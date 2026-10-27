"""Bryce Young, 2023-2026: when did he turn around, and how much of it is him?"""
from common import fetch_ftn, fetch_participation, load_pbp, np, pd, r, write

QB = "B.Young"
SEASONS = [2023, 2024, 2025, 2026]
MIN_DB = {2023: 200, 2024: 200, 2025: 200, 2026: 120}  # qualifying dropbacks (2026 is through Week 7)
SEASON_COLOR = {2023: "gray", 2024: 1, 2025: 2, 2026: 3}

cols = ["season", "season_type", "week", "game_id", "play_id", "posteam", "defteam", "passer", "passer_id",
        "qb_dropback", "epa", "cpoe", "sack", "pass_attempt", "interception", "success", "qb_scramble"]
pbp = load_pbp(SEASONS, columns=cols)
db = pbp[(pbp.season_type == "REG") & (pbp.qb_dropback == 1) & pbp.passer.notna()].dropna(subset=["epa"]).copy()
last_week_2026 = int(db[db.season == 2026].week.max())
print("2026 data through week", last_week_2026)

# ---------- season summaries and ranks ----------
qs = db.groupby(["season", "passer"]).agg(dropbacks=("epa", "size"), epa=("epa", "mean"), cpoe=("cpoe", "mean"),
                                          sack_rate=("sack", "mean"), success=("success", "mean")).reset_index()
qs = qs[qs.dropbacks >= qs.season.map(MIN_DB)]
qs["rank"] = qs.groupby("season").epa.rank(ascending=False, method="min").astype(int)
qs["n"] = qs.groupby("season").passer.transform("size")
young = qs[qs.passer == QB].set_index("season")
print(young.round(3))

# ---------- game log ----------
games = (db[db.passer == QB].groupby(["season", "week", "defteam"])
         .agg(n=("epa", "size"), epa=("epa", "mean"), cpoe=("cpoe", "mean")).reset_index())
starts = games[games.n >= 15].reset_index(drop=True)
starts["idx"] = np.arange(1, len(starts) + 1)
starts["label"] = [f"{s} Wk {w} vs {o}" for s, w, o in zip(starts.season, starts.week, starts.defteam)]
starts["roll6"] = (starts.epa * starts.n).rolling(6, min_periods=3).sum() / starts.n.rolling(6, min_periods=3).sum()
season_start = starts.groupby("season").idx.min()

# Per-play EPA variance on his dropbacks: each game's mean has variance sigma2 / n
sigma2 = db[db.passer == QB].epa.var()
league_mean = qs.epa.mean()
# Between-QB spread of true talent (method of moments on 2023-25 qualified QB-seasons)
full = qs[qs.season < 2026]
noise = (db.epa.var() / full.dropbacks).mean()
tau2 = max(full.epa.var() - noise, 1e-4)
print(f"sigma per play {np.sqrt(sigma2):.3f}, between-QB sd of true talent {np.sqrt(tau2):.3f}")


# ---------- Bayesian change-point model ----------
def segment_log_ml(y, n, m0=league_mean, s0=np.sqrt(tau2)):
    """Log marginal likelihood of game means y (with n plays each) sharing one unknown mean ~ N(m0, s0^2)."""
    v = sigma2 / n
    a = 1 / s0 ** 2 + np.sum(1 / v)
    resid = y - m0
    return -0.5 * (np.sum(np.log(2 * np.pi * v)) + np.log(1 + s0 ** 2 * np.sum(1 / v))
                   + np.sum(resid ** 2 / v) - np.sum(resid / v) ** 2 / a)


y, n = starts.epa.to_numpy(), starts.n.to_numpy()
N, MIN_SEG = len(y), 3
log_none = segment_log_ml(y, n)
taus = np.arange(MIN_SEG, N - MIN_SEG + 1)  # change happens after game tau
log_one = np.array([segment_log_ml(y[:t], n[:t]) + segment_log_ml(y[t:], n[t:]) for t in taus])
post_one = np.exp(log_one - log_one.max())
post_one /= post_one.sum()
best_tau = int(taus[post_one.argmax()])
# Bayes factor: one change point (uniform over locations) vs. none
log_bf = np.log(np.mean(np.exp(log_one - log_one.max()))) + log_one.max() - log_none
before, after = starts.iloc[:best_tau], starts.iloc[best_tau:]
mean_before = np.average(before.epa, weights=before.n)
mean_after = np.average(after.epa, weights=after.n)
print(f"most likely change after game {best_tau} ({starts.label[best_tau - 1]}), next game {starts.label[best_tau]};"
      f" P={post_one.max():.2f}; log BF vs none = {log_bf:.1f}; before {mean_before:+.3f} after {mean_after:+.3f}")
top3 = sorted(zip(post_one, taus), reverse=True)[:3]
print("top change points:", [(starts.label[t], round(p, 3)) for p, t in top3])
order = np.argsort(-post_one)
cred = taus[order[:np.searchsorted(np.cumsum(post_one[order]), 0.9) + 1]]
cred_seasons = sorted({int(starts.season[t - 1]) for t in cred} | {int(starts.season[t]) for t in cred})
p_in_2024 = post_one[[starts.season[t] == 2024 or (starts.season[t - 1] == 2024) for t in taus]].sum()
print("90% credible set spans", starts.label[min(cred) - 1], "to", starts.label[max(cred) - 1], "| P(change during 2024 incl. benching) =", round(p_in_2024, 3))

# Two change points
pairs = [(a, b) for a in range(MIN_SEG, N) for b in range(a + MIN_SEG, N - MIN_SEG + 1)]
log_two = np.array([segment_log_ml(y[:a], n[:a]) + segment_log_ml(y[a:b], n[a:b]) + segment_log_ml(y[b:], n[b:])
                    for a, b in pairs])
best_pair = pairs[int(log_two.argmax())]
log_bf_two = (np.log(np.mean(np.exp(log_two - log_two.max()))) + log_two.max()) - \
             (np.log(np.mean(np.exp(log_one - log_one.max()))) + log_one.max())
seg_means = [np.average(y[s], weights=n[s]) for s in (slice(0, best_pair[0]), slice(*best_pair), slice(best_pair[1], N))]
print("best two change points after", [starts.label[t - 1] for t in best_pair], "segment means", np.round(seg_means, 3),
      "log BF two vs one", round(log_bf_two, 2))

fig_games = {
    "type": "line", "height": 320,
    "x": {"label": "Start number (2023 Week 1 = 1)", "format": "int", "min": 1, "max": N},
    "y": {"label": "EPA per dropback", "format": "signed1"},
    "rules": [{"axis": "y", "value": 0}] + [{"axis": "x", "value": float(i) - 0.5, "label": str(s), "muted": True}
                                            for s, i in season_start.items()],
    "series": [{"name": "Single game", "color": "gray", "dash": False, "kind": "dots",
                "points": [[int(i), r(e, 3)] for i, e in zip(starts.idx, starts.epa)]},
               {"name": "Rolling 6-game average", "color": 1,
                "points": [[int(i), r(e, 3)] for i, e in zip(starts.idx, starts.roll6) if pd.notna(e)]}],
    "tooltipLabels": list(starts.label),
}
fig_cp = {
    "type": "line", "height": 240,
    "x": {"label": "Change point after start number", "format": "int", "min": 1, "max": N},
    "y": {"label": "Posterior probability", "format": "pct0", "min": 0},
    "rules": [{"axis": "x", "value": float(i) - 0.5, "label": str(s), "muted": True} for s, i in season_start.items()],
    "series": [{"name": "Probability the change happened right after this start", "color": 1, "kind": "bins",
                "bins": [[r(t - 0.4, 2), r(t + 0.4, 2), r(p, 4)] for t, p in zip(taus, post_one)]}],
}

# ---------- FTN charting: what he controls vs. what's around him ----------
ftn = pd.concat([pd.read_parquet(fetch_ftn(s)) for s in SEASONS], ignore_index=True)
ftn = ftn.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
fcols = ["game_id", "play_id", "is_qb_fault_sack", "is_interception_worthy", "is_throw_away", "is_qb_out_of_pocket",
         "is_catchable_ball", "is_drop", "read_thrown", "is_play_action", "is_screen_pass", "is_motion"]
d = db.merge(ftn[fcols], on=["game_id", "play_id"], how="inner")
print("FTN match rate", round(len(d) / len(db), 3))
att = d[(d.pass_attempt == 1) & (d.sack == 0)]
aimed = att[~att.is_throw_away]
read = att[att.read_thrown.isin(["1", "2", "CHK"])]


def rates(group):
    keys = ["season", "passer"]
    out = pd.DataFrame({
        "qb_fault_sack": d.groupby(keys).is_qb_fault_sack.mean(),
        "sack": d.groupby(keys).sack.mean(),
        "int_worthy": att.groupby(keys).is_interception_worthy.mean(),
        "throwaway": att.groupby(keys).is_throw_away.mean(),
        "catchable": aimed.groupby(keys).is_catchable_ball.mean(),
        "drop": aimed[aimed.is_catchable_ball].groupby(keys).is_drop.mean(),
        "first_read": read.groupby(keys).read_thrown.apply(lambda s: (s == "1").mean()),
        "checkdown": read.groupby(keys).read_thrown.apply(lambda s: (s == "CHK").mean()),
        "out_of_pocket": d.groupby(keys).is_qb_out_of_pocket.mean(),
        "play_action": d.groupby(keys).is_play_action.mean(),
        "screen": att.groupby(keys).is_screen_pass.mean(),
        "motion": d.groupby(keys).is_motion.mean(),
    })
    out["fault_share"] = out.qb_fault_sack / out.sack
    return out


ftn_rates = rates(d)
ftn_rates = ftn_rates[ftn_rates.index.isin(list(zip(qs.season, qs.passer)))]
yr = ftn_rates.xs(QB, level="passer")
lg = ftn_rates.groupby("season").median()
print("Young:\n", yr.round(3).T, "\nLeague median:\n", lg.round(3).T)

def ftn_fig(metrics, height=280):
    return {
        "type": "bar", "orient": "v", "height": height,
        "categories": [lab for _, lab in metrics],
        "y": {"label": "Rate", "format": "pct0", "min": 0},
        "series": [{"name": str(s), "color": SEASON_COLOR[s], "values": [r(yr.loc[s, k], 4) for k, _ in metrics],
                    "tips": [[["League median", f"{lg.loc[s, k]:.1%}"]] for k, _ in metrics]} for s in SEASONS],
    }


fig_ftn_reads = ftn_fig([("catchable", "Catchable throws"), ("first_read", "Thrown to first read"),
                         ("checkdown", "Checkdowns")])
fig_ftn_mistakes = ftn_fig([("qb_fault_sack", "Sacks charted as his fault"), ("int_worthy", "Interception-worthy throws"),
                            ("throwaway", "Throwaways"), ("drop", "Receiver drops (of catchable)")])

# ---------- pressure (charted participation data exists through 2025) ----------
part = []
for s in [2023, 2024, 2025]:
    pt = pd.read_parquet(fetch_participation(s), columns=["nflverse_game_id", "play_id", "was_pressure"])
    part.append(pt.rename(columns={"nflverse_game_id": "game_id"}))
part = pd.concat(part)
dp = db[db.season < 2026].merge(part, on=["game_id", "play_id"]).dropna(subset=["was_pressure"])
dp["was_pressure"] = dp.was_pressure.astype(bool)
pres = dp.groupby(["season", "passer"]).agg(pressure_rate=("was_pressure", "mean"))
pres["p2s"] = dp[dp.was_pressure].groupby(["season", "passer"]).sack.mean()
pres["epa_pressured"] = dp[dp.was_pressure].groupby(["season", "passer"]).epa.mean()
pres["epa_clean"] = dp[~dp.was_pressure].groupby(["season", "passer"]).epa.mean()
pres = pres[pres.index.isin(list(zip(qs.season, qs.passer)))]
yp = pres.xs(QB, level="passer")
lp = pres.groupby("season").median()
print("pressure Young\n", yp.round(3), "\nleague median\n", lp.round(3))

# ---------- how much should we trust 2026? empirical-Bayes shrinkage ----------
q26 = qs[qs.season == 2026].copy()
q26["noise"] = db.epa.var() / q26.dropbacks
q26["shrunk"] = (tau2 * q26.epa + q26.noise * league_mean) / (tau2 + q26.noise)
q26["post_sd"] = np.sqrt(tau2 * q26.noise / (tau2 + q26.noise))
q26["shrunk_rank"] = q26.shrunk.rank(ascending=False, method="min").astype(int)
y26 = q26[q26.passer == QB].iloc[0]
# Probability he's a top-10 QB this season, by simulation from each QB's posterior
rng = np.random.default_rng(3)
draws = rng.normal(q26.shrunk.to_numpy()[:, None], q26.post_sd.to_numpy()[:, None], size=(len(q26), 20000))
ranks = (-draws).argsort(axis=0).argsort(axis=0) + 1
yi = int(np.where(q26.passer.to_numpy() == QB)[0][0])
p_top10 = (ranks[yi] <= 10).mean()
p_above_avg = (draws[yi] > league_mean).mean()
rank_ci = np.quantile(ranks[yi], [0.05, 0.95])
print(f"2026: raw {y26.epa:+.3f} (#{int(y26['rank'])} of {len(q26)}), shrunk {y26.shrunk:+.3f} ± {y26.post_sd:.3f} "
      f"(#{int(y26.shrunk_rank)}), P(top 10) {p_top10:.2f}, P(above avg) {p_above_avg:.2f}, 90% rank {rank_ci}")

# Same shrinkage applied to his 2025 season, for comparison
q25 = qs[qs.season == 2025]
y25 = q25[q25.passer == QB].iloc[0]
noise25 = db.epa.var() / y25.dropbacks
shrunk25 = (tau2 * y25.epa + noise25 * league_mean) / (tau2 + noise25)

fig_shrink = {
    "type": "scatter", "height": 380,
    "x": {"label": "Raw EPA per dropback, 2026 (through Week %d)" % last_week_2026, "format": "signed2"},
    "y": {"label": "Shrunk estimate (empirical Bayes)", "format": "signed2"},
    "trend": {"slope": 1, "intercept": 0, "label": "No shrinkage"},
    "groups": [{"name": "Other qualified QBs", "color": "gray", "r": 4,
                "labels": list(q26.sort_values("epa").passer.iloc[[0, -1]]),
                "points": [{"x": r(x.epa, 4), "y": r(x.shrunk, 4), "label": x.passer,
                            "tip": [["QB", x.passer], ["Dropbacks", str(int(x.dropbacks))], ["Raw", f"{x.epa:+.3f} (#{int(x.rank)})"],
                                    ["Shrunk", f"{x.shrunk:+.3f} (#{int(x.shrunk_rank)})"]]}
                           for x in q26[q26.passer != QB].itertuples()]},
               {"name": "Bryce Young", "color": 2, "r": 6, "labels": [QB], "forceLabels": True,
                "points": [{"x": r(y26.epa, 4), "y": r(y26.shrunk, 4), "label": QB,
                            "tip": [["QB", QB], ["Dropbacks", str(int(y26.dropbacks))], ["Raw", f"{y26.epa:+.3f} (#{int(y26['rank'])})"],
                                    ["Shrunk", f"{y26.shrunk:+.3f} (#{int(y26.shrunk_rank)})"]]}]}],
}

write("bryceYoung", {
    "charts": {"games": fig_games, "changePoint": fig_cp, "ftnReads": fig_ftn_reads, "ftnMistakes": fig_ftn_mistakes,
               "shrink": fig_shrink},
    "stats": {
        "lastWeek2026": last_week_2026,
        "seasons": {int(s): {k: (r(v, 4) if isinstance(v, float) else int(v)) for k, v in row.items() if k != "passer"}
                    for s, row in young.iterrows()},
        "starts": N, "sigma": r(np.sqrt(sigma2), 3), "tau": r(np.sqrt(tau2), 3),
        "change": {"lastBefore": starts.label[best_tau - 1], "firstAfter": starts.label[best_tau], "p": r(post_one.max(), 3),
                   "logBf": r(log_bf, 2), "p2024": r(p_in_2024, 3), "credFrom": starts.label[min(cred) - 1], "credTo": starts.label[max(cred) - 1], "meanBefore": r(mean_before, 3), "meanAfter": r(mean_after, 3),
                   "top": [[starts.label[t], r(p, 3)] for p, t in top3],
                   "two": {"after": [starts.label[t - 1] for t in best_pair], "means": [r(m, 3) for m in seg_means],
                           "logBf": r(log_bf_two, 2)}},
        "ftn": {"young": {int(s): {k: r(v, 4) for k, v in row.items()} for s, row in yr.iterrows()},
                "league": {int(s): {k: r(v, 4) for k, v in row.items()} for s, row in lg.iterrows()}},
        "pressure": {"young": {int(s): {k: r(v, 4) for k, v in row.items()} for s, row in yp.iterrows()},
                     "league": {int(s): {k: r(v, 4) for k, v in row.items()} for s, row in lp.iterrows()}},
        "shrink": {"raw": r(y26.epa, 4), "rank": int(y26["rank"]), "n": len(q26), "dropbacks": int(y26.dropbacks),
                   "shrunk": r(y26.shrunk, 4), "sd": r(y26.post_sd, 4), "shrunkRank": int(y26.shrunk_rank),
                   "pTop10": r(p_top10, 3), "pAboveAvg": r(p_above_avg, 3), "rankCi": [int(x) for x in rank_ci],
                   "shrunk2025": r(shrunk25, 4), "raw2025": r(y25.epa, 4), "leagueMean": r(league_mean, 4)},
    },
})
