"""Opponent-adjusted EPA with ridge regression, for the NFL (2021-2024) and FBS (2023-2024)."""
import scipy.sparse as sp
from sklearn.linear_model import Ridge

from common import cfb_file, cfb_pbp, load_games, load_pbp, np, pd, r, write

LAMBDAS = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
NFL_YEARS = range(2021, 2025)   # complete seasons only
CFB_YEARS = (2023, 2024)
NFL_EX, CFB_EX = 2024, 2024     # example seasons for the single-season charts
rng = np.random.default_rng(5)


def design(d, off, dfn, home, teams):
    """Sparse one-hot design: an offense column and a defense column per team, plus home field."""
    idx = {t: i for i, t in enumerate(teams)}
    n, t = len(d), len(teams)
    rows = np.arange(n)
    X_off = sp.csr_matrix((np.ones(n), (rows, d[off].map(idx))), shape=(n, t))
    X_def = sp.csr_matrix((np.ones(n), (rows, d[dfn].map(idx))), shape=(n, t))
    X_home = sp.csr_matrix(d[off].eq(d[home]).astype(float).to_numpy().reshape(-1, 1))
    return sp.hstack([X_off, X_def, X_home]).tocsr()


def ridge_ratings(d, lam, off="posteam", dfn="defteam", home="home_team", y="epa"):
    teams = sorted(set(d[off]) | set(d[dfn]))
    m = Ridge(alpha=lam).fit(design(d, off, dfn, home, teams), d[y].to_numpy())
    t = len(teams)
    return (pd.Series(m.coef_[:t], teams), pd.Series(m.coef_[t:2 * t], teams), m.coef_[-1])


def raw_ratings(d, off="posteam", dfn="defteam", y="epa"):
    mu = d[y].mean()
    return d.groupby(off)[y].mean() - mu, d.groupby(dfn)[y].mean() - mu


def backtest(plays, games, weeks, ratings_fn, cols):
    """For each week, rate teams on the weeks before it, then score that week's games."""
    off, dfn, home, y, season = cols
    rows = []
    for s in sorted(plays[season].unique()):
        for w in weeks:
            train = plays[(plays[season] == s) & (plays.week < w)]
            test = games[(games.season == s) & (games.week == w)]
            if test.empty or train.empty:
                continue
            nets = {k: o - d for k, (o, d) in ratings_fn(train).items()}
            for g in test.itertuples():
                row = {"season": s, "week": w, "margin": g.margin, "spread": getattr(g, "spread", np.nan)}
                for k, net in nets.items():
                    row[k] = net.get(g.home, np.nan) - net.get(g.away, np.nan)
                rows.append(row)
    return pd.DataFrame(rows)


def all_ratings(train, cols):
    off, dfn, home, y, _ = cols
    out = {"raw": raw_ratings(train, off, dfn, y)}
    for lam in LAMBDAS:
        o, d, _ = ridge_ratings(train, lam, off, dfn, home, y)
        out[lam] = (o, d)
    return out


def boot_gain(bt, a, b, n=3000):
    """Paired bootstrap over games of corr(b, margin) - corr(a, margin)."""
    x = bt[[a, b, "margin"]].dropna().to_numpy()
    gains = []
    for _ in range(n):
        s = x[rng.integers(0, len(x), len(x))]
        gains.append(np.corrcoef(s[:, 1], s[:, 2])[0, 1] - np.corrcoef(s[:, 0], s[:, 2])[0, 1])
    return np.quantile(gains, [0.025, 0.975])


# ======================= NFL =======================
NFL_COLS = ("posteam", "defteam", "home_team", "epa", "season")
cols = ["season", "season_type", "week", "game_id", "posteam", "defteam", "home_team", "epa", "pass", "rush", "wp"]
nfl = load_pbp(NFL_YEARS, columns=cols)
nfl = nfl[(nfl.season_type == "REG") & ((nfl["pass"] == 1) | (nfl.rush == 1))].dropna(subset=["epa", "posteam", "defteam", "wp"])
nfl = nfl[nfl.wp.between(0.05, 0.95)]  # drop garbage time
g = load_games()
g = g[g.season.isin(NFL_YEARS) & (g.game_type == "REG")].dropna(subset=["result"])
nfl_games = pd.DataFrame({"season": g.season, "week": g.week, "home": g.home_team, "away": g.away_team,
                          "margin": g.result, "spread": g.spread_line})

# Shrinkage path for the example season: how each team's offense coefficient moves as lambda grows
n25 = nfl[nfl.season == NFL_EX]
PATH_LAMBDAS = np.logspace(0, 4, 17)
path = {lam: ridge_ratings(n25, lam)[0] for lam in PATH_LAMBDAS}
raw_off25, raw_def25 = raw_ratings(n25)
PATH_TEAMS = {"DET": 1, "PHI": 2, "CAR": 3, "CLE": 4}
fig_path = {
    "type": "line", "height": 300,
    "x": {"label": "log₁₀ λ (ridge penalty)", "format": "dec1", "min": 0, "max": 4},
    "y": {"label": "Offense coefficient (EPA per play vs. average)", "format": "signed2"},
    "rules": [{"axis": "y", "value": 0}],
    "series": [{"name": t, "color": c, "points": [[r(np.log10(lam), 3), r(path[lam][t], 4)] for lam in PATH_LAMBDAS]}
               for t, c in PATH_TEAMS.items()],
}
for t in PATH_TEAMS:
    print(t, "raw off", round(raw_off25[t], 3), "λ=1", round(path[PATH_LAMBDAS[0]][t], 3), "λ=1000", round(path[1000.0][t], 3))

# Schedule effect in the example season (near-unshrunk ridge vs raw)
o1, d1, home_nfl = ridge_ratings(n25, 1)
nfl_sched_corr = (o1 - d1).corr(raw_off25 - raw_def25)
print(f"NFL {NFL_EX} corr(adjusted net, raw net) =", round(nfl_sched_corr, 3), "home field", round(home_nfl, 4))

nfl_bt = backtest(nfl, nfl_games, range(4, 19), lambda d: all_ratings(d, NFL_COLS), NFL_COLS)
nfl_corr = {k: nfl_bt[k].corr(nfl_bt.margin) for k in ["raw"] + LAMBDAS}
# nflverse spread_line is positive when the home team is favored
spread_corr = nfl_bt.spread.corr(nfl_bt.margin)
best_nfl = max(LAMBDAS, key=lambda k: nfl_corr[k])
nfl_ci = boot_gain(nfl_bt, "raw", best_nfl)
nfl_winners = {k: (np.sign(nfl_bt[k]) == np.sign(nfl_bt.margin)).mean() for k in ["raw", best_nfl]}
print("NFL backtest games:", len(nfl_bt), {k: round(v, 4) for k, v in nfl_corr.items()}, "spread", round(spread_corr, 4),
      "best", best_nfl, "gain CI", nfl_ci.round(4), "winners", {k: round(v, 3) for k, v in nfl_winners.items()})
early = nfl_bt.week <= 8
nfl_early = {k: (nfl_bt[early][k].corr(nfl_bt[early].margin), nfl_bt[~early][k].corr(nfl_bt[~early].margin))
             for k in ["raw", best_nfl]}
print("NFL early/late:", {k: tuple(round(x, 3) for x in v) for k, v in nfl_early.items()})


def lambda_fig(corr, raw_label, extra_rules=()):
    return {
        "type": "line", "height": 280, "markers": True,
        "x": {"label": "log₁₀ λ (ridge penalty)", "format": "dec1", "min": 0, "max": 4},
        "y": {"label": "Correlation with next week's margins", "format": "dec2"},
        "rules": [{"axis": "y", "value": r(corr["raw"], 4), "label": raw_label}] + list(extra_rules),
        "series": [{"name": "Ridge-adjusted EPA", "color": 1,
                    "points": [[r(np.log10(lam), 3), r(corr[lam], 4)] for lam in LAMBDAS]}],
    }


fig_nfl_lambda = lambda_fig(nfl_corr, "Raw EPA", [{"axis": "y", "value": r(spread_corr, 4), "label": "Vegas closing spread"}])

# ======================= College (FBS) =======================
info = pd.read_parquet(cfb_file("cfb_team_info", CFB_EX))
fbs_info = info[info.classification == "fbs"]
FBS = set(fbs_info.school)
P4 = set(fbs_info[fbs_info.conference.isin(["SEC", "Big Ten", "Big 12", "ACC"])].school) | {"Notre Dame"}
CONF = dict(zip(fbs_info.school, fbs_info.conference))
CFB_COLS = ("pos_team", "def_pos_team", "home", "EPA", "season")


def load_cfb(year):
    p = pd.read_parquet(cfb_pbp(year),
                        columns=["game_id", "week", "season_type", "pos_team", "def_pos_team", "home", "away",
                                 "EPA", "wp_before", "rush", "pass"])
    p = p[(p.season_type == "regular") & ((p.rush == 1) | (p["pass"] == 1))].dropna(subset=["EPA", "wp_before"])
    p = p[p.pos_team.isin(FBS) & p.def_pos_team.isin(FBS) & p.wp_before.between(0.05, 0.95)]
    return p.assign(season=year)


cfb = pd.concat([load_cfb(y) for y in CFB_YEARS], ignore_index=True)
sched = pd.concat([pd.read_parquet(cfb_file("cfb_schedules", y)) for y in CFB_YEARS])
sched = sched[(sched.season_type == "regular") & sched.home_team.isin(FBS) & sched.away_team.isin(FBS)
              & ~sched.neutral_site.astype(bool)].dropna(subset=["home_points", "away_points"])
cfb_games = pd.DataFrame({"season": sched.season, "week": sched.week, "home": sched.home_team, "away": sched.away_team,
                          "margin": sched.home_points - sched.away_points})

c25 = cfb[cfb.season == CFB_EX]
co, cd, home_cfb = ridge_ratings(c25, 1, *CFB_COLS[:4])
craw_o, craw_d = raw_ratings(c25, "pos_team", "def_pos_team", "EPA")
cnet, crawnet = co - cd, craw_o - craw_d
cfb_sched_corr = cnet.corr(crawnet)
rank = lambda s: s.rank(ascending=False, method="min").astype(int)
moves = (rank(crawnet) - rank(cnet))
print(f"CFB {CFB_EX} corr(adjusted, raw) =", round(cfb_sched_corr, 3), "home field", round(home_cfb, 4),
      "teams", len(cnet), "max move", int(moves.abs().max()))

# External check against the sportsdataverse opponent-adjusted ratings
sdv = pd.read_parquet(cfb_file("cfb_ratings", CFB_EX)).assign(team_id=lambda x: pd.to_numeric(x.team_id))
sdv = fbs_info.assign(team_id=pd.to_numeric(fbs_info.team_id))[["team_id", "school"]].merge(sdv, on="team_id").set_index("school")
check = {"ridge": cnet.reindex(sdv.index).corr(sdv.adj_net), "raw": crawnet.reindex(sdv.index).corr(sdv.adj_net)}
print("corr with sportsdataverse adj_net:", {k: round(v, 3) for k, v in check.items()})

# Strength of schedule: average adjusted net rating of opponents faced
opp_strength = c25.groupby("pos_team").def_pos_team.apply(lambda s: cnet.reindex(s.unique()).mean())
LABEL = ["Clemson", "Ohio State", "Indiana", "Texas Tech", "Oregon"] + list(moves.sort_values().index[:3]) + list(moves.sort_values().index[-3:])


def cfb_point(t):
    return {"x": r(crawnet[t], 4), "y": r(cnet[t], 4), "label": t,
            "tip": [["Team", f"{t} ({CONF.get(t, '')})"], ["Raw net EPA", f"{crawnet[t]:+.3f} (#{rank(crawnet)[t]})"],
                    ["Adjusted", f"{cnet[t]:+.3f} (#{rank(cnet)[t]})"], ["Opponents' avg rating", f"{opp_strength.get(t, np.nan):+.3f}"]]}


fig_cfb_scatter = {
    "type": "scatter", "height": 420,
    "x": {"label": "Raw net EPA per play", "format": "signed2"},
    "y": {"label": "Opponent-adjusted net EPA per play (ridge)", "format": "signed2"},
    "trend": {"slope": 1, "intercept": 0, "label": "No change"},
    "groups": [{"name": "Group of Five", "color": "gray", "r": 4, "labels": LABEL,
                "points": [cfb_point(t) for t in cnet.index if t not in P4]},
               {"name": "Power Four + Notre Dame", "color": 1, "r": 4, "labels": LABEL,
                "points": [cfb_point(t) for t in cnet.index if t in P4 and t != "Clemson"]},
               {"name": "Clemson", "color": 2, "r": 6, "labels": ["Clemson"], "forceLabels": True, "points": [cfb_point("Clemson")]}],
}
print("biggest risers:", moves.sort_values().tail(5).to_dict(), "fallers:", moves.sort_values().head(5).to_dict())
print("Clemson:", {"raw_rank": int(rank(crawnet)["Clemson"]), "adj_rank": int(rank(cnet)["Clemson"]),
                   "raw_off": round(craw_o["Clemson"], 3), "adj_off": round(co["Clemson"], 3),
                   "raw_def": round(craw_d["Clemson"], 3), "adj_def": round(cd["Clemson"], 3),
                   "opp": round(opp_strength["Clemson"], 3), "opp_rank": int(opp_strength.rank(ascending=False)["Clemson"])})
p4_gap = {"P4 mean move": float(moves[moves.index.isin(P4)].mean()), "G5 mean move": float(moves[~moves.index.isin(P4)].mean())}
print(p4_gap)

cfb_bt = backtest(cfb, cfb_games, range(4, 15), lambda d: all_ratings(d, CFB_COLS), CFB_COLS)
cfb_corr = {k: cfb_bt[k].corr(cfb_bt.margin) for k in ["raw"] + LAMBDAS}
best_cfb = max(LAMBDAS, key=lambda k: cfb_corr[k])
cfb_ci = boot_gain(cfb_bt, "raw", best_cfb)
cfb_winners = {k: (np.sign(cfb_bt[k]) == np.sign(cfb_bt.margin)).mean() for k in ["raw", best_cfb]}
print("CFB backtest games:", len(cfb_bt), {k: round(v, 4) for k, v in cfb_corr.items()}, "best", best_cfb,
      "gain CI", cfb_ci.round(4), "winners", {k: round(v, 3) for k, v in cfb_winners.items()})
fig_cfb_lambda = lambda_fig(cfb_corr, "Raw EPA")

write("ridgeAdjustment", {
    "charts": {"path": fig_path, "nflLambda": fig_nfl_lambda, "cfbScatter": fig_cfb_scatter, "cfbLambda": fig_cfb_lambda},
    "stats": {
        "nfl": {"schedCorr": r(nfl_sched_corr), "home": r(home_nfl, 4), "games": len(nfl_bt),
                "corr": {str(k): r(v, 4) for k, v in nfl_corr.items()}, "spread": r(spread_corr, 4), "best": best_nfl,
                "gainCi": [r(x, 4) for x in nfl_ci], "winners": {str(k): r(v, 4) for k, v in nfl_winners.items()},
                "earlyLate": {str(k): [r(x, 4) for x in v] for k, v in nfl_early.items()},
                "path": {t: {"raw": r(raw_off25[t], 4), "l1": r(path[PATH_LAMBDAS[0]][t], 4),
                             "l1000": r(path[1000.0][t], 4)} for t in PATH_TEAMS}},
        "cfb": {"schedCorr": r(cfb_sched_corr), "home": r(home_cfb, 4), "teams": len(cnet), "maxMove": int(moves.abs().max()),
                "check": {k: r(v) for k, v in check.items()}, "games": len(cfb_bt),
                "corr": {str(k): r(v, 4) for k, v in cfb_corr.items()}, "best": best_cfb,
                "gainCi": [r(x, 4) for x in cfb_ci], "winners": {str(k): r(v, 4) for k, v in cfb_winners.items()},
                "risers": {k: int(v) for k, v in moves.sort_values().tail(5).items()},
                "fallers": {k: int(v) for k, v in moves.sort_values().head(5).items()},
                "moveByGroup": {k: r(v, 2) for k, v in p4_gap.items()},
                "clemson": {"rawRank": int(rank(crawnet)["Clemson"]), "adjRank": int(rank(cnet)["Clemson"]),
                            "rawOff": r(craw_o["Clemson"], 4), "adjOff": r(co["Clemson"], 4),
                            "rawDef": r(craw_d["Clemson"], 4), "adjDef": r(cd["Clemson"], 4),
                            "opp": r(opp_strength["Clemson"], 4),
                            "oppRank": int(opp_strength.rank(ascending=False)["Clemson"])}},
    },
})
