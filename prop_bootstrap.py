"""Bootstrap + Poisson pricing of Jalen Hurts' Super Bowl LVII pass-attempt prop, and a backtest."""
from scipy import stats

from common import load_games, load_player_stats, np, pd, r, write

LINE = 31.5
OVER_PRICE = +100
rng = np.random.default_rng(42)


def implied_probability(money_line):
    return -money_line / (-money_line + 100) if money_line < 0 else 100 / (money_line + 100)


ps = load_player_stats()
qb = ps[(ps.position == "QB") & (ps.attempts >= 1)].copy()
hurts = qb[(qb.player_name == "J.Hurts") & qb.season.isin([2021, 2022])]
sb = hurts[(hurts.season == 2022) & (hurts.week == 22)]
history = hurts.drop(sb.index)
x = history.attempts.to_numpy()
actual = int(sb.attempts.iat[0])
print(f"n={len(x)} mean={x.mean():.2f} median={np.median(x)} sd={x.std(ddof=1):.2f} var/mean={x.var(ddof=1) / x.mean():.2f}; SB actual={actual}")

ip = implied_probability(OVER_PRICE)

# Bootstrap the mean
B = 10_000
boot = rng.choice(x, size=(B, len(x)), replace=True).mean(axis=1)
lo, hi = np.quantile(boot, [0.025, 0.975])
print(f"bootstrap mean 95% CI: {lo:.2f}–{hi:.2f}")

p_over = 1 - stats.poisson.cdf(int(LINE), x.mean())
p_over_lo = 1 - stats.poisson.cdf(int(LINE), lo)
p_over_hi = 1 - stats.poisson.cdf(int(LINE), hi)
p_under_pos_ev = (boot < LINE).mean()
p_emp = (x > LINE).mean()
# Negative binomial with the same mean and the observed variance
m, v = x.mean(), x.var(ddof=1)
nb_n, nb_p = m * m / (v - m), m / v
p_nb = 1 - stats.nbinom.cdf(int(LINE), nb_n, nb_p)
# Normal approximation, as a reference
p_norm = 1 - stats.norm.cdf(LINE, m, np.sqrt(v))
print(f"P(over): poisson={p_over:.3f} [{p_over_lo:.3f}, {p_over_hi:.3f}] empirical={p_emp:.3f} negbin={p_nb:.3f} normal={p_norm:.3f}; P(under +EV)={p_under_pos_ev:.3f}")

# Figure 1: the games + Poisson
bins = np.arange(12, 52, 3)
counts, _ = np.histogram(x, bins=bins)
k = np.arange(10, 54)
fig_hist = {
    "type": "line", "height": 300,
    "x": {"label": "Pass attempts in a game", "format": "int", "min": 10, "max": 52},
    "y": {"label": "Share of games", "format": "pct0", "min": 0},
    "rules": [{"axis": "x", "value": LINE, "label": "Line 31.5"}],
    "series": [{"name": f"Hurts, {len(x)} games before the Super Bowl", "color": 1, "kind": "bins",
                "bins": [[int(bins[i]), int(bins[i + 1]), r(c / len(x) / 3, 4), int(c)] for i, c in enumerate(counts)]},
               {"name": f"Poisson(λ = {m:.1f})", "color": 2,
                "points": [[int(i), r(stats.poisson.pmf(i, m), 4)] for i in k]},
               {"name": "Negative binomial (same mean, observed spread)", "color": 3, "dash": True,
                "points": [[int(i), r(stats.nbinom.pmf(i, nb_n, nb_p), 4)] for i in k]}],
    "note": "Bars are scaled to a per-attempt density so they share the axis with the curves.",
}

# Figure 2: bootstrapped means
bb = np.arange(25, 36.01, 0.25)
bc, _ = np.histogram(boot, bins=bb)
fig_boot = {
    "type": "line", "height": 280,
    "x": {"label": "Bootstrapped mean pass attempts per game", "format": "int", "min": 25, "max": 36},
    "y": {"label": "Share of 10,000 resamples", "format": "pct0", "min": 0},
    "rules": [{"axis": "x", "value": LINE, "label": "Line 31.5"},
              {"axis": "x", "value": r(lo, 2), "label": "2.5%", "muted": True},
              {"axis": "x", "value": r(hi, 2), "label": "97.5%", "muted": True}],
    "series": [{"name": "Bootstrapped means", "color": 1, "kind": "bins",
                "bins": [[r(bb[i], 2), r(bb[i + 1], 2), r(c / B, 4), int(c)] for i, c in enumerate(bc)]}],
}

# Figure 3: edge as a function of the true rate
lams = np.quantile(boot, np.linspace(0.005, 0.995, 120))
curve = [[r(l, 3), r(1 - stats.poisson.cdf(int(LINE), l), 4)] for l in lams]
fig_edge = {
    "type": "line", "height": 300,
    "x": {"label": "Hurts' true pass attempts per game (λ)", "format": "dec0"},
    "y": {"label": "Probability the over hits", "format": "pct0", "min": 0, "max": 0.8},
    "rules": [{"axis": "y", "value": r(ip, 3), "label": "Implied by +100"}],
    "bands": [{"series": 0, "ref": r(ip, 3), "below": {"color": "good", "label": "Under is +EV"},
               "above": {"color": "critical", "label": "Over is +EV"}}],
    "series": [{"name": "P(over 31.5) under a Poisson model", "color": 1, "points": curve}],
}

# Backtest: every QB start 2021-2023 with 8+ prior starts in the last two seasons.
def nb_params(x, weights=None):
    """Method-of-moments negative binomial (mean, variance), falling back to Poisson if not overdispersed."""
    w = np.ones(len(x)) if weights is None else weights
    m = np.average(x, weights=w)
    v = np.average((x - m) ** 2, weights=w) * len(x) / (len(x) - 1)
    return m, v


def p_over_nb(line, m, v):
    if v <= m:
        return 1 - stats.poisson.cdf(int(line), m)
    return 1 - stats.nbinom.cdf(int(line), m * m / (v - m), m / v)


# Game script: how many points the QB's team was favored by (closing spread; negative = underdog).
games = load_games()
fav = {}
for gm in games.dropna(subset=["spread_line"]).itertuples():
    fav[(gm.season, gm.week, gm.home_team)] = gm.spread_line
    fav[(gm.season, gm.week, gm.away_team)] = -gm.spread_line

qb = qb.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
qb["favored_by"] = [fav.get(k) for k in zip(qb.season, qb.week, qb.recent_team)]
print("spread matched for", round(qb[qb.season >= 2015].favored_by.notna().mean(), 3), "of QB games since 2015")
qb["favored_by"] = qb.favored_by.fillna(0.0)


def qb_games(min_season):
    """Yield (game row, prior attempts, relative spread) for every QB start with 8+ prior qualifying starts."""
    for pid, g in qb.groupby("player_id"):
        att = g.attempts.to_numpy()
        spr = g.favored_by.to_numpy()
        seasons = g.season.to_numpy()
        for i in range(len(g)):
            if seasons[i] < min_season or att[i] < 15:
                continue
            keep = att[max(0, i - 34):i] >= 15
            prior = att[max(0, i - 34):i][keep]
            if len(prior) >= 8:
                # How much more (or less) favored this game is than the QB's usual game
                rel = spr[i] - spr[max(0, i - 34):i][keep].mean()
                yield g.iloc[i], prior, rel


# Fit the spread effect on 2015-2020 games only, before the backtest window.
fit_rows = [(row.attempts - np.average(prior, weights=0.5 ** (np.arange(len(prior))[::-1] / 8)), rel)
            for row, prior, rel in qb_games(2015) if row.season <= 2020]
fit_rows = np.array(fit_rows)
spread_b, spread_a = np.polyfit(fit_rows[:, 1], fit_rows[:, 0], 1)
print(f"spread effect: {spread_b:+.3f} attempts per point more favored than usual (intercept {spread_a:+.2f}), n={len(fit_rows)}")

rows = []
for row, prior, rel in qb_games(2021):
    att_i = row.attempts
    if True:
        lam = prior.mean()
        nb = nb_params(prior)
        # Recency-weighted version: each game back counts a little less (half-life of 8 games)
        w = 0.5 ** (np.arange(len(prior))[::-1] / 8)
        nb_w = nb_params(prior, w)
        m_s = nb_w[0] + spread_a + spread_b * rel
        nb_ws = (m_s, nb_w[1] * m_s / nb_w[0])  # keep the same dispersion ratio
        # Price lines from 8 below to 8 above the player's average, like alt-lines
        for d in (-8, -4, 0, 4, 8):
            line = np.round(lam) + d + 0.5
            rows.append({"offset": d, "poisson": 1 - stats.poisson.cdf(int(line), lam),
                         "empirical": (prior > line).mean(),
                         "negbin": p_over_nb(line, *nb), "negbin_recent": p_over_nb(line, *nb_w),
                         "negbin_spread": p_over_nb(line, *nb_ws),
                         "over": int(att_i > line), "disp": prior.var(ddof=1) / lam})
bt = pd.DataFrame(rows)
print(f"backtest n={len(bt)}, over rate={bt.over.mean():.3f}")
METHODS = [("poisson", "Poisson on the mean"), ("empirical", "Resample the games"),
           ("negbin", "Negative binomial"), ("negbin_recent", "Negative binomial, recent games weighted"),
           ("negbin_spread", "+ point spread")]
calib = bt.groupby("offset").agg(**{m: (m, "mean") for m, _ in METHODS}, act=("over", "mean"), n=("over", "size"))
print(calib.round(3))
method_scores = {}
for col, _ in METHODS:
    method_scores[col] = {"brier": ((bt[col] - bt.over) ** 2).mean(),
                          "calib_err": (calib[col] - calib.act).abs().mean()}
    print(col, {k: round(v, 4) for k, v in method_scores[col].items()})
fig_methods = {
    "type": "bar", "orient": "h", "height": 220,
    "categories": [lab for _, lab in METHODS],
    "y": {"label": "Average calibration error across the five lines (percentage points)", "format": "dec1", "min": 0},
    "series": [{"name": "Calibration error", "color": 1,
                "values": [r(method_scores[m]["calib_err"] * 100, 2) for m, _ in METHODS],
                "tips": [[["Brier score", f"{method_scores[m]['brier']:.4f}"]] for m, _ in METHODS]}],
}
brier_base = bt.groupby("offset").over.transform(lambda o: ((o.mean() - o) ** 2)).mean()
OFFSET_LABELS = {-8: "Avg − 8", -4: "Avg − 4", 0: "At avg", 4: "Avg + 4", 8: "Avg + 8"}
fig_calib = {
    "type": "bar", "orient": "v", "height": 300,
    "categories": [OFFSET_LABELS[o] for o in calib.index],
    "y": {"label": "Probability the over hits", "format": "pct0", "min": 0, "max": 1},
    "series": [{"name": "Poisson said", "color": 1, "values": [r(v) for v in calib.poisson]},
               {"name": "Resampled games said", "color": 2, "values": [r(v) for v in calib.empirical]},
               {"name": "Actually happened", "color": "ink", "values": [r(v) for v in calib.act],
                "tips": [[["QB games", f"{n:,}"]] for n in calib.n]}],
}
hurts_w = 0.5 ** (np.arange(len(x))[::-1] / 8)
p_nb_recent = p_over_nb(LINE, *nb_params(x, hurts_w))
print("Hurts recency-weighted NB P(over):", round(p_nb_recent, 3), "weighted mean", round(np.average(x, weights=hurts_w), 2))
disp_med = bt.disp.median()
print("median dispersion", disp_med, "share > 1:", (bt.disp > 1).mean())

write("propPredictionBootstrap", {
    "charts": {"hist": fig_hist, "boot": fig_boot, "edge": fig_edge, "calibration": fig_calib, "methods": fig_methods},
    "stats": {"n": len(x), "mean": r(m, 2), "median": r(np.median(x), 1), "sd": r(np.sqrt(v), 2),
              "dispersion": r(v / m, 2), "ciLow": r(lo, 2), "ciHigh": r(hi, 2), "implied": r(ip, 3),
              "pOver": r(p_over, 3), "pOverLow": r(p_over_lo, 3), "pOverHigh": r(p_over_hi, 3),
              "pEmp": r(p_emp, 3), "pNegBin": r(p_nb, 3), "pNorm": r(p_norm, 3),
              "pUnderPosEv": r(p_under_pos_ev, 3), "pNegBinRecent": r(p_nb_recent, 3),
              "spreadB": r(spread_b, 3), "spreadA": r(spread_a, 2), "actual": actual,
              "bt": {"n": len(bt) // 5, "calib": {str(o): {k: r(calib.loc[o, k]) for k in [m for m, _ in METHODS] + ["act"]} for o in calib.index},
                     "methods": {m: {k: r(v, 4) for k, v in sc.items()} for m, sc in method_scores.items()}, "dispMedian": r(disp_med, 2),
                     "dispShare": r((bt.disp > 1).mean(), 3), "brierBase": r(brier_base, 4),
                     }},
})
