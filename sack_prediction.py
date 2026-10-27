"""Sack probability on dropbacks with logistic regression, 2020-2022 train / 2023 test."""
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from common import load_pbp, np, pd, r, write

pbp = load_pbp([2020, 2021, 2022, 2023], participation=True)
drop = pbp[(pbp.season_type == "REG") & (pbp["pass"] == 1) & (pbp.play_type != "no_play")
           & (pbp.qb_spike == 0)].copy()
drop["obvious_pass"] = ((drop.down == 3) & (drop.ydstogo >= 6)).astype(int)
drop["shotgun"] = drop.shotgun.astype(float)
FEATURES = ["down", "ydstogo", "yardline_100", "game_seconds_remaining", "score_differential",
            "defenders_in_box", "number_of_pass_rushers", "shotgun", "xpass", "obvious_pass"]
df = drop.dropna(subset=FEATURES + ["sack"]).copy()
df = df[df.down.between(1, 4) & df.number_of_pass_rushers.between(1, 11)]
print(len(drop), "dropbacks;", len(df), "with participation data;", df.groupby("season").sack.mean().round(4).to_dict())

base = df.sack.mean()

# --- Exploratory: rates, not counts
def rate_table(col, cats, labels):
    out = []
    for c, lab in zip(cats, labels):
        sub = df[c(df[col])] if callable(c) else df[df[col] == c]
        out.append((lab, sub.sack.mean(), len(sub), sub.sack.sum()))
    return out


downs = rate_table("down", [1, 2, 3, 4], ["1st", "2nd", "3rd", "4th"])
rushers = rate_table("number_of_pass_rushers", [lambda s: s <= 3, 4, 5, lambda s: s >= 6],
                     ["3 or fewer", "4", "5", "6+"])


def rate_fig(rows, label, color=1):
    total = sum(n for _, _, n, _ in rows)
    return {
        "type": "bar", "orient": "v", "height": 260,
        "categories": [x[0] for x in rows],
        "y": {"label": label, "format": "pct1", "min": 0},
        "rules": [{"axis": "y", "value": r(base, 4), "label": f"All dropbacks {base:.1%}"}],
        "series": [{"name": "Sack rate", "color": color, "values": [r(x[1], 4) for x in rows],
                    "tips": [[["Dropbacks", f"{x[2]:,} ({x[2] / total:.0%} of all)"], ["Sacks", f"{int(x[3]):,}"]]
                             for x in rows]}],
    }


fig_rushers = rate_fig(rushers, "Sack rate")
for row in downs + rushers:
    print(row)

# Down x distance heatmap
dist_bins = [(1, 3, "1–3"), (4, 6, "4–6"), (7, 10, "7–10"), (11, 99, "11+")]
heat, heat_n = [], []
for d in [1, 2, 3, 4]:
    row, nrow = [], []
    for lo, hi, _ in dist_bins:
        sub = df[(df.down == d) & df.ydstogo.between(lo, hi)]
        row.append(r(sub.sack.mean(), 4) if len(sub) >= 100 else None)
        nrow.append(len(sub))
    heat.append(row)
    heat_n.append(nrow)
fig_heat = {
    "type": "heatmap", "height": 230,
    "x": {"label": "Yards to go", "categories": [b[2] for b in dist_bins]},
    "y": {"label": "Down", "categories": ["1st", "2nd", "3rd", "4th"]},
    "values": heat, "counts": heat_n, "format": "pct1",
    "scale": {"type": "seq", "min": 0, "max": 0.12},
}

# Pressure (charted by FTN from 2023 on): the post-snap event that sacks come from
p23 = df[(df.season == 2023) & df.was_pressure.notna()].copy()
p23["was_pressure"] = p23.was_pressure.astype(bool)
pressure_rate = p23.was_pressure.mean()
sack_if_pressured = p23[p23.was_pressure].sack.mean()
sack_if_clean = p23[~p23.was_pressure].sack.mean()
share_of_sacks_pressured = p23[p23.sack == 1].was_pressure.mean()
print("pressure", pressure_rate, sack_if_pressured, sack_if_clean, share_of_sacks_pressured)
rush_groups = [(lambda s: s <= 3, "3 or fewer"), (lambda s: s == 4, "4"), (lambda s: s == 5, "5"),
               (lambda s: s >= 6, "6+")]
pr_rows = []
for f, lab in rush_groups:
    sub = p23[f(p23.number_of_pass_rushers)]
    pr_rows.append((lab, sub.was_pressure.mean(), sub.sack.mean(), sub[sub.was_pressure].sack.mean(), len(sub)))
fig_pressure = {
    "type": "bar", "orient": "v", "height": 280,
    "categories": [x[0] for x in pr_rows],
    "y": {"label": "Share of 2023 dropbacks", "format": "pct0", "min": 0},
    "series": [{"name": "Pressured", "color": 2, "values": [r(x[1], 4) for x in pr_rows],
                "tips": [[["Dropbacks", f"{x[4]:,}"]] for x in pr_rows]},
               {"name": "Sacked", "color": 1, "values": [r(x[2], 4) for x in pr_rows],
                "tips": [[["Sack rate when pressured", f"{x[3]:.0%}"]] for x in pr_rows]}],
}
print(pr_rows)

# --- Models: train on 2020-22, test on 2023
train, test = df[df.season < 2023], df[df.season == 2023]
X_train = pd.get_dummies(train[FEATURES], columns=["down"], dtype=float)
X_test = pd.get_dummies(test[FEATURES], columns=["down"], dtype=float).reindex(columns=X_train.columns, fill_value=0)
y_train, y_test = train.sack.astype(int), test.sack.astype(int)

models = {
    "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=50, n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, random_state=42, eval_metric="logloss"),
}
preds, scores = {}, []
baseline = np.full(len(y_test), y_train.mean())
scores.append(("Always predict the average", brier_score_loss(y_test, baseline), 0.5, log_loss(y_test, baseline)))
for name, m in models.items():
    m.fit(X_train, y_train)
    p = m.predict_proba(X_test)[:, 1]
    preds[name] = p
    scores.append((name, brier_score_loss(y_test, p), roc_auc_score(y_test, p), log_loss(y_test, p)))
for s in scores:
    print(f"{s[0]:28s} brier={s[1]:.5f} auc={s[2]:.3f} logloss={s[3]:.4f}")
b0 = scores[0][1]
fig_scores = {
    "type": "bar", "orient": "h", "height": 200,
    "categories": [s[0] for s in scores],
    "y": {"label": "AUC on 2023 dropbacks (0.5 = coin flip)", "format": "dec2", "min": 0.5, "max": 0.75},
    "series": [{"name": "AUC", "values": [r(s[2]) for s in scores],
                "colors": ["gray"] + [1] * (len(scores) - 1),
                "tips": [[["Brier score", f"{s[1]:.4f}"], ["Brier skill vs. average", f"{1 - s[1] / b0:+.1%}"]]
                         for s in scores]}],
}

# Logistic regression coefficients as odds ratios per 1 SD (or per category)
lr = models["Logistic regression"]
coefs = pd.Series(lr[-1].coef_[0], index=X_train.columns)
sds = X_train.std()
NAMES = {"ydstogo": "Yards to go", "yardline_100": "Yards to end zone", "game_seconds_remaining": "Time remaining",
         "score_differential": "Score differential", "defenders_in_box": "Defenders in box",
         "number_of_pass_rushers": "Pass rushers", "shotgun": "Shotgun", "xpass": "Pass likelihood (xpass)",
         "obvious_pass": "Obvious passing down", "down_1.0": "1st down", "down_2.0": "2nd down",
         "down_3.0": "3rd down", "down_4.0": "4th down"}
odds = np.exp(coefs).sort_values()
print(odds.round(3))
fig_odds = {
    "type": "bar", "orient": "h", "baseline": 1,
    "categories": [NAMES.get(k, k) for k in odds.index],
    "y": {"label": "Odds multiplier for a sack, per 1 standard deviation", "format": "dec2",
          "min": 0.85, "max": 1.3},
    "series": [{"name": "Odds ratio", "values": [r(v) for v in odds],
                "colors": [1 if v >= 1 else 2 for v in odds],
                "tips": [[["Raw coefficient", f"{coefs[k]:+.3f}"], ["1 SD of feature", f"{sds[k]:.2f}"]]
                         for k in odds.index]}],
    "legend": [{"name": "Raises sack odds", "color": 1}, {"name": "Lowers sack odds", "color": 2}],
}

# Calibration of the logistic model
frac, mean_pred = calibration_curve(y_test, preds["Logistic regression"], n_bins=10, strategy="quantile")
fig_calib = {
    "type": "line", "height": 300, "square": True,
    "x": {"label": "Predicted sack probability", "format": "pct0", "min": 0, "max": 0.16},
    "y": {"label": "Actual sack rate", "format": "pct0", "min": 0, "max": 0.16},
    "markers": True,
    "series": [{"name": "Perfect calibration", "color": "gray", "dash": True, "points": [[0, 0], [0.16, 0.16]]},
               {"name": "Logistic regression (2023)", "color": 1,
                "points": [[r(x, 4), r(y, 4)] for x, y in zip(mean_pred, frac)]}],
}
lr_p = preds["Logistic regression"]
top = test.assign(p=lr_p).sort_values("p")
dec_lo, dec_hi = top.head(len(top) // 10).sack.mean(), top.tail(len(top) // 10).sack.mean()
print("lowest decile", dec_lo, "highest decile", dec_hi, "pred range", lr_p.min(), lr_p.max())


# --- Follow-up: is turning pressure into sacks a QB trait? (FTN pressure data, 2023-2025)
newer = load_pbp([2023, 2024, 2025], participation=True)
nd = newer[(newer.season_type == "REG") & (newer["pass"] == 1) & (newer.play_type != "no_play")
           & (newer.qb_spike == 0) & newer.passer_id.notna()].copy()
nd["was_pressure"] = nd.was_pressure.astype(bool)
qb = (nd.groupby(["season", "passer_id", "passer", "posteam"])
      .agg(dropbacks=("sack", "size"), pressures=("was_pressure", "sum"), sacks=("sack", "sum"))
      .reset_index())
# A QB can play for two teams in a season; keep his main team.
qb = (qb.sort_values("dropbacks").groupby(["season", "passer_id"])
      .agg(passer=("passer", "last"), team=("posteam", "last"), dropbacks=("dropbacks", "sum"),
           pressures=("pressures", "sum"), sacks=("sacks", "sum")).reset_index())
qb["pressure_rate"] = qb.pressures / qb.dropbacks
qb["p2s"] = qb.sacks / qb.pressures
qb["sack_rate"] = qb.sacks / qb.dropbacks
qbq = qb[qb.pressures >= 100]

q23 = qbq[qbq.season == 2023].sort_values("p2s")
fig_qb = {
    "type": "bar", "orient": "h", "compact": True,
    "categories": [f"{x.passer} ({x.team})" for x in q23.itertuples()],
    "y": {"label": "Share of pressures that became sacks, 2023", "format": "pct0", "min": 0},
    "rules": [{"axis": "y", "value": r(sack_if_pressured, 4), "label": "League"}],
    "series": [{"name": "Pressure-to-sack rate", "color": 1, "values": [r(v, 4) for v in q23.p2s],
                "tips": [[["Pressures", f"{int(x.pressures)}"], ["Sacks", f"{int(x.sacks)}"],
                          ["Pressured on", f"{x.pressure_rate:.0%} of dropbacks"]] for x in q23.itertuples()]}],
}
print(q23[["passer", "team", "pressures", "p2s", "pressure_rate"]].round(3).to_string())

nx = qbq.copy()
nx["season"] -= 1
qpairs = qbq.merge(nx, on=["season", "passer_id"], suffixes=("", "_next"))
qstab = {k: qpairs[k].corr(qpairs[k + "_next"]) for k in ["p2s", "pressure_rate", "sack_rate"]}
print("QB year-to-year (n=%d):" % len(qpairs), {k: round(v, 2) for k, v in qstab.items()})
rng = np.random.default_rng(3)
qstab_ci = {}
for k in qstab:
    bs = []
    for _ in range(5000):
        smp = qpairs.iloc[rng.integers(0, len(qpairs), len(qpairs))]
        bs.append(smp[k].corr(smp[k + "_next"]))
    qstab_ci[k] = np.quantile(bs, [0.025, 0.975])
print("95% CIs:", {k: v.round(2).tolist() for k, v in qstab_ci.items()})
fig_qb_stab = {
    "type": "scatter", "height": 360, "square": True,
    "x": {"label": "Pressure-to-sack rate, one season", "format": "pct0"},
    "y": {"label": "Pressure-to-sack rate, next season", "format": "pct0"},
    "trend": {"slope": 1, "intercept": 0, "label": "Same both years"},
    "groups": [{"name": "QB, consecutive seasons (2023→24, 2024→25)", "color": 1,
                "labels": ["J.Allen", "P.Mahomes", "B.Young", "Z.Wilson", "J.Burrow", "L.Jackson", "C.Williams", "J.Goff", "T.Tagovailoa"],
                "points": [{"x": r(x.p2s, 4), "y": r(x.p2s_next, 4), "label": x.passer,
                            "tip": [["QB", f"{x.passer} ({int(x.season)}→{int(x.season) + 1})"],
                                    ["First season", f"{x.p2s:.1%}"], ["Next season", f"{x.p2s_next:.1%}"]]}
                           for x in qpairs.itertuples()]}],
}

# Does knowing the quarterback help the pre-snap model? Train on 2024, test on 2025.
# QB features come from the *previous* season, so nothing leaks from the season being predicted.
nd["obvious_pass"] = ((nd.down == 3) & (nd.ydstogo >= 6)).astype(int)
nd["shotgun"] = nd.shotgun.astype(float)
league_prior = qb.groupby("season").apply(lambda g: pd.Series({"p2s": g.sacks.sum() / g.pressures.sum(),
                                                               "pressure_rate": g.pressures.sum() / g.dropbacks.sum()}))
prior = qb[qb.pressures >= 50][["season", "passer_id", "p2s", "pressure_rate"]].copy()
prior["season"] += 1
nd = nd.merge(prior.rename(columns={"p2s": "qb_prior_p2s", "pressure_rate": "qb_prior_pressure"}),
              on=["season", "passer_id"], how="left")
for col, key in (("qb_prior_p2s", "p2s"), ("qb_prior_pressure", "pressure_rate")):
    nd[col] = nd[col].fillna(nd.season.map(league_prior[key].shift(1)).fillna(league_prior[key].iloc[0]))
nd = nd.dropna(subset=FEATURES)
nd = nd[nd.down.between(1, 4) & nd.number_of_pass_rushers.between(1, 11)]
tr, te = nd[nd.season == 2024], nd[nd.season == 2025]
qb_scores = {}
for name, extra in (("Pre-snap features", []), ("+ QB's prior-season sack tendencies", ["qb_prior_p2s", "qb_prior_pressure"])):
    Xa = pd.get_dummies(tr[FEATURES + extra], columns=["down"], dtype=float)
    Xb = pd.get_dummies(te[FEATURES + extra], columns=["down"], dtype=float).reindex(columns=Xa.columns, fill_value=0)
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xa, tr.sack)
    pp = m.predict_proba(Xb)[:, 1]
    qb_scores[name] = (roc_auc_score(te.sack, pp), brier_score_loss(te.sack, pp))
base_b = brier_score_loss(te.sack, np.full(len(te), tr.sack.mean()))
for k, (a, b) in qb_scores.items():
    print(f"{k:40s} AUC={a:.3f} Brier={b:.5f} skill={1 - b / base_b:.4f}")

write("sackPredictionLR", {
    "charts": {"byRushers": fig_rushers, "downDistance": fig_heat, "pressure": fig_pressure, "qbP2s": fig_qb, "qbStability": fig_qb_stab,
               "models": fig_scores, "odds": fig_odds, "calibration": fig_calib},
    "stats": {"dropbacks": len(df), "train": len(train), "test": len(test), "base": r(base, 4),
              "scores": [{"name": s[0], "brier": r(s[1], 5), "auc": r(s[2]), "skill": r(1 - s[1] / b0, 4)} for s in scores],
              "decileLow": r(dec_lo, 4), "decileHigh": r(dec_hi, 4), "pressureRate": r(pressure_rate, 4), "sackIfPressured": r(sack_if_pressured, 4),
              "sackIfClean": r(sack_if_clean, 5), "sacksPressured": r(share_of_sacks_pressured, 4),
              "qbStab": {k: r(v) for k, v in qstab.items()}, "qbStabCi": {k: [r(c) for c in v] for k, v in qstab_ci.items()}, "qbPairs": len(qpairs),
              "qbModel": {k: {"auc": r(a), "skill": r(1 - b / base_b, 4)} for k, (a, b) in qb_scores.items()},
              "downs": [[d[0], r(d[1], 4), d[2]] for d in downs], "rushers": [[d[0], r(d[1], 4), d[2]] for d in rushers]},
})
