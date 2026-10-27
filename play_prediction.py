"""Predicting run vs. pass with a random forest: train 2020-2022, test on 2023."""
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from common import load_pbp, np, pd, r, write

pbp = load_pbp([2020, 2021, 2022, 2023])
plays = pbp[(pbp.season_type == "REG") & pbp.play_type.isin(["pass", "run"])
            & ((pbp["pass"] == 1) | (pbp.rush == 1)) & pbp.down.notna()].copy()
plays["is_pass"] = (plays["pass"] == 1).astype(int)
plays["obvious_pass"] = (((plays.down == 3) & (plays.ydstogo >= 7))
                         | ((plays.score_differential <= -10) & (plays.game_seconds_remaining <= 600))).astype(int)
plays["two_minute"] = (plays.half_seconds_remaining <= 120).astype(int)
plays["is_home"] = (plays.posteam == plays.home_team).astype(int)

SITUATION = ["down", "ydstogo", "yardline_100", "qtr", "half_seconds_remaining", "game_seconds_remaining",
             "score_differential", "goal_to_go", "obvious_pass", "two_minute", "is_home"]
PRESNAP = SITUATION + ["shotgun", "no_huddle"]
plays = plays.dropna(subset=PRESNAP + ["xpass"])
train, test = plays[plays.season < 2023], plays[plays.season == 2023].copy()
print(f"train={len(train):,} test={len(test):,} pass rate train={train.is_pass.mean():.3f} test={test.is_pass.mean():.3f}")

# League-wide tendencies: pass rate by down and distance
dist_bins = [(1, 2, "1–2"), (3, 4, "3–4"), (5, 7, "5–7"), (8, 10, "8–10"), (11, 99, "11+")]
heat, heat_n = [], []
for d in [1, 2, 3, 4]:
    row, nrow = [], []
    for lo, hi, _ in dist_bins:
        sub = plays[(plays.down == d) & plays.ydstogo.between(lo, hi)]
        row.append(r(sub.is_pass.mean(), 3) if len(sub) >= 150 else None)
        nrow.append(len(sub))
    heat.append(row)
    heat_n.append(nrow)
fig_heat = {
    "type": "heatmap", "height": 230,
    "x": {"label": "Yards to go", "categories": [b[2] for b in dist_bins]},
    "y": {"label": "Down", "categories": ["1st", "2nd", "3rd", "4th"]},
    "values": heat, "counts": heat_n, "format": "pct0",
    "scale": {"type": "div", "min": 0.2, "max": 0.95, "mid": 0.5},
}
print(np.array(heat))


def fit(features):
    m = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=20, n_jobs=-1, random_state=42)
    m.fit(train[features], train.is_pass)
    return m, m.predict_proba(test[features])[:, 1]


rf_sit, p_sit = fit(SITUATION)
rf_pre, p_pre = fit(PRESNAP)
test["p_sit"], test["p_pre"] = p_sit, p_pre

results = [
    ("Always guess pass", np.full(len(test), train.is_pass.mean()), "gray"),
    ("nflfastR xpass", test.xpass.to_numpy(), "gray"),
    ("Random forest: game situation", p_sit, 1),
    ("Random forest: + shotgun & no-huddle", p_pre, 1),
]
scores = []
for name, p, color in results:
    acc = accuracy_score(test.is_pass, p >= 0.5)
    auc = roc_auc_score(test.is_pass, p) if p.std() > 0 else 0.5
    scores.append({"name": name, "acc": acc, "auc": auc, "brier": brier_score_loss(test.is_pass, p),
                   "logloss": log_loss(test.is_pass, p), "color": color})
    print(f"{name:40s} acc={acc:.3f} auc={auc:.3f} brier={scores[-1]['brier']:.4f}")
fig_acc = {
    "type": "bar", "orient": "h", "height": 220,
    "categories": [s["name"] for s in scores],
    "y": {"label": "Accuracy on every 2023 run/pass play", "format": "pct0", "min": 0.5, "max": 0.85},
    "series": [{"name": "Accuracy", "values": [r(s["acc"], 4) for s in scores], "colors": [s["color"] for s in scores],
                "tips": [[["AUC", f"{s['auc']:.3f}"], ["Brier score", f"{s['brier']:.4f}"]] for s in scores]}],
}

# Classification report for the situational model
pred = (p_sit >= 0.5).astype(int)
y = test.is_pass.to_numpy()
cm = [[int(((y == a) & (pred == b)).sum()) for b in (0, 1)] for a in (0, 1)]
report = {}
for lab, cls in (("Run", 0), ("Pass", 1)):
    tp = ((pred == cls) & (y == cls)).sum()
    prec, rec = tp / (pred == cls).sum(), tp / (y == cls).sum()
    report[lab] = {"precision": r(prec), "recall": r(rec), "f1": r(2 * prec * rec / (prec + rec)), "n": int((y == cls).sum())}
print(cm, report)

# Permutation importance on the test season
NAMES = {"down": "Down", "ydstogo": "Yards to go", "yardline_100": "Yards to end zone", "qtr": "Quarter",
         "half_seconds_remaining": "Seconds left in half", "game_seconds_remaining": "Seconds left in game",
         "score_differential": "Score differential", "goal_to_go": "Goal to go", "obvious_pass": "Obvious passing down",
         "two_minute": "Two-minute drill", "is_home": "Home team", "shotgun": "Shotgun", "no_huddle": "No huddle"}
sample = test.sample(20_000, random_state=0)
imp = permutation_importance(rf_pre, sample[PRESNAP], sample.is_pass, scoring="accuracy", n_repeats=5,
                             random_state=0, n_jobs=-1)
imp = pd.Series(imp.importances_mean, index=PRESNAP).sort_values(ascending=False)
print(imp.round(4))
fig_imp = {
    "type": "bar", "orient": "h",
    "categories": [NAMES[k] for k in imp.index],
    "y": {"label": "Accuracy lost when the feature is shuffled", "format": "pct1", "min": 0},
    "series": [{"name": "Permutation importance", "color": 1, "values": [r(max(v, 0), 4) for v in imp]}],
}

# Calibration
frac, mean_p = calibration_curve(test.is_pass, p_pre, n_bins=10)
fig_calib = {
    "type": "line", "height": 300, "square": True,
    "x": {"label": "Predicted pass probability", "format": "pct0", "min": 0, "max": 1},
    "y": {"label": "Actual pass rate", "format": "pct0", "min": 0, "max": 1},
    "markers": True,
    "series": [{"name": "Perfect calibration", "color": "gray", "dash": True, "points": [[0, 0], [1, 1]]},
               {"name": "Random forest (2023)", "color": 1,
                "points": [[r(a, 3), r(b, 3)] for a, b in zip(mean_p, frac)]}],
}
confident = (np.abs(p_pre - 0.5) >= 0.3)
print("confident share", confident.mean(), "acc there", accuracy_score(y[confident], p_pre[confident] >= 0.5))

# Predictability by team in 2023
test["correct"] = ((p_pre >= 0.5).astype(int) == test.is_pass).astype(int)
teams = test.groupby("posteam").agg(acc=("correct", "mean"), pass_rate=("is_pass", "mean"),
                                   gap=("is_pass", "mean")).sort_values("acc", ascending=False)
teams["gap"] = teams.pass_rate - test.groupby("posteam").p_pre.mean()
print(teams.head(5).round(3), teams.tail(5).round(3))
fig_teams = {
    "type": "bar", "orient": "h", "compact": True,
    "categories": list(teams.index),
    "y": {"label": "Model accuracy on that team's 2023 plays", "format": "pct0", "min": 0.6, "max": 0.82},
    "rules": [{"axis": "y", "value": r(scores[3]["acc"], 4), "label": "League"}],
    "series": [{"name": "Accuracy", "color": 1, "values": [r(v, 4) for v in teams.acc],
                "tips": [[["Actual pass rate", f"{row.pass_rate:.1%}"],
                          ["Pass rate vs. model", f"{row.gap:+.1%}"]] for row in teams.itertuples()]}],
}

# What-if scenarios
SCEN = [
    ("1st & 10 at own 25, tied, 1st quarter", dict(down=1, ydstogo=10, yardline_100=75, qtr=1, half_seconds_remaining=1500, game_seconds_remaining=3300, score_differential=0, goal_to_go=0, obvious_pass=0, two_minute=0, is_home=1, shotgun=0, no_huddle=0)),
    ("Same play, but in shotgun", dict(down=1, ydstogo=10, yardline_100=75, qtr=1, half_seconds_remaining=1500, game_seconds_remaining=3300, score_differential=0, goal_to_go=0, obvious_pass=0, two_minute=0, is_home=1, shotgun=1, no_huddle=0)),
    ("3rd & 2 at midfield, up 3, 3rd quarter", dict(down=3, ydstogo=2, yardline_100=50, qtr=3, half_seconds_remaining=900, game_seconds_remaining=900 + 900, score_differential=3, goal_to_go=0, obvious_pass=0, two_minute=0, is_home=1, shotgun=1, no_huddle=0)),
    ("3rd & 8 at own 30, down 4, 4th quarter", dict(down=3, ydstogo=8, yardline_100=70, qtr=4, half_seconds_remaining=400, game_seconds_remaining=400, score_differential=-4, goal_to_go=0, obvious_pass=1, two_minute=0, is_home=1, shotgun=1, no_huddle=0)),
    ("2nd & 6, up 10, 3:00 left in the game", dict(down=2, ydstogo=6, yardline_100=55, qtr=4, half_seconds_remaining=180, game_seconds_remaining=180, score_differential=10, goal_to_go=0, obvious_pass=0, two_minute=0, is_home=1, shotgun=0, no_huddle=0)),
    ("1st & goal at the 2, 2nd quarter", dict(down=1, ydstogo=2, yardline_100=2, qtr=2, half_seconds_remaining=600, game_seconds_remaining=2400, score_differential=0, goal_to_go=1, obvious_pass=0, two_minute=0, is_home=1, shotgun=0, no_huddle=0)),
]
scen = [(label, float(rf_pre.predict_proba(pd.DataFrame([f])[PRESNAP])[0, 1])) for label, f in SCEN]
for s in scen:
    print(f"{s[1]:.2f}  {s[0]}")
fig_scen = {
    "type": "bar", "orient": "h", "height": 270, "baseline": 0.5,
    "categories": [s[0] for s in scen],
    "y": {"label": "Predicted pass probability", "format": "pct0", "min": 0, "max": 1},
    "series": [{"name": "Pass probability", "values": [r(s[1], 3) for s in scen],
                "colors": [1 if s[1] >= 0.5 else 2 for s in scen]}],
    "legend": [{"name": "Leans pass", "color": 1}, {"name": "Leans run", "color": 2}],
}


# --- Follow-up: other model families, and does knowing the team help?
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Each team's pass rate over expected in the *previous* season (needs 2019 for the 2020 rows).
prev = load_pbp([2019], columns=["season", "season_type", "posteam", "pass", "rush", "xpass", "play_type"])
prev = prev[(prev.season_type == "REG") & prev.play_type.isin(["pass", "run"])]
both = pd.concat([prev, plays[["season", "posteam", "pass", "xpass"]]])
proe_by = both.dropna(subset=["xpass", "posteam"]).groupby(["season", "posteam"]).apply(
    lambda g: g["pass"].mean() - g.xpass.mean()).rename("prior_proe").reset_index()
proe_by["season"] += 1
for df_ in (train, test):
    df_["prior_proe"] = df_.merge(proe_by, on=["season", "posteam"], how="left").prior_proe.fillna(0).to_numpy()

team_cols = sorted(plays.posteam.unique())
def with_team(df_):
    return pd.concat([df_[PRESNAP].reset_index(drop=True),
                      pd.get_dummies(df_.posteam.reset_index(drop=True)).reindex(columns=team_cols, fill_value=0).astype(float)], axis=1)

variants = []
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(train[PRESNAP], train.is_pass)
variants.append(("Logistic regression", lr.predict_proba(test[PRESNAP])[:, 1], "gray"))
variants.append(("Random forest (from above)", p_pre, 1))
xgb = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    random_state=42, eval_metric="logloss").fit(train[PRESNAP], train.is_pass)
variants.append(("XGBoost", xgb.predict_proba(test[PRESNAP])[:, 1], 2))
rf_team = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=20, n_jobs=-1, random_state=42)
rf_team.fit(with_team(train), train.is_pass)
variants.append(("Random forest + team", rf_team.predict_proba(with_team(test))[:, 1], 1))
rf_proe, p_proe = fit(PRESNAP + ["prior_proe"])
variants.append(("Random forest + last season's PROE", p_proe, 1))
xgb_proe = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         random_state=42, eval_metric="logloss").fit(train[PRESNAP + ["prior_proe"]], train.is_pass)
variants.append(("XGBoost + last season's PROE", xgb_proe.predict_proba(test[PRESNAP + ["prior_proe"]])[:, 1], 2))

variant_scores = []
for name, pv, color in variants:
    acc = accuracy_score(test.is_pass, pv >= 0.5)
    variant_scores.append({"name": name, "acc": acc, "auc": roc_auc_score(test.is_pass, pv),
                           "brier": brier_score_loss(test.is_pass, pv), "color": color})
    print(f"{name:40s} acc={acc:.4f} auc={variant_scores[-1]['auc']:.4f} brier={variant_scores[-1]['brier']:.4f}")
fig_variants = {
    "type": "bar", "orient": "h", "height": 260,
    "categories": [v["name"] for v in variant_scores],
    "y": {"label": "Accuracy on 2023 plays", "format": "pct1", "min": 0.7, "max": 0.76},
    "series": [{"name": "Accuracy", "values": [r(v["acc"], 4) for v in variant_scores],
                "colors": [v["color"] for v in variant_scores],
                "tips": [[["AUC", f"{v['auc']:.3f}"], ["Brier score", f"{v['brier']:.4f}"]] for v in variant_scores]}],
    "legend": [{"name": "Logistic regression", "color": "gray"}, {"name": "Random forest", "color": 1},
               {"name": "XGBoost", "color": 2}],
}
# Team-level effect of the PROE feature
test["p_proe"] = p_proe
team_gain = test.assign(c0=((test.p_pre >= .5) == test.is_pass), c1=((test.p_proe >= .5) == test.is_pass)).groupby("posteam")[["c0", "c1"]].mean()
team_gain["gain"] = team_gain.c1 - team_gain.c0
print("teams helped most:", team_gain.gain.sort_values().tail(4).round(3).to_dict(), "hurt most:", team_gain.gain.sort_values().head(3).round(3).to_dict())
print("ATL/PHI/IND prior PROE:", proe_by[(proe_by.season == 2023) & proe_by.posteam.isin(["ATL", "PHI", "IND", "KC"])].round(3).to_dict("records"))

write("playPredictionRFC", {
    "charts": {"downDistance": fig_heat, "accuracy": fig_acc, "importance": fig_imp, "calibration": fig_calib,
               "teams": fig_teams, "scenarios": fig_scen, "variants": fig_variants},
    "stats": {"train": len(train), "test": len(test), "passRate": r(test.is_pass.mean(), 4),
              "scores": [{k: (r(v, 4) if isinstance(v, float) else v) for k, v in s.items()} for s in scores],
              "cm": cm, "report": report, "confidentShare": r(confident.mean(), 3),
              "confidentAcc": r(accuracy_score(y[confident], p_pre[confident] >= 0.5), 3),
              "teamsTop": [[t, r(a, 3)] for t, a in teams.acc.head(3).items()],
              "teamsBottom": [[t, r(a, 3)] for t, a in teams.acc.tail(3).items()],
              "scenarios": [[s[0], r(s[1], 3)] for s in scen],
              "variants": [{k: (r(v, 4) if isinstance(v, float) else v) for k, v in d.items()} for d in variant_scores],
              "teamGain": {k: r(v, 4) for k, v in team_gain.gain.items()}},
})
