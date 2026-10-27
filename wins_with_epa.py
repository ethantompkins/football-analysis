"""Which EPA metrics track (and predict) wins? 2020-2023 regular seasons."""
from common import load_games, load_pbp, np, pd, r, write

YEARS = [2020, 2021, 2022, 2023]          # the main analysis
ALL_YEARS = YEARS + [2024, 2025]          # extra seasons for the forecasting test
OFF, DEF, ALL = 1, 2, "ink"

pbp = load_pbp(ALL_YEARS, columns=["season", "season_type", "week", "pass", "rush", "epa", "posteam", "defteam"])
pbp = pbp[pbp.season_type == "REG"]
plays = pbp[(pbp["pass"] == 1) | (pbp["rush"] == 1)].dropna(subset=["epa", "posteam", "defteam"])

games = load_games()
games = games[games.season.isin(ALL_YEARS) & (games.game_type == "REG")].dropna(subset=["result"])
rows = []
for g in games.itertuples():
    for team, opp, margin in ((g.home_team, g.away_team, g.result), (g.away_team, g.home_team, -g.result)):
        rows.append({"season": g.season, "week": g.week, "team": team, "opp": opp, "margin": margin,
                     "win": 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)})
results = pd.DataFrame(rows)


def epa_table(p):
    off = p.groupby(["season", "posteam"])
    dfn = p.groupby(["season", "defteam"])
    t = pd.DataFrame({
        "off_epa": off.epa.mean(),
        "off_pass": p[p["pass"] == 1].groupby(["season", "posteam"]).epa.mean(),
        "off_rush": p[p.rush == 1].groupby(["season", "posteam"]).epa.mean(),
    })
    t.index.names = ["season", "team"]
    d = pd.DataFrame({
        "def_epa": dfn.epa.mean(),
        "def_pass": p[p["pass"] == 1].groupby(["season", "defteam"]).epa.mean(),
        "def_rush": p[p.rush == 1].groupby(["season", "defteam"]).epa.mean(),
    })
    d.index.names = ["season", "team"]
    t = t.join(d)
    t["net_epa"] = t.off_epa - t.def_epa
    return t


def record(res):
    g = res.groupby(["season", "team"])
    return pd.DataFrame({"win_pct": g.win.mean(), "wins": g.win.sum(), "games": g.win.size(),
                         "point_diff": g.margin.sum(), "pd_per_game": g.margin.mean()})


season = epa_table(plays[plays.season.isin(YEARS)]).join(record(results[results.season.isin(YEARS)]))
print(season.describe().round(3).T[["mean", "min", "max"]])

# Defensive EPA is "allowed", so lower is better: flip its sign for correlations so every bar
# reads "higher is better" and correlations are positive.
METRICS = [
    ("pd_per_game", "Point differential / game", ALL),
    ("net_epa", "Net EPA / play (off − def)", ALL),
    ("off_epa", "Offense EPA / play", OFF),
    ("off_pass", "Offense pass EPA / play", OFF),
    ("off_rush", "Offense rush EPA / play", OFF),
    ("def_epa", "Defense EPA / play (allowed)", DEF),
    ("def_pass", "Defense pass EPA / play (allowed)", DEF),
    ("def_rush", "Defense rush EPA / play (allowed)", DEF),
]
sign = {k: (-1 if k.startswith("def") else 1) for k, _, _ in METRICS}
corrs = {k: sign[k] * season[k].corr(season.win_pct) for k, _, _ in METRICS}
for k, v in corrs.items():
    print(f"{k:12s} r={v:+.3f} r2={v * v:.3f}")
order = sorted(METRICS, key=lambda m: -corrs[m[0]])
fig_corr = {
    "type": "bar", "orient": "h",
    "categories": [m[1] for m in order],
    "y": {"label": "Correlation with win % (r)", "format": "dec2", "min": 0, "max": 1},
    "series": [{"name": "Correlation", "values": [r(corrs[m[0]]) for m in order],
                "colors": [m[2] for m in order],
                "tips": [[["r²", f"{corrs[m[0]] ** 2:.2f}"], ["Team-seasons", str(len(season))]] for m in order]}],
    "legend": [{"name": "Offense", "color": OFF}, {"name": "Defense (sign flipped)", "color": DEF},
               {"name": "Whole team", "color": ALL}],
}

# Linear fit: win % from net EPA
slope, intercept = np.polyfit(season.net_epa, season.win_pct, 1)
resid = season.win_pct - (slope * season.net_epa + intercept)
season["resid_wins"] = resid * season.games
lucky = season.sort_values("resid_wins")
print("unluckiest", lucky.head(4)[["net_epa", "wins", "resid_wins"]].round(2))
print("luckiest", lucky.tail(4)[["net_epa", "wins", "resid_wins"]].round(2))
label_keys = list(lucky.head(3).index) + list(lucky.tail(3).index) + \
    list(season.net_epa.nlargest(2).index) + list(season.net_epa.nsmallest(2).index)


def ts_point(idx, row):
    yr, tm = idx
    return {"x": r(row.net_epa), "y": r(row.win_pct), "label": f"{tm} {str(yr)[2:]}",
            "tip": [["Team", f"{yr} {tm}"], ["Record", f"{row.wins:g}–{row.games - row.wins:g}"],
                    ["Net EPA/play", f"{row.net_epa:+.3f}"],
                    ["Wins vs. expected", f"{row.resid_wins:+.1f}"]]}


fig_net = {
    "type": "scatter", "height": 400,
    "x": {"label": "Net EPA per play (offense minus defense)", "format": "signed2"},
    "y": {"label": "Win %", "format": "pct0", "min": 0, "max": 1},
    "trend": {"slope": r(slope, 5), "intercept": r(intercept, 5), "label": "Linear fit"},
    "groups": [{"name": "Team-season", "color": 1, "r": 4,
                "points": [ts_point(i, row) for i, row in season.iterrows()],
                "labels": [f"{tm} {str(yr)[2:]}" for yr, tm in label_keys]}],
}

# 2023 offense vs defense map
s23 = season.loc[2023]
fig_map = {
    "type": "scatter", "height": 440, "square": True,
    "x": {"label": "Offense EPA per play →  better", "format": "signed2"},
    "y": {"label": "Defense EPA allowed per play →  better", "format": "signed2", "reverse": True},
    "rules": [{"axis": "x", "value": r(s23.off_epa.mean())}, {"axis": "y", "value": r(s23.def_epa.mean())}],
    "quadrants": {"tr": "Good offense, good defense", "tl": "Defense carries", "br": "Offense carries",
                  "bl": "Neither"},
    "groups": [{"name": "Playoff team", "color": 1, "labelAll": True, "points": []},
               {"name": "Missed playoffs", "color": "gray", "labelAll": True, "points": []}],
}
playoff = set(load_games().query("season == 2023 and game_type != 'REG'")[["home_team", "away_team"]]
              .values.ravel())
for tm, row in s23.iterrows():
    fig_map["groups"][0 if tm in playoff else 1]["points"].append(
        {"x": r(row.off_epa), "y": r(row.def_epa), "label": tm,
         "tip": [["Record", f"{row.wins:g}–{row.games - row.wins:g}"],
                 ["Offense EPA/play", f"{row.off_epa:+.3f}"], ["Defense EPA allowed", f"{row.def_epa:+.3f}"]]})

# Predictive test: first half of the season vs. win % in the second half
def season_halves(yr):
    weeks = sorted(results[results.season == yr].week.unique())
    cut = weeks[len(weeks) // 2]
    p1 = plays[(plays.season == yr) & (plays.week < cut)]
    first = epa_table(p1).join(record(results[(results.season == yr) & (results.week < cut)]))
    second = record(results[(results.season == yr) & (results.week >= cut)])
    return first.join(second.add_suffix("_2nd"))


halves = pd.concat([season_halves(yr) for yr in YEARS])
PRED = [("win_pct", "Win % (first half)", ALL)] + METRICS[:5] + METRICS[5:]
pred = {k: (-1 if k.startswith("def") else 1) * halves[k].corr(halves.win_pct_2nd) for k, _, _ in PRED}
for k, v in pred.items():
    print(f"predict 2nd half: {k:12s} r={v:+.3f}")
porder = sorted(PRED, key=lambda m: -pred[m[0]])
fig_pred = {
    "type": "bar", "orient": "h",
    "categories": [m[1].replace(" (first half)", "") for m in porder],
    "y": {"label": "Correlation of first-half metric with second-half win %", "format": "dec2", "min": 0},
    "series": [{"name": "Correlation", "values": [r(pred[m[0]]) for m in porder],
                "colors": ["gray" if m[0] == "win_pct" else m[2] for m in porder]}],
    "legend": [{"name": "Offense", "color": OFF}, {"name": "Defense (sign flipped)", "color": DEF},
               {"name": "Whole team", "color": ALL}, {"name": "Record so far", "color": "gray"}],
}


# --- Follow-up 1: how precise are the correlations? Bootstrap the 128 team-seasons.
rng = np.random.default_rng(11)
boot = {k: [] for k, _, _ in METRICS}
for _ in range(5000):
    sample = season.iloc[rng.integers(0, len(season), len(season))]
    for k, _, _ in METRICS:
        boot[k].append(sign[k] * sample[k].corr(sample.win_pct))
boot = {k: np.array(v) for k, v in boot.items()}
corr_ci = {k: np.quantile(v, [0.025, 0.975]) for k, v in boot.items()}
p_off_gt_def = (boot["off_epa"] > boot["def_epa"]).mean()
p_pass_gt_rush = (boot["off_pass"] > boot["off_rush"]).mean()
for k, v in corr_ci.items():
    print(f"{k:12s} r={corrs[k]:.3f} 95% CI [{v[0]:.3f}, {v[1]:.3f}]")
print("P(offense r > defense r) =", p_off_gt_def, " P(pass r > rush r) =", p_pass_gt_rush)

# --- Follow-up 2: does this season predict next season? (2020->21 ... 2024->25)
full = epa_table(plays).join(record(results))
nxt = full.reset_index()
nxt["season"] -= 1
pairs = full.join(nxt.set_index(["season", "team"])[["win_pct"]].add_suffix("_next"), how="inner")
YOY = [("win_pct", "Win %"), ("pd_per_game", "Point differential / game"), ("net_epa", "Net EPA / play"),
       ("off_epa", "Offense EPA / play"), ("def_epa", "Defense EPA / play (allowed)")]
yoy = {k: ((-1 if k.startswith("def") else 1) * pairs[k].corr(pairs.win_pct_next)) for k, _ in YOY}
same = {k: ((-1 if k.startswith("def") else 1) * full[k].corr(full.win_pct)) for k, _ in YOY}
print("next-season r:", {k: round(v, 3) for k, v in yoy.items()}, "n =", len(pairs))

# --- Follow-up 3: a real forecast. Fit on 2020-23 first halves, predict 2024-25 second halves.
from sklearn.linear_model import LinearRegression

test_h = pd.concat([season_halves(yr) for yr in [2024, 2025]])
MODELS = [("Everyone finishes .500", None), ("First-half record, as is", "asis"),
          ("Regression: record", ["win_pct"]), ("Regression: net EPA", ["net_epa"]),
          ("Regression: net EPA + record", ["net_epa", "win_pct"]),
          ("Regression: offense, defense + record", ["off_epa", "def_epa", "win_pct"])]
forecast, errors = [], {}
for name, feats in MODELS:
    if feats is None:
        fc = np.full(len(test_h), 0.5)
    elif feats == "asis":
        fc = test_h.win_pct.to_numpy()
    else:
        lm = LinearRegression().fit(halves[feats], halves.win_pct_2nd)
        fc = lm.predict(test_h[feats])
        if feats == ["net_epa", "win_pct"]:
            best_pred, best_coef = fc, dict(zip(feats, lm.coef_))
    errors[name] = np.abs(fc - test_h.win_pct_2nd.to_numpy()) * test_h.games_2nd.to_numpy()
    mae_wins = errors[name].mean()
    forecast.append((name, mae_wins))
    print(f"{name:40s} MAE = {mae_wins:.2f} wins over the second half")
print("net EPA + record coefficients:", best_coef)
# Is the gap between the record and net EPA forecasts real? Paired bootstrap over the 64 teams.
d = errors["Regression: net EPA"] - errors["First-half record, as is"]
gap_ci = np.quantile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)], [0.025, 0.975])
test_corr = {k: test_h[k].corr(test_h.win_pct_2nd) for k in ["win_pct", "net_epa", "pd_per_game"]}
print("MAE gap (net EPA - record):", round(d.mean(), 2), "95% CI", gap_ci.round(2), "2024-25 r:", {k: round(v, 3) for k, v in test_corr.items()})

fig_forecast = {
    "type": "bar", "orient": "h", "height": 250,
    "categories": [f[0] for f in forecast],
    "y": {"label": "Average miss, in wins, over the second half of 2024 and 2025", "format": "dec2", "min": 0},
    "series": [{"name": "Mean absolute error (wins)", "values": [r(f[1]) for f in forecast],
                "colors": ["gray", "ink", "ink", 1, 1, 1]}],
    "legend": [{"name": "No information", "color": "gray"}, {"name": "Record only", "color": "ink"},
               {"name": "Uses EPA", "color": 1}],
}
fig_forecast_pts = {
    "type": "scatter", "height": 380, "square": True,
    "x": {"label": "Forecast second-half win % (net EPA + record)", "format": "pct0", "min": 0, "max": 1},
    "y": {"label": "Actual second-half win %", "format": "pct0", "min": 0, "max": 1},
    "trend": {"slope": 1, "intercept": 0, "label": "Perfect forecast"},
    "groups": [{"name": "2024 and 2025 teams", "color": 1, "r": 4, "points": [
        {"x": r(pv), "y": r(row.win_pct_2nd), "label": f"{tm} {str(yr)[2:]}",
         "tip": [["Team", f"{yr} {tm}"], ["First half", f"{row.wins:g}–{row.games - row.wins:g}"],
                 ["Forecast 2nd half", f"{pv * row.games_2nd:.1f} wins"],
                 ["Actual 2nd half", f"{row.wins_2nd:g}–{row.games_2nd - row.wins_2nd:g}"]]}
        for pv, ((yr, tm), row) in zip(best_pred, test_h.iterrows())]}],
}
miss = pd.Series(np.abs(best_pred - test_h.win_pct_2nd.to_numpy()) * test_h.games_2nd.to_numpy(), index=test_h.index)
print("biggest misses:", miss.sort_values().tail(4).round(1).to_dict())

write("winsWithEpa", {
    "charts": {"correlations": fig_corr, "netEpa": fig_net, "map2023": fig_map, "predictive": fig_pred,
               "forecast": fig_forecast, "forecastPoints": fig_forecast_pts},
    "stats": {"n": len(season), "corr": {k: r(v) for k, v in corrs.items()},
              "pred": {k: r(v) for k, v in pred.items()},
              "slope": r(slope, 4), "intercept": r(intercept, 4),
              "winsPerTenth": r(slope * 0.1 * 17, 2),
              "corrCi": {k: [r(v[0]), r(v[1])] for k, v in corr_ci.items()},
              "pOffGtDef": r(p_off_gt_def), "pPassGtRush": r(p_pass_gt_rush),
              "yoy": {k: r(v) for k, v in yoy.items()}, "same": {k: r(v) for k, v in same.items()}, "yoyN": len(pairs),
              "forecast": [[f[0], r(f[1])] for f in forecast], "gapCi": [r(g, 2) for g in gap_ci],
              "testCorr": {k: r(v) for k, v in test_corr.items()}},
})
