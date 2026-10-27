"""Run rate vs. run efficiency, 2023: San Francisco vs. Minnesota."""
from common import load_games, load_pbp, np, pd, r, write

FOCUS = {"SF": 1, "MIN": 2}  # team -> categorical color slot
STATES = [(-99, -9, "Down 9+"), (-8, -1, "Down 1–8"), (0, 0, "Tied"), (1, 8, "Up 1–8"), (9, 99, "Up 9+")]

pbp = load_pbp([2023])
pbp = pbp[pbp.season_type == "REG"]
plays = pbp[(pbp["pass"] == 1) | (pbp["rush"] == 1)].dropna(subset=["epa", "posteam"]).copy()
plays["state"] = pd.cut(plays.score_differential, [s[0] - 0.5 for s in STATES] + [99.5],
                        labels=[s[2] for s in STATES])

runs = plays[plays.rush == 1]
team = pd.DataFrame({
    "run_rate": plays.groupby("posteam").rush.mean(),
    "rush_epa": runs.groupby("posteam").epa.mean(),
    "rush_success": runs.groupby("posteam").success.mean(),
    "ypc": runs.groupby("posteam").yards_gained.mean(),
    "pass_epa": plays[plays["pass"] == 1].groupby("posteam").epa.mean(),
})
neutral = plays[(plays.down <= 2) & plays.wp.between(0.2, 0.8) & (plays.half_seconds_remaining > 120)]
team["neutral_run_rate"] = neutral.groupby("posteam").rush.mean()
team["rush_epa_rank"] = team.rush_epa.rank(ascending=False).astype(int)
team["run_rate_rank"] = team.run_rate.rank(ascending=False).astype(int)
team["pass_epa_rank"] = team.pass_epa.rank(ascending=False).astype(int)
team["neutral_rank"] = team.neutral_run_rate.rank(ascending=False).astype(int)

league = {
    "run_rate": plays.rush.mean(),
    "rush_epa": runs.epa.mean(),
    "neutral_run_rate": neutral.rush.mean(),
}
for t in FOCUS:
    print(t, team.loc[t].round(3).to_dict())
print("league", {k: round(v, 3) for k, v in league.items()})

# Figure 1: every team's run rate vs. rushing efficiency
def team_point(t):
    row = team.loc[t]
    return {"x": r(row.run_rate), "y": r(row.rush_epa), "label": t,
            "tip": [["Run rate", f"{row.run_rate:.1%} (#{row.run_rate_rank})"],
                    ["Rush EPA/play", f"{row.rush_epa:+.3f} (#{row.rush_epa_rank})"]]}

others = [t for t in team.index if t not in FOCUS]
fig_scatter = {
    "type": "scatter", "height": 380,
    "x": {"label": "Run rate (share of run + pass plays)", "format": "pct0"},
    "y": {"label": "Rush EPA per play", "format": "signed2"},
    "rules": [{"axis": "x", "value": r(league["run_rate"]), "label": "League avg"},
              {"axis": "y", "value": r(league["rush_epa"]), "label": "League avg"}],
    "groups": [{"name": "Other teams", "color": "gray", "points": [team_point(t) for t in others],
                "labels": ["LAC", "BAL", "WAS", "KC", "ATL", "CHI"]}]
              + [{"name": t, "color": c, "points": [team_point(t)], "labels": [t]} for t, c in FOCUS.items()],
}

# Figure 2: where each team's snaps happened
share = {t: plays[plays.posteam == t].state.value_counts(normalize=True).reindex([s[2] for s in STATES]) for t in FOCUS}
fig_states = {
    "type": "bar", "orient": "v", "height": 300,
    "categories": [s[2] for s in STATES],
    "y": {"label": "Share of offensive plays", "format": "pct0"},
    "series": [{"name": t, "color": c, "values": [r(v) for v in share[t]]} for t, c in FOCUS.items()],
}

# Figure 3: run rate by score state
def by_state(d):
    return d.groupby("state", observed=False).rush.mean().reindex([s[2] for s in STATES])

fig_run_by_state = {
    "type": "bar", "orient": "v", "height": 300,
    "categories": [s[2] for s in STATES],
    "y": {"label": "Run rate", "format": "pct0", "min": 0},
    "series": [{"name": t, "color": c, "values": [r(v) for v in by_state(plays[plays.posteam == t])]}
               for t, c in FOCUS.items()]
              + [{"name": "League", "color": "gray", "values": [r(v) for v in by_state(plays)]}],
}

# Figure 4: every team-game, run rate vs. final margin
games = load_games()
games = games[(games.season == 2023) & (games.game_type == "REG")]
margins = {}
for g in games.itertuples():
    margins[(g.game_id, g.home_team)] = g.result
    margins[(g.game_id, g.away_team)] = -g.result
opp = {(g.game_id, g.home_team): g.away_team for g in games.itertuples()}
opp.update({(g.game_id, g.away_team): g.home_team for g in games.itertuples()})
tg = plays.groupby(["game_id", "posteam"]).agg(run_rate=("rush", "mean"), week=("week", "first")).reset_index()
tg["margin"] = [int(margins[(a, b)]) for a, b in zip(tg.game_id, tg.posteam)]
tg["opp"] = [opp[(a, b)] for a, b in zip(tg.game_id, tg.posteam)]
slope, intercept = np.polyfit(tg.margin, tg.run_rate, 1)
corr = tg[["margin", "run_rate"]].corr().iloc[0, 1]
print(f"team-game margin vs run rate: r={corr:.2f}, slope={slope * 100:.2f} pts of run rate per point of margin")


def game_point(row):
    res = "W" if row.margin > 0 else ("L" if row.margin < 0 else "T")
    return {"x": int(row.margin), "y": r(row.run_rate),
            "tip": [["Game", f"{row.posteam} vs {row.opp}, week {row.week}"],
                    ["Result", f"{res} {row.margin:+d}"],
                    ["Run rate", f"{row.run_rate:.1%}"]]}


fig_games = {
    "type": "scatter", "height": 380,
    "x": {"label": "Final scoring margin", "format": "signed0"},
    "y": {"label": "Run rate in that game", "format": "pct0"},
    "trend": {"slope": r(slope, 5), "intercept": r(intercept, 4), "label": "All teams trend"},
    "rules": [{"axis": "x", "value": 0}],
    "groups": [{"name": "All other team-games", "color": "gray", "r": 3,
                "points": [game_point(x) for x in tg[~tg.posteam.isin(FOCUS)].itertuples()]}]
              + [{"name": t, "color": c, "points": [game_point(x) for x in tg[tg.posteam == t].itertuples()]}
                 for t, c in FOCUS.items()],
}

# Figure 5: who carried the ball
rushers = (runs[runs.posteam.isin(FOCUS)]
           .groupby(["posteam", "rusher_player_name"])
           .agg(carries=("epa", "size"), epa=("epa", "mean"), ypc=("yards_gained", "mean"))
           .reset_index())
rushers = rushers[rushers.carries >= 30].sort_values("epa", ascending=False)
fig_rushers = {
    "type": "bar", "orient": "h",
    "categories": [f"{x.rusher_player_name} ({x.posteam})" for x in rushers.itertuples()],
    "y": {"label": "EPA per carry", "format": "signed2"},
    "series": [{"name": "EPA per carry", "values": [r(v) for v in rushers.epa],
                "colors": [FOCUS[t] for t in rushers.posteam],
                "tips": [[["Team", x.posteam], ["Carries", str(x.carries)], ["Yards/carry", f"{x.ypc:.1f}"]]
                         for x in rushers.itertuples()]}],
    "legend": [{"name": t, "color": c} for t, c in FOCUS.items()],
}
print(rushers.round(3).to_string())


# --- Follow-up: pass rate over expected (PROE), 2020-2025
# nflfastR's xpass is the probability of a dropback given the situation (down, distance,
# field position, score, clock, win probability). PROE = actual pass rate - mean(xpass).
cols = ["season", "season_type", "game_id", "posteam", "pass", "rush", "xpass", "epa"]
hist = load_pbp(range(2020, 2026), columns=cols)
hist = hist[(hist.season_type == "REG") & ((hist["pass"] == 1) | (hist.rush == 1))].dropna(subset=["xpass", "posteam", "epa"])
proe = hist.groupby(["season", "posteam"]).agg(pass_rate=("pass", "mean"), xpass=("xpass", "mean"))
proe["proe"] = proe.pass_rate - proe.xpass
p23 = proe.loc[2023].sort_values("proe")
for t in FOCUS:
    print(t, p23.loc[t].round(3).to_dict(), "rank (most pass-heavy=1):", int(p23.proe.rank(ascending=False)[t]))

wide = proe.proe.unstack("season")
wide_pr = proe.pass_rate.unstack("season")
stab = {"proe": [wide[y].corr(wide[y + 1]) for y in range(2020, 2025)],
        "pass_rate": [wide_pr[y].corr(wide_pr[y + 1]) for y in range(2020, 2025)]}
print("year-to-year r:", {k: [round(v, 2) for v in vs] for k, vs in stab.items()})

fig_proe = {
    "type": "bar", "orient": "h", "compact": True,
    "categories": list(p23.index[::-1]),
    "y": {"label": "Pass rate over expected, 2023", "format": "signed1pct", "min": -0.1, "max": 0.08},
    "series": [{"name": "PROE", "values": [r(v, 4) for v in p23.proe[::-1]],
                "colors": [FOCUS.get(t, "gray") for t in p23.index[::-1]],
                "tips": [[["Actual pass rate", f"{row.pass_rate:.1%}"], ["Expected (xpass)", f"{row.xpass:.1%}"]]
                         for row in p23[::-1].itertuples()]}],
    "legend": [{"name": t, "color": c} for t, c in FOCUS.items()] + [{"name": "Other teams", "color": "gray"}],
}
fig_proe_hist = {
    "type": "line", "height": 260, "markers": True,
    "x": {"label": "Season", "format": "year", "min": 2020, "max": 2025},
    "y": {"label": "Pass rate over expected", "format": "signed1pct"},
    "rules": [{"axis": "y", "value": 0}],
    "series": [{"name": t, "color": c, "points": [[int(y), r(wide.loc[t, y], 4)] for y in range(2020, 2026)]}
               for t, c in FOCUS.items()],
}
print(wide.loc[list(FOCUS)].round(3))

# How much of the raw gap is situation? Expected pass rates differ by this much:
gap_actual = proe.loc[(2023, "MIN"), "pass_rate"] - proe.loc[(2023, "SF"), "pass_rate"]
gap_expected = proe.loc[(2023, "MIN"), "xpass"] - proe.loc[(2023, "SF"), "xpass"]
print(f"pass-rate gap MIN-SF actual {gap_actual:.3f}, expected from situation {gap_expected:.3f}")

# Is the 2-point neutral run-rate gap real? Bootstrap by resampling games (plays within a game are correlated).
rng = np.random.default_rng(7)
nt = {t: neutral[neutral.posteam == t].groupby("game_id").rush.agg(["sum", "count"]) for t in FOCUS}
diffs = []
for _ in range(10_000):
    rates = {}
    for t, g in nt.items():
        idx = rng.integers(0, len(g), len(g))
        rates[t] = g["sum"].to_numpy()[idx].sum() / g["count"].to_numpy()[idx].sum()
    diffs.append(rates["SF"] - rates["MIN"])
ci = np.quantile(diffs, [0.025, 0.975])
print(f"neutral run-rate gap SF-MIN {team.loc['SF', 'neutral_run_rate'] - team.loc['MIN', 'neutral_run_rate']:.3f}, 95% CI {ci.round(3)}")

tiles = {t: {k: r(team.loc[t, k]) for k in team.columns} for t in FOCUS}
write("percentageRunOffense", {
    "charts": {"runVsEpa": fig_scatter, "gameStates": fig_states, "runByState": fig_run_by_state,
               "gameMargin": fig_games, "rushers": fig_rushers,
               "proe": fig_proe, "proeHistory": fig_proe_hist},
    "stats": {"teams": tiles, "league": {k: r(v) for k, v in league.items()},
              "marginCorr": r(corr), "marginSlope": r(slope * 100),
              "proe": {t: {k: r(v, 4) for k, v in p23.loc[t].items()} for t in FOCUS},
              "proeStability": {k: r(np.mean(v)) for k, v in stab.items()},
              "gapActual": r(gap_actual, 4), "gapExpected": r(gap_expected, 4), "neutralGapCi": [r(c, 4) for c in ci]},
})
