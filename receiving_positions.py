"""How WRs, TEs and RBs are used in the passing game, 2020-2023."""
from common import load_pbp, load_player_stats, np, pd, r, write

YEARS = [2020, 2021, 2022, 2023]
FOCUS_YEAR = 2023
POS = {"WR": 1, "TE": 2, "RB": 3}

pbp = load_pbp(YEARS)
targets = pbp[(pbp.season_type == "REG") & (pbp.pass_attempt == 1) & (pbp.sack == 0)
              & pbp.receiver_player_id.notna()].copy()

stats = load_player_stats()
stats = stats[stats.season.isin(YEARS)]
pos = (stats.groupby(["season", "player_id"]).position
       .agg(lambda s: s.mode().iat[0]).replace({"FB": "RB"}).rename("pos"))
targets = targets.join(pos, on=["season", "receiver_player_id"])
targets = targets[targets.pos.isin(POS)]
targets["yac"] = targets.yards_after_catch.where(targets.complete_pass == 1)
print(targets.groupby("season").pos.value_counts(normalize=True).unstack().round(3))

t23 = targets[targets.season == FOCUS_YEAR]
g = t23.groupby("pos")
summary = pd.DataFrame({
    "targets": g.size(),
    "receptions": g.complete_pass.sum(),
    "yards": g.yards_gained.sum(),
    "tds": g.pass_touchdown.sum(),
    "catch_rate": g.complete_pass.mean(),
    "adot": g.air_yards.mean(),
    "yac": g.yac.mean(),
    "ypt": g.yards_gained.mean(),
    "epa_per_target": g.epa.mean(),
    "success": g.success.mean(),
    "first_down_rate": g.first_down_pass.mean(),
}).reindex(list(POS))
for c in ["targets", "receptions", "yards", "tds"]:
    summary[c + "_share"] = summary[c] / summary[c].sum()
print(summary.round(3).T)

# Figure 1: share of the passing game by position
SHARE = [("targets_share", "Targets"), ("receptions_share", "Receptions"),
         ("yards_share", "Receiving yards"), ("tds_share", "Receiving TDs")]
fig_share = {
    "type": "bar", "orient": "v", "height": 300,
    "categories": [s[1] for s in SHARE],
    "y": {"label": f"Share of {FOCUS_YEAR} total", "format": "pct0", "min": 0},
    "series": [{"name": p, "color": c, "values": [r(summary.loc[p, k]) for k, _ in SHARE]} for p, c in POS.items()],
}

# Figure 2: where targets are thrown (depth distribution)
BINS = list(range(-10, 41, 2))
depth = {}
for p in POS:
    a = t23[t23.pos == p].air_yards.dropna().clip(-10, 39.9)
    counts, _ = np.histogram(a, bins=BINS)
    depth[p] = counts / counts.sum()
fig_depth = {
    "type": "line", "height": 300,
    "x": {"label": "Air yards at target (depth of throw)", "format": "int", "min": -10, "max": 40},
    "y": {"label": "Share of that position's targets", "format": "pct0", "min": 0},
    "rules": [{"axis": "x", "value": 0, "label": "Line of scrimmage"}],
    "series": [{"name": p, "color": c,
                "points": [[(BINS[i] + BINS[i + 1]) / 2, r(v, 4)] for i, v in enumerate(depth[p])]}
               for p, c in POS.items()],
}
behind = {p: r((t23[t23.pos == p].air_yards <= 0).mean()) for p in POS}
deep = {p: r((t23[t23.pos == p].air_yards >= 20).mean()) for p in POS}
print("behind LOS", behind, "20+ air yards", deep)

# Figure 3: how each position gains its yards (per reception)
comp = t23[t23.complete_pass == 1]
per_rec = comp.groupby("pos").agg(air=("air_yards", "mean"), yac=("yac", "mean")).reindex(list(POS))
fig_yards = {
    "type": "bar", "orient": "h", "stacked": True, "height": 200,
    "categories": list(POS),
    "y": {"label": "Yards per reception", "format": "dec1"},
    "series": [{"name": "Air yards (caught)", "color": 1, "values": [r(v, 2) for v in per_rec.air]},
               {"name": "Yards after catch", "color": 2, "values": [r(v, 2) for v in per_rec.yac]}],
}
print(per_rec.round(2))

# Figure 4: every player with 50+ targets
pl = (t23.groupby(["receiver_player_id", "receiver_player_name", "pos", "posteam"])
      .agg(tgt=("epa", "size"), adot=("air_yards", "mean"), yac=("yac", "mean"), epa=("epa", "mean"),
           rec=("complete_pass", "sum"), yds=("yards_gained", "sum"))
      .reset_index())
pl = pl[pl.tgt >= 50]
LABELS = ["C.McCaffrey", "D.Samuel", "R.Rice", "T.Kelce", "G.Kittle", "K.Pitts", "T.Hill", "C.Lamb",
          "M.Evans", "A.Ekeler", "D.London", "Ju.Watson", "S.Barkley"]
fig_players = {
    "type": "scatter", "height": 420,
    "x": {"label": "Average depth of target (aDOT, yards)", "format": "dec0"},
    "y": {"label": "Yards after catch per reception", "format": "dec0", "min": 0},
    "groups": [{"name": p, "color": c, "labels": LABELS,
                "points": [{"x": r(x.adot, 2), "y": r(x.yac, 2), "label": x.receiver_player_name,
                            "tip": [["Player", f"{x.receiver_player_name} ({x.posteam}, {p})"],
                                    ["Targets", str(x.tgt)], ["aDOT", f"{x.adot:.1f}"],
                                    ["YAC/rec", f"{x.yac:.1f}"], ["EPA/target", f"{x.epa:+.2f}"]]}
                           for x in pl[pl.pos == p].itertuples()]}
               for p, c in POS.items()],
}

# Figure 5: efficiency per target at the same depth
DEPTHS = [(-99, 0, "Behind LOS"), (0, 10, "0–9 yards"), (10, 20, "10–19 yards"), (20, 99, "20+ yards")]
t23 = t23.assign(depth=pd.cut(t23.air_yards, [d[0] for d in DEPTHS] + [99], right=False,
                              labels=[d[2] for d in DEPTHS]))
eff = t23.groupby(["pos", "depth"], observed=False).agg(epa=("epa", "mean"), n=("epa", "size")).reset_index()
fig_eff = {
    "type": "bar", "orient": "v", "height": 300,
    "categories": [d[2] for d in DEPTHS],
    "y": {"label": "EPA per target", "format": "signed2"},
    "series": [{"name": p, "color": c,
                "values": [r(eff[(eff.pos == p) & (eff.depth == d[2])].epa.iat[0]) for d in DEPTHS],
                "tips": [[["Targets", str(int(eff[(eff.pos == p) & (eff.depth == d[2])].n.iat[0]))]] for d in DEPTHS]}
               for p, c in POS.items()],
}
print(eff.pivot(index="pos", columns="depth", values="epa").round(3))
print(eff.pivot(index="pos", columns="depth", values="n"))

# Figure 6: target share trend
trend = targets.groupby("season").pos.value_counts(normalize=True).unstack()


# --- Follow-up: yards after catch over expected (YACOE)
# nflfastR's xyac model estimates the YAC an average receiver would get on each completion,
# given where the catch happened (depth, field position, down and distance, etc.).
comp23 = t23[(t23.complete_pass == 1)].dropna(subset=["yac", "xyac_mean_yardage"]).copy()
comp23["yacoe"] = comp23.yac - comp23.xyac_mean_yardage
pos_yacoe = comp23.groupby("pos").agg(yac=("yac", "mean"), xyac=("xyac_mean_yardage", "mean"),
                                      yacoe=("yacoe", "mean")).reindex(list(POS))
print(pos_yacoe.round(2))

pl_y = (comp23.groupby(["receiver_player_id", "receiver_player_name", "pos", "posteam"])
        .agg(rec=("yacoe", "size"), yac=("yac", "mean"), xyac=("xyac_mean_yardage", "mean"), yacoe=("yacoe", "mean"))
        .reset_index())
pl_y = pl_y[pl_y.rec >= 50].sort_values("yacoe", ascending=False)
show = pd.concat([pl_y.head(10), pl_y.tail(5)])
print(show.round(2).to_string())
fig_yacoe = {
    "type": "bar", "orient": "h",
    "categories": [f"{x.receiver_player_name} ({x.posteam})" for x in show.itertuples()],
    "y": {"label": "YAC over expected per reception, 2023", "format": "signed1"},
    "series": [{"name": "YACOE per reception", "values": [r(v, 2) for v in show.yacoe],
                "colors": [POS[p] for p in show.pos],
                "tips": [[["Position", x.pos], ["Receptions", str(x.rec)], ["YAC per rec", f"{x.yac:.1f}"],
                          ["Expected YAC", f"{x.xyac:.1f}"]] for x in show.itertuples()]}],
    "legend": [{"name": p, "color": c} for p, c in POS.items()],
}

# Is it a skill? Correlate each player's season with his next season (2020->21 ... 2024->25).
cols = ["season", "season_type", "pass_attempt", "sack", "complete_pass", "receiver_player_id",
        "air_yards", "yards_after_catch", "xyac_mean_yardage"]
allp = load_pbp(range(2020, 2026), columns=cols)
allp = allp[(allp.season_type == "REG") & (allp.pass_attempt == 1) & (allp.sack == 0) & allp.receiver_player_id.notna()]
cmp_ = allp[allp.complete_pass == 1].dropna(subset=["yards_after_catch", "xyac_mean_yardage"])
per = pd.DataFrame({
    "targets": allp.groupby(["receiver_player_id", "season"]).size(),
    "adot": allp.groupby(["receiver_player_id", "season"]).air_yards.mean(),
    "catch_rate": allp.groupby(["receiver_player_id", "season"]).complete_pass.mean(),
    "rec": cmp_.groupby(["receiver_player_id", "season"]).size(),
    "yac": cmp_.groupby(["receiver_player_id", "season"]).yards_after_catch.mean(),
    "xyac": cmp_.groupby(["receiver_player_id", "season"]).xyac_mean_yardage.mean(),
})
per["yacoe"] = per.yac - per.xyac
per = per[per.rec >= 40].reset_index()
nxt = per.copy()
nxt["season"] -= 1
pairs = per.merge(nxt, on=["receiver_player_id", "season"], suffixes=("", "_next"))
STAB = [("adot", "Depth of target (aDOT)"), ("xyac", "Expected YAC (role)"), ("yac", "YAC per reception"),
        ("catch_rate", "Catch rate"), ("yacoe", "YAC over expected")]
stab = {k: pairs[k].corr(pairs[k + "_next"]) for k, _ in STAB}
print("player year-to-year r (n=%d):" % len(pairs), {k: round(v, 2) for k, v in stab.items()})
fig_stab = {
    "type": "bar", "orient": "h", "height": 230,
    "categories": [lab for _, lab in STAB],
    "y": {"label": "Correlation with the same player's next season", "format": "dec2", "min": 0, "max": 1},
    "series": [{"name": "Year-to-year r", "color": 1, "values": [r(stab[k]) for k, _ in STAB]}],
}

write("receivingPositions", {
    "charts": {"share": fig_share, "depth": fig_depth, "yards": fig_yards, "players": fig_players, "eff": fig_eff,
               "yacoe": fig_yacoe, "yacStability": fig_stab,
               },
    "stats": {"summary": {p: {k: r(v) for k, v in summary.loc[p].items()} for p in POS},
              "behindLos": behind, "deep": deep,
              "posYacoe": {p: {k: r(v, 2) for k, v in pos_yacoe.loc[p].items()} for p in POS},
              "stability": {k: r(v) for k, v in stab.items()}, "stabilityN": len(pairs),
              "perRec": {p: {"air": r(per_rec.loc[p, "air"], 1), "yac": r(per_rec.loc[p, "yac"], 1)} for p in POS},
              "trend": {p: {int(y): r(trend.loc[y, p]) for y in YEARS} for p in POS},
              },
})
