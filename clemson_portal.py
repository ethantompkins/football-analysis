"""Did avoiding the transfer portal cost Clemson? Team-level portal, talent and ratings, 2015-2026."""
import glob
import urllib.request

import statsmodels.formula.api as smf

from common import DATA_DIR, cfb_file, np, pd, r, write

CLEMSON = 228  # ESPN team id
CFB_DIR = DATA_DIR / "cfb"
SDV = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
TABLES = ["cfb_team_portal", "cfb_team_talent", "cfb_returning_production", "cfb_ratings", "cfb_schedules"]
YEARS = range(2015, 2027)


def load(table):
    CFB_DIR.mkdir(parents=True, exist_ok=True)
    for y in YEARS:
        path = CFB_DIR / f"{table}_{y}.parquet"
        if not path.exists():
            print(f"downloading {table}_{y}")
            urllib.request.urlretrieve(f"{SDV}/{table}/{table}_{y}.parquet", path)
    d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(CFB_DIR / f"{table}_20*.parquet")))],
                  ignore_index=True)
    if "team_id" in d.columns:  # some tables store the ESPN id as text
        d["team_id"] = pd.to_numeric(d.team_id, errors="coerce").astype("Int64")
    return d


portal, talent, returning, ratings, sched = (load(t) for t in TABLES)

# The 2026 season is in progress: use the weekly ratings snapshot and only games played by then.
AS_OF_WEEK, AS_OF_DATE = 4, "2026-09-30"
weekly = pd.read_parquet(cfb_file("cfb_ratings_weekly", 2026))
weekly["team_id"] = pd.to_numeric(weekly.team_id, errors="coerce").astype("Int64")
snap = weekly[weekly.through_week == AS_OF_WEEK].drop(columns="through_week")
ratings = pd.concat([ratings[ratings.season != 2026], snap[ratings.columns]], ignore_index=True)
sched = sched[(sched.season != 2026) | (sched.start_date < AS_OF_DATE)]

panel = (ratings
         .merge(portal, on=["season", "team_id"])
         .merge(talent[["season", "team_id", "team", "talent_composite", "talent_rank"]], on=["season", "team_id"])
         .merge(returning[["season", "team_id", "overall_returning"]], on=["season", "team_id"], how="left"))
z = lambda s: (s - s.mean()) / s.std()
panel["talent_z"] = panel.groupby("season").talent_composite.transform(z)
panel["out_rate"] = panel.transfers_out_n / panel.roster_n
prev = panel[["season", "team_id", "net_z"]].assign(season=panel.season + 1).rename(columns={"net_z": "prev_net_z"})
panel = panel.merge(prev, on=["season", "team_id"], how="left")
panel["tin100"] = panel.transfer_talent_in / 100
panel["tout100"] = panel.transfer_talent_out / 100
top25 = panel.talent_rank <= 25
clem = panel[panel.team_id == CLEMSON].set_index("season")

# Records, for context
games = sched.dropna(subset=["home_points", "away_points"])
games = games[(games.home_id == CLEMSON) | (games.away_id == CLEMSON)]
cpts = np.where(games.home_id == CLEMSON, games.home_points, games.away_points)
opts = np.where(games.home_id == CLEMSON, games.away_points, games.home_points)
record = pd.DataFrame({"season": games.season, "w": cpts > opts, "l": cpts < opts}).groupby("season").sum()
print(record.T)

# --- Figure 1: the dynasty and the slide, in standard-deviation units
fig_arc = {
    "type": "line", "height": 280, "markers": True,
    "x": {"label": "Season", "format": "year", "min": 2015, "max": 2026},
    "y": {"label": "Standard deviations above the FBS average", "format": "dec1"},
    "rules": [{"axis": "y", "value": 0}],
    "series": [{"name": "Roster talent (recruiting composite)", "color": 2,
                "points": [[int(y), r(clem.loc[y, "talent_z"], 3)] for y in YEARS]},
               {"name": "On-field rating (opponent-adjusted net EPA)", "color": 1,
                "points": [[int(y), r(clem.loc[y, "net_z"], 3)] for y in YEARS]}],
}
for y in YEARS:
    print(y, "talent rank", int(clem.loc[y, "talent_rank"]), "net rank", int(clem.loc[y, "net_rank"]),
          "talent_z", round(clem.loc[y, "talent_z"], 2), "net_z", round(clem.loc[y, "net_z"], 2))

# --- Figure 2: transfers in per season
PORTAL_YEARS = range(2019, 2027)
by_year = pd.DataFrame({
    "clemson": clem.transfers_in_n.reindex(PORTAL_YEARS),
    "top25": panel[top25].groupby("season").transfers_in_n.median().reindex(PORTAL_YEARS),
    "fbs": panel.groupby("season").transfers_in_n.median().reindex(PORTAL_YEARS),
})
rank_in = {y: int(panel[panel.season == y].transfers_in_n.rank(ascending=False, method="min")[panel[panel.season == y].team_id == CLEMSON].iat[0])
           for y in PORTAL_YEARS}
n_teams = panel.groupby("season").size().to_dict()
print(by_year, rank_in)
fig_in = {
    "type": "line", "height": 280, "markers": True,
    "x": {"label": "Season", "format": "year", "min": 2019, "max": 2026},
    "y": {"label": "Transfers added", "format": "int", "min": 0},
    "series": [{"name": "Clemson", "color": 2, "points": [[int(y), int(v)] for y, v in by_year.clemson.items()]},
               {"name": "Median top-25 talent roster", "color": 1, "points": [[int(y), r(v, 1)] for y, v in by_year.top25.items()]},
               {"name": "Median FBS team", "color": "gray", "points": [[int(y), r(v, 1)] for y, v in by_year.fbs.items()]}],
}

# --- Figure 3: retention, i.e. share of the roster leaving through the portal
ret = pd.DataFrame({
    "clemson": clem.out_rate.reindex(PORTAL_YEARS),
    "top25": panel[top25].groupby("season").out_rate.median().reindex(PORTAL_YEARS),
    "fbs": panel.groupby("season").out_rate.median().reindex(PORTAL_YEARS),
})
rank_out = {y: int(panel[panel.season == y].out_rate.rank(method="min")[panel[panel.season == y].team_id == CLEMSON].iat[0])
            for y in PORTAL_YEARS}
rank_out_peers = {}
for y in PORTAL_YEARS:
    peers = panel[(panel.season == y) & (top25 | (panel.team_id == CLEMSON))]
    rank_out_peers[y] = (int(peers.out_rate.rank(method="min")[peers.team_id == CLEMSON].iat[0]), len(peers))
print(ret.round(3), rank_out, "among top-25 talent:", rank_out_peers)
fig_out = {
    "type": "line", "height": 280, "markers": True,
    "x": {"label": "Season", "format": "year", "min": 2019, "max": 2026},
    "y": {"label": "Share of roster that transferred out", "format": "pct0", "min": 0},
    "series": [{"name": "Clemson", "color": 2, "points": [[int(y), r(v, 4)] for y, v in ret.clemson.items()]},
               {"name": "Median top-25 talent roster", "color": 1, "points": [[int(y), r(v, 4)] for y, v in ret.top25.items()]},
               {"name": "Median FBS team", "color": "gray", "points": [[int(y), r(v, 4)] for y, v in ret.fbs.items()]}],
}

# --- Figure 4: who underperformed their talent, 2021-2025 (the one-time-transfer era)
era = panel[panel.season.between(2021, 2025)].copy()
talent_fit = smf.ols("net_z ~ talent_z + C(season)", data=era).fit()
era["gap"] = talent_fit.resid
elite = (era[era.talent_rank <= 15].groupby("team")
         .agg(seasons=("gap", "size"), gap=("gap", "mean"), talent_rank=("talent_rank", "mean"),
              net_rank=("net_rank", "mean"), transfers_in=("transfers_in_n", "mean"))
         .query("seasons >= 3").sort_values("gap"))
print(elite.round(2))
info = pd.read_parquet(cfb_file("cfb_team_info", 2025))[["team_id", "school"]]
SCHOOL = dict(zip(panel.team, panel.team_id.map(dict(zip(pd.to_numeric(info.team_id), info.school)))))
SHORT = lambda t: SCHOOL.get(t) or t
fig_gap = {
    "type": "bar", "orient": "h",
    "categories": [SHORT(t) for t in elite.index],
    "y": {"label": "On-field rating vs. what roster talent predicts (SD per season)", "format": "signed1"},
    "series": [{"name": "Gap", "values": [r(v, 3) for v in elite.gap],
                "colors": [2 if t.startswith("Clemson") else "gray" for t in elite.index],
                "tips": [[["Avg talent rank", f"{x.talent_rank:.1f}"], ["Avg rating rank", f"{x.net_rank:.1f}"],
                          ["Transfers in / yr", f"{x.transfers_in:.1f}"], ["Seasons in top 15", str(x.seasons)]]
                         for x in elite.itertuples()]}],
}

# --- The model: what predicts a team's rating, once last year and talent are known?
train = panel[panel.season.between(2022, 2025)].dropna(subset=["prev_net_z", "overall_returning"])
FORMULA = "net_z ~ prev_net_z + talent_z + overall_returning + tin100 + tout100 + C(season)"
model = smf.ols(FORMULA, data=train).fit(cov_type="cluster", cov_kwds={"groups": train.team_id.astype(int)})
print(model.summary().tables[1], "R2", round(model.rsquared, 3), "n", len(train))
coef = {k: {"b": r(model.params[k], 4), "lo": r(model.conf_int().loc[k, 0], 4), "hi": r(model.conf_int().loc[k, 1], 4),
            "p": r(model.pvalues[k], 4)} for k in ["prev_net_z", "talent_z", "overall_returning", "tin100", "tout100"]}

# Robustness: count of transfers instead of talent points, and change-in-rating as the outcome
alt_n = smf.ols("net_z ~ prev_net_z + talent_z + overall_returning + transfers_in_n + transfers_out_n + C(season)",
                data=train).fit(cov_type="cluster", cov_kwds={"groups": train.team_id.astype(int)})
print("per transfer in:", round(alt_n.params["transfers_in_n"], 4), alt_n.conf_int().loc["transfers_in_n"].round(4).tolist())

# Clemson: actual vs. model, and a counterfactual with a typical top-25 portal class
clem_rows = panel[(panel.team_id == CLEMSON) & panel.season.between(2022, 2026)].copy()
clem_rows["pred"] = model.predict(clem_rows.assign(season=clem_rows.season.clip(upper=2025)))
typical = panel[top25].groupby("season").transfer_talent_in.median()
clem_rows["typical_tin100"] = clem_rows.season.map(typical) / 100
clem_rows["pred_typical"] = clem_rows.pred + model.params["tin100"] * (clem_rows.typical_tin100 - clem_rows.tin100)
sd_adj = panel.groupby("season").adj_net.std()


def rank_for(season, value_z):
    s = panel[panel.season == season].net_z
    return int((s > value_z).sum() + 1)


for x in clem_rows.itertuples():
    print(int(x.season), "actual", round(x.net_z, 2), "pred", round(x.pred, 2), "with typical portal", round(x.pred_typical, 2),
          "| tin", x.transfer_talent_in, "typical", typical[x.season],
          "| rank actual", int(x.net_rank), "rank if typical", rank_for(x.season, x.net_z + (x.pred_typical - x.pred)))
fig_cf = {
    "type": "bar", "orient": "v", "height": 300,
    "categories": [str(int(s)) + (f" ({AS_OF_WEEK} games)" if s == 2026 else "") for s in clem_rows.season],
    "y": {"label": "On-field rating (SD above FBS average)", "format": "dec1", "min": 0},
    "series": [{"name": "Actual", "color": 2, "values": [r(v, 3) for v in clem_rows.net_z]},
               {"name": "Model, with Clemson's real portal use", "color": "gray", "values": [r(v, 3) for v in clem_rows.pred]},
               {"name": "Model, with a typical top-25 portal class", "color": 1, "values": [r(v, 3) for v in clem_rows.pred_typical]}],
}

# How much of the 2023-25 shortfall could the portal explain?
window = clem_rows[clem_rows.season.between(2023, 2025)]
shortfall = (window.net_z - window.pred_typical).mean()
portal_part = (window.pred_typical - window.pred).mean()
resid_part = (window.net_z - window.pred).mean()
print(f"2023-25 mean: actual vs typical-portal model {shortfall:.2f} SD; portal accounts for {portal_part:.2f}, unexplained {resid_part:.2f}")

write("clemsonPortal", {
    "charts": {"arc": fig_arc, "transfersIn": fig_in, "retention": fig_out, "talentGap": fig_gap, "counterfactual": fig_cf},
    "stats": {
        "record": {int(k): [int(v.w), int(v.l)] for k, v in record.iterrows()},
        "clemson": {int(y): {k: r(clem.loc[y, k], 3) for k in ["talent_rank", "net_rank", "talent_z", "net_z", "transfers_in_n",
                                                                "transfers_out_n", "out_rate", "transfer_talent_in"]} for y in YEARS},
        "rankIn": rank_in, "rankOut": rank_out, "rankOutPeers": rank_out_peers, "nTeams": {int(k): int(v) for k, v in n_teams.items()},
        "medians": {"in": by_year.round(1).to_dict(), "out": ret.round(4).to_dict()},
        "elite": {t: {k: r(v, 3) for k, v in row.items()} for t, row in elite.iterrows()},
        "model": {"n": len(train), "r2": r(model.rsquared, 3), "coef": coef,
                  "perTransfer": r(alt_n.params["transfers_in_n"], 4),
                  "perTransferCi": [r(v, 4) for v in alt_n.conf_int().loc["transfers_in_n"]]},
        "counterfactual": [{k: r(getattr(x, k), 3) for k in ["season", "net_z", "pred", "pred_typical", "transfer_talent_in", "typical_tin100", "net_rank"]}
                           for x in clem_rows.itertuples()],
        "shortfall": r(shortfall, 3), "portalPart": r(portal_part, 3), "residPart": r(resid_part, 3),
        "sdAdjNet": r(sd_adj.mean(), 3),
    },
})
