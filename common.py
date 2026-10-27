"""Shared helpers for the blog analyses.

Data comes from the public nflverse releases. Files are downloaded once into
NFL_DATA_DIR (default: .cache) and reused on later runs.
"""
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("NFL_DATA_DIR", ROOT / ".cache"))
# Chart data for the blog; point BLOG_DATA_DIR at the site's assets/data/blog to update it directly.
OUT_DIR = Path(os.environ.get("BLOG_DATA_DIR", ROOT / "output"))

RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"
SOURCES = {
    "pbp_{y}.parquet": RELEASES + "/pbp/play_by_play_{y}.parquet",
    "part_{y}.parquet": RELEASES + "/pbp_participation/pbp_participation_{y}.parquet",
    "player_stats.parquet": RELEASES + "/player_stats/player_stats.parquet",
    "games.csv": "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
}


def _fetch(name, url):
    path = DATA_DIR / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def load_pbp(years, participation=False, columns=None):
    frames = []
    for y in years:
        pbp = pd.read_parquet(_fetch(f"pbp_{y}.parquet", SOURCES["pbp_{y}.parquet"].format(y=y)), columns=columns)
        if participation:
            part = pd.read_parquet(_fetch(f"part_{y}.parquet", SOURCES["part_{y}.parquet"].format(y=y)))
            part = part.rename(columns={"nflverse_game_id": "game_id"})
            part = part[["game_id", "play_id", "defenders_in_box", "number_of_pass_rushers",
                         "time_to_throw", "was_pressure", "offense_formation"]]
            pbp = pbp.merge(part, on=["game_id", "play_id"], how="left")
        frames.append(pbp)
    return pd.concat(frames, ignore_index=True)


def fetch_ftn(year):
    """FTN charting data (2022 on): play-action, catchable balls, QB-fault sacks, reads."""
    return _fetch(f"ftn_{year}.parquet", f"{RELEASES}/ftn_charting/ftn_charting_{year}.parquet")


def fetch_participation(year):
    return _fetch(f"part_{year}.parquet", SOURCES["part_{y}.parquet"].format(y=year))


SDV = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"


def cfb_file(table, year):
    """A sportsdataverse college table, e.g. cfb_file("cfb_ratings", 2024)."""
    return _fetch(f"cfb/{table}_{year}.parquet", f"{SDV}/{table}/{table}_{year}.parquet")


def cfb_pbp(year):
    """cfbfastR college play-by-play for one season."""
    return _fetch(f"cfb/cfb_pbp_{year}.parquet", f"{SDV}/cfbfastR_cfb_pbp/play_by_play_{year}.parquet")


def load_player_stats():
    return pd.read_parquet(_fetch("player_stats.parquet", SOURCES["player_stats.parquet"]))


def load_games():
    return pd.read_csv(_fetch("games.csv", SOURCES["games.csv"]))


def r(x, n=3):
    """Round for JSON output, passing through None/NaN as null."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), n)


def write(slug, payload):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{slug}.json"
    path.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1024:.1f} KB)")
