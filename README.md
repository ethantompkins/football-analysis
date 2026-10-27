# football-analysis

The analysis code behind the posts on [ethantompkins.com](https://ethantompkins.com/blogs.html). Each script downloads the public
data it needs, runs the analysis, and writes the numbers and chart data for one post.

| Script | Post |
|---|---|
| [`bryce_young.py`](bryce_young.py) | [Measuring Bryce Young's Turnaround](https://ethantompkins.com/blogs/bryceYoung.html) |
| [`clemson_portal.py`](clemson_portal.py) | [Clemson and the Transfer Portal](https://ethantompkins.com/blogs/clemsonPortal.html) |
| [`ridge_adjustment.py`](ridge_adjustment.py) | [Ridge Regression to Adjust Opponents' Statistics](https://ethantompkins.com/blogs/ridgeAdjustment.html) |
| [`percentage_run_offense.py`](percentage_run_offense.py) | [Run Rate and Game Script: 49ers vs. Vikings, 2023](https://ethantompkins.com/blogs/percentageRunOffense.html) |
| [`wins_with_epa.py`](wins_with_epa.py) | [EPA Metrics and Team Wins](https://ethantompkins.com/blogs/winsWithEpa.html) |
| [`receiving_positions.py`](receiving_positions.py) | [Receiving Usage and Efficiency by Position](https://ethantompkins.com/blogs/receivingPositions.html) |
| [`play_prediction.py`](play_prediction.py) | [Run/Pass Prediction with a Random Forest](https://ethantompkins.com/blogs/playPredictionRFC.html) |
| [`prop_bootstrap.py`](prop_bootstrap.py) | [Prop Bet Pricing with Bootstrap Resampling](https://ethantompkins.com/blogs/propPredictionBootstrap.html) |
| [`sack_prediction.py`](sack_prediction.py) | [Sack Prediction with Logistic Regression](https://ethantompkins.com/blogs/sackPredictionLR.html) |

Shared helpers for downloading and caching data live in [`common.py`](common.py).

## Running

Requires Python 3.11+.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python ridge_adjustment.py
```

The first run of each script downloads several hundred MB of play-by-play data into `.cache/`; later runs reuse it.
Results are written to `output/<post>.json`, the chart and summary data the post's page reads.

| Variable | Default | Purpose |
|---|---|---|
| `NFL_DATA_DIR` | `.cache` | Where downloaded data is stored |
| `BLOG_DATA_DIR` | `output` | Where each script writes its JSON |

## Data

- NFL play-by-play, participation, FTN charting, player stats, and schedules from [nflverse](https://github.com/nflverse/nflverse-data).
- College play-by-play, schedules, ratings, recruiting, talent, returning production, and transfer portal tables from the
  [sportsdataverse](https://github.com/sportsdataverse/sportsdataverse-data) releases behind [cfbfastR](https://cfbfastr.sportsdataverse.org/).

Results for in-progress seasons will change as new weeks are published.
