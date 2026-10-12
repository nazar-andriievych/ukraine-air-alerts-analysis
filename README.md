# Ukraine air alerts analysis

Time series analysis of air raid alerts in Ukraine. The goal is to forecast the **probability of an air raid alert per raion in each of the next 24 hours**.

- Task description: [`docs/TASK.md`](docs/TASK.md)
- Problem statement, decisions and data findings: [`docs/PROBLEM.md`](docs/PROBLEM.md)

## Data
- **Alerts:** the official alerts from the public [ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset), pinned to a fixed commit (see `pipeline/config.py`). It covers 2022-03-15 to 2026-09-07. We use data up to **2026-09-06**, the last complete day.
- **Regions:** `data/reference/regions.json` holds the oblast → raion → hromada hierarchy with IDs. It is generated from `regions_uk.json` (Ukrainian names) by `scripts/translate_regions.py`. Names follow the dataset, including old raion names.

All timestamps and day boundaries are in **UTC**.

## Setup
Requires Python 3.13.

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt   # Windows
.venv/bin/python -m pip install -r requirements.txt       # Linux / macOS
```

## Run the pipeline
```
python -m pipeline              # all steps
python -m pipeline gold         # one step
python -m pipeline --force      # also re-download the dataset
python -m pipeline.baselines test   # score the baselines on the test split (default: validation)
python -m pipeline.models --calendar utc   # the models with UTC instead of Kyiv calendar features
python -m pipeline weather                  # download weather (Open-Meteo) for the weather experiment
python -m pipeline models --weather oracle  # weather experiment: upper bound with actual (future) weather
```

The download is skipped when the file is already there at the pinned commit. Silver and gold take a few seconds and are always rebuilt.

| Step | Output | What it does |
|---|---|---|
| `download` (bronze) | `data/bronze/official_data_en.csv` | Downloads the raw dataset at the pinned commit, unchanged, and records a checksum. |
| `silver` | `data/silver/alerts.parquet` | Cleans the alerts. One row per source alert, with oblast/raion/hromada uids. |
| `gold` | `data/gold/raion_daily.parquet` | Alert hours per raion per UTC day. |
|  | `data/gold/raion_hourly.parquet` | 1 if the raion was under alert at any moment in the UTC hour, else 0: the forecasting target. |
|  | `data/gold/raion_intervals.parquet` | Merged alert intervals per raion (no overlaps). |
| `features` | `data/features/origins.parquet` | One row per raion per forecast origin (every UTC hour) with features computed only from data before it, and its split (train / validation / test). |
| `baselines` | `reports/baselines_validation*.csv` | Brier score of the baselines per horizon, and skill against the 28-day rate overall and per alert-rate band. |
| `models` | `reports/models_validation*.csv`, `reports/*.png` | Trains logistic regression (one per horizon) and LightGBM on train origins (about 2 minutes) and scores them on validation with the baselines; writes the charts. |
| `weather` (not in `all`) | `data/bronze/weather/*.json`, `data/silver/weather_actual.parquet`, `data/silver/weather_forecast.parquet` | Downloads hourly weather per oblast centre from Open-Meteo (ERA5 actual weather and day-ahead `_previous_day2` forecasts; points from `data/reference/oblast_centers.json`) and builds one row per hour and oblast. Used only by the weather experiment. |
| `models --weather oracle` / `forecast` | `reports/models_validation_weather_*.csv` | Weather experiment: logistic regression with and without weather at the target hour, on validation. `oracle` uses the actual weather (leaky, an upper bound). Weather was **not adopted** (see `docs/PROBLEM.md`, "Weather experiment"). |
| `final` | `reports/final_test*.csv`, `reports/final_test_*.png` | Retrains the chosen model (logistic regression) on train + validation and scores it and the baselines on the test split, once. |

Silver cleaning:
- removes exact duplicates (38% of rows)
- drops Luhansk and Crimea, where alerts are permanent
- removes zero-length alerts and alerts starting after the data end
- cuts alerts still running at the data end off there
- fixes rows that have a raion name in the hromada field

Silver keeps the source `level`, `raion` and `hromada` of each alert, so finer features can be built from it later.

Gold spreads each alert across the raions it covered:
- oblast alert → every raion of that oblast
- raion alert → that raion
- hromada alert → its whole raion

Kyiv City has no raions, so it is treated as a single unit. Overlapping alerts of a raion are merged so no time is counted twice, then split at UTC midnight (daily table) or at each UTC hour (hourly table). Gold has every raion for every day and every hour from 2022-03-16 (the first complete day) to 2026-09-06, with 0 where there was no alert.

`data/bronze/`, `data/silver/` and `data/gold/` are not committed; run the pipeline to rebuild them.

## Tests
```
python -m pytest
```

To regenerate the English regions file: `python scripts/translate_regions.py`.

## Layout
```
data/reference/   regions.json, regions_uk.json, oblast_centers.json   (committed)
data/bronze/      raw downloaded data              (generated)
data/silver/      cleaned alerts                   (generated)
data/gold/        daily and hourly alerts per raion (generated)
data/features/    features per raion and origin    (generated)
docs/             task and problem statement
pipeline/         pipeline steps and config
reports/          evaluation results
scripts/          one-off helper scripts
tests/            pytest tests for every step
```
