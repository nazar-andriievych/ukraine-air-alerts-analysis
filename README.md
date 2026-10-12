# Ukraine air alerts analysis

Time series analysis of air raid alerts in Ukraine. The goal is to forecast **daily air raid alert hours per raion for the next 7 days**.

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
```

The download is skipped when the file is already there at the pinned commit. Silver and gold take a few seconds and are always rebuilt.

| Step | Output | What it does |
|---|---|---|
| `download` (bronze) | `data/bronze/official_data_en.csv` | Downloads the raw dataset at the pinned commit, unchanged, and records a checksum. |
| `silver` | `data/silver/alerts.parquet` | Cleans the alerts. One row per source alert, with oblast/raion/hromada uids. |
| `gold` | `data/gold/raion_daily.parquet` | Alert hours per raion per UTC day: the forecasting target. |

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

Kyiv City has no raions, so it is treated as a single unit. Overlapping alerts of a raion are merged so no time is counted twice, then split at UTC midnight. Gold has every raion for every day from 2022-03-16 (the first complete day) to 2026-09-06, with 0 on days without alerts.

`data/bronze/`, `data/silver/` and `data/gold/` are not committed; run the pipeline to rebuild them.

## Tests
```
python -m pytest
```

To regenerate the English regions file: `python scripts/translate_regions.py`.

## Layout
```
data/reference/   regions.json, regions_uk.json   (committed)
data/bronze/      raw downloaded data              (generated)
data/silver/      cleaned alerts                   (generated)
data/gold/        daily alert hours per raion      (generated)
docs/             task and problem statement
pipeline/         pipeline steps and config
scripts/          one-off helper scripts
tests/            pytest tests for silver and gold
```
