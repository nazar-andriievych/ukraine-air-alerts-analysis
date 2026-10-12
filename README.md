# Ukraine air alerts analysis

Time series analysis of air raid alerts in Ukraine. The goal is to forecast **daily air raid alert hours per oblast for the next 7 days**.

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
python -m pipeline download     # one step
python -m pipeline --force      # rebuild even if inputs are unchanged
```

Steps skip themselves when their inputs haven't changed.

| Step | Output | What it does |
|---|---|---|
| `download` (bronze) | `data/bronze/official_data_en.csv` | Downloads the raw dataset at the pinned commit, unchanged, and records a checksum. |
| `silver` | `data/silver/raion_alerts.parquet` | Cleans the alerts and turns them into one row per raion per alert. |

Silver cleaning:
- removes exact duplicates (38% of rows)
- drops Luhansk and Crimea, where alerts are permanent
- removes zero-length alerts and alerts after the data end
- fixes rows that have a raion name in the hromada field

Silver then spreads each alert across the raions it covered:
- oblast alert → every raion of that oblast
- raion alert → that raion
- hromada alert → its whole raion

Kyiv City has no raions, so it is treated as a single unit. The original `level` and `hromada` columns are kept.

`data/bronze/` and `data/silver/` are not committed; run the pipeline to rebuild them.

To regenerate the English regions file: `python scripts/translate_regions.py`.

## Layout
```
data/reference/   regions.json, regions_uk.json   (committed)
data/bronze/      raw downloaded data              (generated)
data/silver/      cleaned raion-level alerts       (generated)
docs/             task and problem statement
pipeline/         pipeline steps and config
scripts/          one-off helper scripts
```
