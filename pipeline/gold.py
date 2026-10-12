"""Build the gold layer: daily alert hours per raion, the forecasting target.

Usage: python -m pipeline.gold

Input:  data/silver/alerts.parquet, data/reference/regions.json
Output: data/gold/raion_daily.parquet

Each alert is spread to the raions it covered:
  - oblast alert  -> every raion of the oblast (Kyiv City has no raions and is a single unit)
  - raion alert   -> that raion
  - hromada alert -> its whole raion
Overlapping alerts of a raion are merged, so no time is counted twice, and the merged time is
split at UTC midnight. The table has every raion for every day from DATA_START up to DATA_END,
with 0 on days without alerts.
"""
import json

import pandas as pd

from pipeline import silver
from pipeline.config import DATA_END, DATA_START, GOLD
from pipeline.regions import load_raions

OUTPUT = GOLD / "raion_daily.parquet"
COLUMNS = ["raion_uid", "raion", "oblast_uid", "oblast", "day", "alert_hours"]
DAY = pd.Timedelta(days=1)


def expand_to_raions(alerts: pd.DataFrame, raions: pd.DataFrame) -> pd.DataFrame:
    """One interval (raion_uid, started_at, finished_at) per raion an alert covered."""
    times = ["started_at", "finished_at"]
    whole_oblast = alerts["level"] == "oblast"
    by_oblast = (alerts.loc[whole_oblast, ["oblast_uid", *times]]
                 .merge(raions[["oblast_uid", "raion_uid"]], on="oblast_uid"))
    by_raion = alerts.loc[~whole_oblast, ["raion_uid", *times]]
    return pd.concat([by_oblast[["raion_uid", *times]], by_raion], ignore_index=True)


def merge_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    """Merge overlapping or touching intervals within each raion."""
    iv = intervals.sort_values(["raion_uid", "started_at"]).reset_index(drop=True)
    reach = iv.groupby("raion_uid")["finished_at"].cummax()
    prev_reach = reach.groupby(iv["raion_uid"]).shift()
    run = (prev_reach.isna() | (iv["started_at"] > prev_reach)).cumsum()
    return (iv.groupby(run)
            .agg(raion_uid=("raion_uid", "first"), started_at=("started_at", "min"),
                 finished_at=("finished_at", "max"))
            .reset_index(drop=True))


def split_by_day(intervals: pd.DataFrame) -> pd.DataFrame:
    """Alert hours per (raion_uid, day), splitting intervals at UTC midnight."""
    first_day = intervals["started_at"].dt.floor("D")
    n_days = (intervals["finished_at"].dt.ceil("D") - first_day) // DAY
    rows = intervals.loc[intervals.index.repeat(n_days)]
    day = first_day.loc[rows.index] + rows.groupby(level=0).cumcount() * DAY
    start = rows["started_at"].where(rows["started_at"] > day, day)
    end = rows["finished_at"].where(rows["finished_at"] < day + DAY, day + DAY)
    hours = (end - start).dt.total_seconds() / 3600
    return (pd.DataFrame({"raion_uid": rows["raion_uid"], "day": day, "alert_hours": hours})
            .groupby(["raion_uid", "day"], as_index=False)["alert_hours"].sum())


def daily_table(hours: pd.DataFrame, raions: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Every raion x every day in [start, end), with 0 hours where there was no alert."""
    days = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D", inclusive="left")
    grid = pd.MultiIndex.from_product([raions["raion_uid"], days], names=["raion_uid", "day"])
    table = (hours.set_index(["raion_uid", "day"])["alert_hours"]
             .reindex(grid, fill_value=0.0)
             .reset_index()
             .merge(raions[["raion_uid", "raion", "oblast_uid", "oblast"]], on="raion_uid"))
    return table[COLUMNS]


def build() -> None:
    if not silver.OUTPUT.exists():
        raise SystemExit(f"{silver.OUTPUT} not found; run `python -m pipeline silver` first")

    print(f"build {OUTPUT.name}")
    raions = load_raions()
    intervals = merge_intervals(expand_to_raions(pd.read_parquet(silver.OUTPUT), raions))
    table = daily_table(split_by_day(intervals), raions, DATA_START, DATA_END)

    GOLD.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUTPUT, index=False)
    print(json.dumps({
        "rows": len(table),
        "raions": int(table["raion_uid"].nunique()),
        "first_day": str(table["day"].min().date()),
        "last_day": str(table["day"].max().date()),
        "mean_alert_hours": round(float(table["alert_hours"].mean()), 2),
        "share_of_days_without_alerts": round(float((table["alert_hours"] == 0).mean()), 3),
        "max_alert_hours": round(float(table["alert_hours"].max()), 4),
    }, indent=2))


if __name__ == "__main__":
    build()
