"""Score the baselines on the validation origins (default) or the test origins.

Usage: python -m pipeline.baselines [validation|test]

Input:  data/features/origins.parquet, data/gold/raion_hourly.parquet
Output: reports/baselines_<split>_by_horizon.csv  Brier score per horizon and baseline
        reports/baselines_<split>.csv             Brier and skill against the 28-day rate,
                                                  overall and per alert-rate band

Baselines for horizon h (the hour starting at origin + (h - 1) hours):
  persistence  1 if under alert at the origin, else 0, for every horizon
  rate_28d     share of the 672 hours (28 days) before the origin under alert  <- the one to beat
  hour_of_day  share of the same UTC hour under alert on each of the 28 days before the target hour
"""
import sys

import pandas as pd

from pipeline import evaluate, features, gold
from pipeline.config import REPORTS, SPLITS

BASELINES = ["persistence", "rate_28d", "hour_of_day"]
REFERENCE = "rate_28d"
DAYS = 28


def hour_of_day_rate(flags: pd.DataFrame, days: int = DAYS) -> pd.DataFrame:
    """For each hour u: the mean flag at u - 24 h, u - 48 h, ..., u - days * 24 h.

    For any horizon up to 24 h this uses only hours before the origin.
    """
    return sum(flags.shift(24 * k) for k in range(1, days + 1)) / days


def score_table(origins: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """One row per (origin, raion, horizon): the outcome and every baseline's probability."""
    by_hour = hour_of_day_rate(flags)
    parts = []
    for h in range(1, features.HORIZON + 1):
        target = origins["origin"] + (h - 1) * features.HOUR
        parts.append(pd.DataFrame({
            "origin": origins["origin"], "raion_uid": origins["raion_uid"], "horizon": h,
            "alert": features.at(flags, target, origins["raion_uid"]),
            "persistence": origins["alert_now"].astype(float),
            "rate_28d": origins["share_672h"].astype(float),
            "hour_of_day": features.at(by_hour, target, origins["raion_uid"]),
        }))
    return pd.concat(parts, ignore_index=True)


def run(split: str = "validation") -> None:
    if split not in ("validation", "test"):
        raise SystemExit(f"split must be validation or test, not {split!r}")
    if not features.OUTPUT.exists():
        raise SystemExit(f"{features.OUTPUT} not found; run `python -m pipeline features` first")

    print(f"score baselines on {split}")
    origins = pd.read_parquet(features.OUTPUT, filters=[("split", "==", split)])
    flags = features.flag_matrix(pd.read_parquet(gold.HOURLY_OUTPUT))
    scored = score_table(origins, flags)
    bands = evaluate.alert_rate_bands(flags, pd.Timestamp(SPLITS[split][0]))
    by_horizon = evaluate.brier_by_horizon(scored, BASELINES)
    summary = evaluate.skill_summary(scored, BASELINES, REFERENCE, bands)

    REPORTS.mkdir(parents=True, exist_ok=True)
    by_horizon.round(5).to_csv(REPORTS / f"baselines_{split}_by_horizon.csv")
    summary.round(5).to_csv(REPORTS / f"baselines_{split}.csv", index=False)
    print(f"{scored['origin'].nunique()} origins x {scored['raion_uid'].nunique()} raions x "
          f"{features.HORIZON} horizons = {len(scored)} forecasts")
    print("Brier at horizons 1, 6, 24:")
    print(by_horizon.loc[[1, 6, 24]].round(3).to_string())
    print(f"Skill against {REFERENCE} (95% interval):")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    run(*sys.argv[1:2])
