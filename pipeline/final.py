"""Final test run: retrain the chosen model on train + validation and score the test split once.

Usage: python -m pipeline.final

Input:  data/features/origins.parquet, data/gold/raion_hourly.parquet
Output: reports/final_test.csv             Brier and skill against the 28-day rate, overall and per
                                           alert-rate band, with 95% day-block bootstrap intervals
        reports/final_test_by_horizon.csv  Brier per horizon and forecast
        reports/final_test_*.png           skill by horizon, calibration, example raions

The model was chosen on validation before this run (docs/PROBLEM.md): logistic regression, one per
horizon, Kyiv-time calendar features, no weather. It is trained the same way as in pipeline/models.py
(a random third of origins, one horizon from each horizon group), but on train and validation
origins together. Only the chosen model and the baselines are scored: scoring other candidates on
test would invite choosing among them by the test result.
"""
import numpy as np
import pandas as pd

from pipeline import baselines, evaluate, features, gold, models, plots
from pipeline.config import REPORTS, SPLITS
from pipeline.regions import load_raions

TZ = models.TIMEZONES["kyiv"]
FORECASTS = [*baselines.BASELINES, "logreg"]


def run(fit_splits: tuple[str, ...] = ("train", "validation"), split: str = "test") -> None:
    """Fit on `fit_splits`, score `split`. Other arguments are only for checking the code on validation."""
    if not features.OUTPUT.exists():
        raise SystemExit(f"{features.OUTPUT} not found; run `python -m pipeline features` first")
    rng = np.random.default_rng(models.SEED)
    flags = features.flag_matrix(pd.read_parquet(gold.HOURLY_OUTPUT))

    fit_origins = pd.read_parquet(features.OUTPUT, filters=[("split", "in", list(fit_splits))])
    train = models.sample_rows(fit_origins, flags, TZ, models.TRAIN_SHARE, rng)
    print(f"train rows: {len(train)} ({train['origin'].nunique()} origins, "
          f"{train['origin'].min():%Y-%m-%d} to {train['origin'].max():%Y-%m-%d})")
    del fit_origins
    print("fit logistic regression (24 models)")
    logreg = models.fit_logreg(train)
    del train

    test = pd.read_parquet(features.OUTPUT, filters=[("split", "==", split)])
    scored = baselines.score_table(test, flags)
    for h in range(1, features.HORIZON + 1):
        rows = models.with_horizon(test, h, flags, TZ)
        scored.loc[(scored["horizon"] == h).to_numpy(), "logreg"] = (
            logreg[h].predict_proba(models.logreg_inputs(rows))[:, 1])

    bands = evaluate.alert_rate_bands(flags, pd.Timestamp(SPLITS[split][0]))
    by_horizon = evaluate.brier_by_horizon(scored, FORECASTS)
    summary = evaluate.skill_summary(scored, FORECASTS, baselines.REFERENCE, bands)

    REPORTS.mkdir(parents=True, exist_ok=True)
    by_horizon.round(5).to_csv(REPORTS / f"final_{split}_by_horizon.csv")
    summary.round(5).to_csv(REPORTS / f"final_{split}.csv", index=False)
    names = load_raions().set_index("raion_uid")["raion"]
    plots.skill_by_horizon(by_horizon, baselines.REFERENCE, REPORTS / f"final_{split}_skill_by_horizon.png", split)
    plots.reliability(scored, ["rate_28d", "logreg"], REPORTS / f"final_{split}_reliability.png", split)
    plots.examples(scored, names, REPORTS / f"final_{split}_examples.png", "logreg", split)

    print(f"{scored['origin'].nunique()} origins x {scored['raion_uid'].nunique()} raions x "
          f"{features.HORIZON} horizons = {len(scored)} forecasts")
    print("Brier at horizons 1, 3, 6, 12, 24:")
    print(by_horizon.loc[[1, 3, 6, 12, 24]].round(4).to_string())
    print(f"Skill against {baselines.REFERENCE} (95% interval):")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    run()
