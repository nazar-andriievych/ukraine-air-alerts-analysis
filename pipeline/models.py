"""Train logistic regression and LightGBM on train origins, score them on validation with the baselines.

Usage: python -m pipeline.models [--calendar utc]

Input:  data/features/origins.parquet, data/gold/raion_hourly.parquet
Output: reports/models_validation.csv             Brier and skill against the 28-day rate,
                                                  overall and per alert-rate band
        reports/models_validation_by_horizon.csv  Brier per horizon and forecast
        reports/models_validation_lightgbm_vs_logreg.csv  skill of LightGBM against logistic regression
        reports/lightgbm_importance.csv           LightGBM feature importance (gain)
        reports/*.png                             charts (see pipeline/plots.py)
With --calendar utc the calendar features use UTC instead of Kyiv time and only the two summary
tables are written, with a _utc suffix, to compare the two on validation.

Training rows: a random third of the training origins (all raions), and for each (origin, raion)
one random horizon from each of 1-8, 9-16 and 17-24 h, about 4.3M rows. Forecasts from neighbouring
hours and horizons are nearly the same example, so this loses little and keeps memory low.
Both models train on exactly these rows.

- logreg:   one logistic regression per horizon. Numeric features are scaled; the spell length is
            capped at 48 h; the 28-day rate also enters as a logit, so the model can reproduce the
            baseline; raion, oblast, hour and weekday are one-hot. Default L2 penalty.
- lightgbm: one model for all raions and horizons. The number of trees is picked by early stopping
            on a sample of validation rows (the test split is never used here).
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipeline import baselines, evaluate, features, gold, plots
from pipeline.config import REPORTS, SPLITS
from pipeline.regions import load_raions

SEED = 0
TRAIN_SHARE = 1 / 3
HORIZON_GROUPS = [(1, 8), (9, 16), (17, 24)]
SPELL_CAP = 48  # hours, logistic regression only
NUMERIC = ["alert_now", "spell_hours", "alerts_24h", *[f"share_{k}h" for k in features.SHARE_WINDOWS],
           "oblast_now", "national_now"]
CALENDAR = ["target_hour", "target_weekday"]
CATEGORICAL = ["raion_uid", "oblast_uid"]
LIGHTGBM_FEATURES = [*NUMERIC, "horizon", *CALENDAR, *CATEGORICAL]
LOGREG_NUMERIC = [*NUMERIC, "logit_rate_28d"]
LOGREG_ONE_HOT = [*CATEGORICAL, *CALENDAR]
LIGHTGBM_PARAMS = dict(objective="binary", n_estimators=2000, learning_rate=0.05, num_leaves=63,
                       min_child_samples=500, subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
                       random_state=SEED, verbose=-1)
FORECASTS = [*baselines.BASELINES, "logreg", "lightgbm"]
TIMEZONES = {"kyiv": "Europe/Kyiv", "utc": "UTC"}


def with_horizon(origins: pd.DataFrame, horizon, flags: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Origin rows paired with a horizon (a number or one per row): adds calendar and the outcome."""
    rows = origins.assign(horizon=horizon)
    target = rows["origin"] + (rows["horizon"] - 1) * features.HOUR
    local = target.dt.tz_convert(tz)
    rows["target_hour"] = local.dt.hour
    rows["target_weekday"] = local.dt.dayofweek
    rows["alert"] = features.at(flags, target, rows["raion_uid"])
    return rows


def sample_rows(origins: pd.DataFrame, flags: pd.DataFrame, tz: str, origin_share: float,
                rng: np.random.Generator) -> pd.DataFrame:
    """A share of the origins (all raions), each with one random horizon per horizon group."""
    times = origins["origin"].unique()
    keep = rng.choice(len(times), size=round(len(times) * origin_share), replace=False)
    picked = origins[origins["origin"].isin(times[keep])]
    return pd.concat([with_horizon(picked, rng.integers(lo, hi + 1, len(picked)), flags, tz)
                      for lo, hi in HORIZON_GROUPS], ignore_index=True)


def as_categories(rows: pd.DataFrame, categories: dict[str, list]) -> pd.DataFrame:
    """Same category codes in every table, so LightGBM sees raions consistently."""
    return rows.assign(**{c: pd.Categorical(rows[c], categories=categories[c]) for c in CATEGORICAL})


def logreg_inputs(rows: pd.DataFrame) -> pd.DataFrame:
    rate = rows["share_672h"].clip(1 / 672, 1 - 1 / 672)
    return rows.assign(spell_hours=rows["spell_hours"].clip(upper=SPELL_CAP),
                       logit_rate_28d=np.log(rate / (1 - rate)))[[*LOGREG_NUMERIC, *LOGREG_ONE_HOT]]


def fit_logreg(train: pd.DataFrame) -> dict[int, object]:
    """One scaled + one-hot logistic regression per horizon."""
    models = {}
    for h, rows in train.groupby("horizon"):
        model = make_pipeline(
            ColumnTransformer([("numeric", StandardScaler(), LOGREG_NUMERIC),
                               ("one_hot", OneHotEncoder(handle_unknown="ignore"), LOGREG_ONE_HOT)]),
            LogisticRegression(max_iter=2000))
        models[h] = model.fit(logreg_inputs(rows), rows["alert"])
    return models


def fit_lightgbm(train: pd.DataFrame, valid: pd.DataFrame) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**LIGHTGBM_PARAMS)
    return model.fit(train[LIGHTGBM_FEATURES], train["alert"],
                     eval_X=(valid[LIGHTGBM_FEATURES],), eval_y=(valid["alert"],),
                     callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])


def score(origins: pd.DataFrame, flags: pd.DataFrame, tz: str, logreg: dict, lightgbm: lgb.LGBMClassifier,
          categories: dict[str, list]) -> pd.DataFrame:
    """Baselines and both models for every (origin, raion, horizon) of `origins`."""
    scored = baselines.score_table(origins, flags)
    for h in range(1, features.HORIZON + 1):
        rows = as_categories(with_horizon(origins, h, flags, tz), categories)
        block = (scored["horizon"] == h).to_numpy()  # score_table keeps the origin row order per horizon
        scored.loc[block, "logreg"] = logreg[h].predict_proba(logreg_inputs(rows))[:, 1]
        scored.loc[block, "lightgbm"] = lightgbm.predict_proba(rows[LIGHTGBM_FEATURES])[:, 1]
    return scored


def run(calendar: str = "kyiv") -> None:
    if not features.OUTPUT.exists():
        raise SystemExit(f"{features.OUTPUT} not found; run `python -m pipeline features` first")
    tz, suffix = TIMEZONES[calendar], "" if calendar == "kyiv" else f"_{calendar}"
    rng = np.random.default_rng(SEED)

    flags = features.flag_matrix(pd.read_parquet(gold.HOURLY_OUTPUT))
    categories = {"raion_uid": list(flags.columns),
                  "oblast_uid": sorted(pd.read_parquet(features.OUTPUT, columns=["oblast_uid"])["oblast_uid"].unique())}
    train_origins = pd.read_parquet(features.OUTPUT, filters=[("split", "==", "train")])
    train = as_categories(sample_rows(train_origins, flags, tz, TRAIN_SHARE, rng), categories)
    del train_origins
    val_origins = pd.read_parquet(features.OUTPUT, filters=[("split", "==", "validation")])
    early_stop = as_categories(sample_rows(val_origins, flags, tz, 1.0, rng), categories)
    print(f"train rows: {len(train)} ({train['origin'].nunique()} origins), "
          f"early-stopping rows: {len(early_stop)}, calendar: {tz}")

    print("fit logistic regression (24 models)")
    logreg = fit_logreg(train)
    print("fit LightGBM")
    lightgbm = fit_lightgbm(train, early_stop)
    print(f"LightGBM trees: {lightgbm.best_iteration_}")
    del train, early_stop

    scored = score(val_origins, flags, tz, logreg, lightgbm, categories)
    bands = evaluate.alert_rate_bands(flags, pd.Timestamp(SPLITS["validation"][0]))
    by_horizon = evaluate.brier_by_horizon(scored, FORECASTS)
    summary = evaluate.skill_summary(scored, FORECASTS, baselines.REFERENCE, bands)
    # The choice rule: LightGBM is kept only if it beats logistic regression beyond the interval.
    head_to_head = evaluate.skill_summary(scored, ["lightgbm"], "logreg", bands)

    REPORTS.mkdir(parents=True, exist_ok=True)
    by_horizon.round(5).to_csv(REPORTS / f"models_validation{suffix}_by_horizon.csv")
    summary.round(5).to_csv(REPORTS / f"models_validation{suffix}.csv", index=False)
    head_to_head.round(5).to_csv(REPORTS / f"models_validation{suffix}_lightgbm_vs_logreg.csv", index=False)
    if calendar == "kyiv":
        importance = (pd.Series(lightgbm.booster_.feature_importance("gain"), index=LIGHTGBM_FEATURES)
                      .sort_values(ascending=False))
        (importance / importance.sum()).round(4).rename("gain_share").to_csv(REPORTS / "lightgbm_importance.csv")
        names = load_raions().set_index("raion_uid")["raion"]
        plots.all_charts(scored, by_horizon, importance, names, REPORTS)

    print("Brier at horizons 1, 3, 6, 12, 24:")
    print(by_horizon.loc[[1, 3, 6, 12, 24]].round(4).to_string())
    print(f"Skill against {baselines.REFERENCE} (95% interval):")
    print(summary[summary["forecast"].isin(["logreg", "lightgbm"])].round(3).to_string(index=False))
    print("Skill of LightGBM against logistic regression (95% interval):")
    print(head_to_head.round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and score the models on validation.")
    parser.add_argument("--calendar", choices=list(TIMEZONES), default="kyiv")
    run(parser.parse_args().calendar)
