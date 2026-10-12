"""Weather experiment: logistic regression with and without weather at the target hour, on validation.

Usage: python -m pipeline models --weather oracle|forecast   (same as python -m pipeline.models --weather ...)

Input:  data/features/origins.parquet, data/gold/raion_hourly.parquet,
        data/silver/weather_actual.parquet, data/silver/weather_forecast.parquet (python -m pipeline weather)
Output: reports/models_validation_weather_<mode>.csv             Brier and skill against the 28-day rate,
                                                                  overall and per alert-rate band
        reports/models_validation_weather_<mode>_by_horizon.csv  Brier per horizon and forecast
        reports/models_validation_weather_<mode>_vs_<ref>.csv    skill of the weather model against <ref>

Weather of the raion's oblast at the target hour is added to the logistic regression as scaled
numeric features (precipitation as log(1 + mm)). Everything else is the same as pipeline/models.py,
including the training rows: the same seed draws the same sample, and every variant of a mode trains
on exactly the same rows.

- oracle:   ACTUAL (ERA5) weather at the target hour: wind speed, gusts, precipitation, low cloud,
            temperature. It is not known at the origin. UPPER BOUND, USES FUTURE WEATHER, NOT A RESULT.
            Variants: logreg (no weather) and logreg_oracle, trained on the full training period.
- forecast: the day-ahead forecast for the target hour from a model run started at least 48 h before
            it (Open-Meteo `_previous_day2`), so it was published before the origin (see
            `check_published`). Wind, gusts and precipitation exist from 2024-01-20 12:00 UTC, so:
              logreg              no weather, full training period (the current model)
              logreg_same_period  no weather, training origins from WEATHER_START only
              logreg_weather      with weather, the same rows as logreg_same_period
"""
import numpy as np
import pandas as pd

from pipeline import baselines, evaluate, features, gold, models, weather
from pipeline.config import REPORTS, SPLITS

WEATHER_START = pd.Timestamp("2024-01-21", tz="UTC")  # first full UTC day with day2 wind and precipitation
DAY2_LEAD = pd.Timedelta(hours=48)  # a _previous_day2 value comes from a run started >= 48 h before its hour
PUBLISH_DELAY = pd.Timedelta(hours=12)  # generous bound on how long after its start a run is published
MODES = {
    "oracle": {"source": "actual", "columns": ["wind_speed_10m", "wind_gusts_10m", "precipitation",
                                               "cloud_cover_low", "temperature_2m"],
               "variants": ["logreg", "logreg_oracle"], "reference": "logreg"},
    "forecast": {"source": "forecast", "columns": ["wind_speed_10m", "wind_gusts_10m", "precipitation",
                                                   "cloud_cover", "temperature_2m"],
                 "variants": ["logreg", "logreg_same_period", "logreg_weather"],
                 "reference": "logreg_same_period"},
}


def wx(column: str) -> str:
    return f"wx_{column}"


def check_published(origin: pd.Series, target: pd.Series) -> None:
    """Fail unless every day2 forecast for `target` was published before `origin`.

    Open-Meteo stamps the hour starting at `target` by its end, target + 1 h. Its day2 value comes
    from a run started at least DAY2_LEAD before that stamp and published within PUBLISH_DELAY.
    """
    published = target + features.HOUR - DAY2_LEAD + PUBLISH_DELAY
    if (published > origin).any():
        raise ValueError("a day2 forecast would be published after the origin")


def target_weather(rows: pd.DataFrame, matrices: dict[str, pd.DataFrame], columns: list[str],
                   mode: str) -> pd.DataFrame:
    """Add the oblast's weather at each row's target hour (origin + (horizon - 1) h) as wx_* columns."""
    target = rows["origin"] + (rows["horizon"] - 1) * features.HOUR
    if mode == "forecast":
        check_published(rows["origin"], target)
    added = {}
    for c in columns:
        values = features.at(matrices[c], target, rows["oblast_uid"].astype(str))
        added[wx(c)] = np.log1p(values) if c == "precipitation" else values
    return rows.assign(**added)


def weather_matrices(mode: str) -> dict[str, pd.DataFrame]:
    path = weather.OUTPUTS[MODES[mode]["source"]]
    if not path.exists():
        raise SystemExit(f"{path} not found; run `python -m pipeline weather` first")
    matrices = weather.matrices(pd.read_parquet(path))
    # Reindex to every hour of the data, so hours without weather give NaN instead of a KeyError.
    hours = pd.date_range(SPLITS["train"][0], SPLITS["test"][1], freq="h", inclusive="left")
    return {c: m.reindex(hours) for c, m in matrices.items()}


def fit_variants(train: pd.DataFrame, mode: str) -> dict[str, dict]:
    columns = [wx(c) for c in MODES[mode]["columns"]]
    if mode == "oracle":
        train = train.dropna(subset=columns)
        print(f"train rows: {len(train)} ({train['origin'].nunique()} origins)")
        return {"logreg": models.fit_logreg(train), "logreg_oracle": models.fit_logreg(train, columns)}
    print(f"train rows, full period: {len(train)} ({train['origin'].nunique()} origins)")
    fitted = {"logreg": models.fit_logreg(train)}
    same = train[train["origin"] >= WEATHER_START].dropna(subset=columns)
    print(f"train rows from {WEATHER_START.date()}: {len(same)} ({same['origin'].nunique()} origins), "
          f"dropped for missing weather: {int((train['origin'] >= WEATHER_START).sum()) - len(same)}")
    fitted["logreg_same_period"] = models.fit_logreg(same)
    fitted["logreg_weather"] = models.fit_logreg(same, columns)
    return fitted


def run(mode: str, calendar: str = "kyiv") -> None:
    if mode not in MODES:
        raise SystemExit(f"weather must be one of {list(MODES)}, not {mode!r}")
    if not features.OUTPUT.exists():
        raise SystemExit(f"{features.OUTPUT} not found; run `python -m pipeline features` first")
    spec, tz = MODES[mode], models.TIMEZONES[calendar]
    suffix = f"_weather_{mode}" + ("" if calendar == "kyiv" else f"_{calendar}")
    columns, extra = spec["columns"], [wx(c) for c in spec["columns"]]
    if mode == "oracle":
        print("ORACLE: actual weather at the target hour. Upper bound, uses future weather, not a result.")
    rng = np.random.default_rng(models.SEED)  # same seed and draw order as models.run: same training rows

    flags = features.flag_matrix(pd.read_parquet(gold.HOURLY_OUTPUT))
    categories = {"raion_uid": list(flags.columns),
                  "oblast_uid": sorted(pd.read_parquet(features.OUTPUT, columns=["oblast_uid"])["oblast_uid"].unique())}
    matrices = weather_matrices(mode)
    train_origins = pd.read_parquet(features.OUTPUT, filters=[("split", "==", "train")])
    train = models.sample_rows(train_origins, flags, tz, models.TRAIN_SHARE, rng)
    del train_origins
    train = models.as_categories(target_weather(train, matrices, columns, mode), categories)
    fitted = fit_variants(train, mode)
    del train

    val_origins = pd.read_parquet(features.OUTPUT, filters=[("split", "==", "validation")])
    scored = baselines.score_table(val_origins, flags)
    for h in range(1, features.HORIZON + 1):
        rows = models.as_categories(
            target_weather(models.with_horizon(val_origins, h, flags, tz), matrices, columns, mode), categories)
        if rows[extra].isna().any().any():
            raise SystemExit(f"validation rows without weather at horizon {h}")
        block = (scored["horizon"] == h).to_numpy()  # score_table keeps the origin row order per horizon
        for name, per_horizon in fitted.items():
            use = extra if name in ("logreg_oracle", "logreg_weather") else ()
            scored.loc[block, name] = per_horizon[h].predict_proba(models.logreg_inputs(rows, use))[:, 1]

    variants, reference = spec["variants"], spec["reference"]
    forecasts = [*baselines.BASELINES, *variants]
    bands = evaluate.alert_rate_bands(flags, pd.Timestamp(SPLITS["validation"][0]))
    by_horizon = evaluate.brier_by_horizon(scored, forecasts)
    summary = evaluate.skill_summary(scored, forecasts, baselines.REFERENCE, bands)
    head_to_head = evaluate.skill_summary(scored, [variants[-1]], reference, bands)
    skill_by_horizon = 1 - by_horizon[variants].div(by_horizon[baselines.REFERENCE], axis=0)
    REPORTS.mkdir(parents=True, exist_ok=True)
    by_horizon.join(skill_by_horizon.add_prefix("skill_")).round(5).to_csv(
        REPORTS / f"models_validation{suffix}_by_horizon.csv")
    summary.round(5).to_csv(REPORTS / f"models_validation{suffix}.csv", index=False)
    head_to_head.round(5).to_csv(REPORTS / f"models_validation{suffix}_vs_{reference}.csv", index=False)
    if mode == "forecast":  # the adoption rule also compares with the full-period model
        evaluate.skill_summary(scored, ["logreg_weather"], "logreg", bands).round(5).to_csv(
            REPORTS / f"models_validation{suffix}_vs_logreg.csv", index=False)

    label = " (UPPER BOUND, uses future weather)" if mode == "oracle" else ""
    print(f"Skill against {baselines.REFERENCE} by horizon{label}:")
    print(skill_by_horizon.loc[[1, 3, 6, 12, 24]].round(4).to_string())
    print(f"Skill against {baselines.REFERENCE} (95% interval){label}:")
    print(summary[summary["forecast"].isin(variants)].round(4).to_string(index=False))
    print(f"Skill of {variants[-1]} against {reference} (95% interval){label}:")
    print(head_to_head.round(4).to_string(index=False))
