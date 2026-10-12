"""Scores shared by the baselines and the models.

A scored table has one row per (origin, raion_uid, horizon), the outcome in `alert` (0 or 1),
and one column of probabilities per forecast.

- Brier score: mean of (probability - outcome)^2. 0 is perfect; lower is better.
- Brier skill score against a reference: 1 - Brier(forecast) / Brier(reference).
  0.10 means 10% lower Brier than the reference; below 0 means worse than the reference.
"""
import numpy as np
import pandas as pd

DAY = pd.Timedelta(days=1)
HOUR = pd.Timedelta(hours=1)
BAND_DAYS = 84  # alert-rate bands are measured over the 12 weeks before the evaluated split
BAND_EDGES = [0, 0.1, 0.2, 0.6, np.inf]
BAND_LABELS = ["<10%", "10-20%", "20-60%", ">=60%"]


def squared_errors(scored: pd.DataFrame, forecasts: list[str]) -> pd.DataFrame:
    return scored[forecasts].sub(scored["alert"], axis=0).pow(2)


def brier_by_horizon(scored: pd.DataFrame, forecasts: list[str]) -> pd.DataFrame:
    """Brier score per horizon (rows) and forecast (columns)."""
    return squared_errors(scored, forecasts).groupby(scored["horizon"]).mean()


def alert_rate_bands(flags: pd.DataFrame, before: pd.Timestamp) -> pd.Series:
    """Band of each raion by its share of hours under alert in the BAND_DAYS days before `before`.

    Measured before the evaluated period, so raions are not grouped by the outcomes being scored.
    """
    rate = flags.loc[before - BAND_DAYS * DAY: before - HOUR].mean()
    return pd.cut(rate, BAND_EDGES, labels=BAND_LABELS, right=False).astype(str)


def skill_summary(scored: pd.DataFrame, forecasts: list[str], reference: str, bands: pd.Series,
                  n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Brier and skill against `reference`, overall and per band, with 95% bootstrap intervals.

    Forecasts from neighbouring hours overlap, so rows are not independent. The bootstrap resamples
    whole days of origins (the same days for every forecast) instead of single rows.
    """
    errors = squared_errors(scored, list(dict.fromkeys([*forecasts, reference])))
    day = scored["origin"].dt.floor("D")
    band = scored["raion_uid"].map(bands)
    rng = np.random.default_rng(seed)
    rows = []
    for group in ["all", *BAND_LABELS]:
        mask = band.notna() if group == "all" else band == group
        if not mask.any():
            continue
        daily = errors[mask].groupby(day[mask]).sum()
        sample = rng.integers(len(daily), size=(n_boot, len(daily)))
        ref, ref_boot = daily[reference].sum(), daily[reference].to_numpy()[sample].sum(axis=1)
        for forecast in forecasts:
            boot = 1 - daily[forecast].to_numpy()[sample].sum(axis=1) / ref_boot
            rows.append({
                "group": group, "raions": int(scored.loc[mask, "raion_uid"].nunique()),
                "forecast": forecast, "brier": errors.loc[mask, forecast].mean(),
                "skill": 1 - daily[forecast].sum() / ref,
                "skill_low": np.quantile(boot, 0.025), "skill_high": np.quantile(boot, 0.975),
            })
    return pd.DataFrame(rows)
