import pandas as pd
import pytest

from pipeline.evaluate import alert_rate_bands, brier_by_horizon, skill_summary


def scored_table() -> pd.DataFrame:
    """Two days, two raions, horizon 1; forecast `good` is closer to the outcome than `ref`."""
    origins = pd.to_datetime(["2024-01-01 00:00", "2024-01-02 00:00"], utc=True)
    return pd.DataFrame({
        "origin": origins.repeat(2), "raion_uid": ["a", "b"] * 2, "horizon": 1,
        "alert": [1, 0, 1, 0], "good": [0.9, 0.1, 0.8, 0.2], "ref": [0.5, 0.5, 0.5, 0.5],
    })


def test_brier_by_horizon():
    brier = brier_by_horizon(scored_table(), ["good", "ref"])
    assert brier.loc[1, "good"] == pytest.approx((0.01 + 0.01 + 0.04 + 0.04) / 4)
    assert brier.loc[1, "ref"] == pytest.approx(0.25)


def test_skill_against_reference_with_interval():
    bands = pd.Series({"a": "20-60%", "b": "<10%"})
    summary = skill_summary(scored_table(), ["good", "ref"], "ref", bands, n_boot=200).set_index(["group", "forecast"])
    good = summary.loc[("all", "good")]
    assert good["skill"] == pytest.approx(1 - 0.025 / 0.25)
    assert good["skill_low"] <= good["skill"] <= good["skill_high"]
    assert summary.loc[("all", "ref"), "skill"] == 0
    assert summary.loc[("<10%", "good"), "raions"] == 1


def test_bands_use_only_hours_before_the_split():
    index = pd.date_range("2024-01-01", periods=24 * 100, freq="h", tz="UTC")
    flags = pd.DataFrame({"quiet": 0, "busy": 0}, index=index)
    before = index[24 * 90]
    flags.loc[:before - pd.Timedelta(hours=1), "busy"] = 1  # busy until the split starts
    flags.loc[before:, "quiet"] = 1  # quiet raion only alerts inside the split
    bands = alert_rate_bands(flags, before)
    assert bands.to_dict() == {"quiet": "<10%", "busy": ">=60%"}
