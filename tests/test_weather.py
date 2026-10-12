import numpy as np
import pandas as pd
import pytest

from pipeline.weather import chunks, matrices, response_frame
from pipeline.weather_models import check_published, target_weather

START = pd.Timestamp("2026-01-01", tz="UTC")
HOUR = pd.Timedelta(hours=1)
COLUMNS = ["wind_speed_10m", "precipitation"]


def weather_table(hours: int, rng: np.random.Generator) -> pd.DataFrame:
    """Random hourly weather for oblasts "1" and "2", one row per (hour, oblast_uid)."""
    index = pd.MultiIndex.from_product([pd.date_range(START, periods=hours, freq="h"), ["1", "2"]],
                                       names=["hour", "oblast_uid"])
    return pd.DataFrame({c: rng.uniform(0, 20, len(index)) for c in COLUMNS}, index=index).reset_index()


def rows_at(origin: pd.Timestamp) -> pd.DataFrame:
    """Every horizon of one origin for raions "11" (oblast "1") and "2" (oblast "2")."""
    return pd.DataFrame([(origin, uid, oblast, h) for uid, oblast in [("11", "1"), ("2", "2")]
                         for h in range(1, 25)], columns=["origin", "raion_uid", "oblast_uid", "horizon"])


def test_open_meteo_hours_are_moved_to_the_hour_they_describe():
    # Open-Meteo stamps an hour by its end: precipitation at 01:00 fell during 00:00-01:00.
    hourly = {"time": ["2026-01-01T01:00", "2026-01-01T02:00"], "precipitation_previous_day2": [0.5, None]}
    frame = response_frame(hourly, ["precipitation"], "_previous_day2")
    assert frame["hour"].tolist() == [START, START + HOUR]
    assert frame["precipitation"].iloc[0] == 0.5 and np.isnan(frame["precipitation"].iloc[1])


def test_chunks_cover_the_range_without_gaps_or_overlaps():
    parts = chunks("2022-03-16", "2023-01-10", months=6)
    assert parts[0][0] == "2022-03-16" and parts[-1][1] == "2023-01-10"
    for (_, end), (start, _) in zip(parts, parts[1:]):
        assert pd.Timestamp(start) - pd.Timestamp(end) == pd.Timedelta(days=1)


def test_target_weather_is_the_oblast_value_at_origin_plus_horizon_minus_1():
    table = weather_table(24 * 5, np.random.default_rng(0))
    origin = START + 72 * HOUR
    out = target_weather(rows_at(origin), matrices(table), COLUMNS, "forecast").set_index(["raion_uid", "horizon"])
    expected = table.set_index(["hour", "oblast_uid"])
    assert out.loc[("11", 3), "wx_wind_speed_10m"] == pytest.approx(expected.loc[(origin + 2 * HOUR, "1"), "wind_speed_10m"])
    assert out.loc[("2", 1), "wx_precipitation"] == pytest.approx(np.log1p(expected.loc[(origin, "2"), "precipitation"]))


def test_weather_features_do_not_change_when_the_future_changes():
    """Leakage check: change all actual weather from the origin on, and every forecast that is
    published after the origin; the origin's forecast-mode features must stay the same."""
    rng = np.random.default_rng(1)
    origin = START + 24 * 6 * HOUR
    actual, forecast = weather_table(24 * 10, rng), weather_table(24 * 10, rng)
    actual_later, forecast_later = actual.copy(), forecast.copy()
    actual_later.loc[actual_later["hour"] >= origin, COLUMNS] = rng.uniform(0, 20, (int((actual["hour"] >= origin).sum()), 2))
    # A day2 value for an hour comes from a run started >= 48 h before the hour's end stamp; runs
    # started at or after the origin can only cover hours from origin + 47 h on.
    unpublished = forecast_later["hour"] + HOUR - pd.Timedelta(hours=48) >= origin
    forecast_later.loc[unpublished, COLUMNS] = -1.0

    rows = rows_at(origin)
    before = target_weather(rows, matrices(forecast), COLUMNS, "forecast")
    after = target_weather(rows, matrices(forecast_later), COLUMNS, "forecast")
    pd.testing.assert_frame_equal(before, after)
    # The check can see leakage: the oracle (actual weather at the target hour) does change.
    oracle_before = target_weather(rows, matrices(actual), COLUMNS, "oracle")
    oracle_after = target_weather(rows, matrices(actual_later), COLUMNS, "oracle")
    assert not oracle_before.equals(oracle_after)


def test_forecasts_must_be_published_before_the_origin():
    origin = pd.Series([START + 100 * HOUR])
    check_published(origin, origin + 23 * HOUR)  # horizon 24 with day2: fine
    with pytest.raises(ValueError):  # a value 40 h ahead would need a run that is not out yet
        check_published(origin, origin + 40 * HOUR)
