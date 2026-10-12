import numpy as np
import pandas as pd

from pipeline.models import logreg_inputs, sample_rows, with_horizon

START = pd.Timestamp("2026-03-28", tz="UTC")


def flags(hours: int = 72) -> pd.DataFrame:
    index = pd.date_range(START, periods=hours, freq="h")
    return pd.DataFrame({"11": (np.arange(hours) % 5 == 0).astype("int8"), "12": 0}, index=index)


def origins(times, uids=("11", "12")) -> pd.DataFrame:
    rows = [(t, u) for t in times for u in uids]
    return pd.DataFrame(rows, columns=["origin", "raion_uid"])


def test_calendar_is_the_target_hour_in_kyiv_time_across_dst():
    # Kyiv moves from UTC+2 to UTC+3 at 2026-03-29 01:00 UTC.
    base = origins([pd.Timestamp("2026-03-29 00:00", tz="UTC")], ["11"])
    h1 = with_horizon(base, 1, flags(), "Europe/Kyiv")
    h3 = with_horizon(base, 3, flags(), "Europe/Kyiv")
    assert (h1["target_hour"].item(), h3["target_hour"].item()) == (2, 5)
    assert h1["target_weekday"].item() == 6  # Sunday


def test_outcome_is_the_flag_of_the_target_hour():
    base = origins([START + pd.Timedelta(hours=3)], ["11"])
    assert with_horizon(base, 3, flags(), "UTC")["alert"].item() == 1  # hour 5 is flagged
    assert with_horizon(base, 2, flags(), "UTC")["alert"].item() == 0


def test_sample_rows_takes_a_share_of_origins_and_one_horizon_per_group():
    times = pd.date_range(START, periods=30, freq="h")
    rows = sample_rows(origins(times), flags(), "UTC", 1 / 3, np.random.default_rng(0))
    assert rows["origin"].nunique() == 10
    per_row = rows.groupby(["origin", "raion_uid"])["horizon"].apply(sorted)
    assert len(per_row) == 20
    for horizons in per_row:
        assert [1 <= horizons[0] <= 8, 9 <= horizons[1] <= 16, 17 <= horizons[2] <= 24] == [True] * 3


def test_logreg_inputs_cap_the_spell_and_keep_the_rate_logit_finite():
    rows = pd.DataFrame({"alert_now": [0, 1], "spell_hours": [500.0, 2.0], "alerts_24h": [0, 3],
                         **{f"share_{k}h": [0.0, 1.0] for k in [1, 3, 6, 24, 168, 672]},
                         "oblast_now": [0, 1], "national_now": [0.1, 0.2], "raion_uid": ["11", "12"],
                         "oblast_uid": ["1", "1"], "target_hour": [0, 1], "target_weekday": [0, 1]})
    inputs = logreg_inputs(rows)
    assert inputs["spell_hours"].tolist() == [48, 2]
    assert np.isfinite(inputs["logit_rate_28d"]).all()
