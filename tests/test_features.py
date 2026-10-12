import numpy as np
import pandas as pd
import pytest

from pipeline.baselines import hour_of_day_rate, score_table
from pipeline.features import assign_split, origin_features, share_features, state_features

START = pd.Timestamp("2024-01-01", tz="UTC")
HOUR = pd.Timedelta(hours=1)


def ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def intervals(*rows) -> pd.DataFrame:
    """(raion_uid, started_at, finished_at) rows, as in gold's merged intervals."""
    return pd.DataFrame([(uid, ts(a), ts(b)) for uid, a, b in rows],
                        columns=["raion_uid", "started_at", "finished_at"])


def flags_from(iv: pd.DataFrame, uids: list[str], hours: int) -> pd.DataFrame:
    """Hour x raion flags: 1 if any interval of the raion overlaps the hour."""
    index = pd.date_range(START, periods=hours, freq="h")
    flags = pd.DataFrame(0, index=index, columns=pd.Index(uids, name="raion_uid"), dtype="int8")
    for r in iv.itertuples():
        flags.loc[(index < r.finished_at) & (index + HOUR > r.started_at), r.raion_uid] = 1
    return flags


def test_state_at_origin_uses_only_the_past():
    iv = intervals(("11", "2024-01-01 10:00", "2024-01-01 12:30"))
    origins = pd.DatetimeIndex([ts("2024-01-01 10:00"), ts("2024-01-01 11:00"), ts("2024-01-01 13:00")])
    state = state_features(iv, origins, pd.Index(["11", "12"]), START)
    # an alert starting exactly at the origin is not known yet
    assert state["alert_now"]["11"].tolist() == [0, 1, 0]
    assert state["spell_hours"]["11"].tolist() == [10, 1, 0.5]  # quiet since data start, in alert 1 h, quiet 0.5 h
    assert state["alerts_24h"]["11"].tolist() == [0, 1, 1]
    assert state["spell_hours"]["12"].tolist() == [10, 11, 13]  # never alerted: hours since data start


def test_share_excludes_the_origin_hour():
    flags = flags_from(intervals(("11", "2024-01-01 02:00", "2024-01-01 03:00")), ["11"], 6)
    share = share_features(flags)
    assert share["share_1h"]["11"].tolist()[1:] == [0, 0, 1, 0, 0]  # origin 03:00 sees hour 02:00
    assert share["share_3h"]["11"].iloc[4] == pytest.approx(1 / 3)


def test_context_is_the_oblast_and_national_share_under_alert():
    iv = intervals(("11", "2024-01-01 00:30", "2024-01-01 05:00"))
    flags = flags_from(iv, ["11", "12", "2"], 3)
    oblast_of = pd.Series({"11": "1", "12": "1", "2": "2"})
    table = origin_features(flags, iv, oblast_of, START).set_index(["origin", "raion_uid"])
    row = table.loc[(ts("2024-01-01 01:00"), "12")]
    assert (row["alert_now"], row["oblast_now"], row["national_now"]) == (0, 0.5, pytest.approx(1 / 3))


def test_features_do_not_change_when_the_future_changes():
    """Leakage check: scramble everything from the origin on; the origin's features must stay the same."""
    rng = np.random.default_rng(1)
    uids, hours, origin = ["11", "12", "2"], 24 * 40, START + 24 * 35 * HOUR
    oblast_of = pd.Series({"11": "1", "12": "1", "2": "2"})
    rows = []
    for _ in range(300):
        a = START + pd.Timedelta(minutes=int(rng.integers(0, hours * 60)))
        rows.append((rng.choice(uids), str(a), str(a + pd.Timedelta(minutes=int(rng.integers(5, 600))))))
    iv = intervals(*rows)
    iv = iv[~iv.duplicated()]

    future = iv.copy()
    future = future[future["started_at"] < origin]  # drop alerts that start after the origin ...
    running = future["finished_at"] > origin
    future.loc[running, "finished_at"] = origin + 100 * HOUR  # ... change when running ones end ...
    future = pd.concat([future, intervals(("12", str(origin + HOUR), str(origin + 9 * HOUR)))])  # ... add new ones
    flags = flags_from(iv, uids, hours)
    scrambled = flags.copy()
    scrambled.loc[origin:] = 1 - scrambled.loc[origin:]

    before = origin_features(flags, iv, oblast_of, START)
    after = origin_features(scrambled, future, oblast_of, START)
    pd.testing.assert_frame_equal(before[before["origin"] == origin].reset_index(drop=True),
                                  after[after["origin"] == origin].reset_index(drop=True))


def test_origin_split_needs_all_24_target_hours_inside():
    origins = pd.Series([ts("2026-05-17 00:00"), ts("2026-05-17 01:00"), ts("2026-05-18 00:00"),
                         ts("2026-08-30 23:00"), ts("2026-09-06 00:00"), ts("2026-09-06 01:00")])
    assert assign_split(origins).tolist() == ["train", "none", "validation", "test", "test", "none"]


def test_hour_of_day_rate_uses_the_same_hour_on_earlier_days():
    flags = pd.DataFrame({"11": [1 if h % 24 == 5 else 0 for h in range(24 * 30)]},
                         index=pd.date_range(START, periods=24 * 30, freq="h"))
    rate = hour_of_day_rate(flags)
    assert rate["11"].iloc[24 * 29 + 5] == 1
    assert rate["11"].iloc[24 * 29 + 6] == 0
    assert np.isnan(rate["11"].iloc[24 * 2 + 5])  # fewer than 28 earlier days: no rate


def test_score_table_labels_horizon_h_with_the_hour_starting_h_minus_1_after_origin():
    flags = pd.DataFrame({"11": [0] * 60}, index=pd.date_range(START, periods=60, freq="h"))
    flags.iloc[10 + 3, 0] = 1  # hour 13:00
    origins = pd.DataFrame({"origin": [START + 10 * HOUR], "raion_uid": ["11"],
                            "alert_now": [0], "share_672h": [0.2]})
    scored = score_table(origins, flags)
    assert scored.loc[scored["alert"] == 1, "horizon"].tolist() == [4]
    assert (scored["rate_28d"] == 0.2).all()
