import pandas as pd

from pipeline.gold import daily_table, expand_to_raions, merge_intervals, split_by_day


def ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def alerts(*rows) -> pd.DataFrame:
    """Silver-shaped alerts: (level, oblast_uid, raion_uid, started_at, finished_at)."""
    df = pd.DataFrame(rows, columns=["level", "oblast_uid", "raion_uid", "started_at", "finished_at"])
    df["started_at"] = df["started_at"].map(ts)
    df["finished_at"] = df["finished_at"].map(ts)
    return df


def gold(raions, *rows, start="2024-01-01", end="2024-01-04") -> pd.DataFrame:
    intervals = merge_intervals(expand_to_raions(alerts(*rows), raions))
    return daily_table(split_by_day(intervals), raions, start + "T00:00:00Z", end + "T00:00:00Z")


def hours(table: pd.DataFrame, raion_uid: str) -> list[float]:
    return list(table[table["raion_uid"] == raion_uid].sort_values("day")["alert_hours"])


def test_oblast_alert_covers_every_raion_and_kyiv_is_one_unit(raions):
    table = gold(raions,
                 ("oblast", "1", None, "2024-01-01 10:00", "2024-01-01 12:00"),
                 ("oblast", "2", None, "2024-01-01 10:00", "2024-01-01 11:00"))
    assert hours(table, "11") == [2, 0, 0]
    assert hours(table, "12") == [2, 0, 0]
    assert hours(table, "2") == [1, 0, 0]


def test_hromada_alert_counts_for_its_whole_raion_only(raions):
    table = gold(raions, ("hromada", "1", "12", "2024-01-01 10:00", "2024-01-01 13:00"))
    assert hours(table, "12") == [3, 0, 0]
    assert hours(table, "11") == [0, 0, 0]


def test_overlapping_alerts_are_not_counted_twice(raions):
    table = gold(raions,
                 ("oblast", "1", None, "2024-01-01 10:00", "2024-01-01 12:00"),
                 ("raion", "1", "11", "2024-01-01 11:00", "2024-01-01 13:00"),
                 ("hromada", "1", "11", "2024-01-01 11:30", "2024-01-01 11:45"))
    assert hours(table, "11") == [3, 0, 0]
    assert hours(table, "12") == [2, 0, 0]


def test_alert_is_split_at_utc_midnight(raions):
    table = gold(raions, ("raion", "1", "11", "2024-01-01 22:00", "2024-01-03 03:00"))
    assert hours(table, "11") == [2, 24, 3]


def test_alert_ending_at_midnight_adds_nothing_to_next_day(raions):
    table = gold(raions, ("raion", "1", "11", "2024-01-01 23:00", "2024-01-02 00:00"))
    assert hours(table, "11") == [1, 0, 0]


def test_table_has_every_raion_for_every_day(raions):
    table = gold(raions, ("raion", "1", "11", "2024-01-02 10:00", "2024-01-02 11:00"))
    assert len(table) == len(raions) * 3
    assert not table.duplicated(["raion_uid", "day"]).any()
    assert set(table["raion_uid"]) == {"11", "12", "2"}  # the excluded oblast is not there


def test_days_outside_the_table_range_are_dropped(raions):
    table = gold(raions, ("raion", "1", "11", "2023-12-31 20:00", "2024-01-01 02:00"))
    assert hours(table, "11") == [2, 0, 0]
