import pandas as pd
import pytest

from pipeline.config import DATA_END
from pipeline.silver import attach_uids, clean


def raw(*rows) -> pd.DataFrame:
    """Rows shaped like official_data_en.csv: (oblast, raion, hromada, level, started_at, finished_at)."""
    df = pd.DataFrame(rows, columns=["oblast", "raion", "hromada", "level", "started_at", "finished_at"])
    df["source"] = "official"
    return df


def test_clean_removes_duplicates_excluded_and_empty_alerts():
    df = clean(raw(
        ("Alpha oblast", None, None, "oblast", "2024-01-01 10:00:00+00:00", "2024-01-01 11:00:00+00:00"),
        ("Alpha oblast", None, None, "oblast", "2024-01-01 10:00:00+00:00", "2024-01-01 11:00:00+00:00"),
        ("Luhanska oblast", None, None, "oblast", "2024-01-01 10:00:00+00:00", "2024-01-01 11:00:00+00:00"),
        ("Alpha oblast", None, None, "oblast", "2024-01-02 10:00:00+00:00", "2024-01-02 10:00:00+00:00"),
    ))
    assert len(df) == 1
    assert list(df["alert_id"]) == [0]


def test_clean_cuts_alerts_running_at_data_end():
    end = pd.Timestamp(DATA_END)
    df = clean(raw(
        ("Alpha oblast", None, None, "oblast", str(end - pd.Timedelta(hours=1)), str(end + pd.Timedelta(hours=5))),
        ("Alpha oblast", None, None, "oblast", str(end + pd.Timedelta(hours=1)), str(end + pd.Timedelta(hours=2))),
    ))
    assert len(df) == 1
    assert df["finished_at"].iloc[0] == end


def test_clean_turns_raion_name_in_hromada_field_into_raion_alert():
    df = clean(raw(("Alpha oblast", "North raion", "North raion", "hromada",
                    "2024-01-01 10:00:00+00:00", "2024-01-01 11:00:00+00:00")))
    assert df["level"].iloc[0] == "raion"
    assert pd.isna(df["hromada"].iloc[0])


def test_attach_uids_per_level(raions):
    df = attach_uids(clean(raw(
        ("Alpha oblast", None, None, "oblast", "2024-01-01 10:00:00+00:00", "2024-01-01 11:00:00+00:00"),
        ("Alpha oblast", "North raion", None, "raion", "2024-01-01 12:00:00+00:00", "2024-01-01 13:00:00+00:00"),
        ("Alpha oblast", "South raion", "Lake hromada", "hromada",
         "2024-01-01 14:00:00+00:00", "2024-01-01 15:00:00+00:00"),
    )), raions).set_index("level")
    assert list(df["oblast_uid"]) == ["1", "1", "1"]
    assert pd.isna(df.loc["oblast", "raion_uid"])
    assert df.loc["raion", "raion_uid"] == "11"
    # The same hromada name exists in North raion as 111; the raion decides which one it is.
    assert (df.loc["hromada", "raion_uid"], df.loc["hromada", "hromada_uid"]) == ("12", "121")


def test_attach_uids_fails_on_unknown_raion(raions):
    alerts = clean(raw(("Alpha oblast", "Missing raion", None, "raion",
                        "2024-01-01 10:00:00+00:00", "2024-01-01 11:00:00+00:00")))
    with pytest.raises(ValueError, match="match no raion"):
        attach_uids(alerts, raions)
