"""Build the features table: one row per raion per forecast origin, using only data before it.

Usage: python -m pipeline.features

Input:  data/gold/raion_hourly.parquet, data/gold/raion_intervals.parquet
Output: data/features/origins.parquet

A forecast origin is a moment on the hour (UTC), e.g. 2026-07-13 00:00, when a forecast is made.
It forecasts the next 24 hours: horizon h is the hour starting at origin + (h - 1) hours.
Every feature uses only alerts and hours before the origin. Calendar features depend on the
horizon, so they are added when a model is trained, not stored here.

Each origin gets the split its 24 target hours fall in (config.SPLITS). Origins whose targets cross
a split boundary, and origins with less than 28 days of history, are left out.
"""
import json

import numpy as np
import pandas as pd

from pipeline import gold
from pipeline.config import DATA_START, FEATURES, SPLITS

OUTPUT = FEATURES / "origins.parquet"
HOUR = pd.Timedelta(hours=1)
HORIZON = 24
SHARE_WINDOWS = [1, 3, 6, 24, 168, 672]  # hours; 672 h = 28 days
HISTORY = max(SHARE_WINDOWS) * HOUR
US_PER_HOUR = 3_600_000_000


def flag_matrix(hourly: pd.DataFrame) -> pd.DataFrame:
    """Alert flags as an hour x raion matrix; rows are hour starts."""
    return hourly.pivot(index="hour", columns="raion_uid", values="alert")


def at(matrix: pd.DataFrame, times: pd.Series, raion_uids: pd.Series) -> np.ndarray:
    """Values of an hour x raion matrix at the given (time, raion) pairs."""
    rows = matrix.index.get_indexer(times)
    cols = matrix.columns.get_indexer(raion_uids)
    if (rows < 0).any() or (cols < 0).any():
        raise KeyError("time or raion outside the matrix")
    return matrix.to_numpy()[rows, cols]


def _us(times) -> np.ndarray:
    """UTC times as integer microseconds since the epoch."""
    return pd.DatetimeIndex(times).as_unit("us").asi8


def share_features(flags: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Share of the k hours before each origin under alert (origin x raion).

    The origin's own hour is not included: it is still in the future at the origin.
    """
    return {f"share_{k}h": flags.rolling(k).mean().shift(1) for k in SHARE_WINDOWS}


def state_features(intervals: pd.DataFrame, origins: pd.DatetimeIndex, raion_uids: pd.Index,
                   data_start: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Alert state at each origin from merged alert intervals (origin x raion).

    alert_now:   1 if an alert started before the origin and had not ended by it
    spell_hours: hours since the current alert started, or since the last alert ended
                 (since data_start if the raion had no alert yet)
    alerts_24h:  number of alerts that started in the 24 hours before the origin
    """
    o, first = _us(origins), _us([data_start])[0]
    now = np.zeros((len(o), len(raion_uids)), dtype=np.int8)
    spell = np.empty((len(o), len(raion_uids)), dtype=np.float32)
    count = np.zeros((len(o), len(raion_uids)), dtype=np.int16)
    by_raion = dict(tuple(intervals.groupby("raion_uid")))
    for c, uid in enumerate(raion_uids):
        if uid not in by_raion:
            spell[:, c] = (o - first) / US_PER_HOUR
            continue
        iv = by_raion[uid].sort_values("started_at")
        starts, ends = _us(iv["started_at"]), _us(iv["finished_at"])
        last = np.searchsorted(starts, o, side="left") - 1  # last alert that started before the origin
        seen = last >= 0
        j = np.maximum(last, 0)
        under = seen & (ends[j] > o)
        since = np.where(under, starts[j], np.where(seen, ends[j], first))
        now[:, c] = under
        spell[:, c] = (o - since) / US_PER_HOUR
        count[:, c] = last + 1 - np.searchsorted(starts, o - 24 * US_PER_HOUR, side="left")
    frame = lambda values: pd.DataFrame(values, index=origins, columns=raion_uids)
    return {"alert_now": frame(now), "spell_hours": frame(spell), "alerts_24h": frame(count)}


def context_features(alert_now: pd.DataFrame, oblast_of: pd.Series) -> dict[str, pd.DataFrame]:
    """Share of the raion's oblast and of the whole country under alert at the origin."""
    oblast = alert_now.T.groupby(oblast_of.reindex(alert_now.columns).to_numpy()).transform("mean").T
    country = np.repeat(alert_now.mean(axis=1).to_numpy()[:, None], alert_now.shape[1], axis=1)
    return {"oblast_now": oblast,
            "national_now": pd.DataFrame(country, index=alert_now.index, columns=alert_now.columns)}


def origin_features(flags: pd.DataFrame, intervals: pd.DataFrame, oblast_of: pd.Series,
                    data_start: pd.Timestamp) -> pd.DataFrame:
    """One row per (origin, raion) for every hour of `flags` as an origin."""
    origins, uids = flags.index, flags.columns
    state = state_features(intervals, origins, uids, data_start)
    matrices = {**state, **share_features(flags), **context_features(state["alert_now"], oblast_of)}
    table = pd.DataFrame({
        "origin": origins.repeat(len(uids)),
        "raion_uid": np.tile(uids.to_numpy(), len(origins)),
        "oblast_uid": np.tile(oblast_of.reindex(uids).to_numpy(), len(origins)),
    })
    for name, matrix in matrices.items():
        values = matrix.to_numpy().ravel()
        table[name] = values.astype(np.float32) if values.dtype.kind == "f" else values
    return table


def assign_split(origins: pd.Series) -> pd.Series:
    """The split all 24 target hours of each origin fall in, or "none" if they cross a boundary."""
    split = pd.Series("none", index=origins.index)
    for name, (start, end) in SPLITS.items():
        split[(origins >= pd.Timestamp(start)) & (origins + HORIZON * HOUR <= pd.Timestamp(end))] = name
    return split


def build() -> None:
    if not gold.INTERVALS_OUTPUT.exists():
        raise SystemExit(f"{gold.INTERVALS_OUTPUT} not found; run `python -m pipeline gold` first")

    print(f"build {OUTPUT.name}")
    hourly = pd.read_parquet(gold.HOURLY_OUTPUT)
    oblast_of = hourly.drop_duplicates("raion_uid").set_index("raion_uid")["oblast_uid"]
    data_start = pd.Timestamp(DATA_START)
    table = origin_features(flag_matrix(hourly), pd.read_parquet(gold.INTERVALS_OUTPUT), oblast_of, data_start)
    table["split"] = assign_split(table["origin"])
    table = table[(table["origin"] >= data_start + HISTORY) & (table["split"] != "none")].reset_index(drop=True)

    FEATURES.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUTPUT, index=False)
    print(json.dumps({
        "rows": len(table),
        "features": [c for c in table.columns if c not in ("origin", "raion_uid", "oblast_uid", "split")],
        **{name: {"origins": int(part["origin"].nunique()),
                  "first": str(part["origin"].min()), "last": str(part["origin"].max())}
           for name, part in table.groupby("split", sort=False)},
    }, indent=2))


if __name__ == "__main__":
    build()
