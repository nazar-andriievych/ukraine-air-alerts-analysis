"""Build the silver layer: cleaned alerts, one row per source alert.

Usage: python -m pipeline.silver

Input:  data/bronze/official_data_en.csv, data/reference/regions.json
Output: data/silver/alerts.parquet

Each alert keeps its source `level` (oblast / raion / hromada) and gets the uid of its oblast,
of its raion (raion and hromada alerts) and of its hromada (hromada alerts).
Alerts still running at DATA_END are cut off there. Timestamps are UTC.
Spreading alerts to the raions they covered is the job of the gold layer.
"""
import json

import pandas as pd

from pipeline.config import BRONZE, DATA_END, EXCLUDED_OBLASTS, SILVER
from pipeline.regions import load_raions

SOURCE = BRONZE / "official_data_en.csv"
OUTPUT = SILVER / "alerts.parquet"
COLUMNS = ["alert_id", "level", "oblast", "oblast_uid", "raion", "raion_uid", "hromada", "hromada_uid",
           "started_at", "finished_at"]


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.drop(columns="source").drop_duplicates()
    print(f"  {len(raw) - len(df):>7} exact duplicates removed")

    excluded = df["oblast"].isin(EXCLUDED_OBLASTS)
    print(f"  {excluded.sum():>7} alerts in excluded oblasts removed")
    df = df[~excluded].copy()

    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)

    bad = df["finished_at"] <= df["started_at"]
    print(f"  {bad.sum():>7} alerts with zero or negative duration removed")
    df = df[~bad]

    end = pd.Timestamp(DATA_END)
    late = df["started_at"] >= end
    print(f"  {late.sum():>7} alerts starting at or after {DATA_END} removed")
    df = df[~late].copy()

    running = df["finished_at"] > end
    print(f"  {running.sum():>7} alerts still running at {DATA_END} cut off there")
    df.loc[running, "finished_at"] = end

    # A few rows have the raion name in the hromada field: they are raion alerts.
    raion_in_hromada = df["hromada"].notna() & (df["hromada"] == df["raion"])
    print(f"  {raion_in_hromada.sum():>7} 'hromada' rows that are really raion alerts fixed")
    df.loc[raion_in_hromada, "hromada"] = None
    df.loc[raion_in_hromada, "level"] = "raion"

    df = df.reset_index(drop=True)
    df.insert(0, "alert_id", df.index)
    return df


def attach_uids(alerts: pd.DataFrame, raions: pd.DataFrame) -> pd.DataFrame:
    """Add oblast_uid, raion_uid and hromada_uid; fail on any name that is not in regions.json."""
    out = alerts.copy()
    oblast_uids = dict(zip(raions["oblast"], raions["oblast_uid"]))
    out["oblast_uid"] = out["oblast"].map(oblast_uids)
    unknown = out["oblast_uid"].isna()
    if unknown.any():
        raise ValueError(f"{unknown.sum()} alerts name an oblast not in regions.json:\n"
                         f"{out[unknown]['oblast'].drop_duplicates()}")

    # Oblast alerts have no raion; they are spread to raions in the gold layer.
    raion_uids = dict(zip(zip(raions["oblast"], raions["raion"]), raions["raion_uid"]))
    below_oblast = out["level"] != "oblast"
    out["raion_uid"] = [raion_uids.get((o, r)) if b else None
                        for o, r, b in zip(out["oblast"], out["raion"], below_oblast)]
    unknown = below_oblast & out["raion_uid"].isna()
    if unknown.any():
        raise ValueError(f"{unknown.sum()} alerts match no raion in regions.json:\n"
                         f"{out[unknown][['oblast', 'raion']].drop_duplicates()}")

    # Hromada names repeat across raions (even within one oblast), so look them up by raion + name.
    hromada_uids = {(r.raion_uid, name): uid for r in raions.itertuples() for name, uid in r.hromadas.items()}
    out["hromada_uid"] = [hromada_uids.get((u, h)) for u, h in zip(out["raion_uid"], out["hromada"])]
    unknown = out["hromada"].notna() & out["hromada_uid"].isna()
    if unknown.any():
        raise ValueError(f"{unknown.sum()} hromada alerts name a hromada not in regions.json:\n"
                         f"{out[unknown][['raion', 'hromada']].drop_duplicates()}")

    return out[COLUMNS]


def build() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"{SOURCE} not found; run `python -m pipeline download` first")

    print(f"build {OUTPUT.name}")
    alerts = attach_uids(clean(pd.read_csv(SOURCE)), load_raions())

    SILVER.mkdir(parents=True, exist_ok=True)
    alerts.to_parquet(OUTPUT, index=False)
    print(json.dumps({
        "alerts": len(alerts),
        "alerts_by_level": alerts["level"].value_counts().to_dict(),
        "first_start": str(alerts["started_at"].min()),
        "last_finish": str(alerts["finished_at"].max()),
    }, indent=2))


if __name__ == "__main__":
    build()
