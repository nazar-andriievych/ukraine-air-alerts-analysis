"""Build the silver layer: clean alerts, one row per affected raion.

Usage: python -m pipeline.silver [--force]

Input:  data/bronze/official_data_en.csv, data/reference/regions.json
Output: data/silver/raion_alerts.parquet

Each source alert becomes one row per raion it covers:
  - oblast alert  -> every raion of the oblast (Kyiv City has no raions and is a single unit)
  - raion alert   -> that raion
  - hromada alert -> its whole raion
`level` and `hromada` record where each row came from. Timestamps are UTC.
Overlapping rows are kept as they are; merging them is the job of the next layer.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from pipeline.config import BRONZE, DATA_END, EXCLUDED_OBLASTS, REGIONS, SILVER

SOURCE = BRONZE / "official_data_en.csv"
OUTPUT = SILVER / "raion_alerts.parquet"
MANIFEST = SILVER / "manifest.json"
COLUMNS = ["alert_id", "oblast", "oblast_uid", "raion", "raion_uid", "level", "hromada",
           "hromada_uid", "started_at", "finished_at"]


def inputs_hash() -> str:
    """Hash of everything the output depends on, so unchanged inputs can be skipped."""
    h = hashlib.sha256()
    for path in (SOURCE, REGIONS, Path(__file__), Path(__file__).with_name("config.py")):
        h.update(path.read_bytes())
    return h.hexdigest()


def load_raions() -> pd.DataFrame:
    """One row per raion we forecast for: oblast, oblast_uid, raion, raion_uid, hromadas (name -> uid)."""
    regions = json.loads(REGIONS.read_text(encoding="utf-8"))
    rows = []
    for oblast in regions["oblasts"]:
        if oblast["name"] in EXCLUDED_OBLASTS:
            continue
        # An oblast without raions (Kyiv City) is treated as a single raion of itself.
        raions = oblast["raions"] or [{"uid": oblast["uid"], "name": oblast["name"], "hromadas": []}]
        for raion in raions:
            rows.append({
                "oblast": oblast["name"], "oblast_uid": oblast["uid"],
                "raion": raion["name"], "raion_uid": raion["uid"],
                "hromadas": {h["name"]: h["uid"] for h in raion["hromadas"]},
            })
    return pd.DataFrame(rows)


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

    late = df["started_at"] >= pd.Timestamp(DATA_END)
    print(f"  {late.sum():>7} alerts starting at or after {DATA_END} removed")
    df = df[~late].copy()

    # A few rows have the raion name in the hromada field: they are raion alerts.
    raion_in_hromada = df["hromada"].notna() & (df["hromada"] == df["raion"])
    print(f"  {raion_in_hromada.sum():>7} 'hromada' rows that are really raion alerts fixed")
    df.loc[raion_in_hromada, "hromada"] = None
    df.loc[raion_in_hromada, "level"] = "raion"

    df = df.reset_index(drop=True)
    df.insert(0, "alert_id", df.index)
    return df


def expand_to_raions(alerts: pd.DataFrame, raions: pd.DataFrame) -> pd.DataFrame:
    keys = raions.drop(columns="hromadas")
    whole_oblast = alerts["level"] == "oblast"
    by_oblast = alerts[whole_oblast].drop(columns="raion").merge(keys, on="oblast", how="left")
    by_raion = alerts[~whole_oblast].merge(keys, on=["oblast", "raion"], how="left")
    out = pd.concat([by_oblast, by_raion], ignore_index=True)

    unmatched = out[out["raion_uid"].isna()]
    if len(unmatched):
        raise ValueError(f"{len(unmatched)} alerts match no raion in regions.json:\n"
                         f"{unmatched[['oblast', 'raion']].drop_duplicates()}")

    # Hromada names repeat across raions (even within one oblast), so look them up by raion + name.
    hromada_uids = {(r.raion_uid, name): uid for r in raions.itertuples() for name, uid in r.hromadas.items()}
    out["hromada_uid"] = [hromada_uids.get((u, h)) for u, h in zip(out["raion_uid"], out["hromada"])]
    unknown = out["hromada"].notna() & out["hromada_uid"].isna()
    if unknown.any():
        raise ValueError(f"{unknown.sum()} hromada alerts name a hromada not in regions.json:\n"
                         f"{out[unknown][['raion', 'hromada']].drop_duplicates()}")

    return out[COLUMNS].sort_values(["raion_uid", "started_at", "alert_id"]).reset_index(drop=True)


def build(force: bool = False) -> None:
    current_hash = inputs_hash()
    if not force and OUTPUT.exists() and MANIFEST.exists():
        if json.loads(MANIFEST.read_text(encoding="utf-8")).get("inputs_hash") == current_hash:
            print(f"skip  {OUTPUT.name} (inputs unchanged)")
            return

    print(f"build {OUTPUT.name}")
    alerts = clean(pd.read_csv(SOURCE))
    raions = load_raions()
    silver = expand_to_raions(alerts, raions)

    SILVER.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(OUTPUT, index=False)
    summary = {
        "source_alerts": len(alerts),
        "rows": len(silver),
        "raions_with_alerts": int(silver["raion_uid"].nunique()),
        "raions_in_regions_json": len(raions),
        "rows_by_level": silver["level"].value_counts().to_dict(),
        "first_start": str(silver["started_at"].min()),
        "last_start": str(silver["started_at"].max()),
    }
    print(json.dumps(summary, indent=2))
    MANIFEST.write_text(json.dumps({"inputs_hash": current_hash, **summary}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild even if inputs are unchanged")
    build(parser.parse_args().force)
