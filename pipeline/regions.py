"""The oblast -> raion -> hromada hierarchy from data/reference/regions.json."""
import json

import pandas as pd

from pipeline.config import EXCLUDED_OBLASTS, REGIONS


def load_raions(regions: dict | None = None) -> pd.DataFrame:
    """One row per raion we forecast for: oblast, oblast_uid, raion, raion_uid, hromadas (name -> uid).

    `regions` is the parsed regions.json; it is read from disk when not given.
    """
    if regions is None:
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
