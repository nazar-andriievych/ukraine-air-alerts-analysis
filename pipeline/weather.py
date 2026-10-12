"""Download hourly weather per oblast from Open-Meteo and build the silver weather tables.

Usage: python -m pipeline.weather [--force]

Input:  data/reference/oblast_centers.json (one point per oblast: its administrative centre)
Output: data/bronze/weather/<source>_<start>_<end>.json   raw API responses, cached
        data/silver/weather_actual.parquet     ERA5 reanalysis (what the weather was)
        data/silver/weather_forecast.parquet   day-ahead forecasts issued at least 48 h before the hour

Both tables have one row per (hour, oblast_uid). `hour` is the start of the UTC hour the values
describe: Open-Meteo stamps an hour by its end (precipitation and gusts are totals / maxima over
the preceding hour), so its timestamp minus 1 h is our hour. Raions use their oblast's weather.

Sources (free, no key, hourly, timezone=UTC):
- actual:   archive-api.open-meteo.com/v1/archive (ERA5). Complete for every variable used here.
- forecast: previous-runs-api.open-meteo.com/v1/forecast with the `_previous_day2` suffix: the value
            from a model run started at least 48 h before the hour. Wind, gusts, precipitation and
            cloud cover exist only from 2024-01-20 12:00 UTC; low cloud is not available, so total
            cloud cover is used. See docs/PROBLEM.md, "Weather experiment", for why day2 and not day1
            or the Historical Forecast API.
Downloads are skipped when the cached file exists (unless --force).
"""
import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from pipeline.config import BRONZE, DATA_END, DATA_START, REFERENCE, SILVER

CENTERS = REFERENCE / "oblast_centers.json"
CACHE = BRONZE / "weather"
HOUR = pd.Timedelta(hours=1)
VARIABLES = ["temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m"]
SOURCES = {
    "actual": {"url": "https://archive-api.open-meteo.com/v1/archive",
               "variables": [*VARIABLES, "cloud_cover_low"], "suffix": "", "start": "2022-03-16"},
    "forecast": {"url": "https://previous-runs-api.open-meteo.com/v1/forecast",
                 "variables": [*VARIABLES, "cloud_cover"], "suffix": "_previous_day2", "start": "2023-12-01"},
}
# The last hour we need is 2026-09-06 23:00, which Open-Meteo stamps 2026-09-07 00:00.
LAST_DAY = "2026-09-07"
CHUNK_MONTHS = 6
PAUSE = 20  # seconds between requests, to stay under the per-minute limit
OUTPUTS = {name: SILVER / f"weather_{name}.parquet" for name in SOURCES}


def load_centers() -> pd.DataFrame:
    if not CENTERS.exists():
        raise SystemExit(f"{CENTERS} not found; run `python scripts/oblast_centers.py` first")
    return pd.DataFrame(json.loads(CENTERS.read_text(encoding="utf-8")))


def points(centers: pd.DataFrame) -> pd.DataFrame:
    """Unique weather points (Kyivska oblast and Kyiv City share Kyiv)."""
    return centers.drop_duplicates(["lat", "lon"])[["name", "lat", "lon"]].reset_index(drop=True)


def chunks(start: str, end: str, months: int = CHUNK_MONTHS) -> list[tuple[str, str]]:
    """[start, end] split into date ranges of at most `months` months (inclusive end dates)."""
    edges = pd.date_range(start, end, freq=f"{months}MS").union([pd.Timestamp(start)])
    starts = list(edges) if edges[-1] <= pd.Timestamp(end) else list(edges[:-1])
    ends = [s - pd.Timedelta(days=1) for s in starts[1:]] + [pd.Timestamp(end)]
    return [(s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")) for s, e in zip(starts, ends)]


def fetch(url: str, params: dict) -> bytes:
    """GET with retries; waits a minute on rate limiting."""
    full = f"{url}?{urllib.parse.urlencode(params)}"
    for attempt in range(6):
        try:
            with urllib.request.urlopen(full, timeout=300) as response:
                body = response.read()
            json.loads(body)  # fail early on a truncated body
            return body
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504):
                raise SystemExit(f"{error.code} from {url}: {error.read()[:300]!r}")
            wait = 65 if error.code == 429 else 10 * (attempt + 1)
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, ConnectionError):
            wait = 10 * (attempt + 1)
        print(f"  retry in {wait} s")
        time.sleep(wait)
    raise SystemExit(f"download failed: {full}")


def download(force: bool = False) -> None:
    where = points(load_centers())
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, source in SOURCES.items():
        for start, end in chunks(source["start"], LAST_DAY):
            path = CACHE / f"{name}_{start}_{end}.json"
            if path.exists() and not force:
                continue
            print(f"download {path.name}")
            path.write_bytes(fetch(source["url"], {
                "latitude": ",".join(map(str, where["lat"])), "longitude": ",".join(map(str, where["lon"])),
                "hourly": ",".join(v + source["suffix"] for v in source["variables"]),
                "timezone": "UTC", "wind_speed_unit": "ms", "start_date": start, "end_date": end,
            }))
            time.sleep(PAUSE)


def response_frame(hourly: dict, variables: list[str], suffix: str) -> pd.DataFrame:
    """One location's `hourly` block as rows per hour start; None values become NaN."""
    frame = pd.DataFrame({v: np.array(hourly[v + suffix], dtype=float) for v in variables})
    frame.insert(0, "hour", pd.to_datetime(hourly["time"]).tz_localize("UTC") - HOUR)
    return frame


def parse(name: str, centers: pd.DataFrame) -> pd.DataFrame:
    """All cached responses of a source as one row per (hour, oblast_uid)."""
    source, where = SOURCES[name], points(centers)
    parts = []
    for start, end in chunks(source["start"], LAST_DAY):
        responses = json.loads((CACHE / f"{name}_{start}_{end}.json").read_bytes())
        responses = responses if isinstance(responses, list) else [responses]
        if len(responses) != len(where):
            raise SystemExit(f"{name} {start}: {len(responses)} locations, expected {len(where)}")
        for point, response in zip(where.itertuples(), responses):
            frame = response_frame(response["hourly"], source["variables"], source["suffix"])
            frame.insert(1, "name", point.name)
            parts.append(frame)
    table = pd.concat(parts, ignore_index=True)
    table = table[(table["hour"] >= pd.Timestamp(DATA_START)) & (table["hour"] < pd.Timestamp(DATA_END))]
    table = centers[["oblast_uid", "name"]].merge(table, on="name").drop(columns="name")
    return table.sort_values(["hour", "oblast_uid"]).reset_index(drop=True).astype(
        {v: "float32" for v in source["variables"]})


def coverage(table: pd.DataFrame) -> dict:
    """First hour with a value, and the share of hours from then on that have one, per variable."""
    report = {}
    for column in table.columns.drop(["hour", "oblast_uid"]):
        present = table.loc[table[column].notna(), "hour"]
        after = table[table["hour"] >= present.min()] if len(present) else table.iloc[:0]
        report[column] = {"first": str(present.min()) if len(present) else None,
                          "share_present_after_first": round(float(after[column].notna().mean()), 5)
                          if len(after) else 0.0}
    return report


def build(force: bool = False) -> None:
    download(force)
    centers = load_centers()
    SILVER.mkdir(parents=True, exist_ok=True)
    for name, path in OUTPUTS.items():
        table = parse(name, centers)
        table.to_parquet(path, index=False)
        print(f"build {path.name}: {len(table)} rows, {table['oblast_uid'].nunique()} oblasts")
        print(json.dumps(coverage(table), indent=2))


def matrices(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Each weather variable as an hour x oblast_uid matrix (rows are hour starts)."""
    columns = table.columns.drop(["hour", "oblast_uid"])
    return {c: table.pivot(index="hour", columns="oblast_uid", values=c) for c in columns}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download weather and build the silver weather tables.")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    build(parser.parse_args().force)
