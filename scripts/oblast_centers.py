"""Look up the administrative centre of each oblast with the Open-Meteo geocoding API.

Usage: python scripts/oblast_centers.py

Input:  data/reference/regions.json (oblast uids and names)
Output: data/reference/oblast_centers.json  [{oblast_uid, oblast, name, lat, lon, geonames_id}]

Each oblast gets one weather point: its administrative centre. Donetska oblast uses Kramatorsk,
its temporary administrative centre since 2014 (Donetsk is occupied). Kyivska oblast and Kyiv
City share Kyiv, where the oblast administration sits. Names are spelled as in GeoNames
(Zaporizhzhya, Mykolayiv), which the geocoder searches. The script stops if a name does not
resolve to a regional or national capital (feature code PPLA / PPLC) in Ukraine, except
Kramatorsk, which geocodes as a raion centre (PPLA2).
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGIONS = ROOT / "data" / "reference" / "regions.json"
OUTPUT = ROOT / "data" / "reference" / "oblast_centers.json"
URL = "https://geocoding-api.open-meteo.com/v1/search"

CENTERS = {
    "Vinnytska oblast": "Vinnytsia", "Volynska oblast": "Lutsk", "Dnipropetrovska oblast": "Dnipro",
    "Donetska oblast": "Kramatorsk", "Zhytomyrska oblast": "Zhytomyr", "Zakarpatska oblast": "Uzhhorod",
    "Zaporizka oblast": "Zaporizhzhya", "Ivano-Frankivska oblast": "Ivano-Frankivsk",
    "Kyivska oblast": "Kyiv", "Kirovohradska oblast": "Kropyvnytskyi", "Lvivska oblast": "Lviv",
    "Kyiv City": "Kyiv", "Mykolaivska oblast": "Mykolayiv", "Odeska oblast": "Odesa",
    "Poltavska oblast": "Poltava", "Rivnenska oblast": "Rivne", "Sumska oblast": "Sumy",
    "Ternopilska oblast": "Ternopil", "Kharkivska oblast": "Kharkiv", "Khersonska oblast": "Kherson",
    "Khmelnytska oblast": "Khmelnytskyi", "Cherkaska oblast": "Cherkasy",
    "Chernivetska oblast": "Chernivtsi", "Chernihivska oblast": "Chernihiv",
}
NOT_A_CAPITAL = {"Kramatorsk"}


def lookup(name: str) -> dict:
    query = urllib.parse.urlencode({"name": name, "country": "UA", "count": 10, "language": "en"})
    for attempt in range(8):  # the API sometimes returns an empty body; retry
        try:
            with urllib.request.urlopen(f"{URL}?{query}", timeout=30) as response:
                results = json.load(response).get("results", [])
            break
        except (json.JSONDecodeError, OSError):
            time.sleep(min(2 ** attempt, 30))
    else:
        sys.exit(f"{name}: geocoding failed")
    allowed = {"PPLA", "PPLC"} | ({"PPLA2"} if name in NOT_A_CAPITAL else set())
    matches = [r for r in results if r.get("country_code") == "UA" and r.get("feature_code") in allowed]
    if not matches:
        sys.exit(f"{name}: no Ukrainian regional centre found in {results}")
    best = max(matches, key=lambda r: r.get("population") or 0)
    return {"name": best["name"], "lat": best["latitude"], "lon": best["longitude"],
            "admin1": best.get("admin1"), "geonames_id": best["id"]}


def main() -> None:
    oblasts = json.loads(REGIONS.read_text(encoding="utf-8"))["oblasts"]
    rows = []
    for oblast in oblasts:
        if oblast["name"] not in CENTERS:
            continue  # excluded oblasts (Luhansk, Crimea)
        found = lookup(CENTERS[oblast["name"]])
        print(f"{oblast['name']:26} -> {found['name']} ({found['admin1']}) {found['lat']}, {found['lon']}")
        rows.append({"oblast_uid": oblast["uid"], "oblast": oblast["name"], "name": found["name"],
                     "lat": found["lat"], "lon": found["lon"], "geonames_id": found["geonames_id"]})
        time.sleep(1)
    if len(rows) != len(CENTERS):
        sys.exit(f"expected {len(CENTERS)} oblasts, found {len(rows)}")
    OUTPUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
