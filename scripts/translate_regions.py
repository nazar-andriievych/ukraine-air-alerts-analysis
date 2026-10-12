"""Convert data/reference/regions_uk.json (Ukrainian names) to data/reference/regions.json (English).

Names are transliterated with the official Ukrainian national system (KMU Resolution No. 55, 2010),
which is the same system the English alerts dataset uses. A few names are overridden by hand.
If data/bronze/official_data_en.csv exists, every transliterated name is checked against it.

Usage: python scripts/translate_regions.py
"""
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "reference" / "regions_uk.json"
DST = ROOT / "data" / "reference" / "regions.json"
DATASET = ROOT / "data" / "bronze" / "official_data_en.csv"

# Letters that are written differently at the start of a word.
WORD_START = {"є": "ye", "ї": "yi", "й": "y", "ю": "yu", "я": "ya"}
LETTERS = {
    "а": "a", "б": "b", "в": "v", "г": "h", "ґ": "g", "д": "d", "е": "e", "є": "ie", "ж": "zh",
    "з": "z", "и": "y", "і": "i", "ї": "i", "й": "i", "к": "k", "л": "l", "м": "m", "н": "n",
    "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "shch", "ю": "iu", "я": "ia", "ь": "", "'": "", "’": "", "ʼ": "",
}

# Names whose English form in the dataset is not a plain transliteration.
OVERRIDES = {
    "м. Київ": "Kyiv City",
    # Renamed in 2024, but the dataset still uses the old names, so regions.json uses them too.
    "Володимирський район": "Volodymyr-Volynskyi raion",
    "Шептицький район": "Chervonohradskyi raion",
    "Берестинський район": "Krasnohradskyi raion",
    "м. Шахтарськ та Шахтарська територіальна громада": "m. Pershotravensk ta Pershotravenska terytorialna hromada",
}


def transliterate(text: str) -> str:
    if text in OVERRIDES:
        return OVERRIDES[text]
    text = text.replace("зг", "zgh").replace("Зг", "Zgh")
    out = []
    for i, ch in enumerate(text):
        low = ch.lower()
        word_start = i == 0 or not text[i - 1].isalpha() and text[i - 1] not in "'’ʼ"
        latin = (WORD_START if word_start and low in WORD_START else LETTERS).get(low, ch)
        out.append(latin.capitalize() if ch.isupper() else latin)
    return "".join(out)


def convert(node: dict) -> dict:
    result = {"uid": node["uid"], "name": transliterate(node["title"])}
    for child_key in ("raions", "hromadas"):
        if child_key in node:
            result[child_key] = [convert(c) for c in node[child_key]]
    return result


def check_against_dataset(oblasts: list) -> None:
    names = {"oblast": set(), "raion": set(), "hromada": set()}
    with DATASET.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for level in names:
                if row[level]:
                    names[level].add(row[level])

    ours = {
        "oblast": {o["name"] for o in oblasts},
        "raion": {r["name"] for o in oblasts for r in o["raions"]},
        "hromada": {h["name"] for o in oblasts for r in o["raions"] for h in r["hromadas"]},
    }
    for level in names:
        missing = sorted(names[level] - ours[level])
        print(f"{level}: {len(names[level]) - len(missing)}/{len(names[level])} dataset names found in regions.json")
        for name in missing:
            print(f"   not found: {name}")


def main() -> None:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    oblasts = [convert(o) for o in data["oblasts"]]
    out = {"generated_at": data["generated_at"], "source": SRC.name, "oblasts": oblasts}
    DST.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {DST.relative_to(ROOT)}")
    if DATASET.exists():
        check_against_dataset(oblasts)
    else:
        print("dataset not downloaded; run `python -m pipeline.download` to validate names", file=sys.stderr)


if __name__ == "__main__":
    main()
