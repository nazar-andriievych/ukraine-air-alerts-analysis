"""Paths and constants shared by pipeline steps."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BRONZE = DATA / "bronze"
REFERENCE = DATA / "reference"

# Public dataset: https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset
# Pinned to a fixed commit so every run sees exactly the same data.
DATASET_REPO = "Vadimkin/ukrainian-air-raid-sirens-dataset"
DATASET_COMMIT = "d10c80fa696bab42822b9f05b423fc8a30e79d20"
DATASET_FILES = ["datasets/official_data_en.csv"]

SILVER = DATA / "silver"
REGIONS = REFERENCE / "regions.json"

# Permanent alerts there, so they are not forecast.
EXCLUDED_OBLASTS = {"Luhanska oblast", "Avtonomna Respublika Krym"}

# All times are UTC. The last complete day is 2026-09-06; alerts starting at or after this are dropped.
DATA_END = "2026-09-07T00:00:00Z"
