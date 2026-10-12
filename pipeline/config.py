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
GOLD = DATA / "gold"
FEATURES = DATA / "features"
REPORTS = ROOT / "reports"
REGIONS = REFERENCE / "regions.json"

# Permanent alerts there, so they are not forecast.
EXCLUDED_OBLASTS = {"Luhanska oblast", "Avtonomna Respublika Krym"}

# All times are UTC. The dataset starts at 2022-03-15 16:10, so the first complete day is 2022-03-16.
DATA_START = "2022-03-16T00:00:00Z"
# The last complete day is 2026-09-06. Alerts starting at or after this are dropped,
# and alerts still running at this moment are cut off here.
DATA_END = "2026-09-07T00:00:00Z"

# Splits by time (UTC), as [start, end). An origin belongs to a split when all 24 hours it
# forecasts fall inside it. The test period is the last 8 weeks, validation the 8 weeks before.
TRAIN_END = "2026-05-18T00:00:00Z"
VALIDATION_END = "2026-07-13T00:00:00Z"
SPLITS = {
    "train": (DATA_START, TRAIN_END),
    "validation": (TRAIN_END, VALIDATION_END),
    "test": (VALIDATION_END, DATA_END),
}
