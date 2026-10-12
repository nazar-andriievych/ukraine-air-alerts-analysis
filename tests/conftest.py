import pandas as pd
import pytest

from pipeline.regions import load_raions

# A tiny hierarchy: one oblast with two raions that share a hromada name, Kyiv City without
# raions, and an excluded oblast.
REGIONS = {"oblasts": [
    {"uid": "1", "name": "Alpha oblast", "raions": [
        {"uid": "11", "name": "North raion", "hromadas": [{"uid": "111", "name": "Lake hromada"}]},
        {"uid": "12", "name": "South raion", "hromadas": [{"uid": "121", "name": "Lake hromada"}]},
    ]},
    {"uid": "2", "name": "Kyiv City", "raions": []},
    {"uid": "3", "name": "Luhanska oblast", "raions": [{"uid": "31", "name": "East raion", "hromadas": []}]},
]}


@pytest.fixture
def raions() -> pd.DataFrame:
    return load_raions(REGIONS)

