"""Run pipeline steps in order: python -m pipeline [download|silver|gold|all] [--force]"""
import argparse
from functools import partial

from pipeline import download, gold, silver

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("step", choices=["download", "silver", "gold", "all"], nargs="?", default="all")
parser.add_argument("--force", action="store_true", help="re-download the dataset even if it is present")
args = parser.parse_args()

# Only the download is skipped when already done; silver and gold take seconds and always rebuild.
STEPS = {"download": partial(download.download, args.force), "silver": silver.build, "gold": gold.build}

for name, run in STEPS.items():
    if args.step in (name, "all"):
        run()
