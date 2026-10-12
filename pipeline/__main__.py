"""Run pipeline steps in order: python -m pipeline [download|silver|gold|features|baselines|models|final|all] [--force]"""
import argparse
from functools import partial

from pipeline import baselines, download, features, final, gold, models, silver

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("step", choices=["download", "silver", "gold", "features", "baselines", "models", "final", "all"], nargs="?", default="all")
parser.add_argument("--force", action="store_true", help="re-download the dataset even if it is present")
args = parser.parse_args()

# Only the download is skipped when already done; the other steps always rebuild.
# Baselines are scored on validation here; `python -m pipeline.baselines test` scores the test split.
STEPS = {"download": partial(download.download, args.force), "silver": silver.build, "gold": gold.build,
         "features": features.build, "baselines": baselines.run, "models": models.run,
         "final": final.run}

for name, run in STEPS.items():
    if args.step in (name, "all"):
        run()
