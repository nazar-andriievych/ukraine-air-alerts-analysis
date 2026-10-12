"""Run pipeline steps in order: python -m pipeline [download|silver|all] [--force]"""
import argparse

from pipeline import download, silver

STEPS = {"download": download.download, "silver": silver.build}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("step", choices=[*STEPS, "all"], nargs="?", default="all")
parser.add_argument("--force", action="store_true", help="redo steps even if inputs are unchanged")
args = parser.parse_args()

for name, run in STEPS.items():
    if args.step in (name, "all"):
        run(args.force)
