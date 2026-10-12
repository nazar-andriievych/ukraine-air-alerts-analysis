"""Download the raw official alerts dataset into data/bronze/ at a pinned commit.

Usage: python -m pipeline.download [--force]

Files are stored unchanged. A manifest records the commit and SHA-256 of each file,
so re-running skips files that are already present and intact.
"""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from pipeline.config import BRONZE, DATASET_COMMIT, DATASET_FILES, DATASET_REPO

MANIFEST = BRONZE / "manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"repo": DATASET_REPO, "commit": DATASET_COMMIT, "files": {}}


def download(force: bool = False) -> None:
    BRONZE.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    if manifest.get("commit") != DATASET_COMMIT:
        manifest = {"repo": DATASET_REPO, "commit": DATASET_COMMIT, "files": {}}

    for remote in DATASET_FILES:
        local = BRONZE / Path(remote).name
        recorded = manifest["files"].get(local.name)
        if not force and local.exists() and recorded == sha256(local):
            print(f"skip  {local.name} (already at {DATASET_COMMIT[:7]})")
            continue

        url = f"https://raw.githubusercontent.com/{DATASET_REPO}/{DATASET_COMMIT}/{remote}"
        print(f"fetch {url}")
        tmp = local.with_suffix(local.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(local)
        manifest["files"][local.name] = sha256(local)

    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    download(parser.parse_args().force)
