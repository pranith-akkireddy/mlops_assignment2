"""Fetch and preprocess the Cats vs Dogs dataset.

Resolution order for the source archive:
  1. --zip / DATA_ZIP explicit path
  2. an already-extracted data/raw/PetImages directory
  3. any kagglecatsanddogs*.zip sitting in data/raw/
  4. download from DATA_URL (used by CI, which caches the archive)
"""

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import preprocess  # noqa: E402

# Microsoft's public mirror of the Kaggle Cats vs Dogs corpus (no auth required).
DATA_URL = os.environ.get(
    "DATA_URL",
    "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip",
)


def resolve_archive(explicit=None):
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"Archive not found: {explicit}")
        return explicit
    env_zip = os.environ.get("DATA_ZIP")
    if env_zip and os.path.exists(env_zip):
        return env_zip
    if os.path.isdir(os.path.join(preprocess.RAW_DIR, "PetImages")):
        return None  # already extracted
    os.makedirs(preprocess.RAW_DIR, exist_ok=True)
    local = [
        os.path.join(preprocess.RAW_DIR, n)
        for n in sorted(os.listdir(preprocess.RAW_DIR))
        if n.lower().startswith("kagglecatsanddogs") and n.lower().endswith(".zip")
    ]
    if local:
        return local[0]
    return download_archive()


def download_archive():
    os.makedirs(preprocess.RAW_DIR, exist_ok=True)
    dest = os.path.join(preprocess.RAW_DIR, "kagglecatsanddogs_5340.zip")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"Using cached archive {dest}")
        return dest
    print(f"Downloading dataset from {DATA_URL} ...")
    tmp = dest + ".part"
    with urllib.request.urlopen(DATA_URL, timeout=120) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh, length=1024 * 1024)
    os.replace(tmp, dest)
    print(f"Downloaded to {dest}")
    return dest


def extract(archive):
    """Extract PetImages/ into data/raw, skipping the archive's stray files."""
    target = os.path.join(preprocess.RAW_DIR, "PetImages")
    if os.path.isdir(target) and any(os.scandir(target)):
        print(f"Reusing extracted images at {target}")
        return target
    print(f"Extracting {archive} ...")
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.namelist() if "PetImages/" in m and not m.endswith("/")]
        if not members:
            raise RuntimeError(f"No PetImages/ entries inside {archive}")
        zf.extractall(preprocess.RAW_DIR, members=members)
    # Some mirrors nest PetImages one level deeper.
    if not os.path.isdir(target):
        for root, dirs, _ in os.walk(preprocess.RAW_DIR):
            if "PetImages" in dirs:
                os.replace(os.path.join(root, "PetImages"), target)
                break
    print(f"Extracted to {target}")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", default=None)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=int(os.environ.get("MAX_PER_CLASS", "3000")),
        help="Cap images per class to keep the linear baseline tractable.",
    )
    parser.add_argument("--augment-copies", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    archive = resolve_archive(args.zip_path)
    root = extract(archive) if archive else os.path.join(preprocess.RAW_DIR, "PetImages")

    paths, labels = preprocess.discover_images(
        root=root, max_per_class=args.max_per_class, seed=args.seed
    )
    print(f"Found {len(paths)} readable images "
          f"({int((labels == 0).sum())} cat / {int((labels == 1).sum())} dog)")

    data = preprocess.preprocess_data(
        paths,
        labels,
        seed=args.seed,
        augment_copies=args.augment_copies,
        persist=True,
    )
    print(
        "Split sizes -> train {} (incl. {}x augmentation) | val {} | test {}".format(
            data["X_train"].shape[0],
            args.augment_copies,
            data["X_val"].shape[0],
            data["X_test"].shape[0],
        )
    )
    print(f"Feature dimension: {data['X_train'].shape[1]}")
    print("Data fetched and preprocessed successfully.")


if __name__ == "__main__":
    main()
