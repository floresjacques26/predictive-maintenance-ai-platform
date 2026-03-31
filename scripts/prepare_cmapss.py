"""Prepare and validate the NASA CMAPSS dataset.

Downloads the dataset (if not already present), validates file integrity,
and prints a summary of each sub-dataset.

Usage
-----
# Check / download data automatically:
python scripts/prepare_cmapss.py

# Only validate already-downloaded files:
python scripts/prepare_cmapss.py --no-download

# Show summary for a specific subset:
python scripts/prepare_cmapss.py --subset FD002

Download source
---------------
https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cmapss_loader import CMAPSSLoader, VALID_SUBSETS
from src.utils.logger import get_logger

logger = get_logger("prepare_cmapss")

DOWNLOAD_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
DEFAULT_DATA_DIR = Path("data/cmapss")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare NASA CMAPSS dataset.")
    p.add_argument(
        "--data-dir", default=str(DEFAULT_DATA_DIR),
        help="Directory to store CMAPSS files.",
    )
    p.add_argument(
        "--subset", choices=list(VALID_SUBSETS) + ["all"], default="all",
        help="Sub-dataset to validate / summarise.",
    )
    p.add_argument(
        "--no-download", action="store_true",
        help="Skip download attempt; only validate existing files.",
    )
    p.add_argument(
        "--failure-horizon", type=int, default=30,
        help="Failure horizon used when computing positive-rate statistics.",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Download
# ──────────────────────────────────────────────────────────────────────────────

def _files_present(data_dir: Path, subset: str) -> bool:
    return (data_dir / f"train_{subset}.txt").exists()


def _all_files_present(data_dir: Path, subsets: list[str]) -> bool:
    return all(_files_present(data_dir, s) for s in subsets)


def download_cmapss(data_dir: Path) -> None:
    """Download and extract the CMAPSS zip archive."""
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading CMAPSS dataset from:\n  {DOWNLOAD_URL}")

    try:
        # Stream download with progress
        response = urllib.request.urlopen(DOWNLOAD_URL, timeout=120)
        total = int(response.headers.get("Content-Length", 0))
        chunk_size = 1 << 16  # 64 KB
        downloaded = 0
        buf = io.BytesIO()

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            buf.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = 100 * downloaded / total
                print(f"\r  {downloaded / 1e6:.1f} MB / {total / 1e6:.1f} MB  ({pct:.0f}%)", end="", flush=True)

        print()
        buf.seek(0)
        logger.info("Download complete. Extracting…")

        with zipfile.ZipFile(buf) as zf:
            for member in zf.namelist():
                # Only extract .txt files, stripping any subdirectory prefix
                if member.lower().endswith(".txt"):
                    filename = Path(member).name
                    dest = data_dir / filename
                    dest.write_bytes(zf.read(member))
                    logger.info(f"  Extracted → {dest}")

        logger.info(f"CMAPSS dataset ready in {data_dir.resolve()}")

    except Exception as exc:
        logger.error(f"Download failed: {exc}")
        logger.error(
            "\nManual download:\n"
            f"  1. Visit {DOWNLOAD_URL}\n"
            f"  2. Unzip and place .txt files in: {data_dir.resolve()}\n"
            f"  3. Re-run this script to validate."
        )
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Validation & summary
# ──────────────────────────────────────────────────────────────────────────────

def validate_and_summarise(
    data_dir: Path,
    subsets: list[str],
    failure_horizon: int,
) -> None:
    all_ok = True
    for subset in subsets:
        print(f"\n{'='*50}")
        loader = CMAPSSLoader(
            data_dir=data_dir,
            subset=subset,
            failure_horizon=failure_horizon,
        )
        present = loader.files_present()
        for fname, ok in present.items():
            status = "✓" if ok else "✗ MISSING"
            print(f"  {status}  {fname}")

        if not present.get(f"train_{subset}.txt"):
            print(f"  [SKIP] Cannot load {subset} — train file missing.")
            all_ok = False
            continue

        try:
            print(loader.summary("train"))
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            all_ok = False

    print()
    if all_ok:
        logger.info("All requested CMAPSS files validated successfully.")
    else:
        logger.warning("Some files are missing or could not be loaded.")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    subsets = list(VALID_SUBSETS) if args.subset == "all" else [args.subset]

    # Check if any train files are missing
    missing = [s for s in subsets if not _files_present(data_dir, s)]

    if missing and not args.no_download:
        logger.info(f"Missing train files for: {missing}")
        download_cmapss(data_dir)
    elif missing:
        logger.warning(
            f"Files for {missing} are missing and --no-download was set.\n"
            f"Place .txt files in: {data_dir.resolve()}"
        )

    validate_and_summarise(data_dir, subsets, args.failure_horizon)


if __name__ == "__main__":
    main()
