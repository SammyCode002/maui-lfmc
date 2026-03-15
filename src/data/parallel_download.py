"""
parallel_download.py - Parallel GEE TIF downloader for Globe-LFMC samples.

Replaces the sequential download in download_tifs.py with a thread pool.
Uses the same download_tif_for_sample function — safe to run alongside or
after a previous sequential run (already-downloaded files are skipped).

GEE allows ~20 concurrent requests per project. Default is 10 workers to
stay comfortably under the limit. Increase with --workers if your quota allows.

Usage:
    # Use pre-built labels CSV (fastest — skip Excel processing)
    python -m src.data.parallel_download \
        --project ace-shine-392702 \
        --labels C:/Users/LazyB/Downloads/allenai-lfmc/data/labels/lfmc_data_conus.csv \
        --output data/tifs/ \
        --workers 10

    # Limit to a test batch
    python -m src.data.parallel_download \
        --project ace-shine-392702 \
        --labels C:/path/to/lfmc_data_conus.csv \
        --output data/tifs/ \
        --workers 10 \
        --limit 500
"""

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import pandas as pd
from tqdm import tqdm

from src.data.download_tifs import download_tif_for_sample, pad_dates
from src.data.globe_lfmc_filter import REGION_BOUNDS

logger = logging.getLogger(__name__)


def build_sample_list(labels_csv: Path, limit: int | None = None) -> list[dict]:
    """
    Load pre-built labels CSV, deduplicate by lat/lon/date,
    and return list of sample dicts ready for downloading.
    """
    df = pd.read_csv(labels_csv)
    df["sampling_date"] = pd.to_datetime(df["sampling_date"])

    grouped = df.groupby(
        ["latitude", "longitude", "sampling_date"], as_index=False
    ).agg({"sorting_id": "first", "lfmc_value": "mean"})

    if limit is not None:
        grouped = grouped.head(limit)

    samples = grouped.to_dict("records")
    logger.info("Total unique samples to download: %d", len(samples))
    return samples


def download_worker(sample: dict, output_dir: Path, project: str) -> tuple[int, str]:
    """
    Worker function called by each thread.
    Returns (sorting_id, status) where status is 'downloaded', 'skipped', or 'failed'.
    """
    sorting_id = int(sample["sorting_id"])
    output_path = output_dir / f"{sorting_id}.tif"

    if output_path.exists():
        return sorting_id, "skipped"

    sampling_date = pd.to_datetime(sample["sampling_date"]).date()
    start_date, end_date = pad_dates(sampling_date)

    success = download_tif_for_sample(
        sorting_id=sorting_id,
        lat=float(sample["latitude"]),
        lon=float(sample["longitude"]),
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        project=project,
    )

    return sorting_id, "downloaded" if success else "failed"


def run_parallel_download(
    project: str,
    output_dir: Path,
    labels_csv: Path,
    workers: int = 10,
    limit: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing Google Earth Engine...")
    ee.Initialize(project=project)

    already = len(list(output_dir.glob("*.tif")))
    logger.info("TIFs already on disk: %d", already)

    samples = build_sample_list(labels_csv, limit)
    total = len(samples)

    downloaded = 0
    skipped = 0
    failed = 0
    start_time = time.time()

    print(f"\nStarting parallel download: {total} samples, {workers} workers\n")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_worker, s, output_dir, project): s
            for s in samples
        }

        with tqdm(total=total, unit="tif", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                try:
                    sorting_id, status = future.result()
                    if status == "downloaded":
                        downloaded += 1
                    elif status == "skipped":
                        skipped += 1
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1
                    logger.warning("Worker exception: %s", e)

                elapsed = time.time() - start_time
                done = downloaded + skipped + failed
                rate = done / (elapsed / 3600) if elapsed > 0 else 0
                remaining = total - done
                eta_h = remaining / rate if rate > 0 else 0

                pbar.set_postfix(
                    downloaded=downloaded,
                    skipped=skipped,
                    failed=failed,
                    rate=f"{rate:.0f}/hr",
                    eta=f"{eta_h:.1f}h",
                )
                pbar.update(1)

    elapsed_h = (time.time() - start_time) / 3600
    print(f"\nDone in {elapsed_h:.1f}h")
    print(f"  Downloaded : {downloaded}")
    print(f"  Skipped    : {skipped} (already existed)")
    print(f"  Failed     : {failed}")
    print(f"  Total TIFs : {len(list(output_dir.glob('*.tif')))}")


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(description="Parallel GEE TIF downloader")
    parser.add_argument("--project", required=True, help="GEE project ID")
    parser.add_argument(
        "--labels", type=Path, required=True,
        help="Path to labels CSV (e.g. allenai-lfmc/data/labels/lfmc_data_conus.csv)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/tifs/"),
        help="Output directory (default: data/tifs/)"
    )
    parser.add_argument(
        "--workers", type=int, default=10,
        help="Number of parallel download threads (default: 10, max ~20 for GEE)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap number of samples (useful for testing)"
    )
    args = parser.parse_args()

    run_parallel_download(
        project=args.project,
        output_dir=args.output,
        labels_csv=args.labels,
        workers=args.workers,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
