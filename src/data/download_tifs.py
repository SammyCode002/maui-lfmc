"""
Download satellite data from Google Earth Engine for Globe-LFMC sample locations.

Creates one GeoTIFF per sample, named {sorting_id}.tif, containing:
- 12 monthly composites of: S1 (VV,VH), S2 (10 bands), ERA5, TerraClimate, VIIRS
- Spatial bands: SRTM elevation, DynamicWorld, WorldCereal
- Static bands: LandScan population

This matches the exact format expected by Galileo's Dataset class and the
AllenAI LFMC training pipeline (github.com/allenai/lfmc).

Usage:
    # Download a small test batch (100 samples)
    python -m src.data.download_tifs --labels data/labels/lfmc_data_conus.csv \\
        --output data/tifs/ --project YOUR_GEE_PROJECT --limit 100

    # Download all CONUS samples (runs overnight, ~90k TIFs)
    python -m src.data.download_tifs --labels data/labels/lfmc_data_conus.csv \\
        --output data/tifs/ --project YOUR_GEE_PROJECT

    # Download for a custom region bounding box (e.g. Maui inference)
    python -m src.data.download_tifs --bbox 20.5,156.7,21.1,155.9 \\
        --start 2023-01-01 --end 2023-12-31 \\
        --output data/tifs/maui_2023/ --project YOUR_GEE_PROJECT
"""

import argparse
import logging
import shutil
from datetime import date, datetime
from pathlib import Path

import ee
import pandas as pd
import requests
from tqdm import tqdm

from galileo.data.earthengine.ee_bbox import EEBoundingBox
from galileo.data.earthengine.eo import create_ee_image

logger = logging.getLogger(__name__)

# 1km x 1km bounding box around each sample point (500m each side)
SURROUNDING_METRES = 500


def pad_dates(end_date: date) -> tuple[date, date]:
    """Compute 12-month window ending ~30 days after the sampling date."""
    import calendar
    from datetime import timedelta

    new_end = end_date + timedelta(days=30)
    last_day = calendar.monthrange(new_end.year, new_end.month)[1]
    new_end = date(new_end.year, new_end.month, last_day)
    start = date(new_end.year - 1, new_end.month, 1)
    return start, new_end


def download_tif_for_sample(
    sorting_id: int,
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
    output_dir: Path,
    project: str,
) -> bool:
    """
    Download one GeoTIFF for a single Globe-LFMC sample using GEE url mode.
    Returns True if downloaded successfully, False if skipped or failed.
    """
    output_path = output_dir / f"{sorting_id}.tif"
    if output_path.exists():
        return False  # Already downloaded

    ee_bbox = EEBoundingBox.from_centre(
        mid_lat=lat,
        mid_lon=lon,
        surrounding_metres=SURROUNDING_METRES,
    )

    try:
        img = create_ee_image(
            polygon=ee_bbox.to_ee_polygon(),
            start_date=start_date,
            end_date=end_date,
        )

        url = img.getDownloadURL({
            "region": ee_bbox.to_ee_polygon(),
            "scale": 10,
            "filePerBand": False,
            "format": "GEO_TIFF",
        })

        r = requests.get(url, stream=True, timeout=120)
        if r.status_code != 200:
            logger.warning(f"Failed to download {sorting_id}: HTTP {r.status_code}")
            return False

        with output_path.open("wb") as f:
            shutil.copyfileobj(r.raw, f)

        return True

    except ee.ee_exception.EEException as e:
        logger.warning(f"GEE error for sample {sorting_id}: {e}")
        return False
    except Exception as e:
        logger.warning(f"Unexpected error for sample {sorting_id}: {e}")
        return False


def download_for_labels(
    labels_csv: Path,
    output_dir: Path,
    project: str,
    limit: int | None = None,
) -> None:
    """
    Download GeoTIFFs for all samples in the Globe-LFMC labels CSV.

    Args:
        labels_csv: Path to lfmc_data_conus.csv
        output_dir: Directory to save TIF files
        project: GEE project ID (e.g. 'ace-shine-392702')
        limit: Max number of downloads (None = all)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing Google Earth Engine with project: %s", project)
    ee.Initialize(project=project)

    df = pd.read_csv(labels_csv)
    df["sampling_date"] = pd.to_datetime(df["sampling_date"])

    # Group by lat/lon/date and take first sorting_id (matches allenai labels.py)
    grouped = df.groupby(
        ["latitude", "longitude", "sampling_date"], as_index=False
    ).agg({"sorting_id": "first", "lfmc_value": "mean"})

    if limit is not None:
        grouped = grouped.head(limit)

    already_downloaded = len(list(output_dir.glob("*.tif")))
    logger.info(
        "Downloading TIFs for %d samples (%d already exist)",
        len(grouped), already_downloaded
    )

    downloaded = 0
    skipped = 0
    failed = 0

    for _, row in tqdm(grouped.iterrows(), total=len(grouped), desc="Downloading TIFs"):
        sampling_date = row["sampling_date"].date()
        start_date, end_date = pad_dates(sampling_date)

        result = download_tif_for_sample(
            sorting_id=int(row["sorting_id"]),
            lat=float(row["latitude"]),
            lon=float(row["longitude"]),
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            project=project,
        )

        if result is True:
            downloaded += 1
        elif result is False:
            skipped += 1
        else:
            failed += 1

    logger.info(
        "Done. Downloaded: %d, Skipped (existing): %d, Failed: %d",
        downloaded, skipped, failed
    )


def download_for_bbox(
    bbox: tuple[float, float, float, float],
    start_date: date,
    end_date: date,
    output_dir: Path,
    project: str,
    identifier: str,
) -> None:
    """
    Download a single large GeoTIFF for a bounding box area (e.g. Maui County).
    Used for inference map generation.

    Args:
        bbox: (min_lat, min_lon, max_lat, max_lon) in WGS84
        start_date: Start of 12-month period
        end_date: End of 12-month period
        output_dir: Output directory
        project: GEE project ID
        identifier: Output filename stem (e.g. 'maui_2023_08')
    """
    from galileo.data.bbox import BBox

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{identifier}.tif"

    if output_path.exists():
        logger.info("%s already exists, skipping", output_path)
        return

    logger.info("Initializing Google Earth Engine with project: %s", project)
    ee.Initialize(project=project)

    min_lat, min_lon, max_lat, max_lon = bbox
    ee_polygon = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

    logger.info(
        "Creating GEE image for bbox %s, dates %s to %s",
        bbox, start_date, end_date
    )

    img = create_ee_image(
        polygon=ee_polygon,
        start_date=start_date,
        end_date=end_date,
    )

    logger.info("Requesting download URL (this may take a few minutes)...")
    url = img.getDownloadURL({
        "region": ee_polygon,
        "scale": 10,
        "filePerBand": False,
        "format": "GEO_TIFF",
    })

    logger.info("Downloading...")
    r = requests.get(url, stream=True, timeout=600)
    r.raise_for_status()

    with output_path.open("wb") as f:
        shutil.copyfileobj(r.raw, f)

    logger.info("Saved to %s", output_path)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(
        description="Download GEE satellite data for LFMC training or Maui inference"
    )
    parser.add_argument("--project", required=True, help="GEE project ID")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for TIF files")

    # Mode 1: download for training labels
    parser.add_argument(
        "--labels",
        type=Path,
        help="Path to lfmc_data_conus.csv (for training data download)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of downloads (useful for testing)"
    )

    # Mode 2: download for a bounding box (inference)
    parser.add_argument(
        "--bbox",
        type=str,
        help="Bounding box as min_lat,min_lon,max_lat,max_lon (e.g. 20.5,156.7,21.1,155.9)"
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date YYYY-MM-DD (for bbox mode)"
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date YYYY-MM-DD (for bbox mode)"
    )
    parser.add_argument(
        "--identifier",
        type=str,
        default="area",
        help="Output filename stem for bbox mode (e.g. maui_2023_08)"
    )

    args = parser.parse_args()

    if args.labels:
        download_for_labels(
            labels_csv=args.labels,
            output_dir=args.output,
            project=args.project,
            limit=args.limit,
        )
    elif args.bbox:
        min_lat, min_lon, max_lat, max_lon = [float(x) for x in args.bbox.split(",")]
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
        download_for_bbox(
            bbox=(min_lat, min_lon, max_lat, max_lon),
            start_date=start_date,
            end_date=end_date,
            output_dir=args.output,
            project=args.project,
            identifier=args.identifier,
        )
    else:
        parser.error("Must provide either --labels or --bbox")


if __name__ == "__main__":
    main()
