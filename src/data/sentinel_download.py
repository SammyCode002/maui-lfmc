"""
Sentinel-1/2 Satellite Imagery Download for LFMC Training

Downloads satellite imagery for each Globe-LFMC sample location.
Uses Microsoft Planetary Computer (free) or Google Earth Engine.

YOU NEED TO SET UP:
1. Planetary Computer API key: https://planetarycomputer.microsoft.com/
   OR
2. Google Earth Engine account: https://earthengine.google.com/

Usage:
    python -m src.data.sentinel_download \
        --sites data/processed/globe_lfmc_conus_west.csv \
        --output data/satellite/ \
        --provider planetary_computer
"""

import argparse
import logging
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("sentinel_download")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# Sentinel-2 L2A bands used by Galileo
# These are the 10 optical bands at various resolutions (all resampled to 10m)
S2_BANDS = [
    "B02",  # Blue (10m)
    "B03",  # Green (10m)
    "B04",  # Red (10m)
    "B05",  # Vegetation Red Edge 1 (20m)
    "B06",  # Vegetation Red Edge 2 (20m)
    "B07",  # Vegetation Red Edge 3 (20m)
    "B08",  # NIR (10m)
    "B8A",  # Narrow NIR (20m)
    "B11",  # SWIR 1 (20m)
    "B12",  # SWIR 2 (20m)
]

# Sentinel-1 RTC bands
S1_BANDS = ["VV", "VH"]


@dataclass
class SampleLocation:
    """A Globe-LFMC sample location with metadata for satellite download."""
    latitude: float
    longitude: float
    sampling_date: str
    lfmc_value: float
    sample_id: str
    # Size of the patch to download (in meters)
    patch_size_m: int = 2560  # 256 pixels at 10m resolution


def load_sample_locations(csv_path: str) -> list[SampleLocation]:
    """Load filtered Globe-LFMC sample locations from CSV."""
    start = time.time()
    logger.info(f"INPUT  | Loading samples from: {csv_path}")

    df = pd.read_csv(csv_path)
    samples = []
    for idx, row in df.iterrows():
        samples.append(SampleLocation(
            latitude=row["latitude"],
            longitude=row["longitude"],
            sampling_date=str(row.get("sampling_date", "")),
            lfmc_value=row.get("lfmc_value", np.nan),
            sample_id=f"sample_{idx:06d}",
        ))

    elapsed = time.time() - start
    logger.info(f"OUTPUT | Loaded {len(samples)} sample locations")
    logger.info(f"TIMING | Load took {elapsed:.3f}s")
    return samples


def download_via_planetary_computer(
    sample: SampleLocation,
    output_dir: Path,
    num_timesteps: int = 12,
) -> Optional[Path]:
    """
    Download Sentinel-1/2 imagery from Microsoft Planetary Computer.

    This downloads 12 monthly timesteps of satellite data centered on
    the sample date, matching what Galileo expects as input.

    Args:
        sample: Location and date to download for
        output_dir: Where to save the downloaded data
        num_timesteps: Number of monthly timesteps (default 12)

    Returns:
        Path to saved numpy file, or None if download failed
    """
    try:
        import pystac_client
        import planetary_computer
        import rioxarray
        import xarray as xr
    except ImportError:
        logger.error(
            "Missing dependencies. Install with:\n"
            "  pip install pystac-client planetary-computer rioxarray"
        )
        return None

    start = time.time()
    logger.info(f"INPUT  | Downloading for {sample.sample_id} at "
                f"({sample.latitude:.4f}, {sample.longitude:.4f})")

    # Connect to Planetary Computer STAC catalog
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # Define the area of interest (small bounding box around the point)
    # 0.025 degrees is roughly 2.5km at Hawaii's latitude
    buffer = 0.025
    bbox = [
        sample.longitude - buffer,
        sample.latitude - buffer,
        sample.longitude + buffer,
        sample.latitude + buffer,
    ]

    # Parse the sample date and create a 12-month search window
    from datetime import datetime, timedelta
    try:
        sample_date = datetime.fromisoformat(sample.sampling_date[:10])
    except (ValueError, TypeError):
        logger.warning(f"STATUS | Invalid date for {sample.sample_id}, skipping")
        return None

    # 6 months before and after the sample date
    start_date = sample_date - timedelta(days=180)
    end_date = sample_date + timedelta(days=180)
    time_range = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"

    # Search for Sentinel-2 L2A scenes
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=time_range,
        query={"eo:cloud_cover": {"lt": 30}},  # Max 30% cloud cover
    )

    items = list(search.items())
    logger.info(f"STATUS | Found {len(items)} Sentinel-2 scenes")

    if len(items) == 0:
        logger.warning(f"STATUS | No scenes found for {sample.sample_id}")
        return None

    # Group by month and take the least cloudy scene per month
    monthly_scenes = {}
    for item in items:
        month_key = item.datetime.strftime("%Y-%m")
        cloud = item.properties.get("eo:cloud_cover", 100)
        if month_key not in monthly_scenes or cloud < monthly_scenes[month_key][1]:
            monthly_scenes[month_key] = (item, cloud)

    # Select up to num_timesteps monthly scenes
    sorted_months = sorted(monthly_scenes.keys())
    selected = [monthly_scenes[m][0] for m in sorted_months[:num_timesteps]]

    logger.info(f"STATUS | Selected {len(selected)} monthly composites")

    # Download and stack the bands
    # Each timestep: [H, W, len(S2_BANDS)]
    # Final shape: [T, H, W, len(S2_BANDS)]
    all_timesteps = []
    for item in selected:
        bands_data = []
        for band_name in S2_BANDS:
            try:
                asset = item.assets[band_name]
                signed_asset = planetary_computer.sign(asset)
                da = rioxarray.open_rasterio(signed_asset.href)
                # Crop to our bbox and resample to 10m
                da_crop = da.rio.clip_box(*bbox)
                bands_data.append(da_crop.values[0])  # Remove band dim
            except Exception as e:
                logger.warning(f"STATUS | Failed to download {band_name}: {e}")
                bands_data.append(None)

        if all(b is not None for b in bands_data):
            # Stack bands: [H, W, num_bands]
            timestep = np.stack(bands_data, axis=-1)
            all_timesteps.append(timestep)

    if len(all_timesteps) == 0:
        logger.warning(f"STATUS | No valid timesteps for {sample.sample_id}")
        return None

    # Stack timesteps: [T, H, W, num_bands]
    data = np.stack(all_timesteps, axis=0)

    # Save as numpy file
    save_path = output_dir / f"{sample.sample_id}.npz"
    np.savez_compressed(
        save_path,
        s2_data=data,
        latitude=sample.latitude,
        longitude=sample.longitude,
        lfmc_value=sample.lfmc_value,
        sampling_date=sample.sampling_date,
        months=[s.datetime.strftime("%Y-%m") for s in selected],
    )

    elapsed = time.time() - start
    logger.info(f"OUTPUT | Saved {data.shape} to {save_path}")
    logger.info(f"TIMING | Download took {elapsed:.1f}s")
    return save_path


def download_via_gee(
    sample: SampleLocation,
    output_dir: Path,
    num_timesteps: int = 12,
) -> Optional[Path]:
    """
    Download satellite imagery via Google Earth Engine.

    Alternative to Planetary Computer. Requires GEE authentication.
    See: https://developers.google.com/earth-engine/guides/python_install
    """
    try:
        import ee
    except ImportError:
        logger.error("Install earthengine-api: pip install earthengine-api")
        return None

    start = time.time()
    logger.info(f"INPUT  | GEE download for {sample.sample_id}")

    # Initialize Earth Engine
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

    # Define point and buffer
    point = ee.Geometry.Point([sample.longitude, sample.latitude])
    region = point.buffer(1280)  # 1280m radius = ~2560m patch

    # Parse date range
    from datetime import datetime, timedelta
    try:
        sample_date = datetime.fromisoformat(sample.sampling_date[:10])
    except (ValueError, TypeError):
        return None

    start_date = (sample_date - timedelta(days=180)).strftime("%Y-%m-%d")
    end_date = (sample_date + timedelta(days=180)).strftime("%Y-%m-%d")

    # Get Sentinel-2 collection
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(point)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
          .select(S2_BANDS))

    # Get monthly composites (median)
    months = []
    current = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    while current < end and len(months) < num_timesteps:
        month_start = current.strftime("%Y-%m-%d")
        next_month = current.replace(day=28) + timedelta(days=4)
        month_end = next_month.replace(day=1).strftime("%Y-%m-%d")

        monthly = s2.filterDate(month_start, month_end).median()
        months.append(monthly)
        current = next_month.replace(day=1)

    # Download as numpy arrays
    # (You would use ee.batch.Export or geemap for actual downloads)
    logger.info(f"STATUS | Created {len(months)} monthly composites via GEE")
    logger.info("STATUS | Use geemap or ee.batch.Export to download rasters")

    elapsed = time.time() - start
    logger.info(f"TIMING | GEE setup took {elapsed:.1f}s")
    return None  # Placeholder, actual download needs export


def run_download_pipeline(
    sites_csv: str,
    output_dir: str,
    provider: str = "planetary_computer",
    max_samples: Optional[int] = None,
):
    """Run the satellite download pipeline for all sample locations."""
    start = time.time()
    logger.info("=" * 60)
    logger.info("Satellite Imagery Download Pipeline")
    logger.info("=" * 60)

    samples = load_sample_locations(sites_csv)
    if max_samples:
        samples = samples[:max_samples]
        logger.info(f"STATUS | Limited to {max_samples} samples")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    download_fn = {
        "planetary_computer": download_via_planetary_computer,
        "gee": download_via_gee,
    }[provider]

    successful = 0
    failed = 0
    for i, sample in enumerate(samples):
        logger.info(f"STATUS | Processing {i+1}/{len(samples)}")
        result = download_fn(sample, out)
        if result:
            successful += 1
        else:
            failed += 1

    elapsed = time.time() - start
    logger.info(f"OUTPUT | Downloaded: {successful}, Failed: {failed}")
    logger.info(f"TIMING | Total: {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Sentinel imagery")
    parser.add_argument("--sites", required=True, help="CSV with sample locations")
    parser.add_argument("--output", default="data/satellite/")
    parser.add_argument("--provider", default="planetary_computer",
                       choices=["planetary_computer", "gee"])
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    run_download_pipeline(args.sites, args.output, args.provider, args.max_samples)
