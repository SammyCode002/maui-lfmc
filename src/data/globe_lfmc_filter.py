"""
Globe-LFMC 2.0 Data Filter for Hawaii / Maui County

Filters the Globe-LFMC 2.0 dataset to samples relevant for training
an LFMC model for Maui County, Hawaii.

Download Globe-LFMC 2.0 from: https://doi.org/10.6084/m9.figshare.24312164

Usage:
    python -m src.data.globe_lfmc_filter \
        --input data/raw/Globe-LFMC-2.0.xlsx \
        --output data/processed/ \
        --region maui
"""

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import numpy as np

# 4x4 Debug Logging: inputs, outputs, timing, status
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("globe_lfmc_filter")


REGION_BOUNDS = {
    "hawaii_state": {
        "lat_min": 18.5, "lat_max": 22.5,
        "lon_min": -161.0, "lon_max": -154.0,
        "description": "All Hawaiian Islands",
    },
    "maui": {
        "lat_min": 20.5, "lat_max": 21.1,
        "lon_min": -156.7, "lon_max": -155.9,
        "description": "Maui Island (including Kahoolawe)",
    },
    "conus_west": {
        "lat_min": 30.0, "lat_max": 49.0,
        "lon_min": -125.0, "lon_max": -100.0,
        "description": "Western CONUS (most Globe-LFMC samples here)",
    },
    "conus": {
        "lat_min": 24.0, "lat_max": 49.5,
        "lon_min": -125.0, "lon_max": -66.0,
        "description": "Continental United States",
    },
}


def load_globe_lfmc(filepath: str) -> pd.DataFrame:
    """Load Globe-LFMC 2.0 dataset from Excel file."""
    start = time.time()
    logger.info(f"INPUT  | Loading Globe-LFMC from: {filepath}")
    try:
        df = pd.read_excel(filepath, sheet_name="LFMC Data")
    except Exception:
        logger.warning("Could not find 'LFMC Data' sheet, trying first sheet")
        df = pd.read_excel(filepath, sheet_name=0)
    elapsed = time.time() - start
    logger.info(f"OUTPUT | Loaded {len(df)} samples, {len(df.columns)} columns")
    logger.info(f"TIMING | Load took {elapsed:.1f}s")
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to consistent format."""
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if "latitude" in cl or cl == "lat":
            col_map[col] = "latitude"
        elif "longitude" in cl or cl == "lon":
            col_map[col] = "longitude"
        elif "lfmc" in cl and ("value" in cl or "%" in cl):
            col_map[col] = "lfmc_value"
        elif "sampling" in cl and "date" in cl:
            col_map[col] = "sampling_date"
        elif "species" in cl:
            col_map[col] = "species"
        elif "country" in cl:
            col_map[col] = "country"
    df = df.rename(columns=col_map)
    logger.info(f"STATUS | Standardized columns: {list(col_map.values())}")
    return df


def filter_by_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """Filter samples to a geographic bounding box."""
    start = time.time()
    if region not in REGION_BOUNDS:
        raise ValueError(f"Unknown region. Options: {list(REGION_BOUNDS.keys())}")
    bounds = REGION_BOUNDS[region]
    logger.info(f"INPUT  | Filtering {len(df)} samples to: {region}")
    mask = (
        (df["latitude"] >= bounds["lat_min"])
        & (df["latitude"] <= bounds["lat_max"])
        & (df["longitude"] >= bounds["lon_min"])
        & (df["longitude"] <= bounds["lon_max"])
    )
    filtered = df[mask].copy()
    elapsed = time.time() - start
    logger.info(f"OUTPUT | {len(filtered)} samples in {region}")
    logger.info(f"TIMING | Filter took {elapsed:.3f}s")
    if len(filtered) == 0:
        logger.warning(f"STATUS | NO samples in {region}! Use conus_west for transfer learning.")
    return filtered


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add year, month, season columns from sampling date."""
    if "sampling_date" not in df.columns:
        return df
    df["sampling_date"] = pd.to_datetime(df["sampling_date"], errors="coerce")
    df["year"] = df["sampling_date"].dt.year
    df["month"] = df["sampling_date"].dt.month
    df["day_of_year"] = df["sampling_date"].dt.dayofyear
    season_map = {12: "winter", 1: "winter", 2: "winter",
                  3: "spring", 4: "spring", 5: "spring",
                  6: "summer", 7: "summer", 8: "summer",
                  9: "fall", 10: "fall", 11: "fall"}
    df["season"] = df["month"].map(season_map)
    return df


def compute_summary_stats(df: pd.DataFrame, region: str) -> dict:
    """Compute summary statistics for filtered data."""
    stats = {"region": region, "total_samples": len(df)}
    if "lfmc_value" in df.columns and len(df) > 0:
        stats.update({
            "lfmc_mean": round(float(df["lfmc_value"].mean()), 2),
            "lfmc_std": round(float(df["lfmc_value"].std()), 2),
            "lfmc_min": round(float(df["lfmc_value"].min()), 2),
            "lfmc_max": round(float(df["lfmc_value"].max()), 2),
        })
    if "year" in df.columns and len(df) > 0:
        stats["year_range"] = f"{int(df['year'].min())}-{int(df['year'].max())}"
    if "species" in df.columns and len(df) > 0:
        stats["unique_species"] = int(df["species"].nunique())
    return stats


def run_filter_pipeline(input_path, output_dir, region="maui", also_save_conus=True):
    """Run the complete filtering pipeline."""
    pipeline_start = time.time()
    logger.info("=" * 60)
    logger.info("Globe-LFMC Filtering Pipeline")
    logger.info("=" * 60)

    df = load_globe_lfmc(input_path)
    df = standardize_columns(df)
    df = add_temporal_features(df)
    df_region = filter_by_region(df, region)
    stats = compute_summary_stats(df_region, region)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if len(df_region) > 0:
        df_region.to_csv(out / f"globe_lfmc_{region}.csv", index=False)
    with open(out / f"globe_lfmc_{region}_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    if also_save_conus and region != "conus_west":
        logger.info("Also saving CONUS-west for baseline training...")
        df_conus = filter_by_region(df, "conus_west")
        if len(df_conus) > 0:
            df_conus.to_csv(out / "globe_lfmc_conus_west.csv", index=False)

    elapsed = time.time() - pipeline_start
    logger.info(f"TIMING | Full pipeline: {elapsed:.1f}s")
    print(f"\nSamples in {region}: {stats['total_samples']}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter Globe-LFMC 2.0")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--region", default="maui", choices=list(REGION_BOUNDS.keys()))
    args = parser.parse_args()
    run_filter_pipeline(args.input, args.output, args.region)
