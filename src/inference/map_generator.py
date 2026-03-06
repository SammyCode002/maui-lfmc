"""
LFMC Map Generator for Maui County

Generates wall-to-wall (spatially complete) LFMC maps by:
1. Downloading satellite imagery for the entire county
2. Tiling into overlapping patches
3. Running inference on each patch
4. Stitching results into a single GeoTIFF

Output: Monthly GeoTIFF files with LFMC values (0-200%)

Usage:
    python -m src.inference.map_generator \
        --config configs/inference.yaml \
        --year 2023 \
        --month 8
"""

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger("map_generator")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# Maui County geographic bounds
MAUI_BOUNDS = {
    "lat_min": 20.5,
    "lat_max": 21.1,
    "lon_min": -156.7,
    "lon_max": -155.9,
}

# UTM Zone 4N for Hawaii
MAUI_CRS = "EPSG:32604"
PIXEL_SIZE_M = 10  # 10 meter resolution


@dataclass
class MapConfig:
    """Configuration for map generation."""
    checkpoint_path: str        # Path to trained model checkpoint
    output_dir: str = "outputs/maps"
    patch_size: int = 256       # Pixels per patch side
    overlap: int = 32           # Overlap between patches (pixels)
    batch_size: int = 16        # Patches per GPU batch
    nodata_value: float = -9999.0


def compute_grid_tiles(
    bounds: dict,
    patch_size: int,
    overlap: int,
    pixel_size_m: float = 10.0,
) -> list[dict]:
    """
    Compute a grid of overlapping tiles covering the study area.

    Each tile is a dict with lat/lon bounds that will be downloaded
    and fed through the model.

    Args:
        bounds: Geographic bounds (lat_min, lat_max, lon_min, lon_max)
        patch_size: Size of each tile in pixels
        overlap: Overlap between adjacent tiles in pixels
        pixel_size_m: Ground resolution in meters

    Returns:
        List of tile dicts with bounds and pixel coordinates
    """
    start = time.time()

    # Convert pixel sizes to approximate degrees
    # At Maui's latitude (~20.8N), 1 degree lat ~ 111km, 1 degree lon ~ 104km
    lat_deg_per_pixel = pixel_size_m / 111000.0
    lon_deg_per_pixel = pixel_size_m / 104000.0

    patch_lat = patch_size * lat_deg_per_pixel
    patch_lon = patch_size * lon_deg_per_pixel
    step_lat = (patch_size - overlap) * lat_deg_per_pixel
    step_lon = (patch_size - overlap) * lon_deg_per_pixel

    tiles = []
    lat = bounds["lat_min"]
    row = 0
    while lat < bounds["lat_max"]:
        lon = bounds["lon_min"]
        col = 0
        while lon < bounds["lon_max"]:
            tiles.append({
                "lat_min": lat,
                "lat_max": min(lat + patch_lat, bounds["lat_max"]),
                "lon_min": lon,
                "lon_max": min(lon + patch_lon, bounds["lon_max"]),
                "row": row,
                "col": col,
            })
            lon += step_lon
            col += 1
        lat += step_lat
        row += 1

    elapsed = time.time() - start
    logger.info(f"OUTPUT | Grid: {row} rows x {col} cols = {len(tiles)} tiles")
    logger.info(f"TIMING | Grid computation: {elapsed:.3f}s")
    return tiles


def predict_tile(model, tile_data, device) -> np.ndarray:
    """
    Run LFMC prediction on a single tile of satellite imagery.

    Args:
        model: Trained GalileoLFMC model
        tile_data: Preprocessed satellite data for the tile
        device: torch device

    Returns:
        LFMC predictions array [H, W]
    """
    import torch

    model.eval()
    with torch.no_grad():
        # tile_data should already be formatted as Galileo input
        if isinstance(tile_data, np.ndarray):
            tile_data = torch.from_numpy(tile_data).unsqueeze(0).to(device)

        predictions = model(tile_data)
        return predictions.squeeze().cpu().numpy()


def stitch_tiles(
    predictions: list[Tuple[dict, np.ndarray]],
    bounds: dict,
    pixel_size_m: float = 10.0,
    nodata: float = -9999.0,
) -> Tuple[np.ndarray, dict]:
    """
    Stitch predicted tiles into a single continuous map.

    For overlapping regions, takes the average of predictions.
    This reduces edge artifacts between tiles.

    Returns:
        (lfmc_map, transform_info) tuple
    """
    start = time.time()

    lat_deg_per_pixel = pixel_size_m / 111000.0
    lon_deg_per_pixel = pixel_size_m / 104000.0

    # Compute output raster dimensions
    height = int(np.ceil((bounds["lat_max"] - bounds["lat_min"]) / lat_deg_per_pixel))
    width = int(np.ceil((bounds["lon_max"] - bounds["lon_min"]) / lon_deg_per_pixel))

    # Accumulator and count arrays for averaging overlaps
    lfmc_sum = np.zeros((height, width), dtype=np.float64)
    count = np.zeros((height, width), dtype=np.float64)

    for tile_info, pred in predictions:
        # Compute pixel coordinates for this tile
        row_start = int((tile_info["lat_min"] - bounds["lat_min"]) / lat_deg_per_pixel)
        col_start = int((tile_info["lon_min"] - bounds["lon_min"]) / lon_deg_per_pixel)

        h, w = pred.shape
        row_end = min(row_start + h, height)
        col_end = min(col_start + w, width)
        h_clip = row_end - row_start
        w_clip = col_end - col_start

        lfmc_sum[row_start:row_end, col_start:col_end] += pred[:h_clip, :w_clip]
        count[row_start:row_end, col_start:col_end] += 1.0

    # Average where we have predictions, nodata elsewhere
    lfmc_map = np.where(count > 0, lfmc_sum / count, nodata).astype(np.float32)

    # Flip vertically because raster convention is top-down
    lfmc_map = np.flipud(lfmc_map)

    transform_info = {
        "width": width,
        "height": height,
        "lat_min": bounds["lat_min"],
        "lat_max": bounds["lat_max"],
        "lon_min": bounds["lon_min"],
        "lon_max": bounds["lon_max"],
        "pixel_size_m": pixel_size_m,
        "crs": MAUI_CRS,
    }

    elapsed = time.time() - start
    logger.info(f"OUTPUT | Stitched map: {height}x{width} pixels")
    logger.info(f"TIMING | Stitching: {elapsed:.3f}s")
    return lfmc_map, transform_info


def save_geotiff(
    lfmc_map: np.ndarray,
    transform_info: dict,
    output_path: str,
    nodata: float = -9999.0,
) -> None:
    """
    Save LFMC map as a GeoTIFF file.

    Requires rasterio. The output GeoTIFF includes:
    - Proper CRS (UTM Zone 4N)
    - Geotransform for georeferencing
    - NoData value for ocean/invalid pixels
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError:
        logger.error("Install rasterio: pip install rasterio")
        # Fallback: save as numpy
        np.save(output_path.replace(".tif", ".npy"), lfmc_map)
        logger.info(f"OUTPUT | Saved as numpy: {output_path.replace('.tif', '.npy')}")
        return

    height, width = lfmc_map.shape
    transform = from_bounds(
        transform_info["lon_min"],
        transform_info["lat_min"],
        transform_info["lon_max"],
        transform_info["lat_max"],
        width,
        height,
    )

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=np.float32,
        crs=transform_info["crs"],
        transform=transform,
        nodata=nodata,
        compress="lzw",
    ) as dst:
        dst.write(lfmc_map, 1)
        dst.update_tags(
            description="Live Fuel Moisture Content (LFMC) prediction",
            units="percent",
            model="Galileo-Tiny fine-tuned on Globe-LFMC 2.0",
        )

    logger.info(f"OUTPUT | Saved GeoTIFF: {output_path}")


def generate_monthly_map(
    model,
    year: int,
    month: int,
    config: MapConfig,
    device=None,
) -> str:
    """
    Generate a complete LFMC map for one month.

    This is the main entry point for map generation.

    Args:
        model: Trained GalileoLFMC model
        year: Target year (e.g., 2023)
        month: Target month (1-12)
        config: Map generation configuration
        device: torch device

    Returns:
        Path to the saved GeoTIFF file
    """
    import torch

    start = time.time()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info(f"Generating LFMC map for {year}-{month:02d}")
    logger.info("=" * 60)

    # Step 1: Compute tile grid
    tiles = compute_grid_tiles(MAUI_BOUNDS, config.patch_size, config.overlap)

    # Step 2: Download satellite imagery for each tile
    # (This would call sentinel_download functions)
    logger.info("STATUS | Downloading satellite imagery for tiles...")
    logger.info("STATUS | (This step requires satellite data access)")

    # Step 3: Run predictions
    predictions = []
    for i, tile in enumerate(tiles):
        if (i + 1) % 50 == 0:
            logger.info(f"STATUS | Processing tile {i+1}/{len(tiles)}")

        # TODO: Load actual satellite data for this tile
        # For now, create placeholder
        # In production: tile_data = download_and_preprocess_tile(tile, year, month)
        fake_pred = np.random.uniform(50, 150, (config.patch_size, config.patch_size))
        predictions.append((tile, fake_pred.astype(np.float32)))

    # Step 4: Stitch tiles
    lfmc_map, transform_info = stitch_tiles(predictions, MAUI_BOUNDS, nodata=config.nodata_value)

    # Step 5: Save
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / f"lfmc_maui_{year}_{month:02d}.tif")
    save_geotiff(lfmc_map, transform_info, output_path, config.nodata_value)

    elapsed = time.time() - start
    logger.info(f"TIMING | Map generation: {elapsed:.1f}s")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LFMC maps")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--output-dir", default="outputs/maps")
    args = parser.parse_args()

    config = MapConfig(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )

    # NOTE: Load your trained model here
    # model = load_trained_model(args.checkpoint)
    print(f"Map generation configured for {args.year}-{args.month:02d}")
    print(f"Output dir: {args.output_dir}")
    print("Load your trained model and call generate_monthly_map()")
