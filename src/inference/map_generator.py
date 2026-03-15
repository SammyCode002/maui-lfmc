"""
LFMC Map Generator for Maui County

Generates wall-to-wall (spatially complete) monthly LFMC maps by:
1. Downloading satellite imagery for Maui County from GEE
2. Tiling into overlapping 32x32 pixel patches (320m x 320m)
3. Running the trained Galileo-LFMC model on each patch
4. Stitching results into a single GeoTIFF at 10m resolution

The trained model is applied zero-shot to Hawaii — it was trained on CONUS
data but generalizes to Maui via the Galileo foundation model's pretrained
representations. This is the same approach used by Johnson et al. (2025)
for the LA Palisades/Eaton fire case studies.

Usage:
    # Generate map for August 2023 (month of Lahaina fire)
    python -m src.inference.map_generator \\
        --checkpoint checkpoints/conus/finetuned_model.pth \\
        --galileo-config path/to/galileo-data \\
        --year 2023 --month 8 \\
        --project ace-shine-392702 \\
        --output outputs/maps/

    # Generate all months for 2023
    python -m src.inference.map_generator \\
        --checkpoint checkpoints/conus/finetuned_model.pth \\
        --galileo-config path/to/galileo-data \\
        --year 2023 --all-months \\
        --project ace-shine-392702 \\
        --output outputs/maps/
"""

import argparse
import logging
from datetime import date
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Maui County geographic bounds (WGS84)
MAUI_BOUNDS = {
    "min_lat": 20.5,
    "max_lat": 21.1,
    "min_lon": -156.7,
    "max_lon": -155.9,
}

MAUI_CRS = "EPSG:4326"
PIXEL_SIZE_M = 10


def load_finetuned_model(
    checkpoint_path: Path,
    galileo_config_dir: Path,
):
    """Load the fine-tuned FineTuningModel from a checkpoint."""
    import json

    import torch
    from galileo.galileo import Encoder
    from galileo.utils import device
    from lfmc.core.finetuning import FineTuningModel
    import torch.nn as nn

    # Load encoder config
    config_path = galileo_config_dir / "models" / "tiny" / "config.json"
    with config_path.open("r") as f:
        config = json.load(f)
    encoder = Encoder(**config["model"]["encoder"])

    head = nn.Linear(encoder.embedding_size, 1)
    model = FineTuningModel(encoder, head).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info("Loaded model from %s", checkpoint_path)
    return model, device


def generate_monthly_map(
    year: int,
    month: int,
    checkpoint_path: Path,
    galileo_config_dir: Path,
    gee_project: str,
    output_dir: Path,
    patch_hw: int = 32,
    patch_overlap: int = 4,
) -> Path:
    """
    Generate a wall-to-wall LFMC map for Maui County for a given month.

    Strategy:
    - Download 12-month satellite composite ending in the target month from GEE
    - Tile Maui into overlapping 32x32 pixel patches (320m x 320m at 10m)
    - Run model inference on each patch
    - Stitch with overlap averaging to remove edge artifacts
    - Save as GeoTIFF

    Args:
        year: Target year
        month: Target month (1-12)
        checkpoint_path: Path to finetuned_model.pth
        galileo_config_dir: Path to galileo-data directory
        gee_project: GEE project ID
        output_dir: Where to save output GeoTIFF
        patch_hw: Patch height/width in pixels (default 32 = 320m)
        patch_overlap: Overlap between patches in pixels

    Returns:
        Path to saved GeoTIFF
    """
    import ee
    import rasterio
    from rasterio.transform import from_bounds

    from galileo.data.config import NORMALIZATION_DICT_FILENAME
    from galileo.data.dataset import Dataset, Normalizer
    from galileo.data.earthengine.eo import create_ee_image
    from galileo.utils import masked_output_np_to_tensor
    from lfmc.core.finetuning import FineTuningModel

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"lfmc_maui_{year}_{month:02d}.tif"

    if output_path.exists():
        logger.info("Map already exists: %s", output_path)
        return output_path

    # Initialize GEE
    ee.Initialize(project=gee_project)

    # Load model
    model, device = load_finetuned_model(checkpoint_path, galileo_config_dir)

    # Load normalizer
    norm_path = galileo_config_dir / NORMALIZATION_DICT_FILENAME
    normalizing_dicts = Dataset.load_normalization_values(norm_path)
    normalizer = Normalizer(std=True, normalizing_dicts=normalizing_dicts)

    # Compute 12-month window ending at target month
    end_date = date(year, month, 28)  # last ~day of target month
    from datetime import timedelta
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    end_date = date(year, month, last_day)
    start_date = date(end_date.year - 1, end_date.month, 1)

    logger.info(
        "Generating LFMC map for %d-%02d (data window: %s to %s)",
        year, month, start_date, end_date
    )

    # Compute grid of tiles covering Maui
    tiles = _compute_tile_grid(MAUI_BOUNDS, patch_hw, patch_overlap)
    logger.info("Tiling Maui into %d patches (%dx%d px each)", len(tiles), patch_hw, patch_hw)

    # Build output raster dimensions
    lat_range = MAUI_BOUNDS["max_lat"] - MAUI_BOUNDS["min_lat"]
    lon_range = MAUI_BOUNDS["max_lon"] - MAUI_BOUNDS["min_lon"]
    # ~111km per degree lat, ~104km per degree lon at Maui's latitude
    height_px = int(lat_range * 111000 / PIXEL_SIZE_M)
    width_px = int(lon_range * 104000 / PIXEL_SIZE_M)

    lfmc_sum = np.zeros((height_px, width_px), dtype=np.float64)
    count = np.zeros((height_px, width_px), dtype=np.float64)

    n_success = 0
    n_failed = 0

    for i, tile in enumerate(tiles):
        if (i + 1) % 50 == 0:
            logger.info("Processing tile %d/%d (%d ok, %d failed)", i + 1, len(tiles), n_success, n_failed)

        try:
            pred = _predict_tile(
                tile=tile,
                start_date=start_date,
                end_date=end_date,
                model=model,
                normalizer=normalizer,
                device=device,
                patch_hw=patch_hw,
                gee_project=gee_project,
            )

            # Place prediction in output raster
            row_start = int((tile["min_lat"] - MAUI_BOUNDS["min_lat"]) / (PIXEL_SIZE_M / 111000))
            col_start = int((tile["min_lon"] - MAUI_BOUNDS["min_lon"]) / (PIXEL_SIZE_M / 104000))
            row_end = min(row_start + patch_hw, height_px)
            col_end = min(col_start + patch_hw, width_px)
            h = row_end - row_start
            w = col_end - col_start

            lfmc_sum[row_start:row_end, col_start:col_end] += pred[:h, :w]
            count[row_start:row_end, col_start:col_end] += 1.0
            n_success += 1

        except Exception as e:
            logger.debug("Tile %d failed: %s", i, e)
            n_failed += 1

    logger.info(
        "Inference complete: %d/%d tiles successful", n_success, n_success + n_failed
    )

    # Average overlapping predictions
    lfmc_map = np.where(count > 0, lfmc_sum / count, -9999.0).astype(np.float32)
    lfmc_map = np.flipud(lfmc_map)  # Raster convention: top-down

    # Save GeoTIFF
    transform = from_bounds(
        MAUI_BOUNDS["min_lon"], MAUI_BOUNDS["min_lat"],
        MAUI_BOUNDS["max_lon"], MAUI_BOUNDS["max_lat"],
        width_px, height_px,
    )

    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=height_px,
        width=width_px,
        count=1,
        dtype=np.float32,
        crs=MAUI_CRS,
        transform=transform,
        nodata=-9999.0,
        compress="lzw",
    ) as dst:
        dst.write(lfmc_map, 1)
        dst.update_tags(
            description="Live Fuel Moisture Content (LFMC)",
            units="percent (0-302)",
            year=str(year),
            month=str(month),
            model="Galileo-Tiny fine-tuned on Globe-LFMC 2.0 CONUS",
            reference="Johnson et al. 2025, arXiv:2506.20132",
        )

    logger.info("Saved LFMC map: %s", output_path)
    return output_path


def _compute_tile_grid(bounds: dict, patch_hw: int, overlap: int) -> list[dict]:
    """Compute overlapping tile grid covering the bounds."""
    # Degrees per pixel at Maui's latitude
    lat_per_px = PIXEL_SIZE_M / 111000.0
    lon_per_px = PIXEL_SIZE_M / 104000.0

    patch_lat = patch_hw * lat_per_px
    patch_lon = patch_hw * lon_per_px
    step_lat = (patch_hw - overlap) * lat_per_px
    step_lon = (patch_hw - overlap) * lon_per_px

    tiles = []
    lat = bounds["min_lat"]
    while lat < bounds["max_lat"]:
        lon = bounds["min_lon"]
        while lon < bounds["max_lon"]:
            tiles.append({
                "min_lat": lat,
                "max_lat": min(lat + patch_lat, bounds["max_lat"]),
                "min_lon": lon,
                "max_lon": min(lon + patch_lon, bounds["max_lon"]),
            })
            lon += step_lon
        lat += step_lat

    return tiles


def _predict_tile(
    tile: dict,
    start_date: date,
    end_date: date,
    model,
    normalizer,
    device,
    patch_hw: int,
    gee_project: str,
) -> np.ndarray:
    """Download satellite data for a tile and run model inference."""
    import shutil
    import tempfile

    import ee
    import requests
    import rioxarray

    from galileo.data.dataset import Dataset
    from galileo.data.earthengine.eo import create_ee_image
    from galileo.utils import masked_output_np_to_tensor
    from lfmc.core.finetuning import FineTuningModel

    polygon = ee.Geometry.Rectangle([
        tile["min_lon"], tile["min_lat"],
        tile["max_lon"], tile["max_lat"],
    ])

    img = create_ee_image(polygon, start_date, end_date)
    url = img.getDownloadURL({
        "region": polygon,
        "scale": PIXEL_SIZE_M,
        "filePerBand": False,
        "format": "GEO_TIFF",
    })

    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        shutil.copyfileobj(r.raw, f)
        tmp_path = Path(f.name)

    try:
        # Parse TIF using Galileo's dataset loader
        dataset = Dataset.__new__(Dataset)
        dataset.output_hw = patch_hw
        dataset.output_timesteps = 12
        dataset.h5py_folder = None

        # Use month array based on start month
        import types
        def _month_array(self, tif_path, num_timesteps):
            return np.fmod(np.arange(start_date.month - 1, start_date.month - 1 + num_timesteps), 12)
        dataset.month_array_from_file = types.MethodType(_month_array, dataset)

        dataset_output = dataset._tif_to_array(tmp_path).normalize(normalizer)
        s_t_x, sp_x, t_x, st_x, months = dataset_output

        # Subset to patch_hw x patch_hw
        from galileo.data.dataset import Dataset as D
        s_t_x, sp_x, t_x, st_x, months = D.subset_image(
            s_t_x, sp_x, t_x, st_x, months,
            size=patch_hw, num_timesteps=12
        )

        # Make masks (all zeros = include everything)
        from galileo.data.dataset import (
            SPACE_BAND_GROUPS_IDX, SPACE_TIME_BANDS_GROUPS_IDX,
            STATIC_BAND_GROUPS_IDX, TIME_BAND_GROUPS_IDX,
        )
        s_t_m = np.zeros([patch_hw, patch_hw, 12, len(SPACE_TIME_BANDS_GROUPS_IDX)])
        sp_m = np.zeros([patch_hw, patch_hw, len(SPACE_BAND_GROUPS_IDX)])
        t_m = np.zeros([12, len(TIME_BAND_GROUPS_IDX)])
        st_m = np.zeros([len(STATIC_BAND_GROUPS_IDX)])

        tensors = masked_output_np_to_tensor(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months)
        tensors = [t.unsqueeze(0).to(device) for t in tensors]

        s_t_x_t, sp_x_t, t_x_t, st_x_t, s_t_m_t, sp_m_t, t_m_t, st_m_t, months_t = tensors

        with torch.no_grad():
            pred = model(s_t_x_t, sp_x_t, t_x_t, st_x_t,
                         s_t_m_t, sp_m_t, t_m_t, st_m_t, months_t)
            # pred is [1, 1] normalized 0-1, scale to LFMC %
            lfmc_value = pred[0, 0].item() * 302.0

        return np.full((patch_hw, patch_hw), lfmc_value, dtype=np.float32)

    finally:
        tmp_path.unlink(missing_ok=True)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(
        description="Generate monthly LFMC maps for Maui County"
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to finetuned_model.pth")
    parser.add_argument("--galileo-config", type=Path, required=True,
                        help="Path to galileo-data directory")
    parser.add_argument("--project", required=True,
                        help="GEE project ID")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, default=None,
                        help="Month (1-12). Omit with --all-months for full year.")
    parser.add_argument("--all-months", action="store_true",
                        help="Generate all 12 months for the given year")
    parser.add_argument("--output", type=Path, default=Path("outputs/maps"))

    args = parser.parse_args()

    months = list(range(1, 13)) if args.all_months else [args.month]

    for month in months:
        output_path = generate_monthly_map(
            year=args.year,
            month=month,
            checkpoint_path=args.checkpoint,
            galileo_config_dir=args.galileo_config,
            gee_project=args.project,
            output_dir=args.output,
        )
        logger.info("Generated: %s", output_path)


if __name__ == "__main__":
    main()
