"""
LFMC Map Visualization

Creates publication-quality visualizations of LFMC maps,
matching the style from Johnson et al. (2025).

Color scheme: Red (dry, 0%) -> Yellow -> Green (moist, 180%)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("plot_maps")


# LFMC color scheme matching Johnson et al.
# Red = dry (fire risk), Green = moist (safe)
LFMC_CMAP_COLORS = [
    (0.0, "#8B0000"),    # 0%: dark red
    (0.22, "#FF4500"),   # 40%: orange-red
    (0.33, "#FFA500"),   # 60%: orange
    (0.44, "#FFD700"),   # 80%: gold (critical threshold)
    (0.56, "#ADFF2F"),   # 100%: yellow-green
    (0.67, "#32CD32"),   # 120%: lime green
    (0.78, "#228B22"),   # 140%: forest green
    (1.0, "#006400"),    # 180%: dark green
]


def create_lfmc_colormap():
    """Create a custom colormap for LFMC visualization."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    positions = [c[0] for c in LFMC_CMAP_COLORS]
    colors = [c[1] for c in LFMC_CMAP_COLORS]
    return LinearSegmentedColormap.from_list("lfmc", list(zip(positions, colors)))


def plot_single_map(
    lfmc_data: np.ndarray,
    title: str = "LFMC Map",
    output_path: Optional[str] = None,
    vmin: float = 0,
    vmax: float = 180,
    nodata: float = -9999.0,
    figsize: tuple = (10, 8),
) -> None:
    """
    Plot a single LFMC map.

    Args:
        lfmc_data: 2D array of LFMC values
        title: Plot title (e.g., "Aug 2023")
        output_path: Save to file if provided
        vmin, vmax: Color scale range
        nodata: NoData value to mask
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Mask nodata values
    masked = np.ma.masked_where(
        (lfmc_data == nodata) | np.isnan(lfmc_data),
        lfmc_data,
    )

    cmap = create_lfmc_colormap()
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.axis("off")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("LFMC (%)", fontsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"OUTPUT | Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_monthly_comparison(
    maps: dict,
    output_path: Optional[str] = None,
    vmin: float = 0,
    vmax: float = 180,
    nodata: float = -9999.0,
) -> None:
    """
    Plot a grid of monthly LFMC maps (like Figure 1 in Johnson et al.).

    Args:
        maps: Dict of {label: lfmc_array}, e.g., {"Aug 2021": array, ...}
        output_path: Save to file if provided
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n = len(maps)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(5 * ncols + 1, 5 * nrows))
    gs = GridSpec(nrows, ncols + 1, width_ratios=[1] * ncols + [0.05], figure=fig)

    cmap = create_lfmc_colormap()

    for i, (label, data) in enumerate(maps.items()):
        row, col = divmod(i, ncols)
        ax = fig.add_subplot(gs[row, col])

        masked = np.ma.masked_where(
            (data == nodata) | np.isnan(data), data
        )
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.axis("off")

    # Shared colorbar
    cbar_ax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("LFMC (%)", fontsize=12)

    plt.suptitle("Live Fuel Moisture Content - Maui County", fontsize=16, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"OUTPUT | Saved comparison to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_lfmc_timeseries(
    monthly_means: dict,
    output_path: Optional[str] = None,
) -> None:
    """
    Plot LFMC time series (monthly mean across Maui).

    Useful for seeing seasonal patterns and year-to-year variability.

    Args:
        monthly_means: Dict of {"YYYY-MM": mean_lfmc_value}
    """
    import matplotlib.pyplot as plt
    from datetime import datetime

    dates = sorted(monthly_means.keys())
    values = [monthly_means[d] for d in dates]
    date_objs = [datetime.strptime(d, "%Y-%m") for d in dates]

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    ax.plot(date_objs, values, "b-o", markersize=4, linewidth=1.5)
    ax.axhline(y=80, color="red", linestyle="--", alpha=0.7, label="Critical threshold (80%)")

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Mean LFMC (%)", fontsize=12)
    ax.set_title("Monthly Mean LFMC - Maui County", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"OUTPUT | Saved timeseries to {output_path}")
    else:
        plt.show()

    plt.close()


def load_geotiff(filepath: str) -> np.ndarray:
    """Load LFMC map from GeoTIFF file."""
    try:
        import rasterio
        with rasterio.open(filepath) as src:
            return src.read(1)
    except ImportError:
        return np.load(filepath.replace(".tif", ".npy"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize LFMC maps")
    parser.add_argument("--input", required=True, help="GeoTIFF or directory of GeoTIFFs")
    parser.add_argument("--output", default="outputs/plots/")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        data = load_geotiff(str(input_path))
        plot_single_map(data, title=input_path.stem, output_path=str(output_dir / f"{input_path.stem}.png"))
    elif input_path.is_dir():
        maps = {}
        for tif in sorted(input_path.glob("*.tif")):
            data = load_geotiff(str(tif))
            label = tif.stem.replace("lfmc_maui_", "").replace("_", " ")
            maps[label] = data
        if maps:
            plot_monthly_comparison(maps, output_path=str(output_dir / "comparison.png"))
