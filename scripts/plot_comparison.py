"""
Multi-year August LFMC comparison figure.

Produces a 2x2 grid of Aug 2021 / Aug 2022 / Aug 2023 / Aug 2024 maps,
matching the style shown in the internship project description (Johnson et al.).

Ocean pixels are masked. Lahaina fire origin marked on Aug 2023 panel.

Usage:
    python scripts/plot_comparison.py
    python scripts/plot_comparison.py --output docs/aug_comparison.png
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from pathlib import Path
import tempfile
import urllib.request
import zipfile

REPO = Path(__file__).parent.parent
MAPS_DIR = REPO / "outputs/maps"
DEFAULT_OUT = REPO / "outputs/plots/aug_comparison.png"

BOUNDS = dict(min_lat=20.5, max_lat=21.1, min_lon=-156.7, max_lon=-155.9)
LAHAINA_LON, LAHAINA_LAT = -156.677, 20.875

YEARS = [2021, 2022, 2023, 2024]

CMAP_COLORS = [
    (0.00, "#8B0000"),
    (0.22, "#FF4500"),
    (0.33, "#FFA500"),
    (0.44, "#FFD700"),
    (0.56, "#ADFF2F"),
    (0.67, "#32CD32"),
    (0.78, "#228B22"),
    (1.00, "#006400"),
]
cmap = LinearSegmentedColormap.from_list("lfmc", [(p, c) for p, c in CMAP_COLORS])
cmap.set_bad(color="none")


def get_maui_land():
    import geopandas as gpd
    import shapely.geometry as geom

    ne_candidates = [
        Path("C:/Users/LazyB/AppData/Local/Temp/ne_10m_land/ne_10m_land.shp"),
        Path.home() / "ne_10m_land/ne_10m_land.shp",
        Path(tempfile.gettempdir()) / "ne_10m_land/ne_10m_land.shp",
    ]
    shp = None
    for c in ne_candidates:
        if c.exists():
            shp = c
            break

    if shp is None:
        url = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip"
        zip_path = Path(tempfile.gettempdir()) / "ne_10m_land.zip"
        out_dir = Path(tempfile.gettempdir()) / "ne_10m_land"
        print("Downloading Natural Earth 10m land...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out_dir)
        shp = out_dir / "ne_10m_land.shp"

    world = gpd.read_file(shp)
    maui_box = geom.box(
        BOUNDS["min_lon"] - 0.05, BOUNDS["min_lat"] - 0.05,
        BOUNDS["max_lon"] + 0.05, BOUNDS["max_lat"] + 0.05,
    )
    return world.clip(maui_box)


def load_and_mask(tif_path, land):
    import rasterio
    from rasterio.features import geometry_mask

    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)
        transform = src.transform
        height, width = data.shape

    ocean_mask = geometry_mask(
        list(land.geometry),
        transform=transform,
        invert=False,
        out_shape=(height, width),
    )
    masked = np.ma.masked_where(
        (data == -9999) | np.isnan(data) | ocean_mask, data
    )
    return masked


def main(output_path):
    # Check which years are available
    available = {}
    missing = []
    for year in YEARS:
        tif = MAPS_DIR / f"lfmc_maui_{year}_08.tif"
        if tif.exists():
            available[year] = tif
        else:
            missing.append(year)

    if not available:
        print("No August TIF files found. Run map_generator first.")
        return

    if missing:
        print(f"Note: Missing Aug {missing} — will show placeholder panels.")

    print(f"Loading coastline...")
    land = get_maui_land()

    extent = [BOUNDS["min_lon"], BOUNDS["max_lon"], BOUNDS["min_lat"], BOUNDS["max_lat"]]

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#0d1117")

    gs = GridSpec(2, 2, figure=fig, hspace=0.08, wspace=0.05)

    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]

    for ax, year in zip(axes, YEARS):
        ax.set_facecolor("#0d1117")

        if year in available:
            print(f"  Plotting Aug {year}...")
            masked = load_and_mask(available[year], land)

            im = ax.imshow(
                masked, cmap=cmap, vmin=0, vmax=180,
                extent=extent, origin="upper", aspect="auto",
                interpolation="bilinear",
            )
            land.boundary.plot(ax=ax, color="white", linewidth=0.6, alpha=0.7)

            # Stats
            valid = masked.compressed()
            if len(valid) > 0:
                mean_val = valid.mean()
                pct_danger = (valid < 80).mean() * 100
                ax.text(
                    0.98, 0.04,
                    f"mean {mean_val:.0f}%  |  {pct_danger:.0f}% danger",
                    transform=ax.transAxes,
                    color="#FFD700" if pct_danger > 20 else "#7a9e9e",
                    fontsize=8, ha="right", va="bottom", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#0d1117", alpha=0.8, ec="none"),
                )

        else:
            # Placeholder
            ax.text(0.5, 0.5, "Map generating...",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#2a4e4e", fontsize=12)

        # Lahaina marker on 2023 panel
        if year == 2023 and year in available:
            ax.plot(LAHAINA_LON, LAHAINA_LAT,
                    marker="*", color="#FF4500", markersize=12, zorder=10,
                    markeredgecolor="white", markeredgewidth=0.5)

        # Year label
        fire_note = "  ★ Lahaina fire" if year == 2023 else ""
        label_color = "#f97316" if year == 2023 else "#e8f4f4"
        ax.set_title(f"Aug {year}{fire_note}", color=label_color,
                     fontsize=13, fontweight="bold", pad=8)

        ax.set_xlim(BOUNDS["min_lon"], BOUNDS["max_lon"])
        ax.set_ylim(BOUNDS["min_lat"], BOUNDS["max_lat"])
        ax.tick_params(colors="#4a6e6e", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1a3232")

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.76])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 180))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("LFMC (%)", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    cbar.ax.axhline(y=80, color="#FFD700", linewidth=1.5, linestyle="--")
    cbar.ax.text(2.5, 80, " 80%\n critical", color="#FFD700",
                 fontsize=7, va="center", transform=cbar.ax.transData)

    fig.suptitle(
        "Live Fuel Moisture Content — Maui County\nAugust 2021–2024",
        color="white", fontsize=15, fontweight="bold", y=0.98,
    )
    fig.text(0.5, 0.01,
             "Galileo-LFMC  |  Zero-shot transfer from CONUS  |  NASA Harvest Internship",
             ha="center", color="#2a4e4e", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.output)
