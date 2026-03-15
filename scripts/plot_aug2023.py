"""
Improved August 2023 LFMC map visualization.
- Ocean pixels masked using Natural Earth 10m land polygon
- Accurate geographic extent (lat/lon axes)
- Natural Earth coastline overlay
- Lahaina fire origin marker
- Critical threshold annotation
"""
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — save to file only
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import rasterio
from rasterio.features import geometry_mask
import urllib.request
import zipfile
import tempfile
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
REPO = Path(__file__).parent.parent
TIF = REPO / "outputs/maps/lfmc_maui_2023_08.tif"
OUT = REPO / "outputs/plots/lfmc_maui_2023_08_v2.png"

# ── Maui bounds ───────────────────────────────────────────────────────────────
BOUNDS = dict(min_lat=20.5, max_lat=21.1, min_lon=-156.7, max_lon=-155.9)

# Lahaina fire origin (Lahainaluna Rd area)
LAHAINA_LON, LAHAINA_LAT = -156.677, 20.875

# ── colormap ──────────────────────────────────────────────────────────────────
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
cmap = LinearSegmentedColormap.from_list(
    "lfmc", [(p, c) for p, c in CMAP_COLORS]
)
cmap.set_bad(color="none")  # masked pixels → transparent


def get_maui_land():
    """Return Natural Earth 10m land GeoDataFrame clipped to Maui bounds."""
    import geopandas as gpd
    import shapely.geometry as geom

    ne_candidates = [
        Path("C:/Users/LazyB/AppData/Local/Temp/ne_10m_land/ne_10m_land.shp"),
        Path.home() / "ne_10m_land/ne_10m_land.shp",
        Path(tempfile.gettempdir()) / "ne_10m_land/ne_10m_land.shp",
        REPO / "docs/ne_10m_land/ne_10m_land.shp",
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


# ── load TIF ──────────────────────────────────────────────────────────────────
with rasterio.open(TIF) as src:
    data = src.read(1).astype(np.float32)
    transform = src.transform
    crs = src.crs
    height, width = data.shape

nodata = -9999.0

# ── mask ocean pixels using land polygon ──────────────────────────────────────
land = get_maui_land()

# Burn land polygon into a boolean mask (True = land, False = ocean)
from rasterio.features import geometry_mask
land_shapes = [geom for geom in land.geometry]
ocean_mask = geometry_mask(
    land_shapes,
    transform=transform,
    invert=False,   # True where NOT land = ocean
    out_shape=(height, width),
)

# Combine: mask nodata + ocean
masked = np.ma.masked_where(
    (data == nodata) | np.isnan(data) | ocean_mask,
    data,
)

# Image extent: [left, right, bottom, top]
extent = [BOUNDS["min_lon"], BOUNDS["max_lon"], BOUNDS["min_lat"], BOUNDS["max_lat"]]

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

# LFMC raster (ocean pixels are transparent)
im = ax.imshow(
    masked,
    cmap=cmap,
    vmin=0, vmax=180,
    extent=extent,
    origin="upper",
    aspect="auto",
    interpolation="bilinear",
)

# Coastline boundary
land.boundary.plot(ax=ax, color="white", linewidth=0.8, alpha=0.8)

# Lahaina marker
ax.plot(
    LAHAINA_LON, LAHAINA_LAT,
    marker="*", color="#FF4500", markersize=14, zorder=10,
    markeredgecolor="white", markeredgewidth=0.5,
)
ax.annotate(
    "  Lahaina\n  Fire Origin",
    xy=(LAHAINA_LON, LAHAINA_LAT),
    xytext=(LAHAINA_LON + 0.12, LAHAINA_LAT - 0.07),
    color="white", fontsize=9, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="white", lw=0.7),
    bbox=dict(boxstyle="round,pad=0.2", fc="#FF4500", alpha=0.7, ec="none"),
)

# Critical threshold annotation
ax.text(
    0.98, 0.04,
    "\u26a0 <80% LFMC = Critical Fire Danger",
    transform=ax.transAxes,
    color="#FFD700", fontsize=8, ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a1a", alpha=0.8, ec="#FFD700", lw=0.6),
)

# Axes styling
ax.set_xlim(BOUNDS["min_lon"], BOUNDS["max_lon"])
ax.set_ylim(BOUNDS["min_lat"], BOUNDS["max_lat"])
ax.set_xlabel("Longitude", color="#aaaaaa", fontsize=10)
ax.set_ylabel("Latitude", color="#aaaaaa", fontsize=10)
ax.tick_params(colors="#aaaaaa", labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor("#333333")

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
cbar.set_label("LFMC (%)", color="white", fontsize=11)
cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
cbar.ax.axhline(y=80, color="#FFD700", linewidth=1.5, linestyle="--")

# Title
ax.set_title(
    "Live Fuel Moisture Content \u2014 Maui County\nAugust 2023  (month of Lahaina fire)",
    color="white", fontsize=13, fontweight="bold", pad=12,
)

# Model note
ax.text(
    0.01, 0.01,
    "Galileo-LFMC  |  Zero-shot transfer from CONUS",
    transform=ax.transAxes,
    color="#666666", fontsize=7.5, ha="left", va="bottom",
)

plt.tight_layout()
plt.savefig(OUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {OUT}")
plt.close()
