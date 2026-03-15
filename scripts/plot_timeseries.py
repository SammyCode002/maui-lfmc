"""
Monthly mean LFMC time series for Maui County (2023).

Reads all available monthly TIFs and plots:
  - Mean LFMC per month
  - % of land below 80% critical threshold per month
  - Lahaina fire date marker

Usage:
    python scripts/plot_timeseries.py
    python scripts/plot_timeseries.py --year 2023 --output outputs/plots/timeseries_2023.png
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime

REPO = Path(__file__).parent.parent
MAPS_DIR = REPO / "outputs/maps"
DEFAULT_OUT = REPO / "outputs/plots/timeseries_2023.png"

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

# Lahaina fire: Aug 8, 2023
LAHAINA_DATE = datetime(2023, 8, 8)


def load_tif_stats(tif_path):
    """Return (mean, min, max, pct_below_80) for a TIF, masking nodata."""
    import rasterio
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)

    valid = data[(data != -9999) & ~np.isnan(data) & (data > 0)]
    if len(valid) == 0:
        return None

    return {
        "mean": float(valid.mean()),
        "min":  float(valid.min()),
        "max":  float(valid.max()),
        "pct_danger": float((valid < 80).mean() * 100),
        "n_pixels": len(valid),
    }


def main(year, output_path):
    months_data = {}
    for month in range(1, 13):
        mm = f"{month:02d}"
        tif = MAPS_DIR / f"lfmc_maui_{year}_{mm}.tif"
        if tif.exists():
            stats = load_tif_stats(tif)
            if stats:
                months_data[month] = stats
                print(f"  {MONTH_NAMES[month-1]} {year}: mean={stats['mean']:.1f}%  danger={stats['pct_danger']:.1f}%")

    if not months_data:
        print(f"No TIF files found for {year}. Run map_generator first.")
        return

    months = sorted(months_data.keys())
    dates = [datetime(year, m, 15) for m in months]
    means = [months_data[m]["mean"] for m in months]
    mins  = [months_data[m]["min"]  for m in months]
    maxs  = [months_data[m]["max"]  for m in months]
    dangers = [months_data[m]["pct_danger"] for m in months]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.patch.set_facecolor("#0d1117")

    for ax in (ax1, ax2):
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#7a9e9e", labelsize=9)
        ax.grid(True, color="#1a3232", linewidth=0.6, alpha=0.8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1a3232")

    # ── Top panel: Mean LFMC ──────────────────────────────────────────────────
    ax1.fill_between(dates, mins, maxs, color="#32CD32", alpha=0.12, label="Min–Max range")
    ax1.plot(dates, means, "o-", color="#32CD32", linewidth=2,
             markersize=6, markeredgecolor="white", markeredgewidth=0.5, label="Mean LFMC")
    ax1.axhline(80, color="#FFD700", linewidth=1.2, linestyle="--", alpha=0.8,
                label="80% critical threshold")

    # Lahaina marker
    if datetime(year, 1, 1) <= LAHAINA_DATE <= datetime(year, 12, 31):
        ax1.axvline(LAHAINA_DATE, color="#FF4500", linewidth=1.5, linestyle=":",
                    alpha=0.9, zorder=5)
        ax1.text(LAHAINA_DATE, 195,
                 " ★ Lahaina\n   Aug 8",
                 color="#FF4500", fontsize=8, va="top", fontweight="bold")

    ax1.set_ylabel("LFMC (%)", color="#7a9e9e", fontsize=10)
    ax1.set_ylim(0, 200)
    ax1.legend(fontsize=8, facecolor="#091414", edgecolor="#1a3232",
               labelcolor="#7a9e9e", loc="upper left")
    ax1.set_title(f"Monthly LFMC — Maui County {year}",
                  color="white", fontsize=13, fontweight="bold", pad=10)

    # ── Bottom panel: % below critical ───────────────────────────────────────
    bar_colors = ["#FF4500" if d > 30 else "#FFD700" if d > 15 else "#32CD32"
                  for d in dangers]
    ax2.bar(dates, dangers, width=25, color=bar_colors, alpha=0.8, zorder=3)
    ax2.axhline(30, color="#FF4500", linewidth=1, linestyle="--", alpha=0.6)

    if datetime(year, 1, 1) <= LAHAINA_DATE <= datetime(year, 12, 31):
        ax2.axvline(LAHAINA_DATE, color="#FF4500", linewidth=1.5,
                    linestyle=":", alpha=0.9, zorder=5)

    ax2.set_ylabel("% land below 80% LFMC", color="#7a9e9e", fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.set_xlabel("Month", color="#7a9e9e", fontsize=10)

    fig.text(0.5, 0.01,
             "Galileo-LFMC  |  Zero-shot transfer from CONUS  |  NASA Harvest Internship",
             ha="center", color="#2a4e4e", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.year, args.output)
