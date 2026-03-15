"""
Convenience script to generate LFMC maps for multiple years.

Runs map_generator sequentially for each year/month combination,
skipping any TIF that already exists.

Usage:
    # All months for a single year
    python scripts/run_all_years.py --years 2024

    # All months for multiple years
    python scripts/run_all_years.py --years 2021 2022 2024 2025

    # August only across multiple years (for multi-year comparison figure)
    python scripts/run_all_years.py --years 2021 2022 2024 --months 8

    # Dry run to see what would be generated
    python scripts/run_all_years.py --years 2023 2024 --dry-run
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
MAPS_DIR = REPO / "outputs/maps"
CHECKPOINT = REPO / "checkpoints/conus/finetuned_model.pth"
GALILEO_CONFIG = Path("C:/Users/LazyB/Downloads/allenai-lfmc/submodules/galileo/data")
GEE_PROJECT = "ace-shine-392702"
PYTHON = sys.executable

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate LFMC maps for multiple years")
    parser.add_argument("--years", nargs="+", type=int, required=True,
                        help="Years to generate (e.g. 2021 2022 2024)")
    parser.add_argument("--months", nargs="+", type=int,
                        default=list(range(1, 13)),
                        help="Months to generate (default: all 12). E.g. --months 8")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be run without executing")
    args = parser.parse_args()

    todo = []
    skipped = []

    for year in sorted(args.years):
        for month in sorted(args.months):
            mm = f"{month:02d}"
            out = MAPS_DIR / f"lfmc_maui_{year}_{mm}.tif"
            if out.exists():
                skipped.append(f"{year}-{mm}")
            else:
                todo.append((year, month))

    if skipped:
        logger.info("Skipping %d already-generated maps: %s", len(skipped), skipped)

    if not todo:
        logger.info("Nothing to generate — all maps already exist.")
        return

    logger.info("Will generate %d map(s): %s",
                len(todo), [f"{y}-{m:02d}" for y, m in todo])

    if args.dry_run:
        logger.info("Dry run — exiting without generating.")
        return

    for i, (year, month) in enumerate(todo, 1):
        logger.info("[%d/%d] Generating %d-%02d...", i, len(todo), year, month)
        cmd = [
            PYTHON, "-m", "src.inference.map_generator",
            "--checkpoint", str(CHECKPOINT),
            "--galileo-config", str(GALILEO_CONFIG),
            "--year", str(year),
            "--month", str(month),
            "--project", GEE_PROJECT,
            "--output", str(MAPS_DIR),
        ]
        result = subprocess.run(cmd, cwd=REPO)
        if result.returncode != 0:
            logger.error("Failed for %d-%02d — continuing with next.", year, month)
        else:
            logger.info("Done: %d-%02d", year, month)

    logger.info("All done. Run 'python scripts/update_webmap.py' to sync to the web map.")


if __name__ == "__main__":
    main()
