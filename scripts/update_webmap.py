"""
Sync finished LFMC TIF files to the web map and update the AVAILABLE set.

Run this after new maps are generated to push them to the web map repo
and update the JavaScript AVAILABLE set so new months unlock automatically.

Usage:
    python scripts/update_webmap.py
    python scripts/update_webmap.py --webmap-dir C:/path/to/maui-lfmc-web
    python scripts/update_webmap.py --dry-run
"""
import argparse
import re
import shutil
from pathlib import Path

REPO = Path(__file__).parent.parent
MAPS_DIR = REPO / "outputs/maps"
DEFAULT_WEBMAP = Path("C:/Users/LazyB/Downloads/maui-lfmc-web")
WEBMAP_DATA = DEFAULT_WEBMAP / "public/data"
INDEX_HTML = DEFAULT_WEBMAP / "public/index.html"


def get_available_tifs():
    """Return list of (year, month, path) for all finished TIFs."""
    result = []
    for tif in sorted(MAPS_DIR.glob("lfmc_maui_????_??.tif")):
        parts = tif.stem.split("_")  # lfmc, maui, YYYY, MM
        if len(parts) == 4:
            year, month = int(parts[2]), int(parts[3])
            result.append((year, month, tif))
    return result


def get_current_available_set(html_content):
    """Parse current AVAILABLE set from index.html."""
    match = re.search(r"const AVAILABLE = new Set\(\[(.*?)\]\)", html_content, re.DOTALL)
    if not match:
        return set()
    items = re.findall(r"'(\d{4}-\d{2})'", match.group(1))
    return set(items)


def build_available_set_str(keys):
    """Render the AVAILABLE set JS string."""
    sorted_keys = sorted(keys)
    items = ", ".join(f"'{k}'" for k in sorted_keys)
    return f"const AVAILABLE = new Set([{items}]);"


def main(webmap_dir, dry_run):
    webmap_data = webmap_dir / "public/data"
    index_html = webmap_dir / "public/index.html"

    tifs = get_available_tifs()
    if not tifs:
        print("No TIF files found in outputs/maps/")
        return

    print(f"Found {len(tifs)} TIF(s) in outputs/maps/")

    # Copy TIFs to web map
    new_keys = set()
    for year, month, src_path in tifs:
        key = f"{year}-{month:02d}"
        new_keys.add(key)
        dst = webmap_data / src_path.name
        if dst.exists():
            print(f"  [skip] {src_path.name} already in web map")
        else:
            if not dry_run:
                webmap_data.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst)
            print(f"  [copy] {src_path.name} -> {dst}")

    # Update AVAILABLE set in index.html
    html = index_html.read_text(encoding="utf-8")
    current_keys = get_current_available_set(html)
    all_keys = current_keys | new_keys

    if all_keys == current_keys:
        print("AVAILABLE set already up to date — no changes needed.")
    else:
        added = all_keys - current_keys
        print(f"  Adding {len(added)} new key(s) to AVAILABLE: {sorted(added)}")

        new_set_str = build_available_set_str(all_keys)
        updated_html = re.sub(
            r"const AVAILABLE = new Set\(\[.*?\]\);",
            new_set_str,
            html,
            flags=re.DOTALL,
        )

        if not dry_run:
            index_html.write_text(updated_html, encoding="utf-8")
            print(f"  Updated {index_html}")
        else:
            print(f"  [dry-run] Would write: {new_set_str}")

    if dry_run:
        print("\nDry run complete — no files written.")
    else:
        print("\nDone. Run 'vercel --prod' from the web map directory to deploy.")
        print(f"  cd {webmap_dir} && vercel --prod")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync TIFs to web map")
    parser.add_argument("--webmap-dir", type=Path, default=DEFAULT_WEBMAP)
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without writing files")
    args = parser.parse_args()
    main(args.webmap_dir, args.dry_run)
