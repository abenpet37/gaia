"""Process the latest NISAR pass of every track over the East African Rift and mosaic it.

Streams only the needed HDF5 data from ASF (no full downloads), writes displacement
GeoTIFFs to manipulate/output, then rebuilds the map overlay as one regional mosaic.

    python sat-data/process_rift.py --direction ASCENDING --clean
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import download_nisar as nisar  # noqa: E402

# Afar triangle (N) to Mozambique rift (S)
RIFT_BBOX = (28.0, -20.0, 45.0, 15.0)


def select_latest_per_track(products: list[dict], direction: str | None) -> list[dict]:
    """Keep every frame from the most recent acquisition day of each (direction, track)."""
    def key(p: dict) -> tuple[str, str]:
        parts = p["fileName"].split("_")
        return parts[6], parts[5]

    latest: dict[tuple[str, str], str] = {}
    for p in products:
        day = p["startTime"][:10]
        latest[key(p)] = max(latest.get(key(p), ""), day)
    selected = [p for p in products if p["startTime"][:10] == latest[key(p)]]
    if direction:
        selected = [p for p in selected if key(p)[0] == direction[0].upper()]
    return selected


def bar(done: int, total: int, started: float, width: int = 30) -> str:
    filled = int(width * done / max(1, total))
    elapsed = time.time() - started
    eta = (elapsed / done) * (total - done) if done else 0
    return (
        f"\r[{'#' * filled}{'-' * (width - filled)}] {done}/{total} "
        f"({done * 100 // max(1, total)}%)  elapsed {int(elapsed // 60)}m  eta {int(eta // 60)}m   "
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=("ASCENDING", "DESCENDING"), help="default: both")
    parser.add_argument("--start", default="2026-09-01")
    parser.add_argument("--end", default="2026-11-15")
    parser.add_argument("--maturity", default="PROVISIONAL", choices=("VALIDATED", "PROVISIONAL", "BETA"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--clean", action="store_true", help="delete existing rasters in manipulate/output first")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    products = nisar.search_products(RIFT_BBOX, args.start, args.end, args.maturity)
    selected = select_latest_per_track(products, args.direction)
    print(f"{len(products)} granules found; {len(selected)} selected (latest pass per track).")
    if args.dry_run:
        return 0

    out_dir = ROOT_DIR / "manipulate" / "output"
    if args.clean:
        removed = 0
        for f in out_dir.glob("*.tif"):
            f.unlink()
            removed += 1
        print(f"Removed {removed} existing raster(s).")

    started = time.time()
    print(bar(0, len(selected), started), end="", flush=True)

    def on_granule_done(finished: int, total: int, filename: str) -> None:
        print(bar(finished, total, started), end="", flush=True)

    ok, errors = nisar.remote_process_products(
        selected, nisar._earthdata_token(), workers=args.workers, on_granule_done=on_granule_done
    )
    print(f"\n{ok} succeeded, {len(errors)} failed.")
    for err in errors:
        print("  FAILED:", err[:200])

    subprocess.run(
        [sys.executable, str(ROOT_DIR / "trans" / "prepare_overlay.py"), "--mosaic", "--full-swath"],
        cwd=ROOT_DIR,
        check=False,
    )
    return 1 if errors and not ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
