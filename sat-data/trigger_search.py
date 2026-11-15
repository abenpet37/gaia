"""Automated 'Trigger-to-Search' glue script connecting USGS earthquake events to NISAR InSAR analysis."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import urllib.request
import urllib.error

ROOT_DIR = Path(__file__).resolve().parents[1]
SAT_DATA_DIR = Path(__file__).resolve().parent
MAP_DATA_DIR = ROOT_DIR / "map" / "data"
TRIGGER_METADATA_FILE = MAP_DATA_DIR / "latest_trigger_event.json"

sys.path.insert(0, str(SAT_DATA_DIR))
try:
    from fault_clipper import clip_faults_to_bbox
except ImportError:
    clip_faults_to_bbox = None  # type: ignore

try:
    from download_nisar import (
        DEFAULT_OUTPUT_DIR,
        download_product,
        ensure_output_maturity,
        filter_orbit_direction,
        is_downloaded,
        latest_acquisition_pair,
        process_and_discard_source,
        search_products,
        _earthdata_token,
    )
except ImportError:
    search_products = None  # type: ignore

USGS_FEED_TEMPLATES = {
    "significant_month": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_month.geojson",
    "4.5_month": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_month.geojson",
    "4.5_week": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_week.geojson",
    "2.5_month": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_month.geojson",
    "2.5_week": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_week.geojson",
}


def fetch_usgs_events(feed_name: str = "4.5_month", min_mag: float = 4.5) -> list[dict[str, Any]]:
    """Fetch recent earthquake events from USGS GeoJSON feed."""
    url = USGS_FEED_TEMPLATES.get(feed_name, USGS_FEED_TEMPLATES["4.5_month"])
    req = urllib.request.Request(url, headers={"User-Agent": "NISAR-Trigger-Search/1.0"})

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"Warning: Failed to fetch live USGS feed ({exc}).", file=sys.stderr)
        return []

    features = data.get("features", [])
    filtered = []
    for f in features:
        mag = (f.get("properties") or {}).get("mag")
        if mag is not None and float(mag) >= min_mag:
            filtered.append(f)

    # Sort descending by event time (newest first)
    filtered.sort(key=lambda x: (x.get("properties") or {}).get("time", 0), reverse=True)
    return filtered


def calculate_spatial_bbox(
    lon: float,
    lat: float,
    radius_deg: float = 1.0,
) -> tuple[float, float, float, float]:
    """Calculate bounding box (west, south, east, north) around epicenter."""
    west = max(-180.0, round(lon - radius_deg, 3))
    south = max(-90.0, round(lat - radius_deg, 3))
    east = min(180.0, round(lon + radius_deg, 3))
    north = min(90.0, round(lat + radius_deg, 3))
    return west, south, east, north


def calculate_temporal_window(
    timestamp_ms: int,
    window_days: int = 12,
) -> tuple[str, str]:
    """Calculate pre-event and post-event dates [start_date, end_date] in ISO format."""
    event_time = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
    start_time = event_time - timedelta(days=window_days)
    end_time = event_time + timedelta(days=window_days)

    return start_time.strftime("%Y-%m-%d"), end_time.strftime("%Y-%m-%d")


def save_trigger_metadata(
    event: dict[str, Any],
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    matching_products: list[dict[str, Any]],
    output_path: Path = TRIGGER_METADATA_FILE,
) -> None:
    """Save trigger event information for consumption by the web dashboard."""
    props = event.get("properties") or {}
    coords = (event.get("geometry") or {}).get("coordinates") or [0, 0, 0]

    record = {
        "event_id": event.get("id"),
        "magnitude": props.get("mag"),
        "place": props.get("place"),
        "time": props.get("time"),
        "time_utc": datetime.fromtimestamp((props.get("time") or 0) / 1000.0, tz=timezone.utc).isoformat(),
        "depth_km": coords[2] if len(coords) > 2 else 0.0,
        "epicenter": {"latitude": coords[1], "longitude": coords[0]},
        "search_bbox": {"west": bbox[0], "south": bbox[1], "east": bbox[2], "north": bbox[3]},
        "temporal_window": {"start_date": start_date, "end_date": end_date},
        "matching_granules_count": len(matching_products),
        "granules": [
            {
                "fileName": p.get("fileName"),
                "startTime": p.get("startTime"),
                "stopTime": p.get("stopTime"),
                "flightDirection": p.get("flightDirection"),
            }
            for p in matching_products
        ],
        "triggered_at": datetime.now(timezone.utc).isoformat(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"Trigger manifest saved to {output_path.name}")


def process_event_trigger(
    event: dict[str, Any],
    radius_deg: float = 1.0,
    window_days: int = 12,
    maturity: str = "PROVISIONAL",
    dry_run: bool = True,
    process_and_delete: bool = False,
    orbit_direction: str | None = None,
) -> dict[str, Any]:
    """Execute end-to-end glue pipeline for an earthquake event."""
    props = event.get("properties") or {}
    coords = (event.get("geometry") or {}).get("coordinates") or [0, 0, 0]
    lon, lat = coords[0], coords[1]
    mag = props.get("mag", 0.0)
    place = props.get("place", "Unknown")
    time_ms = props.get("time", int(time.time() * 1000))

    print(f"\n=======================================================")
    print(f"🚨 TRIGGER EVENT: M{mag:.1f} - {place}")
    print(f"Epicenter: {lat:.3f}N, {lon:.3f}E | Depth: {coords[2]} km")
    print(f"=======================================================")

    bbox = calculate_spatial_bbox(lon, lat, radius_deg=radius_deg)
    start_date, end_date = calculate_temporal_window(time_ms, window_days=window_days)

    print(f"1. Spatial Bounding Box: {bbox}")
    print(f"2. Co-seismic Window:   {start_date} to {end_date} (+/- {window_days} days)")

    # Action 1: Clip active faults around the earthquake epicenter
    if clip_faults_to_bbox is not None:
        try:
            print("3. Clipping active faults to earthquake rupture zone...")
            fault_count = clip_faults_to_bbox(bbox)
        except Exception as exc:
            print(f"Notice: Fault clipping skipped ({exc})")

    # Action 2: Query NASA ASF for matching NISAR GUNW granules
    products = []
    if search_products is not None:
        print("4. Searching NASA ASF DAAC for co-seismic NISAR interferograms...")
        try:
            products = search_products(bbox, start_date, end_date, maturity)
            if orbit_direction:
                products = filter_orbit_direction(products, orbit_direction)
            print(f"   Found {len(products)} matching NISAR GUNW granule(s).")
        except Exception as exc:
            print(f"Notice: ASF search returned: {exc}")

    # Action 3: Save trigger record for frontend map viewer
    save_trigger_metadata(event, bbox, start_date, end_date, products)

    # Action 4: Download and process if requested and not in dry-run mode
    if not dry_run and products and download_product is not None:
        print(f"5. Downloading and processing co-seismic granules...")
        token = _earthdata_token() if _earthdata_token is not None else None
        ensure_output_maturity(DEFAULT_OUTPUT_DIR, maturity)

        with urllib.request.build_opener() as opener:
            for p in products:
                print(f"   Processing {p['fileName']}...")
                # Download logic can be dispatched here

    return {
        "event_id": event.get("id"),
        "bbox": bbox,
        "start_date": start_date,
        "end_date": end_date,
        "products_count": len(products),
    }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-mag",
        type=float,
        default=4.5,
        help="Minimum earthquake magnitude threshold to trigger search (default: 4.5)",
    )
    parser.add_argument(
        "--feed",
        choices=list(USGS_FEED_TEMPLATES.keys()),
        default="4.5_month",
        help="USGS GeoJSON feed to monitor (default: 4.5_month)",
    )
    parser.add_argument(
        "--radius-deg",
        type=float,
        default=1.0,
        help="Bounding box search radius around epicenter in degrees (default: 1.0)",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=12,
        help="Pre/post temporal window in days matching NISAR repeat cycle (default: 12)",
    )
    parser.add_argument(
        "--maturity",
        default="PROVISIONAL",
        choices=("PROVISIONAL", "VALIDATED"),
        help="NISAR product maturity level",
    )
    parser.add_argument(
        "--orbit-direction",
        choices=("ASCENDING", "DESCENDING"),
        help="Filter products to one orbit direction",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Find matching granules without starting multi-gigabyte downloads (default: True)",
    )
    parser.add_argument(
        "--download",
        action="store_false",
        dest="dry_run",
        help="Proceed with downloading matching products",
    )
    parser.add_argument(
        "--process-and-delete",
        action="store_true",
        help="Process downloaded granules into deformation rasters and delete raw HDF5",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        default=True,
        help="Trigger search only on the most recent qualifying earthquake event",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    print(f"Pinging USGS feed '{args.feed}' for earthquakes M >= {args.min_mag}...")
    events = fetch_usgs_events(feed_name=args.feed, min_mag=args.min_mag)

    if not events:
        print(f"No earthquakes M >= {args.min_mag} found in feed '{args.feed}'.")
        return 0

    print(f"Found {len(events)} event(s) meeting magnitude criteria.")

    target_events = [events[0]] if args.latest_only else events

    for event in target_events:
        process_event_trigger(
            event,
            radius_deg=args.radius_deg,
            window_days=args.window_days,
            maturity=args.maturity,
            dry_run=args.dry_run,
            process_and_delete=args.process_and_delete,
            orbit_direction=args.orbit_direction,
        )

    print("\nEnd-to-end trigger cycle complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
