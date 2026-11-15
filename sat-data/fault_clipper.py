"""Clip global active fault databases (GEM / USGS) to an earthquake bounding box."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MASTER_PATH = Path(__file__).resolve().parent / "gem_active_faults_global.geojson"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "map" / "faults_clip.geojson"
DEFAULT_FALLBACK_FAULTS = ROOT_DIR / "map" / "active_faults.json"
GEM_REMOTE_URL = (
    "https://raw.githubusercontent.com/GEMScienceTools/gem-global-active-faults/master/geojson/gem_active_faults.geojson"
)


def ensure_master_database(master_path: Path = DEFAULT_MASTER_PATH, timeout: int = 60) -> Path:
    """Ensure the global fault database is present locally; download if missing."""
    if master_path.is_file() and master_path.stat().st_size > 0:
        return master_path

    # If the fallback active_faults.json exists and master doesn't, we can start with fallback if offline
    print(f"Master fault database not found at {master_path.name}; downloading from GEM...")
    try:
        response = requests.get(GEM_REMOTE_URL, timeout=timeout, stream=True)
        response.raise_for_status()
        master_path.parent.mkdir(parents=True, exist_ok=True)
        with open(master_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)
        print(f"Downloaded master fault database ({master_path.stat().st_size / (1024 * 1024):.1f} MB).")
        return master_path
    except Exception as exc:
        print(f"Warning: Could not download GEM master faults from web ({exc}).")
        if DEFAULT_FALLBACK_FAULTS.is_file():
            print(f"Using local regional fault database {DEFAULT_FALLBACK_FAULTS.name} as fallback.")
            return DEFAULT_FALLBACK_FAULTS
        raise


def _line_intersects_bbox(coordinates: list[Any], west: float, south: float, east: float, north: float) -> bool:
    """Check if any point or segment in coordinate array intersects the bounding box."""
    for item in coordinates:
        if isinstance(item[0], list):
            if _line_intersects_bbox(item, west, south, east, north):
                return True
        else:
            lon, lat = float(item[0]), float(item[1])
            if west <= lon <= east and south <= lat <= north:
                return True
    return False


def _standardize_properties(props: dict[str, Any]) -> dict[str, Any]:
    """Normalize properties from GEM/USGS to match the map dashboard schema."""
    name = props.get("name") or props.get("fault_name") or props.get("FaultName") or "Active Fault"
    fault_type = props.get("type") or props.get("slip_type") or props.get("SlipType") or "Normal / Strike-slip"
    slip_rate = props.get("slip_rate") or props.get("slip_rate_avg") or props.get("SlipRate") or "Active"
    rift_system = props.get("rift_system") or props.get("catalog_name") or props.get("system") or "Global Fault System"
    hazard = props.get("hazard") or props.get("activity_confidence") or "High - Active Tectonic Structure"

    if isinstance(slip_rate, (int, float)):
        slip_rate = f"{slip_rate:.1f} mm/yr"

    return {
        "name": str(name),
        "type": str(fault_type),
        "slip_rate": str(slip_rate),
        "rift_system": str(rift_system),
        "hazard": str(hazard),
    }


def clip_faults_geopandas(
    master_path: Path,
    bbox: tuple[float, float, float, float],
    output_path: Path,
) -> int:
    """Clip master faults using GeoPandas with spatial indexing."""
    import geopandas as gpd  # type: ignore

    west, south, east, north = bbox
    buffer = 0.25
    buffered_bbox = (west - buffer, south - buffer, east + buffer, north + buffer)

    gdf = gpd.read_file(master_path, bbox=buffered_bbox)
    if gdf.empty:
        output_path.write_text('{"type": "FeatureCollection", "features": []}', encoding="utf-8")
        return 0

    rename_map = {
        "fault_name": "name",
        "slip_type": "type",
        "slip_rate_avg": "slip_rate",
        "activity_confidence": "hazard",
    }
    gdf = gdf.rename(columns={k: v for k, v in rename_map.items() if k in gdf.columns})

    for col in ("name", "type", "slip_rate", "rift_system", "hazard"):
        if col not in gdf.columns:
            gdf[col] = "Active"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GeoJSON")
    return len(gdf)


def clip_faults_native(
    master_path: Path,
    bbox: tuple[float, float, float, float],
    output_path: Path,
) -> int:
    """Clip master faults using fast native JSON streaming without heavy geospatial C-dependencies."""
    west, south, east, north = bbox
    buffer = 0.25
    b_west, b_south, b_east, b_north = west - buffer, south - buffer, east + buffer, north + buffer

    with open(master_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])
    clipped_features = []

    for feat in features:
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates") or []
        if _line_intersects_bbox(coords, b_west, b_south, b_east, b_north):
            standard_props = _standardize_properties(feat.get("properties") or {})
            clipped_features.append({
                "type": "Feature",
                "properties": standard_props,
                "geometry": geom,
            })

    output_fc = {
        "type": "FeatureCollection",
        "name": "Clipped_Active_Faults",
        "features": clipped_features,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_fc, f, indent=2)

    return len(clipped_features)


def clip_faults_to_bbox(
    bbox: tuple[float, float, float, float],
    output_path: Path = DEFAULT_OUTPUT_PATH,
    master_path: Path = DEFAULT_MASTER_PATH,
) -> int:
    """Clip active fault database to bounding box (west, south, east, north)."""
    if not master_path.is_file():
        try:
            ensure_master_database(master_path)
        except Exception:
            if DEFAULT_FALLBACK_FAULTS.is_file():
                master_path = DEFAULT_FALLBACK_FAULTS

    try:
        count = clip_faults_geopandas(master_path, bbox, output_path)
    except ImportError:
        count = clip_faults_native(master_path, bbox, output_path)

    print(f"Clipped {count} active fault segment(s) to {output_path.name}")
    return count


def parse_bbox_string(value: str) -> tuple[float, float, float, float]:
    parts = [float(p.strip()) for p in value.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must contain four coordinates: west,south,east,north")
    west, south, east, north = parts
    if west >= east or south >= north:
        raise ValueError("bbox must be ordered west < east and south < north")
    return west, south, east, north


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", required=True, help="Bounding box as west,south,east,north in decimal degrees")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output GeoJSON path")
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER_PATH, help="Master global fault database path")
    args = parser.parse_args(argv)

    try:
        bbox = parse_bbox_string(args.bbox)
        count = clip_faults_to_bbox(bbox, output_path=args.output, master_path=args.master)
        print(f"Success: {count} faults written to {args.output}")
        return 0
    except Exception as exc:
        print(f"Error clipping faults: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
