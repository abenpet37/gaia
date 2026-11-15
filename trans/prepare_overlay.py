"""Prepare the generated deformation GeoTIFF for display in the map page."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import warnings
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import array_bounds, from_origin
from rasterio.warp import calculate_default_transform, reproject, transform_bounds


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RASTER_DIR = ROOT_DIR / "manipulate" / "output"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
OVERLAY_IMAGE_NAME = "deformation_overlay.png"
OVERLAY_CONFIG_NAME = "deformation-overlay.js"
# PNG overlays carry no georeferencing by design; positions come from the config bounds.
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

ACQUISITION_PAIR = re.compile(r"_(\d{8}T\d{6})_(\d{8}T\d{6})_")
ADDIS_BOUNDS = (38.60, 8.80, 38.95, 9.15)
ADDIS_CENTER = (38.757, 8.981)


def print_progress(current: int, total: int, label: str, done: bool = False) -> None:
    """Single-line terminal progress bar."""
    width = 30
    filled = int(width * current / max(1, total))
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{bar}] {current}/{total} {label}", end="\n" if done else "", flush=True)


def _orbit_direction(raster_path: Path) -> str:
    if "_A_" in raster_path.stem:
        return "ASCENDING"
    if "_D_" in raster_path.stem:
        return "DESCENDING"
    raise ValueError(f"Cannot determine orbit direction from {raster_path.name}")


def _acquisition_pair(raster_path: Path) -> tuple[str, str]:
    match = ACQUISITION_PAIR.search(raster_path.stem)
    if match is None:
        raise ValueError(f"Cannot determine acquisition dates from {raster_path.name}")
    return match.group(1)[:8], match.group(2)[:8]


COLOR_SUFFIX = "_los_displacement_color.tif"
NUMERIC_SUFFIX = "_los_displacement_cm.tif"


def _numeric_sibling(color_path: Path) -> Path | None:
    """The float LOS-displacement raster written next to a colour raster, if any."""
    if not color_path.name.endswith(COLOR_SUFFIX):
        return None
    candidate = color_path.with_name(color_path.name[: -len(COLOR_SUFFIX)] + NUMERIC_SUFFIX)
    if not candidate.is_file():
        return None
    try:
        with rasterio.open(candidate) as source:
            if source.count != 1 or source.crs is None:
                return None
    except rasterio.errors.RasterioError:
        return None
    return candidate


def _write_mosaic(
    raster_paths: list[Path],
    image_path: Path,
    max_dimension: int,
    clip_bounds: tuple[float, float, float, float] | None = None,
) -> list[list[float]]:
    # Deduplicate frames (e.g. prefer standard 2000 frame over 4000 narrow swath)
    seen_frames: dict[str, Path] = {}
    for raster_path in raster_paths:
        parts = raster_path.stem.split("_")
        # Frame numbers repeat across tracks, so the key must include the track.
        frame_id = f"{parts[5]}_{parts[7]}" if len(parts) > 7 else raster_path.stem
        variant = parts[9] if len(parts) > 9 else ""
        if frame_id not in seen_frames or variant == "2000":
            seen_frames[frame_id] = raster_path

    valid_candidates: list[tuple[float, tuple[float, float, float, float], Path]] = []
    for raster_path in seen_frames.values():
        with rasterio.open(raster_path) as source:
            if source.count != 4 or source.crs is None:
                raise ValueError(f"{raster_path.name} must be a four-band georeferenced RGBA raster")
            raster_bounds = transform_bounds(
                source.crs, "EPSG:4326", *source.bounds, densify_pts=21
            )
            if clip_bounds is not None:
                west, south, east, north = raster_bounds
                clip_west, clip_south, clip_east, clip_north = clip_bounds
                raster_bounds = (
                    max(west, clip_west),
                    max(south, clip_south),
                    min(east, clip_east),
                    min(north, clip_north),
                )
                if raster_bounds[0] >= raster_bounds[2] or raster_bounds[1] >= raster_bounds[3]:
                    continue
            valid_candidates.append((raster_bounds[1], raster_bounds, raster_path))

    if not valid_candidates:
        raise ValueError("No input rasters overlap the requested mosaic bounds")

    # Sort sequentially along-track by latitude to guarantee adjacent overlap alignment
    valid_candidates.sort(key=lambda item: item[0])

    west = min(c[1][0] for c in valid_candidates)
    south = min(c[1][1] for c in valid_candidates)
    east = max(c[1][2] for c in valid_candidates)
    north = max(c[1][3] for c in valid_candidates)
    resolution = max(east - west, north - south) / max_dimension
    width = max(1, math.ceil((east - west) / resolution))
    height = max(1, math.ceil((north - south) / resolution))
    transform = from_origin(west, north, resolution, resolution)

    # Accumulate continuous scalar displacement and feather weights
    mosaic_scalar = np.zeros((height, width), dtype=np.float32)
    mosaic_weights = np.zeros((height, width), dtype=np.float32)

    total_tiles = len(valid_candidates)
    for tile_index, (_, tile_bounds, raster_path) in enumerate(valid_candidates, start=1):
        # Reproject only into this tile's window of the canvas (much faster on big canvases).
        col0 = max(0, int(math.floor((tile_bounds[0] - west) / resolution)))
        col1 = min(width, int(math.ceil((tile_bounds[2] - west) / resolution)))
        row0 = max(0, int(math.floor((north - tile_bounds[3]) / resolution)))
        row1 = min(height, int(math.ceil((north - tile_bounds[1]) / resolution)))
        if col1 <= col0 or row1 <= row0:
            continue
        win_w, win_h = col1 - col0, row1 - row0
        win_transform = from_origin(
            west + col0 * resolution, north - row0 * resolution, resolution, resolution
        )
        print_progress(tile_index, total_tiles, f"mosaic {image_path.name}")

        numeric_path = _numeric_sibling(raster_path)
        if numeric_path is not None:
            # Physical LOS displacement in cm: comparable across frames and dates, so
            # the mosaic never has to guess a per-frame scale.
            with rasterio.open(numeric_path) as source:
                tile_scalar = np.full((win_h, win_w), np.nan, dtype=np.float32)
                reproject(
                    source=rasterio.band(source, 1),
                    destination=tile_scalar,
                    src_transform=source.transform,
                    src_crs=source.crs,
                    src_nodata=source.nodata if source.nodata is not None else -9999.0,
                    dst_transform=win_transform,
                    dst_crs="EPSG:4326",
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                    init_dest_nodata=True,
                )
            tile_valid = np.isfinite(tile_scalar)
            tile_scalar = np.where(tile_valid, tile_scalar, 0.0).astype(np.float32)
            physical_units = True
        else:
            # No numeric raster (e.g. processed with --color-only): rebuild an approximate
            # scalar from the colour ramp. Each frame was coloured with its own range.
            with rasterio.open(raster_path) as source:
                tile = np.zeros((4, win_h, win_w), dtype=np.uint8)
                for band in range(1, 5):
                    reproject(
                        source=rasterio.band(source, band),
                        destination=tile[band - 1],
                        src_transform=source.transform,
                        src_crs=source.crs,
                        dst_transform=win_transform,
                        dst_crs="EPSG:4326",
                        resampling=Resampling.bilinear if band < 4 else Resampling.nearest,
                        init_dest_nodata=True,
                    )
            tile_valid = tile[3] > 10
            r = tile[0].astype(np.float32)
            g = tile[1].astype(np.float32)
            b = tile[2].astype(np.float32)
            # Reconstruct normalized continuous scalar deformation in [-1, 1]
            tile_scalar = np.where(r >= b, 1.0 - g / 255.0, -(1.0 - g / 255.0)).astype(np.float32)
            tile_scalar[~tile_valid] = 0.0
            physical_units = False

        if not np.any(tile_valid):
            continue

        win = (slice(row0, row1), slice(col0, col1))
        m_scalar = mosaic_scalar[win]      # views: updates write into the full canvas
        m_weights = mosaic_weights[win]

        existing_valid = m_weights > 0.05
        overlap = tile_valid & existing_valid

        # Seamless leveling: every interferogram has its own arbitrary phase reference,
        # so match the offset to what is already in the mosaic across the overlap.
        if np.sum(overlap) > 50:
            m_ov = m_scalar[overlap]
            t_ov = tile_scalar[overlap]
            m_med, t_med = float(np.median(m_ov)), float(np.median(t_ov))
            scale = 1.0
            if not physical_units:
                # Colour-derived values also differ in scale between frames; correct it
                # within a bounded range since it is only an estimate.
                m_spread = float(np.percentile(m_ov, 90) - np.percentile(m_ov, 10))
                t_spread = float(np.percentile(t_ov, 90) - np.percentile(t_ov, 10))
                if t_spread > 1e-6 and m_spread > 1e-6:
                    scale = min(2.0, max(0.5, m_spread / t_spread))
            tile_scalar[tile_valid] = (tile_scalar[tile_valid] - t_med) * scale + m_med

        # Smooth boundary feathering to eliminate rectangular tile seams
        tile_w = np.zeros((win_h, win_w), dtype=np.float32)
        tile_w[tile_valid] = 1.0
        try:
            from scipy.ndimage import distance_transform_edt
            if np.any(tile_valid):
                # Pad with an invalid border so the tile's own straight edges count as edges.
                padded = np.pad(tile_valid, 1, constant_values=False)
                dist = distance_transform_edt(padded)[1:-1, 1:-1]
                # Smoothstep ramp over a wide band gives an invisible seam between frames.
                feather_dist = max(2.0, min(120.0, float(np.max(dist) * 0.5)))
                ramp = np.clip(dist / feather_dist, 0.0, 1.0)
                tile_w = (ramp * ramp * (3.0 - 2.0 * ramp)).astype(np.float32)
                tile_w[tile_valid] = np.maximum(tile_w[tile_valid], 1e-3)
        except Exception:
            pass

        tot_w = m_weights + tile_w
        update_mask = tot_w > 0
        m_scalar[update_mask] = (
            m_scalar[update_mask] * m_weights[update_mask]
            + tile_scalar[update_mask] * tile_w[update_mask]
        ) / tot_w[update_mask]
        m_weights[update_mask] = np.maximum(m_weights[update_mask], tile_w[update_mask])
    print_progress(total_tiles, total_tiles, f"mosaic {image_path.name}", done=True)

    valid_mosaic = mosaic_weights > 0
    if not np.any(valid_mosaic):
        raise ValueError("The mosaic contains no valid pixels")

    # Global normalization for consistent, coherent dynamic range across entire swath
    vals = mosaic_scalar[valid_mosaic]
    p98 = float(np.percentile(np.abs(vals), 98))
    norm_factor = p98 if (math.isfinite(p98) and p98 > 0) else 1.0
    scaled = np.clip(mosaic_scalar / norm_factor, -1.0, 1.0)
    intensity = np.abs(scaled)

    mosaic = np.zeros((4, height, width), dtype=np.uint8)
    mosaic[0] = np.where(scaled >= 0, 255, 255 * (1.0 - intensity)).astype(np.uint8)
    mosaic[1] = (255 * (1.0 - intensity)).astype(np.uint8)
    mosaic[2] = np.where(scaled < 0, 255, 255 * (1.0 - intensity)).astype(np.uint8)
    mosaic[3] = valid_mosaic.astype(np.uint8) * 255

    # Leaflet stretches an image evenly in Web Mercator, but this mosaic has rows evenly
    # spaced in latitude. Resample rows to Mercator spacing so features line up on the map
    # (otherwise they are off by up to ~0.3 degrees of latitude on a large swath).
    def _merc(lat_deg: np.ndarray) -> np.ndarray:
        return np.log(np.tan(np.pi / 4 + np.radians(lat_deg) / 2))

    y_top, y_bottom = _merc(np.float64(north)), _merc(np.float64(south))
    row_y = y_top + (np.arange(height) + 0.5) / height * (y_bottom - y_top)
    row_lat = np.degrees(2 * np.arctan(np.exp(row_y)) - np.pi / 2)
    src_rows = np.clip(np.round((north - row_lat) / resolution - 0.5).astype(int), 0, height - 1)
    mosaic = mosaic[:, src_rows, :]

    with rasterio.open(
        image_path,
        "w",
        driver="PNG",
        width=width,
        height=height,
        count=4,
        dtype="uint8",
    ) as image:
        image.write(mosaic)
    return [[south, west], [north, east]]


def prepare_overlay_mosaic(
    raster_paths: list[Path],
    output_dir: Path,
    max_dimension: int = 1024,
    timeseries_raster_paths: list[Path] | None = None,
    clip_bounds: tuple[float, float, float, float] | None = None,
    full_swath: bool = False,
) -> tuple[list[Path], Path]:
    if not raster_paths:
        raise ValueError("No color rasters were provided for the mosaic")
    if max_dimension < 2:
        raise ValueError("Maximum mosaic dimension must be at least 2 pixels")

    # Ignore stray rasters whose names do not carry NISAR acquisition dates.
    raster_paths = [p for p in raster_paths if ACQUISITION_PAIR.search(p.stem)]
    if not raster_paths:
        raise ValueError("No color rasters with NISAR acquisition dates were provided")
    newest_pair = max(_acquisition_pair(path) for path in raster_paths)
    pair_rasters = [path for path in raster_paths if _acquisition_pair(path) == newest_pair]

    # Look for Addis Ababa scenes across all input rasters
    addis_candidates = []
    for path in raster_paths:
        with rasterio.open(path) as source:
            if source.crs is None:
                continue
            west, south, east, north = transform_bounds(
                source.crs, "EPSG:4326", *source.bounds, densify_pts=21
            )
            covers_center = (
                west <= ADDIS_CENTER[0] <= east
                and south <= ADDIS_CENTER[1] <= north
            )
            intersects_city = (
                west < ADDIS_BOUNDS[2]
                and east > ADDIS_BOUNDS[0]
                and south < ADDIS_BOUNDS[3]
                and north > ADDIS_BOUNDS[1]
            )
            if covers_center and intersects_city:
                addis_candidates.append(path)

    addis_rasters = []
    if addis_candidates:
        newest_addis_pair = max(_acquisition_pair(path) for path in addis_candidates)
        addis_rasters = [
            path for path in addis_candidates
            if _acquisition_pair(path) == newest_addis_pair
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    layers: dict[str, dict[str, object]] = {}
    image_paths: list[Path] = []

    # Addis Ababa city layer (placed first so it is the primary default layer!)
    if addis_rasters:
        directions = {_orbit_direction(path) for path in addis_rasters}
        direction = "DESCENDING" if "DESCENDING" in directions else sorted(directions)[0]
        paths = [
            path for path in addis_rasters
            if _orbit_direction(path) == direction
        ]
        image_name = f"deformation_overlay_addis_{direction.lower()}.png"
        image_path = output_dir / image_name
        image_paths.append(image_path)
        layers["ADDIS ABABA"] = {
            "imageUrl": "../trans/" + image_name,
            "bounds": _write_mosaic(
                paths,
                image_path,
                max_dimension=max_dimension,
                clip_bounds=ADDIS_BOUNDS,
            ),
            "label": "Addis Ababa city detail only (small clipped area)",
            "kind": "pairwise",
        }

    # Regional layers generated only when full_swath is requested or when no city scenes exist
    if full_swath or not addis_rasters:
        if full_swath:
            # Mosaic everything: for every frame (direction, track, frame number) use its most
            # recent acquisition pair. Most frames come from the newest pass of their track; a
            # frame missing from that pass is filled from the next most recent pass that has it.
            def _frame_key(path: Path) -> tuple[str, str, str]:
                parts = path.stem.split("_")
                return (_orbit_direction(path), parts[5], parts[7] if len(parts) > 7 else path.stem)

            latest_per_frame: dict[tuple[str, str, str], tuple[str, str]] = {}
            for path in raster_paths:
                key = _frame_key(path)
                latest_per_frame[key] = max(latest_per_frame.get(key, ("", "")), _acquisition_pair(path))
            regional_rasters = [
                path for path in raster_paths
                if _acquisition_pair(path) == latest_per_frame[_frame_key(path)]
            ]
            regional_dimension = max_dimension
        else:
            regional_rasters = pair_rasters
            regional_dimension = max_dimension
        directions = {_orbit_direction(path) for path in regional_rasters}
        for direction in sorted(directions):
            paths = [p for p in regional_rasters if _orbit_direction(p) == direction]
            image_name = f"deformation_overlay_regional_{direction.lower()}.png"
            image_path = output_dir / image_name
            try:
                bounds = _write_mosaic(
                    paths,
                    image_path,
                    max_dimension=regional_dimension,
                    clip_bounds=clip_bounds,
                )
                image_paths.append(image_path)
                layers[f"REGIONAL - {direction}"] = {
                    "imageUrl": "../trans/" + image_name,
                    "bounds": bounds,
                    "label": f"LOS displacement - {direction.title()} (full rift mosaic)",
                    "kind": "pairwise",
                }
            except ValueError:
                continue

    if not layers:
        raise ValueError("No valid mosaic layer could be generated from the provided rasters")

    if timeseries_raster_paths:
        image_paths.extend(
            _add_timeseries_layers(
                timeseries_raster_paths,
                output_dir,
                layers,
                max_dimension,
                full_swath=full_swath,
            )
        )

    # The map selects the first layer and zooms to it, so list the regional mosaics first;
    # otherwise the page opens on the tiny Addis Ababa tile instead of the whole rift.
    layers = dict(sorted(layers.items(), key=lambda item: 0 if item[0].startswith("REGIONAL") else 1))

    config_path = output_dir / OVERLAY_CONFIG_NAME
    config_path.write_text(
        "window.DEFORMATION_OVERLAY = "
        + json.dumps({"layers": layers}, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    return image_paths, config_path


def prepare_overlay(raster_path: Path, output_dir: Path) -> tuple[Path, Path]:
    if not raster_path.is_file():
        raise FileNotFoundError(f"Raster does not exist: {raster_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / OVERLAY_IMAGE_NAME
    config_path = output_dir / OVERLAY_CONFIG_NAME

    with rasterio.open(raster_path) as source:
        if source.count != 4:
            raise ValueError("Expected a four-band RGBA color raster")
        if source.crs is None:
            raise ValueError("Raster does not have a coordinate reference system")

        destination_crs = "EPSG:4326"
        transform, width, height = calculate_default_transform(
            source.crs,
            destination_crs,
            source.width,
            source.height,
            *source.bounds,
        )
        rgba = np.zeros((4, height, width), dtype=np.uint8)
        for band in range(1, 5):
            reproject(
                source=rasterio.band(source, band),
                destination=rgba[band - 1],
                src_transform=source.transform,
                src_crs=source.crs,
                dst_transform=transform,
                dst_crs=destination_crs,
                resampling=Resampling.nearest,
                init_dest_nodata=True,
            )

    with rasterio.open(
        image_path,
        "w",
        driver="PNG",
        width=width,
        height=height,
        count=4,
        dtype="uint8",
    ) as image:
        image.write(rgba)

    west, south, east, north = array_bounds(height, width, transform)
    overlay_config = {
        "imageUrl": "../trans/" + OVERLAY_IMAGE_NAME,
        "bounds": [[south, west], [north, east]],
    }
    config_path.write_text(
        "window.DEFORMATION_OVERLAY = "
        + json.dumps(overlay_config, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    return image_path, config_path


def _write_velocity_overlay(
    raster_path: Path,
    image_path: Path,
    max_dimension: int,
    clip_bounds: tuple[float, float, float, float] | None = None,
) -> list[list[float]]:
    with rasterio.open(raster_path) as source:
        if source.count != 1 or source.crs is None:
            raise ValueError(
                f"{raster_path.name} must be a single-band georeferenced velocity raster"
            )
        destination_crs = "EPSG:4326"
        transform, width, height = calculate_default_transform(
            source.crs,
            destination_crs,
            source.width,
            source.height,
            *source.bounds,
        )
        if clip_bounds is not None:
            src_bounds = transform_bounds(
                source.crs, destination_crs, *source.bounds, densify_pts=21
            )
            west = max(src_bounds[0], clip_bounds[0])
            south = max(src_bounds[1], clip_bounds[1])
            east = min(src_bounds[2], clip_bounds[2])
            north = min(src_bounds[3], clip_bounds[3])
            if west >= east or south >= north:
                raise ValueError(f"{raster_path.name} does not overlap requested bounds")
            resolution = max(
                abs(transform.a),
                abs(transform.e),
                (east - west) / max_dimension,
                (north - south) / max_dimension,
            )
            width = max(1, math.ceil((east - west) / resolution))
            height = max(1, math.ceil((north - south) / resolution))
            transform = from_origin(west, north, resolution, resolution)
        else:
            scale = min(1.0, max_dimension / max(width, height))
            width = max(1, math.ceil(width * scale))
            height = max(1, math.ceil(height * scale))
            transform, width, height = calculate_default_transform(
                source.crs,
                destination_crs,
                source.width,
                source.height,
                *source.bounds,
                dst_width=width,
                dst_height=height,
            )
        velocity = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(source, 1),
            destination=velocity,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=transform,
            dst_crs=destination_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
            init_dest_nodata=True,
        )

    valid = np.isfinite(velocity)
    if not valid.any():
        raise ValueError(f"{raster_path.name} contains no valid velocity pixels")
    color_range = float(np.percentile(np.abs(velocity[valid]), 98))
    if not math.isfinite(color_range) or color_range <= 0:
        color_range = 1.0
    scaled = np.clip(velocity / color_range, -1.0, 1.0)
    intensity = np.abs(scaled)
    rgba = np.zeros((4, height, width), dtype=np.uint8)
    rgba[0] = np.where(scaled >= 0, 220, 255 * (1 - intensity)).astype(np.uint8)
    rgba[1] = (255 * (1 - intensity)).astype(np.uint8)
    rgba[2] = np.where(scaled < 0, 220, 255 * (1 - intensity)).astype(np.uint8)
    rgba[3] = valid.astype(np.uint8) * 255

    with rasterio.open(
        image_path,
        "w",
        driver="PNG",
        width=width,
        height=height,
        count=4,
        dtype="uint8",
    ) as image:
        image.write(rgba)
    west, south, east, north = array_bounds(height, width, transform)
    return [[south, west], [north, east]]


def _add_timeseries_layers(
    raster_paths: list[Path],
    output_dir: Path,
    layers: dict[str, dict[str, object]],
    max_dimension: int,
    full_swath: bool = False,
) -> list[Path]:
    image_paths: list[Path] = []
    for raster_path in sorted(raster_paths):
        prefix = "_secular_tectonic_velocity_cm_yr"
        if not raster_path.stem.endswith(prefix):
            raise ValueError(
                f"{raster_path.name} is not a secular tectonic velocity raster"
            )
        group_key = raster_path.stem[: -len(prefix)]
        if "_A_" in group_key:
            direction = "ASCENDING"
        elif "_D_" in group_key:
            direction = "DESCENDING"
        else:
            direction = "UNKNOWN"
        with rasterio.open(raster_path) as source:
            if source.crs is None:
                raise ValueError(f"{raster_path.name} has no coordinate reference system")
            bounds = transform_bounds(
                source.crs, "EPSG:4326", *source.bounds, densify_pts=21
            )
        covers_addis = (
            bounds[0] <= ADDIS_CENTER[0] <= bounds[2]
            and bounds[1] <= ADDIS_CENTER[1] <= bounds[3]
        )
        if not covers_addis and not full_swath:
            continue

        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", group_key).strip("_")
        image_name = f"deformation_velocity_{safe_name}.png"
        image_path = output_dir / image_name
        image_paths.append(image_path)
        layer_name = f"TIME SERIES - {direction} - {group_key}"
        clip = ADDIS_BOUNDS if (covers_addis and not full_swath) else None
        layers[layer_name] = {
            "imageUrl": "../trans/" + image_name,
            "bounds": _write_velocity_overlay(
                raster_path,
                image_path,
                max_dimension,
                clip_bounds=clip,
            ),
            "label": f"Time-series velocity ({direction.lower()})",
            "kind": "timeseries",
        }
    return image_paths


def prepare_timeseries_overlay(
    raster_paths: list[Path],
    output_dir: Path,
    max_dimension: int = 4096,
    full_swath: bool = False,
) -> tuple[list[Path], Path]:
    if not raster_paths:
        raise ValueError("No secular velocity rasters were provided")
    if max_dimension < 2:
        raise ValueError("Maximum overlay dimension must be at least 2 pixels")

    output_dir.mkdir(parents=True, exist_ok=True)
    layers: dict[str, dict[str, object]] = {}
    image_paths = _add_timeseries_layers(
        raster_paths,
        output_dir,
        layers,
        max_dimension,
        full_swath=full_swath,
    )
    if not image_paths:
        raise ValueError(
            "No valid time-series velocity rasters found"
        )

    config_path = output_dir / OVERLAY_CONFIG_NAME
    config_path.write_text(
        "window.DEFORMATION_OVERLAY = "
        + json.dumps({"layers": layers}, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    return image_paths, config_path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a generated RGBA deformation raster for the map overlay."
    )
    parser.add_argument(
        "--raster",
        type=Path,
        help="color GeoTIFF to overlay (default: newest color GeoTIFF in manipulate/output)",
    )
    parser.add_argument("--raster-dir", type=Path, default=DEFAULT_RASTER_DIR)
    parser.add_argument(
        "--mosaic",
        action="store_true",
        help="combine the newest matching acquisition pair into separate orbit-direction overlays",
    )
    parser.add_argument(
        "--full-swath",
        action="store_true",
        help="combine all scenes in the newest pair into regional swath mosaics without cropping",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        help="optional west,south,east,north bounding box to clip regional mosaic",
    )
    parser.add_argument(
        "--time-series",
        action="store_true",
        help="include time-series velocity as a selectable map layer (requires --mosaic)",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=None,
        help="maximum width or height for the overlay (default: 1024, or 4096 with --full-swath)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    if args.max_dimension is None:
        # Whole-swath mosaics need more pixels than a city-sized overlay to stay legible.
        args.max_dimension = 4096 if args.full_swath else 1024
    if args.time_series and not args.mosaic:
        print("--time-series requires --mosaic.", file=sys.stderr)
        return 2
    if args.time_series:
        timeseries_rasters = sorted(
            args.raster_dir.glob("*_secular_tectonic_velocity_cm_yr.tif")
        )
        if not timeseries_rasters:
            print(
                f"No time-series velocity rasters found in {args.raster_dir}; "
                "run manipulate/generate_deformation.py --time-series first.",
                file=sys.stderr,
            )
            timeseries_rasters = []
    clip_bounds = None
    if args.bbox:
        try:
            parts = [float(p.strip()) for p in args.bbox.split(",")]
            if len(parts) == 4 and parts[0] < parts[2] and parts[1] < parts[3]:
                clip_bounds = (parts[0], parts[1], parts[2], parts[3])
            else:
                raise ValueError("bbox must be west,south,east,north with west < east and south < north")
        except ValueError as exc:
            print(f"Error parsing --bbox: {exc}", file=sys.stderr)
            return 2

    if args.mosaic:
        rasters = sorted(args.raster_dir.glob("*_los_displacement_color.tif"))
        if not rasters:
            print(
                f"No color GeoTIFFs found in {args.raster_dir}; "
                "run manipulate/generate_deformation.py first.",
                file=sys.stderr,
            )
            return 1
        try:
            image_paths, config_path = prepare_overlay_mosaic(
                rasters,
                DEFAULT_OUTPUT_DIR,
                args.max_dimension,
                timeseries_raster_paths=timeseries_rasters if args.time_series else None,
                clip_bounds=clip_bounds,
                full_swath=args.full_swath,
            )
        except (OSError, rasterio.errors.RasterioError, ValueError) as exc:
            print(f"Error preparing raster mosaics: {exc}", file=sys.stderr)
            return 1
        for image_path in image_paths:
            print(f"Created {image_path}")
        print(f"Created {config_path}")
        return 0

    raster_path = args.raster
    if raster_path is None:
        rasters = sorted(
            args.raster_dir.glob("*_los_displacement_color.tif"),
            key=lambda path: path.stat().st_mtime,
        )
        if not rasters:
            print(
                f"No color GeoTIFFs found in {args.raster_dir}; "
                "run manipulate/generate_deformation.py first.",
                file=sys.stderr,
            )
            return 1
        raster_path = rasters[-1]

    try:
        image_path, config_path = prepare_overlay(raster_path, DEFAULT_OUTPUT_DIR)
    except (OSError, rasterio.errors.RasterioError, ValueError) as exc:
        print(f"Error preparing raster overlay: {exc}", file=sys.stderr)
        return 1

    print(f"Created {image_path}")
    print(f"Created {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
