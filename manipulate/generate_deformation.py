"""Convert NISAR GUNW unwrapped phase into georeferenced displacement rasters."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime
import json
import math
from collections import deque
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds, transform_geom
from rasterio.windows import Window
import rasterio.features

try:
    from scipy.ndimage import binary_dilation
except ImportError:
    binary_dilation = None


DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_FAULTS_PATH = Path(__file__).resolve().parent.parent / "map" / "active_faults.json"
SPEED_OF_LIGHT_M_S = 299_792_458.0
GRID_PATH = "science/LSAR/GUNW/grids/frequency{frequency}"
RADAR_GRID_PATH = "science/LSAR/GUNW/metadata/radarGrid"
ACQUISITION_TIMES = re.compile(
    r"_(\d{8}T\d{6})_(\d{8}T\d{6})_"
    r"(\d{8}T\d{6})_(\d{8}T\d{6})_"
)
TIMESERIES_MIN_SPAN_DAYS = 365


def _read_scalar(dataset: h5py.Dataset, name: str) -> float:
    value = np.asarray(dataset[()]).item()
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{dataset.name} has an invalid {name}: {value!r}")
    return result


def _coordinate_spacing(values: np.ndarray, name: str) -> float:
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain at least two finite coordinates")
    differences = np.diff(values)
    spacing = float(np.median(differences))
    tolerance = max(abs(spacing) * 1e-6, 1e-8)
    if spacing == 0 or not np.allclose(differences, spacing, rtol=1e-6, atol=tolerance):
        raise ValueError(f"{name} must be regularly spaced and strictly monotonic")
    return spacing


def _fill_value(dataset: h5py.Dataset) -> float | None:
    value = dataset.attrs.get("_FillValue")
    if value is None:
        return None
    return float(np.asarray(value).item())


def _valid_pixels(
    phase: np.ndarray,
    coherence: np.ndarray,
    phase_fill: float | None,
    coherence_fill: float | None,
    min_coherence: float,
) -> np.ndarray:
    valid = np.isfinite(phase) & np.isfinite(coherence)
    if phase_fill is not None and math.isfinite(phase_fill):
        valid &= phase != phase_fill
    if coherence_fill is not None and math.isfinite(coherence_fill):
        valid &= coherence != coherence_fill
    valid &= coherence >= min_coherence
    return valid


def apply_multilooking(
    array: np.ndarray,
    factor: int,
    nodata: float | None = None,
    is_mask: bool = False,
) -> np.ndarray:
    """Perform spatial block downsampling/multilooking to suppress phase noise, increase interferometric SNR, and accelerate computation.

    Averages blocks of size (factor x factor). Handles both 2D (height, width) and 3D (bands, height, width) arrays.
    Trims array boundaries that are not exact multiples of factor.
    Properly handles nodata and non-finite (NaN/Inf) values by averaging only valid pixels within each block.
    If all pixels in a block are invalid/nodata, assigns nodata (or NaN if nodata is None).
    """
    if factor <= 1:
        return array.copy()
    if factor < 1:
        raise ValueError("Multilook factor must be a positive integer >= 1")

    if array.ndim == 3:
        multilooked_slices = [
            apply_multilooking(array[b], factor=factor, nodata=nodata, is_mask=is_mask)
            for b in range(array.shape[0])
        ]
        return np.stack(multilooked_slices, axis=0)

    if array.ndim != 2:
        raise ValueError(f"Multilooking requires 2D or 3D array, got ndim={array.ndim}")

    height, width = array.shape
    h_trim = (height // factor) * factor
    w_trim = (width // factor) * factor
    if h_trim == 0 or w_trim == 0:
        raise ValueError(
            f"Array dimensions ({height}, {width}) are smaller than multilook factor {factor}"
        )

    trimmed = array[:h_trim, :w_trim]
    new_h = h_trim // factor
    new_w = w_trim // factor

    if is_mask:
        blocks = (trimmed > 0).reshape(new_h, factor, new_w, factor)
        true_counts = blocks.sum(axis=(1, 3))
        return (true_counts >= (factor * factor / 2)).astype(array.dtype)

    valid_mask = np.isfinite(trimmed)
    if nodata is not None and math.isfinite(nodata):
        valid_mask &= (trimmed != nodata)

    valid_blocks = valid_mask.reshape(new_h, factor, new_w, factor)
    valid_counts = valid_blocks.sum(axis=(1, 3))

    values_zeroed = np.where(valid_mask, trimmed, 0.0)
    values_blocks = values_zeroed.reshape(new_h, factor, new_w, factor)
    values_sum = values_blocks.sum(axis=(1, 3))

    fill_val = nodata if nodata is not None else np.nan
    result = np.full((new_h, new_w), fill_val, dtype=np.float32)
    has_valid = valid_counts > 0
    result[has_valid] = (values_sum[has_valid] / valid_counts[has_valid]).astype(np.float32)
    return result


def update_transform_multilook(
    transform: rasterio.Affine,
    factor: int,
) -> rasterio.Affine:
    """Update affine geotransform for multilooked raster preserving top-left origin."""
    if factor <= 1:
        return transform
    return rasterio.Affine(
        transform.a * factor,
        transform.b * factor,
        transform.c,
        transform.d * factor,
        transform.e * factor,
        transform.f,
    )


def _resolve_polarization(
    frequency_group: h5py.Group,
    polarization: str,
) -> str:
    interferogram = frequency_group["unwrappedInterferogram"]
    available = [
        pol
        for pol in ("HH", "VV")
        if pol in interferogram and "unwrappedPhase" in interferogram[pol]
    ]
    if polarization == "auto":
        if not available:
            raise ValueError("GUNW product contains no HH or VV unwrapped phase dataset")
        return "HH" if "HH" in available else "VV"
    if polarization not in available:
        raise ValueError(
            f"GUNW product does not contain {polarization} unwrapped phase data"
        )
    return polarization


def _colorize(displacement: np.ndarray, valid: np.ndarray, color_range_cm: float) -> np.ndarray:
    safe_disp = np.nan_to_num(displacement, nan=0.0)
    scaled = np.clip(safe_disp / color_range_cm, -1.0, 1.0)
    intensity = np.abs(scaled)
    rgba = np.zeros((4, *displacement.shape), dtype=np.uint8)
    rgba[0] = np.where(scaled >= 0, 255, 255 * (1 - intensity)).astype(np.uint8)
    rgba[1] = (255 * (1 - intensity)).astype(np.uint8)
    rgba[2] = np.where(scaled < 0, 255, 255 * (1 - intensity)).astype(np.uint8)
    rgba[3] = valid.astype(np.uint8) * 255
    return rgba


def _sample_color_range(
    phase_dataset: h5py.Dataset,
    coherence_dataset: h5py.Dataset,
    wavelength_m: float,
    min_coherence: float,
    phase_fill: float | None,
    coherence_fill: float | None,
    phase_correction: np.ndarray | None = None,
) -> float:
    height, width = phase_dataset.shape
    stride = max(1, math.ceil(max(height, width) / 512))
    phase = np.asarray(phase_dataset[::stride, ::stride], dtype=np.float32)
    coherence = np.asarray(coherence_dataset[::stride, ::stride], dtype=np.float32)
    valid = _valid_pixels(
        phase, coherence, phase_fill, coherence_fill, min_coherence
    )
    if phase_correction is not None:
        correction = phase_correction[::stride, ::stride]
        valid &= np.isfinite(correction)
        phase = phase - np.nan_to_num(correction, nan=0.0)
    if not valid.any():
        raise ValueError("No valid pixels remain after applying the coherence mask")
    displacement = phase[valid] * (wavelength_m / (4 * math.pi)) * 100
    color_range = float(np.percentile(np.abs(displacement), 98))
    return color_range if color_range > 0 else 1.0


def _ascending_axis(values: np.ndarray, name: str) -> tuple[np.ndarray, bool]:
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain at least two finite coordinates")
    differences = np.diff(values)
    if np.all(differences > 0):
        return values, False
    if np.all(differences < 0):
        return values[::-1], True
    raise ValueError(f"{name} must be strictly monotonic")


def _bracket(axis: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, ...]:
    inside = np.isfinite(values) & (values >= axis[0]) & (values <= axis[-1])
    upper = np.searchsorted(axis, values, side="right")
    upper = np.clip(upper, 1, axis.size - 1)
    lower = upper - 1
    fraction = (values - axis[lower]) / (axis[upper] - axis[lower])
    return lower, upper, fraction, inside


def _interpolate_phase_screen(
    screen: np.ndarray,
    height_axis: np.ndarray,
    y_axis: np.ndarray,
    x_axis: np.ndarray,
    dem: np.ndarray,
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
) -> np.ndarray:
    height, width = dem.shape
    output = np.full((height, width), np.nan, dtype=np.float32)
    x_grid = np.broadcast_to(x_coordinates[None, :], (height, width))

    for row_start in range(0, height, 32):
        row_stop = min(row_start + 32, height)
        row_dem = dem[row_start:row_stop]
        y_grid = np.broadcast_to(
            y_coordinates[row_start:row_stop, None], row_dem.shape
        )
        x_block = x_grid[row_start:row_stop]
        z0, z1, wz, valid_z = _bracket(height_axis, row_dem)
        y0, y1, wy, valid_y = _bracket(y_axis, y_grid)
        x0, x1, wx, valid_x = _bracket(x_axis, x_block)
        valid = valid_z & valid_y & valid_x

        interpolated = np.zeros(row_dem.shape, dtype=np.float32)
        invalid_corner = np.zeros(row_dem.shape, dtype=bool)
        for iz, wz_part in ((z0, 1 - wz), (z1, wz)):
            for iy, wy_part in ((y0, 1 - wy), (y1, wy)):
                for ix, wx_part in ((x0, 1 - wx), (x1, wx)):
                    weight = wz_part * wy_part * wx_part
                    value = screen[iz, iy, ix]
                    finite = np.isfinite(value)
                    invalid_corner |= (weight > 0) & ~finite
                    interpolated += np.where(finite, value, 0) * weight
        output[row_start:row_stop] = np.where(
            valid & ~invalid_corner, interpolated, np.nan
        )
    return output


def _read_dem_grid(
    dem_path: Path,
    transform: rasterio.Affine,
    epsg: int,
    height: int,
    width: int,
) -> np.ndarray:
    if not dem_path.is_file():
        raise ValueError(f"DEM file does not exist: {dem_path}")
    dem_grid = np.full((height, width), np.nan, dtype=np.float32)
    with rasterio.open(dem_path) as dem_source:
        if dem_source.count != 1:
            raise ValueError("DEM must contain exactly one elevation band")
        if dem_source.crs is None:
            raise ValueError("DEM must have a defined coordinate reference system")
        reproject(
            source=rasterio.band(dem_source, 1),
            destination=dem_grid,
            src_transform=dem_source.transform,
            src_crs=dem_source.crs,
            src_nodata=dem_source.nodata,
            dst_transform=transform,
            dst_crs=f"EPSG:{epsg}",
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    if not np.isfinite(dem_grid).any():
        raise ValueError("DEM does not overlap the GUNW grid")
    return dem_grid


def _has_valid_embedded_troposphere(product: h5py.File, epsg: int) -> bool:
    if RADAR_GRID_PATH not in product:
        return False
    radar_grid = product[RADAR_GRID_PATH]
    required = (
        "heightAboveEllipsoid",
        "xCoordinates",
        "yCoordinates",
        "projection",
        "wetTroposphericPhaseScreen",
        "hydrostaticTroposphericPhaseScreen",
    )
    if any(name not in radar_grid for name in required):
        return False
    try:
        radar_epsg_val = radar_grid["projection"].attrs.get("epsg_code")
        if radar_epsg_val is None:
            radar_epsg_val = radar_grid["projection"][()]
        if int(np.asarray(radar_epsg_val).item()) != epsg:
            return False
        h_ds = radar_grid["heightAboveEllipsoid"]
        y_ds = radar_grid["yCoordinates"]
        x_ds = radar_grid["xCoordinates"]
        expected_shape = (h_ds.size, y_ds.size, x_ds.size)
        wet = radar_grid["wetTroposphericPhaseScreen"]
        hyd = radar_grid["hydrostaticTroposphericPhaseScreen"]
        if wet.shape != expected_shape or hyd.shape != expected_shape:
            return False
        for ds in (wet, hyd):
            units = ds.attrs.get("units", "")
            if isinstance(units, bytes):
                units = units.decode("ascii", errors="replace")
            if str(units).strip().lower() not in ("rad", "radian", "radians"):
                return False
        return True
    except Exception:
        return False


def _empirical_phase_elevation_correction(
    phase: np.ndarray,
    coherence: np.ndarray,
    dem_oriented: np.ndarray,
    phase_fill: float | None = None,
    coherence_fill: float | None = None,
    min_coherence: float = 0.7,
    max_outlier_mad: float = 2.5,
) -> tuple[np.ndarray, float, float]:
    """Perform robust empirical linear phase-elevation regression and return tropospheric screen.

    Extracts high-coherence background pixels (coherence >= min_coherence), excludes localized
    deformation zones using median absolute deviation (MAD), fits Δφ = a*h + b using Huber IRLS,
    and returns (phi_tropo, a, b).
    """
    valid = (
        np.isfinite(phase)
        & np.isfinite(coherence)
        & np.isfinite(dem_oriented)
        & (coherence >= min_coherence)
    )
    if phase_fill is not None and math.isfinite(phase_fill):
        valid &= (phase != phase_fill)
    if coherence_fill is not None and math.isfinite(coherence_fill):
        valid &= (coherence != coherence_fill)

    # Fallback to available valid pixels if high-coherence pixels are sparse
    if np.sum(valid) < 4:
        valid = (
            np.isfinite(phase)
            & np.isfinite(coherence)
            & np.isfinite(dem_oriented)
            & (coherence >= 0.3)
        )
        if phase_fill is not None and math.isfinite(phase_fill):
            valid &= (phase != phase_fill)
        if coherence_fill is not None and math.isfinite(coherence_fill):
            valid &= (coherence != coherence_fill)

    if np.sum(valid) < 2:
        raise ValueError(
            "Insufficient valid overlapping pixels between interferogram and DEM to fit empirical tropospheric correction"
        )

    phi_cand = phase[valid].astype(np.float64)
    h_cand = dem_oriented[valid].astype(np.float64)

    # Exclude localized deformation zones and unwrapping blunders using robust MAD
    med_phi = float(np.median(phi_cand))
    mad_phi = float(np.median(np.abs(phi_cand - med_phi)))
    scale_phi = 1.4826 * mad_phi
    if scale_phi > 1e-4:
        bg_mask = np.abs(phi_cand - med_phi) <= max_outlier_mad * scale_phi
        if np.sum(bg_mask) >= 2:
            phi_cand = phi_cand[bg_mask]
            h_cand = h_cand[bg_mask]

    # Robust linear regression: phi = a * h + b using Huber Iteratively Reweighted Least Squares
    h_range = float(np.ptp(h_cand))
    if h_range < 1e-3:
        a = 0.0
        b = float(np.median(phi_cand))
    else:
        X = np.column_stack((h_cand, np.ones(h_cand.size, dtype=np.float64)))
        y = phi_cand
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        for _ in range(5):
            res = y - X @ beta
            mad = float(np.median(np.abs(res - np.median(res))))
            sigma = max(1.4826 * mad, 1e-5)
            u = res / sigma
            k = 1.345
            w = np.where(np.abs(u) <= k, 1.0, k / np.maximum(np.abs(u), 1e-8))
            W = np.sqrt(w)
            beta = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)[0]
        a = float(beta[0])
        b = float(beta[1])

    phi_tropo = np.where(np.isfinite(dem_oriented), a * dem_oriented + b, np.nan).astype(np.float32)
    return phi_tropo, a, b


def _read_tropospheric_phase_correction(
    product: h5py.File,
    dem_path: Path,
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    transform: rasterio.Affine,
    epsg: int,
    flip_x: bool,
    flip_y: bool,
) -> np.ndarray:
    radar_grid_path = RADAR_GRID_PATH
    required = (
        "heightAboveEllipsoid",
        "xCoordinates",
        "yCoordinates",
        "projection",
        "wetTroposphericPhaseScreen",
        "hydrostaticTroposphericPhaseScreen",
    )
    missing = [
        f"{radar_grid_path}/{name}"
        for name in required
        if f"{radar_grid_path}/{name}" not in product
    ]
    if missing:
        raise ValueError(
            "Cannot apply the embedded tropospheric correction; product is "
            f"missing {', '.join(missing)}"
        )
    if not dem_path.is_file():
        raise ValueError(f"DEM file does not exist: {dem_path}")

    radar_grid = product[radar_grid_path]
    radar_epsg_value = radar_grid["projection"].attrs.get("epsg_code")
    if radar_epsg_value is None:
        radar_epsg_value = radar_grid["projection"][()]
    radar_epsg = int(np.asarray(radar_epsg_value).item())
    if radar_epsg != epsg:
        raise ValueError(
            f"Radar-grid EPSG:{radar_epsg} does not match GUNW EPSG:{epsg}"
        )

    height_axis, flip_height = _ascending_axis(
        np.asarray(radar_grid["heightAboveEllipsoid"][:], dtype=np.float64),
        "radar-grid heightAboveEllipsoid",
    )
    radar_x, flip_radar_x = _ascending_axis(
        np.asarray(radar_grid["xCoordinates"][:], dtype=np.float64),
        "radar-grid xCoordinates",
    )
    radar_y, flip_radar_y = _ascending_axis(
        np.asarray(radar_grid["yCoordinates"][:], dtype=np.float64),
        "radar-grid yCoordinates",
    )
    expected_shape = (height_axis.size, radar_y.size, radar_x.size)
    screens = []
    for name in ("wetTroposphericPhaseScreen", "hydrostaticTroposphericPhaseScreen"):
        dataset = radar_grid[name]
        units = dataset.attrs.get("units", "")
        if isinstance(units, bytes):
            units = units.decode("ascii", errors="replace")
        if str(units).strip().lower() not in ("rad", "radian", "radians"):
            raise ValueError(
                f"{dataset.name} must be expressed in radians; found units {units!r}"
            )
        if dataset.shape != expected_shape:
            raise ValueError(
                f"{dataset.name} has shape {dataset.shape}; expected "
                f"(height, y, x)={expected_shape}"
            )
        screen = np.asarray(dataset[:], dtype=np.float32)
        if flip_height:
            screen = screen[::-1, :, :]
        if flip_radar_y:
            screen = screen[:, ::-1, :]
        if flip_radar_x:
            screen = screen[:, :, ::-1]
        screens.append(screen)

    dem_grid = _read_dem_grid(
        dem_path, transform, epsg, y_coordinates.size, x_coordinates.size
    )

    correction = np.zeros_like(dem_grid, dtype=np.float32)
    valid = np.isfinite(dem_grid)
    for screen in screens:
        component = _interpolate_phase_screen(
            screen,
            height_axis,
            radar_y,
            radar_x,
            dem_grid,
            x_coordinates,
            y_coordinates,
        )
        valid &= np.isfinite(component)
        correction += np.nan_to_num(component, nan=0.0)
    correction[~valid] = np.nan
    if not np.isfinite(correction).any():
        raise ValueError(
            "The DEM and GUNW grid do not overlap the embedded phase-screen coverage"
        )

    if flip_x:
        correction = correction[:, ::-1]
    if flip_y:
        correction = correction[::-1, :]
    return correction


def _compute_tropospheric_phase_correction(
    product: h5py.File,
    dem_path: Path,
    phase_dataset: h5py.Dataset,
    coherence_dataset: h5py.Dataset,
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    transform: rasterio.Affine,
    epsg: int,
    flip_x: bool,
    flip_y: bool,
    phase_fill: float | None = None,
    coherence_fill: float | None = None,
) -> tuple[np.ndarray, dict[str, str]]:
    """Compute tropospheric phase screen using embedded 3D weather screens or empirical fallback."""
    if not dem_path.is_file():
        raise ValueError(f"DEM file does not exist: {dem_path}")

    height, width = y_coordinates.size, x_coordinates.size

    # Check if HDF5 metadata contains valid embedded 3D weather screens
    has_embedded = _has_valid_embedded_troposphere(product, epsg)
    if has_embedded:
        try:
            correction = _read_tropospheric_phase_correction(
                product=product,
                dem_path=dem_path,
                x_coordinates=x_coordinates,
                y_coordinates=y_coordinates,
                transform=transform,
                epsg=epsg,
                flip_x=flip_x,
                flip_y=flip_y,
            )
            tags = {
                "tropospheric_correction": (
                    "Embedded GUNW wet and hydrostatic phase screens, interpolated "
                    "to the geocoded DEM height"
                ),
                "troposphere_method": "embedded_weather_screens",
                "troposphere_sign_convention": (
                    "Screens subtracted as requested; verify sign for the GUNW product version"
                ),
            }
            return correction, tags
        except Exception as exc:
            print(
                f"Notice: Embedded tropospheric screen could not be applied ({exc}); "
                "falling back to empirical phase-elevation correction."
            )

    # Empirical Phase-Elevation Fallback
    dem_grid = _read_dem_grid(dem_path, transform, epsg, height, width)
    dem_oriented = dem_grid.copy()
    if flip_x:
        dem_oriented = dem_oriented[:, ::-1]
    if flip_y:
        dem_oriented = dem_oriented[::-1, :]

    phase = np.asarray(phase_dataset[:], dtype=np.float32)
    coherence = np.asarray(coherence_dataset[:], dtype=np.float32)

    phi_tropo, slope, intercept = _empirical_phase_elevation_correction(
        phase=phase,
        coherence=coherence,
        dem_oriented=dem_oriented,
        phase_fill=phase_fill,
        coherence_fill=coherence_fill,
        min_coherence=0.7,
    )

    tags = {
        "tropospheric_correction": (
            f"Empirical linear phase-elevation model: phi_tropo = {slope:.6e} * h + {intercept:.4f} (rad)"
        ),
        "troposphere_method": "empirical_phase_elevation",
        "troposphere_phase_elevation_slope_rad_m": f"{slope:.6e}",
        "troposphere_phase_elevation_intercept_rad": f"{intercept:.4f}",
        "troposphere_sign_convention": (
            "Empirical screen subtracted: phi_corrected = unwrappedPhase - (a*h + b)"
        ),
    }
    return phi_tropo, tags


def _parse_pair_raster(path: Path) -> tuple[str, datetime, datetime]:
    match = ACQUISITION_TIMES.search(path.stem)
    if match is None:
        raise ValueError(
            f"Cannot determine both acquisition intervals from {path.name}"
        )
    timestamps = [
        datetime.strptime(value, "%Y%m%dT%H%M%S")
        for value in match.groups()
    ]
    reference_time = timestamps[0] + (timestamps[1] - timestamps[0]) / 2
    secondary_time = timestamps[2] + (timestamps[3] - timestamps[2]) / 2
    if reference_time == secondary_time:
        raise ValueError(f"{path.name} has identical reference and secondary times")
    return path.stem[: match.start()], reference_time, secondary_time


def _read_aligned_window(
    source: rasterio.DatasetReader,
    destination: np.ndarray,
    transform: rasterio.Affine,
    crs: rasterio.crs.CRS,
) -> np.ndarray:
    destination.fill(np.nan)
    reproject(
        source=rasterio.band(source, 1),
        destination=destination,
        src_transform=source.transform,
        src_crs=source.crs,
        src_nodata=source.nodata,
        dst_transform=transform,
        dst_crs=crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
        init_dest_nodata=True,
    )
    return destination


def _baseline_matrix(
    pairs: list[tuple[int, int]], date_count: int
) -> np.ndarray:
    matrix = np.zeros((len(pairs), date_count - 1), dtype=np.float64)
    for row, (reference, secondary) in enumerate(pairs):
        if reference:
            matrix[row, reference - 1] = -1
        if secondary:
            matrix[row, secondary - 1] = 1
    return matrix


def _network_observations(
    datasets: list[rasterio.DatasetReader],
    window: Window,
    transform: rasterio.Affine,
    crs: rasterio.crs.CRS,
) -> tuple[np.ndarray, np.ndarray]:
    tile_transform = rasterio.windows.transform(window, transform)
    observations = np.empty(
        (len(datasets), int(window.height), int(window.width)), dtype=np.float32
    )
    for index, source in enumerate(datasets):
        _read_aligned_window(source, observations[index], tile_transform, crs)
    valid = np.isfinite(observations)
    return observations, valid


def _choose_reference_pixel(
    datasets: list[rasterio.DatasetReader],
    transform: rasterio.Affine,
    crs: rasterio.crs.CRS,
    baseline: np.ndarray,
    width: int,
    height: int,
    block_size: int = 128,
    blunder_masks: dict[int, np.ndarray] | None = None,
) -> tuple[int, int, np.ndarray]:
    candidates: list[tuple[int, float, int, int, tuple[int, ...]]] = []
    for row in range(0, height, block_size):
        for column in range(0, width, block_size):
            window = Window(
                column,
                row,
                min(block_size, width - column),
                min(block_size, height - row),
            )
            observations, valid = _network_observations(
                datasets, window, transform, crs
            )
            if blunder_masks:
                win_h, win_w = int(window.height), int(window.width)
                for edge_idx in range(len(datasets)):
                    if edge_idx in blunder_masks:
                        mask_block = blunder_masks[edge_idx][
                            row : row + win_h, column : column + win_w
                        ]
                        valid[edge_idx] &= ~mask_block
            counts = valid.sum(axis=0)
            if not counts.any():
                continue
            mean_motion = np.divide(
                np.where(valid, np.abs(observations), 0).sum(axis=0),
                counts,
                out=np.full(counts.shape, np.inf, dtype=np.float64),
                where=counts > 0,
            )
            patterns, inverse = np.unique(
                valid.reshape(len(datasets), -1).T,
                axis=0,
                return_inverse=True,
            )
            flat_motion = mean_motion.ravel()
            for pattern_index, pattern in enumerate(patterns):
                edge_indices = tuple(np.flatnonzero(pattern))
                if (
                    not edge_indices
                    or np.linalg.matrix_rank(baseline[list(edge_indices)])
                    != baseline.shape[1]
                ):
                    continue
                pixels = np.flatnonzero(inverse == pattern_index)
                flat_index = int(pixels[np.argmin(flat_motion[pixels])])
                local_row, local_column = np.unravel_index(flat_index, counts.shape)
                score = (
                    len(edge_indices),
                    float(mean_motion[local_row, local_column]),
                    row + int(local_row),
                    column + int(local_column),
                    edge_indices,
                )
                candidates.append(score)

    candidates.sort(key=lambda item: (-item[0], item[1]))
    for _, _, row, column, edge_indices in candidates:
        local_baseline = baseline[list(edge_indices)]
        if np.linalg.matrix_rank(local_baseline) != baseline.shape[1]:
            continue
        reference_values = np.empty(len(datasets), dtype=np.float64)
        reference_values.fill(np.nan)
        pixel_window = Window(column, row, 1, 1)
        for edge_index in edge_indices:
            source = datasets[edge_index]
            value = np.full((1, 1), np.nan, dtype=np.float32)
            _read_aligned_window(
                source,
                value,
                rasterio.windows.transform(pixel_window, transform),
                crs,
            )
            reference_values[edge_index] = float(value[0, 0])
        return row, column, reference_values
    raise ValueError(
        "No pixel has a connected, full-rank interferogram network; "
        "check spatial overlap and coherence masks"
    )


def _invert_network_tile(
    observations: np.ndarray,
    valid: np.ndarray,
    baseline: np.ndarray,
    reference_values: np.ndarray,
    model: np.ndarray,
    pinv_model: np.ndarray | None = None,
    pinv_cache: dict[tuple[int, ...], np.ndarray | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_count, tile_height, tile_width = observations.shape
    pixel_count = tile_height * tile_width
    flat_observations = observations.reshape(edge_count, pixel_count)
    flat_valid = valid.reshape(edge_count, pixel_count).copy()
    flat_valid &= np.isfinite(reference_values)[:, None]
    flat_outputs = [
        np.full(pixel_count, -9999.0, dtype=np.float32) for _ in range(3)
    ]
    if pinv_model is None:
        pinv_model = np.linalg.pinv(model)

    patterns, inverse = np.unique(flat_valid.T, axis=0, return_inverse=True)
    for pattern_index, pattern in enumerate(patterns):
        pixels = np.flatnonzero(inverse == pattern_index)
        edge_indices = np.flatnonzero(pattern)
        if edge_indices.size < baseline.shape[1]:
            continue
        pattern_key = tuple(edge_indices.tolist())
        pinv_design = None
        if pinv_cache is not None and pattern_key in pinv_cache:
            pinv_design = pinv_cache[pattern_key]
            if pinv_design is None:
                continue
        else:
            design = baseline[edge_indices]
            if np.linalg.matrix_rank(design) != baseline.shape[1]:
                if pinv_cache is not None:
                    pinv_cache[pattern_key] = None
                continue
            pinv_design = np.linalg.pinv(design)
            if pinv_cache is not None:
                pinv_cache[pattern_key] = pinv_design

        values = (
            flat_observations[np.ix_(edge_indices, pixels)]
            - reference_values[edge_indices, None]
        )
        cumulative = pinv_design @ values
        cumulative = np.vstack((np.zeros((1, pixels.size)), cumulative))
        coefficients = pinv_model @ cumulative
        fitted = model @ coefficients
        residual = cumulative - fitted
        flat_outputs[0][pixels] = coefficients[1].astype(np.float32)
        flat_outputs[1][pixels] = np.sqrt(
            np.sum(coefficients[2:6] ** 2, axis=0)
        ).astype(np.float32)
        flat_outputs[2][pixels] = np.sqrt(
            np.mean(residual**2, axis=0)
        ).astype(np.float32)
    return tuple(
        output.reshape(tile_height, tile_width) for output in flat_outputs
    )


def find_interferogram_triplets(
    edges: list[tuple[int, int]]
) -> list[tuple[int, int, int, int, int, int]]:
    """Identify closed triangular loops across dates (A -> B, B -> C, A -> C).

    Returns a list of tuples: (A, B, C, idx_AB, idx_BC, idx_AC)
    where A < B < C are date indices and idx_* are edge indices in the edge list.
    """
    edge_map = {edge: idx for idx, edge in enumerate(edges)}
    triplets = []
    for (a, b), idx_ab in edge_map.items():
        for (b2, c), idx_bc in edge_map.items():
            if b == b2 and (a, c) in edge_map:
                idx_ac = edge_map[(a, c)]
                triplets.append((a, b, c, idx_ab, idx_bc, idx_ac))
    return triplets


def validate_temporal_triplets(
    ordered_items: list[tuple[Path, datetime, datetime]],
    datasets: list[rasterio.DatasetReader],
    edges: list[tuple[int, int]],
    dates: list[datetime],
    first_transform: rasterio.Affine,
    first_crs: rasterio.crs.CRS,
    width: int,
    height: int,
    wavelength_m: float,
    block_size: int = 128,
    closure_threshold_rad: float = 1.8 * math.pi,
    flag_blunder_ratio: float = 0.25,
    output_dir: Path | None = None,
    group_key: str = "network",
) -> tuple[
    list[tuple[Path, datetime, datetime]],
    list[rasterio.DatasetReader],
    list[tuple[int, int]],
    np.ndarray,
    dict[int, np.ndarray],
    dict[str, object],
]:
    """Validate temporal triplet phase-closure residuals before SBAS matrix inversion.

    Identifies closed loops (A -> B, B -> C, A -> C) and evaluates:
        Δφ_loop = φ_AB + φ_BC - φ_AC
    Masks out pixels exceeding closure_threshold_rad and flags/drops corrupted pairs
    while preserving the full-rank connectivity (rank == len(dates) - 1) of matrix B.
    """
    triplets = find_interferogram_triplets(edges)
    closure_threshold_cm = closure_threshold_rad * (100.0 * wavelength_m / (4.0 * math.pi))

    if not triplets:
        baseline = _baseline_matrix(edges, len(dates))
        summary = {
            "group_key": group_key,
            "triplets_total": 0,
            "message": "Baseline network is an open tree without closed loops.",
            "pairs_total": len(ordered_items),
            "pairs_flagged": 0,
            "dropped_pairs": [],
            "downweighted_pairs": [],
            "active_pairs_count": len(edges),
            "network_rank": int(np.linalg.matrix_rank(baseline)),
        }
        print(f"[{group_key}] 0 closed triplets found; baseline network is an open spanning tree.")
        return ordered_items, datasets, edges, baseline, {}, summary

    num_edges = len(ordered_items)
    edge_triplets_count = np.zeros(num_edges, dtype=np.int64)
    edge_triplets_failed = np.zeros(num_edges, dtype=np.int64)
    edge_blunder_pixel_mask = np.zeros((num_edges, height, width), dtype=bool)

    triplet_stats = []
    for a, b, c, idx_ab, idx_bc, idx_ac in triplets:
        edge_triplets_count[idx_ab] += 1
        edge_triplets_count[idx_bc] += 1
        edge_triplets_count[idx_ac] += 1
        triplet_stats.append({
            "dates": (dates[a].strftime("%Y-%m-%d"), dates[b].strftime("%Y-%m-%d"), dates[c].strftime("%Y-%m-%d")),
            "indices": (a, b, c),
            "edge_indices": (idx_ab, idx_bc, idx_ac),
            "pair_names": (ordered_items[idx_ab][0].name, ordered_items[idx_bc][0].name, ordered_items[idx_ac][0].name),
            "valid_pixels": 0,
            "blunder_pixels": 0,
            "max_closure_rad": 0.0,
            "mean_closure_rad": 0.0,
            "sum_closure_rad": 0.0,
            "failed": False,
        })

    rad_per_cm = (4.0 * math.pi) / (100.0 * wavelength_m)
    for row in range(0, height, block_size):
        for col in range(0, width, block_size):
            win_w = min(block_size, width - col)
            win_h = min(block_size, height - row)
            window = Window(col, row, win_w, win_h)
            observations, valid = _network_observations(
                datasets, window, first_transform, first_crs
            )

            for t_idx, (_, _, _, m_ab, m_bc, m_ac) in enumerate(triplets):
                overlap = valid[m_ab] & valid[m_bc] & valid[m_ac]
                if not overlap.any():
                    continue

                d_loop = (
                    observations[m_ab][overlap]
                    + observations[m_bc][overlap]
                    - observations[m_ac][overlap]
                )
                phi_loop_rad = d_loop * rad_per_cm
                abs_phi_rad = np.abs(phi_loop_rad)
                blunder_submask = abs_phi_rad > closure_threshold_rad

                st = triplet_stats[t_idx]
                valid_count = int(overlap.sum())
                blunder_count = int(blunder_submask.sum())
                st["valid_pixels"] += valid_count
                st["blunder_pixels"] += blunder_count
                st["sum_closure_rad"] += float(abs_phi_rad.sum())
                st["max_closure_rad"] = max(st["max_closure_rad"], float(abs_phi_rad.max()))

                if blunder_count > 0:
                    tile_blunder = np.zeros((win_h, win_w), dtype=bool)
                    tile_blunder[overlap] = blunder_submask
                    edge_blunder_pixel_mask[m_ab, row : row + win_h, col : col + win_w] |= tile_blunder
                    edge_blunder_pixel_mask[m_bc, row : row + win_h, col : col + win_w] |= tile_blunder
                    edge_blunder_pixel_mask[m_ac, row : row + win_h, col : col + win_w] |= tile_blunder

    for t_idx, st in enumerate(triplet_stats):
        st["mean_closure_rad"] = (
            st["sum_closure_rad"] / st["valid_pixels"]
            if st["valid_pixels"] > 0
            else 0.0
        )
        st["blunder_ratio"] = (
            st["blunder_pixels"] / st["valid_pixels"]
            if st["valid_pixels"] > 0
            else 0.0
        )
        st["failed"] = st["blunder_ratio"] >= flag_blunder_ratio
        if st["failed"]:
            idx_ab, idx_bc, idx_ac = st["edge_indices"]
            edge_triplets_failed[idx_ab] += 1
            edge_triplets_failed[idx_bc] += 1
            edge_triplets_failed[idx_ac] += 1

    candidate_corrupted_indices: list[tuple[float, float, float, int]] = []
    edge_evaluations = []
    for e_idx in range(num_edges):
        n_part = int(edge_triplets_count[e_idx])
        n_fail = int(edge_triplets_failed[e_idx])
        fail_rate = n_fail / n_part if n_part > 0 else 0.0
        part_stats = [
            st for st in triplet_stats if e_idx in st["edge_indices"]
        ]
        mean_blunder = (
            float(np.mean([st["blunder_ratio"] for st in part_stats]))
            if part_stats
            else 0.0
        )
        item = ordered_items[e_idx]
        span_days = (item[2] - item[1]).total_seconds() / 86400.0
        is_corrupted = (n_part > 0) and (fail_rate >= 0.5) and (mean_blunder >= flag_blunder_ratio)
        if is_corrupted:
            candidate_corrupted_indices.append((fail_rate, mean_blunder, span_days, e_idx))

        edge_evaluations.append({
            "index": e_idx,
            "filename": ordered_items[e_idx][0].name,
            "ref_date": ordered_items[e_idx][1].strftime("%Y-%m-%d"),
            "sec_date": ordered_items[e_idx][2].strftime("%Y-%m-%d"),
            "span_days": round(span_days, 1),
            "triplets_participated": n_part,
            "triplets_failed": n_fail,
            "triplet_failure_rate": fail_rate,
            "mean_blunder_rate": mean_blunder,
            "is_corrupted": is_corrupted,
            "action": "KEEP",
        })

    candidate_corrupted_indices.sort(key=lambda x: (-x[0], -x[1], -x[2]))
    dropped_indices: set[int] = set()
    downweighted_indices: set[int] = set()
    current_edge_indices = list(range(num_edges))
    unresolved_failed_triplets = {
        t_idx for t_idx, st in enumerate(triplet_stats) if st["failed"]
    }

    for _, _, _, e_idx in candidate_corrupted_indices:
        edge_failed_triplets = {
            t_idx for t_idx in unresolved_failed_triplets
            if e_idx in triplet_stats[t_idx]["edge_indices"]
        }
        if not edge_failed_triplets:
            edge_evaluations[e_idx]["action"] = "KEEP"
            continue

        test_indices = [i for i in current_edge_indices if i != e_idx]
        test_edges = [edges[i] for i in test_indices]
        test_baseline = _baseline_matrix(test_edges, len(dates))
        if np.linalg.matrix_rank(test_baseline) == len(dates) - 1:
            dropped_indices.add(e_idx)
            current_edge_indices = test_indices
            edge_evaluations[e_idx]["action"] = "DROPPED"
            unresolved_failed_triplets.difference_update(edge_failed_triplets)
        else:
            downweighted_indices.add(e_idx)
            edge_evaluations[e_idx]["action"] = "DOWN-WEIGHTED_AND_PIXEL_MASKED"

    keep_indices = [i for i in range(num_edges) if i not in dropped_indices]
    active_items = [ordered_items[i] for i in keep_indices]
    active_datasets = [datasets[i] for i in keep_indices]
    active_edges = [edges[i] for i in keep_indices]
    active_baseline = _baseline_matrix(active_edges, len(dates))

    active_blunder_masks: dict[int, np.ndarray] = {}
    for new_idx, old_idx in enumerate(keep_indices):
        if old_idx in downweighted_indices and np.any(edge_blunder_pixel_mask[old_idx]):
            active_blunder_masks[new_idx] = edge_blunder_pixel_mask[old_idx]

    summary = {
        "group_key": group_key,
        "wavelength_m": wavelength_m,
        "closure_threshold_rad": closure_threshold_rad,
        "closure_threshold_cm": round(closure_threshold_cm, 3),
        "triplets_total": len(triplets),
        "triplets_failed": int(sum(1 for st in triplet_stats if st["failed"])),
        "pairs_total": num_edges,
        "pairs_flagged": len(dropped_indices) + len(downweighted_indices),
        "dropped_pairs": [ordered_items[i][0].name for i in sorted(dropped_indices)],
        "downweighted_pairs": [ordered_items[i][0].name for i in sorted(downweighted_indices)],
        "active_pairs_count": len(active_edges),
        "network_rank": int(np.linalg.matrix_rank(active_baseline)),
        "triplet_details": triplet_stats,
        "pair_evaluations": edge_evaluations,
    }

    # Print diagnostic summary log
    print("=" * 80)
    print(f"TEMPORAL TRIPLET PHASE-CLOSURE VALIDATION REPORT: {group_key}")
    print("=" * 80)
    print(f"Network Topology: {len(dates)} dates, {num_edges} pairs")
    print(f"Carrier Wavelength: {wavelength_m:.4f} m | Closure Threshold: {closure_threshold_rad:.3f} rad ({closure_threshold_cm:.2f} cm)")
    print(f"Triplets Analyzed: {len(triplets)} closed loops ({summary['triplets_failed']} failed)")
    for idx, st in enumerate(triplet_stats, 1):
        status = "FAIL" if st["failed"] else "PASS"
        print(
            f"  Loop {idx:2d} ({st['dates'][0]} -> {st['dates'][1]} -> {st['dates'][2]}): "
            f"Valid={st['valid_pixels']:d} px, Mean|dphi|={st['mean_closure_rad']:.3f} rad, "
            f"Max|dphi|={st['max_closure_rad']:.3f} rad, BlunderRate={st['blunder_ratio']:.1%} [{status}]"
        )
    if candidate_corrupted_indices:
        print(f"Flagged Corrupted Pairs ({len(candidate_corrupted_indices)}):")
        for _, _, _, e_idx in candidate_corrupted_indices:
            ev = edge_evaluations[e_idx]
            print(
                f"  * {ev['filename']} ({ev['ref_date']} -> {ev['sec_date']}): "
                f"Participated in {ev['triplets_participated']} loops ({ev['triplets_failed']} failed, "
                f"mean blunder rate {ev['mean_blunder_rate']:.1%}) -> ACTION: {ev['action']}"
            )
    else:
        print("Flagged Corrupted Pairs: None (All closed loops verified within closure threshold)")
    print(f"Inversion Network: {len(active_edges)} of {num_edges} pairs active (Rank {summary['network_rank']} preserved)")
    print("=" * 80)

    if output_dir is not None:
        summary_path = output_dir / f"{group_key}_triplet_closure_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    return (
        active_items,
        active_datasets,
        active_edges,
        active_baseline,
        active_blunder_masks,
        summary,
    )


def compute_stable_background_mask(
    displacement: np.ndarray,
    nodata: float = -9999.0,
    transform: rasterio.Affine | None = None,
    crs: rasterio.crs.CRS | str | None = None,
    faults_path: Path | None = None,
    fault_buffer_px: int = 5,
    gradient_mad_threshold: float = 2.5,
    outlier_mad_threshold: float = 3.0,
) -> np.ndarray:
    """Identify stable background pixels for spatial ramp estimation.

    Excludes:
      1. Water bodies, unwrapping voids, and missing data (nodata or non-finite).
      2. High-gradient deformation zones (spatial gradient magnitude > median + 2.5*MAD).
      3. Active fault zones (rasterized and dilated from fault GeoJSON, and high-MAD outliers).
    """
    valid = np.isfinite(displacement)
    if nodata is not None and math.isfinite(nodata):
        valid &= (displacement != nodata)

    if np.sum(valid) < 3:
        return valid

    # 1. High-gradient deformation screening (finite differencing on valid background)
    disp_filled = np.where(valid, displacement, np.nan)
    med_val = float(np.nanmedian(displacement[valid]))
    disp_filled = np.nan_to_num(disp_filled, nan=med_val)

    grad_y, grad_x = np.gradient(disp_filled)
    grad_mag = np.hypot(grad_x, grad_y)

    grad_valid = grad_mag[valid]
    if grad_valid.size > 0:
        med_grad = float(np.median(grad_valid))
        mad_grad = float(np.median(np.abs(grad_valid - med_grad)))
        scale_grad = 1.4826 * mad_grad
        if scale_grad > 1e-5:
            high_grad_mask = grad_mag > (med_grad + gradient_mad_threshold * scale_grad)
        else:
            high_grad_mask = np.zeros_like(valid, dtype=bool)
    else:
        high_grad_mask = np.zeros_like(valid, dtype=bool)

    # 2. Localized deformation zone screening using robust MAD
    d_valid = displacement[valid]
    med_d = float(np.median(d_valid))
    mad_d = float(np.median(np.abs(d_valid - med_d)))
    scale_d = 1.4826 * mad_d
    if scale_d > 1e-4:
        outlier_disp_mask = np.abs(displacement - med_d) > (outlier_mad_threshold * scale_d)
    else:
        outlier_disp_mask = np.zeros_like(valid, dtype=bool)

    # 3. Active fault zones screening (vector GeoJSON)
    fault_mask = np.zeros(displacement.shape, dtype=bool)
    candidate_fault_path = faults_path if faults_path is not None else DEFAULT_FAULTS_PATH
    if (
        candidate_fault_path is not None
        and candidate_fault_path.is_file()
        and transform is not None
        and crs is not None
    ):
        try:
            with open(candidate_fault_path, "r", encoding="utf-8") as f:
                geojson_data = json.load(f)
            features = geojson_data.get("features", [])
            geoms_to_rasterize = []
            height, width = displacement.shape

            for feat in features:
                geom = feat.get("geometry")
                if not geom:
                    continue
                try:
                    proj_geom = transform_geom("EPSG:4326", crs, geom)
                    geoms_to_rasterize.append((proj_geom, 1))
                except Exception:
                    continue

            if geoms_to_rasterize:
                burned = rasterio.features.rasterize(
                    geoms_to_rasterize,
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    default_value=1,
                    dtype=np.uint8,
                )
                fault_pixels = (burned == 1)
                if fault_buffer_px > 0 and binary_dilation is not None:
                    struct = np.ones(
                        (2 * fault_buffer_px + 1, 2 * fault_buffer_px + 1), dtype=bool
                    )
                    fault_mask = binary_dilation(fault_pixels, structure=struct)
                else:
                    fault_mask = fault_pixels
        except Exception:
            pass

    stable_mask = valid & (~high_grad_mask) & (~outlier_disp_mask) & (~fault_mask)

    # Solvability fallback: ensure at least 3 non-collinear pixels for 2D plane fitting
    if np.sum(stable_mask) < 3:
        stable_mask = valid & (~high_grad_mask)
        if np.sum(stable_mask) < 3:
            stable_mask = valid.copy()

    return stable_mask


def fit_spatial_ramp(
    displacement: np.ndarray,
    stable_mask: np.ndarray,
) -> tuple[float, float, float]:
    """Fit a 1st-order 2D plane (z = a*x + b*y + c) over stable pixels.

    Sets up design matrix X using pixel coordinates (x=column, y=row)
    and computes least-squares solution:
    coeffs, _, _, _ = np.linalg.lstsq(X, y_displacement, rcond=None)

    Returns (a, b, c) where:
      a: x-gradient (columns) in cm/pixel
      b: y-gradient (rows) in cm/pixel
      c: plane offset in cm
    """
    height, width = displacement.shape
    cols, rows = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    x_stable = cols[stable_mask]
    y_stable = rows[stable_mask]
    y_displacement = displacement[stable_mask].astype(np.float64)

    if x_stable.size < 3:
        offset = float(np.median(y_displacement)) if y_displacement.size > 0 else 0.0
        return 0.0, 0.0, offset

    X = np.column_stack((x_stable, y_stable, np.ones_like(x_stable)))
    coeffs, _, _, _ = np.linalg.lstsq(X, y_displacement, rcond=None)
    a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    return a, b, c


def detrend_displacement_array(
    displacement: np.ndarray,
    stable_mask: np.ndarray,
    nodata: float = -9999.0,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Evaluate fitted plane across full grid and subtract from displacement raster."""
    height, width = displacement.shape
    a, b, c = fit_spatial_ramp(displacement, stable_mask)
    cols, rows = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    plane = a * cols + b * rows + c
    valid = np.isfinite(displacement)
    if nodata is not None and math.isfinite(nodata):
        valid &= (displacement != nodata)
    detrended = np.where(valid, displacement - plane, nodata).astype(np.float32)
    return detrended, plane.astype(np.float32), (a, b, c)


def detrend_displacement_raster(
    raster_path: Path,
    output_dir: Path | None = None,
    output_name: str | None = None,
    faults_path: Path | None = None,
    fault_buffer_px: int = 5,
    gradient_mad_threshold: float = 2.5,
    outlier_mad_threshold: float = 3.0,
) -> tuple[Path, dict[str, float | str]]:
    """Remove long-wavelength orbital tilts and regional atmospheric gradients from a displacement GeoTIFF.

    1. Computes a stable background mask excluding active faults, water bodies, and high-gradient deformation.
    2. Fits a 1st-order 2D plane (z = a*x + b*y + c) over stable pixels using least-squares:
       coeffs, _, _, _ = np.linalg.lstsq(X, y_displacement, rcond=None)
    3. Subtracts the fitted plane across the entire grid.
    4. Saves detrended GeoTIFF and logs estimated gradient coefficients for quality control.
    """
    if not raster_path.is_file():
        raise ValueError(f"Displacement raster does not exist: {raster_path}")

    out_dir = output_dir or raster_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = output_name or f"{raster_path.stem}_detrended.tif"
    out_path = out_dir / out_name

    with rasterio.open(raster_path) as src:
        displacement = src.read(1)
        nodata = src.nodata if src.nodata is not None else -9999.0
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()
        tags = src.tags().copy()
        description = src.descriptions[0] if src.descriptions else "LOS displacement"

    stable_mask = compute_stable_background_mask(
        displacement=displacement,
        nodata=nodata,
        transform=transform,
        crs=crs,
        faults_path=faults_path,
        fault_buffer_px=fault_buffer_px,
        gradient_mad_threshold=gradient_mad_threshold,
        outlier_mad_threshold=outlier_mad_threshold,
    )

    detrended, plane, (a, b, c) = detrend_displacement_array(
        displacement=displacement,
        stable_mask=stable_mask,
        nodata=nodata,
    )

    valid_mask = np.isfinite(displacement)
    if nodata is not None and math.isfinite(nodata):
        valid_mask &= (displacement != nodata)
    total_valid = int(np.sum(valid_mask))
    stable_count = int(np.sum(stable_mask))
    pct_stable = (100.0 * stable_count / total_valid) if total_valid > 0 else 0.0

    res_x = abs(transform.a) if transform is not None else 1.0
    res_y = abs(transform.e) if transform is not None else 1.0
    slope_x_cm_km = (a / res_x) * 1000.0 if res_x > 0 else 0.0
    slope_y_cm_km = (b / res_y) * 1000.0 if res_y > 0 else 0.0

    qc_info: dict[str, float | str] = {
        "source_file": raster_path.name,
        "detrended_file": out_path.name,
        "ramp_a_cm_per_px": a,
        "ramp_b_cm_per_px": b,
        "ramp_c_offset_cm": c,
        "ramp_slope_x_cm_per_km": slope_x_cm_km,
        "ramp_slope_y_cm_per_km": slope_y_cm_km,
        "stable_pixels_count": stable_count,
        "total_valid_pixels": total_valid,
        "stable_pixels_pct": pct_stable,
    }

    print(
        f"[Spatial Ramp Detrending] {raster_path.name}:\n"
        f"  Plane model: z = ({a:+.6e})*x + ({b:+.6e})*y + ({c:+.4f}) cm\n"
        f"  Gradients: x = {a:+.4e} cm/px ({slope_x_cm_km:+.4f} cm/km), "
        f"y = {b:+.4e} cm/px ({slope_y_cm_km:+.4f} cm/km), offset = {c:+.2f} cm\n"
        f"  Stable background: {stable_count:,}/{total_valid:,} pixels ({pct_stable:.1f}%)"
    )

    tags.update({
        "spatial_ramp_detrended": "true",
        "ramp_plane_model": f"z = {a:.6e}*x + {b:.6e}*y + {c:.4f} cm",
        "ramp_slope_x_cm_per_px": f"{a:.6e}",
        "ramp_slope_y_cm_per_px": f"{b:.6e}",
        "ramp_offset_c_cm": f"{c:.4f}",
        "ramp_slope_x_cm_per_km": f"{slope_x_cm_km:.4f}",
        "ramp_slope_y_cm_per_km": f"{slope_y_cm_km:.4f}",
        "stable_pixels_count": str(stable_count),
        "stable_pixels_percentage": f"{pct_stable:.2f}%",
    })

    profile.update(dtype="float32", nodata=nodata)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(detrended, 1)
        dst.update_tags(**tags)
        dst.set_band_description(1, f"{description} (detrended)")

    return out_path, qc_info


# Alias for convenience
detrend_spatial_ramp = detrend_displacement_raster


def solve_deformation_timeseries(
    raster_paths: list[Path],
    output_dir: Path,
    minimum_span_days: int = TIMESERIES_MIN_SPAN_DAYS,
    block_size: int = 128,
    closure_threshold_rad: float = 1.8 * math.pi,
    flag_blunder_ratio: float = 0.25,
    skip_triplet_validation: bool = False,
    detrend_spatial_ramp: bool = False,
    faults_path: Path | None = None,
    num_workers: int = 0,
    multilook: int = 1,
) -> list[Path]:
    if minimum_span_days < TIMESERIES_MIN_SPAN_DAYS:
        raise ValueError(
            f"minimum time span cannot be less than {TIMESERIES_MIN_SPAN_DAYS} days"
        )
    if block_size < 1:
        raise ValueError("time-series block size must be positive")
    if multilook < 1:
        raise ValueError("multilook factor must be a positive integer >= 1")

    if num_workers <= 0:
        effective_workers = min(os.cpu_count() or 4, 8)
    else:
        effective_workers = num_workers

    groups: dict[str, list[tuple[Path, datetime, datetime]]] = {}
    for path in raster_paths:
        group_key, reference_time, secondary_time = _parse_pair_raster(path)
        groups.setdefault(group_key, []).append(
            (path, reference_time, secondary_time)
        )
    if not groups:
        raise ValueError("No numeric GUNW pair rasters were provided")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for group_key, items in sorted(groups.items()):
        dates = sorted({time for _, first, second in items for time in (first, second)})
        if len(dates) < 6:
            raise ValueError(
                f"{group_key} has {len(dates)} acquisition dates; at least six "
                "are required for the linear plus annual/semi-annual model"
            )
        span_days = (dates[-1] - dates[0]).total_seconds() / 86400
        if span_days < minimum_span_days:
            raise ValueError(
                f"{group_key} spans {span_days:.1f} days; at least "
                f"{minimum_span_days} days are required for seasonal fitting"
            )
        date_index = {date: index for index, date in enumerate(dates)}
        ordered_items = sorted(
            items, key=lambda item: (item[1], item[2], item[0].name)
        )
        edges = [
            (date_index[reference], date_index[secondary])
            for _, reference, secondary in ordered_items
        ]
        if any(reference >= secondary for reference, secondary in edges):
            raise ValueError(f"{group_key} contains reversed acquisition intervals")
        baseline = _baseline_matrix(edges, len(dates))
        if np.linalg.matrix_rank(baseline) != len(dates) - 1:
            raise ValueError(f"{group_key} interferogram network is disconnected")

        if detrend_spatial_ramp:
            detrended_dir = output_dir / "detrended"
            detrended_dir.mkdir(parents=True, exist_ok=True)
            detrended_items = []
            detrend_qc_list = []
            for path, ref_time, sec_time in ordered_items:
                det_path, qc = detrend_displacement_raster(
                    raster_path=path,
                    output_dir=detrended_dir,
                    faults_path=faults_path,
                )
                detrended_items.append((det_path, ref_time, sec_time))
                detrend_qc_list.append(qc)
            ordered_items = detrended_items
            qc_summary_file = output_dir / f"{group_key}_spatial_ramp_detrend_summary.json"
            qc_summary_file.write_text(json.dumps(detrend_qc_list, indent=2, default=str), encoding="utf-8")

        with ExitStack() as stack:
            datasets = [
                stack.enter_context(rasterio.open(path))
                for path, _, _ in ordered_items
            ]
            first = datasets[0]
            if first.crs is None or first.count != 1:
                raise ValueError(
                    f"{ordered_items[0][0].name} must be a single-band georeferenced raster"
                )
            if any(
                dataset.crs is None or dataset.count != 1
                for dataset in datasets
            ):
                raise ValueError(
                    f"{group_key} contains a raster without one georeferenced band"
                )
            width, height = first.width, first.height

            # Temporal triplet phase-closure validation step before matrix inversion
            wavelength_m = SPEED_OF_LIGHT_M_S / 1.2575e9
            first_freq = first.tags().get("center_frequency_hz")
            if first_freq:
                try:
                    wavelength_m = SPEED_OF_LIGHT_M_S / float(first_freq)
                except (ValueError, TypeError):
                    pass

            if not skip_triplet_validation:
                (
                    ordered_items,
                    datasets,
                    edges,
                    baseline,
                    blunder_masks,
                    triplet_summary,
                ) = validate_temporal_triplets(
                    ordered_items=ordered_items,
                    datasets=datasets,
                    edges=edges,
                    dates=dates,
                    first_transform=first.transform,
                    first_crs=first.crs,
                    width=width,
                    height=height,
                    wavelength_m=wavelength_m,
                    block_size=block_size,
                    closure_threshold_rad=closure_threshold_rad,
                    flag_blunder_ratio=flag_blunder_ratio,
                    output_dir=output_dir,
                    group_key=group_key,
                )
            else:
                blunder_masks = {}
                triplet_summary = {"triplets_total": 0, "status": "skipped"}

            row_ref, col_ref, reference_values = _choose_reference_pixel(
                datasets,
                first.transform,
                first.crs,
                baseline,
                width,
                height,
                block_size,
                blunder_masks=blunder_masks,
            )

            elapsed_years = np.array(
                [
                    (date - dates[0]).total_seconds() / (365.2425 * 86400)
                    for date in dates
                ],
                dtype=np.float64,
            )
            model = np.column_stack(
                (
                    np.ones(len(dates)),
                    elapsed_years,
                    np.sin(2 * np.pi * elapsed_years),
                    np.cos(2 * np.pi * elapsed_years),
                    np.sin(4 * np.pi * elapsed_years),
                    np.cos(4 * np.pi * elapsed_years),
                )
            )
            if np.linalg.matrix_rank(model) < model.shape[1]:
                raise ValueError(
                    f"{group_key} acquisition dates cannot support a full-rank "
                    "annual/semi-annual model"
                )

            output_names = (
                f"{group_key}_secular_tectonic_velocity_cm_yr.tif",
                f"{group_key}_seasonal_amplitude_cm.tif",
                f"{group_key}_residual_noise_map.tif",
            )
            out_width = width // multilook if multilook > 1 else width
            out_height = height // multilook if multilook > 1 else height
            out_transform = (
                update_transform_multilook(first.transform, multilook)
                if multilook > 1
                else first.transform
            )
            profile = {
                "driver": "GTiff",
                "width": out_width,
                "height": out_height,
                "count": 1,
                "dtype": "float32",
                "crs": first.crs,
                "transform": out_transform,
                "nodata": -9999.0,
                "compress": "deflate",
                "num_threads": "all_cpus",
            }
            outputs = [
                stack.enter_context(
                    rasterio.open(output_dir / name, "w", **profile)
                )
                for name in output_names
            ]
            tags = {
                "network_group": group_key,
                "orbit_direction": (
                    "ASCENDING" if "_A_" in group_key else "DESCENDING"
                    if "_D_" in group_key else "UNKNOWN"
                ),
                "acquisition_dates": ",".join(
                    date.isoformat(timespec="seconds") for date in dates
                ),
                "baseline_edges": str(len(edges)),
                "reference_pixel_row": str(row_ref // multilook if multilook > 1 else row_ref),
                "reference_pixel_column": str(col_ref // multilook if multilook > 1 else col_ref),
                "reference_method": (
                    "Maximum valid edge coverage, then minimum mean absolute "
                    "pair displacement; not externally validated as stable"
                ),
                "spatial_ramp_detrended": "true" if detrend_spatial_ramp else "false",
                "seasonal_model": (
                    "linear trend + annual and semi-annual sine/cosine; "
                    "seasonal amplitude is root-sum-square of both harmonics"
                ),
                "residual_noise": "RMS of temporal model residuals in centimeters",
                "triplets_evaluated": str(triplet_summary.get("triplets_total", 0)),
                "triplets_failed": str(triplet_summary.get("triplets_failed", 0)),
                "pairs_dropped": str(len(triplet_summary.get("dropped_pairs", []))),
                "pairs_downweighted": str(len(triplet_summary.get("downweighted_pairs", []))),
                "triplet_closure_threshold_rad": f"{closure_threshold_rad:.3f}",
                "multicore_workers": str(effective_workers),
                "multilook_factor": str(multilook),
            }
            if multilook > 1:
                tags["reference_pixel_original_row"] = str(row_ref)
                tags["reference_pixel_original_column"] = str(col_ref)
                tags["multilook_resolution_x_m"] = f"{abs(out_transform.a):.3f}"
                tags["multilook_resolution_y_m"] = f"{abs(out_transform.e):.3f}"

            output_units = ("centimeters per year", "centimeters", "centimeters")
            for output, name, units in zip(outputs, output_names, output_units):
                output.set_band_description(1, name.removesuffix(".tif"))
                output.update_tags(**tags, units=units)

            pinv_model = np.linalg.pinv(model)
            pinv_cache: dict[tuple[int, ...], np.ndarray | None] = {
                tuple(range(baseline.shape[0])): np.linalg.pinv(baseline)
            }

            tile_step = block_size * multilook if multilook > 1 else block_size
            tile_jobs = []
            for row in range(0, height, tile_step):
                for column in range(0, width, tile_step):
                    src_w = min(tile_step, width - column)
                    src_h = min(tile_step, height - row)
                    if multilook > 1:
                        dst_w = src_w // multilook
                        dst_h = src_h // multilook
                        if dst_w == 0 or dst_h == 0:
                            continue
                        src_w = dst_w * multilook
                        src_h = dst_h * multilook
                        out_window = Window(column // multilook, row // multilook, dst_w, dst_h)
                    else:
                        out_window = Window(column, row, src_w, src_h)
                    src_window = Window(column, row, src_w, src_h)
                    tile_jobs.append((src_window, out_window, row, column, src_h, src_w))

            def _process_one_tile(
                obs: np.ndarray, val: np.ndarray, out_win: Window
            ) -> tuple[Window, tuple[np.ndarray, np.ndarray, np.ndarray]]:
                tile_results = _invert_network_tile(
                    obs,
                    val,
                    baseline,
                    reference_values,
                    model,
                    pinv_model=pinv_model,
                    pinv_cache=pinv_cache,
                )
                return out_win, tile_results

            def _get_tile_data(
                src_win: Window, out_win: Window, r: int, c: int, sh: int, sw: int
            ) -> tuple[np.ndarray, np.ndarray, Window]:
                observations, valid = _network_observations(
                    datasets, src_win, first.transform, first.crs
                )
                if blunder_masks:
                    for edge_idx in range(len(datasets)):
                        if edge_idx in blunder_masks:
                            mask_block = blunder_masks[edge_idx][
                                r : r + sh, c : c + sw
                            ]
                            valid[edge_idx] &= ~mask_block
                if multilook > 1:
                    # Mark invalid / blunder-masked pixels as nodata first so they are
                    # excluded from the block averages and the masks survive multilooking.
                    observations = np.where(valid, observations, -9999.0).astype(np.float32)
                    observations = apply_multilooking(
                        observations, factor=multilook, nodata=-9999.0
                    )
                    valid = np.isfinite(observations) & (observations != -9999.0)
                return observations, valid, out_win

            if effective_workers > 1 and len(tile_jobs) > 1:
                with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                    # Bounded in-flight window: reads overlap with inversion and memory
                    # stays limited instead of holding every tile at once.
                    pending: deque = deque()

                    def _drain_one() -> None:
                        out_win, results = pending.popleft().result()
                        for output, result in zip(outputs, results):
                            output.write(result, 1, window=out_win)

                    for job in tile_jobs:
                        obs, val, out_win = _get_tile_data(*job)
                        pending.append(executor.submit(_process_one_tile, obs, val, out_win))
                        if len(pending) >= effective_workers * 2:
                            _drain_one()
                    while pending:
                        _drain_one()
            else:
                for job in tile_jobs:
                    obs, val, out_win = _get_tile_data(*job)
                    _, results = _process_one_tile(obs, val, out_win)
                    for output, result in zip(outputs, results):
                        output.write(result, 1, window=out_win)
            output_paths.extend(output_dir / name for name in output_names)
    return output_paths


def generate_rasters(
    source_path: Path,
    output_dir: Path,
    polarization: str = "auto",
    frequency: str = "A",
    min_coherence: float = 0.3,
    range_cm: float | None = None,
    color_only: bool = False,
    subtract_troposphere: bool = False,
    dem_path: Path | None = None,
    multilook: int = 1,
) -> tuple[Path, Path]:
    if not 0 <= min_coherence <= 1:
        raise ValueError("minimum coherence must be between 0 and 1")
    if range_cm is not None and (not math.isfinite(range_cm) or range_cm <= 0):
        raise ValueError("color range must be a positive finite number of centimeters")
    if subtract_troposphere and dem_path is None:
        raise ValueError("a geocoded DEM path is required for tropospheric correction")
    if not subtract_troposphere and dem_path is not None:
        raise ValueError("DEM path can only be used with tropospheric correction")
    if multilook < 1:
        raise ValueError("multilook factor must be a positive integer >= 1")

    grid_path = GRID_PATH.format(frequency=frequency)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(source_path, "r") as product:
        if grid_path not in product:
            raise ValueError(f"Product is missing the GUNW grid: {grid_path}")
        frequency_group = product[grid_path]
        pol = _resolve_polarization(frequency_group, polarization)
        layer = frequency_group[f"unwrappedInterferogram/{pol}"]
        required_datasets = (
            "unwrappedPhase",
            "coherenceMagnitude",
            "xCoordinates",
            "yCoordinates",
        )
        missing = [name for name in required_datasets if name not in layer]
        if missing:
            raise ValueError(
                f"Product is missing {pol} grid metadata or data: {', '.join(missing)}"
            )

        # Locate projection dataset (can be in layer, unwrappedInterferogram, or frequency_group)
        projection_dataset = (
            layer.get("projection")
            or frequency_group.get("unwrappedInterferogram/projection")
            or frequency_group.get("projection")
        )
        if projection_dataset is None:
            raise ValueError(
                f"Product is missing {pol} grid metadata or data: projection"
            )

        if "centerFrequency" not in frequency_group:
            raise ValueError(f"Product is missing {grid_path}/centerFrequency")

        phase_dataset = layer["unwrappedPhase"]
        coherence_dataset = layer["coherenceMagnitude"]
        x_coordinates = np.asarray(layer["xCoordinates"][:], dtype=np.float64)
        y_coordinates = np.asarray(layer["yCoordinates"][:], dtype=np.float64)
        if phase_dataset.ndim != 2 or phase_dataset.shape != coherence_dataset.shape:
            raise ValueError("Phase and coherence datasets must be matching 2D rasters")
        if phase_dataset.shape != (y_coordinates.size, x_coordinates.size):
            raise ValueError("Raster dimensions do not match the coordinate arrays")

        x_spacing = _coordinate_spacing(x_coordinates, "xCoordinates")
        y_spacing = _coordinate_spacing(y_coordinates, "yCoordinates")
        flip_x = x_spacing < 0
        flip_y = y_spacing > 0
        if flip_x:
            x_coordinates = x_coordinates[::-1]
        if flip_y:
            y_coordinates = y_coordinates[::-1]

        epsg_value = projection_dataset.attrs.get("epsg_code")
        if epsg_value is None:
            epsg_value = projection_dataset[()]
        epsg = int(np.asarray(epsg_value).item())
        if epsg <= 0:
            raise ValueError(f"Product contains an invalid projection EPSG code: {epsg}")

        center_frequency_hz = _read_scalar(
            frequency_group["centerFrequency"], "center frequency"
        )
        if center_frequency_hz <= 0:
            raise ValueError("Product center frequency must be positive")
        wavelength_m = SPEED_OF_LIGHT_M_S / center_frequency_hz
        phase_fill = _fill_value(phase_dataset)
        coherence_fill = _fill_value(coherence_dataset)
        transform = from_origin(
            float(x_coordinates[0] - abs(x_spacing) / 2),
            float(y_coordinates[0] + abs(y_spacing) / 2),
            abs(x_spacing),
            abs(y_spacing),
        )
        phase_correction = None
        tropo_tags: dict[str, str] = {}
        if subtract_troposphere and dem_path is not None:
            phase_correction, tropo_tags = _compute_tropospheric_phase_correction(
                product=product,
                dem_path=dem_path,
                phase_dataset=phase_dataset,
                coherence_dataset=coherence_dataset,
                x_coordinates=x_coordinates,
                y_coordinates=y_coordinates,
                transform=transform,
                epsg=epsg,
                flip_x=flip_x,
                flip_y=flip_y,
                phase_fill=phase_fill,
                coherence_fill=coherence_fill,
            )
        color_range_cm = range_cm or _sample_color_range(
            phase_dataset,
            coherence_dataset,
            wavelength_m,
            min_coherence,
            phase_fill,
            coherence_fill,
            phase_correction,
        )

        height, width = phase_dataset.shape
        out_height = height // multilook if multilook > 1 else height
        out_width = width // multilook if multilook > 1 else width
        if multilook > 1:
            # Multilooking trims the trailing *source* rows/columns that do not fill a
            # whole block. On a flipped grid those sit at the north/west edge of the
            # output, so the origin has to move in by the trimmed amount.
            trimmed_columns = (width - out_width * multilook) if flip_x else 0
            trimmed_rows = (height - out_height * multilook) if flip_y else 0
            out_transform = update_transform_multilook(
                transform * rasterio.Affine.translation(trimmed_columns, trimmed_rows),
                multilook,
            )
        else:
            out_transform = transform
        numeric_path = output_dir / f"{source_path.stem}_los_displacement_cm.tif"
        color_path = output_dir / f"{source_path.stem}_los_displacement_color.tif"
        numeric_profile = {
            "driver": "GTiff",
            "height": out_height,
            "width": out_width,
            "count": 1,
            "dtype": "float32",
            "crs": f"EPSG:{epsg}",
            "transform": out_transform,
            "nodata": -9999.0,
            "compress": "deflate",
            "num_threads": "all_cpus",
        }
        color_profile = {
            **numeric_profile,
            "count": 4,
            "dtype": "uint8",
            "nodata": None,
        }
        tags = {
            "source": source_path.name,
            "polarization": pol,
            "frequency": f"frequency{frequency}",
            "units": "centimeters",
            "center_frequency_hz": str(center_frequency_hz),
            "minimum_coherence": str(min_coherence),
            "color_range_cm": str(color_range_cm),
            "multilook_factor": str(multilook),
            "phase_to_displacement": (
                (
                    "(unwrappedPhase - empiricalTroposphericPhaseScreen) * (speed_of_light / centerFrequency) / (4*pi)"
                    if tropo_tags.get("troposphere_method") == "empirical_phase_elevation"
                    else "(unwrappedPhase - wetTroposphericPhaseScreen - hydrostaticTroposphericPhaseScreen) * (speed_of_light / centerFrequency) / (4*pi)"
                )
                if subtract_troposphere
                else "unwrappedPhase * (speed_of_light / centerFrequency) / (4*pi)"
            ),
            "sign_convention": "Retained from the source GUNW unwrapped phase; verify before directional interpretation.",
        }
        if multilook > 1:
            tags["multilook_resolution_x_m"] = f"{abs(out_transform.a):.3f}"
            tags["multilook_resolution_y_m"] = f"{abs(out_transform.e):.3f}"
        if subtract_troposphere:
            tags.update(tropo_tags)

        with ExitStack() as stack:
            numeric_output = (
                None
                if color_only
                else stack.enter_context(rasterio.open(numeric_path, "w", **numeric_profile))
            )
            color_output = stack.enter_context(
                rasterio.open(color_path, "w", **color_profile)
            )
            if numeric_output is not None:
                numeric_output.set_band_description(1, "LOS displacement (cm)")
                numeric_output.update_tags(**tags)
            color_output.colorinterp = (
                ColorInterp.red,
                ColorInterp.green,
                ColorInterp.blue,
                ColorInterp.alpha,
            )
            color_output.set_band_description(1, "Red")
            color_output.set_band_description(2, "Green")
            color_output.set_band_description(3, "Blue")
            color_output.set_band_description(4, "Alpha")
            color_output.update_tags(**tags, color_scale="blue-negative, white-zero, red-positive")

            chunk_step = 512
            if multilook > 1:
                chunk_step = max(multilook, ((512 // multilook) or 1) * multilook)

            for row_start in range(0, height, chunk_step):
                row_stop = min(row_start + chunk_step, height)
                if multilook > 1:
                    row_stop = row_start + ((row_stop - row_start) // multilook) * multilook
                    if row_stop <= row_start:
                        continue
                phase = np.asarray(
                    phase_dataset[row_start:row_stop, :], dtype=np.float32
                )
                coherence = np.asarray(
                    coherence_dataset[row_start:row_stop, :], dtype=np.float32
                )
                valid = _valid_pixels(
                    phase,
                    coherence,
                    phase_fill,
                    coherence_fill,
                    min_coherence,
                )
                if phase_correction is not None:
                    correction = phase_correction[row_start:row_stop, :]
                    valid &= np.isfinite(correction)
                    phase = phase - np.nan_to_num(correction, nan=0.0)
                displacement = phase * (wavelength_m / (4 * math.pi)) * 100
                numeric = np.where(valid, displacement, -9999.0).astype(np.float32)

                if multilook > 1:
                    numeric = apply_multilooking(numeric, factor=multilook, nodata=-9999.0)
                    valid_ml = np.isfinite(numeric) & (numeric != -9999.0)
                    rgba = _colorize(
                        np.where(valid_ml, numeric, 0), valid_ml, color_range_cm
                    )
                    if flip_x:
                        numeric = numeric[:, ::-1]
                        rgba = rgba[:, :, ::-1]
                    if flip_y:
                        numeric = numeric[::-1, :]
                        rgba = rgba[:, ::-1, :]
                    ml_row_start = row_start // multilook
                    ml_row_stop = row_stop // multilook
                    ml_row_h = ml_row_stop - ml_row_start
                    ml_out_row = out_height - ml_row_stop if flip_y else ml_row_start
                    window = Window(0, ml_out_row, out_width, ml_row_h)
                else:
                    rgba = _colorize(
                        np.where(valid, displacement, 0), valid, color_range_cm
                    )
                    if flip_x:
                        numeric = numeric[:, ::-1]
                        rgba = rgba[:, :, ::-1]
                    if flip_y:
                        numeric = numeric[::-1, :]
                        rgba = rgba[:, ::-1, :]
                    output_row = height - row_stop if flip_y else row_start
                    window = Window(0, output_row, width, row_stop - row_start)

                if numeric_output is not None:
                    numeric_output.write(numeric, 1, window=window)
                color_output.write(rgba, window=window)

    return numeric_path, color_path


def decompose_2d_displacement(
    asc_raster_path: Path,
    desc_raster_path: Path,
    output_dir: Path,
    incidence_angle_deg: float = 37.0,
    heading_deg: float = 12.0,
    color_range_cm: float | None = None,
) -> tuple[Path, Path, Path, Path]:
    if not asc_raster_path.is_file():
        raise FileNotFoundError(f"Ascending raster does not exist: {asc_raster_path}")
    if not desc_raster_path.is_file():
        raise FileNotFoundError(f"Descending raster does not exist: {desc_raster_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(asc_raster_path) as asc, rasterio.open(desc_raster_path) as desc:
        if asc.crs is None or desc.crs is None:
            raise ValueError("Both rasters must have a coordinate reference system")
        if asc.count != 1 or desc.count != 1:
            raise ValueError("Both rasters must be single-band numeric displacement GeoTIFFs")

        bounds_asc = transform_bounds(asc.crs, "EPSG:4326", *asc.bounds, densify_pts=21)
        bounds_desc = transform_bounds(desc.crs, "EPSG:4326", *desc.bounds, densify_pts=21)

        west = max(bounds_asc[0], bounds_desc[0])
        south = max(bounds_asc[1], bounds_desc[1])
        east = min(bounds_asc[2], bounds_desc[2])
        north = min(bounds_asc[3], bounds_desc[3])
        if west >= east or south >= north:
            raise ValueError("Ascending and Descending rasters do not geographically overlap")

        if asc.crs.to_epsg() == 4326 and desc.crs.to_epsg() == 4326:
            res_x = min(abs(asc.transform.a), abs(desc.transform.a))
            res_y = min(abs(asc.transform.e), abs(desc.transform.e))
        else:
            res_x = 0.0008
            res_y = 0.0008
        width = max(1, math.ceil((east - west) / res_x))
        height = max(1, math.ceil((north - south) / res_y))
        transform = from_origin(west, north, res_x, res_y)
        dst_crs = "EPSG:4326"

        d_asc = np.full((height, width), np.nan, dtype=np.float32)
        d_desc = np.full((height, width), np.nan, dtype=np.float32)

        reproject(
            source=rasterio.band(asc, 1),
            destination=d_asc,
            src_transform=asc.transform,
            src_crs=asc.crs,
            src_nodata=asc.nodata,
            dst_transform=transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        reproject(
            source=rasterio.band(desc, 1),
            destination=d_desc,
            src_transform=desc.transform,
            src_crs=desc.crs,
            src_nodata=desc.nodata,
            dst_transform=transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    valid = np.isfinite(d_asc) & np.isfinite(d_desc)
    if not np.any(valid):
        raise ValueError("No overlapping valid pixels found between ascending and descending rasters")

    theta = math.radians(incidence_angle_deg)
    alpha = math.radians(heading_deg)
    s_u = math.cos(theta)
    s_e_desc = math.sin(theta) * math.cos(alpha)

    d_up = np.full((height, width), np.nan, dtype=np.float32)
    d_east = np.full((height, width), np.nan, dtype=np.float32)

    d_up[valid] = (d_asc[valid] + d_desc[valid]) / (2.0 * s_u)
    d_east[valid] = (d_desc[valid] - d_asc[valid]) / (2.0 * s_e_desc)

    prefix = "2D_DECOMPOSITION"
    up_numeric_path = output_dir / f"{prefix}_vertical_elevation_change_cm.tif"
    up_color_path = output_dir / f"{prefix}_vertical_elevation_change_color.tif"
    east_numeric_path = output_dir / f"{prefix}_east_west_displacement_cm.tif"
    east_color_path = output_dir / f"{prefix}_east_west_displacement_color.tif"

    color_range_up = color_range_cm or float(np.percentile(np.abs(d_up[valid]), 98))
    color_range_east = color_range_cm or float(np.percentile(np.abs(d_east[valid]), 98))
    if not math.isfinite(color_range_up) or color_range_up <= 0:
        color_range_up = 1.0
    if not math.isfinite(color_range_east) or color_range_east <= 0:
        color_range_east = 1.0

    up_rgba = _colorize(d_up, valid, color_range_up)
    east_rgba = _colorize(d_east, valid, color_range_east)

    for path, data in ((up_numeric_path, d_up), (east_numeric_path, d_east)):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=1,
            dtype="float32",
            crs=dst_crs,
            transform=transform,
            nodata=np.nan,
        ) as dest:
            dest.write(data, 1)

    for path, rgba in ((up_color_path, up_rgba), (east_color_path, east_rgba)):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=4,
            dtype="uint8",
            crs=dst_crs,
            transform=transform,
        ) as dest:
            for band in range(1, 5):
                dest.write(rgba[band - 1], band)
                dest.set_band_description(band, ("red", "green", "blue", "alpha")[band - 1])
            dest.colorinterp = (
                ColorInterp.red,
                ColorInterp.blue,
                ColorInterp.green,
                ColorInterp.alpha,
            )

    return up_numeric_path, up_color_path, east_numeric_path, east_color_path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate georeferenced LOS deformation rasters from NISAR GUNW HDF5 products."
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    input_group.add_argument(
        "--input-file",
        type=Path,
        help="process one GUNW product instead of all products in the input directory",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--polarization",
        choices=("auto", "HH", "VV"),
        default="auto",
        help="use HH when available, otherwise VV (default: auto)",
    )
    parser.add_argument("--frequency", choices=("A", "B"), default="A")
    parser.add_argument(
        "--min-coherence",
        type=float,
        default=0.3,
        help="mask pixels below this coherence magnitude (0 to 1; default: 0.3)",
    )
    parser.add_argument(
        "--range-cm",
        type=float,
        help="symmetric color stretch in centimeters (default: sampled 98th percentile)",
    )
    parser.add_argument(
        "--color-only",
        action="store_true",
        help="write only the RGBA color raster, not the numeric displacement raster",
    )
    parser.add_argument(
        "--subtract-troposphere",
        action="store_true",
        help="subtract the product's embedded wet and hydrostatic phase screens",
    )
    parser.add_argument(
        "--dem-path",
        type=Path,
        help=(
            "single-band geocoded DEM in meters above the WGS84 ellipsoid "
            "(required with --subtract-troposphere)"
        ),
    )
    parser.add_argument(
        "--time-series",
        action="store_true",
        help=(
            "after generating pair rasters, invert all numeric pair rasters "
            "in the output directory and fit secular/seasonal models"
        ),
    )
    parser.add_argument(
        "--minimum-span-days",
        type=int,
        default=TIMESERIES_MIN_SPAN_DAYS,
        help=(
            "minimum acquisition span for seasonal fitting in days "
            f"(default: {TIMESERIES_MIN_SPAN_DAYS})"
        ),
    )
    parser.add_argument(
        "--time-series-only",
        action="store_true",
        help="solve an existing numeric pair-raster stack without reprocessing HDF5 products",
    )
    parser.add_argument(
        "--closure-threshold-rad",
        type=float,
        default=1.8 * math.pi,
        help="phase closure residual threshold in radians for triplet validation (default: 1.8*pi ~ 5.655)",
    )
    parser.add_argument(
        "--flag-blunder-ratio",
        type=float,
        default=0.25,
        help="fraction of blunder pixels in a triplet loop required to flag pair corruption (default: 0.25)",
    )
    parser.add_argument(
        "--skip-triplet-validation",
        action="store_true",
        help="skip temporal triplet phase-closure validation before matrix inversion",
    )
    parser.add_argument(
        "--detrend-spatial-ramp",
        action="store_true",
        help="remove long-wavelength spatial ramp / tilt from displacement rasters before time-series inversion",
    )
    parser.add_argument(
        "--faults-path",
        type=Path,
        default=None,
        help="GeoJSON path of active tectonic faults to exclude from stable background estimation",
    )
    parser.add_argument(
        "--multilook",
        type=int,
        default=1,
        choices=[1, 2, 4, 8, 16],
        help="hardware/spatial multilooking factor for block downsampling and speckle noise reduction (default: 1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="number of parallel CPU worker threads for block matrix inversion (default: 0 for all CPUs)",
    )
    parser.add_argument(
        "--decompose-2d",
        nargs=2,
        type=Path,
        metavar=("ASC_TIF", "DESC_TIF"),
        help="decompose matching Ascending and Descending numeric rasters into vertical elevation change and East-West plate motion",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.decompose_2d:
        try:
            up_num, up_col, east_num, east_col = decompose_2d_displacement(
                args.decompose_2d[0],
                args.decompose_2d[1],
                args.output_dir,
                color_range_cm=args.range_cm,
            )
            print(f"Generated 2D vertical elevation change: {up_num.name} and {up_col.name}")
            print(f"Generated 2D east-west plate displacement: {east_num.name} and {east_col.name}")
            return 0
        except Exception as exc:
            print(f"Error during 2D vector decomposition: {exc}", file=sys.stderr)
            return 1

    if args.subtract_troposphere and args.dem_path is None:
        parser.error("--dem-path is required with --subtract-troposphere")
    if args.dem_path is not None and not args.subtract_troposphere:
        parser.error("--dem-path can only be used with --subtract-troposphere")
    if args.minimum_span_days < TIMESERIES_MIN_SPAN_DAYS:
        parser.error(
            f"--minimum-span-days cannot be less than {TIMESERIES_MIN_SPAN_DAYS}"
        )
    if args.time_series_only and not args.time_series:
        parser.error("--time-series-only requires --time-series")
    if args.time_series_only:
        products: list[Path] = []
    elif args.input_file is not None:
        if not args.input_file.is_file():
            print(f"Input file does not exist: {args.input_file}", file=sys.stderr)
            return 1
        products = [args.input_file]
    else:
        if not args.input_dir.is_dir():
            print(f"Input directory does not exist: {args.input_dir}", file=sys.stderr)
            return 1
        products = sorted(args.input_dir.glob("*.h5"))
    if not products and not args.time_series_only:
        print(f"No .h5 GUNW products found in {args.input_dir}", file=sys.stderr)
        return 1

    try:
        for source_path in products:
            numeric_path, color_path = generate_rasters(
                source_path,
                args.output_dir,
                polarization=args.polarization,
                frequency=args.frequency,
                min_coherence=args.min_coherence,
                range_cm=args.range_cm,
                color_only=args.color_only,
                subtract_troposphere=args.subtract_troposphere,
                dem_path=args.dem_path,
                multilook=args.multilook,
            )
            if args.color_only:
                print(f"Generated {color_path.name}")
            else:
                print(f"Generated {numeric_path.name} and {color_path.name}")
        if args.time_series:
            numeric_rasters = sorted(
                args.output_dir.glob("*_los_displacement_cm.tif")
            )
            time_series_paths = solve_deformation_timeseries(
                numeric_rasters,
                args.output_dir,
                minimum_span_days=args.minimum_span_days,
                closure_threshold_rad=args.closure_threshold_rad,
                flag_blunder_ratio=args.flag_blunder_ratio,
                skip_triplet_validation=args.skip_triplet_validation,
                detrend_spatial_ramp=args.detrend_spatial_ramp,
                faults_path=args.faults_path,
                num_workers=args.workers,
                multilook=args.multilook,
            )
            for path in time_series_paths:
                print(f"Generated {path.name}")
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
