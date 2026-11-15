"""Download unique NISAR GUNW products for surface-deformation analysis."""

from __future__ import annotations

import argparse
import getpass
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import requests


ASF_SEARCH_URL = "https://api.daac.asf.alaska.edu/services/search/param"
NISAR_DOWNLOAD_HOST = "nisar.asf.earthdatacloud.nasa.gov"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "manipulate" / "data"
DEFAULT_RASTER_OUTPUT_DIR = DEFAULT_OUTPUT_DIR.parent / "output"
GENERATOR_PATH = DEFAULT_OUTPUT_DIR.parent / "generate_deformation.py"
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
MAX_RESULTS = 5000
MATURITY_FILE = ".nisar_maturity"
EARTHDATA_TOKEN_FILE = Path(__file__).resolve().parent / "earthdata_token.txt"
# A bare file name: no separators, no drive letters, no leading dot (so no "..").
_SAFE_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    try:
        west, south, east, north = (float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "bbox must be west,south,east,north in decimal degrees"
        ) from exc

    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise argparse.ArgumentTypeError(
            "bbox must be ordered west,south,east,north and within "
            "longitude/latitude ranges (antimeridian-crossing boxes are unsupported)"
        )
    return west, south, east, north


def bbox_to_wkt(bbox: tuple[float, float, float, float]) -> str:
    west, south, east, north = bbox
    return (
        f"POLYGON(({west} {south},{east} {south},{east} {north},"
        f"{west} {north},{west} {south}))"
    )


def validate_date_range(start: str, end: str) -> None:
    def parse(value: str) -> datetime:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    try:
        start_time = parse(start)
        end_time = parse(end)
    except ValueError as exc:
        raise ValueError("start and end must be ISO dates or timestamps") from exc

    if start_time > end_time:
        raise ValueError("start must be earlier than or equal to end")


def iter_products(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, list):
        for item in value:
            yield from iter_products(item)
    elif isinstance(value, dict):
        if "downloadUrl" in value or "fileName" in value:
            yield value
        else:
            for item in value.values():
                yield from iter_products(item)


def unique_products(products: Iterator[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for product in products:
        filename = product.get("fileName")
        url = product.get("downloadUrl")
        if not isinstance(filename, str) or not isinstance(url, str):
            raise ValueError("ASF returned a product without a filename or download URL")
        if Path(filename).name != filename:
            raise ValueError(f"ASF returned an unsafe product filename: {filename!r}")

        product_id = product.get("groupID") or product.get("granuleName") or url
        unique.setdefault(str(product_id), product)
    return sorted(unique.values(), key=lambda item: item["fileName"])


def latest_product(products: list[dict[str, Any]]) -> dict[str, Any]:
    def acquisition_time(product: dict[str, Any]) -> str:
        value = product.get("stopTime") or product.get("startTime")
        if not isinstance(value, str):
            raise ValueError("ASF returned a product without an acquisition timestamp")
        return value

    return max(products, key=acquisition_time)


def latest_acquisition_pair(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def pair_dates(product: dict[str, Any]) -> tuple[str, str]:
        start = product.get("startTime")
        stop = product.get("stopTime")
        if not isinstance(start, str) or not isinstance(stop, str):
            raise ValueError("ASF returned a product without acquisition timestamps")
        return start[:10], stop[:10]

    newest_pair = max(pair_dates(product) for product in products)
    return [product for product in products if pair_dates(product) == newest_pair]


def filter_orbit_direction(
    products: list[dict[str, Any]],
    direction: str | None,
) -> list[dict[str, Any]]:
    if direction is None:
        return products
    return [
        product
        for product in products
        if str(product.get("flightDirection", "")).upper() == direction
    ]


def sort_products_by_proximity(
    products: list[dict[str, Any]],
    target_bbox: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    """Sort products so granules closest to the target bounding box center appear first."""
    west, south, east, north = target_bbox
    target_lat = (south + north) / 2.0
    target_lon = (west + east) / 2.0

    def _dist_sq(p: dict[str, Any]) -> float:
        fp = p.get("stringFootprint", "")
        if "((" in fp and "))" in fp:
            try:
                coords_str = fp.split("((")[1].split("))")[0]
                pts = [pt.strip().split() for pt in coords_str.split(",") if pt.strip()]
                lons = [float(pt[0]) for pt in pts]
                lats = [float(pt[1]) for pt in pts]
                c_lon = sum(lons) / len(lons)
                c_lat = sum(lats) / len(lats)
                return (c_lon - target_lon) ** 2 + (c_lat - target_lat) ** 2
            except Exception:
                pass
        return 9999.0

    return sorted(products, key=_dist_sq)


def search_products(
    bbox: tuple[float, float, float, float],
    start: str,
    end: str,
    maturity: str,
) -> list[dict[str, Any]]:
    params = {
        "dataset": "NISAR",
        "processingLevel": "GUNW",
        "productionConfiguration": "PR",
        "dataMaturity": maturity,
        "intersectsWith": bbox_to_wkt(bbox),
        "start": start,
        "end": end,
        "maxResults": MAX_RESULTS,
        "output": "json",
    }
    response = requests.get(ASF_SEARCH_URL, params=params, timeout=60)
    response.raise_for_status()
    products = unique_products(iter_products(response.json()))
    if len(products) >= MAX_RESULTS:
        raise ValueError(
            f"ASF returned the {MAX_RESULTS}-product limit; narrow the area or date range"
        )
    return products


def is_downloaded(output_dir: Path, filename: str) -> bool:
    path = output_dir / filename
    return path.is_file() and path.stat().st_size > 0


def _earthdata_token() -> str:
    token = os.environ.get("EARTHDATA_TOKEN")
    if token:
        return token

    if EARTHDATA_TOKEN_FILE.is_file():
        token = EARTHDATA_TOKEN_FILE.read_text(encoding="utf-8").strip()
        if token:
            return token

    token = getpass.getpass("Earthdata bearer token (input hidden): ").strip()
    if not token:
        raise ValueError("An Earthdata bearer token is required to download products")
    return token


def ensure_output_maturity(output_dir: Path, maturity: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / MATURITY_FILE
    if marker.is_file():
        existing_maturity = marker.read_text(encoding="ascii").strip()
        if existing_maturity != maturity:
            raise ValueError(
                f"{output_dir} already contains {existing_maturity or 'unknown'} "
                "maturity data; choose a separate output directory to avoid mixing releases"
            )
        return

    if any(output_dir.glob("*.h5")):
        raise ValueError(
            f"{output_dir} contains untracked HDF5 products without a maturity marker; "
            "choose an empty output directory to avoid mixing releases"
        )
    marker.write_text(maturity + "\n", encoding="ascii")


def safe_product_filename(product: dict[str, Any]) -> str:
    """Return the product's file name, refusing anything that is not a bare name.

    Product dicts can come straight from an HTTP client, and the name is joined
    onto local directories, so path separators, drive letters and ``..`` must
    never get through.
    """
    filename = product.get("fileName")
    if not isinstance(filename, str) or not _SAFE_FILENAME.fullmatch(filename):
        raise ValueError(f"Refusing unsafe product file name: {filename!r}")
    return filename


def download_product(
    product: dict[str, Any],
    output_dir: Path,
    session: requests.Session,
    progress_callback: Any = None,
) -> bool:
    import time

    filename = safe_product_filename(product)
    url = product["downloadUrl"]
    parsed_url = urlparse(url)
    if parsed_url.scheme != "https" or parsed_url.hostname != NISAR_DOWNLOAD_HOST:
        raise ValueError(f"Refusing unexpected NISAR download URL: {url}")

    output_path = output_dir / filename
    if is_downloaded(output_dir, filename):
        return False

    partial_path = output_path.with_name(output_path.name + ".part")
    completed = False
    start_time = time.time()

    total_bytes = 0

    # Plain single-stream download
    try:
        with session.get(url, stream=True, timeout=(30, 180)) as response:
            if response.status_code in (401, 403):
                raise RuntimeError(
                    "ASF requires Earthdata Login for downloads. Set the "
                    "EARTHDATA_TOKEN environment variable to a valid bearer token."
                )
            response.raise_for_status()

            signature = bytearray()
            if total_bytes <= 0:
                total_bytes = int(response.headers.get("Content-Length", "0") or 0)
            downloaded_bytes = 0
            last_percent = -1
            with partial_path.open("wb") as output_file:
                for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                    if not chunk:
                        continue
                    if len(signature) < len(HDF5_SIGNATURE):
                        signature.extend(chunk[: len(HDF5_SIGNATURE) - len(signature)])
                    output_file.write(chunk)
                    downloaded_bytes += len(chunk)
                    now = time.time()
                    el = max(0.1, now - start_time)
                    spd = (downloaded_bytes / (1024 * 1024)) / el
                    if total_bytes > 0:
                        percent = min(100, downloaded_bytes * 100 // total_bytes)
                        rem = max(0, total_bytes - downloaded_bytes)
                        eta_s = int(rem / max(1, spd * 1024 * 1024))
                        eta_str = f"{eta_s}s" if eta_s < 60 else f"{eta_s//60}m {eta_s%60}s"
                        if percent != last_percent:
                            if progress_callback:
                                progress_callback(
                                    "downloading",
                                    percent,
                                    f"⬇️ {percent}% ({downloaded_bytes / (1024*1024):.0f}/{total_bytes / (1024*1024):.0f} MB) • {spd:.1f} MB/s • ETA: {eta_str}",
                                )
                            else:
                                print(f"{percent}%", end="\r", flush=True)
                            last_percent = percent
                    else:
                        if progress_callback and (downloaded_bytes % (8 * 1024 * 1024) < 4 * 1024 * 1024):
                            progress_callback("downloading", -1, f"⬇️ {downloaded_bytes / (1024*1024):.0f} MB received • {spd:.1f} MB/s")

            if bytes(signature) != HDF5_SIGNATURE:
                raise ValueError(f"Downloaded product is not a valid HDF5 file: {filename}")

        os.replace(partial_path, output_path)
        completed = True
        return True
    finally:
        if not completed and partial_path.exists():
            partial_path.unlink(missing_ok=True)


def process_and_discard_source(
    source_path: Path,
    keep_numeric: bool = True,
) -> Path:
    numeric_path = DEFAULT_RASTER_OUTPUT_DIR / f"{source_path.stem}_los_displacement_cm.tif"
    color_path = DEFAULT_RASTER_OUTPUT_DIR / f"{source_path.stem}_los_displacement_color.tif"
    if not color_path.is_file() or color_path.stat().st_mtime < source_path.stat().st_mtime:
        cmd = [
            sys.executable,
            str(GENERATOR_PATH),
            "--input-file",
            str(source_path),
            "--output-dir",
            str(DEFAULT_RASTER_OUTPUT_DIR),
        ]
        if not keep_numeric:
            cmd.append("--color-only")
        result = subprocess.run(
            cmd,
            cwd=DEFAULT_OUTPUT_DIR.parents[1],
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"Processing failed for {source_path.name}; source file was kept."
            )
    if not color_path.is_file() or color_path.stat().st_size == 0:
        raise RuntimeError(
            f"Expected color raster was not created for {source_path.name}; "
            "source file was kept."
        )

    if not keep_numeric and numeric_path.exists():
        numeric_path.unlink()
    source_path.unlink()
    status_kept = f"kept {color_path.name}" + (f" and {numeric_path.name}" if keep_numeric else "")
    print(f"Processed and removed source {source_path.name}; {status_kept}")
    return color_path


def remote_process_product(
    product: dict[str, Any],
    session: requests.Session,
    progress_callback=None,
    crop_bbox: tuple[float, float, float, float] | None = None,
    color_only: bool = False,
) -> Path:
    """Process a NISAR GUNW product **without** downloading the full HDF5 file.

    Uses HTTP range requests to stream only the HDF5 chunks needed by
    ``generate_deformation.py`` (unwrapped phase, coherence, incidence angle,
    coordinate arrays, and a handful of scalar attributes).  Typical data
    fetched is 5–15 % of the full file size.

    Parameters
    ----------
    product:
        ASF product metadata dict (must contain ``downloadUrl`` and ``fileName``).
    session:
        A :class:`requests.Session` with the ``Authorization: Bearer <token>``
        header already set.
    progress_callback:
        Optional callable ``(step: str, pct: int, detail: str)`` for live
        progress reporting.
    crop_bbox:
        Optional ``(west, south, east, north)`` in degrees. When given, only
        the pixels inside this window are streamed (large transfer saving) and
        the output raster covers just that window. Default: whole frame.

    Returns
    -------
    Path
        The generated colour GeoTIFF in ``manipulate/output/``.
    """
    import sys as _sys
    from pathlib import Path as _Path
    # Lazy import to avoid a hard dependency at module level
    try:
        from remote_hdf5 import RemoteHDF5File
    except ImportError:
        # Try the sat-data sibling directory
        _sys.path.insert(0, str(_Path(__file__).resolve().parent))
        from remote_hdf5 import RemoteHDF5File

    url = product["downloadUrl"]
    filename = safe_product_filename(product)
    parsed_url = urlparse(url)
    if parsed_url.scheme != "https" or parsed_url.hostname != NISAR_DOWNLOAD_HOST:
        raise ValueError(f"Refusing unexpected NISAR download URL: {url}")

    color_path = DEFAULT_RASTER_OUTPUT_DIR / f"{_Path(filename).stem}_los_displacement_color.tif"
    numeric_path = DEFAULT_RASTER_OUTPUT_DIR / f"{_Path(filename).stem}_los_displacement_cm.tif"
    # A sidecar marks outputs produced from a cropped window, so a cached full-frame
    # result is never reused for a crop (or vice versa).
    crop_marker = color_path.with_suffix(".crop")
    if (
        crop_bbox is None
        and not crop_marker.exists()
        and color_path.is_file()
        and color_path.stat().st_size > 0
    ):
        if progress_callback:
            progress_callback("granule_done", 100, f"Already processed: {color_path.name}")
        return color_path

    DEFAULT_RASTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bytes_fetched_total = [0]
    file_size = [None]

    def _on_fetch(fetched: int, total_fetched: int) -> None:
        bytes_fetched_total[0] = total_fetched
        if progress_callback and file_size[0]:
            # Rough estimate: data fetch accounts for ~80% of progress
            # (the remaining 20% is processing)
            raw_pct = min(80, int(total_fetched * 80 / file_size[0]))
            mb_fetched = total_fetched / 1024 / 1024
            mb_total = file_size[0] / 1024 / 1024
            progress_callback(
                "streaming",
                raw_pct,
                f"Streaming {mb_fetched:.0f} / {mb_total:.0f} MB from ASF cloud",
            )

    if progress_callback:
        progress_callback("connect", 2, f"Connecting to ASF cloud: {filename}")

    remote_file = RemoteHDF5File(url, session, chunk_cache_mb=1, progress_callback=_on_fetch)
    file_size[0] = remote_file.file_size

    if progress_callback:
        size_gb = (file_size[0] or 0) / 1024**3
        progress_callback("stream_start", 5, f"File is {size_gb:.2f} GB — streaming needed bands only")

    # Write the remote file wrapper to a temporary in-memory staging path.
    # generate_deformation.py expects a local path, so we write just the needed
    # slices to a small temporary HDF5 that only contains the datasets we need.
    import h5py
    import numpy as np
    import tempfile
    import os

    # generate_deformation.py names its outputs after the input file, so the
    # stub must carry the real granule filename.
    import shutil
    tmp_dir = _Path(tempfile.mkdtemp(prefix="nisar_remote_"))
    tmp_path = tmp_dir / filename

    try:
        if progress_callback:
            progress_callback("extract", 10, "Opening remote HDF5 and extracting needed datasets…")

        # Essential datasets required by generate_deformation.py
        DATASETS_TO_COPY = [
            # Unwrapped phase and coherence
            "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/{pol}/unwrappedPhase",
            "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/{pol}/coherenceMagnitude",
            "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/{pol}/projection",
            "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/projection",
            "science/LSAR/GUNW/grids/frequencyA/projection",
            # Coordinates
            "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/{pol}/xCoordinates",
            "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/{pol}/yCoordinates",
            "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/{pol}/xCoordinateSpacing",
            "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/{pol}/yCoordinateSpacing",
            # Center frequency for carrier wavelength calculation
            "science/LSAR/GUNW/grids/frequencyA/centerFrequency",
            "science/LSAR/GUNW/metadata/radarGrid/projection",
        ]

        with h5py.File(remote_file, "r") as src:
            # Resolve polarization
            try:
                pol_grp = src["science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"]
                pol = "HH" if "HH" in pol_grp else ("VV" if "VV" in pol_grp else None)
            except KeyError:
                pol = None
            if not pol:
                raise RuntimeError("Cannot find HH or VV unwrapped phase in remote HDF5")

            if progress_callback:
                progress_callback("extract", 15, f"Polarization: {pol}  — fetching datasets…")

            window = None
            if crop_bbox is not None:
                window = _bbox_to_pixel_window(src, pol, crop_bbox)

            # Create a minimal local HDF5 with just what we need
            with h5py.File(tmp_path, "w") as dst:
                def _copy_dataset(src_path: str) -> None:
                    path = src_path.replace("{pol}", pol)
                    if path not in src:
                        return
                    ds = src[path]
                    group_path = "/".join(path.split("/")[:-1])
                    ds_name = path.split("/")[-1]
                    # Create parent groups
                    dst.require_group(group_path)
                    if window is not None and ds_name in ("unwrappedPhase", "coherenceMagnitude"):
                        r0, r1, c0, c1 = window
                        out = dst[group_path].create_dataset(
                            ds_name,
                            data=ds[r0:r1, c0:c1],
                            chunks=True,
                            compression="gzip",
                        )
                        _copy_portable_attrs(ds, out)
                    elif window is not None and ds_name in ("xCoordinates", "yCoordinates"):
                        r0, r1, c0, c1 = window
                        lo, hi = (c0, c1) if ds_name == "xCoordinates" else (r0, r1)
                        out = dst[group_path].create_dataset(ds_name, data=ds[lo:hi])
                        _copy_portable_attrs(ds, out)
                    else:
                        # Copy with all attributes
                        src.copy(path, dst[group_path], name=ds_name)

                total_ds = len(DATASETS_TO_COPY)
                for idx, ds_path in enumerate(DATASETS_TO_COPY):
                    _copy_dataset(ds_path)
                    if progress_callback:
                        pct = 15 + int((idx + 1) / total_ds * 55)
                        progress_callback("extract", pct, f"Fetched dataset {idx + 1}/{total_ds}")

                # Ensure projection is accessible under the polarization layer
                pol_layer_path = f"science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/{pol}"
                if pol_layer_path in dst and "projection" not in dst[pol_layer_path]:
                    for proj_candidate in [
                        "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/projection",
                        "science/LSAR/GUNW/grids/frequencyA/projection",
                        "science/LSAR/GUNW/metadata/radarGrid/projection",
                    ]:
                        if proj_candidate in dst:
                            dst[f"{pol_layer_path}/projection"] = dst[proj_candidate]
                            break

                # Copy top-level and frequency group attributes
                for group_path in ("science/LSAR/GUNW", "science/LSAR/GUNW/grids/frequencyA"):
                    if group_path in src and group_path in dst:
                        _copy_portable_attrs(src[group_path], dst[group_path])

        if progress_callback:
            mb = tmp_path.stat().st_size / 1024 / 1024
            remote_mb = bytes_fetched_total[0] / 1024 / 1024
            progress_callback(
                "process",
                72,
                f"Extracted {mb:.1f} MB locally (fetched {remote_mb:.0f} MB from remote). Processing…",
            )

        # Now run generate_deformation.py on the small local stub
        import subprocess
        result = subprocess.run(
            [
                _sys.executable,
                str(GENERATOR_PATH),
                "--input-file", str(tmp_path),
                "--output-dir", str(DEFAULT_RASTER_OUTPUT_DIR),
                *(["--color-only"] if color_only else []),
            ],
            cwd=str(GENERATOR_PATH.parent.parent),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"generate_deformation.py failed for {filename}:\n{result.stderr[:500]}"
            )

        if progress_callback:
            progress_callback("granule_done", 95, f"GeoTIFF generated: {color_path.name}")

        if not color_path.is_file() or color_path.stat().st_size == 0:
            raise RuntimeError(
                f"Expected output {color_path.name} was not created by generate_deformation.py.\n"
                f"stdout: {result.stdout[:300]}\nstderr: {result.stderr[:300]}"
            )

        if crop_bbox is not None:
            crop_marker.write_text(",".join(str(v) for v in crop_bbox), encoding="ascii")
        else:
            crop_marker.unlink(missing_ok=True)
        if color_only and numeric_path.exists():
            numeric_path.unlink()
        return color_path

    finally:
        remote_file.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _is_reference_dtype(dtype) -> bool:
    """True when an HDF5 dtype is, or contains, object/region references."""
    import h5py
    import numpy as np

    if dtype is None:
        return False
    if h5py.check_ref_dtype(dtype) is not None:
        return True
    base = h5py.check_vlen_dtype(dtype)
    if base is not None and base not in (str, bytes):
        return _is_reference_dtype(np.dtype(base))
    if dtype.fields:
        return any(_is_reference_dtype(field[0]) for field in dtype.fields.values())
    if dtype.subdtype is not None:
        return _is_reference_dtype(dtype.subdtype[0])
    return False


def _copy_portable_attrs(source, destination) -> None:
    """Copy HDF5 attributes, skipping references that only mean something in the source file.

    Dimension-scale bookkeeping (``DIMENSION_LIST`` / ``REFERENCE_LIST``) points at
    objects by address; written into another file it would dangle or fail outright.
    """
    for key in source.attrs:
        if key in ("DIMENSION_LIST", "REFERENCE_LIST"):
            continue
        try:
            if _is_reference_dtype(source.attrs.get_id(key).dtype):
                continue
            destination.attrs[key] = source.attrs[key]
        except (TypeError, ValueError, OSError):
            # An attribute h5py cannot round-trip is not worth failing the granule for.
            continue


def _bbox_to_pixel_window(src, pol: str, bbox: tuple[float, float, float, float]):
    """Return (row0, row1, col0, col1) of the grid covering a lon/lat bbox, or raise."""
    import numpy as np
    from rasterio.warp import transform_bounds

    base = "science/LSAR/GUNW/grids/frequencyA"
    layer = src[f"{base}/unwrappedInterferogram/{pol}"]
    proj = None
    for cand in (f"{base}/unwrappedInterferogram/{pol}/projection",
                 f"{base}/unwrappedInterferogram/projection",
                 f"{base}/projection"):
        if cand in src:
            proj = src[cand]
            break
    if proj is None:
        raise RuntimeError("Cannot crop: projection dataset not found")
    epsg_value = proj.attrs.get("epsg_code")
    if epsg_value is None:
        epsg_value = proj[()]
    epsg = int(np.asarray(epsg_value).item())

    xs = np.asarray(layer["xCoordinates"][:], dtype=np.float64)
    ys = np.asarray(layer["yCoordinates"][:], dtype=np.float64)
    x_min, y_min, x_max, y_max = transform_bounds("EPSG:4326", f"EPSG:{epsg}", *bbox, densify_pts=21)
    cols = np.where((xs >= x_min) & (xs <= x_max))[0]
    rows = np.where((ys >= y_min) & (ys <= y_max))[0]
    if cols.size < 2 or rows.size < 2:
        raise RuntimeError("Requested crop window does not overlap this granule")
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def _process_granule_worker(
    product: dict[str, Any],
    token: str,
    crop_bbox: tuple[float, float, float, float] | None,
    color_only: bool,
) -> str:
    """Process one granule in a worker process (module-level so it can be pickled)."""
    callback = None
    progress_queue = _worker_progress_queue
    if progress_queue is not None:
        name = product.get("fileName")
        last: list[Any] = [None]

        def callback(step: str, pct: int, detail: str) -> None:
            # Byte-level progress fires once per block; only forward real changes.
            if (step, pct) == last[0]:
                return
            last[0] = (step, pct)
            try:
                progress_queue.put((name, step, pct, detail))
            except Exception:  # noqa: BLE001 - progress must never break processing
                pass

    with requests.Session() as session:
        session.headers["Authorization"] = f"Bearer {token}"
        return str(
            remote_process_product(
                product,
                session,
                progress_callback=callback,
                crop_bbox=crop_bbox,
                color_only=color_only,
            )
        )


# Set in each worker process by the pool initializer; None means "no live progress".
_worker_progress_queue = None


def _init_progress_worker(progress_queue) -> None:
    global _worker_progress_queue
    _worker_progress_queue = progress_queue


def remote_process_products(
    products: list[dict[str, Any]],
    token: str,
    workers: int = 4,
    progress_callback=None,
    crop_bbox: tuple[float, float, float, float] | None = None,
    color_only: bool = False,
    on_granule_done=None,
) -> tuple[int, list[str]]:
    """Stream-process several granules in parallel worker *processes*.

    h5py holds a global lock during every HDF5 call (including the network reads
    done through ``RemoteHDF5File``), so threads would run granules one at a time;
    separate processes give real parallelism.

    Returns ``(succeeded_count, error_messages)``; failures never discard the
    granules that did succeed. ``progress_callback(step, overall_pct, detail)``
    receives overall 0-90% progress and never a terminal "done" step.
    """
    import multiprocessing
    import threading
    from concurrent.futures import ProcessPoolExecutor, as_completed

    products = [p for p in products if p.get("fileName")]
    total = len(products)
    if not total:
        return 0, []

    # Per-granule 0-100 progress, combined into one monotonic overall 0-90 figure.
    granule_pct: dict[str, int] = {}
    progress_lock = threading.Lock()
    overall_high = [0]

    def _report(name: str, step: str, pct: int, detail: str) -> None:
        with progress_lock:
            granule_pct[name] = max(granule_pct.get(name, 0), min(100, max(0, int(pct))))
            overall = int(sum(granule_pct.values()) / total * 0.9)
            overall_high[0] = overall = max(overall_high[0], overall)
        progress_callback(step, overall, detail)

    # Workers are separate processes, so their progress comes back over a queue
    # that a small thread drains into ``progress_callback``.
    context = multiprocessing.get_context()
    progress_queue = context.Queue() if progress_callback else None
    drain_thread = None
    if progress_queue is not None:
        numbering = {p["fileName"]: i for i, p in enumerate(products, start=1)}

        def _drain() -> None:
            while True:
                item = progress_queue.get()
                if item is None:
                    return
                name, step, pct, detail = item
                if step == "granule_done":
                    continue  # reported from the parent once the future resolves
                prefix = f"[granule {numbering.get(name, '?')}/{total}] " if total > 1 else ""
                try:
                    _report(name, step, pct, prefix + detail)
                except Exception:  # noqa: BLE001 - a UI callback must not stall the drain
                    pass

        drain_thread = threading.Thread(target=_drain, daemon=True)
        drain_thread.start()

    finished = 0
    errors: list[str] = []
    try:
        with ProcessPoolExecutor(
            max_workers=max(1, min(workers, total)),
            mp_context=context,
            initializer=_init_progress_worker,
            initargs=(progress_queue,),
        ) as pool:
            futures = {
                pool.submit(_process_granule_worker, p, token, crop_bbox, color_only): p
                for p in products
            }
            for future in as_completed(futures):
                product = futures[future]
                finished += 1
                try:
                    future.result()
                    detail = f"Finished {product['fileName']}"
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{product['fileName']}: {exc}")
                    detail = f"Failed {product['fileName']}"
                if progress_callback:
                    _report(product["fileName"], "granule_done", 100, f"[{finished}/{total}] {detail}")
                if on_granule_done:
                    on_granule_done(finished, total, product["fileName"])
    finally:
        if progress_queue is not None:
            progress_queue.put(None)
            drain_thread.join(timeout=10)
            progress_queue.close()
    return total - len(errors), errors


def _progress_summary(
    completed: int,
    total: int,
    downloaded: int,
    skipped: int,
) -> str:
    return (
        f"Files: {completed}/{total} complete, {total - completed} remaining "
        f"({downloaded} downloaded this run, {skipped} already present)."
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search ASF for NISAR geocoded unwrapped interferograms (GUNW) "
            "and download each matching product once."
        )
    )
    parser.add_argument(
        "--bbox",
        required=True,
        type=parse_bbox,
        metavar="W,S,E,N",
        help="area of interest as west,south,east,north decimal degrees",
    )
    parser.add_argument("--start", required=True, help="first acquisition date (ISO 8601)")
    parser.add_argument("--end", required=True, help="last acquisition date (ISO 8601)")
    parser.add_argument(
        "--maturity",
        choices=("VALIDATED", "PROVISIONAL", "BETA"),
        default="VALIDATED",
        help="use one data maturity per time series (default: VALIDATED)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"download directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list matching products without downloading them",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="download only the matching product with the latest acquisition time",
    )
    parser.add_argument(
        "--latest-pair",
        action="store_true",
        help="download every matching granule from the latest acquisition-date pair",
    )
    parser.add_argument(
        "--orbit-direction",
        choices=("ASCENDING", "DESCENDING"),
        help="limit products to one viewing direction for a coherent mosaic",
    )
    parser.add_argument(
        "--process-and-delete",
        action="store_true",
        help="process each product by streaming only the needed HDF5 data from ASF (no full download)",
    )
    parser.add_argument(
        "--discard-numeric",
        action="store_true",
        help="discard numeric cm rasters to minimize storage (disables subsequent time-series analysis)",
    )
    parser.add_argument(
        "--clip-faults",
        action="store_true",
        default=True,
        help="dynamically clip active faults to the target bounding box for the map dashboard (default: True)",
    )
    parser.add_argument(
        "--no-clip-faults",
        action="store_false",
        dest="clip_faults",
        help="skip clipping active faults",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        validate_date_range(args.start, args.end)
        products = search_products(args.bbox, args.start, args.end, args.maturity)
        if not products:
            print("No matching NISAR GUNW products were found.")
            return 0

        if args.latest_only and args.latest_pair:
            parser.error("--latest-only and --latest-pair cannot be used together")
        if args.latest_pair:
            products = latest_acquisition_pair(products)
        if args.latest_only:
            products = [latest_product(products)]
        if args.orbit_direction:
            products = filter_orbit_direction(products, args.orbit_direction)
            if not products:
                print(f"No matching {args.orbit_direction} products were found.")
                return 0

        print(f"Found {len(products)} unique {args.maturity} GUNW product(s).")
        for product in products:
            print(f"  {product['fileName']}")

        if getattr(args, "clip_faults", True) and args.bbox:
            try:
                from fault_clipper import clip_faults_to_bbox
                clip_faults_to_bbox(args.bbox)
            except Exception as fault_err:
                print(f"Notice: Active fault clipping skipped ({fault_err}).")

        if args.dry_run:
            return 0

        ensure_output_maturity(args.output, args.maturity)
        downloaded = 0
        skipped = 0
        processed = 0
        token = _earthdata_token()
        with requests.Session() as session:
            if token:
                session.headers["Authorization"] = f"Bearer {token}"

            print(_progress_summary(0, len(products), downloaded, skipped))
            for index, product in enumerate(products, start=1):
                print(f"[{index}/{len(products)}] {product['fileName']}")
                session.headers["Authorization"] = "Bearer " + token
                if args.process_and_delete:
                    remote_process_product(product, session, color_only=args.discard_numeric)
                    processed += 1
                elif is_downloaded(args.output, product["fileName"]):
                    skipped += 1
                    print(f"Already present; skipped {product['fileName']}")
                elif download_product(product, args.output, session):
                    downloaded += 1
                    print(f"Downloaded {product['fileName']}")

                print(_progress_summary(index, len(products), downloaded, skipped))

        print(
            f"Complete: {downloaded} downloaded, {skipped} already present, "
            f"{processed} processed."
        )
        return 0
    except (requests.RequestException, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
