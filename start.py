"""Prepare available deformation data and open the interactive map."""

from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "manipulate" / "data"
OUTPUT_DIR = ROOT / "manipulate" / "output"
TRANS_DIR = ROOT / "trans"
INDEX_PATH = ROOT / "map" / "map.html"
GENERATOR_PATH = ROOT / "manipulate" / "generate_deformation.py"
OVERLAY_PREPARER_PATH = TRANS_DIR / "prepare_overlay.py"
OVERLAY_CONFIG_PATH = TRANS_DIR / "deformation-overlay.js"


def _python_314_command() -> list[str] | None:
    if sys.version_info[:2] == (3, 14):
        return [sys.executable]
    for candidate in [
        Path(r"C:\ProgramData\anaconda3\python.exe"),
        Path(r"C:\Program Files\Python314\python.exe"),
    ]:
        if candidate.is_file():
            return [str(candidate)]
    launcher = shutil.which("py")
    return [launcher, "-3.14"] if launcher else None


def _run_script(command: list[str], script: Path, args: list[str] | None = None) -> bool:
    result = subprocess.run([*command, str(script), *(args or [])], cwd=ROOT, check=False)
    if result.returncode:
        print(f"Warning: {script.relative_to(ROOT)} exited with code {result.returncode}.")
        return False
    return True


def _find_free_port(start: int = 8000, end: int = 8100) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free localhost port found in range {start}-{end}")


def _start_http_server(directory: Path, port: int) -> subprocess.Popen[str]:
    server_script = ROOT / "server.py"
    if server_script.is_file():
        command = [
            sys.executable,
            str(server_script),
            "--port",
            str(port),
            "--no-open",
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "http.server",
            str(port),
            "--bind",
            "127.0.0.1",
            "--directory",
            str(directory),
        ]
    return subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def _map_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/"


def _color_rasters() -> list[Path]:
    if not OUTPUT_DIR.is_dir():
        return []
    return list(OUTPUT_DIR.glob("*_los_displacement_color.tif"))


def _needs_generation(products: list[Path]) -> bool:
    for product in products:
        raster_path = OUTPUT_DIR / f"{product.stem}_los_displacement_color.tif"
        numeric_path = OUTPUT_DIR / f"{product.stem}_los_displacement_cm.tif"
        if (
            not raster_path.is_file()
            or not numeric_path.is_file()
            or raster_path.stat().st_mtime < product.stat().st_mtime
            or numeric_path.stat().st_mtime < product.stat().st_mtime
        ):
            return True
    return False


def initialize() -> None:
    products = list(INPUT_DIR.glob("*.h5")) if INPUT_DIR.is_dir() else []
    if products and _needs_generation(products):
        python_314 = _python_314_command()
        if python_314 is None:
            print("Warning: Python 3.14 is required to generate rasters, but was not found.")
        else:
            print("Generating deformation rasters from local GUNW products...")
            _run_script(python_314, GENERATOR_PATH)
    elif not products:
        print("No local GUNW products found; skipping raster generation.")

    rasters = _color_rasters()
    if not rasters:
        print("No generated color raster found; the map will open with the overlay disabled.")
        return

    if sys.version_info[:2] == (3, 14):
        python_commands = [[sys.executable]]
    else:
        python_commands = [[sys.executable]]
        python_314 = _python_314_command()
        if python_314 and python_314 not in python_commands:
            python_commands.append(python_314)

    time_series_ready = False
    numeric_rasters = list(OUTPUT_DIR.glob("*_los_displacement_cm.tif"))
    if numeric_rasters:
        for command in python_commands:
            print("Preparing time-series layers when the local stack supports them...")
            if _run_script(
                command,
                GENERATOR_PATH,
                ["--time-series", "--time-series-only"],
            ):
                time_series_ready = True
                break
        if not time_series_ready:
            print(
                "Time-series layers are unavailable; the map will still show "
                "the latest-pair deformation overlay."
            )
    else:
        print(
            "Time-series layers are unavailable because no numeric pair rasters "
            "are present; retain them and download a connected stack spanning "
            "at least one year."
        )

    overlay_args = ["--mosaic", "--full-swath"]
    if time_series_ready:
        overlay_args.append("--time-series")
    for command in python_commands:
        print("Preparing the deformation overlay...")
        if _run_script(command, OVERLAY_PREPARER_PATH, overlay_args):
            return
    print("Warning: overlay preparation failed; the map will open without the overlay.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="prepare available raster and overlay assets without opening the browser",
    )
    args = parser.parse_args()

    if not INDEX_PATH.is_file():
        print(f"Map page not found: {INDEX_PATH}", file=sys.stderr)
        return 1

    initialize()

    port = _find_free_port()
    server = _start_http_server(ROOT, port)
    map_url = _map_url(port)
    print(f"Serving project map at {map_url}")

    if not args.no_open:
        opened = webbrowser.open_new_tab(map_url)
        if not opened:
            print(f"Open this map page in a browser: {map_url}")
    else:
        print("Map server started; open the page in a browser when ready.")

    try:
        server.wait()
    except KeyboardInterrupt:
        print("\nShutting down map server...")
        server.terminate()
        server.wait()
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
