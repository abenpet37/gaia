# NISAR Tectonic Deformation & Earthquake Monitor

This application downloads NASA-ISRO Synthetic Aperture Radar (**NISAR**) geocoded unwrapped interferogram (GUNW) products, converts radar phase data into line-of-sight (LOS) surface displacement rasters, decomposes 2D vertical elevation change vs. horizontal tectonic plate motion, and maps deformation alongside live **USGS earthquake feeds** and **active tectonic fault traces** on interactive Leaflet basemaps.

## Project layout

- `sat-data/` — ASF search and NISAR GUNW download tools (with Earthdata authentication and numeric raster retention).
- `manipulate/` — Phase-to-displacement processing, 2D vector decomposition (vertical uplift/subsidence vs. East-West tectonic plate motion), time-series SBAS secular velocity inversion, and raster storage (`data/` and `output/`).
- `trans/` — Reprojection and multi-swath mosaic preparation (both regional rift swathes and local city presets) for Leaflet rendering.
- `map/` — Interactive dashboard with live USGS earthquake feeds (depth color-coded, magnitude scaled), active tectonic fault traces, global plate boundaries, opacity controls, and multi-layer basemaps.

## Requirements and setup

Use Python 3.14 (such as Anaconda3 Python 3.14). Install dependencies from the repository root:

```powershell
pip install -r sat-data\requirements.txt -r manipulate\requirements.txt
```

Downloading products requires an Earthdata Login bearer token. The downloader checks the ignored local file `sat-data\earthdata_token.txt`, then prompts with hidden input if neither the file nor the `EARTHDATA_TOKEN` environment variable contains a token:

```powershell
$env:EARTHDATA_TOKEN = "<your Earthdata bearer token>"
```

## Workflow

### 1. Search for and download products

Run a dry search first (bounding boxes are `west,south,east,north` in decimal degrees):

```powershell
python sat-data\download_nisar.py --bbox 32.0,-6.5,41.0,10.5 --start 2026-10-01 --end 2026-11-14 --maturity PROVISIONAL --dry-run
```

Remove `--dry-run` to download matching products. To download the latest pair and automatically process while saving disk space:

```powershell
python sat-data\download_nisar.py --bbox 32.0,-6.5,41.0,10.5 --start 2026-10-01 --end 2026-11-14 --maturity PROVISIONAL --latest-pair --process-and-delete
```

> **Note**: `--process-and-delete` now safely removes the multi-gigabyte raw HDF5 files while **retaining both the color GeoTIFF and the numeric float32 GeoTIFF (`*_los_displacement_cm.tif`)**, enabling subsequent time-series analysis and 2D decomposition. Add `--discard-numeric` only if you need to discard numeric rasters.

### 2. Generate deformation rasters & 2D Vector Decomposition

```powershell
python manipulate\generate_deformation.py
```

For each GUNW product in `manipulate\data`, the processor writes two GeoTIFFs to `manipulate\output`:
- `*_los_displacement_color.tif` — RGBA visualization (blue for subsidence/motion away from satellite, red for uplift/motion toward satellite).
- `*_los_displacement_cm.tif` — Numeric float32 LOS displacement in centimeters.

#### 2D Vector Decomposition (Vertical Elevation Change vs. Plate Drift)
When both Ascending and Descending numeric rasters are available over an active tectonic zone, decompose them to isolate true vertical ground elevation change from horizontal plate rifting:

```powershell
python manipulate\generate_deformation.py --decompose-2d path\to\ascending_cm.tif path\to\descending_cm.tif
```

This outputs:
- `2D_DECOMPOSITION_vertical_elevation_change_cm.tif` & `*_color.tif` (pure uplift/subsidence)
- `2D_DECOMPOSITION_east_west_displacement_cm.tif` & `*_color.tif` (pure tectonic plate opening/strike-slip)

#### Time-Series Secular Velocity Inversion
To solve the connected network across multi-date acquisitions spanning at least one year:

```powershell
python manipulate\generate_deformation.py --time-series
```

### 3. Launch the Interactive Monitor

Run the primary launcher or the lightweight Folium viewer:

```powershell
# Full multi-panel dashboard
python start.py

# Or launch the lightweight Folium / Leaflet viewer
python viewer.py
```
*(or run `.\start.cmd`)*

The launcher creates missing deformation rasters, generates regional swath mosaics and urban focus layers, launches a local web server, and opens the dashboard in your default browser at `http://127.0.0.1:<port>/map/map.html` (or `viewer.html`).

### 4. Interactive Map Features

- **Live USGS Earthquakes**: Toggle live earthquake feeds (Past 7 Days or Past 30 Days) filtered by magnitude ($M \ge 2.5, 4.0, 5.0$). Circle radii scale with earthquake magnitude; marker colors indicate focal depth:
  - 🔴 **< 30 km**: Shallow crustal (highest surface shaking and rupture risk)
  - 🟠 **30 - 70 km**: Intermediate
  - 🟡 **70 - 150 km**: Deep crust / mantle
  - 🔵 **> 150 km**: Deep subduction zone
- **Active Fault Traces**: Displays high-resolution active tectonic fault lines (rift border faults, transform faults, subduction zones) with detailed tooltips showing fault name, kinematic type, slip rate, and seismic hazard.
- **Tectonic Plates**: Global Peter Bird (PB2002) plate boundaries.
- **InSAR Layer & Opacity Control**: Switch between Regional Swath (Ascending / Descending), urban presets, or time-series velocity, and adjust opacity via the top-bar slider.
- **View Navigation Presets**: One-click navigation to **InSAR Swath**, **East African Rift**, **Addis Ababa**, or **Global Plates**.

## Tests

Run the full unit test suite from the repository root:

```powershell
python -m unittest discover -s sat-data -p "test_*.py"
python -m unittest discover -s manipulate -p "test_*.py"
python -m unittest discover -s trans -p "test_*.py"
```
