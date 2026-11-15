# NISAR deformation raster generation

The downloader saves GUNW HDF5 products to `manipulate\data`. Use Python 3.14
because the current Python 3.15 environment does not have an h5py Windows wheel.
Install dependencies from the repository root and generate rasters with:

```powershell
py -3.14 -m pip install -r sat-data\requirements.txt -r manipulate\requirements.txt
py -3.14 manipulate\generate_deformation.py
```

For each product, the processor writes two georeferenced GeoTIFFs to
`manipulate\output`:

- `*_los_displacement_color.tif`: RGBA color raster, with blue for negative
  phase-derived displacement and red for positive.
- `*_los_displacement_cm.tif`: numeric float raster in centimeters.

The default polarization is HH when present, otherwise VV. Adjust the coherence
mask, color stretch, polarization, input, or output directory as needed:

```powershell
py -3.14 manipulate\generate_deformation.py --min-coherence 0.4 --range-cm 5
py -3.14 manipulate\generate_deformation.py --polarization VV --input-dir path\to\data
```

For products containing NISAR's embedded wet and hydrostatic tropospheric phase
screens, subtract those screens before converting phase to displacement:

```powershell
py -3.14 manipulate\generate_deformation.py --subtract-troposphere --dem-path path\to\dem.tif
```

The DEM must be a single-band geocoded raster in meters above the WGS84
ellipsoid and cover the GUNW grid. The processor interpolates the embedded
three-dimensional screen grids in height and map coordinates onto the GUNW
grid, then subtracts their sum from `unwrappedPhase`. These are phase values in
radians on NISAR's radar-grid geometry, so no additional incidence-angle factor
is applied. It fails if the screens
or usable DEM coverage are missing; it does not estimate an elevation-based
phase regression. This uses the correction layers supplied in GUNW, not an
independent PyAPS/ERA5 download. Without `--subtract-troposphere`, existing
uncorrected processing is unchanged. Confirm that subtraction matches the
phase-screen sign convention for the product version before scientific use.

The displacement is derived from the GUNW `unwrappedPhase` and `centerFrequency`
using `phase * (speed of light / center frequency) / (4 * pi)`. With the
tropospheric option enabled, `phase` means the input phase minus the embedded
wet and hydrostatic phase screens. The output retains the source phase sign;
verify its convention before interpreting positive or negative as motion
toward or away from the satellite. This remains a relative LOS measurement;
ionospheric effects and other residual errors are not removed. The color
stretch uses the 98th percentile of sampled valid absolute displacements unless
`--range-cm` is supplied.

To prepare the newest scene covering the Addis Ababa city center as a
city-cropped overlay, run:

```powershell
py -3.14 trans\prepare_overlay.py --mosaic
```

For a multi-date stack, first download and process all desired GUNW products,
then run the generator with `--time-series` to invert the numeric pair rasters:

```powershell
py -3.14 manipulate\generate_deformation.py --time-series
py -3.14 trans\prepare_overlay.py --mosaic --time-series
```

For an existing numeric pair-raster stack, add `--time-series-only` to skip
reprocessing the source HDF5 files. The launcher attempts this automatically
and exposes available velocity rasters in the map's **Layer** selector.

The inversion constructs a reference-to-secondary baseline matrix from each
product's four acquisition timestamps and solves each connected, valid pixel
network by least squares. It groups rasters by their complete pre-timestamp
product identity, which keeps different orbit/frame geometries separate.
Acquisitions must span at least 365 days (configurable upward) and provide
enough independent dates for a full-rank linear, annual, and semi-annual model.
The automatic reference is the pixel with maximum valid edge coverage and then
minimum mean absolute pair displacement; it is only a heuristic, not a
verified stable ground point. Check it before scientific interpretation.

Each network writes `*_secular_tectonic_velocity_cm_yr.tif`,
`*_seasonal_amplitude_cm.tif` (root-sum-square of annual and semi-annual
harmonic amplitudes), and `*_residual_noise_map.tif` (temporal fit RMS in cm).
The map's `--time-series` mode overlays the secular-velocity layers per track;
seasonal and residual rasters remain available as numeric GeoTIFFs. This is an
equal-weight inversion; coherence is used only through the per-pair mask, not
as a continuous inversion weight.

Then open `map\map.html`. The map is restricted to Addis Ababa and nearby
surroundings; the LOS deformation checkbox controls visibility, and the Layer
selector switches between latest-pair displacement and available time-series
velocity. Pass
`--raster path\to\product_los_displacement_color.tif` to prepare one specific
product instead.

The map uses only scenes that cover the city center and crops the overlay to
the metro area rather than resampling the entire regional swath. Current GUNW
rasters have an 80 m grid spacing, so zooming focuses the view but cannot add
finer spatial information than the source product contains.

See the [ASF GUNW product guide](https://nisar-docs.asf.alaska.edu/gunw/) for
the product layers and their limitations.
