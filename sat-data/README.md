# NISAR deformation-data downloader

`download_nisar.py` searches NASA's Alaska Satellite Facility (ASF) archive for
NISAR **GUNW** products: geocoded, unwrapped interferograms that include
coherence information. This is a more direct starting point for surface
deformation monitoring than downloading raw imagery or several alternate
product types.

The downloader searches only standard-production GUNW products at one
selected data-maturity level. It removes duplicate catalog records, skips
non-empty product files already in the output directory, and writes downloads
to a temporary file before renaming them. It does not calculate deformation
rate or earthquake probability; a time-series analysis of the downloaded
interferograms is still required.

## Setup

Use Python 3.14 for both scripts. The current Python 3.15 environment does not
have an h5py Windows wheel, which is required by the raster processor.

```powershell
py -3.14 -m pip install -r sat-data\requirements.txt -r manipulate\requirements.txt
```

Create an Earthdata Login account before downloading. If
`EARTHDATA_TOKEN` is not set, the downloader first checks the ignored local file
`sat-data\earthdata_token.txt`, then prompts for a bearer token using hidden
terminal input. Put only the token on the first line of that file; it is
gitignored and must not be committed. Product searches and dry runs do not need
a token.

## Search and download

Example area of interest (bbox order is west, south, east, north):

```powershell
py -3.14 sat-data\download_nisar.py --bbox -124,32,-114,42 --start 2026-10-21 --end 2026-11-14 --dry-run
```

Remove `--dry-run` to download matching products to `manipulate\data`. The
default maturity is `VALIDATED`; use `--maturity PROVISIONAL` to search older
provisional acquisitions where validated products are not yet available. Use
one maturity consistently within a time series because different processing
releases can differ. The downloader records the selected maturity in the output
directory and refuses to mix maturity levels there. Use a separate output
directory if you need to keep multiple maturities.

Add `--latest-only` to download just the matching product with the latest
acquisition end time instead of every product in the search area. This is one
GUNW granule, not a mosaic covering the whole area.

For a wider map, `--latest-pair` downloads all matching granules from the most
recent acquisition-date pair. The map builder mosaics these into separate
ascending and descending layers; choose the orbit direction in the map rather
than treating the two viewing geometries as one measurement. Product
footprints may still leave gaps in the selected region.

For low-storage operation, add `--process-and-delete`. Each selected product
is downloaded (or reused if already present), processed individually, and its
HDF5 source is deleted only after a non-empty color raster is verified. The
numeric displacement raster is removed to save space; color rasters remain for
the overlay. This is intentionally destructive to the source HDF5 products.

The ASF archive may require Earthdata authentication to download. Set
`EARTHDATA_TOKEN` in the environment, then rerun the command without
`--dry-run`.
Downloads show a percentage and transferred size when ASF provides a content
length. If it does not, the downloader shows the amount received.
After each selected product is skipped, downloaded, or processed, it also
prints the completed-file count, remaining count, and how many new files have
been downloaded during the current run.

## Dynamic Global Active Fault Clipping

When downloading or searching across any global earthquake area (`--bbox west,south,east,north`),
the pipeline automatically clips the master global active fault database (GEM / USGS) to the exact
viewing window and saves it to `map/faults_clip.geojson` for the interactive map viewer:

```powershell
# Run standalone fault clipping
py -3.14 sat-data\fault_clipper.py --bbox -124,32,-114,42 --output map\faults_clip.geojson
```

Add `--no-clip-faults` to `download_nisar.py` if you wish to bypass fault clipping.

## Automated "Trigger-to-Search" Monitoring

`trigger_search.py` automates the loop connecting live USGS earthquake triggers to NASA ASF searches:
1. Pings the live USGS earthquake feed (`4.5_month.geojson` or `significant_month.geojson`).
2. Extracts epicenter coordinates $(lat, lon)$ and event origin time $t_0$.
3. Dynamically calculates spatial bounding box $(\pm 1^\circ)$ and co-seismic temporal window $(\pm 12\text{ days})$.
4. Clips active fault traces around the epicenter.
5. Queries NASA ASF DAAC for matching co-seismic NISAR GUNW granules.
6. Saves `map/data/latest_trigger_event.json` for live map status display.

```powershell
# Run a dry search on the latest M >= 4.5 earthquake
py -3.14 sat-data\trigger_search.py --min-mag 4.5 --dry-run

# Run full co-seismic download and processing for significant earthquakes
py -3.14 sat-data\trigger_search.py --feed significant_month --download --process-and-delete
```

## Generate deformation rasters

After downloading products, run:

```powershell
py -3.14 manipulate\generate_deformation.py
```

The processor reads each GUNW HDF5 product in `manipulate\data` and writes a
georeferenced, colorized GeoTIFF plus a numeric LOS-displacement GeoTIFF to
`manipulate\output`. It converts unwrapped phase to centimeters using each
product's center frequency and masks pixels below 0.3 coherence by default.
See [the manipulation guide](../manipulate/README.md) for options and
interpretation notes.
