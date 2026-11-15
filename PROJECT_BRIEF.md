# Project brief: NISAR Tectonic Deformation & Hazard Monitor

Purpose of this file: give an AI assistant everything needed to explain this project and
help build a 10-slide presentation. Everything here comes from the repository and from the
processing run done on 2026-11-15. Items marked **(unverified)** were not confirmed by running them.

## 1. One-paragraph summary

A pipeline plus an interactive web map that uses NASA-ISRO **NISAR** radar satellite data to
measure ground deformation along the **East African Rift** (Afar in the north to Mozambique in
the south). It finds new data after earthquakes, converts radar phase into centimeters of
ground motion, stitches the results into regional mosaics, and shows them next to live
earthquakes, active faults and critical infrastructure, with a hazard score for decision support.

## 2. The science, in plain language

- **NISAR**: a radar satellite that repeatedly images the same ground. Radar works through
  cloud and at night.
- **InSAR**: comparing two radar images of the same place taken days apart. The change in
  radar phase tells how far the ground moved toward or away from the satellite.
- **LOS (line of sight)**: the direction from the ground to the satellite. Measured motion is
  only the component along it, in centimeters.
- **GUNW product**: NISAR's "geocoded unwrapped interferogram", already placed on a map grid,
  with phase and a coherence (signal-quality) layer. Source: Alaska Satellite Facility (ASF).
- **L-band and S-band**: NISAR carries an L-band radar (NASA/JPL, about 24 cm) and an S-band radar
  (ISRO, about 3.2 GHz = 9.4 cm). Every product this project processed is L-band (ASF tags them L-SAR; the files'
  center frequency is 1.229 GHz = 24.4 cm). ASF carries the NASA L-band data; ISRO releases the S-band data
  separately through its Bhoonidhi portal, which this project has not used.
- **Ascending vs descending**: the satellite passes in two directions, so it sees the ground
  from opposite sides. Combining both separates **vertical** motion (uplift/subsidence) from
  **east-west** motion (plate opening). One direction alone cannot do this.
- **Why the rift**: the African plate is slowly splitting. Hazards there include earthquakes,
  magma/dike intrusions (Afar), and landslides.
- Displacement formula: phase x (speed of light / center frequency) / (4 pi). Color: blue =
  motion away from the satellite, red = toward it (the project README says to verify the sign
  convention before interpreting it this way). Low-coherence pixels are masked.

## 3. How it works, step by step (architecture slide)

1. **Start** - two ways in. A user can enter an area and dates on the dashboard (or on the command line) and start a search directly. Or the automatic **trigger** (`sat-data/trigger_search.py`) reads the live USGS earthquake feed, takes an
   epicenter, builds a bounding box (about +/-1 degree) and a time window (about +/-12 days),
   clips faults around it, and searches ASF for matching granules. Saves
   `map/data/latest_trigger_event.json`.
2. **Search** (`sat-data/download_nisar.py`): queries the ASF API for NISAR GUNW granules by
   area, dates, orbit direction and data maturity (VALIDATED / PROVISIONAL / BETA).
3. **Remote processing** (`sat-data/remote_hdf5.py` + `remote_process_product`): instead of
   downloading about 1 GB per granule, it streams only the needed HDF5 datasets using HTTP
   range requests (about 6% of the file). It resolves the download redirect once, reads in
   4 MB blocks with background read-ahead, and retries failed requests up to 6 times.
4. **Displacement** (`manipulate/generate_deformation.py`): phase to LOS displacement GeoTIFFs
   (a numeric float32 cm raster and a color RGBA raster per granule). Also supports:
   coherence masking, optional tropospheric correction from phase screens embedded in the
   product, multilooking, 2D decomposition (vertical + east-west) from an ascending and a
   descending raster, and time-series velocity inversion (needs acquisitions spanning 1 year).
5. **Mosaic** (`trans/prepare_overlay.py`): stitches all frames into one regional overlay per
   direction. Leveling matches offset and scale between neighbours; feathered blending hides
   frame boundaries; rows are resampled to Web Mercator so the overlay lines up on the map.
6. **Dashboard** (`map/map.html`, `map/map-state.js`, Leaflet; the site root `/` is a separate GAIA landing page and `/map` opens the dashboard): shows the overlays, live USGS
   earthquakes (circle size = magnitude, color = depth), GEM active fault traces clipped to the
   area, plate boundaries, Ethiopia critical infrastructure (15 features), several basemaps,
   an opacity slider and a layer selector. Buttons jump to the rift, Addis Ababa, or global view.
7. **Backend** (`server.py`, Flask): endpoints for trigger search, NISAR search, fault clipping,
   saving the Earthdata token, remote processing, and live job progress (server-sent events).
8. **Launchers**: `start.py` / `start.cmd` (full dashboard), `viewer.py` (lightweight Folium viewer).

## 4. Hazard analysis modules (extra layers of the project)

- `manipulate/multi_sensor_fusion.py`: combines InSAR deformation speed with other triggers to
  reduce false alarms and give a 0-100 risk score. Two modes: hydro-geological (landslides:
  InSAR creep + NASA GPM IMERG 72-hour rainfall + NASA SMAP soil moisture) and
  tectonic/volcanic (crustal opening or dike inflation + USGS earthquake swarm activity + SO2
  gas). Risk bands on the map: Green (0-29), Yellow (30-54), Orange (55-79), Red/Evacuate (80-100).
- `manipulate/inverse_velocity.py`: time-to-failure forecasting using the Fukuzono/Voight
  inverse-velocity method. When 1/velocity trends to zero, velocity is heading to infinity,
  which marks the projected failure time (slope collapse or pre-eruption pressurization).
- `sat-data/api_connectors.py`: connectors for ASF, GPM IMERG, SMAP, USGS and Sentinel-5P
  TROPOMI SO2, with offline fallback data so demos still work without internet.
- Case studies in `map/data/`: 2025 Afar rifting / Hayli Gubbi eruption (Afar Depression,
  magma dike, weekly timeline) and a Gofa landslide. **(unverified)** The weekly values in these
  files look like curated replay data for demonstration, not live measurements; confirm before
  presenting them as real results.

## 5. Results from the run (use for results slides)

- Scope: the latest acquisition pass of every satellite track over the rift
  (bbox 28 to 45 E, 20 S to 15 N), data maturity PROVISIONAL, search window 2026-09-01 to
  2026-11-15. ASF returned 1,507 granules in total; the latest pass per track is 339 of them.
- Processed all 339: 176 of 176 ascending, 163 of 163 descending. (Two granules, track 129
  frames 008 and 009, failed on the first attempts with network errors and succeeded on a later retry.)
- One granule is about 1 GB and covers about 330 x 330 km (about 110,000 km2) on a roughly
  80 m grid (about 4,200 x 4,100 pixels). Streaming one tested granule fetched about 59 MB of 981 MB.
- Outputs: per-granule displacement rasters in `manipulate/output/`, and two full-rift
  mosaics (ascending, descending) in `trans/` (about 2,400 x 4,100 pixels each) plus a small
  clipped Addis Ababa detail layer.
- Test suite: 80+ automated tests pass (as of the last run).

## 6. Engineering problems solved (challenges slide)

- The remote reader returned short reads at block boundaries, which corrupted compressed data
  ("filter returned failure during read"). Fixed so every read returns exactly the bytes asked.
- The HDF5 library holds a global lock even during network reads, so running granules in
  threads gave no speedup. Switched to separate processes.
- Streaming output files were named after a temp file, so the expected output was never found.
  The temporary file now carries the real granule name.
- Progress reported "done" after the first granule, ending the whole job early. Fixed.
- Overlay misplacement of up to about 0.28 degrees (about 30 km) from projection differences.
  Fixed by resampling to Web Mercator.
- Seams between neighbouring frames: fixed with scale-and-offset leveling and a smooth
  feathered blend.
- Failed granules no longer discard successful ones; the overlay is built from what succeeded.

## 7. Honest limitations (say these on a slide)

- Provisional-maturity data; use one maturity per time series.
- The mosaic combines passes from different days, so it is a coverage view, not one moment.
- LOS motion only; a long smooth color gradient across the swath is probably a data ramp
  (orbital or atmospheric) that is not removed. Ionospheric errors are not removed.
- The automatic reference pixel for time series is a heuristic, not a verified stable point.
- The map shows one overlay layer at a time (ascending or descending), not the combined
  vertical/east-west product.
- The project estimates deformation; it does not calculate earthquake probability.

## 8. Suggested 10-slide outline

1. Title and mission
2. The problem: the East African Rift and why monitoring matters
3. How InSAR and NISAR measure ground motion (LOS, ascending vs descending)
4. Data and scale (granules, size, coverage, 339-granule rift run)
5. Pipeline architecture (trigger, search, stream, process, mosaic, map)
6. Smart data access: streaming about 6% of each file instead of downloading
7. Mosaicking: blending frames and combining both directions
8. The interactive map and hazard layers (faults, earthquakes, infrastructure, risk bands)
9. Results and engineering challenges solved
10. Limitations, impact and next steps

## 9. Likely next steps (ideas, not commitments)

Combine ascending and descending data into the vertical and east-west layers for the whole rift;
remove the large-scale ramp; run time-series velocity once a year of acquisitions exists;
show ascending and descending together on the map; connect the risk score to live data feeds.
Application in AI: the displacement rasters are analysis-ready inputs for predictive models (the repo's
risk-scoring and inverse-velocity forecasting code are rule/physics-based, not trained ML). No model has been
trained; that needs a longer time series and labelled events.
