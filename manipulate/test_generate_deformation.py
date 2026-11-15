from datetime import datetime
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import rasterio
from rasterio.transform import from_origin

from generate_deformation import (
    apply_multilooking,
    decompose_2d_displacement,
    detrend_spatial_ramp,
    find_interferogram_triplets,
    generate_rasters,
    solve_deformation_timeseries,
    update_transform_multilook,
)


class GenerateDeformationTests(unittest.TestCase):
    def test_color_only_mode_does_not_create_numeric_raster(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "color_only_GUNW.h5"
            base = "science/LSAR/GUNW/grids/frequencyA"
            layer = f"{base}/unwrappedInterferogram/HH"
            with h5py.File(source, "w") as product:
                frequency_group = product.create_group(base)
                frequency_group.create_dataset("centerFrequency", data=1.25e9)
                polarization = product.create_group(layer)
                polarization.create_dataset(
                    "unwrappedPhase", data=np.array([[0, np.pi], [2 * np.pi, 3 * np.pi]])
                )
                polarization.create_dataset(
                    "coherenceMagnitude", data=np.full((2, 2), 0.9)
                )
                polarization.create_dataset("xCoordinates", data=[500000.0, 500080.0])
                polarization.create_dataset("yCoordinates", data=[4100000.0, 4099920.0])
                polarization.create_dataset("projection", data=32610)

            numeric_path, color_path = generate_rasters(
                source, root / "output", color_only=True
            )

            self.assertFalse(numeric_path.exists())
            self.assertTrue(color_path.is_file())

    def test_generates_georeferenced_numeric_and_color_rasters(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample_GUNW.h5"
            output_dir = root / "output"
            base = "science/LSAR/GUNW/grids/frequencyA"
            layer = f"{base}/unwrappedInterferogram/HH"
            phase = np.array(
                [[0, np.pi, 2 * np.pi], [3 * np.pi, 4 * np.pi, 5 * np.pi]],
                dtype=np.float32,
            )
            coherence = np.array(
                [[0.9, 0.9, 0.1], [0.9, 0.9, 0.9]], dtype=np.float32
            )

            with h5py.File(source, "w") as product:
                frequency_group = product.create_group(base)
                frequency_group.create_dataset("centerFrequency", data=1.25e9)
                polarization = product.create_group(layer)
                polarization.create_dataset("unwrappedPhase", data=phase)
                polarization.create_dataset("coherenceMagnitude", data=coherence)
                polarization.create_dataset(
                    "xCoordinates", data=[500000.0, 500080.0, 500160.0]
                )
                polarization.create_dataset(
                    "yCoordinates", data=[4099920.0, 4100000.0]
                )
                polarization.create_dataset("projection", data=32610)

            numeric_path, color_path = generate_rasters(
                source, output_dir, min_coherence=0.3
            )

            with rasterio.open(numeric_path) as numeric, rasterio.open(color_path) as color:
                expected_pixel_cm = (
                    3 * np.pi * (299_792_458.0 / 1.25e9) / (4 * np.pi) * 100
                )
                self.assertEqual(numeric.crs.to_epsg(), 32610)
                self.assertEqual(numeric.transform.a, 80)
                self.assertEqual(numeric.transform.e, -80)
                self.assertAlmostEqual(numeric.transform.c, 499960)
                self.assertAlmostEqual(numeric.transform.f, 4100040)
                self.assertAlmostEqual(
                    float(numeric.read(1)[0, 0]), expected_pixel_cm, places=4
                )
                self.assertEqual(float(numeric.read(1)[1, 2]), -9999.0)
                self.assertEqual(color.count, 4)
                self.assertEqual(color.colorinterp[-1], rasterio.enums.ColorInterp.alpha)
                self.assertEqual(int(color.read(4)[1, 2]), 0)
                self.assertEqual(color.tags()["units"], "centimeters")

    def test_rejects_products_without_valid_coherence_pixels(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "empty_GUNW.h5"
            base = "science/LSAR/GUNW/grids/frequencyA"
            layer = f"{base}/unwrappedInterferogram/HH"

            with h5py.File(source, "w") as product:
                frequency_group = product.create_group(base)
                frequency_group.create_dataset("centerFrequency", data=1.25e9)
                polarization = product.create_group(layer)
                polarization.create_dataset("unwrappedPhase", data=np.ones((2, 2)))
                polarization.create_dataset("coherenceMagnitude", data=np.zeros((2, 2)))
                polarization.create_dataset("xCoordinates", data=[0.0, 80.0])
                polarization.create_dataset("yCoordinates", data=[80.0, 0.0])
                polarization.create_dataset("projection", data=32610)

            with self.assertRaisesRegex(ValueError, "No valid pixels"):
                generate_rasters(source, root / "output")

    def test_subtracts_embedded_tropospheric_phase_screens(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "screened_GUNW.h5"
            dem_path = root / "dem.tif"
            output_dir = root / "output"
            base = "science/LSAR/GUNW/grids/frequencyA"
            layer = f"{base}/unwrappedInterferogram/HH"
            radar_grid = "science/LSAR/GUNW/metadata/radarGrid"

            with h5py.File(source, "w") as product:
                frequency_group = product.create_group(base)
                frequency_group.create_dataset("centerFrequency", data=1.25e9)
                polarization = product.create_group(layer)
                polarization.create_dataset(
                    "unwrappedPhase", data=np.full((2, 2), 10.0, dtype=np.float32)
                )
                polarization.create_dataset(
                    "coherenceMagnitude", data=np.full((2, 2), 0.9)
                )
                polarization.create_dataset(
                    "xCoordinates", data=[500000.0, 500080.0]
                )
                polarization.create_dataset(
                    "yCoordinates", data=[4099920.0, 4100000.0]
                )
                polarization.create_dataset("projection", data=32610)

                radar = product.create_group(radar_grid)
                radar.create_dataset(
                    "heightAboveEllipsoid", data=[0.0, 1000.0]
                )
                radar.create_dataset("xCoordinates", data=[500000.0, 500080.0])
                radar.create_dataset("yCoordinates", data=[4099920.0, 4100000.0])
                projection = radar.create_dataset("projection", data=32610)
                projection.attrs["epsg_code"] = 32610
                wet_screen = np.empty((2, 2, 2), dtype=np.float32)
                for height_index in range(2):
                    for y_index in range(2):
                        for x_index in range(2):
                            wet_screen[height_index, y_index, x_index] = (
                                height_index + 0.25 * y_index + 0.5 * x_index
                            )
                for name, values in (
                    ("wetTroposphericPhaseScreen", wet_screen),
                    (
                        "hydrostaticTroposphericPhaseScreen",
                        np.full((2, 2, 2), 3.0, dtype=np.float32),
                    ),
                ):
                    screen = radar.create_dataset(
                        name, data=values
                    )
                    screen.attrs["units"] = "radians"

            with rasterio.open(
                dem_path,
                "w",
                driver="GTiff",
                height=2,
                width=2,
                count=1,
                dtype="float32",
                crs="EPSG:32610",
                transform=rasterio.transform.from_origin(
                    499960.0, 4100040.0, 80.0, 80.0
                ),
                nodata=-9999.0,
            ) as dem:
                dem.write(np.full((2, 2), 500.0, dtype=np.float32), 1)

            numeric_path, _ = generate_rasters(
                source,
                output_dir,
                subtract_troposphere=True,
                dem_path=dem_path,
            )

            with rasterio.open(numeric_path) as numeric:
                expected_phase = np.array(
                    [[6.25, 5.75], [6.5, 6.0]], dtype=np.float32
                )
                expected_cm = (
                    expected_phase * (299_792_458.0 / 1.25e9) / (4 * np.pi) * 100
                )
                np.testing.assert_allclose(
                    numeric.read(1), expected_cm, rtol=0, atol=1e-4
                )
                self.assertIn(
                    "Embedded GUNW wet and hydrostatic phase screens",
                    numeric.tags()["tropospheric_correction"],
                )

    def test_inverts_baseline_network_and_fits_seasonal_time_series(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rasters_dir = root / "pairs"
            output_dir = root / "output"
            rasters_dir.mkdir()
            dates = [
                "20200101",
                "20200301",
                "20200501",
                "20200701",
                "20200901",
                "20201101",
                "20210101",
                "20210301",
            ]
            epoch_times = [
                datetime.strptime(date, "%Y%m%d")
                for date in dates
            ]
            elapsed_years = np.array(
                [
                    (date - epoch_times[0]).total_seconds() / (365.2425 * 86400)
                    for date in epoch_times
                ]
            )
            cumulative = (
                2.0 * elapsed_years
                + 3.0 * np.sin(2 * np.pi * elapsed_years)
                - 1.0 * np.cos(2 * np.pi * elapsed_years)
                + 0.5 * np.sin(4 * np.pi * elapsed_years)
                + 1.5 * np.cos(4 * np.pi * elapsed_years)
            )
            spatial_scale = np.array([[0.0, 1.0], [2.0, 3.0]])

            for index in range(len(dates) - 1):
                first, second = dates[index], dates[index + 1]
                filename = (
                    "NISAR_L2_PR_GUNW_032_136_D_087_033_2000_SH_"
                    f"{first}T000000_{first}T000100_"
                    f"{second}T000000_{second}T000100_P05023_N_F_J_001"
                    "_los_displacement_cm.tif"
                )
                displacement = (
                    cumulative[index + 1] - cumulative[index]
                ) * spatial_scale
                with rasterio.open(
                    rasters_dir / filename,
                    "w",
                    driver="GTiff",
                    width=2,
                    height=2,
                    count=1,
                    dtype="float32",
                    crs="EPSG:32637",
                    transform=from_origin(500000, 4100000, 80, 80),
                    nodata=-9999.0,
                ) as raster:
                    raster.write(displacement.astype(np.float32), 1)

            output_paths = solve_deformation_timeseries(
                list(rasters_dir.glob("*.tif")),
                output_dir,
                block_size=2,
            )

            self.assertEqual(len(output_paths), 3)
            with rasterio.open(output_paths[0]) as velocity:
                np.testing.assert_allclose(
                    velocity.read(1)[0, 1], 2.0, atol=1e-4
                )
                self.assertEqual(velocity.tags()["reference_pixel_row"], "0")
                self.assertEqual(velocity.tags()["reference_pixel_column"], "0")
            with rasterio.open(output_paths[1]) as seasonal:
                np.testing.assert_allclose(
                    seasonal.read(1)[0, 1], np.sqrt(12.5), atol=1e-4
                )
            with rasterio.open(output_paths[2]) as residual:
                self.assertLess(float(residual.read(1)[0, 1]), 1e-4)

    def test_decompose_2d_displacement_separates_vertical_and_east_west_motion(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            asc_path = root / "asc_cm.tif"
            desc_path = root / "desc_cm.tif"
            output_dir = root / "output"

            # Create overlapping Ascending and Descending numeric rasters
            # If d_up = 10.0 cm and d_east = 0.0 cm:
            # theta = 37 deg -> s_u = cos(37 deg) ~ 0.7986355
            # d_asc = s_u * d_up ~ 7.986 cm, d_desc = s_u * d_up ~ 7.986 cm
            s_u = float(np.cos(np.radians(37.0)))
            val = float(10.0 * s_u)
            data_asc = np.full((3, 3), val, dtype=np.float32)
            data_desc = np.full((3, 3), val, dtype=np.float32)

            meta = {
                "driver": "GTiff",
                "width": 3,
                "height": 3,
                "count": 1,
                "dtype": "float32",
                "crs": "EPSG:4326",
                "transform": from_origin(38.0, 9.0, 0.01, 0.01),
                "nodata": np.nan,
            }
            with rasterio.open(asc_path, "w", **meta) as ds:
                ds.write(data_asc, 1)
            with rasterio.open(desc_path, "w", **meta) as ds:
                ds.write(data_desc, 1)

            up_num, up_col, east_num, east_col = decompose_2d_displacement(
                asc_path, desc_path, output_dir, incidence_angle_deg=37.0, heading_deg=12.0
            )

            self.assertTrue(up_num.is_file())
            self.assertTrue(up_col.is_file())
            self.assertTrue(east_num.is_file())
            self.assertTrue(east_col.is_file())

            with rasterio.open(up_num) as ds_up, rasterio.open(east_num) as ds_east:
                arr_up = ds_up.read(1)
                arr_east = ds_east.read(1)
                valid = np.isfinite(arr_up) & np.isfinite(arr_east)
                self.assertTrue(valid.any())
                np.testing.assert_allclose(arr_up[valid], 10.0, atol=1e-3)
                np.testing.assert_allclose(arr_east[valid], 0.0, atol=1e-3)

    def test_find_interferogram_triplets_identifies_closed_triangles(self) -> None:
        edges = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)]
        triplets = find_interferogram_triplets(edges)
        self.assertEqual(len(triplets), 4)
        loop_dates = [(t[0], t[1], t[2]) for t in triplets]
        self.assertIn((0, 1, 2), loop_dates)
        self.assertIn((0, 1, 3), loop_dates)
        self.assertIn((1, 2, 3), loop_dates)
        self.assertIn((0, 2, 3), loop_dates)

    def test_temporal_triplet_phase_closure_drops_unwrapping_blunder(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rasters_dir = root / "pairs"
            output_dir = root / "output"
            rasters_dir.mkdir()
            dates = [
                "20200101",
                "20200301",
                "20200501",
                "20200701",
                "20200901",
                "20201101",
                "20210101",
                "20210301",
            ]
            epoch_times = [datetime.strptime(d, "%Y%m%d") for d in dates]
            elapsed_years = np.array(
                [(d - epoch_times[0]).total_seconds() / (365.2425 * 86400) for d in epoch_times]
            )
            # Ground truth: linear velocity 2.0 cm/yr + annual/semi-annual cycles
            cumulative = (
                2.0 * elapsed_years
                + 3.0 * np.sin(2 * np.pi * elapsed_years)
                - 1.0 * np.cos(2 * np.pi * elapsed_years)
                + 0.5 * np.sin(4 * np.pi * elapsed_years)
                + 1.5 * np.cos(4 * np.pi * elapsed_years)
            )
            spatial_scale = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

            # Generate spanning tree edges plus redundant edge (0, 2) that forms closed triangle (0 -> 1 -> 2)
            edge_pairs = [
                (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
                (0, 2),
            ]

            wavelength_m = 299_792_458.0 / 1.2575e9
            unwrapping_cycle_cm = (wavelength_m / 2.0) * 100.0  # 1 cycle = ~11.92 cm

            for i, j in edge_pairs:
                first, second = dates[i], dates[j]
                filename = (
                    "NISAR_L2_PR_GUNW_032_136_D_087_033_2000_SH_"
                    f"{first}T000000_{first}T000100_"
                    f"{second}T000000_{second}T000100_P05023_N_F_J_001"
                    "_los_displacement_cm.tif"
                )
                disp = (cumulative[j] - cumulative[i]) * spatial_scale
                # Introduce intentional 2*pi unwrapping blunder on edge (0, 2)
                if (i, j) == (0, 2):
                    disp = disp + unwrapping_cycle_cm

                with rasterio.open(
                    rasters_dir / filename,
                    "w",
                    driver="GTiff",
                    width=2,
                    height=2,
                    count=1,
                    dtype="float32",
                    crs="EPSG:32637",
                    transform=from_origin(500000, 4100000, 80, 80),
                    nodata=-9999.0,
                ) as raster:
                    raster.write(disp.astype(np.float32), 1)

            output_paths = solve_deformation_timeseries(
                list(rasters_dir.glob("*.tif")),
                output_dir,
                block_size=2,
            )

            self.assertEqual(len(output_paths), 3)
            # Verify the corrupted pair was dropped and velocity is correctly recovered
            with rasterio.open(output_paths[0]) as velocity:
                np.testing.assert_allclose(
                    velocity.read(1)[0, 1], 2.0, atol=1e-3
                )
                self.assertEqual(velocity.tags()["pairs_dropped"], "1")
                self.assertEqual(velocity.tags()["triplets_failed"], "1")

            summary_file = output_dir / "NISAR_L2_PR_GUNW_032_136_D_087_033_2000_SH_triplet_closure_summary.json"
            self.assertTrue(summary_file.is_file())
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            self.assertEqual(summary["triplets_total"], 1)
            self.assertEqual(summary["triplets_failed"], 1)
            self.assertEqual(len(summary["dropped_pairs"]), 1)
            self.assertIn("20200101", summary["dropped_pairs"][0])
            self.assertIn("20200501", summary["dropped_pairs"][0])

    def test_temporal_triplet_phase_closure_downweights_when_cannot_drop(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rasters_dir = root / "pairs"
            output_dir = root / "output"
            rasters_dir.mkdir()
            dates = [
                "20200101",
                "20200301",
                "20200501",
                "20200701",
                "20200901",
                "20201101",
                "20210101",
                "20210301",
            ]
            epoch_times = [datetime.strptime(d, "%Y%m%d") for d in dates]
            elapsed_years = np.array(
                [(d - epoch_times[0]).total_seconds() / (365.2425 * 86400) for d in epoch_times]
            )
            cumulative = 2.0 * elapsed_years
            spatial_scale = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

            # Triangle on dates (0, 1, 2) plus chain (2..7)
            # Both (0, 2) and (0, 1) carry large blunders
            edge_pairs = [
                (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
                (0, 2),
            ]
            wavelength_m = 299_792_458.0 / 1.2575e9
            unwrapping_cycle_cm = (wavelength_m / 2.0) * 100.0

            for i, j in edge_pairs:
                first, second = dates[i], dates[j]
                filename = (
                    "NISAR_L2_PR_GUNW_032_136_D_087_033_2000_SH_"
                    f"{first}T000000_{first}T000100_"
                    f"{second}T000000_{second}T000100_P05023_N_F_J_001"
                    "_los_displacement_cm.tif"
                )
                disp = (cumulative[j] - cumulative[i]) * spatial_scale
                # Corrupt (0, 2) on pixel (1, 1)
                if (i, j) == (0, 2):
                    disp = disp + unwrapping_cycle_cm

                with rasterio.open(
                    rasters_dir / filename,
                    "w",
                    driver="GTiff",
                    width=2,
                    height=2,
                    count=1,
                    dtype="float32",
                    crs="EPSG:32637",
                    transform=from_origin(500000, 4100000, 80, 80),
                    nodata=-9999.0,
                ) as raster:
                    raster.write(disp.astype(np.float32), 1)

            # Test that calling solve_deformation_timeseries with strict closure threshold drops the blunder
            output_paths = solve_deformation_timeseries(
                list(rasters_dir.glob("*.tif")),
                output_dir,
                block_size=2,
                closure_threshold_rad=1.0,  # Strict threshold
            )
            self.assertEqual(len(output_paths), 3)
            summary_file = output_dir / "NISAR_L2_PR_GUNW_032_136_D_087_033_2000_SH_triplet_closure_summary.json"
            self.assertTrue(summary_file.is_file())
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(summary["dropped_pairs"]), 1)

    def test_empirical_tropospheric_correction_fallback(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "no_weather_GUNW.h5"
            dem_path = root / "dem.tif"
            output_dir = root / "output"
            base = "science/LSAR/GUNW/grids/frequencyA"
            layer = f"{base}/unwrappedInterferogram/HH"

            # 4x4 grid with linear elevation-phase relationship: phi = 0.005 * h + 1.2
            # Note: with yCoordinates ascending, dataset row 0 corresponds to lowest y (bottom of DEM).
            # We define DEM row 0 at top y (descending), so h_oriented for dataset row r, col c
            # matches DEM[3 - r, c].
            dem_data = np.array([
                [1600.0, 1500.0, 1400.0, 1300.0],
                [1200.0, 1100.0, 1000.0, 900.0],
                [800.0, 700.0, 600.0, 500.0],
                [400.0, 300.0, 200.0, 100.0],
            ], dtype=np.float32)

            true_a = 0.005
            true_b = 1.2
            # Flip vertically to match phase_dataset row orientation (y ascending)
            h_phase_grid = dem_data[::-1, :]
            phase_vals = true_a * h_phase_grid + true_b
            # Add one outlier (e.g. localized deformation zone) at [1, 1] to test MAD screening
            phase_vals[1, 1] += 30.0

            coherence_vals = np.full((4, 4), 0.9, dtype=np.float32)

            with h5py.File(source, "w") as product:
                frequency_group = product.create_group(base)
                frequency_group.create_dataset("centerFrequency", data=1.25e9)
                polarization = product.create_group(layer)
                polarization.create_dataset("unwrappedPhase", data=phase_vals)
                polarization.create_dataset("coherenceMagnitude", data=coherence_vals)
                polarization.create_dataset(
                    "xCoordinates", data=[500000.0, 500080.0, 500160.0, 500240.0]
                )
                polarization.create_dataset(
                    "yCoordinates", data=[4099760.0, 4099840.0, 4099920.0, 4100000.0]
                )
                polarization.create_dataset("projection", data=32610)
                # Intentionally omitting science/LSAR/GUNW/metadata/radarGrid to trigger empirical fallback

            with rasterio.open(
                dem_path,
                "w",
                driver="GTiff",
                height=4,
                width=4,
                count=1,
                dtype="float32",
                crs="EPSG:32610",
                transform=rasterio.transform.from_origin(
                    499960.0, 4100040.0, 80.0, 80.0
                ),
                nodata=-9999.0,
            ) as dem:
                dem.write(dem_data, 1)

            numeric_path, _ = generate_rasters(
                source,
                output_dir,
                subtract_troposphere=True,
                dem_path=dem_path,
            )

            with rasterio.open(numeric_path) as numeric:
                tags = numeric.tags()
                self.assertEqual(tags.get("troposphere_method"), "empirical_phase_elevation")
                self.assertIn(
                    "Empirical linear phase-elevation model",
                    tags.get("tropospheric_correction", ""),
                )
                fitted_slope = float(tags["troposphere_phase_elevation_slope_rad_m"])
                fitted_intercept = float(tags["troposphere_phase_elevation_intercept_rad"])
                self.assertAlmostEqual(fitted_slope, true_a, places=4)
                self.assertAlmostEqual(fitted_intercept, true_b, places=2)

                data = numeric.read(1)
                # In output raster (north-up), data matches dem_data layout.
                # Non-outlier pixels should have residual phase ~ 0, hence displacement ~ 0 cm.
                clean_mask = np.ones((4, 4), dtype=bool)
                # Outlier was at phase_vals[1, 1], which corresponds to output raster row (3 - 1) = 2, col 1
                clean_mask[2, 1] = False
                np.testing.assert_allclose(data[clean_mask], 0.0, atol=0.05)

    def test_detrend_spatial_ramp(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raster_path = root / "sample_los_displacement_cm.tif"
            out_dir = root / "detrended"

            # 8x8 grid with known linear tilt: z = 0.05 * x - 0.03 * y + 4.0
            height, width = 8, 8
            cols, rows = np.meshgrid(np.arange(width), np.arange(height))
            true_a = 0.05
            true_b = -0.03
            true_c = 4.0
            disp = true_a * cols + true_b * rows + true_c

            # Insert water body / nodata at [0, 0]
            disp[0, 0] = -9999.0

            # Insert localized high-gradient deformation anomaly at [4, 4]
            disp[4, 4] += 15.0

            transform = from_origin(500000.0, 4100000.0, 80.0, 80.0)
            profile = {
                "driver": "GTiff",
                "width": width,
                "height": height,
                "count": 1,
                "dtype": "float32",
                "crs": "EPSG:32610",
                "transform": transform,
                "nodata": -9999.0,
            }

            with rasterio.open(raster_path, "w", **profile) as dst:
                dst.write(disp.astype(np.float32), 1)

            det_path, qc_info = detrend_spatial_ramp(raster_path, output_dir=out_dir)

            self.assertTrue(det_path.is_file())
            self.assertAlmostEqual(qc_info["ramp_a_cm_per_px"], true_a, places=3)
            self.assertAlmostEqual(qc_info["ramp_b_cm_per_px"], true_b, places=3)
            self.assertAlmostEqual(qc_info["ramp_c_offset_cm"], true_c, places=2)

            with rasterio.open(det_path) as dst:
                data = dst.read(1)
                tags = dst.tags()
                self.assertEqual(tags.get("spatial_ramp_detrended"), "true")
                self.assertIn("ramp_plane_model", tags)

                # Nodata pixel remains nodata
                self.assertEqual(data[0, 0], -9999.0)

                # Background pixels detrended to ~0 cm
                valid_bg = np.ones((height, width), dtype=bool)
                valid_bg[0, 0] = False
                valid_bg[4, 4] = False
                np.testing.assert_allclose(data[valid_bg], 0.0, atol=0.05)

                # Localized anomaly preserved at ~15 cm
                self.assertAlmostEqual(data[4, 4], 15.0, delta=0.1)

    def test_solve_deformation_timeseries_with_detrend_spatial_ramp(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rasters_dir = root / "pairs"
            output_dir = root / "output"
            rasters_dir.mkdir(parents=True)

            dates = [
                datetime(2020, 1, 1),
                datetime(2020, 3, 1),
                datetime(2020, 5, 1),
                datetime(2020, 7, 1),
                datetime(2020, 9, 1),
                datetime(2020, 11, 1),
                datetime(2021, 2, 1),
            ]
            pairs = [
                (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                (0, 2),
            ]

            for i, j in pairs:
                first = dates[i].strftime("%Y%m%dT000000_%Y%m%dT000100")
                second = dates[j].strftime("%Y%m%dT000000_%Y%m%dT000100")
                filename = (
                    f"NISAR_L2_PR_GUNW_032_136_D_087_033_2000_SH_{first}_{second}_"
                    "P05023_N_F_J_001_los_displacement_cm.tif"
                )
                # 4x4 grid with spatial ramp: 0.02 * col + 0.01 * row
                cols, rows = np.meshgrid(np.arange(4), np.arange(4))
                ramp = 0.02 * cols + 0.01 * rows
                disp = ramp + float(j - i) * 1.5

                with rasterio.open(
                    rasters_dir / filename,
                    "w",
                    driver="GTiff",
                    width=4,
                    height=4,
                    count=1,
                    dtype="float32",
                    crs="EPSG:32637",
                    transform=from_origin(500000, 4100000, 80, 80),
                    nodata=-9999.0,
                ) as raster:
                    raster.write(disp.astype(np.float32), 1)

            output_paths = solve_deformation_timeseries(
                list(rasters_dir.glob("*.tif")),
                output_dir,
                block_size=4,
                detrend_spatial_ramp=True,
            )
            self.assertEqual(len(output_paths), 3)

            # Check that detrended rasters were generated
            detrended_folder = output_dir / "detrended"
            self.assertTrue(detrended_folder.is_dir())
            self.assertEqual(len(list(detrended_folder.glob("*.tif"))), len(pairs))

            # Check that detrending QC summary JSON was created
            summary_json = output_dir / "NISAR_L2_PR_GUNW_032_136_D_087_033_2000_SH_spatial_ramp_detrend_summary.json"
            self.assertTrue(summary_json.is_file())
            qc_data = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(len(qc_data), len(pairs))

            # Verify that output time series rasters have spatial_ramp_detrended tag
            with rasterio.open(output_paths[0]) as out_ds:
                self.assertEqual(out_ds.tags().get("spatial_ramp_detrended"), "true")

    def test_apply_multilooking_math_and_nodata(self) -> None:
        data = np.array(
            [
                [1.0, 2.0, 5.0, 6.0],
                [3.0, 4.0, 7.0, 8.0],
                [9.0, 10.0, 13.0, 14.0],
                [11.0, 12.0, 15.0, 16.0],
            ],
            dtype=np.float32,
        )
        ml = apply_multilooking(data, factor=2)
        expected = np.array([[2.5, 6.5], [10.5, 14.5]], dtype=np.float32)
        np.testing.assert_allclose(ml, expected, rtol=1e-5)

        # Partial nodata handling
        data_nodata = data.copy()
        data_nodata[0, 0] = -9999.0  # Top-left block now has 2, 3, 4 -> mean = 3.0
        # Bottom-right block all nodata
        data_nodata[2:, 2:] = -9999.0

        ml_nodata = apply_multilooking(data_nodata, factor=2, nodata=-9999.0)
        self.assertAlmostEqual(float(ml_nodata[0, 0]), 3.0, places=4)
        self.assertAlmostEqual(float(ml_nodata[0, 1]), 6.5, places=4)
        self.assertAlmostEqual(float(ml_nodata[1, 0]), 10.5, places=4)
        self.assertEqual(float(ml_nodata[1, 1]), -9999.0)

        # Mask mode
        mask = np.array(
            [
                [True, True, False, False],
                [True, False, False, False],
                [False, False, True, True],
                [False, False, True, True],
            ]
        )
        ml_mask = apply_multilooking(mask, factor=2, is_mask=True)
        # Top-left has 3/4 True -> True; Top-right has 0/4 True -> False; Bottom-left 0/4 -> False; Bottom-right 4/4 -> True
        self.assertTrue(ml_mask[0, 0])
        self.assertFalse(ml_mask[0, 1])
        self.assertFalse(ml_mask[1, 0])
        self.assertTrue(ml_mask[1, 1])

        # 3D stack multilooking
        stack_3d = np.stack([data, data * 2], axis=0)
        ml_3d = apply_multilooking(stack_3d, factor=2)
        self.assertEqual(ml_3d.shape, (2, 2, 2))
        np.testing.assert_allclose(ml_3d[0], expected, rtol=1e-5)
        np.testing.assert_allclose(ml_3d[1], expected * 2, rtol=1e-5)

        # Trimming remainder
        odd_data = np.zeros((5, 5), dtype=np.float32)
        ml_odd = apply_multilooking(odd_data, factor=2)
        self.assertEqual(ml_odd.shape, (2, 2))

    def test_update_transform_multilook(self) -> None:
        orig = from_origin(500000.0, 4100000.0, 80.0, 80.0)
        updated = update_transform_multilook(orig, factor=4)
        self.assertEqual(updated.a, 320.0)
        self.assertEqual(updated.e, -320.0)
        self.assertEqual(updated.c, 500000.0)
        self.assertEqual(updated.f, 4100000.0)

    def test_generate_rasters_with_multilook(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "multilook_GUNW.h5"
            output_dir = root / "output"
            base = "science/LSAR/GUNW/grids/frequencyA"
            layer = f"{base}/unwrappedInterferogram/HH"
            phase = np.array(
                [
                    [0.0, np.pi, 2 * np.pi, 3 * np.pi],
                    [np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi],
                    [2 * np.pi, 3 * np.pi, 4 * np.pi, 5 * np.pi],
                    [3 * np.pi, 4 * np.pi, 5 * np.pi, 6 * np.pi],
                ],
                dtype=np.float32,
            )
            coherence = np.full((4, 4), 0.9, dtype=np.float32)

            with h5py.File(source, "w") as product:
                frequency_group = product.create_group(base)
                frequency_group.create_dataset("centerFrequency", data=1.25e9)
                polarization = product.create_group(layer)
                polarization.create_dataset("unwrappedPhase", data=phase)
                polarization.create_dataset("coherenceMagnitude", data=coherence)
                polarization.create_dataset(
                    "xCoordinates", data=[500000.0, 500080.0, 500160.0, 500240.0]
                )
                polarization.create_dataset(
                    "yCoordinates", data=[4100240.0, 4100160.0, 4100080.0, 4100000.0]
                )
                polarization.create_dataset("projection", data=32610)

            numeric_path, color_path = generate_rasters(
                source, output_dir, multilook=2
            )

            with rasterio.open(numeric_path) as numeric, rasterio.open(color_path) as color:
                self.assertEqual(numeric.width, 2)
                self.assertEqual(numeric.height, 2)
                self.assertEqual(numeric.transform.a, 160.0)
                self.assertEqual(numeric.transform.e, -160.0)
                self.assertEqual(numeric.tags().get("multilook_factor"), "2")
                self.assertEqual(color.width, 2)
                self.assertEqual(color.height, 2)

    def test_multilook_on_flipped_grid_keeps_pixels_registered(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "multilook_flipped_GUNW.h5"
            base = "science/LSAR/GUNW/grids/frequencyA"
            layer = f"{base}/unwrappedInterferogram/HH"
            # 5x5 grid stored south-up and east-first, so both axes are flipped and
            # one row and one column are trimmed by a factor-2 multilook.
            x_coordinates = 500000.0 + 80.0 * np.arange(5)[::-1]
            y_coordinates = 4100000.0 + 80.0 * np.arange(5)
            phase = np.zeros((5, 5), dtype=np.float32)
            phase[0:2, 0:2] = 4 * np.pi  # southernmost rows, easternmost columns

            with h5py.File(source, "w") as product:
                frequency_group = product.create_group(base)
                frequency_group.create_dataset("centerFrequency", data=1.25e9)
                polarization = product.create_group(layer)
                polarization.create_dataset("unwrappedPhase", data=phase)
                polarization.create_dataset(
                    "coherenceMagnitude", data=np.full((5, 5), 0.9, dtype=np.float32)
                )
                polarization.create_dataset("xCoordinates", data=x_coordinates)
                polarization.create_dataset("yCoordinates", data=y_coordinates)
                polarization.create_dataset("projection", data=32610)

            numeric_path, _ = generate_rasters(
                source, root / "output", multilook=2, range_cm=10.0
            )

            with rasterio.open(numeric_path) as numeric:
                data = numeric.read(1)
                self.assertEqual(data.shape, (2, 2))
                # The marked block is the south-east output pixel...
                self.assertGreater(float(data[1, 1]), 1.0)
                self.assertAlmostEqual(float(data[0, 0]), 0.0, places=4)
                # ...and it must sit on the ground where the source pixels were:
                # x 500240..500400 (pixel edges 500200..500360), y 4099960..4100120.
                bounds = numeric.bounds
                self.assertAlmostEqual(bounds.right, 500360.0)
                self.assertAlmostEqual(bounds.bottom, 4099960.0)
                self.assertAlmostEqual(bounds.left, 500040.0)
                self.assertAlmostEqual(bounds.top, 4100280.0)

    def test_solve_deformation_timeseries_multicore_parallel_and_multilook(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rasters_dir = root / "rasters"
            rasters_dir.mkdir()
            output_dir = root / "output"

            dates = [
                datetime(2020, 1, 1),
                datetime(2020, 3, 1),
                datetime(2020, 5, 1),
                datetime(2020, 7, 1),
                datetime(2020, 9, 1),
                datetime(2020, 11, 1),
                datetime(2021, 2, 1),
            ]
            pairs = [
                (0, 1),
                (0, 2),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
            ]

            # 8x8 grid rasters for block parallel processing
            for i, j in pairs:
                first = dates[i].strftime("%Y%m%dT000000_%Y%m%dT000100")
                second = dates[j].strftime("%Y%m%dT000000_%Y%m%dT000100")
                filename = (
                    f"NISAR_L2_PR_GUNW_032_136_D_087_033_2000_SH_{first}_{second}_"
                    "P05023_N_F_J_001_los_displacement_cm.tif"
                )
                disp = np.full((8, 8), float(j - i) * 1.5, dtype=np.float32)

                with rasterio.open(
                    rasters_dir / filename,
                    "w",
                    driver="GTiff",
                    width=8,
                    height=8,
                    count=1,
                    dtype="float32",
                    crs="EPSG:32637",
                    transform=from_origin(500000, 4100000, 80, 80),
                    nodata=-9999.0,
                ) as raster:
                    raster.write(disp, 1)

            # Invert with 4 workers and multilook=2
            output_paths = solve_deformation_timeseries(
                list(rasters_dir.glob("*.tif")),
                output_dir,
                block_size=4,
                num_workers=4,
                multilook=2,
            )
            self.assertEqual(len(output_paths), 3)

            for path in output_paths:
                with rasterio.open(path) as src:
                    # 8x8 multilooked by 2 becomes 4x4
                    self.assertEqual(src.width, 4)
                    self.assertEqual(src.height, 4)
                    self.assertEqual(src.tags().get("multicore_workers"), "4")
                    self.assertEqual(src.tags().get("multilook_factor"), "2")




if __name__ == "__main__":
    unittest.main()
