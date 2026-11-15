import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from prepare_overlay import (
    prepare_overlay,
    prepare_overlay_mosaic,
    prepare_timeseries_overlay,
)


class PrepareOverlayTests(unittest.TestCase):
    def test_reprojects_rgba_raster_and_writes_leaflet_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "deformation_color.tif"
            rgba = np.zeros((4, 3, 4), dtype=np.uint8)
            rgba[0] = 220
            rgba[1] = 30
            rgba[2] = 10
            rgba[3] = 255
            with rasterio.open(
                source_path,
                "w",
                driver="GTiff",
                width=4,
                height=3,
                count=4,
                dtype="uint8",
                crs="EPSG:32610",
                transform=from_origin(500000, 4100000, 100, 100),
            ) as source:
                source.write(rgba)

            image_path, config_path = prepare_overlay(source_path, root / "output")

            with rasterio.open(image_path) as image:
                self.assertEqual(image.count, 4)
                self.assertEqual(image.colorinterp[-1], rasterio.enums.ColorInterp.alpha)
                image_rgba = image.read()
                visible = image_rgba[3] == 255
                self.assertTrue(visible.any())
                self.assertTrue(np.all(image_rgba[0][visible] == 220))
                self.assertTrue(np.all(image_rgba[1][visible] == 30))
                self.assertTrue(np.all(image_rgba[2][visible] == 10))

            config_text = config_path.read_text(encoding="utf-8")
            config = json.loads(config_text.split("=", 1)[1].strip().rstrip(";"))
            self.assertEqual(config["imageUrl"], "../trans/deformation_overlay.png")
            self.assertLess(config["bounds"][0][0], config["bounds"][1][0])
            self.assertLess(config["bounds"][0][1], config["bounds"][1][1])

    def test_rejects_rasters_without_rgba_bands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raster_path = Path(directory) / "single_band.tif"
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                width=2,
                height=2,
                count=1,
                dtype="uint8",
                crs="EPSG:4326",
                transform=from_origin(0, 2, 1, 1),
            ) as raster:
                raster.write(np.ones((1, 2, 2), dtype=np.uint8))

            with self.assertRaisesRegex(ValueError, "four-band RGBA"):
                prepare_overlay(raster_path, Path(directory) / "output")

    def test_creates_velocity_overlay_per_track(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raster_path = (
                root
                / "NISAR_L2_PR_GUNW_032_136_D_087_033_2000_SH_"
                "secular_tectonic_velocity_cm_yr.tif"
            )
            velocity = np.array([[-2.0, -1.0], [1.0, 2.0]], dtype=np.float32)
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                width=2,
                height=2,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=from_origin(38.60, 9.15, 0.1, 0.1),
                nodata=-9999.0,
            ) as source:
                source.write(velocity, 1)

            image_paths, config_path = prepare_timeseries_overlay(
                [raster_path], root / "overlays", max_dimension=64
            )

            self.assertEqual(len(image_paths), 1)
            config = json.loads(
                config_path.read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";")
            )
            self.assertEqual(
                list(config["layers"]), [
                    "TIME SERIES - DESCENDING - "
                    "NISAR_L2_PR_GUNW_032_136_D_087_033_2000_SH"
                ]
            )
            with rasterio.open(image_paths[0]) as image:
                rgba = image.read()
                self.assertEqual(image.count, 4)
                self.assertTrue(np.all(rgba[3] == 255))
                self.assertTrue(np.any(rgba[0] != rgba[2]))

    def test_combines_pair_and_time_series_options_in_map_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair_path = (
                root
                / "NISAR_L2_PR_GUNW_032_136_D_087_033_2000_SH_"
                "20261009T153517_20261009T153551_"
                "20261021T153517_20261021T153551_P05023_N_F_J_001"
                "_los_displacement_color.tif"
            )
            velocity_path = (
                root
                / "NISAR_L2_PR_GUNW_032_136_D_087_033_2000_SH"
                "_secular_tectonic_velocity_cm_yr.tif"
            )
            rgba = np.zeros((4, 4, 4), dtype=np.uint8)
            rgba[0] = 220
            rgba[3] = 255
            common = {
                "driver": "GTiff",
                "width": 4,
                "height": 4,
                "crs": "EPSG:4326",
                "transform": from_origin(38.60, 9.15, 0.1, 0.1),
            }
            with rasterio.open(
                pair_path,
                "w",
                count=4,
                dtype="uint8",
                **common,
            ) as pair:
                pair.write(rgba)
            with rasterio.open(
                velocity_path,
                "w",
                count=1,
                dtype="float32",
                nodata=-9999.0,
                **common,
            ) as velocity:
                velocity.write(np.arange(16, dtype=np.float32).reshape(4, 4), 1)

            _, config_path = prepare_overlay_mosaic(
                [pair_path],
                root / "overlays",
                max_dimension=64,
                timeseries_raster_paths=[velocity_path],
            )
            config = json.loads(
                config_path.read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";")
            )
            self.assertEqual(
                set(config["layers"]),
                {
                    "ADDIS ABABA",
                    "TIME SERIES - DESCENDING - "
                    "NISAR_L2_PR_GUNW_032_136_D_087_033_2000_SH",
                },
            )
            self.assertEqual(
                config["layers"]["ADDIS ABABA"]["kind"], "pairwise"
            )
            self.assertEqual(
                config["layers"][
                    "TIME SERIES - DESCENDING - "
                    "NISAR_L2_PR_GUNW_032_136_D_087_033_2000_SH"
                ]["kind"],
                "timeseries",
            )

    def test_map_overlay_only_includes_newest_scene_covering_addis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "rasters"
            source_dir.mkdir()
            output_dir = root / "output"
            raster_names = (
                "NISAR_L2_PR_GUNW_032_136_D_087_033_2000_SH_20261009T153517_20261009T153551_20261021T153517_20261021T153551_P05023_N_F_J_001_los_displacement_color.tif",
                "NISAR_L2_PR_GUNW_032_129_A_009_033_2000_SH_20261009T031148_20261009T031222_20261021T031148_20261021T031223_P05023_N_F_J_001_los_displacement_color.tif",
                "NISAR_L2_PR_GUNW_031_129_A_009_033_2000_SH_20260901T031148_20260901T031222_20260913T031148_20260913T031223_P05023_N_F_J_001_los_displacement_color.tif",
            )
            for name in raster_names:
                rgba = np.zeros((4, 50, 70), dtype=np.uint8)
                rgba[0] = 220
                rgba[3] = 255
                with rasterio.open(
                    source_dir / name,
                    "w",
                    driver="GTiff",
                    width=70,
                    height=50,
                    count=4,
                    dtype="uint8",
                    crs="EPSG:4326",
                    transform=from_origin(38.4, 9.2, 0.01, 0.01),
                ) as source:
                    source.write(rgba)

            image_paths, config_path = prepare_overlay_mosaic(
                list(source_dir.glob("*.tif")), output_dir, max_dimension=64
            )

            self.assertEqual(
                [path.name for path in image_paths],
                ["deformation_overlay_addis_descending.png"],
            )
            config = json.loads(
                config_path.read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";")
            )
            self.assertEqual(set(config["layers"]), {"ADDIS ABABA"})
            for image_path in image_paths:
                with rasterio.open(image_path) as image:
                    self.assertEqual(image.count, 4)
                    self.assertGreater(int(image.read(4).sum()), 0)

    def test_creates_a_clipped_addis_city_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "rasters"
            source_dir.mkdir()
            name = (
                "NISAR_L2_PR_GUNW_032_093_D_084_033_2000_SH_"
                "20261006T155835_20261006T155909_20261018T155835_"
                "20261018T155909_P05023_N_F_J_001_los_displacement_color.tif"
            )
            rgba = np.zeros((4, 50, 70), dtype=np.uint8)
            rgba[0] = 220
            rgba[3] = 255
            with rasterio.open(
                source_dir / name,
                "w",
                driver="GTiff",
                width=70,
                height=50,
                count=4,
                dtype="uint8",
                crs="EPSG:4326",
                transform=from_origin(38.4, 9.2, 0.01, 0.01),
            ) as source:
                source.write(rgba)

            image_paths, config_path = prepare_overlay_mosaic(
                [source_dir / name], root / "output"
            )

            city_images = [path for path in image_paths if "addis" in path.name]
            self.assertEqual(len(city_images), 1)
            with rasterio.open(city_images[0]) as image:
                self.assertGreater(image.width, 500)
                self.assertGreater(image.height, 400)
                self.assertEqual(image.count, 4)
            config = json.loads(
                config_path.read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";")
            )
            self.assertIn("ADDIS ABABA", config["layers"])

    def test_full_swath_creates_regional_and_addis_layers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "rasters"
            source_dir.mkdir()
            name = (
                "NISAR_L2_PR_GUNW_032_093_D_084_033_2000_SH_"
                "20261006T155835_20261006T155909_20261018T155835_"
                "20261018T155909_P05023_N_F_J_001_los_displacement_color.tif"
            )
            rgba = np.zeros((4, 50, 70), dtype=np.uint8)
            rgba[0] = 220
            rgba[3] = 255
            with rasterio.open(
                source_dir / name,
                "w",
                driver="GTiff",
                width=70,
                height=50,
                count=4,
                dtype="uint8",
                crs="EPSG:4326",
                transform=from_origin(38.4, 9.2, 0.01, 0.01),
            ) as source:
                source.write(rgba)

            image_paths, config_path = prepare_overlay_mosaic(
                [source_dir / name], root / "output", max_dimension=64, full_swath=True
            )
            config = json.loads(
                config_path.read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";")
            )
            self.assertIn("ADDIS ABABA", config["layers"])
            self.assertIn("REGIONAL - DESCENDING", config["layers"])
            self.assertEqual(len(image_paths), 2)

    def test_full_swath_mosaics_latest_pass_of_every_track(self) -> None:
        def make(directory: Path, track: str, frame: str, day: str, west: float) -> Path:
            name = (
                f"NISAR_L2_PR_GUNW_032_{track}_A_{frame}_033_2000_SH_"
                f"{day}T030613_{day}T030648_20261101T030614_"
                "20261101T030648_P05023_N_F_J_001_los_displacement_color.tif"
            )
            rgba = np.zeros((4, 40, 40), dtype=np.uint8)
            rgba[0] = 220
            rgba[3] = 255
            with rasterio.open(
                directory / name, "w", driver="GTiff", width=40, height=40, count=4,
                dtype="uint8", crs="EPSG:4326", transform=from_origin(west, -1.0, 0.01, 0.01),
            ) as source:
                source.write(rgba)
            return directory / name

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src = root / "rasters"
            src.mkdir()
            # Same frame number on two tracks, acquired on different days, far apart in longitude.
            paths = [
                make(src, "128", "175", "20261009", 36.0),
                make(src, "143", "175", "20261001", 40.0),
            ]
            _, config_path = prepare_overlay_mosaic(
                paths, root / "output", max_dimension=200, full_swath=True
            )
            config = json.loads(
                config_path.read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";")
            )
            (south, west), (north, east) = config["layers"]["REGIONAL - ASCENDING"]["bounds"]
            self.assertAlmostEqual(west, 36.0, places=1)
            self.assertAlmostEqual(east, 40.4, places=1)

    def test_full_swath_fills_missing_frame_from_older_pass(self) -> None:
        def make(directory: Path, frame: str, day: str, west: float) -> Path:
            name = (
                f"NISAR_L2_PR_GUNW_032_128_A_{frame}_033_2000_SH_"
                f"{day}T030613_{day}T030648_20261101T030614_"
                "20261101T030648_P05023_N_F_J_001_los_displacement_color.tif"
            )
            rgba = np.zeros((4, 40, 40), dtype=np.uint8)
            rgba[0] = 220
            rgba[3] = 255
            with rasterio.open(
                directory / name, "w", driver="GTiff", width=40, height=40, count=4,
                dtype="uint8", crs="EPSG:4326", transform=from_origin(west, -1.0, 0.01, 0.01),
            ) as source:
                source.write(rgba)
            return directory / name

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src = root / "rasters"
            src.mkdir()
            # Same track: frame 175 only exists in an OLDER pass, frame 176 in the newest pass.
            paths = [
                make(src, "175", "20260920", 36.0),
                make(src, "176", "20261009", 38.0),
            ]
            _, config_path = prepare_overlay_mosaic(
                paths, root / "output", max_dimension=200, full_swath=True
            )
            config = json.loads(
                config_path.read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";")
            )
            (south, west), (north, east) = config["layers"]["REGIONAL - ASCENDING"]["bounds"]
            self.assertAlmostEqual(west, 36.0, places=1)    # the older-pass frame is included
            self.assertAlmostEqual(east, 38.4, places=1)

    def test_creates_regional_mosaic_when_outside_addis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "rasters"
            source_dir.mkdir()
            # Coordinates in Kenya (-1.0 deg lat, 36.8 deg lon)
            name = (
                "NISAR_L2_PR_GUNW_032_128_A_175_033_2000_SH_"
                "20261009T030613_20261009T030648_20261021T030614_"
                "20261021T030648_P05023_N_F_J_001_los_displacement_color.tif"
            )
            rgba = np.zeros((4, 40, 40), dtype=np.uint8)
            rgba[0] = 180
            rgba[3] = 255
            with rasterio.open(
                source_dir / name,
                "w",
                driver="GTiff",
                width=40,
                height=40,
                count=4,
                dtype="uint8",
                crs="EPSG:4326",
                transform=from_origin(36.8, -1.0, 0.01, 0.01),
            ) as source:
                source.write(rgba)

            image_paths, config_path = prepare_overlay_mosaic(
                [source_dir / name], root / "output", max_dimension=64
            )
            config = json.loads(
                config_path.read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";")
            )
            self.assertNotIn("ADDIS ABABA", config["layers"])
            self.assertIn("REGIONAL - ASCENDING", config["layers"])
            self.assertEqual(len(image_paths), 1)


    def test_full_swath_respects_a_small_max_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src = root / "rasters"
            src.mkdir()
            name = (
                "NISAR_L2_PR_GUNW_032_128_A_175_033_2000_SH_"
                "20261009T030613_20261009T030648_20261021T030614_"
                "20261021T030648_P05023_N_F_J_001_los_displacement_color.tif"
            )
            rgba = np.zeros((4, 40, 40), dtype=np.uint8)
            rgba[0] = 180
            rgba[3] = 255
            with rasterio.open(
                src / name, "w", driver="GTiff", width=40, height=40, count=4,
                dtype="uint8", crs="EPSG:4326", transform=from_origin(36.8, -1.0, 0.01, 0.01),
            ) as source:
                source.write(rgba)

            image_paths, _ = prepare_overlay_mosaic(
                [src / name], root / "output", max_dimension=32, full_swath=True
            )
            with rasterio.open(image_paths[0]) as image:
                self.assertLessEqual(max(image.width, image.height), 32)

    def test_mosaic_uses_numeric_rasters_for_physical_values(self) -> None:
        # Two frames side by side. The colour rasters are identical and uninformative;
        # only the numeric rasters differ, so the output can only reflect them if the
        # mosaic reads the centimetre values (frame B has the larger displacement).
        def make(directory: Path, frame: str, west: float, cm: float) -> Path:
            stem = (
                f"NISAR_L2_PR_GUNW_032_128_A_{frame}_033_2000_SH_"
                "20261009T030613_20261009T030648_20261021T030614_"
                "20261021T030648_P05023_N_F_J_001"
            )
            common = dict(
                driver="GTiff", width=40, height=40, crs="EPSG:4326",
                transform=from_origin(west, -1.0, 0.01, 0.01),
            )
            color = np.zeros((4, 40, 40), dtype=np.uint8)
            color[0] = 255
            color[1] = 128
            color[2] = 128
            color[3] = 255
            with rasterio.open(
                directory / f"{stem}_los_displacement_color.tif", "w",
                count=4, dtype="uint8", **common,
            ) as out:
                out.write(color)
            with rasterio.open(
                directory / f"{stem}_los_displacement_cm.tif", "w",
                count=1, dtype="float32", nodata=-9999.0, **common,
            ) as out:
                out.write(np.full((40, 40), cm, dtype=np.float32), 1)
            return directory / f"{stem}_los_displacement_color.tif"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src = root / "rasters"
            src.mkdir()
            paths = [make(src, "175", 36.8, 0.5), make(src, "176", 37.4, 4.0)]
            image_paths, _ = prepare_overlay_mosaic(
                paths, root / "output", max_dimension=120, full_swath=True
            )
            with rasterio.open(image_paths[0]) as image:
                rgba = image.read()
            left = rgba[:, rgba.shape[1] // 2, 5]
            right = rgba[:, rgba.shape[1] // 2, rgba.shape[2] - 6]
            self.assertEqual(int(left[3]), 255)
            self.assertEqual(int(right[3]), 255)
            # Colour is more saturated red where the displacement is larger.
            self.assertLess(int(right[1]), int(left[1]))


if __name__ == "__main__":
    unittest.main()
