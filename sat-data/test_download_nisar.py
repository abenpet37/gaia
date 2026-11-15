import argparse
import contextlib
import io
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from download_nisar import (
    DEFAULT_OUTPUT_DIR,
    HDF5_SIGNATURE,
    bbox_to_wkt,
    create_parser,
    download_product,
    ensure_output_maturity,
    _earthdata_token,
    filter_orbit_direction,
    is_downloaded,
    latest_acquisition_pair,
    latest_product,
    parse_bbox,
    process_and_discard_source,
    _progress_summary,
    unique_products,
)


class _InlineExecutor:
    """Minimal ProcessPoolExecutor stand-in that runs jobs synchronously in-process."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        if initializer:
            initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args):
        from concurrent.futures import Future

        future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as exc:  # noqa: BLE001
            future.set_exception(exc)
        return future


class DownloadNisarTests(unittest.TestCase):
    def test_default_download_directory_is_manipulate_data(self) -> None:
        self.assertEqual(DEFAULT_OUTPUT_DIR, Path(__file__).resolve().parents[1] / "manipulate" / "data")
        args = create_parser().parse_args(
            ["--bbox", "-124,32,-114,42", "--start", "2026-10-21", "--end", "2026-11-14"]
        )
        self.assertEqual(args.output, DEFAULT_OUTPUT_DIR)
        self.assertTrue(args.clip_faults)

        no_clip_args = create_parser().parse_args(
            ["--bbox", "-124,32,-114,42", "--start", "2026-10-21", "--end", "2026-11-14", "--no-clip-faults"]
        )
        self.assertFalse(no_clip_args.clip_faults)

    def test_bbox_is_validated_and_converted_to_wkt(self) -> None:
        bbox = parse_bbox("-124,32,-114,42")
        self.assertEqual(
            bbox_to_wkt(bbox),
            "POLYGON((-124.0 32.0,-114.0 32.0,-114.0 42.0,-124.0 42.0,-124.0 32.0))",
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_bbox("-114,32,-124,42")

    def test_duplicate_catalog_records_are_returned_once(self) -> None:
        product = {
            "fileName": "scene.h5",
            "downloadUrl": "https://nisar.asf.earthdatacloud.nasa.gov/scene.h5",
            "groupID": "unique-product",
        }
        products = unique_products(iter([product, product]))
        self.assertEqual(products, [product])

    def test_latest_product_uses_the_most_recent_acquisition_end(self) -> None:
        older = {"fileName": "older.h5", "stopTime": "2026-10-10T00:00:00Z"}
        newer = {"fileName": "newer.h5", "stopTime": "2026-10-21T00:00:00Z"}
        self.assertIs(latest_product([older, newer]), newer)

    def test_latest_product_requires_acquisition_timestamp(self) -> None:
        with self.assertRaisesRegex(ValueError, "acquisition timestamp"):
            latest_product([{"fileName": "untimed.h5"}])

    def test_latest_acquisition_pair_keeps_all_granules_from_latest_dates(self) -> None:
        old = {
            "fileName": "old.h5",
            "startTime": "2026-09-01T10:00:00Z",
            "stopTime": "2026-09-13T10:00:00Z",
        }
        first = {
            "fileName": "first.h5",
            "startTime": "2026-10-09T15:00:00Z",
            "stopTime": "2026-10-21T15:00:00Z",
        }
        second = {
            "fileName": "second.h5",
            "startTime": "2026-10-09T03:00:00Z",
            "stopTime": "2026-10-21T03:00:00Z",
        }
        self.assertEqual(latest_acquisition_pair([old, first, second]), [first, second])

    def test_orbit_direction_filters_product_list(self) -> None:
        ascending = {"fileName": "a.h5", "flightDirection": "ASCENDING"}
        descending = {"fileName": "d.h5", "flightDirection": "DESCENDING"}
        self.assertEqual(
            filter_orbit_direction([ascending, descending], "DESCENDING"),
            [descending],
        )

    def test_earthdata_token_prompts_without_echo_when_not_in_environment(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("download_nisar.EARTHDATA_TOKEN_FILE") as token_file,
            patch(
                "download_nisar.getpass.getpass", return_value="test-token"
            ) as prompt,
        ):
            token_file.is_file.return_value = False
            self.assertEqual(_earthdata_token(), "test-token")
        prompt.assert_called_once_with("Earthdata bearer token (input hidden): ")

    def test_earthdata_token_rejects_empty_prompt(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("download_nisar.EARTHDATA_TOKEN_FILE") as token_file,
            patch("download_nisar.getpass.getpass", return_value="  "),
            self.assertRaisesRegex(ValueError, "token is required"),
        ):
            token_file.is_file.return_value = False
            _earthdata_token()

    def test_earthdata_token_reads_local_file_before_prompt(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch(
            "download_nisar.EARTHDATA_TOKEN_FILE"
        ) as token_file, patch("download_nisar.getpass.getpass") as prompt:
            token_file.is_file.return_value = True
            token_file.read_text.return_value = "  file-token \n"
            self.assertEqual(_earthdata_token(), "file-token")
        prompt.assert_not_called()

    def test_existing_nonempty_product_is_not_downloaded_again(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "scene.h5").write_bytes(b"existing data")
            self.assertTrue(is_downloaded(output_dir, "scene.h5"))
            (output_dir / "empty.h5").write_bytes(b"")
            self.assertFalse(is_downloaded(output_dir, "empty.h5"))

    def test_download_shows_percentage_when_content_length_is_available(self) -> None:
        with TemporaryDirectory() as directory:
            response = MagicMock()
            response.status_code = 200
            response.headers = {"Content-Length": str(len(HDF5_SIGNATURE))}
            response.iter_content.return_value = [HDF5_SIGNATURE]
            response.__enter__.return_value = response
            session = MagicMock()
            session.get.return_value = response
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                downloaded = download_product(
                    {
                        "fileName": "scene.h5",
                        "downloadUrl": "https://nisar.asf.earthdatacloud.nasa.gov/scene.h5",
                    },
                    Path(directory),
                    session,
                )

            self.assertTrue(downloaded)
            self.assertIn("100%", output.getvalue())

    def test_process_and_discard_removes_source_only_after_color_raster_exists(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scene.h5"
            source.write_bytes(b"product")
            raster_dir = root / "rasters"
            color = raster_dir / "scene_los_displacement_color.tif"
            numeric = raster_dir / "scene_los_displacement_cm.tif"

            def process(*args: object, **kwargs: object) -> MagicMock:
                raster_dir.mkdir()
                color.write_bytes(b"color raster")
                numeric.write_bytes(b"numeric raster")
                return MagicMock(returncode=0)

            with (
                patch("download_nisar.DEFAULT_RASTER_OUTPUT_DIR", raster_dir),
                patch("download_nisar.subprocess.run", side_effect=process),
            ):
                result = process_and_discard_source(source, keep_numeric=True)

            self.assertEqual(result, color)
            self.assertFalse(source.exists())
            self.assertTrue(numeric.exists())
            self.assertTrue(color.is_file())

    def test_process_and_discard_can_discard_numeric_when_requested(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scene.h5"
            source.write_bytes(b"product")
            raster_dir = root / "rasters"
            color = raster_dir / "scene_los_displacement_color.tif"
            numeric = raster_dir / "scene_los_displacement_cm.tif"

            def process(*args: object, **kwargs: object) -> MagicMock:
                raster_dir.mkdir()
                color.write_bytes(b"color raster")
                numeric.write_bytes(b"numeric raster")
                return MagicMock(returncode=0)

            with (
                patch("download_nisar.DEFAULT_RASTER_OUTPUT_DIR", raster_dir),
                patch("download_nisar.subprocess.run", side_effect=process),
            ):
                result = process_and_discard_source(source, keep_numeric=False)

            self.assertEqual(result, color)
            self.assertFalse(source.exists())
            self.assertFalse(numeric.exists())
            self.assertTrue(color.is_file())

    def test_process_failure_keeps_source_file(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "scene.h5"
            source.write_bytes(b"product")
            with patch(
                "download_nisar.subprocess.run",
                return_value=MagicMock(returncode=1),
            ), self.assertRaisesRegex(RuntimeError, "source file was kept"):
                process_and_discard_source(source)
            self.assertTrue(source.is_file())

    def test_progress_summary_reports_completed_and_remaining_file_counts(self) -> None:
        self.assertEqual(
            _progress_summary(7, 29, 5, 2),
            "Files: 7/29 complete, 22 remaining (5 downloaded this run, 2 already present).",
        )

    def test_output_directory_cannot_mix_maturity_levels(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            ensure_output_maturity(output_dir, "VALIDATED")
            with self.assertRaisesRegex(ValueError, "avoid mixing releases"):
                ensure_output_maturity(output_dir, "PROVISIONAL")


    def test_unsafe_product_file_names_are_rejected(self) -> None:
        from download_nisar import safe_product_filename

        good = "NISAR_L2_PR_GUNW_032_128_A_175_033_2000_SH_20261009T030613.h5"
        self.assertEqual(safe_product_filename({"fileName": good}), good)
        for bad in (
            "C:\\Users\\x\\target.h5",
            "..\\..\\evil.h5",
            "../evil.h5",
            "sub/dir.h5",
            "/abs.h5",
            ".hidden",
            "",
            None,
        ):
            with self.assertRaises(ValueError, msg=repr(bad)):
                safe_product_filename({"fileName": bad})

    def test_remote_process_refuses_a_path_traversing_file_name(self) -> None:
        from download_nisar import remote_process_product

        product = {
            "fileName": "..\\..\\evil.h5",
            "downloadUrl": "https://nisar.asf.earthdatacloud.nasa.gov/x.h5",
        }
        with self.assertRaises(ValueError):
            remote_process_product(product, MagicMock())

    def test_reference_attributes_are_not_copied_between_files(self) -> None:
        import h5py
        import numpy as np
        from download_nisar import _copy_portable_attrs

        with TemporaryDirectory() as directory:
            with h5py.File(Path(directory) / "src.h5", "w") as src, h5py.File(
                Path(directory) / "dst.h5", "w"
            ) as dst:
                scale = src.create_dataset("x", data=np.arange(3))
                data = src.create_dataset("data", data=np.zeros((2, 3)))
                data.attrs["units"] = "radians"
                data.attrs["scale_ref"] = scale.ref
                data.attrs["axis"] = np.array([1, 2])
                out = dst.create_dataset("data", data=np.zeros((2, 3)))

                _copy_portable_attrs(data, out)

                self.assertEqual(out.attrs["units"], "radians")
                self.assertEqual(list(out.attrs["axis"]), [1, 2])
                self.assertNotIn("scale_ref", out.attrs)

    def test_progress_from_workers_reaches_the_callback(self) -> None:
        import download_nisar

        events = []

        def fake_worker(product, token, crop_bbox, color_only):
            download_nisar._worker_progress_queue.put((product["fileName"], "extract", 40, "halfway"))
            return "ok"

        # An in-process executor stands in for real worker processes.
        with patch.object(download_nisar, "_process_granule_worker", fake_worker), patch(
            "concurrent.futures.ProcessPoolExecutor", _InlineExecutor
        ):
            done, errors = download_nisar.remote_process_products(
                [{"fileName": "a.h5"}], "t", progress_callback=lambda *a: events.append(a)
            )
        self.assertEqual((done, errors), (1, []))
        steps = [e[0] for e in events]
        self.assertIn("extract", steps)
        self.assertIn("granule_done", steps)
        # Worker events are relayed by a thread, so one may land after granule_done;
        # the overall figure must still never go backwards or past 90.
        percents = [e[1] for e in events]
        self.assertEqual(percents, sorted(percents))
        self.assertEqual(max(percents), 90)


if __name__ == "__main__":
    unittest.main()
