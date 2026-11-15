"""Tests for automated USGS trigger-to-search glue script."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from trigger_search import (
    calculate_spatial_bbox,
    calculate_temporal_window,
    create_parser,
    fetch_usgs_events,
    process_event_trigger,
    save_trigger_metadata,
)


class TriggerSearchTests(unittest.TestCase):
    def test_calculate_spatial_bbox_standard(self) -> None:
        bbox = calculate_spatial_bbox(lon=38.75, lat=9.0, radius_deg=1.0)
        self.assertEqual(bbox, (37.75, 8.0, 39.75, 10.0))

    def test_calculate_spatial_bbox_clamps_at_poles_and_antimeridian(self) -> None:
        bbox = calculate_spatial_bbox(lon=179.5, lat=-89.5, radius_deg=1.0)
        self.assertEqual(bbox, (178.5, -90.0, 180.0, -88.5))

    def test_calculate_temporal_window_12_days(self) -> None:
        # 2026-10-15 12:00:00 UTC = 1792065600000 ms
        timestamp_ms = 1792065600000
        start, end = calculate_temporal_window(timestamp_ms, window_days=12)
        self.assertEqual(start, "2026-10-03")
        self.assertEqual(end, "2026-10-27")

    def test_save_trigger_metadata_writes_json(self) -> None:
        event = {
            "id": "us7000test",
            "properties": {"mag": 5.4, "place": "20 km S of Hawassa, Ethiopia", "time": 1792065600000},
            "geometry": {"coordinates": [38.5, 7.0, 10.0]},
        }
        bbox = (37.5, 6.0, 39.5, 8.0)
        start, end = "2026-10-03", "2026-10-27"
        products = [{"fileName": "NISAR_test.h5", "flightDirection": "ASCENDING"}]

        with TemporaryDirectory() as tmp_dir:
            meta_path = Path(tmp_dir) / "latest_trigger_event.json"
            save_trigger_metadata(event, bbox, start, end, products, output_path=meta_path)

            self.assertTrue(meta_path.is_file())
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(data["event_id"], "us7000test")
            self.assertEqual(data["magnitude"], 5.4)
            self.assertEqual(data["matching_granules_count"], 1)
            self.assertEqual(data["epicenter"]["longitude"], 38.5)

    @patch("trigger_search.urllib.request.urlopen")
    def test_fetch_usgs_events_filters_magnitude(self, mock_urlopen) -> None:
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "features": [
                {"properties": {"mag": 4.2, "time": 100}},
                {"properties": {"mag": 5.1, "time": 200}},
                {"properties": {"mag": 6.0, "time": 300}},
            ]
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        events = fetch_usgs_events(min_mag=4.5)
        self.assertEqual(len(events), 2)
        # Verify sorted newest first
        self.assertEqual(events[0]["properties"]["mag"], 6.0)
        self.assertEqual(events[1]["properties"]["mag"], 5.1)

    def test_parser_options(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["--min-mag", "5.0", "--radius-deg", "1.5"])
        self.assertEqual(args.min_mag, 5.0)
        self.assertEqual(args.radius_deg, 1.5)
        self.assertTrue(args.dry_run)


if __name__ == "__main__":
    unittest.main()
