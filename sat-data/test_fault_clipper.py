"""Tests for dynamic fault clipping to earthquake / InSAR bounding boxes."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fault_clipper import (
    _line_intersects_bbox,
    _standardize_properties,
    clip_faults_native,
    clip_faults_to_bbox,
    parse_bbox_string,
)


class FaultClipperTests(unittest.TestCase):
    def test_parse_bbox_string_validates_order(self) -> None:
        bbox = parse_bbox_string("36.0, 7.0, 40.0, 10.0")
        self.assertEqual(bbox, (36.0, 7.0, 40.0, 10.0))

        with self.assertRaises(ValueError):
            parse_bbox_string("40.0, 7.0, 36.0, 10.0")  # west > east

        with self.assertRaises(ValueError):
            parse_bbox_string("36.0, 7.0, 40.0")  # incomplete

    def test_line_intersects_bbox_detection(self) -> None:
        # Inside box
        coords = [[38.5, 8.5], [38.7, 8.9]]
        self.assertTrue(_line_intersects_bbox(coords, 38.0, 8.0, 39.0, 9.0))

        # Outside box
        coords_outside = [[10.0, 20.0], [10.5, 20.5]]
        self.assertFalse(_line_intersects_bbox(coords_outside, 38.0, 8.0, 39.0, 9.0))

    def test_standardize_properties_maps_gem_fields(self) -> None:
        gem_props = {
            "fault_name": "San Andreas Fault",
            "slip_type": "Right-lateral strike-slip",
            "slip_rate_avg": 25.0,
            "activity_confidence": "High",
        }
        res = _standardize_properties(gem_props)
        self.assertEqual(res["name"], "San Andreas Fault")
        self.assertEqual(res["type"], "Right-lateral strike-slip")
        self.assertEqual(res["slip_rate"], "25.0 mm/yr")
        self.assertEqual(res["hazard"], "High")

    def test_clipping_native_extracts_matching_features(self) -> None:
        sample_fc = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"name": "Target Fault", "type": "Normal", "slip_rate": "2.0 mm/yr"},
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[38.7, 8.9], [38.8, 9.0]],
                    },
                },
                {
                    "type": "Feature",
                    "properties": {"name": "Faraway Fault", "type": "Thrust", "slip_rate": "5.0 mm/yr"},
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[-120.0, 35.0], [-121.0, 36.0]],
                    },
                },
            ],
        }

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            master_file = temp_path / "master.geojson"
            master_file.write_text(json.dumps(sample_fc), encoding="utf-8")
            output_file = temp_path / "faults_clip.geojson"

            # Query Addis Ababa area (38.0, 8.0 to 39.0, 9.5)
            count = clip_faults_native(master_file, (38.0, 8.0, 39.0, 9.5), output_file)

            self.assertEqual(count, 1)
            self.assertTrue(output_file.is_file())

            clipped_data = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertEqual(len(clipped_data["features"]), 1)
            self.assertEqual(clipped_data["features"][0]["properties"]["name"], "Target Fault")


if __name__ == "__main__":
    unittest.main()
