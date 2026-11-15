"""Tests for lightweight Folium local web viewer."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from viewer import build_folium_map, find_free_port


class ViewerTests(unittest.TestCase):
    def test_find_free_port(self) -> None:
        port = find_free_port(start=8800, end=8900)
        self.assertGreaterEqual(port, 8800)
        self.assertLessEqual(port, 8900)

    def test_build_folium_map_creates_valid_leaflet_html(self) -> None:
        map_obj = build_folium_map(center=(9.0, 38.75), zoom=6, opacity=0.8)
        rendered_html = map_obj.get_root().render()

        # Verify Folium Leaflet elements and base layers
        self.assertIn("L.map", rendered_html)
        self.assertIn("CartoDB Dark (Night Mode)", rendered_html)
        self.assertIn("OpenStreetMap", rendered_html)


if __name__ == "__main__":
    unittest.main()
