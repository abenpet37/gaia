"""Lightweight local web viewer using Python, Flask / HTTP server, and Folium (Leaflet.js).

Layers:
1. CartoDB Dark & OpenStreetMap Basemaps
2. Clipped GEM Active Fault Lines (contrasting sharp amber/black)
3. Processed NISAR Displacement GeoTIFF / Overlay (with transparency)
4. USGS Earthquake Epicenter Marker with full magnitude/depth/date popup
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import webbrowser
from pathlib import Path
from typing import Any

import folium
from folium import plugins

ROOT_DIR = Path(__file__).resolve().parent
MAP_DIR = ROOT_DIR / "map"
MAP_DATA_DIR = MAP_DIR / "data"
TRANS_DIR = ROOT_DIR / "trans"
MANIPULATE_OUTPUT_DIR = ROOT_DIR / "manipulate" / "output"

FAULTS_CLIP_PATH = MAP_DIR / "faults_clip.geojson"
FAULTS_MASTER_PATH = MAP_DIR / "active_faults.json"
TRIGGER_METADATA_PATH = MAP_DATA_DIR / "latest_trigger_event.json"
OVERLAY_CONFIG_PATH = TRANS_DIR / "deformation-overlay.js"


def find_free_port(start: int = 8000, end: int = 8100) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return 8000


def build_folium_map(
    center: tuple[float, float] = (9.0, 38.75),
    zoom: int = 6,
    opacity: float = 0.75,
) -> folium.Map:
    """Construct an interactive Folium Leaflet map with all 4 required layers."""
    # Read trigger event if present to center the map
    trigger_data: dict[str, Any] | None = None
    if TRIGGER_METADATA_PATH.is_file():
        try:
            with open(TRIGGER_METADATA_PATH, "r", encoding="utf-8") as f:
                trigger_data = json.load(f)
                if trigger_data and "epicenter" in trigger_data:
                    center = (trigger_data["epicenter"]["latitude"], trigger_data["epicenter"]["longitude"])
                    zoom = 8
        except Exception:
            pass

    # 1. Base Map: CartoDB Dark Matter with OpenStreetMap toggle
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles=None,  # Start clean to add custom named basemaps
        control_scale=True,
    )

    folium.TileLayer(
        tiles="CartoDB dark_matter",
        name="CartoDB Dark (Night Mode)",
        attr='&copy; <a href="https://carto.com/">CARTO</a>',
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        control=True,
    ).add_to(m)

    # 2. Processed NISAR Displacement Raster Overlay (with transparency)
    # Check deformation-overlay.js or direct PNG overlays in trans/
    overlay_added = False
    if OVERLAY_CONFIG_PATH.is_file():
        try:
            content = OVERLAY_CONFIG_PATH.read_text(encoding="utf-8")
            json_str = content.split("window.DEFORMATION_OVERLAY = ", 1)[-1].rstrip(";\n ")
            config = json.loads(json_str)
            layers = config.get("layers", {})

            for name, layer in layers.items():
                img_url = layer.get("imageUrl", "").replace("../", "")
                img_path = ROOT_DIR / img_url
                bounds = layer.get("bounds")
                if img_path.is_file() and bounds:
                    folium.raster_layers.ImageOverlay(
                        image=str(img_path),
                        bounds=bounds,
                        opacity=opacity,
                        name=f"InSAR: {layer.get('label', name)}",
                        interactive=False,
                        cross_origin=False,
                    ).add_to(m)
                    overlay_added = True
        except Exception as exc:
            print(f"Notice: Could not parse overlay config ({exc})")

    if not overlay_added:
        # Fallback to standard overlay if present
        regional_overlay = TRANS_DIR / "deformation_overlay_regional_ascending.png"
        if regional_overlay.is_file():
            folium.raster_layers.ImageOverlay(
                image=str(regional_overlay),
                bounds=[[-6.55, 32.12], [20.32, 40.84]],
                opacity=opacity,
                name="InSAR LOS Displacement (Ascending)",
            ).add_to(m)

    # 3. Clipped GEM Active Fault Lines (Contrasting Sharp Amber)
    fault_file = FAULTS_CLIP_PATH if (FAULTS_CLIP_PATH.is_file() and FAULTS_CLIP_PATH.stat().st_size > 100) else FAULTS_MASTER_PATH
    if fault_file.is_file():
        try:
            with open(fault_file, "r", encoding="utf-8") as f:
                faults_geojson = json.load(f)

            features = faults_geojson.get("features", [])
            if features:
                sample_props = features[0].get("properties", {}) or {}
                fields = [k for k in ["name", "type", "slip_rate", "hazard"] if k in sample_props]
                alias_map = {"name": "Fault:", "type": "Kinematics:", "slip_rate": "Slip Rate:", "hazard": "Seismic Hazard:"}
                aliases = [alias_map[k] for k in fields]
                tooltip = folium.GeoJsonTooltip(fields=fields, aliases=aliases, sticky=True) if fields else None

                folium.GeoJson(
                    faults_geojson,
                    name="Active Fault Lines (GEM/USGS)",
                    style_function=lambda x: {
                        "color": "#f59e0b",
                        "weight": 3.0,
                        "opacity": 0.95,
                    },
                    tooltip=tooltip,
                ).add_to(m)
        except Exception as exc:
            print(f"Notice: Failed loading faults GeoJSON ({exc})")

    # 4. USGS Earthquake Epicenter Marker with detailed popup
    if trigger_data and "epicenter" in trigger_data:
        epi = trigger_data["epicenter"]
        mag = trigger_data.get("magnitude", 0.0)
        place = trigger_data.get("place", "Unknown location")
        depth = trigger_data.get("depth_km", 0.0)
        time_utc = trigger_data.get("time_utc", "Unknown")
        window = trigger_data.get("temporal_window", {})
        granules_cnt = trigger_data.get("matching_granules_count", 0)

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 210px; line-height: 1.4;">
            <h4 style="margin: 0 0 4px 0; color: #dc2626; font-size: 14px;">
                ⚡ M{mag:.1f} Earthquake Trigger
            </h4>
            <div style="font-size: 12px; font-weight: bold; color: #1e293b; margin-bottom: 6px;">
                {place}
            </div>
            <table style="font-size: 11px; width: 100%; border-collapse: collapse;">
                <tr><td style="color: #64748b;">Depth:</td><td><b>{depth:.1f} km</b></td></tr>
                <tr><td style="color: #64748b;">Date (UTC):</td><td>{time_utc}</td></tr>
                <tr><td style="color: #64748b;">Co-seismic:</td><td><b>{window.get('start_date', 'N/A')} &rarr; {window.get('end_date', 'N/A')}</b></td></tr>
                <tr><td style="color: #64748b;">NISAR Pairs:</td><td><b style="color: #0284c7;">{granules_cnt} matched</b></td></tr>
            </table>
        </div>
        """

        folium.CircleMarker(
            location=[epi["latitude"], epi["longitude"]],
            radius=12,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"Trigger Event: M{mag:.1f} - {place}",
            color="#ffffff",
            weight=2.5,
            fill=True,
            fill_color="#ef4444",
            fill_opacity=0.9,
        ).add_to(m)

    # Layer Switcher & Fullscreen control
    folium.LayerControl(collapsed=False).add_to(m)
    plugins.Fullscreen(position="topleft").add_to(m)

    return m


def run_viewer(port: int = 8080, no_open: bool = False) -> None:
    """Generate and launch the local viewer server."""
    port = find_free_port(port, port + 50)
    map_obj = build_folium_map()
    viewer_html_path = MAP_DIR / "viewer.html"
    map_obj.save(str(viewer_html_path))
    print(f"Generated clean multi-layer Folium map at {viewer_html_path.name}")

    # Use Flask if available, otherwise Python's built-in http.server
    url = f"http://127.0.0.1:{port}/map/viewer.html"

    try:
        from flask import Flask, send_from_directory  # type: ignore

        app = Flask(__name__, static_folder=str(ROOT_DIR))

        @app.route("/")
        def index():
            return send_from_directory(str(MAP_DIR), "viewer.html")

        @app.route("/<path:path>")
        def serve_static(path):
            return send_from_directory(str(ROOT_DIR), path)

        print(f"Starting Flask viewer on http://127.0.0.1:{port}...")
        if not no_open:
            webbrowser.open_new_tab(f"http://127.0.0.1:{port}/")
        app.run(host="127.0.0.1", port=port, debug=False)
    except ImportError:
        # Seamless zero-dependency fallback using built-in http.server
        import http.server
        import socketserver

        class LocalHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(ROOT_DIR), **kwargs)

        print(f"Serving local Folium viewer at {url}...")
        if not no_open:
            webbrowser.open_new_tab(url)
        with socketserver.TCPServer(("127.0.0.1", port), LocalHandler) as httpd:
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nViewer stopped.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080, help="Local server port (default: 8080)")
    parser.add_argument("--no-open", action="store_true", help="Do not automatically open browser tab")
    args = parser.parse_args(argv)

    run_viewer(port=args.port, no_open=args.no_open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
