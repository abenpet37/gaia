import json
from pathlib import Path
import folium
import requests
from branca.element import Element

output_dir = Path(__file__).resolve().parent


def add_view_state_script(map_object):
    map_object.get_root().html.add_child(
        Element('<script src="map-state.js" defer></script>')
    )


# Load tectonic plate boundaries (local cache first, fallback to network)
plates_file = output_dir / "tectonic_plates.json"
if plates_file.is_file():
    print("Loading cached tectonic plate data...")
    plates_data = json.loads(plates_file.read_text(encoding="utf-8"))
else:
    url = "https://raw.githubusercontent.com/fraxen/tectonicplates/master/GeoJSON/PB2002_boundaries.json"
    print("Fetching tectonic plate data from web...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    plates_data = response.json()
    plates_file.write_text(json.dumps(plates_data), encoding="utf-8")

# Load active tectonic fault traces
faults_file = output_dir / "active_faults.json"
faults_data = None
if faults_file.is_file():
    print("Loading active faults data...")
    faults_data = json.loads(faults_file.read_text(encoding="utf-8"))

plate_style_function = lambda x: {
    'fillColor': 'transparent',
    'color': '#ff3333',
    'weight': 2.5,
    'opacity': 0.85
}

fault_style_function = lambda x: {
    'fillColor': 'transparent',
    'color': '#f59e0b',
    'weight': 3.0,
    'opacity': 0.95
}


def add_tectonic_layers(map_object):
    folium.GeoJson(
        plates_data,
        name='Tectonic Plates',
        style_function=plate_style_function,
        tooltip=folium.GeoJsonTooltip(fields=['Name'], aliases=['Plate:'], sticky=True)
    ).add_to(map_object)

    if faults_data:
        folium.GeoJson(
            faults_data,
            name='Active Fault Traces',
            style_function=fault_style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=['name', 'type', 'slip_rate'],
                aliases=['Fault:', 'Type:', 'Slip Rate:'],
                sticky=True
            )
        ).add_to(map_object)


# Version 1: OpenStreetMap (standard map with roads, cities, labels)
m1 = folium.Map(location=[10, 38], zoom_start=4, tiles='https://{s}.basemaps.cartocdn.com/rasterTiles/voyager/{z}/{x}/{y}.png', attr='&copy; OpenStreetMap contributors &copy; CARTO', subdomains='abcd', no_wrap=True, max_bounds=True, min_zoom=2)
add_tectonic_layers(m1)
folium.LayerControl().add_to(m1)
add_view_state_script(m1)
m1.save(output_dir / "tectonic_osm.html")
print("Saved: tectonic_osm.html (OpenStreetMap)")

# Version 2: Esri World Topo (terrain, elevation, cities, labels)
m2 = folium.Map(
    location=[10, 38], zoom_start=4,
    no_wrap=True, max_bounds=True, min_zoom=2,
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
    attr='Esri World Topographic Map'
)
add_tectonic_layers(m2)
folium.LayerControl().add_to(m2)
add_view_state_script(m2)
m2.save(output_dir / "tectonic_topo.html")
print("Saved: tectonic_topo.html (Esri World Topo)")

# Version 3: Esri World Imagery (satellite with labels overlay)
m3 = folium.Map(
    location=[10, 38], zoom_start=4,
    no_wrap=True, max_bounds=True, min_zoom=2,
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri World Imagery'
)
add_tectonic_layers(m3)
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
    attr='Esri, HERE, Garmin, (c) OpenStreetMap contributors, and the GIS user community',
    name='Boundaries & Place Names',
    overlay=True,
    control=True,
    show=True,
    no_wrap=True
).add_to(m3)
folium.LayerControl().add_to(m3)
add_view_state_script(m3)
m3.save(output_dir / "tectonic_satellite.html")
print("Saved: tectonic_satellite.html (Esri Satellite with boundaries and place names)")

# Version 4: Esri World Street Map (detailed map with labels)
m4 = folium.Map(
    location=[10, 38], zoom_start=4,
    no_wrap=True, max_bounds=True, min_zoom=2,
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
    attr='Esri World Street Map'
)
add_tectonic_layers(m4)
folium.LayerControl().add_to(m4)
add_view_state_script(m4)
m4.save(output_dir / "tectonic_voyager.html")
print("Saved: tectonic_voyager.html (Esri World Street Map)")

# Version 5: OpenTopoMap (terrain, elevation, and contour lines)
m5 = folium.Map(
    location=[10, 38], zoom_start=4,
    no_wrap=True, max_bounds=True, min_zoom=2,
    tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
    attr='Map data: &copy; OpenStreetMap contributors, SRTM | Map style: &copy; OpenTopoMap (CC-BY-SA)'
)
add_tectonic_layers(m5)
folium.LayerControl().add_to(m5)
add_view_state_script(m5)
m5.save(output_dir / "tectonic_terrain.html")
print("Saved: tectonic_terrain.html (OpenTopoMap)")
