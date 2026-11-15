import folium
import requests
from branca.element import Element
from pathlib import Path


def add_view_state_script(map_object):
    map_object.get_root().html.add_child(
        Element('<script src="map-state.js" defer></script>')
    )


# Fetch tectonic plate boundaries.
url = "https://raw.githubusercontent.com/fraxen/tectonicplates/master/GeoJSON/PB2002_boundaries.json"

print("Fetching tectonic plate data...")
response = requests.get(url, timeout=30)
response.raise_for_status()
plates_data = response.json()

# Create a base map centered on the Pacific Ocean
m = folium.Map(
    location=[20, 0],
    zoom_start=2,
    no_wrap=True, max_bounds=True, min_zoom=2,
    tiles='https://{s}.basemaps.cartocdn.com/rasterTiles/voyager/{z}/{x}/{y}.png', attr='&copy; OpenStreetMap contributors &copy; CARTO', subdomains='abcd'
)

# Add tectonic plate boundaries
style_function = lambda x: {
    'fillColor': 'transparent',
    'color': '#ff4444',
    'weight': 2,
    'opacity': 0.8
}

folium.GeoJson(
    plates_data,
    name='Tectonic Plates',
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(
        fields=['Name'],
        aliases=['Plate:'],
        sticky=True
    )
).add_to(m)

# Add layer control
folium.LayerControl().add_to(m)
add_view_state_script(m)

# Save to HTML
output_path = Path(__file__).resolve().parent / "tectonic_plates_map.html"
m.save(output_path)
print(f"Map saved to {output_path}")

# Also create a version with satellite imagery
m2 = folium.Map(
    location=[20, 0],
    zoom_start=2,
    no_wrap=True, max_bounds=True, min_zoom=2,
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri World Imagery'
)

folium.GeoJson(
    plates_data,
    name='Tectonic Plates',
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(
        fields=['Name'],
        aliases=['Plate:'],
        sticky=True
    )
).add_to(m2)

folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
    attr='Esri, HERE, Garmin, (c) OpenStreetMap contributors, and the GIS user community',
    name='Boundaries & Place Names',
    overlay=True,
    control=True,
    show=True,
    no_wrap=True
).add_to(m2)

folium.LayerControl().add_to(m2)
add_view_state_script(m2)
m2.save(Path(__file__).resolve().parent / "tectonic_plates_satellite.html")
print("Satellite map saved to tectonic_plates_satellite.html")