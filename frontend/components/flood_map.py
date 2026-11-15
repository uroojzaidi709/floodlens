import folium
from streamlit_folium import st_folium


def display_flood_map(layers, names, key="flood_map", height=520):
    """Dark base map + the pipeline's real PNG overlays (all share the same bounds)."""
    b = layers["bounds_leaflet"]
    center = [(b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2]
    m = folium.Map(location=center, zoom_start=8, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, HERE, Garmin, OpenStreetMap contributors", name="Dark Map", show=True).add_to(m)
    folium.TileLayer(
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, HERE, Garmin, OpenStreetMap contributors", name="Street Map", show=False).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite View", show=False).add_to(m)
    for n in names:
        if n in layers["layers"]:
            folium.raster_layers.ImageOverlay(
                image=layers["layers"][n]["path"], bounds=b, opacity=0.85,
                name=layers["layers"][n]["label"], show=n.startswith("flood_threshold")).add_to(m)
    folium.LayerControl(position="topright").add_to(m)
    m.fit_bounds(b)
    st_folium(m, use_container_width=True, height=height, returned_objects=[], key=key)
