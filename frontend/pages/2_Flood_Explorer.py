import json

import streamlit as st

from common import (setup, demo_banner, na, get, fmt, load_img, load_layers, funnel_rows, footer)
from components.metric_cards import display_metric_card
from components.image_comparison import display_image_comparison
from components.flood_map import display_flood_map
from components.map_legend import display_map_legend
from components.flood_chart import display_funnel_chart

event, S = setup("Flood Explorer", "🌍")

st.title("Flood Explorer")
st.caption(f"Event: {event} | Study area: Dadu, Larkana and Kamber Shahdadkot (Sentinel-1, one orbit, descending). "
           "Pre image 18 Aug 2022, post image 30 Aug 2022.")
demo_banner(S)

ex = S.get("exposure", {}) or {}
tm = S.get("threshold_map", {}) or {}
lc = ex.get("landcover_km2", {}) or {}
roads = ex.get("roads")
fac = ex.get("facilities") or {}

st.subheader("Exposure at a glance")
r1 = st.columns(3)
r2 = st.columns(3)
with r1[0]:
    display_metric_card("Flooded Area", fmt(tm.get("flood_km2")), "km²", "waves", "Threshold map, water seen at pass time")
with r1[1]:
    display_metric_card("People in Flood", fmt(ex.get("people_in_flood")), "", "groups", "WorldPop 2020 grid, not a census")
with r1[2]:
    display_metric_card("Cropland Flooded", fmt(lc.get("cropland")), "km²", "grass", "ESA WorldCover")
with r2[0]:
    display_metric_card("Built-up Flooded", fmt(lc.get("built-up"), 1), "km²", "home_work", "Buildings and settlements")
with r2[1]:
    display_metric_card("Roads Flooded", fmt(roads.get("road_km_flooded"), 1) if roads else "n/a", "km" if roads else "",
                        "add_road", "OpenStreetMap roads" if roads else "OpenStreetMap not fetched yet")
with r2[2]:
    hs = (f"{get(fac, 'health', 'in_flood', default='n/a')} / {get(fac, 'education', 'in_flood', default='n/a')}"
          if fac else "n/a")
    display_metric_card("Hospitals / Schools", hs, "", "local_hospital",
                        "In flooded area" if fac else "OpenStreetMap not fetched yet")
st.caption("People = WorldPop 2020 grid estimate, not a census count.")

L = load_layers(event, "event")
st.divider()
st.subheader("Geospatial flood map")
if L is None:
    na("Map layers were not generated for this event.")
else:
    names = [n for n in ("flood_threshold", "flood_ml", "s1_vh_post", "s1_vh_pre") if n in L["layers"]]
    left, right = st.columns([2, 1])
    with left:
        display_flood_map(L, names, key=f"map_{event}")
    with right:
        display_map_legend()
        st.caption("Layers: flood mask (on), ML map and radar images (switch on in the layer control).")

st.divider()
st.subheader("Before and after")
if L is not None:
    pre, post, flood = load_img(L, "s1_vh_pre"), load_img(L, "s1_vh_post"), load_img(L, "flood_threshold")
    if pre is not None and post is not None:
        show = st.toggle("Show flood mask on the right image", value=True)
        right_img = post
        if show and flood is not None:
            from PIL import Image
            right_img = Image.alpha_composite(post, flood.resize(post.size))
        display_image_comparison(pre, right_img, "Before (18 Aug 2022)", "Flood (30 Aug 2022)")
        st.caption("Drag the slider. Dark areas in radar are smooth surfaces such as water. The pre image may "
                   "already contain some water, so the mask is a minimum estimate of new water.")

st.divider()
st.subheader("How we cleaned the map")
try:
    display_funnel_chart(funnel_rows(S))
    st.caption("Each step removes pixels that are probably not new flood water: permanent water (JRC), steep "
               "terrain (radar shadow) and speckle noise.")
except Exception:
    na("Cleaning steps not available.")

st.download_button("Download summary.json", json.dumps(S, indent=2), file_name=f"{event}_summary.json")

with st.expander("Understanding the flood map"):
    st.markdown("- **Flood mask:** pixels that are dark in the flood image, darker than before, not permanent water "
                "and not steep terrain.\n- **Layer control:** switch base maps and radar layers.\n"
                "- **Exposure numbers** are computed by overlaying the mask on WorldPop and WorldCover.")
with st.expander("Scientific limitations"):
    st.markdown(
        "- Dry sandy ground can look dark in radar and may be mistaken for water, so the area is likely an "
        "overestimate.\n- The pre image (18 Aug 2022) may already contain some water.\n"
        "- Radar speckle, vegetation, buildings and viewing geometry affect results.\n"
        "- No ground truth exists for Sindh 2022, so no accuracy is claimed.\n"
        "- This is a research prototype and not for emergency decisions.")
footer()
