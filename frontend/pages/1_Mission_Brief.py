import streamlit as st

from common import setup, demo_banner, fmt, get, footer
from components.metric_cards import display_metric_card

event, S = setup("Mission Brief", "🛰️")

st.title("Mission Brief")
st.caption("FloodLens: seeing the 2022 Sindh flood through radar")
demo_banner(S)

st.markdown(
    "In 2022 floods covered large parts of Upper Sindh. Clouds and night hide floods from normal satellite "
    "photos, but **radar sees through both**. FloodLens maps the flood with Sentinel-1, adds NASA-ISRO's new "
    "NISAR L-band radar, estimates who and what was exposed, and shows how far the result can be trusted.")

ex = S.get("exposure", {}) or {}
c1, c2, c3, c4 = st.columns(4)
with c1:
    display_metric_card("Flooded Area", fmt(get(S, "threshold_map", "flood_km2")), "km²", "waves",
                        "Threshold map, water seen at pass time")
with c2:
    display_metric_card("People in Flood", fmt(ex.get("people_in_flood")), "", "groups",
                        "WorldPop 2020 estimate, not a census")
with c3:
    display_metric_card("Cropland Flooded", fmt(get(ex, "landcover_km2", "cropland")), "km²", "grass",
                        "ESA WorldCover classes")
with c4:
    display_metric_card("Forensic Status", str(get(S, "forensics", "status", default="n/a")).replace("_", " "),
                        "", "verified_user", "Checks passed, not 'verified correct'")

st.divider()
st.subheader("The problem and our solution")
p1, p2, p3 = st.columns(3)
with p1.container(border=True):
    st.markdown("### 1. The problem")
    st.markdown(
        "- After a flood, relief teams need to know **where the water went and who and what it reached**.\n"
        "- Clouds and night hide floods from normal satellites. Radar sees through both, but radar data is hard to use.\n"
        "- Research already shows radar can map floods. The gap: a map alone does not say **what it means for "
        "people and crops**, and a reader cannot tell **whether to trust it**.")
with p2.container(border=True):
    st.markdown("### 2. Our solution")
    st.markdown(
        "- A flood map from **Sentinel-1** for Upper Sindh 2022 (Dadu, Larkana, Kamber Shahdadkot).\n"
        "- It is turned into **numbers people can act on**: people, cropland, roads, health and school facilities.\n"
        "- **NISAR L-band** (2026) shown next to it, so the new NASA-ISRO satellite is part of the story.\n"
        "- A **trust layer**: every input and output is hashed, and checks show the data is comparable.")
with p3.container(border=True):
    st.markdown("### 3. What is different")
    st.markdown(
        "- **Trust built in:** SHA-256 manifest, same-orbit check and a tamper demo, not just a map.\n"
        "- **Honest scope:** no accuracy claimed for Sindh 2022. The ML model is tested only on a held-out "
        "event (Pakistan 2017).\n"
        "- **Explained:** SHAP shows why the model calls a pixel flood.\n"
        "- **One reproducible pipeline** from Earth Engine to this app, with C-band and L-band together.")
st.caption("We do not claim to be the first to map this flood. Our contribution is making the result "
           "decision-ready, explainable and checkable.")

st.divider()
st.subheader("How radar sees water")
a, b, c = st.columns(3)
a.markdown("**1. Water is dark**\n\nSmooth water bounces the radar signal away from the satellite.")
b.markdown("**2. Land and crops are bright**\n\nRough ground scatters the signal back.")
c.markdown("**3. Before minus after**\n\nA pixel that turns dark between two dates is new water.")

st.subheader("Two satellites")
st.table({
    "": ["Band", "Wavelength", "Role in FloodLens", "Good at"],
    "Sentinel-1": ["C-band", "about 5.5 cm", "Maps the 2022 flood", "Open water, frequent repeat passes"],
    "NISAR": ["L-band", "about 24 cm", "2026 monsoon view over Dadu", "Seeing deeper into crops and vegetation"],
})

st.subheader("Benchmark and references")
refs = [
    ("Chapman et al. 2015: L-band radar for inundation mapping", "https://doi.org/10.3390/rs70505440",
     "Scientific basis for L-band complementing C-band in flood and wetland mapping."),
    ("NASA SVS 5661: NISAR view of a flood in Mozambique", "https://svs.gsfc.nasa.gov/5661",
     "NASA's own NISAR flood example; we follow the same before/after idea over Dadu."),
    ("Upper Sindh 2022 Sentinel-1 study (Ecological Questions, 2026)", "https://doi.org/10.12775/EQ.2026.027",
     "Regional benchmark for our Sentinel-1 flood extent."),
    ("Sen1Floods11 (Bonafilia et al. 2020)", "https://github.com/cloudtostreet/Sen1Floods11",
     "Hand-labelled flood chips used to train and test the ML model."),
]
cols = st.columns(2)
for i, (t, u, why) in enumerate(refs):
    with cols[i % 2].container(border=True):
        st.markdown(f"**[{t}]({u})**")
        st.caption(why)

with st.container(border=True):
    st.markdown("**Honest scope.** We do not claim an accuracy for Sindh 2022, because no ground truth exists. "
                "The ML model's accuracy is measured on a held-out Pakistan 2017 event. NISAR launched in 2025, "
                "so it did not observe the 2022 flood.")
footer()
