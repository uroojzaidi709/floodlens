import streamlit as st

from common import setup, demo_banner, fmt, get, footer

event, S = setup("Context & Benchmarks", "📚")

st.title("Context and Benchmarks")
st.caption("How our result sits next to published work and official reports")
demo_banner(S)

st.subheader("Sanity check against PDMA (people affected, 2022)")
ex = S.get("exposure", {}) or {}
c1, c2 = st.columns(2)
with c1.container(border=True):
    st.markdown("**PDMA Sindh 2022 report (affected people)**")
    st.table({"District": ["Dadu", "Larkana"], "Affected (per PDMA)": ["849,380", "1,217,428"]})
    st.caption("Source: PDMA Sindh 2022 flood report. Totals inside that report are not fully consistent, so we cite "
               "district figures only.")
with c2.container(border=True):
    st.markdown("**FloodLens (people inside flooded pixels)**")
    st.metric("People in flood, whole study area", fmt(ex.get("people_in_flood")))
    st.caption("WorldPop 2020 grid. Study area also includes Kamber Shahdadkot.")
st.info("The two numbers are the same order of magnitude. They measure different things (people affected vs people "
        "in flooded pixels), so this is a sanity check, not a validation.", icon="ℹ️")

st.divider()
st.subheader("Published benchmarks")
refs = [
    ("Chapman et al. 2015", "L-band radar for inundation mapping", "https://doi.org/10.3390/rs70505440",
     "Why we pair L-band (NISAR) with C-band (Sentinel-1)."),
    ("NASA SVS 5661", "NISAR view of a flood in Mozambique", "https://svs.gsfc.nasa.gov/5661",
     "NASA's NISAR flood workflow; we follow the same before/after idea over Dadu."),
    ("Ecological Questions 2026", "Upper Sindh 2022, Sentinel-1", "https://doi.org/10.12775/EQ.2026.027",
     "Same region and satellite; our regional benchmark."),
    ("Bonafilia et al. 2020", "Sen1Floods11", "https://github.com/cloudtostreet/Sen1Floods11",
     "Hand-labelled flood chips used to train and test the ML model."),
]
cols = st.columns(2)
for i, (a, t, u, why) in enumerate(refs):
    with cols[i % 2].container(border=True):
        st.markdown(f"**{a}**")
        st.markdown(f"[{t}]({u})")
        st.caption(why)

st.divider()
st.subheader("2026 context")
st.markdown("The 2026 Sindh monsoon was minor (PDMA situation reports to 21 Aug 2026 list 40 houses damaged and no "
            "crop loss). That is why the NISAR scenes over Dadu show surface change and not a flood.")
footer()
