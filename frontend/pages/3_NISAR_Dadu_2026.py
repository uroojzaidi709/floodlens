import streamlit as st

from common import setup, demo_banner, na, get, fmt, load_img, load_layers, footer
from components.metric_cards import display_metric_card
from components.image_comparison import display_image_comparison

event, S = setup("NISAR Dadu 2026", "🛰️")

st.title("NISAR L-band over Dadu, 2026")
st.caption("A 30 km window, 20 Jun vs 31 Aug 2026 (NISAR L2 GCOV, track 92 descending)")
demo_banner(S)

with st.container(border=True):
    st.markdown("**Please read first.** NISAR launched in 2025, so it did **not** observe the 2022 flood. The 2026 "
                "monsoon in Sindh was minor. These scenes show surface change over Dadu, not a flood. The 2022 "
                "flood evidence is Sentinel-1.")

N = load_layers(event, "nisar")
if N is None:
    na("NISAR layers not generated for this event. Run run_pipeline.py with --nisar-dir.")
else:
    view = st.radio("View", ["Colour composite (RGB)", "HH", "HV", "Change HH", "Change HV"], horizontal=True)
    if view.startswith("Colour"):
        a, b = load_img(N, "nisar_rgb_before"), load_img(N, "nisar_rgb_after")
        if a is not None and b is not None:
            display_image_comparison(a, b, "20 Jun 2026", "31 Aug 2026")
        else:
            na("RGB layers missing. Re-run nisar_extract.py, then run_pipeline.py.")
    elif view in ("HH", "HV"):
        k = view.lower()
        a, b = load_img(N, f"nisar_{k}_before"), load_img(N, f"nisar_{k}_after")
        if a is not None and b is not None:
            display_image_comparison(a, b, "20 Jun 2026", "31 Aug 2026")
        else:
            na("Layer missing.")
    else:
        k = "hh" if view.endswith("HH") else "hv"
        img = load_img(N, f"nisar_{k}_change")
        if img is not None:
            st.image(img, caption="After minus before, in dB. Blue = darker, red = brighter.",
                     use_container_width=True)
        else:
            na("Layer missing.")
    with st.expander("Reading the colours", expanded=False):
        st.markdown("Colour composite: **red = HH, green = HV, blue = HH/HV ratio**. Dark = smooth surface or water. "
                    "Bright = rough ground or double-bounce (for example buildings or flooded crops). "
                    "The dark diagonal strip is a sub-swath no-signal gap, masked out of the numbers.")

st.divider()
st.subheader("Change statistics")
nis = S.get("nisar") or {}
pols = nis.get("polarizations") or get(nis, "stats", default={}) or {}
if pols:
    cols = st.columns(len(pols))
    for col, (pol, v) in zip(cols, pols.items()):
        with col:
            st.markdown("**" + {"HHHH": "HH polarization", "HVHV": "HV polarization"}.get(pol, pol) + "**")
            display_metric_card("Mean change", fmt(v.get("mean_change_db"), 2), "dB", "swap_vert", "After minus before")
            display_metric_card("Brighter than +3 dB", fmt(v.get("pct_brighter_than_plus3db"), 1), "%", "trending_up",
                                "Share of pixels")
            display_metric_card("Darker than -3 dB", fmt(v.get("pct_darker_than_minus3db"), 1), "%", "trending_down",
                                "Share of pixels")
    mask = get(nis, "low_signal_mask", "pct_of_aoi")
    if mask is not None:
        st.caption(f"A {mask}% no-signal strip (sub-swath gap, dark in both dates) is masked out.")
else:
    na("NISAR statistics not available.")

st.divider()
l, r = st.columns(2)
with l.container(border=True):
    st.markdown("### What could be driving the change? (hypotheses)")
    st.markdown("- Crop growth in irrigated fields during the monsoon season (brighter HH).\n"
                "- Shifting Indus channels or sandbars.\n\nThese are hypotheses, not findings.")
with r.container(border=True):
    st.markdown("### C-band vs L-band")
    st.markdown("L-band has a longer wavelength (about 24 cm vs about 5.5 cm), so it penetrates vegetation "
                "deeper than Sentinel-1.")
st.caption("Product: NISAR L2 GCOV, PROVISIONAL (CRID P05023), 20 m, track 92 descending, frame 75. "
           "The NISAR window is 20 m and the Sentinel-1 stack is 30 m, so they are shown on separate views.")
footer()


