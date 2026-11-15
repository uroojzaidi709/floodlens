import streamlit as st

from common import setup, demo_banner, na, load_img, load_layers, footer
from components.image_comparison import display_image_comparison

event, S = setup("Compare Imagery", "🔍")

st.title("Compare Imagery")
st.caption("Sentinel-1 radar before and during the 2022 flood (same orbit, same direction)")
demo_banner(S)

L = load_layers(event, "event")
if L is None:
    na("Layers not generated for this event.")
else:
    band = st.radio("Band", ["VH (best for water)", "VV"], horizontal=True)
    k = "vh" if band.startswith("VH") else "vv"
    pre = load_img(L, f"s1_{k}_pre")
    post = load_img(L, f"s1_{k}_post")
    if pre is None or post is None:
        na("This band is not available for this event. Try VH.")
    else:
        opts = ["None"] + [lbl for n, lbl in (("flood_threshold", "Threshold flood mask"),
                                                ("flood_ml", "ML flood mask (upper bound)")) if n in L["layers"]]
        mask_sel = st.radio("Overlay on the flood image", opts, horizontal=True)
        right = post
        name = {"Threshold flood mask": "flood_threshold", "ML flood mask (upper bound)": "flood_ml"}.get(mask_sel)
        if name:
            m = load_img(L, name)
            if m is not None:
                from PIL import Image
                right = Image.alpha_composite(post, m.resize(post.size))
        display_image_comparison(pre, right, "Before (18 Aug 2022)", "Flood (30 Aug 2022)")
        st.caption("Dark = smooth surface such as water. The overlay shows pixels the method calls flood.")

c1, c2 = st.columns(2)
with c1.container(border=True):
    st.markdown("**How to read it**\n\nDrag the slider across the image. Water spreading across the plain turns "
                "dark between the two dates.")
with c2.container(border=True):
    st.markdown("**Caution**\n\nThe before image (18 Aug 2022) may already contain some water. Dry sandy ground can "
                "also look dark. No ground truth exists for Sindh 2022.")
footer()
