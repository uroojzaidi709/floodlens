import pandas as pd
import streamlit as st

from common import ROOT, setup, demo_banner, na, get, fmt, explain_what_if, footer
from components.metric_cards import display_metric_card

event, S = setup("Model & Explainability", "🧠")

st.title("Model & Explainability")
st.caption("LightGBM flood classifier trained on Sen1Floods11, tested on a held-out event")
demo_banner(S)

ml = S.get("ml_map")
if not ml:
    na("The ML map has not been generated yet (the model has not been trained). The threshold map is the main result.")
else:
    hv = ml.get("heldout_validation")
    st.subheader("Tested on an event the model never saw (Pakistan 2017)")
    if hv:
        keys = ["precision", "recall", "f1", "iou"]
        st.table(pd.DataFrame({"ML (LightGBM)": [get(hv, "ml", k) for k in keys],
                               "Threshold (VH < -18 dB)": [get(hv, "baseline", k) for k in keys]},
                              index=["Precision", "Recall", "F1", "IoU"]))
        st.caption("Accuracy comes from the held-out event. Sindh 2022 has no ground truth. Both methods over-call "
                   "water on dry ground (low precision), and ML is somewhat better.")
    else:
        na("Held-out validation report not found.")

    agr = get(ml, "agreement_with_threshold", "iou")
    c1, c2 = st.columns(2)
    with c1:
        display_metric_card("ML vs Threshold Agreement", fmt(agr, 2), "IoU", "compare_arrows",
                            "Agreement between two maps, not accuracy")

    st.subheader("Three flood-area estimates for Sindh 2022")
    px = S.get("pixel_m", 30.0)
    both = get(ml, "agreement_with_threshold", "both_pixels")
    k1, k2, k3 = st.columns(3)
    with k1:
        display_metric_card("Threshold map", fmt(get(S, "threshold_map", "flood_km2")), "km²", "waves", "Main result")
    with k2:
        display_metric_card("ML map", fmt(ml.get("flood_km2")), "km²", "psychology", "Upper bound")
    with k3:
        display_metric_card("Both agree", fmt(both * px * px / 1e6) if both else "n/a", "km²", "check_circle",
                            "Cautious estimate")
    st.caption("The ML map flags much more area than the threshold map. On the held-out Pakistan event both methods "
               "over-call water on dry, sandy ground, so the ML area is an **upper bound**. The area where both "
               "agree is the more cautious estimate. All three are water seen at pass time, with no ground truth "
               "for Sindh 2022.")

st.divider()
st.subheader("What the model looks at (SHAP)")
shap_png = ROOT / "outputs" / event / "shap_importance.png"
if shap_png.exists():
    st.image(str(shap_png), caption=f"Method: {get(S, 'explainability', 'method', default='n/a')}",
             use_container_width=True)
    st.caption("A positive SHAP value pushes a pixel toward 'flood'. It explains the model, not the physics. "
               "VH backscatter is the strongest driver.")
else:
    na("SHAP chart not available yet.")

st.divider()
st.subheader("What-if: change the radar values")
model_path = ROOT / "ml" / "flood_lgbm.joblib"
if model_path.exists():
    c1, c2 = st.columns(2)
    vv = c1.slider("VV (dB)", -35.0, 5.0, -15.0, 0.5)
    vh = c2.slider("VH (dB)", -40.0, 0.0, -22.0, 0.5)
    try:
        r = explain_what_if(model_path, vv, vh)
        prob = r.get("probability_flood", r.get("probability", r.get("prob")))
        if prob is not None:
            st.progress(min(max(float(prob), 0.0), 1.0), text=f"Flood probability: {float(prob):.0%}")
        contrib = r.get("contributions")
        if contrib:
            from components.flood_chart import display_bar_chart
            if isinstance(contrib, list):
                contrib = {d["feature"]: d["shap"] for d in contrib}
            s = pd.Series(contrib).sort_values()
            display_bar_chart(list(s.index), list(s.values), "Contribution of each input (positive = toward flood)",
                              "SHAP value (log-odds)")
    except Exception:
        na("Per-pixel explanation not available (shap or lightgbm missing).")
else:
    na("Train the model first (python pipeline/train_model.py) to enable the what-if sliders.")

st.divider()
st.subheader("Facts about the model")
st.table({"Item": ["Model", "Training data", "Test", "Inputs", "Uses a pre-flood image?", "Baseline"],
          "Value": ["LightGBM classifier", "Sen1Floods11 hand-labelled chips (446)",
                    "Leave-one-event-out (Pakistan 2017 held out)",
                    "VV dB, VH dB, VV minus VH, VH local std, VV local std (5 features)",
                    "No, post-flood radar only, so it transfers to Sindh",
                    "VH below -18 dB threshold"]})
footer()
