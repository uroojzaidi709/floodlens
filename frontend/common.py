"""Shared helpers for every page: backend import, event selector, real-data loaders and small UI pieces."""
import sys
from pathlib import Path

import streamlit as st

FRONTEND = Path(__file__).resolve().parent
ROOT = FRONTEND.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(FRONTEND))

from services import (check_uploaded_file, explain_what_if, funnel_rows, list_events,  # noqa: E402,F401
                      load_layers, load_manifest, load_summary, tamper_check)
from styles.theme import apply_theme  # noqa: E402

SCOPE = "Shows water seen at satellite pass time. Not a forecast."
GITHUB = "https://github.com/"  # TODO: put the repository link here


def setup(title, icon="🌍"):
    """Page config + theme + global event selector. Returns (event, summary)."""
    st.set_page_config(page_title=f"{title} | FloodLens", page_icon=icon, layout="wide",
                       initial_sidebar_state="expanded")
    apply_theme()
    st.markdown("""<style>
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    [data-testid="stMetricValue"] > div { white-space: normal !important; overflow: visible !important; }
    [data-testid="stMetricLabel"] p { white-space: normal !important; overflow: visible !important; }
    </style>""", unsafe_allow_html=True)

    events = list_events()
    with st.sidebar:
        st.markdown("**Event**")
        if not events:
            st.error("No results in outputs/. Run tools/make_demo_fixture.py or run_pipeline.py")
            st.stop()
        cur = st.session_state.get("event_sel")
        default = events.index("sindh2022") if "sindh2022" in events else 0
        event = st.selectbox("Event", events, index=events.index(cur) if cur in events else default,
                             label_visibility="collapsed")
        st.session_state["event_sel"] = event
        st.caption(SCOPE)
        st.markdown(f"[GitHub repository]({GITHUB})")
    summary = load_summary(event)
    return event, summary


def na(msg="Not available yet."):
    st.info(msg, icon="⏳")


def get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def fmt(x, nd=0):
    if x is None:
        return "n/a"
    try:
        return f"{x:,.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def load_img(layers, name):
    from PIL import Image
    try:
        return Image.open(layers["layers"][name]["path"]).convert("RGBA")
    except Exception:
        return None


def demo_banner(summary):
    if "WARNING" in summary:
        st.warning("DEMO DATA: " + summary["WARNING"])


def footer():
    st.divider()
    st.caption("FloodLens | NASA Space Apps Challenge 2026")
