
import streamlit as st
from pathlib import Path
import base64

def apply_theme():

    base_dir = Path(__file__).resolve().parent.parent

    css_file = base_dir / "styles" / "global.css"
    logo_file = base_dir / "assets" / "floodlens_logo.svg"

    css = css_file.read_text(encoding="utf-8")

    logo = base64.b64encode(
        logo_file.read_bytes()
    ).decode("utf-8")

    st.markdown(
        f"""
        <style>
        {css}

        [data-testid="stSidebarNav"]::before {{
            background-image: url(
                'data:image/svg+xml;base64,{logo}'
            );
            background-repeat: no-repeat;
            background-position: center top;
            background-size: 76px 76px;
            padding-top: 88px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.markdown("""
        <div class="sidebar-footer">
            <p class="prototype-status">
                ● Research Prototype
            </p>
            <p class="sidebar-event">
                NASA Space Apps Challenge 2026
            </p>
            <p class="sidebar-tagline">
                See Beyond the Surface.
            </p>
        </div>
        """, unsafe_allow_html=True)
