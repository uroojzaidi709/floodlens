import streamlit as st
from styles.theme import apply_theme

st.set_page_config(
    page_title="About | FloodLens",
    page_icon="🌍",
    layout="wide"
)
apply_theme()

st.markdown("""
<div style="text-align:center;padding:40px 10px;">
    <h1 style="color:#00B8C4;font-size:65px;">
        FloodLens
    </h1>
    <h2 style="color:#EAF4FF;">
        See Beyond the Surface.
    </h2>
    <p style="color:#A9BDCF;font-size:18px;">
        NASA Space Apps Challenge 2026
        | Dancing with the SARs
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

st.subheader("Our Mission")

st.write(
    "FloodLens is an Earth observation platform "
    "designed to transform satellite data into "
    "accessible and meaningful flood intelligence."
)

st.write(
    "By combining Synthetic Aperture Radar (SAR), "
    "machine learning, digital forensics and "
    "interactive geospatial visualization, "
    "FloodLens aims to support a better understanding "
    "of flooding events and their geographical impact."
)

st.divider()

st.subheader("Our Core Objectives")

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.header("🛰️")
        st.subheader("Observe")
        st.write(
            "Explore satellite observations "
            "and monitor changes in surface water."
        )

with col2:
    with st.container(border=True):
        st.header("🧠")
        st.subheader("Analyze")
        st.write(
            "Apply machine learning to identify "
            "and visualize flooded regions."
        )

with col3:
    with st.container(border=True):
        st.header("🛡️")
        st.subheader("Verify")
        st.write(
            "Support scientific transparency through "
            "data provenance and integrity verification."
        )

st.divider()

st.subheader("Why Synthetic Aperture Radar?")

st.write(
    "Synthetic Aperture Radar is an active remote "
    "sensing technology that uses microwave signals "
    "to observe the Earth's surface."
)

c1, c2 = st.columns(2)

with c1:
    st.markdown("**Advantages of SAR**")

    st.markdown("""
    - Day and night observations
    - Ability to observe through most cloud cover
    - Monitoring of surface water
    - Large-scale geographical observations
    """)

with c2:
    st.markdown("**Applications in Flood Monitoring**")

    st.markdown("""
    - Flood extent mapping
    - Surface water change detection
    - Historical flood analysis
    - Disaster impact assessment
    """)

st.caption(
    "SAR flood detection can be affected by terrain, "
    "vegetation, urban structures and radar geometry."
)

st.divider()

st.subheader("How FloodLens Works")

st.caption("The workflow used in this project.")

workflow = [
    "Satellite Data Acquisition",
    "SAR Image Preprocessing",
    "Data Integrity and Metadata Verification",
    "Machine Learning Flood Segmentation",
    "Flood Extent Analysis",
    "Interactive Visualization"
]

for index, stage in enumerate(workflow):

    st.markdown(f"""
    <div style="
        background:#14283B;
        padding:18px;
        border-radius:10px;
        border-left:4px solid #00B8C4;
        text-align:center;
        color:#EAF4FF;">
        <b>{index + 1}. {stage}</b>
    </div>
    """, unsafe_allow_html=True)

    if index < len(workflow) - 1:

        st.markdown("""
        <div style="
            text-align:center;
            color:#00B8C4;
            font-size:26px;">
            ↓
        </div>
        """, unsafe_allow_html=True)

st.divider()

st.subheader("Meet the Team")

t1, t2, t3 = st.columns(3)

with t1:
    with st.container(border=True):

        st.header("💻")
        st.subheader("Your Name")

        st.caption(
            "Frontend Development & Visualization"
        )

        st.markdown("""
        - Interactive dashboards
        - Geospatial visualization
        - User interface design
        - Frontend integration
        """)

with t2:
    with st.container(border=True):

        st.header("🛰️")
        st.subheader("Urooj")

        st.caption(
            "SAR Processing & Digital Forensics"
        )

        st.markdown("""
        - Satellite data acquisition
        - SAR preprocessing
        - Metadata validation
        - Data integrity and provenance
        """)

with t3:
    with st.container(border=True):

        st.header("🤖")
        st.subheader("Amna")

        st.caption(
            "Machine Learning & Flood Detection"
        )

        st.markdown("""
        - Flood detection models
        - Image segmentation
        - Model evaluation
        - Flood extent analysis
        """)

st.divider()

st.subheader("Technology Stack")

technologies = [
    "Python",
    "Streamlit",
    "Folium",
    "Plotly",
    "Pandas",
    "NumPy",
    "SAR Remote Sensing",
    "Machine Learning",
    "Digital Forensics"
]

badges = "".join([
    f"""
    <span style="
        display:inline-block;
        background:#14283B;
        color:#00E5FF;
        padding:10px 18px;
        margin:5px;
        border:1px solid #24455B;
        border-radius:20px;">
        {tech}
    </span>
    """
    for tech in technologies
])

st.markdown(badges, unsafe_allow_html=True)

st.divider()

st.subheader("Earth Observation Missions")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):

        st.subheader("NISAR")

        st.write(
            "NASA-ISRO Synthetic Aperture Radar, "
            "a joint Earth observation mission "
            "developed by NASA and ISRO."
        )

        st.link_button(
            "Explore NISAR",
            "https://nisar.jpl.nasa.gov/"
        )

with col2:
    with st.container(border=True):

        st.subheader("Sentinel-1")

        st.write(
            "A Copernicus Earth observation mission "
            "providing radar imagery for environmental "
            "monitoring and other applications."
        )

        st.link_button(
            "Explore Sentinel-1",
            "https://www.esa.int/"
        )

st.divider()

st.subheader("Project Status & Limitations")

st.info(
    "FloodLens is a research prototype built for the NASA Space Apps "
    "Challenge 2026. It runs on real Sentinel-1 (2022) and NISAR (2026) data, "
    "a LightGBM model with SHAP explanations, and a SHA-256 provenance manifest."
)

st.write(
    "Flood extent is the water seen at satellite pass time. There is no ground "
    "truth for Sindh 2022, NISAR data is provisional, and results should not be "
    "used for emergency response decisions."
)

st.divider()

st.markdown("""
<div style="text-align:center;padding:25px;">
    <h2 style="color:#00B8C4;">
        See Beyond the Surface.
    </h2>
    <p style="color:#A9BDCF;">
        Earth Observation • Artificial Intelligence
        • Scientific Transparency
    </p>
    <p style="color:#A9BDCF;">
        Developed for NASA Space Apps Challenge 2026
    </p>
</div>
""", unsafe_allow_html=True)


