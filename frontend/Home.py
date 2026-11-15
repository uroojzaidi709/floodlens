
import streamlit as st
from pathlib import Path
import base64

st.set_page_config(
    page_title="FloodLens | Earth Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent

css_file = BASE_DIR / "styles" / "home.css"
hero_file = BASE_DIR / "assets" / "images" / "earth_hero.png"
logo_file = BASE_DIR / "assets" / "floodlens_logo.svg"
video_file = BASE_DIR / "static" / "earth_orbit.mp4"

css = css_file.read_text(encoding="utf-8")

hero_image = base64.b64encode(
    hero_file.read_bytes()
).decode("utf-8")

logo_image = base64.b64encode(
    logo_file.read_bytes()
).decode("utf-8")


# AUTOMATIC PAGE DETECTION

def find_page(keyword):
    for file in sorted((BASE_DIR / "pages").glob("*.py")):
        filename = file.stem.lower()
        filename = filename.replace("_", "").replace(" ", "").replace("-", "")

        if keyword.lower() in filename:
            return f"pages/{file.name}"

    return None


dashboard_page = find_page("missionbrief")
satellite_page = find_page("flood")
about_page = find_page("about")


# LOAD CSS

st.markdown(
    f"<style>{css}</style>",
    unsafe_allow_html=True
)


# TOP NAVIGATION

nav1, nav2, nav3, nav4, nav5 = st.columns(
    [3, 1, 1, 1.4, 1],
    vertical_alignment="center"
)

with nav1:
    st.markdown(f"""
    <div class="floodlens-brand">
        <img src="data:image/svg+xml;base64,{logo_image}">
        <span>FloodLens</span>
    </div>
    """, unsafe_allow_html=True)

with nav2:
    st.page_link("Home.py", label="Home")

with nav3:
    if dashboard_page:
        st.page_link(
            dashboard_page,
            label="Mission Brief"
        )
    else:
        st.caption("Mission Brief unavailable")

with nav4:
    if satellite_page:
        st.page_link(
            satellite_page,
            label="Flood Explorer"
        )
    else:
        st.caption("Flood Explorer unavailable")

with nav5:
    if about_page:
        st.page_link(
            about_page,
            label="About"
        )
    else:
        st.caption("About unavailable")


# VIDEO BACKGROUND

video_html = ""

if video_file.exists():
    video_html = """
    <video class="hero-video"
           autoplay
           muted
           loop
           playsinline
           preload="auto"
           aria-hidden="true">
        <source src="./app/static/earth_orbit.mp4"
                type="video/mp4">
    </video>
    """
else:
    st.warning(
        "Video file not found. Displaying static background."
    )


# HERO SECTION

st.html(f"""
<div class="hero-section" style="
    background-image: url('data:image/png;base64,{hero_image}');
    background-size: cover;
    background-position: center;
">

    {video_html}

    <div class="hero-overlay"></div>

    <div class="hero-content">

        <div class="hero-eyebrow">
            EARTH OBSERVATION • FLOOD INTELLIGENCE
        </div>

        <div class="hero-title">
            See Beyond<br>
            the <span>Surface.</span>
        </div>

        <div class="hero-description">
            A new perspective on our changing planet.
            Explore the power of satellite intelligence,
            machine learning and scientific transparency
            in understanding floods.
        </div>

    </div>

</div>
""")


# EXPLORE BUTTON

left, center, right = st.columns([2, 1, 2])

with center:
    if st.button(
        "Explore FloodLens →",
        use_container_width=True,
        disabled=dashboard_page is None
    ):
        st.switch_page(dashboard_page)


# PLATFORM FEATURES

st.markdown("""
<div class="section-heading">
    <p>OBSERVE • ANALYZE • VERIFY</p>
    <h2>Explore a Changing Planet.</h2>
</div>
""", unsafe_allow_html=True)


space1, col1, col2, col3, space2 = st.columns(
    [0.2, 1, 1, 1, 0.2]
)

with col1:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-number">01 / OBSERVE</span>
        <h3>Satellite Intelligence</h3>
        <p>
            Explore Earth observation data and
            discover geographical changes through
            Synthetic Aperture Radar technology.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-number">02 / ANALYZE</span>
        <h3>AI Flood Detection</h3>
        <p>
            Discover how machine learning can
            identify flooded regions and reveal
            patterns in surface water.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-number">03 / VERIFY</span>
        <h3>Scientific Transparency</h3>
        <p>
            Understand the origin, processing
            history and integrity of satellite
            observations.
        </p>
    </div>
    """, unsafe_allow_html=True)


# FOOTER

# ABOUT FLOODLENS

about_file = BASE_DIR / "assets" / "images" / "about_image.jpg"

if about_file.exists():

    about_image = base64.b64encode(
        about_file.read_bytes()
    ).decode("utf-8")

    st.html(f"""
    <section class="about-section" id="about">

        <div class="about-header">
            <div class="about-eyebrow">
                BEHIND THE TECHNOLOGY
            </div>

            <h2>About FloodLens</h2>
        </div>

        <div class="about-grid">

            <div class="about-visual">

                <img
                    class="about-image"
                    src="data:image/jpeg;base64,{about_image}"
                    alt="Satellite imagery of Sindh, Pakistan"
                >

                <div class="about-image-caption">
                    SINDH, PAKISTAN | EARTH OBSERVATION IMAGERY
                </div>

            </div>

            <div class="about-content">

                <div class="about-eyebrow">
                    OUR MISSION
                </div>

                <h2>
                    Understanding Earth.
                    <span>Protecting Tomorrow.</span>
                </h2>

                <p>
                    Floods are among the most devastating
                    natural disasters, affecting communities,
                    infrastructure and ecosystems across
                    the world.
                </p>

                <p>
                    Inspired by the devastating floods of
                    Pakistan, FloodLens aims to bridge the
                    gap between satellite intelligence and
                    meaningful environmental insights.
                </p>

                <p>
                    By combining Synthetic Aperture Radar,
                    machine learning and scientific
                    transparency, our platform aims to make
                    flood observations more accessible,
                    understandable and trustworthy.
                </p>

                <p>
                    Developed for the NASA Space Apps
                    Challenge 2026.
                </p>

            </div>

        </div>

        <div class="about-stats">

            <div class="about-stat">
                <h3>SAR</h3>
                <p>EARTH OBSERVATION</p>
            </div>

            <div class="about-stat">
                <h3>AI</h3>
                <p>FLOOD INTELLIGENCE</p>
            </div>

            <div class="about-stat">
                <h3>03</h3>
                <p>TEAM MEMBERS</p>
            </div>

        </div>

    </section>
    """)

else:
    st.warning("About section image not found.")


st.markdown("""
<div class="home-footer">
    <p>F L O O D L E N S</p>
    <p>Earth Observation • Artificial Intelligence • Scientific Transparency</p>
    <p>NASA Space Apps Challenge 2026 | Research Prototype</p>
    <p>Homepage visuals are for illustrative purposes only.</p>
</div>
""", unsafe_allow_html=True)
