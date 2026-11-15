
import streamlit as st
from html import escape


def display_metric_card(title, value, unit, icon, description):

    st.html(f"""
    <div class="fl-metric-card">

        <div class="fl-metric-header">

            <span class="fl-metric-title">
                {escape(str(title))}
            </span>

            <span class="material-symbols-outlined fl-metric-icon">
                {escape(str(icon))}
            </span>

        </div>

        <div class="fl-metric-value">
            {escape(str(value))}
            <span class="fl-metric-unit">
                {escape(str(unit))}
            </span>
        </div>

        <div class="fl-metric-description">
            {escape(str(description))}
        </div>

    </div>
    """)


def display_flood_metrics(
    extent,
    percentage,
    observations,
    confidence
):

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        display_metric_card(
            "Flood Extent",
            extent,
            "km²",
            "waves",
            "Estimated inundated area"
        )

    with col2:
        display_metric_card(
            "Flooded Region",
            percentage,
            "%",
            "map",
            "Percentage of surveyed area"
        )

    with col3:
        display_metric_card(
            "Observations",
            observations,
            "",
            "satellite_alt",
            "Satellite observations"
        )

    with col4:
        display_metric_card(
            "Model Confidence",
            confidence,
            "%",
            "analytics",
            "Estimated prediction confidence"
        )
