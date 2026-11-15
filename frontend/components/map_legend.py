import streamlit as st


def display_map_legend(items=None):
    """items: list of (hex colour, label). Default describes the layers on the Flood Explorer map."""
    items = items or [("#38A9DB", "Flood mask (threshold map, main result)"),
                      ("#8A8F98", "Sentinel-1 VH radar (dark = smooth surface or water)")]
    rows = "".join(
        f"""<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
            <span style="display:inline-block;width:16px;height:16px;background:{c};border-radius:3px;"></span>
            {label}</div>""" for c, label in items)
    st.html(f"""
    <div style="background-color:#101F30;border:1px solid #243F56;border-radius:10px;
                padding:18px;margin-top:10px;color:#EAF4FF;">
        <h4 style="color:#00B8C4;letter-spacing:2px;margin-top:0;">MAP LEGEND</h4>
        {rows}
        <hr style="border-color:#243F56;">
        <p style="font-size:12px;color:#A9BDCF;margin-bottom:0;">
            Water seen at satellite pass time. Use the layer control (top right of the map) to switch layers.</p>
    </div>""")
