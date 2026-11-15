import streamlit as st
import plotly.graph_objects as go


def display_funnel_chart(rows):
    """Bar chart of how many km2 survive each cleaning step (rows from services.funnel_rows)."""
    steps = [r["step"] for r in rows]
    km2 = [r["km2"] for r in rows]
    fig = go.Figure(go.Bar(
        y=steps, x=km2, orientation="h", marker=dict(color="#00B8C4"),
        text=[f"{v:,.0f} km²" for v in km2], textposition="auto",
        hovertemplate="%{y}<br>%{x:,.1f} km²<extra></extra>"))
    fig.update_layout(
        paper_bgcolor="#081321", plot_bgcolor="#081321", font=dict(color="#EAF4FF"),
        xaxis=dict(title="Area (km²)", gridcolor="#243F56"),
        yaxis=dict(autorange="reversed", showgrid=False),
        margin=dict(l=30, r=30, t=30, b=30), height=360, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def display_bar_chart(labels, values, title, xtitle):
    fig = go.Figure(go.Bar(
        y=labels, x=values, orientation="h", marker=dict(color="#00B8C4"),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(
        title=title, paper_bgcolor="#081321", plot_bgcolor="#081321", font=dict(color="#EAF4FF"),
        xaxis=dict(title=xtitle, gridcolor="#243F56"), yaxis=dict(showgrid=False),
        margin=dict(l=30, r=30, t=60, b=30), height=320, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
