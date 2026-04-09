"""
Climate Risk Geospatial Assessment Dashboard
=============================================
Interactive Streamlit dashboard for exploring multi-hazard climate risk
analysis results across regions, scenarios, and time horizons.

Run: streamlit run dashboard/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
import os
from openai import OpenAI

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Climate Risk Assessment",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');

    .stApp { font-family: 'DM Sans', sans-serif; }

    .metric-card {
        background: linear-gradient(135deg, #d8f3dc 0%, #b7e4c7 100%);
        border-radius: 12px;
        padding: 20px 24px;
        color: #1b4332;
        margin-bottom: 8px;
        border-left: 4px solid #2d6a4f;
    }
    .metric-card .label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #52796f;
        margin-bottom: 4px;
    }
    .metric-card .value {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 2px;
        color: #1b4332;
    }
    .metric-card .delta {
        font-size: 13px;
        color: #bc4749;
    }
    .metric-card .delta.positive { color: #2d6a4f; }

    .section-header {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #52796f;
        border-bottom: 2px solid #95d5b2;
        padding-bottom: 8px;
        margin: 32px 0 16px 0;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #d8f3dc 0%, #b7e4c7 100%);
    }
    div[data-testid="stSidebar"] .stMarkdown { color: #1b4332; }

    .tcfd-pillar-card {
        background: linear-gradient(135deg, #f0faf4 0%, #e0f4e8 100%);
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 5px solid #2d6a4f;
        margin-bottom: 12px;
    }
    .tcfd-badge {
        display: inline-block;
        background: #2d6a4f;
        color: white;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .disclaimer-box {
        background: #fff9e6;
        border: 1px solid #f9c74f;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 12px;
        color: #7d6608;
        margin-top: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ── Data loading ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = Path(__file__).parent.parent / "data"
    return {
        "risk_scores": pd.read_csv(base / "output" / "risk_scores.csv"),
        "portfolio": pd.read_csv(base / "output" / "portfolio_summary.csv"),
        "eal": pd.read_csv(base / "output" / "expected_annual_loss.csv"),
        "delta": pd.read_csv(base / "output" / "delta_analysis.csv"),
        "grid": pd.read_csv(base / "output" / "grid_risk_output.csv"),
        "composite": pd.read_csv(base / "processed" / "composite_hazard.csv"),
    }

data = load_data()

# ── Scenario config ────────────────────────────────────────────────────
SCENARIO_COLORS = {
    "SSP1-2.6": "#2ecc71",
    "SSP2-4.5": "#f39c12",
    "SSP5-8.5": "#e74c3c",
    "historical": "#3498db",
}
SCENARIO_LABELS = {
    "SSP1-2.6": "SSP1-2.6 (Sustainability)",
    "SSP2-4.5": "SSP2-4.5 (Middle of the Road)",
    "SSP5-8.5": "SSP5-8.5 (Fossil-Fuelled)",
    "historical": "Historical Baseline",
}
REGION_LABELS = {
    "mato_grosso": "🇧🇷 Mato Grosso, Brazil",
    "rhine_meuse": "🇳🇱 Rhine-Meuse, Netherlands",
    "punjab": "🇮🇳 Punjab, India",
}

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌍 Climate Risk Assessment")
    st.markdown("**Multi-Hazard Physical Risk Analysis**")
    st.markdown("---")

    selected_scenario = st.selectbox(
        "Climate Scenario",
        options=["SSP1-2.6", "SSP2-4.5", "SSP5-8.5"],
        format_func=lambda x: SCENARIO_LABELS[x],
        index=1,
    )

    selected_year = st.select_slider(
        "Time Horizon",
        options=[2030, 2050, 2080],
        value=2050,
    )

    selected_regions = st.multiselect(
        "Regions",
        options=list(REGION_LABELS.keys()),
        default=list(REGION_LABELS.keys()),
        format_func=lambda x: REGION_LABELS[x],
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#52796f;'>
    <b>Data Sources</b><br>
    Climate: CMIP6/SSP projections<br>
    Exposure: MapSPAM, FAO statistics<br>
    Hazards: GEV return period analysis<br><br>
    <b>Kehinde Damilola Akindele</b><br>
    Geospatial Data Analyst
    </div>
    """, unsafe_allow_html=True)


# ── Helper functions ───────────────────────────────────────────────────
def fmt_usd(val):
    if val >= 1e9:
        return f"${val/1e9:.1f}B"
    elif val >= 1e6:
        return f"${val/1e6:.0f}M"
    else:
        return f"${val:,.0f}"

def metric_card(label, value, delta=None, delta_positive=False):
    delta_html = ""
    if delta:
        cls = "positive" if delta_positive else ""
        delta_html = f'<div class="delta {cls}">{delta}</div>'
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {delta_html}
    </div>
    """


# ══════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD — Tabs
# ══════════════════════════════════════════════════════════════════════

st.markdown("# 🌍 Climate Risk Geospatial Assessment")

tab_risk, tab_tcfd = st.tabs([
    "📊 Climate Risk Assessment",
    "📋 TCFD Disclosure Extractor",
])


# ──────────────────────────────────────────────────────────────────────
# TAB 1 — Climate Risk Assessment
# ──────────────────────────────────────────────────────────────────────
with tab_risk:

    st.markdown(f"**Scenario:** {SCENARIO_LABELS[selected_scenario]} · **Horizon:** {selected_year} · **Regions:** {len(selected_regions)}")

    # ── Key Metrics Row ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Portfolio Overview</div>', unsafe_allow_html=True)

    port = data["portfolio"]
    current = port[(port["scenario"] == selected_scenario) & (port["period"] == str(selected_year))]

    if not current.empty:
        row = current.iloc[0]
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(metric_card(
                "Portfolio Exposure",
                fmt_usd(row["portfolio_value_usd"]),
                f"{len(selected_regions)} regions",
            ), unsafe_allow_html=True)

        with col2:
            st.markdown(metric_card(
                "Expected Annual Loss",
                fmt_usd(row["portfolio_eal"]),
                f"{row['portfolio_loss_ratio_pct']}% of portfolio",
            ), unsafe_allow_html=True)

        with col3:
            st.markdown(metric_card(
                "Average Risk Score",
                f"{row['avg_risk_score']:.1f} / 100",
                f"Max: {row['max_risk_score']:.1f}",
            ), unsafe_allow_html=True)

        with col4:
            ref = port[(port["scenario"] == "SSP1-2.6") & (port["period"] == str(selected_year))]
            if not ref.empty and selected_scenario != "SSP1-2.6":
                diff = row["portfolio_eal"] - ref.iloc[0]["portfolio_eal"]
                st.markdown(metric_card(
                    "Excess Loss vs SSP1-2.6",
                    fmt_usd(abs(diff)),
                    f"{'Higher' if diff > 0 else 'Lower'} than sustainability path",
                ), unsafe_allow_html=True)
            else:
                st.markdown(metric_card(
                    "Scenario Path",
                    row["scenario_label"],
                    f"{selected_year} projection",
                ), unsafe_allow_html=True)

    # ── Risk Map ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Geospatial Risk Map</div>', unsafe_allow_html=True)

    REGION_CENTERS = {
        "mato_grosso": {"lat": -12.5, "lon": -56.0, "zoom": 4},
        "rhine_meuse":  {"lat": 51.5,  "lon": 5.5,   "zoom": 5},
        "punjab":       {"lat": 30.5,  "lon": 75.0,  "zoom": 5},
    }
    MAP_STYLES = {
        "Light (Natural)": "carto-positron",
        "Street Map":      "open-street-map",
        "Dark":            "carto-darkmatter",
    }
    RISK_COLORS = {
        "Low":       "#52b788",
        "Moderate":  "#f9c74f",
        "High":      "#f8961e",
        "Very High": "#e74c3c",
    }

    grid = data["grid"]
    grid_filtered = grid[
        (grid["scenario"] == selected_scenario)
        & (grid["period"] == str(selected_year))
        & (grid["region"].isin(selected_regions))
    ].copy()

    if not grid_filtered.empty:
        ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 2])
        with ctrl1:
            map_metric = st.selectbox(
                "Metric layer",
                ["Composite Risk", "Flood", "Drought", "Extreme Heat", "Expected Loss"],
            )
        with ctrl2:
            view_mode = st.radio("View mode", ["Heatmap", "Points"], horizontal=True)
        with ctrl3:
            map_style_label = st.selectbox("Map style", list(MAP_STYLES.keys()))

        ctrl4, ctrl5 = st.columns([2, 2])
        with ctrl4:
            risk_threshold = st.slider(
                "Min composite risk threshold",
                min_value=0.0, max_value=1.0, value=0.0, step=0.05,
                help="Hide grid cells below this composite risk level",
            )
        with ctrl5:
            if view_mode == "Heatmap":
                heat_radius = st.slider("Heatmap blur radius", min_value=5, max_value=40, value=18, step=1)
            else:
                point_size = st.slider("Point size", min_value=4, max_value=20, value=8, step=1)

        metric_map = {
            "Composite Risk": "composite_prob",
            "Flood":          "flood_prob",
            "Drought":        "drought_prob",
            "Extreme Heat":   "heat_prob",
            "Expected Loss":  "loss_ratio",
        }
        color_col = metric_map[map_metric]
        mapbox_style = MAP_STYLES[map_style_label]

        grid_filtered = grid_filtered[grid_filtered["composite_prob"] >= risk_threshold].copy()

        if len(selected_regions) == 1:
            region_center = REGION_CENTERS.get(selected_regions[0], {"lat": 20, "lon": 20, "zoom": 3})
            map_center = {"lat": region_center["lat"], "lon": region_center["lon"]}
            map_zoom = region_center["zoom"]
        else:
            map_center = {"lat": 20, "lon": 20}
            map_zoom = 2

        if len(grid_filtered) > 4000:
            grid_sample = grid_filtered.sample(4000, random_state=42)
        else:
            grid_sample = grid_filtered.copy()

        range_max = 1.0 if color_col != "loss_ratio" else float(grid_sample["loss_ratio"].quantile(0.95))

        map_col1, map_col2 = st.columns([3, 1])

        with map_col1:
            if view_mode == "Heatmap":
                fig_map = px.density_mapbox(
                    grid_sample,
                    lat="lat", lon="lon",
                    z=color_col,
                    radius=heat_radius,
                    color_continuous_scale="RdYlGn_r",
                    range_color=[0, range_max],
                    hover_data={
                        "region": True,
                        "composite_prob": ":.1%",
                        "flood_prob": ":.1%",
                        "drought_prob": ":.1%",
                        "heat_prob": ":.1%",
                        "lat": False,
                        "lon": False,
                    },
                    mapbox_style=mapbox_style,
                    zoom=map_zoom,
                    center=map_center,
                    height=540,
                )
            else:
                grid_sample["Risk Category"] = grid_sample["risk_category"]
                grid_sample["Flood %"] = (grid_sample["flood_prob"] * 100).round(1)
                grid_sample["Drought %"] = (grid_sample["drought_prob"] * 100).round(1)
                grid_sample["Heat %"] = (grid_sample["heat_prob"] * 100).round(1)
                grid_sample["Composite %"] = (grid_sample["composite_prob"] * 100).round(1)
                grid_sample["Prod. Value"] = grid_sample["total_production_value"].apply(fmt_usd)
                grid_sample["Exp. Loss"] = grid_sample["total_expected_loss"].apply(fmt_usd)

                fig_map = px.scatter_mapbox(
                    grid_sample,
                    lat="lat", lon="lon",
                    color=color_col,
                    color_continuous_scale="RdYlGn_r",
                    range_color=[0, range_max],
                    size=[point_size] * len(grid_sample),
                    size_max=point_size,
                    hover_name="region",
                    hover_data={
                        "Risk Category": True,
                        "Composite %": True,
                        "Flood %": True,
                        "Drought %": True,
                        "Heat %": True,
                        "Prod. Value": True,
                        "Exp. Loss": True,
                        "lat": False,
                        "lon": False,
                        color_col: False,
                    },
                    mapbox_style=mapbox_style,
                    zoom=map_zoom,
                    center=map_center,
                    height=540,
                )

            fig_map.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                coloraxis_colorbar=dict(
                    title=map_metric,
                    thickness=15,
                    tickformat=".0%" if color_col != "loss_ratio" else ".1f",
                ),
            )
            st.plotly_chart(fig_map, use_container_width=True)

            if grid_filtered.empty:
                st.info("No grid cells meet the current risk threshold. Lower the slider to see more data.")

        with map_col2:
            st.markdown("**Regional Summary**")
            HAZARD_ICONS = {"Flood": "💧", "Drought": "🌵", "Heat": "🌡️"}
            for region_key in selected_regions:
                rg = grid_filtered[grid_filtered["region"] == region_key]
                if rg.empty:
                    st.markdown(f"*{REGION_LABELS[region_key]}* — no cells above threshold")
                    continue
                avg_risk = rg["composite_prob"].mean()
                avg_loss = rg["total_expected_loss"].sum()
                top_cat = rg["risk_category"].value_counts().idxmax()
                color_dot = RISK_COLORS.get(top_cat, "#999")
                hazard_means = {
                    "Flood":   rg["flood_prob"].mean(),
                    "Drought": rg["drought_prob"].mean(),
                    "Heat":    rg["heat_prob"].mean(),
                }
                dom_hazard = max(hazard_means, key=hazard_means.get)
                st.markdown(f"""
                **{REGION_LABELS[region_key]}**
                - Avg risk: `{avg_risk:.1%}`
                - Total EAL: `{fmt_usd(avg_loss)}`
                - Dominant risk: <span style='color:{color_dot}'>**{top_cat}**</span>
                - Dominant hazard: **{HAZARD_ICONS[dom_hazard]} {dom_hazard}** (`{hazard_means[dom_hazard]:.1%}`)
                - Grid cells: `{len(rg)}`
                """, unsafe_allow_html=True)
                st.divider()

    # ── Scenario Comparison ───────────────────────────────────────────
    st.markdown('<div class="section-header">Scenario Comparison</div>', unsafe_allow_html=True)

    sc_col1, sc_col2 = st.columns(2)

    with sc_col1:
        risk_df = data["risk_scores"]
        risk_future = risk_df[
            (risk_df["scenario"] != "historical")
            & (risk_df["region"].isin(selected_regions))
        ].copy()

        risk_agg = risk_future.groupby(["scenario", "period"]).agg(
            total_eal=("total_eal", "sum"),
            avg_risk_score=("risk_score", "mean"),
        ).reset_index()
        risk_agg["period"] = risk_agg["period"].astype(int)

        fig_eal = px.line(
            risk_agg, x="period", y="total_eal",
            color="scenario",
            color_discrete_map=SCENARIO_COLORS,
            markers=True,
            labels={"total_eal": "Expected Annual Loss (USD)", "period": "Year", "scenario": "Scenario"},
            title="Expected Annual Loss Trajectory",
        )
        fig_eal.update_layout(
            template="plotly_white",
            plot_bgcolor="rgba(240,249,240,0.5)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=400,
            yaxis_tickprefix="$",
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_eal, use_container_width=True)

    with sc_col2:
        fig_risk = px.line(
            risk_agg, x="period", y="avg_risk_score",
            color="scenario",
            color_discrete_map=SCENARIO_COLORS,
            markers=True,
            labels={"avg_risk_score": "Average Risk Score", "period": "Year", "scenario": "Scenario"},
            title="Risk Score Trajectory",
        )
        fig_risk.update_layout(
            template="plotly_white",
            plot_bgcolor="rgba(240,249,240,0.5)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=400,
            yaxis_range=[0, 100],
            legend=dict(orientation="h", y=-0.15),
        )
        fig_risk.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.08, line_width=0, annotation_text="Very High Risk", annotation_position="top left")
        fig_risk.add_hrect(y0=50, y1=70, fillcolor="orange", opacity=0.06, line_width=0)
        st.plotly_chart(fig_risk, use_container_width=True)

    # ── Regional Hazard Breakdown ─────────────────────────────────────
    st.markdown('<div class="section-header">Regional Hazard Breakdown</div>', unsafe_allow_html=True)

    eal_df = data["eal"]
    eal_filtered = eal_df[
        (eal_df["scenario"] == selected_scenario)
        & (eal_df["period"] == str(selected_year))
        & (eal_df["region"].isin(selected_regions))
    ].copy()

    if not eal_filtered.empty:
        hz_col1, hz_col2 = st.columns(2)

        with hz_col1:
            eal_melt = eal_filtered.melt(
                id_vars=["region", "crop"],
                value_vars=["flood_eal", "drought_eal", "heat_eal"],
                var_name="hazard", value_name="eal_value",
            )
            eal_melt["hazard"] = eal_melt["hazard"].map({
                "flood_eal": "Flood",
                "drought_eal": "Drought",
                "heat_eal": "Extreme Heat",
            })
            eal_by_region = eal_melt.groupby(["region", "hazard"])["eal_value"].sum().reset_index()
            eal_by_region["region_label"] = eal_by_region["region"].map(lambda r: REGION_LABELS.get(r, r))

            fig_hz = px.bar(
                eal_by_region, x="region_label", y="eal_value",
                color="hazard",
                color_discrete_map={"Flood": "#3498db", "Drought": "#e67e22", "Extreme Heat": "#e74c3c"},
                labels={"eal_value": "Expected Annual Loss (USD)", "region_label": "Region", "hazard": "Hazard"},
                title="EAL by Hazard Type per Region",
                barmode="stack",
            )
            fig_hz.update_layout(
                template="plotly_white",
                plot_bgcolor="rgba(240,249,240,0.5)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=400,
                yaxis_tickprefix="$",
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_hz, use_container_width=True)

        with hz_col2:
            crop_eal = eal_filtered.groupby("crop")["total_eal"].sum().reset_index()
            crop_eal = crop_eal.sort_values("total_eal", ascending=True)

            fig_crop = px.bar(
                crop_eal, x="total_eal", y="crop",
                orientation="h",
                color="total_eal",
                color_continuous_scale="YlOrRd",
                labels={"total_eal": "Expected Annual Loss (USD)", "crop": "Crop"},
                title="EAL by Crop Type",
            )
            fig_crop.update_layout(
                template="plotly_white",
                plot_bgcolor="rgba(240,249,240,0.5)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=400,
                xaxis_tickprefix="$",
                showlegend=False,
            )
            st.plotly_chart(fig_crop, use_container_width=True)

    # ── Delta Analysis ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Change from Baseline</div>', unsafe_allow_html=True)

    delta_df = data["delta"]
    delta_filtered = delta_df[
        (delta_df["scenario"] == selected_scenario)
        & (delta_df["region"].isin(selected_regions))
    ].copy()

    if not delta_filtered.empty:
        delta_filtered["region_label"] = delta_filtered["region"].map(lambda r: REGION_LABELS.get(r, r))
        delta_filtered["period"] = delta_filtered["period"].astype(int)

        d_col1, d_col2 = st.columns(2)

        with d_col1:
            fig_delta_eal = px.bar(
                delta_filtered, x="period", y="eal_change_pct",
                color="region_label",
                barmode="group",
                labels={"eal_change_pct": "EAL Change (%)", "period": "Year", "region_label": "Region"},
                title=f"EAL Change from Baseline — {selected_scenario}",
                color_discrete_sequence=["#3498db", "#2ecc71", "#e74c3c"],
            )
            fig_delta_eal.update_layout(
                template="plotly_white",
                plot_bgcolor="rgba(240,249,240,0.5)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=380,
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_delta_eal, use_container_width=True)

        with d_col2:
            fig_delta_score = px.bar(
                delta_filtered, x="period", y="risk_score_change",
                color="region_label",
                barmode="group",
                labels={"risk_score_change": "Risk Score Change", "period": "Year", "region_label": "Region"},
                title=f"Risk Score Change from Baseline — {selected_scenario}",
                color_discrete_sequence=["#3498db", "#2ecc71", "#e74c3c"],
            )
            fig_delta_score.update_layout(
                template="plotly_white",
                plot_bgcolor="rgba(240,249,240,0.5)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=380,
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_delta_score, use_container_width=True)

    # ── Detailed Risk Table ──────────────────────────────────────────
    st.markdown('<div class="section-header">Detailed Risk Scores</div>', unsafe_allow_html=True)

    risk_table = data["risk_scores"][
        data["risk_scores"]["region"].isin(selected_regions)
    ].copy()
    risk_table["region_label"] = risk_table["region"].map(lambda r: REGION_LABELS.get(r, r))

    display_cols = ["region_label", "scenario", "period", "risk_score", "risk_rating",
                    "total_eal", "loss_ratio", "mean_composite_prob"]
    risk_display = risk_table[display_cols].copy()
    risk_display.columns = ["Region", "Scenario", "Period", "Risk Score", "Rating",
                             "Expected Annual Loss", "Loss Ratio", "Composite Prob"]
    risk_display["Expected Annual Loss"] = risk_display["Expected Annual Loss"].apply(fmt_usd)
    risk_display["Loss Ratio"] = (risk_display["Loss Ratio"] * 100).round(1).astype(str) + "%"
    risk_display["Composite Prob"] = (risk_display["Composite Prob"] * 100).round(1).astype(str) + "%"

    st.dataframe(
        risk_display.sort_values(["Region", "Scenario", "Period"]),
        use_container_width=True,
        height=400,
        hide_index=True,
    )

    # ── Methodology footer ───────────────────────────────────────────
    with st.expander("📋 Methodology & Data Sources"):
        st.markdown("""
        ### Methodology

        **Hazard Modelling:**
        - **Floods**: GEV (Generalized Extreme Value) distribution fitted to annual maximum precipitation series.
          Return periods of 5, 10, 25, 50, and 100 years computed.
        - **Drought**: Consecutive Dry Days (CDD) mapped to exceedance probability via logistic transformation.
          Severity levels: moderate (30d), severe (60d), extreme (90d).
        - **Extreme Heat**: Fraction of growing season exceeding crop-specific thermal thresholds.

        **Financial Risk:**
        - **Expected Annual Loss (EAL)** = Σ (Hazard Probability × Exposure Value × Vulnerability Factor)
        - Vulnerability factors are crop-specific damage functions
        - Exposure values from FAO agricultural production statistics

        **Climate Scenarios (CMIP6/SSP):**
        - **SSP1-2.6**: Sustainability pathway, +1.5°C by 2100
        - **SSP2-4.5**: Middle of the road, +2.7°C by 2100
        - **SSP5-8.5**: Fossil-fuelled development, +4.3°C by 2100

        ### Data Sources
        - Copernicus Climate Data Store (CMIP6 projections)
        - WRI Aqueduct (flood hazard maps)
        - MapSPAM (global crop production data)
        - FAO (agricultural production statistics)
        """)


# ──────────────────────────────────────────────────────────────────────
# TAB 2 — TCFD Disclosure Extractor
# ──────────────────────────────────────────────────────────────────────
with tab_tcfd:

    # ── NL Banks catalogue ─────────────────────────────────────────────
    NL_BANKS = {
        "— Select a bank —": None,
        "ING Group": {
            "full_name": "ING Groep N.V.",
            "hq": "Amsterdam",
            "type": "Universal bank (retail, wholesale, insurance)",
            "assets": "~€967 billion total assets",
            "notes": "Listed on Euronext Amsterdam. Major climate commitments including Paris Agreement alignment and net-zero ambition by 2050. Publishes annual Climate Report alongside Annual Report.",
        },
        "ABN AMRO": {
            "full_name": "ABN AMRO Bank N.V.",
            "hq": "Amsterdam",
            "type": "Universal bank (retail, private, corporate)",
            "assets": "~€413 billion total assets",
            "notes": "State-owned (Dutch government holds majority stake). Signatory to Net-Zero Banking Alliance. Publishes integrated annual report with dedicated climate/ESG sections.",
        },
        "Rabobank": {
            "full_name": "Coöperatieve Rabobank U.A.",
            "hq": "Utrecht",
            "type": "Cooperative bank (food & agri focus)",
            "assets": "~€674 billion total assets",
            "notes": "World's leading food & agriculture bank. Strong focus on transition finance for agri sector. Publishes annual Sustainability Report aligned with GRI and TCFD.",
        },
        "de Volksbank": {
            "full_name": "de Volksbank N.V.",
            "hq": "Utrecht",
            "type": "Retail bank (social mandate)",
            "assets": "~€74 billion total assets",
            "notes": "State-owned. Brands include SNS, ASN Bank, RegioBank, BLG Wonen. Mission-driven with explicit social and environmental targets. TCFD reporting embedded in annual report.",
        },
        "Triodos Bank": {
            "full_name": "Triodos Bank N.V.",
            "hq": "Zeist",
            "type": "Sustainable/ethical bank",
            "assets": "~€22 billion total assets",
            "notes": "Pioneer in sustainable banking. Only finances companies/projects with positive environmental or social impact. Comprehensive impact reporting aligned with TCFD and SDGs.",
        },
        "NIBC Bank": {
            "full_name": "NIBC Bank N.V.",
            "hq": "The Hague",
            "type": "Mid-market corporate & retail bank",
            "assets": "~€26 billion total assets",
            "notes": "Focuses on mid-market clients and specific sectors (energy, real estate, food). Growing focus on sustainable finance and ESG integration in credit risk.",
        },
        "BNG Bank": {
            "full_name": "Bank Nederlandse Gemeenten N.V.",
            "hq": "The Hague",
            "type": "Public sector bank",
            "assets": "~€165 billion total assets",
            "notes": "Finances Dutch municipalities, provinces, housing corporations, and healthcare. Majority owned by the Dutch State. Strong focus on sustainable lending and green bonds.",
        },
        "NWB Bank": {
            "full_name": "Nederlandse Waterschapsbank N.V.",
            "hq": "The Hague",
            "type": "Water authority bank",
            "assets": "~€107 billion total assets",
            "notes": "Exclusively finances Dutch water authorities and social housing. High ESG relevance given climate-adaptation mandate. Regular issuer of sustainability bonds.",
        },
        "Van Lanschot Kempen": {
            "full_name": "Van Lanschot Kempen N.V.",
            "hq": "Amsterdam / 's-Hertogenbosch",
            "type": "Wealth management & investment bank",
            "assets": "~€32 billion client assets under management",
            "notes": "Oldest independent bank in the Netherlands (founded 1737). Focus on wealth management and sustainable investing. Publishes annual sustainability report with TCFD disclosures.",
        },
        "ASN Bank": {
            "full_name": "ASN Bank (brand of de Volksbank)",
            "hq": "The Hague",
            "type": "Sustainable retail bank",
            "assets": "Part of de Volksbank (~€74B group)",
            "notes": "Pioneer Dutch sustainable savings/investment bank. Climate-neutral since 2018. Aligns entire portfolio with 1.5°C pathway. Detailed annual impact report.",
        },
        "Achmea Bank": {
            "full_name": "Achmea Bank N.V.",
            "hq": "Tilburg",
            "type": "Insurance-linked savings & mortgage bank",
            "assets": "~€20 billion total assets",
            "notes": "Part of Achmea insurance group. Focuses on mortgage lending and savings. Climate risk embedded via group-level ESG policy and SFDR disclosures.",
        },
        "Bunq": {
            "full_name": "Bunq B.V.",
            "hq": "Amsterdam",
            "type": "Neobank / digital bank",
            "assets": "~€8 billion total assets",
            "notes": "Europe's second-largest neobank. Markets itself as 'the bank of The Free' with environmental angle — plants trees per transaction. Growing ESG and climate reporting.",
        },
    }

    TCFD_PILLARS = {
        "Governance": {
            "icon": "🏛️",
            "color": "#2d6a4f",
            "description": "Board oversight and management roles in climate risk",
            "recommended_disclosures": [
                "Board's oversight of climate-related risks and opportunities",
                "Management's role in assessing and managing climate-related risks and opportunities",
            ],
        },
        "Strategy": {
            "icon": "🎯",
            "color": "#1a759f",
            "description": "Climate risks/opportunities and their impact on business strategy",
            "recommended_disclosures": [
                "Short, medium, and long-term climate-related risks and opportunities identified",
                "Impact on the organisation's businesses, strategy, and financial planning",
                "Resilience of strategy under different climate scenarios",
            ],
        },
        "Risk Management": {
            "icon": "⚠️",
            "color": "#e67e22",
            "description": "How climate risks are identified, assessed, and managed",
            "recommended_disclosures": [
                "Processes for identifying and assessing climate-related risks",
                "Processes for managing climate-related risks",
                "Integration into overall enterprise risk management",
            ],
        },
        "Metrics & Targets": {
            "icon": "📏",
            "color": "#8b1a1a",
            "description": "Metrics, GHG emissions data, and targets",
            "recommended_disclosures": [
                "Metrics used to assess climate-related risks and opportunities",
                "Scope 1, 2, and 3 greenhouse gas emissions",
                "Targets used to manage climate risks and performance against targets",
            ],
        },
    }

    # ── Header ─────────────────────────────────────────────────────────
    st.markdown("## 📋 TCFD Disclosure Extractor")
    st.markdown(
        "Select a Dutch bank to extract **TCFD-aligned climate disclosures** synthesised from "
        "publicly available sustainability reports, climate reports, and annual filings. "
        "Powered by OpenAI."
    )

    with st.expander("ℹ️ About the TCFD Framework"):
        st.markdown("""
        The **Task Force on Climate-related Financial Disclosures (TCFD)**, established by the
        Financial Stability Board, provides a framework for companies to disclose climate-related
        risks and opportunities across four pillars:

        | Pillar | Focus |
        |--------|-------|
        | 🏛️ **Governance** | Board and management oversight of climate risks |
        | 🎯 **Strategy** | Impact of climate risks/opportunities on business strategy |
        | ⚠️ **Risk Management** | How climate risks are identified, assessed, managed |
        | 📏 **Metrics & Targets** | GHG emissions, climate KPIs, and reduction targets |

        Since 2023, TCFD disclosures are mandatory for large Dutch financial institutions under
        DNB/AFM regulations and EU CSRD requirements.
        """)

    st.markdown("---")

    # ── API key ────────────────────────────────────────────────────────
    api_key = ""
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Enter your OpenAI API key. Get one at platform.openai.com/api-keys",
        )

    # ── Bank selector ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Bank Selection</div>', unsafe_allow_html=True)

    selected_bank = st.selectbox(
        "Select a Dutch bank",
        options=list(NL_BANKS.keys()),
        index=0,
    )

    bank_info = NL_BANKS[selected_bank]

    if bank_info:
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            st.markdown(f"**Type:** {bank_info['type']}")
        with bc2:
            st.markdown(f"**HQ:** {bank_info['hq']}")
        with bc3:
            st.markdown(f"**Scale:** {bank_info['assets']}")
        st.caption(bank_info["notes"])

    st.markdown("")

    extract_btn = st.button(
        f"🔍 Extract TCFD Disclosures — {selected_bank}" if bank_info else "Select a bank above",
        disabled=(not bank_info or not api_key),
        type="primary",
        use_container_width=True,
    )

    # ── Extraction ─────────────────────────────────────────────────────
    if extract_btn and bank_info and api_key:

        prompt = f"""You are an expert climate finance analyst specialising in TCFD (Task Force on Climate-related Financial Disclosures) reporting and the Dutch banking sector. You have comprehensive knowledge of sustainability reports, climate reports, and annual filings published by Dutch financial institutions.

Extract and synthesise TCFD-aligned climate disclosures for **{selected_bank}** ({bank_info['full_name']}).

Bank context: {bank_info['notes']}

Structure your response with the following four TCFD pillars. Under each pillar, address ALL recommended disclosures precisely. Be specific, factual, and grounded in known public information from this bank's sustainability/climate reporting. Where exact figures are available in their public reports, include them. Where data is not publicly disclosed, explicitly state "Not publicly disclosed" or "Partially disclosed" and explain what IS known.

---

## GOVERNANCE

**a) Board oversight of climate-related risks and opportunities**
Describe the board-level structure, committees, and oversight mechanisms for climate risk.

**b) Management's role in assessing and managing climate-related risks and opportunities**
Describe executive roles (CRO, CSO, etc.), management committees, and processes for climate risk.

---

## STRATEGY

**a) Climate-related risks and opportunities identified**
List specific physical risks (acute/chronic) and transition risks (policy, technology, market, reputational) and opportunities, across short (0–3y), medium (3–10y), and long-term (10y+) horizons.

**b) Impact on business, strategy, and financial planning**
How do identified climate risks and opportunities affect the bank's lending portfolio, investment decisions, product offerings, and financial planning?

**c) Resilience of strategy under climate scenarios**
Which climate scenarios does the bank use (e.g., IEA NZE, NGFS, RCP scenarios)? What are the results of scenario analysis on the portfolio?

---

## RISK MANAGEMENT

**a) Process for identifying and assessing climate-related risks**
Describe the methodology, tools, and data sources used for physical and transition risk assessment.

**b) Process for managing climate-related risks**
Describe risk appetite, limits, credit screening, engagement, and exclusion policies.

**c) Integration into overall enterprise risk management**
How are climate risks integrated into the bank's overall risk framework (ICAAP, credit risk, operational risk)?

---

## METRICS & TARGETS

**a) Metrics used to assess climate-related risks and opportunities**
List specific KPIs: financed emissions intensity, green asset ratio, climate VaR, PCAF scores, etc.

**b) Scope 1, 2, and 3 GHG emissions**
Provide the most recent available Scope 1, 2, and 3 (financed emissions) data with units and reference year. Include PCAF data quality scores if disclosed.

**c) Targets and performance**
List specific net-zero targets, intermediate milestones, sectoral decarbonisation targets, and progress reported against them.

---

Close with a brief **Summary Assessment** (3–4 sentences) rating the overall maturity and completeness of this bank's TCFD disclosures compared to Dutch banking sector peers."""

        with st.spinner(f"Extracting TCFD disclosures for {selected_bank}…"):
            try:
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                tcfd_text = response.choices[0].message.content
                st.session_state[f"tcfd_{selected_bank}"] = tcfd_text

            except Exception as e:
                if "auth" in str(e).lower() or "api key" in str(e).lower():
                    st.error("Invalid API key. Please check your OpenAI API key and try again.")
                else:
                    st.error(f"Extraction failed: {e}")
                st.stop()

    # ── Display results ────────────────────────────────────────────────
    cache_key = f"tcfd_{selected_bank}"
    if cache_key in st.session_state and bank_info:
        tcfd_text = st.session_state[cache_key]

        st.markdown('<div class="section-header">TCFD Disclosure Results</div>', unsafe_allow_html=True)
        st.markdown(f"#### {selected_bank} — TCFD Climate Disclosure Summary")

        # Split into pillar tabs
        pillar_tabs = st.tabs([
            f"{v['icon']} {k}" for k, v in TCFD_PILLARS.items()
        ])

        pillar_names = list(TCFD_PILLARS.keys())
        split_markers = ["## GOVERNANCE", "## STRATEGY", "## RISK MANAGEMENT", "## METRICS & TARGETS"]

        # Parse sections from the full text
        sections = {}
        for i, marker in enumerate(split_markers):
            start = tcfd_text.find(marker)
            if start == -1:
                sections[pillar_names[i]] = "_Section not found in response._"
                continue
            end = tcfd_text.find(split_markers[i + 1]) if i + 1 < len(split_markers) else len(tcfd_text)
            sections[pillar_names[i]] = tcfd_text[start:end].strip()

        for i, (pillar_name, pillar_meta) in enumerate(TCFD_PILLARS.items()):
            with pillar_tabs[i]:
                st.markdown(
                    f"<div class='tcfd-badge' style='background:{pillar_meta['color']};'>"
                    f"{pillar_meta['icon']} {pillar_name}</div>",
                    unsafe_allow_html=True,
                )
                st.caption(pillar_meta["description"])
                st.markdown("**TCFD Recommended Disclosures:**")
                for rd in pillar_meta["recommended_disclosures"]:
                    st.markdown(f"- {rd}")
                st.markdown("---")
                st.markdown(sections.get(pillar_name, "_Not available._"))

        # Summary section
        summary_start = tcfd_text.find("Summary Assessment")
        if summary_start != -1:
            st.markdown("---")
            st.markdown("### Overall Assessment")
            st.info(tcfd_text[summary_start:].replace("**Summary Assessment**", "").strip())

        # Download button
        st.markdown("---")
        st.download_button(
            label="⬇️ Download Full TCFD Report (Markdown)",
            data=f"# TCFD Disclosure Report — {selected_bank}\n\n{tcfd_text}",
            file_name=f"TCFD_{selected_bank.replace(' ', '_')}_disclosure.md",
            mime="text/markdown",
            use_container_width=True,
        )

        st.markdown(
            "<div class='disclaimer-box'>"
            "⚠️ <b>Disclaimer:</b> These disclosures are AI-synthesised from publicly available "
            "sustainability reports and annual filings. They are intended for research and reference "
            "purposes only. Always verify against official bank publications. This is not financial "
            "or legal advice."
            "</div>",
            unsafe_allow_html=True,
        )

    elif bank_info and not api_key:
        st.warning("Please enter your Anthropic API key above to extract disclosures.")
    elif not bank_info:
        # Show pillar overview when no bank is selected
        st.markdown('<div class="section-header">TCFD Framework Overview</div>', unsafe_allow_html=True)
        ov_cols = st.columns(4)
        for i, (pillar_name, meta) in enumerate(TCFD_PILLARS.items()):
            with ov_cols[i]:
                st.markdown(
                    f"<div class='tcfd-pillar-card'>"
                    f"<div style='font-size:28px;'>{meta['icon']}</div>"
                    f"<div style='font-weight:700; color:{meta['color']}; margin:6px 0;'>{pillar_name}</div>"
                    f"<div style='font-size:12px; color:#52796f;'>{meta['description']}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ── Footer ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#52796f; font-size:12px;'>"
    "Climate Risk Geospatial Assessment · Kehinde Damilola Akindele"
    "</div>",
    unsafe_allow_html=True,
)
