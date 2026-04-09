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
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px 24px;
        color: white;
        margin-bottom: 8px;
    }
    .metric-card .label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #8892b0;
        margin-bottom: 4px;
    }
    .metric-card .value {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .metric-card .delta {
        font-size: 13px;
        color: #ff6b6b;
    }
    .metric-card .delta.positive { color: #51cf66; }

    .section-header {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #64748b;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 8px;
        margin: 32px 0 16px 0;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    div[data-testid="stSidebar"] .stMarkdown { color: #cbd5e1; }
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
    <div style='font-size:11px; color:#64748b;'>
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
# MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════

st.markdown("# 🌍 Climate Risk Geospatial Assessment")
st.markdown(f"**Scenario:** {SCENARIO_LABELS[selected_scenario]} · **Horizon:** {selected_year} · **Regions:** {len(selected_regions)}")

# ── Key Metrics Row ────────────────────────────────────────────────────
st.markdown('<div class="section-header">Portfolio Overview</div>', unsafe_allow_html=True)

port = data["portfolio"]
current = port[(port["scenario"] == selected_scenario) & (port["period"] == str(selected_year))]
baseline = port[port["scenario"] == "SSP1-2.6"]  # use lowest scenario as reference

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
        # Compare to SSP1-2.6 same year
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


# ── Risk Map ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Geospatial Risk Map</div>', unsafe_allow_html=True)

grid = data["grid"]
grid_filtered = grid[
    (grid["scenario"] == selected_scenario)
    & (grid["period"] == str(selected_year))
    & (grid["region"].isin(selected_regions))
].copy()

if not grid_filtered.empty:
    map_col1, map_col2 = st.columns([3, 1])

    with map_col1:
        map_metric = st.radio(
            "Map layer",
            ["Composite Risk", "Flood", "Drought", "Extreme Heat", "Expected Loss"],
            horizontal=True,
        )

        metric_map = {
            "Composite Risk": "composite_prob",
            "Flood": "flood_prob",
            "Drought": "drought_prob",
            "Extreme Heat": "heat_prob",
            "Expected Loss": "loss_ratio",
        }
        color_col = metric_map[map_metric]

        # Subsample for performance
        if len(grid_filtered) > 3000:
            grid_sample = grid_filtered.sample(3000, random_state=42)
        else:
            grid_sample = grid_filtered

        fig_map = px.scatter_mapbox(
            grid_sample,
            lat="lat", lon="lon",
            color=color_col,
            color_continuous_scale="YlOrRd",
            range_color=[0, 1] if color_col != "loss_ratio" else [0, grid_sample["loss_ratio"].quantile(0.95)],
            size_max=8,
            size=[6] * len(grid_sample),
            hover_data=["region", "composite_prob", "flood_prob", "drought_prob", "heat_prob"],
            mapbox_style="carto-darkmatter",
            zoom=2,
            center={"lat": 20, "lon": 20},
            height=500,
        )
        fig_map.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_colorbar=dict(title=map_metric, thickness=15),
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with map_col2:
        st.markdown("**Regional Summary**")
        for region_key in selected_regions:
            rg = grid_filtered[grid_filtered["region"] == region_key]
            if rg.empty:
                continue
            avg_risk = rg["composite_prob"].mean()
            avg_loss = rg["total_expected_loss"].sum()
            st.markdown(f"""
            **{REGION_LABELS[region_key]}**
            - Avg risk: `{avg_risk:.2%}`
            - Total EAL: `{fmt_usd(avg_loss)}`
            - Grid cells: `{len(rg)}`
            """)


# ── Scenario Comparison ───────────────────────────────────────────────
st.markdown('<div class="section-header">Scenario Comparison</div>', unsafe_allow_html=True)

sc_col1, sc_col2 = st.columns(2)

with sc_col1:
    # EAL trajectory across time horizons
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
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=400,
        yaxis_tickprefix="$",
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_eal, use_container_width=True)

with sc_col2:
    # Risk score trajectory
    fig_risk = px.line(
        risk_agg, x="period", y="avg_risk_score",
        color="scenario",
        color_discrete_map=SCENARIO_COLORS,
        markers=True,
        labels={"avg_risk_score": "Average Risk Score", "period": "Year", "scenario": "Scenario"},
        title="Risk Score Trajectory",
    )
    fig_risk.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=400,
        yaxis_range=[0, 100],
        legend=dict(orientation="h", y=-0.15),
    )
    fig_risk.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.08, line_width=0, annotation_text="Very High Risk", annotation_position="top left")
    fig_risk.add_hrect(y0=50, y1=70, fillcolor="orange", opacity=0.06, line_width=0)
    st.plotly_chart(fig_risk, use_container_width=True)


# ── Regional Hazard Breakdown ─────────────────────────────────────────
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
        # Stacked bar: EAL by hazard type per region
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
        eal_by_region["region_label"] = eal_by_region["region"].map(
            lambda r: REGION_LABELS.get(r, r)
        )

        fig_hz = px.bar(
            eal_by_region, x="region_label", y="eal_value",
            color="hazard",
            color_discrete_map={"Flood": "#3498db", "Drought": "#e67e22", "Extreme Heat": "#e74c3c"},
            labels={"eal_value": "Expected Annual Loss (USD)", "region_label": "Region", "hazard": "Hazard"},
            title="EAL by Hazard Type per Region",
            barmode="stack",
        )
        fig_hz.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=400,
            yaxis_tickprefix="$",
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_hz, use_container_width=True)

    with hz_col2:
        # Crop-level breakdown
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
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=400,
            xaxis_tickprefix="$",
            showlegend=False,
        )
        st.plotly_chart(fig_crop, use_container_width=True)


# ── Delta Analysis ────────────────────────────────────────────────────
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
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
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
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=380,
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_delta_score, use_container_width=True)


# ── Detailed Risk Table ──────────────────────────────────────────────
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


# ── Methodology footer ───────────────────────────────────────────────
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

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#64748b; font-size:12px;'>"
    "Climate Risk Geospatial Assessment · Kehinde Damilola Akindele · 2026"
    "</div>",
    unsafe_allow_html=True,
)
