# 🌍 Climate Risk Geospatial Assessment

**Multi-Hazard Physical Risk Analysis for Agricultural Regions Under SSP Climate Scenarios**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://climate-risk-assessment.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This project builds a **geospatial climate risk assessment pipeline** that quantifies the physical risk of climate hazards — **floods, droughts, and extreme heat** — on agricultural regions across multiple countries. It uses climate scenario data (CMIP6/SSP projections), overlays them with geospatial crop exposure data, and produces **financial risk indicators** including Expected Annual Loss (EAL) and composite risk scores.

### Key Features

- **Climate Scenario Analysis**: Processes CMIP6 projections under SSP1-2.6, SSP2-4.5, and SSP5-8.5 scenarios across 2030–2080 time horizons
- **Multi-Hazard Modelling**: Quantifies flood, drought, and extreme heat exceedance probabilities using return period analysis (GEV/Gumbel distributions)
- **Financial Risk Quantification**: Computes Expected Annual Loss (EAL) and composite risk scores by overlaying hazard probabilities with crop exposure data
- **Interactive Dashboard**: Streamlit-based dashboard with scenario comparison maps, risk heatmaps, and regional drill-down
- **AI Report Analysis**: LLM-powered extraction of TCFD-aligned climate disclosures from sustainability reports

### Study Regions

| Region | Key Crops | Primary Hazards | Relevance |
|--------|-----------|----------------|-----------|
| Mato Grosso, Brazil | Soy, maize, cotton | Drought, heat stress | Major agri-commodity sourcing |
| Rhine-Meuse Delta, Netherlands | Dairy, arable crops | River flooding | European agriculture hub |
| Punjab, India | Wheat, rice | Extreme heat, monsoon variability | Food security critical |

## Project Structure

```
climate-risk-assessment/
├── src/
│   ├── data_generator.py        # Synthetic CMIP6-style data generation
│   ├── hazard_modelling.py      # Hazard probability modelling (GEV/Gumbel)
│   ├── exposure_analysis.py     # Crop exposure & financial impact
│   ├── risk_indicators.py       # EAL, risk scores, portfolio aggregation
│   └── config.py                # Project configuration & constants
├── dashboard/
│   └── app.py                   # Streamlit dashboard
├── scripts/
│   └── run_pipeline.py          # End-to-end pipeline runner
├── data/
│   ├── raw/                     # Raw climate projection data
│   ├── processed/               # Processed hazard layers
│   └── output/                  # Final risk indicators & results
├── docs/
│   └── methodology.md           # Technical methodology documentation
├── tests/
│   └── test_pipeline.py         # Unit tests
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Installation

```bash
git clone https://github.com/Damkez/climate-risk-assessment.git
cd climate-risk-assessment
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
python scripts/run_pipeline.py
```

This will:
1. Generate/load climate projection data for all study regions
2. Run hazard probability modelling for floods, droughts, and extreme heat
3. Compute crop exposure overlays and financial impact
4. Generate risk indicators, EAL estimates, and portfolio aggregations
5. Save all outputs to `data/output/`

### 3. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

## Methodology

### Climate Scenarios (SSP)

- **SSP1-2.6** (Sustainability): Low emissions, +1.5–2°C by 2100
- **SSP2-4.5** (Middle of the Road): Moderate emissions, +2–3°C by 2100
- **SSP5-8.5** (Fossil-Fuelled): High emissions, +4–5°C by 2100

### Hazard Modelling

For each hazard type, we compute exceedance probabilities under baseline and future scenarios:

- **Floods**: Return period analysis using Generalized Extreme Value (GEV) distributions fitted to annual maximum precipitation series
- **Drought**: Standardized Precipitation-Evapotranspiration Index (SPEI) derived from temperature and precipitation projections
- **Extreme Heat**: Frequency analysis of days exceeding crop-specific thermal thresholds

### Financial Risk Quantification

Expected Annual Loss (EAL) is computed as:

```
EAL = Σ (Hazard Probability × Exposure Value × Vulnerability Factor)
```

Where:
- **Hazard Probability**: Exceedance probability from hazard modelling
- **Exposure Value**: Crop production value (USD) from FAO/MapSPAM data
- **Vulnerability Factor**: Crop-specific damage functions relating hazard intensity to yield loss

## Tech Stack

- **Python** (GeoPandas, XArray, Rasterio, SciPy, NumPy, Pandas)
- **Streamlit** + **Plotly** + **Folium** for interactive dashboard
- **SciPy** for statistical modelling (GEV, Gumbel distributions)
- **Docker** for containerized deployment

## Data Sources

- [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu) — CMIP6 projections
- [WRI Aqueduct](https://www.wri.org/aqueduct) — Flood hazard maps
- [MapSPAM](https://www.mapspam.info) — Global crop data
- [FAO](https://www.fao.org/faostat) — Agricultural production statistics

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

**Kehinde Damilola Akindele**
Geospatial Data Analyst | Climate Risk Analytics
[LinkedIn](https://www.linkedin.com/in/damilola-akindele-1349a112b/) | [GitHub](https://github.com/Damkez)
