# Technical Methodology

## 1. Climate Data & Scenarios

This project analyzes physical climate risk under three Shared Socioeconomic Pathways (SSP) from the CMIP6 framework:

**SSP1-2.6 (Sustainability):** Assumes strong climate mitigation, reaching net-zero around 2070. Global mean temperature rise of approximately 1.5°C above pre-industrial by 2100.

**SSP2-4.5 (Middle of the Road):** Moderate mitigation efforts with continued but slowing emissions growth. Global warming of approximately 2.7°C by 2100.

**SSP5-8.5 (Fossil-Fuelled Development):** High energy demand met predominantly by fossil fuels with minimal mitigation. Warming of approximately 4.3°C by 2100.

For each scenario, projections are generated at three time horizons: 2030 (near-term), 2050 (mid-century), and 2080 (late-century), against a 1985–2014 historical baseline.

## 2. Hazard Modelling

### 2.1 Flood Hazard

Flood risk is assessed through extreme precipitation analysis using the Generalized Extreme Value (GEV) distribution. Annual maximum daily precipitation series are fitted with the GEV distribution parameterized by location (μ), scale (σ), and shape (ξ) parameters.

Return period thresholds are computed for 5, 10, 25, 50, and 100-year events. For future scenarios, GEV parameters are adjusted based on projected changes in extreme precipitation intensity.

The exceedance probability for a given return period T is: P(exceed) = 1/T

### 2.2 Drought Hazard

Drought probability is derived from Consecutive Dry Days (CDD), which represents the longest annual sequence of days with precipitation below 1mm. The CDD metric is mapped to exceedance probability using a logistic transformation:

P(drought) = 1 / (1 + exp(-(CDD - threshold) / (threshold × 0.3)))

Three severity levels are defined: moderate (30 days), severe (60 days), and extreme (90 days).

### 2.3 Extreme Heat Hazard

Heat stress probability is computed as the fraction of the growing season (180 days) during which temperatures exceed crop-specific thermal thresholds. These thresholds reflect the temperature above which significant yield reductions begin:

- Wheat: 32°C
- Soy: 35°C
- Rice: 35°C
- Maize: 37°C

### 2.4 Composite Hazard

The composite multi-hazard probability combines individual hazard probabilities under the assumption of independent occurrence:

P(composite) = 1 - (1 - P_flood) × (1 - P_drought) × (1 - P_heat)

## 3. Financial Risk Quantification

### 3.1 Exposure

Agricultural exposure is defined as the annual production value (USD) of crops within each grid cell, sourced from FAO production statistics and MapSPAM spatial crop allocation data.

### 3.2 Vulnerability

Vulnerability factors represent the fraction of crop value lost per unit of hazard occurrence. These are crop-specific and hazard-specific, calibrated from published agricultural damage studies.

### 3.3 Expected Annual Loss (EAL)

EAL is the primary financial risk metric, computed as:

EAL = Σ_h (P_h × V_h × E)

Where:
- P_h = annual exceedance probability for hazard h
- V_h = vulnerability factor for hazard h and crop type
- E = exposure value (production value in USD)

### 3.4 Risk Score

A normalized composite risk score (0–100) combines hazard probability and loss ratio:

Risk Score = 0.4 × (Composite Probability × 100) + 0.6 × (Loss Ratio × 100)

Risk ratings: Very Low (0–15), Low (15–30), Medium (30–50), High (50–70), Very High (70–100).

## 4. Study Regions

**Mato Grosso, Brazil:** Major soy, maize, and cotton producing region. Primary hazards are drought and extreme heat, which affect growing season length and crop yields. Total agricultural production value: ~$28.5B.

**Rhine-Meuse Delta, Netherlands:** Intensive dairy and arable farming in a low-lying river delta. Primary hazards are river flooding (exacerbated by sea-level rise) and drought during summer growing seasons. Total agricultural production value: ~$9.2B.

**Punjab, India:** Critical wheat and rice producing region feeding a large population. Primary hazards are extreme heat (especially during wheat filling) and monsoon variability causing both floods and droughts. Total agricultural production value: ~$12.8B.

## 5. Limitations

- Synthetic data is used to demonstrate the pipeline; production deployment requires real CMIP6 data from the Copernicus Climate Data Store
- Hazard interactions (e.g., drought followed by flood) are not modelled
- Vulnerability factors are simplified constant values; real damage functions are nonlinear
- Spatial resolution is coarser than operational risk assessment would require
- Financial exposure is based on national/regional averages, not actual portfolio data
