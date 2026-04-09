"""
Data Generator Module
=====================
Generates synthetic CMIP6-style climate projection data for study regions.
In production, this would be replaced by CDS API downloads of real CMIP6 data.
The synthetic data mirrors the statistical properties of real climate projections
to demonstrate the full pipeline.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json

from src.config import (
    REGIONS, SCENARIOS, TIME_HORIZONS, BASELINE_PERIOD,
    RAW_DIR, PROCESSED_DIR, RANDOM_SEED, HEAT_THRESHOLDS,
)


def generate_grid(region_key: str) -> pd.DataFrame:
    """Generate a regular lat/lon grid for a study region."""
    region = REGIONS[region_key]
    lon_min, lat_min, lon_max, lat_max = region["bbox"]
    res = region["grid_resolution"]

    lons = np.arange(lon_min, lon_max, res)
    lats = np.arange(lat_min, lat_max, res)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    points = []
    for lon, lat in zip(lon_grid.flatten(), lat_grid.flatten()):
        points.append({"lon": round(lon, 4), "lat": round(lat, 4)})

    df = pd.DataFrame(points)
    df["grid_id"] = range(len(df))
    df["region"] = region_key
    return df


def generate_baseline_climate(grid: pd.DataFrame, region_key: str, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate baseline (historical) climate statistics per grid cell.
    Simulates 30-year baseline means for temperature, precipitation, and
    derived hazard indicators.
    """
    rng = np.random.default_rng(seed)
    region = REGIONS[region_key]
    n = len(grid)

    # Regional baseline climate characteristics
    climate_profiles = {
        "mato_grosso": {
            "temp_mean": 26.5, "temp_std": 1.5,
            "precip_annual_mm": 1800, "precip_std": 300,
            "precip_max_daily_mean": 85, "precip_max_daily_std": 20,
            "dry_days_mean": 120, "dry_days_std": 25,
        },
        "rhine_meuse": {
            "temp_mean": 10.5, "temp_std": 1.0,
            "precip_annual_mm": 850, "precip_std": 120,
            "precip_max_daily_mean": 55, "precip_max_daily_std": 15,
            "dry_days_mean": 45, "dry_days_std": 12,
        },
        "punjab": {
            "temp_mean": 24.0, "temp_std": 2.0,
            "precip_annual_mm": 650, "precip_std": 180,
            "precip_max_daily_mean": 110, "precip_max_daily_std": 35,
            "dry_days_mean": 200, "dry_days_std": 30,
        },
    }

    cp = climate_profiles[region_key]

    # Add spatial gradient (latitude-dependent variation)
    lat_norm = (grid["lat"].values - grid["lat"].min()) / max(grid["lat"].max() - grid["lat"].min(), 0.01)

    data = {
        "grid_id": grid["grid_id"].values,
        "lon": grid["lon"].values,
        "lat": grid["lat"].values,
        "region": region_key,
        "period": "baseline",
        "scenario": "historical",

        # Temperature: mean annual temp with latitude gradient
        "temp_mean_annual_c": (
            cp["temp_mean"]
            + rng.normal(0, cp["temp_std"] * 0.3, n)
            - lat_norm * 2.0  # cooler at higher latitudes
        ),

        # Precipitation: annual total
        "precip_annual_mm": (
            cp["precip_annual_mm"]
            + rng.normal(0, cp["precip_std"], n)
            + lat_norm * 100  # slight gradient
        ),

        # Annual maximum daily precipitation (for flood analysis)
        "precip_max_daily_mm": (
            cp["precip_max_daily_mean"]
            + rng.normal(0, cp["precip_max_daily_std"], n)
        ),

        # Consecutive dry days (for drought analysis)
        "consecutive_dry_days": (
            cp["dry_days_mean"]
            + rng.normal(0, cp["dry_days_std"], n)
            - lat_norm * 15
        ),

        # Heat stress days (days > crop threshold)
        "heat_stress_days": rng.poisson(
            lam=max(5, 15 + (cp["temp_mean"] - 20) * 3), size=n
        ).astype(float),
    }

    df = pd.DataFrame(data)
    # Ensure physical bounds
    df["precip_annual_mm"] = df["precip_annual_mm"].clip(lower=100)
    df["precip_max_daily_mm"] = df["precip_max_daily_mm"].clip(lower=10)
    df["consecutive_dry_days"] = df["consecutive_dry_days"].clip(lower=5)
    df["heat_stress_days"] = df["heat_stress_days"].clip(lower=0, upper=200)

    return df


def project_climate(baseline_df: pd.DataFrame, scenario: str, year: int,
                     seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Project climate variables forward under a given SSP scenario and time horizon.
    Applies scenario-specific scaling factors with spatial noise.
    """
    rng = np.random.default_rng(abs(seed + hash(scenario) + year) % (2**31))
    sc = SCENARIOS[scenario]
    n = len(baseline_df)

    # Interpolate temperature delta based on year
    if year <= 2050:
        frac = (year - 2025) / (2050 - 2025)
        temp_delta = sc["temp_delta_2050"] * frac
    else:
        frac = (year - 2050) / (2080 - 2050)
        temp_delta = sc["temp_delta_2050"] + (sc["temp_delta_2080"] - sc["temp_delta_2050"]) * frac

    # Precipitation change scales with time
    precip_factor = 1.0 + (sc["precip_change_factor"] - 1.0) * (year - 2025) / 55

    proj = baseline_df.copy()
    proj["period"] = str(year)
    proj["scenario"] = scenario

    # Temperature projection with spatial noise
    proj["temp_mean_annual_c"] = (
        baseline_df["temp_mean_annual_c"]
        + temp_delta
        + rng.normal(0, 0.3, n)
    )

    # Precipitation projection (extremes intensify faster than means)
    proj["precip_annual_mm"] = (
        baseline_df["precip_annual_mm"] * precip_factor
        + rng.normal(0, 30, n)
    ).clip(lower=50)

    # Extreme precipitation intensifies 1.5x faster than mean
    proj["precip_max_daily_mm"] = (
        baseline_df["precip_max_daily_mm"] * (precip_factor ** 1.5)
        + rng.normal(0, 8, n)
    ).clip(lower=10)

    # Drought days increase with warming (nonlinear)
    drought_increase = temp_delta * 8 + (temp_delta ** 2) * 2
    proj["consecutive_dry_days"] = (
        baseline_df["consecutive_dry_days"]
        + drought_increase
        + rng.normal(0, 5, n)
    ).clip(lower=5)

    # Heat stress days increase significantly with warming
    heat_increase = temp_delta * 12 + (temp_delta ** 2) * 4
    proj["heat_stress_days"] = (
        baseline_df["heat_stress_days"]
        + heat_increase
        + rng.poisson(lam=max(1, temp_delta * 3), size=n)
    ).clip(lower=0, upper=365)

    return proj


def generate_annual_maxima_series(baseline_df: pd.DataFrame, region_key: str,
                                   n_years: int = 30, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate synthetic annual maximum precipitation series for GEV fitting.
    Creates n_years of annual maxima per grid cell for return period analysis.
    """
    rng = np.random.default_rng(seed + 999)
    records = []

    for _, row in baseline_df.iterrows():
        mu = row["precip_max_daily_mm"]
        sigma = mu * 0.25
        # GEV-distributed annual maxima (shape=0.1, positive = heavy tail)
        annual_max = rng.gumbel(loc=mu, scale=sigma, size=n_years)
        annual_max = np.maximum(annual_max, 5.0)

        for yr_idx, val in enumerate(annual_max):
            records.append({
                "grid_id": row["grid_id"],
                "year": BASELINE_PERIOD[0] + yr_idx,
                "annual_max_precip_mm": val,
            })

    return pd.DataFrame(records)


def generate_all_data() -> dict:
    """
    Master function: generate all climate data for all regions, scenarios, and time horizons.
    Returns dict of DataFrames keyed by type.
    """
    print("=" * 60)
    print("STAGE 1: Data Generation & Preprocessing")
    print("=" * 60)

    all_grids = []
    all_baseline = []
    all_projections = []
    all_annual_maxima = []

    for region_key in REGIONS:
        region = REGIONS[region_key]
        print(f"\n  Region: {region['name']}")

        # Generate spatial grid
        grid = generate_grid(region_key)
        all_grids.append(grid)
        print(f"    Grid cells: {len(grid)}")

        # Generate baseline climate
        baseline = generate_baseline_climate(grid, region_key)
        all_baseline.append(baseline)
        print(f"    Baseline climate generated ({BASELINE_PERIOD[0]}-{BASELINE_PERIOD[1]})")

        # Generate annual maxima for GEV fitting
        annual_max = generate_annual_maxima_series(baseline, region_key)
        all_annual_maxima.append(annual_max)
        print(f"    Annual maxima series: {len(annual_max)} records")

        # Generate projections for each scenario × time horizon
        for scenario in SCENARIOS:
            for year in TIME_HORIZONS:
                proj = project_climate(baseline, scenario, year,
                                        seed=abs(RANDOM_SEED + hash(region_key)) % (2**31))
                all_projections.append(proj)
            print(f"    Projected: {scenario}")

    # Concatenate
    grids_gdf = pd.concat(all_grids, ignore_index=True)
    baseline_df = pd.concat(all_baseline, ignore_index=True)
    projections_df = pd.concat(all_projections, ignore_index=True)
    annual_maxima_df = pd.concat(all_annual_maxima, ignore_index=True)

    # Combine baseline + projections into one climate dataset
    climate_df = pd.concat([baseline_df, projections_df], ignore_index=True)

    # Save raw data
    grids_gdf.to_csv(RAW_DIR / "study_grids.csv", index=False)
    climate_df.to_csv(RAW_DIR / "climate_projections.csv", index=False)
    annual_maxima_df.to_csv(RAW_DIR / "annual_maxima_precip.csv", index=False)

    print(f"\n  Total climate records: {len(climate_df):,}")
    print(f"  Saved to: {RAW_DIR}")

    return {
        "grids": grids_gdf,
        "climate": climate_df,
        "annual_maxima": annual_maxima_df,
    }


if __name__ == "__main__":
    generate_all_data()
