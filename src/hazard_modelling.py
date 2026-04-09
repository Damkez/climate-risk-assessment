"""
Hazard Modelling Module
=======================
Computes hazard exceedance probabilities for floods, droughts, and extreme heat.
Uses statistical extreme value analysis (GEV/Gumbel) for flood return periods
and threshold-based analysis for drought and heat stress.
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

from src.config import (
    REGIONS, SCENARIOS, TIME_HORIZONS, HAZARD_TYPES,
    RETURN_PERIODS, HEAT_THRESHOLDS, PROCESSED_DIR, RANDOM_SEED,
)


def fit_gev_return_periods(annual_maxima_df: pd.DataFrame, climate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit GEV distribution to annual maximum precipitation series and compute
    exceedance probabilities for different return periods.

    For future scenarios, the GEV parameters are shifted according to projected
    changes in extreme precipitation.
    """
    print("\n  Fitting GEV distributions for flood hazard...")

    # Get baseline data
    baseline = climate_df[climate_df["scenario"] == "historical"].copy()

    records = []

    for region_key in REGIONS:
        region_baseline = baseline[baseline["region"] == region_key]
        region_maxima = annual_maxima_df[
            annual_maxima_df["grid_id"].isin(region_baseline["grid_id"])
        ]

        for grid_id in region_baseline["grid_id"].unique():
            cell_maxima = region_maxima[region_maxima["grid_id"] == grid_id]["annual_max_precip_mm"].values

            if len(cell_maxima) < 10:
                continue

            # Fit GEV to baseline annual maxima
            try:
                shape, loc, scale = stats.genextreme.fit(cell_maxima)
            except Exception:
                # Fallback to Gumbel (shape=0)
                loc, scale = stats.gumbel_r.fit(cell_maxima)
                shape = 0.0

            cell_info = region_baseline[region_baseline["grid_id"] == grid_id].iloc[0]

            # Baseline return period analysis
            for rp in RETURN_PERIODS:
                exceed_prob = 1.0 / rp
                # Survival function threshold
                if shape != 0:
                    threshold = stats.genextreme.ppf(1 - exceed_prob, shape, loc=loc, scale=scale)
                else:
                    threshold = stats.gumbel_r.ppf(1 - exceed_prob, loc=loc, scale=scale)

                records.append({
                    "grid_id": grid_id,
                    "region": region_key,
                    "lon": cell_info["lon"],
                    "lat": cell_info["lat"],
                    "hazard": "flood",
                    "scenario": "historical",
                    "period": "baseline",
                    "return_period_years": rp,
                    "threshold_mm": threshold,
                    "annual_exceed_prob": exceed_prob,
                    "gev_shape": shape,
                    "gev_loc": loc,
                    "gev_scale": scale,
                })

            # Future scenarios: shift GEV parameters
            for scenario in SCENARIOS:
                sc = SCENARIOS[scenario]
                for year in TIME_HORIZONS:
                    future_row = climate_df[
                        (climate_df["grid_id"] == grid_id)
                        & (climate_df["scenario"] == scenario)
                        & (climate_df["period"] == str(year))
                    ]
                    if future_row.empty:
                        continue
                    future_row = future_row.iloc[0]

                    # Shift location parameter based on projected extreme precip change
                    precip_ratio = future_row["precip_max_daily_mm"] / max(cell_info["precip_max_daily_mm"], 1)
                    future_loc = loc * precip_ratio
                    future_scale = scale * (precip_ratio ** 0.8)

                    for rp in RETURN_PERIODS:
                        exceed_prob = 1.0 / rp
                        if shape != 0:
                            threshold = stats.genextreme.ppf(1 - exceed_prob, shape,
                                                              loc=future_loc, scale=future_scale)
                        else:
                            threshold = stats.gumbel_r.ppf(1 - exceed_prob,
                                                            loc=future_loc, scale=future_scale)

                        records.append({
                            "grid_id": grid_id,
                            "region": region_key,
                            "lon": cell_info["lon"],
                            "lat": cell_info["lat"],
                            "hazard": "flood",
                            "scenario": scenario,
                            "period": str(year),
                            "return_period_years": rp,
                            "threshold_mm": threshold,
                            "annual_exceed_prob": exceed_prob,
                            "gev_shape": shape,
                            "gev_loc": future_loc,
                            "gev_scale": future_scale,
                        })

    df = pd.DataFrame(records)
    print(f"    Flood hazard records: {len(df):,}")
    return df


def compute_drought_probability(climate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute drought exceedance probabilities based on consecutive dry days.
    Uses empirical thresholds and a logistic transformation to map CDD to probability.
    """
    print("\n  Computing drought exceedance probabilities...")

    # Drought severity thresholds (consecutive dry days)
    drought_thresholds = {
        "moderate": 30,
        "severe": 60,
        "extreme": 90,
    }

    records = []

    for _, row in climate_df.iterrows():
        cdd = row["consecutive_dry_days"]

        for severity, threshold in drought_thresholds.items():
            # Probability increases with CDD relative to threshold
            # Using a sigmoid/logistic function
            z = (cdd - threshold) / (threshold * 0.3)
            prob = 1.0 / (1.0 + np.exp(-z))
            prob = np.clip(prob, 0.01, 0.99)

            records.append({
                "grid_id": row["grid_id"],
                "region": row["region"],
                "lon": row["lon"],
                "lat": row["lat"],
                "hazard": "drought",
                "scenario": row["scenario"],
                "period": row["period"],
                "severity": severity,
                "consecutive_dry_days": cdd,
                "threshold_days": threshold,
                "annual_exceed_prob": prob,
            })

    df = pd.DataFrame(records)
    print(f"    Drought hazard records: {len(df):,}")
    return df


def compute_heat_stress_probability(climate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute extreme heat exceedance probabilities based on heat stress days
    relative to crop-specific thermal thresholds.
    """
    print("\n  Computing extreme heat exceedance probabilities...")

    records = []

    for _, row in climate_df.iterrows():
        region_key = row["region"]
        region = REGIONS[region_key]
        heat_days = row["heat_stress_days"]

        # Compute probability for each crop in this region
        for crop in region["key_crops"]:
            threshold_temp = HEAT_THRESHOLDS.get(crop, 35.0)

            # Adjust heat days based on temperature relative to threshold
            temp_excess = row["temp_mean_annual_c"] - (threshold_temp - 10)
            effective_heat_days = heat_days * max(0.5, 1 + temp_excess * 0.05)

            # Probability: fraction of growing season affected
            growing_season_days = 180
            prob = min(effective_heat_days / growing_season_days, 0.95)
            prob = max(prob, 0.01)

            records.append({
                "grid_id": row["grid_id"],
                "region": region_key,
                "lon": row["lon"],
                "lat": row["lat"],
                "hazard": "extreme_heat",
                "scenario": row["scenario"],
                "period": row["period"],
                "crop": crop,
                "heat_stress_days": heat_days,
                "effective_heat_days": effective_heat_days,
                "temp_mean_c": row["temp_mean_annual_c"],
                "thermal_threshold_c": threshold_temp,
                "annual_exceed_prob": prob,
            })

    df = pd.DataFrame(records)
    print(f"    Heat stress records: {len(df):,}")
    return df


def compute_composite_hazard_probability(flood_df: pd.DataFrame,
                                          drought_df: pd.DataFrame,
                                          heat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute composite hazard probability per grid cell by combining individual
    hazard probabilities. Uses the 'at least one hazard' probability:
    P(composite) = 1 - (1-P_flood)(1-P_drought)(1-P_heat)
    """
    print("\n  Computing composite hazard probabilities...")

    # Aggregate to single probability per cell/scenario/period
    # Flood: use 25-year return period as reference
    flood_agg = flood_df[flood_df["return_period_years"] == 25].groupby(
        ["grid_id", "region", "lon", "lat", "scenario", "period"]
    )["annual_exceed_prob"].mean().reset_index()
    flood_agg.rename(columns={"annual_exceed_prob": "flood_prob"}, inplace=True)

    # Drought: use severe threshold
    drought_agg = drought_df[drought_df["severity"] == "severe"].groupby(
        ["grid_id", "region", "lon", "lat", "scenario", "period"]
    )["annual_exceed_prob"].mean().reset_index()
    drought_agg.rename(columns={"annual_exceed_prob": "drought_prob"}, inplace=True)

    # Heat: average across crops
    heat_agg = heat_df.groupby(
        ["grid_id", "region", "lon", "lat", "scenario", "period"]
    )["annual_exceed_prob"].mean().reset_index()
    heat_agg.rename(columns={"annual_exceed_prob": "heat_prob"}, inplace=True)

    # Merge
    composite = flood_agg.merge(drought_agg, on=["grid_id", "region", "lon", "lat", "scenario", "period"], how="outer")
    composite = composite.merge(heat_agg, on=["grid_id", "region", "lon", "lat", "scenario", "period"], how="outer")

    # Fill missing with low probability
    for col in ["flood_prob", "drought_prob", "heat_prob"]:
        composite[col] = composite[col].fillna(0.02)

    # Composite probability
    composite["composite_prob"] = 1 - (
        (1 - composite["flood_prob"])
        * (1 - composite["drought_prob"])
        * (1 - composite["heat_prob"])
    )

    # Risk category
    composite["risk_category"] = pd.cut(
        composite["composite_prob"],
        bins=[0, 0.1, 0.25, 0.5, 0.75, 1.0],
        labels=["Very Low", "Low", "Medium", "High", "Very High"],
    )

    return composite


def run_hazard_modelling(data: dict) -> dict:
    """
    Master function: run all hazard modelling.
    """
    print("\n" + "=" * 60)
    print("STAGE 2: Hazard Probability Modelling")
    print("=" * 60)

    climate_df = data["climate"]
    annual_maxima_df = data["annual_maxima"]

    # Flood hazard (GEV)
    flood_df = fit_gev_return_periods(annual_maxima_df, climate_df)

    # Drought hazard
    drought_df = compute_drought_probability(climate_df)

    # Extreme heat hazard
    heat_df = compute_heat_stress_probability(climate_df)

    # Composite
    composite_df = compute_composite_hazard_probability(flood_df, drought_df, heat_df)

    # Save
    flood_df.to_csv(PROCESSED_DIR / "flood_hazard.csv", index=False)
    drought_df.to_csv(PROCESSED_DIR / "drought_hazard.csv", index=False)
    heat_df.to_csv(PROCESSED_DIR / "heat_hazard.csv", index=False)
    composite_df.to_csv(PROCESSED_DIR / "composite_hazard.csv", index=False)

    print(f"\n  Composite hazard records: {len(composite_df):,}")
    print(f"  Saved to: {PROCESSED_DIR}")

    return {
        **data,
        "flood": flood_df,
        "drought": drought_df,
        "heat": heat_df,
        "composite": composite_df,
    }


if __name__ == "__main__":
    from src.data_generator import generate_all_data
    data = generate_all_data()
    run_hazard_modelling(data)
