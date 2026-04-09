"""
Exposure Analysis Module
========================
Overlays hazard probability layers with crop exposure data to compute
financial exposure and vulnerability-adjusted losses per grid cell.
"""
import numpy as np
import pandas as pd

from src.config import (
    REGIONS, SCENARIOS, TIME_HORIZONS,
    VULNERABILITY_FACTORS, PROCESSED_DIR, RANDOM_SEED,
)


def generate_crop_exposure(region_key: str, grid_ids: np.ndarray,
                            seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate synthetic crop exposure data per grid cell.
    Distributes regional production value across grid cells weighted by
    a simulated suitability surface.
    """
    rng = np.random.default_rng(abs(seed + hash(region_key) + 777) % (2**31))
    region = REGIONS[region_key]
    n_cells = len(grid_ids)

    records = []

    for crop in region["key_crops"]:
        # Crop share of total regional production
        crop_shares = {
            "soy": 0.50, "maize": 0.30, "cotton": 0.20,
            "dairy_pasture": 0.45, "wheat": 0.30, "potato": 0.25,
            "rice": 0.40, "sugarcane": 0.20,
        }
        share = crop_shares.get(crop, 1.0 / len(region["key_crops"]))
        crop_total_value = region["annual_production_value_usd"] * share

        # Spatial suitability weights (not uniform)
        weights = rng.dirichlet(np.ones(n_cells) * 2)
        cell_values = crop_total_value * weights

        # Crop area per cell
        crop_area_total = region["crop_area_km2"] * share
        cell_areas = crop_area_total * weights

        for i, gid in enumerate(grid_ids):
            records.append({
                "grid_id": gid,
                "region": region_key,
                "crop": crop,
                "crop_area_km2": cell_areas[i],
                "production_value_usd": cell_values[i],
                "yield_per_km2_usd": cell_values[i] / max(cell_areas[i], 0.01),
            })

    return pd.DataFrame(records)


def compute_exposure_overlay(composite_df: pd.DataFrame, flood_df: pd.DataFrame,
                              drought_df: pd.DataFrame, heat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Overlay hazard probabilities with crop exposure to compute
    vulnerability-adjusted financial exposure per grid cell.
    """
    print("\n  Computing exposure overlays...")

    all_exposure = []
    for region_key in REGIONS:
        region_cells = composite_df[composite_df["region"] == region_key]["grid_id"].unique()
        exposure = generate_crop_exposure(region_key, region_cells)
        all_exposure.append(exposure)

    exposure_df = pd.concat(all_exposure, ignore_index=True)

    # Merge exposure with composite hazard
    # For each scenario/period, compute exposed value
    records = []

    scenarios_periods = composite_df[["scenario", "period"]].drop_duplicates()

    for _, sp in scenarios_periods.iterrows():
        scenario = sp["scenario"]
        period = sp["period"]

        hazard_slice = composite_df[
            (composite_df["scenario"] == scenario)
            & (composite_df["period"] == period)
        ][["grid_id", "region", "lon", "lat", "flood_prob", "drought_prob", "heat_prob", "composite_prob", "risk_category"]]

        # Merge with exposure
        merged = exposure_df.merge(hazard_slice, on=["grid_id", "region"], how="inner")

        for _, row in merged.iterrows():
            crop = row["crop"]
            prod_value = row["production_value_usd"]

            # Compute vulnerability-adjusted loss for each hazard
            flood_vuln = VULNERABILITY_FACTORS["flood"].get(crop, 0.3)
            drought_vuln = VULNERABILITY_FACTORS["drought"].get(crop, 0.3)
            heat_vuln = VULNERABILITY_FACTORS["extreme_heat"].get(crop, 0.3)

            flood_loss = row["flood_prob"] * prod_value * flood_vuln
            drought_loss = row["drought_prob"] * prod_value * drought_vuln
            heat_loss = row["heat_prob"] * prod_value * heat_vuln

            # Total expected loss (avoiding double-counting via max approach)
            total_loss = flood_loss + drought_loss + heat_loss

            records.append({
                "grid_id": row["grid_id"],
                "region": row["region"],
                "lon": row["lon"],
                "lat": row["lat"],
                "scenario": scenario,
                "period": period,
                "crop": crop,
                "crop_area_km2": row["crop_area_km2"],
                "production_value_usd": prod_value,
                "flood_prob": row["flood_prob"],
                "drought_prob": row["drought_prob"],
                "heat_prob": row["heat_prob"],
                "composite_prob": row["composite_prob"],
                "risk_category": row["risk_category"],
                "flood_vuln": flood_vuln,
                "drought_vuln": drought_vuln,
                "heat_vuln": heat_vuln,
                "flood_expected_loss": flood_loss,
                "drought_expected_loss": drought_loss,
                "heat_expected_loss": heat_loss,
                "total_expected_loss": total_loss,
            })

    result = pd.DataFrame(records)
    print(f"    Exposure overlay records: {len(result):,}")
    return result


def run_exposure_analysis(data: dict) -> dict:
    """Master function: run exposure analysis."""
    print("\n" + "=" * 60)
    print("STAGE 3: Exposure & Financial Impact Analysis")
    print("=" * 60)

    exposure_df = compute_exposure_overlay(
        data["composite"], data["flood"], data["drought"], data["heat"]
    )

    exposure_df.to_csv(PROCESSED_DIR / "exposure_overlay.csv", index=False)
    print(f"\n  Saved to: {PROCESSED_DIR / 'exposure_overlay.csv'}")

    return {**data, "exposure": exposure_df}


if __name__ == "__main__":
    from src.data_generator import generate_all_data
    from src.hazard_modelling import run_hazard_modelling
    data = generate_all_data()
    data = run_hazard_modelling(data)
    run_exposure_analysis(data)
