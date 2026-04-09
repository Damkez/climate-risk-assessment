"""
Risk Indicators Module
======================
Computes final risk indicators: Expected Annual Loss (EAL), composite risk scores,
and portfolio-level aggregations by region, crop, and scenario.
"""
import numpy as np
import pandas as pd

from src.config import REGIONS, SCENARIOS, TIME_HORIZONS, OUTPUT_DIR


def compute_expected_annual_loss(exposure_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Expected Annual Loss (EAL) aggregated by region, scenario, period, and crop.
    EAL = Σ (hazard_probability × exposure_value × vulnerability_factor)
    """
    print("\n  Computing Expected Annual Loss (EAL)...")

    eal = exposure_df.groupby(["region", "scenario", "period", "crop"]).agg(
        total_crop_area_km2=("crop_area_km2", "sum"),
        total_production_value_usd=("production_value_usd", "sum"),
        flood_eal=("flood_expected_loss", "sum"),
        drought_eal=("drought_expected_loss", "sum"),
        heat_eal=("heat_expected_loss", "sum"),
        total_eal=("total_expected_loss", "sum"),
        mean_composite_prob=("composite_prob", "mean"),
        grid_cells=("grid_id", "nunique"),
    ).reset_index()

    # Loss ratio
    eal["loss_ratio"] = eal["total_eal"] / eal["total_production_value_usd"].clip(lower=1)
    eal["loss_ratio_pct"] = (eal["loss_ratio"] * 100).round(2)

    print(f"    EAL records: {len(eal)}")
    return eal


def compute_risk_scores(eal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute normalized composite risk scores (0-100) per region and scenario.
    Combines hazard probability, financial exposure, and loss ratio into
    a single interpretable score.
    """
    print("\n  Computing composite risk scores...")

    # Aggregate to region-scenario-period level
    risk = eal_df.groupby(["region", "scenario", "period"]).agg(
        total_production_value_usd=("total_production_value_usd", "sum"),
        total_eal=("total_eal", "sum"),
        mean_composite_prob=("mean_composite_prob", "mean"),
        n_crops=("crop", "nunique"),
    ).reset_index()

    risk["loss_ratio"] = risk["total_eal"] / risk["total_production_value_usd"].clip(lower=1)

    # Normalized risk score (0-100)
    # Weighted combination of hazard probability and loss ratio
    prob_score = risk["mean_composite_prob"] * 100
    loss_score = (risk["loss_ratio"] * 100).clip(upper=100)
    risk["risk_score"] = (0.4 * prob_score + 0.6 * loss_score).clip(0, 100).round(1)

    # Risk rating
    risk["risk_rating"] = pd.cut(
        risk["risk_score"],
        bins=[0, 15, 30, 50, 70, 100],
        labels=["Very Low", "Low", "Medium", "High", "Very High"],
    )

    # Add region name
    risk["region_name"] = risk["region"].map(lambda r: REGIONS[r]["name"])
    risk["country"] = risk["region"].map(lambda r: REGIONS[r]["country"])

    print(f"    Risk score records: {len(risk)}")
    return risk


def compute_portfolio_summary(eal_df: pd.DataFrame, risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute portfolio-level summary statistics across all regions.
    """
    print("\n  Computing portfolio summary...")

    # By scenario and period
    portfolio = risk_df.groupby(["scenario", "period"]).agg(
        portfolio_value_usd=("total_production_value_usd", "sum"),
        portfolio_eal=("total_eal", "sum"),
        avg_risk_score=("risk_score", "mean"),
        max_risk_score=("risk_score", "max"),
        n_regions=("region", "nunique"),
    ).reset_index()

    portfolio["portfolio_loss_ratio_pct"] = (
        portfolio["portfolio_eal"] / portfolio["portfolio_value_usd"].clip(lower=1) * 100
    ).round(2)

    # Scenario label
    portfolio["scenario_label"] = portfolio["scenario"].map(
        lambda s: SCENARIOS.get(s, {}).get("label", s)
    )

    return portfolio


def compute_delta_analysis(risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute change in risk metrics relative to baseline for each scenario/period.
    """
    print("\n  Computing delta analysis (change from baseline)...")

    baseline = risk_df[risk_df["scenario"] == "historical"][
        ["region", "total_eal", "loss_ratio", "risk_score", "mean_composite_prob"]
    ].rename(columns={
        "total_eal": "baseline_eal",
        "loss_ratio": "baseline_loss_ratio",
        "risk_score": "baseline_risk_score",
        "mean_composite_prob": "baseline_prob",
    })

    future = risk_df[risk_df["scenario"] != "historical"].copy()
    delta = future.merge(baseline, on="region", how="left")

    delta["eal_change_pct"] = (
        (delta["total_eal"] - delta["baseline_eal"]) / delta["baseline_eal"].clip(lower=1) * 100
    ).round(1)
    delta["risk_score_change"] = (delta["risk_score"] - delta["baseline_risk_score"]).round(1)
    delta["prob_change"] = (delta["mean_composite_prob"] - delta["baseline_prob"]).round(4)

    return delta


def generate_grid_level_output(exposure_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a grid-level output dataset suitable for mapping in the dashboard.
    Aggregates across crops to give one row per grid cell per scenario/period.
    """
    grid_output = exposure_df.groupby(
        ["grid_id", "region", "lon", "lat", "scenario", "period"]
    ).agg(
        total_production_value=("production_value_usd", "sum"),
        total_expected_loss=("total_expected_loss", "sum"),
        flood_prob=("flood_prob", "first"),
        drought_prob=("drought_prob", "first"),
        heat_prob=("heat_prob", "first"),
        composite_prob=("composite_prob", "first"),
        risk_category=("risk_category", "first"),
        n_crops=("crop", "nunique"),
    ).reset_index()

    grid_output["loss_ratio"] = (
        grid_output["total_expected_loss"]
        / grid_output["total_production_value"].clip(lower=1)
    )

    return grid_output


def run_risk_indicators(data: dict) -> dict:
    """Master function: compute all risk indicators."""
    print("\n" + "=" * 60)
    print("STAGE 4: Risk Indicators & Portfolio Aggregation")
    print("=" * 60)

    exposure_df = data["exposure"]

    # EAL
    eal_df = compute_expected_annual_loss(exposure_df)

    # Risk scores
    risk_df = compute_risk_scores(eal_df)

    # Portfolio summary
    portfolio_df = compute_portfolio_summary(eal_df, risk_df)

    # Delta analysis
    delta_df = compute_delta_analysis(risk_df)

    # Grid-level output
    grid_output = generate_grid_level_output(exposure_df)

    # Save all outputs
    eal_df.to_csv(OUTPUT_DIR / "expected_annual_loss.csv", index=False)
    risk_df.to_csv(OUTPUT_DIR / "risk_scores.csv", index=False)
    portfolio_df.to_csv(OUTPUT_DIR / "portfolio_summary.csv", index=False)
    delta_df.to_csv(OUTPUT_DIR / "delta_analysis.csv", index=False)
    grid_output.to_csv(OUTPUT_DIR / "grid_risk_output.csv", index=False)

    print(f"\n  Portfolio summary:")
    for _, row in portfolio_df.iterrows():
        if row["scenario"] != "historical":
            print(f"    {row['scenario']} ({row['period']}): "
                  f"EAL = ${row['portfolio_eal']:,.0f} | "
                  f"Loss ratio = {row['portfolio_loss_ratio_pct']}% | "
                  f"Avg risk score = {row['avg_risk_score']:.1f}")

    print(f"\n  All outputs saved to: {OUTPUT_DIR}")

    return {
        **data,
        "eal": eal_df,
        "risk_scores": risk_df,
        "portfolio": portfolio_df,
        "delta": delta_df,
        "grid_output": grid_output,
    }


if __name__ == "__main__":
    from src.data_generator import generate_all_data
    from src.hazard_modelling import run_hazard_modelling
    from src.exposure_analysis import run_exposure_analysis
    data = generate_all_data()
    data = run_hazard_modelling(data)
    data = run_exposure_analysis(data)
    run_risk_indicators(data)
