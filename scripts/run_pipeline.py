#!/usr/bin/env python3
"""
Pipeline Runner
===============
Executes the full climate risk assessment pipeline end-to-end:
  Stage 1: Data generation / ingestion
  Stage 2: Hazard probability modelling
  Stage 3: Exposure & financial impact analysis
  Stage 4: Risk indicators & portfolio aggregation
"""
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generator import generate_all_data
from src.hazard_modelling import run_hazard_modelling
from src.exposure_analysis import run_exposure_analysis
from src.risk_indicators import run_risk_indicators


def main():
    print("╔" + "═" * 58 + "╗")
    print("║  Climate Risk Geospatial Assessment Pipeline             ║")
    print("║  Multi-Hazard Physical Risk Analysis                     ║")
    print("║  SSP Scenarios: SSP1-2.6, SSP2-4.5, SSP5-8.5            ║")
    print("╚" + "═" * 58 + "╝")

    t0 = time.time()

    # Stage 1: Data generation
    data = generate_all_data()

    # Stage 2: Hazard modelling
    data = run_hazard_modelling(data)

    # Stage 3: Exposure analysis
    data = run_exposure_analysis(data)

    # Stage 4: Risk indicators
    data = run_risk_indicators(data)

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time: {elapsed:.1f} seconds")
    print(f"  Grid cells processed: {len(data['grids']):,}")
    print(f"  Climate records: {len(data['climate']):,}")
    print(f"  Risk output records: {len(data['grid_output']):,}")
    print(f"\n  Dashboard ready: run `streamlit run dashboard/app.py`")

    return data


if __name__ == "__main__":
    main()
