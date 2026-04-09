"""
Unit tests for the Climate Risk Assessment pipeline.
Run: python -m pytest tests/test_pipeline.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from src.config import REGIONS, SCENARIOS, TIME_HORIZONS, HAZARD_TYPES, RETURN_PERIODS
from src.data_generator import generate_grid, generate_baseline_climate, project_climate
from src.hazard_modelling import fit_gev_return_periods, compute_drought_probability
from src.exposure_analysis import generate_crop_exposure


class TestDataGenerator:
    """Tests for synthetic data generation."""

    def test_grid_generation(self):
        grid = generate_grid("mato_grosso")
        assert len(grid) > 0
        assert "lon" in grid.columns
        assert "lat" in grid.columns
        assert "grid_id" in grid.columns
        assert grid["lon"].min() >= REGIONS["mato_grosso"]["bbox"][0]
        assert grid["lat"].min() >= REGIONS["mato_grosso"]["bbox"][1]

    def test_baseline_climate(self):
        grid = generate_grid("rhine_meuse")
        baseline = generate_baseline_climate(grid, "rhine_meuse")
        assert len(baseline) == len(grid)
        assert "temp_mean_annual_c" in baseline.columns
        assert "precip_annual_mm" in baseline.columns
        # Netherlands should be temperate
        assert baseline["temp_mean_annual_c"].mean() < 20
        assert baseline["precip_annual_mm"].mean() > 500

    def test_projections_warmer_than_baseline(self):
        grid = generate_grid("punjab")
        baseline = generate_baseline_climate(grid, "punjab")
        proj_2080 = project_climate(baseline, "SSP5-8.5", 2080)
        # SSP5-8.5 at 2080 should be significantly warmer
        assert proj_2080["temp_mean_annual_c"].mean() > baseline["temp_mean_annual_c"].mean()

    def test_scenario_ordering(self):
        """Higher forcing scenarios should produce more warming."""
        grid = generate_grid("mato_grosso")
        baseline = generate_baseline_climate(grid, "mato_grosso")
        proj_low = project_climate(baseline, "SSP1-2.6", 2080)
        proj_high = project_climate(baseline, "SSP5-8.5", 2080)
        assert proj_high["temp_mean_annual_c"].mean() > proj_low["temp_mean_annual_c"].mean()

    def test_all_regions_generate(self):
        for region_key in REGIONS:
            grid = generate_grid(region_key)
            baseline = generate_baseline_climate(grid, region_key)
            assert len(baseline) > 0
            assert baseline["precip_annual_mm"].min() > 0


class TestHazardModelling:
    """Tests for hazard probability computations."""

    def test_drought_probability_bounds(self):
        grid = generate_grid("punjab")
        baseline = generate_baseline_climate(grid, "punjab")
        drought = compute_drought_probability(baseline)
        assert drought["annual_exceed_prob"].min() >= 0.01
        assert drought["annual_exceed_prob"].max() <= 0.99

    def test_drought_increases_with_warming(self):
        grid = generate_grid("mato_grosso")
        baseline = generate_baseline_climate(grid, "mato_grosso")
        proj = project_climate(baseline, "SSP5-8.5", 2080)
        d_base = compute_drought_probability(baseline)
        d_proj = compute_drought_probability(proj)
        # Average severe drought prob should increase
        base_severe = d_base[d_base["severity"] == "severe"]["annual_exceed_prob"].mean()
        proj_severe = d_proj[d_proj["severity"] == "severe"]["annual_exceed_prob"].mean()
        assert proj_severe > base_severe


class TestExposure:
    """Tests for crop exposure calculations."""

    def test_exposure_sums_to_regional_value(self):
        grid = generate_grid("mato_grosso")
        exposure = generate_crop_exposure("mato_grosso", grid["grid_id"].values)
        total = exposure["production_value_usd"].sum()
        expected = REGIONS["mato_grosso"]["annual_production_value_usd"]
        # Should be approximately equal (floating point)
        assert abs(total - expected) / expected < 0.01

    def test_all_crops_present(self):
        grid = generate_grid("rhine_meuse")
        exposure = generate_crop_exposure("rhine_meuse", grid["grid_id"].values)
        expected_crops = set(REGIONS["rhine_meuse"]["key_crops"])
        actual_crops = set(exposure["crop"].unique())
        assert expected_crops == actual_crops


class TestGEVFitting:
    """Tests for GEV distribution fitting."""

    def test_gev_fit_valid(self):
        # Generate sample data and fit
        rng = np.random.default_rng(42)
        sample = rng.gumbel(loc=50, scale=12, size=30)
        shape, loc, scale = stats.genextreme.fit(sample)
        assert scale > 0
        assert np.isfinite(shape)

    def test_return_period_ordering(self):
        """Higher return periods should have higher thresholds."""
        rng = np.random.default_rng(42)
        sample = rng.gumbel(loc=50, scale=12, size=30)
        shape, loc, scale = stats.genextreme.fit(sample)
        thresholds = []
        for rp in RETURN_PERIODS:
            t = stats.genextreme.ppf(1 - 1.0/rp, shape, loc=loc, scale=scale)
            thresholds.append(t)
        # Should be monotonically increasing
        for i in range(1, len(thresholds)):
            assert thresholds[i] > thresholds[i-1]


if __name__ == "__main__":
    try:
        import pytest
        pytest.main([__file__, "-v"])
    except ImportError:
        # Run tests manually without pytest
        import traceback
        test_classes = [TestDataGenerator, TestHazardModelling, TestExposure, TestGEVFitting]
        passed = 0
        failed = 0
        for cls in test_classes:
            instance = cls()
            for name in dir(instance):
                if name.startswith("test_"):
                    try:
                        getattr(instance, name)()
                        print(f"  PASS  {cls.__name__}.{name}")
                        passed += 1
                    except Exception as e:
                        print(f"  FAIL  {cls.__name__}.{name}: {e}")
                        failed += 1
        print(f"\n{passed} passed, {failed} failed")
