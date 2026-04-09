"""
Configuration module for Climate Risk Geospatial Assessment.
Contains all project constants, region definitions, scenario parameters,
and crop exposure data.
"""
import os
from pathlib import Path

# ── Project paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"

for d in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Climate scenarios (SSP) ────────────────────────────────────────────
SCENARIOS = {
    "SSP1-2.6": {
        "label": "Sustainability",
        "description": "Low emissions, strong mitigation",
        "temp_delta_2050": 1.2,   # °C above pre-industrial
        "temp_delta_2080": 1.5,
        "precip_change_factor": 1.03,  # fractional change
        "color": "#2ecc71",
    },
    "SSP2-4.5": {
        "label": "Middle of the Road",
        "description": "Moderate emissions, some mitigation",
        "temp_delta_2050": 1.8,
        "temp_delta_2080": 2.7,
        "precip_change_factor": 1.06,
        "color": "#f39c12",
    },
    "SSP5-8.5": {
        "label": "Fossil-Fuelled Development",
        "description": "High emissions, minimal mitigation",
        "temp_delta_2050": 2.4,
        "temp_delta_2080": 4.3,
        "precip_change_factor": 1.10,
        "color": "#e74c3c",
    },
}

TIME_HORIZONS = [2030, 2050, 2080]
BASELINE_PERIOD = (1985, 2014)

# ── Study regions ──────────────────────────────────────────────────────
REGIONS = {
    "mato_grosso": {
        "name": "Mato Grosso, Brazil",
        "country": "Brazil",
        "bbox": [-60.0, -18.0, -50.0, -8.0],   # [lon_min, lat_min, lon_max, lat_max]
        "center": [-55.0, -13.0],
        "grid_resolution": 0.25,  # degrees
        "primary_hazards": ["drought", "extreme_heat"],
        "key_crops": ["soy", "maize", "cotton"],
        "crop_area_km2": 95000,
        "annual_production_value_usd": 28_500_000_000,
    },
    "rhine_meuse": {
        "name": "Rhine-Meuse Delta, Netherlands",
        "country": "Netherlands",
        "bbox": [3.3, 51.2, 7.2, 53.5],
        "center": [5.25, 52.35],
        "grid_resolution": 0.1,
        "primary_hazards": ["flood", "drought"],
        "key_crops": ["dairy_pasture", "wheat", "potato"],
        "crop_area_km2": 18000,
        "annual_production_value_usd": 9_200_000_000,
    },
    "punjab": {
        "name": "Punjab, India",
        "country": "India",
        "bbox": [73.8, 29.5, 76.8, 32.5],
        "center": [75.3, 31.0],
        "grid_resolution": 0.25,
        "primary_hazards": ["extreme_heat", "flood"],
        "key_crops": ["wheat", "rice", "sugarcane"],
        "crop_area_km2": 41000,
        "annual_production_value_usd": 12_800_000_000,
    },
}

# ── Hazard parameters ──────────────────────────────────────────────────
HAZARD_TYPES = ["flood", "drought", "extreme_heat"]

# Crop-specific thermal thresholds (°C) for extreme heat
HEAT_THRESHOLDS = {
    "soy": 35.0,
    "maize": 37.0,
    "cotton": 38.0,
    "wheat": 32.0,
    "rice": 35.0,
    "potato": 30.0,
    "dairy_pasture": 33.0,
    "sugarcane": 38.0,
}

# Vulnerability factors: fraction of crop value lost per unit hazard intensity
VULNERABILITY_FACTORS = {
    "flood": {
        "soy": 0.45, "maize": 0.50, "cotton": 0.40,
        "wheat": 0.55, "rice": 0.30, "potato": 0.60,
        "dairy_pasture": 0.35, "sugarcane": 0.25,
    },
    "drought": {
        "soy": 0.55, "maize": 0.60, "cotton": 0.35,
        "wheat": 0.50, "rice": 0.65, "potato": 0.55,
        "dairy_pasture": 0.45, "sugarcane": 0.30,
    },
    "extreme_heat": {
        "soy": 0.40, "maize": 0.45, "cotton": 0.20,
        "wheat": 0.50, "rice": 0.35, "potato": 0.45,
        "dairy_pasture": 0.40, "sugarcane": 0.15,
    },
}

# Return periods for flood analysis (years)
RETURN_PERIODS = [5, 10, 25, 50, 100]

# ── Random seed for reproducibility ───────────────────────────────────
RANDOM_SEED = 42
