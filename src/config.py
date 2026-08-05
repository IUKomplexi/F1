"""Central configuration - single source of truth for the F1 pipeline.

All paths, season bounds, decay parameters, and named constants live here.
Modules import from this package instead of hardcoding magic values.
"""

import logging
import os
import sys

# Ensure ``src/`` is importable even when a script is launched from the repo
# root (e.g. ``python run_pipeline.py``).  Harmless when already on sys.path.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# --- API (bronze layer) -----------------------------------------------------
BASE_URL = "https://api.jolpi.ca/ergast/f1/"
THROTTLE_DELAY = 0.5
HEADERS = {
    "User-Agent": "F1_DATA&ANALYTICS_UNI/1.0",
    "Accept": "application/json",
}

# --- Data paths --------------------------------------------------------------
BRONZE_DIR = os.path.join("data", "bronze")
SILVER_DIR = os.path.join("data", "silver")
GOLD_DIR = os.path.join("data", "gold")

RESULTS_PATH = os.path.join(SILVER_DIR, "fact_results.parquet")
QUALIFYING_PATH = os.path.join(SILVER_DIR, "fact_qualifying.parquet")
SPRINTS_PATH = os.path.join(SILVER_DIR, "fact_sprints.parquet")
GOLD_PATH = os.path.join(GOLD_DIR, "f1_feature_matrix.parquet")

# --- Season bounds ------------------------------------------------------------
FIRST_SEASON = 2014
TEST_SEASON = 2026
VAL_SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]
REGULATION_RESET_SEASONS = {2022, 2026}
SUMMER_BREAK_ROUND = 13

# --- EWMA decay parameters ----------------------------------------------------
# Imported from the grid-search artifact (src/decay_params.py); fall back to
# the previously-optimized values if the artifact is missing or malformed.
try:
    from decay_params import BEST_DECAY_PARAMS

    OFF_SEASON_DECAY = float(BEST_DECAY_PARAMS["off_season_decay"])
    REGULATION_RESET_DECAY = float(BEST_DECAY_PARAMS["regulation_reset_decay"])
    SUMMER_BREAK_DECAY = float(BEST_DECAY_PARAMS["summer_break_decay"])
    DRIVER_REG_RESET_DECAY = float(BEST_DECAY_PARAMS["driver_reg_reset_decay"])
except (ImportError, KeyError, TypeError, ValueError):
    OFF_SEASON_DECAY = 0.3
    REGULATION_RESET_DECAY = 0.05
    SUMMER_BREAK_DECAY = 0.7
    DRIVER_REG_RESET_DECAY = 0.7

EWMA_ALPHA = 0.4

# --- Imputation constants ------------------------------------------------------
DNF_RATE_FALLBACK = 0.17          # historical reliability fills
RETENTION_FALLBACK = 0.55         # track retention index fills
TEAMMATE_DELTA_FILL = 1.5         # +/- when one teammate lacks a Q time
PACE_DELTA_FALLBACK = 5.0         # pace delta when no Q time available
RETENTION_WINDOW_YEARS = 5        # lookback for track retention
RELIABILITY_WINDOW_RACES = 10     # lookback for DNF-rate features


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once; safe to call from any entry point."""
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        root.addHandler(handler)
    root.setLevel(level)
