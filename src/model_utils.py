"""Update model_utils.py with points metrics and PU manufacturer mapping."""

import numpy as np
import pandas as pd

# F1 championship points mapping (2025+ rules) extended with sub-1.0 decay tail
# for positions outside the points. Values are multiplied by 100 to produce
# integers (required by LambdaRank) while preserving the non-linear scale.
F1_POINTS_RELEVANCE: dict[int, int] = {
    1: 2500,
    2: 1800,
    3: 1500,
    4: 1200,
    5: 1000,
    6: 800,
    7: 600,
    8: 400,
    9: 200,
    10: 100,
    11: 50,
    12: 40,
    13: 30,
    14: 25,
    15: 20,
    16: 15,
    17: 10,
    18: 8,
    19: 5,
    20: 2,
}

# Actual F1 championship points per finishing position (1-indexed)
F1_ACTUAL_POINTS: dict[int, float] = {
    1: 25.0,
    2: 18.0,
    3: 15.0,
    4: 12.0,
    5: 10.0,
    6: 8.0,
    7: 6.0,
    8: 4.0,
    9: 2.0,
    10: 1.0,
}

_MIN_RELEVANCE = 1


def position_to_relevance(positions: pd.Series) -> np.ndarray:
    """Convert positionOrder values to F1 points-scaled integer relevance labels."""
    return np.array(
        [F1_POINTS_RELEVANCE.get(int(p), _MIN_RELEVANCE) for p in positions],
        dtype=int,
    )


def get_label_gain() -> list[float]:
    """Build the label_gain parameter for LightGBM LambdaRank."""
    max_label = max(F1_POINTS_RELEVANCE.values())
    return [float(i) for i in range(max_label + 1)]


_CLASSIFIED_STATUSES = {"Finished"}
_LAPPED_PATTERN = "Lap"


def classify_status(status_series: pd.Series) -> pd.Series:
    """Derive a binary is_classified column from the status string.
    
    1 = classified finisher (Finished or Lapped)
    0 = DNF / DNS / DSQ
    """
    is_finished = status_series.isin(_CLASSIFIED_STATUSES)
    is_lapped = status_series.str.contains(_LAPPED_PATTERN, na=False)
    return (is_finished | is_lapped).astype(int)


def expected_points_error(y_true_pos: pd.Series | np.ndarray, y_pred_pos: pd.Series | np.ndarray) -> float:
    """Compute Expected Points Error (Weighted MAE where weights = F1 points scale).
    
    Measures the average error in predicted championship points earned per driver.
    """
    y_true = np.asarray(y_true_pos, dtype=int)
    y_pred = np.asarray(y_pred_pos, dtype=int)
    
    true_pts = np.array([F1_ACTUAL_POINTS.get(int(p), 0.0) for p in y_true])
    pred_pts = np.array([F1_ACTUAL_POINTS.get(int(p), 0.0) for p in y_pred])
    
    return float(np.mean(np.abs(true_pts - pred_pts)))


def get_pu_manufacturer(constructor_id: str, season: int) -> str:
    """Map constructorId and season to Power Unit Manufacturer.
    
    Returns one of: 'mercedes', 'ferrari', 'renault', 'honda_rbpt', 'audi'
    """
    c = str(constructor_id).lower()
    s = int(season)
    
    # 1. Mercedes PU
    if c in ("mercedes", "williams"):
        return "mercedes"
    if c in ("force_india", "racing_point") and s <= 2020:
        return "mercedes"
    if c == "aston_martin" and 2021 <= s <= 2025:
        return "mercedes"
    if c == "mclaren" and (s == 2014 or s >= 2021):
        return "mercedes"
    if c == "lotus_f1" and s == 2015:
        return "mercedes"
    if c == "manor" and s == 2016:
        return "mercedes"
    if c == "alpine" and s >= 2026:
        return "mercedes"
        
    # 2. Ferrari PU
    if c in ("ferrari", "haas"):
        return "ferrari"
    if c in ("sauber", "alfa") and s != 2026:  # Sauber became Audi in 2026
        return "ferrari"
    if c in ("marussia", "manor") and s <= 2015:
        return "ferrari"
    if c in ("toro_rosso",) and s == 2016:
        return "ferrari"
    if c == "cadillac" and s >= 2026:
        return "ferrari"
        
    # 3. Honda / RBPT PU
    if c in ("red_bull", "toro_rosso", "alphatauri", "rb") and s >= 2019:
        return "honda_rbpt"
    if c in ("toro_rosso", "alphatauri", "rb") and s == 2018:
        return "honda_rbpt"
    if c == "mclaren" and 2015 <= s <= 2017:
        return "honda_rbpt"
    if c == "aston_martin" and s >= 2026:
        return "honda_rbpt"
        
    # 4. Renault PU
    if c == "renault":
        return "renault"
    if c == "alpine" and s <= 2025:
        return "renault"
    if c in ("red_bull", "toro_rosso", "caterham", "lotus_f1") and s <= 2018:
        return "renault"
    if c == "mclaren" and 2018 <= s <= 2020:
        return "renault"
        
    # 5. Audi PU
    if c in ("audi",) or (c == "sauber" and s >= 2026):
        return "audi"
        
    # Default fallback if unknown
    return "other"
