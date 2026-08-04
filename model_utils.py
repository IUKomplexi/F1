"""Shared utilities for F1 predictive model training and evaluation."""

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

# Minimum relevance for positions beyond 20 (e.g. historically larger grids)
_MIN_RELEVANCE = 1


def position_to_relevance(positions: pd.Series) -> np.ndarray:
    """Convert positionOrder values to F1 points-scaled integer relevance labels.

    Args:
        positions: Series of integer finishing positions (1-indexed).

    Returns:
        NumPy array of integer relevance scores suitable for LambdaRank.
    """
    return np.array(
        [F1_POINTS_RELEVANCE.get(int(p), _MIN_RELEVANCE) for p in positions],
        dtype=int,
    )


def get_label_gain() -> list[float]:
    """Build the label_gain parameter for LightGBM LambdaRank.

    LightGBM maps integer label i to label_gain[i]. We need enough entries
    to cover the maximum relevance label (2500).

    Returns:
        List of floats where index i maps to gain value i (identity mapping).
    """
    max_label = max(F1_POINTS_RELEVANCE.values())
    return [float(i) for i in range(max_label + 1)]


# Status values that indicate a driver was "classified" (finished the race,
# even if lapped). Everything else is a DNF / DNS / DSQ.
_CLASSIFIED_STATUSES = {"Finished"}
_LAPPED_PATTERN = "Lap"  # catches "+1 Lap", "+2 Laps", "Lapped", etc.


def classify_status(status_series: pd.Series) -> pd.Series:
    """Derive a binary is_classified column from the status string.

    Returns:
        Integer Series: 1 = classified finisher, 0 = DNF/DNS/DSQ.
    """
    is_finished = status_series.isin(_CLASSIFIED_STATUSES)
    is_lapped = status_series.str.contains(_LAPPED_PATTERN, na=False)
    return (is_finished | is_lapped).astype(int)
