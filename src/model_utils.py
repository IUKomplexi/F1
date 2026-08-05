"""Shared helpers: relevance mapping, PU manufacturer mapping, metrics, and
canonical feature/model definitions for the F1 prediction pipeline."""

import numpy as np
import pandas as pd
import lightgbm as lgb

from config import EWMA_ALPHA, REGULATION_RESET_SEASONS

# Canonical clean-v3 feature set consumed by every training/eval script.
FEATURES: list[str] = [
    "qualifying_pos",
    "grid_penalty_delta",
    "pace_delta_pct",
    "teammate_delta_pct",
    "teammate_h2h_form",
    "sprint_finish",
    "driver_pts_lag",
    "team_pts_lag",
    "driver_form_ewma",
    "constructor_form_ewma",
    "track_retention_idx",
    "track_retention_confidence",
    "driver_dnf_rate_10",
    "constructor_dnf_rate_10",
    "driver_team_dnf_rate_10",
    "pu_dnf_rate_10",
    "circuit_dnf_rate_5yr",
]

TARGET = "positionOrder"
CAT_COLS: list[str] = ["circuitId", "driverId", "constructorId", "regulatory_era"]

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


def make_ranker(
    n_estimators: int = 150,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    min_child_samples: int = 20,
    eval_at: list[int] | None = None,
    **kwargs,
):
    """Build a configured LGBMRanker with the shared LambdaRank settings.

    Centralizes the objective/metric/label_gain boilerplate; callers only
    override the hyperparameters that differ between experiments.
    """
    params: dict = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "label_gain": get_label_gain(),
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "min_child_samples": min_child_samples,
        "random_state": 42,
    }
    if eval_at is not None:
        params["eval_at"] = eval_at
    params.update(kwargs)
    return lgb.LGBMRanker(**params)


def evaluate_predictions(
    test_df: pd.DataFrame,
    pred_col: str,
    target: str = TARGET,
) -> dict[str, float]:
    """Compute the shared benchmark metrics for a prediction column.

    Returns expected points error, P1/Top3/Top10 accuracy, per-race mean
    Spearman rho, and per-race MAE - all averaged across the test set.
    """
    total_races = int(test_df["raceId"].nunique())

    exp_pts_err = expected_points_error(test_df[target], test_df[pred_col])

    p1 = int(((test_df[target] == 1) & (test_df[pred_col] == 1)).sum())
    p1_acc = (p1 / total_races) * 100 if total_races else 0.0

    top3 = int(((test_df[target] <= 3) & (test_df[pred_col] <= 3)).sum())
    top3_acc = (top3 / (total_races * 3)) * 100 if total_races else 0.0

    top10 = int(((test_df[target] <= 10) & (test_df[pred_col] <= 10)).sum())
    top10_acc = (top10 / (total_races * 10)) * 100 if total_races else 0.0

    rhos: list[float] = []
    maes: list[float] = []
    for _, g in test_df.groupby("raceId"):
        r = g[target].corr(g[pred_col], method="spearman")
        if pd.notna(r):
            rhos.append(float(r))
        maes.append(float(np.mean(np.abs(g[target] - g[pred_col]))))
    avg_rho = float(np.mean(rhos)) if rhos else 0.0
    avg_mae = float(np.mean(maes)) if maes else 0.0

    return {
        "exp_pts_err": exp_pts_err,
        "p1_acc": p1_acc,
        "top3_acc": top3_acc,
        "top10_acc": top10_acc,
        "spearman_rho": avg_rho,
        "mae": avg_mae,
    }


def compute_decayed_ewma(
    df_work: pd.DataFrame,
    off_season_decay: float,
    regulation_reset_decay: float,
    summer_break_decay: float,
    driver_reg_reset_decay: float,
    summer_break_round: int,
) -> tuple[pd.Series, pd.Series]:
    """Recompute driver and constructor EWMA features with decay parameters.

    Shared by the gold feature builder and the decay-parameter grid search so
    the feature math is defined exactly once.
    """
    df_work = df_work.copy()
    df_work["total_weekend_pts"] = df_work.get("points", 0.0) + df_work.get("sprint_points", 0.0)

    historical_median_pos = float(df_work["positionOrder"].median())

    # Driver EWMA
    driver_ewma = (
        df_work.groupby("driverId")["positionOrder"]
        .transform(lambda x: x.shift(1).ewm(alpha=EWMA_ALPHA, min_periods=1).mean())
        .fillna(historical_median_pos)
    )

    # Driver regulation reset decay
    for reset_season in REGULATION_RESET_SEASONS:
        reset_mask = df_work["season"] == reset_season
        first_round = df_work.loc[reset_mask, "round"].min() if reset_mask.any() else None
        if first_round is not None:
            driver_reset_mask = reset_mask & (df_work["round"] == first_round)
            driver_ewma.loc[driver_reset_mask] *= driver_reg_reset_decay

    # Constructor EWMA
    df_team_pts_per_race = (
        df_work.groupby(["season", "round", "constructorId"])["total_weekend_pts"]
        .sum()
        .reset_index()
        .sort_values(by=["season", "round"])
    )
    df_team_pts_per_race["constructor_form_ewma"] = (
        df_team_pts_per_race.groupby("constructorId")["total_weekend_pts"]
        .transform(lambda x: x.shift(1).ewm(alpha=EWMA_ALPHA, min_periods=1).mean())
        .fillna(0.0)
    )

    df_merged = pd.merge(
        df_work[["season", "round", "constructorId"]],
        df_team_pts_per_race[["season", "round", "constructorId", "constructor_form_ewma"]],
        on=["season", "round", "constructorId"],
        how="left",
    )
    constructor_ewma = df_merged["constructor_form_ewma"].fillna(0.0).copy()

    # Season start decay
    season_first_rounds = df_work.groupby("season")["round"].transform("min")
    is_season_start = df_work["round"] == season_first_rounds
    constructor_ewma.loc[is_season_start] *= off_season_decay

    # Regulation reset decay
    for reset_season in REGULATION_RESET_SEASONS:
        reset_mask = is_season_start & (df_work["season"] == reset_season)
        constructor_ewma.loc[reset_mask] *= regulation_reset_decay

    # Summer break decay
    is_post_summer = df_work["round"] == summer_break_round
    constructor_ewma.loc[is_post_summer] *= summer_break_decay

    return driver_ewma, constructor_ewma
