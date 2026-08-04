"""Fast parallelized grid search optimizer for EWMA decay parameters in F1 feature matrix."""

import concurrent.futures
import itertools
import os
import sys
import time
from typing import Any
import lightgbm as lgb
import numpy as np
import pandas as pd

from model_utils import (
    get_label_gain,
    position_to_relevance,
    expected_points_error,
)

GOLD_PATH = "./data/gold/f1_feature_matrix.parquet"

FEATURES = [
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
VAL_SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]
REGULATION_RESET_SEASONS = {2022, 2026}


def compute_decayed_ewma(
    df_work: pd.DataFrame,
    off_season_decay: float,
    regulation_reset_decay: float,
    summer_break_decay: float,
    driver_reg_reset_decay: float,
    summer_break_round: int,
) -> tuple[pd.Series, pd.Series]:
    """Fast vectorized recomputation of driver and constructor EWMA with decay parameters."""
    df_work = df_work.copy()
    df_work["total_weekend_pts"] = df_work.get("points", 0.0) + df_work.get("sprint_points", 0.0)
    
    historical_median_pos = float(df_work["positionOrder"].median())

    # Driver EWMA
    driver_ewma = (
        df_work.groupby("driverId")["positionOrder"]
        .transform(lambda x: x.shift(1).ewm(alpha=0.4, min_periods=1).mean())
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
        .transform(lambda x: x.shift(1).ewm(alpha=0.4, min_periods=1).mean())
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


def _evaluate_single_combination(args: tuple[pd.DataFrame, tuple[float, float, float, float, int]]) -> tuple[float, dict[str, Any]]:
    df_base, (os_d, reg_d, sb_d, dr_d, sb_r) = args
    df_eval = df_base.copy()
    
    driver_ewma, constructor_ewma = compute_decayed_ewma(
        df_eval,
        off_season_decay=os_d,
        regulation_reset_decay=reg_d,
        summer_break_decay=sb_d,
        driver_reg_reset_decay=dr_d,
        summer_break_round=sb_r,
    )
    
    df_eval["driver_form_ewma"] = driver_ewma
    df_eval["constructor_form_ewma"] = constructor_ewma

    fold_errors: list[float] = []

    for val_season in VAL_SEASONS:
        train_mask = df_eval["season"] < val_season
        val_mask = df_eval["season"] == val_season

        train_fold = df_eval[train_mask]
        val_fold = df_eval[val_mask].copy()

        if len(val_fold) == 0:
            continue

        train_groups = train_fold.groupby("raceId", sort=False).size().to_numpy()
        val_groups = val_fold.groupby("raceId", sort=False).size().to_numpy()

        y_train_rel = position_to_relevance(train_fold[TARGET])
        y_val_rel = position_to_relevance(val_fold[TARGET])

        ranker = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            label_gain=get_label_gain(),
            n_estimators=80,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=20,
            random_state=42,
            n_jobs=1,
            verbosity=-1,
        )

        ranker.fit(
            X=train_fold[FEATURES],
            y=y_train_rel,
            group=train_groups,
            eval_X=val_fold[FEATURES],
            eval_y=y_val_rel,
            eval_group=[val_groups],
        )

        raw_preds = ranker.predict(val_fold[FEATURES])
        val_fold["pred_score"] = np.asarray(raw_preds, dtype=float)
        val_fold["pred_pos"] = (
            val_fold.groupby("raceId")["pred_score"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

        err = expected_points_error(val_fold[TARGET], val_fold["pred_pos"])
        fold_errors.append(err)

    mean_err = float(np.mean(fold_errors))
    params = {
        "off_season_decay": os_d,
        "regulation_reset_decay": reg_d,
        "summer_break_decay": sb_d,
        "driver_reg_reset_decay": dr_d,
        "summer_break_round": sb_r,
    }
    return mean_err, params


def main():
    print("=== FAST PARALLEL GRID SEARCH FOR EWMA DECAY PARAMETERS ===")

    if not os.path.exists(GOLD_PATH):
        print(f"Error: Gold feature matrix not found at {GOLD_PATH}")
        sys.exit(1)

    df_base = pd.read_parquet(GOLD_PATH)

    off_season_range = [0.3, 0.5, 0.7, 0.9]
    reg_reset_range = [0.05, 0.1, 0.2, 0.4]
    summer_break_range = [0.7, 0.85, 1.0]
    driver_reg_range = [0.1, 0.3, 0.5, 0.7]
    summer_round_range = [13, 14, 15]

    all_combos = list(itertools.product(
        off_season_range,
        reg_reset_range,
        summer_break_range,
        driver_reg_range,
        summer_round_range
    ))

    total_combinations = len(all_combos)
    num_workers = min(os.cpu_count() or 4, 8)
    print(f"Testing {total_combinations} configurations across 6 CV folds using {num_workers} parallel workers...")

    start_time = time.time()
    best_score = float("inf")
    best_params: dict[str, Any] = {}

    tasks = [(df_base, combo) for combo in all_combos]

    completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_evaluate_single_combination, task): task for task in tasks}

        for future in concurrent.futures.as_completed(futures):
            completed += 1
            score, params = future.result()

            if score < best_score:
                best_score = score
                best_params = params
                print(f"[{completed}/{total_combinations}] NEW BEST Expected Points Error: {best_score:.4f} pts | Params: {best_params}", flush=True)
            elif completed % 50 == 0 or completed == total_combinations:
                print(f"Progress: [{completed}/{total_combinations}] ({completed/total_combinations*100:.1f}%) complete... Best so far: {best_score:.4f} pts", flush=True)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE!")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Best Expected Points Error: {best_score:.4f} pts/driver/race")
    print("Optimal Parameters:")
    for k, v in best_params.items():
        print(f"  {k} = {v}")
    print("=" * 60)

    # Save best parameters to a python file / dict for easy import
    out_code = f"# Optimal EWMA decay parameters derived from grid search\n"
    out_code += f"BEST_DECAY_PARAMS = {best_params}\n"
    with open("./best_decay_params.py", "w") as f:
        f.write(out_code)
    print("Saved optimal parameters to ./best_decay_params.py")


if __name__ == "__main__":
    main()
