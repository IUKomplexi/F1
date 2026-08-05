import os
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import GOLD_PATH
from model_utils import get_label_gain, position_to_relevance


def run_cross_validation() -> None:
    print("=== CROSS-VALIDATION (2014-2025) ===")

    if not os.path.exists(GOLD_PATH):
        raise FileNotFoundError(f" Feature matrix not found at {GOLD_PATH}")

    df = pd.read_parquet(GOLD_PATH)
    features = [
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
    target = "positionOrder"

    
    # Validation Seasonss
    val_seasons = [2020, 2021, 2022, 2023, 2024, 2025]
    fold_results: list[dict[str, Any]] = []

    for val_season in val_seasons:
        train_mask = df["season"] < val_season
        val_mask = df["season"] == val_season

        train_fold = df[train_mask].copy()
        val_fold = df[val_mask].copy()

        train_groups = train_fold.groupby("raceId", sort=False).size().to_numpy()
        val_groups = val_fold.groupby("raceId", sort=False).size().to_numpy()

        # Convert target to F1 championship points-scaled relevance
        y_train_rel = position_to_relevance(train_fold[target])
        y_val_rel = position_to_relevance(val_fold[target])

        # LGBM Ranker
        ranker = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            label_gain=get_label_gain(),
            n_estimators=100,
            learning_rate=0.03,
            num_leaves=15,          
            min_child_samples=20,    # Requires at least 1 race of data per leaf
            random_state=42,
        )

        ranker.fit(  # pyright: ignore[reportUnknownMemberType]
            X=train_fold[features],
            y=y_train_rel,
            group=train_groups,
            eval_X=val_fold[features],
            eval_y=y_val_rel,
            eval_group=[val_groups],
        )

        # Predictions
        raw_preds = ranker.predict(val_fold[features])
        val_fold["lgbm_score"] = np.asarray(raw_preds, dtype=float)
        val_fold["lgbm_pred_pos"] = (
            val_fold.groupby("raceId")["lgbm_score"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

        total_races = len(val_groups)

        # Pace Rho
        pace_rhos = [
            float(g[target].corr(g["qualifying_pos"], method="spearman"))
            for _, g in val_fold.groupby("raceId")
            if g["qualifying_pos"].nunique() > 1
        ]
        avg_pace_rho = float(np.mean(pace_rhos)) if pace_rhos else 0.0

        # LTR Model Rho
        model_rhos = [
            float(g[target].corr(g["lgbm_pred_pos"], method="spearman"))
            for _, g in val_fold.groupby("raceId")
            if g["lgbm_pred_pos"].nunique() > 1
        ]
        avg_model_rho = float(np.mean(model_rhos)) if model_rhos else 0.0

        p1_correct = int(((val_fold[target] == 1) & (val_fold["lgbm_pred_pos"] == 1)).sum())
        p1_acc = (p1_correct / total_races) * 100

        fold_results.append(
            {
                "Validation Season": val_season,
                "Races": total_races,
                "Pace Rho (Benchmark)": f"{avg_pace_rho:.4f}",
                "Model Rho (LGBMRanker)": f"{avg_model_rho:.4f}",
                "Model P1 Acc (%)": f"{p1_acc:.2f}%",
                "Beats Pace Baseline?": "YES" if avg_model_rho > avg_pace_rho else "NO",
            }
        )

    cv_df = pd.DataFrame(fold_results)
    print("\n" + cv_df.to_string(index=False))


if __name__ == "__main__":
    run_cross_validation()