"""Train and evaluate Clean F1 Predictive Model (v3).

- Pruned feature set (removed collinear & redundant features)
- All categoricals dropped (relies on continuous proxies)
- 5 historical reliability features added
- Evaluated on 2026 test set using Expected Points Error, P1/Top3/Top10 accuracy, MAE, and Spearman Rho.
"""

import os
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import GOLD_PATH
from model_utils import (
    get_label_gain,
    position_to_relevance,
    expected_points_error,
)

# Clean feature set (v3)
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


def train_and_evaluate_clean_model() -> None:
    print("=== CLEAN F1 PREDICTIVE MODEL (v3) EVALUATION ===\n")

    if not os.path.exists(GOLD_PATH):
        raise FileNotFoundError(f"Gold feature matrix not found at {GOLD_PATH}")

    df = pd.read_parquet(GOLD_PATH)
    df = df.sort_values(by=["season", "round"]).reset_index(drop=True)

    # Chronological Split: Train (2014-2025) | Test (2026)
    train_df = df[df["season"] < 2026].copy()
    test_df = df[df["season"] == 2026].copy()

    print(f"Train Dataset: {train_df.shape[0]} entries (Seasons {train_df['season'].min()}-{train_df['season'].max()})")
    print(f"Test Dataset:  {test_df.shape[0]} entries (Season 2026, {test_df['raceId'].nunique()} races)")
    print(f"Features:      {len(FEATURES)} numeric features (0 categoricals, 0 interaction terms)")

    # Dynamic group sizes for LTR
    train_groups = train_df.groupby("raceId", sort=False).size().to_numpy()
    test_groups = test_df.groupby("raceId", sort=False).size().to_numpy()

    # Convert target to F1 championship points-scaled relevance
    y_train_rel = position_to_relevance(train_df[TARGET])
    y_test_rel = position_to_relevance(test_df[TARGET])

    # Initialize and Train LGBMRanker
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[1, 3, 10],
        label_gain=get_label_gain(),
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )

    ranker.fit(
        X=train_df[FEATURES],
        y=y_train_rel,
        group=train_groups,
        eval_X=test_df[FEATURES],
        eval_y=y_test_rel,
        eval_group=[test_groups],
    )

    # Predictions
    raw_preds = ranker.predict(test_df[FEATURES])
    test_df["clean_pred_score"] = np.asarray(raw_preds, dtype=float)
    test_df["clean_pred_pos"] = (
        test_df.groupby("raceId")["clean_pred_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # Championship Leader Baseline
    test_df["championship_pred_pos"] = (
        test_df.groupby("raceId")["driver_pts_lag"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # Benchmark Comparison
    benchmarks: dict[str, str] = {
        "1. Pure Pace (Qualifying)": "qualifying_pos",
        "2. Championship Leader": "championship_pred_pos",
        "3. Clean LTR Model (v3)": "clean_pred_pos",
    }

    total_races = len(test_groups)
    results: list[dict[str, Any]] = []

    for name, col in benchmarks.items():
        # Expected Points Error
        exp_pts_err = expected_points_error(test_df[TARGET], test_df[col])

        # P1 Accuracy
        p1 = int(((test_df[TARGET] == 1) & (test_df[col] == 1)).sum())
        p1_acc = (p1 / total_races) * 100

        # Top-3 Podium Accuracy
        top3 = int(((test_df[TARGET] <= 3) & (test_df[col] <= 3)).sum())
        top3_acc = (top3 / (total_races * 3)) * 100

        # Top-10 Accuracy
        top10 = int(((test_df[TARGET] <= 10) & (test_df[col] <= 10)).sum())
        top10_acc = (top10 / (total_races * 10)) * 100

        # Spearman Rank Correlation
        rhos: list[float] = []
        for _, g in test_df.groupby("raceId"):
            r = g[TARGET].corr(g[col], method="spearman")
            if pd.notna(r):
                rhos.append(float(r))
        avg_rho = float(np.mean(rhos)) if rhos else 0.0

        # Mean Absolute Error
        maes: list[float] = []
        for _, g in test_df.groupby("raceId"):
            maes.append(float(np.mean(np.abs(g[TARGET] - g[col]))))
        avg_mae = float(np.mean(maes)) if maes else 0.0

        results.append({
            "Strategy": name,
            "Exp Pts Error": f"{exp_pts_err:.3f} pts",
            "P1 Acc": f"{p1_acc:.1f}%",
            "Top-3 Acc": f"{top3_acc:.1f}%",
            "Top-10 Acc": f"{top10_acc:.1f}%",
            "Spearman Rho": f"{avg_rho:.4f}",
            "MAE": f"{avg_mae:.2f}",
        })

    summary_df = pd.DataFrame(results)
    print("=== BENCHMARK COMPARISON (2026 TEST SET) ===")
    print(summary_df.to_string(index=False))

    # Feature Importance Ranking
    importances = ranker.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": importances,
    }).sort_values(by="Importance", ascending=False)

    print("\n--- Feature Importance Summary (v3 Clean Model) ---")
    print(importance_df.to_string(index=False))


if __name__ == "__main__":
    train_and_evaluate_clean_model()
