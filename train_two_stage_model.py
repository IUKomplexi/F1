"""Two-Stage F1 Prediction Model.

Stage 1: Binary DNF Survival Classifier (LGBMClassifier)
    Predicts P(driver finishes / is classified) for each race entry.

Stage 2: Learning-to-Rank on Classified Finishers (LGBMRanker)
    Ranks only the classified finishers using F1 points-scaled relevance.

Merge: Classified drivers ranked by Stage 2 score (positions 1–N),
    DNF-predicted drivers appended at positions N+1 onwards ordered by
    descending P(classified).
"""

import os
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from model_utils import get_label_gain, position_to_relevance

GOLD_PATH = "./data/gold/f1_feature_matrix.parquet"

# --- Feature Lists ---
# Stage 1 and Stage 2 share the same feature set; the difference is in
# training data (all entries vs classified-only) and target variable.
FEATURES = [
    "circuitId",
    "driverId",
    "constructorId",
    "grid",
    "qualifying_pos",
    "grid_penalty_delta",
    "pace_delta_pct",
    "teammate_delta_pct",
    "teammate_h2h_form",
    "is_sprint_weekend",
    "sprint_finish",
    "driver_pts_lag",
    "team_pts_lag",
    "team_point_contribution_pct",
    "driver_form_ewma",
    "constructor_form_ewma",
    "track_retention_idx",
    "pace_x_overtaking",
    "grid_x_retention",
    "regulatory_era",
]

CAT_COLS = ["circuitId", "driverId", "constructorId", "regulatory_era"]
TARGET = "positionOrder"


def run_two_stage_evaluation() -> None:
    print("=== TWO-STAGE MODEL EVALUATION ===\n")

    if not os.path.exists(GOLD_PATH):
        raise FileNotFoundError(f"Gold feature matrix not found at {GOLD_PATH}")

    df = pd.read_parquet(GOLD_PATH)
    for col in CAT_COLS:
        df[col] = df[col].astype("category")
    df = df.sort_values(by=["season", "round"]).reset_index(drop=True)

    # --- Chronological Split ---
    train_df = df[df["season"] < 2026].copy()
    test_df = df[df["season"] == 2026].copy()

    print(f"Train: {train_df.shape[0]} entries ({train_df['season'].min()}-{train_df['season'].max()})")
    print(f"Test:  {test_df.shape[0]} entries (2026, {test_df['raceId'].nunique()} races)")
    print(f"Train DNF rate: {(1 - train_df['is_classified'].mean()) * 100:.1f}%")
    print(f"Test  DNF rate: {(1 - test_df['is_classified'].mean()) * 100:.1f}%")

    # ================================================================
    # STAGE 1: Survival Classifier
    # ================================================================
    print("\n--- Stage 1: DNF Survival Classifier ---")

    clf = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        random_state=42,
    )

    clf.fit(  # pyright: ignore[reportUnknownMemberType]
        X=train_df[FEATURES],
        y=train_df["is_classified"],
        categorical_feature=CAT_COLS,
    )

    # Predict survival probability for test set
    survival_probs = clf.predict_proba(test_df[FEATURES])[:, 1]
    test_df["survival_prob"] = survival_probs

    # Stage 1 accuracy
    test_df["predicted_classified"] = (test_df["survival_prob"] >= 0.5).astype(int)
    s1_accuracy = (test_df["predicted_classified"] == test_df["is_classified"]).mean() * 100
    print(f"Stage 1 Classification Accuracy: {s1_accuracy:.1f}%")

    # Stage 1 feature importances
    s1_imp = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": clf.feature_importances_,
    }).sort_values("Importance", ascending=False)
    print("\nStage 1 Top-5 Features:")
    print(s1_imp.head(5).to_string(index=False))

    # ================================================================
    # STAGE 2: LTR Ranker (trained on classified finishers only)
    # ================================================================
    print("\n--- Stage 2: Classified Finisher Ranking ---")

    train_classified = train_df[train_df["is_classified"] == 1].copy()
    print(f"Stage 2 training on {len(train_classified)} classified entries "
          f"(dropped {len(train_df) - len(train_classified)} DNFs from training)")

    # Recompute group sizes for classified-only races
    train_groups_s2 = train_classified.groupby("raceId", sort=False).size().to_numpy()

    # F1 points-scaled relevance for classified finishers
    # Re-rank within each race since DNFs are removed (positionOrder stays original)
    y_train_s2 = position_to_relevance(train_classified[TARGET])

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

    ranker.fit(  # pyright: ignore[reportUnknownMemberType]
        X=train_classified[FEATURES],
        y=y_train_s2,
        group=train_groups_s2,
        categorical_feature=CAT_COLS,
    )

    # Stage 2 feature importances
    s2_imp = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": ranker.feature_importances_,
    }).sort_values("Importance", ascending=False)
    print("\nStage 2 Top-5 Features:")
    print(s2_imp.head(5).to_string(index=False))

    # ================================================================
    # MERGE: Combine Stage 1 + Stage 2 predictions
    # ================================================================
    print("\n--- Merged Two-Stage Predictions ---")

    # Get Stage 2 rank scores for ALL test entries (the ranker can score anyone)
    raw_s2_scores = ranker.predict(test_df[FEATURES])
    test_df["rank_score"] = np.asarray(raw_s2_scores, dtype=float)

    # Composite scoring: classified drivers get rank score, DNF-predicted get pushed down
    # Strategy: Use a large negative offset for DNF-predicted drivers so they always
    # sort below classified drivers, then order by survival probability within DNFs
    test_df["composite_score"] = np.where(
        test_df["predicted_classified"] == 1,
        test_df["rank_score"],
        -1000.0 + test_df["survival_prob"]  # far below any rank score, ordered by survival prob
    )

    # Final predicted position: rank by composite score within each race
    test_df["twostage_pred_pos"] = (
        test_df.groupby("raceId")["composite_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # ================================================================
    # EVALUATION: Compare all strategies
    # ================================================================
    print("\n=== BENCHMARK COMPARISON (2026 TEST SET) ===\n")

    # Single-stage model for comparison (trained on ALL data, same hyperparams)
    train_groups_single = train_df.groupby("raceId", sort=False).size().to_numpy()
    y_train_single = position_to_relevance(train_df[TARGET])

    ranker_single = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=get_label_gain(),
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )
    ranker_single.fit(  # pyright: ignore[reportUnknownMemberType]
        X=train_df[FEATURES],
        y=y_train_single,
        group=train_groups_single,
        categorical_feature=CAT_COLS,
    )
    test_df["single_score"] = np.asarray(
        ranker_single.predict(test_df[FEATURES]), dtype=float
    )
    test_df["single_pred_pos"] = (
        test_df.groupby("raceId")["single_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # Championship leader baseline
    test_df["championship_pred_pos"] = (
        test_df.groupby("raceId")["driver_pts_lag"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    benchmarks: dict[str, str] = {
        "1. Starting Grid":              "grid",
        "2. Pure Pace (Qualifying)":     "qualifying_pos",
        "3. Championship Leader":         "championship_pred_pos",
        "4. Single-Stage LTR (v2)":      "single_pred_pos",
        "5. Two-Stage Model":            "twostage_pred_pos",
    }

    test_groups = test_df.groupby("raceId", sort=False).size().to_numpy()
    total_races = len(test_groups)
    results: list[dict[str, Any]] = []

    for name, col in benchmarks.items():
        # P1 accuracy
        p1 = int(((test_df[TARGET] == 1) & (test_df[col] == 1)).sum())
        p1_acc = (p1 / total_races) * 100

        # Top-3 podium accuracy
        top3 = int(((test_df[TARGET] <= 3) & (test_df[col] <= 3)).sum())
        top3_acc = (top3 / (total_races * 3)) * 100

        # Top-10 accuracy
        top10 = int(((test_df[TARGET] <= 10) & (test_df[col] <= 10)).sum())
        top10_acc = (top10 / (total_races * 10)) * 100

        # Spearman rank correlation
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
            "P1 Acc": f"{p1_acc:.1f}%",
            "Top-3 Acc": f"{top3_acc:.1f}%",
            "Top-10 Acc": f"{top10_acc:.1f}%",
            "Spearman Rho": f"{avg_rho:.4f}",
            "MAE": f"{avg_mae:.2f}",
        })

    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))

    # Print the actual full Stage 2 feature importances
    print("\n--- Stage 2 Full Feature Importances ---")
    print(s2_imp.to_string(index=False))


if __name__ == "__main__":
    run_two_stage_evaluation()
