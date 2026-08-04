import os
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

GOLD_PATH = "./data/gold/f1_feature_matrix.parquet"


def evaluate_benchmarks() -> None:
    print("=== F1 PREDICTIVE BASELINE BENCHMARKING ===")

    # 1. Load Feature Matrix
    if not os.path.exists(GOLD_PATH):
        raise FileNotFoundError(f"Gold feature matrix not found at {GOLD_PATH}")

    df = pd.read_parquet(GOLD_PATH)
    cat_cols = ["circuitId", "driverId", "constructorId", "regulatory_era"]
    for col in cat_cols:
        df[col] = df[col].astype("category")

    df = df.sort_values(by=["season", "round"]).reset_index(drop=True)

    # 2. Train-Test Split (Train: 2014-2025 | Test: 2026)
    train_df = df[df["season"] < 2026].copy()
    test_df = df[df["season"] == 2026].copy()

    features = [
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
        "regulatory_era",
    ]
    target = "positionOrder"

    train_groups = train_df.groupby("raceId", sort=False).size().to_numpy()
    test_groups = test_df.groupby("raceId", sort=False).size().to_numpy()

    max_pos_floor = int(max(df[target].max(), 30))
    y_train_rel = (max_pos_floor - train_df[target]).astype(int)
    y_test_rel = (max_pos_floor - test_df[target]).astype(int)

    # 3. Train Learning-to-Rank Model
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[1, 3, 10],
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )

    ranker.fit(  # pyright: ignore[reportUnknownMemberType]
        X=train_df[features],
        y=y_train_rel,
        group=train_groups,
        eval_X=test_df[features],
        eval_y=y_test_rel,
        eval_group=[test_groups],
        categorical_feature=cat_cols,
    )

    # 4. Generate Predictions & Rank Drivers
    raw_preds = ranker.predict(test_df[features])
    test_df["lgbm_score"] = np.asarray(raw_preds, dtype=float)
    test_df["lgbm_pred_pos"] = (
        test_df.groupby("raceId")["lgbm_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # 5. Championship Leader Baseline Ranking (Higher points = lower rank number)
    test_df["championship_pred_pos"] = (
        test_df.groupby("raceId")["driver_pts_lag"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # 6. Benchmark Evaluation
    benchmarks: dict[str, str] = {
        "1. Starting Grid Baseline": "grid",
        "2. Pure Pace Baseline": "qualifying_pos",
        "3. Championship Leader Baseline": "championship_pred_pos",
        "4. LTR Model (LGBMRanker)": "lgbm_pred_pos",
    }

    results: list[dict[str, Any]] = []
    total_races = len(test_groups)

    for name, col in benchmarks.items():
        # Winner (P1) Accuracy
        p1_correct = int(
            ((test_df[target] == 1) & (test_df[col] == 1)).sum()
        )
        p1_acc = (p1_correct / total_races) * 100

        # Podium (P1-P3) Accuracy
        podium_correct = int(
            ((test_df[target] <= 3) & (test_df[col] <= 3)).sum()
        )
        podium_acc = (podium_correct / (total_races * 3)) * 100

        # Overall Grid Rank Correlation (Spearman Rho across all positions)
        corrs: list[float] = []
        for _, group in test_df.groupby("raceId"):
            corr_val = group[target].corr(group[col], method="spearman")
            if pd.notna(corr_val):
                corrs.append(float(corr_val))
        avg_rho = float(np.mean(corrs)) if corrs else 0.0

        results.append(
            {
                "Strategy / Baseline": name,
                "P1 Accuracy (%)": f"{p1_acc:.2f}%",
                "Podium Accuracy (%)": f"{podium_acc:.2f}%",
                "Grid Spearman Rho (ρ)": f"{avg_rho:.4f}",
            }
        )

    summary_df = pd.DataFrame(results)
    print("\n" + summary_df.to_string(index=False))


if __name__ == "__main__":
    evaluate_benchmarks()