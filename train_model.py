import os

import lightgbm as lgb
import numpy as np
import pandas as pd

GOLD_PATH = "./data/gold/f1_feature_matrix.parquet"


def train_lgbm_ranker() -> None:
    print("=== MODEL TRAINING ===")

    if not os.path.exists(GOLD_PATH):
        raise FileNotFoundError(f"Gold feature matrix not found at {GOLD_PATH}")

    df = pd.read_parquet(GOLD_PATH)

    # Convert object/string columns to category for LightGBM
    cat_cols = ["circuitId", "driverId", "constructorId", "regulatory_era"]
    for col in cat_cols:
        df[col] = df[col].astype("category")

    # Sort chronologically to maintain time-series structure
    df = df.sort_values(by=["season", "round"]).reset_index(drop=True)

    # 2. Chronological Split (Train: 2014-2025 | Test: 2026)
    train_df = df[df["season"] < 2026].copy()
    test_df = df[df["season"] == 2026].copy()

    print(
        f"Train Dataset Shape: {train_df.shape} (Seasons {train_df['season'].min()}-{train_df['season'].max()})"
    )
    print(f"Test Dataset Shape:  {test_df.shape} (Season 2026)")

    # 3. Define Feature List and Target Variable
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

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    # 4. Compute Dynamic Group Sizes per Race (Required for LTR)
    train_groups = train_df.groupby("raceId", sort=False).size().to_numpy()
    test_groups = test_df.groupby("raceId", sort=False).size().to_numpy()

    # Convert target variable to non-negative relevance scores for LambdaRank (P1 = highest relevance score)
    # Using 30 guarantees all relevance labels stay >= 0 even with large historical grids
    max_pos_floor = int(max(df[target].max(), 30))
    y_train_rel = (max_pos_floor - y_train).astype(int)
    y_test_rel = (max_pos_floor - y_test).astype(int)

    # 5. Initialize and Train LGBMRanker
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
        X=X_train,
        y=y_train_rel,
        group=train_groups,
        eval_X=X_test,
        eval_y=y_test_rel,
        eval_group=[test_groups],
        categorical_feature=cat_cols,
    )

    # 6. Evaluate Predictions on 2026 Test Set
    raw_preds = ranker.predict(X_test)
    test_df["predicted_rank_score"] = np.asarray(raw_preds, dtype=float)

    # Rank drivers within each 2026 race based on predicted relevance score
    test_df["predicted_pos"] = (
        test_df.groupby("raceId")["predicted_rank_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # Calculate Top-1 and Top-3 Accuracy
    p1_matches = (test_df["positionOrder"] == 1) & (
        test_df["predicted_pos"] == 1
    )
    p1_accuracy = (p1_matches.sum() / len(test_groups)) * 100

    top3_matches = (test_df["positionOrder"] <= 3) & (
        test_df["predicted_pos"] <= 3
    )
    top3_accuracy = (top3_matches.sum() / (len(test_groups) * 3)) * 100

    print("\n=== MODEL EVALUATION (2026 TEST SET) ===")
    print(f"Winner Prediction Accuracy (P1):  {p1_accuracy:.2f}%")
    print(f"Podium Prediction Accuracy (P1-P3): {top3_accuracy:.2f}%")

    # 7. Feature Importance Ranking
    importances = ranker.feature_importances_
    importance_df = pd.DataFrame(
        {
            "Feature": features,
            "Importance": importances,
        }
    ).sort_values(by="Importance", ascending=False)

    print("\n--- Feature Importance Summary ---")
    print(importance_df.to_string(index=False))


if __name__ == "__main__":
    train_lgbm_ranker()