import logging

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from config import GOLD_PATH, setup_logging

logger = logging.getLogger(__name__)


def run_smoke_test(parquet_path: str = GOLD_PATH) -> None:
    logger.info("=== RUNNING RANDOM FOREST SMOKE TEST ===")

    try:
        df: pd.DataFrame = pd.read_parquet(parquet_path)
    except FileNotFoundError:
        logger.error("Could not find %s. Run Phase 3 first.", parquet_path)
        return

    # 1. Isolate Features (X) and Target (y)
    drop_cols = ["raceId", "driverId", "constructorId", "positionOrder", "status", "is_classified"]

    numeric_df: pd.DataFrame = df.select_dtypes(include=['number'])
    features: list[str] = [c for c in numeric_df.columns if c not in drop_cols]

    X: pd.DataFrame = numeric_df[features].fillna(0)
    y: pd.Series = numeric_df["positionOrder"]

    # 2. Train a shallow Random Forest directly on the full dataset
    logger.info("Fitting diagnostic model...")
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X, y)

    # 3. Extract and analyze feature importances
    importances = rf.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    logger.info("\n--- Top 10 Feature Importances ---")
    logger.info(importance_df.head(10).to_string(index=False))

    # 4. The Leakage Alarm Logic
    top_importance = float(importance_df.iloc[0]["Importance"])
    top_feature = str(importance_df.iloc[0]["Feature"])

    logger.info("\n--- SMOKE TEST VERDICT ---")
    if top_importance > 0.85:
        logger.warning("[CRITICAL WARNING] Massive Data Leakage Detected!")
        logger.warning(
            "Feature '%s' holds %.1f%% of the model's predictive power.",
            top_feature,
            top_importance * 100,
        )
        logger.warning(
            "This feature is acting as a proxy for the final result. "
            "Remove it from your Gold Layer before proceeding to LGBM."
        )
    else:
        logger.info("[PASSED] No single feature dominates the model.")
        logger.info(
            "Your pipeline is clear of obvious post-race leakage. "
            "You are safe to begin LightGBM training."
        )


if __name__ == "__main__":
    setup_logging()
    run_smoke_test()
