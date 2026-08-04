import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def run_smoke_test(parquet_path: str) -> None:
    print("=== RUNNING RANDOM FOREST SMOKE TEST ===")
    
    try:
        df: pd.DataFrame = pd.read_parquet(parquet_path)
    except FileNotFoundError:
        print(f"Error: Could not find {parquet_path}. Run Phase 3 first.")
        return

    # 1. Isolate Features (X) and Target (y)
    drop_cols = ["raceId", "driverId", "constructorId", "positionOrder", "status", "is_classified"]
    
    numeric_df: pd.DataFrame = df.select_dtypes(include=['number'])
    features: list[str] = [c for c in numeric_df.columns if c not in drop_cols]
    
    # Type hints help satisfy strict Pylance checks
    X: pd.DataFrame = numeric_df[features].fillna(0) 
    y: pd.Series = numeric_df["positionOrder"]

    # 2. Train a shallow Random Forest directly on the full dataset
    print("Fitting diagnostic model...")
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X, y)

    # 3. Extract and analyze feature importances
    importances = rf.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print("\n--- Top 10 Feature Importances ---")
    print(importance_df.head(10).to_string(index=False))

    # 4. The Leakage Alarm Logic
    top_importance = float(importance_df.iloc[0]["Importance"])
    top_feature = str(importance_df.iloc[0]["Feature"])

    print("\n--- SMOKE TEST VERDICT ---")
    if top_importance > 0.85:
        print("[CRITICAL WARNING] Massive Data Leakage Detected!")
        print(f"Feature '{top_feature}' holds {top_importance*100:.1f}% of the model's predictive power.")
        print("This feature is acting as a proxy for the final result. Remove it from your Gold Layer before proceeding to LGBM.")
    else:
        print("[PASSED] No single feature dominates the model.")
        print("Your pipeline is clear of obvious post-race leakage. You are safe to begin LightGBM training.")

if __name__ == "__main__":
    run_smoke_test("./data/gold/f1_feature_matrix.parquet")