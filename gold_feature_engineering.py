import os
from typing import Any

import numpy as np
import pandas as pd

SILVER_DIR = "./data/silver"
GOLD_DIR = "./data/gold"

os.makedirs(GOLD_DIR, exist_ok=True)


def map_regulatory_era(season: int) -> str:
    if 2014 <= season <= 2021:
        return "V6_Hybrid_v1"
    elif 2022 <= season <= 2025:
        return "Ground_Effect"
    else:
        return "2026_Reset"


def compute_track_retention(df_results: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    circuits: list[str] = df_results["circuitId"].unique().tolist()

    for circuit in circuits:
        circuit_df: pd.DataFrame = df_results[
            df_results["circuitId"] == circuit
        ].copy()
        seasons: list[int] = sorted(
            [int(s) for s in circuit_df["season"].unique()]
        )

        for season in seasons:
            past_df: pd.DataFrame = circuit_df[
                (circuit_df["season"] < season)
                & (circuit_df["season"] >= season - 5)
            ]
            if len(past_df) >= 20:
                corr_val = past_df["grid"].corr(
                    past_df["positionOrder"], method="spearman"
                )
                retention_idx = float(corr_val) if pd.notna(corr_val) else 0.5
            else:
                retention_idx = 0.5

            records.append(
                {
                    "circuitId": circuit,
                    "season": season,
                    "track_retention_idx": retention_idx,
                }
            )

    return pd.DataFrame(records).drop_duplicates(subset=["circuitId", "season"])


def generate_gold_features() -> pd.DataFrame:
    print("=== PHASE 3: GOLD LAYER FEATURE ENGINEERING ===")

    # 1. Load Silver Tables
    df_results = pd.read_parquet(os.path.join(SILVER_DIR, "fact_results.parquet"))
    df_qualifying = pd.read_parquet(
        os.path.join(SILVER_DIR, "fact_qualifying.parquet")
    )
    df_sprints = pd.read_parquet(os.path.join(SILVER_DIR, "fact_sprints.parquet"))

    df_results = df_results.sort_values(by=["season", "round"]).reset_index(
        drop=True
    )

    # 2. Join Base Tables
    df = pd.merge(
        df_results, df_qualifying, on=["raceId", "driverId"], how="left"
    )
    df = pd.merge(df, df_sprints, on=["raceId", "driverId"], how="left")

    # 3. Grid Penalty Delta
    df["qualifying_pos"] = df["qualifying_pos"].fillna(df["grid"])
    df["grid_penalty_delta"] = df["grid"] - df["qualifying_pos"]

    # 4. Pace Delta Percentage
    pole_times = df.groupby("raceId")["driver_best_q_ms"].transform("min")
    df["pace_delta_pct"] = (
        (df["driver_best_q_ms"] - pole_times) / pole_times
    ) * 100

    # 5. Teammate Relativization Metrics
    df_teammate_info = df[
        ["raceId", "constructorId", "driverId", "driver_best_q_ms", "pace_delta_pct", "positionOrder"]
    ].copy()
    
    df_merged_tm = pd.merge(
        df,
        df_teammate_info,
        on=["raceId", "constructorId"],
        suffixes=("", "_tm"),
    )
    
    df_tm_only = df_merged_tm[
        df_merged_tm["driverId"] != df_merged_tm["driverId_tm"]
    ].copy()

    # Consolidate teammate data extraction
    tm_data = (
        df_tm_only.groupby(["raceId", "driverId"])
        [["driver_best_q_ms_tm", "pace_delta_pct_tm", "positionOrder_tm"]]
        .first()
        .reset_index()
    )
    df = pd.merge(df, tm_data, on=["raceId", "driverId"], how="left")

    # FIX: Smart Pace Imputation (Teammate + 1% penalty, else Race Max + 1%)
    race_max_pace = df.groupby("raceId")["pace_delta_pct"].transform("max")
    df["pace_delta_pct"] = (
        df["pace_delta_pct"]
        .fillna(df["pace_delta_pct_tm"] + 1.0)
        .fillna(race_max_pace + 1.0)
        .fillna(5.0)  # Absolute fallback for extreme edge cases
    )

    # FIX: Teammate Delta (Penalize missing times instead of returning 0.0)
    df["teammate_delta_ms"] = df["driver_best_q_ms"] - df["driver_best_q_ms_tm"]
    
    # If driver has no time but teammate does -> +1000ms penalty
    # If teammate has no time but driver does -> -1000ms reward
    # If both are missing -> 0.0 draw
    df["teammate_delta_ms"] = np.where(
        df["driver_best_q_ms"].isna() & df["driver_best_q_ms_tm"].notna(), 1000.0,
        np.where(
            df["driver_best_q_ms"].notna() & df["driver_best_q_ms_tm"].isna(), -1000.0,
            df["teammate_delta_ms"].fillna(0.0)
        )
    )

    # FIX: H2H Logic (Double DNF prevention - assume >15 is a DNF proxy draw)
    df["h2h_win"] = np.where(
        (df["positionOrder"] > 15) & (df["positionOrder_tm"] > 15), 
        0.5, 
        (df["positionOrder"] < df["positionOrder_tm"]).astype(float)
    )

    df["teammate_h2h_form"] = (
        df.groupby("driverId")["h2h_win"]
        .transform(lambda x: x.shift(1).ewm(alpha=0.4, min_periods=1).mean())
        .fillna(0.5)
    )

    # 6. Sprint Handling
    df["is_sprint_weekend"] = df["sprint_finish"].notna().astype(int)
    df["sprint_finish"] = df["sprint_finish"].fillna(df["qualifying_pos"])
    df["sprint_points"] = df["sprint_points"].fillna(0.0)
    df["total_weekend_pts"] = df["points"] + df["sprint_points"]

    # 7. Strictly Pre-Race Lagged Standings Math (Round - 1)
    df = df.sort_values(by=["season", "round"]).reset_index(drop=True)

    # Driver Points Lag
    df["driver_pts_cum"] = df.groupby(["season", "driverId"])[
        "total_weekend_pts"
    ].cumsum()
    df["driver_pts_lag"] = (
        df.groupby(["season", "driverId"])["driver_pts_cum"]
        .shift(1)
        .fillna(0.0)
    )

    # Team Points Lag (Aggregate per race first to prevent intra-team row shifting)
    df_team_standings = (
        df.groupby(["season", "round", "constructorId"])["total_weekend_pts"]
        .sum()
        .reset_index()
        .sort_values(by=["season", "round"])
    )
    df_team_standings["team_pts_cum"] = df_team_standings.groupby(
        ["season", "constructorId"]
    )["total_weekend_pts"].cumsum()
    
    df_team_standings["team_pts_lag"] = (
        df_team_standings.groupby(["season", "constructorId"])["team_pts_cum"]
        .shift(1)
        .fillna(0.0)
    )

    df = pd.merge(
        df,
        df_team_standings[
            ["season", "round", "constructorId", "team_pts_lag"]
        ],
        on=["season", "round", "constructorId"],
        how="left",
    )

    # Driver Point Contribution Percentage
    df["team_point_contribution_pct"] = np.where(
        df["team_pts_lag"] > 0,
        (df["driver_pts_lag"] / df["team_pts_lag"]) * 100,
        50.0,
    )

    # 8. EWMA Form Calculation (alpha=0.4, shifted by 1)
    
    # FIX: Calculate historical median for rookie cold-starts instead of hardcoding
    historical_median_pos = df["positionOrder"].median()

    df["driver_form_ewma"] = (
        df.groupby("driverId")["positionOrder"]
        .transform(lambda x: x.shift(1).ewm(alpha=0.4, min_periods=1).mean())
        .fillna(historical_median_pos)
    )

    df_team_pts_per_race = (
        df.groupby(["season", "round", "constructorId"])["total_weekend_pts"]
        .sum()
        .reset_index()
        .sort_values(by=["season", "round"])
    )
    df_team_pts_per_race["constructor_form_ewma"] = (
        df_team_pts_per_race.groupby("constructorId")["total_weekend_pts"]
        .transform(lambda x: x.shift(1).ewm(alpha=0.4, min_periods=1).mean())
        .fillna(0.0)
    )
    df = pd.merge(
        df,
        df_team_pts_per_race[
            ["season", "round", "constructorId", "constructor_form_ewma"]
        ],
        on=["season", "round", "constructorId"],
        how="left",
    )

    # 9. Circuit Profiling & Regulations Era
    df_retention = compute_track_retention(df_results)
    df = pd.merge(df, df_retention, on=["circuitId", "season"], how="left")
    df["track_retention_idx"] = df["track_retention_idx"].fillna(0.5)
    df["regulatory_era"] = df["season"].apply(map_regulatory_era)

    # 10. Schema Export
    gold_features = [
        "raceId",
        "season",
        "round",
        "circuitId",
        "driverId",
        "constructorId",
        "grid",
        "qualifying_pos",
        "grid_penalty_delta",
        "pace_delta_pct",
        "teammate_delta_ms",
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
        "positionOrder",
    ]

    df_gold = df[gold_features].copy()
    output_path = os.path.join(GOLD_DIR, "f1_feature_matrix.parquet")
    df_gold.to_parquet(output_path, index=False)

    print("=== COMPLETE: Gold Layer Feature Matrix Ready ===")
    print(f"Saved: {output_path} | Shape: {df_gold.shape}")
    return df_gold


if __name__ == "__main__":
    generate_gold_features()