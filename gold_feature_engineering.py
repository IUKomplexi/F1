import os
from typing import Any

import numpy as np
import pandas as pd

from model_utils import classify_status

SILVER_DIR = "./data/silver"
GOLD_DIR = "./data/gold"

os.makedirs(GOLD_DIR, exist_ok=True)

# --- EWMA Decay Constants ---
# Off-season decay: halve constructor momentum (winter development uncertainty)
OFF_SEASON_DECAY = 0.5
# Regulation reset: aggressive 90% decay (car hierarchy is unknown)
REGULATION_RESET_DECAY = 0.1
# Summer break: mild 15% decay (teams bring focused upgrades)
SUMMER_BREAK_DECAY = 0.85
# Regulation reset seasons
REGULATION_RESET_SEASONS = {2022, 2026}
# Approximate post-summer-break round (hardcoded midpoint)
SUMMER_BREAK_ROUND = 14
# Driver form regulation reset decay (talent transfers, but baseline shifts)
DRIVER_REG_RESET_DECAY = 0.3


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

    df_results = pd.read_parquet(os.path.join(SILVER_DIR, "fact_results.parquet"))
    df_qualifying = pd.read_parquet(os.path.join(SILVER_DIR, "fact_qualifying.parquet"))
    df_sprints = pd.read_parquet(os.path.join(SILVER_DIR, "fact_sprints.parquet"))

    df_results = df_results.sort_values(by=["season", "round"]).reset_index(drop=True)

    df = pd.merge(df_results, df_qualifying, on=["raceId", "driverId"], how="left")
    df = pd.merge(df, df_sprints, on=["raceId", "driverId"], how="left")

    df["qualifying_pos"] = df["qualifying_pos"].fillna(df["grid"])
    df["grid_penalty_delta"] = df["grid"] - df["qualifying_pos"]

    pole_times = df.groupby("raceId")["driver_best_q_ms"].transform("min")
    df["pace_delta_pct"] = ((df["driver_best_q_ms"] - pole_times) / pole_times) * 100

    df_teammate_info = df[
        ["raceId", "constructorId", "driverId", "driver_best_q_ms", "pace_delta_pct", "positionOrder"]
    ].copy()
    
    df_merged_tm = pd.merge(
        df, df_teammate_info, on=["raceId", "constructorId"], suffixes=("", "_tm")
    )
    
    df_tm_only = df_merged_tm[df_merged_tm["driverId"] != df_merged_tm["driverId_tm"]].copy()

    tm_data = (
        df_tm_only.groupby(["raceId", "driverId"])
        [["driver_best_q_ms_tm", "pace_delta_pct_tm", "positionOrder_tm"]]
        .first()
        .reset_index()
    )
    df = pd.merge(df, tm_data, on=["raceId", "driverId"], how="left")

    race_max_pace = df.groupby("raceId")["pace_delta_pct"].transform("max")
    df["pace_delta_pct"] = (
        df["pace_delta_pct"]
        .fillna(df["pace_delta_pct_tm"] + 1.0)
        .fillna(race_max_pace + 1.0)
        .fillna(5.0) 
    )

    # FIX: Shift to teammate delta percentages to scale linearly with circuit lap length
    df["teammate_delta_pct"] = ((df["driver_best_q_ms"] - df["driver_best_q_ms_tm"]) / df["driver_best_q_ms_tm"]) * 100
    
    # 1.5% penalty ~1.05s in Monaco and ~1.5s in Spa.
    df["teammate_delta_pct"] = np.where(
        df["driver_best_q_ms"].isna() & df["driver_best_q_ms_tm"].notna(), 1.5,
        np.where(
            df["driver_best_q_ms"].notna() & df["driver_best_q_ms_tm"].isna(), -1.5,
            df["teammate_delta_pct"].fillna(0.0)
        )
    )

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

    df["is_sprint_weekend"] = df["sprint_finish"].notna().astype(int)
    df["sprint_finish"] = df["sprint_finish"].fillna(df["qualifying_pos"])
    df["sprint_points"] = df["sprint_points"].fillna(0.0)
    df["total_weekend_pts"] = df["points"] + df["sprint_points"]

    df = df.sort_values(by=["season", "round"]).reset_index(drop=True)

    df["driver_pts_cum"] = df.groupby(["season", "driverId"])["total_weekend_pts"].cumsum()
    df["driver_pts_lag"] = df.groupby(["season", "driverId"])["driver_pts_cum"].shift(1).fillna(0.0)

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
        df_team_standings[["season", "round", "constructorId", "team_pts_lag"]],
        on=["season", "round", "constructorId"],
        how="left",
    )

    df["team_point_contribution_pct"] = np.where(
        df["team_pts_lag"] > 0,
        (df["driver_pts_lag"] / df["team_pts_lag"]) * 100,
        50.0,
    )

    historical_median_pos = df["positionOrder"].median()

    df["driver_form_ewma"] = (
        df.groupby("driverId")["positionOrder"]
        .transform(lambda x: x.shift(1).ewm(alpha=0.4, min_periods=1).mean())
        .fillna(historical_median_pos)
    )

    # --- Driver EWMA: Regulation Reset Decay ---
    # Driver talent persists but baseline performance expectation shifts
    for reset_season in REGULATION_RESET_SEASONS:
        reset_mask = df["season"] == reset_season
        first_round = df.loc[reset_mask, "round"].min() if reset_mask.any() else None
        if first_round is not None:
            driver_reset_mask = reset_mask & (df["round"] == first_round)
            df.loc[driver_reset_mask, "driver_form_ewma"] *= DRIVER_REG_RESET_DECAY

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
        df_team_pts_per_race[["season", "round", "constructorId", "constructor_form_ewma"]],
        on=["season", "round", "constructorId"],
        how="left",
    )

    # --- Constructor EWMA Decay ---
    # a) Off-season decay: halve momentum at Round 1 of each season
    season_first_rounds = df.groupby("season")["round"].transform("min")
    is_season_start = df["round"] == season_first_rounds
    df.loc[is_season_start, "constructor_form_ewma"] *= OFF_SEASON_DECAY

    # b) Regulation reset: aggressive additional decay at regulation change seasons
    for reset_season in REGULATION_RESET_SEASONS:
        reset_mask = is_season_start & (df["season"] == reset_season)
        df.loc[reset_mask, "constructor_form_ewma"] *= REGULATION_RESET_DECAY

    # c) Summer break decay: mild decay at post-summer-break round
    is_post_summer = df["round"] == SUMMER_BREAK_ROUND
    df.loc[is_post_summer, "constructor_form_ewma"] *= SUMMER_BREAK_DECAY

    df_retention = compute_track_retention(df_results)
    df = pd.merge(df, df_retention, on=["circuitId", "season"], how="left")
    df["track_retention_idx"] = df["track_retention_idx"].fillna(0.5)
    df["regulatory_era"] = df["season"].apply(map_regulatory_era)

    # --- Interaction Features ---
    # Pace advantage is worth MORE on high-overtaking tracks (low retention index)
    df["pace_x_overtaking"] = df["pace_delta_pct"] * (1 - df["track_retention_idx"])
    # Grid advantage is worth MORE on low-overtaking / processional tracks
    df["grid_x_retention"] = df["grid"] * df["track_retention_idx"]

    # --- Classification Target (for two-stage model) ---
    df["is_classified"] = classify_status(df["status"])

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
        "teammate_delta_pct",  # FIX: Updated feature name
        "teammate_h2h_form",
        "is_sprint_weekend",
        "sprint_finish",
        "driver_pts_lag",
        "team_pts_lag",
        "team_point_contribution_pct",
        "driver_form_ewma",
        "constructor_form_ewma",
        "track_retention_idx",
        "pace_x_overtaking",   # NEW: Interaction feature
        "grid_x_retention",    # NEW: Interaction feature
        "regulatory_era",
        "status",              # FIX: Passed status through to Gold for Train filtering
        "is_classified",       # NEW: Two-stage model target (Stage 1)
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