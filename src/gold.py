import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from config import (
    DNF_RATE_FALLBACK,
    DRIVER_REG_RESET_DECAY,
    GOLD_DIR,
    GOLD_PATH,
    OFF_SEASON_DECAY,
    PACE_DELTA_FALLBACK,
    REGULATION_RESET_DECAY,
    RETENTION_FALLBACK,
    RETENTION_WINDOW_YEARS,
    RESULTS_PATH,
    QUALIFYING_PATH,
    SPRINTS_PATH,
    SUMMER_BREAK_DECAY,
    SUMMER_BREAK_ROUND,
    TEAMMATE_DELTA_FILL,
    setup_logging,
)
from model_utils import classify_status, compute_decayed_ewma, get_pu_manufacturer

logger = logging.getLogger(__name__)

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

    # Calculate global mean retention index across all historical circuits
    all_corrs: list[float] = []
    for circuit in circuits:
        c_df = df_results[df_results["circuitId"] == circuit]
        for s in c_df["season"].unique():
            p_df = c_df[(c_df["season"] < s) & (c_df["season"] >= s - 5)]
            if len(p_df) >= 5:
                r = p_df["grid"].corr(p_df["positionOrder"], method="spearman")
                if pd.notna(r):
                    all_corrs.append(float(r))
    global_mean_retention = float(np.mean(all_corrs)) if all_corrs else RETENTION_FALLBACK

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
            past_years = past_df["season"].nunique()
            confidence = min(past_years / RETENTION_WINDOW_YEARS, 1.0)

            if len(past_df) >= 10:
                corr_val = past_df["grid"].corr(
                    past_df["positionOrder"], method="spearman"
                )
                raw_retention = float(corr_val) if pd.notna(corr_val) else global_mean_retention
            else:
                raw_retention = global_mean_retention

            # Confidence weighting blend with global mean
            retention_idx = confidence * raw_retention + (1.0 - confidence) * global_mean_retention

            records.append(
                {
                    "circuitId": circuit,
                    "season": season,
                    "track_retention_idx": retention_idx,
                    "track_retention_confidence": confidence,
                }
            )

    return pd.DataFrame(records).drop_duplicates(subset=["circuitId", "season"])


def generate_gold_features(
    off_season_decay: float = OFF_SEASON_DECAY,
    regulation_reset_decay: float = REGULATION_RESET_DECAY,
    summer_break_decay: float = SUMMER_BREAK_DECAY,
    driver_reg_reset_decay: float = DRIVER_REG_RESET_DECAY,
    summer_break_round: int = SUMMER_BREAK_ROUND,
) -> pd.DataFrame:
    logger.info("=== PHASE 3: GOLD LAYER FEATURE ENGINEERING (CLEAN v3) ===")

    df_results = pd.read_parquet(RESULTS_PATH)
    df_qualifying = pd.read_parquet(QUALIFYING_PATH)
    df_sprints = pd.read_parquet(SPRINTS_PATH)

    # Exclude DNS (Did Not Start) entries
    dns_mask = df_results["status"].str.contains("DNS|Did not start", case=False, na=False)
    dns_count = int(dns_mask.sum())
    df_results = df_results[~dns_mask].copy()
    logger.info("Removed %d DNS entries from dataset.", dns_count)

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
        .fillna(PACE_DELTA_FALLBACK) 
    )

    df["teammate_delta_pct"] = ((df["driver_best_q_ms"] - df["driver_best_q_ms_tm"]) / df["driver_best_q_ms_tm"]) * 100
    
    df["teammate_delta_pct"] = np.where(
        df["driver_best_q_ms"].isna() & df["driver_best_q_ms_tm"].notna(), TEAMMATE_DELTA_FILL,
        np.where(
            df["driver_best_q_ms"].notna() & df["driver_best_q_ms_tm"].isna(), -TEAMMATE_DELTA_FILL,
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
        .transform(lambda x: x.shift(1).ewm(alpha=EWMA_ALPHA, min_periods=1).mean())
        .fillna(0.5)
    )

    # Sprint handling: Do NOT impute sprint_finish with qualifying_pos. Keep NaN for non-sprint weekends.
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

    # Driver & constructor form EWMA with decay (shared implementation)
    df["driver_form_ewma"], df["constructor_form_ewma"] = compute_decayed_ewma(
        df,
        off_season_decay=off_season_decay,
        regulation_reset_decay=regulation_reset_decay,
        summer_break_decay=summer_break_decay,
        driver_reg_reset_decay=driver_reg_reset_decay,
        summer_break_round=summer_break_round,
    )

    # Track retention with confidence weighting
    df_retention = compute_track_retention(df_results)
    df = pd.merge(df, df_retention, on=["circuitId", "season"], how="left")
    df["track_retention_idx"] = df["track_retention_idx"].fillna(RETENTION_FALLBACK)
    df["track_retention_confidence"] = df["track_retention_confidence"].fillna(0.0)
    df["regulatory_era"] = df["season"].apply(map_regulatory_era)

    # --- Classification Target ---
    df["is_classified"] = classify_status(df["status"])
    df["is_dnf"] = 1 - df["is_classified"]

    # --- Power Unit Manufacturer Mapping ---
    df["pu_manufacturer"] = [
        get_pu_manufacturer(str(c), int(s)) for c, s in zip(df["constructorId"], df["season"])
    ]

    # --- 5 Historical Reliability Features (10-race rolling, shift(1) to avoid leakage) ---
    df["driver_dnf_rate_10"] = (
        df.groupby("driverId")["is_dnf"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        .fillna(DNF_RATE_FALLBACK)
    )

    df["constructor_dnf_rate_10"] = (
        df.groupby("constructorId")["is_dnf"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        .fillna(DNF_RATE_FALLBACK)
    )

    df["driver_team_dnf_rate_10"] = (
        df.groupby(["driverId", "constructorId"])["is_dnf"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        .fillna(df["driver_dnf_rate_10"])
    )

    df["pu_dnf_rate_10"] = (
        df.groupby("pu_manufacturer")["is_dnf"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        .fillna(DNF_RATE_FALLBACK)
    )

    # Circuit 5-year rolling DNF rate
    circuit_dnf_records: list[dict[str, Any]] = []
    for circuit in df["circuitId"].unique():
        c_df = df[df["circuitId"] == circuit]
        for season in sorted(c_df["season"].unique()):
            past_c = c_df[(c_df["season"] < season) & (c_df["season"] >= season - 5)]
            if len(past_c) >= 10:
                c_rate = float((1.0 - past_c["is_classified"]).mean())
            else:
                c_rate = DNF_RATE_FALLBACK
            circuit_dnf_records.append({
                "circuitId": circuit,
                "season": season,
                "circuit_dnf_rate_5yr": c_rate,
            })
    df_circuit_dnf = pd.DataFrame(circuit_dnf_records).drop_duplicates(subset=["circuitId", "season"])
    df = pd.merge(df, df_circuit_dnf, on=["circuitId", "season"], how="left")
    df["circuit_dnf_rate_5yr"] = df["circuit_dnf_rate_5yr"].fillna(DNF_RATE_FALLBACK)

    # Output Clean Gold Feature Matrix
    gold_features = [
        "raceId",
        "season",
        "round",
        "circuitId",
        "driverId",
        "constructorId",
        "pu_manufacturer",
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
        "regulatory_era",
        "status",
        "is_classified",
        "positionOrder",
    ]

    df_gold = df[gold_features].copy()
    df_gold.to_parquet(GOLD_PATH, index=False)

    logger.info("=== COMPLETE: Gold Layer Feature Matrix Ready (Clean v3) ===")
    logger.info("Saved: %s | Shape: %s", GOLD_PATH, df_gold.shape)
    return df_gold


if __name__ == "__main__":
    setup_logging()
    generate_gold_features()