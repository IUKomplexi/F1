import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from config import BRONZE_DIR, RESULTS_PATH, SILVER_DIR, SPRINTS_PATH, QUALIFYING_PATH, setup_logging

logger = logging.getLogger(__name__)

os.makedirs(SILVER_DIR, exist_ok=True)


def parse_time_to_ms(time_str: str | None) -> float | None:
    if not time_str or time_str == r"\N":
        return None
    try:
        parts = time_str.split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60000 + float(parts[1]) * 1000
        elif len(parts) == 1:
            return float(parts[0]) * 1000
    except ValueError:
        return None
    return None


def process_results() -> pd.DataFrame:
    logger.info("--- Parsing Fact Results ---")
    results_list: list[dict[str, Any]] = []

    for file in os.listdir(BRONZE_DIR):
        if file.endswith(".json") and "results" in file:
            filepath = os.path.join(BRONZE_DIR, file)
            with open(filepath, "r", encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)

            races: list[dict[str, Any]] = (
                data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            )
            for race in races:
                season = int(race.get("season", 0))
                if season < 2014:
                    continue

                round_no = int(race.get("round", 0))
                circuit_id = race.get("Circuit", {}).get("circuitId")
                race_id = f"{season}_{round_no}"

                for result in race.get("Results", []):
                    results_list.append(
                        {
                            "raceId": race_id,
                            "season": season,
                            "round": round_no,
                            "circuitId": circuit_id,
                            "driverId": result.get("Driver", {}).get("driverId"),
                            "constructorId": result.get("Constructor", {}).get(
                                "constructorId"
                            ),
                            "grid": result.get("grid"),
                            "positionOrder": result.get("position"),
                            "points": result.get("points"),
                            "status": result.get("status"),  # FIX: Extracted status for DNF filtering
                        }
                    )

    df = pd.DataFrame(results_list)
    df.replace(to_replace=r"\N", value=np.nan, inplace=True)
    
    df = df.drop_duplicates(subset=["raceId", "driverId"]).reset_index(drop=True)
    
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
    df["positionOrder"] = pd.to_numeric(df["positionOrder"], errors="coerce")
    df["points"] = pd.to_numeric(df["points"], errors="coerce")

    df.to_parquet(RESULTS_PATH, index=False)
    logger.info("Saved: %s | Shape: %s", RESULTS_PATH, df.shape)
    return df


def process_qualifying() -> pd.DataFrame:
    # (Remains identical to original implementation)
    logger.info("--- Parsing Fact Qualifying ---")
    qual_list: list[dict[str, Any]] = []

    for file in os.listdir(BRONZE_DIR):
        if file.endswith(".json") and "qualifying" in file:
            filepath = os.path.join(BRONZE_DIR, file)
            with open(filepath, "r", encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)

            races: list[dict[str, Any]] = (
                data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            )
            for race in races:
                season = int(race.get("season", 0))
                if season < 2014:
                    continue

                round_no = int(race.get("round", 0))
                race_id = f"{season}_{round_no}"

                for qual in race.get("QualifyingResults", []):
                    q1_ms = parse_time_to_ms(qual.get("Q1"))
                    q2_ms = parse_time_to_ms(qual.get("Q2"))
                    q3_ms = parse_time_to_ms(qual.get("Q3"))

                    valid_times = [t for t in [q1_ms, q2_ms, q3_ms] if t is not None]
                    best_q_ms = min(valid_times) if valid_times else np.nan

                    qual_list.append(
                        {
                            "raceId": race_id,
                            "driverId": qual.get("Driver", {}).get("driverId"),
                            "qualifying_pos": pd.to_numeric(
                                qual.get("position"), errors="coerce"
                            ),
                            "q1_ms": q1_ms,
                            "q2_ms": q2_ms,
                            "q3_ms": q3_ms,
                            "driver_best_q_ms": best_q_ms,
                        }
                    )

    df = pd.DataFrame(qual_list).drop_duplicates(subset=["raceId", "driverId"])
    df.to_parquet(QUALIFYING_PATH, index=False)
    logger.info("Saved: %s | Shape: %s", QUALIFYING_PATH, df.shape)
    return df


def process_sprints() -> pd.DataFrame:
    # (Remains identical to original implementation)
    logger.info("--- Parsing Fact Sprints ---")
    sprint_list: list[dict[str, Any]] = []

    for file in os.listdir(BRONZE_DIR):
        if file.endswith(".json") and "sprint" in file:
            filepath = os.path.join(BRONZE_DIR, file)
            with open(filepath, "r", encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)

            races: list[dict[str, Any]] = (
                data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            )
            for race in races:
                season = int(race.get("season", 0))
                if season < 2014:
                    continue

                round_no = int(race.get("round", 0))
                race_id = f"{season}_{round_no}"

                for sprint in race.get("SprintResults", []):
                    sprint_list.append(
                        {
                            "raceId": race_id,
                            "driverId": sprint.get("Driver", {}).get("driverId"),
                            "sprint_finish": pd.to_numeric(
                                sprint.get("position"), errors="coerce"
                            ),
                            "sprint_points": pd.to_numeric(
                                sprint.get("points"), errors="coerce"
                            ),
                        }
                    )

    df = pd.DataFrame(sprint_list).drop_duplicates(subset=["raceId", "driverId"])
    df.to_parquet(SPRINTS_PATH, index=False)
    logger.info("Saved: %s | Shape: %s", SPRINTS_PATH, df.shape)
    return df


if __name__ == "__main__":
    setup_logging()
    logger.info("=== SILVER LAYER PROCESSING ===")
    process_results()
    process_qualifying()
    process_sprints()