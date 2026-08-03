import json
import os
from typing import Any

import numpy as np
import pandas as pd

BRONZE_DIR = "./data/bronze"
SILVER_DIR = "./data/silver"

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
    print("--- Parsing Fact Results ---")
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
                        }
                    )

    df = pd.DataFrame(results_list)
    df.replace(to_replace=r"\N", value=np.nan, inplace=True)
    
    # Deduplicate across combined Driver/Constructor JSON caches
    df = df.drop_duplicates(subset=["raceId", "driverId"]).reset_index(drop=True)
    
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
    df["positionOrder"] = pd.to_numeric(df["positionOrder"], errors="coerce")
    df["points"] = pd.to_numeric(df["points"], errors="coerce")

    output_path = os.path.join(SILVER_DIR, "fact_results.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Saved: {output_path} | Shape: {df.shape}")
    return df


def process_qualifying() -> pd.DataFrame:
    print("--- Parsing Fact Qualifying ---")
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
    output_path = os.path.join(SILVER_DIR, "fact_qualifying.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Saved: {output_path} | Shape: {df.shape}")
    return df


def process_sprints() -> pd.DataFrame:
    print("--- Parsing Fact Sprints ---")
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
    output_path = os.path.join(SILVER_DIR, "fact_sprints.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Saved: {output_path} | Shape: {df.shape}")
    return df


if __name__ == "__main__":
    print("=== SILVER LAYER PROCESSING ===")
    process_results()
    process_qualifying()
    process_sprints()