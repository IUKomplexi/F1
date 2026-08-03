import json
import os
from typing import Any

import numpy as np
import pandas as pd

BRONZE_DIR = "./data/bronze"
SILVER_DIR = "./data/silver"

# Ensure Silver directory exists
os.makedirs(SILVER_DIR, exist_ok=True)


def process_results() -> pd.DataFrame:
    print("--- Parsing Fact Results ---")
    results_list: list[dict[str, Any]] = []

    # Iterate through all files in the Bronze cache
    for file in os.listdir(BRONZE_DIR):
        if file.endswith(".json") and "results" in file:
            filepath = os.path.join(BRONZE_DIR, file)

            with open(filepath, "r", encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)

            mr_data: dict[str, Any] = data.get("MRData", {})
            race_table: dict[str, Any] = mr_data.get("RaceTable", {})
            races: list[dict[str, Any]] = race_table.get("Races", [])

            for race in races:
                season = int(race.get("season", 0))

                # 1. Enforce Modern Era Cutoff (Phase 2 Data Cleansing)
                if season < 2014:
                    continue

                round_no = int(race.get("round", 0))
                circuit_dict: dict[str, Any] = race.get("Circuit", {})
                circuit_id = circuit_dict.get("circuitId")

                # Create unique grouping key for LTR model
                race_id = f"{season}_{round_no}"

                results: list[dict[str, Any]] = race.get("Results", [])
                for result in results:
                    driver_dict: dict[str, Any] = result.get("Driver", {})
                    constructor_dict: dict[str, Any] = result.get("Constructor", {})

                    # 2. Extract only pre-race signals, target, and points
                    row: dict[str, Any] = {
                        "raceId": race_id,
                        "season": season,
                        "round": round_no,
                        "circuitId": circuit_id,
                        "driverId": driver_dict.get("driverId"),
                        "constructorId": constructor_dict.get("constructorId"),
                        "grid": result.get("grid"),  # Official start pos
                        "positionOrder": result.get("position"),  # Target variable (Y)
                        "points": result.get("points"),  # Retained for standings math
                    }
                    results_list.append(row)

    df = pd.DataFrame(results_list)

    # /N to np.nan
    df.replace(to_replace=r"\N", value=np.nan, inplace=True)

    # Type cast for mathematical safety
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
    df["positionOrder"] = pd.to_numeric(df["positionOrder"], errors="coerce")
    df["points"] = pd.to_numeric(df["points"], errors="coerce")

    # 4. Save to Silver Layer as Parquet
    output_path = os.path.join(SILVER_DIR, "fact_results.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Saved: {output_path} | Shape: {df.shape}")

    return df


if __name__ == "__main__":
    print("=== SILVER LAYER PROCESSING ===")
    df_results = process_results()