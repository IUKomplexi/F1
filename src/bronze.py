import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

# --- Config ---
BASE_URL = "https://api.jolpi.ca/ergast/f1/"
BRONZE_DIR = "./data/bronze"
HEADERS = {
    "User-Agent": "F1_DATA&ANALYTICS_UNI/1.0",
    "Accept": "application/json",
}
THROTTLE_DELAY = 0.5

os.makedirs(BRONZE_DIR, exist_ok=True)


def fetch_paginated_endpoint(endpoint_path: str, filename_prefix: str):
    """Fetch Data and create JSON Cache"""
    limit = 100
    offset = 0
    total = None
    endpoint_path = endpoint_path.lstrip("/")

    while total is None or offset < total:
        
        # Checkpoint
        safe_filename = f"{filename_prefix}_offset_{offset}.json".replace("/", "_")
        filepath = os.path.join(BRONZE_DIR, safe_filename)

        if os.path.exists(filepath):
            print(f"    [Checkpoint] File exists: {filepath}. Skipping API call.")
            with open(filepath, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            
            mr_data = cached_data.get("MRData", {})
            total = int(mr_data.get("total", 0))
            offset += limit
            continue # Skip rest jump to offset
        
        time.sleep(THROTTLE_DELAY)  # Rate Limits

        # Create URL query
        params = urllib.parse.urlencode({"limit": limit, "offset": offset})
        url = f"{BASE_URL}{endpoint_path}.json?{params}"

        req = urllib.request.Request(url, headers=HEADERS)

        # Retry loop for HTTP errors
        retries = 5
        backoff = 2
        response_data = None

        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req) as response:
                    response_data = json.loads(response.read().decode("utf-8"))
                    break
            except urllib.error.HTTPError as e:
                if e.code == 400:
                    print(f"    [Warning] Invalid request for endpoint '{endpoint_path}', skipping.")
                    return
                if e.code in [429, 500, 502, 503, 504] and attempt < retries - 1:
                    retry_after = e.headers.get("Retry-After")
                    wait_for = (
                        int(retry_after)
                        if retry_after and retry_after.isdigit()
                        else backoff
                    )
                    print(f"[Warning] HTTP {e.code} on attempt {attempt + 1}. Retrying in {wait_for}s...")
                    time.sleep(wait_for)
                    backoff = max(backoff * 2, wait_for * 2)
                else:
                    print(f"    [Error] HTTP Failure: {e.code} - {e.reason}")
                    return
            except urllib.error.URLError as e:
                print(f"Network error occurred: {e}")

        if not response_data:
            break

        mr_data = response_data.get("MRData", {})
        total = int(mr_data.get("total", 0))

        # Save JSON payload to disk
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(response_data, f, indent=4)

        print(f" Saved: {filepath} ({offset}/{total})")

        if total == 0:
            break

        offset += limit


def get_active_entities(year: int = 2026):
    """Get active grid, parse Grid"""
    print(f"\n--- Active Grid for {year} ---")

    fetch_paginated_endpoint(f"/{year}/drivers", f"active_drivers_{year}")
    fetch_paginated_endpoint(f"/{year}/constructors", f"active_constructors_{year}")

    active_drivers: set[str] = set()
    active_constructors: set[str] = set()

    for file in os.listdir(BRONZE_DIR):
        file_path = os.path.join(BRONZE_DIR, file)
        if file.startswith(f"active_drivers_{year}") and file.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for d in data.get("MRData", {}).get("DriverTable", {}).get("Drivers", []):
                    active_drivers.add(d["driverId"])

        elif file.startswith(f"active_constructors_{year}") and file.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for c in data.get("MRData", {}).get("ConstructorTable", {}).get("Constructors", []):
                    active_constructors.add(c["constructorId"])

    return list(active_drivers), list(active_constructors)


def get_historical_races(driver_ids: list[str], start_year: int = 2014) -> set[tuple[int, int]]:
    """Fetch all races participated in by the active drivers to build a master race list."""
    print("\n--- Compiling Master Race List ---")
    unique_races: set[tuple[int, int]] = set()

    for driver in driver_ids:
        path = f"/drivers/{driver}/races"
        prefix = f"driver_{driver}_races"
        fetch_paginated_endpoint(path, prefix)

        # Parse the saved race lists to extract (season, round)
        for file in os.listdir(BRONZE_DIR):
            if file.startswith(prefix) and file.endswith(".json"):
                filepath = os.path.join(BRONZE_DIR, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
                    for race in races:
                        season = int(race.get("season", 0))
                        round_no = int(race.get("round", 0))
                        
                        # We only need the Hybrid Era onwards for our features
                        if season >= start_year:
                            unique_races.add((season, round_no))
                            
    return unique_races


def main():
    print("=== BRONZE LAYER PIPELINE: NO SURVIVORSHIP BIAS ===")

    driver_ids, _ = get_active_entities(year=2026)
    print(f"Grid Analysis: Found {len(driver_ids)} drivers active for 2026.")

    master_races = get_historical_races(driver_ids, start_year=2014)
    print(f"Master Race List: Found {len(master_races)} unique hybrid-era races featuring active drivers.")

    print("\n--- Extracting Full Grid Historical Records ---")
    
    # Sort chronologically to maintain sanity in the logs
    for season, round_no in sorted(master_races):
        print(f"\n Processing Race: {season} Round {round_no}")
        
        # 1. Full Race Results
        fetch_paginated_endpoint(
            f"/{season}/{round_no}/results", 
            f"race_{season}_{round_no}_results"
        )
        
        # 2. Full Qualifying Results
        fetch_paginated_endpoint(
            f"/{season}/{round_no}/qualifying", 
            f"race_{season}_{round_no}_qualifying"
        )
        
        # 3. Full Sprint Results (Ergast gracefully returns empty if no sprint occurred)
        fetch_paginated_endpoint(
            f"/{season}/{round_no}/sprint", 
            f"race_{season}_{round_no}_sprint"
        )

    print("\n=== COMPLETE: Full Historical Grids Cached ===")


if __name__ == "__main__":
    main()