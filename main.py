import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

# --- Configuration ---
BASE_URL = "https://api.jolpi.ca/ergast/f1/"
BRONZE_DIR = "./data/bronze"
HEADERS = {
    "User-Agent": "F1_DATA&ANALYTICS_UNI/1.0",
    "Accept": "application/json",
}
THROTTLE_DELAY = 0.5

# Check if dir exists
os.makedirs(BRONZE_DIR, exist_ok=True)


def fetch_paginated_endpoint(endpoint_path: str, filename_prefix: str):
    """Fetch Data and create JSON Cache"""
    limit = 100
    offset = 0
    total = None
    endpoint_path = endpoint_path.lstrip("/")

    while total is None or offset < total:
        time.sleep(THROTTLE_DELAY)  # Rate Limits
        
        # Create URL query 
        params = urllib.parse.urlencode({"limit": limit, "offset": offset})
        url = f"{BASE_URL}{endpoint_path}?{params}"

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
                    print(
                        f"    [Warning] Invalid request for endpoint '{endpoint_path}', skipping."
                    )
                    return
                if e.code in [429, 500, 502, 503, 504] and attempt < retries - 1:
                    retry_after = e.headers.get("Retry-After")
                    wait_for = int(retry_after) if retry_after and retry_after.isdigit() else backoff
                    print(
                        f"[Warning] HTTP {e.code} on attempt {attempt+1}. Retrying in {wait_for}s..."
                    )
                    time.sleep(wait_for)
                    backoff = max(backoff * 2, wait_for * 2)
                else:
                    print(f"    [Error] HTTP Failure: {e.code} - {e.reason}")
                    return
            except urllib.error.URLError as e: # Catch the specific network error
                print(f"Network error occurred: {e}")
    

        if not response_data:
            break

        mr_data = response_data.get("MRData", {})
        total = int(mr_data.get("total", 0))

        # Save JSON payload to disk
        safe_filename = f"{filename_prefix}_offset_{offset}.json".replace(
            "/", "_"
        )
        filepath = os.path.join(BRONZE_DIR, safe_filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(response_data, f, indent=4)

        print(f" Saved: {filepath} ({offset}/{total})")

        if total == 0:
            break

        offset += limit


def get_active_entities(year: int = 2026):
    """ Get active grid, parse Grid """
    print(f"\n---Active Grid for {year} ---")

    fetch_paginated_endpoint(f"/{year}/drivers", f"active_drivers_{year}")
    fetch_paginated_endpoint(
        f"/{year}/constructors", f"active_constructors_{year}"
    )

    active_drivers: set[str] = set()
    active_constructors: set[str] = set()


    # Parse through cache
    for file in os.listdir(BRONZE_DIR):
        file_path = os.path.join(BRONZE_DIR, file)
        if file.startswith(f"active_drivers_{year}"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for d in data["MRData"]["DriverTable"]["Drivers"]:
                    active_drivers.add(d["driverId"])

        elif file.startswith(f"active_constructors_{year}"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for c in data["MRData"]["ConstructorTable"]["Constructors"]:
                    active_constructors.add(c["constructorId"])

    return list(active_drivers), list(active_constructors)


def main():
    print("=== STANDARD LIB PIPELINE SETUP ===")

    driver_ids, constructor_ids = get_active_entities(year=2026)
    print(
        f"Grid Analysis: Found {len(driver_ids)} drivers and {len(constructor_ids)} constructors active for 2026."
    )

    print("\n--- Extracting Historical Driver Records ---")
    driver_endpoints = ["/results", "/qualifying"]
    for driver in driver_ids:
        print(f" Processing Profile: {driver}")
        for endpoint in driver_endpoints:
            path = f"/drivers/{driver}/{endpoint.strip('/')}"
            prefix = f"driver_{driver}_{endpoint.strip('/')}"
            fetch_paginated_endpoint(path, prefix)

    print("\n--- Extracting Historical Constructor Records ---")
    constructor_endpoints = [
        "/results",
        "/qualifying",
    ]
    for constructor in constructor_ids:
        print(f" Processing Profile: {constructor}")
        for endpoint in constructor_endpoints:
            path = f"/constructors/{constructor}/{endpoint.strip('/')}"
            prefix = f"constructor_{constructor}_{endpoint.strip('/')}"
            fetch_paginated_endpoint(path, prefix)

    print("\n=== COMPLETE:Cache Is Ready ===")

# --- Phase 2 ---
# --- Phase 3 ---
# --- Phase 4 ---
if __name__ == "__main__":
    main()
