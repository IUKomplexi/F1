# F1 Predictive Modeling: Implementation Plan

## 1. Executive Task

The core objective is to predict the final race classification (`positionOrder`) using *only* data available after qualifying and strictly before the race starts (zero data leakage).

The architecture relies on a robust Bronze-Silver-Gold data pipeline, caching data locally to respect the Jolpica-F1 API limits. The predictive core will utilize a Learning-to-Rank (LTR) algorithm (`LGBMRanker` or `XGBRanker`) to handle the ordinal nature of a closed-system race grid.

---

## 2. Technical Architecture & Data Layers

The pipeline follows a strict, resource-efficient, and leakage-proof design.

### 2.1 API Ingestion (The Bronze Layer)
*   **Goal:** Create a local, immutable cache of raw API responses.
*   **Action:** Query the Jolpica-F1 API. Save every JSON response directly to disk.
*   **Rules:**
    *   Implement sleep mechanisms (e.g., a mathematical throttle of `time.sleep(0.25)`) and exponential backoff to respect rate limits.
    *   Use a custom User-Agent header (e.g., `"F1Predictor_Pipeline/1.0"`).
    *   Handle pagination by iterating the `offset` parameter until `MRData.total` is reached.
*   **Storage:** Local directory (e.g., `./data/bronze/`).

### 2.2 Data Cleansing & Normalization (The Silver Layer)
*   **Goal:** Parse the raw JSON cache, handle API quirks, and structure the data into normalized tabular formats.
*   **Action:** Extract relevant fields from endpoints (`/results`, `/qualifying`, `/driverStandings`, `/constructorStandings`, `/sprint`).
*   **Rules:**
    *   Convert Jolpica's literal null string representation (`"\N"`) to programmatic nulls (e.g., `np.nan`) globally to prevent mathematical distortion.
    *   Drop all post-race/leakage fields immediately (e.g., `laps`, `time`, `fastestLap`, `statusId`, `pitstops`).
    *   Convert raw qualifying lap time strings into milliseconds.
    *   Create base relational tables (e.g., `dim_drivers`, `dim_races`, `fact_qualifying`).
*   **Storage:** Local Parquet files (`./data/silver/`).

### 2.3 Feature Engineering (The Gold Layer)
*   **Goal:** Synthesize the final feature matrix (`X`) and target variable (`Y`) for model training, isolating driver skill from machinery performance.
*   **Action:** Join Silver layer tables and calculate complex, derived metrics.
*   **Rules:**
    *   **The Target:** `Y = positionOrder` (Integer, guarantees full sequence even for DNFs/DNS).
    *   **Temporal Boundary:** Ensure features like championship standings are joined strictly using `round - 1`.
    *   **Format:** Export exclusively as Parquet (`.parquet`) to preserve data types.
*   **Storage:** `./data/gold/f1_feature_matrix.parquet`.

---

## 3. Step-by-Step Implementation Roadmap

### Phase 1: Pipeline Setup & Bronze Extraction
1.  **Define Entities:** Query the `/2026/drivers` and `/2026/constructors` endpoints to get the active grid.
2.  **Historical Extraction:** Loop through each active `driverId` and `constructorId`.
3.  **Endpoint Iteration:** Fetch `/results`, `/qualifying`, `/driverStandings`, and `/constructorStandings` for their entire career history.
4.  **Save to Disk:** Store all JSONs in the Bronze layer.

### Phase 2: Silver Layer Processing
1.  **Parse Results:** Extract `raceId` (season + round), `driverId`, `constructorId`, `circuitId`, `grid` (starting position), and the target `positionOrder`. Drop all other in-race metrics.
2.  **Parse Qualifying:** Extract `position` (qualifying rank) and raw lap times (`q1`, `q2`, `q3`). Convert times to milliseconds.
3.  **Parse Standings:** Extract `points` and `position` for both drivers and constructors.
4.  **Parse Sprints (if applicable):** Extract sprint `positionOrder`.

### Phase 3: Gold Layer Feature Engineering
This phase calculates the pre-race signals, relative team dynamics, and handles edge cases.

1.  **Pace Delta Calculation (`pace_delta_pct`):** 
    *   *Formula:* `(driver_best_q_time - pole_q3_time) / pole_q3_time * 100`.
    *   *Crash/Missing Data Imputation:* If a driver crashes and has no Q time (API returns `"\N"`), impute using their teammate's pace delta + a standard penalty margin, or use the slowest time of the session + margin.
2.  **Team-Internal Performance Relativization:**
    *   *Pace Gap:* Calculate `teammate_delta_ms` (`t_driver - t_teammate`) to isolate driver talent from car pace.
    *   *Teammate Head-to-Head Form (`teammate_h2h_form`):* An EWMA of the team-internal battle. Evaluates to `1` if the driver finished ahead of the teammate in `positionOrder`, and `0` if behind.
    *   *Point Contribution:* Calculate `team_point_contribution_pct` (Driver Points / Total Team Points) to define the clear #1 driver dynamically.
3.  **Grid Penalty Indicator:**
    *   *Formula:* `grid - qualifying_position`. (Positive values indicate a penalty).
4.  **Sprint Imputation:**
    *   If `is_sprint_weekend == 1`, use the sprint `positionOrder`.
    *   If `is_sprint_weekend == 0`, impute the sprint feature with the driver's `qualifying_position` for that weekend. Do not use 0.
55.  **Historical Form (EWMA - Separated):**
    *   *Driver Form:* Calculate an Exponentially Weighted Moving Average (e.g., alpha=0.4) of the driver's last 3 finishing positions.
    *   *Constructor Form (`constructor_form_ewma`):* EWMA ($\\alpha=0.4$) of the aggregated team points to capture pure machinery upgrade cycles. Prevents team changes (e.g., Hamilton to Ferrari) from polluting car performance metrics.
    *   *Rookie Cold Start:* Impute rookie EWMA with the historical median of all F1 rookies over their first 3 races. Let the model rely on the team's constructor form proxy for car potential. Do not use 0 or leave as NaN.
6.  **Circuit & Environmental Profiling:**
    *   *Track Position Retention Index:* Spearman rank correlation between `grid` and `positionOrder` over the past 5 years at the specific circuit (e.g., high for Monaco, low for Spa).
    *   *Regulatory Era Tag:* Categorical encoding based on technical regulations ("V6_Hybrid", "2026_Reset") to insulate against structural historical data breaks.
7.  **Lagged Standings Join:**
    *   Join Driver and Constructor standings specifically using `races.round - 1`.

### Phase 4: Model Training (Learning-to-Rank)
1.  **Algorithm Choice:** Use `LightGBM (LGBMRanker)` with `lambdarank` loss, specifically designed to optimize ordinal placement within a defined group.
2.  **Group Definition:** The model must rank drivers *within a specific race*.
    *   Create a `group` array defining the number of entrants per `raceId`. Calculate this dynamically (e.g., `X.groupby('raceId').size()`). Do not hardcode to 20, as grid sizes vary historically and DNS (Did Not Start) cases exist.
3.  **Feature Types:** Define `circuitId`, `driverId`, and `constructorId` as categorical features within LightGBM. Add `regulatory_era` (e.g., "V6_Hybrid", "2026_Reset") as a categorical tag.
4.  **Validation & Chronological Split:** 
    *   First, execute a simple Random Forest regression/classification as a pure pipeline sanity check ("Smoke Test"). A suspiciously high accuracy immediately exposes a data leakage field.
    *   For final validation, use a strict chronological time-series split (e.g., train on data up to 2025, evaluate performance exclusively on the live races of the 2026 season). Do not use random cross-validation.
4.  **Evaluation:** Use NDCG (Normalized Discounted Cumulative Gain) or MAP (Mean Average Precision) focused on the Top 3 / Top 10 positions.

---

## 4. Final Feature Schema (Parquet / Gold Layer / (!Draft)

| Feature Name | Data Type | Source / Engineering Logic | Risk / Notes |
| :--- | :--- | :--- | :--- |
| `raceId` | String | Base identifier (`Season_Round`) | Grouping key for LTR. |
| `season` | Int | API: `/races` | Temporal marker. |
| `round` | Int | API: `/races` | Chronological race index. |
| `circuitId` | Category | API: `/races` | Categorical track encoding. |
| `driverId` | Category | API: `/drivers` | Categorical driver encoding. |
| `constructorId` | Category | API: `/constructors` | Categorical team encoding. |
| `grid` | Int | API: `/results` | Official start pos (post-penalties). |
| `qualifying_pos` | Int | API: `/qualifying` | Raw speed rank. |
| `grid_penalty_delta` | Int | `grid - qualifying_pos` | Captures out-of-position fast cars. |
| `pace_delta_pct` | Float | `(Q_time - Pole_time)/Pole_time` | Requires imputation for `\N`. |
| `teammate_delta_ms` | Float | `t_driver - t_teammate` | Relativizes qualifying pace against the same car. |
| `teammate_h2h_form` | Float | EWMA of team-internal finishing battle | Determines internal No.1 vs No.2 status. |
| `is_sprint_weekend`| Int/Bool| API: Schedule check | 1 if sprint, 0 otherwise. |
| `sprint_finish` | Int | API: `/sprint` | Impute with `qualifying_pos` if 0. |
| `driver_pts_lag` | Float | API: `/driverStandings` | Must be `round - 1`. |
| `team_pts_lag` | Float | API: `/constructorStandings`| Must be `round - 1`. |
| `driver_form_ewma` | Float | Calculated (Past 3 races, driver only) | Driver consistency; rookie fallback applied. |
| `constructor_form_ewma`| Float | Calculated (Past 3 races, team points) | Machinery performance; insulates team swaps. |
| `track_retention_idx` | Float | Spearman rank correlation (Past 5 years) | Track profiling metric (Monaco vs Spa). |
| `regulatory_era` | Category | Map based on `season` | Accounts for major rule-change resets. |
| **`target_pos`** | **Int** | **API: `/results (positionOrder)`** | **The variable to predict (Y).** |

---

## 5. Summary Mantra

*   **First:** Establish a stable, leakage-free data flow (Bronze -> Silver -> Gold) with strict programmatic `\N` null conversion.
*   **Second:** Implement advanced feature engineering that isolates and relativizes driver performance against their teammate (`teammate_delta_ms`, `teammate_h2h_form`).
*   **Third:** Train the Learning-to-Rank model with dynamic groups and enforce strict chronological validation after a pipeline smoke test.
