# Refactor Plan — F1 Prediction Codebase

Status of this document: **live plan** for the readability & maintainability
refactor of the F1 race-outcome prediction pipeline. Supersedes the archived
plans in `docs/archive/PLAN/`.

Progress legend:
- ✅ **DONE** — verified complete
- 🔶 **IN PROGRESS / PARTIAL** — started, remaining steps listed
- ⬜ **TODO** — not started

---

## 1. Centralize config + constants (`src/config.py`) — ✅ DONE

- ✅ Create `src/config.py` as the single source of truth:
  - API constants: `BASE_URL`, `THROTTLE_DELAY`, `HEADERS`
  - Data paths: `BRONZE_DIR`, `SILVER_DIR`, `GOLD_DIR`, `RESULTS_PATH`,
    `QUALIFYING_PATH`, `SPRINTS_PATH`, `GOLD_PATH`
  - Season bounds: `FIRST_SEASON` (2014), `TEST_SEASON` (2026),
    `VAL_SEASONS` (2020–2025), `REGULATION_RESET_SEASONS`, `SUMMER_BREAK_ROUND`
  - EWMA decay parameters imported from the grid-search artifact
    `src/decay_params.BEST_DECAY_PARAMS`, with hardcoded fallbacks
  - Named imputation constants: `DNF_RATE_FALLBACK` (0.17),
    `RETENTION_FALLBACK` (0.55), `TEAMMATE_DELTA_FILL` (1.5),
    `PACE_DELTA_FALLBACK` (5.0), `EWMA_ALPHA` (0.4), window sizes
- ✅ Add `setup_logging()` helper in `config.py`; sys.path bootstrap for `src/`
- ✅ Wire every production module to import from `config` (bronze, silver, gold,
  model_utils, train_clean, cross_validate, optimize_decay, smoketest)
- ✅ Replace `print` diagnostics with `logging` in the above modules
- ✅ Delete the old root-level `config.py` stub
- ⬜ Remaining: `src/two_stage.py` still hardcodes `GOLD_PATH` (it is being
  retired in Phase 4, so it stays as-is until then)
- Verify: no `"./data/..."` hardcodes remain in `src/*.py` (except
  `two_stage.py`), all scripts import cleanly

## 2. De-duplicate shared ML code into `model_utils.py` — 🔶 PARTIAL

- ✅ Add canonical `FEATURES` (17-feature clean v3 list), `TARGET`, `CAT_COLS`
  to `model_utils.py`
- ✅ Add `make_ranker(...)` LGBMRanker factory
- ✅ Add `evaluate_predictions(...)` benchmark helper (Expected Pts Error,
  P1/Top3/Top10 acc, Spearman rho, MAE)
- ✅ Move `compute_decayed_ewma(...)` into `model_utils.py`; `gold.py` now calls
  it (kills the duplicated EWMA/decay block)
- ⬜ Rewire `train_clean.py` to import `FEATURES`/`TARGET`/`make_ranker`/
  `evaluate_predictions` from `model_utils`, deleting its local copies
- ⬜ Rewire `optimize_decay.py` to import `FEATURES` from `model_utils`
- ⬜ Rewire `cross_validate.py` to use shared `FEATURES` + `make_ranker`
- Verify: run `python src/train_clean.py` end-to-end (gold parquet exists) and
  confirm identical output vs. pre-refactor behavior

## 3. Refactor `silver.py` and `gold.py` — ⬜ TODO

- `silver.py`: extract one generic `parse_bronze_files(file_keyword,
  row_extractor)` helper; rewrite the 3 near-identical loops
  (`process_results` / `process_qualifying` / `process_sprints`) on top of it
- `gold.py`: split the ~250-line `generate_gold_features` monolith into named
  builders — `add_teammate_features`, `add_form_ewma`, `add_standings_features`,
  `add_reliability_features`, `add_track_retention`, `finalize_gold_matrix` —
  with the pipeline function only orchestrating
- Verify: rebuild gold from existing bronze; assert regenerated parquet matches
  current one (same columns, same row count, no NaN in FEATURES)

## 4. Retire/delete stale scripts; add CLI + orchestrator — ⬜ TODO

- `git mv src/two_stage.py` → `src/legacy/two_stage.py` (broken against the
  gold schema; decision: retire to legacy)
- `git rm src/legacy/train_model.py` + `src/legacy/evaluate_baselines.py`
  (superseded by `train_clean.py`; decision: delete)
- Update `docs/archive/PLAN/README.md` / legacy marker to reflect new contents
- Fix `optimize_decay.py` output path → `src/decay_params.py`
- Add lightweight `argparse` CLI (`--season`/`--paths`, defaults preserve
  current behavior) to `train_clean.py`, `cross_validate.py`,
  `optimize_decay.py`
- Implement `run_pipeline.py --stage bronze|silver|gold|train|all` (bootstraps
  `src/` on `sys.path`, dispatches to existing functions)
- Verify: `python run_pipeline.py --help` and `python src/train_clean.py
  --help` work; `git status` shows the moves/deletions

## 5. Automated checks + docs — ⬜ TODO

- Add `pytest` to `requirements.txt`; add root `conftest.py` putting `src/` on
  `sys.path` for the flat imports
- `tests/test_schema.py`: load gold parquet, assert it contains exactly the
  canonical `FEATURES` + target columns (kills the stale-script class of bug)
- `tests/test_utils.py`: unit tests for `parse_time_to_ms`, `classify_status`,
  `position_to_relevance`, `compute_decayed_ewma`
- Update `README.md` structure section (config in `src/`, new legacy contents,
  test command)
- Verify: `./.venv/Scripts/python.exe -m pytest tests/ -q` passes; README
  matches final tree

---

## Decisions locked in

- `src/two_stage.py` → **retire to `src/legacy/`** (not fixed)
- `src/legacy/train_model.py`, `evaluate_baselines.py` → **delete**
- pandas **stay on 3.0.5**; pin `pandas<3` only if a stage breaks

## Assumptions

- `config.py` lives in `src/` (scripts run as `python src/x.py`; root entry
  points bootstrap `sys.path` via `config.py`)
- `pytest` is the test runner
- `optimize_decay.py`'s stale write path (`./best_decay_params.py`) is fixed to
  `src/decay_params.py`
