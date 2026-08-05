# AGENTS.md — F1 Race-Outcome Prediction Pipeline

Python pipeline that predicts F1 race finishing positions with a LightGBM
LambdaRank model trained on Ergast/jolpi.ca API data (hybrid era 2014+).

## Project

- Stack: Python 3.14, `lightgbm`, `pandas`, `numpy`, `scikit-learn`, `pyarrow`
- Venv: `.venv/` (already created, deps installed)
- Entry points: `src/bronze.py` → `src/silver.py` → `src/gold.py` →
  `src/train_clean.py` (each is a standalone script with `if __name__ == "__main__"`)
- Live plan: `docs/REFACTOR_PLAN.md` (archived old plans: `docs/archive/PLAN/`)

## Commands

All scripts run from the **repo root** (data paths are CWD-relative):

```bash
.venv/Scripts/python.exe src/bronze.py            # fetch raw JSON → data/bronze/ (checkpointed)
.venv/Scripts/python.exe src/silver.py            # parse → data/silver/*.parquet (3 facts)
.venv/Scripts/python.exe src/gold.py              # features → data/gold/f1_feature_matrix.parquet
.venv/Scripts/python.exe src/train_clean.py       # train + evaluate clean v3 model (test = 2026)
.venv/Scripts/python.exe src/cross_validate.py    # chronological CV 2020–2025
.venv/Scripts/python.exe src/optimize_decay.py    # EWMA decay grid search → src/decay_params.py
.venv/Scripts/python.exe src/smoketest.py         # leakage diagnostic
```

- No build step, no linter configured, no test runner yet (tests/ are empty
  stubs; pytest planned — see REFACTOR_PLAN phase 5)
- `data/` is gitignored: parquet regeneration requires running silver+gold
  after bronze; bronze is already populated (826 cached JSON files)

## Architecture

- `src/config.py` — single source of truth: API constants, all data paths,
  season bounds, EWMA decay params (from `decay_params.py` artifact),
  imputation constants, `setup_logging()`. Imports it into its own `sys.path`
  bootstrap so root scripts can import `src` modules.
- `src/bronze.py` — checkpointed paginated API fetch into `data/bronze/`.
- `src/silver.py` — 3 near-identical parse loops → `fact_results` /
  `fact_qualifying` / `fact_sprints` parquet.
- `src/gold.py` — merges silver facts; ~250-line `generate_gold_features`
  monolith → `f1_feature_matrix.parquet` (28 columns incl. target
  `positionOrder`, `status`, `is_classified`).
- `src/model_utils.py` — shared ML code: canonical `FEATURES` (17 clean-v3
  features) / `TARGET` / `CAT_COLS`, `make_ranker()`, `evaluate_predictions()`,
  `compute_decayed_ewma()`, relevance/points mappings, PU manufacturer map,
  `classify_status`, `parse_time_to_ms` (in silver.py).
- `src/decay_params.py` — generated artifact `BEST_DECAY_PARAMS` (grid-search
  output; consumed by config.py).
- `src/legacy/` — stale/broken scripts (`train_model.py`,
  `evaluate_baselines.py`; `two_stage.py` still to be moved here): reference
  features absent from the gold schema, do not run them.
- `run_pipeline.py`, `tests/` — empty placeholders (planned).

## Conventions

- **Never hardcode paths or magic numbers in modules** — import from
  `src/config.py` (paths, seasons, decay, imputation constants).
- **Flat imports**: `from config import ...` / `from model_utils import ...` —
  works because scripts run from within `src/`; do not convert to package
  relative imports unless the run model changes.
- **`FEATURES` in `model_utils.py` is the canonical feature contract** — the
  gold parquet and every training script must match it; scripts must not
  define their own feature lists.
- **Logging, not print** — call `setup_logging()` in `__main__` blocks and use
  module-level `logger = logging.getLogger(__name__)`.
- **Type hints** on public functions; `pandas`/`numpy` idioms throughout.
- **Run from repo root** — `data/` resolves relative to CWD (running from
  `src/` creates an empty `src/data/`).
- Refactor in progress: check `docs/REFACTOR_PLAN.md` for status before
  touching modules; verify behavior against existing parquet outputs.

## Notes

- (empty — add quick facts here as they surface)
