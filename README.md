# F1 Race-Outcome Prediction Pipeline

Predicts F1 race finishing positions using a lightgbm LambdaRank model trained
on data fetched from the Ergast/F1 API (via jolpi.ca).

## Structure

```
config.py                 # future single source of truth for paths/constants (stub)
run_pipeline.py           # future end-to-end orchestrator (stub)
src/                      # production Python modules
  bronze.py               # API fetch + checkpointing (raw JSON cache)
  silver.py               # 3 parquet fact tables from bronze
  gold.py                 # gold feature matrix (training input)
  model_utils.py          # relevance mapping, PU manufacturer mapping, metrics
  decay_params.py         # generated EWMA decay params (grid-search artifact)
  train_clean.py          # clean v3 LGBMRanker train + eval
  cross_validate.py       # chronological cross-validation
  optimize_decay.py       # EWMA decay parameter grid search
  two_stage.py            # two-stage DNF / rank model
  smoketest.py            # quick diagnostic model
  legacy/                 # superseded/stale scripts (deprecated, see below)
tests/                    # placeholder test suite
data/                     # bronze/silver/gold (gitignored)
docs/                     # live docs; archive/ holds deprecated PLAN docs
```

## Pipeline stages

1. **Bronze** — `python src/bronze.py` fetches raw race/driver data into
   `data/bronze/*.json` with checkpointing.
2. **Silver** — `python src/silver.py` parses bronze into three parquet facts
   under `data/silver/`: `fact_results`, `fact_qualifying`, `fact_sprints`.
3. **Gold** — `python src/gold.py` builds the feature matrix
   `data/gold/f1_feature_matrix.parquet`.
4. **Train** — `python src/train_clean.py` trains/evaluates the clean v3 model
   on the 2026 test set.

## Notes

- The data directory is gitignored; regenerate with the bronze/silver/gold
  stages if absent.
- `src/legacy/` holds `train_model.py` and `evaluate_baselines.py`, which are
  superseded or stale against the current gold schema. Do not rely on them.
