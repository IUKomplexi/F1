"""Central configuration — single source of truth for the F1 pipeline.

In the flat-layout cleanup this module is a placeholder stub. The moving refactor
will graduate paths, season bounds, decay parameters and named constants here so
the moving modules stop hardcoding magic numbers.

Expected contents (filled during the config refactor step):
    - Data paths: data/bronze, data/silver, data/gold + parquet filenames
    - Season bounds: first season, test season (2026), validation seasons
    - EWMA decay parameters imported from src.decay_params (not duplicated)
    - Named constants for imputation values (DNF rate 0.17, retention 0.55, etc.)
    - Logging setup helper

DO NOT put runtime logic here; import from this module everywhere instead.
"""
