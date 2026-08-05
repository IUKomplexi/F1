"""Schema contract test for the gold feature matrix.

Planned (guards against the stale-script class of bug):
    - Load data/gold/f1_feature_matrix.parquet
    - Assert it contains EXACTLY the canonical FEATURES columns from
      src.model_utils (no missing, no unexpected)
    - Every script that imports FEATURES will then fail fast here if a
      feature was dropped or renamed.

Also verify the shared LGBMRanker factory and evaluate_predictions helpers
return well-formed outputs.
"""
