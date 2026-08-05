"""End-to-end pipeline orchestrator.

Placeholder stub for the future orchestrator that chains the whole run:

    bronze (fetch/checkpoint) -> silver (facts) -> gold (features) -> train

Planned CLI:
    python run_pipeline.py --stage all            # full pipeline
    python run_pipeline.py --stage gold           # rebuild gold features only
    python run_pipeline.py --stage train --season 2026

Implementation will dispatch to the modules under src/ once the moving
refactor lands. Intentionally empty for now.
"""
