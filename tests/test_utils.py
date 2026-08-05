"""Unit tests for shared helper logic.

Planned coverage:
    - parse_time_to_ms: "1:23.456", "23.456", "\\N", None, garbage strings
    - classify_status: Finished / Lap / DNF / DNS / DSQ statuses
    - position_to_relevance / get_label_gain / expected_points_error
    - compute_decayed_ewma math (once extracted to a shared module)
"""
