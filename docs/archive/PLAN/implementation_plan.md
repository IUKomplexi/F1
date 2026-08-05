# F1 Model v2: Four Critical Improvements

Upgrade the existing F1 predictive model with four targeted changes to address the weaknesses identified in the critical audit. The goal is for the LTR model to **consistently and meaningfully outperform** the naive starting-grid baseline.

---

## Proposed Changes

### 1. F1 Points-Scaled Relevance Target

#### [MODIFY] [train_model.py](file:///d:/Code/F1/train_model.py)
#### [MODIFY] [evaluate_baselines.py](file:///d:/Code/F1/evaluate_baselines.py)
#### [MODIFY] [cross_validate_model.py](file:///d:/Code/F1/cross_validate_model.py)

**Current Problem**: The LambdaRank relevance label is `30 - positionOrder`, giving the ranker equal "budget" to resolve the P1↔P2 split and the P19↔P20 split.

**Change**: Replace the linear mapping with the actual F1 championship points curve, extended with a sub-1.0 decay tail for positions outside the points:

| Pos | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20+ |
|-----|---|---|---|---|---|---|---|---|---|----|----|----|----|----|----|----|----|----|----|-----|
| Relevance | 25 | 18 | 15 | 12 | 10 | 8 | 6 | 4 | 2 | 1 | 0.5 | 0.4 | 0.3 | 0.25 | 0.2 | 0.15 | 0.1 | 0.08 | 0.05 | 0.02 |

This forces the ranker to spend its tree splits resolving the high-value top of the grid, where real competitive separation matters.

**Implementation**: Create a shared utility function `position_to_relevance(pos)` in a new file and use it across all three training scripts.

---

### 2. Two-Stage Model Architecture (DNF Survival + Classified Ranking)

#### [NEW] [train_two_stage_model.py](file:///d:/Code/F1/train_two_stage_model.py)

**Current Problem**: The model tries to predict final `positionOrder` for all drivers, but ~17% of all race entries are DNFs (mechanical failure, crashes, collisions). These are **stochastic events** that no pre-race feature can predict, yet they currently poison the ranker's loss function.

**Architecture**:

```
┌─────────────────────────────────┐
│  Stage 1: Survival Classifier   │
│  LGBMClassifier (binary)        │
│  Target: is_classified (0/1)    │
│  → Predicts P(driver finishes)  │
└───────────────┬─────────────────┘
                │ Only classified
                ▼ drivers pass
┌─────────────────────────────────┐
│  Stage 2: LTR Ranker            │
│  LGBMRanker (lambdarank)        │
│  Target: F1 points relevance    │
│  → Ranks classified finishers   │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Merge: Combine predictions     │
│  Classified → ranked by Stage 2 │
│  DNF-predicted → appended last, │
│  ordered by survival probability│
└─────────────────────────────────┘
```

**Classification target definition** (derived from `status` column already in Gold layer):
- `is_classified = 1` if status is `"Finished"` or contains `"Lap"` (lapped finishers)
- `is_classified = 0` otherwise (Retired, Collision, Engine, Brakes, DNS, DSQ, etc.)

**Stage 2 training data**: Only rows where `is_classified == 1`. Group sizes must be recomputed dynamically per race since DNF'd drivers are excluded.

**Inference pipeline**:
1. Run Stage 1 on all 20 drivers → get `P(classified)` scores
2. For drivers predicted classified (above threshold), run Stage 2 → get rank scores
3. Rank classified drivers by Stage 2 score (positions 1–N)
4. Append DNF-predicted drivers at positions N+1 onwards, ordered by descending `P(classified)`

> [!IMPORTANT]
> The existing single-stage scripts ([train_model.py](file:///d:/Code/F1/train_model.py), [evaluate_baselines.py](file:///d:/Code/F1/evaluate_baselines.py), [cross_validate_model.py](file:///d:/Code/F1/cross_validate_model.py)) will be preserved. The two-stage model is a **new** script to compare side-by-side.

---

### 3. Overtaking Difficulty Feature & Pace×Track Interaction

#### [MODIFY] [gold_feature_engineering.py](file:///d:/Code/F1/gold_feature_engineering.py)

**Your question**: *"How do I differentiate between high and low overtaking tracks?"*

**Answer**: You already have the data — `track_retention_idx` is the Spearman correlation between starting grid and finishing position over the past 5 years at each circuit. This is precisely an overtaking-difficulty proxy:

| Track Type | `track_retention_idx` | Meaning |
|:---|:---:|:---|
| **High-overtaking** (Vegas, Imola, COTA) | ~0.42–0.47 | Grid position shuffles heavily → pace matters more |
| **Moderate** (Spa, Interlagos, Red Bull Ring) | ~0.50–0.57 | Balanced |
| **Low-overtaking / processional** (Shanghai, Paul Ricard, Catalunya) | ~0.66–0.72 | Grid position mostly holds → starting pos. matters more |

**New interaction features** to add in Gold layer:

```python
# Pace advantage is worth MORE on high-overtaking tracks
df["pace_x_overtaking"]  = df["pace_delta_pct"] * (1 - df["track_retention_idx"])

# Grid advantage is worth MORE on low-overtaking tracks
df["grid_x_retention"]   = df["grid"] * df["track_retention_idx"]
```

These let the model learn that being fast but starting low is recoverable at COTA but fatal at Monaco, without hardcoding a threshold.

---

### 4. Constructor EWMA with Season Decay, Regulation Reset & Summer Break

#### [MODIFY] [gold_feature_engineering.py](file:///d:/Code/F1/gold_feature_engineering.py)

**Current Problem**: The `constructor_form_ewma` carries smoothed momentum across season boundaries with zero decay. Evidence from the data:
- Mercedes entering 2022 (new Ground Effect regs): EWMA = **31.1** (highest), but in reality they dropped to 3rd-best car
- Williams entering 2022: EWMA = **5.3e-10** (effectively zero), making early-season predictions useless

**Change**: Apply three decay mechanisms:

#### a) Off-Season Decay (γ = 0.5)
At Round 1 of each new season, halve the constructor EWMA. This acknowledges winter development uncertainty while retaining *some* historical signal:
```python
# After computing raw EWMA, apply season-start decay
is_season_start = df.groupby(['season', 'constructorId']).cumcount() == 0
df.loc[is_season_start, 'constructor_form_ewma'] *= 0.5
```

#### b) Regulation Reset Decay (γ = 0.1)
At the start of major regulation eras (`2022` Ground Effect, `2026` Reset), apply an aggressive 90% decay — essentially saying "we know almost nothing about car performance hierarchy":
```python
regulation_resets = {2022, 2026}
is_reg_reset = is_season_start & df['season'].isin(regulation_resets)
df.loc[is_reg_reset, 'constructor_form_ewma'] *= 0.1  # 0.5 * 0.1 = 0.05 total
```

#### c) Summer Break Decay (γ = 0.85)
After the summer break (approximately rounds 13-15 depending on the calendar), teams historically bring significant upgrades. Apply a mild 15% decay to the running EWMA for the post-summer round:
```python
# Summer break typically falls around round 13-15
# Apply decay to the first post-break race
SUMMER_BREAK_ROUND = 14  # approximate midpoint
is_post_summer = df['round'] == SUMMER_BREAK_ROUND
df.loc[is_post_summer, 'constructor_form_ewma'] *= 0.85
```

> [!NOTE]
> The same decay logic should apply to `driver_form_ewma` for **regulation resets only** (γ=0.3). Driver talent transfers across regulation eras, but the baseline performance expectation still shifts. Off-season and summer-break decay should **not** apply to driver form, as driver skill is continuous.

---

## Feature Schema Changes

| Change | Feature Name | Type | Added/Modified |
|:---|:---|:---|:---|
| Interaction | `pace_x_overtaking` | Float | **NEW** |
| Interaction | `grid_x_retention` | Float | **NEW** |
| Decay | `constructor_form_ewma` | Float | **MODIFIED** (decay logic) |
| Decay | `driver_form_ewma` | Float | **MODIFIED** (regulation reset decay only) |
| Classification | `is_classified` | Int | **NEW** (added in Gold, used as Stage 1 target) |

---

## Files to Change

| File | Action | Description |
|:---|:---|:---|
| [model_utils.py](file:///d:/Code/F1/model_utils.py) | **NEW** | Shared utility: `position_to_relevance()`, `F1_POINTS_RELEVANCE` mapping, `classify_status()` |
| [gold_feature_engineering.py](file:///d:/Code/F1/gold_feature_engineering.py) | **MODIFY** | Add interaction features, EWMA decay logic, `is_classified` column |
| [train_two_stage_model.py](file:///d:/Code/F1/train_two_stage_model.py) | **NEW** | Two-stage DNF survival + classified ranking pipeline |
| [train_model.py](file:///d:/Code/F1/train_model.py) | **MODIFY** | Use F1 points relevance from `model_utils` |
| [evaluate_baselines.py](file:///d:/Code/F1/evaluate_baselines.py) | **MODIFY** | Use F1 points relevance, add two-stage baseline comparison |
| [cross_validate_model.py](file:///d:/Code/F1/cross_validate_model.py) | **MODIFY** | Use F1 points relevance, add two-stage cross-validation |

---

## Verification Plan

### Automated Tests
1. Re-run `python smoketest.py` to verify no leakage from new features
2. Re-run `python cross_validate_model.py` with the updated features and compare Spearman ρ across all validation seasons (2020–2026)
3. Run `python train_two_stage_model.py` and compare P1, Top-3, MAE, and ρ against both the single-stage model and the starting-grid baseline
4. Verify the two-stage model **consistently beats the grid baseline on Spearman ρ** (the metric where the v1 model failed)

### Manual Verification
- Inspect constructor EWMA values at 2022 Round 1 (post-regulation reset) to confirm the decay brought Mercedes/Red Bull/Williams closer to a neutral baseline
- Spot-check that DNF-predicted drivers are correctly appended at bottom positions in the two-stage output

## Open Questions

> [!IMPORTANT]
> **Summer break round number**: The summer break falls at different rounds each year (typically Round 13–15). Should I:
> - **A)** Hardcode Round 14 as the approximate midpoint for all seasons, or
> - **B)** Build a per-season lookup mapping the actual post-summer-break round number? This would be more accurate but requires manual curation or API schedule parsing.

> [!NOTE]
> **Stage 1 threshold**: For the two-stage model, I'll initially use `P(classified) >= 0.5` as the default cutoff for routing drivers to Stage 2 vs. labeling them as predicted-DNF. We can tune this threshold after seeing initial results.
