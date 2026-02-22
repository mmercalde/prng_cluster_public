# SESSION_CHANGELOG_20260222_S107.md

## Session 107 — February 22, 2026

### Focus: Step 2 v4.1 — Subset-Selection Objective Design + Architecture Clarification

---

## Summary

Session exposed the WSI v4.0 tautology on Zeus (WSI=0.9997), diagnosed root cause,
clarified Step 2 architecture, and produced the v4.1 run_trial() draft for TB review.
Also established file header standard for line-count tracking.

---

## Key Events

### 1. test_both_modes Clarification
Confirmed via live code that `test_both_modes=True` means each trial runs BOTH
constant-skip AND variable-skip (_hybrid) sieves, forward and reverse for each.
Not related to fwd/rev symmetry — the matching mean=0.2537 for both is a
synthetic data artifact, not a flag behavior.

### 2. GitHub Push — S104/S105/S106 + Step 1 outputs
Two clean commits pushed to both repos:
- Commit 1: apply_s105_scorer_worker_v4_0.py + 3 session changelogs
- Commit 2: Step 1 test run outputs (NPZ, configs, working files)
Both private (prng_cluster_project) and public (prng_cluster_public) synced.

### 3. WSI v4.0 Tautology — Confirmed on Zeus

Smoke test result:
```
Parameters: rm1=10, rm2=100, rm3=500, max_offset=5, temporal_window_size=50
wf=0.016  wr=0.164  w3=0.820
WSI = 0.999673
```

Root cause: w3 = rm3/(rm1+rm2+rm3) = 500/610 = 0.820
scores ≈ 0.820 * (fwd*rev) = 0.820 * quality
corr(scores, quality) ≈ 1.0 structurally — Optuna landscape is flat.

TB ruling request filed: docs/TB_RULING_REQUEST_STEP2_v4_1_OBJECTIVE.md
Pushed to both repos (commit ffcc08b).

### 4. Architecture Deep-Dive — What Step 2 Actually Does

Extended investigation confirmed:
- survivor_scorer.py: 3-lane residue filtering ALIVE and working (lines 208, 285, 419)
- rm1/rm2/rm3 feed into SurvivorScorer as residue_mods list via translation at line 99-103
- scorer_trial_worker.py v4.0: SurvivorScorer was REMOVED — params now just weight NPZ signals
- This is why v4.0 is meaningless — params have no real work to do without SurvivorScorer

### 5. TB Ruling Received (S107)

TB approved v4.1 direction: subset-selection objective.

Key rulings:
- rm1/rm2/rm3 + max_offset define a filter (selector over seeds) — correct framing
- Primary quality signal: bidirectional_selectivity (not fwd*rev)
- fwd/rev used only as secondary balance regularizer (0.25 weight)
- temporal_window_size: controls keep-rate target band (pending TODO-B)
- Preferred mask: k-of-3 (seed passes if ≥2 of 3 residue conditions met)

Concrete TB formula:
```
mask     = vote_count >= 2  (k-of-3)
sel_norm = (mean(selectivity[mask]) - global_min) / (global_max - global_min)
bal      = 1 - abs(mean(fwd[mask]) - mean(rev[mask]))
size_pen = abs(log((keep + eps) / target_keep))
objective = clip(sel_norm * (0.75 + 0.25*bal) - lambda * size_pen, -1, 1)
```

### 6. v4.1 run_trial() Draft Produced

File: run_trial_v4_1_draft.py
AST: CLEAN
Lines: 319 (including _log_trial_metrics helper)

Key implementation decisions:
- NPZ stats hardcoded from Zeus run: sel_min=1.010, sel_max=2.471
- target_keep=0.10 (10% = ~674 seeds), band [1%, 40%]
- lambda=0.30 size penalty weight
- S101 per-trial RNG preserved: random.seed(trial_num)
- WATCHER CLI compatibility preserved
- Per-trial metric logging: subset_n, keep, sel_mean, sel_p25/p75, bal, size_pen, objective

Two open TODOs pending TB clarification:
- TODO-A: Do rm1/rm2/rm3 act as literal filters downstream (Steps 3-6)?
- TODO-B: Does trial_number in NPZ give temporal_window_size real temporal semantics?

Draft submitted to TB for approval.

---

## Project Standard Established — File Header Line Counts

**Standard:** When touching any file, add/update header block with:
```python
# filename.py (vX.Y - description)
# Last modified : YYYY-MM-DD
# Session       : SXXX
# Expected lines: ~NNN
#   section_name()  : ~NN  (lines START-END)
#   section_name()  : ~NN  (lines START-END)
```

**Rules:**
1. Total line count approximate — >15% deviation is a red flag
2. Section anchors are the critical part — catch targeted truncation
3. Update header on every patch, not just major versions
4. Patcher scripts must print wc -l of result after applying
5. Only document functions/classes >20 lines

**Applies:** Forward only — retrofit only when touching a file.
**Rationale:** Addresses recurring truncation/refactor incidents where
git history showed what changed but not whether the result was complete.

---

## Files Produced This Session

| File | Destination |
|------|-------------|
| TB_RULING_REQUEST_STEP2_v4_1_OBJECTIVE.md | docs/ (committed) |
| run_trial_v4_1_draft.py | Downloads (awaiting TB approval) |
| SESSION_CHANGELOG_20260222_S107.md | docs/ (this file) |

---

## TODOs (Updated)

1. TB ruling on v4.1 run_trial() draft — ⏳ SUBMITTED THIS SESSION
2. TODO-A: Confirm rm1/rm2/rm3 role in Steps 3-6 (literal filter vs feature param)
3. TODO-B: Confirm temporal_window_size semantic (keep-rate vs temporal diversity)
4. Update S103 changelog with Part2 fix — still pending
5. Regression diagnostics for gate_true validation
6. Remove 27 stale project files
7. Phase 9B.3 heuristics (deferred)

---

## Git Commits This Session

| Hash | Description |
|------|-------------|
| 7047c68 | S104/S105/S106: scorer_trial_worker v4.0 patcher + changelogs |
| 26e3fa6 | S106: Step 1 test run outputs + working files |
| ffcc08b | docs(S107): TB ruling request — Step 2 v4.1 objective redesign |

---

## Next Session Start

1. Check for TB ruling on run_trial_v4_1_draft.py
2. If approved — write patcher script (add npz_selectivity + npz_trial_number
   to load_data(), drop in new run_trial() + _log_trial_metrics())
3. Apply patcher, smoke test 3 param sets — verify WSI varies across trials
4. Distribute to rigs, run full Step 2

---

*Session 107 — Team Alpha*
*WSI v4.0 tautology confirmed + diagnosed. v4.1 draft submitted to TB.*
