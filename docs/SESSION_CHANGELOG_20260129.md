# Session Changelog — January 29, 2026

## Overview

**Focus:** Selfplay Architecture Design & Inner Episode Trainer Implementation  
**Duration:** Extended session  
**Outcome:** Major architecture decisions finalized, first selfplay component complete

---

## Major Accomplishments

### 1. LightGBM GPU Benchmarking (Completed from Jan 28)

**Finding:** CPU is 8-11x faster than GPU for tree models on mining rigs.

| Rig | CPU (12 models) | GPU (12 models) | CPU Advantage |
|-----|-----------------|-----------------|---------------|
| rig-6600 | 1.12s | 8.79s | **7.9x faster** |
| rig-6600b | 1.08s | 11.92s | **11x faster** |

**Root Cause:** ROCm OpenCL has ~1.2s initialization overhead regardless of dataset size.

**Decision:** Tree models (LightGBM, XGBoost, CatBoost) run on CPU only for selfplay inner episodes.

---

### 2. Selfplay Architecture Proposal (v1.1 APPROVED)

**Document:** `SELFPLAY_ARCHITECTURE_PROPOSAL_v1_0.md`

**Key Decisions:**
- Outer episodes: 26 GPUs via coordinators (mandatory)
- Inner episodes: CPU tree models only (NO neural_net)
- Learning telemetry: Observational only (no control path)

**Resource Allocation:**

| Phase | Zeus | rig-6600 | rig-6600b |
|-------|------|----------|-----------|
| Outer (Sieving) | 2× 3080 Ti | 12× RX 6600 | 12× RX 6600 |
| Inner (ML) | 3 CPU workers | 2 CPU workers | 2 CPU workers |

---

### 3. Authority Contract Ratified (v1.1)

**Document:** `CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md`

**The One-Sentence Rule:**
> "Selfplay explores. Chapter 13 decides. WATCHER enforces."

**Six Invariants:**
1. Promotion Authority — Only Chapter 13 updates active policy
2. Ground Truth Isolation — Only Chapter 13 sees live outcomes
3. Selfplay Output Status — Hypotheses, not decisions
4. Coordinator Requirement — GPU work uses coordinators
5. Telemetry Usage — Informs but doesn't solely decide
6. Safe Fallback — Validated baseline always recoverable

---

### 4. Chapter 13 Implementation Progress Updated (v1.5)

**Document:** `CHAPTER_13_IMPLEMENTATION_PROGRESS_v1_5.md`

**Changes:**
- Added Phase 8: Selfplay Integration
- Added selfplay architecture overview
- Added learning telemetry schema
- Updated session history

---

### 5. Inner Episode Trainer (v1.0.3 COMPLETE)

**File:** `inner_episode_trainer.py`

**Features:**
- LightGBM, XGBoost, CatBoost (NO neural_net)
- K-fold cross-validation (k=3 for speed)
- Proxy metrics: R², MAE, fold_std, train_val_gap
- Fitness function with overfit penalty
- Numerical stability: train_val_gap capped at 5.0
- OMP_NUM_THREADS safety
- Nested features auto-detection

**Test Results (75K survivors):**

| Model | R² | Fitness | Time |
|-------|-----|---------|------|
| LightGBM | 0.9999 | 0.82 | 786ms |
| XGBoost | 1.0000 | 0.35 | 743ms |
| CatBoost | 1.0000 | **0.83** 🏆 | 1133ms |

**Team Beta Verdict:** Approved for Phase 8 integration.

---

## Files Created/Modified

### New Files
| File | Location | Purpose |
|------|----------|---------|
| `SELFPLAY_ARCHITECTURE_PROPOSAL_v1_0.md` | `docs/` | Approved architecture |
| `SELFPLAY_INTEGRATION_PROGRESS_v1_0.md` | `docs/` | Implementation tracker |
| `CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md` | `docs/` | Authority contract |
| `CHAPTER_13_IMPLEMENTATION_PROGRESS_v1_5.md` | `docs/` | Updated progress |
| `inner_episode_trainer.py` | root | Inner episode ML training |

### Version History
| File | Version | Changes |
|------|---------|---------|
| inner_episode_trainer.py | 1.0.0 | Initial implementation |
| inner_episode_trainer.py | 1.0.1 | Team Beta: R² floor, NaN guard, OMP safety |
| inner_episode_trainer.py | 1.0.2 | Nested features loading fix |
| inner_episode_trainer.py | 1.0.3 | train_val_gap cap for numerical stability |

---

## Key Learnings

### 1. CPU vs GPU for Tree Models
- LightGBM GPU only helps at 1M+ samples
- ROCm OpenCL has contention issues with parallel processes
- i5 CPUs (6-core) achieve 10-11 models/sec

### 2. Selfplay Architecture
- Must use coordinators for GPU work (no direct SSH)
- CPU work doesn't need ROCm stagger/batching
- Inner episodes produce hypotheses, not decisions

### 3. Numerical Stability
- Near-zero MSE causes train_val_gap explosion
- Capping at 5.0 prevents fitness poisoning
- Warning message confirms guardrail engagement

---

## Next Session: Selfplay Orchestrator

**Priority:** Build `selfplay_orchestrator.py`

**Components:**
1. Outer episode trigger (calls coordinators)
2. Inner episode trigger (calls inner_episode_trainer)
3. Optuna integration (parameter optimization)
4. Policy candidate emission (`learned_policy_candidate.json`)

**Estimated Effort:** 2-3 hours

---

## Git Commands

```bash
cd ~/distributed_prng_analysis

# Stage new files
git add docs/SELFPLAY_ARCHITECTURE_PROPOSAL_v1_0.md
git add docs/SELFPLAY_INTEGRATION_PROGRESS_v1_0.md
git add docs/CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md
git add docs/CHAPTER_13_IMPLEMENTATION_PROGRESS_v1_5.md
git add docs/SESSION_CHANGELOG_20260129.md
git add inner_episode_trainer.py

# Commit
git commit -m "feat: Selfplay architecture + inner episode trainer v1.0.3

- SELFPLAY_ARCHITECTURE_PROPOSAL_v1_0.md: Approved by Team Beta
- CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md: 6 invariants ratified
- inner_episode_trainer.py: CPU tree models with proxy metrics
- CHAPTER_13_IMPLEMENTATION_PROGRESS_v1_5.md: Added Phase 8

Benchmarks: CPU 8-11x faster than GPU for tree models
Inner Episode: CatBoost wins (fitness=0.83) on 75K survivors"

# Push
git push origin main
```

---

*Session ended: January 29, 2026*
