# Phase 9B.2 Integration Specification

**Version:** 1.0.0  
**Date:** 2026-01-30  
**Status:** Team Beta Approved  
**Target:** `selfplay_orchestrator.py` v1.0.6 → v1.1.0

---

## Overview

This document specifies exactly how to integrate `policy_conditioned_episode.py` into the existing `selfplay_orchestrator.py`.

**New Files Required:**
```
distributed_prng_analysis/
├── policy_transform.py           # Phase 9B.1 (already created)
├── policy_conditioned_episode.py # Phase 9B.2 (already created)
├── learned_policy_active.json    # Read by selfplay (written by Chapter 13)
├── learned_policy_candidate.json # Written by selfplay (read by Chapter 13)
└── policy_history/               # Archive of all candidates
```

---

## Integration Points

### 1. New Import Block

Add to the imports section of `selfplay_orchestrator.py`:

```python
# Phase 9B.2: Policy-Conditioned Learning
from policy_conditioned_episode import (
    condition_episode,
    create_policy_candidate,
    emit_policy_candidate,
    load_active_policy,
    run_conditioned_episode,
    PolicyConditionedResult,
    PolicyCandidate,
)
```

### 2. New CLI Arguments

Add to the argument parser:

```python
parser.add_argument('--policy-conditioned', action='store_true', default=False,
                    help='Enable policy-conditioned learning (Phase 9B)')
parser.add_argument('--no-emit-candidate', action='store_true', default=False,
                    help='Disable candidate emission (for testing)')
```

### 3. Episode Runner Modification

**BEFORE (v1.0.6):**
```python
def run_inner_episode(survivors, episode_number, model_type="catboost"):
    """Run inner episode training."""
    result = inner_episode_trainer.train(
        survivors=survivors,
        model_type=model_type,
        n_jobs=22,
    )
    return result
```

**AFTER (v1.1.0):**
```python
def run_inner_episode(
    survivors,
    episode_number,
    model_type="catboost",
    *,
    policy_conditioned=False,
    project_root=".",
):
    """
    Run inner episode training with optional policy conditioning.
    
    Phase 9B.2: If policy_conditioned=True, survivors are transformed
    using the active policy before training.
    """
    # Phase 9B.2: Load active policy
    active_policy, _ = load_active_policy(project_root)
    
    # Phase 9B.2: Condition survivors (or passthrough if disabled)
    if policy_conditioned:
        cond_result = condition_episode(
            survivors,
            project_root=project_root,
        )
        training_survivors = cond_result.conditioned_survivors
        
        # Log conditioning results
        logger.info(
            f"[Episode {episode_number}] Policy conditioning: "
            f"{cond_result.survivors_before} → {cond_result.survivors_after} "
            f"(policy={cond_result.active_policy_id})"
        )
        for log_entry in cond_result.transform_log:
            logger.debug(f"  {log_entry}")
    else:
        cond_result = None
        training_survivors = survivors
    
    # Train model
    result = inner_episode_trainer.train(
        survivors=training_survivors,
        model_type=model_type,
        n_jobs=22,
    )
    
    # Phase 9B.2: Add conditioning metadata to result
    result["policy_conditioned"] = policy_conditioned
    result["survivor_count"] = len(training_survivors)
    if cond_result:
        result["conditioning"] = cond_result.to_dict()
    
    return result, active_policy
```

### 4. Candidate Emission

**BEFORE (v1.0.6):**
```python
def run_episode_loop(args):
    for episode in range(args.num_episodes):
        survivors = load_survivors(args.survivors_file)
        result = run_inner_episode(survivors, episode, args.model_type)
        
        # Log results
        logger.info(f"Episode {episode}: fitness={result['fitness']:.4f}")
        
        # Update telemetry
        update_learning_telemetry(result)
```

**AFTER (v1.1.0):**
```python
def run_episode_loop(args):
    for episode in range(args.num_episodes):
        survivors = load_survivors(args.survivors_file)
        
        # Run with policy conditioning
        result, active_policy = run_inner_episode(
            survivors,
            episode,
            args.model_type,
            policy_conditioned=args.policy_conditioned,
            project_root=args.project_root,
        )
        
        # Log results
        logger.info(f"Episode {episode}: fitness={result['fitness']:.4f}")
        
        # Phase 9B.2: Create and emit policy candidate
        if args.policy_conditioned and not args.no_emit_candidate:
            candidate = create_policy_candidate(
                episode_number=episode,
                parent_policy=active_policy,
                training_result=result,
            )
            emit_policy_candidate(
                candidate,
                project_root=args.project_root,
            )
            logger.info(
                f"Emitted candidate: {candidate.policy_id} "
                f"(fitness_delta={candidate.fitness_delta})"
            )
        
        # Update telemetry
        update_learning_telemetry(result)
```

---

## CLI Usage Examples

### Baseline Mode (unchanged behavior)
```bash
python3 selfplay_orchestrator.py \
    --survivors survivors_with_scores.json \
    --num-episodes 10 \
    --model-type catboost
```

### Policy-Conditioned Mode (Phase 9B.2)
```bash
python3 selfplay_orchestrator.py \
    --survivors survivors_with_scores.json \
    --num-episodes 10 \
    --model-type catboost \
    --policy-conditioned \
    --project-root ~/distributed_prng_analysis
```

### Testing Mode (no candidate emission)
```bash
python3 selfplay_orchestrator.py \
    --survivors survivors_with_scores.json \
    --num-episodes 1 \
    --policy-conditioned \
    --no-emit-candidate
```

---

## Data Flow Diagram

```
                              ┌─────────────────────────────┐
                              │  learned_policy_active.json │
                              │  (written by Chapter 13)    │
                              └─────────────┬───────────────┘
                                            │
                                            ▼
┌───────────────────┐         ┌─────────────────────────────┐
│   Raw Survivors   │────────▶│     condition_episode()     │
│   (from Step 3)   │         │                             │
└───────────────────┘         │  • Load active policy       │
                              │  • Apply transforms         │
                              │  • Log audit trail          │
                              └─────────────┬───────────────┘
                                            │
                                            ▼
                              ┌─────────────────────────────┐
                              │  Conditioned Survivors      │
                              │  (filtered/weighted/masked) │
                              └─────────────┬───────────────┘
                                            │
                                            ▼
                              ┌─────────────────────────────┐
                              │  inner_episode_trainer.py   │
                              │  (unchanged)                │
                              └─────────────┬───────────────┘
                                            │
                                            ▼
                              ┌─────────────────────────────┐
                              │     Training Result         │
                              │  • fitness                  │
                              │  • val_r2                   │
                              │  • train_val_gap            │
                              └─────────────┬───────────────┘
                                            │
                                            ▼
                              ┌─────────────────────────────┐
                              │  create_policy_candidate()  │
                              │                             │
                              │  • Set parent_policy_id     │
                              │  • Calculate fitness_delta  │
                              │  • Compute fingerprint      │
                              └─────────────┬───────────────┘
                                            │
                                            ▼
                              ┌─────────────────────────────┐
                              │  emit_policy_candidate()    │
                              │                             │
                              │  → learned_policy_candidate │
                              │  → policy_history/          │
                              └─────────────┬───────────────┘
                                            │
                                            ▼
                              ┌─────────────────────────────┐
                              │  Chapter 13 Validation      │
                              │  (separate process)         │
                              │                             │
                              │  If ACCEPT:                 │
                              │  → learned_policy_active    │
                              └─────────────────────────────┘
```

---

## Invariants (Must Be Preserved)

| Invariant | Enforcement |
|-----------|-------------|
| Selfplay never writes `learned_policy_active.json` | Only `emit_policy_candidate()` is called |
| Policy conditioning is optional | `--policy-conditioned` flag gates all 9B behavior |
| Transforms are pure functional | `apply_policy()` returns new list, never mutates |
| Safety floor enforced | `ABSOLUTE_MIN_SURVIVORS = 50` in policy_transform.py |
| Audit trail preserved | `policy_history/` archives all candidates |
| Fingerprint enables deduplication | Same policy params → same fingerprint |

---

## Testing Checklist

After integration, verify:

```bash
# 1. Unit tests pass
python3 policy_transform.py --test
python3 policy_conditioned_episode.py --test

# 2. Baseline mode unchanged
python3 selfplay_orchestrator.py --survivors test.json --num-episodes 1
# Should work exactly as before

# 3. Policy-conditioned mode works
echo '{}' > learned_policy_active.json  # Empty active policy
python3 selfplay_orchestrator.py --survivors test.json --num-episodes 1 --policy-conditioned
# Should log "Policy conditioning: X → X (policy=baseline_empty)"

# 4. Candidate emitted
ls learned_policy_candidate.json  # Should exist
cat learned_policy_candidate.json  # Should have valid JSON

# 5. Archive created
ls policy_history/  # Should have candidate files

# 6. Chapter 13 can read candidate
python3 chapter_13_acceptance.py --validate-selfplay learned_policy_candidate.json
```

---

## Version Bump

Update version in `selfplay_orchestrator.py`:

```python
# Before
VERSION = "1.0.6"

# After
VERSION = "1.1.0"
```

Changelog entry:
```
## [1.1.0] - 2026-01-30

### Added
- Phase 9B.2: Policy-conditioned learning integration
- `--policy-conditioned` CLI flag
- `--no-emit-candidate` CLI flag for testing
- Policy candidate emission with lineage tracking
- Fitness delta calculation (vs parent policy)

### Changed
- `run_inner_episode()` now accepts policy_conditioned parameter
- Training results include conditioning metadata when enabled

### Dependencies
- Requires: policy_transform.py (Phase 9B.1)
- Requires: policy_conditioned_episode.py (Phase 9B.2)
```

---

## Git Commands

```bash
cd ~/distributed_prng_analysis

# Stage new files
git add policy_transform.py
git add policy_conditioned_episode.py
git add docs/PHASE_9B2_INTEGRATION_SPEC.md

# Commit Phase 9B modules
git commit -m "feat: Phase 9B - Policy-Conditioned Learning modules

- policy_transform.py v1.0.0: Pure functional transforms
- policy_conditioned_episode.py v1.0.0: Episode conditioning
- PHASE_9B2_INTEGRATION_SPEC.md: Integration guide

Total: 41 unit tests passing
Ready for selfplay_orchestrator.py integration"

# After modifying selfplay_orchestrator.py
git add selfplay_orchestrator.py
git commit -m "feat: selfplay_orchestrator.py v1.1.0 - Phase 9B.2 integration

- Added --policy-conditioned flag
- Added --no-emit-candidate flag
- Integrated condition_episode() before training
- Added candidate emission with lineage tracking
- Fitness delta calculation vs parent policy

Phase 9B.2 complete."

git push origin main
```

---

## Next: Phase 9B.3 (Future)

**Proposal Heuristics** — Automatic transform updates based on training outcomes.

The placeholder in `policy_conditioned_episode.py`:
```python
def propose_transform_update(current_transforms, training_result, conditioning_result):
    # Phase 9B.2: Just inherit
    return current_transforms.copy()
    
    # Phase 9B.3 will add:
    # - If fitness improved, tighten filter
    # - If fitness dropped, loosen filter
    # - If overfit detected, add masking
```

This is intentionally deferred — manual policy tuning should be validated first.

---

*End of Integration Specification*
