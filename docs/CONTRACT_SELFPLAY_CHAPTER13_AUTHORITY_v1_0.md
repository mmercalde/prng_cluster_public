# CONTRACT: Selfplay / Chapter 13 / WATCHER Authority Boundaries
## Version 1.0 — January 29, 2026

**Status:** RATIFIED  
**Approved By:** Team Beta + User  
**Binding On:** All future implementation

---

> **OWNER AMENDMENT, 2026-11-24 (v1.2) — read before citing anything below.**
> The body of this contract is unchanged. Where it conflicts with the three points here, these
> govern (owner statements of 2026-11-24, word for word in project doc
> `claude/TFM_LEARNING_LOOP_OWNER_INTENT_20261124.md`).
>
> 1. **The promotion gate is walk-forward on back draws.** A selfplay candidate is judged by
>    replaying it on historical draws selfplay did not score itself on, against the active
>    settings. This replaces "REAL draws" / "real-draw validation" as the promotion test in
>    Invariant 1, Invariant 3, the Complete Flow step 3 and the Enforcement checklist.
>    Chapter 13 still grades every live draw afterwards and still detects the regime change.
>    Prior record: S176/S179 hold the KPI triggers audit-only "until a walk-forward baseline
>    exists" (`docs/CHAPTER_13_AUDIT.md:90`, `:192`).
> 2. **Selfplay is meant to run continuously**, not only when triggered. The trigger table
>    stays valid as the list of events that can also start or redirect it.
> 3. **Selfplay may be rewarded by hits on historical draws it is allowed to use.** The proxy
>    reward table stays as additional signals. Invariant 2 is unchanged: selfplay never sees
>    live outcomes, and never the back draws reserved for the gate.
>
> "Approved By: Team Beta + User" and "Binding" below predate the owner rulings of 2026-11-11:
> Michael is the final authority and Beta's reviews are not binding (`CLAUDE.md`).

---

## The One-Sentence Rule

**Selfplay explores. Chapter 13 decides. WATCHER enforces.**

---

## Authority Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                      CHAPTER 13                             │
│                 (Governor / Arbiter)                        │
│                                                             │
│  SOLE AUTHORITY OVER:                                       │
│    • Ground-truth outcomes (live draw results)              │
│    • Promotion / rejection decisions                        │
│    • learned_policy_active.json                             │
│    • Shadow evaluation (candidate vs active)                │
│                                                             │
│  MAY: Authorize or request selfplay runs via WATCHER        │
│  NEVER: Execute selfplay logic directly                     │
│  NEVER: Explore parameter space                             │
└─────────────────────────────────────────────────────────────┘
                              │
                    authorizes / validates
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      WATCHER AGENT                          │
│                 (Operator / Enforcer)                       │
│                                                             │
│  RESPONSIBLE FOR:                                           │
│    • Triggering selfplay (drift/schedule/manual)            │
│    • Spawning and managing workers                          │
│    • Health monitoring and cleanup                          │
│    • Enforcing invariants                                   │
│    • Executing Chapter 13's decisions                       │
│                                                             │
│  MAY: Start/stop selfplay jobs                              │
│  NEVER: Decide promotion                                    │
│  NEVER: Trust selfplay output without Chapter 13 validation │
└─────────────────────────────────────────────────────────────┘
                              │
                           executes
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        SELFPLAY                             │
│              (Optimization Engine / Hypothesis Generator)   │
│                                                             │
│  RESPONSIBLE FOR:                                           │
│    • Running outer episodes (GPU sieving via coordinators)  │
│    • Running inner episodes (CPU ML training)               │
│    • Exploring parameter space                              │
│    • Learning statistical policy from proxy rewards         │
│    • Writing learned_policy_candidate.json                  │
│                                                             │
│  MAY: Access historical data and derived structure          │
│  MAY: Optimize proxy rewards (R², stability, entropy, etc)  │
│  NEVER: Access ground-truth outcomes (live draw results)    │
│  NEVER: Promote policies                                    │
│  NEVER: Modify learned_policy_active.json                   │
│  NEVER: Affect production directly                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Access Matrix

| Data Type | Selfplay | Chapter 13 | WATCHER |
|-----------|----------|------------|---------|
| Historical draws (25 years) | ✅ YES | ✅ YES | ✅ YES |
| Derived structure (windows, survivors, features) | ✅ YES | ✅ YES | ✅ YES |
| Cached artifacts (outer/inner cache) | ✅ YES | ✅ YES | ✅ YES |
| Learning telemetry | ✅ WRITE | ✅ READ | ✅ READ |
| **Ground-truth outcome (did tonight hit?)** | ❌ **NO** | ✅ **YES** | ❌ NO |
| `learned_policy_candidate.json` | ✅ WRITE | ✅ READ | ✅ READ |
| `learned_policy_active.json` | ❌ NO | ✅ **WRITE** | ✅ READ |

---

## File Ownership

| File | Owner | Others |
|------|-------|--------|
| `learned_policy_active.json` | **Chapter 13 ONLY** | Read-only |
| `learned_policy_candidate.json` | Selfplay | Chapter 13 validates |
| `learning_health.json` (telemetry) | Selfplay | Read-only for all |
| `watcher_policies.json` | WATCHER | Read by all |
| `post_draw_diagnostics.json` | Chapter 13 | Read-only |

---

## Invariants (Violation = System Bug)

### Invariant 1: Promotion Authority
```
No component may update learned_policy_active.json 
without real-draw validation by Chapter 13.
```

### Invariant 2: Ground Truth Isolation
```
No component except Chapter 13 may observe 
ground-truth outcomes (live draw hit/miss).
```

### Invariant 3: Selfplay Output Status
```
Selfplay outputs are HYPOTHESES, not DECISIONS.
They must pass Chapter 13's promotion gate before 
affecting production.
```

### Invariant 4: Coordinator Requirement
```
GPU sieving work MUST use coordinator.py / scripts_coordinator.py.
Direct SSH to rigs for GPU work is FORBIDDEN.
```

### Invariant 5: Telemetry Usage
```
Telemetry may inform diagnostics and human review,
but MUST NOT be the sole input to promotion,
execution, or parameter selection.
```

### Invariant 6: Safe Fallback
```
At all times, a validated baseline policy MUST exist
and be recoverable without selfplay or retraining.
```

---

## What Selfplay Optimizes (Proxy Rewards)

Selfplay learns from **proxy signals**, not ground truth:

| Proxy Reward | Description | Correlated With |
|--------------|-------------|-----------------|
| Validation R² | Model fit quality | Generalization |
| Stability across folds | Consistency | Robustness |
| Pool concentration | Prediction focus | Hit rate (indirect) |
| Survivor consistency | Repeat performers | Pattern validity |
| Model agreement | Ensemble consensus | Confidence calibration |
| Overfit penalty | Train/val gap | Generalization |
| Confidence mass | High-conf predictions | Actionable output |

**These are correlated with hit rate — but they are NOT hit rate.**

Only Chapter 13 can measure actual hit rate against real draws.

---

## Trigger Responsibility

| Trigger Source | Who Detects | Who Starts Selfplay | Who Validates Output |
|----------------|-------------|---------------------|----------------------|
| Manual command | User | WATCHER | Chapter 13 |
| Scheduled run | Cron/system | WATCHER | Chapter 13 |
| Drift detected | Chapter 13 | WATCHER | Chapter 13 |
| Policy staleness | WATCHER | WATCHER | Chapter 13 |
| Hit rate collapse | Chapter 13 | WATCHER | Chapter 13 |

**Note:** Chapter 13 may *recommend* selfplay, but WATCHER *executes* it.

---

## Forbidden Actions (Explicit)

### Chapter 13 MUST NOT:
- Execute selfplay logic
- Explore parameter space
- Write to `learned_policy_candidate.json`
- Bypass WATCHER for execution

### Selfplay MUST NOT:
- Access live draw outcomes
- Write to `learned_policy_active.json`
- Promote policies
- Affect production directly
- Bypass coordinators for GPU work

### WATCHER MUST NOT:
- Decide promotion (only execute it)
- Trust selfplay output without Chapter 13 validation
- Override Chapter 13 decisions
- Expose ground-truth outcomes to selfplay

---

## The Complete Flow (Reference)

```
1. TRIGGER PHASE
   ├── Chapter 13 detects drift/collapse → recommends exploration
   ├── WATCHER receives recommendation
   └── WATCHER triggers selfplay

2. EXPLORATION PHASE (Selfplay)
   ├── Outer episodes: GPU sieving (via coordinators)
   ├── Inner episodes: CPU ML training (tree models)
   ├── Optuna optimization: parameter search
   ├── Proxy reward evaluation: R², stability, etc.
   └── Output: learned_policy_candidate.json

3. VALIDATION PHASE (Chapter 13)
   ├── Shadow evaluation: candidate vs active on REAL draws
   ├── Hit rate comparison
   ├── Regression checks
   └── Promotion decision: ACCEPT or REJECT

4. PROMOTION PHASE (Chapter 13 → WATCHER)
   ├── If ACCEPT: Chapter 13 updates learned_policy_active.json
   ├── WATCHER applies new policy to production
   └── Telemetry records promotion event

5. MONITORING PHASE (Continuous)
   ├── Chapter 13 watches live performance
   ├── WATCHER monitors health
   └── Cycle repeats on next trigger
```

---

## Rationale Summary

| Design Choice | Rationale |
|---------------|-----------|
| Selfplay can't see live outcomes | Prevents overfitting to sparse real signals |
| Chapter 13 owns promotion | Single arbiter of truth prevents conflicts |
| WATCHER mediates execution | Safety layer between governance and machinery |
| Proxy rewards only | Enables fast exploration without contamination |
| Coordinator requirement | Prevents ROCm/SSH storms (proven infrastructure) |
| Telemetry informs but doesn't decide | Real draws remain final arbiter |
| Safe fallback required | Enables deterministic rollback and disaster recovery |

---

## Enforcement

Any code that violates these invariants is **a bug**, not a feature request.

Code reviews MUST verify:
- [ ] Selfplay never reads live draw outcomes
- [ ] Only Chapter 13 writes to `learned_policy_active.json`
- [ ] GPU work uses coordinators
- [ ] Telemetry is not the sole input to any automated decision
- [ ] Promotion requires Chapter 13 validation with real draws
- [ ] A validated baseline policy exists and is recoverable

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-29 | Initial contract ratified |
| 1.1 | 2026-01-29 | Team Beta improvements: Tightened Invariant 5, added Invariant 6 (Safe Fallback) |
| 1.2 | 2026-11-24 | Owner amendment (notice at top): walk-forward on back draws is the promotion gate; selfplay runs continuously; reward by hits on permitted historical draws. Body unchanged |

---

**END OF CONTRACT**
