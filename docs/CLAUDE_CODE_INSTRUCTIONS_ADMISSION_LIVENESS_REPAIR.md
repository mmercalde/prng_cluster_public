# CLAUDE_CODE_INSTRUCTIONS_ADMISSION_LIVENESS_REPAIR.md — REV1

**The §4.3 silent hang: separate admission liveness from execution maintenance.**

Team Beta confirmed this a **Phase 7 blocker** and **promoted it ahead of the remaining bounded
Phase 6 work.** It is the next isolated change after P0.5 Q2 closure (`8600e75`).

**Base:** current `main` on VM 101. Claude Code as `michael`, venv `~/venvs/torch`. Implement
and iterate; you do **NOT** commit, push, or run WATCHER. STOP at the gate.

---

## 0. The defect

`docs/FLEET_STATE_REQUIREMENTS_v1.md` §4.3. Everything that keeps a trial moving sits inside
one condition:

```
miner/range_miner_coordinator.py:3715
    if len(eligible) >= expected_workers and stage_idx < len(workflow_stages):
:3727        self.assign_stripes(...)
:3731        self._dispatch_pending(...)
:3737        self.process_lease_expiry(run_id, eligible)
```

`assign_stripes`, `_dispatch_pending`, `process_lease_expiry` **and the stage advance** are all
gated on it. And `serve_timeout` defaults to `None` — **correctly**, by Beta's earlier ruling,
because a multi-billion-seed scan exceeds any fixed wall clock.

Two consequences:

- **Before assignment:** if fewer than `expected_workers` daemons ever register, the loop
  accepts connections forever. No assignment, no dispatch, no error, no timeout. The 15 s read
  deadline (`:3698-3711`) only drops connections that never complete a *frame*; registered idle
  workers are exempt.
- **After assignment:** if worker deaths drop `len(eligible)` below the threshold,
  `process_lease_expiry` **stops being called.** Dead workers' stripes stay `claimed` with
  expired leases nobody processes.

> **The Blocker-3 failure matrix is unreachable in exactly the situation it exists for.**

Under Phase 7 — 50 trials, autonomous, WATCHER at the wheel — one mid-run GPU loss produces an
indefinite hang with no signal to react to.

## 1. The approved repair (Beta, Ruling 1 — follow exactly)

**This is not solved with a finite overall `serve_timeout`.** The `None` default stays.

The repair **separates admission liveness from execution maintenance**:

**Admission (bounded):**
- Before each stage is assigned, wait for `expected_workers` under a positive finite
  **`worker_admission_timeout`**.
- **Default 180 seconds**, matching the existing PWC readiness window.
- **Do not reset it because workers connect, disconnect, or churn.** Reset **only** at a genuine
  new-stage boundary.
- If the threshold is not reached: call `fail_trial` with **run ID, stage, expected count and
  eligible count**, and propagate a terminal failure to WATCHER.

**Maintenance (unbounded):**
- **Once a stage is assigned**, `_dispatch_pending`, `process_lease_expiry` and
  stage-completion evaluation must run **regardless of whether the current eligible count
  remains above `expected_workers`.**

**Explicitly forbidden by the ruling:**
- **Do not reduce `expected_workers` dynamically.**
- **Do not change the Blocker-3 constant/hybrid failure matrix** (`:2892-2969`). It is correct;
  it simply must become reachable.
- **Do not change the numerical interpretation of `worker_pool_size` in this patch.** Its
  per-rig vs fleet-wide unit defect belongs to the later fleet-authority work.

## 2. The behaviour this restores

| situation | expected |
|---|---|
| initial shortage | **bounded explicit failure** at the admission timeout |
| constant-phase worker loss (phases 1/2 — TFM's `java_lcg` path) | lease expiry reaches the existing **immediate trial failure** |
| hybrid loss (phases 3/4) | the existing **one-reassignment** policy executes |
| final work completed, worker disconnects afterwards | **the trial can still commit** |
| healthy long scan | **unbounded by wall-clock duration** |

That last two rows matter: the repair must not make a *successful* trial fail because a worker
dropped after finishing, and it must not impose any duration ceiling on a healthy run.

## 3. Gates — Beta specified these

Add to the appropriate existing harness (or a new one — say which and why):

| gate | asserts |
|---|---|
| G-ADMISSION-TIMEOUT | below threshold **before** assignment → **terminal failure**, not a hang. Message carries run ID, stage, expected, eligible |
| G-CROSS-CONSTANT | threshold crossing **after** assignment, constant phase → **Blocker-3 failure** fires |
| G-CROSS-HYBRID | hybrid reassignment executes **below the original threshold** |
| G-FINAL-STAGE | final-stage completion **below threshold** still commits |
| G-LONG-HEALTHY | healthy long-running control with `serve_timeout=None` — **no admission timeout fires**, no ceiling imposed |
| **G-MUTANT** | **restoring the outer threshold guard must turn the new gates red** |

**G-MUTANT is the one that proves the rest.** A repair to a hang is easy to assert and hard to
demonstrate; reverting to the enclosing guard and showing the gates red is what makes them
meaningful. Beta named it explicitly.

**Testing a hang is the hard part.** State how you did it — a bounded harness timeout that
distinguishes *"the code under test failed correctly"* from *"the test gave up"* is the minimum.
A gate that passes because it timed out is a vacuous pass (VIR-2), and this deliverable is
specifically about a system that hangs instead of failing.

## 4. Out of scope

- **Beta's Q1 refinement** — the local-run execution set. Still **unauthorized**; it must come
  through the shared Resolved Execution Set (Beta Ruling 4), not a special case here.
- **The Resolved Execution Set itself** (Beta Ruling 2) — step 4 in the sequence, not this.
- `worker_pool_size` unit semantics — deferred by the ruling.
- The Blocker-3 matrix — reachable, not rewritten.
- The dataset authority, P0.5, the published dataset, the pointer.
- Any skip work — after bounded Phase 6.
- **Do not add a finite `serve_timeout`.** Beta's `None` ruling stands.

## 5. Verification-integrity controls (VIR-1…6)

- **execution proof** — the gates drive the real `serve_trial` loop, not a reimplementation of
  its logic.
- **clean control** — G-LONG-HEALTHY: a healthy run is unaffected, and the admission timeout
  does **not** fire.
- **fault-injection control** — G-MUTANT, per Beta.
- **completion sentinel** — explicit `PASS | FAIL | UNAVAILABLE | INCOMPLETE`.
- **unavailable-observer** — if a case cannot be exercised without real hardware failure, mark
  it `UNAVAILABLE` and say so. **Do not simulate a hang and call it observed.**
- **audit claim scope** — declare searched and unavailable surfaces.

## 6. Non-regression

D1.1 · D4 · D5 · D6 3.A · **D6-threshold 17/17** · D6.1 · **threshold-propagation 5/5** ·
Chapter1-P0 12/12 · **P0.5 dataset authority 38/38 with `--fleet`** · Phase 3 ·
**Phase 4 63/63**.

Gate 22 and `G-MINER-UNCHANGED` will both see changed `miner/` files — register with rationale,
**append rather than rewrite**, and check whether another session touched those harnesses first.
`G-MINER-UNCHANGED` was strengthened during P0.5 to grep registered diffs for threshold tokens;
**keep that strengthening intact.**

## 7. Report

The change with `file:line`. Where the admission wait sits relative to assignment, and how the
"reset only at a genuine stage boundary" rule is enforced. How the hang cases were tested and
how a correct failure is distinguished from a harness timeout. The gate matrix and the G-MUTANT
result. Confirmation that the Blocker-3 matrix, `expected_workers` and `worker_pool_size`
semantics are unchanged, and that no finite `serve_timeout` was introduced. Then STOP.
**Do not commit.**
