# PROPOSAL: S156 — Persistent Worker Coordinator Lifecycle Fix
**Version:** 2.0  
**Date:** 2026-03-25  
**Author:** Claude (Team Alpha Lead Dev)  
**Status:** APPROVED WITH MODIFICATIONS — Team Beta 2026-03-25  
**Scope:** `persistent_worker_coordinator.py`, `window_optimizer_integration_final.py`

---

## 1. Problem Statement

### 1.1 Observed Failure

rrig6600c consistently crashes during Step 1 production runs when running 8 concurrent GPU
workers. The remote crash monitor captured the definitive signature:

```
py_procs=6  → PageTables=15,000 kB   ← workers from previous trial (not yet cleaned up)
py_procs=8  → PageTables=16,000 kB
py_procs=10 → PageTables=17,000 kB   ← new trial spawning ON TOP of previous session state
py_procs=12 → PageTables=19,124 kB   ← kernel freeze / D-state
DOWN
```

The machine crashes at ~13,000-19,000 kB PageTables — the exact threshold where ROCm context
page table allocations for 10-12 concurrent python processes exhaust available kernel page
table memory.

### 1.2 Root Cause — Team Beta Confirmed

> *"The root cause is that the 'persistent' worker pool is being recreated every trial instead
> of once per optimization session, and that repeated spawn/teardown cycle is the most likely
> reason rrig6600c hits the multi-process ROCm/PageTables crash cliff."*

The specific bug is in `window_optimizer_integration_final.py` — `run_trial_persistent()`:

```python
def run_trial_persistent(...) -> Dict[str, Any]:
    pwc = PersistentWorkerCoordinator(...)  # ← NEW INSTANCE EVERY TRIAL
    pwc.startup()                           # ← SPAWNS 8 WORKERS EVERY TRIAL
    try:
        # run passes...
    finally:
        pwc.shutdown()                      # ← TEARDOWN EVERY TRIAL
```

This creates a full spawn/teardown cycle PER TRIAL. With 3 trials = 3 spawn cycles.

### 1.3 Why rrig6600 and rrig6600b Don't Crash

rrig6600 and rrig6600b never crash mid-trial so their `pwc.shutdown()` always executes
cleanly. The next trial starts with a clean slate.

After a crash on rrig6600c, the production path may re-spawn workers onto the node without
proving the node is clean or that prior session state has been fully reaped and released.
This allows worker accumulation across trials, eventually causing PageTable overflow.

### 1.4 Missing Invariant

There is no hard invariant enforcing one live worker per (hostname, gpu_id) slot across the
run. No pre-spawn validation checks for pre-existing `sieve_gpu_worker` processes.

---

## 2. Approved Fixes

### Phase A — Immediate Safety Bandaid

**Status:** Temporary safety patch — NOT the root fix. Labeled explicitly as bandaid.

**What:** Add targeted pre-spawn cleanup to `PersistentWorkerCoordinator.startup()`.

**Requirements per TB:**
- Use targeted match pattern (not broad pkill)
- SIGTERM first, SIGKILL only on timeout
- Log exact processes found and reaped
- Scope by `--persistent` flag and `--gpu-id`

```python
# [S156-BANDAID] Pre-spawn targeted cleanup — temporary safety net only.
# Root fix is session-scoped PWC (Phase B). This prevents stale worker
# accumulation on nodes where prior session state was not fully reaped.
import subprocess as _subprocess
for node in self.nodes:
    if self._is_localhost(node.hostname):
        continue
    if not self._is_rocm(node):
        continue
    try:
        # Find matching persistent workers on this node
        find_cmd = (
            f"ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "
            f"-o BatchMode=yes "
            f"{node.username}@{node.hostname} "
            f"'pgrep -af \"sieve_gpu_worker.*--persistent\" || echo none'"
        )
        r = _subprocess.run(find_cmd, shell=True, capture_output=True,
                            text=True, timeout=10)
        found = r.stdout.strip()
        if found and found != "none":
            self.logger.info(f"  [S156] {node.hostname}: found stale workers: {found}")
            # SIGTERM first
            term_cmd = (
                f"ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "
                f"-o BatchMode=yes "
                f"{node.username}@{node.hostname} "
                f"'pkill -15 -f \"sieve_gpu_worker.*--persistent\" 2>/dev/null; sleep 2; "
                f"pkill -9 -f \"sieve_gpu_worker.*--persistent\" 2>/dev/null; "
                f"sleep 1; echo reaped'"
            )
            r2 = _subprocess.run(term_cmd, shell=True, capture_output=True,
                                 text=True, timeout=15)
            if "reaped" in r2.stdout:
                self.logger.info(f"  [S156] {node.hostname}: stale workers reaped")
            else:
                self.logger.warning(f"  [S156] {node.hostname}: reap uncertain")
        else:
            self.logger.info(f"  [S156] {node.hostname}: no stale workers found")
    except Exception as e:
        self.logger.warning(f"  [S156] {node.hostname}: pre-spawn cleanup failed: {e}")
# Allow ROCm contexts to fully release after cleanup
import time as _time
_time.sleep(2)
```

---

### Phase B — Architectural Correction (Root Fix)

**What:** Create ONE `PersistentWorkerCoordinator` for the entire Step 1 session.

**Ownership:** Per TB — session-scoped PWC ownership lives in the **Step 1 execution/
integration layer**, not inside `OptunaBayesianSearch`. The persistent-worker choice is an
execution-engine concern, not an Optuna-strategy concern.

**Current (wrong):**
```
run_trial_persistent()          ← called per trial
  pwc = NEW PersistentWorkerCoordinator()
  pwc.startup()
  run passes
  pwc.shutdown()
```

**Correct:**
```
Step 1 integration layer         ← session scope
  pwc = PersistentWorkerCoordinator()   ← ONCE
  pwc.startup()                          ← ONCE
  for each trial:
    run_trial_persistent(pwc=pwc)        ← REUSE
    pwc.reset_for_new_trial()            ← between trials
  pwc.shutdown()                         ← ONCE
```

**New method `reset_for_new_trial()` — minimal contract:**
```python
def reset_for_new_trial(self):
    """Reset per-trial state while keeping workers alive.
    
    Primary contract: worker-pool continuity and health.
    - Verify or respawn dead (non-quarantined) workers
    - Clear per-trial progress state
    - Do NOT destroy the persistent pool
    - Do NOT mutate persistent handles beyond health recovery
    """
    # Health check — respawn any dead non-quarantined workers
    for handle in self.workers:
        if not handle.quarantined:
            self._ensure_worker_alive(handle)
    
    # Clear per-trial progress writer only
    if self._progress_writer:
        try:
            self._progress_writer.finish()
        except Exception:
            pass
        self._progress_writer = None
    
    # Log exact pool health per node
    for node in self.nodes:
        if self._is_localhost(node.hostname):
            continue
        expected = min(self.worker_pool_size, node.gpu_count)
        alive = sum(1 for w in self.workers
                    if w.node.hostname == node.hostname
                    and w.alive and not w.quarantined)
        self.logger.info(
            f"[S156] {node.hostname}: trial reset — {alive}/{expected} workers alive"
        )
```

---

### Phase C — Required Guardrails

**C1. One-worker-per-slot invariant**

Before spawning (node, gpu_id), verify the slot is clean:

```python
def _check_slot_clean(self, handle: WorkerHandle) -> bool:
    """Verify no sieve_gpu_worker already running for this gpu_id on this node."""
    cmd = (
        f"ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "
        f"-o BatchMode=yes "
        f"{handle.node.username}@{handle.node.hostname} "
        f"'pgrep -af \"sieve_gpu_worker.*--gpu-id {handle.gpu_id}.*--persistent\" "
        f"|| echo clean'"
    )
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        if "clean" in r.stdout:
            return True
        # Slot occupied — attempt graceful reap
        self.logger.warning(
            f"  [S156] {handle.node.hostname}:GPU{handle.gpu_id} slot occupied — reaping"
        )
        reap_cmd = (
            f"ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "
            f"-o BatchMode=yes "
            f"{handle.node.username}@{handle.node.hostname} "
            f"'pkill -15 -f \"sieve_gpu_worker.*--gpu-id {handle.gpu_id}.*--persistent\" "
            f"2>/dev/null; sleep 2; "
            f"pkill -9 -f \"sieve_gpu_worker.*--gpu-id {handle.gpu_id}.*--persistent\" "
            f"2>/dev/null; echo reaped'"
        )
        r2 = subprocess.run(reap_cmd, shell=True, capture_output=True, text=True, timeout=15)
        return "reaped" in r2.stdout
    except Exception as e:
        self.logger.warning(f"  [S156] slot check failed: {e}")
        return False
```

**C2. Exact startup verification (no soft tolerance)**

Per TB — replace 75% soft threshold with exact per-node accounting:

```python
def verify_pool_ready(self) -> Dict[str, Dict]:
    """Return exact pool health per node. Caller decides policy."""
    health = {}
    for node in self.nodes:
        if self._is_localhost(node.hostname):
            continue
        expected = min(self.worker_pool_size, node.gpu_count)
        alive = sum(1 for w in self.workers
                    if w.node.hostname == node.hostname
                    and w.alive and not w.quarantined)
        quarantined = sum(1 for w in self.workers
                          if w.node.hostname == node.hostname
                          and w.quarantined)
        health[node.hostname] = {
            "expected": expected,
            "alive": alive,
            "quarantined": quarantined,
            "degraded": alive < expected
        }
        if alive < expected:
            self.logger.warning(
                f"[S156] DEGRADED POOL: {node.hostname} "
                f"{alive}/{expected} alive, {quarantined} quarantined"
            )
        else:
            self.logger.info(
                f"[S156] {node.hostname}: pool healthy {alive}/{expected}"
            )
    return health
```

**C3. Explicit remote shutdown reap**

Per TB — shutdown must not rely solely on stdin pipe close:

```python
# Add to shutdown() after existing worker teardown:
# [S156] Explicit remote reap — verify no workers remain after shutdown
for node in self.nodes:
    if self._is_localhost(node.hostname):
        continue
    try:
        reap_cmd = (
            f"ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "
            f"-o BatchMode=yes "
            f"{node.username}@{node.hostname} "
            f"'pkill -9 -f \"sieve_gpu_worker.*--persistent\" 2>/dev/null; "
            f"remaining=$(pgrep -c -f sieve_gpu_worker 2>/dev/null || echo 0); "
            f"echo \"shutdown_reap_done:$remaining\"'"
        )
        r = subprocess.run(reap_cmd, shell=True, capture_output=True,
                           text=True, timeout=15)
        if "shutdown_reap_done:0" in r.stdout:
            self.logger.info(f"[S156] {node.hostname}: shutdown reap confirmed clean")
        else:
            self.logger.warning(f"[S156] {node.hostname}: shutdown reap result: {r.stdout.strip()}")
    except Exception as e:
        self.logger.warning(f"[S156] {node.hostname}: shutdown reap failed: {e}")
```

---

## 3. Success Criteria

Per TB — success is defined as:

1. **Worker count bounded** — py_procs on rrig6600c stays at or below expected pool size
   across all trials
2. **PageTables non-climbing** — PageTables do not rise trial-over-trial during a 3-trial run
3. **No absolute PageTables target** — specific kB values are hardware-dependent and not
   promised

---

## 4. Implementation Plan

### Phase A — Immediate (S156, tonight)
- [x] Build `apply_s156_pwc_prestartup_cleanup.py` with targeted cleanup
- [ ] Deploy to Zeus
- [ ] Commit with message `fix(s156-bandaid): targeted pre-spawn cleanup in PWC startup`
- [ ] Dual-push

### Phase B+C — Architectural (S157)
- [ ] Refactor Step 1 integration layer for session-scoped PWC
- [ ] Add `reset_for_new_trial()` to PWC
- [ ] Add `_check_slot_clean()` invariant
- [ ] Add `verify_pool_ready()` with exact accounting
- [ ] Add explicit shutdown reap
- [ ] smoke test: `sweep_preprod.sh` (50M seeds, 3 trials)
- [ ] Production test: full sweep with rrig6600c at 8 workers
- [ ] Confirm: py_procs bounded, PageTables non-climbing across trials

---

## 5. Files Modified

| File | Change | Phase |
|------|--------|-------|
| `persistent_worker_coordinator.py` | Targeted pre-spawn cleanup in `startup()` | A |
| `persistent_worker_coordinator.py` | `reset_for_new_trial()` | B |
| `persistent_worker_coordinator.py` | `_check_slot_clean()` invariant | C |
| `persistent_worker_coordinator.py` | `verify_pool_ready()` exact accounting | C |
| `persistent_worker_coordinator.py` | Explicit shutdown reap | C |
| `window_optimizer_integration_final.py` | Session-scoped PWC, pass to `run_trial_persistent()` | B |

---

*Filed: 2026-03-25*  
*Team Beta Decision: APPROVED WITH MODIFICATIONS — 2026-03-25*  
*Phase A patch ready for deployment*  
*Phase B+C targeted for S157*
