# PROPOSAL: S158D — ZMQ+SQLite Distributed Sieve Coordinator
**Version:** 1.0  
**Date:** 2026-03-29  
**Author:** Claude (Team Alpha Lead Dev)  
**Status:** Submitted for Team Beta review  
**Scope:** New standalone files only — 3 existing files minimally modified  
**Risk:** Low — purely additive, activated by flag, PWC untouched

---

## 1. Motivation

The current PersistentWorkerCoordinator (PWC) maintains 8 long-lived SSH
connections per rig throughout the entire run. Each connection holds open
stdin/stdout/stderr pipes on Zeus. Under multi-rig load this has caused:

- Kernel panics during worker spawn (rrig6600c repeatedly)
- GIL thrashing from 537 simultaneous blocked SSH pipe threads
- Respawn storms when pipe timeouts cascade

The fundamental issue: **Zeus babysits 24 SSH processes instead of GPUs
doing pure compute.** A crypto miner does not maintain persistent SSH pipes
to its GPUs.

---

## 2. Architecture

### Current PWC model (push via SSH stdin pipe):
```
Zeus → SSH pipe → Worker (persistent, idle between jobs)
Zeus ← SSH pipe ← Worker (result)
16 SSH connections open for entire run duration
```

### Proposed ZMQ+SQLite model (pull via TCP):
```
Zeus  → ZMQ PUSH (port 5557) → Workers pull jobs
Workers → ZMQ PUSH → Zeus PULL (port 5558) ← results
SQLite tracks chunk state: pending → claimed → done

SSH used ONCE per rig at startup to launch workers.
After that: zero SSH connections during compute.
```

### Worker lifecycle (like a crypto miner):
```
boot once → ROCm init once → loop:
  connect to Zeus ZMQ
  pull job
  GPU compute
  push result
  pull next job
```

---

## 3. Files

### New files (zero impact on existing code):

| File | Purpose | Lines |
|------|---------|-------|
| `zmq_sqlite_coordinator.py` | Zeus-side ZMQ server + SQLite job queue | ~300 |
| `zmq_sqlite_worker.py` | Rig-side GPU worker, ZMQ client | ~200 |

### Modified files (additive only):

| File | Change | Lines added |
|------|--------|-------------|
| `window_optimizer_integration_final.py` | ZMQ-SQLite gate (mirrors PWC gate) | ~30 |
| `window_optimizer.py` | `--use-zmq-sqlite` flag | ~10 |
| `agent_manifests/window_optimizer.json` | Flag in args_map + default_params | ~4 |

### Untouched:
- `persistent_worker_coordinator.py` — unchanged
- `coordinator.py` — unchanged
- `watcher_agent.py` — transparent, passes flags from manifest
- Steps 2-6, Chapter 13/14, all Pydantic models — no references to Step 1 IPC

---

## 4. Backwards Compatibility

- `--use-zmq-sqlite` is `false` by default in manifest
- Without the flag: identical behavior to today
- With the flag: new coordinator, same output files produced
- `run_trial_zmq_sqlite()` returns identical dict to `run_trial_persistent()`
- `_build_test_result_from_pw()` works unchanged on both

---

## 5. Key Improvements Over PWC

| Property | PWC | ZMQ+SQLite |
|----------|-----|-----------|
| SSH connections during compute | 16-24 persistent | 0 |
| Thread count on Zeus | 537 (unbounded) | 1 per ZMQ socket |
| Worker crash isolation | One crash = respawn storm | One crash = that chunk retried |
| Rig crash isolation | Affects Zeus coordinator | Zeus continues, rig reconnects |
| Job persistence on crash | Lost | SQLite survives |
| Spawn mechanism | 8 SSH processes per rig | 1 SSH call per rig |
| ROCm init | Once per worker startup | Once per worker startup |

---

## 6. Validation Evidence

- rrig6600 + rrig6600c 2-rig test completed cleanly: **100% — 537/537 chunks**
- 14,046 bidirectional survivors found
- 34:15 elapsed, zero crashes
- This is the first ever clean multi-rig run

---

## 7. Dependencies

```bash
pip install pyzmq --break-system-packages  # Zeus + all rigs
```

`pyzmq` is a single C-extension package, actively maintained, widely used
in distributed compute. No other new dependencies.

---

## 8. Request

Team Beta approval to:
1. Deploy `zmq_sqlite_coordinator.py` and `zmq_sqlite_worker.py`
2. Apply `apply_s158d_zmq_sqlite_integration.py` to add the flag
3. Set `use_zmq_sqlite: false` in manifest (disabled by default)
4. Run isolation test with `--use-zmq-sqlite` flag on rrig6600 alone
5. If stable, enable as default and retire PWC path
