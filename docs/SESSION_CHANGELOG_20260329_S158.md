# SESSION CHANGELOG — S158
**Date:** 2026-03-29  
**Session:** S158  
**Author:** Claude (Team Alpha Lead Dev)  
**Status:** Complete

---

## Summary

Long session focused on multi-rig crash root cause investigation, isolation
testing, multiple patch cycles, and design of a replacement coordinator
architecture (ZMQ+SQLite).

---

## Key Findings

### Root Cause — Multi-Rig Crashes
- **PWC launches 537 threads simultaneously** (one per chunk) with no throttle
- Under multi-rig load, GIL thrashing causes SSH pipe read timeouts
- Workers marked dead → respawn storm → rig kernel panic
- **fd exhaustion ruled out**: Zeus fd count peaked at 81 (limit 1,048,576)
- **Hardware ruled out**: both rigs stable in isolation, crash only together
- **PageTables misread corrected**: low PageTables at crash = symptom of dying rig, not cause

### Isolation Test Results
| Test | Config | Result |
|------|--------|--------|
| rrig6600 isolated | 8 AMD + Zeus | ✅ Full 537-chunk sweep |
| rrig6600c isolated (forward) | 8 AMD + Zeus | ✅ Full 537-chunk sweep |
| rrig6600c isolated (reverse) | 8 AMD + Zeus | ⚠️ Stalled chunk 88/537 — hang evidence |
| rrig6600 + rrig6600c | 16 AMD + Zeus | ✅ 537/537 chunks, 14,046 bidi survivors |
| All 3 rigs | 24 AMD + Zeus | ❌ Hard kernel panic on rrig6600c |

### First Ever Clean 2-Rig Run
- rrig6600 + rrig6600c completed 537/537 chunks
- 14,046 bidirectional survivors
- 34:15 elapsed, zero crashes
- Proves S158B dispatch fix is stable for 2-rig config

---

## Commits This Session

| Commit | Description |
|--------|-------------|
| `ca2f2c3` | fix(s158): graceful empty NPZ on zero survivors |
| `b0235e7` | fix(s158b-v3): bounded dispatch — progress-aware ThreadPoolExecutor |
| `8d62491` | fix(s158c): ROCM_CLIFF_NODES inter-node spawn stagger |
| `c9e7221` | fix(s158c): increase inter-node stagger 10s→30s |
| `d1301e1` | revert(s158c): roll back cliff-node stagger — caused spawn crashes |

---

## Patches Deployed

### S158 — Zero-survivor NPZ guard (`ca2f2c3`)
- `convert_survivors_to_binary.py` — early exit when n==0
- Prevents `ValueError: zero-size array` crash

### S158B-v3 — Bounded dispatch (`b0235e7`)
- `persistent_worker_coordinator.py` — ThreadPoolExecutor replaces unbounded thread launch
- `max_workers = min(num_workers, len(chunks))`
- Progress-aware `wait()` loop — no permanent hangs
- `shutdown(wait=False, cancel_futures=True)`
- TB approved after 3 revision cycles

### S158C — REVERTED
- Inter-node stagger caused spawn crashes on rrig6600 and rrig6600b
- Was solving the wrong problem (crash during spawn vs crash during dispatch)
- Fully reverted at `d1301e1`

---

## Infrastructure Changes

### Zeus ulimit
- Raised from 1,024 → 1,048,576
- Set in `/etc/security/limits.conf`
- Note: did NOT fix crashes (fd count peaked at 81, far below limit)

### seed_cap_amd
- Restored to 2,000,000 (production value)

### Coverage
- Reset to seed_start=0 for clean Run 1

---

## New Architecture — S158D ZMQ+SQLite

### Problem with PWC
- 8 persistent SSH connections per rig during entire run
- Zeus babysits 24 SSH processes instead of GPUs doing pure compute
- Spawn phase hammers rigs simultaneously

### Solution
- ZMQ PUSH/PULL for job dispatch (TCP, no SSH during compute)
- SQLite for durable job state (Zeus-only writer, WAL mode)
- Workers launched ONCE via SSH, then run independently
- Like crypto miners — connect, pull job, compute, push result, repeat

### Files Created
- `zmq_sqlite_coordinator.py` — Zeus-side coordinator (~400 lines)
- `zmq_sqlite_worker.py` — Rig-side GPU worker (~180 lines)
- `apply_s158d_zmq_sqlite_integration.py` — Integration patch script
- `docs/PROPOSAL_ZMQ_SQLITE_COORDINATOR_S158D_v1_0.md`

### Existing Files Modified (minimal)
- `window_optimizer_integration_final.py` — ZMQ-SQLite gate added
- `window_optimizer.py` — `--use-zmq-sqlite` flag added
- `agent_manifests/window_optimizer.json` — flag in args_map + defaults

### TB Guardrails Implemented (v2)
1. SQLite schema: `lease_expires_at`, `attempt_count`, `claimed_by`
2. Zeus sole SQLite writer — workers ZMQ only
3. Idempotent result ingestion — duplicate chunk_id silently ignored
4. Worker identity explicit: `"hostname:gpuN"` bound to claims
5. JSON only — no pickle
6. venv install instructions

### TB Status
- S158D approved with mandatory guardrails
- Risk rated Medium (not Low as proposed)
- Validation ladder: 1-rig → 2-rig → 3-rig before default promotion

---

## Monitoring Tools Created
- `monitor_all.sh` — launches web dashboard + 3 gnome-terminal tabs
  (Live CLI, Page Memory, Crash Monitor)

---

## TODO Carried Forward
1. S158D validation — rrig6600 isolation test with `--use-zmq-sqlite`
2. rrig6600b 2-rig test (rrig6600 + rrig6600b) — never done
3. Full 3-rig test with S158B — rrig6600b still not tested with others
4. Install pyzmq on all nodes before S158D testing
5. SESSION_CHANGELOG committed and dual-pushed (this file)
6. Coverage reset to seed_start=0 before next production run

---

## Invariants Verified This Session
- `bidirectional_survivors_binary.npz` git-tracked ✅
- `watcher_policies.json` version-controlled ✅
- Dual-push enforced on all commits ✅
- Per-worker `CUPY_CACHE_DIR` active (S157) ✅
- S158B bounded dispatch active ✅
- S158C stagger REVERTED ✅
