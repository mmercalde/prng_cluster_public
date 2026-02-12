# SESSION CHANGELOG — S80
**Date:** 2026-02-11  
**Session:** 80  
**Focus:** Soak C v2.0.0 — Operational Autonomy Validation  

---

## Summary

Validated WATCHER v2.0.0 daemon through full operational autonomy soak test. Discovered and fixed lifecycle coupling bug (`self.running` shared between daemon and pipeline). Introduced `approval_route` policy for explicit authority routing between orchestrator and WATCHER. Deployed permanent GPU stability fixes to all 3 mining rigs.

---

## Changes

### 1. `approval_route` Policy (chapter_13_orchestrator.py)

**Problem:** With `test_mode=true` and `auto_approve_in_test_mode=true`, the orchestrator auto-executed internally and never created `pending_approval.json`. WATCHER daemon's `_poll_pending_approval()` was untestable.

**Fix:** New policy field `approval_route` with values `"orchestrator"` (legacy) or `"watcher"` (daemon route). Enum-validated with fallback to orchestrator.

- `approval_route = "orchestrator"` → Chapter 13 retains execution authority
- `approval_route = "watcher"` → Chapter 13 emits approval artifacts, WATCHER executes

**Lines changed:** ~8 lines in `chapter_13_orchestrator.py`

### 2. `_pipeline_running` Lifecycle Separation (watcher_agent.py)

**Problem:** `self.running` controlled both daemon loop and pipeline loop. When `run_pipeline()` completed, it set `self.running = False`, killing the daemon.

**Root cause:** Classic shared-state collision between two independent control loops.

**Fix:** Introduced `self._pipeline_running` for pipeline-only lifecycle control.

| Lifecycle | Control Flag |
|---|---|
| Daemon loop | `self.running` |
| Pipeline execution | `self._pipeline_running` |
| SIGTERM | kills both |
| CLI --run-pipeline | unaffected |

**Lines changed:** 5 (init, pipeline start, pipeline loop, pipeline finish, signal handler)

### 3. `approved_by` Audit Clarity (watcher_agent.py)

**Problem:** CLI `--approve` path archived with `approved_by: "watcher_daemon"` even though triggered via CLI.

**Fix:** `_archive_pending_approval()` now accepts `approved_by` parameter. CLI path passes `"watcher_cli"`, daemon path uses default `"watcher_daemon"`.

### 4. GPU Rig Stability Fixes (all 3 rigs)

**Deployed to:** rig-6600 (192.168.3.120), rig-6600b (192.168.3.154), rig-6600c (192.168.3.162)

- **udev rule** (`/etc/udev/rules.d/99-amdgpu-perf.rules`): Auto-set `perf=high` when GPUs detected
- **GFXOFF disable** (`amdgpu.gfxoff=0` in GRUB): Prevents soft lockup crashes from GPU sleep/wake storms
- **Reboot required** on all rigs for GFXOFF to take effect

**Root cause of rig-6600b crash:** Kernel soft lockup (`CPU#2 stuck for 26s` in `systemd-udevd`) during GPU enumeration on PCIe Gen1 with 8 GPUs.

---

## Soak C v2.0.0 Results

### Test Configuration
```
test_mode: true
auto_approve_in_test_mode: true
skip_escalation_in_test_mode: true
approval_route: watcher
```

### Full Path Validated
```
synthetic_draw_injector (60s interval)
    → new_draw.flag
        → chapter_13_orchestrator (detect, diagnose, trigger, LLM analysis)
            → pending_approval.json
                → WATCHER daemon (_poll_pending_approval)
                    → auto-approve (test_mode dual-flag check)
                        → run_pipeline(steps 3→6)
                            → archive to watcher_requests/archive/
                                → wait for next approval → repeat
```

### Metrics

| Metric | Result | Criteria |
|---|---|---|
| Duration | 46 minutes | >45 min ✅ |
| Daemon cycles | 69 | Sustained ✅ |
| Archives created | 39 | Accumulating ✅ |
| RSS memory | 258→263 MB (+5MB) | No leak ✅ |
| SIGTERM response | 1.558s | <2s ✅ |
| PID cleanup | Removed on stop | ✅ |
| State persistence | 69 cycles saved | ✅ |
| Double executions | 0 | ✅ |
| Tracebacks | 0 | ✅ |
| Telegram notifications | Working | ✅ |
| LLM (DeepSeek-R1-14B) | Grammar-constrained responses | ✅ |

### Kill Test
```
$ time PYTHONPATH=. python3 agents/watcher_agent.py --stop
Sending SIGTERM to daemon (PID 3755)...
Daemon stopped successfully
real 0m1.558s
```

PID file removed. daemon_state.json updated to state: STOPPED.

---

## Commits

| Hash | Description |
|---|---|
| `d4bba47` | fix: audit clarity — CLI approval uses approved_by='watcher_cli' |
| `66620be` | feat: Session 80 — approval_route, _pipeline_running fix, Soak C v2.0.0 PASSED |
| `53ee6fc` | restore: production policies after Soak C v2.0.0 |

---

## Files Modified

| File | Type | Change |
|---|---|---|
| agents/watcher_agent.py | MODIFIED | _pipeline_running, approved_by param (+4 lines) |
| chapter_13_orchestrator.py | MODIFIED | approval_route routing (~8 lines) |
| watcher_policies.json | MODIFIED | Added approval_route field |
| fix_gpu_rigs.sh | NEW | GPU stability deployment script |

---

## Architectural Significance

**Old Soak C (Session 60):** Validated epistemic autonomy — orchestrator detects, evaluates, auto-approves internally. Pipeline never actually dispatched through WATCHER.

**Soak C v2.0.0 (Session 80):** Validated operational autonomy — orchestrator emits approval artifact, WATCHER daemon detects, approves, dispatches pipeline, archives. Full authority separation proven.

This is the transition from "detection system" to "operational autonomous system."

---

## Policy Reference (Production vs Test)

| Policy Flag | Production | Test (Soak) | Purpose |
|---|---|---|---|
| `test_mode` | `false` | `true` | Enable synthetic injection, bypass safety gates |
| `auto_approve_in_test_mode` | `false` | `true` | WATCHER auto-approves without human |
| `skip_escalation_in_test_mode` | `false` | `true` | Suppress mandatory escalation |
| `approval_route` | `"orchestrator"` | `"watcher"` | Who owns execution authority |

---

## Next Steps

1. **Documentation sweep** — Update COMPLETE_OPERATING_GUIDE, Chapter 10, Chapter 13
2. **Reboot rigs** — Activate GFXOFF disable on all 3 mining rigs  
3. **Real learning cycle** — Invalidate outputs, run full pipeline through WATCHER with actual GPU training
4. **Chunk 3** (deferred) — APScheduler, scraper subprocess, decision chain persistence

---

*End of Session 80*
