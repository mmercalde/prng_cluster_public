# S151 Chat Prompt — PRNG Distributed Analysis System

## Session Priority

**P0 — Start here:**
1. `--force-step` flag for WATCHER — freshness check blocks every resume/restart
2. `sweep_run2.sh` — warm-start from Run 1 best params via `enqueue_trial()`

**P1 — After P0:**
3. Check Run 1 status — how many trials completed, NPZ accumulator count
4. If Run 1 complete: launch Run 2
5. Chapter 13 wire-up (critical path — autonomous feedback loop)

---

## Current System State (end of S150)

**All commits synced at `bd9db51` on both remotes.**

### What was validated in S149+S150
- **S149-B:** AMD 4-worker ceiling removed — `Device(gpu_id)` direct selection,
  no HIP/CUDA masking. All 3 rigs running 8 workers each (24 AMD + 2 Zeus = 26 GPUs)
- **Per-trial NPZ checkpoint:** Survivors written after every survivor-producing trial.
  No more end-of-run data loss risk.
- **slim_v1 IPC:** Workers return parallel arrays instead of list-of-dicts.
  Pass 3/4 throughput: 3,774 s/s → 84,000 s/s per rig (**22x gain**).
- **Run 1 running:** Study `window_opt_1774109563`, seed range 660M→1.73B,
  50 trials, Trial 8+ active, 708 seeds in NPZ accumulator.

### Run 1 status (as of S150 end)
| Item | Value |
|---|---|
| Study | `window_opt_1774109563` |
| Seed range | 660,000,000 → 1,733,741,824 |
| Trials completed | 7 |
| Best config | W5_O41 — 701 bidirectional |
| NPZ accumulator | 708 seeds |
| Checkpoint | Active |
| Workers | 8 per rig × 3 rigs + Zeus = 26 GPUs |

### Key files changed in S149+S150
| File | Commit | Change |
|---|---|---|
| `sieve_gpu_worker.py` | `c1c698d`, `65461c9` | Device(gpu_id), slim_v1 output |
| `persistent_worker_coordinator.py` | `c1c698d`, `65461c9` | Remove HIP/CUDA masking, slim_v1 parser |
| `window_optimizer_bayesian.py` | `5320108` | Per-trial NPZ checkpoint |
| `window_optimizer_integration_final.py` | `c39bba0` | strategy._survivor_accumulator wiring |
| `agent_manifests/window_optimizer.json` | `265889c` | success_condition fix |
| `apply_s149b_device_gpu_id.py` | `c1c698d` | S149-B patch script |
| `apply_s149_npz_checkpoint.py` | `5320108` | NPZ checkpoint patch script |
| `apply_s150_slim_v1_ipc.py` | `65461c9` | slim_v1 patch script |

### Architecture invariants added S149+S150
- **[S149-B]** Workers see all GPUs — no HIP/CUDA/ROCR masking in spawner
- **[S149-B]** `Device(gpu_id)` direct selection in `sieve_gpu_worker.py`
- **[S149-B]** `run_sieve_job(job, gpu_id)` explicit parameter with mismatch assertion
- **[S149-CKPT]** NPZ checkpoint fires after every survivor-producing trial (atomic)
- **[S149-CKPT]** `strategy._survivor_accumulator` set AFTER `strategy_map.get()`
- **[S150]** slim_v1 IPC — parallel arrays, JSON-line outer protocol preserved
- **[S150]** `_is_hybrid` driven from job context, not survivor content
- **[S150]** Coordinator enforces hybrid arrays present for hybrid jobs
- **[S150]** Legacy dict-list parser preserved for rollout safety
- **[S150]** Rigs have no git — deploy worker updates via scp from Zeus

---

## Infrastructure

**Zeus:** `rzeus`, `~/distributed_prng_analysis/`, `~/venvs/torch/bin/activate`  
**Rigs:** `rrig6600` (192.168.3.120), `rrig6600b` (192.168.3.154), `rrig6600c` (192.168.3.162)  
**Dashboard:** `45.32.131.224:5002`  
**Git:** dual-push `origin` + `public` always  
**Rig deploys:** No git on rigs — use scp from Zeus

### Monitor Run 1
```bash
# From Zeus directly:
cd ~/distributed_prng_analysis
tail -f logs/sweep_run1_production.log | grep --line-buffered -E 'S149-CKPT|NEW BEST|SAVE.*Trial|Bidirectional|Worker pool ready'

# NPZ accumulator count:
python3 -c "import numpy as np; d=np.load('bidirectional_survivors_all.npz'); print('Seeds:', len(d[list(d.keys())[0]]))"
```

### Resume Run 1 (if stopped)
```bash
# ALWAYS delete optimal_window_config.json first (freshness skip bug)
rm -f optimal_window_config.json
source ~/venvs/torch/bin/activate
PYTHONPATH=. python3 agents/watcher_agent.py --clear-halt
bash sweep_run1.sh --resume
```

### Kill all workers
```bash
ssh rzeus "pkill -f 'watcher_agent.py'; pkill -f 'window_optimizer.py'"
ssh rrig6600 "pkill -f sieve_gpu_worker 2>/dev/null"
ssh rrig6600b "pkill -f sieve_gpu_worker 2>/dev/null"
ssh rrig6600c "pkill -f sieve_gpu_worker 2>/dev/null"
```

### Deploy worker updates to rigs (no git)
```bash
scp rzeus:~/distributed_prng_analysis/sieve_gpu_worker.py ~/Downloads/
scp ~/Downloads/sieve_gpu_worker.py rrig6600:~/distributed_prng_analysis/
scp ~/Downloads/sieve_gpu_worker.py rrig6600b:~/distributed_prng_analysis/
scp ~/Downloads/sieve_gpu_worker.py rrig6600c:~/distributed_prng_analysis/
```

---

## P0: --force-step flag design

The freshness check in WATCHER skips Step 1 if `optimal_window_config.json`
exists and is recent. This blocks every resume and restart. Current workaround:
manually `rm -f optimal_window_config.json` before every launch.

**Proposed fix:** Add `--force-step N` to `watcher_agent.py` that bypasses
freshness check for step N. Wire into `sweep_run1.sh --resume` automatically.

**Files:** `agents/watcher_agent.py`, `sweep_run1.sh`, `sweep_run2.sh`

---

## P1: sweep_run2.sh design

After Run 1 completes all 50 trials:
- Read best params from `optimal_window_config.json`
- Create fresh Optuna study (do NOT use `add_trials()` — scores are range-specific)
- `enqueue_trial()` with Run 1 best params as trial 0 (warm-start only)
- Coverage tracker auto-advances `seed_start` beyond Run 1 range
- Save study name to `logs/sweep_run2_study_name.txt`

**4-run plan:**
| Run | Seed range | Status |
|---|---|---|
| Run 1 | 0 → 1,073,741,824 | Active |
| Run 2 | 1,073,741,824 → 2,147,483,648 | Pending |
| Run 3 | 2,147,483,648 → 3,221,225,472 | Pending |
| Run 4 | 3,221,225,472 → 4,294,967,295 | Pending |

---

## Backlog

- S110 root cleanup (884 files in project root)
- sklearn warnings in Step 5
- Remove dead CSV writer from `coordinator.py`
- Regression diagnostics gate → set to True
- Chapter 13 wire-up: `dispatch_selfplay()`, `dispatch_learning_loop()`
- Selfplay NN two-part fix
- Walk-forward simulation
- Remove legacy dict-list coordinator parser (after Run 1 proves slim_v1 stable)
- GPU utilization measurement Pass 3/4 (1s rocm-smi polling)
