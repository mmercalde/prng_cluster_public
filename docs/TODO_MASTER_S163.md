# MASTER TODO LIST — S163
**Compiled:** 2026-03-02 (S114) | **Updated:** 2026-04-06 (S163)
**Status:** S163 active. NPZ bug fixed. TCP-PWC production-ready. First complete Step 1 run achieved (S162 VICTORY — 887 bidirectional survivors).

---

## ✅ COMPLETED SINCE S159

| Item | Session |
|------|---------|
| ZMQ 3-rig + 4-node validation ladder | S159–S160 |
| S159G ROCm env vars (`HSA_ENABLE_SDMA=0` etc.) deployed + confirmed on all rigs | S159G |
| Coordinator env-invariant check (cgroup-based PID discovery, D-Bus-free) | S160 |
| `_best_effort_gpu_cleanup()` wired into ZMQ worker inter-chunk loop | S160 |
| amdgpu-dkms 6.12.12 root cause confirmed — stock kernel cannot handle 8 concurrent compute procs per GPU | S162 |
| All 3 rigs on `amdgpu-dkms 6.12.12`, packages pinned — do not upgrade | S162 |
| TCP-PWC transport implemented + validated — 2,240,701 sps, 26 GPUs, 10x over SSH-PWC | S161 |
| Two-phase startup (online→init→ready, lazy ROCm init, parallel GPU warmup) | S161 |
| Web dashboard integrated with TCP workers (hostname, worker_id, GPU data) | S161 |
| **First complete Step 1 run EVER** — 887 bidirectional survivors, 42:36, 26/26 GPUs | S162 |
| `optimal_window_config.json` written (W6_O64_evening_S3-37_FT0.68_RT0.7) | S162 |
| `bidirectional_survivors.json` written (887 seeds) | S162 |
| `convert_survivors_to_binary.py` NPZ bug fixed (duplicate `import numpy as np`) | S163 |
| NPZ conversion verified OK — 887 survivors, 64.9x compression | S163 |

---

## 🔴 P1 — HIGH PRIORITY (This Session / Next)

### Run Step 2
- [ ] **Run Step 2 — Scorer Meta-Optimizer** with 887 survivors from S162 victory run:
  ```bash
  ssh rzeus "cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate && \
    PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 2 --end-step 2 \
    2>&1 | tee logs/s163_step2_run1.log"
  ```
  NPZ is now valid. This is the immediate next action after S163 P1/P2 items land.

---

## 🟠 P2 — MEDIUM PRIORITY (Active S163 Work)

### S163 — Remove `free_all_blocks()` Race Condition
- [ ] **Implement S163 proposal** — remove `free_all_blocks()` from `_best_effort_gpu_cleanup()`
  in `sieve_gpu_worker.py`. Add sampling-gated memory instrumentation behind `S163_MEM_DEBUG=1`.
  TB proposal: `docs/PROPOSAL_FREE_ALL_BLOCKS_REPLACEMENT_v1_0.md` — **approved**.
  - Sample every 25 chunks (not every chunk)
  - Log before AND after cleanup: `pool_used`, `pool_total`, `n_free_blocks`, `VmRSS`, `VmSize`
  - Threshold breach logging always active (not gated)
  - Staged validation: 500K → 1M → 2M seeds
  - Enables raising `seed_cap_amd` beyond 100K if 2M passes clean

### Zeus `cudaErrorDevicesUnavailable` Fix
- [ ] **Write TB proposal** for Zeus GPU keepalive fix.
  - 250/10,738 chunks failed Trial 3 reverse hybrid — Zeus GPUs entering P8 idle between chunks
  - Error at `sieve_filter.py:291` — `Device.__exit__` after kernel completes (context lost between chunks)
  - Three options (TB to rule):
    - Option A: `sudo nvidia-smi -pm 1` persistence mode before launch
    - Option B: Tie `_localhost_semaphore` to `gpu_count` or `max_per_node`
    - Option C: CuPy device context keepalive on Zeus local sieve path
  - Current impact: ~2.3% of Trial 3 reverse hybrid seeds missed (retried successfully)

### TCP-PWC Wire-up in WATCHER Manifest
- [ ] Wire `--pwc-transport tcp` into WATCHER `agent_manifests/window_optimizer.json`
  `default_params` so it activates automatically without manual flag.
- [ ] Sync `default_params.use_zmq_sqlite` (true) vs `param_docs.use_zmq_sqlite.default` (false) — inconsistency flagged S159, still unresolved. TB recommendation: keep false until 4-node TCP soak fully complete.

### NPZ Warning Tuning
- [ ] **Fix low-variance warning** in `convert_survivors_to_binary.py` — warning fires incorrectly
  when match rate has natural discrete distribution (small window_size). Warning should be aware
  of `window_size` before flagging low unique-value count. Pre-S103 degenerate case (all identical
  values) is distinct from natural discretization (W6 → max 7 possible values). Medium priority.

---

## 🟡 P3 — MEDIUM PRIORITY (Next 3-6 Sessions)

### Chapter 13 — Autonomy Wire-up
- [ ] Wire `dispatch_selfplay()` into WATCHER post-Step-6
- [ ] Wire `dispatch_learning_loop()` into WATCHER
- [ ] Wire Chapter 13 orchestrator into WATCHER daemon
- [ ] Integration test: WATCHER → Chapter 13 → Selfplay full loop

### Selfplay NN Fix
- [ ] Remove forbidden guard + add y-normalization to `inner_episode_trainer.py` selfplay path
  (S121 fix applied to `train_single_trial.py` at commit `6e5f76c` — not yet in selfplay)

### Pre-warm CuPy Kernel Cache
- [ ] Pre-warm CuPy kernel cache on rigs — reduces cold-start ROCm init 90s → ~10s
  (identified S161, not yet implemented)

### Session-Scoped Worker Persistence
- [ ] TCP worker persistence across trials within a session (currently workers restart per trial)

### Ephemeral Coordinator Benchmark
- [ ] Run original ephemeral coordinator benchmark for complete comparison table
  (ZMQ vs SSH-PWC vs TCP-PWC vs ephemeral — only three of four measured)

---

## 🟢 P4 — DEFERRED

- [ ] **S110 root cleanup** — 884 stray files in project root. Low urgency.
- [ ] **sklearn warnings in Step 5** — harmless deprecation warnings.
- [ ] **Remove dead CSV writer from `coordinator.py`**
- [ ] **Regression diagnostic gate=True**
- [ ] **S103 Part 2**
- [ ] **Phase 9B.3**
- [ ] **k_folds runtime clamp** — `val_fold_size < 3000` edge case. TB review needed.
- [ ] **Upload stale files to Claude Project** — `agents/watcher_agent.py`,
  `persistent_worker_coordinator.py`, `window_optimizer_integration_final.py`,
  `hybrid_strategy.py`, updated chapter docs.
- [ ] **S159 Hardening** — `cleanup()` should not delete .npz chunk files if output write failed.
  File: `zmq_sqlite_coordinator.py` → `run_sieve_pass()` finalization block.
- [ ] **Confirm `skip_sequences`/`strategy_ids` empty-list behavior** in ZMQ result path.

---

## Architecture Invariants (never break)
- All 3 rigs pinned: `amdgpu-dkms 6.12.12` — never run `apt upgrade` without checking holds
- Zeus semaphore: `_localhost_semaphore = threading.Semaphore(2)`
- `seed_cap_amd=100000` is current stable config — do not raise without S163 free_all_blocks validation
- `seed_cap_nvidia=5000000`
- `bidirectional_survivors_binary.npz` always git-tracked; commit after every Step 1 run
- `watcher_policies.json` version-controlled
- Dual-push every commit: `git push origin main && git push public main`
- Never commit from Claude sandbox
- TB approval required before architectural changes
- ZMQ ports: job=5557, result=5558
- Required ROCm vars: `HSA_ENABLE_SDMA=0`, `HSA_ENABLE_RUNTIME_POWER_MGMT=0`, `AMDGPU_NO_POWER_PROFILE=1`, `HSA_OVERRIDE_GFX_VERSION=10.3.0`, `ROCR_VISIBLE_DEVICES=<gpu_id>`
- TCP-PWC is current default transport for production runs
- Fix forward, never restore from backup
