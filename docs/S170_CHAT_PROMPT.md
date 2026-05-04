# S170 Chat Prompt
**Date:** 2026-04-25+
**Branch:** s167-clean
**HEAD:** a6cd55e
**Focus:** Stability curve testing + DPM harness + BoTorch dual-GPU

---

## Cluster State

- Zeus: 2× RTX 3080Ti, `45.32.131.224` / `192.168.3.127`, alias `rzeus`
- rrig6600 (`192.168.3.120`): 8× RX 6600 — `cwsr_enable=0 mcbp=0` ✅
- rrig6600b (`192.168.3.154`): 8× RX 6600 — stock ✅
- rrig6600c (`192.168.3.162`): 8× RX 6600 — stock (cwsr causes faster crashes on this rig) ✅

**All rigs:** amdgpu-dkms 6.12.12, GDM disabled, multi-user.target permanent

**Zeus venv:** `~/venvs/torch/bin/activate`
**Rig venv:** `~/rocm_env/bin/activate`

---

## Active Branch: s167-clean

### Commits
```
a6cd55e docs(s167/s168/s169): session changelog 2026-04-24
b6cabe9 feat(s168/s169): startup jitter + per-worker pacing anti-hammer
4f2648e fix(s167): remove watcher warm-start leak; add s168a passive startup diagnostics
a6bc546 fix(s166): session_idx warm-start + bidirectional list clear  ← stable baseline
```

### Active Env Vars for S168/S169
```bash
PRNG_PWC_STARTUP_DIAG=1               # Enable startup telemetry → logs/pwc_startup_diag_simple.jsonl
PRNG_PWC_FIRST_ASSIGN_JITTER_SEC=3    # S168 startup de-sync jitter
PRNG_PWC_PER_WORKER_MIN_GAP_SEC=0.02  # S169 per-worker pacing
```

Both S168/S169 default OFF — must be explicitly enabled.

---

## Priority 1 — Stability Curve Test

### Purpose
Determine the maximum stable `seed_cap_amd` under sustained multi-trial load with S168+S169 enabled.

### Test Matrix
```
CAP=100000  TRIALS=5
CAP=150000  TRIALS=5
CAP=200000  TRIALS=5
```

### Command Template
```bash
CAP=100000
TRIALS=5
ssh rzeus "cd ~/distributed_prng_analysis && \
  rm -f logs/pwc_startup_diag_simple.jsonl optimal_window_config.json && \
  truncate -s 0 logs/netconsole_all_rigs.log && \
  source ~/venvs/torch/bin/activate && \
  PRNG_PWC_STARTUP_DIAG=1 \
  PRNG_PWC_FIRST_ASSIGN_JITTER_SEC=3 \
  PRNG_PWC_PER_WORKER_MIN_GAP_SEC=0.02 \
  PYTHONPATH=. nohup python3 agents/watcher_agent.py \
  --run-pipeline --start-step 1 --end-step 1 --force-step 1 \
  --params '{\"min_workers\": 24, \"seed_cap_amd\": '$CAP', \"window_trials\": '$TRIALS'}' \
  > logs/stability_cap_${CAP}_t${TRIALS}_\$(date +%H%M).log 2>&1 & echo PID: \$!"
```

Repeat with `CAP=150000` then `CAP=200000`.

### After Each Run — Check Results
```bash
ssh rzeus "cd ~/distributed_prng_analysis && \
  latest=\$(ls -t logs/stability_cap_*.log | head -1); \
  echo LOG=\$latest && \
  grep -E 'seeds/sec|Elapsed|Pipeline Summary|Step 1|ERROR|Traceback|FAILED|Complete' \$latest | tail -20 && \
  echo '--- netconsole ---' && tail -20 logs/netconsole_all_rigs.log"
```

### Results Table
```
cap      trials   result   elapsed   avg seeds/s   netconsole   script_write_failed   crashed_rig
100k     5        ?        ?         ?             ?            ?                     none
150k     5        ?        ?         ?             ?            ?                     ?
200k     5        ?        ?         ?             ?            ?                     ?
```

### Pass/Fail Criteria
```
PASS = all trials complete, netconsole clean, no script write failures
WARN = completes but netconsole has amdgpu/KFD faults OR script write failures
FAIL = any rig crash/reset/unreachable
```

### Important: Track Both Failure Modes
1. **GPU faults** — netconsole `GCVM_L2_PROTECTION_FAULT`, `qcm fence timeout`
2. **Transport failures** — `[PWC-TCP] 192.168.3.120:GPU* script write failed`

---

## Priority 2 — DPM Harness (after stability curve)

### Goal
Find optimal stable DPM settings for all 3 AMD rigs: target **900mV / 2100-2200MHz**

### Background
Previous Kaspa OC profile used **2250MHz / -150mV** on all 24 AMD GPUs. Current rigs are running stock DPM settings. Need to implement a proper DPM harness to test and validate optimal settings specifically for ROCm compute workloads (not mining).

### Test Approach
1. Start conservative: 900mV / 2100MHz
2. Run stability test at each setting
3. Increase frequency in 50MHz steps until instability
4. Record: crash frequency, GPU temp, power draw, seeds/sec

### DPM Commands
```bash
# Check current DPM on all rigs
for rig in rrig6600 rrig6600b rrig6600c; do
  echo "=== $rig ==="
  ssh $rig "for card in /sys/class/drm/card[0-7]/device; do
    echo \"\$(basename \$(dirname \$card)): \$(cat \$card/pp_dpm_sclk 2>/dev/null | grep '*')\"
  done"
done

# Force manual DPM level
for rig in rrig6600 rrig6600b rrig6600c; do
  ssh $rig "for card in /sys/class/drm/card[0-7]/device; do
    echo manual > \$card/power_dpm_force_performance_level
  done"
done
```

### Note
Requires Team Beta proposal before implementing persistent DPM service changes.

---

## Priority 3 — BoTorch Dual-GPU (after DPM harness)

### Goal
Replace Optuna TPE with BoTorch Bayesian optimization for the final Optuna step in the pipeline. Use both Zeus RTX 3080Ti GPUs for parallel acquisition function optimization.

### Background
BoTorch (PyTorch-based) provides:
- GPU-accelerated Gaussian process fitting
- qEI/qNEI acquisition functions for parallel candidate selection
- Better sample efficiency than TPE on expensive black-box functions
- Natural dual-GPU parallelism via `torch.device`

### Current State
- BoTorch is installed on Zeus (`pip_list.txt` confirms)
- Target: `window_optimizer_bayesian.py` — replace or augment `OptunaBayesianSearch`
- Must be proposed to Team Beta before implementation

### Proposed Integration Point
```python
# In window_optimizer_bayesian.py
# After N warmup trials (e.g. 5), switch from TPE to BoTorch
# Use both Zeus GPUs: device_0 = cuda:0, device_1 = cuda:1
# BoTorch fits GP on completed trials, generates next batch
```

### Acceptance Criteria (TB will require)
1. Falls back to Optuna TPE if BoTorch fails
2. Does NOT affect AMD rig workers
3. Uses only Zeus GPUs for optimization compute
4. Existing `--resume-study` passthrough still works
5. No changes to S167 WATCHER fix

---

## Key Commands Reference

```bash
# Kill everything
ssh rzeus "pkill -9 -f watcher_agent; pkill -9 -f window_optimizer"
for rig in rrig6600 rrig6600b rrig6600c; do
  ssh $rig "pkill -9 -f pwc_worker_service 2>/dev/null"
done

# Reset seed coverage
ssh rzeus "cd ~/distributed_prng_analysis && \
  source ~/venvs/torch/bin/activate && \
  python3 reset_seed_coverage.py java_lcg"

# Clear netconsole
ssh rzeus "truncate -s 0 ~/distributed_prng_analysis/logs/netconsole_all_rigs.log"

# Web dashboard
ssh rzeus "bash -c 'cd ~/distributed_prng_analysis && \
  source ~/venvs/torch/bin/activate && \
  nohup python3 web_dashboard.py > logs/dashboard.log 2>&1 </dev/null &' && \
  sleep 2 && ss -tlnp | grep 5002"

# Monitor all
bash ~/monitor_all.sh

# Live netconsole
ssh rzeus "tail -f ~/distributed_prng_analysis/logs/netconsole_all_rigs.log"

# Check git status
ssh rzeus "cd ~/distributed_prng_analysis && git log --oneline -5 && git status --short"
```

---

## Open Issues

1. **rrig6600 script write failed** — occurs during Trial 2+ worker restart phase on rrig6600 (.120). Non-fatal (trial completes) but indicates transport/I/O contention. Needs investigation.
2. **rrig6600c page fault signature** — `gfxhub / SQC (inst) / GCVM_L2_PROTECTION_FAULT` — different from rrig6600's `qcm fence timeout`. May require different mitigation than cwsr.
3. **S103 Part 2** — deferred from earlier sessions.
4. **Selfplay NN fix** — `inner_episode_trainer.py` still has hardcoded forbidden guard blocking NN in selfplay. Fix: remove forbidden check + add y-normalization to selfplay path.

---

## Repo

- Private: `git@github.com:mmercalde/prng_cluster_project.git`
- Public: `https://github.com/mmercalde/prng_cluster_public`
- Clone: `git clone https://github.com/mmercalde/prng_cluster_public.git`
- Active branch: `s167-clean`
- Dual-push always: `git push origin s167-clean && git push public s167-clean`
