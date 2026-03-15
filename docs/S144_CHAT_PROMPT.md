# S144 Session Chat Prompt
**Date:** 2026-03-14
**Previous Session:** S143
**Last Commits:** `51aed27` (S142) + S143 docs commit (pending push)

---

## System State

Distributed PRNG functional mimicry system. Modular, configurable, ML & AI
compatible. Not lottery-specific — designed for any deterministic PRNG output.

**Cluster:** Zeus (2×RTX 3080Ti, `rzeus`) + 2 rigs (8×RX6600 each)
**Project root:** `~/distributed_prng_analysis/`
**Public repo:** `github.com/mmercalde/prng_cluster_public`
**Dashboard:** `http://45.32.131.224:5002`

---

## What Was Completed in S143

S143 was an exploratory research session — PA Pick 3 experiment.

### PA Pick 3 Scraper — `pa_pick3_scraper.py` Rev 1.1
- Adapted from `daily3_scraper.py` Rev 1.5
- Handles PA Wild Ball format (`NNNWWild` → strip suffix, take first 3 chars)
- Output identical format to `daily3.json` — fully pipeline-compatible
- PA data: 18,003 draws, 2000–2026, full 0–999 space

### PA Sieve Validation Harness — `pa_sieve_validation_harness.py`
- CPU-side 3-tier harness validating sieve correctness
- Key finding: reverse sieve = forward kernel fed `draws[::-1]`, NOT backward LCG
- Confirmed: forward/reverse are independent filters, not mirrors

### PA Step 1 Experiment (15 trials, 10M seeds, direct `window_optimizer.py`)
- **389,041 total bidirectional survivors** across 15 trials
- Best trial: W3_O59_evening — 220,168 bidirectional survivors
- Forward (744,306) ≠ Reverse (632,454) — clean independent sieves confirmed
- W2/W3 dominate — very short persistence regime
- W8 trials produced only 8–12 survivors

### Key Finding
PA Pick 3 midday draws are confirmed software RNG (official PA docs state:
"A Random Number Generator (RNG), consisting of secure computerized systems,
selects at random the numbers"). Evening draws are physical ball machines.
The session-split follow-up (midday vs evening independently) is required to
isolate which session drives the signal. Experiment deferred.

### Architectural Violation Identified
Warm-start hardcode in `window_optimizer_bayesian.py` ~line 547:
```python
_ws_params = {'window_size':8,'offset':43,'skip_min':5,
              'skip_max':56,'forward_threshold':0.49,'reverse_threshold':0.49}
study.enqueue_trial(_ws_params)
```
This is CA-specific empirical data hardcoded in a general-purpose optimizer.
Confirmed to suppress PA signal in S143. Fix is P1 for S144.

### System Reverted
- `agent_manifests/window_optimizer.json` → `daily3.json` ✅
- `agent_manifests/prediction.json` → `daily3.json` ✅
- PA output files deleted (bidirectional_survivors.json, optimal_window_config.json, etc.) ✅
- `bidirectional_survivors_binary.npz` left intact (git-tracked) ✅
- Working directory clean — ready for CA Step 1 run ✅

---

## P1 TODO for S144

### 1. Fix Warm-Start Hardcode (FIRST)
**File:** `window_optimizer_bayesian.py` ~line 547
**Change:** Remove hardcoded W8_O43 enqueue. Drive warm-start entirely from
`trial_history_context` — if all 6 params present, enqueue; otherwise Optuna
explores freely. CA manifest supplies W8_O43 explicitly via `default_params`.

```python
# REMOVE this entire block:
if not _resume:
    _ws_params = {'window_size':8,'offset':43,...}
    study.enqueue_trial(_ws_params)

# REPLACE with:
if not _resume and trial_history_context:
    if all(v is not None for v in [_ww, _wo, _wsk, _wsx, _wf, _wr]):
        study.enqueue_trial(_ws_params)
        print(f"   🌡️  Warm-start from trial history: {_ws_source}")
    # else: no warm-start — Optuna explores freely
```

### 2. Clean CA Step 1 Run
After warm-start fix verified, launch clean CA Step 1:
```bash
ssh rzeus "cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate && \
nohup bash -c 'PYTHONPATH=. python3 agents/watcher_agent.py \
--run-pipeline --start-step 0 --end-step 1 \
> logs/ca_step01_s144.log 2>&1' &"
```

### 3. Wire dispatch_selfplay() into WATCHER post-Step-6
### 4. Wire Chapter 13 orchestrator into WATCHER daemon

---

## Current File State on Zeus

```
~/distributed_prng_analysis/
  daily3.json                          ← CA data (canonical)
  pa_pick3.json                        ← PA data (experiment, keep for future)
  pa_pick3_scraper.py                  ← Rev 1.1
  pa_sieve_validation_harness.py       ← S143 harness
  bidirectional_survivors_binary.npz   ← S138 data (PA-contaminated, will be
                                          overwritten on next clean CA Step 1)
  agent_manifests/window_optimizer.json ← lottery_file = daily3.json ✅
  agent_manifests/prediction.json      ← lottery_history = daily3.json ✅
  docs/SESSION_CHANGELOG_20260314_S143.md
  docs/TODO_MASTER_S143.md
```

No `bidirectional_survivors.json`, `optimal_window_config.json`,
`train_history.json`, `holdout_history.json` — working directory clean.

---

## Session Start Checklist

```bash
# 1. Clone public repo
git clone --depth 1 https://github.com/mmercalde/prng_cluster_public.git \
    /home/claude/prng_cluster_public

# 2. Verify manifest reverted correctly
ssh rzeus "grep lottery_file \
    ~/distributed_prng_analysis/agent_manifests/window_optimizer.json"
# Expected: "lottery_file": "daily3.json"

# 3. Verify working directory clean
ssh rzeus "ls ~/distributed_prng_analysis/bidirectional_survivors.json \
    ~/distributed_prng_analysis/optimal_window_config.json 2>/dev/null \
    || echo 'Clean — ready for Step 1'"

# 4. Check DB state
ssh rzeus "cd ~/distributed_prng_analysis && \
    source ~/venvs/torch/bin/activate && \
    PYTHONPATH=. python3 -c \"
import sqlite3
conn = sqlite3.connect('prng_analysis.db')
rows = conn.execute('SELECT trial_number, window_size, offset, session, trial_score FROM step1_trial_history ORDER BY trial_score DESC LIMIT 5').fetchall()
print(f'Trial history: {len(rows)} rows')
for r in rows: print(f'  T{r[0]}: W{r[1]}_O{r[2]}_{r[3]} score={r[4]:.0f}')
conn.close()
\""
```

---

## Architecture Invariants
- Step order static: 0→1→2→3→4→5→6
- Authority: Chapter 13 decides, WATCHER executes
- GPU isolation: parent never initializes CUDA before NN subprocess spawn
- Manifest param governance: every CLI param in `default_params` or dropped
- `bidirectional_survivors_binary.npz` never gitignored — explicit commit after Step 1
- Never restore from backup — fix forward
- Always dual-push: `git push origin main && git push public main`
- PYTHONPATH=. required for all WATCHER commands
- No `--params` on Step 1 WATCHER launch — silently dropped, Optuna loads from manifest
- Always run in nohup, not tmux

---

## Key Architecture — Warm-Start Flow (post-fix)
```
TRSE Step 0
  → writes trse_context.json (regime type, warm-start candidates)
      ↓
Step 1 WATCHER launch
  → reads step1_trial_history from DB (recent successful trials)
  → builds trial_history_context with warm_start_* params
  → passes to window_optimizer_bayesian.py
      ↓
OptunaBayesianSearch.search()
  → if trial_history_context has all 6 warm-start params → enqueue Trial 0
  → else → Optuna explores freely (correct for new datasets)
```
