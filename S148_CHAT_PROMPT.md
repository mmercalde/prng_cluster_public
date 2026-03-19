# S148 Chat Prompt — PRNG Distributed Analysis System

## Session Priority

**P1 — Start here:**
1. Resume production sweep Run 1 — clear halt, delete stale outputs, relaunch
2. Monitor hybrid forward prune rate from Run 1 logs
3. Chapter 13 wire-up (critical path — autonomous feedback loop)

---

## Current System State (end of S147)

**All commits synced at `bf2549b` on both remotes.**

### What was completed in S147
- 5 chapter docs patched with S146 PWC invariants (commits `9ca2671` + `c797045`)
- Dashboard fixes: trial_stats persistence + GPU clock → seeds/s display (`60393a0`)
- Step 1 execution flow documented — both paths with pruning gates (`0e01dcb`)
- Q0/Q1/Q2 patches deployed and verified 19/19 on live Zeus hardware (`667ece5`)
- Verification scripts committed (`bf2549b`)
- ser8 docs folder cleaned — 34 canonical files retained
- TB ruling request submitted and rulings applied (Q4 staged architecture rejected by Michael)

### Q0/Q1/Q2 Patches (commit `667ece5`)

| Patch | File | Change |
|-------|------|--------|
| Q0A | `persistent_worker_coordinator.py` | Hybrid forward gate — skip Pass 4 when fwd=0 |
| Q0B | `window_optimizer_integration_final.py` | Same gate in legacy coordinator path |
| Q1 | `agents/watcher_agent.py` | `{1: 0}` in step_timeout_overrides → infinite timeout |
| Q2 | `persistent_worker_coordinator.py` | balanced_hybrid single strategy for full scan |

**Apply order invariant:** Q2 before Q0A (Q0A references `_hybrid_strategies` from Q2).
**Deployment:** `apply_s147_q0_q1_q2.py --dry-run` then live. Full mode only.

### Sweep Run 1 Status
- Launched S147, failed after 120 minutes (Q1 bug — now fixed)
- Trial 1 partially complete: constant-skip passes done (719 survivors), hybrid timed out
- NPZ still at 666 seeds (nothing written — Trial 1 never completed)
- WATCHER halted — halt file present
- Study: `window_opt_1773792529`

---

## Infrastructure

**Zeus:** `rzeus`, `~/distributed_prng_analysis/`, `~/venvs/torch/bin/activate`
**Rigs:** `rrig6600` (192.168.3.120), `rrig6600b` (192.168.3.154), `rrig6600c` (192.168.3.162)
**Dashboard:** `45.32.131.224:5002` (VPS port forwards to Zeus:5000)
**Git:** dual-push `origin` + `public` always

### Resume sweep Run 1
```bash
# Clear halt
ssh rzeus "cd ~/distributed_prng_analysis && \
  source ~/venvs/torch/bin/activate && \
  python3 -c 'from agents.safety import clear_halt; clear_halt()'"

# Delete stale output files
ssh rzeus "rm -f ~/distributed_prng_analysis/optimal_window_config.json \
                  ~/distributed_prng_analysis/bidirectional_survivors.json"

# Relaunch
ssh rzeus "cd ~/distributed_prng_analysis && \
  source ~/venvs/torch/bin/activate && \
  nohup bash sweep_run1.sh > logs/sweep_run1_production.log 2>&1 &"

# Monitor
ssh rzeus "tail -f ~/distributed_prng_analysis/logs/sweep_run1_production.log"
```

### Dashboard
```bash
ssh rzeus "cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate && \
  nohup python3 web_dashboard.py > logs/dashboard.log 2>&1 & sleep 3 && ss -tlnp | grep 5000"
```

### Kill all workers
```bash
ssh rzeus "pkill -f 'watcher_agent.py'; pkill -f 'window_optimizer.py'"
ssh rrig6600  "pkill -f sieve_gpu_worker 2>/dev/null"
ssh rrig6600b "pkill -f sieve_gpu_worker 2>/dev/null"
ssh rrig6600c "pkill -f sieve_gpu_worker 2>/dev/null"
```

---

## P1 TODO (S148 order — do not reorder)

1. Resume production sweep Run 1
2. Monitor hybrid forward prune rate
3. Chapter 13 wire-up

## Backlog (do not reorder)
- Project knowledge uploads: `agents/watcher_agent.py`, `persistent_worker_coordinator.py`,
  `window_optimizer_integration_final.py`, `hybrid_strategy.py`, updated 5 chapter docs
- sklearn warnings Step 5
- Remove CSV writer from coordinator.py
- Regression diagnostics gate = True
- S103 Part 2
- Phase 9B.3 (deferred)
- S110 root cleanup (884 files)
- sweep_run2.sh, sweep_run3.sh, sweep_run4.sh

---

## Key Architecture Notes for S148

### Step 1 timing with Q2 (single strategy)
- Expected per-trial: ~2.5-3 hours (was ~9-11 hours with 5 strategies)
- 50 trials → ~125-150 hours for Run 1 (5-6 days continuous)
- Step 1 timeout: infinite (Q1 fix — `{1: 0}` in overrides)

### Q0 gate monitoring
First session after resuming, check logs for gate firing:
```bash
ssh rzeus "grep 'Hybrid forward zero survivors\|Q0 gate' \
  ~/distributed_prng_analysis/logs/sweep_run1_production.log | head -20"
```
If gate fires frequently → good (saves Pass 4 time).
If gate never fires → hybrid forward always finds survivors (expected for CA data).

### NPZ protocol
After Run 1 completes:
```bash
git add -f bidirectional_survivors_all.npz bidirectional_survivors_binary.npz
git commit -m "data(S148): sweep run 1 complete — seeds 610M→1.68B"
git push origin main && git push public main
```

### Project knowledge files that need updating
- `agents/watcher_agent.py` — project has stale 1,863-line root copy, canonical is 3,084 lines
- `persistent_worker_coordinator.py` — not in project knowledge at all
- `window_optimizer_integration_final.py` — not in project knowledge at all
- `hybrid_strategy.py` — not in project knowledge at all
- 5 updated chapter docs from S147 (already on ser8, need upload to Claude Project)
