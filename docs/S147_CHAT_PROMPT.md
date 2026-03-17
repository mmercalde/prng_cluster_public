# S147 Chat Prompt — PRNG Distributed Analysis System

## Session Priority

**P1 — Start here:**
1. Update chapter documentation per `docs/DOCUMENTATION_UPDATES_S146.md` — patch the 5 chapter files with S146 PWC invariants, hybrid kernel signatures, dashboard fixes
2. Launch production sweep Run 1 — `bash sweep_run1.sh` (manifest is restored to 1B seeds, 50 trials)
3. Chapter 13 wire-up (critical path — autonomous feedback loop)

---

## Current System State (end of S146)

**All commits synced at `8ac4047` on both remotes.**

### What was validated in S146
- Persistent Worker Coordinator (PWC) fully operational — all 4 sieve passes working
- 313 bidirectional survivors found (274 constant + 40 variable skip)
- 666 total in NPZ accumulator (`bidirectional_survivors_all.npz`)
- WATCHER confidence 1.00 — PROCEED on Step 1
- Web dashboard working — all routes 200, live throughput, live survivor counts
- `sweep_preprod.sh` validated end-to-end

### Key files changed in S146
| File | Commit | Change |
|------|--------|--------|
| `persistent_worker_coordinator.py` | `ec3cd1f` | localhost semaphore, ProgressWriter, log_gpu_result, update_trial_stats, full strategy dict, import time fix |
| `sieve_gpu_worker.py` | `cf1d1dc` | hybrid kernel sig split, phase2_threshold, coerce_threshold, custom_params, int32 casts, count clamp, full strategy dict |
| `web_dashboard.py` | `89087c5` | read_progress() always returns complete trial_stats |
| `sweep_preprod.sh` | `8ac4047` | NEW — pre-production validation script (50M seeds, 5 trials) |
| `agent_manifests/window_optimizer.json` | `8ac4047` | Restored to production: max_seeds=1B, trials=50, seed_start=0 |

### Architecture invariants added S146 (CRITICAL)
- PWC `_localhost_semaphore = threading.Semaphore(2)` — required for Zeus local dispatch
- Forward hybrid kernel tail: `threshold, unsigned long long a, unsigned long long c`
- Reverse hybrid kernel tail: `threshold, int offset` (a,c hardcoded inside)
- Hybrid uses `phase2_threshold` not `min_match_threshold` for both kernel AND post-filter
- Strategies must be full `StrategyConfig.to_dict()` — all 6 fields required
- `worker_pool_size=4` validated (not 8)
- `JOB_TIMEOUT_S=600` (not 300)
- PWC must call `log_gpu_result()` after every chunk for dashboard throughput
- PWC must call `update_trial_stats()` after each trial for dashboard survivor counts

---

## Infrastructure

**Zeus:** `rzeus`, `~/distributed_prng_analysis/`, `~/venvs/torch/bin/activate`
**Rigs:** `rrig6600` (192.168.3.120), `rrig6600b` (192.168.3.154), `rrig6600c` (192.168.3.162)
**Dashboard:** `45.32.131.224:5002`
**Git:** dual-push `origin` + `public` always

### Launch commands
```bash
# Pre-production test (50M seeds, 5 trials)
bash sweep_preprod.sh

# Production sweep Run 1 (1B seeds, 50 trials)
bash sweep_run1.sh

# Monitor
tail -f logs/sweep_run1_production.log
# or
tail -f logs/sweep_preprod.log
```

### Kill all workers
```bash
ssh rzeus "pkill -f 'watcher_agent.py'; pkill -f 'window_optimizer.py'"
ssh rrig6600 "pkill -f sieve_gpu_worker 2>/dev/null"
ssh rrig6600b "pkill -f sieve_gpu_worker 2>/dev/null"
ssh rrig6600c "pkill -f sieve_gpu_worker 2>/dev/null"
```

---

## Documentation to patch (S147 task)

Apply content from `docs/DOCUMENTATION_UPDATES_S146.md` to these 5 files:

1. `docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md` — append PWC S146 invariants section
2. `docs/CHAPTER_1_WINDOW_OPTIMIZER.md` — add persistent worker mode section
3. `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` — update sieve execution path
4. `docs/CHAPTER_12_WATCHER_AGENT.md` — update Step 1 execution path
5. `docs/COMPLETE_OPERATING_GUIDE_v2_0.md` — add PWC operating procedures

---

## P1 TODO (S147 order — do not reorder)

1. Chapter documentation updates (5 files per DOCUMENTATION_UPDATES_S146.md)
2. Launch production sweep Run 1
3. Chapter 13 wire-up (critical path — autonomous feedback loop)
4. Linear complexity Tier 1B
5. Binary matrix rank

## Backlog
- S110 root cleanup (884 files)
- sklearn warnings in Step 5
- Remove CSV writer from coordinator.py
- Regression diagnostics gate = True
- S103 Part 2
- Phase 9B.3 (deferred)
- Fix soak_s130.sh (calls coordinator.py directly, cannot test PWC)
- sweep_run2.sh, sweep_run3.sh, sweep_run4.sh

---

## NPZ Accumulator Status
- `bidirectional_survivors_all.npz` — 666 seeds, 22 fields
- `bidirectional_survivors_binary.npz` — 666 seeds (Step 2-6 input)
- Coverage: 0→510M (prior) + 560M→610M (S146 preprod)
- Next seed_start for Run 1: 0 (fresh, full 1B range)
