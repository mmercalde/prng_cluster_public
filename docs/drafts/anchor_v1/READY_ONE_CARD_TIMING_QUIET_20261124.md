# READY · One-card timing, 2²⁶ on quiet cores (2026-11-24): committed, 79.6 s per window

- **Lead answer:** 17:03, at the end of `logs/e2e_session/LEAD_ANSWERS_ONE_CARD_TIMING_20261124.txt`.
- **Owner:** go 14:51, cap one-off 14:58 for this test only; owner told 15:43.
- **Evidence:** `data/anchor_v1_logs/onecard_timing/q26/` in this worktree. It holds `driver.log`, `optimizer.out`, `worker_gpu0.log`, `ledger.txt`, `card_mem.csv` (203 samples at 0.5 s), the load files, `search_open.log` and the main-database checks.
- **Settings:** run H's, unchanged from s26: window 12, anchor 25, midday, 0.30/0.30, skip 0-16, skip delivery on, [2³¹, 2³¹+2²⁶).
- **Run tree:** `~/onecard_q26_20261124`, a detached worktree of `41ad509`.

## Driver change: pre-flight only (disclosed)
Main had moved from `41ad509` to `827f4b4` since RERUN2, with two docs-only owner commits (`f090c69`, `827f4b4`). They touch `AGENTS.md`, `CLAUDE.md`, `docs/CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md` and `docs/TFM_PROJECT_FACTS_SKILL.md` (`driver.log:2`).

The driver's pre-flight required main to be at exactly `41ad509`, so it would have refused. I widened that one check. It now passes only if all three hold:
- `41ad509` is an ancestor of main's HEAD;
- main is clean;
- the diff from `41ad509` to HEAD touches only `*.md` files.

The run tree is still built at `41ad509`, so the code under test is the same as s24 and s26. Nothing else in the driver changed (`run_onecard.sh`, lines after `# [QUIET run, 17:03]`).

**Lead: if this check should have been a stop, say so and I will mark the result void.**

## Plain summary
- **On quiet cores, a 2²⁶ trial takes 79.6 s, against 153.2 s on shared cores: 1.9 times faster.** Windows per day at 2²⁶ rise from about 560 to about **1,085**.
- **Survivor handling cost about 72-75 µs per survivor here, against about 160 µs shared.** About half of the shared figure was contention.
- **The card is still not the limit.** It averaged 0.3 % use during phase 3 and peaked at 16 %. Phase 3 waits on the host CPU even on a quiet core.
- **Survivor counts are identical to s26** in every phase. The search id is also the same (`s1s-fa1f3378…`), as expected from identical parameters.

## Per phase (miner ledger `stripes`, one stripe per phase, one worker)

| Phase | Quiet s | Shared s (s26) | Survivors (both runs) | Card use, quiet (mean / peak) |
|---|---|---|---|---|
| 1 `java_lcg` | 3.13 | 6.09 | 0 | — |
| 2 `java_lcg_reverse` | 1.18 | 2.59 | 0 | — |
| 3 `java_lcg_hybrid` | 54.09 | 105.30 | 592,791 | 0.3 % / 6 % (108 samples) |
| 4 `java_lcg_hybrid_reverse` | 14.31 | 27.29 | 148,709 | 4.9 % / 14 % (28 samples) |
| **Stripes total** | **72.70** | 141.3 | — | — |
| **Trial** (`trials.created_at` → `finalized_at`) | **79.57** | 153.2 | final 3,044 rows | — |
| Wall (launch → optimizer exit) | 96.4 | 184.8 | — | — |
| **Windows per day** (86,400 ÷ trial time) | **≈ 1,085** | ≈ 560 | — | — |

**How the per-survivor cost was worked out:** subtract run H's volume-gate kernel times, as in RERUN2, then divide by the survivor count.
- phase 3: (54.09 − 9.9) s ÷ 592,791 ≈ **75 µs** per survivor;
- phase 4: (14.31 − 3.7) s ÷ 148,709 ≈ **71 µs** per survivor.

**Fixed cost per trial** (trial time minus stripes) was 6.9 s, against 12.0 s shared.

## 2²⁸ extrapolation [UNVERIFIED]
Scaling each stripe ×4 in seeds and adding the 6.9 s fixed cost gives about 4 × 72.7 + 6.9 ≈ **298 s per trial**, or **≈ 290 windows per day** on one quiet card.
- This is not measured at 2²⁸.
- It assumes the survivor count grows linearly with the range, as phase 3 did from 2²⁴ to 2²⁶ (148,004 → 592,791, ×4.005).
- Phases 1 and 2 grew less than ×4 from 2²⁴ to 2²⁶, so ×4 probably overstates them slightly.

## Card, memory, load
- **Peak card memory:** 2,495 MiB, including `sunshine`'s 264 MiB, the same as s24 and s26.
- **Peak RSS** (optimizer plus worker): 969 MiB. The memory guard did not fire, and 11 GiB was available at the start.
- **Load:**

  | | Load | Time |
  |---|---|---|
  | Before | 0.53 / 0.82 / 3.82 | 17:02:45 |
  | After | 0.99 / 0.90 / 3.52 | 17:04:30 |

  Before the run, no process above 1.5 % CPU was on cores 6 or 7; the top entries were idle `claude` sessions (`load_before.txt`). The learning batch had finished.

## Scope kept and state left
- **The search, credit and publication stayed inside the run tree**, as in s24 and s26:
  - S7 credited `[2147483648, 2214592512]`;
  - generation published, 3,044 rows;
  - coverage `955129c5105a8ab4`, all in the run tree's own databases (`optimizer.out`).

  **No credit is claimed.**
- **Main tree unchanged:** `prng_analysis.db` is 307200 bytes, mtime 1795145381, before and after, and `git status` is empty.
- **Processes:** no `window_optimizer.py` or `range_miner_worker.py` python process (`ps`/`awk` count 0). The worker was stopped by its recorded PID, 116863.
- **Card and shared memory:** the card shows `sunshine` only, and `/dev/shm` has 0 entries.
- **Run trees left for the lead to remove:** this run adds `~/onecard_q26_20261124` and `~/onecard_q26_staging_20261124` to the five listed in RERUN2.

## Question
- **Q2 (open since RERUN2).** Lead: the quiet figure is in. Does the macro view take 2²⁶ per window (≈ 1,085 a day on one quiet card) or 2²⁸ (≈ 290 a day, UNVERIFIED)?
