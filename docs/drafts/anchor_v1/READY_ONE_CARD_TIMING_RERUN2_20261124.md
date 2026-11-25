# READY · One-card timing, variant a (run H's settings), 2²⁴ and 2²⁶ (2026-11-24): both trials committed

- Lead answer: 15:32, at the end of `logs/e2e_session/LEAD_ANSWERS_ONE_CARD_TIMING_20261124.txt`.
- Owner: go 14:51, cap one-off 14:58, for this test only.
- Evidence: `data/anchor_v1_logs/onecard_timing/{s24,s26}/` in this worktree. Each holds the driver log, `optimizer.out`, `worker_gpu0.log`, `ledger.txt`, `card_mem.csv`, the load files, `search_open.log` and the main-database checks.
- Driver: `onecard_timing/run_onecard.sh`. The one change since the last rerun is that the worker now starts only once `:5700` is listening (driver lines after `# [RERUN2 driver change, lead 15:32]`). No project code changed.

## Plain summary
**Times are upper bounds.** Cores 6 and 7 were shared all along with two learning processes, PIDs 89642 and 75783, at about 96-98 % each. The card averaged 4-5 % use during phase 3, so phase 3 was waiting on the CPU, not the card.

- **Survivor handling dominates.** It costs about 160 µs per survivor, five times my earlier 30 µs estimate, so my earlier per-window figures were too low.
- **Corrected windows per day at 2²⁸:** about **155** (trial time about 560 s), on these shared cores, extrapolated from the two measured points [UNVERIFIED at 2²⁸].
- **Measured windows per day:** about 1,660 at 2²⁴ and about 560 at 2²⁶.

## Per phase (miner ledger `stripes`, one stripe per phase, one worker)

| Phase | 2²⁴ s | 2²⁴ survivors | 2²⁶ s | 2²⁶ survivors | 2²⁸ s, linear in seeds [UNVERIFIED] |
|---|---|---|---|---|---|
| 1 `java_lcg` | 4.68 | 0 | 6.09 | 0 | ≈ 12 |
| 2 `java_lcg_reverse` | 1.08 | 0 | 2.59 | 0 | ≈ 9 |
| 3 `java_lcg_hybrid` | 26.68 | 148,004 | 105.30 | 592,791 | ≈ 420 |
| 4 `java_lcg_hybrid_reverse` | 7.66 | 37,416 | 27.29 | 148,709 | ≈ 106 |
| **Stripes total** | 40.1 | — | 141.3 | — | ≈ 546 |
| **Trial** (`trials.created_at` → `finalized_at`) | **52.2** | final 789 rows | **153.2** | final 3,044 rows | **≈ 560** |
| Wall, launch → optimizer exit (includes startup and publication) | 64.1 | | 184.8 | | ≈ 670 [UNVERIFIED] |
| **Windows per day** (86,400 ÷ trial time) | **≈ 1,660** | | **≈ 560** | | **≈ 155** |

**Notes on the table:**
- **Fixed cost per trial** (trial time minus the stripes) was 12.0 s at both sizes.
- **Phase 3 matched the volume gate's prediction:** 592,791 survivors against the gate's 592,050 from run H.
- **Survivor cost**, taking the kernel times from run H's volume gate and dividing the rest by the survivor count:
  - phase 3: (105.3 − 9.9) s ÷ 592,791 ≈ 161 µs per survivor;
  - phase 4: (27.3 − 3.7) s ÷ 148,709 ≈ 159 µs per survivor.
- **Against my earlier estimates** (`NOTE_SINGLE_CARD_SIEVE_20261124.md` §2):
  - 2²⁴: estimated 30 s, measured 52 s;
  - 2²⁶: estimated 65 s, measured 153 s;
  - 2²⁸: estimated 200-240 s, now about 560 s.
- **Settling the 2²⁸ figure** needs one 2²⁸ trial, or a rerun on quiet cores, which might lower it.

## Card, memory, load
- **Peak card memory:** 2,495 MiB, including `sunshine`'s 264 MiB, so about 2.2 GiB for the miner. It was the same at both sizes.
- **Card use:** peaked at 71 %; the average during phase 3 was 4.9 % (2²⁴) and 3.9 % (2²⁶), from 0.5 s samples.
- **Process memory:** peak RSS for optimizer plus worker was 838 MiB (2²⁴) and 967 MiB (2²⁶). The memory guard never fired, and MemAvailable stayed at 9 GiB or more.
- **Load:**

  | Run | Before | After |
  |---|---|---|
  | 2²⁴ | 10.40 / 9.46 / 8.86 (15:31:39) | 10.36 / 9.64 / 8.97 (15:32:54) |
  | 2²⁶ | 10.24 / 9.67 / 9.00 (15:33:20) | 10.25 / 10.02 / 9.27 (15:36:35) |

  The learning batch ran on all 8 cores throughout, with two of its processes on 6 and 7.

## Scope kept and state left
- **Searches, credit and publication stayed inside each run tree's own databases.** As route (i) expected, the code credited S7 automatically, published a run-tree generation and certified run-tree coverage:
  - 2²⁴: search `s1s-fa00b55a…`, 789 rows, coverage `ef149af46e60c63d`;
  - 2²⁶: search `s1s-fa1f3378…`, 3,044 rows, coverage `a3a57df50fa64c36`.
  
  **No credit is claimed from either.**
- **The main tree is unchanged:** `prng_analysis.db` is still 307200 bytes, mtime 1795145381, before and after each run, and `git status` is empty.
- **Processes:** no `window_optimizer.py` or `range_miner_worker.py` python process (`ps`/`awk` count 0). Each worker was stopped by its recorded PID.
- **Card and memory:** the card shows `sunshine` only. `/dev/shm` has 0 entries. Nothing was written to `~/miner_staging` or `/tmp`.
- **Run trees left in place for the lead to remove.** All are detached worktrees of `41ad509`:

  | Tree | State | Staging dir |
  |---|---|---|
  | `~/onecard_a24_20261124` | unused | `~/onecard_a24_staging_20261124` |
  | `~/onecard_a24r_20261124` | skip-switch refusal | `~/onecard_a24r_staging_20261124` |
  | `~/onecard_r24_20261124` | admission timeout | `~/onecard_r24_staging_20261124` |
  | `~/onecard_s24_20261124` | committed | `~/onecard_s24_staging_20261124` |
  | `~/onecard_s26_20261124` | committed | `~/onecard_s26_staging_20261124` |

## Questions
- **Q1.** Lead: repeat 2²⁶ on quiet cores 6 and 7, since phase 3 waits on the CPU, to see how much of the 160 µs per survivor is contention?
- **Q2.** Lead: at about 155 windows a day at 2²⁸ (about 560 a day at 2²⁶), does the macro view take 2²⁶ per window, or wait for a quiet-core figure?
