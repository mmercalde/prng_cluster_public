# READY · One-card timing rerun, variant a at 2²⁴ (2026-11-24): trial ABORTED at worker admission, no timing taken

- Lead answers: `logs/e2e_session/LEAD_ANSWERS_ONE_CARD_TIMING_20261124.txt` (15:18, route (i)).
- Owner: go 14:51, cap one-off 14:58.
- Evidence: `data/anchor_v1_logs/onecard_timing/r24/` in this worktree. It holds the driver log, `optimizer.out`, `worker_gpu0.log`, the ledger read-out, `search_open.log`, `params.json`, load before and after, the card monitor and the main-database checks. The driver is `onecard_timing/run_onecard.sh`.

## Plain summary
- **No timing was measured.** The trial was aborted by the coordinator's admission gate: 0 of 2²⁴ seeds swept, 0 stripes dispatched, 0 survivors in every phase.
  > `worker_admission_timeout: stage 0 (family 'java_lcg', phase 1) expected 1 eligible worker(s); 0 admitted after 180.1s`
  
  (`optimizer.out:84`; ledger `trials` row `distributed_config_t1_c72a02fd`, state `aborted`.)
- **The cause is my driver, not the code.** It started the worker as soon as the optimizer printed `[EXEC-SET]` (15:17:52), before the coordinator was listening on port 5700 (admission opened 15:17:54, `optimizer.out:65`).
  - The worker makes one connection attempt, with no retry (`range_miner_worker.py:2280`). It got `ConnectionRefusedError` and exited.
  - The coordinator then waited the full 180 s with `live_count 0` (`optimizer.out:66-83`).
- **Follow-on error:** the optimizer then raised `MinerIngressError`. The aborted trial had no validated threshold-provenance record, so D6 refused to ingest it. This is the expected refusal on an aborted trial, not a second fault.
- Per the lead's rule I stopped. I did not patch or rerun, and 2²⁶ was not attempted.

**Figures asked for:**

| Asked for | Result |
|---|---|
| Seconds per window per phase | **none** (no phase ran). Wall time was 185.4 s, almost all of it the 180 s admission wait |
| Survivors per phase | **none** |
| Peak card memory | 264 MiB, sunshine's resting level; the worker never reached the card |
| Peak RSS (optimizer + worker) | 205 MiB |
| Windows per day at 2²⁸ | **not corrected**; the earlier estimate of about 360-430 [UNVERIFIED] stands |
| Load (cores 6,7 shared with the learning batch) | before 9.53 / 8.94 / 8.25 (15:17:45); after 8.91 / 8.92 / 8.37 (15:21:00) |

## What worked (new, measured)
- **Search opened in the run tree only:** `s1s-fa00b55a6832…`, `kind pinned`, `constant_skip_delivery trial_range`, effective window 12 / anchor 25 / midday / 0.30 / 0.30 / skip 0-16.
  - The main tree's `prng_analysis.db` was unchanged (size 307200, mtime 1795145381, before and after).
  - The run tree stayed clean, so the skip-switch preflight that refused the first run passed.
- **Execution set** was `PARTIAL nodes=['localhost'] gpus=1 remote=False admission=1`. Admission used `expected_workers=1 (source=execution_set)`.
- **Nothing published or credited.** `optimizer.out` has no `PUBLISHED`, `credited` or `certified` line, and the trial is `aborted` in the run tree's miner ledger.

## State left
- **Processes:** no `window_optimizer.py` or `range_miner_worker.py` python process (`ps`/`awk` count 0). `pgrep -af 'range_miner_worke[r]'` matched only the shell running it.
- **Card and memory:** card shows `sunshine` only (PID 3219); `/dev/shm` has 0 entries. Main tree: `git status` empty.
- **Run trees and staging dirs left in place for the lead to remove** (per Q2). All are detached worktrees of `41ad509`:

  | Tree | State | Staging dir | Contents |
  |---|---|---|---|
  | `~/onecard_a24_20261124` | unused | `~/onecard_a24_staging_20261124` | empty |
  | `~/onecard_a24r_20261124` | refused, skip preflight | `~/onecard_a24r_staging_20261124` | empty |
  | `~/onecard_r24_20261124` | this rerun; has the search record in its `prng_analysis.db` | `~/onecard_r24_staging_20261124` | `miner_ledger.db`, `threshold_provenance.json` |

## What would fix it (driver only, no code; not done)
- Start the worker only after the coordinator is listening: wait for `ss -ltn` to show `:5700` (or for the `[ADMISSION]` line) before launching it.
- That is still inside the 180 s window, with a fresh tag and tree.

## Question
- **Q1.** Lead: may I rerun with that one driver change (worker starts after `:5700` is listening), new tag, 2²⁴ then 2²⁶?
