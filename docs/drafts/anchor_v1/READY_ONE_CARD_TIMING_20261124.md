# READY · One-card timing test, variant a (2026-11-24): REFUSED before dispatch, no timing taken

Lead GO: `logs/e2e_session/GO_ONE_CARD_TIMING_20261124.txt` (15:05). Owner one-off: 14:58. Prep: `PREP_ONE_CARD_TIMING_20261124.md`.
Evidence: `data/anchor_v1_logs/onecard_timing/{a24,a24r}/` in this worktree. It holds the driver `run_onecard.sh`, the driver log, `optimizer.out`, the worker log, load before and after, the card monitor and `/dev/shm` before and after.

## Plain summary
- **No timing was measured.** Seeds swept 0 of 2²⁴, stripes dispatched 0, survivors 0 in every phase.
- The 2²⁴ trial was **refused by Step 1 itself** 4.1 s after launch, before any study or dispatch:
  > `Step1SearchPreflightError: --deliver-skip-range needs a Step 1 search (--step1-search-id) opened with deliver_skip_range: true, so its trials get a study of their own`
  
  Source: `window_optimizer_integration_final.py:1767-1770`, `_step1_skip_delivery_preflight`.
- Per the GO, I stopped. I did not patch and did not rerun, and the 2²⁶ run was not attempted.
- **The cause is my error.** My single-card note said a direct run needs no search id. That is true with skip delivery off, but run H's settings switch it on, and the skip switch then requires a search of its own. The prep missed this.

**Figures the lead asked for:**

| Asked for | Result |
|---|---|
| Seconds per window per phase | **none** (no phase ran) |
| Survivors per phase | **none** |
| Peak card memory | 264 MiB: sunshine's resting level, the miner added nothing. 19 samples at 0.5 s |
| Windows per day at 2²⁸ | **not corrected**: the earlier figure (about 360-430, [UNVERIFIED]) stands unmeasured |

## What ran, in order
1. **First attempt (tag `a24`), 15:00:14-15:00:15. Launched nothing.**
   - Pre-flight passed, and the run tree was built.
   - Then my driver exited on `set -u` inside the venv's `activate` (line 70, `LD_LIBRARY_PATH` unbound).
   - No optimizer, worker or card use. This was a fault in my wrapper, not in the test. I fixed the wrapper (`set +u` around `source`) and started again under a new tag.
2. **Second attempt (tag `a24r`), 15:01:08-15:01:18.**
   - Pre-flight passed:
     - card: only `sunshine` (PID 3219);
     - no miner worker; port 5700 free;
     - `/dev/shm` empty;
     - main clean at `41ad509`; dataset digest `513648160d35…` OK;
     - 9 GiB available.
   - The run tree was clean, with inputs identical to main (`inputs.sha256`).
   - The optimizer resolved `[EXEC-SET] … PARTIAL nodes=['localhost'] gpus=1 remote=False admission=1` and dropped the three rigs. **So the one-card execution set works as the note said.**
   - Then the refusal above, rc 1.
   - The worker (PID 100502) started 15:01:12 and exited at once with `ConnectionRefusedError`, because the coordinator was already gone.
   - Staging is empty: no `miner_ledger.db` was created.
3. **Load:** taken under the learning agent's batch.
   - Before: 7.27 / 7.33 / 7.63; after: 7.30 / 7.34 / 7.62. Uptime 5:14 → 5:15.
   - Two `wm_step5.py` processes (PIDs 89642 and 75783) were pinned to cores 6,7, the cores this test used (PREP §4).

## State left
- **Processes:**
  - `pgrep -af 'range_miner_worke[r]'` returned nothing (rc 1).
  - No `window_optimizer.py` or `range_miner_worker.py` python process (`ps`/`awk`).
  - Card: `sunshine` only. `/dev/shm`: 0 entries.
- **Nothing published:** no generation, no search opened, no credit. The main tree is clean.
- **Left on disk:** two detached worktrees of `41ad509`, registered in main's `.git/worktrees`. Both are unused and clean, and both are the lead's to keep or remove (`git worktree remove`):
  - `~/onecard_a24_20261124`
  - `~/onecard_a24r_20261124`
  
  Also two empty staging directories, `~/onecard_a24{,r}_staging_20261124`.
- **Prep commands file:** the `$TT/data/` paths were moved to `$TT/logs/` before the run. `data/` is not gitignored and would have dirtied the tree that `finalize_run` checks. The file is otherwise unchanged and still lacks the search step.

## What would unblock it (for the lead and owner; not done)
- **(i) Open a Step 1 search in the run tree's own `prng_analysis.db`**, `kind: pinned`, `deliver_skip_range: true`, as run H did. Its record:
  > `STEP1_SEARCH_OPENED s1s-a15a5425… space=pinned`
  
  (`~/skipsw_run_thr3030/logs/stage4/search_open.log`). The command was WATCHER `--step1-search-open` (`RUNBOOK_RUN_I_20261123.md:366`).
  - Pass that id to `--step1-search-id`.
  - The trial would then be eligible for S7 credit in that isolated tree only. The GO says "no search credit claimed", so this needs the lead's word.
- **(ii) Drop `--deliver-skip-range`** (run A's arm).
  - This is no longer "run H's settings as they are".
  - In runs A and B (thresholds 0.71/0.47) the switch did not change phase rows (`RESULT_SKIP_SWITCH_RIG_RUNS_20261122.md` §2-§3). Whether that holds at 0.30 is not measured.

## Questions
- **Q1.** Lead: route (i) with a search confined to the run tree, or route (ii)?
- **Q2.** Lead: keep or remove the two unused run trees?
