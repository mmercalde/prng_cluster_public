# PREP · One-card sieve timing test (2026-11-24, prepare only, nothing launched)

Task: `logs/e2e_session/TASK_ONE_CARD_TIMING_PREP_20261124.txt`; owner go `OWNER_GO_macro_one_card_timing_20261124.txt` ("Yes on the macro", 14:51). Commands: `PREP_ONE_CARD_TIMING_20261124_commands.sh` beside this file (`bash -n` clean, never run).

**Plain summary.**
- One trial, one window, 2²⁶ seeds `[2³¹, 2³¹+2²⁶)`, on the 3080 Ti, run as a direct `window_optimizer.py --use-range-miner --execution-set-nodes localhost`.
- Build the run tree from **main `41ad509`**. **Brief II is not needed for timing.**
- Variant **a** (run H's settings) needs the owner's one-off past the cap. Variant **b** (run B's thresholds) should pass the gate with no one-off.
- **Cores 6 and 7 are not free now**: two learning-agent processes are pinned there (§4).

## Prior record
Read for this note: the `NOTE_SINGLE_CARD_SIEVE_20261124.md` Prior record (searches unchanged), plus:
- `docs/RESULT_SKIP_SWITCH_RIG_RUNS_20261122.md` §1, §3;
- run H's `~/skipsw_run_thr3030/logs/{gate12_20261122_143253.log:45,311-323, stage4/volume_gate.log}`;
- `scripts/survivor_volume_gate.py:1-52, 288-322`;
- `config_manifests/survivor_volume.cap`;
- `PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md` §7.3; `PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:58`;
- `RUNBOOK_RUN_I_20261123.md` b03-b04.

Labels:
- one-card route: ALREADY RECORDED (the earlier note);
- variant b's measured basis: ALREADY RECORDED (run B);
- the gate's minimum evaluable range (§2): NEW;
- the core conflict: NEW, point-in-time.

Not searched: ser8, the private repository.

## 1. Run tree, staging, pre-flight
- **Paths:** run tree `~/onecard_{a,b}_20261124` (a git worktree of `41ad509`) and staging `~/onecard_{a,b}_staging_20261124`, one pair per variant. Each has its own coverage ledger and miner ledger.
- **Kept off-limits:**
  - nothing goes under the main tree, `~/miner_staging` or `/tmp`;
  - worker output and the CuPy cache go to `$TT/logs/` (gitignored; `data/` is not, and would dirty the tree that `finalize_run` checks) (the launcher's local default writes the CuPy cache to `/tmp`, `launch_fleet_manual.sh:345`);
  - `--miner-output-dir` is explicit so nothing lands in `/dev/shm/prng/miner` (the worker's default, `range_miner_worker.py:2938-2940`);
  - Optuna storage follows the checkout (`window_optimizer_bayesian.py:14-23`).
- **Why `41ad509`:**
  - It is main HEAD, landed and clean. `git diff 00e2ee7 41ad509` is one doc file, so the code is identical to run H's tree.
  - Anchor 25 (run H's) is inside main's 0-100.
  - Brief II changes only which anchors can be placed and adds the screen objective (CPU, about 0.3 s). It touches no kernel, stripe or survivor path, so the timing carries over.
- **Pre-flight (block P1, read only):**
  - Card: only `sunshine`. At 14:56 it was PID 3219, 248 MiB, 0 %.
  - No `range_miner_worker.py` process; port 5700 free (0 at 14:56).
  - `/dev/shm` empty, 0 % used (after the 09:47 reboot).
  - Main is clean at HEAD; dataset sha256 is `513648160d35…`.
  - MemAvailable at least 6 GiB (9 GiB at 14:56). VM100 idle is the lead's check.
- **Launch:** optimizer first; then, once `[EXEC-SET]` appears, start the worker with `nohup`. It must connect within the 180 s admission window (`range_miner_coordinator.py:417`).
- **Stopping:** by recorded PIDs only. A memory guard stops both if MemAvailable falls below 2 GiB.
- **Peak memory: not known.** Run H's coordinator carried about 4× variant a's volume to completion on this box, but its peak memory was not recorded [UNVERIFIED].
- **Timing read-out:** from the miner ledger, opened with `?mode=ro&immutable=1`:
  - `trials.created_at` / `finalized_at`;
  - each stripe's `elapsed_s` and `survivors_total`.

## 2. Two variants (both 2²⁶ seeds, window 12, anchor 25, midday, skip 0-16, skip delivery on, both modes)

| | Thresholds fwd/rev | Gate prediction, phases 3 / 4 (survivors) | Gate verdict | Measures |
|---|---|---|---|---|
| **a** | 0.30 / 0.30 (run H) | about 592,000 / 149,000 (run H's sampled counts ÷ 4 for phase 4) | **REFUSE**: needs the one-off | the real per-window cost at the macro settings, survivor handling included |
| **b** | 0.71 / 0.47 (run B) | about 6 / 750 (run B's PROCEED at 2²⁸: 24 and 2,995, `~/skipsw_run_on/logs/stage4/volume_gate.log:6-10`) | **PROCEED** expected | kernel plus fixed per-trial cost with almost no survivors |

- **a − b** gives the survivor-handling cost.
- **What b changes about the sieve:**
  - It raises both thresholds, so phase 3 drops about 100,000-fold and the survivor set is a different, far smaller one.
  - Its screen scores are not comparable with a's. It does not give a usable macro configuration.
  - **b is a timing variant only, not a proposal to change the sieve.** Run H's thresholds stay the baseline.
- **Why not go below 2²⁶ for b:** the gate cannot evaluate a range under 2²⁶.
  - `min_sample_seeds` is 2²⁶ (cap file), and the estimator refuses a sample larger than the range (`survivor_volume_estimator.py:216-217`).
  - So a 2²⁴ run gets no PROCEED at any thresholds; at 2²⁴, only the owner's one-off would allow it.
  - On a 2²⁶ range the gate samples the whole range, so its count is the real per-phase count. That needs `--phase4-sample-seeds 2²⁶`, because the default 4 × 2²⁶ exceeds the range.
- **Card time:** gate about 30 s per variant, plus the trial. Estimates [UNVERIFIED]: a about 65 s, b about 40 s. Together a + b come to about 3 minutes.

## 3. What the cap protects
- **F3: "A run cannot overwhelm the rigs"** (`PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:58`).
  - The cap is the largest volume a complete certified run has carried. Run 3 froze after 13,146,485 survivors.
  - It is raised only by the owner (`PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md` §7.3; cap file evidence: 50,000, owner, 2026-11-19).
- **For this test:** no rig is involved. What is exposed is VM101 itself: coordinator memory (15.5 GiB, no swap) and staging disk (478 GiB free).
- **Suggested one-sentence question:** "May the one-card timing test run at run H's settings past the 50,000 cap? It predicts about 592,000 phase-3 survivors, a quarter of what run H carried to completion; no rig is used, and the run is stopped by a memory guard."

## 4. Cores
- **Need:** two cores, one for the optimizer/coordinator (with its 4 staging threads) and one for the worker. The commands use 6 and 7.
- **Now (14:56):** all 8 cores are busy. The learning agent's `wm_step5.py` processes are pinned to 0,1 / 2,3 / 4,5, and **two more, PIDs 89642 and 75783, are pinned to 6,7** (`taskset -cp`). That is five busy processes, already at the lead's limit of about 5.
- **Ask:** clear 6 and 7 (those two PIDs). The test then makes 3 + 2 = 5 busy cores, within the rule.
- If 6 and 7 stay shared, variant a's time is an upper bound.

## Questions
- **Q1.** Owner: the one-off for variant a (the sentence in §3)?
- **Q2.** Lead: run b first, so a gate PROCEED and a timing exist before the one-off is asked?
- **Q3.** Lead: will PIDs 89642 and 75783 be moved off 6 and 7?
