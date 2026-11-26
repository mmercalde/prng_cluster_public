# DRAFT: Step 2 diagnostic harness, run-scoped and non-certifying (2026-11-22)

**Summary:**
1. **What it is.** `scripts/diagnostics/step2_diagnostic_harness.py` runs Step 2's existing trial code in-process: `load_data`, `run_trial` with the v4.3 objective, and the search space from `generate_scorer_jobs.py`. It reads one NPZ by path and writes only into a new `--out-dir`.
2. **What it avoids.** It never calls the Step 2 launcher or the legacy converter, and never reads the summary JSON. It writes nothing to the main tree, `scorer_trial_results`, `/dev/shm/prng` or any rig. GPU imports, network and subprocesses are blocked.
3. **Processes.** Six separate worker processes by default, one fresh process per trial (owner 16:00). For a given seed the results are identical whatever the worker count: the self-test checks 6 workers against 1.
4. **Self-test: 8/8 PASS** (3 trials on a 500-row slice, CPU, about 1 s per harness call). It found and fixed one write: importing the Step 2 modules wrote `.pyc` files into the code tree.
5. **The 100-trial pass on run H's 12,156 rows was NOT run.** The command for the lead is in §5.
6. **Measured in the self-test, and predicted by reading for run H:** on this population the objective has one live term. Every row is variable-skip and every row has `trial_number` 1, so the enrichment term is exactly 0 and coverage is exactly 1. The objective reduces to 0.20 − 0.10·|ln(keep/0.10)|: it ranks parameter sets only by how close their residue mask keeps 10% of the sample.
7. **Yardstick:** Step 2 output is a parameter set, not a hit-rate result. Only pools scored on unseen draws against `k/1000` settle value.

**Open questions for the owner:** see §6.

- **Author and scope:** Claude (Team Alpha), worktree `/home/michael/tfm_worktrees/pwc-offset`, uncommitted.
- **Files:** new files only; no existing file changed.
- **Authority:** owner 15:57 ("yes, work on the Step 2 test") and 16:00 (multi-core). Route Q1(a) of `docs/drafts/NOTE_STEP2_READINESS_ON_RUNH_20261122.md`.

## Prior record

- **Searches:** this draft adds no new search. It rests on the readiness note's Prior record (P1–P7, same day), which covers Step 2's governed items: Chapter 3 audit F1–F16 and N1–N3, Chapter 12 B1, and `TODO_CONSOLIDATED.md:91` and `:157`.
- **Precedent for a non-certifying harness that drives production functions directly:** `tests/phase6/wall_ab_gate.py:37-49` ("HARNESS: the optimizer loop … the trial itself is the production call") and `tests/phase6/sampler_control_arm.py:1-60`.

**Labels:**

| Item | Label |
|---|---|
| The harness | **NEW** (a diagnostic, not a pipeline change) |
| F5 / N1 (`UnboundLocalError` when rows ≤ `sample_size`) | **ALREADY RECORDED**; reproduced by self-test C |
| An objective with a dead term on a single-mode population | **ALREADY RECORDED** in lineage (`TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md`; audit F6, F8). The **zero enrichment on run H** is **NEW** as a measured fact |
| Anything that changes a ruling | **none** |

**Not searched:** as in the readiness note.

## 1. Files (new, uncommitted)

| File | sha256 |
|---|---|
| `scripts/diagnostics/step2_diagnostic_harness.py` | `ca642042fed56ad5…` |
| `scripts/diagnostics/step2_diagnostic_harness_selftest.py` | `15dfffe21e636739…` |
| `docs/drafts/DRAFT_STEP2_DIAGNOSTIC_HARNESS_20261122.md` | this file |

**The imported code is byte-identical to main `41ad509` (`cmp`):** `scorer_trial_worker.py`, `generate_scorer_jobs.py`, `utils/survivor_loader.py`, `utils/__init__.py`. The worktree HEAD (`ab2d330`) is recorded with the files' sha256 in every `run_record.json`.

**Self-test outputs** (kept; `data/` is not `/tmp`): `data/step2_diag_selftest_kpp2swue/` (final) and `pz29zoxt/`. There are also earlier runs, `data/step2_diag_selftest_{2f972s0z,emrcwbgi,71hxhyi0}/`. Those three show the `.pyc` finding and the stdout-file false alarm described in §4.

## 2. What it does

1. **Refuses before writing anything** if:
   - `--npz` is not an existing `.npz`, or names the summary JSON;
   - `--out-dir` exists, or lies under `~/distributed_prng_analysis` or `/dev/shm/prng`, or inside any `scorer_trial_results`;
   - its parent is missing.
2. **Reads git HEAD** and `git status` for the imported files: the only subprocess, read-only.
3. **Installs guards:**
   - `sys.dont_write_bytecode`;
   - an import blocker for `torch`, `cupy`, `cupyx`, `pycuda`, `jax`, `tensorflow`;
   - `CUDA_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` set to `""`;
   - socket connect raises, and attempts are counted;
   - subprocess and `os.system` raise, and attempts are counted.
4. **Creates `--out-dir`** and routes logging to `worker.log` there. The worker module's stderr handler is lowered to WARNING.
5. **Imports `scorer_trial_worker`** with a clean `sys.argv`, since the worker parses `--gpu-id` at import (`scorer_trial_worker.py:73-98`).
   - It unregisters the worker's atexit GPU cleanup (`:125-149`).
   - It binds the worker's `load_survivors` to `allow_fallback=False`.
6. **Records the input:** NPZ path, resolved path, sha256, row count, keys and `skip_mode` counts. Then it calls `stw.load_data(npz)` once in the parent.
7. **Creates an Optuna study** in `--out-dir`, with the same direction and pruner as `generate_scorer_jobs.py:94-107` and a seeded TPE sampler. It pre-samples all trials with `define_search_space(trial, sample_size)` plus `optuna_trial_number` (`:116-120`).
8. **Runs each trial** in its own forked process (`Pool(workers, maxtasksperchild=1)`), calling `stw.run_trial(...)`. The status is `scored`, `scored_-1.0` (the worker's reject value) or `raised` (with traceback).
9. **Tells the study** each objective, and `-inf` for raised trials, as the real path does.
10. **Writes into `--out-dir`:**
    - `trials.jsonl`: params, status, objective, kept count, seconds, process id and error per trial;
    - `trial_masks_packbits.npz`: each trial's 0/1 selection mask, lossless, since the worker's `scores` array is the mask (`:460-466`);
    - `NON_CERTIFYING_DIAGNOSTIC_optimal_scorer_config.json`: the best params inside a labelled wrapper;
    - `optuna_step2_diagnostic.db`;
    - `run_record.json`: wall and evaluation time, peak RSS (self and children), input identity, code identity, worker count, CPU count and affinity, distinct process ids, guard counters, the GPU modules found in `sys.modules`, and counts.
11. **Prints** the scored, scored −1.0 and raised counts, the best trial, the input sha256 and row count, the worker and CPU counts, wall time, RSS and guard state. Exit code 3 if any GPU module was loaded.

## 3. Every difference from the real Step 2 path

The real path is `run_scorer_meta_optimizer.sh` → `generate_scorer_jobs.py` → `scripts_coordinator.py` → `scorer_trial_worker.py` (CLI) → `coordinator.collect_scorer_results`.

| # | Real path | Harness | Effect on trial results |
|---|---|---|---|
| D1 | Ramdisk preload of the NPZ and histories to `/dev/shm/prng/step2` on every node (`run_scorer_meta_optimizer.sh:14-24`) | none; NPZ read by path | none (same bytes) |
| D2 | NPZ auto-conversion block (`:140-150`). It calls the converter on certified trees (F1) | absent | real path exits 1 today; harness proceeds |
| D3 | Code push to three rigs (`:257-261`) | absent | none |
| D4 | Jobs read `/dev/shm/prng/step2/<name>` (`generate_scorer_jobs.py:31-39`, `:126-128`) | `--npz` path; default is the bundle copy, sha256 `3f6a884d…`, equal to the generation artifact | none |
| D5 | Loader with fallback on (`scorer_trial_worker.py:166`; `utils/survivor_loader.py:106`, `:160-168`) | fallback off | none for an existing `.npz`; harness can never load JSON |
| D6 | `load_data` reads `optimal_window_config.json` beside the NPZ (`:273-288`). On the ramdisk it is absent (only 3 files preloaded, `:18-21`) | beside the bundle NPZ it is also absent (`run_record.input.optimal_window_config_beside_npz`) | none: both default to `java_lcg`, 1000; `run_trial` does not use them (`:293-294`, signature only) |
| D7 | History args passed and ignored (`:524-526`) | not passed | none |
| D8 | TPE sampler unseeded (`generate_scorer_jobs.py:101-107`) | `TPESampler(seed=--seed)` | different parameter draws; reproducible |
| D9 | Study DB `./optuna_studies/scorer_meta_opt_<epoch>.db` in the launch directory (`:221-224`) | `--out-dir/optuna_step2_diagnostic.db` | none |
| D10 | One `python3 scorer_trial_worker.py …` process per job via `subprocess.run(..., cwd=node.script_path)` (`scripts_coordinator.py:893`). Each loads the NPZ itself. Batches ≤ 6 per node with a 5 s cooldown (`:105-106`); localhost runs 2 at a time | one forked process per trial, inheriting the NPZ loaded once in the parent; `--workers` (default 6) at a time | none on values: sampling is seeded per trial from `optuna_trial_number` (`:354-355`); the objective is deterministic |
| D11 | Worker GPU binding from `--gpu-id` (`:73-98`) and atexit cleanup that imports `cupy` and `torch` (`:125-149`) | blocked or unregistered | none: the objective is numpy only |
| D12 | Result written to `~/distributed_prng_analysis/scorer_trial_results/trial_NNNN.json` (`:496-517`), pulled and deleted (`coordinator.py:2599-2614`, `:2635-2664`) | returned in-process; never written there | real path can lose results (F10, `collect_scorer_results(1)`); harness cannot |
| D13 | `study.tell(trial_id, accuracy)` for results whose accuracy is not None; errors carry `-inf` (`run_scorer_meta_optimizer.sh:290-297`; worker `:633`) | `study.tell(trial, objective or -inf)` for every trial | same values (Optuna 4.4.0 records `-inf` as COMPLETE, checked) |
| D14 | `optimal_scorer_config.json` = bare `best.params` in the launch directory (`:306-307`) | `NON_CERTIFYING_DIAGNOSTIC_optimal_scorer_config.json`, wrapped and labelled, only in `--out-dir` | deliberate: no consumer can mistake it |
| D15 | Logs to the worker's stderr, captured by `scripts_coordinator` | `--out-dir/worker.log` | none |
| D16 | `--legacy-scoring` branch (broken, F11) | not offered | n/a |
| D17 | WATCHER evaluation by file existence (B1) | none: the harness reports counts and exit codes | n/a |

## 4. Self-test (run, CPU only)

**Command:** `python3 -B -u scripts/diagnostics/step2_diagnostic_harness_selftest.py`. Final run, on the final harness file: `data/step2_diag_selftest_kpp2swue/`, rc 0. The run A values below are from `data/step2_diag_selftest_pz29zoxt/`, identical code apart from one comment.

**Setup:**
- The slice is the first 500 rows of every array of the run H NPZ (`slice500.npz`, sha256 `0b13746a…`).
- The harness runs as separate subprocesses with `PYTHONDONTWRITEBYTECODE` removed from their environment, so the harness's own bytecode guard is what is tested.

| Check | Result |
|---|---|
| A: 3 trials, 6 workers, sample 200, seed 7. rc 0; 3 records; NON-CERTIFYING config and no `optimal_scorer_config.json`; no GPU module loaded; 0 socket and 0 subprocess attempts; 3 distinct process ids; workers and CPU count recorded (6 / 8) | PASS |
| A2: same with 1 worker; identical params, objectives, statuses and masks | PASS |
| B: reusing A's `--out-dir` is refused; A's files unchanged | PASS |
| C: sample 5000 on 500 rows; the governed F5/N1 `UnboundLocalError` is counted as `raised 1` | PASS |
| D: `--out-dir` under `~/distributed_prng_analysis` is refused and not created | PASS |
| E: `--npz` = run H's summary JSON is refused | PASS |
| W1: listed before and after, unchanged: `~/distributed_prng_analysis/{scorer_trial_results,optuna_studies}`, `/dev/shm/prng`, the worktree's `scorer_trial_results` and `optuna_studies`, and the `optimal_scorer_config.json` / `scorer_jobs.json` files in the main tree, the run H tree and the worktree | PASS |
| W2: no file newer than the start marker under `~/distributed_prng_analysis`, `~/skipsw_run_thr3030` or the worktree, outside the temp directory | PASS |

**What the earlier runs showed:**
- **First run, W2 failed on two counts:**
  - The real finding: importing the Step 2 modules wrote `__pycache__/scorer_trial_worker.cpython-310.pyc` and `generate_scorer_jobs.cpython-310.pyc` into the code tree. Fixed with `sys.dont_write_bytecode` before any import. I deleted the two `.pyc` files my run had created and re-ran; none was written.
  - The other count was my own `tee` log inside the worktree.
- **A later run failed W2 only on my stdout redirect file.** The final run printed to the terminal only.

**Self-test values (run A):**

| Trial | kept / 200 | enrich | coverage | size penalty | objective |
|---|---|---|---|---|---|
| 0 | 11 | 0.0 | 1.0 | 0.598 | 0.140 |
| 1 | 31 | 0.0 | 1.0 | 0.438 | 0.156 |
| 2 | 18 | 0.0 | 1.0 | 0.105 | 0.189 |

- **Wall time:** 1.0 s per harness call.
- **Peak RSS:** 76.6 MB in the parent, 62.6 MB per child.

## 5. The full pass: NOT run. Command for the lead to review

```bash
cd /home/michael/tfm_worktrees/pwc-offset && source ~/venvs/torch/bin/activate && \
mkdir -p logs/step2_diag_20261122 && \
nohup python3 -u scripts/diagnostics/step2_diagnostic_harness.py \
  --npz /home/michael/distributed_prng_analysis/logs/skipsw_session/bundle_thr3030/survivors.npz \
  --out-dir /home/michael/tfm_worktrees/pwc-offset/data/step2_diag_runH_20261122 \
  --trials 100 --workers 6 --sample-size 5000 --seed 0 \
  > logs/step2_diag_20261122/runH_100.log 2>&1 &
echo $! > logs/step2_diag_20261122/runH_100.pid
```

- **Expected peak RSS** [UNVERIFIED, scaled from the self-test]: under 100 MB in the parent plus about 70 MB per child, 6 at a time, so under 1 GB.
- **Expected wall time** [UNVERIFIED]: about a minute or less. Each trial scores a 5,000-row sample with numpy, plus a fork.
- **VM101 has 15.5 GiB and no swap;** this is far inside it.
- **What to read afterwards:** `run_record.json` `counts` (scored, scored −1.0, raised) and `trials.jsonl`.
- **Predicted by reading, not measured on the 12,156 rows:** every scored trial has enrich 0 and coverage 1, and the best trial is the one whose mask keeps closest to 10%.

## 6. What it does not prove, and open questions

**What it does not prove:**
- **Not a hit-rate result.** Step 2 output is a parameter set chosen by the v4.3 objective from NPZ columns. No draw is read, seen or unseen. Only pools scored on unseen draws against `k/1000` settle value (`docs/PITFALL_STATE_RECOVERY_YARDSTICK.md` §1).
- **Not a Step 2 run.** It exercises none of: the launcher, ramdisk, conversion, dispatch, the fleet, collection, WATCHER, or the main-tree output paths (§3). It does not show that the real path would give the same best parameters: the sampler is seeded here (D8).
- **Not certifying.** No artifact may feed Step 3. The config file is wrapped and labelled for that reason.
- **Says nothing about the governed defects** except that F5/N1 reproduces.

**Open questions:**
- **Q1, the objective.** On run H's population, enrich ≡ 0 and coverage ≡ 1 (all rows `skip_mode` 1, all `trial_number` 1). Step 2 therefore selects only on keep fraction. Is a Step 2 pass on this population still wanted as a pipeline exercise, knowing the choice carries no other signal? The objective is ruled (S107 v4.3).
- **Q2, the seed.** The real path is unseeded. Is `--seed 0` acceptable, or should the lead run two or three seeds to show the spread?
- **Q3, landing.** An untracked or new `.py` under `scripts/` trips the Phase-4 coexistence gate (Gate 22, which fails on stray untracked `.py`), until landed. Should these files land as a tracked diagnostic, or stay as drafts?
- **Q4, review.** The harness is not in the before-commit Beta review classes (coordinator, worker and kernel code, the verifier and launch path for runs that count). Send it to Beta anyway?
- **Q5, the next step.** If Step 3 is to consume a diagnostic Step 2 result, it needs its own isolated route. `run_step3_full_scoring.sh` has the same ramdisk and fleet surfaces [not traced here].
