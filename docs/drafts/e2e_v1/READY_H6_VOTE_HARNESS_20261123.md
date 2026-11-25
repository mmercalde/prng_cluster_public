# READY: H6, the Step 6 vote harness (2026-11-23, rev 5: follow-up vote pools built, NOT COUNTED, no outcome read)

- **Author:** Claude (Team Alpha), worktree `/home/michael/tfm_worktrees/e2e-vote`, branch `e2e-vote` at `41ad509`.
- **State:** uncommitted, not landed, not sent to Beta.
- **Task:** `logs/e2e_session/TASK_VOTE.txt` (H6) under `TASK_COMMON.txt`, main tree.
- **Route:** `MAP_E2E_THIN_PATH_20261122.md` §2.4 (4), in worktree e2e-map.
- **Rev 5:** carries out `logs/e2e_session/TASK_FOLLOWUP_VOTE.txt` (sha `561e05e0…`) under the owner's go of 14:40 (`logs/e2e_session/OWNER_GO_followups_20261123.txt`, `005d9a5c…`, main tree). Seven **follow-up, descriptive, NOT COUNTED** pool files were built. **No outcome was read.** Results are in §3c. The harness and self-test are unchanged from rev 4.
- **Rev 4:** carries out `logs/e2e_session/TASK_VOTE_RUNH.txt` under the owner's go of 13:26 (`logs/e2e_session/OWNER_GO_vote_and_selfplay_20261123.txt`, main tree). The run H pools for arms A1 and A0 are built over the whole grade block. **No outcome was read.** Results are in §3b.
- **Rev 3:** adds two things.
  - The lead's NN rule: a `neural_net` sidecar fails closed unless the generator loaded the input scaler and the wrapper loaded `y_mean`/`y_std`. Harness-level only; the module is not edited.
  - The GPU-path checks, run under the owner's go of 08:29 (`logs/e2e_session/OWNER_GO_gpu_fixture_checks_20261123.txt`, main tree): synthetic fixtures only, on the VM101 RTX 3080 Ti.
  Results are in §3a.
- **Rev 2:** carries out the lead's decision on rev 1's Q2: "one format, defined once". H6 now writes the grader's `tfm.e2e.pool.v1` schema exactly as `~/tfm_worktrees/e2e-grader/scripts/diagnostics/e2e_grader.py` reads it. A new gate, G12, feeds H6 output through that grader's loader. The grader was not edited.

**Null results first (rev 5, NOT COUNTED).**
- **0 follow-up pools graded. No pool was compared with any draw. No hit count of any kind exists for the follow-up files.** The grader was not run; only its loader functions read the files (§3c.4).
- **Abstentions: 0 of 1,807 in each of the 7 files. Empty pools: 0.** Failures: none. The memory guard never fired (lowest MemAvailable 9,453 MiB).
- **At pool size 300, no draw in any of the three files has 300 distinct values** (max 275 or 276). At pool size 100, only 9 or 10 of 1,807 draws have 100 distinct values. A fixed Hit@100 or Hit@300 on distinct values would be unavailable on nearly every draw; the distinct k per draw is recorded (§3c.3).
- **A1 was not run on the label block [14454, 16261)**, as the task said: its target was built from those draws.

**Null results first (rev 4).**
- **0 pools graded. No pool was compared with any draw. No hit count of any kind exists for run H.** The grader was not run. Only its loader functions read the pool files (§3b.5).
- **Abstentions: 0 of 1,807 per arm. Empty pools: 0.** Failures: none. The memory guard never fired.
- **Under the grader's default `out_of_block_records=error`, the loader REJECTS both pool files**, because the block holds evening records and the grade block is midday only. With `exclude_and_count` it ACCEPTS both: 904 eligible midday records per arm, 903 evening records excluded and counted (§3b.5).
- **Fixed Hit@20 is unavailable on 315 of 1,807 A1 draws and 309 of 1,807 A0 draws**, where the 20 raw votes give 18 or 19 distinct values (§3b.3).

**Null results first (rev 3 and earlier).**
- **0 pools graded on any real draw. No hit-rate evidence of any kind.**
- The G12 grader run graded H6 fixture pools against the **synthetic** fixture dataset: A0fx 0 hits of 30, A1fx 1 hit of 30, both "NULL: not above chance after correction". These numbers check the format only. They are not results.
- Up to rev 3, run H was not touched. Nothing ran on run H data, except one read-only numpy read of the run H NPZ's `forward_matches`/`reverse_matches` (§5 F7, rev 1). Rev 4 is the run H vote, under the 13:26 go.
- The GPU was used **only** under the 08:29 go, for 37 s, on synthetic fixtures. No commit, no push, no land. The grader worktree was read only (§3, G12).

**Plain summary.**
1. `scripts/diagnostics/e2e_vote_harness.py` produces, for every target index t in a declared block, the pool Step 6 itself emits for that draw. It calls `PredictionGenerator.generate_predictions(survivors, history[:t], None, k)` and copies none of the ranking or vote code.
2. **Format:** H6 writes `pools.jsonl` with the grader's fields only:
   - `schema`, `arm_id`, `config_id`, `dataset_sha256`, `data_cutoff_index`;
   - `target{combined_index, date, session}`;
   - `pool`, `pool_sha256`, `abstain`, and `abstain_reason` when abstaining.
   All other per-draw detail goes to `vote_diagnostics.jsonl`, which the grader never reads.
3. **G12 passes.** The grader's own reader functions accepted three H6 fixture arms:
   - learned A1fx: 30 of 30 graded, 1 pending;
   - unlearned A0fx: 30 of 30 graded, 1 pending;
   - gate-blocked Ablk: 30 of 30 abstained, 1 pending.
   The grader's own `pool_sha256` and `k` agreed on every record. The grader's full CLI also ran on the output: exit 0, `controls_ok=True`.
4. **TODO-E2E-8 still passes.** At t = len(history) the H6 pool equals the Step 6 CLI pool byte for byte (4 of 4 cases). Inside the history it equals the CLI on `history[:t]` at 3 of 3 sampled t.
5. **CPU self-test (run 6): 13 of 13 gates pass, 14 of 14 mutants killed.**
   - The grader-field mutants M11–M14 all corrupt a grader field (cutoff, target identity, pool hash, dataset hash), and G12 kills each one.
   - The run took 77.8 s; peak RSS was 692 MB.
   - Multi-core through `--workers`; output byte-identical at 1 and 4.
6. **GPU path tested (rev 3): 3 of 3 GPU gates pass, 2 of 2 NN mutants killed.**
   - With the card visible, H6 = CLI byte for byte for LightGBM, XGBoost and a Step 5-trained `neural_net` on `cuda`. NN walk-forward = CLI, and 2 forked workers = 1.
   - The NN check refuses a sidecar without the scaler and one without `y_mean`/`y_std`. On those same sidecars the CLI exits 0 with a pool.
   - Wall 36.1 s, peak card memory 922 MiB (658 MiB above the 264 MiB baseline), peak RSS 1,282 MB.
7. **On CPU the NN path fails closed.** Both the harness and the CLI stop with `RuntimeError: No CUDA GPUs are available`, because `pg:352` passes `device="cuda"` whenever cupy imports. An NN winner needs the GPU for H6.

## Prior record

**Searches run** (worktree `docs/`, `/bin/grep -rliE <pattern> PROJECT_FILE_CATALOG.md TB_RULING_* TB_RULING_REQUEST_* PROPOSAL_* CHAPTER_* *_AUDIT.md TODO_CONSOLIDATED.md BACKLOG.md LEADS.md exchange`):

| # | Pattern | Hits read |
|---|---|---|
| S1 | `e2e_vote\|vote harness\|H6\b` | `CHAPTER_6/7/11_AUDIT.md`, X-GCVM files. Only unrelated tokens; **no prior H6 document** |
| S2 | `walk[- ]forward` | `TB_RULING_S176_WATCHER_KPI.md` Phase C (cited by the map), S177/S179, KPI proposals, Ch7/Ch13 audits (file list) |
| S3 | `tie[- ]?break\|argsort\|ties in (the )?rank` | `CHAPTER_7_AUDIT.md:88-89`; `TB_RULING_S178_KPI_GOVERNANCE.md` §5 (`:238-262`); `TB_RULING_S179_IMPLEMENTATION_AUTH.md` §2.5 (`:142-174`) |
| S4 | `fallback_mean\|mean fallback\|mean-of-features` | `CHAPTER_7_AUDIT.md` B2 (`:200-212`) |
| S5 | `next_idx\|len\(lottery_history\)\|len\(history\)` | `CHAPTER_7_AUDIT.md` Q2 (`:71-107`), O3 |
| S6 | `zero-fill\|missing columns\|0\.0 for missing` | file list; Ch6 B5 cited by the map §2.4 row 5 |
| S7 | `GPU_AVAILABLE\|cupy\|Could not load config\|reinforcement_engine_config\|optimal_window_config` | `CHAPTER_7_AUDIT.md` N3 (`:318-321`) |
| S8 | `MULTI_MODEL_AVAILABLE` | `docs/` (all): **none** |
| S9 (rev 2) | read `e2e_grader.py` INPUT FORMATS (`:106-129`), `SCHEMA_POOL` (`:148`), `pool_sha256` (`:193-196`), `load_dataset` (`:233-255`), `read_pool_records` (`:359-379`), `resolve_records` (`:386-532`) | grader worktree, read-only, file sha256 `a4d96fe7ac441a8befc93f55a638eb70e36e7d0cd70f24eaeb60d08b7a52169a` |

**Documents that already decide or record the subject:**
- The H6 route and TODO-E2E-8/9/12: map §2.4 (4), §3.
- **The pool schema: the lead's decision (2026-11-23), defined by the grader's reader** (S9).
- No tie-break at `pg:836`: Ch7 `:88-89`; S178 §5; S179 §2.5 (approved, not implemented).
- Mean fallback: Ch7 B2. Config fallback: Ch7 N3. Combined-index vote, skip 0: Ch7 Q2/O3. Repeats kept: Ch7 B3; S179 §2.5.
- Cutoff and walk-forward rule: S176 Phase C, as the grader enforces it (`e2e_grader.py:424-427`).

**Labels:**

| Item | Label |
|---|---|
| H6 harness, run record, diagnostics file | **NEW** (built here) |
| Pool record schema `tfm.e2e.pool.v1` | **ALREADY RECORDED**: lead decision; defined by `e2e_grader.py`. H6 conforms and defines nothing |
| `--data-cutoff-index` as a declared argument | **ALREADY RECORDED** (lead decision); the `t` option is **NEW** |
| `--pending-date`/`--pending-session` declared arguments | **NEW**: the grader requires date and session strings on a pending target (`e2e_grader.py:439-442`) |
| Empty pool written as `abstain: false, pool: []` | **NEW** as H6 behaviour: it defers to the grader's declared `empty_pool` rule (`e2e_grader.py:457-463`) |
| Byte identity with the CLI (TODO-E2E-8) | **ALREADY RECORDED** as required; **NEW** as a measured pass |
| Ties set 16/20 of A0's slots on run H | **NEW** (measured, rev 1 §5 F7) |
| Mean fallback precision; missing sidecar-free constructor; cupy device choice | as in rev 1: **ALREADY RECORDED** (Ch7 B2) with a NEW precision; **NEW**; **NEW** |
| **CHANGES A RULING** | **none** |

**Not searched:**
- the host main tree beyond the read-only reads named here;
- the ser8 archives, the private repository, project files outside the repository;
- `docs/exchange/` bodies;
- the grader's tests and drafts (only `e2e_grader.py` was read).

## 1. What was built

| File | Lines | sha256 |
|---|---|---|
| `scripts/diagnostics/e2e_vote_harness.py` (untracked, **rev 4**) | 1125 | `99791061cde4c05fcf7e1dd831c05a04bda7f389e4000ce524f58bac55774544` |
| `tests/test_e2e_vote_harness.py` (untracked, **rev 4**) | 1367 | `f5bc4f597a0287c1c711ed5b5c9e6806e924c18a77ef3687eea17d5aa29575c5` |
| `logs/e2e_vote/selftest_run7.log` (CPU, rev 4, ignored) | — | `65ccd320c7326c87bc11d9f0d613c1c8eba86d844efcd222b9068962dd7ae337` |
| `data/e2e_vote_selftest/20261123_133235/selftest_summary.json` (rev 4) | — | `f3e5b1c367a4444b997daac9607de42450affc52f7478e4906a0beda6d4b486e` |
| rev 3 harness / test (superseded) | 1024 / 1281 | `f36139ef…` / `9bc69a1d…` |
| `logs/e2e_vote/selftest_run6.log` (CPU, ignored) | — | `78cdbbddd619a7d063cad60e693984b75309bfdb724b59ae072377ab18191fa8` |
| `logs/e2e_vote/selftest_gpu_run2.log` (GPU, ignored) | — | `d55fb45f3dd300c942ae04463535510f9f9ef8e30d4ae6127afcd6ba97a9b796` |
| `data/e2e_vote_selftest/20261123_083531/selftest_gpu_summary.json` | — | `a6286234cfd5d5f17bfefae73eb2bbcb176bc49ad0705bb26f01f06970d87f9c` |

CPU run 6 and GPU run 2 used the rev 3 files. **CPU run 7 used the rev 4 files** and is the run behind every run H execution. The GPU suite was **not** rerun on the rev 4 files. Rev 4 changes the dataset loader, the history handed past a seal, and two sha checks. It does not touch the device path. The rev 2 hashes are superseded.

Gate 22 will red on the untracked `scripts/diagnostics/e2e_vote_harness.py` until it is landed. This is expected; the allowlist was not widened.

**How it calls the module** (`pg` = `prediction_generator.py` at `41ad509`, `vh` = the harness):
- **`vh:459-479`** `build_config`: `PredictionConfig.from_json(--config)`, then the CLI's own override order (`pg:1018-1029`).
  - Forced: `use_dual_sieve=False` (TODO-E2E-9) and `save_predictions=False`.
  - Refused: a missing config.
- **`vh:516-567`** `build_generator`: `PredictionGenerator(config, logger)` is loaded once and `gen.model` is wrapped in `RankOnce` (`vh:241-325`).
  - The first `predict(X)` runs the model. Every later call must present a byte-identical X and gets the same scores; any other X stops the run.
  - For the unlearned arms, the generator is built with `_construct_unlearned` (`vh:481-490`, TODO-E2E-V1).
  - **Rev 3:** for a `neural_net` sidecar, `check_nn_normalisation` (`vh:518-555`, called at `vh:583-584`) runs before the model is wrapped. It refuses unless:
    - `gen._normalize_features` is true, and `_scaler_mean`/`_scaler_scale` are present, finite and of schema length (`pg:385-449` is best-effort, and `pg:817-824` skips a length mismatch silently);
    - the wrapper's `_y_mean`/`_y_std` are loaded and finite, with `y_std > 0` (`neural_net_wrapper.py:437-441`; without them `predict()` returns normalised outputs, `:357-359`).
    - `y_std > 0` is my addition to the lead's rule: a zero std collapses every score to one value.
  - The run record gets `ranking_info.nn_normalisation` (scaler length, schema length, `y_mean`, `y_std`, device).
- **`vh:576-597`** `vote_one`: per draw, `gen.generate_predictions(survivors, combined[:t], None, k=config.k)`, the CLI's own call (`pg:1066-1071`).
  - The module ranks (`pg:836`) and votes `seq[next_idx]` with `next_idx = t` (`pg:839-850`).
- **`vh:599-657`** `target_for`, `cutoff_for`, `make_line`: build the grader record and the diagnostics line.
- **`vh:659-687`** `run_block`: the first draw runs in the parent and fixes the ranking; the other draws run in forked workers.
- **Checks after the vote:**
  - `vh:878-880`: one record per t, in order, with pools and diagnostics in step;
  - `vh:889-895`: the model ran once, and `len(block)-1` draws reused the ranking.
- **Input checks before the vote** (`vh:796-824`):
  - the dataset is loaded (`vh:207-226`), and `--history` must equal its draws (`vh:228-238`);
  - a pending target needs declared date and session;
  - an integer cutoff must be ≤ the block start.

**Parameters** (none fixed; declared defaults in `--help` and in `run_record.json` → `declared_parameters`):

| Parameter | Default | Notes |
|---|---|---|
| `--dataset` | **required** | frozen `[{date, session, draw}]`. Its sha256 goes on every record; `target.date`/`target.session` come from it |
| `--history` | the dataset's draws | optional Step 6-style history file(s); must equal the dataset's draws value for value |
| `--data-cutoff-index` | **required** | an integer ≤ every target in the block, or `t` (cutoff = target index, i.e. `history[:t]`). Written on every record |
| `--pending-date`, `--pending-session` | none | required only when the block includes t = len(dataset) |
| `--arm-id` | `<ranking>/<vote-arm>[...]` | must match the grader's arm list; give it explicitly (e.g. `A0`, `A1`) |
| `--k`, `--pool-size` | the config's values | CLI semantics: the pool is the first `k` of `pool_size` ranked votes (`pg:653`) |
| `--block-start`, `--block-end` | required | half-open; `--block-end` may be len(dataset)+1 |
| `--ranking` | `sidecar` | or the unlearned keys `file_order`, `seed_asc`, `field_sum` (`--ranking-fields`, `--ranking-npz`) |
| `--tie-break` | `module` | `module` = numpy argsort as in `pg:836` (CLI-identical); `seed_asc` = S178 §5's contract; `file_order` |
| `--vote-arm` | `step6_baseline` | `per_skip --arm-skip s` is built and tested; `midday_stream_index` is a hook (`vh:408-421`) |
| `--on-missing-feature` | `fail` | `record` = zero-fill as the CLI does (`pg:798`) |
| `--workers`, `--allow-gpu`, `--out-dir` | 1 / off / required | the out-dir must not exist |

## 2. Output files

- **`pools.jsonl`**: one `tfm.e2e.pool.v1` record per target in the block, in target order. The fields are exactly those the grader reads (`e2e_grader.py:106-114`, `:404-477`):

  ```
  {"schema": "tfm.e2e.pool.v1", "arm_id": str, "config_id": str, "dataset_sha256": hex,
   "data_cutoff_index": int, "target": {"combined_index": int, "date": str, "session": str},
   "pool": [int, ...], "pool_sha256": hex, "abstain": bool, "abstain_reason": str (only if abstain)}
  ```

  - **`pool`:** Step 6's raw list, order and repeats kept.
  - **`pool_sha256`:** the grader's definition, `sha256(canonical_json(pool).encode("ascii"))` (`e2e_grader.py:193-196`, `vh:146-149`).
  - **`config_id`:** sha256 of the canonical JSON of every identity input. These are the input sha256s (survivors, config, history files, dataset, sidecar, checkpoint, NPZ), the declared cutoff, ranking, fields, tie-break, vote arm and skip, the effective `PredictionConfig.to_dict()`, and the harness and `prediction_generator.py` sha256s. The grader's arm list must carry the same value; it is printed in `run_record.json`.
  - **`abstain: true`** only for a Step 6 signal-gate skip (`abstain_reason: "signal_gate:<status>"`).
  - **An empty pool** (`pg:631-640`, or success with no predictions) is written as `abstain: false, pool: []`, so the grader's declared `empty_pool` rule decides. Its cause goes to the diagnostics file.
  - Records carry no outcome and no hit.
- **`vote_diagnostics.jsonl`** (`tfm.e2e.vote_diagnostics.v1`, never read by the grader): one line per target with `arm_id`, `combined_index`, `block_pos`, `vote_index`, `pending`, `vote_status` (`pool` | `abstain` | `empty`), `note`, `pool_len`, `raw_scores`, `k_requested`, `pool_size_requested` and `model_type`.
- **`run_record.json`:** as in rev 1. It adds the dataset path, sha256 and rows, the cutoff and pending declarations, empty-pool counts, and the diagnostics file sha256.
- All files are written `.partial`-then-rename. The record is written last.

## 3. Fixture and test results

**Fixture** (built in the test under `data/e2e_vote_selftest/<stamp>/fixture/`; no run H data):
- **Step 3 rows:** from `full_scoring_worker.score_survivors` on CPU, 40 seeds plus one duplicated row (41), sorted by score as the `.sh` does.
- **Rev 2 change: the labels come from a label block only.** Step 3's holdout argument is combined [240, 270), so **[270, 300) is seen by nothing upstream**. That is the map's Q1 route and the block G12 grades. The rest of the fixture is as in rev 1:
  - synthetic histories: 240 train and 60 holdout;
  - a synthetic dataset (`date`, alternating `session`, `draw`); row 245 is 000;
  - Step 5's loader and `compute_signal_quality`;
  - LightGBM and XGBoost CPU checkpoints with a 3.4.0-shape sidecar (**fixture-made**);
  - the gate-blocked and missing-checkpoint variants;
  - Step 1-style NPZs.
- **The oracle for learned pools is the real CLI** as a subprocess (cwd = fixture dir, CUDA hidden).
- **Format check on every H6 run:** the test's `run_harness` checks that every `pools.jsonl` record has exactly the grader field set and the schema id, and lines up one-for-one with its diagnostics line.

**Gates** (run 4, `logs/e2e_vote/selftest_run4.log`):

| Gate | Result |
|---|---|
| G1 byte identity at t = len(history) vs CLI | **PASS** 4/4 (LightGBM cfg / k=10 / k=25 ps=30; XGBoost). `pool`, `raw_scores` and `pool_sha256` identical; the pending target carries the declared date/session |
| G2 walk-forward t = 250…299 | **PASS**: 50/50 in order; 3/3 sampled t = CLI on `history[:t]`; target date/session = dataset rows; cutoff `t` = target; no outcome field anywhere |
| G3 ranking once | **PASS**: 1 model call, 49/49 reuse, X [41, 89] |
| G4 workers | **PASS**: `pools.jsonl` and `vote_diagnostics.jsonl` sha256 equal at 4 and 1 workers |
| G5 abstentions | **PASS**: 21/21 records `abstain: true` (`signal_gate:degenerate`), 1 pending |
| G6 fail-closed | **PASS**, 10/10 refusals: out-dir exists; bare-int row; missing checkpoint, both schema modes; block past len+1; `midday_stream_index`; dataset misaligned with history; **pending target undeclared**; **cutoff > target**; missing config |
| G7 per_skip | **PASS** (skip 0 = baseline 50/50; skip 3 = CLI on rows with `skip: 3`; X unchanged) |
| G8 unlearned keys | **PASS** (`file_order`, `seed_asc`, `field_sum` = independent oracle; `field_sum` without the NPZ refused) |
| G9 repeats kept | **PASS** (41 slots, 38 distinct) |
| G10 tie-break | **PASS** (tied key: `seed_asc` = oracle; tie report: group of 24 at the boundary, 3 slots set by tie order) |
| G11 missing schema feature | **PASS** (refused by default; `record` = CLI) |
| **G12 grader reader** | **PASS**, details below |

**G12 in detail.**
- **Read-only load.** The grader is imported from its worktree path with `sys.dont_write_bytecode = True`.
- **Three H6 arms on block [270, 301):**
  - `A1fx`: learned LightGBM, `--data-cutoff-index 270`;
  - `A0fx`: `file_order`, `--data-cutoff-index t`;
  - `Ablk`: gate-blocked sidecar.
- **Declarations written under the test's data dir:**
  - grade block `{270, 301, [evening, midday], frequency_control_block [0, 240)}`;
  - arm list with each arm's `config_id` from its run record and `upstream_unseen_from_index` 270;
  - abstention rule `{allow_abstention: true, allowed_reasons: null, empty_pool: "error"}`.
- **Calls, all the grader's own, unmodified:** `read_pool_records`, `load_dataset`, `load_declaration`, `parse_grade_block`, `parse_arms`, `parse_abstention`, `resolve_records`.
- **Credited assertions:**
  - no `GraderError`;
  - 30 eligible draws;
  - A1fx and A0fx each 30 graded and 1 pending;
  - Ablk 30 abstained and 1 pending;
  - for every record, the grader's `pool_sha256` = H6's = `G.pool_sha256(pool)`, `k_raw` = `len(pool)` and `k` = distinct count;
  - the grader file sha256 is unchanged (`a4d96fe7…`) and `scripts/diagnostics/` has no `__pycache__`, before and after.
- **Observation, not credited:** the grader's full CLI (`python3 -B`, `--no-provenance`, `--control-reps 50`, out-dir under this test's data dir) on A1fx + A0fx exited 0 with `controls_ok=True`. It wrote `SHA256SUMS`, `SUMMARY.md`, `per_draw.jsonl` and `summary.json`. Its hit lines are on synthetic data (see Null results).
- **Checked after the run:** the grader sha256 was unchanged and `scripts/diagnostics/` held only `e2e_grader.py`.

**Mutants.** A mutant counts as killed only if it was applied, its hit counter is > 0, and the credited gate's own `GateFail` tag fired.

| Mutant | Credited gate | Result |
|---|---|---|
| M1 vote at t−1 | G2 | KILLED |
| M2 ranking reversed | G1 | KILLED |
| M3 features drift after the first draw | G3 | KILLED |
| M4 a record's target shifted | GI (block integrity) | KILLED |
| M5 repeats collapsed | G9 | KILLED |
| M6 per_skip ignored | G7 | KILLED |
| M7 no-model refusal removed | G6 | KILLED |
| M8 reverse survivors / dual-sieve on | G1 | KILLED |
| M9 tie-break ignored | G10 | KILLED |
| M10 missing-feature check blind | G11 | KILLED |
| **M11** cutoff written as t+1 | G12 (grader: "data_cutoff_index 271 > target 270") | KILLED (93 hits) |
| **M12** target date/session from the next row | G12 (grader: stale target) | KILLED (93 hits) |
| **M13** `pool_sha256` by `json.dumps` default separators | G12 (grader: "pool_sha256 does not match") | KILLED (93 hits) |
| **M14** wrong `dataset_sha256` | G12 (grader: "differs from the bound dataset") | KILLED (93 hits) |

93 hits = 31 records × 3 arms.

**Rev 3 additions to the CPU suite** (run 6, `logs/e2e_vote/selftest_run6.log`):
- The fixture adds three NN sidecars, all trained by Step 5's own `train_single_trial.train_neural_net` (hidden [16, 8], 20 epochs, on CPU in CPU mode and on GPU in GPU mode):
  - `models_nn`: `normalize_features=True`, with scaler and `y_mean`/`y_std`;
  - `models_nn_noscaler`: `normalize_features=False`;
  - `models_nn_noy`: `models_nn`'s checkpoint with `y_mean`/`y_std` removed.
- **G13cpu: PASS.** With CUDA hidden, the NN sidecar gives no pool. Harness: `RuntimeError: No CUDA GPUs are available`. CLI: exit 1, same error.
- Totals: **13/13 gates, 14/14 mutants killed, 77.8 s, peak RSS 692 MB.**

**Measured run time and memory** (run 4; run 6 above for rev 3):
- self-test 86.7 s wall; peak RSS 712 MB (`/usr/bin/time -v`); largest child 695 MiB;
- scale probe (synthetic, 12,156 × 89, history 18,068): 0.732 s per draw on 1 worker, 0.164 s effective on 6, about 0.62 GB per process.
- **Extrapolation [UNVERIFIED]:** 3,614 draws take ≈ 44 min on 1 core and ≈ 10 min on 6.
- **Memory caveat** as in rev 1: real Step 3 rows are larger. Measure the parent on a 10-draw block before choosing a worker count (CLAUDE.md rule 9).

## 3a. GPU-path checks (rev 3; owner go 08:29, synthetic fixtures only)

**Limits kept:**
- no run H data, no rigs, no Step 1–5 pipeline run;
- `nvidia-smi` was checked before each run; the only process was `sunshine` (allowlisted), 264 MiB, 0 % use. The test also waits up to 300 s if any other compute process is on the card;
- each run was well under the few-minute limit (36 s);
- card memory stayed under 3 GB, as the test asserts;
- the card was back at 264 MiB afterwards.

**Mode:** `python3 -u tests/test_e2e_vote_harness.py --gpu`. CUDA is visible to the test, the CLI subprocesses, and H6 (`--allow-gpu`).

| Gate | Result |
|---|---|
| GG1 byte identity at t = len(history) vs CLI, card visible | **PASS** 3/3: LightGBM, XGBoost, `neural_net` (`device=cuda`). `pool` and `raw_scores` identical |
| GG2 NN walk-forward t = 290…299 on GPU | **PASS**: t ∈ {290, 299} = CLI on `history[:t]`; 1 model call; `--workers 2` `pools.jsonl` sha256 = `--workers 1`. The forked children reuse cached scores and never touched CUDA |
| G13 NN normalisation check | **PASS**, details below |

**G13 in detail:**
- Complete sidecar: accepted, report `{normalize_features: true, scaler_len 89 = schema_len 89, y_mean 0.11278, y_std 0.01727, device cuda}`.
- `nn_noscaler`: refused, "input scaler not loaded".
  - **Observed:** the CLI on the same sidecar exits 0, `status: success`, 20 predictions, raw_scores [0.0733, 0.0955].
- `nn_noy`: refused, "y_mean/y_std not loaded".
  - **Observed:** the CLI exits 0, `status: success`, 20 predictions, raw_scores [0.0722, 0.2353]. These are normalised-scale outputs, which the CLI reports as success.

| Mutant | Credited gate | Result |
|---|---|---|
| M15 `check_nn_normalisation` removed | G13 ("nn_noscaler: accepted") | KILLED |
| M16 y-part blinded (y_mean/y_std faked during the check only) | G13 ("nn_noy: accepted") | KILLED |

- **GPU run 1, a test defect:** M15 was not credited because G13 read the complete-case report first and raised `KeyError`. I reordered G13 so the refusals are asserted first, then reran.
- **Measured (GPU run 2):** wall 36.1 s (`/usr/bin/time` 37.3 s); peak card memory 922 MiB total, 658 MiB above the 264 MiB baseline (163 samples at 0.2 s); peak RSS 1,282 MB.

**Not tested on GPU:**
- CatBoost (no fixture);
- run-H-sized NN inference;
- more than 2 workers with CUDA;
- whether GPU NN scores equal CPU NN scores. This cannot be tested: the CPU NN path does not load (V6).

## 3b. Run H vote (rev 4; owner go 13:26; TASK_VOTE_RUNH)

**Prior record for rev 4.**
- **Searches:** `/bin/grep -rlIiE "e2e_runH_vote|TASK_VOTE_RUNH|seal-from|grade block.*seal"` over the main tree's `PROJECT_FILE_CATALOG.md TB_RULING_* TB_RULING_REQUEST_* PROPOSAL_* CHAPTER_* *_AUDIT.md TODO_CONSOLIDATED.md BACKLOG.md LEADS.md exchange/`: **no hits**. The same scope for `sealed until grading|grade block`: no hits in the main tree.
- **Read:** `X-E2E-GRADE_R1_BETA_REVIEW.md` (copy found in worktree `pwc-offset`, `docs/exchange/from_beta/`). It is not in the main tree's `docs/exchange/from_beta/`.
- **Already decides:**
  - Owner ruling Q1, 08:08 (`OWNER_RULING_e2e_Q1_20261123.txt`): label/grade cut, grade block sealed until grading, midday counted.
  - Beta R1 Q3 (`:61-65`): combined index for both sessions.
  - Beta R1 Q4 (`:69-72`): same 20-vote cap for both arms, raw votes and actual k kept, Hit@20 only when 20 distinct values exist.
  - The task file's declarations.
- **Labels:**
  - targets, vote rule, config, arms, cutoffs: **ALREADY RECORDED** (task, owner Q1, Beta R1 Q3–Q4, this note Q3/Q6);
  - `--seal-from` (V10), `--expect-*-sha256`, the loader-only check: **NEW**;
  - the loader's default rejection of evening records: **NEW** (measured);
  - **CHANGES A RULING: none.**
- **Not searched:** the ser8 archives, the private repository, files outside the repository, and `docs/exchange/` bodies other than the one named.

### 3b.1 Declared before the run

`data/e2e_runH_vote/run_record_declared.json` (sha256 `890c6b3ff26f68a507d7fcf4a9e7aa8839a381de02cf618359ff3c2f4fff46a4`) was written after self-test run 7 and before the first run H execution. It holds the targets, vote rule, config, both arms and their cutoffs, the seal, the stop rules and the code hashes. The launcher, `data/e2e_runH_vote/run_arm.sh`, has sha256 `0a8e0912…d9`. Its first version failed before running anything (`set -u` against the venv's activate script, `LD_LIBRARY_PATH: unbound variable`). It was fixed, and the declaration was not changed.

- **Targets:** combined [16261, 18068), every record, both sessions: 1,807 = 904 midday + 903 evening. Each has its original combined index, date and session from `daily3.json`, sha256 `513648160d35…68f6` (verified).
  - That file is the one input outside `e2e-map/data/e2e_runH/`. The task names it by sha256, and it is the only source of the evening grade-block identities.
- **No outcome read: `--seal-from 16261`.** The harness hashes the dataset bytes. It parses the dataset with a hook that replaces the draw value of every row ≥ 16261 before the row exists; the run record states 16,261 held and 1,807 discarded. History is `copy/train_history.json` (`0fe05436…`, 14,454) + `copy/label_block_history.json` (`703fe276…`, 1,807). It equals the dataset's held draws value for value (`check_aligned`).
  - Self-test G14 proves the seal: a dataset whose sealed draws are not even integers gives byte-identical pools and diagnostics to the unsealed run on the real dataset.
- **Config:** `prediction_generator_config.json` (`d77c3ac8…`) with `--k 20 --pool-size 20`. Effective: `use_dual_sieve=False` (forced), `prng_type java_lcg`, `mod 1000`, `default_skip 0`. All of it is in `config_id`.
- **Survivors:** `step3_full/survivors_with_scores.json`, sha256 `37c8b146…187b` (verified), 12,156 rows, 89 schema features present in every row (0 missing).
- **Vote:** Step 6 as built, `next_idx = t`.
  - The per-survivor skip, by the module's own rule (`pg:530-553`), is **0 for all 12,156 rows**: no `skip`, `metadata.skip` or `features.skip_hypothesis` is present, so the default applies. The Step 1 NPZ's skip range is [0, 16]. The vote uses 0, as Step 6 does (TODO-E2E-12).

| Arm | Ranking | Tie order | `data_cutoff_index` | `config_id` |
|---|---|---|---|---|
| **A1** (counted) | frozen Step 5 winner, CatBoost. Sidecar `55e6bbdc…`, checkpoint `c2abe721…`, both **checked by the harness at load** (`--expect-*-sha256`) | module argsort; **no boundary tie** (12,156 distinct scores) | 16261 | `bd932b3ed234fad278376d89798f89ec97a700a951d51c3b11420bdeb41d68bc` |
| **A0** (descriptive) | `forward_matches + reverse_matches` from `copy/bidirectional_survivors_binary.npz` (`3f6a884d…`), 12,156/12,156 by seed | `seed_asc`; **16 of 20 slots set by tie order** (boundary group 39 rows at 0.91667; 7 distinct scores; F7 reproduced) | `t` (Q6) | `52cbe2119a7f2451a52e3f875bf6f9243a21d2c93d47cf125b42808f26bbb2f1` |

- **The checkpoint the module loads** is the absolute path in the sidecar, `step45/catboost/models/reinforcement/best_model.cbm` (`model_factory.py:318-325`). Its sha256 equals the frozen `winner/best_model.cbm`.
- **A0's cutoff `t`** is a declaration looser than the fact. H6 handed A0 the same sealed history as A1, so no draw ≥ 16261 reached either arm.

### 3b.2 Smoke (first 5 targets, [16261, 16266), 4 workers)

| Arm | Records | Wall | Peak RSS (main) | Card | Min MemAvailable |
|---|---|---|---|---|---|
| A1 | 5 pools | 7.2 s | 694 MB | 264 MiB (baseline, untouched) | 10,348 MiB |
| A0 | 5 pools | 7.2 s | 688 MB | 264 MiB | 10,318 MiB |

- **The CatBoost winner loads, on CPU.**
  - The harness hides CUDA (`CUDA_VISIBLE_DEVICES=""`, run record).
  - The wrapper logs `use_gpu=True, devices=0:1`, but that only sets fit parameters (`catboost_wrapper.py:144-153`). `load` builds a plain `CatBoostRegressor().load_model` (`:308-324`), and `predict` calls `model.predict(X)` (`:278-291`).
  - The card stayed at its 264 MiB baseline.
  - **Why CPU:** the model is called once, on 12,156 × 89. The per-draw cost is the module's sequence generation, which runs on CPU. The module does not need the card for this model, so the run stayed off it. That also left the card free for the selfplay task.
- **CPU work used several workers:** 4 for the smoke, 6 for the full runs (8 cores).

### 3b.3 Full block (nohup, 6 workers, A1 then A0)

| | A1 | A0 |
|---|---|---|
| **Pool records written** | **1,807** (904 midday, 903 evening) | **1,807** (904 midday, 903 evening) |
| Targets = [16261, 18068) one for one, in order | yes | yes |
| Abstentions / empty pools | **0 / 0** | **0 / 0** |
| Raw k (votes) | 20 on every draw | 20 on every draw |
| **Distinct k**: min / median / max | 18 / 20 / 20 | 18 / 20 / 20 |
| Draws with 20 distinct | **1,492** of 1,807 (midday 746 of 904) | **1,498** of 1,807 (midday 748 of 904) |
| Distinct-k histogram | 18: 25, 19: 290, 20: 1,492 | 18: 23, 19: 286, 20: 1,498 |
| **Ranking taken once** | model calls 1; 1,806 draws reused it, X sha checked each time | same: 1 call, 1,806 reused |
| **Top-20 survivor set constant** | yes: 1 distinct vector of the 20 voting scores across all 1,807 draws; the 20 are distinct | yes: 1 distinct vector across 1,807; the 20 are distinct tie-break positions |
| `pools.jsonl` sha256 | `d0a80f8eca7031e9470dbbc5113c4d16b3b29fcfc74029a8d5a8ec48a340aa12` | `7c1cc332bf29daee4cf751339a6b4b344c147bbd23a57faba292364e44014506` |
| `vote_diagnostics.jsonl` sha256 | `7dcdd537cccd0269…` | `70e6d4c62fdf8366…` |
| `run_record.json` sha256 | `002fbc699114befc7a45069604767e3ea7731f9dac1c78bebb79f740ce46a472` | `39efab240bd4709b933b96d385f8300dc0dc5a4eb01717649e5355b37ce3d9ab` |
| Wall (`/usr/bin/time`) | 3:49.7 (vote 224.0 s) | 5:35.1 (vote 329.9 s) |
| Peak RSS: main / largest child | 694 MB / 672 MiB | 689 MB / 672 MiB |
| Process-group RSS sum (counts shared pages more than once) | 3,556 MiB | 3,223 MiB |
| Min MemAvailable / card | 9,462 MiB / 264 MiB (baseline) | 9,565 MiB / 264 MiB |

- **How the constant top 20 is established.** Step 6's result does not name the voting seeds. The proof has three parts:
  - one score vector serves every draw (RankOnce checks the X sha and the call count);
  - the module's argsort is deterministic;
  - the 20 voting scores are identical, and distinct, on every draw.
- **Distinct k below 20 means repeats.** The same 20 survivors vote on every draw. On 315 (A1) and 309 (A0) draws, two or three of them generate the same value at that index. These pools are written raw, as Step 6 emits them (Beta R1 Q4: report variable breadth; fixed Hit@20 unavailable on those draws).
- Paths: `data/e2e_runH_vote/full_A1/`, `full_A0/`, with `*.log`, `*.time`, `*.watch.log` and `*.summary` beside them.

### 3b.4 Not tested or not done

- No grader run, no outcome matching, no hit count (by instruction).
- The GPU suite was not rerun on the rev 4 harness files (see §1).
- NN, LightGBM and XGBoost winners were not voted on run H; only the frozen CatBoost winner was.
- No arm other than A1 and A0, and no other k.

### 3b.5 The grader LOADER only (schema read, no outcome matching)

- **Script:** `data/e2e_runH_vote/analyse_and_load.py` (`d9945b42…`). Output: `analysis.json` (`3cd9364a…`) and `analyse_and_load.log` (`ffb848b6…`).
- **Grader:** `e2e_grader.py`, imported read-only from its worktree. Its sha256 was `a4d96fe7…` before and after, and no `__pycache__` was written.
- **Called:**
  - `read_pool_records`, `parse_grade_block`, `parse_arms`, `parse_abstention`, `resolve_records`.
  - Grade block [16261, 18068), sessions `["midday"]`; the frequency-control block is declared as [0, 14454) because the parser requires one.
  - Arms: A1 with `upstream_unseen_from_index` 16261, A0 with 14454 (Q6); each arm's `config_id` taken from its run record.
- **Not called:**
  - The grader's `load_dataset`, which parses and validates every draw.
  - `grade_draw`, and any statistic.
  - `resolve_records` instead gets the dataset identity from H6's sealed loader as (date, session, None). It reads only date and session (`e2e_grader.py:434-435`, `:520-521`). The dataset sha256 is the hash of the file bytes.

| Loader run | Verdict |
|---|---|
| A1 alone, grader default `out_of_block_records=error` | **REJECT**: "pools line 1: target 16261 (evening) is outside the grade block (declared rule: error)" |
| A0 alone, default | **REJECT**: same message |
| A1 alone, `exclude_and_count` | **ACCEPT**: 904 eligible; 904 bound, 0 abstained, 0 pending; 903 excluded (evening) |
| A0 alone, `exclude_and_count` | **ACCEPT**: same counts |
| A1 + A0 together, `exclude_and_count` | **ACCEPT**: 904 eligible; per arm 904 bound, 903 evening excluded |

- **Meaning:** the files are in the grader's schema. Every record's config_id, dataset sha, cutoff ≤ target, target identity, pool values and pool sha pass. The midday completeness check passes (904 of 904 per arm).
- A grading run on these files must declare `--out-of-block-records exclude_and_count`. The alternative, a both-session grade block, would count evening, and the ruling counts midday only. Either way it is a grader declaration; H6 was asked to write both sessions.

## 3c. Follow-up votes on run H (rev 5; owner go 14:40; TASK_FOLLOWUP_VOTE). FOLLOW-UP, DESCRIPTIVE, NOT COUNTED

**Prior record for rev 5.**
- **Searches:** `/bin/grep -rlIiE "e2e_runH_vote_followup|TASK_FOLLOWUP_VOTE|follow-?up vote|pool size 300|pool_size 300"` over the main tree's `PROJECT_FILE_CATALOG.md TB_RULING_* TB_RULING_REQUEST_* PROPOSAL_* CHAPTER_* *_AUDIT.md TODO_CONSOLIDATED.md BACKLOG.md LEADS.md exchange/`: **no hits** (rc 1).
- **Already decides:**
  - the task file and owner go named in the header;
  - the run H rules of §3b (seal, cutoffs, Beta R1 Q3–Q4, owner Q1);
  - the lead-handling rule that follow-ups are run, not noted.
- **Note:** the owner go text quotes the first counted grade's A1 total. That was read. No setting depends on it; every setting comes from the task file (declaration, `owner_go.note`).
- **Labels:**
  - the seven files, the arms, blocks and pool sizes: **ALREADY RECORDED** (task);
  - the flags `--k N --pool-size N` as the way to raise the cap: **NEW** (this declaration, from `pg:653`, `:841`);
  - **CHANGES A RULING: none.**
- **Not searched:** the ser8 archives, the private repository, files outside the repository, and `docs/exchange/` bodies.

### 3c.1 Declared before the first run

- **Declaration:** `data/e2e_runH_vote_followup/run_record_declared.json` (`32de5f9a1a2cebffd92feddce50c15cfa169a5eb60476875c1b1b551c066664f`), written at about 14:43 PST before any follow-up run. No arm, k or setting beyond it was run.
- **Pool-size flags:**
  - The cap of ranked survivor votes N is `--pool-size N`, which is `ranked_idx[:pool_size]` (`pg:841`).
  - It is used together with `--k N`, because the module truncates with `pool_predictions[:k]` (`pg:653`).
  - Both are CLI overrides of `prediction_generator_config.json` (`pg:1018-1021`). They are recorded in `effective_config` and so in `config_id`.
  - The vote rule is otherwise unchanged: step6_baseline, `next_idx = t`, the module's survivor skip (0 on all 12,156 rows), raw pools with repeats.
- **Common inputs:** as in §3b.1. The same config, survivors and dataset are used, the dataset is sealed at each file's block start, and the harness is `99791061…` (self-test run 7). CPU, 6 forked workers, nohup, one run after another.
- **History:**
  - F1 (seal 16261): train + label files.
  - F2 (seal 14454): train file only.
- **Launcher:** `run_one.sh` (`4850a015…`). Each run gets its own process group, `/usr/bin/time -v`, and a watcher that kills the group if MemAvailable < 2048 MiB.
- **Smoke:** F1_A1_p300 shape on [16261, 16266), 4 workers, in `smoke_p300/` (not one of the seven). rc 0, 14.4 s, RSS 694 MB.

### 3c.2 Run

- **Chain:** started 14:43:50; `CHAIN_EXIT=0` (`chain.nohup.log`). All seven runs had rc 0.
- **Memory:** lowest MemAvailable was 9,453 MiB. Group RSS sum (6 children sharing pages) peaked at 3,221–3,614 MiB; the declared figure was about 3.5 GB.
- **Card:** 264 MiB throughout, the sunshine baseline, so the card was not used.
- **Wall time:** pool size changed CPU time, not memory, as declared. The chain ran from 14:43:50 to 16:31:48 (1:47:58, `chain.nohup.log` mtime); the seven run walls sum to 1:47:53.

### 3c.3 Per-file results (no outcome)

Every file has 1,807 records over [block start, block end), one per target in order: 904 midday and 903 evening. Every file has **0 abstentions and 0 empty pools**. Raw k equals the pool size on every record. The ranking was taken once (1 model call, 1,806 reuses). There is 1 distinct raw-score vector across the draws, and the scores inside each pool are distinct, so the same top-N vote on every draw.

| File | Arm | Block | Pool = k | Cutoff | Distinct k min / median / mean / max | Draws with N distinct | Wall | Max RSS | `config_id` | `pools.jsonl` sha256 |
|---|---|---|---|---|---|---|---|---|---|---|
| F1_A1_p100 | A1 | [16261, 18068) | 100 | 16261 | 87 / 95 / 95.261 / 100 | 9 | 9:21.95 | 704,492 kB | `3486fd0f9f7787300f55046400394d34eb326354cd9f45be300b96ea96651c71` | `38e189f57d80a4b96649a0d5e5358bb3d19f14b72b1e8cdc92dee452093e3909` |
| F1_A1_p300 | A1 | [16261, 18068) | 300 | 16261 | 243 / 259 / 259.431 / 275 | 0 | 23:35.74 | 730,296 kB | `160de1c8504cf61c46d0b53f62a75763890a477885af70b34eca944258a765e3` | `137dc3750d9983b4b0e87fc6639cbc8d3f4aee58f30aaf60b15a364dd36b25ee` |
| F1_A0_p100 | A0 | [16261, 18068) | 100 | t | 86 / 95 / 95.164 / 100 | 10 | 11:10.40 | 688,588 kB | `62cbc7c4509e2fa2b7c35b2b5516aad16f86f875093ae61096dab2123442bffe` | `d957201332ef1f58f4678750b5f3540a6b4e9e712c4b1198e0718a68effb16b3` |
| F1_A0_p300 | A0 | [16261, 18068) | 300 | t | 240 / 259 / 259.1 / 276 | 0 | 24:57.54 | 688,448 kB | `608e7f5b8bc4f07bbcf9be9769a7934a4c8eac4120ba15f5fad74cfc8f68d78c` | `c2fff597a23388ea617629483222c96ec48ee0cabe0336c28f6e6de2c4f73256` |
| F2_A0_p20 | A0 | [14454, 16261) | 20 | t | 18 / 20 / 19.806 / 20 | 1,494 | 5:24.95 | 688,196 kB | `668224dd1166fa5a07b899f9a338e1fe497cbb4b64aa7a51633dddf6c9898613` | `60043f58790e61fd3f06a06c6dc61c8b9b15d83239467df476dd0e8118b070d2` |
| F2_A0_p100 | A0 | [14454, 16261) | 100 | t | 86 / 95 / 95.193 / 100 | 10 | 10:21.49 | 688,552 kB | `2c9ea597b4e5108aa003eea07696e930b2d77448c3a538ca4bbab7a2aa6ee9ae` | `74314b13b2a881546ee7fb6b3506aa523d480f2a7b265ed60dacdf5f1e92dc0c` |
| F2_A0_p300 | A0 | [14454, 16261) | 300 | t | 243 / 260 / 259.534 / 275 | 0 | 23:01.10 | 688,312 kB | `966e3755cec7eb4a5e3072950ae78571a763cb8af541a04f7ec9bd34d6214fe2` | `871c960093b27492bb2798226cdf7b9b1ac829c10cf46bdb471932ef4b2e207e` |

- **Sources:**
  - wall time and max RSS come from each `<name>.time` (`/usr/bin/time -v`);
  - the other columns come from `verification.json`;
  - histograms and per-session distinct k are in `verification.json`, under `files.<name>`.
- **Checks per file, by `verify_and_load.py`** (`99351d3a…`), all passed:
  - targets equal the block;
  - pool sha equals the run record;
  - k and pool_size, seal, cutoff and arm_id are as declared;
  - the dataset, survivors, config and harness shas match;
  - for A1, the checkpoint sha matches and the cutoff is 16261 on every record;
  - for A0, the ranking is field_sum with seed_asc, and the cutoff equals t on every record;
  - no outcome field is present.

### 3c.4 The grader LOADER only

Each file was loaded alone, with these declarations:
- `out_of_block_records=exclude_and_count`;
- grade block = the file's block, sessions `["midday"]`;
- frequency-control block [0, 14454);
- `upstream_unseen_from_index`: A1 16261 and A0 14454 on F1; A0 14454 on F2.

The dataset identity came from H6's sealed loader at the file's own seal, as (date, session, None). The functions used were `read_pool_records`, `parse_grade_block`, `parse_arms`, `parse_abstention` and `resolve_records`. Not called: the grader's `load_dataset`, `grade_draw`, or any statistic.

| File | Verdict | Eligible | Bound | Abstained | Pending | Excluded (evening) |
|---|---|---|---|---|---|---|
| all 7 | **ACCEPT** | 904 | 904 | 0 | 0 | 903 |

- **Grader:** sha256 `a4d96fe7ac441a8befc93f55a638eb70e36e7d0cd70f24eaeb60d08b7a52169a` before and after.
- **`__pycache__`:** the grader's `scripts/diagnostics/__pycache__/e2e_grader.cpython-310.pyc` exists. Its mtime is 13:51:31, which is after this note's rev 4 loader check (13:46:47) and before the follow-up chain (14:43:50). Its state did not change during this check. The verification field `unchanged_no_pycache: true` means only "same before and after". It was not written by this session's follow-up check, which imports with `dont_write_bytecode`. Which process wrote it at 13:51 is **[UNVERIFIED]**; it may be the grader session's counted grade.

### 3c.5 DONE

- **DONE file:** `data/e2e_runH_vote_followup/DONE.json` (`836a7cb6a3f95854a89e828f84b565f10865ab28f72aacae73b0b56629a84acd`). For each of the seven files it lists path, arm, arm_id, block, pool size, k, config_id, cutoff rule, seal and sha256, plus the declaration and verification shas.
- **Other files:** `verification.json` (`29357fca74132e8b97fb8a100443d4728701b5294411532adeeec018718837bd`) and `verify_and_load.log` (`49c0895e…`).
- **Not done:** no grading, no hit count, no comparison with any draw. No other arm, k or block was run.

## 4. TODO list

| TODO | Shortcut | Anchor |
|---|---|---|
| E2E-8 | H6 calls the module's ranking and vote; byte identity tested (G1, G2) | `pg:777-850`; `vh:576-597` |
| E2E-9 | dual-sieve path untouched | `pg:603-618`; `vh:476` |
| E2E-12 | vote at the combined index, survivor skip (0 on Step 3 rows) | `pg:839`, `:846`; `vh:408-421` |
| E2E-V1 | unlearned arms patch `MULTI_MODEL_AVAILABLE` for `__init__` only, then set `model_meta` | `pg:288-294`, `:829`; `vh:481-490`, `:556-567` |
| E2E-V2 | CLI plumbing mirrored (config overrides; survivors/history loaders) | `pg:1018-1029`, `:1042-1056`; `vh:170-205`, `:459-479` |
| E2E-V3 | `midday_stream_index` hook only | `vh:408-421` |
| E2E-V4 | `--tie-break` via rank positions before the module's argsort; default `module` | `vh:280-287`; S178 §5 |
| E2E-V5 | `--on-missing-feature` default `fail` | `pg:798`; `vh:503-514` |
| E2E-V6 | **Tested in rev 3.** On CPU the NN load fails closed: `RuntimeError: No CUDA GPUs are available`, because `pg:48-53`/`:352` choose `cuda` whenever cupy imports. On the GPU it equals the CLI (§3a). An NN run needs `--allow-gpu` and an owner go | `pg:350-363` |
| E2E-V10 (rev 4) | `--seal-from`: dataset rows at or after the seal are read for date/session only. The parse hook discards their draw values (`load_dataset`, `vh:226-270`), and votes past the seal get a None-padded history of length t (`history_for`, `vh:273-283`). The padding is safe only because the module reads `len(lottery_history)` alone for dict survivors (`pg:839`; `pg:788` is the int-survivor path, refused by H6). A future module read of the history would raise on `None`, not compute | `vh:226-283`, `:906-931`; self-test G14, M17, M18 |
| E2E-V9 (rev 3) | `neural_net` sidecars need the input scaler and `y_mean`/`y_std` loaded (lead rule); the module stays best-effort | `vh:518-555`; `pg:385-449`; `neural_net_wrapper.py:437-441` |
| E2E-V7 (rev 2) | pending target date/session are declared arguments, not derived from a calendar | `vh:599-604`, `:814-818` |
| E2E-V8 (rev 2) | one constant cutoff per run, or `t`. There is no per-arm cutoff law beyond the declaration | `vh:607-608`, `:821-824` |

## 5. Facts in `prediction_generator.py` met on the fixture route

These are unchanged from rev 1 except as noted below: F1 (`pg:288-294`, no sidecar-free constructor) through F10 (`pg:905`, cwd `predictions/`). The headline ones:
- **F3:** a missing checkpoint exits 1 when the schema is required; the mean fallback with `status: success` happens only with `--no-require-schema`.
- **F5:** `prng_type`/`mod` are inherited from the cwd's `optimal_window_config.json`.
- **F7:** on run H, `forward_matches + reverse_matches` has 5 distinct values over 12,156 rows, so 16 of A0's 20 slots are set by tie order.

Rev 2 met no new stop in `prediction_generator.py`. The schema change is H6-side only.

**Rev 3, measured:**
- **F4 update (`pg:48-53`, `:352`).** With CUDA hidden, the `"cuda"` device choice makes an NN checkpoint fail to load (`RuntimeError: No CUDA GPUs are available`), in both the CLI and H6. Trees are unaffected.
- **F11 (`pg:385-449`, `:814-824`; `neural_net_wrapper.py:437-441`, `:357-359`).** On GPU, an NN sidecar without the scaler, or without `y_mean`/`y_std`, gives a CLI pool reported as `status: success`, exit 0. H6 refuses both (V9).

## 6. Open questions

- **Q1 (owner): A0 tie-break.** `--tie-break seed_asc` (S178 §5's contract) or `module` (Step 6 as built) for the first A0 grade? 16 of 20 slots depend on it (F7).
- **Q2: closed** by the lead's decision. One schema, defined by the grader's reader.
- **Q3 (owner/lead): the run H Step 6 config.** `prediction_generator_config.json --k 20 --pool-size 20`, or Step 4's file once it exists? It goes into `config_id`.
- **Q4 (lead): workers and memory on run H.** Measure first.
- **Q5 (lead): NN winner.** The answer is now measured. H6 handles an NN winner only on the GPU (`--allow-gpu`), with the scaler and `y_mean`/`y_std` present. On CPU it fails closed. A run H NN vote would need its own owner go for the card.
- **Q7 (lead, rev 4): grading declaration.** The run H pool files carry evening records, as the task asked. The grader's default rejects them against a midday grade block. The grading run needs `--out-of-block-records exclude_and_count`, declared before outcomes are read (§3b.5).
- **Q8 (lead, rev 4): A0's cutoff `t`.** It was declared as Q6 proposed. H6 actually handed A0 only [0, 16261), the same as A1. Should a later run declare 16261 for A0 instead? It changes A0's `config_id`, not its pools.
- **Q6 (lead, rev 2): the declared cutoff for run H arms.** Used in rev 4 as proposed. The likely values were:
  - A1: the label-block end (the grader's `upstream_unseen_from_index` and H6's `--data-cutoff-index` both set there);
  - A0: `t`, with `upstream_unseen_from_index` at the holdout start.
  The values are a declaration; H6 writes whatever is declared and refuses a cutoff after a target.
