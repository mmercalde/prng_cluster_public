# READY: H3, the run-copy builder and the Step 3 harness (2026-11-23)

**Author and scope:** Claude (Team Alpha, agent 4), worktree `/home/michael/tfm_worktrees/e2e-map`, branch `e2e-map` at `41ad509`, uncommitted. The task is `logs/e2e_session/TASK_STEP3.txt` (H3) under `TASK_COMMON.txt`. The map is `docs/drafts/e2e_v1/MAP_E2E_THIN_PATH_20261122.md` §2.1 and §4.

**Null result first:**
- **No run H row was read into the harness or scored.** Every test ran on synthetic fixtures. The CPU tests ran first. GPU checks followed after the owner's go at 08:29 (§7).
- **Nothing was written outside this worktree's `data/`.** Gate W checked this, at file level.
- **The CUDA path is now tested on synthetic fixtures** (§7). It ran clean, and its values match CPU to float32 rounding. One limit stands: it has not been tested on run H data.

## Prior record

**Searches run.** Worktree at `41ad509`, `/bin/grep -rlniE`, scope `docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`, `docs/TB_RULING_REQUEST_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*`, `docs/*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/LEADS.md`, `docs/exchange/`:

| # | Pattern | Hits |
|---|---|---|
| S1 | `AGGREGATE_EOF\|aggregation heredoc\|inline aggregat` | none |
| S2 | `step ?3 harness\|run copy\|in-process.*step ?3` | none |
| S3 | `F-PARITY` | `CHAPTER_4_AUDIT.md`, `CHAPTER_4_FULL_SCORING.md`, `CHAPTER_CORRECTIONS_REPORT_20261111.md`, `TODO_CONSOLIDATED.md:62`, `:161` (P0-B, authorised, not implemented) |
| S4 | `_cpu_batch_generate\|unreduced` | `CHAPTER_4_AUDIT.md` (B4), `CHAPTER_4_FULL_SCORING.md`, `CHAPTER_8_AUDIT.md`, `CHAPTER_8_PRNG_REGISTRY.md`, `CHAPTER_CORRECTIONS_REPORT_20261111.md`, `exchange/from_beta/X-FINDINGS-PLAN_R1_BETA_REVIEW.md` |
| S5 | `holdout_quality.*0\.0\|written as 0` | `CHAPTER_13_AUDIT.md`, `CHAPTER_13_LIVE_FEEDBACK_LOOP.md`. Also recorded in Beta `X-STEP2-WIRING_R1` §2 ("the current worker catches label errors and writes zero"; pwc-offset worktree) |

The map's own Prior record (P1–P10) still applies and is not repeated here.

**Pattern source:**
- `~/tfm_worktrees/pwc-offset/scripts/diagnostics/step2_diagnostic_harness.py`, for the guards, path refusals, RSS/time record and git identity.
- Its self-test, for the no-write gate.

**Labels:**

| Item | Label |
|---|---|
| In-process Step 3 harness and run-copy builder | **NEW** (route recorded in map §2.1 (4), TODO-E2E-1, -3) |
| Sequential-fallback, CPU-generate and holdout_quality-zero detection | the defects are **ALREADY RECORDED** (Ch4 B6 / P0-B, Ch4 B4, Beta X-STEP2-WIRING R1 §2). Making them fail closed *in this harness* is **NEW**. Production is unchanged |
| The aggregation heredoc is at `run_step3_full_scoring.sh:275-442`, not `:299-408` | **NEW** correction of my map's §2.1 and the task text. `:299-408` lies inside the heredoc |
| **CHANGES A RULING** | none |

**Not searched:** ser8, the private repository, the `apply_s*.py` and changelog corpora, `docs/exchange/` contents beyond file names.

## 1. What was built

All three files are new, untracked, under `scripts/diagnostics/`. They trip Gate 22 until landed, which is expected; the allowlist is not widened.

| File | Lines | sha256 |
|---|---|---|
| `e2e_build_run_copy.py` | 432 | `bd4d1ae8…3a911a9` |
| `e2e_step3_harness.py` | 788 | `344ba862…9411cab9` (08:20: CUDA fail-closed check added, §6) |
| `e2e_step3_selftest.py` | 437 | `588bc549…3e983084` |

None touches a governed or hand-landed file, the main tree, `/dev/shm/prng`, `~/miner_staging` or a rig.

### 1.1 `e2e_build_run_copy.py`, the run copy

Run as `build()` (`:227`). It writes under `--out-dir`, which must not exist and must lie under `<worktree>/data/`.

**Files it writes:**
- `bidirectional_survivors_binary.npz`: a byte copy of the resolved NPZ. Its sha256 must equal `--generation-digest`, checked on the source, the copy, and the source again after copying.
- `train_history.json`: a byte copy.
- `label_block_history.json`: `holdout[0 : label_count]`, a bare int list in Step 3's holdout format.
- `grade_block_records.json`: `holdout[grade_start:]` as `{combined_index, holdout_index, date, session, draw}`, filtered by `--grade-sessions`.
- `configs/<name>`: byte copies of every `--config`.
- `run_record.json`: every declared parameter; every input sha256 (with an optional `--dataset-digest` check); every output sha256; block boundaries as index and date; per-session counts; excluded counts; git HEAD; the builder's own sha256.
- **The full holdout is deliberately not copied.** No consumer can be fed it by default.

**Refusals,** all exit 2 with `REFUSED`, before anything is created:
- an existing out-dir, or one outside `data/`;
- an out-dir under the main tree, any `~/skipsw_run_*` tree (run H included), `/dev/shm/prng`, `~/miner_staging` or `~/miner_ledger_archive`, checked after resolving symlinked parents (`check_out_dir`, `:129`);
- a wrong NPZ or dataset digest;
- an NPZ that needs pickle or fails `utils.canonical_arrays.validate_array_bundle` (22 arrays, order, dtypes; `check_npz`, `:166`);
- duplicate seeds, unless `--allow-duplicate-seeds` (recorded);
- a dataset draw sequence that differs from train + holdout;
- label/grade overlap, or an empty grade block.

**The split and the session filter are required arguments with no defaults** (owner Q1 is open).
- **`--label-start` must be 0** (`:285`). Step 3 reads the holdout file from combined index `len(train_history)` (`full_scoring_worker.py:299-305`), so a later-starting label block would be read at the wrong positions. TODO-E2E-16.
- **The label block is never session-filtered,** for the same reason. Only the grade listing is filtered, and the excluded draws are counted, not deleted.

### 1.2 `e2e_step3_harness.py`, Step 3 in-process

**What it runs:** the production code, unchanged and imported. The production signatures are checked first against a pinned list (`EXPECTED_SIGNATURES`); a mismatch refuses the run.
- **Chunker:** `generate_step3_scoring_jobs.generate_jobs` (`:629`).
  - It runs with cwd = out-dir, so `scoring_chunks/` and `full_scoring_results/` are written there.
  - `load_survivors` is bound with `allow_fallback=False`.
  - `remote_base_path` is the out-dir. The job list is kept as `scoring_jobs.RECORD_ONLY.json`.
  - The effective `prng_type`/`mod` are read back from the job args, so the config override is honoured exactly as in production.
  - The chunk files must reproduce the NPZ seed order.
- **Per chunk** (`run_chunk`, `:443`): one fresh `spawn` process per chunk (`maxtasksperchild=1`, `:692`).
  - It makes the same calls as `full_scoring_worker.main()` (`:577-656`): `load_survivors_with_metadata`, `load_lottery_history` ×2, and `score_survivors(...)` (`:490`).
  - It writes the chunk JSON exactly as the worker does (`json.dump(results, f, indent=2)`, `full_scoring_worker.py:673-674`).
- **Aggregation:** `aggregate_like_launcher` (`:286`) reproduces the `AGGREGATE_EOF` heredoc (`run_step3_full_scoring.sh:275-442`): the global-feature merge, the `chunk_*.json` order, the score sort and the dump. TODO-E2E-3.
- **Phase 6 floor:** `validate_like_launcher` (`:347`), `VALIDATE_EOF` (`:451-489`, ≥46 features).
- **Checks the launcher does not make** (`harness_checks`, `:362`):
  - row count = input rows, and the seed set equal to the input;
  - one feature key set across **all** rows;
  - non-finite and non-numeric value counts;
  - the ranges of `holdout_quality` and `holdout_hits`.

**Guards** (`install_guards`, `:144`), in the parent and in every chunk process:
- sockets blocked and counted; `subprocess.*` and `os.system` blocked and counted (the one git query runs before the guard);
- `cupy`/`cupyx`/`pycuda`/`jax`/`tensorflow` imports blocked; torch is allowed;
- no `.pyc` writes;
- `--device cpu` (default) hides every GPU before torch is imported, and a chunk fails if torch still sees one.

**Fail-closed rules.** This is stricter than production, which takes these fallbacks silently. Each rule can be relaxed only by a named flag:

| Production fallback | Record | Detected by | Flag |
|---|---|---|---|
| batch → sequential path (6 of 18 metadata fields) | Ch4 B6, P0-B | result shape (no `metadata.worker_gpu`, `full_scoring_worker.py:394-401` vs `:465-466`) **and** the log line `:439` | `--allow-sequential-fallback` |
| torch generator → `_cpu_batch_generate` (unreduced) | Ch4 B4 | log line `survivor_scorer.py:577` **and** `pred_max > mod-1` | `--allow-cpu-generate-fallback` |
| `holdout_quality` error written as 0.0 | Beta X-STEP2-WIRING R1 §2 | empty `holdout_features` **and** the log lines (`:424-430`, `:485-491`) | `--allow-holdout-quality-failure` |
| per-seed error record | — | `error` key, log `:494` | `--allow-seed-errors` |
| global features not computed (the launcher prints and continues, `:355-357`) | — | exception | `--allow-global-feature-failure` |
| any failed chunk | — | — | **no flag:** no aggregate is written |

- A failed chunk is written as `FAILED_chunk_NNNN.json`, which never matches the aggregator's glob.
- A failed aggregate is renamed `FAILED_survivors_with_scores*.json`.
- **`--chunks`** (`'0'`, `'0-3'`, `'all'`) runs a subset, and the output is then `survivors_with_scores.PARTIAL.json`.

**Parameters:**
- every value has a declared default and is recorded;
- `--chunk-size` (1000, the launcher's `:70`), `--workers` (1), `--torch-threads` (1), `--device` (cpu), `--gpu-ids`, `--prng-type`/`--mod`, `--batch-size`, `--min-features` (46), `--config`.

**Multi-core:** yes. `--workers N` runs N chunk processes at once.

**`run_record.json` records:** status, partial, args, inputs with sha256 (each verified against the run copy's record), effective `prng_type`/`mod`, per-chunk status/flags/seconds/peak RSS/pid, parent and chunk-max peak RSS, wall and eval seconds, git HEAD plus the sha256 of every production file used, signatures, guards, the output sha256, and the self-test mutant name when one is active.

### 1.3 `e2e_step3_selftest.py`

18 gates, run as `~/venvs/torch/bin/python3 -B scripts/diagnostics/e2e_step3_selftest.py --workers 3`.
- **The fixture** is generated from a fixed RNG seed: 100 rows in the 22-array contract, both `skip_mode` values, 300 train and 120 holdout draws, alternating sessions.
- **The mutants** switch the production fallbacks on through `E2E_STEP3_SELFTEST_MUTANT`. The harness honours it only when set, and writes the name into every record (`_apply_chunk_mutant`, `:415`). For each mutant the test proves three things:
  - it was applied (`mutant` in the chunk record);
  - the mutated function was called (`mutant_calls > 0`);
  - the credited check stopped it (the error text names that check).

  Then the same mutant runs with its `--allow-*` flag and the chunk passes, which shows that the credited check, and no other, was the stop.

## 2. Test results

**Last run:** `data/e2e_step3_selftest_20261123_075557/`, log `data/selftest_run3.log`. **18 of 18 gates pass.** Wall 87.0 s, 155 % CPU. Peak RSS 470 MB: `/usr/bin/time -v`, the largest single process in the tree.

| Gate | Result |
|---|---|
| B1 copy and record (label block = holdout[:60], grade block starts at combined index 360, all output sha256 match, full holdout not copied) | PASS |
| B2 `--grade-sessions midday`: 30 listed, 30 excluded and counted | PASS |
| B3 eleven refusals, nothing created; B3b duplicate seeds allowed by flag and recorded | PASS |
| H1 all 3 chunks, 3 workers: every row through, 91 features, 0 socket and 0 subprocess attempts, torch saw no GPU, 3 distinct pids | PASS (wall 3.3 s, chunk RSS 432 MB) |
| H2 1 worker gives an identical aggregate (timestamps stripped) | PASS |
| H3 **parity:** the launcher's own heredoc (lines 275–442, extracted from the file at test time), run on H1's chunk files, gives a **byte-identical** `survivors_with_scores.json` | PASS |
| H4 `--chunks 0` → `.PARTIAL.json`, 40 rows | PASS |
| H5 five harness refusals (grade block as holdout, tampered input, out-dir in main tree, `--gpu-ids` with cpu, existing out-dir) | PASS |
| H6 Phase 6 floor fails closed (rc 1, `FAILED_` output, no clean output) | PASS |
| M1 `batch_raise`: sequential fallback caught (40 rows plus the log); passes with its flag | PASS |
| M2 `torch_generate_raise`: unreduced predictions caught (40 rows plus the log); passes with its flag | PASS |
| M3 `holdout_quality_raise`: 40 zero labels caught; passes with its flag | PASS |
| M4 `seed_raise`: 40 seed errors caught. With the flags the chunk passes and the aggregate still fails at Phase 6 | PASS |
| M5 `agg_drop_row` and `agg_drop_feature`: `harness_checks` raise | PASS |
| M6 `agg_sort_ascending`: caught by the parity compare | PASS |
| W no change to the watched main-tree, run H and worktree Step 3 paths; file-level `git status` outside `data/` unchanged; no new `__pycache__` | PASS |

- **Run 1 of the self-test** showed W failing on a directory-level `git status` difference created by my own `tee` into `data/`. W was then made file-level, excluding `data/`, the designated output area.
- **W has no deliberate mutant,** because proving it would mean writing into the main tree. TODO-E2E-18.

**Sizing run, synthetic rows at run H's history size.** `data/e2e_step3_sizing_20261123_075401/`: 2,000 synthetic rows, train 14,454, label block 1,807, 2 chunks of 1,000, 2 workers, CPU.

| Measure | Value |
|---|---|
| status | ok, 91 features, 0 non-finite, 0 non-numeric |
| wall | 31.8 s (eval 29.2 s) |
| per chunk | 24.2 s and 29.1 s |
| peak RSS per chunk process | 1.53 GB |
| peak RSS, parent | 119 MB |
| output | 15.1 MB for 2,000 rows |

**Projection for run H** (12,156 rows = 13 chunks; [UNVERIFIED], scaled from the synthetic run):
- with `--workers 4`: about 2–3 min, peak about 6.2 GB (4 × 1.53 + parent);
- with `--workers 2`: about 4 min, about 3.2 GB;
- output about 92 MB, and parent RSS during aggregation under 1.5 GB [UNVERIFIED].

VM101 had 11 GB available at the time.

**Selfplay command check (synthetic only).** The single-episode command of §4 ran on the 2,000-row synthetic harness output:
- rc 0, 6.4 s, 413 MB;
- lightgbm, xgboost and catboost each trained;
- outputs only `telemetry/` and `catboost_info/` in its cwd;
- `find -newer` showed nothing written in the main tree or in the worktree outside that directory.

This proves the hand-off format. It says nothing about run H.

## 3. TODO list (shortcuts and known limits, with anchors)

| TODO | What | Anchor |
|---|---|---|
| E2E-3 | Aggregation is reproduced, not called. Guarded by parity gate H3, which reads the launcher text at test time, so a launcher edit breaks the gate rather than drifting silently | `run_step3_full_scoring.sh:275-442`; `e2e_step3_harness.py:286` |
| E2E-14 | The launcher computes the aggregate's global features with `mod` hardcoded to 1000, and the harness reproduces that. With `--mod` ≠ 1000 the aggregate's global features and the worker's disagree | `run_step3_full_scoring.sh:340`; `e2e_step3_harness.py:286` |
| E2E-15 | The harness is stricter than production. Production still takes all four fallbacks silently; P0-B / F-PARITY remain unimplemented | `TODO_CONSOLIDATED.md:62`, `:161`; `full_scoring_worker.py:438-499` |
| E2E-16 | `--label-start` fixed at 0 and the label block not session-filtered (Step 3's position law) | `full_scoring_worker.py:299-305`; builder `:285` |
| E2E-17 | **Closed for fixtures, 08:30–08:33** (§7): cuda path, the no-device refusal and GPU sharing by 2–3 workers all ran. Still open: run H data, and 4 workers on one GPU at chunk-size 1000 (not run; it would exceed the 3 GB check limit) | `e2e_step3_harness.py` `device_env` |
| E2E-18 | Gate W has no deliberate mutant | selftest `snapshot()` |
| E2E-19 | `selfplay_orchestrator.py` silently uses its default config when `--config` is missing, so the §4 command passes an absolute path and checks it exists | `selfplay_orchestrator.py` (`config_path`, after `:1167`) |
| E2E-20 | `generate_jobs`' `timeout=7200` is passed but has no effect in-process; no per-chunk timeout exists | `e2e_step3_harness.py:629` |
| E2E-12, -13 (map) | Step 3 still regenerates every run H row as constant `java_lcg` at skip 0, and `holdout_quality` uses `skip_min` = 0. The harness measures under these and fixes nothing | map §2.1 rows 7–9 |

## 4. Commands for the owner-approved run on run H (lead to bring; do NOT run before Q1 and the go)

The values come from files read today:
- generation digest: `logs/skipsw_session/receipt_thr3030.json` `artifact_sha256`, equal to `sha256sum` of both NPZ paths;
- dataset digest: the same receipt's `job_identity.dataset_sha256`, equal to `sha256sum ~/skipsw_run_thr3030/daily3.json`.

`<N>` and `<SESSIONS>` are the owner's Q1 answers.

```bash
cd /home/michael/tfm_worktrees/e2e-map
source ~/venvs/torch/bin/activate
mkdir -p data/e2e_runH logs

# 1. run copy (reads run H, writes only under data/)
python3 -B scripts/diagnostics/e2e_build_run_copy.py \
  --npz /home/michael/skipsw_run_thr3030/bidirectional_survivors_binary.npz \
  --generation-digest 3f6a884d6536cbbdd6b23b1e8f8c7b85946d001ca1b1de49e0c2fd9dac0882d0 \
  --train-history /home/michael/skipsw_run_thr3030/train_history.json \
  --holdout-history /home/michael/skipsw_run_thr3030/holdout_history.json \
  --dataset /home/michael/skipsw_run_thr3030/daily3.json \
  --dataset-digest 513648160d356617c22a1e543ae1c9c65f4921ec21718989308b1f70c00768f6 \
  --config /home/michael/skipsw_run_thr3030/optimal_scorer_config.json \
  --config /home/michael/skipsw_run_thr3030/optimal_window_config.json \
  --label-start 0 --label-count <N> --grade-sessions <SESSIONS> \
  --out-dir data/e2e_runH/copy 2>&1 | tee logs/e2e_runH_build.log

# 2. smoke: chunk 0 only (1,000 rows; about 30 s, about 1.6 GB)
nohup python3 -u -B scripts/diagnostics/e2e_step3_harness.py \
  --run-copy data/e2e_runH/copy --config configs/optimal_scorer_config.json \
  --chunks 0 --workers 1 --device cpu \
  --out-dir data/e2e_runH/step3_smoke > logs/e2e_runH_step3_smoke.log 2>&1 &
echo $! > logs/e2e_runH_step3_smoke.pid

# 3. after the smoke record says status ok: all 13 chunks
#    (4 workers: about 2–3 min, peak about 6.2 GB projected; use --workers 2 for about 3.2 GB)
nohup python3 -u -B scripts/diagnostics/e2e_step3_harness.py \
  --run-copy data/e2e_runH/copy --config configs/optimal_scorer_config.json \
  --chunks all --workers 4 --device cpu \
  --out-dir data/e2e_runH/step3_full > logs/e2e_runH_step3_full.log 2>&1 &
echo $! > logs/e2e_runH_step3_full.pid
```

**Read after each run:** `run_record.json` fields `status`, `checks.rows` (must be 1,000 for the smoke, then 12,156), `chunks[*].flags`, `peak_rss_kib`, `wall_seconds`, and `output.sha256`. If any fail-closed rule fires, the run stops there with rc 1. No `--allow-*` flag should be added without the owner.

**Selfplay single episode** (after step 3 is status ok; CPU; synthetic check 6 s and 413 MB at 2,000 rows):

```bash
cd /home/michael/tfm_worktrees/e2e-map
test -f configs/selfplay_config.json || { echo "config missing"; exit 1; }   # TODO-E2E-19
mkdir -p data/e2e_runH/selfplay_ep1 && cd data/e2e_runH/selfplay_ep1
nohup python3 -u -B /home/michael/tfm_worktrees/e2e-map/selfplay_orchestrator.py \
  --config /home/michael/tfm_worktrees/e2e-map/configs/selfplay_config.json \
  --survivors /home/michael/tfm_worktrees/e2e-map/data/e2e_runH/step3_full/survivors_with_scores.json \
  --single-episode --policy-conditioned --no-emit-candidate --n-jobs 4 --project-root . \
  > selfplay_ep1.log 2>&1 &
echo $! > selfplay_ep1.pid
```

**What the selfplay episode yields** (map §2.6):
- per-model R², fitness and top features, fitted to `holdout_hits` computed on the **label block only**;
- `score` and `confidence` are among its inputs (map Q4), so their importance must be reported as such;
- no hit-rate evidence.

## 5. Open questions

1. **Q1, still open (map):** the label/grade split `<N>` and the grade sessions. The builder refuses to run without both.
2. **Workers for the full run:** 4 (about 6.2 GB projected) or 2 (about 3.2 GB)?
3. **Device:** superseded by the owner direction of 08:19. See §6.
4. **A gap between the label and grade blocks** (`--grade-start` > label end) is supported, and defaults to none. Does the owner want one?
5. **Landing:** the three files are new and untracked under `scripts/diagnostics/`, so Gate 22 reds until they land. Should they go to Beta first? This is not a mandatory class: it is not coordinator, worker or kernel code, and not a counted-run launch path.

## 6. Device and cores (owner direction 08:19, `logs/e2e_session/OWNER_DIRECTION_ml_device_20261123.txt`)

**Owner, verbatim:** "if there is no advantage to running the ml on CPU they must be run multi core/thread. If needed we can all locate more cores. The first choice is GPU."

**What did not change:** device and workers stay arguments (`--device cpu|cuda`, `--gpu-ids`, `--workers`, `--torch-threads`). I did not use the graphics card. The only GPU contact was `nvidia-smi --query-gpu=name,memory.total,memory.used`, which reported an RTX 3080 Ti with 12,288 MiB and 264 MiB used.

**Added 08:20, fail-closed:** with `--device cuda` a chunk now fails if torch sees no CUDA device. Without this, a cuda run would score on CPU without saying so (`survivor_scorer.py:117`). The check also records `cuda_device_name` per chunk.
- Superseded: it was run at 08:30 and fails closed as designed (§7, G0).
- The CPU self-test still passes 18/18 (`data/e2e_step3_selftest_20261123_082034/`, 85 s, 470 MB).

### 6.1 Where a chunk's time goes (measured, CPU)

Source: the synthetic sizing run's chunk log (1,000 rows, train 14,454, label block 1,807, 1 torch thread), from the log timestamps in `data/e2e_step3_sizing_20261123_075401/h/logs/chunk_0000.log`:

| Phase | Time | Runs on | GPU can help? |
|---|---|---|---|
| process start, imports | ≈ 3 s (24.2 s chunk − 21.0 s scoring) | CPU | no |
| `holdout_hits`: Python generator loop, about 16k steps per seed (`full_scoring_worker.py:274-310`) | 6.1 s | CPU always (`prng_registry` CPU reference) | **no** |
| batched features (`extract_ml_features_batch`) | 9.3 s | torch on `self.device`, **plus** per-seed battery features in numpy (`survivor_scorer.py:735-760`) | **partly**: only the torch part. The split inside the 9.3 s was not measured |
| per-seed `holdout_quality` loop: `extract_ml_features` on the label block, one seed at a time (`full_scoring_worker.py:405-430`) | 5.5 s | torch on `self.device`, small tensors per seed | doubtful: per-seed kernel launches [UNVERIFIED] |
| total scoring | 21.0 s | | |

**So at least 11.6 s of each 21 s chunk is CPU-only whatever `--device` says.** A GPU can shorten only part of the remaining 9.3 s, plus perhaps some of the 5.5 s.
- The GPU gain on Step 3 is therefore bounded and unmeasured [UNVERIFIED].
- Whichever device is used, the CPU-bound share has to be spread across cores. Each chunk process is single-threaded Python, so the parallelism comes from `--workers`.
- In cuda mode, the workers share one GPU. Each chunk's prediction tensor is about 116 MB (1,000 × 14,454 int64), against 12 GB of VRAM. Per-process CUDA context size is [UNVERIFIED].

### 6.2 Commands (lead brings them to the owner; run only after Q1 and the go)

All commands run from `/home/michael/tfm_worktrees/e2e-map` with `source ~/venvs/torch/bin/activate`, on the run copy of §4 step 1.

**GPU smoke, one chunk** (first choice per the owner). Expect at most the CPU smoke's ~1.6 GB RSS plus a CUDA context; first CUDA use [UNVERIFIED].
```bash
nohup python3 -u -B scripts/diagnostics/e2e_step3_harness.py \
  --run-copy data/e2e_runH/copy --config configs/optimal_scorer_config.json \
  --chunks 0 --workers 1 --device cuda --gpu-ids 0 \
  --out-dir data/e2e_runH/step3_smoke_gpu > logs/e2e_runH_step3_smoke_gpu.log 2>&1 &
echo $! > logs/e2e_runH_step3_smoke_gpu.pid
```

**CPU smoke, the same chunk,** for comparison. This is §4 step 2 with `--out-dir data/e2e_runH/step3_smoke`. It takes about 30 s and about 1.6 GB (measured on synthetic rows).

**Compare the two smokes.** Reporting only, no pass threshold declared: `holdout_hits` is computed on CPU in both modes and must be equal; torch float features may differ slightly between devices.
```bash
python3 -I -c "
import json
a={r['seed']:r for r in json.load(open('data/e2e_runH/step3_smoke/survivors_with_scores.PARTIAL.json'))}
b={r['seed']:r for r in json.load(open('data/e2e_runH/step3_smoke_gpu/survivors_with_scores.PARTIAL.json'))}
assert a.keys()==b.keys(), 'seed sets differ'
print('holdout_hits equal:', all(a[s]['holdout_hits']==b[s]['holdout_hits'] for s in a))
print('holdout_quality max|d|:', max(abs(a[s]['holdout_quality']-b[s]['holdout_quality']) for s in a))
d={k:max(abs(a[s]['features'][k]-b[s]['features'][k]) for s in a) for k in a[next(iter(a))]['features']}
print('features with |d|>0:', {k:v for k,v in sorted(d.items(), key=lambda x:-x[1]) if v>0})
for n in ('step3_smoke','step3_smoke_gpu'):
    r=json.load(open(f'data/e2e_runH/{n}/run_record.json')); c=r['chunks'][0]
    print(n, r['status'], 'chunk s', c['seconds'], 'RSS KiB', c['peak_rss_kib'], c.get('cuda_device_name'))
"
```

**Full run, GPU plus multi-core** (if the GPU smoke is ok and the comparison is acceptable to the owner): four chunk processes share GPU 0 and use 4 cores for the CPU-bound share.
```bash
nohup python3 -u -B scripts/diagnostics/e2e_step3_harness.py \
  --run-copy data/e2e_runH/copy --config configs/optimal_scorer_config.json \
  --chunks all --workers 4 --device cuda --gpu-ids 0 \
  --out-dir data/e2e_runH/step3_full > logs/e2e_runH_step3_full.log 2>&1 &
echo $! > logs/e2e_runH_step3_full.pid
```

**Full run, CPU multi-core** (the fallback if the GPU smoke fails or gains nothing). This is §4 step 3: `--workers 4 --device cpu`.
- Projected about 2–3 min and about 6.2 GB peak (4 × 1.53 GB measured per process, plus the parent).
- `--workers 6` would give about 9.3 GB projected, against 11 GB available and zero swap (CLAUDE.md rule 9). I do not recommend it without more memory.
- 13 chunks at 4 workers is 4 rounds; at 6 workers it is 3. The saving is about one chunk-time, about 30 s.

### 6.3 Recommendation for run H

**Updated by §7.1 with measured GPU numbers (08:33). §7.1 governs; this section is kept as written at 08:20.**

1. **Smoke chunk 0 on GPU and on CPU, then compare.** The GPU path is the owner's first choice and has never run; the CPU path is measured. The pair costs about 1 min and gives what is missing: the GPU's real gain on the 9.3 s batch share, its memory, and whether device changes any value.
2. **For the full run, use `--device cuda --gpu-ids 0 --workers 4`** if the GPU smoke is status ok, its chunk time is at or below the CPU smoke's, and the owner accepts the comparison.
3. **Otherwise use `--device cpu --workers 4`.**

**Why both choices use 4 workers:**
- On CPU, at least 11.6 of every 21 s of scoring is single-threaded Python that no GPU touches (§6.1).
- Four processes keep memory near 6.2 GB on a 15.5 GB box with no swap.
- Either way the run is minutes, not hours: 12,156 rows, about 2–3 min projected on CPU.

**More cores (the owner's offer):** they would shorten the CPU-bound share in proportion to the cores added. Each extra worker also needs about 1.5 GB of RAM. At this size the whole run is already a few minutes, so more cores matter more for Step 5 and for selfplay at scale than for this Step 3.

**Already multi-core:**
- The harness uses `--workers`, one process per chunk.
- The selfplay command uses `--n-jobs 4`.
- Step 5's tree models and NN use the GPU (map §2.3).

## 7. GPU-path checks on synthetic fixtures (owner go 08:29, `logs/e2e_session/OWNER_GO_gpu_fixture_checks_20261123.txt`)

**Result: the GPU path is tested, and it works on synthetic fixtures.** No run H data, no rigs.

**Card state:** before every check, `nvidia-smi --query-compute-apps` showed only `sunshine` (PID 2421, 248 MiB; allowlisted). Card baseline was 264 MiB used. After the last check it was back to `sunshine` only and 264 MiB.

**How it was measured:**
- **Card memory:** `nvidia-smi --query-gpu=memory.used -lms 200`, sampled over each run. The sampler was started and stopped by recorded PID.
- **Host RSS:** `/usr/bin/time -v`, plus the harness's per-chunk `ru_maxrss`.
- **Location:** `data/e2e_step3_gpu_checks_20261123_082953/`.

| Check | Fixture | Command difference | rc / status | Wall | Card peak (above baseline) | Peak RSS |
|---|---|---|---|---|---|---|
| G0 no-device refusal | 100 rows | `--device cuda --gpu-ids 7 --chunks 0` | rc 1, chunk raised `--device cuda but torch sees no CUDA device (CUDA_VISIBLE_DEVICES='7')` | ≈ 4 s | 0 MiB | 420 MB |
| G1 | 100 rows, 3 chunks × 40 | `--device cuda --gpu-ids 0 --workers 1` | rc 0, ok, every chunk reports `NVIDIA GeForce RTX 3080 Ti` | 9.9 s | 295 MiB | 851 MB |
| G2 | same | `--workers 3` (3 processes share GPU 0) | rc 0, ok | 3.7 s | 885 MiB | 851 MB |
| G3 | 2,000 rows, train 14,454, label 1,807; chunk 0 of 1,000 | `--chunk-size 1000 --chunks 0 --workers 1` | rc 0, ok | 21.3 s (chunk scoring 16.7 s) | **1,389 MiB** | 965 MB |
| G4 | same, all 2,000 rows | `--chunk-size 500 --workers 2` (4 chunks, 2 share GPU 0) | rc 0, ok | 29.5 s (chunks 12.5–14.0 s each) | 1,694 MiB | 905 MB |

**Every check stayed under the 3 GB card-memory limit.** The largest was 1,958 MiB total, sunshine included. All fail-closed flags were zero: no CPU-generate fallback, no sequential fallback, no `holdout_quality` failure, no seed error.

**Phase split, one 1,000-row chunk at run H's history size.** Same synthetic fixture, log timestamps:

| Phase | CPU (§6.1) | GPU (G3) |
|---|---|---|
| `holdout_hits` (Python loop) | 6.1 s | 6.0 s |
| batched features | 9.3 s | **4.7 s** |
| per-seed `holdout_quality` loop | 5.5 s | 6.0 s |
| scoring total | 21.0 s | **16.7 s** (−20 %) |
| host RSS per chunk process | 1.53 GB | 0.97 GB |

**Values, GPU against CPU, on the same synthetic inputs:**
- `holdout_hits`: **identical** on every row (100 of 100, 1,000 of 1,000 and 2,000 of 2,000).
- **Feature key sets identical.**
- 20 of 91 features (100-row fixture) and 26 of 91 (2,000-row fixture) differ at float32 rounding level: largest relative difference 2.1e-7, largest absolute 6.1e-5 (`pred_std`, `residual_std`). `holdout_quality` max |d| is 2.4e-9; `score` max |d| is 1.5e-8.
- **GPU chunk size 500 against 1,000: 0 of 91 features differ.** On the GPU, the chunking does not change any value.
- **Consequence:** because `score` can differ by about 1e-8, two rows with near-equal scores can swap places in the aggregate's sort (`run_step3_full_scoring.sh` score sort, reproduced at `e2e_step3_harness.py:286`). The rows and their values are the same; their order is not guaranteed to be byte-identical between devices. Not a defect, recorded. A byte-level comparison across devices must match rows by seed, as the §6.2 compare does.

### 7.1 Recommendation for run H, updated with these numbers

**GPU first, per the owner, with several workers sharing the card.**
- Each GPU chunk process uses about 1.4 GB of card memory at chunk-size 1,000, measured. Four processes would project to about 5.8 GB of the 12 GB card, sunshine included. Host RSS would project to about 1 GB each, so about 4 GB in total.
- The GPU saves about 4 s of each 21 s chunk (−20 %, measured). The remaining 12 s are CPU-only Python, which is why the parallelism still has to come from `--workers`.

1. **Smoke, run H chunk 0 on GPU:** the §6.2 command (`--chunks 0 --workers 1 --device cuda --gpu-ids 0`). Measured equivalent: about 21 s, 1.4 GB card, 1 GB RSS.
2. **Optional CPU smoke of the same chunk**, and the §6.2 compare (by seed): `holdout_hits` must be equal, and the features are expected to differ only at float32 rounding, as measured here.
3. **Full run:** `--device cuda --gpu-ids 0 --workers 4 --chunk-size 1000`.
   - Projected about 1.5–2 min [UNVERIFIED: 4 GPU-sharing workers were not run at chunk-size 1000, because that would exceed this morning's 3 GB check limit].
   - Projected card about 5.8 GB, host about 4 GB.
   - If the owner wants the card kept under 3 GB on the real run too, use `--chunk-size 500 --workers 2`, as measured in G4: 1.7 GB card, about 13 s per 500-row chunk. For 12,156 rows that is 25 chunks, about 2.7 min [projection].
4. **CPU multi-core** (`--device cpu --workers 4`, about 2–3 min, about 6.2 GB host) remains the fallback if the GPU smoke fails.

**What this does not tell us:** nothing about run H's values or about hit rate. It shows only that the GPU path runs, fails closed, stays within memory, and matches CPU to float32 rounding on made-up rows.

## 8. Run H: the run copy and the Step 3 smoke on chunk 0 (owner go 08:57, ruling Q1 08:08)

**Authority:** `logs/e2e_session/OWNER_GO_step3_smoke_runH_20261123.txt` and `logs/e2e_session/OWNER_RULING_e2e_Q1_20261123.txt` (main tree).

**Scope run, exactly:** the run copy, then chunk 0 (1,000 of 12,156 rows), once on GPU and then once on CPU. **The other 12 chunks, Step 4, Step 5 and selfplay were not run.**

### 8.1 Run copy, `data/e2e_runH/copy/` (log `logs/e2e_runH_build.log`)

- **NPZ:** the source `~/skipsw_run_thr3030/bidirectional_survivors_binary.npz` resolves to the generation file. Its sha256 is `3f6a884d6536cbbdd6b23b1e8f8c7b85946d001ca1b1de49e0c2fd9dac0882d0`, which equals the `artifact_sha256` in `logs/skipsw_session/receipt_thr3030.json`. It was checked before the copy, on the copy, and again after copying. 12,156 rows; passes the 22-array contract; seeds unique.
- **Dataset:** sha256 `513648160d35…` equals the receipt's `dataset_sha256`, and its draw sequence equals train (14,454) + holdout (3,614).
- **Declared in the run record:** `label_start 0`, `label_count 1807`, `grade_start 1807`, `grade_sessions ['midday']`, `gap_between_label_and_grade 0`, label block not session-filtered.

| Block | Holdout index | Combined index | Dates | Midday / evening | Beta figure |
|---|---|---|---|---|---|
| label (to Step 3) | [0, 1807) | [14454, 16261) | 2021-03-17 → 2023-09-06 | **904 / 903** | 904 / 903 ✓ |
| grade (sealed) | [1807, 3614) | [16261, 18068) | 2023-09-07 → 2026-02-26 | **904 / 903**, 904 midday listed for grading, 903 evening counted as excluded | 904 / 903 ✓ |

**The cut lands between complete dates.**
- No date appears in both blocks: the intersection of the label-block and grade-block date sets is empty.
- The last label record is 2023-09-06 midday; the first grade record is 2023-09-07 evening.
- Each date around the cut holds both sessions.

**Two dataset facts found while checking.** Both are reported only; nothing was changed, and whether they are already recorded was not searched [UNVERIFIED]:
1. **Within every date the dataset stores evening before midday** (8,514 of 8,514 same-date pairs). So "contiguous" in record order puts a date's evening draw before its midday draw. My cut is unaffected because it falls between dates.
2. **Step 1's 80/20 split** (`window_optimizer.py:1170-1175`, by count) **cuts 2021-03-17 in two.** Its evening draw is the last train record (combined 14,453). Its midday draw is the first label record (combined 14,454). The label block's first date is therefore shared with train. This is not the Q1 cut, which lies between 2023-09-06 and 2023-09-07.

### 8.2 Step 3 smoke, chunk 0, GPU then CPU

Both runs used `nohup`, one after the other. Before each, `nvidia-smi` showed only `sunshine` (248 MiB; card baseline 264 MiB); after the CPU run it was back to `sunshine` only.

**Commands, as run:**
- GPU: `--chunks 0 --workers 1 --device cuda --gpu-ids 0`, `--out-dir data/e2e_runH/step3_smoke_gpu`.
- CPU: `--chunks 0 --workers 1 --device cpu`, `--out-dir data/e2e_runH/step3_smoke`.
- Both: `--run-copy data/e2e_runH/copy --config configs/optimal_scorer_config.json`.
- Logs: `logs/e2e_runH_step3_smoke_gpu.log` and `logs/e2e_runH_step3_smoke.log` (worktree). Card memory and time records: `data/e2e_runH/smoke_measure/`.

| Measure | GPU (`--device cuda`) | CPU (`--device cpu`) |
|---|---|---|
| rc / status | 0 / ok (PARTIAL: 1 of 13 chunks, 1,000 rows) | 0 / ok (PARTIAL) |
| wall, whole harness (`/usr/bin/time`) | 22.9 s | 26.8 s |
| chunk process time | 18.8 s | 22.8 s |
| scoring (worker log) | 16.0 s: holdout_hits 5.9, batch 4.2, holdout_quality loop 5.9 | 20.0 s: holdout_hits 5.9, batch 8.9, holdout_quality loop 5.3 |
| card memory, peak above baseline | **1,389 MiB** (1,653 total incl. sunshine) | 0 (no CUDA device visible) |
| peak RSS, chunk process | 966 MB | 1,536 MB |
| peak RSS, parent | 111 MB | 114 MB |
| device reported | `NVIDIA GeForce RTX 3080 Ti` | none (torch saw no GPU, as required) |
| fail-closed flags (fallbacks, seed errors, zero labels) | all 0 | all 0 |
| guards (socket / subprocess / blocked imports) | 0 / 0 / none | 0 / 0 / none |
| features | 91 (14 global), 0 non-finite, 0 non-numeric | same |
| inputs | NPZ `3f6a884d…`, train `0fe05436…`, label block `703fe276…` | same |
| effective prng_type / mod | java_lcg / 1000 | same |

**Comparison, matched by seed, 1,000 rows:**
- seed sets equal; feature key sets equal on every row; `holdout_features` key sets equal on every row;
- **`holdout_hits` equal on all 1,000 rows;**
- `holdout_quality` max |d| = 2.5e-9; `score` max |d| = 1.5e-8;
- **25 of 91 features differ, all at float32 rounding.** The largest absolute difference is `residual_std` at 6.1e-5. The largest relative difference is 5.0e-6, on `residue_8_kl_divergence`, a near-zero value.
- This matches the synthetic-fixture result (§7). The two output files are not byte-identical: aggregate sha256 `d57b1d3f…` on CPU, `03f6f2cc…` on GPU. This is expected from float rounding and from the score sort (§7).

**Facts about the chunk 0 values, no verdict:**
- `holdout_hits` over the 1,807-draw label block: min 0, max 0.00387 (7 of 1,807), mean 0.0010, 8 distinct values.
- `holdout_quality`: 0.2324–0.2430, 1,000 distinct values.
- These are per-vehicle labels for Step 5, not pool hit rates. They say nothing about `k/1000`.

**What this smoke does and does not tell us:**
- **It shows** that Step 3 runs end to end on real run H rows in isolation: on the GPU and on CPU, with the label block only, within memory, and with no fallback taken. Device choice changes values only at float32 rounding.
- **It does not show** anything about the other 11,156 rows, about Steps 4–6, or about hit rate.
- The recorded rows still apply: every row regenerated as constant `java_lcg` at skip 0; `holdout_quality` read at `skip_min` = 0; forward/reverse matches not merged (map §2.1 rows 7–10).

**Full-run projection, updated with the run H numbers:**
- GPU, `--workers 4 --chunk-size 1000`: about 1.5–2 min, about 5.8 GB of card memory, about 4 GB host [projection: 4 workers sharing the GPU not yet measured].
- CPU, `--workers 4`: about 2–3 min, about 6.4 GB host.

### 8.3 The owner's question in the 08:57 go: "the earlier bug where we were only using a small section of the draw data"

These are facts from this run's records. I have not searched which specific bug the owner means, so this is not a verdict on whether it is resolved:
- **Step 3 (this smoke)** computed every feature against the whole train history, 14,454 draws. Prediction tensor 1,000 × 14,454 (`survivor_scorer.py:547-582`; train sha256 `0fe05436…` in the run record).
- **Step 3's labels** (`holdout_hits`, `holdout_quality`) used the whole 1,807-draw label block.
- **Step 1's sieve for run H** used a 12-draw midday window at offset 25 (`optimal_window_config.json`: `window_size 12`, `offset 25`, `sessions ['midday']`). This is unchanged by anything here. The recorded item "the analysis window reaches 150 of 18,068 records" (`docs/PITFALL_STATE_RECOVERY_YARDSTICK.md` §5) concerns Step 1's window, not Step 3.

## 9. Full Step 3 on the run H copy, all 13 chunks (owner go 10:30)

Owner go: `logs/e2e_session/OWNER_GO_step3_full_runH_20261123.txt` (main tree). Step 4, Step 5, the vote, selfplay and grading were not started. This is a NON-CERTIFYING DIAGNOSTIC.

### 9.1 Pre-checks and launch

- At 10:31, before launch, `nvidia-smi` showed only `sunshine` (248 MiB). The card had 11,775 MiB free (264 used of 12,288), against the go's floor of 8 GB. The host had 11.9 GB available with no swap. Expected peak was about 5.8 GB of card memory and about 4 GB of host memory (§8.2 projection).
- **Command, run with nohup** from the worktree with the torch venv, wrapped in `/usr/bin/time -v`:
  `scripts/diagnostics/e2e_step3_harness.py --run-copy data/e2e_runH/copy --config configs/optimal_scorer_config.json --chunks all --workers 4 --device cuda --gpu-ids 0 --out-dir data/e2e_runH/step3_full`
- Log: `logs/e2e_runH_step3_full.log`. PGID: `logs/e2e_runH_step3_full.pgid` (40870). Time record: `data/e2e_runH/full_measure/full.time`.
- **Watchdog:** `data/e2e_runH/full_measure/watch.sh` (sha256 `b310233b…`). Once a second it sampled total card memory and the run's process-group RSS. It would have stopped the group if card memory passed 8,192 MiB or a `FAILED_chunk` file appeared. **It never fired.**
- **Launch slip, recorded:** a `&&` chain backgrounded the whole command, so the first watchdog was given the wrapper shell's group (40867) instead of the harness's group (40870). I replaced it 21 s in (10:32:10). In that window the card reading is valid, because the card is sampled globally (`watch_wrongpgid.log`), but the RSS reading and the kill switch did not cover the harness. Only one harness instance ran.

### 9.2 Result (plain facts, no verdict)

| Measure | Value |
|---|---|
| exit / status | 0 / `ok`, `partial: false`, 13 of 13 chunks `ok`, no `FAILED_chunk` file |
| wall time | 97.6 s harness (`run_record.json`); 1:39.1 under `/usr/bin/time` |
| chunk process time | 7.6 s (chunk 12, 156 rows) to 33.0 s; 12 full chunks 21.8–33.0 s |
| peak card memory | **5,820 MiB total**, i.e. 5,556 MiB above the 264 MiB baseline (at about 10:31:5x, in the first 20 s). After 10:32:10 the peak was 4,726 MiB. Never near 8 GB |
| peak RSS | single process: 965,848 KiB (largest chunk process; `/usr/bin/time` and the run record agree). Parent: 251,900 KiB. Whole process group: 3,866,688 KiB (≈3.9 GB), sampled from 10:32:10 only |
| device | `NVIDIA GeForce RTX 3080 Ti` in all 13 chunks; torch saw CUDA in all 13 |
| fail-closed flags, summed over chunks | seed_errors 0, sequential_fallback 0/0, cpu_generate 0, unreduced_pred 0, holdout_quality empty/failed 0/0, seed_failed 0 |
| guards | socket 0, subprocess 0, blocked imports 0 (chunks and parent) |
| after the run | `nvidia-smi` showed `sunshine` only (248 MiB) |

### 9.3 Output checks

- **Rows:** 12,156 in (NPZ, 12,156 unique seeds) and 12,156 out (12,156 unique seeds). The seed sets are equal and none was dropped. The aggregate read 13 files: 12 × 1,000 plus 156.
- **Features:** 91 on every row, one key set across all rows, and 0 non-finite or non-numeric values. Phase 6 floor ≥46 met. `holdout_features` has 77 entries on every row.
- **Label fields:** `holdout_hits` and `holdout_quality` are top-level fields on all 12,156 rows, with 0 None and 0 non-finite values.
- **Inputs used** (run record): the NPZ `3f6a884d…` (equal to the generation digest), `train_history.json` (14,454 draws) and the holdout `label_block_history.json` (1,807 draws). The grade block was not read.
- **Output:** `data/e2e_runH/step3_full/survivors_with_scores.json`, 92,213,036 bytes, sha256 `37c8b14600462069cc70ea9aebbcab38fb08f17283f87661d20048a43a7b187b`. This equals the digest in `run_record.json` `output`.
- **Run record:** `data/e2e_runH/step3_full/run_record.json`. It includes argv and args, `effective` (java_lcg, mod 1000, 13/13 chunks, 12,156/12,156 rows), input and config sha256, the run-copy record sha256 `8acdf26f…`, code identity (HEAD `41ad509`, untracked harness files with their sha256), checked `inspect.signature` lists, per-chunk status, timing, RSS, flags, guards and output sha256, aggregate per-file counts, `aggregate_error: null`, `selftest_mutant: null`, host and python. No field is missing against the record format the self-test checks.

### 9.4 Spread of the labels, 12,156 rows

| Field | min | median | max | distinct values | mean |
|---|---|---|---|---|---|
| `holdout_hits` | 0.0 | 0.0011068 (2 of 1,807) | 0.0049806 (9 of 1,807) | 10 | 0.000998 |
| `holdout_quality` | 0.230463 | 0.237072 | 0.243863 | 12,156 | 0.237113 |

- `holdout_hits` as counts out of 1,807: 0 → 2,066 rows; 1 → 3,560; 2 → 3,197; 3 → 1,957; 4 → 926; 5 → 333; 6 → 89; 7 → 23; 8 → 4; 9 → 1.
- These are per-vehicle Step 5 training labels on the label block. They are not pool hit rates and say nothing about `k/1000` on the grade block.
- The recorded limits of §8.2 still apply: rows are regenerated as constant `java_lcg` at skip 0, `holdout_quality` is read at `skip_min` = 0, and forward and reverse matches are not merged.

### 9.5 Not done

Step 4, Step 5, the vote, selfplay and every grade were not started, by the go. Nothing was committed. Nothing was written in the main tree or in the run H tree.

## 10. Step 4 and Step 5 on the run H copy (TASK_STEP45, owner go 10:40)

Task file: `logs/e2e_session/TASK_STEP45.txt` (main tree). Go: `OWNER_RULING_step5_go_ch14_beta_holds_20261123.txt`, "Please proceed with step 5". NON-CERTIFYING DIAGNOSTIC. **The scores below are cross-validation R² of a model predicting the Step 3 label `holdout_quality`. They are training scores, not hit rates, and they say nothing about `k/1000`.** The vote, selfplay and every grade were not run. The grade block was not opened.

### 10.0 Null results first

- **No family's CV score is meaningfully above zero.** Mean 20-fold CV R²: CatBoost +0.00187, NN +0.00036, XGBoost −0.00467, LightGBM −0.00563. On the internal 20 % validation split, the final fits are all slightly negative: CatBoost −0.00017, NN −0.0016, XGBoost −0.0044, LightGBM −0.0052. Plain fact, no verdict: on these 12,156 rows, none of the four families predicts `holdout_quality` much better than its mean.
- **Step 5's signal gate reported `weak`, not degenerate.** `holdout_quality` has 11,984 unique values, variance 3.53e-6, std 0.00188 and mean 0.2371. Exit 2 did not occur.
- No family failed and no stop rule fired. Memory available never fell below 9,840 MiB.

### 10.1 Prior record for this section

- **Searches run:** `grep` of `docs/CHAPTER_6_AUDIT.md` for `permutation|unseeded|seed`, for the B-list, and for `sidecar` with `optuna|tree`. Map §2.2–§2.3 and §3, re-read. Not searched: the other docs families listed in CLAUDE.md, the host, ser8, the private repo.
- **The unseeded 80/20 split is ALREADY RECORDED** (`CHAPTER_6_AUDIT.md:140`). B1, B3, B4, B7 and B8 are ALREADY RECORDED (Ch6 audit, map §2.3 row 7).
- Labelled NEW below, by reading only: the tree sidecars carry no `optuna` block; the NN worker's normalisation lines are dropped from the Step 5 log; a failed NN fold becomes r2 = −999.0 and is averaged in without an error; fold NPZ paths are relative to cwd; `provenance.compare_models_used` is true on the NN sidecar of a single-model run.

### 10.2 What was run and how

- **Pre-run record:** `data/e2e_runH/step45/run_record_step45.json`, sha256 `10aa6bac…`. Written before any run and unchanged since. It holds:
  - the winner rule verbatim, and how each part is checked;
  - the inputs bound by sha256: the Step 3 output `37c8b146…`, train history `0fe05436…`, the label block, and the grade block's digest `5fc3c94a…` from the run-copy record, with `opened: false`;
  - every declared parameter.
- **Results:** `data/e2e_runH/step45/results_step45.json`, sha256 `14893ca6…`. Per-family checks: `check_<family>.json`.
- **Step 4:** run directly in an empty `data/e2e_runH/step45/step4/`, on CPU (`CUDA_VISIBLE_DEVICES=""`):
  `adaptive_meta_optimizer.py --config <worktree>/adaptive_meta_optimizer_config.json --window-results optimal_window_config.json --lottery-data train_history.json --mode full --apply`.
  - Exit 0, 0.72 s, 87 MB RSS.
  - Output: `optimization_results/meta_optimization_results.json` (`cdfaf8fb…`) and `reinforcement_engine_config.json` (`2ef10879…`). It gives 476 survivors (range 250–952), architecture [256, 128, 64] and 150 epochs, at confidence 24.5 %. The window-optimizer source confidence is 0.00 %, so its values are the constants (map §2.2 row 4, ruled TB Q4).
  - **Unconsumed (TODO-E2E-5).** No family opened it: 0 opens of `reinforcement_engine_config|meta_optimization|optimization_results` in any strace, and 0 references in `meta_prediction_optimizer_anti_overfit.py` or `train_single_trial.py`. This is B7.
- **Step 5:** four direct calls, one family at a time, each under nohup. Each ran in its own new directory `data/e2e_runH/step45/<family>/`, so `models/`, `optuna_studies/` and `diagnostics_outputs/` started empty. No `--compare-models`. Common arguments: `--trials 20 --k-folds 20 --enable-diagnostics`. 20 folds is the manifest value (`agent_manifests/reinforcement.json` `default_params.k_folds`).
  - **NN:** also `--normalize-features --use-leaky-relu`, which compare-models Option A forces (`:240-245`).
  - **LightGBM:** `S88_FORCE_LGBM_CPU=1`, as the compare-models parent sets it (`:264-265`).
  - Not passed: `--enable-vmap`. Compare-models children never receive it (B2), and it is inert without persistent workers (`:2051-2057`).
- **Before each family,** `nvidia-smi` showed only `sunshine`. Before XGBoost, another agent's test held the card, so I waited until it left: PID 101574, `tests/test_s172_attempt6_remediation.py`, cwd `~/tfm_worktrees/anchor-brief2`. Waiting ran from 11:40 to 11:41.
- **Additions, all disclosed, with no module code changed:**
  1. **`scripts/diagnostics/e2e_step5_shim.py`** (`1a7534d3…`) runs the module unchanged as `__main__`.
     - It seeds numpy once (declared split seed 42) so all four families get the same 80/20 split. The module's `np.random.permutation` at `:1514` is unseeded. The KFold at `:2034` is seeded but draws from that split.
     - It records every child process call (argv, cwd, the resolved script and its sha256, rc, stdout and stderr) without changing what the module receives.
     - **TODO-E2E-21:** seed the split in the module as a declared parameter. **TODO-E2E-22:** have the module log and fail on child stderr itself.
  2. **`data/e2e_runH/step45/run_one.sh`** (`50618834…`):
     - copies the worktree's `train_single_trial.py` into the run dir, where the final NN call runs it by relative path from cwd (`:2563`, `:2602`);
     - passes an absolute `--output-dir`, because the fold child runs with `cwd = module dir` (`:1949-1953`) and Python 3.10 `mkdtemp` returns a relative path;
     - redirects `TMPDIR`, `XDG_CACHE_HOME`, `CUDA_CACHE_PATH` and `MPLCONFIGDIR` into the run dir;
     - sets `CUDA_VISIBLE_DEVICES=0`.
  3. A watchdog (`watch5.sh`) stops the group if MemAvailable falls below 2 GB. It never fired. The tree families also ran under `strace -f -e trace=%file`.
- **Self-test on a synthetic 300 × 91 fixture first** (`data/e2e_runH/step45/selftest/`):
  - two LightGBM calls gave the same permutation;
  - the NN at 2 trials × 2 folds made 5 child calls, all rc 0, and the CAT-B and S121 lines were captured;
  - the checkpoint had the four normalisation keys;
  - a 1-trial call takes the single-shot path, not Optuna (`:2726`), so the fold path was tested with 2 trials.

### 10.3 Per-family results

| | neural_net | lightgbm | xgboost | catboost |
|---|---|---|---|---|
| exit status | 0 | 0 | 0 | 0 |
| **CV score** (mean 20-fold R², best of 20) | **+0.000364** (trial 11) | **−0.005625** (trial 8) | **−0.004667** (trial 13) | **+0.001866** (trial 2) |
| worst trial CV | −0.01415 | −0.1866 | −0.2787 | −0.00106 |
| trials COMPLETE | 20 | 20 | 20 | 20 |
| final fit R², internal 20 % val | −0.0016 | −0.0052 | −0.0044 | −0.00017 |
| wall time | 29:35 | 17:41 | 5:36 | 9:20 |
| CPU (`time`) | 122 % | 662 % (multi-threaded) | 138 % | 112 % |
| peak card memory, whole card incl. 264 MiB baseline | 2,672 MiB | 772 MiB | 2,682 MiB | **11,465 MiB** (CatBoost reserves the card; accepted for this run) |
| peak RSS, largest single process | 984 MB | 721 MB | 795 MB | 822 MB |
| peak RSS, process group | 1.13 GB | 0.73 GB | 0.80 GB | 0.83 GB |
| device actually used | children `cuda:0 (NVIDIA GeForce RTX 3080 Ti)` (all 401) | CPU | `cuda:0` (log) | `task_type GPU` (checkpoint) |
| child calls / failures | 401 (400 folds + 1 final) / 0 | 1 (`lscpu`) / 0 | 0 / 0 | 0 / 0 |
| split, first-permutation sha256 | `bb7f87a2…` | `bb7f87a2…` | `bb7f87a2…` | `bb7f87a2…` |
| checkpoint | `best_model.pth` `62e2d15f…`, 89 features | `best_model.txt` `6aae5022…`, 89 features, 148 trees | `best_model.json` `03cdde2e…`, 89 features, 196 rounds | `best_model.cbm` `c2abe721…`, 89 features, 6 trees |
| sidecar `best_model.meta.json` | see `check_neural_net.json` | see `check_lightgbm.json` | see `check_xgboost.json` | `55e6bbdc…` |

- **CV score source:** the NN sidecar's `optuna.best_value`, which equals the study db. **The tree sidecars have no `optuna` block** (NEW by reading: they are written by `MultiModelTrainer.save_model`, `:1216`, while only the NN path writes `optuna`, `:2908`). For the trees the score is the study db's best COMPLETE value, cross-checked against the log's `Best R²` line.
- **Coverage gaps, recorded:**
  - NN: the watchdog was not sampling for the first 23 s (10:50:28–10:50:51); it had been given a PID that had already exited.
  - LightGBM: process-group RSS for the first 12 s read 0, because `ps -g` selects by session, not by process group. It was fixed and restarted; the card samples in that window are valid.
  - `watch5.sh` therefore differs from the pre-run record's hash (`37af0a61…` → `5d9f51f5…`). The only change is that RSS line.
  - Card memory is the whole card. Before each start only `sunshine` was present, but another agent's process could have joined during a run without being separated out.

### 10.4 Winner by the declared rule

- **Eligible: all four.** Each had exit 0, no child failure, a fresh sidecar in a new directory, a checkpoint that loads on CPU in a separate process with feature count 89 equal to the sidecar's, and a finite CV score.
- **Same folds:** one permutation sha256 `bb7f87a2…`, k = 20, KFold `random_state` 42, 12,156 rows each. **Same trial budget:** 20 COMPLETE each.
- **Ranking:** catboost +0.001866 > neural_net +0.000364 > xgboost −0.004667 > lightgbm −0.005625. No tie, so the tie order was not used.
- **Winner: CatBoost.** Frozen before any grade into `data/e2e_runH/step45/winner/`, files mode 0444 and directory 0555:

| File | sha256 |
|---|---|
| `best_model.cbm` | `c2abe7212bc9242aa77967bbade63ba7bdf8c3b558ee20cb246fbe9ecca73c93` |
| `best_model.meta.json` | `55e6bbdc5c0a9fefa0f286c7fd39cb132422b6bab1f1e262a92c72482b5e7f71` |

- The margin between first and second is 0.0015 R² on a target with std 0.00188. No statistical test was run between families, and none was declared. Reported as a fact, not a ranking claim.

### 10.5 NN normalisation check (owner 08:14)

- **Log: NO.** The two lines are absent from `data/e2e_runH/step45/neural_net.log`. `train_single_trial.py` prints them to its own stderr (`:493-495`, `:510-512`). The Step 5 parent captures that stderr and drops it on success (`meta_prediction_optimizer_anti_overfit.py:1949-1983` folds, `:2602` final). So no Step 5 log has ever shown them. This is NEW by reading.
- **Child stderr, captured by the shim: YES, in all 401 of 401 calls.** Final call (`shim_capture/call_0400.stderr.txt`):
  - `[CAT-B] Input normalization applied: 89 features, scale range [0.0000, 14557.4648]`
  - `[S121] Y normalization: mean=0.237096 std=0.001882 range=[-3.525, 3.164]`
- **Checkpoint: YES.**
  - `best_model.pth` keys: `best_epoch, dropout, feature_count, hidden_layers, normalize_features, scaler_mean, scaler_scale, state_dict, use_leaky_relu, y_mean, y_std`.
  - `scaler_mean` and `scaler_scale` are shape (89,) and finite. `y_mean` 0.237096, `y_std` 0.001882, `normalize_features` True, `use_leaky_relu` True.
- **Facts from the same check, no verdict:**
  - `scaler_scale` min is 7.4e-8, so one feature is near-constant and the zero-scale guard (`train_single_trial.py:490`) catches only an exact 0. That feature is multiplied by about 1.35e7 after centring.
  - `scaler_mean` max is 2.37e6, so the raw features span very different scales.
  - The children logged `[S96A] Full-batch mode`; whether Optuna's `batch_size` has any effect in that mode was not checked [UNVERIFIED].

### 10.6 Where `train_single_trial.py` came from (TODO-E2E-7)

- **Fold calls (400):** `/home/michael/tfm_worktrees/e2e-map/train_single_trial.py`, cwd = worktree root.
- **Final call (1):** `/home/michael/tfm_worktrees/e2e-map/data/e2e_runH/step45/neural_net/train_single_trial.py`, cwd = run dir.
- Both sha256 `59cd21d8febc50dfca8c1181a26787a079995ed6c718ae59480fe3055bee3111`, the same as the worktree file. **No call used the main tree.** The fallback at `subprocess_trial_coordinator.py:199-202` belongs to the compare-models coordinator path, which this run did not use.

### 10.7 Map §2.3 blockers that applied (plain facts)

- **B1, test split never evaluated.** All four sidecars have `test_mae: 0.0` and `overfit_ratio: 1.0`, constants. The 2,431-row test split (20 % of 12,156) was never scored.
- **B3, final NN not trained with Optuna's architecture.** Optuna's best is 2 layers [114, 112], dropout 0.2038. The checkpoint's `hidden_layers` is [256, 128, 64], with dropout 0.2038 taken from params. The child's stdout reports `architecture [256, 128, 64]` for the call that received `--params {"n_layers": 2, "layer_0": 114, "layer_1": 112, …}`. Best epoch 3.
- **B4, CatBoost ignores sampled `iterations` and `depth`.** Optuna's best is `iterations 234, depth 9`. The saved model has `iterations 200, depth 8`, the defaults read from `n_estimators`/`max_depth` (`:1105-1106`), and 6 trees. Every one of the 20 CatBoost trials therefore trained the same architecture, varying only `learning_rate`.
- **B7, Step 4 output unread.** As in §10.2: no file opens and no code references.
- **B8, holdout window not data.** All four sidecars record `holdout_window {start_draw 14455, end_draw 15454, draw_count 1000}`. That is the `--holdout-draws` default (`:3074`). The labels were made on the 1,807-draw label block (combined positions 14,454–16,260).
- **Also observed, no verdict:**
  - The NN sidecar says `provenance.compare_models_used: true` on a single-model run; the tree sidecars say `false`.
  - The NN sidecar says `hardware.cuda_available: false`, describing the parent, while the training children ran on CUDA.
  - Tree families initialise CUDA in the parent (`:3145`), so LightGBM on CPU still held about 250 MiB of card memory.
  - XGBoost warned "Falling back to prediction using DMatrix due to mismatched devices" (data on CPU, model on `cuda:0`). That warning is about speed only.

### 10.8 Writes outside the worktree

- **The tree families, under `strace`, wrote no file outside the worktree.** Each made exactly two short-lived POSIX semaphores in `/dev/shm` (`sem.XXXXXX` and `sem.joblib-<pid>-…`), created and unlinked within the run. This is kernel IPC from Python multiprocessing and joblib, not a file left behind, and I judged it not to be a stop condition. Recorded here so the owner can rule otherwise.
- **The NN run was not under `strace`.**
  - Its temp files went to its own `_env/tmp`.
  - The once-a-second `/dev/shm` poll saw nothing in 1,698 samples.
  - `outside_check.sh` after the run listed only things carrying other sessions' names: main-tree `.git/objects` and `.git/worktrees/{ch14-learn, anchor-brief2, selfplay-nn}`, and `/tmp/ch1_p0_*`, `/tmp/ch2_gate_*`, `/tmp/s172_l7_*`, `gate12_*`, `d6_2_mutants_*`, `lead_read_server.log`.
  - Two empty directories, `/tmp/tmpukfv2xme` (11:05:33) and `/tmp/tmpqqvhbm0n` (11:09:26), carry no name. Their origin was not proven [UNVERIFIED]. My run's `TMPDIR` pointed inside the run dir.
- **Main tree:** `git status` clean before and after. Nothing was written in the run H tree, `~/miner_staging` or `/dev/shm/prng`.

### 10.9 New TODOs

| TODO | Shortcut | Anchor |
|---|---|---|
| E2E-21 | Split seed carried by a shim, not the module | `meta_prediction_optimizer_anti_overfit.py:1514`; `e2e_step5_shim.py` |
| E2E-22 | Child stderr (normalisation lines, fold failures) visible only through the shim; a failed fold becomes r2 = −999.0 and is averaged in without error | `:1949-1983`, `:1957`, `:2118`, `:2602` |
| E2E-23 | Tree sidecars lack `optuna`; the CV score is read from the study db | `:1216`, `:2908` |
| E2E-24 | `run_one.sh` copies `train_single_trial.py` into cwd and passes an absolute `--output-dir`, to work around relative paths split between two cwds | `:1949-1953`, `:2508-2531`, `:2563` |

Nothing committed. Stopped after this section, as instructed: no vote, no selfplay, no grade.
