<!--
Written 2026-11-21 by a Claude Code session (Team Alpha) in /home/michael/tfm_worktrees/pwc-backup, at the Alpha
lead's request; finalised the same day with agent 2's hashes and the end-to-end synthetic check. Relayed at the
owner's "Proceed" (12:21 PST).
-->

# X-PWC-STAGE4 · R1 · ALPHA REVIEW REQUEST · Stage 4 tooling

**Owner approval to relay:** 2026-11-21. Alpha lead, 12:15 PST: "Shall I send the Stage 4 tooling to Beta when both
agents finish and I've checked it myself?" Owner, 12:21 PST, verbatim: **"Proceed"**. One round.

## Prior record

**Searches run** (`/bin/grep -rliE`, worktree `/home/michael/tfm_worktrees/pwc-backup` at `6d75f71`), pattern
`(separate|isolated|scratch|fresh) (run )?(tree|worktree|output root|accumulator)|output_root|NOT COMPARED|extract_(pwc|miner)_bundle|stage ?4 (session|runbook)|prior_row_count`:

| Scope | Files with hits |
|---|---|
| `docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`, `docs/TB_RULING_REQUEST_*`, `docs/PROPOSAL_*`, `docs/*_AUDIT.md` | 0 |
| `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/LEADS.md` | 0 |
| `docs/CHAPTER_*` | 2 (`CHAPTER_2_BIDIRECTIONAL_SIEVE.md:1280`, `CHAPTER_8_PRNG_REGISTRY.md:1470`: "not compared" about rig source; not this subject) |
| `docs/exchange/` | 2, read and listed below |
| `docs/RUN4_LAUNCH_CHECKLIST.md`, `docs/drafts/RUN4_FIRST_RUN_CHECKLIST_20261118.md`, `docs/drafts/pwc_backup_stage3_v1/` | 0 |

**Documents that already decide or record the subject:**

| Document | What it records | Label |
|---|---|---|
| `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:47` | Owner, 2026-11-20 15:55: "once finished your testing should run each and verify the results are the same" | ALREADY RECORDED. The tools under review serve it |
| `docs/drafts/pwc_backup_stage3_v1/PLAN.md` §F (`:282-379`) | Stage 4 design: inputs held equal (F2), outputs compared (F3), smallest session (F4) | ALREADY RECORDED |
| `scripts/compare_engine_outputs.py` at `6d75f71` (`390c285`) | The comparator as reviewed in X-PWC-STAGE3; it said writing bundles from real output "is Stage 4 work (an extractor per engine)" | ALREADY RECORDED. Changed here |
| `docs/exchange/from_beta/X-FINDINGS-PLAN_R1_BETA_REVIEW.md:86` | Your note: publication selects L2 winners and merges a prior generation before counting final rows (`utils/run_finalizer.py:1613-1659`) | ALREADY RECORDED. It is the premise of Q7 |
| `docs/exchange/from_beta/X-PWC-STAGE2_R2_BETA_REVIEW.md:37` | The five-strategy GPU question stays `NEEDS EVIDENCE` for the owner's Stage 4 sessions | ALREADY RECORDED. Not answered here; no run has happened |
| `docs/drafts/pwc_backup_stage4_v1/REDEPLOY_AND_SESSION_PLAN.md` §4b gap 3 | The comparator read a phase absent from both bundles as EQUAL (`compare_engine_outputs.py:131-135` at `6d75f71`) | ALREADY RECORDED. The fix is under review (Q1) |
| `/home/michael/run4_launch.sh` (outside the repository) | Run 4's launcher: a copy of `gate12_launch.sh` differing only in the `--params` block | ALREADY RECORDED. Precedent for the derived launcher (Q8) |

**Labels for what this request raises:**
- **NEW:** the `NOT COMPARED` status and exit 3; `build_inputs` and `exact_float32`; both extractors; the tests; the
  derived launcher; the separate-run-tree premise.
- Nothing here **CHANGES A RULING**.

**Not searched:** the host, Proxmox, ser8 archives, the private repository, project files outside the repository, the
rigs, the live `prng_analysis.db` and `~/miner_staging`.

## The request

- **Review class:** before commit, at the owner's decision. Two parts touch the before-commit classes:
  - the derived launcher is a launch path;
  - the extractors and the comparator are verifier-like code.

  The runs they serve do not count: they write no live coverage, because each run uses its own tree and database.
- **Base:** `6d75f71` (`6d75f71d7b637c7a329429aef3b0ba29a806a355`). Nothing is committed.
- **Trees to review:**
  - `/home/michael/tfm_worktrees/pwc-backup`: the comparator, the PWC extractor, both test files, and the runbook for
    context. The diff is `git -C /home/michael/tfm_worktrees/pwc-backup diff 6d75f71`, plus the untracked files below.
  - `/home/michael/tfm_worktrees/pwc-offset` (agent 2, base `6d75f71`, files untracked there):
    `scripts/extract_miner_bundle.py`, `tests/test_extract_miner_bundle.py` and
    `docs/drafts/pwc_backup_stage4_v1/EXTRACT_MINER_README.md`. That worktree also holds unrelated uncommitted work
    (the X-SKIP-F1 thread and a `LEDGER.md` edit); only the three files listed are under review here.
  - `/tmp/s4combo`: the Alpha lead's scratch copy holding all six code and test files together, byte-identical to
    the two worktrees (sha256 checked, table below). The end-to-end check ran there.
- **Attached in this worktree (all six code/test files and the launcher diff are here):**
  - agent 2's `scripts/extract_miner_bundle.py` and `tests/test_extract_miner_bundle.py`, copied unchanged from
    `pwc-offset` with `cp -p`. Their sha256 values equal the table below (checked after the copy), as in the Stage 3
    combined-tree precedent;
  - the derived launcher's three edits as a unified diff: `docs/drafts/pwc_backup_stage4_v1/stage4_miner_launch.diff`
    (sha256 `dd83011558e44e0e5dc592f31b388b7fa1d95e85f0463e3e4a3d8b81d7c20fc8`), made from the dry-run file. It is
    the same content as the diff quoted below.
- **Scratch location:** the `TMPDIR` the relay sets for this run (outside the repository, `AGENTS.md` §3). Please keep
  throwaway checks there.
  - Both suites write only under `<worktree>/logs/test_compare_engine_outputs/` and `<worktree>/logs/test_extract_pwc_bundle/`,
    which are git-ignored and removed by each run.
  - The extractor suite calls the real `finalize_run` with `output_root` inside that `logs/` folder.
  - Neither suite touches Optuna storage, a GPU, SSH or a rig.
- **Yardstick:** TFM is judged by whether its prediction pools hit more often than `k/1000` on draws the sieve never
  saw. These tools check engine equality, an engineering property. No verdict they print says anything about hit
  rate, and none should be read as a quality claim about either engine's survivors.

## What is NOT verified (read this first)

- **No real run.** Neither tool has read a real RANGE-MINER or PWC Stage 4 output. No Stage 4 session has run. Every
  extractor test uses a synthetic run tree:
  - the pass files are shaped like `run_sieve_pass` output;
  - the generation is built by the real `finalize_run` from synthetic candidates;
  - the ledger uses the real table definitions with hand-inserted rows.
- **Assumptions only a real run can confirm:**
  - The exact on-disk layout of a finished PWC run in a fresh tree: four `results/window_opt_*_12_25_t0.json` files at or
    under 100,000 survivors, `.s172_accumulator/current`, and `prng_analysis.db` in the working directory.
  - That the published NPZ's seeds equal the union of PWC's forward∩reverse sets. The extractor refuses otherwise; if
    this belief is wrong, it refuses on real data.
  - That a fresh run tree holds every git-ignored file a Step 1 run needs.
- **`scripts/extract_miner_bundle.py` is agent 2's.** This session read it in full (521 lines) before finalising this
  request, and ran it in the end-to-end check. It did not run agent 2's test suite itself.
- **The derived launcher was dry-run only.** The edit snippet was applied to `gate12_launch.sh` at `6d75f71`, writing
  to a scratch file; `diff` and `bash -n` were checked. It has never launched anything.
- **Test evidence is from this session's earlier runs.** No full test pass was run for this request; the counts are
  the last recorded results:
  - `tests/test_compare_engine_outputs.py`: 30/30;
  - `tests/test_extract_pwc_bundle.py`: 17/17.

  The files have not changed since, as checked by the sha256 values below (computed when this request was drafted).
  The Alpha lead reports that all three test files (the two above and `tests/test_extract_miner_bundle.py`) exit 0
  in `/tmp/s4combo`; this session did not repeat that run.
- **The end-to-end check is synthetic** (section "End-to-end synthetic check" below). It shows the two extractors'
  bundles fit together through the comparator. It says nothing about real engine output.

## Files under review

| File | Status | Lines | sha256 |
|---|---|---|---|
| `scripts/compare_engine_outputs.py` | modified (base sha256 `434fa3ba…ffa77`) | 483 | `454eb7b3046f9fbe8647c17879c50fc4de6b0dd4e713200b04acbeb5b259e6df` |
| `tests/test_compare_engine_outputs.py` | modified | 441 | `fc28b46c00d17d31030d8d0f8b707cbbd3facb49cc5c316d89d2044073cee3d1` |
| `scripts/extract_pwc_bundle.py` | new | 383 | `d32b76e0ddd58751281e61a5e246ae5c7e82ae4eed4696b0d415c4f22d15560c` |
| `tests/test_extract_pwc_bundle.py` | new | 376 | `923b89df4dd9886f5dd50951e577cb32fb190884ae6304a4b95afb53529ce0a1` |
| `scripts/extract_miner_bundle.py` (agent 2, `pwc-offset`) | new | 521 | `4c27042f44c1308d24b4ff9f0ee68b236cbb1eeb7a00102cfaeb4e68210ee611` |
| `tests/test_extract_miner_bundle.py` (agent 2, `pwc-offset`) | new | 595 | `ed705ff8c05cc34352ac2c091498ff8849e6abe4167b2ff3722eb55790461347` |
| `docs/drafts/pwc_backup_stage4_v1/EXTRACT_MINER_README.md` (agent 2, `pwc-offset`) | new, context | 226 | `6de12671e0060da0cb5b73c20413ee2e436375c6fa70cff535488fad500c473e` |
| `/home/michael/stage4_miner_launch.sh` | not yet created; specified as a diff (below) | — | — |
| `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md` | context only, not under review | — | `d964ce9916f5935675c1fd65e4763915a2cac2f95271845da7a7face5b3bad6c` |

Before relay, re-check every sha256 against the tree. If a file changed after drafting, this table must be updated.

## What changed

### Comparator (`scripts/compare_engine_outputs.py`)

1. **Three statuses per row: `EQUAL`, `DIFFERENT`, `NOT COMPARED`.** Each row keeps `equal`, and adds `status` and
   `compared`. Every NOT COMPARED row is built by one function, `_nc`; the code-mutant gate patches it.
2. **The presence rule, `_presence`:**

   | Bundle A | Bundle B | Status |
   |---|---|---|
   | key absent | key absent | NOT COMPARED |
   | null (or absent) | null (or absent) | NOT COMPARED |
   | null | value | NOT COMPARED (detail names which engine does not carry it) |
   | absent | value | DIFFERENT ("present in B only") |
   | value | value | compared |

   - Null is the extractor's way of saying "this engine's outputs do not carry it".
   - An empty list is data: `{"seeds": [], "match_rates": []}` compares EQUAL to another empty phase.
   - An empty mapping in both bundles (e.g. `phase_counts: {}`) is NOT COMPARED.
   - The expected keys are reported even when neither bundle has them: `phases` 1-4, `bidirectional` constant/variable,
     `phase_counts` 1-4, and `credit` granted/swept_intervals.
   - At `6d75f71` a phase absent from both bundles read EQUAL (`:131-135`), and so did a null leaf in both.
3. **Verdicts and exits**, in precedence order:
   - `NOT COMPARABLE`, exit 2: any input DIFFERENT, or no input EQUAL. Empty inputs now give exit 2; before, they
     passed vacuously.
   - `DIFFERENT`, exit 1.
   - `SAME WHERE COMPARED`, exit 3: NEW. No difference, but at least one NOT COMPARED field; the printed verdict names
     them.
   - `SAME`, exit 0: only when nothing was left uncompared.
4. **`load_bundle`:** a compared section may be an object or `null`; `npz` may be a path or `null`. Every compared
   top-level key must still be present.
5. **`build_inputs(params, provenance, dataset_sha256)`**, shared by both extractors. It builds `inputs` from:
   - `param.<k>` for every WATCHER `--params` key not in `ENGINE_PARAM_KEYS`;
   - `run.<k>` for `prng_base`, `skip_modes_executed`, `seed_start`, `seed_end_exclusive` and `repository_commit`, from
     `provenance.json`;
   - `run.dataset_sha256`, from the `certified_coverage` row.

   Engine keys go to `geometry`, which is allowed to differ. `seed_start` and `max_seeds` are among them, because WATCHER
   overwrites both from the open search (`agents/watcher_agent.py:2306-2308`).
6. **`exact_float32(values, where)`** returns each value as the float32 it already is. It raises `BundleError` for a value
   not exactly representable as float32, and for a string or a bool. It never rounds.
7. **Docstring:** now states the agreed field source for every bundle section, the presence rule, and the exit codes.

### PWC extractor (`scripts/extract_pwc_bundle.py`, new)

It is read-only on the run tree and writes only into a new, empty `--out-dir`. It reads:
1. The four per-pass files `results/window_opt_{forward,reverse,forward_hybrid,reverse_hybrid}_<W>_<O>_t<T>.json`.
   PWC writes them at `persistent_worker_coordinator.py:2504-2535`, with the names from `:2715`, `:2742`, `:2787` and
   `:2812`. Each pass is checked for:
   - `pwc_pass_accounting_v1`;
   - the expected `prng_type`;
   - `requested_interval` equal to the generation's interval;
   - `transport == "tcp"`;
   - a modification time no later than `provenance.json`'s (the stale-file guard).
2. **The trial completion record**, which a run never writes to disk (`:2654-2665`). The extractor rebuilds it with the
   imported `build_pwc_trial_completion` (`:725-771`). It requires `trial_complete` and every required pass `complete`.
   `credit.swept_intervals` comes from it.
3. **Phases.**
   - With full lists, the seeds are stably sorted and the match rates go through `exact_float32`, with
     `survivor_count == len(survivors)` checked.
   - A summary-only file (more than 100,000 survivors) gives `null`.
   - `bidirectional` is forward∩reverse per mode, from the lists, and `null` if either list is missing.
4. **The generation** at `.s172_accumulator/current`:
   - it is refused unless `parent_generation_id` is null and `prior_row_count == 0`;
   - the NPZ's sha256 must equal `artifact_sha256`;
   - the arrays must be in the frozen 22-array order;
   - the NPZ is copied next to `bundle.json` and re-hashed.
5. **Cross-checks, refused on failure:**
   - the NPZ seeds equal the union of the bidirectional sets;
   - each `skip_mode`'s NPZ rows lie inside that mode's set.
6. **The ledger** `prng_analysis.db`, opened `?mode=ro&immutable=1`:
   - `certified_coverage` by `artifact_sha256` gives `credit.granted`, `certified_interval` and `dataset_sha256`;
   - `step1_chunk_link` by `generation_id` gives `linked_interval`.
7. **Refused paths:** a run tree or ledger under `~/distributed_prng_analysis` or `~/miner_staging`. The paths come
   from `os.path.expanduser`, and tests inject their own list.

### Miner extractor (`scripts/extract_miner_bundle.py`, agent 2)

Summarised by this session from its reading of the file; agent 2's own account is `EXTRACT_MINER_README.md`.

- **Sources:**
  - `<staging-dir>/miner_ledger.db`: the `trials` row (must be `committed`/`done`), `stripes` (swept intervals and
    per-phase `survivors_total`), `shards` (cross-check of each stripe's count against its current attempt), and
    `trial_context`;
  - the published generation, with the same parent-free, hash-check and copy-byte-for-byte rules as the PWC
    extractor;
  - the run ledger (`certified_coverage`, `step1_chunk_link`);
  - the `--params` JSON. `inputs` comes from the comparator's `build_inputs`.
- **Written as null (NOT COMPARED)**, each listed with its reason in `provenance.not_available`: `phases.1-4`, both
  `bidirectional` sets, and every `distinct_survivors`. A finished miner run keeps no per-phase seed lists, and the NPZ
  keeps one row per seed. `survivor_rows` is null too when a stripe count is NULL or disagrees with its shards.
- **Sqlite reads** are immutable, or, when a `-wal` is non-empty, made on a byte copy checked unchanged across the
  copy.
- **Refusals (exit 2):**
  - a live path;
  - not a RANGE-MINER run;
  - an uncommitted trial;
  - a stripe not `done`, or stripes with a gap or overlap;
  - stripe phases that do not match the skip modes;
  - a parent generation or a hash mismatch;
  - another trial's rows in the NPZ;
  - a missing ledger;
  - a non-empty `--out-dir`, or one inside the run;
  - a comparator without `build_inputs`.
- **Warnings only:**
  - swept or certified interval off the generation's;
  - a dataset digest mismatch between ledgers;
  - the F-1 constant skip range read from this tree's code (`read_skip_range_code`);
  - NPZ array order.
- **Differences from the PWC extractor** (both inside the agreed schema):
  - it fills `finished_at`, `duration_s`, `worker_ids`, `job_ids` and `logs`, all allowed to differ;
  - it takes `--staging-dir` and `--output-root` where the PWC extractor takes `--run-dir`;
  - it has no stale-file guard; the miner ledger's `committed` state plays that role.

### Tests

- **`tests/test_compare_engine_outputs.py`:** the 13 original gates are unchanged in what they assert. Their
  `differing()` helper now counts `status == DIFFERENT`. Added:
  - N1-N14, including N9 (absent in one bundle only → DIFFERENT), N10 (a NOT COMPARED field plus a real difference →
    exit 1) and N11 (an empty phase is compared EQUAL);
  - `G-CODE-MUTANT`: `_nc` patched to return EQUAL, the pre-fix behaviour; all 12 dependent N gates must fail under it;
  - `G-BUILD-INPUTS` and `G-EXACT-FLOAT32`;
  - an exit-3 case in `G-SUBPROCESS-EXIT`.
- **`tests/test_extract_pwc_bundle.py`:**
  - `G-CLEAN`: two identical synthetic PWC trees give two bundles, and the comparator reports exit 0, SAME;
  - `G-READONLY`: the run tree is byte-identical afterwards, with no `-wal`/`-shm`;
  - `G-SUMMARY`: a summary-only phase is null, and the comparator gives exit 3;
  - R1-R13, one per refusal;
  - `G-SUBPROCESS`.

### The derived launcher: three edits to `gate12_launch.sh`

The runbook (`SESSION_RUNBOOK.md` §3 A4) makes `/home/michael/stage4_miner_launch.sh` from the run tree's
`gate12_launch.sh` (identical to `6d75f71`) with an asserted-replace snippet. Dry-run result, `diff gate12_launch.sh
<derived>`:

```diff
79c79
< cd ~/distributed_prng_analysis || exit 1
---
> cd /home/michael/stage4_run_miner || exit 1
468,472c468
<   --params '{"use_persistent_workers": false, "use_range_miner": true,
<              "worker_pool_size": 25,
<              "seed_start": 0, "max_seeds": 2147483648,
<              "miner_stripe_size": 67108864, "test_both_modes": true,
<              "prng_type": "java_lcg", "window_trials": 1, "n_parallel": 1}' \
---
>   --params "$(cat logs/stage4/params.json)" \
523c519
< tail -f "$LOG"
---
> echo "launcher done; follow $LOG by reading it, not by tail -f"
```

- `:79` is the only main-tree path in the script (`grep -n distributed_prng_analysis gate12_launch.sh`).
- `logs/stage4/params.json` is Run 4's params JSON (`docs/RUN4_LAUNCH_CHECKLIST.md:313`) with:
  - `max_seeds` 268435456;
  - `warm_start_skip_min` 0 and `warm_start_skip_max` 16;
  - `staging_dir` `/home/michael/stage4_miner_staging`.
- The `tail -f` removal lets the launcher exit under `nohup`.
- Every gate in the launcher is unchanged. That includes the GPU fail-close gate's one `rocm-smi` per rig, before any
  worker starts (`preflight_check.py:118-124`).

## End-to-end synthetic check (run 2026-11-21, `/tmp/s4combo` only)

Script: `/tmp/s4combo/logs/e2e_stage4/e2e.py` (sha256 `72f7e74c…b218b`). Full output:
`/tmp/s4combo/logs/e2e_stage4/e2e_output.txt` (sha256 `b9671c1c…d44a18`). CPU only; nothing outside `/tmp/s4combo`
was written.

**Fixture:** one synthetic result, a seed→rate map per phase (agent 2's test maps, interval `[1000, 1400)`), written
as both:
- a finished RANGE-MINER run: the real `MinerLedger` schema, with `trials`, `trial_context`, `stripes` and `shards`
  consistent with the maps;
- a finished PWC run: four pass files in the `run_sieve_pass` shape.

Both share one published generation, built by the real `build_mode_records` and `finalize_run`, and one run ledger
(real `CoverageLedger.record_publication` plus a `step1_chunk_link` row). The params files differ only in engine
keys.

| Case | Bundles | Exit | Verdict |
|---|---|---|---|
| 1. Same result | miner bundle vs PWC bundle, same tree | **3** | `SAME WHERE COMPARED` |
| 2. One-seed difference | the same miner bundle vs a PWC bundle from a second tree whose phases 1 and 2 carry seed 1003 in place of 1002 (1002 is the constant-only bidirectional seed) | **1** | `DIFFERENT` |

- **Case 1:** 52 EQUAL rows (17 inputs; `phase_counts.*.survivor_rows`; all `credit` fields; the 22 NPZ arrays and
  both order rows), 0 DIFFERENT, and 10 NOT COMPARED: `phases.1-4`, `bidirectional.constant/variable`, and
  `phase_counts.1-4.distinct_survivors`, each "A's engine outputs do not carry it (null); B does".
- **Case 2:** the one DIFFERENT row is `npz.seeds  first differing index 0: A=1002 B=1003 (1 of 3 differ)`; the same
  10 NOT COMPARED rows; 51 EQUAL.
- **Self-checks:** the PWC extractor's both passed in both trees. The miner extractor printed no warning: the trial's
  `[0,16]` equals the worker's constant default, so the F-1 check is silent.

**Limits of this check:**
- In case 1 both bundles read the same NPZ and the same run ledger, so the equality of `npz.*`, `credit` and the
  `run.*` inputs is by construction. What case 1 shows is that the two extractors' bundles are schema-compatible and
  that the comparator returns exit 3, not 0, when the miner side carries no seed lists.
- Case 2 uses a separate tree for PWC (its own generation and ledger), so the comparison across independent trees
  works too.
- No real engine output was involved.

## Real-world scenarios

1. **RANGE-MINER keeps no per-phase seed lists on disk** (`window_optimizer_integration_final.py:1256-1258`); PWC's
   pass files do. The miner bundle has `phases.N: null`, the PWC bundle has lists, and the comparator gives
   `phases.N` NOT COMPARED, so at best exit 3. At `6d75f71` this read EQUAL when absent in both.
2. **Both runs in the main tree.** The second generation's parent is the first, so its NPZ carries the first run's
   rows. A PWC run that missed rows would still compare EQUAL on the NPZ. The extractor refuses any generation with a
   parent. The runbook uses one fresh `git worktree` per engine.
3. **A phase over 100,000 survivors in PWC** writes a summary-only file, so that phase is null and NOT COMPARED, never
   EQUAL and never DIFFERENT.
4. **A difference that never reaches the NPZ.** Example: a seed that only one engine finds in phase 1 alone, not
   bidirectional. With the miner bundle carrying no seed lists, `phases.1` is NOT COMPARED. The difference shows only
   as a `phase_counts.1.survivor_rows` mismatch, and only if the counts differ (a swapped seed keeps the count). This
   is the cost of exit 3, and why exit 3 is never read as "the same". Case 2 was caught through the NPZ because the
   changed seed was bidirectional.
5. **A stale `results/` file from an earlier run in the same tree.** It is refused when newer than the generation's
   provenance, or when its interval differs. A stale file older than the provenance with an identical interval would
   pass. The runbook's fresh trees are the defence against that case.

## Questions (allowed answers: NO OBJECTION · OBJECTION · SUGGESTION · NEEDS EVIDENCE; LIVE/THEORETICAL on every OBJECTION)

- **Q1, the presence rule.** Is NOT COMPARED for "null on either side" sound? In particular, can it hide a real
  difference that the old "present in one only → DIFFERENT" rule would have caught? Alpha's position: an extractor bug
  that nulls a section can now hide a difference, but never as SAME (exit 3 names the field). An absent key stays
  DIFFERENT.
- **Q2, exit codes.** Is the precedence 2 > 1 > 3 > 0 right? Is "no input EQUAL means not comparable" right? And is
  exit 3 a safe addition for any caller that treated non-zero as "differ"? No caller in the repository reads this exit
  code today [UNVERIFIED beyond this tree's `scripts/` and `tests/`].
- **Q3, `build_inputs`.** Is `ENGINE_PARAM_KEYS` the right split? Does any job-defining key wrongly go to `geometry`, or
  any engine key wrongly stay in `inputs`? Is taking the interval and commit from `provenance.json`, and the dataset
  digest from `certified_coverage`, the right source of "the same job"?
- **Q4, `exact_float32`.** Refusing, rather than rounding, a rate that is not exactly float32: is that right, given
  that PWC's rates are GPU float32 → `.tolist()` → JSON (`sieve_filter.py:252`, `:318`, `:488`)? Is a stable sort by
  seed, keeping each repeated seed's first rate, the right canonical order?
- **Q5, `extract_pwc_bundle.py`:** are its sources, its rebuilt completion record (the imported builder) and its 13
  refusals sufficient? Is anything it should refuse missing? Specific points:
  - the stale-file guard by modification time;
  - the union cross-check, given L2 keeps one row per seed (`utils/run_finalizer.py:714-744`);
  - the live-path list.
- **Q6, `extract_miner_bundle.py`** (agent 2, sha256 `4c27042f…ee611`): does it follow the comparator's field list exactly,
  so that a difference is the engines' and not the extractors'?
- **Q7, the separate-run-tree premise.** Is a parent-free generation the right requirement? Does a fresh `git worktree`
  per engine isolate the accumulator, `prng_analysis.db`, Optuna storage and the miner ledger, as cited?
  - `window_optimizer_integration_final.py:3578`
  - `database_system.py:46`
  - `window_optimizer_bayesian.py:23`
  - `miner/range_miner_coordinator.py:12371-12372`, `:12537`
- **Q8, the derived launcher.** Do the three edits leave every gate of the RANGE-MINER launch path intact? Is a copy
  outside the repository (the Run 4 precedent) acceptable for a launch path?
- **Q9, tests.** Do the gates test behaviour, not assignments? Does the code mutant show the fix is tested? Is any
  refusal untested?

## Positions table (please fill in)

| Question | Position | Severity |
|---|---|---|
| Q1 Presence rule | | |
| Q2 Exit codes | | |
| Q3 `build_inputs` | | |
| Q4 `exact_float32` and order | | |
| Q5 `extract_pwc_bundle.py` | | |
| Q6 `extract_miner_bundle.py` | | |
| Q7 Separate run trees | | |
| Q8 Derived launcher | | |
| Q9 Tests | | |
