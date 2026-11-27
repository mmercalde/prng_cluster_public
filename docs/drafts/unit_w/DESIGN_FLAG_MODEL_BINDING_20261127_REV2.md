# DESIGN NOTE REV2: the why-report flag names the model that was flagged, with proof (unit W3)

Date: 2026-11-27. Author: Team Alpha build agent. Base: main `14815a7`, worktree
`/home/michael/tfm_worktrees/w3_binding_design`. REV1 (`DESIGN_FLAG_MODEL_BINDING_20261127.md`) is kept untouched; this
note replaces its sections 3 to 7. Inputs: Beta's design review `docs/exchange/from_beta/X-W3-BINDING_R1_BETA_REVIEW.md`
(main tree) and the owner's four decisions of 2026-11-27 10:26 PST, verbatim "1.yes. 2 yes 3. yes 4 yes.".

Abbreviations as in REV1: `wa` = `agents/watcher_agent.py`, `thc` = `training_health_check.py`,
`td` = `training_diagnostics.py`, `mpo` = `meta_prediction_optimizer_anti_overfit.py`, `tst` = `train_single_trial.py`,
`wrf` = `why_report_flags.py`, `wrr` = `scripts/diagnostics/why_report_runner.py`, `rj` = `agent_manifests/reinforcement.json`.
Line numbers are on `14815a7` unless marked.

## 1. Prior record

### 1.1 Searches run

REV1 section 1.1 searches 1 to 7 stand (REV1 lines 19 to 45). Added for REV2, from the worktree root:

1. `for g in docs/PROJECT_FILE_CATALOG.md 'docs/TB_RULING_*' 'docs/TB_RULING_REQUEST_*' 'docs/PROPOSAL_*' 'docs/CHAPTER_*' 'docs/*_AUDIT.md' docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md docs/exchange/; do git grep -l -E 'as many tree|tree models at once|run-local|run_local|staging dir|step5_identity|binding_run_id|model_id|G-2|grade_set_by|X-W3' -- "$g"; done`
   Hits read: `docs/TODO_CONSOLIDATED.md:86` (miner staging dir, other subject), `:443` (host state, other subject),
   `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:1185` ("G-2" is a sieve gap label, other subject),
   `docs/exchange/from_beta/X-BENCHING-SWITCH_R1_BETA_REVIEW.md:39` (the G-2 fix, recorded). The other exchange hits
   are `model_id`/`staging` in PWC and miner reviews, other subjects.
2. `git grep -n -i -E "tree.*(parallel|at once|concurrent)" -- docs/TODO_CONSOLIDATED.md docs/BACKLOG.md`
   -> `docs/TODO_CONSOLIDATED.md:168` "Phase 3B: parallel tree trials", status REVIEW.
3. Read in full: `logs/e2e_session/OWNER_RULING_benching_switch_20261126.txt` (main tree),
   `docs/drafts/loop_v1/READY_G2_GRADE_SCOPE_FIX_20261126.md`, `PLAN_CH14_DIAGNOSTICS.md` rows CD-05 (`:187`) and CD-07
   (`:174`), `docs/drafts/loop_v1/DRAFT_WATCHER_CD07_20261123.diff`, `docs/PROPOSAL_PHASE3_CONCURRENT_TRIAL_BATCHING_v1_0.md`
   (sections 1 to 4, file table `:353-355`), `docs/drafts/loop_v1/X-BENCHING-SWITCH_R2_ALPHA_CLOSING_NOTE_benching_switch.md:60-80`.

### 1.2 Documents that decide or record the subject (added to REV1 D1 to D15)

| # | Document and place | What it records |
|---|---|---|
| D16 | `OWNER_RULING_benching_switch_20261126.txt:7` (07:14) | Owner yes to the G-2 fix: fixed grading with `grade_scope all_families` must report the family that set the grade. |
| D17 | `READY_G2_GRADE_SCOPE_FIX_20261126.md:34-62`; closing note `X-BENCHING-SWITCH_R2_ALPHA_CLOSING_NOTE_benching_switch.md:65-73`; `thc:881-893` | A promoted family's grade carries its own report, report path, source and freshness; the winner's are kept under `winner_*`. **The flag must follow the same report** (Beta Finding 1). |
| D18 | `PLAN_CH14_DIAGNOSTICS.md:187` (CD-05) | Tree reports on the paths that run, with a **run-scoped per-family path** `diagnostics_outputs/{run_id}/{model}_diagnostics.json` next to the canonical copy. Built on 14815a7: the canonical paths only (`mpo:2836-2839`); the run-scoped path is not built (`git grep` finds no writer). |
| D19 | `docs/TODO_CONSOLIDATED.md:168`; `PROPOSAL_PHASE3_CONCURRENT_TRIAL_BATCHING_v1_0.md` section 4 | Parallel tree work in Step 5 (Phase 3B), status REVIEW. This is the "as many tree models at once as fit" item the owner asked about. |
| D20 | Owner decisions 2026-11-27 10:26 (task brief) | 1 identity on the flag line (option A; amends D9 Item 7); 2 strict stale guard when WATCHER passes an id, no-id behaviour exactly as 14815a7; 3 tree pair recorded in the sidecar by the final-model writer, report not reordered; 4 hash deadline setting, default 10 s. |

### 1.3 Labels for what REV2 adds

| Item | Label |
|---|---|
| Option A, identity on the one flag line | CHANGES A RULING (D9 Item 7); **owner amended it, D20 item 1** |
| Strict stale guard when WATCHER issued an id | ALREADY RECORDED as CD-07 (D1); strictness is NEW, owner yes D20 item 2 |
| Pair taken by the final-model writer, recorded in the sidecar | CHANGES A RULING (D8 named report and sidecar); **owner yes D20 item 3** |
| Run-local staging dir `diagnostics_outputs/<run id>/<model id>/` | NEW. Extends CD-05's run-scoped path (D18) by one model-id level |
| Identity keyed by model id, more than one model per family | NEW (forward compatibility for D19) |
| Explicit binding request `--binding-run-id`, separate from `--diagnostics-run-id` | NEW (Beta Q5) |
| Compare publishes a child's identity only on returncode 0 | NEW (Beta Q2) |
| Latest flag occurrence decides; an unbound W3 occurrence never falls back | NEW (Beta Q2) |
| Flag follows the exact graded report tuple, also a promoted non-winner | ALREADY RECORDED as an order (D16, D17); design NEW |
| Hash deadline, unbound on expiry | NEW, owner yes D20 item 4 |
| Runner takes the checkpoint path from the flag | NEW (Beta Q6) |

### 1.4 Not searched

The host and Proxmox, rigs, ser8 archives, the private repository, project files outside the repository, live
`diagnostics_outputs/` and `models/` on VM101.

## 2. Beta's points, checked against the code

| Beta point | Check on 14815a7 | Verdict |
|---|---|---|
| Finding 1: G-2 and CD-05 run-scoped path omitted | `thc:881-893` (promoted result carries its own report); CD-05 row `:187` names `diagnostics_outputs/{run_id}/{model}_diagnostics.json`; no writer of it in code | Correct. Folded in (sections 3, 4.4). |
| Q1: two Step 5 runs on one output dir can pair A's report with B's checkpoint | `save_model` writes `best_model.<ext>` in `output_dir` (`mpo:1306-1327`); every compare child writes `models/reinforcement` (`mpo:262-268`, no `--output-dir`); REV1 C9 hashed the public path after writing | Correct. Fixed by run-local staging (4.1). |
| Q2a: an unbound occurrence falls back to an older digest or the live model | `wrr:202-216` takes the latest line **with** a sha; `wrr:562-563` no sha means the live checkpoint | Correct. Fixed (4.6). |
| Q2b: child writes a full pair then exits 1, still accepted | `mpo:308-325` archives before looking at `rc`; `mpo:379` returns 0 | Correct. Fixed (4.2). |
| Q2c: fixed grading promotes a non-winner; flag must follow its tuple | `thc:849-852` promotion, `:869` `_severity_to_action` raises the flag, `:888-893` report path set after | Correct. Fixed (4.4). |
| Q3: one id per dispatch; stamp also a result loaded from disk | `wa:2510-2512` returns `_find_results(5)` (file on disk) | Correct. Built (4.5). |
| Q4: legacy NN collector and fallback emitter | `tst:787-806`: `_nn_diag.save` else `_emit_nn_diagnostics` fallback | Correct. Fallback path writes no pair, stays unbound. |
| Q5a: full mode already makes an id; it alone must not turn W3 on | `td:142-164` generates an id at full; `mpo:269-278` forwards it to children | Correct. Separate flag (4.1). |
| Q5b: best-effort and atomic writes | `mpo:2817-2842` already catches; restore precedes summary writes `mpo:353-379` | Correct. Built (4.7). |
| Q5c: no-id stale path and the registry reset | `thc:816-822` fixed grader can give `stale` from the pointer alone with no id; REV1 C5 would then stop a reset | Correct. C5 keyed on the id (4.5). |
| Q5d: with an id, an unstamped report falls through to mtime | `thc:920-946` | Correct. Strict rule (4.3), owner D20 item 2. |
| Q6: hand the checkpoint path to the runner; gate reaching LL-8 | `wrr:243-252` searches by sidecar only; LL-8 command `wrr:594-597` | Correct. Built (4.6). |
| Q7: deadline on the hash | `thc:1304-1316` appends synchronously | Correct. Built (4.8), owner D20 item 4. |

No Beta point was found wrong.

## 3. The binding rule (REV2)

WATCHER makes one run id per Step 5 dispatch and passes it to Step 5 as an **explicit binding request**
(`--binding-run-id`). Only that request turns W3 on. With it, each final-model writer in Step 5 writes its report bytes
and its checkpoint bytes into its own run-local staging dir `diagnostics_outputs/<run id>/<model id>/` first, takes both
sha256 values from those staged files, and then writes the public files from the same bytes. After Step 5 (single model)
or the compare child (compare) exits 0, an identity record `identity.json` is published in that staging dir.

The flag may name a checkpoint only when every link holds:

1. WATCHER's run id = the graded report's `diagnostics_run_id` (strict: no id on the report is a failure);
2. flagged family = the graded report's own `model_type`, read before the S93 override;
3. exactly one published `identity.json` under `diagnostics_outputs/<run id>/` has this run id, this family, and
   `report_sha256` = sha256 of the exact bytes that were graded (and the same model id when the report carries one);
4. for the neural net, the graded report's own `checkpoint_sha256` = the record's `checkpoint_sha256`;
5. the staged checkpoint file named by the record hashes, within the deadline, to the record's `checkpoint_sha256`.

Then the flag line names the staged checkpoint path, its dir and sha256 (`binding: bound`). Else `binding: unbound` with
the first failed link as the reason. The flag is never delayed past the deadline, dropped, or doubled.

## 4. Design

### 4.1 Producer provenance for the pair (Beta Q1, Q5a; owner D20 item 3)

- **Explicit request.** New Step 5 options `--binding-run-id`, `--binding-run-started-at`, `--binding-model-id`. Only a
  non-empty, path-safe `--binding-run-id` (`[A-Za-z0-9._-]`, at most 96 characters, not starting with a dot) turns W3 on.
  `--diagnostics-run-id` alone, and the id level `full` makes by itself (`td:153-156`), never do.
  With the request, Step 5 sets `diagnostics_run_id` to the binding id when none was given, so the report and the
  compare pointer carry the same id.
- **Run-local staging is the rule.** Model id: `--binding-model-id` if given, else `<family>-<8 hex>`. Compare passes
  `cmp-<family>` to each child. The dir `diagnostics_outputs/<run id>/<model id>/` belongs to one writer: the run id is
  unique per dispatch (8 random hex, `td:134-139`), the model id is unique inside the run. No other code writes there.
- **Trees (level full; the only level with a tree report, CD-05).** `_ch14_emit_inline_tree_report` first writes the
  report exactly as on 14815a7 (those three lines are kept byte for byte; an existing Chapter 14 mutant anchors on
  them), then serializes the report once to bytes and writes them to the staging dir, then over
  `{family}_diagnostics.json` and the canonical copy (if the staged write fails, the 14815a7 report stays). Then
  `save_model` saves the same fitted object to `staging/best_model.<ext>` and copies those bytes to the public path. The
  public write order (report, then checkpoint, then sidecar) is unchanged; the report is not reordered.
- **Neural net (any level with diagnostics on).** `_run_nn_via_subprocess` (the final NN path only; Optuna folds use
  `_run_nn_optuna_trial`) passes the request and the staging dir to `tst`. `tst` saves the checkpoint to
  `staging/best_model.pth`, copies it to its public trial path, and writes the report with `diagnostics_run_id`,
  `checkpoint_sha256` (of the staged file it wrote) and `binding_model_id`, from one byte string, to staging and to the
  public report paths. The fallback emitter (`tst:805-806`) gets no request and stays unbound.
- **Who may publish.** Only the final-model writers above write `identity.pending.json`, after checking the staged
  files themselves (NN: the staged report's `checkpoint_sha256` must equal the staged checkpoint's sha256). A trial
  report can never be published: trial and fold paths never receive the request.
- **Sidecar.** The pair is also written into the public sidecar as a new top-level block `step5_identity` (owner D20
  item 3). `agent_metadata`, `schema_version` and every existing key are unchanged.

### 4.2 Publication and the compare exit gate (Beta Q2b)

- Single model: `main` publishes `identity.pending.json` -> `identity.json` only on the normal exit-0 path, never for a
  compare child (env `S88_COMPARE_MODELS_CHILD`), never on the degenerate exit 2. It publishes only the record the
  accepted final path wrote in this process (`_w3_publish_final`, `w3_publish(expect=...)`); the NN parent writes its
  pending record only after the worker's metrics were accepted, so an inline-fallback run has no pair and stays
  unbound, and a rejected worker's leftover `identity.pending.json` is never published (fix pass 2, Beta X-W3-BINDING
  R2 Q2).
- Compare: the parent publishes a child's record only when that child's `returncode == 0`. A child that wrote a full
  pair and exited 1 leaves only `identity.pending.json`, which the health check never reads.
- Publication re-checks the staged files against the record and writes `identity.json` by temp file + `os.replace`, so
  the file is visible only when complete.

### 4.3 Stale guard (owner D20 item 2; Beta Q5c, Q5d)

- `check_training_health(..., binding_run_id=ID)` sets `expected_run_id = ID` and `strict_run_id`. Strict: a report with
  no `diagnostics_run_id` or another id is `stale`; pointer, start time and mtime are not consulted. A stale report
  is never graded with an id, whatever `stale_report_action` says: the setting `grade` gives the not-graded
  `PROCEED_WITH_NOTE` / `stale` result, with `stale_report_action_overridden` saying why (fix pass 2, Beta X-W3-BINDING
  R2 Q5; `training_health_check.py` `_stale_result`). `PROCEED_WITH_NOTE` and `RETRY` keep their own not-graded
  result. No id: `grade` grades the stale report, as on 14815a7.
- Fixed grading with an id reads the compare pointer once; the parsed object that passed the id check is the one that
  routes (fix pass 2, Beta R2 Q5). Each report read keeps its identity tuple with it; the winner's read is reused in
  the per-family pass, and the grade-setter's own tuple goes to the flag (fix pass 2, Beta R2 Q3).
- Fixed grading with an id: a compare pointer whose `diagnostics_run_id` is not this id is ignored and the canonical
  report is graded (Beta Q5 "current single-model report next to a leftover pointer").
- No id: every path as on 14815a7, including the fixed grader's pointer rule and `stale` from the pointer alone.
- WATCHER: a `stale` result does not reset the skip registry **only when the result carries `binding_run_id`**. A no-id
  `stale` keeps the 14815a7 reset (Beta Q5).

### 4.4 Which report the flag follows (Beta Q2c, Finding 1)

With an id, every report load keeps a tuple `(sha256 of the exact bytes parsed, path, own model_type before any
override, diagnostics_run_id, checkpoint_sha256, binding_model_id)`. Before `_severity_to_action` can raise the flag,
`run_ctx['graded_report']` is set to the tuple of the report that set the grade: the canonical report on the legacy path,
the winner's report on the fixed path, or the promoted family's report when G-2 promotes a non-winner.

### 4.5 WATCHER (owner patch)

- `run_step`, Step 5, when the manifest declares `binding_run_id`: make `(id, started) = new_run_id()` after the merge
  and before the command is built; always overwrite any caller value; log it; stamp `step5_run_id` and
  `step5_run_started_at` on every dict returned after that point, including one loaded from disk.
- Health call: `binding_run_id` and `run_started_at` are passed only when the result carries an id. With no id the call
  is the 14815a7 call.
- `_handle_training_health`: no registry reset for `stale` when `health['binding_run_id']` is set.
- Manifest: `"binding_run_id": null, "binding_run_started_at": null` (the builder skips `None`, `wa:2433-2434`).
- Registry first (D13): two STRING entries, "set by WATCHER only; never proposed by an LLM".

### 4.6 Flag line and runner (owner D20 item 1; Beta Q2a, Q6)

- With an id, the one flag line also carries `binding_schema: "w3/1"`, `binding`, `binding_reason`, `step5_run_id`,
  `model_id`, `models_dir`, `checkpoint_path`, `checkpoint_file`, `checkpoint_sha256`, `report_path`, `report_sha256`,
  `flagged_model_error`. With no id the line is byte-identical to 14815a7 and the returned flag is as 14815a7.
- Runner `flagged_model`: if the flag's **latest** line has `binding_schema`, it decides alone. Bound: its sha and
  explicit checkpoint path. Unbound: the run fails with reason `no_checkpoint`, error class `FlagUnbound`, and the
  line's reason text; nothing is attached, the flag stays open; no older sha and no live model is used. (Corrected in
  fix pass 2, Beta X-W3-BINDING R2 Q6: this line said reason `flag_unbound`, which the runner never used. The W1
  runner's reason set is fixed: lead decision D-2, an unusable checkpoint is `no_checkpoint`, and D-5, a fixed set of
  failure reasons (eleven when D-5 was written; twelve in `REASONS` at 14815a7, the twelfth `no_graded_draws`), `docs/exchange/to_beta/X-W1-RUNNER_R1_ALPHA_REVIEW_REQUEST_why_report_runner.md:15`;
  `REASONS` at `scripts/diagnostics/why_report_runner.py:98`. The unbound stop is at `why_report_runner.py:591-595`.) A latest line without `binding_schema` keeps the 14815a7
  rule (latest line with a sha, else the live model).
- Runner `checkpoint_by_sha`: after the walk-forward dirs and `--models-dir`, the flag's explicit `checkpoint_path` is
  tried and accepted only on a sha match, before the flag's `models_dir` sidecar route.

### 4.7 Failure behaviour (Beta Q5b)

Every W3 write is in its own `try`. A staging failure falls back to the 14815a7 public write and leaves no pending
record (so the run is unbound, never wrong). No W3 code can change Step 5's exit code, the compare winner rule or the
winner restore (the publish step runs after the child's return code is read and touches only the staging dir), or the
health action (the binding is computed inside its own guard; any error gives `unbound` with the error).

### 4.8 Deadline (owner D20 item 4; Beta Q7)

All binding file I/O (record search, reads, checkpoint hash) runs in one worker thread. The health check waits at most
`training_diagnostics.why_flag_binding_deadline_seconds` (default 10; a missing, non-number, non-positive or non-finite
value uses 10 and says so in the reason). On expiry or read error the flag is appended once, unbound, with the reason. The
worker is a daemon thread; a stuck read can outlive the check (it holds no lock and writes nothing). Owner measurement:
largest saved model 1,133,848 bytes, sha256sum 0.00 s warm cache; cold cache not measured [UNVERIFIED].

## 5. Forward compatibility: many tree models per family in one Step 5 run (D19)

What that future work can rely on (built here):
- identity records are keyed by model id, one dir per model, so many records per family per run can coexist;
- the binding picks the record whose `report_sha256` equals the graded bytes, and refuses (unbound, "ambiguous") when
  more than one record matches;
- a report written with a request carries `binding_model_id`; the binding checks it against the record;
- staging dirs cannot collide while each concurrent writer gets its own model id.

What it must still do [UNVERIFIED; not built, not designed here]:
- give every concurrent writer its own `--binding-model-id` (today one model per family per process: `cmp-<family>` or
  a random one);
- stop concurrent writers sharing the public `best_model.*`, `{family}_diagnostics.json` and canonical report paths,
  or grade from the staged reports instead of the public ones;
- decide which model of a family the health check grades and the flag names when several exist;
- extend the compare parent's publish step to more than one child per family.

## 6. Tests (all CPU, synthetic files, stubbed children; `tests/test_flag_model_binding.py`)

Every REV1 section 5 case, plus Beta's: interleaved writers before the pair is recorded; bound then unbound; unbound
alone with a usable substitute present; child writes a full pair then exits 1; fixed grading with a healthy winner and a
critical non-winner (`all_families`); full level with no binding request; injected read and write errors; deadline
expiry; absent-is-today gates against `git show 14815a7` copies; an installed-code gate that SKIPs when the owner patch
is not installed and has a negative control. Source mutants for the main gates.

## 7. What does not change

Training maths, model selection, the compare winner rule and restore, `_s88_extract_score` keys, the sidecar's existing
keys, `agent_metadata.run_id` (Step 6), Chapter 13 reads, the inference bundle, the automatic why-report switch (off).
With no id: argv, reports, sidecars, compare summaries and archives, health results, the skip registry and the flag line
are as on 14815a7.

## 8. Open questions

1. Owner: the staging dirs grow by one small dir per final model per Step 5 run with an id (report plus one checkpoint
   copy, about 1 MB at the measured largest). Prune with the diagnostics history cap, or keep?
2. Owner: at level `legacy` (WATCHER's default today) only the neural net can be bound; tree flags stay unbound with the
   reason "no tree report at level legacy". The `diagnostics_level: full` manifest entry (your 2026-11-23 yes) is not
   in this patch.
