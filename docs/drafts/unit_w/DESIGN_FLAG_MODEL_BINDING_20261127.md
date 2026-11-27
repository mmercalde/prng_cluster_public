# DESIGN NOTE: the why-report flag names the model that was flagged, with proof (unit W3)

Date: 2026-11-27. Author: Team Alpha investigation agent. Base: main `14815a7`, worktree
`/home/michael/tfm_worktrees/w3_binding_design`. Design only. No code was written, nothing was committed, no pipeline,
training or GPU was run. Team Beta reviews this design before any code is written.

Abbreviations used for paths: `wa` = `agents/watcher_agent.py`, `thc` = `training_health_check.py`,
`td` = `training_diagnostics.py`, `mpo` = `meta_prediction_optimizer_anti_overfit.py`, `tst` = `train_single_trial.py`,
`wrf` = `why_report_flags.py`, `wrr` = `scripts/diagnostics/why_report_runner.py`, `rj` = `agent_manifests/reinforcement.json`,
`ib` = `models/inference_bundle.py`. Every line number is on `14815a7` unless it says otherwise.

What this note is about, in one sentence: the flag must not name a checkpoint unless the files prove that the checkpoint
came from the same Step 5 run, and the same model family, as the diagnostics report that was graded.

---

## 1. Prior record

### 1.1 Searches run (exact commands, in the worktree unless marked)

1. Hit counts per required scope:
   `git grep -l -E -i 'CD-07|run.identity|checkpoint_sha256|flagged_model|flag.*binding|binding.*flag|unbound|parent_run_id|parent-run-id|diagnostics_run_id|output.dir.*compare|compare.*output.dir|TODO-WHY|W3' -- <scope>`
   for each scope: `docs/PROJECT_FILE_CATALOG.md`, `'docs/TB_RULING_*'`, `'docs/TB_RULING_REQUEST_*'`, `'docs/PROPOSAL_*'`,
   `'docs/CHAPTER_*'`, `'docs/*_AUDIT.md'`, `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/LEADS.md`,
   `docs/exchange/`, `docs/drafts/`, `logs/e2e_session/`.
2. The same, narrowed and read line by line:
   `git grep -n -E 'CD-07|run identity|checkpoint_sha256|flagged_model|unbound|parent_run_id|parent-run-id|diagnostics_run_id|TODO-WHY|X-W3|unit W3' -- docs/PROJECT_FILE_CATALOG.md 'docs/TB_RULING_*' 'docs/TB_RULING_REQUEST_*' 'docs/PROPOSAL_*' 'docs/CHAPTER_*' 'docs/*_AUDIT.md' docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md`
3. `git grep -l -E '<pattern of 2>' -- docs/exchange docs/drafts logs/e2e_session`, then
   `git grep -n -E 'CD-07|run identity|diagnostics_run_id' -- <the ten drafts and exchange files it listed>`.
4. `git ls-files docs/drafts/loop_v1` and `find . -name 'DRAFT_WATCHER_CD07*'` (the CD-07 draft diff exists and is tracked).
5. Owner rulings (main tree, git-ignored, read only):
   `/bin/grep -l -i -E "run id|run_id|cd-07|binding|checkpoint|flag" ~/distributed_prng_analysis/logs/e2e_session/OWNER_RULING*.txt`,
   then read the four hits plus `OWNER_RULING_defaults_20261123.txt` and `OWNER_RULING_switch_registration_order_20261125.txt`.
6. Code: `git grep -n "diagnostics_run_id\|diagnostics-run-id\|diagnostics_run_started_at" -- '*.py' '*.json'`;
   `git grep -n "compare_models_summary.json"` (who writes the file the legacy check reads);
   `git grep -n "agent_metadata\|parent_run_id\|run_id" -- prediction_generator.py` (what Step 6 reads from the sidecar);
   `git grep -n "diagnostics_level\|diagnostics_run" -- agents/registry/` (registry state);
   `git grep -n "run_subprocess_comparison"` (second compare path).
7. Plan outside the repository, read in full for sections 6.3 to 6.9:
   `~/tfm_worktrees/_staging/ll78/PLAN_LAND_LL7_LL8_20261126.md`.

Result notes. Search 2 hits in the registers were almost all the word "unbounded" (not relevant). The relevant register hits
are `parent_run_id` in Chapter 6 and 7 and in `docs/PROPOSAL_Unified_Agent_Context_Framework_v3_2_8.md:60-62`, which say
Step 6 reads the sidecar's `agent_metadata.run_id`. `docs/TODO_CONSOLIDATED.md` had 0 hits. `docs/LEADS.md:891-897` holds
TODO-WHY-1. `logs/e2e_session/` does not exist in the worktree; the owner rulings were read in the main tree.

### 1.2 Documents that already decide, schedule or record the subject

| # | Document and place | What it records |
|---|---|---|
| D1 | `docs/drafts/ch14_v1/PLAN_CH14_DIAGNOSTICS.md:174` (CD-07) | Step 5 stamps a run id and start time into every report; WATCHER passes the run it just finished; the check returns `stale` for another run's report. WATCHER line is hand-landed, before-commit review. |
| D2 | `docs/drafts/loop_v1/READY_LOOP_DIAG_2_20261123.md:49-52, :80-84, :98, :135, :222-228` | CD-07 built at level `full` only; key named `diagnostics_run_id` (not `run_id`, which the S88 summary uses); manifest keys declared `null` so argv is unchanged; "set by WATCHER only"; limit 2: without WATCHER's id an old consistent set is not caught. |
| D3 | `docs/drafts/loop_v1/DRAFT_WATCHER_CD07_20261123.diff` (tracked) | The draft WATCHER hunks: fill the id when the manifest declares it, keep it on `self`, pass it to the health call, and do not reset the skip registry on `stale`. Not applied. |
| D4 | `docs/drafts/loop_v1/READY_LAND_CH14_DIAG_20261125.md:345-347` | Owner items still open: the CD-07 WATCHER line and the Step 5 manifest entry for `--diagnostics-level`. |
| D5 | `docs/CHAPTER_6_AUDIT.md:134-139` (section 3 item 2) and `:179-186` (B2) | Compare mode drops `--output-dir` (and `--k-folds`, `--enable-vmap`, `--persistent-workers`); the archive and the winner restore go through `models/reinforcement`. |
| D6 | `PLAN_CH14_DIAGNOSTICS.md:173` (CD-06, B6) and `:187` (CD-05, B3+B4) | The legacy check reads a compare summary file name no code writes; tree families get a report only at level `full`. |
| D7 | `docs/exchange/from_beta/X-W2-HOOK_R2_BETA_REVIEW.md` Q3, Q4, Finding 1 | Verified: compare publishes under `models/reinforcement` while the flag hashed another dir; a stale report plus a same-family checkpoint gave a flag naming the wrong model; archived sidecars can point at the canonical path. |
| D8 | `docs/exchange/to_beta/X-W2-HOOK_R2_ALPHA_CLOSING_NOTE.md` (Q3/Q4 line) | Owner ordered the real binding: run id from WATCHER to Step 5 and the health check (CD-07); Step 5 writes run id and checkpoint sha256 into report and sidecar; compare archive records each family's own sha256; a design note goes to Beta first. |
| D9 | `docs/drafts/unit_w/READY_W2_20261127.md:38-63` (Item 4), `:65-72` (Item 5), `:74-82` (Item 7) | What each side carries today; a second flag line would break `read_flags` counting; owner accepted 2026-11-27 that the returned flag carries the identity record with the switch off, flag line on disk unchanged. |
| D10 | `~/tfm_worktrees/_staging/ll78/PLAN_LAND_LL7_LL8_20261126.md` sections 6.4, 6.6 items 3 and 5 | The flag needs run ID, `models_dir` and checkpoint sha256; "the report's checkpoint sha equals the flag's"; switch off is today. |
| D11 | `docs/exchange/from_beta/X-W1-RUNNER_R1_BETA_REVIEW.md:62` | "Flag-time identity remains the recorded W2 gap." |
| D12 | `OWNER_RULING_defaults_20261123.txt` item 3 | Owner yes: full diagnostics on by default once the manifest entry exists (flips only when that work lands, with his yes). |
| D13 | `OWNER_RULING_switch_registration_order_20261125.txt` | Order: registry repair, then switch registrations in the registry, then the hand-landed manifest entries. |
| D14 | `OWNER_RULING_benching_switch_20261126.txt` 07:05-07:10 | A failing family is flagged for a human or AI to check, never skipped. |
| D15 | `prediction_generator.py:321-325`; `docs/CHAPTER_7_PREDICTION_GENERATOR.md:1150-1151` | Step 6 reads the sidecar's `agent_metadata.run_id` as its parent run id. This must not change. |

### 1.3 Label for each item this note raises

| Item | Label |
|---|---|
| WATCHER makes one run id per Step 5 dispatch and passes it to Step 5 and to the health check | ALREADY RECORDED (D1, D3, D8) |
| The id travels on `run_step`'s returned result, not on `self`; WATCHER always overwrites any id a caller passed | NEW (refines D3, answers D7 Q3's invocation-local point) |
| The stale guard and the `stale_report_action` setting | ALREADY RECORDED, built (D1, D2; `thc:904-962`) |
| A `stale` result does not reset the skip registry | ALREADY RECORDED (D3 hunk 2) |
| Run id stamped into reports at level `legacy` too, when the id is given | NEW (D2 built it at `full` only) |
| Sidecar block `step5_identity` with run id, checkpoint sha256 and the report sha256 | NEW design of an ordered item (D8) |
| Per-run identity file `diagnostics_outputs/step5_identity_<run id>.json`, one writer | NEW |
| Compare: per-family identity taken from each child's own sidecar, not from whatever is in the canonical dir | ALREADY RECORDED as an order (D8); design NEW |
| Compare ignores `--output-dir` (B2) | ALREADY RECORDED (D5). Status. Not fixed here. |
| Archived sidecar points at the canonical path; the runner resolves the canonical file first | ALREADY RECORDED (D7 Q4). Runner fix: NEW |
| Identity fields written on the one flag line (option A) | CHANGES A RULING: plan 6.6 item 5 as amended by the owner on 2026-11-27 (D9 Item 7: "the flag line on disk is unchanged"). Needs the owner. |
| Identity in a separate file keyed by flag line (option B) | NEW, no ruling changed |
| Registry entries before the manifest entries | ALREADY RECORDED (D13) |
| Full level by default | ALREADY RECORDED (D12). Not needed by this design. |
| The legacy check's compare summary file has no writer | ALREADY RECORDED (D6). Status. |
| At level legacy no tree family writes a report | ALREADY RECORDED (D6, CD-05). Status. |
| Today an old report can be graded and counted (audit C3) | ALREADY RECORDED (D1). Status. |

### 1.4 Not searched

The host and Proxmox, the rigs, ser8 archives, the private repository, project files outside the repository (except the
plan in 1.1 item 7), and any live `diagnostics_outputs/` or `models/` on VM101. No real Step 5 output was read.

---

## 2. As-built facts (on `14815a7`)

### 2.1 Run identity on each side today

| Side | What identity it carries | Where |
|---|---|---|
| Diagnostics report, level `legacy` (the default) | None. Only `model_type`, `generated_at` (local time), `schema_version`. | `td:90` (default level), `td:507-532` (`save`), `td:523` |
| Diagnostics report, level `full` | `diagnostics_run_id`, `diagnostics_run_started_at` | `td:454-460` (`_add_full_report_fields`) |
| Where the run id comes from at `full` | `--diagnostics-run-id` or generated `s5-<UTC second>-<8 hex>` | `td:134-139` (`new_run_id`), `td:142-164`; at legacy the options bundle is `None` (`td:148-150`) |
| Child flags for the id | Only at `full`; `[]` at legacy | `td:167-181` (`[]` at `:169-170`) |
| Trial worker report (NN, trees) | Run id only at `full` | `tst:1008-1023` (`_diag_opts` only when level is `full`), `tst:934` (arg) |
| Sidecar `best_model.meta.json` | `agent_metadata.run_id = "step5_<local start, to the second>"`, `parent_run_id` (CLI only). No diagnostics id, no checkpoint sha256, no report reference. | `mpo:1359-1364` (inline writer), `mpo:2941` and `:2997`, `:3043-3049` (existing-checkpoint writer), degenerate writer from `mpo:3075` |
| Compare per-run summary `compare_models_summary_<S88 id>.json` | S88 run id (UTC to the second, `mpo:79-81`, `:224`), per family `returncode`, `archived_artifacts`, `score_best_effort`. No sha256. | `mpo:239-247`, `:322-326`, `:365-368` |
| Compare pointer `compare_models_summary_latest.json` | S88 id plus `diagnostics_run_id`, at level `full` only | `mpo:369-376` |
| Compare archive `models/reinforcement/compare_models/<S88 id>/<family>/` | Copies of whatever `best_model.*` files are in `models/reinforcement` after each child, whatever the child's return code | `mpo:312-318`, `_s88_copy_if_exists` `mpo:87-92` |
| WATCHER | Keeps the Step 5 `output_dir` it asked for on `self._step5_output_dir`; makes no run id | `wa:1035`, `wa:2468` |
| Health check result and flag | `_flagged_model` always returns `models_dir: None`, `checkpoint_sha256: None`, `flagged_model_error: "unbound: ..."`; nothing is hashed | `thc:1246-1266`, `thc:1313-1315` |

### 2.2 What WATCHER passes

- **To Step 5:** the manifest's `default_params` (`rj:173-186`: `compare_models: true`, `output_dir: models/reinforcement`,
  `enable_diagnostics: true`, `model_type: catboost`, ...), plus caller params, but only keys the manifest declares
  (`wa:2033-2040`). `None` and `""` values are left off the command line (`wa:2433-2434`). No diagnostics level and no run
  id are declared, so WATCHER's Step 5 runs at level `legacy` with no run id.
- **To the health check:** `check_training_health(models_dir=self._step5_output_dir)` only (`wa:3200`). No
  `expected_run_id`, no `run_started_at`.
- The health check runs only when the step is 5, the decision was `proceed`, and Step 5 was not freshness-skipped
  (`wa:3188-3197`). A freshness skip returns `{"success": True, "skipped": True}` before any command is built (`wa:2013`).

### 2.3 The stale guard and when it is active

- `report_freshness` (`thc:904-947`): the caller's `expected_run_id` decides first (equal: fresh; other id: stale); then
  the compare pointer's id; then the caller's `run_started_at` (report's own start time, else the file's mtime); else
  `unverified`.
- **Legacy grading path** (the live one: `watcher_policies.json` has no `grading_rules` key, checked with a python read):
  the guard runs **only when the caller passes `expected_run_id` or `run_started_at`** (`thc:317-326`, test at `:319`). WATCHER passes
  neither, so on `14815a7` the guard is off in the live path.
- **Fixed grading path:** the guard always runs, with the pointer id when there is one (`thc:816-822` winner, `:835` each family).
- A stale result uses `stale_report_action` (default `PROCEED_WITH_NOTE`), never touches the skip registry and is never
  archived (`thc:949-962`). WATCHER today still calls `reset_skip_registry` for any non-critical `PROCEED*`
  (`wa:2586-2588`); D3 hunk 2 changes that for `stale`.
- **Legacy S93 override:** the legacy check renames the report's family to the winner named in
  `diagnostics_outputs/compare_models_summary.json` (`thc:328-347`, `:346`). `git grep` finds no code that writes that
  file name (only old patch scripts and a test mutant). Compare writes `compare_models_summary_<S88 id>.json`. So the
  override fires only on a leftover file (D6).

### 2.4 Where compare drops `--output-dir`, and how the archive is filled

- `mpo:3194-3195` defines `--output-dir` (default `models/reinforcement`).
- `mpo:3283-3307`: with `--compare-models` and no `S88_COMPARE_MODELS_CHILD`, `main` returns
  `_s88_run_compare_models({...})`. The dict has no `output_dir`, no `parent_run_id`, no `k_folds`. This is where the
  value is dropped.
- Each child command (`mpo:262-268`) is `--survivors --lottery-data --model-type m --trials N`, plus `--enable-diagnostics`
  and, at level `full` only, the diagnostics child flags (`mpo:269-278`), plus NN flags. No `--output-dir`, so the child
  writes `models/reinforcement` (argparse default).
- Families run in the order `neural_net, lightgbm, xgboost, catboost` (`mpo:148`), minus skipped ones.
- After each child, the parent copies six file names from `models/reinforcement` into the family's archive dir
  (`mpo:312-318`). It does not look at the return code first. If the child wrote nothing, the archive gets the previous
  child's or an older run's files. The score comes from the archived sidecar (`mpo:320`).
- The winner is the highest score (`mpo:328-351`); its archived files are copied back into `models/reinforcement`
  (`mpo:354-363`). The parent returns 0 whatever the children did (`mpo:379`).
- Sidecars carry `checkpoint_path` as written by the child, which is the canonical path
  (`mpo:1306`, `:1338`; `mpo:3022-3024`). `ib:156-165` loads the written path first if it exists, else
  `<models_dir>/<basename>`. So an archived sidecar resolves to the canonical file while that file exists.

### 2.5 Which report is graded, by path (reading, not run)

- Optuna folds get no `--enable-diagnostics` (`mpo:1962-1964`).
- **NN final model:** the NN subprocess `tst` gets `--enable-diagnostics` (`mpo:2665-2670`); it saves the checkpoint
  (`tst:774`) and then the report `neural_net_diagnostics.json` and the canonical copy (`tst:787-795`). `mpo` renames
  `neural_net_trial-1.pth` to `best_model.pth` (`mpo:2695-2703`); the bytes do not change.
- **Tree final model:** inline trainer (`mpo:1903-1907`). A report is written only at level `full`
  (`_ch14_emit_inline_tree_report`, `mpo:2810-2843`, writes `{family}_diagnostics.json` and the canonical copy). At legacy
  no tree report is written (D6).
- Every report writer also copies itself to `diagnostics_outputs/training_diagnostics.json` (`tst:98-106`, `mpo:2839`).
- So at level legacy in WATCHER's default compare run, the canonical report is the NN child's report from this run, if
  NN ran. In a single-model tree run at legacy, the canonical report is whatever an earlier run left.
- The sidecar is written after the report on both final paths (`mpo:1907` then `:1926`; NN report inside `tst`, sidecar by
  `mpo` after the subprocess returns).

### 2.6 Retries

- On a health `RETRY`, WATCHER builds new params from the caller's `params` (not from `final_params`), invalidates Step 5
  freshness, deletes `diagnostics_outputs/training_diagnostics.json` (S99), and loops back to Step 5 (`wa:3203-3238`,
  `:2663-2686`). The next loop pass calls `run_step(5, ...)` again (`wa:3127`), which builds a new command.
- A Step 5 that exits non-zero returns `success: False` (`wa:2480-2487`); the health check does not run for it.

### 2.7 How the flag line is read

- `_raise_why_flag` appends one JSON line (`thc:1303-1307`) and only then adds the unbound record to the returned dict
  (`thc:1313-1315`).
- `read_flags` counts every line of a `flag_id` as one more repeat and moves `last_flagged_at` (`wrf:133-135`). It reads
  only named keys, so extra keys on a line are ignored (`wrf:120-140`).
- The runner takes the **latest** line of the flag that has `checkpoint_sha256` (`wrr:202-216`), then looks for that sha
  in walk-forward frozen dirs, then `--models-dir`, then the flag's `models_dir` (`wrr:219-257`). For the flag's dir it calls
  `current_checkpoint`, which uses `ib.resolve_checkpoint_path`, which prefers the sidecar's written (canonical) path
  (`wrr:174-199`).

---

## 3. The binding rule (plain language)

The flag may name a checkpoint only when one chain of matching records links it to the report that was graded. WATCHER
makes a fresh run id just before it starts Step 5. The graded report must carry exactly that run id, and its own family
must be the flagged family. Step 5's identity record for that run id must list the flagged family. That entry must hold
the sha256 of the very report bytes that were graded. The family's sidecar must carry the same run id, the same family and
a checkpoint sha256. The checkpoint file found next to that sidecar must hash to that same sha256. If every link holds, the
flag names that checkpoint, its folder and its sha256. If any link is missing or differs, the flag says "unbound" and names
the first link that failed. A flag is never delayed or dropped because of this.

Short form: **WATCHER's run id = report's run id = identity record's run id = sidecar's run id; flagged family = report's
own family = sidecar's family; sha256(graded report) = recorded report sha256; sha256(checkpoint file) = recorded checkpoint
sha256.**

---

## 4. The smallest set of changes

Principle: every new value is optional. When WATCHER passes no run id, every file, argv and return value is exactly as on
`14815a7`.

| # | File | Function | What is added | When the new value is absent | Who lands it |
|---|---|---|---|---|---|
| C1 | `agents/registry/parameter_registry.py` | Step 5 registry (`:357`) | `diagnostics_run_id`, `diagnostics_run_started_at` as STRING, default `None`, "set by WATCHER only, never proposed by an LLM" (text from D2:80-84) | Not used until the manifest declares them | agent (registry first, per D13) |
| C2 | `agent_manifests/reinforcement.json` | `default_params` | `"diagnostics_run_id": null, "diagnostics_run_started_at": null` (D2:98) | `null` is skipped by the builder (`wa:2433`): argv byte-identical | **owner** (hand-landed dir) |
| C3 | `agents/watcher_agent.py` | `run_step` | When `step == 5` and the manifest declares the key: make `(id, started) = training_diagnostics.new_run_id()` after the merge and before the command is built; **always overwrite** any value a caller passed; log it. Put `step5_run_id` and `step5_run_started_at` on every dict `run_step` returns after that point (success, failure, timeout). | Key not declared: nothing generated, returns unchanged | **owner** |
| C4 | `agents/watcher_agent.py` | `run_pipeline` health block (`wa:3200`) | `check_training_health(models_dir=..., expected_run_id=results.get("step5_run_id"), run_started_at=results.get("step5_run_started_at"))`. Read from this pass's `results`, not from `self`. | Both `None`: the call behaves as today | **owner** |
| C5 | `agents/watcher_agent.py` | `_handle_training_health` (`wa:2586-2588`) | Do not reset the skip registry when severity is `stale` (D3 hunk 2) | No `stale` result exists without an id: unchanged | **owner** |
| C6 | `training_diagnostics.py` | `save` and a new helper `identity_child_flags(run_id, started)` | When the object was given a run id, write `diagnostics_run_id` and `diagnostics_run_started_at` into the report **at any level**. The helper returns the two flags when an id is given, at any level. | No id: report bytes as today (legacy identity gates still hold) | agent |
| C7 | `train_single_trial.py` | `main` (`tst:1008-1023`) and the report writers | At legacy, if `--diagnostics-run-id` is given, give it to the report object. For NN only, also write `checkpoint_sha256` of the file it just saved (`tst:774`) into the report. | No id: unchanged | agent |
| C8 | `meta_prediction_optimizer_anti_overfit.py` | compare dispatch (`mpo:3285-3307`), child command (`mpo:262-278`), NN subprocess command (`mpo:2665-2670`) | Thread `--diagnostics-run-id` and `--diagnostics-run-started-at` to compare children and to the NN subprocess at any level, when given | No id: no new flags | agent |
| C9 | `meta_prediction_optimizer_anti_overfit.py` | all four sidecar writers: `MultiModelTrainer.save_model` (`mpo:1266-1377`), `_save_existing_checkpoint_sidecar` (`mpo:2980-3073`), `_save_degenerate_sidecar` (`mpo:3075`), via `save_best_model` | When an id is given: a new top-level block `step5_identity = {run_id, run_started_at, model_type, checkpoint_file (basename), checkpoint_sha256, report_path, report_sha256, report_run_id}`. The report is `diagnostics_outputs/{family}_diagnostics.json`; its sha is recorded only if it carries this run id, else `report_sha256: null` with `report_reason`. Degenerate: checkpoint fields `null`. `agent_metadata` is not touched. | No id: no block; sidecar as today | agent |
| C10 | `meta_prediction_optimizer_anti_overfit.py` | single-model `main` after `optimizer.run` (`mpo:3382`), only when not a compare child | Write `diagnostics_outputs/step5_identity_<run id>.json`: `{run_id, mode: single, families: {F: {sidecar_path (absolute), sidecar_sha256, ...step5_identity fields}}}` | No id: no file | agent |
| C11 | `meta_prediction_optimizer_anti_overfit.py` | `_s88_run_compare_models` (after `mpo:318`, and at the end) | After each child: read the **archived** sidecar; accept it for family `m` only if `step5_identity.run_id == id` and `model_type == m` and the archived checkpoint file (by basename, in the archive dir) hashes to `checkpoint_sha256`; else record `identity: null` with the reason (for example "child exit 1, archive holds another family's sidecar"). At the end write the same identity file with `mode: compare`, the S88 id, the winner, and one entry per family pointing at the archive dir. | No id: no reads, no file; archive and restore exactly as today | agent |
| C12 | `training_health_check.py` | report loading (`thc:298-301` legacy; `_load_json` on the fixed path) | Keep the sha256 of the exact bytes that were parsed, and the report's own `model_type` before the S93 override (`thc:346`); put both and the graded path into `run_ctx` | Kept only in `run_ctx`; no output changes | agent |
| C13 | `training_health_check.py` | `_flagged_model` replaced by `_bind_flagged_model(family, run_ctx)` | Applies the rule in section 3, in order, and returns `binding: bound|unbound`, `models_dir`, `checkpoint_path`, `checkpoint_sha256`, `step5_run_id`, `report_sha256`, `flagged_model_error`. Hashes the checkpoint only after every cheaper link has passed. Never raises. | `expected_run_id` absent: the unbound record exactly as today ("WATCHER passes no run id") | agent |
| C14 | `training_health_check.py` | `_raise_why_flag` (`thc:1268-1317`) | Compute the binding **before** the append, and store it per the chosen option (below) | No id: line and return value as today | agent |
| C15 | `scripts/diagnostics/why_report_runner.py` | `checkpoint_by_sha` flag-dir branch (`wrr:243-252`) | For the flag's own dir: try `<flag models_dir>/<basename>` (or the flag's `checkpoint_path`) first and accept only on a sha match, before `current_checkpoint` | Flag without these fields: as today | agent |
| C16 | `tests/test_flag_model_binding.py` (new) and updates to `tests/test_why_report_auto.py`, `tests/test_why_report_flags.py` | gates and mutants of section 5 | | | agent |

### 4.1 Answers to the specific coverage points

- **One run id per Step 5 run, made by WATCHER (CD-07).** C3 makes it right before the command is built, in `run_step`.
  A freshness skip returns at `wa:2013`, before this point, so a skipped Step 5 has no id. The id is carried on the
  returned dict, so the health check reads the id of the dispatch it is checking, not a value left on the instance.
- **Every diagnostics level.** Yes for the run id (C6, C7): it is two small keys, written only when an id is given.
  Raising every run to level `full` is not needed. Its cost: different report content, the NN gradient probe and tree
  staged predictions on every run (time and peak RSS not measured; [UNVERIFIED]); it is a separate owner yes (D12) that
  waits for its own manifest entry. Note: at legacy no tree report exists, so at legacy only the NN family can ever be
  bound. That is the same set of families that can be graded fresh today.
- **Checkpoint sha256 in the report.** Only the NN report can carry it, because `tst` writes the report after the
  checkpoint (`tst:774` then `:792`). The inline tree report is written before the sidecar and checkpoint save
  (`mpo:1907` then `:1926`). Writing the report after the save would change the order of Step 5's file writes, so the
  design does not do that. Instead the sidecar records the report's sha256 (C9), which ties the two in the other direction.
- **Compare, per family, from the child's own output.** The child writes `step5_identity` into its own sidecar (C9).
  The parent accepts an archive entry only if that sidecar names this run and this family and the archived checkpoint
  hashes right (C11). An archive filled from another child's leftovers fails these checks and is recorded as `null`.
- **Health retries.** A retry goes back through `run_step`, so it is a new Step 5 run with a **new id**. The id of the
  failed attempt is never reused. The skip count still spans attempts (it is per family, `thc:1103-1184`); the flag
  names the checkpoint of the attempt whose report raised it.
- **Single-model and compare paths.** Single: identity file from `main` (C10), sidecar in the `--output-dir` Step 5
  actually used. Compare: identity file from the parent (C11), each family pointing at its archive dir. The health check
  never guesses a dir from WATCHER's `output_dir`; it reads the identity file named by the run id.
  The second compare path (`run_subprocess_comparison`, `mpo:1219`, saving at `mpo:1720`) is not on WATCHER's route
  (reading); it gets the sidecar block through `save_best_model` and no identity file, so its flags stay unbound.
- **Where the identity is stored for the flag.**
  - **Option A (proposed): on the one flag line.** C14 computes the binding before `thc:1307`, so the fields go on the
    single line that is written anyway. `read_flags` ignores extra keys (`wrf:120-140`), so `repeats` and
    `last_flagged_at` stay right. The runner already reads `checkpoint_sha256` and `models_dir` from lines
    (`wrr:202-216`). Cost: with the owner patch installed, the line on disk gains keys whether the automatic switch is on
    or off. That changes D9 Item 7 ("flag line on disk unchanged"). **Needs the owner.** Gate `W2.flag_line_base_unbound`
    is rewritten to "no id given: line as base; id given: line as base plus these keys".
  - **Option B (alternative): a separate append-only file** `diagnostics_outputs/why_report_flag_models.jsonl`, one
    record per flag line, keyed by `(flag_id, flagged_at)`, written right after the flag line. The flag line stays
    byte-identical. `read_flags` is untouched. The runner's `flagged_model` reads this file instead of the flag lines.
    Cost: two files to keep in step; a crash between the two writes leaves a flag with no identity (which reads as
    unbound, the safe side); one more file with no retention rule.

---

## 5. Every way it can still name a wrong model, and the test that catches it

All tests are CPU only, on synthetic files and stubbed children, with a mutant for each gate that removes the check and must
turn the gate red.

| Case | How it goes wrong today or could go wrong | What stops it | Test (gate / mutant) |
|---|---|---|---|
| Stale report | An old NN report sits in the canonical file; a single-model tree run writes no report (2.5); the check grades the old one | Report run id is not WATCHER's id: unbound (and CD-07 returns `stale`, not graded) | B.stale_report: old report with id X, run id Y gives `stale` and no flag; also a report with no id at all. Mutant: skip the run-id link |
| Stale sidecar | A child fails before saving; the canonical sidecar is the previous family's or an older run's | Sidecar run id or family differs: archive entry `null` (C11), unbound | B.stale_sidecar: child stub exits 1 and writes nothing. Mutant: accept archive without the run-id check |
| Compare winner vs non-winner | NN is flagged; the canonical dir holds the catboost winner | Binding uses the NN entry of the identity file (archive dir), never the canonical dir | B.non_winner: four stub children, catboost wins, NN critical; flag names the NN archive file and its sha. Mutant: use the canonical sidecar |
| S93 override renames the family | A leftover `compare_models_summary.json` renames the NN report to the winner (`thc:346`) | Flagged family must equal the report's own family read before the override (C12) | B.override_family: leftover file naming catboost, NN report. Unbound. Mutant: compare with the overridden family |
| Relocated dir | WATCHER asks for `output_dir=X`; compare writes the canonical dir anyway (B2) | The health check reads the identity file, which records where files really are | B.relocated: compare with `output_dir` override; flag `models_dir` is the archive dir, not X. Mutant: use WATCHER's `models_dir` |
| Archived sidecar points at the canonical path | `ib:160-161` picks the canonical file, which holds another model | Binding hashes `<sidecar dir>/<basename>`; runner C15 does the same, sha must match | B.archive_canonical_path: canonical file with other bytes exists. Mutant: resolve through `resolve_checkpoint_path` first |
| Two WATCHER runs close together | S88 ids and sidecar `run_id` are to the second (`mpo:79-81`, `:2941`); two compare runs in one second share one archive dir | WATCHER ids carry 8 random hex (`td:134-139`); every archive entry is checked against the run id and sha at write time and again at flag time | B.same_second: two runs, same S88 second, second overwrites archive; first run's binding becomes unbound, never names the second's file. Mutant: drop the flag-time re-hash |
| Step 5 fails after writing a checkpoint | Non-zero exit; checkpoint and sidecar with id E1 left behind | No health check for a failed step (`wa:2480-2487`); the next dispatch has id E2, so E1 files never match | B.failed_then_retry: stub writes then exits 1; retry stub writes E2; flag (if any) names E2 only. Mutant: reuse the id across dispatches |
| Compare child fails, parent exits 0 | `mpo:379` returns 0; that family's archive holds leftovers | C11 records `null` for that family | Covered by B.stale_sidecar |
| Skipped Step 5 | Freshness skip; no new training | No id is made (`wa:2013` is before C3); the health check is not called (`wa:3188-3197`) | B.skipped: skip result; health stub not called; no id generated. Mutant: generate the id before the freshness return |
| Resumed pipeline | Start at Step 6 after an earlier Step 5 | No Step 5 dispatch in this invocation, so no id and no health check; nothing on `self` is read | B.resume_at_6 on a fresh instance and on an instance that ran Step 5 before. Mutant: read the id from `self` |
| Health retry | Retry deletes the canonical report (S99, `wa:3229-3236`); a retry child may fail to write a report | New id per retry; the old report is gone or carries the old id | B.retry_new_id: two dispatches, two ids, the second health call gets the second. Mutant: keep the first id |
| Report rewritten between grade and flag | A second process rewrites the canonical report | sha256 of the bytes graded (C12) must equal the recorded report sha | B.report_swapped: rewrite between load and bind. Mutant: hash the file again instead of the graded bytes |
| A trial report carries the run id | If any trial path wrote a report with the run id after the final one, the sidecar could record a trial's report | Today no Optuna fold gets `--enable-diagnostics` (`mpo:1962-1964`); the sidecar records the report as it is at save time; NN reports also carry the checkpoint sha (C7) | B.one_report_per_family: tiny CPU Step 5 with 2 trials counts report writes with the id: exactly one per family. Residual risk named in 8 |
| Id passed by a caller | An operator or LLM passes `diagnostics_run_id` in params; two dispatches share it | C3 always overwrites | B.caller_id_overwritten. Mutant: fill only when empty (as in D3) |

Plus the "absent is today" gates: no id given, so argv, reports, sidecars, compare summaries, archive contents, health
results and flag line are byte-identical to `14815a7` (timestamps excepted, as in the existing legacy identity gates).

---

## 6. What does NOT change

- **Step 5 training results:** no training code is touched. Only file writes after training (sidecar block, identity
  file, two report keys, two child flags). Gate: the existing bit-identical NN and tree identity tests, run with an id.
- **Model selection:** the compare winner rule (`mpo:328-351`) and the restore (`mpo:354-363`) are unchanged. The score
  read (`_s88_extract_score`, `mpo:94-139`) finds `training_metrics.r2` first; the new block is named `step5_identity`
  and holds no key containing "r2", so the fallback scan cannot pick it up.
- **Argv when the new values are absent:** byte-identical (manifest `null`s are skipped, `wa:2433`).
- **Step 6:** reads `agent_metadata.run_id` (`prediction_generator.py:321-325`) and loads through `ib.resolve_checkpoint_path`
  (`ib:156-165`). Neither is changed. The sidecar keeps `schema_version` 3.4.0 and every existing key.
- **Chapter 13:** reads the sidecar's `model_type` and friends (`chapter_13_orchestrator.py:594-600`); no existing key changes.
- **The automatic why-report switch:** stays off; the hook is untouched.
- **Benching, skip counts, grading rules:** unchanged, apart from C5 (`stale` never resets a skip count) and CD-07 itself:
  once WATCHER passes the id, the legacy path stops grading old reports. That is CD-07's recorded purpose (D1), and it
  does change WATCHER actions in the case "old report left behind" (audit C3). It must be stated at landing.

---

## 7. Landing plan

**Commit 1 (one commit, owner lands; non-owner files plus the owner patch record):**
`agents/registry/parameter_registry.py` (C1), `training_diagnostics.py` (C6), `train_single_trial.py` (C7),
`meta_prediction_optimizer_anti_overfit.py` (C8 to C11), `training_health_check.py` (C12 to C14),
`scripts/diagnostics/why_report_runner.py` (C15), the test files (C16), and
`docs/drafts/unit_w/W3_OWNER_LAND.patch` (record).

**The owner patch** (`W3_OWNER_LAND.patch`, applied by hand in the same sitting):
- `agent_manifests/reinforcement.json`: two `null` keys in `default_params` (C2).
- `agents/watcher_agent.py`: the id in `run_step` and on its returns (C3), the health call (C4), the `stale` line (C5).
  About 15 lines.
- `watcher_policies.json`: no change.

`step 5` is a before-commit review class for WATCHER (launch path), and the `mpo` changes touch Step 5's file writes, so the
code itself also goes to Beta before commit (AGENTS.md review classes; D1 "WATCHER line before-commit").

**Proof (same form as W2):** fresh clone of main at landing time, the non-owner files copied in with sha256 matching, CPU
only (`CUDA_VISIBLE_DEVICES=""`), one suite at a time (memory: concurrent suites flake).
- Phase A, without the owner patch: new binding suite (owner-dependent gates SKIP, shown in the summary line), plus
  `test_why_report_auto`, `test_why_report_runner`, `test_why_report_flags`, benching suite and mutants, Chapter 14 wave 1,
  wave 2 and landing-defaults suites (legacy identity gates), LL feature settings. All green; argv identical.
- Phase B, owner patch applied in the clone: the same, plus the WATCHER gates of section 5 on a faked `run_pipeline`, and an
  installed-code gate that reads the installed WATCHER with a negative control (removing C4 turns it red), as `W2.installed_hook`.
- Restart any running WATCHER from the finished tree before relying on it (D7 Q6).
- Untracked new test files make Gate 22 red until landed (known; memory note on Gate 22).

---

## 8. Open questions

**For the owner (his choices):**
1. Option A (identity on the one flag line; changes your 2026-11-27 acceptance that the line on disk is unchanged) or
   option B (separate identity file, line unchanged)? Alpha proposes A.
2. Land the registry entries (C1) for `diagnostics_run_id` now, ahead of the other Step 5 switch registrations, given your
   2026-11-25 order (registry, then manifest)?
3. Once WATCHER passes the id, the legacy check stops grading old reports (CD-07). An old report left behind will give
   `stale` and `PROCEED_WITH_NOTE` instead of a grade. Accept that as part of this landing?
4. Should the `diagnostics_level: full` manifest entry (your 2026-11-23 yes) land with this, so tree families can be bound
   too? Not required by this design.
5. The identity files grow by one small file per Step 5 run. Keep them all, or prune with the diagnostics history cap?

**For Beta:** see the request draft (`BETA_REQUEST_DRAFT_X-W3-BINDING_R1.md`). In short: is the rule sufficient; which
wrong-model path is missing; is one id per Step 5 dispatch the right unit; is "run id at every level, sha in the sidecar"
right; what in Step 5 must this not disturb; is hashing the checkpoint inside the health check acceptable (it is
synchronous before Step 6; sizes on the host are [UNVERIFIED]); and is the "trial report with the run id" residual risk
closed by the one-report-per-family gate.

**Limits.** Nothing here was run. Every Step 5 path statement is a reading of `14815a7`. A bound flag still needs a graded
walk-forward block for that checkpoint before the runner can explain it (plan 6.5, recorded `no_graded_block`). This note
measures nothing about pool hits against `k/1000`; it only decides which model a flag may name.
