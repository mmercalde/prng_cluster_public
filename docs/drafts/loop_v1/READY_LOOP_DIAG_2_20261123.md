# READY · TASK_LOOP_DIAG_2 (window c14d, worktree ch14-diag) · CH14-DIAG wave 2 · 2026-11-23

## Prior record

**Searches run** (worktree at `41ad509`; `/bin/grep -rl`, so `.json` and ignored files are not skipped).
Scope: `docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`, `docs/TB_RULING_REQUEST_*`, `docs/PROPOSAL_*`,
`docs/CHAPTER_*`, `docs/*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/LEADS.md`, `docs/exchange/`.

| Term | Files found |
|---|---|
| `training_health_log`, `stale_report`, `expiry_mode`, `history_prune`, `dead_unit`, `llm_prompt_fields`, `diagnostics_run_id`, `advisor_reads_training_health` | 0 each |
| `unused_feature_pct` | `CHAPTER_14_AUDIT.md`, `CHAPTER_14_TRAINING_DIAGNOSTICS.md`, `CHAPTER_CORRECTIONS_REPORT_20261111.md` |
| `skip_duration_hours` | the three above, plus `CHAPTER_14_IMPLEMENTATION_PROPOSAL_S69.md` |
| `R-66`, `R-67`, `R-68`, `R-69`, `R-74` | `CHAPTER_CORRECTIONS_REPORT_20261111.md` |

The full search set behind this wave is in the plan, `docs/drafts/ch14_v1/PLAN_CH14_DIAGNOSTICS.md` §Prior record.

**Also read:**
- The lead's task files: `TASK_LOOP_DIAG_2.txt`, `OWNER_DIRECTION_autonomy_switches_20261123.txt`, `TASK_SWITCH_REGISTER(_ADDENDUM_1).txt`, `NOTE_zeus_reboot_20261123.txt`.
- The existing switch machinery: `agents/registry/parameter_registry.py` (ParameterSpec, the Step 5 entry), `agents/manifest/agent_manifest.py`, `agent_manifests/reinforcement.json`, and WATCHER's command builder at `agents/watcher_agent.py:2012-2034` and `:2379-2422`.
- Window g's output, read only: `~/tfm_worktrees/e2e-grader/docs/drafts/loop_v1/READY_SWITCH_CHECK_20261123.md` and `SWITCH_CHECK_20261123.md`. It lists my wave 1 flags as "not registered" and "no run record" (rows S00245, S00246).

**Documents that decide or record these items:**
- B7–B10 and B16 = R-66–R-69 and R-74 (`CHAPTER_CORRECTIONS_REPORT_20261111.md`).
- D12 dead neurons (`docs/CHAPTER_14_AUDIT.md`).
- My plan's rows: CD-07 :174, CD-09 :175, CD-10 :176, CD-08 :177, CD-11 :178, CD-23 :189, CD-12 :195; owner questions Q1 and Q2 at :297-298.
- TODO-CH14-4: `READY_LOOP_DIAG_20261123.md:154`.

**Labels:**
- ALREADY RECORDED: CD-08, CD-09, CD-10, CD-11, CD-12 (R-74, R-67, R-69, R-68, R-66), and TODO-CH14-4 (my wave 1 note).
- NEW (as labelled in the plan): CD-07 and CD-23.
- CHANGES A RULING: none.
  - The CD-09 and CD-10 defaults stay at today's behaviour. Q1 and Q2 remain the owner's.
  - CD-23 keeps the old definition as the default, which differs from the plan (see "Choices made").

**Not searched:** the host, ser8 archives, the private repository, project files outside the repository, and Zeus host logs.

## Plain summary

I built wave 2 of my diagnostics plan (lanes A, B and E) plus TODO-CH14-4, in the ch14-diag worktree. All new behaviour sits behind declared settings, and every default is today's behaviour. Nothing is committed, nothing is landed, and no GPU or rig was used.

**Tests (CPU):**
- Wave 1 and wave 2 together: **73 of 73 tests pass** (36 + 37).
- **Wave 1 mutants: 22 of 22 killed**, re-checked against the changed code.
- **Wave 2 mutants: 29 of 29 killed.** Each one fails on its own tagged assertion, with the mutated line shown to have run.
- **Training is still bit-identical.** The NN weights and metrics match with diagnostics off, at the old dead-unit definition and at the new one, in both batch modes (`test_cd23_cd07_nn_bit_identical`). The wave 1 tree and NN identity tests still pass.

**What each item does now:**
- **CD-07, run identity and stale guard:**
  - At `--diagnostics-level full`, Step 5 creates one run id and start time and passes them to every child. They are stamped into every report and the compare-models pointer as `diagnostics_run_id` and `diagnostics_run_started_at`. I did not reuse `run_id`, because the compare summary already uses that key for its S88 id.
  - The health check now grades a report from another run as `stale`. A stale report never counts toward a model skip and is never archived.
  - The WATCHER line is a **draft diff only, not applied**: `docs/drafts/loop_v1/DRAFT_WATCHER_CD07_20261123.diff`. It passes `git apply --check`, and `agents/watcher_agent.py` is unchanged (sha256 `cc82f7c4…08f4`).
- **CD-08:** level-full reports carry `unused_feature_pct`, overall, per column class, and excluding constant columns. The fixed grading rules can grade it, behind a setting.
- **CD-09:** model skips can expire after `skip_duration_hours`, as the policy text says. Both readers use the same rule.
- **CD-10:** the health-check archive can be capped. Pruned files are moved (the default mode) or deleted. Chapter 13 `root_cause_*` files are kept unless a setting includes them.
- **CD-11:** one append-only log, `diagnostics_outputs/training_health_log.jsonl`, with one writer (the health check, every result). The Strategy Advisor reads it only behind a default-off setting.
- **CD-12:** the LLM prompt can be built from the fields the reports actually contain, behind a setting. No LLM is called in any test.
- **CD-23:** a dead-unit metric that can fire, behind a setting. A unit counts as dead when its pre-activation is ≤ 0 on every probe row. The old definition stays selectable and remains the default.
- **TODO-CH14-4:** the run-level column set now comes from the feature manifest's `run_context` alias. The manifest's hash is unchanged (`b2fabc87…860f`).
  - **There is no trial id anywhere.** Run H's Step 5 rows have only `seed`, `score`, `metadata`, `features` and `holdout_*` at the top level, and the manifest names none. So the `trial` column class stays "unavailable", and the report now gives that reason.

These are readings of how training went, not evidence about TFM, which is judged only by pool hits against k/1000 on unseen draws.

**After the 19:25 reboot.** Everything had been saved by 19:28. The final test run had finished: 73 passed, `pytest_w1w2_final2.log`. Only the wave 2 mutant run was in flight; it was re-run on cores 2,3 at nice 15 with 2 jobs, and 29/29 were killed. Nothing else was re-run.

## Settings added, in the existing registry and manifest form

### Step 5 CLI (`meta_prediction_optimizer_anti_overfit.py`; the same flags exist on `train_single_trial.py`, which only Step 5 calls)

All of these act at `--diagnostics-level full` only; at legacy none of them is read.

```python
# agents/registry/parameter_registry.py -> "meta_prediction_optimizer_anti_overfit.py".parameters
ParameterSpec(name="diagnostics_level", cli_flag="--diagnostics-level", param_type=ParamType.CHOICE,
              choices=["legacy", "full"], default="legacy",
              description="Chapter 14 report content; training identical at every level (wave 1)"),
ParameterSpec(name="gradient_probe_rows", cli_flag="--gradient-probe-rows", param_type=ParamType.INT,
              min_value=0, default=None,
              description="NN eval-mode probe rows (level full; None = worker default 512; 0 = no probe) (wave 1)"),
ParameterSpec(name="diagnostics_run_id", cli_flag="--diagnostics-run-id", param_type=ParamType.STRING,
              default=None, description="CD-07 run id stamped into reports and pointer; generated when absent. "
                                        "Set by WATCHER, never proposed by an LLM"),
ParameterSpec(name="diagnostics_run_started_at", cli_flag="--diagnostics-run-started-at",
              param_type=ParamType.STRING, default=None, description="CD-07 run start, UTC ISO; set by WATCHER"),
ParameterSpec(name="dead_unit_definition", cli_flag="--dead-unit-definition", param_type=ParamType.CHOICE,
              choices=["linear_zero", "always_negative_preact"], default="linear_zero",
              description="CD-23 NN dead-unit metric (report only)"),
ParameterSpec(name="unused_gradient_epsilon", cli_flag="--unused-gradient-epsilon", param_type=ParamType.FLOAT,
              min_value=0.0, default=None, description="CD-08 NN column unused below this mean |input gradient| (None = 1e-8)"),
ParameterSpec(name="feature_manifest", cli_flag="--feature-manifest", param_type=ParamType.STRING, default=None,
              description="TODO-CH14-4 feature manifest path (None = models/feature_manifest.json when present)"),
```

```json
// agent_manifests/reinforcement.json (hand-landed). Every default_params value is null, so the WATCHER
// builder skips it (watcher_agent.py:2425) and today's command line is byte-identical.
// No args_map entry is needed: the fallback underscore->hyphen (:2415) gives exactly these flags.
"default_params":   {"diagnostics_level": null, "gradient_probe_rows": null, "diagnostics_run_id": null,
                     "diagnostics_run_started_at": null, "dead_unit_definition": null,
                     "unused_gradient_epsilon": null, "feature_manifest": null},
"parameter_bounds": {"diagnostics_level": {"type": "categorical", "values": ["legacy", "full"], "default": "legacy"},
                     "gradient_probe_rows": {"type": "int", "min": 0, "default": null},
                     "dead_unit_definition": {"type": "categorical", "values": ["linear_zero", "always_negative_preact"], "default": "linear_zero"},
                     "unused_gradient_epsilon": {"type": "float", "min": 0.0, "default": null}}
```

**One difference from window g's fragment.** It proposes `{"diagnostics_level": "legacy"}`. That behaves the same, but it adds `--diagnostics-level legacy` to every Step 5 command. `null` keeps the command line byte-identical.

### Health-check, advisor and LLM settings (`watcher_policies.json` → `training_diagnostics.<key>`; absent key = default)

There is no CLI for these. WATCHER reaches them by reading the policy file (`training_health_check.py:_load_policies`). Written as ParameterSpec with an empty `cli_flag`:

```python
# proposed registry entry "training_health_check.py" (policy keys; cli_flag empty)
ParameterSpec(name="grading_rules", param_type=ParamType.CHOICE, choices=["legacy", "fixed"], default="legacy"),          # wave 1
ParameterSpec(name="early_stop_round_base", param_type=ParamType.INT, min_value=0, max_value=1, default=0),             # wave 1, fixed only
ParameterSpec(name="early_stop_min_rounds", param_type=ParamType.INT, min_value=1, default=2),                         # wave 1, fixed only
ParameterSpec(name="grade_scope", param_type=ParamType.CHOICE, choices=["winner", "all_families"], default="winner"),   # wave 1, fixed only
ParameterSpec(name="unused_feature_grading", param_type=ParamType.CHOICE, choices=["off", "all", "exclude_constant"], default="off"),  # CD-08, fixed only
ParameterSpec(name="stale_report_action", param_type=ParamType.CHOICE, choices=["PROCEED_WITH_NOTE", "RETRY", "grade"], default="PROCEED_WITH_NOTE"),  # CD-07
ParameterSpec(name="model_skip_rules.expiry_mode", param_type=ParamType.CHOICE, choices=["until_reset", "hours"], default="until_reset"),  # CD-09
ParameterSpec(name="history_prune", param_type=ParamType.CHOICE, choices=["off", "on"], default="off"),                 # CD-10
ParameterSpec(name="history_prune_mode", param_type=ParamType.CHOICE, choices=["move", "delete"], default="move"),      # CD-10
ParameterSpec(name="history_max_files", param_type=ParamType.INT, min_value=1, default=None),                          # CD-10, None = td.MAX_HISTORY_FILES (100)
ParameterSpec(name="history_archive_dir", param_type=ParamType.STRING, default="diagnostics_outputs/history_archive/"),  # CD-10
ParameterSpec(name="history_prune_include_root_cause", param_type=ParamType.BOOL, default=False),                      # CD-10
ParameterSpec(name="training_health_log", param_type=ParamType.CHOICE, choices=["off", "on"], default="off"),          # CD-11 writer
ParameterSpec(name="advisor_reads_training_health", param_type=ParamType.BOOL, default=False),                         # CD-11 reader (parameter_advisor.py)
ParameterSpec(name="llm_prompt_fields", param_type=ParamType.CHOICE, choices=["legacy", "report"], default="legacy"),   # CD-12 (diagnostics_llm_analyzer.py)
```

**Who may change them (suggested):**
- Report-content settings: the lead.
- Settings that change WATCHER's actions (`grading_rules`, `stale_report_action`, `expiry_mode`, `history_prune*`, `advisor_reads_training_health`): the owner.
- `diagnostics_run_id` and `diagnostics_run_started_at`: WATCHER only.

Every value is validated, and an unknown value fails closed. In the health check that becomes a `PROCEED_WITH_NOTE` with a module-error note (`test_cd07_unknown_setting_value_fails_closed`).

**Run record:**
- Every level-full report now carries `diagnostics_settings`: level, epsilon, manifest path and sha256, dead-unit definition, and probe rows (`training_diagnostics.py:447`, `:640`).
- Every health-log line carries the grading and wave 2 settings in force (`training_health_check.py:952`).

## Anchors

| Item | Where |
|---|---|
| Settings constants, options bundle, child flags | `training_diagnostics.py:118` (constants), `:134` (`new_run_id`), `:142` (`diagnostics_options_from_args`), `:167` (`diagnostics_child_flags`) |
| CD-07 stamp | `training_diagnostics.py:459` |
| CD-07 freshness rules | `training_health_check.py:841` (`report_freshness`), `:886` (`_stale_result`) |
| CD-07 call sites | legacy path `:287` (only when the caller names its run); fixed path `:783` (winner) and `:800` (families) |
| CD-07 pointer stamp, child flags | `meta_prediction_optimizer_anti_overfit.py:347`, `:251`, `:2620`, bundle built `:3208` |
| CD-08 metric | `training_diagnostics.py:1395`, written `:481-484` |
| CD-08 grading | `training_health_check.py:592` (value), `:667` (grading) |
| CD-09 | `training_health_check.py:901` (`skip_active`), `:1232`, `:1259`; Step 5 filter `meta_prediction_optimizer_anti_overfit.py:179` |
| CD-10 | `training_diagnostics.py:1689` (prune; defaults unchanged); `training_health_check.py:930`, called `:1132` |
| CD-11 writer | `training_health_check.py:952`, called from `check_training_health` (`:188`) |
| CD-11 reader | `parameter_advisor.py:578`; gate `:492`, `:706`, `:783`; prompt line `agents/contexts/advisor_bundle.py:210` |
| CD-12 | `diagnostics_llm_analyzer.py:118`, `:144`, `:183`, `:313` |
| CD-23 | `training_diagnostics.py:595` (`_preact_hooks`), `:646` (`_dead_history`) |
| TODO-CH14-4 | `training_diagnostics.py:184` (`load_run_level_names`), `:1562`, `:1567` |
| Worker | `train_single_trial.py:660` (NN), `:127` (tree), `:934` (argparse), `:1018` (options) |

## Hunks in the two shared files (vs 41ad509)

**`meta_prediction_optimizer_anti_overfit.py`**, 13 hunk headers (`git diff -U0`), 10 of them from wave 1:
1. `@@ -172 / -174` (wave 2): the S90 skip filter calls `training_health_check.skip_active`. The old comparison is the fallback if that import fails.
2. `@@ -238 +245,7`: the compare child gets its flags from `td.diagnostics_child_flags`. This replaces the wave 1 lines. A caller without the bundle has one built.
3. `@@ -328 +342,8`: the pointer is written (wave 1) and now also carries `diagnostics_run_id` and `diagnostics_run_started_at`.
4. `@@ -1836 +1858` and `@@ -2763 +2836` (wave 1): the inline tree report calls.
5. `@@ -2532 +2554,10` (wave 1): `feature_names` in the NPZ.
6. `@@ -2588 +2618,3`: the NN subprocess flags come from the helper.
7. `@@ -2721 +2754,40` (wave 1): the helpers. Wave 2 adds one `options=` line.
8. `@@ -3081 +3155,24`: argparse, with 2 wave 1 flags and 5 wave 2 flags.
9. `@@ -3107 +3205,4` (wave 2): the options bundle, built after `parse_args`.
10. `@@ -3123 +3225,6` and `@@ -3184 +3292,3`: the `args_dict` key and the optimizer attribute.

**`train_single_trial.py`**, 21 hunk headers. Wave 2 adds no new regions; it only extends wave 1's:
- the tree full emit (adds `options=`);
- the NN attach block (now one call to `td.make_nn_diagnostics_full`, shorter than wave 1's);
- argparse (5 flags);
- the `_diag_opts` dict (5 keys).

The `train_neural_net` signature line is unchanged since wave 1. It still overlaps with the selfplay-nn worktree.

**Other files, none shared or hand-landed** (checked against `scripts/owner_land.py:86-87` and `GOVERNED_FILES` in `scripts/gate12_parity_gate.py`):
`training_diagnostics.py`, `training_health_check.py`, `parameter_advisor.py` (+57), `diagnostics_llm_analyzer.py` (+114), `agents/contexts/advisor_bundle.py` (+2).

## Tests and evidence (`data/ch14/w2/logs/`)

| What | Result | File (sha256) |
|---|---|---|
| Wave 1 + wave 2 tests | 73 passed, 50 s wall, peak RSS 740 MB | `pytest_w1w2_final2.log` (`ffb8d2e8…6e`) |
| Wave 1 mutants (re-check) | 22/22 killed | `mutants_w1_final.json` (`9809dcf0…89`) |
| Wave 2 mutants (after the reboot) | 29/29 killed, 48.5 s, 2 jobs | `mutants_w2_final2.json` (`aabf7ed6…c8`) |

- Per-mutant logs: `mutant_logs_w1/`, `mutant_logs_w2/`.
- Test files: `tests/test_ch14_diag_wave2.py` (`033a388e…3b`) and `tests/ch14_wave2_mutants.py` (`e0e8e728…60`).

**Default-unchanged gates against `git show 41ad509`:**
- the wave 1 legacy-grade, NN-report and tree-report identity gates;
- `get_skip_status` and `is_model_skipped`;
- `_prune_history_fifo` at its defaults;
- the default LLM prompt, byte for byte;
- `build_advisor_bundle` without the new key, byte for byte.

**Changes to wave 1 test files:**
- Mutant M16 now targets the new child-flag line (same test, same tag).
- `tests/ch14_cpu_patch.py` sets lightgbm `num_threads` from the process's allowed cores instead of `os.cpu_count()`, so it respects `taskset`. Test-only.

**Fixed during the build:**
- Three wave 1 CD-06 tests failed once: the first stamp key, `run_id`, collided with the S88 compare id. Renamed to `diagnostics_run_id`.
- RETRY results carry no `consecutive_critical`, so the log now also records the skip-registry count.
- Three mutants first failed before their tagged line; those reads were made safe.

**Cleanup:** scratch folders holding untracked `.py` copies (head copies, mutant roots) were removed. No `.py` remains under `data/`. My four `tests/*.py` files are untracked: Gate 22's sensitivity applies until they land.

## Choices made (say if you want them the other way)

- **CD-23 default is `linear_zero` (today's).** The plan said to default to the new definition; the task rule (defaults = today's behaviour) won.
- **CD-10:** the prune switch `history_prune` is off by default (today the archive is never pruned). When it is on, the mode defaults to `move`, as the lead asked.
- **CD-11:** the log writer is also off by default, because a new file is a new behaviour.
- **CD-07:** on the legacy grading path the stale guard runs only when a caller passes its run (the draft WATCHER line); on the fixed path it always runs.
- **CD-12:** the `llm_on` call-site setting is not built, because that call site is in `agents/watcher_agent.py` (hand-landed). "Every family's grade" in the prompt uses each report's own severity, not WATCHER's grade.

## UNVERIFIED and limits

1. **The WATCHER draft has not been run.** It is checked only with `git apply --check`.
2. **Stale detection without the WATCHER id is partial.** A pointer and family reports that all come from one older full-level run agree with each other and are not flagged. Only WATCHER's expected id, or its start time, catches that.
3. **CD-09 `hours` mode does not reset the counter after expiry.** One more critical after a skip expires re-skips at once. The policy text does not say which is meant.
4. **The CD-11 advisor reader adds a signal and one prompt line only.** `FocusAreaClassifier` (`MODEL_DIVERSITY`) is not rescored.
5. **The health log is append-only and has no cap**, at about one line per check.
6. **CD-23 measures on the eval-mode probe batch** (BatchNorm running statistics), not on train-mode batches. With LeakyReLU, a "dead" unit still passes 0.01 × its gradient.
7. **No Step 5 run with these settings on real data.** That needs the lead's queued GPU run (approved in principle, started by the lead).

## Owner questions (one line each, held with Q1–Q3 from wave 1)

1. Should a model skip expire after `skip_duration_hours` (`expiry_mode: hours`) or last until a non-critical check (today's)? (Plan Q1)
2. When the history cap is on, should the oldest files be moved (the default now) or deleted? (Plan Q2)
3. Should the dead-unit metric default to the new `always_negative_preact` definition?
4. Should `training_health_log` be on, so the Strategy Advisor has something to read?

I've stopped here.
