# PLAN — Chapter 14 diagnostics made correct and complete (CH14-DIAG, 2026-11-23)

**Author:** Team Alpha (Claude Code), worktree `~/tfm_worktrees/ch14-diag`, base `41ad509`.
**Task:** `logs/e2e_session/TASK_CH14_DIAG.txt` under `TASK_CH14_COMMON.txt` (main tree).
**Authority:** owner ruling 2026-11-23 10:40 (`logs/e2e_session/OWNER_RULING_step5_go_ch14_beta_holds_20261123.txt`):
Chapter 14 is to be implemented, Beta holds no longer stop work, owner and lead authorisation are the only
ones required. Owner directions 10:32 and 10:35: selfplay and ML must eventually be able to adjust pipeline
settings.
**Kind:** plan only. No code, no run, no GPU, no rig. Nothing was changed except this file.

---

## Prior record

### Searches run (worktree at `41ad509`, `/bin/grep` so `.json` and ignored files are not skipped)

```
P='training_health\|training_diagnostics\|best_val_round\|skip_registry\|feature_gradient\|enable_diagnostics\|Chapter 14\|CH14\|tensorboard\|unused_feature\|record_event\|actual_mean'
/bin/grep -l -i "$P" docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_* docs/TB_RULING_REQUEST_* docs/PROPOSAL_* \
     docs/CHAPTER_* docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md
/bin/grep -rl -i "$P" docs/exchange/ docs/drafts/
/bin/grep -n -i "$P" docs/CHAPTER_6_AUDIT.md docs/CHAPTER_10_AUDIT.md docs/CHAPTER_11_AUDIT.md docs/CHAPTER_7_AUDIT.md \
     docs/CHAPTER_5_AUDIT.md docs/PROJECT_FILE_CATALOG.md docs/CHAPTER_CORRECTIONS_REPORT_20261111.md
/bin/grep -n -i 'chapter 14\|ch14\|selfplay\|diagnos\|tensorboard\|dashboard' docs/TODO_CONSOLIDATED.md
git diff --stat a15b524 41ad509 -- <every code file this plan names>
```

Hits: catalog (`:299`, `:729`); 10 `PROPOSAL_*` (read: `CATEGORY_B_NN_TRAINING_ENHANCEMENTS_v1_0`, `EPISTEMIC_AUTONOMY_UNIFIED_v1_3`;
the rest are keyword hits on `enable_diagnostics` / feature importance, read at the hit, none decides a Chapter 14
item); `CHAPTER_*` and `*_AUDIT.md` (Ch5, Ch6, Ch10, Ch11, Ch13 progress v3_1..v3_9, Ch14 set); `TODO_CONSOLIDATED.md`;
`CHAPTER_CORRECTIONS_REPORT_20261111.md`. **No hits** in `TB_RULING_*` (30 files), `TB_RULING_REQUEST_*`, `BACKLOG.md`
(for these terms; its `:94-101` WATCHER RETRY log item was found by reading), `LEADS.md`, `docs/exchange/`, `docs/drafts/`.

Read in full or at the cited sections: `CHAPTER_14_TRAINING_DIAGNOSTICS.md` (§15–§19 first, then every correction note
in §1–§14), `CHAPTER_14_AUDIT.md` with Addendum and Addendum 2, `CHAPTER_14_IMPLEMENTATION_PROPOSAL_S69.md`,
`TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md` (all), `S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md:424-428`,
`TODO_CONSOLIDATED.md` (b) rows `:105-126`, `TODO_SELFPLAY_AND_LLM_AUTONOMY.md:1-140`, the e2e map
(`~/tfm_worktrees/e2e-map/docs/drafts/e2e_v1/MAP_E2E_THIN_PATH_20261122.md`, headings and every diagnostics /
Step 5 / selfplay line), `S111_TEAM_BETA_BRIEFING.md:113-157`, `CLAUDE_CODE_INSTRUCTIONS_CHAIN_C_TRUTHFULNESS_HOTFIX.md:33-44`,
`PIPELINE_BEHAVIOUR_MODEL.md:993-996`, `SESSION_CHANGELOG_20260216_S94.md:59`, `SESSION_CHANGELOG_20260213_S83.md:14`, `:46`,
`AGENTS.md` §2. `CHAPTER_13_AUDIT.md` was opened; it carries no Chapter 14 diagnostics item in this scope (B13–B15 are
the CH14-LEARN plan's scope).

### Documents that already decide, authorise, schedule or record the subject

| document | section | what it records |
|---|---|---|
| `CHAPTER_14_TRAINING_DIAGNOSTICS.md` | §16.1, §16.2, §17 | B1–B18, S1–S10, open items |
| `CHAPTER_14_AUDIT.md` | §4–§5, Addendum A.1–A.4, Addendum 2 | evidence; host counts; S10 authority corrected to the owner's S111 hold |
| `CHAPTER_CORRECTIONS_REPORT_20261111.md` | §2.11 R-60…R-76; D-28…D-30 | **the repair register**: every B item in this plan is an OPEN R-number; B11 (R-70) is GOVERNED |
| `CHAPTER_14_IMPLEMENTATION_PROPOSAL_S69.md` | §8 `:374-382` | Phases 2, 4, 5, 7, 8 deferred ("foundation first", "optional, human-only") |
| `TODO_CONSOLIDATED.md` | `:122-126`, `:294`, `:333` | Ch14 Phase 4, 5, 8 remainder, 9, task 3.6 as open rows |
| `TFM_SYSTEM_MAP…v1_2.md` | §8 (namespaces), §9 (no duplicated feature lists), §13 P0-A, §14 | globals are context, not survivor evidence; one canonical feature-schema producer; "no diagnostic module may maintain a manually duplicated feature-name list" |
| `S111_TEAM_BETA_BRIEFING.md` + `SESSION_CHANGELOG_20260225_S111.md:80-84` | G1–G6 | Beta approved G1–G6; owner held the patch ("Hold off on S112 patch") |
| `CHAIN_C_TRUTHFULNESS_HOTFIX.md:33-44` | Beta position | "The six proposals must not be wired through"; "does not authorize adding the six fields to `reinforcement.json`" |
| `SESSION_CHANGELOG_20260216_S94.md:59` | TB Trim #1 | no diagnostics on Optuna folds ("avoids 100-file explosion") |
| `SESSION_CHANGELOG_20260213_S83.md:14`, `:46` | session decision | selfplay trend observe-only |
| `BACKLOG.md:94-101` | §4 | three `[WATCHER][RETRY]` log lines still claim a change |
| `CHAPTER_11_AUDIT.md:218`, `:224` | X1, S9 cross-ref | `f{i}` importance keys absorbed as X1; dashboard deferral |
| `CHAPTER_6_AUDIT.md:447-450` | cross-refs | B1 and S10 recorded there as cross-references only |

### Label per item raised here

Every work item below is **ALREADY RECORDED** (it implements a recorded R-number, S-item or TODO row, cited in its row)
except these, labelled **NEW**: CD-07 (report freshness / run identity guard — the stale-grading consequence is recorded as
audit C3, but no repair item exists), CD-13 (data-driven column classes in every report — the system map requires the
classification, §8–§9, but no Chapter 14 item asks for it), CD-23 (dead-neuron metric that can fire — D12 is recorded as
DESCRIPTION-ONLY; making the metric able to fire is new work), CD-00 (offline re-grade mode). Items that lift an S-item
are **CHANGES A RULING** in the old sense; under the owner's 10:40 ruling they are to-dos, and each still names the record
it supersedes: CD-15 (S1, Beta's Chain C position), CD-20 (S7, TB Trim #1), CD-18 (S8), CD-21/CD-22 (S9), CD-14 (S10,
the owner's own S111 hold — this one is the owner's to lift, Q4).

### Anchor re-verification at `41ad509`

`git diff --stat a15b524 41ad509` over every code file named here shows changes only in `agents/watcher_agent.py`
(+505/−77), `agents/watcher_dispatch.py` (+70) and `scripts/owner_land.py` (new). All chapter anchors in the other files
hold and were re-opened this session. WATCHER anchors moved; the current ones, opened this session:

| chapter anchor (a15b524) | now (41ad509) | what |
|---|---|---|
| `wa:2861-2868` | `agents/watcher_agent.py:3180-3190` | health-check call site |
| `wa:2897-2901` | `:3216-3225` | canonical diagnostics file deleted before RETRY |
| `wa:2262-2309` | `:2555-2609` | `_handle_training_health` |
| `wa:2375-2386` | `:2653-2681` | `_build_retry_params`; log lines `:2671`, `:2675`, `:2679` |
| `wa:2336-2343` | `:2611-2651` | `_is_within_policy_bounds`, six-field whitelist `:2629-2636` |
| `wa:1813-1820` | `:2026-2034` | step-scoped `default_params` merge filter |
| `wa:2400-2432` | `:2689-2823` | LLM refinement on RETRY (now with S179 truthful status logging) |

### Not searched

The host's untracked state (`diagnostics_outputs/` contents were read by the 2026-11-11 pass, not re-read now);
ser8 archives; the private repository beyond this clone; project files not in the repository; `~/dashboard_work`
(the RANGE-MINER-era dashboard lives there, `TODO_CONSOLIDATED.md:279`); the other Chapter 14 plan
(`~/tfm_worktrees/ch14-learn`) — not read, scopes split by the lead.

---

## Plain summary

**What works today.** The diagnostics code exists and is safe: it never fails Step 5 (`train_single_trial.py:119-158`,
`:610-618`; `training_health_check.py:80-120`). On the WATCHER path Step 5 runs with diagnostics on
(`agent_manifests/reinforcement.json:179`). The neural net writes a real per-epoch report (loss curves, layer
gradient norms). WATCHER reads one report after Step 5 and maps it to proceed / retry / skip, with retries capped at 2
(`agents/watcher_agent.py:3180-3225`). The selfplay fold capture, the trend check, the LLM grammar and schema all exist.

**What does not work.** Almost nothing reaches a correct conclusion:
- Every report the health check has ever graded came out **critical** (107 of 107 on the host), mostly because it reads
  a key the producer never writes (B1). The skip and retry actions downstream of that grade are therefore driven by a
  defect, not by training health.
- Only the neural net produces a report on the production path; the three tree models produce none (B3), and where
  tree code does run, XGBoost/LightGBM rounds and importances are not captured (B4) and features are named `f4`, `f6`
  instead of their names (B5). The NN never captures per-feature gradients (B18), so its gradient-spread check cannot
  fire either (B2), and its dead-neuron metric reads 0 on every layer by construction (D12).
- The health check judges whatever model wrote the canonical file last, not the compare-models winner (B6), and can
  re-grade an old report (audit C3: 35 of 78 archived checks re-graded a report already graded).
- The LLM gets model type, severity and issue strings only (B7); its proposals, and WATCHER's own retry changes, never
  reach Step 5 (S1), so a retry re-runs the same configuration.
- Skip expiry (B8), the incident/advisor bridge (B9), history pruning (B10) and `unused_feature_pct` (B16) are declared
  and not implemented. Selfplay diagnostics are off (§9.1), the degrading-trend incident is swallowed (B12), the trend
  acts on nothing (S8), and the 8.5/8.6 tests assert on literals (B17).
- Any importance list can be headed by one of the **36 of 89 columns that carry one value per trial or per run**
  (§15.1), and no diagnostic says so.

**How much work.** **24 work items, about 65 agent-hours** including their tests (about 61 without TensorBoard),
plus landing and review rounds. Four lanes can run in parallel from the start. **Wave 1 (9 items, about 20 h) gives
the owner the first real indication:** a Step 5 report for all four families with named importances, NN gradients and
a column-class tag, graded by a health check whose grade is no longer automatically critical, plus a re-grade of the
archived reports and of any Step 5 run made since.

**What these diagnostics are and are not.** They are instruments on training. None of them is a verdict on TFM.
TFM is judged by whether its pools hit more often than `k/1000` on draws the sieve never saw. Items that let a
diagnostic **act** (CD-09 skip, CD-15 retry, CD-18 selfplay stop) change which model ranks the pools. Each one of
those ships with its action off by default, and the measurement that settles whether to switch it on is the e2e grader
(`~/tfm_worktrees/e2e-grader`, read only by this plan): pools from the run where the action fired against pools from
the run where it did not, on the same sealed grade block, each against `k/1000`.

**Answer on the unbuilt phases (S9).** The **web `/training` page is worth building, but only after wave 1**: before
that, three of four families have nothing to plot, features are unnamed and the NN gradient chart is empty. A small
read-only page (CD-21, about 5 h) is wave 3. **TensorBoard is not worth building now.** It is human-only by the
chapter's own decision (§10.1), has no consumer, and once wave 1 lands the JSON reports plus CD-21 carry the same loss,
importance and gradient information. It becomes worth its 4 h only if an NN investigation needs weight histograms or the
graph view. `tensorboard` is already importable in `~/venvs/torch` (`importlib.util.find_spec`, this session), so it
needs no install when wanted. Recorded as CD-22, last.

---

## Work items

Columns: **Gets** = what the owner gets, in plain words. **HL/GOV** = owner-hand-landed (`scripts/owner_land.py:86-87`:
`agents/watcher_agent.py`, `agent_manifests/`, `miner/`, `gate12_launch.sh`, `scripts/gate12_parity_gate.py`) or in
`GOVERNED_FILES` (`scripts/gate12_parity_gate.py:174-185`). No file in this plan is in `GOVERNED_FILES`.
**Beta class** = per `AGENTS.md` §2 table (`:62-66`): before-commit is coordinator/worker/kernel code (`miner/`,
`prng_registry.py`, `sieve_gpu_worker.py`), the verifier and launch path for runs that count, and the evidence of a
counted run; tools, monitoring and the dashboard are after-commit. `agents/watcher_agent.py` is the pipeline launcher
(CLAUDE.md §1.3); this plan classes changes to its dispatch as **before-commit (launch path)**. The diagnostics modules
are monitoring and are **after-commit**. Step 5 trainer edits that are proven not to change the trained model are
after-commit; an edit that does change it when switched on (CD-14) is after-commit as code, and any counted run that
uses it has its evidence reviewed before acceptance (`AGENTS.md:66`).

Every new behaviour is a declared setting with a stated default. Unless a row says otherwise, **the default keeps
today's behaviour**, and the test set includes a "default unchanged" gate. Every test runs on CPU on a small synthetic
fixture, with mutants that are shown to apply, run the mutated path and reach the credited assertion (TASK_COMMON
rule 8). Abbreviations: `thc` `training_health_check.py`, `td` `training_diagnostics.py`, `tst` `train_single_trial.py`,
`mpo` `meta_prediction_optimizer_anti_overfit.py`, `dla` `diagnostics_llm_analyzer.py`, `so` `selfplay_orchestrator.py`,
`iet` `inner_episode_trainer.py`.

### A. Health check and WATCHER path (§7)

| # | item (record) | Gets | files | HL/GOV | depends on | size h | test | Beta class |
|---|---|---|---|---|---|---|---|---|
| CD-01 | **B1** early-stop ratio key (R-60) | A grade that is not automatically critical. The early-stop check reads `best_val_round`, the key every real report carries (`td:239`); legacy `best_round` kept as fallback for the self-test fixture. The ratio's 0-based index is stated in the report. | `thc:392` | — | — | 1.5 | report shaped like the host's real schema-1.1.0 file with a late best round grades `ok`; one with best round 1 of 17 grades critical; mutant (read `best_round` only) turns the healthy case critical | after-commit |
| CD-02 | **B2** gradient-spread key (R-61), consumer and producer | The NN gradient-spread bound can fire. Reader uses `model_specific.gradient_health.gradient_spread_ratio`; producer takes it from the **last round that carries** `feature_gradients`, not only `_round_data[-1]` (`td:595-598` misses it whenever the last epoch is not a detail round) | `thc:373`, `td:595-598` | — | CD-03 for real values | 1 | report with spread 5000 grades critical; spread 50 grades ok; mutant (old key) never fires; producer test with last epoch not a multiple of `capture_every_n` still carries the spread | after-commit |
| CD-06 | **B6** compare-models winner (R-65), plus per-family grading | The check judges the model that actually won, and reports every family's grade. One filename and key: `mpo:325` also writes `diagnostics_outputs/compare_models_summary_latest.json` (pointer, per-run file kept); reader takes `winner_best_effort.model_type` (it is a dict, `mpo:311`). Setting `training_diagnostics.grade_scope: winner \| all_families` (default `winner`, today's intent) | `thc:155-170`, `mpo:325` | — | CD-05 (tree reports exist) | 2 | fixture compare summary naming `catboost` makes the check grade the catboost report even when the NN wrote last; mutant (old filename) grades the NN; `all_families` lists four grades | after-commit |
| CD-07 | stale report guard (NEW; audit C3) | A check never grades an old report. Step 5 stamps a `run_id` and start time into every report; the check compares them with the Step 5 run WATCHER just finished, and returns `stale` → action from setting `stale_report_action` (default `PROCEED_WITH_NOTE`) | `td` (stamp), `mpo` (pass run id), `thc` (compare), `agents/watcher_agent.py:3188` (pass run id/start) | **HL** (`watcher_agent.py`) | — | 2 | report older than the run start → `stale`; same-run report graded normally; mutant (skip compare) grades the stale file | thc/td/mpo after-commit; WATCHER line before-commit (launch path) |
| CD-09 | **B8** skip expiry (R-67) | Skips behave as the policy file says, or the policy file says what they do. Both modes as setting `model_skip_rules.expiry_mode: hours \| until_reset`; `hours` uses `last_critical` and `skip_duration_hours`, applied at both readers (`thc:667-689` `is_model_skipped`, `mpo:174` exclusion filter) | `thc`, `mpo:151-178`, `watcher_policies.json` | — | CD-01 (else every run counts toward a skip) | 2 | clock-injected test: skip active at 23 h, lifted at 25 h in `hours`; never lifted in `until_reset`; mutant (ignore expiry) | after-commit |
| CD-10 | **B10** FIFO history pruning (R-69) | The history folder stops growing without bound (111 files on host). The live writer `thc:_archive_diagnostics` calls `_prune_history_fifo` (`td:918`); cap and mode are settings `history_max_files` (default 100, `td:72`) and `history_prune_mode: delete \| move` (Q2) | `thc:539-562`, `td:918` | — | — | 1.5 | 105 fixture files → 100 kept, oldest gone (or moved); `root_cause_*` files from Chapter 13 counted per the setting, not silently evicted; mutant (no call) keeps 105 | after-commit |
| CD-08 | **B16** `unused_feature_pct` (R-74) | The declared bound is computed and graded: share of the 89 columns with zero gain (trees) or input-gradient magnitude below a declared epsilon (NN). Reported also **per column class** (CD-13), so "36 constant columns are unused" is visible as expected, not alarming | `td` (compute), `thc:331-430` (grade) | — | CD-03, CD-05, CD-13 | 2 | fixture with 50 of 89 zero-importance columns → warning at 40 %; mutant (metric not computed) never warns | after-commit |
| CD-11 | **B9** incident and advisor bridge (R-68) | Training health reaches the Strategy Advisor. One artifact, `diagnostics_outputs/training_health_log.jsonl` (append-only): **one writer** (`thc`, after every check, including skips and retries), **one reader** (`parameter_advisor.py` `DiagnosticsLoader`, new `load_training_health()`; `MODEL_DIVERSITY` scoring may read it behind setting `advisor_reads_training_health`, default `false`). The §7.5 undefined functions are not created; this replaces them | `thc`, `parameter_advisor.py:492-520` | — | CD-01, CD-07 | 3 | check run → one JSONL line with action, severity, model, run id; advisor loader returns it; default-off gate shows advisor output unchanged; mutant (writer off) → loader empty | after-commit |
| CD-00 | offline re-grade mode (NEW) | The first indication without running anything: `python3 training_health_check.py --regrade <dir> --out data/ch14/regrade_<ts>.json` re-grades every report in a directory with the fixed rules, read-only, and tabulates old vs new grade. Run over the host's 107 archives and over any Step 5 run made since (e.g. the lead's run H Step 5) | `thc` (`__main__`, `:777` area) | — | CD-01, CD-02, CD-06 | 2 | fixture dir of 5 reports → table matches hand count; input files byte-identical after (sha256); multi-core not needed (JSON reads) | after-commit |

### B. What the reports contain (§3, §15)

| # | item (record) | Gets | files | HL/GOV | depends on | size h | test | Beta class |
|---|---|---|---|---|---|---|---|---|
| CD-03 | **B18** NN per-feature gradients (R-76) | The NN report finally says which inputs the network is sensitive to. On each detail epoch, a separate pass: model in `eval()`, a `requires_grad` copy of a fixed training batch, `torch.autograd.grad(loss, x)` — no parameter gradient, no BatchNorm running-stat update, no RNG draw on the training stream; then restore `train()`. Batch size of the probe is a setting (`gradient_probe_rows`, default 512) | `td:417-500`, `tst:600-660` | — | CD-04 (names) | 3 | CPU tiny net: report carries `feature_gradients.top_10` with real names and `spread_ratio ≠ 1.0`; **final `state_dict` bit-identical with diagnostics on vs off** (training unaltered, §2.3); mutant (probe skipped) → no `feature_gradients` → red | after-commit |
| CD-04 | **B5** feature names into every report (R-64) | Importances say `lane_agreement_8`, not `f4`. `mpo` adds the 89 sidecar names (`self.feature_schema['feature_names']`, `mpo:593`) to the trial worker's NPZ handoff (`mpo:2532`; a private temp file, **not** the 22-array survivor contract) and to the inline trainers; `tst` passes them to `NNDiagnostics()` (`tst:614` passes none today) and to `_emit_tree_diagnostics` (`tst:147`). Missing names → report says `names_source: positional` instead of guessing | `mpo:2532`, `:1876`, `:2565`, `tst:112-158`, `:614`, `:890-896` | — | — | 2 | report names equal sidecar names in order; mutant (names dropped) → `names_source: positional` and the gate fails; old NPZ without names still loads | after-commit |
| CD-05 | **B3 + B4** tree diagnostics on the paths that run (R-62, R-63) | All four families get a report. Final inline models (`mpo:1832`, `:2761` → `MultiModelTrainer._train_xgboost/_lightgbm/_catboost`, `mpo:998-1125`, sklearn-API objects, no early stopping) and the trial worker's Booster path (`tst:216`, `:309`, `:323`). Val curve from `evals_result()` / `evals_result_` / `get_evals_result()`; **train curve computed afterwards by staged prediction** (`iteration_range` / `num_iteration`), so the `fit` call and its `eval_set` are not touched; importance by gain (`get_booster().get_score`, `feature_importance('gain')`, `get_feature_importance`). `XGBDiagnostics`/`LGBDiagnostics` (`td:683-690`) get the capture code. Per-family report path `diagnostics_outputs/{run_id}/{model}_diagnostics.json`; canonical copy unchanged. **Settles UNVERIFIED #5** (Booster `feature_importances_`) by an executed CPU check inside the test | `mpo:998-1125`, `:1832`, `:2761`, `tst:180-435`, `td:607-700` | — | CD-04 | 5 | 200-row fixture per family: report has `rounds_captured == n_estimators`, named gain importance, both curves; **predictions bit-identical on vs off**; mutant (capture removed) → no tree report; device CPU, `n_jobs` set so LightGBM/XGBoost use several cores | after-commit |
| CD-13 | column classes in every report (NEW; system map §8–§9; §15.1, S10 as it affects reading) | Every importance, gradient and attribution line is tagged `local`, `trial`, `run` or `constant`, so "the model leans on `forward_count`" reads as "on a trial-level value". **Computed from the training matrix, not from a list** (system map §9 forbids duplicated feature lists): `constant` = one value over all rows; `run` = one value over all rows and name in the sidecar's global set; `trial` = one value within each trial group when the rows carry a trial id; else `local`. Report adds `importance_share_by_class` | `td` (classifier + report field), `mpo`/`tst` (pass X stats, no X copy) | — | CD-04 | 3 | fixture with planted constant, trial-constant and local columns → classes correct; real-shaped 89-column fixture → 36 non-local (B 13, C 17, D 6) when rows are single-generation; mutant (classifier off) → no tags → red | after-commit |
| CD-23 | dead-neuron metric that can fire (D12, NEW repair) | `nn_dead_neuron_pct` measures something real. Today it counts exact zeros on the `nn.Linear` output, which sits before BatchNorm and the activation, and production uses LeakyReLU (`mpo:245-246`): 242/242 host entries read 0.0. New definition, as setting `dead_unit_definition: linear_zero \| always_negative_preact` (default the new one; old kept selectable): a unit is dead if its pre-activation is ≤ 0 for every row of the probe batch | `td:402-415` | — | CD-03 (shares the probe) | 2 | net with a unit forced negative → reported dead; healthy net → 0 %; mutant (old definition) misses the forced unit | after-commit |

### C. The LLM bundle (§8)

| # | item (record) | Gets | files | HL/GOV | depends on | size h | test | Beta class |
|---|---|---|---|---|---|---|---|---|
| CD-12 | **B7** prompt from schema 1.1.0 (R-66) | The LLM is shown the rounds, best round, overfit gap, named top features with their column class, NN layer and gradient health, and every family's grade, instead of type and severity only. Keys read from `training_summary` and `model_specific.*` (`dla:178-206`); old keys kept as fallback for old archives. Setting `llm_on: retry \| warning \| never` (default `retry`, today's call site `wa:2689`). Grammar and schema unchanged | `dla:112-230` | — | CD-01…CD-05, CD-13 | 2.5 | prompt built from a real-shaped report contains best round, gap and ≥5 named features with classes; mutant (old keys only) → those fields 0/empty → red; no LLM is called in tests (prompt builder only) | after-commit |

### D. Diagnostics on Optuna folds (S7) and retry parameters reaching Step 5 (S1)

| # | item (record) | Gets | files | HL/GOV | depends on | size h | test | Beta class |
|---|---|---|---|---|---|---|---|---|
| CD-20 | **S7** per-fold diagnostics during the search (supersedes TB Trim #1, S94:59) | NN and tree health per Optuna trial and fold, without overwriting the final report. Setting `--diagnostics-on-folds` (default off, `mpo:1891`). `tst` gains `--diagnostics-output PATH` (default today's paths); `mpo` passes `diagnostics_outputs/{run_id}/optuna/trial_{n}_fold_{k}.json`; the canonical file is never written from a fold. A per-trial summary (worst severity, best-round ratio, spread) goes into Optuna `user_attrs`. Acting on it (pruning a trial on a critical fold) is a second setting `prune_on_fold_critical`, default off, because it changes the search | `mpo:1857-1900`, `:2113`, `tst:860-870` | — | CD-03, CD-04, CD-05, CD-10 | 4 | 2-trial × 2-fold CPU study: 4 fold files, canonical untouched, `user_attrs` present; study best value identical on vs off with pruning off; mutant (fold writes canonical) → red | after-commit |
| CD-15 | **S1** retry parameters reach Step 5 (supersedes Beta's Chain C position; R-number none) | A retry that differs from the failed run, with the log saying what was actually applied. (a) Declare in `agent_manifests/reinforcement.json` `default_params` the keys the retry and the LLM can propose — `dropout`, `normalize_features`, `use_leaky_relu`, `learning_rate`, `n_estimators`, `max_depth`, `nn_activation` (`wa:2629-2636`, `:2669-2679`) — each with a null/no-op default so an ordinary run is unchanged. (b) Step 5 CLI routes: `--dropout`, `--normalize-features`, `--use-leaky-relu` exist (`mpo:3086-3092`); the other four are new **pins** that fix the value Optuna would otherwise choose (Q5). (c) Fact to state in the log: in `--compare-models` mode the NN already forces normalisation and LeakyReLU (`mpo:240-246`), so those two proposals are no-ops there. (d) The three `[WATCHER][RETRY]` log lines (`wa:2671`, `:2675`, `:2679`; `BACKLOG.md:94-101`) report recommended / applied / effective, the same three-leg rule as `TODO_SELFPLAY…:93-104` | `agent_manifests/reinforcement.json`, `agents/watcher_agent.py:2653-2823`, `:2026-2034`, `mpo` argparse and objective | **HL** (both) | CD-01 (else every run retries), CD-07, CD-12 | 4 | synthetic critical report → retry → dispatched command line contains `--dropout 0.4`; Step 5 dry-parse shows effective value; mutant (manifest key removed) → flag absent and log says `validated_not_applied`; default run's command line byte-identical to today | **before-commit (launch path)** |

### E. Selfplay episode diagnostics (§9.1–9.2)

Coordinate with `~/tfm_worktrees/selfplay-nn` (TASK NN), which edits `iet` and `configs/selfplay_config.json` at the same
time. This lane starts from whatever that worktree lands, or the two are merged by the lead before landing.

| # | item (record) | Gets | files | HL/GOV | depends on | size h | test | Beta class |
|---|---|---|---|---|---|---|---|---|
| CD-16 | **§9.1** episode diagnostics switched on, with real tree train curves | Every selfplay episode records training health per fold. `--enable-diagnostics` CLI on `so` (today config-only: `SelfplayConfig.from_file` accepts the key, `so:226-229`; none in `configs/selfplay_config.json`); key `"enable_diagnostics": true` added to the tracked config (the switch the owner asked for; code default stays `False`, `so:213`); `agents/watcher_dispatch.py:171-175` passes the flag when the config says so. Fix the tree train curve: XGBoost records val as train (`iet:339`), LightGBM has only a val set (`iet:322-328`) — use the CD-05 staged-prediction helper, so the `fit` call is unchanged | `so`, `iet:310-376`, `configs/selfplay_config.json`, `agents/watcher_dispatch.py` | — (`watcher_dispatch.py` is not on the hand-landed list) | CD-05 helper | 3 | one `--single-episode` on a CPU fixture: `episode_diagnostics` has real `best_round_ratio`, non-zero overfit gap for XGB/LGB; **fitness and candidate identical on vs off**; mutant (flag not threaded) → `episode_diagnostics` empty | after-commit |
| CD-17 | **B12** trend incident recorded (R-71) | A degrading trend is written down. `LearningTelemetry.record_event(event_type, details)` added, writing through the existing `_append_to_log` (`modules/learning_telemetry.py:580-595`); the bare `except: pass` at `so:852-853` logs the exception instead of discarding it | `modules/learning_telemetry.py`, `so:840-853` | — | — | 1 | real `_check_episode_training_trend` with a declining history → one `training_quality_degrading` line in the health log; mutant (method removed) → test red (no longer swallowed) | after-commit |
| CD-18 | **S8** trend can act (supersedes S83 observe-only) | A degrading trend can stop a selfplay run. The run loop uses the return value (discarded today, `so:524`); setting `trend_action: observe \| stop` (default `observe`). `stop` ends the run after the current episode, records why, emits no candidate from the degraded episodes. Anything else ("adjust") is Q3 | `so:515-530`, `:826-870` | — | CD-16, CD-17 | 3 | 5-episode fixture with planted declining ratios: `observe` runs 5, `stop` runs 3 and records the reason; mutant (return ignored) → 5 episodes under `stop` | after-commit |
| CD-19 | **B17** tests that exercise production code (R-75) | Tests 8.5/8.6 prove what they claim. New tests call `InnerEpisodeTrainer.train` and `_check_episode_training_trend`; the literal-asserting tests (`test_phase_8_soak.py:137-196`) stay, marked as structure-only | new `tests/test_ch14_selfplay_diagnostics.py` | — | CD-16, CD-17 | 2 | the tests themselves, each with a mutant on the production function they cover | after-commit |

### F. Constant columns as a training setting (S10, §15.1)

| # | item (record) | Gets | files | HL/GOV | depends on | size h | test | Beta class |
|---|---|---|---|---|---|---|---|---|
| CD-14 | **S10** `actual_mean`/`actual_std` exclusion and guardrail G1 (owner's S111 hold) | A switch, not a removal: `--exclude-features` (default today's list, `mpo:1481-1482`) and `--constant-feature-policy report \| exclude` (default `report`: G1 preflight from the CD-13 classifier prints and records near-constant columns, trains as today). With `exclude`, the schema hash changes, the sidecar records it, and Step 6 accepts by that hash as it does now. The columns stay in Step 3 output; nothing upstream changes. Declared in the Step 5 manifest so WATCHER and later the ML can set it. G2–G6 are listed in the record (`S111_TEAM_BETA_BRIEFING.md:115-131`) and are **not** in this plan (Ch6/Ch11 scope; G4 overlaps the Optuna study naming) | `mpo:550-600`, `:1476-1486`, argparse; `agent_manifests/reinforcement.json` | **HL** (manifest) | CD-13 | 4 | default run → feature list and schema hash identical to today; `exclude` → 87 columns, new hash, Step 6 loads the sidecar on a fixture; mutant (flag parsed but not applied) → 89 columns → red | code after-commit; manifest is the WATCHER interface → before-commit (launch path) |

S5 and S6 (per-seed sieve evidence and skip-shape placeholders entering the model) are the CH14-LEARN plan's scope.
They interact with CD-13 only through the classifier, which needs no change when new columns arrive: it computes classes
from data.

### G. Human views (S9)

| # | item (record) | Gets | files | HL/GOV | depends on | size h | test | Beta class |
|---|---|---|---|---|---|---|---|---|
| CD-21 | **S9** `/training` page (Phase 4, `TODO_CONSOLIDATED.md:124`) | A read-only page: loss curves per family, named importance with class tags, NN layer and gradient health, the health-check history table (from CD-11's log). Reads files only; starts nothing | `web_dashboard.py` (routes `:1943-2225`) | — | wave 1, CD-11 | 5 | Flask test client on fixture reports: page renders four families; missing reports render "absent", not an error | after-commit (dashboard) |
| CD-22 | **S9** TensorBoard (Phase 5, `TODO_CONSOLIDATED.md:125`) — **not recommended now** | Weight histograms and graph view for a human NN session. Setting `--tensorboard-dir` (default off) on `tst` only | `tst` | — | CD-03 | 4 | writer creates event files on a CPU fixture; default run writes none | after-commit |

### Totals

| group | items | hours |
|---|---|---|
| A. health check, WATCHER | 9 (CD-00, 01, 02, 06, 07, 08, 09, 10, 11) | 17 |
| B. report content | 5 (CD-03, 04, 05, 13, 23) | 15 |
| C. LLM | 1 (CD-12) | 2.5 |
| D. folds, retry | 2 (CD-15, 20) | 8 |
| E. selfplay | 4 (CD-16, 17, 18, 19) | 9 |
| F. constant columns | 1 (CD-14) | 4 |
| G. human views | 2 (CD-21, 22) | 9 |
| **total** | **24** | **≈ 64.5** (≈ 60.5 without CD-22) |

Hours are agent build-and-self-test time on CPU fixtures, not landing or review time. Rough: ±40 %.

---

## Proposed order — waves, one worktree per lane

Lanes are cut by file, so parallel worktrees do not edit the same file.

| lane | owns | |
|---|---|---|
| **A** | `training_health_check.py`, `parameter_advisor.py`, `watcher_policies.json` | |
| **B** | `training_diagnostics.py`, `train_single_trial.py` | |
| **C** | `meta_prediction_optimizer_anti_overfit.py` | |
| **D** | `selfplay_orchestrator.py`, `inner_episode_trainer.py`, `modules/learning_telemetry.py`, `configs/selfplay_config.json`, `agents/watcher_dispatch.py`, new selfplay tests | after or merged with `selfplay-nn` |
| **E** | `diagnostics_llm_analyzer.py` | |
| **F (owner hand-lands)** | `agents/watcher_agent.py`, `agent_manifests/reinforcement.json` | |
| **G** | `web_dashboard.py` | |

**Wave 1 — first real indication (4 worktrees in parallel, ≈ 20 h of work; longest lane B ≈ 10 h).**
- Lane A: CD-01, CD-02 (reader), CD-06 (reader), CD-00.
- Lane B: CD-03, CD-02 (producer), CD-04 (worker side), CD-05 (Booster side), CD-13 (classifier).
- Lane C: CD-04 (names into NPZ and inline trainers), CD-05 (inline final path), CD-06 (pointer writer).
- Lane D: CD-17 (one-hour fix, independent).

What the owner sees at the end of wave 1: the re-grade table over the 107 archived checks (old grade vs fixed grade),
and, on the next Step 5 run the owner or lead authorises (any run H Step 5 made before wave 1 lands runs **today's** code; its
reports can be re-graded by CD-00 but will lack tree reports and gradients — a second Step 5 with the wave-1 code is
what shows all four families), a report per family with named, class-tagged importance and NN gradients.

**Wave 2 — complete and wired (≈ 27 h).**
- Lane A: CD-07 (thc side), CD-08, CD-09, CD-10, CD-11.
- Lane B: CD-23, and `--diagnostics-output` for CD-20.
- Lane C: CD-20 (fold paths), CD-07 (run id stamp).
- Lane D: CD-16, CD-18, CD-19.
- Lane E: CD-12.

**Wave 3 — acting and human views (≈ 17 h), with the owner.**
- Lane F (owner hand-lands, Beta before-commit): CD-15, CD-07's WATCHER line, CD-14's manifest entries.
- Lane C: CD-14 (code).
- Lane G: CD-21.
- Last, only if wanted: CD-22.

Then Chapter 14 Phase 9 (first diagnostic investigation, `TODO_CONSOLIDATED.md:123`) becomes possible: one Step 5 run
on real data with everything on, owner-initiated, and a written reading of it. Whether any acting setting (CD-09 expiry,
CD-15 retry, CD-18 stop) is switched on afterwards is settled by the e2e grader on the sealed grade block, as stated in
the summary.

---

## Owner questions (one line each)

1. **Q1 (CD-09, B8):** should a model skip expire after `skip_duration_hours` as the policy file says, or last until a non-critical check (today's behaviour)?
2. **Q2 (CD-10, B10):** when diagnostics history exceeds its cap, delete the oldest files (the chapter's design) or move them to an archive folder?
3. **Q3 (CD-18, S8):** on a degrading selfplay trend, is stopping the run the action you want, or should it do something else (the record defines no "adjust")?
4. **Q4 (CD-14, S10):** once the exclusion is a switch, should its default flip to excluding `actual_mean`/`actual_std` (new schema hash, retrain), or stay off until the grader shows a difference?
5. **Q5 (CD-15, S1):** may a retry pin `learning_rate`, `n_estimators`, `max_depth` over what Optuna picks, or should retries change only the NN flags and dropout?
6. **Q6 (CD-06):** should a compare-models run retry when the winner is healthy but another family is critical, or grade the winner only (today's intent)?
7. **Q7 (CD-22, S9):** build TensorBoard now (not recommended), or leave it until an NN investigation asks for weight histograms?

---

## UNVERIFIED

- **Booster `feature_importances_`** (chapter §17.1, audit #5): not executed. CD-05's test settles it on CPU.
- **Whether Step 3's `survivors_with_scores.json` rows carry `trial_number`** (the canonical record does,
  `utils/canonical_records.py:229`): not checked in a Step 3 output. Without it, CD-13 cannot separate `trial` from
  `local` within one file and reports `trial_class: unavailable`.
- **The 36-of-89 count on run H's survivors**: the chapter's count is for a single-generation set (§15.1). Run H's
  Step 3 output was not read.
- **Which WATCHER action followed each of the 2026-03 checks** (chapter §17.3): not re-read.
- **Host state** (`diagnostics_outputs/` counts: 107 checks, 111 history files, 44 LLM proposals): taken from the
  2026-11-11 addendum, not re-listed now.
- **Hours**: estimates, not measured.
