# CLAUDE CODE REPORT: STEP 1 RERUN POLICY AND WHAT COVERAGE RECORDS, STAGE A (READ-ONLY), 2026-11-13

**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_STEP1_RERUN_POLICY_STAGE_A_20261113.md` (commit `87b3182`).
**Worktree:** `/home/michael/tfm_worktrees/step1_rerun_a`, detached at `87b3182145aa44f2053cd271ef9497923dcb005f`.
**Host / user:** VM101, `michael`, venv `~/venvs/torch`. **Model:** Opus.
**Review class:** document. The amendment that follows from this report goes to Beta before any code.

Every anchor below is a `file:line` read in this session, in the worktree at `87b3182`. Q4, Q7 and
Q8, and parts of Q5 and Q6, were traced by two read-only helper agents launched from this session.
Their claims marked NEW, and the strongest claims used here, were re-read at source by the lead
session; those anchors are marked **(re-read)**. Everything else from the helpers is cited as they
reported it. It is read-path evidence, not execution.

---

## Prior record

**Searches run, scope `docs/` in the worktree at `87b3182`.** Pattern
`rerun policy|re-run policy|regime shift|study_identity|search identity|UNIQUE\(run_id|step1_trial_history|run_id collision|_trials_to_run|seed_domain_complete|COMPLETE generates|partition key`
(case-insensitive). The output is saved in `data/step1_rerun_a/prior_record_search.txt`.

| Scope | Files hit |
|---|---|
| `PROJECT_FILE_CATALOG.md` | 0 |
| `TB_RULING_*` | 2: `TB_RULING_S176_WATCHER_KPI.md:414`, `TB_RULING_S177_KPI_GOVERNANCE.md:215` (regime-shift wording only) |
| `TB_RULING_REQUEST_*` | 0 |
| `PROPOSAL_*` | 2: `PROPOSAL_EPISTEMIC_AUTONOMY_UNIFIED_v1_3.md:398`, `PROPOSAL_v1_3_FINAL_ACCEPTANCE.md:255` (Step 1 "Static / Regime shift") |
| `CHAPTER_*` | 8. The relevant ones are `CHAPTER_1_WINDOW_OPTIMIZER.md:841`, `CHAPTER_13_LIVE_FEEDBACK_LOOP.md`, `CHAPTER_13_IMPLEMENTATION_PROGRESS_v1_2.md:74,193` and `CHAPTER_14_AUDIT.md:237` |
| `*_AUDIT.md` | 1: `CHAPTER_14_AUDIT.md` |
| `TODO_CONSOLIDATED.md`, `BACKLOG.md`, `LEADS.md` | 0 each |
| `exchange/*/*` | 2: the X-RUN4-RANGE R1 request and the Beta review |

Follow-up searches, all across `docs/`:

- `UNIQUE\(run_id|run_id, trial_number|step1_\{prng_base\}_\{|INSERT OR IGNORE INTO step1_trial_history` hit `SESSION_CHANGELOG_20260314_S142.md`, `CLAUDE_CODE_REPORT_OPTUNA_TO_FULL_SWEEP_PIVOT.md:240`, `TEAM_ALPHA_D6_2_IDENTITY_ADDENDUM.md:22,50,64` and `PIPELINE_BEHAVIOUR_MODEL.md:1379`.
- `_trials_to_run|max_iterations - _resumed|remaining: 0|n_trials=0` hit `SESSION_CHANGELOG_20260312_S138.md:69-72`, `CHAPTER_1_PATCH_S114.md:120`, `SESSION_CHANGELOG_20260302_S114.md:76` and `S172_D6_2_IMPLEMENTATION_REPORT.md:355`.
- `freshness` across the lineage report, `PIPELINE_BEHAVIOUR_MODEL.md`, `CHAPTER_14_AUDIT.md`, TODO and BACKLOG hit only `PIPELINE_BEHAVIOUR_MODEL.md:187` (the Step 0 TRSE lock, a different item).
- `watcher_agent=None` hit `CHAPTER_13_AUDIT.md:99` and `CHAPTER_13_LIVE_FEEDBACK_LOOP.md:892`.
- `dispatch` together with `complete` in `CLAUDE_CODE_REPORT_WATCHER_FOLLOWUP_REPAIRS.md` hit `:68`.
- `input-manifest digest|accumulator input` hit 6 files (helper search), including `TFM_PROJECT_FACTS_SKILL.md:638-646`, `PHASE_6_P0_SCOPING_v1.md:470` and `DATASET_PUBLICATION_SCHEMA_v1.md:111`.

**Every record in the brief's own prior-record table was re-read for this report and stands as
cited there.** The brief's table also covers the ALREADY RECORDED status of: the Beta ruling of
2026-08-07 (cursor law), R1 2026-08-08 (`dataset_sha256` is provenance, not a partition key), the
ledger's fields, the S145-R1 four-session plan, the purpose-lineage regime-rerun chain, fact file
§2.10b, WATCHER F-1 (COMPLETE is a clean stop), Route-A R2, the owner's automation statement,
S179 §8 BOOTSTRAP, and Beta's Q2/Q3 LIVE objection.

**Each item this report raises, with its label:**

| # | Item | Label |
|---|---|---|
| P1 | `step1_{prng_base}_{seed_start}` is not unique across reruns | ALREADY RECORDED: `TEAM_ALPHA_D6_2_IDENTITY_ADDENDUM.md:62-66`, `PIPELINE_BEHAVIOUR_MODEL.md:1378-1395`. The live ledger observation in Q2 is new **evidence** on a recorded status |
| P2 | Under `UNIQUE(run_id, trial_number)` + `INSERT OR IGNORE`, colliding rows are dropped silently | ALREADY RECORDED for partitions: `SESSION_CHANGELOG_20260314_S142.md:25-27,52-54`. Its reach to **reruns of the same seed range** is NEW as a statement. The mechanism is the same |
| P3 | A resumed study runs `max_iterations − completed` trials, and an explicit `study_name` bypasses the ceiling | ALREADY RECORDED (OPEN): `SESSION_CHANGELOG_20260312_S138.md:69-72` |
| P4 | Consequence of P3 for coverage: under the four-session plan as written, a chunk could be published and certified after zero trials ran | NEW. Read path only, not executed (Q3) |
| P5 | The certified row has `study_identity = NULL`, because the study name is captured only on the `n_parallel > 1` path | NEW (Q2) |
| P6 | `step1_trial_history.session` is NULL in every row. The writer reads `time_of_day`/`session` but the trial carries `session_idx` | NEW (Q2) |
| P7 | The finalizer records no dataset identity and merges a run on a new dataset onto a generation scored on the old one | ALREADY RECORDED as a ruling not yet implemented ("6-P1 accumulator input wall = exact input-manifest digest match"): `TFM_PROJECT_FACTS_SKILL.md:638-646`, `PHASE_6_P0_SCOPING_v1.md:470`, `DATASET_PUBLICATION_SCHEMA_v1.md:111`. Status, not finding |
| P8 | A regime-shift rerun with no new draws skips Step 1 through the freshness check | NEW (bounded search above) |
| P9 | `chapter_13_triggers.py --approve` and `chapter_13_orchestrator.py --approve` run nothing (`watcher_agent=None`) | ALREADY RECORDED: `CHAPTER_13_AUDIT.md:99`, `CHAPTER_13_LIVE_FEEDBACK_LOOP.md:892`. Row 8 of the lineage report's chain (`CLAUDE_CODE_REPORT_STEP1_PURPOSE_LINEAGE.md` §5.2) is inconsistent with that audit |
| P10 | On the `dispatch_learning_loop` route, COMPLETE is logged as `STEP_FAILED` rather than as a clean stop | ALREADY RECORDED as a route: `CLAUDE_CODE_REPORT_WATCHER_FOLLOWUP_REPAIRS.md:68` (its §5 item 3) |
| P11 | The S179 governance gates (§3, §4, §5) and the delegated-run gates G2/G3 are not wired | ALREADY RECORDED: `CLAUDE_CODE_REPORT_DELEGATED_RUN_GATES.md` §5 (`:414-447`), as reported by the helper. The S179 absence is a bounded code search (Q8) |
| P12 | The WATCHER cursor overwrites any operator `seed_start` when the cursor is above 0, so no WATCHER route can request a covered interval | ALREADY RECORDED in substance: Beta X-RUN4-RANGE Q1/Q2, lineage report §5. Status |
| P13 | Options in Q9 | Each is a design input. Q9 labels each one CHANGES A RULING where it does |

**Not searched:** the host outside VM101's repository, ser8 archives, the private repository,
project PDFs, rigs, the Proxmox hosts, the miner ledger files (`~/miner_staging`,
`~/miner_ledger_archive`; not opened), the Optuna study databases, and live systemd or cron state.

---

## First-step output

Saved verbatim in `data/step1_rerun_a/first_step.txt`. Recorded at 2026-11-13T20:32:16-08:00.

- `pgrep -af "beta_relay|range_miner_coordinator|watcher_agent|window_optimizer"` returned exactly
  two lines, PIDs 24698 and 24704. Both are the Bash shell running the `pgrep` itself: the pattern
  matched the shell's own command text, the hazard named in the box-wide rule 8. **No relay,
  coordinator, WATCHER or optimizer process was running.** Before the SQLite read, a second
  `pgrep -l` matched only `34213 bash`, the same self-match.
- `git status --porcelain` in the main tree was empty. In the worktree, after creating scratch, it
  showed `?? data/` (scratch only; no `.py`).
- `git worktree list` at that time:
  ```
  /home/michael/distributed_prng_analysis    87b3182 [main]
  /home/michael/tfm_worktrees/owner_land     5e100b3 (detached HEAD)
  /home/michael/tfm_worktrees/rm_gates       fbd8264 (detached HEAD)
  /home/michael/tfm_worktrees/rm_reads       30d94bd (detached HEAD)
  /home/michael/tfm_worktrees/step1_rerun_a  87b3182 (detached HEAD)
  /home/michael/tfm_worktrees/wp1_stage_a    0066b6e (detached HEAD)
  /home/michael/tfm_worktrees/wp2_stage_a    0066b6e (detached HEAD)
  ```

**SQLite, under addendum §A:**

- Before opening, `stat` of `/home/michael/distributed_prng_analysis/prng_analysis.db*` showed
  only `prng_analysis.db`: 106496 bytes, mtime `2026-08-21 21:29:03.138555356 -0700`. There was no
  `-wal`, `-shm` or `-journal` file.
- I opened `file:...prng_analysis.db?mode=ro&immutable=1` with plain SQL only, and never
  constructed `CoverageLedger` or `DistributedPRNGDatabase`.
- After reading, the size and mtime were identical, and the main-tree directory listing filtered to
  `prng_analysis*` was unchanged (`DIR UNCHANGED`). Nothing was created beside the original. The
  only `diff` line was path formatting (relative before, absolute after). Files:
  `data/step1_rerun_a/db_stat_{before,after}.txt` and `dir_{before,after}.txt`. Query output:
  `data/step1_rerun_a/ledger_read.txt`.

---

## Summary (ten lines)

1. **Null results first.** No route exists today to run Step 1 again on seeds already certified: not WATCHER, not Chapter 13, not the dispatcher. Steps that need a person (R5) appear at 7 of the 11 points of the unattended walk. Five are gaps with no ruling, and the delegated-run and S179 gates that would lift the launch gate are not wired.
2. Exactly one row is certified: `java_lcg`, `{constant, variable}`, `[0, 2³¹)`, `study_identity NULL`. The cursor is therefore OPEN at 2³¹, and the next WATCHER Step 1 run covers `[2³¹, 2³¹+max_seeds)`.
3. **R3 is not met.** The cursor tests only family and mode set. Dataset is stored only as a digest on the coverage row. Session, window, thresholds, study and trial count are absent from the row and the sidecar. The trial-level record (`step1_trial_history`) has NULL session, a non-unique run id, and silently drops a second search's trials.
4. **R1 and R2 are met across runs for seeds only.** The first-gap cursor picks up after the last *published* chunk. A run that dies leaves no coverage. A checkpoint resume exists for `n_parallel == 1`, but needs a person to supply the run id and `--resume-study`.
5. **R2 for Optuna is broken as written.** The plan resumes one study across four chunks with an unchanged trial budget. The code runs `max_iterations − completed` trials, so chunk 2 onward would run 0 trials. Nothing found refuses publishing and certifying that chunk (read path, not executed).
6. **R4 is blocked twice.** After COMPLETE, WATCHER builds no command. Before COMPLETE, a regime-shift rerun with unchanged draws is skipped by the Step 1 freshness check. With new draws it advances the frontier rather than repeating the search.
7. A second search of certified seeds, if it could be published, would be merged onto the prior generation by best per-seed score across datasets. It would add a second overlapping coverage row that changes no cursor. There is no dataset input wall (ALREADY RECORDED as a ruling not yet implemented).
8. A new dataset version changes the frozen dataset identity and invalidates a checkpoint resume. It does not change the cursor, the residues for a given window anchored from the start of the file, or the generation lineage.
9. Run 4 needs the cursor OPEN at exactly 2³¹. The soak needs it OPEN with room for its range. F4 as worded ("COMPLETE over `[0, 2³²)`") would then refuse the soak.
10. Four options are given, A to D. B, a search-scoped cursor beside the historical union, is the only one that meets R1–R4 without discarding history. It CHANGES the 2026-08-07 cursor ruling's scheduling role and the F4 definition, and needs the owner. R5 then depends on wiring the existing unwired gates, not on the cursor.

---

## Q1 — Who reads the cursor

Search: `grep -rn --include=*.py -E "certified_cursor|first_uncovered_seed|CURSOR_STATUS_COMPLETE|CURSOR_STATUS_OPEN|is_complete|record_publication|get_next_seed_start|CoverageLedger\("` plus `seed_coverage_ledger` imports. Results by file:

| Consumer | What it reads | On OPEN | On COMPLETE | Anchor |
|---|---|---|---|---|
| `utils/seed_coverage_ledger.py` | `certified_cursor(prng_base, required_modes)` → `first_uncovered_seed` over records with matching `prng_base` and `required_modes ⊆ executed` | `CursorResult(OPEN, next_seed_start=first gap)` | `CursorResult(COMPLETE, next_seed_start=None)` | `:1029-1055`, `:994-1019`, `:551-605` |
| `database_system.py` (wrapper) | `get_certified_cursor` canonicalises `prng_type` and `test_both_modes`, with no default for `test_both_modes`. It constructs `CoverageLedger` lazily (runs `init_schema` on a writer) | returns the result | returns the result | `:350-355`, `:357-385` |
| `database_system.py` `get_next_seed_start` | the same cursor | returns the int | returns `None` | `:387-433`. Production callers: **none** (the only callers are in `tests/test_seed_domain_cursor_amendment.py`; `apply_s140_*` is a one-shot patch script) |
| **WATCHER** `run_step` | `_db.get_certified_cursor(prng_type, test_both_modes=bool(...))`, only when `step == 1 and 'seed_start' in final_params` (always true under the manifest default `seed_start: 0`) | cursor > 0: **overwrites** `final_params['seed_start']`. No explicit `study_name` forces `resume_study=False`; an explicit one forces `resume_study=True`. Cursor 0: `seed_start` is left as supplied, though the log says "using seed_start=0". Then the domain wall checks `seed_start + max_seeds` | returns `{success: False, blocked_by: "seed_domain_complete", seed_domain_complete: True, next_seed_start: None}`; no command is built. A cursor exception blocks Step 1 (`seed_coverage_cursor_unavailable`) | `agents/watcher_agent.py:2027-2066`, `:2068-2085`, `:2087-2110`, `:2126-2140` |
| WATCHER `run_pipeline` | `results.get("seed_domain_complete") is True` | — | `break`: clean stop, no halt file, no evaluation, **no later steps** | `:2902-2909` |
| `agents/watcher_dispatch.py` `dispatch_learning_loop` | the `run_step(step)` result | — | `success is False` → `STEP_FAILED`, loop aborted (COMPLETE is logged as a failure) | `:344`, `:365-381` **(re-read)**; ALREADY RECORDED `CLAUDE_CODE_REPORT_WATCHER_FOLLOWUP_REPAIRS.md:68` |
| **Finalizer ledger write** | not a reader. `record_certified_coverage(artifact, dataset_sha256=frozen.sha256, study_identity=_mp_study_name or None)` right after `finalize_run` returns | — | — | `window_optimizer_integration_final.py:3056-3060` |
| `window_optimizer.py` CLI | does **not** read the cursor. It applies the domain wall only, so a direct CLI run starts wherever `--seed-start` says | n/a | n/a | `window_optimizer.py:756-770` |
| `window_optimizer.py` `run_with_config` | does **not** read the cursor. The plan is `[i·max_seeds, +max_seeds)` from 0; the wall checks the whole plan | n/a | n/a | `:1132-1173` |
| Launch harness `tests/run4_prelaunch_provenance_check.py` | stub `get_certified_cursor`, `is_complete = False`, cursor fixed at 0 | — | — | `:136-152` (`:139`, `:147`) |
| Fixture `tests/fixtures/run4_routing_clean_control.py` | stub, `is_complete = False` | — | — | `:79-90` |
| `tests/test_gate12_cleantree_admission.py` | stub `get_certified_cursor` in generated source | — | — | `:440` |
| `tests/test_watcher_followup_repairs.py` | anchors on `if _cursor.is_complete:` | — | — | `:183` |
| `tests/test_seed_domain_cursor_amendment.py` | the full suite (G-CURSOR-FIRST-GAP, G-CURSOR-COMPLETE, G-PUBLICATION-BINDS-COVERAGE, R1-A/B/C …) | — | `arm_complete_generates_no_watcher_run` | `:1584-1641` |
| `miner/range_miner_coordinator.py` `is_complete` | **not a consumer.** It is per-stripe reconciliation | — | — | `:1107-1110`, `:4278` |

---

## Q2 — What is recorded, against R3

**Live certified ledger** (`data/step1_rerun_a/ledger_read.txt`): one row.

```
run_id step1_java_lcg_0 · study_identity None · prng_base java_lcg · modes constant,variable
seed_domain_contract v1.1-stratum · [0, 2147483648) · dataset_sha256 513648160d35…68f6
repository_commit e9ca800a… · artifact_sha256 aecc5d50… · generation_id gen-20260818T013219550933Z-step1_java_lcg_0
recorded_at 2026-08-18T01:32:19Z
```

**Where each item is stored.** Column (1) is the coverage row (`utils/seed_coverage_ledger.py:646-666`).
Column (2) is the generation sidecar (`utils/run_finalizer.py:222-255`, built `:1660-1700`). Column (3)
is the miner ledger's `trial_context` (`miner/range_miner_coordinator.py:1384-1400`). Column (4) is
`step1_trial_history` (live schema in `ledger_read.txt`; `database_system.py:123-150`). Columns (2)
and (3) come from the helper and were spot-checked: the finalizer has zero `dataset` matches
(**re-read**: `grep -c -i dataset utils/run_finalizer.py` returned `0`).

| Item | (1) Coverage row | (2) Sidecar | (3) Miner trial_context | (4) step1_trial_history | Tested by the cursor? |
|---|---|---|---|---|---|
| Dataset version | `dataset_sha256` only; no version_id | **nowhere** | `dataset_sha256`, `residue_sha256` | **nowhere** | **No** (R1 ruling, `:884-888`) |
| Session | no | no | `sessions_json` | column exists; **NULL in all 56 live rows** (P6) | No |
| Window, thresholds | no | no (per-row NPZ arrays carry `window_size` and `offset`, no thresholds) | `window_size`, `window_anchor_val`, `generator_phase`, `forward_threshold`, `reverse_threshold` | `window_size`, `offset`, both thresholds | No |
| Skip settings | `skip_modes_executed` (set) | `skip_modes_executed`; per-row `skip_min`/`skip_max`/`skip_mode` | `skip_min`, `skip_max` | `skip_min`, `skip_max` | **Mode set only** (containment) |
| Study identity | `study_identity`, nullable. **NULL in the live row** (P5) | no | no | `study_name` = the operator-supplied value, `''` unless set | No ("never consulted by the cursor", `:884-890`) |
| Run id | `step1_{base}_{start}` | same | `{stem}_t{n}_{uuid8}` (`:12473`) | `step1_{base}_{start}`, part of the UNIQUE key | No |
| Trial count | no | no (`raw_candidate_count`, `l2_winner_count` only) | one trial's number | derivable, but lossy (below) | No |

**Mechanisms behind the gaps, as read:**

- **P5, `study_identity` NULL.** The write passes `locals().get('_mp_study_name')`
  (`window_optimizer_integration_final.py:3059`). `_mp_study_name` is assigned only inside
  `if n_parallel > 1:` (`:2080`, assignments `:2311-2343`). The certified path is `n_parallel == 1`
  (manifest default `n_parallel: 1`; the D6.2 scope wall `:2027`). There, the study name lives in
  `window_optimizer_bayesian.py:662,738` (`window_opt_<time>`) and never reaches the ledger.
- **P6, session NULL.** The writer stores `params.get('time_of_day', params.get('session'))`
  (`database_system.py:476`). The objective's parameter is `session_idx`
  (`window_optimizer_bayesian.py:535`, `:557`). Neither key is present, so the value is NULL.
- **P1/P2, run id reuse and dropped trials.** Schema `UNIQUE(run_id, trial_number)` (live) plus
  `INSERT OR IGNORE` (`database_system.py:465-466`); exceptions are swallowed (`:482-483`). The
  run id is `step1_{prng_base}_{seed_start}` (`window_optimizer_integration_final.py:2833`).
  **Live evidence.** Run id `step1_java_lcg_0` holds trials 0–1 recorded 2026-04-23 with
  `seed_range_end 1073741824`, and trials 2–8 recorded 2026-08-21 with `seed_range_end 10000000`.
  These are two different searches under one key, and the August run's trials 0–1 are absent. That
  pattern is consistent with the silent drop; it does not prove which run wrote what. The certified
  attempt-9 run (2026-08-18, `[0, 2³¹)`) has no row whose `seed_range_end` is 2147483648. The live
  table therefore cannot say which settings produced the one certified interval. That record
  exists only in the miner ledger, which was not opened here.
- `write_downstream_score` (Chapter 13) updates `WHERE run_id=?` (`database_system.py:489-494`;
  caller `chapter_13_orchestrator.py:313`). A hit-rate annotation for one search therefore lands
  on every search sharing that seed start.

**The checkpoint run context binds more than the coverage row.** `run_context_components` binds
dataset version_id, filename and sha256, repository commit, prng_base, modes, seed range and
`execution_set_id` (`utils/checkpoint_d6_2.py:502-553`; `window_optimizer_integration_final.py:1002-1015`).
It still does not bind window, session, threshold or trial budget.

---

## Q3 — R1 and R2 as built

| Continuation | Mechanism | Works on the RANGE-MINER path today? | Anchor |
|---|---|---|---|
| **Within one run** (checkpoint + resume) | D6.2 checkpoint under `.s172_checkpoint/<run_id>/`. The run id is `PRNG_CHECKPOINT_RUN_ID`, else `hostname-pid-…`. Resume needs `--resume-checkpoint <id>` **and** `--resume-study`, and a loaded study with trials. A digest mismatch (dataset, commit, range, execution set) fails closed | Scoped to `n_parallel == 1`; `n_parallel > 1` is rejected. Helper: the miner ledger keeps its own per-trial run ids, and coordinator death is "not resumable" per the delegated proposal §8 (`:192`) | `window_optimizer_integration_final.py:429,486-494`, `:2027-2045`, `:1001-1088`; `utils/checkpoint_d6_2.py:1024-1047`; `window_optimizer_bayesian.py:719-765` |
| **Across runs** (the cursor) | First gap in the certified union. Credit only after publication | Yes. The WATCHER block is upstream of the backend choice ("on the PWC path and the miner path alike", `watcher_agent.py:2021-2025`) | Q1 |
| **Across runs, Optuna** (`study_name`) | WATCHER keeps continuity only if `study_name` is set explicitly (`:2090-2108`). Without a name, `resume_study=True` picks the **newest** `window_opt_*.db` by mtime, regardless of range (`window_optimizer_bayesian.py:664-705`) | Mechanically yes, but see P4 | — |

**Chunk size.**
- WATCHER uses `final_params['max_seeds']`, with a fallback of 10,000,000 if absent
  (`watcher_agent.py:2045-2047`).
- The manifest default is **`max_seeds: 1073741824` (2³⁰)**, the four-session chunk, in
  `agent_manifests/window_optimizer.json` `default_params`. It maps to `--max-seeds` through
  `args_map` (`"max-seeds": "seed_count"` in the action `args_map`; WATCHER reads `max_seeds`).
- The miner splits a chunk into 2²⁶ macro-stripes (`miner_stripe_size` 67,108,864;
  `miner/range_miner_coordinator.py:603`, `:619`).
- **What stops a session at 2³⁰:** nothing but that `max_seeds` value. The run sweeps
  `[cursor, cursor + max_seeds)` and then ends. The domain wall refuses only past 2³²
  (`:2126-2140`).
- No loop starts the next chunk. A person re-invokes (Q8).

**P4: the four-session plan's Optuna continuity runs zero trials (read path, not executed).**
- The plan resumes one study for runs 2–4 by setting `study_name` and changing nothing else
  (`docs/PROPOSAL_S145_R1_Progressive_Empirical_Sweep.md` §3.3 and §6.3, `:260-268`; 50 trials per
  session, §4).
- The code runs `_trials_to_run = max_iterations - _resumed_completed`
  (`window_optimizer_bayesian.py:806`, `study.optimize(n_trials=_trials_to_run)` at `:827`). With
  an explicit `study_name` the resume is taken even when `completed ≥ max_iterations` (`:692`).
- So run 2, with `window_trials` unchanged, runs `50 − 50 = 0` trials over its new range.
- The finalizer has no minimum-candidate check: attempt 9 legitimately published zero rows, and no
  `empty`/`zero` refusal was found in `utils/run_finalizer.py`.
- The coverage write follows any successful `finalize_run` (`window_optimizer_integration_final.py:2992-3060`).
- **Prediction tied to hit rate:** a chunk certified this way contributes no survivors to the pool
  scored against `k/1000`. It still removes that range from the cursor, so no later run searches
  it. **Measurement that settles it:** a CPU-only harness run with an explicit `study_name` whose
  completed count equals `window_trials`, intercepting dispatch. Record `_trials_to_run`, whether
  `finalize_run` is reached, and whether `record_certified_coverage` is called. This session ran
  nothing. The overrun side of the same arithmetic is already OPEN (`SESSION_CHANGELOG_20260312_S138.md:69-72`).

**A run that dies part-way:**
- **Coverage ledger:** nothing. Credit is bound to publication (`utils/seed_coverage_ledger.py:38-44`;
  the write sits after `finalize_run`, `:3036-3060`).
- **Generation:** `current` is unchanged; the finalizer is fail-closed (`utils/run_finalizer.py:1559-1571`).
- **Checkpoint:** survives under its run id.
- **`step1_trial_history`:** rows for completed trials.
- **Miner ledger:** stripe states.
- **Where the next run starts:** the next WATCHER run starts at the same cursor, the dead chunk's
  start. Without `--resume-checkpoint` it is a **fresh** search of that whole chunk, and its trials
  collide in `step1_trial_history` on the same run id (P2).

---

## Q4 — The regime-shift rerun

Documented chain: `CHAPTER_13_LIVE_FEEDBACK_LOOP.md:117-120` (Step 1 "re-run only on regime
shift"), `:504-511` (§10.4), `:755-780`; recorded in full in `CLAUDE_CODE_REPORT_STEP1_PURPOSE_LINEAGE.md`
§5 (`:586-660`, re-read).

**Trigger.** The orchestrator daemon polls `new_draw.flag` (`chapter_13_orchestrator.py:509-516`).
Diagnostics read `lottery_history.json` (`chapter_13_diagnostics.py:52,781`).
`REGIME_SHIFT_POSSIBLE` requires both decay > 0.5 and churn > 0.4 (lineage report §5.2 rows 1–4).
`evaluate_triggers` raises REGIME_SHIFT → FULL_PIPELINE → steps `[1..6]`
(`chapter_13_triggers.py:290-302`, `:313-319`, `:454-455`).

**The three human gates, as enforced:**

1. **Approval always required.** `requires_approval=True  # v1: always require approval`
   (`chapter_13_triggers.py:369`). The request is written to `pending_approval.json` (`:419-448`).
   The orchestrator's "auto" branches only set a label (`chapter_13_orchestrator.py:410-425`).
   Policies `test_mode:false` and `auto_approve_in_test_mode:false` (`watcher_policies.json:7,102`).
2. **The WATCHER daemon does not approve.** It dispatches only if `test_mode` and `auto_approve`
   (`agents/watcher_agent.py:3398-3433`). Otherwise it sends a notice and puts the request back to
   pending (`:3436-3477`).
3. **A person runs `agents/watcher_agent.py --approve`** (`:3845-3883`). That calls
   `watcher.run_pipeline(min(steps), max(steps))` **with no params** (`:3878`, **re-read**). The
   other two `--approve` commands run nothing (`chapter_13_triggers.py:895-899`, `:603-608`;
   `chapter_13_orchestrator.py:124-128`, **re-read**). ALREADY RECORDED, P9.

**How it reaches the cursor.**
- With no params, Step 1 gets manifest defaults: `seed_start 0`, `study_name ""`,
  `resume_study false`, `max_seeds 2³⁰`, `window_trials 3`, `use_range_miner false`,
  `resume_checkpoint ""`.
- **Before the cursor, the freshness check runs:** Step 1 is skipped as "fresh" when its output is
  newer than its input (`agents/watcher_agent.py:1835-1858`, **re-read**). Only `--force-step`
  bypasses it, and that is CLI-only (helper: `:3694`).
- **P8 (NEW):** a regime shift raised from decay and churn with **no new draw file** returns
  `{success: True, skipped: True}` for Step 1, and the pipeline goes on to Steps 2–6 on the old
  survivors.

**What it would do today:**

- **OPEN at 2³¹** (the live state). With new draws, so Step 1 is not fresh, the cursor's 2³¹
  overwrites `seed_start` (`:2087-2089`). No `study_name` means a fresh study (`:2094-2096`). The
  run is `[2³¹, 2³¹+2³⁰)` with 3 trials on the PWC path, since `use_range_miner` defaults to
  false. It is **not** a rerun of the searched range: it advances the frontier. Beta built
  `--seed-start 2147483648 --max-seeds 2147483648` only with Run-4 pins
  (`X-RUN4-RANGE_R1_BETA_REVIEW.md` Q1). Without new draws, Step 1 is skipped (P8).
- **COMPLETE.** `run_step` returns `seed_domain_complete` (`:2068-2085`). `run_pipeline` breaks
  (`:2902-2909`), so Steps 2–6 of the FULL_PIPELINE also do not run. The approve path archives the
  request as `completed_by_cli` anyway (`:3880-3883`, **re-read**). On the dispatcher route the
  same result is `STEP_FAILED` (P10).

---

## Q5 — Publishing a second search of the same seeds (read only)

There is no WATCHER route to it (Q1, P12). A direct `window_optimizer.py --seed-start 0 …` run is
refused only by the domain wall (`window_optimizer.py:756-770`). If such a run published:

- **Merge with the prior generation.** `finalize_run` uses `current` as the prior automatically. A
  supplied prior must match it (`utils/run_finalizer.py:1269-1313`, `:1640-1650`).
  - L2 within the run keeps one winner per seed (`:690-748`).
  - L3 against the prior matches by exact seed. **A new row replaces a prior row only on a strictly
    greater float32 score; every unmatched prior seed is kept** (`:786-806`, **re-read**).
  - Scores computed on two datasets or two configurations are compared directly. Nothing in the 22
    arrays identifies the dataset (`utils/canonical_arrays.py:98-123`).
- **Parent identity.** `current` is a symlink under `<cwd>/.s172_accumulator/`
  (`utils/run_finalizer.py:124-126`, swap `:449-460`, `:1512-1515`). The sidecar carries
  `parent_generation_id` and the parent artifact and sidecar sha (`:1663-1665`). The chain is linear
  and validated to the root (`:1176-1258`). `generation_id` = `gen-<timestamp>-<run_id>`
  (`:1652`, **re-read**), so a repeated run id still gets a new directory.
- **Input-identity wall.** The prior is checked for schema and encoding versions, `canonical_map_hash`,
  **`prng_base` equality (modes may differ)** (`:1340-1358`, **re-read** `:1352-1358`), ordering and
  the chain. The current run is checked for a non-empty run id (`:1581`), a clean tree (`:1589-1596`)
  and candidates inside its seed range (`:1598-1612`).
  - **Not checked:** dataset digest (no `dataset` in the file), window, session or threshold, or any
    relation between the seed range and the prior.
  - ALREADY RECORDED as an unimplemented ruling (P7).
- **Ledger write.** A different artifact, generation or dataset gives a different `coverage_id`
  (`utils/seed_coverage_ledger.py:762-774`, `:821-841`), so a bare `INSERT` adds a **second,
  overlapping row** (`:957-977`). The cursor merges overlaps for computation only (`:538-541`), so
  it does not move. History keeps both rows. The append-only triggers are untouched.
- **Trial history.** Collides on `(step1_java_lcg_0, n)` and is silently dropped (P2).

---

## Q6 — New draws

| Part | Changes? | Anchor |
|---|---|---|
| Dataset authority | **Yes.** `FrozenDataset` (path, filename, sha256, size, record_count, version_id, lineage_id, predecessor_sha256 …). Frozen once per process; a different identity in the same process raises. It binds the coordinator's per-run digest, `trial_context.dataset_sha256` and the coverage row's `dataset_sha256` | `miner/dataset_authority.py:229-254`, `:594-623`, `:659-675`; `agents/watcher_agent.py:749-763`; integration `:3058` |
| Residue window | **No, for a given (window_size, offset, sessions).** The window is anchored from the start of the file (`data[anchor:anchor+window_size]`), and `daily3.json` is ordered oldest first. An appended draw leaves those residues unchanged; it changes the digest and the anchor domain. The order inside the published version file is [UNVERIFIED] | `miner/range_miner_worker.py:906-931`; `reverse_sieve_filter_INTEGRATED.py:57` (clamps instead); `window_optimizer.py:74,143` |
| Cursor | **No.** Identity is `(prng_base, modes)` | `utils/seed_coverage_ledger.py:271-320`, `:994-1019`; R1 ruling `:884-888` |
| Generation lineage | **No.** The sidecar has no dataset field, and a new-dataset run merges onto the old-dataset `current` (Q5) | `utils/run_finalizer.py:222-255` |
| Checkpoint resume | **Yes, invalidated.** The digest includes the dataset | `utils/checkpoint_d6_2.py:536-553`, `:1024-1047` |
| Step 1 freshness | **Yes.** Newer input makes Step 1 not fresh, so it will run and advance the cursor (Q4) | `agents/watcher_agent.py:1835-1858` |

Fact file §2.10b (`docs/TFM_PROJECT_FACTS_SKILL.md:632-648`, quoted by the helper): *"production
re-optimization is **per-session**"*; *"Append-only does NOT make prior scores valid on the next
version … Prefix-only merging **not approved**"*; the accumulator input wall is an *"exact
input-manifest digest match"*; *"input identity is a lineage invariant, not annotation."* The code
in this table does not yet meet that text (P7).

The note above about the residue window does not make old scores valid on new draws. It is a fact
about one input to a score, not about the score.

---

## Q7 — The soak and Run 4

- **Run 4.** Pins `max_seeds 2147483648`, `window_trials 1`, `n_parallel 1`, `--start-step 1
  --end-step 1` (`docs/S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md:26-31`). The interval is open at A3
  (`docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:77`). It needs **OPEN with the cursor at exactly
  2³¹** (2³¹ + 2³¹ = 2³² passes `:2126-2140`). That is the live state. The ruling's "full 2³¹ seed
  domain used by Attempt 9" (`docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:148`), i.e.
  `[0, 2³¹)`, is not reachable through WATCHER, because the cursor overrides `seed_start`. Beta
  already accepted the upper half under narrowed claims (X-RUN4-RANGE Q1).
- **Soak.** 50 trials (overriding the manifest's 3), `n_parallel 1`, empty `resume_checkpoint`, 25
  workers (`docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_7_SOAK.md:114-120`), as `--start-step 1
  --end-step 1` (completion plan C2, `:101`). The doc names no seed range (helper grep). It needs
  **OPEN** with `cursor + max_seeds ≤ 2³²`. Ordered after F4, it cannot start (Beta Q2 LIVE).
- **F4, verbatim** (`docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:59`):
  > | F4 | The whole space is swept | The coverage ledger reports `java_lcg`, modes `{constant, variable}`, **COMPLETE** over `[0, 2³²)`, with no host freeze during the sweep |

---

## Q8 — Every manual step and human gate (R5)

The walk assumes nothing runs unless code starts it. "Ruling + lift" names what the ruling says
must be shown before the gate can be removed.

| # | Step | Who acts today | Gate or gap | Ruling, and what lifts it | Anchor |
|---|---|---|---|---|---|
| 1 | A new `daily3.json` version arrives | Person runs `daily3_scraper.py --json --recent`. The systemd unit points to a script that does not exist | **GAP** (owner: intentional for now, `TEAM_ALPHA_DATASET_LIFECYCLE_FINDINGS.md:80-83`; Alpha proposes a reviewed deliverable, `:85-87`) | none | `daily3_scraper.py:14,99` |
| 2 | Chapter 13's input (`lottery_history.json` + `new_draw.flag`) | Person drops a file in `incoming_draws/` or runs the injector. Nothing connects it to `daily3.json` | **GAP** | none | `draw_ingestion_daemon.py:45-48,265-282` |
| 3 | Trigger → approval | Person | **DELIBERATE** | Ch13 v1 `:33`; §20 "only activate after stability is proven" (`:1400`); S179 §8 BOOTSTRAP, "no autonomous retraining enabled" (`docs/TB_RULING_S179_IMPLEMENTATION_AUTH.md:513-532`). **Lift:** reach GOVERNED with S179's three binding gates built. Not found in code (`manual-governance-override|BLOCKED_BY_GOVERNANCE|audit_only`, all `*.py`) | `chapter_13_triggers.py:369`; `agents/watcher_agent.py:3398-3477` |
| 4 | The approve that executes | Person, `watcher_agent.py --approve` | DELIBERATE (as 3). The two inert `--approve` commands are a **GAP** (P9) | as 3 | `:3845-3883` |
| 5 | Search chosen (pins, trials, miner on) | Person, `--run-pipeline --params` | **DELIBERATE** for pins (Route-A: the operator CLI is the only privileged seam). Manifest `use_range_miner:false` and `window_trials:3` are **GAP** (completion plan C4, `:103`) | `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:12-24` | `agents/watcher_agent.py:3919-3932` |
| 6 | Launch | Person | **DELIBERATE** | `CLAUDE.md` §1 rule 3. **Lift:** `PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md` §11 items 1–6 (`:222-229`) plus §12 text. G2/G3 are committed but unwired; `config_manifests/run_authorizations/` and `env_freeze_baseline.cfg` do not exist (helper `ls`). Owner: "Agents can launch. That is the whole point. We need guard rails." (`:30`) | — |
| 7 | Chunking | Code: `max_seeds` | none (automatic) | — | Q3 |
| 8 | The run dies | WATCHER escalates and writes a halt file; person runs `--clear-halt` | **DELIBERATE** | Delegated §4.9 "No relaunch after a failure", which no authorization unlocks (`:67`, `:83`); §8 coordinator death "not resumable" (`:192`). **No lift condition is stated** | `agents/watcher_agent.py:968-993`, `:1599` |
| 9 | Resume | Person supplies the checkpoint run id (`hostname-pid-…`, found in the log), `--resume-study` and the study name | **GAP**: no ruling asks a person to carry the id. Discovery by "newest directory" is deliberately banned (D6.2 §4.1), so automation needs a recorded handle | — | Q3 |
| 10 | Publish + coverage credit | Code | none (automatic, fail-closed) | — | Q3 |
| 11 | Next chunk | Person re-invokes. No loop in WATCHER or Chapter 13 | **GAP** (no loop). The launch itself is gated by 6. Under delegation, the SWEEP class would cover it once §11 is met (`:199-201`) | — | `agents/watcher_agent.py:2860-3027` |
| 12 | Continue the Optuna study across chunks | Person edits `study_name` in the manifest (plan §6.3) and must also raise `window_trials`, which the plan does not say (P4) | **GAP** | — | `PROPOSAL_S145_R1…md:260-268` |
| 13 | Rerun called for, with draws unchanged | Step 1 skipped by freshness (P8) | **GAP** | none | `:1835-1858` |
| 14 | Rerun called for, after COMPLETE | No route exists | **GAP**, pending owner ruling (Beta Q2 LIVE). A fix CHANGES A RULING | `CLAUDE_CODE_INSTRUCTIONS_SEED_DOMAIN_CURSOR_R1.md:16` | `agents/watcher_agent.py:2068-2085` |

**Counted against the brief's walk** (arrive, choose, start, chunk, die/resume, publish, next
chunk, rerun), with 11 points:

- **Deliberate gates:** 3/11 (trigger approval, launch, relaunch after failure). Pin selection is
  also deliberate, inside "choose".
- **Gaps:** 5/11 (data fetch, resume handle, next-chunk loop, study continuation, rerun with no
  route).
- **Person-free:** 3/11 (chunking, publish, credit).

**R5 depends on the gates in rows 3 and 6 being built and wired.** The cursor cannot supply that.

---

## Q9 — Options (not implemented)

Beta's keep-list holds for every option: domain bounds `[0, 2³²)`; publication before credit;
append-only history; fail-closed lookup; no reset or deletion of old credit.

**Cross-cutting items every option except A needs anyway.** Each is a status or a NEW item above,
not a part of the option:

- (x1) a unique search or run identity in `step1_trial_history`, together with the
  `downstream_score` join (P1/P2, recorded);
- (x2) the trial budget on a resumed study (P3/P4);
- (x3) the study name reaching the ledger on `n_parallel == 1` (P5);
- (x4) session recorded (P6);
- (x5) the dataset input wall in the finalizer (P7, an existing ruling);
- (x6) the regime rerun not skipped by freshness (P8).

### A — Change nothing; document it

- **R1/R2:** seeds yes; Optuna continuity no (P4 stands).
- **R3:** no.
- **R4:** no. COMPLETE is terminal, and a rerun needs a person to run `window_optimizer.py`
  directly, outside the cursor, with its credit merged as in Q5.
- **R5:** unchanged (Q8).
- **Edits:** documentation only (fact file, plan F4 note, Chapter 13).
- **Rulings changed:** none, but it is inconsistent with the owner's R4 and with S145-R1's
  rejection of "Step 1 retired permanently" (`:22`).
- **Suites and gates:** none.
- **COMPLETE / F4:** the end of Step 1 for `java_lcg` on this domain; the soak cannot run after F4.

### B — Search-scoped cursor beside the historical union (Beta's suggested shape)

- **Design:**
  - Keep `certified_coverage` and its union cursor unchanged, as **historical traversal**.
  - Add an append-only **search** record. Its identity: frozen dataset id (version_id + sha256),
    session selection, a digest of the declared search space and configuration, the study name,
    and an optional `repeat_of` / `reason` (`new_dataset`, `regime_shift`, `operator`).
  - Coverage rows gain a `search_id` (new column, or a sibling table keyed by `coverage_id`).
  - WATCHER asks for the **search** cursor: the first gap among rows with that `search_id`.
    COMPLETE now means "this search finished its traversal".
  - A new search is opened explicitly: automatically when a new dataset version is frozen, or by
    a trigger or operator request through the existing gates. A resume continues the same
    `search_id`.
- **R1:** yes, within a search. A dead run resumes at its search's first gap.
- **R2:** yes. Chunks are `max_seeds` within a search; needs x2.
- **R3:** yes, as the search identity; needs x3/x4.
- **R4:** yes. A new dataset version or a regime shift opens a new search over `[0, 2³²)`.
- **R5:** the search choice and opening can be code (new frozen dataset → new search). Launch and
  approval stay gated (Q8 rows 3 and 6) until S179 and the delegated gates are wired.
- **Edits:**
  - `utils/seed_coverage_ledger.py` (schema, and a migration that leaves old rows untouched with a
    NULL `search_id`, read as "historical search 0");
  - `database_system.py` (wrapper);
  - `agents/watcher_agent.py` S145 block `:2027-2110`;
  - `window_optimizer_integration_final.py` `:3056-3060`;
  - `agent_manifests/window_optimizer.json` (a search parameter);
  - Chapter 13 trigger payload, for regime reruns.
- **Rulings changed:**
  - Beta 2026-08-07 §4/§6: the cursor law's role as the **scheduling** authority moves from the
    union to the search;
  - the F-1 COMPLETE semantics (`CLAUDE_CODE_INSTRUCTIONS_WATCHER_FOLLOWUP_REPAIRS_20261111.md:57-78`);
  - **not** R1 2026-08-08 for the historical union (dataset stays provenance there). It becomes an
    identity component of a new object, which Beta named as the reviewable alternative
    (X-RUN4-RANGE Q2).
- **Suites and gates:**
  - **Seed-domain suite** (`tests/test_seed_domain_cursor_amendment.py:1584-1641`): G-CURSOR-*,
    R1-B, the schema-drift arm (`_COLUMNS` is asserted exactly, `utils/seed_coverage_ledger.py:732-759`),
    and G-PUBLICATION-BINDS-COVERAGE all need extension. None should be weakened.
  - **Route-A routing suite** (`tests/test_s172_run4_routing_patch.py`): it imports and mutates
    `agents/watcher_agent.py`, so it needs a scoped refresh (Beta Q4).
  - The **fixture and prelaunch stubs** of `get_certified_cursor` change shape.
  - **Gate 22** (`tests/test_s172_phase4_coordinator.py:1621-1700`): `agents/watcher_agent.py`,
    `database_system.py` and `utils/seed_coverage_ledger.py` are not in its `allowed` set, so it
    reds while those edits are uncommitted.
  - **Rig parity set** (`scripts/gate12_parity_gate.py:174-185`): none of these files is governed,
    so no parity change.
- **COMPLETE / F4:** COMPLETE is per search. F4 must be reworded (owner) as "search S over
  `[0, 2³²)` COMPLETE", or "the historical union COMPLETE", which stays reportable. The soak then
  runs as its own search.

### C — Dataset-partitioned cursor key

- **Design:** add `dataset_sha256` (or version_id) to the cursor predicate. A new dataset version
  gives a fresh first gap at 0.
- **R1/R2:** yes, per dataset.
- **R3:** dataset only.
- **R4:** **partly**. New draws reopen it, but a regime shift on unchanged draws, a new
  configuration or a repeat search does not (Beta Q2: "Dataset partitioning alone does not enable
  another geometry or repeated search").
- **R5:** a new freeze reopens automatically; the gates are unchanged.
- **Edits:** `utils/seed_coverage_ledger.py` `:994-1055` and `canonical_coverage_identity`;
  `database_system.py`; WATCHER `:2053`.
- **Rulings changed:** **R1 2026-08-08** (`CLAUDE_CODE_INSTRUCTIONS_SEED_DOMAIN_CURSOR_R1.md:7-16`;
  executable `utils/seed_coverage_ledger.py:884-888`) and the 2026-08-07 identity definition.
- **Suites and gates:** the seed-domain suite's R1-B containment matrix and its
  `arm_canonical_identity_unifies_hybrid_and_base`; Route-A refresh; Gate 22 as in B. Rig parity
  is unaffected.
- **COMPLETE / F4:** per dataset version. F4 would have to name the dataset. Every published draw
  reopens the domain, which at the current throughput may never finish. That is a scheduling fact
  for the owner, not a verdict.

### D — Keep the cursor; add an authorised rerun override

- **Design:** a flag-selectable `--rerun-of <coverage_id|range>` together with a recorded
  authorization (the delegated proposal's authorization file is the natural carrier). It lets
  WATCHER dispatch a covered interval and records a second overlapping row, which is already
  legal (Q5).
- **R1/R2:** unchanged; no resume of a rerun.
- **R3:** only what the authorization records.
- **R4:** yes, manually.
- **R5:** **no**. A person must author each rerun, unless the authorization is generated by a
  governed trigger, which needs S179 wired.
- **Edits:** WATCHER `:2087-2110`; manifest; authorization schema
  (`scripts/run_authorization_gate.py`, unwired).
- **Rulings changed:** the 2026-08-07 §6 statement that the cursor alone selects `seed_start` for a
  certified generation; F-1.
- **Suites and gates:** the seed-domain G-CURSOR-COMPLETE arm `arm_complete_generates_no_watcher_run`
  would need an explicit override arm; Route-A refresh (a new privileged seam, which the R2 review
  says needs mutation evidence, `:12-24`); Gate 22 as above. Rig parity is unaffected.
- **COMPLETE / F4:** unchanged in meaning, with "COMPLETE, rerun authorised" as a state.

**Comparison.**
- **B** is the only option that meets R1–R4 together while keeping the historical union intact.
- **C** is smaller but meets R4 only for new draws.
- **D** is smallest and fails R5.
- **A** fails R3 and R4.
- **All four** leave R5's launch and approval gates where they are. Those are lifted by wiring S179
  and the delegated gates, a separate track that already exists.

These options are design inputs for Alpha's amendment. None is a recommendation to change the
baseline algorithm. The current cursor stays as the baseline, and any option would be
flag-selectable until Beta has reviewed it and the owner has ruled.

---

## UNAVAILABLE

- The miner ledger (`~/miner_staging`, `~/miner_ledger_archive`): not opened. Attempt 9's trial
  settings, its study and whether a zero-trial path occurred there are UNAVAILABLE from this report.
- Optuna study databases (`optuna_studies/*.db`): not opened.
- Execution of P4 (zero-trial resume → publish → credit): not run. It is a read-path finding.
- Execution of P8 (regime rerun skipped by freshness): not run. It is read-path.
- The order inside the published dataset version file: not read.
- Live state of systemd, cron and the Chapter 13 daemons: not checked.
- Route-A mutation suite, seed-domain suite and Gate 22: not run (read-only brief; the Route-A M1b
  arm writes `agents/watcher_agent.py`).
- ser8, rigs, the private repository and project PDFs: not searched.

---

## Commands used

- Worktree: `git worktree add --detach /home/michael/tfm_worktrees/step1_rerun_a HEAD`; `mkdir -p data/step1_rerun_a`.
- First step: `pgrep -af …`, `git status --porcelain` (main tree and worktree), `git worktree list`, `git rev-parse HEAD`, written to `data/step1_rerun_a/first_step.txt`.
- Source reads with `sed -n`, `grep -n`, `/bin/grep` and `awk` over:
  - `utils/seed_coverage_ledger.py`, `database_system.py`, `agents/watcher_agent.py`, `agents/watcher_dispatch.py`;
  - `window_optimizer.py`, `window_optimizer_bayesian.py`, `window_optimizer_integration_final.py`;
  - `utils/run_finalizer.py`, `utils/checkpoint_d6_2.py`, `miner/range_miner_coordinator.py`;
  - `chapter_13_triggers.py`, `chapter_13_orchestrator.py`;
  - `agent_manifests/window_optimizer.json` (via `python3 -c json.load`);
  - `tests/test_s172_phase4_coordinator.py`, `tests/test_seed_domain_cursor_amendment.py`, `tests/test_s172_run4_routing_patch.py`, `scripts/gate12_parity_gate.py`;
  - the docs cited above.
- SQLite: `stat -c '%n %s %y' prng_analysis.db*` before and after; `ls -la --time-style=full-iso | grep prng_analysis` before and after with `diff`; `PYTHONDONTWRITEBYTECODE=1 python3 -I` with `sqlite3.connect("file:…/prng_analysis.db?mode=ro&immutable=1", uri=True)`, plain `SELECT`/`PRAGMA table_info` only, written to `data/step1_rerun_a/ledger_read.txt`.
- Prior-record searches: written to `data/step1_rerun_a/prior_record_search.txt`, plus the follow-up greps listed in the Prior record.
- Two read-only helper agents (Q4/Q7/Q8 and Q5/Q6/Q2-support), instructed: no edits, no SQLite, no runs. Their NEW claims were re-read at source and marked **(re-read)**.

No file outside this worktree was written. No existing file was edited. Nothing was committed, pushed, branched or tagged. Nothing was launched.
