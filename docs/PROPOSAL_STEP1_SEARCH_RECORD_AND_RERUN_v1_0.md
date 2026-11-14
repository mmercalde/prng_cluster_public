# PROPOSAL — STEP 1 SEARCH RECORD AND RERUN — v1.0

**Destination:** `docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_0.md`
**From:** Team Alpha · **Owner:** Michael · **Date:** 2026-11-14 · **Base commit:** `c8e8275`
**Status:** design for Beta's review. No code exists and none is written until the owner has
Beta's review in hand. The code that follows changes WATCHER and the Step 1 launch path, so
it is BEFORE-COMMIT class.

## Prior record

**Searches run** (fresh public clone, `d60a714` to `c8e8275`; VM101 was not read by Alpha):

1. `grep -rlIiE "search[_ ]id|search identity|rerun|re-run|re-sweep|resweep|re-optimi[sz]|retire|per-session|new draws|new dataset version|next dataset version|four[- ]session|progressive sweep|seed_domain_complete|COMPLETE generates|accumulator input|input-manifest digest|study_identity|walk-forward"`
   Files hit: `PROJECT_FILE_CATALOG.md` 1 · `TB_RULING_*` 9 · `TB_RULING_REQUEST_*` 5 ·
   `PROPOSAL_*` 18 · `CHAPTER_*` 28 · `*_AUDIT.md` 10 · `TODO_CONSOLIDATED.md` 1 ·
   `BACKLOG.md` 1 · `LEADS.md` 0 · `exchange/` 6 · `CLAUDE_CODE_REPORT_*` 31 ·
   `CLAUDE_CODE_INSTRUCTIONS_*` 50 · `TEAM_ALPHA_*` 21 · `SESSION_CHANGELOG_*` 62.
   Hit lines were read in the catalog, TODO, BACKLOG, the S176, S177 and S179 rulings, the
   window-anchor ruling, both KPI governance proposals, the delegated-fleet proposal and
   `PIPELINE_BEHAVIOUR_MODEL.md`.
2. `grep -rnI "fingerprint_registry|watcher_registry_hooks|pre_run_check|post_run_record" --include=*.py .`,
   then `docs/CHAPTER_12_AUDIT.md` and `CHAPTER_10_AUDIT.md` for the same terms.
3. `grep -rnI --include=*.py -E "step1_trial_history|get_best_step1_params|downstream_score" .`
4. `grep -rlIiE "warm[-_ ]start" docs/TB_RULING_*` and the S140, S141, S166 and S167 changelogs.
5. Project knowledge search: "window optimizer re-run when new draws arrive, regime shift"
   (8 hits; `combined_agent_framework` §11 read).

**Read in full:** the Stage A report and its brief; `CLAUDE_CODE_REPORT_STEP1_PURPOSE_LINEAGE.md`;
`PROPOSAL_S145_R1_Progressive_Empirical_Sweep.md`; the X-RUN4-RANGE request and Beta's review;
`CLAUDE_CODE_INSTRUCTIONS_SEED_DOMAIN_CURSOR_AMENDMENT.md`; `RELAY_PROTOCOL.md`; `CLAUDE.md`;
`AGENTS.md`; the pitfall record; the 2026-11-13 handoff.
**Read at the cited lines:** every other anchor in this document.

**Documents that already decide or record the subject:**

| # | Document | What it says | Label |
|---|---|---|---|
| 1 | Beta ruling 2026-08-07, as quoted in `docs/CLAUDE_CODE_INSTRUCTIONS_SEED_DOMAIN_CURSOR_AMENDMENT.md` §1–§7 | Domain `[0, 2³²)`. Append-only ledger. Each certified interval binds, at minimum, a list that includes **"search/study identity"**. Cursor = first gap in the certified union. COMPLETE generates no WATCHER run. Optuna and the strategy system are not to be changed under cover of the cursor | ALREADY RECORDED. §2.4 and §3 below CHANGE its §4 (the cursor law's role) |
| 2 | Beta ruling 2026-08-08, `docs/CLAUDE_CODE_INSTRUCTIONS_SEED_DOMAIN_CURSOR_R1.md:7-16` | Approved and closed: first-gap cursor, explicit COMPLETE, append-only design, `dataset_sha256` as provenance and **not** a v1 partition key | ALREADY RECORDED. Not changed for the history record |
| 3 | `docs/PROPOSAL_S145_R1_Progressive_Empirical_Sweep.md:22`, §3.3, §3.4, §7.4 | "Step 1 retired permanently" **rejected**. Four sessions of 2³⁰ seeds, one Optuna study resumed across them. Survivors accumulate, merged by best per-seed score. Step 1 "remains active" until a post-sweep yield test is met | ALREADY RECORDED |
| 4 | `docs/CHAPTER_13_LIVE_FEEDBACK_LOOP.md:205-212`, `:275-283`, `:729-735`, `:1046-1053` | Steps 1, 2, 4 are static: "re-run only on regime shift". Steps 3, 5, 6 re-run in the learning loop. Step 1's parameters are never changed autonomously. Regime shift = window decay, survivor churn, an LLM drift flag, or operator override | ALREADY RECORDED. The owner confirmed it on 2026-11-13 (§0, D3) |
| 5 | `docs/CLAUDE_CODE_REPORT_STEP1_PURPOSE_LINEAGE.md` §4–§5 | The sweep is global per family and never per draw. A regime rerun exists, is human-gated at three points, and enters through the same frontier, so it sweeps the next uncovered block | ALREADY RECORDED (2026-08-08) |
| 6 | `docs/CLAUDE_CODE_INSTRUCTIONS_WATCHER_FOLLOWUP_REPAIRS_20261111.md:57-78` | COMPLETE is a clean stop: no halt file, no escalation, no later step | ALREADY RECORDED. §2.4 keeps the clean stop and changes what it reports |
| 7 | `docs/SESSION_CHANGELOG_20260424_S167_S168_S169.md:19-51` | Beta approved removing the automatic warm-start: `get_best_step1_params()` is not called during Step 1. Fresh runs must not become warm-started runs | ALREADY RECORDED. Not changed (§2.10) |
| 8 | `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:9`, `:46`; `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:13`, `:50` | The seven warm-start keys are routable only when the operator supplies them. `_operator_pin_params` and `_pin_bundle` are privileged seams | ALREADY RECORDED. Not changed |
| 9 | `docs/TFM_PROJECT_FACTS_SKILL.md:632-648` (§2.10b); `docs/BACKLOG.md:104-120` | Re-optimisation is per-session. Scores are not valid on the next dataset version. The accumulator input wall is an exact input-digest match (ruled, not built). Which artifact a per-session run resolves against is open | ALREADY RECORDED. Bears on §2.2 and §9 |
| 10 | `docs/TB_RULING_S176_WATCHER_KPI.md:456`, `:692-725`; `docs/TB_RULING_S179_IMPLEMENTATION_AUTH.md:205-300`, `:513-546` | Reruns become autonomous only after a walk-forward baseline exists. The walk-forward forbids draw t or later from reaching window selection. BOOTSTRAP: every regime trigger is audit-only. A `pipeline_rerun` request must pass a generic gate | ALREADY RECORDED. §2.5 routes through it and lifts nothing |
| 11 | `docs/TB_RULING_WINDOW_ANCHOR_PROPOSAL_V1_0.md:15`, `:42`; fact file §2.3 (`:365`) | The 22-array wall stays closed. New facts go in metadata | ALREADY RECORDED. Honoured (§2.10) |
| 12 | `agents/fingerprint_registry.py:1-24`; `docs/CHAPTER_12_AUDIT.md:74-78` | A Beta-approved (2026-01-02) store of which data context and generator family were tried, for Step 5. WATCHER does not call it | ALREADY RECORDED. Not reused and not replaced (§5) |
| 13 | `docs/CLAUDE_CODE_REPORT_STEP1_RERUN_POLICY_STAGE_A.md` P1–P13, Q1–Q9 | The as-built map and four options. Option B is the base of this design | ALREADY RECORDED (`d47f48c`) |
| 14 | `docs/exchange/from_beta/X-RUN4-RANGE_R1_BETA_REVIEW.md` Q2, Q3 | LIVE objection: completing the sweep would refuse the soak and later Step 1 work. Suggested shape: keep the history ledger, add an explicit search bound to dataset, session, configuration and a search identity. Settle this before Run 4 | ALREADY RECORDED |
| 15 | `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:44-65`, `:77`, `:101` | Finished = F1–F4. The soak follows | ALREADY RECORDED. §4 CHANGES F4's wording and adds F5 |
| 16 | `docs/TEAM_ALPHA_D6_2_IDENTITY_ADDENDUM.md:62-66`; `docs/SESSION_CHANGELOG_20260312_S138.md:69-72` | `step1_{prng_base}_{seed_start}` is not unique across reruns. The resumed-study trial count is an open item | ALREADY RECORDED |

**Labels for what this proposal raises:**

| Item | Label |
|---|---|
| S1 The search record and the chunk link (§2.2, §2.3) | NEW as objects. They fill the "search/study identity" binding that record 1 already requires |
| S2 WATCHER schedules from the search's own cursor (§2.4) | CHANGES A RULING: record 1 §4 and record 6 |
| S3 A search is opened only explicitly (§2.5) | NEW. Consistent with records 4 and 10 |
| S4 Step 1 is due when its search has a gap (§2.6) | NEW (Stage A P8) |
| S5 Trial budget per chunk; no credit for a chunk with no completed trial (§2.7) | NEW rule on a recorded open item (record 16; Stage A P3, P4) |
| S6 Trial rows keyed by search (§2.8) | NEW. The fault is recorded (Stage A P1, P2, P5, P6) |
| S7 Attempt 9 becomes search 0; Run 4 continues it (§2.9) | NEW |
| S8 Keys for Optuna and the ML; no wiring (§2.10) | NEW. Changes nothing in records 7, 8 or 11 |
| S9 F4 reworded, F5 added (§4) | CHANGES A RULING: record 15 (owner decision D2) |

**Not searched:** VM101; the miner ledgers and Optuna study databases, so attempt 9's recorded
settings were not read; ser8; the private repository. `CHAPTER_1_WINDOW_OPTIMIZER.md` was not
read in full: its statements here come through the lineage report's quotations.

---

## 0. The owner's decisions this design carries

Requirements R1–R5 are in the Stage A brief (`87b3182`): pick up where a search stopped;
cover the space in chunks; track **what** was run; be able to run again; automated.

Decisions of 2026-11-13 and 2026-11-14, in the owner's words:

- **D1 — reach.** Asked whether the record would be within reach of Optuna and the ML:
  "i believe so- or at least coded so that we can add that functionality later once we
  arrive at those steps- maybe bounce it off beta?"
- **D2 — finished.** The plan's wording exists "as we spent months on various other methods
  which failed", and "it must me able to run the seed space when required-i.e new draws- or
  the code request or requires i t" (quoted as typed).
- **D3 — when Step 1 runs again.** "1 as chapter 13- with all the requisites that a rerun
  and the learning ml will require- ie- survivors- windows- sessions etc etc". Option 1 was:
  new draws go to Steps 3, 5 and 6; Step 1 must be able to run again and does so when a
  regime shift is called, by the code or by him.
- **D4 — order.** "Fix first", then Run 4.

## 1. The problem, as read

1. The certified record holds one interval: `java_lcg`, both modes, `[0, 2³¹)`, attempt 9.
   WATCHER overwrites any start with the first gap (`agents/watcher_agent.py:2087-2089`).
2. When the union reaches `[0, 2³²)`, WATCHER builds no Step 1 command for that family and
   mode set (`:2068-2085`), whatever dataset, session or window is asked for. The cursor
   tests family and modes only (`utils/seed_coverage_ledger.py:634-640`, `:1029-1055`).
3. The record does not say what was run. The certified row's study identity is NULL
   (Stage A P5). Session is NULL in all 56 trial rows (P6). Two searches share one run id,
   and the second one's trials are dropped (P1, P2; `database_system.py:462-483`).
4. A continued study runs `max_iterations − completed` trials
   (`window_optimizer_bayesian.py:806`, `:827`). Under the four-session plan as written that
   is zero trials from chunk 2 on, and nothing refuses to certify that chunk (P4; read, not
   executed).
5. A regime rerun with no newer input is skipped by the freshness check before the cursor
   is read (`agents/watcher_agent.py:1835-1858`; P8; read, not executed).
6. The hit-rate write-back lands on every search that shares a seed start
   (`database_system.py:484-499`).

None of this is a statement about predictive value. Attempt 9 published 0 rows at one window
setting; that is an observation at one geometry. The yardstick is unchanged: pool hits
against `k/1000` on draws the pipeline never saw.

## 2. The design

### 2.1 Two records with two meanings

- **History** (`certified_coverage`, unchanged): which seed intervals have been traversed
  and published for a family and mode set. Its union and its COMPLETE stay reportable. It
  is no longer what decides where Step 1 starts.
- **Search** (new): one named piece of Step 1 work, with what it ran on and how far it
  has got.

Beta's sentence from X-RUN4-RANGE Q2 is adopted as the meaning of a history row: it
certifies traversal under that run's inputs. It does not certify that every window was
tried, that its scores hold on another dataset, or that later optimisation is unnecessary.

### 2.2 The search record

Append-only, in the same database, in a new module. `utils/seed_coverage_ledger.py`, its
table and its suite are not edited.

| Field | Content |
|---|---|
| `search_id` | Content hash of the fields below |
| `prng_base`, `required_modes` | As the history record's identity |
| `dataset_version_id`, `dataset_sha256` | From the frozen dataset (`miner/dataset_authority.py:229-254`) |
| `sessions_declared` | The session options the search may sample |
| `search_space_digest`, `search_space` | The declared bounds, or the pins for a pinned run; sampler; pruning |
| `study_name` | The Optuna study this search uses |
| `trials_per_chunk`, `chunk_seeds` | The budget for each chunk and the chunk size |
| `opened_reason` | `operator`, `regime_shift`, `soak`, `history_import` |
| `opened_by_ref` | The approval request id or the operator invocation |
| `repeat_of` | An earlier `search_id`, or NULL |
| `opened_utc`, `repository_commit` | Provenance |

### 2.3 The chunk link

One append-only row per published chunk: `coverage_id`, `search_id`, `completed_trials`,
`generation_id`. It is written in the same publication hook as the history row
(`window_optimizer_integration_final.py:3056-3060`), after publication, never before.

**Credit rule.** A chunk counts toward its search only if it was published **and**
`completed_trials ≥ 1`. A chunk with no completed trial still gets its history row, because
the seeds were traversed. It does not advance the search.

### 2.4 Which search WATCHER runs

- Each family and mode set has at most one **open** search. WATCHER continues it.
- Step 1's start is the first gap among that search's credited chunks. The same pure
  function is used (`first_uncovered_seed`, `utils/seed_coverage_ledger.py:551-605`), over
  a different set of intervals. The pre-dispatch domain wall is unchanged.
- Before dispatch WATCHER checks that the frozen dataset and the search-space digest equal
  the search record's. A mismatch refuses, with both values named. **A search finishes on
  the dataset version it was opened on.**
- When the search's own union is complete, or no search is open, Step 1 stops cleanly, as
  F-1 requires today. The message names the search and says how one is opened.
- A lookup failure still blocks Step 1. Nothing falls back to seed 0.

### 2.5 How a search is opened

Only explicitly. Three routes, and none of them is new authority:

| Route | Authority | State today |
|---|---|---|
| Operator, through the certified CLI seam (`--run-pipeline --params`) | The Route-A seam, unchanged | Usable |
| An approved Chapter 13 full-pipeline request (regime shift) | `agents/watcher_agent.py:3845-3883`, after the S179 gate | Human approval today. Audit-only under BOOTSTRAP until S179's gates are built |
| History import (§2.9) | This proposal, once | One-time |

A new dataset version does **not** open a search (D3). New draws go to Steps 3, 5 and 6.
Opening a search closes the previous open one for that family and mode set; its rows are
kept.

### 2.6 When Step 1 is due

For Step 1 only, "due" is decided by the search: it is due when the selected search has a
gap. File times no longer skip it. Every other step keeps the freshness check.

### 2.7 Trial budget

Each chunk of a search runs `trials_per_chunk` new trials, counted from the study's
completed total at the start of that chunk. A resume inside one chunk keeps the present
rule (the remainder of that chunk's budget).

### 2.8 The trial record

A new append-only table of Step 1 trials, keyed `(search_id, coverage chunk, trial ordinal)`,
holding the window, position, skip range, **session**, thresholds, score and survivor counts.
The study name is written on the `n_parallel == 1` path. Chapter 13's hit-rate write-back
targets a `search_id`. `step1_trial_history` is kept as it is, as telemetry with no
authority, the same disposition the 2026-08-07 ruling gave the old seed tracker.

### 2.9 The existing history

- Attempt 9's certified row is linked to **search 0**, opened as `history_import`. Its
  settings are read from the archived miner ledger on VM101 at implementation, under the
  binding addendum §A. They are not taken from memory or from this document.
- Run 4 carries attempt 9's pins on the same dataset. If the import shows the same dataset
  and the same pins, Run 4 is chunk 2 of search 0 and completes both search 0 and the
  history union. If they differ, Run 4 opens its own search and the difference is reported
  to the owner before launch.

### 2.10 Reach for Optuna and the ML (D1)

Keys are put in place. Nothing is wired.

- **Optuna.** Within a search the study is the memory, as now. Across searches nothing is
  carried automatically: records 7 and 8 stand. What is new is that every trial row and
  every hit-rate write-back carries a `search_id`, and a search can name the one it
  repeats. A later feature that starts a new search from an earlier one's results would
  CHANGE records 7 and 8 and needs its own ruling.
- **ML.** Which search a surviving row came from is derivable without touching the 22
  arrays or the finalizer: the link row gives `generation_id → search_id`, and a row that
  differs from its parent generation's row, or is absent there, came from that generation's
  run (strict-greater merge, `utils/run_finalizer.py:786-806`, as read by Stage A Q5). No
  feature is added now.

## 3. What changes in the rulings

| Ruling | Changes | Does not change |
|---|---|---|
| Beta 2026-08-07 §4 (cursor law) | WATCHER's start comes from the open search's first gap, not the history union's | First-gap arithmetic; the domain; append-only; publication before credit; fail-closed lookup; no reset or deletion |
| Beta 2026-08-07 §3 | The "search/study identity" binding becomes a real record | The history row and its `coverage_id` |
| Beta 2026-08-08 R1 | Nothing | Dataset stays provenance on the history record. It is an identity field of a different object |
| F-1 (2026-11-11) | The clean stop reports the search's state | No halt file, no escalation, no later step |
| Completion plan F4 | See §4 | F1–F3 |
| S167; Route-A | Nothing | No automatic warm-start; the privileged seams |

## 4. Finished (D2)

- **F4, reworded:** the history record reports `java_lcg`, modes `{constant, variable}`,
  COMPLETE over `[0, 2³²)`, with no host freeze during the sweep. This is the proof that
  the method can sweep the whole space. It is not a claim that every window was tried.
- **F5, new:** with the history COMPLETE, a newly opened search yields a Step 1 command,
  and no existing row is changed or deleted. Shown first with dispatch intercepted (§7,
  M2), then for real by the soak.

## 5. What this proposal does not do

- **The dataset input wall and cross-dataset survivor merge** (record 9; Stage A P7). A
  search on a new dataset version would today merge onto the old generation by best score.
  That needs the owner's decision before such a search first publishes. Run 4 is not
  affected: same dataset.
- **The next-chunk loop, launch and approval gates** (Stage A Q8 rows 3, 6, 11). R5 for
  those depends on the S179 and delegated-run gates, which are separate tracks.
- **Walk-forward.** It becomes a client of this record. It is not designed here.
- **The fingerprint registry.** Left as it is. Whether a search's identity should later
  feed it is not decided here.
- **Which session options a certifying search may declare.** BACKLOG §5 governs.

## 6. Files and suites

Edits expected: a new search-ledger module; `database_system.py` (wrappers, the new trial
table); `agents/watcher_agent.py` (`:2027-2110`, and the Step 1 freshness branch);
`window_optimizer_integration_final.py` (`:3056-3060`, the study name);
`window_optimizer_bayesian.py` (`:806`); `window_optimizer.py` and
`agent_manifests/window_optimizer.json` (the search id as a declared parameter on all three
hops, fact file §2.15). Not edited: `utils/seed_coverage_ledger.py`, `utils/run_finalizer.py`,
`miner/`, the kernels, the 22-array contract.

Suites, as Stage A Q9 lists them: the seed-domain suite must stay green **unedited**; the
Route-A routing evidence needs the scoped refresh Beta asked for; Gate 22 is red while the
edits are uncommitted; the rig parity set is not touched.

## 7. Measurements that settle it

CPU-only, dispatch intercepted, throwaway database. No fleet.

| # | Measurement | Settles |
|---|---|---|
| M1 | Before any edit: continue a study whose completed count equals the budget; record trials run, whether publication is reached, whether coverage is written | P4 as a measured fact |
| M2 | History COMPLETE, then open a search: a Step 1 command is built from seed 0; no existing row changes | F5 |
| M3 | A chunk with zero completed trials: history row written, search not advanced | §2.3 |
| M4 | A search opened on dataset A, current dataset B: refused before dispatch, both named | §2.4 |
| M5 | A regime request with unchanged input files and an open gap: Step 1 is not skipped | §2.6, P8 |
| M6 | Two searches from the same seed start: both sets of trial rows present, write-back lands on one | §2.8 |
| M7 | The seed-domain suite, unedited, on the changed tree | The history record is untouched |

## 8. Order

1. Beta reviews this design. The owner rules on what Beta raises.
2. One brief for one Claude Code session on Opus, with M1 run first.
3. Beta reviews the code before commit.
4. The Route-A evidence refresh, then Run 4, then Brief II and the soak.

## 9. For the owner, after Beta

The cross-dataset merge in §5 is the one decision this design leaves open that will come due.

## 10. What Alpha did not verify

- Attempt 9's settings, study and trial count: in the miner ledger, not opened.
- How WATCHER can make Step 1 freeze a **named** dataset version. `dataset_authority`
  supports it (`miner/dataset_authority.py:558`); no route was found by grep in
  `window_optimizer.py` or `agents/watcher_agent.py`.
- Facts 4 and 5 in §1 by execution.
- The finalizer's merge lines, the Gate 22 list and the suite arm names: taken from the
  Stage A report.
- Whether the S179 gates exist on VM101. The Chapter 13 chapter says they did not land.
