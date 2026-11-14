# PROPOSAL — STEP 1 SEARCH RECORD AND RERUN — v1.1

**Destination:** `docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md`
**From:** Team Alpha · **Owner:** Michael · **Date:** 2026-11-14 · **Base commit:** `b2d2822`
**Supersedes:** v1.0 (`eac05ec`), after Beta's review `X-STEP1-SEARCH_R1_BETA_REVIEW.md` (`b2d2822`).
**Status:** design, accepted by the owner for implementation on 2026-11-14. Owner ruling of
that date: there is no second design round with Beta; Beta reviews the code instead. The
code changes WATCHER and the Step 1 launch path, so it is BEFORE-COMMIT class.

## Prior record

**Searches run** (fresh public clone, `d60a714` to `b2d2822`; VM101 was not read by Alpha):

1. `grep -rlIiE "search[_ ]id|search identity|rerun|re-run|re-sweep|resweep|re-optimi[sz]|retire|per-session|new draws|new dataset version|next dataset version|four[- ]session|progressive sweep|seed_domain_complete|COMPLETE generates|accumulator input|input-manifest digest|study_identity|walk-forward"`
   Files hit: `PROJECT_FILE_CATALOG.md` 1 · `TB_RULING_*` 9 · `TB_RULING_REQUEST_*` 5 ·
   `PROPOSAL_*` 18 · `CHAPTER_*` 28 · `*_AUDIT.md` 10 · `TODO_CONSOLIDATED.md` 1 ·
   `BACKLOG.md` 1 · `LEADS.md` 0 · `exchange/` 6 · `CLAUDE_CODE_REPORT_*` 31 ·
   `CLAUDE_CODE_INSTRUCTIONS_*` 50 · `TEAM_ALPHA_*` 21 · `SESSION_CHANGELOG_*` 62.
   Hit lines were read in the catalog, TODO, BACKLOG, the S176, S177 and S179 rulings, the
   window-anchor ruling, both KPI governance proposals, the delegated-fleet proposal and
   `PIPELINE_BEHAVIOUR_MODEL.md`.
2. `grep -rnI "fingerprint_registry|watcher_registry_hooks|pre_run_check|post_run_record" --include=*.py .`,
   then the Chapter 10 and 12 audits.
3. `grep -rnI --include=*.py -E "step1_trial_history|get_best_step1_params|downstream_score" .`
4. `grep -rlIiE "warm[-_ ]start" docs/TB_RULING_*` and the S140, S141, S166 and S167 changelogs.
5. Project knowledge search: "window optimizer re-run when new draws arrive, regime shift".
6. **For v1.1:** full read of Beta's R1 review; read of every record its Finding 1 names, at
   the lines it gives; `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:140-182`;
   `docs/CLAUDE_CODE_REPORT_RANGE_MINER_VM101_READS.md:195-292`, which v1.0 had read only
   to `:90`; and the source lines Beta cites for the dataset route, the zero-trial path, the
   CLI seam and the suite's COMPLETE arm.

**Read in full:** the Stage A report and its brief; `CLAUDE_CODE_REPORT_STEP1_PURPOSE_LINEAGE.md`;
`PROPOSAL_S145_R1_Progressive_Empirical_Sweep.md`; the X-RUN4-RANGE request and review;
`CLAUDE_CODE_INSTRUCTIONS_SEED_DOMAIN_CURSOR_AMENDMENT.md`; Beta's X-STEP1-SEARCH R1 review;
`RELAY_PROTOCOL.md`; `CLAUDE.md`; `AGENTS.md`; the pitfall record; the 2026-11-13 handoff.
**Read at the cited lines:** every other anchor in this document.

**Documents that already decide or record the subject:**

| # | Document | What it says | Label |
|---|---|---|---|
| 1 | Beta ruling 2026-08-07, as quoted in `docs/CLAUDE_CODE_INSTRUCTIONS_SEED_DOMAIN_CURSOR_AMENDMENT.md` §1–§7 | Domain `[0, 2³²)`. Append-only ledger. Each certified interval binds a list that includes "search/study identity". Cursor = first gap in the certified union. COMPLETE generates no WATCHER run. A range is certified only after canonical publication | ALREADY RECORDED. §2.4 and §3 CHANGE its §4 (the cursor law's role) |
| 2 | Beta ruling 2026-08-08, `docs/CLAUDE_CODE_INSTRUCTIONS_SEED_DOMAIN_CURSOR_R1.md:7-16`; `docs/CLAUDE_CODE_INSTRUCTIONS_SEED_DOMAIN_CURSOR_R2.md:145-155` | `dataset_sha256` is provenance, not a v1 partition key. WATCHER overwrites the start only when the cursor is above 0 (a recorded certification boundary) | ALREADY RECORDED. The first is kept. The second is closed for the search cursor (§2.4) |
| 3 | `docs/PROPOSAL_S145_R1_Progressive_Empirical_Sweep.md:22`, §3.3, §3.4, §7.4 | "Step 1 retired permanently" rejected. Four sessions, one study resumed across them. Survivors accumulate by best per-seed score | ALREADY RECORDED |
| 4 | `docs/CHAPTER_13_LIVE_FEEDBACK_LOOP.md:205-212`, `:275-283`, `:729-735`, `:1046-1053` | Step 1 re-runs only on regime shift. New draws re-run Steps 3, 5, 6 | ALREADY RECORDED. Confirmed by the owner (§0, D3) |
| 5 | `docs/CLAUDE_CODE_REPORT_STEP1_PURPOSE_LINEAGE.md` §4–§5 | The sweep is global per family. The regime rerun is human-gated and enters through the same frontier | ALREADY RECORDED |
| 6 | `docs/CLAUDE_CODE_INSTRUCTIONS_WATCHER_FOLLOWUP_REPAIRS_20261111.md:57-78` | COMPLETE is a clean stop | ALREADY RECORDED. Kept; what it reports changes |
| 7 | `docs/SESSION_CHANGELOG_20260424_S167_S168_S169.md:19-51` | No automatic warm-start | ALREADY RECORDED. Not changed |
| 8 | `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:9`, `:46`; `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:13`, `:48-50` | The seven warm-start keys are routable only when the operator supplies them. `_operator_pin_params` and `_pin_bundle` are privileged seams; a new production caller needs governance review | ALREADY RECORDED. Not changed (§2.5) |
| 9 | `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:154-179` | Attempt 9 ran a fused pre-advance of 25. Run 4 runs `generator_phase = 0`. Their populations are not regression-equivalent. Run 4 certifies operation, not population identity | ALREADY RECORDED. **Missed by v1.0.** It decides §2.9 |
| 10 | `docs/CLAUDE_CODE_REPORT_RANGE_MINER_VM101_READS.md` R1–R3 (`:94-292`) | Attempt 9's certified row, generation, sidecar and miner trial context, read on VM101 on 2026-11-13: window 12, `offset_val` 25, sessions `["midday"]`, skip 6–99, thresholds 0.71 / 0.47, dataset `513648160d35…`, four phases of 32 stripes done over `[0, 2³¹)`, trial committed. The sidecar does not name the miner run id | ALREADY RECORDED (`2930a37`). **Missed by v1.0**, which said these settings had not been read |
| 11 | `docs/TFM_PROJECT_FACTS_SKILL.md:632-648`; `docs/DATASET_PUBLICATION_SCHEMA_v1.md:107-115`; `docs/BACKLOG.md:104-120` | Per-session re-optimisation. Scores are not valid on the next dataset version. The accumulator input wall is an exact input-manifest digest match (ruled, not built). The session-scoped authority is open | ALREADY RECORDED. Bears on §2.2, §2.6 |
| 12 | `docs/TB_RULING_S176_WATCHER_KPI.md:456`, `:692-727`; `docs/TB_RULING_S179_IMPLEMENTATION_AUTH.md:225-300`, `:513-546` | Reruns become autonomous only after a walk-forward baseline. A rerun request needs the consumer gate **and** the execution-chokepoint recheck. BOOTSTRAP: regime triggers audit-only | ALREADY RECORDED. §2.5 lifts nothing |
| 13 | `docs/CLAUDE_CODE_INSTRUCTIONS_S172_D6_2_CHECKPOINT_RECONCILIATION.md:167-208`; `docs/CLAUDE_CODE_INSTRUCTIONS_S172_D6_2_BOUNDED_REPAIR.md:72-105`, `:128-146`; `docs/TEAM_ALPHA_D6_2_IDENTITY_ADDENDUM.md:62-66` | Resume inside a run: an explicit checkpoint run id, a context digest, proof that an existing study was loaded, record ordinals kept apart from Optuna trial numbers, certified for `n_parallel == 1` only. The Step 1 run id is not unique across reruns | ALREADY RECORDED. **The first two were missed by v1.0** (Beta Finding 1). Govern §2.7 |
| 14 | `docs/CHAPTER_1_WINDOW_OPTIMIZER.md:810-828`; `docs/CLAUDE_CODE_INSTRUCTIONS_S172_THRESHOLD_REPAIR.md:49-51`, `:102-130` | One threshold resolver. Requested, payload and effective values are recorded separately and read at the executor. A historical study is not classed as executed without evidence | ALREADY RECORDED. **Missed by v1.0** (Beta Finding 1). Govern §2.8, §2.9 |
| 15 | `docs/TB_RULING_WINDOW_ANCHOR_PROPOSAL_V1_0.md:15`, `:42`; fact file §2.3 | The 22-array wall stays closed | ALREADY RECORDED. Honoured |
| 16 | `agents/fingerprint_registry.py:1-24`; `docs/CHAPTER_12_AUDIT.md:74-78` | A Beta-approved store of tried data contexts for Step 5. Not wired | ALREADY RECORDED. Not reused, not replaced |
| 17 | `docs/CLAUDE_CODE_REPORT_STEP1_RERUN_POLICY_STAGE_A.md` P1–P13, Q1–Q9; `docs/SESSION_CHANGELOG_20260312_S138.md:69-72` | The as-built map and four options. The resumed-study trial count is an open item | ALREADY RECORDED |
| 18 | `docs/exchange/from_beta/X-RUN4-RANGE_R1_BETA_REVIEW.md` Q2, Q3; `docs/exchange/from_beta/X-STEP1-SEARCH_R1_BETA_REVIEW.md` | The LIVE objection this design answers. Beta's R1 positions on v1.0 | ALREADY RECORDED |
| 19 | `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:44-65`, `:77`, `:96-101` | Finished = F1–F4. Brief II and the soak come after finished | ALREADY RECORDED. §4 CHANGES F4's wording and adds F5a |

**Labels for what this proposal raises:**

| Item | Label |
|---|---|
| S1 The search record, its lifecycle events and the chunk link (§2.2, §2.3) | NEW. They fill record 1's "search/study identity" binding |
| S2 WATCHER schedules from the search's own cursor, including when it is 0 (§2.4) | CHANGES A RULING: record 1 §4, record 6, and the boundary in record 2 |
| S3 A search is opened only explicitly; stored settings carry no authority (§2.5) | NEW. Consistent with records 4, 8 and 12 |
| S4 Step 1 is due when its search has a gap (§2.6) | NEW (Stage A P8) |
| S5 Credit needs evidence of a completed sweep; budget per chunk (§2.3, §2.7) | NEW rule on a recorded open item. **Changed from v1.0 on Beta's LIVE objection** |
| S6 Trial rows and evaluations keyed by search (§2.8) | NEW. The fault is recorded (Stage A P1, P2, P5, P6) |
| S7 Attempt 9 imported as a search from the recorded evidence; Run 4 is its own search (§2.9) | NEW. **Changed from v1.0** on record 9 |
| S8 Keys for Optuna and the ML; no wiring (§2.10) | NEW. Changes nothing in records 7, 8 or 15 |
| S9 A search on a different dataset is refused until the owner decides the merge (§2.6) | NEW restriction on a recorded open item (record 11) |
| S10 F4 reworded; F5a added; F5b after finished (§4) | CHANGES A RULING: record 19 (owner decisions D2 and D5) |

**Not searched:** VM101; the miner ledgers and Optuna study databases directly; ser8; the
private repository. `CHAPTER_1_WINDOW_OPTIMIZER.md` was read at `:810-828` only.

---

## 0. The owner's decisions this design carries

Requirements R1–R5 are in the Stage A brief (`87b3182`): pick up where a search stopped;
cover the space in chunks; track **what** was run; be able to run again; automated.

- **D1 — reach.** On whether the record is within reach of Optuna and the ML: "i believe
  so- or at least coded so that we can add that functionality later once we arrive at those
  steps- maybe bounce it off beta?"
- **D2 — finished.** The plan's wording exists "as we spent months on various other methods
  which failed", and "it must me able to run the seed space when required-i.e new draws- or
  the code request or requires i t" (quoted as typed).
- **D3 — when Step 1 runs again.** "1 as chapter 13- with all the requisites that a rerun
  and the learning ml will require- ie- survivors- windows- sessions etc etc". Option 1 was:
  new draws go to Steps 3, 5 and 6; Step 1 must be able to run again and does so when a
  regime shift is called, by the code or by him.
- **D4 — order.** "Fix first", then Run 4.
- **D5 — the split of the fifth condition** (2026-11-14, on Beta's Q10): "agreed". The
  intercepted test belongs to finished; the soak's confirmation stays after finished.

## 0.1 What changed from v1.0

| Beta R1 | Position | Disposition in v1.1 |
|---|---|---|
| Q1, Q2, Q3, Q5, Q8 | NO OBJECTION | Kept. Beta's implementation obligations under Q2, Q5 and Q8 are written into §2.3–§2.5 and §2.10 |
| Q4 | OBJECTION · LIVE | **Accepted. v1.0 was wrong.** The sieve runs inside a trial, so no completed trial means no seeds were swept. §2.3 now gives no history credit and no search credit without evidence of a completed sweep |
| Q6 | SUGGESTION | Accepted: the route to a named dataset version, the session record, and an enforced refusal for a different dataset (§2.2, §2.4, §2.6) |
| Q7 | SUGGESTION | Accepted: a search owns its study; evaluations are separate records (§2.7, §2.8) |
| Q9 | NEEDS EVIDENCE | Answered from record 10, which v1.0 and the review both missed. Unknowns are stated as unknown (§2.9) |
| Q10 | SUGGESTION | Accepted, and agreed by the owner (D5; §4) |
| Q11 | SUGGESTION | Accepted: Beta's seven evidence groups become the code brief's gate list (§7) |
| Q12 / Finding 1 | SUGGESTION | Accepted: records 13 and 14 added |

## 1. The problem, as read

1. The certified record holds one interval: `java_lcg`, both modes, `[0, 2³¹)`, attempt 9.
   WATCHER overwrites a supplied start with the first gap when that gap is above 0
   (`agents/watcher_agent.py:2087-2089`).
2. When the union reaches `[0, 2³²)`, WATCHER builds no Step 1 command for that family and
   mode set (`:2068-2085`), whatever dataset, session or window is asked for.
3. The record does not say what was run. The certified row's study identity is NULL
   (Stage A P5). Session is NULL in all 56 trial rows (P6). Two searches share one run id
   and the second one's trials are dropped (P1, P2; `database_system.py:462-483`).
4. A continued study runs `max_iterations − completed` trials
   (`window_optimizer_bayesian.py:806`, `:827`). With no trial run, the no-result return is
   taken (`:855-869`), and the integration layer still hands the configured interval to the
   finalizer (`window_optimizer_integration_final.py:2923`, `:3001-3011`), after which the
   history row is written (`:3056-3060`). Read, not executed. M1 settles it.
5. A regime rerun with no newer input is skipped by the freshness check before the cursor
   is read (`agents/watcher_agent.py:1835-1858`; P8; read, not executed).
6. The hit-rate write-back lands on every search that shares a seed start
   (`database_system.py:484-499`).

None of this is a statement about predictive value. Attempt 9 completed its sweep and
published 0 rows at one window setting; that is an observation at one geometry. The
yardstick is unchanged: pool hits against `k/1000` on draws the pipeline never saw.

## 2. The design

### 2.1 Two records with two meanings

- **History** (`certified_coverage`, unchanged): which seed intervals have been swept and
  published for a family and mode set. Its union and its COMPLETE stay reportable. It no
  longer decides where Step 1 starts.
- **Search** (new): one named piece of Step 1 work, with what it ran on and how far it
  has got.

A history row certifies a sweep under that run's inputs. It does not certify that every
window was tried, that its scores hold on another dataset, or that later optimisation is
unnecessary (Beta, X-RUN4-RANGE Q2).

### 2.2 The search record

Append-only, in the same database, in a new module with its own schema version and
validation. `utils/seed_coverage_ledger.py`, its table and its ledger assertions are not
edited. The new tables are not a second certification door.

| Field | Content |
|---|---|
| `search_id` | Hash of a versioned, canonical encoding of the fields below. A retry of the same opening gives the same id; an explicit repeat gives a new one |
| `prng_base`, `required_modes` | As the history record's identity |
| `range_start`, `range_end_exclusive` | The seed range this search is to cover. Default `[0, 2³²)` |
| `dataset_version_id`, `dataset_sha256`, `input_manifest_sha256`, `dataset_lineage_id` | From the frozen dataset (`miner/dataset_authority.py:229-254`) |
| `session_options`, `session_filter_rule` | The session sets the search may sample, **as labels**, with the index-to-set mapping and the ordering rule. An index alone is not recorded as the session |
| `search_space`, `search_space_digest` | The declared bounds, or the pins for a pinned run, in effective terms (window, anchor, generator phase, skip range, thresholds); sampler; pruning |
| `study_name` | The Optuna study this search owns (§2.7) |
| `trials_per_chunk`, `chunk_seeds` | Budget per chunk and chunk size |
| `opened_reason`, `opened_by_ref` | `operator`, `regime_shift`, `soak` or `history_import`; the invocation or request id |
| `repeat_of` | An earlier `search_id`, or NULL. Descriptive only. It grants nothing |
| `opened_utc`, `repository_commit` | Provenance |

**Lifecycle.** Opening, superseding and completing a search are append-only events. At
most one search is open per family and mode set, enforced in one transaction. A missing or
conflicting selector refuses. It never becomes a new search at seed 0.

### 2.3 The chunk, its link, and credit

**Chunk opening.** Before a chunk's first trial, an append-only chunk-open event records the
chunk's identity, its seed interval, the study's trial count at that moment (the budget
baseline), the budget target, and the checkpoint run id. A restart reads this record. It
does not recompute a baseline.

**Credit rule (changed from v1.0).** A chunk is credited only on evidence that this run
completed a sweep of that interval under every required mode: at least one trial in state
COMPLETE, belonging to this chunk, whose sieve finished all required phases over the whole
interval.

- **No such evidence:** nothing is published as certified. There is no history row and no
  search credit. The attempt and its reason are kept in a non-certifying audit record.
- **A completed sweep with zero survivors** is credited. Credit needs execution evidence,
  not a survivor count.
- **A recovered chunk** is credited only if the completed work is shown to belong to that
  chunk. A study-wide completed count is never that evidence.

**The link.** One append-only row per credited chunk: `coverage_id`, `search_id`, chunk id,
trial counts by state, `generation_id`. The interval, family, modes and generation are
derived from the certified publication witness and checked against the selected search
(`utils/seed_coverage_ledger.py:862-907`). The caller cannot assert them. The write is
idempotent. If it fails after publication, the failure surfaces, the search is not
advanced, and the retry uses the same witness.

### 2.4 Which search WATCHER runs

- WATCHER continues the one open search for the family and mode set.
- Step 1's start is the first gap among that search's credited chunks inside its declared
  range, **assigned even when it is 0**. The same pure function is used
  (`first_uncovered_seed`, `utils/seed_coverage_ledger.py:551-605`). The pre-dispatch
  domain wall is unchanged.
- **Dataset.** A search finishes on the dataset version it was opened on. WATCHER, the
  Step 1 child, the checkpoint identity and the publication hook must all resolve that
  named version (`agents/watcher_agent.py:748-763`, `:1930-1936`;
  `window_optimizer.py:1822-1839`; `miner/dataset_authority.py:548-563`). That complete
  route is not shown yet and is a gate of the code brief. Until it is shown, a search
  whose dataset is no longer the current pointer **refuses**, with both versions named.
- **Configuration.** The session mapping, search-space digest, pins, sampler, pruning and
  budget presented for a run must equal the search record's. A mismatch refuses.
- The result distinguishes three states durably: no open search, search complete, lookup
  failure. The first two are clean stops as F-1 requires. A lookup failure blocks Step 1.
  Nothing falls back to seed 0.

### 2.5 How a search is opened, and what carries authority

Only explicitly:

| Route | Authority | State today |
|---|---|---|
| Operator, through the existing CLI seam (`agents/watcher_agent.py:3919-3932`) | That seam, unchanged. It needs the scoped routing review and evidence refresh already planned, not a new permission | Usable |
| An approved Chapter 13 full-pipeline request (regime shift) | Only after **both** S179 gates exist: the request-consumer gate and the execution-chokepoint recheck. A request marked approved is not a substitute | Not enabled by this design. Audit-only under BOOTSTRAP |
| History import (§2.9) | This proposal, once | One-time |

**Stored settings are never authority.** For a pinned search the operator supplies the pins
on every run, as now. WATCHER compares them with the record. Nothing reconstructs pins from
a stored search and places them in `_operator_pin_params` or `_pin_bundle`.

A new dataset version does not open a search (D3). An open gap makes authorised work due.
It does not authorise a launch.

### 2.6 When Step 1 is due, and when it is refused

- For Step 1 only, "due" is decided by the search: it is due when the selected search has a
  gap. File times no longer skip it. Other steps keep the freshness check.
- **Different dataset.** A search whose dataset differs from the one behind the current
  accumulated generation is refused before dispatch. That holds until the accumulator input
  wall (record 11) is built or the owner decides otherwise. A new `search_id` does not
  license merging unlike inputs.

### 2.7 Study ownership, budget and resume

- **Ownership.** A search owns one study. A new search gets a distinct study. A study that
  already belongs to another search, or a name that does not match the record, refuses.
- **Budget.** Each chunk attempts `trials_per_chunk` trials, counted from the baseline in
  its chunk-open event. COMPLETE, PRUNED and FAIL each consume one. The counts by state are
  recorded. A chunk that used its budget without a COMPLETE trial is closed as uncredited.
- **Resume inside a chunk** follows the D6.2 rulings unchanged (record 13): explicit
  checkpoint run id, context digest, proof that the existing study was loaded, record
  ordinals separate from Optuna numbers, `n_parallel == 1`.

### 2.8 Trial and evaluation records

- **Trials.** A new append-only table keyed `(search_id, chunk, record ordinal)`, holding
  the Optuna trial number separately, the session **label**, window, anchor, generator
  phase, skip range, thresholds as requested and as effective at the executor (record 14),
  state, score and survivor counts. The study name is written on the `n_parallel == 1` path.
- **Evaluations.** A hit-rate result is a separate append-only record: `search_id`, the
  generation or pool evaluated, the target draws and the information cutoff. Trial rows are
  not updated. A search-level hit rate is an aggregate observation. It is not attribution
  to a trial.
- `step1_trial_history` is kept as it is, as telemetry with no authority.

### 2.9 The existing history

**Attempt 9.** Its certified row is linked to a search by the ordinary import path, with the
ordinary validation. "Search 0" is a display label. The evidence is already on record
(record 10) and is re-read on VM101 at implementation under the binding addendum §A.

| Imported fact | Value on record | Status |
|---|---|---|
| Coverage row, artifact, generation, commit | `c6f28aedf7af12cd…`, `aecc5d50…`, `gen-20260818T013219550933Z-step1_java_lcg_0`, `e9ca800` | Observed |
| Interval and modes | `[0, 2³¹)`, `{constant, variable}`; four phases, 32 stripes each, done; trial committed | Observed |
| Dataset | `daily3-20260801T145551443433Z-513648160d35.json`, sha256 `513648160d35…68f6`, lineage `daily3-combined-L001` | Observed |
| Session | `["midday"]` | Observed |
| Window, skip, thresholds | 12; 6–99; 0.71 / 0.47 | Observed |
| Pre-advance | `offset_val` 25, the fused pre-advance of that date (record 9) | Observed. Recorded as it ran, not as `generator_phase 0` |
| Link from the generation to the miner run | By matching commit, interval, modes and time. The sidecar names no miner run id | Derived |
| Optuna study name; trial budget | Not read | Unknown, recorded as unknown |

If the re-read cannot support a credited import, attempt 9 stays history only, and the
consequence is reported before any launch. Nothing is invented to obtain a start.

**Run 4.** It is not a continuation of attempt 9's search: its effective configuration
differs (record 9). Run 4 opens its own pinned search, by the operator, with declared range
`[2³¹, 2³²)`. Its first gap is therefore 2³¹. When it is credited, the history union is
complete. Run 4's claim boundary is the one the ruling set and is not widened here.

### 2.10 Reach for Optuna and the ML (D1)

Keys are put in place. Nothing is wired.

- **Optuna.** Within a search the study is the memory. Across searches nothing is carried.
  `repeat_of` is descriptive. A later carry-forward proposal would have to name its source
  and mapping, separate resume from reuse, obtain review of any new privileged caller, keep
  dataset and session compatibility, and show benefit against a fresh-search baseline on
  unseen draws at comparable budgets, with the choosing data separate from the judging data.
- **ML.** The search of a **retained winning row** is derivable from the generation chain
  and the link rows, because the merge keeps the prior row unless the new score is strictly
  greater (`utils/run_finalizer.py:757-807`). A missing ancestor or link means the origin
  is **unknown**. It is never assumed to be the current search. The derivation says nothing
  about losing or tied candidates. No array and no finalizer edit is proposed.

## 3. What changes in the rulings

| Ruling | Changes | Does not change |
|---|---|---|
| Beta 2026-08-07 §4 (cursor law) | WATCHER's start comes from the open search's first gap | First-gap arithmetic; the domain; append-only; publication before credit; fail-closed lookup; no reset or deletion |
| Beta 2026-08-07 §3 | The "search/study identity" binding becomes a real record. Publication as certified now also needs evidence of a completed sweep | The history row, its `coverage_id` and its door |
| Beta 2026-08-08 R1; R2 boundary | The cursor-zero boundary is closed for the search cursor | Dataset stays provenance on the history record |
| F-1 (2026-11-11) | The clean stop reports the search's state | No halt file, no escalation, no later step |
| Completion plan F4 | See §4 | F1–F3; Brief II and the soak after finished |
| S167; Route-A; D6.2; threshold provenance | Nothing | All of them |

## 4. Finished (D2, D5)

- **F4, reworded:** the history record reports `java_lcg`, modes `{constant, variable}`,
  COMPLETE over `[0, 2³²)`, every credited interval backed by evidence of a completed
  sweep, with no host freeze. It shows the method can sweep the whole space. It is not a
  claim about every window or about hit rate.
- **F5a, new, part of finished, before Run 4:** with the history COMPLETE in a throwaway
  database and dispatch intercepted, a newly opened search yields a Step 1 command, and no
  existing row is changed or deleted.
- **F5b, after finished:** the soak confirms it in production, in the plan's existing
  order after Brief II.

## 5. What this proposal does not do

- **Decide the cross-dataset survivor merge.** §2.6 refuses such a search until the owner
  decides. Run 4 is not affected: same dataset as attempt 9.
- **Decide the session-scoped authority** (BACKLOG §5). The record holds what is needed to
  apply that decision. Credit for one session stream is not credit for every declared
  option.
- **The next-chunk loop, launch and approval gates.** Separate tracks (S179; the delegated
  gates).
- **Walk-forward.** It becomes a client of this record.
- **The fingerprint registry.** Left as it is.

## 6. Files and suites

Edits expected: a new search-ledger module; `database_system.py`; `agents/watcher_agent.py`
(`:2027-2110` and the Step 1 freshness branch); `window_optimizer_integration_final.py`
(the credit gate before publication, `:3056-3060`, the study name);
`window_optimizer_bayesian.py` (`:806`); `window_optimizer.py` and
`agent_manifests/window_optimizer.json` (the search id as a declared parameter on all three
hops, proven to arrive from a WATCHER-shaped params dict). Not edited:
`utils/seed_coverage_ledger.py`, `utils/run_finalizer.py`, `miner/`, the kernels, the
22-array contract.

**Suite plan.** The history ledger's own assertions stay as they are. The seed-domain
suite's WATCHER arm (`tests/test_seed_domain_cursor_amendment.py:675-704`) tests the
scheduling authority that this design changes: its fixture change is named explicitly in
the code brief, and a paired case is added, "history COMPLETE and a search open gives a
command". No obsolete fallback is kept to hold the old fixture green. The Route-A evidence
gets its scoped refresh. Gate 22 is red while the edits are uncommitted. The rig parity set
is not touched.

## 7. Evidence before code review

CPU-only, isolated state, objective and dispatch intercepted. No fleet.

**First, before any edit — M1:** continue a study whose completed count equals the budget;
report actual objective calls, the candidate and publication outcome, and the credit writes,
each separately.

Then Beta's seven groups (R1 Q11), which the code brief carries in full as its gate list:

1. Credit: no work; a completed sweep with zero survivors; valid recovered work.
2. Budget and resume: two chunks in one study; crash inside chunk 2; PRUNED and FAIL; the
   D6.2 controls.
3. Identity: explicit repeat; retry; study reuse refused; one open selector.
4. Publication and link failures; wrong generation, dataset or mode; gaps, overlaps, exact
   endpoints.
5. Effective input: named version A after the pointer moves to B; each configuration field
   mutated alone; parameters proven to arrive; threshold controls kept.
6. Authority and freshness: the clean stops; no default search on error; no bypass of
   BOOTSTRAP; no inherited pins.
7. Provenance and import: retained, new, improved and tied rows; missing link; import
   replay and rejection.

## 8. Order

1. No second design review (owner, 2026-11-14). This revision answers Beta's R1 in §0.1.
   Beta sees those answers when it reviews the code.
2. One brief for one Claude Code session on Opus, M1 first:
   `docs/CLAUDE_CODE_INSTRUCTIONS_STEP1_SEARCH_RECORD_STAGE1_20261114.md`. It builds what
   Run 4 needs (stage 1). The rest of this design is stage 2, after Run 4, and is listed
   in that brief.
3. Beta reviews the code before commit, through the relay, carried by that session.
4. F5a shown. The Route-A evidence refresh. Run 4. Then Brief II and the soak.

## 9. For the owner, later

The cross-dataset merge (§2.6) comes due before the first search on newer draws. The
session-scoped authority (BACKLOG §5) comes due with it.

## 10. What Alpha did not verify

- Attempt 9's facts in §2.9 come from the reads report of 2026-11-13, not from a read by
  Alpha. Its Optuna study and trial budget are unknown.
- The complete route by which Step 1 freezes a named dataset version.
- Facts 4 and 5 in §1 by execution.
- Whether the S179 gates exist on VM101. The Chapter 13 chapter says they did not land.
- The Gate 22 list and the other suite arm names: from the Stage A report.
