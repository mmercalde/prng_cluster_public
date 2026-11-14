# CLAUDE CODE INSTRUCTIONS — STEP 1 SEARCH RECORD, STAGE 1 (what Run 4 needs)

## Prior record

**Searches run (public mirror, HEAD `b2d2822`, 2026-11-14):**

- The 19 governing records listed in `docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md`
  "Prior record" are carried here by reference. They were found with the searches listed
  there, over `docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`, `docs/TB_RULING_REQUEST_*`,
  `docs/PROPOSAL_*`, `docs/CHAPTER_*`, `docs/*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`,
  `docs/BACKLOG.md`, `docs/LEADS.md` and `docs/exchange/`.
- For this brief: `grep -rnIiE "search_id|step1_search|search_ledger|open_search" --include=*.py .`
  (excluding `apply_s*.py` and `docs/`).
- `grep -n "exhaustive_progress" docs/CLAUDE_CODE_REPORT_STEP1_RERUN_POLICY_STAGE_A.md docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_0.md docs/exchange/from_beta/X-STEP1-SEARCH_R1_BETA_REVIEW.md` (no hit).
- `grep -n "GOVERNED_FILES\|BETA_MINIMUM_FILES" scripts/gate12_parity_gate.py`.
- Read: `docs/exchange/RELAY_PROTOCOL.md` §1–§3, `AGENTS.md` §5, `CLAUDE.md` "Prior record
  first", `docs/CLAUDE_CODE_INSTRUCTIONS_OWNER_LAND_SCRIPT_20261113.md` §0,
  `docs/exchange/to_beta/X-0001_R1_ALPHA_REVIEW_REQUEST_prod_shape_run_scoping.md` (header).

**Documents that already decide this subject:**

| Document | What it settles | Label |
|---|---|---|
| `docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md` | The design this brief implements. Owner decisions D1–D5 in its §0 | ALREADY RECORDED. This brief adds no design |
| `docs/exchange/from_beta/X-STEP1-SEARCH_R1_BETA_REVIEW.md` (`b2d2822`) | Beta's review of v1.0. One LIVE objection (Q4, credit), accepted in v1.1 §2.3. Seven evidence groups (Q11) | ALREADY RECORDED. The groups are this brief's gate list |
| Owner, 2026-11-14 | No second design round with Beta. Beta reviews the code, through the relay, carried by this session | ALREADY RECORDED in v1.1 "Status" and §8 |
| `docs/CLAUDE_CODE_REPORT_STEP1_RERUN_POLICY_STAGE_A.md` (`d47f48c`) | The map of the code as built | ALREADY RECORDED |
| `docs/CLAUDE_CODE_REPORT_RANGE_MINER_VM101_READS.md` R1–R3 (`2930a37`) | Attempt 9's facts on VM101 | ALREADY RECORDED. Not re-read by this session (stage 2 imports it) |
| `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:154-179` | Run 4's claim boundary | ALREADY RECORDED. Not changed |
| `docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md` §A | How a session reads SQLite | ALREADY RECORDED. Binding here |

**Items this brief raises:**

| Item | Label |
|---|---|
| The split into stage 1 (before Run 4) and stage 2 (after Run 4) | NEW. Alpha's scoping. v1.1 §8 names it. Beta is asked about the boundary (Q10 below) |
| The name `search_id` is already taken: `exhaustive_progress.search_id` (`database_system.py:96-105`, written at `window_optimizer.py:928` as `step1_<prng>_<seed_start>`). The new identifier is named `step1_search_id` everywhere | NEW. Found while writing this brief. `exhaustive_progress` is not edited |
| A pre-dispatch refusal when a chunk would run zero trials (§2, S8) | NEW, conditional on M1. It is the stage 1 stand-in for v1.1 §2.7's budget |
| Close and status commands at the operator seam (§2, S2) | NEW. v1.1 §2.2 names the lifecycle events; this gives the operator a way to write them |

**Not searched:** VM101's live state, ser8 archives, the private repository, the project
PDFs. Line anchors below were read at `b2d2822`. **Re-read each one before relying on it.**

---

**From:** Team Alpha · **Owner:** Michael · **Run on:** VM101, as `michael`, venv
`source ~/venvs/torch/bin/activate` · **Model:** Opus
**Base commit:** the commit that lands this brief (its parent is `b2d2822`). Record
`git rev-parse HEAD` in your first-step output.
**Review class:** BEFORE-COMMIT. This changes WATCHER and the Step 1 launch path.
**Report:** `docs/CLAUDE_CODE_REPORT_STEP1_SEARCH_RECORD_STAGE1.md`, in your worktree.

**Why.** Once `[0, 2³²)` is certified for `java_lcg`, WATCHER builds no Step 1 command for
that family again (`agents/watcher_agent.py:2068-2085`). Run 4 would create that state. The
owner ruled "Fix first": the way out is built and shown before Run 4 runs.

**The yardstick (CLAUDE.md).** TFM is judged by prediction-pool hit rate against `k/1000` on
unseen draws. Nothing here is judged by state recovery or a "true seed". Do not write
either into code, comments, tests or the report.

## 0. Rules for this session

- **No commit, push, branch or tag.** Michael lands everything.
- **Worktree:** `git worktree add --detach /home/michael/tfm_worktrees/step1_search HEAD`.
  Write only inside it. The one exception is the review request file, §4.
- **No fleet, no GPU, no SSH to any rig.** CPU only. Dispatch and the sieve are intercepted
  in every test.
- **Isolated state.** Every database, ledger, study, staging directory and output your
  tests touch is a throwaway under the worktree or a temp directory. In the first step,
  print the paths of the live database, the live coverage ledger, the live miner ledger
  and the halt file. You open the live databases read-only, as the binding addendum §A
  says, or not at all. You never write to them and never create or remove the halt file.
- **Not edited:** `utils/seed_coverage_ledger.py`, `utils/run_finalizer.py`, everything
  under `miner/`, the kernels, `prng_registry.py`, the 22-array contract,
  `exhaustive_progress` and its two functions, `scripts/owner_land.py`,
  `scripts/beta_relay.py`, `gate12_launch.sh`.
- **Not changed in behaviour:** S167 (no automatic warm start), Route-A containment, the
  D6.2 resume rulings, threshold provenance, the pre-dispatch domain wall, the freshness
  check for steps other than Step 1.
- **Rig parity set.** If any file you need to edit is in `GOVERNED_FILES` or
  `BETA_MINIMUM_FILES` (`scripts/gate12_parity_gate.py:174`, `:189`), stop and report.
- **Long commands** run under `nohup`, never tmux.
- **Another session is working in parallel** in `/home/michael/tfm_worktrees/volume_limit`
  on different files. Do not read from or write to that directory.
- **First step:** `pgrep -af beta_relay.py`; `git rev-parse HEAD`; `git status --porcelain`
  in the main tree; create the worktree; activate the venv; the live paths above. Record
  the output. Then run, in the worktree at the base commit, every suite in §3 "Existing
  suites" and record each result. That is the baseline.

## 1. M1 — measure before any edit

CPU only, isolated state, the objective and dispatch intercepted so that no sieve runs.

1. **Run A:** Step 1 through `window_optimizer.py` on a fresh named study, `--trials 1`,
   chunk `[0, N)`. One trial completes.
2. **Run B:** the same study resumed (`--study-name` the same, resume on), `--trials 1`,
   chunk `[N, 2N)`. `window_optimizer_bayesian.py:806` gives
   `_trials_to_run = max_iterations - _resumed_completed`.
3. **Run C:** as run B, with the manifest defaults instead (`study_name ''`,
   `resume_study False`).

For runs B and C report, each as its own line: the number of objective calls; what
`_checkpoint_finalizer_input` returned (`window_optimizer_integration_final.py:2923`);
whether `_finalize_run_d3_5` published (`:3001-3011`); whether `record_publication` wrote a
certified row for the chunk (`:3056-3060`); whether a row reached `step1_trial_history`.

Write the M1 section of the report before you edit anything. If M1 shows that v1.1 §2.3
cannot be implemented as written, stop and report.

## 2. What to build

Implement `docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md` §2.1–§2.7, limited to the
items below. Where this brief and the proposal differ, the proposal governs and you report
the difference.

| # | Item | Proposal | Where |
|---|---|---|---|
| S1 | **The search ledger.** A new module with its own schema version and exact-schema validation, in the same database as the coverage ledger. Tables: the search record (every field in §2.2); lifecycle events (opened, superseded, completed); the chunk-open event; the chunk link; a non-certifying record of attempts that earned no credit, with the reason. Append-only: no code path updates or deletes a row, and a test proves it. At most one open search per family and mode set, enforced in one transaction. The search's cursor is `first_uncovered_seed` (`utils/seed_coverage_ledger.py:551-605`), imported and applied to that search's credited chunks inside its declared range | §2.2, §2.3 | new `utils/step1_search_ledger.py`; accessors in `database_system.py` beside `:342-385` |
| S2 | **Opening, closing and reading a search, by the operator only.** At the WATCHER CLI seam (`agents/watcher_agent.py:3919-3932`): open a search (range, reason, and the pins supplied through the existing `--params` route), close one (a lifecycle event with a reason), print status. Opening writes the record and exits. It runs no step. Opening is not a manifest parameter and is not readable from a params dict, a config file, a stored record or an environment variable. A retry of the same opening gives the same id. A second open search for the same family and mode set refuses. The dataset fields come from the current frozen dataset (`miner/dataset_authority.py:229-254`), read only; if it cannot be resolved, refuse | §2.2, §2.5 | `agents/watcher_agent.py` |
| S3 | **WATCHER schedules from the open search.** In `agents/watcher_agent.py:2027-2110`: Step 1's start is the first gap of the open search, **assigned even when it is 0**. Three durable results: no open search (clean stop), search complete (clean stop), lookup failure (blocks Step 1). Nothing falls back to seed 0 and nothing opens a search by default. The clean stop keeps F-1's form (`:2902-2909`): no halt file, no escalation, no later step | §2.4 | `agents/watcher_agent.py` |
| S4 | **Refusals before dispatch.** The dataset presented for the run must be the one the search was opened on. The session mapping, pins, search-space digest, sampler, pruning and budget presented must equal the record. Each mismatch refuses and names both values. Stored settings are never authority: nothing reads pins out of a search record into `_operator_pin_params` or `_pin_bundle` | §2.4, §2.5, §2.6 | `agents/watcher_agent.py` |
| S5 | **Step 1 is due when the search has a gap.** For Step 1 only, file times no longer skip it (`:1835-1858`). Every other step keeps the freshness check | §2.6 | `agents/watcher_agent.py` |
| S6 | **`step1_search_id` on all three hops,** as a declared parameter: manifest, `window_optimizer.py`, the integration signature. Proven to arrive from a WATCHER-shaped params dict. A Step 1 run reached without it, through WATCHER, refuses | §6 | `agent_manifests/window_optimizer.json`, `window_optimizer.py`, `window_optimizer_integration_final.py` |
| S7 | **The credit gate.** Nothing is published as certified, and no coverage row is written, unless at least one trial in state COMPLETE belongs to this chunk and its sieve finished every required phase over the whole interval. Zero survivors from a completed sweep is credited. The gate sits in the caller; the finalizer and the ledger are not edited. An attempt that earns no credit is written to the non-certifying record with its reason | §2.3 | `window_optimizer_integration_final.py` before `:3001-3011` and `:3056-3060` |
| S8 | **The chunk-open event and the link.** Before the chunk's first trial: chunk identity, interval, the study's trial count at that moment, the checkpoint run id. After a credited publication: one link row, with `coverage_id`, `step1_search_id`, chunk id, trial counts by state, `generation_id`. Interval, family, modes and generation are taken from the publication witness and checked against the search; the caller cannot assert them. Idempotent. If the link write fails after publication, the failure surfaces, the search does not advance, and a retry uses the same witness. **If M1 showed zero objective calls in run B:** a chunk whose study has already met its budget refuses before dispatch, naming the counts | §2.3, §2.7 | `window_optimizer_integration_final.py`, `utils/step1_search_ledger.py` |
| S9 | **Study ownership.** A search owns one study. A study that already belongs to another search, or a name that does not match the record, refuses | §2.7 | as needed |

**Stage 2. Do not build it. List anything you had to touch from it.** The per-chunk trial
budget (`window_optimizer_bayesian.py:806`); the per-search trial table and evaluation
records (§2.8); the attempt 9 import (§2.9); the route to a named dataset version after the
pointer has moved (§2.4; stage 1 refuses instead); the ML origin index (§2.10); the
Chapter 13 opening route (§2.5).

## 3. Tests

A new suite, in the style of the existing ones, CPU only, isolated state. Each case that
guards a refusal has a mutant that turns it red, and the report shows the mutant red.

| Group (Beta R1 Q11) | Stage 1 cases |
|---|---|
| 1 Credit | No COMPLETE trial: no certified publication, no coverage row, no link, one non-certifying record. A completed sweep with zero survivors: credited. A trial COMPLETE in the study but belonging to another chunk: not credit for this one |
| 2 Budget and resume | The M1 case after the change: refused before dispatch, or runs, as S8 decides; never a silent no-op. PRUNED-only and FAIL-only chunks earn no credit. The D6.2 controls still pass |
| 3 Identity | Retry of an opening gives the same id. An explicit repeat gives a new id with `repeat_of` set. A second open search refuses. Study reuse refuses. Close, then open, works |
| 4 Publication and link | Link failure after publication surfaces and does not advance. Wrong generation, dataset or mode refuses. Gaps, overlaps and exact endpoints at 0, 2³¹ and 2³² |
| 5 Effective input | Each configuration field mutated alone refuses. The dataset pointer moved after opening refuses, naming both versions. `step1_search_id` arrives on all three hops. Threshold provenance controls still pass |
| 6 Authority and freshness | The three WATCHER results. No default search on error. No opening from a params dict, a stored record or the environment. No pins inherited from a record. Step 1 due with a gap even when its output file is fresh; other steps unchanged |
| **F5a** | In a throwaway database with history COMPLETE over `[0, 2³²)` and dispatch intercepted: with no open search WATCHER stops cleanly; after the operator opens a search, WATCHER builds a Step 1 command starting at that search's first gap; no existing row in any table is changed or deleted (row counts and a content hash, before and after) |
| **Run 4 shape** | History holds `[0, 2³¹)`. The operator opens a pinned search over `[2³¹, 2³²)` with the Run 4 pins (the built command at `docs/SESSION_CHANGELOG_20260828_RUN4_ROUTING_PATCH.md:212-222`). WATCHER builds `--seed-start 2147483648 --max-seeds 2147483648` with all seven pins unchanged. Different pins refuse |

**Existing suites.** Run before and after, and explain every difference:
`tests/test_seed_domain_cursor_amendment.py`, `tests/test_s172_run4_routing_patch.py`,
`tests/run4_prelaunch_provenance_check.py`, `tests/test_watcher_followup_repairs.py`,
`tests/test_gate12_cleantree_admission.py`, `tests/test_s172_phase4_coordinator.py`, and any
other suite that imports a file you edited.

- The WATCHER arm of the seed-domain suite (`tests/test_seed_domain_cursor_amendment.py:675-704`,
  `arm_complete_generates_no_watcher_run`) tests the authority this brief changes. Change its
  fixture explicitly, say so in the report, and add the paired case: history COMPLETE and a
  search open gives a command.
- Keep no obsolete fallback in the code to hold an old fixture green.
- Gate 22 is red while the edits are uncommitted. Report it as that and nothing else.

## 4. Review by Beta, carried by this session

When the code and tests are done and you have stopped editing:

1. Write the report (§5) in the worktree.
2. `pgrep -af beta_relay.py`. If a relay is running, wait 60 seconds and check again.
3. Write the request file in the **main tree**, the only write you make outside the
   worktree: `/home/michael/distributed_prng_analysis/docs/exchange/to_beta/X-STEP1-SEARCH-CODE_R1_ALPHA_REVIEW_REQUEST_search_record_stage1.md`.
   Layout as `AGENTS.md` §5 and `X-0001_R1`: it opens with `## Prior record`; review class
   BEFORE-COMMIT; base commit; every file under review by absolute worktree path, marked new
   or modified; a line saying the files are in a worktree and are not committed; then the
   questions below, unchanged, each with the allowed answers.
4. Start the relay from the main tree at once, as `docs/exchange/RELAY_PROTOCOL.md` §2
   says. Write nothing in the main tree while it runs.
5. Act on Beta's positions by the protocol's table. At most three rounds. You never
   overrule Beta and never argue it into agreement. You stop when you disagree with a LIVE
   objection, when a fix is outside §2, or when Beta cannot read the worktree.

**Questions for Beta (use as written):**

1. Credit (S7): can anything be published as certified, or any coverage row be written,
   without a COMPLETE trial of this chunk that finished every required phase?
2. Scheduling (S3): is the start taken from the open search's first gap in every case,
   including 0, and are the three results distinct and durable?
3. Authority (S2, S4): is there any route by which a search is opened, or pins are
   supplied, other than the operator's command line?
4. The search ledger (S1): append-only in fact, one open search enforced in one
   transaction, and not a second certification door?
5. The link (S8): derived from the publication witness, idempotent, and safe when it fails
   after publication?
6. Refusals (S4): dataset and each configuration field, before dispatch, naming both values?
7. Parameters (S6): does `step1_search_id` arrive on all three hops from a WATCHER-shaped
   params dict?
8. Existing suites: is every changed fixture justified, and is any obsolete fallback kept?
9. Untouched: the coverage ledger, the finalizer, `miner/`, the kernels, the 22-array
   contract, S167, Route-A containment, D6.2 and threshold provenance.
10. The stage boundary: is anything left to stage 2 that Run 4 needs to be safe?
11. Any governing record this work missed.

## 5. Report

`docs/CLAUDE_CODE_REPORT_STEP1_SEARCH_RECORD_STAGE1.md`, opening with `## Prior record`.
Then: first-step output; M1; each item S1–S9 with what was built and `file:line`; the test
table with counts and the mutants shown red; existing suites before and after; anything
touched from stage 2; Beta's last position per question and what was done; anything
unresolved with both positions; what you did not verify.

End with the landing list: every new or modified file with its sha256, in two groups.
Group 1, files the landing script takes. Group 2, files it refuses and Michael lands by
hand (`agents/watcher_agent.py`, anything under `agent_manifests/`).

## 6. Stop gates

Stop and report, without working around it, when: M1 contradicts the design; an item needs
an edit to a file listed as not edited; a file you must edit is in the rig parity set; a
test can only pass by writing to live state; Beta holds a LIVE objection you disagree with
or whose fix is outside §2; the third round ends; Beta cannot read the worktree.
