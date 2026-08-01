# Claude Code Brief — S177 Resubmission: Proposal v1.1 + Analyzer v2.1 — v1

**Runs on:** VM101, as `michael`, from `/home/michael/distributed_prng_analysis`.
**Context:** Team Beta issued CONDITIONAL APPROVAL of the S177 governance-states
proposal (ruling: `docs/TB_RULING_S177_KPI_GOVERNANCE.md` — read it FIRST; it is
the spec for this session). Eight proposal blockers + six analyzer fixes are
required before implementation is approved. This session produces the
resubmission: **Proposal v1.1 and Analyzer v2.1. Documents and one tool only —
NO runtime implementation.**

---

## CONCURRENT-SESSION RULES (read before anything else)

Another Claude Code agent is actively working S172 Phase 5 on THIS tree.

- **Your write-set is EXACTLY three files:**
  `docs/PROPOSAL_WATCHER_KPI_GOVERNANCE_STATES_v1_1.md`,
  `watcher_kpi_metricC_deterministic_v2_1.py`,
  `docs/SESSION_CHANGELOG_YYYYMMDD_S<N>.md`.
  Plus append-only memory writes. NOTHING else. Creating or editing any other
  file is out of scope.
- **Never touch the S172 lane:** `miner/`, `tests/test_s172*`,
  `window_optimizer_integration_final.py`, `docs/CLAUDE_CODE_INSTRUCTIONS_S172*`.
  Do not read-modify-write anything the other agent may be editing.
- **All runtime files are READ-ONLY this session:** `chapter_13_orchestrator.py`,
  `chapter_13_triggers.py`, `chapter_13_diagnostics.py`, `watcher_agent.py`,
  `watcher_policies.json`, manifests. You trace them; you do not change them.
- **Expect transient tree churn.** If a file appears missing or changed
  unexpectedly, re-check after a pause before concluding anything (last
  session's "missing .py" alarm was a race with the S172 agent). Never
  "restore" or recreate a file you believe lost without re-verifying.
- **Do not run git commands that alter state** (add/commit/stash/checkout/etc.).
  Read-only git (status, log, diff, rev-parse) is fine. The other agent's D0
  work may be staged/committed by Michael at any time — that is not your
  concern and must not be disturbed.
- Standard rules: no commits/pushes, no pipeline runs, no policy edits,
  read source before every claim (file:line), inspect.signature() discipline,
  use **/bin/grep** for any search that must cover .json files (the shell grep
  here is a ugrep wrapper honoring .gitignore — recorded in memory).

---

## Task 0 — Read the ruling; build the blocker checklist

Read `docs/TB_RULING_S177_KPI_GOVERNANCE.md` in full. Produce a working
checklist mapping each of the 8 blockers + 6 analyzer fixes to where it will be
addressed. This checklist becomes an appendix of Proposal v1.1 (TB can verify
coverage 1:1).

---

## Task 1 — Source-trace the control flow (Blockers 1 & 2) — READ-ONLY

The heart of the resubmission. Trace, with file:line citations:

1. **The full Chapter13Orchestrator post-draw flow:** diagnostics generation →
   `trigger_manager.evaluate_triggers(diagnostics)` → LLM analysis → proposal
   validation → approval-request creation → routing (WATCHER route AND the
   direct orchestrator route TB identified).
2. **The exact insertion point for the primary governance gate:** immediately
   after `evaluate_triggers(...)` returns, before any LLM/approval/request
   step. Quote the surrounding lines.
3. **The approval/execution boundary** (`request_approval()` /
   `approve_request()` in the trigger manager) for the defense-in-depth gate:
   where a stale or manually created request would be re-checked against
   current governance state. Quote it.
4. **The evaluator's priority-selection logic:** where multiple fired
   candidates are collapsed to one `TriggerEvaluation` and the rest survive
   only as names in `metrics["all_triggers"]`. This defines what the
   structured-candidates contract (TB Blocker 2) must replace.

**Deliverable 1:** a trace section (goes into v1.1) with every hook point
pinned to file:line, including the TB-required audit-only outcome shape
(record hypothetical trigger, dispatched=false, approval_requested=false, skip
LLM unless observation-requested, outcome="audit_only_trigger").

---

## Task 2 — Complete trigger registry (Blocker 3) — READ-ONLY

Enumerate EVERY action-producing trigger in the live evaluator: hit-rate
collapse, consecutive misses, confidence drift, periodic N_DRAWS,
window-decay+churn regime shift, RETRAIN_RECOMMENDED, REGIME_SHIFT_POSSIBLE,
LLM-proposed actions — plus anything else found in source. For each: source
location, action it can produce, and its proposed governance mapping.
Classify periodic N_DRAWS as a **cadence trigger** (per TB §2.5/Blocker 3),
not a prospective-performance metric. Define the fail-closed default verbatim:
unmapped trigger → enforcement=audit_only, dispatch prohibited, config warning.

---

## Task 3 — Hit@K canonical-source contract (Blocker 5) — READ-ONLY

Trace what the diagnostics actually load (`prediction_pool.json`, single
20-pool) and what artifacts exist that could serve independent tiers
(`ranked_predictions`-style full list? explicit tight/balanced/wide pools per
TB Option A/B?). Read `prediction_generator.py` output surface to determine
which option the current pipeline can already supply vs what it would need to
emit. Recommend Option A or B **based on what exists in source**, and specify
the per-tier record TB requires (source artifact, requested K, unique K,
duplicate policy, ranking contract, hit boolean, derived null rate). State
plainly: Hit@100/300 must NOT be derived from the current 20-pool best_rank.

---

## Task 4 — Write Proposal v1.1

Revise into `docs/PROPOSAL_WATCHER_KPI_GOVERNANCE_STATES_v1_1.md` (v1.0 left
untouched for the audit trail). Must contain TB §5's ten elements, using
Tasks 1–3 as the evidence base:

1. Correct enforcement hook in `chapter_13_orchestrator.py` (Task 1, file:line).
2. Defense-in-depth gate at approval/execution (Task 1).
3. Structured trigger-candidate contract, governance filtering BEFORE priority
   selection (Task 1.4; adopt TB's candidates/selected/audit_only JSON shape).
4. Complete trigger mapping + fail-closed default (Task 2).
5. Fail-closed state/enforcement validation: global_state + per-metric state
   per TB's preferred design, the allowed-enforcement matrix, invalid/missing
   config → BOOTSTRAP+audit_only, schema version field.
6. Canonical Hit@20/100/300 source contract (Task 3).
7. Schema split into four sections: hard_invariants /
   structural_quality_metrics / prospective_outcome_metrics /
   cadence_and_regime_triggers — with every "active" hard invariant
   source-traced to its runtime consumer (do NOT claim min_survivor_count or
   any gate is enforced without the trace; NO_EXACT_HITS is performance, not
   structural; LOW_POOL_COVERAGE is a summary flag, not a hard gate).
8. One idempotent KPI-event persistence contract: choose immutable per-draw
   JSON **or** append-only JSONL (state the choice and why), full TB field
   list (schema_version → source commit), idempotency rules (same
   draw_id+fingerprint → no duplicate; changed fingerprint → new revision with
   provenance), and if JSONL: atomic append, locking, crash recovery,
   duplicate detection, strict encoding.
9. Derived null rates: store draw_space/requested_k/unique_k, derive
   p_null = unique_k/draw_space, record derived rate per KPI event. No
   hardcoded 0.02/0.10/0.30.
10. Exact implementation file list + test plan (what changes where when
    implementation IS approved, with the tests that gate each change).

Header must carry the same authority block as v1.0 (DRAFT — for TB review;
recommend-only; changes nothing) plus a v1.0→v1.1 delta section and the
Task 0 blocker-coverage appendix.

---

## Task 5 — Write Analyzer v2.1

`watcher_kpi_metricC_deterministic_v2_1.py` (v2 untouched). Apply TB §4's six
fixes exactly:

A. `pool_size <= draw_space` validation (fatal).
B. Degenerate rates: q==1 → first m-miss run at exactly m draws; q==0 → inf.
C. Strict JSON: `allow_nan=False`; infinite/unavailable → null + status field.
D. Rename to `expected_draws_to_first_fire_at_uniform_null`; no "false fire"
   wording anywhere at the null; keep verdict FIRES-WITHIN-HORIZON-AT-NULL.
E. Validate collapse_threshold finite/in-range; max_misses exact integer.
F. Record analyzed source commit (git rev-parse HEAD, read-only) in findings.

Then verify: run v2.1 against the real `watcher_policies.json`; confirm the
null reproductions still land at 1.0204 (collapse) and 5.3146 (5-miss); run
every invalid-input path TB tested (no policy, pool=0, space=0, misses=0, bad
horizon, rate outside [0,1]) PLUS the new ones (pool>space, rate=0, rate=1,
non-integer misses) and show each fails loudly. Save output to
`watcher_kpi_metricC_v2_1_findings.json` — strict JSON, verify it parses.

---

## Task 6 — Changelog and stop

`docs/SESSION_CHANGELOG_YYYYMMDD_S<N>.md`: the blocker checklist with
addressed-at pointers, v2→v2.1 diff summary, any concurrent-session
observations. Deliver all three files for Michael → Team Beta. **Stop after
delivering.** No implementation, no Phase A, no walk-forward, no policy edits.
