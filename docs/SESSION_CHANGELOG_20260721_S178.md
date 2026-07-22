# Session Changelog — 2026-07-21 — S178 (S177 resubmission: Proposal v1.1 + Analyzer v2.1)

**Session:** S178
**Author:** Team Alpha (Claude)
**Track:** WATCHER KPI Governance — TB conditional-approval resubmission
(`CLAUDE_CODE_BRIEF_S177_RESUBMISSION_v1.md`)
**Ruling:** `docs/TB_RULING_S177_KPI_GOVERNANCE.md` (CONDITIONAL APPROVAL)
**Status:** Recommend-only. No commits, no policy edits, no runtime implementation, no cluster
runs. Documents + one analyzer tool only. Delivered to Michael → Team Beta.
**Tree head (unchanged all session):** `0c3166a630be321809f415bb28af28e319d0fe1b`.

---

## Summary

Produced the S177 resubmission: **Proposal v1.1** and **Analyzer v2.1**. Tasks 0–3 (all
read-only source traces) formed the evidence base; the Task 1 control-flow trace was reported
to Michael and the gate placement confirmed before any document was written. Every code
location cited is a read-only trace on `0c3166a`; nothing runtime was modified.

**Gate placement confirmed by Michael before drafting:**
- Primary governance gate → `chapter_13_orchestrator.py` between `:363` and `:365` (right
  after `evaluate_triggers()` + run-counter bump, before the `should_trigger` branch / Step 3
  LLM / all four `request_approval()` sinks).
- Defense-in-depth gate → inside `chapter_13_triggers.py` `approve_request()` before `:537`
  (`execute_learning_loop`).
- Hit@K canonical source → **Option A** (full ranked list), accepted.
- `SELFPLAY_RECOMMENDED` included in the trigger registry as found.

## Deliverables (write-set)

1. `docs/PROPOSAL_WATCHER_KPI_GOVERNANCE_STATES_v1_1.md` — ten TB §5 elements, v1.0→v1.1 delta
   (§0), blocker-coverage appendix (§14). v1.0 left intact for the audit trail.
2. `watcher_kpi_metricC_deterministic_v2_1.py` — six TB §4 edge fixes on top of v2 (intact).
3. `docs/SESSION_CHANGELOG_20260721_S178.md` — this file.
4. `watcher_kpi_metricC_v2_1_findings.json` — analyzer's canonical output (Task 5 mandate; see
   write-set note below).

## Blocker checklist — addressed-at pointers (1:1 with TB ruling)

| TB item | Correction | Addressed at |
|---------|-----------|--------------|
| B1 enforcement gate wrong point | Primary gate in orchestrator + defense-in-depth at approval | Proposal §3, §3.1 |
| B2 evaluator returns one trigger | Structured candidates; govern-filter before priority sort | Proposal §4 |
| B3 schema misses live triggers | 8-trigger registry; verbatim fail-closed default; N_DRAWS=cadence; SELFPLAY_RECOMMENDED incl. | Proposal §5 |
| B4 global vs per-metric state | global_state + per-metric state; allowed matrix; missing/invalid→BOOTSTRAP+audit_only; schema_version | Proposal §6 |
| B5 one canonical pool source | Option A full ranked list; per-tier record; Hit@100/300 ≠ 20-pool best_rank | Proposal §7 |
| B6 gates vs quality blended | Four sections; hard invariants source-traced or GAP; NO_EXACT_HITS/LOW_POOL_COVERAGE reclassified; min_survivor not claimed | Proposal §8 |
| B7 persistence source of truth | Single append-only JSONL ledger; full field list; idempotency; atomic/lock/recovery/dedup/encoding | Proposal §9 |
| B8 derived null rates | p_null = unique_k/draw_space; store draw_space/requested_k/unique_k | Proposal §10 |
| §5 el.10 files + test plan | 9-file implementation list with gating tests | Proposal §13 |
| Analyzer A pool≤space | fatal validation | v2.1 + Proposal §11 |
| Analyzer B p_hit=0 miss run | p_miss==1→m, p_miss==0→inf | v2.1 + §11 |
| Analyzer C strict JSON | allow_nan=False; inf/unavailable→null+status | v2.1 + §11 |
| Analyzer D "fire at null" wording | rename field; --fire-horizon; verdict kept | v2.1 + §11 |
| Analyzer E threshold/integer validation | collapse_threshold finite∈[0,1]; max_misses exact int | v2.1 + §11 |
| Analyzer F source commit | git rev-parse HEAD in findings | v2.1 + §11 |

## Task 1 control-flow trace (reported + confirmed)

Post-draw flow `run_cycle()` (`chapter_13_orchestrator.py:251–460`): diagnostics `:282` →
S140b Hit@K-from-one-best_rank `:305–309` → root cause (observe-only) `:321–355` →
`evaluate_triggers()` `:359` → run-counter `:363` → `should_trigger` early-exit `:365–369` →
LLM `:373–390` → validate + `request_approval()` at `:415/:422/:429/:440`. Both the
orchestrator route (default `approval_route="orchestrator"`, `:417–425`) and the WATCHER route
(`:413–416`) create the actionable request via the trigger manager — confirming TB's finding
that a WATCHER-only gate (v1.0) cannot cover the orchestrator route. Priority collapse at
`chapter_13_triggers.py:349–354` (`best = triggered[0]`; losers survive only as names in
`metrics["all_triggers"]` `:367`) is the point the structured-candidate contract replaces.

## Analyzer v2 → v2.1 diff summary

- **Fix A:** new `pool_size <= draw_space` fatal check in `analyze()`.
- **Fix B:** `expected_draws_to_miss_run` — `q>=1.0 → float(m)` (was `inf`); `q<=0.0 → inf`.
- **Fix C:** `json.dump(..., allow_nan=False)`; new `waiting_time_field()` maps inf/NaN/None →
  `(null, status)`; every finding gains an `expected_draws_status` field; assumed block gains
  per-metric status fields.
- **Fix D:** field `expected_draws_to_first_false_fire_at_null` →
  `expected_draws_to_first_fire_at_uniform_null`; `false_alarm_horizon_draws` →
  `fire_horizon_draws`; flag `--false-alarm-horizon` → `--fire-horizon` (old name retained as a
  DEPRECATED alias). Verdict `FIRES-WITHIN-HORIZON-AT-NULL` unchanged.
- **Fix E:** new `_validate_exact_integer()` (rejects bool/NaN/inf/fractional) for
  `max_consecutive_misses`; `collapse_threshold` validated finite ∈ [0,1].
- **Fix F:** new `source_commit()` (read-only `git rev-parse HEAD`) → `probe.analyzed_source_commit`.
- Probe name → `watcher_kpi_metricC_deterministic_v2_1`; default `--out` →
  `watcher_kpi_metricC_v2_1_findings.json`.

### Verification run (all passed)

- **Main path** (`--pool-size 20 --policies watcher_policies.json`): collapse `1.0204`, 5-miss
  `5.3146`, both `finite`; `analyzed_source_commit=0c3166a…`; strict JSON parses.
- **Fail-loudly (SystemExit) paths:** no-policy/no-flags; pool=0; draw_space=0; max_misses=0;
  fire-horizon≤0; assumed-rate∉[0,1]; **pool>space** (fix A); **non-integer misses 5.5 via
  policy** (fix E); **collapse_threshold 1.5** (fix E). All exit 1 with a `[FATAL]` message.
- **Degenerate-but-VALID (fix B/C), NOT errors:** `--assumed-healthy-hit-rate 0` → miss-run
  `5.0` (finite, was inf in v2); `--assumed-healthy-hit-rate 1` → collapse/miss `null` +
  `"infinite_never_fires"`, strict JSON parses, zero `Infinity`/`NaN` tokens.
- Deprecated alias `--false-alarm-horizon` still functions.

**⚠️ Brief vs fix-B nuance (flagged for TB):** the S177 brief Task 5 listed `rate=0` and
`rate=1` under "show each fails loudly." Fix B explicitly defines those as **valid** degenerate
inputs (deterministic `m` / infinite→null), so forcing them to error would contradict the fix
TB required. v2.1 therefore handles rate=0/rate=1 correctly (finite / null+status) instead of
erroring, and only genuinely-invalid inputs fail loudly. Raising for confirmation rather than
silently choosing.

## Concurrent-session observations

- The S172 Phase-5 agent is active on this tree. Its lane (`miner/range_miner_coordinator.py`,
  `tests/test_s172_phase4_coordinator.py`, `window_optimizer_integration_final.py`, and the
  newly-appeared `tests/test_s172_phase5_d0.py`) shows churn in `git status` — **untouched by
  this session.** No read-modify-write on any S172 file. No state-altering git run.
- No "missing file" alarms this session; tree stayed consistent. HEAD never moved (`0c3166a`).
- **Untracked `.py` note:** this session adds one stray untracked `.py`
  (`watcher_kpi_metricC_deterministic_v2_1.py`, in the write-set). Per memory
  `phase4-gate22-untracked-py-sensitivity`, the Phase-4 coexistence gate reds on any stray
  untracked `.py`; the S172 agent is on Phase 5, but flag in case a Phase-4 gate is re-run
  before Michael stages these deliverables.

## Write-set note (for Michael)

The brief's CONCURRENT-SESSION rules pinned the write-set to exactly three files + memory.
Task 5 additionally mandates saving analyzer output to `watcher_kpi_metricC_v2_1_findings.json`
— the analyzer's designed default artifact, in the KPI lane (not S172), and the direct
companion TB reviewed for v2. I produced it as Task 5 requires and am flagging the four-vs-three
discrepancy explicitly rather than silently resolving it.

## Non-scope (unchanged)

No thresholds selected; no autonomous enforcement; `watcher_policies.json` and all runtime code
UNCHANGED; Phase C walk-forward deferred to post-S172 Phase 7; `minimum_hit_rate` deprecation
sequence per Proposal §5 (v1.0)/§13; D1/D2 resolved (stale `_find_results` map = separate P3).

## Delivered

`docs/PROPOSAL_WATCHER_KPI_GOVERNANCE_STATES_v1_1.md`,
`watcher_kpi_metricC_deterministic_v2_1.py`,
`watcher_kpi_metricC_v2_1_findings.json`,
`docs/SESSION_CHANGELOG_20260721_S178.md` → Michael → Team Beta. Stopped after delivery: no
implementation, no Phase A, no walk-forward, no policy edits.
