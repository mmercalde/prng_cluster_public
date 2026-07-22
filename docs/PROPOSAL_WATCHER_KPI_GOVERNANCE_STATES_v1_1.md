# PROPOSAL — WATCHER KPI Governance States (BOOTSTRAP → CALIBRATING → GOVERNED) v1.1

**Date:** 2026-07-21
**Session:** S178 (S177 resubmission — TB conditional-approval revision)
**Author:** Team Alpha
**Status:** DRAFT PROPOSAL — for Team Beta code-review before any implementation is scoped.
**Authority:** Recommend-only. This document changes nothing. No thresholds are selected;
no autonomous enforcement is enabled; `watcher_policies.json` and all runtime code are
UNCHANGED. Every code location below is a **read-only trace on tree `0c3166a`**, not an edit.
**Basis:** `docs/TB_RULING_S177_KPI_GOVERNANCE.md` (CONDITIONAL APPROVAL) resolving the
eight blockers on `docs/PROPOSAL_WATCHER_KPI_GOVERNANCE_STATES_v1_0.md`.
**Verified tree head:** `git rev-parse HEAD == 0c3166a630be321809f415bb28af28e319d0fe1b`
(matches the head TB reviewed against).

> Scope discipline: v1.1 corrects the **control-flow and metric-source contracts** TB
> ruled unresolved. It still selects **no** numerical thresholds — those come only after
> the Phase C walk-forward produces empirical distributions (§12, non-scope).

---

## 0. v1.0 → v1.1 delta (what changed and why)

| # | v1.0 said | TB blocker | v1.1 correction | §  |
|---|-----------|-----------|-----------------|----|
| 1 | Gate in WATCHER (`watcher_agent.py`); `chapter_13_triggers.py` unchanged | B1 | Primary gate moves **into `chapter_13_orchestrator.py`** immediately after `evaluate_triggers()`, before LLM/proposal/approval/routing | §3 |
| 2 | "pure evaluator requires no change" | B2 | Evaluator must expose a **structured candidate list**; governance filtering happens **before** priority selection | §4 |
| 3 | Schema covered Hit@K, Lift@K, misses, pool-structure only | B3 | **Complete trigger registry** (8 action producers) + **fail-closed default** for any unmapped trigger | §5 |
| 4 | `state` only at global level; metrics carry `enforcement` | B4 | `global_state` **and** per-metric `state`; allowed-enforcement matrix; invalid/missing → `BOOTSTRAP+audit_only`; schema version field | §6 |
| 5 | Assumed `generate_diagnostics()` can record independent Hit@20/100/300 | B5 | Canonical **Option A** full-ranked-list source contract; the current single 20-pool `best_rank` **must not** back Hit@100/300 | §7 |
| 6 | `pool_structure_metrics` all `audit_only` (blended) | B6 | Four-section split; **hard invariants source-traced** to a runtime consumer or marked GAP | §8 |
| 7 | Two histories referenced (diag JSON archives + trigger history + proposed JSONL) | B7 | **One canonical** append-only JSONL KPI-event ledger + idempotency contract; diag archives demoted to non-canonical | §9 |
| 8 | Hardcoded `null_rate` 0.02/0.10/0.30 | B8 | **Derived** `p_null = unique_k / draw_space`; store `draw_space/requested_k/unique_k` | §10 |
| — | (analyzer v2) | §4 A–F | Analyzer **v2.1** delivered alongside (six edge fixes); see §11 | §11 |

Unchanged and re-affirmed from v1.0 (TB §2 approved without modification): the three-state
model, the metric-class separation, the `minimum_hit_rate` deprecation sequence, D1/D2
resolved, no thresholds / no autonomous enforcement. Those sections are carried forward
compressed; the audit trail lives in v1.0 (left intact).

---

## 1. The state model (TB §2.1 — approved, carried forward)

| State | Meaning | Performance/cadence/regime triggers | Hard-invariant gates |
|-------|---------|-------------------------------------|----------------------|
| **BOOTSTRAP** | No empirical TFM baseline exists | **audit-only** (record hypothetical; do NOT dispatch) | **ACTIVE** |
| **CALIBRATING** | Enough walk-forward observations to estimate distributions | **shadow** (candidate thresholds; counted; human review) | ACTIVE |
| **GOVERNED** | Per-trigger evidence bundle satisfied (TB §5.3/Phase E) | **active** (per-trigger) | ACTIVE |

Transitions are per-trigger (individual `state` field, §6). Entry/exit criteria map to TB
Phases A–E exactly as in v1.0 §1; no count/stability bar is chosen here.

---

## 2. Live post-draw control flow (evidence base for §3–§4)

Full trace of `Chapter13Orchestrator.run_cycle()` (`chapter_13_orchestrator.py:251–460`),
pinned file:line, tree `0c3166a`:

```
:282  diagnostics = generate_diagnostics()                         # Step 1
:290  save_diagnostics(diagnostics)
:292-319  [S140b] derives hit_at_20/100/300 from ONE best_rank ➜ DB (see §7)
:321-355  Step 1b post-draw root cause (observe-only; does not gate)
:359  trigger_eval = self.trigger_manager.evaluate_triggers(diagnostics)   # Step 2  ◀── PRIMARY GATE
:360  result["steps"]["triggers"] = trigger_eval.to_dict()
:363  self.trigger_manager.increment_run_counter()
:365-369  if not trigger_eval.should_trigger: return "no_action_needed"
:371  logger.info("Trigger fired: ...")
:373-390  Step 3  LLM analysis  (proposal = llm_advisor.analyze_diagnostics :386)
:392-441  Step 4  validate + create approval request, via one of:
          :415  request_approval(trigger_eval)   # approval_route == "watcher"
          :422  request_approval(trigger_eval)   # v1 human-approval (default/orchestrator)
          :429  request_approval(trigger_eval)   # ESCALATE
          :440  request_approval(trigger_eval)   # no-LLM trigger-based path
```

TB's key structural finding confirmed in source: the actionable request is created through
**both** an orchestrator route (`:417–425`, default `approval_route="orchestrator"`) and a
WATCHER route (`:413–416`, `approval_route="watcher"`), and the trigger manager owns the
downstream execution path (`request_approval()` → `approve_request()` → learning loop). A
WATCHER-only dispatch gate (v1.0's proposal) therefore cannot cover the orchestrator route.

---

## 3. Blocker 1 — Primary enforcement gate in the orchestrator

**Location: `chapter_13_orchestrator.py`, inserted between `:363` and `:365`** — immediately
after `evaluate_triggers()` returns (and the run-counter bump), and **before** the
`should_trigger` branch, Step 3 LLM (`:373`), proposal validation (`:395`), and all four
`request_approval()` sinks (`:415/:422/:429/:440`).

Surrounding lines quoted (unchanged source):

```python
359   trigger_eval = self.trigger_manager.evaluate_triggers(diagnostics)
360   result["steps"]["triggers"] = trigger_eval.to_dict()
361
362   # Increment run counter
363   self.trigger_manager.increment_run_counter()
364
      # ◀── §3 PRIMARY GOVERNANCE GATE INSERTS HERE ──▶
365   if not trigger_eval.should_trigger:
366       logger.info("... No triggers fired - system healthy")
367       result["outcome"] = "no_action_needed"
368       self._log_cycle(result)
369       return result
```

**Gate behavior.** The gate consumes the structured candidate contract (§4), resolves each
candidate's governance `enforcement`, and for an **audit-only** outcome performs verbatim:

```
record the hypothetical trigger  (candidate + governance state → KPI-event ledger §9)
mark dispatched = false
mark approval_requested = false
skip LLM action analysis  unless explicitly requested for observation
return outcome = "audit_only_trigger"
```

So on an audit-only trigger the orchestrator returns before `:373`; no LLM call, no proposal,
no `request_approval()` on either route. Only a candidate whose per-metric `enforcement ==
"active"` **and** whose `global_state == GOVERNED` (matrix §6) is allowed to proceed into the
existing `:371→:441` dispatch flow.

**Why here, not in WATCHER:** this single point sits upstream of both the orchestrator and
WATCHER request routes, satisfying TB §3-B1 ("audit-only enforcement must be applied in the
Chapter 13 orchestrator before LLM/approval/request creation"). The WATCHER dispatch gate may
still exist as belt-and-suspenders, but is no longer the *only* gate.

### 3.1 Defense-in-depth gate (Blocker 1, second gate)

A stale or hand-created `pending_approval.json` must not execute if governance has since moved
to audit-only/BOOTSTRAP. Second gate: **inside `Chapter13TriggerManager.approve_request()`
(`chapter_13_triggers.py:516–549`), before `:537`**:

```python
516   def approve_request(self) -> bool:
523       request = self.check_approval()          # reads pending_approval.json
...
535       # Execute the learning loop
      # ◀── §3.1 DEFENSE-IN-DEPTH GATE INSERTS HERE (re-read governance state) ──▶
536       steps = request.get("steps_to_run", [3, 5, 6])
537       success = self.execute_learning_loop(steps)   # runs pipeline
```

The gate re-loads current `kpi_governance`, re-derives the request's `governance_key`, and
refuses execution (records `dispatched=false`, `outcome="blocked_by_governance_at_approval"`)
unless that key is currently `active`/`GOVERNED`. This closes the "old request bypasses
current state" path TB flagged.

---

## 4. Blocker 2 — Structured trigger candidates before priority selection

**Current collapse point (`chapter_13_triggers.py`).** Seven conditions append to a local
`triggered` list (`:250, :261, :272, :283, :297, :306, :313`); the list is then sorted by a
fixed priority order and **only the top element survives as actionable**:

```python
349   triggered.sort(key=lambda x: (priority_order.index(x[0]) ... , -x[2]))
354   best = triggered[0]
356   return TriggerEvaluation(should_trigger=True, trigger_type=best[0], ...
367       metrics={..., "all_triggers": [t[0].value for t in triggered]})   # losers = names only
```

The problem TB identified is real in source: a high-priority `audit_only` trigger can mask a
lower-priority `active` one (which is lost), and — conversely — a legacy trigger omitted from
the schema stays actionable even in BOOTSTRAP.

**Correction.** The evaluator exposes the full candidate set **before** the sort/select at
`:349–354`, and governance filtering runs first. Adopt TB's contract:

```json
{
  "candidates": [
    {
      "trigger_type": "hit_rate_collapse",
      "action": "learning_loop",
      "confidence": 0.95,
      "reasoning": "...",
      "governance_key": "prospective_outcome_metrics.hit20",
      "enforcement": "audit_only"
    }
  ],
  "selected_active_trigger": null,
  "audit_only_triggers": ["hit_rate_collapse"]
}
```

**Order of operations (new):** build `candidates` (one per fired condition, each tagged with
its `governance_key` + resolved `enforcement`) → partition into `active`-eligible vs
`audit_only` by the matrix (§6) → **then** apply the existing priority sort *within the
active-eligible set only* to choose `selected_active_trigger` (or `null`). Every non-active
candidate is recorded in `audit_only_triggers` and written to the KPI-event ledger (§9); none
is silently dropped. `TriggerEvaluation.to_dict()` (`:92–102`) gains `candidates` /
`selected_active_trigger` / `audit_only_triggers`; the legacy `metrics["all_triggers"]`
name-only list is retained for backward compatibility but is no longer the governance record.

---

## 5. Blocker 3 — Complete trigger registry + fail-closed default

Every **action-producing** trigger in the live evaluator (`chapter_13_triggers.py`) and its
upstream flags (`chapter_13_diagnostics.py`), with source, action, and governance mapping.
"Enf. (BOOTSTRAP)" is the enforcement each carries in the default state — all `audit_only`.

| # | Trigger | Source (file:line) | Action produced | Governance section (§8) | Enf. (BOOTSTRAP) |
|---|---------|--------------------|-----------------|--------------------------|------------------|
| 1 | `CONSECUTIVE_MISSES` | triggers.py:247–254 | LEARNING_LOOP (3→5→6) | prospective_outcome_metrics | audit_only |
| 2 | `CONFIDENCE_DRIFT` | triggers.py:257–266 | LEARNING_LOOP | cadence_and_regime_triggers (drift) | audit_only |
| 3 | `HIT_RATE_COLLAPSE` | triggers.py:268–277 | LEARNING_LOOP | prospective_outcome_metrics | audit_only |
| 4 | `N_DRAWS` (periodic) | triggers.py:279–288 | LEARNING_LOOP | **cadence_and_regime_triggers (cadence)** | audit_only |
| 5 | `REGIME_SHIFT` (decay+churn) | triggers.py:290–302 | FULL_PIPELINE (1→6) | cadence_and_regime_triggers (regime) | audit_only |
| 6 | `LLM_PROPOSED` (`RETRAIN_RECOMMENDED`) | triggers.py:304–311; diag flag :661–662 | LEARNING_LOOP | cadence_and_regime_triggers (regime) | audit_only |
| 7 | `REGIME_SHIFT_POSSIBLE` | triggers.py:313–319; diag flag :664–665 | FULL_PIPELINE | cadence_and_regime_triggers (regime) | audit_only |
| 8 | `SELFPLAY_RECOMMENDED` | triggers.py:819–842 (`should_request_selfplay`) → `request_selfplay` :762 | selfplay retrain request (WATCHER-authorized) | cadence_and_regime_triggers (regime) | audit_only |

Notes on classification (per TB §2.5 / Blocker 3):
- **#4 `N_DRAWS` is a cadence trigger**, not a prospective-performance metric — it fires on
  `runs_since_retrain >= retrain_after_n_draws` (a clock), independent of any Hit@K outcome.
- **#8 `SELFPLAY_RECOMMENDED`** is included as found (per Michael's confirmation). It is
  already gated on WATCHER authorization (`requires_watcher_approval=True`, triggers.py:795)
  and produces a request artifact, not direct execution — but it *is* action-producing, so it
  gets an explicit `audit_only` mapping rather than relying on the WATCHER gate alone.
- `MANUAL` / `SELFPLAY_RETRAIN` enum members (triggers.py:68–69) are not auto-dispatched by
  the evaluator and are out of the automatic-registry scope; a manual request is a human act.

**Fail-closed default (verbatim, TB §3-B3):**

```
Any trigger with no governance mapping:
    enforcement = audit_only
    dispatch    = prohibited
    emit a configuration warning naming the unmapped trigger_type
```

The gate (§3) resolves an unknown `governance_key` to `audit_only` and logs
`"[GOVERNANCE][WARN] unmapped trigger <type> → audit_only (fail-closed)"`. It must **not**
fall through to the legacy actionable path.

---

## 6. Blocker 4 — Fail-closed state / enforcement validation

**Preferred design (TB §3-B4): `global_state` + per-metric `state`.**

```json
{
  "kpi_governance": {
    "schema_version": "1.1.0",
    "global_state": "BOOTSTRAP",
    "_state_note": "global_state = system lifecycle phase; per-metric state = per-trigger transition.",
    "prospective_outcome_metrics": {
      "hit20": { "state": "BOOTSTRAP", "enforcement": "audit_only", "...": "..." }
    }
  }
}
```

- `global_state` = system lifecycle phase.
- per-metric `state` = that trigger's individual transition (a trigger may reach GOVERNED
  while others remain BOOTSTRAP).
- `enforcement` = per-trigger operational status ∈ `{audit_only, shadow, active}`.

**Allowed-enforcement matrix (validated at load; illegal combinations fail closed):**

| Global state | Allowed enforcement values |
|--------------|----------------------------|
| BOOTSTRAP    | `audit_only` |
| CALIBRATING  | `audit_only`, `shadow` |
| GOVERNED     | `audit_only`, `shadow`, `active` |

A config with `global_state=BOOTSTRAP` and any metric `enforcement=active` **fails closed**:
the loader clamps that metric to `audit_only` and emits a warning; it does not honor `active`.

**Missing / malformed / unknown value → `BOOTSTRAP + audit_only` (never legacy).** If
`kpi_governance` is absent (it is absent today — confirmed: no `kpi_governance` key in
`watcher_policies.json` on `0c3166a`), unparseable, or contains an unrecognized state/
enforcement token, the loader returns the safe default for **every** trigger and must **not**
silently fall back to the currently-active legacy triggers. A `schema_version` mismatch is
treated the same way (fail closed, warn).

---

## 7. Blocker 5 — Canonical Hit@20/100/300 source (Option A)

**Current single-pool reality, source-traced:**
- The diagnostic engine loads **one** file: `DEFAULT_PREDICTION_POOL = "prediction_pool.json"`
  (`chapter_13_diagnostics.py:50`), a single `{"predictions": [...], "pool_size": N}` shape
  (`load_predictions` :100–118, default `pool_size` 20).
- `generate_diagnostics()` (`:750`) passes that one pool to `compute_prediction_validation()`
  (`:811`), which computes a single `best_rank = min(hit_ranks)` (`:258`).
- The orchestrator's S140b block derives **all three tiers from that one `best_rank`**:

```python
307   _hit_at_20  = 1.0 if _best_rank is not None and _best_rank <= 20  else 0.0
308   _hit_at_100 = 1.0 if _best_rank is not None and _best_rank <= 100 else 0.0
309   _hit_at_300 = 1.0 if _best_rank is not None and _best_rank <= 300 else 0.0   # chapter_13_orchestrator.py
```

With a 20-entry pool, `best_rank ∈ [1,20] ∪ {None}`, so Hit@100 and Hit@300 **collapse onto
Hit@20** — they carry no independent information. This is exactly TB's Blocker 5.

**Root cause is at emit time, not ranking time:** `prediction_generator._build_prediction_pool`
ranks the *full* survivor set (`ranked_idx = np.argsort(predicted_quality)[::-1]`,
`prediction_generator.py:836`) but then **truncates to the top `pool_size`**
(`for idx in ranked_idx[:pool_size]`, `:841`) before `_save_predictions` persists it
(`:896–951`, canonical `predictions/next_draw_prediction.json`). The full ranked list exists
in memory and is discarded.

**Recommendation: Option A — full `ranked_predictions.json`.** Chosen over Option B
(explicit tight/balanced/wide pools) because the ranking Option A needs **already exists** at
`:836`; Option A is a **serialization change** (persist the full ranked list, or at least the
top ≥300 unique), whereas Option B would require the generator to construct and reconcile three
separate pools. Hit@K is then: *the actual draw appears within the first K unique ranked
outputs.*

**Per-tier record (TB §3-B5), stored per KPI event (§9):**

| Field | Meaning |
|-------|---------|
| `source_artifact` | `predictions/ranked_predictions.json` (Option A) |
| `requested_k` | 20 / 100 / 300 |
| `unique_k` | distinct ranked outputs actually available in the first K (≤ requested_k) |
| `duplicate_policy` | how duplicates in the ranked list are collapsed before counting K |
| `ranking_contract` | ordering key = descending `predicted_quality` (`:836`), ties broken by rank index |
| `actual_hit` | boolean: actual draw ∈ first `unique_k` |
| `null_rate` | derived (§10): `unique_k / draw_space` |

**Stated plainly:** Hit@100 and Hit@300 **MUST NOT** be derived from the current 20-pool
`best_rank`. Producing the full ranked artifact (Option A) is a **required prerequisite** for
Phase A metric plumbing; until it exists, only Hit@20 is a real measurement and Hit@100/300
must be reported `unavailable`, not fabricated.

---

## 8. Blocker 6 — Four-section schema; hard invariants source-traced

Per TB §3-B6, the single blended `pool_structure_metrics` block is split into four sections.
**No gate is labeled "active" without a traced runtime consumer.**

```json
"kpi_governance": {
  "hard_invariants": {},
  "structural_quality_metrics": {},
  "prospective_outcome_metrics": {},
  "cadence_and_regime_triggers": {}
}
```

### 8.1 `hard_invariants` — active in all states, each with its runtime-consumer trace

| Invariant | Runtime consumer (traced) | Verdict |
|-----------|---------------------------|---------|
| Required artifact exists / stage output fresher than inputs | `check_output_freshness()` `watcher_agent.py:419` via `get_step_io_from_manifest()` `:386–416` (manifest `primary_output`, hard-fail if missing) | **TRACED → active** |
| Manifest-driven input/output validation | `get_step_io_from_manifest()` `watcher_agent.py:406,414` | **TRACED → active** |
| Pool not empty | `compute_prediction_validation` handles `pool_size==0` by returning zeros (`chapter_13_diagnostics.py:223–234`) — **does not block/gate** | **NOT a gate today → GAP; keep audit_only until a gate is implemented** |
| Pool ≤ outcome space | no runtime check found in the diagnostics/trigger path (analyzer v2.1 checks it, runtime does not) | **GAP → not claimed active** |
| Feature-schema hash valid / finite values / weights normalized / duplicate bound | not located in the diagnostics path this session (NPZ contract wall is S172 Phase 5, CLAUDE.md §6) | **GAP → not claimed active** |
| Minimum survivor population | `selfplay.min_survivor_count: 1000` exists in policy, but **no consumer traced in the Chapter-13 path** (it lives under `selfplay`) | **NOT proven enforced → audit_only until traced** |

Per TB's explicit instruction, `min_survivor_count` is **not** asserted as an enforced gate
without a trace, and the GAP rows are flagged for implementation review rather than claimed.

### 8.2 `structural_quality_metrics` — soft, need calibration (audit_only until distributions known)

weight concentration (`topK_weight_share`), `prediction_entropy`, `effective_pool_size`,
`pool_to_pool_stability`, normal `duplicate_count` distribution, `outcome_space_breadth`,
`unique_prediction_count`. All `audit_only`.

Reclassified out of "structural gate" per TB §3-B6:
- **`NO_EXACT_HITS`** (`chapter_13_diagnostics.py:606–607`) is a **performance miss**, not a
  structural non-degeneracy gate → prospective_outcome, audit_only.
- **`LOW_POOL_COVERAGE`** (`:609–610`) is a **summary breadth flag**, not an action-producing
  hard gate (it only nudges `pool_size` in `generate_recommended_actions` :715–716) →
  structural_quality, audit_only.

### 8.3 `prospective_outcome_metrics`

`hit20/hit100/hit300` (§7), `lift20/lift100/lift300`, `consecutive_misses`, plus the
reclassified `NO_EXACT_HITS`. All `audit_only`; null rates derived (§10).

### 8.4 `cadence_and_regime_triggers`

`N_DRAWS` (cadence), `REGIME_SHIFT` (decay+churn), `REGIME_SHIFT_POSSIBLE`, `CONFIDENCE_DRIFT`
(drift), `RETRAIN_RECOMMENDED`/`LLM_PROPOSED`, `SELFPLAY_RECOMMENDED`. All `audit_only`.

---

## 9. Blocker 7 — One canonical idempotent KPI-event ledger

**Choice: a single append-only JSONL event ledger** (`kpi_events.jsonl`) as the source of
truth. Rationale: (1) Phase C/D walk-forward needs one clean, ordered, cheaply-replayable time
series — a directory of per-draw JSON files would have to be enumerated and sorted on every
replay; (2) idempotency is naturally expressed as last-revision-wins over an appended ledger;
(3) it is genuinely *one* history, satisfying TB's "do not maintain two independent canonical
histories."

**Non-canonical, explicitly:** the existing `post_draw_diagnostics.json`,
`.previous_diagnostics.json`, and the timestamped diagnostic archives remain **diagnostics
artifacts**, not a second KPI-event history; the trigger-history writer (`_save_trigger_history`
`chapter_13_triggers.py:189–195`) is likewise not the canonical KPI record. Only
`kpi_events.jsonl` is canonical for KPI events.

**Record fields (TB §3-B7, full list):**

```
schema_version, draw_id, session, draw_timestamp,
prediction_artifact_fingerprint,
metric_values{ hit20, hit100, hit300, lift@K, structural_quality..., consecutive_misses,
               window_decay, survivor_churn, confidence_drift },
global_governance_state, per_trigger_enforcement_state,
all_trigger_candidates (§4), selected_active_trigger, hypothetical_action,
approval_requested, dispatched, cycle_id, source_commit
```

**Idempotency rules (TB §3-B7):**
- same `draw_id` + same `prediction_artifact_fingerprint` → **idempotent, no duplicate append**.
- same `draw_id` + **changed** `fingerprint` → **new revision** appended, carrying
  `supersedes: <prior_revision_id>` and a monotonically increasing `revision` for provenance;
  readers take the highest `revision` per `draw_id`.

**JSONL operational contract (required because JSONL was chosen):**
- **Atomic append:** open `O_APPEND`, serialize the record to a single compact line
  (`json.dumps(..., allow_nan=False)` + `"\n"`), one `write()` of the full buffer; POSIX
  guarantees an `O_APPEND` write below `PIPE_BUF` is not interleaved. Never partial-format.
- **Locking:** advisory `fcntl.flock(LOCK_EX)` around the read-index-then-append critical
  section so idempotency checks and the append are atomic w.r.t. concurrent writers.
- **Crash recovery:** records are line-delimited and self-contained; on read, a trailing torn
  line (no terminating `\n` or failing `json.loads`) is discarded — a crash mid-append loses at
  most the final uncommitted record, never corrupts prior events.
- **Duplicate detection:** an in-memory index keyed by `(draw_id, fingerprint)` built from a
  one-pass scan on open; append is skipped on an exact key match.
- **Strict encoding:** UTF-8, one JSON object per line, newline-terminated, `allow_nan=False`
  (no `Infinity`/`NaN` tokens — same discipline the analyzer §11 enforces).

**Hook point:** the ledger append is added at the single producer that already assembles the
metrics — after `generate_diagnostics()` and after the §3 gate resolves candidates, at the end
of Step 2 in `run_cycle()` — so there is exactly one write site per draw, no new pipeline stage.

---

## 10. Blocker 8 — Derived null rates

Remove the hardcoded `"null_rate": 0.02 / 0.10 / 0.30`. Those hold only for a 1,000-outcome
space with distinct pools of exactly 20/100/300 and drift if draw space, pool sizes, or
duplicate collapse (reducing unique K) change.

**Contract:** store per KPI event `draw_space`, `requested_k`, `unique_k`; **derive**

```
p_null = unique_k / draw_space
```

and record the derived rate in each KPI event for auditability. No `null_rate` literal appears
in `kpi_governance`; the schema carries `draw_space` (resolved from the active game config) and
the deriver, not the answer. (For CA Daily 3, `draw_space = 1000`; distinct-pool sanity:
20/1000=0.02, 100/1000=0.10, 300/1000=0.30 — matching the old literals, now derived not baked.)

---

## 11. Analyzer v2.1 (delivered alongside — TB §4 A–F)

`watcher_kpi_metricC_deterministic_v2_1.py` (v2 left intact). Six edge fixes, each verified:

| Fix | Change | Verification |
|-----|--------|--------------|
| A | `pool_size <= draw_space` (fatal) | `--pool-size 1001 --draw-space 1000` → `[FATAL] ... cannot exceed draw_space (TB §4.A)` |
| B | `p_miss==1` → run at draw `m`; `p_miss==0` → inf | rate=0 → miss-run `5.0` (was inf); rate=1 → inf→null |
| C | strict JSON `allow_nan=False`; inf/unavailable → null + status | rate=1 output has `null` + `"infinite_never_fires"`, parses, no `Infinity` token |
| D | rename → `expected_draws_to_first_fire_at_uniform_null`; `--fire-horizon` (alias `--false-alarm-horizon`); no "false fire" at null | field-name diff confirmed vs v2; verdict `FIRES-WITHIN-HORIZON-AT-NULL` kept |
| E | `collapse_threshold` finite & ∈ [0,1]; `max_misses` exact integer | `max_consecutive_misses=5.5` (via policy) → `[FATAL] ... exact integer`; `--collapse-threshold 1.5` → `[FATAL] ... range [0,1]` |
| F | record `git rev-parse HEAD` in findings | `analyzed_source_commit = 0c3166a...` |

**Null reproductions preserved** against the real `watcher_policies.json` (`--pool-size 20`):
collapse `1.0204`, 5-miss `5.3146` — both `finite`. Findings saved to
`watcher_kpi_metricC_v2_1_findings.json` (strict JSON, parses). Full invalid-input matrix
(no-policy, pool=0, space=0, misses=0, bad horizon, rate∉[0,1], pool>space, non-integer misses,
threshold∉[0,1]) all fail loudly; degenerate-but-valid rate=0/rate=1 are handled per fix B/C
(finite `m` / null+status) rather than forced to error — flagged here because the S177 brief
listed rate=0/rate=1 under "fails loudly", but fix B defines them as **valid** degenerate
inputs, so treating them as errors would contradict the fix. See changelog for the full matrix.

---

## 12. Explicit non-scope (unchanged from v1.0 §7)

- **Phase C walk-forward deferred** to post-S172 Phase 7 (GPU/cluster-heavy; must not create a
  launch-storm competing with S172 Phase 5–7 acceptance runs).
- **No thresholds selected** — every performance/cadence/regime field is `null` / `audit_only`.
- **No autonomous enforcement enabled** — GOVERNED activation is per-trigger, gated on the
  TB §5.3 evidence bundle after Phase D.
- **D1/D2 resolved** on `main` (accepted); stale `_find_results` `step_files` map
  (`watcher_agent.py:1317–1322`) remains a separate P3 cleanup, outside this KPI work.
- **This proposal edits nothing.** All §3/§3.1/§4/§9 code locations are read-only traces; the
  edits they describe happen only if and when TB approves implementation.

---

## 13. Implementation file list + test plan (Blocker 10 — for TB scoping, not executed here)

| # | File | Change | Gating test(s) |
|---|------|--------|----------------|
| 1 | `watcher_policies.json` | Add `kpi_governance` (schema_version, global_state=BOOTSTRAP, four sections, all `audit_only`); mark `convergence_targets.minimum_hit_rate` deprecated | schema-load test: valid parse; matrix rejects `BOOTSTRAP+active`; missing block → BOOTSTRAP+audit_only |
| 2 | `chapter_13_triggers.py` | `evaluate_triggers()` returns structured candidates (§4); governance filter before priority sort; `TriggerEvaluation.to_dict()` gains candidate fields | unit: two fired triggers (audit_only high-prio + active low-prio) → active one selected, both recorded; unmapped trigger → audit_only + warning |
| 3 | `chapter_13_orchestrator.py` | §3 primary gate between `:363`/`:365`; audit-only outcome shape; skip LLM unless observation-requested | cycle test: audit_only candidate → returns `outcome="audit_only_trigger"`, no LLM call, no `request_approval`, ledger row `dispatched=false` |
| 4 | `chapter_13_triggers.py` | §3.1 defense-in-depth gate in `approve_request()` before `:537` | test: stale `pending_approval.json` for now-audit_only key → `approve_request()` refuses, records `blocked_by_governance_at_approval` |
| 5 | `prediction_generator.py` | Option A: persist full `ranked_predictions.json` (§7) | test: ranked artifact has ≥300 unique ranked outputs; Hit@100/300 differ from Hit@20 on a constructed case |
| 6 | `chapter_13_diagnostics.py` | Independent Hit@20/100/300 from ranked list; per-tier record; derived `p_null` (§10) | test: Hit@K counts unique-K membership; `p_null = unique_k/draw_space`; no hardcoded 0.02/0.10/0.30 |
| 7 | `chapter_13_diagnostics.py` / new `kpi_event_ledger.py` | Single append-only `kpi_events.jsonl` writer (§9): atomic append, flock, dedup, strict encoding, revision-on-fingerprint-change | test: same draw_id+fingerprint twice → one row; changed fingerprint → revision row; torn final line discarded on read; concurrent append no interleave |
| 8 | (config) | Elevate/keep hard invariants active; close §8.1 GAP rows (pool-empty, pool≤space, finite/schema/normalized, min_survivor trace) | per-gate test that each claimed-active invariant actually blocks on violation |
| 9 | `watcher_policies.json` | Remove `minimum_hit_rate` after schema lands + re-confirm zero consumers | `/bin/grep` re-confirmation shows no new consumer |

All nine land only after Team Beta reviews this v1.1.

---

## 14. Blocker-coverage appendix (Task 0 — TB can verify 1:1)

| TB item | Addressed in |
|---------|--------------|
| Blocker 1 — enforcement gate wrong point | §3 (primary, orchestrator `:363/:365`) + §3.1 (defense-in-depth, `approve_request` before `:537`) |
| Blocker 2 — evaluator returns one trigger | §4 (structured candidates; filter before priority) |
| Blocker 3 — schema misses live triggers | §5 (8-trigger registry + verbatim fail-closed default; N_DRAWS as cadence; SELFPLAY_RECOMMENDED included) |
| Blocker 4 — global vs per-metric state | §6 (global_state + per-metric state; allowed matrix; missing/invalid → BOOTSTRAP+audit_only; schema_version) |
| Blocker 5 — one canonical pool source | §7 (Option A full ranked list; per-tier record; Hit@100/300 not from 20-pool best_rank) |
| Blocker 6 — gates vs quality blended | §8 (four sections; hard invariants source-traced or marked GAP; NO_EXACT_HITS/LOW_POOL_COVERAGE reclassified; min_survivor not claimed) |
| Blocker 7 — persistence source of truth | §9 (single JSONL ledger; full field list; idempotency; atomic/lock/recovery/dedup/encoding) |
| Blocker 8 — derived null rates | §10 (`p_null=unique_k/draw_space`; store draw_space/requested_k/unique_k; no literals) |
| §5 element 10 — files + test plan | §13 |
| Analyzer fix A — pool≤space | §11 / v2.1 (verified) |
| Analyzer fix B — p_hit=0 miss run | §11 / v2.1 (verified) |
| Analyzer fix C — strict JSON | §11 / v2.1 (verified) |
| Analyzer fix D — "fire at null" wording | §11 / v2.1 (verified) |
| Analyzer fix E — threshold/integer validation | §11 / v2.1 (verified) |
| Analyzer fix F — source commit | §11 / v2.1 (verified) |

---

## 15. References

- `docs/TB_RULING_S177_KPI_GOVERNANCE.md` — the conditional-approval ruling this revises to
- `docs/PROPOSAL_WATCHER_KPI_GOVERNANCE_STATES_v1_0.md` — prior version (intact audit trail)
- `docs/TB_RULING_S176_WATCHER_KPI.md` — original ruling
- `chapter_13_orchestrator.py:251–460` — post-draw cycle (gate points §3/§3.1)
- `chapter_13_triggers.py:201–370, 419–549` — evaluator + approval/execution boundary
- `chapter_13_diagnostics.py:50,204–279,750–811` — single-pool diagnostic source (§7)
- `prediction_generator.py:836–862, 896–951` — full ranking truncated at emit (§7)
- `watcher_agent.py:386–450` — manifest-driven structural validation (§8.1)
- `watcher_kpi_metricC_deterministic_v2_1.py` — analyzer v2.1 (§11)
- `watcher_policies.json` — current policy (UNCHANGED; no `kpi_governance` key present)
