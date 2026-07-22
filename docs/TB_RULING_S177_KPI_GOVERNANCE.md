# Team Beta Review — S177 WATCHER KPI Governance Proposal and Analyzer v2

## Review basis

Team Beta reviewed:

* `PROPOSAL_WATCHER_KPI_GOVERNANCE_STATES_v1_0.md`
* `SESSION_CHANGELOG_20260720_S177.md`
* `watcher_kpi_metricC_deterministic_v2.py`
* `watcher_kpi_metricC_v2_findings.json`

The review was cross-checked against the current live GitHub `main` branch of `mmercalde/prng_cluster_public`.

Current verified repository head remains:

```text
0c3166a630be321809f415bb28af28e319d0fe1b
```

The analyzer was also executed directly against a representative `watcher_policies.json`, including normal operation and invalid-input paths.

---

# 1. Executive ruling

## Status: CONDITIONAL APPROVAL — REVISION REQUIRED BEFORE IMPLEMENTATION

Team Beta approves the core architectural direction:

```text
BOOTSTRAP → CALIBRATING → GOVERNED
```

Team Beta also approves:

* separating prospective Hit@K outcomes from pool-structure measurements;
* keeping all new numerical performance thresholds unset;
* operating uncalibrated performance triggers in audit/shadow mode;
* retaining structural and catastrophic-safety gates from the first cycle;
* deprecating the unwired synthetic `minimum_hit_rate`;
* recognizing D1 and D2 as resolved on current `main`;
* deferring empirical calibration until the causally correct walk-forward exists.

The proposal is thoughtful and substantially incorporates the S176 ruling.

However, implementation is **not yet approved** because several control-flow and metric-source contracts remain unresolved. The most important issue is that the proposed enforcement gate is placed at the wrong point in the live Chapter 13 flow.

---

# 2. Items approved without modification

## 2.1 Three governance phases

The three-state model is correct:

* `BOOTSTRAP`: no empirical TFM baseline; performance decisions audit-only.
* `CALIBRATING`: candidate policies evaluated in shadow mode.
* `GOVERNED`: individually validated triggers may become active.

The proposal correctly avoids manufacturing sample counts, windows, or collapse floors before real distributions exist.

## 2.2 Separate metric classes

The distinction between:

```text
Prospective outcomes:
Hit@20, Hit@100, Hit@300, Lift@K

Pool structure:
weight share, breadth, entropy, duplicates, stability
```

is approved.

Weight concentration is not the same thing as future-draw inclusion, and neither class should substitute for the other.

## 2.3 `minimum_hit_rate` disposition

The S177 follow-up reports a corrected `/bin/grep` search and confirms zero runtime consumers on tree `0c3166a`. The methodology correction concerning the `ugrep` wrapper and `.gitignore` is valuable and should remain in the changelog.

Approved sequence:

1. Introduce the replacement governance schema.
2. Mark `minimum_hit_rate` deprecated.
3. Reconfirm no consumer after implementation.
4. Remove it in a later cleanup.

## 2.4 D1 and D2 resolution

Approved.

The current manifests and manifest-driven output validation resolve the previously reported Stage 3 and Stage 4 defects. The stale `_find_results()` map may remain a separate P3 cleanup, but it is not part of the KPI-governance implementation.

## 2.5 No thresholds and no autonomous enforcement

Approved.

The proposal remains recommend-only and does not alter `watcher_policies.json`, launch the cluster, or claim empirical performance.

---

# 3. Blocking proposal corrections

## Blocker 1 — The enforcement gate is proposed at the wrong control-flow point

The proposal says:

> The caller, identified as WATCHER / `watcher_agent.py`, should inspect governance state and suppress dispatch, while `chapter_13_triggers.py` remains unchanged.

That is not sufficient for the current live architecture.

The live `Chapter13Orchestrator`:

1. generates diagnostics;
2. evaluates triggers;
3. proceeds into LLM analysis;
4. validates a proposal;
5. creates an approval request.

It can create that actionable request either through the WATCHER approval route or directly through the orchestrator route.

The trigger manager also exposes its own approval and execution path. `request_approval()` creates an actionable Steps 3/5/6 or full-pipeline request, and `approve_request()` executes the learning loop.

### Required correction

The primary enforcement gate must occur in `chapter_13_orchestrator.py` immediately after:

```python
trigger_eval = self.trigger_manager.evaluate_triggers(diagnostics)
```

and before:

* LLM analysis;
* proposal validation;
* approval-request creation;
* WATCHER request routing.

For an audit-only trigger, the orchestrator should:

```text
record the hypothetical trigger
mark dispatched=false
mark approval_requested=false
skip LLM action analysis unless explicitly requested for observation
return outcome="audit_only_trigger"
```

Add a second defense-in-depth gate at the approval/execution boundary so that an old or manually created request cannot bypass current governance state.

A WATCHER dispatch gate may still exist, but it cannot be the only gate.

---

## Blocker 2 — The evaluator returns only one prioritized trigger

The live evaluator collects multiple trigger conditions, sorts them, and returns only the highest-priority trigger as the actionable `TriggerEvaluation`. The other fired triggers survive only as names in `metrics["all_triggers"]`.

This creates a governance problem.

Example:

```text
highest-priority trigger = audit_only
lower-priority trigger   = active
```

If the caller merely suppresses the selected audit-only trigger, the lower active trigger is lost.

The reverse can also occur: a legacy trigger omitted from the new schema may remain actionable even though the intended state is BOOTSTRAP.

### Required correction

The trigger layer must expose a structured candidate list before priority selection.

Recommended contract:

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

Priority selection must occur **after** governance filtering.

The proposal's statement that the pure evaluator requires no change is therefore not approved.

---

## Blocker 3 — The schema does not cover every live trigger

The proposed schema includes:

* Hit@K;
* Lift@K;
* consecutive misses;
* pool-structure metrics.

The current live trigger evaluator also contains:

* confidence drift;
* periodic `N_DRAWS`;
* window-decay plus survivor-churn regime shift;
* `RETRAIN_RECOMMENDED`;
* `REGIME_SHIFT_POSSIBLE`;
* LLM-proposed actions.

Without a complete trigger registry, omitted legacy triggers may continue generating requests during BOOTSTRAP.

### Required correction

Add governance entries for every action-producing trigger, or define an explicit fail-closed default:

```text
Any trigger with no governance mapping:
    enforcement = audit_only
    dispatch = prohibited
    emit configuration warning
```

The periodic ten-draw trigger should be classified separately as a cadence trigger rather than a prospective-performance metric.

---

## Blocker 4 — Global state and per-metric state are inconsistent

The proposal states that governance state is per-KPI, but the schema contains:

```json
"state": "BOOTSTRAP"
```

only at the global level. Individual metrics contain `enforcement`, not `state`.

That can work, but the contract must be stated accurately.

### Required correction

Choose one of these designs:

### Preferred design

```json
{
  "global_state": "BOOTSTRAP",
  "metrics": {
    "hit20": {
      "state": "BOOTSTRAP",
      "enforcement": "audit_only"
    }
  }
}
```

Or explicitly define:

```text
global_state = system lifecycle phase
enforcement  = per-trigger operational status
```

Then provide and validate the allowed matrix:

| Global state | Allowed enforcement        |
| ------------ | -------------------------- |
| BOOTSTRAP    | audit_only                 |
| CALIBRATING  | audit_only, shadow         |
| GOVERNED     | audit_only, shadow, active |

A configuration such as:

```text
global_state=BOOTSTRAP
enforcement=active
```

must fail closed.

If `kpi_governance` is missing, malformed, or contains an unknown value, the safe default must be:

```text
BOOTSTRAP + audit_only
```

It must not silently fall back to the currently active legacy triggers.

Add a governance schema version.

---

## Blocker 5 — Live diagnostics currently have only one canonical pool source

The proposal assumes that `generate_diagnostics()` can record independent:

```text
Hit@20
Hit@100
Hit@300
```

The live diagnostic engine currently loads one file:

```text
prediction_pool.json
```

Its documented format contains one `predictions` array and one `pool_size`, with the example/default pool size of 20.

`generate_diagnostics()` loads that single pool and sends it to `compute_prediction_validation()`.

Therefore, independent Hit@100 and Hit@300 cannot be inferred safely from the current diagnostic source unless it contains the complete ranked prediction list.

The live orchestrator currently derives Hit@20, Hit@100, and Hit@300 from one `best_rank`. If the input pool contains only 20 predictions, those three values collapse to the same event.

### Required correction

Before implementing the schema, define the canonical source for each tier:

```text
Option A:
full ranked_predictions.json
Hit@K = actual appears in first K unique ranked outputs

Option B:
prediction_pools.json
explicit tight/balanced/wide pool arrays
```

For each tier, record:

* source artifact;
* requested K;
* actual unique pool size;
* duplicate policy;
* ranking contract;
* actual-hit boolean;
* null rate derived from unique K and draw space.

Do not implement Hit@100/300 by reusing the current 20-pool `best_rank`.

This is a required prerequisite for Phase A metric plumbing.

---

## Blocker 6 — Structural gates and structural quality metrics are blended

The proposal says structural gates remain active, but its `pool_structure_metrics` entries all begin `audit_only`.

These are not all the same type.

### Hard invariants — active immediately

Examples:

```text
required artifact exists
schema valid
finite values
pool not empty
pool cannot exceed outcome space
required sidecar/hash valid
weight total valid
prohibited duplicates absent
```

### Soft structural-quality KPIs — need calibration

Examples:

```text
weight concentration
entropy
effective pool size
pool-to-pool stability
normal duplicate distribution
```

These may need audit or shadow mode until their expected real-TFM distributions are known.

Additionally:

* `NO_EXACT_HITS` is a performance miss, not a structural non-degeneracy gate.
* `LOW_POOL_COVERAGE` is currently a summary breadth flag, not an action-producing hard gate.
* the existence of `min_survivor_count` in policy does not prove that the live path consumes it as an enforced gate.

### Required correction

Create separate schema sections:

```json
"hard_invariants": {},
"structural_quality_metrics": {},
"prospective_outcome_metrics": {},
"cadence_and_regime_triggers": {}
```

Every claimed existing gate must be source-traced to its runtime consumer before being labeled active.

---

## Blocker 7 — Persistence design needs one source of truth and idempotency

The proposal references:

* existing archived diagnostic JSON files;
* a proposed per-draw JSONL series;
* the existing trigger-history writer.

The live diagnostics system already writes:

* the current `post_draw_diagnostics.json`;
* `.previous_diagnostics.json`;
* one timestamped archived JSON file per diagnostic run.

The existing trigger-history writer is not presently a complete record of every evaluation. Approval requests are archived when approved or rejected, not necessarily when an audit-only trigger is first evaluated.

### Required correction

Define one canonical immutable KPI-event record per draw.

It must include:

```text
schema_version
draw_id
session
draw timestamp
prediction artifact fingerprint
metric values
global governance state
per-trigger enforcement state
all trigger candidates
selected active trigger
hypothetical action
approval_requested
dispatched
cycle_id
source commit
```

It must also handle reprocessing safely:

```text
same draw_id + same prediction fingerprint → idempotent/no duplicate
same draw_id + changed prediction fingerprint → new revision with provenance
```

Choose either:

* immutable per-draw JSON files as the source of truth; or
* an append-only JSONL event ledger.

Do not maintain two independent canonical histories.

If JSONL is used, specify atomic append, locking, crash recovery, duplicate detection, and strict JSON encoding.

---

## Blocker 8 — Null rates should be derived, not duplicated

The proposal hardcodes:

```json
"hit20":  {"null_rate": 0.02},
"hit100": {"null_rate": 0.10},
"hit300": {"null_rate": 0.30}
```

Those values are correct for a 1,000-outcome space and distinct pools of exactly 20, 100, and 300.

They will drift if:

* draw space changes;
* pool sizes change;
* duplicates reduce unique K.

### Required correction

Store or resolve:

```text
draw_space
requested_k
unique_k
```

and derive:

p_null = unique_k / draw_space

The derived rate may be recorded in each KPI event for auditability.

---

# 4. Analyzer v2 review

## Status: CONDITIONAL APPROVAL

The main-path analyzer result is correct and matches the delivered findings JSON.
Team Beta directly verified that it:

* compiles;
* reads policy values;
* fails when no policy or explicit values are supplied;
* rejects zero pool size;
* rejects zero draw space;
* rejects `max_misses=0`;
* rejects an invalid false-alarm horizon;
* rejects an assumed hit rate outside `[0,1]`;
* reproduces `1.0204` draws to the first collapse fire at the null;
* reproduces `5.3146` draws to the first five-miss run at the null.

The twelve original S176 corrections are substantially implemented.

Four additional edge corrections are required.

## Analyzer fix A — validate `pool_size <= draw_space`

The current tool accepts:

```text
pool_size=1001
draw_space=1000
```

This produces an invalid null probability greater than one and nonsensical run arithmetic.

Add:

```python
if pool_size > draw_space:
    raise SystemExit("[FATAL] pool_size cannot exceed draw_space.")
```

## Analyzer fix B — correct the zero-hit-rate degenerate case

For an assumed hit rate of zero:

```text
p_miss = 1
```

The first run of `m` misses occurs deterministically at draw `m`.

The current function returns infinity.

Correct behavior:

```python
if q == 1:
    return float(m)
if q == 0:
    return float("inf")
```

## Analyzer fix C — prohibit non-standard JSON infinities

Python's default `json.dump()` may emit:

```text
Infinity
```

which is not strict JSON.

Use strict output:

```python
json.dump(result, f, indent=2, allow_nan=False)
```

Represent infinite/unavailable expectations as `null` plus an explanatory status.

## Analyzer fix D — do not call a null event a false alarm

The uniform random null is not the measured healthy TFM baseline.

Therefore:

```text
expected_draws_to_first_false_fire_at_null
```

should be renamed:

```text
expected_draws_to_first_fire_at_uniform_null
```

A "false alarm" can only be defined relative to an accepted healthy operating distribution or explicit null-governance policy.

The existing verdict:

```text
FIRES-WITHIN-HORIZON-AT-NULL
```

is acceptable.

Also add validation that:

```text
collapse_threshold is finite and within the supported rate range
max_misses is an exact integer
```

and record the analyzed source commit in the findings.

After these changes, Analyzer v2 is approved for retention as a read-only governance-analysis tool.

---

# 5. Required Alpha resubmission

Team Alpha should return:

## Proposal v1.1

Containing:

1. Correct enforcement hook in `chapter_13_orchestrator.py`.
2. Defense-in-depth gate at approval/execution.
3. Structured trigger-candidate list before priority selection.
4. Complete mapping for every current action-producing trigger.
5. Fail-closed state/enforcement validation.
6. Canonical source contract for Hit@20/100/300.
7. Split between hard invariants and soft structural KPIs.
8. One idempotent KPI-event persistence contract.
9. Derived null-rate logic.
10. Exact implementation files and test plan.

## Analyzer v2.1

Containing:

1. `pool_size <= draw_space` validation.
2. Correct `p_hit=0` miss-run result.
3. Strict JSON output.
4. "Fire at null" wording instead of "false fire."
5. Threshold/integer validation.
6. Source commit in output.

No policy edits or runtime implementation should begin before Team Beta reviews those revisions.

---

# 6. Final Team Beta disposition

> **S177 successfully resolves the earlier D1/D2 uncertainty, confirms the orphaned synthetic `minimum_hit_rate`, substantially corrects the deterministic analyzer, and proposes the correct three-stage governance direction. The architecture is approved in principle. Implementation is not yet approved because audit-only enforcement must be applied in the Chapter 13 orchestrator before LLM/approval/request creation, trigger candidates must be governance-filtered before priority selection, all live trigger types must be covered fail-closed, and independent Hit@20/100/300 metrics require a canonical full-pool source that the current single 20-pool diagnostic path does not provide. Alpha should submit Proposal v1.1 and Analyzer v2.1 with the corrections above before modifying `watcher_policies.json` or runtime code.**
