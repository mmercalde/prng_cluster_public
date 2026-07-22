# Team Beta Review — S178 WATCHER KPI Governance Proposal v1.1 and Analyzer v2.1

## Verification basis

Team Beta reviewed:

* `PROPOSAL_WATCHER_KPI_GOVERNANCE_STATES_v1_1.md`
* `watcher_kpi_metricC_deterministic_v2_1.py`

The architectural claims were cross-checked against the current live GitHub `main` branch. The repository head remains:

```text
0c3166a630be321809f415bb28af28e319d0fe1b
```

The analyzer was compiled and executed through its normal path, invalid-input matrix, degenerate probability cases, threshold edge cases, strict-JSON path, and provenance path.

---

# 1. Executive ruling

## Proposal v1.1

**Status: ARCHITECTURE APPROVED IN PRINCIPLE — FOUR MANDATORY AMENDMENTS BEFORE IMPLEMENTATION**

Proposal v1.1 successfully resolves most of Team Beta's S177 blockers. In particular, it correctly moves the primary governance gate into the Chapter 13 orchestrator, introduces governance filtering before trigger prioritization, expands the trigger registry, separates hard invariants from soft metrics, defines a canonical ranked-prediction source, derives null rates, and selects one canonical KPI ledger.

A complete rewrite is not required.

Team Alpha should issue a short v1.2 addendum covering the four remaining issues below. Runtime implementation may begin after those amendments are incorporated into the implementation contract.

## Analyzer v2.1

**Status: REVISION REQUIRED — TWO REPRODUCIBLE LOGIC/PROVENANCE DEFECTS**

The primary null calculations and requested input validations are correct. However, the optional assumed-rate sensitivity calculation can contradict the configured trigger behavior, and the recorded source commit is derived from the process working directory rather than an explicitly identified source repository.

Analyzer v2.2 is required before retention as an authoritative governance artifact.

---

# 2. Proposal corrections accepted from v1.1

## 2.1 Correct primary enforcement location

Approved.

The primary gate belongs in `Chapter13Orchestrator.run_cycle()` immediately after trigger evaluation and before:

* LLM analysis;
* proposal validation;
* approval-request generation;
* WATCHER routing.

This now covers both the orchestrator and WATCHER approval routes.

## 2.2 Governance filtering before priority selection

Approved.

The full candidate set must survive evaluation. Governance filtering must occur before selecting the highest-priority active trigger.

The proposed structure:

```json
{
  "candidates": [],
  "selected_active_trigger": null,
  "audit_only_triggers": []
}
```

is appropriate.

## 2.3 Fail-closed unmapped-trigger behavior

Approved.

Any trigger without a governance mapping must resolve to:

```text
enforcement = audit_only
dispatch = prohibited
configuration warning = emitted
```

It must never fall through to the current legacy action path.

## 2.4 Canonical Hit@K source

Approved in direction.

Team Beta agrees with Option A: preserve the full ranked output from Step 6 rather than deriving Hit@100 and Hit@300 from the current truncated 20-entry pool.

The live generator currently ranks all survivors, then truncates the ranked indexes to `pool_size` before producing the saved prediction result.

Therefore, an additional ranked artifact is required before Hit@100 and Hit@300 become real measurements.

## 2.5 Hard-invariant versus soft-quality separation

Approved.

The proposal now correctly distinguishes:

* active hard invariants with verified runtime consumers;
* invariants that remain implementation gaps;
* structural-quality metrics requiring empirical calibration;
* prospective outcome metrics;
* cadence and regime triggers.

## 2.6 Derived null rates

Approved.

The correct contract is:

p_null = unique_K / draw_space

rather than hardcoded 0.02, 0.10, and 0.30 values.

---

# 3. Mandatory proposal amendment 1 — SELFPLAY requires its own governance gate

The primary orchestrator gate covers the normal `evaluate_triggers()` cycle, but the SELFPLAY request path is separate.

The live `request_selfplay()` method directly creates a pending JSON request under `watcher_requests/`, marks it as requiring WATCHER approval, and records it in trigger history.

Merely placing `SELFPLAY_RECOMMENDED` in the registry does not guarantee that its request-creation path is governed.

## Required amendment

Add explicit enforcement at both locations:

### Before SELFPLAY request creation

`request_selfplay()` or its caller must resolve the SELFPLAY governance entry before writing an actionable request.

For `audit_only`:

```text
record hypothetical SELFPLAY recommendation
do not create a pending watcher request
dispatched = false
approval_requested = false
```

For `shadow`:

```text
record candidate and review metadata
do not create an executable request
```

For `active`:

```text
create the normal WATCHER-authorized request
```

### At the WATCHER SELFPLAY consumer

WATCHER must re-read current governance state before authorizing or executing an existing SELFPLAY request.

This provides the same stale-request protection proposed for normal retraining approvals.

---

# 4. Mandatory proposal amendment 2 — define `shadow` and validate per-metric state

The proposal defines `shadow` as an allowed enforcement value in CALIBRATING, but the control-flow section only defines:

* `audit_only`;
* `active`.

The runtime behavior of `shadow` is not specified.

## Required shadow contract

A shadow trigger should:

```text
evaluate a candidate calibrated threshold
record whether it would have fired
record its hypothetical action
count false alarms, overlap and recovery
optionally generate a non-executable human-review artifact
never create an executable approval request
never dispatch pipeline work
```

The distinction should be:

```text
audit_only:
    collect raw metric and legacy-trigger observations;
    no candidate calibrated policy is asserted.

shadow:
    evaluate a specific candidate policy;
    record what that candidate would have done;
    still prohibit execution.
```

## State consistency

The current allowed matrix validates `global_state` against `enforcement`, but it must also validate each metric's own `state`.

For example, this must be rejected:

```json
{
  "global_state": "GOVERNED",
  "metric": {
    "state": "BOOTSTRAP",
    "enforcement": "active"
  }
}
```

Required rules:

```text
metric state BOOTSTRAP   → audit_only only
metric state CALIBRATING → audit_only or shadow
metric state GOVERNED    → audit_only, shadow or active
```

A per-metric state must not exceed the global lifecycle state:

```text
global BOOTSTRAP   → no metric may be CALIBRATING or GOVERNED
global CALIBRATING → no metric may be GOVERNED
global GOVERNED    → individual metrics may remain in any lower state
```

Any inconsistent combination must fail closed to `audit_only`.

---

# 5. Mandatory proposal amendment 3 — make the ranked Hit@K contract deterministic

The current proposal says ties are broken by rank index. The live source does not establish that guarantee.

It currently uses:

```python
ranked_idx = np.argsort(predicted_quality)[::-1]
```

The default NumPy sort is not a sufficient documented deterministic tie-breaking contract for governance artifacts.

The duplicate-collapse rule also needs to be defined operationally rather than merely stored as a descriptive field.

## Required ranking contract

Use an explicitly deterministic ordering, such as:

```text
primary key: predicted_quality descending
secondary key: stable survivor identifier ascending
tertiary key: original source index ascending
```

Persist both:

```text
raw_rank
unique_rank
```

For duplicate predicted outcome values:

```text
first occurrence in the deterministic ranked order wins
later occurrences of the same outcome are retained only as provenance,
not assigned another unique rank
```

The generator must continue through the raw ranked survivor list until it has collected the requested number of unique outcome values, rather than truncating the first 300 raw survivor rows and assuming they represent 300 unique predictions.

## Insufficient unique values

When fewer than K unique outcomes are available:

```text
requested_k = 300
unique_k = 217
hit300_available = false
hit300 = null
```

Do not label a 217-entry pool as Hit@300.

A separate observation may record:

```text
hit_at_available_k
available_k = 217
null_rate = 217 / draw_space
```

but that is not the same KPI as Hit@300.

---

# 6. Mandatory proposal amendment 4 — correct the KPI ledger lifecycle

The proposal's selection of one append-only canonical ledger is sound, but three details require correction.

## 6.1 `PIPE_BUF` does not govern regular-file append atomicity

The statement that POSIX guarantees an `O_APPEND` write below `PIPE_BUF` will not interleave is not the correct basis for a regular JSONL file. `PIPE_BUF` applies to pipes and FIFOs.

The ledger should rely on:

```text
one cooperating lock protocol
fcntl.flock(LOCK_EX)
O_APPEND
one encoded write while holding the lock
flush and fsync before releasing the lock when durable audit persistence is required
```

All writers must use the same lock.

The torn-final-line recovery rule remains appropriate.

## 6.2 The idempotency key is incomplete

This key is insufficient:

```text
draw_id + prediction_artifact_fingerprint
```

The same predictions may be re-evaluated after:

* a governance-policy change;
* a source-code change;
* a trigger-schema change;
* a state transition;
* a corrected diagnostic implementation.

Suppressing that second evaluation would lose important audit history.

Use an evaluation identity containing at least:

```text
draw_id
prediction_artifact_fingerprint
governance_policy_fingerprint
evaluator_schema_version
source_commit
```

An exact match is idempotent.

A change to any component creates a new evaluation revision.

## 6.3 Approval and execution happen after initial evaluation

A single completed row cannot always contain final values for:

```text
approval_requested
approved
dispatched
execution_result
```

because human approval may occur later.

Use one canonical event ledger, but allow multiple lifecycle events:

```text
KPI_EVALUATED
APPROVAL_REQUESTED
APPROVAL_APPROVED
APPROVAL_REJECTED
DISPATCH_STARTED
DISPATCH_COMPLETED
DISPATCH_FAILED
BLOCKED_BY_GOVERNANCE
```

Each later event references:

```text
evaluation_id
draw_id
request_id
```

This remains one canonical ledger while preserving append-only history.

Do not overwrite the original evaluation event.

---

# 7. Manual execution path — document as a privileged override

The live CLI exposes:

```text
--execute
```

which calls `execute_standalone()` directly.

That method runs the selected pipeline scripts without an approval-request round trip.

This is an explicit human operation rather than an autonomous trigger, so Team Beta does not require it to be removed.

However, the implementation contract must classify it as a privileged override.

Recommended requirements:

```text
explicit --manual-governance-override
mandatory --reason
operator identity
source commit
policy fingerprint
requested steps
ledger event
```

Without the override flag, direct execution should still respect governance state.

This prevents the command from becoming an undocumented bypass during later autonomous operation.

---

# 8. Analyzer v2.1 — confirmed improvements

The following fixes work correctly:

* `pool_size > 0`;
* `draw_space > 0`;
* `pool_size <= draw_space`;
* `max_misses >= 1`;
* exact-integer validation for policy-provided `max_misses`;
* collapse threshold finite and within `[0,1]`;
* positive fire horizon;
* strict JSON without `Infinity`;
* hit rate zero produces a five-miss run at draw five;
* hit rate one produces `null` plus `infinite_never_fires`;
* primary null outputs remain approximately 1.0204 and 5.3146 draws.

The source implementation reflects the requested v2.1 corrections.

---

# 9. Analyzer defect 1 — assumed-rate sensitivity ignores threshold shape

The primary analyzer correctly recognizes three possible collapse-trigger shapes:

```text
threshold = 0:
    fires on neither hit nor miss

0 < threshold <= 1/pool_size:
    fires on misses only

threshold > 1/pool_size:
    fires on every draw
```

However, the optional `assumed_healthy_sensitivity` block always calculates:

```python
1 / p_miss
```

as though the collapse trigger fires on misses only.

Team Beta reproduced the contradiction:

```text
collapse_threshold = 0
primary shape       = does not fire
assumed-rate wait   = 1.25 draws
```

and:

```text
collapse_threshold = 1
primary shape       = fires on every draw
assumed-rate wait   = 1.25 draws
```

For the second case, the correct wait is one draw. For the first case, the trigger never fires.

## Required v2.2 correction

Apply the same trigger-shape logic to the assumed-rate block:

```python
if fires_on_hit and fires_on_miss:
    fire_probability = 1.0
elif fires_on_miss and not fires_on_hit:
    fire_probability = 1.0 - assumed_hit_rate
else:
    fire_probability = 0.0
```

Then derive the waiting time from that probability.

---

# 10. Analyzer defect 2 — source provenance follows the working directory

The current function runs:

```python
git rev-parse HEAD
```

without identifying which repository is being analyzed.

Team Beta verified:

* running outside a Git repository records `null`;
* running the analyzer from an unrelated temporary Git repository records that unrelated repository's commit as `analyzed_source_commit`.

Therefore, the field does not reliably prove which TFM tree was analyzed.

## Required v2.2 correction

Add an explicit argument:

```text
--repo-root /path/to/prng_cluster_public
```

Resolve provenance with:

```text
git -C <repo-root> rev-parse HEAD
git -C <repo-root> status --porcelain
```

Record:

```text
analyzed_repo_root
analyzed_source_commit
analyzed_tree_dirty
policy_file_path
policy_file_sha256
analyzer_file_sha256
```

For an authoritative run, failure to resolve the repository commit should be fatal rather than silently writing `null`.

Also reject a Boolean `collapse_threshold`; the current `float(True)` path accepts it as `1.0`.

---

# 11. Required Alpha deliverables

## Short proposal v1.2 addendum

No full rewrite is necessary. The addendum should specify:

1. SELFPLAY request and execution governance gates.
2. Exact `shadow` behavior.
3. Per-metric state consistency rules.
4. Deterministic stable ranking and duplicate collapse.
5. Hit@K unavailable behavior when fewer than K unique outputs exist.
6. Correct JSONL locking/durability wording.
7. Expanded evaluation identity and lifecycle event model.
8. Privileged manual-execution override policy.

## Analyzer v2.2

Required fixes:

1. Apply threshold shape to assumed-rate sensitivity.
2. Resolve source provenance from explicit `--repo-root`.
3. Record dirty state and file fingerprints.
4. Fail when authoritative source provenance cannot be established.
5. Reject Boolean collapse thresholds.

---

# 12. Final Team Beta disposition

> **Proposal v1.1 resolves the central S177 control-flow and metric-source blockers and is approved as the governing architectural direction. Implementation remains paused only for a short v1.2 addendum covering the separate SELFPLAY request path, precise shadow/state semantics, deterministic unique Hit@K ranking, and the append-only ledger lifecycle/idempotency contract. Analyzer v2.1 passes its primary null and validation matrix but is not yet authoritative because its assumed-rate sensitivity ignores the configured threshold shape and its source commit is taken from the current working directory rather than an explicitly selected TFM repository. Submit the focused v1.2 addendum and Analyzer v2.2; no further broad architectural rewrite is required.**
