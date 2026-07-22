# Team Beta Binding Review — S179 Governance Addendum v1.2 and Analyzer v2.2

## Verification basis

Team Beta reviewed:

* `PROPOSAL_WATCHER_KPI_GOVERNANCE_STATES_v1_2_ADDENDUM.md`
* `watcher_kpi_metricC_deterministic_v2_2.py`
* `SESSION_CHANGELOG_20260721_S179.md`
* `watcher_kpi_metricC_v2_2_findings.json`

The relevant control-flow claims were cross-checked against the current live GitHub `main` source at:

```text
0c3166a630be321809f415bb28af28e319d0fe1b
```

The analyzer was independently compiled and executed through its normal path, threshold-shape cases, invalid-policy cases, provenance failures, dirty-tree detection and non-authoritative mode.

---

# 1. Executive ruling

## Governance architecture

**Status: APPROVED FOR IMPLEMENTATION WITH THREE BINDING CODE-LEVEL CONDITIONS**

Proposal v1.1 plus the v1.2 addendum now forms an acceptable architectural contract for:

```text
BOOTSTRAP → CALIBRATING → GOVERNED
```

The v1.2 addendum successfully closes the previously identified gaps concerning:

* SELFPLAY request governance;
* stale-request protection;
* exact shadow-mode behavior;
* per-metric state consistency;
* deterministic ranking;
* duplicate collapse;
* insufficient unique Hit@K depth;
* JSONL locking and durability;
* append-only lifecycle events;
* privileged manual execution.

No further broad proposal rewrite is required.

Three implementation details identified below are binding. They can be incorporated directly into the implementation task and tests; a proposal v1.3 is not required.

## Analyzer v2.2

**Status: APPROVED**

Analyzer v2.2 corrects the v2.1 threshold-shape contradiction and binds provenance to an explicitly selected repository rather than the current working directory.

The delivered findings reproduce the expected live-policy results:

```text
instantaneous collapse:
    expected first fire at uniform null = 1.0204 draws

five consecutive misses:
    expected first run at uniform null = 5.3146 draws
```

The findings also correctly disclose:

```text
source commit = 0c3166a630be321809f415bb28af28e319d0fe1b
tree dirty    = true
authoritative = true
```

---

# 2. Addendum items approved

## 2.1 SELFPLAY pre-creation gate

Approved.

`request_selfplay()` currently creates an actionable pending request file and records it in trigger history. The proposed gate inside that function is therefore correctly positioned to prevent request creation under `audit_only` or `shadow`. The live function writes the request directly under `watcher_requests/` and labels it as requiring WATCHER authorization.

Required behavior remains:

```text
audit_only:
    record hypothetical recommendation
    create no pending request
    approval_requested = false
    dispatched = false

shadow:
    evaluate candidate policy
    create no executable request
    dispatched = false

active:
    create normal WATCHER request
```

## 2.2 SELFPLAY dispatch chokepoint

Approved.

`dispatch_selfplay()` is the correct defense-in-depth execution chokepoint. It performs actual SELFPLAY launch work after only its existing dry-run and halt checks.

The governance recheck must occur before any actual execution, VRAM shutdown or process launch.

## 2.3 Shadow semantics

Approved.

The addendum now makes the necessary distinction:

```text
audit_only = raw observation, no candidate calibrated policy

shadow = evaluate a specific candidate policy and record what it
         would have done, but prohibit execution
```

This distinction must appear in both policy validation and ledger output.

## 2.4 Per-metric state rules

Approved.

Both constraints must be enforced:

```text
metric state controls allowed enforcement

and

metric state cannot exceed global lifecycle state
```

Every inconsistent or unknown combination fails closed to `audit_only`.

## 2.5 Deterministic ranking and duplicate collapse

Approved.

The required ordering is:

```text
predicted quality descending
stable survivor identifier ascending
original source index ascending
```

The current generator ranks predictions and then truncates to `pool_size`, so it does not currently preserve a full deterministic unique ranking.

Implementation must persist:

```text
raw_rank
unique_rank
predicted outcome
quality
stable survivor identifier
source index
duplicate-of reference, when applicable
```

The first occurrence of an outcome wins its `unique_rank`. Later occurrences remain provenance records but do not consume unique-rank positions.

## 2.6 Hit@K unavailability

Approved.

When fewer than K unique predictions exist:

```text
hitK_available = false
hitK = null
```

Do not silently relabel Hit@217 as Hit@300.

## 2.7 Ledger correction

Approved.

The addendum correctly strikes the inappropriate `PIPE_BUF` claim and replaces it with a shared locking protocol:

```text
flock(LOCK_EX)
O_APPEND
one encoded write while holding the lock
flush + fsync when durable audit persistence is required
```

The append-only lifecycle-event model is also approved.

---

# 3. Binding condition 1 — make the request-consumer gate generic and early

The addendum proposes a SELFPLAY-specific check in `process_chapter_13_request()` immediately before the request-type routing branch.

That gate must instead be:

1. **generic across every action-producing request type**, and
2. **performed immediately after loading and identifying the request**, before LLM validation or Strategy Advisor enrichment.

The live consumer supports:

```text
selfplay_retrain
learning_loop
pipeline_rerun
```

It currently loads the request, then invokes LLM validation and optional Strategy Advisor processing before reaching the request-type route.

It then dispatches all three action types:

* SELFPLAY through `dispatch_selfplay()`;
* partial retraining through `dispatch_learning_loop()`;
* full pipeline rerun through `dispatch_learning_loop(scope="full")`.

## Required implementation

Immediately after:

```python
request_type = request.get("request_type")
```

resolve:

```text
governance_key
global_state
metric_state
enforcement
request provenance
manual-override status
```

Then apply:

```text
audit_only:
    archive or retain as BLOCKED_BY_GOVERNANCE
    append ledger event
    do not invoke LLM validation
    do not invoke Strategy Advisor
    do not dispatch

shadow:
    record candidate-policy evaluation
    optionally create a non-executable review artifact
    do not invoke executable dispatch

active:
    continue through normal validation and routing
```

Unknown request types or unmapped governance keys remain fail-closed.

This generic gate prevents a hand-created or stale `learning_loop` or `pipeline_rerun` request from bypassing the governance protections applied to orchestrator-created approvals.

The scanner processes every JSON request in `watcher_requests/`, so this consumer-side gate is necessary even if normal request producers are governed.

---

# 4. Binding condition 2 — guard the learning-loop chokepoint

The v1.2 addendum correctly adds an authoritative recheck to `dispatch_selfplay()`, but the equivalent learning-loop execution chokepoint must also be guarded.

The live `dispatch_learning_loop()` can execute:

```text
Steps 3→5→6
Steps 1→6
arbitrary validated step sequences
```

After its dry-run and halt checks, it directly begins executing pipeline steps.

## Required implementation

At the beginning of `dispatch_learning_loop()`:

```text
resolve action scope:
    partial learning loop
    full pipeline rerun
    custom manual scope

resolve governance or privileged override

refuse execution unless:
    the originating trigger is active/GOVERNED
    or
    a valid privileged manual override is present
```

This is the defense-in-depth counterpart to the generic request-consumer gate.

It protects against:

* direct CLI dispatch;
* stale request files;
* manually constructed request files;
* future callers that bypass `process_chapter_13_request()`.

---

# 5. Binding condition 3 — complete the privileged-override inventory

The addendum identifies:

```text
chapter_13_triggers.py --execute
watcher_agent.py --dispatch-selfplay
```

The live WATCHER CLI exposes two additional direct execution routes:

```text
--dispatch-learning-loop
--run-pipeline
```

`--dispatch-learning-loop` calls `dispatch_learning_loop()` directly, while `--run-pipeline` calls `WatcherAgent.run_pipeline()` directly.

These are explicit human operations, so they do not need to be removed. They must, however, be included in the same privileged-override contract.

## Complete privileged command set

```text
chapter_13_triggers.py --execute
watcher_agent.py --dispatch-selfplay
watcher_agent.py --dispatch-learning-loop
watcher_agent.py --run-pipeline
```

Each requires:

```text
--manual-governance-override
--reason
operator identity
requested action or steps
source commit
working-tree dirty state
policy fingerprint
ledger event
```

Without the override flag:

```text
the command must respect current governance state
```

For the initial real TFM cycle during BOOTSTRAP, Michael may deliberately invoke the pipeline using this explicit manual override. That preserves operator authority while ensuring the first baseline-building run is not mistaken for autonomous KPI dispatch.

Add an explicit ledger event type:

```text
MANUAL_OVERRIDE_REQUESTED
MANUAL_OVERRIDE_EXECUTED
MANUAL_OVERRIDE_FAILED
```

or an equivalent clearly typed event family.

---

# 6. Evaluation-identity strengthening

The addendum defines:

```text
draw_id
prediction artifact fingerprint
governance policy fingerprint
evaluator schema version
source commit
```

This is acceptable for clean committed code.

However, the delivered authoritative analyzer run records:

```text
analyzed_tree_dirty = true
```

Two evaluations can therefore share the same commit while running different uncommitted evaluator code.

## Required implementation detail

Add one of:

```text
evaluator_code_fingerprint
```

or hashes of the relevant governance/evaluator source files to `evaluation_id`.

Recommended identity:

```text
draw_id
prediction_artifact_fingerprint
governance_policy_fingerprint
evaluator_schema_version
evaluator_code_fingerprint
source_commit
```

Also record:

```text
source_tree_dirty
dirty_paths
```

This prevents an exact-idempotency match from suppressing a materially different evaluation performed on the same commit with uncommitted changes.

This is an implementation-level strengthening, not a request for another proposal revision.

---

# 7. Analyzer v2.2 independent verification

Team Beta independently confirmed:

## Normal policy case

```text
pool_size = 20
draw_space = 1000
collapse_threshold = 0.01
max_misses = 5

collapse first fire = 1.0204 draws
five-miss run       = 5.3146 draws
```

## Threshold-shape cases

```text
collapse_threshold = 0
assumed hit rate    = 0.20
result:
    fire probability = 0
    wait = null
    status = infinite_never_fires
```

```text
collapse_threshold = 1
assumed hit rate    = 0.20
result:
    fire probability = 1
    wait = 1 draw
```

These correct the v2.1 contradiction.

## Validation paths

Confirmed fail-loud behavior for:

```text
Boolean collapse threshold
non-integer max_misses
missing authoritative repo root
nonexistent repo root
pool size greater than draw space
invalid threshold range
invalid hit-rate range
invalid horizon
```

## Provenance paths

Confirmed:

```text
valid explicit repo:
    real commit recorded
    authoritative=true

dirty repo:
    analyzed_tree_dirty=true

--no-provenance:
    authoritative=false
    commit=null
```

## Artifact integrity

The SHA-256 of the uploaded Analyzer v2.2 source exactly matches the analyzer hash recorded in the delivered findings:

```text
cf6e1c94cd99a74f2ce1095c5257431ee7163b198122bc470de42e2f31604c9c
```

The S179 changelog accurately records the analyzer changes and test matrix.

---

# 8. Implementation authorization

Team Alpha is authorized to begin the implementation described by:

```text
Proposal v1.1
+
v1.2 Addendum
+
the three binding conditions in this ruling
```

Implementation must remain:

```text
BOOTSTRAP by default
all performance/cadence/regime triggers audit_only
no numerical KPI thresholds selected
no autonomous retraining enabled
hard invariants only where a runtime consumer and blocking behavior are tested
```

Required review sequence:

1. Implement policy schema and fail-closed resolver.
2. Implement trigger-candidate structure.
3. Implement generic request-consumer and dispatch chokepoint gates.
4. Implement ranked unique prediction artifact.
5. Implement independent Hit@K diagnostics.
6. Implement canonical lifecycle ledger.
7. Add privileged override controls.
8. Run CPU-only unit/integration tests.
9. Submit the implementation diff and test results to Team Beta.
10. Do not launch the historical walk-forward until S172 Phase 7 scheduling permits it.

---

# 9. Final Team Beta ruling

> **The S179 v1.2 addendum is accepted and Analyzer v2.2 is approved. Team Alpha may proceed to implementation without another proposal revision. The implementation must apply governance generically to SELFPLAY, learning-loop and full-pipeline request consumers; add defense-in-depth checks to both `dispatch_selfplay()` and `dispatch_learning_loop()`; classify all four direct execution commands as privileged manual overrides; and strengthen event idempotency with an evaluator-code fingerprint when the source tree may be dirty. Subject to those binding conditions, the WATCHER KPI governance architecture is approved for BOOTSTRAP-mode implementation.**
