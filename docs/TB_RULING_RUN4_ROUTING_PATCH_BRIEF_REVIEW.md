# TB RULING — RUN-4 ROUTE-A IMPLEMENTATION BRIEF REVIEW

**Received:** 2026-08-27 · **Recorded by:** Team Alpha (verbatim; delivered via ser8 → VM101) · **Status:** BINDING
**Reviews:** `docs/S172_RUN4_ROUTING_PATCH_BRIEF_ROUTE_A.md` (427 lines, sha256 `a56433b3390f4831ceaa7893cb0fba9b54ca56b2413f464a7d1dc245e9a7b9e4`)

## Dispositions

| Item | Disposition |
|---|---|
| Route-A brief architecture (literal seven-key allowlist, no manifest defaults, explicit operator provenance, all-seven-or-none at WATCHER, eighth key permanently stripped, no opportunistic wiring) | **APPROVED** |
| Brief §3.1–§3.4 (four source-derived corrections to Beta's earlier implementation shape) | **RATIFIED**, all four |
| **BLOCKER 1** — frozen operator bundle ownership | **REVISION REQUIRED** — authority is **pipeline-invocation-local**: captured defensively before any retry/LLM mutation, owned by the invocation context, threaded explicitly to every Step-1 `run_step()` of that invocation, discarded at invocation end. NOT `self._step1_pin_bundle`, NOT module global, NOT daemon/persisted state. Scope MAY expand (e.g. `run_pipeline()`) if the call graph requires threading — do not hide the lifetime problem to keep a smaller diff. Gate 10 tests the stronger shape: same `WatcherAgent` instance → pinned pipeline completes → second unpinned pipeline → zero warm-start routing, zero pin provenance |
| **BLOCKER 2** — G-UNPINNED-IDENTICAL oracle | **REVISION REQUIRED** — the pinned-executable design violates **EXEC-PIN-1** (pinned `run_step` in the live namespace can consume the post-patch `_step1_declared_params` and false-green). Preferred (strongly): capture the deterministic unpinned command control from **clean `69ca910` BEFORE the first edit** — record base commit, deterministic fixture inputs, exact argv list, artifact hash, completion proof; post-patch gate compares list-equal against that capture. Fallback only: full EXEC-PIN-1 compliance — pin the entire resolved-name dependency closure, never `run_step` alone |
| G-EXACT | **TIGHTENED** — value **and type** equality, not Python object identity (`is`) |
| M2 | **SPLIT REQUIRED** — **M2a**: invocation-local authority becomes process/agent-lifetime authority; **M2b**: pins contaminate ordinary/default `final_params` and appear on a subsequent unpinned invocation. Both detected while the pinned path stays green. M1 and M3 mandatory as written |
| Provenance marker | **APPROVED** — absence on unpinned runs preferable to null/empty; the Run-4 acceptance record retains the actual generated argv alongside the marker (marker proves authority; argv proves what it requested); no decision logic consumes it |
| Scope rule (replaces "two definitions") | Minimum definitions necessary to capture the operator bundle once, thread it invocation-locally, open WALL 1/WALL 2 only under that authority, and maintain declaration parity |
| Still prohibited | manifest `default_params` · optimizer/integration changes · direct threshold wiring · `search_strategy` · `seed_count` · search-bounds changes · six-of-seven downstream repair · L-1 · ingress byte-bound work |
| Run-4 geometry and claim boundary | **UNCHANGED** — routing patch certifies routing and containment only; Run 4 remains the later production proof for volume/plumbing/completion, never population equivalence |
| Process | Once the four bounded edits are incorporated, **Alpha may proceed directly to the Route-A patch** — no further conceptual routing ruling required. Beta reviews the patch itself per the original cycle |

## Required bounded edits (from DISPOSITION)

1. Specify and test true pipeline-invocation-local ownership/threading of operator pin authority; expand the changed-definition scope if the call graph requires it.
2. Replace the contaminated pinned-executable G-UNPINNED-IDENTICAL oracle with a pre-edit `69ca910` clean control, or fully comply with EXEC-PIN-1.
3. Define G-EXACT as value/type equality, not object identity.
4. Split ambiguous M2 into concrete persistence mutants (M2a, M2b).

---

## RULING BODY (VERBATIM)

## TEAM BETA REVIEW — RUN-4 ROUTE-A IMPLEMENTATION BRIEF

**VERDICT: ARCHITECTURE APPROVED; TWO BLOCKING CORRECTIONS REQUIRED BEFORE IMPLEMENTATION. NO ROUTING REDESIGN REQUIRED.**

The brief has the right containment model. Alpha's four source-derived corrections to my prior implementation guidance are accepted: `_step1_declared_params` is not the command-building wall; the internal-only set actually has eight members; `args_map` membership cannot safely define routability; and the downstream six-of-seven check cannot enforce the all-seven-or-none contract required for Run 4. 

The chosen architecture is also the right one: **literal seven-key allowlist, no manifest defaults, explicit operator provenance, all-seven-or-none at WATCHER, eighth key permanently stripped, and no opportunistic wiring of the hazardous threshold/search/seed declarations.** 

Two issues must be fixed in the brief before Claude implements it.

### 1. BLOCKER — the frozen operator bundle must have an explicit invocation-local owner

The design says the operator's `--params` is captured once into an immutable per-invocation record, then legitimate retry/replay may continue using it. That is correct. But the brief does not yet establish **where that record lives and how every `run_step()` invocation receives it**. 

This is load-bearing because WATCHER is long-lived in daemon mode; instance state can survive between pipelines. The existing architecture explicitly separates daemon lifetime from individual pipeline lifetime. 

Binding rule:

> **The explicit-pin authority belongs to one pipeline invocation, not to the `WatcherAgent` object, the daemon, module state, `retry_params`, or persisted state.**

Therefore:

* capture a defensive immutable copy of the original seven operator values **before** any retry/LLM mutation;
* own it in the invocation-local pipeline context;
* explicitly pass that same frozen record to every Step-1 `run_step()` call belonging to that invocation;
* discard it when that pipeline invocation ends;
* a later pipeline on the same `WatcherAgent` begins with no pin authority unless its own operator input supplies all seven again.

**Do not implement this as `self._step1_pin_bundle` or a module global.**

If the actual call graph requires modifying `run_pipeline()` or another caller to thread that record, **that definition is in scope.** The present "two definitions" scope must expand rather than hiding the lifetime problem to preserve a smaller diff.

Gate 10 should test the stronger shape:

**same `WatcherAgent` instance → pinned pipeline invocation completes → second unpinned pipeline invocation → zero warm-start routing and zero pin provenance.**

That proves invocation isolation rather than merely two calls to `run_step()`.

### 2. BLOCKER — `G-UNPINNED-IDENTICAL` currently violates EXEC-PIN-1

The proposed gate executes `run_step` source pinned at `69ca910` to produce the pre-patch command oracle. 

But this project already established the **pinned-executable-source hazard**: old executable source evaluated against live helpers is not an old-code oracle. That became standing rule EXEC-PIN-1 after Brief I. 

Here the risk is concrete, not theoretical:

* pinned `run_step` calls `_ensure_execution_set`;
* `_ensure_execution_set` calls `_step1_declared_params`;
* this patch itself proposes changing `_step1_declared_params`.

So if the pinned `69ca910` `run_step` is executed using the live patched module namespace, its supposedly pre-patch control can consume the **post-patch helper**. The exact gate intended to prove "unpinned is unchanged" can therefore false-green.

Because implementation has **not started yet**, use the stronger solution:

> **Capture the deterministic unpinned command control from clean `69ca910` before the first edit.**

Record:

* base commit;
* deterministic fixture inputs;
* exact generated argv/list;
* hash of the captured artifact;
* completion proof.

After the patch, `G-UNPINNED-IDENTICAL` compares the patched result list-equal against that pre-edit captured result.

That avoids executing historical code against a live namespace entirely.

If Alpha instead insists on reconstructing the old executable later, EXEC-PIN-1 applies fully: derive the pinned function's resolved-name dependency closure and pin every changed helper it reaches. Merely pinning `run_step` is **not acceptable**.

I strongly prefer the clean pre-edit control while `69ca910` is still untouched.

### 3. The four corrections to Beta's earlier implementation shape are RATIFIED

**§3.1:** accepted. The actual two command walls in `run_step` must be changed; updating only `_step1_declared_params` would not route anything. 

**§3.2:** accepted. `warm_start_session` is the eighth internal-only key and stays unconditionally internal. It must never acquire the underscore-to-hyphen fallback route.

**§3.3:** accepted. The exact seven-name literal `frozenset` is superior to deriving authorization from `args_map`. The latter would accidentally authorize the two threshold overrides plus `search_strategy` and `seed_count`.

**§3.4:** accepted. All-seven-or-none must be enforced at WATCHER. The downstream optimizer's six-key test plus `session_idx=0` default is a real latent hazard, but its repair remains outside this Run-4 patch. Record it under WATCHER-MANIFEST-ROUTING-1.

### 4. Acceptance gate set — APPROVED with two tightenings

The proposed gate structure is strong: 126 partial subsets, explicit eighth-key check, LLM/retry synthesis resistance, positive provenance evidence, exact allowlist proof, and optimizer one-trial verification. 

Two wording/implementation corrections:

For **G-EXACT**, "identity of value" must mean **value and type equality**, not Python object identity (`is`). Floats and integers must be compared semantically, with `session_idx == 1`, thresholds exactly surviving the intended conversion, etc.

For **M2**, choose a concrete mutant. The current description—

> "make the frozen record process-global / carry the bundle into defaults"

—describes two different defects. 

Split it:

* **M2a:** invocation-local authority becomes process/agent-lifetime authority;
* **M2b:** explicit pins contaminate ordinary/default `final_params` and appear on a subsequent unpinned invocation.

Both should be detected while the original pinned path remains green. That is exactly the dangerous failure shape: Run 4 works, but S167 containment has silently been destroyed.

M1 and M3 remain mandatory as written.

### 5. Provenance marker — APPROVED

`step1_pin_source=explicit_operator_warm_start` is acceptable, and its **absence** on unpinned execution is preferable to a null/empty placeholder. 

For the eventual Run-4 acceptance record, retain the actual generated command containing all seven values alongside the marker. The marker proves authority; the argv proves what that authority actually requested.

No decision logic may consume the provenance marker.

### 6. Scope after correction

Authorized executable surface remains narrowly WATCHER-only, but do not artificially hold it to two changed functions if invocation-local threading requires another caller.

The proper scope rule is:

> **minimum definitions necessary to capture the original operator bundle once, thread it invocation-locally, open WALL 1/WALL 2 only under that authority, and maintain declaration parity.**

Still prohibited:

* manifest `default_params`;
* optimizer/integration changes;
* direct threshold wiring;
* `search_strategy`;
* `seed_count`;
* search-bounds changes;
* six-of-seven downstream repair;
* L-1;
* ingress byte-bound work.

### 7. Run-4 geometry remains approved

Nothing in this review changes the previously approved pinned geometry or claim boundary.

The routing patch certifies **routing and containment only**. Run 4 remains the later production proof for volume/plumbing/completion, never population equivalence. 

## DISPOSITION

**Route-A brief:** **APPROVED IN ARCHITECTURE, REVISION REQUIRED BEFORE IMPLEMENTATION.**

Required bounded edits:

1. specify and test true **pipeline-invocation-local** ownership/threading of operator pin authority; expand the changed-definition scope if the call graph requires it;
2. replace the contaminated pinned-executable `G-UNPINNED-IDENTICAL` oracle with a **pre-edit `69ca910` clean control**, or fully comply with EXEC-PIN-1;
3. define G-EXACT as value/type equality, not object identity;
4. split ambiguous M2 into concrete persistence mutants.

No other redesign requested. Once those are incorporated, Alpha may proceed directly with the Route-A patch; another conceptual routing ruling is not required. 
