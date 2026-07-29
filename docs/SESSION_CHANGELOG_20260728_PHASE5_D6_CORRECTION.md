# SESSION CHANGELOG — 2026-07-28 — S172 Phase 5 D6 CORRECTION PASS

**Scope:** the correction pass Team Beta required before D6 can advance. D6 was
HELD on one correctness blocker; the architecture was approved. This session did
NOT rework D6 — the approved D6 build stayed in place and only the blocker, the
two supporting contracts Beta named, the residue asymmetry, and two documentation
signposts were touched.

**Authority:** `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D6_CORRECTION.md` (REV1),
which encodes Beta's D6 disposition and Beta's response to the correction plan.
Both binding.

**Base:** HEAD `a22216c` (docs-only commits sitting on the D6 code freeze point
`2a6e0f8`), D6 changes uncommitted in the working tree. Nothing was committed or
pushed from the agent sandbox; WATCHER was not run.

---

## 1. The blocker, and the fix

`build_stripe_assign_payload` emitted **no threshold field**. The worker therefore
ran `coerce_threshold(payload.get("min_match_threshold", None), 0.25)` and filtered
at a hardcoded `0.25`, while Optuna swept `forward_threshold` / `reverse_threshold`
as real per-trial, per-direction hyperparameters. The optimizer was certifying
results for a configuration other than the one it requested. The 3.B smoke masked
it by running `--threshold` at its `0.25` default — the same value the broken path
fell back to.

**The fix is one canonical path.** The parent resolves the directional threshold
per stripe and the worker never chooses:

```
WindowConfig.forward_threshold / reverse_threshold
  -> parent direction resolution (§6.8 phase table)
  -> canonical stripe payload
  -> worker
  -> executor
  -> kernel
```

* `build_stripe_assign_payload` now takes `phase`, `forward_threshold` and
  `reverse_threshold` as **required keyword-only arguments with no defaults**, so
  it is structurally impossible to build a D6 payload with no threshold. Direction
  comes from `workflow_phase_semantics` (phases 1,3 → forward; 2,4 → reverse),
  which already fails closed on an unknown phase.
* `_dispatch_pending` threads the trial's requested thresholds from the persisted
  trial context into that one chokepoint. No getattr default, no `or 0.25`.
* The worker's `0.25` fallback **survives only for legacy pre-D6 payloads** that
  carry no threshold field at all. A newly generated D6 payload that relies on it
  is a defect, and G8 catches it.

### 1.1 Intentional internal API break (Beta §6)

`build_stripe_assign_payload`'s `phase` / `forward_threshold` /
`reverse_threshold` are **required keyword-only parameters with no defaults, and
this is an intentional internal API break** — recorded as such rather than
smoothed over.

The rationale is the defect itself: **silent omission was the original bug**, so
omission must now **fail loudly**. A default — any default — would reintroduce
exactly the failure mode this pass exists to close, just one layer further in.
Calling the builder without a threshold now raises `TypeError` at the call site
instead of producing a payload that quietly filters at 0.25.

* The **two in-tree callers were updated**: `_dispatch_pending` (production) and
  the Phase-4 coordinator gate's direct-construction fixture.
* **Any out-of-tree caller must be updated, not accommodated.** Restoring a
  default to keep such a caller working would let it produce semantically
  ambiguous work — results certified against a threshold nobody requested. A
  loud break at the call site is the correct outcome.

## 2. `min_match_threshold` / `phase2_threshold` — the explicit contract

Beta required that these two fields have defined, tested meanings, read from what
the hybrid executor **actually needs** rather than inferred from the field names.
Read at source:

* constant-skip stripes (phases 1,2) run `_constant_prefix`, whose only threshold
  scalar is `ctx.threshold`, sourced from `min_match_threshold`;
* hybrid/variable stripes (phases 3,4) run `_hybrid_prefix`, whose only threshold
  scalar is `ctx.hybrid_threshold`, sourced from `phase2_threshold` when present.
  **`ctx.threshold` is never passed to a hybrid kernel.**

So each stripe launches exactly **one** kernel with exactly **one** threshold
scalar, and which payload key feeds it is decided by **skip mode, not by stage**.
`phase2_threshold` is a legacy PWC name for "the hybrid kernel's single threshold",
not a second pass over the data. Emitting both keys with the same resolved
directional value is therefore the only assignment under which the tuned
per-direction threshold reaches the kernel in all four §6.8 phases.

**The invariant is asserted, not assumed:** the payload builder raises if the pair
disagrees, and the worker re-validates on receipt and raises the new
`ThresholdContractError` (non-retryable — a contradictory pair is a contract
violation, not a transient fault). The full contract is documented in the payload
builder's docstring.

## 3. Effective-threshold provenance

`WindowConfig` alone is not evidence a value reached execution. Three legs are now
recorded separately and never reconciled by assumption:

| leg | source |
|---|---|
| `requested` | `WindowConfig.forward/reverse_threshold`, via the trial context |
| `payload` | what `build_stripe_assign_payload` actually emitted |
| `effective` | what the kernel actually filtered at, reported back by the worker |

* `SubStripeOutcome.effective_threshold` is set from the very local the kernel arg
  was built from (`hybrid_threshold` for hybrid, `threshold` for constant) — not
  recomputed from the payload, which would only prove the payload agrees with
  itself.
* It rides `SubStripeResultMessage.effective_threshold` and
  `StripeCompleteMessage.effective_threshold` — **additive, defaulted** envelope
  fields, so the Phase-2 framing contract is intact. It deliberately does **not**
  ride the canonical `payload_bytes`, so the spool/inline byte schema and its
  sha256 are untouched (D5 contract intact).

### 3.1 Protocol compatibility — the exact claim (Beta §4)

The compatibility claim is deliberately narrow, and the narrow form is the only
one supported by what was tested:

> **The change is backward-decode compatible: D6 code can decode pre-D6 messages
> whose new provenance fields are absent.**

**No claim of mixed-version wire compatibility is made.** That would require
demonstrating that pre-D6 *consumers* accept and ignore the extra fields emitted
by D6 peers. That was **not tested** and is **not part of the acceptance
contract**. Nothing in this pass should be read as clearing a mixed-version
fleet.

**Schema-level optionality does not weaken runtime enforcement (Beta's
requirement — now met).** Optional-with-default is a *decode* accommodation for
legacy messages, not a licence to treat a missing value as fine. Per Beta's
commit ruling this is stated as accomplished fact, not as status: **the D6 parent
fails closed when a required effective threshold is absent, when effective ≠
assigned, when a stripe's sub-stripes disagree, or when the constant/hybrid
equality contract is violated.**

**Legacy absence is representable, never accepted as proof of correct D6
execution.**

### 3.1.1 THE LEGACY-CLASSIFICATION RULE (locked, verbatim)

> **"Absence of a generation marker is not sufficient to classify work as legacy.
> Only an explicitly recognized legacy marker may relax D6 provenance
> requirements."**

This rule is binding and is the reason the D6 gate cannot be satisfied by a
silent omission. Its practical consequence, stated without hedging: **for current
D6 work, ALL THREE of the following fail closed —**

| missing thing | outcome |
|---|---|
| missing `d6_generated` | **fails closed.** The flag defaults to `False` only in the get-or-create skeleton and is set `True` explicitly by `record_assignment_threshold` at dispatch. A record that reached the validator without ever being marked is a record whose origin is unknown — and unknown is not legacy. It is never silently treated as relaxable. |
| missing assignment record | **fails closed.** `validate_threshold_provenance` raises when a run has no assignment records at all, and a per-stripe record with no assigned threshold is itself a violation. A trial cannot be certified on the basis that nothing was recorded. |
| missing provenance (no effective value reported) | **fails closed.** Absent provenance on a D6-generated assignment is a `ThresholdProvenanceError`, not a legacy case. |

The distinction is explicit in code, not implicit: every assignment record carries
a `d6_generated` flag set by `record_assignment_threshold`, and
`build_stripe_assign_payload` cannot emit a payload without a resolved threshold,
so in the current serve path **every** assignment is D6-generated. The flag exists
so that any future explicitly-recognised legacy work has a defined, visible place
to be marked — never a silent default, and never inferred from absence.

**Where this rule is enforced and proved:** `validate_threshold_provenance`
(`miner/range_miner_coordinator.py`) enforces it; the gate
`tests/test_s172_phase5_d6_threshold_path.py` proves it — **G10** covers both the
per-sub-stripe missing value and the wholly-absent case ("NO sub-stripe reported
an effective threshold"), **G13** covers the downstream refusal where an absent
`validated` flag is rejected exactly like a `False`, and the **M-prov-missing**
mutant proves G10 actually bites. G10's docstring already carries the substance of
this rule ("Optional schema fields make legacy absence representable; they do NOT
make provenance optional for a D6 run"); see the note in §11 about quoting the
locked sentence there verbatim.

| # | enforced at runtime | where |
|---|---|---|
| 1 effective threshold absent | **fails closed** — `ThresholdProvenanceError`, trial aborted, never committed | `validate_threshold_provenance` |
| 2 effective ≠ assigned | **fails closed** — per sub-stripe *and* for the stripe-complete roll-up | `validate_threshold_provenance` |
| 3 sub-stripes disagree | **fails closed** — plus the worker still refuses to report a single value for a disagreeing stripe, so disagreement cannot masquerade as agreement even before the parent sees it | `validate_threshold_provenance`, `MinerWorker.handle_stripe` |
| 3b stripe-complete ≠ sub-stripe consensus | **fails closed** | `validate_threshold_provenance` |
| 4 constant/hybrid equality violated | **fails closed** (unchanged) — `MinerMetadataError` at construction; `ThresholdContractError` on worker receipt, routed **non-retryable** | `build_stripe_assign_payload`, `SieveExecutor.execute` |

**Placement.** The gate runs inside `serve_trial` **immediately before
`commit_trial`** — therefore before Phase-5 assembly, before candidate ingress,
before accumulator mutation and before `finalize_run`. A violating trial can never
reach certification. A second, independent wall sits at the ingress boundary:
`_build_test_result_from_miner` refuses any `miner_result` whose
`threshold_provenance.validated` is not `True`, **and an absent flag is refused
exactly like a `False`** — absence means the evidence was never checked, which is
not a neutral state.

**Primary-exception discipline (D5 REV2 §7).** On violation the parent logs the
violation, persists the provenance record *first* (so the evidence survives),
then aborts the trial — and **re-raises the original `ThresholdProvenanceError`
unchanged**. An abort or cleanup failure is caught and logged separately and can
never replace or obscure the primary diagnostic; nothing is chained onto it. The
error lists **every** violation found, so the first does not mask the rest.

`ThresholdProvenanceError` is a dedicated exception; `ThresholdContractError`
remains the worker-side constant/hybrid contract violation.
* A stripe whose sub-stripes disagreed reports **no** single effective value and
  logs the disagreement, rather than averaging or silently picking the first.
* `RangeMinerCoordinator.threshold_provenance()` returns the three-leg audit
  record; `serve_trial` returns it and writes `threshold_provenance.json` beside
  the staged output, so "what did the kernel actually filter at?" is answerable
  from the run's own artifacts. It carries a `validated` flag set **only** by the
  parent's fail-closed gate.
* **One registry is the single source of truth** for both the audit record and
  the enforcement gate (`_assignment_provenance`, keyed `run_id ->
  (stripe_id, attempt)`), so what is *reported* and what is *enforced* cannot
  drift apart. Superseded retry attempts are excluded: holding a trial to results
  the retry matrix already discarded would fail a run that actually recovered.

## 4. Residue asymmetry — **FIXED via shared authority** (not guarded)

The preferred route completed cleanly; the pre-dispatch eligibility guard fallback
was **not** needed.

`_get_residues_for_config` never passed `sessions`, while the worker rebuilt its
window **with** the session filter applied. For a both-sessions trial the filter is
a no-op and the two agreed by luck; for a single-session trial (`['midday']` /
`['evening']`) they diverged, the coordinator stamped a `residue_sha256` the worker
could not reproduce, and **every stripe failed the Blocker-6 residue check
non-retryably**.

The fix is one canonical derivation function, `load_residue_window`, whose inputs
include the session selection. Both sides consume it: the worker through
`ResidueResolver`'s default loader, and the coordinator side through the new
`_miner_residues_for_config` at the `use_range_miner` call site. **The session
filter is not duplicated.** `_get_residues_for_config` is untouched, so the PWC and
ZMQ call sites keep their existing derivation byte-for-byte (out of scope).

## 5. Documentation signposts

* **Item 1 (added):** the ⚠️ BLOCKED-BY tripwire under Part B of
  `docs/TODO_SELFPLAY_AND_LLM_AUTONOMY.md`, plus a status note recording that the
  D6 fix is a **precondition, not a green light** — Part B must still route through
  the single chokepoint and must audit `recommended` / `approved-applied` /
  `effective`, never recording an adaptation as applied unless the effective
  execution value matches.
* **Item 2 (added):** the invariant NOTE at `build_stripe_assign_payload` stating
  every threshold source must pass through this chokepoint. It is a comment —
  documentation, not enforcement.
* **Item 3 (DROPPED per Beta ruling):** `watcher_policies.json` is **untouched**.
  Beta rejected the ad-hoc `_parameter_application_note` (an unvalidated field
  either breaks strict policy parsing or becomes ignored metadata, and a note does
  not make `"parameter_application": true` truthful). **The discrepancy — that
  `parameter_application: true` is advisory-only in reality
  (`diagnostics_analysis_schema.py:76`) — is recorded in the autonomy TODO and
  this changelog only**, flagged for the dedicated Part-B implementation to resolve
  properly.

## 6. D5 writer-freeze exception (Beta §7.7 / 4A), recorded verbatim

> "D6 introduces one approved post-D5 extension to `AssemblingPhase5Sink`: an
> optional assembly-backend seam whose `None` path is the exact pre-D6 behavior."

The writer is no longer described as unconditionally frozen.

## 7. Explicitly NOT touched

* `_flush_npz_incremental` — remains **D6.1**, a separate high-priority repair. Not
  opportunistically fixed here (it would shift flush cadence). The existing
  G-FLUSH-CADENCE gate still pins current behaviour and is green.
* PWC / ZMQ ingress, the D3.25 four-map contract, `TestResult` shape — unchanged.
* `serial_reference` stays the default; `process_sharded` stays unpromoted.
* `persistent/pwc_protocol.py` — untouched.

## 8. Files changed this pass

| file | change |
|---|---|
| `miner/range_miner_coordinator.py` | threshold chokepoint + direction resolution + contract assert + provenance registry, `ThresholdProvenanceError`, `validate_threshold_provenance`, the pre-commit fail-closed gate, provenance record/write |
| `miner/range_miner_worker.py` | `ThresholdContractError`, contract re-validation, effective-threshold capture + reporting, canonical `load_residue_window` |
| `miner/range_miner_protocol.py` | two additive defaulted `effective_threshold` fields |
| `window_optimizer_integration_final.py` | `_miner_residues_for_config` (shared authority) wired at the miner call site; the threshold-provenance ingress wall ahead of candidate ingress |
| `tests/test_s172_phase5_d6_threshold_path.py` | **NEW** — the correction's acceptance harness |
| `tests/smoke_s172_phase5_d6_zeus_single_gpu.py` | asymmetric `--forward-threshold` / `--reverse-threshold`, per-direction survivor counts, provenance verification |
| `tests/test_s172_phase4_coordinator.py` | gate-22 allowlist extended; `_FakeWorker` now reports `effective_threshold` |
| `tests/test_s172_phase5_d1_workflow.py`, `tests/test_s172_phase5_d2_directional_uniqueness.py` | their serve-path worker stubs now report `effective_threshold`, mirroring the real worker |
| `tests/test_s172_phase5_d6_production_adapter.py` | `_committed_run` fixture carries the provenance record `serve_trial` now returns |
| `tests/test_s172_phase5_d3_25_candidate_ingress.py` | its hand-transcribed `serve_trial` key oracle gains the eighth key `threshold_provenance` — the oracle did its job and flagged the drift |
| `docs/TODO_SELFPLAY_AND_LLM_AUTONOMY.md` | signpost item 1 + Part-B status note |

## 9. Results

**New gate — `tests/test_s172_phase5_d6_threshold_path.py`: 17/17 green, 11 mutant
kills.** Beta's nine threshold checks at asymmetric `forward=0.31 / reverse=0.47`
(G1–G9), the five parent-side enforcement checks (G10–G13), and the four residue
session cases (R1–R3 identical ordered residues for both / midday-only /
evening-only, R4 assignment round-trip). G4 and G9 drive the **real**
`SieveExecutor` against the real 3080 Ti with real compiled kernels, capturing at
the single kernel entry — they fail loudly rather than skip if cupy is absent.
G10–G12b drive the **real** validator on **real** dispatched assignments,
perturbing exactly one condition each.

Mutants, all four-part-rule proved:

| mutant | killed by |
|---|---|
| **M-drop** (payload emits no threshold field) | G8 (`AssertionError`: payload has no `min_match_threshold`), and independently G1 |
| **M-collapse** (`forward_threshold` applied to both directions) | G2 (`phase 2 carries [0.31…], expected 0.47`), and independently G5 |
| **M-swap** (forward↔reverse exchanged) | **G6 only.** G5 is explicitly asserted to *survive* this mutant — two consistently-reversed branches still look asymmetric, which is exactly why the swap needed its own detector |
| **M-prov-missing** (absent effective value accepted) | G10 — the parent returned instead of raising, so a trial with no physical evidence would commit |
| **M-prov-mismatch** (assigned/effective mismatch accepted) | G11 — a stripe that filtered at 0.25 while assigned 0.47 would be certified |
| **M-prov-disagree** — a **diagnostic-specific enforcement mutant** | G12. Classified precisely: it removes the **cross-sub-stripe disagreement detector and its diagnostic**, *not* all abort paths. The assigned-vs-effective check still aborts the trial, so this mutant does not open a certification hole; what it destroys is the ability to *identify* that a stripe filtered at two different thresholds — the condition's own detector and its named diagnostic. G12 kills it by requiring the disagreement to be named. Reported this way deliberately rather than as a "disagreement accepted" kill, which would overstate it |
| **M-prov-nogate** (the `validate_threshold_provenance` call removed from `serve_trial`) | G13 — enforcement absent entirely |
| **M-residue** (coordinator side drops the session filter) | R1–R3 (`midday only`: coordinator `[100, 700, 101]` vs worker `[100, 101, 102]`), and independently R4 (`residue_sha256` mismatch) |

**Non-regression (exact):**

| suite | result |
|---|---|
| Phase 4 coordinator | **63/63** green |
| D1.1 engine | **18/18** green |
| D4 serial backend | **8/8** green |
| D5 process_sharded | **24/24** green (18 mutants) |
| D6 production adapter | **9/9** green (16 mutants) — includes G-FLUSH-CADENCE, still pinning pre-D6 flush behaviour |

Test edits required by the enforcement, each a fixture made **more** faithful to
production, never a weakened assertion:

* the three serve-path worker stubs (`test_s172_phase4_coordinator`,
  `..._d1_workflow`, `..._d2_directional_uniqueness`) now echo the assignment's
  resolved threshold as `effective_threshold`, exactly as the real worker does.
  A stub that omitted it was modelling a worker that violates the D6 contract,
  and the parent correctly refused it.
* `test_s172_phase5_d6_production_adapter`'s `_committed_run` fixture now carries
  the provenance record `serve_trial` returns. A committed serve result always
  has `validated: True` — the fixture could not otherwise represent a result
  production can produce. The refusal path itself is proved by G13, not weakened
  here.
* `tests/test_s172_phase4_coordinator.py`'s gate-22 coexistence allowlist, which
  reds on any changed `.py` outside its declared set.

PWC/ZMQ/`pwc_protocol` remain unmodified.

### DEFERRED — fragility-hardening item, explicitly NOT folded into this commit

> **"Stage isolation currently depends on the caller-maintained `dispatched` set
> plus the stage barrier; `_dispatch_pending` itself iterates every claimed
> stripe."**

Recorded as a **later fragility-hardening item**. It is **not** a defect in this
commit and **no production change was made for it**.

How it surfaced: while building the enforcement fixture I hit it directly.
`_dispatch_pending` iterates *every* CLAIMED stripe in the run, not only the
current stage's, so stage isolation rests on two things outside the function — the
caller's `dispatched` set and the stage barrier (which only advances once a
stage's stripes are all DONE). Production is correct because `serve_trial` keeps
one `dispatched` set for the whole serve loop and never advances a stage early. My
first fixture passed a *fresh* set per phase and therefore silently re-dispatched
phases 1–3 under phase 4's threshold; G9 and G12 caught it. The fixture now
mirrors production and additionally asserts every dispatched message carries the
expected phase.

Why deferred rather than fixed here: making `_dispatch_pending` self-isolating
(filtering by the stage's phase itself) is a control-flow change to the dispatch
path, outside the scope Beta authorised for this correction, and this commit's
threshold work does not depend on it. It belongs in a dedicated pass with its own
gates — folding it in now would mix an unrequested behavioural change into a
correctness fix that is otherwise fully proved.

**Real-silicon smoke — `tests/smoke_s172_phase5_d6_zeus_single_gpu.py`,
asymmetric `forward=0.31 / reverse=0.47`**:

> **These are the D6 SMOKE-TEST defaults only** — the smoke's `--forward-threshold`
> / `--reverse-threshold` flags replaced its old single `--threshold 0.25`, which
> could not distinguish a working path from the fallback (0.25 *is* the fallback).
> **No production default was changed.** The production backward-compatible
> default remains **0.25**, and it applies to exactly one case: a genuinely legacy
> payload that omits the threshold fields entirely. New D6 payloads always carry
> explicit, direction-resolved values, so the production default is never the
> value a D6 trial filters at. The production `WindowConfig` / optimizer threshold
> defaults were **not** intentionally changed and were **not** changed in fact —
> `git diff` shows no edit to any `forward_threshold` / `reverse_threshold`
> default in `window_optimizer_integration_final.py` or `window_optimizer.py`.

* seeds `[0, 8,000,000)`, window 3, stripe 4,000,000, substripe cap 1,000,000,
  real `miner/range_miner_worker.py` process on the passed-through RTX 3080 Ti,
  trial returned in 23.3 s.
* **Survivor counts by direction:** forward (phase 1, threshold 0.31) =
  **398,156**; reverse (phase 2, threshold 0.47) = **383**; bidirectional
  intersection = **319**. The split is itself physical evidence the two
  directions filtered at different values — a collapsed or fallback run cannot
  produce it.
* **Three provenance values, from the trial's own audit record:**
  `requested` = forward 0.31 / reverse 0.47; `payload` = `{phase 1: [0.31],
  phase 2: [0.47]}`; `effective` = `{phase 1: [0.31], phase 2: [0.47]}`.
  **requested == payload == effective in both directions.**
* **`validated: true`** in `threshold_provenance.json` — set only by the parent's
  fail-closed gate, immediately before `commit_trial`. On real silicon the four
  conditions were therefore **enforced**, not merely recorded. The smoke now
  asserts this rather than printing it. Re-run after the enforcement landed
  (it touches the parent's commit sequence): **identical results** — 398,156 /
  383 / 319, 24.3 s.
* Certified generation produced; 22-array bundle validated in the frozen order;
  sidecar schema `s172.d3_5.provenance.v1.1`; final rows 319;
  **Step-2 load-back `format=npz`, `npz_version=3`, `count=319`,
  `fallback_used=False`.**
* Snapshot-repo commit `3643ee65`, `tree_clean=True` (the harness snapshot
  described in the smoke's header note 3, not a project commit).

## 10. Fallback parity

`fallback parity: code=[not re-checked this pass], env=[not re-checked this pass]`
— the correction pass ran entirely on VM 101; `.127` was not booted, so the
two-pass review (§5 of CLAUDE.md) was not performed. No dependency changed this
pass, so no new env capture was required.

## 11. Note on the gate cross-reference for the locked legacy rule

Beta's item (1) asked for the legacy-classification rule to be locked verbatim in
the changelog **and referenced in the gate**, in an instruction that also
specified *documentation only — no code, no gates* and required confirming that
**no `.py` changed**. Writing the verbatim sentence into
`tests/test_s172_phase5_d6_threshold_path.py` would have modified a `.py`, so the
hard constraint was honoured and the rule is cross-referenced here instead: §3.1.1
names the enforcing function (`validate_threshold_provenance`), the proving gates
(**G10**, **G13**) and the proving mutant (**M-prov-missing**).

The gate already carries the substance — G10's docstring reads *"Optional schema
fields make legacy absence representable; they do NOT make provenance optional for
a D6 run"* — so nothing about the rule is undocumented at the gate. If Beta wants
the locked sentence quoted **verbatim** in G10's docstring, that is a one-line
docstring addition; it is deliberately deferred rather than done silently, because
it would change a `.py` in a commit authorised as changelog-only. It can be
applied and re-proved docs-only (AST identity with docstrings stripped) on
request.

---

**Do not commit from the sandbox.** Team Alpha review, then Team Beta. After both
pass: Michael commits D6 + this correction, then runs the release-grade smoke from
the clean real repository and records the commit-linked certified generation.
