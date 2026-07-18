# TB Ruling Request — Blocker-6 worker-side `dataset_sha256` comparison: reject-on-absence vs compare-when-present
**Session:** S172 Phase 4 (coordinator implementation)
**Author:** Team Alpha
**Date:** 2026-07-18
**Priority:** P1 — on the critical path; the Phase-3 `ResidueResolver` patch is
Stage 0 of implementation and cannot start until this is ruled.
**Related:** `docs/S172_PHASE4_BRIEF.md` Blocker 6 (lines 153–168) + residue
contract (lines 225–227); `docs/PROPOSAL_S172_RANGE_MINER_v1_4_5.md` §3.A;
`miner/range_miner_worker.py` `ResidueResolver` (`:579-623`), non-retryable routing
(`:1135-1136`); `tests/test_s172_phase3_worker.py` gate 9 (B1 residue window).

---

## Question

Blocker 6 requires patching Phase 3's `ResidueResolver` to compare a
**payload-supplied `dataset_sha256`** against the worker-local file hash and
**fail non-retryably on mismatch**. It also states `dataset_sha256` is
**"mandatory for dataset identity."**

When the coordinator supplies a `dataset_sha256`, the mismatch → non-retryable
behavior is unambiguous and not in question. The narrow question is what the
**worker** does when a payload arrives with **no** `dataset_sha256` key at all:

- **(a) Compare-when-present:** worker compares if the field is present, fails
  non-retryably on mismatch, and tolerates absence (treats "mandatory" as a
  coordinator-side obligation Phase 4 always satisfies); **or**
- **(b) Reject-on-absence:** worker treats a missing `dataset_sha256` as a hard
  non-retryable failure, enforcing mandatoriness at the worker itself.

---

## Background — code-verified

**Current resolver behavior.** `ResidueResolver.resolve()` computes
`dataset_sha = self._file_hasher(dataset)` (`range_miner_worker.py:601`) and uses
it **only** as a cache key — it never compares it to anything payload-supplied.
This is the exact defect Blocker 6 identifies: "Phase 4 would send a field the
worker silently ignores."

**Non-retryable routing already exists.** `ResidueError` subclasses
(`ResidueResolutionError`, `ResidueVerificationError`) already route to
`stripe_error(retryable=False)` at `range_miner_worker.py:1135-1136`. So either
ruling is a small patch; no new exception type or routing change is needed. A
mismatch raising `ResidueVerificationError` is already correct under both (a) and
(b).

**The `residue_sha256` precedent (same file, same method).** The resolver's
existing `residue_sha256` check is **compare-when-present**: `if residue_sha:`
gates the verification, and absence falls through without error
(`range_miner_worker.py:611-619`). Option (a) makes `dataset_sha256` behave
identically to the field already beside it; option (b) makes the two
identity fields behave differently.

**The tension in the brief.** Blocker 6 says `dataset_sha256` is "mandatory," but
the same Blocker's final sentence requires: *"re-run the Phase-3 harness to
confirm non-regression."* Phase-3 gate 9 (B1 residue window) and the fake-loader
fixtures drive `resolve()` **without** a `dataset_sha256` key
(`tests/test_s172_phase3_worker.py`). Under option (b), those calls begin raising
`ResidueVerificationError` and the Phase-3 harness regresses — which the brief
forbids in the same paragraph that introduces the requirement.

**Where "mandatory" is enforced under option (a).** The brief's residue contract
(lines 225–227) states every `StripeAssignMessage.payload` carries
`dataset_sha256` (mandatory) and `residue_sha256` (mandatory), and that
"`run_trial_miner()` already has the exact residue sequence, so the coordinator
can always compute it." Under option (a), Phase 4 asserts both fields present on
every assign (never optional), so a production payload without `dataset_sha256`
cannot occur — mandatoriness holds end-to-end, enforced at the sender.

---

## Options

**Option A — compare-when-present at the worker; mandatoriness enforced
coordinator-side (Team Alpha recommended).**
Worker: `expected = payload.get("dataset_sha256")`; if present and
`!= dataset_sha` → `ResidueVerificationError` (non-retryable); absence tolerated.
Coordinator: asserts `dataset_sha256` present on every `StripeAssignMessage`
(Stage 4 payload contract). Phase-3 harness unchanged, stays 14/14 green.
- *Pro:* satisfies Blocker 6's operative requirement (mismatch fails
  non-retryably); preserves Phase-3 non-regression exactly as the brief demands;
  matches the adjacent `residue_sha256` semantics; single enforcement point.
- *Con:* a hypothetical non-Phase-4 caller could invoke the worker without the
  field and not be caught at the worker (out of current scope — no such caller
  exists; the assignment contract has one producer).

**Option B — reject-on-absence at the worker.**
Worker treats missing `dataset_sha256` as a non-retryable failure.
- *Pro:* mandatoriness enforced at the worker; defends against a future
  out-of-contract caller.
- *Con:* regresses the Phase-3 harness unless gate 9 + fake-loader fixtures are
  updated to supply `dataset_sha256`; the brief requires the Phase-3 patch be
  "surgical" and non-regressing. Would make `dataset_sha256` and `residue_sha256`
  behave differently in the same method for no stated reason.

**Option C — reject-on-absence, and update the Phase-3 harness in the same patch.**
Option B plus editing `tests/test_s172_phase3_worker.py` gate 9 and fixtures so
absence no longer occurs in-test.
- *Pro:* full worker-side enforcement without a "regression" per se.
- *Con:* expands a Phase-4-driven change into edits of the approved Phase-3
  acceptance harness; arguably relitigates Phase-3 gate 9's contract, which Beta
  already approved (rev-3, `dbe3d0e`).

---

## What we need ruled

1. **(a) compare-when-present or (b)/(c) reject-on-absence** at the worker for a
   missing `dataset_sha256`.
2. If (b) or (c): confirm Beta authorizes the corresponding Phase-3 harness edit
   and that it does **not** count as relitigating the approved Phase-3 contract.
3. Confirm the mismatch path — `ResidueVerificationError` (non-retryable) — is
   correct regardless of the above (Team Alpha reads this as settled by Blocker 6;
   confirming for the record).

---

## Recommendation (Team Alpha, non-binding)

**Option A.** It satisfies every operative word of Blocker 6 — a payload-supplied
`dataset_sha256` that mismatches the worker-local file hash fails non-retryably —
while honoring the same Blocker's explicit non-regression requirement and matching
the compare-when-present semantics already used for `residue_sha256` in the same
method. Mandatoriness is not weakened: it is relocated to the single producer
(Phase 4's assign path), which the brief itself says can always compute the field.
The patch stays surgical (one comparison added; cache-key logic, `residue_sha256`
verification, and routing all untouched), and the Phase-3 harness remains 14/14
without edits to Beta-approved acceptance gates.
