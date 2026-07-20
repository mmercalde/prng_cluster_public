# Team Alpha Review Record — S172 Phase 4 (RANGE-MINER coordinator)
**Reviewer:** Team Alpha (lead dev)
**Date:** 2026-07-18
**Verdict:** PASS — ready for Team Beta binding code review.
**Method:** file-vs-source. Every claim below was read from the actual delivered
files (not the implementer's summaries) and diffed against the clean clone at
`6661b04`. Where a prior Beta round caught a defect by extrapolation, that exact
area was read line-by-line.

---

## Scope reviewed

| File | Change | Lines |
|---|---|---|
| `miner/range_miner_worker.py` | Blocker-6 `ResidueResolver` patch (Stage 0) | +19 in `resolve()`, diff otherwise nil |
| `tests/test_s172_phase3_worker.py` | Phase-3 harness 14 → 17 gates | 3 new gates + fixtures supply `dataset_sha256` |
| `miner/range_miner_coordinator.py` | 63-line stub → full coordinator | 2141 lines |
| `tests/test_s172_phase4_coordinator.py` | New — 36 brief gates + 1 non-regression | 37 `_check` calls |

Harness result: **36/36 brief gates green, exit 0** (the 37th `_check` is the
Phase-0/1/2/3 subprocess non-regression). CPU-only, loopback, stubbed transfer +
stubbed `Phase5Sink`.

---

## Binding rulings — verified implemented to the letter

**Ruling 1 (Blocker 6, Option C).** `resolve()` performs, BEFORE the cache return
(`if key in self._cache`) and BEFORE `self._loader(...)`: absent `dataset_sha256`
→ `ResidueResolutionError`; mismatch vs locally-computed hash →
`ResidueVerificationError`. Reuses the existing cache-key hash (file hashed once).
Plain `!=`. Both inherit `ResidueError`; routing to `stripe_error(retryable=False)`
verified unchanged at `range_miner_worker.py:1152-1154`. Worker diff is confined
entirely to `resolve()` — nothing else in 1346 lines moved. Phase-3 harness: valid
fixtures updated to supply `dataset_sha256`; gate 15 (missing→non-retryable), gate
16 (mismatch→non-retryable), gate 17 (warm cache cannot bypass a later mismatch —
asserts the loader is NOT re-called, proving the check precedes the cache return).

**Ruling 2 (L7, Option A).** `abort_trial()` order: persist aborted → fence/cancel
active stripes → **synchronous** `Phase5Sink.abort_trial()` → only on successful
return delete staged files + release reservations. On raise/timeout: files +
reservations RETAINED (no deletion in the except branch), `cleanup_status='failed'`,
idempotent by `(event_id, run_id)` via `{run_id}:abort` + first-guard. Off the
dispatch thread via `submit_abort()` → single-worker cleanup executor. No
`TrialAbortAck`/async path added. Gate 34: consumed shard's file exists DURING the
sync stub, deleted only AFTER return. Gate 28: retain-on-failure + idempotent retry.

---

## Per-stage source findings

**Stage 1 — ledger/state machine/L8.** Shard PK `(run_id, stripe_id, attempt,
sub_index)` — the many-shards-per-stripe cardinality (the prior Beta-caught trap) is
correct; `stripes` separately keyed `(run_id, stripe_id)`. L8 predicate checks all
five invariants independently, counts `distinct sub_index` (not row count), guards
empty-shard vacuity. `_coverage_exact` cursor-walk rejects both gap and overlap.
`reclaim_expired_leases` bound to `state='claimed'` → staging never compute-reclaimed;
generation bumped on fence. `expected_substripes` via imported `select_seed_cap`.
`MAX_ATTEMPTS` appears only in a "do not reuse" comment.

**Stage 2 — identity/caps/fencing.** `_validate_caps` iterates ALL FOUR caps
(amd/nvidia/amd_hybrid/nvidia_hybrid — the hybrids were the prior omission); explicit
`bool` guard so `True` can't pass as an int cap; missing/non-positive/mismatch →
quarantine (durable, not dropped, not silently defaulted). `can_assign_variant`
requires the exact concrete variant; quarantined refused; `assign_stripes` leaves an
ineligible worker's stripe pending. `accept_stripe_message` enforces the full L1
conjunction (worker_id, claimed_by, current_attempt==recorded, state-permits); every
reject is read-only. Spool-root guard uses the sibling-prefix-safe
`startswith(root+sep)` idiom with `normpath` `..`-collapse. Gate 24 proves attempt-1
ledger byte-unchanged after a stale attempt-0 message and attempt-0's shard stays
keyed under attempt 0.

**Stage 3 — staging/reservations/L5 fence.** Inline normalization IMPORTS
`build_substripe_payload_bytes` (byte-identity, not a re-mirror); gate 13 asserts
`raw == pb`. Reservations summed globally over `status='held'`, both marks enforced,
back-pressure returns None. The L5 fence sits in `_finalize_stage` after materialize
but before any ledger update/enqueue/delete_remote; stale → own file removed, own
reservation released, publishes nothing; gate 32 drives the actual generation-bump
race. Verify-before-delete: mismatch fails before `_delete_remote` (gate 15).
`_fail_and_release` enforces remove-file-FIRST-then-release. Gate 36 asserts zero
reservation leak across all six pathologies. SC1 `retry_remote_delete` idempotent,
no shard duplication; gate 31 attempt-count (1→2→3, no-op stays 3) verified correct.

**Stage 4 — retry matrix/lifecycle/Phase5Sink.** `handle_stripe_failure` implements
all five matrix rows exactly (non-retryable→fail; constant 1/2→fail-closed; hybrid
first→reassign to a DIFFERENT variant-capable worker + `phase_degraded`; hybrid
second→fail; lease-expiry→same policy). `_pick_other_worker` requires
`can_assign_variant`. Blocker-2 publish correctly relocated: `_finalize_stage` no
longer publishes; `finalize_stripe` calls `publish_attempt` only on full L8, then
`done`; `cleanup_attempt` invalidates a failed attempt's shards (publishes nothing).
`ack_by_event_id` keyed solely by event_id, releases exactly once, A-never-touches-B.
Payload builder makes `dataset_sha256` + `residue_sha256` mandatory (closes the
Ruling-1 both-ends loop).

**Stage 5 — integration/coexistence/non-regression.** Gate 20 is a real end-to-end:
live `RangeMinerWorker` daemon over loopback + real Stage-0-patched `ResidueResolver`;
wrong `dataset_sha256` → `stripe_error(retryable=False)` → identity gate → matrix row
4 → trial aborted, sink got one abort. Gate 21 asserts the coordinator module
namespace is clean of Phase-5 assembly (no numpy, no `range_miner_npz_writer`, no
`EXPECTED_NPZ_KEYS`/`assemble_arrays`/`run_contract_wall`) — inspection, not just
grep. Gate 22: `git status` shows only the four deliverables changed; PWC + ZMQ
import unmodified with their trial entrypoints; `run_trial_miner` builds a real
coordinator and plumbs all L4 config. Gate 23: Phase 0/1/2/3 harnesses re-run as
subprocesses, all exit 0, including the 17-gate patched Phase-3.

---

## Deviations (all reviewed, all legitimate, all in-scope)

1. **Blocker-2 publish relocation** — a correction toward the brief, not away from
   it. Attempt-scoped publish at completion is exactly Blocker 2; the earlier
   per-shard enqueue was the weaker model. Stage-3 guarantees re-verified intact
   under the new publish point.
2. **`run_trial_miner` live serve loop is integration-time** — the coordinator
   object + all lifecycle logic is complete and gate-validated; only the live
   socket-server loop over the fleet is stubbed via injectable `_serve`. This is the
   CT100-key / Phase-6-7 boundary, explicitly out of Phase-4 scope. No-`_serve` path
   raises `NotImplementedError` preserving the Phase-1 stub contract.
3. **`remote_delete_attempts` counts the successful attempt** — defensible attempt
   count; idempotent no-op retry does not increment; gate 31 asserts it; no shard
   duplication.

---

## Open items for Michael before commit (non-blocking)

- **Version-string drift:** worker docstring reads "rev-2" and the Phase-3 harness
  header "rev-3" — cosmetic, bump to match actual state so Beta doesn't flag a stale
  marker.
- **Changelog count wording:** "36/36 gates" vs harness's 37 `_check` calls — the
  37th is the non-regression check; a one-line clarification avoids a question.

## Standing

This is a Team **Alpha** pass — file-vs-source verification that the implementation
matches the brief + both binding rulings. It is NOT the binding gate. Sequence from
here: **Team Beta binding code review → Michael commits + dual-pushes.** The four code
deliverables + the changelog are uncommitted on VM 101 as written.
