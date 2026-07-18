# TB Ruling Request — L7 trial-abort discard: sync `abort_trial()` vs async `TrialAbortAck`
**Session:** S172 Phase 4 (coordinator implementation)
**Author:** Team Alpha
**Date:** 2026-07-18
**Priority:** P2 — non-blocking. Both options are already brief-compliant; this is
a selection, not a scope change. Lands in Stage 4, so Stages 0–3 proceed
regardless of the ruling.
**Related:** `docs/S172_PHASE4_BRIEF.md` L7 (lines 470–480), L6 `Phase5Sink`
interface (lines 442–468), L3 whole-trial abort (lines 341–353), gate 34 (lines
510–512); `docs/PROPOSAL_S172_RANGE_MINER_v1_4_5.md` §3.A (Phase 4/5 ownership
boundary).

---

## Question

L7 requires resolving the abort-discard race — Phase 5 may be mid-read of a
provisional local shard path when a `TrialAbort` deletes it — and **explicitly
offers a choice**:

> Require ONE of:
> - **Async:** `TrialAbort → Phase 5 stops/finishes reads + discards provisional
>   state → TrialAbortAck → Phase 4 deletes remaining local files + releases
>   reservations`; OR
> - **Sync:** an `abort_trial()` call whose successful return **guarantees** Phase
>   5 no longer references any trial-owned path.

Both satisfy the binding invariant (L7, verbatim): *"a local file backing an
unacked, actively-consumed shard MUST remain until the discard acknowledgement.
Phase 4 does not delete out from under Phase 5."* We ask Beta to select which
variant Phase 4 implements, since the choice fixes the `Phase5Sink.abort_trial`
contract that Phase 5 must later honor.

---

## Background — code-verified

**Phase 5 does not exist yet.** `range_miner_npz_writer.py` is not in the repo at
`6661b04`; Phase 4 defines the `Phase5Sink` seam (L6) as an injected interface a
future Phase 5 implements, and the Phase-4 harness drives it with a stub. So this
ruling sets the contract Phase 5 inherits — it is cheaper to fix now than after
Phase 5 is built against it.

**The `Phase5Sink` interface (L6, brief lines 447–452).**
```python
class Phase5Sink:
    def publish_shard(self, manifest: "ShardReadyManifest") -> None: ...
    def commit_trial(self, event: "TrialCommit") -> None: ...
    def abort_trial(self, event: "TrialAbort") -> None: ...
```
`abort_trial` is already synchronous in signature (returns `None`, no ack
callback declared). The async variant would additionally require a
`TrialAbortAck` entry point (a fourth sink method or a callback), which L6 does
not currently list.

**Gate 34 (brief lines 510–512) is variant-agnostic in wording:**
> A shard is unacked and actively consumed when TrialAbort fires → its local file
> REMAINS until discard-ack (TrialAbortAck **or** sync abort_trial return); only
> then is it deleted + reservation released.

The gate names both discharge mechanisms, so either implementation is testable
against it. Under sync, the stub `abort_trial` asserts the actively-consumed
shard file still exists **during** the call and is deleted only **after** it
returns. Under async, the stub emits `TrialAbortAck` and the harness asserts the
file survives until the ack fires.

**Interaction with L3 (whole-trial abort).** L3 requires abort to invalidate
every provisional shard of the trial, release all reservations, and be terminal +
idempotent. Under sync, all of that happens inside/after the single
`abort_trial()` return — one code path, no in-flight ack state to reconcile.
Under async, L3's "once safe" cleanup is gated on the ack, so the coordinator
must hold per-trial pending-abort state until `TrialAbortAck` arrives, which
composes with the L5 async-task fencing generation counter but adds a second
in-flight lifecycle.

---

## Options

**Option A — Sync `abort_trial()` (Team Alpha recommended).**
`abort_trial(event)`'s successful return guarantees Phase 5 references no
trial-owned path; Phase 4 then deletes local files + releases reservations. No
new sink method.
- *Pro:* matches the L6 signature as written (no fourth method); single abort
  code path; deterministic, race-free harness for gate 34 (no ack timing);
  simplest composition with L3 terminal/idempotent cleanup.
- *Con:* Phase 5's future `abort_trial` implementation must complete its
  read-drain/discard **before returning** — i.e. it may block the coordinator's
  calling context for the drain duration. Acceptable given abort is a terminal,
  low-frequency path.

**Option B — Async `TrialAbortAck`.**
`TrialAbort` emitted; Phase 5 drains + discards, then calls back
`TrialAbortAck`; Phase 4 then deletes + releases.
- *Pro:* never blocks the coordinator on Phase 5's drain; more symmetric with the
  L2 shard-ack seam (which is already async/event-id-keyed).
- *Con:* requires a `TrialAbortAck` entry point not in the L6 interface list;
  adds a second in-flight lifecycle (pending-abort state) to reconcile with L5
  generation fencing and L3 terminality; more surface for a stuck/never-acked
  abort to leak reservations, needing its own timeout.

---

## What we need ruled

1. **Sync (Option A) or async (Option B)** for L7 discharge.
2. If async: confirm the `Phase5Sink` interface (L6) is extended with a
   `TrialAbortAck` mechanism (method or callback), and that a pending-abort
   timeout is in scope for Phase 4 (to bound a never-acked abort).
3. Confirm the L7 invariant holds either way (Team Alpha reads it as satisfied by
   both; confirming for the record): the actively-consumed local file survives
   until discharge, and Phase 4 never deletes out from under Phase 5.

---

## Recommendation (Team Alpha, non-binding)

**Option A (sync).** It matches the `Phase5Sink` signature exactly as L6 defines
it, needs no interface addition, and gives gate 34 a fully deterministic assertion
(file present during the call, deleted only after return) with no ack-timing race
to stub. It composes most cleanly with L3's terminal/idempotent whole-trial
cleanup — one abort path rather than a two-phase emit/ack lifecycle that would add
its own timeout and pending-state reconciliation against the L5 generation
counter. The only cost — `abort_trial` may block for Phase 5's drain — is
acceptable on a terminal, low-frequency failure path. Async remains a clean future
change if Phase 5's drain proves long enough to warrant non-blocking abort, but we
see no present reason to pay its complexity.
