# TB Binding Rulings — S172 Phase 4 (dataset_sha256 enforcement + L7 abort discharge)
**Session:** S172 Phase 4 (coordinator implementation)
**Recorded by:** Team Alpha
**Date:** 2026-07-18
**Status:** BINDING. Rulings issued by Team Beta in response to
`TB_RULING_REQUEST_BLOCKER6_DATASET_SHA_S172_PHASE4.md` and
`TB_RULING_REQUEST_L7_ABORT_DISCARD_S172_PHASE4.md`.
**Effect:** Unblocks the Stage 0 resolver patch; fixes the Phase 5 abort interface
for Stage 4. Implemented in `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE4.md`
(Stage 0 + Stage 4).

---

## Ruling 1 — Worker-side `dataset_sha256` enforcement

**Decision: Option C — reject on absence AND update the Phase 3 harness.**
(Team Alpha had recommended Option A / compare-when-present; **overridden**.)

`dataset_sha256` is mandatory at both ends of the assignment contract:
- Phase 4 must include it in every `StripeAssignMessage.payload`.
- The worker must reject an assignment that omits it.
- The worker must reject an assignment whose value differs from the locally
  computed dataset hash.

Compare-when-present was rejected: it would permit a coordinator regression to
silently bypass the identity check. The worker is the final authority immediately
before GPU execution and must **fail closed** rather than assume the producer met
its obligation.

The required-field and equality checks must run **before any cache return or
residue loading**. Required behavior:
```python
expected_dataset_sha = payload.get("dataset_sha256")
if not expected_dataset_sha:
    raise ResidueResolutionError(
        "assignment payload missing mandatory dataset_sha256"
    )

actual_dataset_sha = self._file_hasher(dataset)
if actual_dataset_sha != expected_dataset_sha:
    raise ResidueVerificationError(
        f"dataset_sha256 mismatch: payload={expected_dataset_sha}, "
        f"computed={actual_dataset_sha}"
    )
```
Plain `!=` — the hash is an integrity identifier, not a secret; no constant-time
compare needed.

**Exception/retry semantics:**
- Missing → `ResidueResolutionError`.
- Mismatch → `ResidueVerificationError` (Team Alpha's proposed mismatch path
  **confirmed**).
- Both inherit `ResidueError`; both produce `StripeErrorMessage(retryable=False)`
  through the worker's existing routing.

**Phase 3 harness modification AUTHORIZED AND REQUIRED** — not a relitigation of
Phase 3. "Non-regression" means all *valid* Phase 3 behavior still passes; old
fixtures emitting now-invalid payloads (no `dataset_sha256`) must be updated.
Required test changes to `tests/test_s172_phase3_worker.py`:
- Existing valid resolver fixtures compute + supply `dataset_sha256`.
- Add a missing-`dataset_sha256` test → fails non-retryably.
- Add a mismatched-`dataset_sha256` test → fails non-retryably.
- Add a cache-safety test proving a cached window cannot bypass a later hash
  mismatch.
- Preserve existing different-window and `residue_sha256` coverage.

Narrowly limited to `dataset_sha256`; `residue_sha256` remains mandatory on every
production assignment per the approved brief.

---

## Ruling 2 — L7 trial-abort discharge

**Decision: Option A — synchronous `abort_trial()`.** (Matches Team Alpha's
recommendation.)

Approved interface (unchanged from L6):
```python
class Phase5Sink:
    def publish_shard(self, manifest: ShardReadyManifest) -> None: ...
    def commit_trial(self, event: TrialCommit) -> None: ...
    def abort_trial(self, event: TrialAbort) -> None: ...
```
A successful return from `phase5_sink.abort_trial(event)` **guarantees** Phase 5
has stopped/completed all reads for that trial, holds no reference to any
trial-owned staged path, has discarded every provisional shard + partial assembly
state, and will reject/harmlessly ignore later stale manifests. Only after that
return may Phase 4 delete remaining Zeus-local staged files, release byte + file
reservations, and complete cleanup bookkeeping.

**No asynchronous acknowledgement protocol.** Phase 4 must NOT add `TrialAbortAck`,
an abort callback, a second pending-abort protocol, or a separate ack-timeout
state machine. The async variant is not approved for Phase 4.

**Dispatch-thread requirement.** Synchronous contract, but the call must NOT
execute inside the socket receive/dispatch loop. Route through the coordinator's
lifecycle / cleanup executor:
```
terminal failure detected
→ persist trial state = aborted
→ fence all active assignments
→ schedule synchronous Phase5Sink.abort_trial()
→ wait for successful completion
→ remove staged files
→ release reservations
```

**Failure/timeout.** If `abort_trial()` raises, times out, or the coordinator
exits mid-cleanup: the trial stays terminally aborted; `TrialCommit` stays
permanently prohibited; staged files + reservations are **retained** (never
deleted merely because abort delivery was attempted); cleanup status becomes
`pending`/`failed`; the sync abort call is retried idempotently. Use the
configured `staging_timeout` as the initial bound unless a separately configurable
`phase5_abort_timeout` is introduced.

**Idempotency** by immutable terminal event identity `(event_id, run_id)`:
repeated calls for the same abort event return successfully after confirming no
Phase 5 references remain.

**Gate 34** tests: local file exists while the synchronous stub executes → stub
drains and returns → only then does Phase 4 delete the file + release its
reservation.

L7 invariant confirmed: Phase 4 must never delete a staged file while Phase 5 may
still be reading it.

---

## Final decisions (summary)

| Request | Binding ruling |
|---|---|
| Missing `dataset_sha256` | Reject non-retryably at the worker (`ResidueResolutionError`) |
| Phase 3 harness changes | Authorized and required |
| Dataset hash mismatch | `ResidueVerificationError`, non-retryable |
| Trial-abort discharge | Synchronous `abort_trial()` |
| Async `TrialAbortAck` | Not approved for Phase 4 |
