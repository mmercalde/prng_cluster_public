# S172 Phase 4 — Coordinator: Implementation Brief (rev-4, post-Beta rev-3 review)

> **rev-4 status:** Team Beta accepted rev-3 ("correctly incorporates every item…
> accepted ownership boundary intact… narrow rev-4 required, not an architectural
> rewrite") pending **four narrow lifecycle details**. rev-4 PRESERVES all of rev-3
> (Blockers 1–7, Decisions A/B, L1–L4, SC1, gates 1–31) and ADDS sections **L5–L8**
> + gates **32–36**. Reconciliation fields verified at source
> (`StripeCompleteMessage.substripes_done`/`survivors_total`, `range_miner_protocol.py:133-134`).

> **rev-3 status:** Team Beta accepted rev-2's architecture ("correctly absorbs the
> seven prior rulings… do not relitigate or rewrite the accepted architecture") and
> rejected pending **four lifecycle additions** + one schema correction. rev-3
> PRESERVES everything below (Blockers 1–7, Decisions A/B, all rev-2 gates) and ADDS
> sections **L1–L4 + schema correction SC1** and new gates **24–31**. Nothing in the
> accepted architecture is changed. Both code-touching additions verified at source
> (`range_miner_protocol.py`: result/complete/error messages carry NO `attempt` — only
> `StripeAssignMessage` does, line 109; `range_miner_coordinator.py:40-41`: only
> `seed_cap_nvidia`/`seed_cap_amd` exposed, no hybrid caps).

**For:** Claude Code on VM 101 as `michael`, `/home/michael/distributed_prng_analysis`.
**Authoritative spec:** `docs/PROPOSAL_S172_RANGE_MINER_v1_4_5.md` (`13c5a1a`, TB-approved)
— read §3.A, §6.7, §12.3, §12.4, §15. **Base:** `zmq_sqlite_coordinator.py`.
**Prereqs:** Phases 0–3 committed (`…dbe3d0e`); confirm `HEAD ≥ 13c5a1a`.

**Status:** rev-1 was REJECTED by Team Beta (7 blockers + 2 binding decisions).
This rev folds all of them in. Beta's rulings are binding — implement, do not
relitigate. The root cause of rev-1's defects: it modeled the coordinator on ZMQ's
**one-job-one-result** shape, but the miner protocol is **one stripe → many
sub-stripe results + one StripeComplete**. Every state-model fix below flows from
correcting that.

Phase 5/6/7 remain out of scope. The coordinator MUST NOT assemble the 22 arrays,
dedup, order, or run the contract wall (v1.4.5 §3.A).

---

## Deliverable
Flesh out `miner/range_miner_coordinator.py` + `tests/test_s172_phase4_coordinator.py`.
CPU-testable on loopback (fake workers, stubbed transfer); no GPU/rig needed to pass.

---

## BLOCKER 1 — shard-level result ledger (not stripe-level)

A worker partitions ONE assigned stripe into MANY sub-stripes (`range_miner_worker.py`
:483,498 — each `SubStripe` has a `sub_index`), emits one `SubStripeResultMessage`
per sub-stripe, THEN one `StripeCompleteMessage`. A one-row-per-stripe table
overwrites sub-stripe records. **Wrong cardinality.**

Replace `job_results` with a **shard-level** table keyed by
`(run_id, stripe_id, attempt, sub_index)`, recording:
```
worker_id, seed_start, seed_count, survivor_count,
remote_spool_path, local_staged_path, size_bytes, sha256,
staging_status, created_at, verified_at
```
A stripe becomes `done` **only after ALL** of:
- `StripeCompleteMessage` arrived;
- every expected `sub_index` exists;
- sub-stripe seed ranges **exactly cover** the assigned stripe (no gap/overlap);
- every shard is locally staged AND hash-verified;
- reported survivor totals reconcile (sum of shard `survivor_count` == StripeComplete total).

---

## BLOCKER 2 — attempt-scoped staging; no partial-attempt leak; trial lifecycle

A worker can emit several good sub-stripe results then fail a later one. rev-1
staged+enqueued each immediately → attempt-0 shards could reach Phase 5 even though
the stripe is retried as attempt 1. **Forbidden.**

Required flow — **publish per attempt, only when whole:**
```
attempt-scoped staging → verify ALL shards → receive StripeComplete
  → validate complete stripe coverage → PUBLISH that attempt's manifests to Phase 5
```
On attempt failure:
- invalidate + remove ALL local shards from that attempt;
- never publish them as committed Phase 5 input;
- retry the complete stripe per policy (Blocker 3).

At **trial** level, Phase 4 emits explicit lifecycle events to Phase 5 (without
these, Phase 5 cannot enforce "no partial dataset"):
```
ShardReadyManifest   # provisional trial input (per verified, complete attempt)
TrialCommit          # all stripes succeeded → provisional input becomes committed
TrialAbort           # terminal failure → Phase 5 discards provisional input
```

---

## BLOCKER 3 — phase-specific retry matrix (not "one retry for all")

rev-1 generalized "one retry per failure." **Wrong.** v1.4.5 §12.3 distinguishes
sieve-workflow phases (the 4-phase test-both-modes flow, §6.8): workflow **phases
1/2 = constant** (fail closed), workflow **phases 3/4 = hybrid** (one retry then
fail). And `StripeErrorMessage.retryable` (`range_miner_protocol.py:147`, default
True; False short-circuits) must short-circuit.

**Binding matrix — implement exactly:**
| Condition | Action |
|---|---|
| Workflow phase 1/2 (constant) failure | **Fail trial immediately** (fail closed) |
| Workflow phase 3/4 (hybrid) retryable **first** failure | Reassign once to another eligible GPU; mark `phase_degraded=True` in agent_metadata |
| Workflow phase 3/4 (hybrid) retryable **second** failure | **Fail trial** |
| Any `retryable=False` failure | **Fail trial immediately** — never consumes the retry |
| Lease expiry | Apply the **same phase-specific** policy above |

No reduced-sample objective, no partial dataset (§12.3). Do NOT inherit ZMQ's
`MAX_ATTEMPTS=3`.

---

## BLOCKER 4 — inline results normalized to Zeus-local file manifests

S175 §6.7.A: Phase 5 processes receive ONLY compact manifests and open Zeus-local
staged files themselves; large parsed objects MUST NOT cross process IPC.
Therefore an inline `SubStripeResultMessage` **must not** be enqueued as an inline
dict. Phase 4 must, for inline results:
- canonically serialize the inline payload (same `s172_substripe_v1` bytes);
- verify its `size_bytes` and `sha256`;
- **atomically write it as a Zeus-local staged shard**;
- enqueue the SAME path-based manifest used for remote spools.

Phase 5 sees ONE uniform input type:
```python
{ "local_spool_path": "...", "expected_size": ..., "expected_sha256": ...,
  "stripe_id": "...", "attempt": ..., "sub_index": ..., "trial_metadata": {...} }
```

---

## BLOCKER 5 — state machine must include async staging

`pending → claimed → done/failed` cannot represent async staging. After
`StripeCompleteMessage` the GPU is free but transfers may still run; if the stripe
stays `claimed`, its compute lease can expire → **incorrect duplicate reassignment.**

Add a real staging state (or equivalent durable fields):
```
pending → claimed → staging → done
                          ↘ failed
```
Rules:
- heartbeats renew leases **only while `claimed`**;
- lease reclaim applies **only to active compute claims** (not `staging`);
- `staging` has its own timeout/failure handling;
- `done` means every shard locally verified AND published;
- **worker completion ≠ ledger completion.**

---

## BLOCKER 6 — enforce residue identity (requires a small Phase 3 patch)

Verified at source: `ResidueResolver` computes `dataset_sha = self._file_hasher(dataset)`
and uses it ONLY as a cache key (`range_miner_worker.py:601`). It never compares a
**payload-supplied `dataset_sha256`** → Phase 4 would send a field the worker
silently ignores.

Binding contract revision:
- `residue_sha256` is **MANDATORY**, not "strongly preferred." `run_trial_miner()`
  already has the exact residue sequence, so the coordinator can always compute it.
- `dataset_sha256` remains **mandatory** for dataset identity.
- **Patch `ResidueResolver`** (Phase 3 file) to compare a supplied `dataset_sha256`
  against the worker-local file hash and **fail non-retryably on mismatch**. This is
  a small, targeted Phase-3 change — call it out in the changelog as a Phase-4-driven
  Phase-3 patch, and re-run the Phase-3 harness to confirm non-regression.

---

## BLOCKER 7 — macro-stripe vs sub-stripe sizing (do not conflate)

The coordinator assigns **macro-stripes**; the WORKER selects the effective
family/backend cap and partitions the macro-stripe into GPU-safe sub-stripes itself
(`range_miner_worker.py:_partition…` :498). rev-1's Gate 1 wrongly implied every
macro-stripe must fit one GPU cap.

Binding cap rule:
- `miner_stripe_size` controls **macro-stripe** scheduling.
- Worker-advertised `seed_caps` (in `RegisterMessage.capabilities`) are the runtime
  source for **sub-stripe** sizing.
- At **registration**, the coordinator **validates** advertised caps against the
  centrally-resolved config (WATCHER manifest §7, 6-level precedence). Mismatch →
  **reject or quarantine** the worker; never silently pick one value.
- Coordinator records `expected_substripes = ceil(stripe.seed_count / advertised_effective_cap)`.
- Coordinator **verifies the worker advertises the exact concrete variant** before
  assigning it (ties to Phase-3 `SUPPORTED_VARIANTS`).
- **Gate 1 must NOT require macro-stripe ≤ one GPU cap.**

---

## BINDING DECISION A — node identity: connection-bound, NOT string parsing

Do **not** add `hostname` to every `SubStripeResultMessage`, and do **not** blindly
parse hostname from `worker_id`. The protocol already provides `RegisterMessage.hostname`,
`RegisterMessage.worker_id`, and inherited `worker_id` on every message. At registration:
- **bind the TCP connection** to the registered worker;
- map `worker_id` → registered hostname + configured **node record**;
- **reject** any later message whose `worker_id` doesn't match that connection;
- use the **node configuration's** SSH address + username for transfer — not a
  parsed hostname.

## BINDING DECISION B — remote deletion: coordinator-owned transfer adapter

No new protocol message. The transfer abstraction exposes:
```
fetch_remote(...)
delete_remote(...)
```
`delete_remote()` may run **only after** local size/hash verification succeeds.
Record cleanup status durably; retry failed deletions idempotently. Test wording:
"**invokes remote release/deletion only after successful local verification**"
(not "signals the worker its spool is deletable").

---

## Reuse from `zmq_sqlite_coordinator.py` (adapted)
- SQLite **sole-writer** discipline + `_write_lock` — keep.
- Lease-reclaim **mechanism** (`:229-264`) — reuse, but phase-specific policy
  (Blocker 3) and staging-aware (Blocker 5).
- Dispatch/collect **loop shape** (`:855-895`) — reuse skeleton.
- **Do NOT** reuse: `MAX_ATTEMPTS=3`, the one-row result table, the serial
  `np.load` collection (Phase 5's job), ZMQ sockets (use Phase-2 framed TCP).

## Residue/window contract Phase 4 MUST supply
Every `StripeAssignMessage.payload`: `dataset`, `dataset_sha256` (mandatory),
`window_size`, `sessions`, `offset`, `residue_sha256` (mandatory). Ties Blocker 6.

## Coexistence (§8)
Activate via `use_range_miner`; PWC + ZMQ paths untouched (Phase-6 comparators).

---

## Test harness gates (`tests/test_s172_phase4_coordinator.py`)
CPU-only, loopback fake workers, stubbed `fetch_remote`/`delete_remote`.

1. **Macro-stripe partition + assign** — no gap/overlap; macro-stripe MAY exceed one
   GPU cap; `expected_substripes` computed from advertised cap. (Blocker 7)
2. **Multiple sub-stripe results under one stripe** — N `SubStripeResultMessage` +
   one `StripeComplete`; shard-level ledger has N rows keyed by sub_index. (B1)
3. **Missing / duplicate / overlapping sub_index** — stripe NOT marked done;
   coverage validation rejects each pathology. (B1)
4. **Shard-level done conditions** — stripe `done` only when StripeComplete + all
   sub_index present + exact coverage + all staged+verified + totals reconcile. (B1)
5. **Staging state** — after StripeComplete, stripe enters `staging`, not `done`;
   compute lease reclaim does NOT fire during `staging`; no duplicate reassign. (B5)
6. **StripeComplete before transfers finish** — coordinator waits for staging to
   verify before `done`. (B5)
7. **Partial-attempt cleanup before retry** — attempt-0 emits 2 good shards then a
   failure; all attempt-0 local shards invalidated/removed; NOT published; stripe
   retried whole as attempt 1. (B2)
8. **TrialCommit vs TrialAbort** — success → TrialCommit publishes committed input;
   terminal failure → TrialAbort, provisional input discarded. (B2)
9. **Phase 1/2 immediate failure** (constant) — fails trial immediately, no retry. (B3)
10. **Phase 3/4 one-retry-then-fail** (hybrid) — first retryable failure reassigns
    once (to a DIFFERENT eligible worker) + `phase_degraded`; second fails trial. (B3)
11. **`retryable=False` immediate failure** — fails trial immediately, does NOT
    consume the retry. (B3)
12. **Lease expiry applies phase-specific policy** — expiry in constant phase fails;
    in hybrid phase reassigns once. (B3)
13. **Inline result normalized** — a small inline result is canonically serialized,
    size+sha256 verified, atomically written Zeus-local, enqueued as the SAME
    path-manifest as remote spools. (B4)
14. **Remote staging happy path (stubbed)** — fetch_remote → re-hash → match →
    enqueue path-manifest → mark verified → delete_remote invoked ONLY after verify. (Decision B)
15. **Hash mismatch** — transferred bytes ≠ advertised sha256 → failed sub-stripe
    (→ retry path); delete_remote NOT invoked. (§15)
16. **Byte reservation / high-water mark** — reserve before transfer so
    `staged_bytes + incoming_size <= high_water_mark`; back-pressure at the mark;
    staged bytes never exceed it. (§15)
17. **Spool path restricted to registered worker's configured spool root** — a
    manifest whose path escapes the worker's configured spool root is rejected. (security)
18. **Connection-bound identity** — a message whose `worker_id` doesn't match its
    bound connection is rejected. (Decision A)
19. **Cap / supported-variant mismatch at registration** — a worker advertising caps
    inconsistent with central config, or a variant it can't support, is rejected/
    quarantined. (Blocker 7)
20. **Worker-local dataset SHA mismatch** — payload `dataset_sha256` ≠ worker file
    hash → resolver fails non-retryably (requires the Blocker-6 Phase-3 patch). (B6)
21. **No assembly in dispatch thread** — coordinator never imports/calls Phase-5
    assembly, never builds 22 arrays, never runs contract wall. (§3.A)
22. **Coexistence non-interference** — `use_range_miner` selects miner; PWC/ZMQ
    importable + unmodified.
23. **Non-regression** — Phases 0/1/2/3 harnesses still green (subprocess), INCLUDING
    the Blocker-6-patched Phase-3 resolver.

Exit 0 = all green (Phase 4 shippable pending Beta re-review). Exit 1 = DO NOT COMMIT.

---

---

# rev-3 ADDITIONS (Team Beta rev-2 review — four lifecycle blockers + schema fix)

## L1 — Stale-attempt fencing (delayed-message safety)

Verified at source: `attempt` is on `StripeAssignMessage` (`range_miner_protocol.py:109`)
but **NOT** on `SubStripeResultMessage` / `StripeCompleteMessage` / `StripeErrorMessage`.
So after a lease expires and attempt 1 runs on worker B, delayed attempt-0 messages
from worker A cannot be distinguished by attempt at the message level — and the
coordinator must NOT bind them to the current attempt just by `stripe_id` lookup.

**Binding rule (no protocol expansion needed — bind durably at assignment):**
Every stripe-flow message is accepted ONLY when ALL hold:
```
connection.worker_id == message.worker_id
AND ledger.claimed_by == message.worker_id
AND ledger.current_attempt == the connection's recorded assignment attempt
AND the stripe state permits that message type
```
When an attempt fails or expires, **fence that assignment before requeueing**: record
the superseded (worker, attempt) so any later message from the old worker is stale →
ignored or logged, and MUST NOT alter the new attempt's ledger.

This composes with Decision A (connection-bound identity): the connection already
knows its worker; L1 adds that the connection also carries its **assignment attempt**,
and the ledger's `current_attempt` is the authority.

## L2 — Bounded-staging release / Phase 5 acknowledgement seam

rev-2 reserves bytes before transfer but never defines when they STOP counting.
Releasing on mere-enqueue lets queued files exceed the mark; never releasing
deadlocks staging. Define the acknowledgement seam now:
```
reserve bytes → write/verify local shard → enqueue ShardReadyManifest
  → Phase 5 acknowledges shard consumed or safely internalized
  → Phase 4 deletes/releases local staged file → release byte + file reservation
```
The high-water calculation MUST count every local staged file **not yet acked AND
deleted**:
```
reserved_or_staged_bytes + incoming_size <= high_water_mark   (bytes)
staged_or_pending_files  + 1             <= high_water_files   (count)
```
Durable shard fields for the seam: `phase5_status`, `phase5_enqueued_at`,
`phase5_acked_at`, `local_cleanup_status`, `local_deleted_at`.

(Phase 5 does not exist yet; Phase 4 defines and emits the seam and treats the ack
as an interface Phase 5 will call. For the harness, the ack is stubbed.)

## L3 — Whole-trial abort cleanup

A terminal failure can occur AFTER other stripes have completed and published
provisional manifests. On `TrialAbort`, Phase 4 MUST:
- emit **exactly one** terminal abort event;
- **prevent any subsequent `TrialCommit`** (abort is terminal, idempotent);
- invalidate **every provisional shard for the entire trial** (not just the failed
  stripe's);
- instruct Phase 5 to discard them;
- remove all Phase-4-owned local staged files once safe;
- release ALL corresponding byte + file reservations;
- retry any pending **remote** deletions idempotently;
- mark all still-pending or active stripes **cancelled/aborted**.

## L4 — Hybrid-cap + staging-resource configuration wiring

Verified at source: `run_trial_miner()` (`range_miner_coordinator.py:40-41`) exposes
only `seed_cap_nvidia` / `seed_cap_amd` — **no hybrid caps**, yet Blocker 7 requires
validating against all four advertised caps. The deliverable MUST add and wire:
```python
seed_cap_nvidia_hybrid: int = 2_500_000
seed_cap_amd_hybrid:    int = 1_000_000
```
following the approved **6-level precedence** (§12.4/§7) and compared against the
worker's complete `seed_caps` advertisement. **Missing keys, non-positive values, or
mismatch → quarantine the worker** (ties Blocker 7 registration validation).

Also expose/configure (NOT bury as constants) the new Phase-4 resources:
```
staging_high_water_bytes
staging_high_water_files
staging_dir
compute_lease_timeout
staging_timeout
```

## SC1 — Schema correction: durable remote-deletion status (Decision B)

Decision B requires remote-deletion status to be durable, but the rev-2 shard table
omitted the fields. Add to the shard-level table:
```
remote_delete_status
remote_delete_attempts
remote_delete_error
remote_deleted_at
```
Remote-deletion failure does **not** invalidate a verified shard, but must remain
visible and retryable (idempotent retry).

## rev-3 ADDED GATES (append to the harness; keep gates 1–23)

24. **Stale-attempt fencing (L1).** attempt-0 lease expires → attempt 1 assigned to
    worker B → a delayed result/complete from worker A arrives → message REJECTED;
    attempt-1 ledger unchanged.
25. **Queue-does-not-release capacity (L2).** Enqueuing a ShardReadyManifest alone
    does NOT release its byte/file reservation; only Phase 5 ack + local delete does.
    Assert reserved bytes stay counted until ack+delete.
26. **Phase 5 ack releases capacity (L2).** After stubbed Phase 5 ack + local delete,
    the reservation is released and new staging can proceed up to the mark.
27. **High-water counts unacked staged files (L2).** With several enqueued-but-unacked
    shards, `reserved_or_staged_bytes` includes them; a new transfer that would exceed
    the mark is back-pressured.
28. **Whole-trial abort (L3).** Two stripes complete + publish provisional manifests,
    a third fails terminally → TrialAbort: no committed trial input remains, all
    provisional shards invalidated, all staging reservations released, no leaked
    files, subsequent TrialCommit refused.
29. **Four-cap validation + quarantine (L4).** A worker advertising all four caps
    consistent with config registers; one with a missing/zero/mismatched hybrid cap
    is quarantined.
30. **Staging resources configurable (L4).** `staging_high_water_bytes/files`,
    `staging_dir`, `compute_lease_timeout`, `staging_timeout` are injectable config,
    not hardcoded constants.
31. **Durable remote-delete status (SC1).** A failed remote deletion records
    status/attempts/error, leaves the verified shard valid, and is retried
    idempotently without duplicating the shard.

---

# rev-4 ADDITIONS (Team Beta rev-3 review — four narrow lifecycle details)

## L5 — Fence asynchronous TASKS, not only socket messages

L1 fences delayed *messages*. But an old attempt can have a `fetch_remote()` or
inline-write task **already running** when fenced; its completion callback could
later mark a stale shard verified, enqueue a stale manifest, invoke `delete_remote`,
or consume/release the NEW attempt's reservation.

**Binding rule:** every async task and callback carries the immutable key:
```
(run_id, stripe_id, attempt, sub_index, staging_generation)
```
Before ANY final rename, ledger update, enqueue, or remote deletion, the callback
MUST verify the attempt is still active AND the trial is not aborted. A **stale
completion** must:
- delete its own temporary local file;
- release ONLY its own reservation;
- perform **no** publication (no ledger verify, no manifest enqueue, no delete_remote).

`staging_generation` increments whenever an assignment is fenced/requeued, so even a
same-(stripe,attempt) collision after reuse is distinguishable.

## L6 — Precise Phase 5 interface (injected sink + event identity)

`ShardReadyManifest`/`TrialCommit`/`TrialAbort`/"ack" were concepts; make them a
concrete injected interface so Phase 4 is testable with a stub and Phase 5 later
implements it:
```python
class Phase5Sink:
    def publish_shard(self, manifest: "ShardReadyManifest") -> None: ...
    def commit_trial(self, event: "TrialCommit") -> None: ...
    def abort_trial(self, event: "TrialAbort") -> None: ...
```
`ShardReadyManifest` immutable fields (at least):
```
event_id            # immutable unique id — the ack key
run_id
workflow_phase      # 1/2 constant, 3/4 hybrid (§6.8)
stripe_id
attempt
sub_index
local_spool_path
expected_size
expected_sha256
trial_metadata
```
**Acknowledgement references `event_id`, NOT a path or stripe_id.** Duplicate acks
are idempotent and MUST NEVER release another shard's reservation (ack→event_id→
exactly one reservation).

## L7 — Trial-abort discard acknowledgement (race-free)

L3's "instructs Phase 5 to discard… removes files once safe" leaves a race: Phase 5
may be mid-read of a provisional local path when abort deletes it. Require ONE of:
- **Async:** `TrialAbort → Phase 5 stops/finishes reads + discards provisional state
  → TrialAbortAck → Phase 4 deletes remaining local files + releases reservations`; OR
- **Sync:** a `abort_trial()` call whose successful return **guarantees** Phase 5 no
  longer references any trial-owned path.

Either way: **a local file backing an unacked, actively-consumed shard MUST remain
until the discard acknowledgement.** Phase 4 does not delete out from under Phase 5.

## L8 — Complete stripe-accounting invariant + failure-path reservation cleanup

Verified at source: `StripeCompleteMessage` carries `substripes_done` +
`survivors_total` (`range_miner_protocol.py:133-134`) — both are authoritative
reconciliation inputs. Before a stripe → `done`, require ALL:
```
StripeComplete.substripes_done == expected_substripes == count(distinct shard sub_index)
sum(shard.seed_count)      == stripe.seed_count
sum(shard.survivor_count)  == StripeComplete.survivors_total
exact contiguous seed coverage of the stripe (no gap/overlap)
```
Any mismatch → the stripe does NOT complete (feeds the failure/retry path per Blocker 3).

**Reservation cleanup for EVERY non-success path** — capacity released ONLY after the
associated temp/staged file is actually removed:
```
fetch exception | hash mismatch | atomic-write failure |
staging timeout | stale callback (L5) | trial abort (L7)
```

## rev-4 ADDED GATES (append; keep gates 1–31)

32. **Stale async-task fencing (L5).** attempt-0's `fetch_remote` completes AFTER
    attempt 1 is assigned → the callback finds its attempt inactive → deletes its temp
    file, releases only its own reservation, publishes nothing; attempt-1 state
    unchanged.
33. **Event-id ack idempotency (L6).** Two acks for the same `event_id` release its
    reservation exactly once; an ack for event A never releases event B's reservation.
34. **Abort-discard race (L7).** A shard is unacked and actively consumed when
    TrialAbort fires → its local file REMAINS until discard-ack (TrialAbortAck or
    sync abort_trial return); only then is it deleted + reservation released.
35. **Full completion reconciliation (L8).** A stripe with substripes_done ==
    expected == distinct sub_index count, seed_count sum == stripe.seed_count,
    survivor_count sum == survivors_total, contiguous coverage → done. Break any one
    (short a sub_index / seed sum off / survivor sum off / gap) → NOT done.
36. **Failure-path reservation cleanup (L8).** For each of fetch-exception,
    hash-mismatch, atomic-write-failure, staging-timeout, stale-callback, trial-abort:
    the temp/staged file is removed AND only then is capacity released. Assert no
    reservation leaks on any path.

---

## Workflow
Read LIVE `zmq_sqlite_coordinator.py`, `miner/range_miner_protocol.py`,
`miner/range_miner_worker.py` before writing. The Blocker-6 change touches Phase 3's
`ResidueResolver` — make it surgical and re-run the Phase-3 harness. Iterate on 101
(loopback). **Do NOT commit or push.** Write `SESSION_CHANGELOG…S172_PHASE4.md`.
Report when green + STOP for Beta re-review.

## Out of scope
22-array assembly / dedup / ordering / contract wall (Phase 5, §6.7); the
serial-vs-process_sharded backends (Phase 5); four-path acceptance + high-survivor
benchmark (Phase 6, §16); soak (Phase 7); real multi-rig execution (needs CT100
keys + static IP).
