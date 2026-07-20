# Claude Code Instructions — S172 Phase 4 Coordinator (staged implementation)

**From:** Team Alpha lead (orchestrator)
**For:** Claude Code on VM 101 (`zeus-ubuntu`, 192.168.3.177), user `michael`,
working dir `/home/michael/distributed_prng_analysis`.
**Date:** 2026-07-18

---

## 0. Ground rules (non-negotiable)

1. **The spec is `docs/S172_PHASE4_BRIEF.md` (rev-4, Team-Beta approved, 36 gates).**
   Read it IN FULL before writing any code. This document does NOT restate it —
   where this document and the brief disagree, **the brief governs**. Also read
   `docs/PROPOSAL_S172_RANGE_MINER_v1_4_5.md` §3.A, §6.7, §12.3, §12.4, §15.
2. **Verify HEAD first:** `git log --oneline -1` must show `6661b04`. If not, STOP
   and report.
3. **Read live source before every claim.** Before writing code that touches or
   mimics `zmq_sqlite_coordinator.py`, `miner/range_miner_protocol.py`, or
   `miner/range_miner_worker.py`, open the file and read the relevant region.
   Four Beta rejection rounds on this project came from extrapolating instead of
   reading. Do not cite a line number you have not viewed in this session.
4. **Do NOT commit, push, or run the WATCHER pipeline.** (Deny-rules enforce this;
   respect them.) Michael commits after Beta approval.
5. Venv: `source ~/venvs/torch/bin/activate`. Run harnesses with
   `PYTHONPATH=. python3 tests/test_s172_phase4_coordinator.py`.
6. **Implement in the stages below, in order.** Each stage's gates must be green
   before starting the next. Do not write the whole coordinator in one pass —
   that is exactly how rev-1 got the cardinality wrong.
7. All 36 gates are CPU-only, loopback, with stubbed `fetch_remote`/`delete_remote`
   and a stubbed `Phase5Sink`. No GPU, no rigs, no real SSH.
8. When all 36 gates + non-regression are green: write
   `docs/SESSION_CHANGELOG_YYYYMMDD_S172_PHASE4.md`, then **STOP and report**.
   Do not proceed to Phase 5. Beta review comes next.

## Deliverables

| File | Action |
|---|---|
| `miner/range_miner_coordinator.py` | Flesh out (currently a 63-line stub). Keep `run_trial_miner()` importable with its existing signature, extended per L4. |
| `tests/test_s172_phase4_coordinator.py` | New. All 36 gates from the brief, numbered to match. |
| `miner/range_miner_worker.py` | ONE surgical patch (Blocker 6, Stage 0 below). Nothing else. |
| `docs/SESSION_CHANGELOG_YYYYMMDD_S172_PHASE4.md` | Session changelog incl. the Blocker-6 "Phase-4-driven Phase-3 patch" callout and the two flagged decisions (below). |

---

## Stage 0 — Blocker-6 Phase-3 patch (do this FIRST, in isolation)

**BINDING: Team Beta ruled Option C — reject-on-absence AND reject-on-mismatch,
both non-retryable. This OVERRIDES the earlier "compare-when-present"
recommendation. Implement Beta's ruling verbatim; do not compare-when-present.**

Target: `miner/range_miner_worker.py`, `ResidueResolver.resolve()` (~line 579–623).
Verified at source: `dataset_sha = self._file_hasher(dataset)` is computed and used
ONLY as a cache key. The required-field + equality checks MUST run **before any
cache return or residue loading** (i.e. before the `if key in self._cache` return
and before `self._loader(...)`).

Required behavior (Beta-specified, implement exactly):
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
- Plain `!=` comparison — this hash is an integrity identifier, NOT a secret; no
  constant-time compare needed (Beta-specified).
- `ResidueResolutionError` (missing) and `ResidueVerificationError` (mismatch)
  both inherit `ResidueError` and both route to `stripe_error(retryable=False)`
  via the existing routing at `range_miner_worker.py:1135-1136` — verify this
  routing still holds after the patch. Do NOT add a new exception class.
- The resolver already computes the local hash for the cache key; reuse that
  computation — do not hash the file twice. Placement is the point: the checks
  gate the method BEFORE cache return and BEFORE loading.
- Do NOT touch the `residue_sha256` verification or the cache-key structure
  otherwise.

**Phase-3 harness update is AUTHORIZED AND REQUIRED (Beta ruling).** "Non-regression"
now means all *valid* Phase-3 behavior still passes — old fixtures emitting
payloads that are now explicitly invalid (no `dataset_sha256`) MUST be updated.
Edit `tests/test_s172_phase3_worker.py`:
- Existing valid resolver fixtures (incl. gate 9 / B1 residue window) must compute
  and supply `dataset_sha256` matching the fake loader's dataset file hash.
- ADD a missing-`dataset_sha256` test → asserts `ResidueResolutionError` →
  non-retryable.
- ADD a mismatched-`dataset_sha256` test → asserts `ResidueVerificationError` →
  non-retryable.
- ADD a cache-safety test proving a cached window CANNOT bypass a later hash
  mismatch (i.e. a first resolve populates cache; a second resolve with a
  mismatched `dataset_sha256` still raises — the checks run before cache return).
- PRESERVE existing different-window and `residue_sha256` verification coverage.

This ruling is narrowly limited to `dataset_sha256`. `residue_sha256` stays
mandatory on every production assignment (coordinator-supplied, Stage 4).

Then: `PYTHONPATH=. python3 tests/test_s172_phase3_worker.py` → all gates green
INCLUDING the new ones (harness count grows from 14). If any *valid-behavior*
gate breaks, STOP and report before proceeding.

---

## Stage 1 — Ledger + state machine + reconciliation (gates 1, 2, 3, 4, 5, 6, 35)

Build the durable core in `miner/range_miner_coordinator.py`:

- **SQLite schema.** Reuse the sole-writer discipline from
  `zmq_sqlite_coordinator.py` (`self._write_lock = threading.Lock()`, every write
  under it, WAL mode). Do NOT reuse `MAX_ATTEMPTS=3` or the one-row `job_results`
  table.
  - **Stripe table:** keyed `(run_id, stripe_id)`; fields incl. `seed_start`,
    `seed_count`, `state` (`pending|claimed|staging|done|failed|cancelled`),
    `claimed_by`, `current_attempt`, `staging_generation`, `expected_substripes`,
    `lease_expires_at`, `phase` (workflow phase 1–4), `family_name`,
    `phase_degraded`.
  - **Shard table (Blocker 1 + SC1 + L2):** keyed
    `(run_id, stripe_id, attempt, sub_index)`; fields per the brief:
    `worker_id, seed_start, seed_count, survivor_count, remote_spool_path,
    local_staged_path, size_bytes, sha256, staging_status, created_at, verified_at`
    PLUS SC1: `remote_delete_status, remote_delete_attempts, remote_delete_error,
    remote_deleted_at` PLUS L2 seam: `phase5_status, phase5_enqueued_at,
    phase5_acked_at, local_cleanup_status, local_deleted_at`.
- **State machine (Blocker 5):** `pending → claimed → staging → done / failed`.
  Heartbeats renew leases only while `claimed`; lease reclaim (adapt the mechanism
  from `zmq_sqlite_coordinator.py:229-264`) applies ONLY to `claimed`; `staging`
  has its own `staging_timeout`; `done` requires local verification AND publish.
- **Macro-stripe partitioner (Blocker 7):** contiguous, no gap/overlap over
  `total_seeds`; macro-stripes MAY exceed one GPU cap;
  `expected_substripes = ceil(seed_count / advertised_effective_cap)` recorded at
  assignment using the assigned worker's advertised cap for the resolved variant
  (hybrid vs constant selects between the four caps — mirror
  `select_seed_cap()` logic, `range_miner_worker.py:468-476`).
- **L8 completion predicate** — a stripe becomes `done` ONLY when ALL hold:
  `StripeComplete.substripes_done == expected_substripes == count(distinct sub_index)`;
  `sum(shard.seed_count) == stripe.seed_count`;
  `sum(shard.survivor_count) == StripeComplete.survivors_total`;
  exact contiguous coverage (no gap/overlap); every shard staged + hash-verified.
  Any mismatch → failure/retry path (Stage 4 policy), never `done`.

Gate targets this stage: 1, 2, 3, 4, 5, 6, 35. (Gates 5/6 need only a minimal
staging stub at this point — full staging is Stage 3; structure the harness so
these gates use the injectable adapter you'll complete in Stage 3.)

---

## Stage 2 — Identity, registration validation, fencing (gates 17, 18, 19, 24, 29, 30)

- **Decision A — connection-bound identity.** At registration, bind the TCP
  connection (Phase-2 framed wire; server-side counterpart of the worker's
  `MinerFramedSocket`) to `RegisterMessage.worker_id` + hostname + configured node
  record. Reject any later message whose `worker_id` ≠ the connection's bound id.
  Transfers use the NODE CONFIG's SSH address/user — never a parsed hostname.
- **Blocker 7 + L4 — registration validation.** The worker advertises
  `capabilities = {"supported_variants": [...], "seed_caps": {"amd", "nvidia",
  "amd_hybrid", "nvidia_hybrid"}}` (verified: `range_miner_worker.py:1073-1077`,
  `VramCaps` at `:456-464`). Validate ALL FOUR caps against centrally-resolved
  config; missing key, non-positive value, or mismatch → **quarantine** (worker
  registered-but-ineligible, durably visible; never silently pick a value). Verify
  the exact concrete variant is advertised before assigning a stripe.
- **L4 config wiring:** extend `run_trial_miner()` with
  `seed_cap_nvidia_hybrid: int = 2_500_000` and
  `seed_cap_amd_hybrid: int = 1_000_000`, honoring the 6-level precedence
  (§12.4/§7). Also make `staging_high_water_bytes`, `staging_high_water_files`,
  `staging_dir`, `compute_lease_timeout`, `staging_timeout` injectable config —
  NOT module constants.
- **L1 — stale-attempt message fencing.** Result/complete/error messages carry NO
  `attempt` (verified: only `StripeAssignMessage` has it,
  `range_miner_protocol.py:109`). Accept a stripe-flow message ONLY when:
  connection.worker_id == message.worker_id AND ledger.claimed_by ==
  message.worker_id AND ledger.current_attempt == the connection's recorded
  assignment attempt AND the state permits that message type. On failure/expiry,
  fence the superseded (worker, attempt) BEFORE requeueing; stale messages are
  logged and MUST NOT touch the new attempt's ledger.
- **Gate 17 — spool-root restriction.** A `spool_path` outside the registered
  worker's configured spool root is rejected (path-normalize before comparing;
  guard `..` traversal).

---

## Stage 3 — Staging pipeline + reservations (gates 13, 14, 15, 16, 25, 26, 27, 31, 32, 36)

- **Transfer adapter (Decision B):** injectable object exposing
  `fetch_remote(...)` / `delete_remote(...)`. Stubbed in the harness. No new
  protocol message. `delete_remote()` runs ONLY after local size+sha256
  verification; failures recorded durably in SC1 columns and retried idempotently
  (a failed deletion never invalidates a verified shard — gate 31).
- **Reservations (L2 + §15):** reserve BEFORE transfer;
  `reserved_or_staged_bytes + incoming <= staging_high_water_bytes` and
  `staged_or_pending_files + 1 <= staging_high_water_files`; back-pressure at the
  mark. Capacity counts every local staged file NOT yet (Phase-5-acked AND
  deleted). Mere enqueue releases nothing (gate 25); ack + local delete releases
  (gate 26).
- **Inline normalization (Blocker 4):** an inline `SubStripeResultMessage` is
  canonically re-serialized to the SAME `s172_substripe_v1` bytes — use
  `json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")`
  exactly as `build_substripe_payload_bytes()` (`range_miner_worker.py:861-880`;
  import or mirror it — if you mirror, add a harness assertion that both
  serializations are byte-identical for a sample payload). Verify `size_bytes` +
  `sha256`, atomically write Zeus-local (temp in same dir → fsync →
  `os.replace`, mirroring `spool_payload_atomic`), enqueue the SAME path-manifest
  shape as remote spools. Phase 5 sees ONE uniform input type.
- **L5 — async TASK fencing.** Every async task/callback carries the immutable key
  `(run_id, stripe_id, attempt, sub_index, staging_generation)`. Before ANY final
  rename, ledger update, enqueue, or `delete_remote`, the callback re-checks the
  attempt is active AND the trial is not aborted. Stale completion: delete own
  temp file, release ONLY own reservation, publish nothing.
  `staging_generation` increments on every fence/requeue.
- **L8 failure-path cleanup (gate 36):** for fetch-exception, hash-mismatch,
  atomic-write-failure, staging-timeout, stale-callback, trial-abort — the
  temp/staged file is removed FIRST, and only then is capacity released. No
  reservation leak on any path (harness must assert reserved bytes/files return
  to zero after each pathology).

---

## Stage 4 — Retry matrix, trial lifecycle, Phase5Sink (gates 7, 8, 9, 10, 11, 12, 28, 33, 34)

- **Blocker 3 — retry matrix, implement EXACTLY** (workflow phase from
  `StripeAssignMessage.phase`, 1–4 per §6.8):
  - phase 1/2 (constant) failure → fail trial immediately;
  - phase 3/4 (hybrid) retryable first failure → reassign ONCE to a DIFFERENT
    eligible worker, set `phase_degraded=True` in agent_metadata;
  - phase 3/4 retryable second failure → fail trial;
  - ANY `retryable=False` → fail trial immediately, retry NOT consumed;
  - lease expiry → same phase-specific policy. No `MAX_ATTEMPTS=3`.
- **Blocker 2 — attempt-scoped publish.** Stage per attempt; publish an attempt's
  manifests to Phase 5 ONLY after: all shards verified + StripeComplete + full L8
  reconciliation. On attempt failure: invalidate + remove ALL that attempt's local
  shards; never publish; retry the WHOLE stripe per the matrix.
- **L6 — injected `Phase5Sink`:** `publish_shard(manifest)`, `commit_trial(event)`,
  `abort_trial(event)`, plus a shard-ack entry point keyed by **`event_id`**
  (immutable unique id on each `ShardReadyManifest`; manifest fields per the brief:
  `event_id, run_id, workflow_phase, stripe_id, attempt, sub_index,
  local_spool_path, expected_size, expected_sha256, trial_metadata`). Duplicate
  acks idempotent — a reservation releases exactly once; ack for event A never
  touches event B (gate 33).
- **L7 — abort-discard race. BINDING: Team Beta ruled Option A — synchronous
  `abort_trial()`. No async ack.** Implement exactly:
  - A successful return from `phase5_sink.abort_trial(event)` GUARANTEES Phase 5
    has stopped/completed all reads for that trial, holds no reference to any
    trial-owned staged path, has discarded every provisional shard + partial
    assembly state, and will reject/harmlessly ignore later stale manifests.
    ONLY AFTER that successful return may Phase 4 delete remaining Zeus-local
    staged files, release byte + file reservations, and complete cleanup
    bookkeeping.
  - **Do NOT add** any of: `TrialAbortAck`, an abort callback, a second
    pending-abort protocol, or a separate ack-timeout state machine. (Beta:
    async variant explicitly not approved for Phase 4.)
  - **Dispatch-thread requirement:** the call is synchronous but MUST NOT run
    inside the socket receive/dispatch loop. Route abort cleanup through the
    coordinator's lifecycle / cleanup executor, in this order:
    `terminal failure detected → persist trial state = aborted → fence all
    active assignments → schedule synchronous Phase5Sink.abort_trial() → wait
    for successful completion → remove staged files → release reservations`.
    This keeps the network dispatcher responsive while preserving synchronous
    discharge.
  - **Failure/timeout:** if `abort_trial()` raises, times out, or the coordinator
    exits mid-cleanup — the trial stays terminally aborted; `TrialCommit` stays
    permanently prohibited; **staged files + reservations are RETAINED (never
    deleted merely because abort delivery was attempted)**; cleanup status
    becomes `pending`/`failed`; the sync abort call is retried idempotently. Use
    the configured `staging_timeout` as the initial bound UNLESS you add a
    separately configurable `phase5_abort_timeout` (either is acceptable per Beta).
  - **Idempotency by `(event_id, run_id)`:** repeated calls for the same abort
    event return successfully after confirming no Phase 5 references remain.
  - **Gate 34:** local file exists WHILE the sync stub executes → stub drains and
    returns → ONLY THEN Phase 4 deletes the file + releases its reservation.
- **L3 — whole-trial abort:** exactly one terminal abort event; subsequent
  `TrialCommit` refused (abort terminal + idempotent); EVERY provisional shard of
  the trial invalidated; all Phase-4 local staged files removed once safe; ALL
  reservations released; pending remote deletions retried idempotently; all
  pending/active stripes marked cancelled.
- **Payload contract (ties Stage 0):** every `StripeAssignMessage.payload`
  carries `dataset`, `dataset_sha256` (MANDATORY — coordinator computes it),
  `window_size`, `sessions`, `offset`, `residue_sha256` (MANDATORY —
  `run_trial_miner()` has the residues; compute via
  `sha256_residues`, same function the worker uses). Assert both present in the
  assign path — never optional.

---

## Stage 5 — Integration, coexistence, non-regression (gates 20, 21, 22, 23)

- Wire `run_trial_miner()` to drive the coordinator (loopback-testable; the
  harness may drive the coordinator object directly for most gates and exercise
  `run_trial_miner` argument plumbing separately).
- Gate 20: end-to-end through a fake worker using the REAL patched
  `ResidueResolver` with a wrong `dataset_sha256` → `stripe_error(retryable=False)`
  → trial fails immediately (matrix row 4).
- Gate 21: coordinator module imports no Phase-5 assembly, builds no 22-array
  structures, runs no contract wall (assert on module imports + a source grep in
  the harness is acceptable).
- Gate 22: `use_range_miner` selects the miner path; PWC + ZMQ modules remain
  importable and unmodified (`git diff --name-only` in the harness must show only
  the three code deliverables).
- Gate 23: subprocess re-run of Phase 0/1/2/3 harnesses — ALL green, including
  the Stage-0-patched resolver.

---

## Harness conventions

Match `tests/test_s172_phase3_worker.py`: numbered gates with PASS/FAIL/SKIP
colored output, module docstring listing every gate, `_ROOT` sys.path insert,
exit 0 = all green / exit 1 = DO NOT COMMIT. Number gates 1–36 exactly as the
brief numbers them. No gate may be skipped — all 36 are CPU-runnable.

## Reporting format (when green)

Report: per-stage summary; confirmation that both Beta rulings are implemented as
specified (Stage 0 = reject-on-absence + reject-on-mismatch, both non-retryable,
checks before cache return, Phase-3 harness updated with the 3 new tests +
fixtures supplying `dataset_sha256`; Stage 4 = synchronous `abort_trial()` off the
dispatch thread, no async ack, retained-on-failure, idempotent by
`(event_id, run_id)`); any deviation from the brief or rulings WITH the exact line
it deviates from and why; harness output (all 36 Phase-4 gates + Phase 0/1/2/3
re-runs, the Phase-3 count now > 14); and the changelog path. The changelog must
record the Blocker-6 change as a "Phase-4-driven Phase-3 patch" and cite both
binding rulings (Ruling 1 Option C, Ruling 2 Option A). Then STOP. Team Alpha
reviews the actual files against source, then Team Beta binding review, then
Michael commits.
