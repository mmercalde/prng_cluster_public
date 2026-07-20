# Claude Code Instructions — S172 Phase 4 CORRECTION (Beta rejection: real serve path)

**From:** Team Alpha lead
**For:** Claude Code on VM 101, `~/distributed_prng_analysis`, user `michael`
**Date:** 2026-07-18
**Status:** Team Beta REJECTED the Phase-4 submission — one release blocker. This
corrects it. Do NOT start Phase 5.

---

## Why this correction exists (read first)

Team Beta's binding ruling: the coordinator's **real serve path is the central
Phase-4 deliverable**, not a Phase-6/7 integration detail. The current
`run_trial_miner()` raises `NotImplementedError` when `_serve` is not injected, and
`_default_serve()` is a stub that also raises. That is the blocker. Real AMD rigs /
CT100 keys remain out of scope, but a **functioning coordinator server** — one that
speaks the Phase-2 framed TCP protocol — is in scope and required.

The good news: every coordinator method the serve loop must call already exists and
is Beta-context-verified (`register_worker`, `accept_stripe_message`,
`record_substripe_result`, `record_stripe_complete`, `handle_stripe_failure`,
`process_lease_expiry`, `finalize_stripe`, `commit_trial`, `abort_trial`,
`build_stripe_assign_payload`, the staging/reservation methods). You are NOT
redesigning anything. You are adding the socket server that wires them together —
essentially lifting the hand-rolled loop that gate 20's harness already uses over
loopback into a real coordinator method.

**Do not touch** the ledger, retry matrix, staging, reservations, fencing, abort,
or resolver logic. Those passed Alpha review and Beta did not contest them. This
correction is additive plus deleting two `NotImplementedError` raises.

---

## Part 1 — Implement the real default serve path

### 1a. Add `RangeMinerCoordinator.serve_trial(self, context) -> Dict[str, Any]`

A real framed-TCP server loop. It must:

- **Bind/listen** on the configured miner port (add a `miner_port` field to
  `CoordinatorConfig` if absent; default it, and let `run_trial_miner` pass it
  through). Use `SO_REUSEADDR`, a sane `settimeout`, bind host from config
  (default `0.0.0.0` or `127.0.0.1` — pick and document; loopback is fine for the
  gate). Port 0 must be selectable so the gate can grab an ephemeral port; expose
  the bound port so a caller/harness can read it (e.g. stash on the coordinator or
  accept a pre-bound socket — see 1c).
- **Accept connections** and read framed messages via the SAME server-side framing
  the worker uses (`MinerFramedSocket` / the Phase-2 protocol — reuse it, do not
  re-mirror). One connection per worker; support at least two concurrent workers
  (thread-per-connection is acceptable and simplest; if you use threads, guard all
  ledger access — the ledger already serializes writes under `_write_lock`, so that
  holds, but do not introduce new shared mutable state without a lock).
- **On `RegisterMessage`:** call `register_worker(...)` with the node config
  resolved from the connection + config (Decision A — identity binds worker_id +
  the configured node record; transfers use the node config SSH address/user, never
  a parsed hostname). Bind the connection to that worker identity. A quarantined
  registration stays connected-but-ineligible; do not assign it stripes.
- **Dispatch `StripeAssignMessage`:** partition/assign via the existing
  `assign_stripes(...)` path, build the payload via `build_stripe_assign_payload`
  (mandatory `dataset_sha256` + `residue_sha256`), send framed to the bound worker.
- **Process inbound** `SubStripeResultMessage`, `StripeCompleteMessage`,
  `StripeErrorMessage`, `HeartbeatMessage`: gate EVERY one through
  `accept_stripe_message(...)` (L1 fencing) BEFORE mutating the ledger; a rejected
  message is logged and dropped, touches no ledger state. Route accepted messages
  to the existing handlers (`record_substripe_result`, `record_stripe_complete` →
  `finalize_stripe`, `handle_stripe_failure` for errors, lease renewal for
  heartbeats). Inline results normalize through the existing `stage_inline_shard`;
  spooled results through the existing remote-staging path (the transfer adapter is
  injectable — in the gate it is stubbed; in production it is the real one).
- **Retry/lease:** run `process_lease_expiry(...)` on a timer or between polls so
  expired compute leases route through the matrix. Reassignment to a DIFFERENT
  eligible worker must actually re-dispatch to that worker's connection.
- **Terminate** when the trial reaches a terminal state — all stripes done →
  `commit_trial(...)`; or a failure → `abort_trial(...)` (the SYNCHRONOUS L7 path,
  which already runs its cleanup through the cleanup executor off the dispatch
  thread — call `abort_trial`/`fail_trial`, do not re-implement discharge).
- **Clean shutdown:** close all worker sockets, the listen socket, the cleanup
  executor (`self._cleanup_executor.shutdown(wait=True)` if created), and ensure
  staged-resource bookkeeping is consistent. Send `MinerShutdownMessage` to
  connected workers before closing where the protocol supports it.
- **Return the real trial result** — a dict with at least the run_id, terminal
  trial state (committed/aborted), per-stripe completion summary, and the published
  manifests (or a reference to them). NEVER raise `NotImplementedError`.

Keep it single-trial: `serve_trial` runs one trial's lifecycle to terminal state
and returns. Do not build a multi-trial daemon.

### 1b. Wire the default in `run_trial_miner`

Replace the `if _serve is None: raise NotImplementedError(...)` with:
```python
if _serve is None:
    _serve = coordinator.serve_trial
```
and delete the stub `_default_serve` raise (or repoint it at `serve_trial`). Keep
the `_serve` injection seam intact for tests. `run_trial_miner` still builds the
`CoordinatorConfig` + ledger + coordinator + trial exactly as now, then calls
`_serve(coordinator, context)` — the only change is the default is real.

### 1c. Testability

The gate needs to run the server without a real fleet. Support an ephemeral port
(bind port 0) and expose the bound address so the gate can connect fake workers.
Two clean options — pick one and document it:
- `serve_trial` binds internally and stashes `self.bound_addr` before accepting, and
  the gate reads it after starting `serve_trial` in a thread; OR
- `serve_trial` accepts an optional pre-bound `listen_sock` (the gate binds port 0,
  passes it in, reads `getsockname()` itself). This is usually easier to test
  deterministically.

---

## Part 2 — Mandatory new gate (Beta-required)

Add a gate that calls **`run_trial_miner()` WITHOUT injecting `_serve`**, driving
loopback fake workers over REAL framed sockets. It must prove all six:

1. workers register (real `RegisterMessage` over the framed socket);
2. stripes are assigned (real `StripeAssignMessage` received by the fake workers);
3. at least one inline OR spooled result traverses the REAL server path
   (`serve_trial`, not a harness loop) and is staged/verified;
4. the trial reaches a terminal state (committed or aborted);
5. `run_trial_miner()` returns normally with the real result dict;
6. NO `NotImplementedError` occurs anywhere.

**Use TWO fake workers** so the same test can exercise hybrid reassignment to a
DIFFERENT worker (drive one hybrid-phase worker to a retryable failure/lease expiry
and assert the stripe re-dispatches to the second worker, `phase_degraded=True`).

Number it to fit the harness (e.g. gate 24b or gate 37 — match the existing
numbering convention; the brief's 36 are fixed, so an added integration gate should
be clearly labeled as the Beta-required serve-path gate). The gate is CPU-only,
loopback, stubbed transfer + stubbed `Phase5Sink` — same constraints as the rest.

Fake workers: reuse the real `RangeMinerWorker` over loopback where practical (as
gate 20 does), or a minimal framed-socket fake that speaks register → receive
assign → send result/complete. Real `MinerFramedSocket` framing either way — Beta
requires the REAL server path be exercised, so no shortcutting the wire.

---

## Part 3 — The two cosmetic corrections (Beta also requires these before commit)

1. **Stale revision labels:** bump the Phase-3 worker docstring "rev-2" and the
   Phase-3 harness header "rev-3" to match actual state (they drifted during the
   Stage-0 patch).
2. **Harness count wording:** state clearly (harness docstring + changelog) that the
   Phase-4 harness is **36 Phase-4 gates PLUS one subprocess non-regression check**
   (37 `_check` calls total), so the "36/36 vs 37" is unambiguous.

Also update `docs/SESSION_CHANGELOG_20260718_S172_PHASE4.md`: replace deviation #2
(which claimed the live serve loop was out of scope) with a note that the real
default `serve_trial` path is now implemented per Beta's binding rejection, and that
ONLY the real worker fleet / CT100 keys remain Phase-6/7 (that part IS still out of
scope — a real coordinator server binding to loopback fakes is in scope and done).

---

## Verify + report

- `PYTHONPATH=. python3 tests/test_s172_phase4_coordinator.py` → all gates green
  including the new serve-path gate, exit 0.
- `PYTHONPATH=. python3 tests/test_s172_phase3_worker.py` → still green (17 gates).
- Confirm `git status --porcelain` still shows ONLY the same code deliverables
  (coordinator, worker, the two harnesses) + the changelog — the serve path lives in
  `range_miner_coordinator.py`, so no new files unless you add a small server helper
  module (avoid it; keep it in the coordinator).

Report: what `serve_trial` does, how the new gate drives the real path with two fake
workers, the reassignment assertion, confirmation the two `NotImplementedError`
raises are gone, and the harness output. Then STOP — Team Alpha re-reviews the serve
path + new gate against source, then Team Beta binding re-review, then Michael
commits. Do NOT commit, push, or run the WATCHER pipeline.
