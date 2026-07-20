# Team Alpha Review Record — S172 Phase 4 (RANGE-MINER coordinator) — REV 2
**Reviewer:** Team Alpha (lead dev)
**Date:** 2026-07-18
**Verdict:** PASS — ready for Team Beta binding re-review.
**Supersedes:** the rev-1 Alpha record (pre-serve-path-correction).
**Method:** file-vs-source. Every claim below was read from the actual delivered
files (not the implementer's summaries) and diffed against the clean clone at
`6661b04`.

---

## Context

Team Beta REJECTED the first Phase-4 submission on ONE release blocker:
`run_trial_miner()` raised `NotImplementedError` without an injected `_serve` — i.e.
the coordinator had no real server. Beta ruled the serve path is the central Phase-4
deliverable (real AMD rigs / CT100 keys remain Phase 6/7; a functioning coordinator
server does not). This rev-2 record covers the correction. Everything Beta did NOT
contest (ledger, retry matrix, staging, reservations, fencing, abort, resolver) was
already verified in rev-1 and is unchanged.

**Alpha's rev-1 miss, stated plainly:** the rev-1 Alpha review accepted "deviation
#2 (live serve loop out of scope)" as legitimate. It was not — it was the missing
deliverable. Beta caught it. The tell was that gate 20 exercised the protocol over a
socket via a *harness-built* loop, not the coordinator's own serve path; rev-1 did
not push on that seam. Rev-2 verifies the real `serve_trial` path directly.

---

## Scope reviewed (correction delta)

| File | Change in this correction |
|---|---|
| `miner/range_miner_coordinator.py` | Added real `serve_trial()` + `_serve_dispatch`/`_serve_register`/`_dispatch_pending`/`_drop_conn`; `run_trial_miner` defaults `_serve` to `coordinator.serve_trial`; both blocking `NotImplementedError` raises deleted. |
| `tests/test_s172_phase4_coordinator.py` | Added gate 37 (real serve path, two workers). |
| `tests/test_s172_phase1_scaffolding.py` | Gate 2 updated: asserts `run_trial_miner` is the real wired entrypoint (was: asserts `NotImplementedError`). |

Harness result: **37/37 checks green, exit 0** (36 brief gates + gate 37 + the
subprocess non-regression). Phase-3 harness still 17/17.

---

## Beta's required correction — verified at source

| Beta requirement | Status (read from source) |
|---|---|
| Real default serve path (`_serve = coordinator.serve_trial`) | ✓ `run_trial_miner`: `if _serve is None: return coordinator.serve_trial(context)`; both `NotImplementedError` raises deleted (the 5 remaining are `TransferAdapter`/`Phase5Sink` abstract-method stubs) |
| Bind/listen on configured port (+ ephemeral for tests) | ✓ `serve_trial` binds `config.miner_host:miner_port` or a pre-bound `listen_sock`; `bound_addr` stashed; owns-vs-borrowed socket distinguished for shutdown |
| Accept + validate `RegisterMessage`; bind connection→identity | ✓ `_serve_register` → `register_worker` (Decision A); quarantined workers stay connected-but-ineligible (excluded from the eligible pool) |
| Dispatch `StripeAssignMessage` | ✓ `assign_stripes` once the eligible pool registers + `_dispatch_pending` with coordinator-computed mandatory `dataset_sha256` + `residue_sha256` |
| Process results/completion/errors/heartbeats **through L1 fence** | ✓ `_serve_dispatch`: EVERY stripe-flow message gated through `accept_stripe_message` before any ledger mutation; rejected/unbound messages logged and dropped, mutating nothing |
| Continue until `TrialCommit`/`TrialAbort` | ✓ `_terminal()` loop; terminal accounting → `commit_trial` (all done) / `fail_trial` (any not done); `serve_timeout` safety fail |
| Clean shutdown (sockets, executor, staged resources) | ✓ `finally`: shutdown msg to workers, close all framed sockets, close listen sock (if owned), `cleanup_executor.shutdown(wait=True)` |
| Return real trial result, never `NotImplementedError` | ✓ result dict: run_id, state, committed, workers_registered, per-stripe summary, manifests, bound_addr |
| **Gate 37: `run_trial_miner()` no `_serve`, 2 workers, real framed sockets, all 6 points + hybrid reassignment** | ✓ drives the REAL default `serve_trial` (no injected `_serve`); two `_FakeWorker`s over real `MinerFramedSocket`; asserts register / assign / real-path staged+verified+published / terminal committed / real dict / no `NotImplementedError`; worker A (fail, attempt 0) → re-dispatch to worker B (attempt 1, `phase_degraded=True`) → stripe `done`, trial `committed` |

`_serve_dispatch`, `_serve_register`, `serve_trial`, and the shutdown path were each
read line-by-line. Gate 37 was verified to drive the actual default path (not a
harness loop) — the specific thing rev-1 failed to check.

## Collateral change — legitimate

`test_s172_phase1_scaffolding.py` gate 2 previously asserted `run_trial_miner` raises
`NotImplementedError` — the exact raise Beta ordered deleted, so the gate HAD to
change. The new gate 2 asserts the real wired contract: `run_trial_miner` builds the
coordinator + creates a `running` trial (via an injected `_serve` capture) and raises
no `NotImplementedError`. This asserts the new correct behavior — it is not a defang.
Gate 22's changed-files allowlist was widened to include this test, with a comment.

## rev-1 findings that remain valid (unchanged code)

Ledger/state-machine/L8 (shard cardinality correct; lease-reclaim skips staging;
exact coverage); identity/four-cap validation (all four incl. hybrids, `bool` guard,
quarantine-not-drop); L1 fencing; staging/reservations/L5 async fence (byte-identical
inline via imported fn, global reservation accounting, verify-before-delete, zero-leak
across six pathologies, real race driven); retry matrix (all five rows, no
`MAX_ATTEMPTS`); L7 sync abort off the dispatch thread with retain-on-failure +
`(event_id, run_id)` idempotency; Blocker-2 attempt-scoped publish; event_id ack
idempotency; coexistence (PWC/ZMQ unmodified); Phase 0/1/2/3 non-regression.

## Open items for Michael before commit (housekeeping, non-blocking)

- **`python3_with_venv.sh` is a STRAY in `git status`** — it is the venv wrapper, not
  a Phase-4 deliverable. Do NOT `git add` it into the Phase-4 commit.
- **Working-prompt docs** (`CLAUDE_CODE_INSTRUCTIONS_*`, `CLAUDE_CODE_CORRECTION_*`)
  are untracked — commit to `docs/` for the trail or leave out; Alpha's lean is to
  keep working prompts out of git, but it is Michael's call.
- **Changelog nits:** the header still says "the two flagged decisions" (now three);
  the fallback-parity line still says "4 files uncommitted" (now 5). Body is correct;
  cosmetic only.
- **Version strings:** confirm the worker docstring / Phase-3 harness header rev labels
  were bumped (Beta's cosmetic ask).

## Standing

Team **Alpha** pass — file-vs-source verification that the serve-path correction and
gate 37 satisfy Beta's binding rejection, and that the rest of Phase 4 is unchanged
and still correct. NOT the binding gate. Sequence: **Team Beta binding re-review →
Michael commits + dual-pushes** the five code/test deliverables + the changelog.
