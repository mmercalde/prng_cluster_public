# Claude Code Instructions — S172 Phase 4 CORRECTION 5 (Beta: real-resource bounds)

**From:** Team Alpha lead
**For:** Claude Code on VM 101, `~/distributed_prng_analysis`, user `michael`
**Date:** 2026-07-19
**Status:** Team Beta REJECTED resubmission 4 — three narrow, directly-reproducible
blockers. Beta's own words: "Once those three fixes and gates are green, this should
finally be at approval threshold." Do NOT start Phase 5. Do NOT commit/push.

---

## Read first

The common theme in two of three: a gate checked the REGISTRY (entry count) but not the
ACTUAL RESOURCE (bytes on disk, live threads) — so the gate passed while the real resource
leaked. Every new/replacement gate below MUST assert the real resource, not a bookkeeping
proxy.

Same discipline: each defect gets a fix AND a gate that FAILS on current code, using the
real resource. Do not weaken existing gates. The accepted ledger/retry/resolver/dispatch
logic is NOT to be redesigned — these are three localized fixes.

---

## Defect 1 — large remote spools bypass the oversized-attempt check

`_attempt_footprint` estimates `need_files * INLINE_BYTE_LIMIT` (48 MiB/file). That ceiling
is valid ONLY for inline messages. A REMOTE spool exists because its payload is at least
that large and can be much larger. Beta: 70 MiB remote spool, 60 MiB high-water →
`_attempt_exceeds_highwater` returned None (admitted), then the job loops forever on
`StagingBackPressure` because the 70 MiB reservation never fits.

**Fix:** admission must use the ADVERTISED ACTUAL `size_bytes` of received remote shards, not
the inline ceiling. `record_substripe_result` already carries `size_bytes` and
`remote_spool_path`. Specifically:
- **One shard whose advertised `size_bytes` exceeds `staging_high_water_bytes`** → fail
  IMMEDIATELY (non-retryable capacity error); it can never fit, so never admit or defer it.
- **Accumulated advertised bytes for an attempt exceeding the byte high-water** → fail and
  clean the attempt (do not admit-then-loop).
- Remote spools must NOT use `INLINE_BYTE_LIMIT` as their size estimate. For an inline
  result, the payload IS bounded by the inline ceiling — keep that for inline. For a remote
  result, use the message's `size_bytes` (known at `enqueue_staging` time from the
  `SubStripeResultMessage`).

Practically: the per-shard byte check belongs where the shard's real size is known — at
`enqueue_staging` for that shard, using `msg.size_bytes` for a remote result. A single
oversized remote shard fails fast there; the attempt-footprint check can keep the
files-count guard, but the BYTE guard must be driven by actual advertised sizes as shards
arrive, not a static per-file estimate.

**Gate:** a 70 MiB remote result with a 60 MiB high-water → the shard/attempt fails
IMMEDIATELY (capacity error), the stripe does NOT sit in `claimed`/`staging` looping, no
perpetual back-pressure. Assert the failure fired and the stripe left the waiting state.

---

## Defect 2 — orphan-fetch registry is bounded but the actual threads are not

`_register_orphan_fetch` RAISES when the registry is full — but the fetch thread was ALREADY
STARTED before the register call, so it stays alive and becomes UNTRACKED. Beta: cap 2, seven
hung fetches → registry has 2 entries but SEVEN live `miner-fetch` threads. Gate 59 only
checks `len(coord._orphan_fetch_threads) <= cap`, so it passes while real threads accumulate.

**Fix:** enforce capacity BEFORE starting a non-cancellable fetch thread. No new fallback
fetch thread may be launched while the orphan allowance is exhausted:
- Check the orphan budget (pruned of finished threads) BEFORE `threading.Thread(...).start()`.
- If the budget is exhausted, do NOT start the thread — fail that staging job with a capacity
  error (`StagingError`) and route it through the failure path. The whole point is that a hung
  transport surfaces as a capacity error instead of spawning yet another zombie thread.
- Native adapter timeout/cancellation remains preferable — if `TransferAdapter` exposes a real
  timeout, use it so the thread actually dies and no orphan is created.

**Gate (replaces gate 59):** launch N > cap permanently-blocked fetches. Assert the number of
LIVE `miner-fetch` threads (counted via `threading.enumerate()`, filtered by thread name) stays
≤ cap — NOT merely the registry length. This is Beta's exact probe. Also assert the excess jobs
fail with a capacity error.

---

## Defect 3 — disconnected workers remain eligible for new assignments

`_drop_conn(rawsock, fs_by_sock, worker_by_sock, fs_by_worker)` removes the socket from
`fs_by_sock`/`worker_by_sock`/`fs_by_worker` but NOT from `wconn_by_worker`, `self.connections`,
or `registered`. `serve_trial` builds `_eligible()` from `wconn_by_worker`, so a worker whose
socket is already gone still receives new stripes; `_dispatch_pending` then can't send them,
leaving the stripe claimed until lease expiry. Beta: A disconnected, run_s0 still claimed by A.

**Fix:**
- When a socket is dropped, remove/mark-offline the worker EVERYWHERE the eligible pool is
  built from: `wconn_by_worker`, `self.connections`, and `registered` (or a live/offline flag
  those consult). Extend `_drop_conn`'s signature to receive these (or make them instance
  state it can reach).
- `serve_trial` must construct the assignment pool from CURRENTLY LIVE, registered sockets —
  a worker with no live socket is not eligible.
- If a worker disconnects while a stripe is ALREADY claimed by it, route that stripe through
  the existing phase-specific failure / lease policy (do NOT invent a new path) — but do NOT
  hand it NEW work.
- Guard against the identity subtlety from Defect 3 (C4): dropping A's socket must not evict a
  DIFFERENT live socket that legitimately holds the same worker_id after a fenced replacement.
  Evict by the (socket→worker) mapping that is being dropped, and only clear `wconn_by_worker`
  for that worker_id if the dropped socket is the one currently bound to it.

**Gate:** A and B register; A disconnects (`_drop_conn`) BEFORE assignment; then every new
compatible stripe is assigned to B, and NONE to A. Assert A is not in the eligible pool and no
stripe is left claimed-by-A-and-unsendable.

---

## Packaging (Beta's recurring execution blocker — fix it properly this time)

The archive still wasn't self-contained: it omitted `utils/prng_encoding.py` and
`persistent_worker_coordinator.py`, so Gate 22 (PWC import), Gate 23 (`test_prng_encoding`
importing `utils.prng_encoding`), and Phase-3 (16/17) failed on packaging, not code.

**For the next archive, include EVERY import dependency the harnesses touch.** Determine the
real set — from the repo, run:
```
cd ~/distributed_prng_analysis
python3 - <<'PY'
import ast, os, sys
roots = ["tests/test_s172_phase4_coordinator.py","tests/test_s172_phase3_worker.py",
         "tests/test_prng_encoding.py","tests/test_s172_phase1_scaffolding.py",
         "tests/test_s172_phase2_protocol.py","miner/range_miner_coordinator.py",
         "window_optimizer_integration_final.py"]
# crude: print local modules imported so we can package them
PY
```
or more simply, from a CLEAN checkout in /tmp, run the full suite and add whatever it reports
`ModuleNotFoundError` for until it runs green, then package exactly that set. Known missing:
`utils/prng_encoding.py`, `persistent_worker_coordinator.py`. Verify the full suite runs green
from the /tmp clean copy of the archive BEFORE resubmitting — that is the real test of
"self-contained."

---

## Verify + report

- Full harness green from a CLEAN /tmp extraction of the archive (not just from the live repo)
  — exit 0, including the new D1 gate, the REPLACED gate 59 (live-thread count via
  `threading.enumerate()`), and the new D3 gate. Each new/replaced gate FAILS on current code.
- Phase-3 17/17 from the clean copy.
- Report per defect: the fix, the gate, why it catches the original, and — for D2 — proof the
  gate counts LIVE threads not registry entries. Then STOP. Team Alpha adversarial re-review
  (tracing: 70 MiB remote admission, live-thread bound, disconnected-worker eligibility), then
  Team Beta. Do NOT commit/push.
