# SESSION CHANGELOG — 2026-07-19 — S172 Phase 4 CORRECTION 5 (real-resource bounds)

**Team Alpha (Claude) implementing. Team Beta is the binding approval authority.**
Instructions: `docs/CLAUDE_CODE_CORRECTION5_S172_PHASE4_REALBOUNDS.md`.
Scope: fix THREE narrow, directly-reproducible blockers (Beta: "once these are green
this should finally be at approval threshold") + make the resubmission archive
self-contained. The common theme on two: a gate checked the REGISTRY/bookkeeping but
not the ACTUAL RESOURCE (bytes on disk, live threads), so it passed while the real
thing leaked — every new/replacement gate now asserts the REAL resource. The accepted
ledger/retry/resolver/dispatch logic was **not** redesigned.

**Status: harness GREEN from a CLEAN /tmp extraction — Phase 4 = 61/61, Phase 3 =
17/17.** gate 59 replaced; every new/replaced gate verified to FAIL on the pre-C5 (C4)
code. NOT committed/pushed; WATCHER not run.

Files changed: `miner/range_miner_coordinator.py`,
`tests/test_s172_phase4_coordinator.py` (gate 59 replaced + gates 60–61).
fallback parity: code=current, env=ok (no dependency change this session).

---

## Defect 1 — large remote spools bypassed the oversized check (fix by ACTUAL bytes)

`_attempt_footprint` estimated `need_files * INLINE_BYTE_LIMIT` (48 MiB/file). That
ceiling is valid only for INLINE messages; a REMOTE spool exists because its payload
is at least that large and can be much larger. A 70 MiB remote spool passed a 60 MiB
`_attempt_exceeds_highwater` (48 MiB estimate ≤ 60 MiB), was admitted, then looped
forever on `StagingBackPressure` because the 70 MiB reservation never fit.

**Fix:** the BYTE guard is now driven by ACTUAL advertised sizes, at `enqueue_staging`
where the shard's real size is known:
- `_attempt_exceeds_highwater` is now FILES-only (the byte-estimate was removed).
- A single shard whose advertised `size_bytes` exceeds `staging_high_water_bytes`
  → **fail immediately** (non-retryable capacity error); never admitted or deferred.
- An attempt whose ACCUMULATED advertised bytes (`_attempt_actual_bytes`, summed from
  the ledger's recorded shard `size_bytes`) exceed the byte high-water → fail and
  clean, never admit-then-loop.
- Inline keeps the inline-ceiling semantics (its payload IS bounded by that ceiling).

**Gate 60** (`gate60_large_remote_spool_fails_fast`): a 70 MiB REMOTE result under a
60 MiB high-water → trial fails immediately, the stripe leaves `claimed`/`staging`,
nothing admitted/deferred, and the oversized spool is NEVER fetched (`fetch_calls ==
[]`). Asserts the REAL resource (advertised bytes); the files-only footprint guard is
shown to return None (it can't catch it). Pre-fix: `.result(timeout=5)` raises —
the job dead-loops on back-pressure.

## Defect 2 — orphan-fetch registry was bounded but the real threads were not

`_register_orphan_fetch` RAISED when the registry was full — but the fetch thread was
ALREADY started before that call, so it stayed alive and became UNTRACKED. Cap 2,
seven hung fetches → registry had 2 entries but SEVEN live `miner-fetch` threads. The
old gate 59 only checked `len(_orphan_fetch_threads) <= cap`, so it passed while real
threads accumulated.

**Fix:** enforce capacity BEFORE launching a non-cancellable fetch thread.
`_reserve_orphan_slot` prunes finished threads and admits only if the LIVE count is
below cap — BEFORE `threading.Thread(...).start()`. If the budget is exhausted the
thread is NOT started; the job fails with a `StagingError` (transport capacity). A
thread that completes in time releases its reserved slot (`_release_orphan_slot`); a
timed-out one keeps it (it's a live orphan). Native adapter cancellation is still
preferred (`_adapter_supports_timeout` → pass `timeout` so the thread actually dies).

**Gate 59 REPLACED** (`gate59_orphan_fetch_threads_live_bound`): launch N=7 > cap=2
permanently-blocked fetches; assert the number of LIVE `miner-fetch` threads —
**counted via `threading.enumerate()` filtered by thread name, not registry length**
— stays ≤ cap, and the excess jobs fail with a capacity error (`timeouts == cap`,
`cap_errors == N - cap`; only `cap` threads ever launched). This is Beta's exact
probe. Pre-fix: the gate reports "7 live miner-fetch threads exceeds cap 2".

## Defect 3 — disconnected workers stayed eligible for new assignments

`_drop_conn` removed the socket from `fs_by_sock`/`worker_by_sock`/`fs_by_worker` but
NOT from `wconn_by_worker`, `self.connections`, or `registered`. `serve_trial` builds
`_eligible()` from `wconn_by_worker`, so a worker whose socket was gone still received
new stripes; `_dispatch_pending` then couldn't send them, leaving the stripe claimed
until lease expiry.

**Fix:** `_drop_conn` now evicts the worker identity from EVERY structure the eligible
pool is built from — `fs_by_worker`, `wconn_by_worker`, `self.connections`,
`registered` — guarded so it only fires when the dropped socket is the one CURRENTLY
bound to the worker_id (a fenced replacement that legitimately rebound the same
worker_id to a DIFFERENT live socket is NOT evicted). All three serve-loop callers
(eof, read-deadline, reject-dup-worker) pass the extra maps; the reject-dup socket was
never bound, so the guard leaves the original worker intact. A worker that disconnects
while holding a claimed stripe is handled by the EXISTING lease/matrix policy (no new
path) — it simply gets no NEW work.

**Gate 61** (`gate61_disconnected_worker_not_eligible`): A and B register; A is dropped
BEFORE assignment; A is evicted from `wconn_by_worker` / `connections` / `registered` /
`fs_by_worker`; the eligible pool is `[B]`; both new java_lcg stripes go to B and NONE
is left claimed by the disconnected A. Pre-fix: A remains in `wconn_by_worker`.

---

## Packaging — self-contained archive (Beta's recurring execution blocker, fixed)

The full suite now runs GREEN from a CLEAN /tmp extraction (verified by rebuilding a
brand-new dir from the live repo and running there — not just from the live repo).
The prior archive omitted `utils/prng_encoding.py`, `persistent_worker_coordinator.py`,
`sieve_gpu_worker.py`, the `integration/` package, and a few others, so Gate 22, Gate
23, and Phase-3 failed on PACKAGING, not code. The exact self-contained dep set (33
files) is:

```
adaptive_thresholds.py                 miner/range_miner_worker.py
agent_manifests/window_optimizer.json  persistent/__init__.py
hybrid_strategy.py                     persistent/active_job_state.py
integration/__init__.py                persistent/pwc_protocol.py
integration/coordinator_adapter.py     persistent/pwc_result_normalizer.py
integration/metadata_writer.py         persistent/pwc_transport_base.py
integration/sieve_integration.py       persistent/pwc_transport_ssh.py
miner/__init__.py                      persistent/pwc_transport_tcp.py
miner/range_miner_coordinator.py       persistent/pwc_worker_service.py
miner/range_miner_protocol.py          persistent_worker_coordinator.py
prng_registry.py                       sieve_gpu_worker.py
utils/__init__.py                      window_optimizer.py
utils/prng_encoding.py                 window_optimizer_integration_final.py
utils/survivor_loader.py               zmq_sqlite_coordinator.py
tests/test_prng_encoding.py            tests/test_s172_phase1_scaffolding.py
tests/test_s172_phase2_protocol.py     tests/test_s172_phase3_worker.py
tests/test_s172_phase4_coordinator.py
```

Package EXACTLY this set. Verified: `PYTHONPATH=<extract> python3
tests/test_s172_phase4_coordinator.py` → 61/61; `…test_s172_phase3_worker.py` → 17/17,
from a fresh /tmp copy.

## Verification

- Full harness GREEN from a clean /tmp extraction: **Phase 4 = 61/61**, **Phase 3 =
  17/17** (also green on the live repo).
- Pre-fix failure confirmed per defect: gates 59/60 fail against the full C4
  coordinator (7 live threads > cap 2; back-pressure dead-loop `.result` TimeoutError);
  gate 61 fails under a targeted D3 revert (A stays in `wconn_by_worker`).

Next: Team Alpha adversarial re-review (70 MiB remote admission, live-thread bound,
disconnected-worker eligibility), then Team Beta. Do NOT commit/push/run WATCHER.
