# SESSION CHANGELOG — 2026-07-19 — S172 Phase 4 CORRECTION 6 (one coherent admission byte-model)

**Team Alpha (Claude) implementing. Team Beta is the binding approval authority.**
Instructions: `docs/CLAUDE_CODE_CORRECTION6_S172_PHASE4_ADMISSION.md`.
Scope: fix the ONE remaining blocker — admission used TWO contradictory byte models.
Beta confirmed all prior defects fixed (61/61 + 17/17, archive self-contained); this
is the last one. The accepted ledger/retry/resolver/dispatch logic was NOT redesigned.

**Status: harness GREEN from a CLEAN /tmp extraction — Phase 4 = 63/63, Phase 3 =
17/17.** Two new gates (62/63); gate 41 updated + flagged for the new model. Every
new/changed gate verified to FAIL on the pre-C6 (C5) code. NOT committed/pushed;
WATCHER not run.

Files changed: `miner/range_miner_coordinator.py`,
`tests/test_s172_phase4_coordinator.py` (gates 62/63 + gate-41 update).
fallback parity: code=current, env=ok (no dependency change this session).

---

## The defect: two contradictory byte models

- `enqueue_staging`'s per-shard guard used ACTUAL advertised `size_bytes` (C5, correct).
- `_try_admit_locked` → `_attempt_footprint` still budgeted `expected_substripes *
  INLINE_BYTE_LIMIT` (static 48 MiB/file) into `self._admitted` and checked THAT
  against high-water.

The static 48 MiB/substripe estimate is simultaneously too LARGE for tiny inline
results (Beta repro 1: 2 substripes × 48 MiB = 96 MiB > 60 MiB high-water → a
135-byte job permanently deferred, no capacity event can ever fit a static 96 MiB
into 60 MiB) and too SMALL for remote spools (Beta repro 2: 96 MiB estimate admits
BOTH of two 70 MiB×2 attempts under 200 MiB high-water → A.0 + B.0 reserve 140 MiB,
each second shard needs +70 MiB (210 > 200) → both wait forever, neither publishes →
circular wait).

## The fix — ONE coherent model: serialized attempt-level staging (Approach A)

Chose Beta's recommended Approach A (serialize) over B (dynamic dual-budget) — it is
simpler and structurally cannot circular-wait, and the miner's bottleneck is GPU
compute, not staging concurrency (Phase 5 acks release capacity promptly).

- `_try_admit_locked` is now a PURE SERIALIZATION GATE: admit an attempt only if it is
  already admitted, OR no OTHER attempt currently holds partial staging capacity
  (`self._admitted` empty). At most ONE attempt actively stages at a time; a second
  attempt DEFERS (bounded, as before) and resumes when the first completes + publishes
  and its capacity is released via the ack path. `_admitted` maps a live attempt key →
  True; it NO LONGER stores a static byte budget.
- Within the single admitted attempt, per-shard reservation continues to use ACTUAL
  `size_bytes` (the C5 guard). Because only one attempt stages at a time, its shards
  cannot be starved by another attempt's partial occupancy — no circular wait.
- The BYTE decision lives entirely in `enqueue_staging`, driven by ACTUAL advertised
  sizes: a single shard `size_bytes > staging_high_water_bytes`, or an admitted
  attempt whose ACCUMULATED actual bytes (`_attempt_actual_bytes`) exceed high-water →
  fail fast (non-retryable), releasing admission + reservations and routing via the
  matrix. `_attempt_exceeds_highwater` is now a FILES-only sanity guard.
- `_attempt_footprint` (and its `INLINE_BYTE_LIMIT` byte estimate) is REMOVED from the
  admission path; the `INLINE_BYTE_LIMIT` import is removed from the coordinator (it
  now appears only in an explanatory comment, never in the byte decision — the inline
  per-shard ceiling is enforced in `range_miner_worker.py`).

### Why this satisfies Beta's four guarantees
1. **Tiny actual attempt not rejected:** no static estimate; a 135-byte attempt sees
   only its real ~270 bytes ≤ high-water → admitted (gate 62).
2. **Remote not under-budgeted:** admission is serialization, not a 48 MiB estimate;
   the per-shard/accumulated ACTUAL-byte checks catch oversize (gate 60, still green).
3. **No two-attempt circular wait:** only one attempt stages at a time; the second
   never partially occupies capacity, so they cannot wait on each other (gate 63).
4. **Actual total over high-water fails explicitly:** `_attempt_actual_bytes >
   high_water_bytes` (or a single shard over) → fail fast + clean (gates 55/60).

## Mandatory new gates

- **Gate 62** (`gate62_tiny_inline_admission`): 2 expected shards, 60 MiB byte
  high-water, ~100-byte inline payloads → the attempt STAGES and COMPLETES + PUBLISHES;
  `_deferred` stays empty. Pre-fix (C5): the 96 MiB static estimate deferred it FOREVER
  (`.result(timeout)` → TimeoutError).
- **Gate 63** (`gate63_cross_attempt_remote_serialized`): two attempts × two 70 MiB
  remote shards, 200 MiB global high-water, delivered in the poison interleaving
  A.0,B.0,A.1,B.1 → attempt A stages both shards and PUBLISHES (real publish path),
  attempt B WAITS without partially consuming capacity (no B reservation, `reserved_files
  == 2`), no circular wait (A completed). Pre-fix (C5): both admit, A.1/B.1 circular-wait
  forever, published = 0 (A.1's future never resolves → TimeoutError).

Both use the real publish lifecycle (no manual acks of unpublished shards).

## Existing gate whose behavior changed (flagged, not force-passed)

- **Gate 41** (`gate41_slow_fetch_nonblocking`, Defect 4a) — UPDATED for the serialized
  model. The DEFECT it guards (a slow/blocked fetch must NOT stall the dispatch thread)
  is unchanged and still asserted (both `_serve_dispatch` calls return promptly). What
  changed: worker B's result is now DEFERRED while attempt A is actively staging
  (cross-attempt STAGING parallelism is intentionally traded away for no-circular-wait),
  then resumes after A completes + publishes. This is the correct new behavior, flagged
  here. Pre-fix (C5) fails the updated gate (it staged B in parallel).
- Gates **43 / 55 / 56 / 60** (the C3 deadlock gate and the C4/C5 admission gates)
  confirmed STILL PASS under the serialized model — their expectations are unchanged.

## Verification

- Full harness GREEN from a CLEAN /tmp extraction (build tar → fresh extract → run
  there): **Phase 4 = 63/63**, **Phase 3 = 17/17** (also green on the live repo).
- `grep INLINE_BYTE_LIMIT miner/range_miner_coordinator.py` → only a comment; not used
  in any admission byte decision.
- Pre-fix failure confirmed: gates 62 (permanent defer → timeout), 63 (circular wait →
  A.1 future TimeoutError), and the updated 41 (parallel stage) all FAIL on the C5
  coordinator.

## Self-contained archive (unchanged 33-file set from C5)

Only `miner/range_miner_coordinator.py` and `tests/test_s172_phase4_coordinator.py`
changed this round — both already in the C5 dep set — so the tar file list is
IDENTICAL to Correction 5 (33 files). Verified green from a fresh /tmp extraction of
`/tmp/s172_phase4_c6.tar.gz`. The exact list:

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

Next: Team Alpha adversarial re-review (tiny-inline stages, two-remote-attempt
serialization / no-circular-wait, oversized fail-fast still works), then Team Beta.
Do NOT commit/push/run WATCHER.
