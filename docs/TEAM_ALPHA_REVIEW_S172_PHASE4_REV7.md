# Team Alpha Review Record — S172 Phase 4 (RANGE-MINER coordinator) — REV 7
**Reviewer:** Team Alpha (lead dev)
**Date:** 2026-07-19
**Verdict:** PASS — addresses the last binding blocker; ready for Team Beta final re-review.
**Supersedes:** rev-6. Rev-7 covers Correction 6 — the single admission byte-model defect
(the only blocker in Beta's most recent verdict, which confirmed everything else fixed).
**Method:** file-vs-source, adversarial. Traced both of Beta's reproductions against source:
the tiny-inline false deadlock and the two-remote-attempt circular wait.

---

## The single defect and the fix

Admission used two contradictory byte models: `enqueue_staging` checked ACTUAL advertised
`size_bytes`, but `_try_admit_locked` → `_attempt_footprint` budgeted
`expected_substripes × INLINE_BYTE_LIMIT` (static 48 MiB/file) against high-water. That
static estimate is simultaneously too large for tiny inline results (false deadlock) and too
small for remote spools (real cross-attempt deadlock).

**Fix (Beta's recommended Approach A — serialized attempt-level staging), verified at source:**
`_try_admit_locked` is now a PURE SERIALIZATION GATE — it admits an attempt iff that exact
attempt is already admitted OR no OTHER attempt is currently staging (`self._admitted` empty);
`self._admitted[key] = True` holds a live-attempt marker, never a byte budget. At most one
attempt stages at a time; a second defers (bounded) and resumes when the first
completes + publishes and its capacity releases. Per-shard reservation continues to use ACTUAL
`size_bytes` (the C5 guard). `INLINE_BYTE_LIMIT` is removed from the admission path entirely
(a source grep shows it now appears only in one explanatory comment). The byte fail-fast
(single shard > high-water, or an admitted attempt's ACCUMULATED actual bytes > high-water →
fail fast + clean) lives entirely in `enqueue_staging` on real sizes.

Release path (so the deferred attempt resumes): `finalize_stripe` → `_release_admission`
(`self._admitted.pop(key)`) → `_pump_deferred` (coordinator.py ~2285 / ~2118).

---

## Why it satisfies Beta's four guarantees

1. **Tiny attempt not rejected** — no static estimate exists; a 135-byte attempt sees only
   its real bytes and is admitted. **Gate 62:** 2 expected shards, 60 MiB high-water,
   ~100-byte payloads → stages, completes, publishes; `_deferred` empty. (Pre-fix: 2 × 48 MiB
   = 96 MiB > 60 MiB → deferred forever, `.result()` → TimeoutError.)
2. **Remote not under-budgeted** — admission is serialization, not a 48 MiB estimate; actual
   per-shard byte checks still catch oversize (gate 60, still green).
3. **No two-attempt circular wait** — only one attempt stages at a time; the second never
   partially occupies capacity. **Gate 63:** two attempts × two 70 MiB shards, 200 MiB
   high-water, POISON interleave (A.0, B.0, A.1, B.1) → attempt A stages BOTH shards
   (`SH_VERIFIED`), completes via the REAL `record_stripe_complete` → `finalize_stripe`, and
   publishes 2 manifests; attempt B holds ZERO reservations (`get_reservation_by_event(...)
   is None` for both B shards; `reserved_files() == 2` — only A's). No circular wait. (Pre-fix:
   static 96 MiB estimate admits both; A.0+B.0 hold 140 MiB; each second shard needs +70 MiB
   (210 > 200) → both wait forever, published = 0.)
4. **Actual-over-high-water fails explicitly** — `enqueue_staging` byte guard on accumulated
   actual bytes → fail fast + clean (gates 55/60).

---

## Scope reviewed (Correction-6 delta)

| File | Change |
|---|---|
| `miner/range_miner_coordinator.py` | `_try_admit_locked` → pure serialization gate; `_attempt_footprint` / `INLINE_BYTE_LIMIT` removed from admission; `_attempt_exceeds_highwater` now files-only (byte decision moved to `enqueue_staging` on actual sizes). |
| `tests/test_s172_phase4_coordinator.py` | Gates 62, 63 added; gate 41 updated (B now serializes rather than staging in parallel — flagged as correct new behavior, not a force-pass). Harness now 64 `_check` (63 gates + subprocess non-regression). |

Harness: **63/63 green from a CLEAN /tmp extraction** (build tar → extract fresh → run),
Phase-3 17/17 from the same extraction. Both new gates verified to FAIL on the pre-C6 (C5)
code (gate 62 → C5 defers forever; gate 63 → C5 circular-waits, published = 0).

Only 2 files changed, both already in the C5 self-contained set, so the 33-file archive list
is identical to C5 plus this changelog + this review.

---

## Cross-cutting confirmations

- The accepted ledger / retry-matrix / resolver / dispatch / staging-fence logic
  (rev-1..rev-6) was NOT redesigned; this is one localized change to the admission model.
- No existing gate weakened. Gate 41's expected behavior legitimately changed under
  serialization (B defers instead of parallel-staging) and was updated + flagged, not
  force-passed. Gates 43/55/56/60 pass unchanged.
- The two contradictory byte models are eliminated: admission is serialization, all byte
  decisions use actual advertised sizes.

## Open items for Michael before commit (housekeeping, non-blocking)

- `python3_with_venv.sh` stays out of the Phase-4 commit.
- COMMIT set = the actual Phase-4 deliverables changed across this whole effort
  (`range_miner_coordinator.py`, `range_miner_worker.py`, the phase-1/3/4 harnesses,
  `window_optimizer_integration_final.py`) + the Phase-4 changelog(s). The 33-file archive is
  for Beta's EXECUTION, not the commit set — do not commit unchanged deps that are already in
  the repo.

## Standing

Team **Alpha** pass — adversarial file-vs-source verification that the single admission
byte-model defect is fixed (serialized attempt-level staging, all byte decisions on actual
sizes) and gate-covered by Beta's two reproductions. NOT the binding gate. Sequence: **Team
Beta final binding re-review → Michael commits + dual-pushes.**
