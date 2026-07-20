# Claude Code Instructions — S172 Phase 4 CORRECTION 6 (Beta: one coherent admission byte-model)

**From:** Team Alpha lead
**For:** Claude Code on VM 101, `~/distributed_prng_analysis`, user `michael`
**Date:** 2026-07-19
**Status:** Team Beta REJECTED resubmission 5 — ONE remaining blocker. Beta confirmed all
prior defects fixed, archive self-contained, 61/61 + 17/17. This is the last blocker. Do
NOT start Phase 5. Do NOT commit/push.

---

## The single defect: two contradictory byte models

Admission uses TWO incompatible byte estimates:
- `enqueue_staging` per-shard guard uses ACTUAL advertised `size_bytes` (correct, from C5).
- `_try_admit_locked` → `_attempt_footprint` still budgets an attempt as
  `expected_substripes * INLINE_BYTE_LIMIT` (static 48 MiB/file) and reserves THAT against
  high-water in `self._admitted`.

These cannot coexist. Beta reproduced both failure directions:

**Repro 1 (tiny inline falsely deadlocks):** 2 expected substripes, 60 MiB high-water,
135-byte payloads → `_attempt_footprint` = 96 MiB > 60 MiB → attempt permanently deferred,
no capacity event can ever make a static 96 MiB fit 60 MiB. A hundreds-of-bytes job never
starts.

**Repro 2 (two remote attempts circular-wait):** 200 MiB high-water, attempt A = two 70 MiB
remote shards, attempt B = two 70 MiB remote shards. Static estimate = 96 MiB/attempt, so
BOTH admit. First shard of each reserves → 140 MiB held. Each second shard needs +70 MiB
(140+70 > 200) → both wait forever; neither completes, neither publishes, no ack releases the
140 MiB. Circular wait.

Root cause: static 48 MiB-per-substripe admission is simultaneously too LARGE for small
inline results and too SMALL for remote spools (which exist precisely because they exceed the
inline ceiling).

---

## Required correction — ONE coherent byte model

Admission must use a single byte model driven by ACTUAL advertised shard sizes, and must
guarantee an admitted attempt can complete its WHOLE remaining footprint without circular
wait. Beta's four guarantees:

1. a tiny actual attempt is NOT permanently rejected because of `INLINE_BYTE_LIMIT`;
2. a remote attempt is NOT admitted using an underestimated 48 MiB-per-file budget;
3. two partially-staged attempts cannot wait on one another indefinitely;
4. an attempt whose ACTUAL total exceeds high-water fails explicitly and cleans up.

Beta offers two acceptable approaches. **Pick the simpler, safer one unless you find a
concrete reason not to: serialize attempt-level staging.**

### Recommended: serialize attempt-level staging (approach A)

The core problem is *partial interleaving* of multiple attempts causing circular wait. The
cleanest guarantee: **at most one attempt may be actively staging (partially occupying
staging capacity) at a time.** A second attempt is not admitted to begin staging until the
first attempt is complete AND published (its capacity released via the ack path).

Concretely:
- Replace the static-footprint admission with an attempt-level GATE: `_try_admit_locked`
  admits an attempt only if NO other attempt currently holds partial staging capacity (i.e.
  `self._admitted` is empty OR already contains this exact attempt key). While one attempt is
  admitted-and-incomplete, others DEFER (bounded, as now) and resume when it releases.
- Within the single admitted attempt, per-shard reservation continues to use ACTUAL
  `size_bytes` (the C5 guard). Because only one attempt stages at a time, its shards can't be
  starved by another attempt's partial occupancy — no circular wait.
- Keep the fail-fast checks driven by actual bytes: a single shard `> high_water_bytes`, or an
  admitted attempt whose ACTUAL accumulated bytes exceed high-water, fails explicitly and
  cleans up (releases admission + reservations, routes via matrix). This satisfies guarantee 4
  and handles the "even one attempt can't fit" case.
- Remove `_attempt_footprint`'s `INLINE_BYTE_LIMIT` estimate from the admission path entirely
  (or keep only a files-count sanity guard; the BYTE decision must come from actual advertised
  sizes, never the static 48 MiB). Guarantee 1 (tiny inline) and 2 (remote under-estimate) both
  fall out of using actual bytes + single-attempt serialization.

This trades a little cross-attempt parallelism for correctness — acceptable, since Phase 5 acks
release capacity promptly on attempt completion, and the miner's throughput bottleneck is GPU
compute, not staging concurrency.

### Alternative (approach B) — only if you deliberately want cross-attempt parallelism

Maintain an atomic, dynamically-expandable per-attempt budget based on actual advertised shard
sizes, and refuse to admit a second attempt unless the SUM of both attempts' worst-case
remaining actual footprints fits under high-water (so partial interleaving can never create
circular wait). This is more complex and easier to get subtly wrong; use A unless there's a
concrete throughput reason.

Whichever you choose, `_admitted` must track ACTUAL bytes (or be a pure serialization gate),
never the static estimate.

---

## Mandatory gates (Beta-specified)

**Gate — tiny-inline admission:** 2 expected shards, 60 MiB byte high-water, ~100-byte
payloads → the attempt STAGES and COMPLETES; no indefinitely-deferred future. (Pre-fix: 96 MiB
static estimate → permanently deferred.)

**Gate — cross-attempt remote admission:** two attempts × two 70 MiB shards, 200 MiB global
high-water → ONE attempt makes forward progress and PUBLISHES; the other WAITS without
partially consuming capacity; no circular wait. (Pre-fix: both admit, 140 MiB held, both second
shards wait forever, published = 0.)

Both gates must FAIL on the current code and pass on the fix, using the REAL lifecycle
(publish/ack, no manual acks of unpublished shards).

Do NOT weaken any existing gate. Confirm the existing admission gates (55/56/60 and the C3
deadlock gate 43) still pass under the new model — if the new model changes their expected
behavior, update them to the correct new behavior and flag it, do not force-pass.

---

## Verify + report

- Full harness green from a CLEAN /tmp extraction (build tar → extract fresh → run there),
  including the two new gates. Phase-3 17/17 from the extraction.
- Confirm the admission path no longer references `INLINE_BYTE_LIMIT` for the BYTE decision
  (grep it — it should appear only for the inline per-shard ceiling, not attempt admission).
- Give the exact self-contained tar file list again (it may be unchanged from C5 + these
  edits).
- Update the changelog: "Correction 6" — the one coherent byte model, which approach (A or B)
  and why, the two new gates, and any existing gate whose expected behavior changed under the
  new model.

Report: the chosen model, why it satisfies all four Beta guarantees, the two gates + why they
catch the original, and the clean-extraction result. Then STOP. Team Alpha adversarial
re-review (tracing: tiny-inline stages, two-remote-attempt serialization/no-circular-wait,
oversized-fail-fast still works), then Team Beta. Do NOT commit/push.
