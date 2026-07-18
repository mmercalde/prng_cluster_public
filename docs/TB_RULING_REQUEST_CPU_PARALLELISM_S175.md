# TB Ruling Request — Host-CPU parallelism for RANGE-MINER survivor collection / NPZ assembly
**Session:** S175 (for Phase 4/5 scoping)
**Author:** Team Alpha
**Date:** 2026-07-18
**Priority:** P1 — high-survivor throughput; decision needed BEFORE Phase 4 coordinator design
**Related:** `TB_RULING_REQUEST_IPC_SERIALIZATION_S150.md` (prior ruling on the same
high-survivor bottleneck, PWC side), `S172_SIEVE_PATH_VERIFICATION_SCOPE.md`, spec §12.1
(EXPECTED_NPZ_KEYS contract wall), §4.4 (22-array NPZ consumer surface).

---

## Question

For the RANGE-MINER coordinator (Phase 4) and NPZ write-back (Phase 5), should
host-side **survivor collection + 22-array NPZ assembly** be parallelized across
the coordinator host CPU (Zeus i9-9920X, 12C/24T), e.g. via `multiprocessing`, or
should it inherit the existing **single-threaded, GIL-bound assembly + incremental
flush (S152)** pattern?

This affects whether PWC's high-survivor throughput collapse is *removed* by the
miner or merely *relocated* from the transport layer to the assembly layer.

---

## Background — what is and isn't parallel today (code-verified)

**Dispatch IS parallel (not the concern).** `coordinator.py` uses
`ThreadPoolExecutor` (lines 1231, 1897, 2079) + `threading.Thread` worker loops
(1645, 1757) to dispatch GPU jobs across nodes. Those threads `ssh → python
worker` and wait — I/O-bound, so the GIL is released and threading is the correct,
effective choice for keeping 26 GPUs fed. No change requested here.

**Assembly is NOT parallel (the concern).** `convert_survivors_to_binary.py`
builds the 22-array NPZ as **~22 sequential Python list comprehensions**, one per
array, each walking the entire survivor list:
```python
seeds        = np.array([s['seed']              for s in survivors], dtype=np.uint32)
window_size  = np.array([s.get('window_size',0) for s in survivors], dtype=np.int32)
score        = np.array([s.get('score',0.0)     for s in survivors], dtype=np.float32)
# ... 22 arrays total, each a full pure-Python pass over every survivor
```
This is single-threaded, GIL-bound Python. The terminal `np.savez[_compressed]`
releases the GIL, but the **record marshaling — the part that scales with survivor
count — runs on one core** of the i9's 24 threads.

**The incumbent pull-coordinator (ZMQ+SQLite) also collects serially — code-verified.**
`zmq_sqlite_coordinator.py` is the closest existing analog to the Phase 4 miner
coordinator (pull-based, SQLite `pending→claimed→done/failed` ledger, per-chunk
`.npz` payload files + `result_path` in the ledger). Its result-loading is a
**serial loop**:
```python
for row in rows:
    data = np.load(npz_path, allow_pickle=True)
    results.append({...})
```
So single-threaded collection is the **incumbent pattern the miner would inherit
by default** — this is not a hypothetical; it is what the existing pull
coordinator does.

**Existing mitigations (both are chunking, not parallelism):**
- `_flush_npz_incremental` (S152) — flushes NPZ per trial to bound peak assembly
  size, still single-core.
- S159 moved payloads out of the DB into per-chunk `.npz` files (`result_path`
  in the ledger) — the direct ancestor of the miner spool. S159B found
  **`np.savez` uncompressed is 71× faster than `savez_compressed`** — a hard-won
  perf lesson that must carry into the miner's spool/NPZ path.

## Why this matters now (PWC history → miner risk)

Per S150, PWC's high-survivor degradation was **27×** (~1.99M s/s low-survivor →
~73K s/s high-survivor), and S150's root cause was explicitly *IPC result
serialization* in the worker→coordinator path — NOT GPU, NOT assembly. PWC's stable
record was ~1.86–2.0M s/s at pool=6 (~90% of the ~2.07M s/s ceiling).

RANGE-MINER's spool design (Phase 3, byte-exact `s172_substripe_v1`, size-based
inline/spool under `MAX_FRAME_BYTES`) **directly removes the S150 transport
bottleneck** — large results go to disk (`/dev/shm/prng/miner` preferred), only a
reference crosses the wire. Good.

**But the coordinator still must collect, sha256-verify, and assemble every
survivor into the 22-array NPZ.** If that assembly stays single-threaded, the
high-survivor bottleneck is not eliminated — it moves from the socket (PWC) to a
single CPU core (miner). On a genuine high-survivor flood, one-core marshaling of
millions of survivor records could become the new ceiling, capping the throughput
the spool design was meant to protect.

We do NOT yet have a high-survivor throughput number for the miner — Phase 3
tested contract correctness, not performance. So this is a design decision to make
before Phase 4, not a measured defect.

## Options

**Option A — `multiprocessing` pool for verification + assembly (recommended for eval).**
Shard survivors across worker processes (bypassing the GIL): each process
sha256-verifies its spool shard and builds partial arrays; parent concatenates and
writes the NPZ. Pros: uses the i9's 24 threads for the CPU-bound part; scales with
survivor count. Cons: process-spawn + IPC/pickle overhead (ironic given S150 —
must confirm it doesn't reintroduce a serialization cost); concatenation still
serial; interacts with the §12.1 contract wall (validation must run once on the
assembled NPZ, or per-shard-then-final).

**Option B — inherit single-threaded assembly + incremental flush (S152).**
Keep the current pattern; rely on incremental per-trial flush to bound peak
assembly size. Pros: simplest; preserves the proven §12.1 validation path; zero
new failure modes. Cons: does not use the 24 threads; if per-trial survivor counts
are large, the single-core marshal is still the ceiling.

**Option C — GIL-releasing vectorization without multiprocessing.**
Replace the 22 per-field Python comprehensions with a structured-array / pandas
bulk parse that pushes the per-record loop into C (GIL-released), staying
single-process. Pros: big constant-factor win without process overhead; keeps one
validation path. Cons: still one core (no 24-thread scaling), but may be *enough*
if the constant factor dominates; requires the survivor records arrive in a
bulk-parseable form (the spool `s172_substripe_v1` JSON already is structured).

## What we need ruled

1. Which option (A/B/C, or a staged combination — e.g. C first, A only if
   benchmarked-insufficient) for Phase 4/5.
2. Whether the §12.1 EXPECTED_NPZ_KEYS contract wall must run once on the final
   assembled NPZ (implies assembly completes before validation) or may run
   per-shard — this constrains A.
3. Whether **high-survivor throughput** must be an explicit Phase 6/7 acceptance
   dimension (a deliberately high-survivor trial with measured s/s), so the
   chosen option is validated against the exact case PWC failed — rather than
   Phase 6 proving byte-identity only on a low-survivor happy path.

## Additional finding — all prior distribution paths remain viable (verification asset)

The S172 spec keeps every prior distribution engine as a live, flag-selectable
path on the coordinator: `use_persistent_workers` (PWC), `use_zmq_sqlite` (ZMQ),
and the new `use_range_miner` — none are deleted; RANGE-MINER is additive
(spec lines 125-127). This is normally framed as a safety property (old path
untouched). It is also a **verification and benchmarking asset that should be used
deliberately, not left implicit:**

1. **Correctness oracle for Phase 6.** Phase 6's byte-identity check currently
   compares miner output against "the PWC path." But ZMQ+SQLite is *also* a
   proven path producing the same 22-array NPZ contract. Running the SAME input
   through PWC, ZMQ, AND range-miner gives a **three-way cross-check** — if all
   three produce byte-identical NPZs, correctness confidence is far higher than a
   single pairwise comparison, and any two-vs-one divergence localizes the fault.

2. **Throughput benchmark baseline.** The prior paths have *known, recorded*
   throughput numbers (PWC: ~1.99M s/s low-survivor peak / ~1.86M sps stable
   pool=6 / ~73K s/s high-survivor collapse — S150). These are ready-made
   baselines to benchmark range-miner against on the SAME hardware and SAME
   trials — including, critically, the **high-survivor case** where PWC is known
   to collapse. Range-miner's whole thesis (spool removes the IPC bottleneck) is
   directly testable by re-running PWC's high-survivor trial under the miner and
   comparing s/s.

3. **Regression guard.** Because the old paths still run, a range-miner
   regression can always be bisected against a known-good engine on demand — not
   just against git history.

**Requested as part of this ruling (or noted for Phase 6 scoping):** that Phase 6
acceptance explicitly use the surviving PWC/ZMQ paths as (a) additional
byte-identity oracles and (b) throughput baselines, *including a high-survivor
trial*, rather than validating range-miner in isolation. The prior paths were
hard-won; keeping them as live comparators turns that sunk cost into ongoing
verification leverage.

## Recommendation (Team Alpha, non-binding)

Stage it: **C first** (vectorize the assembly, single-process, cheap, preserves the
validation path), and add a **high-survivor benchmark to Phase 6/7 acceptance**. Go
to **A** (multiprocessing) only if the benchmark shows the single-core assembly is
the ceiling. Avoid committing to A blind, since S150's lesson is that naive
serialization/IPC can itself become the bottleneck — process-pool pickle overhead
must be measured, not assumed away.
