# PLAN — next rig arms V1 (cold cache, run #2) and V2 (warm cache, run #1)

**Status:** V2 approved by the owner and running as Run C (2026-11-16 ~17:4x). V1 is prepared, not run.

## Prior record

- `docs/LEADS.md` L-7: these arms are its open follow-ups. **ALREADY RECORDED.**
- Beta X-GCVM-LOCATED R1 Q6: prefers a same-boot cold-cache B. That is V1.
- Register E30 (`22bb145`, an in-process "pre-compile" that, by reading, compiled nothing) and the
  "warm-up barrier didn't help" entry (01-23). Neither tested a warm on-disk cache on a first run, so V2 is
  **NEW**.

**Not searched:** ser8, the private repository.

## What separates the two

| | Run #1 after boot (no earlier run's driver state) | Run #2+ (state carried from an earlier run) |
|---|---|---|
| cold cache (compile, ~0.3–0.7 s load) | A, 11-15 attempts 1 and 3: **clean ×3** | **V1** |
| warm cache (cache hit, ~1 ms load) | **V2 = Run C** | B, 11-15 attempt 2, 11-15 confirm 1: **fault ×3** |

**Readings:**

| Arm | Outcome | Reading |
|---|---|---|
| V2 | faults | the carried state is not necessary; the warm fast path is sufficient |
| V2 | clean | the earlier run's carried state is implicated |
| V1 | faults | the warm fast path is not necessary |
| V1 | clean | the fast path is implicated |

All of these are single events per arm. Beta's caution applies: recurrence is probabilistic, so repeat
before reading a clean run as exclusion.

## Cache portability (answers "are entries portable, keyed how?")

- **The key.** CuPy 13.5.1 `cupy/cuda/compiler.py:873-945` (read on the CT) names each cache file
  `sha1(key_src).hsaco`. `key_src` is built from:
  - `env = (arch, options, hiprtc version, backend)` (`:912`);
  - the preprocessed empty-file base;
  - the full kernel **source** and the extra source (`:922`).

  **The key contains no device index, PCI address, UUID, host name or path.**
- **Measured.** On rrig6600b, all 8 worker caches hold the same 8 file names with byte-identical contents.
  The phase-1/2 entries (`190aa0bf…`, `757a409d…`) compiled independently in July/August are byte-identical
  across all three CTs' persistent caches.
- **Integrity.** Each file is a 40-byte SHA-1 of the binary followed by the binary. On a hit CuPy verifies
  that hash before `mod.load` (`:937-945`); on a mismatch it recompiles and rewrites the file. Mtimes are not
  consulted.
- **Conclusion.** Entries are portable between workers and between rigs, provided all of these hold:
  - same arch (`gfx1032`, confirmed in all 8 binaries);
  - same CuPy version;
  - same hiprtc/ROCm version (6.4.3 on all CTs);
  - identical kernel source (same commit, parity-gated).

## Files to copy (V2, as done by Alpha for Run C)

- **Source:** rrig6600b CT `.156` `/tmp/cupy_cache_gpu{0..7}/`: 8 files per directory, 64 in all, identical
  sets.
- **Target:** `.122` and `.164` CTs, `/tmp/cupy_cache_gpu{0..7}/`.
  - Create each directory with `mkdir -p` first, as `michael`, mode 0775 like the originals.
  - All 8 `.hsaco` files go into **each** of the 8 directories, because the launcher sets
    `CUPY_CACHE_DIR=/tmp/cupy_cache_gpu$N` per worker (`scripts/launch_fleet_manual.sh:327-329`).
- **The eight entries:**

  | Entry | Kernel |
  |---|---|
  | `df5f754f…` | arange u32 (`warm_gpu`) |
  | `7eadd335…` | add u32 (`warm_gpu`) |
  | `42ec350d…` | sum (`warm_gpu`) |
  | `9c7ba1c0…` | arange u64 (seeds) |
  | `757a409d…` | `java_lcg_flexible_sieve` |
  | `190aa0bf…` | `java_lcg_reverse_sieve` |
  | `f72c9211…` | `java_lcg_hybrid_multi_strategy_sieve` |
  | `46d35db4…` | `java_lcg_hybrid_reverse_sieve` |

- **Check:** the same combined checksum on all three CTs. Measured before Run C: `59ec272b9863654a` by
  `find cupy_cache_gpu* -name '*.hsaco' | sort | xargs sha256sum | sha256sum`; Alpha's `7f14fae2…` by
  another method; identical on all three CTs.
- **Survival:** `/tmp` is cleared at every CT start, so the copy must be redone after each boot. A persistent
  copy is kept in `logs/rig_gpu_fault_20261115/offline_1908/ct_192.168.3.156.tar` (11-15). Its phase-4 entry
  is the same `46d35db4…` name.

## V1 — what it needs

1. A boot on which one full run has already happened. The first run of the next boot qualifies; so does Run
   C's boot, if no reset follows it.
2. Before launch, on each CT: `rm -rf /tmp/cupy_cache_gpu*` (CT write; owner/Alpha). Confirm that zero
   `.hsaco` files remain.
3. The recorder:
   - armed on all three hosts (re-arm after any reboot);
   - stream running, with **one** reader per host (see the incident below);
   - T0, then `auto --out runV1`.
4. The same launch as B: new search record, `RIG_CLEANUP_TORCH=off`, fresh nonce.
5. Expected signature of a cold path: phase-4 code map ~300–700 ms after dispatch. If it is, the arm is valid.

## V2 (Run C) — what it needed

Done by Alpha:
- fresh boot of `.121`/`.163`, with no run on this boot;
- caches copied as above;
- tracing armed on `.121`/`.163`; `.155` still armed;
- T0 retaken with markers.

Expected signature of a warm path: phase-4 code map ~10–30 ms after dispatch and buffer→code ~1 ms on `.122`
and `.164`. Run C's report checks this.

## Recorder incident to fix before V1

- **What happened:** the `stream` reader's remote `cat trace_pipe` survives when the local reader is stopped.
  An orphan on `.155` (pid 96263, from 16:16) held `trace_pipe`, so new readers got EBUSY. I stopped it by
  exact PID at 17:40. Events on `.155` between ~17:08 and 17:40 are lost; the host was idle.
- **Proposed fix (not applied):** run the remote reader under a parent that exits when SSH closes, e.g.
  `exec timeout --foreground` with a heartbeat, or a `trap` on SIGPIPE. Add `disarm`/`stream --stop`
  handling that kills its own remote reader by recorded PID.
