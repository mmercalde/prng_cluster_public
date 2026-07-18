# SESSION_CHANGELOG — 2026-07-17 — S172 Phase 3 rev-2 (Team Beta fix brief)

**Team Alpha implementation. NOT committed/pushed — Michael commits after Team
Beta re-approves.** Ref: `docs/S172_PHASE3_FIX_BRIEF.md` (Beta APPROVED w/ 3
clarifications). Phase 3 rev-1 was REJECTED; this session implements all five
release-blockers and the three clarifications exactly as ruled.

## Files changed
- `miner/range_miner_worker.py` — reworked for all five blockers.
- `tests/test_s172_phase3_worker.py` — 8 original gates + 6 new blocking gates (14 total).

## Audited hybrid ABIs (read from LIVE prng_registry.py — NOT extrapolated)
Every hybrid kernel signature was extracted from its `kernel_source` string and
verified before writing the builder. **No STOP condition triggered** — all 24
covered concrete variants exist in `KERNEL_REGISTRY` with kernel_name +
kernel_source, and all are satisfiable. No Route-A erratum needed.

Forward-hybrid common 13-prefix (all): seeds, residues, survivors, match_rates,
skip_sequences, strategy_ids, survivor_count, int32(n_seeds), int32(k),
strategy_max_misses, strategy_tolerances, int32(n_strategies), float32(threshold).
Family FORWARD tails (verified):
- java_lcg_hybrid   : uint64 a, uint64 c                      → 15 (no offset)
- lcg32_hybrid      : uint32 a, uint32 c, uint32 m, int32 off → 17
- minstd_hybrid     : uint32 a, uint32 m_val                  → 15 (no offset)
- pcg32_hybrid      : uint64 increment, int32 offset          → 15
- xorshift32_hybrid : int32 shift_a, shift_b, shift_c         → 16 (no offset)
- xorshift128_hybrid: int32 dummy1, dummy2, dummy3            → 16 (no offset)
REVERSE hybrids (ALL families identical): 13-prefix + int32(offset) → 14.
seed_type uint64 ONLY for java_lcg; the other five are uint32 (read per family).

**Empirical ABI validation:** all 12 hybrid variants (6 fwd + 6 rev) COMPILED and
LAUNCHED on the real 3080 Ti with zero arg-count/type errors and produced
survivors — a wrong ABI would make CuPy raise at launch.

## The five blocker fixes
- **B1 — per-assignment residue window (CORRECTNESS).** Removed the process-
  lifetime `self.draws`. `ResidueResolver` resolves the window per assignment from
  `StripeAssignMessage.payload`, keyed on
  `(dataset_reference, dataset_sha256, window_size, canonical_sessions, offset)`.
  Sessions canonicalized (sorted tuple); dataset CONTENT fingerprint always
  computed (never path-only, fresh load — no stale pathname cache); a coordinator
  `residue_sha256` takes precedence for keying AND is verified (mismatch →
  `stripe_error` retryable=False). Missing window fields → `ResidueResolutionError`
  (retryable=False), never silent stale data.
- **B2 — real atomic spool (DATA LOSS + frame overflow).** Byte-exact schema
  `s172_substripe_v1`; `payload_bytes = json.dumps(obj, sort_keys=True,
  separators=(",",":"))`; size/sha256 taken over `payload_bytes` (inline mode
  carries the SAME logical object). Inline/spool chosen by MEASURED framed-message
  size ≤ 48 MiB (16 MiB headroom under the 64 MiB cap), NOT survivor count. Atomic
  write: temp in same dir → fsync → `os.replace`; worker removes abandoned temp;
  output dir `--miner-output-dir` / auto `/dev/shm/prng/miner` → `~/miner_output`.
- **B3 — exception-safe full cleanup (STABILITY).** All GPU alloc/launch/extract
  wrapped in `try/finally`; the finally does guarded per-array `del` then a shared
  `_best_effort_gpu_cleanup()` (gc + torch sync/empty_cache + CuPy default & pinned
  pool `free_all_blocks`, each guarded) — replicating sieve_gpu_worker.py:78-94 /
  :331-348. Runs after EVERY sub-stripe, success or exception.
- **B4 — Route B non-Java hybrid builders (SPEC-REQUIRED).** `_reject_hybrid`
  removed; all 6 covered families build all 4 variants (constant/_reverse/_hybrid/
  _hybrid_reverse) with the audited per-family ABIs. `COVERED_FAMILIES` unchanged
  (all six). Handshake advertises EXACT concrete variants via `supported_variants()`:
  an explicit `SUPPORTED_VARIANTS` table, each variant validated against BOTH
  `KERNEL_REGISTRY` and a working builder branch; the list is the sorted union of
  validated variants (24). A missing/malformed variant raises `VariantStopCondition`
  (proposed Route-A erratum) rather than silently disappearing.
- **B5 — tests for the dangerous paths.** New blocking gates 9-14 (below).

## Harness — 14/14 gates green
Run under `~/venvs/torch` on VM 101 (cupy 13.5.1, 3080 Ti), exit 0. GPU gates 7
and 12 ran for real. Also 14/14 under system `python3` with cupy absent (gates 7
& 12 skip cleanly) — CPU contract-parity.
New gates: 9 [B1] two-window correctness + residue_sha verify + missing-field
fail; 10 [B2a] spool file written, path/size/sha256 set, inline=None, re-hash
matches; 11 [B2b] size-based selection (5 huge survivors spool, 5000 tiny inline —
proves size not count); 12 [B3] forced launch exception still runs cleanup hook +
daemon serves next assignment; 13 [B4] register advertises 24 concrete variants
incl. hybrids, no uncovered claimed; 14 [B4/§11.I] lcg32 + minstd test_both_modes
4-phase dispatch through correct builders + cap tiers.
Harness output states plainly: CPU-only green = contract-validation only, not
ROCm deploy-readiness (Phase 6).

## Kept from rev-1 (Beta-approved, unchanged)
Declarative dtype-tagged builders; uint64 Java ABI preserved through
materialization; uncovered-family hard-fail before cupy/launch; sub-stripe
partitioning; serialized writes + heartbeat thread; coordinator-retry kept OUT of
the worker (worker only reports `stripe_error`).

## Fallback parity
code=current, env=ok (no new deps; uses existing cupy + sieve_gpu_worker /
prng_registry already on VM 101).
