---
Status: Binding Fix Brief (Superseded by rev-3)
Phase: S172 Phase 3
Applies to: range_miner_worker.py
Superseded by: PHASE3_FINAL_APPROVAL_REV3.md
---

# S172 Phase 3 — FIX BRIEF v2 (Team Beta APPROVED w/ 3 clarifications, rev-2)

**For:** Claude Code on VM 101 as `michael`, `/home/michael/distributed_prng_analysis`.
**Status:** Phase 3 rev-1 was REJECTED by Team Beta. This brief specifies the
required changes. Beta's ruling is binding — implement it, do not relitigate.
**Read first:** the rev-1 files (`miner/range_miner_worker.py`,
`tests/test_s172_phase3_worker.py`), and the ABIs below (audited from the LIVE
`prng_registry.py` — do NOT extrapolate).

Beta approved: declarative builders, uint64 Java ABI preservation, uncovered-family
hard-fail before cupy/launch, sub-stripe partitioning, serialized writes +
heartbeat thread, coordinator-retry kept out of the worker. **Keep all of that.**

Five blockers must be fixed. All are release-blocking. Each needs a blocking test.

---

## Blocker 1 — per-assignment residue window (CORRECTNESS)

**Defect:** `SieveExecutor` loads `draws` ONCE at process start (`main()` →
`load_draws_cached`) and every `stripe_assign` reuses `self.draws`. A long-lived
worker serves different Window Optimizer trials where `window_size`, `sessions`,
`offset` — and thus the residues — change. The payload carries thresholds but
cannot replace the window. This silently runs a valid kernel against WRONG-TRIAL
data. Worse than a crash.

**Fix:** resolve/validate the residue window **per assignment**, keyed by the
window identity carried on `StripeAssignMessage.payload`. Concretely:

- **Cache identity must include dataset CONTENT identity (Beta clarification 1).**
  A path-only key is unsafe: the file at that path can change while the daemon
  is alive, and a newly-ingested draw would reuse a stale cached window. Key on:
  ```
  (dataset_reference, dataset_sha256, window_size, canonical_sessions, offset)
  ```
  Rules (all mandatory):
  - `sessions` MUST be canonicalized (e.g. a sorted tuple) before it enters the key.
  - A coordinator-provided `residue_sha256` (on the payload) TAKES PRECEDENCE: if
    present, key/verify on it.
  - If only a dataset path is supplied, compute or validate a dataset content
    fingerprint (`dataset_sha256`) — never hit the cache on pathname alone.
  - When `residue_sha256` is supplied, the loaded residue sequence MUST be
    verified against it; mismatch → `stripe_error` (retryable=False).
- On each assignment: read the window params from `payload`, build the full key,
  look up or load residues for that key, use those. If the payload lacks the
  fields to resolve the window and no residue reference is provided, fail the
  sub-stripe with a clear `stripe_error` (retryable=False) — never silently use
  stale `self.draws`.
- Remove the "load once in main(), pass to executor" coupling. The executor (or
  a residue-cache it holds) resolves per assignment.

**Blocking test:** two assignments in one daemon session with DIFFERENT window
params (e.g. different offset/window_size) must produce residues from the correct
window each time. Assert the executor used the right residue set for each (inject
a fake loader that records the key it was asked for).

---

## Blocker 2 — real spool transport (DATA LOSS + frame overflow)

**Defect A:** for `count > INLINE_SURVIVOR_LIMIT` the code sets `msg.inline = None`
and sends neither survivors nor a spool path — the coordinator gets a count only.
Data is destroyed.
**Defect B:** the inline/spool decision is on survivor COUNT, not serialized byte
size. A hybrid survivor carries a full skip-sequence; well under 50k hybrid
survivors can exceed the 64 MiB frame cap and kill the connection.

**Fix:** implement real atomic spooling now (this can no longer defer to Phase 5).

**Exact spool byte format (Beta clarification 2 — Phase 5 must know the bytes):**
```python
payload_obj = {
    "schema_version": "s172_substripe_v1",
    "stripe_id":  assign.stripe_id,
    "sub_index":  sub.sub_index,
    "seed_start": sub.seed_start,
    "seed_count": sub.seed_count,
    "survivors":  outcome.survivors,
}
payload_bytes = json.dumps(
    payload_obj, separators=(",", ":"), sort_keys=True
).encode("utf-8")
```
Then, invariants (all mandatory):
- `size_bytes == len(payload_bytes)`.
- `sha256 == hashlib.sha256(payload_bytes).hexdigest()`.
- The spool file contains EXACTLY `payload_bytes`.
- Inline mode carries the SAME logical `payload_obj`.
- Hashing occurs AFTER serialization, over `payload_bytes` — never over
  reconstructed Python objects.

**Inline vs spool selection:**
- Choose by **encoded byte size with headroom below `MAX_FRAME_BYTES`**, not by
  survivor count. Inline only if the size of the COMPLETE framed
  `SubStripeResultMessage` (measure it, or conservatively bound it — not just the
  survivor list) is ≤ **48 MiB** (leaves substantial room under the 64 MiB hard
  cap; justify the constant in a comment).

**Atomic write + file ownership (Beta clarification 2):**
- Write to a temp path in the SAME dir, fsync, `os.replace` to final (atomic).
- Populate `spool_path` / `size_bytes` / `sha256`, set `inline=None`.
- Use miner output dir resolution (`--miner-output-dir` / auto-detect
  `/dev/shm/prng/miner` → `~/miner_output`, per S172_INFRASTRUCTURE_INTERFACE §5).
- Ownership rules: coordinator verifies hash BEFORE consuming; coordinator removes
  the spool only AFTER verified collection; the WORKER removes abandoned temp
  files; final spool files remain until acknowledged or coordinator cleanup.
- NPZ contract-wall formatting is still Phase 5 — but the spool FILE with correct
  bytes/path/size/sha256 must exist and be readable now. Phase 5 consumes it.

**Blocking tests:** (a) a result forced over the inline byte threshold writes a
spool file, and `spool_path`/`size_bytes`/`sha256` are set with `inline=None`;
re-reading the file and re-hashing matches `sha256`. (b) a hybrid result whose
encoded size approaches the frame cap goes to spool, not inline (proves size-based,
not count-based).

---

## Blocker 3 — exception-safe full GPU cleanup (STABILITY)

**Defect:** rev-1 does only `del` + `gc.collect()`, and only on the success path.
If allocation/launch/extraction throws, cleanup is skipped — yet the daemon stays
alive for the next assignment. That is the accumulated-VRAM/VM-pressure failure
mode S172 exists to prevent (see S154 OOM).

**Fix:** match the proven worker's cleanup (`sieve_gpu_worker.py:328-348`):
- Wrap ALL GPU allocation/launch/extraction in `try/finally`.
- In `finally`, after per-array `del` (guarded with try/except NameError as the
  live worker does), call a shared `_best_effort_gpu_cleanup()` that does what the
  proven path does: `gc.collect()`, torch sync + cache clear if torch present,
  CuPy default memory-pool `free_all_blocks()`, CuPy pinned-memory-pool
  `free_all_blocks()`. Read the live `_best_effort_gpu_cleanup` implementation and
  replicate it; guard each step so a missing torch/cupy doesn't crash cleanup.
- Cleanup runs after EVERY sub-stripe, success or exception.

**Blocking test:** force a GPU-path exception (e.g. monkeypatch the launch to raise
after allocation) and assert the cleanup hook ran (spy on
`_best_effort_gpu_cleanup`) and the daemon accepted a subsequent assignment.

---

## Blocker 4 — Route B: implement non-Java hybrid builders (SPEC-REQUIRED)

**Ruling (binding):** Route B. Frozen spec v1.4.4 §11.I requires ≥3 base families
(java_lcg, lcg32, minstd) to pass with `test_both_modes=True`, and
`resolve_kernel_families()` auto-adds `_hybrid`/`_hybrid_reverse`. So
`_reject_hybrid()` for non-Java families contradicts acceptance. Beta owns the
§5.3 defect (it gave only the Java hybrid ABI). **Do not** narrow to Route A —
that needs a formal erratum, which is Beta's to issue, not an implementation choice.

**AUDITED ABIs (from LIVE `prng_registry.py` — implement these, do not extrapolate):**

Common forward-hybrid prefix (13 elements) for ALL families:
```
seeds, residues, survivors, match_rates, skip_sequences, strategy_ids,
survivor_count, int32(n_seeds), int32(k), strategy_max_misses,
strategy_tolerances, int32(n_strategies), float32(threshold)
```
Then the family-specific FORWARD tail:
- `java_lcg_hybrid`   (:1007): `uint64 a, uint64 c`                  → 15 args, NO offset
- `lcg32_hybrid`      (:2191): `uint32 a, uint32 c, uint32 m, int32 offset` → 17 args
- `minstd_hybrid`     (:1138): `uint32 a, uint32 m_val`              → 15 args, NO offset
- `pcg32_hybrid`      (:2095): `uint64 increment, int32 offset`      → 15 args
- `xorshift32_hybrid` (:864):  `int32 shift_a, shift_b, shift_c`     → 16 args, NO offset
- `xorshift128_hybrid`(:1276): `int32 dummy1, dummy2, dummy3`        → 16 args, NO offset

REVERSE hybrids — ALL identical shape (constants hardcoded in-kernel):
```
<13-element prefix>, int32(offset)   → 14 args
```
Verified for `java_lcg_hybrid_reverse` (:3172), `lcg32_hybrid_reverse` (:2447),
`minstd_hybrid_reverse` (:3305); the other three follow the same reverse pattern —
CONFIRM each by reading its signature before writing the builder.

**Note the per-family differences** (this is why extrapolation fails): java forward
has uint64 a,c and no offset; lcg32 forward has uint32 a,c,m AND a trailing offset;
minstd forward has uint32 a,m_val and no offset. Replicate each verbatim.

**Implementation:** replace `_reject_hybrid` with real forward-hybrid branches for
all 6 covered families. `seed_dtype` stays uint64 only for java_lcg (registry
seed_type); the other five are uint32 seeds — read each family's `seed_type` from
its config, do not assume. Since all six forward-hybrid kernels exist, KEEP
`COVERED_FAMILIES` as all six (do not narrow the advertised set).

**Handshake capability — variant-aware derivation (Beta clarification 3).**
Advertise EXACT variants, not a base-family list — BUT do NOT derive them as
`every builder key × all four suffixes` (that recreates the overclaiming problem).
Instead use an explicit variant-support table AND validate each concrete variant
against BOTH `KERNEL_REGISTRY` and the builder's supported branch:
```python
SUPPORTED_VARIANTS = {
    "java_lcg": {"java_lcg", "java_lcg_reverse",
                 "java_lcg_hybrid", "java_lcg_hybrid_reverse"},
    "lcg32":    {"lcg32", "lcg32_reverse",
                 "lcg32_hybrid", "lcg32_hybrid_reverse"},
    # ... minstd, pcg32, xorshift32, xorshift128 (their four variants each)
}
```
- The registration list is the **sorted union of successfully VALIDATED concrete
  variants** — each must exist in `KERNEL_REGISTRY` AND have a working builder branch.
- A missing or malformed registry variant MUST trigger the STOP condition below
  (propose a Route-A erratum for that variant) — it must NOT simply disappear from
  the handshake. Silent disappearance is itself an overclaim/underclaim defect.

**STOP condition (Beta-mandated):** if reading any kernel shows a variant is
unusable or incomplete (missing kernel_name, malformed source, arity that can't be
satisfied), STOP implementation for that family and return a proposed Beta erratum
for Route A on that family — do NOT silently narrow capability or ship a guessed ABI.

---

## Blocker 5 — tests for the dangerous paths

Add blocking gates (keep the existing 8; these extend them):
- **Two draw windows:** (blocker 1) two assignments, different window params,
  correct residues each.
- **Spooled result:** (blocker 2a) over-threshold result writes a spool file with
  correct path/size/sha256, inline=None, re-hash matches.
- **Size-based selection:** (blocker 2b) a hybrid result near the frame cap spools
  rather than inlines.
- **Cleanup after exception:** (blocker 3) forced GPU exception still runs full
  cleanup; daemon serves the next assignment.
- **Exact capability advertisement:** (blocker 4) `register` advertises concrete
  variants including the hybrid variants now built; assert the hybrid variants are
  present and no unsupported variant is claimed.
- **Non-Java full-mode:** (blocker 4/§11.I) a `test_both_modes=True` workflow for
  lcg32 and minstd runs all four phases through the correct builders (CPU arg-shape
  assertions for each of the 4 variants; GPU smoke may stay skippable but the
  builder/dispatch path must be exercised on CPU).
- Keep the GPU gate skippable BUT make clear in the harness output that a CPU-only
  green is contract-validation only, not ROCm deploy-readiness (that's Phase 6).

Add per-family forward-hybrid arg-shape assertions matching the audited ABIs above
(lengths: java 15, lcg32 17, minstd 15, pcg32 15, xorshift32 16, xorshift128 16;
all reverse = 14; verify trailing dtypes per family).

---

## Workflow

Iterate the edit→harness loop on 101 until all gates green (including the new ones).
GPU smoke runs for real on the 3080 Ti. **Do NOT commit or push** — Michael commits
after Team Beta re-approves. Update the SESSION_CHANGELOG with the five fixes and
the audited-ABI note. When green, report and STOP for review.
