# SESSION_CHANGELOG — 2026-07-18 — S172 Phase 3 rev-3 (Team Beta rev-2 rejection)

**Team Alpha implementation. NOT committed/pushed — Michael commits after Team
Beta re-approves.** Ref: `docs/S172_PHASE3_FIX_BRIEF_v3.md`. rev-2 was REJECTED;
this session fixes the two remaining blockers + four test corrections in the
TIGHT scope Beta specified. Nothing Beta approved was touched.

## Files changed
- `miner/range_miner_worker.py` — reverse-constant builders + spool guard + one
  docstring (residue-reference clarification).
- `tests/test_s172_phase3_worker.py` — gates 2, 7, 9, 11, 12, 14 corrected.

## Reverse-constant ABI audit (self-verified against LIVE prng_registry.py)
Extracted all six fixed-skip REVERSE kernel signatures directly from their
`kernel_source` strings:
`java_lcg_reverse_sieve`, `lcg32_reverse_sieve`, `minstd_reverse_sieve`,
`pcg32_reverse_sieve`, `xorshift32_reverse_sieve`, `xorshift128_reverse_sieve`.
**All six are identical:**
`candidate_seeds, residues, survivors, match_rates, best_skips, survivor_count,
int n_candidates, int k, int skip_min, int skip_max, float threshold, int offset`
= `_constant_prefix` + `int32(offset)` = **12 args, NO family tail** (generator
params a/c/m/shifts/increment are hardcoded inside the reverse kernel body).
Seed pointer: `unsigned long long` for java_lcg, `unsigned int` for the other
five. Matches Beta's ruling exactly.

## Blocker fixes
- **B1 (v3) — reverse-CONSTANT builders had the wrong ABI.** rev-2 appended the
  forward family tail on the reverse path (13–15 args). Fixed: every constant
  builder now splits the reverse branch → `_constant_prefix + _offset_tail` = 12
  args, with NO family tail. Forward-constant and both hybrid branches are
  UNCHANGED. Final per-family arg counts:
  | base | fwd-const | rev-const | fwd-hybrid | rev-hybrid |
  |---|---|---|---|---|
  | java_lcg | 14 | **12** | 15 | 14 |
  | lcg32 | 15 | **12** | 17 | 14 |
  | minstd | 14 | **12** | 15 | 14 |
  | pcg32 | 13 | **12** | 15 | 14 |
  | xorshift32 | 15 | **12** | 16 | 14 |
  | xorshift128 | 15 | **12** | 16 | 14 |
- **B2 (v3) — oversized inline candidate threw before the spool branch.** rev-2's
  `if len(message_to_bytes(candidate)) <= INLINE_BYTE_LIMIT` framed a known-large
  candidate, so a >64 MiB body raised `ValueError` inside the guard and aborted
  stripe handling. Fixed to Beta's guard: spool immediately when
  `len(payload_bytes) >= INLINE_BYTE_LIMIT` (never frame it); otherwise measure,
  and treat a framing `ValueError` as "must spool". `INLINE_BYTE_LIMIT` stays
  48 MiB.

## Test corrections
- **Gate 2:** added REVERSE-constant assertions — every fixed-skip reverse variant
  is exactly 12 args = `_constant_prefix + int32(offset)`, no family tail,
  `best_skips` present, `skip_sequences` absent. Forward/reverse are no longer
  assumed identical.
- **Gate 7 (GPU smoke):** extended to also launch a REVERSE-CONSTANT variant
  (`java_lcg_reverse`) on hardware, so a 12-vs-14 arg error surfaces on the GPU,
  not only in a CPU shape assertion.
- **Gate 9 (B1):** now drives TWO `stripe_assign`s with different window params
  THROUGH `SieveExecutor.execute()` (not just `resolve()` directly); a recording
  loader proves execute() requests a fresh residue identity per assignment (two
  distinct keys, correct residues each). Works on GPU (full run) and CPU-only
  (resolve records, then the cupy import raises after resolution). Direct-unit
  checks retained.
- **Gate 11 (B2b):** keeps the size-vs-count check and ADDS a REAL 64 MiB cap
  crossing WITHOUT shrinking the threshold: a genuine >64 MiB inline payload
  (a) spools under production config via the payload-size guard and does NOT
  raise (the actual regression proof vs the rev-2 bug), and (b) exercises the
  `ValueError`-catch net by RAISING the limit above the frame cap so framing hits
  `encode_frame`'s 64 MiB `ValueError`, which must be caught and spooled.
- **Gate 12 (B3):** subsequent assignment now runs through the SAME executor
  (restore the real launch hook via `del ex._gpu_launch`), not a fresh instance —
  proving the daemon that survived the exception serves the next assignment.
- **Gate 14 (B4/§11.I):** `expected_len` map fixed — reverse-constant phase is 12
  (was `11+tail+1`); forward-constant `11+tail+1`, forward-hybrid `HYBRID_FWD_LEN`,
  reverse-hybrid 14.

## Non-blocking residue-reference clarification — chose (a)
Documented on `ResidueResolver`: Phase 4's `stripe_assign.payload` ALWAYS supplies
`dataset*` + `window_size` (+ optional `sessions`/`offset`/`residue_sha256`). No
bare residue-reference path is implemented (it would be dead code given the current
coordinator direction); `resolve()` fails CLEARLY with `ResidueResolutionError`
if those fields are absent (existing guard), never falling back to a stale window.

## Harness — 14/14 gates green
Under `~/venvs/torch` on VM 101 (cupy 13.5.1, 3080 Ti), exit 0. GPU gates 7 & 12
ran for real; gate 7 launched `java_lcg` AND `java_lcg_reverse`. Also 14/14 under
system `python3` with cupy absent (gates 7 & 12 skip cleanly) — CPU contract-parity.

**Empirical reverse-ABI validation:** all 6 reverse-CONSTANT variants compiled and
launched on the 3080 Ti with the 12-arg ABI, zero arg-count/type errors (a wrong
count would make CuPy raise at launch).

## Unchanged (Beta-approved, per v3 scope)
Six forward-hybrid ABIs; reverse-HYBRID builders (14 args); content-keyed
`ResidueResolver`; canonical `s172_substripe_v1` spool format; `supported_variants()`
validation + `VariantStopCondition`; the `try/finally` `_best_effort_gpu_cleanup`
structure.

## Fallback parity
code=current, env=ok (no new deps).
