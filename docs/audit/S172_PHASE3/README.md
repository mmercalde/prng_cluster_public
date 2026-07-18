# S172 Phase 3 — Audit Trail (Worker Daemon)

**Status:** Approved (rev-3)
**Phase:** S172 Phase 3 — per-GPU RANGE-MINER worker daemon
**Applies to:** `miner/range_miner_worker.py`, `tests/test_s172_phase3_worker.py`
**Authoritative spec:** `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md` (frozen `1f6c0c5`)

This directory preserves the governance trail for S172 Phase 3, which required
**three review rounds** and uncovered **two real production defects**. Per Team
Beta policy, a phase earns a permanent audit trail when its review cycle
discovers a material correctness/safety/data-loss defect, changes a frozen
requirement, requires multiple binding rejection rounds, or sets a precedent.
Phase 3 met the first and third criteria.

## Why this phase was preserved

Two defects were caught in review that a changelog outcome-summary alone would
not explain:

1. **Incorrect fixed-skip *reverse* kernel ABIs.** rev-1/rev-2 appended
   forward-only generator params to the reverse constant builders. The live
   reverse kernels take `_constant_prefix + int32(offset) = 12 args`, with
   generator params hardcoded in-kernel. Wrong arg counts would mis-launch every
   reverse-constant sieve. Caught because Beta read the kernel source rather than
   extrapolating from the forward layout.

2. **Spool path failed exactly at the frame cap.** The inline/spool decision
   called `message_to_bytes(candidate)`, whose framing encoder raises `ValueError`
   past `MAX_FRAME_BYTES` (64 MiB) — so an oversized result raised *before*
   reaching the spool branch, aborting stripe handling. The exact frame-overflow
   failure the spool design was meant to remove.

Both are recorded here with their reasoning, not just their fix.

## Timeline

| Round | Outcome | Artifact | Blockers |
|-------|---------|----------|----------|
| rev-1 | REJECTED | `PHASE3_INITIAL_REVIEW.md` | 5 (residue window, spool data-loss, cleanup, hybrid-family/spec contradiction, missing tests) |
| rev-2 | REJECTED | `PHASE3_FIX_BRIEF_REV2.md` | 2 (reverse-constant ABI, spool frame-cap overflow) + 4 test corrections |
| rev-3 | APPROVED | `PHASE3_FINAL_APPROVAL_REV3.md` | none — 14/14 gates green on RTX 3080 Ti |

## Artifact index

- **`PHASE3_INITIAL_REVIEW.md`** — Team Beta's rev-1 rejection: the five original
  blockers and the Route B ruling (implement non-Java hybrid builders).
- **`PHASE3_FIX_BRIEF_REV2.md`** — the binding five-blocker + three-clarification
  implementation brief (audited kernel ABIs from `prng_registry.py`).
- **`PHASE3_FINAL_APPROVAL_REV3.md`** — the two remaining blockers, their fixes,
  and Beta's final approval with validation evidence.

## Related (outside this audit dir)

- Operational summary + final test result: session changelog
  (`docs/SESSION_CHANGELOG_20260718_S172_PHASE3_REV3.md`).
- What is / isn't computationally proven for the four sieve paths:
  `docs/S172_SIEVE_PATH_VERIFICATION_SCOPE.md`.

## Standing note carried out of this phase

Phase 3 verified the **contract** (correct args marshaled to the frozen kernels,
correct output shape) for all four sieve paths. It did **not** verify the sieve
**computation** through the miner — that is Phase 6 byte-identity acceptance
against the proven PWC path. See the verification-scope note above.
