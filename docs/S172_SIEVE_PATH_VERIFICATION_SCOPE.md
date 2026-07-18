# S172 — Sieve Path Verification Scope (what is / isn't proven, and where)

**Purpose:** pin down exactly what the RANGE-MINER work has verified about the
four sieve paths, so nobody later mistakes "Phase 3 green" for "the sieve
computes correct survivors through the miner." Those are different claims. This
note is the standing reference; update the "status" column as phases land.

Author: Team Alpha. Ruling authority for scope changes: Team Beta.

---

## The four sieve paths (× 6 covered families = 24 concrete variants)

RANGE-MINER replaces the GPU sieve **inside Step 1 (Window Optimizer)** — the
"candidate seeds × trials" full-seed-space sweep that produced the ~17K-launch
GCVM_L2 crash under PWC. It does NOT reimplement the sieve math; the compiled
CUDA kernels in `prng_registry.py` are the frozen, proven artifact. The miner
feeds those kernels arguments and marshals their output.

| # | Path | Variant suffix | Kernel arg count (per family) |
|---|------|----------------|-------------------------------|
| 1 | forward sieve, constant skip-gap | `<base>`            | java14, lcg32 15, minstd14, pcg32 13, xs32 15, xs128 15 |
| 2 | reverse sieve, constant skip-gap | `<base>_reverse`   | **12 (all families)** — params hardcoded in-kernel |
| 3 | forward sieve, variable skip-gap | `<base>_hybrid`    | java15, lcg32 17, minstd15, pcg32 15, xs32 16, xs128 16 |
| 4 | reverse sieve, variable skip-gap | `<base>_hybrid_reverse` | **14 (all families)** — params hardcoded in-kernel |

("variable skip-gap" == "hybrid" == multi-strategy skip. "constant skip-gap" ==
fixed skip_min/skip_max sweep.)

---

## Two DIFFERENT claims — do not conflate

**Claim A — CONTRACT: "correct data & parameters are created / marshaled."**
The miner builds the exact argument list (count, dtype, which params are passed
vs. hardcoded) each kernel expects, and returns survivors in the right tuple
shape. This is what Phase 3 verified.

**Claim B — COMPUTATION: "the sieve produces the CORRECT survivors."**
Given a real window, the seeds that *should* survive actually do. This is the
sieve math. It lives in the frozen kernels and was established by the OLD PWC
path. Phase 3 does NOT re-verify it. It is proven for the miner by **Phase 6
byte-identity** (miner output == PWC output on identical input).

A GPU launch that "runs without an arg-count/type error and returns a
well-formed result" proves Claim A on hardware. It does NOT prove Claim B —
the harness feeds SYNTHETIC windows and asserts STRUCTURE (`count>=0`, tuple
shape), never "seed X should survive and did." (This is why the smoke tests
correctly show 0 survivors: made-up 10-draw window, 256 seeds, no true
generating seed exists to find.)

---

## Verification status per path (as of Phase 3 rev-3, 2026-07-18)

| Path | Claim A: ABI/params asserted (CPU) | Claim A: real GPU launch | Claim B: correct survivors |
|------|-----------------------------------|--------------------------|----------------------------|
| 1 fwd constant       | ✅ gate 2                | ✅ gate 7 (`java_lcg`) — standing regression | ❌ deferred → Phase 6 |
| 2 rev constant       | ✅ gate 2 (12-arg)       | ✅ gate 7 (`java_lcg_reverse`) standing; all 6 launched in rev-3 review (not all in standing suite) | ❌ deferred → Phase 6 |
| 3 fwd variable/hybrid| ✅ gate 2 (per-family)   | ⚠️ all 12 hybrids launched per Alpha rev-2 report — NOT pinned in standing suite | ❌ deferred → Phase 6 |
| 4 rev variable/hybrid| ✅ gate 2 (14-arg)       | ⚠️ launched in review — NOT pinned in standing suite | ❌ deferred → Phase 6 |

Legend: ✅ locked in the standing harness · ⚠️ verified once during review but not
a permanent regression gate · ❌ not this phase's job.

---

## Open items this surfaces (decide when scoping Phase 6/7)

1. **GPU-launch regression coverage is partial.** The standing harness (gate 7)
   only pins forward-constant + reverse-constant `java_lcg` as repeatable GPU
   launches. The other 22 variants were launched during review but aren't
   permanent regression gates. DECISION NEEDED: add a full 24-variant GPU
   launch-success gate (Phase 6/7), or accept review-time verification. Note:
   even a full launch gate proves only Claim A, never Claim B.

2. **Claim B (correctness) has exactly one planned proof: Phase 6 byte-identity.**
   That test only exercises the variants a given run actually uses. If a variant
   is never hit by the Phase 6 run set, its computational correctness through the
   miner is unproven even after Phase 6. DECISION NEEDED: ensure the Phase 6
   acceptance run set exercises all four paths (ideally all 24 variants), or
   explicitly scope which variants are in production use (TFM sieve targets
   java_lcg only — so java_lcg's 4 variants are the must-prove set; the other 5
   families' hybrids exist and pass Claim A but may never be production-exercised).

3. **A stronger optional correctness check (not required by spec):** seed a known
   java_lcg value, generate a real window from it, confirm the sieve recovers
   that seed through the miner. This would prove Claim B directly for the forward
   path without waiting on full byte-identity. Out of Phase 3 scope; nice-to-have
   for Phase 6 confidence.

---

## One-line summary

Phase 3 proved the miner hands the frozen sieve kernels the **right arguments**
for all four paths and returns output in the **right shape** (Claim A). It did
**not** prove the four paths compute the **right survivors** through the miner
(Claim B) — that is Phase 6's byte-identity job, and only for the variants that
run actually exercises.
