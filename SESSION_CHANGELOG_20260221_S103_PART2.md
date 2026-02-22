# SESSION CHANGELOG S103 PART 2 — Reverse Sieve Bug Fix
Date: 2026-02-21
Commit: 213fe51

## Bug Found
After Step 1 re-run verification, all three survivor files had identical counts:
- forward: 49,305 / reverse: 49,305 / bidirectional: 49,305
- All 49,305 seeds had forward_match_rate == reverse_match_rate
- Root cause: reverse sieve was passing same prng_type as forward

## Root Cause
window_optimizer_integration_final.py:
- Line 161: reverse_args.prng_type = prng_base         # 'java_lcg' (WRONG)
- Line 287: reverse_args_hybrid.prng_type = prng_hybrid # 'java_lcg_hybrid' (WRONG)

The reverse sieve was running the same forward kernel against the same data,
intersecting with itself, always producing identical counts.

sieve_filter.py auto-flips residue array when '_reverse' is in prng_family name.
Without the _reverse suffix, the flip never happened.

## Fix Applied
- Line 161: reverse_args.prng_type = prng_base + '_reverse'
- Line 287: reverse_args_hybrid.prng_type = prng_hybrid + '_reverse'
- Lines 159, 283: print statements updated to show correct prng name

## Verification
Step 1 re-run with 100 trials, 50K seeds:
- forward:       10,912
- reverse:        7,080
- bidirectional:  4,245  (bid/fwd ratio: 0.389)
- PASS — genuine bidirectional filtering confirmed

## S103 Complete Status
Part 1 (per-seed match rates): FIXED + VERIFIED
Part 2 (reverse sieve prng_type): FIXED + VERIFIED
Step 1 fully operational for the first time.
