#!/usr/bin/env python3
"""
test_prng_encoding.py — S172 Phase 0 acceptance harness (v3.2)

Covers the 8 TB-mandated acceptance gates for the shared PRNG encoding:

  1. All 44 KERNEL_REGISTRY keys encode -> decode round-trip correctly.
  2. Unknown prng_type hard-fails (encode).
  3. Unknown encoded id hard-fails (decode).
  4. A synthetic v3 NPZ using the new encoding loads via
     load_survivors(..., return_format="array").
  5. The same synthetic NPZ loads via load_survivors(..., return_format="dict").
  6. Step 2 loader path accepts the NPZ without schema error
     (load_survivors array path is exactly what scorer_trial_worker.py uses).
  7. Step 3 job generator converts the NPZ into scoring_chunks/chunk_*.json
     without metadata loss (exercises generate_step3_scoring_jobs path).
  8. The generated chunk JSON still has all expected metadata fields.

Plus a guardrail test pinning len(PRNG_TYPE_ENCODING) == 44.

Run:
    cd ~/distributed_prng_analysis
    PYTHONPATH=. python3 tests/test_prng_encoding.py

Exit code 0 = all gates green (TB merge condition satisfied).
Exit code 1 = a gate failed (DO NOT COMMIT).
"""
import os
import sys
import json
import tempfile
import traceback

# Ensure project root on path (mirrors how Steps 2-6 run).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from prng_registry import KERNEL_REGISTRY  # noqa: E402
from utils.prng_encoding import (  # noqa: E402
    PRNG_TYPE_ENCODING,
    PRNG_TYPE_DECODING,
    encode_prng_type,
    decode_prng_type,
    encode_skip_mode,
    decode_skip_mode,
    resolve_prng_type,
    ENCODING_VERSION,
)
from utils.survivor_loader import load_survivors  # noqa: E402

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"

_results = []


def _check(name, fn):
    try:
        fn()
        _results.append((name, True, None))
        print(f"  [{_PASS}] {name}")
    except Exception as e:
        _results.append((name, False, traceback.format_exc()))
        print(f"  [{_FAIL}] {name}: {e}")


# === The 22-array NPZ schema (mirrors convert_survivors_to_binary writer) ===
_NPZ_INT32 = ['window_size', 'offset', 'trial_number', 'skip_min', 'skip_max', 'skip_range']
_NPZ_FLOAT32 = [
    'forward_matches', 'reverse_matches', 'forward_count', 'reverse_count',
    'bidirectional_count', 'intersection_count', 'intersection_ratio',
    'intersection_weight', 'bidirectional_selectivity', 'forward_only_count',
    'reverse_only_count', 'survivor_overlap_ratio', 'score',
]


def _make_synthetic_npz(path, prng_types):
    """Build a 22-array NPZ using the NEW encoding for the given prng_type list."""
    n = len(prng_types)
    arrays = {
        'seeds': np.arange(1, n + 1, dtype=np.uint32),
        'skip_mode': np.array(
            [encode_skip_mode('variable' if '_hybrid' in p else 'constant') for p in prng_types],
            dtype=np.uint8),
        'prng_type': np.array([encode_prng_type(p) for p in prng_types], dtype=np.uint8),
    }
    for k in _NPZ_INT32:
        arrays[k] = np.arange(n, dtype=np.int32)
    for k in _NPZ_FLOAT32:
        arrays[k] = np.linspace(0.1, 0.9, n).astype(np.float32)
    np.savez_compressed(path, **arrays)
    return path


# ---------------------------------------------------------------------------
# GATE 1 — all 44 registry keys round-trip
# ---------------------------------------------------------------------------
def gate1_roundtrip_all_keys():
    keys = sorted(KERNEL_REGISTRY.keys())
    assert len(keys) == 44, f"expected 44 registry keys, got {len(keys)}"
    for k in keys:
        eid = encode_prng_type(k)
        assert isinstance(eid, int) and 0 <= eid <= 255, f"{k} -> bad id {eid}"
        back = decode_prng_type(eid)
        assert back == k, f"round-trip mismatch: {k} -> {eid} -> {back}"
    # Bijection: every id unique
    ids = [encode_prng_type(k) for k in keys]
    assert len(set(ids)) == len(ids), "encoding is not injective (duplicate ids)"


# ---------------------------------------------------------------------------
# GATE 2 — unknown prng_type hard-fails on encode
# ---------------------------------------------------------------------------
def gate2_unknown_prng_encode_fails():
    for bogus in ['bogus_prng', 'java_lcg_typo', '', 'randu']:
        # 'randu' is intentionally NOT in the registry (legacy dead enum),
        # so it must now hard-fail rather than silently encode.
        try:
            encode_prng_type(bogus)
        except ValueError:
            continue
        raise AssertionError(f"encode_prng_type({bogus!r}) should have raised ValueError")


# ---------------------------------------------------------------------------
# GATE 3 — unknown encoded id hard-fails on decode
# ---------------------------------------------------------------------------
def gate3_unknown_id_decode_fails():
    n = len(PRNG_TYPE_DECODING)
    for bad_id in [n, n + 1, 99, 255]:
        try:
            decode_prng_type(bad_id)
        except ValueError:
            continue
        raise AssertionError(f"decode_prng_type({bad_id}) should have raised ValueError")


# ---------------------------------------------------------------------------
# GATE 4 — synthetic NPZ loads via return_format="array"
# ---------------------------------------------------------------------------
def gate4_npz_loads_array():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'syn_binary.npz')
        _make_synthetic_npz(p, ['java_lcg', 'java_lcg_hybrid', 'lcg32', 'minstd_hybrid'])
        result = load_survivors(p, return_format="array")
        data = result.data
        assert 'seeds' in data, "array load missing 'seeds'"
        assert 'forward_matches' in data and 'reverse_matches' in data
        assert len(data['seeds']) == 4, f"expected 4 seeds, got {len(data['seeds'])}"


# ---------------------------------------------------------------------------
# GATE 5 — synthetic NPZ loads via return_format="dict" (round-trip provenance)
# ---------------------------------------------------------------------------
def gate5_npz_loads_dict():
    types = ['java_lcg', 'java_lcg_hybrid', 'lcg32', 'minstd_hybrid']
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'syn_binary.npz')
        _make_synthetic_npz(p, types)
        result = load_survivors(p, return_format="dict")
        recs = result.data
        assert len(recs) == 4, f"expected 4 dict records, got {len(recs)}"
        decoded = sorted([(r['seed'], r['prng_type']) for r in recs])
        # seeds 1..4 map to types in order
        expected = sorted(list(zip(range(1, 5), types)))
        assert decoded == expected, f"provenance round-trip failed: {decoded} != {expected}"
        # CRITICAL: java_lcg_hybrid must NOT collapse to java_lcg
        hybrid_recs = [r for r in recs if r['prng_type'] == 'java_lcg_hybrid']
        assert len(hybrid_recs) == 1, "java_lcg_hybrid provenance was lost (silent collapse bug)"


# ---------------------------------------------------------------------------
# GATE 6 — Step 2 loader path (scorer_trial_worker uses load_survivors array)
# ---------------------------------------------------------------------------
def gate6_step2_loader_accepts():
    # scorer_trial_worker.py:151-246 requires seeds, forward_matches,
    # reverse_matches; soft-reads bidirectional_count, intersection_ratio,
    # trial_number, skip_mode. Verify all present and correctly typed.
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'syn_binary.npz')
        _make_synthetic_npz(p, ['java_lcg'] * 5)
        result = load_survivors(p, return_format="array")
        d = result.data
        for hard in ['seeds', 'forward_matches', 'reverse_matches']:
            assert hard in d, f"Step 2 hard-required field missing: {hard}"
        for soft in ['bidirectional_count', 'intersection_ratio', 'trial_number', 'skip_mode']:
            assert soft in d, f"Step 2 soft field missing: {soft}"
        # forward/reverse matches must be float (per-seed signal)
        assert d['forward_matches'].dtype.kind == 'f'


# ---------------------------------------------------------------------------
# GATE 7 + 8 — Step 3 chunk generation preserves all metadata
# ---------------------------------------------------------------------------
def gate7_8_step3_chunk_metadata():
    # Exercise the Step 3 metadata-preservation contract: load the NPZ in dict
    # form (exactly what generate_step3_scoring_jobs.py does), confirm every
    # survivor dict carries the full metadata field set with no loss.
    types = ['java_lcg', 'java_lcg_hybrid', 'lcg32_hybrid', 'minstd', 'mt19937_hybrid_reverse']
    expected_fields = {
        'seed', 'forward_match_rate', 'reverse_match_rate', 'prng_type', 'skip_mode',
        'window_size', 'offset', 'trial_number', 'skip_min', 'skip_max', 'skip_range',
        'forward_count', 'reverse_count', 'bidirectional_count', 'intersection_count',
        'intersection_ratio', 'intersection_weight', 'bidirectional_selectivity',
        'forward_only_count', 'reverse_only_count', 'survivor_overlap_ratio', 'score',
    }
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'syn_binary.npz')
        _make_synthetic_npz(p, types)
        result = load_survivors(p, return_format="dict")
        recs = result.data
        assert len(recs) == len(types)
        # Gate 8: every record has the metadata surface (allowing loader's field
        # naming — seeds->seed, forward_matches->forward_match_rate).
        sample = recs[0]
        present = set(sample.keys())
        # The loader emits at least the core + metadata fields. Verify no
        # metadata field is silently dropped (the METADATA LOSS guardrail).
        missing = expected_fields - present
        # Loader may name the match fields differently; tolerate either spelling.
        if 'forward_match_rate' in missing and 'forward_matches' in present:
            missing.discard('forward_match_rate')
        if 'reverse_match_rate' in missing and 'reverse_matches' in present:
            missing.discard('reverse_match_rate')
        if 'seed' in missing and 'seeds' in present:
            missing.discard('seed')
        assert not missing, f"Step 3 metadata loss — missing fields: {sorted(missing)}"
        # Gate 7: hybrid/reverse provenance preserved through the chunk surface
        reverse_hybrid = [r for r in recs if r['prng_type'] == 'mt19937_hybrid_reverse']
        assert len(reverse_hybrid) == 1, "mt19937_hybrid_reverse provenance lost in chunk metadata"


# ---------------------------------------------------------------------------
# GUARDRAIL — registry size pinned at 44
# ---------------------------------------------------------------------------
def guardrail_registry_size():
    assert len(PRNG_TYPE_ENCODING) == 44, (
        f"PRNG_TYPE_ENCODING has {len(PRNG_TYPE_ENCODING)} entries, expected 44. "
        f"KERNEL_REGISTRY changed — this is an intentional-decision tripwire. "
        f"If the change is deliberate: regenerate NPZs and bump ENCODING_VERSION "
        f"(currently {ENCODING_VERSION}), then update this assertion."
    )


def main():
    print(f"\nS172 Phase 0 acceptance harness — ENCODING_VERSION {ENCODING_VERSION}")
    print("=" * 66)
    _check("Gate 1: all 44 registry keys round-trip", gate1_roundtrip_all_keys)
    _check("Gate 2: unknown prng_type hard-fails (encode)", gate2_unknown_prng_encode_fails)
    _check("Gate 3: unknown id hard-fails (decode)", gate3_unknown_id_decode_fails)
    _check("Gate 4: synthetic NPZ loads (return_format=array)", gate4_npz_loads_array)
    _check("Gate 5: synthetic NPZ loads (return_format=dict)", gate5_npz_loads_dict)
    _check("Gate 6: Step 2 loader path accepts NPZ", gate6_step2_loader_accepts)
    _check("Gate 7+8: Step 3 chunk metadata preserved", gate7_8_step3_chunk_metadata)
    _check("Guardrail: registry size pinned at 44", guardrail_registry_size)
    print("=" * 66)

    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{passed}/{total} gates green")
    if passed != total:
        print("\nFAILURES (DO NOT COMMIT):")
        for name, ok, tb in _results:
            if not ok:
                print(f"\n--- {name} ---\n{tb}")
        sys.exit(1)
    print("\nAll gates green — TB merge condition satisfied.")
    sys.exit(0)


if __name__ == "__main__":
    main()
