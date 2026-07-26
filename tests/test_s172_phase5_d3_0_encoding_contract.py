#!/usr/bin/env python3
"""
test_s172_phase5_d3_0_encoding_contract.py — S172 Phase-5 Deliverable D3.0
acceptance harness (legacy seam correction: canonical PRNG/skip encoding +
rectangular 22-array empty output).

D3.0 is the NARROW seam correction described in
docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3_0.md. Phase 0 replaced the
divergent inline PRNG encoding tables with a registry-derived canonical module
(`utils/prng_encoding.py`) that hard-fails on unknown identities — but that fix
never reached either NPZ writer:

  - convert_survivors_to_binary.py        PRNG_TYPE_ENCODING / SKIP_MODE_ENCODING
  - window_optimizer_integration_final.py _PRNG_ENC / _SKIP_ENC (live Step-1
                                          NPZ producer, inline in the accumulator)

Both carried a local 12-entry table with a silent `.get(..., 0)` fallback, so
`java_lcg_hybrid` (canonical 1) was silently written as `java_lcg` (0), and
seven shared keys disagreed with canonical outright.

INDEPENDENT ORACLE (the G9 lesson, binding). This harness does NOT import
PRNG_TYPE_ENCODING / SKIP_MODE_ENCODING and use them as its own expected values.
Every asserted numeric id is an INTEGER LITERAL hand-transcribed below, and the
22 array names / order / dtypes are hand-transcribed from the on-disk contract
(convert_survivors_to_binary.py v3.1 `np.savez_compressed` call order), not
read back out of the code under test.

THE STANDALONE WRITER IS EXERCISED LIVE. `convert_survivors_to_binary` is
imported and driven end-to-end (JSON in -> NPZ on disk -> np.load out). Nothing
is copied into this file — editing the production writer changes what this
harness runs. Where a check is about the encoding contract itself rather than
about one writer, it also asserts directly against the canonical
`utils/prng_encoding` entry points.

MIGRATION-GATE RETIREMENT (S172 Phase-5 D3.5, Team Beta ruling). D3.0 originally
exercised a SECOND live writer: the inline `_survivors_to_arrays` closure in the
S145-R1 accumulator block of `window_optimizer_integration_final.py`, extracted
from live source by AST line-range because a nested closure cannot be imported.
That seam existed only until the shared finalizer replaced the inline
run-finalization block. D3.5 removed it, so every inline-writer assertion here
has reached its planned expiry and is REMOVED — not stubbed, not snapshotted,
and not kept alive as dead production code. The standalone-writer and canonical
assertions are unchanged and unweakened; the retired seam's successor is
`utils/run_finalizer.py`, gated by `tests/test_s172_phase5_d3_5_finalizer.py`.

Ten checks (E1-E10), each constructed to FAIL on the wrong behavior. Against the
UNMODIFIED tree at 2d37b77 the expected result is:

    E1  PASS  (java_lcg is 0 in both the legacy table and canonical)
    E2  FAIL  <- java_lcg_hybrid absent from legacy table, .get(...,0) -> 0
    E3  FAIL  <- legacy java_lcg_reverse = 1, canonical = 3
    E4  FAIL  <- unknown prng_type silently becomes 0 instead of raising
    E5  FAIL  <- unknown skip_mode silently becomes 0 instead of raising
    E6  FAIL  <- randu/randu_reverse silently encode as 10/11 instead of raising
    E7  PASS  (canonical skip_mode {constant:0, variable:1} == legacy table)
    E8  FAIL  <- empty input writes ONE array (seeds=[]), not 22
    E9  PASS  (known-identity java_lcg golden; must STAY green post-fix)
    E10 FAIL  <- mixed constant+hybrid prng_type column collapses to {0}
                 (asserted against the standalone converter post-D3.5)

Run:
    cd ~/distributed_prng_analysis
    source ~/venvs/torch/bin/activate
    PYTHONPATH=. python3 tests/test_s172_phase5_d3_0_encoding_contract.py
Exit 0 = all green. Exit 1 = a check failed (DO NOT COMMIT).
"""
import json
import os
import sys
import tempfile
import traceback

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import convert_survivors_to_binary as CSB  # noqa: E402



# ─────────────────────────────────────────────────────────────────────────────
# Hand-transcribed oracle — NOT imported from the code under test.
# ─────────────────────────────────────────────────────────────────────────────

# The 22 arrays in their frozen on-disk order (convert_survivors_to_binary.py
# v3.1 savez_compressed call order), each with its frozen dtype. The six
# `*_count` arrays are float32 despite being logically integral — that is the
# existing on-disk contract and is reproduced deliberately.
NPZ_CONTRACT = [
    ("seeds",                     "uint32"),
    ("forward_matches",           "float32"),
    ("reverse_matches",           "float32"),
    ("window_size",               "int32"),
    ("offset",                    "int32"),
    ("trial_number",              "int32"),
    ("skip_min",                  "int32"),
    ("skip_max",                  "int32"),
    ("skip_range",                "int32"),
    ("forward_count",             "float32"),
    ("reverse_count",             "float32"),
    ("bidirectional_count",       "float32"),
    ("intersection_count",        "float32"),
    ("intersection_ratio",        "float32"),
    ("intersection_weight",       "float32"),
    ("bidirectional_selectivity", "float32"),
    ("forward_only_count",        "float32"),
    ("reverse_only_count",        "float32"),
    ("survivor_overlap_ratio",    "float32"),
    ("score",                     "float32"),
    ("skip_mode",                 "uint8"),
    ("prng_type",                 "uint8"),
]
NPZ_KEYS = {name for name, _ in NPZ_CONTRACT}

# Canonical ids, written as integer literals (registry-derived, alphabetical
# over the 44 KERNEL_REGISTRY keys). Deliberately NOT imported.
ID_JAVA_LCG = 0
ID_JAVA_LCG_HYBRID = 1
ID_JAVA_LCG_REVERSE = 3
SKIP_CONSTANT = 0
SKIP_VARIABLE = 1

# ─────────────────────────────────────────────────────────────────────────────
# E9 golden — the pre-fix 22-array output for a known-identity java_lcg fixture.
# Captured from the UNMODIFIED tree at 2d37b77 and transcribed as literals. All
# float values are exact binary fractions, so float32 round-trips are exact.
# ─────────────────────────────────────────────────────────────────────────────

E9_FIXTURE = [
    {
        "seed": 12345,
        "forward_match_rate": 0.25, "reverse_match_rate": 0.5,
        "window_size": 6, "offset": 54, "trial_number": 3,
        "skip_min": 1, "skip_max": 9, "skip_range": "2-7",
        "forward_count": 10.0, "reverse_count": 11.0, "bidirectional_count": 12.0,
        "intersection_count": 13.0, "intersection_ratio": 0.5,
        "intersection_weight": 1.5, "bidirectional_selectivity": 0.125,
        "forward_only_count": 4.0, "reverse_only_count": 5.0,
        "survivor_overlap_ratio": 0.75, "score": 0.375,
        "skip_mode": "constant", "prng_type": "java_lcg",
    },
    {
        # no 'prng_type' — exercises the caller-side prng_base resolution step
        "seed": 999,
        "forward_match_rate": 0.75, "reverse_match_rate": 0.125,
        "window_size": 5, "offset": 3, "trial_number": 1,
        "skip_min": 0, "skip_max": 2, "skip_range": [4, 10],
        "forward_count": 1.0, "reverse_count": 2.0, "bidirectional_count": 3.0,
        "intersection_count": 4.0, "intersection_ratio": 0.25,
        "intersection_weight": 0.5, "bidirectional_selectivity": 0.0625,
        "forward_only_count": 6.0, "reverse_only_count": 7.0,
        "survivor_overlap_ratio": 0.5, "score": 0.4375,
        "skip_mode": "variable", "prng_base": "java_lcg",
    },
    # bare record — every field defaults, including the prng_type resolution
    {"seed": 7},
]

E9_GOLDEN = {
    "seeds":                     [12345, 999, 7],
    "forward_matches":           [0.25, 0.75, 0.0],
    "reverse_matches":           [0.5, 0.125, 0.0],
    "window_size":               [6, 5, 0],
    "offset":                    [54, 3, 0],
    "trial_number":              [3, 1, 0],
    "skip_min":                  [1, 0, 0],
    "skip_max":                  [9, 2, 0],
    "skip_range":                [5, 6, 0],
    "forward_count":             [10.0, 1.0, 0.0],
    "reverse_count":             [11.0, 2.0, 0.0],
    "bidirectional_count":       [12.0, 3.0, 0.0],
    "intersection_count":        [13.0, 4.0, 0.0],
    "intersection_ratio":        [0.5, 0.25, 0.0],
    "intersection_weight":       [1.5, 0.5, 0.0],
    "bidirectional_selectivity": [0.125, 0.0625, 0.0],
    "forward_only_count":        [4.0, 6.0, 0.0],
    "reverse_only_count":        [5.0, 7.0, 0.0],
    "survivor_overlap_ratio":    [0.75, 0.5, 0.0],
    "score":                     [0.375, 0.4375, 0.0],
    "skip_mode":                 [0, 1, 0],
    "prng_type":                 [0, 0, 0],
}


# ─────────────────────────────────────────────────────────────────────────────
# Live-source drivers for the two writers.
# ─────────────────────────────────────────────────────────────────────────────

def run_convert_writer_with_key_order(records):
    """Drive convert_survivors_to_binary end-to-end.

    Returns ``(arrays, key_order)`` where ``key_order`` is ``tuple(z.files)``
    read INSIDE the loaded-NPZ context — i.e. the physical on-disk entry order,
    not the iteration order of anything this harness built.
    """
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "survivors.json")
        npz = os.path.join(tmp, "survivors_binary.npz")
        meta = os.path.join(tmp, "survivors_binary.meta.json")
        with open(src, "w") as f:
            json.dump(records, f)
        CSB.convert_json_to_npz(src, npz, meta)
        with np.load(npz) as z:
            return {k: z[k].copy() for k in z.files}, tuple(z.files)


def run_convert_writer(records):
    """Drive convert_survivors_to_binary end-to-end; return {name: ndarray}."""
    arrays, _key_order = run_convert_writer_with_key_order(records)
    return arrays


# `load_inline_writer()`, the module-level `_inline_survivors_to_arrays`
# binding and `run_inline_writer()` lived here until D3.5 removed the closure
# they read. See the migration-gate retirement note in the module docstring;
# they are deleted rather than adapted, because nothing remains in production
# for them to extract and a frozen copy would gate this file against itself.


def _rec(**kw):
    """Minimal survivor record with a seed, plus overrides."""
    base = {"seed": 1}
    base.update(kw)
    return base


def _convert_prng_ids(prng_type):
    """prng_type column (as a plain int list) from the live standalone writer."""
    recs = [_rec(seed=101, prng_type=prng_type)]
    return [int(v) for v in run_convert_writer(recs)["prng_type"]]


def _assert_raises_valueerror(label, fn):
    try:
        fn()
    except ValueError:
        return
    except Exception as exc:  # wrong exception type is also a failure
        raise AssertionError(
            f"{label}: expected ValueError, got {type(exc).__name__}: {exc}"
        )
    raise AssertionError(f"{label}: expected ValueError, but the call succeeded "
                         f"(silent fallback still present)")


# ─────────────────────────────────────────────────────────────────────────────
# Gates
# ─────────────────────────────────────────────────────────────────────────────

def e1_java_lcg_agrees_everywhere():
    """java_lcg encodes to literal 0 through the standalone writer AND canonical."""
    from utils.prng_encoding import encode_prng_type
    conv = _convert_prng_ids("java_lcg")
    assert conv == [ID_JAVA_LCG], f"convert writer java_lcg -> {conv}, expected [0]"
    assert encode_prng_type("java_lcg") == ID_JAVA_LCG, "canonical java_lcg != 0"


def e2_hybrid_is_one_not_zero():
    """java_lcg_hybrid encodes to canonical 1, NOT collapsed to 0.

    FAILS PRE-FIX: the 12-entry legacy table has no 'java_lcg_hybrid' key, so
    `.get(..., 0)` mislabels every hybrid survivor as java_lcg.
    """
    from utils.prng_encoding import encode_prng_type
    conv = _convert_prng_ids("java_lcg_hybrid")
    assert conv == [ID_JAVA_LCG_HYBRID], (
        f"convert writer java_lcg_hybrid -> {conv}, expected [1] "
        f"(got [0] = silent collapse to java_lcg)" if conv == [ID_JAVA_LCG]
        else f"convert writer java_lcg_hybrid -> {conv}, expected [1]"
    )
    assert encode_prng_type("java_lcg_hybrid") == ID_JAVA_LCG_HYBRID, (
        "canonical java_lcg_hybrid != 1")


def e3_shared_key_now_matches_canonical():
    """java_lcg_reverse encodes to canonical 3 (the legacy table said 1).

    FAILS PRE-FIX. One of the seven shared keys that disagreed on value.
    """
    from utils.prng_encoding import encode_prng_type
    conv = _convert_prng_ids("java_lcg_reverse")
    assert conv == [ID_JAVA_LCG_REVERSE], (
        f"convert writer java_lcg_reverse -> {conv}, expected [3]")
    assert encode_prng_type("java_lcg_reverse") == ID_JAVA_LCG_REVERSE, (
        "canonical java_lcg_reverse != 3")


def e4_unknown_prng_type_raises():
    """An unknown prng_type raises ValueError from the writer AND canonical.

    FAILS PRE-FIX: `.get(unknown, 0)` returns 0 and the NPZ silently claims the
    survivor was java_lcg.
    """
    from utils.prng_encoding import encode_prng_type
    unknown = "definitely_not_a_registered_prng"
    recs = [_rec(seed=202, prng_type=unknown)]
    _assert_raises_valueerror("convert writer / unknown prng_type",
                              lambda: run_convert_writer(recs))
    _assert_raises_valueerror("canonical encode_prng_type / unknown prng_type",
                              lambda: encode_prng_type(unknown))


def e5_unknown_skip_mode_raises():
    """An unknown skip_mode raises ValueError from the writer AND canonical.

    FAILS PRE-FIX: `.get(unknown, 0)` silently reports 'constant'.
    """
    from utils.prng_encoding import encode_skip_mode
    recs = [_rec(seed=303, prng_type="java_lcg", skip_mode="sideways")]
    _assert_raises_valueerror("convert writer / unknown skip_mode",
                              lambda: run_convert_writer(recs))
    _assert_raises_valueerror("canonical encode_skip_mode / unknown skip_mode",
                              lambda: encode_skip_mode("sideways"))


def e6_randu_identities_raise():
    """randu / randu_reverse raise ValueError — legacy ids 10/11 are NOT preserved.

    Per the Team Beta randu disposition these identities are unsupported and
    unreachable through the current registry-backed kernel producer path. This
    gate asserts ONLY that they now hard-fail; it makes no claim about whether
    they were ever historically emitted.

    FAILS PRE-FIX: the legacy table encodes them as 10 and 11.
    """
    from utils.prng_encoding import encode_prng_type
    for name in ("randu", "randu_reverse"):
        recs = [_rec(seed=404, prng_type=name)]
        _assert_raises_valueerror(f"convert writer / {name}",
                                  lambda r=recs: run_convert_writer(r))
        _assert_raises_valueerror(f"canonical encode_prng_type / {name}",
                                  lambda n=name: encode_prng_type(n))


def e7_skip_mode_numerics_unchanged():
    """skip_mode numerics are unchanged by the source-of-truth swap.

    constant -> literal 0, variable -> literal 1, through both writers. This is
    asserted rather than assumed: if canonical ever diverges from the legacy
    table this gate is the stop condition.
    """
    from utils.prng_encoding import encode_skip_mode
    recs = [_rec(seed=1, prng_type="java_lcg", skip_mode="constant"),
            _rec(seed=2, prng_type="java_lcg", skip_mode="variable")]
    for label, arrays in (("convert", run_convert_writer(recs)),):
        got = [int(v) for v in arrays["skip_mode"]]
        assert got == [SKIP_CONSTANT, SKIP_VARIABLE], (
            f"{label} writer skip_mode -> {got}, expected [0, 1]")
    assert encode_skip_mode("constant") == SKIP_CONSTANT
    assert encode_skip_mode("variable") == SKIP_VARIABLE


# [D3.0 correction round, Team Beta] E8 previously compared a SET and merely
# iterated NPZ_CONTRACT, so it documented an order it did not enforce: swapping
# two adjacent keys in the production writer's _EMPTY_NPZ_DTYPES left the whole
# gate 10/10 green. The tuple below is a SECOND, DELIBERATELY INDEPENDENT
# hand-transcription of the frozen savez order, existing solely for E8's
# positional assertion.
#
# DO NOT "dedupe" this against NPZ_CONTRACT, _EMPTY_NPZ_DTYPES, the production
# writer, or any shared schema constant. Deriving it from a shared source is
# exactly the defect this correction removes — two independent transcriptions
# that must agree is the whole point.
E8_EXPECTED_KEY_ORDER = (
    "seeds",
    "forward_matches",
    "reverse_matches",
    "window_size",
    "offset",
    "trial_number",
    "skip_min",
    "skip_max",
    "skip_range",
    "forward_count",
    "reverse_count",
    "bidirectional_count",
    "intersection_count",
    "intersection_ratio",
    "intersection_weight",
    "bidirectional_selectivity",
    "forward_only_count",
    "reverse_only_count",
    "survivor_overlap_ratio",
    "score",
    "skip_mode",
    "prng_type",
)


def e8_empty_output_is_rectangular():
    """Empty input produces exactly 22 zero-length arrays, in the frozen ORDER,
    with the frozen dtypes.

    FAILS PRE-FIX: convert_survivors_to_binary writes ONE array (seeds=[]), so an
    empty artifact is structurally different from a non-empty one.

    The positional assertion uses tuple(z.files) read inside the loaded-NPZ
    context — the physical on-disk entry order — against the independent
    E8_EXPECTED_KEY_ORDER transcription, so a reordering of the production
    writer's dtype map reds this gate instead of passing silently.
    """
    arrays, key_order = run_convert_writer_with_key_order([])
    assert len(arrays) == 22, (
        f"empty NPZ has {len(arrays)} arrays, expected 22: {sorted(arrays)}")
    assert set(arrays) == NPZ_KEYS, (
        f"empty NPZ key set mismatch; missing={sorted(NPZ_KEYS - set(arrays))} "
        f"unexpected={sorted(set(arrays) - NPZ_KEYS)}")
    assert key_order == E8_EXPECTED_KEY_ORDER, (
        "empty NPZ key ORDER differs from the frozen savez order.\n"
        f"  on disk:  {list(key_order)}\n"
        f"  expected: {list(E8_EXPECTED_KEY_ORDER)}\n"
        "  first divergence at index " + str(next(
            (i for i, (a, b) in enumerate(zip(key_order, E8_EXPECTED_KEY_ORDER))
             if a != b), min(len(key_order), len(E8_EXPECTED_KEY_ORDER)))))
    for name, dtype in NPZ_CONTRACT:
        assert arrays[name].dtype == np.dtype(dtype), (
            f"empty NPZ {name} dtype {arrays[name].dtype}, expected {dtype}")
        assert len(arrays[name]) == 0, (
            f"empty NPZ {name} length {len(arrays[name])}, expected 0")


def e9_known_identity_output_unchanged():
    """Known-identity java_lcg output is byte-for-byte what the pre-fix tree wrote.

    All 22 arrays are compared against the hand-transcribed pre-fix golden, for
    the standalone writer. This is the behavior-preservation gate: it is green
    pre-fix and must STAY green post-fix.
    """
    for label, arrays in (("convert", run_convert_writer(E9_FIXTURE)),):
        assert set(arrays) == NPZ_KEYS, (
            f"{label} writer key set mismatch: {sorted(set(arrays) ^ NPZ_KEYS)}")
        for name, dtype in NPZ_CONTRACT:
            expected = np.array(E9_GOLDEN[name], dtype=np.dtype(dtype))
            got = arrays[name]
            assert got.dtype == np.dtype(dtype), (
                f"{label} writer {name} dtype {got.dtype}, expected {dtype}")
            assert np.array_equal(got, expected), (
                f"{label} writer {name} changed: got {got.tolist()}, "
                f"pre-fix golden {expected.tolist()}")


def e10_mixed_column_carries_both_ids():
    """The live standalone writer's prng_type column for a mixed java_lcg
    constant/hybrid fixture carries BOTH canonical ids, not a collapsed {0}.

    FAILS PRE-FIX: the hybrid rows silently take the .get(...,0) fallback, so the
    whole column reads java_lcg and the distinction never reaches the ML feature
    surface.

    Reframed from the inline writer to the standalone converter by the D3.5
    migration-gate retirement; the assertion itself — a mixed column must carry
    0 AND 1, positionally — is unchanged.
    """
    recs = [
        _rec(seed=10, prng_type="java_lcg", skip_mode="constant"),
        _rec(seed=11, prng_type="java_lcg_hybrid", skip_mode="constant"),
        _rec(seed=12, prng_type="java_lcg", skip_mode="constant"),
        _rec(seed=13, prng_type="java_lcg_hybrid", skip_mode="constant"),
    ]
    col = run_convert_writer(recs)["prng_type"]
    got = [int(v) for v in col]
    assert got == [ID_JAVA_LCG, ID_JAVA_LCG_HYBRID,
                   ID_JAVA_LCG, ID_JAVA_LCG_HYBRID], (
        f"convert writer prng_type column -> {got}, expected [0, 1, 0, 1]")
    assert set(got) == {ID_JAVA_LCG, ID_JAVA_LCG_HYBRID}, (
        f"convert writer prng_type distinct values {sorted(set(got))}, "
        f"expected {{0, 1}} (a collapsed {{0}} is the pre-fix bug)")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

_results = []


def _check(name, fn):
    try:
        fn()
    except Exception:
        _results.append((name, False, traceback.format_exc()))
        print(f"  FAIL  {name}")
        return
    _results.append((name, True, ""))
    print(f"  ok    {name}")


def main():
    print("=" * 78)
    print("S172 Phase-5 D3.0 — canonical encoding + rectangular empty output gate")
    print("=" * 78)
    _check("E1: java_lcg -> 0 through the standalone writer + canonical",
           e1_java_lcg_agrees_everywhere)
    _check("E2: java_lcg_hybrid -> 1, not collapsed to 0",
           e2_hybrid_is_one_not_zero)
    _check("E3: java_lcg_reverse -> 3 (legacy said 1)",
           e3_shared_key_now_matches_canonical)
    _check("E4: unknown prng_type raises ValueError (writer + canonical)",
           e4_unknown_prng_type_raises)
    _check("E5: unknown skip_mode raises ValueError (writer + canonical)",
           e5_unknown_skip_mode_raises)
    _check("E6: randu / randu_reverse raise ValueError (ids 10/11 gone)",
           e6_randu_identities_raise)
    _check("E7: skip_mode numerics unchanged (constant 0 / variable 1)",
           e7_skip_mode_numerics_unchanged)
    _check("E8: empty input -> 22 rectangular zero-length arrays, frozen order",
           e8_empty_output_is_rectangular)
    _check("E9: known-identity java_lcg output == pre-fix golden (22/22)",
           e9_known_identity_output_unchanged)
    _check("E10: standalone writer mixed column carries {0, 1}",
           e10_mixed_column_carries_both_ids)

    print("=" * 78)
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{passed}/{total} D3.0 gate checks green\n")
    if passed != total:
        print("FAILURES (DO NOT COMMIT):\n")
        for name, ok, tb in _results:
            if not ok:
                print(f"--- {name} ---\n{tb}")
        return 1
    print("All D3.0 gate checks green — the standalone NPZ writer is on the "
          "canonical encoding and the empty case is rectangular (pending "
          "Team Alpha + "
          "Team Beta review).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
