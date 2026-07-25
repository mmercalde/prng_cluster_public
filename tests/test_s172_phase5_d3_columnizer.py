#!/usr/bin/env python3
"""
test_s172_phase5_d3_columnizer.py — S172 Phase-5 Deliverable D3 acceptance
harness (shared backend-neutral 24 -> 22 columnizer + structural validator).

Spec: docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3.md (REV3, Team Beta
approved), frozen against HEAD 66f0425.

Ten checks C1-C10, each constructed to FAIL on the wrong behavior (REV3 §1.2).

INDEPENDENT ORACLE — the G9 / E8 lesson, binding (REV3 §1.3). This harness does
NOT import `CANONICAL_ARRAY_CONTRACT`, `CANONICAL_RECORD_FIELDS`,
`_RENAMED_SOURCE_FIELDS` or `BASE_PRNG_FAMILIES` from the module under test and
assert against them. The 22 array names, their frozen ORDER, their dtypes, the
canonical 24 input field names and the full 24 -> 22 mapping are all
HAND-TRANSCRIBED below from the on-disk contract
(`convert_survivors_to_binary._EMPTY_NPZ_DTYPES` :50-73 and the identical
`np.savez_compressed` call order :200-223) and from
`miner/range_miner_npz_writer.CANONICAL_RECORD_FIELDS` :150-158 — read once,
written out here as literals. Asserting a constant against itself is the exact
defect corrected in D1.1's G9 and again in D3.0's E8.

The two numeric identity literals used below (`java_lcg` -> 0,
`java_lcg_hybrid` -> 1; `constant` -> 0, `variable` -> 1) are likewise INTEGER
LITERALS, hand-transcribed exactly as D3.0's harness does — not read back out of
`utils/prng_encoding`. They are the first two keys of the alphabetically-sorted
KERNEL_REGISTRY, so they are also the least likely to shift.

C8 drives BOTH live legacy columnizers for parity. `convert_survivors_to_binary`
is imported and driven end-to-end (JSON in -> NPZ on disk -> np.load out). The
inline `_survivors_to_arrays` is a nested closure inside a ~2000-line function
and is not importable, so it is extracted FROM LIVE SOURCE by AST line-range
(the same technique D3.0's harness uses) — editing the production seam changes
what this gate runs; nothing is copied into this file.

C10 is the Rule-2 mutation proof. Each mutant is a TEXTUAL edit applied to the
live `utils/canonical_arrays.py` source and exec'd into a fresh namespace; the
full C1-C9 suite is then re-run against that mutated module and must go RED.
The production file on disk is never modified.

Against the unmodified tree the expected result is 10/10 green.
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import os
import sys
import tempfile
import textwrap
import traceback
from typing import Dict, List, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import utils.canonical_arrays as PROD                      # noqa: E402
import convert_survivors_to_binary as CSB                  # noqa: E402

_MODULE_PATH = os.path.join(_ROOT, "utils", "canonical_arrays.py")
_WOIF = os.path.join(_ROOT, "window_optimizer_integration_final.py")


# ═════════════════════════════════════════════════════════════════════════════
# HAND-TRANSCRIBED ORACLE  (never imported from the code under test)
# ═════════════════════════════════════════════════════════════════════════════

# The frozen 22-array contract: name, dtype, in frozen order.
EXPECTED_CONTRACT: Tuple[Tuple[str, str], ...] = (
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
)
EXPECTED_ARRAY_NAMES: Tuple[str, ...] = tuple(n for n, _d in EXPECTED_CONTRACT)

# The canonical 24-field input record.
EXPECTED_RECORD_FIELDS: Tuple[str, ...] = (
    "seed", "forward_match_rate", "reverse_match_rate", "score",
    "window_size", "offset", "skip_min", "skip_max", "skip_range", "sessions",
    "trial_number", "prng_base", "skip_mode", "prng_type",
    "forward_count", "reverse_count", "bidirectional_count",
    "intersection_count", "intersection_ratio",
    "forward_only_count", "reverse_only_count",
    "survivor_overlap_ratio", "bidirectional_selectivity", "intersection_weight",
)

# The 24 -> 22 mapping, written out in full. 24 - 2 = 22: `sessions` and
# `prng_base` do not become arrays; two fields are RENAMED; 20 map 1:1.
EXPECTED_ARRAY_SOURCE: Dict[str, str] = {
    "seeds":                     "seed",                # renamed
    "forward_matches":           "forward_match_rate",  # renamed
    "reverse_matches":           "reverse_match_rate",  # renamed
    "window_size":               "window_size",
    "offset":                    "offset",
    "trial_number":              "trial_number",
    "skip_min":                  "skip_min",
    "skip_max":                  "skip_max",
    "skip_range":                "skip_range",
    "forward_count":             "forward_count",
    "reverse_count":             "reverse_count",
    "bidirectional_count":       "bidirectional_count",
    "intersection_count":        "intersection_count",
    "intersection_ratio":        "intersection_ratio",
    "intersection_weight":       "intersection_weight",
    "bidirectional_selectivity": "bidirectional_selectivity",
    "forward_only_count":        "forward_only_count",
    "reverse_only_count":        "reverse_only_count",
    "survivor_overlap_ratio":    "survivor_overlap_ratio",
    "score":                     "score",
    "skip_mode":                 "skip_mode",           # str -> uint8 id
    "prng_type":                 "prng_type",           # str -> uint8 id
}
EXPECTED_NON_ARRAY_FIELDS: Tuple[str, ...] = ("sessions", "prng_base")

# Hand-transcribed identity ids (integer literals, D3.0 style).
ID_JAVA_LCG = 0
ID_JAVA_LCG_HYBRID = 1
ID_SKIP_CONSTANT = 0
ID_SKIP_VARIABLE = 1

# Hand-transcribed sanity assertions on the oracle itself.
assert len(EXPECTED_CONTRACT) == 22
assert len(EXPECTED_RECORD_FIELDS) == 24
assert len(EXPECTED_ARRAY_SOURCE) == 22
assert set(EXPECTED_ARRAY_SOURCE) == set(EXPECTED_ARRAY_NAMES)
assert (set(EXPECTED_ARRAY_SOURCE.values()) | set(EXPECTED_NON_ARRAY_FIELDS)
        == set(EXPECTED_RECORD_FIELDS))


# ═════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════════════

def rec_constant(**over) -> dict:
    """A valid canonical 24-field record in CONSTANT skip mode."""
    r = {
        "seed":                      7,
        "forward_match_rate":        0.25,
        "reverse_match_rate":        0.75,
        "score":                     0.5,
        "window_size":               60,
        "offset":                    3,
        "skip_min":                  0,
        "skip_max":                  7,
        "skip_range":                7,
        "sessions":                  ["midday", "evening"],
        "trial_number":              11,
        "prng_base":                 "java_lcg",
        "skip_mode":                 "constant",
        "prng_type":                 "java_lcg",
        "forward_count":             40,
        "reverse_count":             25,
        "bidirectional_count":       10,
        "intersection_count":        10,
        "intersection_ratio":        0.2,
        "forward_only_count":        30,
        "reverse_only_count":        15,
        "survivor_overlap_ratio":    0.25,
        "bidirectional_selectivity": 1.6,
        "intersection_weight":       0.16,
    }
    r.update(over)
    return r


def rec_variable(**over) -> dict:
    """A valid canonical 24-field record in VARIABLE skip mode."""
    r = rec_constant()
    r.update({
        "seed":                      4294967295,   # uint32 max
        "forward_match_rate":        1.0,
        "reverse_match_rate":        0.0,
        "score":                     0.5,
        "window_size":               120,
        "offset":                    -5,
        "skip_min":                  -3,
        "skip_max":                  9,
        "skip_range":                12,
        "sessions":                  [],
        "trial_number":              12,
        "prng_base":                 "java_lcg",
        "skip_mode":                 "variable",
        "prng_type":                 "java_lcg_hybrid",
        "forward_count":             3,
        "reverse_count":             0,
        "bidirectional_count":       0,
        "intersection_count":        0,
        "intersection_ratio":        0.0,
        "forward_only_count":        3,
        "reverse_only_count":        0,
        "survivor_overlap_ratio":    0.0,
        "bidirectional_selectivity": 3.0,
        "intersection_weight":       0.0,
    })
    r.update(over)
    return r


def _f32(values) -> np.ndarray:
    return np.array(values, dtype=np.float32)


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _expect_valueerror(label, fn, must_mention=()):
    """Assert `fn()` raises a ValueError whose text mentions each fragment."""
    try:
        fn()
    except ValueError as exc:
        text = str(exc)
        for frag in must_mention:
            _assert(frag in text,
                    f"{label}: ValueError text does not mention {frag!r}: {text}")
        return str(exc)
    except Exception as exc:
        raise AssertionError(
            f"{label}: expected ValueError, got {type(exc).__name__}: {exc}")
    raise AssertionError(
        f"{label}: expected ValueError, but the call SUCCEEDED (fail-closed lost)")


# ═════════════════════════════════════════════════════════════════════════════
# Live legacy columnizers (C8 parity)
# ═════════════════════════════════════════════════════════════════════════════

def run_convert_writer(records) -> Tuple[Dict[str, np.ndarray], Tuple[str, ...]]:
    """Drive convert_survivors_to_binary end-to-end; return (arrays, key order).

    `key_order` is `tuple(z.files)` read inside the loaded-NPZ context — the
    physical on-disk entry order, not the iteration order of anything built here.
    """
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "survivors.json")
        npz = os.path.join(tmp, "survivors_binary.npz")
        meta = os.path.join(tmp, "survivors_binary.meta.json")
        with open(src, "w") as f:
            json.dump(records, f)
        with contextlib.redirect_stdout(io.StringIO()):
            CSB.convert_json_to_npz(src, npz, meta)
        with np.load(npz) as z:
            return {k: z[k].copy() for k in z.files}, tuple(z.files)


def load_inline_writer():
    """Extract `_survivors_to_arrays` FROM LIVE SOURCE and return it callable.

    Same AST line-range technique as D3.0's harness: locate the FunctionDef in
    the module AST, walk BACKWARDS over the contiguous run of local imports and
    underscore-name assignments that immediately precede it (the encoding seam),
    and exec that exact line range. Whatever the seam actually does is what runs.
    """
    with open(_WOIF) as f:
        src = f.read()
    tree = ast.parse(src)

    parent_body, idx = None, None
    for node in ast.walk(tree):
        for _field, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            for i, child in enumerate(value):
                if isinstance(child, ast.FunctionDef) and child.name == "_survivors_to_arrays":
                    parent_body, idx = value, i
    _assert(parent_body is not None,
            "_survivors_to_arrays not found in window_optimizer_integration_final.py")

    start = idx
    while start > 0:
        prev = parent_body[start - 1]
        if isinstance(prev, (ast.Import, ast.ImportFrom)):
            start -= 1
            continue
        if isinstance(prev, ast.Assign) and prev.targets and all(
            isinstance(t, ast.Name) and t.id.startswith("_") for t in prev.targets
        ):
            start -= 1
            continue
        break

    first_line = parent_body[start].lineno
    last_line = parent_body[idx].end_lineno
    block = textwrap.dedent("\n".join(src.splitlines()[first_line - 1:last_line]))
    ns = {"__name__": "_d3_inline_seam"}
    exec(compile(block, f"{_WOIF}:{first_line}-{last_line}", "exec"), ns)  # noqa: S102
    fn = ns.get("_survivors_to_arrays")
    _assert(callable(fn), "extracted _survivors_to_arrays is not callable")
    return fn


_inline_survivors_to_arrays = load_inline_writer()


# ═════════════════════════════════════════════════════════════════════════════
# Mutation engine (C10)
# ═════════════════════════════════════════════════════════════════════════════

with open(_MODULE_PATH) as _f:
    _PROD_SRC = _f.read()


class _Mod:
    """A thin namespace wrapper so gate functions can run against a mutant."""

    def __init__(self, ns):
        self._ns = ns

    def __getattr__(self, name):
        try:
            return self._ns[name]
        except KeyError:
            raise AttributeError(name)


def build_mutant(replacements: List[Tuple[str, str]]) -> _Mod:
    """Apply textual replacements to the LIVE module source and exec the result.

    Every anchor must occur exactly once; a vanished anchor means the mutation
    no longer models what it claims to, which is itself a red.
    """
    src = _PROD_SRC
    for old, new in replacements:
        n = src.count(old)
        _assert(n == 1, f"mutation anchor occurs {n} times (expected 1): {old!r}")
        src = src.replace(old, new)
    ns = {"__name__": "_d3_mutant", "__file__": _MODULE_PATH}
    exec(compile(src, f"{_MODULE_PATH}:MUTANT", "exec"), ns)  # noqa: S102
    return _Mod(ns)


# --- shared mutation fragments ----------------------------------------------
_MISSING_LINE = "    missing = _RECORD_FIELD_SET - keys"
_FIELD_ACCESSOR = "    return record[name]"


def _relax_missing(*fields: str) -> Tuple[str, str]:
    """Disable the missing-key wall for `fields` only (all of them if empty).

    Scoping the relaxation matters: a blanket `missing = set()` lets an
    UNRELATED field's absence blow up as a raw KeyError, so the mutant dies for
    an incidental reason and the evidence no longer demonstrates the defect
    being modelled. Narrowing it means the gate must red on the intended case.
    """
    if not fields:
        return (_MISSING_LINE, "    missing = set()  # MUTANT: wall disabled")
    allowed = ", ".join(repr(f) for f in fields)
    return (_MISSING_LINE,
            f"    missing = _RECORD_FIELD_SET - keys - {{{allowed}}}  # MUTANT")


MUTANTS: List[Tuple[str, List[Tuple[str, str]]]] = [
    # ---- output-shape mutants -------------------------------------------
    ("M01 dropped array (score removed from the contract)", [
        ('        ("score",                     "float32"),\n', ""),
    ]),
    ("M02 added array (23rd array emitted past the postcondition)", [
        ("    validate_array_bundle(arrays)\n    return arrays",
         '    validate_array_bundle(arrays)\n'
         '    arrays["mutant_extra"] = arrays["seeds"].astype(np.float32)\n'
         "    return arrays"),
    ]),
    ("M03 reordered keys (skip_mode <-> prng_type)", [
        ('        ("skip_mode",                 "uint8"),\n'
         '        ("prng_type",                 "uint8"),',
         '        ("prng_type",                 "uint8"),\n'
         '        ("skip_mode",                 "uint8"),'),
    ]),
    ("M04 renamed key (score -> scores at emit)", [
        ("    return arrays",
         '    return {("scores" if _k == "score" else _k): _v '
         "for _k, _v in arrays.items()}"),
    ]),
    ("M05 wrong dtype (forward_count float32 -> int32)", [
        ('        ("forward_count",             "float32"),',
         '        ("forward_count",             "int32"),'),
    ]),
    ("M06 match-rate rename swapped (forward <-> reverse)", [
        ('    "forward_matches": "forward_match_rate",\n'
         '    "reverse_matches": "reverse_match_rate",',
         '    "forward_matches": "reverse_match_rate",\n'
         '    "reverse_matches": "forward_match_rate",'),
    ]),
    ("M07 sessions emitted as a 23rd array", [
        ("    return arrays",
         '    return {**arrays, "sessions": arrays["seeds"].astype(np.float32)}'),
    ]),
    ("M08 prng_base emitted as a 23rd array", [
        ("    return arrays",
         '    return {**arrays, "prng_base": arrays["seeds"].astype(np.float32)}'),
    ]),
    # ---- validation-relaxation mutants -----------------------------------
    ("M09 restored silent default for a missing field", [
        _relax_missing("forward_match_rate"),
        (_FIELD_ACCESSOR, "    return record.get(name, 0.0)  # MUTANT"),
    ]),
    ("M10 restored '-> java_lcg' terminal default", [
        _relax_missing("prng_type"),
        (_FIELD_ACCESSOR,
         '    if name == "prng_type":  # MUTANT: legacy resolution chain\n'
         '        return record.get("prng_type",\n'
         '                          record.get("prng_base", "java_lcg"))\n'
         "    return record[name]"),
    ]),
    ("M11 accepting a missing explicit prng_type (derived from prng_base)", [
        _relax_missing("prng_type"),
        (_FIELD_ACCESSOR,
         '    if name == "prng_type" and "prng_type" not in record:  # MUTANT\n'
         '        _b = record.get("prng_base")\n'
         '        return _b if record.get("skip_mode") == "constant" '
         'else f"{_b}_hybrid"\n'
         "    return record[name]"),
    ]),
    ("M12 cross-direction match-rate fallback", [
        _relax_missing("forward_match_rate", "reverse_match_rate"),
        (_FIELD_ACCESSOR,
         '    if name == "forward_match_rate":  # MUTANT: Ruling G violated\n'
         '        return record.get("forward_match_rate",\n'
         '                          record.get("reverse_match_rate", 0.0))\n'
         '    if name == "reverse_match_rate":\n'
         '        return record.get("reverse_match_rate",\n'
         '                          record.get("forward_match_rate", 0.0))\n'
         "    return record[name]"),
    ]),
    ("M13 ignoring an inconsistent prng_base/prng_type", [
        ("    if prng_type != expected_prng_type:",
         "    if False:  # MUTANT: identity inconsistency ignored"),
    ]),
    ("M14 allowing silent integer overflow", [
        ("    if not (low <= value <= high):",
         "    if False:  # MUTANT: destination range check removed"),
    ]),
    ("M15 registry membership only, base-family restriction omitted [A1]", [
        ("    if prng_base not in BASE_PRNG_FAMILIES:",
         "    if prng_base not in KERNEL_REGISTRY:  # MUTANT: membership only"),
    ]),
    ("M16 post-conversion np.float32 finiteness check removed [A2]", [
        ("    if not bool(np.isfinite(np.float32(value))):",
         "    if False:  # MUTANT: float32 representability unchecked"),
    ]),
    ("M17 accepting a fractional count [A2]", [
        ("        if value != math.floor(value):",
         "        if False:  # MUTANT: count integrality unchecked"),
    ]),
    # ---- ordering / identity mutants -------------------------------------
    ("M18 unique-seed wall (collapses C6's cross-mode pair)", [
        ("    return arrays",
         "    _seen, _keep = set(), []  # MUTANT: unique-seed wall\n"
         '    for _i, _s in enumerate(arrays["seeds"].tolist()):\n'
         "        if _s not in _seen:\n"
         "            _seen.add(_s)\n"
         "            _keep.append(_i)\n"
         "    _idx = np.array(_keep, dtype=np.int64)\n"
         "    return {_k: _v[_idx] for _k, _v in arrays.items()}"),
    ]),
    ("M19 sorting records inside D3 (by mode)", [
        ("    for index, record in enumerate(records):",
         "    for index, record in enumerate(  # MUTANT: mode-first sort\n"
         '            sorted(records, key=lambda r: str(r.get("skip_mode", "")))):'),
    ]),
    ("M20 sorting records inside D3 (by seed)", [
        ("    for index, record in enumerate(records):",
         "    for index, record in enumerate(  # MUTANT: seed sort\n"
         '            sorted(records, key=lambda r: r.get("seed", 0))):'),
    ]),
    ("M21 internal validate_array_bundle call removed + malformed bundle", [
        ("    validate_array_bundle(arrays)\n    return arrays",
         '    arrays["score"] = arrays["score"].reshape(-1, 1)  # MUTANT\n'
         "    return arrays  # MUTANT: postcondition self-check removed"),
    ]),
]


# ═════════════════════════════════════════════════════════════════════════════
# C1 — 22 arrays, exact names, exact ORDER, exact dtypes
# ═════════════════════════════════════════════════════════════════════════════

def c1_contract_shape(mod=PROD):
    arrays = mod.records_to_arrays([rec_constant(), rec_variable()])

    observed = tuple(arrays.keys())
    _assert(len(observed) == 22,
            f"expected exactly 22 arrays, got {len(observed)}: {list(observed)}")
    # Asserted as a TUPLE, not a set — D3.0's E8 lesson.
    _assert(observed == EXPECTED_ARRAY_NAMES,
            f"array name/ORDER mismatch.\n  expected {list(EXPECTED_ARRAY_NAMES)}"
            f"\n  observed {list(observed)}")

    for name, dtype_name in EXPECTED_CONTRACT:
        arr = arrays[name]
        _assert(isinstance(arr, np.ndarray),
                f"{name!r} is {type(arr).__name__}, not ndarray")
        _assert(arr.dtype == np.dtype(dtype_name),
                f"{name!r} dtype {arr.dtype}, expected {dtype_name}")
        _assert(arr.ndim == 1,
                f"{name!r} ndim {arr.ndim} (shape {arr.shape}), expected 1-D")
        _assert(arr.shape[0] == 2,
                f"{name!r} length {arr.shape[0]}, expected 2")

    # The public contract constant, if exposed, must agree in order + dtype.
    contract = tuple((n, np.dtype(d)) for n, d in mod.CANONICAL_ARRAY_CONTRACT)
    _assert(contract == tuple((n, np.dtype(d)) for n, d in EXPECTED_CONTRACT),
            f"CANONICAL_ARRAY_CONTRACT disagrees with the hand oracle: {contract}")


# ═════════════════════════════════════════════════════════════════════════════
# C2 — value correctness against hand-computed expectations, both renames
# ═════════════════════════════════════════════════════════════════════════════

def c2_values(mod=PROD):
    arrays = mod.records_to_arrays([rec_constant(), rec_variable()])

    expected = {
        "seeds":                     np.array([7, 4294967295], dtype=np.uint32),
        # RENAME: forward_match_rate -> forward_matches
        "forward_matches":           _f32([0.25, 1.0]),
        # RENAME: reverse_match_rate -> reverse_matches
        "reverse_matches":           _f32([0.75, 0.0]),
        "window_size":               np.array([60, 120], dtype=np.int32),
        "offset":                    np.array([3, -5], dtype=np.int32),
        "trial_number":              np.array([11, 12], dtype=np.int32),
        "skip_min":                  np.array([0, -3], dtype=np.int32),
        "skip_max":                  np.array([7, 9], dtype=np.int32),
        "skip_range":                np.array([7, 12], dtype=np.int32),
        "forward_count":             _f32([40.0, 3.0]),
        "reverse_count":             _f32([25.0, 0.0]),
        "bidirectional_count":       _f32([10.0, 0.0]),
        "intersection_count":        _f32([10.0, 0.0]),
        "intersection_ratio":        _f32([0.2, 0.0]),
        "intersection_weight":       _f32([0.16, 0.0]),
        "bidirectional_selectivity": _f32([1.6, 3.0]),
        "forward_only_count":        _f32([30.0, 3.0]),
        "reverse_only_count":        _f32([15.0, 0.0]),
        "survivor_overlap_ratio":    _f32([0.25, 0.0]),
        "score":                     _f32([0.5, 0.5]),
        "skip_mode":                 np.array([ID_SKIP_CONSTANT, ID_SKIP_VARIABLE],
                                              dtype=np.uint8),
        "prng_type":                 np.array([ID_JAVA_LCG, ID_JAVA_LCG_HYBRID],
                                              dtype=np.uint8),
    }
    _assert(set(expected) == set(EXPECTED_ARRAY_NAMES), "oracle expectation drift")

    for name in EXPECTED_ARRAY_NAMES:
        got, want = arrays[name], expected[name]
        _assert(got.dtype == want.dtype,
                f"{name!r} dtype {got.dtype} != expected {want.dtype}")
        _assert(np.array_equal(got, want),
                f"{name!r} values {got.tolist()} != expected {want.tolist()}")

    # The renames are load-bearing and must NOT be symmetric: prove the two
    # rate columns are distinguishable, so a swap cannot pass silently.
    _assert(not np.array_equal(arrays["forward_matches"], arrays["reverse_matches"]),
            "fixture is rate-symmetric; a forward<->reverse swap would be invisible")


# ═════════════════════════════════════════════════════════════════════════════
# C3 — `sessions` and `prng_base` are absent from the output
# ═════════════════════════════════════════════════════════════════════════════

def c3_non_array_fields_absent(mod=PROD):
    for records in ([], [rec_constant()], [rec_constant(), rec_variable()]):
        arrays = mod.records_to_arrays(records)
        for field in EXPECTED_NON_ARRAY_FIELDS:
            _assert(field not in arrays,
                    f"{field!r} must NOT become an array (24 - 2 = 22), but the "
                    f"output carries it: {list(arrays.keys())}")
        _assert(len(arrays) == 22,
                f"expected 22 arrays, got {len(arrays)}: {list(arrays.keys())}")


# ═════════════════════════════════════════════════════════════════════════════
# C4 — empty input -> 22 rectangular zero-length arrays, frozen order/dtypes
# ═════════════════════════════════════════════════════════════════════════════

def c4_empty_is_rectangular(mod=PROD):
    for empty in ([], iter([]), (r for r in [])):
        arrays = mod.records_to_arrays(empty)
        observed = tuple(arrays.keys())
        _assert(observed == EXPECTED_ARRAY_NAMES,
                f"empty-case name/ORDER mismatch: {list(observed)}")
        for name, dtype_name in EXPECTED_CONTRACT:
            arr = arrays[name]
            _assert(arr.dtype == np.dtype(dtype_name),
                    f"empty {name!r} dtype {arr.dtype}, expected {dtype_name}")
            _assert(arr.ndim == 1, f"empty {name!r} ndim {arr.ndim}, expected 1-D")
            _assert(arr.shape == (0,),
                    f"empty {name!r} shape {arr.shape}, expected (0,) — the "
                    f"empty artifact must be RECTANGULAR (D3.0 E8)")


# ═════════════════════════════════════════════════════════════════════════════
# C5 — fail-closed matrix
# ═════════════════════════════════════════════════════════════════════════════

def c5_fail_closed(mod=PROD):
    R = mod.records_to_arrays

    # --- every one of the 24 canonical fields missing in turn --------------
    # Index 1 of a 2-record list, so the reported index must be the REAL index.
    for field in EXPECTED_RECORD_FIELDS:
        bad = rec_variable()
        del bad[field]
        _expect_valueerror(
            f"missing {field!r}",
            lambda b=bad: R([rec_constant(), b]),
            must_mention=(field, "record 1"))

    # --- an extra, unexpected field ---------------------------------------
    _expect_valueerror(
        "extra field",
        lambda: R([rec_constant(seed_recovery_hint="nope")]),
        must_mention=("seed_recovery_hint", "record 0"))

    # --- explicitly listed omitted-field cases (subset of the loop above,
    #     restated because REV3 §5-C5 names them individually) -------------
    for field in ("sessions", "prng_base", "prng_type",
                  "forward_match_rate", "reverse_match_rate"):
        bad = rec_constant()
        del bad[field]
        _expect_valueerror(f"missing {field!r} (explicit)",
                           lambda b=bad: R([b]), must_mention=(field, "record 0"))

    # --- `sessions` shape violations (validated though never an array) -----
    for bad_sessions, label in (
        ("midday", "scalar string"),
        (("midday",), "tuple"),
        (None, "None"),
        (["midday", 7], "non-string member"),
    ):
        _expect_valueerror(
            f"sessions {label}",
            lambda s=bad_sessions: R([rec_constant(sessions=s)]),
            must_mention=("sessions", "record 0"))

    # --- identity inconsistency -------------------------------------------
    _expect_valueerror(
        "inconsistent prng_base/prng_type",
        lambda: R([rec_constant(prng_base="lcg32", prng_type="java_lcg")]),
        must_mention=("record 0",))
    _expect_valueerror(
        "inconsistent skip_mode/prng_type (variable but non-hybrid type)",
        lambda: R([rec_constant(skip_mode="variable", prng_type="java_lcg")]),
        must_mention=("record 0",))
    _expect_valueerror(
        "inconsistent skip_mode/prng_type (constant but hybrid type)",
        lambda: R([rec_constant(prng_type="java_lcg_hybrid")]),
        must_mention=("record 0",))
    _expect_valueerror(
        "valid base with an unrelated-but-valid prng_type",
        lambda: R([rec_constant(prng_base="java_lcg", prng_type="lcg32")]),
        must_mention=("record 0",))

    # --- unknown identities -------------------------------------------------
    _expect_valueerror(
        "unknown explicit base",
        lambda: R([rec_constant(prng_base="not_a_family",
                                prng_type="not_a_family")]),
        must_mention=("not_a_family",))
    _expect_valueerror(
        "unknown explicit type",
        lambda: R([rec_constant(prng_type="randu")]),
        must_mention=("randu",))
    _expect_valueerror(
        "unknown skip_mode",
        lambda: R([rec_constant(skip_mode="sometimes")]),
        must_mention=("sometimes",))

    # --- [A1] base-family restriction: directional / derived identities ----
    for bad_base in ("java_lcg_reverse", "java_lcg_hybrid",
                     "java_lcg_hybrid_reverse"):
        # Constructed so the EQUALITY rule alone would pass — this is exactly
        # the hole registry-membership-only validation leaves open.
        _expect_valueerror(
            f"prng_base = {bad_base!r} (derived identity, equality-consistent)",
            lambda b=bad_base: R([rec_constant(prng_base=b, prng_type=b,
                                               skip_mode="constant")]),
            must_mention=(bad_base, "record 0"))

    # --- adversarial numerics ---------------------------------------------
    numeric_cases = [
        ("negative seed",              dict(seed=-1),                     "seed"),
        ("seed above uint32",          dict(seed=2 ** 32),                "seed"),
        ("int32 overflow window_size", dict(window_size=2 ** 31),         "window_size"),
        ("int32 underflow offset",     dict(offset=-2 ** 31 - 1),         "offset"),
        ("NaN forward match rate",     dict(forward_match_rate=float("nan")),
                                                                          "forward_match_rate"),
        ("infinite score",             dict(score=float("inf")),          "score"),
        ("bool as an integer field",   dict(trial_number=True),           "trial_number"),
        ("bool as a float field",      dict(survivor_overlap_ratio=True),
                                                                          "survivor_overlap_ratio"),
        ("float32 overflow (finite in Python)",
                                       dict(bidirectional_selectivity=1e300),
                                                                          "bidirectional_selectivity"),
        ("fractional count",           dict(forward_count=1.5),           "forward_count"),
        ("negative count",             dict(reverse_count=-1),            "reverse_count"),
        ("boolean count",              dict(intersection_count=True),     "intersection_count"),
        # --- the three unit-interval fields, both ends of the bound ---------
        # REV3 §4.5 freezes ONE rule over three fields:
        #     forward_match_rate, reverse_match_rate, score  in [0.0, 1.0]
        # Coverage has to be per-field AND per-end, because the two ways of
        # breaking it are independent: reclassifying a single field out of
        # _UNIT_INTERVAL_ARRAYS drops that field's ceiling only, while weakening
        # the shared comparison drops one END across all three at once. A row
        # that probes a different field, or the same field at the other end,
        # does not cover either case.
        #
        # The pre-existing rows probed only reverse_match_rate's ceiling; the
        # score rows probed non-finiteness and non-numericity, which fail at
        # checks 3 and 1 long before the bounds branch is ever reached.
        ("match rate above 1.0",       dict(reverse_match_rate=1.5),      "reverse_match_rate"),
        ("forward_match_rate above 1.0",
                                       dict(forward_match_rate=1.5),      "forward_match_rate"),
        ("score above 1.0",            dict(score=1.5),                   "score"),
        ("score below 0.0",            dict(score=-0.1),                  "score"),
        ("negative ratio",             dict(intersection_ratio=-0.1),     "intersection_ratio"),
        ("non-numeric float field",    dict(score="0.5"),                 "score"),
        ("float in an integer field",  dict(skip_min=1.5),                "skip_min"),
    ]
    for label, over, field in numeric_cases:
        # The rejection must IDENTIFY THE OFFENDING FIELD, not merely raise.
        # Matched on the QUOTED form (`'score'`, the `{field!r}` rendering every
        # numeric rejection in `_check_int`/`_check_float` uses), so a bare
        # substring collision cannot satisfy the assertion and a message that
        # blames some other field reds.
        _expect_valueerror(label, lambda o=over: R([rec_constant(**o)]),
                           must_mention=(repr(field), "record 0"))

    # --- [A3] an iterator input is consumed EXACTLY once -------------------
    records = [rec_constant(), rec_variable(), rec_constant(seed=99)]
    seen: List[int] = []

    def _counting():
        for r in records:
            seen.append(1)
            yield r

    arrays = R(_counting())
    _assert(len(seen) == 3, f"generator traversed {len(seen)} times, expected 3")
    _assert(arrays["seeds"].tolist() == [7, 4294967295, 99],
            f"generator input mis-ordered: {arrays['seeds'].tolist()}")
    # A one-shot iterator (already exhausted) must yield an empty bundle, not
    # a crash — proof that nothing re-traverses or calls len()/indexing.
    once = iter(records)
    list(once)
    _assert(R(once)["seeds"].shape == (0,), "exhausted iterator did not yield 0 rows")


# ═════════════════════════════════════════════════════════════════════════════
# C6 — cross-mode seed survives (no unique-seed wall)
# ═════════════════════════════════════════════════════════════════════════════

def c6_cross_mode_seed_survives(mod=PROD):
    seed = 424242
    const = rec_constant(seed=seed, forward_match_rate=0.9, reverse_match_rate=0.1,
                         score=0.5, forward_count=11, reverse_count=2,
                         bidirectional_selectivity=5.5)
    var = rec_variable(seed=seed, forward_match_rate=0.2, reverse_match_rate=0.8,
                       score=0.5, forward_count=4, reverse_count=9,
                       bidirectional_selectivity=0.5)
    arrays = mod.records_to_arrays([const, var])

    _assert(arrays["seeds"].tolist() == [seed, seed],
            f"cross-mode duplicate seed collapsed: {arrays['seeds'].tolist()} "
            f"(a unique-seed wall is forbidden — duplication is legitimate "
            f"until L2)")
    _assert(arrays["skip_mode"].tolist() == [ID_SKIP_CONSTANT, ID_SKIP_VARIABLE],
            f"mode column wrong: {arrays['skip_mode'].tolist()}")
    _assert(arrays["prng_type"].tolist() == [ID_JAVA_LCG, ID_JAVA_LCG_HYBRID],
            f"prng_type column wrong: {arrays['prng_type'].tolist()}")
    # each row keeps ITS OWN mode's rates and aggregates
    _assert(np.array_equal(arrays["forward_matches"], _f32([0.9, 0.2])),
            f"forward rates not per-mode: {arrays['forward_matches'].tolist()}")
    _assert(np.array_equal(arrays["reverse_matches"], _f32([0.1, 0.8])),
            f"reverse rates not per-mode: {arrays['reverse_matches'].tolist()}")
    _assert(np.array_equal(arrays["forward_count"], _f32([11.0, 4.0])),
            f"forward_count not per-mode: {arrays['forward_count'].tolist()}")
    _assert(np.array_equal(arrays["bidirectional_selectivity"], _f32([5.5, 0.5])),
            f"selectivity not per-mode: "
            f"{arrays['bidirectional_selectivity'].tolist()}")


# ═════════════════════════════════════════════════════════════════════════════
# C7 — exact order preservation [C1]
# ═════════════════════════════════════════════════════════════════════════════

def _nontrivial_sequence() -> List[dict]:
    """Deliberately neither seed-sorted nor mode-grouped."""
    return [
        rec_variable(seed=900, forward_match_rate=0.11, score=0.11,
                     forward_count=9,  trial_number=3),
        rec_constant(seed=12,  forward_match_rate=0.22, score=0.22,
                     forward_count=8,  trial_number=1),
        rec_variable(seed=400, forward_match_rate=0.33, score=0.33,
                     forward_count=7,  trial_number=4),
        rec_constant(seed=900, forward_match_rate=0.44, score=0.44,
                     forward_count=6,  trial_number=2),
        rec_constant(seed=12,  forward_match_rate=0.55, score=0.55,
                     forward_count=5,  trial_number=0),
    ]


def c7_order_preserved(mod=PROD):
    records = _nontrivial_sequence()
    seeds = [r["seed"] for r in records]
    modes = [ID_SKIP_CONSTANT if r["skip_mode"] == "constant" else ID_SKIP_VARIABLE
             for r in records]
    _assert(seeds != sorted(seeds), "fixture is seed-sorted; C7 would be blind")
    _assert(modes != sorted(modes), "fixture is mode-grouped; C7 would be blind")

    arrays = mod.records_to_arrays(records)
    _assert(arrays["seeds"].tolist() == seeds,
            f"seed order changed: {arrays['seeds'].tolist()} != {seeds}")
    _assert(arrays["skip_mode"].tolist() == modes,
            f"mode order changed: {arrays['skip_mode'].tolist()} != {modes}")
    _assert(np.array_equal(arrays["forward_matches"],
                           _f32([r["forward_match_rate"] for r in records])),
            "forward_matches order changed")
    _assert(np.array_equal(arrays["trial_number"],
                           np.array([r["trial_number"] for r in records],
                                    dtype=np.int32)),
            "trial_number order changed")

    # Shuffling the input produces the SAME corresponding shuffle in all 22.
    perm = [3, 0, 4, 2, 1]
    shuffled = [records[i] for i in perm]
    shuffled_arrays = mod.records_to_arrays(shuffled)
    for name in EXPECTED_ARRAY_NAMES:
        want = arrays[name][np.array(perm, dtype=np.int64)]
        _assert(np.array_equal(shuffled_arrays[name], want),
                f"{name!r} is not shuffle-equivariant: "
                f"{shuffled_arrays[name].tolist()} != {want.tolist()} — D3 "
                f"must own NO ordering policy [REV3 C1]")


# ═════════════════════════════════════════════════════════════════════════════
# C8 — parity with BOTH corrected post-D3.0 legacy columnizers
# ═════════════════════════════════════════════════════════════════════════════

def c8_parity_with_legacy(mod=PROD):
    # All 24 fields explicit, so NEITHER legacy writer's compatibility fallback
    # is exercised; already in the desired order (D3 preserves input order).
    records = [
        rec_constant(seed=7),
        rec_variable(seed=4294967295),
        rec_constant(seed=123456, forward_match_rate=0.0, reverse_match_rate=1.0,
                     score=0.5, offset=-5, skip_min=-3, skip_range=12),
    ]
    for r in records:
        _assert(set(r) == set(EXPECTED_RECORD_FIELDS),
                f"parity fixture is not the full explicit 24: {sorted(set(r))}")

    d3 = mod.records_to_arrays(records)

    legacy_convert, key_order = run_convert_writer(records)
    _assert(key_order == EXPECTED_ARRAY_NAMES,
            f"convert_survivors_to_binary on-disk key order drifted: "
            f"{list(key_order)}")

    legacy_inline = _inline_survivors_to_arrays(records)

    for label, legacy in (("convert_survivors_to_binary", legacy_convert),
                          ("inline _survivors_to_arrays", legacy_inline)):
        for name in EXPECTED_ARRAY_NAMES:
            _assert(name in legacy, f"{label}: missing array {name!r}")
            got, want = d3[name], legacy[name]
            _assert(got.dtype == want.dtype,
                    f"{label}: {name!r} dtype {got.dtype} != legacy {want.dtype}")
            _assert(np.array_equal(got, want),
                    f"{label}: {name!r} values differ — D3 {got.tolist()} vs "
                    f"legacy {want.tolist()}. A real semantic difference beyond "
                    f"the Ruling-G finding is a STOP condition (REV3 §7).")


# ═════════════════════════════════════════════════════════════════════════════
# C9 — the validator, on HAND-BUILT bundles
# ═════════════════════════════════════════════════════════════════════════════

def c9_validator(mod=PROD):
    V = mod.validate_array_bundle

    def good() -> Dict[str, np.ndarray]:
        """Hand-built, not produced by records_to_arrays."""
        return {name: np.zeros(3, dtype=np.dtype(dt))
                for name, dt in EXPECTED_CONTRACT}

    V(good())  # the hand-built well-formed bundle must PASS

    # dropped key
    b = good(); del b["score"]
    _expect_valueerror("validator: dropped key", lambda: V(b))

    # added key
    b = good(); b["mutant_extra"] = np.zeros(3, dtype=np.float32)
    _expect_valueerror("validator: added key", lambda: V(b))

    # reordered keys (same names, same dtypes, wrong ITERATION order)
    src = good()
    b = {n: src[n] for n in list(EXPECTED_ARRAY_NAMES[:-2])
         + [EXPECTED_ARRAY_NAMES[-1], EXPECTED_ARRAY_NAMES[-2]]}
    _assert(set(b) == set(EXPECTED_ARRAY_NAMES) and len(b) == 22,
            "reorder fixture changed the key SET, not just the order")
    _expect_valueerror("validator: reordered keys", lambda: V(b))

    # renamed key
    src = good()
    b = {("scores" if n == "score" else n): src[n] for n in EXPECTED_ARRAY_NAMES}
    _expect_valueerror("validator: renamed key", lambda: V(b))

    # wrong dtype
    b = good(); b["forward_count"] = np.zeros(3, dtype=np.int32)
    _expect_valueerror("validator: wrong dtype", lambda: V(b))

    # unequal lengths
    b = good(); b["offset"] = np.zeros(2, dtype=np.int32)
    _expect_valueerror("validator: unequal lengths", lambda: V(b))

    # two-dimensional array (matching outer length, still not a 1-D column)
    b = good(); b["score"] = np.zeros((3, 1), dtype=np.float32)
    _expect_valueerror("validator: 2-D array", lambda: V(b))

    # scalar (0-d) array
    b = good(); b["score"] = np.float32(0.0).reshape(())
    _expect_valueerror("validator: scalar array", lambda: V(b))

    # non-NumPy value under one key
    b = good(); b["score"] = [0.0, 0.0, 0.0]
    _expect_valueerror("validator: non-ndarray value", lambda: V(b))

    # a zero-length bundle is rectangular and legal
    V({name: np.zeros(0, dtype=np.dtype(dt)) for name, dt in EXPECTED_CONTRACT})


# ═════════════════════════════════════════════════════════════════════════════
# C10 — mutation proof
# ═════════════════════════════════════════════════════════════════════════════

_MUTATION_TARGETS = (
    ("C1", c1_contract_shape),
    ("C2", c2_values),
    ("C3", c3_non_array_fields_absent),
    ("C4", c4_empty_is_rectangular),
    ("C5", c5_fail_closed),
    ("C6", c6_cross_mode_seed_survives),
    ("C7", c7_order_preserved),
    ("C8", c8_parity_with_legacy),
    ("C9", c9_validator),
)

_MUTATION_REPORT: List[str] = []


def c10_mutation_proof(mod=PROD):
    """Every mutant must be KILLED by at least one of C1-C9.

    `mod` is accepted for signature uniformity but deliberately unused: C10
    always mutates the live production source, never a mutant of a mutant.
    """
    survivors = []
    for label, replacements in MUTANTS:
        try:
            mutant = build_mutant(replacements)
        except Exception as exc:                       # anchor gone / bad syntax
            _MUTATION_REPORT.append(
                f"  {label}\n      KILLED at build: "
                f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}")
            continue

        reds = []
        for gate_name, gate_fn in _MUTATION_TARGETS:
            try:
                gate_fn(mutant)
            except Exception as exc:
                first = str(exc).splitlines()[0] if str(exc) else ""
                reds.append((gate_name, f"{type(exc).__name__}: {first[:150]}"))
        if reds:
            sig = "; ".join(g for g, _ in reds)
            lines = [f"  {label}", f"      KILLED by [{sig}]"]
            lines += [f"        {g} red -> {msg}" for g, msg in reds]
            _MUTATION_REPORT.append("\n".join(lines))
        else:
            survivors.append(label)
            _MUTATION_REPORT.append(f"  {label}\n      *** SURVIVED ***")

    _assert(not survivors,
            f"{len(survivors)} mutant(s) SURVIVED the C1-C9 gate suite — the "
            f"gate does not fail on the wrong behavior (REV3 §1.2): {survivors}")
    _assert(len(MUTANTS) == 21,
            f"expected the 21-mutant C10 set, found {len(MUTANTS)}")


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════

_results: List[Tuple[str, bool, str]] = []


def _check(name, fn):
    try:
        fn()
        _results.append((name, True, ""))
        print(f"  ok    {name}")
    except Exception:
        _results.append((name, False, traceback.format_exc()))
        print(f"  FAIL  {name}")


def main():
    print("=" * 78)
    print("S172 Phase-5 D3 — canonical 24 -> 22 columnizer + structural validator")
    print("=" * 78)

    _check("C1: 22 arrays, exact names, exact ORDER, exact dtypes", c1_contract_shape)
    _check("C2: value correctness incl. both match-rate renames", c2_values)
    _check("C3: sessions + prng_base absent from the output", c3_non_array_fields_absent)
    _check("C4: empty input -> 22 rectangular zero-length arrays", c4_empty_is_rectangular)
    _check("C5: fail-closed matrix (24 missing, extra, identity, numerics)", c5_fail_closed)
    _check("C6: cross-mode duplicate seed survives (no unique-seed wall)",
           c6_cross_mode_seed_survives)
    _check("C7: exact order preservation + shuffle equivariance", c7_order_preserved)
    _check("C8: parity with BOTH live legacy columnizers", c8_parity_with_legacy)
    _check("C9: validator on hand-built bundles", c9_validator)
    _check("C10: mutation proof (21 mutants, each killed by C1-C9)",
           c10_mutation_proof)

    if _MUTATION_REPORT:
        print("\n" + "-" * 78)
        print("C10 mutation evidence (red signature per mutant):")
        print("-" * 78)
        for line in _MUTATION_REPORT:
            print(line)

    print("=" * 78)
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{passed}/{total} D3 gate checks green\n")
    if passed != total:
        print("FAILURES (DO NOT COMMIT):\n")
        for name, ok, tb in _results:
            if not ok:
                print(f"--- {name} ---\n{tb}")
        return 1
    print("All D3 gate checks green — the shared 24 -> 22 columnizer is "
          "order-preserving, fail-closed and parity-clean against both live "
          "legacy writers (pending Team Alpha + Team Beta review).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
