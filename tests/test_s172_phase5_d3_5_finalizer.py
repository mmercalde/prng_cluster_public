#!/usr/bin/env python3
"""
test_s172_phase5_d3_5_finalizer.py — S172 Phase-5 Deliverable D3.5 acceptance
harness (shared run finalizer: L2 winner selection, L3 array-domain merge,
immutable-generation publication).

Spec: docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3_5.md (REV3.1, Team Beta
approved), frozen against HEAD 70cd6f0.
Extended by docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3_5_B.md (REV2, Team
Beta approved), frozen against HEAD 46a3828 — Seed-Domain v1.1, gates S1-S9.

Gates F1-F51 and S1-S9, plus the F26 mutation set (which carries the S9 mutants).
Every gate is constructed to FAIL on the wrong behavior (REV3.1 §1.2).

INDEPENDENT ORACLES — the G9 / E8 / C1 lesson, binding (§1.3). This harness does
NOT import `CANONICAL_ARRAY_CONTRACT`, `SIDECAR_REQUIRED_KEYS`, `ALL_NPZ_NAME`,
`ACCUMULATOR_DIRNAME` or any other production constant and assert against it.
The 22 array names and their frozen ORDER and dtypes, the 32 sidecar keys, the
directory layout, the alias names, the identity encodings and every tie outcome
are HAND-TRANSCRIBED below as literals, read once from the spec and from the
on-disk contract. Asserting a constant against itself is the exact defect
corrected in D1.1's G9, again in D3.0's E8 and again in D3's C1.

SYNTHETIC PRIORS ONLY (§1.4). No real prior generation exists — Ruling F is a
clean start — and none is required. Priors here are produced by the finalizer's
own publication path and then, for the negative gates, corrupted on disk. Where
a `current` pointer is built by hand it is built to the SAME rules production
uses; §8b is never bypassed.

F26 is the Rule-2 mutation proof: each mutant is a TEXTUAL edit applied to the
live `utils/run_finalizer.py` (or, for the integration mutants, to the live
`window_optimizer_integration_final.py`) source and exec'd into a fresh
namespace. The production files on disk are never modified.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import shutil
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.run_finalizer as PROD                              # noqa: E402

_MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "utils", "run_finalizer.py")
_INTEGRATION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "window_optimizer_integration_final.py")

with open(_MODULE_PATH) as _f:
    _PROD_SRC = _f.read()
with open(_INTEGRATION_PATH) as _f:
    _INTEGRATION_SRC = _f.read()


# ═════════════════════════════════════════════════════════════════════════════
# Hand-transcribed oracles — literals, never imported from the module under test
# ═════════════════════════════════════════════════════════════════════════════

# The frozen 22 arrays, in the frozen ORDER, with the frozen dtypes. Transcribed
# from convert_survivors_to_binary._EMPTY_NPZ_DTYPES / the savez call order.
ORACLE_ARRAYS: Tuple[Tuple[str, str], ...] = (
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
ORACLE_ARRAY_NAMES = tuple(n for n, _ in ORACLE_ARRAYS)

# The exact 32 sidecar keys — the 23 of §7.3 plus the nine seed-domain fields of
# D3.5-B REV2 §3 — IN GLOBAL ALPHABETICAL ORDER, hand-transcribed from the brief
# and never imported from `SIDECAR_REQUIRED_KEYS`. `sidecar_sha256` is
# deliberately NOT one of them [REV3 C1].
ORACLE_SIDECAR_KEY_ORDER: Tuple[str, ...] = (
    "artifact_schema_version",
    "artifact_sha256",
    "canonical_map_hash",
    "created_at",
    "encoding_contract_version",
    "exhaustive_over",                      # NEW
    "external_seed_transform",              # NEW
    "final_row_count",
    "generation_id",
    "l2_winner_count",
    "parent_artifact_sha256",
    "parent_generation_id",
    "parent_sidecar_sha256",
    "prior_row_count",
    "prng_base",
    "raw_candidate_count",
    "repository_commit",
    "repository_tree_clean",
    "row_count",
    "run_id",
    "seed_count",
    "seed_domain_contract",                 # NEW
    "seed_domain_end_exclusive",            # NEW
    "seed_domain_start",                    # NEW
    "seed_effective_bits",                  # NEW
    "seed_end_exclusive",
    "seed_high16_prefix",                   # NEW
    "seed_semantics",                       # NEW
    "seed_start",
    "seed_storage_dtype",                   # NEW
    "sidecar_schema_version",
    "skip_modes_executed",
)
ORACLE_SIDECAR_KEYS = frozenset(ORACLE_SIDECAR_KEY_ORDER)

# The nine frozen seed-domain values (D3.5-B REV2 §3), hand-transcribed. Every
# one is a fixed v1.1 constant — none is caller-supplied.
ORACLE_SEED_DOMAIN: Tuple[Tuple[str, object], ...] = (
    ("seed_semantics",            "internal_state"),
    ("seed_storage_dtype",        "uint32"),
    ("seed_effective_bits",       32),
    ("seed_high16_prefix",        0),
    ("seed_domain_contract",      "v1.1-stratum"),
    ("seed_domain_start",         0),
    ("seed_domain_end_exclusive", 4294967296),
    ("exhaustive_over",           "high16=0 stratum only"),
    ("external_seed_transform",   None),
)
ORACLE_SEED_DOMAIN_NAMES = tuple(n for n, _ in ORACLE_SEED_DOMAIN)

# The seven stratum-identifying fields of §5's coordinate identity. Retained as
# documentation; a certified lineage must in fact agree on all NINE [R3].
ORACLE_STRATUM_IDENTITY = (
    "seed_domain_contract", "seed_semantics", "seed_storage_dtype",
    "seed_effective_bits", "seed_high16_prefix", "seed_domain_start",
    "seed_domain_end_exclusive",
)

# §5's per-link contract — fourteen fields. The five below are already required
# properties of a homogeneous certified lineage and, at 46a3828, NONE of them
# was compared link-by-link: `_validate_chain` checked hashes, ids, cycles and
# existence only [R1].
ORACLE_LINEAGE_CONTRACT_FIELDS = (
    "prng_base", "artifact_schema_version", "sidecar_schema_version",
    "encoding_contract_version", "canonical_map_hash",
)
ORACLE_LINEAGE_KEYS = ORACLE_LINEAGE_CONTRACT_FIELDS + ORACLE_SEED_DOMAIN_NAMES

# The three version strings. Exactly ONE moves in D3.5-B (§4).
ORACLE_SIDECAR_SCHEMA_VERSION = "s172.d3_5.provenance.v1.1"
ORACLE_PRE_V1_1_SIDECAR_SCHEMA_VERSION = "s172.d3_5.provenance.v1"
ORACLE_ARTIFACT_SCHEMA_VERSION = "s172.d3.arrays.v1"
ORACLE_ENCODING_CONTRACT_VERSION = "s172.phase0.encoding.v1"

# Layout (§7.1), transcribed from the spec.
ORACLE_ACCUM_DIR = ".s172_accumulator"
ORACLE_GENERATIONS = "generations"
ORACLE_CURRENT = "current"
ORACLE_SIDECAR_NAME = "provenance.json"
ORACLE_ALL_NPZ = "bidirectional_survivors_all.npz"
ORACLE_BINARY_NPZ = "bidirectional_survivors_binary.npz"

# Identity encodings — INTEGER LITERALS, exactly as D3.0 and D3 transcribe them.
# `java_lcg` and `java_lcg_hybrid` are the first two keys of the alphabetically
# sorted KERNEL_REGISTRY.
ORACLE_JAVA_LCG_ID = 0
ORACLE_JAVA_LCG_HYBRID_ID = 1
ORACLE_SKIP_CONSTANT_ID = 0
ORACLE_SKIP_VARIABLE_ID = 1

ORACLE_UINT32_EXCLUSIVE_MAX = 4294967296        # 2**32, written out


# ═════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════════════

_TEMP_ROOTS: List[str] = []


def fresh_root() -> Path:
    path = tempfile.mkdtemp(prefix="d3_5_gate_")
    _TEMP_ROOTS.append(path)
    return Path(path)


def cand(seed: int, trial: int, mode: str, score: float, **over) -> dict:
    """A canonical 24-field record. Keys written out, never derived."""
    base = over.pop("prng_base", "java_lcg")
    record = {
        "seed":                      seed,
        "forward_match_rate":        score,
        "reverse_match_rate":        score,
        "score":                     score,
        "window_size":               12,
        "offset":                    3,
        "skip_min":                  1,
        "skip_max":                  4,
        "skip_range":                3,
        "sessions":                  ["midday", "evening"],
        "trial_number":              trial,
        "prng_base":                 base,
        "skip_mode":                 mode,
        "prng_type":                 base if mode == "constant" else base + "_hybrid",
        "forward_count":             9.0,
        "reverse_count":             7.0,
        "bidirectional_count":       4.0,
        "intersection_count":        4.0,
        "intersection_ratio":        0.25,
        "forward_only_count":        5.0,
        "reverse_only_count":        3.0,
        "survivor_overlap_ratio":    0.44,
        "bidirectional_selectivity": 1.2857,
        "intersection_weight":       0.25,
    }
    record.update(over)
    return record


def publish(root: Path, candidates, RF=PROD, **over):
    kwargs = dict(
        output_root=root,
        run_id="step1_java_lcg_0",
        prng_base="java_lcg",
        skip_modes_executed=("constant", "variable"),
        seed_start=0,
        seed_count=1000,
        repository_commit="a" * 40,
        repository_tree_clean=True,
    )
    kwargs.update(over)
    return RF.finalize_run(candidates, **kwargs)


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _expect_raises(label, error_types, fn, must_mention=()):
    try:
        fn()
    except error_types as exc:
        text = str(exc)
        for fragment in must_mention:
            _assert(fragment in text,
                    f"{label}: raised {type(exc).__name__} but its message does "
                    f"not mention {fragment!r}: {text}")
        return exc
    except Exception as exc:                    # noqa: BLE001
        raise AssertionError(
            f"{label}: expected {error_types}, got "
            f"{type(exc).__name__}: {exc}") from exc
    raise AssertionError(f"{label}: expected {error_types}, nothing raised")


def read_current(root: Path):
    pointer = root / ORACLE_ACCUM_DIR / ORACLE_CURRENT
    if not os.path.islink(pointer):
        return None
    return os.readlink(pointer)


def load_bundle(path: Path) -> Tuple[Dict[str, np.ndarray], Tuple[str, ...]]:
    with np.load(path) as handle:
        order = tuple(handle.files)
        arrays = {name: handle[name] for name in order}
    return arrays, order


def sha256_path(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _docstring_nodes(tree: ast.AST) -> set:
    """Every node that is a DOCSTRING, so source gates can ignore prose.

    The finalizer's docstrings deliberately NAME the things it must never do
    ("populates neither MinerTrialAssembly.binary_npz_path...", "the
    convert_survivors_to_binary.py subprocess..."). A blunt substring search
    would therefore red on the very comments that document the prohibition, and
    would also miss a real call written with a differently-spelled literal. The
    gates below assert over executable references instead.
    """
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                found.add(id(body[0].value))
    return found


def _executable_strings(tree: ast.AST) -> List[str]:
    docstrings = _docstring_nodes(tree)
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docstrings]


def _referenced_names(tree: ast.AST) -> set:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split(".")[0])
            names.add(node.name.split(".")[-1])
    return names


# ═════════════════════════════════════════════════════════════════════════════
# F1-F8 — L2 / L3 semantics (Ruling D T1-T8)
# ═════════════════════════════════════════════════════════════════════════════

def f1_new_new_unequal(RF=PROD):
    """trial 8 constant 0.70 vs trial 3 variable 0.80 -> trial 3 variable."""
    root = fresh_root()
    result = publish(root, [cand(5, 8, "constant", 0.70),
                            cand(5, 3, "variable", 0.80)], RF=RF)
    arrays, _ = load_bundle(result.binary_npz_path)
    _assert(list(arrays["seeds"]) == [5], f"expected one row, got {arrays['seeds']}")
    _assert(int(arrays["trial_number"][0]) == 3,
            f"F1: higher score must win — expected trial 3, got "
            f"{int(arrays['trial_number'][0])}")
    _assert(int(arrays["skip_mode"][0]) == ORACLE_SKIP_VARIABLE_ID,
            "F1: the winner is the variable-mode record")
    _assert(int(arrays["prng_type"][0]) == ORACLE_JAVA_LCG_HYBRID_ID,
            "F1: a variable winner carries the hybrid prng_type id")


def f2_equal_score_lower_trial_wins(RF=PROD):
    """LOAD-BEARING: rejects a global mode-first rule.

    trial 3 variable 0.80 vs trial 8 constant 0.80 -> trial 3 VARIABLE. If mode
    were ordered before trial_number the constant record would win.
    """
    root = fresh_root()
    result = publish(root, [cand(5, 3, "variable", 0.80),
                            cand(5, 8, "constant", 0.80)], RF=RF)
    arrays, _ = load_bundle(result.binary_npz_path)
    _assert(int(arrays["trial_number"][0]) == 3,
            f"F2: equal score must fall to the LOWER trial_number — expected 3, "
            f"got {int(arrays['trial_number'][0])}")
    _assert(int(arrays["skip_mode"][0]) == ORACLE_SKIP_VARIABLE_ID,
            "F2: constant-before-variable is a WITHIN-TRIAL tiebreak only; a "
            "global mode-first rule would have picked the trial-8 constant row")


def f3_same_trial_constant_wins(RF=PROD):
    root = fresh_root()
    result = publish(root, [cand(5, 4, "variable", 0.80),
                            cand(5, 4, "constant", 0.80)], RF=RF)
    arrays, _ = load_bundle(result.binary_npz_path)
    _assert(int(arrays["skip_mode"][0]) == ORACLE_SKIP_CONSTANT_ID,
            "F3: within one trial, constant beats variable on an equal score")


def f4_float32_tie(RF=PROD):
    """Two Python floats differing only beyond float32 precision are a TIE."""
    a = 0.8
    b = float(np.float64(np.float32(0.8)) + 1e-12)
    _assert(a != b, "F4 fixture: the two Python floats must differ")
    _assert(np.float32(a) == np.float32(b),
            "F4 fixture: the two values must be equal under float32")
    root = fresh_root()
    result = publish(root, [cand(5, 9, "constant", b),
                            cand(5, 2, "constant", a)], RF=RF)
    arrays, _ = load_bundle(result.binary_npz_path)
    _assert(int(arrays["trial_number"][0]) == 2,
            f"F4: values equal under float32 are an exact tie and the LOWER "
            f"trial wins — expected 2, got {int(arrays['trial_number'][0])}. "
            f"Comparing pre-rounding Python floats would have picked trial 9.")


def f5_prior_new_unequal_replaces(RF=PROD):
    root = fresh_root()
    publish(root, [cand(5, 1, "constant", 0.40)], RF=RF)
    result = publish(root, [cand(5, 2, "constant", 0.90)], RF=RF)
    arrays, _ = load_bundle(result.binary_npz_path)
    _assert(list(arrays["seeds"]) == [5], "F5: one row per seed after L3")
    _assert(abs(float(arrays["score"][0]) - 0.90) < 1e-6,
            f"F5: a strictly greater new score replaces the prior row, got "
            f"{float(arrays['score'][0])}")
    _assert(result.prior_row_count == 1, "F5: the prior contributed one row")


def f6_prior_tie_retained_byte_for_byte(RF=PROD):
    """Equal score RETAINS the prior, byte-for-byte in every one of the 22 arrays."""
    root = fresh_root()
    first = publish(root, [cand(5, 1, "variable", 0.80),
                           cand(11, 1, "constant", 0.30)], RF=RF)
    prior_arrays, _ = load_bundle(first.binary_npz_path)
    prior_row = {n: prior_arrays[n][list(prior_arrays["seeds"]).index(5)]
                 for n in ORACLE_ARRAY_NAMES}

    # Same score, different trial AND different mode — the L2 tiebreakers must
    # NOT reach across the L3 boundary and displace a certified row.
    result = publish(root, [cand(5, 7, "constant", 0.80)], RF=RF)
    arrays, _ = load_bundle(result.binary_npz_path)
    idx = list(arrays["seeds"]).index(5)
    for name in ORACLE_ARRAY_NAMES:
        got = arrays[name][idx]
        want = prior_row[name]
        _assert(got.tobytes() == want.tobytes(),
                f"F6: array {name!r} for the tied seed is not byte-identical to "
                f"the retained prior row (prior {want!r}, got {got!r})")
    _assert(int(arrays["skip_mode"][idx]) == ORACLE_SKIP_VARIABLE_ID,
            "F6: the retained row is the prior's variable-mode row")


def f7_same_trial_same_mode_collision(RF=PROD):
    root = fresh_root()
    exc = _expect_raises(
        "F7", RF.RunFinalizerError,
        lambda: publish(root, [cand(5, 4, "constant", 0.80),
                               cand(5, 4, "constant", 0.20)], RF=RF),
        must_mention=("same-trial",))
    _assert(isinstance(exc, RF.AccumulatorConsistencyError),
            f"F7: expected a dedicated accumulator-consistency error, got "
            f"{type(exc).__name__}")
    _assert(read_current(root) is None,
            "F7: nothing may be published when the accumulator is inconsistent")


def f8_order_independence(RF=PROD):
    batch = [
        cand(5, 8, "constant", 0.70), cand(5, 3, "variable", 0.80),
        cand(5, 3, "constant", 0.80), cand(9, 2, "constant", 0.55),
        cand(9, 6, "variable", 0.55), cand(2, 1, "constant", 0.10),
    ]
    reference = None
    # A deterministic set of permutations — no RNG, so a failure is reproducible.
    for rotation in range(len(batch)):
        shuffled = batch[rotation:] + batch[:rotation]
        for ordering in (shuffled, list(reversed(shuffled))):
            root = fresh_root()
            result = publish(root, list(ordering), RF=RF)
            arrays, _ = load_bundle(result.binary_npz_path)
            signature = tuple(
                (int(arrays["seeds"][i]), int(arrays["trial_number"][i]),
                 int(arrays["skip_mode"][i]), float(arrays["score"][i]))
                for i in range(len(arrays["seeds"])))
            if reference is None:
                reference = signature
            _assert(signature == reference,
                    f"F8: L2 selection is order-dependent — {signature} vs "
                    f"{reference}")
    # Hand-computed expected outcome, not read back from the module.
    # Scores are compared in the artifact's own float32 domain — the stored
    # column is float32, so 0.10 is 0.10000000149011612 there.
    expected = tuple(
        (seed, trial, mode, float(np.float32(score))) for seed, trial, mode, score in (
            (2, 1, ORACLE_SKIP_CONSTANT_ID, 0.10),
            (5, 3, ORACLE_SKIP_CONSTANT_ID, 0.80),
            (9, 2, ORACLE_SKIP_CONSTANT_ID, 0.55),
        ))
    _assert(reference == expected,
            f"F8: unexpected winner set {reference}, expected {expected}")


# ═════════════════════════════════════════════════════════════════════════════
# F9-F15 — publication and crash safety
# ═════════════════════════════════════════════════════════════════════════════

class _Patch:
    """Temporarily replace module attributes (the injectable §7.2 seams)."""

    def __init__(self, module, **attrs):
        self.module, self.attrs, self.saved = module, attrs, {}

    def __enter__(self):
        for name, value in self.attrs.items():
            self.saved[name] = getattr(self.module, name)
            setattr(self.module, name, value)
        return self

    def __exit__(self, *exc):
        for name, value in self.saved.items():
            setattr(self.module, name, value)
        return False


def _boom(*_a, **_kw):
    raise RuntimeError("injected publication failure")


def _published_state(root: Path):
    """A fingerprint of everything a reader could observe as current."""
    pointer = read_current(root)
    if pointer is None:
        return None
    gen = root / ORACLE_ACCUM_DIR / pointer
    return (pointer,
            sha256_path(gen / ORACLE_BINARY_NPZ),
            sha256_path(gen / ORACLE_SIDECAR_NAME))


def f9_publication_failure_leaves_prior_intact(RF=PROD):
    root = fresh_root()
    publish(root, [cand(5, 1, "constant", 0.40)], RF=RF)
    before = _published_state(root)
    with _Patch(RF, _write_npz=_boom):
        _expect_raises("F9", Exception,
                       lambda: publish(root, [cand(9, 1, "constant", 0.90)], RF=RF))
    after = _published_state(root)
    _assert(before == after,
            f"F9: the prior artifact AND sidecar must be byte-identical after a "
            f"failed publication — {before} vs {after}")


def f10_merge_failure_writes_nothing(RF=PROD):
    root = fresh_root()
    publish(root, [cand(5, 1, "constant", 0.40)], RF=RF)
    before = _published_state(root)
    generations = root / ORACLE_ACCUM_DIR / ORACLE_GENERATIONS
    dirs_before = sorted(p.name for p in generations.iterdir())
    with _Patch(RF, _l3_merge=_boom):
        _expect_raises("F10", Exception,
                       lambda: publish(root, [cand(9, 1, "constant", 0.90)], RF=RF))
    _assert(_published_state(root) == before, "F10: prior generation changed")
    _assert(sorted(p.name for p in generations.iterdir()) == dirs_before,
            "F10: a merge failure must write no canonical artifact at all")


def f11_sidecar_write_failure(RF=PROD):
    root = fresh_root()
    publish(root, [cand(5, 1, "constant", 0.40)], RF=RF)
    before = _published_state(root)
    with _Patch(RF, _write_and_fsync_bytes=_boom):
        _expect_raises("F11", Exception,
                       lambda: publish(root, [cand(9, 1, "constant", 0.90)], RF=RF))
    _assert(_published_state(root) == before,
            "F11: a sidecar write failure must accept no new generation")
    generations = root / ORACLE_ACCUM_DIR / ORACLE_GENERATIONS
    _assert(not any(p.name.startswith(".tmp-") for p in generations.iterdir()),
            "F11: the unreferenced staging directory must not be left behind as "
            "a would-be generation")


def f12_parent_artifact_hash_mismatch(RF=PROD):
    root = fresh_root()
    g1 = publish(root, [cand(5, 1, "constant", 0.40)], RF=RF)
    publish(root, [cand(9, 1, "constant", 0.50)], RF=RF)
    # Corrupt the ANCESTOR's payload; its child's recorded parent_artifact_sha256
    # no longer describes it.
    victim = g1.generation_dir / ORACLE_BINARY_NPZ
    data = bytearray(victim.read_bytes())
    data[-1] ^= 0xFF
    victim.write_bytes(bytes(data))
    before = read_current(root)
    exc = _expect_raises("F12", RF.RunFinalizerError,
                         lambda: publish(root, [cand(11, 1, "constant", 0.60)], RF=RF))
    _assert(isinstance(exc, RF.PriorGenerationError),
            f"F12: expected a prior-generation rejection, got {type(exc).__name__}")
    _assert(read_current(root) == before, "F12: current changed")


def f13_prior_without_sidecar(RF=PROD):
    root = fresh_root()
    g1 = publish(root, [cand(5, 1, "constant", 0.40)], RF=RF)
    (g1.generation_dir / ORACLE_SIDECAR_NAME).unlink()
    exc = _expect_raises("F13", RF.RunFinalizerError,
                         lambda: publish(root, [cand(9, 1, "constant", 0.50)], RF=RF))
    _assert(isinstance(exc, RF.PriorGenerationError),
            f"F13: a prior without a sidecar must FAIL CLOSED with a "
            f"prior-generation rejection, got {type(exc).__name__}")


def f14_uncertified_historical_artifact_never_imported(RF=PROD):
    """An uncertified NPZ is never imported — assert the raise AND that no row
    of it reaches the output. There is no filename-based trust (Ruling F)."""
    root = fresh_root()
    accum = root / ORACLE_ACCUM_DIR
    generations = accum / ORACLE_GENERATIONS
    generations.mkdir(parents=True)
    historical = generations / ("hist-" + "0" * 4 + "--" + "b" * 64)
    historical.mkdir()
    forensic_seed = 777
    arrays = {
        name: np.array([_historical_value(name, forensic_seed)], dtype=dtype)
        for name, dtype in ORACLE_ARRAYS
    }
    for name in (ORACLE_ALL_NPZ, ORACLE_BINARY_NPZ):
        with open(historical / name, "wb") as handle:
            np.savez_compressed(handle, **arrays)
    os.symlink(f"{ORACLE_GENERATIONS}/{historical.name}", accum / ORACLE_CURRENT)

    exc = _expect_raises("F14", RF.RunFinalizerError,
                         lambda: publish(root, [cand(5, 1, "constant", 0.90)], RF=RF))
    _assert(isinstance(exc, RF.PriorGenerationError),
            f"F14: expected a prior-generation rejection, got {type(exc).__name__}")
    # And nothing from it leaked anywhere.
    for gen_dir in generations.iterdir():
        if gen_dir == historical:
            continue
        published, _ = load_bundle(gen_dir / ORACLE_BINARY_NPZ)
        _assert(forensic_seed not in [int(s) for s in published["seeds"]],
                "F14: a row of the uncertified historical artifact reached a "
                "published generation")


def _historical_value(name, seed):
    if name == "seeds":
        return seed
    if name in ("skip_mode", "prng_type"):
        return 0
    if name in ("window_size", "offset", "trial_number", "skip_min", "skip_max",
                "skip_range"):
        return 1
    return 0.5


def f15_no_fallback_writer(RF=PROD, source=None):
    """Source-level AND behavioral: no fallback writer, no subprocess spawn."""
    src = _PROD_SRC if source is None else source
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _assert(alias.name.split(".")[0] != "subprocess",
                        "F15 (source): the finalizer imports `subprocess`")
        if isinstance(node, ast.ImportFrom):
            _assert((node.module or "").split(".")[0] != "subprocess",
                    "F15 (source): the finalizer imports from `subprocess`")
        if isinstance(node, ast.Attribute) and node.attr in (
                "system", "popen", "execv", "execvp", "spawnv", "fork"):
            _assert(False, f"F15 (source): the finalizer calls os.{node.attr}")
    for literal in _executable_strings(tree):
        _assert("convert_survivors_to_binary" not in literal,
                f"F15 (source): the finalizer carries the executable literal "
                f"{literal!r} — the legacy fallback converter")
    _assert("subprocess" not in _referenced_names(tree),
            "F15 (source): the finalizer references `subprocess`")

    # Behavioral spy — fails the gate if either is called during a real run.
    import subprocess as _subprocess
    tripped: List[str] = []

    def spy(label, original):
        def wrapper(*a, **kw):
            tripped.append(label)
            return original(*a, **kw)
        return wrapper

    saved = {
        "run": _subprocess.run, "Popen": _subprocess.Popen,
        "call": _subprocess.call, "check_output": _subprocess.check_output,
    }
    saved_os = {"system": os.system, "popen": os.popen}
    for name, original in saved.items():
        setattr(_subprocess, name, spy(f"subprocess.{name}", original))
    for name, original in saved_os.items():
        setattr(os, name, spy(f"os.{name}", original))
    try:
        root = fresh_root()
        publish(root, [cand(5, 1, "constant", 0.40)], RF=RF)
        with _Patch(RF, _write_npz=_boom):
            _expect_raises("F15", Exception,
                           lambda: publish(root, [cand(9, 1, "constant", 0.9)], RF=RF))
    finally:
        for name, original in saved.items():
            setattr(_subprocess, name, original)
        for name, original in saved_os.items():
            setattr(os, name, original)
    _assert(not tripped,
            f"F15 (behavioral): the finalizer spawned {tripped} — no fallback "
            f"writer may be invoked under any circumstance")


# ═════════════════════════════════════════════════════════════════════════════
# F16-F21 — coverage and domain
# ═════════════════════════════════════════════════════════════════════════════

def f16_seed_below_start(RF=PROD):
    root = fresh_root()
    exc = _expect_raises(
        "F16", RF.RunFinalizerError,
        lambda: publish(root, [cand(99, 1, "constant", 0.5)],
                        RF=RF, seed_start=100, seed_count=50))
    _assert(isinstance(exc, RF.CoverageValidationError),
            f"F16: expected a coverage rejection, got {type(exc).__name__}")
    _assert(read_current(root) is None, "F16: nothing may be published")


def f17_seed_at_or_above_end(RF=PROD):
    root = fresh_root()
    for seed in (150, 151):
        _expect_raises(
            f"F17(seed={seed})", RF.CoverageValidationError,
            lambda s=seed: publish(root, [cand(s, 1, "constant", 0.5)],
                                   RF=RF, seed_start=100, seed_count=50))
    # the last in-range seed is accepted
    publish(root, [cand(149, 1, "constant", 0.5)], RF=RF,
            seed_start=100, seed_count=50)


# The two anchors of the F18 mutant, shared by the gate's demonstration and by
# F26's permanent mutation set. Kept at module level so both use the SAME mutant
# and neither can drift from the other.
F18_PYTHON_ADDITION = ("    end_exclusive = start + count           "
                       "# Python ints — cannot wrap")
F18_UINT32_ADDITION = "    end_exclusive = int(np.uint32(start) + np.uint32(count))"
F18_ORDERING_CHECK = ("    if not (start < end_exclusive <= "
                      "SEED_DOMAIN_EXCLUSIVE_MAX):")
F18_ORDERING_REMOVED = "    if False:"

# The boundary case, written out as literals.
F18_SEED_START = 4294967286              # 2**32 - 10
F18_SEED_COUNT = 100
F18_PYTHON_END = 4294967386              # the TRUE end, in Python integers
F18_WRAPPED_END = 90                     # what fixed-width unsigned produces


def f18_overflow_rejected(RF=PROD):
    """The declared interval must be rejected by the PYTHON-INTEGER domain check.

    Boundary case: seed_start = 2**32 - 10, seed_count = 100. In Python integers
    the end is 4294967386, outside the frozen [0, 2**32) seed domain. In
    fixed-width unsigned arithmetic it WRAPS to 90, and the run's declared
    interval silently becomes [4294967286, 90) — a sweep that never happened.

    ATTRIBUTION IS THE POINT. Three things are excluded by construction so that a
    red here can mean nothing else:

      * the candidate list is EMPTY, so no candidate validation, no L2, no
        ordering and no identity wall can contribute;
      * the run metadata is otherwise well-formed and the tree is clean, so no
        parameter or sidecar rejection can contribute;
      * every publication step is left untouched, so no unrelated publication
        failure can contribute.

    The error message must name the UNWRAPPED end, 4294967386. That value can
    only exist if the addition was performed in unbounded Python integers — a
    fixed-width implementation cannot produce it, because it never computes it.
    So the assertion does not merely observe "something raised"; it observes the
    specific arithmetic domain the spec requires.
    """
    _assert(F18_SEED_START + F18_SEED_COUNT == F18_PYTHON_END,
            "F18 fixture: the Python-integer end must be 4294967386")
    _assert(int(np.uint32(F18_SEED_START) + np.uint32(F18_SEED_COUNT))
            == F18_WRAPPED_END,
            "F18 fixture: fixed-width unsigned addition must wrap to 90")

    root = fresh_root()
    _expect_raises(
        "F18", RF.CoverageValidationError,
        lambda: publish(root, [], RF=RF,
                        seed_start=F18_SEED_START, seed_count=F18_SEED_COUNT),
        must_mention=(str(F18_PYTHON_END),))
    _assert(read_current(root) is None,
            "F18: an interval that escapes the uint32 seed domain must publish "
            "nothing")

    if RF is PROD:
        _f18_demonstrate_false_coverage()


def _f18_demonstrate_false_coverage():
    """Show what the defective implementation would actually certify.

    The mutant is the strengthened form the spec binds: fixed-width unsigned
    arithmetic AND removal of the interval-ordering check that would otherwise
    reject the wrapped result incidentally. Removing only the arithmetic is not
    enough to demonstrate anything — the wrapped end is necessarily below
    `start`, so the ordering check rejects it for a reason that has nothing to do
    with the arithmetic domain. Both anchors together are what isolate the
    Python-integer requirement.

    With both gone, `finalize_run` CERTIFIES a generation whose sidecar claims
    the run swept [4294967286, 90). No seed can satisfy an inverted interval, so
    the artifact is empty — and an empty artifact carrying false coverage
    metadata is worse than a loud failure, because the next run inherits it as a
    validated parent and the claim is then chain-authenticated.
    """
    defective = build_mutant([
        (F18_PYTHON_ADDITION, F18_UINT32_ADDITION),
        (F18_ORDERING_CHECK, F18_ORDERING_REMOVED),
    ])
    root = fresh_root()
    leaked = publish(root, [], RF=defective,
                     seed_start=F18_SEED_START, seed_count=F18_SEED_COUNT)

    _assert(read_current(root) is not None,
            "F18: the defective implementation should have PUBLISHED — if it "
            "did not, this mutant no longer models the defect")
    _assert(leaked.seed_end_exclusive == F18_WRAPPED_END,
            f"F18: the defective implementation should certify the wrapped end "
            f"{F18_WRAPPED_END}, got {leaked.seed_end_exclusive}")
    sidecar = json.loads(leaked.sidecar_path.read_text())
    _assert(sidecar["seed_start"] == F18_SEED_START
            and sidecar["seed_end_exclusive"] == F18_WRAPPED_END
            and sidecar["seed_count"] == F18_SEED_COUNT,
            f"F18: the defective implementation should have written the false "
            f"interval [{F18_SEED_START}, {F18_WRAPPED_END}) into the sidecar, "
            f"got [{sidecar['seed_start']}, {sidecar['seed_end_exclusive']})")
    _assert(sidecar["seed_end_exclusive"] < sidecar["seed_start"],
            "F18: the certified interval should be INVERTED — that inversion is "
            "the false coverage claim")

    # And the claim is inheritable: a following run adopts it as a validated
    # parent, so the falsehood becomes chain-authenticated rather than isolated.
    child = publish(root, [], RF=defective,
                    seed_start=F18_SEED_START, seed_count=F18_SEED_COUNT)
    _assert(child.parent_generation_id == leaked.generation_id,
            "F18: the false-coverage generation should have been accepted as a "
            "certified parent by the next run")


def f19_declared_coverage_outside_domain(RF=PROD):
    root = fresh_root()
    for label, kwargs in (
            ("negative start", dict(seed_start=-1, seed_count=10)),
            ("start at 2**32", dict(seed_start=ORACLE_UINT32_EXCLUSIVE_MAX,
                                    seed_count=10)),
            ("zero count", dict(seed_start=0, seed_count=0)),
            ("negative count", dict(seed_start=0, seed_count=-5)),
            ("bool start", dict(seed_start=True, seed_count=10)),
            ("bool count", dict(seed_start=0, seed_count=True)),
            ("float count", dict(seed_start=0, seed_count=10.0))):
        _expect_raises(f"F19 ({label})", RF.CoverageValidationError,
                       lambda k=kwargs: publish(
                           root, [cand(5, 1, "constant", 0.5)], RF=RF, **k))


def f20_candidate_seed_outside_domain(RF=PROD):
    root = fresh_root()
    # A seed at/above 2**32 cannot also be inside a legal declared interval, so
    # the domain wall is what must reject it.
    _expect_raises(
        "F20", RF.RunFinalizerError,
        lambda: publish(root, [cand(ORACLE_UINT32_EXCLUSIVE_MAX, 1, "constant", 0.5)],
                        RF=RF, seed_start=0, seed_count=ORACLE_UINT32_EXCLUSIVE_MAX))
    _expect_raises(
        "F20 (negative)", RF.RunFinalizerError,
        lambda: publish(root, [cand(-1, 1, "constant", 0.5)], RF=RF))
    _expect_raises(
        "F20 (bool)", RF.RunFinalizerError,
        lambda: publish(root, [cand(True, 1, "constant", 0.5)], RF=RF))


def f21_valid_range_records_coverage(RF=PROD):
    root = fresh_root()
    result = publish(root, [cand(1200, 3, "constant", 0.5)], RF=RF,
                     seed_start=1000, seed_count=500)
    _assert((result.seed_start, result.seed_count, result.seed_end_exclusive)
            == (1000, 500, 1500),
            f"F21: RunArtifactResult coverage is "
            f"{(result.seed_start, result.seed_count, result.seed_end_exclusive)}")
    sidecar = json.loads(result.sidecar_path.read_text())
    _assert((sidecar["seed_start"], sidecar["seed_count"],
             sidecar["seed_end_exclusive"]) == (1000, 500, 1500),
            "F21: the sidecar must record the same coverage")


# ═════════════════════════════════════════════════════════════════════════════
# F22-F25 — contract
# ═════════════════════════════════════════════════════════════════════════════

def f22_published_bundle_contract(RF=PROD):
    root = fresh_root()
    result = publish(root, [cand(9, 1, "constant", 0.5),
                            cand(2, 1, "constant", 0.6),
                            cand(31, 2, "variable", 0.7)], RF=RF)
    for path in (result.binary_npz_path, result.all_npz_path):
        arrays, order = load_bundle(path)
        _assert(order == ORACLE_ARRAY_NAMES,
                f"F22 ({path.name}): stored key ORDER is {order}, expected "
                f"{ORACLE_ARRAY_NAMES}")
        for name, dtype in ORACLE_ARRAYS:
            _assert(arrays[name].dtype == np.dtype(dtype),
                    f"F22 ({path.name}): {name!r} dtype is "
                    f"{arrays[name].dtype}, expected {dtype}")
            _assert(arrays[name].ndim == 1,
                    f"F22 ({path.name}): {name!r} is not 1-D")
        seeds = [int(s) for s in arrays["seeds"]]
        _assert(seeds == sorted(seeds) and len(set(seeds)) == len(seeds),
                f"F22: the published bundle must be globally seed-ascending and "
                f"unique, got {seeds}")
        _assert(seeds == [2, 9, 31], f"F22: unexpected seed order {seeds}")
    # And it satisfies the production validator too.
    arrays, _ = load_bundle(result.binary_npz_path)
    PROD.validate_array_bundle(arrays)


def f23_clean_start(RF=PROD):
    root = fresh_root()
    result = publish(root, [cand(5, 1, "constant", 0.5),
                            cand(5, 2, "variable", 0.9),
                            cand(7, 1, "constant", 0.5)], RF=RF)
    _assert(result.parent_generation_id is None
            and result.parent_artifact_sha256 is None
            and result.parent_sidecar_sha256 is None,
            "F23: a clean start carries no parent references")
    sidecar = json.loads(result.sidecar_path.read_text())
    for key in ("parent_generation_id", "parent_artifact_sha256",
                "parent_sidecar_sha256"):
        _assert(sidecar[key] is None, f"F23: sidecar {key} must be null")
    _assert(result.l2_winner_count == 2, "F23: two seeds -> two L2 winners")
    _assert(result.final_row_count == 2 == sidecar["row_count"],
            "F23: row count equals the L2 winner count on a clean start")
    _assert(result.prior_row_count == 0, "F23: no prior rows")


def f24_sidecar_artifact_hash(RF=PROD):
    root = fresh_root()
    result = publish(root, [cand(5, 1, "constant", 0.5)], RF=RF)
    sidecar = json.loads(result.sidecar_path.read_text())
    actual = sha256_path(result.binary_npz_path)
    _assert(sidecar["artifact_sha256"] == actual,
            f"F24: sidecar artifact_sha256 {sidecar['artifact_sha256']} != the "
            f"published file's actual hash {actual}")
    _assert(result.artifact_sha256 == actual, "F24: result hash disagrees")
    _assert(sha256_path(result.all_npz_path) == actual,
            "F24: the two published names must carry byte-identical payloads")


def f25_miner_path_fields_remain_none(RF=PROD):
    from miner.range_miner_npz_writer import MinerTrialAssembly
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(MinerTrialAssembly)}
    for name in ("binary_npz_path", "all_npz_path"):
        _assert(name in fields, f"F25: MinerTrialAssembly has no {name!r}")
        _assert(fields[name].default is None,
                f"F25: {name!r} must default to None (deprecated, Ruling E)")
    # The finalizer must never construct or touch a MinerTrialAssembly. Asserted
    # over executable references, not raw text: the module docstring names the
    # class precisely to record the prohibition.
    tree = ast.parse(_PROD_SRC)
    _assert("MinerTrialAssembly" not in _referenced_names(tree),
            "F25: the finalizer references MinerTrialAssembly — its path fields "
            "remain deprecated and permanently None (Ruling E)")
    _assert(not any("miner" in (n.module or "").split(".")
                    for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)),
            "F25: the finalizer imports from the miner package")


# ═════════════════════════════════════════════════════════════════════════════
# F27-F37 — REV2 additions
# ═════════════════════════════════════════════════════════════════════════════

def f27_malformed_losing_candidate(RF=PROD):
    """seed X valid @0.9 + seed X missing `sessions` @0.4 -> the run FAILS."""
    root = fresh_root()
    loser = cand(5, 2, "variable", 0.40)
    del loser["sessions"]
    exc = _expect_raises(
        "F27", RF.RunFinalizerError,
        lambda: publish(root, [cand(5, 1, "constant", 0.90), loser], RF=RF),
        must_mention=("sessions",))
    _assert(isinstance(exc, RF.CandidateValidationError),
            f"F27: expected a candidate-validation rejection, got "
            f"{type(exc).__name__}")
    _assert(read_current(root) is None,
            "F27: no generation may be published when a LOSING candidate is "
            "malformed — it must not vanish during selection")


def f28_l3_is_array_domain(RF=PROD, source=None):
    """No 22 -> 24 reconstruction anywhere: source-level plus behavioral."""
    src = _PROD_SRC if source is None else source
    tree = ast.parse(src)
    merge = next((n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_l3_merge"), None)
    _assert(merge is not None, "F28: _l3_merge is missing")
    for node in ast.walk(merge):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            _assert(node.func.id not in ("records_to_arrays", "build_mode_records"),
                    f"F28 (source): _l3_merge calls {node.func.id} — L3 operates "
                    f"on ARRAYS; reconstructing records is impossible without "
                    f"inventing `sessions` and `prng_base`")
        if isinstance(node, ast.Constant) and node.value in (
                "sessions", "prng_base", "forward_match_rate"):
            _assert(False,
                    f"F28 (source): _l3_merge references the record-domain field "
                    f"{node.value!r}")

    # Behavioral: a retained prior row is byte-identical to its SOURCE array,
    # read independently from the prior file before the merge ran.
    root = fresh_root()
    first = publish(root, [cand(5, 1, "variable", 0.80),
                           cand(40, 2, "constant", 0.10)], RF=RF)
    prior_arrays, _ = load_bundle(first.binary_npz_path)
    result = publish(root, [cand(5, 9, "constant", 0.10),
                            cand(77, 1, "constant", 0.95)], RF=RF)
    arrays, _ = load_bundle(result.binary_npz_path)
    for seed in (5, 40):
        i_new = list(arrays["seeds"]).index(seed)
        i_old = list(prior_arrays["seeds"]).index(seed)
        for name in ORACLE_ARRAY_NAMES:
            _assert(arrays[name][i_new].tobytes()
                    == prior_arrays[name][i_old].tobytes(),
                    f"F28: retained prior row seed={seed} array {name!r} was not "
                    f"copied directly from its existing typed array")


def f29_single_commit_visibility(RF=PROD):
    """Artifact, sidecar and BOTH root aliases flip through ONE pointer swap."""
    root = fresh_root()
    all_alias = root / ORACLE_ALL_NPZ
    bin_alias = root / ORACLE_BINARY_NPZ
    observations: List[dict] = []

    original = RF._replace_symlink

    def spy(target, link_path, tmp_link_path):
        observations.append({
            "when": "before",
            "current": read_current(root),
            "all_resolves": all_alias.exists(),
            "bin_resolves": bin_alias.exists(),
            "all_is_link": os.path.islink(all_alias),
            "bin_is_link": os.path.islink(bin_alias),
        })
        original(target, link_path, tmp_link_path)
        observations.append({
            "when": "after",
            "current": read_current(root),
            "all_resolves": all_alias.exists(),
            "bin_resolves": bin_alias.exists(),
        })

    with _Patch(RF, _replace_symlink=spy):
        first = publish(root, [cand(5, 1, "constant", 0.5)], RF=RF)
    before, after = observations
    _assert(before["current"] is None, "F29: current existed before the commit")
    _assert(before["all_is_link"] and before["bin_is_link"],
            "F29: both root aliases must already be symlinks before the commit")
    _assert(not before["all_resolves"] and not before["bin_resolves"],
            "F29: before the commit both aliases must still be DANGLING — "
            "nothing is visible")
    _assert(after["all_resolves"] and after["bin_resolves"]
            and after["current"] is not None,
            "F29: one pointer swap must make artifact, sidecar and both aliases "
            "visible together")
    _assert(sha256_path(all_alias) == first.artifact_sha256
            and sha256_path(bin_alias) == first.artifact_sha256,
            "F29: the root aliases must resolve to the certified artifact")
    _assert((root / ORACLE_ACCUM_DIR / ORACLE_CURRENT / ORACLE_SIDECAR_NAME).is_file(),
            "F29: the sidecar becomes visible through the same pointer")

    # Generation 2: everything flips from the OLD generation to the NEW one at
    # the same instant.
    observations.clear()
    with _Patch(RF, _replace_symlink=spy):
        second = publish(root, [cand(9, 1, "constant", 0.5)], RF=RF)
    before, after = observations
    _assert(before["current"].endswith(first.generation_dir.name),
            "F29: before the second commit current still names generation 1")
    _assert(after["current"].endswith(second.generation_dir.name),
            "F29: after the swap current names generation 2")
    _assert(sha256_path(all_alias) == second.artifact_sha256,
            "F29: the aliases followed the single pointer swap")


def f30_fsync_before_pointer_swap(RF=PROD):
    calls: List[str] = []
    originals = {name: getattr(RF, name) for name in
                 ("_fsync_file", "_fsync_dir", "_write_and_fsync_bytes",
                  "_replace_symlink", "_atomic_rename")}

    def wrap(name):
        original = originals[name]

        def wrapper(*a, **kw):
            calls.append(f"{name}:{Path(a[0]).name if a else ''}")
            return original(*a, **kw)
        return wrapper

    root = fresh_root()
    with _Patch(RF, **{n: wrap(n) for n in originals}):
        publish(root, [cand(5, 1, "constant", 0.5)], RF=RF)

    swap = next(i for i, c in enumerate(calls) if c.startswith("_replace_symlink"))
    before = calls[:swap]

    npz_fsyncs = [c for c in before
                  if c.startswith("_fsync_file:") and c.endswith(".npz")]
    _assert(len(npz_fsyncs) >= 2,
            f"F30: both NPZ files must be fsynced before the pointer swap, saw "
            f"{npz_fsyncs} in {calls}")
    _assert(any(c.startswith("_write_and_fsync_bytes:") for c in before),
            f"F30: the sidecar must be written and fsynced before the swap: {calls}")
    dir_fsyncs = [c for c in before if c.startswith("_fsync_dir:")]
    _assert(len(dir_fsyncs) >= 2,
            f"F30: the staging directory AND generations/ must be fsynced before "
            f"the swap, saw {dir_fsyncs} in {calls}")
    rename = next((i for i, c in enumerate(calls)
                   if c.startswith("_atomic_rename")), None)
    _assert(rename is not None and rename < swap,
            f"F30: the generation rename must precede the pointer swap: {calls}")
    _assert(any(c.startswith("_fsync_dir:") for c in calls[rename:swap]),
            f"F30: generations/ must be fsynced between the rename and the swap: "
            f"{calls}")


def f31_failure_propagates_through_optimize_window(RF=PROD, source=None):
    """The finalizer must not sit inside the caller's swallow wrapper [B4]."""
    src = _INTEGRATION_SRC if source is None else source
    tree = ast.parse(src)

    parents: Dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node

    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "_finalize_run_d3_5"]
    _assert(len(calls) == 1,
            f"F31: expected exactly one finalizer call site, found {len(calls)}")

    node = calls[0]
    swallowing: List[int] = []
    walker = node
    while id(walker) in parents:
        walker = parents[id(walker)]
        if isinstance(walker, ast.Try) and walker.handlers:
            swallowing.append(walker.lineno)
    _assert(not swallowing,
            f"F31 (source): the finalizer call is nested inside try/except "
            f"block(s) at line(s) {swallowing}. A fail-closed finalizer inside "
            f"the broad swallow-and-fallback wrapper would print a warning and "
            f"still RETURN SUCCESS.")

    # `return results` must be reachable only AFTER the finalizer call.
    returns = [n for n in ast.walk(tree)
               if isinstance(n, ast.Return) and isinstance(n.value, ast.Name)
               and n.value.id == "results"]
    _assert(returns, "F31: optimize_window no longer returns `results`")
    _assert(all(r.lineno > node.lineno for r in returns),
            "F31: a `return results` precedes canonical finalization")

    # Behavioral: the extracted canonical region really does propagate.
    block = _extract_canonical_block(src)
    namespace = _canonical_block_namespace()
    root = namespace["_root_for_gate"]
    raised = []
    try:
        exec(compile(block, "<d3.5-canonical-block>", "exec"), namespace)  # noqa: S102
    except RuntimeError as exc:
        raised.append(exc)
    _assert(raised, "F31 (behavioral): the injected finalizer failure was "
                    "swallowed instead of propagating out of the block")
    _assert(not namespace.get("_reached_end"),
            "F31 (behavioral): execution continued past the failed finalization")
    _assert(read_current(root) is None, "F31: current must be unchanged")


_CANONICAL_BLOCK_START = "        _repo_commit_d3_5, _repo_clean_d3_5 = _repository_state()"
_CANONICAL_BLOCK_END = "        print(f\"{'='*80}\\n\")"


def _extract_canonical_block(src: str) -> str:
    lines = src.splitlines()
    try:
        start = lines.index(_CANONICAL_BLOCK_START)
    except ValueError:
        raise AssertionError(
            "F31: the canonical finalization region could not be located — its "
            "anchor line has changed, so this gate no longer runs what it "
            "claims to.")
    end = next(i for i in range(start, len(lines))
               if lines[i] == _CANONICAL_BLOCK_END)
    return textwrap.dedent("\n".join(lines[start:end + 1])) + "\n_reached_end = True\n"


def _canonical_block_namespace() -> dict:
    root = fresh_root()

    def exploding_finalize(*_a, **_kw):
        raise RuntimeError("injected canonical finalization failure")

    return {
        "_repository_state": lambda: ("b" * 40, True),
        "_finalize_run_d3_5": exploding_finalize,
        "_Path_d3_5": Path,
        "_BINARY_NPZ_NAME_d3_5": ORACLE_BINARY_NPZ,
        "_raw_candidates_d3_5": [cand(5, 1, "constant", 0.5)],
        "test_both_modes": True,
        "prng_base": "java_lcg",
        "seed_start": 0,
        "seed_count": 1000,
        "json": json,
        "_root_for_gate": root,
        "_reached_end": False,
    }


def f32_current_unchanged_after_every_injected_failure(RF=PROD):
    """Parameterized across the §7.2 seams for steps 1-11."""
    seams = ("_mkdir", "_write_npz", "_sha256_file", "_canonical_json_bytes",
             "_write_and_fsync_bytes", "_read_bytes", "_fsync_file",
             "_fsync_dir", "_atomic_rename")
    for seam in seams:
        root = fresh_root()
        publish(root, [cand(5, 1, "constant", 0.40)], RF=RF)
        before = _published_state(root)
        with _Patch(RF, **{seam: _boom}):
            _expect_raises(f"F32 ({seam})", Exception,
                           lambda: publish(root, [cand(9, 1, "constant", 0.9)],
                                           RF=RF))
        after = _published_state(root)
        _assert(before == after,
                f"F32: injecting a failure at {seam} changed the current "
                f"generation — {before} vs {after}")


def f33_parent_sidecar_hash_mismatch(RF=PROD):
    root = fresh_root()
    g1 = publish(root, [cand(5, 1, "constant", 0.40)], RF=RF)
    publish(root, [cand(9, 1, "constant", 0.50)], RF=RF)
    # Modify the ANCESTOR's provenance METADATA only — the payload is untouched,
    # so a payload-only chain would not notice.
    sidecar_path = g1.generation_dir / ORACLE_SIDECAR_NAME
    payload = json.loads(sidecar_path.read_text())
    payload["repository_commit"] = "c" * 40
    sidecar_path.write_bytes(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8"))
    before = read_current(root)
    exc = _expect_raises("F33", RF.RunFinalizerError,
                         lambda: publish(root, [cand(11, 1, "constant", 0.6)], RF=RF))
    _assert(isinstance(exc, RF.PriorGenerationError),
            f"F33: expected a prior-generation rejection, got {type(exc).__name__}")
    _assert(read_current(root) == before, "F33: current changed")


def _rewrite_prior_arrays(RF, root: Path, mutate):
    """Rewrite the CURRENT generation's payload in place and re-anchor its
    sidecar + hash-bound directory name, so exactly one property is broken."""
    pointer = read_current(root)
    gen = root / ORACLE_ACCUM_DIR / pointer
    arrays, order = load_bundle(gen / ORACLE_BINARY_NPZ)
    mutate(arrays)
    ordered = {name: arrays[name] for name in order}
    for name in (ORACLE_ALL_NPZ, ORACLE_BINARY_NPZ):
        target = gen / name
        target.unlink()
        with open(target, "wb") as handle:
            np.savez_compressed(handle, **ordered)
    payload = json.loads((gen / ORACLE_SIDECAR_NAME).read_text())
    payload["artifact_sha256"] = sha256_path(gen / ORACLE_BINARY_NPZ)
    payload["row_count"] = int(arrays["seeds"].shape[0])
    payload["final_row_count"] = payload["row_count"]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True).encode("utf-8")
    (gen / ORACLE_SIDECAR_NAME).write_bytes(raw)
    new_name = f"{payload['generation_id']}--{hashlib.sha256(raw).hexdigest()}"
    new_dir = gen.parent / new_name
    gen.rename(new_dir)
    pointer_path = root / ORACLE_ACCUM_DIR / ORACLE_CURRENT
    pointer_path.unlink()
    os.symlink(f"{ORACLE_GENERATIONS}/{new_name}", pointer_path)


def f34_prior_seeds_unsorted_or_duplicated(RF=PROD):
    for label, mutate in (
            ("unsorted", lambda a: [a.__setitem__(n, a[n][::-1])
                                    for n in ORACLE_ARRAY_NAMES]),
            ("duplicated", lambda a: [a.__setitem__(
                n, np.concatenate([a[n][:1], a[n]])) for n in ORACLE_ARRAY_NAMES])):
        root = fresh_root()
        publish(root, [cand(5, 1, "constant", 0.40),
                       cand(9, 1, "constant", 0.50)], RF=RF)
        _rewrite_prior_arrays(RF, root, mutate)
        exc = _expect_raises(f"F34 ({label})", RF.RunFinalizerError,
                             lambda: publish(root, [cand(11, 1, "constant", 0.6)],
                                             RF=RF))
        _assert(isinstance(exc, RF.PriorGenerationError),
                f"F34 ({label}): expected a prior-generation rejection, got "
                f"{type(exc).__name__}")


def f35_prior_invalid_ids(RF=PROD):
    for label, name, value in (("skip_mode", "skip_mode", 200),
                               ("prng_type", "prng_type", 250)):
        root = fresh_root()
        publish(root, [cand(5, 1, "constant", 0.40)], RF=RF)
        _rewrite_prior_arrays(
            RF, root,
            lambda a, n=name, v=value: a.__setitem__(
                n, np.full(a[n].shape, v, dtype=a[n].dtype)))
        exc = _expect_raises(f"F35 ({label})", RF.RunFinalizerError,
                             lambda: publish(root, [cand(9, 1, "constant", 0.6)],
                                             RF=RF))
        _assert(isinstance(exc, RF.PriorGenerationError),
                f"F35 ({label}): expected a prior-generation rejection, got "
                f"{type(exc).__name__}")


def f36_legacy_deduplicator_has_no_authority(RF=PROD, source=None):
    """Source-level AND behavioral retirement of the score-only deduplicator."""
    src = _INTEGRATION_SRC if source is None else source
    tree = ast.parse(src)
    for node in ast.walk(tree):
        _assert(not (isinstance(node, ast.FunctionDef)
                     and node.name == "deduplicate_survivors"),
                "F36 (source): `deduplicate_survivors` is still defined; the "
                "legacy score-only selector must not determine any canonical or "
                "full bidirectional survivor output")
        _assert(not (isinstance(node, ast.Name)
                     and node.id == "deduplicate_survivors"),
                "F36 (source): `deduplicate_survivors` is still referenced")
        _assert(not (isinstance(node, ast.Name) and node.id == "bidirectional_deduped"),
                "F36 (source): the deduplicated list still feeds an output")

    # The JSON is no longer an independently deduplicated survivor list.
    dumps = [n for n in ast.walk(tree)
             if isinstance(n, ast.Constant) and n.value == "bidirectional_survivors.json"]
    _assert(dumps, "F36: bidirectional_survivors.json is no longer written")
    _assert("canonical input for Steps 2-6" not in src,
            "F36 (source): the comment still describes the JSON as the canonical "
            "Steps 2-6 input")

    # Behavioral: on a batch where the legacy score-only lexsort and the L2 key
    # DISAGREE, the published winner is L2's.
    batch = [cand(5, 8, "constant", 0.80), cand(5, 3, "variable", 0.80)]
    legacy = _legacy_score_only_dedup(batch)
    _assert(legacy[0]["trial_number"] == 8,
            f"F36 fixture: the legacy selector must pick trial 8 here, it picked "
            f"{legacy[0]['trial_number']}")
    root = fresh_root()
    result = publish(root, batch, RF=RF)
    arrays, _ = load_bundle(result.binary_npz_path)
    _assert(int(arrays["trial_number"][0]) == 3,
            f"F36 (behavioral): the canonical NPZ follows the legacy score-only "
            f"selector (trial {int(arrays['trial_number'][0])}) instead of the "
            f"explicit L2 key (trial 3)")


def _legacy_score_only_dedup(survivor_list):
    """The retired selector, HAND-TRANSCRIBED from `70cd6f0:1684-1700`.

    `lexsort((-scores, seeds))` — seed ascending, score descending, with NO
    trial_number and NO skip_mode key. Kept here as an oracle so the gate can
    show the two disagree; it is never imported from production.
    """
    if not survivor_list:
        return []
    seeds = np.array([s["seed"] for s in survivor_list], dtype=np.int64)
    scores = np.array([s.get("score", 0.0) for s in survivor_list], dtype=np.float32)
    order = np.lexsort((-scores, seeds))
    sorted_seeds = seeds[order]
    keep = np.concatenate(([True], sorted_seeds[1:] != sorted_seeds[:-1]))
    return [survivor_list[i] for i in order[keep]]


def f37_dirty_tree_cannot_certify(RF=PROD):
    root = fresh_root()
    exc = _expect_raises(
        "F37", RF.RunFinalizerError,
        lambda: publish(root, [cand(5, 1, "constant", 0.5)], RF=RF,
                        repository_tree_clean=False))
    _assert(isinstance(exc, RF.RunParameterError),
            f"F37: expected a run-parameter rejection, got {type(exc).__name__}")
    _assert(read_current(root) is None,
            "F37: a dirty working tree must not produce a certified generation — "
            "the first certified baseline must not claim a commit SHA while "
            "running uncommitted source")


# ═════════════════════════════════════════════════════════════════════════════
# F38-F47 — REV3 additions
# ═════════════════════════════════════════════════════════════════════════════

def f38_first_generation_alias_bootstrap(RF=PROD):
    root = fresh_root()
    seen: List[Tuple[bool, bool, bool, bool]] = []
    original = RF._replace_symlink

    def spy(target, link_path, tmp_link_path):
        seen.append((
            os.path.islink(root / ORACLE_ALL_NPZ),
            os.path.islink(root / ORACLE_BINARY_NPZ),
            (root / ORACLE_ALL_NPZ).exists(),
            (root / ORACLE_BINARY_NPZ).exists(),
        ))
        return original(target, link_path, tmp_link_path)

    with _Patch(RF, _replace_symlink=spy):
        publish(root, [cand(5, 1, "constant", 0.5)], RF=RF)
    _assert(seen, "F38: the commit point was never reached")
    all_link, bin_link, all_res, bin_res = seen[0]
    _assert(all_link and bin_link,
            "F38: both root aliases must exist as symlinks BEFORE the commit")
    _assert(not all_res and not bin_res,
            "F38: before the commit both aliases must be DANGLING")
    for name in (ORACLE_ALL_NPZ, ORACLE_BINARY_NPZ):
        _assert((root / name).exists(),
                f"F38: {name} must be valid immediately after the commit")
        _assert(os.readlink(root / name)
                == f"{ORACLE_ACCUM_DIR}/{ORACLE_CURRENT}/{name}",
                f"F38: {name} must be the exact static compatibility symlink")


def f39_conflicting_root_alias(RF=PROD):
    for label, make in (
            ("regular file", lambda p: p.write_bytes(b"not a symlink")),
            ("wrong-target symlink", lambda p: os.symlink("/dev/null", p)),
            ("directory", lambda p: p.mkdir())):
        for name in (ORACLE_ALL_NPZ, ORACLE_BINARY_NPZ):
            root = fresh_root()
            make(root / name)
            exc = _expect_raises(
                f"F39 ({label}, {name})", RF.RunFinalizerError,
                lambda: publish(root, [cand(5, 1, "constant", 0.5)], RF=RF))
            _assert(isinstance(exc, RF.PublicationError),
                    f"F39: expected a publication rejection, got "
                    f"{type(exc).__name__}")
            _assert(read_current(root) is None, "F39: nothing may be published")
            if label == "regular file":
                _assert((root / name).read_bytes() == b"not a symlink",
                        "F39: an existing regular file must never be silently "
                        "replaced")


def f40_failure_after_bootstrap_before_commit(RF=PROD):
    root = fresh_root()
    with _Patch(RF, _write_npz=_boom):
        _expect_raises("F40", Exception,
                       lambda: publish(root, [cand(5, 1, "constant", 0.5)], RF=RF))
    for name in (ORACLE_ALL_NPZ, ORACLE_BINARY_NPZ):
        _assert(os.path.islink(root / name),
                f"F40: {name} should remain as a harmless alias")
        _assert(not (root / name).exists(),
                f"F40: {name} must still be DANGLING — nothing was accepted")
    _assert(read_current(root) is None, "F40: no generation may be accepted")
    generations = root / ORACLE_ACCUM_DIR / ORACLE_GENERATIONS
    _assert(not list(generations.iterdir()),
            "F40: no generation directory may survive the failure")


def f41_candidate_prng_base_mismatch(RF=PROD):
    root = fresh_root()
    exc = _expect_raises(
        "F41", RF.RunFinalizerError,
        lambda: publish(root, [cand(5, 1, "constant", 0.5, prng_base="xorshift32")],
                        RF=RF))
    _assert(isinstance(exc, RF.RunIdentityError),
            f"F41: expected a run-identity rejection, got {type(exc).__name__}")
    _assert(read_current(root) is None, "F41: nothing may be published")


def f42_candidate_mode_not_executed(RF=PROD):
    root = fresh_root()
    exc = _expect_raises(
        "F42", RF.RunFinalizerError,
        lambda: publish(root, [cand(5, 1, "variable", 0.5)], RF=RF,
                        skip_modes_executed=("constant",)))
    _assert(isinstance(exc, RF.RunIdentityError),
            f"F42: expected a run-identity rejection, got {type(exc).__name__}")

    # A mode that ran and produced ZERO rows is legitimate — the executed set
    # comes from configuration, never from candidate inference.
    root2 = fresh_root()
    result = publish(root2, [cand(5, 1, "constant", 0.5)], RF=RF,
                     skip_modes_executed=("constant", "variable"))
    sidecar = json.loads(result.sidecar_path.read_text())
    _assert(sidecar["skip_modes_executed"] == ["constant", "variable"],
            f"F42: an executed mode with zero survivors must still be recorded, "
            f"got {sidecar['skip_modes_executed']}")


def f43_prior_identity_inconsistent(RF=PROD):
    """skip_mode=constant carrying <base>_hybrid — both IDs individually valid."""
    root = fresh_root()
    publish(root, [cand(5, 1, "constant", 0.40)], RF=RF)
    _rewrite_prior_arrays(
        RF, root,
        lambda a: a.__setitem__("prng_type", np.full(
            a["prng_type"].shape, ORACLE_JAVA_LCG_HYBRID_ID, dtype=a["prng_type"].dtype)))
    exc = _expect_raises("F43", RF.RunFinalizerError,
                         lambda: publish(root, [cand(9, 1, "constant", 0.6)], RF=RF),
                         must_mention=("identity-inconsistent",))
    _assert(isinstance(exc, RF.PriorGenerationError),
            f"F43: expected a prior-generation rejection, got {type(exc).__name__}")


def f44_detached_prior(RF=PROD):
    root = fresh_root()
    g1 = publish(root, [cand(5, 1, "constant", 0.40)], RF=RF)
    g2 = publish(root, [cand(9, 1, "constant", 0.50)], RF=RF)
    # g1 is no longer current: merging against it would fork the lineage.
    exc = _expect_raises(
        "F44 (detached)", RF.RunFinalizerError,
        lambda: publish(root, [cand(11, 1, "constant", 0.6)], RF=RF,
                        prior_generation_dir=g1.generation_dir))
    _assert(isinstance(exc, RF.PriorGenerationError),
            f"F44: expected a prior-generation rejection, got {type(exc).__name__}")
    _assert(read_current(root).endswith(g2.generation_dir.name),
            "F44: current changed")
    # A prior supplied while `current` is absent is equally a fork.
    root2 = fresh_root()
    _expect_raises(
        "F44 (no current)", RF.PriorGenerationError,
        lambda: publish(root2, [cand(5, 1, "constant", 0.5)], RF=RF,
                        prior_generation_dir=g1.generation_dir))
    # And the matching prior IS accepted.
    publish(root, [cand(11, 1, "constant", 0.6)], RF=RF,
            prior_generation_dir=g2.generation_dir)


def _write_generation(directory: Path, payload: dict, arrays=None) -> str:
    """Hand-build a generation directory under the SAME rules production uses."""
    directory.mkdir(parents=True, exist_ok=True)
    arrays = arrays or {name: np.array([], dtype=dtype)
                        for name, dtype in ORACLE_ARRAYS}
    for name in (ORACLE_ALL_NPZ, ORACLE_BINARY_NPZ):
        with open(directory / name, "wb") as handle:
            np.savez_compressed(handle, **arrays)
    payload = dict(payload)
    payload["artifact_sha256"] = sha256_path(directory / ORACLE_BINARY_NPZ)
    payload["row_count"] = int(arrays["seeds"].shape[0])
    payload["final_row_count"] = payload["row_count"]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True).encode("utf-8")
    (directory / ORACLE_SIDECAR_NAME).write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def f45_recursive_chain_failures(RF=PROD):
    # (a) missing ancestor
    root = fresh_root()
    g1 = publish(root, [cand(5, 1, "constant", 0.40)], RF=RF)
    publish(root, [cand(9, 1, "constant", 0.50)], RF=RF)
    shutil.rmtree(g1.generation_dir)
    _expect_raises("F45 (missing ancestor)", RF.PriorGenerationError,
                   lambda: publish(root, [cand(11, 1, "constant", 0.6)], RF=RF))

    # (b) modified ancestor sidecar (metadata only)
    root = fresh_root()
    g1 = publish(root, [cand(5, 1, "constant", 0.40)], RF=RF)
    publish(root, [cand(9, 1, "constant", 0.50)], RF=RF)
    path = g1.generation_dir / ORACLE_SIDECAR_NAME
    payload = json.loads(path.read_text())
    payload["run_id"] = "tampered"
    path.write_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=True).encode("utf-8"))
    _expect_raises("F45 (modified ancestor)", RF.PriorGenerationError,
                   lambda: publish(root, [cand(11, 1, "constant", 0.6)], RF=RF))

    # (c) repeated generation ID.
    #
    # A TRUE hash cycle is not constructible: A's directory name embeds
    # sha256(A's sidecar), and B's `parent_sidecar_sha256` must equal it while
    # A's must equal sha256(B's sidecar) — a circular pre-image problem. The
    # `seen` guard is reached by the constructible case, a REPEATED generation
    # id along an otherwise hash-consistent chain, which is exercised here.
    root = fresh_root()
    seed_gen = publish(root, [cand(5, 1, "constant", 0.40)], RF=RF)
    generations = root / ORACLE_ACCUM_DIR / ORACLE_GENERATIONS
    template = json.loads((seed_gen.sidecar_path).read_text())

    rootgen = dict(template)
    rootgen["generation_id"] = "dup"
    rootgen["run_id"] = "root"
    rootgen.update(parent_generation_id=None, parent_artifact_sha256=None,
                   parent_sidecar_sha256=None)
    root_hash = _write_generation(generations / "dup--" + "0" * 0, rootgen) \
        if False else None
    tmp = generations / "_staging_root"
    root_hash = _write_generation(tmp, rootgen)
    tmp.rename(generations / f"dup--{root_hash}")
    root_artifact = json.loads(
        (generations / f"dup--{root_hash}" / ORACLE_SIDECAR_NAME).read_text()
    )["artifact_sha256"]

    child = dict(template)
    child["generation_id"] = "dup"
    child["run_id"] = "child"
    child.update(parent_generation_id="dup", parent_artifact_sha256=root_artifact,
                 parent_sidecar_sha256=root_hash)
    tmp = generations / "_staging_child"
    child_hash = _write_generation(tmp, child)
    tmp.rename(generations / f"dup--{child_hash}")

    pointer = root / ORACLE_ACCUM_DIR / ORACLE_CURRENT
    pointer.unlink()
    os.symlink(f"{ORACLE_GENERATIONS}/dup--{child_hash}", pointer)
    _expect_raises("F45 (repeated generation id)", RF.PriorGenerationError,
                   lambda: publish(root, [cand(9, 1, "constant", 0.6)], RF=RF),
                   must_mention=("repeats generation id",))

    # (d) a "clean-start root" carrying a non-null parent reference.
    root = fresh_root()
    seed_gen = publish(root, [cand(5, 1, "constant", 0.40)], RF=RF)
    generations = root / ORACLE_ACCUM_DIR / ORACLE_GENERATIONS
    payload = json.loads(seed_gen.sidecar_path.read_text())
    payload["parent_generation_id"] = "phantom"
    tmp = generations / "_staging_partial"
    bad_hash = _write_generation(tmp, payload)
    tmp.rename(generations / f"partial--{bad_hash}")
    payload2 = json.loads(
        (generations / f"partial--{bad_hash}" / ORACLE_SIDECAR_NAME).read_text())
    _assert(payload2["generation_id"] != "partial",
            "F45 fixture guard")           # the id/name disagreement is separate
    # Re-stamp so the ONLY defect is the partial parent reference.
    payload["generation_id"] = "partial"
    shutil.rmtree(generations / f"partial--{bad_hash}")
    tmp = generations / "_staging_partial2"
    bad_hash = _write_generation(tmp, payload)
    tmp.rename(generations / f"partial--{bad_hash}")
    pointer = root / ORACLE_ACCUM_DIR / ORACLE_CURRENT
    pointer.unlink()
    os.symlink(f"{ORACLE_GENERATIONS}/partial--{bad_hash}", pointer)
    _expect_raises("F45 (partial parent reference)", RF.PriorGenerationError,
                   lambda: publish(root, [cand(9, 1, "constant", 0.6)], RF=RF),
                   must_mention=("partially null",))


def f46_prior_numeric_domains(RF=PROD):
    cases = (
        ("NaN directional rate", "forward_matches", float("nan")),
        ("fractional count", "forward_count", 2.5),
        ("negative metric", "intersection_ratio", -0.1),
        ("score above 1", "score", 1.5),
    )
    for label, name, value in cases:
        root = fresh_root()
        publish(root, [cand(5, 1, "constant", 0.40)], RF=RF)
        _rewrite_prior_arrays(
            RF, root,
            lambda a, n=name, v=value: a.__setitem__(
                n, np.full(a[n].shape, v, dtype=a[n].dtype)))
        exc = _expect_raises(f"F46 ({label})", RF.RunFinalizerError,
                             lambda: publish(root, [cand(9, 1, "constant", 0.6)],
                                             RF=RF))
        _assert(isinstance(exc, RF.PriorGenerationError),
                f"F46 ({label}): expected a prior-generation rejection, got "
                f"{type(exc).__name__}")

    # bidirectional_selectivity ABOVE 1 stays legal — no generic ceiling.
    root = fresh_root()
    publish(root, [cand(5, 1, "constant", 0.40)], RF=RF)
    _rewrite_prior_arrays(
        RF, root,
        lambda a: a.__setitem__("bidirectional_selectivity", np.full(
            a["bidirectional_selectivity"].shape, 10.0,
            dtype=a["bidirectional_selectivity"].dtype)))
    publish(root, [cand(9, 1, "constant", 0.6)], RF=RF)


def f47_sidecar_hash_binding(RF=PROD):
    root = fresh_root()
    first = publish(root, [cand(5, 1, "constant", 0.40)], RF=RF)
    payload = json.loads(first.sidecar_path.read_text())
    _assert(set(payload) == ORACLE_SIDECAR_KEYS,
            f"F47: sidecar key set mismatch. missing "
            f"{sorted(ORACLE_SIDECAR_KEYS - set(payload))}, unexpected "
            f"{sorted(set(payload) - ORACLE_SIDECAR_KEYS)}")
    _assert("sidecar_sha256" not in payload,
            "F47: the sidecar payload must contain NO sidecar_sha256 key — a "
            "file cannot contain its own hash")
    stored = first.sidecar_path.read_bytes()
    _assert(first.sidecar_sha256 == hashlib.sha256(stored).hexdigest(),
            "F47: RunArtifactResult.sidecar_sha256 must equal SHA-256 over the "
            "FINAL STORED BYTES of provenance.json")
    _assert(first.generation_dir.name.endswith("--" + first.sidecar_sha256),
            "F47: the generation directory name must embed that same hash")

    # The child records it as parent_sidecar_sha256.
    second = publish(root, [cand(9, 1, "constant", 0.50)], RF=RF)
    child = json.loads(second.sidecar_path.read_text())
    _assert(child["parent_sidecar_sha256"] == first.sidecar_sha256,
            f"F47: the child must record the parent's stored-bytes hash, got "
            f"{child['parent_sidecar_sha256']}")
    _assert(child["parent_artifact_sha256"] == first.artifact_sha256
            and child["parent_generation_id"] == first.generation_id,
            "F47: the chain must cover data AND metadata")


# ═════════════════════════════════════════════════════════════════════════════
# F48-F51 — REV3.1 additions
# ═════════════════════════════════════════════════════════════════════════════

def f48_modified_current_sidecar(RF=PROD):
    root = fresh_root()
    first = publish(root, [cand(5, 1, "constant", 0.40)], RF=RF)
    path = first.sidecar_path
    payload = json.loads(path.read_text())
    payload["repository_commit"] = "f" * 40
    path.write_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=True).encode("utf-8"))
    before = read_current(root)
    exc = _expect_raises("F48", RF.RunFinalizerError,
                         lambda: publish(root, [cand(9, 1, "constant", 0.5)], RF=RF),
                         must_mention=("modified since publication",))
    _assert(isinstance(exc, RF.PriorGenerationError),
            f"F48: expected a prior-generation rejection, got {type(exc).__name__}")
    _assert(read_current(root) == before, "F48: current changed")


def f49_malformed_current_target(RF=PROD):
    accum_rel = f"{ORACLE_ACCUM_DIR}"

    def _fresh_with_generation():
        root = fresh_root()
        gen = publish(root, [cand(5, 1, "constant", 0.40)], RF=RF)
        return root, gen

    # (a) wrong embedded sidecar hash
    root, gen = _fresh_with_generation()
    pointer = root / ORACLE_ACCUM_DIR / ORACLE_CURRENT
    wrong = gen.generation_dir.parent / (
        gen.generation_id + "--" + "e" * 64)
    shutil.copytree(gen.generation_dir, wrong)
    pointer.unlink()
    os.symlink(f"{ORACLE_GENERATIONS}/{wrong.name}", pointer)
    _expect_raises("F49 (wrong embedded hash)", RF.PriorGenerationError,
                   lambda: publish(root, [cand(9, 1, "constant", 0.5)], RF=RF))

    # (b) a target escaping generations/
    root, gen = _fresh_with_generation()
    pointer = root / ORACLE_ACCUM_DIR / ORACLE_CURRENT
    pointer.unlink()
    os.symlink(f"../{ORACLE_GENERATIONS}/{gen.generation_dir.name}", pointer)
    _expect_raises("F49 (escaping target)", RF.PriorGenerationError,
                   lambda: publish(root, [cand(9, 1, "constant", 0.5)], RF=RF),
                   must_mention=("direct child",))

    # (c) a non-directory target
    root, gen = _fresh_with_generation()
    pointer = root / ORACLE_ACCUM_DIR / ORACLE_CURRENT
    flat = gen.generation_dir.parent / ("flat--" + "d" * 64)
    flat.write_bytes(b"not a directory")
    pointer.unlink()
    os.symlink(f"{ORACLE_GENERATIONS}/{flat.name}", pointer)
    _expect_raises("F49 (non-directory target)", RF.PriorGenerationError,
                   lambda: publish(root, [cand(9, 1, "constant", 0.5)], RF=RF),
                   must_mention=("not a real directory",))

    # (d) parsed generation_id disagreeing with sidecar.generation_id
    root, gen = _fresh_with_generation()
    pointer = root / ORACLE_ACCUM_DIR / ORACLE_CURRENT
    renamed = gen.generation_dir.parent / (
        "someone-else--" + gen.sidecar_sha256)
    gen.generation_dir.rename(renamed)
    pointer.unlink()
    os.symlink(f"{ORACLE_GENERATIONS}/{renamed.name}", pointer)
    _expect_raises("F49 (id disagreement)", RF.PriorGenerationError,
                   lambda: publish(root, [cand(9, 1, "constant", 0.5)], RF=RF),
                   must_mention=("but the sidecar claims",))

    # (e) `current` present but not a symlink at all
    root, gen = _fresh_with_generation()
    pointer = root / ORACLE_ACCUM_DIR / ORACLE_CURRENT
    pointer.unlink()
    pointer.write_bytes(b"regular file")
    _expect_raises("F49 (non-symlink pointer)", RF.PriorGenerationError,
                   lambda: publish(root, [cand(9, 1, "constant", 0.5)], RF=RF))
    _ = accum_rel


def f50_current_present_prior_omitted_merges(RF=PROD):
    """Omitting the optional argument must NOT silently start a new lineage."""
    root = fresh_root()
    first = publish(root, [cand(5, 1, "constant", 0.40),
                           cand(40, 1, "constant", 0.30)], RF=RF)
    second = publish(root, [cand(9, 1, "constant", 0.50)], RF=RF)   # prior omitted
    _assert(second.parent_generation_id == first.generation_id,
            f"F50: the child must automatically adopt current's target as its "
            f"parent, got {second.parent_generation_id}")
    _assert(second.prior_row_count == 2,
            f"F50: the prior's rows must be merged, got prior_row_count="
            f"{second.prior_row_count}")
    arrays, _ = load_bundle(second.binary_npz_path)
    _assert([int(s) for s in arrays["seeds"]] == [5, 9, 40],
            f"F50: omitting the prior silently started a new lineage — seeds "
            f"{[int(s) for s in arrays['seeds']]}")


def f51_post_swap_durability_failure(RF=PROD):
    root = fresh_root()
    first = publish(root, [cand(5, 1, "constant", 0.40)], RF=RF)

    original_fsync_dir = RF._fsync_dir
    original_swap = RF._replace_symlink
    state = {"swapped": False}

    def swap_spy(*a, **kw):
        result = original_swap(*a, **kw)
        state["swapped"] = True
        return result

    def fsync_dir_spy(path):
        if state["swapped"]:
            raise RuntimeError("injected post-swap durability failure")
        return original_fsync_dir(path)

    with _Patch(RF, _fsync_dir=fsync_dir_spy, _replace_symlink=swap_spy):
        exc = _expect_raises(
            "F51", RF.RunFinalizerError,
            lambda: publish(root, [cand(9, 1, "constant", 0.50)], RF=RF))
    _assert(isinstance(exc, RF.PublicationDurabilityError),
            f"F51: a post-swap failure needs a dedicated "
            f"PublicationDurabilityError, got {type(exc).__name__}")
    _assert(not isinstance(exc, RF.PublicationError),
            "F51: it must NOT be reportable as 'nothing published' — step 13 is "
            "the logical commit")
    _assert(read_current(root) is not None
            and not read_current(root).endswith(first.generation_dir.name),
            "F51: the pointer swap DID commit; the new generation is current")

    # The next invocation performs §7.1b recovery validation and may accept it.
    third = publish(root, [cand(11, 1, "constant", 0.60)], RF=RF)
    arrays, _ = load_bundle(third.binary_npz_path)
    _assert([int(s) for s in arrays["seeds"]] == [5, 9, 11],
            f"F51: the committed generation must be accepted on recovery once "
            f"directory, artifact, sidecar and hash-bound pointer all validate; "
            f"got {[int(s) for s in arrays['seeds']]}")

    # ...and only if they do. A corrupted committed tip is still refused.
    root2 = fresh_root()
    publish(root2, [cand(5, 1, "constant", 0.40)], RF=RF)
    state2 = {"swapped": False}

    def swap_spy2(*a, **kw):
        result = original_swap(*a, **kw)
        state2["swapped"] = True
        return result

    def fsync_dir_spy2(path):
        if state2["swapped"]:
            raise RuntimeError("injected post-swap durability failure")
        return original_fsync_dir(path)

    with _Patch(RF, _fsync_dir=fsync_dir_spy2, _replace_symlink=swap_spy2):
        _expect_raises("F51 (second)", RF.PublicationDurabilityError,
                       lambda: publish(root2, [cand(9, 1, "constant", 0.5)], RF=RF))
    tip = root2 / ORACLE_ACCUM_DIR / read_current(root2)
    payload = json.loads((tip / ORACLE_SIDECAR_NAME).read_text())
    payload["run_id"] = "tampered-after-commit"
    (tip / ORACLE_SIDECAR_NAME).write_bytes(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8"))
    _expect_raises("F51 (corrupt recovery)", RF.PriorGenerationError,
                   lambda: publish(root2, [cand(11, 1, "constant", 0.6)], RF=RF))


# ═════════════════════════════════════════════════════════════════════════════
# S1-S9 — D3.5-B Seed-Domain v1.1 (REV2 §6)
# ═════════════════════════════════════════════════════════════════════════════

def _canonical_bytes(payload: dict) -> bytes:
    """Serialize under the SAME rules production uses (§8b is never bypassed)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def _reanchor_tip(root: Path, mutate) -> dict:
    """Rewrite the CURRENT tip's sidecar through `mutate`, then re-anchor its
    hash-bound directory name and the `current` pointer.

    Every hash stays internally consistent, so the ONLY surviving defect is the
    mutated field — a rejection can never be attributed to a stale hash.
    """
    gen = root / ORACLE_ACCUM_DIR / read_current(root)
    payload = json.loads((gen / ORACLE_SIDECAR_NAME).read_text())
    mutate(payload)
    raw = _canonical_bytes(payload)
    (gen / ORACLE_SIDECAR_NAME).write_bytes(raw)
    new_name = f"{payload['generation_id']}--{hashlib.sha256(raw).hexdigest()}"
    gen.rename(gen.parent / new_name)
    pointer = root / ORACLE_ACCUM_DIR / ORACLE_CURRENT
    pointer.unlink()
    os.symlink(f"{ORACLE_GENERATIONS}/{new_name}", pointer)
    return payload


def _reanchor_parent(root: Path, mutate) -> None:
    """Two-generation chain: rewrite the PARENT's sidecar through `mutate`, then
    re-anchor the parent directory name, the child's parent_* references, the
    child's own directory name and the `current` pointer.

    Structural validity is preserved end to end — sidecar hashes, artifact
    hashes, generation ids and the pointer all agree — so the chain can fail on
    nothing except the mutated lineage field.
    """
    generations = root / ORACLE_ACCUM_DIR / ORACLE_GENERATIONS
    child_dir = root / ORACLE_ACCUM_DIR / read_current(root)
    child = json.loads((child_dir / ORACLE_SIDECAR_NAME).read_text())
    _assert(child["parent_generation_id"] is not None,
            "fixture: the tip must already have a parent")
    parent_dir = generations / (f"{child['parent_generation_id']}--"
                                f"{child['parent_sidecar_sha256']}")
    _assert(parent_dir.is_dir(), f"fixture: parent {parent_dir.name} is missing")

    parent = json.loads((parent_dir / ORACLE_SIDECAR_NAME).read_text())
    mutate(parent)
    raw = _canonical_bytes(parent)
    (parent_dir / ORACLE_SIDECAR_NAME).write_bytes(raw)
    parent_hash = hashlib.sha256(raw).hexdigest()
    new_parent = generations / f"{parent['generation_id']}--{parent_hash}"
    parent_dir.rename(new_parent)

    child["parent_generation_id"] = parent["generation_id"]
    child["parent_sidecar_sha256"] = parent_hash
    child["parent_artifact_sha256"] = sha256_path(new_parent / ORACLE_BINARY_NPZ)
    raw_child = _canonical_bytes(child)
    (child_dir / ORACLE_SIDECAR_NAME).write_bytes(raw_child)
    new_child = (f"{child['generation_id']}--"
                 f"{hashlib.sha256(raw_child).hexdigest()}")
    child_dir.rename(generations / new_child)
    pointer = root / ORACLE_ACCUM_DIR / ORACLE_CURRENT
    pointer.unlink()
    os.symlink(f"{ORACLE_GENERATIONS}/{new_child}", pointer)


def _two_generation_chain(RF, root: Path):
    first = publish(root, [cand(5, 1, "constant", 0.40)], RF=RF)
    second = publish(root, [cand(9, 1, "constant", 0.50)], RF=RF)
    return first, second


def s1_sidecar_key_set_is_exactly_32(RF=PROD):
    """The published sidecar's key set is EXACTLY the 32 hand-transcribed keys.

    Read off the STORED bytes, in their stored order, rather than from
    `SIDECAR_REQUIRED_KEYS` — asserting a constant against itself is the G9 /
    E8 / C1 defect. Production serializes with `sort_keys=True`, so the stored
    order IS the global alphabetical order the brief specifies.
    """
    root = fresh_root()
    result = publish(root, [cand(5, 1, "constant", 0.40)], RF=RF)
    stored = json.loads(result.sidecar_path.read_text(),
                        object_pairs_hook=lambda pairs: tuple(k for k, _ in pairs))
    _assert(stored == ORACLE_SIDECAR_KEY_ORDER,
            f"S1: the sidecar key tuple is not the 32 hand-transcribed keys in "
            f"global alphabetical order.\n  missing: "
            f"{[k for k in ORACLE_SIDECAR_KEY_ORDER if k not in stored]}\n"
            f"  unexpected: {[k for k in stored if k not in ORACLE_SIDECAR_KEYS]}"
            f"\n  order: {list(stored)}")
    _assert(len(stored) == 32, f"S1: expected 32 keys, got {len(stored)}")
    _assert("sidecar_sha256" not in stored,
            "S1: a file cannot contain its own hash [REV3 C1]")

    # An UNEXPECTED key fails closed too — the wall is exact, not a lower bound.
    root2 = fresh_root()
    publish(root2, [cand(5, 1, "constant", 0.40)], RF=RF)
    _reanchor_tip(root2, lambda p: p.__setitem__("seed_domain_note", "extra"))
    _expect_raises("S1 (extra key)", RF.PriorGenerationError,
                   lambda: publish(root2, [cand(9, 1, "constant", 0.5)], RF=RF))


def s2_published_generation_carries_the_nine_values(RF=PROD):
    """Every published v1.1 sidecar carries the nine fields at their §3 values,
    with the correct strict types, and none of them is caller-supplied."""
    for label, seed, seed_count in (("low stratum", 5, 1000),
                                    # A seed ABOVE 2**16 — an implementation
                                    # that derived the prefix from the candidate
                                    # maximum would report high16 = 1 here.
                                    ("seed above 2**16", 70000, 100000)):
        root = fresh_root()
        result = publish(root, [cand(seed, 1, "constant", 0.40)], RF=RF,
                         seed_count=seed_count)
        payload = json.loads(result.sidecar_path.read_text())
        for key, expected in ORACLE_SEED_DOMAIN:
            _assert(key in payload, f"S2 ({label}): sidecar lacks {key!r}")
            got = payload[key]
            _assert(got == expected and type(got) is type(expected),
                    f"S2 ({label}): sidecar {key!r} is {got!r} "
                    f"({type(got).__name__}), expected {expected!r} "
                    f"({type(expected).__name__})")
        _assert(payload["sidecar_schema_version"]
                == ORACLE_SIDECAR_SCHEMA_VERSION,
                f"S2 ({label}): sidecar_schema_version is "
                f"{payload['sidecar_schema_version']!r}, expected "
                f"{ORACLE_SIDECAR_SCHEMA_VERSION!r}")
        # The stratum domain is NOT the per-run coverage interval; both coexist.
        _assert(payload["seed_start"] == 0 and payload["seed_count"] == seed_count,
                f"S2 ({label}): the per-run coverage fields were overwritten by "
                f"the stratum-domain fields")

    # --- the nine are module-owned, never part of the public API ------------
    params = set(inspect.signature(RF.finalize_run).parameters)
    leaked = sorted(params & set(ORACLE_SEED_DOMAIN_NAMES))
    _assert(not leaked,
            f"S2: finalize_run exposes seed-domain field(s) {leaked} as caller "
            f"arguments. Every one is a module-owned constant; a caller-supplied "
            f"value is exactly how a run publishes a sidecar claiming a stratum "
            f"other than the one the uint32 domain wall enforced.")
    root = fresh_root()
    _expect_raises(
        "S2 (caller-supplied prefix)", TypeError,
        lambda: publish(root, [cand(5, 1, "constant", 0.40)], RF=RF,
                        seed_high16_prefix=1))


def s3_fail_closed_matrix(RF=PROD):
    """One case per seed-domain field, plus the bool-as-int hazard. Each is a
    structurally valid, correctly re-anchored generation whose ONLY defect is
    the named field."""
    cases = [
        ("missing seed_high16_prefix",
         lambda p: p.pop("seed_high16_prefix"), "seed_high16_prefix"),
        ("seed_semantics mislabelled",
         lambda p: p.__setitem__("seed_semantics", "external_input"),
         "seed_semantics"),
        ("seed_storage_dtype widened",
         lambda p: p.__setitem__("seed_storage_dtype", "uint64"),
         "seed_storage_dtype"),
        ("seed_effective_bits claims the full state",
         lambda p: p.__setitem__("seed_effective_bits", 48),
         "seed_effective_bits"),
        ("seed_high16_prefix claims another stratum",
         lambda p: p.__setitem__("seed_high16_prefix", 1),
         "seed_high16_prefix"),
        ("seed_domain_start shifted",
         lambda p: p.__setitem__("seed_domain_start", 1), "seed_domain_start"),
        ("seed_domain_end_exclusive claims 2**48",
         lambda p: p.__setitem__("seed_domain_end_exclusive", 2 ** 48),
         "seed_domain_end_exclusive"),
        ("exhaustive_over mislabelled",
         lambda p: p.__setitem__("exhaustive_over",
                                 "the full 48-bit internal state"),
         "exhaustive_over"),
        ("external_seed_transform non-null",
         lambda p: p.__setitem__("external_seed_transform", "identity"),
         "external_seed_transform"),
        # bool-as-int. `True` is caught by the value pin as well, but `False`
        # is NOT: False == 0 in Python, so only a bool-rejecting integer guard
        # can refuse it. Both are asserted.
        ("seed_high16_prefix is True (bool)",
         lambda p: p.__setitem__("seed_high16_prefix", True),
         "seed_high16_prefix"),
        ("seed_high16_prefix is False (bool == 0)",
         lambda p: p.__setitem__("seed_high16_prefix", False),
         "seed_high16_prefix"),
        ("seed_domain_start is False (bool == 0)",
         lambda p: p.__setitem__("seed_domain_start", False),
         "seed_domain_start"),
    ]
    for label, mutate, named in cases:
        root = fresh_root()
        publish(root, [cand(5, 1, "constant", 0.40)], RF=RF)
        _reanchor_tip(root, mutate)
        before = read_current(root)
        exc = _expect_raises(f"S3 ({label})", RF.RunFinalizerError,
                             lambda: publish(root, [cand(9, 1, "constant", 0.5)],
                                             RF=RF),
                             must_mention=(named,))
        _assert(isinstance(exc, RF.PriorGenerationError),
                f"S3 ({label}): expected a prior-generation rejection, got "
                f"{type(exc).__name__}")
        _assert(read_current(root) == before,
                f"S3 ({label}): current moved — the rejection was not "
                f"fail-closed")


def s4_pre_v1_1_sidecar_fails_closed(RF=PROD):
    """A genuine pre-v1.1 (23-key) sidecar is REJECTED, never silently read as
    `high16 = 0`. No compatibility reader is authorized (§4)."""
    root = fresh_root()
    publish(root, [cand(5, 1, "constant", 0.40)], RF=RF)

    def _downgrade(payload):
        for name in ORACLE_SEED_DOMAIN_NAMES:
            payload.pop(name, None)
        payload["sidecar_schema_version"] = \
            ORACLE_PRE_V1_1_SIDECAR_SCHEMA_VERSION

    old = _reanchor_tip(root, _downgrade)
    _assert(len(old) == 23,
            f"S4 fixture: the downgraded sidecar must carry the historical 23 "
            f"keys, it carries {len(old)}")
    _assert(not (set(old) & set(ORACLE_SEED_DOMAIN_NAMES)),
            "S4 fixture: the downgraded sidecar still carries a seed-domain key")
    before = read_current(root)
    exc = _expect_raises("S4", RF.RunFinalizerError,
                         lambda: publish(root, [cand(9, 1, "constant", 0.5)],
                                         RF=RF))
    _assert(isinstance(exc, RF.PriorGenerationError),
            f"S4: expected a prior-generation rejection, got "
            f"{type(exc).__name__}")
    _assert(read_current(root) == before, "S4: current moved")

    # The version string alone must also fail closed, with the nine fields
    # present and correct — a v1.1 payload wearing a pre-v1.1 label is still a
    # generation written under a different contract.
    root2 = fresh_root()
    publish(root2, [cand(5, 1, "constant", 0.40)], RF=RF)
    _reanchor_tip(root2, lambda p: p.__setitem__(
        "sidecar_schema_version", ORACLE_PRE_V1_1_SIDECAR_SCHEMA_VERSION))
    _expect_raises("S4 (old version label)", RF.PriorGenerationError,
                   lambda: publish(root2, [cand(9, 1, "constant", 0.5)], RF=RF))


def s5_valid_v1_1_publishes_and_chains(RF=PROD):
    """A valid v1.1 generation publishes, and a two-generation chain whose links
    agree on every stratum field validates recursively."""
    root = fresh_root()
    first, second = _two_generation_chain(RF, root)
    parent = json.loads(first.sidecar_path.read_text())
    child = json.loads(second.sidecar_path.read_text())

    _assert(child["parent_sidecar_sha256"] == first.sidecar_sha256
            and child["parent_generation_id"] == first.generation_id,
            "S5: the child does not reference its parent")
    for key in ORACLE_STRATUM_IDENTITY:
        _assert(child[key] == parent[key],
                f"S5: child and parent disagree on stratum field {key!r}: "
                f"{child[key]!r} vs {parent[key]!r}")
    for key, expected in ORACLE_SEED_DOMAIN:
        _assert(parent[key] == expected and child[key] == expected,
                f"S5: {key!r} is not the frozen v1.1 value in both links")

    # A third generation walks the whole chain to the clean-start root.
    third = publish(root, [cand(11, 1, "constant", 0.60)], RF=RF)
    _assert(third.parent_generation_id == second.generation_id,
            "S5: the third generation did not link to the second")
    _assert(parent["parent_generation_id"] is None,
            "S5: the first generation must be a clean-start root")
    arrays, _ = load_bundle(third.binary_npz_path)
    _assert(sorted(int(s) for s in arrays["seeds"]) == [5, 9, 11],
            f"S5: the merged bundle lost rows: {list(arrays['seeds'])}")


def s6_parent_disagreement_fails_closed(RF=PROD):
    """A child whose PARENT disagrees fails closed — parameterized across all
    nine seed-domain fields [R3] and across the five contract fields of §5's
    per-link contract, none of which was compared per link at 46a3828 [R1].

    The parent is re-anchored, so its sidecar hash, artifact hash, generation id
    and the child's references all remain valid: the only defect is the field.
    """
    disagreements = {
        "seed_semantics":            "external_input",
        "seed_storage_dtype":        "uint64",
        "seed_effective_bits":       48,
        "seed_high16_prefix":        1,
        "seed_domain_contract":      "v2.0-full-state",
        "seed_domain_start":         1,
        "seed_domain_end_exclusive": 2 ** 48,
        "exhaustive_over":           "the full 48-bit internal state",
        "external_seed_transform":   "identity",
        "prng_base":                 "lcg32",
        "artifact_schema_version":   "s172.d3.arrays.v0",
        "sidecar_schema_version":    ORACLE_PRE_V1_1_SIDECAR_SCHEMA_VERSION,
        "encoding_contract_version": "s172.phase0.encoding.v0",
        "canonical_map_hash":        "b" * 64,
    }
    _assert(set(disagreements) == set(ORACLE_LINEAGE_KEYS),
            f"S6 fixture: the case table must cover exactly the fourteen "
            f"per-link fields; it differs by "
            f"{sorted(set(disagreements) ^ set(ORACLE_LINEAGE_KEYS))}")

    for key in ORACLE_LINEAGE_KEYS:
        root = fresh_root()
        _two_generation_chain(RF, root)
        _reanchor_parent(root, lambda p, k=key: p.__setitem__(
            k, disagreements[k]))
        before = read_current(root)
        exc = _expect_raises(f"S6 ({key})", RF.RunFinalizerError,
                             lambda: publish(root, [cand(11, 1, "constant", 0.6)],
                                             RF=RF),
                             must_mention=(key,))
        _assert(isinstance(exc, RF.PriorGenerationError),
                f"S6 ({key}): expected a prior-generation rejection, got "
                f"{type(exc).__name__}")
        _assert(read_current(root) == before, f"S6 ({key}): current moved")


def s7_cross_domain_lineage_mismatch(RF=PROD):
    """[R4] A child claiming a DIFFERENT seed-domain contract / storage domain
    from its v1.1 parent fails specifically on seed-domain incompatibility —
    not on a malformed hash, key or JSON type.

    This establishes the invariant a future v2 must retain: a different
    seed-domain contract requires a NEW CLEAN ROOT. No v2 writer and no v2
    sidecar schema is introduced here.
    """
    def _contract_only(payload):
        # The MINIMAL cross-domain claim: the storage domain is unchanged, so
        # the rejection can only name the contract field itself.
        payload["seed_domain_contract"] = "v2.0-full-48bit-state"

    def _full_v2_claim(payload):
        payload["seed_domain_contract"] = "v2.0-full-48bit-state"
        payload["seed_storage_dtype"] = "uint64"
        payload["seed_effective_bits"] = 48
        payload["seed_domain_end_exclusive"] = 2 ** 48
        payload["exhaustive_over"] = "the full 48-bit internal state"

    for label, mutate, named in (
            ("contract only", _contract_only, ("seed_domain_contract",)),
            ("contract + storage domain", _full_v2_claim,
             ("seed-domain field",))):
        root = fresh_root()
        first, _second = _two_generation_chain(RF, root)
        child = _reanchor_tip(root, mutate)

        # Fixture guards: the chain is structurally intact, so nothing but the
        # seed-domain claim can be the cause.
        _assert(set(child) == ORACLE_SIDECAR_KEYS,
                f"S7 ({label}) fixture: the child's key set must stay exactly "
                f"the 32 keys")
        _assert(child["parent_sidecar_sha256"] == first.sidecar_sha256
                and child["parent_generation_id"] == first.generation_id,
                f"S7 ({label}) fixture: the parent references must stay valid")
        parent = json.loads(first.sidecar_path.read_text())
        _assert(parent["seed_domain_contract"] == "v1.1-stratum",
                f"S7 ({label}) fixture: the parent must remain v1.1")
        tip = root / ORACLE_ACCUM_DIR / read_current(root)
        _assert(tip.name.endswith(
            "--" + hashlib.sha256(
                (tip / ORACLE_SIDECAR_NAME).read_bytes()).hexdigest()),
            f"S7 ({label}) fixture: the tip directory must stay hash-bound to "
            f"its sidecar")

        before = read_current(root)
        exc = _expect_raises(f"S7 ({label})", RF.RunFinalizerError,
                             lambda: publish(root,
                                             [cand(11, 1, "constant", 0.6)],
                                             RF=RF),
                             must_mention=named)
        _assert(isinstance(exc, RF.PriorGenerationError),
                f"S7 ({label}): expected a prior-generation rejection, got "
                f"{type(exc).__name__}")
        text = str(exc)
        named_fields = [n for n in ORACLE_SEED_DOMAIN_NAMES if n in text]
        _assert(named_fields,
                f"S7 ({label}): the rejection names no seed-domain field, so "
                f"it is not attributable to seed-domain lineage "
                f"incompatibility: {text}")
        for wrong in ("hashes to", "is not valid", "must be a JSON object",
                      "missing required key", "unexpected key"):
            _assert(wrong not in text,
                    f"S7 ({label}): the failure is attributed to {wrong!r}, "
                    f"not to the seed-domain contract: {text}")
        _assert(read_current(root) == before, f"S7 ({label}): current moved")


def s8_other_versions_and_array_contract_unchanged(RF=PROD):
    """ARTIFACT_SCHEMA_VERSION and ENCODING_CONTRACT_VERSION are unchanged from
    46a3828, and the 22-array contract — names, order, dtypes — is untouched."""
    root = fresh_root()
    result = publish(root, [cand(5, 1, "constant", 0.40),
                            cand(9, 2, "variable", 0.50)], RF=RF)
    payload = json.loads(result.sidecar_path.read_text())
    _assert(payload["artifact_schema_version"] == ORACLE_ARTIFACT_SCHEMA_VERSION,
            f"S8: artifact_schema_version moved to "
            f"{payload['artifact_schema_version']!r}; D3.5-B changes no array, "
            f"no key order and no dtype, so it must stay "
            f"{ORACLE_ARTIFACT_SCHEMA_VERSION!r}")
    _assert(payload["encoding_contract_version"]
            == ORACLE_ENCODING_CONTRACT_VERSION,
            f"S8: encoding_contract_version moved to "
            f"{payload['encoding_contract_version']!r}; the encoding maps are "
            f"unchanged, so it must stay {ORACLE_ENCODING_CONTRACT_VERSION!r}")
    _assert(RF.ARTIFACT_SCHEMA_VERSION == ORACLE_ARTIFACT_SCHEMA_VERSION
            and RF.ENCODING_CONTRACT_VERSION == ORACLE_ENCODING_CONTRACT_VERSION,
            "S8: a module constant disagrees with the 46a3828 literal")

    for path in (result.all_npz_path, result.binary_npz_path):
        arrays, order = load_bundle(path)
        _assert(order == ORACLE_ARRAY_NAMES,
                f"S8: {path.name} array order changed: {order}")
        for name, dtype in ORACLE_ARRAYS:
            _assert(arrays[name].dtype == np.dtype(dtype),
                    f"S8: {path.name} {name!r} dtype is {arrays[name].dtype}, "
                    f"expected {dtype}")
            _assert(arrays[name].ndim == 1,
                    f"S8: {path.name} {name!r} is not 1-D")
        _assert(arrays["seeds"].shape[0] == 2,
                f"S8: {path.name} row count changed")


# ═════════════════════════════════════════════════════════════════════════════
# F26 — mutation proof
# ═════════════════════════════════════════════════════════════════════════════

class _Mod:
    """A namespace wrapper so gate functions can run against a mutant module."""

    def __init__(self, ns):
        self._ns = ns

    def __getattr__(self, name):
        try:
            return self._ns[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name == "_ns":
            object.__setattr__(self, name, value)
        else:
            self._ns[name] = value


def build_mutant(replacements: List[Tuple[str, str]], source=None) -> _Mod:
    """Apply textual replacements to the LIVE source and exec the result.

    Every anchor must occur exactly once; a vanished anchor means the mutant no
    longer models what it claims to, which is itself a red.
    """
    src = _PROD_SRC if source is None else source
    for old, new in replacements:
        count = src.count(old)
        _assert(count == 1,
                f"mutation anchor occurs {count} times (expected 1): {old!r}")
        src = src.replace(old, new)
    # `@dataclass` resolves forward references through `sys.modules[__module__]`,
    # so the mutant namespace has to be registered under a real module object or
    # RunArtifactResult cannot be built at all.
    _MUTANT_COUNTER[0] += 1
    module_name = f"_d3_5_mutant_{_MUTANT_COUNTER[0]}"
    import types
    module = types.ModuleType(module_name)
    module.__file__ = _MODULE_PATH
    sys.modules[module_name] = module
    ns = module.__dict__
    try:
        exec(compile(src, f"{_MODULE_PATH}:MUTANT", "exec"), ns)  # noqa: S102
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return _Mod(ns)


_MUTANT_COUNTER = [0]


def mutate_integration(replacements: List[Tuple[str, str]]) -> str:
    src = _INTEGRATION_SRC
    for old, new in replacements:
        count = src.count(old)
        _assert(count == 1,
                f"integration mutation anchor occurs {count} times: {old!r}")
        src = src.replace(old, new)
    return src


_MUTATION_REPORT: List[str] = []


# --- the mutants ------------------------------------------------------------
_M_L2_CALL = "    winners = _select_l2_winners(raw_candidates)"
_M_STRICT_GT = "        new_replaces = matched & (new_scores > prior_scores[pos_clipped])"
_M_L2_KEY = "    return (score32, -trial, mode_rank)"
_M_F32 = '    score32 = float(np.float32(record["score"]))'
_M_MERGED_INIT = "    merged: Dict[str, np.ndarray] = {}"
_M_PUBLISH_EXCEPT = "    except BaseException:\n        if not committed and os.path.isdir(tmp_dir):"
_M_FINAL_VALIDATE = "        validate_array_bundle(final_arrays)"
_M_STAGED_VALIDATE = "                validate_array_bundle(stored)"
_M_SORT_RETURN = "    ordered = {name: arrays[name][order] for name in _ARRAY_NAMES}"
_M_HASH_STEP = "        artifact_sha256 = _sha256_file(tmp_dir / CANONICAL_NPZ_NAME)"
_M_SIDECAR_MISSING = "    if not sidecar_path.is_file():"
_M_COVERAGE_CALL = "    _validate_candidate_coverage(raw_candidates, start, end_exclusive)"
_M_L3_CALL = "    merged = _l3_merge(winner_arrays,"
_M_RAW_WALL = "    _validate_raw_candidates(raw_candidates)"
_M_SWAP = """        _replace_symlink(
            f"{GENERATIONS_DIRNAME}/{final_dir.name}",
            accumulator_root / CURRENT_POINTER_NAME,
            accumulator_root / f".{CURRENT_POINTER_NAME}.tmp",
        )"""
_M_GEN_FSYNC = "        # 11. fsync generations/\n        _fsync_dir(generations_dir)"
_M_COMMITTED = "        committed = True"
_M_PARENT_SIDECAR = '        "parent_sidecar_sha256": None if prior is None else prior.sidecar_sha256,'
_M_CREATED_AT = '        "created_at": created_at,\n    }'
_M_IDENTITY_CALL = "    _validate_candidate_identity(raw_candidates, prng_base, modes)"
_M_PRIOR_MATCH = "    if not same:"
_M_CHAIN_CALL = "    _validate_chain(generations_dir, sidecar)"
_M_BOOTSTRAP_CALL = "    _bootstrap_root_aliases(output_root)"
_M_TIP_HASH = "    if actual_hash != expected_sidecar_hash:"
_M_FINAL_NAME = '        final_dir = generations_dir / f"{generation_id}--{sidecar_sha256}"'
_M_REOPEN = "        stored_sidecar_bytes = _read_bytes(sidecar_path)"
_M_WRITE_FSYNC = "        handle.write(data)"
_M_PRIOR_AUTO = "    if prior_generation_dir is None:\n        return target"

MUTANTS: List[Tuple[str, List[Tuple[str, str]], List[str]]] = [
    ("generic max-sort over prior + raw candidates (no L2 key)",
     [(_M_L2_CALL,
       "    winners = list({int(r['seed']): r for r in "
       "sorted(raw_candidates, key=lambda r: float(np.float32(r['score'])))}"
       ".values())")],
     ["F2", "F3", "F4", "F7", "F8"]),

    ("equal-score L3 replaces instead of retaining",
     [(_M_STRICT_GT, _M_STRICT_GT.replace(">", ">="))],
     ["F6"]),

    ("L2 orders mode before trial_number",
     [(_M_L2_KEY, "    return (score32, mode_rank, -trial)")],
     ["F2"]),

    ("Python-float instead of float32 comparison domain",
     [(_M_F32, '    score32 = float(record["score"])')],
     ["F4"]),

    ("in-place mutation of the prior bundle",
     [(_M_MERGED_INIT,
       '    prior_arrays["score"][:] = 0.0\n' + _M_MERGED_INIT)],
     ["F6", "F28"]),

    ("restored fallback-writer call on publication failure",
     [(_M_PUBLISH_EXCEPT,
       "    except BaseException:\n"
       "        import subprocess\n"
       "        subprocess.run(['python3', 'convert_survivors_to_binary.py'],\n"
       "                       check=False)\n"
       "        if not committed and os.path.isdir(tmp_dir):")],
     ["F15"]),

    ("validate_array_bundle skipped before publication",
     [(_M_FINAL_VALIDATE, "        final_arrays = final_arrays"),
      (_M_STAGED_VALIDATE, "                stored = stored"),
      (_M_SORT_RETURN,
       '    ordered = {name: (arrays[name][order].astype("int64")\n'
       '                      if name == "window_size" else arrays[name][order])\n'
       "               for name in _ARRAY_NAMES}")],
     ["F22"]),

    ("sidecar serialized before the artifact hash is known",
     [(_M_HASH_STEP, '        artifact_sha256 = "0" * 64')],
     ["F24"]),

    ("prior accepted without a sidecar",
     [(_M_SIDECAR_MISSING, "    if False:")],
     ["F13"]),

    ("local coverage check dropped",
     [(_M_COVERAGE_CALL, "    pass  # mutant: coverage check dropped")],
     ["F16", "F17"]),

    ("L3 before columnization of L2 winners",
     [(_M_L3_CALL, "    merged = _l3_merge(records_to_arrays(raw_candidates),")],
     ["F1", "F8"]),

    ("only L2 winners validated, not every raw candidate",
     [(_M_RAW_WALL, "    pass  # mutant: raw wall removed")],
     ["F27"]),

    ("two independent root-file renames instead of one pointer commit",
     [(_M_SWAP,
       "        _root_m = accumulator_root.parent\n"
       "        for _n_m in (ALL_NPZ_NAME, BINARY_NPZ_NAME):\n"
       "            _p_m = _root_m / _n_m\n"
       "            if os.path.lexists(_p_m):\n"
       "                os.unlink(_p_m)\n"
       "            os.replace(final_dir / _n_m, _p_m)")],
     ["F29"]),

    ("pointer swap before the generations/ directory fsync",
     [(_M_GEN_FSYNC, "        pass  # mutant: fsync moved after the swap"),
      (_M_COMMITTED, "        committed = True\n        _fsync_dir(generations_dir)")],
     ["F30"]),

    ("parent sidecar hash omitted from the chain",
     [(_M_PARENT_SIDECAR, '        "parent_sidecar_sha256": None,')],
     ["F47"]),

    ("sidecar_sha256 written into the sidecar payload",
     [(_M_CREATED_AT,
       '        "created_at": created_at,\n        "sidecar_sha256": "0" * 64,\n    }')],
     ["F47"]),

    ("candidate identity unchecked against run identity",
     [(_M_IDENTITY_CALL, "    pass  # mutant: identity wall removed")],
     ["F41", "F42"]),

    ("prior accepted without resolving the live current pointer",
     [(_M_PRIOR_MATCH, "    if False:")],
     ["F44"]),

    ("parent chain recorded but never recursively validated",
     [(_M_CHAIN_CALL, "    pass  # mutant: chain validation removed")],
     ["F12", "F33", "F45"]),

    ("root aliases created AFTER the current commit",
     [(_M_BOOTSTRAP_CALL, "    pass  # mutant: bootstrap deferred"),
      ("    # 15. only NOW is a result object constructed.",
       "    _bootstrap_root_aliases(output_root)\n"
       "    # 15. only NOW is a result object constructed.")],
     ["F29", "F38"]),

    ("ancestors validated but the current-tip hash check omitted",
     [(_M_TIP_HASH, "    if False:")],
     ["F48"]),

    ("generation directory named without the sidecar hash",
     [(_M_FINAL_NAME,
       '        final_dir = generations_dir / f"{generation_id}"')],
     ["F50"]),

    ("sidecar hashed from the in-memory buffer, not the reopened stored bytes",
     [(_M_REOPEN, "        stored_sidecar_bytes = sidecar_bytes"),
      (_M_WRITE_FSYNC, '        handle.write(data + b"\\n")')],
     ["F47"]),

    ("prior silently omitted when current exists (clean start instead of merge)",
     [(_M_PRIOR_AUTO, "    if prior_generation_dir is None:\n        return None")],
     ["F50"]),

    # The F18 strengthened form, permanently registered. BOTH anchors are
    # required: fixed-width unsigned arithmetic alone is rejected incidentally by
    # the interval-ordering check, so only removing that check too isolates the
    # Python-integer domain requirement. The kill must be attributable to the
    # missing domain check — F18 publishes ZERO candidates and leaves every
    # publication step intact, so nothing else can produce the red.
    ("seed_end_exclusive in fixed-width unsigned + ordering check removed "
     "(certifies false coverage [4294967286, 90))",
     [(F18_PYTHON_ADDITION, F18_UINT32_ADDITION),
      (F18_ORDERING_CHECK, F18_ORDERING_REMOVED)],
     ["F18"]),
]

# The two integration-file mutants (the defect lives in the caller, not the
# finalizer), each carrying the gate that must kill it.
INTEGRATION_MUTANTS: List[Tuple[str, List[Tuple[str, str]], str]] = [
    ("swallowed integration exception",
     [("        _artifact_d3_5 = _finalize_run_d3_5(\n            _raw_candidates_d3_5,",
       "        try:\n            _artifact_d3_5 = _finalize_run_d3_5(\n"
       "                _raw_candidates_d3_5,"),
      ("            repository_tree_clean=_repo_clean_d3_5,\n        )",
       "            repository_tree_clean=_repo_clean_d3_5,\n"
       "            )\n"
       "        except Exception as _swallowed:\n"
       "            print(f'warning: {_swallowed}')\n"
       "            _artifact_d3_5 = None")],
     "F31"),
    ("score-only legacy dedup left active",
     [("        _raw_candidates_d3_5 = survivor_accumulator['bidirectional']",
       "        def deduplicate_survivors(survivor_list):\n"
       "            return survivor_list\n"
       "        _raw_candidates_d3_5 = deduplicate_survivors(\n"
       "            survivor_accumulator['bidirectional'])")],
     "F36"),
]

# --- S9: the D3.5-B mutation set (REV2 §6) ----------------------------------
_M_SEED_PREFIX_PAYLOAD = '        "seed_high16_prefix": SEED_HIGH16_PREFIX,'
_M_MISSING_KEYS = "    missing = _SIDECAR_REQUIRED_KEY_SET - keys"
_M_READ_SIDECAR_VALIDATE = \
    "    _validate_sidecar_payload(payload, generation_dir.name)"
_M_LINEAGE_LOOP = "        for key in _LINEAGE_INVARIANT_KEYS:"
_M_SEED_VALUE_PIN = "        if payload[key] != expected:"
_M_ARTIFACT_VERSION = 'ARTIFACT_SCHEMA_VERSION = "s172.d3.arrays.v1"'
_M_SEED_INT_LOOP = (
    "    for key in _SEED_DOMAIN_INT_FIELDS:\n"
    "        _require_int(payload[key], f\"{label}: sidecar {key!r}\",\n"
    "                     PriorGenerationError)")
_M_FINALIZE_SIG = ("    prior_generation_dir: Optional[Path] = None,\n"
                   ") -> RunArtifactResult:")

S9_MUTANTS: List[Tuple[str, List[Tuple[str, str]], List[str]]] = [
    ("a seed-domain value INFERRED from the candidate maximum instead of the "
     "frozen module constant",
     [(_M_SEED_PREFIX_PAYLOAD,
       '        "seed_high16_prefix": (max((int(r["seed"]) '
       'for r in raw_candidates), default=0) >> 16),')],
     ["S2"]),

    ("seed_high16_prefix silently defaults to 0 when absent instead of "
     "failing closed",
     [(_M_MISSING_KEYS,
       "    payload = dict(payload)\n"
       '    payload.setdefault("seed_high16_prefix", 0)\n'
       "    keys = set(payload)\n"
       "    missing = _SIDECAR_REQUIRED_KEY_SET - keys")],
     ["S3"]),

    ("a compatibility reader accepts a pre-v1.1 23-key sidecar and interprets "
     "it as high16 = 0",
     [(_M_READ_SIDECAR_VALIDATE,
       '    if payload.get("sidecar_schema_version") == '
       '"s172.d3_5.provenance.v1":\n'
       "        payload = {**{_n: _v for _n, _v in SEED_DOMAIN_FIELDS}, "
       "**payload}\n"
       '        payload["sidecar_schema_version"] = SIDECAR_SCHEMA_VERSION\n'
       "    _validate_sidecar_payload(payload, generation_dir.name)")],
     ["S4"]),

    # BOTH anchors are required, for the same reason F18 needs two: the
    # seed-domain fields are guarded by TWO deliberately redundant layers — the
    # per-link lineage comparison and the payload value pin — so removing only
    # one leaves the other catching the parent. Removing both isolates the
    # seed-domain lineage contract itself, and S6 attributes the red to the
    # seed_high16_prefix link (every other S6 case still reds correctly, so
    # nothing else can produce this kill).
    ("the per-link lineage check omits the nine seed-domain fields (payload "
     "value pin for seed_high16_prefix removed too)",
     [(_M_LINEAGE_LOOP, "        for key in _LINEAGE_INVARIANT_KEYS[:5]:"),
      (_M_SEED_VALUE_PIN,
       '        if key != "seed_high16_prefix" and payload[key] != expected:')],
     ["S6"]),

    # Single anchor, uniquely attributable: an ANCESTOR's prng_base is compared
    # NOWHERE else. `_load_prior_generation` compares only the selected tip.
    ("the per-link lineage check omits prng_base",
     [(_M_LINEAGE_LOOP, "        for key in _LINEAGE_INVARIANT_KEYS[1:]:")],
     ["S6"]),

    ("ARTIFACT_SCHEMA_VERSION bumped although no array, key order or dtype "
     "changed",
     [(_M_ARTIFACT_VERSION, 'ARTIFACT_SCHEMA_VERSION = "s172.d3.arrays.v2"')],
     ["S8"]),

    # `False == 0`, so the value pin cannot see it; only the bool-rejecting
    # integer guard can.
    ("bool accepted as an integer seed-domain field (bare isinstance int)",
     [(_M_SEED_INT_LOOP,
       "    for key in _SEED_DOMAIN_INT_FIELDS:\n"
       "        if not isinstance(payload[key], int):\n"
       "            raise PriorGenerationError(\n"
       '                f"{label}: sidecar {key!r} must be an int, got "\n'
       '                f"{payload[key]!r}.")')],
     ["S3"]),

    ("a seed-domain value promoted to a finalize_run caller argument",
     [(_M_FINALIZE_SIG,
       "    prior_generation_dir: Optional[Path] = None,\n"
       "    seed_high16_prefix: int = SEED_HIGH16_PREFIX,\n"
       ") -> RunArtifactResult:"),
      (_M_SEED_PREFIX_PAYLOAD,
       '        "seed_high16_prefix": seed_high16_prefix,')],
     ["S2"]),
]

_GATE_BY_NAME = {}


def _run_mutation_set(label_prefix, mutants):
    """Shared mutation driver: build, run the nominated gates, report the red."""
    survivors: List[str] = []
    for label, replacements, gate_names in mutants:
        try:
            mutant = build_mutant(replacements)
        except AssertionError as exc:
            _MUTATION_REPORT.append(
                f"  RED  [{label_prefix}] {label}\n         anchor: {exc}")
            continue
        killed_by, signature = None, ""
        for gate_name in gate_names:
            try:
                _GATE_BY_NAME[gate_name](mutant)
            except Exception as exc:                        # noqa: BLE001
                killed_by = gate_name
                signature = (f"{type(exc).__name__}: "
                             f"{str(exc).splitlines()[0][:150]}")
                break
        if killed_by is None:
            survivors.append(f"{label} (expected kill by {gate_names})")
            _MUTATION_REPORT.append(f"  SURVIVED  [{label_prefix}] {label}")
        else:
            _MUTATION_REPORT.append(
                f"  RED  [{label_prefix}] {label}\n         killed by "
                f"{killed_by}: {signature}")
    return survivors


def s9_mutation_proof(RF=PROD):
    """The D3.5-B mutation set. Each mutant is a TEXTUAL edit to the live
    `utils/run_finalizer.py` exec'd into a fresh namespace; the file on disk is
    never modified."""
    _assert(RF is PROD, "S9 runs only against the production module")
    survivors = _run_mutation_set("S9", S9_MUTANTS)
    _assert(not survivors,
            "S9: mutants survived their gates:\n  " + "\n  ".join(survivors))


def f26_mutation_proof(RF=PROD):
    _assert(RF is PROD, "F26 runs only against the production module")
    survivors: List[str] = []

    for label, replacements, gate_names in MUTANTS:
        try:
            mutant = build_mutant(replacements)
        except AssertionError as exc:
            _MUTATION_REPORT.append(f"  RED  {label}\n         anchor: {exc}")
            continue
        killed_by, signature = None, ""
        for gate_name in gate_names:
            try:
                _GATE_BY_NAME[gate_name](mutant)
            except Exception as exc:                        # noqa: BLE001
                killed_by = gate_name
                signature = f"{type(exc).__name__}: {str(exc).splitlines()[0][:150]}"
                break
        if killed_by is None:
            survivors.append(f"{label} (expected kill by {gate_names})")
            _MUTATION_REPORT.append(f"  SURVIVED  {label}")
        else:
            _MUTATION_REPORT.append(
                f"  RED  {label}\n         killed by {killed_by}: {signature}")

    for label, replacements, gate_name in INTEGRATION_MUTANTS:
        mutated = mutate_integration(replacements)
        try:
            if gate_name == "F31":
                f31_failure_propagates_through_optimize_window(PROD, source=mutated)
            else:
                f36_legacy_deduplicator_has_no_authority(PROD, source=mutated)
        except Exception as exc:                            # noqa: BLE001
            _MUTATION_REPORT.append(
                f"  RED  {label}\n         killed by {gate_name}: "
                f"{type(exc).__name__}: {str(exc).splitlines()[0][:150]}")
        else:
            survivors.append(f"{label} (expected kill by {gate_name})")
            _MUTATION_REPORT.append(f"  SURVIVED  {label}")

    _assert(not survivors,
            "F26: mutants survived their gates:\n  " + "\n  ".join(survivors))


_GATE_BY_NAME.update({
    "F1": f1_new_new_unequal,
    "F2": f2_equal_score_lower_trial_wins,
    "F3": f3_same_trial_constant_wins,
    "F4": f4_float32_tie,
    "F5": f5_prior_new_unequal_replaces,
    "F6": f6_prior_tie_retained_byte_for_byte,
    "F7": f7_same_trial_same_mode_collision,
    "F8": f8_order_independence,
    "F12": f12_parent_artifact_hash_mismatch,
    "F13": f13_prior_without_sidecar,
    "F15": f15_no_fallback_writer,
    "F16": f16_seed_below_start,
    "F17": f17_seed_at_or_above_end,
    "F18": f18_overflow_rejected,
    "F22": f22_published_bundle_contract,
    "F24": f24_sidecar_artifact_hash,
    "F27": f27_malformed_losing_candidate,
    "F28": f28_l3_is_array_domain,
    "F29": f29_single_commit_visibility,
    "F30": f30_fsync_before_pointer_swap,
    "F33": f33_parent_sidecar_hash_mismatch,
    "F38": f38_first_generation_alias_bootstrap,
    "F41": f41_candidate_prng_base_mismatch,
    "F42": f42_candidate_mode_not_executed,
    "F44": f44_detached_prior,
    "F45": f45_recursive_chain_failures,
    "F47": f47_sidecar_hash_binding,
    "F48": f48_modified_current_sidecar,
    "F50": f50_current_present_prior_omitted_merges,
    "S2": s2_published_generation_carries_the_nine_values,
    "S3": s3_fail_closed_matrix,
    "S4": s4_pre_v1_1_sidecar_fails_closed,
    "S6": s6_parent_disagreement_fails_closed,
    "S8": s8_other_versions_and_array_contract_unchanged,
})


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


GATES = [
    ("F1: new/new unequal -> higher score wins", f1_new_new_unequal),
    ("F2: equal score, different trial AND mode -> lower trial wins",
     f2_equal_score_lower_trial_wins),
    ("F3: equal score, same trial -> constant wins", f3_same_trial_constant_wins),
    ("F4: equal under float32 -> tie, lower trial wins", f4_float32_tie),
    ("F5: prior/new unequal -> new replaces in the ARRAY domain",
     f5_prior_new_unequal_replaces),
    ("F6: prior/new tie -> prior retained byte-for-byte in all 22 arrays",
     f6_prior_tie_retained_byte_for_byte),
    ("F7: same-trial/same-mode collision -> accumulator-consistency error",
     f7_same_trial_same_mode_collision),
    ("F8: ordering independence", f8_order_independence),
    ("F9: publication failure -> prior artifact AND sidecar byte-identical",
     f9_publication_failure_leaves_prior_intact),
    ("F10: merge failure -> no canonical artifact written", f10_merge_failure_writes_nothing),
    ("F11: sidecar write failure -> no accepted generation", f11_sidecar_write_failure),
    ("F12: parent-artifact-hash mismatch -> fail closed",
     f12_parent_artifact_hash_mismatch),
    ("F13: prior without sidecar -> fail closed", f13_prior_without_sidecar),
    ("F14: uncertified historical NPZ never imported",
     f14_uncertified_historical_artifact_never_imported),
    ("F15: no fallback subprocess or legacy writer (source + behavioral)",
     f15_no_fallback_writer),
    ("F16: candidate below seed_start -> fail", f16_seed_below_start),
    ("F17: candidate at/above seed_end_exclusive -> fail", f17_seed_at_or_above_end),
    ("F18: overflow rejected by the Python-integer domain check; the "
     "fixed-width mutant certifies false coverage", f18_overflow_rejected),
    ("F19: declared coverage outside [0, 2**32) -> fail",
     f19_declared_coverage_outside_domain),
    ("F20: candidate seed outside [0, 2**32) -> fail",
     f20_candidate_seed_outside_domain),
    ("F21: valid range succeeds; coverage in result AND sidecar",
     f21_valid_range_records_coverage),
    ("F22: published artifact = frozen 22 keys/order/dtypes, seed-ascending",
     f22_published_bundle_contract),
    ("F23: clean start -> parent_* null; rows == L2 winners", f23_clean_start),
    ("F24: sidecar artifact_sha256 == the published file's actual hash",
     f24_sidecar_artifact_hash),
    ("F25: MinerTrialAssembly path fields remain None",
     f25_miner_path_fields_remain_none),
    ("F27: malformed LOSING raw candidate -> fail before L2",
     f27_malformed_losing_candidate),
    ("F28: L3 is array-domain; no 22->24 reconstruction", f28_l3_is_array_domain),
    ("F29: artifact + sidecar + both aliases visible through ONE swap",
     f29_single_commit_visibility),
    ("F30: NPZ/sidecar/directory fsyncs all precede the pointer swap",
     f30_fsync_before_pointer_swap),
    ("F31: finalizer failure propagates through optimize_window",
     f31_failure_propagates_through_optimize_window),
    ("F32: current unchanged after every injected failure (steps 1-11)",
     f32_current_unchanged_after_every_injected_failure),
    ("F33: parent SIDECAR hash mismatch -> fail closed",
     f33_parent_sidecar_hash_mismatch),
    ("F34: duplicate/unsorted prior seeds -> fail closed",
     f34_prior_seeds_unsorted_or_duplicated),
    ("F35: invalid prior skip_mode / prng_type ids -> fail closed",
     f35_prior_invalid_ids),
    ("F36: legacy score-only dedup has no canonical authority",
     f36_legacy_deduplicator_has_no_authority),
    ("F37: dirty repository state cannot certify", f37_dirty_tree_cannot_certify),
    ("F38: first-generation alias bootstrap before the commit",
     f38_first_generation_alias_bootstrap),
    ("F39: conflicting regular file / wrong-target alias -> fail closed",
     f39_conflicting_root_alias),
    ("F40: failure after bootstrap, before commit -> dangling aliases only",
     f40_failure_after_bootstrap_before_commit),
    ("F41: candidate prng_base != run identity -> fail",
     f41_candidate_prng_base_mismatch),
    ("F42: candidate mode not executed -> fail; zero-row mode succeeds",
     f42_candidate_mode_not_executed),
    ("F43: prior row identity-inconsistent -> fail closed",
     f43_prior_identity_inconsistent),
    ("F44: detached prior / prior without current -> fail closed", f44_detached_prior),
    ("F45: recursive chain failures each fail closed", f45_recursive_chain_failures),
    ("F46: malformed prior numeric domains each fail closed", f46_prior_numeric_domains),
    ("F47: no sidecar_sha256 key; hash over the final stored bytes",
     f47_sidecar_hash_binding),
    ("F48: modified current provenance.json -> next run fails closed",
     f48_modified_current_sidecar),
    ("F49: malformed current target -> fail closed", f49_malformed_current_target),
    ("F50: current present + prior omitted -> automatic merge", f50_current_present_prior_omitted_merges),
    ("F51: post-swap fsync failure -> PublicationDurabilityError + recovery",
     f51_post_swap_durability_failure),
    ("S1: sidecar required key set is exactly the 32 hand-transcribed keys",
     s1_sidecar_key_set_is_exactly_32),
    ("S2: a published v1.1 generation carries the nine seed-domain values",
     s2_published_generation_carries_the_nine_values),
    ("S3: seed-domain fail-closed matrix (one case per field + bool-as-int)",
     s3_fail_closed_matrix),
    ("S4: a pre-v1.1 sidecar fails closed, never read as high16 = 0",
     s4_pre_v1_1_sidecar_fails_closed),
    ("S5: a valid v1.1 chain publishes and validates recursively",
     s5_valid_v1_1_publishes_and_chains),
    ("S6: parent disagreement on any per-link field fails closed",
     s6_parent_disagreement_fails_closed),
    ("S7: cross-domain lineage mismatch fails on the seed-domain contract",
     s7_cross_domain_lineage_mismatch),
    ("S8: ARTIFACT/ENCODING versions and the 22-array contract unchanged",
     s8_other_versions_and_array_contract_unchanged),
    ("F26: mutation proof", f26_mutation_proof),
    ("S9: D3.5-B mutation proof", s9_mutation_proof),
]


def main():
    print("=" * 78)
    print("S172 Phase-5 D3.5 — shared run finalizer (L2 / L3 / immutable "
          "generation publication)")
    print("=" * 78)

    for name, fn in GATES:
        _check(name, fn)

    if _MUTATION_REPORT:
        print("\n" + "-" * 78)
        print("F26 mutation evidence (red signature per mutant):")
        print("-" * 78)
        for line in _MUTATION_REPORT:
            print(line)

    for path in _TEMP_ROOTS:
        shutil.rmtree(path, ignore_errors=True)

    print("=" * 78)
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{passed}/{total} D3.5 gate checks green\n")
    if passed != total:
        print("FAILURES (DO NOT COMMIT):\n")
        for name, ok, tb in _results:
            if not ok:
                print(f"--- {name} ---\n{tb}")
        return 1
    print("All D3.5 gate checks green — the shared finalizer selects L2 winners "
          "in the record domain, merges L3 in the array domain, and publishes "
          "immutable chain-authenticated generations through a single "
          "pointer commit, each labelled with the Seed-Domain v1.1 stratum "
          "contract and homogeneous across every lineage link (pending Team "
          "Alpha + Team Beta review).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
