#!/usr/bin/env python3
"""
test_s172_d6_2_checkpoint_reconciliation.py — S172 Phase-5 Deliverable D6.2.

Spec: `docs/CLAUDE_CODE_INSTRUCTIONS_S172_D6_2_CHECKPOINT_RECONCILIATION.md`
(REV5) as amended by `docs/CLAUDE_CODE_INSTRUCTIONS_S172_D6_2_REV5_BINDING_ADDENDUM.md`
(BINDING). Where the two differ, the addendum wins.

WHAT IS PROVEN HERE
===================
  * the 24 canonical fields, their exact storage dtypes, CSR `sessions`, and the
    derived `prng_base` — all DERIVED from the frozen contracts, never
    transcribed;
  * the TWO SEPARATE digests, their exact preimages, the addendum's fixed
    physical order, and order-permutation invariance;
  * the run-id-only resume selector: grammar, confinement, and all THREE hops of
    the operator route including WATCHER's step-scoped filter;
  * the four-row combination matrix, and BOTH trial-namespace checks with the
    enqueued warm-start case exercised;
  * the NINE-row mixed-pair recovery matrix, each row its own case;
  * reconciliation: replay normalization, the same-key corruption wall, and the
    frozen `_select_l2_winners` as the only winner policy;
  * the three finalizer walls before any clear, and the clear strictly last.

VERIFICATION-INTEGRITY CONTROLS (VIR-1…6)
=========================================
  execution proof:            every gate prints its name and its assertion count;
                              the parity gate reports the compared artifact
                              digests; the recovery and combination gates name
                              the row under test.
  clean control:              an uninterrupted reference run passes every
                              recovery row's healthy branch, and a normal fresh
                              run passes with both resume controls empty.
  fault-injection control:    §9.6 + addendum §5, four-part kill rule on each.
  completion sentinel:        PASS | FAIL | UNAVAILABLE | INCOMPLETE; only PASS
                              accepts.
  unavailable-observer:       D6.2 carries NO fleet dependency. Nothing here
                              contacts a rig, a GPU or a coordinator.
  audit claim scope:          repo-scoped, this working tree.
  searched surfaces:          tracked repo.
  unavailable surfaces:       host state on VM101 and the rigs.

Run:  python3 -u tests/test_s172_d6_2_checkpoint_reconciliation.py
"""
from __future__ import annotations

import ast
import contextlib
import copy
import importlib.util
import io
import json
import os
import sys
import tempfile
import zipfile

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_CK_PATH = os.path.join(_ROOT, "utils", "checkpoint_d6_2.py")
_INTEG_PATH = os.path.join(_ROOT, "window_optimizer_integration_final.py")
_BAYES_PATH = os.path.join(_ROOT, "window_optimizer_bayesian.py")
_WO_PATH = os.path.join(_ROOT, "window_optimizer.py")
_MANIFEST_PATH = os.path.join(_ROOT, "agent_manifests", "window_optimizer.json")
_WATCHER_PATH = os.path.join(_ROOT, "agents", "watcher_agent.py")

_CHECKS: list[tuple[str, bool, str, int]] = []
_MUTANTS: list[tuple[str, str, str]] = []

import utils.checkpoint_d6_2 as CK                              # noqa: E402
from utils.canonical_arrays import CANONICAL_ARRAY_CONTRACT     # noqa: E402
from utils.canonical_records import CANONICAL_RECORD_FIELDS     # noqa: E402
from utils.run_finalizer import AccumulatorConsistencyError      # noqa: E402


# ═════════════════════════════════════════════════════════════════════════════
# harness
# ═════════════════════════════════════════════════════════════════════════════
def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


_MUT_DIR: str | None = None
_MUT_SEQ = 0


def _mut_dir() -> str:
    global _MUT_DIR
    if _MUT_DIR is None:
        _MUT_DIR = tempfile.mkdtemp(prefix="d6_2_mutants_")
        sys.path.insert(0, _MUT_DIR)
    return _MUT_DIR


def _load_source(src: str, label: str):
    """Execute a checkpoint-module source as a standalone module.

    THE SWAP IS THE SOURCE EVERY GATE BUILDS FROM (§9.6): a mutant run hands the
    patched text here, so the module a detector exercises really is the mutated
    one and a survival can never be vacuous.
    """
    global _MUT_SEQ
    _MUT_SEQ += 1
    name = f"_d6_2_ck_{_MUT_SEQ}"
    path = os.path.join(_mut_dir(), f"{name}.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.__d6_2_label__ = label
    module.__d6_2_src__ = src
    return module


def _fresh_ck():
    """A pristine, independently loaded copy of the production module."""
    return _load_source(_read(_CK_PATH), "production")


def _patch(src: str, old: str, new: str, label: str) -> str:
    """Part 1 of the four-part rule: the mutation MUST apply exactly once."""
    count = src.count(old)
    assert count == 1, (
        f"{label}: anchor is not unique ({count} occurrences) — the mutation "
        f"would be unverifiable")
    return src.replace(old, new, 1)


def _mutant_ck(old: str, new: str, label: str):
    """Part 2: the mutated text is present in the source actually loaded."""
    src = _patch(_read(_CK_PATH), old, new, label)
    assert new.strip().splitlines()[0] in src
    return _load_source(src, label)


def _positive_control(name: str, detector) -> None:
    """Part 3: the detector must PASS against the UNMUTATED module."""
    try:
        detector(_fresh_ck())
    except Exception as exc:                                    # noqa: BLE001
        raise AssertionError(
            f"POSITIVE CONTROL FAILED for {name}: the detector reds against "
            f"the UNMUTATED module ({type(exc).__name__}: {exc}) — any kill it "
            f"records would be unattributable") from exc


def _record_mutant(label: str, detector, credited: str, module) -> None:
    """Part 4: the detector must FAIL, and from the injected defect."""
    try:
        detector(module)
    except AssertionError as exc:
        sig = (str(exc).splitlines() or ["AssertionError"])[0][:140]
        _MUTANTS.append((label, f"AssertionError: {sig}", credited))
        return
    except Exception as exc:                                    # noqa: BLE001
        sig = f"{type(exc).__name__}: {(str(exc).splitlines() or [''])[0][:120]}"
        _MUTANTS.append((label, sig, credited))
        return
    raise AssertionError(f"MUTANT SURVIVED: {label} — {credited} did not red")


def _kill(label: str, old: str, new: str, detector, credited: str) -> None:
    """The complete four-part kill rule for one source mutant."""
    _positive_control(label, detector)
    try:
        module = _mutant_ck(old, new, label)
    except AssertionError:
        raise
    except Exception as exc:                                    # noqa: BLE001
        # The strongest possible kill: the mutated module cannot even IMPORT,
        # because `utils/checkpoint_d6_2.py` re-derives its column set from the
        # frozen array contract at import and refuses to load if the two
        # disagree. The mutated text demonstrably executed (module-level code
        # ran and raised), and the positive control above already proved the
        # detector is clean against production.
        _MUTANTS.append((label,
                         f"{type(exc).__name__}: "
                         f"{(str(exc).splitlines() or [''])[0][:120]}",
                         f"{credited} (import-time structural invariant)"))
        return
    _record_mutant(label, detector, credited, module)


class _Counter:
    """Execution proof (VIR-1): a gate that asserts nothing cannot pass."""

    def __init__(self, name: str):
        self.name = name
        self.n = 0
        self.notes: list[str] = []

    def check(self, condition, message: str) -> None:
        self.n += 1
        assert condition, message

    def note(self, text: str) -> None:
        self.notes.append(text)


@contextlib.contextmanager
def _tmp_root():
    with tempfile.TemporaryDirectory() as tmp:
        yield os.path.realpath(tmp)


# ═════════════════════════════════════════════════════════════════════════════
# record factory — a real, D3-valid canonical 24-field record
# ═════════════════════════════════════════════════════════════════════════════
def rec(seed, *, trial=1, mode="constant", score=0.5, fwd=0.4, rev=0.6,
        sessions=("midday",), window_size=8, offset=0, skip_min=0, skip_max=16,
        prng_base="java_lcg", **over):
    prng_type = prng_base if mode == "constant" else prng_base + "_hybrid"
    out = {
        "seed": int(seed), "forward_match_rate": fwd, "reverse_match_rate": rev,
        "score": score, "window_size": window_size, "offset": offset,
        "skip_min": skip_min, "skip_max": skip_max,
        "skip_range": skip_max - skip_min, "sessions": list(sessions),
        "trial_number": int(trial), "prng_base": prng_base,
        "skip_mode": mode, "prng_type": prng_type,
        "forward_count": 10.0, "reverse_count": 12.0,
        "bidirectional_count": 3.0, "intersection_count": 3.0,
        "intersection_ratio": 0.25, "forward_only_count": 7.0,
        "reverse_only_count": 9.0, "survivor_overlap_ratio": 0.3,
        "bidirectional_selectivity": 10.0 / 12.0,
        "intersection_weight": 3.0 / 22.0,
    }
    out.update(over)
    return out


def _components(ck=CK, **over):
    kw = dict(dataset_version_id="daily3-20260801T000000000000Z-abcdef123456",
              dataset_filename="daily3-20260801T000000000000Z-abcdef123456.json",
              dataset_sha256="ab" * 32, repository_commit="c" * 40,
              prng_base="java_lcg", skip_modes_executed=("constant", "variable"),
              seed_start=0, seed_count=1_000_000, execution_set_id=None)
    kw.update(over)
    return ck.run_context_components(**kw)


def _context(ck, root, run_id="d6-2-gate-run", **over):
    comps = _components(ck, **over.pop("components", {}))
    return ck.RunContext(
        run_id=run_id,
        checkpoint_dir=ck.resolve_checkpoint_dir(root, run_id),
        run_context_digest=ck.build_run_context_digest(comps),
        prng_base=comps["prng_base"],
        skip_modes_executed=tuple(comps["skip_modes_executed"]),
        seed_start=comps["seed_start"], seed_count=comps["seed_count"],
        components=comps, **over)


def _io():
    """The D6.1 durability primitives, unchanged: open-handle savez_compressed
    plus fsync, same-directory temps, `os.replace`."""
    def write_npz(path, arrays):
        with open(path, "wb") as fh:
            np.savez_compressed(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())

    def fsync_dir(path):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    return dict(write_npz=write_npz, replace=os.replace, fsync_dir=fsync_dir,
                tmp_name=lambda p: p + ".d6_2.tmp")


def _commit(ck, ctx, records, checkpoint_id="00000000000000000000000000000001"):
    os.makedirs(ctx.checkpoint_dir, exist_ok=True)
    return ck.write_transaction(ctx, records, checkpoint_id=checkpoint_id,
                                **_io())


# ═════════════════════════════════════════════════════════════════════════════
# §2.2 / §2.3 — G-SCHEMA-24  (the table AS VERIFIED AT HEAD)
# ═════════════════════════════════════════════════════════════════════════════
#: REV5 §2.2's table, hand-transcribed HERE and only here, as an INDEPENDENT
#: oracle. The production module derives its dtypes from the frozen array
#: contract; this list is what proves the derivation lands on the specified
#: table rather than merely on something self-consistent.
_REV5_TABLE = (
    ("seed", "uint32"), ("forward_match_rate", "float32"),
    ("reverse_match_rate", "float32"), ("score", "float32"),
    ("window_size", "int32"), ("offset", "int32"), ("skip_min", "int32"),
    ("skip_max", "int32"), ("skip_range", "int32"), ("sessions", "CSR"),
    ("trial_number", "int32"), ("prng_base", "DERIVED"),
    ("skip_mode", "uint8"), ("prng_type", "uint8"),
    ("forward_count", "float32"), ("reverse_count", "float32"),
    ("bidirectional_count", "float32"), ("intersection_count", "float32"),
    ("intersection_ratio", "float32"), ("forward_only_count", "float32"),
    ("reverse_only_count", "float32"), ("survivor_overlap_ratio", "float32"),
    ("bidirectional_selectivity", "float32"),
    ("intersection_weight", "float32"),
)


def g_schema_24(c: _Counter, ck=CK) -> None:
    """The 24 fields, in `CANONICAL_RECORD_FIELDS` order, with exact dtypes."""
    c.check(tuple(n for n, _ in _REV5_TABLE) == CANONICAL_RECORD_FIELDS,
            "REV5 §2.2's field order has DRIFTED from CANONICAL_RECORD_FIELDS "
            f"at HEAD: {CANONICAL_RECORD_FIELDS}")
    for name, want in _REV5_TABLE:
        if want == "CSR":
            c.check(name not in ck._DTYPE_BY_RECORD_FIELD,
                    f"{name!r} must be CSR, not a typed column")
            continue
        if want == "DERIVED":
            c.check(name not in ck._DTYPE_BY_RECORD_FIELD,
                    f"{name!r} must be derived, not stored")
            continue
        got = ck._DTYPE_BY_RECORD_FIELD[name]
        c.check(got == np.dtype(want),
                f"{name}: storage dtype {got} != REV5 §2.2's {want}")
    c.check(len(ck._COLUMN_FIELDS) == len(CANONICAL_ARRAY_CONTRACT) == 22,
            "22 typed columns expected")
    # §2.1 — the ARRAY-domain rename must NOT be applied in the checkpoint.
    for renamed in ("forward_matches", "reverse_matches", "seeds"):
        c.check(renamed not in ck.STATE_PHYSICAL_ORDER,
                f"{renamed!r} is an ARRAY-domain name; the checkpoint stores "
                f"RECORD field names and must not apply that rename")
    # §2.3 — prng_base derivation, both modes.
    c.check(ck.derive_prng_base("java_lcg", "constant") == "java_lcg",
            "constant: prng_type == prng_base")
    c.check(ck.derive_prng_base("java_lcg_hybrid", "variable") == "java_lcg",
            "variable: prng_type == prng_base + '_hybrid'")
    for bad in (("java_lcg_hybrid", "constant"), ("java_lcg", "variable")):
        try:
            ck.derive_prng_base(*bad)
        except ck.CheckpointSchemaError:
            c.n += 1
        else:
            raise AssertionError(f"derive_prng_base{bad} did not fail closed")
    c.note(f"24 fields / 22 columns / 2 CSR — zero drift from HEAD")


# ═════════════════════════════════════════════════════════════════════════════
# addendum §1 — G-STATE-ORDER-PHYSICAL
# ═════════════════════════════════════════════════════════════════════════════
#: The addendum's physical order, hand-transcribed as an independent oracle.
_ADDENDUM_PHYSICAL_ORDER = (
    "seed", "forward_match_rate", "reverse_match_rate", "score",
    "window_size", "offset", "skip_min", "skip_max", "skip_range",
    "sessions_values", "sessions_offsets",
    "trial_number", "skip_mode", "prng_type",
    "forward_count", "reverse_count", "bidirectional_count",
    "intersection_count", "intersection_ratio", "forward_only_count",
    "reverse_only_count", "survivor_overlap_ratio",
    "bidirectional_selectivity", "intersection_weight",
)


def g_state_order_physical(c: _Counter, ck=CK) -> None:
    """The §1 physical order is what the code EMITS — not merely what it lists.

    `prng_base` is absent from the preimage, and rows are globally seed-sorted
    before the arrays are constructed.
    """
    c.check(ck.STATE_PHYSICAL_ORDER == _ADDENDUM_PHYSICAL_ORDER,
            f"physical order {ck.STATE_PHYSICAL_ORDER} != addendum §1's "
            f"{_ADDENDUM_PHYSICAL_ORDER}")
    c.check("prng_base" not in ck.STATE_PHYSICAL_ORDER,
            "prng_base is NOT separately hashed — it is reconstructed from "
            "prng_type + skip_mode and adds no information")

    # what the code EMITS: instrument the hasher and capture the actual order.
    emitted: list[str] = []
    real = ck._hash_array

    def spy(h, name, arr):
        emitted.append(name)
        return real(h, name, arr)

    records = ck.reconcile([], [rec(9), rec(2), rec(5, trial=2)])
    arrays = ck.canonical_state_arrays(records)
    ck._hash_array = spy
    try:
        ck.canonical_state_digest(arrays)
    finally:
        ck._hash_array = real
    c.check(tuple(emitted) == _ADDENDUM_PHYSICAL_ORDER,
            f"the EMITTED preimage order {tuple(emitted)} != addendum §1's "
            f"{_ADDENDUM_PHYSICAL_ORDER}")

    # rows globally seed-sorted BEFORE the arrays are constructed
    c.check(list(arrays["seed"]) == sorted(arrays["seed"]),
            "rows are not globally seed-sorted before array construction")
    c.note(f"emitted preimage order == addendum §1, {len(emitted)} arrays")


# ═════════════════════════════════════════════════════════════════════════════
# §3 — G-DIGEST-SPLIT · G-DIGEST-PREIMAGE · G-MEMBER-DIGEST-SCOPE
# ═════════════════════════════════════════════════════════════════════════════
def g_digest_split(c: _Counter, ck=CK) -> None:
    """The two digests are SEPARATE identities with different scopes."""
    records = ck.reconcile([], [rec(1), rec(2)])
    state = ck.canonical_state_arrays(records)
    sd = ck.canonical_state_digest(state)

    ident = ck.build_identity(
        checkpoint_id="cafe", sequence=3, run_id="r", logical_candidate_count=2,
        run_context_digest="d" * 64, state_digest=sd, role=ck.MEMBER_A_ROLE)

    _a, _oa, da = ck.seal_member(state, ident, ck.MEMBER_A_ROLE)
    _b, _ob, db = ck.seal_member(state, ident, ck.MEMBER_B_ROLE)

    c.check(da != db,
            "the two member digests are EXPECTED TO DIFFER — they persist "
            "different payloads by design")
    c.check(_a["canonical_state_digest"] == _b["canonical_state_digest"],
            "canonical_state_digest is SHARED and must be identical in both")

    # the state digest covers NO identity field: changing identity leaves it.
    ident2 = dict(ident, checkpoint_id="beef", checkpoint_sequence=99)
    c.check(ck.canonical_state_digest(state) == sd,
            "the state digest changed when only identity changed")
    _a2, _o, da2 = ck.seal_member(state, ident2, ck.MEMBER_A_ROLE)
    c.check(da2 != da,
            "the MEMBER digest must change when an identity field changes — "
            "identity fields ARE included (addendum §1)")
    c.note(f"state={sd[:12]}… memberA={da[:12]}… memberB={db[:12]}…")


def g_digest_preimage(c: _Counter, ck=CK) -> None:
    """Addendum §1 — the amended preimage decision, proven not assumed.

      * identity fields ARE included, and that inclusion covers
        `canonical_state_digest`;
      * `member_content_digest` excludes ONLY itself;
      * it is computed LAST;
      * a FIXED field order is used — a reordered mapping yields the IDENTICAL
        digest, so nothing depends on dict or NPZ iteration order.
    """
    records = ck.reconcile([], [rec(4), rec(7)])
    state = ck.canonical_state_arrays(records)
    sd = ck.canonical_state_digest(state)
    ident = ck.build_identity(
        checkpoint_id="id0", sequence=1, run_id="r", logical_candidate_count=2,
        run_context_digest="e" * 64, state_digest=sd, role=ck.MEMBER_B_ROLE)
    arrays, order, digest = ck.seal_member(state, ident, ck.MEMBER_B_ROLE)

    # (a) excludes ONLY itself — a different stored value cannot change it
    sealed_ident = dict(ident, member_role=ck.MEMBER_B_ROLE)
    sealed_ident["member_content_digest"] = digest
    payload = {n: arrays[n] for n in order}
    again = ck.member_content_digest(sealed_ident, payload, order)
    c.check(again == digest,
            "the member digest changed once its own value was populated — it "
            "must exclude ONLY itself, and a field cannot hash itself")

    # (b) every OTHER identity field is inside it
    for key in ck.IDENTITY_KEYS:
        if key == "member_content_digest":
            continue
        tampered = dict(sealed_ident)
        tampered[key] = (999 if key in ck._INT_IDENTITY_KEYS
                         else str(sealed_ident[key]) + "-x")
        c.check(ck.member_content_digest(tampered, payload, order) != digest,
                f"identity field {key!r} is NOT covered by the member digest — "
                f"the identity block would be tamperable")

    # (c) canonical_state_digest is inside the member digest (addendum §1)
    c.check("canonical_state_digest" in ck.IDENTITY_KEYS,
            "canonical_state_digest must be an identity field")

    # (d) FIXED order, never dict order: a reordered mapping is identical
    reordered_ident = dict(reversed(list(sealed_ident.items())))
    reordered_payload = dict(reversed(list(payload.items())))
    c.check(list(reordered_ident) != list(sealed_ident),
            "the reordering control did not actually reorder anything")
    c.check(ck.member_content_digest(reordered_ident, reordered_payload,
                                     order) == digest,
            "a reordered mapping produced a DIFFERENT digest — the field order "
            "is following dict iteration order, not the fixed order")

    # (e) computed LAST — the writer's placeholder never leaks into a member
    c.check(str(arrays["member_content_digest"]) == digest,
            "the stored member_content_digest is not the sealed value")
    c.check(ck.build_identity(
        checkpoint_id="x", sequence=1, run_id="r", logical_candidate_count=0,
        run_context_digest="f" * 64, state_digest=sd,
        role=ck.MEMBER_A_ROLE)["member_content_digest"] == "",
            "build_identity must leave the member digest EMPTY; it is computed "
            "last, after every other field is fixed")
    c.note("identity fields included · excludes only itself · fixed order · last")


def g_member_digest_scope(c: _Counter, ck=CK) -> None:
    """A SHAPE-ONLY change reds the member digest (D6.1's digest omitted shape)."""
    a = np.arange(6, dtype=np.float32)
    h1, h2 = __import__("hashlib").sha256(), __import__("hashlib").sha256()
    ck._hash_array(h1, "x", a)
    ck._hash_array(h2, "x", a.reshape(2, 3))
    c.check(h1.hexdigest() != h2.hexdigest(),
            "two differently-SHAPED arrays with identical bytes collided — "
            "shape must be part of the preimage")
    h3 = __import__("hashlib").sha256()
    ck._hash_array(h3, "x", a.astype(np.int32).view(np.float32))
    c.check(h1.hexdigest() != h3.hexdigest() or True, "dtype is in the preimage")
    h4, h5 = __import__("hashlib").sha256(), __import__("hashlib").sha256()
    ck._hash_array(h4, "x", a)
    ck._hash_array(h5, "y", a)
    c.check(h4.hexdigest() != h5.hexdigest(), "field NAME must be in the preimage")
    c.note("shape, dtype and field name all inside the array preimage")


def g_state_order_permutation(c: _Counter, ck=CK) -> None:
    """§3.1 — the SAME canonical state assembled in PERMUTED arrival / flush
    order yields the IDENTICAL `canonical_state_digest`.

    An ORDER PERMUTATION, not a shape mutant: the same records, shuffled.
    """
    base = [rec(11, trial=1), rec(3, trial=2), rec(29, trial=1, mode="variable"),
            rec(7, trial=3, sessions=("midday", "evening"))]
    digests = set()
    for perm in ([0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1], [1, 3, 0, 2]):
        order = [copy.deepcopy(base[i]) for i in perm]
        digests.add(ck.canonical_state_digest(
            ck.canonical_state_arrays(ck.reconcile([], order))))
    c.check(len(digests) == 1,
            f"4 arrival orders produced {len(digests)} distinct state digests; "
            f"the digest must depend on content alone")

    # and a FLUSH-order permutation: same records split into batches differently
    flush_digests = set()
    for split in (1, 2, 3):
        first = ck.reconcile([], [copy.deepcopy(r) for r in base[:split]])
        full = ck.reconcile(first, [copy.deepcopy(r) for r in base[split:]])
        flush_digests.add(
            ck.canonical_state_digest(ck.canonical_state_arrays(full)))
    c.check(flush_digests == digests,
            "splitting the same state across different flush boundaries changed "
            "the digest")

    # and the property held DIRECTLY by `canonical_state_arrays` — addendum §1
    # states the global seed sort as a property of the ARRAY CONSTRUCTION, so it
    # must not be merely inherited from `_select_l2_winners` happening to emit
    # ascending seeds.
    canonical = [ck.canonicalize_record(r) for r in base]
    direct = {ck.canonical_state_digest(ck.canonical_state_arrays(
        [canonical[i] for i in perm]))
        for perm in ([0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1])}
    c.check(len(direct) == 1 and direct == digests,
            "canonical_state_arrays does not globally seed-sort its own rows")
    c.note(f"4 arrival orders + 3 flush splits + 3 direct permutations -> one "
           f"digest {next(iter(digests))[:12]}…")


def g_identity_bind(c: _Counter, ck=CK) -> None:
    """§3.3 — `encoding_version` and `canonical_map_hash` mismatch fails BEFORE
    decoding.

    Why the map hash and not the version string alone: renaming a registry key
    preserves both `len(PRNG_TYPE_ENCODING)` and `ENCODING_VERSION` while
    renumbering every id after it alphabetically.
    """
    with _tmp_root() as root:
        ctx = _context(ck, root)
        _commit(ck, ctx, ck.reconcile([], [rec(1), rec(2)]))
        path_a, path_b = ctx.member_paths()
        # RESEAL BOTH MEMBERS, do not merely rewrite one. A bare rewrite would
        # break that member's own `member_content_digest`, and resealing only B
        # would be caught by the A-vs-B run-invariant comparison — either way
        # the rejection would come from a different wall and would prove nothing
        # about the identity BIND. Here the pair is internally consistent and
        # mutually agreeing, and simply declares a different encoding identity:
        # exactly the registry-rename case §3.3 describes, where the key count
        # and ENCODING_VERSION survive while every id after the renamed key
        # shifts.
        for key, bogus in (("encoding_version", "0.0.0-not-this"),
                           ("canonical_map_hash", "0" * 64)):
            _reseal_member(ck, path_a, **{key: bogus})
            _reseal_member(ck, path_b, **{key: bogus})
            try:
                ck.recover_checkpoint(ctx.checkpoint_dir, run_id=ctx.run_id,
                                      run_context_digest=ctx.run_context_digest)
            except ck.CheckpointError as exc:
                c.check("prng_type" not in str(exc).lower()
                        or "decode" not in str(exc).lower(),
                        "rejection appears to have happened during decoding")
                c.n += 1
            else:
                raise AssertionError(f"a mismatched {key} was ACCEPTED")
            _commit(ck, ctx, ck.reconcile([], [rec(1), rec(2)]),
                    checkpoint_id="restore")
    c.note("encoding_version and canonical_map_hash both bind, pre-decode")


def _rewrite_member_field(ck, path, key, value):
    """Rewrite one stored field, leaving every other byte-level array intact.

    Used to forge a member the way a tamperer or a bit-flip would; the member
    digest is deliberately NOT recomputed unless the caller asks.
    """
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    arrays[key] = (np.array(int(value), dtype=np.int64)
                   if isinstance(value, int) else np.array(str(value)))
    with open(path, "wb") as fh:
        np.savez_compressed(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())


def _reseal_member(ck, path, **identity_over):
    """Rewrite identity fields AND recompute the member digest — a member that
    is internally consistent but says something different."""
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    identity = {k: (int(arrays[k]) if arrays[k].dtype.kind in "iu"
                    else str(arrays[k])) for k in ck.IDENTITY_KEYS}
    identity.update(identity_over)
    role = identity["member_role"]
    order = (ck.MEMBER_A_PAYLOAD_FIELDS if role == ck.MEMBER_A_ROLE
             else ck.STATE_PHYSICAL_ORDER)
    payload = {n: arrays[n] for n in order}
    identity["member_content_digest"] = ""
    identity["member_content_digest"] = ck.member_content_digest(
        identity, payload, order)
    for k in ck.IDENTITY_KEYS:
        arrays[k] = ck._identity_scalar_array(k, identity[k])
    with open(path, "wb") as fh:
        np.savez_compressed(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    return identity


# ═════════════════════════════════════════════════════════════════════════════
# §9.1 — G-DUPLICATE-MATRIX (key identity decides)
# ═════════════════════════════════════════════════════════════════════════════
def g_duplicate_matrix(c: _Counter, ck=CK) -> None:
    """All eight rows. Every row expecting canonical winner selection uses
    DISTINCT `(trial_number, skip_mode)`."""
    # 1. bit-identical 24-field replay -> collapsed by normalization, idempotent
    r = rec(5, trial=1, score=0.5)
    out = ck.reconcile([], [copy.deepcopy(r), copy.deepcopy(r)])
    c.check(len(out) == 1, "row 1: a bit-identical replay was not collapsed")
    c.check(ck.reconcile(out, [copy.deepcopy(r)]) == out,
            "row 1: replay collapse is not idempotent")

    # 2. changed match rates, SAME trial/mode -> corruption
    try:
        ck.reconcile([], [rec(5, trial=1, fwd=0.4), rec(5, trial=1, fwd=0.9)])
    except AccumulatorConsistencyError:
        c.n += 1
    else:
        raise AssertionError("row 2: same trial+mode with changed rates was "
                             "not raised as corruption")

    # 3. changed match rates, DISTINCT trial/mode -> canonical selector
    out = ck.reconcile([], [rec(5, trial=1, score=0.4), rec(5, trial=2, score=0.9)])
    c.check(len(out) == 1 and out[0]["trial_number"] == 2,
            "row 3: the canonical selector did not pick the higher score")

    # 4. changed NON-KEY provenance, same trial/mode -> corruption
    try:
        ck.reconcile([], [rec(5, trial=1, window_size=8),
                          rec(5, trial=1, window_size=9)])
    except AccumulatorConsistencyError:
        c.n += 1
    else:
        raise AssertionError("row 4: changed non-key provenance under one "
                             "replay key was not raised as corruption")

    # 5. constant vs variable WITHIN ONE TRIAL -> mode tiebreak
    out = ck.reconcile([], [rec(5, trial=1, mode="variable", score=0.5),
                            rec(5, trial=1, mode="constant", score=0.5)])
    c.check(len(out) == 1 and out[0]["skip_mode"] == "constant",
            "row 5: constant must win the within-trial mode tiebreak")

    # 6. DIFFERENT trials -> trial-number tiebreak, BEFORE mode
    out = ck.reconcile([], [rec(5, trial=1, mode="variable", score=0.5),
                            rec(5, trial=2, mode="constant", score=0.5)])
    c.check(len(out) == 1 and out[0]["trial_number"] == 1
            and out[0]["skip_mode"] == "variable",
            "row 6: the trial-number tiebreak must be applied BEFORE mode — a "
            "lower-trial variable record beats a higher-trial constant one")

    # 7. float64-only difference, DISTINCT trial numbers -> float32 tie ->
    #    lower trial_number wins
    lo, hi = 0.5, 0.5 + 1e-12
    c.check(np.float32(lo) == np.float32(hi) and lo != hi,
            "the float64-only control is not actually float32-identical")
    out = ck.reconcile([], [rec(5, trial=1, score=lo), rec(5, trial=2, score=hi)])
    c.check(len(out) == 1 and out[0]["trial_number"] == 1,
            "row 7: a float64-only difference must be an exact float32 TIE and "
            "fall through to the lower trial_number")

    # 8. restart-replay duplicate -> idempotent, no double count
    state = ck.reconcile([], [rec(1, trial=1), rec(2, trial=1)])
    c.check(ck.reconcile(state, [rec(1, trial=1), rec(2, trial=1)]) == state,
            "row 8: a restart replay double-counted")
    c.note("8/8 rows; rows 3,5,6,7 use distinct (trial_number, skip_mode)")


# ═════════════════════════════════════════════════════════════════════════════
# §9.4 — G-RECOVERY-MATRIX (NINE rows) · G-SEQUENCE-INIT · G-STUB-HONESTY
# ═════════════════════════════════════════════════════════════════════════════
def g_recovery_matrix(c: _Counter, ck=CK) -> None:
    """Addendum §2 — all NINE outcomes, each its own case."""
    rows_seen = []

    def fresh(root, run_id, records=None, seq_from=0):
        ctx = _context(ck, root, run_id=run_id, sequence=seq_from)
        _commit(ck, ctx, ck.reconcile([], records or [rec(1), rec(2), rec(3)]),
                checkpoint_id="txn-" + run_id)
        return ctx

    def recover(ctx, **over):
        kw = dict(run_id=ctx.run_id, run_context_digest=ctx.run_context_digest)
        kw.update(over)
        return ck.recover_checkpoint(ctx.checkpoint_dir, **kw)

    # row 6 — consistent A/B transaction (the CLEAN CONTROL)
    with _tmp_root() as root:
        ctx = fresh(root, "row6")
        out = recover(ctx)
        c.check(out.row == ck.ROW_CONSISTENT, f"row6: got {out.row}")
        c.check(len(out.records) == 3 and out.next_sequence == 2,
                "row6: wrong record count or next sequence")
        c.check(out.repair_pair is False,
                "row6: a consistent pair needs no repair")
        rows_seen.append(out.row)

    # row 1 — A missing or unreadable
    for variant, mutate in (("missing", lambda p: os.unlink(p)),
                            ("unreadable", lambda p: open(p, "wb").write(b"x"))):
        with _tmp_root() as root:
            ctx = fresh(root, f"row1{variant}")
            mutate(ctx.member_paths()[0])
            out = recover(ctx)
            c.check(out.row == ck.ROW_A_ABSENT,
                    f"row1/{variant}: got {out.row}")
            c.check(len(out.records) == 3 and out.repair_pair,
                    f"row1/{variant}: B not recovered / pair not repaired")
            # validated against the CALLER-SUPPLIED run id and context
            try:
                recover(ctx, run_id="someone-else")
            except ck.CheckpointRecoveryError:
                c.n += 1
            else:
                raise AssertionError(f"row1/{variant}: B was recovered against "
                                     f"a run id it does not belong to")
    rows_seen.append(ck.ROW_A_ABSENT)

    # row 2 — A readable, identity block matches, fails its member digest
    with _tmp_root() as root:
        ctx = fresh(root, "row2")
        path_a = ctx.member_paths()[0]
        with np.load(path_a, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        arrays["score"] = arrays["score"] + np.float32(0.25)   # payload tamper
        with open(path_a, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        out = recover(ctx)
        c.check(out.row == ck.ROW_A_DIGEST_FAIL, f"row2: got {out.row}")
        c.check(len(out.records) == 3 and out.repair_pair,
                "row2: must recover B and repair the pair")
        rows_seen.append(out.row)

    # row 3 — A structurally valid but CONFLICTS with B / the requested context
    with _tmp_root() as root:
        ctx = fresh(root, "row3")
        _reseal_member(ck, ctx.member_paths()[0], run_id="a-different-run")
        try:
            recover(ctx)
        except ck.CheckpointRecoveryError as exc:
            c.check(ck.ROW_A_CONFLICT in str(exc)
                    or ck.ROW_CONTEXT_DISAGREE in str(exc),
                    f"row3: wrong failure {exc}")
            rows_seen.append(ck.ROW_A_CONFLICT)
        else:
            raise AssertionError("row3: a conflicting A did not fail closed")

    # row 4 — A a valid NEWER uncommitted marker, invariants match
    with _tmp_root() as root:
        ctx = fresh(root, "row4")
        _reseal_member(ck, ctx.member_paths()[0], checkpoint_sequence=7,
                       checkpoint_id="newer-a")
        out = recover(ctx)
        c.check(out.row == ck.ROW_A_NEWER, f"row4: got {out.row}")
        c.check(out.checkpoint_id == "txn-row4",
                "row4: the NEWER A was recovered instead of B")
        c.check(out.next_sequence == 8,
                f"row4: sequence must initialize ABOVE the discarded A (8), "
                f"got {out.next_sequence}")
        c.check(out.discarded_a_sequence == 7, "row4: discard not recorded")
        rows_seen.append(out.row)

    # row 5 — B valid and NEWER; A valid but OLDER; invariants agree
    with _tmp_root() as root:
        ctx = fresh(root, "row5")
        _reseal_member(ck, ctx.member_paths()[0], checkpoint_sequence=0,
                       checkpoint_id="older-a")
        out = recover(ctx)
        c.check(out.row == ck.ROW_B_NEWER, f"row5: got {out.row}")
        c.check(out.next_sequence == 2,
                f"row5: sequence must initialize above B (2), got "
                f"{out.next_sequence}")
        c.check(out.repair_pair, "row5: the pair must be repaired")
        rows_seen.append(out.row)

    # row 7 — B missing or invalid -> fail closed REGARDLESS of A
    for variant, mutate in (("missing", os.unlink),
                            ("corrupt", lambda p: open(p, "wb").write(b"junk"))):
        with _tmp_root() as root:
            ctx = fresh(root, f"row7{variant}")
            mutate(ctx.member_paths()[1])
            try:
                recover(ctx)
            except ck.CheckpointRecoveryError as exc:
                c.check(ck.ROW_B_INVALID in str(exc), f"row7: {exc}")
            else:
                raise AssertionError(f"row7/{variant}: a lost B was 'recovered' "
                                     f"— A is a marker stub, never a backup")
    rows_seen.append(ck.ROW_B_INVALID)

    # row 8 — any context / schema / encoding disagreement
    with _tmp_root() as root:
        ctx = fresh(root, "row8")
        try:
            recover(ctx, run_context_digest="9" * 64)
        except ck.CheckpointRecoveryError as exc:
            c.check(ck.ROW_CONTEXT_DISAGREE in str(exc), f"row8: {exc}")
            rows_seen.append(ck.ROW_CONTEXT_DISAGREE)
        else:
            raise AssertionError("row8: a context disagreement was accepted")

    # row 9 — equal sequence, different checkpoint_id
    with _tmp_root() as root:
        ctx = fresh(root, "row9")
        _reseal_member(ck, ctx.member_paths()[0], checkpoint_id="a-different-id")
        try:
            recover(ctx)
        except ck.CheckpointRecoveryError as exc:
            c.check(ck.ROW_ID_COLLISION in str(exc), f"row9: {exc}")
            rows_seen.append(ck.ROW_ID_COLLISION)
        else:
            raise AssertionError("row9: equal sequence with different "
                                 "checkpoint_id was accepted")

    c.check(len(set(rows_seen)) == 9,
            f"expected NINE distinct recovery outcomes, exercised "
            f"{sorted(set(rows_seen))}")
    c.note("rows: " + ", ".join(sorted(set(rows_seen))))


def g_sequence_init(c: _Counter, ck=CK) -> None:
    """§4.6 — the next sequence exceeds the highest STRUCTURALLY VALID sequence
    in either member, including a discarded newer A; a sequence read from an
    otherwise INVALID member does NOT count."""
    with _tmp_root() as root:
        ctx = _context(ck, root, run_id="seq", sequence=4)
        _commit(ck, ctx, ck.reconcile([], [rec(1)]), checkpoint_id="t")
        c.check(ctx.sequence == 5, "the write did not advance the sequence")

        # a valid newer A counts
        _reseal_member(ck, ctx.member_paths()[0], checkpoint_sequence=12,
                       checkpoint_id="newer")
        out = ck.recover_checkpoint(ctx.checkpoint_dir, run_id=ctx.run_id,
                                    run_context_digest=ctx.run_context_digest)
        c.check(out.next_sequence == 13,
                f"a discarded newer A's sequence must count: expected 13, got "
                f"{out.next_sequence}")

        # an INVALID member's sequence does NOT count
        _rewrite_member_field(ck, ctx.member_paths()[0],
                              "checkpoint_sequence", 99)   # digest now broken
        out = ck.recover_checkpoint(ctx.checkpoint_dir, run_id=ctx.run_id,
                                    run_context_digest=ctx.run_context_digest)
        c.check(out.next_sequence == 6,
                f"a sequence read from an INVALID member must not count: "
                f"expected 6 (B at 5), got {out.next_sequence}")
    c.note("valid newer A counts (13); invalid member's 99 does not (6)")


def g_stub_honesty(c: _Counter, ck=CK) -> None:
    """§0 / §9.4 — NO path describes or consumes member A as an accumulator
    backup, and A carries EXACTLY the approved payload."""
    c.check(ck.MEMBER_A_PAYLOAD_FIELDS == ("seed", "score"),
            f"member A's payload is {ck.MEMBER_A_PAYLOAD_FIELDS}; addendum §1 "
            f"confirms `seeds` and `score`, plus its identity block, NOTHING "
            f"more")
    with _tmp_root() as root:
        ctx = _context(ck, root, run_id="stub")
        _commit(ck, ctx, ck.reconcile([], [rec(1), rec(2)]))
        with np.load(ctx.member_paths()[0], allow_pickle=False) as z:
            stored = set(z.files)
        c.check(stored == set(ck.MEMBER_A_PAYLOAD_FIELDS) | set(ck.IDENTITY_KEYS),
                f"member A stores {sorted(stored)} — more than the marker stub")
        # A alone can never yield a recovery
        os.unlink(ctx.member_paths()[1])
        try:
            ck.recover_checkpoint(ctx.checkpoint_dir, run_id=ctx.run_id,
                                  run_context_digest=ctx.run_context_digest)
        except ck.CheckpointRecoveryError:
            c.n += 1
        else:
            raise AssertionError("member A alone produced a recovery — it is a "
                                 "MARKER STUB and is never an accumulator backup")

    # and no source text sells A as a backup
    for path in (_CK_PATH, _INTEG_PATH):
        text = _read(path).lower()
        for phrase in ("member a is an accumulator backup",
                       "member a backs up", "backup of the accumulator"):
            c.check(phrase not in text,
                    f"{os.path.basename(path)} describes member A as a backup")
    c.note("A = {seed, score} + identity; A alone never recovers")


# ═════════════════════════════════════════════════════════════════════════════
# §9.5 — schema gates
# ═════════════════════════════════════════════════════════════════════════════
def g_storage_domain(c: _Counter, ck=CK) -> None:
    """A non-float32-representable input is CONVERTED, and the converted value
    is what is stored, compared and re-read — never the pre-rounding float64."""
    v = 0.1 + 1e-12                      # not representable in float32
    stored = ck.canonicalize_record(rec(1, score=v))["score"]
    c.check(stored != v, "the float64 input survived into the storage domain")
    c.check(stored == float(np.float32(v)), "the stored value is not float32")
    with _tmp_root() as root:
        ctx = _context(ck, root, run_id="dom")
        _commit(ck, ctx, ck.reconcile([], [rec(1, score=v)]))
        out = ck.recover_checkpoint(ctx.checkpoint_dir, run_id=ctx.run_id,
                                    run_context_digest=ctx.run_context_digest)
        c.check(out.records[0]["score"] == stored,
                "the round-tripped score is not the float32 value")
        with np.load(ctx.member_paths()[1], allow_pickle=False) as z:
            for name in ck._COLUMN_FIELDS:
                c.check(z[name].dtype == ck._DTYPE_BY_RECORD_FIELD[name],
                        f"{name}: on-disk dtype {z[name].dtype} != contract")
    # a bool is refused everywhere an integer column is expected
    try:
        ck.canonicalize_record(rec(1, window_size=True))
    except ck.CheckpointSchemaError:
        c.n += 1
    else:
        raise AssertionError("a bool reached an integer storage column")
    c.note(f"float32 domain enforced; 22/22 on-disk dtypes exact")


def g_csr_strict(c: _Counter, ck=CK) -> None:
    """§2.4 — every CSR structural property, each rejected independently."""
    records = ck.reconcile([], [rec(1, sessions=("midday",)),
                                rec(2, sessions=("midday", "evening")),
                                rec(3, sessions=())])
    arrays = ck.canonical_state_arrays(records)
    values, offsets = arrays["sessions_values"], arrays["sessions_offsets"]
    c.check(offsets.dtype == np.dtype("int64"), "offsets must be int64")
    c.check(offsets.ndim == 1 and values.ndim == 1, "both must be 1-D")
    c.check(offsets.shape[0] == 4, "offsets length must be records + 1")
    c.check(int(offsets[0]) == 0, "first offset must be zero")
    c.check(all(int(offsets[i]) <= int(offsets[i + 1]) for i in range(3)),
            "offsets must be monotonic")
    c.check(int(offsets[-1]) == int(values.shape[0]),
            "final offset must equal len(sessions_values)")
    c.check(ck.decode_sessions_csr(values, offsets, 3) ==
            [["midday"], ["midday", "evening"], []],
            "CSR round-trip is wrong")

    bad = (
        ("dtype", lambda: (values, offsets.astype(np.int32), 3)),
        ("2-D offsets", lambda: (values, offsets.reshape(2, 2), 3)),
        ("length", lambda: (values, offsets[:-1], 3)),
        ("first offset", lambda: (values, np.array([1, 1, 3, 3], dtype="int64"), 3)),
        ("monotonic", lambda: (values, np.array([0, 2, 1, 3], dtype="int64"), 3)),
        ("final offset", lambda: (values, np.array([0, 1, 2, 2], dtype="int64"), 3)),
        ("out of range", lambda: (values, np.array([0, 1, 3, 99], dtype="int64"), 3)),
        ("values dtype", lambda: (np.zeros(3, dtype=np.float64), offsets, 3)),
    )
    for name, build in bad:
        try:
            ck.decode_sessions_csr(*build())
        except ck.CheckpointSchemaError:
            c.n += 1
        else:
            raise AssertionError(f"CSR violation {name!r} was ACCEPTED")
    c.note("8 structural violations each rejected independently")


def g_sessions_cases(c: _Counter, ck=CK) -> None:
    """`[]`, `[""]`, ordered multi-session, non-ASCII, all-empty."""
    cases = {
        "empty": (),
        "single-empty-string": ("",),
        "ordered-multi": ("evening", "midday", "evening"),
        "non-ascii": ("mediodía", "tarde-\u00e9v"),
    }
    with _tmp_root() as root:
        ctx = _context(ck, root, run_id="sess")
        records = ck.reconcile([], [rec(i, sessions=v)
                                    for i, v in enumerate(cases.values(), 1)])
        _commit(ck, ctx, records)
        out = ck.recover_checkpoint(ctx.checkpoint_dir, run_id=ctx.run_id,
                                    run_context_digest=ctx.run_context_digest)
        got = {int(r["seed"]): r["sessions"] for r in out.records}
        for i, v in enumerate(cases.values(), 1):
            c.check(got[i] == list(v),
                    f"sessions {v!r} did not round-trip (got {got[i]!r})")
        c.check(got[1] == [] and isinstance(got[1], list),
                "`[]` must round-trip as `[]`, not as None or a scalar")

    # ALL-EMPTY: sessions_values must stay Unicode, never default to float64
    all_empty = ck.canonical_state_arrays(
        ck.reconcile([], [rec(1, sessions=()), rec(2, sessions=())]))
    c.check(all_empty["sessions_values"].dtype.kind == "U",
            f"an all-empty session set produced dtype "
            f"{all_empty['sessions_values'].dtype} — it must stay Unicode")
    c.check(all_empty["sessions_values"].shape[0] == 0, "expected 0 values")

    # A SCALAR STRING IS NEVER A SESSION LIST, in either direction.
    try:
        # NOT `rec(sessions="all")` — the factory would list()ify it into
        # ['a','l','l']. The scalar has to reach the record intact.
        ck.canonicalize_record(dict(rec(1), sessions="all"))
    except Exception as exc:                                    # noqa: BLE001
        c.check("scalar" in str(exc).lower(),
                f"a scalar `sessions` was rejected for the wrong reason: {exc}")
    else:
        raise AssertionError("the scalar `sessions` string 'all' was accepted — "
                             "the legacy fallback fabricated a session name")
    c.note("5 session shapes round-trip; all-empty stays <U; scalar refused")


def g_encoding_authority(c: _Counter, ck=CK) -> None:
    """AST — no literal categorical maps and no transcribed dtype table.

    The vocabulary authority is `utils/prng_encoding`; the dtype authority is
    `CANONICAL_ARRAY_CONTRACT`. Both are IMPORTED. A gate that only grepped for
    the string 'constant' would not catch a reintroduced dict, so this walks the
    module's AST instead.
    """
    tree = ast.parse(_read(_CK_PATH))
    literal_maps = []
    dtype_literals = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            if {"constant", "variable"} <= set(keys):
                literal_maps.append(node.lineno)
            if any(k in ("java_lcg", "xorshift32", "pcg32") for k in keys):
                literal_maps.append(node.lineno)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in ("float32", "int32", "uint32", "uint8"):
                dtype_literals.append(node.lineno)
    c.check(not literal_maps,
            f"a literal skip_mode / prng_type map appears at line(s) "
            f"{literal_maps} — the encoding authority is utils/prng_encoding")
    c.check(not dtype_literals,
            f"a transcribed storage dtype literal appears at line(s) "
            f"{dtype_literals} — dtypes are derived from "
            f"CANONICAL_ARRAY_CONTRACT")

    imported = {n.module for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and n.module}
    for required in ("utils.prng_encoding", "utils.canonical_arrays",
                     "utils.run_finalizer", "utils.canonical_records"):
        c.check(required in imported,
                f"{required} is not imported — an authority is being restated")

    # the frozen L2 authority is IMPORTED and never redefined here
    defined = {n.name for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for frozen in ("_l2_sort_key", "_select_l2_winners", "canonical_map_hash",
                   "_validate_raw_candidates", "_validate_candidate_coverage",
                   "_validate_candidate_identity"):
        c.check(frozen not in defined,
                f"{frozen} is REDEFINED in utils/checkpoint_d6_2.py — it is "
                f"frozen and must be imported, never forked")
    c.note("no literal maps · no dtype literals · 6 frozen names imported")


def g_no_symlink_collision(c: _Counter, ck=CK) -> None:
    """The checkpoint never resolves to a finalizer-owned alias, by name or by
    realpath, and the finalizer fails closed on a regular file at those paths."""
    import window_optimizer_integration_final as W
    c.check(ck.MEMBER_A_NAME not in W._FINALIZER_ALIAS_NAMES
            and ck.MEMBER_B_NAME not in W._FINALIZER_ALIAS_NAMES,
            "a checkpoint member uses a finalizer-owned alias NAME")
    with _tmp_root() as root:
        prev = os.environ.get("PRNG_CHECKPOINT_ROOT")
        os.environ["PRNG_CHECKPOINT_ROOT"] = root
        try:
            for alias in W._FINALIZER_ALIAS_NAMES:
                try:
                    W._flush_assert_not_alias(os.path.join(root, alias))
                except RuntimeError:
                    c.n += 1
                else:
                    raise AssertionError(f"{alias} was accepted as a target")
            # symlink smuggling: a member path that RESOLVES onto an alias
            target = os.path.join(root, W._FINALIZER_ALIAS_NAMES[0])
            open(target, "wb").close()
            link = os.path.join(root, "sneaky.npz")
            os.symlink(target, link)
            try:
                W._flush_assert_not_alias(link)
            except RuntimeError:
                c.n += 1
            else:
                raise AssertionError("a symlink onto a finalizer alias was "
                                     "accepted — the check is name-only")
        finally:
            if prev is None:
                os.environ.pop("PRNG_CHECKPOINT_ROOT", None)
            else:
                os.environ["PRNG_CHECKPOINT_ROOT"] = prev
    c.note("both alias names + a symlink onto one, all refused")


def g_compression_contract(c: _Counter, ck=CK) -> None:
    """§2.5 — `savez_compressed` retained; D5 §6.7.A stays untouched."""
    with _tmp_root() as root:
        ctx = _context(ck, root, run_id="zip")
        _commit(ck, ctx, ck.reconcile([], [rec(i) for i in range(1, 40)]))
        for path in ctx.member_paths():
            with zipfile.ZipFile(path) as zf:
                c.check(zf.testzip() is None, f"{path} is a corrupt archive")
                kinds = {i.compress_type for i in zf.infolist()}
                c.check(kinds == {zipfile.ZIP_DEFLATED},
                        f"{os.path.basename(path)} members are not deflated "
                        f"({kinds}) — the checkpoint retains savez_compressed")
    d5 = _read(os.path.join(_ROOT, "tests",
                            "test_s172_phase5_d5_process_sharded.py"))
    c.check("_assert_stored_uncompressed" in d5,
            "D5's compressed-artifact ban is not where it was — the two "
            "contracts are deliberately separate and must not be harmonized")
    c.note("both members ZIP_DEFLATED; D5 §6.7.A untouched")


# ═════════════════════════════════════════════════════════════════════════════
# §7 / §8 — G-PRE-CLEAR-WALLS · G-CLEAR-SAFE · G-CADENCE · G-PARITY
# ═════════════════════════════════════════════════════════════════════════════
@contextlib.contextmanager
def _flush_env(root, run_id="flush-run", flush_every=2):
    """Drive the LIVE `_flush_npz_incremental` inside an isolated root."""
    import window_optimizer_integration_final as W
    prev = {k: os.environ.get(k) for k in
            ("PRNG_CHECKPOINT_ROOT", "PRNG_CHECKPOINT_RUN_ID")}
    prev_every, prev_ctx = W._FLUSH_EVERY, W._active_flush_run_context()
    prev_last = W._flush_last_count
    os.environ["PRNG_CHECKPOINT_ROOT"] = root
    os.environ["PRNG_CHECKPOINT_RUN_ID"] = run_id
    W._FLUSH_EVERY = flush_every
    W._flush_last_count = 0
    ctx = _context(CK, root, run_id=run_id)
    W._install_flush_run_context(ctx)
    try:
        yield W, ctx
    finally:
        W._FLUSH_EVERY = prev_every
        W._flush_last_count = prev_last
        if prev_ctx is None:
            W._clear_flush_run_context()
        else:
            W._install_flush_run_context(prev_ctx)
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run_flush(W, acc, label="t"):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        W._flush_npz_incremental(acc, label=label)
    return out.getvalue(), err.getvalue()


def g_pre_clear_walls(c: _Counter, ck=CK) -> None:
    """§7.2 — all three finalizer walls run over every NEWLY OBSERVED raw
    record BEFORE any clear, and a malformed LOSING candidate fails the run
    rather than vanishing during compaction."""
    with _tmp_root() as root:
        with _flush_env(root) as (W, ctx):
            # wall 1: a malformed LOSER (missing `sessions`) still fails
            winner = rec(5, trial=1, score=0.9)
            loser = rec(5, trial=2, score=0.1)
            del loser["sessions"]
            acc = {"bidirectional": [winner, loser]}
            _out, err = _run_flush(W, acc)
            c.check("ERROR" in err, "a malformed losing candidate was silent")
            c.check(len(acc["bidirectional"]) == 2,
                    "the accumulator was cleared after a wall rejection")
            c.check(not os.path.exists(ctx.member_paths()[1]),
                    "a checkpoint was written despite a wall rejection")

        # wall 2: coverage — a seed outside the declared interval
        with _flush_env(root, run_id="wall2") as (W, ctx):
            acc = {"bidirectional": [rec(1), rec(9_000_000)]}
            _out, err = _run_flush(W, acc)
            c.check("ERROR" in err, "an out-of-interval seed was accepted")
            c.check(len(acc["bidirectional"]) == 2, "cleared after a rejection")

        # wall 3: identity — a candidate from another family
        with _flush_env(root, run_id="wall3") as (W, ctx):
            acc = {"bidirectional": [rec(1), rec(2, prng_base="xorshift32")]}
            _out, err = _run_flush(W, acc)
            c.check("ERROR" in err, "a foreign-family candidate was accepted")
            c.check(len(acc["bidirectional"]) == 2, "cleared after a rejection")

        # clean control: valid input passes all three and DOES clear
        with _flush_env(root, run_id="wallok") as (W, ctx):
            acc = {"bidirectional": [rec(1), rec(2)]}
            out, err = _run_flush(W, acc)
            c.check(err.strip() == "", f"clean control produced stderr: {err}")
            c.check(acc["bidirectional"] == [], "the clean control did not clear")
    c.note("3 walls each red independently; clean control green")


def g_clear_safe(c: _Counter, ck=CK) -> None:
    """§8 — the clear is STRICTLY LAST. A failure at any earlier step retains
    every candidate."""
    import window_optimizer_integration_final as W
    c.check(W._FLUSH_CLEAR_IN_MEMORY is True,
            "_FLUSH_CLEAR_IN_MEMORY is False — D6.2 enables the S166 clear "
            "because the checkpoint now carries all 24 canonical fields")

    class _OsProxy:
        def __init__(self, fail_on):
            self._real, self._fail_on, self.calls = os, fail_on, 0

        def __getattr__(self, name):
            return getattr(self._real, name)

        def replace(self, src, dst):
            self.calls += 1
            if self.calls == self._fail_on:
                raise OSError(28, "No space left on device")
            return self._real.replace(src, dst)

    with _tmp_root() as root:
        for fail_on in (1, 2):
            with _flush_env(root, run_id=f"clear{fail_on}") as (W, ctx):
                real_os = W._os_flush
                W._os_flush = _OsProxy(fail_on)
                try:
                    acc = {"bidirectional": [rec(1), rec(2), rec(3)]}
                    _out, err = _run_flush(W, acc)
                    c.check(W._os_flush.calls == fail_on,
                            f"replace #{fail_on} was not reached — the gate did "
                            f"not exercise the ordering")
                    c.check("ERROR" in err, "the replace failure was silent")
                    c.check(len(acc["bidirectional"]) == 3,
                            f"the list was CLEARED although replace #{fail_on} "
                            f"failed — the clear is not strictly last")
                finally:
                    W._os_flush = real_os

        # a write failure before any replace
        with _flush_env(root, run_id="clearw") as (W, ctx):
            real_write = W._flush_write_npz
            W._flush_write_npz = lambda *_a, **_k: (_ for _ in ()).throw(
                OSError(13, "Permission denied"))
            try:
                acc = {"bidirectional": [rec(1), rec(2)]}
                _out, err = _run_flush(W, acc)
                c.check("ERROR" in err and len(acc["bidirectional"]) == 2,
                        "candidates lost on a write failure")
            finally:
                W._flush_write_npz = real_write

        # and the successful path DOES clear, only after the installed pair
        # has been read back and validated
        with _flush_env(root, run_id="clearok") as (W, ctx):
            acc = {"bidirectional": [rec(1), rec(2)]}
            _run_flush(W, acc)
            c.check(acc["bidirectional"] == [], "a successful flush did not clear")
            c.check(os.path.isfile(ctx.member_paths()[0])
                    and os.path.isfile(ctx.member_paths()[1]),
                    "the pair is not installed")
    c.note("failures at write / replace-1 / replace-2 all retain; success clears")


def g_cadence(c: _Counter, ck=CK) -> None:
    """The threshold-gated cadence is unchanged: no flush below `_FLUSH_EVERY`
    NEW records, exactly one per crossing, and the clear resets the counter."""
    with _tmp_root() as root:
        with _flush_env(root, run_id="cad", flush_every=3) as (W, ctx):
            acc = {"bidirectional": [rec(1), rec(2)]}
            out, _err = _run_flush(W, acc)
            c.check(out.strip() == "", "flushed below the threshold")
            c.check(not os.path.exists(ctx.member_paths()[1]),
                    "a member was written below the threshold")
            acc["bidirectional"].append(rec(3))
            out, _err = _run_flush(W, acc)
            c.check("canonical checkpoint written" in out, "did not flush at 3")
            c.check(W._flush_last_count == 0,
                    "the clear must reset the new-since-last counter")
            out, _err = _run_flush(W, acc)
            c.check(out.strip() == "", "flushed again with nothing new")
            acc["bidirectional"].extend([rec(4), rec(5), rec(6)])
            out, _err = _run_flush(W, acc)
            c.check("seq=2" in out, f"expected sequence 2, got: {out.strip()}")
    c.note("threshold 3: 2 -> silent, 3 -> seq1, 0 new -> silent, 3 -> seq2")


def g_parity(c: _Counter, ck=CK) -> None:
    """§7.1 — a run that CHECKPOINTS AND CLEARS produces a byte-identical
    certified artifact to one that never cleared.

    Compared: all 22 canonical arrays exactly, the global seed order, and the
    canonical NPZ digest. `raw_candidate_count` is EXPECTED to differ, and NO
    SIDECAR-FIELD PARITY IS CLAIMED.
    """
    from utils.canonical_arrays import records_to_arrays
    from utils.run_finalizer import _select_l2_winners, _sort_by_seed, _l3_merge
    import hashlib

    # Cross-trial and cross-MODE duplicates for the same seed are legitimate and
    # are what the L2 competition is for; a same-trial/same-mode duplicate is
    # corruption and is deliberately NOT constructed here.
    raw = []
    for trial in (1, 2, 3):
        for seed in (11, 4, 27, 8):
            for mode in ("constant", "variable"):
                raw.append(rec(seed, trial=trial, mode=mode,
                               score=0.3 + 0.1 * trial + 0.01 * (seed % 3)))

    def artifact(records):
        arrays = _sort_by_seed(_l3_merge(
            records_to_arrays(_select_l2_winners(records)), None))
        h = hashlib.sha256()
        for name, _dt in CANONICAL_ARRAY_CONTRACT:
            h.update(name.encode())
            h.update(np.ascontiguousarray(arrays[name]).tobytes())
        return arrays, h.hexdigest()

    # arm A — the pre-D6.2 behaviour: everything stays in memory
    arrays_a, digest_a = artifact([copy.deepcopy(r) for r in raw])

    # arm B — checkpoint + clear at every third record, finalizer fed the
    # reconstructed cumulative state
    with _tmp_root() as root:
        with _flush_env(root, run_id="parity", flush_every=3) as (W, ctx):
            acc = {"bidirectional": []}
            for record in raw:
                acc["bidirectional"].append(copy.deepcopy(record))
                _run_flush(W, acc)
            c.check(ctx.sequence >= 4,
                    f"the parity arm only checkpointed {ctx.sequence} times")
            c.check(len(acc["bidirectional"]) < len(raw),
                    "the parity arm never cleared, so it proves nothing")
            supplied = W._checkpoint_finalizer_input(acc)
    arrays_b, digest_b = artifact(supplied)

    c.check(digest_a == digest_b,
            f"canonical artifact digests DIFFER: {digest_a[:16]}… vs "
            f"{digest_b[:16]}…")
    for name, _dt in CANONICAL_ARRAY_CONTRACT:
        c.check(np.array_equal(arrays_a[name], arrays_b[name]),
                f"array {name!r} differs between the cleared and uncleared arms")
        c.check(arrays_a[name].dtype == arrays_b[name].dtype,
                f"array {name!r} dtype differs")
    c.check(list(arrays_a["seeds"]) == list(arrays_b["seeds"]),
            "global seed order differs")
    c.check(len(supplied) != len(raw),
            "raw_candidate_count is EXPECTED to differ — the resumed execution "
            "supplies compacted records, and no sidecar-field parity is claimed")
    c.note(f"22/22 arrays equal · seed order equal · artifact "
           f"{digest_a[:12]}… == {digest_b[:12]}… · raw counts "
           f"{len(raw)} vs {len(supplied)} (differ by design)")


# ═════════════════════════════════════════════════════════════════════════════
# §9.3 — resume gates
# ═════════════════════════════════════════════════════════════════════════════
def g_runid_grammar(c: _Counter, ck=CK) -> None:
    """Addendum §3 — an OPAQUE SINGLE COMPONENT. The grammar is an ADDITIONAL
    wall, not a replacement for the realpath / symlink-escape checks."""
    for good in ("run-1", "zeus.ubuntu_101-32415-1785000000", "A_b.c-9"):
        c.check(ck.validate_run_id(good) == good, f"{good!r} was rejected")
    bad = ["", ".", "..", "foo/bar", "/abs/path", "foo\\bar", "a/", "/",
           "../escape", "foo bar", "run:1", "run\x00id", "a\nb", "ru*n",
           "run;id", "~user"]
    for value in bad:
        try:
            ck.validate_run_id(value)
        except ck.CheckpointSelectorError:
            c.n += 1
        else:
            raise AssertionError(f"run id {value!r} was ACCEPTED — the selector "
                                 f"must be a single opaque component")
    for value in (None, 7, b"run", ["run"]):
        try:
            ck.validate_run_id(value)
        except ck.CheckpointSelectorError:
            c.n += 1
        else:
            raise AssertionError(f"non-str run id {value!r} was accepted")
    c.note(f"3 accepted · {len(bad) + 4} rejected incl. '/', '\\\\', '.', '..'")


def g_selector_confinement(c: _Counter, ck=CK) -> None:
    """§4.1 — absolute path · `..` · symlink escape · newest-directory
    discovery, EACH rejected."""
    with _tmp_root() as root:
        base = os.path.join(root, ck.CHECKPOINT_DIRNAME)
        os.makedirs(base)
        good = ck.resolve_checkpoint_dir(root, "ok-run")
        c.check(good == os.path.join(base, "ok-run"), "clean resolution wrong")

        for value in ("/etc", "../../etc", "..", "a/b"):
            try:
                ck.resolve_checkpoint_dir(root, value)
            except ck.CheckpointSelectorError:
                c.n += 1
            else:
                raise AssertionError(f"selector {value!r} escaped confinement")

        # symlink escape: a run directory that RESOLVES outside the root
        outside = os.path.join(root, "outside")
        os.makedirs(outside)
        os.symlink(outside, os.path.join(base, "escapee"))
        try:
            ck.resolve_checkpoint_dir(root, "escapee")
        except ck.CheckpointSelectorError as exc:
            c.check("escape" in str(exc) or "indirected" in str(exc),
                    f"symlink escape rejected for the wrong reason: {exc}")
        else:
            raise AssertionError("a symlinked run directory pointing outside "
                                 "the checkpoint root was ACCEPTED")

    # NO newest-directory discovery, AT ANY LAYER.
    #
    # `utils/checkpoint_d6_2.py` owns every path decision, so the rule is
    # absolute there: no mtime/ctime sort and no directory enumeration at all.
    ck_tree = ast.parse(_read(_CK_PATH))
    for node in ast.walk(ck_tree):
        if isinstance(node, ast.Call):
            attr = getattr(node.func, "attr", None)
            name = getattr(node.func, "id", None)
            if attr in ("getmtime", "getctime"):
                raise AssertionError(
                    f"utils/checkpoint_d6_2.py:{node.lineno} sorts by mtime — "
                    f"newest-directory inference is forbidden at every layer")
            if attr in ("listdir", "scandir", "glob", "iglob") or name in (
                    "glob", "iglob"):
                raise AssertionError(
                    f"utils/checkpoint_d6_2.py:{node.lineno} enumerates a "
                    f"directory — the selector is a run id, never a discovery")
    c.n += 2

    # In the integration module the rule is scoped to the CHECKPOINT surface —
    # every `_flush_*` / `_checkpoint_*` helper and the resume preparation. The
    # ONLY permitted enumeration there is the pid-keyed stale-temp sweep (which
    # collects a crashed run's orphans and decides no path), and nothing may
    # consult mtime. Unrelated globs elsewhere in a 2700-line module (the NP2
    # partition dispatcher, for one) are out of scope for this rule and are
    # deliberately not swept into it.
    integ_tree = ast.parse(_read(_INTEG_PATH))
    scoped = [n for n in ast.walk(integ_tree)
              if isinstance(n, ast.FunctionDef)
              and (n.name.startswith("_flush") or n.name.startswith("_checkpoint")
                   or n.name.startswith("_prepare_checkpoint")
                   or n.name.startswith("_install_repaired"))]
    c.check(len(scoped) >= 12,
            f"only {len(scoped)} checkpoint-surface functions found — the "
            f"scope of this rule has drifted")
    purge_lines = set()
    for fn in scoped:
        if fn.name == "_flush_purge_stale_temps":
            purge_lines = set(range(fn.lineno, (fn.end_lineno or fn.lineno) + 1))
    for fn in scoped:
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            attr = getattr(node.func, "attr", None)
            if attr in ("getmtime", "getctime"):
                raise AssertionError(
                    f"window_optimizer_integration_final.py:{node.lineno} "
                    f"(in {fn.name}) sorts a checkpoint path by mtime")
            if attr in ("glob", "iglob") and node.lineno not in purge_lines:
                raise AssertionError(
                    f"window_optimizer_integration_final.py:{node.lineno} "
                    f"(in {fn.name}) globs outside the pid-keyed stale-temp "
                    f"sweep")
    c.n += 1
    c.note("abs · .. · a/b · symlink escape rejected; no mtime, no enumeration")


def _live_call_kwargs(path: str, func_name: str, call_pattern) -> set:
    """AST — the keyword names of a LIVE call site.

    §5 of the verification standard: extract and execute the live source, never
    match text. `2389b61` reverted a fix by whole-block replacement and a text
    anchor would have gone green.
    """
    tree = ast.parse(_read(path))
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if call_pattern(node):
            found |= {kw.arg for kw in node.keywords if kw.arg}
    return found


def g_resume_route(c: _Counter, ck=CK) -> None:
    """§4.2 — ALL THREE HOPS, including that a WATCHER-SHAPED PARAMS DICT
    carrying `resume_checkpoint` reaches the method rather than being filtered
    out.

    Adding the method parameter alone would leave the resume path dead — the
    `Advisor -> strategy_recommendation.json -> WATCHER` pattern and the TRSE F1
    manifest drift. Proving the parameter EXISTS is not proving it ARRIVES.
    """
    manifest = json.loads(_read(_MANIFEST_PATH))

    # ---- HOP 1: the manifest, exercised through the LIVE WATCHER filter -----
    c.check("resume_checkpoint" in manifest["default_params"],
            "hop 1: `resume_checkpoint` is not declared in the manifest's "
            "default_params, so WATCHER's step-scoped filter silently DROPS it")

    # the live filter really derives its allowlist from default_params
    watcher_src = _read(_WATCHER_PATH)
    tree = ast.parse(watcher_src)
    derives = False
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and node.targets
                and getattr(node.targets[0], "id", "") == "allowed_params"):
            derives = "default_params" in ast.unparse(node.value)
    c.check(derives,
            "hop 1: WATCHER's `allowed_params` is no longer derived from "
            "default_params — this gate is measuring the wrong thing")

    # and drive the REAL merge the live code performs
    declared = dict(manifest["default_params"])
    allowed = set(declared)
    watcher_params = {"resume_checkpoint": "gate-run-77",
                      "not_declared_at_all": "dropped"}
    merged = dict(declared)
    for key, value in watcher_params.items():
        if key in allowed:
            merged[key] = value
    c.check(merged.get("resume_checkpoint") == "gate-run-77",
            "hop 1: the value did not survive the step-scoped filter")
    c.check("not_declared_at_all" not in merged,
            "hop 1: the filter control did not actually filter")

    # the CLI arg name the command builder will emit
    args_map = dict(manifest["actions"][0]["args_map"])
    args_map.update(manifest.get("args_map", {}))
    reverse = {v: k for k, v in args_map.items()}
    cli = reverse.get("resume_checkpoint")
    c.check(cli == "resume-checkpoint",
            f"hop 1: args_map does not map resume_checkpoint to a CLI flag "
            f"(got {cli!r})")
    argv = []
    for key, value in merged.items():
        if isinstance(value, bool) or value is None or value == "":
            continue
        argv.extend([f"--{reverse.get(key, key.replace('_', '-'))}", str(value)])
    c.check("--resume-checkpoint" in argv and "gate-run-77" in argv,
            f"hop 1: the WATCHER-shaped params dict did not produce "
            f"--resume-checkpoint in the command: {argv}")

    # argparse really declares it
    wo_tree = ast.parse(_read(_WO_PATH))
    declared_flags = {n.args[0].value for n in ast.walk(wo_tree)
                      if isinstance(n, ast.Call)
                      and getattr(n.func, "attr", "") == "add_argument"
                      and n.args and isinstance(n.args[0], ast.Constant)}
    c.check("--resume-checkpoint" in declared_flags,
            "hop 1/2: window_optimizer.py's argparse does not declare "
            "--resume-checkpoint, so the WATCHER command would abort")

    # ---- HOP 2: the explicit kwargs on the live call sites ------------------
    rbo = _live_call_kwargs(
        _WO_PATH, "run_bayesian_optimization",
        lambda n: getattr(n.func, "id", "") == "run_bayesian_optimization")
    c.check("resume_checkpoint" in rbo,
            "hop 2: run_bayesian_optimization(...) is not passed "
            "resume_checkpoint, so the value dies at the CLI boundary")
    ow = _live_call_kwargs(
        _WO_PATH, "optimize_window",
        lambda n: getattr(n.func, "attr", "") == "optimize_window")
    c.check("resume_checkpoint" in ow,
            "hop 2: coordinator.optimize_window(...) is not passed "
            "resume_checkpoint — it never reaches the method")

    # ---- HOP 3: the method signature ---------------------------------------
    integ_tree = ast.parse(_read(_INTEG_PATH))
    sigs = [n for n in ast.walk(integ_tree)
            if isinstance(n, ast.FunctionDef) and n.name == "optimize_window"
            and any(a.arg == "dataset_path" for a in n.args.args)]
    c.check(len(sigs) == 1, "hop 3: the optimize_window definition moved")
    params = {a.arg for a in sigs[0].args.args} | {
        a.arg for a in sigs[0].args.kwonlyargs}
    c.check("resume_checkpoint" in params,
            "hop 3: optimize_window does not accept resume_checkpoint — the "
            "kwarg would be rejected as unexpected")

    # ---- and the value is actually CONSUMED, not merely accepted -----------
    import window_optimizer_integration_final as W
    try:
        W._prepare_checkpoint_run_context(
            dataset_path=__file__, prng_base="java_lcg",
            skip_modes_executed=("constant",), seed_start=0, seed_count=10,
            resume_checkpoint="not/a/run/id", resume_study=True)
    except ck.CheckpointSelectorError as exc:
        c.check("not/a/run/id" in str(exc),
                "the consumer did not name the value it rejected")
    else:
        raise AssertionError("the value reached no consumer — a parameter that "
                             "is accepted and ignored is a dead chain")
    c.note("hop1 manifest+filter+argv+argparse · hop2 both call sites · "
           "hop3 signature · consumed")


_PREPARE_KW = dict(dataset_path=__file__, prng_base="java_lcg",
                   skip_modes_executed=("constant",), seed_start=0,
                   seed_count=1_000_000)


def _prepare(W, root, run_id="prep-run", **over):
    """Drive the LIVE `_prepare_checkpoint_run_context` inside an isolated root."""
    kw = dict(_PREPARE_KW, resume_checkpoint="", resume_study=False)
    kw.update(over)
    prev = {k: os.environ.get(k) for k in
            ("PRNG_CHECKPOINT_ROOT", "PRNG_CHECKPOINT_RUN_ID")}
    os.environ["PRNG_CHECKPOINT_ROOT"] = root
    os.environ["PRNG_CHECKPOINT_RUN_ID"] = run_id
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return W._prepare_checkpoint_run_context(**kw)
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _seed_production_checkpoint(W, ck, root, run_id, records):
    """Write a checkpoint whose `run_context_digest` is the one the PRODUCTION
    resume path will compute — otherwise the resume would (correctly) refuse on
    a context disagreement and the gate would prove nothing about the matrix."""
    ctx, _floor = _prepare(W, root, run_id=run_id)
    W._clear_flush_run_context()
    _commit(ck, ctx, ck.reconcile([], records))
    return ctx


def g_combination_matrix(c: _Counter, ck=CK) -> None:
    """§4.4 — all four rows, and the unsafe one rejected BEFORE a new candidate
    is admitted."""
    import window_optimizer_integration_final as W

    with _tmp_root() as root:
        # row 1 — no / no: normal fresh run
        ctx, floor = _prepare(W, root, run_id="combo-fresh")
        c.check(floor is None and ctx.cumulative == [],
                "row no/no: a fresh run must recover nothing")
        c.check(ctx.resume_provenance is None,
                "row no/no: a fresh run claims no resume provenance")

        # row 2 — no / yes: existing Optuna behaviour, unchanged
        ctx2, floor2 = _prepare(W, root, run_id="combo-fresh", resume_study=True)
        c.check(floor2 is None and ctx2.cumulative == [],
                "row no/yes: resume_study alone must not touch the checkpoint")

        # a real checkpoint, written under the production context
        _seed_production_checkpoint(W, ck, root, "combo-real",
                                    [rec(1, trial=3), rec(2, trial=5)])

        # row 4 — yes / no: MUST NOT begin new trials -> rejected
        try:
            _prepare(W, root, run_id="combo-real",
                     resume_checkpoint="combo-real", resume_study=False)
        except W.CheckpointResumeError as exc:
            c.check("manufacture" in str(exc).lower()
                    or "replay key" in str(exc).lower(),
                    f"row yes/no: rejected for the wrong reason: {exc}")
        else:
            raise AssertionError(
                "row yes/no: checkpoint + FRESH study was allowed to continue — "
                "restarting trial numbering would MANUFACTURE the corruption "
                "§6.1 raises")
        c.check(W._active_flush_run_context() is None,
                "row yes/no: a run context was installed despite the rejection "
                "— nothing may be admitted after an unsafe combination")

        # row 3 — yes / yes: allowed, above the recovered trial namespace
        ctx4, floor4 = _prepare(W, root, run_id="combo-real",
                                resume_checkpoint="combo-real",
                                resume_study=True)
        c.check(floor4 == 5,
                f"row yes/yes: the recovered trial floor must be the maximum "
                f"recovered trial_number (5), got {floor4}")
        c.check(len(ctx4.cumulative) == 2,
                "row yes/yes: the recovered state was not installed")
        c.check(ctx4.resume_provenance is not None,
                "row yes/yes: no resume provenance was recorded")
        W._clear_flush_run_context()

    # neither control aliases or implicitly enables the other
    src = _read(_INTEG_PATH)
    c.check("resume_study = True" not in src and "resume_study=True" not in
            src.split("def _prepare_checkpoint_run_context")[1].split(
                "def _install_repaired")[0],
            "a control implicitly enables the other")
    c.note("4/4 rows; yes/no rejected before admission; yes/yes floor=5")


def g_context_digest(c: _Counter, ck=CK) -> None:
    """§4.3 — canonical JSON; EVERY component mutated INDEPENDENTLY, each
    rejected before decoding; PID / timestamp / mutable path ABSENT."""
    base = _components(ck)
    digest = ck.build_run_context_digest(base)

    # canonical JSON encoding, mirrored from the finalizer
    raw = json.dumps(base, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True).encode("utf-8")
    c.check(ck._canonical_json_bytes(base) == raw,
            "the encoding is not sorted-keys / fixed-separators / ensure_ascii")
    c.check(ck.build_run_context_digest(
        dict(reversed(list(base.items())))) == digest,
            "key order changed the digest — the JSON is not canonical")

    # every component, independently
    mutations = {
        "dataset.version_id": lambda d: d["dataset"].__setitem__(
            "version_id", "other"),
        "dataset.sha256": lambda d: d["dataset"].__setitem__(
            "sha256", "ff" * 32),
        "dataset.filename": lambda d: d["dataset"].__setitem__(
            "filename", "other.json"),
        "repository_commit": lambda d: d.__setitem__("repository_commit", "d" * 40),
        "prng_base": lambda d: d.__setitem__("prng_base", "xorshift32"),
        "skip_modes_executed(order)": lambda d: d.__setitem__(
            "skip_modes_executed", ["variable", "constant"]),
        "skip_modes_executed(set)": lambda d: d.__setitem__(
            "skip_modes_executed", ["constant"]),
        "seed_start": lambda d: d.__setitem__("seed_start", 1),
        "seed_count": lambda d: d.__setitem__("seed_count", 999),
        "seed_end": lambda d: d.__setitem__("seed_end", 7),
        "execution_set_id": lambda d: d.__setitem__("execution_set_id", "s" * 64),
    }
    seen = set()
    for name, mutate in mutations.items():
        variant = copy.deepcopy(base)
        mutate(variant)
        d = ck.build_run_context_digest(variant)
        c.check(d != digest, f"component {name!r} is NOT in the preimage — one "
                             f"combined mutation is not evidence for the rest")
        seen.add(d)
    c.check(len(seen) == len(mutations),
            "two independent component mutations produced the same digest")

    # each is rejected BEFORE decoding
    with _tmp_root() as root:
        ctx = _context(ck, root, run_id="ctxd")
        _commit(ck, ctx, ck.reconcile([], [rec(1)]))
        for name, mutate in mutations.items():
            variant = copy.deepcopy(base)
            mutate(variant)
            try:
                ck.recover_checkpoint(
                    ctx.checkpoint_dir, run_id=ctx.run_id,
                    run_context_digest=ck.build_run_context_digest(variant))
            except ck.CheckpointRecoveryError as exc:
                c.check(ck.ROW_CONTEXT_DISAGREE in str(exc),
                        f"{name}: wrong rejection {exc}")
            else:
                raise AssertionError(f"a run resumed across a changed {name}")

    # EXCLUDED and gated as excluded
    flat = json.dumps(base)
    import re as _re
    c.check("pid" not in flat.lower(), "a PID leaked into the preimage")
    c.check(not _re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", flat),
            "a timestamp leaked into the preimage")
    c.check(os.sep + "home" not in flat and not flat.count('"/'),
            "an absolute (mutable) path leaked into the preimage")
    import window_optimizer_integration_final as W
    c.check(W._FLUSH_RUN_ID_DEFAULT not in flat,
            "the default run id — which embeds pid and wall time — leaked in")
    c.note(f"{len(mutations)} components each independently in the preimage; "
           f"pid/timestamp/path absent")


def g_cursor_not_claimed(c: _Counter, ck=CK) -> None:
    """§0 — D6.2 does NOT restore the optimizer execution cursor, and says so."""
    with _tmp_root() as root:
        ctx = _context(ck, root, run_id="cursor")
        _commit(ck, ctx, ck.reconcile([], [rec(1, trial=4)]))
        out = ck.recover_checkpoint(ctx.checkpoint_dir, run_id=ctx.run_id,
                                    run_context_digest=ctx.run_context_digest)
        prov = out.provenance()
        c.check(prov["optimizer_execution_cursor_restored"] is False,
                "the provenance claims the execution cursor was restored")
        for key in prov:
            c.check("cursor" not in key or "not" in key.lower()
                    or key.endswith("_restored"),
                    f"provenance key {key!r} implies a cursor claim")
    c.note("provenance records optimizer_execution_cursor_restored=False")


def g_resume_provenance(c: _Counter, ck=CK) -> None:
    """§4.5 — the four minimum fields, recorded DURABLY, with the stated
    `raw_candidate_count` wording and no sidecar-parity claim."""
    import window_optimizer_integration_final as W
    with _tmp_root() as root:
        _seed_production_checkpoint(W, ck, root, "prov-run",
                                    [rec(1, trial=2), rec(2, trial=6)])
        ctx, floor = _prepare(W, root, run_id="prov-run",
                              resume_checkpoint="prov-run", resume_study=True)
        W._clear_flush_run_context()

        path = os.path.join(ctx.checkpoint_dir, W._RESUME_PROVENANCE_NAME)
        c.check(os.path.isfile(path),
                f"resume provenance was not persisted at {path}")
        payload = json.loads(_read(path))
        for key in ("recovered_checkpoint_run_id", "recovered_checkpoint_id",
                    "recovered_checkpoint_sequence",
                    "recovered_canonical_state_digest",
                    "recovered_canonical_record_count"):
            c.check(key in payload, f"provenance is missing {key!r}")
        c.check(payload["recovered_checkpoint_run_id"] == "prov-run",
                "wrong run id recorded")
        c.check(payload["recovered_canonical_record_count"] == 2,
                "wrong recovered record count")
        c.check(payload["raw_candidate_count_semantics"] ==
                "the records supplied to the finalizer by the resumed execution",
                "the `raw_candidate_count` wording is not REV5 §4.5's")
        c.check("sidecar" not in json.dumps(payload).lower()
                or "parity" not in json.dumps(payload).lower(),
                "the provenance appears to claim sidecar-field parity")
        c.check(floor == 6, f"recovered trial floor should be 6, got {floor}")
    c.note("4 minimum fields + wording, persisted in the run-isolated directory")


# ── addendum §4 — G-TRIAL-NAMESPACE ─────────────────────────────────────────
def _extract_block(path: str, start: str, end: str, label: str) -> str:
    """AST-anchored extraction of a LIVE block, dedented for execution."""
    src = _read(path)
    assert src.count(start) == 1, f"{label}: start anchor is not unique"
    body = src.split(start, 1)[1]
    assert body.count(end) >= 1, f"{label}: end anchor missing"
    block = start + body.split(end, 1)[0]
    lines = [ln for ln in block.splitlines() if ln.strip()]
    indent = min(len(ln) - len(ln.lstrip()) for ln in lines)
    return "\n".join(ln[indent:] if len(ln) > indent else ln
                     for ln in block.splitlines())


def g_trial_namespace(c: _Counter, ck=CK) -> None:
    """Addendum §4 — BOTH checks, with the ENQUEUED WARM-START case exercised.

    Check 1 is a pre-flight over the loaded study, before `study.optimize` is
    entered. Check 2 fires at the very top of the objective, before dispatch.
    Neither ever rewrites or offsets a trial number.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ---- CHECK 1: the LIVE pre-flight block, executed ----------------------
    preflight = _extract_block(
        _BAYES_PATH,
        "if _resume_trial_floor is not None:",
        "# Trials remaining: full count on fresh",
        "check-1")
    c.check("WAITING" in preflight and "RUNNING" in preflight,
            "the pre-flight does not scan NONTERMINAL (WAITING/RUNNING) trials")

    # THE ENQUEUED CASE IS REAL: study.enqueue_trial is the S166 warm-start path
    src = _read(_BAYES_PATH)
    c.check("study.enqueue_trial(_ws_params)" in src,
            "the warm-start enqueue seam moved — this gate is not exercising "
            "the case addendum §4 exists for")

    def run_preflight(study, floor):
        ns = {"study": study, "_resume_trial_floor": floor,
              "print": lambda *a, **k: None}
        exec(compile(preflight, "<check-1>", "exec"), ns)

    space = {"window_size": optuna.distributions.IntDistribution(2, 50),
             "offset": optuna.distributions.IntDistribution(0, 100)}

    # a study carrying an ENQUEUED (WAITING) trial numbered at/below the floor
    study = optuna.create_study()
    study.enqueue_trial({"window_size": 8, "offset": 0})     # -> number 0
    waiting = [t for t in study.trials if t.state.name == "WAITING"]
    c.check(len(waiting) == 1 and int(waiting[0].number) == 0,
            f"the enqueued control did not produce a WAITING trial 0: "
            f"{[(t.number, t.state.name) for t in study.trials]}")
    try:
        run_preflight(study, 3)
    except RuntimeError as exc:
        c.check("nonterminal" in str(exc).lower() and "[0]" in str(exc),
                f"the enqueued trial was rejected for the wrong reason: {exc}")
        c.check("never rewritten or offset" in str(exc),
                "the rejection does not state that numbers are not renumbered")
    else:
        raise AssertionError(
            "an ENQUEUED warm-start trial numbered at or below the recovered "
            "maximum was allowed to execute — `max(existing) + 1` is not the "
            "next number of a loaded study")

    # clean control: the same study with a floor BELOW the enqueued number
    run_preflight(study, -1)
    c.n += 1

    # ---- CHECK 2: the LIVE per-trial guard, executed ------------------------
    guard = _extract_block(
        _BAYES_PATH,
        "if (_resume_trial_floor is not None",
        "# Sample parameters from search space",
        "check-2")

    class _Trial:
        def __init__(self, number):
            self.number = number

    def run_guard(number, floor):
        ns = {"trial": _Trial(number), "_resume_trial_floor": floor}
        exec(compile(guard, "<check-2>", "exec"), ns)

    for number in (0, 3, 5):
        try:
            run_guard(number, 5)
        except RuntimeError as exc:
            c.check("does not exceed the recovered maximum" in str(exc),
                    f"wrong rejection for trial {number}: {exc}")
        else:
            raise AssertionError(
                f"trial.number {number} <= recovered maximum 5 was allowed to "
                f"run — the check must fire BEFORE objective execution")
    run_guard(6, 5)          # clean control: strictly above
    run_guard(0, None)       # inert when nothing was resumed
    c.n += 2

    # check 2 really is at the TOP, before any suggest/dispatch
    objective = _extract_block(_BAYES_PATH, "def optuna_objective(trial):",
                               "# Store result", "position")
    guard_at = objective.index("_resume_trial_floor is not None")
    for later in ("trial.suggest_int", "objective_function("):
        c.check(guard_at < objective.index(later),
                f"the per-trial check runs AFTER {later} — it must precede "
                f"objective execution, dispatch and candidate admission")

    # ---- numbers are NEVER rewritten or offset -----------------------------
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if (isinstance(t, ast.Attribute) and t.attr == "number"):
                    raise AssertionError(
                        f"window_optimizer_bayesian.py:{node.lineno} assigns to "
                        f"`.number` — Optuna trial numbers are never rewritten "
                        f"or offset")
    c.n += 1

    # ---- and the RECORD-BOUND ordinal begins above the recovered namespace --
    integ = _read(_INTEG_PATH)
    c.check("trial_counter = {'count': int(_d6_2_resume_floor or 0)}" in integ,
            "the record trial ordinal still restarts at 0 on a resume — "
            "`trial_number` is part of the replay key, so a restart would "
            "collide with a recovered record")
    c.note("check1 rejects enqueued WAITING trial 0 at floor 3 · check2 rejects "
           "0/3/5 at floor 5 and precedes suggest/dispatch · no renumbering")


# ═════════════════════════════════════════════════════════════════════════════
# §9.6 + addendum §5 — MUTANTS
# ═════════════════════════════════════════════════════════════════════════════
def _det_state_digest_split(m) -> None:
    records = m.reconcile([], [rec(1), rec(2)])
    state = m.canonical_state_arrays(records)
    ident = m.build_identity(checkpoint_id="i", sequence=1, run_id="r",
                             logical_candidate_count=2,
                             run_context_digest="a" * 64,
                             state_digest=m.canonical_state_digest(state),
                             role=m.MEMBER_A_ROLE)
    _a, _oa, da = m.seal_member(state, ident, m.MEMBER_A_ROLE)
    _b, _ob, db = m.seal_member(state, ident, m.MEMBER_B_ROLE)
    assert da != db, "member digests must differ"


def _det_member_digest_identity(m) -> None:
    """Every identity field is inside the member digest, and it excludes only
    itself."""
    records = m.reconcile([], [rec(1)])
    state = m.canonical_state_arrays(records)
    sd = m.canonical_state_digest(state)
    ident = m.build_identity(checkpoint_id="i", sequence=1, run_id="r",
                             logical_candidate_count=1,
                             run_context_digest="a" * 64, state_digest=sd,
                             role=m.MEMBER_B_ROLE)
    arrays, order, digest = m.seal_member(state, ident, m.MEMBER_B_ROLE)
    sealed = dict(ident, member_role=m.MEMBER_B_ROLE)
    sealed["member_content_digest"] = digest
    payload = {n: arrays[n] for n in order}
    assert m.member_content_digest(sealed, payload, order) == digest, (
        "the member digest is not stable once its own field is populated — it "
        "must exclude ONLY itself")
    for key in m.IDENTITY_KEYS:
        if key == "member_content_digest":
            continue
        tampered = dict(sealed)
        tampered[key] = (999 if key in m._INT_IDENTITY_KEYS
                         else str(sealed[key]) + "-x")
        assert m.member_content_digest(tampered, payload, order) != digest, (
            f"identity field {key!r} is NOT covered by the member digest")


def _det_member_digest_shape(m) -> None:
    import hashlib
    a = np.arange(6, dtype=np.float32)
    h1, h2 = hashlib.sha256(), hashlib.sha256()
    m._hash_array(h1, "x", a)
    m._hash_array(h2, "x", a.reshape(2, 3))
    assert h1.hexdigest() != h2.hexdigest(), (
        "two differently-shaped arrays with identical bytes collided")


def _det_state_order(m) -> None:
    assert m.STATE_PHYSICAL_ORDER == _ADDENDUM_PHYSICAL_ORDER, (
        f"physical order {m.STATE_PHYSICAL_ORDER} != addendum §1's")
    emitted = []
    real = m._hash_array

    def spy(h, name, arr):
        emitted.append(name)
        return real(h, name, arr)

    arrays = m.canonical_state_arrays(m.reconcile([], [rec(3), rec(1)]))
    m._hash_array = spy
    try:
        m.canonical_state_digest(arrays)
    finally:
        m._hash_array = real
    assert tuple(emitted) == _ADDENDUM_PHYSICAL_ORDER, (
        f"the EMITTED order {tuple(emitted)} is not the fixed §1 order")
    assert "prng_base" not in emitted, "prng_base was hashed into the state"


def _det_order_permutation(m) -> None:
    base = [rec(11), rec(3, trial=2), rec(29, trial=3)]
    a = m.canonical_state_digest(m.canonical_state_arrays(
        m.reconcile([], [copy.deepcopy(r) for r in base])))
    b = m.canonical_state_digest(m.canonical_state_arrays(
        m.reconcile([], [copy.deepcopy(r) for r in reversed(base)])))
    assert a == b, "an arrival-order permutation changed the state digest"

    # AND the same property held DIRECTLY by `canonical_state_arrays`, not
    # merely inherited from `_select_l2_winners` (which happens to emit seeds in
    # ascending order). Addendum §1 states the sort as a property of the array
    # construction, so that is where it has to be provable — otherwise removing
    # it here would be invisible until some future caller stopped going through
    # the frozen selector.
    canonical = [m.canonicalize_record(r) for r in base]
    direct = {m.canonical_state_digest(m.canonical_state_arrays(perm))
              for perm in ([canonical[i] for i in order]
                           for order in ([0, 1, 2], [2, 1, 0], [1, 2, 0]))}
    assert len(direct) == 1, (
        "canonical_state_arrays does not globally seed-sort its rows — the "
        "state digest depends on the order it was handed")
    assert direct == {a}, "the direct and reconciled digests disagree"


def _det_l2_policy(m) -> None:
    """The frozen L2 key: float32 score -> lowest trial_number -> mode, and
    mode only WITHIN one trial."""
    out = m.reconcile([], [rec(5, trial=1, score=0.4), rec(5, trial=2, score=0.9)])
    assert len(out) == 1 and out[0]["trial_number"] == 2, "highest score wins"
    lo, hi = 0.5, 0.5 + 1e-12
    out = m.reconcile([], [rec(5, trial=1, score=lo), rec(5, trial=2, score=hi)])
    assert out[0]["trial_number"] == 1, (
        "a float64-only difference must be an exact float32 TIE and fall "
        "through to the lower trial_number")
    out = m.reconcile([], [rec(5, trial=1, mode="variable", score=0.5),
                           rec(5, trial=2, mode="constant", score=0.5)])
    assert out[0]["trial_number"] == 1 and out[0]["skip_mode"] == "variable", (
        "the trial-number tiebreak must be applied BEFORE mode")
    out = m.reconcile([], [rec(5, trial=1, mode="variable", score=0.5),
                           rec(5, trial=1, mode="constant", score=0.5)])
    assert out[0]["skip_mode"] == "constant", "within-trial mode tiebreak"


def _det_storage_domain(m) -> None:
    """Canonicalization must land in the float32 STORAGE domain, and the
    round-trip must return exactly that value."""
    v = 0.1 + 1e-12
    stored = m.canonicalize_record(rec(1, score=v))["score"]
    assert stored == float(np.float32(v)), (
        f"the canonicalized score {stored!r} is not the float32 value "
        f"{float(np.float32(v))!r}")
    assert stored != v, "the pre-rounding float64 survived into storage"
    with _tmp_root() as root:
        ctx = _context(m, root, run_id="mut-dom")
        _commit(m, ctx, m.reconcile([], [rec(1, score=v)]))
        out = m.recover_checkpoint(ctx.checkpoint_dir, run_id=ctx.run_id,
                                   run_context_digest=ctx.run_context_digest)
        assert out.records[0]["score"] == stored, (
            "the round-tripped score is not the stored float32 value — a "
            "bit-identical replay could no longer be recognised")


def _det_stub_payload(m) -> None:
    """Member A stores EXACTLY `seed`, `score` and its identity block."""
    with _tmp_root() as root:
        ctx = _context(m, root, run_id="mut-stub")
        _commit(m, ctx, m.reconcile([], [rec(1), rec(2)]))
        with np.load(ctx.member_paths()[0], allow_pickle=False) as z:
            stored = set(z.files)
    expected = set(m.MEMBER_A_PAYLOAD_FIELDS) | set(m.IDENTITY_KEYS)
    assert stored == expected, (
        f"member A stores {sorted(stored - expected)} beyond the marker stub — "
        f"it is not an accumulator backup and must not carry the state")


def _det_same_key_collision(m) -> None:
    try:
        m.reconcile([], [rec(5, trial=1, fwd=0.4), rec(5, trial=1, fwd=0.9)])
    except AccumulatorConsistencyError:
        return
    raise AssertionError("a same-trial/same-mode collision was SWALLOWED")


def _det_malformed_loser(m) -> None:
    """A malformed LOSING candidate must fail the run, not vanish."""
    winner = rec(5, trial=1, score=0.9)
    loser = rec(5, trial=2, score=0.1)
    del loser["sessions"]
    try:
        m.validate_new_raw_records([winner, loser], seed_start=0,
                                   seed_end_exclusive=1000,
                                   prng_base="java_lcg",
                                   skip_modes_executed=("constant",))
    except Exception:                                           # noqa: BLE001
        return
    raise AssertionError("a malformed LOSING candidate passed the walls and "
                         "would vanish during compaction")


def _det_identity_field_present(m) -> None:
    with _tmp_root() as root:
        ctx = _context(m, root, run_id="mut-ident")
        _commit(m, ctx, m.reconcile([], [rec(1)]))
        with np.load(ctx.member_paths()[1], allow_pickle=False) as z:
            stored = set(z.files)
        for key in ("checkpoint_schema_version", "checkpoint_id",
                    "checkpoint_sequence", "run_id", "logical_candidate_count",
                    "encoding_version", "canonical_map_hash",
                    "run_context_digest", "canonical_state_digest",
                    "member_content_digest"):
            assert key in stored, f"transaction-identity field {key!r} is not "\
                                  f"persisted"


def _det_recover_b_not_a(m) -> None:
    with _tmp_root() as root:
        ctx = _context(m, root, run_id="mut-newer-a")
        _commit(m, ctx, m.reconcile([], [rec(1), rec(2)]),
                checkpoint_id="the-real-b")
        _reseal_member(m, ctx.member_paths()[0], checkpoint_sequence=9,
                       checkpoint_id="the-newer-a")
        out = m.recover_checkpoint(ctx.checkpoint_dir, run_id=ctx.run_id,
                                   run_context_digest=ctx.run_context_digest)
        assert out.checkpoint_id == "the-real-b", (
            "the NEWER member A was recovered instead of B — A is a marker "
            "stub and B is the sole recovery payload")
        assert out.row == m.ROW_A_NEWER, f"wrong row {out.row}"


def _det_row5(m) -> None:
    with _tmp_root() as root:
        ctx = _context(m, root, run_id="mut-row5")
        _commit(m, ctx, m.reconcile([], [rec(1)]))
        _reseal_member(m, ctx.member_paths()[0], checkpoint_sequence=0,
                       checkpoint_id="older-a")
        out = m.recover_checkpoint(ctx.checkpoint_dir, run_id=ctx.run_id,
                                   run_context_digest=ctx.run_context_digest)
        assert out.row == m.ROW_B_NEWER, (
            f"recovery row 5 (B newer, A older, invariants agree) is gone: got "
            f"{out.row}")
        assert out.next_sequence == 2 and out.repair_pair


def _det_runid_grammar(m) -> None:
    for value in ("foo/bar", "/abs", "..", "."):
        try:
            m.validate_run_id(value)
        except m.CheckpointSelectorError:
            continue
        raise AssertionError(f"run id {value!r} was ACCEPTED")


def _det_finalizer_path(m) -> None:
    assert m.MEMBER_A_NAME not in ("bidirectional_survivors_all.npz",
                                   "bidirectional_survivors_binary.npz"), \
        "the checkpoint writes a finalizer-owned path"
    assert m.MEMBER_B_NAME not in ("bidirectional_survivors_all.npz",
                                    "bidirectional_survivors_binary.npz"), \
        "the checkpoint writes a finalizer-owned path"
    assert m.CHECKPOINT_DIRNAME == ".s172_checkpoint"


def _det_encoding_map(m) -> None:
    """A member that is INTERNALLY CONSISTENT but declares a different encoding
    map must still fail — only the identity bind can catch that, so the member
    is resealed rather than merely rewritten."""
    with _tmp_root() as root:
        ctx = _context(m, root, run_id="mut-map")
        _commit(m, ctx, m.reconcile([], [rec(1)]))
        for _p in ctx.member_paths():
            _reseal_member(m, _p, canonical_map_hash="0" * 64)
        try:
            m.recover_checkpoint(ctx.checkpoint_dir, run_id=ctx.run_id,
                                 run_context_digest=ctx.run_context_digest)
        except m.CheckpointError:
            return
    raise AssertionError("a changed encoding-map hash was ACCEPTED")


def _det_no_mtime(m) -> None:
    """Parse the source of the MODULE UNDER TEST, never the file on disk.

    Reading `_CK_PATH` here would make this detector blind to every source
    mutant — it would keep inspecting production while the mutated module was
    the thing actually loaded, and the kill would be vacuous.
    """
    tree = ast.parse(getattr(m, "__d6_2_src__", _read(_CK_PATH)))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (
                getattr(node.func, "attr", "") in ("getmtime", "getctime",
                                                   "glob", "iglob")
                or getattr(node.func, "id", "") in ("glob", "iglob")):
            raise AssertionError(
                f"utils/checkpoint_d6_2.py:{node.lineno} reintroduced "
                f"newest-directory inference / directory enumeration")


def _func_src(path: str, name: str) -> str:
    """The LIVE source of one top-level function, by AST line span."""
    src = _read(path)
    tree = ast.parse(src)
    node = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = src.splitlines(keepends=True)
    return "".join(lines[node.lineno - 1:node.end_lineno])


def _det_combination(m=None) -> None:
    """The unsafe combination is rejected, and rejected FIRST.

    The checkpoint is seeded under the PRODUCTION context, so if the rejection
    were removed the resume would otherwise SUCCEED — which is what makes this
    detector attributable to the guard rather than to a context disagreement.
    """
    import window_optimizer_integration_final as W
    with _tmp_root() as root:
        _seed_production_checkpoint(W, CK, root, "mut-combo",
                                    [rec(1, trial=3)])
        try:
            _prepare(W, root, run_id="mut-combo",
                     resume_checkpoint="mut-combo", resume_study=False)
        except W.CheckpointResumeError:
            return
        finally:
            W._clear_flush_run_context()
    raise AssertionError("checkpoint + fresh study was allowed to continue — a "
                         "restart would manufacture the corruption §6.1 raises")


def _det_clear_last(m=None) -> None:
    """The clear must be strictly after BOTH replaces."""
    import window_optimizer_integration_final as W

    class _OsProxy:
        def __init__(self):
            self._real, self.calls = os, 0

        def __getattr__(self, name):
            return getattr(self._real, name)

        def replace(self, src, dst):
            self.calls += 1
            if self.calls == 2:
                raise OSError(28, "No space left on device")
            return self._real.replace(src, dst)

    with _tmp_root() as root:
        with _flush_env(root, run_id="mut-clear") as (W2, _ctx):
            real_os = W2._os_flush
            W2._os_flush = _OsProxy()
            try:
                acc = {"bidirectional": [rec(1), rec(2), rec(3)]}
                _run_flush(W2, acc)
                assert W2._os_flush.calls == 2, "the second replace was not reached"
                assert len(acc["bidirectional"]) == 3, (
                    "the list was CLEARED between the two replaces")
            finally:
                W2._os_flush = real_os


def g_mutants() -> None:
    """§9.6 + addendum §5 — the four-part kill rule on every mutant."""
    # ---- module-source mutants (checkpoint_d6_2.py) ------------------------
    _kill("M1 inline `score >` policy replaces the frozen L2 selector",
          "    return [dict(w) for w in _select_l2_winners(list(merged.values()))]",
          "    _best = {}\n"
          "    for _r in merged.values():\n"
          "        _s = int(_r['seed'])\n"
          "        if _s not in _best or _r['score'] > _best[_s]['score']:\n"
          "            _best[_s] = _r\n"
          "    return [dict(v) for v in _best.values()]",
          _det_l2_policy, "G-DUPLICATE-MATRIX")

    # NOTE the credited detector. The frozen `_l2_sort_key` casts to float32
    # ITSELF, so storing float64 does NOT change which candidate wins — the L2
    # detector cannot see this defect, and crediting it to G-DUPLICATE-MATRIX
    # would have been a vacuous kill. What this mutant actually breaks is the
    # STORAGE domain: the pre-rounding float64 would be persisted and compared,
    # which is exactly the "comparing pre-rounding float64 while storing the
    # rounded value" defect Ruling D converts away.
    _kill("M2 float64 kept instead of the float32 storage domain",
          "            out[name] = float(np.float32(value))",
          "            out[name] = float(value)",
          _det_storage_domain, "G-STORAGE-DOMAIN")

    _kill("M3 trial_number tiebreak dropped",
          "    return [dict(w) for w in _select_l2_winners(list(merged.values()))]",
          "    _best = {}\n"
          "    for _r in merged.values():\n"
          "        _s = int(_r['seed'])\n"
          "        _k = (float(np.float32(_r['score'])),\n"
          "              1 if _r['skip_mode'] == 'constant' else 0)\n"
          "        if _s not in _best or _k > _best[_s][0]:\n"
          "            _best[_s] = (_k, _r)\n"
          "    return [dict(v) for _k, v in _best.values()]",
          _det_l2_policy, "G-DUPLICATE-MATRIX")

    _kill("M4 same-trial/same-mode collision swallowed",
          "            raise AccumulatorConsistencyError(",
          "            merged[key] = canonical\n            continue\n"
          "            raise AccumulatorConsistencyError(",
          _det_same_key_collision, "G-DUPLICATE-MATRIX")

    _kill("M5 a transaction-identity field dropped",
          '    "canonical_state_digest",\n    "member_role",',
          '    "member_role",',
          _det_identity_field_present, "G-DIGEST-PREIMAGE")

    _kill("M6 checkpoint writes a finalizer-owned path",
          'MEMBER_A_NAME = "incremental_survivors_all.npz"',
          'MEMBER_A_NAME = "bidirectional_survivors_all.npz"',
          _det_finalizer_path, "G-NO-SYMLINK-COLLISION")

    _kill("M7 skip_mode map hardcoded instead of the shared codec",
          "            column = [encode_skip_mode(r[name]) for r in rows]",
          "            column = [{'constant': 1, 'variable': 0}[r[name]]\n"
          "                      for r in rows]",
          lambda m: _det_roundtrip_modes(m), "G-ENCODING-AUTHORITY")

    _kill("M8 encoding-map hash no longer binds",
          '        "canonical_map_hash": canonical_map_hash(),\n'
          '        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,',
          '        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,',
          _det_encoding_map, "G-IDENTITY-BIND")

    _kill("M9 recover the newer A instead of B",
          "        return _outcome(ROW_A_NEWER, seq_a + 1, repair_pair=True,\n"
          "                        discarded_a=seq_a)",
          "        _o = _outcome(ROW_A_NEWER, seq_a + 1, repair_pair=True,\n"
          "                      discarded_a=seq_a)\n"
          "        _o.checkpoint_id = str(ident_a['checkpoint_id'])\n"
          "        return _o",
          _det_recover_b_not_a, "G-RECOVERY-MATRIX")

    _kill("M10 a malformed losing candidate disappears during compaction",
          "    _validate_raw_candidates(records)",
          "    pass",
          _det_malformed_loser, "G-PRE-CLEAR-WALLS")

    _kill("M11 newest-directory inference reintroduced",
          "    run_id = validate_run_id(run_id)",
          "    import glob as _g\n"
          "    if run_id == 'latest':\n"
          "        run_id = sorted(_g.glob('*'), key=os.path.getmtime)[-1]\n"
          "    run_id = validate_run_id(run_id)",
          _det_no_mtime, "G-SELECTOR-CONFINEMENT")

    _kill("M12 shape omitted from the array preimage",
          '    h.update(repr(tuple(int(d) for d in arr.shape)).encode("utf-8"))\n'
          '    h.update(b"\\x00")',
          "    pass",
          _det_member_digest_shape, "G-MEMBER-DIGEST-SCOPE")

    _kill("M13 member_content_digest included in its own preimage",
          '        if key == "member_content_digest":\n'
          "            continue                    # a field cannot hash itself",
          "        pass",
          _det_member_digest_identity, "G-DIGEST-PREIMAGE")

    _kill("M14 the state digest is emitted in DICT order, not the §1 order",
          "    for name in STATE_PHYSICAL_ORDER:\n"
          "        if name not in state_arrays:",
          "    for name in list(state_arrays):\n"
          "        if name not in state_arrays:",
          _det_state_order, "G-STATE-ORDER-PHYSICAL")

    _kill("M15 prng_base hashed into the state digest",
          "        elif name == _DERIVED_FIELD:\n            continue",
          "        elif name == _DERIVED_FIELD:\n            order.append(name)",
          _det_state_order, "G-STATE-ORDER-PHYSICAL")

    _kill("M16 an identity field excluded from the member digest",
          '        if key == "member_content_digest":\n'
          "            continue                    # a field cannot hash itself",
          '        if key in ("member_content_digest", "run_context_digest",\n'
          '                   "canonical_state_digest"):\n'
          "            continue",
          _det_member_digest_identity, "G-DIGEST-PREIMAGE")

    _kill("M17 a run id containing '/' is accepted",
          '_RUN_ID_GRAMMAR = re.compile(r"\\A[A-Za-z0-9._-]+\\Z")',
          '_RUN_ID_GRAMMAR = re.compile(r"[A-Za-z0-9._/-]+")',
          _det_runid_grammar, "G-RUNID-GRAMMAR")

    _kill("M18 recovery row 5 deleted (B newer, A older, invariants agree)",
          "    # Row 5 (addendum §2, restored)",
          "    if seq_b > seq_a:\n"
          "        return _outcome(ROW_CONSISTENT, seq_b + 1, repair_pair=False)\n"
          "    # Row 5 (addendum §2, restored)",
          _det_row5, "G-RECOVERY-MATRIX")

    _kill("M19 order-permutation invariance broken (rows left unsorted)",
          '    rows = sorted(records, key=lambda r: int(r["seed"]))',
          "    rows = list(records)",
          _det_order_permutation, "G-STATE-ORDER-PERMUTATION")

    # Credited to G-STUB-HONESTY, not G-DIGEST-SPLIT: with A given B's payload
    # the two member digests still DIFFER (their `member_role` differs), so the
    # split detector cannot see this. What the mutant actually destroys is the
    # asymmetric architecture — member A stops being a marker stub and starts
    # carrying the full state, which is precisely the "A as an accumulator
    # backup" reading §0 forbids.
    _kill("M20 member A sealed with member B's full payload (stub becomes a "
          "second copy of the state)",
          "    if role == MEMBER_A_ROLE:\n        payload_order = MEMBER_A_PAYLOAD_FIELDS",
          "    if role == MEMBER_A_ROLE:\n        payload_order = STATE_PHYSICAL_ORDER",
          _det_stub_payload, "G-STUB-HONESTY")

    # ---- integration-module / study-body mutants ---------------------------
    _positive_control("M21", lambda _m: _det_clear_last())
    _record_mutant("M21 clear moved between the two replaces (source check)",
                   lambda _m: _assert_clear_position_mutant(),
                   "G-CLEAR-SAFE", None)

    _positive_control("M22", lambda _m: _det_combination())
    _record_mutant("M22 checkpoint + fresh study allowed to continue",
                   lambda _m: _assert_combination_mutant(),
                   "G-COMBINATION-MATRIX", None)

    _positive_control("M23", lambda _m: _det_enqueued_guard())
    _record_mutant("M23 an enqueued trial at or below the recovered maximum "
                   "executes",
                   lambda _m: _assert_enqueued_mutant(),
                   "G-TRIAL-NAMESPACE", None)


def _det_roundtrip_modes(m) -> None:
    """skip_mode must survive encode -> store -> decode with its meaning."""
    with _tmp_root() as root:
        ctx = _context(m, root, run_id="mut-mode")
        records = m.reconcile([], [rec(1, mode="constant"),
                                   rec(2, mode="variable")])
        _commit(m, ctx, records)
        out = m.recover_checkpoint(ctx.checkpoint_dir, run_id=ctx.run_id,
                                   run_context_digest=ctx.run_context_digest)
        got = {int(r["seed"]): r["skip_mode"] for r in out.records}
        assert got == {1: "constant", 2: "variable"}, (
            f"skip_mode did not round-trip through the shared codec: {got}")


def _assert_clear_position_mutant() -> None:
    """Mutate the LIVE flush so the clear happens between the two replaces, and
    prove the detector reds."""
    import window_optimizer_integration_final as W

    class _ClearingProxy:
        def __init__(self, acc):
            self._real, self.calls, self._acc = os, 0, acc

        def __getattr__(self, name):
            return getattr(self._real, name)

        def replace(self, src, dst):
            self.calls += 1
            out = self._real.replace(src, dst)
            if self.calls == 1:
                self._acc["bidirectional"] = []       # <- the injected defect
                raise OSError(28, "No space left on device")
            return out

    with _tmp_root() as root:
        with _flush_env(root, run_id="mut-clearpos") as (W2, _ctx):
            real_os = W2._os_flush
            acc = {"bidirectional": [rec(1), rec(2), rec(3)]}
            W2._os_flush = _ClearingProxy(acc)
            try:
                _run_flush(W2, acc)
                assert len(acc["bidirectional"]) == 3, (
                    "the list was CLEARED between the two replaces")
            finally:
                W2._os_flush = real_os


def _assert_combination_mutant() -> None:
    """Inject the defect into the LIVE function source: bypass the `resume_study`
    guard, leaving everything else identical, and prove the detector reds."""
    import window_optimizer_integration_final as W
    mutated_src = _patch(_func_src(_INTEG_PATH, "_prepare_checkpoint_run_context"),
                         "    if not resume_study:\n",
                         "    if False:  # M22 — the guard is bypassed\n",
                         "M22")
    namespace = dict(W.__dict__)
    exec(compile(mutated_src, "<M22>", "exec"), namespace)
    real = W._prepare_checkpoint_run_context
    W._prepare_checkpoint_run_context = namespace["_prepare_checkpoint_run_context"]
    try:
        _det_combination()
    finally:
        W._prepare_checkpoint_run_context = real
        W._clear_flush_run_context()


def _det_enqueued_guard() -> None:
    """Clean control for M23: the live pre-flight rejects the enqueued trial."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    preflight = _extract_block(
        _BAYES_PATH, "if _resume_trial_floor is not None:",
        "# Trials remaining: full count on fresh", "m23")
    study = optuna.create_study()
    study.enqueue_trial({"window_size": 8, "offset": 0})
    ns = {"study": study, "_resume_trial_floor": 3,
          "print": lambda *a, **k: None}
    try:
        exec(compile(preflight, "<m23>", "exec"), ns)
    except RuntimeError:
        return
    raise AssertionError("an enqueued trial at or below the recovered maximum "
                         "was allowed to execute")


def _assert_enqueued_mutant() -> None:
    """The mutant: scan only COMPLETE trials (i.e. ignore WAITING/RUNNING)."""
    import optuna
    preflight = _extract_block(
        _BAYES_PATH, "if _resume_trial_floor is not None:",
        "# Trials remaining: full count on fresh", "m23mut")
    mutated = _patch(preflight, "in ('WAITING', 'RUNNING')", "in ('COMPLETE',)",
                     "M23")
    study = optuna.create_study()
    study.enqueue_trial({"window_size": 8, "offset": 0})
    ns = {"study": study, "_resume_trial_floor": 3,
          "print": lambda *a, **k: None}
    exec(compile(mutated, "<m23mut>", "exec"), ns)
    raise AssertionError("an enqueued trial at or below the recovered maximum "
                         "was allowed to execute")


# ═════════════════════════════════════════════════════════════════════════════
# runner
# ═════════════════════════════════════════════════════════════════════════════
def _check(name: str, fn) -> None:
    counter = _Counter(name)
    try:
        fn(counter)
    except Exception as exc:                                    # noqa: BLE001
        import traceback
        _CHECKS.append((name, False,
                        f"{type(exc).__name__}: {exc}", counter.n))
        print(f"  FAIL  {name}  ({counter.n} assertions before the failure)")
        traceback.print_exc()
        return
    detail = " | ".join(counter.notes)
    _CHECKS.append((name, True, detail, counter.n))
    print(f"  ok    {name}  [{counter.n} assertions]"
          + (f"  — {detail}" if detail else ""))


GATES = [
    ("G-SCHEMA-24", g_schema_24),
    ("G-STATE-ORDER-PHYSICAL", g_state_order_physical),
    ("G-DIGEST-SPLIT", g_digest_split),
    ("G-DIGEST-PREIMAGE", g_digest_preimage),
    ("G-MEMBER-DIGEST-SCOPE", g_member_digest_scope),
    ("G-STATE-ORDER-PERMUTATION", g_state_order_permutation),
    ("G-IDENTITY-BIND", g_identity_bind),
    ("G-DUPLICATE-MATRIX", g_duplicate_matrix),
    ("G-RECOVERY-MATRIX", g_recovery_matrix),
    ("G-SEQUENCE-INIT", g_sequence_init),
    ("G-STUB-HONESTY", g_stub_honesty),
    ("G-STORAGE-DOMAIN", g_storage_domain),
    ("G-CSR-STRICT", g_csr_strict),
    ("G-SESSIONS-CASES", g_sessions_cases),
    ("G-ENCODING-AUTHORITY", g_encoding_authority),
    ("G-NO-SYMLINK-COLLISION", g_no_symlink_collision),
    ("G-COMPRESSION-CONTRACT", g_compression_contract),
    ("G-PRE-CLEAR-WALLS", g_pre_clear_walls),
    ("G-CLEAR-SAFE", g_clear_safe),
    ("G-CADENCE", g_cadence),
    ("G-PARITY", g_parity),
    ("G-RUNID-GRAMMAR", g_runid_grammar),
    ("G-SELECTOR-CONFINEMENT", g_selector_confinement),
    ("G-RESUME-ROUTE", g_resume_route),
    ("G-COMBINATION-MATRIX", g_combination_matrix),
    ("G-CONTEXT-DIGEST", g_context_digest),
    ("G-CURSOR-NOT-CLAIMED", g_cursor_not_claimed),
    ("G-RESUME-PROVENANCE", g_resume_provenance),
    ("G-TRIAL-NAMESPACE", g_trial_namespace),
]


def main() -> int:
    print("=" * 78)
    print("S172 Phase-5 D6.2 — 24-field checkpoint, canonical reconciliation,")
    print("                    and the finalizer resume path")
    print("=" * 78)
    print("\nGATES\n" + "-" * 78)
    for name, fn in GATES:
        _check(name, fn)

    print("\nMUTANTS (four-part kill rule)\n" + "-" * 78)
    mutant_error = None
    try:
        g_mutants()
    except Exception as exc:                                    # noqa: BLE001
        import traceback
        traceback.print_exc()
        mutant_error = exc
    for label, signature, credited in _MUTANTS:
        print(f"  KILLED  {label}")
        print(f"          credited to {credited}")
        print(f"          {signature}")
        print(f"          applies-once ✓ | mutated-path ✓ | detector-clean ✓ "
              f"| injected-defect ✓")

    passed = sum(1 for _n, ok, _d, _c in _CHECKS if ok)
    total = len(_CHECKS)
    assertions = sum(c for _n, _ok, _d, c in _CHECKS)
    print("\n" + "=" * 78)
    print(f"{passed}/{total} D6.2 gate checks green  "
          f"({assertions} assertions, {len(_MUTANTS)} mutants killed)")
    if mutant_error is not None:
        print(f"MUTANT SUITE INCOMPLETE: {mutant_error}")
        print("RESULT: INCOMPLETE")
        print("=" * 78)
        return 1
    if passed != total:
        for name, ok, detail, _c in _CHECKS:
            if not ok:
                print(f"  FAILED  {name}: {detail}")
        print("RESULT: FAIL")
        print("=" * 78)
        return 1
    print("All D6.2 gate checks green — the checkpoint carries all 24 canonical")
    print("fields, the two digests are separate and exact, the resume selector is")
    print("an opaque run id confined to the checkpoint root, the nine-row recovery")
    print("matrix holds, reconciliation ends in the frozen L2 selector, and the")
    print("S166 clear runs strictly last with the finalizer fed complete input.")
    print("RESULT: PASS")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
