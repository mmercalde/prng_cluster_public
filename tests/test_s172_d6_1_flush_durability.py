#!/usr/bin/env python3
"""
S172 Phase-5 D6.1 — incremental NPZ atomic flush and durability repair.

Beta's framing is the operative one: INCREMENTAL DURABILITY DID NOT EXIST. The
S152 helper has failed on every single invocation since it was written, and a
broad `except Exception` turned each failure into an ignorable stdout warning.

Five defects are covered here — the four the D6.1 brief names, plus one the
brief did not anticipate and which made the briefed repair unsafe:

  D1  the `.npz` suffix bug, present twice: the temp NAME lacked `.npz`, so
      `np.savez_compressed` appended one and wrote `...flush.tmp.npz`, and the
      following `os.replace("...flush.tmp", ...)` raised FileNotFoundError.
  D2  the broad `except Exception` swallowed every failure into one warning.
  D3  the in-memory list-clear sat inside the same `try` after both replaces.
      It was only ACCIDENTALLY protective (the exception fired first); fixing
      D1 makes that ordering load-bearing, so it is gated, not assumed.
  D4  the S166 comment asserted "data is safe in NPZ" — a guarantee that has
      never once held.
  D5* PATH COLLISION (found in D6.1, not in the brief). The helper wrote to
      `bidirectional_survivors_all.npz` / `..._binary.npz` in the run root.
      Since D3.5 those two names are COMPATIBILITY SYMLINKS OWNED BY THE
      FINALIZER (`run_finalizer._bootstrap_root_aliases`), which FAILS CLOSED
      if a regular file appears at either path. Repairing D1 alone would have
      replaced both symlinks with regular 4-array files and made every
      subsequent `finalize_run` raise PublicationError — permanently breaking
      generation publication. G-NO-ALIAS-COLLISION pins the repair.

The checkpoint therefore lives in its own namespace.

[S172 Phase-5 D6.2] PORTED, NOT WEAKENED — and this suite is no longer the
authority on the payload.
--------------------------------------------------------------------------
D6.1 deliberately left the S166 in-memory clear DISABLED and stamped the
checkpoint `s172-d6.1-four-field-v1`, because four arrays cannot reconstruct the
24 CANONICAL_RECORD_FIELDS the D3.5 finalizer consumes. D6.2 supplies those 24
fields, so BOTH of those facts are now deliberately false:

  * `_CHECKPOINT_SCHEMA_VERSION` is D6.2's marker — REV5 §3.3 requires the
    four-field marker to CHANGE;
  * `_FLUSH_CLEAR_IN_MEMORY` is True — REV5 §8;
  * the four-field content digest and `_flush_inspect_pair` no longer exist:
    D6.2 splits the identity into `canonical_state_digest` (shared, content
    only) and `member_content_digest` (per member), and classification moved to
    `utils.checkpoint_d6_2.recover_checkpoint`'s nine-row matrix.

Every assertion that pinned one of those three has been RE-POINTED at the
replacement property, and nothing else in this suite has been relaxed. The
DURABILITY properties D6.1 exists for — the `.npz` suffix defeat, per-file
atomic replacement, temp cleanup on every path, the pid-keyed stale-temp purge,
path conditions, alias isolation, failure visibility, cadence and the
clear-strictly-last ordering — are all still gated here, against the D6.2 flush.
Payload correctness (the 24 fields, both digests, CSR sessions, reconciliation,
resume and the recovery matrix) belongs to
`tests/test_s172_d6_2_checkpoint_reconciliation.py`.

Oracles are hand-transcribed. Every gate must FAIL on wrong behaviour, proven
by the mutants under the four-part kill rule.

Run:  python3 -u tests/test_s172_d6_1_flush_durability.py
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_INTEG_PATH = "window_optimizer_integration_final.py"
_INTEG_FULL = os.path.join(_ROOT, _INTEG_PATH)

#: [D6.2] The flush's log prefix moved with its payload.
_TAG = "[S172-D6.2-CHECKPOINT]"

_CHECKS: list[tuple[str, bool, str]] = []
_MUTANTS: list[tuple[str, str, str, str]] = []


# ═════════════════════════════════════════════════════════════════════════════
# THE FLUSH SECTION, SLICED — the unit under test
# ═════════════════════════════════════════════════════════════════════════════
# The S152/D6.1 flush block is self-contained: it imports its own `os`, `numpy`
# and `sys` aliases and references nothing else in the 2000-line integration
# module. Slicing it lets every gate and every mutant load a real, executable
# copy in milliseconds without dragging in torch/optuna — and makes "the
# mutated path executes" trivially provable.
_SLICE_START = "import os as _os_flush"
_SLICE_END = "# END [S152] incremental flush helper"


def _flush_section_src(src: str | None = None) -> str:
    src = _read(_INTEG_FULL) if src is None else src
    assert src.count(_SLICE_START) == 1, (
        f"slice start anchor {_SLICE_START!r} is not unique — the flush "
        f"section boundary has drifted")
    assert src.count(_SLICE_END) == 1, (
        f"slice end anchor {_SLICE_END!r} is not unique — the flush section "
        f"boundary has drifted")
    body = src.split(_SLICE_START, 1)[1].split(_SLICE_END, 1)[0]
    section = _SLICE_START + body
    # drift detector: the slice must really contain the unit under test
    for required in ("def _flush_npz_incremental(", "def _flush_write_npz(",
                     "def _flush_tmp_name(", "def _flush_remove_temps(",
                     "_FLUSH_CLEAR_IN_MEMORY", "_CHECKPOINT_DIRNAME"):
        assert required in section, (
            f"the sliced flush section is missing {required!r} — the slice is "
            f"not the unit this suite believes it is testing")
    return section


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


_MUT_DIR: str | None = None
_MUT_SEQ = 0


def _mut_dir() -> str:
    global _MUT_DIR
    if _MUT_DIR is None:
        _MUT_DIR = tempfile.mkdtemp(prefix="d6_1_mutants_")
        sys.path.insert(0, _MUT_DIR)
    return _MUT_DIR


def _load_section(section_src: str, label: str):
    """Execute a flush-section source as a standalone module."""
    global _MUT_SEQ
    _MUT_SEQ += 1
    name = f"_d6_1_flush_{_MUT_SEQ}"
    path = os.path.join(_mut_dir(), f"{name}.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(section_src)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.__d6_1_label__ = label
    module.__d6_1_src__ = section_src
    return module


# The source every `_fresh()` builds from. Normally the production section; a
# mutant run swaps it, so EVERY module a gate constructs internally is the
# mutated one. Without this a gate that calls `_fresh()` would silently keep
# testing production and the mutant would survive vacuously.
_ACTIVE_SRC: str | None = None


@contextlib.contextmanager
def _active(src: str):
    global _ACTIVE_SRC
    prev = _ACTIVE_SRC
    _ACTIVE_SRC = src
    try:
        yield
    finally:
        _ACTIVE_SRC = prev


#: [D6.2] The declared seed interval every gate's candidates must fall inside.
#: `_validate_candidate_coverage` is one of the three walls the flush now runs,
#: so a gate's seeds are no longer arbitrary integers.
_GATE_SEED_START = 0
_GATE_SEED_COUNT = 1_000_000


def _install_context(mod, run_id: str | None = None):
    """[D6.2] Install a run context on a freshly loaded flush section.

    The flush FAILS CLOSED with no context — an absent context means nobody
    established what this run is, so the three walls cannot run and a written
    checkpoint could never be verified on resume. Every gate therefore installs
    one, exactly as `optimize_window` does at run start.
    """
    import utils.checkpoint_d6_2 as _ck
    run_id = run_id or mod._flush_run_id()
    components = _ck.run_context_components(
        dataset_version_id="daily3-20260801T000000000000Z-abcdef123456",
        dataset_filename="daily3-20260801T000000000000Z-abcdef123456.json",
        dataset_sha256="ab" * 32, repository_commit="c" * 40,
        prng_base="java_lcg", skip_modes_executed=("constant",),
        seed_start=_GATE_SEED_START, seed_count=_GATE_SEED_COUNT,
        execution_set_id=None)
    context = _ck.RunContext(
        run_id=run_id,
        checkpoint_dir=_ck.resolve_checkpoint_dir(mod._flush_checkpoint_root(),
                                                  run_id),
        run_context_digest=_ck.build_run_context_digest(components),
        prng_base="java_lcg", skip_modes_executed=("constant",),
        seed_start=_GATE_SEED_START, seed_count=_GATE_SEED_COUNT,
        components=components)
    mod._install_flush_run_context(context)
    return context


def _fresh(clear_in_memory: bool = False, flush_every: int | None = None,
           context: bool = True):
    """A pristine copy of the flush section under test.

    [D6.2] `clear_in_memory` defaults to False so the pre-existing gates keep
    observing the accumulator after a flush; the PRODUCTION default is True and
    is pinned separately by `g_clear_after`.
    """
    src = _ACTIVE_SRC if _ACTIVE_SRC is not None else _flush_section_src()
    mod = _load_section(src, "production" if _ACTIVE_SRC is None else "mutant")
    if flush_every is not None:
        mod._FLUSH_EVERY = flush_every
    mod._flush_last_count = 0
    mod._FLUSH_CLEAR_IN_MEMORY = clear_in_memory
    mod.__d6_1_needs_context__ = context
    return mod


def _patch(src: str, old: str, new: str, label: str) -> str:
    """Part 1 of the four-part rule: the mutation MUST actually apply once."""
    count = src.count(old)
    assert count == 1, (
        f"{label}: anchor is not unique ({count} occurrences) — the mutation "
        f"would be unverifiable")
    return src.replace(old, new, 1)


def _mutant(old: str, new: str, label: str, *, clear_in_memory: bool = False,
            flush_every: int | None = None):
    mod = _load_section(_patch(_flush_section_src(), old, new, label), label)
    if flush_every is not None:
        mod._FLUSH_EVERY = flush_every
    mod._flush_last_count = 0
    mod._FLUSH_CLEAR_IN_MEMORY = clear_in_memory
    return mod


def _executed(src: str, marker: str, label: str) -> None:
    """Part 2: the mutated text is present in the source the gate will load."""
    assert marker in src, (
        f"{label}: the mutated text is absent from the source under test — the "
        f"mutant did not take effect")


def _positive_control(name: str, detector) -> None:
    """Part 3: the detector must PASS against the UNMUTATED section."""
    try:
        detector()
    except Exception as exc:                                    # noqa: BLE001
        raise AssertionError(
            f"POSITIVE CONTROL FAILED for {name}: the detector reds against "
            f"the UNMUTATED section ({type(exc).__name__}: {exc}) — any kill "
            f"it records would be unattributable") from exc


def _record(label: str, detector, credited: str, marker=None, src=None):
    """Parts 3+4: run `detector` and require it to FAIL from the defect."""
    if src is not None and marker is not None:
        _executed(src, marker, label)
    try:
        detector()
    except AssertionError as exc:
        sig = str(exc).splitlines()[0][:150] or "AssertionError"
        _MUTANTS.append((label, f"AssertionError: {sig}", credited,
                         "applies-once ✓ | mutated-path ✓ | detector-clean ✓ "
                         "| injected-defect ✓"))
        return
    except Exception as exc:                                    # noqa: BLE001
        sig = f"{type(exc).__name__}: {str(exc).splitlines()[0][:130]}"
        _MUTANTS.append((label, sig, credited,
                         "applies-once ✓ | mutated-path ✓ | detector-clean ✓ "
                         "| injected-defect ✓"))
        return
    raise AssertionError(f"MUTANT SURVIVED: {label} — {credited} did not red")


# ═════════════════════════════════════════════════════════════════════════════
# helpers
# ═════════════════════════════════════════════════════════════════════════════
_RUN_SEQ = 0


@contextlib.contextmanager
def _in_tmp(run_id: str | None = None):
    """Isolate a gate: temp dir as the STABLE snapshot root, plus a distinct
    run id so no two gates can share a snapshot directory.

    `PRNG_CHECKPOINT_ROOT` is what keeps writes out of the repo now that the
    snapshot root is deliberately NOT the CWD (Beta path condition 2).
    """
    global _RUN_SEQ
    _RUN_SEQ += 1
    cwd = os.getcwd()
    prev_root = os.environ.get("PRNG_CHECKPOINT_ROOT")
    prev_run = os.environ.get("PRNG_CHECKPOINT_RUN_ID")
    with tempfile.TemporaryDirectory() as tmp:
        real = os.path.realpath(tmp)
        os.environ["PRNG_CHECKPOINT_ROOT"] = real
        os.environ["PRNG_CHECKPOINT_RUN_ID"] = run_id or f"gate-run-{_RUN_SEQ}"
        try:
            os.chdir(real)
            yield real
        finally:
            os.chdir(cwd)
            for k, v in (("PRNG_CHECKPOINT_ROOT", prev_root),
                         ("PRNG_CHECKPOINT_RUN_ID", prev_run)):
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def _cands(seeds, score=0.5, trial=1, mode="constant"):
    """[D6.2] FULL canonical 24-field records.

    The pre-D6.2 four-field stub is no longer sufficient: the flush runs
    `_validate_raw_candidates` (D3's strict 24-field validator) over every newly
    observed record before anything is written or cleared, so a gate feeding
    four fields would fail the wall rather than exercise the durability property
    it is testing.
    """
    prng_type = "java_lcg" if mode == "constant" else "java_lcg_hybrid"
    return [{
        "seed": int(s), "forward_match_rate": 0.4, "reverse_match_rate": 0.6,
        "score": score, "window_size": 8, "offset": 0, "skip_min": 0,
        "skip_max": 16, "skip_range": 16, "sessions": ["midday"],
        "trial_number": int(trial), "prng_base": "java_lcg",
        "skip_mode": mode, "prng_type": prng_type,
        "forward_count": 10.0, "reverse_count": 12.0,
        "bidirectional_count": 3.0, "intersection_count": 3.0,
        "intersection_ratio": 0.25, "forward_only_count": 7.0,
        "reverse_only_count": 9.0, "survivor_overlap_ratio": 0.3,
        "bidirectional_selectivity": 10.0 / 12.0,
        "intersection_weight": 3.0 / 22.0,
    } for s in seeds]


def _ckpt_paths(mod, root=None):
    """The RUN-ISOLATED snapshot directory and its two members.

    Resolved through the module's own `_flush_checkpoint_dir()` so the gates
    follow the production path rule (stable root + `<run_id>/`) instead of
    re-deriving it and drifting.
    """
    d = mod._flush_checkpoint_dir()
    return (os.path.join(d, mod._CHECKPOINT_ALL_NAME),
            os.path.join(d, mod._CHECKPOINT_BINARY_NAME), d)


def _run(mod, acc, label="t"):
    """Invoke the flush, capturing stdout and stderr.

    [D6.2] The run context is installed HERE, not in `_fresh`, because it is
    derived from `PRNG_CHECKPOINT_ROOT` / `PRNG_CHECKPOINT_RUN_ID` and those are
    only set once a gate has entered `_in_tmp()`. Installing it at the single
    call site keeps every existing gate body unchanged.
    """
    if getattr(mod, "__d6_1_needs_context__", True):
        _ctx = mod._active_flush_run_context()
        if _ctx is None or _ctx.checkpoint_dir != mod._flush_checkpoint_dir():
            _install_context(mod)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        mod._flush_npz_incremental(acc, label=label)
    return out.getvalue(), err.getvalue()


def _pair_status(mod, allp, binp):
    """[D6.2] Classify the on-disk pair through the PRODUCTION recovery matrix.

    `_flush_inspect_pair` and its `_PAIR_*` vocabulary are gone: D6.2 replaced
    the single four-field content digest with two separate identities and moved
    classification into `utils.checkpoint_d6_2.recover_checkpoint`, whose nine
    rows are gated in `tests/test_s172_d6_2_checkpoint_reconciliation.py`. This
    helper reduces that richer answer to the coarse label D6.1's durability
    gates need, WITHOUT reimplementing any of the decision.

    Returns one of: "absent" · "consistent" · the recovery row name · "failed:…".
    """
    import utils.checkpoint_d6_2 as _ck
    if not os.path.exists(allp) and not os.path.exists(binp):
        return "absent"
    ctx = mod._active_flush_run_context()
    try:
        outcome = _ck.recover_checkpoint(
            os.path.dirname(binp), run_id=ctx.run_id,
            run_context_digest=ctx.run_context_digest)
    except Exception as exc:                                    # noqa: BLE001
        return f"failed:{type(exc).__name__}"
    return ("consistent" if outcome.row == _ck.ROW_CONSISTENT else outcome.row)


def _temps_in(d):
    if not os.path.isdir(d):
        return []
    return sorted(p for p in os.listdir(d) if p.endswith(".tmp"))


class _OsProxy:
    """Delegates to the real `os`, with injectable failures.

    Used to reach failure points that cannot otherwise be provoked
    deterministically (a mid-sequence `os.replace` fault). Everything not
    overridden passes straight through, so the module under test behaves
    exactly as in production apart from the injected fault.
    """

    def __init__(self, fail_replace_on=None, exc=None):
        self._real = os
        self._fail_on = fail_replace_on      # 1-based call index, or None
        self._exc = exc or OSError(28, "No space left on device")
        self.replace_calls = 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    def replace(self, src, dst):
        self.replace_calls += 1
        if self._fail_on is not None and self.replace_calls == self._fail_on:
            raise self._exc
        return self._real.replace(src, dst)


def _seeds_on_disk(path):
    """[D6.2] The seed column is `seed`, not `seeds`.

    REV5 §2.1 is explicit: the checkpoint stores RECORD field names, and the
    `seed -> seeds` / `*_match_rate -> *_matches` renames belong to the ARRAY
    domain only ("do not apply that rename here"). Member A's approved payload
    is that record field plus `score`.
    """
    with np.load(path) as z:
        return sorted(int(s) for s in z["seed"])


# ═════════════════════════════════════════════════════════════════════════════
# G-SUFFIX
# ═════════════════════════════════════════════════════════════════════════════
def g_suffix(mod=None):
    """The temp target cannot be suffix-rewritten by NumPy.

    The PROPERTY, not the mechanism: the file NumPy actually creates must equal
    the path `os.replace` consumes. This is gated against a temp name that does
    NOT end in `.npz` — precisely the shape that broke D1 — so the gate proves
    the write mechanism defeats the rewrite, and would still hold for any
    future temp name.
    """
    mod = _fresh(flush_every=2) if mod is None else mod

    # (1) the temp name really is the D1-shaped one (no `.npz` tail)
    tmp_name = mod._flush_tmp_name("/x/y/incremental_survivors_all.npz")
    assert not tmp_name.endswith(".npz"), (
        f"the temp name {tmp_name!r} ends in .npz — this gate must exercise "
        f"the suffix-rewrite-prone shape, or it proves nothing")

    # (2) the file NumPy creates == the path handed to it, exactly
    with _in_tmp() as root:
        target = os.path.join(root, "probe.tmp")
        before = set(os.listdir(root))
        mod._flush_write_npz(target, {"seeds": np.array([1], dtype=np.uint64)})
        created = set(os.listdir(root)) - before
        assert created == {"probe.tmp"}, (
            f"NumPy created {sorted(created)}, not exactly {{'probe.tmp'}} — "
            f"the implicit .npz suffix was applied (this is D1)")
        assert os.path.isfile(target), f"{target} was not created"
        # and it is a real, loadable npz
        with np.load(target) as z:
            assert list(z["seeds"]) == [1]

    # (3) end to end: a real flush leaves NO `*.tmp.npz` debris anywhere
    with _in_tmp() as root:
        acc = {"bidirectional": _cands([1, 2, 3])}
        out, err = _run(mod, acc)
        allp, binp, cdir = _ckpt_paths(mod, root)
        assert os.path.isfile(allp), (
            f"the flush did not land {allp} — stdout={out!r} stderr={err!r}")
        assert os.path.isfile(binp), f"the flush did not land {binp}"
        strays = [p for p in os.listdir(cdir) if p.endswith(".tmp.npz")]
        assert not strays, (
            f"NumPy suffix-rewrote a temp target: {strays} (this is D1)")


# ═════════════════════════════════════════════════════════════════════════════
# G-ATOMIC-ACCUM / G-ATOMIC-BINARY
# ═════════════════════════════════════════════════════════════════════════════
def g_atomic(mod=None):
    """Each final NPZ is either the COMPLETE prior content or the COMPLETE new
    content — never partial, and never absent once it has existed."""
    mod = _fresh(flush_every=2) if mod is None else mod

    with _in_tmp() as root:
        allp, binp, _ = _ckpt_paths(mod, root)

        # establish a prior generation of the checkpoint
        acc = {"bidirectional": _cands([1, 2])}
        _run(mod, acc)
        assert _seeds_on_disk(allp) == [1, 2]
        assert _seeds_on_disk(binp) == [1, 2]

        # a failure while building the SECOND payload must leave BOTH finals
        # exactly as they were: no final name is touched until both temps exist
        mod._flush_last_count = 0
        boom = {"n": 0}
        real_write = mod._flush_write_npz

        def _second_write_fails(tmp_path, arrays):
            boom["n"] += 1
            if boom["n"] == 2:
                raise OSError(28, "No space left on device")
            return real_write(tmp_path, arrays)

        mod._flush_write_npz = _second_write_fails
        try:
            acc2 = {"bidirectional": _cands([7, 8, 9])}
            _out, err = _run(mod, acc2)
        finally:
            mod._flush_write_npz = real_write

        assert boom["n"] == 2, "the second write was never attempted"
        assert "ERROR" in err, "a write failure was not surfaced"
        assert _seeds_on_disk(allp) == [1, 2], (
            "G-ATOMIC-ACCUM: the accumulator checkpoint changed although the "
            "flush failed — a final name was touched before both temps were "
            "complete")
        assert _seeds_on_disk(binp) == [1, 2], (
            "G-ATOMIC-BINARY: the binary checkpoint changed although the "
            "flush failed")

        # both files remain complete, loadable npz archives
        for p in (allp, binp):
            with zipfile.ZipFile(p) as zf:
                assert zf.testzip() is None, f"{p} is a corrupt archive"


# ═════════════════════════════════════════════════════════════════════════════
# G-CLEAR-AFTER
# ═════════════════════════════════════════════════════════════════════════════
def g_clear_after(mod=None):
    """The in-memory list clears ONLY after BOTH replaces have succeeded.

    The clear is DISABLED in production (`_FLUSH_CLEAR_IN_MEMORY = False`,
    because the 4-array checkpoint cannot reconstruct the 24
    CANONICAL_RECORD_FIELDS the D3.5 finalizer reads from this list). The
    ORDERING property is gated regardless, with the flag forced on, so that
    enabling the clear later is a one-line change against a proven gate.
    """
    # (1) production default: the clear does NOT run, so every candidate still
    #     reaches the finalizer
    prod = _fresh(flush_every=2) if mod is None else mod
    assert prod._FLUSH_CLEAR_IN_MEMORY is False, (
        "_FLUSH_CLEAR_IN_MEMORY is enabled — the D3.5 finalizer consumes the "
        "in-memory list and needs all 24 canonical fields; a 4-array "
        "checkpoint cannot restore them")
    with _in_tmp():
        acc = {"bidirectional": _cands([1, 2, 3])}
        _run(prod, acc)
        assert len(acc["bidirectional"]) == 3, (
            "the accumulator was cleared with the clear disabled")

    # (2) with the clear ENABLED, it runs only after a fully successful flush
    on = _fresh(clear_in_memory=True, flush_every=2) if mod is None else mod
    on._FLUSH_CLEAR_IN_MEMORY = True
    on._flush_last_count = 0
    with _in_tmp() as root:
        allp, binp, _ = _ckpt_paths(on, root)
        acc = {"bidirectional": _cands([1, 2, 3])}
        _run(on, acc)
        assert os.path.isfile(allp) and os.path.isfile(binp), \
            "the flush did not succeed, so this gate cannot judge the clear"
        assert acc["bidirectional"] == [], (
            "the clear did not run after a fully successful flush")

    # (3) a failure at the SECOND replace must leave the list intact — the
    #     clear is strictly after both
    on2 = _fresh(clear_in_memory=True, flush_every=2)
    with _in_tmp():
        proxy = _OsProxy(fail_replace_on=2)
        on2._os_flush = proxy
        acc = {"bidirectional": _cands([4, 5, 6])}
        _out, err = _run(on2, acc)
        assert proxy.replace_calls == 2, (
            f"the second replace was not reached ({proxy.replace_calls} "
            f"calls) — this gate did not exercise the ordering")
        assert "ERROR" in err, "the replace failure was not surfaced"
        assert len(acc["bidirectional"]) == 3, (
            "the list was cleared although the SECOND replace failed — the "
            "clear is not strictly after both replaces")

    # (4) a failure at the FIRST replace likewise retains everything
    on3 = _fresh(clear_in_memory=True, flush_every=2)
    with _in_tmp():
        proxy = _OsProxy(fail_replace_on=1)
        on3._os_flush = proxy
        acc = {"bidirectional": _cands([4, 5, 6])}
        _out, err = _run(on3, acc)
        assert proxy.replace_calls == 1
        assert len(acc["bidirectional"]) == 3, (
            "the list was cleared although the FIRST replace failed")


# ═════════════════════════════════════════════════════════════════════════════
# G-RETAIN-ON-FAIL
# ═════════════════════════════════════════════════════════════════════════════
def g_retain_on_fail(mod=None):
    """Failure injected at each of four points → ZERO candidate loss, always.

    Injection points: before the first write, between write and replace,
    between the two replaces, and after both. The clear is forced ON for this
    gate, because with it off retention is trivially true and the gate would
    prove nothing about the failure paths.
    """
    seeds = [11, 22, 33, 44]

    def _fresh_on():
        m = _fresh(clear_in_memory=True, flush_every=2)
        m._FLUSH_CLEAR_IN_MEMORY = True
        return m

    # (a) before the first write
    m = _fresh_on()
    with _in_tmp():
        def _no_write(_tmp, _arrays):
            raise OSError(13, "Permission denied")
        m._flush_write_npz = _no_write
        acc = {"bidirectional": _cands(seeds)}
        _out, err = _run(m, acc)
        assert "ERROR" in err
        assert len(acc["bidirectional"]) == 4, "(a) candidates lost"

    # (b) between write and replace (first replace faults)
    m = _fresh_on()
    with _in_tmp():
        proxy = _OsProxy(fail_replace_on=1)
        m._os_flush = proxy
        acc = {"bidirectional": _cands(seeds)}
        _out, err = _run(m, acc)
        assert proxy.replace_calls == 1
        assert "ERROR" in err
        assert len(acc["bidirectional"]) == 4, "(b) candidates lost"

    # (c) between the two replaces
    m = _fresh_on()
    with _in_tmp():
        proxy = _OsProxy(fail_replace_on=2)
        m._os_flush = proxy
        acc = {"bidirectional": _cands(seeds)}
        _out, err = _run(m, acc)
        assert proxy.replace_calls == 2
        assert "ERROR" in err
        assert len(acc["bidirectional"]) == 4, "(c) candidates lost"

    # (d) after both replaces (the directory fsync faults)
    m = _fresh_on()
    with _in_tmp() as root:
        allp, binp, _ = _ckpt_paths(m, root)

        def _fsync_boom(_d):
            raise OSError(5, "I/O error")
        m._flush_fsync_dir = _fsync_boom
        acc = {"bidirectional": _cands(seeds)}
        _out, err = _run(m, acc)
        assert "ERROR" in err, "the post-replace failure was not surfaced"
        assert os.path.isfile(allp) and os.path.isfile(binp), \
            "(d) both replaces should already have landed"
        assert len(acc["bidirectional"]) == 4, (
            "(d) candidates lost — a failure AFTER both replaces still cleared "
            "the list, so the data exists only on disk in a 4-array projection")


# ═════════════════════════════════════════════════════════════════════════════
# G-NO-TEMP-LEAK
# ═════════════════════════════════════════════════════════════════════════════
def g_no_temp_leak(mod=None):
    """No temp file remains after success or after ANY failure path."""
    # success
    m = _fresh(flush_every=2) if mod is None else mod
    with _in_tmp() as root:
        _allp, _binp, cdir = _ckpt_paths(m, root)
        acc = {"bidirectional": _cands([1, 2, 3])}
        _run(m, acc)
        assert _temps_in(cdir) == [], (
            f"temp files survived a SUCCESSFUL flush: {_temps_in(cdir)}")

    # failure at each replace, and at the write
    for label, setup in (
        ("first replace", lambda mm: setattr(mm, "_os_flush",
                                             _OsProxy(fail_replace_on=1))),
        ("second replace", lambda mm: setattr(mm, "_os_flush",
                                              _OsProxy(fail_replace_on=2))),
    ):
        mm = _fresh(flush_every=2)
        with _in_tmp() as root:
            _a, _b, cdir = _ckpt_paths(mm, root)
            setup(mm)
            acc = {"bidirectional": _cands([1, 2, 3])}
            _run(mm, acc)
            assert _temps_in(cdir) == [], (
                f"temp files survived a failure at the {label}: "
                f"{_temps_in(cdir)}")

    # a crashed run's orphan (a temp owned by a dead pid) is collected
    mm = _fresh(flush_every=2)
    with _in_tmp() as root:
        _a, _b, cdir = _ckpt_paths(mm, root)
        os.makedirs(cdir, exist_ok=True)
        # pid 2**22 is above /proc/sys/kernel/pid_max on any normal box
        orphan = os.path.join(cdir, f"{mm._CHECKPOINT_ALL_NAME}.flush-4194303.tmp")
        with open(orphan, "wb") as fh:
            fh.write(b"debris")
        acc = {"bidirectional": _cands([1, 2, 3])}
        _run(mm, acc)
        assert not os.path.exists(orphan), (
            "a crashed run's orphaned temp was not collected by the next flush")

    # a LIVE sibling's temp is NOT collected (parallel partition workers share
    # one CWD — a blind sweep would delete an in-flight write)
    mm = _fresh(flush_every=2)
    with _in_tmp() as root:
        _a, _b, cdir = _ckpt_paths(mm, root)
        os.makedirs(cdir, exist_ok=True)
        live = os.path.join(cdir,
                            f"{mm._CHECKPOINT_ALL_NAME}.flush-{os.getpid()}.tmp")
        with open(live, "wb") as fh:
            fh.write(b"in flight")
        n = mm._flush_purge_stale_temps(cdir)
        assert n == 0 and os.path.exists(live), (
            "the purge deleted a LIVE process's in-flight temp")


# ═════════════════════════════════════════════════════════════════════════════
# G-CUMULATIVE
# ═════════════════════════════════════════════════════════════════════════════
def g_cumulative(mod=None):
    """Repeated flushes preserve EXACT cumulative counts; merge-by-seed dedup
    (highest score wins) and the prior-checkpoint merge behave as before."""
    mod = _fresh(flush_every=1) if mod is None else mod

    with _in_tmp() as root:
        allp, binp, _ = _ckpt_paths(mod, root)

        # flush 1 — seeds 1,2,3
        acc = {"bidirectional": _cands([1, 2, 3], score=0.5)}
        _run(mod, acc)
        assert _seeds_on_disk(allp) == [1, 2, 3], _seeds_on_disk(allp)

        # flush 2 — new seeds 4,5 plus a REPEAT of 2 at a HIGHER score.
        # With the clear disabled the list still holds 1..3, so this also
        # exercises re-flushing already-persisted candidates (idempotence).
        # [D6.2] the re-observation of seed 2 carries a DISTINCT trial number.
        # `(seed, trial_number, skip_mode)` is the replay key: two different
        # scores under ONE key is corruption, not a competition, and the flush
        # now (correctly) raises `AccumulatorConsistencyError` on it. Winner
        # selection between DIFFERENT trials is what this gate is about.
        acc["bidirectional"] = _cands([1, 2, 3], score=0.5) + \
            _cands([4, 5], score=0.5) + _cands([2], score=0.99, trial=2)
        _run(mod, acc)
        assert _seeds_on_disk(allp) == [1, 2, 3, 4, 5], (
            f"cumulative seed set wrong: {_seeds_on_disk(allp)}")
        assert _seeds_on_disk(binp) == [1, 2, 3, 4, 5]

        # highest score per seed wins
        with np.load(allp) as z:
            got = dict(zip((int(s) for s in z["seed"]),
                           (float(v) for v in z["score"])))
        assert abs(got[2] - 0.99) < 1e-6, (
            f"dedup did not keep the highest score for seed 2: {got[2]}")
        assert abs(got[1] - 0.5) < 1e-6, got[1]

        # [D6.2] The CUMULATIVE CANONICAL STATE is what carries seeds forward:
        # drop the in-memory list entirely and the persisted seeds must survive.
        # Note what changed and why — D6.1 re-read member A from disk and merged
        # it; D6.2 never does, because member A is a MARKER STUB and must never
        # be consumed as an accumulator backup. The cumulative state lives in
        # the run context (and is seeded from member B on resume), so the
        # property this gate asserts is unchanged while the mechanism is now the
        # one the asymmetric architecture allows.
        acc["bidirectional"] = _cands([6], score=0.5)
        mod._flush_last_count = 0
        _run(mod, acc)
        assert _seeds_on_disk(allp) == [1, 2, 3, 4, 5, 6], (
            f"the prior-checkpoint merge lost seeds: {_seeds_on_disk(allp)}")


# ═════════════════════════════════════════════════════════════════════════════
# G-CRASH-RESTART
# ═════════════════════════════════════════════════════════════════════════════
def g_crash_restart(mod=None):
    """The three crash points, each asserting what a RESTART observes, and that
    the next flush self-repairs an inconsistent pair.

    Sequential-atomic with self-repair — NOT jointly atomic. The pair can be
    inconsistent; the claim is that it is DETECTABLE and REPAIRED, and that no
    candidate is lost in the meantime.
    """
    # ── (a) crash BEFORE any replace ────────────────────────────────────────
    m = _fresh(flush_every=2)
    with _in_tmp() as root:
        allp, binp, cdir = _ckpt_paths(m, root)
        acc = {"bidirectional": _cands([1, 2])}
        _run(m, acc)                                   # a good prior exists
        assert _seeds_on_disk(allp) == [1, 2]

        m._flush_last_count = 0
        proxy = _OsProxy(fail_replace_on=1)
        m._os_flush = proxy
        acc["bidirectional"] = _cands([1, 2, 3, 4])
        _run(m, acc)
        # RESTART SEES: both finals at their complete PRIOR content, no temps,
        # and every candidate still in memory.
        assert _seeds_on_disk(allp) == [1, 2], "(a) _all was modified"
        assert _seeds_on_disk(binp) == [1, 2], "(a) _binary was modified"
        assert _temps_in(cdir) == [], "(a) temp debris"
        assert len(acc["bidirectional"]) == 4, "(a) candidates lost"

    # ── (b) crash BETWEEN the two replaces ──────────────────────────────────
    m = _fresh(flush_every=2)
    with _in_tmp() as root:
        allp, binp, cdir = _ckpt_paths(m, root)
        acc = {"bidirectional": _cands([1, 2])}
        _run(m, acc)

        m._flush_last_count = 0
        real_os = m._os_flush
        proxy = _OsProxy(fail_replace_on=2)
        m._os_flush = proxy
        acc["bidirectional"] = _cands([1, 2, 3, 4])
        _run(m, acc)

        # RESTART SEES: a MIXED pair — member A advanced, member B did not.
        assert _seeds_on_disk(allp) == [1, 2, 3, 4], "(b) member A did not advance"
        assert _seeds_on_disk(binp) == [1, 2], "(b) member B advanced"
        # ...and it is detected by TRANSACTION IDENTITY. Seed-set comparison is
        # NOT the detector — see G-TRANSACTION-IDENTITY for the case where the
        # seed sets are identical and only a score differs.
        # [D6.2] A is at n+1, B at n — the A-FIRST crash. The recovery matrix
        # names it row 4 ("A a valid NEWER uncommitted marker"): A is discarded,
        # B is recovered, and the repaired sequence initializes ABOVE A. The
        # blanket "higher valid sequence wins" rule would recover A here, which
        # is why the A cases are disambiguated at all.
        import utils.checkpoint_d6_2 as _ck_b
        assert _pair_status(m, allp, binp) == _ck_b.ROW_A_NEWER, (
            f"(b) the mixed pair was not classified as the A-first crash: "
            f"{_pair_status(m, allp, binp)}")
        # each file is INDIVIDUALLY complete and loadable
        for p in (allp, binp):
            with zipfile.ZipFile(p) as zf:
                assert zf.testzip() is None, f"(b) {p} is corrupt"
        assert _temps_in(cdir) == [], "(b) temp debris"
        assert len(acc["bidirectional"]) == 4, "(b) candidates lost"

        # SELF-REPAIR: the next flush restores consistency, losing nothing
        m._os_flush = real_os
        m._flush_last_count = 0
        _run(m, acc)
        assert _seeds_on_disk(allp) == [1, 2, 3, 4]
        assert _seeds_on_disk(binp) == [1, 2, 3, 4], (
            "(b) the next flush did not self-repair the inconsistent pair")

    # ── (c) crash AFTER both replaces ───────────────────────────────────────
    m = _fresh(flush_every=2)
    with _in_tmp() as root:
        allp, binp, cdir = _ckpt_paths(m, root)
        acc = {"bidirectional": _cands([1, 2, 3, 4])}
        _run(m, acc)
        # RESTART SEES: a consistent, complete pair; nothing to repair.
        assert _seeds_on_disk(allp) == [1, 2, 3, 4]
        assert _seeds_on_disk(binp) == [1, 2, 3, 4]
        assert _seeds_on_disk(allp) == _seeds_on_disk(binp), \
            "(c) the pair is inconsistent after a fully successful flush"
        assert _temps_in(cdir) == [], "(c) temp debris"
        # re-flushing the same candidates is IDEMPOTENT (no double counting)
        m._flush_last_count = 0
        _run(m, acc)
        assert _seeds_on_disk(allp) == [1, 2, 3, 4], (
            "(c) replaying a flush after a post-replace crash double-counted")


# ═════════════════════════════════════════════════════════════════════════════
# G-CADENCE
# ═════════════════════════════════════════════════════════════════════════════
def g_cadence(mod=None):
    """`_FLUSH_EVERY` / `_flush_last_count` entry gating is unchanged, and the
    gate now pins the SUCCESSFUL flush rather than the failed attempt."""
    # (1) the entry-gate text is byte-identical to the frozen S152 rule
    ORACLE_CADENCE_GATE = (
        '    bidi = accumulator.get("bidirectional", [])\n'
        "    current_count = len(bidi)\n"
        "\n"
        "    new_since_last = current_count - _flush_last_count\n"
        "    if new_since_last < _FLUSH_EVERY:\n"
        "        return  # not enough new survivors yet"
    )
    assert ORACLE_CADENCE_GATE in _read(_INTEG_FULL), (
        "the flush ENTRY GATE has drifted — D6.1 repairs the write, never the "
        "cadence rule")

    # (2) the env override still governs the default
    m = _fresh()
    assert isinstance(m._FLUSH_EVERY, int) and m._FLUSH_EVERY > 0
    assert 'environ.get("PRNG_FLUSH_EVERY", "10")' in _read(_INTEG_FULL), (
        "the PRNG_FLUSH_EVERY override was removed")

    # (3) below threshold: nothing happens at all — no dir, no file, no output
    m = _fresh(flush_every=10)
    with _in_tmp() as root:
        acc = {"bidirectional": _cands(range(9))}
        out, err = _run(m, acc, label="below")
        assert _TAG not in out, f"fired below threshold: {out!r}"
        assert err == "", f"stderr below threshold: {err!r}"
        assert os.listdir(root) == [], (
            f"the flush wrote below threshold: {os.listdir(root)}")

        # (4) at threshold: it fires AND the checkpoint actually lands
        acc["bidirectional"] = _cands(range(10))
        out, err = _run(m, acc, label="at")
        assert _TAG in out, f"did not fire at threshold: {out!r}"
        allp, binp, _ = _ckpt_paths(m, root)
        assert os.path.isfile(allp) and os.path.isfile(binp), (
            "the at-threshold flush did not land a checkpoint — this gate now "
            "pins SUCCESS, not the pre-D6.1 failed attempt")
        assert m._flush_success_count == 1, m._flush_success_count
        assert m._flush_failure_count == 0, m._flush_failure_count

        # (5) `_flush_last_count` advanced, so the NEXT call is gated again
        assert m._flush_last_count == 10, m._flush_last_count
        out, _err = _run(m, acc, label="again")
        assert _TAG not in out, (
            "the flush fired again with no new survivors — the cadence gate "
            "no longer advances")

    # (6) the D3.25 one-flush-per-trial invariant: EXACTLY ONE call site in
    #     each adapter, unchanged by D6.1
    import ast
    tree = ast.parse(_read(_INTEG_FULL))
    for fn_name in ("_build_test_result_from_pw", "_build_test_result_from_miner"):
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == fn_name)
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name)
                 and n.func.id == "_flush_npz_incremental"]
        assert len(calls) == 1, (
            f"{fn_name} makes {len(calls)} flush calls — the D3.25 "
            f"one-flush-per-trial invariant has shifted")


# ═════════════════════════════════════════════════════════════════════════════
# G-VISIBLE-FAILURE
# ═════════════════════════════════════════════════════════════════════════════
def g_visible_failure(mod=None):
    """An unexpected write failure is SURFACED, not swallowed (D2).

    The contract: non-fatal to the trial, but loud on stderr, counted, and
    distinguishable by tier. Pre-D6.1 every failure — including a total,
    permanent outage — produced one stdout "Warning:" and nothing else.
    """
    # (1) a WRITE FAILURE (OSError) is loud on stderr and counted
    m = _fresh(flush_every=2) if mod is None else mod
    with _in_tmp():
        proxy = _OsProxy(fail_replace_on=1)
        m._os_flush = proxy
        acc = {"bidirectional": _cands([1, 2, 3])}
        out, err = _run(m, acc)
        assert "ERROR" in err, (
            f"a write failure was not surfaced on stderr — stdout={out!r} "
            f"stderr={err!r}")
        assert "Traceback" in err, "no traceback accompanied the failure"
        assert m._flush_failure_count == 1, (
            f"_flush_failure_count is {m._flush_failure_count} — a soak or "
            f"WATCHER cannot observe the failure")
        assert m._flush_last_error is not None
        assert m._flush_success_count == 0

    # (2) the helper stays NON-FATAL: it returns normally, never raises
    m2 = _fresh(flush_every=2)
    with _in_tmp():
        m2._os_flush = _OsProxy(fail_replace_on=1)
        acc = {"bidirectional": _cands([1, 2, 3])}
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            rv = m2._flush_npz_incremental(acc, label="nf")
        assert rv is None, "the helper must stay non-fatal to the trial"

    # (3) an UNEXPECTED (non-OSError) failure is surfaced under its own tier
    m3 = _fresh(flush_every=2)
    with _in_tmp():
        def _weird(_t, _a):
            raise RuntimeError("contract violation")
        m3._flush_write_npz = _weird
        acc = {"bidirectional": _cands([1, 2, 3])}
        out, err = _run(m3, acc)
        assert "UNEXPECTED ERROR" in err, (
            f"an unexpected failure was not distinguished from a disk "
            f"condition — stderr={err!r}")
        assert m3._flush_failure_count == 1

    # (4) [D6.2] A corrupt member A on disk is simply OVERWRITTEN, quietly.
    #
    # D6.1 warned here because it MERGED the prior member A back in and had to
    # say when it could not. D6.2 never reads member A as data — A is a MARKER
    # STUB, and consuming it as an accumulator backup is precisely what §0
    # forbids — so there is no "could not merge the prior" condition left to
    # report. The property that survives, and is asserted, is that a corrupt
    # prior member does NOT escalate to an error and does NOT stop the flush.
    m4 = _fresh(flush_every=2)
    with _in_tmp() as root:
        allp, _binp, cdir = _ckpt_paths(m4, root)
        os.makedirs(cdir, exist_ok=True)
        with open(allp, "wb") as fh:
            fh.write(b"not an npz at all")
        acc = {"bidirectional": _cands([1, 2, 3])}
        out, err = _run(m4, acc)
        assert "ERROR" not in err, (
            f"a corrupt prior member was escalated to an error: {err!r}")
        assert m4._flush_success_count == 1, (
            "the flush did not proceed past a corrupt prior member")
        assert _seeds_on_disk(allp) == [1, 2, 3]

    # (5) [D6.2] NO RUN CONTEXT is a loud, fail-closed condition — and it clears
    #     nothing. An absent context is not a neutral "unknown": it means nobody
    #     established the run identity, the declared seed interval or the
    #     `run_context_digest`, so the three walls cannot run and the checkpoint
    #     could never be verified on resume.
    m5 = _fresh(flush_every=2, context=False)
    m5._FLUSH_CLEAR_IN_MEMORY = True
    with _in_tmp():
        m5._clear_flush_run_context()
        acc = {"bidirectional": _cands([1, 2, 3])}
        out, err = _run(m5, acc)
        assert "ERROR" in err and "no run context" in err, (
            f"a missing run context was not surfaced: stdout={out!r} "
            f"stderr={err!r}")
        assert len(acc["bidirectional"]) == 3, (
            "the accumulator was cleared with no run context installed")
        assert m5._flush_failure_count == 1


# ═════════════════════════════════════════════════════════════════════════════
# G-TRANSACTION-IDENTITY   [Beta blocker — seed-set comparison is insufficient]
# ═════════════════════════════════════════════════════════════════════════════
def g_transaction_identity(mod=None):
    """Beta's counterexample: a mixed pair whose SEED SETS ARE IDENTICAL.

    Old pair holds seed 42 @ 0.40. The next transaction holds seed 42 @ 0.90.
    A crash after replacing member A leaves A=0.90 / B=0.40 — two different
    transactions, one seed set. Seed-set comparison reports agreement; only
    transaction identity detects it.

    This gate asserts BOTH halves: that seed-set comparison genuinely cannot see
    the difference (so the gate has teeth), and that the production detector
    classifies the pair correctly.

    [D6.2] RE-POINTED, NOT RELAXED. `_flush_inspect_pair` / `_PAIR_*` /
    `four_field_content_digest` are gone — D6.2 split the single content digest
    into `canonical_state_digest` (shared, content only) and
    `member_content_digest` (per member, identity included) and moved
    classification into `recover_checkpoint`'s NINE-ROW MATRIX. The blocker's
    property is unchanged and is asserted here against that matrix; the matrix's
    own nine rows are gated in
    `tests/test_s172_d6_2_checkpoint_reconciliation.py` (G-RECOVERY-MATRIX).
    """
    import utils.checkpoint_d6_2 as _ck

    # ── the score-only case ─────────────────────────────────────────────────
    m = _fresh(flush_every=1)
    with _in_tmp() as root:
        allp, binp, _ = _ckpt_paths(m, root)

        acc = {"bidirectional": _cands([42], score=0.40)}
        _run(m, acc)
        assert _pair_status(m, allp, binp) == "consistent"

        # transaction 2: SAME seed, higher score, a DISTINCT trial (the replay
        # key is (seed, trial_number, skip_mode) — one key with two contents is
        # corruption, not a new transaction). Crash after the first replace.
        m._flush_last_count = 0
        proxy = _OsProxy(fail_replace_on=2)
        m._os_flush = proxy
        acc["bidirectional"] = _cands([42], score=0.90, trial=2)
        _out, err = _run(m, acc)
        assert proxy.replace_calls == 2, "the mixed state was not produced"
        assert "ERROR" in err

        # the pair really is mixed, and the seed sets really are identical
        with np.load(allp) as z:
            a_score = max(float(v) for v in z["score"])
        with np.load(binp) as z:
            b_score = max(float(v) for v in z["score"])
        assert abs(a_score - 0.90) < 1e-6, a_score
        assert abs(b_score - 0.40) < 1e-6, b_score
        assert _seeds_on_disk(allp) == _seeds_on_disk(binp) == [42], (
            "this gate requires identical seed sets, or it does not test "
            "Beta's counterexample at all")

        # THE POINT: seed-set comparison sees agreement across a mixed pair
        seed_set_says_agree = _seeds_on_disk(allp) == _seeds_on_disk(binp)
        assert seed_set_says_agree, "the premise of the blocker has changed"

        # ...and the production detector does not. It reports the A-FIRST crash
        # (row 4), which is a DIFFERENT answer from "consistent" — reached
        # through the identity block, never through the seed sets.
        status = _pair_status(m, allp, binp)
        assert status == _ck.ROW_A_NEWER, (
            f"a mixed pair with identical seed sets was classified {status!r}, "
            f"not the A-first crash — seed-set comparison is not sufficient and "
            f"this is exactly Beta's counterexample")

    # ── the match-rate-only case (same hole) ────────────────────────────────
    m2 = _fresh(flush_every=1)
    with _in_tmp() as root:
        allp, binp, _ = _ckpt_paths(m2, root)
        acc = {"bidirectional": _cands([7], score=0.5)}
        acc["bidirectional"][0]["forward_match_rate"] = 0.10
        acc["bidirectional"][0]["reverse_match_rate"] = 0.10
        _run(m2, acc)
        m2._flush_last_count = 0
        m2._os_flush = _OsProxy(fail_replace_on=2)
        nxt = _cands([7], score=0.5, trial=2)
        nxt[0]["forward_match_rate"] = 0.95
        nxt[0]["reverse_match_rate"] = 0.95
        acc["bidirectional"] = nxt
        _run(m2, acc)
        assert _seeds_on_disk(allp) == _seeds_on_disk(binp) == [7]
        assert _pair_status(m2, allp, binp) == _ck.ROW_A_NEWER, (
            "a mixed pair differing only in match rates was not detected")

    # ── the load outcomes Beta specified ────────────────────────────────────
    m3 = _fresh(flush_every=1)
    with _in_tmp() as root:
        allp, binp, cdir = _ckpt_paths(m3, root)
        _install_context(m3)

        # absent
        assert _pair_status(m3, allp, binp) == "absent"

        # matching identity + digests -> accept
        acc = {"bidirectional": _cands([1, 2, 3])}
        _run(m3, acc)
        assert _pair_status(m3, allp, binp) == "consistent"

        # both members carry the full identity block, and they AGREE on
        # everything a normal installed pair must agree on. `member_role` and
        # `member_content_digest` are EXPECTED to differ — they persist
        # different payloads by design — and that difference is asserted, not
        # merely tolerated.
        ia, _pa, _oa = _ck.read_member(allp)
        ib, _pb, _ob = _ck.read_member(binp)
        for k in _ck.TRANSACTION_INVARIANT_KEYS:
            assert ia[k] == ib[k], f"identity field {k!r} differs across members"
        assert ia["member_content_digest"] != ib["member_content_digest"], (
            "the two members report the SAME member_content_digest; A is a "
            "marker stub and B carries the state, so equal digests mean one of "
            "them is not what it claims")
        assert ia["member_role"] == _ck.MEMBER_A_ROLE
        assert ib["member_role"] == _ck.MEMBER_B_ROLE
        assert ia["checkpoint_schema_version"] == m3._CHECKPOINT_SCHEMA_VERSION
        assert ia["logical_candidate_count"] == 3, ia
        assert ia["run_id"] == m3._flush_run_id()
        assert isinstance(ia["checkpoint_sequence"], int)

        # a tampered member fails its OWN digest under an unchanged identity
        tampered = os.path.join(cdir, "tampered.npz")
        with np.load(binp) as z:
            payload = {k: z[k] for k in z.files}
        payload["score"] = payload["score"] + np.float32(0.25)
        with open(tampered, "wb") as fh:
            np.savez_compressed(fh, **payload)
        os.replace(tampered, binp)
        assert _pair_status(m3, allp, binp).startswith("failed:"), (
            "a tampered member B under one transaction identity was not "
            "reported as unrecoverable — B is the SOLE recovery payload")

        # one unreadable member -> B lost -> fail closed regardless of A
        with open(binp, "wb") as fh:
            fh.write(b"truncated garbage")
        assert _pair_status(m3, allp, binp).startswith("failed:")

        # neither valid -> recovery fails visibly, in-memory untouched
        with open(allp, "wb") as fh:
            fh.write(b"also garbage")
        acc_before = _cands([1, 2, 3])
        acc2 = {"bidirectional": list(acc_before)}
        assert _pair_status(m3, allp, binp).startswith("failed:")
        assert acc2["bidirectional"] == acc_before, (
            "inspecting an unrecoverable pair touched the in-memory records")

    # ── a pre-D6.2 member with no identity block is REFUSED, not guessed ────
    m4 = _fresh(flush_every=1)
    with _in_tmp() as root:
        allp, binp, cdir = _ckpt_paths(m4, root)
        os.makedirs(cdir, exist_ok=True)
        _install_context(m4)
        for p in (allp, binp):
            with open(p, "wb") as fh:
                np.savez_compressed(fh, seeds=np.array([1], dtype=np.uint64),
                                    score=np.array([0.5], dtype=np.float32))
        assert _pair_status(m4, allp, binp).startswith("failed:"), (
            "a member with no identity block was accepted — an unversioned "
            "file must be refused, not interpreted")

        # ...and so is a member stamped with the D6.1 FOUR-FIELD marker. That is
        # what the schema version is for: D6.2 must be able to tell the interim
        # format apart at a glance rather than mis-decode it.
        with np.load(binp) as z:
            pass
        legacy = {"seeds": np.array([1], dtype=np.uint64),
                  "score": np.array([0.5], dtype=np.float32),
                  "checkpoint_schema_version": np.array(
                      "s172-d6.1-four-field-v1"),
                  "checkpoint_id": np.array("x"),
                  "checkpoint_sequence": np.array(1, dtype=np.int64),
                  "run_id": np.array(m4._flush_run_id()),
                  "logical_candidate_count": np.array(1, dtype=np.int64),
                  "four_field_content_digest": np.array("0" * 64)}
        with open(binp, "wb") as fh:
            np.savez_compressed(fh, **legacy)
        assert _pair_status(m4, allp, binp).startswith("failed:"), (
            "a D6.1 four-field member was accepted by the D6.2 reader")


# ═════════════════════════════════════════════════════════════════════════════
# G-PATH-CONDITIONS   [Beta's six .s172_checkpoint/ conditions]
# ═════════════════════════════════════════════════════════════════════════════
def g_path_conditions(mod=None):
    """Git-ignored · CWD-independent · run-isolated · same-filesystem ·
    never a finalizer alias · explicitly versioned."""
    mod = _fresh(flush_every=1) if mod is None else mod

    # (1) git-ignored
    gi = _read(os.path.join(_ROOT, ".gitignore"))
    assert ".s172_checkpoint/" in gi, (
        ".s172_checkpoint/ is not git-ignored — snapshot state would be "
        "committable and would trip Phase-4 gate 22")

    # (2) NOT dependent on the process CWD
    with _in_tmp() as root:
        before = mod._flush_checkpoint_dir()
        sub = os.path.join(root, "deep", "nested")
        os.makedirs(sub, exist_ok=True)
        cwd = os.getcwd()
        try:
            os.chdir(sub)
            after = mod._flush_checkpoint_dir()
        finally:
            os.chdir(cwd)
        assert before == after, (
            f"the snapshot directory moved when the process chdir'd "
            f"({before} -> {after}) — it must resolve from a stable root")

        # and a flush after chdir lands in the SAME directory
        allp, _binp, _ = _ckpt_paths(mod, root)
        acc = {"bidirectional": _cands([1, 2])}
        try:
            os.chdir(sub)
            _run(mod, acc)
        finally:
            os.chdir(cwd)
        assert os.path.isfile(allp), (
            "a flush issued from a different CWD did not land in the stable "
            "snapshot directory")

    # (3) run-isolated: two run ids never share a directory
    with _in_tmp(run_id="run-A") as root:
        a_dir = mod._flush_checkpoint_dir()
        acc = {"bidirectional": _cands([1, 2])}
        mod._flush_last_count = 0
        _run(mod, acc)
        with _in_tmp(run_id="run-B"):
            pass
    assert "run-A" in a_dir, (
        f"the snapshot directory {a_dir} carries no run identity — "
        f"consecutive or concurrent runs could collide")

    m_a, m_b = _fresh(flush_every=1), _fresh(flush_every=1)
    with tempfile.TemporaryDirectory() as shared:
        shared = os.path.realpath(shared)
        prev = os.environ.get("PRNG_CHECKPOINT_ROOT")
        os.environ["PRNG_CHECKPOINT_ROOT"] = shared
        try:
            os.environ["PRNG_CHECKPOINT_RUN_ID"] = "run-A"
            da = m_a._flush_checkpoint_dir()
            os.environ["PRNG_CHECKPOINT_RUN_ID"] = "run-B"
            db = m_b._flush_checkpoint_dir()
        finally:
            os.environ.pop("PRNG_CHECKPOINT_RUN_ID", None)
            if prev is None:
                os.environ.pop("PRNG_CHECKPOINT_ROOT", None)
            else:
                os.environ["PRNG_CHECKPOINT_ROOT"] = prev
    assert da != db, (
        f"two runs sharing one root resolved to the same snapshot directory "
        f"({da}) — they would overwrite each other")

    # (4) temp and destination in the same directory / filesystem
    with _in_tmp() as root:
        allp, binp, cdir = _ckpt_paths(mod, root)
        for final in (allp, binp):
            tmp = mod._flush_tmp_name(final)
            assert os.path.dirname(tmp) == os.path.dirname(final), (
                f"temp {tmp} is not in the destination directory — os.replace "
                f"would not be atomic")
        os.makedirs(cdir, exist_ok=True)
        mod._flush_assert_same_filesystem(mod._flush_tmp_name(allp), allp)
        # ...and the guard actually fires when they differ
        try:
            mod._flush_assert_same_filesystem("/tmp/elsewhere.tmp", allp)
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "the same-filesystem guard did not fire for a temp in another "
                "directory")

    # (5) never resolves to a finalizer alias — fail closed, not by convention
    for alias in ("bidirectional_survivors_all.npz",
                  "bidirectional_survivors_binary.npz"):
        try:
            mod._flush_assert_not_alias(f"/anywhere/{alias}")
        except RuntimeError:
            continue
        raise AssertionError(
            f"the alias guard accepted {alias} — the finalizer fails closed on "
            f"a regular file at that path")

    # (6) the schema version is explicit and carried IN the artifact
    assert isinstance(mod._CHECKPOINT_SCHEMA_VERSION, str)
    assert mod._CHECKPOINT_SCHEMA_VERSION, "the schema version is empty"
    # [D6.2] REV5 §3.3: "checkpoint_schema_version — UPDATE — the four-field
    # marker must change". The version is now imported from
    # `utils/checkpoint_d6_2` rather than restated here, so there is exactly one
    # authority for the marker that tells the two formats apart.
    assert "four-field" not in mod._CHECKPOINT_SCHEMA_VERSION, (
        f"the schema version is still {mod._CHECKPOINT_SCHEMA_VERSION!r} — the "
        f"D6.1 four-field marker MUST change now that the payload carries all "
        f"24 canonical fields, or a D6.1 member and a D6.2 member would be "
        f"indistinguishable")
    import utils.checkpoint_d6_2 as _ck_ver
    assert mod._CHECKPOINT_SCHEMA_VERSION == _ck_ver.CHECKPOINT_SCHEMA_VERSION, (
        "the flush section restates the schema version instead of importing it")
    with _in_tmp() as root:
        allp, binp, _ = _ckpt_paths(mod, root)
        mod._flush_last_count = 0
        _run(mod, {"bidirectional": _cands([1, 2])})
        for p in (allp, binp):
            ident, _payload, _order = _ck_ver.read_member(p)
            assert ident["checkpoint_schema_version"] == \
                mod._CHECKPOINT_SCHEMA_VERSION


# ═════════════════════════════════════════════════════════════════════════════
# G-NO-ALIAS-COLLISION   [D6.1 — the defect the brief did not anticipate]
# ═════════════════════════════════════════════════════════════════════════════
def g_no_alias_collision(mod=None):
    """The checkpoint never writes the finalizer-owned root paths.

    Since D3.5, `bidirectional_survivors_all.npz` and `..._binary.npz` in the
    run root are compatibility SYMLINKS owned by
    `run_finalizer._bootstrap_root_aliases`, which fails closed if a regular
    file appears at either. Repairing D1 without relocating the checkpoint
    would have replaced both symlinks with regular 4-array files and made every
    subsequent finalization raise PublicationError.
    """
    from utils import run_finalizer as RF

    mod = _fresh(flush_every=2) if mod is None else mod
    ROOT_NAMES = ("bidirectional_survivors_all.npz",
                  "bidirectional_survivors_binary.npz")
    # the oracle is only meaningful if these really are the finalizer's names
    assert RF.ALL_NPZ_NAME == ROOT_NAMES[0], RF.ALL_NPZ_NAME
    assert RF.BINARY_NPZ_NAME == ROOT_NAMES[1], RF.BINARY_NPZ_NAME

    # (1) the checkpoint targets are NOT the finalizer's names
    assert mod._CHECKPOINT_ALL_NAME not in ROOT_NAMES, (
        f"the checkpoint writes {mod._CHECKPOINT_ALL_NAME} — a "
        f"finalizer-owned path")
    assert mod._CHECKPOINT_BINARY_NAME not in ROOT_NAMES
    assert mod._CHECKPOINT_DIRNAME != "", "the checkpoint has no namespace"

    # (2) end to end: with the aliases in place, flushes leave them untouched
    #     and the finalizer still accepts them afterwards
    with _in_tmp() as root:
        rootp = Path(root)
        RF._bootstrap_root_aliases(rootp)
        for n in ROOT_NAMES:
            assert os.path.islink(rootp / n), f"{n} was not bootstrapped"

        m = _fresh(flush_every=1)
        acc = {"bidirectional": _cands([1, 2, 3])}
        _run(m, acc)
        acc["bidirectional"] = _cands([4, 5, 6])
        m._flush_last_count = 0
        _run(m, acc)

        allp, _binp, _ = _ckpt_paths(m, root)
        assert os.path.isfile(allp), "the flush did not land its checkpoint"

        for n in ROOT_NAMES:
            p = rootp / n
            assert os.path.islink(p), (
                f"the flush replaced the finalizer-owned symlink {n} with a "
                f"regular file — the next finalize_run would raise "
                f"PublicationError and generation publication would be "
                f"permanently broken")

        # the finalizer's own fail-closed check still passes
        RF._bootstrap_root_aliases(rootp)          # must not raise

    # (3) the failure this gate exists to prevent is REAL: a regular file at
    #     those paths does make the finalizer fail closed
    with _in_tmp() as root:
        rootp = Path(root)
        with open(rootp / ROOT_NAMES[0], "wb") as fh:
            fh.write(b"regular file")
        try:
            RF._bootstrap_root_aliases(rootp)
        except RF.PublicationError:
            pass
        else:
            raise AssertionError(
                "the finalizer no longer fails closed on a regular file at an "
                "alias path — this gate's premise has changed")


# ═════════════════════════════════════════════════════════════════════════════
# G-COMPRESSION-CONTRACT
# ═════════════════════════════════════════════════════════════════════════════
def g_compression_contract(mod=None):
    """The CHECKPOINT may be compressed AND D5's ARTIFACT ban is untouched.

    Two genuinely different contracts. The hazard is conflation, so this gate
    proves they remain separate rather than harmonising them.
    """
    # (1) the checkpoint IS compressed — deliberate, for a file rewritten every
    #     _FLUSH_EVERY survivors
    m = _fresh(flush_every=2) if mod is None else mod
    with _in_tmp() as root:
        allp, binp, _ = _ckpt_paths(m, root)
        acc = {"bidirectional": _cands([1, 2, 3])}
        _run(m, acc)
        for p in (allp, binp):
            with zipfile.ZipFile(p) as zf:
                types = {i.filename: i.compress_type for i in zf.infolist()}
            assert types and all(t == zipfile.ZIP_DEFLATED
                                 for t in types.values()), (
                f"the checkpoint is not compressed: {types}")

    # (2) D5's ARTIFACT ban is untouched: the writer still calls `np.savez`,
    #     and M6a's anchor text is intact
    asw_src = _read(os.path.join(_ROOT, "miner", "assembly_shard_worker.py"))
    assert asw_src.count("np.savez(fh, **payload)") == 1, (
        "D5's M6a mutation anchor `np.savez(fh, **payload)` is gone — the "
        "artifact codec changed, or M6a can no longer be applied")
    assert "np.savez_compressed(" not in asw_src, (
        "the artifact writer now compresses — D5 §6.7.A forbids it and M6a "
        "reds on compress_type=8")

    # (3) and the artifact really is stored uncompressed, by D5's own rule
    import miner.range_miner_npz_writer as RMW
    import miner.assembly_shard_worker as ASW
    proj = RMW.build_validated_projection(
        [5, 9, 1], np.array([0.1, 0.2, 0.3], dtype=np.float64))
    # built through the PRODUCTION identity builder so it cannot drift
    identity = ASW._identity_from(
        {"stripe_id": "s0", "sub_index": 0, "attempt": 0,
         "trial_metadata": {"workflow_phase": 1, "direction": "forward",
                            "skip_mode": "constant", "prng_type": "java_lcg"}},
        "d6_1-compression-probe")
    with _in_tmp() as root:
        apath = os.path.join(root, "artifact.npz")
        ASW.write_projection_artifact(apath, proj, identity)
        with zipfile.ZipFile(apath) as zf:
            for info in zf.infolist():
                assert info.compress_type == zipfile.ZIP_STORED, (
                    f"{info.filename} is compressed "
                    f"(compress_type={info.compress_type}); §6.7.A requires "
                    f"np.savez, not savez_compressed")

    # (4) the distinction is DOCUMENTED at the checkpoint, so nobody
    #     "harmonizes" the two and reds a D5 gate
    integ = _read(_INTEG_FULL)
    assert "§6.7.A" in integ and "M6a" in integ, (
        "the checkpoint no longer documents why its compression contract "
        "differs from D5's artifact ban — that comment is the only thing "
        "stopping a well-meaning harmonisation")


# ═════════════════════════════════════════════════════════════════════════════
# G-COMMENT-TRUTH  (D4)
# ═════════════════════════════════════════════════════════════════════════════
def g_comment_truth(mod=None):
    """The S166 comment no longer asserts a guarantee that does not hold."""
    integ = _read(_INTEG_FULL)
    for forbidden, why in (
        ("data is safe in NPZ",
         "the S166 comment's guarantee has never held (D4)"),
        ("DETECTABLE (the two seed sets differ)",
         "seed-set comparison does NOT detect a mixed pair — Beta's "
         "counterexample disproves it (see G-TRANSACTION-IDENTITY)"),
    ):
        assert forbidden not in integ, (
            f"the code still claims {forbidden!r} — {why}; a load-bearing "
            f"comment documenting a false guarantee is itself the D4 defect")

    # the replacement states the properties that DO hold, and no more
    #
    # [D6.2] Four D6.1 phrases are RETIRED because the facts they stated are no
    # longer true, and a comment restating a superseded fact is the same D4
    # defect in the other direction:
    #   "NON-AUTHORITATIVE" / "not a canonical accumulator checkpoint" — the
    #       checkpoint IS canonical now; it carries all 24 fields;
    #   "PROVISIONAL SNAPSHOT MAINTENANCE ONLY" — the merge is real
    #       reconciliation ending in the frozen `_select_l2_winners`;
    #   "THE IN-MEMORY LIST REMAINS THE FINALIZER'S AUTHORITATIVE SOURCE" — the
    #       finalizer is now fed the reconstructed cumulative state, because the
    #       in-memory list is cleared.
    # What is still TRUE, and still required, is everything below.
    for required in ("SEQUENTIAL-ATOMIC WITH SELF-REPAIR",
                     "explicitly NOT jointly atomic",
                     "_FLUSH_CLEAR_IN_MEMORY",
                     "TRANSACTION IDENTITY",
                     "MARKER / COMPATIBILITY STUB",
                     "SOLE RECOVERY PAYLOAD"):
        assert required in integ, (
            f"the repaired documentation does not state {required!r}")

    # the retired claims must be GONE, not merely unasserted
    for retired, why in (
        ("NON-AUTHORITATIVE, FOUR-FIELD INCREMENTAL SNAPSHOT",
         "the checkpoint now carries all 24 canonical fields"),
        ("PROVISIONAL SNAPSHOT MAINTENANCE ONLY",
         "the merge is reconciliation through the frozen L2 selector"),
        ("THE IN-MEMORY LIST REMAINS THE FINALIZER'S AUTHORITATIVE SOURCE",
         "the finalizer is fed the reconstructed cumulative state"),
    ):
        assert retired not in integ, (
            f"the code still claims {retired!r} — {why}, so that statement is "
            f"now false and is the D4 defect in the other direction")

    # and the scope disclaimer that STILL holds
    for required in ("does not restore the optimizer execution cursor",):
        assert required in integ, (
            f"the code does not disclaim {required!r} — D6.2 must not be read "
            f"as providing it")


# ═════════════════════════════════════════════════════════════════════════════
# MUTATION PROOF
# ═════════════════════════════════════════════════════════════════════════════
def g_mutants():
    src = _flush_section_src()

    def _kill2(label, mutated_src, marker, gate, credited):
        """Run `gate` with the MUTATED section active, and require it to red.

        `_active` is what makes this honest: every module the gate builds
        internally via `_fresh()` comes from `mutated_src`, so the mutated path
        is the one actually executed — part 2 of the four-part rule. Part 3 is
        the positive control immediately below, which requires the same gate to
        PASS against the unmutated section, so the kill is attributable.
        """
        _positive_control(f"{label} detector", gate)

        def _run():
            with _active(mutated_src):
                gate()
        _record(label, _run, credited, marker, mutated_src)

    # ── M1: restore the un-suffixed temp NAME passed to savez_compressed ────
    old_write = ("    with open(tmp_path, \"wb\") as _fh:\n"
                 "        _np_flush.savez_compressed(_fh, **arrays)\n"
                 "        _fh.flush()\n"
                 "        _os_flush.fsync(_fh.fileno())")
    new_write = "    _np_flush.savez_compressed(tmp_path, **arrays)"
    _kill2("M1 temp name is suffix-rewritten by NumPy",
           _patch(src, old_write, new_write, "M1"),
           "savez_compressed(tmp_path", g_suffix, "G-SUFFIX")

    # ── M2: move the list-clear BEFORE the write transaction ────────────────
    # [D6.2] RE-ANCHORED. The two `os.replace` calls moved into
    # `utils.checkpoint_d6_2.write_transaction`, which also validates the
    # installed pair before returning; the clear is the step after it. Hoisting
    # the clear above that call is the same injected defect — candidates dropped
    # before the checkpoint is known to have landed — and G-CLEAR-AFTER must
    # still red on it.
    old_order = ("        # ── §8 steps 2-7: write, validate, replace A, "
                 "replace B, validate ────\n"
                 "        _txn = _write_checkpoint_transaction(")
    new_order = ("        if _FLUSH_CLEAR_IN_MEMORY:\n"
                 "            accumulator[\"bidirectional\"] = []\n"
                 "        _txn = _write_checkpoint_transaction(")
    old_order_present = old_order
    _kill2("M2 list-clear moved before the write transaction",
           _patch(src, old_order, new_order, "M2"),
           "accumulator[\"bidirectional\"] = []\n        _txn =",
           g_clear_after, "G-CLEAR-AFTER")

    # ── M3: clear the list on a FAILED write ────────────────────────────────
    old_fail = ("        print(f\"[S172-D6.2-CHECKPOINT] ERROR: checkpoint write "
                "FAILED \"")
    new_fail = ("        accumulator[\"bidirectional\"] = []\n"
                "        print(f\"[S172-D6.2-CHECKPOINT] ERROR: checkpoint write "
                "FAILED \"")
    _kill2("M3 candidates cleared on a failed write",
           _patch(src, old_fail, new_fail, "M3"),
           "accumulator[\"bidirectional\"] = []\n        print",
           g_retain_on_fail, "G-RETAIN-ON-FAIL")

    # ── M4: leave the temp files behind ─────────────────────────────────────
    _kill2("M4 temp files are never removed",
           _patch(src, "        _flush_remove_temps(_tmp, _tmp_bin)",
                  "        pass  # temps deliberately leaked", "M4"),
           "temps deliberately leaked", g_no_temp_leak, "G-NO-TEMP-LEAK")

    # ── M5: re-broaden the exception handler to swallow everything ──────────
    old_exc = "    except OSError as _fe:"
    new_exc = ("    except Exception as _fe:\n"
               "        print(f\"[S172-D6.2-CHECKPOINT] Warning: checkpoint \"\n"
               "              f\"failed (non-fatal): {_fe}\")\n"
               "        return\n"
               "    except OSError as _fe:")
    _kill2("M5 exception handler swallows the failure again",
           _patch(src, old_exc, new_exc, "M5"),
           "Warning: checkpoint ", g_visible_failure,
           "G-VISIBLE-FAILURE")

    # ── M6: drop the CUMULATIVE state from the reconciliation ───────────────
    # [D6.2] RE-ANCHORED. D6.1 carried seeds forward by re-reading member A from
    # disk and merging it; D6.2 never does that (A is a MARKER STUB, never an
    # accumulator backup) and carries them in the run context's cumulative
    # state instead. Dropping that input is the same injected defect — earlier
    # candidates silently disappearing from the checkpoint — and G-CUMULATIVE
    # must still red on it.
    _kill2("M6 cumulative state dropped from reconciliation",
           _patch(src, "        _cumulative = _reconcile_candidates(_ctx.cumulative, _new_records)",
                  "        _cumulative = _reconcile_candidates([], _new_records)",
                  "M6"),
           "_reconcile_candidates([], _new_records)", g_cumulative,
           "G-CUMULATIVE")

    # ── M7: point the checkpoint back at the finalizer-owned root names ─────
    # [D6.2] RE-ANCHORED. The three names are IMPORTED from
    # `utils/checkpoint_d6_2` rather than assigned here, so the mutant overrides
    # them after the import instead of editing an assignment that no longer
    # exists. The injected defect is identical: the checkpoint aimed at the two
    # finalizer-owned root aliases, which `finalize_run` fails closed on.
    m7_src = _patch(
        src, "_CHECKPOINT_TMP_SUFFIX  = \".flush-{pid}.tmp\"",
        "_CHECKPOINT_DIRNAME     = \".\"\n"
        "_CHECKPOINT_ALL_NAME    = \"bidirectional_survivors_all.npz\"\n"
        "_CHECKPOINT_BINARY_NAME = \"bidirectional_survivors_binary.npz\"\n"
        "_CHECKPOINT_TMP_SUFFIX  = \".flush-{pid}.tmp\"", "M7")
    _kill2("M7 checkpoint writes the finalizer-owned root paths", m7_src,
           "_CHECKPOINT_ALL_NAME    = \"bidirectional_survivors_all.npz\"",
           g_no_alias_collision, "G-NO-ALIAS-COLLISION")

    # ── M8: the fail-closed run-context guard removed ───────────────────────
    # [D6.2] REPLACED, and the reason is stated rather than the check quietly
    # dropped. D6.1's M8 reverted `_flush_inspect_pair` to a SEED-SET-ONLY
    # comparison — Beta's blocker. That function no longer exists: D6.2 moved
    # pair classification into `utils.checkpoint_d6_2.recover_checkpoint`, which
    # is a different module and therefore outside this suite's sliced unit. The
    # blocker's property is still gated — G-TRANSACTION-IDENTITY above asserts
    # it against the live matrix — and the MUTANT for it now lives in
    # `tests/test_s172_d6_2_checkpoint_reconciliation.py` (M9 "recover the newer
    # A instead of B", M18 "recovery row 5 deleted"), where the code it mutates
    # actually is.
    #
    # What replaces it here is the defect this suite CAN still inject: removing
    # the fail-closed guard that refuses to write or clear without a run
    # context. An absent context is not a neutral "unknown" — it means nobody
    # established the run identity, the declared seed interval or the
    # `run_context_digest`, so the three walls cannot run and the checkpoint
    # could never be verified on resume.
    old_guard = ("    _ctx = _active_flush_run_context()\n"
                 "    if _ctx is None:\n"
                 "        # FAIL CLOSED, LOUDLY, AND CLEAR NOTHING.")
    new_guard = ("    _ctx = _active_flush_run_context()\n"
                 "    if False:  # M8 — the fail-closed guard is removed\n"
                 "        # FAIL CLOSED, LOUDLY, AND CLEAR NOTHING.")
    _kill2("M8 the fail-closed run-context guard removed",
           _patch(src, old_guard, new_guard, "M8"),
           "if False:  # M8", g_visible_failure, "G-VISIBLE-FAILURE")


# ═════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═════════════════════════════════════════════════════════════════════════════
def _check(name, fn):
    try:
        fn()
    except Exception as exc:                                    # noqa: BLE001
        import traceback
        _CHECKS.append((name, False, f"{type(exc).__name__}: {exc}"))
        print(f"  [\033[91mFAIL\033[0m] {name}")
        traceback.print_exc()
        return
    _CHECKS.append((name, True, ""))
    print(f"  [\033[92mPASS\033[0m] {name}")


def main():
    print("=" * 78)
    print("S172 Phase-5 D6.1 — incremental NPZ atomic flush & durability")
    print("=" * 78)

    _check("G-SUFFIX: the temp target cannot be .npz-rewritten by NumPy",
           g_suffix)
    _check("G-ATOMIC-ACCUM/BINARY: complete prior or complete new, never partial",
           g_atomic)
    _check("G-CLEAR-AFTER: the list clears only after BOTH replaces succeed",
           g_clear_after)
    _check("G-RETAIN-ON-FAIL: zero candidate loss at four injection points",
           g_retain_on_fail)
    _check("G-NO-TEMP-LEAK: no temp survives success or any failure",
           g_no_temp_leak)
    _check("G-CUMULATIVE: exact cumulative counts, dedup + prior merge intact",
           g_cumulative)
    _check("G-CRASH-RESTART: three crash points, detectable + self-repairing",
           g_crash_restart)
    _check("G-TRANSACTION-IDENTITY: mixed pair caught when seed sets match",
           g_transaction_identity)
    _check("G-PATH-CONDITIONS: ignored, CWD-free, run-isolated, same-fs, versioned",
           g_path_conditions)
    _check("G-CADENCE: entry gating unchanged; pins SUCCESSFUL flush",
           g_cadence)
    _check("G-VISIBLE-FAILURE: failures are surfaced, tiered and counted",
           g_visible_failure)
    _check("G-NO-ALIAS-COLLISION: finalizer-owned root paths never written",
           g_no_alias_collision)
    _check("G-COMPRESSION-CONTRACT: checkpoint compressed, D5 artifact ban intact",
           g_compression_contract)
    _check("G-COMMENT-TRUTH: the false S166 guarantee is gone (D4)",
           g_comment_truth)
    _check("G-MUTANTS: mutation proof (8 mutants, four-part rule)", g_mutants)

    print("\n" + "-" * 78)
    print("MUTANTS")
    print("-" * 78)
    for label, sig, credited, parts in _MUTANTS:
        print(f"  KILLED  {label}")
        print(f"          credited to {credited}")
        print(f"          {sig}")
        print(f"          {parts}")

    passed = sum(1 for _n, ok, _e in _CHECKS if ok)
    total = len(_CHECKS)
    print("\n" + "=" * 78)
    for name, ok, err in _CHECKS:
        if not ok:
            print(f"  FAILED: {name}\n          {err}")
    print(f"{passed}/{total} D6.1 gate checks green  ({len(_MUTANTS)} mutants killed)")
    if passed == total:
        print("All D6.1 gate checks green (ported to the D6.2 payload) — the "
              "checkpoint writes,\nreplaces atomically per file, detects an "
              "interrupted replacement by transaction\nidentity, cleans up its "
              "temps on every path, is loud on failure, is isolated from\nthe "
              "finalizer, and clears the in-memory list strictly last.\n"
              "Payload correctness (24 fields, both digests, CSR sessions, "
              "reconciliation,\nresume and the nine-row recovery matrix) is "
              "gated by\ntests/test_s172_d6_2_checkpoint_reconciliation.py.")
    print("=" * 78)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
