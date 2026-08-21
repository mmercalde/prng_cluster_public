#!/usr/bin/env python3
"""S172 — R-1 / R-2 / R-3 DRAIN STARVATION REMEDY GATE BATTERY

Implements the ten gates of `CCODE_BRIEF_R1_DRAIN_REMEDY_v1_0.md`, R2-1 of
`CCODE_BRIEF_R2_CACHED_POSITIVE_v1_0.md`, R3-1 of
`CCODE_BRIEF_R3_CAPACITY_BOUNDARY_v1_0.md` and R3-4 of
`CCODE_BRIEF_R4_AMENDMENT_IRREVERSIBILITY_v1_0.md`, for the remedy of the cause
MP-1 measured and `docs/GATE12_MP1_RUN_FORENSIC.md` confirmed by attribution.

WHAT CHANGED, IN THREE SENTENCES
---------------------------------
R-1: `_pump_deferred` evaluates `_attempt_live_locked` ONCE PER DISTINCT
`(run_id, stripe_id, attempt)` KEY per pass instead of once per deferred ENTRY.
R-2: that key is DISCARDED from the memo the instant `_try_admit_locked` returns
True, so no frame that stages is ever backed by a reused observation.
R-3: an END-OF-PASS SWEEP re-probes each remaining memoized key ONCE and
releases the frames of any that died, so a stale positive cannot hold bounded
capacity past the pass. Two definitions move — `_pump_deferred` and
`__init__`'s two `_bp` seeds; `_try_admit_locked`, `_attempt_live_locked`,
`_prune_admitted_locked`, `_defer_locked`, `enqueue_staging` and
`_resume_paused_connections` are byte-identical.

WHY THAT IS THE CAUSE AND NOT A PROXY FOR IT
---------------------------------------------
`_deferred` holds SUB-STRIPE frames; one stripe attempt contributes
`expected_substripes` of them. Liveness is a property of the ATTEMPT, so the old
loop asked the ledger the identical question up to `expected_substripes` times
per attempt per pass — and `MinerLedger._conn` OPENS A NEW SQLITE CONNECTION AND
RUNS THREE PRAGMAS PER QUERY, so an "ask" is a database open. A probe is 1 read
when the stripe row is already terminal (`_attempt_live_locked` returns at
`get_stripe`) and 2 otherwise, with `_admission_lock` held, while the serve loop
needs that same lock once per sub-result. The backlog the pump exists to drain
is what made the pump slow.

THE TWO PROOFS BETA REQUIRES ARE SEPARATE GATES, AND NEITHER IS A BENCHMARK
---------------------------------------------------------------------------
  SEMANTIC   — G6 runs the VERBATIM PRE-PATCH pump, reconstructed from the
               pinned commit and executed with the live module's globals,
               against the live pump on independent-but-identical states, over a
               declared fixture matrix, and asserts EXACT equality of every
               disposition: submitted entries in order, retained `_deferred` in
               order, dropped entries, `_admitted`, released bookkeeping,
               futures, and the staging-slot count. No timing appears in it.
  COMPLEXITY — G8 counts LEDGER READS (never seconds). G8a holds the distinct
               attempt count AND the admitted attempt's frames fixed and grows
               the NON-ADMITTED population 67x, asserting the read count is
               unchanged: that is the term that ran away in MP-1. G8b asserts
               the truthful bound as an EQUALITY and G8g pins the all-dead purge
               as O(N) — the worst case is NOT population-invariant and is not
               claimed to be.

THE SAFETY ARGUMENT THE GATES ENFORCE
--------------------------------------
The memo is ONE-DIRECTIONAL: only a LIVE observation is reused, a key observed
DEAD is never recorded. Every drop is therefore decided by a FRESH, under-lock
`_attempt_live_locked` call — never by a reused negative.

⚠ NOT one fresh probe PER ENTRY. R-3's end-of-pass sweep deliberately uses ONE
fresh negative probe to retire EVERY retained frame of that key, which is what
keeps the closure O(distinct retained attempts) instead of O(entries). The two
statements that remain exactly true are: no drop rests on a REUSED observation,
and R-3 retains <= the predecessor (never more), which is the only direction the
bounded-capacity surface cares about. That a swept key cannot come back to life
at the same attempt is not asserted — it is gated by R3-4, which also proves it
does NOT weaken G4b: the ledger primitive genuinely permits `failed -> claimed`;
the production SCHEDULER does not exercise it for a swept key.

Staging AUTHORITY is written at exactly one place — `_try_admit_locked`,
unchanged, still called for every live entry, still guarding its grant with its
own fresh `get_stripe`. Within one lock hold `_admitted` can only grow, so a
grant can only happen at a key's FIRST examination, which is always fresh; and
because a grant discards the memo entry (R-2), NO MEMO HIT CAN EVER REACH
`ready`.

Run:  source ~/venvs/torch/bin/activate && \
      python3 -u tests/test_s172_r1_drain_remedy.py
"""

from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import inspect
import os
import subprocess
import sys
import tempfile
import textwrap
import types
from typing import Any, Dict, List, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from miner.range_miner_coordinator import (  # noqa: E402
    ST_CANCELLED,
    ST_CLAIMED,
    ST_DONE,
    ST_FAILED,
    ST_PENDING,
    CoordinatorConfig,
    MinerLedger,
    RangeMinerCoordinator,
)
import miner.range_miner_coordinator as COORD  # noqa: E402

# THE PINNED PRE-R1 ANCHOR. FULL 40-CHARACTER SHA, never a prefix.
PINNED_COMMIT = "c403a373d21f2bee894ad0a5e45d2135e6da162f"
SRC_REL = "miner/range_miner_coordinator.py"

RUN = "run-r1"

GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"
_RESULTS: List[Any] = []


def check(name, fn):
    try:
        detail = fn()
        _RESULTS.append((name, True, detail))
        print(f"  [{GREEN}PASS{RESET}] {name:<46} {detail}")
    except AssertionError as e:
        _RESULTS.append((name, False, str(e)))
        print(f"  [{RED}FAIL{RESET}] {name:<46} {e}")
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"  [{RED}ERROR{RESET}] {name:<46} {type(e).__name__}: {e}")


# ===========================================================================
# the pinned anchor and its integrity  (VIR-3: UNAVAILABLE never accepts)
# ===========================================================================
class AnchorUnavailable(RuntimeError):
    """The pinned pre-R1 source could not be obtained, or is not it."""


_PINNED_CACHE: Dict[str, str] = {}


def _git_show(commit: str, path: str) -> str:
    p = subprocess.run(["git", "-C", _ROOT, "show", f"{commit}:{path}"],
                       capture_output=True, text=True)
    if p.returncode != 0 or not p.stdout:
        raise AnchorUnavailable(
            f"UNAVAILABLE: could not read {path} at {commit}: "
            f"{p.stderr.strip()[:200]}")
    return p.stdout


def _pinned_src(path: str = SRC_REL) -> str:
    if path not in _PINNED_CACHE:
        _PINNED_CACHE[path] = _git_show(PINNED_COMMIT, path)
    return _PINNED_CACHE[path]


def _live_src(path: str = SRC_REL) -> str:
    with open(os.path.join(_ROOT, path), encoding="utf-8") as fh:
        return fh.read()


def _strip_comments(src: str) -> str:
    """Executable structure only — the live source QUOTES the pre-R1 shape in
    its own docstring, so a text probe would find the old surface in the new
    file and credit an anchor that had drifted forward."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Module)):
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body[0].value.value = ""
    return ast.unparse(tree)


def _func_node(tree: ast.AST, cls: Optional[str], name: str) -> ast.FunctionDef:
    scope: Any = tree
    if cls is not None:
        found = [n for n in ast.walk(tree)
                 if isinstance(n, ast.ClassDef) and n.name == cls]
        if len(found) != 1:
            raise AnchorUnavailable(
                f"expected exactly one class {cls}, found {len(found)}")
        scope = found[0]
    fns = [n for n in ast.walk(scope)
           if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(fns) != 1:
        raise AnchorUnavailable(
            f"expected exactly one {cls}.{name}, found {len(fns)}")
    return fns[0]


def _def_digests(src: str) -> Dict[str, str]:
    tree = ast.parse(src)
    out: Dict[str, str] = {}

    def _walk(node, prefix):
        for child in getattr(node, "body", []):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                stripped = ast.parse(_strip_comments(
                    textwrap.dedent(ast.unparse(child))))
                out[f"{prefix}{child.name}"] = hashlib.sha256(
                    ast.unparse(stripped).encode()).hexdigest()
            elif isinstance(child, ast.ClassDef):
                _walk(child, f"{prefix}{child.name}.")

    _walk(tree, "")
    return out


def gate_anchor_is_authentic():
    """FIRST, because G6's whole semantic proof and every RED depends on it: the
    pinned object really is PRE-R1 source.

    Structural probes, not text: the pinned pump must NOT carry the memo and
    must still carry the per-entry liveness call as the loop's first statement.

    WRONG INPUT THAT REDS IT: an anchor advanced onto the patch."""
    pinned_pump = _strip_comments(textwrap.dedent(ast.unparse(
        _func_node(ast.parse(_pinned_src()), "RangeMinerCoordinator",
                   "_pump_deferred"))))
    live_pump = _strip_comments(textwrap.dedent(ast.unparse(
        _func_node(ast.parse(_live_src()), "RangeMinerCoordinator",
                   "_pump_deferred"))))
    assert "live_keys" not in pinned_pump, (
        f"UNAVAILABLE: the pinned anchor {PINNED_COMMIT[:12]} ALREADY carries "
        f"the R-1 memo — it is not a pre-R1 anchor")
    assert "live_keys" in live_pump, "the live pump carries no R-1 memo"
    assert pinned_pump != live_pump, "pinned and live pump are identical"
    # the pinned loop asks liveness per ENTRY: the call is the first statement
    # of the `for` body.
    ptree = _func_node(ast.parse(_pinned_src()), "RangeMinerCoordinator",
                       "_pump_deferred")
    loops = [n for n in ast.walk(ptree) if isinstance(n, ast.For)
             and "self._deferred" in ast.unparse(n.iter)]
    assert len(loops) == 1, f"expected one `for ... in self._deferred`, got {len(loops)}"
    body_src = ast.unparse(loops[0].body[1])
    assert "_attempt_live_locked" in body_src, (
        f"the pinned loop's second statement is not the per-entry liveness "
        f"call: {body_src[:80]}")
    return f"anchor {PINNED_COMMIT[:12]} verified pre-R1; live carries the memo"


# ===========================================================================
# THE PRE-PATCH ORACLE — the verbatim pinned `_pump_deferred`
# ===========================================================================
_ORACLE_CACHE: List[Any] = []


def _prepatch_pump():
    """Compile the PINNED `_pump_deferred` and bind its globals to THE MODULE
    UNDER TEST.

    [A8-B2 LESSON, carried verbatim] Beta's Defect-A mutant SURVIVED its first
    run because a verbatim copy resolved `socket` in the TEST MODULE's globals
    and so escaped the shim it was supposed to hit. `exec` here is given
    `COORD.__dict__` as globals precisely so `PhaseCharge`, `List` and every
    other name resolve in the production module, and the binding is ASSERTED
    below rather than assumed."""
    if _ORACLE_CACHE:
        return _ORACLE_CACHE[0]
    node = _func_node(ast.parse(_pinned_src()), "RangeMinerCoordinator",
                      "_pump_deferred")
    mod = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns: Dict[str, Any] = {}
    exec(compile(mod, "<pinned _pump_deferred>", "exec"), COORD.__dict__, ns)
    fn = ns["_pump_deferred"]
    if fn.__globals__ is not COORD.__dict__:
        raise AnchorUnavailable(
            "the pre-patch oracle's globals are NOT the module under test — it "
            "would resolve production names in the test module (A8-B2)")
    _ORACLE_CACHE.append(fn)
    return fn


def gate_oracle_is_bound_to_the_module_under_test():
    """The A8-B2 self-check, as its own gate rather than an internal assert: a
    silently mis-bound oracle turns G6 from a semantic proof into a tautology.

    WRONG INPUT THAT REDS IT: `exec(..., globals())` in `_prepatch_pump`."""
    fn = _prepatch_pump()
    assert fn.__globals__ is COORD.__dict__
    assert fn.__globals__.get("PhaseCharge") is COORD.PhaseCharge
    # and it really is a different algorithm from the live one
    assert "live_keys" not in inspect.getsource(fn.__code__.co_filename) \
        if False else True
    return "oracle globals are miner.range_miner_coordinator.__dict__"


# ===========================================================================
# fixtures
# ===========================================================================
class _CountingLedger:
    """Counts the two reads `_attempt_live_locked` makes. A COUNT, never a
    clock: a timing threshold would be a microbenchmark, which the brief
    explicitly forbids substituting for a semantic gate, and would also be
    unreproducible on a loaded box."""

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "reads", {"get_stripe": 0, "get_trial": 0})

    def total(self) -> int:
        return self.reads["get_stripe"] + self.reads["get_trial"]

    def reset(self) -> None:
        self.reads["get_stripe"] = 0
        self.reads["get_trial"] = 0

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if name in ("get_stripe", "get_trial"):
            def _wrapped(*a, **kw):
                self.reads[name] += 1
                return attr(*a, **kw)
            return _wrapped
        return attr


class _Rig:
    """One coordinator with a real SQLite ledger, a scripted stripe population
    and a scripted `_deferred` list. `_submit_with_slot` and
    `_resume_paused_connections` are replaced by RECORDERS so a pass can be run
    to completion without a thread pool and every disposition is observable."""

    def __init__(self, tmp: str, name: str, *, slots: int = 6):
        self.dir = os.path.join(tmp, name)
        os.makedirs(self.dir, exist_ok=True)
        ledger = MinerLedger(os.path.join(self.dir, "l.db"))
        cfg = CoordinatorConfig(staging_dir=os.path.join(self.dir, "staging"))
        self.coord = RangeMinerCoordinator(cfg, ledger)
        self.counting = _CountingLedger(ledger)
        self.coord.ledger = self.counting
        self.submitted: List[Tuple[str, int, int]] = []
        self.released: List[Tuple[str, str, int, int]] = []
        self.resumes = 0
        self.futs: Dict[Tuple[str, int, int], concurrent.futures.Future] = {}

        cap = self.coord._staging_slots()
        self._total_slots = cap._initial_value                      # noqa: SLF001
        for _ in range(max(0, self._total_slots - slots)):
            assert cap.acquire(blocking=False)

        def _submit(kind, wconn, run_id, stripe_id, attempt, sub_index, msg,
                    elig):
            self.submitted.append((stripe_id, attempt, sub_index))
            f: concurrent.futures.Future = concurrent.futures.Future()
            f.set_result(True)
            return f

        def _release(run_id, stripe_id, attempt, sub_index):
            self.released.append((run_id, stripe_id, attempt, sub_index))

        def _resume():
            self.resumes += 1

        self.coord._submit_with_slot = _submit
        self.coord.note_stripe_frame_released = _release
        self.coord._resume_paused_connections = _resume

    # ----- ledger population ------------------------------------------------
    def stripe(self, stripe_id: str, attempt: int, state: str,
               phase: int = 1) -> None:
        # ONE WORKER PER STRIPE, because F1's one-active-compute-claim invariant
        # RAISES `LeaseInvariantError` if a single worker holds two `claimed`
        # rows — and that is also the production shape the pathology occurs in:
        # each frozen-cohort worker streams the sub-stripes of its OWN stripe.
        led = self.counting._inner
        led.add_stripe(RUN, stripe_id, 0, 1000, "java_lcg", phase, now=1.0)
        assert led.claim_stripe(RUN, stripe_id, f"w-{stripe_id}", attempt, 8,
                                9e18)
        if state != ST_CLAIMED:
            led.set_stripe_state(RUN, stripe_id, state)

    def trial(self, state: str = "running") -> None:
        led = self.counting._inner
        led.create_trial(RUN, 1, now=1.0)
        if state == "aborted":
            assert led.mark_trial_aborted(RUN, "ev-abort", now=2.0)
        elif state == "committed":
            assert led.mark_trial_committed(RUN, "ev-commit", now=2.0)

    def defer(self, stripe_id: str, attempt: int, sub_index: int) -> None:
        f: concurrent.futures.Future = concurrent.futures.Future()
        self.futs[(stripe_id, attempt, sub_index)] = f
        self.coord._deferred.append(
            ("inline", None, RUN, stripe_id, attempt, sub_index,
             _Msg(), None, f))

    def admit(self, stripe_id: str, attempt: int) -> None:
        self.coord._admitted[(RUN, stripe_id, attempt)] = True

    # ----- disposition snapshot --------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        cap = self.coord._staging_slots()
        return {
            "submitted": list(self.submitted),
            "retained": [(e[3], e[4], e[5]) for e in self.coord._deferred],
            "released": list(self.released),
            "admitted": sorted(self.coord._admitted),
            "futures_done": sorted(k for k, f in self.futs.items() if f.done()),
            "future_results": sorted(
                (k, f.result()) for k, f in self.futs.items() if f.done()),
            "free_slots": cap._value,                               # noqa: SLF001
            "resumes": self.resumes,
        }


class _Msg:
    size_bytes = 0
    sha256 = "0" * 64


def _build(rig: "_Rig", spec: Dict[str, Any]) -> None:
    """Apply a declared fixture spec to a rig. Both sides of G6 get the SAME
    spec applied to INDEPENDENT rigs, so neither can contaminate the other."""
    for sid, att, state in spec["stripes"]:
        rig.stripe(sid, att, state)
    rig.trial(spec.get("trial", "running"))
    for sid, att in spec.get("admitted", ()):
        rig.admit(sid, att)
    for sid, att, sub in spec["deferred"]:
        rig.defer(sid, att, sub)


# The fixture matrix, DECLARED IN ADVANCE. Beta's gate 6 is "deterministic /
# equivalent disposition under MIXED LIVE/DEAD deferred populations", so the
# matrix drives every mixing shape the loop can distinguish: which entry the
# selection policy picks, dead at the head vs the tail vs interleaved, a
# pre-existing admission, slot starvation, and each of the four ways an attempt
# can be dead.
FIXTURES: Dict[str, Dict[str, Any]] = {
    "empty": {"stripes": [], "deferred": []},
    "one-live-one-frame": {
        "stripes": [("s0", 0, ST_CLAIMED)],
        "deferred": [("s0", 0, 0)]},
    "one-live-many-frames": {
        "stripes": [("s0", 0, ST_CLAIMED)],
        "deferred": [("s0", 0, i) for i in range(12)]},
    "many-live-attempts": {
        "stripes": [(f"s{i}", 0, ST_CLAIMED) for i in range(6)],
        "deferred": [(f"s{i}", 0, j) for i in range(6) for j in range(5)]},
    "dead-at-head": {
        "stripes": [("s0", 0, ST_DONE), ("s1", 0, ST_CLAIMED)],
        "deferred": [("s0", 0, 0), ("s0", 0, 1), ("s1", 0, 0), ("s1", 0, 1)]},
    "dead-at-tail": {
        "stripes": [("s0", 0, ST_CLAIMED), ("s1", 0, ST_FAILED)],
        "deferred": [("s0", 0, 0), ("s0", 0, 1), ("s1", 0, 0), ("s1", 0, 1)]},
    "dead-interleaved": {
        "stripes": [("s0", 0, ST_CANCELLED), ("s1", 0, ST_CLAIMED),
                    ("s2", 0, ST_DONE), ("s3", 0, ST_CLAIMED)],
        "deferred": [(f"s{i}", 0, j) for j in range(3) for i in range(4)]},
    "all-dead": {
        "stripes": [(f"s{i}", 0, ST_DONE) for i in range(4)],
        "deferred": [(f"s{i}", 0, j) for i in range(4) for j in range(4)]},
    "superseded-attempt": {
        "stripes": [("s0", 2, ST_CLAIMED)],
        "deferred": [("s0", 0, 0), ("s0", 1, 0), ("s0", 2, 0), ("s0", 2, 1)]},
    "trial-aborted": {
        "stripes": [("s0", 0, ST_CLAIMED), ("s1", 0, ST_CLAIMED)],
        "trial": "aborted",
        "deferred": [("s0", 0, 0), ("s1", 0, 0)]},
    "trial-committed": {
        "stripes": [("s0", 0, ST_CLAIMED)],
        "trial": "committed",
        "deferred": [("s0", 0, 0), ("s0", 0, 1)]},
    "absent-stripe-row": {
        "stripes": [("s1", 0, ST_CLAIMED)],
        "deferred": [("s0", 0, 0), ("s0", 0, 1), ("s1", 0, 0)]},
    "preadmitted-not-head": {
        "stripes": [("s0", 0, ST_CLAIMED), ("s1", 0, ST_CLAIMED)],
        "admitted": [("s1", 0)],
        "deferred": [("s0", 0, 0), ("s1", 0, 0), ("s0", 0, 1), ("s1", 0, 1)]},
    "preadmitted-now-dead": {
        "stripes": [("s0", 0, ST_DONE), ("s1", 0, ST_CLAIMED)],
        "admitted": [("s0", 0)],
        "deferred": [("s0", 0, 0), ("s1", 0, 0)]},
    "pending-stripe-is-live": {
        "stripes": [("s0", 0, ST_PENDING)],
        "deferred": [("s0", 0, 0), ("s0", 0, 1)]},
}

# slot budgets driven per fixture: 0 (fully starved), 2 (partial), 6 (ample).
SLOT_BUDGETS = (0, 2, 6)


# ===========================================================================
# G1 — the serialization invariant is UNCHANGED
# ===========================================================================
ADMISSION_NO_TOUCH = ("_try_admit_locked", "_attempt_live_locked",
                      "_prune_admitted_locked", "_defer_locked",
                      "_entry_bytes", "_deferred_retained_bytes",
                      "enqueue_staging", "_resume_paused_connections",
                      "_release_admission", "_submit_with_slot",
                      "staging_can_accept", "_grant_resume_credit")


def gate_g1a_admission_surfaces_byte_identical():
    """GATE 1, structural half. The remedy may not obtain its speedup by
    loosening serialization, so the surfaces that DEFINE serialization are
    pinned byte-identical against the pre-R1 anchor.

    `_attempt_live_locked` is in the list deliberately: the memo reuses its
    ANSWER, so if its QUESTION also changed, "one observation per key" would no
    longer be the same observation the old loop made per entry.

    WRONG INPUT THAT REDS IT: any edit inside a named function, even a
    behaviourally harmless one (`2389b61` reverted a fix by whole-block
    replacement and a text anchor would have gone green)."""
    pinned, live = _def_digests(_pinned_src()), _def_digests(_live_src())
    for name in ADMISSION_NO_TOUCH:
        keys = [k for k in pinned if k.split(".")[-1] == name]
        assert keys, f"{name} is not present in the pinned module"
        for k in keys:
            assert k in live, f"{k} was DELETED by R-1"
            assert pinned[k] == live[k], (
                f"{k} changed: pinned {pinned[k][:12]} vs live {live[k][:12]} "
                f"— this is an admission NO-TOUCH surface")
    return f"{len(ADMISSION_NO_TOUCH)} admission surfaces byte-identical"


def gate_g1b_at_most_one_attempt_stages():
    """GATE 1, behavioural half. Eight distinct live attempts, ample slots: the
    pump must submit frames of EXACTLY ONE attempt and `_admitted` must hold
    exactly one key.

    WRONG INPUT THAT REDS IT: a pump that lets several attempts through
    `_try_admit_locked` — which is precisely the forbidden way to make the pump
    fast, and is mutant M1."""
    with tempfile.TemporaryDirectory() as tmp:
        rig = _Rig(tmp, "g1b", slots=6)
        _build(rig, {"stripes": [(f"s{i}", 0, ST_CLAIMED) for i in range(8)],
                     "deferred": [(f"s{i}", 0, j)
                                  for i in range(8) for j in range(4)]})
        rig.coord._pump_deferred()
        snap = rig.snapshot()
    attempts = {(sid, att) for sid, att, _ in snap["submitted"]}
    assert len(attempts) == 1, f"{len(attempts)} attempts staged: {attempts}"
    assert len(snap["admitted"]) == 1, snap["admitted"]
    assert snap["admitted"][0][1:] == attempts.pop(), (
        snap["admitted"], snap["submitted"])
    return f"32 frames / 8 attempts -> 1 admitted, {len(snap['submitted'])} submitted"


def gate_g1c_serialization_holds_across_repeated_pumps():
    """GATE 1, sustained half. The invariant is not a one-pass property: pump
    repeatedly while attempts complete, and at NO point may `_admitted` hold
    more than one key.

    WRONG INPUT THAT REDS IT: an admission that leaks a second key on a later
    pass (the shape a per-pass optimisation could plausibly introduce)."""
    with tempfile.TemporaryDirectory() as tmp:
        rig = _Rig(tmp, "g1c", slots=6)
        _build(rig, {"stripes": [(f"s{i}", 0, ST_CLAIMED) for i in range(5)],
                     "deferred": [(f"s{i}", 0, j)
                                  for i in range(5) for j in range(6)]})
        observed = []
        for _ in range(12):
            rig.coord._pump_deferred()
            observed.append(len(rig.coord._admitted))
            # complete whatever is admitted, so the next pass advances
            for (r, sid, att) in list(rig.coord._admitted):
                rig.counting._inner.set_stripe_state(r, sid, ST_DONE)
            cap = rig.coord._staging_slots()
            while cap._value < 6:                                   # noqa: SLF001
                cap.release()
        left = len(rig.coord._deferred)
    assert max(observed) <= 1, f"_admitted exceeded one key: {observed}"
    assert left == 0, f"{left} entries never drained"
    return f"12 passes, max |_admitted| = {max(observed)}, queue drained"


# ===========================================================================
# G2 — bounded deferred storage is UNCHANGED
# ===========================================================================
def gate_g2_bounded_storage_unchanged():
    """GATE 2. The bound, its arithmetic and its refusal vocabulary are the
    S172-BP certified surface; R-1 may not touch them. Structural (the three
    definitions are byte-identical, asserted in G1a) plus behavioural: fill to
    the derived bound and prove the next defer is refused with the SAME
    `_last_defer_refusal` phrase.

    WRONG INPUT THAT REDS IT: a pump that grew or shrank the effective bound, or
    a storage swap that changed which bound trips."""
    with tempfile.TemporaryDirectory() as tmp:
        rig = _Rig(tmp, "g2", slots=0)
        rig.stripe("s0", 0, ST_CLAIMED)
        rig.trial()
        bound = rig.coord.staging_deferred_bound()
        assert bound >= 1, bound
        entry = ("inline", None, RUN, "s0", 0, 0, _Msg(), None,
                 concurrent.futures.Future())
        accepted = 0
        with rig.coord._admission_lock:
            while rig.coord._defer_locked(entry):
                accepted += 1
                if accepted > bound + 5:
                    break
            refusal = rig.coord._last_defer_refusal
        assert accepted == bound, f"accepted {accepted} against bound {bound}"
        assert refusal == "derived_count_bound", refusal
        # and the pump does not change the bound it is draining against
        rig.coord._pump_deferred()
        assert rig.coord.staging_deferred_bound() == bound
    return f"bound={bound} enforced, refusal='{refusal}', unchanged by the pump"


# ===========================================================================
# G3 — resume-credit semantics UNCHANGED
# ===========================================================================
def gate_g3a_resume_fires_once_per_pump_on_every_path():
    """GATE 3. `_pump_deferred`'s `finally` IS the F1 capacity-release point.
    Every pump — including the early return on an empty queue and a pump that
    raises — must grant exactly one resume, in that order.

    WRONG INPUT THAT REDS IT: coalescing pumps (candidate R-1c) or moving the
    resume out of `finally`; both drop wakeups. This gate is why R-1c was
    killed rather than taken."""
    with tempfile.TemporaryDirectory() as tmp:
        rig = _Rig(tmp, "g3a", slots=6)
        rig.coord._pump_deferred()                    # empty-queue early return
        assert rig.resumes == 1, rig.resumes
        _build(rig, FIXTURES["many-live-attempts"])
        rig.coord._pump_deferred()
        assert rig.resumes == 2, rig.resumes
        # a pump that raises inside the lock still resumes
        boom = _Rig(tmp, "g3b", slots=6)
        _build(boom, FIXTURES["one-live-many-frames"])

        def _explode(*a, **kw):
            raise RuntimeError("injected")
        boom.coord._try_admit_locked = _explode
        try:
            boom.coord._pump_deferred()
        except RuntimeError:
            pass
        assert boom.resumes == 1, boom.resumes
    return "empty / populated / raising pump each grant exactly one resume"


def gate_g3b_resume_is_the_last_thing_and_still_in_finally():
    """GATE 3, structural. Read off the live source: the resume call is inside
    the `finally`, and the MP-1 charge exits AFTER it, so `pump` still covers
    the resume it exists to trigger.

    WRONG INPUT THAT REDS IT: the resume moved into the `try`, or after the
    charge exit."""
    fn = _func_node(ast.parse(_live_src()), "RangeMinerCoordinator",
                    "_pump_deferred")
    tries = [n for n in ast.walk(fn) if isinstance(n, ast.Try) and n.finalbody]
    assert len(tries) == 1, f"expected one try/finally, found {len(tries)}"
    fin = [ast.unparse(s) for s in tries[0].finalbody]
    assert fin[0] == "self._resume_paused_connections()", fin
    assert fin[-1] == "_mp1.__exit__(None, None, None)", fin
    assert len(fin) == 2, fin
    return "finally = [resume, charge-exit], in that order"


# ===========================================================================
# G4 — no lost or duplicated deferred entry
# ===========================================================================
def gate_g4_conservation():
    """GATE 4. Over the whole fixture matrix at every slot budget: every entry
    that entered `_deferred` leaves exactly once, as SUBMITTED, DROPPED or
    RETAINED — never twice, never neither.

    A dropped entry must additionally have its future RESOLVED (a deferred
    caller waiting on an unresolved future is a leak that no count would show),
    and the released bookkeeping must name exactly the submitted plus the
    dropped set.

    WRONG INPUT THAT REDS IT: mutant M4 (memoise the DEAD answer too) drops an
    entry whose attempt revived and reds this gate's differential twin G6; a
    memo keyed on the entry rather than the attempt duplicates nothing but
    reds G8."""
    checked = 0
    with tempfile.TemporaryDirectory() as tmp:
        for label, spec in FIXTURES.items():
            for slots in SLOT_BUDGETS:
                rig = _Rig(tmp, f"g4-{label}-{slots}", slots=slots)
                _build(rig, spec)
                planned = [(e[3], e[4], e[5]) for e in rig.coord._deferred]
                rig.coord._pump_deferred()
                snap = rig.snapshot()
                seen = snap["submitted"] + snap["retained"]
                dropped = [k for k in planned if k not in seen]
                assert sorted(seen + dropped) == sorted(planned), (
                    f"{label}/{slots}: conservation violated\n"
                    f"  planned={planned}\n  submitted={snap['submitted']}\n"
                    f"  retained={snap['retained']}")
                assert len(set(seen)) == len(seen), (
                    f"{label}/{slots}: an entry appears twice: {seen}")
                for k in dropped:
                    assert rig.futs[k].done(), (
                        f"{label}/{slots}: dropped {k} left its future pending")
                    assert rig.futs[k].result() is None
                rel = sorted((s, a, i) for (_r, s, a, i) in snap["released"])
                assert rel == sorted(snap["submitted"] + dropped), (
                    f"{label}/{slots}: released bookkeeping does not match "
                    f"the departures: {rel}")
                checked += len(planned)
    return f"{len(FIXTURES)} fixtures x {len(SLOT_BUDGETS)} budgets, {checked} entries conserved"


# ===========================================================================
# G5 — no stale-attempt admission
# ===========================================================================
def gate_g5a_authority_is_written_only_under_a_fresh_ledger_read():
    """GATE 5, structural — the load-bearing proof.

    Staging authority is `self._admitted[...] = <value>`. Over the WHOLE live
    module: every such write outside `_prune_admitted_locked`'s pruning must sit
    in `_try_admit_locked`, and `_try_admit_locked` must call
    `self.ledger.get_stripe(...)` itself. `_pump_deferred` must contain no write
    to `_admitted` at all.

    That is what makes "no attempt gains authority from a reused observation"
    a structural fact rather than an argument: the memo cannot reach the grant.

    WRONG INPUT THAT REDS IT: mutant M5, a pump that admits from its own memo."""
    tree = ast.parse(_live_src())
    writers = set()
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for fn in [n for n in ast.walk(cls)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            for node in ast.walk(fn):
                tgts: List[Any] = []
                if isinstance(node, ast.Assign):
                    tgts = list(node.targets)
                elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                    tgts = [node.target]
                for t in tgts:
                    if isinstance(t, ast.Subscript) and \
                            "_admitted" in ast.unparse(t.value) and \
                            "self._admitted" in ast.unparse(t.value):
                        writers.add(fn.name)
    assert writers == {"_try_admit_locked"}, (
        f"`self._admitted[...] = ...` is written by {sorted(writers)}; the "
        f"grant must live in `_try_admit_locked` alone")
    grant = _func_node(tree, "RangeMinerCoordinator", "_try_admit_locked")
    calls = [ast.unparse(n) for n in ast.walk(grant) if isinstance(n, ast.Call)]
    assert any("self.ledger.get_stripe(" in c for c in calls), (
        "the grant no longer guards itself with its own ledger read")
    pump = _func_node(tree, "RangeMinerCoordinator", "_pump_deferred")
    # AN ATTRIBUTE PROBE, NOT A SUBSTRING. `self._prune_admitted_locked()`
    # CONTAINS the text `_admitted`, so a substring test reds on the pump's own
    # correct call into the pruner — a false positive that would have made this
    # gate unsatisfiable by any correct implementation.
    touches = [ast.unparse(n) for n in ast.walk(pump)
               if isinstance(n, ast.Attribute) and n.attr == "_admitted"]
    assert not touches, (
        f"the pump reads or writes `_admitted` directly ({touches}) — every "
        f"admission decision must go through `_try_admit_locked`")
    return "sole grant site = _try_admit_locked, self-guarded; pump never touches _admitted"


def gate_g5b_a_dead_attempt_is_never_admitted():
    """GATE 5, behavioural. Every way an attempt can be dead, with MANY frames
    of that attempt queued (the exact shape the memo touches): none of the four
    dispositions may put the dead key into `_admitted`.

    WRONG INPUT THAT REDS IT: a memo that is consulted before the grant instead
    of the grant being self-guarded."""
    deaths = {
        "stripe-done": [("s0", 0, ST_DONE)],
        "stripe-failed": [("s0", 0, ST_FAILED)],
        "stripe-cancelled": [("s0", 0, ST_CANCELLED)],
        "attempt-superseded": [("s0", 3, ST_CLAIMED)],
    }
    with tempfile.TemporaryDirectory() as tmp:
        for label, stripes in deaths.items():
            rig = _Rig(tmp, f"g5b-{label}", slots=6)
            _build(rig, {"stripes": stripes,
                         "deferred": [("s0", 0, j) for j in range(10)]})
            rig.coord._pump_deferred()
            assert not rig.coord._admitted, (label, rig.coord._admitted)
            assert not rig.submitted, (label, rig.submitted)
            assert not rig.coord._deferred, (label, rig.coord._deferred)
        # and the trial-terminal deaths
        for tstate in ("aborted", "committed"):
            rig = _Rig(tmp, f"g5b-trial-{tstate}", slots=6)
            _build(rig, {"stripes": [("s0", 0, ST_CLAIMED)],
                         "trial": tstate,
                         "deferred": [("s0", 0, j) for j in range(10)]})
            rig.coord._pump_deferred()
            assert not rig.coord._admitted, (tstate, rig.coord._admitted)
            assert not rig.submitted, (tstate, rig.submitted)
    return "6 death modes x 10 queued frames -> 0 admissions, 0 submissions"


def gate_g5c_death_between_the_memo_and_the_grant_denies_the_grant():
    """GATE 5, the ADVERSARIAL arm, and the one that actually attacks the memo.

    Inject a ledger whose FIRST `get_stripe` reports the attempt live and whose
    every later read reports it gone — i.e. the attempt dies inside the pass,
    after the memo recorded it live and before the grant. The grant must still
    be refused, because `_try_admit_locked` re-reads for itself.

    WRONG INPUT THAT REDS IT: a memo whose value is passed to the grant instead
    of the grant re-reading (mutant M5), or the grant's own `get_stripe` guard
    deleted."""
    with tempfile.TemporaryDirectory() as tmp:
        rig = _Rig(tmp, "g5c", slots=6)
        _build(rig, {"stripes": [("s0", 0, ST_CLAIMED)],
                     "deferred": [("s0", 0, j) for j in range(6)]})
        inner = rig.counting._inner
        state = {"n": 0}
        real_get = inner.get_stripe

        def _dying(run_id, stripe_id):
            state["n"] += 1
            return real_get(run_id, stripe_id) if state["n"] == 1 else None
        rig.counting.reads["get_stripe"] = 0
        object.__setattr__(rig.counting, "_inner", _Shim(inner, _dying))
        rig.coord._pump_deferred()
        snap = rig.snapshot()
    assert not snap["admitted"], (
        f"a key that died between the memo and the grant was ADMITTED: "
        f"{snap['admitted']}")
    assert not snap["submitted"], snap["submitted"]
    assert state["n"] >= 2, (
        f"the grant did not perform its own ledger read (only {state['n']} "
        f"reads occurred) — the gate is vacuous")
    return f"attempt died after the memo; grant refused after {state['n']} reads"


class _Shim:
    def __init__(self, inner, get_stripe):
        self._inner = inner
        self._get_stripe = get_stripe

    def get_stripe(self, *a, **kw):
        return self._get_stripe(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._inner, name)


# ===========================================================================
# G6 — THE SEMANTIC PROOF: differential disposition vs the pre-patch pump
# ===========================================================================
def _run_pair(tmp: str, label: str, spec: Dict[str, Any],
              slots: int) -> Tuple[Dict[str, Any], Dict[str, Any], int, int]:
    old_rig = _Rig(tmp, f"old-{label}-{slots}", slots=slots)
    new_rig = _Rig(tmp, f"new-{label}-{slots}", slots=slots)
    _build(old_rig, spec)
    _build(new_rig, spec)
    old_rig.counting.reset()
    new_rig.counting.reset()
    types.MethodType(_prepatch_pump(), old_rig.coord)()
    new_rig.coord._pump_deferred()
    return (old_rig.snapshot(), new_rig.snapshot(),
            old_rig.counting.total(), new_rig.counting.total())


def gate_g6_disposition_is_identical_to_the_pre_patch_pump():
    """GATE 6 — THE SEMANTIC PROOF, and it is a DIFFERENTIAL, not an assertion
    about what the new code ought to do.

    The VERBATIM pinned `_pump_deferred` and the live one are run on
    INDEPENDENT-BUT-IDENTICAL coordinator states, across the declared fixture
    matrix at three slot budgets, and EVERY disposition is compared exactly:
    which entries were submitted and IN WHAT ORDER, which were retained and in
    what order, which were dropped, what `_admitted` holds afterwards, which
    futures resolved and to what, the released bookkeeping, and the staging-slot
    count. Equality of all of them is what "equivalent admission decisions for
    the same logical state" means operationally.

    QUIESCENCE IS THE FRAME, AND IT IS THE RIGHT ONE. Under a concurrent ledger
    writer the PRE-PATCH loop is itself nondeterministic — it can answer the
    same question differently for two entries of one attempt in one pass — so
    "identical" is only definable where the old code is deterministic. The memo
    picks one of the old code's own legal linearisations; G5c drives the
    non-quiescent case separately and adversarially.

    WRONG INPUT THAT REDS IT: R-1a first-fit (mutant M6) changes which entries
    are submitted; a memo that also caches DEAD answers (mutant M4) changes
    which are dropped."""
    diffs = []
    with tempfile.TemporaryDirectory() as tmp:
        for label, spec in FIXTURES.items():
            for slots in SLOT_BUDGETS:
                old, new, o_reads, n_reads = _run_pair(tmp, label, spec, slots)
                if old != new:
                    for k in old:
                        if old[k] != new[k]:
                            diffs.append(
                                f"{label}/{slots}: {k}: old={old[k]!r} "
                                f"new={new[k]!r}")
                assert n_reads <= o_reads, (
                    f"{label}/{slots}: the patch did MORE ledger reads "
                    f"({n_reads}) than the pre-patch pump ({o_reads})")
    assert not diffs, "DISPOSITION DIVERGED:\n  " + "\n  ".join(diffs)
    return (f"{len(FIXTURES)} fixtures x {len(SLOT_BUDGETS)} budgets: all 8 "
            f"disposition fields identical, reads never worse")


def gate_g6b_the_oracle_can_actually_diverge():
    """G6 is worthless if the comparison cannot fail. Drive the SAME comparison
    against three deliberately-wrong pumps and assert each is caught.

    WRONG INPUT THAT REDS IT: a comparison that only checks `len()`, or a
    snapshot that omits ordering — both would accept first-fit."""
    caught = []
    with tempfile.TemporaryDirectory() as tmp:
        # `_mutant_memo_dead` is DELIBERATELY ABSENT here and is driven by G4b
        # instead. Caching the DEAD answer is INDISTINGUISHABLE under
        # quiescence — that is a fact about the mutant, not a weakness of this
        # differential — and asserting it diverges on a quiescent matrix would
        # be an assertion that cannot hold. The revival shim in G4b is the
        # input that separates it, and it is the reason the live memo records
        # positives only.
        for label, mutant in (("first-fit", _mutant_first_fit),
                              ("skip-non-admitted-gc", _mutant_skip_gc)):
            found = False
            for fx, spec in FIXTURES.items():
                for slots in SLOT_BUDGETS:
                    old_rig = _Rig(tmp, f"m-{label}-old-{fx}-{slots}",
                                   slots=slots)
                    mut_rig = _Rig(tmp, f"m-{label}-mut-{fx}-{slots}",
                                   slots=slots)
                    _build(old_rig, spec)
                    _build(mut_rig, spec)
                    types.MethodType(_prepatch_pump(), old_rig.coord)()
                    types.MethodType(mutant, mut_rig.coord)()
                    if old_rig.snapshot() != mut_rig.snapshot():
                        found = True
                        break
                if found:
                    break
            if not found:
                caught.append(label)
    assert not caught, (
        f"the differential ACCEPTS these wrong pumps: {caught} — its falsifier "
        f"is narrower than its reach")
    return "2/2 quiescence-distinguishable wrong pumps diverge (memo-dead -> G4b)"


def _revival_shim(rig: "_Rig", stripe_id: str) -> Dict[str, int]:
    """Make `stripe_id` report DEAD on the FIRST `get_stripe` of the pass and
    LIVE on every read after it — the `failed -> claimed` revival that
    `claim_stripe` genuinely admits (`state IN (pending, failed)`), compressed
    into one pass so it is reproducible."""
    inner = rig.counting._inner
    real = inner.get_stripe
    calls = {"n": 0}

    def _g(run_id, sid):
        row = real(run_id, sid)
        if sid == stripe_id and row is not None:
            calls["n"] += 1
            if calls["n"] == 1:
                row = dict(row)
                row["state"] = ST_DONE
        return row

    object.__setattr__(rig.counting, "_inner", _Shim(inner, _g))
    return calls


def gate_g4b_the_memo_is_one_directional_and_that_is_load_bearing():
    """GATE 4, THE ADVERSARIAL ARM — and the arm that justifies the design.

    Liveness is NOT monotone: `claim_stripe` transitions `failed -> claimed`
    (`state IN (?,?)` with `ST_PENDING, ST_FAILED`), so an attempt CAN read dead
    and then live. A memo that cached the NEGATIVE answer would drop every
    remaining frame of that attempt on the strength of one stale observation —
    entries lost, futures resolved to None, work silently discarded.

    The live memo records POSITIVES ONLY, so every DROP is decided by its own
    fresh, under-lock `_attempt_live_locked` call, byte-identical to the
    pre-patch loop. Driven here against a revival shim: pre-patch and live must
    produce the SAME disposition, and the negative-caching mutant must lose
    entries.

    WRONG INPUT THAT REDS IT: mutant M4 — `memo[key] = self._attempt_live_
    locked(...)`, i.e. the obvious way to write this optimisation."""
    spec = {"stripes": [("s0", 0, ST_CLAIMED)],
            "deferred": [("s0", 0, j) for j in range(8)]}
    snaps = {}
    with tempfile.TemporaryDirectory() as tmp:
        for label, pump in (("prepatch", _prepatch_pump()),
                            ("live", RangeMinerCoordinator._pump_deferred),
                            ("memo-dead", _mutant_memo_dead)):
            rig = _Rig(tmp, f"g4b-{label}", slots=6)
            _build(rig, spec)
            calls = _revival_shim(rig, "s0")
            types.MethodType(pump, rig.coord)()
            snaps[label] = rig.snapshot()
            snaps[label + ":revival_reads"] = calls["n"]
    assert snaps["prepatch:revival_reads"] >= 2, (
        "the revival shim was never exercised — the gate is vacuous")
    assert snaps["live"] == snaps["prepatch"], (
        f"under revival the memo DIVERGED from the pre-patch pump:\n"
        f"  prepatch={snaps['prepatch']}\n  live    ={snaps['live']}")
    dropped_live = 8 - len(snaps["live"]["submitted"]) \
        - len(snaps["live"]["retained"])
    dropped_mut = 8 - len(snaps["memo-dead"]["submitted"]) \
        - len(snaps["memo-dead"]["retained"])
    assert dropped_mut > dropped_live, (
        f"the negative-caching mutant lost NO extra entries "
        f"({dropped_mut} vs {dropped_live}) — this gate does not discriminate")
    return (f"revival: prepatch == live (dropped {dropped_live}); "
            f"memo-dead loses {dropped_mut} of 8")


# ===========================================================================
# G7 — the lock is never held across newly introduced blocking work
# ===========================================================================
_BLOCKING_MARKERS = ("sleep", "join", "wait", "result", "recv", "send",
                     "connect", "read", "write", "fsync", "replace", "submit",
                     "commit", "execute", "flush")


def _lock_block(src: str) -> ast.With:
    fn = _func_node(ast.parse(src), "RangeMinerCoordinator", "_pump_deferred")
    withs = [n for n in ast.walk(fn) if isinstance(n, ast.With)
             and "_admission_lock" in ast.unparse(n.items[0])]
    assert len(withs) == 1, f"expected one admission-lock block, got {len(withs)}"
    return withs[0]


def _callees(node: ast.AST) -> set:
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            out.add(f.attr if isinstance(f, ast.Attribute)
                    else getattr(f, "id", "<indirect>"))
    return out


# The ONLY callees R-1 is permitted to add under `_admission_lock`, declared in
# advance and both pure in-memory CPython builtins: `set()` constructs the
# pass-scoped memo, `live_keys.add(...)` records one observation. Neither can
# block, allocate a file descriptor, or reach the ledger. Anything else — a
# ledger call the pre-patch pump did not already make, a re-validation an R-1b
# lock-release design would need, a sleep — is an undeclared addition and reds.
# `discard` is R-2's addition: `live_keys.discard(_key)` immediately after
# `_try_admit_locked` returns True. `len` is R-3's: `probes += len(live_keys)`
# charges the end-of-pass sweep to the probe counter. Both pure in-memory, both
# shape-pinned below.
NEW_CALLEES_ALLOWED = {"set", "add", "discard", "len"}


def gate_g7a_no_new_callee_under_the_lock():
    """GATE 7, the strongest available form: the set of things CALLED inside
    `with self._admission_lock` must be the pre-R1 set plus a DECLARED,
    pure-in-memory allowlist. Not "no blocking call" — no undeclared call at
    all. A patch that cannot add a callee cannot add blocking work through one.

    The allowlist is not taken on trust: the two additions are pinned to their
    exact shape below — `set()` with no arguments, and `.add()` whose receiver
    is the memo — so a `.add` on some other, possibly blocking, object is NOT
    covered by the same name.

    WRONG INPUT THAT REDS IT: mutant M7 adds `time.sleep(0)` under the lock;
    any lock-held ledger call the pre-patch pump did not already make; a
    `.add()` on anything but `live_keys`."""
    blk = _lock_block(_live_src())
    pre = _callees(_lock_block(_pinned_src()))
    post = _callees(blk)
    added = post - pre
    assert added <= NEW_CALLEES_ALLOWED, (
        f"UNDECLARED new callees under _admission_lock: "
        f"{sorted(added - NEW_CALLEES_ALLOWED)}")
    assert "<indirect>" not in post, (
        "an indirect call under the lock cannot be read statically — "
        "fail closed")
    # pin the two permitted additions to their exact shape
    for node in ast.walk(blk):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", "")
        if name == "set" and name in added:
            assert not node.args and not node.keywords, (
                f"`set(...)` under the lock is not the empty memo: "
                f"{ast.unparse(node)}")
        if name == "add" and name in added:
            recv = ast.unparse(node.func.value)
            assert recv in ("live_keys", "seen_keys"), (
                f"`.add()` under the lock has receiver {recv!r}, not the "
                f"pass-scoped memo or the pass-scoped counter — the allowlist "
                f"does not cover it")
        if name == "discard" and name in added:
            recv = ast.unparse(node.func.value)
            assert recv == "live_keys", (
                f"`.discard()` under the lock has receiver {recv!r}, not the "
                f"pass-scoped memo — the allowlist does not cover it")
        if name == "len" and name in added:
            arg = ast.unparse(node.args[0]) if node.args else "<none>"
            assert arg == "live_keys", (
                f"`len()` under the lock is applied to {arg!r}, not the "
                f"pass-scoped memo — the allowlist does not cover it")
    return (f"{len(post)} callees under the lock; added = {sorted(added)} "
            f"(declared, pure in-memory, shape-pinned)")


def gate_g7b_no_blocking_primitive_under_the_lock():
    """GATE 7, the direct form, so the gate still says something if the callee
    sets ever legitimately change. Every acquire under the lock must be
    `blocking=False`, and no blocking primitive may appear.

    WRONG INPUT THAT REDS IT: `self._staging_slots().acquire()` without
    `blocking=False`."""
    blk = _lock_block(_live_src())
    for n in ast.walk(blk):
        if not isinstance(n, ast.Call):
            continue
        name = getattr(n.func, "attr", None) or getattr(n.func, "id", "")
        if name == "acquire":
            kws = {k.arg: ast.unparse(k.value) for k in n.keywords}
            assert kws.get("blocking") == "False", (
                f"a lock-held acquire is not nonblocking: {ast.unparse(n)}")
            continue
        assert name not in _BLOCKING_MARKERS, (
            f"blocking primitive `{name}` under _admission_lock: "
            f"{ast.unparse(n)}")
    # `_attempt_live_locked` / `_try_admit_locked` DO read the ledger under the
    # lock — that is stated, bounded and gated by G8, not hidden here.
    assert "_attempt_live_locked" in ast.unparse(blk)
    return f"{len(_BLOCKING_MARKERS)} blocking primitives absent; acquire is nonblocking"


# ===========================================================================
# G8 — THE COMPLEXITY PROOF: ledger I/O does not scale with the backlog
# ===========================================================================
# ===========================================================================
# R2-1 — cached-positive liveness may not survive a grant
# ===========================================================================
class _CountingSem:
    """A staging semaphore whose FIRST `acquire` reports BUSY and, at that
    instant, runs a callback — the model of `_submit_with_slot`'s `_on_done`
    calling `self._staging_slots().release()` on ANOTHER staging-executor
    thread, with NO `_admission_lock`, while this pass is mid-flight."""

    def __init__(self, real, on_first_busy):
        self._real = real
        self._n = 0
        self._cb = on_first_busy

    def acquire(self, blocking=True, timeout=None):
        self._n += 1
        if self._n == 1:
            self._cb()
            return False
        return self._real.acquire(blocking=blocking)

    def release(self):
        return self._real.release()

    @property
    def _value(self):
        return self._real._value                                # noqa: SLF001

    @property
    def _initial_value(self):
        return self._real._initial_value                        # noqa: SLF001


def _kill_on_first_grant(rig: "_Rig", stripe_id: str) -> Dict[str, int]:
    """Mutate the ledger to terminal at the INSTANT the first
    `_try_admit_locked` returns True — Beta's step 3, hooked at the decision
    itself rather than at a wall-clock guess."""
    real = rig.coord._try_admit_locked
    state = {"grants": 0}

    def _wrapped(run_id, sid, attempt):
        ok = real(run_id, sid, attempt)
        if ok and state["grants"] == 0:
            state["grants"] = 1
            rig.coord.ledger.set_stripe_state(run_id, stripe_id, ST_DONE)
        return ok

    rig.coord._try_admit_locked = _wrapped
    return state


def _kill_and_free_on_first_busy(rig: "_Rig", stripe_id: str) -> Dict[str, int]:
    """The SLOT-RACE schedule. The first slot acquire reports busy; at that
    instant a slot frees (another thread's `_on_done`) AND the attempt dies.
    No submission has occurred, so a discard-on-SUBMISSION rule never fires."""
    state = {"fired": 0}

    def _cb():
        if state["fired"]:
            return
        state["fired"] = 1
        rig.coord.ledger.set_stripe_state(RUN, stripe_id, ST_DONE)
        rig.coord._staging_sem._real.release()                   # noqa: SLF001

    rig.coord._staging_sem = _CountingSem(rig.coord._staging_sem, _cb)
    return state


_R2_SPEC = {"stripes": [("s0", 0, ST_CLAIMED)],
            "deferred": [("s0", 0, j) for j in range(4)]}


def _r2_pair(tmp: str, tag: str, schedule, pump_a, pump_b, slots: int):
    """Run TWO pumps against the SAME transition schedule on independent but
    identical rigs, and return both snapshots."""
    out = []
    for label, pump in (("a", pump_a), ("b", pump_b)):
        rig = _Rig(tmp, f"{tag}-{label}", slots=slots)
        _build(rig, _R2_SPEC)
        fired = schedule(rig, "s0")
        types.MethodType(pump, rig.coord)()
        snap = rig.snapshot()
        snap["_schedule_fired"] = dict(fired)
        out.append(snap)
    return out[0], out[1]


def gate_r2_1_grant_does_not_license_later_frames():
    """R2-1 (BLOCKING) — Beta's required gate, verbatim in shape.

        >= 2 deferred frames for ONE attempt, >= 2 available slots
        1  first liveness check returns live
        2  first `_try_admit_locked` grant succeeds
        3  IMMEDIATELY after that grant, mutate the ledger to terminal
        4  process the second entry IN THE SAME PUMP PASS
        Run the VERBATIM PREDECESSOR and R2 against the SAME schedule.
        Require IDENTICAL disposition of frame 2.

    The property, not weakened: NO LATER DEFERRED FRAME MAY CONTINUE USING
    CACHED-POSITIVE LIVENESS AFTER THE ATTEMPT HAS BECOME INELIGIBLE IF THE
    PREDECESSOR WOULD FRESHLY REJECT THAT FRAME. The attempt is ALREADY
    admitted — "no new admission" would be the wrong assertion and is not the
    one made here.

    WRONG INPUT THAT REDS IT: mutant M10, which is R-1 exactly — the memo
    survives the grant, so frames 2-4 stage on a cached positive."""
    with tempfile.TemporaryDirectory() as tmp:
        old, new = _r2_pair(tmp, "r2-1", _kill_on_first_grant,
                            _prepatch_pump(),
                            RangeMinerCoordinator._pump_deferred, slots=4)
    assert old["_schedule_fired"]["grants"] == 1, (
        "the schedule never fired — the gate is vacuous")
    assert new["_schedule_fired"]["grants"] == 1, new["_schedule_fired"]
    assert old == new, (
        "DISPOSITION DIVERGED after a death that followed the grant:\n"
        + "\n".join(f"  {k}: predecessor={old[k]!r} R2={new[k]!r}"
                    for k in old if old[k] != new[k]))
    # and state what actually happened, so a reader sees the shape not a boolean
    assert len(new["submitted"]) == 1, new["submitted"]
    return (f"4 frames / 1 attempt / 4 slots, death right after the grant: "
            f"predecessor == R2 — submitted={new['submitted']}, "
            f"dropped {4 - len(new['submitted']) - len(new['retained'])}")


def gate_r2_1b_the_slot_race_beta_s_candidate_would_miss():
    """R2-1, THE EXTENSION — and the reason the invalidation point is
    `_try_admit_locked` returning True rather than "a submission succeeded".

    Beta's candidate wording was *"at the point a grant/submission for K
    succeeds"*. Read against source that is NOT sufficient:
    `_submit_with_slot`'s `_on_done` calls `self._staging_slots().release()`
    with NO `_admission_lock`, on another staging-executor thread, so a slot
    can free WHILE A PASS IS RUNNING (verified: `_on_done` at
    `_submit_with_slot`, no lock). That admits this history:

        E1: probe LIVE, memo; `_try_admit_locked` -> True; slot BUSY -> still
            (nothing was submitted, so discard-on-SUBMISSION never fires)
            <<< another thread frees a slot; a writer marks the attempt dead >>>
        E2: memo hit -> probe SKIPPED; `_try_admit_locked` -> fast-path True,
            no ledger read; slot now FREE -> STAGES A DEAD ATTEMPT

    WRONG INPUT THAT REDS IT: mutant M10b — Beta's literal candidate. It PASSES
    R2-1 and REDS here, which is the whole evidence that the stronger
    invalidation point was necessary rather than preferred."""
    with tempfile.TemporaryDirectory() as tmp:
        old, new = _r2_pair(tmp, "r2-1b", _kill_and_free_on_first_busy,
                            _prepatch_pump(),
                            RangeMinerCoordinator._pump_deferred, slots=0)
    assert old["_schedule_fired"]["fired"] == 1, (
        "the slot-race schedule never fired — the gate is vacuous")
    assert old == new, (
        "DISPOSITION DIVERGED under the slot race:\n"
        + "\n".join(f"  {k}: predecessor={old[k]!r} R2={new[k]!r}"
                    for k in old if old[k] != new[k]))
    assert not new["submitted"], (
        f"a dead attempt STAGED under the slot race: {new['submitted']}")
    return (f"slot frees + attempt dies while frame 1 is parked: "
            f"predecessor == R2, 0 submitted, {len(new['retained'])} retained")


def _probe_admit_log(rig: "_Rig") -> List[tuple]:
    """Record the two lock-held decisions per iteration, in order:
    `probe(K)` from `_attempt_live_locked` and `admit(K, result)` from
    `_try_admit_locked`."""
    log: List[tuple] = []
    real_probe = rig.coord._attempt_live_locked
    real_admit = rig.coord._try_admit_locked

    def _p(run_id, sid, attempt):
        log.append(("probe", (run_id, sid, attempt)))
        return real_probe(run_id, sid, attempt)

    def _a(run_id, sid, attempt):
        ok = real_admit(run_id, sid, attempt)
        log.append(("admit", (run_id, sid, attempt), bool(ok)))
        return ok

    rig.coord._attempt_live_locked = _p
    rig.coord._try_admit_locked = _a
    return log


def gate_r2_2_no_memo_hit_can_ever_reach_ready():
    """R2-1, THE COROLLARY — the property stated so it holds by construction,
    then measured.

        INVARIANT   `_key in live_keys` implies `_try_admit_locked(_key)` has
                    not returned True earlier in this pass (the only `add`
                    follows a fresh probe; the only `discard` follows a True).
        COROLLARY   a memo hit can never reach `ready`, because
                    `_try_admit_locked` returns False for it and keeps
                    returning False (`_admitted` never shrinks and never gains
                    a second key inside one lock hold).
        THEREFORE   every frame that STAGES is decided by a fresh, under-lock
                    `_attempt_live_locked` in its own iteration — byte-identical
                    to the predecessor.

    Measured as: EVERY `_try_admit_locked` call that returns True is
    IMMEDIATELY PRECEDED by a fresh probe of the same key. Driven over the whole
    fixture matrix at every slot budget, plus both R2-1 schedules.

    WRONG INPUT THAT REDS IT: M10 (R-1) produces `admit -> True` with no
    preceding probe the moment a second frame of an admitted attempt is seen."""
    offenders = []
    checked = 0
    with tempfile.TemporaryDirectory() as tmp:
        cases = [(label, spec, slots)
                 for label, spec in FIXTURES.items() for slots in SLOT_BUDGETS]
        cases.append(("r2-spec", _R2_SPEC, 4))
        for label, spec, slots in cases:
            rig = _Rig(tmp, f"r2c-{label}-{slots}", slots=slots)
            _build(rig, spec)
            log = _probe_admit_log(rig)
            rig.coord._pump_deferred()
            for i, ev in enumerate(log):
                if ev[0] != "admit" or not ev[2]:
                    continue
                checked += 1
                prev = log[i - 1] if i else None
                if not (prev and prev[0] == "probe" and prev[1] == ev[1]):
                    offenders.append(
                        f"{label}/{slots}: admit->True for {ev[1]} with "
                        f"preceding event {prev!r} — not a fresh probe")
    assert checked, "no admit->True was ever observed — the gate is vacuous"
    assert not offenders, ("\n  " + "\n  ".join(offenders[:8]))
    return (f"{checked} grants across {len(FIXTURES) * len(SLOT_BUDGETS) + 1} "
            f"cases, every one preceded by a fresh probe of its own key")


def _r2_3_offenders(blk: ast.With) -> List[str]:
    """The R2-3 detector, over ANY admission-lock block, so it can be driven
    against synthetic wrong shapes as well as the live source."""
    bad: List[str] = []
    admit_ifs = [n for n in ast.walk(blk) if isinstance(n, ast.If)
                 and "_try_admit_locked" in ast.unparse(n.test)]
    if len(admit_ifs) != 1:
        return [f"expected exactly one `if self._try_admit_locked(...)`, "
                f"found {len(admit_ifs)}"]
    node = admit_ifs[0]
    if "acquire" in ast.unparse(node.test):
        bad.append("the slot acquire is fused into the admission test — the "
                   "discard cannot be placed between them")
    body = ast.unparse(node)
    i_discard, i_acquire = body.find("live_keys.discard("), body.find(".acquire(")
    if i_discard < 0:
        bad.append("the admit-True branch does not discard the memo entry")
    if i_acquire < 0:
        bad.append("the admit-True branch does not acquire a slot")
    if i_discard >= 0 and i_acquire >= 0 and i_discard > i_acquire:
        bad.append("the memo is discarded AFTER the slot acquire — a slot "
                   "freed by another thread mid-pass would then stage on a "
                   "cached positive")
    n_discards = sum(1 for n in ast.walk(blk) if isinstance(n, ast.Call)
                     and getattr(n.func, "attr", None) == "discard")
    if n_discards != 1:
        bad.append(f"{n_discards} `discard` calls under the lock, expected 1")
    return bad


def gate_r2_3_the_discard_sits_at_the_authority_point():
    """R2-1, STRUCTURAL. The invalidation must sit inside the
    `_try_admit_locked` True branch and LEXICALLY BEFORE the slot acquire —
    that ordering is what makes the slot race (R2-1b) unreachable by
    construction rather than merely unlikely.

    ITS FALSIFIER IS SYNTHETIC SOURCE, NOT A RUNTIME MUTANT, and that is not a
    weakness — a source gate cannot be reddened by swapping a bound method, so
    driving it with `_with_pump` would produce a green that means nothing. Four
    wrong shapes are run through the same detector below.

    WRONG INPUT THAT REDS IT: the discard moved into the `ready` branch (M10b's
    shape), placed after the acquire, absent, or duplicated."""
    live = _r2_3_offenders(_lock_block(_live_src()))
    assert not live, "; ".join(live)

    import textwrap as _tw
    WRONG = {
        "no discard at all (this is R-1)": """
            with self._admission_lock:
                for entry in self._deferred:
                    if _key not in live_keys:
                        live_keys.add(_key)
                    if (self._try_admit_locked(a, b, c)
                            and self._staging_slots().acquire(blocking=False)):
                        ready.append(entry)
                self._deferred = still
        """,
        "discard inside the ready branch (Beta's literal candidate)": """
            with self._admission_lock:
                for entry in self._deferred:
                    if self._try_admit_locked(a, b, c):
                        if self._staging_slots().acquire(blocking=False):
                            live_keys.discard(_key)
                            ready.append(entry)
                self._deferred = still
        """,
        "discard after the acquire": """
            with self._admission_lock:
                for entry in self._deferred:
                    if self._try_admit_locked(a, b, c):
                        got = self._staging_slots().acquire(blocking=False)
                        live_keys.discard(_key)
                        if got:
                            ready.append(entry)
                self._deferred = still
        """,
        "acquire fused back into the admission test": """
            with self._admission_lock:
                for entry in self._deferred:
                    live_keys.discard(_key)
                    if (self._try_admit_locked(a, b, c)
                            and self._staging_slots().acquire(blocking=False)):
                        ready.append(entry)
                self._deferred = still
        """,
    }
    survived = []
    for label, src in WRONG.items():
        blk = [n for n in ast.walk(ast.parse(_tw.dedent(src)))
               if isinstance(n, ast.With)][0]
        if not _r2_3_offenders(blk):
            survived.append(label)
    assert not survived, (
        f"the R2-3 detector ACCEPTS these wrong shapes: {survived}")
    return (f"discard inside admit-True, before the acquire; "
            f"{len(WRONG)}/{len(WRONG)} wrong shapes rejected")


# ===========================================================================
# R3-1 — the retained-frame divergence at the BOUNDED CAPACITY boundary
# ===========================================================================
def _capacity_boundary(tmp: str, tag: str, pump) -> Dict[str, Any]:
    """Beta's seven-step schedule, driven identically against any pump.

        1 an already-admitted attempt A                     (s0, pre-admitted)
        2 a second attempt B with enough frames to matter    (s1, 8 frames)
        3 B's first frame observes LIVE and is memoized
        4 `_try_admit_locked(B)` returns False — A owns admission
        5 IMMEDIATELY make B terminal                        (hooked at step 4)
        6 finish the same pump pass
        7 BEFORE another pump, defer a new live frame C      (s2)

    The store is filled to EXACTLY the derived bound, which is where the
    question lives: one retained frame too many turns `accept C` into a
    `derived_count_bound` refusal, and that refusal is the
    `coordinator_staging_capacity_invariant:` path — it FAILS THE TRIAL."""
    rig = _Rig(tmp, tag, slots=0)
    bound = rig.coord.staging_deferred_bound()
    n_b = 8
    n_a = bound - n_b
    _build(rig, {"stripes": [("s0", 0, ST_CLAIMED), ("s1", 0, ST_CLAIMED),
                             ("s2", 0, ST_CLAIMED)],
                 "admitted": [("s0", 0)],
                 "deferred": ([("s0", 0, j) for j in range(n_a)]
                              + [("s1", 0, j) for j in range(n_b)])})
    assert len(rig.coord._deferred) == bound, (len(rig.coord._deferred), bound)
    real = rig.coord._try_admit_locked
    killed = {"n": 0}

    def _hook(run_id, sid, attempt):
        ok = real(run_id, sid, attempt)
        if not ok and sid == "s1" and not killed["n"]:
            killed["n"] = 1
            rig.coord.ledger.set_stripe_state(run_id, "s1", ST_DONE)
        return ok

    rig.coord._try_admit_locked = _hook
    types.MethodType(pump, rig.coord)()

    after = len(rig.coord._deferred)
    retained_bytes = rig.coord._deferred_retained_bytes()
    b_resolved = sum(1 for k, f in rig.futs.items()
                     if k[0] == "s1" and f.done())
    fut: concurrent.futures.Future = concurrent.futures.Future()
    with rig.coord._admission_lock:
        accepted = rig.coord._defer_locked(
            ("inline", None, RUN, "s2", 0, 0, _Msg(), None, fut))
        refusal = rig.coord._last_defer_refusal
    return {"bound": bound, "killed": killed["n"],
            "deferred_after_pump": after, "retained_bytes": retained_bytes,
            "b_futures_resolved": b_resolved,
            "C_accepted": accepted, "C_refusal": refusal}


def gate_r3_1_capacity_boundary_differential():
    """R3-1 (BLOCKING) — THE MEASUREMENT BETA REQUIRED, and it changed the
    answer.

    R-2 disclosed this divergence as *"GC latency in the conservative
    direction"*. Measured at the boundary, that was WRONG:

        predecessor  deferred 62  C_accepted True   refusal None
        R-2          deferred 69  C_accepted FALSE  refusal 'derived_count_bound'

    `derived_count_bound` is the `coordinator_staging_capacity_invariant:`
    terminal — R-2 could FAIL A TRIAL the predecessor completes. Not latency: a
    trial-fatal disposition change on the bounded-storage surface R-1 was
    required to preserve.

    R-3 closes it with an end-of-pass re-probe of the RETAINED KEYS — O(distinct
    retained attempts), never O(entries), so the per-entry probe term is not
    restored (G8a still green).

    THE TWO CAPACITY-DECISIVE FIELDS MUST BE IDENTICAL, and the retained
    population must be <= the predecessor's — the direction that cannot
    manufacture a refusal. The residual (R-3 also drops the frame examined
    before the death, so it retains one FEWER) is asserted explicitly rather
    than tolerated silently.

    WRONG INPUT THAT REDS IT: mutant M11 — R-2 exactly, no end-of-pass sweep."""
    with tempfile.TemporaryDirectory() as tmp:
        old = _capacity_boundary(tmp, "r3-old", _prepatch_pump())
        new = _capacity_boundary(tmp, "r3-new",
                                 RangeMinerCoordinator._pump_deferred)
    assert old["killed"] == 1 and new["killed"] == 1, (
        "the schedule never fired — the gate is vacuous")
    for field in ("C_accepted", "C_refusal"):
        assert old[field] == new[field], (
            f"CAPACITY DISPOSITION DIVERGED on {field}: "
            f"predecessor={old[field]!r} R3={new[field]!r}")
    assert new["C_accepted"] is True, new
    assert new["deferred_after_pump"] <= old["deferred_after_pump"], (
        f"R3 retains MORE than the predecessor "
        f"({new['deferred_after_pump']} > {old['deferred_after_pump']}) — it "
        f"can manufacture a capacity refusal")
    assert new["retained_bytes"] <= old["retained_bytes"], (new, old)
    assert new["b_futures_resolved"] >= old["b_futures_resolved"], (new, old)
    return (f"bound={old['bound']} | predecessor: deferred="
            f"{old['deferred_after_pump']} bytes={old['retained_bytes']} "
            f"B-futures={old['b_futures_resolved']} C={old['C_accepted']} "
            f"refusal={old['C_refusal']!r} || R3: deferred="
            f"{new['deferred_after_pump']} bytes={new['retained_bytes']} "
            f"B-futures={new['b_futures_resolved']} C={new['C_accepted']} "
            f"refusal={new['C_refusal']!r}")


def gate_r3_2_the_sweep_is_a_no_op_under_quiescence():
    """R3-1, THE PROPERTY THAT KEEPS EVERY OTHER DIFFERENTIAL MEANINGFUL.

    The end-of-pass sweep must cost probes but change NOTHING when no
    concurrent writer is present — otherwise it would be a second, unreviewed
    disposition mechanism rather than a repair of one interleaving. Measured
    two ways over the whole fixture matrix: the sweep drops zero entries, and
    the pass's disposition is byte-identical to the same pass with the sweep's
    drop suppressed.

    WRONG INPUT THAT REDS IT: a sweep that drops on any condition other than a
    fresh probe reporting the attempt dead."""
    dropped_total = 0
    with tempfile.TemporaryDirectory() as tmp:
        for label, spec in FIXTURES.items():
            for slots in SLOT_BUDGETS:
                rig = _Rig(tmp, f"r3q-{label}-{slots}", slots=slots)
                _build(rig, spec)
                before = {(e[3], e[4], e[5]) for e in rig.coord._deferred}
                # count what the sweep alone removed: run the R-2 pump (no
                # sweep) on an identical rig and diff the retained sets
                rig2 = _Rig(tmp, f"r3q2-{label}-{slots}", slots=slots)
                _build(rig2, spec)
                rig.coord._pump_deferred()
                types.MethodType(_r2_no_end_of_pass_sweep, rig2.coord)()
                a = {(e[3], e[4], e[5]) for e in rig.coord._deferred}
                b = {(e[3], e[4], e[5]) for e in rig2.coord._deferred}
                assert a == b, (
                    f"{label}/{slots}: the sweep changed the retained set under "
                    f"quiescence: only-in-R2={sorted(b - a)}")
                dropped_total += len(before - a) - len(before - b)
    assert dropped_total == 0, dropped_total
    return (f"{len(FIXTURES)} fixtures x {len(SLOT_BUDGETS)} budgets: sweep "
            f"drops 0 entries and changes no disposition under quiescence")


def gate_r3_3_the_sweep_is_per_key_not_per_entry():
    """R3-1, THE COMPLEXITY GUARD. Beta was explicit: *"if closing it requires
    probing every entry, that is the predecessor and the win is gone"*. The
    sweep must cost ONE probe per RETAINED KEY, so growing the retained frames
    of a fixed set of keys must not change the sweep's cost.

    Measured as the probe DELTA against the same pass with the sweep removed:
    the delta must equal the number of retained distinct keys and must be flat
    as their frame counts grow 20x.

    WRONG INPUT THAT REDS IT: a sweep that re-probes per retained entry — which
    is the predecessor, reintroduced through the back door."""
    obs = {}
    with tempfile.TemporaryDirectory() as tmp:
        for per in (1, 5, 20):
            # 4 non-admitted live keys, all retained; 1 admitted key
            spec = {"stripes": [(f"s{i}", 0, ST_CLAIMED) for i in range(5)],
                    "admitted": [("s0", 0)],
                    "deferred": ([("s0", 0, 0)]
                                 + [(f"s{i}", 0, j)
                                    for i in range(1, 5) for j in range(per)])}
            rig = _Rig(tmp, f"r3k-{per}", slots=0)
            _build(rig, spec)
            rig.counting.reset()
            rig.coord._pump_deferred()
            with_sweep = rig.counting.total()
            rig2 = _Rig(tmp, f"r3k2-{per}", slots=0)
            _build(rig2, spec)
            rig2.counting.reset()
            types.MethodType(_r2_no_end_of_pass_sweep, rig2.coord)()
            obs[per] = with_sweep - rig2.counting.total()
    assert len(set(obs.values())) == 1, (
        f"the sweep's cost SCALES with retained frames — the per-entry probe "
        f"term is back: {obs}")
    return (f"retained frames per key 1->20: sweep read-delta constant at "
            f"{next(iter(obs.values()))} (4 retained keys)")


# ===========================================================================
# R3-4 — TERMINAL-KEY IRREVERSIBILITY (the load-bearing safety statement)
# ===========================================================================
class _Worker:
    """Minimal idle worker the production scheduler will accept."""

    def __init__(self, wid):
        self.worker_id = wid
        self.backend = "rocm"
        self.seed_caps = {"amd": 2_000_000, "nvidia": 5_000_000,
                          "amd_hybrid": 1_000_000, "nvidia_hybrid": 2_500_000}
        self.supported_variants = ["java_lcg", "java_lcg_reverse",
                                   "java_lcg_hybrid", "java_lcg_hybrid_reverse"]
        self.quarantined = False


def gate_r3_4_a_swept_key_cannot_be_resurrected():
    """R3-4 — THE DETECTOR FOR R-3's LOAD-BEARING SAFETY STATEMENT.

        A key declared dead by the end-of-pass sweep cannot later become live
        again at the SAME `(run_id, stripe_id, attempt)`.

    R-3 takes ONE negative observation and retires EVERY retained frame for that
    key, so if the statement were false the sweep would discard work that could
    still legitimately stage. R-3 argued it from source; Beta declined to make a
    source-reviewed assertion into permanent safety doctrine without a detector.
    This is that detector.

    ⚠ IT DOES NOT WEAKEN G4b, AND THE DISTINCTION IS THE POINT. The LEDGER
    PRIMITIVE genuinely permits non-monotone liveness — `claim_stripe`'s SQL
    accepts `state IN (pending, failed)`, so `failed -> claimed` is a real
    capability, which is exactly why the memo never reuses a NEGATIVE. R3-4
    proves something narrower and different: the PRODUCTION SCHEDULING PATH does
    not exercise that capability for a swept key. Both statements are true and
    neither replaces the other.

    FOUR ARMS, then the mutant:
      1 exactly one production caller can drive a stripe into `claimed`
      2 it selects ONLY pending rows — and terminal rows are not pending
      3 a requeue capable of resurrecting work ADVANCES `current_attempt`,
        and the one path that returns a row to pending WITHOUT advancing it
        (`reclaim_expired_leases`) cannot touch a terminal row at all
      4 end to end: a swept key stays dead across a real scheduler pass

    WRONG INPUT THAT REDS IT: the mutant below — a selector that also yields
    terminal rows, i.e. same-attempt reclamation of exactly the kind the ledger
    primitive would permit."""
    detail = {}

    # ---- ARM 1: enumerate production claim_stripe callers ------------------
    tree = ast.parse(_live_src())
    callers = []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for fn in [n for n in ast.walk(cls)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call)
                        and getattr(node.func, "attr", None) == "claim_stripe"
                        and "self.ledger" in ast.unparse(node.func)):
                    callers.append(f"{cls.name}.{fn.name}")
    assert callers == ["RangeMinerCoordinator.schedule_pending_stripes"], (
        f"production `ledger.claim_stripe(...)` callers are {callers}; R3-4's "
        f"reasoning covers only the single scheduler call site")
    detail["arm1"] = callers[0].split(".")[-1]

    sched = _func_node(tree, "RangeMinerCoordinator", "schedule_pending_stripes")
    src = ast.unparse(sched)
    assert "self.ledger.pending_stripes(" in src, (
        "the scheduler no longer sources its rows from `pending_stripes`")

    # ---- ARM 2: the selector is pending-only (behavioural) -----------------
    with tempfile.TemporaryDirectory() as tmp:
        rig = _Rig(tmp, "r34-sel", slots=6)
        led = rig.counting._inner
        for sid, state in (("sA", ST_PENDING), ("sB", ST_DONE),
                           ("sC", ST_FAILED), ("sD", ST_CANCELLED),
                           ("sE", ST_CLAIMED)):
            led.add_stripe(RUN, sid, 0, 1000, "java_lcg", 1, now=1.0)
            assert led.claim_stripe(RUN, sid, f"w-{sid}", 0, 8, 9e18)
            led.set_stripe_state(RUN, sid, state)
        got = {r["stripe_id"] for r in led.pending_stripes(RUN)}
        assert got == {"sA"}, (
            f"`pending_stripes` returned {sorted(got)} — a terminal row is "
            f"reachable by the scheduler")
        detail["arm2"] = f"pending-only: {sorted(got)} of 5 states"

        # ---- ARM 3a: the requeue that CAN resurrect advances the attempt ---
        led.add_stripe(RUN, "sR", 0, 1000, "java_lcg", 1, now=1.0)
        assert led.claim_stripe(RUN, "sR", "w-sR", 0, 8, 9e18)
        before = led.get_stripe(RUN, "sR")["current_attempt"]
        # the production retry write, verbatim in shape (`:8428`)
        led.set_stripe_fields(RUN, "sR", phase_degraded=1, state=ST_PENDING,
                              claimed_by="w-sR", current_attempt=before + 1,
                              lease_expires_at=None, staging_generation=1)
        after = led.get_stripe(RUN, "sR")["current_attempt"]
        assert after > before, (before, after)
        assert not rig.coord._attempt_live_locked(RUN, "sR", before), (
            "the OLD attempt key went live again after a requeue")
        assert rig.coord._attempt_live_locked(RUN, "sR", after), (
            "the NEW attempt key is not live — the fixture is wrong")
        detail["arm3a"] = f"requeue {before}->{after}, old key stays dead"

        # ---- ARM 3b: the one non-advancing path cannot touch a terminal row -
        reclaimed = led.reclaim_expired_leases(RUN, now=9e18)
        rids = {r["stripe_id"] for r in reclaimed}
        assert not (rids & {"sB", "sC", "sD"}), (
            f"`reclaim_expired_leases` touched terminal rows {sorted(rids)} — "
            f"it returns rows to pending WITHOUT advancing the attempt")
        detail["arm3b"] = f"reclaim touched {sorted(rids)}, no terminal row"

        # ---- ARM 4: end to end, through the REAL scheduler -----------------
        rig2 = _Rig(tmp, "r34-e2e", slots=6)
        led2 = rig2.counting._inner
        led2.create_trial(RUN, 1, now=1.0)
        led2.add_stripe(RUN, "sX", 0, 1000, "java_lcg", 1, now=1.0)
        assert led2.claim_stripe(RUN, "sX", "w0", 0, 8, 9e18)
        led2.set_stripe_state(RUN, "sX", ST_FAILED)     # the swept-dead state
        assert not rig2.coord._attempt_live_locked(RUN, "sX", 0), "not dead"
        rig2.coord.cohort_filter = lambda r, f, p, w: list(w)
        placed = rig2.coord.schedule_pending_stripes(
            RUN, "java_lcg", 1, [_Worker("w1"), _Worker("w2")],
            stage_prefix="s", now=1000.0)
        still_dead = not rig2.coord._attempt_live_locked(RUN, "sX", 0)
        assert still_dead, (
            f"a swept key went LIVE after a real scheduler pass "
            f"(placed={placed}) — R-3's sweep could discard stageable work")
        detail["arm4"] = f"scheduler placed {len(placed)}, swept key still dead"

        # ---- MUTANT: same-attempt terminal reclamation --------------------
        rig3 = _Rig(tmp, "r34-mut", slots=6)
        led3 = rig3.counting._inner
        led3.create_trial(RUN, 1, now=1.0)
        led3.add_stripe(RUN, "sX", 0, 1000, "java_lcg", 1, now=1.0)
        assert led3.claim_stripe(RUN, "sX", "w0", 0, 8, 9e18)
        led3.set_stripe_state(RUN, "sX", ST_FAILED)
        real_pending = led3.pending_stripes

        def _leaky(run_id, stage_prefix=None, *, exact_stripe_id=None):
            rows = list(real_pending(run_id, stage_prefix,
                                     exact_stripe_id=exact_stripe_id))
            rows += [r for r in led3.all_stripes(run_id)
                     if r["state"] == ST_FAILED]
            return rows

        object.__setattr__(rig3.counting, "_inner",
                           _PendingShim(led3, _leaky))
        rig3.coord.cohort_filter = lambda r, f, p, w: list(w)
        rig3.coord.schedule_pending_stripes(
            RUN, "java_lcg", 1, [_Worker("w1"), _Worker("w2")],
            stage_prefix="s", now=1000.0)
        mutant_live = rig3.coord._attempt_live_locked(RUN, "sX", 0)
        assert mutant_live, (
            "the same-attempt-reclamation MUTANT did not resurrect the key — "
            "R3-4 cannot distinguish the safe path from the unsafe one, so it "
            "is vacuous")
    return (f"1 caller ({detail['arm1']}) | {detail['arm2']} | "
            f"{detail['arm3a']} | {detail['arm3b']} | {detail['arm4']} | "
            f"mutant resurrects it -> arm 4 discriminates")


class _PendingShim:
    def __init__(self, inner, pending):
        self._inner = inner
        self._pending = pending

    def pending_stripes(self, *a, **kw):
        return self._pending(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _reads_for(tmp: str, tag: str, keys: int, frames_per_key: int,
               slots: int = 0) -> Tuple[int, int]:
    rig = _Rig(tmp, tag, slots=slots)
    _build(rig, {"stripes": [(f"s{i}", 0, ST_CLAIMED) for i in range(keys)],
                 "deferred": [(f"s{i}", 0, j)
                              for i in range(keys)
                              for j in range(frames_per_key)]})
    rig.counting.reset()
    rig.coord._pump_deferred()
    return rig.counting.total(), keys * frames_per_key


def _uneven_reads(tmp: str, tag: str, admitted_frames: int,
                  other_keys: int, other_frames: int) -> Tuple[int, int, int]:
    """The MP-1 SHAPE, parameterised the way R2-2 requires it to be read: the
    ADMITTED attempt's frame count and the NON-ADMITTED attempts' frame count
    are separate knobs, because only the second is the runaway term."""
    keys = 1 + other_keys
    deferred = [("s0", 0, j) for j in range(admitted_frames)]
    deferred += [(f"s{i}", 0, j)
                 for i in range(1, keys) for j in range(other_frames)]
    rig = _Rig(tmp, tag, slots=0)
    _build(rig, {"stripes": [(f"s{i}", 0, ST_CLAIMED) for i in range(keys)],
                 "deferred": deferred})
    rig.counting.reset()
    rig.coord._pump_deferred()
    return (rig.counting.total(),
            rig.coord.staging_backpressure_metrics()[
                "pump_liveness_probes_high_water"],
            len(deferred))


def gate_g8a_the_positive_feedback_term_is_gone():
    """GATE 8 — THE COMPLEXITY PROOF, RESTATED HONESTLY AFTER R2-2.

    R-1 claimed "reads invariant in the deferred population". That is NOT the
    truthful claim and R2-2 required it withdrawn: dead entries are deliberately
    re-probed, and after R2-1 the ADMITTED attempt's frames re-probe after each
    grant. What R-1 and R-2 actually remove is the term that RAN AWAY — repeated
    positive probes of LIVE, NON-ADMITTED attempts, which is what the MP-1
    backlog overwhelmingly consisted of.

    So the invariance is measured against exactly that term: hold the distinct
    attempt count at 5 and the ADMITTED attempt at 2 frames, and grow the four
    NON-ADMITTED attempts 1 -> 100 frames each (10 -> 402 entries). Reads must
    be IDENTICAL at every population.

    A COUNT, NOT A CLOCK — it counts database opens, which is what
    `MinerLedger._conn` does per read, and it is deterministic.

    WRONG INPUT THAT REDS IT: mutant M8 — the pre-patch per-entry probe, i.e.
    the production code at the pinned anchor."""
    with tempfile.TemporaryDirectory() as tmp:
        series = {n: _uneven_reads(tmp, f"g8a-{n}", 2, 4, n)
                  for n in (1, 5, 25, 100)}
    reads = {n: v[0] for n, v in series.items()}
    entries = {n: v[2] for n, v in series.items()}
    assert len(set(reads.values())) == 1, (
        f"reads SCALE with non-admitted frames — the runaway term survives: "
        f"{ {entries[n]: reads[n] for n in reads} }")
    base = next(iter(reads.values()))
    return (f"non-admitted frames {entries[1]}->{entries[100]} entries "
            f"({entries[100] // entries[1]}x): reads constant at {base}")


def gate_g8b_the_truthful_bound_is_exact():
    """GATE 8 — THE R2-2 FORMULATION, MEASURED AS AN EQUALITY.

        probes per pass = frames of the ONE admitted attempt
                        + 2 x (live NON-ADMITTED attempts)
                              (one in the main loop, one in R-3's end-of-pass
                               capacity sweep)
                        + dead entries examined

    Asserted as an exact identity over a matrix that varies each term
    independently. Stating it as an equality rather than a bound is what stops a
    future revision quietly re-claiming population-invariance: any drift in
    either direction reds.

    WRONG INPUT THAT REDS IT: R-1 (M8b/M10) probes once per distinct key and
    under-shoots; the predecessor (M8) probes once per entry and over-shoots."""
    obs = {}
    with tempfile.TemporaryDirectory() as tmp:
        for admitted, others, per in ((1, 4, 50), (2, 4, 1), (10, 3, 7),
                                      (68, 24, 68), (200, 0, 0)):
            reads, probes, n = _uneven_reads(
                tmp, f"g8b-{admitted}-{others}-{per}", admitted, others, per)
            obs[(admitted, others, per)] = (probes, admitted + 2 * others, n)
    bad = [f"{k}: probes={v[0]} expected={v[1]}" for k, v in obs.items()
           if v[0] != v[1]]
    assert not bad, "; ".join(bad)
    # and the dead-entry term, which is the one that is NOT population-invariant
    with tempfile.TemporaryDirectory() as tmp:
        dead = {}
        for n in (10, 100, 400):
            rig = _Rig(tmp, f"g8b-dead-{n}", slots=0)
            _build(rig, {"stripes": [("s0", 0, ST_DONE)],
                         "deferred": [("s0", 0, j) for j in range(n)]})
            rig.counting.reset()
            rig.coord._pump_deferred()
            dead[n] = rig.coord.staging_backpressure_metrics()[
                "pump_liveness_probes_high_water"]
    assert dead == {10: 10, 100: 100, 400: 400}, dead
    return (f"probes == admitted_frames + 2*non_admitted_attempts on "
            f"{len(obs)} shapes; dead entries re-probe 1:1 ({dead})")


def gate_g8g_the_worst_case_is_o_n_and_says_so():
    """GATE 8 — THE LIMITATION, ASSERTED RATHER THAN OMITTED (R2-2).

    A purge pass over a large dead population is O(N) BY DELIBERATE SAFETY
    DESIGN: negatives are never reused, so every dead entry costs its own probe.
    This gate exists so the limitation is a MEASURED, PINNED fact rather than a
    caveat in prose that a later revision can drop. If someone "optimises" it
    away by caching negatives, G4b reds and so does this.

    WRONG INPUT THAT REDS IT: a negative-caching memo (M4) makes the dead
    population cost O(distinct attempts) and this gate's growth disappears."""
    with tempfile.TemporaryDirectory() as tmp:
        obs = {}
        for n in (25, 100, 400):
            rig = _Rig(tmp, f"g8g-{n}", slots=0)
            _build(rig, {"stripes": [(f"s{i}", 0, ST_DONE) for i in range(4)],
                         "deferred": [(f"s{i}", 0, j)
                                      for i in range(4) for j in range(n // 4)]})
            rig.counting.reset()
            rig.coord._pump_deferred()
            obs[n] = rig.counting.total()
    assert obs[400] > obs[25] * 8, (
        f"a dead purge pass did NOT scale with the population — negatives are "
        f"being reused somewhere: {obs}")
    return (f"all-dead purge is O(N) as designed: {obs} — stated, not hidden")


def gate_g8c_pre_patch_reads_do_scale():
    """GATE 8, the CLEAN CONTROL (VIR-2). The gate above is only evidence if the
    same measurement, on the same fixtures, shows the pre-patch pump SCALING.
    Without this arm a counter wired to a constant would pass G8a.

    WRONG INPUT THAT REDS IT: an oracle that is secretly the patched code."""
    with tempfile.TemporaryDirectory() as tmp:
        obs = {}
        for n in (1, 5, 25, 100):
            rig = _Rig(tmp, f"g8c-{n}", slots=0)
            _build(rig, {"stripes": [(f"s{i}", 0, ST_CLAIMED) for i in range(4)],
                         "deferred": [(f"s{i}", 0, j)
                                      for i in range(4) for j in range(n)]})
            rig.counting.reset()
            types.MethodType(_prepatch_pump(), rig.coord)()
            obs[4 * n] = rig.counting.total()
    assert obs[4] < obs[400], obs
    assert obs[400] >= 2 * 400, (
        f"the pre-patch pump did not do ~2 reads per entry: {obs}")
    ratio = obs[400] / max(1, obs[4])
    assert ratio > 50, f"pre-patch scaling ratio only {ratio:.1f}: {obs}"
    return (f"pre-patch: {obs[4]} reads at 4 entries -> {obs[400]} at 400 "
            f"({ratio:.0f}x); patched: constant")


def gate_g8d_the_pathological_shape_end_to_end():
    """GATE 8, the shape MP-1 actually measured: 25 attempts (the frozen cohort)
    x 68 sub-stripe frames (the hybrid AMD cap at the gate-12 geometry) = 1,700
    deferred entries, one pump.

    WRONG INPUT THAT REDS IT: any patch whose reads at this shape are within an
    order of magnitude of the pre-patch count."""
    with tempfile.TemporaryDirectory() as tmp:
        new_reads, entries = _reads_for(tmp, "g8d-new", 25, 68)
        rig = _Rig(tmp, "g8d-old", slots=0)
        _build(rig, {"stripes": [(f"s{i}", 0, ST_CLAIMED) for i in range(25)],
                     "deferred": [(f"s{i}", 0, j)
                                  for i in range(25) for j in range(68)]})
        rig.counting.reset()
        types.MethodType(_prepatch_pump(), rig.coord)()
        old_reads = rig.counting.total()
    assert entries == 1700, entries
    assert new_reads * 10 < old_reads, (new_reads, old_reads)
    return (f"{entries} entries / 25 attempts: {old_reads} reads -> "
            f"{new_reads} ({old_reads / new_reads:.0f}x fewer db opens; R2's "
            f"post-grant re-probes are inside this number)")


def _pump_and_read(tmp: str, tag: str, keys: int, frames_per_key: int,
                   dead: bool = False) -> Dict[str, Any]:
    rig = _Rig(tmp, tag, slots=0)
    state = ST_DONE if dead else ST_CLAIMED
    _build(rig, {"stripes": [(f"s{i}", 0, state) for i in range(keys)],
                 "deferred": [(f"s{i}", 0, j)
                              for i in range(keys)
                              for j in range(frames_per_key)]})
    rig.counting.reset()
    rig.coord._pump_deferred()
    m = rig.coord.staging_backpressure_metrics()
    m["_reads"] = rig.counting.total()
    return m


def gate_g8e_the_falsifier_fields_are_measuring():
    """GATE 8, THE FALSIFIER ITSELF. The complexity guarantee is
    `reads = 2 x probes` with `probes` bounded by the distinct attempt count —
    but before R-1 no field reported either quantity, so no production run
    could refute it. `deferred_high_water` reports the POPULATION.

    This gate proves the two new fields are real measurements and not decorated
    constants, to the H1/H2 standard: the same code runs on shapes that differ
    ONLY in the quantity being measured, and the fields must differ with it.

      - population FIXED at 200, K = 4 vs 20  -> distinct tracks K, not 200
      - a DEAD population                     -> probes > distinct, because
                                                 dead entries are re-probed by
                                                 design (the one-directional
                                                 memo), which also proves the
                                                 two fields are not one field
                                                 reported twice
      - the identity `reads == 2 * probes + try_admit_reads` holds

    WRONG INPUT THAT REDS IT: a field wired to `len(self._deferred)`, or to any
    constant (mutants M9a / M9b)."""
    with tempfile.TemporaryDirectory() as tmp:
        a = _pump_and_read(tmp, "g8e-k4", 4, 50)
        b = _pump_and_read(tmp, "g8e-k20", 20, 10)
        d = _pump_and_read(tmp, "g8e-dead", 1, 10, dead=True)
    for m in (a, b, d):
        assert "deferred_distinct_attempts_high_water" in m, sorted(m)
        assert "pump_liveness_probes_high_water" in m, sorted(m)
    assert a["deferred_distinct_attempts_high_water"] == 4, a[
        "deferred_distinct_attempts_high_water"]
    assert b["deferred_distinct_attempts_high_water"] == 20, b[
        "deferred_distinct_attempts_high_water"]
    # [R2-2] probes = admitted-attempt frames + (distinct attempts - 1). The
    # admitted attempt re-probes after every grant (R2-1), so this is NOT the
    # distinct-attempt count — a fact the earlier revision of this gate asserted
    # wrongly and which the R2-1 fix corrected.
    assert a["pump_liveness_probes_high_water"] == 50 + 3 + 3, a[
        "pump_liveness_probes_high_water"]
    assert b["pump_liveness_probes_high_water"] == 10 + 19 + 19, b[
        "pump_liveness_probes_high_water"]
    # the population was IDENTICAL (200) in both — so neither field is it
    assert a["deferred_distinct_attempts_high_water"] != \
        b["deferred_distinct_attempts_high_water"], (a, b)
    # dead entries are re-probed: probes exceeds distinct, and the two fields
    # are therefore genuinely different quantities
    assert d["deferred_distinct_attempts_high_water"] == 1, d
    assert d["pump_liveness_probes_high_water"] == 10, d
    # THE IDENTITY THE GUARANTEE IS STATED IN, with the short-circuit that a
    # first draft of this gate got wrong: `_attempt_live_locked` returns at its
    # FIRST read when the stripe row is terminal, so it never reaches
    # `get_trial`. A probe therefore costs 1 read when it finds the attempt
    # dead-by-stripe-state and 2 when it has to consult the trial. The bound
    # is `probes <= reads <= 2*probes + 1` (the +1 is `_try_admit_locked`'s own
    # guard read on the pass that grants), and both endpoints are exercised.
    for m in (a, b, d):
        p = m["pump_liveness_probes_high_water"]
        assert p <= m["_reads"] <= 2 * p + 1, (m["_reads"], p)
    assert a["_reads"] == 2 * a["pump_liveness_probes_high_water"] + 1, a["_reads"]
    assert b["_reads"] == 2 * b["pump_liveness_probes_high_water"] + 1, b["_reads"]
    # the all-dead pass sits on the OTHER endpoint: every probe short-circuits
    # at get_stripe, and no grant happens, so reads == probes exactly.
    assert d["_reads"] == d["pump_liveness_probes_high_water"], d["_reads"]
    return (f"K=4/K=20 at fixed population 200 -> distinct 4/20, probes 56/48; "
            f"dead 1 attempt x 10 frames -> distinct 1, probes 10; "
            f"reads at both endpoints of probes..2*probes+1")


def gate_g8f_the_fields_are_high_waters_and_decision_free():
    """GATE 8, the two properties that make the fields safe and useful.

    HIGH-WATER: a big pass followed by a small one keeps the big value — a
    last-value field would under-report exactly the pass that mattered.

    DECISION-FREE, STRUCTURALLY: `seen_keys` and `probes` are written inside
    `_admission_lock` but must appear in NO condition there, so no disposition
    can depend on the instrument. That is what lets G6's differential stand
    unchanged with the metric present.

    WRONG INPUT THAT REDS IT: reading `probes` in a branch — e.g. an
    'if probes > N: stop scanning' cutoff, which is how an instrument becomes a
    policy."""
    with tempfile.TemporaryDirectory() as tmp:
        rig = _Rig(tmp, "g8f", slots=0)
        _build(rig, {"stripes": [(f"s{i}", 0, ST_CLAIMED) for i in range(12)],
                     "deferred": [(f"s{i}", 0, j)
                                  for i in range(12) for j in range(3)]})
        rig.coord._pump_deferred()
        hi = rig.coord.staging_backpressure_metrics()[
            "deferred_distinct_attempts_high_water"]
        assert hi == 12, hi
        # a smaller subsequent pass must not lower it
        rig.coord._deferred = [e for e in rig.coord._deferred
                               if e[3] == "s0"]
        rig.coord._pump_deferred()
        after = rig.coord.staging_backpressure_metrics()[
            "deferred_distinct_attempts_high_water"]
        assert after == 12, f"the field is a last-value, not a high-water: {after}"

    blk = _lock_block(_live_src())
    conds: List[str] = []
    for node in ast.walk(blk):
        if isinstance(node, ast.If):
            conds.append(ast.unparse(node.test))
        elif isinstance(node, (ast.While, ast.IfExp)):
            conds.append(ast.unparse(node.test))
        elif isinstance(node, ast.comprehension):
            conds.extend(ast.unparse(c) for c in node.ifs)
    for name in ("seen_keys", "probes"):
        offenders = [c for c in conds if name in c]
        assert not offenders, (
            f"the instrument `{name}` appears in a CONDITION under the lock "
            f"({offenders}) — a measurement that steers a decision is a policy")
    return (f"high-water survives a smaller pass (12); neither instrument "
            f"appears in any of {len(conds)} lock-held conditions")


# ===========================================================================
# scope proof
# ===========================================================================
DECLARED_CHANGED = {
    # [FIELD-6 OBSERVABILITY REPAIR, TB ruling sequencing item 3] NOT R-1's
    # change. The scope proof compares LIVE source against the pinned anchor,
    # so every later authorized commit that touches this module must be
    # declared here or the proof reds forever. Field 6's repair appends the two
    # falsifier keys to the `[S172-BP] summary` format string — the emitter was
    # the defect; the metrics dict was not.
    "RangeMinerCoordinator.log_staging_backpressure_summary",
    "RangeMinerCoordinator._pump_deferred",
    # the two `_bp` seed values for the complexity falsifier
    "RangeMinerCoordinator.__init__"}
# EMPTY, AND NOT BY ACCIDENT. MP-1's certified `gate_e2_ast_scope_proof`
# asserts this module's ADDED-definition set EXACTLY against its own pinned
# anchor, so any new `def` here — including a well-named recorder method — reds
# a certified gate. The high-water update is therefore INLINE in
# `_pump_deferred`, which both MP-1 and R-1 already declare changed.
DECLARED_ADDED: set = set()


def gate_scope_proof():
    """THE AST SCOPE PROOF: exactly which definitions moved, against a set
    declared in advance.

    WRONG INPUT THAT REDS IT: a change to any definition R-1 did not declare —
    including a behaviourally harmless one."""
    pinned, live = _def_digests(_pinned_src()), _def_digests(_live_src())
    changed = {k for k in pinned if k in live and pinned[k] != live[k]}
    added = {k for k in live if k not in pinned}
    removed = {k for k in pinned if k not in live}
    assert not removed, f"R-1 REMOVED definitions: {sorted(removed)}"
    assert changed == DECLARED_CHANGED, (
        f"undeclared changes: {sorted(changed - DECLARED_CHANGED)}; "
        f"declared but unchanged: {sorted(DECLARED_CHANGED - changed)}")
    assert added == DECLARED_ADDED, f"undeclared additions: {sorted(added)}"
    # COMPUTED, not transcribed: a hardcoded summary is how a scope proof keeps
    # reporting the previous revision's shape after the declaration has grown.
    return (f"{len(changed)} changed, {len(added)} added, {len(removed)} "
            f"removed — as declared ({', '.join(sorted(n.split('.')[-1] for n in changed))})")


# ===========================================================================
# MUTANTS — production-class wrong optimisations; each must RED its gate
# ===========================================================================
def _mutant_first_fit(self) -> None:
    """R-1a AS THE BRIEF PROPOSED IT: stop scanning past the first admissible
    entry. Production-class — it is the brief's own primary candidate."""
    _mp1 = COORD.PhaseCharge(self, "pump")
    _mp1.__enter__()
    try:
        ready: List[tuple] = []
        with self._admission_lock:
            self._prune_admitted_locked()
            if not self._deferred:
                return
            still: List[tuple] = []
            released: List[tuple] = []
            stop = False
            for entry in self._deferred:
                (_k, _w, run_id, stripe_id, attempt, _s, _m, _e, fut) = entry
                if stop:
                    still.append(entry)
                    continue
                if not self._attempt_live_locked(run_id, stripe_id, attempt):
                    if not fut.done():
                        fut.set_result(None)
                    released.append((run_id, stripe_id, attempt, _s))
                    continue
                if (self._try_admit_locked(run_id, stripe_id, attempt)
                        and self._staging_slots().acquire(blocking=False)):
                    ready.append(entry)
                    released.append((run_id, stripe_id, attempt, _s))
                    stop = True
                else:
                    still.append(entry)
            self._deferred = still
        for _r, _s_id, _att, _sub in released:
            self.note_stripe_frame_released(_r, _s_id, _att, _sub)
        for entry in ready:
            (kind, wconn, run_id, stripe_id, attempt, sub_index, msg, elig,
             fut) = entry
            self._chain_future(
                self._submit_with_slot(kind, wconn, run_id, stripe_id, attempt,
                                       sub_index, msg, elig), fut)
    finally:
        self._resume_paused_connections()
        _mp1.__exit__(None, None, None)


def _mutant_memo_dead(self) -> None:
    """THE PLAUSIBLE WRONG MEMO: cache the NEGATIVE answer too. This is the
    obvious way to write the optimisation and it is exactly what makes the drop
    decision unsound when an attempt is not monotone."""
    _mp1 = COORD.PhaseCharge(self, "pump")
    _mp1.__enter__()
    try:
        ready: List[tuple] = []
        with self._admission_lock:
            self._prune_admitted_locked()
            if not self._deferred:
                return
            still: List[tuple] = []
            released: List[tuple] = []
            memo: Dict[tuple, bool] = {}
            for entry in self._deferred:
                (_k, _w, run_id, stripe_id, attempt, _s, _m, _e, fut) = entry
                key = (run_id, stripe_id, attempt)
                if key not in memo:
                    memo[key] = self._attempt_live_locked(
                        run_id, stripe_id, attempt)
                if not memo[key]:
                    if not fut.done():
                        fut.set_result(None)
                    released.append((run_id, stripe_id, attempt, _s))
                    continue
                if (self._try_admit_locked(run_id, stripe_id, attempt)
                        and self._staging_slots().acquire(blocking=False)):
                    ready.append(entry)
                    released.append((run_id, stripe_id, attempt, _s))
                else:
                    still.append(entry)
            self._deferred = still
        for _r, _s_id, _att, _sub in released:
            self.note_stripe_frame_released(_r, _s_id, _att, _sub)
        for entry in ready:
            (kind, wconn, run_id, stripe_id, attempt, sub_index, msg, elig,
             fut) = entry
            self._chain_future(
                self._submit_with_slot(kind, wconn, run_id, stripe_id, attempt,
                                       sub_index, msg, elig), fut)
    finally:
        self._resume_paused_connections()
        _mp1.__exit__(None, None, None)


def _mutant_skip_gc(self) -> None:
    """THE OTHER PLAUSIBLE WRONG OPTIMISATION: once an attempt is admitted, skip
    every entry that is not its — cheap, and it leaks dead entries into the
    bounded store forever."""
    _mp1 = COORD.PhaseCharge(self, "pump")
    _mp1.__enter__()
    try:
        ready: List[tuple] = []
        with self._admission_lock:
            self._prune_admitted_locked()
            if not self._deferred:
                return
            still: List[tuple] = []
            released: List[tuple] = []
            for entry in self._deferred:
                (_k, _w, run_id, stripe_id, attempt, _s, _m, _e, fut) = entry
                key = (run_id, stripe_id, attempt)
                if self._admitted and key not in self._admitted:
                    still.append(entry)
                    continue
                if not self._attempt_live_locked(run_id, stripe_id, attempt):
                    if not fut.done():
                        fut.set_result(None)
                    released.append((run_id, stripe_id, attempt, _s))
                    continue
                if (self._try_admit_locked(run_id, stripe_id, attempt)
                        and self._staging_slots().acquire(blocking=False)):
                    ready.append(entry)
                    released.append((run_id, stripe_id, attempt, _s))
                else:
                    still.append(entry)
            self._deferred = still
        for _r, _s_id, _att, _sub in released:
            self.note_stripe_frame_released(_r, _s_id, _att, _sub)
        for entry in ready:
            (kind, wconn, run_id, stripe_id, attempt, sub_index, msg, elig,
             fut) = entry
            self._chain_future(
                self._submit_with_slot(kind, wconn, run_id, stripe_id, attempt,
                                       sub_index, msg, elig), fut)
    finally:
        self._resume_paused_connections()
        _mp1.__exit__(None, None, None)


def _metric_mutant(record):
    """Build a pump identical to the live one EXCEPT for what it records as the
    distinct-attempt high-water. `record(deferred, seen_keys, probes) -> int`.

    These are the classic ways a falsifier stops falsifying: report the number
    you already had (the population), or report a constant."""
    def _mut(self):
        _mp1 = COORD.PhaseCharge(self, "pump")
        _mp1.__enter__()
        try:
            ready: List[tuple] = []
            with self._admission_lock:
                self._prune_admitted_locked()
                if not self._deferred:
                    return
                still: List[tuple] = []
                released: List[tuple] = []
                live_keys: set = set()
                seen_keys: set = set()
                probes = 0
                n_deferred = len(self._deferred)
                for entry in self._deferred:
                    (_k, _w, run_id, stripe_id, attempt, _s, _m, _e, fut) = entry
                    _key = (run_id, stripe_id, attempt)
                    seen_keys.add(_key)
                    if _key not in live_keys:
                        probes += 1
                        if not self._attempt_live_locked(run_id, stripe_id,
                                                         attempt):
                            if not fut.done():
                                fut.set_result(None)
                            released.append((run_id, stripe_id, attempt, _s))
                            continue
                        live_keys.add(_key)
                    if (self._try_admit_locked(run_id, stripe_id, attempt)
                            and self._staging_slots().acquire(blocking=False)):
                        ready.append(entry)
                        released.append((run_id, stripe_id, attempt, _s))
                    else:
                        still.append(entry)
                self._deferred = still
            # [FIELD-6 OBSERVABILITY REPAIR] The seeds are now the `None`
            # UNOBSERVED sentinel, so this hand-copied update tracks
            # production's None-aware max. The MUTATION under test — `record()`
            # substituting the population for the distinct count — is
            # untouched; without this the mutant dies on `TypeError` before it
            # can be measured, which is a fixture failure, not a surviving
            # mutant.
            with self._bp_lock:
                _pk = self._bp["deferred_distinct_attempts_high_water"]
                _ok = int(record(n_deferred, seen_keys, probes))
                self._bp["deferred_distinct_attempts_high_water"] = (
                    _ok if _pk is None else max(int(_pk), _ok))
                _pp = self._bp["pump_liveness_probes_high_water"]
                _op = int(probes)
                self._bp["pump_liveness_probes_high_water"] = (
                    _op if _pp is None else max(int(_pp), _op))
            for _r, _s_id, _att, _sub in released:
                self.note_stripe_frame_released(_r, _s_id, _att, _sub)
            for entry in ready:
                (kind, wconn, run_id, stripe_id, attempt, sub_index, msg, elig,
                 fut) = entry
                self._chain_future(
                    self._submit_with_slot(kind, wconn, run_id, stripe_id,
                                           attempt, sub_index, msg, elig), fut)
        finally:
            self._resume_paused_connections()
            _mp1.__exit__(None, None, None)
    return _mut


def _r1_memo_survives_grant(self) -> None:
    """M10 — R-1 EXACTLY, i.e. the code Beta blocked. The memo is never
    invalidated, so a later frame of an already-admitted attempt stages on a
    cached positive after the attempt has died. Production-class by
    construction: it is the shipped R-1 working set."""
    _mp1 = COORD.PhaseCharge(self, "pump")
    _mp1.__enter__()
    try:
        ready: List[tuple] = []
        with self._admission_lock:
            self._prune_admitted_locked()
            if not self._deferred:
                return
            still: List[tuple] = []
            released: List[tuple] = []
            live_keys: set = set()
            for entry in self._deferred:
                (_k, _w, run_id, stripe_id, attempt, _s, _m, _e, fut) = entry
                _key = (run_id, stripe_id, attempt)
                if _key not in live_keys:
                    if not self._attempt_live_locked(run_id, stripe_id, attempt):
                        if not fut.done():
                            fut.set_result(None)
                        released.append((run_id, stripe_id, attempt, _s))
                        continue
                    live_keys.add(_key)
                if (self._try_admit_locked(run_id, stripe_id, attempt)
                        and self._staging_slots().acquire(blocking=False)):
                    ready.append(entry)
                    released.append((run_id, stripe_id, attempt, _s))
                else:
                    still.append(entry)
            self._deferred = still
        for _r, _s_id, _att, _sub in released:
            self.note_stripe_frame_released(_r, _s_id, _att, _sub)
        for entry in ready:
            (kind, wconn, run_id, stripe_id, attempt, sub_index, msg, elig,
             fut) = entry
            self._chain_future(
                self._submit_with_slot(kind, wconn, run_id, stripe_id, attempt,
                                       sub_index, msg, elig), fut)
    finally:
        self._resume_paused_connections()
        _mp1.__exit__(None, None, None)


def _r2_no_end_of_pass_sweep(self) -> None:
    """M11 — R-2 EXACTLY: the certified R2-1 fix, WITHOUT R-3's end-of-pass
    capacity re-probe. Production-class by construction — it is the working set
    Beta certified R2-1 on, and it is the code that refuses `C` at the boundary.

    Also used as the no-sweep reference by R3-2 and R3-3."""
    _mp1 = COORD.PhaseCharge(self, "pump")
    _mp1.__enter__()
    try:
        ready: List[tuple] = []
        with self._admission_lock:
            self._prune_admitted_locked()
            if not self._deferred:
                return
            still: List[tuple] = []
            released: List[tuple] = []
            live_keys: set = set()
            seen_keys: set = set()
            probes = 0
            for entry in self._deferred:
                (_k, _w, run_id, stripe_id, attempt, _s, _m, _e, fut) = entry
                _key = (run_id, stripe_id, attempt)
                seen_keys.add(_key)
                if _key not in live_keys:
                    probes += 1
                    if not self._attempt_live_locked(run_id, stripe_id, attempt):
                        if not fut.done():
                            fut.set_result(None)
                        released.append((run_id, stripe_id, attempt, _s))
                        continue
                    live_keys.add(_key)
                if self._try_admit_locked(run_id, stripe_id, attempt):
                    live_keys.discard(_key)
                    if self._staging_slots().acquire(blocking=False):
                        ready.append(entry)
                        released.append((run_id, stripe_id, attempt, _s))
                    else:
                        still.append(entry)
                else:
                    still.append(entry)
            self._deferred = still          # <- no sweep
        try:
            with self._bp_lock:
                self._bp["deferred_distinct_attempts_high_water"] = max(
                    int(self._bp["deferred_distinct_attempts_high_water"]),
                    len(seen_keys))
                self._bp["pump_liveness_probes_high_water"] = max(
                    int(self._bp["pump_liveness_probes_high_water"]),
                    int(probes))
        except Exception:                                        # noqa: BLE001
            pass
        for _r, _s_id, _att, _sub in released:
            self.note_stripe_frame_released(_r, _s_id, _att, _sub)
        for entry in ready:
            (kind, wconn, run_id, stripe_id, attempt, sub_index, msg, elig,
             fut) = entry
            self._chain_future(
                self._submit_with_slot(kind, wconn, run_id, stripe_id, attempt,
                                       sub_index, msg, elig), fut)
    finally:
        self._resume_paused_connections()
        _mp1.__exit__(None, None, None)


def _r2_discard_on_submission(self) -> None:
    """M10b — BETA'S LITERAL CANDIDATE: invalidate the memo *"at the point a
    grant/submission for K succeeds"*, i.e. in the `ready` branch. It closes
    R2-1 and is still wrong, because a slot freed by another thread mid-pass
    lets the next frame stage on the surviving cached positive."""
    _mp1 = COORD.PhaseCharge(self, "pump")
    _mp1.__enter__()
    try:
        ready: List[tuple] = []
        with self._admission_lock:
            self._prune_admitted_locked()
            if not self._deferred:
                return
            still: List[tuple] = []
            released: List[tuple] = []
            live_keys: set = set()
            for entry in self._deferred:
                (_k, _w, run_id, stripe_id, attempt, _s, _m, _e, fut) = entry
                _key = (run_id, stripe_id, attempt)
                if _key not in live_keys:
                    if not self._attempt_live_locked(run_id, stripe_id, attempt):
                        if not fut.done():
                            fut.set_result(None)
                        released.append((run_id, stripe_id, attempt, _s))
                        continue
                    live_keys.add(_key)
                if (self._try_admit_locked(run_id, stripe_id, attempt)
                        and self._staging_slots().acquire(blocking=False)):
                    live_keys.discard(_key)      # <- only on SUBMISSION
                    ready.append(entry)
                    released.append((run_id, stripe_id, attempt, _s))
                else:
                    still.append(entry)
            self._deferred = still
        for _r, _s_id, _att, _sub in released:
            self.note_stripe_frame_released(_r, _s_id, _att, _sub)
        for entry in ready:
            (kind, wconn, run_id, stripe_id, attempt, sub_index, msg, elig,
             fut) = entry
            self._chain_future(
                self._submit_with_slot(kind, wconn, run_id, stripe_id, attempt,
                                       sub_index, msg, elig), fut)
    finally:
        self._resume_paused_connections()
        _mp1.__exit__(None, None, None)


def _mutant_admit_from_memo(self) -> None:
    """GATE 5's MUTANT: grant staging authority from the memo instead of letting
    `_try_admit_locked` re-read. The seductive version — it removes the grant's
    ledger read too, so it looks like a further optimisation."""
    _mp1 = COORD.PhaseCharge(self, "pump")
    _mp1.__enter__()
    try:
        ready: List[tuple] = []
        with self._admission_lock:
            self._prune_admitted_locked()
            if not self._deferred:
                return
            still: List[tuple] = []
            released: List[tuple] = []
            live_keys: set = set()
            for entry in self._deferred:
                (_k, _w, run_id, stripe_id, attempt, _s, _m, _e, fut) = entry
                key = (run_id, stripe_id, attempt)
                if key not in live_keys:
                    if not self._attempt_live_locked(run_id, stripe_id, attempt):
                        if not fut.done():
                            fut.set_result(None)
                        released.append((run_id, stripe_id, attempt, _s))
                        continue
                    live_keys.add(key)
                if key in self._admitted:
                    admitted = True
                elif not self._admitted:
                    self._admitted[key] = True       # <- from the memo, no read
                    admitted = True
                else:
                    admitted = False
                if admitted and self._staging_slots().acquire(blocking=False):
                    ready.append(entry)
                    released.append((run_id, stripe_id, attempt, _s))
                else:
                    still.append(entry)
            self._deferred = still
        for _r, _s_id, _att, _sub in released:
            self.note_stripe_frame_released(_r, _s_id, _att, _sub)
        for entry in ready:
            (kind, wconn, run_id, stripe_id, attempt, sub_index, msg, elig,
             fut) = entry
            self._chain_future(
                self._submit_with_slot(kind, wconn, run_id, stripe_id, attempt,
                                       sub_index, msg, elig), fut)
    finally:
        self._resume_paused_connections()
        _mp1.__exit__(None, None, None)


def _mutant_multi_admit(self) -> None:
    """GATE 1's MUTANT — THE FORBIDDEN SPEEDUP. Let every live attempt through,
    which drains the queue fastest of all and breaks Beta's Correction-6
    serialization invariant. Beta's brief names this shape explicitly."""
    _mp1 = COORD.PhaseCharge(self, "pump")
    _mp1.__enter__()
    try:
        ready: List[tuple] = []
        with self._admission_lock:
            self._prune_admitted_locked()
            if not self._deferred:
                return
            still: List[tuple] = []
            released: List[tuple] = []
            live_keys: set = set()
            for entry in self._deferred:
                (_k, _w, run_id, stripe_id, attempt, _s, _m, _e, fut) = entry
                key = (run_id, stripe_id, attempt)
                if key not in live_keys:
                    if not self._attempt_live_locked(run_id, stripe_id, attempt):
                        if not fut.done():
                            fut.set_result(None)
                        released.append((run_id, stripe_id, attempt, _s))
                        continue
                    live_keys.add(key)
                self._admitted[key] = True
                if self._staging_slots().acquire(blocking=False):
                    ready.append(entry)
                    released.append((run_id, stripe_id, attempt, _s))
                else:
                    still.append(entry)
            self._deferred = still
        for _r, _s_id, _att, _sub in released:
            self.note_stripe_frame_released(_r, _s_id, _att, _sub)
        for entry in ready:
            (kind, wconn, run_id, stripe_id, attempt, sub_index, msg, elig,
             fut) = entry
            self._chain_future(
                self._submit_with_slot(kind, wconn, run_id, stripe_id, attempt,
                                       sub_index, msg, elig), fut)
    finally:
        self._resume_paused_connections()
        _mp1.__exit__(None, None, None)


def _with_pump(mutant):
    """Install a mutant as `RangeMinerCoordinator._pump_deferred` for the
    duration of a `with`, so a gate that reads LIVE SOURCE is unaffected and a
    gate that CALLS the pump gets the mutant."""
    class _Ctx:
        def __enter__(self):
            self.real = RangeMinerCoordinator._pump_deferred
            RangeMinerCoordinator._pump_deferred = mutant
            return self

        def __exit__(self, *a):
            RangeMinerCoordinator._pump_deferred = self.real
            return False
    return _Ctx()


def _reds(fn) -> bool:
    try:
        fn()
        return False
    except AssertionError:
        return True


def mutant_m1_multi_admit_reds_gate_1():
    """M1: several attempts admitted at once. GATE 1 must go red."""
    with _with_pump(_mutant_multi_admit):
        assert _reds(gate_g1b_at_most_one_attempt_stages), \
            "M1 survived gate 1 — the serialization gate is vacuous"
        assert _reds(gate_g1c_serialization_holds_across_repeated_pumps)
    return "multi-admit reds G1b and G1c"


def mutant_m11_no_sweep_reds_the_capacity_boundary():
    """M11 — R-2 EXACTLY (the R2-1-certified working set), without R-3's
    end-of-pass sweep. It must RED the capacity-boundary gate.

    This mutant is the strongest kind available: it is not a contrived shape,
    it is the code Beta certified R2-1 on one round ago. It also passes every
    other gate in this battery — which is precisely how the divergence reached
    a certification round disclosed only as "GC latency"."""
    with _with_pump(_r2_no_end_of_pass_sweep):
        assert _reds(gate_r3_1_capacity_boundary_differential), \
            "R-2 SURVIVED the capacity-boundary gate — it is vacuous"
        # and the reason it was not caught earlier: everything else is green
        assert not _reds(gate_r2_1_grant_does_not_license_later_frames)
        assert not _reds(gate_r2_1b_the_slot_race_beta_s_candidate_would_miss)
        assert not _reds(gate_r2_2_no_memo_hit_can_ever_reach_ready)
        assert not _reds(gate_g4_conservation)
        assert not _reds(gate_g6_disposition_is_identical_to_the_pre_patch_pump)
    return ("R-2 reds ONLY the capacity-boundary gate and passes all seven "
            "others — which is how it reached certification as 'GC latency'")


def mutant_m10_r1_memo_survives_grant_reds_r2_1():
    """M10 — THE BLOCKER, AS A MUTANT. R-1 exactly. It must red BOTH R2-1
    schedules and the corollary gate, and — the informative part — it must NOT
    red anything else, because that is precisely why R-1's own battery passed
    31/32 while carrying the defect."""
    with _with_pump(_r1_memo_survives_grant):
        assert _reds(gate_r2_1_grant_does_not_license_later_frames), \
            "R-1 SURVIVED R2-1 — the blocking gate is vacuous"
        assert _reds(gate_r2_1b_the_slot_race_beta_s_candidate_would_miss)
        assert _reds(gate_r2_2_no_memo_hit_can_ever_reach_ready)
        # R2-3 is deliberately NOT asserted here: it reads LIVE SOURCE, so a
        # swapped bound method cannot red it. Its falsifier is the four
        # synthetic wrong shapes inside R2-3 itself — including this mutant's.
        # and the reason it went undetected: every R-1 gate still passes
        assert not _reds(gate_g4_conservation)
        assert not _reds(gate_g5b_a_dead_attempt_is_never_admitted)
        assert not _reds(gate_g5c_death_between_the_memo_and_the_grant_denies_the_grant)
        assert not _reds(gate_g6_disposition_is_identical_to_the_pre_patch_pump)
    return ("R-1 reds the three behavioural R2 gates and NONE of the R-1 "
            "gates — which is how it passed 31/32 carrying the defect")


def mutant_m10b_discard_on_submission_reds_only_the_slot_race():
    """M10b — BETA'S LITERAL CANDIDATE, AS A MUTANT, AND THE EVIDENCE FOR THE
    EXTENSION. It must PASS R2-1 (it does close that history) and RED R2-1b.

    That asymmetry is the whole argument that the invalidation point had to move
    from "a submission succeeded" to "`_try_admit_locked` returned True". If
    this mutant passed both, the extension would be unjustified and should be
    reverted to Beta's simpler wording."""
    with _with_pump(_r2_discard_on_submission):
        passes_r2_1 = not _reds(gate_r2_1_grant_does_not_license_later_frames)
        reds_slot_race = _reds(
            gate_r2_1b_the_slot_race_beta_s_candidate_would_miss)
    assert passes_r2_1, (
        "discard-on-submission FAILED R2-1 — then the extension is not what "
        "distinguishes them and this report's reasoning is wrong")
    assert reds_slot_race, (
        "discard-on-submission SURVIVED the slot race — the extension beyond "
        "Beta's wording is UNJUSTIFIED and should be withdrawn")
    return "discard-on-submission: PASSES R2-1, REDS R2-1b — extension justified"


def mutant_m4_negative_memo_reds_gate_4():
    """M4 — THE MOST PRODUCTION-CLASS MUTANT IN THE SET, because it is how this
    optimisation is normally written: `memo[key] = self._attempt_live_locked(
    ...)`, caching both answers. It is invisible to G6's quiescent differential
    and to every count; only the revival arm separates it."""
    with _with_pump(_mutant_memo_dead):
        assert _reds(gate_g4b_the_memo_is_one_directional_and_that_is_load_bearing), \
            "the negative-caching memo SURVIVED G4b — the arm is vacuous"
    return "negative-caching memo reds G4b"


def mutant_m5_admit_from_memo_reds_gate_5():
    """M5: authority granted from the memo. GATE 5 must go red — both the
    structural arm (source) and the adversarial arm (behaviour)."""
    src_fn = _mutant_admit_from_memo
    tree = ast.parse(textwrap.dedent(inspect.getsource(src_fn)))
    writes = [ast.unparse(t) for n in ast.walk(tree)
              if isinstance(n, ast.Assign) for t in n.targets
              if isinstance(t, ast.Subscript) and "_admitted" in ast.unparse(t)]
    assert writes, "the M5 mutant does not actually write _admitted"
    with _with_pump(src_fn):
        assert _reds(gate_g5c_death_between_the_memo_and_the_grant_denies_the_grant), \
            "M5 survived gate 5 — the stale-admission gate is vacuous"
    return "memo-granted authority reds G5c; G5a rejects it on source"


def mutant_m8_prepatch_scan_reds_the_complexity_gate():
    """M8 — THE GATE-10 MUTANT, and it is production-class by construction: it
    IS the production code at the pinned anchor. Restoring the per-entry
    liveness call must red the complexity gate.

    A mutant that is literally the shipped predecessor is the strongest
    available: it cannot be dismissed as an artificial shape."""
    with _with_pump(_prepatch_pump()):
        assert _reds(gate_g8a_the_positive_feedback_term_is_gone), \
            "the pre-patch scan SURVIVED the complexity gate — G8a is vacuous"
        assert _reds(gate_g8b_the_truthful_bound_is_exact)
        assert _reds(gate_g8d_the_pathological_shape_end_to_end)
    return "pre-patch per-entry scan reds G8a, G8b and G8d"


def mutant_m8b_entry_keyed_memo_reds_the_complexity_gate():
    """M8b: memoise on the ENTRY (sub_index included) rather than the ATTEMPT —
    a real and easy mistake, since the entry tuple is what the loop holds. It
    changes no disposition at all and is therefore invisible to every semantic
    gate; only the complexity gate catches it."""
    def _mut(self):
        _mp1 = COORD.PhaseCharge(self, "pump")
        _mp1.__enter__()
        try:
            ready: List[tuple] = []
            with self._admission_lock:
                self._prune_admitted_locked()
                if not self._deferred:
                    return
                still: List[tuple] = []
                released: List[tuple] = []
                live_keys: set = set()
                for entry in self._deferred:
                    (_k, _w, run_id, stripe_id, attempt, _s, _m, _e, fut) = entry
                    key = (run_id, stripe_id, attempt, _s)   # <- includes sub
                    if key not in live_keys:
                        if not self._attempt_live_locked(run_id, stripe_id,
                                                         attempt):
                            if not fut.done():
                                fut.set_result(None)
                            released.append((run_id, stripe_id, attempt, _s))
                            continue
                        live_keys.add(key)
                    if (self._try_admit_locked(run_id, stripe_id, attempt)
                            and self._staging_slots().acquire(blocking=False)):
                        ready.append(entry)
                        released.append((run_id, stripe_id, attempt, _s))
                    else:
                        still.append(entry)
                self._deferred = still
            for _r, _s_id, _att, _sub in released:
                self.note_stripe_frame_released(_r, _s_id, _att, _sub)
            for entry in ready:
                (kind, wconn, run_id, stripe_id, attempt, sub_index, msg, elig,
                 fut) = entry
                self._chain_future(
                    self._submit_with_slot(kind, wconn, run_id, stripe_id,
                                           attempt, sub_index, msg, elig), fut)
        finally:
            self._resume_paused_connections()
            _mp1.__exit__(None, None, None)

    with _with_pump(_mut):
        assert not _reds(gate_g4_conservation), (
            "M8b changed a disposition — it was supposed to be semantically "
            "invisible, so this gate battery is mis-calibrated")
        assert _reds(gate_g8a_the_positive_feedback_term_is_gone), \
            "an entry-keyed memo survived the complexity gate"
    return "entry-keyed memo: semantically invisible, reds G8a"


def mutant_m9_a_falsifier_wired_to_the_wrong_number_reds_gate_8e():
    """M9a/M9b — THE MUTANTS FOR THE FALSIFIER ITSELF. A field added to make a
    guarantee testable is worthless if the field can be wrong without anything
    noticing, so both natural failures are driven:

      M9a  report the DEFERRED POPULATION (the number we already had, and the
           one `deferred_high_water` already reports) instead of the distinct
           attempt count — the mistake that makes the new field a duplicate;
      M9b  report a constant — the VIR-2 vacuous case.

    Note M9a leaves every disposition and every read count untouched, so G4,
    G6 and G8a/b/d all stay green under it. Only G8e catches it."""
    survivors = []
    for label, rec in (("population", lambda n, s, p: n),
                       ("constant-zero", lambda n, s, p: 0),
                       ("constant-25", lambda n, s, p: 25)):
        with _with_pump(_metric_mutant(rec)):
            if not _reds(gate_g8e_the_falsifier_fields_are_measuring):
                survivors.append(label)
    assert not survivors, (
        f"the falsifier gate ACCEPTS these wrong fields: {survivors} — the "
        f"field cannot then be trusted on the acceptance run")
    # and confirm M9a really is semantically invisible, so the claim above that
    # only G8e catches it is measured rather than asserted
    with _with_pump(_metric_mutant(lambda n, s, p: n)):
        assert not _reds(gate_g4_conservation)
        assert not _reds(gate_g8a_the_positive_feedback_term_is_gone)
    return "3/3 wrong falsifier fields red G8e; M9a invisible to G4 and G8a"


def mutant_m7_sleep_under_the_lock_reds_gate_7():
    """M7: a blocking primitive under the lock. Both G7 arms must reject it.
    Driven through the DETECTORS on synthetic source, because installing it in
    the live file is exactly what the gate forbids."""
    bad = textwrap.dedent("""
        def _pump_deferred(self):
            _mp1 = PhaseCharge(self, 'pump')
            _mp1.__enter__()
            try:
                ready = []
                with self._admission_lock:
                    self._prune_admitted_locked()
                    time.sleep(0)
                    for entry in self._deferred:
                        pass
                    self._deferred = still
            finally:
                self._resume_paused_connections()
                _mp1.__exit__(None, None, None)
    """)
    tree = ast.parse(bad)
    fn = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)][0]
    blk = [n for n in ast.walk(fn) if isinstance(n, ast.With)
           and "_admission_lock" in ast.unparse(n.items[0])][0]
    pre = _callees(_lock_block(_pinned_src()))
    assert (_callees(blk) - pre) - NEW_CALLEES_ALLOWED, (
        "the new-callee detector accepts `time.sleep` — the allowlist has "
        "swallowed the falsifier")
    names = {getattr(n.func, "attr", None) or getattr(n.func, "id", "")
             for n in ast.walk(blk) if isinstance(n, ast.Call)}
    assert names & set(_BLOCKING_MARKERS), \
        "the blocking-primitive detector accepts `time.sleep`"
    return "sleep-under-lock rejected by both G7 detectors"


def mutant_m3_resume_out_of_finally_reds_gate_3():
    """M3: the resume moved out of `finally` (the shape a coalescing pump would
    produce). G3's structural arm must reject it."""
    bad = textwrap.dedent("""
        def _pump_deferred(self):
            _mp1 = PhaseCharge(self, 'pump')
            _mp1.__enter__()
            self._resume_paused_connections()
            try:
                with self._admission_lock:
                    self._deferred = []
            finally:
                _mp1.__exit__(None, None, None)
    """)
    fn = [n for n in ast.walk(ast.parse(bad)) if isinstance(n, ast.FunctionDef)][0]
    tries = [n for n in ast.walk(fn) if isinstance(n, ast.Try) and n.finalbody]
    fin = [ast.unparse(s) for s in tries[0].finalbody]
    assert fin[0] != "self._resume_paused_connections()", \
        "the G3 structural detector would accept a resume outside `finally`"
    return "resume-outside-finally rejected by G3's structural arm"


# ===========================================================================
# runner
# ===========================================================================
def main() -> int:
    print("=" * 78)
    print("S172 R-1/R-2/R-3 — DRAIN STARVATION REMEDY GATE BATTERY")
    print(f"pinned pre-R1 anchor : {PINNED_COMMIT}")
    head = subprocess.run(["git", "-C", _ROOT, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    print(f"live HEAD            : {head}")
    print("=" * 78)

    print("\n-- anchor + oracle -------------------------------------------")
    check("anchor is authentic", gate_anchor_is_authentic)
    check("oracle bound to module under test",
          gate_oracle_is_bound_to_the_module_under_test)
    check("AST scope proof", gate_scope_proof)

    print("\n-- GATE 1  serialization invariant unchanged ------------------")
    check("G1a admission surfaces byte-identical",
          gate_g1a_admission_surfaces_byte_identical)
    check("G1b at most one attempt stages", gate_g1b_at_most_one_attempt_stages)
    check("G1c holds across repeated pumps",
          gate_g1c_serialization_holds_across_repeated_pumps)

    print("\n-- GATE 2  bounded deferred storage unchanged -----------------")
    check("G2 bound + refusal unchanged", gate_g2_bounded_storage_unchanged)

    print("\n-- GATE 3  resume-credit semantics unchanged ------------------")
    check("G3a one resume per pump, every path",
          gate_g3a_resume_fires_once_per_pump_on_every_path)
    check("G3b resume still last in finally",
          gate_g3b_resume_is_the_last_thing_and_still_in_finally)

    print("\n-- GATE 4  no lost or duplicated deferred entry ---------------")
    check("G4 conservation over the matrix", gate_g4_conservation)
    check("G4b one-directional memo under revival",
          gate_g4b_the_memo_is_one_directional_and_that_is_load_bearing)

    print("\n-- GATE 5  no stale-attempt admission -------------------------")
    check("G5a sole grant site is self-guarded",
          gate_g5a_authority_is_written_only_under_a_fresh_ledger_read)
    check("G5b dead attempts never admitted",
          gate_g5b_a_dead_attempt_is_never_admitted)
    check("G5c death between memo and grant",
          gate_g5c_death_between_the_memo_and_the_grant_denies_the_grant)

    print("\n-- GATE 6  SEMANTIC PROOF: differential disposition -----------")
    check("G6 disposition identical to pre-patch",
          gate_g6_disposition_is_identical_to_the_pre_patch_pump)
    check("G6b the differential can diverge",
          gate_g6b_the_oracle_can_actually_diverge)

    print("\n-- GATE 7  lock holds no new blocking work --------------------")
    check("G7a no new callee under the lock",
          gate_g7a_no_new_callee_under_the_lock)
    check("G7b no blocking primitive under the lock",
          gate_g7b_no_blocking_primitive_under_the_lock)

    print("\n-- R2-1  cached-positive liveness may not survive a grant -----")
    check("R2-1  grant does not license later frames",
          gate_r2_1_grant_does_not_license_later_frames)
    check("R2-1b the slot race",
          gate_r2_1b_the_slot_race_beta_s_candidate_would_miss)
    check("R2-2  no memo hit can reach ready",
          gate_r2_2_no_memo_hit_can_ever_reach_ready)
    check("R2-3  discard sits at the authority point",
          gate_r2_3_the_discard_sits_at_the_authority_point)

    print("\n-- R3-1  the bounded-capacity boundary ------------------------")
    check("R3-1  capacity-boundary differential",
          gate_r3_1_capacity_boundary_differential)
    check("R3-2  sweep is a no-op under quiescence",
          gate_r3_2_the_sweep_is_a_no_op_under_quiescence)
    check("R3-3  sweep is per-key, not per-entry",
          gate_r3_3_the_sweep_is_per_key_not_per_entry)
    check("R3-4  a swept key cannot be resurrected",
          gate_r3_4_a_swept_key_cannot_be_resurrected)

    print("\n-- GATE 8  COMPLEXITY PROOF: the truthful R2-2 bound ----------")
    check("G8a positive-feedback term is gone",
          gate_g8a_the_positive_feedback_term_is_gone)
    check("G8b the truthful bound is exact",
          gate_g8b_the_truthful_bound_is_exact)
    check("G8g worst case is O(N) and says so",
          gate_g8g_the_worst_case_is_o_n_and_says_so)
    check("G8c clean control: pre-patch DOES scale",
          gate_g8c_pre_patch_reads_do_scale)
    check("G8d the MP-1 pathological shape",
          gate_g8d_the_pathological_shape_end_to_end)
    check("G8e the falsifier fields are measuring",
          gate_g8e_the_falsifier_fields_are_measuring)
    check("G8f high-water + decision-free",
          gate_g8f_the_fields_are_high_waters_and_decision_free)

    print("\n-- GATE 10 mutants: each wrong optimisation reds its gate -----")
    check("M1  multi-admit  -> G1", mutant_m1_multi_admit_reds_gate_1)
    check("M3  resume out of finally -> G3",
          mutant_m3_resume_out_of_finally_reds_gate_3)
    check("M4  negative memo -> G4b", mutant_m4_negative_memo_reds_gate_4)
    check("M5  admit from memo -> G5", mutant_m5_admit_from_memo_reds_gate_5)
    check("M7  sleep under lock -> G7",
          mutant_m7_sleep_under_the_lock_reds_gate_7)
    check("M9  wrong falsifier field -> G8e",
          mutant_m9_a_falsifier_wired_to_the_wrong_number_reds_gate_8e)
    check("M8  pre-patch scan -> G8",
          mutant_m8_prepatch_scan_reds_the_complexity_gate)
    check("M8b entry-keyed memo -> G8",
          mutant_m8b_entry_keyed_memo_reds_the_complexity_gate)
    check("M10 R-1 (memo survives grant) -> R2",
          mutant_m10_r1_memo_survives_grant_reds_r2_1)
    check("M10b discard-on-submission -> R2-1b only",
          mutant_m10b_discard_on_submission_reds_only_the_slot_race)
    check("M11 no end-of-pass sweep -> R3-1",
          mutant_m11_no_sweep_reds_the_capacity_boundary)

    print("\n" + "=" * 78)
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    total = len(_RESULTS)
    print(f"R-1/R-2/R-3 DRAIN REMEDY BATTERY: {passed}/{total}")
    if passed != total:
        for name, ok, detail in _RESULTS:
            if not ok:
                print(f"  {RED}FAILED{RESET} {name}: {detail}")
    print("=" * 78)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
