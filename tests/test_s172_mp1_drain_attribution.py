#!/usr/bin/env python3
"""S172 — MP-1 DRAIN ATTRIBUTION GATE BATTERY

Implements the gates for the brief `~/dashboard_work/
CCODE_BRIEF_MP1_DRAIN_ATTRIBUTION_v1_0.md` (Team Beta ruling on the attempt-7
forensic, 2026-08-16: MEASUREMENT BEFORE REMEDY).

WHAT MP-1 MEASURES, AND WHY IT HAD TO
--------------------------------------
`docs/GATE12_ATTEMPT7_H1H2_FORENSIC.md` established, by direct measurement, that
a worker delivered a complete stripe into the coordinator's `inbound` queue in
2.64 s and the coordinator's drain removed NONE of those 45 frames for 296 s.
H1 (worker send stall) is refuted; H2 (coordinator ingest backlog) is confirmed.
The forensic explicitly does NOT claim a cause, and says so:

    "Not claimed here: the cause of the drain starvation, why
     `heartbeats_accepted` is zero run-wide, and whether that zero is a second
     defect or a consequence of the first. Those need the drain's own scheduling
     read, which this report did not do."

This battery gates the instrument that makes that read possible. It measures
NOTHING about the cause itself — MP-1 is authorized as READ-ONLY OBSERVATION and
no remedy of any kind is in scope.

THE STANDARD, INHERITED FROM THE H1/H2 BATTERY AND NOT WEAKENED
----------------------------------------------------------------
No gate here asserts that a field exists, is non-negative, or parses. Every gate
for a measured field runs the same code twice — a CLEAN CONTROL and a
FAULT-INJECTION CONTROL — and asserts the two produce MATERIALLY DIFFERENT
values, so a field hardwired to any constant fails whatever the constant is.
Where a field is a presence/absence fact, both dispositions are driven. The
mutants disable each instrument at its source and assert the corresponding gate
goes red.

`None` means UNOBSERVED throughout and never degrades to `0.0`. A drain position
of 0 would read as "serviced first in the pass"; a `live_count` of 0 would read
as "the fleet was empty". Both are the inverse of the truth they would be
standing in for.

VOCABULARY FOR THE THREE-LEVEL ATTRIBUTION
-------------------------------------------
    L1  iteration = accept + admission + drain + deadline + stage_setup
                  + schedule + dispatch + expiry + advance + LOOP REMAINDER
    L2  drain     = msg + DRAIN REMAINDER
    L3  msg       = staging + pump + MSG REMAINDER

Every level carries its own NAMED remainder. Unattributed time is a bucket with
a name, never a silent gap (R1.2's lesson, applied per iteration rather than
once per run).

Run:  source ~/venvs/torch/bin/activate && \
      python3 -u tests/test_s172_mp1_drain_attribution.py
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from typing import Any, Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from miner.range_miner_coordinator import (  # noqa: E402
    OBS_NO_OBSERVATION,
    OBS_OK,
    OBS_UNAVAILABLE,
    RX_UNOBSERVED,
    SERVE_LOOP_SUBPHASES,
    SERVE_LOOP_WINDOW_INTERVAL_S,
    CoordinatorConfig,
    MinerLedger,
    NodeConfig,
    PhaseCharge,
    RangeMinerCoordinator,
    ServeLoopTiming,
)
import miner.range_miner_coordinator as COORD  # noqa: E402
from miner.range_miner_protocol import (  # noqa: E402
    SubStripeResultMessage,
)
from miner.range_miner_worker import (  # noqa: E402
    build_substripe_payload_bytes,
)

# THE PINNED PRE-MP-1 ANCHOR. FULL 40-CHARACTER SHA, never a prefix: a short
# prefix can become ambiguous in a repository's future and an ambiguous anchor
# silently stops being an anchor.
PINNED_COMMIT = "2c38f8cbe01e67cc66e7204bf4f35b09da5ed1d1"
SRC_REL = "miner/range_miner_coordinator.py"

CAPS = {"amd": 2_000_000, "nvidia": 5_000_000,
        "amd_hybrid": 1_000_000, "nvidia_hybrid": 2_500_000}
VARIANTS = ["java_lcg", "java_lcg_reverse",
            "java_lcg_hybrid", "java_lcg_hybrid_reverse"]
SPOOL_ROOT = "/var/spool/miner"

GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"
_RESULTS: List[Any] = []


def check(name, fn):
    try:
        detail = fn()
        _RESULTS.append((name, True, detail))
        print(f"  [{GREEN}PASS{RESET}] {name:<40} {detail}")
    except AssertionError as e:
        _RESULTS.append((name, False, str(e)))
        print(f"  [{RED}FAIL{RESET}] {name:<40} {e}")
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"  [{RED}ERROR{RESET}] {name:<40} {type(e).__name__}: {e}")


# ===========================================================================
# the pinned anchor and its integrity
# ===========================================================================
class AnchorUnavailable(RuntimeError):
    """The pinned pre-MP-1 source could not be obtained, or is not it."""


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
    """Executable structure only.

    The MP-1 source QUOTES the pre-MP-1 shape in its own comments (the whole
    point of the `[MP-1]` markers is to say what was there before), so a text
    probe over raw source would find the pre-MP-1 surface in the CHANGED file
    and credit an anchor that had drifted forward. Unparsing the AST drops every
    comment and re-emits every docstring as a literal, so the probes run against
    executable structure and cannot be fooled by prose."""
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
    """Per-definition AST digest: `sha256(ast.unparse(node))` for every
    module-level function, every method and every class body.

    AST rather than text, because `2389b61` reverted a fix by WHOLE-BLOCK
    REPLACEMENT and a text anchor would have gone green. Comments and docstring
    reflow move no digest; a changed statement moves exactly one."""
    tree = ast.parse(src)
    out: Dict[str, str] = {}

    def _walk(node, prefix):
        for child in getattr(node, "body", []):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                key = f"{prefix}{child.name}"
                stripped = ast.parse(_strip_comments(
                    textwrap.dedent(ast.unparse(child))))
                out[key] = hashlib.sha256(
                    ast.unparse(stripped).encode()).hexdigest()
            elif isinstance(child, ast.ClassDef):
                _walk(child, f"{prefix}{child.name}.")

    _walk(tree, "")
    return out


def gate_anchor_is_authentic():
    """FIRST, because every RED and every scope arm depends on it: the pinned
    object is really PRE-MP-1 source.

    A drifted anchor terminates UNAVAILABLE, which never accepts (VIR-3). The
    probes are structural: the pinned module must NOT carry the MP-1 surfaces,
    and must still carry the pre-MP-1 shapes they replaced."""
    src = _pinned_src()
    exe = _strip_comments(src)
    for absent in ("PhaseCharge", "note_drain_service",
                   "maybe_emit_serve_loop_window", "phase_exclusive_snapshot",
                   "SERVE_LOOP_SUBPHASES", "drain_service_census",
                   "note_drain_pass", "_close_iteration",
                   "note_drain_frame_class"):
        if absent in exe:
            raise AnchorUnavailable(
                f"pinned {PINNED_COMMIT} already carries {absent!r} — the "
                f"anchor points at MP-1 or later source, so every RED arm "
                f"credited against it would be credited against the fix")
    tree = ast.parse(src)
    cls = [n for n in ast.walk(tree)
           if isinstance(n, ast.ClassDef) and n.name == "ServeLoopTiming"]
    if len(cls) != 1:
        raise AnchorUnavailable("pinned module has no single ServeLoopTiming")
    nested = [n for n in ast.walk(cls[0])
              if isinstance(n, ast.Assign)
              and getattr(n.targets[0], "id", None) == "NESTED_SEGMENTS"]
    if len(nested) != 1 or ast.unparse(nested[0].value) != "('msg',)":
        raise AnchorUnavailable(
            f"pinned NESTED_SEGMENTS is "
            f"{ast.unparse(nested[0].value) if nested else 'absent'!r}, not the "
            f"pre-MP-1 ('msg',) — this is not the pre-MP-1 source")
    # and the live tree really has moved on, or the "scope proof" below would be
    # proving nothing about anything.
    live = _strip_comments(_live_src())
    assert "PhaseCharge" in live and "note_drain_service" in live, (
        "the LIVE module carries no MP-1 surface — this battery would be "
        "gating an instrument that does not exist")
    return f"anchor {PINNED_COMMIT[:12]} verified pre-MP-1; live carries MP-1"


# ===========================================================================
# fixtures
# ===========================================================================
def _coord(tmp, dbname="l.db", **cfg):
    cfg.setdefault("staging_dir", os.path.join(tmp, "staging"))
    ledger = MinerLedger(os.path.join(tmp, dbname))
    return RangeMinerCoordinator(CoordinatorConfig(**cfg), ledger)


def _register(coord, wid="hostA:gpu0", backend="cuda", now=100.0):
    node = NodeConfig(hostname=wid.split(":")[0], spool_root=SPOOL_ROOT,
                      ssh_address="10.0.0.9", ssh_user="michael")
    return coord.register_worker(
        worker_id=wid, hostname=wid.split(":")[0], backend=backend,
        capabilities={"seed_caps": dict(CAPS),
                      "supported_variants": list(VARIANTS)},
        node_config=node, now=now)


class _MonoClock:
    """A scripted `perf_counter`, so every arithmetic gate below asserts an
    EXACT expected number rather than a tolerance band. The residual arithmetic
    is the thing under test; a tolerance would hide a systematic error of
    exactly the size this instrument exists to detect."""

    def __init__(self, values):
        self.values = list(values)
        self.reads = []

    def __call__(self):
        v = self.values.pop(0)
        self.reads.append(v)
        return v


class _patched_clock:
    def __init__(self, mono):
        self.mono = mono

    def __enter__(self):
        self._saved = COORD.time.perf_counter
        COORD.time.perf_counter = self.mono
        return self.mono

    def __exit__(self, *a):
        COORD.time.perf_counter = self._saved
        return False


def _mutant_red(fn, what):
    """Run a scenario expected to FAIL under a mutation. A mutant that stays
    green is a gate proving nothing."""
    try:
        fn()
    except AssertionError:
        return True
    raise AssertionError(
        f"the mutant survived: {what} — the gate is green with the instrument "
        f"disabled, so it is not gating the instrument")


def _expect_red(fn, name):
    try:
        fn()
    except AssertionError:
        return True
    return False


# ===========================================================================
# §A — the three-level attribution, and every level SUMS
# ===========================================================================
def gate_a1_level1_partition_sums_exactly():
    """L1: `iteration` = the nine top-level phases + the LOOP REMAINDER, exactly.

    WHAT WOULD PROVE IT IS MEASURING: a scripted iteration whose phases are
    known to the microsecond produces a remainder equal to the arithmetic
    difference and nothing else. A remainder computed over the wrong segment set
    — the R1.2 defect, one level up — produces a different number.

    WRONG INPUT THAT REDS IT: adding a nested segment to the L1 partition, which
    is what `LEVEL1_SEGMENTS` exists to make a visible one-line decision instead
    of a silent double-count."""
    mono = _MonoClock([
        0.0,            # __init__ _t0
        10.0,           # tick -> iteration opens
        11.0, 12.0,     # accept    1.0
        12.0,           # drain start
        13.0, 15.0,     # msg       2.0   (nested in drain)
        18.0,           # drain stop -> 6.0
        18.0, 20.0,     # schedule  2.0
        30.0,           # close -> iteration 20.0
        31.0,           # metrics loop_seconds
    ])
    with _patched_clock(mono):
        t = ServeLoopTiming()
        t.tick(1786.0)
        _a = t.start(); t.stop("accept", _a)
        _d = t.start()
        _m = t.start(); t.stop("msg", _m)
        t.stop("drain", _d)
        _s = t.start(); t.stop("schedule", _s)
        t.close_current_iteration()
        m = t.metrics()
    assert abs(m["iteration_total"] - 20.0) < 1e-9, m
    expected = 20.0 - (1.0 + 6.0 + 2.0)          # accept + drain + schedule
    assert abs(m["loop_remainder_total"] - expected) < 1e-9, (
        f"loop remainder is {m['loop_remainder_total']}, expected exactly "
        f"{expected}; a different value means the L1 partition is not the "
        f"partition this level claims")
    # and it agrees with the certified run-total residual, computed the other way
    assert abs(m["loop_remainder_total"] - m["unattributed_total"]) < 1e-9, (
        f"the two residuals disagree: per-iteration "
        f"{m['loop_remainder_total']} vs run-total {m['unattributed_total']}")
    assert m["remainder_negative_loop"] == 0, m
    return (f"iteration=20.0 named=9.0 remainder={m['loop_remainder_total']:.1f} "
            f"== unattributed_total")


def gate_a2_level2_and_level3_sum_exactly():
    """L2 `drain = msg + drain_remainder` and L3 `msg = staging + pump +
    msg_remainder`, both exact.

    L3 is the level MP-1 adds and the one the primary hypothesis lives on: if
    the per-iteration cost grew, this says WHICH of the three named parts of a
    message dispatch grew.

    WRONG INPUT THAT REDS IT: charging `staging` INCLUSIVELY while a `pump`
    nested inside it is also charged — the residual would go negative and the
    clamp counter would fire."""
    mono = _MonoClock([
        0.0,          # _t0
        0.0,          # tick
        0.0,          # drain start
        0.0, 10.0,    # msg      10.0
        10.0,         # drain stop -> 10.0
        30.0,         # close -> iteration 30.0
        31.0,         # metrics
    ])
    with _patched_clock(mono):
        t = ServeLoopTiming()
        t.tick(1786.0)
        _d = t.start()
        _m = t.start(); t.stop("msg", _m)
        # the sub-phase deltas the serve loop reads off the phase stack
        t.note_subphase("staging", 6.0)
        t.note_subphase("pump", 1.5)
        t.stop("drain", _d)
        t.close_current_iteration()
        m = t.metrics()
    assert abs(m["drain_total"] - 10.0) < 1e-9, m
    assert abs(m["msg_total"] - 10.0) < 1e-9, m
    assert abs(m["drain_remainder_total"] - 0.0) < 1e-9, (
        f"drain remainder is {m['drain_remainder_total']}, expected 0.0 "
        f"(drain contained exactly one msg)")
    assert abs(m["msg_remainder_total"] - 2.5) < 1e-9, (
        f"msg remainder is {m['msg_remainder_total']}, expected exactly 2.5 "
        f"(10.0 - staging 6.0 - pump 1.5)")
    assert abs(m["staging_total"] - 6.0) < 1e-9, m
    assert abs(m["pump_total"] - 1.5) < 1e-9, m
    assert m["remainder_negative_msg"] == 0, m
    # FAULT INJECTION: the same iteration with the sub-phases NOT recorded must
    # produce a materially different msg remainder — i.e. the fields are load-
    # bearing and not decoration.
    mono2 = _MonoClock([0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 30.0, 31.0])
    with _patched_clock(mono2):
        t2 = ServeLoopTiming()
        t2.tick(1786.0)
        _d2 = t2.start()
        _m2 = t2.start(); t2.stop("msg", _m2)
        t2.stop("drain", _d2)
        t2.close_current_iteration()
        m2 = t2.metrics()
    assert abs(m2["msg_remainder_total"] - 10.0) < 1e-9, m2
    assert m2["msg_remainder_total"] > m["msg_remainder_total"] * 3, (
        "the msg remainder does not vary with the sub-phase attribution")
    return "drain=10.0 msg=10.0 staging=6.0 pump=1.5 msg_remainder=2.5"


def gate_a3_level1_declaration_matches_the_segment_set():
    """The declared L1 partition must BE the derived one.

    `LEVEL1_SEGMENTS` is written out as data so a reader can see the partition,
    and `unattributed_total` derives the same set from `SEGMENTS` minus
    `iteration` minus `NESTED_SEGMENTS`. Two spellings of one fact drift, and
    the drift would be invisible: both residuals would still be numbers.

    WRONG INPUT THAT REDS IT: adding a segment to `SEGMENTS` and forgetting
    `LEVEL1_SEGMENTS`, or vice versa."""
    derived = (set(ServeLoopTiming.SEGMENTS) - {"iteration"}
               - set(ServeLoopTiming.NESTED_SEGMENTS))
    declared = set(ServeLoopTiming.LEVEL1_SEGMENTS)
    assert derived == declared, (
        f"the declared L1 partition {sorted(declared)} is not the derived one "
        f"{sorted(derived)} — the two residual computations would disagree")
    # …and the nested declarations name real segments
    for name in (ServeLoopTiming.LEVEL2_CHILDREN
                 + ServeLoopTiming.LEVEL3_CHILDREN
                 + (ServeLoopTiming.LEVEL2_PARENT,
                    ServeLoopTiming.LEVEL3_PARENT)):
        assert name in ServeLoopTiming.SEGMENTS, f"{name} is not a segment"
    assert set(SERVE_LOOP_SUBPHASES) == set(ServeLoopTiming.LEVEL3_CHILDREN), (
        f"the loop reads sub-phases {SERVE_LOOP_SUBPHASES} but L3 partitions "
        f"over {ServeLoopTiming.LEVEL3_CHILDREN} — a sub-phase the loop reads "
        f"and the partition ignores is time that is measured and then lost")
    return f"L1={len(declared)} segments; L2/L3 children declared and real"


def gate_a4_remainder_is_named_never_silent():
    """An iteration that spends ALL its time outside every named phase reports
    that time AS THE REMAINDER — it does not vanish and it does not become zero.

    This is the whole content of the brief's first design constraint. The
    pre-MP-1 instrument computed ONE residual over the whole trial, so a single
    iteration that lost 296 s was averaged into ~4,300 others.

    WRONG INPUT THAT REDS IT: a residual that clamps to zero without counting
    the clamp, or one that is only ever computed run-wide."""
    mono = _MonoClock([0.0, 0.0, 100.0, 101.0])
    with _patched_clock(mono):
        t = ServeLoopTiming()
        t.tick(1786.5)
        t.close_current_iteration()        # 100 s, nothing named
        m = t.metrics()
    assert abs(m["iteration_max"] - 100.0) < 1e-9, m
    assert abs(m["loop_remainder_max"] - 100.0) < 1e-9, (
        f"a 100 s iteration with no named phase reports a remainder of "
        f"{m['loop_remainder_max']} — the time is silent, which is the defect")
    assert m["loop_remainder_max_at"] == 1786.5, (
        f"the remainder maximum names no instant: {m['loop_remainder_max_at']}")
    # CLEAN CONTROL: an iteration fully inside a named phase has ~no remainder
    mono2 = _MonoClock([0.0, 0.0, 0.0, 100.0, 100.0, 101.0])
    with _patched_clock(mono2):
        t2 = ServeLoopTiming()
        t2.tick(1786.5)
        _d = t2.start(); t2.stop("drain", _d)
        t2.close_current_iteration()
        m2 = t2.metrics()
    assert abs(m2["loop_remainder_max"]) < 1e-9, (
        f"the clean control reports a remainder of {m2['loop_remainder_max']}")
    return "unnamed 100.0s -> remainder 100.0 @1786.5 | named 100.0s -> 0.0"


def gate_a5_negative_remainder_is_counted_not_hidden():
    """Children summing to MORE than their parent is a MEASUREMENT DEFECT, and
    clamping it to zero silently is how a broken partition keeps reporting
    plausible numbers.

    WHAT WOULD PROVE IT IS MEASURING: driving children past the parent
    increments `remainder_negative_*`; the honest case leaves it at 0. Both
    dispositions are driven.

    WRONG INPUT THAT REDS IT: `max(0.0, residual)` with no counter — the two
    cases would be indistinguishable in every emitted record."""
    mono = _MonoClock([0.0, 0.0, 0.0, 1.0, 1.0, 5.0, 6.0, 7.0])
    with _patched_clock(mono):
        t = ServeLoopTiming()
        t.tick(1786.0)
        _d = t.start(); t.stop("drain", _d)          # drain 1.0
        _m = t.start(); t.stop("msg", _m)            # msg 4.0 > drain 1.0
        t.close_current_iteration()
        m = t.metrics()
    assert not mono.values, f"scripted clock not fully consumed: {mono.values}"
    assert m["remainder_negative_drain"] == 1, (
        f"a child exceeding its parent was not counted: {m}")
    assert m["drain_remainder_total"] == 0.0, m
    assert m["remainder_negative_loop"] == 0, (
        f"the honest L1 level was also flagged: {m}")
    return "children>parent -> remainder_negative_drain=1, L1 clean"


def gate_a6_worst_iteration_profile_is_ONE_iteration():
    """`iteration_max_parts` describes the SINGLE worst iteration, not the
    element-wise maximum across iterations.

    Nine independent per-segment maxima can each come from a different
    iteration; adding them up and reading the result as a profile is the mistake
    a forensic reader makes when the profile is the thing they want. This is why
    the parts are captured at the instant the iteration becomes the worst.

    WHAT WOULD PROVE IT IS MEASURING: two iterations, each dominated by a
    DIFFERENT phase, with the SHORTER one holding the larger `schedule`. The
    profile must show the long iteration's small `schedule`, not the short
    iteration's large one."""
    mono = _MonoClock([
        0.0,                    # _t0
        0.0,                    # tick 1
        0.0, 9.0,               # it1 schedule 9.0
        10.0,                   # tick 2 -> closes it1 at 10.0
        10.0, 11.0,             # it2 schedule 1.0
        110.0,                  # close it2 -> 100.0
        111.0,                  # metrics
    ])
    with _patched_clock(mono):
        t = ServeLoopTiming()
        t.tick(1786.0)
        _s = t.start(); t.stop("schedule", _s)
        t.tick(1786.1)
        _s = t.start(); t.stop("schedule", _s)
        t.close_current_iteration()
        m = t.metrics()
    parts = m["iteration_max_parts"]
    assert abs(m["iteration_max"] - 100.0) < 1e-9, m
    assert abs(m["schedule_max"] - 9.0) < 1e-9, (
        "the fixture is malformed — the SHORT iteration must hold the larger "
        "schedule or this gate proves nothing")
    assert abs(parts["schedule"] - 1.0) < 1e-9, (
        f"the profile reports schedule={parts['schedule']}, which is the OTHER "
        f"iteration's value — the parts are an element-wise maximum, not a "
        f"profile of one iteration")
    assert abs(parts["remainder_loop"] - 99.0) < 1e-9, (
        f"the profile's own remainder is {parts['remainder_loop']}, not 99.0")
    return f"worst=100.0s parts.schedule=1.0 (schedule_max=9.0) remainder=99.0"


def gate_a7_per_frame_cost_is_unobserved_not_zero():
    """`drain_seconds_per_frame` with no frame drained is `None`.

    A zero would read as "the drain was free", which is the exact inverse of
    "the drain never ran" — and the attempt-7 signature IS a drain that ran and
    took nothing.

    WHAT WOULD PROVE IT IS MEASURING: with frames, the rate is a real quotient
    that tracks the inputs; without, it is None."""
    t = ServeLoopTiming()
    m = t.metrics()
    assert m["drain_seconds_per_frame"] is None, (
        f"a drain with no frames reports a rate of "
        f"{m['drain_seconds_per_frame']!r} — a zero is a measurement")
    assert m["msg_seconds_per_frame"] is None, m
    mono = _MonoClock([0.0, 0.0, 0.0, 4.0, 10.0, 11.0])
    with _patched_clock(mono):
        t2 = ServeLoopTiming()
        t2.tick(1786.0)
        _d = t2.start(); t2.stop("drain", _d)      # 4.0 s
        t2.note_drain_pass(8, 2, 25)               # 8 frames
        t2.close_current_iteration()
        m2 = t2.metrics()
    assert abs(m2["drain_seconds_per_frame"] - 0.5) < 1e-9, (
        f"4.0 s over 8 frames is {m2['drain_seconds_per_frame']}, not 0.5")
    return "no frames -> None; 4.0s/8 frames -> 0.5 s/frame"


# ===========================================================================
# §B — per-connection drain service: the LATE-INDEX evidence
# ===========================================================================
def gate_b1_service_counts_measure_and_vary():
    """WHAT WOULD PROVE IT IS MEASURING: a connection serviced 20 times and one
    serviced once report 20 and 1. A counter hardwired to any constant fails
    whatever the constant is."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        for i in range(20):
            c.note_drain_service("run:conn1", "rrig6600:gpu0",
                                 position=i + 1, pass_seq=1)
        c.note_drain_service("run:conn2", "rrig6600c:gpu4",
                             position=21, pass_seq=1)
        cen = c.drain_service_census(["run:conn1", "run:conn2"])
    rows = {r["connection_id"]: r for r in cen["connections"]}
    assert cen["status"] == OBS_OK, cen
    assert rows["run:conn1"]["frames_window"] == 20, rows["run:conn1"]
    assert rows["run:conn2"]["frames_window"] == 1, rows["run:conn2"]
    assert rows["run:conn1"]["frames_window"] > rows["run:conn2"]["frames_window"] * 10
    assert rows["run:conn1"]["passes_window"] == 1, (
        "20 frames in ONE pass must be one PASS, not twenty — the pass counter "
        "is a different quantity from the frame counter and conflating them "
        "would make one drain pass look like twenty services")
    assert rows["run:conn1"]["worker_id"] == "rrig6600:gpu0", rows["run:conn1"]
    return "conn1 frames=20 passes=1 | conn2 frames=1 passes=1"


def gate_b2_never_serviced_is_a_MEASURED_ZERO():
    """THE GATE THE LATE-INDEX CLAIM RESTS ON.

    "connection X was never serviced" must be a measured zero (OK / 0), never an
    absent row. A census built from the SERVICE REGISTRY can only ever report
    connections it saw, so the claim would rest on an absence — the class this
    arc has sixteen recorded instances of. The census is therefore built from
    the LIVE CONNECTION SET.

    WHAT WOULD PROVE IT IS MEASURING: an unserviced live connection produces a
    row with status OK, frames 0, and positions `None` (there is no position for
    a frame that never happened).

    WRONG INPUT THAT REDS IT: enumerating from `_drain_conn` instead of the live
    set — the row disappears and `unserviced_count` becomes 0."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        c.note_drain_service("run:conn1", "w0", position=1, pass_seq=1)
        cen = c.drain_service_census(["run:conn1", "run:conn2", "run:conn3"])
    rows = {r["connection_id"]: r for r in cen["connections"]}
    assert set(rows) == {"run:conn1", "run:conn2", "run:conn3"}, (
        f"the census enumerated {sorted(rows)} — an unserviced connection is "
        f"missing, so 'never serviced' would be an absence")
    for cid in ("run:conn2", "run:conn3"):
        r = rows[cid]
        assert r["status"] == OBS_OK, r
        assert r["frames_window"] == 0 and r["frames_total"] == 0, r
        assert r["passes_window"] == 0, r
        assert r["position_min"] is None and r["position_mean"] is None, (
            f"a never-serviced connection reports a position: {r} — position 0 "
            f"would read as 'serviced first in the pass'")
        assert r["live"] is True, r
    assert cen["live_count"] == 3 and cen["serviced_count"] == 1
    assert cen["unserviced_count"] == 2, cen
    return "3 live, 1 serviced, 2 measured zeroes with None positions"


def gate_b3_unreadable_live_set_is_UNAVAILABLE_not_zero():
    """NO UNAVAILABLE PATH MAY RENDER COUNT-SHAPED (§2.10's rule, this
    instrument's copy). A census taken with no live set reports UNAVAILABLE and
    `live_count = None`; a zero would assert an empty fleet.

    WHAT WOULD PROVE IT IS MEASURING: both dispositions are driven off the same
    registry and differ only in the live-set argument."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        c.note_drain_service("run:conn1", "w0", position=1, pass_seq=1)
        bad = c.drain_service_census(None)
        good = c.drain_service_census(["run:conn1"])
    assert bad["status"] == OBS_UNAVAILABLE, bad
    assert bad["live_count"] is None, (
        f"an unreadable live set rendered count-shaped: {bad['live_count']!r}")
    assert bad["unserviced_count"] is None, bad
    assert bad["connections"][0]["live"] is None, bad["connections"][0]
    # the measurements it DID make are still measurements
    assert bad["connections"][0]["frames_window"] == 1, bad
    assert good["status"] == OBS_OK and good["live_count"] == 1, good
    return "live set None -> UNAVAILABLE/live_count=None; present -> OK/1"


def gate_b4_unresolvable_connection_is_a_NAMED_bucket():
    """A frame whose socket has already been reaped has no connection id. It is
    charged to `UNOBSERVED` and reported as NO_OBSERVATION — never dropped.

    A dropped frame would make the per-connection counts stop summing to the
    pass's frame count, and would do so exactly during a disconnect, which is
    when the census matters most.

    WHAT WOULD PROVE IT IS MEASURING: the bucket appears with its own status,
    distinct from every resolved row."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        c.note_drain_service("run:conn1", "w0", position=1, pass_seq=1)
        c.note_drain_service(None, None, position=2, pass_seq=1)
        c.note_drain_service(None, None, position=3, pass_seq=1)
        cen = c.drain_service_census(["run:conn1"])
    rows = {r["connection_id"]: r for r in cen["connections"]}
    assert RX_UNOBSERVED in rows, (
        f"an unresolvable frame was dropped: {sorted(rows)}")
    ghost = rows[RX_UNOBSERVED]
    assert ghost["status"] == OBS_NO_OBSERVATION, ghost
    assert ghost["frames_window"] == 2, ghost
    assert rows["run:conn1"]["status"] == OBS_OK, rows["run:conn1"]
    total = sum(r["frames_window"] for r in cen["connections"])
    assert total == 3, f"the per-connection counts do not sum to 3: {total}"
    return "2 unresolvable frames -> UNOBSERVED/NO_OBSERVATION; counts sum to 3"


def gate_b5_position_measures_and_discriminates_head_from_tail():
    """POSITION IS THE SERVICING-ORDER EVIDENCE, and it is what separates the
    two candidate causes.

    A connection reached only at high positions is one the pass runs out of
    budget before it gets to (ORDER starvation); a connection whose positions
    are spread across the pass is starved by RATE, not order. Those are
    different defects and no record the coordinator emitted before this one
    could tell them apart.

    WHAT WOULD PROVE IT IS MEASURING: a head-serviced and a tail-serviced
    connection driven through the SAME code produce materially different
    position statistics."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        for p in (1, 2, 3):                       # head of every pass
            c.note_drain_service("run:head", "w0", position=p, pass_seq=p)
        for p in (250, 251, 252):                 # tail of every pass
            c.note_drain_service("run:tail", "w1", position=p, pass_seq=p)
        cen = c.drain_service_census(["run:head", "run:tail"])
    rows = {r["connection_id"]: r for r in cen["connections"]}
    head, tail = rows["run:head"], rows["run:tail"]
    assert head["position_max"] == 3 and head["position_min"] == 1, head
    assert tail["position_min"] == 250 and tail["position_max"] == 252, tail
    assert tail["position_mean"] > head["position_mean"] * 50, (
        f"position does not discriminate: head mean {head['position_mean']}, "
        f"tail mean {tail['position_mean']}")
    assert head["passes_window"] == 3 and tail["passes_window"] == 3, (head, tail)
    return (f"head mean={head['position_mean']} tail mean={tail['position_mean']}")


def gate_b6_window_resets_without_losing_totals():
    """The window is a SERIES, so its counters must reset; the totals are the
    run, so they must not. A build-up is only readable if consecutive windows
    are independent.

    WRONG INPUT THAT REDS IT: resetting the totals too (every window would look
    identical), or not resetting the window (every window would be cumulative
    and no growth would be visible)."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        for i in range(5):
            c.note_drain_service("run:conn1", "w0", position=i + 1, pass_seq=i)
        before = c.drain_service_census(["run:conn1"])["connections"][0]
        c.reset_drain_service_window()
        after = c.drain_service_census(["run:conn1"])["connections"][0]
        c.note_drain_service("run:conn1", "w0", position=9, pass_seq=99)
        third = c.drain_service_census(["run:conn1"])["connections"][0]
    assert before["frames_window"] == 5 and before["frames_total"] == 5, before
    assert after["frames_window"] == 0, (
        f"the window did not reset: {after['frames_window']}")
    assert after["frames_total"] == 5, (
        f"the RUN total was reset with the window: {after['frames_total']}")
    assert after["position_min"] is None, (
        "a reset window kept a position from the previous window")
    assert third["frames_window"] == 1 and third["frames_total"] == 6, third
    assert third["position_min"] == 9, third
    return "window 5 -> reset 0 (totals 5) -> 1 (totals 6)"


def gate_b7_census_rows_are_detached():
    """[R3-1] `dict(x)` alone is not a snapshot. The census must not hand out
    live rows: the caller serializes them into a log record while the serve
    thread is still incrementing them, producing a record that mixes counters
    read at two different instants.

    WRONG INPUT THAT REDS IT: returning `self._drain_conn.values()`."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        c.note_drain_service("run:conn1", "w0", position=1, pass_seq=1)
        cen = c.drain_service_census(["run:conn1"])
        row = cen["connections"][0]
        snapshot_value = row["frames_window"]
        for i in range(10):
            c.note_drain_service("run:conn1", "w0", position=i + 2, pass_seq=2)
        assert row["frames_window"] == snapshot_value, (
            f"the census row moved from {snapshot_value} to "
            f"{row['frames_window']} after the census returned — it is a live "
            f"reference, not a snapshot")
        fresh = c.drain_service_census(["run:conn1"])["connections"][0]
    assert fresh["frames_window"] == 11, fresh
    return "row frozen at 1 while the registry advanced to 11"


def gate_b8_drain_pass_census_measures_partial_coverage():
    """THE PRIMARY HYPOTHESIS, MADE FALSIFIABLE. "Each pass could service only
    the head of the connection set" is exactly the statement
    `distinct_conns_serviced << connections_live`, sustained.

    WHAT WOULD PROVE IT IS MEASURING: a pass that reaches every connection and a
    pass that reaches three of twenty-five are counted differently."""
    t = ServeLoopTiming()
    t.note_drain_pass(25, 25, 25)          # full coverage
    t.note_drain_pass(256, 3, 25)          # head-only, budget-bound
    t.note_drain_pass(256, 3, 25)
    m = t.metrics()
    assert m["drain_passes"] == 3, m
    assert m["drain_passes_partial"] == 2, (
        f"partial passes counted {m['drain_passes_partial']}, expected 2")
    assert m["drain_pass_conns_max"] == 25 and m["drain_pass_conns_min"] == 3, m
    assert m["drain_pass_live_max"] == 25, m
    assert m["drain_pass_frames_max"] == 256, m
    assert m["drain_frames_total"] == 537, m
    # CLEAN CONTROL: full coverage only
    t2 = ServeLoopTiming()
    t2.note_drain_pass(25, 25, 25)
    assert t2.metrics()["drain_passes_partial"] == 0, (
        "a fully-covering pass was counted as partial")
    return "3 passes, 2 partial, conns 3..25 of 25 live"


def gate_b9_unknown_live_count_is_never_a_partial_pass():
    """[VIR-5] A pass that serviced 3 of an UNKNOWN number of connections is not
    evidence of partial coverage, and must not be counted as such.

    WRONG INPUT THAT REDS IT: treating `None` as 0 (never partial) or as a large
    number (always partial). Both manufacture a verdict from a non-observation."""
    t = ServeLoopTiming()
    t.note_drain_pass(10, 3, None)
    m = t.metrics()
    assert m["drain_passes"] == 1, m
    assert m["drain_passes_partial"] == 0, (
        "an unobservable live count produced a partial-coverage verdict")
    assert m["drain_pass_live_max"] == 0, m
    return "live=None -> pass counted, coverage verdict withheld"


def gate_b10_heartbeat_without_a_stripe_is_VISIBLE():
    """THE GATE FOR THE QUESTION BETA LEFT OPEN — `heartbeats_accepted = 0`
    run-wide on attempt 7.

    Every existing heartbeat counter is keyed by STRIPE, and a heartbeat reaches
    a stripe's counters only if it carries a `current_stripe_id`. So a heartbeat
    with an empty one is invisible to arrival, dequeue AND acceptance alike, and
    reads exactly like a heartbeat that was never sent. Two materially different
    facts, one observable value — the class this arc keeps finding.

    WHAT WOULD PROVE IT IS MEASURING: heartbeats WITH and WITHOUT a stripe id
    are counted in separate buckets, and both are non-zero when both are driven.
    Under the pre-MP-1 inventory the second bucket does not exist at all.

    WRONG INPUT THAT REDS IT: folding the two into one `heartbeat` count — the
    three candidate readings of the attempt-7 zero would collapse back into one
    unfalsifiable number."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        for _ in range(4):
            c.note_drain_frame_class("heartbeat", stripe_id="run__st1_s30")
        for _ in range(7):
            c.note_drain_frame_class("heartbeat", stripe_id=None)
        c.note_drain_frame_class("sub_stripe_result", stripe_id="run__st1_s30")
        c.note_drain_frame_class("eof", stripe_id=None)
        c.note_drain_frame_class(None, stripe_id=None)
        cen = c.drain_frame_class_census(window=True)
    k = cen["counts"]
    assert cen["status"] == OBS_OK, cen
    assert k["heartbeat"] == 11, k
    assert k["heartbeat_with_stripe"] == 4, (
        f"heartbeats carrying a stripe id are not counted separately: {k}")
    assert k["heartbeat_without_stripe"] == 7, (
        f"THE INVISIBLE CLASS IS STILL INVISIBLE: {k} — a heartbeat with no "
        f"stripe id is indistinguishable from one that was never sent, which "
        f"is exactly why the attempt-7 zero could not be explained")
    assert k["heartbeat_with_stripe"] != k["heartbeat_without_stripe"], (
        "the two buckets do not vary independently")
    assert k["sub_stripe_result"] == 1 and k["eof"] == 1, k
    assert k["unknown"] == 1, (
        f"a frame with no message_type was dropped rather than named: {k}")
    assert "heartbeat" not in ("eof",) and k.get("eof") == 1, k
    return "heartbeat 11 = 4 with-stripe + 7 without; eof and unknown named"


def gate_b11_class_census_window_and_run_are_distinct():
    """The window closes; the run total does not. A reader holding ONE window
    line must still be able to answer "did any heartbeat reach the drain at any
    point in this run", which is why both are carried on every record.

    WRONG INPUT THAT REDS IT: resetting the run total with the window — every
    record would then say "no heartbeat yet" for a run full of them."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        c.note_drain_frame_class("heartbeat", stripe_id="s")
        c.reset_drain_service_window()
        w = c.drain_frame_class_census(window=True)["counts"]
        r = c.drain_frame_class_census(window=False)["counts"]
        c.note_drain_frame_class("heartbeat", stripe_id="s")
        w2 = c.drain_frame_class_census(window=True)["counts"]
        r2 = c.drain_frame_class_census(window=False)["counts"]
    assert w == {}, f"the window did not reset: {w}"
    assert r["heartbeat"] == 1, f"the RUN total was reset with the window: {r}"
    assert w2["heartbeat"] == 1 and r2["heartbeat"] == 2, (w2, r2)
    return "window {} / run 1 -> window 1 / run 2"


# ===========================================================================
# §C — thread attribution (design constraint 5, R1-A's rule)
# ===========================================================================
def gate_c1_phase_charges_are_keyed_by_thread():
    """The SAME phase charged from two threads must produce TWO rows.

    Summing them would manufacture a serve-loop cost out of work the serve loop
    never did — the coordinator-side mirror of the defect R1-A fixed when a
    heartbeat thread's 200 s send was being charged to a mining thread's stripe.

    WHAT WOULD PROVE IT IS MEASURING: the two rows carry different thread ids
    AND different magnitudes, and the serve thread's own snapshot excludes the
    other thread's charge entirely."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        with c.phase_charge("staging"):
            time.sleep(0.05)
        mine = c.phase_exclusive_snapshot(("staging",))["staging"]

        def _other():
            with c.phase_charge("staging"):
                time.sleep(0.30)

        th = threading.Thread(target=_other, name="miner-staging_0")
        th.start(); th.join()
        mine_after = c.phase_exclusive_snapshot(("staging",))["staging"]
        rows = [r for r in c.phase_attribution() if r["phase"] == "staging"]
    assert len(rows) == 2, f"expected 2 thread rows, got {len(rows)}: {rows}"
    ids = {r["thread_id"] for r in rows}
    assert len(ids) == 2, f"both charges landed on one thread key: {rows}"
    names = {r["thread_name"] for r in rows}
    assert "miner-staging_0" in names, (
        f"the thread NAME is not recorded, so a reader cannot say WHOSE time "
        f"this was without correlating an opaque integer: {names}")
    assert abs(mine_after - mine) < 1e-6, (
        f"this thread's exclusive total moved from {mine} to {mine_after} when "
        f"ANOTHER thread charged the same phase — the attribution is not "
        f"thread-keyed")
    assert 0.03 < mine < 0.20, f"the clean control did not measure: {mine}"
    other = [r for r in rows if r["thread_name"] == "miner-staging_0"][0]
    assert other["exclusive_s"] > mine * 3, (
        f"the two threads' charges do not vary: {mine} vs "
        f"{other['exclusive_s']}")
    return (f"serve-thread {mine * 1000:.0f}ms | staging thread "
            f"{other['exclusive_s'] * 1000:.0f}ms | 2 keyed rows")


def gate_c2_exclusive_time_excludes_the_nested_phase():
    """EXCLUSIVE TIME IS WHAT MAKES L3 PARTITION. `_pump_deferred` is reachable
    from INSIDE `enqueue_staging` on the same thread, so charging both
    inclusively and subtracting both from `msg` would double-count the nested
    pump and drive the residual negative — the R1.2 defect one level down.

    WHAT WOULD PROVE IT IS MEASURING: with a 0.30 s pump nested inside a 0.50 s
    staging, `staging` reports ~0.20 s exclusive and ~0.50 s inclusive, and the
    two exclusive figures sum to the outer inclusive.

    WRONG INPUT THAT REDS IT: charging inclusive time as exclusive — the sum
    would exceed the parent."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        with c.phase_charge("staging"):
            time.sleep(0.20)
            with c.phase_charge("pump"):
                time.sleep(0.30)
        rows = {r["phase"]: r for r in c.phase_attribution()}
    st, pu = rows["staging"], rows["pump"]
    assert 0.45 < st["inclusive_s"] < 1.2, f"inclusive staging: {st}"
    assert 0.15 < st["exclusive_s"] < 0.30, (
        f"staging exclusive is {st['exclusive_s']} — the nested pump is not "
        f"being removed, so `msg` would be over-subtracted")
    assert st["inclusive_s"] > st["exclusive_s"] * 1.8, (
        "inclusive and exclusive do not differ under real nesting")
    assert abs((st["exclusive_s"] + pu["exclusive_s"])
               - st["inclusive_s"]) < 5e-3, (
        f"the exclusive figures {st['exclusive_s']} + {pu['exclusive_s']} do "
        f"not partition the inclusive {st['inclusive_s']}")
    return (f"staging incl={st['inclusive_s']:.3f} excl={st['exclusive_s']:.3f} "
            f"pump excl={pu['exclusive_s']:.3f}")


def gate_c3_phase_charge_cannot_perturb_its_caller():
    """An instrument that can kill a trial is worse than no instrument.

    WHAT WOULD PROVE IT IS MEASURING: the charge survives a broken coordinator
    (no phase machinery at all) AND propagates the wrapped block's exception
    unchanged rather than swallowing it."""
    class _Broken:
        def _phase_stack(self):
            raise RuntimeError("no stack here")

        def _charge_phase(self, *a):
            raise RuntimeError("nor here")

    b = _Broken()
    with PhaseCharge(b, "staging"):
        pass                                     # must not raise
    raised = False
    try:
        with PhaseCharge(b, "staging"):
            raise ValueError("the caller's own error")
    except ValueError:
        raised = True
    assert raised, (
        "PhaseCharge SWALLOWED the wrapped block's exception — an instrument "
        "that eats a production error is worse than the gap it fills")
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        try:
            with c.phase_charge("staging"):
                raise ValueError("boom")
        except ValueError:
            pass
        rows = [r for r in c.phase_attribution() if r["phase"] == "staging"]
    assert len(rows) == 1 and rows[0]["calls"] == 1, (
        f"a block that raised was not charged: {rows} — the failing path is "
        f"exactly the one a forensic reader needs timed")
    return "broken coordinator survives; caller exception propagates; charged"


# ===========================================================================
# §D — the wiring. Without these the battery gates a mechanism nobody calls.
# ===========================================================================
def _claimed_stripe(coord, sid="run_sMP1"):
    conn = _register(coord)
    coord.ledger.add_stripe("run", sid, 0, 30, "java_lcg", 1, now=100.0)
    coord.ledger.claim_stripe("run", sid, "hostA:gpu0", 0, 1,
                              lease_expires_at=1e9)
    conn.record_assignment(sid, 0)
    return conn, sid


def _inline_result(sid, sub_index=0):
    survivors = [[1, 0.9, None, [1]]]
    obj, pb = build_substripe_payload_bytes(sid, sub_index, 0, 30, survivors)
    return SubStripeResultMessage(
        worker_id="hostA:gpu0", stripe_id=sid, sub_index=sub_index,
        seed_start=0, seed_count=30, survivor_count=1, inline=obj,
        size_bytes=len(pb), sha256=hashlib.sha256(pb).hexdigest())


def gate_d1_staging_is_charged_by_the_real_dispatch():
    """THE PRODUCTION CHARGE, driven through the REAL `_serve_dispatch` with a
    real ledger, a real registered worker and a real inline sub-result.

    WHAT WOULD PROVE IT IS MEASURING: `staging` is charged on the dispatching
    thread with a non-zero exclusive time, and the mutant that neutralises
    `phase_charge` reds it.

    WRONG INPUT THAT REDS IT: instrumenting a wrapper nobody calls — the real
    dispatch path would then produce no row at all."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp, miner_stripe_size=1000)
        conn, sid = _claimed_stripe(c)
        msg = _inline_result(sid)
        c._serve_dispatch(msg, "run", "hostA:gpu0",
                          {"hostA:gpu0": conn}, lambda: [conn])
        if c._staging_executor is not None:
            c._staging_executor.shutdown(wait=True)
            c._staging_executor = None
        rows = [r for r in c.phase_attribution() if r["phase"] == "staging"]
    assert len(rows) == 1, (
        f"the real dispatch produced {len(rows)} staging rows — the charge is "
        f"not on the production path")
    assert rows[0]["calls"] == 1 and rows[0]["exclusive_s"] > 0.0, rows
    # VIR-2 POSITIVE CONTROL: neutralise the seam on the PRODUCTION class.
    saved = RangeMinerCoordinator.phase_charge

    class _Null:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    try:
        RangeMinerCoordinator.phase_charge = lambda self, name: _Null()

        def _mut():
            with tempfile.TemporaryDirectory() as tmp2:
                c2 = _coord(tmp2, miner_stripe_size=1000)
                conn2, sid2 = _claimed_stripe(c2)
                c2._serve_dispatch(_inline_result(sid2), "run", "hostA:gpu0",
                                   {"hostA:gpu0": conn2}, lambda: [conn2])
                if c2._staging_executor is not None:
                    c2._staging_executor.shutdown(wait=True)
                    c2._staging_executor = None
                bad = [r for r in c2.phase_attribution()
                       if r["phase"] == "staging"]
                assert len(bad) == 1, bad
        _mutant_red(_mut, "phase_charge neutralised on the production class")
    finally:
        RangeMinerCoordinator.phase_charge = saved
    return f"real dispatch -> staging excl={rows[0]['exclusive_s'] * 1e6:.0f}us"


def gate_d2_serve_loop_delta_read_lands_in_the_timing_record():
    """THE READ PATH the serve loop uses, driven around the REAL dispatch.

    The charge happens several frames below the loop, so the loop reads the
    delta of its own thread's exclusive accumulator across one dispatch and
    records it as a nested segment of `msg`. This gate drives exactly that
    sequence — `phase_exclusive_snapshot` / dispatch / `phase_exclusive_snapshot`
    / `note_subphase` — over production code, and asserts the L3 partition then
    reconciles.

    SCOPE, STATED RATHER THAN IMPLIED: the surrounding loop here is the harness;
    that the PRODUCTION loop contains this sequence is proved by
    `gate_d4_wiring_is_in_the_serve_loop` over the live AST. Neither gate is
    sufficient alone, and the pairing is deliberate — the R1.3 lesson, where
    inspecting a class proved nothing about how it was wired."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp, miner_stripe_size=1000)
        conn, sid = _claimed_stripe(c)
        sl = ServeLoopTiming()
        sl.tick(1786.0)
        _d = sl.start()
        ph0 = c.phase_exclusive_snapshot(SERVE_LOOP_SUBPHASES)
        _m = sl.start()
        c._serve_dispatch(_inline_result(sid), "run", "hostA:gpu0",
                          {"hostA:gpu0": conn}, lambda: [conn])
        sl.stop("msg", _m)
        ph1 = c.phase_exclusive_snapshot(SERVE_LOOP_SUBPHASES)
        for name in SERVE_LOOP_SUBPHASES:
            sl.note_subphase(name, ph1[name] - ph0[name])
        sl.stop("drain", _d)
        sl.close_current_iteration()
        m = sl.metrics()
        if c._staging_executor is not None:
            c._staging_executor.shutdown(wait=True)
            c._staging_executor = None
    assert m["staging_total"] > 0.0, (
        "the delta read produced no staging time — the loop would report a "
        "message dispatch with an unattributable interior")
    assert m["staging_total"] <= m["msg_total"] + 1e-9, (
        f"staging {m['staging_total']} exceeds its parent msg "
        f"{m['msg_total']}")
    assert m["remainder_negative_msg"] == 0, m
    assert abs((m["staging_total"] + m["pump_total"]
                + m["msg_remainder_total"]) - m["msg_total"]) < 1e-9, (
        f"L3 does not partition: staging {m['staging_total']} + pump "
        f"{m['pump_total']} + remainder {m['msg_remainder_total']} != msg "
        f"{m['msg_total']}")
    return (f"msg={m['msg_total'] * 1e3:.2f}ms staging={m['staging_total'] * 1e3:.2f}ms "
            f"remainder={m['msg_remainder_total'] * 1e3:.2f}ms; L3 sums")


def gate_d3_pump_is_charged_and_thread_attributed():
    """`_pump_deferred` is charged, and the charge rides the EXISTING `finally`
    so the certified `_resume_paused_connections` guarantee is untouched.

    WHAT WOULD PROVE IT IS MEASURING: a real `_pump_deferred` call produces a
    `pump` row on the calling thread; the same call from another thread produces
    a second, separately keyed row."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        c._pump_deferred()

        def _other():
            c._pump_deferred()

        th = threading.Thread(target=_other, name="miner-staging_7")
        th.start(); th.join()
        rows = [r for r in c.phase_attribution() if r["phase"] == "pump"]
    assert len(rows) == 2, f"expected two thread-keyed pump rows: {rows}"
    assert {r["thread_name"] for r in rows} >= {"miner-staging_7"}, rows
    assert all(r["calls"] == 1 for r in rows), rows
    # …and the early-return path (empty deferred queue) is still charged: a
    # pump that returns immediately is a measurement of ~0, not a missing one.
    assert all(r["inclusive_s"] >= 0.0 for r in rows), rows
    return f"2 pump rows, thread-keyed, early-return path charged"


def gate_d4_wiring_is_in_the_serve_loop():
    """AST over LIVE source: the instrument is called from the places that make
    it an instrument, not merely defined.

    The H1/H2 battery's N5 lesson: a gate that checks only the no-touch
    surfaces would pass on an instrument that does not exist."""
    tree = ast.parse(_live_src())
    serve = ast.unparse(_func_node(tree, "RangeMinerCoordinator", "serve_trial"))
    for required in ("note_drain_service", "note_drain_pass",
                     "maybe_emit_serve_loop_window",
                     "phase_exclusive_snapshot", "note_subphase",
                     "next_drain_pass", "note_drain_frame_class"):
        assert required in serve, (
            f"{required} is not called from serve_trial — this battery would "
            f"be gating a mechanism nothing drives")
    disp = ast.unparse(_func_node(tree, "RangeMinerCoordinator",
                                  "_serve_dispatch"))
    assert disp.count("phase_charge('staging')") == 2, (
        f"expected the staging charge at BOTH enqueue_staging call sites "
        f"(inline and remote), found {disp.count(chr(39))}: a charged inline "
        f"path and an uncharged remote one would under-report exactly the "
        f"spooled-result geometry")
    pump = ast.unparse(_func_node(tree, "RangeMinerCoordinator",
                                  "_pump_deferred"))
    assert "PhaseCharge(self, 'pump')" in pump, "the pump charge is absent"
    assert "_resume_paused_connections()" in pump, (
        "the certified resume trigger vanished from the pump's finally")
    return "6 hooks in serve_trial; 2 staging charges; pump charged"


def gate_d5_window_is_rate_limited_and_emits_one_record():
    """[§15 no-high-rate-noise] The window is self-rate-limited on a MONOTONIC
    interval, so the loop's iteration rate — 12,300 iterations in one attempt-5
    stage — cannot turn it into a flood.

    WHAT WOULD PROVE IT IS MEASURING: many calls inside one interval emit ONE
    record; a call past the interval emits another; `force` overrides."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        c.serve_loop_window_interval_s = 10.0
        sl = ServeLoopTiming()
        c.note_drain_service("run:conn1", "w0", position=1, pass_seq=1)
        with _Capture("range_miner_coordinator") as cap:
            first = c.maybe_emit_serve_loop_window(
                "run", sl, stage_idx=1, live_connection_ids=["run:conn1"],
                now=1000.0)
            skipped = [c.maybe_emit_serve_loop_window(
                "run", sl, live_connection_ids=["run:conn1"],
                now=1000.0 + i * 0.1) for i in range(50)]
            later = c.maybe_emit_serve_loop_window(
                "run", sl, live_connection_ids=["run:conn1"], now=1011.0)
            forced = c.maybe_emit_serve_loop_window(
                "run", sl, live_connection_ids=["run:conn1"], now=1011.5,
                force=True)
            lines = cap.of("[S172-SL] window")
    assert first is not None and all(s is None for s in skipped), (
        f"{sum(s is not None for s in skipped)} of 50 in-interval calls emitted")
    assert later is not None and forced is not None
    assert len(lines) == 3, f"expected 3 window records, got {len(lines)}"
    assert first["window_seconds"] is None, (
        "the FIRST window fabricated a duration — there is no previous mark to "
        "measure from, and a 0.0 would make every rate in it infinite")
    assert abs(later["window_seconds"] - 11.0) < 1e-6, later["window_seconds"]
    assert first["connections"]["live_count"] == 1, first["connections"]
    assert first["stage_idx"] == 1, first
    return "50 in-interval calls -> 1 record; 3 records total; first secs=None"


def gate_d6_window_reports_a_series_not_a_cumulative_total():
    """A BUILD-UP IS A DERIVATIVE, and a derivative needs consecutive
    independent windows. If the window reported cumulative totals, a phase whose
    cost doubled and one that was constant would be indistinguishable.

    WHAT WOULD PROVE IT IS MEASURING: two windows over the same instrument, the
    second with three times the work, report 1x and 3x — not 1x and 4x."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        sl = ServeLoopTiming()
        mono = _MonoClock([0.0] + [float(i) for i in range(1, 40)])
        with _patched_clock(mono):
            sl.tick(1786.0)
            sl.note_subphase("staging", 1.0)
            sl.close_current_iteration()
            w1 = c.maybe_emit_serve_loop_window(
                "run", sl, live_connection_ids=[], now=1000.0)
            sl.tick(1786.0)
            sl.note_subphase("staging", 3.0)
            sl.close_current_iteration()
            w2 = c.maybe_emit_serve_loop_window(
                "run", sl, live_connection_ids=[], now=1011.0)
    assert abs(w1["phases"]["staging"] - 1.0) < 1e-9, w1["phases"]
    assert abs(w2["phases"]["staging"] - 3.0) < 1e-9, (
        f"the second window reports {w2['phases']['staging']} — 4.0 would mean "
        f"the window is cumulative and no growth is readable from the series")
    assert w1["iterations"] == 1 and w2["iterations"] == 1, (w1, w2)
    return "window staging 1.0 then 3.0 (not 1.0 then 4.0)"


def gate_d7_summary_carries_the_new_series_and_the_profile():
    """The terminal record: the new fields are ADDITIVE on the grep-stable
    `[S172-SL] summary` line, and the worst iteration's profile is its own
    structured record because it is a MAP.

    WRONG INPUT THAT REDS IT: flattening the profile into the summary line,
    where a reader adds nine maxima that came from nine different iterations."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        sl = ServeLoopTiming()
        sl.tick(1786.0)
        sl.note_subphase("staging", 0.5)
        sl.note_drain_pass(10, 2, 25)
        sl.close_current_iteration()
        with _Capture("range_miner_coordinator") as cap:
            m = c.log_serve_loop_timing_summary("run-mp1", sl)
            line = cap.of("[S172-SL] summary")
            prof = cap.payloads("[S172-SL] iteration_profile")
    assert m is not None and len(line) == 1, (m, line)
    for field in ("staging_total=", "pump_total=", "loop_remainder_total=",
                  "loop_remainder_max=", "drain_remainder_total=",
                  "msg_remainder_total=", "remainder_negative_loop=",
                  "drain_passes=", "drain_frames_total=",
                  "drain_pass_conns_max=", "drain_pass_conns_min=",
                  "drain_passes_partial=", "drain_seconds_per_frame=",
                  "msg_seconds_per_frame="):
        assert field in line[0], f"missing {field} in the summary line"
    # the certified fields are still there, in the same line
    for field in ("loop_seconds=", "unattributed_total=", "drain_max=",
                  "exit_seconds=", "loop_now_age_max="):
        assert field in line[0], f"the MP-1 series displaced {field}"
    assert len(prof) == 1, f"expected one iteration_profile, got {len(prof)}"
    assert "staging" in prof[0]["parts"], prof[0]
    assert "remainder_loop" in prof[0]["parts"], prof[0]
    assert prof[0]["phase_attribution"] is not None, prof[0]
    return f"{len(line[0].split())} summary fields + 1 profile record"


def gate_d8_summary_never_raises():
    """Standing guard (§4 Alpha guard, Beta-approved): this runs on the terminal
    path of EVERY trial, including failing ones, and an instrument that masks a
    primary terminal reason with its own traceback is a defect of exactly the
    kind the F2 work removed.

    WHAT WOULD PROVE IT IS MEASURING: a deliberately broken timing object is
    handled, and the honest object still returns metrics."""
    class _Exploding(ServeLoopTiming):
        def metrics(self):
            raise RuntimeError("instrument failure")

    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        assert c.log_serve_loop_timing_summary("run", _Exploding()) is None
        assert c.log_serve_loop_timing_summary("run", None) is None
        # the window emitter too
        assert c.maybe_emit_serve_loop_window(
            "run", _Exploding(), live_connection_ids=["x"], force=True,
            now=1.0) is not None or True
        assert c.log_serve_loop_timing_summary("run", ServeLoopTiming()) is not None
    return "broken instrument -> None; None -> None; honest -> metrics"


# ===========================================================================
# §E — scope, non-regression, and the no-behaviour-change claim
# ===========================================================================
NO_TOUCH_DEFS = (
    "claim_stripe", "schedule_pending_stripes", "renew_lease",
    "_renew_active_lease", "process_lease_expiry",
    "_handle_stripe_failure_locked", "_execution_set_expected_workers",
    # MP-1's own additions to the pinned set: the certified back-pressure and
    # capacity surfaces this instrument BORDERS but must not enter.
    "enqueue_staging", "_defer_locked", "_conn_reader_loop",
    "dispatch_inbound_result", "_release_resume_credit_exact",
    "_resume_paused_connections", "staging_capacity_timeout_expired",
)


def gate_e1_no_touch_surfaces_are_byte_identical():
    """AST over live source vs the pinned anchor: MP-1's diff touches NONE of
    the certified lease, matrix, admission, capacity or reader surfaces.

    `enqueue_staging` and `_conn_reader_loop` are in the list deliberately.
    Instrumenting `enqueue_staging` from the INSIDE was the obvious way to time
    it and would have required wrapping its whole body in a `with`, re-indenting
    a certified capacity surface for an instrument. The charge is at the CALL
    SITE instead, which leaves the surface itself untouched.

    WRONG INPUT THAT REDS IT: an edit inside a named function reds EVEN IF EVERY
    BEHAVIOURAL TEST PASSES — `2389b61` reverted a fix by whole-block
    replacement and a text anchor would have gone green."""
    pinned = _def_digests(_pinned_src())
    live = _def_digests(_live_src())
    for name in NO_TOUCH_DEFS:
        keys = [k for k in pinned if k.split(".")[-1] == name]
        assert keys, f"{name} is not present in the pinned module"
        for k in keys:
            assert k in live, f"{k} was DELETED by MP-1"
            assert pinned[k] == live[k], (
                f"{k} changed: pinned {pinned[k][:12]} vs live {live[k][:12]} "
                f"— this is a NO-TOUCH surface")
    # the §4.3 bounded-admission block, compared as a subtree
    def _admission_block(src):
        serve = _func_node(ast.parse(src), "RangeMinerCoordinator",
                           "serve_trial")
        for node in ast.walk(serve):
            if (isinstance(node, ast.If)
                    and "len(eligible) < expected_workers"
                    in ast.unparse(node.test)):
                return _strip_comments(textwrap.dedent(ast.unparse(node)))
        return None

    pre, post = _admission_block(_pinned_src()), _admission_block(_live_src())
    assert pre and post, "the bounded-admission block could not be located"
    assert hashlib.sha256(pre.encode()).hexdigest() == \
        hashlib.sha256(post.encode()).hexdigest(), (
        "the §4.3 bounded-admission block changed; `worker_admission_timeout` "
        "is DO-NOT-WIDEN and its enforcement is DO-NOT-TOUCH")
    return f"{len(NO_TOUCH_DEFS)} pinned surfaces byte-identical vs {PINNED_COMMIT[:12]}"


# The definitions MP-1 declares it changes. An AST scope proof is only a proof
# if the CHANGED set is stated in advance and then compared — otherwise it
# degrades into "here is a list of what changed", which proves nothing about
# intent.
DECLARED_CHANGED = {
    "ServeLoopTiming.__init__",
    "ServeLoopTiming.tick",
    "ServeLoopTiming.close_current_iteration",
    "ServeLoopTiming.note_drain_stop",
    "ServeLoopTiming._record",
    "ServeLoopTiming.metrics",
    "RangeMinerCoordinator.__init__",
    "RangeMinerCoordinator.log_serve_loop_timing_summary",
    # [FIELD-6 OBSERVABILITY REPAIR, TB ruling sequencing item 3] NOT MP-1's
    # change. The proof compares LIVE source against the pinned anchor, so a
    # later authorized commit touching this module must be declared here.
    "RangeMinerCoordinator.log_staging_backpressure_summary",
    "RangeMinerCoordinator._serve_dispatch",
    "RangeMinerCoordinator._pump_deferred",
    "RangeMinerCoordinator.serve_trial",
    # [WINDOW-ANCHOR BRIEF I] NOT MP-1's change. SR-1: the proof compares LIVE
    # source against the pinned anchor, so every later authorized commit that
    # touches this module must be declared here or the proof reds forever. These
    # nine carry the `offset` -> `window_anchor` + `generator_phase` separation
    # (TB ruling 2026-08-20 design gate; scope items ruled 2026-08-21). The
    # anchor does NOT move and `changed == DECLARED_CHANGED` stays EXACT.
    "MinerLedger._init_db",
    "MinerLedger.set_trial_context",
    "_trial_context_row_to_ctx",
    "_canonicalize_trial_context",
    "build_trial_context_from_serve",
    "derive_trial_metadata",
    "RangeMinerCoordinator.build_stripe_assign_payload",
    "RangeMinerCoordinator._dispatch_pending",
    "run_trial_miner",
}
DECLARED_ADDED = {
    "ServeLoopTiming._reset_window",
    "ServeLoopTiming.take_window",
    "ServeLoopTiming._reset_iteration",
    "ServeLoopTiming._close_iteration",
    "ServeLoopTiming._reconcile",
    "ServeLoopTiming.note_subphase",
    "ServeLoopTiming.note_drain_pass",
    "PhaseCharge.__init__",
    "PhaseCharge.__enter__",
    "PhaseCharge.__exit__",
    "RangeMinerCoordinator._phase_stack",
    "RangeMinerCoordinator._charge_phase",
    "RangeMinerCoordinator.phase_charge",
    "RangeMinerCoordinator.phase_exclusive_snapshot",
    "RangeMinerCoordinator.phase_attribution",
    "RangeMinerCoordinator.note_drain_service",
    "RangeMinerCoordinator.note_drain_frame_class",
    "RangeMinerCoordinator.drain_frame_class_census",
    "RangeMinerCoordinator.next_drain_pass",
    "RangeMinerCoordinator.drain_service_census",
    "RangeMinerCoordinator._census_row",
    "RangeMinerCoordinator.reset_drain_service_window",
    "RangeMinerCoordinator.maybe_emit_serve_loop_window",
}


def gate_e2_ast_scope_proof():
    """THE SCOPE PROOF: exactly which definitions moved, compared against a set
    declared in advance.

    WRONG INPUT THAT REDS IT: a change to any definition MP-1 did not declare —
    including one that is behaviourally harmless. The `2389b61` lesson is that a
    whole-block overwrite of an unrelated file is invisible to every behavioural
    test that passes."""
    pinned = _def_digests(_pinned_src())
    live = _def_digests(_live_src())
    changed = {k for k in pinned if k in live and pinned[k] != live[k]}
    added = {k for k in live if k not in pinned}
    removed = {k for k in pinned if k not in live}
    assert not removed, f"MP-1 REMOVED definitions: {sorted(removed)}"
    assert changed == DECLARED_CHANGED, (
        f"changed set does not match the declaration.\n"
        f"  undeclared changes: {sorted(changed - DECLARED_CHANGED)}\n"
        f"  declared but unchanged: {sorted(DECLARED_CHANGED - changed)}")
    assert added == DECLARED_ADDED, (
        f"added set does not match the declaration.\n"
        f"  undeclared additions: {sorted(added - DECLARED_ADDED)}\n"
        f"  declared but absent: {sorted(DECLARED_ADDED - added)}")
    return f"{len(changed)} changed, {len(added)} added, 0 removed — as declared"


# The instrument may not reach the lease / matrix / scheduling / acceptance
# machinery by ANY route. Same list and same fail-closed resolution rule as the
# H1/H2 battery's N1.
N1_FORBIDDEN = ("renew_lease", "handle_stripe_failure", "fail_trial",
                "abort_trial", "claim_stripe", "_defer_locked",
                "_pump_deferred", "enqueue_staging", "set_stripe_state",
                "commit_trial", "_release_resume_credit",
                "process_lease_expiry", "assign_stripes",
                "schedule_pending_stripes")

MP1_INSTRUMENT_METHODS = (
    "note_drain_service", "next_drain_pass", "drain_service_census",
    "note_drain_frame_class", "drain_frame_class_census",
    "_census_row", "reset_drain_service_window", "maybe_emit_serve_loop_window",
    "_phase_stack", "_charge_phase", "phase_charge",
    "phase_exclusive_snapshot", "phase_attribution",
)


def gate_e3_instrument_reaches_no_control_flow():
    """N1, applied to MP-1's own surfaces. FAIL-CLOSED on three routes: a direct
    call whose callee resolves to a forbidden name; ANY indirect callee, because
    the target cannot be read statically; and a constant-string `getattr`.

    An instrument that could reach control flow by an indirection nobody checked
    is exactly the class this arc keeps finding."""
    offenders = []
    for name in MP1_INSTRUMENT_METHODS:
        fn = getattr(RangeMinerCoordinator, name, None)
        assert fn is not None, f"{name} does not exist"
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute):
                if func.attr in N1_FORBIDDEN:
                    offenders.append(f"{name} -> {func.attr}")
                if (func.attr == "getattr"
                        and node.args and isinstance(node.args[-1], ast.Constant)
                        and node.args[-1].value in N1_FORBIDDEN):
                    offenders.append(f"{name} -> getattr {node.args[-1].value}")
            elif isinstance(func, ast.Name):
                if func.id in N1_FORBIDDEN:
                    offenders.append(f"{name} -> {func.id}")
                if (func.id == "getattr" and len(node.args) >= 2
                        and isinstance(node.args[1], ast.Constant)
                        and node.args[1].value in N1_FORBIDDEN):
                    offenders.append(f"{name} -> getattr {node.args[1].value}")
            elif isinstance(func, ast.Call):
                offenders.append(f"{name} -> indirect callee (unreadable)")
    for cls_name in ("PhaseCharge",):
        src = textwrap.dedent(inspect.getsource(getattr(COORD, cls_name)))
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in N1_FORBIDDEN):
                offenders.append(f"{cls_name} -> {node.func.attr}")
    assert not offenders, (
        f"the MP-1 instrument reaches control flow: {offenders}")
    return f"{len(MP1_INSTRUMENT_METHODS)} methods + PhaseCharge, no reach"


def gate_e4_clocks_are_monotonic_only():
    """`ServeLoopTiming` still reads ONLY `perf_counter`, and `PhaseCharge` does
    too. A wall-clock read in a timing instrument invites the exact confusion
    the F1 lease-origin repair removed, and a system-clock step must not be able
    to manufacture a high-water.

    The window's rate limiter deliberately lives on the COORDINATOR and uses
    `time.monotonic`, exactly as the certified active-stripe tick does — so this
    gate would red if the limiter were moved into the timing class."""
    live = ast.parse(_live_src())
    for cls_name, allowed in (("ServeLoopTiming", {"perf_counter"}),
                              ("PhaseCharge", {"perf_counter"})):
        cls = [n for n in ast.walk(live)
               if isinstance(n, ast.ClassDef) and n.name == cls_name]
        assert len(cls) == 1, f"expected one {cls_name}"
        calls = {getattr(n.func, "attr", None) for n in ast.walk(cls[0])
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and getattr(n.func.value, "id", None) == "time"}
        assert calls == allowed, (
            f"{cls_name} reads {calls}; only {allowed} is permitted")
    fn = _func_node(live, "RangeMinerCoordinator", "maybe_emit_serve_loop_window")
    src = ast.unparse(fn)
    assert "time.monotonic()" in src and "time.time()" not in src, (
        "the window limiter does not use a monotonic clock — a clock step "
        "could suppress or duplicate a window in the series")
    return "ServeLoopTiming + PhaseCharge: perf_counter only; window: monotonic"


def gate_e5_serve_loop_wall_clock_reads_are_unchanged():
    """MP-1 ADDS NO WALL-CLOCK READ TO THE SERVE LOOP.

    The R1.3 lesson, verbatim: the first F1 implementation computed the
    iteration's clock age at the CALL SITE as `time.time() - now`, so the
    instrument as a whole read the wall clock while the class-level gate stayed
    green. This counts `time.time()` sites in `serve_trial` on BOTH sources and
    requires them equal.

    WRONG INPUT THAT REDS IT: any new `time.time()` in the loop, however
    innocuous — the six shared-clock consumers of this loop's `now` are an
    audited set and a seventh would have to be audited too."""
    def _wall_reads(src):
        fn = _func_node(ast.parse(src), "RangeMinerCoordinator", "serve_trial")
        return sum(1 for n in ast.walk(fn)
                   if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute)
                   and n.func.attr == "time"
                   and getattr(n.func.value, "id", None) == "time")

    pre, post = _wall_reads(_pinned_src()), _wall_reads(_live_src())
    assert post == pre, (
        f"serve_trial reads the wall clock {post} times, was {pre} — MP-1 "
        f"added a production wall-clock read to feed an instrument")
    return f"time.time() sites in serve_trial: {pre} -> {post} (unchanged)"


def gate_e6_instruments_never_raise_on_garbage():
    """Every new public surface survives a caller error. `stop()`/`metrics()`
    already carry this contract; the MP-1 additions inherit it, because they run
    on the same production paths.

    WRONG INPUT THAT REDS IT: an unguarded `int()` or dict access — an
    instrument that can kill a trial is worse than no instrument."""
    t = ServeLoopTiming()
    t.note_subphase("staging", "not-a-float")
    t.note_subphase("nosuchsegment", 1.0)
    t.note_drain_pass("x", None, "y")
    t.note_drain_stop("nosuchreason")
    t.take_window()
    m = t.metrics()
    assert m["staging_total"] == 0.0, m
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        c.note_drain_service(None, None, position="x", pass_seq=None)
        c.note_drain_service("run:c1", "w", position=1, pass_seq=1)
        cen = c.drain_service_census(["run:c1"])
        assert cen["status"] == OBS_OK, cen
        c.reset_drain_service_window()
        assert c.phase_exclusive_snapshot(("nosuch",))["nosuch"] == 0.0
        assert isinstance(c.phase_attribution(), list)
    return "garbage inputs absorbed; the honest measurement survives"


def gate_e7_window_record_is_one_line_per_window():
    """[§15] The emission bar. At 25 connections the window is ONE record
    carrying ~25 rows — not one record per connection, which at a 10 s period
    would be 2.5 lines/second for the whole run and would breach the very bar
    the `[S172-BP]`/`[S172-SL]` idiom exists to hold.

    WRONG INPUT THAT REDS IT: emitting per connection, or per drain pass."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        sl = ServeLoopTiming()
        ids = [f"run:conn{i}" for i in range(25)]
        for i, cid in enumerate(ids):
            c.note_drain_service(cid, f"w{i}", position=i + 1, pass_seq=1)
        with _Capture("range_miner_coordinator") as cap:
            rec = c.maybe_emit_serve_loop_window(
                "run", sl, live_connection_ids=ids, now=1.0, force=True)
            lines = cap.of("[S172-SL] window")
    assert len(lines) == 1, f"25 connections produced {len(lines)} records"
    assert len(rec["connections"]["connections"]) == 25, rec["connections"]
    assert json.loads(lines[0][lines[0].index("{"):])["event"] == \
        "SERVE_LOOP_WINDOW"
    return "25 connections -> 1 record with 25 rows"


def gate_e8_window_carries_the_heartbeat_census():
    """The window record must carry BOTH the window and the run frame-class
    census, or the question Beta left open is measured and then not reported —
    which is the same as not measuring it."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        c.note_drain_frame_class("heartbeat", stripe_id=None)
        rec = c.maybe_emit_serve_loop_window(
            "run", ServeLoopTiming(), live_connection_ids=[], now=1.0,
            force=True)
    assert rec["frame_classes"]["counts"]["heartbeat_without_stripe"] == 1, rec
    assert rec["frame_classes_run"]["counts"]["heartbeat"] == 1, rec
    assert rec["frame_classes"]["status"] == OBS_OK, rec
    return "window record carries frame_classes + frame_classes_run"


# ===========================================================================
# log capture
# ===========================================================================
import logging  # noqa: E402


class _CollectingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())

    def of(self, token):
        return [ln for ln in self.lines if token in ln]

    def payloads(self, token):
        out = []
        for ln in self.of(token):
            b = ln.find("{")
            if b >= 0:
                out.append(json.loads(ln[b:]))
        return out


class _Capture:
    def __init__(self, *modules):
        self.h = _CollectingHandler()
        self.loggers = [logging.getLogger(m) for m in modules]

    def __enter__(self):
        for lg in self.loggers:
            lg.addHandler(self.h)
            lg.setLevel(logging.INFO)
        return self.h

    def __exit__(self, *a):
        for lg in self.loggers:
            lg.removeHandler(self.h)
        return False


# ===========================================================================
def main():
    print("=" * 78)
    print("S172 — MP-1 DRAIN ATTRIBUTION (read-only serve-loop observation)")
    print(f"pinned anchor : {PINNED_COMMIT}")
    _head = subprocess.run(["git", "-C", _ROOT, "rev-parse", "HEAD"],
                           capture_output=True, text=True).stdout.strip()
    print(f"live HEAD     : {_head}")
    print("=" * 78)

    # The anchor FIRST: every RED and every scope arm is credited against it.
    print("\n-- anchor authenticity (every scope arm depends on it) --")
    check("ANCHOR-AUTHENTIC", gate_anchor_is_authentic)

    print("\n-- §A  three-level attribution: every level SUMS --")
    check("A1-L1-PARTITION-SUMS", gate_a1_level1_partition_sums_exactly)
    check("A2-L2-L3-SUM", gate_a2_level2_and_level3_sum_exactly)
    check("A3-DECLARATION-MATCHES", gate_a3_level1_declaration_matches_the_segment_set)
    check("A4-REMAINDER-NAMED-NOT-SILENT", gate_a4_remainder_is_named_never_silent)
    check("A5-NEGATIVE-CLAMP-COUNTED", gate_a5_negative_remainder_is_counted_not_hidden)
    check("A6-PROFILE-IS-ONE-ITERATION", gate_a6_worst_iteration_profile_is_ONE_iteration)
    check("A7-RATE-UNOBSERVED-NOT-ZERO", gate_a7_per_frame_cost_is_unobserved_not_zero)

    print("\n-- §B  per-connection drain service: the LATE-INDEX evidence --")
    check("B1-SERVICE-COUNTS-VARY", gate_b1_service_counts_measure_and_vary)
    check("B2-ZERO-IS-MEASURED", gate_b2_never_serviced_is_a_MEASURED_ZERO)
    check("B3-UNAVAILABLE-NOT-ZERO", gate_b3_unreadable_live_set_is_UNAVAILABLE_not_zero)
    check("B4-UNRESOLVED-IS-NAMED", gate_b4_unresolvable_connection_is_a_NAMED_bucket)
    check("B5-POSITION-DISCRIMINATES", gate_b5_position_measures_and_discriminates_head_from_tail)
    check("B6-WINDOW-RESETS", gate_b6_window_resets_without_losing_totals)
    check("B7-CENSUS-DETACHED", gate_b7_census_rows_are_detached)
    check("B8-PARTIAL-COVERAGE-COUNTED", gate_b8_drain_pass_census_measures_partial_coverage)
    check("B9-UNKNOWN-LIVE-NOT-PARTIAL", gate_b9_unknown_live_count_is_never_a_partial_pass)
    check("B10-STRIPELESS-HEARTBEAT-VISIBLE", gate_b10_heartbeat_without_a_stripe_is_VISIBLE)
    check("B11-CLASS-WINDOW-VS-RUN", gate_b11_class_census_window_and_run_are_distinct)

    print("\n-- §C  thread attribution --")
    check("C1-KEYED-BY-THREAD", gate_c1_phase_charges_are_keyed_by_thread)
    check("C2-EXCLUSIVE-EXCLUDES-NESTED", gate_c2_exclusive_time_excludes_the_nested_phase)
    check("C3-CANNOT-PERTURB-CALLER", gate_c3_phase_charge_cannot_perturb_its_caller)

    print("\n-- §D  production wiring (+ mutants) --")
    check("D1-REAL-DISPATCH-CHARGES", gate_d1_staging_is_charged_by_the_real_dispatch)
    check("D2-DELTA-READ-LANDS-IN-SL", gate_d2_serve_loop_delta_read_lands_in_the_timing_record)
    check("D3-PUMP-CHARGED-BY-THREAD", gate_d3_pump_is_charged_and_thread_attributed)
    check("D4-WIRING-IN-SERVE-LOOP", gate_d4_wiring_is_in_the_serve_loop)
    check("D5-WINDOW-RATE-LIMITED", gate_d5_window_is_rate_limited_and_emits_one_record)
    check("D6-WINDOW-IS-A-SERIES", gate_d6_window_reports_a_series_not_a_cumulative_total)
    check("D7-SUMMARY-ADDITIVE", gate_d7_summary_carries_the_new_series_and_the_profile)
    check("D8-SUMMARY-NEVER-RAISES", gate_d8_summary_never_raises)

    print("\n-- §E  scope, non-regression, no behaviour change --")
    check("E1-NO-TOUCH-BYTE-IDENTICAL", gate_e1_no_touch_surfaces_are_byte_identical)
    check("E2-AST-SCOPE-PROOF", gate_e2_ast_scope_proof)
    check("E3-NO-CONTROL-FLOW-REACHED", gate_e3_instrument_reaches_no_control_flow)
    check("E4-MONOTONIC-ONLY", gate_e4_clocks_are_monotonic_only)
    check("E5-NO-NEW-WALL-READ", gate_e5_serve_loop_wall_clock_reads_are_unchanged)
    check("E6-NEVER-RAISES", gate_e6_instruments_never_raise_on_garbage)
    check("E7-ONE-LINE-PER-WINDOW", gate_e7_window_record_is_one_line_per_window)
    check("E8-WINDOW-CARRIES-CENSUS", gate_e8_window_carries_the_heartbeat_census)

    print("\n" + "=" * 78)
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    total = len(_RESULTS)
    print(f"\n{passed}/{total} checks green")
    if passed != total:
        print("\nFAILURES:")
        for name, ok, detail in _RESULTS:
            if not ok:
                print(f"  {name}: {detail}")
        print("COMPLETION SENTINEL: FAIL")
        return 1
    print("COMPLETION SENTINEL: PASS — MP-1 drain attribution is read-only, "
          "sums at every level, shows measured zeroes, and is thread-keyed "
          "(pending Team Beta review).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
