#!/usr/bin/env python3
"""
test_s172_phase5_d6_threshold_path.py — S172 Phase 5, D6 CORRECTION PASS gates.

Spec: docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D6_CORRECTION.md (REV1) §4/§5,
encoding Team Beta's D6 disposition.

THE BLOCKER THESE GATES CLOSE
-----------------------------
`build_stripe_assign_payload` emitted no threshold field, so the worker fell back
to a hardcoded 0.25 and Optuna's per-direction `forward_threshold` /
`reverse_threshold` never reached the kernel — the optimizer certified results for
a configuration other than the one it requested. The correction makes the parent
resolve the directional threshold per stripe from the §6.8 phase table and emit it
explicitly; the worker never chooses.

THE NINE CHECKS (Beta §4), all at asymmetric forward=0.31 / reverse=0.47:

  G1  forward assignments carry exactly 0.31
  G2  reverse assignments carry exactly 0.47
  G3  the worker receives each exact value (over the real protocol framing)
  G4  the executor/kernel receives each exact value (real SieveExecutor, real
      kernel-arg materialization, captured at the single kernel entry)
  G5  values are not collapsed (forward != reverse preserved)
  G6  values are not swapped
  G7  a LEGACY payload that omits the field still resolves to 0.25 (back-compat)
  G8  a NEW D6 payload always carries an explicit threshold — no silent fallback
  G9  effective-threshold provenance: requested == payload == effective

PLUS the shared-authority residue fix (Beta §5):

  R1-R3  coordinator-side and worker-side residue derivation produce IDENTICAL
         ORDERED residues for sessions = both / midday-only / evening-only
  R4     the assignment round-trip verifies (no ResidueVerificationError) in all
         three session cases

House rules honoured here:
  * oracles are HAND-TRANSCRIBED literals, never imported from a module under
    test;
  * fixtures drive the REAL producer surface (coordinator -> dispatch -> wire ->
    worker -> executor -> kernel args), never a hand-built payload where a real
    one is reachable;
  * every mutant is proved under the four-part rule (applies-once,
    mutated-path-executed, detector-clean-unmutated, injected-defect).

G4 and G9 execute the REAL GPU executor and therefore require cupy + a visible
device. They FAIL LOUDLY rather than skipping if that is absent: a D6 threshold
gate that cannot reach the kernel has not tested the thing that was broken.

Run:  python tests/test_s172_phase5_d6_threshold_path.py
"""
import ast
import importlib.util
import inspect
import json
import os
import sys
import tempfile
import traceback
from typing import Any, Dict, List, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from miner.range_miner_coordinator import (  # noqa: E402
    CoordinatorConfig,
    MinerLedger,
    NodeConfig,
    RangeMinerCoordinator,
)
from miner.range_miner_protocol import (  # noqa: E402
    StripeAssignMessage,
    SubStripeResultMessage,
    from_dict,
    message_to_bytes,
)
from miner.range_miner_worker import (  # noqa: E402
    ResidueResolver,
    ResidueVerificationError,
    SieveExecutor,
    load_residue_window,
    sha256_residues,
    supported_variants,
)

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"
_results: List[Tuple[str, bool, Any]] = []
_MUTANTS: List[Tuple[str, str, str, str]] = []

_COORD_PATH = os.path.join(_ROOT, "miner", "range_miner_coordinator.py")
_INTEG_PATH = os.path.join(_ROOT, "window_optimizer_integration_final.py")
with open(_COORD_PATH, "r", encoding="utf-8") as _f:
    _COORD_SRC = _f.read()
with open(_INTEG_PATH, "r", encoding="utf-8") as _f:
    _INTEG_SRC = _f.read()


# ═════════════════════════════════════════════════════════════════════════════
# Hand-transcribed oracles — literals, never imported from a module under test
# ═════════════════════════════════════════════════════════════════════════════

# Beta's asymmetric test values. Chosen so a collapse, a swap and the legacy
# fallback are all three distinguishable from each other and from 0.25.
FWD = 0.31
REV = 0.47

# The legacy hardcoded fallback that the blocker exposed
# (range_miner_worker.py: coerce_threshold(..., 0.25)). Kept ONLY for pre-D6
# payloads.
LEGACY_FALLBACK = 0.25

# §6.8 workflow table, hand-transcribed from the spec (NOT imported from
# workflow_phase_semantics, which is itself under test):
#   1 -> forward/constant   2 -> reverse/constant
#   3 -> forward/variable   4 -> reverse/variable
ORACLE_PHASE_TABLE: Dict[int, Tuple[str, str, str]] = {
    1: ("forward", "constant", "java_lcg"),
    2: ("reverse", "constant", "java_lcg_reverse"),
    3: ("forward", "variable", "java_lcg_hybrid"),
    4: ("reverse", "variable", "java_lcg_hybrid_reverse"),
}

# The directional threshold each phase MUST end up filtering at.
ORACLE_PHASE_THRESHOLD: Dict[int, float] = {1: FWD, 2: REV, 3: FWD, 4: REV}

# Index of the single float32 threshold scalar in each kernel's argument list,
# hand-transcribed from the AUDITED ABIs in range_miner_worker.py's header:
#   constant prefix: seeds, residues, survivors, match_rates, best_skips,
#                    survivor_count, n_seeds, k, skip_min, skip_max, THRESHOLD
#   hybrid prefix:   seeds, residues, survivors, match_rates, skip_sequences,
#                    strategy_ids, survivor_count, n_seeds, k,
#                    strategy_max_misses, strategy_tolerances, n_strategies,
#                    THRESHOLD
ORACLE_THRESHOLD_ARG_INDEX = {"constant": 10, "variable": 12}

TOTAL_SEEDS = 4_000_000
STRIPE_SIZE = 2_000_000
SUB_CAP = 1_000_000
WINDOW_SIZE = 3
SPOOL_ROOT = "/var/spool/miner"


# ═════════════════════════════════════════════════════════════════════════════
# Harness
# ═════════════════════════════════════════════════════════════════════════════
def _check(name, fn):
    try:
        fn()
        _results.append((name, True, None))
        print(f"  [{_PASS}] {name}")
    except Exception as e:                                      # noqa: BLE001
        _results.append((name, False, traceback.format_exc()))
        print(f"  [{_FAIL}] {name}: {e}")


def _dataset(tmp: str) -> str:
    """A dataset whose midday and evening rows are DISJOINT in value, so a
    dropped session filter changes the residue window observably rather than by
    luck."""
    rows = []
    for i in range(40):
        rows.append({"date": f"2020-01-{i + 1:02d}", "session": "midday",
                     "draw": 100 + i})
        rows.append({"date": f"2020-01-{i + 1:02d}", "session": "evening",
                     "draw": 700 + i})
    path = os.path.join(tmp, "dataset.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh)
    return path


def _coord(tmp: str, coord_cls=RangeMinerCoordinator, dbname="l.db"):
    ledger_cls = MinerLedger
    cfg_cls = CoordinatorConfig
    if coord_cls is not RangeMinerCoordinator:      # a mutant module
        mod = sys.modules[coord_cls.__module__]
        ledger_cls, cfg_cls = mod.MinerLedger, mod.CoordinatorConfig
    ledger = ledger_cls(os.path.join(tmp, dbname))
    cfg = cfg_cls(staging_dir=os.path.join(tmp, "staging"),
                  miner_stripe_size=STRIPE_SIZE,
                  seed_cap_amd=SUB_CAP, seed_cap_nvidia=SUB_CAP,
                  seed_cap_amd_hybrid=SUB_CAP, seed_cap_nvidia_hybrid=SUB_CAP)
    return coord_cls(cfg, ledger)


def _register(coord, wid="hostA:gpu0"):
    node = NodeConfig(hostname=wid.split(":")[0], spool_root=SPOOL_ROOT,
                      ssh_address="10.0.0.9", ssh_user="michael")
    caps = {"amd": SUB_CAP, "nvidia": SUB_CAP,
            "amd_hybrid": SUB_CAP, "nvidia_hybrid": SUB_CAP}
    return coord.register_worker(
        worker_id=wid, hostname=wid.split(":")[0], backend="cuda",
        capabilities={"seed_caps": caps,
                      "supported_variants": list(supported_variants())},
        node_config=node, now=100.0)


class _CapturingSocket:
    """Stands in for the framed socket at the dispatch boundary and records the
    StripeAssignMessage objects the coordinator actually sent."""

    def __init__(self):
        self.sent: List[StripeAssignMessage] = []

    def send_msg(self, msg):
        self.sent.append(msg)


def _dispatch_assignments(tmp, sessions=("midday", "evening"),
                          coord_cls=RangeMinerCoordinator,
                          fwd=FWD, rev=REV, dbname="l.db"):
    """Drive the REAL assignment path for all four §6.8 phases and return
    {phase: [StripeAssignMessage, ...]} exactly as the coordinator emitted them.

    This is the production route — assign_stripes -> _dispatch_pending ->
    build_stripe_assign_payload — not a hand-built payload."""
    ds = _dataset(tmp)
    coord = _coord(tmp, coord_cls, dbname=dbname)
    run_id = "run_thresh"
    coord.ledger.create_trial(run_id, 7, now=100.0)
    sess = list(sessions)
    residues = load_residue_window(ds, WINDOW_SIZE, sess, 0)
    coord.ledger.set_trial_context(run_id, {
        "trial_number": 7, "window_size": WINDOW_SIZE, "offset": 0,
        "sessions": sess, "skip_min": 0, "skip_max": 16,
        "prng_base": "java_lcg",
        "forward_threshold": fwd, "reverse_threshold": rev,
        "dataset_sha256": coord.__class__.__module__ and
        sys.modules[coord.__class__.__module__].compute_dataset_sha256(ds),
        "residue_sha256": sha256_residues(residues),
    })
    conn = _register(coord)
    ds_sha = sys.modules[coord.__class__.__module__].compute_dataset_sha256(ds)

    out: Dict[int, List[StripeAssignMessage]] = {}
    # ONE `dispatched` set across all four stages, exactly as serve_trial keeps
    # it. This matters: _dispatch_pending iterates every CLAIMED stripe in the
    # run, not only the current stage's, and relies on the caller's `dispatched`
    # set (plus the stage barrier, which only advances once a stage's stripes are
    # all DONE) to avoid re-sending an earlier stage's stripe under a later
    # stage's phase. A fixture that passed a fresh set per phase would silently
    # re-dispatch phases 1-3 with phase 4's threshold and test nothing.
    dispatched: set = set()
    for phase, (_direction, _mode, family) in ORACLE_PHASE_TABLE.items():
        coord.assign_stripes(run_id, family, phase, TOTAL_SEEDS, [conn],
                             stripe_prefix=f"{run_id}__p{phase}", now=100.0)
        sock = _CapturingSocket()
        coord._dispatch_pending(
            run_id, family, phase, {conn.worker_id: sock}, dispatched, ds, ds_sha,
            WINDOW_SIZE, sess, 0, residues, 7, fwd, rev)
        assert sock.sent, f"phase {phase}: no assignment was dispatched"
        for m in sock.sent:
            assert m.phase == phase, (
                f"fixture leaked a phase-{m.phase} assignment into the phase "
                f"{phase} dispatch")
        out[phase] = list(sock.sent)
    return coord, run_id, ds, residues, out


# ═════════════════════════════════════════════════════════════════════════════
# G1 / G2 / G5 / G6 — what the assignments carry
# ═════════════════════════════════════════════════════════════════════════════
def _payload_thresholds(tmp, coord_cls=RangeMinerCoordinator, fwd=FWD, rev=REV):
    _c, _r, _d, _res, sent = _dispatch_assignments(
        tmp, coord_cls=coord_cls, fwd=fwd, rev=rev)
    vals: Dict[int, List[float]] = {}
    for phase, msgs in sent.items():
        vals[phase] = [m.payload.get("min_match_threshold") for m in msgs]
    return vals


def g1_forward_payload(coord_cls=RangeMinerCoordinator):
    with tempfile.TemporaryDirectory() as tmp:
        vals = _payload_thresholds(tmp, coord_cls)
    for phase in (1, 3):
        got = vals[phase]
        assert got and all(v == FWD for v in got), (
            f"G1: phase {phase} (forward) assignments carry {got!r}, "
            f"expected every stripe at exactly {FWD}")


def g2_reverse_payload(coord_cls=RangeMinerCoordinator):
    with tempfile.TemporaryDirectory() as tmp:
        vals = _payload_thresholds(tmp, coord_cls)
    for phase in (2, 4):
        got = vals[phase]
        assert got and all(v == REV for v in got), (
            f"G2: phase {phase} (reverse) assignments carry {got!r}, "
            f"expected every stripe at exactly {REV}")


def g5_not_collapsed(coord_cls=RangeMinerCoordinator):
    """Forward and reverse must remain DISTINCT. Note this check alone cannot
    detect a swap — that is exactly why G6 exists as its own detector."""
    with tempfile.TemporaryDirectory() as tmp:
        vals = _payload_thresholds(tmp, coord_cls)
    fwd_vals = set(vals[1]) | set(vals[3])
    rev_vals = set(vals[2]) | set(vals[4])
    assert len(fwd_vals) == 1 and len(rev_vals) == 1, (
        f"G5: a direction carries more than one value (fwd={fwd_vals!r} "
        f"rev={rev_vals!r})")
    assert fwd_vals != rev_vals, (
        f"G5: forward and reverse thresholds COLLAPSED to the same value "
        f"{fwd_vals!r} — the per-direction tuning was lost")


def g6_not_swapped(coord_cls=RangeMinerCoordinator):
    """The forward phases must carry the FORWARD value specifically. Two
    consistently-reversed branches still look asymmetric, so G5 passes on a swap
    and only this identity check catches it."""
    with tempfile.TemporaryDirectory() as tmp:
        vals = _payload_thresholds(tmp, coord_cls)
    for phase, expected in ORACLE_PHASE_THRESHOLD.items():
        got = set(vals[phase])
        assert got == {expected}, (
            f"G6: phase {phase} ({ORACLE_PHASE_TABLE[phase][0]}) carries {got!r}, "
            f"expected {{{expected}}} — forward/reverse are swapped or misrouted")


# ═════════════════════════════════════════════════════════════════════════════
# G3 — the worker receives each exact value (over the real protocol framing)
# ═════════════════════════════════════════════════════════════════════════════
def g3_worker_receives(coord_cls=RangeMinerCoordinator):
    with tempfile.TemporaryDirectory() as tmp:
        _c, _r, _d, _res, sent = _dispatch_assignments(tmp, coord_cls=coord_cls)
        for phase, msgs in sent.items():
            expected = ORACLE_PHASE_THRESHOLD[phase]
            for msg in msgs:
                # REAL wire round-trip: encode exactly as the coordinator sends
                # it and decode exactly as the worker's recv_msg does.
                raw = message_to_bytes(msg)
                body = json.loads(raw[4:].decode("utf-8"))
                received = from_dict(body)
                assert isinstance(received, StripeAssignMessage)
                mmt = received.payload.get("min_match_threshold")
                p2t = received.payload.get("phase2_threshold")
                assert mmt == expected, (
                    f"G3: phase {phase}: worker received min_match_threshold="
                    f"{mmt!r}, expected {expected}")
                assert p2t == expected, (
                    f"G3: phase {phase}: worker received phase2_threshold="
                    f"{p2t!r}, expected {expected}")


# ═════════════════════════════════════════════════════════════════════════════
# G4 / G7 / G9 — the REAL executor, the REAL kernel arguments
# ═════════════════════════════════════════════════════════════════════════════
class _CapturingExecutor(SieveExecutor):
    """The real SieveExecutor with the single mockable kernel entry captured.
    Everything up to and including kernel-argument materialization is the
    production path; only the launch itself is withheld."""

    def __init__(self, resolver, device_index=0):
        super().__init__(resolver, device_index)
        self.kernel_args = None

    def _gpu_launch(self, kernel, blocks, threads, kernel_args):
        self.kernel_args = kernel_args


def _require_gpu():
    try:
        import cupy  # noqa: F401
    except Exception as e:                                      # noqa: BLE001
        raise AssertionError(
            f"this gate drives the REAL executor and requires cupy + a visible "
            f"device; it does not skip ({type(e).__name__}: {e}). Run it on the "
            f"GPU box (VM 101).")


def _run_executor(payload, family, seed_count=4096, device_index=0):
    """Run the REAL SieveExecutor against `payload`, capturing the kernel args
    and returning (kernel_args, outcome)."""
    _require_gpu()
    ex = _CapturingExecutor(ResidueResolver(), device_index=device_index)
    assign = StripeAssignMessage(
        worker_id="hostA:gpu0", stripe_id="s0", trial_number=7,
        prng_type="java_lcg", family_name=family, seed_start=0,
        seed_count=seed_count, phase=1, payload=payload)
    outcome = ex.execute(assign, 0, seed_count)
    assert ex.kernel_args is not None, "the kernel entry was never reached"
    return ex.kernel_args, outcome


def _kernel_threshold(kernel_args, skip_mode):
    idx = ORACLE_THRESHOLD_ARG_INDEX[skip_mode]
    return round(float(kernel_args[idx]), 6)


def g4_kernel_receives(coord_cls=RangeMinerCoordinator):
    with tempfile.TemporaryDirectory() as tmp:
        _c, _r, _d, _res, sent = _dispatch_assignments(tmp, coord_cls=coord_cls)
        for phase, (_direction, mode, family) in ORACLE_PHASE_TABLE.items():
            payload = dict(sent[phase][0].payload)
            args, _outcome = _run_executor(payload, family)
            got = _kernel_threshold(args, mode)
            expected = ORACLE_PHASE_THRESHOLD[phase]
            assert got == round(expected, 6), (
                f"G4: phase {phase} ({family}) launched the kernel with "
                f"threshold={got!r}, expected {expected} — the value did not "
                f"survive payload -> worker -> executor -> kernel")


def g7_legacy_fallback():
    """A LEGACY (pre-D6) payload carries no threshold field at all; the worker's
    0.25 fallback must still apply, unchanged. This is the ONE case in which the
    fallback is correct."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = _dataset(tmp)
        sess = ["midday", "evening"]
        residues = load_residue_window(ds, WINDOW_SIZE, sess, 0)
        from miner.range_miner_coordinator import compute_dataset_sha256
        legacy = {
            "dataset": ds,
            "dataset_sha256": compute_dataset_sha256(ds),
            "window_size": WINDOW_SIZE,
            "sessions": sess,
            "offset": 0,
            "residue_sha256": sha256_residues(residues),
        }
        assert "min_match_threshold" not in legacy and \
               "phase2_threshold" not in legacy, "the legacy fixture is not legacy"
        for phase, (_d, mode, family) in ORACLE_PHASE_TABLE.items():
            args, outcome = _run_executor(dict(legacy), family)
            got = _kernel_threshold(args, mode)
            assert got == round(LEGACY_FALLBACK, 6), (
                f"G7: a legacy payload for {family} filtered at {got!r}; the "
                f"documented back-compat fallback is {LEGACY_FALLBACK}")
            assert outcome.effective_threshold == LEGACY_FALLBACK, (
                f"G7: effective_threshold={outcome.effective_threshold!r} does "
                f"not report the fallback that was actually used")


def g8_no_silent_fallback(coord_cls=RangeMinerCoordinator):
    """A NEWLY generated D6 payload must ALWAYS carry an explicit threshold.
    Three independent legs: the signature makes omission impossible, omission
    actually raises, and every dispatched payload carries the value."""
    sig = inspect.signature(coord_cls.build_stripe_assign_payload)
    for name in ("phase", "forward_threshold", "reverse_threshold"):
        assert name in sig.parameters, (
            f"G8: build_stripe_assign_payload has no {name!r} parameter — a D6 "
            f"payload could be built with no threshold at all")
        p = sig.parameters[name]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"G8: {name!r} is {p.kind}, expected KEYWORD_ONLY")
        assert p.default is inspect.Parameter.empty, (
            f"G8: {name!r} has default {p.default!r}; a defaulted threshold is a "
            f"silent fallback by another name")

    # Leg 2: the builder's OWN output, called directly, so the presence
    # assertion below is exercised even by a mutant that would crash the
    # dispatch path before it (a kill must be attributable to this check, not
    # to a downstream KeyError).
    with tempfile.TemporaryDirectory() as tmp:
        ds = _dataset(tmp)
        coord = _coord(tmp, coord_cls, dbname="g8direct.db")
        mod = sys.modules[coord.__class__.__module__]
        direct = coord.build_stripe_assign_payload(
            ds, WINDOW_SIZE, ["midday", "evening"], 0, [1, 2, 3],
            dataset_sha256=mod.compute_dataset_sha256(ds),
            phase=1, forward_threshold=FWD, reverse_threshold=REV)
        assert "min_match_threshold" in direct, (
            "G8: build_stripe_assign_payload returned a payload with NO "
            "min_match_threshold — the worker would silently fall back to "
            f"{LEGACY_FALLBACK}")
        assert "phase2_threshold" in direct, (
            "G8: build_stripe_assign_payload returned a payload with NO "
            "phase2_threshold — hybrid stripes would silently fall back")
        assert direct["min_match_threshold"] == FWD, (
            f"G8: direct build carried {direct['min_match_threshold']!r}, "
            f"expected {FWD}")

    # Leg 3: every payload the REAL dispatch path actually emitted.
    with tempfile.TemporaryDirectory() as tmp:
        _c, _r, _d, _res, sent = _dispatch_assignments(tmp, coord_cls=coord_cls)
        for phase, msgs in sent.items():
            for msg in msgs:
                pl = msg.payload
                assert "min_match_threshold" in pl, (
                    f"G8: phase {phase} payload has NO min_match_threshold — the "
                    f"worker would silently fall back to {LEGACY_FALLBACK}")
                assert "phase2_threshold" in pl, (
                    f"G8: phase {phase} payload has NO phase2_threshold — hybrid "
                    f"stripes would silently fall back")
                assert pl["min_match_threshold"] == pl["phase2_threshold"], (
                    f"G8: phase {phase} payload carries a contradictory pair "
                    f"{pl['min_match_threshold']!r} vs {pl['phase2_threshold']!r}")
                assert pl["min_match_threshold"] != LEGACY_FALLBACK, (
                    f"G8: phase {phase} payload carries the legacy fallback value "
                    f"{LEGACY_FALLBACK} — indistinguishable from no threshold at "
                    f"all (the fixture uses {FWD}/{REV} precisely so this is "
                    f"decidable)")


def g9_provenance(coord_cls=RangeMinerCoordinator):
    """requested == payload == effective, for a non-default asymmetric value,
    with `effective` coming back off the REAL executor through the REAL result
    envelope — not recomputed from the payload."""
    with tempfile.TemporaryDirectory() as tmp:
        coord, run_id, _ds, _res, sent = _dispatch_assignments(
            tmp, coord_cls=coord_cls)
        for phase, (_direction, _mode, family) in ORACLE_PHASE_TABLE.items():
            payload = dict(sent[phase][0].payload)
            _args, outcome = _run_executor(payload, family)
            # The effective value travels on the REAL result envelope and is
            # decoded exactly as the coordinator's dispatch decodes it.
            msg = SubStripeResultMessage(
                worker_id="hostA:gpu0", stripe_id=sent[phase][0].stripe_id,
                sub_index=0, seed_start=0, seed_count=4096,
                survivor_count=outcome.count, inline={"survivors": []},
                effective_threshold=outcome.effective_threshold)
            raw = message_to_bytes(msg)
            received = from_dict(json.loads(raw[4:].decode("utf-8")))
            coord.record_substripe_effective(
                run_id, sent[phase][0].stripe_id, 0, 0,
                received.effective_threshold)
            coord.record_stripe_complete_effective(
                run_id, sent[phase][0].stripe_id, 0,
                received.effective_threshold)

        prov = coord.threshold_provenance(run_id, validated=True)
        assert prov["requested"] == {"forward": FWD, "reverse": REV}, (
            f"G9: requested leg is {prov['requested']!r}")
        for phase, expected in ORACLE_PHASE_THRESHOLD.items():
            assert prov["payload"].get(phase) == [expected], (
                f"G9: phase {phase} payload leg is {prov['payload'].get(phase)!r}, "
                f"expected [{expected}]")
            assert prov["effective"].get(phase) == [expected], (
                f"G9: phase {phase} EFFECTIVE leg is "
                f"{prov['effective'].get(phase)!r}, expected [{expected}] — the "
                f"kernel did not filter at what the payload transmitted")
            direction = prov["phase_direction"][phase]
            assert prov["requested"][direction] == expected, (
                f"G9: phase {phase} direction {direction!r} does not tie back to "
                f"the requested value")


# ═════════════════════════════════════════════════════════════════════════════
# G10 / G11 / G12 — the PARENT-SIDE FAIL-CLOSED provenance gate
# (Beta's commit ruling: these three conditions must ABORT the trial, not merely
# be recorded. Each drives the REAL validator on a REAL dispatched assignment.)
# ═════════════════════════════════════════════════════════════════════════════
def _provenance_fixture(tmp, coord_cls=RangeMinerCoordinator):
    """Real dispatch for all four phases, then the worker-side reports recorded
    through the REAL recorders. Returns (coord, run_id, sent) with every stripe
    correctly reporting its assigned threshold — the clean baseline the three
    violation gates each perturb in exactly one way."""
    coord, run_id, _ds, _res, sent = _dispatch_assignments(
        tmp, coord_cls=coord_cls)
    for phase, msgs in sent.items():
        for m in msgs:
            eff = m.payload["min_match_threshold"]
            coord.record_substripe_effective(run_id, m.stripe_id, 0, 0, eff)
            coord.record_substripe_effective(run_id, m.stripe_id, 0, 1, eff)
            coord.record_stripe_complete_effective(run_id, m.stripe_id, 0, eff)
    return coord, run_id, sent


def _prov_error_cls(coord_cls=RangeMinerCoordinator):
    return sys.modules[coord_cls.__module__].ThresholdProvenanceError


def _expect_violation(coord, run_id, coord_cls, needle, gate):
    err = _prov_error_cls(coord_cls)
    try:
        coord.validate_threshold_provenance(run_id)
    except err as exc:
        assert needle in str(exc), (
            f"{gate}: the validator raised, but its diagnostic does not name the "
            f"violation ({needle!r} absent from {str(exc)[:300]!r})")
        return
    raise AssertionError(
        f"{gate}: the parent ACCEPTED the violation — validate_threshold_"
        f"provenance returned instead of raising ThresholdProvenanceError, so a "
        f"trial with unproven kernel filtering would reach commit and "
        f"certification")


def g10_missing_effective(coord_cls=RangeMinerCoordinator):
    """A D6-generated assignment whose effective threshold never came back must
    ABORT the trial. Optional schema fields make legacy absence representable;
    they do NOT make provenance optional for a D6 run."""
    with tempfile.TemporaryDirectory() as tmp:
        coord, run_id, sent = _provenance_fixture(tmp, coord_cls)
        # positive control: the clean fixture validates
        coord.validate_threshold_provenance(run_id)
        victim = sent[1][0]
        coord.record_substripe_effective(run_id, victim.stripe_id, 0, 1, None)
        _expect_violation(coord, run_id, coord_cls,
                          "reported no effective threshold", "G10")

    # ...and the wholly-absent case (no sub-stripe reported at all)
    with tempfile.TemporaryDirectory() as tmp:
        coord, run_id, _ds, _res, _sent = _dispatch_assignments(
            tmp, coord_cls=coord_cls)
        _expect_violation(coord, run_id, coord_cls,
                          "NO sub-stripe reported an effective threshold", "G10")


def g11_mismatch(coord_cls=RangeMinerCoordinator):
    """effective != assigned must ABORT the trial — this is the exact condition
    the whole correction exists to make impossible to certify through."""
    with tempfile.TemporaryDirectory() as tmp:
        coord, run_id, sent = _provenance_fixture(tmp, coord_cls)
        coord.validate_threshold_provenance(run_id)
        victim = sent[2][0]                     # a reverse stripe, assigned 0.47
        # CONSISTENTLY wrong: every leg of this stripe reports 0.25. There is no
        # disagreement and the roll-up matches the consensus, so the ONLY thing
        # wrong is that the kernel filtered at a value nobody assigned — which is
        # exactly the condition under test, isolated from the other two.
        coord.record_substripe_effective(
            run_id, victim.stripe_id, 0, 0, LEGACY_FALLBACK)
        coord.record_substripe_effective(
            run_id, victim.stripe_id, 0, 1, LEGACY_FALLBACK)
        coord.record_stripe_complete_effective(
            run_id, victim.stripe_id, 0, LEGACY_FALLBACK)
        _expect_violation(coord, run_id, coord_cls, "but was assigned", "G11")


def g12_substripes_disagree(coord_cls=RangeMinerCoordinator):
    """Sub-stripes of ONE stripe reporting different effective values must ABORT
    the trial — one stripe cannot have filtered at two thresholds."""
    with tempfile.TemporaryDirectory() as tmp:
        coord, run_id, sent = _provenance_fixture(tmp, coord_cls)
        coord.validate_threshold_provenance(run_id)
        victim = sent[3][0]
        coord.record_substripe_effective(run_id, victim.stripe_id, 0, 1, REV)
        _expect_violation(coord, run_id, coord_cls,
                          "DISAGREE on the effective threshold", "G12")


def g12b_complete_disagrees(coord_cls=RangeMinerCoordinator):
    """The stripe-complete roll-up must agree with the sub-stripe consensus."""
    with tempfile.TemporaryDirectory() as tmp:
        coord, run_id, sent = _provenance_fixture(tmp, coord_cls)
        victim = sent[4][0]
        coord.record_stripe_complete_effective(
            run_id, victim.stripe_id, 0, LEGACY_FALLBACK)
        _expect_violation(coord, run_id, coord_cls, "stripe_complete reports",
                          "G12b")


def g13_placement_before_certification(coord_cls=RangeMinerCoordinator):
    """PLACEMENT: the gate runs before commit, and the downstream wall refuses an
    unvalidated run.

    Leg 1 (AST, on the real source): inside `serve_trial`, the
    `validate_threshold_provenance` call precedes the `commit_trial` call — so
    Phase-5 assembly, candidate ingress, accumulator mutation and finalize_run
    are all downstream of it.
    Leg 2 (runtime, real function): `_build_test_result_from_miner` RAISES on a
    miner_result whose provenance is missing or not validated, before touching
    the accumulator."""
    src = inspect.getsource(coord_cls.serve_trial)
    i_val = src.find("validate_threshold_provenance")
    i_commit = src.find("self.commit_trial(")
    assert i_val != -1, (
        "G13: serve_trial does not call validate_threshold_provenance at all — "
        "there is no parent-side fail-closed gate")
    assert i_commit != -1, "G13: serve_trial does not call commit_trial"
    assert i_val < i_commit, (
        "G13: validate_threshold_provenance is called AFTER commit_trial — the "
        "trial would already be committed (and its assembly stored) before the "
        "kernel filter was proven")

    import window_optimizer_integration_final as WOI
    acc = {"forward_count": 0, "reverse_count": 0, "bidirectional": []}
    for label, result in (
        ("no provenance at all", {"run_id": "r", "committed": True}),
        ("validated=False", {"run_id": "r", "committed": True,
                             "threshold_provenance": {"validated": False}}),
        ("validated missing", {"run_id": "r", "committed": True,
                               "threshold_provenance": {"payload": {}}}),
    ):
        try:
            WOI._build_test_result_from_miner(
                result, acc, _Cfg(["midday", "evening"]), "java_lcg", 1,
                phase5_sink=None)
        except Exception as exc:                                # noqa: BLE001
            assert "threshold provenance" in str(exc).lower(), (
                f"G13[{label}]: raised, but not on the provenance wall: "
                f"{type(exc).__name__}: {str(exc)[:200]}")
        else:
            raise AssertionError(
                f"G13[{label}]: candidate ingress ACCEPTED an unvalidated miner "
                f"result — the accumulator would be mutated and the run could "
                f"reach finalize_run")
        assert acc == {"forward_count": 0, "reverse_count": 0,
                       "bidirectional": []}, (
            f"G13[{label}]: the accumulator was mutated before the wall")


# ═════════════════════════════════════════════════════════════════════════════
# R1-R4 — shared-authority residue derivation (Beta §5)
# ═════════════════════════════════════════════════════════════════════════════
SESSION_CASES = (
    ("both sessions", ["midday", "evening"]),
    ("midday only", ["midday"]),
    ("evening only", ["evening"]),
)


def _coordinator_side_residues(config, dataset_path, derive=None):
    """The coordinator-side consumer under test. `derive` lets a mutant stand in
    for the production function."""
    if derive is not None:
        return derive(config, dataset_path)
    import window_optimizer_integration_final as WOI
    return WOI._miner_residues_for_config(config, dataset_path)


class _Cfg:
    def __init__(self, sessions, window_size=WINDOW_SIZE, offset=0):
        self.sessions = list(sessions)
        self.window_size = window_size
        self.offset = offset


def r_identical_residues(derive=None):
    """R1-R3: identical ORDERED residues on both sides, in every session case."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = _dataset(tmp)
        from miner.range_miner_coordinator import compute_dataset_sha256
        ds_sha = compute_dataset_sha256(ds)
        for label, sessions in SESSION_CASES:
            parent = _coordinator_side_residues(_Cfg(sessions), ds, derive)
            resolver = ResidueResolver()
            worker = resolver.resolve({
                "dataset": ds, "dataset_sha256": ds_sha,
                "window_size": WINDOW_SIZE, "sessions": sessions, "offset": 0,
            })
            assert list(parent) == list(worker), (
                f"R[{label}]: coordinator residues {list(parent)!r} != worker "
                f"residues {list(worker)!r} — the session filter is applied on "
                f"only one side")


def r4_assignment_roundtrip(derive=None):
    """R4: the residue_sha256 the parent stamps verifies on the worker in every
    session case — the failure the asymmetry actually produced."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = _dataset(tmp)
        from miner.range_miner_coordinator import compute_dataset_sha256
        ds_sha = compute_dataset_sha256(ds)
        coord = _coord(tmp)
        for label, sessions in SESSION_CASES:
            parent = _coordinator_side_residues(_Cfg(sessions), ds, derive)
            payload = coord.build_stripe_assign_payload(
                ds, WINDOW_SIZE, sessions, 0, parent, dataset_sha256=ds_sha,
                phase=1, forward_threshold=FWD, reverse_threshold=REV)
            resolver = ResidueResolver()
            try:
                resolver.resolve(payload)
            except ResidueVerificationError as e:
                raise AssertionError(
                    f"R4[{label}]: the worker rejected the parent's "
                    f"residue_sha256 ({e}) — every stripe of this trial would "
                    f"fail non-retryably")


# ═════════════════════════════════════════════════════════════════════════════
# Mutation infrastructure (four-part rule)
# ═════════════════════════════════════════════════════════════════════════════
_MUT_DIR = None
_MUT_SEQ = 0


def _mut_dir():
    global _MUT_DIR
    if _MUT_DIR is None:
        _MUT_DIR = tempfile.mkdtemp(prefix="d6_thresh_mutants_")
        sys.path.insert(0, _MUT_DIR)
    return _MUT_DIR


def _patch(src: str, old: str, new: str, label: str) -> str:
    """Part 1: the mutation MUST actually apply, exactly once. A no-op patch
    would let a mutant survive vacuously (a false green)."""
    count = src.count(old)
    assert count == 1, (
        f"{label}: anchor is not unique ({count} occurrences) — the mutation "
        f"would be unverifiable")
    return src.replace(old, new, 1)


def _load_mutant(src: str, label: str):
    global _MUT_SEQ
    _MUT_SEQ += 1
    name = f"_d6t_mutant_{_MUT_SEQ}"
    path = os.path.join(_mut_dir(), f"{name}.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.__d6_label__ = label
    return module


def _executed(module, marker: str, label: str) -> None:
    """Part 2: the mutated text is present in the module that was loaded."""
    src = inspect.getsource(module)
    assert marker in src, (
        f"{label}: the mutated text is absent from the loaded module — the "
        f"mutant did not take effect")


def _positive_control(name, detector):
    """Part 3: the detector must PASS against the UNMUTATED module."""
    try:
        detector()
    except Exception as exc:                                    # noqa: BLE001
        raise AssertionError(
            f"POSITIVE CONTROL FAILED for {name}: the detector reds against the "
            f"UNMUTATED module ({type(exc).__name__}: {exc}) — any kill it "
            f"records would be unattributable") from exc


def _record(label, detector, credited, marker=None, module=None):
    """Parts 3+4: run `detector` and require it to FAIL."""
    if module is not None and marker is not None:
        _executed(module, marker, label)
    try:
        detector()
    except AssertionError as exc:
        sig = str(exc).splitlines()[0][:150] or type(exc).__name__
        _MUTANTS.append((label, f"AssertionError: {sig}", credited,
                         "applies-once ✓ | mutated-path ✓ | detector-clean ✓ | "
                         "injected-defect ✓"))
        return
    except Exception as exc:                                    # noqa: BLE001
        sig = f"{type(exc).__name__}: {str(exc).splitlines()[0][:130]}"
        _MUTANTS.append((label, sig, credited,
                         "applies-once ✓ | mutated-path ✓ | detector-clean ✓ | "
                         "injected-defect ✓"))
        return
    raise AssertionError(f"MUTANT SURVIVED: {label} — {credited} did not red")


def _survives(label, detector, note):
    """The complement of _record: a detector that must PASS on this mutant,
    proving another detector is genuinely load-bearing."""
    try:
        detector()
    except Exception as exc:                                    # noqa: BLE001
        raise AssertionError(
            f"{label}: {note} — but the detector RED ({type(exc).__name__}: "
            f"{exc}), so the claim it cannot catch this mutant is wrong")


# --- the three required threshold mutants -----------------------------------
_ANCHOR_RESOLVE = (
    "        if direction == \"forward\":\n"
    "            resolved_threshold = float(forward_threshold)\n"
    "        elif direction == \"reverse\":\n"
    "            resolved_threshold = float(reverse_threshold)\n"
)

_ANCHOR_EMIT = (
    "            \"min_match_threshold\": resolved_threshold,   # constant kernels\n"
    "            \"phase2_threshold\": resolved_threshold,      # hybrid kernels\n"
)

_ANCHOR_ASSERT = (
    "        if payload[\"min_match_threshold\"] != payload[\"phase2_threshold\"]:"
)


def _m_drop():
    """M-drop: the payload carries no threshold field — the pre-correction
    behaviour, which sends the worker back to its hardcoded 0.25."""
    label = "M-drop payload emits no threshold field"
    src = _patch(_COORD_SRC, _ANCHOR_EMIT,
                 "            # MUTANT: threshold fields dropped from the payload\n",
                 label)
    src = _patch(src, _ANCHOR_ASSERT,
                 "        if False:  # MUTANT: contract assert removed", label)
    return _load_mutant(src, label), label


def _m_collapse():
    """M-collapse: forward_threshold applied to BOTH directions — per-direction
    tuning silently lost."""
    label = "M-collapse forward threshold applied to both directions"
    src = _patch(
        _COORD_SRC, _ANCHOR_RESOLVE,
        "        if direction == \"forward\":\n"
        "            resolved_threshold = float(forward_threshold)\n"
        "        elif direction == \"reverse\":\n"
        "            resolved_threshold = float(forward_threshold)  "
        "# MUTANT: collapsed to forward\n",
        label)
    return _load_mutant(src, label), label


def _m_swap():
    """M-swap: forward and reverse exchanged. Two consistently-reversed branches
    still look asymmetric, so the not-collapsed check passes — only the explicit
    direction-identity check (G6) catches this."""
    label = "M-swap forward/reverse thresholds exchanged"
    src = _patch(
        _COORD_SRC, _ANCHOR_RESOLVE,
        "        if direction == \"forward\":\n"
        "            resolved_threshold = float(reverse_threshold)  # MUTANT: swapped\n"
        "        elif direction == \"reverse\":\n"
        "            resolved_threshold = float(forward_threshold)  # MUTANT: swapped\n",
        label)
    return _load_mutant(src, label), label


# --- the residue mutant ------------------------------------------------------
def _func_src(src: str, path: str, name: str) -> str:
    tree = ast.parse(src, filename=path)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise AssertionError(f"{name} not found in {path}")


def _m_residue_derive():
    """M-residue: the coordinator-side consumer stops passing the session
    selection — precisely the pre-D6 asymmetry (`_get_residues_for_config` never
    passed `sessions`). Built by patching the REAL function's source."""
    label = "M-residue coordinator side drops the session filter"
    fsrc = _func_src(_INTEG_SRC, _INTEG_PATH, "_miner_residues_for_config")
    mutated = _patch(
        fsrc,
        "    return load_residue_window(dataset_path, config.window_size,\n"
        "                               config.sessions, config.offset)",
        "    return load_residue_window(dataset_path, config.window_size,\n"
        "                               None, config.offset)  "
        "# MUTANT: sessions dropped",
        label)
    ns: Dict[str, Any] = {}
    exec(compile(mutated, "<m-residue>", "exec"), ns)          # noqa: S102
    assert "# MUTANT: sessions dropped" in mutated, "mutant text absent"
    return ns["_miner_residues_for_config"], label, mutated


def run_mutants():
    # ---- M-drop --------------------------------------------------------
    _positive_control("G8", lambda: g8_no_silent_fallback())
    _positive_control("G1", lambda: g1_forward_payload())
    mod, label = _m_drop()
    _record(label, lambda: g8_no_silent_fallback(mod.RangeMinerCoordinator),
            "G8", "MUTANT: threshold fields dropped from the payload", mod)
    _record(label + " (also caught by G1)",
            lambda: g1_forward_payload(mod.RangeMinerCoordinator), "G1")

    # ---- M-collapse ----------------------------------------------------
    _positive_control("G2", lambda: g2_reverse_payload())
    _positive_control("G5", lambda: g5_not_collapsed())
    _positive_control("G6", lambda: g6_not_swapped())
    mod, label = _m_collapse()
    _record(label, lambda: g2_reverse_payload(mod.RangeMinerCoordinator),
            "G2", "# MUTANT: collapsed to forward", mod)
    _record(label + " (also caught by G5/G6)",
            lambda: g5_not_collapsed(mod.RangeMinerCoordinator), "G5")

    # ---- M-swap --------------------------------------------------------
    mod, label = _m_swap()
    # The reason this mutant has to exist in its own right: the not-collapsed
    # detector CANNOT see it.
    _survives(label, lambda: g5_not_collapsed(mod.RangeMinerCoordinator),
              "a swap keeps forward != reverse, so G5 is blind to it")
    _record(label, lambda: g6_not_swapped(mod.RangeMinerCoordinator),
            "G6", "# MUTANT: swapped", mod)

    # ---- the parent-side enforcement mutants (Beta's commit ruling) -----
    _positive_control("G10", lambda: g10_missing_effective())
    _positive_control("G11", lambda: g11_mismatch())
    _positive_control("G12", lambda: g12_substripes_disagree())
    _positive_control("G12b", lambda: g12b_complete_disagrees())
    _positive_control("G13", lambda: g13_placement_before_certification())

    # M-prov-missing: the parent stops requiring an effective value at all, so a
    # D6-generated assignment with absent provenance sails through to commit.
    label = "M-prov-missing absent effective value accepted"
    src = _patch(_COORD_SRC,
                 "            subs = rec[\"sub_effective\"]\n"
                 "            if not subs:\n",
                 "            subs = rec[\"sub_effective\"]\n"
                 "            if False:  # MUTANT: wholly-absent provenance accepted\n",
                 label)
    src = _patch(src,
                 "            missing = sorted(i for i, v in subs.items() if v is None)\n"
                 "            if missing:\n",
                 "            missing = sorted(i for i, v in subs.items() if v is None)\n"
                 "            if False:  # MUTANT: missing effective value accepted\n",
                 label)
    mod = _load_mutant(src, label)
    _record(label, lambda: g10_missing_effective(mod.RangeMinerCoordinator),
            "G10", "# MUTANT: missing effective value accepted", mod)

    # M-prov-mismatch: the parent stops comparing effective against assigned, so
    # a kernel that filtered at a value nobody requested is certified.
    label = "M-prov-mismatch assigned/effective mismatch accepted"
    src = _patch(_COORD_SRC,
                 "            for idx, value in sorted(subs.items()):\n"
                 "                if value is not None and value != assigned:\n",
                 "            for idx, value in sorted(subs.items()):\n"
                 "                if False:  # MUTANT: sub-stripe mismatch accepted\n",
                 label)
    src = _patch(src,
                 "                elif complete != assigned:\n",
                 "                elif False:  # MUTANT: roll-up mismatch accepted\n",
                 label)
    mod = _load_mutant(src, label)
    _record(label, lambda: g11_mismatch(mod.RangeMinerCoordinator),
            "G11", "# MUTANT: sub-stripe mismatch accepted", mod)

    # M-prov-disagree: the parent stops checking that a stripe's sub-stripes
    # agree. NOTE, stated precisely: the residual assigned-vs-effective check
    # still aborts this particular trial (defence in depth), so what this mutant
    # destroys is the CONDITION'S OWN detector — a stripe that demonstrably
    # filtered at two different thresholds is no longer identified as such. G12
    # requires the disagreement to be named, which is why it catches this.
    label = "M-prov-disagree sub-stripe disagreement no longer detected"
    src = _patch(_COORD_SRC,
                 "            distinct = sorted(set(present))\n"
                 "            if len(distinct) > 1:\n",
                 "            distinct = sorted(set(present))\n"
                 "            if False:  # MUTANT: disagreeing sub-stripes accepted\n",
                 label)
    mod = _load_mutant(src, label)
    _record(label, lambda: g12_substripes_disagree(mod.RangeMinerCoordinator),
            "G12", "# MUTANT: disagreeing sub-stripes accepted", mod)

    # M-prov-nogate: the enforcement call is removed from serve_trial entirely,
    # so a violating trial reaches commit_trial and certification.
    label = "M-prov-nogate enforcement removed from serve_trial"
    src = _patch(_COORD_SRC,
                 "                                self.validate_threshold_provenance(run_id)\n",
                 "                                pass  # MUTANT: provenance gate removed\n",
                 label)
    mod = _load_mutant(src, label)
    _record(label,
            lambda: g13_placement_before_certification(mod.RangeMinerCoordinator),
            "G13", "# MUTANT: provenance gate removed", mod)

    # ---- M-residue -----------------------------------------------------
    _positive_control("R1-R3", lambda: r_identical_residues())
    _positive_control("R4", lambda: r4_assignment_roundtrip())
    derive, label, mutated = _m_residue_derive()
    assert "# MUTANT: sessions dropped" in mutated, (
        f"{label}: mutated path not present")
    _record(label, lambda: r_identical_residues(derive), "R1-R3")
    _record(label + " (also caught by R4)",
            lambda: r4_assignment_roundtrip(derive), "R4")


# ═════════════════════════════════════════════════════════════════════════════
def main() -> int:
    print("=" * 78)
    print("S172 Phase 5 D6 CORRECTION — threshold path + residue shared authority")
    print(f"asymmetric fixture: forward={FWD}  reverse={REV}  "
          f"(legacy fallback {LEGACY_FALLBACK})")
    print("=" * 78)

    _check("G1: forward assignments (phases 1,3) carry exactly 0.31",
           g1_forward_payload)
    _check("G2: reverse assignments (phases 2,4) carry exactly 0.47",
           g2_reverse_payload)
    _check("G3: the worker receives each exact value over the real framing",
           g3_worker_receives)
    _check("G4: the real executor launches the kernel with each exact value",
           g4_kernel_receives)
    _check("G5: forward/reverse not collapsed", g5_not_collapsed)
    _check("G6: forward/reverse not swapped", g6_not_swapped)
    _check("G7: a LEGACY payload still resolves to 0.25 (back-compat)",
           g7_legacy_fallback)
    _check("G8: every new D6 payload carries an explicit threshold",
           g8_no_silent_fallback)
    _check("G9: provenance — requested == payload == effective",
           g9_provenance)
    _check("G10: parent ABORTS on an absent effective threshold",
           g10_missing_effective)
    _check("G11: parent ABORTS on assigned != effective", g11_mismatch)
    _check("G12: parent ABORTS on disagreeing sub-stripes",
           g12_substripes_disagree)
    _check("G12b: parent ABORTS when stripe_complete disagrees with consensus",
           g12b_complete_disagrees)
    _check("G13: gate runs before commit; ingress refuses an unvalidated run",
           g13_placement_before_certification)
    _check("R1-R3: identical ordered residues both sides "
           "(both / midday / evening)", r_identical_residues)
    _check("R4: parent residue_sha256 verifies on the worker in every case",
           r4_assignment_roundtrip)

    print("\n" + "-" * 78)
    print("MUTATION PROOF (four-part rule)")
    print("-" * 78)
    _check("MUTANTS: every injected defect is caught", run_mutants)
    for label, sig, credited, parts in _MUTANTS:
        print(f"  [{_PASS}] {label}\n         killed by {credited} — {sig}\n"
              f"         {parts}")

    passed = sum(1 for _n, ok, _t in _results if ok)
    total = len(_results)
    print("\n" + "=" * 78)
    print(f"{passed}/{total} D6 threshold-path checks green "
          f"({len(_MUTANTS)} mutants killed)")
    print("=" * 78)
    if passed != total:
        print("FAILURES (DO NOT COMMIT):\n")
        for name, ok, tb in _results:
            if not ok:
                print(f"--- {name} ---\n{tb}")
        return 1
    print("The Optuna-tuned per-direction sieve threshold now reaches the kernel "
          "unchanged through ONE chokepoint, the effective value is recorded "
          "off the real executor, and the coordinator/worker residue derivation "
          "is a single shared function (pending Team Alpha + Team Beta review).")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        if _MUT_DIR is not None:
            import shutil
            shutil.rmtree(_MUT_DIR, ignore_errors=True)
