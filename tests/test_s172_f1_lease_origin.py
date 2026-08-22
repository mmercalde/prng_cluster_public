#!/usr/bin/env python3
"""S172 F1 — LEASE-ORIGIN REPAIR.  Gates L1-L7 (Beta ruling 2026-08-11).

FALSIFIABLE QUESTION
  Does a newly claimed compute stripe receive its FULL configured compute lease
  measured from the ACTUAL claim operation, regardless of how old the enclosing
  serve-loop iteration is — and did fixing that leave every non-lease consumer of
  the loop clock untouched?

THE DEFECT (gate-12 attempt 4, `distributed_config_t1_c8939b64`, 2026-08-11)
  `serve_trial` captures `now = time.time()` ONCE per iteration and passed that
  same value to `schedule_pending_stripes(..., now=now)`, which resolved it once
  at entry and stamped `now + compute_lease_timeout` into every `claim_stripe` of
  the pass. Measured from the frozen evidence bundle, independently of the brief:

    * stage 2 opened 19:15:42.900 with `claimed=25 queued=7` ([S172-BP]
      burst_exact, `gate12_20260811_190414.log:150`);
    * the concurrency sampler brackets the claim of the last two backlog
      stripes between 19:21:45.667 (`queued_pending=2, compute_active=4`) and
      19:21:47.676 (`queued_pending=0, compute_active=6`, with `rrig6600:gpu1`
      and `zeus-ubuntu-vm:gpu0` newly active);
    * both rows carry `lease_expires_at = 19:22:13.373838` — IDENTICAL to the
      microsecond, which is only possible from one shared origin — i.e.
      19:17:13.373838 + 300.0.

  So the iteration's clock was **272.3 s** old when the scheduler ran, and
  `st1_s30`/`st1_s31` were born with **~26 s of a nominal 300 s lease, ~91% of
  the budget already consumed.** Both produced zero shards; the constant-mode
  matrix then failed the trial, correctly, on a lease that had never been granted.
  `272.3` is where L1's "~270 s ageing" comes from — it is a measurement, not a
  round number.

THE TWO HALVES OF THE REPAIR ARE GATED SEPARATELY, AND BOTH ARE LOAD-BEARING
  (a) `schedule_pending_stripes` reads its clock immediately before EACH
      `claim_stripe` when no clock was injected; (b) production stops injecting
      the enclosing-loop `now`. L6 runs each half alone and shows the surviving
      half is insufficient — a suite that only tested them together could not
      tell which one carried the fix.

RED-ARM PROVENANCE (the discipline Beta required in the clean-tree R1 correction;
`tests/test_gate12_cleantree_admission.py` is the worked example)
  Every RED arm runs the COMMITTED PRE-REPAIR SOURCE, pinned to the IMMUTABLE
  commit 213bfff — never `HEAD`, which becomes the repaired source the instant
  this work is committed, and never a retyped copy. `_pinned_scheduler()` refuses
  to return the pinned object unless it still carries BOTH defect surfaces (the
  once-per-pass clock resolution AND the `now +`-stamped claim), so a wrong or
  drifted anchor reports UNAVAILABLE instead of crediting a RED arm.

DETERMINISM
  No sleeps and no wall-clock races: the module clock is replaced by a scripted
  `_Clock` for every arm, so "the loop timestamp aged 272.3 s" is arithmetic. The
  instrumentation gates additionally script `perf_counter` (`_MonoClock`), so
  "the terminal iteration was the long one" is arithmetic too.

  [R1.5, Beta 2026-08-12] THE READ-COUNT CLAIM, STATED PRECISELY. An earlier
  revision said `time.time()` is called inside `schedule_pending_stripes` "exactly
  once per SUCCESSFUL claim and never otherwise". That is too strong: the read
  happens immediately BEFORE `ledger.claim_stripe`, so a `False` return from a
  concurrent state change or a terminal guard has still consumed a clock read.
  The accurate statement, and the one the gates rely on:

      ONE fresh clock read per CLAIM ATTEMPT REACHING `claim_stripe`; every
      successfully created lease uses the immediately preceding fresh value.

  This is harmless to the repair — an over-count can never manufacture a stale
  origin — and no credited assertion depended on the stronger form: L1-GREEN and
  L2 count reads on passes where every attempt succeeds, so attempts and
  successes coincide there and the arithmetic is identical under either wording.
  The read count remains an execution proof (VIR-1) because the ledger claim path
  contains no clock read of its own.

Run:  source ~/venvs/torch/bin/activate && \
      python3 -u tests/test_s172_f1_lease_origin.py | tee /tmp/f1_lease_origin.log
"""
import ast
import logging
import os
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time as time_module
import traceback
import types
from typing import Any, Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"
_UNAV = "\033[91mUNAV\033[0m"
_results: List[Any] = []


def _check(name, fn):
    try:
        fn()
        _results.append((name, True, None))
        print(f"  [{_PASS}] {name}", flush=True)
    except Exception as e:                                       # noqa: BLE001
        _results.append((name, False, traceback.format_exc()))
        print(f"  [{_FAIL}] {name}: {e}", flush=True)


def _unavailable(name, reason):
    """VIR-3: UNAVAILABLE is a distinct terminal state and never accepts."""
    _results.append((name, False, f"UNAVAILABLE: {reason}"))
    print(f"  [{_UNAV}] {name}: UNAVAILABLE: {reason}", flush=True)


from miner.range_miner_coordinator import (  # noqa: E402
    ST_CLAIMED,
    ST_PENDING,
    ST_STAGING,
    CoordinatorConfig,
    LeaseInvariantError,
    MinerLedger,
    NodeConfig,
    RangeMinerCoordinator,
    ServeLoopTiming,
    TC_COMPUTE_LEASE_EXPIRY,
    TC_WORKER_ADMISSION_TIMEOUT,
    run_trial_miner,
)
import miner.range_miner_coordinator as COORD  # noqa: E402

# --- the attempt-4 constants, all derived above from the frozen bundle --------
LEASE = 300.0
MACRO = 1000
STALE_AGE = 272.3          # measured: 19:21:45.667 - 19:17:13.373838
CAPS = {"amd": 2_000_000, "nvidia": 5_000_000,
        "amd_hybrid": 1_000_000, "nvidia_hybrid": 2_500_000}
VARIANTS = ["java_lcg", "java_lcg_reverse", "java_lcg_hybrid",
            "java_lcg_hybrid_reverse"]
SPOOL_ROOT = "/var/spool/miner"

# [R1.4, Beta 2026-08-12] THE FULL 40-CHARACTER SHA, never the abbreviation. A
# short SHA fails CLOSED if it later becomes ambiguous (`git show` errors, the
# anchor reports UNAVAILABLE, no RED arm is credited), so this is a durability
# correction rather than a false-PASS repair — but a permanent governance anchor
# must not be abbreviated. Same discipline as the clean-tree admission repair.
PINNED_COMMIT = "213bffff512f0e360c40974cbfc9e787c5b005f0"
SRC_REL = "miner/range_miner_coordinator.py"


# ===========================================================================
# fixtures
# ===========================================================================
def _coord(tmp, **cfg):
    cfg.setdefault("miner_stripe_size", MACRO)
    cfg.setdefault("compute_lease_timeout", LEASE)
    cfg.setdefault("staging_dir", os.path.join(tmp, "staging"))
    ledger = MinerLedger(os.path.join(tmp, "l.db"))
    return RangeMinerCoordinator(CoordinatorConfig(**cfg), ledger)


def _register(coord, wid, backend="cuda", variants=None, now=100.0):
    node = NodeConfig(hostname=wid.split(":")[0], spool_root=SPOOL_ROOT,
                      ssh_address="10.0.0.9", ssh_user="michael")
    return coord.register_worker(
        worker_id=wid, hostname=wid.split(":")[0], backend=backend,
        capabilities={"seed_caps": dict(CAPS),
                      "supported_variants": list(variants or VARIANTS)},
        node_config=node, now=now)


class _Clock:
    """A scripted stand-in for the `time` MODULE.

    Only `time()` is scripted; everything else (`perf_counter`, `monotonic`,
    `sleep`, …) delegates to the real module, which is what keeps the
    instrumentation — deliberately built on `perf_counter` — independent of the
    clock these gates control. `reads` is the execution proof: a claim that reads
    no clock did not take the repaired path."""

    def __init__(self, value: float, step: float = 0.0):
        self.value = float(value)
        self.step = float(step)
        self.reads: List[float] = []
        import time as _real
        self._real = _real

    def time(self) -> float:
        v = self.value
        self.reads.append(v)
        self.value += self.step
        return v

    def __getattr__(self, item):
        return getattr(self._real, item)


class _MonoClock:
    """[R1] A scripted stand-in for the `time` MODULE that scripts BOTH clocks.

    `perf_counter()` returns the next value of a fixed script and **raises when
    the script is exhausted** — so an unexpected extra monotonic read is a loud
    failure rather than a silently absorbed one, and `values` being empty at the
    end of a gate is an exact-consumption execution proof (VIR-1).

    `time()` returns `wall`, which the gate may step arbitrarily, and counts its
    reads. `wall_reads == 0` is the credited proof for R1.3: the instrument does
    not touch the wall clock at all, so no system-clock step can move a duration
    it reports. Everything else delegates to the real module."""

    def __init__(self, mono_values, wall: float = 1_786_500_000.0):
        self.values: List[float] = [float(v) for v in mono_values]
        self.reads: List[float] = []
        self.wall = float(wall)
        self.wall_reads = 0
        import time as _real
        self._real = _real

    def perf_counter(self) -> float:
        if not self.values:
            raise AssertionError(
                "scripted perf_counter exhausted — the instrument took more "
                "monotonic readings than the scenario scripts, so the arithmetic "
                "below would not be the arithmetic under test")
        v = self.values.pop(0)
        self.reads.append(v)
        return v

    def time(self) -> float:
        self.wall_reads += 1
        return self.wall

    def __getattr__(self, item):
        return getattr(self._real, item)


class _patched_clock:
    """Install `_Clock` / `_MonoClock` as the coordinator module's `time`."""

    def __init__(self, clock):
        self.clock = clock

    def __enter__(self):
        self._saved = COORD.time
        COORD.time = self.clock
        return self.clock

    def __exit__(self, *exc):
        COORD.time = self._saved
        return False


def _mutant_red(fn, label):
    """VIR-2 positive control: `fn` MUST raise. A mutation the gate cannot detect
    makes the gate vacuous, so each gate proves its own detection power."""
    try:
        fn()
    except Exception:                                            # noqa: BLE001
        return
    raise AssertionError(
        f"MUTANT SURVIVED ({label}) — the gate did not detect it, so it is "
        f"vacuous and proves nothing")


# ===========================================================================
# [R1] the pinned pre-repair anchor, and its integrity
# ===========================================================================
class AnchorUnavailable(RuntimeError):
    """The pinned pre-repair source could not be obtained, or is not it."""


def _git_show(commit: str, path: str) -> str:
    p = subprocess.run(["git", "-C", _ROOT, "show", f"{commit}:{path}"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise AnchorUnavailable(
            f"{commit}:{path} does not resolve: {p.stderr.strip()}")
    return p.stdout


def _func_node(tree: ast.AST, cls: Optional[str], name: str) -> ast.FunctionDef:
    scope = tree
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


def _entry_clock_resolutions(fn: ast.FunctionDef) -> int:
    """Count `now = time.time() if now is None else now` at the function's TOP
    level — the pre-repair once-per-pass resolution."""
    n = 0
    for stmt in fn.body:
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == "now"
                and isinstance(stmt.value, ast.IfExp)):
            n += 1
    return n


def _claim_lease_origin(fn: ast.FunctionDef) -> str:
    """The identifier the lease expiry is computed from, at the `claim_stripe`
    call. Pre-repair this is `now`; post-repair it must be `claim_now`."""
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "claim_stripe"):
            for arg in node.args:
                if (isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add)
                        and isinstance(arg.left, ast.Name)):
                    return arg.left.id
    raise AnchorUnavailable(
        "no `<name> + <lease>` argument found at the claim_stripe call")


def _serve_now_kw_callees(fn: ast.FunctionDef) -> List[str]:
    """Every call inside `fn` that passes the keyword `now=now`, by callee name,
    in source order. This IS the six-site audit, computed rather than asserted."""
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if (kw.arg == "now" and isinstance(kw.value, ast.Name)
                        and kw.value.id == "now"):
                    name = (node.func.attr if isinstance(node.func, ast.Attribute)
                            else getattr(node.func, "id", "?"))
                    out.append((node.lineno, name))
    return [n for _, n in sorted(out)]


_PINNED_CACHE: Dict[str, Any] = {}


def _pinned_module_src() -> str:
    if "src" not in _PINNED_CACHE:
        _PINNED_CACHE["src"] = _git_show(PINNED_COMMIT, SRC_REL)
    return _PINNED_CACHE["src"]


def _pinned_scheduler():
    """The COMMITTED PRE-REPAIR `schedule_pending_stripes`, pinned and
    integrity-checked, returned as a callable ready to bind to a coordinator.

    REFUSES unless the pinned object still carries BOTH defect surfaces. Nothing
    downstream can credit a RED arm from a source that does not carry the defect
    the RED arm exists to demonstrate."""
    src = _pinned_module_src()
    tree = ast.parse(src)
    fn = _func_node(tree, "RangeMinerCoordinator", "schedule_pending_stripes")
    n_entry = _entry_clock_resolutions(fn)
    if n_entry != 1:
        raise AnchorUnavailable(
            f"pinned {PINNED_COMMIT} does not resolve `now` exactly once at "
            f"entry to schedule_pending_stripes (found {n_entry}) — this is not "
            f"the pre-repair source")
    origin = _claim_lease_origin(fn)
    if origin != "now":
        raise AnchorUnavailable(
            f"pinned {PINNED_COMMIT} stamps the lease from {origin!r}, not the "
            f"pass-wide `now` — this is not the pre-repair source")
    if "claim_now" in src:
        raise AnchorUnavailable(
            f"pinned {PINNED_COMMIT} already contains `claim_now` — the anchor "
            f"points at repaired source")
    serve = _func_node(tree, "RangeMinerCoordinator", "serve_trial")
    if "schedule_pending_stripes" not in _serve_now_kw_callees(serve):
        raise AnchorUnavailable(
            f"pinned {PINNED_COMMIT} serve_trial does not inject `now=now` into "
            f"schedule_pending_stripes — the invocation half of the defect is "
            f"absent, so the anchor is wrong")
    body = textwrap.dedent(ast.get_source_segment(src, fn))
    g = dict(vars(COORD))          # real module globals: time, helpers, errors
    exec(compile(body, f"<pinned {PINNED_COMMIT}>", "exec"), g)   # noqa: S102
    return g["schedule_pending_stripes"], g


def _bind_pinned(coord, clock):
    """Bind the pinned pre-repair scheduler to `coord`, with its OWN view of the
    clock, and return the bound method. The pinned copy resolves `time` in the
    globals dict built above, so its clock is replaced there, not by patching the
    live module — the two arms therefore cannot contaminate each other."""
    fn, g = _pinned_scheduler()
    g["time"] = clock
    return types.MethodType(fn, coord)


# ===========================================================================
# shared scenario: one stage, one worker, one stripe held in the backlog
# ===========================================================================
def _late_backlog(coord, loop_t0, n_stripes=2, n_workers=1, fam="java_lcg",
                  phase=1):
    """Open a stage at `loop_t0`, complete the claimed stripe(s) so the worker(s)
    go compute-idle, and leave the rest PENDING. Returns (conns, pending_ids)."""
    conns = [_register(coord, f"host{i}:gpu0") for i in range(n_workers)]
    coord.ledger.create_trial("run", 1, now=loop_t0)
    assigns = coord.assign_stripes(
        "run", fam, phase, MACRO * n_stripes, conns,
        stripe_prefix="run__st0", now=loop_t0)
    for a in assigns:
        if a["claimed"]:
            coord.ledger.record_stripe_complete(
                "run", a["stripe_id"], 0, a["worker_id"], 1, 0)
    assert coord.ledger.compute_busy_worker_ids("run") == set()
    pending = [a["stripe_id"] for a in assigns if not a["claimed"]]
    assert pending, "fixture produced no backlog"
    return conns, pending


# ===========================================================================
# L1 — the exact attempt-4 stale-time mutant, both arms
# ===========================================================================
def l1_red_pre_repair():
    """RED: the PINNED PRE-REPAIR scheduler, invoked exactly as production
    invoked it (`now=<loop clock>`), on a stripe claimed 272.3 s into the
    iteration, reproduces attempt 4's ~26 s lease."""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        T0 = 1_786_500_000.0
        clock = _Clock(T0)                      # stage setup happens at T0
        with _patched_clock(clock):
            conns, pending = _late_backlog(coord, T0)
            claim_at = T0 + STALE_AGE
            clock.value = claim_at
            before = len(clock.reads)
            sched = _bind_pinned(coord, clock)
            placed = sched("run", "java_lcg", 1, conns,
                           stage_prefix="run__st0", now=T0)
        assert len(placed) == 1, placed
        st = coord.ledger.get_stripe("run", placed[0]["stripe_id"])
        lease = st["lease_expires_at"]
        assert lease == T0 + LEASE, (
            f"pinned pre-repair source stamped {lease}, expected the stale "
            f"pass-wide origin {T0 + LEASE}")
        remaining = lease - claim_at
        assert abs(remaining - (LEASE - STALE_AGE)) < 1e-6, remaining
        assert 20.0 < remaining < 35.0, (
            f"the RED arm must reproduce attempt 4's ~26 s residue, got "
            f"{remaining:.3f}s")
        consumed = 100.0 * (1.0 - remaining / LEASE)
        assert consumed > 90.0, consumed
        # VIR-1: the pinned copy read NO clock (its `now` was injected), which is
        # precisely the defect — the claim never consulted the wall at all.
        assert len(clock.reads) == before, (
            "the pinned pre-repair scheduler read the clock; it must not")


def l1_green_repaired():
    """GREEN: the LIVE scheduler, invoked exactly as production now invokes it
    (no `now=`), gives the same late stripe its FULL 300 s from the actual
    claim."""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        T0 = 1_786_500_000.0
        clock = _Clock(T0)
        with _patched_clock(clock):
            conns, pending = _late_backlog(coord, T0)
            claim_at = T0 + STALE_AGE
            clock.value = claim_at
            before = len(clock.reads)
            placed = coord.schedule_pending_stripes(
                "run", "java_lcg", 1, conns, stage_prefix="run__st0")
        assert len(placed) == 1, placed
        st = coord.ledger.get_stripe("run", placed[0]["stripe_id"])
        lease = st["lease_expires_at"]
        assert lease == claim_at + LEASE, (
            f"lease {lease} is not claim_at+{LEASE} ({claim_at + LEASE}); a "
            f"stale pass-wide origin would give {T0 + LEASE}")
        assert lease - claim_at == LEASE
        # VIR-1 EXECUTION PROOF: exactly one clock read, at the claim.
        assert len(clock.reads) - before == 1, (
            f"expected exactly one clock read (one claim), got "
            f"{len(clock.reads) - before}")
        assert clock.reads[-1] == claim_at


# ===========================================================================
# L2 — multiple late claims, each from its OWN claim time
# ===========================================================================
def l2_multiple_late_claims():
    """Two late pending stripes handed to two newly idle workers in ONE pass get
    two DIFFERENT origins — not a shared historical timestamp. Attempt 4's two
    stripes carried leases identical to the microsecond; that is the observable
    this gate makes impossible."""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        T0 = 1_786_500_000.0
        STEP = 7.0
        clock = _Clock(T0)
        with _patched_clock(clock):
            conns, pending = _late_backlog(coord, T0, n_stripes=4, n_workers=2)
            assert len(pending) == 2 and len(conns) == 2
            claim_at = T0 + STALE_AGE
            clock.value, clock.step = claim_at, STEP
            placed = coord.schedule_pending_stripes(
                "run", "java_lcg", 1, conns, stage_prefix="run__st0")
        assert len(placed) == 2, placed
        leases = [coord.ledger.get_stripe("run", p["stripe_id"])["lease_expires_at"]
                  for p in placed]
        assert len(set(leases)) == 2, (
            f"both late claims share one lease origin {leases} — this is exactly "
            f"attempt 4's identical-to-the-microsecond pair")
        assert sorted(leases) == [claim_at + LEASE, claim_at + STEP + LEASE], leases
        # each got a FULL lease from its own claim
        for lease, read in zip(sorted(leases), [claim_at, claim_at + STEP]):
            assert lease - read == LEASE, (lease, read)
        assert set(p["worker_id"] for p in placed) == {c.worker_id for c in conns}


# ===========================================================================
# L3 — no premature constant failure
# ===========================================================================
def _late_claim(coord, clock, T0, fam="java_lcg", phase=1, n_workers=1,
                n_stripes=2):
    conns, pending = _late_backlog(coord, T0, n_stripes=n_stripes,
                                   n_workers=n_workers, fam=fam, phase=phase)
    claim_at = T0 + STALE_AGE
    clock.value = claim_at
    placed = coord.schedule_pending_stripes(
        "run", fam, phase, conns, stage_prefix="run__st0")
    assert placed, "nothing placed"
    return conns, placed, claim_at


def l3_no_premature_expiry():
    """A CONSTANT stripe claimed late must not enter compute_lease_expiry before
    its full lease interval — and the RED arm shows the pre-repair origin failing
    the trial 270 s early."""
    # GREEN: 1 s short of the full lease, nothing expires.
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        T0 = 1_786_500_000.0
        clock = _Clock(T0)
        with _patched_clock(clock):
            conns, placed, claim_at = _late_claim(coord, clock, T0)
            out = coord.process_lease_expiry("run", conns,
                                             now=claim_at + LEASE - 1.0)
        assert out == [], f"premature expiry {out}"
        sid = placed[0]["stripe_id"]
        assert coord.ledger.get_stripe("run", sid)["state"] == ST_CLAIMED
        assert coord.ledger.get_trial("run")["state"] not in ("aborted",
                                                             "committed")
    # RED: the pinned pre-repair origin, checked at the SAME instant the GREEN
    # arm calls healthy — 30 s after the claim — already terminates the trial.
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        T0 = 1_786_500_000.0
        clock = _Clock(T0)
        with _patched_clock(clock):
            conns, pending = _late_backlog(coord, T0)
            claim_at = T0 + STALE_AGE
            clock.value = claim_at
            sched = _bind_pinned(coord, clock)
            placed = sched("run", "java_lcg", 1, conns,
                           stage_prefix="run__st0", now=T0)
            assert placed
            out = coord.process_lease_expiry("run", conns, now=claim_at + 30.0)
        assert out, (
            "the pre-repair arm did NOT expire 30 s into a 300 s lease — the RED "
            "arm is vacuous")
        assert coord.ledger.get_trial("run")["state"] == "aborted"


# ===========================================================================
# L4 — genuine expiry preserved (both phase policies unchanged)
# ===========================================================================
def l4_genuine_expiry_constant():
    """A genuinely silent CONSTANT assignment that consumes the whole refreshed
    lease still fails immediately, with the same terminal class."""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        T0 = 1_786_500_000.0
        clock = _Clock(T0)
        with _patched_clock(clock):
            conns, placed, claim_at = _late_claim(coord, clock, T0)
            out = coord.process_lease_expiry("run", conns,
                                             now=claim_at + LEASE + 1.0)
        assert len(out) == 1, out
        assert out[0]["action"] == "fail_trial", out
        trial = coord.ledger.get_trial("run")
        assert trial["state"] == "aborted", trial["state"]
        assert trial["terminal_class"] == TC_COMPUTE_LEASE_EXPIRY, dict(trial)


def l4_genuine_expiry_hybrid_retry():
    """Hybrid first-expiry semantics are UNCHANGED: one reassignment to a
    different worker, phase_degraded, never an immediate trial failure."""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        T0 = 1_786_500_000.0
        clock = _Clock(T0)
        with _patched_clock(clock):
            conns, placed, claim_at = _late_claim(
                coord, clock, T0, fam="java_lcg_hybrid", phase=3,
                n_workers=2, n_stripes=3)
            out = coord.process_lease_expiry("run", conns,
                                             now=claim_at + LEASE + 1.0)
        assert out, out
        actions = {o["action"] for o in out}
        assert "fail_trial" not in actions, out
        assert actions <= {"reassigned", "requeued"}, out
        assert coord.ledger.get_trial("run")["state"] not in ("aborted",
                                                              "committed")
        for o in out:
            assert o["phase_degraded"] is True, o
            assert o["attempt"] == 1, o


# ===========================================================================
# L5 — the one-active-claim invariant is unchanged
# ===========================================================================
def l5_one_active_invariant():
    """No bulk claiming and no two compute-active claims per worker: one idle
    worker takes exactly ONE stripe per pass however deep the backlog, and the
    ledger primitive still RAISES on a second."""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        T0 = 1_786_500_000.0
        clock = _Clock(T0)
        with _patched_clock(clock):
            conns, pending = _late_backlog(coord, T0, n_stripes=5, n_workers=1)
            assert len(pending) == 4
            clock.value = T0 + STALE_AGE
            placed = coord.schedule_pending_stripes(
                "run", "java_lcg", 1, conns, stage_prefix="run__st0")
            assert len(placed) == 1, f"bulk claim restored: {placed}"
            # a second pass with the worker still busy claims nothing more
            again = coord.schedule_pending_stripes(
                "run", "java_lcg", 1, conns, stage_prefix="run__st0")
            assert again == [], again
        states = [s["state"] for s in coord.ledger.all_stripes("run")]
        assert states.count(ST_CLAIMED) == 1, states
        assert states.count(ST_PENDING) == 3, states
        assert states.count(ST_STAGING) == 1, states
        # and the SQL invariant still raises rather than silently refusing
        other = next(s["stripe_id"] for s in coord.ledger.all_stripes("run")
                     if s["state"] == ST_PENDING)
        try:
            coord.ledger.claim_stripe("run", other, conns[0].worker_id, 0, 1,
                                      T0 + 9999.0)
        except LeaseInvariantError:
            pass
        else:
            raise AssertionError(
                "a second compute-active claim for one worker was accepted")


# ===========================================================================
# L6 — red-first authenticity, ONE HALF OF THE REPAIR AT A TIME
# ===========================================================================
def l6_red_first_authenticity():
    """Reintroducing `now=<stale loop now>` as the lease origin must make the L1
    regression assertion FAIL — and so must reverting only the other half."""

    # MUTANT A: the invocation half reverted. The LIVE (repaired) scheduler, but
    # production hands it the stale loop clock again. Per-claim reading cannot
    # help: an injected clock is honoured verbatim, by design.
    def mutant_invocation():
        with tempfile.TemporaryDirectory() as tmp:
            coord = _coord(tmp)
            T0 = 1_786_500_000.0
            clock = _Clock(T0)
            with _patched_clock(clock):
                conns, pending = _late_backlog(coord, T0)
                claim_at = T0 + STALE_AGE
                clock.value = claim_at
                placed = coord.schedule_pending_stripes(
                    "run", "java_lcg", 1, conns, stage_prefix="run__st0",
                    now=T0)                                   # <-- THE MUTATION
            st = coord.ledger.get_stripe("run", placed[0]["stripe_id"])
            # the L1-GREEN assertion, verbatim
            assert st["lease_expires_at"] == claim_at + LEASE, (
                f"stale origin detected: {st['lease_expires_at']}")

    _mutant_red(mutant_invocation, "L6/A invocation half reverted")

    # MUTANT B: the SCHEDULER half reverted. Production stops injecting (the fix
    # that landed at the call site) but the body is the PINNED PRE-REPAIR one, so
    # it resolves the clock ONCE per pass. Two late claims in one pass then share
    # an origin again — the L2 assertion must fail. This is what proves the
    # per-claim read is load-bearing and not decoration on top of the call-site
    # change.
    def mutant_scheduler():
        with tempfile.TemporaryDirectory() as tmp:
            coord = _coord(tmp)
            T0 = 1_786_500_000.0
            clock = _Clock(T0)
            with _patched_clock(clock):
                conns, pending = _late_backlog(coord, T0, n_stripes=4,
                                               n_workers=2)
                clock.value, clock.step = T0 + STALE_AGE, 7.0
                sched = _bind_pinned(coord, clock)
                placed = sched("run", "java_lcg", 1, conns,
                               stage_prefix="run__st0")      # <-- no now=, fixed
            leases = [coord.ledger.get_stripe(
                "run", p["stripe_id"])["lease_expires_at"] for p in placed]
            assert len(placed) == 2, placed
            # the L2 assertion, verbatim
            assert len(set(leases)) == 2, (
                f"both late claims share one lease origin {leases}")

    _mutant_red(mutant_scheduler, "L6/B scheduler half reverted")


def l6_anchor_self_protection():
    """[R1] An anchor pointing at REPAIRED source must RAISE AnchorUnavailable,
    never silently credit a RED arm. Demonstrated against HEAD's own working
    tree, which now carries the repair."""
    live_src = open(os.path.join(_ROOT, SRC_REL)).read()
    tree = ast.parse(live_src)
    fn = _func_node(tree, "RangeMinerCoordinator", "schedule_pending_stripes")
    assert _entry_clock_resolutions(fn) == 0, (
        "the repaired scheduler still resolves `now` once at entry")
    assert _claim_lease_origin(fn) == "claim_now", (
        "the repaired scheduler does not stamp the lease from `claim_now`")
    # and the guard that would have to reject it
    saved = _PINNED_CACHE.pop("src", None)
    try:
        _PINNED_CACHE["src"] = live_src
        try:
            _pinned_scheduler()
        except AnchorUnavailable:
            pass
        else:
            raise AssertionError(
                "the anchor guard accepted REPAIRED source as the pre-repair "
                "anchor — every RED arm would be vacuous")
    finally:
        _PINNED_CACHE.pop("src", None)
        if saved is not None:
            _PINNED_CACHE["src"] = saved


# ===========================================================================
# L7 — non-lease timing preservation (the required six-site audit, computed)
# ===========================================================================
EXPECTED_PRE = ["fail_trial", "fail_trial", "fail_trial", "fail_trial",
                "fail_trial", "schedule_pending_stripes"]
EXPECTED_POST = ["fail_trial"] * 5


def l7_structural_six_sites():
    """The six shared-`now` consumers, computed from BOTH sources. Exactly ONE is
    modified, and it is the lease seam. Beta's expected result is confirmed
    against source rather than taken on trust."""
    pinned = ast.parse(_pinned_module_src())
    live = ast.parse(open(os.path.join(_ROOT, SRC_REL)).read())
    pre = _serve_now_kw_callees(
        _func_node(pinned, "RangeMinerCoordinator", "serve_trial"))
    post = _serve_now_kw_callees(
        _func_node(live, "RangeMinerCoordinator", "serve_trial"))
    assert pre == EXPECTED_PRE, pre
    assert post == EXPECTED_POST, post
    removed = list(pre)
    for c in post:
        removed.remove(c)
    assert removed == ["schedule_pending_stripes"], (
        f"the patch changed more than the lease seam: removed {removed}")


def l7_serve_loop_clock_capture_unchanged():
    """The serve-loop clock capture is byte-for-byte where it was: exactly one
    `now = time.time()` in serve_trial, still the FIRST statement of the while
    body. `process_lease_expiry` is still called WITHOUT the shared clock —
    Beta's correction to Alpha's preliminary concern, verified against source."""
    for label, src in (("pinned", _pinned_module_src()),
                       ("live", open(os.path.join(_ROOT, SRC_REL)).read())):
        fn = _func_node(ast.parse(src), "RangeMinerCoordinator", "serve_trial")
        # THE loop is the one gated on `_terminal()` — `serve_trial` also holds
        # the bounded inbound-drain loop, and picking "the only while" would be a
        # gate that breaks the next time an inner loop is added rather than a
        # gate about the serve loop.
        whiles = [n for n in ast.walk(fn)
                  if isinstance(n, ast.While)
                  and any(getattr(c.func, "id", None) == "_terminal"
                          for c in ast.walk(n.test) if isinstance(c, ast.Call))]
        assert len(whiles) == 1, (
            f"{label}: {len(whiles)} `while not _terminal()` loops")
        first = whiles[0].body[0]
        assert (isinstance(first, ast.Assign)
                and isinstance(first.targets[0], ast.Name)
                and first.targets[0].id == "now"
                and isinstance(first.value, ast.Call)
                and getattr(first.value.func, "attr", None) == "time"), (
            f"{label}: the loop's first statement is no longer "
            f"`now = time.time()`")
        captures = [n for n in ast.walk(fn)
                    if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id == "now"]
        assert len(captures) == 1, f"{label}: {len(captures)} `now =` captures"
        # process_lease_expiry: never given the shared clock, before or after
        for call in ast.walk(fn):
            if (isinstance(call, ast.Call)
                    and getattr(call.func, "attr", None) == "process_lease_expiry"):
                assert not call.keywords, f"{label}: process_lease_expiry got kwargs"
                assert len(call.args) == 2, f"{label}: {len(call.args)} args"


def l7_non_lease_consumers_behaviour():
    """BEHAVIOURAL arm: drive the REAL serve_trial with a pool it cannot fill and
    prove the admission window — a non-lease consumer of the same `now` — still
    measures from the loop clock and still terminates the trial. Also proves the
    instrumentation is emitted and did not change the terminal outcome."""
    logs: List[str] = []

    class _Cap(logging.Handler):
        def emit(self, rec):
            try:
                logs.append(rec.getMessage())
            except Exception:                                    # noqa: BLE001
                pass

    tmp = tempfile.mkdtemp(prefix="s172_l7_")
    ds = os.path.join(tmp, "dataset.json")
    with open(ds, "w") as fh:
        fh.write('[{"draw":1},{"draw":2},{"draw":3}]')
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind(("127.0.0.1", 0))
    lsock.listen(8)
    holder: Dict[str, Any] = {}
    handler = _Cap()
    COORD.logger.addHandler(handler)
    # The runner sets the ROOT level to CRITICAL to keep the gate output
    # readable; the coordinator's logger inherits that, so without this the
    # handler would receive NOTHING and the gate would report "no terminal
    # recorded" for a run that recorded one — a false red that reads exactly like
    # a real behavioural regression.
    _saved_level = COORD.logger.level
    COORD.logger.setLevel(logging.INFO)

    def run():
        try:
            holder["result"] = run_trial_miner(
                "run-l7", None, 5, "java_lcg", [1, 2, 3], 30, 0.25, 0.25,
                False, ds, worker_pool_size=2,
                staging_dir=os.path.join(tmp, "stg"), listen_sock=lsock,
                family_name="java_lcg", workflow_phase=1,
                skip_min=0, skip_max=0, window_anchor=0, generator_phase=0,
                window_size=3,
                # PRODUCTION DEFAULT: nothing but the code's own admission
                # decision may end this run.
                serve_timeout=None, worker_admission_timeout=2.0)
        except Exception:                                        # noqa: BLE001
            holder["err"] = traceback.format_exc()

    t = threading.Thread(target=run, name="l7-serve", daemon=True)
    t.start()
    t.join(timeout=60.0)
    try:
        assert not t.is_alive(), (
            "serve_trial did not terminate — the admission window no longer "
            "fires, which would be a behavioural change in a non-lease consumer")
        joined = "\n".join(logs)
        assert TC_WORKER_ADMISSION_TIMEOUT in joined or "admission timeout" in joined, (
            "no worker-admission-timeout terminal was recorded")
        # the instrumentation exists, ran, and reported
        sl = [m for m in logs if m.startswith("[S172-SL] summary")]
        assert len(sl) == 1, f"expected exactly one [S172-SL] summary, got {len(sl)}"
        for field in ("loop_seconds=", "iterations=", "iteration_max=",
                      "loop_now_age_max=", "schedule_max=", "dispatch_max=",
                      "expiry_max=", "drain_max=", "unattributed_total=",
                      "exit_seconds="):
            assert field in sl[0], f"missing {field} in {sl[0]}"
        bp = [m for m in logs if m.startswith("[S172-BP] summary")]
        assert len(bp) == 1, "the back-pressure summary was displaced"
        # [R1.1] THE PRODUCTION-WIRING PROOF, on the REAL serve_trial: every
        # iteration the loop ticked is accounted for, INCLUDING the terminal one
        # that ended this trial. `iteration_count < iterations` is the defect
        # signature, and it is the shape attempt 4 would have produced.
        res = holder.get("result")
        assert res is not None, holder.get("err", "run_trial_miner returned none")
        slm = res.get("serve_loop_timing")
        assert slm is not None, "serve_loop_timing is absent from the result"
        assert slm["iterations"] >= 1, slm
        assert slm["iteration_count"] == slm["iterations"], (
            f"{slm['iteration_count']} of {slm['iterations']} iterations "
            f"recorded — the terminal iteration was not closed (R1.1)")
        # and the residual is a residual of a partition: it can never exceed the
        # iteration total, which double-counting nested `msg` could not guarantee
        assert slm["unattributed_total"] <= slm["iteration_total"] + 1e-9, slm
    finally:
        COORD.logger.removeHandler(handler)
        COORD.logger.setLevel(_saved_level)
        try:
            lsock.close()
        except OSError:
            pass


def l7_instrumentation_is_monotonic_only():
    """The instrumentation CLASS may not read the wall clock: a segment timed off
    `time.time()` could be moved by an NTP step or by any gate that controls the
    clock, and — the reason that matters here — a timing instrument that shares a
    clock with the lease seam invites exactly the confusion this repair removes.

    [R1.3] THIS GATE IS NOT SUFFICIENT ON ITS OWN, and that is why it now says so
    in its own name. Inspecting the class proves nothing about how the class is
    WIRED: the first implementation computed the iteration's clock age at the CALL
    SITE as `time.time() - now`, so the instrument as a whole read the wall clock
    while this gate stayed green. `r1_3_call_site_adds_no_wall_read` covers the
    call site and is the other half."""
    live = ast.parse(open(os.path.join(_ROOT, SRC_REL)).read())
    cls = [n for n in ast.walk(live)
           if isinstance(n, ast.ClassDef) and n.name == "ServeLoopTiming"]
    assert len(cls) == 1
    calls = {getattr(n.func, "attr", None) for n in ast.walk(cls[0])
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and getattr(n.func.value, "id", None) == "time"}
    assert calls == {"perf_counter"}, (
        f"ServeLoopTiming reads {calls}; only perf_counter is permitted")
    # and the accumulator is inert: metrics never raise, a bad segment is dropped
    t = ServeLoopTiming()
    t.tick(1.0)
    t.stop("nosuchsegment", t.start())
    t.stop("schedule", "not-a-float")          # a caller error must not raise
    t.note_loop_now_age(float("nan"))          # a nonsense LABEL must not raise
    t.tick(2.0)
    t.close_current_iteration()
    m = t.metrics()
    # [R1.1] TWO ticks, both closed. The superseded form of this assertion —
    # `iterations == 2 and iteration_count == 1` — ENCODED THE DEFECT: it declared
    # the un-recorded terminal iteration to be correct behaviour and turned green
    # on it. Every ticked iteration is now accounted for, which is the invariant,
    # not an arithmetic coincidence of this scenario.
    assert m["iterations"] == 2 and m["iteration_count"] == 2, m
    assert m["schedule_count"] == 0, m


# ===========================================================================
# [R1.1] the terminal iteration is closed — the event the instrument exists for
# ===========================================================================
def _terminal_iteration_scenario(mono):
    """Two iterations where the LAST one is the long one. Returns the metrics.

    Deliberately shaped like attempt 4: a short iteration, then a long one that
    itself ends the trial. Under the pre-R1 instrument the long one was never
    recorded, because `tick()` closes an iteration only when the NEXT one begins
    and a terminal iteration has no next.

    ON THE 272.3 s SCRIPTED HERE (Beta certification, precision correction).
    Attempt 4's terminal iteration had already aged 272.3 s when it reached the
    scheduling pass; that same iteration later terminated the trial. 272.3 s is
    therefore the AGE AT THE SCHEDULING PASS, not the measured full duration of
    that production iteration — the full duration was LONGER and is UNMEASURED,
    which is precisely the gap this instrument closes. Using the number as this
    scenario's deterministic long terminal iteration remains correct: here it is
    a scripted duration chosen to match the one real quantity we have, not a
    claim about how long the production iteration actually ran."""
    with _patched_clock(mono):
        t = ServeLoopTiming()                       # mono[0]  -> _t0
        t.tick(WALL_1)                              # mono[1]  -> opens it 1
        t.tick(WALL_2)                              # mono[2]  -> closes it 1
        t.close_current_iteration()                 # mono[3]  -> closes it 2
        n_reads = len(mono.reads)
        t.close_current_iteration()                 # idempotent: no record, and
        assert len(mono.reads) == n_reads, (        # not even a clock read
            "the second close read the clock — it must short-circuit on the "
            "already-closed mark, or a repeated terminal path would both "
            "re-record and perturb the measurement")
        return t.metrics()                          # mono[4]  -> loop_seconds


WALL_1 = 1_786_500_000.0
WALL_2 = 1_786_500_001.0
SHORT = 1.0


def r1_1_terminal_iteration_is_recorded():
    """The LAST/TERMINAL iteration is the long one and it MUST become
    `iteration_max`, with `max_iteration_at` naming the instant it started."""
    m = _terminal_iteration_scenario(
        _MonoClock([0.0, 10.0, 10.0 + SHORT, 10.0 + SHORT + STALE_AGE, 300.0]))
    assert m["iterations"] == 2, m
    assert m["iteration_count"] == 2, (
        f"only {m['iteration_count']} of {m['iterations']} iterations were "
        f"recorded — the terminal one is missing, which is R1.1 exactly")
    assert abs(m["iteration_max"] - STALE_AGE) < 1e-9, (
        f"iteration_max is {m['iteration_max']}, not the terminal {STALE_AGE}s "
        f"iteration; the instrument would have missed attempt 4's own event")
    assert abs(m["iteration_total"] - (SHORT + STALE_AGE)) < 1e-9, m
    assert m["max_iteration_at"] == WALL_2, (
        f"max_iteration_at is {m['max_iteration_at']}, not the wall instant "
        f"{WALL_2} at which the worst iteration STARTED")
    # VIR-2 POSITIVE CONTROL: neutralise the terminal close on the PRODUCTION
    # class and re-run the identical scenario. The mutated path executes and the
    # credited assertion above must fail.
    _saved = ServeLoopTiming.close_current_iteration
    try:
        ServeLoopTiming.close_current_iteration = lambda self: None
        def _mut():
            mm = _terminal_iteration_scenario(
                _MonoClock([0.0, 10.0, 10.0 + SHORT, 300.0]))
            assert mm["iteration_count"] == 2, mm
            assert abs(mm["iteration_max"] - STALE_AGE) < 1e-9, mm
        _mutant_red(_mut, "close_current_iteration neutralised")
    finally:
        ServeLoopTiming.close_current_iteration = _saved


def _assert_finally_closes_iteration(src: str) -> None:
    """`serve_trial`'s `finally` must close the open iteration BEFORE it starts
    the exit timer — otherwise the terminal iteration's time is either lost or
    silently re-attributed to teardown."""
    fn = _func_node(ast.parse(src), "RangeMinerCoordinator", "serve_trial")
    tries = [n for n in ast.walk(fn)
             if isinstance(n, ast.Try) and n.finalbody
             and any(isinstance(s, ast.Assign)
                     and isinstance(s.targets[0], ast.Name)
                     and s.targets[0].id == "_exit_t0"
                     for s in ast.walk(n))]
    assert len(tries) == 1, (
        f"expected exactly one `finally` carrying the exit timer, found "
        f"{len(tries)}")
    body = tries[0].finalbody
    closes = [i for i, s in enumerate(body)
              if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
              and getattr(s.value.func, "attr", None) == "close_current_iteration"]
    assert closes, (
        "serve_trial's `finally` never closes the current iteration — a terminal "
        "iteration is therefore absent from iteration_max/unattributed_total")
    exits = [i for i, s in enumerate(body)
             if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name)
             and s.targets[0].id == "_exit_t0"]
    assert exits, "the exit timer is no longer started in this `finally`"
    assert closes[0] < exits[0], (
        f"close_current_iteration (stmt {closes[0]}) runs AFTER the exit timer "
        f"starts (stmt {exits[0]}); the loop and the teardown would overlap")
    assert closes[0] == 0, (
        f"close_current_iteration is statement {closes[0]} of the `finally`, not "
        f"the first — anything ahead of it is charged to the loop iteration")


def r1_1_production_closes_the_terminal_iteration():
    """The production wiring, over live source, with a source-level mutant."""
    live = open(os.path.join(_ROOT, SRC_REL)).read()
    _assert_finally_closes_iteration(live)
    mutant = live.replace("            _sl.close_current_iteration()\n", "", 1)
    assert mutant != live, "MUTANT NOT APPLIED — the close line was not found"
    _mutant_red(lambda: _assert_finally_closes_iteration(mutant),
                "terminal close removed from serve_trial's finally")


# ===========================================================================
# [R1.2] unattributed_total is a residual over a PARTITION, not over overlaps
# ===========================================================================
def r1_2_unattributed_excludes_nested_segments():
    """`msg` is timed INSIDE `drain`. Subtracting both charged message dispatch
    twice, so `unattributed_total` was not "loop time inside no named segment".

    The scenario is pure arithmetic and the expected residual is known exactly:

        iteration 20.0  =  accept 1.0 + drain 6.0 + schedule 2.0  +  residual
        drain 6.0       contains  msg 2.0
        residual        = 20.0 - (1.0 + 6.0 + 2.0) = 11.0        <- credited
        double-counted  = 20.0 - (1.0 + 6.0 + 2.0 + 2.0) = 9.0   <- the defect
    """
    mono = _MonoClock([
        0.0,                       # __init__
        10.0,                      # tick  -> iteration opens
        11.0, 12.0,                # accept  1.0
        12.0,                      # drain start
        13.0, 15.0,                # msg     2.0   (nested inside drain)
        18.0,                      # drain stop -> 6.0
        18.0, 20.0,                # schedule 2.0
        30.0,                      # close -> iteration 20.0
        31.0,                      # metrics loop_seconds
        31.0,                      # metrics again, for the mutant
    ])
    with _patched_clock(mono):
        t = ServeLoopTiming()
        t.tick(WALL_1)
        _a = t.start(); t.stop("accept", _a)
        _d = t.start()
        _m = t.start(); t.stop("msg", _m)
        t.stop("drain", _d)
        _s = t.start(); t.stop("schedule", _s)
        t.close_current_iteration()
        m = t.metrics()

        assert abs(m["iteration_total"] - 20.0) < 1e-9, m
        assert abs(m["accept_total"] - 1.0) < 1e-9, m
        assert abs(m["drain_total"] - 6.0) < 1e-9, m
        assert abs(m["schedule_total"] - 2.0) < 1e-9, m
        # the nested total is still REPORTED — R1.2 removes it from the
        # subtraction, it does not stop measuring message dispatch
        assert abs(m["msg_total"] - 2.0) < 1e-9, m
        assert m["msg_count"] == 1, m
        assert m["msg_total"] <= m["drain_total"], (
            "msg is supposed to be nested inside drain; if it is not, the "
            "partition this gate asserts is the wrong partition")
        assert abs(m["unattributed_total"] - 11.0) < 1e-9, (
            f"unattributed_total is {m['unattributed_total']}, expected exactly "
            f"11.0; 9.0 means nested msg time is being subtracted twice")

        # VIR-2 POSITIVE CONTROL: empty the nesting declaration on the PRODUCTION
        # class and recompute through the real `metrics()`. The mutated path
        # executes and the credited assertion must fail.
        _saved = ServeLoopTiming.NESTED_SEGMENTS
        try:
            ServeLoopTiming.NESTED_SEGMENTS = ()
            def _mut():
                bad = t.metrics()
                assert abs(bad["unattributed_total"] - 11.0) < 1e-9, bad
            _mutant_red(_mut, "NESTED_SEGMENTS emptied -> msg subtracted twice")
        finally:
            ServeLoopTiming.NESTED_SEGMENTS = _saved
    assert not mono.values, f"scripted clock not fully consumed: {mono.values}"


# ===========================================================================
# [R1.3] the age is monotonic AT THE CALL SITE, not only inside the class
# ===========================================================================
def r1_3_loop_now_age_is_monotonic():
    """`loop_now_age` is derived from the instrument's own monotonic mark. The
    loop's `now` is retained ONLY as the wall-time label.

    Credited proof: the wall clock is stepped by an hour BETWEEN the tick and the
    measurement, and (a) the reported age is unchanged, (b) the instrument read
    the wall clock **zero** times."""
    mono = _MonoClock([0.0, 10.0, 25.0, 30.0], wall=WALL_1)
    with _patched_clock(mono):
        t = ServeLoopTiming()
        t.tick(WALL_1)
        mono.wall = WALL_1 - 3600.0          # a system-clock step, backwards
        t.note_loop_now_age(WALL_1)
        m = t.metrics()
    assert abs(m["loop_now_age_max"] - 15.0) < 1e-9, (
        f"loop_now_age_max is {m['loop_now_age_max']}, expected the monotonic "
        f"15.0s; a wall-derived age would have absorbed the clock step")
    assert m["loop_now_age_at"] == WALL_1, (
        f"loop_now_age_at is {m['loop_now_age_at']}; the loop `now` must survive "
        f"as the LABEL so a reader can find the instant in the log")
    assert mono.wall_reads == 0, (
        f"the instrument read the wall clock {mono.wall_reads} time(s); it must "
        f"read none")
    assert not mono.values, f"scripted clock not fully consumed: {mono.values}"

    # VIR-2 POSITIVE CONTROL: restore a wall-derived age on the PRODUCTION class
    # and re-run. Both credited assertions must fail.
    _saved = ServeLoopTiming.note_loop_now_age

    def _wall_derived(self, wall_now=None):
        age = time_module.time() - (wall_now or 0.0)
        if age > self.loop_now_age_max:
            self.loop_now_age_max = float(age)
            self.loop_now_age_at = wall_now

    try:
        ServeLoopTiming.note_loop_now_age = _wall_derived
        def _mut():
            mm = _MonoClock([0.0, 10.0, 30.0], wall=WALL_1)
            with _patched_clock(mm):
                t2 = ServeLoopTiming()
                t2.tick(WALL_1)
                mm.wall = WALL_1 - 3600.0
                t2.note_loop_now_age(WALL_1)
                out = t2.metrics()
            assert abs(out["loop_now_age_max"] - 15.0) < 1e-9, out
            assert mm.wall_reads == 0, mm.wall_reads
        _mutant_red(_mut, "note_loop_now_age reverted to a wall-derived age")
    finally:
        ServeLoopTiming.note_loop_now_age = _saved


def _serve_trial_wall_reads(src: str) -> int:
    fn = _func_node(ast.parse(src), "RangeMinerCoordinator", "serve_trial")
    return len([n for n in ast.walk(fn)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "time"
                and getattr(n.func.value, "id", None) == "time"])


def _assert_no_instrument_wall_read(src: str, pinned_reads: int) -> None:
    """No argument of any `_sl.*` call may contain a wall-clock read, AND
    `serve_trial` must hold no more `time.time()` calls than the PRE-REPAIR
    source did. The second half is the one that matters: it is a count of
    PRODUCTION wall reads, so it catches an instrumentation clock read wherever
    in the function it is hidden, not only inside an `_sl.*` argument list."""
    fn = _func_node(ast.parse(src), "RangeMinerCoordinator", "serve_trial")
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and getattr(node.func.value, "id", None) == "_sl"):
            for arg in list(node.args) + [k.value for k in node.keywords]:
                for sub in ast.walk(arg):
                    if (isinstance(sub, ast.Call)
                            and isinstance(sub.func, ast.Attribute)
                            and sub.func.attr == "time"
                            and getattr(sub.func.value, "id", None) == "time"):
                        raise AssertionError(
                            f"`_sl.{node.func.attr}` is fed a `time.time()` read "
                            f"at line {sub.lineno}: the instrument is wall-clock "
                            f"wired even if the class is not")
    live_reads = _serve_trial_wall_reads(src)
    assert live_reads == pinned_reads, (
        f"serve_trial performs {live_reads} `time.time()` reads; the pre-repair "
        f"source performed {pinned_reads}. Instrumentation may not add a "
        f"production wall-clock read (Beta R1.3: do not add another one)")


def r1_3_call_site_adds_no_wall_read():
    """The half `l7 instrumentation is monotonic + inert` cannot see."""
    live = open(os.path.join(_ROOT, SRC_REL)).read()
    pinned_reads = _serve_trial_wall_reads(_pinned_module_src())
    assert pinned_reads > 0, (
        "the pinned serve_trial performs no wall read at all — the count "
        "baseline would be vacuous")
    _assert_no_instrument_wall_read(live, pinned_reads)
    mutant = live.replace("_sl.note_loop_now_age(now)",
                          "_sl.note_loop_now_age(time.time() - now)", 1)
    assert mutant != live, "MUTANT NOT APPLIED — the call site was not found"
    _mutant_red(lambda: _assert_no_instrument_wall_read(mutant, pinned_reads),
                "call site recomputes the age from time.time()")


# ===========================================================================
# runner
# ===========================================================================
def main():
    logging.basicConfig(level=logging.CRITICAL)
    print("=" * 74)
    print("S172 F1 — LEASE-ORIGIN REPAIR: gates L1-L7")
    print(f"pinned pre-repair anchor: {PINNED_COMMIT}:{SRC_REL}")
    print("=" * 74)

    # The anchor is resolved ONCE, up front: if it is unavailable every RED arm
    # must report UNAVAILABLE rather than quietly passing on repaired source.
    anchor_error = None
    try:
        _pinned_scheduler()
        print(f"  anchor OK — {PINNED_COMMIT} still carries both defect "
              f"surfaces\n")
    except AnchorUnavailable as e:
        anchor_error = str(e)
        print(f"  anchor UNAVAILABLE: {anchor_error}\n")

    # EVERY arm in this group consumes the pinned pre-repair source, so the
    # anchor gate below is unconditional. It used to select by substring
    # ("RED" in name or "L6" in name or …), which meant a newly added
    # pinned-source gate whose name matched nothing would run WITHOUT the anchor
    # and report FAIL where VIR-3 requires UNAVAILABLE. Membership of this dict
    # is now the whole rule.
    red_arms = {
        "L1-RED  pre-repair source reproduces the ~26 s lease": l1_red_pre_repair,
        "L3      no premature constant failure (+RED)": l3_no_premature_expiry,
        "L6      red-first authenticity, each half alone": l6_red_first_authenticity,
        "L7      six-site audit, computed from both sources": l7_structural_six_sites,
        "L7      serve-loop clock capture unchanged": l7_serve_loop_clock_capture_unchanged,
        "R1.3    call site adds no wall read (+mutant)": r1_3_call_site_adds_no_wall_read,
    }
    green_arms = {
        "L1-GREEN full lease from the actual claim": l1_green_repaired,
        "L2      multiple late claims, own origins": l2_multiple_late_claims,
        "L4      genuine constant expiry preserved": l4_genuine_expiry_constant,
        "L4      hybrid first-expiry retry unchanged": l4_genuine_expiry_hybrid_retry,
        "L5      one-active invariant unchanged": l5_one_active_invariant,
        "L6      anchor rejects repaired source": l6_anchor_self_protection,
        "L7      non-lease consumers still fire (live)": l7_non_lease_consumers_behaviour,
        "L7      instrumentation is monotonic + inert": l7_instrumentation_is_monotonic_only,
        "R1.1    terminal iteration recorded (+mutant)": r1_1_terminal_iteration_is_recorded,
        "R1.1    production closes it in finally (+mutant)":
            r1_1_production_closes_the_terminal_iteration,
        "R1.2    unattributed excludes nested msg (+mutant)":
            r1_2_unattributed_excludes_nested_segments,
        "R1.3    loop_now_age is monotonic (+mutant)": r1_3_loop_now_age_is_monotonic,
    }

    for name, fn in green_arms.items():
        _check(name, fn)
    for name, fn in red_arms.items():
        if anchor_error is not None:
            _unavailable(name, anchor_error)
            continue
        _check(name, fn)

    print("=" * 74)
    ok = sum(1 for _, p, _ in _results if p)
    for name, passed, tb in _results:
        if not passed:
            print(f"\n--- {name} ---\n{tb}")
    print(f"\n{ok}/{len(_results)} checks green")
    if ok == len(_results):
        print("COMPLETION SENTINEL: PASS — S172 F1 lease-origin gates L1-L7 green "
              "(pending Team Beta review).")
    else:
        print("COMPLETION SENTINEL: FAIL")
    return 0 if ok == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
