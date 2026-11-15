#!/usr/bin/env python3
"""S172 L-4 — the staging-admission stall: a stripe that reaches `done` on EITHER path
releases its admission and resumes deferred work.

Governing brief: docs/CLAUDE_CODE_INSTRUCTIONS_MINER_ADMISSION_STALL_20261114.md (Part 1).
Finding it repairs: docs/LEADS.md L-4; Beta X-MINER-START R1 Q7 (OBJECTION · LIVE).

CPU only. Real coordinator; fake workers over real framed loopback sockets; every ledger
and staging directory is throwaway. NO PUMP NUDGER ANYWHERE: nothing in this suite calls
`_pump_deferred`. The ordering harness (tests/fixtures/admission_stall_driver.py) binds
OBSERVERS that call each real method exactly once when the coordinator does, and imposes
order only by waiting.

Gates
  G-UNIT          the L-4 sequence, deterministic, on the real `_serve_dispatch`: s0
                  fully staged, s1's frame deferred behind s0, then s0's StripeComplete.
                  s1 must stage; s0 must not hold admission; the pump must not run on
                  the dispatch thread
  G-SF-{2,4,25}   staging_first forced (last staging job, and its completion pump,
                  finish BEFORE the StripeComplete is processed), L4_REPS runs each
  G-CF-{2,4,25}   complete_first forced (StripeComplete processed BEFORE the last
                  staging job finishes), L4_REPS runs each
  G-RUN4          Run 4's shape: 25 workers, 32 stripes a phase over [2^31, 2^32), four
                  phases, production caps; free, staging_first and complete_first
  G-ONE           one worker: ledger rows and assignments byte-equal to the control
                  captured on the base tree (7b9184a)
  G-OFFTHREAD     R2: in the StripeComplete branch the pump is HANDED to the staging
                  executor (never called on the dispatch thread), after the finalize,
                  and the branch writes no admission state
  G-SCOPE         R3/SR-1: against 7b9184a exactly one definition changed
                  (`_serve_dispatch`); the admission, pump, resume-credit and finalize
                  surfaces are byte-identical
Mutants are function-level, compiled against the PRODUCTION module globals (A8-B2).
Every gate terminates PASS | FAIL; only PASS accepts. A run whose ordering could not be
forced is not credited (reported INCOMPLETE inside the gate's FAIL).
"""
import ast
import contextlib
import hashlib
import importlib
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
SCRATCH = os.path.join(ROOT, "data", "admission_stall", "suite_tmp")
os.makedirs(SCRATCH, exist_ok=True)

from fixtures import admission_stall_driver as AD          # noqa: E402

REPS = int(os.environ.get("L4_REPS", "20"))
BASE_COMMIT = "7b9184a95cd125c14685008cd306fe0fd9ff3040"
CONTROL = os.path.join(HERE, "fixtures", "admission_stall_one_worker_control_7b9184a.txt")
C_MOD = "miner.range_miner_coordinator"
C_PATH = os.path.join(ROOT, "miner", "range_miner_coordinator.py")
T31, T32 = 2 ** 31, 2 ** 32
SMALL_CAPS = {"nvidia": 125, "amd": 125, "nvidia_hybrid": 125, "amd_hybrid": 125}
PROD_CAPS = {"nvidia": 5_000_000, "amd": 2_000_000,
             "nvidia_hybrid": 2_500_000, "amd_hybrid": 1_000_000}

RESULTS, MUTANTS = [], []


class Fail(AssertionError):
    pass


def ok_(cond, msg):
    if not cond:
        raise Fail(msg)


@contextlib.contextmanager
def quiet():
    root = logging.getLogger()
    level = root.level
    root.setLevel(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        root.setLevel(level)


@contextlib.contextmanager
def tmpdir(prefix):
    d = tempfile.mkdtemp(prefix=prefix, dir=SCRATCH)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def run_gate(name, fn):
    try:
        detail = fn()
        RESULTS.append((name, True, detail or ""))
    except BaseException as e:   # noqa: BLE001
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"[:1500]))


# ---------------------------------------------------------------------------
# function-level mutation against production globals (A8-B2)
# ---------------------------------------------------------------------------
def _def_node(tree, qualname):
    body = tree.body
    node = None
    for part in qualname.split("."):
        node = next((n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                     and n.name == part), None)
        if node is None:
            raise RuntimeError(f"definition {qualname} not found")
        body = node.body
    return node


def def_source(path, qualname, src=None):
    src = src if src is not None else open(path, encoding="utf-8").read()
    node = _def_node(ast.parse(src), qualname)
    start = (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1
    lines = src.splitlines(keepends=True)[start:node.end_lineno]
    return textwrap.dedent("".join(lines))


@contextlib.contextmanager
def patched_def(modname, qualname, old, new):
    mod = importlib.import_module(modname)
    src = def_source(mod.__file__, qualname)
    n = src.count(old)
    if n != 1:
        raise RuntimeError(f"MUTANT ANCHOR ERROR {qualname}: found {n} times: {old[:80]!r}")
    ns = {}
    exec(compile(src.replace(old, new), f"{mod.__file__}<{qualname}>", "exec"),
         mod.__dict__, ns)
    name = qualname.split(".")[-1]
    owner = mod
    for part in qualname.split(".")[:-1]:
        owner = getattr(owner, part)
    saved = owner.__dict__[name]
    setattr(owner, name, ns[name])
    try:
        yield
    finally:
        setattr(owner, name, saved)


def run_mutant(label, qualname, old, new, gate_fn):
    try:
        with patched_def(C_MOD, qualname, old, new):
            try:
                with quiet():
                    det = gate_fn()
                MUTANTS.append((label, False, f"gate stayed GREEN under the mutant: {det}"))
            except BaseException as e:   # noqa: BLE001
                MUTANTS.append((label, True, f"RED: {type(e).__name__}: {str(e)[:300]}"))
    except RuntimeError as e:
        MUTANTS.append((label, False, f"ERROR (not a detection): {e}"))


# ===========================================================================
# G-UNIT — the L-4 sequence on the real _serve_dispatch, deterministic
# ===========================================================================
_CTX = dict(trial_number=7, window_size=5, window_anchor=2, generator_phase=0,
            sessions=["midday", "evening"], skip_min=1, skip_max=9,
            prng_base="java_lcg", forward_threshold=0.40, reverse_threshold=0.45,
            dataset_sha256="d" * 64, residue_sha256="r" * 64)
_VARIANTS = ["java_lcg", "java_lcg_reverse", "java_lcg_hybrid", "java_lcg_hybrid_reverse"]


def _quiesce(coord, rounds=3):
    """Wait until the staging executor has run everything queued so far: a barrier of
    one no-op per executor thread, repeated (a pump may queue further work)."""
    for _ in range(rounds):
        ex = coord._staging_executor
        if ex is None:
            return
        n = max(2, int(getattr(coord.config, "staging_workers", 4)))
        for f in [ex.submit(time.sleep, 0.02) for _ in range(n)]:
            f.result(timeout=30)


def g_unit():
    import hashlib as _h
    from miner.range_miner_coordinator import (CoordinatorConfig, MinerLedger,
                                               NodeConfig, RangeMinerCoordinator)
    from miner.range_miner_protocol import StripeCompleteMessage, SubStripeResultMessage
    from miner.range_miner_worker import build_substripe_payload_bytes
    with tmpdir("unit_") as d:
        led = MinerLedger(os.path.join(d, "l.db"))
        coord = RangeMinerCoordinator(
            CoordinatorConfig(staging_dir=os.path.join(d, "stg"), miner_stripe_size=30),
            led, phase5_sink=AD.ProbeSink())
        led.set_trial_context("run", dict(_CTX))
        callers = []
        real_pump = coord._pump_deferred

        def pump():
            callers.append((sys._getframe(1).f_code.co_name,
                            threading.current_thread().name))
            return real_pump()
        coord._pump_deferred = pump

        def reg(w):
            return coord.register_worker(
                worker_id=w, hostname=w.split(":")[0], backend="cuda",
                capabilities={"seed_caps": dict(PROD_CAPS),
                              "supported_variants": list(_VARIANTS)},
                node_config=NodeConfig(hostname=w.split(":")[0],
                                       spool_root="/var/spool/miner",
                                       ssh_address="10.0.0.9", ssh_user="michael"),
                now=100.0)
        A, B = reg("hostA:gpu0"), reg("hostB:gpu0")
        asg = coord.assign_stripes("run", "java_lcg", 1, 60, [A, B], now=100.0)
        by_w = {a["worker_id"]: a["stripe_id"] for a in asg}
        ok_(by_w == {"hostA:gpu0": "run_s0", "hostB:gpu0": "run_s1"}, f"assignment {asg}")
        A.record_assignment("run_s0", 0)
        B.record_assignment("run_s1", 0)
        wmap = {"hostA:gpu0": A, "hostB:gpu0": B}

        def elig():
            return [A, B]

        def result(w, sid, start):
            obj, pb = build_substripe_payload_bytes(sid, 0, start, 30,
                                                    [[start, 0.9, None, [1]]])
            return SubStripeResultMessage(
                worker_id=w, stripe_id=sid, sub_index=0, seed_start=start,
                seed_count=30, survivor_count=1, inline=obj, size_bytes=len(pb),
                sha256=_h.sha256(pb).hexdigest())

        # 1. s0's only sub-stripe stages completely (job, its tail pump, its callback).
        coord._serve_dispatch(result("hostA:gpu0", "run_s0", 0), "run", "hostA:gpu0",
                              wmap, elig)
        _quiesce(coord)
        st0 = [s["staging_status"] for s in led.get_shards("run", "run_s0", 0)]
        ok_(st0 == ["verified"], f"s0 did not stage: {st0}")
        ok_(led.get_stripe("run", "run_s0")["state"] != "done", "s0 done before its SC")
        # 2. s1's frame arrives while s0 holds admission: it is deferred.
        coord._serve_dispatch(result("hostB:gpu0", "run_s1", 30), "run", "hostB:gpu0",
                              wmap, elig)
        ok_([(e[3], e[5]) for e in coord._deferred] == [("run_s1", 0)],
            f"s1 not deferred: {[(e[3], e[5]) for e in coord._deferred]}")
        ok_(list(coord._admitted) == [("run", "run_s0", 0)],
            f"admission {list(coord._admitted)}")
        # 3. THE L-4 ORDERING: s0's StripeComplete is processed after its last staging
        #    job finished. Nothing else happens afterwards except the coordinator's own
        #    reactions.
        n_before = len(callers)
        coord._serve_dispatch(StripeCompleteMessage(
            worker_id="hostA:gpu0", stripe_id="run_s0", substripes_done=1,
            survivors_total=1), "run", "hostA:gpu0", wmap, elig)
        _quiesce(coord, rounds=6)
        s0 = led.get_stripe("run", "run_s0")["state"]
        st1 = [s["staging_status"] for s in led.get_shards("run", "run_s1", 0)]
        after = callers[n_before:]
        dispatch_pumps = [c for c in callers if c[0] == "_serve_dispatch"]
        held = list(coord._admitted)
        deferred = [(e[3], e[5]) for e in coord._deferred]
        if coord._staging_executor is not None:
            coord._staging_executor.shutdown(wait=True)
        ok_(s0 == "done", f"s0 is {s0}, not done")
        ok_(not dispatch_pumps,
            f"_pump_deferred called directly from _serve_dispatch: {dispatch_pumps}")
        ok_(("run", "run_s0", 0) not in held,
            f"STALL: done stripe s0 still holds admission {held}; deferred {deferred}")
        ok_(not deferred, f"STALL: s1 still deferred {deferred}")
        ok_(st1 == ["verified"], f"s1 did not stage: {st1}")
        ok_(all(not t.startswith("MainThread") for _c, t in after),
            f"a pump after the SC ran on the dispatch thread: {after}")
        return (f"s0 done, admission released, s1 staged; pumps after SC: "
                f"{len(after)} on {sorted({t.split('_')[0] for _c, t in after})}; "
                f"0 called from _serve_dispatch")


# ===========================================================================
# forced orderings — staging_first / complete_first
# ===========================================================================
def _small_shape(n_workers):
    stripes = max(4, 2 * n_workers)
    return dict(total_seeds=250 * stripes, stripe_size=250, test_both_modes=False,
                caps=SMALL_CAPS)


def _ordered_gate(order, n_workers, reps):
    fails, forced_bad, durations = [], [], []
    stripes_seen = None
    for i in range(reps):
        with tmpdir(f"{order[:2]}{n_workers}_") as d:
            with quiet():
                r = AD.run_ordered(d, n_workers=n_workers, order=order,
                                   serve_timeout=60.0, force_wait_s=20.0,
                                   **_small_shape(n_workers))
        durations.append(r["seconds"])
        n_str = sum(r["stripe_states"].values())
        stripes_seen = n_str
        bad = []
        if not r["committed"]:
            bad.append(f"not committed ({r['state']}, {r['stripe_states']})")
        if r["stripe_states"].get("done", 0) != n_str or n_str == 0:
            bad.append(f"stripes {r['stripe_states']}")
        if r["pump_callers"].get("_serve_dispatch"):
            bad.append(f"pump called from _serve_dispatch x{r['pump_callers']['_serve_dispatch']}")
        if r["error"] or r["worker_errors"]:
            bad.append(f"error {(r['error'] or '')[-200:]} {r['worker_errors']}")
        if bad:
            fails.append((i, bad))
        if r["forced_failed"] or r["forced_ok"] != n_str:
            forced_bad.append((i, r["forced_ok"], n_str, r["forced_failed"][:3]))
    ok_(not fails, f"{len(fails)}/{reps} runs failed: {fails[:3]}")
    ok_(not forced_bad,
        f"INCOMPLETE — ordering not forced in {len(forced_bad)}/{reps} runs: {forced_bad[:3]}")
    return (f"{reps}/{reps} committed, {stripes_seen} stripes each, ordering forced on "
            f"every stripe; max {max(durations):.1f}s")


def g_sf(n):
    return lambda: _ordered_gate("staging_first", n, REPS)


def g_cf(n):
    return lambda: _ordered_gate("complete_first", n, REPS)


def _contiguous(rows, ph):
    items = sorted((s, c) for (_sid, s, c, p, _f, _st) in rows if p == ph)
    if not items:
        return None
    cur = items[0][0]
    for s, c in items:
        if s != cur:
            return ("gap", s, cur)
        cur = s + c
    return (items[0][0], cur, len(items))


def g_run4():
    out = []
    for order in ("free", "staging_first", "complete_first"):
        with tmpdir(f"run4_{order}_") as d:
            with quiet():
                r = AD.run_ordered(d, n_workers=25, order=order, seed_start=T31,
                                   total_seeds=T31, stripe_size=2 ** 26,
                                   test_both_modes=True, caps=PROD_CAPS,
                                   serve_timeout=900.0, force_wait_s=120.0)
        ok_(r["committed"], f"{order}: not committed: {r['state']} {r['stripe_states']} "
                            f"{(r['error'] or '')[-300:]}")
        for ph in (1, 2, 3, 4):
            got = _contiguous(r["stripes"], ph)
            ok_(got == (T31, T32, 32), f"{order}: phase {ph}: {got}")
        ok_(r["stripe_states"] == {"done": 128}, f"{order}: {r['stripe_states']}")
        ok_(not r["pump_callers"].get("_serve_dispatch"), f"{order}: dispatch-thread pump")
        if order != "free":
            ok_(not r["forced_failed"] and r["forced_ok"] == 128,
                f"INCOMPLETE {order}: forced {r['forced_ok']}/128 {r['forced_failed'][:3]}")
        out.append(f"{order} {r['seconds']:.0f}s")
    return ("25 workers, 4 phases x 32 stripes over [2^31, 2^32), production caps, "
            "committed: " + ", ".join(out))


# ===========================================================================
# G-ONE — the one-worker control, byte-equal to the base tree
# ===========================================================================
ONE_SHAPE = dict(n_workers=1, order="free", total_seeds=1000, stripe_size=250,
                 test_both_modes=True, caps=SMALL_CAPS, seed_start=0)


def one_worker_run(d):
    with quiet():
        r = AD.run_ordered(d, serve_timeout=120.0, dump_ledger=True, **ONE_SHAPE)
    return r, {"stripes": r["stripes"], "assigns": r["assigns"], "ledger": r["ledger"]}


def g_one():
    with open(CONTROL) as f:
        ctl = json.load(f)
    ok_(ctl.get("completion_sentinel") == "CAPTURE COMPLETE", "control incomplete")
    ok_(ctl["driver_sha256"] == AD.driver_sha256(),
        "the ordering driver changed since the control was captured — comparison invalid")
    base = subprocess.check_output(
        ["git", "-C", ROOT, "show", f"{ctl['head_at_capture']}:miner/range_miner_coordinator.py"])
    ok_(hashlib.sha256(base).hexdigest() == ctl["coordinator_sha256"],
        "control was not captured from the base commit's coordinator")
    live = open(C_PATH, "rb").read()
    ok_(hashlib.sha256(live).hexdigest() != ctl["coordinator_sha256"],
        "live coordinator equals the base — the comparison would be vacuous")
    ok_(ctl["shape"] == {k: (list(v) if isinstance(v, tuple) else v)
                         for k, v in ONE_SHAPE.items()}, "shape changed")
    with tmpdir("one_") as d:
        r, got = one_worker_run(d)
    ok_(r["committed"], f"one-worker run not committed: {r['state']}")
    for k in ("stripes", "assigns", "ledger"):
        if AD.canonical(got[k]) != AD.canonical(ctl[k]):
            if k == "ledger":
                diff = [t for t in sorted(set(got[k]) | set(ctl[k]))
                        if AD.canonical(got[k].get(t)) != AD.canonical(ctl[k].get(t))]
                raise Fail(f"ledger tables differ from the base control: {diff}")
            raise Fail(f"{k} differs from the base control")
    dg = hashlib.sha256(AD.canonical(got).encode()).hexdigest()
    ok_(dg == ctl["digest"], "digest differs")
    n_rows = sum(len(t["rows"]) for t in got["ledger"].values())
    return (f"{len(got['stripes'])} stripes, {len(got['assigns'])} assigns, "
            f"{len(got['ledger'])} tables / {n_rows} rows byte-equal to "
            f"{ctl['head_at_capture'][:7]} ({dg[:12]})")


# ===========================================================================
# G-OFFTHREAD (R2, R4 static) and G-SCOPE (R3, SR-1)
# ===========================================================================
def _sc_branch(fn_node):
    """The `mt == "stripe_complete"` branch that FINALIZES (the dispatch function tests
    that message type in more than one place; only one branch completes a stripe)."""
    found = []
    for n in ast.walk(fn_node):
        if (isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                and isinstance(n.test.left, ast.Name) and n.test.left.id == "mt"
                and any(isinstance(c, ast.Constant) and c.value == "stripe_complete"
                        for c in n.test.comparators)
                and "finalize_stripe" in ast.unparse(ast.Module(body=n.body,
                                                                type_ignores=[]))):
            found.append(n)
    return found[0] if len(found) == 1 else None


def g_offthread():
    src = open(C_PATH, encoding="utf-8").read()
    fn = _def_node(ast.parse(src), "RangeMinerCoordinator._serve_dispatch")
    br = _sc_branch(fn)
    ok_(br is not None, "stripe_complete branch not found")
    body = ast.Module(body=br.body, type_ignores=[])
    calls = [n for n in ast.walk(body) if isinstance(n, ast.Call)]
    direct = [ast.unparse(c) for c in calls
              if isinstance(c.func, ast.Attribute) and c.func.attr == "_pump_deferred"]
    ok_(not direct, f"the StripeComplete branch CALLS the pump on the dispatch thread: {direct}")
    handoffs = [c for c in calls
                if isinstance(c.func, ast.Attribute) and c.func.attr == "submit"
                and isinstance(c.func.value, ast.Call)
                and isinstance(c.func.value.func, ast.Attribute)
                and c.func.value.func.attr == "_staging_exec"
                and len(c.args) == 1 and isinstance(c.args[0], ast.Attribute)
                and c.args[0].attr == "_pump_deferred"]
    ok_(len(handoffs) == 1, f"expected exactly one hand-off of the pump to the staging "
                            f"executor, found {len(handoffs)}")
    fins = [c for c in calls if isinstance(c.func, ast.Attribute)
            and c.func.attr == "finalize_stripe"]
    ok_(len(fins) == 1, f"finalize calls in the branch: {len(fins)}")
    ok_((fins[0].lineno, fins[0].col_offset) < (handoffs[0].lineno, handoffs[0].col_offset),
        "the pump is handed off BEFORE the finalize — it could run before `done` is written")
    forbidden = {"_try_admit_locked", "_release_admission", "_prune_admitted_locked",
                 "_defer_locked", "_submit_with_slot", "enqueue_staging"}
    touched = sorted({c.func.attr for c in calls if isinstance(c.func, ast.Attribute)}
                     & forbidden)
    ok_(not touched, f"the branch reaches admission surfaces directly: {touched}")
    writes = [ast.unparse(n) for n in ast.walk(body)
              if isinstance(n, (ast.Attribute,)) and n.attr in ("_admitted", "_deferred")]
    ok_(not writes, f"the branch names admission state: {writes}")
    return ("StripeComplete branch: finalize, then ONE hand-off "
            "`self._staging_exec().submit(self._pump_deferred)`; no direct pump call; "
            "no admission surface or state named")


def _def_digests(src):
    tree = ast.parse(src)
    out = {}

    def walk(nodes, prefix):
        for n in nodes:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[prefix + n.name] = hashlib.sha256(
                    ast.dump(n, include_attributes=False).encode()).hexdigest()
            elif isinstance(n, ast.ClassDef):
                walk(n.body, prefix + n.name + ".")
    walk(tree.body, "")
    return out


PINNED_SAME = (
    "RangeMinerCoordinator._pump_deferred", "RangeMinerCoordinator._prune_admitted_locked",
    "RangeMinerCoordinator._try_admit_locked", "RangeMinerCoordinator._attempt_live_locked",
    "RangeMinerCoordinator._release_admission", "RangeMinerCoordinator._defer_locked",
    "RangeMinerCoordinator.enqueue_staging", "RangeMinerCoordinator._submit_with_slot",
    "RangeMinerCoordinator._run_staging_job", "RangeMinerCoordinator.finalize_stripe",
    "RangeMinerCoordinator._resume_paused_connections",
    "RangeMinerCoordinator._grant_resume_credit", "RangeMinerCoordinator.staging_can_accept",
    "RangeMinerCoordinator._staging_exec", "RangeMinerCoordinator._on_staging_failed",
    "RangeMinerCoordinator.handle_stripe_failure",
    "RangeMinerCoordinator._handle_stripe_failure_locked",
)
DECLARED_CHANGED_VS_BASE = {"RangeMinerCoordinator._serve_dispatch"}


def g_scope():
    base = subprocess.check_output(
        ["git", "-C", ROOT, "show", f"{BASE_COMMIT}:miner/range_miner_coordinator.py"]
    ).decode()
    b, l = _def_digests(base), _def_digests(open(C_PATH, encoding="utf-8").read())
    for k in PINNED_SAME:
        ok_(k in b and k in l, f"{k} missing")
        ok_(b[k] == l[k], f"{k} changed — a pinned admission/back-pressure surface")
    changed = {k for k in b if k in l and b[k] != l[k]}
    added, removed = set(l) - set(b), set(b) - set(l)
    ok_(not added and not removed, f"added {sorted(added)} removed {sorted(removed)}")
    ok_(changed == DECLARED_CHANGED_VS_BASE,
        f"undeclared {sorted(changed - DECLARED_CHANGED_VS_BASE)}; "
        f"declared but unchanged {sorted(DECLARED_CHANGED_VS_BASE - changed)}")
    return (f"vs {BASE_COMMIT[:7]}: 1 changed (_serve_dispatch), 0 added, 0 removed; "
            f"{len(PINNED_SAME)} admission/pump/resume/finalize/matrix surfaces "
            f"byte-identical")


# ===========================================================================
GATES = [
    ("G-UNIT", g_unit),
    ("G-OFFTHREAD", g_offthread),
    ("G-SCOPE", g_scope),
    ("G-ONE", g_one),
    ("G-SF-2", g_sf(2)), ("G-SF-4", g_sf(4)), ("G-SF-25", g_sf(25)),
    ("G-CF-2", g_cf(2)), ("G-CF-4", g_cf(4)), ("G-CF-25", g_cf(25)),
    ("G-RUN4", g_run4),
]

_SD = "RangeMinerCoordinator._serve_dispatch"
_HANDOFF = "self._staging_exec().submit(self._pump_deferred)"
MUTANT_SPECS = [
    ("MU1 repair removed (base behaviour): unit", _SD, _HANDOFF, "None", g_unit),
    ("MU1 repair removed (base behaviour): staging_first x3, 2 workers", _SD, _HANDOFF,
     "None", lambda: _ordered_gate("staging_first", 2, 3)),
    ("MU2 pump called on the dispatch thread", _SD, _HANDOFF, "self._pump_deferred()",
     g_unit),
    ("MU3 hand-off condition inverted (pump only when NOT complete)", _SD,
     "if _sc_check.is_complete:", "if not _sc_check.is_complete:", g_unit),
]


def main():
    print("=" * 78)
    print(f"S172 L-4 STAGING-ADMISSION STALL — gate battery (CPU only, REPS={REPS})")
    print("=" * 78)
    only = set(a for a in sys.argv[1:] if not a.startswith("-"))
    for name, fn in GATES:
        if only and name not in only:
            continue
        t0 = time.time()
        run_gate(name, fn)
        n, ok, det = RESULTS[-1]
        print(f"  [{'PASS' if ok else 'FAIL'}] {n} ({time.time() - t0:.0f}s): {det}",
              flush=True)
    if not only or "--mutants" in sys.argv:
        print("MUTANTS (each must turn its gate RED)")
        for label, qn, old, new, gate in MUTANT_SPECS:
            run_mutant(label, qn, old, new, gate)
            lab, det, txt = MUTANTS[-1]
            print(f"  [{'RED (detected)' if det else 'NOT DETECTED'}] {lab}: {txt}",
                  flush=True)
    g_ok = sum(1 for _, ok, _ in RESULTS if ok)
    m_ok = sum(1 for _, det, _ in MUTANTS if det)
    print(f"GATES {g_ok}/{len(RESULTS)}  MUTANTS {m_ok}/{len(MUTANTS)} detected")
    for n, ok, d in RESULTS:
        if not ok:
            print(f"--- FAIL {n}\n{d}")
    ok = g_ok == len(RESULTS) and m_ok == len(MUTANTS) and not only
    print(f"COMPLETION SENTINEL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
