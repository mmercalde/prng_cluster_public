#!/usr/bin/env python3
"""S172 — ADMISSION BINDING gate suite (Team Beta repairs A, B and C).

Authority: `docs/CLAUDE_CODE_INSTRUCTIONS_ADMISSION_BINDING_REPAIR.md`.
Beta accepted fleet identity and consumer unification at `63e627f` but withheld
Phase-7 closure pending exactly these repairs.

    REPAIR A  the freeze-after-read property was FALSE as implemented.
              `active_execution_set()` incremented `_READS` only when `_ACTIVE`
              was already non-None, so a consumer could read None, take the
              legacy path, and the set could still be frozen afterwards — the
              exact "a consumer already decided without it" sequence the
              submission called structurally impossible. Reads now count on the
              None case too, and the resolver OWNER gets a private,
              non-consuming peek so it does not trip the guard it exists to arm.

    REPAIR B  admission was bound to nothing. The set recorded one count while
              `serve_trial` derived `expected_workers` independently from
              `context["worker_pool_size"]` — two frozen run facts about one run,
              free to disagree, and they did: a local two-GPU set still waited
              for eight workers the set itself declared could not exist.
                  effective = min(requested pool size, selected worker identities)
              Both counts recorded, the clamp logged and in `set_id` provenance,
              and on the miner path `expected_workers` now comes FROM the set.

    REPAIR C  Q1's executable half. `G-LOCAL` proves one selected node is
              verified and that its failure blocks — it calls `fleet_preflight()`
              directly and proves nothing about miner stage admission. This
              suite drives the REAL `serve_trial` over real framed sockets.

WHY A NEW FILE RATHER THAN GATES APPENDED TO AN EXISTING SUITE
    `tests/test_s172_resolved_execution_set.py` (34/34) and
    `tests/test_s172_admission_liveness.py` (16/16) are both cited as
    non-regression figures by committed artifacts. Growing either would silently
    move a number other documents pin. The one gate that HAD to change in place
    is `g_resolve_once_read_then_freeze`, because it asserted the false property
    itself ("a read of an EMPTY set must not block a later freeze") — it is
    corrected there, and the tally stays 34.

VIR-2 DISCIPLINE — every gate below has a clean control and a fault-injection
control, and the fault injection is a REVERT OF THE REPAIR, not a broken input:
  * repair A: restore the `None`-read exemption -> the empty-read gate must red.
  * repair B: revert `_execution_set_expected_workers` to the context value
              -> the two-local-worker stage-assignment gate must red (it goes
              back to waiting for eight).
A gate that stays green under its own revert is measuring nothing.

VIR-3: terminates in PASS | FAIL | UNAVAILABLE | INCOMPLETE.

Run:
    cd ~/distributed_prng_analysis
    source ~/venvs/torch/bin/activate
    PYTHONPATH=. python3 tests/test_s172_admission_binding.py
"""

import ast
import inspect
import logging
import os
import socket
import sys
import textwrap
import threading
import traceback
from typing import List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests"))

import execution_set as XS                                            # noqa: E402
from execution_set import (                                            # noqa: E402
    ExecutionSetError, resolve_execution_set, freeze_execution_set,
    clear_execution_set, active_execution_set, execution_set_scope,
    admission_expectation, execution_set_provenance,
)
import miner.range_miner_coordinator as RMC                            # noqa: E402

# The REAL-wire driver: `_drive` runs the production `serve_trial` against
# loopback workers on genuine framed sockets, with serve_timeout=None, and
# classifies a run that never decides as `still-hung` (a FAILURE, never a pass).
# Reusing it rather than writing a third copy — gate 37 of the Phase-4 suite
# already certifies the worker fixture speaks the genuine protocol.
import test_s172_admission_liveness as AL                              # noqa: E402

_PASS, _FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
_results: List[tuple] = []
_unavailable: List[tuple] = []

LOCAL_HOSTNAME = socket.gethostname()


def _check(name, fn):
    try:
        detail = fn()
        _results.append((name, True, None))
        print(f"  [{_PASS}] {name}" + (f" — {detail}" if detail else ""),
              flush=True)
    except Exception as e:                                            # noqa: BLE001
        _results.append((name, False, traceback.format_exc()))
        print(f"  [{_FAIL}] {name}: {e}", flush=True)


def _unavail(name, why):
    _unavailable.append((name, why))
    print(f"  [UNAVAILABLE] {name} — {why}", flush=True)


def _miner_set(**kw):
    """A set resolved exactly the way `window_optimizer.main()` resolves one."""
    kw.setdefault("backend", "miner")
    kw.setdefault("invoked_by", "window_optimizer.main")
    kw.setdefault("admission_count", 8)          # the DEFAULT worker-pool request
    return resolve_execution_set(**kw)


def _local_set(admission_count=8):
    """`--execution-set-nodes localhost` — Beta's Q1 topology."""
    return _miner_set(declared_nodes=["localhost"],
                      admission_count=admission_count)


class _AdmissionLog(logging.Handler):
    """Capture the coordinator's `[ADMISSION]` line off a REAL serve_trial.

    VIR-1 execution proof: the binding is read back from what the code under
    test emitted while running, not asserted from this file's source.
    """

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.lines: List[str] = []
        self._lock = threading.Lock()

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:                                             # noqa: BLE001
            return
        if ("[ADMISSION]" in msg or "ADMISSION CLAMPED" in msg
                or "BOUND TO THE FROZEN SET" in msg):
            with self._lock:
                self.lines.append(msg)

    def __enter__(self):
        # NOTE the coordinator's logger is named "range_miner_coordinator"
        # (range_miner_coordinator.py:46), NOT __name__ — attaching to the
        # dotted module path captures nothing and would make every read-back
        # assertion below vacuously unreachable.
        for name in ("range_miner_coordinator", "execution_set"):
            lg = logging.getLogger(name)
            lg.addHandler(self)
            lg.setLevel(logging.INFO)
        return self

    def __exit__(self, *exc):
        # NOTE the coordinator's logger is named "range_miner_coordinator"
        # (range_miner_coordinator.py:46), NOT __name__ — attaching to the
        # dotted module path captures nothing and would make every read-back
        # assertion below vacuously unreachable.
        for name in ("range_miner_coordinator", "execution_set"):
            logging.getLogger(name).removeHandler(self)
        return False

    def find(self, needle) -> Optional[str]:
        with self._lock:
            for line in self.lines:
                if needle in line:
                    return line
        return None


# ===========================================================================
# REPAIR A — the freeze-after-read property, made true
# ===========================================================================

def _legacy_active_execution_set():
    """The PRE-REPAIR reader: count a read only when a set is already frozen.

    This is the exemption Beta refuted, restored verbatim as the fault
    injection. It is deliberately written against the module's real globals so
    the mutant differs from the live code in exactly one condition.
    """
    with XS._LOCK:
        if XS._ACTIVE is not None:
            XS._READS += 1
        return XS._ACTIVE


def g_a_empty_read_blocks_freeze():
    """GATE A1 — an EMPTY consumer read refuses a later freeze.

    The case that matters. A consumer that read None did not merely fail to
    learn the fleet; it went on to behave as though no fleet authority existed
    (that is what every `if s is None: <legacy path>` consumer helper does). A
    set frozen after that did not govern the decision.
    """
    clear_execution_set()
    try:
        assert XS.active_execution_set() is None, "precondition: nothing frozen"
        try:
            freeze_execution_set(_miner_set())
        except ExecutionSetError as e:
            assert "already been read" in str(e), str(e)
            return f"empty read -> freeze refused ({str(e).splitlines()[0][:60]}…)"
        raise AssertionError(
            "a freeze SUCCEEDED after a consumer read None. This is the exact "
            "sequence the submission claimed was structurally impossible.")
    finally:
        clear_execution_set()


def g_a_clean_resolve_freeze_passes():
    """GATE A2 — CLEAN CONTROL: resolve and freeze before any read still passes.

    Without this, gate A1 could be satisfied by a module that refuses every
    freeze — which would break both production entrypoints and prove nothing.
    """
    clear_execution_set()
    try:
        s = _miner_set()
        frozen = freeze_execution_set(s)
        assert frozen.set_id() == s.set_id()
        assert XS._READS == 0, (
            f"resolution and freeze must not consume a read; _READS={XS._READS}")
        # and the set is readable afterwards, which is the whole point
        got = active_execution_set()
        assert got is not None and got.set_id() == s.set_id()
        return f"clean freeze accepted, set_id={s.set_id()[:12]}"
    finally:
        clear_execution_set()


def g_a_idempotent_refreeze_after_consumption():
    """GATE A3 — an IDENTICAL re-freeze after consumption is still idempotent.

    The regression risk created by A1: WATCHER and the CLI resolving the same
    inputs in one process must not become a failure just because consumers have
    since read. The idempotent branch returns from `_ACTIVE is not None` before
    `_READS` is consulted at all, which is why consumption cannot break it.
    """
    clear_execution_set()
    try:
        s = _miner_set()
        freeze_execution_set(s)
        for _ in range(5):
            assert active_execution_set() is not None      # heavily consumed
        assert XS._READS >= 5, XS._READS
        again = freeze_execution_set(_miner_set())
        assert again.set_id() == s.set_id(), "identical re-freeze must be a no-op"
        # a DIFFERENT set is still refused, and for the FROZEN reason, not the
        # read reason — the two refusals must not collapse into one.
        try:
            freeze_execution_set(_local_set(admission_count=1))
            raise AssertionError("a different set must not replace the frozen one")
        except ExecutionSetError as e:
            assert "FROZEN for this run" in str(e), str(e)
        return f"idempotent after {XS._READS} reads; different set still refused"
    finally:
        clear_execution_set()


def g_a_resolver_owner_peek_is_private_and_silent():
    """GATE A4 — the resolver owner's peek exists, does not count, and is used.

    Three things, because two of them alone are cosmetic:
      1. `_peek_execution_set()` returns the set without incrementing `_READS`;
      2. WATCHER's `_ensure_execution_set` — the code that PERFORMS the freeze —
         calls it and NOT `active_execution_set()` (AST over the live source,
         never a text match: `2389b61` reverted a fix by whole-block replacement
         and a text anchor would have gone green);
      3. the peek is private (leading underscore) and is not reachable through
         any consumer helper, so it cannot become a quiet bypass.
    """
    clear_execution_set()
    try:
        assert XS._peek_execution_set() is None
        assert XS._READS == 0, "the peek counted an empty read"
        s = freeze_execution_set(_miner_set())
        assert XS._peek_execution_set().set_id() == s.set_id()
        assert XS._READS == 0, f"the peek counted a read; _READS={XS._READS}"

        # (2) the OWNER uses it — located structurally in the live source.
        import agents.watcher_agent as WA
        src = textwrap.dedent(
            inspect.getsource(WA.WatcherAgent._ensure_execution_set))
        tree = ast.parse(src)
        called = {n.func.id for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "_peek_execution_set" in called, (
            f"_ensure_execution_set does not call the non-consuming peek "
            f"(calls: {sorted(called)})")
        assert "active_execution_set" not in called, (
            "_ensure_execution_set still calls active_execution_set(): the "
            "resolver owner would consume a read and then refuse its own freeze")

        # (3) private, and not re-exported by any consumer helper.
        for helper in (XS.filter_config_nodes, XS.is_admitted_worker,
                       XS.admission_expectation, XS.execution_set_provenance,
                       XS.require_execution_set):
            hsrc = inspect.getsource(helper)
            assert "_peek_execution_set" not in hsrc, (
                f"{helper.__name__} uses the owner-only peek — a consumer read "
                f"that does not count is the defect this repair removes")
        return "peek is silent, private, and used by the resolver owner only"
    finally:
        clear_execution_set()


def g_a_fault_injection_none_exemption():
    """GATE A5 — FAULT INJECTION: restore the `None`-read exemption; A1 must red.

    Beta named this control explicitly (§4). The mutant is the one condition
    that was wrong, nothing else.
    """
    original = XS.active_execution_set
    XS.active_execution_set = _legacy_active_execution_set
    try:
        try:
            g_a_empty_read_blocks_freeze()
        except AssertionError as e:
            assert "structurally impossible" in str(e), str(e)
            return "A1 went RED with the None-read exemption restored"
        raise AssertionError(
            "GATE A1 STAYED GREEN with the pre-repair `None`-read exemption "
            "restored — it is VACUOUS and proves nothing about the repair.")
    finally:
        XS.active_execution_set = original
        clear_execution_set()


# ===========================================================================
# REPAIR B — the four clamp cases, and where expected_workers comes from
# ===========================================================================

def g_b_full_fleet_default_eight():
    """CLAMP CASE 1 — full 26-GPU set, default request 8 -> admission 8.

    The clean control for the whole repair: existing behaviour unchanged.
    """
    s = _miner_set()
    assert len(s.worker_ids()) == 26, len(s.worker_ids())
    assert s.requested_admission_count == 8, s.requested_admission_count
    assert s.admission_count == 8, s.admission_count
    assert s.admission_clamped() is False
    assert "CLAMPED" not in s.describe(), s.describe()
    return f"26 identities, requested 8, admission 8, unclamped"


def g_b_local_set_clamped_to_two():
    """CLAMP CASE 2 — local Zeus set, default request 8, two GPUs -> admission 2.

    The defect, in one line: a local two-GPU set used to wait for eight.
    """
    with _AdmissionLog() as log:
        s = _local_set()
    assert s.worker_ids() == (f"{LOCAL_HOSTNAME}:gpu0", f"{LOCAL_HOSTNAME}:gpu1"), \
        s.worker_ids()
    assert s.requested_admission_count == 8, s.requested_admission_count
    assert s.admission_count == 2, s.admission_count
    assert s.admission_clamped() is True
    # visibly logged...
    line = log.find("ADMISSION CLAMPED")
    assert line and "requested 8" in line and "2 worker identities" in line, \
        f"the clamp was not visibly logged: {log.lines}"
    # ...and visible in `describe()`, which is what the CLI banner, the resolve
    # log and the freeze log all print.
    assert "CLAMPED from requested 8" in s.describe(), s.describe()
    return f"requested 8 -> admission 2; logged: {line[:70]}…"


def g_b_explicit_request_one():
    """CLAMP CASE 3 — local set, explicit request 1 -> admission 1.

    Independence control for case 2: the clamp is `min`, not "always the node's
    GPU count". A request BELOW capacity is honoured unchanged.
    """
    s = _local_set(admission_count=1)
    assert s.requested_admission_count == 1 and s.admission_count == 1, s.describe()
    assert s.admission_clamped() is False
    return "requested 1, capacity 2, admission 1 (no clamp)"


def g_b_zero_negative_and_empty_capacity_fail_at_resolution():
    """CLAMP CASE 4 — zero, negative, or a zero-capacity set FAILS at resolution.

    Not at admission time: these are unsatisfiable before anything is
    allocated, and the 180s bounded window (`ee0db06`) exists to bound a fleet
    that is LATE, not to discover one that is arithmetically impossible.
    """
    for bad in (0, -1, -8):
        try:
            _local_set(admission_count=bad)
            raise AssertionError(f"admission_count={bad} was accepted")
        except ExecutionSetError as e:
            assert "not positive" in str(e), str(e)

    # zero capacity: a set whose selected nodes contribute NO worker identities.
    # Injected at the source the resolver reads (`distributed_config.json` GPU
    # counts), so the failure is produced by the real code path rather than by
    # hand-building a ResolvedExecutionSet the resolver would never emit.
    import json
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp()
    try:
        cfg = json.load(open(os.path.join(_ROOT, "distributed_config.json")))
        for n in cfg["nodes"]:
            n["gpu_count"] = 0
        p = os.path.join(tmp, "distributed_config.json")
        json.dump(cfg, open(p, "w"))
        try:
            _miner_set(declared_nodes=["localhost"], admission_count=8,
                       config_path=p)
            raise AssertionError("a zero-capacity set was accepted")
        except ExecutionSetError as e:
            assert "NO worker identities" in str(e), str(e)
            msg = str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return f"0/-1/-8 refused; zero-capacity refused ({msg.splitlines()[0][:48]}…)"


def g_b_both_counts_in_provenance_and_set_id():
    """Both counts are RECORDED and the clamp is in `set_id` provenance.

    Read back from `to_provenance()` — the object run provenance is written
    from — not from the log line that announced it.
    """
    clamped = _local_set()                       # requested 8 -> effective 2
    asked = _local_set(admission_count=2)        # requested 2 -> effective 2

    p = clamped.to_provenance()
    assert p["requested_admission_count"] == 8, p
    assert p["admission_count"] == 2, p
    assert p["admission_clamped"] is True, p
    assert p["worker_identity_count"] == 2, p

    # A run that ASKED for 8 and was clamped to 2 is not the same run as one
    # that asked for 2, even though both admit two workers. If `set_id` could
    # not tell them apart, the clamp would be invisible to provenance.
    assert clamped.set_id() != asked.set_id(), (
        "set_id does not distinguish a clamped run from an unclamped one — the "
        "clamp is not in the set's identity")
    assert clamped.admission_count == asked.admission_count == 2

    # and the frozen set is readable back through the run-provenance accessor
    with execution_set_scope(clamped):
        live = execution_set_provenance()
    assert live["requested_admission_count"] == 8 and live["admission_count"] == 2
    return (f"requested/effective recorded distinctly; set_id "
            f"{clamped.set_id()[:8]} != {asked.set_id()[:8]}")


def g_b_expected_workers_comes_from_the_set():
    """`expected_workers` is sourced from the SET, and context is not a second
    authority — proven through the coordinator's own resolution helper.

    Three arms:
      * a frozen local set: the helper returns the set's 2 even though the
        context still says 8, and names the set as the source;
      * NO set frozen: the context value is returned unchanged (this is what
        keeps every Phase-4 loopback gate green — a pre-existing behaviour, not
        a bypass; production always freezes a set);
      * the binding in `serve_trial` is structurally single and routed through
        the helper (AST over the live source).
    """
    clear_execution_set()
    try:
        with execution_set_scope(_local_set()):
            got, source = RMC._execution_set_expected_workers(8)
            assert got == 2, f"expected_workers came from context, not the set: {got}"
            assert source.startswith("execution_set("), source
        clear_execution_set()
        got2, source2 = RMC._execution_set_expected_workers(8)
        assert got2 == 8 and "no execution set frozen" in source2, (got2, source2)

        # structural: ONE binding, routed through the helper, never re-bound.
        src = open(RMC.__file__, encoding="utf-8").read()
        serve = [n for n in ast.walk(ast.parse(src))
                 if isinstance(n, ast.FunctionDef) and n.name == "serve_trial"]
        assert len(serve) == 1, len(serve)
        binds = []
        for n in ast.walk(serve[0]):
            targets = (n.targets if isinstance(n, ast.Assign)
                       else [n.target] if isinstance(n, (ast.AugAssign, ast.AnnAssign))
                       else [])
            for tgt in targets:
                names = ([e for e in tgt.elts if isinstance(e, ast.Name)]
                         if isinstance(tgt, ast.Tuple) else
                         [tgt] if isinstance(tgt, ast.Name) else [])
                if any(x.id == "expected_workers" for x in names):
                    binds.append(ast.get_source_segment(src, n))
        assert len(binds) == 1, f"expected_workers is bound {len(binds)} times: {binds}"
        assert "_execution_set_expected_workers" in binds[0], binds[0]
        assert "worker_pool_size" in binds[0], binds[0]
        return "set=2 over context=8; no set -> context 8; bound once via the set"
    finally:
        clear_execution_set()


def g_b_forbidden_unchanged():
    """Beta's explicit prohibitions, checked against the live source.

    The 180s admission timeout, `serve_timeout=None` and the Blocker-3 matrix
    are out of scope for this repair, and `distributed_config.json`'s
    bare-metal addresses are deliberate (CLAUDE.md §3).
    """
    import json
    src = open(RMC.__file__, encoding="utf-8").read()
    assert "DEFAULT_WORKER_ADMISSION_TIMEOUT = 180.0" in src, \
        "the 180s admission timeout changed"
    assert RMC.DEFAULT_WORKER_ADMISSION_TIMEOUT == 180.0, \
        RMC.DEFAULT_WORKER_ADMISSION_TIMEOUT
    assert '_timeout_raw = context.get("serve_timeout", None)' in src
    assert '"serve_timeout": kwargs.get("serve_timeout", None),' in src

    # The Blocker-3 matrix functions are untouched by THIS repair: compare each
    # against the same function as of the last commit.
    import subprocess
    head = subprocess.run(["git", "show", "HEAD:miner/range_miner_coordinator.py"],
                          cwd=_ROOT, capture_output=True, text=True)
    assert head.returncode == 0, head.stderr

    def _fn(s, name):
        hits = [n for n in ast.walk(ast.parse(s))
                if isinstance(n, ast.FunctionDef) and n.name == name]
        assert len(hits) == 1, f"{name}: {len(hits)}"
        return ast.get_source_segment(s, hits[0])

    for fn in ("handle_stripe_failure", "_handle_stripe_failure_locked",
               "_pick_other_worker", "process_lease_expiry"):
        assert _fn(head.stdout, fn) == _fn(src, fn), \
            f"{fn} changed — the Blocker-3 matrix must be untouched"

    cfg = json.load(open(os.path.join(_ROOT, "distributed_config.json")))
    assert [n["hostname"] for n in cfg["nodes"]] == [
        "localhost", "192.168.3.120", "192.168.3.154", "192.168.3.162"], \
        "distributed_config.json's addresses were modified — they are deliberate"
    return "180s timeout, serve_timeout=None, Blocker-3 matrix, addresses unchanged"


# ===========================================================================
# REPAIR C — the six-point miner-path gate, over the REAL serve_trial
# ===========================================================================
# `G-LOCAL` calls `fleet_preflight()` directly. Everything below drives
# `RangeMinerCoordinator.serve_trial` itself, with `serve_timeout=None`, so a
# run that never decides is `still-hung` — a failure, never a pass.

_C_RESULTS = {}

#: `assign_stripes` splits `total_seeds` by `config.miner_stripe_size`
#: (67_108_864, range_miner_coordinator.py:1944), so the 30-seed sizing every
#: existing miner harness uses produces exactly ONE stripe: at most one worker
#: can receive an assignment however many are admitted.
#:
#: A multi-stripe sizing was tried here and REJECTED as a harness change, not
#: adopted and not worked around: a 2-worker / 2-stripe loopback run does not
#: terminate, leaving shards at `staging_status='pending'`. That reproduces with
#: NO execution set frozen and with this repair's code path never entered, so it
#: is independent of admission binding — pre-existing behaviour of the loopback
#: fixture and/or the staging executor on multi-stripe runs. Whether it is a
#: fixture limitation (`_FakeWorker` answers one sub-stripe per stripe) or a
#: production defect is [UNVERIFIED] and out of this brief's scope; it is
#: recorded in the report rather than silently absorbed.
#:
#: CONSEQUENCE FOR WHAT THESE GATES PROVE. "Both workers were admitted" is
#: therefore NOT asserted from "both received a stripe". It is asserted from
#: what actually decides admission: the coordinator only assigns a stage when
#: `len(eligible) >= expected_workers`, so a committed single-stripe trial at
#: `expected_workers=2` IS proof that two workers were admitted — and C5 is the
#: control, where one worker under the same set refuses.
_SINGLE_STRIPE_SEEDS = 30


def _local_specs(n, behavior="complete"):
    return [(LOCAL_HOSTNAME, i, behavior, 0.0) for i in range(n)]


def _drive_with_set(s, *, worker_specs, requested_pool, admission_timeout=20.0,
                    budget=60.0, serve_timeout=None):
    """Drive the production serve loop under a FROZEN set, capturing the
    coordinator's admission log so the binding is read back off the real run."""
    with execution_set_scope(s), _AdmissionLog() as log:
        o = AL._drive(RMC, worker_specs=worker_specs,
                      expected_workers=requested_pool,
                      admission_timeout=admission_timeout,
                      lease_timeout=120.0, serve_timeout=serve_timeout,
                      total_seeds=_SINGLE_STRIPE_SEEDS,
                      budget=budget)
    return o, log


def g_c1_local_resolves_two_identities():
    """C1 — `--execution-set-nodes localhost` resolves TWO eligible identities.

    Also proves the CLI flag reaches `declared_nodes` rather than being parsed
    and dropped, located structurally in `window_optimizer.main()`.
    """
    s = _local_set()
    assert s.node_ids() == ("localhost",), s.node_ids()
    assert s.worker_ids() == (f"{LOCAL_HOSTNAME}:gpu0", f"{LOCAL_HOSTNAME}:gpu1"), \
        s.worker_ids()
    assert s.remote_execution is False, "a local-only set must derive remote=False"

    src = open(os.path.join(_ROOT, "window_optimizer.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    main = [n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "main"]
    assert len(main) == 1
    calls = [n for n in ast.walk(main[0])
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)
             and n.func.id == "_resolve_xset"]
    assert len(calls) == 1, f"{len(calls)} resolver calls in main()"
    kw = {k.arg for k in calls[0].keywords}
    assert {"declared_nodes", "admission_count", "backend"} <= kw, kw
    return f"localhost -> 2 identities {s.worker_ids()}, remote_execution=False"


def g_c2_default_request_bounded_to_two():
    """C2 — the DEFAULT worker-pool request is bounded to effective count 2."""
    s = _local_set()                              # admission_count=8, the default
    assert s.requested_admission_count == 8 and s.admission_count == 2, s.describe()
    with execution_set_scope(s):
        got, source = RMC._execution_set_expected_workers(8)
    assert got == 2 and source.startswith("execution_set("), (got, source)
    return "default request 8 -> effective admission 2 on the miner path"


def g_c3_two_local_workers_cause_assignment():
    """C3 — TWO local workers cause stage assignment, without waiting for eight.

    The executable half Beta said was missing. The run requests the production
    default of 8; the frozen local set bounds it to 2; two loopback workers
    register and the stage is ASSIGNED, dispatched and COMMITTED — with
    `serve_timeout=None`, so nothing but the code's own terminal decision can
    end it. Before this repair the same run sat in the bounded admission window
    waiting for six workers the set had already declared could not exist, and
    ended in `worker admission timeout`.
    """
    o, log = _drive_with_set(_local_set(), worker_specs=_local_specs(2),
                             requested_pool=8, admission_timeout=20.0)
    _C_RESULTS["c3"] = o
    assert not o.still_hung, f"the serve loop hung: {o.summary()}"
    assert o.error is None, o.error
    assert o.committed, (
        f"two local workers did not reach a committed trial: {o.summary()} "
        f"reasons={o.abort_reasons}")
    assert not any("admission timeout" in r for r in o.abort_reasons), \
        f"the run still waited for a pool it could never reach: {o.abort_reasons}"
    # A stage WAS assigned — the coordinator only does that at
    # `len(eligible) >= expected_workers`, so this is the admission evidence.
    assert sum(len(w.assigns_received) for w in o.workers) >= 1, \
        "no stripe was ever dispatched: the stage was never assigned"
    for w in o.workers:
        assert w.worker_id.startswith(LOCAL_HOSTNAME), w.worker_id

    # VIR-1 execution proof: the number came from the SET, read back off the
    # line the coordinator emitted while running.
    line = log.find("[ADMISSION]")
    assert line, f"no admission line was emitted: {log.lines}"
    assert "expected_workers=2" in line, line
    assert "source=execution_set(" in line, line
    # …and the set-side line names the REQUEST it overrode, so the two numbers
    # are both readable off the run rather than one silently replacing the other.
    bound = log.find("BOUND TO THE FROZEN SET")
    assert bound and "worker_pool_size=8" in bound, (
        f"the overridden request was not recorded: {log.lines}")
    return (f"assigned+committed with 2 workers in {o.elapsed:.1f}s; "
            f"{line.split('run ')[-1][:72]}…")


def g_c4_unlisted_third_worker_quarantined():
    """C4 — an unlisted THIRD worker remains quarantined while the run proceeds.

    The clamp lowers the COUNT; it does not lower the BAR. Two arms, because
    the first alone is weak — a stranger receiving no work proves little when
    there is one stripe and two listed workers ahead of it:

      * CLEAN — two listed workers plus a stranger: the run commits, and the
        stranger is dispatched nothing.
      * DISCRIMINATING — ONE listed worker plus a stranger, against the same
        set requiring two. Two workers are connected and well-formed. If the
        stranger counted, admission would be satisfied and the stage assigned.
        It must instead fail naming **1 admitted**, which is only true if the
        unlisted worker never entered the eligible pool.
    """
    specs = _local_specs(2) + [("stranger-rig", 0, "complete", 0.0)]
    o, log = _drive_with_set(_local_set(), worker_specs=specs,
                             requested_pool=8, admission_timeout=20.0)
    _C_RESULTS["c4"] = o
    assert not o.still_hung, f"the serve loop hung: {o.summary()}"
    assert o.committed, f"the run did not commit: {o.summary()} {o.abort_reasons}"
    by_id = {w.worker_id: w for w in o.workers}
    stranger = by_id["stranger-rig:gpu0"]
    assert not stranger.assigns_received, (
        f"an UNLISTED worker was assigned work: {stranger.assigns_received}")

    o2, _ = _drive_with_set(
        _local_set(),
        worker_specs=_local_specs(1) + [("stranger-rig", 0, "complete", 0.0)],
        requested_pool=8, admission_timeout=6.0)
    assert not o2.still_hung, o2.summary()
    reasons = " | ".join(o2.abort_reasons)
    assert "expected 2 eligible worker(s), 1 admitted after" in reasons, (
        f"the unlisted worker COUNTED toward admission — membership was earned "
        f"by connecting: {reasons}")
    assert not o2.committed
    return ("stranger dispatched nothing; and with 2 connected but only 1 "
            "listed, admission still reports 1 admitted")


def g_c5_missing_capacity_hits_the_existing_failure():
    """C5 — missing required local capacity reaches the EXISTING bounded
    admission failure (`ee0db06`), not a second failure mode.

    One of the two listed local workers never arrives. The set says 2, one
    shows up, and the run must fail through the same path and the same reason
    text that the §4.3 repair introduced — bounded, explicit, diagnosable.
    """
    o, log = _drive_with_set(_local_set(), worker_specs=_local_specs(1),
                             requested_pool=8, admission_timeout=6.0)
    _C_RESULTS["c5"] = o
    assert not o.still_hung, (
        f"missing capacity HUNG instead of failing: {o.summary()}")
    assert o.ended_by == "own-decision", o.summary()
    reasons = " | ".join(o.abort_reasons)
    assert "worker admission timeout" in reasons, (
        f"a DIFFERENT failure mode was introduced: {reasons}")
    # the existing reason format, unchanged: run · stage · family · phase ·
    # expected · admitted · elapsed
    assert "expected 2 eligible worker(s), 1 admitted after" in reasons, reasons
    assert "worker_admission_timeout=6.0s" in reasons, reasons
    assert not o.committed
    return f"bounded admission failure, existing path: …{reasons[-96:]}"


def g_c6_full_fleet_default_eight_unchanged():
    """C6 — full-fleet / default-eight behaviour is unchanged.

    Two arms, because either alone is weak:
      * CLEAN — a full 26-GPU set, request 8, eight listed workers: assigned
        and committed exactly as before this repair;
      * NEGATIVE — the same set with only two workers still waits for EIGHT
        and fails naming eight. The clamp must not have quietly lowered the
        full-fleet threshold to whatever showed up (which would be the
        "inferred from who answered" defect wearing this repair's clothes).
    """
    full = _miner_set()
    assert full.admission_count == 8 and full.requested_admission_count == 8

    specs = [("rrig6600", i, "complete", 0.0) for i in range(8)]
    o, log = _drive_with_set(full, worker_specs=specs, requested_pool=8,
                             admission_timeout=30.0, budget=90.0)
    _C_RESULTS["c6"] = o
    assert not o.still_hung, f"the full-fleet arm hung: {o.summary()}"
    assert o.committed, f"full fleet did not commit: {o.summary()} {o.abort_reasons}"
    line = log.find("[ADMISSION]")
    assert line and "expected_workers=8" in line, line

    o2, _ = _drive_with_set(full, worker_specs=[("rrig6600", i, "complete", 0.0)
                                                for i in range(2)],
                            requested_pool=8, admission_timeout=6.0)
    assert not o2.still_hung, o2.summary()
    reasons2 = " | ".join(o2.abort_reasons)
    assert "expected 8 eligible worker(s), 2 admitted after" in reasons2, reasons2
    return ("8 listed workers -> committed at expected_workers=8; "
            "2 workers -> still refuses, naming 8")


def g_c_fault_injection_unbind_admission():
    """FAULT INJECTION for repair B/C — revert `expected_workers` to the raw
    context value; C3 must go red.

    This is the pre-repair line, restored: `serve_trial` derives the count from
    `context["worker_pool_size"]` and ignores the set. C3 then waits for eight
    local workers that the set itself refuses to admit, and dies in the bounded
    admission window — which is exactly the defect Beta named.
    """
    original = RMC._execution_set_expected_workers
    _live_c3 = _C_RESULTS.get("c3")
    RMC._execution_set_expected_workers = (
        lambda pool: (int(pool), "context(PRE-REPAIR MUTANT)"))
    try:
        try:
            g_c3_two_local_workers_cause_assignment()
        except AssertionError as e:
            assert "admission" in str(e).lower() or "commit" in str(e).lower(), str(e)
            return f"C3 went RED with admission unbound from the set: {str(e)[:88]}…"
        raise AssertionError(
            "GATE C3 STAYED GREEN with expected_workers reverted to the raw "
            "context value — it is VACUOUS and proves nothing about the repair.")
    finally:
        RMC._execution_set_expected_workers = original
        # restore the LIVE C3 outcome: the mutant run overwrote it, and the
        # anti-vacuity summary must report the real run, not the injected one.
        if _live_c3 is not None:
            _C_RESULTS["c3"] = _live_c3
        clear_execution_set()


def g_c_mutant_summary():
    """Anti-vacuity roll-up: the miner-path gates ran REAL trials, not stubs."""
    for k in ("c3", "c4", "c5", "c6"):
        o = _C_RESULTS.get(k)
        assert o is not None, f"{k} did not execute"
        assert o.serve_timeout is None, (
            f"{k} ran with an injected wall clock ({o.serve_timeout}s) — a run "
            f"that can be ended by the harness is not evidence of a decision")
        assert not o.still_hung, f"{k} hung"
    assert _C_RESULTS["c3"].run_id and _C_RESULTS["c5"].run_id
    return (f"{len(_C_RESULTS)} real serve_trial runs, serve_timeout=None in "
            f"every one, all terminated by their own decision")


# ===========================================================================
def main():
    print("=" * 74)
    print("S172 — ADMISSION BINDING gate suite (Beta repairs A / B / C)")
    print(f"repo: {_ROOT}")
    print(f"host: {LOCAL_HOSTNAME}")
    print("live arm: serve_timeout=None (production default) — no wall clock")
    print("          can end a miner-path run; the budget is a FAILURE signal")
    print("=" * 74)

    clear_execution_set()

    print("\n-- REPAIR A: the freeze-after-read property, made true --")
    _check("A1: an EMPTY consumer read refuses a later freeze",
           g_a_empty_read_blocks_freeze)
    _check("A2: CLEAN CONTROL — resolve/freeze before any read passes",
           g_a_clean_resolve_freeze_passes)
    _check("A3: an identical re-freeze after consumption is still idempotent",
           g_a_idempotent_refreeze_after_consumption)
    _check("A4: the resolver owner's peek is silent, private, and used",
           g_a_resolver_owner_peek_is_private_and_silent)
    _check("A5: FAULT INJECTION — restoring the None-read exemption reds A1",
           g_a_fault_injection_none_exemption)

    print("\n-- REPAIR B: admission bound to the frozen set --")
    _check("B1 (clamp case 1): full 26-GPU set, request 8 -> admission 8",
           g_b_full_fleet_default_eight)
    _check("B2 (clamp case 2): local set, request 8, two GPUs -> admission 2",
           g_b_local_set_clamped_to_two)
    _check("B3 (clamp case 3): local set, explicit request 1 -> admission 1",
           g_b_explicit_request_one)
    _check("B4 (clamp case 4): zero / negative / zero-capacity fail at resolution",
           g_b_zero_negative_and_empty_capacity_fail_at_resolution)
    _check("B5: both counts recorded; the clamp is in set_id provenance",
           g_b_both_counts_in_provenance_and_set_id)
    _check("B6: expected_workers comes from the SET, not from context",
           g_b_expected_workers_comes_from_the_set)
    _check("B7: 180s admission timeout / serve_timeout / Blocker-3 matrix "
           "unchanged", g_b_forbidden_unchanged)

    print("\n-- REPAIR C: the six-point miner-path gate (REAL serve_trial) --")
    _check("C1: --execution-set-nodes localhost resolves two eligible identities",
           g_c1_local_resolves_two_identities)
    _check("C2: the default pool request is bounded to effective admission 2",
           g_c2_default_request_bounded_to_two)
    _check("C3: two local workers cause stage assignment, without waiting for 8",
           g_c3_two_local_workers_cause_assignment)
    _check("C4: an unlisted third worker remains quarantined",
           g_c4_unlisted_third_worker_quarantined)
    _check("C5: missing local capacity reaches the EXISTING bounded admission "
           "failure", g_c5_missing_capacity_hits_the_existing_failure)
    _check("C6: full-fleet / default-eight behaviour unchanged",
           g_c6_full_fleet_default_eight_unchanged)
    _check("C-MUTANT: FAULT INJECTION — unbinding admission reds C3",
           g_c_fault_injection_unbind_admission)
    _check("C-MUTANT: summary — the miner-path gates ran real trials",
           g_c_mutant_summary)

    clear_execution_set()

    print("=" * 74)
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{passed}/{total} admission-binding checks green")
    if _unavailable:
        print(f"{len(_unavailable)} UNAVAILABLE (not exercised, not assumed):")
        for name, why in _unavailable:
            print(f"  - {name}: {why}")
    if passed != total:
        print("\nFAILURES (DO NOT COMMIT):")
        for name, ok, tb in _results:
            if not ok:
                print(f"\n--- {name} ---\n{tb}")
        print("\nRESULT: FAIL")
        return 1
    print("\nRESULT: PASS" + (" (with UNAVAILABLE surfaces declared)"
                             if _unavailable else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
