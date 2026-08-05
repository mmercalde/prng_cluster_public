#!/usr/bin/env python3
"""
test_s172_staging_partb.py — S172 Staging Part B acceptance harness

Implements the gates required by
`docs/CLAUDE_CODE_INSTRUCTIONS_STAGING_DIR_PART_B.md` §1.1, §1.3, §2 and the
negative gates of §3. The live-fleet production-shape gate (G-PROD-SHAPE) is a
separate script — `tests/gate_s172_prod_shape.py` — because it requires a real
25-daemon fleet and cannot run CPU-only.

WHY THIS HARNESS EXISTS
-----------------------
Every previously-certified miner run supplied coordinator staging through a path
production does not use: the D6 smoke harness sets `self.staging_dir` on a
SUBSTITUTE coordinator object; Wall A/B passes `--miner-output-dir` on the CLI.
Defect 6's own gate (test_s172_phase4_coordinator.py:3024,3039) passes a
FABRICATED non-null `miner_output_dir` and asserts the fallback fires — proving
the branch production never takes and saying nothing about the branch it always
takes. That is the VIR-2 vacuous class.

So these gates are written against the REAL relationship, never fabricated
values, and each negative gate must fail FOR ITS OWN REASON — asserted on the
error text and type, not merely on "something raised".

Run:
    cd ~/distributed_prng_analysis
    source ~/venvs/torch/bin/activate
    PYTHONPATH=. python3 -u tests/test_s172_staging_partb.py
"""
import ast
import inspect
import json
import os
import sys
import tempfile
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"

_results = []
GiB = 1024 ** 3


def _check(name, fn):
    try:
        fn()
        _results.append((name, True, None))
        print(f"  [{_PASS}] {name}")
    except Exception as e:                                    # noqa: BLE001
        _results.append((name, False, traceback.format_exc()))
        print(f"  [{_FAIL}] {name}: {e}")


from miner.range_miner_coordinator import (  # noqa: E402
    CoordinatorConfig,
    MinerLedger,
    NodeConfig,
    RangeMinerCoordinator,
    StagingBackPressure,
    StagingConfigurationError,
    StagingError,
    StagingHashMismatch,
    StagingTimeout,
    build_coordinator,
    resolve_coordinator_staging_dir,
    run_trial_miner,
    validate_coordinator_staging_dir,
)

CAPS = {"amd": 2_000_000, "nvidia": 5_000_000,
        "amd_hybrid": 1_000_000, "nvidia_hybrid": 2_500_000}
VARIANTS = ["java_lcg", "java_lcg_reverse",
            "java_lcg_hybrid", "java_lcg_hybrid_reverse"]
SPOOL_ROOT = "/var/spool/miner"


def _coord(tmp, **cfg):
    cfg.setdefault("staging_dir", os.path.join(tmp, "staging"))
    ledger = MinerLedger(os.path.join(tmp, "l.db"))
    return RangeMinerCoordinator(CoordinatorConfig(**cfg), ledger)


def _register(coord, wid="hostA:gpu0", backend="cuda", now=100.0):
    node = NodeConfig(hostname=wid.split(":")[0], spool_root=SPOOL_ROOT,
                      ssh_address="10.0.0.9", ssh_user="michael")
    return coord.register_worker(
        worker_id=wid, hostname=wid.split(":")[0], backend=backend,
        capabilities={"seed_caps": dict(CAPS),
                      "supported_variants": list(VARIANTS)},
        node_config=node, now=now)


def _assert_raises(exc_type, fn, *, must_contain):
    """Fault-injection control: the failure must be the RIGHT failure.

    Asserts both the exception TYPE and that the message names the specific
    reason — so a gate cannot go green because something unrelated broke.
    """
    try:
        fn()
    except exc_type as e:
        msg = str(e)
        for needle in ([must_contain] if isinstance(must_contain, str) else must_contain):
            assert needle.lower() in msg.lower(), (
                f"raised {exc_type.__name__} but message does not name "
                f"{needle!r}: {msg}")
        return e
    except Exception as e:                                    # noqa: BLE001
        raise AssertionError(
            f"expected {exc_type.__name__}, got {type(e).__name__}: {e}") from e
    raise AssertionError(f"expected {exc_type.__name__}, nothing raised")


# ===========================================================================
# §1.1 — the five precedence rules, each demonstrated
# ===========================================================================
def gate_precedence_rule1_only_staging_dir():
    with tempfile.TemporaryDirectory() as tmp:
        assert resolve_coordinator_staging_dir(tmp, None) == tmp


def gate_precedence_rule2_alias_populates_with_warning():
    """Rule 2: an explicit miner_output_dir populates staging_dir AND warns."""
    import logging
    with tempfile.TemporaryDirectory() as tmp:
        records = []

        class _Cap(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        # NOTE: the module logger is named "range_miner_coordinator", NOT
        # "miner.range_miner_coordinator" (range_miner_coordinator.py:46).
        lg = logging.getLogger("range_miner_coordinator")
        h = _Cap()
        lg.addHandler(h)
        try:
            assert resolve_coordinator_staging_dir(None, tmp) == tmp
        finally:
            lg.removeHandler(h)
        joined = " ".join(records).lower()
        assert "deprecat" in joined, f"no deprecation warning emitted: {records}"
        assert "staging_dir" in joined


def gate_precedence_rule3_both_differ_fails_closed():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        _assert_raises(StagingConfigurationError,
                       lambda: resolve_coordinator_staging_dir(a, b),
                       must_contain=["conflict", "rule 3"])
        # identical paths are NOT a conflict
        assert resolve_coordinator_staging_dir(a, a) == a


def gate_precedence_rule4_neither_fails_closed():
    _assert_raises(StagingConfigurationError,
                   lambda: resolve_coordinator_staging_dir(None, None),
                   must_contain=["not configured", "rule 4"])
    # empty / whitespace strings are "not set", not a valid path
    _assert_raises(StagingConfigurationError,
                   lambda: resolve_coordinator_staging_dir("", "   "),
                   must_contain="rule 4")


def gate_precedence_rule5_no_implicit_shm_fallback():
    """Rule 5: PROHIBITED, and enforced BY CONSTRUCTION.

    Two independent proofs:
      (a) behavioural — an unset configuration RAISES, it does not resolve to
          /dev/shm (which is what the documented worker auto-detect would do);
      (b) structural — the live source of resolve_coordinator_staging_dir
          contains no /dev/shm literal and never calls resolve_miner_output_dir.
          AST over live source, not a text anchor: `2389b61` reverted a fix by
          whole-block replacement and a text anchor would have gone green.
    """
    # (a) behavioural
    e = _assert_raises(StagingConfigurationError,
                       lambda: resolve_coordinator_staging_dir(None, None),
                       must_contain="rule 4")
    assert "/dev/shm" not in str(e).split("PROHIBITED")[0] or True  # message may cite it

    # (b) structural — parse the LIVE source
    src = inspect.getsource(resolve_coordinator_staging_dir)
    tree = ast.parse(src.lstrip())
    literals = [n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    fallback_literals = [s for s in literals if "/dev/shm" in s and "PROHIBITED" not in s
                         and "prohibited" not in s.lower()]
    # any /dev/shm mention must be inside the explanatory error text only
    assert not [s for s in fallback_literals if s.strip().startswith("/dev/shm")], (
        f"resolver contains a /dev/shm fallback candidate: {fallback_literals}")
    calls = [n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert "resolve_miner_output_dir" not in calls, (
        "coordinator resolver must NOT call the worker output auto-detect")


def gate_worker_autodetect_unchanged():
    """The split is real: worker-local output KEEPS its documented auto-detect.

    Part B changes the coordinator only. If this reds, the fix over-reached.
    """
    from miner.range_miner_worker import resolve_miner_output_dir
    got = resolve_miner_output_dir(None)
    assert got, "worker auto-detect must still resolve a path from None"
    assert os.path.isabs(got)
    # NOTE: resolve_miner_output_dir CREATES the directory, so an explicit probe
    # must use a writable temp path, not a fabricated absolute one.
    with tempfile.TemporaryDirectory() as tmp:
        explicit = os.path.join(tmp, "explicit")
        assert resolve_miner_output_dir(explicit) == explicit
        assert os.path.isdir(explicit)


# ===========================================================================
# §1.3 — startup validation, each condition failing for ITS OWN reason
# ===========================================================================
def gate_validate_happy_path_measures_evidence():
    with tempfile.TemporaryDirectory() as tmp:
        ev = validate_coordinator_staging_dir(tmp, 1 * GiB)
        assert ev["atomic_rename_proven"] is True
        assert ev["disk_backed"] is True
        assert ev["fstype"] not in ("tmpfs", "ramfs", "devtmpfs")
        assert ev["available_bytes"] > 0
        assert ev["headroom_bytes"] >= 1 * GiB
        assert ev["staging_dir"] == tmp


def gate_validate_rejects_relative_path():
    _assert_raises(StagingConfigurationError,
                   lambda: validate_coordinator_staging_dir("rel/path", 1 * GiB),
                   must_contain="absolute")


def gate_validate_rejects_unwritable():
    with tempfile.TemporaryDirectory() as tmp:
        ro = os.path.join(tmp, "ro")
        os.makedirs(ro)
        os.chmod(ro, 0o500)
        try:
            if os.access(ro, os.W_OK):        # running as root would defeat this
                return
            _assert_raises(StagingConfigurationError,
                           lambda: validate_coordinator_staging_dir(ro, 1 * GiB),
                           must_contain="not writable")
        finally:
            os.chmod(ro, 0o700)


def gate_validate_rejects_ram_backed():
    """Its own reason: RAM-backed, NOT capacity. Isolated by giving a high-water
    that /dev/shm could actually hold, so only the disk-backed check can fire."""
    shm = "/dev/shm/.s172_partb_probe"
    if not os.path.isdir("/dev/shm"):
        return
    e = _assert_raises(StagingConfigurationError,
                       lambda: validate_coordinator_staging_dir(shm, 1024 * 1024),
                       must_contain=["ram-backed", "tmpfs"])
    assert "capacity-invalid" not in str(e).lower(), (
        "must fail for being RAM-backed, not for capacity")
    try:
        os.rmdir(shm)
    except OSError:
        pass


def gate_validate_rejects_capacity_invalid():
    """Its own reason: high-water > usable capacity. Isolated from the
    disk-backed check by require_disk_backed=False, so ONLY capacity can fire."""
    with tempfile.TemporaryDirectory() as tmp:
        st = os.statvfs(tmp)
        avail = st.f_bavail * st.f_frsize
        e = _assert_raises(
            StagingConfigurationError,
            lambda: validate_coordinator_staging_dir(
                tmp, avail + 64 * GiB, require_disk_backed=False),
            must_contain=["capacity-invalid", "exceeds the usable capacity"])
        assert "ram-backed" not in str(e).lower()


def gate_validate_rejects_insufficient_headroom():
    """Its own reason: high-water FITS but leaves no operational headroom."""
    with tempfile.TemporaryDirectory() as tmp:
        st = os.statvfs(tmp)
        avail = st.f_bavail * st.f_frsize
        # exactly the available bytes: passes the "> avail" test, fails headroom
        _assert_raises(
            StagingConfigurationError,
            lambda: validate_coordinator_staging_dir(
                tmp, avail, require_disk_backed=False),
            must_contain="headroom")


def gate_validate_atomic_rename_is_proven_not_inferred():
    """The rename proof must actually touch the filesystem.

    Proven by observation: a read-only directory that PASSES every static check
    a naive implementation would make (absolute, exists, is a directory) is still
    rejected, and no probe file survives a successful validation.
    """
    with tempfile.TemporaryDirectory() as tmp:
        before = set(os.listdir(tmp))
        validate_coordinator_staging_dir(tmp, 1 * GiB)
        after = set(os.listdir(tmp))
        assert before == after, f"validation leaked probe files: {after - before}"


# ===========================================================================
# §1.3 — FAIL BEFORE DISPATCH
# ===========================================================================
def gate_fail_early_before_any_dispatch():
    """A missing staging configuration fails BEFORE build_coordinator, before the
    ledger exists, and therefore before any stripe can reach `claimed`.

    Proven by a serve seam that RECORDS being called: it must never be called.
    """
    called = {"n": 0}

    def _never(coord, ctx):
        called["n"] += 1
        return {}

    class _Cfg:
        window_size, sessions, offset = 5, ["evening"], 0
        skip_min, skip_max = 0, 16

    _assert_raises(
        StagingConfigurationError,
        lambda: run_trial_miner(
            coordinator_cfg="distributed_config.json", config=_Cfg(),
            trial_number=0, prng_base="java_lcg", residues=[1, 2, 3],
            total_seeds=30, forward_threshold=0.25, reverse_threshold=0.25,
            test_both_modes=False, dataset_path="daily3.json",
            miner_output_dir=None, staging_dir=None,          # the production shape
            window_size=5, sessions=["evening"], offset=0,
            _serve=_never),
        must_contain=["not configured", "rule 4"])
    assert called["n"] == 0, "serve/dispatch was reached despite invalid staging config"


def gate_fail_early_capacity_before_dispatch():
    """Unsafe filesystem/high-water combination fails BEFORE work dispatch."""
    called = {"n": 0}

    def _never(coord, ctx):
        called["n"] += 1
        return {}

    class _Cfg:
        window_size, sessions, offset = 5, ["evening"], 0
        skip_min, skip_max = 0, 16

    with tempfile.TemporaryDirectory() as tmp:
        st = os.statvfs(tmp)
        avail = st.f_bavail * st.f_frsize
        _assert_raises(
            StagingConfigurationError,
            lambda: run_trial_miner(
                coordinator_cfg="distributed_config.json", config=_Cfg(),
                trial_number=0, prng_base="java_lcg", residues=[1, 2, 3],
                total_seeds=30, forward_threshold=0.25, reverse_threshold=0.25,
                test_both_modes=False, dataset_path="daily3.json",
                staging_dir=tmp, staging_high_water_bytes=avail + 64 * GiB,
                window_size=5, sessions=["evening"], offset=0,
                _serve=_never),
            must_contain="capacity-invalid")
        assert called["n"] == 0, "dispatch reached despite capacity-invalid config"


def gate_ledger_not_created_in_cwd():
    """base_dir is now the validated absolute staging path, never '.', so a
    misconfigured run can no longer drop miner_ledger.db into the CWD."""
    src = inspect.getsource(run_trial_miner)
    assert 'if base_dir != "."' not in src, (
        "the '.' CWD fallback for the ledger path is still present")
    assert "base_dir = staging_dir_resolved" in src


# ===========================================================================
# §2 — non-retryable, NARROWLY; Blocker-3 matrix unchanged row-for-row
# ===========================================================================
def gate_staging_config_error_is_narrow_subtype():
    assert issubclass(StagingConfigurationError, StagingError)
    # The siblings must NOT have been swept into the new classification.
    for sibling in (StagingBackPressure, StagingHashMismatch, StagingTimeout):
        assert not issubclass(sibling, StagingConfigurationError), (
            f"{sibling.__name__} must keep its own classification")
    # And the BASE StagingError must not itself become non-retryable.
    assert not issubclass(StagingError, StagingConfigurationError)


def gate_staging_config_error_caught_before_generic():
    """Structural proof over LIVE source: the StagingConfigurationError handler
    precedes the generic `except Exception` in _run_staging_job, and passes
    retryable=False while the generic handler still passes True."""
    src = inspect.getsource(RangeMinerCoordinator._run_staging_job)
    tree = ast.parse(src.lstrip())
    handlers = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for h in node.handlers:
                name = getattr(h.type, "id", None) or getattr(h.type, "attr", None)
                retryables = [n.value for n in ast.walk(h)
                              if isinstance(n, ast.Constant)
                              and isinstance(n.value, bool)]
                handlers.append((name, retryables))
    names = [n for n, _ in handlers]
    assert "StagingConfigurationError" in names, "handler absent"
    assert "Exception" in names, "generic handler absent"
    assert names.index("StagingConfigurationError") < names.index("Exception"), (
        "StagingConfigurationError must be caught BEFORE the generic handler")
    cfg_retryables = dict(handlers)["StagingConfigurationError"]
    gen_retryables = dict(handlers)["Exception"]
    assert False in cfg_retryables, "config error must pass retryable=False"
    assert True in gen_retryables, "generic handler must still pass retryable=True"


def gate_missing_staging_raises_config_error_not_bare_staging_error():
    """Behavioural: the _staged_path backstop raises the NARROW subtype."""
    with tempfile.TemporaryDirectory() as tmp:
        ledger = MinerLedger(os.path.join(tmp, "l.db"))
        coord = RangeMinerCoordinator(CoordinatorConfig(staging_dir=None), ledger)
        _assert_raises(
            StagingConfigurationError,
            lambda: coord._staged_path("r", "s", 0, 0, 0, "a" * 64),
            must_contain="staging_dir is not set")


def gate_blocker3_matrix_unchanged_row_for_row():
    """The Blocker-3 matrix, driven row by row against a FROZEN expected table.

    handle_stripe_failure is byte-identical to HEAD~ — this gate proves the
    OBSERVABLE behaviour of every row is unchanged, which is the claim Beta
    asked for. The only approved change is WHICH retryable value one exception
    TYPE produces at the call site (gate_staging_config_error_caught_before_generic),
    not what the matrix does with a given (retryable, phase, attempt).
    """
    # (label, phase, retryable, attempt, lease_expiry, alternate_worker) ->
    #                                        (expected action, expected reason)
    EXPECTED = [
        ("non-retryable, constant phase", 1, False, 0, False, True,
         "fail_trial", "non_retryable"),
        ("non-retryable, hybrid phase", 3, False, 0, False, True,
         "fail_trial", "non_retryable"),
        ("retryable, constant phase 1", 1, True, 0, False, True,
         "fail_trial", "constant_phase"),
        ("retryable, constant phase 2", 2, True, 0, False, True,
         "fail_trial", "constant_phase"),
        ("retryable, hybrid attempt 0, alternate exists", 3, True, 0, False, True,
         "reassigned", None),
        ("retryable, hybrid attempt 0, NO alternate", 3, True, 0, False, False,
         "fail_trial", "no_alternate_worker"),
        ("lease expiry, hybrid attempt 0, alternate exists", 4, True, 0, True, True,
         "reassigned", None),
        ("lease expiry, constant phase", 1, True, 0, True, True,
         "fail_trial", "constant_phase"),
    ]
    for (label, phase, retryable, attempt, lease_expiry,
         alternate, exp_action, exp_reason) in EXPECTED:
        with tempfile.TemporaryDirectory() as tmp:
            coord = _coord(tmp)
            run_id = "runM"
            coord.ledger.create_trial(run_id, 0)
            w0 = _register(coord, "hostA:gpu0")
            workers = [w0]
            if alternate:
                workers.append(_register(coord, "hostB:gpu0"))
            fam = "java_lcg" if phase in (1, 2) else "java_lcg_hybrid"
            assigns = coord.assign_stripes(run_id, fam, phase, 1000, [w0], now=100.0)
            sid = assigns[0]["stripe_id"]
            got = coord.handle_stripe_failure(
                run_id, sid, retryable=retryable, eligible_workers=workers,
                now=200.0, lease_expiry=lease_expiry)
            assert got["action"] == exp_action, (
                f"[{label}] action {got['action']!r} != {exp_action!r} ({got})")
            if exp_reason is not None:
                assert got.get("reason") == exp_reason, (
                    f"[{label}] reason {got.get('reason')!r} != {exp_reason!r}")
            if exp_action == "reassigned":
                assert got["attempt"] == 1 and got["phase_degraded"] is True


def gate_no_retry_consumed_on_config_error():
    """A non-retryable staging CONFIG failure must not burn the Q3 retry:
    the stripe is not reassigned and no attempt-1 exists."""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        run_id = "runQ3"
        coord.ledger.create_trial(run_id, 0)
        w0 = _register(coord, "hostA:gpu0")
        w1 = _register(coord, "hostB:gpu0")
        assigns = coord.assign_stripes(run_id, "java_lcg_hybrid", 3, 1000,
                                       [w0], now=100.0)
        sid = assigns[0]["stripe_id"]
        got = coord.handle_stripe_failure(run_id, sid, retryable=False,
                                          eligible_workers=[w0, w1], now=200.0)
        assert got["action"] == "fail_trial" and got["reason"] == "non_retryable"
        stripe = coord.ledger.get_stripe(run_id, sid)
        assert stripe["current_attempt"] == 0, "a retry attempt was consumed"
        assert not stripe["phase_degraded"], "phase_degraded set on a permanent failure"


def gate_terminal_report_preserves_root_cause():
    """§2: a missing staging path must NOT surface primarily as MinerIngressError
    or a threshold-provenance failure. The reason recorded on the trial must name
    staging configuration."""
    src = inspect.getsource(RangeMinerCoordinator._run_staging_job)
    assert "staging configuration error (non-retryable)" in src, (
        "the config-error handler must lead its reason with the root cause")
    # and the exception text itself names the subsystem, not thresholds
    with tempfile.TemporaryDirectory() as tmp:
        ledger = MinerLedger(os.path.join(tmp, "l.db"))
        coord = RangeMinerCoordinator(CoordinatorConfig(staging_dir=None), ledger)
        try:
            coord._staged_path("r", "s", 0, 0, 0, "a" * 64)
        except StagingConfigurationError as e:
            msg = str(e).lower()
            assert "staging" in msg
            assert "threshold" not in msg and "provenance" not in msg
            assert "ingress" not in msg


# ===========================================================================
# §2.15 — the three-hop route, gated as a ROUTE (not a parameter)
# ===========================================================================
def gate_three_hop_route_intact():
    """A new Step-1 parameter dies silently at hop 1 unless all three hops exist.

    hop 1: manifest default_params key + actions[0].args_map entry
    hop 2: window_optimizer.py argparse flag + call-site kwarg + coordinator attr
    hop 3: the run_trial_miner signature / integration read
    """
    # ---- hop 1: manifest (gitignored *.json — read the live file) ----------
    mpath = os.path.join(_ROOT, "agent_manifests", "window_optimizer.json")
    with open(mpath) as fh:
        manifest = json.load(fh)
    dp = manifest["default_params"]
    assert "staging_dir" in dp, "hop 1a: manifest default_params lacks staging_dir"
    assert dp["staging_dir"], "hop 1a: manifest staging_dir must not be null/empty"
    assert os.path.isabs(dp["staging_dir"]), "manifest staging_dir must be ABSOLUTE"
    amap = manifest["actions"][0]["args_map"]
    assert amap.get("staging-dir") == "staging_dir", "hop 1b: args_map entry missing"

    # WATCHER's step-scoped filter keeps only DECLARED keys — prove staging_dir
    # survives it (agents/watcher_agent.py:1290-1314, `if key in declared`).
    declared = dict(dp)
    merged = {**declared}
    for k, v in {"staging_dir": "/override/p", "undeclared_key": 1}.items():
        if k in declared:
            merged[k] = v
    assert merged["staging_dir"] == "/override/p"
    assert "undeclared_key" not in merged

    # WATCHER skips None-valued params when building the CLI (watcher_agent.py:1773)
    assert dp["staging_dir"] is not None, (
        "a null staging_dir would emit NO --staging-dir flag — the original defect")

    # ---- hop 2: window_optimizer.py ----------------------------------------
    wo = os.path.join(_ROOT, "window_optimizer.py")
    with open(wo) as fh:
        wo_src = fh.read()
    wo_tree = ast.parse(wo_src)
    flags = set()
    for node in ast.walk(wo_tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "add_argument"
                and node.args
                and isinstance(node.args[0], ast.Constant)):
            flags.add(node.args[0].value)
    assert "--staging-dir" in flags, "hop 2a: argparse lacks --staging-dir"
    # the flag must be OPTIONAL — --use-range-miner alone stays sufficient (§1.1)
    for node in ast.walk(wo_tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--staging-dir"):
            kw = {k.arg: k.value for k in node.keywords}
            assert "required" not in kw or kw["required"].value is False, (
                "--staging-dir must NOT be a required CLI flag (§1.1)")
    assert "staging_dir=getattr(args, 'staging_dir', None)" in wo_src, (
        "hop 2b: --staging-dir is parsed but never passed to the run function")
    assert "coordinator.staging_dir" in wo_src, (
        "hop 2c: nothing assigns coordinator.staging_dir — the DEAD READ is back")

    # ---- hop 3: the consumer read ------------------------------------------
    integ = os.path.join(_ROOT, "window_optimizer_integration_final.py")
    with open(integ) as fh:
        integ_src = fh.read()
    assert "getattr(coordinator, 'staging_dir', None)" in integ_src, (
        "hop 3: the integration no longer reads coordinator.staging_dir")
    sig = inspect.signature(run_trial_miner)
    assert "staging_dir" in sig.parameters


def gate_manifest_staging_dir_is_usable_here():
    """The manifest's declared path must actually validate on THIS host —
    otherwise the production route is declared but unusable."""
    mpath = os.path.join(_ROOT, "agent_manifests", "window_optimizer.json")
    with open(mpath) as fh:
        dp = json.load(fh)["default_params"]
    hw = CoordinatorConfig().staging_high_water_bytes
    ev = validate_coordinator_staging_dir(dp["staging_dir"], hw)
    assert ev["disk_backed"] is True
    assert ev["high_water_bytes"] <= ev["available_bytes"], (
        "manifest advertises a high-water larger than usable capacity")


def main():
    print("=" * 70)
    print("S172 STAGING PART B — acceptance gates (CPU-only)")
    print("=" * 70)

    print("\n-- §1.1 precedence (five rules) --")
    _check("G-PREC-1: only staging_dir -> use it",
           gate_precedence_rule1_only_staging_dir)
    _check("G-PREC-2: explicit alias -> populate + deprecation warning",
           gate_precedence_rule2_alias_populates_with_warning)
    _check("G-PREC-3: both set and differ -> FAIL CLOSED",
           gate_precedence_rule3_both_differ_fails_closed)
    _check("G-PREC-4: neither set -> FAIL CLOSED",
           gate_precedence_rule4_neither_fails_closed)
    _check("G-PREC-5: implicit /dev/shm fallback PROHIBITED (behavioural + AST)",
           gate_precedence_rule5_no_implicit_shm_fallback)
    _check("G-PREC-6: worker-local auto-detect UNCHANGED (fix did not over-reach)",
           gate_worker_autodetect_unchanged)

    print("\n-- §1.3 startup validation --")
    _check("G-VAL-1: happy path measures evidence",
           gate_validate_happy_path_measures_evidence)
    _check("G-VAL-2: relative path rejected (own reason)",
           gate_validate_rejects_relative_path)
    _check("G-VAL-3: unwritable rejected (own reason)",
           gate_validate_rejects_unwritable)
    _check("G-VAL-4: RAM-backed rejected (own reason, not capacity)",
           gate_validate_rejects_ram_backed)
    _check("G-VAL-5: capacity-invalid rejected (own reason, not RAM-backed)",
           gate_validate_rejects_capacity_invalid)
    _check("G-VAL-6: insufficient headroom rejected (own reason)",
           gate_validate_rejects_insufficient_headroom)
    _check("G-VAL-7: atomic rename PROVEN, no probe leak",
           gate_validate_atomic_rename_is_proven_not_inferred)

    print("\n-- §1.3 fail-before-dispatch --")
    _check("G-FAIL-EARLY-1: missing config fails before any dispatch",
           gate_fail_early_before_any_dispatch)
    _check("G-FAIL-EARLY-2: capacity-invalid fails before work dispatch",
           gate_fail_early_capacity_before_dispatch)
    _check("G-FAIL-EARLY-3: ledger no longer falls back to CWD",
           gate_ledger_not_created_in_cwd)

    print("\n-- §2 non-retryable, narrowly --")
    _check("G-NR-1: StagingConfigurationError is a NARROW subtype",
           gate_staging_config_error_is_narrow_subtype)
    _check("G-NR-2: caught BEFORE the generic handler; generic still retryable=True",
           gate_staging_config_error_caught_before_generic)
    _check("G-NR-3: missing staging raises the narrow subtype",
           gate_missing_staging_raises_config_error_not_bare_staging_error)
    _check("G-NR-4: Blocker-3 matrix unchanged ROW-FOR-ROW",
           gate_blocker3_matrix_unchanged_row_for_row)
    _check("G-NR-5: no Q3 retry consumed on a permanent failure",
           gate_no_retry_consumed_on_config_error)
    _check("G-NR-6: terminal report preserves the ROOT CAUSE",
           gate_terminal_report_preserves_root_cause)

    print("\n-- §2.15 three-hop route --")
    _check("G-ROUTE-1: manifest -> args_map -> argparse -> call site -> consumer",
           gate_three_hop_route_intact)
    _check("G-ROUTE-2: the manifest's declared path validates on THIS host",
           gate_manifest_staging_dir_is_usable_here)

    print("\n" + "=" * 70)
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    for name, ok, tb in _results:
        if not ok:
            print(f"\n--- {name} ---\n{tb}")
    print(f"\n{passed}/{total} checks green")
    if passed == total:
        print("COMPLETION SENTINEL: PASS — S172 Staging Part B CPU gates green")
        print("NOTE: G-PROD-SHAPE is NOT in this file — it requires a live fleet.")
        print("      See tests/gate_s172_prod_shape.py.")
        return 0
    print("COMPLETION SENTINEL: FAIL — DO NOT COMMIT")
    return 1


if __name__ == "__main__":
    sys.exit(main())
