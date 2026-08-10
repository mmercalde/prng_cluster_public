#!/usr/bin/env python3
"""S172 — RESOLVED EXECUTION SET gate suite.

Authority: `docs/CLAUDE_CODE_INSTRUCTIONS_RESOLVED_EXECUTION_SET.md` §4.
Evidence:  `docs/FLEET_STATE_REQUIREMENTS_v1.md`.

    G-RESOLVE-ONCE     resolved once, before dataset verification, GPU
                       verification, coordinator construction and dispatch
    G-FROZEN           cannot change mid-run
    G-SAME-RESOLVER    WATCHER and the CLI produce the IDENTICAL set
    G-PROFILE          the selected profile decides the endpoints; both resolve
    G-NO-INFERENCE     a worker that connects but is not in the set does not
                       become eligible
    G-PARTIAL-EXPLICIT a partial set is accepted only when explicitly declared
    G-CONSUMERS        each of the six reads the set
    G-LOCAL            a one-node set verifies one node — and still refuses if
                       that node fails
    G-MUTANT           reverting any consumer to independent resolution turns
                       its gate red

G-NO-INFERENCE and G-MUTANT are the load-bearing pair: the first is the defect
Beta named, the second proves the rest are not vacuous.

VIR-2 discipline: every vacuous-capable detector here has a clean control, a
fault-injection control, and independence between the detector and the thing it
detects. VIR-3: this terminates in PASS | FAIL | UNAVAILABLE | INCOMPLETE.

Run:  source ~/venvs/torch/bin/activate && python3 tests/test_s172_resolved_execution_set.py
"""

import ast
import hashlib
import inspect
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import execution_set as XS                                          # noqa: E402
from execution_set import (                                          # noqa: E402
    ExecutionSetError, ResolvedExecutionSet, ResolvedNode,
    resolve_execution_set, freeze_execution_set, clear_execution_set,
    active_execution_set, execution_set_scope, filter_config_nodes,
    is_admitted_worker,
)

import coordinator as CO                                             # noqa: E402
import preflight_check as PF                                         # noqa: E402
import persistent_worker_coordinator as PWC                          # noqa: E402
import miner.dataset_authority as DA                                 # noqa: E402
import miner.range_miner_coordinator as RMC                          # noqa: E402
from miner.range_miner_coordinator import (                          # noqa: E402
    RangeMinerCoordinator, CoordinatorConfig, MinerLedger, NodeConfig,
)
from miner.range_miner_protocol import RegisterMessage             # noqa: E402
from miner.range_miner_worker import MinerFramedSocket               # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PASS, _FAIL = "PASS", "FAIL"
_results = []
_unavailable = []

CAPS = {"amd": 2_000_000, "nvidia": 5_000_000,
        "amd_hybrid": 1_000_000, "nvidia_hybrid": 2_500_000}
VARIANTS = ["java_lcg", "java_lcg_reverse",
            "java_lcg_hybrid", "java_lcg_hybrid_reverse"]


def _check(name, fn):
    try:
        detail = fn()
        print(f"  [{_PASS}] {name}" + (f" — {detail}" if detail else ""), flush=True)
        _results.append((name, True, ""))
    except Exception:                                                # noqa: BLE001
        print(f"  [{_FAIL}] {name}", flush=True)
        _results.append((name, False, traceback.format_exc()))


def _unavail(name, why):
    print(f"  [UNAVAILABLE] {name} — {why}", flush=True)
    _unavailable.append((name, why))


def _config_gpu_total():
    """Total declared GPUs, read from the authoritative fixture.

    `distributed_config.json` is what `resolve_execution_set` itself reads for
    per-node `gpu_count` (`execution_set.py:640`). Deriving the expectation from
    the same declaration — rather than transcribing a literal — is what keeps a
    certified count correction (localhost 2 -> 1) from leaving a silently stale
    magic number behind. Independent of `ResolvedExecutionSet.gpu_count()`: this
    sums the raw config, that sums resolved nodes.
    """
    with open(os.path.join(REPO, "distributed_config.json")) as fh:
        return sum(int(n.get("gpu_count", 0))
                   for n in json.load(fh).get("nodes", []))


def _cli_set(**kw):
    """A set resolved the way `window_optimizer.main()` resolves one."""
    kw.setdefault("backend", "miner")
    kw.setdefault("invoked_by", "window_optimizer.main")
    kw.setdefault("admission_count", 8)
    return resolve_execution_set(**kw)


# ===========================================================================
# G-RESOLVE-ONCE
# ===========================================================================

def g_resolve_once_read_then_freeze():
    """Freezing AFTER a consumer read is refused — INCLUDING an empty read.

    This is the structural half of "resolved BEFORE dataset verification, GPU
    verification, coordinator construction and dispatch": a set that arrives
    after a consumer already decided did not govern that decision.

    CORRECTED (admission-binding repair A). This gate previously asserted the
    OPPOSITE of its second line — that "a read of an EMPTY set must not block a
    later freeze" — and that exemption is exactly what made the submission's
    freeze-after-read claim false. `active_execution_set()` counted reads only
    when `_ACTIVE` was already non-None, so a consumer could read None, take the
    legacy path, and the set could still be frozen afterwards. The gate encoded
    the hole rather than catching it. It now requires the counted-None
    behaviour, and no longer needs to forge `XS._READS` to reach the refusal:
    a real empty read produces it.
    """
    clear_execution_set()
    try:
        # 1. EMPTY CONSUMER READ -> a later freeze is REFUSED.
        assert active_execution_set() is None                 # the read
        try:
            freeze_execution_set(_cli_set())
            raise AssertionError(
                "freeze after an EMPTY consumer read must be refused: the "
                "consumer read None and took the legacy path, which IS deciding "
                "without the set")
        except ExecutionSetError as e:
            assert "already been read" in str(e), str(e)

        # 2. CLEAN CONTROL — resolve and freeze with NO read first: passes.
        clear_execution_set()
        s = _cli_set()
        frozen = freeze_execution_set(s)
        assert frozen.set_id() == s.set_id()

        # 3. IDEMPOTENT RE-FREEZE AFTER CONSUMPTION — still permitted. Reads
        #    are now counted, so this is the case that must NOT have regressed:
        #    WATCHER and the CLI resolving identical inputs in one process is
        #    not a failure, even after consumers have read.
        assert active_execution_set() is not None             # consumed
        assert active_execution_set() is not None             # twice
        again = freeze_execution_set(_cli_set())
        assert again.set_id() == s.set_id(), "identical re-freeze must be a no-op"

        # 4. and a DIFFERENT set is still refused after consumption
        try:
            freeze_execution_set(_cli_set(declared_nodes=["localhost"],
                                          admission_count=1))
            raise AssertionError("a different set must not replace the frozen one")
        except ExecutionSetError as e:
            assert "FROZEN for this run" in str(e), str(e)
        return ("empty read blocks a later freeze; clean freeze passes; "
                "identical re-freeze after consumption still idempotent")
    finally:
        clear_execution_set()


def _main_source_order():
    """AST-located call sites in `window_optimizer.main()`.

    Located by walking the LIVE source tree, never by matching text: a text
    anchor survives a whole-block replacement (that is how `2389b61` reverted a
    threshold fix and a text gate stayed green).
    """
    src = open(os.path.join(REPO, "window_optimizer.py")).read()
    tree = ast.parse(src)
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    hits = {}
    for node in ast.walk(main):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = (f.id if isinstance(f, ast.Name)
                else f.attr if isinstance(f, ast.Attribute) else None)
        if name in ("_freeze_xset", "_resolve_xset", "run_start_dataset_gate",
                    "MultiGPUCoordinator"):
            hits.setdefault(name, node.lineno)
    return hits


def g_resolve_once_placement():
    """The CLI resolves+freezes BEFORE the dataset gate and before any
    coordinator construction — checked against the live AST of `main()`."""
    hits = _main_source_order()
    for required in ("_resolve_xset", "_freeze_xset", "run_start_dataset_gate"):
        assert required in hits, f"{required} is not called in window_optimizer.main()"
    assert hits["_resolve_xset"] < hits["run_start_dataset_gate"], (
        f"the execution set must be resolved BEFORE the P0.5 dataset gate "
        f"(resolve at line {hits['_resolve_xset']}, gate at "
        f"{hits['run_start_dataset_gate']})")
    assert hits["_freeze_xset"] <= hits["run_start_dataset_gate"], (
        "the set must be FROZEN before dataset verification")
    # MultiGPUCoordinator is constructed in run_bayesian_optimization /
    # run_with_config, both called from main() strictly after the gate; assert
    # main() itself constructs none before the freeze.
    if "MultiGPUCoordinator" in hits:
        assert hits["_freeze_xset"] < hits["MultiGPUCoordinator"], (
            "coordinator construction must follow the freeze")
    return (f"resolve@{hits['_resolve_xset']} < freeze@{hits['_freeze_xset']} "
            f"< p0.5@{hits['run_start_dataset_gate']}")


def g_resolve_once_watcher_placement():
    """WATCHER resolves before its GPU health check and before its P0.5 block."""
    src = open(os.path.join(REPO, "agents", "watcher_agent.py")).read()
    tree = ast.parse(src)
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "WatcherAgent")
    run_step = next(n for n in cls.body
                    if isinstance(n, ast.FunctionDef) and n.name == "run_step")
    order = {}
    for node in ast.walk(run_step):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("_ensure_execution_set", "_run_preflight_check",
                                  "fleet_preflight"):
                order.setdefault(node.func.attr, node.lineno)
    assert "_ensure_execution_set" in order, \
        "WatcherAgent.run_step does not resolve an execution set"
    assert "_run_preflight_check" in order
    assert order["_ensure_execution_set"] < order["_run_preflight_check"], (
        f"the set must be resolved before GPU verification "
        f"({order['_ensure_execution_set']} vs {order['_run_preflight_check']})")
    if "fleet_preflight" in order:
        assert order["_ensure_execution_set"] < order["fleet_preflight"], (
            "the set must be resolved before dataset verification")
    return (f"ensure@{order['_ensure_execution_set']} < "
            f"preflight@{order['_run_preflight_check']}")


# ===========================================================================
# G-FROZEN
# ===========================================================================

def g_frozen_cannot_be_replaced():
    clear_execution_set()
    try:
        a = _cli_set()
        freeze_execution_set(a)
        b = _cli_set(rig_profile="baremetal")
        assert a.set_id() != b.set_id(), "control: the two sets must differ"
        try:
            freeze_execution_set(b)
            raise AssertionError("replacing a frozen set must be refused")
        except ExecutionSetError as e:
            assert "FROZEN" in str(e)
        assert active_execution_set().set_id() == a.set_id(), \
            "the ORIGINAL set must survive the rejected replacement"
        # idempotent re-freeze of the identical set is not a failure
        again = _cli_set()
        assert freeze_execution_set(again).set_id() == a.set_id()
        return "replacement refused, identical re-freeze idempotent"
    finally:
        clear_execution_set()


def g_frozen_survives_config_change():
    """A topology or config change does not alter a run in progress.

    Fault injection, not assertion: the profile map on disk is really rewritten
    under the frozen set, and the set must not move.
    """
    clear_execution_set()
    tmp = tempfile.mkdtemp()
    try:
        pmap_path = os.path.join(tmp, "rig_profiles_config.json")
        shutil.copy(os.path.join(REPO, "rig_profiles_config.json"), pmap_path)
        s = resolve_execution_set(backend="miner", invoked_by="t",
                                  profile_map_path=pmap_path,
                                  config_path=os.path.join(REPO, "distributed_config.json"),
                                  repo_root=REPO)
        freeze_execution_set(s)
        before = active_execution_set().to_provenance()

        # A REAL topology change: the declared boot target flips under the run.
        # (Kept coherent with the files it joins, so the control below exercises
        # the freeze rather than the cross-check.)
        doc = json.load(open(pmap_path))
        assert doc["default_profile"] == "proxmox"
        doc["default_profile"] = "baremetal"
        json.dump(doc, open(pmap_path, "w"))

        after = active_execution_set().to_provenance()
        assert before == after, "the frozen set moved when the config changed"
        assert after["rig_profile"] == "proxmox", \
            "a mid-run profile change leaked into the frozen set"
        eps = {n["endpoint"] for n in after["nodes"]}
        assert "192.168.3.122" in eps and "192.168.3.120" not in eps, eps
        # control: a NEW resolution against the edited file really does differ
        s2 = resolve_execution_set(backend="miner", invoked_by="t",
                                   profile_map_path=pmap_path,
                                   config_path=os.path.join(REPO, "distributed_config.json"),
                                   repo_root=REPO)
        assert s2.set_id() != s.set_id(), \
            "control failed: the edit did not change what a fresh resolve produces"
        return "config rewritten mid-run; frozen set unchanged (fresh resolve differs)"
    finally:
        clear_execution_set()
        shutil.rmtree(tmp, ignore_errors=True)


# ===========================================================================
# G-SAME-RESOLVER
# ===========================================================================

def g_same_resolver_identical_set():
    """WATCHER and the CLI produce the identical set for identical inputs."""
    cli = resolve_execution_set(backend="miner", invoked_by="window_optimizer.main",
                                admission_count=8)
    watcher = resolve_execution_set(backend="miner", invoked_by="watcher_agent.run_step",
                                    admission_count=8)
    assert cli.set_id() == watcher.set_id(), (
        f"same inputs produced different sets:\n  cli={cli.describe()}\n"
        f"  watcher={watcher.describe()}")
    assert cli.invoked_by != watcher.invoked_by, "control: invokers must differ"
    # and a genuinely different input must produce a different set (independence)
    other = resolve_execution_set(backend="pwc", invoked_by="x", admission_count=24)
    assert other.set_id() != cli.set_id(), \
        "control failed: set_id is insensitive to its inputs"
    return f"identical set_id {cli.set_id()[:12]} from both invokers"


def g_same_resolver_one_function():
    """Both entry points call the SAME resolver — not two lookalikes."""
    for path, fn_names in (
        (os.path.join(REPO, "window_optimizer.py"), {"resolve_execution_set"}),
        (os.path.join(REPO, "agents", "watcher_agent.py"), {"resolve_execution_set"}),
    ):
        tree = ast.parse(open(path).read())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "execution_set":
                imported |= {a.name for a in node.names}
        missing = fn_names - imported
        assert not missing, f"{os.path.basename(path)} does not import {missing} " \
                            f"from execution_set (imports: {sorted(imported)})"
    return "both entry points import execution_set.resolve_execution_set"


# ===========================================================================
# G-PROFILE
# ===========================================================================

def g_profile_both_resolve():
    bare = _cli_set(rig_profile="baremetal")
    prox = _cli_set(rig_profile="proxmox")
    b = {n.node_id: n.endpoint for n in bare.nodes}
    p = {n.node_id: n.endpoint for n in prox.nodes}
    assert set(b) == set(p), "the two profiles must describe the SAME logical nodes"
    differing = [k for k in b if b[k] != p[k]]
    assert differing, "the two profiles resolved to identical endpoints — the " \
                      "profile selector would be inert"
    cfg = json.load(open(os.path.join(REPO, "distributed_config.json")))
    cfg_hosts = {n["hostname"] for n in cfg["nodes"]}
    for node_id, ep in b.items():
        if node_id != "localhost":
            assert ep in cfg_hosts, (
                f"baremetal endpoint {ep} for {node_id} is not in "
                f"distributed_config.json")
    prov_path = os.path.join(REPO, "dataset_provisioning.json")
    if os.path.exists(prov_path):
        prov = json.load(open(prov_path))
        for entry in prov.get("datasets", []):
            if entry.get("dataset_logical_name") != "daily3":
                continue
            for n in entry.get("nodes", []):
                assert p[n["node_id"]] == n["ssh_address"], (
                    f"proxmox endpoint for {n['node_id']} disagrees with "
                    f"dataset_provisioning.json")
    return (f"baremetal={sorted(b.values())} proxmox={sorted(p.values())}; "
            f"{len(differing)} node(s) differ")


def g_profile_decides_consumer_endpoints():
    """The profile the SET carries is the address every consumer uses."""
    cfg_nodes = json.load(open(os.path.join(REPO, "distributed_config.json")))["nodes"]
    seen = {}
    for profile in ("baremetal", "proxmox"):
        with execution_set_scope(_cli_set(rig_profile=profile)):
            hosts = [n["hostname"] for n in
                     filter_config_nodes(cfg_nodes, consumer="gate")]
            seen[profile] = hosts
            pf = PF.PreflightChecker(
                config_file=os.path.join(REPO, "distributed_config.json"))
            assert {n["hostname"] for n in pf.nodes} <= set(hosts)
    assert seen["baremetal"] != seen["proxmox"], \
        "consumers saw the same addresses under both profiles"
    assert "192.168.3.120" in seen["baremetal"] and "192.168.3.122" in seen["proxmox"]
    return f"consumers follow the profile: {seen['baremetal'][1]} vs {seen['proxmox'][1]}"


def g_profile_unknown_refused():
    try:
        _cli_set(rig_profile="does-not-exist")
        raise AssertionError("an unknown rig profile must be refused")
    except ExecutionSetError as e:
        assert "unknown rig profile" in str(e)
    return "unknown profile refused"


def g_profile_map_cross_check():
    """A profile map that disagrees with the files it joins is refused.

    Closes FLEET_STATE_REQUIREMENTS_v1 §5.4-4 ("nothing compares the two files").
    """
    tmp = tempfile.mkdtemp()
    try:
        pmap_path = os.path.join(tmp, "rig_profiles_config.json")
        doc = json.load(open(os.path.join(REPO, "rig_profiles_config.json")))
        # clean control: unmodified copy resolves
        json.dump(doc, open(pmap_path, "w"))
        resolve_execution_set(backend="miner", invoked_by="t",
                              profile_map_path=pmap_path, repo_root=REPO)
        # fault: a bare-metal endpoint that contradicts distributed_config.json
        for n in doc["nodes"]:
            if n["node_id"] == "rrig6600b":
                n["endpoints"]["baremetal"] = "192.168.3.199"
        json.dump(doc, open(pmap_path, "w"))
        try:
            resolve_execution_set(backend="miner", invoked_by="t",
                                  profile_map_path=pmap_path, repo_root=REPO)
            raise AssertionError("a contradicted baremetal endpoint must be refused")
        except ExecutionSetError as e:
            assert "INCOHERENT" in str(e) and "rrig6600b" in str(e), str(e)
        return "clean control resolves; contradicted endpoint refused"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ===========================================================================
# G-NO-INFERENCE  (load-bearing)
# ===========================================================================

def _mk_miner_coord(tmp):
    ledger = MinerLedger(os.path.join(tmp, "l.db"))
    return RangeMinerCoordinator(
        CoordinatorConfig(staging_dir=os.path.join(tmp, "staging")), ledger)


def _register_over_socket(coord, worker_id, hostname):
    """Drive the REAL `_serve_register` over a real socketpair.

    Deliberately not `register_worker` directly: the defect Beta named is on the
    serve path, and a gate that skips the serve path is testing a function
    rather than a behaviour.
    """
    a, b = socket.socketpair()
    try:
        fs_by_sock = {a: MinerFramedSocket(a)}
        worker_by_sock, wconn_by_worker, fs_by_worker, registered = {}, {}, {}, []
        msg = RegisterMessage(
            worker_id=worker_id, hostname=hostname, gpu_id=0, gpu_name="x",
            backend="rocm", vram_bytes=8 * 1024 ** 3,
            capabilities={"seed_caps": dict(CAPS),
                          "supported_variants": list(VARIANTS)})
        status = coord._serve_register(msg, a, None, fs_by_sock, worker_by_sock,
                                       wconn_by_worker, fs_by_worker, registered)
        eligible = [w for w in wconn_by_worker.values() if not w.quarantined]
        return status, wconn_by_worker.get(worker_id), eligible
    finally:
        for s in (a, b):
            try:
                s.close()
            except Exception:                                # noqa: BLE001
                pass


def g_no_inference_unlisted_worker():
    """A worker that connects but is not in the set does NOT become eligible."""
    tmp = tempfile.mkdtemp()
    try:
        s = _cli_set()
        listed = s.worker_ids()[2]                # a real rig GPU, e.g. rrig6600:gpu0
        with execution_set_scope(s):
            coord = _mk_miner_coord(tmp)

            # CLEAN CONTROL — a LISTED worker registers and IS eligible.
            st, conn, eligible = _register_over_socket(
                coord, listed, listed.split(":")[0])
            assert st == "ok", st
            assert conn is not None and not conn.quarantined, \
                f"a listed worker must be eligible, got quarantine: " \
                f"{getattr(conn, 'quarantine_reason', None)}"
            assert len(eligible) == 1

            # FAULT INJECTION — an UNLISTED worker connects, fully well-formed,
            # advertising correct caps and variants. It is the set membership,
            # and nothing else, that must refuse it.
            st2, conn2, eligible2 = _register_over_socket(
                coord, "stranger-rig:gpu0", "stranger-rig")
            assert st2 == "ok", "the connection itself is not rejected"
            assert conn2 is not None, "the worker must still be REGISTERED"
            assert conn2.quarantined, \
                "an unlisted worker became ELIGIBLE merely by connecting"
            assert "NOT in the resolved execution set" in (conn2.quarantine_reason or "")
            assert eligible2 == [], \
                f"the unlisted worker entered the eligible pool: {eligible2}"

            # durably recorded, not just an in-memory flag
            row = coord.ledger.get_worker("stranger-rig:gpu0") \
                if hasattr(coord.ledger, "get_worker") else None
            if row is not None:
                assert row.get("status") == "quarantined", row
        return "listed worker eligible; unlisted worker registered-but-ineligible"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def g_no_inference_hostname_spoof():
    """Claiming a listed HOSTNAME is not enough — the GPU identity must be listed."""
    tmp = tempfile.mkdtemp()
    try:
        with execution_set_scope(_cli_set()):
            coord = _mk_miner_coord(tmp)
            # rrig6600 is a real node with 8 GPUs; gpu99 is not in the set.
            st, conn, eligible = _register_over_socket(
                coord, "rrig6600:gpu99", "rrig6600")
            assert st == "ok"
            assert conn.quarantined, "an out-of-range GPU identity became eligible"
            assert eligible == []
        return "rrig6600:gpu99 refused despite a listed hostname"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def g_no_inference_no_set_unchanged():
    """With no set frozen, admission is exactly the pre-existing behaviour.

    This is what keeps every Phase-4 loopback gate green, and it is also the
    independence control for the two gates above: they must fail for the reason
    stated (membership), not because registration broke.
    """
    tmp = tempfile.mkdtemp()
    try:
        clear_execution_set()
        coord = _mk_miner_coord(tmp)
        st, conn, eligible = _register_over_socket(coord, "stranger-rig:gpu0",
                                                   "stranger-rig")
        assert st == "ok" and conn is not None and not conn.quarantined
        assert len(eligible) == 1
        return "no set frozen -> unchanged admission"
    finally:
        clear_execution_set()
        shutil.rmtree(tmp, ignore_errors=True)


def g_no_inference_caps_still_enforced():
    """Set membership does not replace the capability check — both compose."""
    tmp = tempfile.mkdtemp()
    try:
        s = _cli_set()
        listed = s.worker_ids()[2]
        with execution_set_scope(s):
            coord = _mk_miner_coord(tmp)
            node = NodeConfig(hostname=listed.split(":")[0], spool_root=tmp,
                              ssh_address="", ssh_user="")
            bad = dict(CAPS)
            bad["amd"] = 1                       # contradicts central config
            conn = coord.register_worker(
                worker_id=listed, hostname=listed.split(":")[0], backend="rocm",
                capabilities={"seed_caps": bad, "supported_variants": VARIANTS},
                node_config=node, admission_reason=None)
            assert conn.quarantined and "seed_cap" in (conn.quarantine_reason or "")
            # and both reasons survive together
            conn2 = coord.register_worker(
                worker_id="x:gpu0", hostname="x", backend="rocm",
                capabilities={"seed_caps": bad, "supported_variants": VARIANTS},
                node_config=node, admission_reason="NOT in the set")
            assert "NOT in the set" in conn2.quarantine_reason
            assert "ALSO" in conn2.quarantine_reason and "seed_cap" in conn2.quarantine_reason
        return "capability quarantine intact; both reasons recorded together"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ===========================================================================
# G-PARTIAL-EXPLICIT
# ===========================================================================

def g_partial_explicit_only():
    full = _cli_set()
    assert not full.partial and full.declared_nodes is None, \
        "a full set must not be marked partial"
    part = _cli_set(declared_nodes=["localhost", "rrig6600"])
    assert part.partial and part.declared_nodes == ("localhost", "rrig6600")
    assert part.node_ids() == ("localhost", "rrig6600")
    assert len(part.nodes) == 2
    # naming EVERY node is not "partial"
    everything = _cli_set(declared_nodes=list(full.node_ids()))
    assert not everything.partial, \
        "naming the whole fleet must not be recorded as a partial set"
    return f"partial={part.node_ids()} explicit; full set not marked partial"


def g_partial_unknown_node_refused():
    try:
        _cli_set(declared_nodes=["rrig6600", "rrig9999"])
        raise AssertionError("an unknown node name must be refused, not dropped")
    except ExecutionSetError as e:
        assert "rrig9999" in str(e) and "never silently dropped" in str(e)
    try:
        _cli_set(declared_nodes=[])
        raise AssertionError("an empty declaration must be refused")
    except ExecutionSetError as e:
        assert "not a declaration" in str(e)
    return "unknown node refused; empty declaration refused"


def g_partial_not_inferred_from_answers():
    """Nothing in resolution observes reachability.

    The resolver is a pure function of declared inputs: it opens config files
    and nothing else. If it ever learns to probe, this goes red.
    """
    # AST over the resolver AND everything it calls in this module — not a
    # substring scan, which both false-positives on prose ("keeping", "pinged")
    # and false-negatives on an aliased import.
    mod = ast.parse(open(XS.__file__).read())
    fns = {n.name: n for n in ast.walk(mod)
           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    called, walked, stack = set(), set(), ["resolve_execution_set"]
    while stack:
        name = stack.pop()
        if name in walked or name not in fns:
            continue
        walked.add(name)
        for node in ast.walk(fns[name]):
            if isinstance(node, ast.Call):
                f = node.func
                callee = (f.id if isinstance(f, ast.Name)
                          else f.attr if isinstance(f, ast.Attribute) else None)
                if callee:
                    called.add(callee)
                    stack.append(callee)
    forbidden = {"run", "check_output", "Popen", "call", "create_connection",
                 "urlopen", "post", "connect", "gethostbyname", "getaddrinfo"}
    hit = forbidden & called
    assert not hit, (
        f"the resolver's call graph reaches {sorted(hit)} — membership would "
        f"then be inferred from what answered")
    # Independence control: the traversal DOES see the calls that are there, so
    # an empty `hit` means "no network", not "the walker found nothing".
    assert "gethostname" in called and "load_profile_map" in called, \
        f"control: the call-graph walk did not reach known callees ({len(called)} seen)"
    # unreachable declared nodes still resolve: the set is a declaration
    s = _cli_set(rig_profile="baremetal")          # .120/.154/.162 are DOWN today
    # DERIVED, not a magic literal. The authoritative per-node GPU count is
    # `distributed_config.json` — the same file `resolve_execution_set` reads at
    # `execution_set.py:640` (`int(cfg.get("gpu_count", 0))`). Summing it here
    # independently means the certified localhost correction (2 -> 1) moves this
    # expectation on its own; the previous literal `26` was the pre-correction
    # total and went stale silently.
    expected_gpus = _config_gpu_total()
    assert len(s.nodes) == 4 and s.gpu_count() == expected_gpus, \
        f"set reports {s.gpu_count()} GPUs, config declares {expected_gpus}"
    return ("resolution performs no reachability probe; down nodes still "
            f"resolve ({len(s.nodes)} nodes, {s.gpu_count()} GPUs derived "
            f"from distributed_config.json)")


# ===========================================================================
# G-CONSUMERS  (all six)
# ===========================================================================

def g_consumer_legacy_test_connectivity():
    s = _cli_set(declared_nodes=["localhost", "rrig6600"])
    with execution_set_scope(s):
        c = CO.MultiGPUCoordinator(
            config_file=os.path.join(REPO, "distributed_config.json"))
        hosts = [n.hostname for n in c.nodes]
        assert hosts == ["localhost", "192.168.3.122"], hosts
        assert set(c._node_max_concurrent) == set(hosts), \
            "the concurrency map must be keyed by the SAME re-pointed hostnames"
        workers = c.create_gpu_workers()
        # DERIVED from the authoritative set, not a fresh magic literal. The
        # coordinator must create exactly one GPU worker per GPU declared by the
        # nodes in THIS set; `ResolvedExecutionSet.gpu_count()`
        # (`execution_set.py:220-221`) is that authority. The previous literal
        # `2 + 8` baked in the pre-correction localhost count of 2 and went
        # stale when the certified correction made it 1.
        expected_workers = s.gpu_count()
        assert len(workers) == expected_workers, \
            (f"coordinator created {len(workers)} GPU workers but the resolved "
             f"set declares {expected_workers} GPUs "
             f"({[(n.node_id, n.gpu_count) for n in s.nodes]})")
    return (f"MultiGPUCoordinator nodes={hosts}, {len(workers)} GPU workers "
            f"(derived from set gpu_count={expected_workers})")


def g_consumer_pwc_ready_gate():
    s = _cli_set(rig_profile="proxmox")
    with execution_set_scope(s):
        p = PWC.PersistentWorkerCoordinator(
            config_file=os.path.join(REPO, "distributed_config.json"),
            min_workers=24)
        hosts = [n.hostname for n in p.nodes]
        assert "192.168.3.122" in hosts and "192.168.3.120" not in hosts, hosts
        assert p.min_workers == 24, "min_workers must not be rewritten by the set"
        assert p.worker_pool_size == 8, "worker_pool_size semantics must not change"
    return f"PWC nodes={hosts}; min_workers/worker_pool_size unchanged"


def g_consumer_watcher_gpu_health():
    s = _cli_set(rig_profile="proxmox")
    with execution_set_scope(s):
        pf = PF.PreflightChecker(
            config_file=os.path.join(REPO, "distributed_config.json"))
        hosts = [n["hostname"] for n in pf.nodes]
        assert hosts == ["192.168.3.122", "192.168.3.156", "192.168.3.164"], hosts
        assert all(n["gpu_count"] == 8 for n in pf.nodes)
    # NON-BLOCKING must be preserved — asserted on the live source of check_all
    src = inspect.getsource(PF.PreflightChecker.check_all)
    tree = ast.parse(src.lstrip())
    gpu_calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "add_failure"]
    assert "add_warning" in src, "GPU issues must still be warnings"
    for call in gpu_calls:
        assert "GPU" not in ast.dump(call), \
            "a GPU issue became a blocking failure — WATCHER GPU health is " \
            "non-blocking BY DESIGN and that must be preserved"
    return f"preflight nodes={hosts}; GPU health still non-blocking"


def g_consumer_p05_dataset_targets():
    s = _cli_set(rig_profile="proxmox")
    targets = s.dataset_verification_targets()
    ids = [t.node_id for t in targets]
    addrs = [t.ssh_address for t in targets]
    assert ids == ["localhost", "rrig6600", "rrig6600b", "rrig6600c"], ids
    assert addrs[1:] == ["192.168.3.122", "192.168.3.156", "192.168.3.164"], addrs
    assert targets[0].local is True, "the local node must be verified locally"
    # the P0.5 gate really consults the set — checked on the live source
    src = inspect.getsource(DA.run_start_dataset_gate)
    assert "_active_execution_set()" in src and "dataset_verification_targets" in src
    # ...and the vocabulary it must keep is still there
    assert DA.FLEET_STATUS_UNAVAILABLE == "UNAVAILABLE"
    assert DA.FLEET_STATUS_NOT_APPLICABLE == "NOT_APPLICABLE"
    assert "unknown" in inspect.getsource(DA.resolve_absent_fleet_status).lower()
    return f"P0.5 targets {ids} at {addrs}; UNAVAILABLE/NOT_APPLICABLE intact"


def g_consumer_miner_admission():
    s = _cli_set()
    with execution_set_scope(s):
        ok, why = is_admitted_worker(s.worker_ids()[0])
        assert ok and why is None
        bad, reason = is_admitted_worker("nobody:gpu0")
        assert not bad and "NOT in the resolved execution set" in reason
    src = inspect.getsource(RMC.RangeMinerCoordinator._serve_register)
    assert "_execution_set_admission" in src, \
        "_serve_register does not consult the resolved set"
    return "miner registration consults the set"


def g_consumer_boot_notify():
    """boot notify reads the same declared fleet — and still blocks nothing."""
    script = os.path.join(REPO, "scripts", "cluster_boot_notify.sh")
    conf = "/etc/cluster-boot-notify.conf"
    if not os.access(conf, os.R_OK):
        raise RuntimeError(f"UNAVAILABLE: {conf} not readable as this user")
    tmp = tempfile.mkdtemp()
    try:
        # Stub curl so nothing leaves the box and the message body is observable.
        stub = os.path.join(tmp, "curl")
        out = os.path.join(tmp, "posted.txt")
        with open(stub, "w") as f:
            f.write(f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" > {out}\nexit 0\n')
        os.chmod(stub, 0o755)
        env = dict(os.environ, PATH=tmp + os.pathsep + os.environ["PATH"],
                   TFM_ROOT=REPO)
        proc = subprocess.run(["bash", script], env=env, capture_output=True,
                              text=True, timeout=60)
        assert proc.returncode == 0, \
            f"boot notify must exit 0 unconditionally (got {proc.returncode})"
        body = open(out).read() if os.path.exists(out) else ""
        assert "GPUs:" in body, "the GPU verdict line must survive"
        assert "Fleet: node localhost" in body, \
            f"boot notify did not read the fleet definition; posted:\n{body}"
        assert "endpoints" in body
        # fault injection: no profile map -> the line disappears, exit stays 0
        env2 = dict(env, TFM_ROOT=tmp)
        proc2 = subprocess.run(["bash", script], env=env2, capture_output=True,
                               text=True, timeout=60)
        assert proc2.returncode == 0, "a missing fleet definition must not fail boot notify"
        body2 = open(out).read()
        assert "Fleet: node" not in body2 and "GPUs:" in body2
        return "fleet line present; exit 0 preserved; degrades silently"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def g_consumers_none_deleted():
    """All six mechanisms still exist. They were re-pointed, not retired."""
    assert callable(CO.MultiGPUCoordinator.test_connectivity)
    assert callable(PWC.PersistentWorkerCoordinator._tcp_wait_ready)
    assert callable(PF.PreflightChecker.check_gpu_health)
    assert callable(DA.fleet_preflight) and callable(DA.load_provisioning_nodes)
    assert callable(RMC.RangeMinerCoordinator.register_worker)
    assert os.path.exists(os.path.join(REPO, "scripts", "cluster_boot_notify.sh"))
    # and the things §5 forbids touching
    src = open(os.path.join(REPO, "miner", "range_miner_coordinator.py")).read()
    assert "DEFAULT_WORKER_ADMISSION_TIMEOUT = 180.0" in src
    assert 'context.get("serve_timeout", None)' in src
    cfg = json.load(open(os.path.join(REPO, "distributed_config.json")))
    assert [n["hostname"] for n in cfg["nodes"]] == [
        "localhost", "192.168.3.120", "192.168.3.154", "192.168.3.162"], \
        "distributed_config.json's addresses were modified — they are deliberate"
    return "six mechanisms present; admission timeout / serve_timeout / addresses unchanged"


# ===========================================================================
# G-LOCAL
# ===========================================================================

def _frozen_for(path):
    data = open(path, "rb").read()
    return DA.FrozenDataset(
        dataset_logical_name="daily3", path=os.path.abspath(path),
        filename=os.path.basename(path),
        sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data),
        record_count=1, resolution_source="explicit")


def g_local_one_node_verified():
    """A one-node local set verifies exactly that node — Beta's Q1, via the set."""
    tmp = tempfile.mkdtemp()
    try:
        f = os.path.join(tmp, "daily3-local.json")
        open(f, "w").write('[{"a": 1}]')
        s = _cli_set(declared_nodes=["localhost"], admission_count=1)
        assert s.remote_execution is False, \
            "a local-only set must derive remote_execution=False from the SET"
        targets = s.dataset_verification_targets()
        assert len(targets) == 1 and targets[0].local is True
        records = DA.fleet_preflight(_frozen_for(f), targets)
        assert len(records) == 1 and records[0].status == "PASS", records
        return "one-node set verified exactly one node, locally, PASS"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def g_local_still_refuses_on_failure():
    """The refinement narrows the SCOPE of verification, never its strictness."""
    tmp = tempfile.mkdtemp()
    try:
        s = _cli_set(declared_nodes=["localhost"], admission_count=1)
        missing = _frozen_for(os.path.join(REPO, "distributed_config.json"))
        missing = DA.FrozenDataset(**{**missing.to_provenance(),
                                      "path": os.path.join(tmp, "absent.json")})
        try:
            DA.fleet_preflight(missing, s.dataset_verification_targets())
            raise AssertionError("a one-node set whose node FAILS must still refuse")
        except Exception as e:                                # noqa: BLE001
            assert "FAIL BEFORE DISPATCH" in str(e) and "ABSENT" in str(e), str(e)
        return "one-node set with a failing node still refuses before dispatch"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def g_local_not_a_bypass():
    """`remote_execution` is derived from the set, never declared beside it."""
    src = inspect.getsource(XS.resolve_execution_set)
    assert "remote_execution = any(not n.local for n in nodes)" in src, \
        "remote_execution must be DERIVED from the set"
    assert "remote_execution" not in inspect.signature(
        XS.resolve_execution_set).parameters, \
        "remote_execution must not be a caller-supplied input — that is the bypass"
    s = _cli_set()
    assert s.remote_execution is True, \
        "a set containing rigs must report remote execution"
    return "remote_execution derived, not declarable"


# ===========================================================================
# G-MUTANT — revert each consumer to independent resolution; its gate must red
# ===========================================================================

def _legacy_nodes(node_dicts, *, consumer):
    """The pre-work behaviour: decide for yourself, ignore the set."""
    return list(node_dicts)


def _legacy_admission(worker_id):
    """The pre-work behaviour: whoever connects is eligible."""
    return True, None


def _mutant_must_be_red(gate_name, gate_fn, module, attr, replacement):
    original = getattr(module, attr)
    setattr(module, attr, replacement)
    try:
        gate_fn()
    except Exception:                                        # noqa: BLE001
        return f"{gate_name} correctly went RED with {attr} reverted"
    else:
        raise AssertionError(
            f"{gate_name} stayed GREEN after reverting {module.__name__}.{attr} "
            f"to independent resolution — the gate is VACUOUS")
    finally:
        setattr(module, attr, original)


def g_mutant_summary():
    assert _MUTANT_RESULTS, "no mutant was exercised"
    bad = [k for k, v in _MUTANT_RESULTS.items() if not v]
    assert not bad, f"vacuous gates: {bad}"
    return f"{len(_MUTANT_RESULTS)} consumer mutants all turned their gate red"


_MUTANT_RESULTS = {}


# ===========================================================================
# runner
# ===========================================================================

def main():
    print("=" * 70)
    print("S172 — RESOLVED EXECUTION SET gate suite")
    print(f"repo: {REPO}")
    print(f"host: {socket.gethostname()}")
    print("=" * 70)

    clear_execution_set()

    print("\n-- G-RESOLVE-ONCE --")
    _check("G-RESOLVE-ONCE: freeze after a consumer read is refused",
           g_resolve_once_read_then_freeze)
    _check("G-RESOLVE-ONCE: CLI resolves+freezes before dataset verification "
           "and coordinator construction (AST)", g_resolve_once_placement)
    _check("G-RESOLVE-ONCE: WATCHER resolves before GPU and dataset verification "
           "(AST)", g_resolve_once_watcher_placement)

    print("\n-- G-FROZEN --")
    _check("G-FROZEN: a different set cannot replace the frozen one",
           g_frozen_cannot_be_replaced)
    _check("G-FROZEN: a mid-run config change does not alter the run",
           g_frozen_survives_config_change)

    print("\n-- G-SAME-RESOLVER --")
    _check("G-SAME-RESOLVER: WATCHER and the CLI produce the identical set",
           g_same_resolver_identical_set)
    _check("G-SAME-RESOLVER: both entry points import the one resolver",
           g_same_resolver_one_function)

    print("\n-- G-PROFILE --")
    _check("G-PROFILE: both profiles resolve, to different endpoints",
           g_profile_both_resolve)
    _check("G-PROFILE: the profile decides what consumers address",
           g_profile_decides_consumer_endpoints)
    _check("G-PROFILE: an unknown profile is refused", g_profile_unknown_refused)
    _check("G-PROFILE: a map contradicting its sources is refused",
           g_profile_map_cross_check)

    print("\n-- G-NO-INFERENCE (load-bearing) --")
    _check("G-NO-INFERENCE: an unlisted worker registers but is NOT eligible",
           g_no_inference_unlisted_worker)
    _check("G-NO-INFERENCE: a listed hostname does not admit an unlisted GPU",
           g_no_inference_hostname_spoof)
    _check("G-NO-INFERENCE: with no set frozen, admission is unchanged",
           g_no_inference_no_set_unchanged)
    _check("G-NO-INFERENCE: the capability quarantine still applies, and composes",
           g_no_inference_caps_still_enforced)

    print("\n-- G-PARTIAL-EXPLICIT --")
    _check("G-PARTIAL-EXPLICIT: partial only when declared", g_partial_explicit_only)
    _check("G-PARTIAL-EXPLICIT: an unknown or empty declaration is refused",
           g_partial_unknown_node_refused)
    _check("G-PARTIAL-EXPLICIT: membership is never inferred from reachability",
           g_partial_not_inferred_from_answers)

    print("\n-- G-CONSUMERS (all six) --")
    _check("G-CONSUMERS 1/6: P0.5 dataset preflight reads the set",
           g_consumer_p05_dataset_targets)
    _check("G-CONSUMERS 2/6: legacy test_connectivity reads the set",
           g_consumer_legacy_test_connectivity)
    _check("G-CONSUMERS 3/6: PWC ready gate reads the set",
           g_consumer_pwc_ready_gate)
    _check("G-CONSUMERS 4/6: WATCHER GPU health reads the set (still non-blocking)",
           g_consumer_watcher_gpu_health)
    _check("G-CONSUMERS 5/6: miner registration reads the set",
           g_consumer_miner_admission)
    try:
        _check("G-CONSUMERS 6/6: boot notify reads the fleet definition (exit 0)",
               g_consumer_boot_notify)
    except Exception as e:                                   # noqa: BLE001
        _unavail("G-CONSUMERS 6/6: boot notify", str(e))
    _check("G-CONSUMERS: none of the six was deleted; §5 invariants intact",
           g_consumers_none_deleted)

    print("\n-- G-LOCAL --")
    _check("G-LOCAL: a one-node set verifies one node", g_local_one_node_verified)
    _check("G-LOCAL: a one-node set still refuses when that node fails",
           g_local_still_refuses_on_failure)
    _check("G-LOCAL: remote_execution is derived, not a bypass", g_local_not_a_bypass)

    print("\n-- G-MUTANT: each consumer reverted to independent resolution --")
    for gate_name, gate_fn, module, attr in (
        ("G-CONSUMERS/legacy", g_consumer_legacy_test_connectivity,
         CO, "_execution_set_nodes"),
        ("G-CONSUMERS/pwc", g_consumer_pwc_ready_gate,
         PWC, "_execution_set_nodes"),
        ("G-CONSUMERS/preflight", g_consumer_watcher_gpu_health,
         PF, "_execution_set_nodes"),
        ("G-NO-INFERENCE/miner", g_no_inference_unlisted_worker,
         RMC, "_execution_set_admission"),
        ("G-PROFILE/endpoints", g_profile_decides_consumer_endpoints,
         XS, "active_execution_set"),
    ):
        replacement = (_legacy_admission if attr == "_execution_set_admission"
                       else (lambda *_a, **_k: None) if attr == "active_execution_set"
                       else _legacy_nodes)

        def _run(g=gate_name, f=gate_fn, m=module, a=attr, r=replacement):
            out = _mutant_must_be_red(g, f, m, a, r)
            _MUTANT_RESULTS[g] = True
            return out

        _check(f"G-MUTANT/{gate_name} must go red", _run)

    _check("G-MUTANT: summary — the live gates are not vacuous", g_mutant_summary)

    clear_execution_set()

    print("=" * 70)
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{passed}/{total} resolved-execution-set checks green")
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
    if _unavailable:
        print("\nRESULT: INCOMPLETE (all executed checks passed; see UNAVAILABLE)")
        return 2
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
