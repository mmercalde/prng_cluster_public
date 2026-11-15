"""Run-4 routing SUCCESSOR harness — the real-cursor case beside the frozen control.

WHY THIS MODULE EXISTS. `run4_routing_clean_control.py` is FROZEN: its sha256 is
recorded in the 69ca910 STEP-0 capture (`run4_clean_control_69ca910.txt`) and
G-UNPINNED-IDENTICAL refuses to compare if it changes. Beta X-RUN4-RANGE R1 Q4: preserve
that cursor-zero oracle for its original comparison and add a SEPARATE real-cursor case;
X-MINER-START R1 Q6: a successor check must name the stage-1 scheduling changes, use an
independently established pre-miner-change control under the new search collaborator,
and keep the checks on every other argv field and every containment behaviour.

WHAT IT IS. The frozen module's stub boundary, unchanged, with exactly the collaborator
stage 1 (`915d382`) added to Step 1: the open Step 1 SEARCH. Since stage 1, run_step looks
the search up, compares dataset and configuration against the record, assigns
`seed_start` = the search's first gap and `max_seeds` = min(chunk, range end - start),
and routes `--step1-search-id`. The search is opened by the PRODUCTION opener over Run 4's
range [2^31, 2^32) (`run4_search_fixture`), so the start in the built command is read from
a real search record, not a stub. The S145 certified cursor stays stubbed at 0, as in the
frozen module (since stage 1 it no longer sets the Step 1 start).

THE DECLARED DELTA FROM THE FROZEN BOUNDARY — and nothing else (gated by
G-SUCCESSOR-BOUNDARY in tests/test_s172_run4_routing_patch.py, which compares the two
`build_argv` bodies statement by statement):
  * the frozen-dataset stub carries the four identity fields a search records
    (`run4_search_fixture.stub_frozen_dataset`);
  * the stub database is `run4_search_fixture.SearchDB` (search lookup + the same
    empty-history cursor);
  * the build's search is opened before run_step and removed after it.

The constants are IMPORTED from the frozen module, so the parameters and pins of the two
cases cannot drift apart.
"""

import hashlib
import os
import sys
import types

from . import run4_routing_clean_control as _FX
from . import run4_search_fixture as _S

SEVEN = _FX.SEVEN
CONTROL_PARAMS = _FX.CONTROL_PARAMS
PIN_VALUES = _FX.PIN_VALUES
_FakeCompleted = _FX._FakeCompleted

# The stage-1 scheduling changes between the frozen 69ca910 argv and today's, by name.
# A successor comparison against the frozen argv allows exactly these and nothing else.
STAGE1_SCHEDULING_CHANGES = (
    "--seed-start: 0 (S145 cursor, 69ca910) -> the open search's first gap",
    "--step1-search-id <id>: inserted (stage 1, S6)",
)


def module_sha256() -> str:
    with open(__file__, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def build_argv(agent_mod, params):
    """Drive run_step(1, params) on `agent_mod` and return the argv it builds.

    Returns the argv list. Raises RuntimeError if run_step returned without
    reaching dispatch -- an early guard fired and the control is INCOMPLETE
    rather than empty (VIR-5: an unobserved argv is never an empty argv).
    """
    cfg = types.SimpleNamespace(
        manifests_dir=os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "agent_manifests"),
        force_steps=set(),
        get_step_timeout_minutes=lambda step: 600,
    )
    agent = agent_mod.WatcherAgent.__new__(agent_mod.WatcherAgent)
    agent.config = cfg
    agent.retry_counts = {}

    captured = {}

    def _stub_stream(cmd, step, timeout):
        captured["argv"] = list(cmd)
        return _FakeCompleted()

    agent._ensure_execution_set = lambda p=None: object()   # non-dict => proceed
    agent._run_preflight_check = lambda step: (True, "clean-control fixture")
    agent._run_step_streaming = _stub_stream
    agent._run_post_step_cleanup = lambda *a, **kw: None
    agent._find_results = lambda *a, **kw: {}

    saved = {}
    for name, val in (
        ("check_output_freshness", lambda step: (False, "clean-control: stale", False)),
        ("p05_freeze_dataset", lambda run_label="watcher": _S.stub_frozen_dataset(
            "/home/michael/distributed_prng_analysis/daily3.json")),
        ("p05_resolve_dataset_path", lambda p: str(p)),
    ):
        saved[name] = getattr(agent_mod, name)
        setattr(agent_mod, name, val)

    # miner.dataset_authority is imported INSIDE run_step, so the local name
    # binds this module object -- patching its attributes reaches the call sites.
    # Forced down the no-nodes branch so the fleet is never contacted: P0.5's
    # UNAVAILABLE/NOT_APPLICABLE vocabulary is exercised without SSH.
    import miner.dataset_authority as _ds
    ds_saved = {}
    for _n, _v in (
        ("_active_execution_set", lambda: None),
        ("load_provisioning_nodes", lambda *a, **kw: []),
        ("fleet_preflight", lambda *a, **kw: []),
        ("resolve_absent_fleet_status", lambda *a, **kw: "NOT_APPLICABLE"),
        ("default_provisioning_manifest_path", lambda *a, **kw: "<clean-control-stub>"),
        ("write_run_provenance", lambda *a, **kw: None),
    ):
        ds_saved[_n] = getattr(_ds, _n, None)
        setattr(_ds, _n, _v)

    stub_db_mod = types.ModuleType("database_system")
    stub_db_mod.DistributedPRNGDatabase = _S.SearchDB
    prev_db = sys.modules.get("database_system")
    sys.modules["database_system"] = stub_db_mod
    _search_cleanup = None
    try:
        # The build's Step 1 search: pinned when the harness is building a pinned
        # command (it sets SearchDB.pin_bundle), unpinned otherwise.
        _search_cleanup = _S.open_run4_search(agent_mod, agent, params,
                                              _S.SearchDB.pin_bundle)
        agent_mod.WatcherAgent.run_step(agent, 1, dict(params))
    finally:
        if _search_cleanup is not None:
            _search_cleanup()
        for name, val in saved.items():
            setattr(agent_mod, name, val)
        for _n, _v in ds_saved.items():
            if _v is not None:
                setattr(_ds, _n, _v)
        if prev_db is None:
            sys.modules.pop("database_system", None)
        else:
            sys.modules["database_system"] = prev_db

    if "argv" not in captured:
        raise RuntimeError(
            "INCOMPLETE: run_step returned before dispatch; no argv was built. "
            "An early guard fired -- the control was not captured."
        )
    return captured["argv"]
