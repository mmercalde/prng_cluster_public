"""Shared deterministic fixture for the Run-4 Route-A routing patch clean control.

TB ruling `TB_RULING_RUN4_ROUTING_PATCH_BRIEF_REVIEW.md` Blocker 2: the
pre-patch oracle for `G-UNPINNED-IDENTICAL` is a control CAPTURED FROM CLEAN
`69ca910` BEFORE the first edit -- never historical source executed against a
live namespace (EXEC-PIN-1).

THIS MODULE IS THE CONTRACT. The capture (pre-edit) and the gate (post-patch)
both import it and must see byte-identical content; its sha256 is recorded in
the captured artifact and re-verified by the gate. Editing this module
invalidates the control.

Stub boundary, and why each is here. Every stubbed collaborator is one the
routing patch does NOT change; each is stubbed because it reaches live host
state (fleet, dataset manifest, coverage ledger) that is not deterministic
across the capture/gate interval -- notably the S145 certified cursor, which
Run 4 itself would advance. `_s145_domain_wall` is deliberately NOT stubbed:
it is pure arithmetic, so the real one runs.
"""

import hashlib
import json
import os
import sys
import types

BASE_COMMIT = "69ca9100f72adbeaddceddae1f11c09909b8e0c3"

# The seven routable warm-start keys (ruling §1). Literal, never derived. (Brief §3.3)
SEVEN = (
    "warm_start_window", "warm_start_offset", "warm_start_skip_min",
    "warm_start_skip_max", "warm_start_fwd_thresh", "warm_start_rev_thresh",
    "warm_start_session_idx",
)

# Attempt-9 / Run-4 geometry, unpinned form. No warm-start keys.
CONTROL_PARAMS = {
    "prng_type": "java_lcg",
    "seed_start": 0,
    "max_seeds": 2147483648,
    "miner_stripe_size": 67108864,
    "worker_pool_size": 25,
    "test_both_modes": True,
    "use_range_miner": True,
    "use_persistent_workers": False,
    "window_trials": 1,
    "n_parallel": 1,
}

# Run-4 pin values. DIAGNOSTIC ONLY at capture time: pre-patch these are dropped
# at WALL 1, so the captured argv must equal the unpinned argv. That equality is
# the executable proof of the dead chain the brief §2 describes.
PIN_VALUES = {
    "warm_start_window": 12,
    "warm_start_offset": 25,
    "warm_start_skip_min": 6,
    "warm_start_skip_max": 99,
    "warm_start_fwd_thresh": 0.71,
    "warm_start_rev_thresh": 0.47,
    "warm_start_session_idx": 1,
}


def fixture_sha256() -> str:
    with open(__file__, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


class _FakeCompleted:
    """Stands in for the dispatch result. run_step wraps the dispatch call in
    its own try/except, so the interception must RETURN, never raise -- a
    raised sentinel is caught by run_step and reported as an execution error."""

    returncode = 0
    stdout = ""
    stderr = ""


class _StubCursor:
    is_complete = False
    domain_start = 0
    domain_end_exclusive = 4294967296
    covered_seed_count = 0
    certified_interval_count = 0
    next_seed_start = 0


class _StubDB:
    def get_certified_cursor(self, *a, **kw):
        return _StubCursor()


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
        ("p05_freeze_dataset", lambda run_label="watcher": types.SimpleNamespace(
            path="/home/michael/distributed_prng_analysis/daily3.json",
            sha256="0" * 64, size=0, record_count=0, manifest_id="clean-control")),
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
    stub_db_mod.DistributedPRNGDatabase = _StubDB
    prev_db = sys.modules.get("database_system")
    sys.modules["database_system"] = stub_db_mod
    try:
        agent_mod.WatcherAgent.run_step(agent, 1, dict(params))
    finally:
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
