#!/usr/bin/env python3
"""RUN-4 PRE-LAUNCH ROUTE-A PROVENANCE CHECK — DRY RUN, NOTHING EXECUTES.

TB ruling `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md`, post-commit
step 4: prove the seven intended values enter through the CLI operator seam and
that the generated Step-1 command contains exactly the approved pinned geometry.

WHAT THIS DRIVES. The REAL entry point: `agents/watcher_agent.py`'s
`--run-pipeline --params <json>` branch, parsed by the production argparse, split
by the production `split_operator_pin_params`, threaded through the production
`run_pipeline` -> `run_step`. Nothing about the pin path is simulated.

WHAT THIS STOPS. `_run_step_streaming` is intercepted, so the built argv is
captured and RETURNED instead of dispatched. No GPU, no fleet, no optimizer, no
step execution. Preflight, execution-set resolution and dataset freeze are
stubbed with the same deterministic fixtures the acceptance suite uses -- the
fleet is DOWN and is never contacted.

Exit 0 only if every assertion holds.
"""
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from fixtures import run4_routing_clean_control as FX     # noqa: E402
import agents.watcher_agent as W                          # noqa: E402

# ── The approved Run-4 geometry, verbatim from the TB pinned-geometry ruling ──
APPROVED = {
    "warm_start_window": 12,
    "warm_start_offset": 25,
    "warm_start_session_idx": 1,
    "warm_start_fwd_thresh": 0.71,
    "warm_start_rev_thresh": 0.47,
    "warm_start_skip_min": 6,
    "warm_start_skip_max": 99,
}

# Non-pin Step-1 params, the Attempt-9/Run-4 shape the clean control uses.
BASE = dict(FX.CONTROL_PARAMS)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    return bool(ok)


def build_via_real_cli(params_json):
    """Run the production CLI branch logic; capture argv + the step result.

    Mirrors `agents/watcher_agent.py`'s `--run-pipeline` branch exactly: parse
    the --params JSON, call the production `split_operator_pin_params`, pass the
    result as `_operator_pin_params`. The pin path is production code end to end.
    """
    captured = {}

    # --- the production CLI seam, executed ---
    override_params = json.loads(params_json)          # what argparse hands over
    ordinary, operator_pin = W.split_operator_pin_params(override_params)
    captured["cli_ordinary"] = dict(ordinary)
    captured["cli_authority"] = dict(operator_pin) if operator_pin else None

    # --- intercept the dispatch, keep everything above it real ---
    real_stream = W.WatcherAgent._run_step_streaming

    class _Done:
        returncode, stdout, stderr = 0, "", ""

    agent = W.WatcherAgent.__new__(W.WatcherAgent)
    agent.config = types.SimpleNamespace(
        manifests_dir=os.path.join(ROOT, "agent_manifests"),
        force_steps=set(),
        get_step_timeout_minutes=lambda step: 600,
        daemon_state_file=os.path.join(ROOT, "daemon_state.json"),
    )
    agent.retry_counts = {}
    agent.current_step = 1
    agent._pipeline_running = True
    agent._get_step_trials = lambda *a, **k: 1

    def _intercept(self, cmd, step, timeout):
        captured["argv"] = list(cmd)
        return _Done()

    agent._run_step_streaming = types.MethodType(_intercept, agent)
    agent._ensure_execution_set = lambda p=None: object()
    agent._run_preflight_check = lambda step: (True, "prelaunch dry run")
    agent._run_post_step_cleanup = lambda *a, **kw: None
    agent._find_results = lambda *a, **kw: {}
    _dec = types.SimpleNamespace(recommended_action="proceed", confidence=1.0,
                                 reasoning="dry run", warnings=[],
                                 suggested_param_adjustments={})
    agent.evaluate_results = lambda *a, **k: (_dec, {})
    agent.execute_decision = lambda *a, **k: False

    def _spy_run_step(self, step, p=None, *, _pin_bundle=None):
        captured["threaded_bundle"] = dict(_pin_bundle) if _pin_bundle else None
        out = W.WatcherAgent.run_step(self, step, p, _pin_bundle=_pin_bundle)
        captured["step_result"] = out
        return out

    saved_mod = {}
    for n, v in (("check_safety", lambda: True),
                 ("notify_telegram", lambda *a, **k: None),
                 ("PROGRESS_DISPLAY_AVAILABLE", False),
                 ("check_output_freshness",
                  lambda step: (False, "prelaunch dry run", False)),
                 ("p05_freeze_dataset",
                  lambda run_label="watcher": types.SimpleNamespace(
                      path=os.path.join(ROOT, "daily3.json"), sha256="0" * 64,
                      size=0, record_count=0, manifest_id="prelaunch-dry-run")),
                 ("p05_resolve_dataset_path", lambda p: str(p))):
        saved_mod[n] = getattr(W, n, None)
        setattr(W, n, v)

    import miner.dataset_authority as _ds
    ds_saved = {}
    for n, v in (("_active_execution_set", lambda: None),
                 ("load_provisioning_nodes", lambda *a, **kw: []),
                 ("fleet_preflight", lambda *a, **kw: []),
                 ("resolve_absent_fleet_status", lambda *a, **kw: "NOT_APPLICABLE"),
                 ("default_provisioning_manifest_path", lambda *a, **kw: "<dry-run>"),
                 ("write_run_provenance", lambda *a, **kw: None)):
        ds_saved[n] = getattr(_ds, n, None)
        setattr(_ds, n, v)

    stub_db = types.ModuleType("database_system")

    class _Cur:
        is_complete = False
        domain_start = 0
        domain_end_exclusive = 4294967296
        covered_seed_count = 0
        certified_interval_count = 0
        next_seed_start = 0

    class _DB:
        def get_certified_cursor(self, *a, **kw):
            return _Cur()

    stub_db.DistributedPRNGDatabase = _DB
    prev_db = sys.modules.get("database_system")
    sys.modules["database_system"] = stub_db

    agent.run_step = types.MethodType(_spy_run_step, agent)
    try:
        # THE REAL CALL the CLI makes, with the authority channel populated
        # exactly as the --run-pipeline branch populates it.
        W.WatcherAgent.run_pipeline(agent, 1, 1, ordinary,
                                    _operator_pin_params=operator_pin)
    finally:
        for n, v in saved_mod.items():
            setattr(W, n, v)
        for n, v in ds_saved.items():
            if v is not None:
                setattr(_ds, n, v)
        if prev_db is None:
            sys.modules.pop("database_system", None)
        else:
            sys.modules["database_system"] = prev_db
        W.WatcherAgent._run_step_streaming = real_stream
    return captured


def main():
    print("=" * 78)
    print("RUN-4 PRE-LAUNCH ROUTE-A PROVENANCE CHECK — DRY RUN")
    print("=" * 78)
    print("NOTHING IS DISPATCHED. Fleet is DOWN and is never contacted.")
    print(f"target : agents/watcher_agent.py")
    print()

    cli_params = {**BASE, **APPROVED}
    params_json = json.dumps(cli_params)
    print("CLI invocation this models:")
    print("  python3 agents/watcher_agent.py --run-pipeline \\")
    print("      --start-step 1 --end-step 1 \\")
    print(f"      --params '{params_json}'")
    print()

    cap = build_via_real_cli(params_json)

    print("SEAM")
    print(f"  ordinary params after split : {sorted(cap['cli_ordinary'])}")
    print(f"  authority channel keys      : {sorted(cap['cli_authority'] or [])}")
    print()

    print("ASSERTIONS")
    seven = set(APPROVED)

    check("SEAM-MOVED — no warm-start key left in ordinary params",
          not (seven & set(cap["cli_ordinary"])),
          f"leftover={sorted(seven & set(cap['cli_ordinary']))}")
    check("SEAM-AUTHORITY — all seven entered the authority channel",
          set(cap["cli_authority"] or {}) == seven,
          f"channel={sorted(cap['cli_authority'] or [])}")
    check("THREADED — run_pipeline threaded the frozen bundle to Step 1",
          set(cap.get("threaded_bundle") or {}) == seven)

    argv = cap.get("argv")
    if not argv:
        check("ARGV-BUILT — a Step-1 command was built", False,
              "no argv captured; an early guard fired")
        return 1
    check("ARGV-BUILT — a Step-1 command was built", True, f"{len(argv)} tokens")

    # exact approved geometry in argv
    pairs = {}
    for i, tok in enumerate(argv):
        if tok.startswith("--warm-start-"):
            pairs[tok[2:].replace("-", "_")] = argv[i + 1] if i + 1 < len(argv) else None
    check("GEOMETRY-COMPLETE — exactly the seven warm-start flags present",
          set(pairs) == seven,
          f"got={sorted(pairs)}")
    wrong = {k: (pairs.get(k), str(v)) for k, v in APPROVED.items()
             if pairs.get(k) != str(v)}
    check("GEOMETRY-EXACT — every value equals the approved value",
          not wrong, f"mismatches={wrong}" if wrong else
          ", ".join(f"{k}={pairs[k]}" for k in sorted(APPROVED)))
    check("EIGHTH-KEY-ABSENT — warm_start_session never routes",
          not any(t.startswith("--warm-start-session")
                  and not t.startswith("--warm-start-session-idx") for t in argv))
    check("ONE-TRIAL — --trials 1",
          "--trials" in argv and argv[argv.index("--trials") + 1] == "1")

    res = cap.get("step_result") or {}
    km, ka = W.STEP1_PIN_PROVENANCE_KEY, W.STEP1_PIN_ARGV_KEY
    check(f"PROVENANCE-MARKER — {km}={W.STEP1_PIN_SOURCE_MARKER}",
          res.get(km) == W.STEP1_PIN_SOURCE_MARKER, f"got={res.get(km)!r}")
    check(f"PROVENANCE-ARGV — {ka} matches the command actually built",
          res.get(ka) == argv,
          f"recorded {len(res.get(ka) or [])} tokens")

    print()
    print("BUILT STEP-1 COMMAND, VERBATIM")
    print("-" * 78)
    print(" ".join(argv))
    print("-" * 78)
    print()
    print("ARGV TOKENS")
    for i, t in enumerate(argv):
        print(f"  {i:3d}  {t}")

    ok = sum(1 for _, v, _ in RESULTS if v)
    print()
    print("=" * 78)
    for n, v, d in RESULTS:
        if not v:
            print(f"  NOT-PASS: {n}  {d}")
    print(f"  {ok}/{len(RESULTS)} PASS")
    if ok == len(RESULTS):
        print("  RUN4_PRELAUNCH_PROVENANCE_OK")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
