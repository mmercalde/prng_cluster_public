#!/usr/bin/env python3
"""
test_s172_phase1_scaffolding.py — S172 Phase 1 dry-run acceptance harness

Phase 1 is scaffolding + argparse gate + integration hook + WATCHER manifest
bump. No GPU work. This harness proves the scaffolding is wired correctly
without launching any real Step 1 job.

Gates (all block-on-failure):
  1. miner/ package imports; run_trial_miner is callable.
  2. run_trial_miner is the REAL wired coordinator entrypoint (updated for Phase 4
     per Team Beta's binding serve-path ruling): it builds the coordinator + trial
     and drives the default serve path; the `_serve` seam stays injectable. (Was:
     'raises NotImplementedError' — the Phase-4 coordinator is now implemented.)
  3. window_optimizer.py argparse accepts the 4 new flags (--use-range-miner,
     --miner-stripe-size, --miner-substripes, --miner-output-dir).
  4. Argparse-level mutex: --use-range-miner + --use-persistent-workers
     together must exit with a parser.error (mutex, not silent shadow).
  5. WATCHER manifest v1.8.0 has all 4 miner keys in args_map, parameter_bounds,
     and default_params. Values are the ones spec §7 mandates.
  6. Phase 0 non-regression: tests/test_prng_encoding.py still passes
     (Phase 1 must not have broken the encoding module or any import chain).

Run:
    cd ~/distributed_prng_analysis
    PYTHONPATH=. python3 tests/test_s172_phase1_scaffolding.py

Exit code 0 = all gates green (Phase 1 shippable).
Exit code 1 = a gate failed (DO NOT COMMIT).
"""
import json
import os
import subprocess
import sys
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"

_results = []


def _check(name, fn):
    try:
        fn()
        _results.append((name, True, None))
        print(f"  [{_PASS}] {name}")
    except Exception as e:
        _results.append((name, False, traceback.format_exc()))
        print(f"  [{_FAIL}] {name}: {e}")


# ---------------------------------------------------------------------------
# GATE 1 — miner/ package imports; run_trial_miner is callable
# ---------------------------------------------------------------------------
def gate1_miner_package_imports():
    from miner import run_trial_miner
    assert callable(run_trial_miner), "miner.run_trial_miner is not callable"
    # Also confirm the coordinator module resolves and re-exports the same object
    from miner import range_miner_coordinator
    assert run_trial_miner is range_miner_coordinator.run_trial_miner, (
        "miner.run_trial_miner and miner.range_miner_coordinator.run_trial_miner "
        "must be the same object for the integration gate to work."
    )


# ---------------------------------------------------------------------------
# GATE 2 — run_trial_miner is the REAL wired coordinator entrypoint
# ---------------------------------------------------------------------------
def gate2_run_trial_miner_wired():
    """UPDATED for Phase 4 (Team Beta binding ruling — the coordinator serve path
    is the central Phase-4 deliverable): run_trial_miner is no longer a
    NotImplementedError stub. It builds the CoordinatorConfig + durable ledger +
    coordinator, creates the trial, and drives it via the REAL default
    `RangeMinerCoordinator.serve_trial`; `_serve` stays an injectable seam for
    tests. This gate asserts the plumbing (args -> coordinator + trial) and that
    NO NotImplementedError is raised. (Was: 'stub raises NotImplementedError'.)"""
    import tempfile
    from miner import run_trial_miner

    captured = {}

    def _capture(coordinator, ctx):
        captured["run_id"] = ctx["run_id"]
        captured["trial"] = coordinator.ledger.get_trial(ctx["run_id"])
        return {"state": "captured", "run_id": ctx["run_id"]}

    with tempfile.TemporaryDirectory() as tmp:
        out = run_trial_miner(
            coordinator_cfg='distributed_config.json',
            config=None, trial_number=0, prng_base='java_lcg',
            residues=[1, 2, 3], total_seeds=1000,
            forward_threshold=0.01, reverse_threshold=0.01,
            test_both_modes=False, dataset_path='daily3.json',
            staging_dir=os.path.join(tmp, "stg"), _serve=_capture)
    assert out["state"] == "captured", "run_trial_miner must drive the coordinator"
    assert captured["trial"] is not None and captured["trial"]["state"] == "running", (
        "run_trial_miner must build the coordinator + create the trial")


# ---------------------------------------------------------------------------
# GATE 3 — argparse accepts the 4 new flags
# ---------------------------------------------------------------------------
def gate3_argparse_accepts_flags():
    """
    Run window_optimizer.py --help via subprocess and confirm the 4 miner flags
    are listed. --help exits 0 without importing GPU deps, so this is safe in
    any environment.
    """
    result = subprocess.run(
        [sys.executable, os.path.join(_ROOT, 'window_optimizer.py'), '--help'],
        capture_output=True, text=True, timeout=30, cwd=_ROOT,
        env={**os.environ, 'PYTHONPATH': _ROOT},
    )
    if result.returncode != 0:
        raise AssertionError(f"window_optimizer.py --help exited {result.returncode}:\n"
                             f"stdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}")
    out = result.stdout + result.stderr
    for flag in ['--use-range-miner', '--miner-stripe-size', '--miner-substripes', '--miner-output-dir']:
        assert flag in out, f"argparse --help output does not list {flag}"


# ---------------------------------------------------------------------------
# GATE 4 — backend mutex: miner + PWC together must be rejected
# ---------------------------------------------------------------------------
def gate4_backend_mutex():
    """
    Invoke window_optimizer.py with two mutually-exclusive backend flags plus
    a --lottery-file arg (required) and confirm exit code is non-zero AND the
    stderr mentions the mutex rule. We use --strategy random + tiny values so
    the argparse mutex check fires before any expensive work is attempted.
    """
    result = subprocess.run(
        [sys.executable, os.path.join(_ROOT, 'window_optimizer.py'),
         '--strategy', 'random',
         '--lottery-file', '/tmp/does_not_exist_ok.json',
         '--use-range-miner', '--use-persistent-workers'],
        capture_output=True, text=True, timeout=30, cwd=_ROOT,
        env={**os.environ, 'PYTHONPATH': _ROOT},
    )
    assert result.returncode != 0, (
        "argparse should have rejected --use-range-miner + --use-persistent-workers"
    )
    stderr = result.stderr
    assert (
        'only one of' in stderr or 'mutually exclusive' in stderr
        or '--use-persistent-workers' in stderr or '--use-range-miner' in stderr
    ), f"mutex error message not surfaced. Got stderr: {stderr[-500:]}"


# ---------------------------------------------------------------------------
# GATE 5 — WATCHER manifest v1.8.0 has the miner keys
# ---------------------------------------------------------------------------
def gate5_watcher_manifest():
    path = os.path.join(_ROOT, 'agent_manifests', 'window_optimizer.json')
    m = json.load(open(path))
    assert m.get('version') == '1.8.0', f"manifest version should be 1.8.0, got {m.get('version')}"

    args_map = m['actions'][0]['args_map']
    for cli_flag, param_key in [
        ('use-range-miner',  'use_range_miner'),
        ('miner-stripe-size','miner_stripe_size'),
        ('miner-substripes', 'miner_substripes'),
        ('miner-output-dir', 'miner_output_dir'),
    ]:
        assert args_map.get(cli_flag) == param_key, (
            f"args_map[{cli_flag!r}] should be {param_key!r}, got {args_map.get(cli_flag)!r}"
        )

    bounds = m['parameter_bounds']
    for key in ('use_range_miner', 'miner_stripe_size', 'miner_substripes', 'miner_output_dir'):
        assert key in bounds, f"parameter_bounds missing miner key {key!r}"

    defaults = m['default_params']
    # Phase 1: use_range_miner MUST default to False in the manifest (spec §7 —
    # only set true after TB approves miner production readiness at Phase 7).
    assert defaults.get('use_range_miner') is False, (
        "default_params.use_range_miner must be false in Phase 1 (spec §7 comment)"
    )
    assert defaults.get('miner_stripe_size') == 67108864
    assert defaults.get('miner_substripes')  == 8
    assert defaults.get('miner_output_dir')  is None


# ---------------------------------------------------------------------------
# GATE 6 — Phase 0 non-regression
# ---------------------------------------------------------------------------
def gate6_phase0_still_green():
    """
    Re-invoke the Phase 0 acceptance harness (tests/test_prng_encoding.py).
    Phase 1 must not have broken the encoding module or any import chain.
    """
    result = subprocess.run(
        [sys.executable, os.path.join(_ROOT, 'tests', 'test_prng_encoding.py')],
        capture_output=True, text=True, timeout=60, cwd=_ROOT,
        env={**os.environ, 'PYTHONPATH': _ROOT},
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Phase 0 harness regressed under Phase 1 changes:\n"
            f"stdout: {result.stdout[-800:]}\nstderr: {result.stderr[-400:]}"
        )


def main():
    print("\nS172 Phase 1 scaffolding acceptance harness")
    print("=" * 66)
    _check("Gate 1: miner/ package imports; run_trial_miner callable", gate1_miner_package_imports)
    _check("Gate 2: run_trial_miner is the real wired entrypoint",      gate2_run_trial_miner_wired)
    _check("Gate 3: argparse accepts 4 miner flags",                    gate3_argparse_accepts_flags)
    _check("Gate 4: mutex — miner + PWC rejected",                       gate4_backend_mutex)
    _check("Gate 5: WATCHER manifest v1.8.0 has miner keys",              gate5_watcher_manifest)
    _check("Gate 6: Phase 0 encoding harness still green",               gate6_phase0_still_green)
    print("=" * 66)

    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{passed}/{total} gates green")
    if passed != total:
        print("\nFAILURES (DO NOT COMMIT):")
        for name, ok, tb in _results:
            if not ok:
                print(f"\n--- {name} ---\n{tb}")
        sys.exit(1)
    print("\nAll gates green — Phase 1 scaffolding is deploy-ready.")
    sys.exit(0)


if __name__ == "__main__":
    main()
