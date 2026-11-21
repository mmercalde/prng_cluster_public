#!/usr/bin/env python3
"""test_pwc_stage3_r1_fixes.py — the two LIVE objections of Beta X-PWC-STAGE3 R1.

Review: docs/exchange/from_beta/X-PWC-STAGE3_R1_BETA_REVIEW.md
  3b-Q3 (:60, :93)  a nonpositive PWC chunk cap passes startup and hangs chunk planning.
                    Fix: persistent_worker_coordinator.py _startup_refusals refuses
                    seed_cap_amd / seed_cap_nvidia <= 0 before any worker is touched,
                    as RANGE-MINER refuses a nonpositive size
                    (miner/range_miner_coordinator.py:769-770, :789-790;
                    miner/range_miner_worker.py:681-682).
  3b-Q7 (:64, :103) Stage 3 passed the CLI/manifest pool size (24) as PWC's ready minimum,
                    so 18 rig workers + 1 VM101 worker, 19 of 19 ready, was refused.
                    Fix: the PWC call in run_bidirectional_test is again the b2efdef call
                    (plus seed_start and the dial address), so the effective minimum is
                    run_trial_persistent's own default, 1. Owner ruling 2026-08-04:
                    docs/CLAUDE_CODE_INSTRUCTIONS_WARMSTART_RESTORE_PART_B.md:86-100,
                    docs/FLEET_STATE_REQUIREMENTS_v1.md:146-195.

Gates:
  G-Q3-REFUSE   cap 0 and cap -1, for AMD and for NVIDIA, refuse in the REAL startup()
                with PwcSettingRefused naming the cap, and no subprocess call is made.
  G-Q3-NORMAL   a normal cap (25,000) still starts (4 launch scripts) and a sieve pass
                completes 4/4 chunks over [0, 100000).
  G-Q7-CALL     run_bidirectional_test's PWC call (coordinator pwc_min_workers=24 and
                min_workers=24) passes exactly b2efdef's keywords plus seed_start and
                pwc_coordinator_address; min_workers is not passed.
  G-Q7-READY19  Beta's scenario: 3 rigs x 6 workers + 1 VM101 worker. At the effective
                minimum (run_trial_persistent's default), 19 of 19 ready passes the
                startup ready gate and the per-pass dispatch check, and the pass
                completes. Control: at 24 the same 19 is refused (the scenario
                discriminates).

Mutants (textual edits of the live source; killed only by the detector's own
AssertionError, after a passing positive control):
  M-Q3-1 refusal removed; M-Q3-2 `<= 0` weakened to `< 0`; M-Q3-3 NVIDIA cap not checked;
  M-Q7-1 min_workers=pwc_min_workers passed again; M-Q7-2 run_trial_persistent's default
  minimum raised to 24.

CPU only: no GPU, rig, SSH, socket or live ledger. Reuses the fixtures of
tests/test_pwc_backup_step1.py and tests/test_pwc_backup_stage3.py (imported, not run).
Run from the repository root: python3 -u tests/test_pwc_stage3_r1_fixes.py
"""
import ast
import contextlib
import importlib.util
import inspect
import io
import json
import os
import shutil
import subprocess
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.chdir(ROOT)
BASE = "b2efdef"
PWC_PATH = os.path.join(ROOT, "persistent_worker_coordinator.py")
WOI_PATH = os.path.join(ROOT, "window_optimizer_integration_final.py")
OUT = os.path.join(ROOT, "data", "pwc_stage3_r1", "test_out")

import test_pwc_backup_step1 as S1                                 # noqa: E402
import test_pwc_backup_stage3 as S3                                # noqa: E402
import persistent_worker_coordinator as PWC                        # noqa: E402

S1.OUT = OUT                       # the step-1 fixtures write their configs here
RIGS = ("10.9.9.1", "10.9.9.2", "10.9.9.3")   # fake remote ROCm nodes; never dialled


def load_module(src, name, path):
    spec = importlib.util.spec_from_loader(name, loader=None, origin=path)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = path
    sys.modules[name] = mod
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod


def make(M, *, amd=25_000, nvidia=25_000, pool=4, min_workers=None, cfg=None, ready=4):
    kw = {} if min_workers is None else {"min_workers": min_workers}
    with contextlib.redirect_stderr(io.StringIO()):
        p = M.PersistentWorkerCoordinator(
            config_file=cfg or S1.write_config(8), worker_pool_size=pool,
            seed_cap_amd=amd, seed_cap_nvidia=nvidia, pwc_transport="tcp",
            pwc_coordinator_address=S1.DIAL, planned_total_seeds=100_000, **kw)
    p._tcp_transport = S1.FakeTransport(ready)
    p._effective_pool_by_host = {S1.RIG: min(pool, 8)}
    p._dispatch_to_tcp = S1.Worker()
    return p


# ── 3b-Q3 ────────────────────────────────────────────────────────────────────
def d_q3_refuse(M=None):
    M = M or PWC
    rows = []
    for name in ("seed_cap_amd", "seed_cap_nvidia"):
        for cap in (0, -1):
            kw = {"amd": cap} if name == "seed_cap_amd" else {"nvidia": cap}
            p = make(M, **kw)
            ic, exc = S1.startup_for_real(p)
            assert isinstance(exc, M.PwcSettingRefused), \
                f"{name}={cap}: startup did not refuse: {exc!r}"
            assert name in str(exc) and "must be positive" in str(exc), str(exc)
            assert not ic.calls, f"{name}={cap}: {len(ic.calls)} subprocess call(s) before the refusal"
            rows.append(f"{name}={cap}")
    return "refused before any subprocess call: " + ", ".join(rows)


def d_q3_normal(M=None):
    M = M or PWC
    p = make(M)
    ic, exc = S1.startup_for_real(p)
    assert exc is None, f"normal cap: startup failed: {exc!r}"
    assert len(ic.launch_scripts()) == 4, len(ic.launch_scripts())
    p._tcp_transport = S1.FakeTransport(4)
    p._effective_pool_by_host = {S1.RIG: 4}
    p._dispatch_to_tcp = S1.Worker()
    r = S1.sieve(p)
    a = r["pwc_pass_accounting"]
    assert r["pass_complete"] is True and (a["planned"], a["completed"]) == (4, 4), a
    assert a["completed_intervals"] == [[0, 100_000]], a["completed_intervals"]
    return "seed caps 25,000: startup ran (4 launch scripts); pass 4/4 complete over [0, 100000)"


# ── 3b-Q7 ────────────────────────────────────────────────────────────────────
def _pwc_call_keywords(src):
    fn = S3.func(ast.parse(src), "run_bidirectional_test")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "run_trial_persistent"]
    assert len(calls) == 1, f"{len(calls)} run_trial_persistent calls"
    return {k.arg for k in calls[0].keywords}


def d_q7_call(woi_mod=None):
    W = woi_mod or __import__("window_optimizer_integration_final")
    calls, _acc, exc, _tr = S3._pwc_drive(
        W, S3._complete_result(), coord_over={"pwc_min_workers": 24, "min_workers": 24})
    assert exc is None, f"{type(exc).__name__}: {exc}"
    kw = calls["rtp"][0]
    assert "min_workers" not in kw, f"min_workers passed to run_trial_persistent: {kw['min_workers']}"
    base = _pwc_call_keywords(S3.git_show("window_optimizer_integration_final.py"))
    added, removed = set(kw) - base, base - set(kw)
    assert added == {"seed_start", "pwc_coordinator_address"} and not removed, (added, removed)
    return ("coordinator pwc_min_workers=24: the PWC call is b2efdef's plus seed_start and "
            "pwc_coordinator_address; min_workers not passed")


def _fleet_config():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "pwc_cfg_3rigs_plus_vm101.json")
    nodes = [{"hostname": h, "gpu_type": "RX 6600 XT", "gpu_count": 8,
              "python_env": "/nowhere/venv/bin/python", "script_path": "/nowhere/repo",
              "username": "michael"} for h in RIGS]
    nodes.append({"hostname": "localhost", "gpu_type": "RTX 3080 Ti", "gpu_count": 1,
                  "python_env": "/nowhere/venv/bin/python", "script_path": "/nowhere/repo"})
    with open(path, "w") as fh:
        json.dump({"nodes": nodes}, fh)
    return path


def _ready_at(M, min_workers, ready=19):
    p = make(M, pool=6, min_workers=min_workers, cfg=_fleet_config(), ready=ready)
    p._effective_pool_by_host = {h: 6 for h in RIGS}
    shut = []
    p.shutdown = lambda: shut.append(1)
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            n = p._tcp_wait_ready(expected=ready, timeout_s=0.05)
    except RuntimeError as exc:
        return p, None, exc, shut
    return p, n, None, shut


def d_q7_ready19(M=None):
    M = M or PWC
    eff = inspect.signature(M.run_trial_persistent).parameters["min_workers"].default
    p, n, exc, shut = _ready_at(M, eff)
    assert exc is None and n == 19 and not shut, \
        f"effective minimum {eff}: 19 of 19 ready refused: {exc!r}"
    r = S1.sieve(p)
    assert r["pass_complete"] is True, r["pwc_pass_accounting"]
    _p, _n, exc24, shut24 = _ready_at(M, 24)
    assert isinstance(exc24, RuntimeError) and shut24, "control: 19 < 24 was not refused"
    return (f"3x6 rig + 1 VM101 workers: 19/19 ready passes at the effective minimum "
            f"{eff} (startup gate and dispatch check; pass complete); control at 24 refuses")


GATES = [
    ("G-Q3-REFUSE", d_q3_refuse),
    ("G-Q3-NORMAL", d_q3_normal),
    ("G-Q7-CALL", d_q7_call),
    ("G-Q7-READY19", d_q7_ready19),
]

_Q3_GUARD = "            if int(_cap) <= 0:\n"
MUTANTS = [
    ("M-Q3-1 nonpositive-cap refusal removed", PWC_PATH,
     _Q3_GUARD, "            if False:\n", d_q3_refuse),
    ("M-Q3-2 refusal weakened to cap < 0 (cap 0 passes)", PWC_PATH,
     _Q3_GUARD, "            if int(_cap) < 0:\n", d_q3_refuse),
    ("M-Q3-3 NVIDIA cap not checked", PWC_PATH,
     '        for _cap_name in ("seed_cap_amd", "seed_cap_nvidia"):\n',
     '        for _cap_name in ("seed_cap_amd",):\n', d_q3_refuse),
    ("M-Q7-1 pool size passed as min_workers again", WOI_PATH,
     "            pwc_coordinator_address = getattr(coordinator, 'pwc_coordinator_address', None),\n",
     "            pwc_coordinator_address = getattr(coordinator, 'pwc_coordinator_address', None),\n"
     "            min_workers = getattr(coordinator, 'pwc_min_workers', 1),\n", d_q7_call),
    ("M-Q7-2 run_trial_persistent default minimum raised to 24", PWC_PATH,
     "                         min_workers: int = 1,                   # [S163-KARG-PWC] + [S174 ready gate]\n",
     "                         min_workers: int = 24,                  # [S163-KARG-PWC] + [S174 ready gate]\n",
     d_q7_ready19),
]


def run_mutant(label, path, old, new, detector):
    with open(path) as fh:
        src = fh.read()
    n = src.count(old)
    assert n == 1, f"mutant anchor occurs {n} times: {old[:80]!r}"
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        detector()                                                  # positive control
        name = "pwc_mut_r1" if path == PWC_PATH else "woi_mut_r1"
        mod = load_module(src.replace(old, new), name, path)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            detector(mod)
    except AssertionError as exc:
        return f"killed: {str(exc).splitlines()[0][:120]}"
    except Exception as exc:                                        # noqa: BLE001
        raise AssertionError(f"UNATTRIBUTABLE kill: {type(exc).__name__}: {exc}")
    raise AssertionError(f"SURVIVED: {label}")


def main():
    import logging
    logging.disable(logging.CRITICAL)
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT, exist_ok=True)
    print("VERIFICATION test_pwc_stage3_r1_fixes START", flush=True)
    passed = killed = 0
    for name, fn in GATES:
        try:
            print(f"[PASS] {name}: {fn()}", flush=True)
            passed += 1
        except Exception as exc:                                    # noqa: BLE001
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
    for label, path, old, new, det in MUTANTS:
        try:
            print(f"[PASS] {label} -> {run_mutant(label, path, old, new, det)}", flush=True)
            killed += 1
        except Exception as exc:                                    # noqa: BLE001
            print(f"[FAIL] {label}: {type(exc).__name__}: {exc}", flush=True)
    print(f"GATES {passed}/{len(GATES)} PASS")
    print(f"MUTANTS {killed}/{len(MUTANTS)} killed")
    ok = passed == len(GATES) and killed == len(MUTANTS)
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
