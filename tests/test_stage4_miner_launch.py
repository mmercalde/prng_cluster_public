#!/usr/bin/env python3
"""test_stage4_miner_launch.py — the derived Stage 4 RANGE-MINER launcher
(docs/drafts/pwc_backup_stage4_v1/stage4_miner_launch.diff applied to gate12_launch.sh).
CPU only. Nothing is launched: no WATCHER, no fleet, no SSH, no GPU. Files go under
<root>/logs/test_stage4_miner_launch/ (git-ignored). The real halt file
(/tmp/agent_halt) and the live staging dir are never created, opened or removed: the halt
path is redirected by a stand-in `agents/safety/kill_switch.py` in the scratch tree, and
the live staging dir is only compared by path.

Run from the repository root:   python3 -u tests/test_stage4_miner_launch.py

Beta X-PWC-STAGE4 R1 Q8/Q7/Q9:
  L-APPLY      the diff applies to gate12_launch.sh with `patch`; `bash -n` passes
  L-STRUCTURE  params loaded and validated before the first gate, the clean slate and the
               fleet launch; WATCHER gets "$STAGE4_PARAMS"; no --clear-halt; the
               concurrency sampler is never started (no non-comment invocation)
  L-SAMPLER-PATH  the real sampler's default ledger resolves to the manifest's (live)
               staging_dir, which the Stage 4 params never name: why it is not run
  P1-P10       the extracted params block, executed: valid / missing / bad JSON / live
               staging / staging under live / non-empty staging / missing staging / not a
               miner run / a pin missing / halt present (and left in place)
  L-EMPTY-STAGING  the coordinator's own ledger constructor makes a fresh schema in an
               empty staging dir (miner/range_miner_coordinator.py:12371-12374)
  C3-*         [Beta X-PWC-STAGE4 R2 Finding 2] the runbook's PWC launch block (between its
               C3-PWC-LAUNCH markers, run verbatim, nohup stubbed to record its argv): a
               missing, malformed or RANGE-MINER params file, or a failed cd, starts
               NOTHING; a valid PWC params file reaches nohup with that JSON
"""
import json
import os
import shutil
import sqlite3
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
OUT = os.path.join(ROOT, "logs", "test_stage4_miner_launch")
DIFF = os.path.join(ROOT, "docs", "drafts", "pwc_backup_stage4_v1", "stage4_miner_launch.diff")
BASE = os.path.join(ROOT, "gate12_launch.sh")
MANIFEST = os.path.join(ROOT, "agent_manifests", "window_optimizer.json")
DERIVED = os.path.join(OUT, "stage4_miner_launch.sh")

LIVE_STAGING = json.load(open(MANIFEST))["default_params"]["staging_dir"]
PARAMS = {"prng_type": "java_lcg", "seed_start": 0, "max_seeds": 268435456,
          "miner_stripe_size": 67108864, "worker_pool_size": 25, "test_both_modes": True,
          "use_range_miner": True, "use_persistent_workers": False, "window_trials": 1,
          "n_parallel": 1, "warm_start_window": 12, "warm_start_offset": 25,
          "warm_start_session_idx": 1, "warm_start_fwd_thresh": 0.71,
          "warm_start_rev_thresh": 0.47, "warm_start_skip_min": 0, "warm_start_skip_max": 16}


def derived():
    if not os.path.exists(DERIVED):
        r = subprocess.run(["patch", "-s", "-o", DERIVED, BASE, DIFF],
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, (r.stdout, r.stderr)
    return open(DERIVED).read()


def g_apply():
    src = derived()
    r = subprocess.run(["bash", "-n", DERIVED], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    r = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "gate12_launch.sh"], timeout=60)
    assert r.returncode == 0, "gate12_launch.sh differs from HEAD; the diff's base is HEAD's file"
    return f"patch applied ({len(src.splitlines())} lines), bash -n ok, base is HEAD's gate12_launch.sh"


def _code_lines(src):
    return [(i, l) for i, l in enumerate(src.splitlines(), 1) if not l.lstrip().startswith("#")]


def g_structure():
    src = derived()
    code = _code_lines(src)

    def first(pat):
        hits = [i for i, l in code if pat in l]
        assert hits, f"no code line with {pat!r}"
        return hits[0]
    p_end = first("STAGE 4 params validated")
    for pat in ("gate12_cleantree_gate.py --phase admission", "gate12_gpu_gate.py",
                "gate12_parity_gate.py", 'pkill -f "[w]atcher_agent"',
                "./scripts/launch_fleet_manual.sh"):
        assert p_end < first(pat), f"params validated after {pat!r}"
    assert not [l for _, l in code if "--clear-halt" in l], "a command still passes --clear-halt"
    assert not [l for _, l in code if "gate12_concurrency_sampler.py" in l], "the sampler is still started"
    w = [l for _, l in code if '--params "$STAGE4_PARAMS"' in l]
    assert len(w) == 1, w
    assert not [l for _, l in code if "$(cat logs/stage4/params.json)" in l], "inline cat remains"
    assert [l for _, l in code if l.strip() == 'SAMPLER=""'], "SAMPLER is not defined for set -u"
    return ("params validated before cleantree/gpu/parity gates, clean slate and fleet; "
            "WATCHER gets $STAGE4_PARAMS; no --clear-halt; sampler not started")


def g_sampler_path():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gate12_concurrency_sampler", os.path.join(ROOT, "scripts", "gate12_concurrency_sampler.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    p = m.default_ledger_path()             # reads the manifest only; opens no database
    assert p == os.path.join(LIVE_STAGING, "miner_ledger.db"), p
    assert os.path.realpath(PARAMS_STAGING()) != os.path.realpath(LIVE_STAGING)
    return f"the sampler's default ledger is {p} (live); Stage 4 staging is elsewhere, so it is not run"


def PARAMS_STAGING():
    return os.path.join(OUT, "staging_ok")


def block():
    src = derived()
    a = src.index("# --- STAGE4-PARAMS BEGIN ---")
    b = src.index("# --- STAGE4-PARAMS END ---")
    return src[a:b]


def scratch_tree(tag, params, staging=None, halt=False, write_params=True, raw=None):
    t = os.path.join(OUT, tag)
    os.makedirs(os.path.join(t, "logs", "stage4"))
    os.makedirs(os.path.join(t, "agent_manifests"))
    shutil.copyfile(MANIFEST, os.path.join(t, "agent_manifests", "window_optimizer.json"))
    os.makedirs(os.path.join(t, "agents", "safety"))
    for d in ("agents", os.path.join("agents", "safety")):
        open(os.path.join(t, d, "__init__.py"), "w").close()
    halt_path = os.path.join(t, "agent_halt_standin")
    with open(os.path.join(t, "agents", "safety", "kill_switch.py"), "w") as f:
        f.write(f"class KillSwitch:\n    halt_file_path = {halt_path!r}\n")
    if halt:
        open(halt_path, "w").write("someone's halt")
    if write_params:
        with open(os.path.join(t, "logs", "stage4", "params.json"), "w") as f:
            f.write(raw if raw is not None else json.dumps(params))
    script = os.path.join(t, "run_block.sh")
    with open(script, "w") as f:
        f.write("set -u\n" + block() + 'echo "RESULT_PARAMS=$STAGE4_PARAMS"\n')
    r = subprocess.run(["bash", script], cwd=t, capture_output=True, text=True, timeout=60,
                       env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    return r, halt_path


def ok_staging(tag):
    d = os.path.join(OUT, tag + "_staging")
    os.makedirs(d)
    return d


def p_case(tag, params_fn, must, staging_fn=ok_staging, **kw):
    st = staging_fn(tag)
    p = params_fn(dict(PARAMS, staging_dir=st)) if params_fn else None
    r, halt_path = scratch_tree(tag, p, **kw)
    out = r.stdout + r.stderr
    if must is None:
        assert r.returncode == 0, out[-500:]
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT_PARAMS=")]
        got = json.loads(line[0].split("=", 1)[1])
        assert got["staging_dir"] == st and got["use_range_miner"] is True, got
        return f"rc 0; validated params passed on ({len(line[0])} chars)"
    assert r.returncode == 1, (r.returncode, out[-400:])
    assert must in out, out[-400:]
    assert "RESULT_PARAMS=" not in r.stdout, "an empty or invalid --params would have been passed"
    return f"rc 1: {must}"


def p10_halt():
    r = p_case("p10", lambda p: p, "halt file", halt=True)
    halt_path = os.path.join(OUT, "p10", "agent_halt_standin")
    assert os.path.exists(halt_path), "the halt file was cleared"
    return r + "; halt file left in place"


def g_empty_staging_ledger():
    from miner.range_miner_coordinator import MinerLedger
    st = os.path.join(OUT, "fresh_staging")
    os.makedirs(st)
    assert not os.listdir(st)
    db = os.path.join(st, "miner_ledger.db")
    led = MinerLedger(db)
    del led
    con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    tabs = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
    need = {"trials", "stripes", "shards", "trial_context"}
    assert need <= tabs, tabs
    return f"MinerLedger on an empty staging dir creates {sorted(need)} (fresh ledger)"


RUNBOOK = os.path.join(ROOT, "docs", "drafts", "pwc_backup_stage4_v1", "SESSION_RUNBOOK.md")
PWC_PARAMS = {"prng_type": "java_lcg", "use_range_miner": False, "use_persistent_workers": True,
              "pwc_transport": "tcp", "warm_start_window": 12}


def c3_block():
    src = open(RUNBOOK).read()
    a = src.index("# --- C3-PWC-LAUNCH BEGIN ---")
    b = src.index("# --- C3-PWC-LAUNCH END ---")
    return src[a:b]


def c3_case(tag, raw=None, write=True, bad_cd=False):
    base = os.path.join(OUT, "c3_" + tag)
    tp = os.path.join(base, "tree")
    os.makedirs(os.path.join(tp, "logs", "stage4"))
    if write:
        with open(os.path.join(tp, "logs", "stage4", "params.json"), "w") as f:
            f.write(raw if raw is not None else json.dumps(PWC_PARAMS))
    stub = os.path.join(base, "bin")
    os.makedirs(stub)
    rec = os.path.join(base, "nohup_argv.txt")
    with open(os.path.join(stub, "nohup"), "w") as f:
        f.write(f'#!/bin/bash\nprintf "%s\\n" "$@" > {rec!r}\n')
    os.chmod(os.path.join(stub, "nohup"), 0o755)
    script = os.path.join(base, "c3.sh")
    with open(script, "w") as f:
        f.write(c3_block() + "wait\n")
    env = dict(os.environ, PATH=stub + os.pathsep + os.environ["PATH"],
               TP=os.path.join(base, "no_such_tree") if bad_cd else tp)
    r = subprocess.run(["bash", script], cwd=base, capture_output=True, text=True, timeout=60, env=env)
    argv = open(rec).read().splitlines() if os.path.exists(rec) else None
    return r, argv


def c3_valid():
    r, argv = c3_case("valid")
    assert argv is not None, (r.stdout, r.stderr)
    i = argv.index("--params")
    assert json.loads(argv[i + 1]) == PWC_PARAMS, argv
    return "valid PWC params: nohup reached with the validated JSON"


def c3_refused(tag, **kw):
    r, argv = c3_case(tag, **kw)
    assert argv is None, f"{tag}: nohup was started with {argv}"
    assert "PWC LAUNCH REFUSED" in r.stdout, r.stdout[-300:]
    return f"{tag}: nothing started; REFUSED printed"


def runbook_check_line():
    for ln in open(RUNBOOK).read().splitlines():
        if ln.startswith("CHECK='"):
            return ln[len("CHECK='"):-1]
    raise AssertionError("no CHECK= line in the runbook's C5b")


def c5b_check():
    """[Beta X-PWC-SHUTDOWN R1 Q7] the runbook's per-host leftover check: a working probe
    reports PWC_NONE (no real PWC worker here) or PWC_FOUND; pgrep that cannot run reports
    PROBE_UNAVAILABLE with exit 7, never 'none'."""
    chk = runbook_check_line()
    if subprocess.run(["pgrep", "-f", "[p]wc_worker_service"], capture_output=True).returncode == 0:
        return "UNAVAILABLE: a real pwc_worker_service runs on this host; the NONE case cannot be shown"
    r = subprocess.run(["bash", "-c", chk], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and r.stdout.strip() == "PWC_NONE", (r.returncode, r.stdout)
    stub = os.path.join(OUT, "c5b_stub")
    os.makedirs(stub, exist_ok=True)
    with open(os.path.join(stub, "pgrep"), "w") as f:
        f.write("#!/bin/bash\nexit 2\n")
    os.chmod(os.path.join(stub, "pgrep"), 0o755)
    env = dict(os.environ, PATH=stub + os.pathsep + os.environ["PATH"])
    r2 = subprocess.run(["bash", "-c", chk], capture_output=True, text=True, timeout=30, env=env)
    assert r2.returncode == 7 and r2.stdout.strip() == "PROBE_UNAVAILABLE", (r2.returncode, r2.stdout)
    return "working probe: PWC_NONE; pgrep exit 2: PROBE_UNAVAILABLE rc 7 (never 'none')"


def a2b_block():
    src = open(RUNBOOK).read()
    return src[src.index("# --- A2B-PROVISION BEGIN ---"):src.index("# --- A2B-PROVISION END ---")]


def a2b_case(tag, rc_tm, rc_tp):
    base = os.path.join(OUT, "a2b_" + tag)
    trees = {}
    for name, rc in (("tm", rc_tm), ("tp", rc_tp)):
        t = os.path.join(base, name)
        os.makedirs(os.path.join(t, "scripts"))                 # NO logs/stage4: a fresh tree
        with open(os.path.join(t, "scripts", "provision_dataset_fleet.py"), "w") as f:
            f.write(f"print('node rrig6600: PASS (stub)'); raise SystemExit({rc})\n")
        trees[name] = t
    script = os.path.join(base, "a2b.sh")
    with open(script, "w") as f:
        f.write(a2b_block())
    env = dict(os.environ, TM=trees["tm"], TP=trees["tp"])
    r = subprocess.run(["bash", script], cwd=base, capture_output=True, text=True, timeout=60, env=env)
    ok = {n: os.path.exists(os.path.join(t, "logs", "stage4", "provision.ok")) for n, t in trees.items()}
    logs = {n: os.path.exists(os.path.join(t, "logs", "stage4", "provision.log")) for n, t in trees.items()}
    return r, ok, logs


def a2b_ok():
    r, ok, logs = a2b_case("ok", 0, 0)
    assert ok == {"tm": True, "tp": True} and logs == {"tm": True, "tp": True}, (ok, logs, r.stdout)
    return "fresh trees (no logs/stage4): both logs kept, both provision.ok written"


def a2b_fail():
    r, ok, logs = a2b_case("fail", 1, 0)
    assert ok == {"tm": False, "tp": False}, (ok, r.stdout)
    assert "NOT confirmed - STOP" in r.stdout and logs["tm"], r.stdout
    return "provisioner rc 1 on the first tree: STOP, no provision.ok anywhere, its log kept"


GATES = [
    ("L-APPLY", g_apply),
    ("L-STRUCTURE", g_structure),
    ("L-SAMPLER-PATH", g_sampler_path),
    ("P1 valid params, empty staging", lambda: p_case("p1", lambda p: p, None)),
    ("P2 params file missing", lambda: p_case("p2", lambda p: p, "cannot read params", write_params=False)),
    ("P3 params not JSON", lambda: p_case("p3", lambda p: p, "cannot read params", raw="{not json")),
    ("P4 staging_dir is the live one", lambda: p_case(
        "p4", lambda p: dict(p, staging_dir=LIVE_STAGING), "overlaps the live staging_dir")),
    ("P5 staging_dir under the live one", lambda: p_case(
        "p5", lambda p: dict(p, staging_dir=os.path.join(LIVE_STAGING, "stage4")),
        "overlaps the live staging_dir")),
    ("P6 staging not empty", lambda: p_case(
        "p6", lambda p: p, "is not empty",
        staging_fn=lambda t: (lambda d: (open(os.path.join(d, "miner_ledger.db"), "w").close(), d)[1])(ok_staging(t)))),
    ("P7 staging missing", lambda: p_case(
        "p7", lambda p: p, "does not exist", staging_fn=lambda t: os.path.join(OUT, t + "_nope"))),
    ("P8 not a RANGE-MINER run", lambda: p_case(
        "p8", lambda p: dict(p, use_range_miner=False, use_persistent_workers=True), "not a RANGE-MINER run")),
    ("P9 a pin missing", lambda: p_case(
        "p9", lambda p: {k: v for k, v in p.items() if k != "warm_start_skip_min"}, "params lack")),
    ("P10 halt present: abort, never cleared", p10_halt),
    ("L-EMPTY-STAGING", g_empty_staging_ledger),
    ("C3-VALID", c3_valid),
    ("C3-MISSING-PARAMS", lambda: c3_refused("missing", write=False)),
    ("C3-MALFORMED-PARAMS", lambda: c3_refused("malformed", raw="{not json")),
    ("C3-WRONG-ROUTE (RANGE-MINER params)", lambda: c3_refused(
        "wrongroute", raw=json.dumps(dict(PWC_PARAMS, use_range_miner=True, use_persistent_workers=False)))),
    ("C3-BAD-CD", lambda: c3_refused("badcd", bad_cd=True)),
    ("C5B-CHECK probe (Beta X-PWC-SHUTDOWN R1 Q7)", c5b_check),
    ("A2B-OK fresh trees keep their logs", a2b_ok),
    ("A2B-FAIL provisioner failure stops", a2b_fail),
]


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    os.makedirs(PARAMS_STAGING())
    print("VERIFICATION test_stage4_miner_launch START", flush=True)
    passed = unavailable = 0
    for name, fn in GATES:
        try:
            out = fn()
            if isinstance(out, str) and out.startswith("UNAVAILABLE"):
                print(f"[UNAVAILABLE] {name}: {out}", flush=True)
                unavailable += 1
            else:
                print(f"[PASS] {name}: {out}", flush=True)
                passed += 1
        except Exception as exc:                                    # noqa: BLE001
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)
    failed = len(GATES) - passed - unavailable
    print(f"\n{passed}/{len(GATES)} PASS, {unavailable} UNAVAILABLE, {failed} FAIL")
    print("RESULT: " + ("FAIL" if failed else "INCOMPLETE" if unavailable else "PASS"))
    print("VERIFICATION test_stage4_miner_launch COMPLETE", flush=True)
    return 0 if passed == len(GATES) else 1


if __name__ == "__main__":
    sys.exit(main())
