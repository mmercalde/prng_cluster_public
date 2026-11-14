#!/usr/bin/env python3
"""Gates for scripts/survivor_volume_gate.py and utils/survivor_volume_estimator.py —
V-A..V-M and four mutants (V-MUT-1..4). CPU only.

The sampler and the GPU probe are injected. The dataset is a synthetic file written
into a throwaway directory under data/survivor_volume_gate_tests/ (removed at the
end), and each cap-file case uses its own throwaway git repository, as
tests/test_run_authorization_gate.py does. Nothing reaches a GPU, a rig or live state.
The residues, the payload and the phase table still come from the PRODUCTION
loader, builder and workflow table, on the synthetic file.

Prints COMPLETION SENTINEL: PASS only when every gate ran and passed and every
mutant was applied and turned its target gate red.
"""

import builtins
import contextlib
import copy
import importlib.util
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import survivor_volume_gate as LIVE                                 # noqa: E402

GATE_SRC = os.path.join(REPO, "scripts", "survivor_volume_gate.py")
EST_SRC = os.path.join(REPO, "utils", "survivor_volume_estimator.py")
WORK_PARENT = os.path.join(REPO, "data", "survivor_volume_gate_tests")

S26 = 1 << 26
CAP = {"cap_survivors_per_phase": 23515, "compare": "point",
       "min_sample_seeds": S26, "gpu_process_allowlist": ["sunshine"],
       "evidence": "attempt 9 phase 4, run distributed_config_t1_554463d3"}
GPU_OK = {"status": "OK", "gpus": ["0, FAKE GPU, 12288 MiB"],
          "processes": [{"pid": "2419", "name": "sunshine", "used_memory": "248 MiB"}]}
DROP = object()
CTX = {}            # dataset path, filled in main()


def git(repo, *args):
    p = subprocess.run(["git", "-C", repo, "-c", "user.name=test",
                        "-c", "user.email=test@example.invalid", *args],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"git {args} failed: {p.stderr}")
    return p.stdout


def write_dataset(root):
    rng = random.Random(20261114)
    recs = []
    for day in range(120):
        for sess in ("evening", "midday"):
            recs.append({"date": f"2000-{1 + day // 28:02d}-{1 + day % 28:02d}",
                         "session": sess, "draw": rng.randrange(1000)})
    path = os.path.join(root, "synthetic_daily3.json")
    with open(path, "w") as fh:
        json.dump(recs, fh)
    return path


def params(lo=0, hi=1 << 31, **kw):
    d = {"seed_start": lo, "seed_end_exclusive": hi, "prng_type": "java_lcg",
         "window_size": 12, "window_anchor": 25, "generator_phase": 0,
         "sessions": ["midday"], "forward_threshold": 0.71, "reverse_threshold": 0.47,
         "skip_min": 6, "skip_max": 99, "dataset": CTX["dataset"],
         "test_both_modes": True}
    for k, v in kw.items():
        if v is DROP:
            d.pop(k, None)
        else:
            d[k] = v
    return d


class FakeSampler:
    """hits: {phase: int}. mode alters the result for the failure cases."""

    def __init__(self, hits=None, mode=None):
        self.hits = hits or {}
        self.mode = mode
        self.requests = []

    def __call__(self, req):
        self.requests.append(copy.deepcopy(req))
        if self.mode == "raise":
            raise RuntimeError("injected sampler failure")
        if self.mode == "sleep":
            time.sleep(3.0)
        if self.mode == "malformed":
            return {"host_hits": "many"}
        iv = [list(p) for p in req["intervals"]]
        if self.mode == "outside":
            iv[0] = [req["intervals"][0][0] - 10 ** 9 if req["intervals"][0][0] >= 10 ** 9
                     else 1 << 40, req["intervals"][0][1]]
        h = self.hits.get(req["phase"], 0)
        bh = [0] * len(iv)
        if bh:
            bh[0] = h
        return {"host_hits": h, "kernel_hits": h + 1, "intervals": iv, "block_hits": bh,
                "residue_sha256": ("0" * 64 if self.mode == "residue"
                                   else req["payload"]["residue_sha256"]),
                "effective_threshold": req["payload"]["min_match_threshold"],
                "seconds": 0.01, "gpu": "fake:0", "kernel": {"family_name": req["family_name"]},
                "gpu_pool_peak_bytes": None}


class Repo:
    def __init__(self, root, cap=CAP, commit=True, raw=None):
        self.repo = tempfile.mkdtemp(prefix="svg_", dir=root)
        git(self.repo, "init", "-q")
        with open(os.path.join(self.repo, "README"), "w") as fh:
            fh.write("throwaway\n")
        git(self.repo, "add", "README")
        git(self.repo, "commit", "-q", "-m", "init")
        self.path = os.path.join(self.repo, LIVE.CAP_REL)
        if cap is not None or raw is not None:
            self.write(cap, raw)
            if commit:
                git(self.repo, "add", "-f", LIVE.CAP_REL)
                git(self.repo, "commit", "-q", "-m", "cap")

    def write(self, cap, raw=None):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as fh:
            fh.write(raw if raw is not None else json.dumps(cap, indent=1) + "\n")

    def run(self, G, p, sampler=None, probe=GPU_OK, extra=()):
        rec = os.path.join(self.repo, "record.json")
        argv = ["--params", json.dumps(p), "--record-out", rec, *extra]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = G.main(argv, repo_root=self.repo,
                        sampler=sampler if sampler is not None else FakeSampler(),
                        gpu_probe=lambda: probe)
        record = None
        if os.path.exists(rec):
            with open(rec) as fh:
                record = json.load(fh)
        return rc, buf.getvalue(), record


def C(**kw):
    d = dict(CAP)
    for k, v in kw.items():
        if v is DROP:
            d.pop(k)
        else:
            d[k] = v
    return d


def proceed(r):
    return r[0] == 0 and "PROCEED" in r[1] and r[2] and r[2]["verdict"] == "PROCEED"


def verdict(r, word, *needles):
    return (r[0] == 1 and f": {word}" in r[1] and r[2] is not None
            and r[2]["verdict"] == word and all(n in r[1] for n in needles))


# ---------------------------------------------------------------------------
# gates. Each takes (G, root) where G is the gate module (live or mutant).
# ---------------------------------------------------------------------------

def g_VA(G, root):
    r = Repo(root).run(G, params(), FakeSampler({1: 0, 2: 0, 3: 3, 4: 700}))
    rec = r[2] or {}
    ok = (proceed(r) and rec.get("estimate") and len(rec["estimate"]["phases"]) == 4
          and rec["cap_sha256"] and "LOAD" in rec["verdict_kind"]
          and rec["estimate"]["dataset_sha256"] and rec["repository_commit"])
    return ok, r[1]


def g_VB(G, root):
    # range == sample == 2^26, so scale is exactly 1 and projection == hits.
    rp = Repo(root)
    at = rp.run(G, params(0, S26), FakeSampler({1: 0, 2: 0, 3: 23515, 4: 0}),
                extra=["--phase4-sample-seeds", str(S26)])
    over = rp.run(G, params(0, S26), FakeSampler({1: 0, 2: 0, 3: 23516, 4: 0}),
                  extra=["--phase4-sample-seeds", str(S26)])
    ok = proceed(at) and verdict(over, "REFUSE", "phase 3 (java_lcg_hybrid)", "> cap 23515")
    return ok, at[1] + "\n---\n" + over[1]


def g_VC(G, root):
    hits = {1: 0, 2: 0, 3: 0, 4: 2900}   # phase 4 sampled 2^28: scale 8 -> 23200; upper ~24060
    up = Repo(root, C(compare="upper95")).run(G, params(), FakeSampler(hits))
    pt = Repo(root, C(compare="point")).run(G, params(), FakeSampler(hits))
    ph4 = (pt[2] or {}).get("estimate", {}).get("phases", [{}] * 4)[3]
    ok = (verdict(up, "REFUSE", "phase 4", "upper95 figure") and proceed(pt)
          and ph4.get("projection", 1e18) <= 23515 < ph4.get("upper95", 0))
    return ok, up[1] + "\n---\n" + pt[1]


def g_VD(G, root):
    r = Repo(root).run(G, params(), FakeSampler({}))
    ph = (r[2] or {}).get("estimate", {}).get("phases", [])
    ok = (proceed(r) and len(ph) == 4 and all(p["projection"] == 0 for p in ph)
          and all(p["upper95"] > 0 for p in ph)
          and abs(ph[0]["upper95"] - 3.6888794541 * 32) < 1e-3)
    return ok, r[1] + json.dumps([(p["projection"], p["upper95"]) for p in ph])


def g_VE(G, root):
    outs = {
        "absent": Repo(root, cap=None).run(G, params()),
        "unparseable": Repo(root, raw="{not json").run(G, params()),
        "missing": Repo(root, C(compare=DROP)).run(G, params()),
        "untracked": Repo(root, commit=False).run(G, params()),
    }
    rp = Repo(root)
    rp.write(C(cap_survivors_per_phase=10 ** 9))   # valid content, differs from HEAD
    outs["modified"] = rp.run(G, params(), FakeSampler({4: 3000}))
    need = {"absent": "ABSENT", "unparseable": "unparseable", "missing": "missing field",
            "untracked": "UNTRACKED", "modified": "MODIFIED"}
    ok = all(verdict(outs[k], "REFUSE", need[k]) for k in outs)
    return ok, "\n---\n".join(f"{k}: {v[1]}" for k, v in outs.items())


def g_VF(G, root):
    rp = Repo(root)
    miss = rp.run(G, params(window_anchor=DROP))
    rng = rp.run(G, params(forward_threshold={"min": 0.3, "max": 0.75}))
    ok = (verdict(miss, "REFUSE", "missing", "window_anchor")
          and verdict(rng, "REFUSE", "forward_threshold", "range"))
    return ok, miss[1] + "\n---\n" + rng[1]


def g_VG(G, root):
    rp = Repo(root)
    small = rp.run(G, params(), extra=["--sample-seeds", str(S26 - 1)])
    outside = rp.run(G, params(), FakeSampler(mode="outside"))
    ok = (verdict(small, "REFUSE", "min_sample_seeds")
          and verdict(outside, "REFUSE", "outside the requested range"))
    return ok, small[1] + "\n---\n" + outside[1]


def g_VH(G, root):
    r = Repo(root).run(G, params(), FakeSampler(mode="residue"))
    return verdict(r, "REFUSE", "ResidueMismatch"), r[1]


def g_VI(G, root):
    rp = Repo(root)
    absent = rp.run(G, params(), probe={"status": "ABSENT", "detail": "no GPU",
                                        "gpus": [], "processes": []})
    busy = rp.run(G, params(), probe={"status": "OK", "gpus": ["0, G"], "processes": [
        {"pid": "2419", "name": "sunshine"}, {"pid": "77", "name": "/usr/bin/python3"}]})
    allow = rp.run(G, params())          # positive control: sunshine only -> proceeds
    me = rp.run(G, params(), probe={"status": "OK", "gpus": ["0, G"], "processes": [
        {"pid": str(os.getpid()), "name": "python3"}]})   # own context is not "busy"
    ok = (verdict(absent, "UNAVAILABLE", "ABSENT") and verdict(busy, "UNAVAILABLE", "busy")
          and proceed(allow) and proceed(me))
    return ok, "\n---\n".join(x[1] for x in (absent, busy, allow, me))


def g_VJ(G, root):
    rp = Repo(root)
    raised = rp.run(G, params(), FakeSampler(mode="raise"))
    slow = rp.run(G, params(), FakeSampler(mode="sleep"),
                  extra=["--sampler-timeout-s", "0.5"])
    bad = rp.run(G, params(), FakeSampler(mode="malformed"))
    ok = (verdict(raised, "ERROR", "SamplerFailed") and verdict(slow, "ERROR", "SamplerTimeout")
          and verdict(bad, "ERROR", "SamplerMalformed"))
    return ok, "\n---\n".join(x[1] for x in (raised, slow, bad))


BLOCK_MODULE = '''
import time


def block(request):           # importable by module name, as the child requires
    time.sleep(3600)
'''

# Deliberately WITHOUT an `if __name__ == "__main__"` guard: a child that re-imported
# the caller's main module (multiprocessing spawn) would run this file twice, and
# DRIVER_START would be printed twice. V-J2 asserts it is printed exactly once.
LIFETIME_DRIVER = r'''
import json, os, sys, importlib.util
print("DRIVER_START", flush=True)
GATE, EST, REPO_ROOT, MODE, PARAMS = sys.argv[1:6]
sys.path.insert(0, os.path.dirname(os.path.dirname(GATE)))
sys.path.insert(0, os.path.dirname(GATE))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svg_block import block


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


est = load(EST, "utils.survivor_volume_estimator")
gate = load(GATE, "survivor_volume_gate_under_test")
sampler = est.IsolatedProcessSampler(block) if MODE == "process" else block
probe = {"status": "OK", "gpus": ["0, FAKE"], "processes": []}
rc = gate.main(["--params", PARAMS, "--sampler-timeout-s", "1"], repo_root=REPO_ROOT,
               sampler=sampler, gpu_probe=lambda: probe)
child = getattr(sampler, "last_child", None)
alive = None
if child:
    try:
        os.kill(child["pid"], 0)
        alive = True
    except OSError:
        alive = False
print("LIFETIME " + json.dumps({"rc": rc, "child": child, "child_alive": alive}),
      flush=True)
sys.exit(rc)
'''


def _lifetime(G, root, mode):
    """Run the gate in a SEPARATE process against a sampler that blocks for an hour,
    with a 1 s sampler timeout. Returns (exited_in_time, rc, info, output)."""
    rp = Repo(root)
    drv = os.path.join(root, "lifetime_driver.py")
    with open(drv, "w") as fh:
        fh.write(LIFETIME_DRIVER)
    with open(os.path.join(root, "svg_block.py"), "w") as fh:
        fh.write(BLOCK_MODULE)
    est_path = getattr(G.SVE, "__file__", EST_SRC)
    # Output goes to a FILE, not a pipe: a surviving child inherits the driver's
    # stdout, and a pipe would make this wait for the child instead of the driver.
    out_path = os.path.join(root, f"lifetime_{mode}_{time.time_ns()}.out")
    with open(out_path, "w") as out_fh:
        proc = subprocess.Popen([sys.executable, "-u", drv, G.__file__, est_path, rp.repo,
                                 mode, json.dumps(params())], stdout=out_fh,
                                stderr=subprocess.STDOUT, cwd=root)
        try:
            rc = proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            rc = None
    with open(out_path) as fh:
        text = fh.read()
    info = None
    for ln in text.splitlines():
        if ln.startswith("LIFETIME "):
            info = json.loads(ln[len("LIFETIME "):])
    if info and info.get("child_alive"):     # a mutant left its child running: reap it
        import signal
        try:
            os.kill(info["child"]["pid"], signal.SIGKILL)
        except OSError:
            pass
    if rc is None:
        return False, None, info, "gate process did NOT exit within 60 s\n" + text
    return True, rc, info, text


def g_VJ2(G, root):
    """Beta R1 Q6: the gate PROCESS exits on a blocked sampler; the isolated child is
    dead when main returns. An in-process return assertion is not enough."""
    pe, prc, pinfo, pout = _lifetime(G, root, "process")
    te, trc, tinfo, tout = _lifetime(G, root, "thread")
    ok = (pe and prc == 1 and pinfo is not None and pinfo["rc"] == 1
          and pinfo["child"] is not None and pinfo["child"].get("timed_out") is True
          and pinfo["child_alive"] is False and "terminated" in pout
          and pout.count("DRIVER_START") == 1 and tout.count("DRIVER_START") == 1
          and te and trc == 1 and tinfo is not None and tinfo["rc"] == 1
          and "SamplerTimeout" in tout)
    return ok, f"process: exited={pe} rc={prc} {pinfo}\n{pout}\n---\nthread: exited={te} rc={trc} {tinfo}\n{tout}"


def g_VK(G, root):
    rp = Repo(root)
    a, b = FakeSampler(), FakeSampler()
    rp.run(G, params(), a)
    rp.run(G, params(), b)
    c = FakeSampler()
    rp.run(G, params(reverse_threshold=0.48), c)
    ia = [r["intervals"] for r in a.requests]
    ib = [r["intervals"] for r in b.requests]
    ic = [r["intervals"] for r in c.requests]
    spread = all(len(x) == 8 for x in ia) and len({tuple(map(tuple, x)) for x in ia}) >= 1
    ok = len(ia) == 4 and ia == ib and ia != ic and spread
    return ok, json.dumps({"a": ia[:1], "c": ic[:1]})


def g_VL(G, root):
    rp = Repo(root)
    cases = [((0, 1 << 31), 3, 96.0),
             ((1 << 31, 1 << 32), 3, 96.0),
             ((1000, 1000 + 3_000_000_000), 5, 5 * 3_000_000_000 / S26)]
    outs, ok = [], True
    for (lo, hi), h, want in cases:
        r = rp.run(G, params(lo, hi), FakeSampler({3: h}))
        ph = (r[2] or {}).get("estimate", {}).get("phases", [])
        p3 = next((p for p in ph if p["phase"] == 3), None)
        inside = p3 is not None and all(lo <= s and s + c <= hi for s, c in p3["intervals"])
        good = proceed(r) and p3 is not None and abs(p3["projection"] - want) < 1e-6 and inside
        ok = ok and good
        outs.append(f"[{lo},{hi}) want {want} got {p3 and p3['projection']} inside={inside}")
    return ok, "\n".join(outs)


def g_VM(G, root):
    rp = Repo(root)
    p = params()
    gate = rp.run(G, p, FakeSampler({3: 3, 4: 700}))
    opened = []
    real_open = builtins.open

    def spy(file, *a, **kw):
        opened.append(str(file))
        return real_open(file, *a, **kw)

    geo = {k: v for k, v in p.items() if k not in ("seed_start", "seed_end_exclusive")}
    builtins.open = spy
    try:
        est = G.SVE.estimate_volume(geo, p["seed_start"], p["seed_end_exclusive"],
                                    sample_seeds={1: S26, 2: S26, 3: S26, 4: 4 * S26},
                                    sampler=FakeSampler({3: 3, 4: 700}))
    finally:
        builtins.open = real_open
    keys = ("phase", "hits", "kernel_hits", "projection", "lower95", "upper95", "intervals")
    gp = [{k: x[k] for k in keys} for x in (gate[2] or {}).get("estimate", {}).get("phases", [])]
    ep = [{k: x[k] for k in keys} for x in est["phases"]]
    no_cap = not any(o.endswith(".cap") for o in opened)
    no_verdict = "verdict" not in json.dumps(est) and "cap" not in est
    ok = proceed(gate) and gp == ep and len(ep) == 4 and no_cap and no_verdict
    return ok, f"gate==fn {gp == ep} no_cap {no_cap} no_verdict {no_verdict} opened {opened}"


def g_VN(G, root):
    """Beta R1 finding 2: a whole-range request whose size is not a multiple of the
    block count is sampled exactly (census), not refused; blocks tile the range."""
    lo, hi = 1000, 1000 + S26 + 1
    smp = FakeSampler({3: 7})
    r = Repo(root).run(G, params(lo, hi), smp,
                       extra=["--sample-seeds", str(S26 + 1),
                              "--phase4-sample-seeds", str(S26 + 1)])
    ivs = [q["intervals"] for q in smp.requests]
    tiles = all(sum(c for _s, c in iv) == hi - lo and iv[0][0] == lo
                and all(a[0] + a[1] == b[0] for a, b in zip(iv, iv[1:]))
                and iv[-1][0] + iv[-1][1] == hi for iv in ivs)
    p3 = next((p for p in (r[2] or {}).get("estimate", {}).get("phases", [])
               if p["phase"] == 3), None)
    ok = proceed(r) and len(ivs) == 4 and tiles and p3 is not None \
        and p3["projection"] == 7 and p3["block_hits"][0] == 7
    return ok, r[1] + json.dumps(ivs[:1])


GATES = [("V-A", g_VA), ("V-B", g_VB), ("V-C", g_VC), ("V-D", g_VD), ("V-E", g_VE),
         ("V-F", g_VF), ("V-G", g_VG), ("V-H", g_VH), ("V-I", g_VI), ("V-J", g_VJ),
         ("V-J2", g_VJ2), ("V-K", g_VK), ("V-L", g_VL), ("V-M", g_VM), ("V-N", g_VN)]

# (description, file, old, new, target gates)
MUTANTS = {
    "V-MUT-1": ("UNAVAILABLE mapped to PROCEED", "gate",
                "        return done(V_UNAVAILABLE, [why])\n",
                "        return done(V_PROCEED, [why])\n", ["V-I"]),
    "V-MUT-2": ("'>' changed to '>=' at the cap", "gate",
                "        if fig > cap:\n", "        if fig >= cap:\n", ["V-B"]),
    "V-MUT-3": ("projection scaled by the wrong range", "est",
                "    scale = range_size / sample_size\n",
                "    scale = (1 << 31) / sample_size\n", ["V-L"]),
    "V-MUT-4": ("cap-file ownership check removed", "gate",
                "    status, reason, data = EF.tracked_unmodified(repo_root, CAP_REL, runner)\n",
                "    _p = os.path.join(repo_root, CAP_REL)\n"
                "    data = open(_p, 'rb').read() if os.path.isfile(_p) else None\n"
                "    status, reason = ('OK', None) if data is not None else ('ABSENT', 'absent')\n",
                ["V-E"]),
    "V-MUT-5": ("sampler thread made non-daemon (the R1 defect)", "est",
                '    t = threading.Thread(target=run, name="survivor-volume-sampler", daemon=True)\n',
                '    t = threading.Thread(target=run, name="survivor-volume-sampler", daemon=False)\n',
                ["V-J2"]),
    "V-MUT-6": ("child not terminated on timeout", "est",
                "            proc.terminate()\n            try:\n"
                "                proc.wait(self.TERM_GRACE_S)\n"
                "            except subprocess.TimeoutExpired:\n                proc.kill()\n",
                "            pass\n            try:\n"
                "                proc.wait(self.TERM_GRACE_S)\n"
                "            except subprocess.TimeoutExpired:\n                pass\n",
                ["V-J2"]),
}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_mutant(name, which, old, new, root):
    src_path = GATE_SRC if which == "gate" else EST_SRC
    src = open(src_path, encoding="utf-8").read()
    n = src.count(old)
    if n != 1:
        raise RuntimeError(f"{name}: mutation anchor occurs {n} times (need 1) — "
                           f"MUTANT NOT APPLIED")
    tag = name.replace("-", "_")
    path = os.path.join(root, f"{tag}_{os.path.basename(src_path)}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src.replace(old, new))
    mod = _load(path, f"svg_{tag}")
    if which == "gate":
        return mod
    gate = _load(GATE_SRC, f"svg_{tag}_gate")       # a fresh gate bound to the mutant
    gate.SVE = mod
    return gate


def main():
    os.makedirs(WORK_PARENT, exist_ok=True)
    root = tempfile.mkdtemp(prefix="test_svg_", dir=WORK_PARENT)
    CTX["dataset"] = write_dataset(root)
    results = []
    try:
        for name, fn in GATES:
            try:
                ok, out = fn(LIVE, root)
            except Exception as exc:                                # noqa: BLE001
                ok, out = False, f"EXCEPTION {type(exc).__name__}: {exc}"
            results.append((name, ok))
            print(f"[{'PASS' if ok else 'FAIL'}] {name}", flush=True)
            if not ok:
                print("    " + out.replace("\n", "\n    "), flush=True)
        gate_fns = dict(GATES)
        for mname, (desc, which, old, new, targets) in MUTANTS.items():
            try:
                mod = load_mutant(mname, which, old, new, root)
                reds = {}
                for t in targets:
                    try:
                        reds[t] = not gate_fns[t](mod, root)[0]
                    except Exception as exc:                        # noqa: BLE001
                        reds[t] = f"EXCEPTION {type(exc).__name__}: {exc}"
                killed = all(v is True for v in reds.values())
                detail = ", ".join(f"{t} {'red' if r is True else ('STILL GREEN' if r is False else r)}"
                                   for t, r in reds.items())
            except Exception as exc:                                # noqa: BLE001
                killed, detail = False, f"ERROR {type(exc).__name__}: {exc}"
            results.append((mname, killed))
            print(f"[{'PASS' if killed else 'FAIL'}] {mname} ({desc}): {detail}", flush=True)
            if os.environ.get("SVG_SHOW_MUTANT_OUTPUT"):
                for t in targets:
                    print("    " + gate_fns[t](mod, root)[1].replace("\n", "\n    ")[:1500],
                          flush=True)
    finally:
        shutil.rmtree(root, ignore_errors=True)
        try:
            os.rmdir(WORK_PARENT)
        except OSError:
            pass
    expected = len(GATES) + len(MUTANTS)
    passed = sum(1 for _, ok in results if ok)
    print(f"\n{passed}/{expected} gates passed", flush=True)
    if len(results) == expected and passed == expected:
        print("COMPLETION SENTINEL: PASS", flush=True)
        return 0
    print("COMPLETION SENTINEL: FAIL", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
