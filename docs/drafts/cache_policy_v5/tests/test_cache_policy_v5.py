#!/usr/bin/env python3
"""DRAFT tests for X-CACHE-POLICY v5. CPU only; no rig; no real ssh; no real pkill; nothing outside the fixture.

Two modes, same checks:
  pre-landing (default)      builds a LANDED-LAYOUT tree in a temp dir: the three diffs applied to copies of
                             the committed scripts/launch_fleet_manual.sh and gate12_launch.sh, the three new
                             scripts copied to scripts/, Route A taken from tests/fixtures/run4_launch.sh.
  post-landing               python3 -B test_cache_policy_v5.py --landed <repo> [--route-a <installed run4_launch.sh>]
                             runs against <repo>/scripts/*.py, <repo>/scripts/launch_fleet_manual.sh and
                             <repo>/gate12_launch.sh as installed; Route A from --route-a or the fixture.
  mutants                    ... mutants [--landed <repo>]
The fixture root may live anywhere (including /tmp): the stub transport shields the fixture's own path before it
rewrites a CT's /tmp and cache root, and rewrites only at the start of a path token (Beta X-CACHE-POLICY R1 fix).
"""
import argparse
import base64
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
V5 = HERE.parent
REPO = V5.parents[2]
FIXTURE_ROUTE_A = HERE / "fixtures/run4_launch.sh"
RESULTS = []
T = {}          # the tree under test: launcher/gate12/run4 texts, verify/join/footprint paths


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"   <- {str(detail)[:300]}"))


def run(cmd, **kw):
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    kw.setdefault("timeout", 150)
    return subprocess.run(cmd, **kw)


# ======================================================================================= tree setup
def apply_patch(src: Path, diff: Path, dst: Path):
    shutil.copy(src, dst)
    p = run(["patch", "-s", "--no-backup-if-mismatch", str(dst), str(diff)])
    assert p.returncode == 0, f"{diff.name} does not apply to {src}: {p.stdout}{p.stderr}"


def build_tree(work: Path, landed: Path = None, route_a: Path = None):
    if landed:
        root = landed
        T.update(launcher=(root / "scripts/launch_fleet_manual.sh").read_text(),
                 gate12=(root / "gate12_launch.sh").read_text())
        for k, f in (("verify", "cache_records_verify.py"), ("join", "gate12_cache_join_gate.py"), ("footprint", "cache_footprint.py")):
            T[k] = root / "scripts" / f
    else:
        root = work / "tree"
        (root / "scripts").mkdir(parents=True)
        apply_patch(REPO / "scripts/launch_fleet_manual.sh", V5 / "launch_fleet_manual.diff", root / "scripts/launch_fleet_manual.sh")
        apply_patch(REPO / "gate12_launch.sh", V5 / "gate12_launch.diff", root / "gate12_launch.sh")
        for f in ("cache_records_verify.py", "gate12_cache_join_gate.py", "cache_footprint.py"):
            shutil.copy(V5 / "scripts" / f, root / "scripts" / f)
        T.update(launcher=(root / "scripts/launch_fleet_manual.sh").read_text(), gate12=(root / "gate12_launch.sh").read_text())
        for k, f in (("verify", "cache_records_verify.py"), ("join", "gate12_cache_join_gate.py"), ("footprint", "cache_footprint.py")):
            T[k] = root / "scripts" / f
        base = Path("/home/michael/run4_launch.sh")
        if base.exists():               # consistency: the shipped fixture == the diff applied to today's baseline
            chk = work / "run4_check.sh"
            apply_patch(base, V5 / "run4_launch.diff", chk)
            check("Route A fixture equals run4_launch.diff applied to the installed baseline",
                  chk.read_text() == FIXTURE_ROUTE_A.read_text())
    T["run4"] = (route_a or FIXTURE_ROUTE_A).read_text()
    for n in ("launcher", "gate12", "run4"):
        r = run(["bash", "-n"], input=T[n])
        check(f"bash -n {n}", r.returncode == 0, r.stderr)


# ========================================================================================= U-VERIFY
def rec_line(tag, host, gpu, path, pid, t="1700000000.123", **over):
    f = dict(tag=tag, host=host, gpu=gpu, policy="per_run", path=path, created="exclusive", entries="0",
             t_prep=t, pid=pid, env_CUPY_CACHE_DIR=path)
    f.update(over)
    for k in [k for k, v in f.items() if v is None]:
        del f[k]
    return "[cache] OK " + " ".join(f"{k}={v}" for k, v in f.items())


def u_verify(work: Path):
    print("U-VERIFY")
    tag, root = "case.1", "/tmp/cache.root"
    V = str(T["verify"])

    def fixture(sub, mutate=None, slots=None, exp_mut=None):
        d = work / "uv" / sub
        d.mkdir(parents=True)
        rows = []
        for i, (h, g) in enumerate((("rigA", "0"), ("rigA", "1"), ("rigB", "0"))):
            path = f"{root}/cupy_cache_run_{tag}_gpu{g}"
            log = d / f"dispatch_{h}_gpu{g}.log"
            lines = [rec_line(tag, h, g, path, str(100 + i)), f"[dispatch] started pid={100 + i} gpu={g}"]
            if mutate and i == 1:
                lines = mutate(lines, h, g, path, str(100 + i))
            log.write_text("\n".join(lines) + "\n")
            rows.append([f"{h}:gpu{g}", h, f"ct-{h}", g, path, str(log)])
        if exp_mut:
            rows = exp_mut(rows)
        (d / "expect.tsv").write_text("".join("\t".join(r) + "\n" for r in rows))
        return d, (slots if slots is not None else 3)

    def go(d, slots, out=None, pre=None, inject=None):
        out = out or d / "records.tsv"
        if pre:
            pre(out)
        args = ["--tag", tag, "--expect", str(d / "expect.tsv"), "--expected-slots", str(slots), "--out", str(out)]
        if inject:
            code = "import sys, os, runpy\nsys.argv = sys.argv[1:]\n" + inject + "\nrunpy.run_path(sys.argv[0], run_name='__main__')\n"
            return run([sys.executable, "-B", "-c", code, V] + args), out
        return run([sys.executable, "-B", V] + args), out

    d, n = fixture("clean")
    r, out = go(d, n)
    rows = out.read_text().splitlines() if out.exists() else []
    check("clean control publishes exactly once; equal paths on two CTs accepted", r.returncode == 0 and len(rows) == 4, r.stderr)
    r2, _ = go(d, n)
    check("pre-existing TSV refused (a tag is written once)", r2.returncode == 3, r2.stderr)
    cases = {
        "dotted tag mismatch (caseX1 for case.1)": lambda L, h, g, p, pid: [L[0].replace("tag=case.1", "tag=caseX1"), L[1]],
        "dotted root mismatch (/tmp/cacheYroot)": lambda L, h, g, p, pid: [L[0].replace("/tmp/cache.root", "/tmp/cacheYroot"), L[1]],
        "missing host field": lambda L, h, g, p, pid: [rec_line(tag, None, g, p, pid), L[1]],
        "empty host value": lambda L, h, g, p, pid: [rec_line(tag, "", g, p, pid), L[1]],
        "missing t_prep field": lambda L, h, g, p, pid: [rec_line(tag, h, g, p, pid, t=None), L[1]],
        "non-decimal t_prep": lambda L, h, g, p, pid: [rec_line(tag, h, g, p, pid, t="soon"), L[1]],
        "wrong-GPU ACK": lambda L, h, g, p, pid: [L[0], f"[dispatch] started pid={pid} gpu=7"],
        "diagnostic containing 'started pid=' instead of an ACK": lambda L, h, g, p, pid: [L[0], f"[dispatch] worker failed; was started pid={pid} gpu={g}"],
        "a full ACK embedded in a longer line": lambda L, h, g, p, pid: [L[0], f"note: {L[1]} (replayed)"],
        "absent ACK": lambda L, h, g, p, pid: [L[0]],
        "duplicate ACK": lambda L, h, g, p, pid: [L[0], L[1], L[1]],
        "duplicate record": lambda L, h, g, p, pid: [L[0], L[0], L[1]],
        "extra field": lambda L, h, g, p, pid: [L[0] + " note=x", L[1]],
        "host suffix (rigAx for rigA)": lambda L, h, g, p, pid: [L[0].replace(f"host={h} ", f"host={h}x "), L[1]],
        "path suffix": lambda L, h, g, p, pid: [L[0].replace(f"path={p} ", f"path={p}x "), L[1]],
        "record host is not the dispatch host": lambda L, h, g, p, pid: [rec_line(tag, "rigZ", g, p, pid), L[1]],
        "record pid != ACK pid": lambda L, h, g, p, pid: [L[0], f"[dispatch] started pid=999 gpu={g}"],
        "env != path": lambda L, h, g, p, pid: [L[0].replace(f"env_CUPY_CACHE_DIR={p}", "env_CUPY_CACHE_DIR=/tmp/cupy_cache_gpu1"), L[1]],
        "policy shared in record": lambda L, h, g, p, pid: [L[0].replace("policy=per_run", "policy=shared"), L[1]],
    }
    for i, (name, mut) in enumerate(cases.items()):
        d, n = fixture(f"c{i}", mutate=mut)
        r, out = go(d, n)
        check(f"refuses: {name}", r.returncode == 1 and not out.exists() and "DEFECT" in r.stderr, r.stderr + r.stdout)
    d, n = fixture("slots", slots=4)
    r, out = go(d, n)
    check("refuses: expectation count != expected slots", r.returncode == 3 and not out.exists(), r.stderr)
    d, n = fixture("dupexp", exp_mut=lambda rows: rows[:2] + [rows[0][:]])
    r, out = go(d, n)
    check("refuses: duplicate slot in expectations", r.returncode == 3 and not out.exists(), r.stderr)
    d, n = fixture("dupendpath", exp_mut=lambda rows: rows[:2] + [rows[2][:2] + [rows[0][2], rows[2][3], rows[0][4], rows[2][5]]])
    r, out = go(d, n)
    check("refuses: duplicate (endpoint, path) in expectations", r.returncode == 3 and not out.exists(), r.stderr)
    d, n = fixture("unavail")
    r, out = go(d, n, pre=lambda o: o.mkdir())
    check("refuses: TSV path unavailable (a directory)", r.returncode == 3 and out.is_dir(), r.stderr)
    d, n = fixture("rodir")
    od = d / "ro"; od.mkdir(); od.chmod(0o500)
    try:
        r, out = go(d, n, out=od / "records.tsv")
        check("refuses: output directory not writable -> rc 2, nothing published", r.returncode == 2 and not out.exists(), r.stderr)
    finally:
        od.chmod(0o700)
    d, n = fixture("before_rename")
    r, out = go(d, n, inject="def boom(*a, **k): raise OSError('injected: before rename')\nos.replace = boom")
    check("refuses: failure before rename -> rc 2, no TSV, no temp left",
          r.returncode == 2 and not out.exists() and not list(d.glob("records.tsv.tmp.*")), r.stderr)
    d, n = fixture("interrupt")
    r, out = go(d, n, inject="def boom(*a, **k): raise KeyboardInterrupt()\nos.replace = boom")
    check("refuses: interrupt during publish -> rc 2, nothing published", r.returncode == 2 and not out.exists(), r.stderr)
    # [v5] failure AFTER the rename: the directory fsync fails
    dirfail = ("_o = os.open\n"
               "def o(p, f, *a, **k):\n"
               "    if os.path.isdir(p): raise OSError('injected: directory fsync failure after rename')\n"
               "    return _o(p, f, *a, **k)\n"
               "os.open = o")
    d, n = fixture("after_rename")
    r, out = go(d, n, inject=dirfail)
    check("[v5] failure after rename (dir fsync) -> rc 2 AND the published TSV is removed",
          r.returncode == 2 and not out.exists() and "failure after publication" in r.stderr, r.stderr)
    d, n = fixture("after_rename_nounlink")
    r, out = go(d, n, inject=dirfail + "\n_u = os.unlink\n"
                "def u(p, *a, **k):\n    if str(p).endswith('records.tsv'): raise OSError('injected: unlink fails')\n    return _u(p, *a, **k)\n"
                "os.unlink = u")
    check("[v5] failure after rename with unlink failing too -> rc 2, official name renamed to .INVALID",
          r.returncode == 2 and not out.exists() and Path(str(out) + ".INVALID").exists(), r.stderr)


# ======================================================================== stubs (transport, workers)
REWRITE_PY = r'''
import re, sys
c, ct, fix = sys.argv[1], sys.argv[2], sys.argv[3]
SH = "\x00FIXROOT\x00"
c = c.replace(fix, SH)                                   # shield the fixture's own path (may live under /tmp)
lead = r'(?<![^\s="\':(])'                              # rewrite only at the start of a path token
c = re.sub(lead + r'/tmp(?=[/\s"\';)]|$)', lambda m: SH + "/" + ct + "/tmp", c)
c = re.sub(lead + r'/FIXCACHE(?=[/\s"\';)]|$)', lambda m: SH + "/" + ct + "/cache", c)
c = re.sub(lead + r'/proc/', lambda m: (SH + "/" + ct + "/proc/") if sys.argv[4] == "1" else "/proc/", c)
sys.stdout.write(c.replace(SH, fix))
'''
UNWRITE_PY = r'''
import base64, sys
fix, ct = sys.argv[1], sys.argv[2]
def un(b):
    return b.replace(f"{fix}/{ct}/cache".encode(), b"/FIXCACHE").replace(f"{fix}/{ct}/tmp".encode(), b"/tmp")
out = []
for line in sys.stdin.buffer.read().split(b"\n"):
    p = line.split(b" ")
    if line.startswith(b"@@TFMJOIN@@ ") and len(p) in (3, 4):      # the join probe: map inside the payload too
        payload = un(base64.b64decode(p[3])) if len(p) == 4 else b""
        line = b" ".join(p[:3] + [base64.b64encode(payload)])
    out.append(un(line))
sys.stdout.buffer.write(b"\n".join(out))
'''
STUB_SSH = r'''#!/bin/bash
# stub transport: each fake CT gets its own filesystem view; paths are mapped back in what it returns.
tgt=""; while [ $# -gt 0 ]; do case "$1" in -o) shift 2;; -n|-T) shift;; michael@*|root@*) tgt=${1#*@}; shift; break;; *) shift;; esac; done
cmd="$*"
echo "$tgt" >> "$FIX/ssh_calls.log"
case "$tgt" in ct-*) ;; *) exit 0;; esac          # anything else (the real rig IPs in abort paths): never contacted
mkdir -p "$FIX/$tgt/tmp" "$FIX/$tgt/cache"
cmd2=$(python3 "$FIX/rewrite.py" "$cmd" "$tgt" "$FIX" "${FAKE_PROC:-0}") || exit 99
export CT_NAME="$tgt"
if [ -e "$FIX/transport_fails_after_output_$tgt" ]; then PATH="$FIX/$tgt/bin:$PATH" bash -c "$cmd2" | python3 "$FIX/unwrite.py" "$FIX" "$tgt"; exit 255; fi
PATH="$FIX/$tgt/bin:$PATH" bash -c "$cmd2" 2>&1 | python3 "$FIX/unwrite.py" "$FIX" "$tgt"
exit "${PIPESTATUS[0]}"
'''
STANDIN_PY = r'''
import datetime, json, os, subprocess, sys
args = sys.argv[1:]
fix = os.environ["FIX"]
g = args[args.index("--gpu-id") + 1] if "--gpu-id" in args else ""
ct = os.environ.get("CT_NAME", "local")
with open(os.path.join(fix, "worker_pids"), "a") as f:
    f.write(f"{os.getpid()}\n")
if "--sentinel-log-path" in args and "--run-nonce" in args:
    lp = args[args.index("--sentinel-log-path") + 1]
    nonce = args[args.index("--run-nonce") + 1]
    host = subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip()
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    rec = {"event": "SESSION_SENTINEL", "pid": os.getpid(), "run_nonce": nonce, "worker_id": f"{host}:gpu{g}", "log_path": lp}
    with open(lp, "a") as f:
        f.write(f"{ts} WARNING __main__ [MINER-SESSION] SESSION_SENTINEL {json.dumps(rec)}\n")
env = dict(os.environb)
mode = os.environ.get("FIXTURE_ENV_MODE", "")           # "<ct>:<gpu>:<mode>"
if mode and mode.rsplit(":", 1)[0] == f"{ct}:{g}":
    m = mode.rsplit(":", 1)[1]
    path = env.get(b"CUPY_CACHE_DIR", b"")
    if m == "wrong":
        env[b"CUPY_CACHE_DIR"] = b"/wrong"
    elif m == "multi":
        env[b"CUPY_CACHE_DIR"] = path + b"\nextra"
    elif m == "inside":
        env.pop(b"CUPY_CACHE_DIR", None); env[b"OTHER"] = b"x\nCUPY_CACHE_DIR=" + path
    elif m == "missing":
        env.pop(b"CUPY_CACHE_DIR", None)
    elif m == "nonutf":
        env[b"ZZ_BYTES"] = b"\xff\xfe\x80"
os.execve(sys.executable, [sys.executable, "-c", "import time; time.sleep(120)"] + args, env)
'''
STUB_PKILL = '#!/bin/bash\necho "pkill $*" >> "$FIX/pkill_calls.log"\nexit 0\n'
STUB_SENTINEL_GATE = 'import sys\nprint("[stub sentinel gate] PASS (fixture)")\nsys.exit(0)\n'
STUB_LIVENESS_GATE = r'''
import argparse, json, os, re, sys
ap = argparse.ArgumentParser(); ap.add_argument("--run-nonce"); ap.add_argument("--local-log-dir"); ap.add_argument("--evidence-json")
a = ap.parse_args()
d = a.local_log_dir
res = []
for line in open(os.path.join(d, f"cache_expect_{a.run_nonce}.tsv")):
    label, whost, ep, gpu, path, log = line.rstrip("\n").split("\t")
    m = re.search(r"^\[dispatch\] started pid=(\d+) gpu=\d+$", open(log).read(), re.M)
    pid = m.group(1)
    argv = [x for x in open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="replace").split("\0") if x]
    res.append({"worker_id": f"{whost}:gpu{gpu}", "endpoint": ep, "gpu": int(gpu), "pid": pid, "status": "OK",
                "local": False, "argv": argv, "log_path": f"/tmp/minerlogs/gpu{gpu}.log"})
if os.environ.get("FIXTURE_LIVENESS_BADPID"):
    res[0]["pid"] = "999999"
res.append({"worker_id": "zeus:gpu0", "endpoint": "localhost", "gpu": 0, "pid": "1", "status": "OK", "local": True, "argv": []})
json.dump({"run_nonce": a.run_nonce, "expected": len(res), "results": res}, open(a.evidence_json, "w"))
print(f"[stub liveness gate] {len(res)} rows")
'''


def make_fixture(work: Path, name: str, launcher_text: str = None):
    fix = work / name
    (fix / "scripts").mkdir(parents=True)
    (fix / "miner").mkdir()
    (fix / "logs" / "miner_workers").mkdir(parents=True)
    (fix / "scripts/launch_fleet_manual.sh").write_text(launcher_text or T["launcher"])
    (fix / "scripts/launch_fleet_manual.sh").chmod(0o755)
    shutil.copy(T["verify"], fix / "scripts/cache_records_verify.py")
    shutil.copy(T["join"], fix / "scripts/gate12_cache_join_gate.py")
    shutil.copy(T["footprint"], fix / "scripts/cache_footprint.py")
    (fix / "scripts/gate12_sentinel_gate.py").write_text(STUB_SENTINEL_GATE)
    (fix / "scripts/gate12_worker_liveness_gate.py").write_text(STUB_LIVENESS_GATE)
    (fix / "rewrite.py").write_text(REWRITE_PY)
    (fix / "unwrite.py").write_text(UNWRITE_PY)
    (fix / "standin.py").write_text(STANDIN_PY)
    bin_ = fix / "bin"; bin_.mkdir()
    for n, body in (("ssh", STUB_SSH), ("pkill", STUB_PKILL)):
        (bin_ / n).write_text(body); (bin_ / n).chmod(0o755)
    py = fix / "stub_python"; py.write_text(f'#!/bin/bash\nexec python3 "{fix}/standin.py" "$@"\n'); py.chmod(0o755)
    for ct, host in (("ct-a", "rigA"), ("ct-b", "rigB")):
        cb = fix / ct / "bin"; cb.mkdir(parents=True)
        (cb / "hostname").write_text(f'#!/bin/bash\n[ "${{FIXTURE_BAD_HOST:-}}" = "{ct}" ] && echo rigX || echo {host}\n')
        (cb / "hostname").chmod(0o755)
    (fix / "rig_profiles_config.json").write_text(json.dumps({
        "default_profile": "test", "profiles": ["test"],
        "nodes": [{"worker_hostname": "zeus", "config_hostname": "zeus", "endpoints": {"test": "localhost"}, "local": True},
                  {"worker_hostname": "rigA", "config_hostname": "rigA", "endpoints": {"test": "ct-a"}},
                  {"worker_hostname": "rigB", "config_hostname": "rigB", "endpoints": {"test": "ct-b"}}]}))
    (fix / "distributed_config.json").write_text(json.dumps({"nodes": [
        {"hostname": h, "gpu_count": n, "python_env": str(py), "script_path": str(fix)}
        for h, n in (("zeus", 1), ("rigA", 2), ("rigB", 2))]}))
    return fix


def env_for(fix, **extra):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("RIG_CUPY", "RUN_NONCE", "FIXTURE_", "GIT_FAIL"))}
    e.update(PATH=f"{fix / 'bin'}:{os.environ['PATH']}", FIX=str(fix), STAGGER="0", REMOTE_LAUNCH_SETTLE="0.5",
             RIG_CUPY_CACHE_ROOT="/FIXCACHE", RIG_PROFILE="test")
    e.update(extra)
    return e


def cleanup(fix):
    pf = fix / "worker_pids"
    if pf.exists():
        for p in pf.read_text().split():
            try:
                os.kill(int(p), signal.SIGTERM)
            except (ProcessLookupError, ValueError):
                pass


def environ_of(pid):
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return None
    return {l.split(b"=", 1)[0]: l.split(b"=", 1)[1] for l in raw if b"=" in l}


def launch(fix, **env):
    return run(["bash", str(fix / "scripts/launch_fleet_manual.sh"), "127.0.0.1", "5700", str(fix / "logs/miner_workers")],
               env=env_for(fix, **env), cwd=str(fix))


# ======================================================================================= U-STARTUP
def u_startup(work: Path):
    """The CT-side startup post-check fragment, executed against fixture environ files (NUL-exact)."""
    print("U-STARTUP (CT post-check, NUL-safe)")
    s = T["launcher"]
    i = s.index("CACHE_POST_BODY='\n") + len("CACHE_POST_BODY='\n")
    j = s.index("\n'\n", i)
    body = s[i:j]
    assert "ENVF=/proc/$worker_pid/environ" in body, "post-check body changed shape"
    d = work / "ust"; d.mkdir()
    P = "/FIXCACHE/cupy_cache_run_T_gpu3"
    cases = {"correct value": (b"A=1\0CUPY_CACHE_DIR=" + P.encode() + b"\0B=2\0", 0),
             "non-UTF-8 unrelated value + correct value": (b"Z=\xff\xfe\x80\0CUPY_CACHE_DIR=" + P.encode() + b"\0", 0),
             "missing variable": (b"A=1\0B=2\0", 15),
             "name only inside another value": (b"OTHER=x\nCUPY_CACHE_DIR=" + P.encode() + b"\0", 15),
             "value with an embedded newline": (b"CUPY_CACHE_DIR=" + P.encode() + b"\nextra\0", 15),
             "duplicate entry, same value": (b"CUPY_CACHE_DIR=" + P.encode() + b"\0CUPY_CACHE_DIR=" + P.encode() + b"\0", 15),
             "duplicate entry, different value": (b"CUPY_CACHE_DIR=" + P.encode() + b"\0CUPY_CACHE_DIR=/other\0", 15),
             "value with an extra suffix": (b"CUPY_CACHE_DIR=" + P.encode() + b"x\0", 15),
             "environment unreadable": (None, 15),
             "read error reported after a correct-looking count (grep rc 2)": (b"CUPY_CACHE_DIR=" + P.encode() + b"\0", 15)}
    for name, (content, want) in cases.items():
        ef = d / (re.sub(r"\W", "_", name) + ".env")
        if content is not None:
            ef.write_bytes(content)
        script = (f'WC={P}; TAG=T; NN=3; RCE=15; T_PREP=1.5\nsleep 30 & worker_pid=$!\n'
                  + body.replace("ENVF=/proc/$worker_pid/environ", f"ENVF={ef}") + '\nkill $worker_pid 2>/dev/null; exit 0\n')
        env = None
        if "grep rc 2" in name:                    # a grep that counts, then reports an I/O error
            gb = d / "grepfail"; gb.mkdir(exist_ok=True)
            (gb / "grep").write_text('#!/bin/bash\n/bin/grep "$@"\nexit 2\n'); (gb / "grep").chmod(0o755)
            env = dict(os.environ, PATH=f"{gb}:{os.environ['PATH']}")
        r = run(["bash", "-c", script], env=env)
        ok_line = [l for l in r.stdout.splitlines() if l.startswith("[cache] OK ")]
        if want == 0:
            check(f"startup check passes: {name}", r.returncode == 0 and len(ok_line) == 1 and ok_line[0].endswith(f"env_CUPY_CACHE_DIR={P}"),
                  r.stderr + r.stdout)
        else:
            check(f"startup check refuses (rc 15): {name}", r.returncode == 15 and not ok_line and "reason=env_mismatch" in r.stderr,
                  f"rc={r.returncode} {r.stderr}")


# ========================================================================================== U-JOIN
FAKE_SSH_JOIN = r'''#!/bin/bash
# join-gate transport over fixture files: /proc/<pid>/ -> $JOINFIX/<ep>/proc/<pid>/, /tmp/ -> $JOINFIX/<ep>/tmp/
tgt=""; while [ $# -gt 0 ]; do case "$1" in -o) shift 2;; -n) shift;; michael@*) tgt=${1#michael@}; shift; break;; *) shift;; esac; done
cmd2=$(python3 "$JOINFIX/rewrite.py" "$*" "$tgt" "$JOINFIX" 1) || exit 99
if [ -e "$JOINFIX/transport_fails_after_output_$tgt" ]; then bash -c "$cmd2"; exit 255; fi
if [ -e "$JOINFIX/read_fails_after_output_$tgt" ]; then PATH="$JOINFIX/catfail:$PATH" exec bash -c "$cmd2"; fi
exec bash -c "$cmd2"
'''


def u_join(work: Path):
    print("U-JOIN (parked re-read, NUL-safe, sentinel capture)")
    nonce = "gate12-T"
    J = str(T["join"])
    fix = work / "join"
    fix.mkdir()
    (fix / "rewrite.py").write_text(REWRITE_PY)
    ssh = fix / "fake_ssh"; ssh.write_text(FAKE_SSH_JOIN); ssh.chmod(0o755)
    cf = fix / "catfail"; cf.mkdir()
    (cf / "cat").write_text('#!/bin/bash\n/bin/cat "$@"\nexit 1\n'); (cf / "cat").chmod(0o755)   # reads, then reports an error
    slots = (("rigA", "ct-a", "0", "500"), ("rigB", "ct-b", "0", "501"))
    import datetime as _dt
    T_PREP_FIX = f"{_dt.datetime(2026, 11, 17, 1, 0, 0, tzinfo=_dt.timezone.utc).timestamp():.3f}"   # sentinel 5.25 s later

    def sentinel(pid, worker, n=nonce, ts="2026-11-17T01:00:05.250Z"):
        return f'{ts} WARNING __main__ [MINER-SESSION] SESSION_SENTINEL {json.dumps({"event": "SESSION_SENTINEL", "pid": int(pid), "run_nonce": n, "worker_id": worker})}\n'

    def build(sub, tweak=None):
        for p in fix.glob("ct-*"):
            shutil.rmtree(p)
        for f in list(fix.glob("transport_fails_after_output_*")) + list(fix.glob("read_fails_after_output_*")):
            f.unlink()
        d = fix / sub; d.mkdir()
        tsv = ["\t".join(("label", "tag", "host", "endpoint", "gpu", "path", "pid", "t_prep", "env_CUPY_CACHE_DIR", "dispatch_log"))]
        live = {"run_nonce": nonce, "expected": 3, "results": []}
        for h, ep, g, pid in slots:
            path = f"/tmp/cupy_cache_run_{nonce}_gpu{g}"
            tsv.append("\t".join((f"{h}:gpu{g}", nonce, h, ep, g, path, pid, T_PREP_FIX, path, "x.log")))
            argv = ["/py", "range_miner_worker.py", "--gpu-id", g, "--run-nonce", nonce, "--release-deadline", "900"]
            live["results"].append({"worker_id": f"{h}:gpu{g}", "endpoint": ep, "gpu": int(g), "pid": pid, "status": "OK",
                                    "local": False, "argv": argv, "log_path": f"/tmp/minerlogs/gpu{g}.log"})
            pd = fix / ep / "proc" / pid; pd.mkdir(parents=True)
            (pd / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
            (pd / "environ").write_bytes(b"PATH=/bin\0CUPY_CACHE_DIR=" + path.encode() + b"\0")
            ld = fix / ep / "tmp" / "minerlogs"; ld.mkdir(parents=True)
            (ld / f"gpu{g}.log").write_text("some line\n" + sentinel(pid, f"{h}:gpu{g}"))
        live["results"].append({"worker_id": "zeus:gpu0", "endpoint": "localhost", "gpu": 0, "pid": "77", "status": "OK", "local": True, "argv": []})
        if tweak:
            tweak(tsv, live, fix)
        (d / "c.tsv").write_text("\n".join(tsv) + "\n")
        (d / "l.json").write_text(json.dumps(live))
        r = run([sys.executable, "-B", J, "--run-nonce", nonce, "--cache-tsv", str(d / "c.tsv"), "--liveness-json", str(d / "l.json"),
                 "--expected-rig-slots", "2", "--evidence-json", str(d / "ev.json"), "--sentinel-tsv", str(d / "sent.tsv"),
                 "--ssh-cmd", str(ssh)], env=dict(os.environ, JOINFIX=str(fix)))
        ev = json.loads((d / "ev.json").read_text()) if (d / "ev.json").exists() else {}
        return r, ev, d

    r, ev, d = build("clean")
    sent = (d / "sent.tsv").read_text().splitlines() if (d / "sent.tsv").exists() else []
    check("clean join passes; Zeus recorded as local; 2 sentinel rows with start-up seconds",
          r.returncode == 0 and ev.get("verdict") == "PASS" and len(ev.get("local", [])) == 1 and len(sent) == 3
          and all(abs(float(x.split("\t")[6]) - 5.25) < 0.01 for x in sent[1:]), r.stderr + r.stdout + str(sent[:2]))

    def env_bytes(b):
        return lambda tsv, live, fx: (fx / "ct-b" / "proc" / "501" / "environ").write_bytes(b)

    def cmd_bytes(b):
        return lambda tsv, live, fx: (fx / "ct-a" / "proc" / "500" / "cmdline").write_bytes(b)

    def log_text(t):
        return lambda tsv, live, fx: (fx / "ct-a" / "tmp" / "minerlogs" / "gpu0.log").write_text(t)

    P1 = b"/tmp/cupy_cache_run_gate12-T_gpu0"
    good = {"non-UTF-8 unrelated value + correct value": env_bytes(b"Z=\xff\xfe\x80\0CUPY_CACHE_DIR=" + P1 + b"\0")}
    for name, tw in good.items():
        r, ev, _ = build(re.sub(r"\W", "_", name), tw)
        check(f"join passes: {name}", r.returncode == 0 and ev.get("verdict") == "PASS", r.stderr + r.stdout)
    negs = {
        "different parked pid": lambda tsv, live, fx: live["results"][0].update(pid="999"),
        "liveness run_nonce differs": lambda tsv, live, fx: live.update(run_nonce="gate12-OLD"),
        "liveness status not OK": lambda tsv, live, fx: live["results"][1].update(status="GONE"),
        "liveness argv nonce differs": lambda tsv, live, fx: live["results"][0]["argv"].__setitem__(5, "gate12-OLD"),
        "missing cache record": lambda tsv, live, fx: tsv.pop(2),
        "record tag differs": lambda tsv, live, fx: tsv.__setitem__(1, tsv[1].replace("\tgate12-T\t", "\tgate12-OLD\t", 1)),
        "parked environment: variable missing": env_bytes(b"PATH=/bin\0"),
        "parked environment: name only inside another value": env_bytes(b"OTHER=x\nCUPY_CACHE_DIR=" + P1 + b"\0"),
        "parked environment: value with an embedded newline": env_bytes(b"CUPY_CACHE_DIR=" + P1 + b"\nextra\0"),
        "parked environment: duplicate entry": env_bytes(b"CUPY_CACHE_DIR=" + P1 + b"\0CUPY_CACHE_DIR=/other\0"),
        "parked environment: value suffix": env_bytes(b"CUPY_CACHE_DIR=" + P1 + b"x\0"),
        "parked environment unreadable": lambda tsv, live, fx: (fx / "ct-a" / "proc" / "500" / "environ").unlink(),
        "parked cmdline: nonce as a longer token": cmd_bytes(b"/py\0--run-nonce\0gate12-T-extra\0"),
        "parked cmdline: nonce split across tokens": cmd_bytes(b"/py\0--run-nonce\0gate12\0-T\0"),
        "parked cmdline: --run-nonce twice": cmd_bytes(b"/py\0--run-nonce\0gate12-T\0--run-nonce\0gate12-T\0"),
        "parked cmdline unreadable": lambda tsv, live, fx: (fx / "ct-a" / "proc" / "500" / "cmdline").unlink(),
        "transport fails after printing correct output": lambda tsv, live, fx: (fx / "transport_fails_after_output_ct-b").write_text("x"),
        "parked read reports an error after printing correct content": lambda tsv, live, fx: (fx / "read_fails_after_output_ct-b").write_text("x"),
        "sentinel missing": log_text("no sentinel here\n"),
        "sentinel only for another nonce": log_text(sentinel("500", "rigA:gpu0", n="gate12-OLD")),
        "sentinel only for another pid": log_text(sentinel("999", "rigA:gpu0")),
        "sentinel only for another worker": log_text(sentinel("500", "rigA:gpu1")),
        "two matching sentinels (ambiguous)": log_text(sentinel("500", "rigA:gpu0") * 2),
        "sentinel with a non-ISO timestamp": log_text(sentinel("500", "rigA:gpu0", ts="yesterday")),
        "sentinel log unreadable": lambda tsv, live, fx: (fx / "ct-a" / "tmp" / "minerlogs" / "gpu0.log").unlink(),
    }
    for name, tw in negs.items():
        r, ev, d = build(re.sub(r"\W", "_", name), tw)
        check(f"join refuses: {name}", r.returncode == 1 and ev.get("verdict") == "REFUSE" and not (d / "sent.tsv").exists(),
              r.stderr + r.stdout)


# ======================================================================================== U-FOOT
def u_foot(work: Path):
    print("U-FOOT (post-run footprint from TSV paths)")
    fix = make_fixture(work, "foot")
    tsvp = fix / "c.tsv"
    rows = ["\t".join(("label", "tag", "host", "endpoint", "gpu", "path", "pid", "t_prep", "env_CUPY_CACHE_DIR", "dispatch_log"))]
    for ct, h in (("ct-a", "rigA"), ("ct-b", "rigB")):
        cdir = fix / ct / "cache" / "cupy_cache_run_T_gpu0"; cdir.mkdir(parents=True)
        (cdir / "k.hsaco").write_bytes(b"x" * 5000)
        (fix / ct / "cache" / "cupy_cache_run_OLD_gpu0").mkdir()
        rows.append("\t".join((f"{h}:gpu0", "T", h, ct, "0", "/FIXCACHE/cupy_cache_run_T_gpu0", "1", "1.0", "/FIXCACHE/cupy_cache_run_T_gpu0", "x")))
    tsvp.write_text("\n".join(rows) + "\n")
    out = fix / "foot.json"
    r = run([sys.executable, "-B", str(T["footprint"]), "--cache-tsv", str(tsvp), "--out", str(out), "--ssh-cmd", str(fix / "bin/ssh")], env=env_for(fix))
    res = json.loads(out.read_text()) if out.exists() else {}
    loc = fix / "ct-a" / "cache" / "cupy_cache_run_T_gpu0"
    app = int(run(["du", "-s", "--apparent-size", "--block-size=1", str(loc)]).stdout.split()[0])
    alo = int(run(["du", "-s", "--block-size=1", str(loc)]).stdout.split()[0])
    row = (res.get("rows") or [{}])[0]
    check("footprint COMPLETE; apparent and allocated bytes both reported and labelled",
          r.returncode == 0 and res.get("verdict") == "COMPLETE" and row.get("apparent_bytes") == app
          and row.get("allocated_bytes") == alo and "apparent" in res.get("units", {}).get("apparent_bytes", "")
          and "allocated" in res.get("units", {}).get("allocated_bytes", ""), r.stderr + str(row) + f" app={app} alo={alo}")
    check("footprint counts retained cupy_cache_run_* directories per CT (2 each)",
          res.get("per_ct", {}).get("ct-a", {}).get("retained_run_dirs") == 2, str(res.get("per_ct")))
    shutil.rmtree(fix / "ct-b" / "cache" / "cupy_cache_run_T_gpu0")
    out2 = fix / "foot2.json"
    r = run([sys.executable, "-B", str(T["footprint"]), "--cache-tsv", str(tsvp), "--out", str(out2), "--ssh-cmd", str(fix / "bin/ssh")], env=env_for(fix))
    res2 = json.loads(out2.read_text()) if out2.exists() else {}
    check("footprint refuses to call itself complete when a path cannot be measured",
          r.returncode == 1 and res2.get("verdict") == "INCOMPLETE", r.stderr)


# =========================================================================================== E2E
def e2e(work: Path):
    print("E2E (the launcher under test, stubbed transport, real stand-in processes)")
    nonce = "gate12-E2E-1"
    fix = make_fixture(work, "e2e_clean")
    try:
        r = launch(fix, RUN_NONCE=nonce)
        tsv = fix / f"logs/miner_workers/cache_records_{nonce}.tsv"
        rows = tsv.read_text().splitlines()[1:] if tsv.exists() else []
        check("clean per_run: launcher rc 0 and 4 verified rig rows", r.returncode == 0 and len(rows) == 4, r.stdout[-600:] + r.stderr[-600:])
        ok_env = bool(rows)
        for row in rows:
            c = row.split("\t")
            real = c[5].replace("/FIXCACHE/", f"{fix / c[3]}/cache/")
            ev = environ_of(c[6]) or {}
            ok_env &= ev.get(b"CUPY_CACHE_DIR") == real.encode() and Path(real).is_dir() and not any(Path(real).iterdir())
        check("clean per_run: every stand-in's real environment carries its fresh empty path", ok_env)
        zeus_pid = next((l.split("pid=")[1].strip() for l in r.stdout.splitlines() if "[launch]   pid=" in l), None)
        zev = environ_of(zeus_pid) if zeus_pid else None
        check("Zeus control: local stand-in carries /tmp/cupy_cache_local_gpu0, no cache record for it",
              zev and zev.get(b"CUPY_CACHE_DIR") == b"/tmp/cupy_cache_local_gpu0" and not any("zeus" in x for x in rows), str(zeus_pid))
        live = {"run_nonce": nonce, "expected": 5, "results": []}
        for row in rows:
            c = row.split("\t")
            argv = [a for a in Path(f"/proc/{c[6]}/cmdline").read_bytes().decode(errors="replace").split("\0") if a]
            live["results"].append({"worker_id": f"{c[2]}:gpu{c[4]}", "endpoint": c[3], "gpu": int(c[4]), "pid": c[6], "status": "OK",
                                    "local": False, "argv": argv, "log_path": f"/tmp/minerlogs/gpu{c[4]}.log"})
        live["results"].append({"worker_id": "zeus:gpu0", "endpoint": "localhost", "gpu": 0, "pid": zeus_pid, "status": "OK", "local": True, "argv": []})
        lj = fix / "liveness.json"; lj.write_text(json.dumps(live))
        rj = run([sys.executable, "-B", str(T["join"]), "--run-nonce", nonce, "--cache-tsv", str(tsv), "--liveness-json", str(lj),
                  "--expected-rig-slots", "4", "--evidence-json", str(fix / "join.json"), "--sentinel-tsv", str(fix / "sent.tsv"),
                  "--ssh-cmd", str(fix / "bin/ssh")], env=env_for(fix))
        sent = (fix / "sent.tsv").read_text().splitlines()[1:] if (fix / "sent.tsv").exists() else []
        check("clean per_run: join over the REAL parked stand-ins passes; 4 sentinel rows, start-up seconds >= 0",
              rj.returncode == 0 and len(sent) == 4 and all(float(x.split("\t")[6]) >= 0 for x in sent), rj.stderr + rj.stdout)
    finally:
        cleanup(fix)

    def fault(name, prep=None, **env):
        f = make_fixture(work, "e2e_" + name)
        try:
            if prep:
                prep(f)
            rr = launch(f, RUN_NONCE=nonce, **env)
            pub = (f / f"logs/miner_workers/cache_records_{nonce}.tsv").exists()
            logs = "".join(p.read_text() for p in (f / "logs/miner_workers").glob("dispatch_*.log"))
            return rr, pub, logs
        finally:
            cleanup(f)

    rr, pub, _ = fault("exit14", prep=lambda f: (f / "ct-b" / "cache" / f"cupy_cache_run_{nonce}_gpu1").mkdir(parents=True))
    check("exit 14 (pre-existing cache dir): launcher rc != 0, refusal visible, no TSV", rr.returncode != 0 and not pub and "DISPATCH FAILED" in rr.stderr, rr.stderr[-500:])
    for mode in ("wrong", "missing", "inside", "multi"):
        rr, pub, logs = fault("exit15_" + mode, FIXTURE_ENV_MODE=f"ct-a:1:{mode}")
        check(f"exit 15 (stand-in environment '{mode}'): launcher rc != 0, env_mismatch visible, no TSV",
              rr.returncode != 0 and not pub and "reason=env_mismatch" in logs, rr.stderr[-400:] + logs[-300:])
    rr, pub, _ = fault("nonutf", FIXTURE_ENV_MODE="ct-a:1:nonutf")
    check("control: a non-UTF-8 unrelated variable with the correct path passes (rc 0, TSV published)", rr.returncode == 0 and pub, rr.stderr[-500:])
    rr, pub, _ = fault("badrec", FIXTURE_BAD_HOST="ct-a")
    check("bad record (record host != dispatch host): launcher rc != 0, no TSV", rr.returncode != 0 and not pub and "PER-RUN CACHE NOT VERIFIED" in rr.stderr, rr.stderr[-500:])
    fs = make_fixture(work, "e2e_shared")
    try:
        rs = launch(fs, RUN_NONCE=nonce, RIG_CUPY_CACHE_POLICY="shared")
        logs = "".join(p.read_text() for p in (fs / "logs/miner_workers").glob("dispatch_*.log"))
        check("shared control: rc 0, no [cache] lines, no TSV, old per-GPU directory used",
              rs.returncode == 0 and "[cache]" not in logs and not list((fs / "logs/miner_workers").glob("cache_records_*"))
              and (fs / "ct-a" / "tmp" / "cupy_cache_gpu1").is_dir(), rs.stderr[-500:])
    finally:
        cleanup(fs)
    # the fixture root itself under /tmp-like naming must not be rewritten (Beta's SSH stub finding)
    tw = work / "tmp" / "deep"; tw.mkdir(parents=True)
    ft = make_fixture(tw, "e2e_tmpname")
    try:
        rt = launch(ft, RUN_NONCE=nonce)
        check("stub transport: a fixture path containing '/tmp/' is not rewritten (clean run still passes)", rt.returncode == 0, rt.stderr[-500:])
    finally:
        cleanup(ft)


def entry_slice(text: str, through_join: bool) -> str:
    i = text.index("# ---------- 1.9. [X-CACHE-POLICY v5]")
    if through_join:
        j = text.index('echo "POST-RUN (after the workers exit)')
        j = text.index("\n", j) + 1
        return text[i:j] + 'echo "REACHED_SAMPLER"\n'
    j = text.index('GATE-12 ABORTED: fleet dispatch was TRUNCATED')
    j = text.index("\nfi\n", j) + 4
    return text[i:j] + 'echo "REACHED_SENTINEL_GATE"\n'


GIT_STUB = ('git() { case "$1" in rev-parse) [ -n "${GIT_FAIL_HEAD:-}" ] && return 128; echo 0123456789abcdef0123456789abcdef01234567;;\n'
            '  status) [ -n "${GIT_FAIL_STATUS:-}" ] && return 128; printf "";; esac; }\n')


def e2e_entry(work: Path, entry_name: str, entry_text: str):
    print(f"E2E-ENTRY ({entry_name}: policy guard -> provenance -> fleet -> sentinel/liveness stubs -> join -> freeze)")
    nonce = "gate12-G-1"

    def go(name, through, prep=None, **env):
        f = make_fixture(work, re.sub(r"\W", "_", f"ent_{entry_name}_{name}"))   # fixture paths must be shell-safe
        try:
            if prep:
                prep(f)
            (f / "entry_point.sh").write_text(entry_text)
            body = f'STAMP=t_{name}; EVID=logs/ev_{name}.txt; RUN_NONCE={nonce}; : > "$EVID"\n' + GIT_STUB + entry_slice(entry_text, through)
            e = env_for(f, **env)
            if "RIG_CUPY_CACHE_POLICY" not in env:
                e.pop("RIG_CUPY_CACHE_POLICY", None)
            r = run(["bash", "-c", body, str(f / "entry_point.sh")], env=e, cwd=str(f))
            ran = bool(list((f / "logs/miner_workers").glob("dispatch_*.log")))
            freeze = f / f"logs/gate12_t_{name}_cache_freeze"
            man = run(["sha256sum", "-c", "SHA256SUMS"], cwd=str(freeze)) if (freeze / "SHA256SUMS").exists() else None
            return r, ran, freeze, man
        finally:
            cleanup(f)

    r, ran, freeze, man = go("clean", True)
    files = sorted(p.name for p in freeze.iterdir()) if freeze.exists() else []
    check(f"{entry_name}: clean -> join and freeze pass, sampler boundary reached; frozen manifest verifies; sentinel TSV frozen",
          r.returncode == 0 and "REACHED_SAMPLER" in r.stdout and man is not None and man.returncode == 0
          and any(x.endswith("_cache_sentinels.tsv") for x in files) and any(x.startswith("cache_records_") for x in files),
          r.stdout[-500:] + r.stderr[-300:] + str(files))
    for name, env, why in (("shared_exported", {"RIG_CUPY_CACHE_POLICY": "shared"}, "REFUSED"),
                           ("git_head_fails", {"GIT_FAIL_HEAD": "1"}, "git rev-parse HEAD failed"),
                           ("git_status_fails", {"GIT_FAIL_STATUS": "1"}, "git status failed")):
        r, ran, _, _ = go(name, False, **env)
        check(f"{entry_name}: {name} -> refused BEFORE dispatch",
              r.returncode == 1 and not ran and why in r.stdout and "REACHED_SENTINEL_GATE" not in r.stdout, r.stdout[-400:])
    r, ran, _, _ = go("prov_missing", False, prep=lambda f: (f / "scripts/gate12_cache_join_gate.py").unlink())
    check(f"{entry_name}: an unhashable provenance file -> refused before dispatch",
          r.returncode == 1 and not ran and "provenance not recorded" in r.stdout, r.stdout[-400:])
    def bad_hasher(f):                          # prints a well-formed hash, then reports failure
        h = f / "bin" / "sha256sum"
        h.write_text('#!/bin/bash\n/usr/bin/sha256sum "$@"\nexit 1\n'); h.chmod(0o755)
    r, ran, _, _ = go("hash_rc_fails", False, prep=bad_hasher)
    check(f"{entry_name}: sha256sum failing after printing a hash -> refused before dispatch",
          r.returncode == 1 and not ran and "provenance not recorded" in r.stdout, r.stdout[-400:])
    for name, prep, env in (("exit14", lambda f: (f / "ct-a" / "cache" / f"cupy_cache_run_{nonce}_gpu0").mkdir(parents=True), {}),
                            ("exit15", None, {"FIXTURE_ENV_MODE": "ct-b:0:multi"}),
                            ("badrec", None, {"FIXTURE_BAD_HOST": "ct-b"})):
        r, ran, _, _ = go(name, False, prep, **env)
        check(f"{entry_name}: {name} -> exit 1 before the sentinel gate (no coordinator, no release)",
              r.returncode == 1 and "REACHED_SENTINEL_GATE" not in r.stdout and "GATE-12 ABORTED" in r.stdout, r.stdout[-400:])
    r, ran, freeze, _ = go("join_fails", True, FIXTURE_LIVENESS_BADPID="1")
    check(f"{entry_name}: join failure -> exit 1 before the sampler, nothing frozen",
          r.returncode == 1 and "REACHED_SAMPLER" not in r.stdout and "CACHE JOIN GATE" in r.stdout and not freeze.exists(), r.stdout[-400:])
    r, ran, freeze, _ = go("freeze_fails", True, prep=lambda f: (f / "logs" / "gate12_t_freeze_fails_cache_freeze").write_text("blocker"))
    check(f"{entry_name}: freeze failure -> rc 97 reported, exit 1 before the sampler",
          r.returncode == 1 and "REACHED_SAMPLER" not in r.stdout and "rc=97" in r.stdout, r.stdout[-400:])


# ========================================================================================= MUTANTS
MUTANTS = [
    ("V1 substring compare", "verify", '        if rec[k] != v:\n', '        if v not in rec[k]:\n'),
    ("V2 ACK by search, not whole line", "verify", '    acks = [ACK_RE.fullmatch(l) for l in lines]\n', '    acks = [ACK_RE.search(l) for l in lines]\n'),
    ("V3 no post-rename invalidation", "verify", '        if renamed:\n', '        if False:\n'),
    ("V4 expected-slot count ignored", "verify", '    if len(exp) != a.expected_slots:\n', '    if False:\n'),
    ("S1 startup exact match without -x", "launcher", 'NEX=$(LC_ALL=C grep -zxcF -e', 'NEX=$(LC_ALL=C grep -zcF -e'),
    ("S2 startup named-entry count ignored", "launcher", '[ "$NANY" != 1 ] || [ "$NEX" != 1 ]', '[ "$NEX" != 1 ]'),
    ("S3 startup read status ignored", "launcher", 'if [ "$RA" -gt 1 ] || [ "$RX" -gt 1 ] || ', 'if '),
    ("J1 parked environ split on newline", "join", '    ents = [e for e in environ.split(b"\\0") if e]\n', '    ents = [e for e in environ.replace(b"\\0", b"\\n").split(b"\\n") if e]\n'),
    ("J2 parked nonce prefix match", "join", '    if toks[idx[0] + 1] != nonce.encode():\n', '    if not toks[idx[0] + 1].startswith(nonce.encode()):\n'),
    ("J3 parked read status ignored", "join", '        if failed:\n', '        if False:\n'),
    ("J4 transport rc ignored", "join", '        if rc != 0:\n            bad(f"transport rc={rc}"', '        if False:\n            bad(f"transport rc={rc}"'),
    ("J5 sentinel pid ignored", "join", ' and str(rec.get("pid")) == pid and ', ' and '),
    ("J6 ambiguous sentinel accepted", "join", '    if len(hits) != 1:\n', '    if not hits:\n'),
    ("F1 footprint apparent/allocated swapped", "footprint", 'zip(("apparent_bytes", "allocated_bytes",', 'zip(("allocated_bytes", "apparent_bytes",'),
    ("E1 entry: git HEAD status unchecked", "entry", 'if ! GIT_HEAD=$(git rev-parse --verify HEAD 2>/dev/null) || ! [[ "$GIT_HEAD" =~ ^[0-9a-f]{40}$ ]]; then',
     'GIT_HEAD=$(git rev-parse --verify HEAD 2>/dev/null); if false; then'),
    ("E2 entry: git status unchecked", "entry", 'if ! GIT_STATUS_OUT=$(git status --porcelain 2>/dev/null); then',
     'GIT_STATUS_OUT=$(git status --porcelain 2>/dev/null); if false; then'),
    ("E3 entry: hash status unchecked", "entry", '  h=$(sha256sum -- "$1" 2>/dev/null) || return 1', '  h=$(sha256sum -- "$1" 2>/dev/null) || true'),
    ("E4 entry: inherited shared let through", "entry", 'if [ -n "${RIG_CUPY_CACHE_POLICY+set}" ] && [ "$RIG_CUPY_CACHE_POLICY" != per_run ]; then',
     'if false; then'),
    ("E5 entry: sentinel TSV not frozen", "entry", '"$CACHE_JOIN_EVID" "$CACHE_SENT_TSV" "$PROV"', '"$CACHE_JOIN_EVID" "$PROV"'),
]


def run_mutants(work: Path):
    global RESULTS
    import contextlib
    import io
    saved = dict(T)
    out = []
    for i, (name, kind, old, new) in enumerate(MUTANTS):
        T.clear(); T.update(saved)
        src = {"verify": saved["verify"].read_text(), "join": saved["join"].read_text(), "footprint": saved["footprint"].read_text(),
               "launcher": saved["launcher"], "entry": saved["gate12"]}[kind]
        if src.count(old) != 1:
            out.append((name, f"INVALID (pattern count {src.count(old)})")); continue
        mut = src.replace(old, new)
        sub = work / f"mut{i}"; sub.mkdir()
        if kind in ("verify", "join", "footprint"):
            p = sub / saved[kind].name; p.write_text(mut); T[kind] = p
        elif kind == "launcher":
            T["launcher"] = mut
        else:
            T["gate12"] = mut
        RESULTS = []
        with contextlib.redirect_stdout(io.StringIO()):
            {"verify": lambda: (u_verify(sub)), "join": lambda: (u_join(sub)), "footprint": lambda: (u_foot(sub)),
             "launcher": lambda: (u_startup(sub), e2e(sub)), "entry": lambda: (e2e_entry(sub, "gate12_launch.sh", T["gate12"]))}[kind]()
        failed = [r[0] for r in RESULTS if not r[1]]
        out.append((name, f"DETECTED ({failed[0][:80]})" if failed else "SURVIVED"))
        for f in sub.rglob("worker_pids"):
            cleanup(f.parent)
    T.clear(); T.update(saved)
    for n, v in out:
        print(f"{n}: {v}")
    print(f"{sum(v.startswith('DETECTED') for _, v in out)} of {len(out)} mutants detected")
    return all(v.startswith("DETECTED") for _, v in out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="tests", choices=("tests", "mutants"))
    ap.add_argument("--landed", type=Path, default=None)
    ap.add_argument("--route-a", type=Path, default=None)
    a = ap.parse_args()
    (REPO / "logs").mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(dir=str(REPO / "logs"), prefix="cachev5_"))
    try:
        build_tree(work, a.landed, a.route_a)
        if a.mode == "mutants":
            ok = run_mutants(work)
            return 0 if ok else 1
        u_verify(work)
        u_startup(work)
        u_join(work)
        u_foot(work)
        e2e(work)
        e2e_entry(work, "gate12_launch.sh", T["gate12"])
        e2e_entry(work, "run4_launch.sh (Route A fixture)", T["run4"])
    finally:
        for f in work.rglob("worker_pids"):
            cleanup(f.parent)
        subprocess.run(["chmod", "-R", "u+w", str(work)])
        shutil.rmtree(work, ignore_errors=True)
    bad = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(bad)} of {len(RESULTS)} checks pass")
    print("ALL GREEN" if not bad else "NOT GREEN")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
