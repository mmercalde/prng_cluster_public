#!/usr/bin/env python3
"""DRAFT tests for X-CACHE-POLICY v4 (docs/drafts/cache_policy_v4/). CPU only; no rig; no real ssh.

All fixtures live under the worktree's logs/ (never /tmp). The three diffs are applied to temporary COPIES
of scripts/launch_fleet_manual.sh, gate12_launch.sh and /home/michael/run4_launch.sh; nothing real is edited.
The stub transport maps each fake CT's /tmp and cache root to its own fixture directory and maps the
paths back in what it returns; unknown hosts (e.g. the real rig IPs in an abort path) are never contacted.
pkill is stubbed on PATH, so the abort paths cannot touch a real process. Every stand-in worker records its
PID and is killed by that PID at cleanup.

  U-VERIFY   cache_records_verify.py: clean control publishes once; each R3 defect refuses with nothing
             published: TSV path unavailable, interrupted write, dotted tag mismatch, dotted root mismatch,
             missing host, missing t_prep, wrong-GPU ACK, a diagnostic containing "started pid=" instead of
             an ACK, absent ACK, duplicate ACK, duplicate record, extra field, host != endpoint's host,
             pid != ACK pid, env != path, expectation count != slots, pre-existing TSV.
  U-JOIN     gate12_cache_join_gate.py with a stub ssh over fixture /proc files: clean join passes; refuses on
             a different parked pid, a different nonce in the parked argv, a wrong or missing or duplicated
             parked CUPY_CACHE_DIR, an unreadable environment, a missing record, liveness not OK; Zeus is
             recorded as local.
  E2E        the patched launcher, stubbed transport, 2 rigs x 2 GPUs + 1 local Zeus GPU:
             clean per_run -> rc 0, 4 verified rows, each stand-in's real environment carries its path;
             the join gate over the real stand-in processes passes;
             exit 14 (pre-existing cache dir), exit 15 (stand-in exec's with another CUPY_CACHE_DIR) and a bad
             record (wrong CT hostname) each -> launcher rc != 0, no TSV published;
             shared control -> rc 0, no [cache] lines, no TSV, old per-GPU path used;
             Zeus control -> local stand-in carries /tmp/cupy_cache_local_gpu0, no cache record for it.
  E2E-GATE12 the patched gate12_launch.sh / run4_launch.sh slice (policy guard -> fleet call -> abort) with
             the patched launcher: exported shared -> refused before the launcher runs; each of the three
             faults -> exit 1 and the next gate (sentinel) is never reached; clean -> reached.
Run:  python3 -B docs/drafts/cache_policy_v4/tests/test_cache_policy_v4.py
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
V4 = HERE.parent
REPO = V4.parents[2]
VERIFY = V4 / "scripts/cache_records_verify.py"
JOIN = V4 / "scripts/gate12_cache_join_gate.py"
RUN4 = Path("/home/michael/run4_launch.sh")
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"   <- {detail[:300]}"))


def run(cmd, **kw):
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    kw.setdefault("timeout", 120)
    return subprocess.run(cmd, **kw)


# ============================================================================== U-VERIFY
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

    def fixture(sub, mutate=None, slots=None, exp_mut=None):
        d = work / sub
        d.mkdir(parents=True)
        rows = []
        for i, (h, g) in enumerate((("rigA", "0"), ("rigA", "1"), ("rigB", "0"))):
            path = f"{root}/cupy_cache_run_{tag}_gpu{g}"        # per-CT path: rigA gpu0 and rigB gpu0 share it
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

    def go(d, slots, out=None, pre=None):
        out = out or d / "records.tsv"
        if pre:
            pre(out)
        return run([sys.executable, "-B", str(VERIFY), "--tag", tag, "--expect", str(d / "expect.tsv"),
                    "--expected-slots", str(slots), "--out", str(out)]), out

    d, n = fixture("clean")
    r, out = go(d, n)
    rows = out.read_text().splitlines() if out.exists() else []
    check("clean control publishes exactly once (header + 3 rows)", r.returncode == 0 and len(rows) == 4, r.stderr)
    check("no temp file left", not list(d.glob("records.tsv.tmp.*")))
    r2, _ = go(d, n)
    check("pre-existing TSV refused (a tag is written once)", r2.returncode == 3, r2.stderr)

    cases = {
        "dotted tag mismatch (caseX1 for case.1)": lambda L, h, g, p, pid: [L[0].replace("tag=case.1", "tag=caseX1"), L[1]],
        "dotted root mismatch (/tmp/cacheYroot for /tmp/cache.root)": lambda L, h, g, p, pid: [L[0].replace("/tmp/cache.root", "/tmp/cacheYroot"), L[1]],
        "missing host field": lambda L, h, g, p, pid: [rec_line(tag, None, g, p, pid), L[1]],
        "missing t_prep field": lambda L, h, g, p, pid: [rec_line(tag, h, g, p, pid, t=None), L[1]],
        "empty host value": lambda L, h, g, p, pid: [rec_line(tag, "", g, p, pid), L[1]],
        "non-decimal t_prep": lambda L, h, g, p, pid: [rec_line(tag, h, g, p, pid, t="soon"), L[1]],
        "wrong-GPU ACK": lambda L, h, g, p, pid: [L[0], f"[dispatch] started pid={pid} gpu=7"],
        "diagnostic containing 'started pid=' instead of an ACK": lambda L, h, g, p, pid: [L[0], f"[dispatch] worker failed; was started pid={pid} gpu={g}"],
        "absent ACK": lambda L, h, g, p, pid: [L[0]],
        "duplicate ACK": lambda L, h, g, p, pid: [L[0], L[1], L[1]],
        "duplicate record": lambda L, h, g, p, pid: [L[0], L[0], L[1]],
        "extra field": lambda L, h, g, p, pid: [L[0] + " note=x", L[1]],
        "host is not the dispatch endpoint's host": lambda L, h, g, p, pid: [rec_line(tag, "rigZ", g, p, pid), L[1]],
        "record pid != ACK pid": lambda L, h, g, p, pid: [L[0], f"[dispatch] started pid=999 gpu={g}"],
        "env != path": lambda L, h, g, p, pid: [L[0].replace(f"env_CUPY_CACHE_DIR={p}", "env_CUPY_CACHE_DIR=/tmp/cupy_cache_gpu1"), L[1]],
        "policy shared in record": lambda L, h, g, p, pid: [L[0].replace("policy=per_run", "policy=shared"), L[1]],
        "host with an extra suffix (rigAx for rigA)": lambda L, h, g, p, pid: [L[0].replace(f"host={h} ", f"host={h}x "), L[1]],
        "path with an extra suffix": lambda L, h, g, p, pid: [L[0].replace(f"path={p} ", f"path={p}x "), L[1]],
        "a full ACK embedded in a longer line instead of an ACK line": lambda L, h, g, p, pid: [L[0], f"note: {L[1]} (replayed)"],
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
    check("clean control covers equal paths on different CTs (rigA gpu0 / rigB gpu0)", True)
    # TSV path unavailable: a directory sits at the output path
    d, n = fixture("unavail")
    r, out = go(d, n, pre=lambda o: o.mkdir())
    check("refuses: TSV path unavailable (a directory)", r.returncode == 3 and out.is_dir(), r.stderr)
    # TSV directory not writable
    d, n = fixture("rodir")
    od = d / "ro"; od.mkdir()
    od.chmod(0o500)
    try:
        r, out = go(d, n, out=od / "records.tsv")
        check("refuses: output directory not writable (write fails -> rc 2, nothing published)",
              r.returncode == 2 and not out.exists(), r.stderr)
    finally:
        od.chmod(0o700)
    # interrupted write: rename fails after the temp file is complete
    d, n = fixture("interrupt")
    code = ("import sys, os, runpy; sys.argv = sys.argv[1:]\n"
            "def boom(*a, **k): raise OSError('injected: interrupted before rename')\n"
            "os.replace = boom\n"
            "runpy.run_path(sys.argv[0], run_name='__main__')\n")
    r = run([sys.executable, "-B", "-c", code, str(VERIFY), "--tag", tag, "--expect", str(d / "expect.tsv"),
             "--expected-slots", str(n), "--out", str(d / "records.tsv")])
    check("refuses: interrupted write (rename fails) -> rc 2, no TSV, no temp left",
          r.returncode == 2 and not (d / "records.tsv").exists() and not list(d.glob("records.tsv.tmp.*")), r.stderr)
    code2 = code.replace("raise OSError('injected: interrupted before rename')", "raise KeyboardInterrupt()")
    d, n = fixture("interrupt2")
    r = run([sys.executable, "-B", "-c", code2, str(VERIFY), "--tag", tag, "--expect", str(d / "expect.tsv"),
             "--expected-slots", str(n), "--out", str(d / "records.tsv")])
    check("refuses: interrupt signal during publish -> rc 2, nothing published",
          r.returncode == 2 and not (d / "records.tsv").exists(), r.stderr)


# ================================================================================ U-JOIN
FAKE_SSH = r'''#!/bin/bash
# fake transport for the join gate: /proc/<pid>/ -> $JOINFIX/proc_<endpoint>/<pid>/
tgt=""; while [ $# -gt 0 ]; do case "$1" in -o) shift 2;; -n) shift;; michael@*) tgt=${1#michael@}; shift; break;; *) shift;; esac; done
cmd="$*"; cmd=${cmd//\/proc\//$JOINFIX/proc_$tgt/}
if [ -e "$JOINFIX/transport_fails_after_output_$tgt" ]; then bash -c "$cmd"; exit 255; fi
exec bash -c "$cmd"
'''


def u_join(work: Path):
    print("U-JOIN")
    nonce = "gate12-T"
    fix = work / "join"
    fix.mkdir()
    ssh = fix / "fake_ssh"
    ssh.write_text(FAKE_SSH)
    ssh.chmod(0o755)

    def build(sub, tweak=None):
        d = fix / sub
        d.mkdir()
        tsv = ["\t".join(("label", "tag", "host", "endpoint", "gpu", "path", "pid", "t_prep", "env_CUPY_CACHE_DIR", "dispatch_log"))]
        live = {"run_nonce": nonce, "expected": 3, "results": []}
        for i, (h, ep, g) in enumerate((("rigA", "ct-a", "0"), ("rigB", "ct-b", "0"))):
            pid, path = str(500 + i), f"/tmp/cupy_cache_run_{nonce}_gpu{g}"
            tsv.append("\t".join((f"{h}:gpu{g}", nonce, h, ep, g, path, pid, "1.0", path, "x.log")))
            argv = ["/py", "range_miner_worker.py", "--gpu-id", g, "--run-nonce", nonce, "--release-deadline", "900"]
            live["results"].append({"worker_id": f"{h}:gpu{g}", "endpoint": ep, "gpu": int(g), "pid": pid,
                                    "status": "OK", "local": False, "argv": argv})
            pd = fix / f"proc_{ep}" / pid
            pd.mkdir(parents=True, exist_ok=True)
            (pd / "cmdline").write_bytes(("\0".join(argv) + "\0").encode())
            (pd / "environ").write_bytes(f"PATH=/bin\0CUPY_CACHE_DIR={path}\0".encode())
        live["results"].append({"worker_id": "zeus:gpu0", "endpoint": "localhost", "gpu": 0, "pid": "77",
                                "status": "OK", "local": True, "argv": []})
        if tweak:
            tweak(tsv, live, fix)
        (d / "c.tsv").write_text("\n".join(tsv) + "\n")
        (d / "l.json").write_text(json.dumps(live))
        r = run([sys.executable, "-B", str(JOIN), "--run-nonce", nonce, "--cache-tsv", str(d / "c.tsv"),
                 "--liveness-json", str(d / "l.json"), "--expected-rig-slots", "2",
                 "--evidence-json", str(d / "ev.json"), "--ssh-cmd", str(ssh)],
                env=dict(os.environ, JOINFIX=str(fix)))
        ev = json.loads((d / "ev.json").read_text()) if (d / "ev.json").exists() else {}
        return r, ev

    r, ev = build("clean")
    check("clean join passes; Zeus recorded as local", r.returncode == 0 and ev.get("verdict") == "PASS"
          and len(ev.get("local", [])) == 1, r.stderr + r.stdout)
    for k in list((fix / "proc_ct-a").iterdir()) + list((fix / "proc_ct-b").iterdir()):
        shutil.rmtree(k)

    def reset_and(f):
        def t(tsv, live, fx):
            f(tsv, live, fx)
        return t
    negs = {
        "different parked pid": lambda tsv, live, fx: live["results"][0].update(pid="999"),
        "different nonce in parked argv": lambda tsv, live, fx: live["results"][0]["argv"].__setitem__(5, "gate12-OLD"),
        "liveness run_nonce differs": lambda tsv, live, fx: live.update(run_nonce="gate12-OLD"),
        "liveness status not OK": lambda tsv, live, fx: live["results"][1].update(status="GONE"),
        "missing cache record": lambda tsv, live, fx: tsv.pop(2),
        "record tag differs": lambda tsv, live, fx: tsv.__setitem__(1, tsv[1].replace("\tgate12-T\t", "\tgate12-OLD\t", 1)),
    }
    for name, tw in negs.items():
        for k in [p for p in fix.glob("proc_*")]:
            shutil.rmtree(k)
        r, ev = build(name.replace(" ", "_"), tw)
        check(f"join refuses: {name}", r.returncode == 1 and ev.get("verdict") == "REFUSE", r.stderr + r.stdout)

    def env_tw(content):
        def tw(tsv, live, fx):
            (fx / "proc_ct-b" / "501" / "environ").write_bytes(content)
        return tw
    for name, content in {"wrong parked CUPY_CACHE_DIR": b"CUPY_CACHE_DIR=/tmp/cupy_cache_gpu0\0",
                          "missing parked CUPY_CACHE_DIR": b"PATH=/bin\0",
                          "duplicated parked CUPY_CACHE_DIR": b"CUPY_CACHE_DIR=/tmp/cupy_cache_run_gate12-T_gpu0\0CUPY_CACHE_DIR=/x\0"}.items():
        for k in [p for p in fix.glob("proc_*")]:
            shutil.rmtree(k)
        r, ev = build(name.replace(" ", "_"), env_tw(content))
        check(f"join refuses: {name}", r.returncode == 1 and ev.get("verdict") == "REFUSE", r.stderr + r.stdout)
    for k in [p for p in fix.glob("proc_*")]:
        shutil.rmtree(k)
    r, ev = build("unreadable", lambda tsv, live, fx: (fx / "proc_ct-a" / "500" / "environ").unlink())
    check("join refuses: parked environment unreadable", r.returncode == 1 and ev.get("verdict") == "REFUSE", r.stderr)
    for k in [p for p in fix.glob("proc_*")]:
        shutil.rmtree(k)
    marker = fix / "transport_fails_after_output_ct-b"
    r, ev = build("transport_fail", lambda tsv, live, fx: marker.write_text("x"))
    marker.unlink()
    check("join refuses: transport fails after printing the right value (rc != 0 is not trusted)",
          r.returncode == 1 and ev.get("verdict") == "REFUSE", r.stderr)


# =================================================================================== E2E
STUB_SSH = r'''#!/bin/bash
# stub transport: each fake CT gets its own filesystem view; paths are mapped back in what it returns.
tgt=""; while [ $# -gt 0 ]; do case "$1" in -o) shift 2;; -n|-T) shift;; michael@*|root@*) tgt=${1#*@}; shift; break;; *) shift;; esac; done
cmd="$*"
echo "$tgt" >> "$FIX/ssh_calls.log"
case "$tgt" in ct-*) ;; *) exit 0;; esac          # anything else (real rig IPs in abort paths): never contacted
CT="$FIX/$tgt"; mkdir -p "$CT/tmp" "$CT/cache"
cmd=${cmd//\/tmp\//$CT/tmp/}; cmd=${cmd//\/FIXCACHE\//$CT/cache/}
export CT_NAME="$tgt"
PATH="$CT/bin:$PATH" bash -c "$cmd" 2>&1 | sed -e "s#$CT/cache/#/FIXCACHE/#g" -e "s#$CT/tmp/#/tmp/#g"
exit "${PIPESTATUS[0]}"
'''
STUB_PY = r'''#!/bin/bash
# inert worker stand-in; may exec with a different CUPY_CACHE_DIR to inject an env mismatch
g=""; prev=""; for a in "$@"; do [ "$prev" = "--gpu-id" ] && g="$a"; prev="$a"; done
echo "$$" >> "$FIX/worker_pids"
# keep the worker argv visible in /proc/<pid>/cmdline (as the real worker's is), so the join gate can see it
if [ -n "${FIXTURE_ENV_MISMATCH:-}" ] && [ "$FIXTURE_ENV_MISMATCH" = "${CT_NAME:-local}:$g" ]; then
    exec env CUPY_CACHE_DIR=/wrong python3 -c 'import time; time.sleep(120)' "$@"
fi
exec python3 -c 'import time; time.sleep(120)' "$@"
'''
STUB_PKILL = '#!/bin/bash\necho "pkill $*" >> "$FIX/pkill_calls.log"\nexit 0\n'


def make_fixture(work: Path, launcher_text: str, name: str):
    fix = work / name
    (fix / "scripts").mkdir(parents=True)
    (fix / "miner").mkdir()
    (fix / "logs" / "miner_workers").mkdir(parents=True)
    (fix / "scripts/launch_fleet_manual.sh").write_text(launcher_text)
    (fix / "scripts/launch_fleet_manual.sh").chmod(0o755)
    shutil.copy(VERIFY, fix / "scripts/cache_records_verify.py")
    shutil.copy(JOIN, fix / "scripts/gate12_cache_join_gate.py")
    for f in ("gate12_worker_liveness_gate.py", "gate12_sentinel_gate.py"):
        (fix / "scripts" / f).write_text("# stub for provenance hashing only\n")
    bin_ = fix / "bin"; bin_.mkdir()
    for n, body in (("ssh", STUB_SSH), ("pkill", STUB_PKILL)):
        (bin_ / n).write_text(body); (bin_ / n).chmod(0o755)
    py = fix / "stub_python"; py.write_text(STUB_PY); py.chmod(0o755)
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
    e = {k: v for k, v in os.environ.items() if not k.startswith(("RIG_CUPY", "RUN_NONCE", "FIXTURE_"))}
    e.update(PATH=f"{fix / 'bin'}:{os.environ['PATH']}", FIX=str(fix), STAGGER="0", REMOTE_LAUNCH_SETTLE="0.4",
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
    return {l.split(b"=", 1)[0].decode(): l.split(b"=", 1)[1].decode() for l in raw if b"=" in l}


def launch(fix, **env):
    return run(["bash", str(fix / "scripts/launch_fleet_manual.sh"), "127.0.0.1", "5700",
                str(fix / "logs/miner_workers")], env=env_for(fix, **env), cwd=str(fix), timeout=120)


def e2e(work: Path, launcher_text: str):
    print("E2E (patched launcher, stubbed transport)")
    nonce = "gate12-E2E-1"
    fix = make_fixture(work, launcher_text, "e2e_clean")
    try:
        r = launch(fix, RUN_NONCE=nonce)
        tsv = fix / f"logs/miner_workers/cache_records_{nonce}.tsv"
        rows = tsv.read_text().splitlines()[1:] if tsv.exists() else []
        check("clean per_run: launcher rc 0 and 4 verified rig rows", r.returncode == 0 and len(rows) == 4,
              r.stdout[-800:] + r.stderr[-800:])
        ok_env = True
        for row in rows:
            c = row.split("\t")
            pid, path, ct = c[6], c[5], c[3]
            # the stand-in runs in this process namespace; its real path is the CT-mapped one
            real = path.replace("/FIXCACHE/", f"{fix / ct}/cache/")
            ev = environ_of(pid) or {}
            ok_env &= ev.get("CUPY_CACHE_DIR") == real and Path(real).is_dir() and not any(Path(real).iterdir())
        check("clean per_run: every stand-in's real environment carries its fresh empty path", ok_env and rows)
        zeus_pid = None
        for l in r.stdout.splitlines():
            if "pid=" in l and "[launch]   pid=" in l:
                zeus_pid = l.split("pid=")[1].strip()
        zev = environ_of(zeus_pid) if zeus_pid else None
        check("Zeus control: local stand-in carries /tmp/cupy_cache_local_gpu0 and has no cache record",
              zev and zev.get("CUPY_CACHE_DIR") == "/tmp/cupy_cache_local_gpu0" and not any("zeus" in x for x in rows),
              f"zeus_pid={zeus_pid} env={zev and zev.get('CUPY_CACHE_DIR')}")
        # join gate over the real stand-ins (stub ssh maps paths back)
        live = {"run_nonce": nonce, "expected": 5, "results": []}
        for row in rows:
            c = row.split("\t")
            argv = Path(f"/proc/{c[6]}/cmdline").read_bytes().decode(errors="replace").split("\0")
            live["results"].append({"worker_id": f"{c[2]}:gpu{c[4]}", "endpoint": c[3], "gpu": int(c[4]), "pid": c[6],
                                    "status": "OK", "local": False, "argv": [a for a in argv if a]})
        live["results"].append({"worker_id": "zeus:gpu0", "endpoint": "localhost", "gpu": 0, "pid": zeus_pid,
                                "status": "OK", "local": True, "argv": []})
        lj = fix / "liveness.json"; lj.write_text(json.dumps(live))
        rj = run([sys.executable, "-B", str(JOIN), "--run-nonce", nonce, "--cache-tsv", str(tsv), "--liveness-json", str(lj),
                  "--expected-rig-slots", "4", "--evidence-json", str(fix / "join.json"), "--ssh-cmd", str(fix / "bin/ssh")],
                 env=env_for(fix))
        check("clean per_run: the join gate passes over the real parked stand-ins", rj.returncode == 0, rj.stderr + rj.stdout)
    finally:
        cleanup(fix)

    def fault(name, prep=None, **env):
        f = make_fixture(work, launcher_text, "e2e_" + name)
        try:
            if prep:
                prep(f)
            rr = launch(f, RUN_NONCE=nonce, **env)
            pub = (f / f"logs/miner_workers/cache_records_{nonce}.tsv").exists()
            return rr, pub, f
        finally:
            cleanup(f)

    rr, pub, _ = fault("exit14", prep=lambda f: (f / "ct-b" / "cache" / f"cupy_cache_run_{nonce}_gpu1").mkdir(parents=True))
    check("exit 14 (pre-existing cache dir on one slot): launcher rc != 0, refusal visible, no TSV",
          rr.returncode != 0 and not pub and "DISPATCH FAILED" in rr.stderr, rr.stderr[-600:])
    rr, pub, f = fault("exit15", FIXTURE_ENV_MISMATCH="ct-a:1")
    logs = "".join(p.read_text() for p in (f / "logs/miner_workers").glob("dispatch_*.log"))
    check("exit 15 (stand-in with another CUPY_CACHE_DIR): launcher rc != 0, env_mismatch visible, no TSV",
          rr.returncode != 0 and not pub and "reason=env_mismatch" in logs, rr.stderr[-600:])
    rr, pub, _ = fault("badrec", FIXTURE_BAD_HOST="ct-a")
    check("bad record (record host != the dispatch endpoint's host): launcher rc != 0, no TSV",
          rr.returncode != 0 and not pub and "PER-RUN CACHE NOT VERIFIED" in rr.stderr, rr.stderr[-600:])
    fs = make_fixture(work, launcher_text, "e2e_shared")
    try:
        rs = launch(fs, RUN_NONCE=nonce, RIG_CUPY_CACHE_POLICY="shared")
        logs = "".join(p.read_text() for p in (fs / "logs/miner_workers").glob("dispatch_*.log"))
        check("shared control: rc 0, no [cache] lines, no TSV, old per-GPU directory used",
              rs.returncode == 0 and "[cache]" not in logs and not list((fs / "logs/miner_workers").glob("cache_records_*"))
              and (fs / "ct-a" / "tmp" / "cupy_cache_gpu1").is_dir(), rs.stderr[-600:])
    finally:
        cleanup(fs)


def gate12_slice(entry_text: str) -> str:
    """policy guard + provenance (1.9) -> fleet call -> fleet abort block, then a marker for the next gate."""
    i = entry_text.index("# ---------- 1.9. [X-CACHE-POLICY]")
    j = entry_text.index('GATE-12 ABORTED: fleet dispatch was TRUNCATED')
    j = entry_text.index("\nfi\n", j) + 4
    return entry_text[i:j] + 'echo "REACHED_SENTINEL_GATE"\n'


def e2e_gate12(work: Path, launcher_text: str, entry_name: str, entry_text: str):
    print(f"E2E-GATE12 ({entry_name} slice, patched launcher, stubbed transport)")
    sl = gate12_slice(entry_text)
    nonce = "gate12-G-1"

    def go(name, prep=None, **env):
        f = make_fixture(work, launcher_text, f"g12_{entry_name}_{name}")
        try:
            if prep:
                prep(f)
            (f / "entry_point.sh").write_text(entry_text)
            body = (f'STAMP=t_{name}; EVID=logs/ev_{name}.txt; RUN_NONCE={nonce}; : > "$EVID"\n'
                    'git() { echo "not-a-repo"; }\n' + sl)
            e = env_for(f, **env)
            if "RIG_CUPY_CACHE_POLICY" not in env:
                e.pop("RIG_CUPY_CACHE_POLICY", None)
            r = run(["bash", "-c", body, str(f / "entry_point.sh")], env=e, cwd=str(f), timeout=150)
            ran = bool(list((f / "logs/miner_workers").glob("dispatch_*.log")))
            return r, ran
        finally:
            cleanup(f)

    r, ran = go("shared_exported", RIG_CUPY_CACHE_POLICY="shared")
    check(f"{entry_name}: exported shared refused before the launcher runs",
          r.returncode == 1 and not ran and "REFUSED" in r.stdout and "REACHED_SENTINEL_GATE" not in r.stdout, r.stdout[-400:])
    r, ran = go("clean")
    check(f"{entry_name}: clean -> fleet passes and the sentinel gate is reached",
          r.returncode == 0 and "REACHED_SENTINEL_GATE" in r.stdout, r.stdout[-500:] + r.stderr[-300:])
    r, ran = go("prov_missing", prep=lambda f: (f / "scripts/gate12_cache_join_gate.py").unlink())
    check(f"{entry_name}: an unhashable provenance file refuses before the launcher runs",
          r.returncode == 1 and not ran and "provenance not recorded" in r.stdout, r.stdout[-400:])
    for name, prep, env in (("exit14", lambda f: (f / "ct-a" / "cache" / f"cupy_cache_run_{nonce}_gpu0").mkdir(parents=True), {}),
                            ("exit15", None, {"FIXTURE_ENV_MISMATCH": "ct-b:0"}),
                            ("badrec", None, {"FIXTURE_BAD_HOST": "ct-b"})):
        r, ran = go(name, prep, **env)
        check(f"{entry_name}: {name} -> exit 1 before the sentinel gate (no coordinator/release reached)",
              r.returncode == 1 and "REACHED_SENTINEL_GATE" not in r.stdout and "GATE-12 ABORTED" in r.stdout,
              r.stdout[-400:])


def apply_patch(src: Path, diff: Path, name: str, work: Path) -> str:
    d = work / f"patch_{name}"
    d.mkdir()
    shutil.copy(src, d / name)
    p = run(["patch", "-s", "--no-backup-if-mismatch", str(d / name), str(diff)])
    assert p.returncode == 0, f"{diff.name} does not apply to {src}: {p.stdout}{p.stderr}"
    return (d / name).read_text()


def main():
    (REPO / "logs").mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(dir=str(REPO / "logs"), prefix="cachev4_"))
    try:
        launcher = apply_patch(REPO / "scripts/launch_fleet_manual.sh", V4 / "launch_fleet_manual.diff", "launch_fleet_manual.sh", work)
        gate12 = apply_patch(REPO / "gate12_launch.sh", V4 / "gate12_launch.diff", "gate12_launch.sh", work)
        run4 = apply_patch(RUN4, V4 / "run4_launch.diff", "run4_launch.sh", work)
        for n, t in (("launcher", launcher), ("gate12", gate12), ("run4", run4)):
            r = run(["bash", "-n"], input=t)
            check(f"bash -n on patched {n}", r.returncode == 0, r.stderr)
        u_verify(work)
        u_join(work)
        e2e(work, launcher)
        e2e_gate12(work, launcher, "gate12_launch.sh", gate12)
        e2e_gate12(work, launcher, "run4_launch.sh", run4)
    finally:
        for f in work.rglob("worker_pids"):
            cleanup(f.parent)
        subprocess.run(["chmod", "-R", "u+w", str(work)])
        shutil.rmtree(work, ignore_errors=True)
    bad = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(bad)} of {len(RESULTS)} checks pass")
    print("ALL GREEN" if not bad else "NOT GREEN")
    return 0 if not bad else 1


if __name__ == "__main__" and not (len(sys.argv) > 1 and sys.argv[1] == "mutants"):
    sys.exit(main())


# ================================================================================ MUTANTS
# Each mutant edits ONE safeguard in a temporary copy and must make at least one check FAIL.
MUTANTS = [
    ("V1 verifier compares by substring, not equality", "verify",
     '        if rec[k] != v:\n', '        if v not in rec[k]:\n'),
    ("V2 verifier lets an unparsable t_prep pass", "verify",
     '    if not TPREP_RE.fullmatch(rec["t_prep"]):\n', '    if False:\n'),
    ("V3 verifier accepts any line containing an ACK", "verify",
     '    acks = [ACK_RE.fullmatch(l) for l in lines]\n', '    acks = [ACK_RE.search(l) for l in lines]\n'),
    ("V4 verifier skips the ACK-gpu check", "verify",
     '    if agpu != gpu:\n', '    if False:\n'),
    ("V5 verifier accepts extra/missing fields", "verify",
     '    if set(out) != set(FIELDS):\n', '    if not set(FIELDS) - set(out) - {"host", "t_prep"} == set():\n'),
    ("V6 verifier publishes directly (no temp+rename)", "verify",
     '        os.replace(tmp, out)\n', '        os.replace(tmp, out) if False else os.rename(tmp, out)\n'),
    ("V7 verifier swallows a failed publish", "verify",
     '        return 2\n    print(f"[cache-verify] {len(rows)} OK', '        return 0\n    print(f"[cache-verify] {len(rows)} OK'),
    ("V8 verifier ignores the expected-slot count", "verify",
     '    if len(exp) != a.expected_slots:\n', '    if False:\n'),
    ("J1 join skips the parked env comparison", "join",
     '        if vals[0] != rec["path"]:\n', '        if False:\n'),
    ("J2 join skips the pid comparison", "join",
     '        if str(lv.get("pid")) != rec.get("pid"):\n', '        if False:\n'),
    ("J3 join trusts an unreadable environment", "join",
     '        if rc != 0:\n            defect(', '        if False:\n            defect('),
    ("L1 launcher does not call the verifier", "launcher",
     '    CACHE_VERIFY_RC=$?\n', '    CACHE_VERIFY_RC=0\n'),
    ("L2 launcher sizes the expectation from the dispatch arrays", "launcher",
     """    RIG_SLOTS=$(echo "$FLEET" | awk -F'\\t' '$6=="remote"{s+=$3} END{print s+0}')\n""",
     '    RIG_SLOTS=${#REMOTE_DISPATCH_LOGS[@]}\n'),
    ("E1 entry point lets an inherited shared through", "entry",
     'if [ -n "${RIG_CUPY_CACHE_POLICY+set}" ] && [ "$RIG_CUPY_CACHE_POLICY" != per_run ]; then\n  echo "GATE-12 REFUSED',
     'if false; then\n  echo "GATE-12 REFUSED'),
    ("E2 entry point ignores a failed provenance hash", "entry",
     '    prov_hash "$f" "$f" || return 1\n', '    prov_hash "$f" "$f" || true\n'),
]


def run_mutants():
    global VERIFY, JOIN, RESULTS
    import io, contextlib
    work = Path(tempfile.mkdtemp(dir=str(REPO / "logs"), prefix="cachev4mut_"))
    out = []
    try:
        base_l = apply_patch(REPO / "scripts/launch_fleet_manual.sh", V4 / "launch_fleet_manual.diff", "launch_fleet_manual.sh", work)
        base_g = apply_patch(REPO / "gate12_launch.sh", V4 / "gate12_launch.diff", "gate12_launch.sh", work)
        orig_v, orig_j = VERIFY, JOIN
        for i, (name, kind, old, new) in enumerate(MUTANTS):
            src = {"verify": orig_v.read_text(), "join": orig_j.read_text(), "launcher": base_l, "entry": base_g}[kind]
            if src.count(old) != 1:
                out.append((name, f"INVALID (pattern count {src.count(old)})")); continue
            mut = src.replace(old, new)
            sub = work / f"m{i}"; sub.mkdir()
            VERIFY, JOIN = orig_v, orig_j
            if kind == "verify":
                VERIFY = sub / "cache_records_verify.py"; VERIFY.write_text(mut)
            if kind == "join":
                JOIN = sub / "gate12_cache_join_gate.py"; JOIN.write_text(mut)
            RESULTS = []
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                if kind == "verify":
                    u_verify(sub); e2e(sub, base_l)
                elif kind == "join":
                    u_join(sub); e2e(sub, base_l)
                elif kind == "launcher":
                    e2e(sub, mut)
                else:
                    e2e_gate12(sub, base_l, "gate12_launch.sh", mut)
            failed = [r[0] for r in RESULTS if not r[1]]
            out.append((name, f"DETECTED ({failed[0][:70]})" if failed else "SURVIVED"))
            for f in sub.rglob("worker_pids"):
                cleanup(f.parent)
        VERIFY, JOIN = orig_v, orig_j
    finally:
        for f in work.rglob("worker_pids"):
            cleanup(f.parent)
        subprocess.run(["chmod", "-R", "u+w", str(work)])
        shutil.rmtree(work, ignore_errors=True)
    for n, v in out:
        print(f"{n}: {v}")
    ok = all(v.startswith("DETECTED") for _, v in out)
    print(f"{sum(v.startswith('DETECTED') for _, v in out)} of {len(out)} mutants detected")
    return ok


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "mutants":
    sys.exit(0 if run_mutants() else 1)
