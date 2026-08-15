#!/usr/bin/env python3
"""S172 D6-I2 — SENTINEL-CORRELATED WORKER LIVENESS GATE.

WHAT THIS SUITE IS FOR
----------------------
`scripts/gate12_worker_liveness_gate.py` exists because in D6 dry run #2 the
worker-log sentinel gate returned PASS 25/25 over a fleet whose local worker was
already dead. Everything about that was correct except the conclusion: delivery
was proven, liveness was assumed. This suite proves the new gate measures the
thing the old one could not, and — the harder half — that each of its arms can
actually go red.

EVERY ARM STATES WHAT WRONG INPUT MAKES IT RED. That is not decoration. The
`STUB_SSH_OK = "sleep .2; exit 0"` fixture was the seventh recorded instance of a
check that could not fail on the condition it claimed to cover, and it was
written after Beta named the pattern. So:

  * the processes here are REAL processes, with real PIDs, read through real
    /proc — never a mocked cmdline;
  * the mandatory regression drives BOTH gates over ONE fixture and requires
    them to DISAGREE (sentinel PASS, liveness REFUSE), which is the exact D6 #2
    hardware condition and is unsatisfiable by a gate that merely re-reads logs;
  * every mutant is proven APPLIED, EXECUTED and DETECTED.

THE STUB WORKER IS A SHELL SCRIPT NAMED `range_miner_worker.py`, and that is
deliberate rather than lazy: the gate's identity evidence is `/proc/<pid>/cmdline`
and nothing else, so a faithful stand-in is any process whose argv is the
worker's argv. Using the real worker would drag in a GPU and prove nothing extra
about the gate.
"""

import json
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import gate12_sentinel_gate as SG                                   # noqa: E402
import gate12_worker_liveness_gate as LG                            # noqa: E402

_RESULTS = []


def _porcelain():
    out = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                         capture_output=True, text=True).stdout
    return [ln for ln in out.rstrip("\n").splitlines() if ln]


_PORCELAIN_AT_START = _porcelain()

GREEN, RED, YELL, OFF = "\033[92m", "\033[91m", "\033[93m", "\033[0m"


def check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok), False))
    print(f"  [{GREEN}PASS{OFF}] {name}  {detail}" if ok
          else f"  [{RED}FAIL{OFF}] {name}  {detail}")


def unavailable(name, detail=""):
    _RESULTS.append((name, False, True))
    print(f"  [{YELL}UNAVAILABLE{OFF}] {name}  {detail}")


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES — real processes, real logs, real /proc
# ═══════════════════════════════════════════════════════════════════════════

# NO `exec` HERE, DELIBERATELY. `exec sleep 600` would replace the process image
# and with it /proc/<pid>/cmdline: the argv would become `sleep 600` and the
# fixture would stop being a worker at all — 15 arms went red proving it. So `sh`
# stays as the process the gate identifies, and it forks a `sleep` child. That
# child is why `spawn()` opens a new session and `close()` kills the GROUP:
# killing only the parent left ~160 orphaned `sleep`s per run, invisible to a
# leak check scoped to the spawned PIDs — a check narrowed until it could not
# fail, which is the pattern this suite exists to avoid.
STUB_WORKER = "#!/bin/sh\nsleep 600\n"

NONCE = "liveness-suite-nonce"
OLD_NONCE = "liveness-suite-OLD-nonce"

# Every PID this suite starts, so the leak check can be about THIS suite rather
# than about a pattern that other suites' fixtures also match.
_SPAWNED = []


def group_members(pgids):
    """Every live PID whose process GROUP is one of `pgids`.

    Read straight from /proc field 5 (pgrp). This is how a DESCENDANT is seen: a
    leak check that only asks about the PIDs it spawned cannot see the children
    they forked, and the stub worker forks exactly one.
    """
    out = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
        except (OSError, IndexError):
            continue
        try:
            if int(fields[2]) in pgids:
                out.add(int(entry.name))
        except (ValueError, IndexError):
            continue
    return out


def _alive(pid):
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


class Fleet:
    """A synthetic fleet of real processes, with the logs their sentinels would
    have written. Every process is killed on exit, including on failure."""

    def __init__(self, root):
        self.root = Path(root)
        (self.root / "miner").mkdir(parents=True, exist_ok=True)
        (self.root / "logs").mkdir(parents=True, exist_ok=True)
        self.worker = self.root / "miner" / "range_miner_worker.py"
        self.worker.write_text(STUB_WORKER)
        self.worker.chmod(self.worker.stat().st_mode | stat.S_IEXEC)
        self.procs = []

    def spawn(self, gpu, nonce=NONCE, release=None, argv_extra=None,
              entrypoint=None):
        """Start one real process carrying the worker's argv."""
        release = (release if release is not None
                   else self.release_path_for(nonce))
        argv = [str(entrypoint or self.worker),
                "--host", "127.0.0.1", "--port", "5700",
                "--gpu-id", str(gpu), "--device-index", "0"]
        if nonce is not None:
            argv += ["--run-nonce", nonce]
        if release:
            argv += ["--session-release-file", release,
                     "--release-deadline", "900"]
        argv += (argv_extra or [])
        # own session, so `close()` can kill the whole group and no descendant
        # can outlive the fixture
        p = subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
        self.procs.append(p)
        _SPAWNED.append(p.pid)
        return p

    def spawn_unrelated(self):
        """A process that is NOT a worker — for the pid-reuse arms.

        `start_new_session=True` is NOT optional here. Without it this process
        shares the TEST RUNNER'S process group, and `close()`'s group kill
        SIGKILLs the suite itself: cleanup killing its own reporter, which is
        VIR-4 exactly. It did, once, before this helper existed.
        """
        p = subprocess.Popen(["sleep", "600"], start_new_session=True)
        self.procs.append(p)
        _SPAWNED.append(p.pid)
        return p

    def release_path_for(self, nonce=NONCE):
        """The barrier path `spawn()` puts in the process's argv.

        [R2] ONE EXPRESSION, USED BY BOTH the spawned process and the log record
        it would have written. Before R2 the fixture spawned a worker carrying a
        release file under the fixture root while `log()` wrote
        `/tmp/gate12_release_<nonce>` into the WAIT record — two different paths,
        and every arm stayed green because the gate never compared them. The
        fixture was structurally similar to production and not internally
        consistent with it, which is exactly the state that hides a correlation
        defect.
        """
        return f"{self.root}/gate12_release_{nonce}"

    def log(self, worker_id, gpu, pid, nonce=NONCE, events=(),
            hostname="stubhost", wait=True, wait_worker_id=None,
            wait_nonce=None, wait_release_path=None, wait_event=None):
        """Write the log a worker with this identity would have written.

        The record is built the way `_emit_session_event` builds it — one JSON
        object per line, sorted keys — so this suite reads the same shape the
        production emitter produces (verified against the collected D6 dry run #2
        rig logs).

        [R1] `wait=False` writes the log of a worker that emitted its sentinel
        and has NOT YET entered `await_session_release` — the real interval
        between the two calls, and the state the gate used to call ALIVE+PARKED.
        `wait_worker_id` writes a wait record belonging to somebody else.
        """
        path = self.root / "logs" / f"{worker_id.replace(':', '_')}.log"
        lines = []
        rec = {"backend": "rocm", "device_index": 0,
               "event": "SESSION_SENTINEL", "gpu_id": gpu,
               "gpu_name": "AMD Radeon RX 6600 XT", "hostname": hostname,
               "log_path": str(path), "pid": pid, "python": "/usr/bin/python3",
               "run_nonce": nonce, "session_generation": 1,
               "worker_id": worker_id}
        lines.append("[MINER-SESSION] SESSION_SENTINEL "
                     + json.dumps(rec, sort_keys=True))
        if wait:
            # The record carries the SAME path the spawned process carries in
            # argv, unless an arm deliberately breaks one of the four fields.
            # `wait_nonce` sets the AUTHORITATIVE run_nonce independently of the
            # text the remote grep matches on, which is how L-18 builds the
            # record Beta described.
            lines.append("[MINER-SESSION] SESSION_RELEASE_WAIT " + json.dumps(
                {"deadline_s": 900.0,
                 "event": wait_event or "SESSION_RELEASE_WAIT",
                 "release_path": (wait_release_path
                                  if wait_release_path is not None
                                  else self.release_path_for(nonce)),
                 "run_nonce": wait_nonce or nonce, "session_generation": 1,
                 "worker_id": wait_worker_id or worker_id}, sort_keys=True))
        for ev in events:
            lines.append(f"[MINER-SESSION] {ev} " + json.dumps(
                {"event": ev, "run_nonce": nonce, "worker_id": worker_id,
                 "waited_s": 1.0}, sort_keys=True))
        path.write_text("\n".join(lines) + "\n")
        return path

    def close(self):
        # TWO PASSES, and the second is not belt-and-braces. `sh` forks its
        # `sleep` child a moment after Popen returns, so a group kill that lands
        # in that window kills the parent and misses a child that appears
        # immediately after. The second pass re-signals every group that still
        # has members. Verified by the descendant check in PART F, which is what
        # found the single survivor the first version left behind.
        for _ in range(2):
            for p in self.procs:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
            for pid in group_members({p.pid for p in self.procs}):
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            time.sleep(0.05)
        for p in self.procs:
            try:
                p.kill()
            except OSError:
                pass
        for p in self.procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


def target(worker_id, gpu, local=True, endpoint="localhost"):
    return {"worker_id": worker_id, "gpu": gpu, "local": local,
            "endpoint": endpoint, "ssh_user": "michael",
            "node_id": endpoint, "worker_hostname": worker_id.split(":")[0]}


def probe(fleet, worker_id, gpu, nonce=NONCE, log=None):
    lp = str(log or (fleet.root / "logs" / f"{worker_id.replace(':', '_')}.log"))
    return LG.probe_liveness(target(worker_id, gpu), nonce, lp)


# ---- ssh shims, for the REMOTE branch --------------------------------------
SHIM_EXEC = "#!/usr/bin/env bash\nbash -c \"${@: -1}\"\n"
SHIM_255 = ("#!/usr/bin/env bash\n"
            "echo 'ssh: connect to host port 22: No route to host' >&2\n"
            "exit 255\n")
SHIM_TRUNCATED = ("#!/usr/bin/env bash\n"
                  "echo TFM-LIVENESS-BEGIN\necho LOG_READABLE=1\nexit 0\n")
SHIM_GARBAGE = ("#!/usr/bin/env bash\n"
                "echo TFM-LIVENESS-BEGIN\necho wat\necho TFM-LIVENESS-END\n")


class ShimPath:
    """Put a fake `ssh` first on PATH for the duration of the block."""

    def __init__(self, src):
        self.src = src
        self.td = None

    def __enter__(self):
        self.td = tempfile.TemporaryDirectory()
        p = Path(self.td.name) / "ssh"
        p.write_text(self.src)
        p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        self._old = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{self.td.name}:{self._old}"
        return self

    def __exit__(self, *exc):
        os.environ["PATH"] = self._old
        self.td.cleanup()
        return False


def _uncommented(text):
    """Shell/Python `#` comment lines removed."""
    return "\n".join(ln for ln in text.splitlines()
                     if not ln.lstrip().startswith("#"))


def _executable_text(py_src):
    """Python source with every docstring removed — the project's AST idiom, so
    a claim about what the CODE does is not answered by what the prose says."""
    import ast
    tree = ast.parse(py_src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(ast.fix_missing_locations(tree))


def _mutant_span(start_marker, end_marker, name, replacement=""):
    """Excise a whole SPAN of live source, not one line.

    Needed where a requirement is several statements. Deleting only its first
    line leaves the rest to refuse for a neighbouring reason, and the mutant then
    proves defence in depth rather than the property it claims to remove — which
    is how the first M-7 passed as a false detection.
    """
    src = (REPO / "scripts" / "gate12_worker_liveness_gate.py").read_text()
    i = src.index(start_marker)
    j = src.index(end_marker, i)
    mutated = src[:i] + replacement + src[j:]
    assert mutated != src, f"mutation did not apply for {name}"
    assert start_marker not in mutated, f"span not fully removed for {name}"
    mod = types.ModuleType(name)
    mod.__file__ = str(REPO / "scripts" / "gate12_worker_liveness_gate.py")
    exec(compile(mutated, mod.__file__, "exec"), mod.__dict__)   # noqa: S102
    return mod


def _mutant(old, new, name):
    """Compile a source-level mutant of the LIVE gate into a fresh module.

    APPLIED is proven by asserting the substitution changed the text: a mutation
    that silently failed to apply looks exactly like one that was killed.
    """
    src = (REPO / "scripts" / "gate12_worker_liveness_gate.py").read_text()
    assert old in src, f"mutation site not found for {name}"
    mutated = src.replace(old, new, 1)
    assert mutated != src
    mod = types.ModuleType(name)
    mod.__file__ = str(REPO / "scripts" / "gate12_worker_liveness_gate.py")
    exec(compile(mutated, mod.__file__, "exec"), mod.__dict__)   # noqa: S102
    return mod


# ═══════════════════════════════════════════════════════════════════════════
# PART A — THE JOIN, per identity
# ═══════════════════════════════════════════════════════════════════════════

def part_a():
    section("PART A — THE SENTINEL->PID->ARGV JOIN")
    with tempfile.TemporaryDirectory() as td:
        f = Fleet(td)
        try:
            # ---- L-1 the green case ---------------------------------------
            # RED WHEN: any of the eleven requirements is unsatisfied — this is the
            # only shape the gate accepts, so every arm below is a perturbation
            # of exactly this one.
            p = f.spawn(3)
            f.log("stubhost:gpu3", 3, p.pid)
            r = probe(f, "stubhost:gpu3", 3)
            ok, refusals = LG.evaluate([r])
            check("L-1  sentinel + live parked process -> PASS",
                  r["status"] == LG.PROBE_OK and r["reason"] is None and ok,
                  f"pid={r['pid']} argv joined")

            # ---- L-2 the PID is gone --------------------------------------
            # RED WHEN: the gate reads the log and never looks at /proc — which
            # is precisely the sentinel gate's (correct, narrower) contract.
            p2 = f.spawn(4)
            f.log("stubhost:gpu4", 4, p2.pid)
            p2.kill()
            p2.wait(timeout=5)
            r2 = probe(f, "stubhost:gpu4", 4)
            ok2, ref2 = LG.evaluate([r2])
            check("L-2  sentinel delivered, PROCESS GONE -> REFUSE",
                  not ok2 and "pid_not_present" in (r2["reason"] or ""),
                  r2["reason"])

            # ---- L-3 the PID was reused by an unrelated process ------------
            # RED WHEN: existence is treated as identity. A pid that exists is
            # not the worker; this is the difference between `kill -0` and a
            # gate.
            other = f.spawn_unrelated()
            f.log("stubhost:gpu5", 5, other.pid)
            r3 = probe(f, "stubhost:gpu5", 5)
            check("L-3  PID reused by an UNRELATED process -> REFUSE",
                  LG.evaluate([r3])[0] is False
                  and "pid_reused_by_unrelated_process" in (r3["reason"] or ""),
                  r3["reason"])

            # ---- L-4 right process, wrong nonce ---------------------------
            # RED WHEN: the argv nonce is not checked, so LAST run's surviving
            # worker satisfies THIS run's gate. The sentinel record carries the
            # current nonce; only the process argv disagrees.
            p4 = f.spawn(6, nonce="some-other-run")
            f.log("stubhost:gpu6", 6, p4.pid)
            r4 = probe(f, "stubhost:gpu6", 6)
            check("L-4  correct worker process, WRONG NONCE in argv -> REFUSE",
                  LG.evaluate([r4])[0] is False
                  and "wrong_or_missing_run_nonce_in_argv" in (r4["reason"] or ""),
                  r4["reason"])

            # ---- L-5 right process, wrong gpu -----------------------------
            # RED WHEN: the gate counts workers instead of identifying them —
            # 25 live workers all on gpu0 would pass.
            p5 = f.spawn(1)
            f.log("stubhost:gpu7", 7, p5.pid)
            r5 = probe(f, "stubhost:gpu7", 7)
            check("L-5  process is on the WRONG GPU -> REFUSE",
                  LG.evaluate([r5])[0] is False
                  and "wrong_or_missing_gpu_id_in_argv" in (r5["reason"] or ""),
                  r5["reason"])

            # ---- L-6 `--gpu-id 3` is not satisfied by `30` ----------------
            # RED WHEN: argv is matched by substring instead of adjacent pairs.
            # This is the prefix-as-exact defect Beta found in F1/R1, in a new
            # place, so it is measured rather than assumed absent.
            p6 = f.spawn(30)
            f.log("stubhost:gpu3b", 3, p6.pid)
            r6 = probe(f, "stubhost:gpu3b", 3)
            check("L-6  `--gpu-id 3` is NOT satisfied by `--gpu-id 30` -> REFUSE",
                  LG.evaluate([r6])[0] is False
                  and "wrong_or_missing_gpu_id_in_argv" in (r6["reason"] or ""),
                  r6["reason"])

            # ---- L-7 already released -------------------------------------
            # RED WHEN: "alive" is accepted for "parked". A released worker is
            # alive and is no longer in the state the gate is asked about.
            p7 = f.spawn(8)
            f.log("stubhost:gpu8", 8, p7.pid, events=("SESSION_RELEASED",))
            r7 = probe(f, "stubhost:gpu8", 8)
            check("L-7  worker already SESSION_RELEASED -> REFUSE",
                  LG.evaluate([r7])[0] is False
                  and "worker_already_released" in (r7["reason"] or ""),
                  r7["reason"])

            # ---- L-8 already aborted --------------------------------------
            p8 = f.spawn(9)
            f.log("stubhost:gpu9", 9, p8.pid,
                  events=("SESSION_RELEASE_ABORTED",))
            r8 = probe(f, "stubhost:gpu9", 9)
            check("L-8  worker already SESSION_RELEASE_ABORTED -> REFUSE",
                  LG.evaluate([r8])[0] is False
                  and "worker_already_aborted" in (r8["reason"] or ""),
                  r8["reason"])

            # ---- L-9 no barrier arguments ---------------------------------
            # RED WHEN: a worker that is running but was never given a release
            # file is counted as parked. It is running toward REGISTER.
            p9 = f.spawn(10, release="")
            f.log("stubhost:gpu10", 10, p9.pid)
            r9 = probe(f, "stubhost:gpu10", 10)
            check("L-9  no --session-release-file in argv -> REFUSE",
                  LG.evaluate([r9])[0] is False
                  and "no_session_release_file_argument" in (r9["reason"] or ""),
                  r9["reason"])

            # ---- L-10 barrier file belongs to another run -----------------
            p10 = f.spawn(11, release=f"/tmp/gate12_release_{OLD_NONCE}")
            f.log("stubhost:gpu11", 11, p10.pid)
            r10 = probe(f, "stubhost:gpu11", 11)
            check("L-10 barrier file is a PREVIOUS run's -> REFUSE",
                  LG.evaluate([r10])[0] is False
                  and "barrier_file_is_not_this_run" in (r10["reason"] or ""),
                  r10["reason"])

            # ---- L-11 stale sentinel only ---------------------------------
            # RED WHEN: the nonce is not required in the sentinel search, so a
            # previous run's log satisfies this run.
            p11 = f.spawn(12)
            f.log("stubhost:gpu12", 12, p11.pid, nonce=OLD_NONCE)
            r11 = probe(f, "stubhost:gpu12", 12)
            check("L-11 log holds only a PREVIOUS nonce's sentinel -> REFUSE",
                  LG.evaluate([r11])[0] is False
                  and "no_sentinel_record_for_this_nonce" in (r11["reason"] or ""),
                  r11["reason"])

            # ---- L-12 another worker's record in this log -----------------
            # RED WHEN: the gate proves *a* worker is alive rather than *this*
            # worker — the shape that lets one live process satisfy 25 slots.
            p12 = f.spawn(13)
            f.log("stubhost:gpu13", 13, p12.pid)
            r12 = LG.probe_liveness(
                target("stubhost:gpu14", 14), NONCE,
                str(f.root / "logs" / "stubhost_gpu13.log"))
            check("L-12 the log carries ANOTHER identity's sentinel -> REFUSE",
                  LG.evaluate([r12])[0] is False
                  and "sentinel_identity_mismatch" in (r12["reason"] or ""),
                  r12["reason"])

            # ---- L-13 one process claiming two identities -----------------
            # RED WHEN: uniqueness is not checked. Both identities pass every
            # per-identity test; only the cross-identity rule sees it.
            p13 = f.spawn(15)
            f.log("stubhost:gpu15", 15, p13.pid)
            f.log("stubhost:gpu15b", 15, p13.pid)
            ra = probe(f, "stubhost:gpu15", 15)
            rb = LG.probe_liveness(
                target("stubhost:gpu15b", 15), NONCE,
                str(f.root / "logs" / "stubhost_gpu15b.log"))
            okd, refd = LG.evaluate([ra, rb])
            check("L-13 the SAME PID claimed by two identities -> REFUSE",
                  okd is False and any("duplicate_pid" in x for x in refd),
                  refd[0] if refd else "")

            # ---- L-14 PIDs may repeat across HOSTS ------------------------
            # RED WHEN: uniqueness is global. PIDs are per kernel, so a correct
            # 25-worker fleet across four hosts would be refused ~always.
            rc1 = dict(ra)
            rc2 = dict(ra)
            rc2["worker_id"], rc2["endpoint"] = "rrig6600b:gpu0", "192.168.3.156"
            rc1["endpoint"] = "192.168.3.122"
            okh, refh = LG.evaluate([rc1, rc2])
            check("L-14 the same PID on DIFFERENT hosts is NOT a duplicate",
                  okh is True, f"refusals={refh}")

            # ---- L-15 CONFIGURED TO PARK IS NOT PARKED --------------------
            # [R1] THE ARM BETA'S REVIEW WAS RETURNED FOR. Everything is
            # perfect: current sentinel, live process, correct worker, correct
            # nonce, correct gpu, correct --session-release-file, no RELEASED,
            # no ABORTED. The ONLY missing fact is that the worker has not yet
            # emitted SESSION_RELEASE_WAIT — it is between
            # `emit_startup_sentinel()` and `await_session_release()`, which is
            # a real interval, not a hypothetical.
            #
            # RED WHEN: the gate infers "parked" from `--session-release-file`
            # in argv. That argument is a property of how the process was
            # LAUNCHED; it says nothing about where the process has got to. This
            # is the exact shape of a verdict exceeding its measurement, and
            # before R1 this fixture PASSED as ALIVE+PARKED.
            p15 = f.spawn(16)
            f.log("stubhost:gpu16", 16, p15.pid, wait=False)
            r15 = probe(f, "stubhost:gpu16", 16)
            check("L-15 alive + correctly configured but NO "
                  "SESSION_RELEASE_WAIT -> REFUSE",
                  LG.evaluate([r15])[0] is False
                  and "not_parked_no_release_wait_record" in (r15["reason"] or ""),
                  r15["reason"])

            # ---- L-16 the positive control for L-15 -----------------------
            # The SAME fixture with the wait record added must PASS. Without
            # this, L-15 could be red for any reason at all and still look like
            # a detection — the control that makes the arm a measurement.
            p16 = f.spawn(17)
            f.log("stubhost:gpu17", 17, p16.pid)
            r16 = probe(f, "stubhost:gpu17", 17)
            check("L-16 …identical fixture WITH the wait record -> PASS",
                  r16["status"] == LG.PROBE_OK and r16["reason"] is None
                  and LG.evaluate([r16])[0] is True,
                  f"wait={r16['release_waits']}")

            # ---- L-17 the wait record must be THIS worker's ---------------
            # RED WHEN: presence is counted without reading whose it is. Two
            # workers' logs are separate files, but a stale or misrouted record
            # in this file would otherwise let one worker's arrival stand in for
            # another's.
            p17 = f.spawn(18)
            f.log("stubhost:gpu18", 18, p17.pid,
                  wait_worker_id="stubhost:gpu99")
            r17 = probe(f, "stubhost:gpu18", 18)
            check("L-17 the wait record belongs to ANOTHER identity -> REFUSE",
                  LG.evaluate([r17])[0] is False
                  and "release_wait_identity_mismatch" in (r17["reason"] or ""),
                  r17["reason"])

            # ---- L-18 TEXT CONTAINMENT IS NOT SEMANTIC EQUALITY -----------
            # [R2] THE ARM BETA'S REVIEW WAS RETURNED FOR. The record's
            # authoritative `run_nonce` names ANOTHER run, while the current
            # nonce appears in the line as text — inside `release_path`, which
            # is where it legitimately lives. The remote
            # `grep 'SESSION_RELEASE_WAIT' | grep '<nonce>'` therefore ships it,
            # and before R2 `classify()` read only `worker_id` from it and
            # accepted it as this run's parking proof.
            #
            # RED WHEN: the gate treats the grep as authoritative parsing — i.e.
            # a text match standing in for equality. This exact fixture PASSED
            # as ALIVE+PARKED before R2.
            p18 = f.spawn(21)
            f.log("stubhost:gpu21", 21, p18.pid, wait_nonce="some-other-run")
            r18 = probe(f, "stubhost:gpu21", 21)
            check("L-18 wait record carries the nonce as TEXT but its JSON "
                  "run_nonce is another run's -> REFUSE",
                  LG.evaluate([r18])[0] is False
                  and "release_wait_nonce_mismatch" in (r18["reason"] or ""),
                  r18["reason"])

            # ---- L-19 the record must name the LIVE PROCESS's barrier ------
            # RED WHEN: the record is only checked for self-consistency. Correct
            # worker, correct nonce, correct event — and a release_path the
            # running process does not carry. Without this the log record and
            # the process are two separate stories that never have to agree, and
            # the correlation the gate claims is not performed.
            p19 = f.spawn(22)
            f.log("stubhost:gpu22", 22, p19.pid,
                  wait_release_path=f"/tmp/gate12_release_{NONCE}")
            r19 = probe(f, "stubhost:gpu22", 22)
            check("L-19 wait record's release_path is NOT the live process's "
                  "--session-release-file -> REFUSE",
                  LG.evaluate([r19])[0] is False
                  and "release_wait_path_mismatch" in (r19["reason"] or ""),
                  r19["reason"])

            # ---- L-20 the positive control for L-18 and L-19 ---------------
            # Exact worker, exact nonce, exact barrier path. Without this, both
            # arms above could be red for any reason and still look like
            # detections.
            p20 = f.spawn(23)
            f.log("stubhost:gpu23", 23, p20.pid)
            r20 = probe(f, "stubhost:gpu23", 23)
            check("L-20 exact worker + exact nonce + exact barrier path -> PASS",
                  r20["status"] == LG.PROBE_OK and r20["reason"] is None
                  and LG.evaluate([r20])[0] is True,
                  f"wait={r20['release_waits']} "
                  f"path={f.release_path_for()}")

            # ---- L-21 the event field is read, not assumed -----------------
            # RED WHEN: a line is trusted because it CONTAINS the event name.
            # The remote grep matches the string anywhere in the line, so a
            # record whose own `event` is something else still arrives here.
            p21 = f.spawn(24)
            f.log("stubhost:gpu24", 24, p21.pid,
                  wait_event="SESSION_RELEASE_WAIT_SOMETHING_ELSE")
            r21 = probe(f, "stubhost:gpu24", 24)
            check("L-21 the record's own event field is not "
                  "SESSION_RELEASE_WAIT -> REFUSE",
                  LG.evaluate([r21])[0] is False
                  and "release_wait_wrong_event" in (r21["reason"] or ""),
                  r21["reason"])
        finally:
            f.close()


# ═══════════════════════════════════════════════════════════════════════════
# PART B — THE MANDATORY REGRESSION: the exact D6 dry run #2 condition
# ═══════════════════════════════════════════════════════════════════════════

def part_b():
    section("PART B — 25 SENTINELS / 24 LIVE (the D6 #2 hardware condition)")
    with tempfile.TemporaryDirectory() as td:
        f = Fleet(td)
        try:
            targets, logs = [], []
            for i in range(25):
                p = f.spawn(i)
                wid = f"stubhost:gpu{i}"
                logs.append(str(f.log(wid, i, p.pid)))
                targets.append(target(wid, i))
            # 25/25 first: the control. Without it, the 24-live arm below could
            # be red for any reason at all and still look like a detection.
            live = [LG.probe_liveness(t, NONCE, lp)
                    for t, lp in zip(targets, logs)]
            ok25, ref25 = LG.evaluate(live)
            sent25 = [SG.probe_sentinel(t, NONCE, lp)
                      for t, lp in zip(targets, logs)]
            sok25, _ = SG.evaluate(sent25)
            check("R-1  CONTROL: 25 sentinels + 25 live -> BOTH gates PASS",
                  ok25 and sok25, f"liveness={ok25} sentinel={sok25}")

            # Now kill exactly one — the D6 #2 condition, reproduced.
            victim = f.procs[7]
            victim.kill()
            victim.wait(timeout=5)
            sent = [SG.probe_sentinel(t, NONCE, lp)
                    for t, lp in zip(targets, logs)]
            sok, sref = SG.evaluate(sent)
            live24 = [LG.probe_liveness(t, NONCE, lp)
                      for t, lp in zip(targets, logs)]
            lok, lref = LG.evaluate(live24)
            check("R-2  25 sentinels + 24 live -> SENTINEL still PASSES",
                  sok, "delivery was real and the record is still true — the "
                       "sentinel gate is NOT widened")
            check("R-3  25 sentinels + 24 live -> LIVENESS REFUSES",
                  lok is False and len(lref) == 1
                  and "pid_not_present" in lref[0], lref[0] if lref else "")
            check("R-4  COMPOSITE: sentinel PASS alone does not authorize launch",
                  sok and not lok,
                  "the two gates DISAGREE on this fixture, which is the whole "
                  "reason the second one exists")
            # …and the refusal names the identity, not just a count.
            check("R-5  the refusal names WHICH identity is not alive",
                  "stubhost:gpu7" in lref[0], lref[0] if lref else "")
        finally:
            f.close()


# ═══════════════════════════════════════════════════════════════════════════
# PART C — UNAVAILABLE vs ERROR vs a count (the remote branch)
# ═══════════════════════════════════════════════════════════════════════════

def part_c():
    section("PART C — UNAVAILABLE IS NEVER A COUNT")
    t = target("rrig6600:gpu3", 3, local=False, endpoint="192.168.3.122")

    # RED WHEN: `proc.returncode` is not read, so a transport failure arrives as
    # empty stdout and is classified as though the probe had run. Both refuse,
    # so the consequence is evidentiary — and evidence is the point.
    with ShimPath(SHIM_255):
        r = LG.probe_liveness(t, NONCE, "/tmp/minerlogs/gpu3.log")
    check("C-1  ssh transport failure (255) -> UNAVAILABLE, not ERROR",
          r["status"] == LG.PROBE_UNAVAILABLE
          and "ssh_transport_failure" in (r["reason"] or ""), r["reason"])
    check("C-2  …and it REFUSES", LG.evaluate([r])[0] is False)
    rendered = LG.render(r)
    check("C-3  …and never renders count-shaped",
          "UNAVAILABLE" in rendered
          and not re.search(r"\b0/\d", rendered), rendered.strip())

    # RED WHEN: the END sentinel is not required. A truncated record's counts are
    # lower bounds; read as facts they say "no sentinels, no processes" — the
    # UNAVAILABLE-rendered-as-zero defect, one layer out.
    with ShimPath(SHIM_TRUNCATED):
        r2 = LG.probe_liveness(t, NONCE, "/tmp/minerlogs/gpu3.log")
    check("C-4  truncated probe output -> UNAVAILABLE",
          r2["status"] == LG.PROBE_UNAVAILABLE
          and "truncated" in (r2["reason"] or ""), r2["reason"])

    # RED WHEN: unclassifiable output is silently treated as an empty result.
    with ShimPath(SHIM_GARBAGE):
        r3 = LG.probe_liveness(t, NONCE, "/tmp/minerlogs/gpu3.log")
    check("C-5  unclassifiable probe output -> ERROR (not UNAVAILABLE, not 0)",
          r3["status"] == LG.PROBE_ERROR, r3["reason"])

    # The remote branch, executed for real: same classification as local.
    with tempfile.TemporaryDirectory() as td:
        f = Fleet(td)
        try:
            p = f.spawn(3)
            lp = str(f.log("rrig6600:gpu3", 3, p.pid))
            with ShimPath(SHIM_EXEC):
                r4 = LG.probe_liveness(t, NONCE, lp)
            check("C-6  the REMOTE branch performs the same join",
                  r4["status"] == LG.PROBE_OK and r4["reason"] is None
                  and str(r4["pid"]) == str(p.pid), f"pid={r4['pid']}")
        finally:
            f.close()

    # RED WHEN: an unreadable log is reported as "no sentinel found".
    r5 = LG.probe_liveness(target("stubhost:gpu0", 0), NONCE,
                           "/nonexistent/path/gpu0.log")
    check("C-7  unreadable log -> UNAVAILABLE, not 'no sentinel'",
          r5["status"] == LG.PROBE_UNAVAILABLE
          and r5["reason"] == "log_unreadable", r5["reason"])

    # RED WHEN: `evaluate` acquires a default-accept path. A status it does not
    # recognise must refuse, not fall through.
    bogus = {"worker_id": "x:gpu0", "endpoint": "e", "pid": "1",
             "status": "WEIRD", "reason": None, "log_path": "/l"}
    check("C-8  an unrecognised probe status REFUSES by default",
          LG.evaluate([bogus])[0] is False)


# ═══════════════════════════════════════════════════════════════════════════
# PART D — COMPOSITION: where the wall sits, and what may decide a launch
# ═══════════════════════════════════════════════════════════════════════════

def part_d():
    section("PART D — COMPOSITION IN gate12_launch.sh")
    launch = (REPO / "gate12_launch.sh").read_text()
    # The INVOCATIONS, not the first textual mention: this file's header
    # discusses the sampler 300 lines above the line that starts it, and an
    # ordering arm that compares prose positions measures the prose.
    i_sent = launch.find("python3 -u scripts/gate12_sentinel_gate.py --phase verify")
    i_live = launch.find("python3 -u scripts/gate12_worker_liveness_gate.py")
    i_samp = launch.find("setsid nohup python3 -u scripts/gate12_concurrency_sampler.py")
    i_coord = launch.find("nohup env PYTHONPATH=. python3 agents/watcher_agent.py")
    i_rel = launch.find("--phase release")

    # RED WHEN: the wall is placed after the coordinator (where it would be
    # measuring a fleet that has already committed GPU-seconds and frozen a
    # cohort), or before the sentinel gate (where there is no sentinel to
    # correlate with yet).
    check("D-1  liveness gate runs AFTER the sentinel gate",
          -1 < i_sent < i_live, f"sentinel@{i_sent} liveness@{i_live}")
    check("D-2  …and BEFORE the sampler and the coordinator",
          i_live < i_samp and i_live < i_coord,
          f"liveness@{i_live} sampler@{i_samp} coordinator@{i_coord}")
    check("D-3  …and before any release token is written",
          i_live < i_rel, f"release@{i_rel}")

    # RED WHEN: the result is read from the pipeline instead of the gate — `cmd
    # | tee` exits with tee's status, which is 0 essentially always, and the gate
    # becomes decorative. This is the defect class attempt 3 shipped.
    block = launch[i_live:i_live + 1400]
    check("D-4  the verdict is read via ${PIPESTATUS[0]}, not the pipeline",
          "LIVENESS_RC=${PIPESTATUS[0]}" in block)
    check("D-5  a nonzero verdict ABORTS the launch",
          re.search(r'if \[ "\$LIVENESS_RC" -ne 0 \]; then', block) is not None
          and "exit 1" in block)
    check("D-6  a refusal kills the parked fleet rather than leaving it",
          "pkill -f \"[r]ange_miner_worker\"" in block)

    # RED WHEN: `pgrep -c -f` returns as an acceptance input anywhere in the new
    # wall or the gate. Beta retired it: 16 for 8 workers, and blind to identity.
    # Docstrings AND comments stripped before the search. Both files discuss
    # `pgrep` at length — that prose is the record of Beta retiring it, and an
    # assertion that cannot tell the executable from the warning about it would
    # be red for saying the right thing.
    gate_src = (REPO / "scripts" / "gate12_worker_liveness_gate.py").read_text()
    check("D-7  the liveness gate does not use pgrep as an authority",
          "pgrep" not in _executable_text(gate_src)
          and "pgrep" not in _uncommented(block),
          "named only in the prose that retires it")

    # G — the sentinel gate's PASS wording no longer overstates its authority.
    sent_src = (REPO / "scripts" / "gate12_sentinel_gate.py").read_text()
    check("D-8  sentinel PASS no longer says 'Launch may proceed'",
          "Launch may proceed" not in _uncommented(sent_src)
          and "Launch may proceed" in sent_src,
          "gone from the printed text, and the comment records why")
    check("D-9  …and says the remaining prelaunch gates must also pass",
          "remaining prelaunch gates must " in sent_src
          and "proved current-run session-log delivery" in sent_src)

    # RED WHEN: the two gates drift into asking about different workers or
    # reading different files — which would make the join a join in name only.
    check("D-10 the liveness gate derives identities from the SENTINEL gate",
          "SG.log_path_for" in gate_src and "SG.worker_targets" in gate_src
          and "def log_path_for" not in gate_src)


# ═══════════════════════════════════════════════════════════════════════════
# PART E — MUTANTS (applied · executed · detected)
# ═══════════════════════════════════════════════════════════════════════════

def part_e():
    section("PART E — MUTANTS")
    with tempfile.TemporaryDirectory() as td:
        f = Fleet(td)
        try:
            # a live worker, and a dead one, and a reused pid
            alive_p = f.spawn(0)
            f.log("stubhost:gpu0", 0, alive_p.pid)
            dead_p = f.spawn(1)
            f.log("stubhost:gpu1", 1, dead_p.pid)
            dead_p.kill(); dead_p.wait(timeout=5)
            other = f.spawn_unrelated()
            f.log("stubhost:gpu2", 2, other.pid)
            # ONLY the argv nonce is wrong: the barrier file is this run's, so
            # the downstream barrier check cannot mask M-2's removal. Isolating
            # the mutated check is what makes the mutant a measurement.
            wrong_nonce_p = f.spawn(
                3, nonce="another-run",
                release=f"{f.root}/gate12_release_{NONCE}")
            f.log("stubhost:gpu3", 3, wrong_nonce_p.pid)
            rel_p = f.spawn(4)
            f.log("stubhost:gpu4", 4, rel_p.pid, events=("SESSION_RELEASED",))

            def _probe_with(mod, wid, gpu):
                return mod.probe_liveness(
                    target(wid, gpu), NONCE,
                    str(f.root / "logs" / f"{wid.replace(':', '_')}.log"))

            # M-1: liveness stops being required — the PRE-D6-I2 launch
            # authority, in one substitution: a delivered sentinel is accepted
            # whether or not the process it named still exists.
            #
            # NOTE ON THE SHAPE OF THIS MUTANT. Merely deleting the branch
            # (`if False:`) is NOT detected, and that is a fact worth recording
            # rather than hiding: the argv identity check downstream refuses the
            # literal string "ABSENT" too, so the gate has two independent
            # reasons to refuse a dead worker. A mutant must isolate the check it
            # kills, so this one makes the absent case AFFIRMATIVELY pass.
            m1 = _mutant(
                '    if argv_raw == "ABSENT":\n'
                '        result.update(status=PROBE_OK,\n'
                '                      reason=f"pid_not_present:{pid} — the sentinel was "\n'
                '                             f"delivered and the process is GONE")\n'
                '        return result',
                '    if argv_raw == "ABSENT":\n'
                '        result.update(status=PROBE_OK, reason=None)\n'
                '        return result', "lg_m1")
            real = probe(f, "stubhost:gpu1", 1)
            mut = _probe_with(m1, "stubhost:gpu1", 1)
            check("M-1  PID-existence check removed -> DETECTED",
                  LG.evaluate([real])[0] is False
                  and m1.evaluate([mut])[0] is True,
                  "real REFUSES a dead worker, mutant ACCEPTS it")

            # M-2: the argv nonce check is removed.
            m2 = _mutant('    if ("--run-nonce", str(nonce)) not in pairs:',
                         '    if False:', "lg_m2")
            real2 = probe(f, "stubhost:gpu3", 3)
            mut2 = _probe_with(m2, "stubhost:gpu3", 3)
            check("M-2  argv nonce check removed -> DETECTED",
                  LG.evaluate([real2])[0] is False
                  and m2.evaluate([mut2])[0] is True,
                  "the mutant accepts a PREVIOUS run's surviving worker")

            # M-3: the identity check on the process is removed.
            m3 = _mutant(
                "    if not any(a.endswith(WORKER_ENTRYPOINT_BASENAME) for a in argv):",
                "    if False:", "lg_m3")
            real3 = probe(f, "stubhost:gpu2", 2)
            mut3 = _probe_with(m3, "stubhost:gpu2", 2)
            check("M-3  worker-identity check removed -> DETECTED",
                  LG.evaluate([real3])[0] is False
                  and mut3["reason"] != real3["reason"],
                  f"real={real3['reason']!r} mutant={mut3['reason']!r}")

            # M-4: released/aborted no longer disqualify.
            m4 = _mutant("        if int(value):", "        if False:", "lg_m4")
            real4 = probe(f, "stubhost:gpu4", 4)
            mut4 = _probe_with(m4, "stubhost:gpu4", 4)
            check("M-4  released/aborted check removed -> DETECTED",
                  LG.evaluate([real4])[0] is False
                  and m4.evaluate([mut4])[0] is True,
                  "the mutant accepts a worker that has left the barrier")

            # M-5: the cross-identity duplicate-PID rule is removed.
            m5 = _mutant("            if key in seen:", "            if False:",
                         "lg_m5")
            r_ok = probe(f, "stubhost:gpu0", 0)
            dup = dict(r_ok)
            dup["worker_id"] = "stubhost:gpu99"
            check("M-5  duplicate-PID rule removed -> DETECTED",
                  LG.evaluate([r_ok, dup])[0] is False
                  and m5.evaluate([r_ok, dup])[0] is True,
                  "one process satisfying two identities")

            # M-6: UNAVAILABLE becomes acceptable.
            m6 = _mutant(
                '        elif r["status"] == PROBE_UNAVAILABLE:\n'
                '            refusals.append(',
                '        elif r["status"] == PROBE_UNAVAILABLE:\n'
                '            _ = (', "lg_m6")
            with ShimPath(SHIM_255):
                ru = LG.probe_liveness(
                    target("rrig6600:gpu3", 3, local=False,
                           endpoint="192.168.3.122"), NONCE, "/tmp/x.log")
            check("M-6  UNAVAILABLE treated as acceptable -> DETECTED",
                  LG.evaluate([ru])[0] is False
                  and m6.evaluate([ru])[0] is True,
                  "an unreachable rig would authorize a launch")

            # M-7: [R1] the SESSION_RELEASE_WAIT requirement is removed —
            # i.e. the gate as Beta received it in review. It must fail on the
            # L-15 fixture, and the pre-R1 gate did not.
            m7 = _mutant_span(
                "    if not wait_lines:",
                "    result.update(status=PROBE_OK, reason=None)", "lg_m7")
            nowait_p = f.spawn(20)
            f.log("stubhost:gpu20", 20, nowait_p.pid, wait=False)
            real7 = probe(f, "stubhost:gpu20", 20)
            mut7 = _probe_with(m7, "stubhost:gpu20", 20)
            check("M-7  SESSION_RELEASE_WAIT requirement removed -> DETECTED",
                  LG.evaluate([real7])[0] is False
                  and m7.evaluate([mut7])[0] is True,
                  "the mutant calls a worker that never reached the barrier "
                  "ALIVE+PARKED — which is what the pre-R1 gate did")

            # M-8: [R2] the SEMANTIC correlation is removed and the pre-R2
            # behaviour restored — the record is parsed for `worker_id` alone,
            # exactly as R1 left it, so the grep is authoritative again.
            #
            # `_mutant_span`, and the mutant is required to die on the arm that
            # tests what it removed: L-18's fixture (nonce as text only). Three
            # mutants in this arc were first killed by a neighbouring check, so
            # the arm asserts BOTH that live source refuses AND that the mutant
            # accepts THIS fixture.
            m8 = _mutant_span(
                "    valid, rejects = 0, []",
                "    result.update(status=PROBE_OK, reason=None)",
                "lg_m8",
                replacement=(
                    "    wait_ids = set()\n"
                    "    for line in wait_lines:\n"
                    "        wait_ids.add(_parse_sentinel(line).get('worker_id'))\n"
                    "    if wait_ids != {target['worker_id']}:\n"
                    "        result.update(status=PROBE_OK,\n"
                    "                      reason='release_wait_identity_mismatch')\n"
                    "        return result\n"))
            textonly_p = f.spawn(25)
            f.log("stubhost:gpu25", 25, textonly_p.pid,
                  wait_nonce="some-other-run")
            real8 = probe(f, "stubhost:gpu25", 25)
            mut8 = _probe_with(m8, "stubhost:gpu25", 25)
            check("M-8  semantic WAIT correlation removed -> DETECTED",
                  LG.evaluate([real8])[0] is False
                  and "release_wait_nonce_mismatch" in (real8["reason"] or "")
                  and m8.evaluate([mut8])[0] is True,
                  f"live REFUSES on the nonce; mutant accepts "
                  f"(reason={mut8['reason']!r}) — killed on L-18's fixture, the "
                  f"arm that tests what it removed")
        finally:
            f.close()


# ═══════════════════════════════════════════════════════════════════════════
# PART F — harness integrity
# ═══════════════════════════════════════════════════════════════════════════

def part_f():
    section("PART F — HARNESS INTEGRITY")
    check("HI-1  the suite tests the LIVE gate module",
          Path(LG.__file__).resolve()
          == (REPO / "scripts" / "gate12_worker_liveness_gate.py").resolve(),
          LG.__file__)
    check("HI-2  it is joined to the LIVE sentinel gate, not a copy",
          Path(SG.__file__).resolve()
          == (REPO / "scripts" / "gate12_sentinel_gate.py").resolve(),
          SG.__file__)
    # No stub worker may have survived this suite: a leaked process would make a
    # later run's /proc reads answer about the wrong pid.
    #
    # SCOPED TO THE PIDS THIS SUITE SPAWNED, not to a pattern. A pattern match
    # also catches the D6 integration suite's stubs, which made this arm's
    # verdict depend on what ran before it — it is how that suite's orphan leak
    # was found, and the fix belongs there (it is now gated there as HI-2b), not
    # in a wider pattern here.
    leaked = [p for p in _SPAWNED if _alive(p)]
    # DESCENDANTS TOO. The stub worker is `sh` plus a forked `sleep`, so a check
    # scoped to the spawned PIDs alone passed while ~160 orphaned children
    # survived each run. The group read catches them.
    orphans = sorted(group_members(set(_SPAWNED)))
    check("HI-3  no process this suite spawned — or forked — outlived it",
          not leaked and not orphans,
          f"spawned={len(_SPAWNED)} leaked={leaked} orphaned_children={orphans}")
    # HI-5 — THE FIXTURE IS CHECKED AGAINST A REAL PRODUCTION RECORD.
    # Every arm in this suite reads sentinel lines this suite itself wrote, so
    # if the writer encodes the implementer's idea of the format rather than the
    # emitter's, the whole battery is green against a shape production never
    # produces. That is the five-defect pattern (Beta §2.30) in its purest form,
    # so it is measured against the 24 rig logs collected from D6 dry run #2 —
    # produced by the real `_emit_session_event` on real rigs, before this gate
    # existed. If the bundle is absent this is UNAVAILABLE; it is never a pass.
    real = Path.home() / "d6_dryrun2_riglogs_20260814" / "192.168.3.122" / "gpu0.log"
    if not real.exists():
        unavailable("HI-5  fixture record shape matches a REAL rig sentinel",
                    f"collected bundle not present at {real}")
    else:
        line = next((ln for ln in real.read_text(errors="replace").splitlines()
                     if "SESSION_SENTINEL" in ln), None)
        if line is None:
            unavailable("HI-5  fixture record shape matches a REAL rig sentinel",
                        "no SESSION_SENTINEL line in the collected log")
        else:
            prod = LG._parse_sentinel(line)
            with tempfile.TemporaryDirectory() as td:
                f = Fleet(td)
                try:
                    p = f.log("rrig6600:gpu0", 0, 4657, nonce="n")
                    mine = LG._parse_sentinel(
                        next(ln for ln in p.read_text().splitlines()
                             if "SESSION_SENTINEL" in ln))
                finally:
                    f.close()
            missing = sorted(set(prod) - set(mine))
            # The fields `classify()` actually reads must be present in the real
            # record with the types it assumes.
            used = {"pid": int, "gpu_id": int, "worker_id": str,
                    "run_nonce": str}
            typed = all(k in prod and isinstance(prod[k], t)
                        for k, t in used.items())
            check("HI-5  fixture record shape matches a REAL rig sentinel",
                  not missing and typed,
                  f"real keys={len(prod)} fixture keys={len(mine)} "
                  f"missing_from_fixture={missing} "
                  f"real pid={prod.get('pid')!r} gpu_id={prod.get('gpu_id')!r}")

            # [R1] AND THE SAME FOR THE WAIT RECORD, which R1 made an acceptance
            # input. A parser proven only against a record this suite invented
            # is the same blind spot one field further along, and the rig logs
            # carry real SESSION_RELEASE_WAIT records emitted by the production
            # emitter before this gate existed.
            wline = next((ln for ln in real.read_text(errors="replace").splitlines()
                          if "SESSION_RELEASE_WAIT" in ln), None)
            if wline is None:
                unavailable("HI-6  wait-record shape matches a REAL rig record",
                            "no SESSION_RELEASE_WAIT line in the collected log")
            else:
                pw = LG._parse_sentinel(wline)
                check("HI-6  wait-record shape matches a REAL rig record",
                      isinstance(pw.get("worker_id"), str)
                      and isinstance(pw.get("run_nonce"), str)
                      and isinstance(pw.get("release_path"), str)
                      and pw.get("event") == "SESSION_RELEASE_WAIT"
                      and pw["run_nonce"] in pw["release_path"],
                      f"real worker_id={pw.get('worker_id')!r} "
                      f"release_path={pw.get('release_path')!r}")

    # HI-7 — THE INVARIANT R2'S PATH COMPARISON DEPENDS ON, MEASURED IN THE
    # WORKER'S OWN SOURCE.
    #
    # R2 compares the WAIT record's `release_path` against the live process's
    # `--session-release-file` argv value by EXACT EQUALITY. That is only a valid
    # test if the worker emits the argv string verbatim — if it ever normalised,
    # absolutised or defaulted the path, this gate would refuse every healthy
    # worker on the fleet, and the first place anyone would find out is D6 #3.
    #
    # So it is read, not assumed: `main()` must pass `args.session_release_file`
    # straight into `await_session_release`, which must emit that same parameter
    # as `release_path`. Read-only — the worker is not in R2's scope and is not
    # touched.
    #
    # RED WHEN: a normalisation step is introduced anywhere on that path.
    import ast as _ast
    wsrc = (REPO / "miner" / "range_miner_worker.py").read_text()
    wtree = _ast.parse(wsrc)
    emits_param = passes_argv = False
    for node in _ast.walk(wtree):
        if not isinstance(node, _ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "attr", getattr(fn, "id", None))
        if (name == "_emit_session_event" and node.args
                and isinstance(node.args[0], _ast.Constant)
                and node.args[0].value == "SESSION_RELEASE_WAIT"):
            emits_param = any(
                kw.arg == "release_path" and isinstance(kw.value, _ast.Name)
                and kw.value.id == "release_path" for kw in node.keywords)
        if name == "await_session_release" and node.args:
            a0 = node.args[0]
            passes_argv = (isinstance(a0, _ast.Attribute)
                           and a0.attr == "session_release_file"
                           and getattr(a0.value, "id", None) == "args")
    check("HI-7  the worker emits the argv barrier path VERBATIM "
          "(the invariant R2's exact comparison rests on)",
          emits_param and passes_argv,
          f"emits_parameter={emits_param} main_passes_argv={passes_argv}")

    now = _porcelain()
    appeared = [ln for ln in now if ln not in _PORCELAIN_AT_START]
    vanished = [ln for ln in _PORCELAIN_AT_START if ln not in now]
    check("HI-4  the suite changed NOTHING in the working tree",
          not appeared and not vanished,
          f"appeared={appeared} vanished={vanished}")


def main():
    print("=" * 78)
    print("S172 D6-I2 — SENTINEL-CORRELATED WORKER LIVENESS GATE")
    print("=" * 78)
    part_a()
    part_b()
    part_c()
    part_d()
    part_e()
    part_f()

    total = len(_RESULTS)
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    unav = sum(1 for _, _, u in _RESULTS if u)
    section("RESULT")
    for name, ok, u in _RESULTS:
        if not ok:
            print(f"  {'UNAVAILABLE' if u else 'FAILED'}: {name}")
    print(f"\n  {passed}/{total} checks green"
          + (f"  ({unav} UNAVAILABLE)" if unav else ""))
    if passed == total:
        print("COMPLETION SENTINEL: PASS — D6-I2 liveness gate battery is green")
        return 0
    print("COMPLETION SENTINEL: FAIL — see the list above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
