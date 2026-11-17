#!/usr/bin/env python3
"""rig_seam_probe.py — REDUCED single-GPU phase-3 -> phase-4 hand-over probe (one attempt).

Brief: docs/CLAUDE_CODE_INSTRUCTIONS_RIG_GPU_FAULT_20261115.md, T4.1 (as revised through
X-RIG-GPU-FAULT R3 and the Alpha rulings after it). Reviewed in thread X-RIG-GPU-PROBE.

WHAT IT IS
  One fresh process = one ATTEMPT. It is started by rig_seam_controller.py (VM101) on a CT,
  with ROCR_VISIBLE_DEVICES=N exactly as production launches a miner worker
  (scripts/launch_fleet_manual.sh:314-323), and is driven entirely by the controller over a
  TCP control channel: it does nothing GPU-related until told to.
  It is a REDUCED probe: it omits production phases 1/2, the coordinator, the network
  result path and staging. Its non-reproduction means non-reproduction only.

DECLARED HISTORY (first seam)
  startup  : import + detect_gpu + warm_gpu, as RangeMinerWorker does (range_miner_worker.py:1992-1993)
  warm     : phase-3 (java_lcg_hybrid) sub-stripes through the production SieveExecutor
  idle     : sleep, process alive, reverse-hybrid NOT loaded
  seam     : mode "handover" -> phase-4 (java_lcg_hybrid_reverse): the FULL production stripe
             (every sub-stripe, production size, real survivor volume): first load of the reverse
             module + first launch;  mode "resume" -> a full phase-3 stripe again (g2n)
  The phase-3 and phase-4 work is the REAL production stripe at real size: the plan carries
  the run's own stripe range partitioned by production VramCaps (2^26 seeds = 68 x 1,000,000 for
  java_lcg_hybrid*), executed by production SieveExecutor. Nothing is shrunk.

TORCH ARMS (the only difference between P, M and N)
  P : production cleanup, byte-for-byte the logic of range_miner_worker._best_effort_gpu_cleanup
      (import torch; torch.cuda.is_available(); synchronize(); empty_cache(); both CuPy pool
      frees). The probe refuses to run P if the production function's source hash changed.
  M : at the same point, `import torch` (library mapped) but NO torch call; cleanup = gc +
      both CuPy pool frees.
  N : torch is never imported (an import blocker enforces it); cleanup = gc + both CuPy frees.
  P1: per sub-stripe exactly as M (torch mapped, gc + both CuPy frees, no torch call); torch's
      is_available + synchronize + empty_cache run ONCE at the end of each phase. P vs P1 isolates
      HOW OFTEN torch's cleanup runs inside the long-lived process; P1 vs M isolates whether it
      runs at all; M vs N isolates mapping alone.
  P vs M differs by torch's calls (its initialisation + sync + empty_cache, a bundle);
  M vs N differs by torch's mapping alone.

TRACE (what is and is not traced — Beta X-RIG-GPU-PROBE R1 Q6/Q3)
  Traced with ENTRY/EXIT records (per-process monotonic sequence ID, wall time us, pid, host, BDF,
  op, status): import cupy, detect_gpu, the BDF check, warm_gpu, each `execute` (one sub-stripe:
  allocation, H2D copies, kernel lookup, launch, D2H `.get()` and cleanup all INSIDE it), the
  `_get_kernel` lookup (RawKernel construction/cache — completion does NOT prove the device
  module load completed), the kernel launch enqueue, and every cleanup sub-step (gc, each torch
  call, each pool free). Allocation, H2D and D2H are NOT separately traced in the real backend:
  an E1 inside an `execute` interval is reported as "inside execute", operation UNRESOLVED. Records go to the controller by
  UDP as they happen AND to a local JSONL copy (default /dev/shm/rig_seam_probe_<pid>.jsonl).
  A 1 Hz heartbeat record is sent in every state, including idle. Runtime-identity snapshots
  (mapped libamdhip64 / libhsa-runtime64 / libhsakmt paths + open /dev/kfd and /dev/dri fds)
  are taken around the imports and device bring-up, around EACH torch call on P's first cleanup
  and P1's first phase end, and at the end. A device handle that appears between two snapshots is
  associated with the call interval between them; this does NOT identify which runtime copy opened
  it (Beta R1 Q3) — runtime attribution stays UNOBSERVED unless stronger evidence exists.

BACKENDS
  --backend real : production code (needs the repo, CuPy and a visible GPU) — owner runs only.
  --backend fake : no GPU, no cupy, no torch; simulated operations with the same trace, for the
                   off-GPU gates (tests/test_rig_seam_probe_controller.py).
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import socket
import sys
import threading
import time

PROBE_VERSION = 1
# sha256 of inspect.getsource(miner.range_miner_worker._best_effort_gpu_cleanup) at the base
# commit 3579088. Arm P is only meaningful if production still has exactly this logic.
PRODUCTION_CLEANUP_SHA256 = None  # filled at import by _production_cleanup_sha() in real mode
EXPECTED_CLEANUP_SHA256_FILE = "rig_seam_probe.cleanup.sha256"

TORCH_MODES = ("P", "P1", "M", "N")
# Off-GPU gate hooks (fake backend only; ignored by the real backend).
_FAKE_DROP = {int(x) for x in os.environ.get("RSP_FAKE_DROP_SEQ", "").split(",") if x.strip()}


def _fake_nc(body):
    """Append one netconsole-listener-format line to RSP_FAKE_NC_FILE (fake backend only)."""
    path = os.environ.get("RSP_FAKE_NC_FILE")
    if not path:
        return
    host = os.environ.get("RSP_FAKE_NC_HOST", "10.0.0.1")
    t = time.time()
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1e6):06d}"
    with open(path, "a") as f:
        f.write(f"{stamp} {host}: [ 1.000000] {body}\n")
SEAM_MODES = ("handover", "resume")


# ------------------------------------------------------------------------------------------
# Trace
# ------------------------------------------------------------------------------------------
class Trace:
    def __init__(self, udp_addr, local_path, pid, bdf, arm):
        self.seq = 0
        self.pid, self.bdf, self.arm = pid, bdf, arm
        self.host = socket.gethostname()
        self._lock = threading.Lock()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if udp_addr else None
        self._addr = udp_addr
        self._f = open(local_path, "a", buffering=1) if local_path else None

    def emit(self, kind, op=None, **kw):
        with self._lock:
            self.seq += 1
            rec = {"v": PROBE_VERSION, "seq": self.seq, "ts_us": int(time.time() * 1e6),
                   "pid": self.pid, "host": self.host, "bdf": self.bdf, "arm": self.arm,
                   "kind": kind}
            if op is not None:
                rec["op"] = op
            rec.update(kw)
            line = json.dumps(rec, separators=(",", ":"))
            if self._f:
                self._f.write(line + "\n")
            if self.seq in _FAKE_DROP:          # off-GPU gate hook: simulate a lost datagram
                return rec
            if self._sock:
                try:
                    self._sock.sendto(line.encode(), self._addr)
                except OSError:
                    pass
            return rec

    def op(self, name, **kw):
        return _Op(self, name, kw)

    def close(self):
        if self._f:
            self._f.close()


class _Op:
    def __init__(self, tr, name, kw):
        self.tr, self.name, self.kw = tr, name, kw

    def __enter__(self):
        self.tr.emit("entry", self.name, **self.kw)
        return self

    def __exit__(self, et, ev, tb):
        self.tr.emit("exit", self.name, status="ok" if et is None else "error",
                     error=None if et is None else f"{et.__name__}: {ev}"[:500])
        return False


def runtime_identity():
    """Mapped HIP/HSA/thunk libraries and open GPU device fds of THIS process (read-only)."""
    libs = set()
    try:
        with open("/proc/self/maps") as f:
            for ln in f:
                p = ln.split()[-1] if ln.strip() else ""
                if any(k in p for k in ("libamdhip64", "libhsa-runtime64", "libhsakmt")):
                    libs.add(os.path.realpath(p))
    except OSError:
        pass
    fds = []
    try:
        for fd in os.listdir("/proc/self/fd"):
            try:
                t = os.readlink(f"/proc/self/fd/{fd}")
            except OSError:
                continue
            if t.startswith("/dev/kfd") or t.startswith("/dev/dri"):
                fds.append(f"{fd}->{t}")
    except OSError:
        pass
    origin = {p: ("torch_wheel" if "site-packages/torch" in p else
                  "system_rocm" if p.startswith("/opt/rocm") else "other") for p in libs}
    return {"libs": sorted(libs), "origin": origin, "gpu_fds": sorted(fds),
            "kfd_fd_count": sum(1 for x in fds if x.endswith("/dev/kfd"))}


def snap(tr, where):
    tr.emit("identity", where=where, **runtime_identity())


# ------------------------------------------------------------------------------------------
# Torch arms
# ------------------------------------------------------------------------------------------
class _TorchBlocker:
    """Meta-path finder that makes `import torch` fail (arm N)."""
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("rig_seam_probe arm N: torch import blocked")
        return None


def install_torch_blocker():
    sys.meta_path.insert(0, _TorchBlocker())


def _torch_blocked():
    """True iff an import of torch would be refused. Uses find_spec, which consults the meta-path
    finders WITHOUT importing or executing anything."""
    import importlib.util
    try:
        importlib.util.find_spec("torch")
    except ImportError:
        return True
    return False


def _real_torch():
    import torch
    return torch


def make_cleanup(mode, tr, cp_mod, first_flag, torch_import=_real_torch):
    """Return the cleanup callable for arm `mode`, traced per sub-operation.

    P mirrors range_miner_worker._best_effort_gpu_cleanup exactly (each step guarded)."""
    def _gc():
        with tr.op("cleanup_gc"):
            import gc
            gc.collect()

    def _cupy_frees():
        try:
            with tr.op("cupy_free_default_pool"):
                cp_mod.get_default_memory_pool().free_all_blocks()
            with tr.op("cupy_free_pinned_pool"):
                cp_mod.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    def cleanup_P():
        try:
            _gc()
        except Exception:
            pass
        try:
            first = not first_flag["torch_imported"]
            if first:
                snap(tr, "before_import_torch")
            with tr.op("import_torch"):
                torch = torch_import()
            first_flag["torch_imported"] = True
            if first:
                snap(tr, "after_import_torch")
            with tr.op("torch_is_available"):
                avail = torch.cuda.is_available()
            if first:
                snap(tr, "after_torch_is_available")
            if avail:
                with tr.op("torch_synchronize"):
                    torch.cuda.synchronize()
                if first:
                    snap(tr, "after_torch_synchronize")
                with tr.op("torch_empty_cache"):
                    torch.cuda.empty_cache()
                if first:
                    snap(tr, "after_torch_empty_cache")
        except Exception:
            pass
        _cupy_frees()

    def cleanup_M():
        try:
            _gc()
        except Exception:
            pass
        try:
            first = not first_flag["torch_imported"]
            if first:
                snap(tr, "before_import_torch")
            with tr.op("import_torch"):
                torch_import()           # imported, never called
            first_flag["torch_imported"] = True
            if first:
                snap(tr, "after_import_torch")
        except Exception:
            pass
        _cupy_frees()

    def cleanup_N():
        try:
            _gc()
        except Exception:
            pass
        _cupy_frees()

    return {"P": cleanup_P, "P1": cleanup_M, "M": cleanup_M, "N": cleanup_N}[mode]


def make_phase_end(mode, tr, torch_import=_real_torch):
    """Arm P1 only: torch's cleanup once, at the end of a phase. Other arms: no-op."""
    if mode != "P1":
        return lambda: None

    calls = {"n": 0}

    def torch_once():
        calls["n"] += 1
        first = calls["n"] == 1
        try:
            with tr.op("import_torch"):
                torch = torch_import()
            if first:
                snap(tr, "p1_before_torch_is_available")
            with tr.op("torch_is_available"):
                avail = torch.cuda.is_available()
            if first:
                snap(tr, "p1_after_torch_is_available")
            if avail:
                with tr.op("torch_synchronize"):
                    torch.cuda.synchronize()
                if first:
                    snap(tr, "p1_after_torch_synchronize")
                with tr.op("torch_empty_cache"):
                    torch.cuda.empty_cache()
            snap(tr, "after_phase_end_torch")
        except Exception:
            pass
    return torch_once


# ------------------------------------------------------------------------------------------
# Backends
# ------------------------------------------------------------------------------------------
class FakeBackend:
    """No GPU. Simulates the same operation sequence. Behaviour knobs come from the
    environment so the off-GPU gates can force a hang, a slow op or an error."""
    def __init__(self, tr, mode):
        self.tr, self.mode = tr, mode
        self.stub_torch = _StubTorch(tr)
        self.cleanup = make_cleanup(mode, tr, _StubCupy(), {"torch_imported": False},
                                    torch_import=self.stub_torch.importer)
        self.phase_end = make_phase_end(mode, tr, torch_import=self.stub_torch.importer)
        self.sub_s = float(os.environ.get("RSP_FAKE_SUBSTRIPE_S", "0.05"))
        self.hang_at = os.environ.get("RSP_FAKE_HANG_AT", "")   # e.g. "seam:launch"
        self.loaded = set()
        self._phase = None

    def bring_up(self, expected_bdf):
        with self.tr.op("detect_gpu"):
            pass
        with self.tr.op("bdf_check", expected=expected_bdf, observed=expected_bdf):
            pass
        with self.tr.op("warm_gpu"):
            pass
        snap(self.tr, "after_bring_up")
        return expected_bdf

    def run_substripe(self, family, assign, start, count, phase_tag):
        if family not in self.loaded:
            with self.tr.op("module_load", family=family):
                self._maybe_hang(f"{phase_tag}:module_load")
            self.loaded.add(family)
        with self.tr.op("launch", family=family, seed_start=start, seed_count=count):
            self._maybe_hang(f"{phase_tag}:launch")
            time.sleep(self.sub_s)
        with self.tr.op("d2h_get"):
            pass
        self.cleanup()   # production calls cleanup in execute()'s finally (:1331)
        return 0

    def _maybe_hang(self, where):
        if os.environ.get("RSP_FAKE_NC_E1_AT") == where:
            _fake_nc(f"amdgpu {self.tr.bdf}: amdgpu: GCVM_L2_PROTECTION_FAULT_STATUS:0x00801231")
        if self.hang_at and self.hang_at == where:
            self.tr.emit("note", msg=f"fake hang at {where}")
            while True:
                time.sleep(3600)


class RealBackend:
    """Production code path. Owner-run only. Imports nothing GPU-related until bring_up."""
    def __init__(self, tr, mode, repo):
        self.tr, self.mode, self.repo = tr, mode, repo
        sys.path.insert(0, repo)
        self.cp = None
        self.rmw = None
        self.executor = None
        self.first_flag = {"torch_imported": False}

    def bring_up(self, expected_bdf):
        snap(self.tr, "before_import_cupy")
        with self.tr.op("import_cupy"):
            import cupy as cp
        self.cp = cp
        snap(self.tr, "after_import_cupy")
        with self.tr.op("import_range_miner_worker"):
            import miner.range_miner_worker as rmw
        self.rmw = rmw
        if self.mode == "P":
            got = hashlib.sha256(inspect.getsource(rmw._best_effort_gpu_cleanup).encode()).hexdigest()
            want = _expected_cleanup_sha()
            # FAIL CLOSED (Beta X-RIG-GPU-PROBE R1 Q6): an absent, empty or malformed pin refuses P.
            if not want or len(want) != 64 or any(ch not in "0123456789abcdef" for ch in want):
                raise SystemExit(f"arm P refused: cleanup pin missing or malformed ({want!r})")
            if got != want:
                raise SystemExit(f"arm P refused: production cleanup changed ({got} != {want})")
            self.tr.emit("note", msg="production cleanup sha", sha256=got, expected=want)
        # Install the arm's cleanup at the module global the executor calls (:1331).
        rmw._best_effort_gpu_cleanup = make_cleanup(self.mode, self.tr, cp, self.first_flag)
        self.phase_end = make_phase_end(self.mode, self.tr)
        with self.tr.op("detect_gpu"):
            info = rmw.detect_gpu(0)
        observed = _visible_bdf(cp)
        with self.tr.op("bdf_check", expected=expected_bdf, observed=observed,
                        gpu_name=getattr(info, "gpu_name", None)):
            if observed != expected_bdf:
                raise SystemExit(f"BDF mismatch: visible {observed} != planned {expected_bdf}")
        snap(self.tr, "after_detect_gpu")
        with self.tr.op("warm_gpu"):
            rmw.warm_gpu(0)
        snap(self.tr, "after_bring_up")
        self.executor = rmw.SieveExecutor(resolver=rmw.ResidueResolver(), device_index=0)
        self._wrap_launch_and_load()
        return observed

    def _wrap_launch_and_load(self):
        tr, ex = self.tr, self.executor
        orig_launch = ex._gpu_launch

        def traced_launch(kernel, blocks, threads, kernel_args):
            sig = [type(a).__name__ for a in kernel_args]
            with tr.op("launch", blocks=blocks, threads=threads, nargs=len(kernel_args), argtypes=sig):
                return orig_launch(kernel, blocks, threads, kernel_args)
        ex._gpu_launch = traced_launch
        import sieve_gpu_worker as sgw
        orig_get = sgw._get_kernel

        def traced_get_kernel(family, *a, **k):
            with tr.op("module_load_or_cache", family=family):
                return orig_get(family, *a, **k)
        sgw._get_kernel = traced_get_kernel

    def run_substripe(self, family, assign, start, count, phase_tag):
        with self.tr.op("execute", family=family, seed_start=start, seed_count=count):
            out = self.executor.execute(assign, start, count)
        return int(getattr(out, "count", 0))


def _visible_bdf(cp):
    props = cp.cuda.runtime.getDeviceProperties(0)
    dom = int(props.get("pciDomainID", 0))
    return f"{dom:04x}:{int(props['pciBusID']):02x}:{int(props['pciDeviceID']):02x}.0"


def _expected_cleanup_sha():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), EXPECTED_CLEANUP_SHA256_FILE)
    try:
        return open(p).read().split()[0]
    except OSError:
        return None


# ------------------------------------------------------------------------------------------
# Control loop
# ------------------------------------------------------------------------------------------
def _send(sock, obj):
    sock.sendall((json.dumps(obj, separators=(",", ":")) + "\n").encode())


def _recv_lines(sock):
    buf = b""
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            return
        buf += chunk
        while b"\n" in buf:
            ln, buf = buf.split(b"\n", 1)
            if ln.strip():
                yield json.loads(ln)


def main(argv=None):
    ap = argparse.ArgumentParser(description="reduced phase-3/4 seam probe (one attempt)")
    ap.add_argument("--controller", required=True, help="host:port of rig_seam_controller")
    ap.add_argument("--bdf", required=True, help="planned PCI BDF of the visible device")
    ap.add_argument("--torch-mode", choices=TORCH_MODES, required=True)
    ap.add_argument("--attempt-id", required=True)
    ap.add_argument("--backend", choices=("real", "fake"), default="real")
    ap.add_argument("--repo", default=os.path.expanduser("~/distributed_prng_analysis"))
    ap.add_argument("--local-trace", default=None,
                    help="local JSONL copy (default /dev/shm/rig_seam_probe_<pid>.jsonl; '' = none)")
    a = ap.parse_args(argv)

    if a.torch_mode == "N":
        install_torch_blocker()
    host, port = a.controller.rsplit(":", 1)
    port = int(port)
    local = a.local_trace if a.local_trace is not None else f"/dev/shm/rig_seam_probe_{os.getpid()}.jsonl"
    udp = None if (a.backend == "fake" and os.environ.get("RSP_FAKE_NO_UDP") == "1") else (host, port)
    tr = Trace(udp, local or None, os.getpid(), a.bdf, f"{a.torch_mode}")
    tr.emit("torch_guard", torch_mode=a.torch_mode, torch_blocked=_torch_blocked())
    if a.torch_mode == "N" and not _torch_blocked():
        tr.emit("refused", reason="arm N requires the torch import blocker")
        tr.emit("end", rc=3)
        return 3
    tr.emit("start", attempt_id=a.attempt_id, backend=a.backend, torch_mode=a.torch_mode,
            env={k: os.environ.get(k) for k in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                                                 "CUPY_CACHE_DIR", "AMD_LOG_LEVEL", "AMD_LOG_MASK",
                                                 "HSAKMT_DEBUG_LEVEL", "AMD_SERIALIZE_KERNEL",
                                                 "AMD_SERIALIZE_COPY", "HIP_LAUNCH_BLOCKING")})
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(1.0):
            tr.emit("hb")
    threading.Thread(target=heartbeat, daemon=True).start()

    sock = socket.create_connection((host, port), timeout=30)
    sock.settimeout(None)
    _send(sock, {"t": "hello", "pid": os.getpid(), "host": socket.gethostname(), "bdf": a.bdf,
                 "attempt_id": a.attempt_id,
                 "torch_mode": a.torch_mode, "backend": a.backend})
    be = FakeBackend(tr, a.torch_mode) if a.backend == "fake" else RealBackend(tr, a.torch_mode, a.repo)
    plan = None
    rc = 0
    try:
        for msg in _recv_lines(sock):
            cmd = msg.get("cmd")
            tr.emit("cmd", cmd=cmd)
            if cmd == "plan":
                plan = msg
                _send(sock, {"t": "ack", "cmd": "plan"})
            elif cmd == "bring_up":
                with tr.op("bring_up"):
                    be.bring_up(a.bdf)
                _send(sock, {"t": "done", "cmd": "bring_up"})
            elif cmd in ("warm", "seam"):
                mode = msg.get("mode", "warm")
                if cmd == "warm" or mode == "resume":
                    fam, key, tag = plan["family3"], "assign3", ("warm" if cmd == "warm" else "seam")
                else:
                    fam, key, tag = plan["family4"], "assign4", "seam"
                # The FULL production stripe: every sub-stripe range in the plan, at production
                # size (the plan is built from the run's ledger stripe and VramCaps). No timer.
                ranges = plan["ranges3" if key == "assign3" else "ranges4"]
                tr.emit("phase_begin", cmd=cmd, mode=mode, family=fam, substripes_planned=len(ranges))
                n = 0
                total = 0
                assign = _assign_obj(plan[key], a.backend)
                for s0, c0 in ranges:
                    total += be.run_substripe(fam, assign, s0, c0, tag)
                    n += 1
                be.phase_end()
                tr.emit("phase_end", cmd=cmd, mode=mode, substripes=n, survivors=total)
                _send(sock, {"t": "done", "cmd": cmd, "substripes": n, "survivors": total})
            elif cmd == "idle":
                tr.emit("idle_begin", seconds=msg.get("seconds"))
                time.sleep(float(msg.get("seconds", 0)))
                tr.emit("idle_end")
                _send(sock, {"t": "done", "cmd": "idle"})
            elif cmd in ("exit", "stop"):
                _send(sock, {"t": "bye"})
                break
    except SystemExit as e:
        tr.emit("refused", reason=str(e))
        rc = 3
    except Exception as e:  # report and exit; never retry
        tr.emit("error", error=f"{type(e).__name__}: {e}"[:1000])
        rc = 4
    finally:
        snap(tr, "final")
        tr.emit("end", rc=rc)
        if a.backend == "fake" and os.environ.get("RSP_FAKE_NC_POSTPROOF") == "1":
            _fake_nc(f"NCPROOF-fake-{a.attempt_id}")
        stop.set()
        tr.close()
        try:
            sock.close()
        except OSError:
            pass
    return rc


class _StubTorch:
    """Fake-mode torch: records calls, touches no device. Never the real torch."""
    def __init__(self, tr):
        self.tr = tr
        outer = self

        class _Cuda:
            def is_available(self_inner):
                outer.tr.emit("note", msg="stub torch.cuda.is_available")
                return True

            def synchronize(self_inner):
                outer.tr.emit("note", msg="stub torch.cuda.synchronize")

            def empty_cache(self_inner):
                outer.tr.emit("note", msg="stub torch.cuda.empty_cache")
        self.cuda = _Cuda()

    def importer(self):
        if any(isinstance(f, _TorchBlocker) for f in sys.meta_path):
            raise ImportError("rig_seam_probe arm N: torch import blocked")
        self.tr.emit("note", msg="stub torch imported")
        return self


class _StubCupy:
    class _Pool:
        def free_all_blocks(self):
            return None

    def get_default_memory_pool(self):
        return self._Pool()

    def get_default_pinned_memory_pool(self):
        return self._Pool()


def _assign_obj(d, backend):
    if backend == "fake":
        return d
    from miner.range_miner_protocol import from_dict
    return from_dict(d)


if __name__ == "__main__":
    sys.exit(main())
