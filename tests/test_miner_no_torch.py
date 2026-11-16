#!/usr/bin/env python3
"""X-MINER-NO-TORCH — CPU-only gates and mutants for `--cleanup-torch` in miner/range_miner_worker.py.

NO GPU. Every gate runs in a FRESH interpreter with CUDA_VISIBLE_DEVICES, HIP_VISIBLE_DEVICES and
ROCR_VISIBLE_DEVICES empty, and with STUB `cupy` and STUB `torch` packages first on sys.path:
  * stub cupy is numpy-backed (arrays with .get()); its pool `free_all_blocks()` ZEROES every live
    stub array, so a cleanup that ran before the results were read back would corrupt them;
  * stub torch records every call and touches nothing; in arm "off" the worker's blocker must stop
    it being imported at all.

Gates
  G-DEFAULT        default mode is per_substripe; `_best_effort_gpu_cleanup` is byte-unchanged
                   (sha pinned); the call site dispatches to it; the launcher passes no flag by default
  G-OFF-NO-TORCH   mode off: cleanup never imports torch, calls CuPy deviceSynchronize + both frees,
                   `import torch` raises; mode per_substripe still calls torch (stub)
  G-OFF-PREIMPORT  mode off fails closed if torch is already imported
  G-SELFCHECK      mode off fails closed on a second libamdhip64 / libhsa-runtime64 / libcudart or a
                   torch-wheel library; per_substripe only logs; the log line names mode and runtimes
  G-IDENTITY       SieveExecutor.execute (REAL code) on stub cupy with a deterministic fake launch:
                   survivors are byte-identical (JSON) between per_substripe and off, and equal an
                   independently computed expectation, for java_lcg_hybrid_reverse and java_lcg
  G-LAUNCHER       launch_fleet_manual.sh: default command lines carry no --cleanup-torch; the rig and
                   Zeus lines carry RIG_/ZEUS_CLEANUP_TORCH; bad values exit 64

Run:  python3 -B -m pytest -q -p no:cacheprovider tests/test_miner_no_torch.py
      python3 -B tests/test_miner_no_torch.py gate <NAME> --root <dir>
      python3 -B tests/test_miner_no_torch.py mutants
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PIN = "d756acc37805fac3d0c90f5e4d8901fe2f7bd751e3d63db4faabaac0e76d4a4d"   # production cleanup @3579088

STUB_CUPY = r'''
import numpy as _np, weakref as _wr
_LIVE = []                    # weak references to every live stub array
_CALLS = []
_REG = []                     # creation order of arrays inside one execute()
class ndarray(_np.ndarray):
    def get(self):
        return _np.asarray(self).copy()
    def __getitem__(self, i):            # CuPy returns a 0-d device array for a scalar index
        r = super().__getitem__(i)
        return r if isinstance(r, ndarray) else _np.asarray(r).view(ndarray)
def _wrap(a):
    v = _np.asarray(a).view(ndarray); _LIVE.append(_wr.ref(v)); _REG.append(v); return v
uint8, uint32, uint64, int32, float32 = _np.uint8, _np.uint32, _np.uint64, _np.int32, _np.float32
def array(x, dtype=None): return _wrap(_np.array(x, dtype=dtype))
def zeros(n, dtype=None): return _wrap(_np.zeros(n, dtype=dtype))
def arange(a, b, dtype=None): return _wrap(_np.arange(a, b, dtype=dtype))
class _Pool:
    def __init__(self, n): self.n = n
    def free_all_blocks(self):
        _CALLS.append(self.n)
        for r in list(_LIVE):            # freed device memory: every live buffer is now garbage
            v = r()
            if v is None:
                continue
            try: _np.asarray(v)[...] = 0
            except Exception: pass
def get_default_memory_pool(): return _Pool("free_default")
def get_default_pinned_memory_pool(): return _Pool("free_pinned")
class RawKernel:
    def __init__(self, src, name, *a, **k): self.name = name
class _Dev:
    def __init__(self, i=0): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def synchronize(self): _CALLS.append("device_sync")
class _RT:
    is_hip = True
    @staticmethod
    def deviceSynchronize(): _CALLS.append("deviceSynchronize")
class _Stream:
    @staticmethod
    def synchronize(): _CALLS.append("stream_sync")
class _Cuda:
    Device = _Dev
    runtime = _RT()
    class Stream:
        null = _Stream()
cuda = _Cuda()
'''
STUB_TORCH = r'''
import json, os
_LOG = os.environ.get("STUB_TORCH_LOG")
def _rec(m):
    if _LOG:
        with open(_LOG, "a") as f: f.write(m + "\n")
_rec("imported")
class _C:
    def is_available(self): _rec("is_available"); return True
    def synchronize(self): _rec("synchronize")
    def empty_cache(self): _rec("empty_cache")
cuda = _C()
'''


def _stub_dir(work):
    d = Path(work) / "stubs"
    (d / "cupy").mkdir(parents=True, exist_ok=True)
    (d / "cupy" / "__init__.py").write_text(STUB_CUPY)
    (d / "torch").mkdir(exist_ok=True)
    (d / "torch" / "__init__.py").write_text(STUB_TORCH)
    return d


def _run_py(root, work, code, env=None):
    stubs = _stub_dir(work)
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="",
             PYTHONPATH=f"{stubs}:{root}", PYTHONDONTWRITEBYTECODE="1")
    e.update(env or {})
    return subprocess.run([sys.executable, "-B", "-c", code], cwd=str(root), env=e,
                          capture_output=True, text=True, timeout=300)


# ------------------------------------------------------------------------------------------
def gate_default(root, work):
    src = (Path(root) / "miner" / "range_miner_worker.py").read_text()
    tree = ast.parse(src)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_best_effort_gpu_cleanup"][0]
    seg = "".join(src.splitlines(keepends=True)[fn.lineno - 1:fn.end_lineno])
    assert hashlib.sha256(seg.encode()).hexdigest() == PIN, \
        "G-DEFAULT: _best_effort_gpu_cleanup changed — the default path must stay byte-identical"
    p = _run_py(root, work, r'''
import miner.range_miner_worker as w
a = w._parse_args(["--host", "h", "--gpu-id", "0"])
assert a.cleanup_torch == "per_substripe", a.cleanup_torch
w.configure_cleanup_torch(a.cleanup_torch)
called = []
w._best_effort_gpu_cleanup = lambda: called.append("production")
w._gpu_cleanup_dispatch()
assert called == ["production"], called
print("OK")
''')
    assert p.returncode == 0 and "OK" in p.stdout, f"G-DEFAULT: default dispatch wrong: {p.stdout[-500:]} {p.stderr[-800:]}"


def gate_off_no_torch(root, work):
    log = Path(work) / "torch_calls.log"
    p = _run_py(root, work, r'''
import sys, cupy
import miner.range_miner_worker as w
w.configure_cleanup_torch("off")
w._FIRST_CLEANUP_CHECKED = True     # self-check is gated separately
w._gpu_cleanup_dispatch()
assert "torch" not in sys.modules, "torch imported in mode off"
assert cupy._CALLS == ["deviceSynchronize", "free_default", "free_pinned"], cupy._CALLS
try:
    import torch
    raise SystemExit("torch import NOT blocked")
except ImportError:
    pass
print("OK")
''', env={"STUB_TORCH_LOG": str(log)})
    assert p.returncode == 0 and "OK" in p.stdout, f"G-OFF-NO-TORCH: {p.stdout[-500:]} {p.stderr[-800:]}"
    assert not log.exists(), f"G-OFF-NO-TORCH: stub torch was touched in mode off: {log.read_text()}"
    p2 = _run_py(root, Path(work) / "on", r'''
import cupy, miner.range_miner_worker as w
w.configure_cleanup_torch("per_substripe")
w._FIRST_CLEANUP_CHECKED = True
w._gpu_cleanup_dispatch()
print("OK")
''', env={"STUB_TORCH_LOG": str(Path(work) / "torch_on.log")})
    on = (Path(work) / "torch_on.log").read_text().split() if (Path(work) / "torch_on.log").exists() else []
    assert p2.returncode == 0 and on == ["imported", "is_available", "synchronize", "empty_cache"], \
        f"G-OFF-NO-TORCH: per_substripe must keep the torch cleanup: {on} {p2.stderr[-500:]}"


def gate_off_preimport(root, work):
    p = _run_py(root, work, r'''
import torch, miner.range_miner_worker as w
try:
    w.configure_cleanup_torch("off")
    print("NOT REFUSED")
except w.CleanupTorchSelfCheckError:
    print("REFUSED")
''')
    assert "REFUSED" in p.stdout and "NOT" not in p.stdout, f"G-OFF-PREIMPORT: {p.stdout} {p.stderr[-500:]}"


MAPS_ONE = "7f00-7f01 r-xp 0 08:01 1 /opt/rocm-6.4.3/lib/libamdhip64.so.6.4.60403\n" \
           "7f02-7f03 r-xp 0 08:01 2 /opt/rocm-6.4.3/lib/libhsa-runtime64.so.1.15.60403\n"
MAPS_TWO = MAPS_ONE + \
    "7f04-7f05 r-xp 0 08:01 3 /home/m/rocm_env/lib/python3.10/site-packages/torch/lib/libamdhip64.so\n" \
    "7f06-7f07 r-xp 0 08:01 4 /home/m/rocm_env/lib/python3.10/site-packages/torch/lib/libhsa-runtime64.so\n"


def gate_selfcheck(root, work):
    p = _run_py(root, work, r'''
import logging, io, json
import miner.range_miner_worker as w
buf = io.StringIO(); h = logging.StreamHandler(buf); logging.getLogger().addHandler(h); logging.getLogger().setLevel(logging.INFO)
ONE, TWO = %r, %r
w.configure_cleanup_torch("per_substripe")
w.runtime_selfcheck("t_on_two", TWO)                # must only log
w.configure_cleanup_torch("off")
w.runtime_selfcheck("t_off_one", ONE)               # must pass
try:
    w.runtime_selfcheck("t_off_two", TWO); print("NOT REFUSED")
except w.CleanupTorchSelfCheckError as e:
    print("REFUSED")
out = buf.getvalue()
assert "cleanup_torch=off stage=t_off_one" in out and "libamdhip64" in out, out
assert "cleanup_torch=per_substripe stage=t_on_two" in out, out
print("LOGOK")
''' % (MAPS_ONE, MAPS_TWO))
    assert "REFUSED" in p.stdout and "NOT REFUSED" not in p.stdout and "LOGOK" in p.stdout, \
        f"G-SELFCHECK: {p.stdout[-600:]} {p.stderr[-800:]}"


def _write_dataset(path, n=200):
    import random
    rnd = random.Random(1234)
    data = [{"draw": rnd.randrange(1000), "session": "midday" if i % 2 == 0 else "evening"} for i in range(n)]
    Path(path).write_text(json.dumps(data))


IDENTITY_CODE = r'''
import json, sys, numpy as np
import cupy
import miner.range_miner_worker as w
from miner.range_miner_coordinator import RangeMinerCoordinator
from miner.range_miner_protocol import StripeAssignMessage
MODE, DATASET, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
w.configure_cleanup_torch(MODE)
w._FIRST_CLEANUP_CHECKED = True
sessions, ws, anchor = ["midday"], 12, 25
residues = w.load_residue_window(DATASET, ws, sessions, anchor)
import hashlib
dsha = hashlib.sha256(open(DATASET, "rb").read()).hexdigest()
res = {}
for family, phase in (("java_lcg_hybrid_reverse", 4), ("java_lcg", 1)):
    payload = RangeMinerCoordinator.build_stripe_assign_payload(
        None, DATASET, ws, sessions, anchor, residues, dataset_sha256=dsha, phase=phase,
        forward_threshold=0.71, reverse_threshold=0.47, generator_phase=0)
    msg = StripeAssignMessage(stripe_id="s", seed_start=1000, seed_count=5000, family_name=family,
                              phase=phase, payload=payload)
    ex = w.SieveExecutor(resolver=w.ResidueResolver(), device_index=0)
    def fake_launch(kernel, blocks, threads, args, _hyb=w.is_hybrid_family(family)):
        R = cupy._REG
        seeds, surv, rates, cnt = R[1], R[2], R[3], R[4]
        s = np.asarray(seeds).astype(np.uint64)
        sel = s[(s * 2654435761) % 97 < 9]
        c = len(sel)
        np.asarray(surv)[:c] = sel
        np.asarray(rates)[:c] = (0.40 + (sel % 50) / 100.0).astype(np.float32)
        np.asarray(cnt)[0] = c
        if _hyb:
            np.asarray(R[7])[:c] = (sel % 5).astype(np.uint32)
            ss = np.asarray(R[8]).reshape(-1, len(residues))
            for j in range(len(residues)):
                ss[:c, j] = ((sel + j) % 7).astype(np.uint32)
        else:
            np.asarray(R[5])[:c] = (sel % 13).astype(np.uint8)
    ex._gpu_launch = fake_launch
    cupy._REG.clear()
    out = ex.execute(msg, 1000, 5000)
    res[family] = {"survivors": out.survivors, "count": out.count, "eff": out.effective_threshold}
open(OUT, "w").write(json.dumps(res, sort_keys=True))
print("OK")
'''


def _expected():
    import numpy as np
    out = {}
    s = np.arange(1000, 6000, dtype=np.uint64)
    sel = s[(s * 2654435761) % 97 < 9]
    for family, thr, hyb in (("java_lcg_hybrid_reverse", 0.47, True), ("java_lcg", 0.71, False)):
        rows = []
        for v in sel.tolist():
            rate = float(np.float32(0.40 + (v % 50) / 100.0))
            if rate >= thr:
                rows.append([int(v), rate, int(v % 5), [int((v + j) % 7) for j in range(12)]] if hyb
                            else [int(v), rate, None, [int(v % 13)]])
        out[family] = rows
    return out


def gate_identity(root, work):
    Path(work).mkdir(parents=True, exist_ok=True)
    ds = Path(work) / "dataset.json"
    _write_dataset(ds)
    outs = {}
    for mode in ("per_substripe", "off"):
        o = Path(work) / f"out_{mode}.json"
        stubs = _stub_dir(Path(work) / mode)
        e = dict(os.environ, CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="",
                 PYTHONPATH=f"{stubs}:{root}", PYTHONDONTWRITEBYTECODE="1")
        p = subprocess.run([sys.executable, "-B", "-c", IDENTITY_CODE, mode, str(ds), str(o)],
                           cwd=str(root), env=e, capture_output=True, text=True, timeout=300)
        assert p.returncode == 0 and o.exists(), f"G-IDENTITY: execute failed in mode {mode}: {p.stderr[-1500:]}"
        outs[mode] = o.read_bytes()
    assert outs["per_substripe"] == outs["off"], "G-IDENTITY: survivor output differs between modes"
    got = json.loads(outs["off"])
    exp = _expected()
    for fam, rows in exp.items():
        assert got[fam]["survivors"] == rows, f"G-IDENTITY: {fam} survivors != independent expectation"
        assert got[fam]["count"] == len(rows) and len(rows) > 0, f"G-IDENTITY: {fam} count"


def gate_launcher(root, work):
    sh = (Path(root) / "scripts" / "launch_fleet_manual.sh").read_text()
    assert 'RIG_CLEANUP_TORCH="${RIG_CLEANUP_TORCH:-per_substripe}"' in sh, "G-LAUNCHER: rig default"
    assert 'ZEUS_CLEANUP_TORCH="${ZEUS_CLEANUP_TORCH:-per_substripe}"' in sh, "G-LAUNCHER: zeus default"
    assert '$REMOTE_SENTINEL_ARGS $RIG_CT_ARGS' in sh, "G-LAUNCHER: rig worker line lacks RIG_CT_ARGS"
    assert '$LOCAL_SENTINEL_ARGS $ZEUS_CT_ARGS' in sh, "G-LAUNCHER: zeus worker line lacks ZEUS_CT_ARGS"
    # execute just the flag block (no ssh, no launch)
    a = sh.index('RIG_CLEANUP_TORCH="${RIG_CLEANUP_TORCH:-per_substripe}"')
    b = sh.index('ZEUS_CT_ARGS=""; [ "$ZEUS_CLEANUP_TORCH"')
    b = sh.index("\n", b)
    block = sh[a:b] + '\necho "RIG=[$RIG_CT_ARGS] ZEUS=[$ZEUS_CT_ARGS]"\n'
    def run(env):
        return subprocess.run(["bash", "-c", block], env=dict(os.environ, **env), capture_output=True, text=True)
    d = run({})
    assert d.stdout.strip() == "RIG=[] ZEUS=[]", f"G-LAUNCHER: default must pass no flag: {d.stdout}"
    o = run({"RIG_CLEANUP_TORCH": "off"})
    assert o.stdout.strip() == "RIG=[--cleanup-torch off] ZEUS=[]", f"G-LAUNCHER: rig off: {o.stdout}"
    bad = run({"RIG_CLEANUP_TORCH": "of"})
    assert bad.returncode == 64, f"G-LAUNCHER: a bad value must exit 64, got {bad.returncode}"


GATES = {f.__name__[5:]: f for f in (gate_default, gate_off_no_torch, gate_off_preimport,
                                       gate_selfcheck, gate_identity, gate_launcher)}

MUTANTS = [
    ("N1-off-imports-torch", "miner/range_miner_worker.py",
     [('        # Device-wide barrier on CuPy\'s OWN runtime', '        import torch  # mutant\n        # Device-wide barrier on CuPy\'s OWN runtime')],
     "off_no_torch"),
    ("N2-no-blocker", "miner/range_miner_worker.py",
     [("            sys.meta_path.insert(0, _TorchImportBlocker())", "            pass")], "off_no_torch"),
    ("N3-selfcheck-fail-open", "miner/range_miner_worker.py",
     [('            raise CleanupTorchSelfCheckError(f"[NO-TORCH] fail closed at {stage}: " + "; ".join(problems))',
       "            pass")], "selfcheck"),
    ("N4-default-off", "miner/range_miner_worker.py",
     [('choices=CLEANUP_TORCH_MODES, default="per_substripe",', 'choices=CLEANUP_TORCH_MODES, default="off",')],
     "default"),
    ("N5-dispatch-ignores-mode", "miner/range_miner_worker.py",
     [('    if _CLEANUP_TORCH_MODE == "off":\n        _cupy_only_gpu_cleanup()', '    if False:\n        _cupy_only_gpu_cleanup()')],
     "off_no_torch"),
    ("N6-cleanup-before-readback", "miner/range_miner_worker.py",
     [("                self._gpu_launch(kernel, blocks, threads, kernel_args)\n",
       "                self._gpu_launch(kernel, blocks, threads, kernel_args)\n                _gpu_cleanup_dispatch()\n")],
     "identity"),
    ("N7-preimport-not-refused", "miner/range_miner_worker.py",
     [('        if "torch" in sys.modules:\n            raise CleanupTorchSelfCheckError(',
       '        if False:\n            raise CleanupTorchSelfCheckError(')], "off_preimport"),
    ("N8-launcher-drops-rig-flag", "scripts/launch_fleet_manual.sh",
     [("  $REMOTE_SENTINEL_ARGS $RIG_CT_ARGS \\", "  $REMOTE_SENTINEL_ARGS \\")], "launcher"),
    ("N9-duplicate-runtime-not-counted", "miner/range_miner_worker.py",
     [('        for key in ("libamdhip64", "libhsa-runtime64", "libcudart"):\n            if len(ident["libs"][key]) > 1:',
       '        for key in ("libamdhip64", "libhsa-runtime64", "libcudart"):\n            if len(ident["libs"][key]) > 9:'),
      ('        if ident["torch_wheel_libs"]:\n', '        if False:\n')], "selfcheck"),
]


def _copy_root(dst):
    dst = Path(dst)
    for rel in ("miner", "scripts/launch_fleet_manual.sh"):
        s = REPO / rel
        if s.is_dir():
            shutil.copytree(s, dst / rel, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(s, dst / rel)
    for f in ("sieve_gpu_worker.py", "prng_registry.py", "adaptive_thresholds.py", "hybrid_strategy.py"):
        shutil.copy(REPO / f, dst / f)
    for f in REPO.glob("*.py"):           # other top-level modules the miner imports lazily
        if not (dst / f.name).exists():
            shutil.copy(f, dst / f.name)
    return dst


def run_gate_subprocess(name, root, work):
    return subprocess.run([sys.executable, "-B", __file__, "gate", name, "--root", str(root),
                           "--work", str(work)], capture_output=True, text=True, timeout=600)


def run_mutant(mid, rel, pairs, gate):
    with tempfile.TemporaryDirectory(dir=str(REPO / "logs")) as td:
        clean = _copy_root(Path(td) / "clean")
        cp = run_gate_subprocess(gate, clean, Path(td) / "w_clean")
        if cp.returncode != 0:
            return {"mutant": mid, "result": "INVALID", "why": "clean control failed", "out": (cp.stdout + cp.stderr)[-800:]}
        mut = _copy_root(Path(td) / "mut")
        f = mut / rel
        src = f.read_text()
        for old, new in pairs:
            if src.count(old) != 1:
                return {"mutant": mid, "result": "INVALID", "why": f"pattern count {src.count(old)}: {old[:60]!r}"}
            src = src.replace(old, new)
        f.write_text(src)
        mp = run_gate_subprocess(gate, mut, Path(td) / "w_mut")
        out = mp.stdout + mp.stderr
        if mp.returncode == 0:
            return {"mutant": mid, "result": "SURVIVED", "gate": gate}
        if "AssertionError" in out and "G-" in out:
            return {"mutant": mid, "result": "DETECTED", "gate": gate,
                    "assertion": [l for l in out.splitlines() if "AssertionError" in l][-1][-300:]}
        return {"mutant": mid, "result": "INVALID", "why": "died before the credited assertion", "out": out[-800:]}


def _pytest_gate(name):
    os.makedirs(REPO / "logs", exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(REPO / "logs")) as td:
        p = run_gate_subprocess(name, REPO, Path(td) / "w")
        assert p.returncode == 0, (p.stdout + p.stderr)[-2500:]


for _n in GATES:
    globals()[f"test_gate_{_n}"] = (lambda n: (lambda: _pytest_gate(n)))(_n)


def test_mutants_all_detected():
    os.makedirs(REPO / "logs", exist_ok=True)
    res = [run_mutant(*m) for m in MUTANTS]
    print(json.dumps(res, indent=1))
    bad = [r for r in res if r["result"] != "DETECTED"]
    assert not bad, json.dumps(bad, indent=1)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "gate":
        name = sys.argv[2]
        root = sys.argv[sys.argv.index("--root") + 1]
        work = sys.argv[sys.argv.index("--work") + 1] if "--work" in sys.argv else tempfile.mkdtemp()
        Path(work).mkdir(parents=True, exist_ok=True)
        GATES[name](root, work)
        print(f"GATE {name}: PASS")
        sys.exit(0)
    if len(sys.argv) >= 2 and sys.argv[1] == "mutants":
        print(json.dumps([run_mutant(*m) for m in MUTANTS], indent=1))
        sys.exit(0)
    print(__doc__)
