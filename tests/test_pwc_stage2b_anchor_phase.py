#!/usr/bin/env python3
"""PWC-BACKUP STAGE 2b — worker side of the window-anchor / generator-phase split.

CPU-only. No GPU, no rigs, no network, no live ledger. CuPy is replaced by an
in-process stand-in BEFORE sieve_filter is imported, and `GPUSieve` is replaced
by a recorder, so no kernel is compiled or launched. Scratch files live under
this worktree's ignored `logs/` (never /tmp).

Gates
  G-OLD-1   old-style jobs: execute_sieve_job at BASE and in the working tree make
            the same GPUSieve calls with the same arguments and return the same
            result (json-identical), across constant/reverse/hybrid/two-phase jobs,
            anchors 0 and 25, both and single sessions.
  G-OLD-2   old-style jobs: the worker's sieve_job dict is identical to BASE's.
  G-NEW-1   window_anchor selects the draws: equal to RANGE-MINER's
            load_residue_window(path, w, sessions, anchor) and to the plain slice,
            for both-session and single-session jobs; boundary anchor accepted.
  G-NEW-2   generator_phase 0 is what reaches the kernel's trailing argument
            (`offset=` of run_sieve / run_hybrid_sieve), whatever the anchor; the
            result records window_anchor and generator_phase.
  G-REF-1   generator_phase != 0 raises PWC_GENERATOR_PHASE_REFUSED before any
            dataset read or GPUSieve construction.
  G-REF-2   one key without the other, `offset` alongside the new keys, and a
            non-integer value are refused the same way.
  G-REF-3   an out-of-domain anchor (negative, or > N_filtered - window_size) is
            refused, not clamped, and no GPUSieve is constructed.
  G-WRK-1   the worker copies window_anchor and generator_phase into the sieve
            job and drops its DEFAULTED offset; an offset the job itself sent is
            kept (so sieve_filter refuses the mixed job).
  G-WRK-2   worker + sieve_filter end to end: a phase-1 job comes back
            success=False naming PWC_GENERATOR_PHASE_REFUSED; a phase-0 job runs.
  G-PRF-1   the proof script's AST check detects a changed module-level statement
            (mutant: pool limit default 256 -> 512) and passes on an unchanged copy.
  G-PRF-2   the proof script's import check detects the same mutant.
"""
import copy
import importlib.util
import json
import os
import subprocess
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.environ.get("PWC_STAGE2B_BASE", "0caeaf9")
SCRATCH = os.path.join(REPO, "logs", "test_pwc_stage2b")
sys.path.insert(0, REPO)

# --- CuPy stand-in, installed before anything can import the real one ---------
_cp = types.ModuleType("cupy")
class _Pool:
    def set_limit(self, *a, **k): pass
_cp.get_default_memory_pool = lambda: _Pool()
sys.modules["cupy"] = _cp

PASS, FAIL = [], []
def gate(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" — {detail}" if detail and not cond else ""), flush=True)


def _load_module(name, src, path_hint):
    spec = importlib.util.spec_from_loader(name, loader=None, origin=path_hint)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = path_hint
    exec(compile(src, path_hint, "exec"), mod.__dict__)
    sys.modules[name] = mod
    return mod


def _git_show(path):
    return subprocess.run(["git", "-C", REPO, "show", f"{BASE}:{path}"], check=True,
                          capture_output=True, text=True).stdout


# --- the two versions of each file -------------------------------------------
import sieve_filter as SF_NEW                                    # working tree
SF_BASE = _load_module("sieve_filter_base", _git_show("sieve_filter.py"),
                       os.path.join(REPO, "sieve_filter.py@BASE"))
from persistent import pwc_worker_service as WS_NEW
WS_BASE = _load_module("pwc_worker_service_base", _git_show("persistent/pwc_worker_service.py"),
                       os.path.join(REPO, "persistent", "pwc_worker_service.py@BASE"))
from miner.range_miner_worker import load_residue_window       # read-only reference


# --- GPUSieve recorder --------------------------------------------------------
class Recorder:
    def __init__(self):
        self.calls = []
    def factory(self):
        rec = self
        class FakeSieve:
            def __init__(self, gpu_id=0):
                rec.calls.append(("init", {"gpu_id": gpu_id}))
            def _res(self, kw, tag):
                return {"family": kw["prng_family"],
                        "seed_range": {"start": kw["seed_start"], "end": kw["seed_end"]},
                        "survivors": [{"seed": kw["seed_start"], "match_rate": 0.9, "tag": tag}],
                        "stats": {"seeds_tested": kw["seed_end"] - kw["seed_start"],
                                  "survivors_found": 1, "duration_ms": 1.0,
                                  "seeds_per_sec": 1.0}}
            def run_sieve(self, **kw):
                rec.calls.append(("run_sieve", copy.deepcopy(kw)))
                return self._res(kw, "const")
            def run_hybrid_sieve(self, **kw):
                kw2 = copy.deepcopy({k: v for k, v in kw.items() if k != "strategies"})
                kw2.setdefault("miner_abi", False)   # BASE passes none; False is the default
                kw2["n_strategies"] = len(kw.get("strategies") or [])
                rec.calls.append(("run_hybrid_sieve", kw2))
                return self._res(kw, "hybrid")
        return FakeSieve


def run(mod, job):
    rec = Recorder()
    saved = mod.GPUSieve
    mod.GPUSieve = rec.factory()
    try:
        out = mod.execute_sieve_job(copy.deepcopy(job), 0)
    finally:
        mod.GPUSieve = saved
    return rec.calls, out


def run_raises(mod, job):
    rec = Recorder()
    saved = mod.GPUSieve
    mod.GPUSieve = rec.factory()
    try:
        mod.execute_sieve_job(copy.deepcopy(job), 0)
        return None, rec.calls
    except Exception as e:                       # noqa: BLE001 — we inspect it
        return e, rec.calls
    finally:
        mod.GPUSieve = saved


# --- fixture dataset ----------------------------------------------------------
os.makedirs(SCRATCH, exist_ok=True)
DS = os.path.join(SCRATCH, "daily3_fixture.json")
_data = []
for i in range(120):
    e = {"draw": (i * 37 + 11) % 1000, "session": "midday" if i % 2 == 0 else "evening"}
    if i % 5 == 0:
        e["full_state"] = 100000 + i
    _data.append(e)
with open(DS, "w") as f:
    json.dump(_data, f)

from hybrid_strategy import get_all_strategies
STRATS = [s.to_dict() if hasattr(s, "to_dict") else dict(vars(s)) for s in get_all_strategies()]

def job(**kw):
    j = {"job_id": "t", "search_type": "residue_sieve", "dataset_path": DS,
         "seed_start": 1000, "seed_end": 3000, "window_size": 12,
         "min_match_threshold": 0.71, "skip_range": [6, 99],
         "sessions": ["midday", "evening"], "prng_families": ["java_lcg"],
         "strategies": None, "hybrid": False, "phase2_threshold": 0.71}
    j.update(kw)
    return j

OLD_JOBS = []
for fam, hyb in (("java_lcg", False), ("java_lcg_reverse", False),
                 ("java_lcg_hybrid", True), ("java_lcg_hybrid_reverse", True),
                 ("java_lcg", True)):                       # last = two-phase path
    for off in (0, 25):
        for sess in (["midday", "evening"], ["midday"]):
            OLD_JOBS.append(job(prng_families=[fam], hybrid=hyb, offset=off, sessions=sess,
                                strategies=STRATS if hyb else None))
OLD_JOBS.append({k: v for k, v in job().items() if k != "sessions"})   # no sessions, no offset key

print("== G-OLD: jobs without the new keys", flush=True)
same_calls = same_out = True
for j in OLD_JOBS:
    cb, ob = run(SF_BASE, j)
    cn, on = run(SF_NEW, j)
    same_calls &= (cb == cn)
    same_out &= (json.dumps(ob, sort_keys=True) == json.dumps(on, sort_keys=True))
gate("G-OLD-1a GPUSieve calls identical to BASE (%d jobs)" % len(OLD_JOBS), same_calls)
gate("G-OLD-1b results json-identical to BASE (%d jobs)" % len(OLD_JOBS), same_out)
gate("G-OLD-1c old jobs carry no window_anchor/generator_phase in the result",
     all("window_anchor" not in run(SF_NEW, j)[1] for j in OLD_JOBS))
gate("G-OLD-1d old jobs: every hybrid launch keeps the old argument list (miner_abi False)",
     all(c[1]["miner_abi"] is False for j in OLD_JOBS for c in run(SF_NEW, j)[0]
         if c[0] == "run_hybrid_sieve"))

def worker(mod):
    w = mod.PWCWorkerService("test:gpu0", "127.0.0.1", 1, 0, use_rocm=False)
    w._emit_heartbeat = lambda *a, **k: None                 # no files under ~
    seen = []
    w._execute_sieve_job = lambda sj, g: (seen.append(copy.deepcopy(sj)) or {"success": True})
    return w, seen

# [STAGE 2b, owner 2026-11-20] the TCP worker is PWC's entry point and takes
# new-key jobs only, as RANGE-MINER's worker does (miner/range_miner_worker.py:813-860)
wn, sn = worker(WS_NEW)
old_refused = True
for j in [dict(j, lease_id="L", attempt=1) for j in OLD_JOBS] + [
        {"job_id": "c", "dataset_path": DS, "seed_start": 0, "seed_end": 10, "window_size": 12,
         "chunk_offset": 7, "threshold": 0.4, "prng_type": "java_lcg", "is_hybrid": True}]:
    r = wn._execute_job(copy.deepcopy(j))
    code = ("PWC_LEGACY_OFFSET_REFUSED" if ("offset" in j or "chunk_offset" in j)
            else "PWC_WINDOW_ANCHOR_SCHEMA_REFUSED")
    old_refused &= r["success"] is False and code in (r.get("error") or "")
gate("G-OLD-2 worker refuses every old-style job (%d jobs: offset, chunk_offset, neither key)"
     % (len(OLD_JOBS) + 1), old_refused and sn == [])

print("== G-NEW: window_anchor / generator_phase", flush=True)
ok_draws = ok_phase = ok_echo = ok_abi = True
for fam, hyb in (("java_lcg", False), ("java_lcg_reverse", False),
                 ("java_lcg_hybrid", True), ("java_lcg_hybrid_reverse", True)):
    for sess in (["midday", "evening"], ["midday"], ["evening"]):
        for anchor in (0, 25, 7):
            j = job(prng_families=[fam], hybrid=hyb, sessions=sess,
                    strategies=STRATS if hyb else None, window_anchor=anchor, generator_phase=0)
            calls, out = run(SF_NEW, j)
            kern = [c for c in calls if c[0] in ("run_sieve", "run_hybrid_sieve")]
            ref = load_residue_window(DS, 12, sess, anchor)
            filt = [e for e in _data if e["session"] in sess]
            plain = [int(e.get("full_state", e["draw"])) for e in filt[anchor:anchor + 12]]
            ok_draws &= bool(kern) and all(c[1]["residues"] == ref == plain for c in kern)
            ok_phase &= bool(kern) and all(c[1]["offset"] == 0 for c in kern)
            ok_abi &= all(c[1]["miner_abi"] is True for c in kern if c[0] == "run_hybrid_sieve")
            ok_echo &= out.get("success") is True and out.get("window_anchor") == anchor \
                and out.get("generator_phase") == 0
gate("G-NEW-1a draws == miner load_residue_window == slice (anchors 0/7/25, 3 session sets)", ok_draws)
gate("G-NEW-2a kernel trailing argument is generator_phase 0 at every anchor", ok_phase)
gate("G-NEW-2b result records window_anchor and generator_phase", ok_echo)
gate("G-NEW-2d new-key hybrid launches ask for the miner's argument list (miner_abi True)", ok_abi)
# the old path at offset 25 sends 25 to the kernel; the new path at anchor 25 sends 0
_c_old, _ = run(SF_NEW, job(offset=25))
_c_new, _ = run(SF_NEW, job(window_anchor=25, generator_phase=0))
gate("G-NEW-2c same draws as legacy offset=25, kernel arg 25 -> 0",
     _c_old[1][1]["residues"] == _c_new[1][1]["residues"]
     and _c_old[1][1]["offset"] == 25 and _c_new[1][1]["offset"] == 0)
n_mid = sum(1 for e in _data if e["session"] == "midday")
calls, out = run(SF_NEW, job(sessions=["midday"], window_anchor=n_mid - 12, generator_phase=0))
gate("G-NEW-1b boundary anchor N_filtered - window_size accepted", out.get("success") is True)

print("== G-REF: refusals", flush=True)
missing_ds = job(dataset_path=os.path.join(SCRATCH, "does_not_exist.json"))
for ph in (1, -1, 25):
    e, calls = run_raises(SF_NEW, dict(missing_ds, window_anchor=0, generator_phase=ph))
    gate(f"G-REF-1 generator_phase={ph} raises PWC_GENERATOR_PHASE_REFUSED, no dataset read, no GPUSieve",
         isinstance(e, ValueError) and "PWC_GENERATOR_PHASE_REFUSED" in str(e) and calls == [],
         repr(e))
for label, extra in (("window_anchor only", {"window_anchor": 0}),
                     ("generator_phase only", {"generator_phase": 0}),
                     ("offset alongside", {"window_anchor": 0, "generator_phase": 0, "offset": 0}),
                     ("non-integer anchor", {"window_anchor": "x", "generator_phase": 0}),
                     ("None phase", {"window_anchor": 0, "generator_phase": None})):
    e, calls = run_raises(SF_NEW, dict(missing_ds, **extra))
    gate(f"G-REF-2 {label}: refused before any read",
         isinstance(e, ValueError) and "PWC_WINDOW_ANCHOR_SCHEMA_REFUSED" in str(e) and calls == [],
         repr(e))
for anchor in (-1, n_mid - 12 + 1, 10_000):
    calls, out = run(SF_NEW, job(sessions=["midday"], window_anchor=anchor, generator_phase=0))
    gate(f"G-REF-3 anchor {anchor} refused, not clamped, no GPUSieve",
         out.get("success") is False and "PWC_WINDOW_ANCHOR_OUT_OF_DOMAIN" in out.get("error", "")
         and calls == [], out.get("error", ""))

print("== G-WRK: worker pass-through", flush=True)
w, seen = worker(WS_NEW)
w._execute_job(job(window_anchor=25, generator_phase=0))
s = seen[-1]
gate("G-WRK-1a keys copied verbatim, defaulted offset dropped",
     s.get("window_anchor") == 25 and s.get("generator_phase") == 0 and "offset" not in s, str(s))
n_seen = len(seen)
r = w._execute_job(dict(job(window_anchor=25, generator_phase=0), offset=25))
gate("G-WRK-1b offset alongside the new keys: refused by the worker, nothing run",
     r["success"] is False and "PWC_LEGACY_OFFSET_REFUSED" in r["error"] and len(seen) == n_seen)
r1 = w._execute_job(job(generator_phase=0)); r2 = w._execute_job(job(window_anchor=0))
gate("G-WRK-1c one key alone: refused by the worker, nothing run",
     all(x["success"] is False and "PWC_WINDOW_ANCHOR_SCHEMA_REFUSED" in x["error"] for x in (r1, r2))
     and len(seen) == n_seen)

w2 = WS_NEW.PWCWorkerService("test:gpu0", "127.0.0.1", 1, 0, use_rocm=False)
w2._emit_heartbeat = lambda *a, **k: None
rec = Recorder(); saved = SF_NEW.GPUSieve; SF_NEW.GPUSieve = rec.factory()
try:
    w2._execute_sieve_job = SF_NEW.execute_sieve_job
    r1 = w2._execute_job(job(window_anchor=25, generator_phase=1))
    r0 = w2._execute_job(job(window_anchor=25, generator_phase=0))
finally:
    SF_NEW.GPUSieve = saved
gate("G-WRK-2a end to end: phase 1 -> success False naming PWC_GENERATOR_PHASE_REFUSED",
     r1["success"] is False and "PWC_GENERATOR_PHASE_REFUSED" in (r1.get("error") or ""))
gate("G-WRK-2b end to end: phase 0 -> success, anchor recorded in payload",
     r0["success"] is True and r0["payload"].get("window_anchor") == 25
     and r0["payload"].get("generator_phase") == 0)

print("== G-HYB: the real run_hybrid_sieve's kernel arguments", flush=True)
import re as _re
from miner.range_miner_worker import EXPECTED_KERNEL_ARITY           # read-only reference
from prng_registry import get_kernel_info


class _Arr:
    def __getitem__(self, i): return self
    def get(self): return 0


class _Dev:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def synchronize(self): pass


def _scalar(tag):
    return lambda v=None: (tag, v)


_FCP = types.SimpleNamespace(
    int32=_scalar("int32"), uint32=_scalar("uint32"), uint64=_scalar("uint64"),
    float32=_scalar("float32"),
    array=lambda *a, **k: _Arr(), zeros=lambda *a, **k: _Arr(), arange=lambda *a, **k: _Arr(),
    cuda=types.SimpleNamespace(Device=lambda *a: _Dev()))


def hybrid_args(mod, family, *, miner_abi=None, offset=0):
    """Kernel arguments the REAL GPUSieve.run_hybrid_sieve passes, CuPy replaced."""
    launched = []
    sv = object.__new__(mod.GPUSieve)
    sv.device, sv.compiled_kernels = _Dev(), {}
    cfg = get_kernel_info(family)
    sv._get_kernel = lambda fam, params=None: (
        lambda grid, block, args: launched.append(list(args)), cfg)
    saved = mod.cp
    mod.cp = _FCP
    try:
        kw = {} if miner_abi is None else {"miner_abi": miner_abi}
        sv.run_hybrid_sieve(prng_family=family, seed_start=0, seed_end=10, residues=[1] * 12,
                            strategies=STRATS, min_match_threshold=0.5, offset=offset, **kw)
    finally:
        mod.cp = saved
    assert len(launched) == 1, launched
    # scalars are (dtype, value) tuples; device arrays are compared by kind only
    return [x if isinstance(x, tuple) else type(x).__name__ for x in launched[0]], cfg


def declared_params(cfg):
    m = _re.search(_re.escape(cfg["kernel_name"]) + r"\s*\(([^)]*)\)", cfg["kernel_source"])
    return len([x for x in m.group(1).split(",") if x.strip()])


HYB = [f for f in EXPECTED_KERNEL_ARITY if "_hybrid" in f]
ok_miner = ok_decl = ok_old = ok_drop = True
detail = []
for fam in HYB:
    a_new, cfg = hybrid_args(SF_NEW, fam, miner_abi=True)
    a_dflt, _ = hybrid_args(SF_NEW, fam)
    a_base, _ = hybrid_args(SF_BASE, fam)
    ok_miner &= len(a_new) == EXPECTED_KERNEL_ARITY[fam]
    ok_decl &= len(a_new) == declared_params(cfg)
    ok_old &= repr(a_dflt) == repr(a_base)
    no_phase = fam in ("java_lcg_hybrid", "minstd_hybrid", "xorshift32_hybrid", "xorshift128_hybrid")
    ok_drop &= (repr(a_new) == repr(a_base[:-1]) if no_phase else repr(a_new) == repr(a_base))
    detail.append(f"{fam}: pwc {len(a_new)} / miner {EXPECTED_KERNEL_ARITY[fam]} / "
                  f"kernel {declared_params(cfg)} / old {len(a_base)}")
gate("G-HYB-1 new-key launch: argument count == miner EXPECTED_KERNEL_ARITY for all %d hybrids"
     % len(HYB), ok_miner, "; ".join(detail))
gate("G-HYB-2 new-key launch: argument count == the kernel's declared parameters", ok_decl,
     "; ".join(detail))
gate("G-HYB-3 default (legacy/ZMQ callers): arguments identical to BASE for every hybrid", ok_old)
gate("G-HYB-4 the change is only the trailing int32 of the four no-phase forward hybrids", ok_drop)
e = None
try:
    hybrid_args(SF_NEW, "java_lcg_hybrid", miner_abi=True, offset=1)
except Exception as exc:                          # noqa: BLE001
    e = exc
gate("G-HYB-5 new-key launch, no-phase hybrid, phase 1: refused before any kernel launch",
     isinstance(e, ValueError) and "PWC_GENERATOR_PHASE_REFUSED" in str(e), repr(e))
print("    " + "\n    ".join(detail), flush=True)

print("== G-PRF: the proof script detects a mutant", flush=True)
proof = _load_module("pwc_proof", open(os.path.join(REPO, "scripts", "pwc_sieve_filter_import_proof.py")).read(),
                     os.path.join(REPO, "scripts", "pwc_sieve_filter_import_proof.py"))
base_src = _git_show("sieve_filter.py")
mut_src = base_src.replace('"PRNG_CUPY_POOL_LIMIT_MB", "256"', '"PRNG_CUPY_POOL_LIMIT_MB", "512"', 1)
assert mut_src != base_src
gate("G-PRF-1a AST check passes on an unchanged copy",
     proof.ast_check(base_src, base_src, {"execute_sieve_job"})["ok"])
gate("G-PRF-1b AST check fails on the pool-limit mutant",
     not proof.ast_check(base_src, mut_src, {"execute_sieve_job"})["ok"])
mdir = os.path.join(SCRATCH, "mutant"); bdir = os.path.join(SCRATCH, "basecopy")
for d, src in ((mdir, mut_src), (bdir, base_src)):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "sieve_filter.py"), "w") as f:
        f.write(src)
rb, rm = proof.import_report(bdir, "real", False), proof.import_report(mdir, "real", False)
gate("G-PRF-2 import check sees set_limit(268435456) vs set_limit(536870912)",
     rb != rm and "268435456" in json.dumps(rb) and "536870912" in json.dumps(rm))

print(f"\nRESULT: {len(PASS)} passed, {len(FAIL)} failed", flush=True)
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
