#!/usr/bin/env python3
"""PWC-BACKUP STAGE 2b — proof that an edit to sieve_filter.py leaves what
RANGE-MINER's launch process gets from it unchanged.

Why this exists. RANGE-MINER's VM101 launch process imports sieve_filter.py at
module load (window_optimizer_integration_final.py:52-55, inside try/except
ImportError), so its module top level runs inside the miner's process: the ROCm
env prelude, the CuPy import and `get_default_memory_pool().set_limit(...)`. The
rig miner worker never imports it (scripts/gate12_parity_gate.py GOVERNED_FILES).

Two checks, both read-only and CPU-only:

1. AST. Every module-level statement of the file at BASE and in the working
   tree is compared with `ast.dump(..., include_attributes=False)`, in order,
   EXCEPT the top-level functions/classes named with --allow (default:
   execute_sieve_job). The allowed names must exist at the same position on
   both sides (nothing added, removed or moved).

2. Import. For BASE and for the working tree, a fresh `python3 -I` subprocess
   with a minimal fixed environment imports `sieve_filter` and reports:
     * the os.environ diff caused by the import (added / changed / removed);
     * every CuPy access and call made during the import, including the
       pool-limit value passed to set_limit.
   CuPy is replaced by a RECORDING STAND-IN (no GPU, no driver, no context).
   Each version is imported twice: with the real hostname, and with
   socket.gethostname() patched to "rig-6600" so the hostname-gated HSA
   prelude is exercised too. Reports must be identical pairwise.
   `--real-cupy` uses the real CuPy instead and reads get_limit(); it touches
   the GPU and is for an owner-run session only.

Exit 0 = both checks hold; 1 = a difference (printed); 2 = usage/setup error.
Usage:
  python3 scripts/pwc_sieve_filter_import_proof.py [--base ee39404]
          [--file sieve_filter.py] [--allow execute_sieve_job ...] [--real-cupy]
"""
import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CHILD = r'''
import json, os, sys, types, socket
version_dir, repo, host_mode, real_cupy, modname = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1", sys.argv[5]
sys.path[:0] = [version_dir, repo]
events = []
if not real_cupy:
    class _Rec:
        def __init__(self, path): self._p = path
        def __getattr__(self, name):
            if name.startswith("__"): raise AttributeError(name)
            events.append(["getattr", self._p + "." + name]); return _Rec(self._p + "." + name)
        def __call__(self, *a, **k):
            events.append(["call", self._p, [repr(x) for x in a], {kk: repr(v) for kk, v in sorted(k.items())}])
            return _Rec(self._p + "()")
    mod = types.ModuleType("cupy")
    def _mod_getattr(name):
        if name.startswith("__"):
            raise AttributeError(name)
        events.append(["getattr", "cupy." + name])
        return _Rec("cupy." + name)
    mod.__getattr__ = _mod_getattr
    sys.modules["cupy"] = mod
if host_mode != "real":
    socket.gethostname = lambda: host_mode
import importlib
before = dict(os.environ)
mods_before = set(sys.modules)
target = importlib.import_module(modname)
mods_added = sorted(set(sys.modules) - mods_before)
after = dict(os.environ)
rep = {
    "module_file_is_version_dir": os.path.dirname(os.path.abspath(target.__file__)) == os.path.abspath(version_dir),
    "modules_added": mods_added,
    "env_added":   {k: after[k] for k in sorted(after) if k not in before},
    "env_changed": {k: [before[k], after[k]] for k in sorted(after) if k in before and before[k] != after[k]},
    "env_removed": sorted(k for k in before if k not in after),
    "cupy_events": events,
    "GPU_AVAILABLE": getattr(target, "GPU_AVAILABLE", None),
}
if real_cupy:
    import cupy as cp
    rep["pool_limit_bytes"] = cp.get_default_memory_pool().get_limit()
print("@@REPORT@@" + json.dumps(rep, sort_keys=True))
'''

_FIXED_ENV = {
    "PATH": "/usr/bin:/bin",
    "HOME": os.environ.get("HOME", "/home/michael"),
    "LANG": "C.UTF-8",
    "CUDA_VISIBLE_DEVICES": "",      # belt and braces: no device even if real CuPy loads
}


def _git_show(rev, path):
    return subprocess.run(["git", "-C", REPO, "show", f"{rev}:{path}"], check=True,
                          capture_output=True, text=True).stdout


def ast_check(base_src, new_src, allow, allow_added=()):
    """Delegates to scripts/prove_unchanged_outside.py (qualified names, any depth)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import prove_unchanged_outside as _P
    rep = _P.compare(base_src, new_src, allow, allow_added)
    return {"ok": rep["ok"], "problems": rep["problems"], "allowed": rep["allowed"],
            "allowed_added": rep["allowed_added"],
            "base_statements": rep["base_top_level_statements"],
            "new_statements": rep["new_top_level_statements"],
            "checked_definitions": rep["checked_definitions"]}


def _ast_check_top_level_only(base_src, new_src, allow):
    """The Stage 2b v1 check (top-level names only), kept for reference."""
    b, n = ast.parse(base_src).body, ast.parse(new_src).body

    def name(s):
        return s.name if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else None

    out = {"base_statements": len(b), "new_statements": len(n), "allowed": sorted(allow)}
    bpos = [i for i, s in enumerate(b) if name(s) in allow]
    npos = [i for i, s in enumerate(n) if name(s) in allow]
    out["allowed_positions_base"], out["allowed_positions_new"] = bpos, npos
    bk = [ast.dump(s, include_attributes=False) for s in b if name(s) not in allow]
    nk = [ast.dump(s, include_attributes=False) for s in n if name(s) not in allow]
    diffs = [i for i in range(max(len(bk), len(nk)))
             if i >= len(bk) or i >= len(nk) or bk[i] != nk[i]]
    changed = sorted({name(s) for s in b if name(s) in allow
                      for t in n if name(t) == name(s)
                      and ast.dump(s, include_attributes=False) != ast.dump(t, include_attributes=False)})
    out["compared_statements"] = len(bk)
    out["differing_outside_allowed"] = diffs
    out["allowed_names_that_changed"] = changed
    out["ok"] = (not diffs and len(bk) == len(nk) and bpos == npos)
    return out


def import_report(version_dir, host_mode, real_cupy, modname="sieve_filter"):
    p = subprocess.run([sys.executable, "-I", "-c", _CHILD, version_dir, REPO, host_mode,
                        "1" if real_cupy else "0", modname],
                       env=_FIXED_ENV, capture_output=True, text=True, timeout=300)
    for line in p.stdout.splitlines():
        if line.startswith("@@REPORT@@"):
            return json.loads(line[len("@@REPORT@@"):])
    raise RuntimeError(f"child produced no report (rc={p.returncode}): {p.stderr[-2000:]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="0caeaf9")
    ap.add_argument("--file", default="sieve_filter.py")
    ap.add_argument("--allow", nargs="*", default=["execute_sieve_job", "GPUSieve.run_hybrid_sieve"])
    ap.add_argument("--allow-added", nargs="*", default=[])
    ap.add_argument("--real-cupy", action="store_true")
    a = ap.parse_args()
    if a.file not in ("sieve_filter.py", "persistent_worker_coordinator.py"):
        print("only sieve_filter.py and persistent_worker_coordinator.py (both imported by "
              "RANGE-MINER's launch, window_optimizer_integration_final.py:52-58) are "
              "supported for the import check", file=sys.stderr)
        return 2
    modname = a.file[:-3]
    base_src = _git_show(a.base, a.file)
    with open(os.path.join(REPO, a.file)) as f:
        new_src = f.read()

    result = {"base": a.base, "file": a.file, "cupy": "real" if a.real_cupy else "recording stand-in"}
    result["ast"] = ast_check(base_src, new_src, set(a.allow), set(a.allow_added))

    scratch = os.path.join(REPO, "logs", "pwc_sieve_filter_import_proof")   # ignored path, never /tmp
    os.makedirs(scratch, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as td:
        bdir, ndir = os.path.join(td, "base"), os.path.join(td, "new")
        os.makedirs(bdir); os.makedirs(ndir)
        for d, src in ((bdir, base_src), (ndir, new_src)):
            with open(os.path.join(d, a.file), "w") as f:
                f.write(src)
        imp = {}
        for host in ("real", "rig-6600"):
            rb = import_report(bdir, host, a.real_cupy, modname)
            rn = import_report(ndir, host, a.real_cupy, modname)
            imp[host] = {"identical": rb == rn, "base": rb,
                         **({} if rb == rn else {"new": rn})}
        result["import"] = imp

    ok = result["ast"]["ok"] and all(v["identical"] and v["base"]["module_file_is_version_dir"]
                                     for v in result["import"].values())
    result["verdict"] = "HOLDS" if ok else "DOES NOT HOLD"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
