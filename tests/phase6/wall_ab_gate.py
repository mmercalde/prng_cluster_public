#!/usr/bin/env python3
"""
tests/phase6/wall_ab_gate.py — S172 bounded Phase 6, WALL A and WALL B.

Authority: docs/CLAUDE_CODE_INSTRUCTIONS_BOUNDED_PHASE_6.md §1, §2, §7, §9.

WALL A — INTERFACE AND CONSUMER (§1)
    "Steps 2 onward cannot tell which engine produced the data" is the pivot's
    founding rule, and Beta was explicit that it is BROADER than opening the NPZ
    in Step 2. So this gate walks the whole consumer chain on a miner-produced
    CERTIFIED generation:

        certified NPZ
          -> A1  frozen 22-array contract: names, ORDER, shapes, dtypes
          -> A2  utils.canonical_arrays.validate_array_bundle()
          -> A3  Step-2 loader (utils.survivor_loader.load_survivors),
                 fallback_used MUST be False
          -> A4  NPZ -> dict conversion preserves EVERY field
                 (generate_step3_scoring_jobs.extract_survivors_full)
          -> A5  Step-3 chunk generation preserves the contract
                 (chunk_list -> chunk_NNNN.json -> read back)
          -> A6  survivors_with_scores smoke completes without metadata loss
                 (full_scoring_worker.py on a real chunk, real GPU)

WALL B — DETERMINISM AND PLATFORM (§2)
    With identical frozen inputs and configuration:
        B1  CUDA vs ROCm                      — CITED from Phase 6.0, NOT re-run
        B2  repeated run vs repeated run      — FRESH
        B3  serial_reference vs process_sharded — FRESH
        B4  multi-rig: identical results independent of NODE ASSIGNMENT — FRESH
            (the one leg with no prior evidence; Phase 6.0 was single-rig)

    Beta's provenance list is bound in the §PROVENANCE block and written to the
    JSON record.

WHAT IS REAL AND WHAT IS HARNESS
    REAL: the GPUs, the miner worker daemons (separate processes, and separate
          HOSTS for B4), the production coordinator via
          `WOI.run_bidirectional_test(use_range_miner=True)`, the Phase-5
          assembly, `utils.run_finalizer.finalize_run`, the NPZ writer, and every
          consumer named in Wall A.
    REAL: the P0.5 dataset authority. `run_start_dataset_gate` is invoked at run
          start with `miner_backed=True`, so the pointer manifest is resolved,
          the dataset is frozen for the run, and every rig is verified ON TARGET
          before a worker is dispatched. The absolute immutable path is what
          reaches the payload — never the legacy `daily3.json` alias, which does
          not even exist on rrig6600b/c.
    HARNESS: the optimizer loop. Each arm drives ONE trial directly rather than
          running `optimize_window`'s search; the trial itself is the production
          call. Same limitation, same wording, as the D6 3.B smoke.
    HARNESS: repository identity in the default SCRATCH mode — see the D6 smoke
          header. `--release-grade` is passed through unchanged.

Run (all legs, ~15 min):
    source ~/venvs/torch/bin/activate
    python tests/phase6/wall_ab_gate.py --all --json docs/phase6_evidence/wall_ab.json
Selective:
    python tests/phase6/wall_ab_gate.py --wall-a
    python tests/phase6/wall_ab_gate.py --wall-b-repeat --wall-b-backend
    python tests/phase6/wall_ab_gate.py --wall-b-multirig
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils import run_finalizer as RF                                  # noqa: E402
from utils.canonical_arrays import validate_array_bundle               # noqa: E402
from utils.survivor_loader import load_survivors                       # noqa: E402
from window_optimizer import WindowConfig                              # noqa: E402

# ===========================================================================
# LAZY IMPORTS — LOAD-BEARING, DO NOT HOIST TO MODULE LEVEL
# ===========================================================================
# `window_optimizer_integration_final` imports cupy at module level, and the D6
# 3.B smoke module imports WOI. This module must stay CUPY-FREE AT IMPORT TIME,
# for a reason that is not stylistic:
#
#   D5's `process_sharded` backend builds a multiprocessing pool with the SPAWN
#   start method, and a spawn child re-imports the parent's `__main__` module
#   before running its task. `assembly_shard_worker` enforces §6.7.A — "assembly
#   is CPU-only work and a worker must never hold a GPU context" — by refusing
#   any shard worker that finds a GPU module in `sys.modules`
#   (`ShardArtifactError`, assembly_shard_worker.py). So if THIS file imported
#   WOI at module level, every spawn child would import cupy through it and the
#   guard would kill the B3 arm before it started.
#
#   That is exactly what happened on the first run of this gate: the B3 arm died
#   with BrokenProcessPool / "no committed assembly". Verified this session that
#   it is a HARNESS defect and NOT a production one: `window_optimizer.py` — the
#   real Step-1 `__main__` — has cupy ABSENT from `sys.modules` after its own
#   import (it imports WOI lazily, inside `run_bayesian_optimization`), so a
#   production spawn child re-imports a cupy-free `__main__` and the guard
#   passes. Measured both ways, not assumed.
#
# Keeping these imports inside functions restores that property for the gate.
_D6_CACHE = {}


def _d6():
    """The D6 3.B smoke module, loaded on first use.

    Imported (not copied) so the frozen 22-array oracle, the coordinator
    attribute surface and the threshold-provenance check are the SAME objects
    the Phase 6.0 evidence was produced with, rather than a fork that could
    drift. `tests/` has no `__init__.py`, hence the by-path load.
    """
    if "mod" not in _D6_CACHE:
        import importlib.util as _ilu
        smoke_path = os.path.join(_ROOT, "tests",
                                  "smoke_s172_phase5_d6_zeus_single_gpu.py")
        spec = _ilu.spec_from_file_location("_d6_smoke", smoke_path)
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _D6_CACHE["mod"] = mod
    return _D6_CACHE["mod"]


def _woi():
    import window_optimizer_integration_final as WOI
    return WOI


# The frozen 22 array names, hand-transcribed exactly as the D6 smoke
# transcribes them — never imported from the module that PRODUCES them. Kept
# here as a literal (rather than read off `_d6()`) so the oracle is available
# without triggering the cupy import chain, and cross-checked against the
# smoke's copy in `main()`.
ORACLE_ARRAY_NAMES = (
    "seeds", "forward_matches", "reverse_matches", "window_size", "offset",
    "trial_number", "skip_min", "skip_max", "skip_range", "forward_count",
    "reverse_count", "bidirectional_count", "intersection_count",
    "intersection_ratio", "intersection_weight", "bidirectional_selectivity",
    "forward_only_count", "reverse_only_count", "survivor_overlap_ratio",
    "score", "skip_mode", "prng_type",
)

PRNG_BASE = "java_lcg"
# The P0.5 gate takes the LEGACY ALIAS as its argument and resolves the pointer
# manifest that sits beside it (dataset_authority.resolve_dataset_path case 1) —
# the same argument `window_optimizer.py:1486` passes. Handing it the pointer
# manifest directly is wrong: the manifest is a JSON object, not a draw array,
# and `count_records` refuses it. What comes BACK is the absolute immutable
# version path, and that is what reaches every worker.
DATASET_ALIAS = os.path.join(_ROOT, "daily3.json")
POINTER_MANIFEST = os.path.join(_ROOT, "daily3_current.json")

# Every rig endpoint is the CT100 worker address (CLAUDE.md §3 / skill §6),
# because the rigs are currently booted into Proxmox. The bare-metal addresses
# in distributed_config.json are deliberate and are NOT used here.
RIGS = {
    "rrig6600":  dict(host="192.168.3.122", user="michael"),
    "rrig6600b": dict(host="192.168.3.156", user="michael"),
    "rrig6600c": dict(host="192.168.3.164", user="michael"),
}
RIG_PYTHON = "/home/michael/rocm_env/bin/python"
RIG_REPO = "/home/michael/distributed_prng_analysis"
COORDINATOR_ADDR = "192.168.3.177"

_OK = "OK "
_BAD = "BAD"


# ===========================================================================
# Worker targets
# ===========================================================================

class LocalCudaWorker:
    """The RTX 3080 Ti on VM 101 — the D6 3.B path, unchanged."""
    kind = "cuda-local"
    bind_host = "127.0.0.1"

    def __init__(self, device=0, spool=None):
        self.device = device
        self.node_id = socket.gethostname()
        self.spool = spool
        self.label = f"CUDA / local ({self.node_id} device {device})"
        self.proc = None

    def prepare(self):
        if self.spool:
            shutil.rmtree(self.spool, ignore_errors=True)
            os.makedirs(self.spool, exist_ok=True)

    def launch(self, port, caps, log):
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "miner.range_miner_worker",
             "--host", "127.0.0.1", "--port", str(port),
             "--gpu-id", str(self.device), "--device-index", str(self.device),
             "--miner-output-dir", str(self.spool),
             "--seed-cap-nvidia", str(caps), "--seed-cap-amd", str(caps),
             "--seed-cap-nvidia-hybrid", str(caps),
             "--seed-cap-amd-hybrid", str(caps),
             "--heartbeat-interval", "15"],
            cwd=_ROOT, stdout=log, stderr=subprocess.STDOUT)
        return self.proc

    def cleanup(self, port):
        try:
            self.proc.terminate(); self.proc.wait(timeout=30)
        except Exception:
            try: self.proc.kill()
            except Exception: pass
        return ""


class RemoteRigWorker:
    """One RX 6600 XT in a CT100, driven over SSH.

    Modelled on the Phase 6.0 `_RemoteRocmTarget`, but N of these run at once so
    a trial's stripes land on genuinely different physical machines — which is
    the whole point of Wall B's multi-rig leg.
    """
    kind = "rocm-remote"
    bind_host = "0.0.0.0"

    def __init__(self, node_id, host, user, device=0, spool=None):
        self.node_id = node_id
        self.host = host
        self.user = user
        self.device = device
        self.spool = spool or f"/home/michael/s172_wallb_spool_gpu{device}"
        self.label = f"ROCm / remote ({user}@{host} {node_id} device {device})"
        self.proc = None

    def _ssh(self):
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                "-o", "ServerAliveInterval=15", f"{self.user}@{self.host}"]

    def sh(self, cmd, timeout=180):
        r = subprocess.run(self._ssh() + [cmd], capture_output=True, text=True,
                           timeout=timeout)
        return (r.stdout or "") + (r.stderr or "")

    def identity(self):
        return self.sh(
            'hostname; '
            f'cd {RIG_REPO} && echo "worker_sha256=$(sha256sum '
            'miner/range_miner_worker.py | cut -d" " -f1)"; '
            'echo "registry_sha256=$(sha256sum prng_registry.py | cut -d" " -f1)"; '
            f'{RIG_PYTHON} -c "import cupy;rt=cupy.cuda.runtime;'
            'print(\'backend\', \'rocm\' if getattr(rt,\'is_hip\',False) else \'cuda\');'
            f'print(\'gcnArch\', rt.getDeviceProperties({self.device})[\'gcnArchName\'])"')

    def prepare(self):
        self.sh(f'rm -rf {self.spool} && mkdir -p {self.spool}')

    def launch(self, port, caps, log):
        remote = (
            f"cd {RIG_REPO} && exec {RIG_PYTHON} -m miner.range_miner_worker "
            f"--host {COORDINATOR_ADDR} --port {port} "
            f"--gpu-id {self.device} --device-index {self.device} "
            f"--miner-output-dir {self.spool} "
            f"--seed-cap-nvidia {caps} --seed-cap-amd {caps} "
            f"--seed-cap-nvidia-hybrid {caps} --seed-cap-amd-hybrid {caps} "
            f"--heartbeat-interval 15")
        self.proc = subprocess.Popen(self._ssh() + [remote], stdout=log,
                                     stderr=subprocess.STDOUT)
        return self.proc

    def cleanup(self, port):
        try:
            self.proc.terminate(); self.proc.wait(timeout=20)
        except Exception:
            try: self.proc.kill()
            except Exception: pass
        # Bracket trick + drop the shell's own PID: see the Phase 6.0 note.
        return self.sh(
            f'pkill -f "[r]ange_miner_worker.*--port {port}" >/dev/null 2>&1; '
            f'sleep 1; pgrep -af "[r]ange_miner_worker" | awk -v me=$$ "\\$1 != me" '
            f'| wc -l')


# ===========================================================================
# One trial arm
# ===========================================================================

def run_arm(*, label, workers, dataset_path, workdir, cfg, backend=None,
            backend_options=None, release_grade=False):
    """Drive ONE production trial with `workers` attached, then finalize.

    Returns (artifact, bundle, meta). Every worker advertises the SAME four
    seed caps, so `select_seed_cap` resolves identically on CUDA and ROCm and
    the sub-stripe boundaries — hence the canonical record ORDER — match by
    construction rather than by luck.
    """
    work = Path(workdir); work.mkdir(parents=True, exist_ok=True)
    staging = work / "miner_output"; staging.mkdir(parents=True, exist_ok=True)
    gen_root = work / "generation_root"; gen_root.mkdir(parents=True, exist_ok=True)

    config = WindowConfig(
        window_size=cfg["window_size"], offset=cfg["offset"],
        sessions=cfg["sessions"], skip_min=cfg["skip_min"],
        skip_max=cfg["skip_max"],
        forward_threshold=cfg["forward_threshold"],
        reverse_threshold=cfg["reverse_threshold"])

    _D6 = _d6()
    WOI = _woi()
    port = _D6._free_port()
    bind = "127.0.0.1" if all(w.kind == "cuda-local" for w in workers) else "0.0.0.0"
    coordinator = _D6._Coordinator(
        port=port, staging_dir=str(staging), seed_caps=cfg["seed_cap"],
        stripe_size=cfg["stripe_size"],
        substripes=max(1, cfg["stripe_size"] // cfg["seed_cap"]),
        backend=backend, backend_options=backend_options, miner_host=bind)
    # expected_workers follows the number of daemons actually attached; the
    # admission window (ee0db06) is bounded, so a miscount fails loudly rather
    # than hanging.
    coordinator.worker_pool_size = len(workers)

    accumulator = {"forward_count": 0, "reverse_count": 0, "bidirectional": []}
    holder = {}

    def _trial():
        try:
            t0 = time.time()
            holder["result"] = WOI.run_bidirectional_test(
                coordinator, config, dataset_path, cfg["seed_start"],
                cfg["seed_count"], prng_base=PRNG_BASE, test_both_modes=False,
                forward_threshold=cfg["forward_threshold"],
                reverse_threshold=cfg["reverse_threshold"],
                trial_number=1, accumulator=accumulator)
            holder["elapsed"] = time.time() - t0
        except Exception:
            holder["err"] = traceback.format_exc()

    print(f"\n[ARM {label}] bind {bind}:{port}  workers={len(workers)}  "
          f"backend={backend or 'serial_reference (default)'}")
    for w in workers:
        print(f"          - {w.label}")
    t = threading.Thread(target=_trial, name=f"arm-{label}", daemon=True)
    t.start()
    time.sleep(2.0)

    logs = []
    for i, w in enumerate(workers):
        w.spool = w.spool or str(staging / f"spool{i}")
        w.prepare()
        lf = open(work / f"worker_{w.node_id}_{i}.log", "w")
        logs.append(lf)
        w.launch(port, cfg["seed_cap"], lf)

    deadline = time.time() + 1800
    while t.is_alive() and time.time() < deadline:
        time.sleep(2.0)
    t.join(timeout=60)
    alive = [w.proc.poll() is None for w in workers]
    for w in workers:
        w.cleanup(port)
    for lf in logs:
        lf.close()

    if t.is_alive():
        raise AssertionError(f"arm {label}: the miner trial did not terminate")
    if "err" in holder:
        for w, i in zip(workers, range(len(workers))):
            p = work / f"worker_{w.node_id}_{i}.log"
            if p.exists():
                print(f"\n[WORKER LOG {w.node_id}]\n{p.read_text()[-3000:]}")
        raise AssertionError(f"arm {label} failed:\n{holder['err']}")

    tr = holder["result"]
    print(f"[ARM {label}] {holder['elapsed']:.1f}s  forward={tr.forward_count:,} "
          f"reverse={tr.reverse_count:,} bidirectional={tr.bidirectional_count:,} "
          f"raw_candidates={len(accumulator['bidirectional']):,}")
    prov = _D6._report_threshold_provenance(staging, cfg["forward_threshold"],
                                            cfg["reverse_threshold"])

    # --- finalize (same call shape as run-level Step-1 finalization) -------
    snap = _source_snapshot(work / "source_snapshot")
    commit, clean = WOI._repository_state(repo_root=str(snap))
    artifact = RF.finalize_run(
        accumulator["bidirectional"], output_root=gen_root,
        run_id=f"step1_{PRNG_BASE}_{int(cfg['seed_start'])}",
        prng_base=PRNG_BASE, skip_modes_executed=("constant",),
        seed_start=int(cfg["seed_start"]), seed_count=int(cfg["seed_count"]),
        repository_commit=commit, repository_tree_clean=clean)
    with np.load(artifact.binary_npz_path) as npz:
        bundle = {k: npz[k] for k in npz.files}
        order = tuple(npz.files)
    meta = {
        "label": label,
        "workers": [{"node_id": w.node_id, "kind": w.kind, "device": w.device,
                     "label": w.label} for w in workers],
        "worker_alive_through_trial": alive,
        "assembly_backend": backend or "serial_reference (default)",
        "assembly_backend_options": backend_options,
        "bind_host": bind,
        "elapsed_s": round(holder["elapsed"], 1),
        "forward_count": int(tr.forward_count),
        "reverse_count": int(tr.reverse_count),
        "bidirectional_count": int(tr.bidirectional_count),
        "raw_candidates": len(accumulator["bidirectional"]),
        "generation_id": artifact.generation_id,
        "artifact_sha256": artifact.artifact_sha256,
        "sidecar_sha256": artifact.sidecar_sha256,
        "final_row_count": int(artifact.final_row_count),
        "binary_npz_path": str(artifact.binary_npz_path),
        "array_order": list(order),
        "threshold_provenance": prov,
        "snapshot_commit": commit,
    }
    print(f"[ARM {label}] generation {artifact.generation_id} "
          f"artifact_sha256={artifact.artifact_sha256} rows={artifact.final_row_count}")
    return artifact, bundle, meta


def _source_snapshot(dest: Path) -> Path:
    """HEAD's tracked files + working-tree .py overlay, in a throwaway repo.

    Identical in construction to the D6 smoke's snapshot, and for the same
    reason: `finalize_run` refuses a dirty tree and an agent may not commit.
    The recorded SHA identifies a tree byte-identical to the source that ran; it
    is NOT the project's own commit. `--release-grade` is the mode that certifies
    against the real commit, and it is Michael's to run after committing.
    """
    dest.mkdir(parents=True, exist_ok=True)
    tar = subprocess.run(["git", "-C", _ROOT, "archive", "HEAD"], check=True,
                         capture_output=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=tar.stdout, check=True)
    status = subprocess.run(["git", "-C", _ROOT, "status", "--porcelain"],
                            check=True, capture_output=True, text=True).stdout
    for line in status.splitlines():
        rel = line[3:].strip()
        if not rel.endswith(".py"):
            continue
        src = Path(_ROOT) / rel
        if not src.is_file():
            continue
        dst = dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for cmd in (["git", "init", "-q"],
                ["git", "-c", "user.email=w@l", "-c", "user.name=w", "add", "-A"],
                ["git", "-c", "user.email=w@l", "-c", "user.name=w", "commit",
                 "-q", "-m", "wall_ab source snapshot"]):
        subprocess.run(cmd, cwd=str(dest), check=True, capture_output=True)
    return dest


# ===========================================================================
# WALL A
# ===========================================================================

def wall_a(artifact, bundle, workdir, dataset_path, results: dict) -> bool:
    """The six consumer legs, each with a stated failure mode."""
    work = Path(workdir)
    ok = True
    legs = {}

    def leg(name, passed, detail, would_fail_if):
        nonlocal ok
        ok = ok and passed
        legs[name] = {"pass": bool(passed), "detail": detail,
                      "would_fail_if": would_fail_if}
        print(f"  {_OK if passed else _BAD}  {name:<44} {detail}")

    print("\n" + "=" * 78)
    print("WALL A — INTERFACE AND CONSUMER (§1)")
    print("=" * 78)
    print(f"  subject: miner-produced certified generation "
          f"{artifact.generation_id}")
    print(f"  artifact_sha256: {artifact.artifact_sha256}")
    print(f"  rows: {artifact.final_row_count:,}\n")

    # --- A1 frozen 22-array contract -------------------------------------
    with np.load(artifact.binary_npz_path) as npz:
        order = tuple(npz.files)
    n = len(bundle["seeds"])
    shapes = {k: tuple(v.shape) for k, v in bundle.items()}
    dtypes = {k: str(v.dtype) for k, v in bundle.items()}
    a1 = (order == ORACLE_ARRAY_NAMES
          and all(shapes[k] == (n,) for k in ORACLE_ARRAY_NAMES))
    leg("A1 frozen 22-array contract", a1,
        f"{len(order)} arrays, order == frozen oracle, all shaped ({n},)",
        "an array is added, removed, renamed, reordered or not per-row")
    legs["A1 frozen 22-array contract"]["dtypes"] = dtypes

    # --- A2 validate_array_bundle ----------------------------------------
    try:
        validate_array_bundle(bundle)
        leg("A2 validate_array_bundle()", True, "passed",
            "dtype/length/NaN/contract violation in any of the 22 arrays")
    except Exception as e:
        leg("A2 validate_array_bundle()", False, f"raised {type(e).__name__}: {e}",
            "dtype/length/NaN/contract violation in any of the 22 arrays")

    # --- A3 Step-2 loader -------------------------------------------------
    loaded = load_survivors(str(artifact.binary_npz_path))
    a3 = (loaded.format == "npz" and loaded.fallback_used is False
          and loaded.count == artifact.final_row_count)
    leg("A3 Step-2 loader, fallback_used=False", a3,
        f"format={loaded.format} npz_version={loaded.npz_version} "
        f"count={loaded.count:,} fallback_used={loaded.fallback_used}",
        "the loader cannot read the NPZ and silently falls back to JSON")

    # --- A4 NPZ -> dict, every field preserved ----------------------------
    from generate_step3_scoring_jobs import extract_survivors_full, chunk_list
    records = extract_survivors_full(loaded.data)
    expected_fields = set(ORACLE_ARRAY_NAMES) - {"seeds"} | {"seed"}
    got_fields = set(records[0]) if records else set()
    a4 = bool(records) and got_fields == expected_fields and len(records) == n
    leg("A4 NPZ->dict preserves every field", a4,
        f"{len(records):,} records x {len(got_fields)} fields; "
        f"missing={sorted(expected_fields - got_fields)} "
        f"extra={sorted(got_fields - expected_fields)}",
        "extract_survivors_full drops a field — this is the exact regression "
        "that once left 14/47 ML features at zero")
    legs["A4 NPZ->dict preserves every field"]["fields"] = sorted(got_fields)

    # A4 round-trip: every value, every record, against the arrays.
    mismatched = []
    for i, rec in enumerate(records):
        for name in ORACLE_ARRAY_NAMES:
            key = "seed" if name == "seeds" else name
            if rec[key] != bundle[name][i].item():
                mismatched.append((i, name, rec[key], bundle[name][i].item()))
                if len(mismatched) > 5:
                    break
        if len(mismatched) > 5:
            break
    leg("A4b NPZ->dict value round-trip", not mismatched,
        f"{len(records) * 22:,} values compared, {len(mismatched)} mismatched"
        + (f" e.g. {mismatched[:2]}" if mismatched else ""),
        "a value is coerced, truncated or reordered in the conversion")

    # --- A5 Step-3 chunk generation ---------------------------------------
    chunk_dir = work / "wall_a_chunks"
    shutil.rmtree(chunk_dir, ignore_errors=True)
    chunk_dir.mkdir(parents=True)
    chunk_size = max(1, n // 3 + 1)
    chunks = chunk_list(records, chunk_size)
    paths = []
    for i, ch in enumerate(chunks):
        p = chunk_dir / f"chunk_{i:04d}.json"
        p.write_text(json.dumps(ch))
        paths.append(p)
    rt = []
    for p in paths:
        rt.extend(json.loads(p.read_text()))
    a5 = (len(rt) == len(records)
          and all(set(r) == got_fields for r in rt)
          and rt == records)
    leg("A5 Step-3 chunk generation", a5,
        f"{len(chunks)} chunks of <= {chunk_size:,}; {len(rt):,} records read "
        f"back; field set preserved in every chunk; byte round-trip equal",
        "chunking drops records, drops fields, or reorders them")

    # --- A6 survivors_with_scores smoke -----------------------------------
    out = work / "wall_a_scored.json"
    cmd = [sys.executable, "full_scoring_worker.py",
           "--seeds-file", str(paths[0]), "--train-history", dataset_path,
           "--output-file", str(out), "--prng-type", PRNG_BASE,
           "--mod", "1000", "--gpu-id", "0"]
    r = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True,
                       timeout=1800)
    (work / "wall_a_scoring.log").write_text((r.stdout or "") + (r.stderr or ""))
    if r.returncode != 0 or not out.exists():
        leg("A6 survivors_with_scores, no metadata loss", False,
            f"full_scoring_worker exited {r.returncode}; see wall_a_scoring.log",
            "Step 3 cannot consume a miner chunk at all")
    else:
        scored = json.loads(out.read_text())
        sieve_fields = ("forward_count", "reverse_count", "bidirectional_count",
                        "skip_min", "skip_max", "skip_range")
        feats = scored[0].get("features", {}) if scored else {}
        present = [f for f in sieve_fields if f in feats]
        # The value check is what makes this a metadata-LOSS test rather than a
        # key-presence test: the sieve metadata must arrive with the miner's
        # values, not with a scorer default.
        by_seed = {r_["seed"]: r_ for r_ in scored}
        val_bad = []
        for rec in chunks[0]:
            f = by_seed.get(rec["seed"], {}).get("features", {})
            for name in sieve_fields:
                if name in f and float(f[name]) != float(rec[name]):
                    val_bad.append((rec["seed"], name, f[name], rec[name]))
        a6 = (len(scored) == len(chunks[0]) and len(present) == len(sieve_fields)
              and not val_bad)
        leg("A6 survivors_with_scores, no metadata loss", a6,
            f"{len(scored):,} scored, {len(feats)} features/record, "
            f"sieve metadata present={present}, "
            f"value mismatches={len(val_bad)}",
            "the sieve metadata does not survive the chunk->scorer boundary, "
            "or arrives with scorer defaults instead of miner values")
        legs["A6 survivors_with_scores, no metadata loss"]["feature_count"] = len(feats)

    # --- fault-injection control for Wall A -------------------------------
    print("\n  FAULT-INJECTION CONTROL (VIR-2) — each leg must reject a broken "
          "bundle:")
    faults = []

    def fault(name, fn, desc):
        try:
            fn()
            rejected = False
            det = "NOT REJECTED — the leg accepted a broken bundle"
        except Exception as e:
            rejected = True
            det = f"rejected with {type(e).__name__}"
        faults.append({"id": name, "description": desc, "rejected": rejected,
                       "evidence": det})
        print(f"    {_OK if rejected else _BAD}  {name:<34} {desc}")
        print(f"           -> {det}")

    def _drop_array():
        b = dict(bundle); b.pop("reverse_matches")
        validate_array_bundle(b)
    fault("FA1_missing_array", _drop_array,
          "remove `reverse_matches` from the 22-array bundle")

    def _wrong_dtype():
        b = dict(bundle); b["score"] = b["score"].astype(np.float64)
        validate_array_bundle(b)
    fault("FA2_wrong_dtype", _wrong_dtype, "widen `score` from float32 to float64")

    def _ragged():
        b = dict(bundle); b["seeds"] = b["seeds"][:-1]
        validate_array_bundle(b)
    fault("FA3_ragged_length", _ragged, "truncate `seeds` by one row")

    def _metadata_loss():
        # The A4 guardrail: a bundle carrying only `seeds` must be refused.
        extract_survivors_full({"seeds": bundle["seeds"]})
    fault("FA4_metadata_loss", _metadata_loss,
          "hand extract_survivors_full a seeds-only bundle (the 14/47-features "
          "regression shape)")

    def _json_fallback():
        # Step-2 loader pointed at a path with no NPZ: it must not silently
        # succeed with fallback_used=False.
        res = load_survivors(str(work / "does_not_exist.npz"))
        if res.fallback_used is False:
            raise AssertionError("loader reported fallback_used=False on a "
                                 "missing NPZ — this control is the failure")
    fault("FA5_loader_fallback_visible", _json_fallback,
          "point the Step-2 loader at a missing NPZ; a silent "
          "fallback_used=False would be undetectable")

    all_faults_ok = all(f["rejected"] for f in faults)
    results["wall_a"] = {"legs": legs, "faults": faults,
                         "generation_id": artifact.generation_id,
                         "artifact_sha256": artifact.artifact_sha256,
                         "rows": int(artifact.final_row_count)}
    sentinel = "PASS" if (ok and all_faults_ok) else "FAIL"
    results["wall_a"]["sentinel"] = sentinel
    print(f"\n  WALL A SENTINEL: {sentinel}   "
          f"(legs {sum(1 for v in legs.values() if v['pass'])}/{len(legs)}, "
          f"faults rejected {sum(1 for f in faults if f['rejected'])}/{len(faults)})")
    return sentinel == "PASS"


# ===========================================================================
# WALL B comparison
# ===========================================================================

def compare_bundles(a_meta, a_bundle, b_meta, b_bundle, title) -> dict:
    """22-row field-for-field matrix. A divergence is LOCALIZED, never summarised
    to a boolean: report the array, the first differing index, and both values."""
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)
    print(f"  A: {a_meta['label']:<28} artifact_sha256={a_meta['artifact_sha256']}")
    print(f"  B: {b_meta['label']:<28} artifact_sha256={b_meta['artifact_sha256']}")
    rows = []
    all_equal = True
    for i, name in enumerate(ORACLE_ARRAY_NAMES, 1):
        ca, ra = a_bundle.get(name), b_bundle.get(name)
        if ca is None or ra is None:
            all_equal = False
            rows.append({"array": name, "equal": False, "note": "MISSING"})
            print(f"  {i:>2}  {name:<28} MISSING")
            continue
        eq = bool(np.array_equal(ca, ra))
        note = ""
        if not eq:
            all_equal = False
            if ca.shape != ra.shape:
                note = f"SHAPE {ca.shape} vs {ra.shape}"
            else:
                d = np.flatnonzero(ca != ra)
                j = int(d[0])
                note = (f"first differing index {j}: A={ca[j]!r} B={ra[j]!r} "
                        f"({d.size} of {ca.size} differ)")
        rows.append({"array": name, "equal": eq, "note": note})
        print(f"  {i:>2}  {name:<28} {str(ca.dtype):<9} {ca.shape[0]:>6}  "
              f"{str(eq):<6} {note}")
    order_eq = (tuple(a_meta["array_order"]) == tuple(b_meta["array_order"])
                == ORACLE_ARRAY_NAMES)
    sha_eq = a_meta["artifact_sha256"] == b_meta["artifact_sha256"]
    print(f"  canonical ORDER identical to frozen oracle on both: {order_eq}")
    print(f"  artifact_sha256 identical:                          {sha_eq}")
    verdict = all_equal and order_eq and sha_eq
    print(f"  VERDICT: {'IDENTICAL SEMANTIC ARTIFACT' if verdict else 'DIVERGENT'}")
    return {"title": title, "a": a_meta["label"], "b": b_meta["label"],
            "arrays": rows, "all_arrays_equal": all_equal,
            "order_equal": order_eq, "artifact_sha256_equal": sha_eq,
            "identical": verdict}


# ===========================================================================
# Provenance binding (§2)
# ===========================================================================

def provenance_block(frozen, cfg, arms) -> dict:
    tracked_dirty = subprocess.run(
        ["git", "-C", _ROOT, "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True).stdout.strip().splitlines()
    untracked = subprocess.run(
        ["git", "-C", _ROOT, "ls-files", "--others", "--exclude-standard"],
        capture_output=True, text=True).stdout.strip().splitlines()
    commit, clean_incl_untracked = _woi()._repository_state(repo_root=_ROOT)
    with open(POINTER_MANIFEST, "rb") as f:
        pointer_sha = hashlib.sha256(f.read()).hexdigest()
    prov = {
        "repository_commit": commit,
        "repository_tracked_clean": not tracked_dirty,
        "repository_tracked_dirty_paths": tracked_dirty,
        "repository_untracked_count": len(untracked),
        "repository_clean_including_untracked": clean_incl_untracked,
        "dataset_pointer_manifest": POINTER_MANIFEST,
        "dataset_pointer_manifest_sha256": pointer_sha,
        "dataset_lineage_id": getattr(frozen, "lineage_id", None),
        "dataset_version_id": getattr(frozen, "version_id", None),
        "dataset_frozen_path": frozen.path,
        "dataset_sha256": frozen.sha256,
        "dataset_size_bytes": getattr(frozen, "size_bytes", None),
        "dataset_record_count": getattr(frozen, "record_count", None),
        "seed_domain": {"seed_start": cfg["seed_start"],
                        "seed_count": cfg["seed_count"],
                        "exact_range": [cfg["seed_start"],
                                        cfg["seed_start"] + cfg["seed_count"]],
                        "contract": "contiguous uint64 half-open [start, start+count)"},
        "prng_family": PRNG_BASE,
        "prng_variants_executed": ["java_lcg (phase 1, forward constant)",
                                   "java_lcg_reverse (phase 2, reverse constant)"],
        "window_selection": {"window_size": cfg["window_size"],
                             "offset": cfg["offset"],
                             "sessions": cfg["sessions"]},
        "thresholds": {
            "requested": {"forward": cfg["forward_threshold"],
                          "reverse": cfg["reverse_threshold"]},
            "payload": {a["label"]: a["threshold_provenance"].get("payload")
                        for a in arms},
            "effective": {a["label"]: a["threshold_provenance"].get("effective")
                          for a in arms},
        },
        "skip_mode": "constant (test_both_modes=False)",
        "effective_skip_semantics": (
            f"kernel skip range from the assignment payload, default [0,16]; "
            f"trial metadata skip_min={cfg['skip_min']} skip_max={cfg['skip_max']}; "
            f"skip burned BEFORE the first draw and BETWEEN every subsequent "
            f"pair (inter-draw). Constant-skip only — the hybrid kernels' "
            f"skip bounds are dead (skill 2.7 #4) and no hybrid phase ran here."),
        "assembly_backends_exercised": sorted({a["assembly_backend"] for a in arms}),
    }
    print("\n" + "=" * 78)
    print("PROVENANCE BINDING (§2)")
    print("=" * 78)
    for k, v in prov.items():
        if isinstance(v, (dict, list)):
            print(f"  {k}:")
            print("      " + json.dumps(v, indent=2, default=str).replace("\n", "\n      "))
        else:
            print(f"  {k}: {v}")
    return prov


# ===========================================================================
# Main
# ===========================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--wall-a", action="store_true")
    ap.add_argument("--wall-b-repeat", action="store_true")
    ap.add_argument("--wall-b-backend", action="store_true")
    ap.add_argument("--wall-b-multirig", action="store_true")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--seed-count", type=int, default=8_000_000)
    ap.add_argument("--stripe-size", type=int, default=4_000_000)
    ap.add_argument("--seed-cap", type=int, default=1_000_000)
    ap.add_argument("--window-size", type=int, default=3)
    ap.add_argument("--forward-threshold", type=float, default=0.31)
    ap.add_argument("--reverse-threshold", type=float, default=0.47)
    ap.add_argument("--shard-pool-size", type=int, default=4)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    if args.all:
        args.wall_a = args.wall_b_repeat = args.wall_b_backend = True
        args.wall_b_multirig = True

    workdir = Path(args.workdir or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "s172_phase6_wall_ab"))
    workdir.mkdir(parents=True, exist_ok=True)

    cfg = dict(seed_start=args.seed_start, seed_count=args.seed_count,
               stripe_size=args.stripe_size, seed_cap=args.seed_cap,
               window_size=args.window_size, offset=0,
               sessions=["midday", "evening"], skip_min=0, skip_max=16,
               forward_threshold=args.forward_threshold,
               reverse_threshold=args.reverse_threshold)

    results: Dict[str, Any] = {
        "gate": "s172_bounded_phase6_wall_a_wall_b",
        "authority": "docs/CLAUDE_CODE_INSTRUCTIONS_BOUNDED_PHASE_6.md §1,§2",
        "config": cfg,
    }

    # --- P0.5 run-start dataset gate --------------------------------------
    print("=" * 78)
    print("S172 BOUNDED PHASE 6 — WALL A / WALL B GATE")
    print("=" * 78)
    # The local hand-transcribed oracle must agree with the D6 smoke's copy;
    # two transcriptions that disagree would silently weaken A1 and every Wall-B
    # comparison. Checked here (after the cupy-free import window closes) rather
    # than at module level.
    assert ORACLE_ARRAY_NAMES == _d6().ORACLE_ARRAY_NAMES, (
        "the 22-array oracle in wall_ab_gate.py disagrees with the D6 3.B "
        "smoke's transcription")
    print("[ORACLE] 22-array oracle agrees with the D6 3.B smoke transcription")

    from miner import dataset_authority as DA
    frozen = DA.run_start_dataset_gate(
        DATASET_ALIAS, run_label=f"wall_ab_{os.getpid()}",
        miner_backed=True, remote_execution=True)
    print(f"\n[P0.5] dataset FROZEN: {frozen.describe()}")
    dataset_path = frozen.path

    arms: List[dict] = []
    bundles: Dict[str, Any] = {}
    metas: Dict[str, dict] = {}
    ok = True

    def arm(name, workers, backend=None, backend_options=None):
        art, bun, meta = run_arm(
            label=name, workers=workers, dataset_path=dataset_path,
            workdir=workdir / name, cfg=cfg, backend=backend,
            backend_options=backend_options)
        bundles[name] = bun
        metas[name] = meta
        arms.append(meta)
        return art, bun, meta

    # --- baseline arm (also Wall A's subject) -----------------------------
    art1, bun1, m1 = arm("cuda_run1", [LocalCudaWorker(0)])

    if args.wall_a:
        ok = wall_a(art1, bun1, workdir / "wall_a", dataset_path, results) and ok

    wall_b: Dict[str, Any] = {"comparisons": [], "cited": []}

    # --- B1 CUDA vs ROCm: CITED, NOT RE-RUN -------------------------------
    wall_b["cited"].append({
        "leg": "B1 CUDA vs ROCm",
        "status": "CITED — NOT RE-RUN IN THIS SESSION",
        "source": "docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md, commit 23fa413",
        "claim": ("identical artifact_sha256 "
                  "0e0092feeb02e22d28557ddf4d8e421941d6117bcc0448d7f7323ec402c1c4b0 "
                  "across the D6 release-grade generation and both Phase 6.0 runs; "
                  "22/22 arrays field-for-field equal; 398,156 / 383 / 319 "
                  "forward/reverse/bidirectional; no GPU reset and no "
                  "GCVM_L2_PROTECTION_FAULT in the host kernel log"),
        "caveat": ("the Phase 6.0 ROCm run executed the PRE-P0.5 worker "
                   "(miner/range_miner_worker.py at 8e2f5bf). The current "
                   "worker at HEAD is a different file. This citation is "
                   "therefore evidence about the KERNEL and the platform, not "
                   "about the current worker source."),
    })
    print("\n" + "=" * 78)
    print("WALL B — DETERMINISM AND PLATFORM (§2)")
    print("=" * 78)
    for c in wall_b["cited"]:
        print(f"\n  {c['leg']}: {c['status']}")
        print(f"    source : {c['source']}")
        print(f"    claim  : {c['claim']}")
        print(f"    caveat : {c['caveat']}")

    # --- B2 repeat vs repeat ---------------------------------------------
    if args.wall_b_repeat:
        _a2, bun2, m2 = arm("cuda_run2", [LocalCudaWorker(0)])
        c = compare_bundles(m1, bun1, m2, bun2,
                            "B2 REPEATED RUN vs REPEATED RUN (fresh, CUDA)")
        wall_b["comparisons"].append(c); ok = ok and c["identical"]

    # --- B3 serial_reference vs process_sharded ---------------------------
    if args.wall_b_backend:
        _a3, bun3, m3 = arm("cuda_sharded", [LocalCudaWorker(0)],
                            backend="process_sharded",
                            backend_options={"pool_size": args.shard_pool_size})
        c = compare_bundles(m1, bun1, m3, bun3,
                            "B3 serial_reference vs process_sharded (fresh, CUDA)")
        wall_b["comparisons"].append(c); ok = ok and c["identical"]

    # --- B4 multi-rig node-assignment independence ------------------------
    if args.wall_b_multirig:
        print("\n  B4 MULTI-RIG — the leg with NO prior evidence (Phase 6.0 was "
              "single-rig).")
        rig_ids = list(RIGS)
        for rid in rig_ids:
            w = RemoteRigWorker(rid, **RIGS[rid])
            print(f"\n  ---- {rid} identity ----")
            print("    " + w.identity().strip().replace("\n", "\n    "))
        # Arm M1: rigs a+b.  Arm M2: rigs b+c.  Different physical machines,
        # different node assignment, identical everything else.
        wa = [RemoteRigWorker("rrig6600", **RIGS["rrig6600"]),
              RemoteRigWorker("rrig6600b", **RIGS["rrig6600b"])]
        _a4, bun4, m4 = arm("rigs_ab", wa)
        wb = [RemoteRigWorker("rrig6600b", **RIGS["rrig6600b"]),
              RemoteRigWorker("rrig6600c", **RIGS["rrig6600c"])]
        _a5, bun5, m5 = arm("rigs_bc", wb)
        c = compare_bundles(m4, bun4, m5, bun5,
                            "B4a MULTI-RIG {a,b} vs {b,c} — node-assignment "
                            "independence (fresh)")
        wall_b["comparisons"].append(c); ok = ok and c["identical"]
        c = compare_bundles(m1, bun1, m4, bun4,
                            "B4b SINGLE CUDA GPU vs TWO ROCm RIGS — engine "
                            "output independent of fleet shape AND platform (fresh)")
        wall_b["comparisons"].append(c); ok = ok and c["identical"]

    # --- provenance + fault injection for Wall B --------------------------
    results["provenance"] = provenance_block(frozen, cfg, arms)
    results["arms"] = arms

    print("\n  FAULT-INJECTION CONTROL for Wall B (VIR-2): the comparator must "
          "reject a\n  divergent artifact. Perturb one value in a copy of the "
          "baseline bundle:")
    faulted = {k: v.copy() for k, v in bun1.items()}
    if len(faulted["seeds"]):
        faulted["forward_matches"] = faulted["forward_matches"].copy()
        faulted["forward_matches"][0] = faulted["forward_matches"][0] + 1
    fm = dict(m1); fm["label"] = "cuda_run1_FAULTED"
    # Flip the last hex nibble to a DIFFERENT value — `[:-1] + "0"` is a no-op
    # whenever the digest already ends in 0, which would leave the injected
    # "digest differs" condition silently unexercised.
    _last = m1["artifact_sha256"][-1]
    fm["artifact_sha256"] = m1["artifact_sha256"][:-1] + ("1" if _last != "1" else "2")
    cf = compare_bundles(m1, bun1, fm, faulted,
                         "FB1 fault injection: one forward_matches value bumped")
    wall_b["fault_injection"] = {
        "id": "FB1_one_value_perturbed",
        "rejected": not cf["identical"],
        "evidence": f"all_arrays_equal={cf['all_arrays_equal']} "
                    f"artifact_sha256_equal={cf['artifact_sha256_equal']}",
    }
    print(f"  {_OK if not cf['identical'] else _BAD}  "
          f"FB1 comparator rejects a one-value divergence: "
          f"{not cf['identical']}")
    ok = ok and (not cf["identical"])

    sentinel = "PASS" if ok else "FAIL"
    wall_b["sentinel"] = sentinel if (args.wall_b_repeat or args.wall_b_backend
                                      or args.wall_b_multirig) else "UNAVAILABLE"
    results["wall_b"] = wall_b

    print("\n" + "=" * 78)
    print("BOUNDED PHASE 6 — WALL A / WALL B")
    print(f"  Wall A sentinel : {results.get('wall_a', {}).get('sentinel', 'UNAVAILABLE')}")
    print(f"  Wall B sentinel : {wall_b['sentinel']}")
    for c in wall_b["comparisons"]:
        print(f"    {_OK if c['identical'] else _BAD}  {c['title']}")
    print("=" * 78)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(results, indent=2, sort_keys=True, default=str))
        print(f"[RECORD] {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        print("\nSENTINEL: FAIL (unhandled exception)")
        sys.exit(2)
