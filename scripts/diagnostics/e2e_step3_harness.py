#!/usr/bin/env python3
"""
scripts/diagnostics/e2e_step3_harness.py -- NON-CERTIFYING DIAGNOSTIC (e2e thin path).

Runs Step 3 (Full Scoring) IN-PROCESS on an isolated run copy built by
scripts/diagnostics/e2e_build_run_copy.py. Route: docs/drafts/e2e_v1/
MAP_E2E_THIN_PATH_20261122.md §2.1 (4), harness H3; lead task
logs/e2e_session/TASK_STEP3.txt item 2.

WHAT IT CALLS (unchanged production code, imported; signatures checked first):
  generate_step3_scoring_jobs.generate_jobs  -- the chunker. It runs with cwd =
      --out-dir, so it writes scoring_chunks/ and full_scoring_results/ there.
      Its load_survivors is bound with allow_fallback=False.
  full_scoring_worker.load_survivors_with_metadata, .load_lottery_history,
      .score_survivors -- exactly the calls full_scoring_worker.main() makes
      (full_scoring_worker.py:577-656), one chunk per fresh spawned process.
  models.global_state_tracker.GlobalStateTracker -- for the aggregation.

WHAT IT REPRODUCES (TODO-E2E-3): the inline aggregation heredoc of
  run_step3_full_scoring.sh (the AGGREGATE_EOF block, :275-442; the
  map and the task cite :299-408, which lies inside it) as
  aggregate_like_launcher() below, and the Phase 6 floor (VALIDATE_EOF,
  :451-489). The self-test runs the launcher's own heredoc text on the same
  chunk files and requires a byte-identical survivors_with_scores.json.

WHAT IT NEVER CALLS OR TOUCHES: run_step3_full_scoring.sh itself,
  ramdisk_preload.sh, /dev/shm/prng, scripts_coordinator.py (its
  output-path hardcode, scripts_coordinator.py:499-500), distributed_worker.py,
  scp/ssh, any rig, the main tree (write), ~/miner_staging, the run H tree.

GUARDS (parent and every chunk process):
  * sockets: connect / connect_ex / create_connection raise and are counted.
  * subprocess.Popen/run/call/check_output and os.system raise and are counted
    (the parent's one read-only git query runs before the guard).
  * cupy / cupyx / pycuda / jax / tensorflow imports raise. torch is allowed:
    SurvivorScorer needs it (survivor_scorer.py:117).
  * --device cpu (default): CUDA_VISIBLE_DEVICES="" and HIP_VISIBLE_DEVICES=""
    before torch is imported; the chunk FAILS if torch still sees a GPU.
  * no .pyc writes (PYTHONDONTWRITEBYTECODE, sys.dont_write_bytecode).

FAIL-CLOSED (stricter than production; each relaxable only by a named flag):
  * batch path fell back to the sequential path (6 of 18 metadata fields, Ch4
    B6 / F-PARITY): detected by the result shape (no metadata.worker_gpu,
    full_scoring_worker.py:394-401 vs :465-466) and by the log line.
    --allow-sequential-fallback.
  * the torch generator fell back to _cpu_batch_generate (unreduced values,
    Ch4 B4, survivor_scorer.py:575-581, :802-808): log line, or pred_max > mod-1.
    --allow-cpu-generate-fallback.
  * holdout_quality failed and was written as 0.0 (full_scoring_worker.py:424-430, :485-491):
    log line, or empty holdout_features. --allow-holdout-quality-failure.
  * a per-seed error record (full_scoring_worker.py:493-495). --allow-seed-errors.
  * global features could not be computed (the launcher swallows this,
    run_step3_full_scoring.sh:355-357). --allow-global-feature-failure.
  * any chunk failing for any reason: no aggregate is written. No flag.

NOTHING IS REMOVED: every row of every selected chunk goes through and is
kept; the aggregate holds exactly the input seeds (checked). --chunks selects
a subset for a smoke run; the output is then named survivors_with_scores.PARTIAL.json
so no later step can mistake it for a full Step 3 output.

Step 3 output is features + labels, not a hit-rate result (yardstick,
docs/PITFALL_STATE_RECOVERY_YARDSTICK.md). Nothing here is certified.
"""
from __future__ import annotations

import sys as _sys
_sys.dont_write_bytecode = True

import argparse
import datetime as _dt
import functools
import hashlib
import inspect
import json
import logging
import math
import multiprocessing as mp
import os
import re
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

HARNESS_VERSION = "1.0.0-draft-20261123"
LABEL = "NON-CERTIFYING DIAGNOSTIC"
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BLOCKED_MODULES = ("cupy", "cupyx", "pycuda", "jax", "tensorflow")
PARTIAL_NAME = "survivors_with_scores.PARTIAL.json"
FULL_NAME = "survivors_with_scores.json"
FAILED_PREFIX = "FAILED_"            # never matches the aggregator's chunk_*.json glob

# The production signatures this harness was written against (41ad509).
# A change refuses the run rather than calling with a stale contract.
EXPECTED_SIGNATURES = {
    ("generate_step3_scoring_jobs", "generate_jobs"): [
        "survivors_file", "train_history_file", "holdout_history_file", "config_file",
        "chunk_size", "prng_type", "mod", "remote_base_path", "timeout",
        "forward_survivors_file", "reverse_survivors_file"],
    ("full_scoring_worker", "score_survivors"): [
        "seeds", "train_history", "prng_type", "mod", "batch_size", "forward_survivors",
        "reverse_survivors", "scorer_class", "survivor_metadata", "holdout_history"],
    # LL-2 (ch14-learn W1-a) appends LL2_SIGNATURE_TAIL; both forms are accepted (see check_signatures)
    ("full_scoring_worker", "load_survivors_with_metadata"): ["seeds_file"],
    ("full_scoring_worker", "load_lottery_history"): ["history_file"],
}

# Log lines emitted by the production fallbacks (matched as substrings).
LOG_BATCH_FALLBACK = "[BATCH-GPU] Batch processing failed"         # full_scoring_worker.py:439
LOG_CPU_GENERATE = "PyTorch GPU batch failed"                       # survivor_scorer.py:577
LOG_HQ_FAILED = ("[S111] holdout_quality failed", "[S111] holdout_quality fallback failed")
LOG_SEED_FAILED = "Failed to score seed"                            # full_scoring_worker.py:494

_GUARD = {"socket_attempts": 0, "subprocess_attempts": 0, "blocked_imports": []}

# Self-test mutants (scripts/diagnostics/e2e_step3_selftest.py). Active ONLY when
# this environment variable is set; the name is written into every record and
# printed, so a mutated run can never pass for a real one.
MUTANT_ENV = "E2E_STEP3_SELFTEST_MUTANT"
CHUNK_MUTANTS = ("batch_raise", "torch_generate_raise", "holdout_quality_raise", "seed_raise")
PARENT_MUTANTS = ("agg_drop_row", "agg_drop_feature", "agg_sort_ascending")

sys.path.insert(0, str(HERE))
import e2e_build_run_copy as _builder  # noqa: E402  (path checks shared with the builder)


class Refused(Exception):
    pass


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------
class _BlockGPULibs:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED_MODULES:
            _GUARD["blocked_imports"].append(name)
            raise ImportError(f"[{LABEL}] import blocked: {name}")
        return None


def install_guards() -> None:
    for m in BLOCKED_MODULES:
        if m in sys.modules:
            raise RuntimeError(f"blocked module {m} already imported before the guard")
    sys.meta_path.insert(0, _BlockGPULibs())

    import socket

    def _no_net(*a, **k):
        _GUARD["socket_attempts"] += 1
        raise RuntimeError(f"[{LABEL}] network access blocked")

    socket.socket.connect = _no_net            # type: ignore[assignment]
    socket.socket.connect_ex = _no_net         # type: ignore[assignment]
    socket.create_connection = _no_net         # type: ignore[assignment]

    def _no_proc(*a, **k):
        _GUARD["subprocess_attempts"] += 1
        raise RuntimeError(f"[{LABEL}] subprocess blocked")

    subprocess.Popen = _no_proc                 # type: ignore[assignment]
    subprocess.run = _no_proc                   # type: ignore[assignment]
    subprocess.call = _no_proc                  # type: ignore[assignment]
    subprocess.check_call = _no_proc            # type: ignore[assignment]
    subprocess.check_output = _no_proc          # type: ignore[assignment]
    os.system = _no_proc                        # type: ignore[assignment]


def device_env(device: str, gpu_id) -> dict:
    """The environment a chunk process gets before torch is imported."""
    if device == "cpu":
        return {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}
    if device == "cuda":
        if gpu_id is None:
            raise Refused("--device cuda needs --gpu-ids")
        return {"CUDA_VISIBLE_DEVICES": str(int(gpu_id)), "HIP_VISIBLE_DEVICES": ""}
    raise Refused(f"unknown --device {device}")


LL2_SIGNATURE_TAIL = ["per_seed_sieve_features", "fallback_metadata_merge"]
LL2_DEFAULTS = {"per_seed_sieve_features": "off", "fallback_metadata_merge": "legacy6"}


def check_signatures(ll2_requested: bool = False) -> dict:
    """Exact signatures, fail closed. full_scoring_worker.score_survivors may be the form without
    the LL-2 switch (main) or with it; asking for a non-default LL-2 value on the form without it
    refuses (owner, TASK_LAND_WF_VOTE_20261125_ADDENDUM_1)."""
    import importlib
    seen = {}
    for (mod, fn), expected in EXPECTED_SIGNATURES.items():
        f = getattr(importlib.import_module(mod), fn)
        got = list(inspect.signature(f).parameters)
        seen[f"{mod}.{fn}"] = got
        if (mod, fn) == ("full_scoring_worker", "score_survivors"):
            seen["ll2_switch_present"] = got == expected + LL2_SIGNATURE_TAIL
            if got not in (expected, expected + LL2_SIGNATURE_TAIL):
                raise Refused(f"signature of {mod}.{fn} changed: {got} != {expected} (+ {LL2_SIGNATURE_TAIL})")
            if ll2_requested and not seen["ll2_switch_present"]:
                raise Refused(f"LL-2 switch missing: {REPO_ROOT / 'full_scoring_worker.py'} score_survivors "
                              f"has no {LL2_SIGNATURE_TAIL}")
        elif got != expected:
            raise Refused(f"signature of {mod}.{fn} changed: {got} != {expected}")
    return seen


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
sha256_file = _builder.sha256_file


def git_identity() -> dict:
    files = ["generate_step3_scoring_jobs.py", "full_scoring_worker.py", "survivor_scorer.py",
             "holdout_quality.py", "prng_registry.py", "utils/survivor_loader.py",
             "models/global_state_tracker.py", "run_step3_full_scoring.sh",
             "scripts/diagnostics/e2e_step3_harness.py", "scripts/diagnostics/e2e_build_run_copy.py"]
    out = {"repo_root": str(REPO_ROOT)}
    try:
        out["head"] = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                     capture_output=True, text=True, check=True).stdout.strip()
        out["status_of_used_files"] = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", *files],
            capture_output=True, text=True, check=True).stdout.splitlines()
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    out["file_sha256"] = {f: sha256_file(REPO_ROOT / f) for f in files if (REPO_ROOT / f).exists()}
    return out


def parse_chunks(spec: str, n: int) -> list:
    if spec == "all":
        return list(range(n))
    sel = set()
    for part in spec.split(","):
        part = part.strip()
        if re.fullmatch(r"\d+", part):
            sel.add(int(part))
        elif re.fullmatch(r"\d+-\d+", part):
            a, b = map(int, part.split("-"))
            if b < a:
                raise Refused(f"--chunks range {part} is reversed")
            sel.update(range(a, b + 1))
        else:
            raise Refused(f"--chunks: cannot parse {part!r} (use 'all', '0', '0-3', '0,2')")
    bad = [i for i in sel if i >= n]
    if bad:
        raise Refused(f"--chunks names chunk(s) {sorted(bad)} but only {n} exist")
    return sorted(sel)


def verify_inputs(args) -> dict:
    """Every input must be a file of the run copy, with the sha256 its record holds."""
    run_copy = Path(os.path.expanduser(args.run_copy)).resolve()
    rec_p = run_copy / "run_record.json"
    if not rec_p.is_file():
        raise Refused(f"no run_record.json in --run-copy {run_copy}")
    rec = json.loads(rec_p.read_text())
    if rec.get("kind") != "e2e_run_copy":
        raise Refused("--run-copy record is not an e2e_run_copy record")
    outputs = rec.get("outputs", {})

    def member(p: str, what: str) -> tuple:
        path = Path(os.path.expanduser(p))
        if not path.is_absolute():
            path = run_copy / path
        path = path.resolve()
        try:
            rel = str(path.relative_to(run_copy))
        except ValueError:
            raise Refused(f"{what} is not inside the run copy: {path}")
        if rel not in outputs:
            raise Refused(f"{what} ({rel}) is not listed in the run copy's record")
        s = sha256_file(path)
        if s != outputs[rel]:
            raise Refused(f"{what} ({rel}) sha256 {s} != recorded {outputs[rel]}")
        return path, rel, s

    npz, npz_rel, npz_sha = member(args.npz, "--npz")
    if npz_sha != rec["declared_parameters"]["generation_digest"]:
        raise Refused("--npz sha256 differs from the run copy's generation digest")
    train, train_rel, train_sha = member(args.train_history, "--train-history")
    hold, hold_rel, hold_sha = member(args.holdout_history, "--holdout-history")
    if hold_rel != "label_block_history.json":
        # The grade block must never reach Step 3: it would become the label.
        raise Refused(f"--holdout-history must be the run copy's label_block_history.json, got {hold_rel}")
    cfg = None
    if args.config:
        cfg = member(args.config, "--config")
    return {"run_copy": run_copy, "record": rec, "record_sha256": sha256_file(rec_p),
            "npz": (npz, npz_rel, npz_sha), "train": (train, train_rel, train_sha),
            "holdout": (hold, hold_rel, hold_sha), "config": cfg}


# ---------------------------------------------------------------------------
# aggregation -- reproduction of run_step3_full_scoring.sh AGGREGATE_EOF (:275-442)
# ---------------------------------------------------------------------------
def aggregate_like_launcher(results_dir: Path, train_history_file: Path, output_file: Path,
                            allow_global_failure: bool, _sort_ascending: bool = False) -> dict:
    """Same inputs, same order, same merge, same sort, same json.dump as the
    launcher's heredoc. Differences, all fail-closed: a global-feature failure
    raises unless allowed (the launcher prints and continues); a chunk file
    that fails to parse raises (the launcher counts it and continues)."""
    from models.global_state_tracker import GlobalStateTracker

    global_features_prefixed = {}
    global_error = None
    try:
        with open(train_history_file) as f:
            lottery_data = json.load(f)
        if lottery_data and isinstance(lottery_data[0], dict):
            lottery_history = [d.get('draw', d.get('number', 0)) for d in lottery_data]
        else:
            lottery_history = lottery_data
        global_tracker = GlobalStateTracker(lottery_history, {'mod': 1000})   # launcher :340 hardcodes 1000 (TODO-E2E-14)
        global_features = global_tracker.get_global_state()
        global_features_prefixed = {f"global_{k}": v for k, v in global_features.items()}
    except Exception as e:
        global_error = f"{type(e).__name__}: {e}"
        if not allow_global_failure:
            raise RuntimeError(f"global features failed: {global_error}")

    all_survivors = []
    error_count = 0
    files_processed = 0
    per_file = {}
    for result_file in sorted(results_dir.glob("chunk_*.json")):
        with open(result_file, 'r') as f:
            chunk_results = json.load(f)
        n_file = 0
        if isinstance(chunk_results, list):
            for survivor in chunk_results:
                if isinstance(survivor, dict) and 'seed' in survivor:
                    all_survivors.append(survivor)
                    n_file += 1
                    if 'error' in survivor:
                        error_count += 1
        files_processed += 1
        per_file[result_file.name] = n_file
    if not all_survivors:
        raise RuntimeError("No survivors aggregated")

    if global_features_prefixed:
        for survivor in all_survivors:
            if 'features' in survivor:
                survivor['features'].update(global_features_prefixed)
            else:
                survivor['features'] = dict(global_features_prefixed)

    all_survivors.sort(key=lambda x: x.get('score', 0), reverse=not _sort_ascending)
    with open(output_file, 'w') as f:
        json.dump(all_survivors, f, indent=2)
    return {"files_processed": files_processed, "per_file": per_file,
            "total": len(all_survivors), "error_count": error_count,
            "global_features": len(global_features_prefixed), "global_error": global_error,
            "survivors": all_survivors}


def validate_like_launcher(survivors: list, min_features: int) -> dict:
    """Phase 6 floor (run_step3_full_scoring.sh VALIDATE_EOF): required keys on
    the first row and >= 46 features. min_features is the launcher's 46 by default."""
    if not survivors:
        raise RuntimeError("Phase 6: empty output")
    sample = survivors[0]
    missing = [k for k in ('seed', 'score', 'features') if k not in sample]
    if missing:
        raise RuntimeError(f"Phase 6: missing keys {missing}")
    fc = len(sample.get('features', {}))
    if fc < min_features:
        raise RuntimeError(f"Phase 6: only {fc} features (expected {min_features}+)")
    return {"feature_count_first_row": fc}


def harness_checks(survivors: list, expected_seeds: list) -> dict:
    """Checks the launcher does not make. Every row, not just the first."""
    got = [s["seed"] for s in survivors]
    if len(got) != len(expected_seeds):
        raise RuntimeError(f"row count {len(got)} != input rows {len(expected_seeds)}")
    if sorted(got) != sorted(expected_seeds):
        raise RuntimeError("aggregated seeds differ from the input seeds")
    keysets = {}
    nonfinite = 0
    nonnumeric = {}
    for s in survivors:
        ks = tuple(sorted(s.get("features", {})))
        keysets[ks] = keysets.get(ks, 0) + 1
        for k, v in s.get("features", {}).items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                nonnumeric[k] = nonnumeric.get(k, 0) + 1
            elif not math.isfinite(v):
                nonfinite += 1
    if len(keysets) != 1:
        raise RuntimeError(f"rows carry {len(keysets)} different feature key sets")
    keys = list(next(iter(keysets)))
    hq = [s.get("holdout_quality") for s in survivors]
    hh = [s.get("holdout_hits") for s in survivors]
    return {"rows": len(got), "feature_keys": keys, "feature_count": len(keys),
            "global_feature_count": sum(k.startswith("global_") for k in keys),
            "nonfinite_feature_values": nonfinite, "nonnumeric_feature_values": nonnumeric,
            "holdout_quality": {"min": min(hq), "max": max(hq), "distinct": len(set(hq))},
            "holdout_hits": {"min": min(hh), "max": max(hh), "distinct": len(set(hh))}}


# ---------------------------------------------------------------------------
# one chunk, in a fresh spawned process
# ---------------------------------------------------------------------------
class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.hits = {"batch_fallback": 0, "cpu_generate": 0, "hq_failed": 0, "seed_failed": 0}

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return
        if LOG_BATCH_FALLBACK in msg:
            self.hits["batch_fallback"] += 1
        if LOG_CPU_GENERATE in msg:
            self.hits["cpu_generate"] += 1
        if any(p in msg for p in LOG_HQ_FAILED):
            self.hits["hq_failed"] += 1
        if LOG_SEED_FAILED in msg:
            self.hits["seed_failed"] += 1


def _apply_chunk_mutant(name, fsw, SurvivorScorer, out) -> None:
    """Make the PRODUCTION code take one of its silent fallbacks, in this
    process only, and count every call of the mutated function."""
    def counted(fn):
        def wrapper(*a, **k):
            out["mutant_calls"] += 1
            return fn(*a, **k)
        return wrapper

    def boom(*a, **k):
        raise RuntimeError("selftest mutant")

    if name == "batch_raise":                 # -> sequential fallback, full_scoring_worker.py:438-499
        SurvivorScorer.extract_ml_features_batch = counted(boom)
    elif name == "torch_generate_raise":      # -> _cpu_batch_generate, survivor_scorer.py:575-579
        import survivor_scorer as ss
        ss.get_pytorch_gpu_reference = counted(lambda *a, **k: boom)
    elif name == "holdout_quality_raise":     # -> holdout_quality written as 0.0
        fsw.compute_holdout_quality = counted(boom)
    elif name == "seed_raise":                # -> per-seed error record
        SurvivorScorer.extract_ml_features_batch = counted(boom)
        SurvivorScorer.extract_ml_features = counted(boom)
    elif name in PARENT_MUTANTS:
        pass
    else:
        raise RuntimeError(f"unknown {MUTANT_ENV}={name}")


def run_chunk(task: dict) -> dict:
    t0 = time.time()
    out = {"chunk": task["chunk"], "pid": os.getpid(), "status": "raised", "error": None}
    log_path = Path(task["log_dir"]) / f"chunk_{task['chunk']:04d}.log"
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    sys.stdout = os.fdopen(1, "w", buffering=1)
    sys.stderr = os.fdopen(2, "w", buffering=1)
    try:
        os.environ.update(task["env"])
        sys.dont_write_bytecode = True
        install_guards()
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        import atexit
        import torch
        torch.set_num_threads(task["torch_threads"])
        import full_scoring_worker as fsw
        atexit.unregister(fsw._best_effort_gpu_cleanup)
        from survivor_scorer import SurvivorScorer
        cuda_seen = bool(torch.cuda.is_available())
        out["torch_cuda_available"] = cuda_seen
        if task["device"] == "cpu" and cuda_seen:
            raise RuntimeError("--device cpu but torch sees a CUDA device")
        if task["device"] == "cuda":
            # fail closed: without this a cuda run would silently score on CPU
            # (survivor_scorer.py:117 picks cuda only if available). UNTESTED at
            # run time: no GPU use was allowed while building (TODO-E2E-17).
            if not cuda_seen:
                raise RuntimeError(f"--device cuda but torch sees no CUDA device "
                                   f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})")
            out["cuda_device_name"] = torch.cuda.get_device_name(0)

        cap = _Capture()
        logging.getLogger().addHandler(cap)
        mutant = os.environ.get(MUTANT_ENV)
        out["mutant"] = mutant
        out["mutant_calls"] = 0
        if mutant:
            _apply_chunk_mutant(mutant, fsw, SurvivorScorer, out)

        # == full_scoring_worker.main() :577-656, minus argparse/GPU env/print ==
        survivors_full = fsw.load_survivors_with_metadata(task["chunk_file"])
        seeds = [s['seed'] for s in survivors_full]
        survivor_metadata = {s['seed']: s for s in survivors_full}
        train_history = fsw.load_lottery_history(task["train_history"])
        holdout_history = fsw.load_lottery_history(task["holdout_history"])
        if not seeds:
            raise RuntimeError("No seeds found in input file")
        if not train_history:
            raise RuntimeError("No lottery history found in training file")
        if not holdout_history:
            raise RuntimeError("holdout (label block) history is empty")
        results = fsw.score_survivors(
            seeds=seeds, train_history=train_history, prng_type=task["prng_type"],
            mod=task["mod"], batch_size=task["batch_size"], forward_survivors=None,
            reverse_survivors=None, scorer_class=SurvivorScorer,
            survivor_metadata=survivor_metadata, holdout_history=holdout_history,
            **({k: task[k] for k in LL2_SIGNATURE_TAIL} if task["ll2_switch_present"] else {}))

        # == fail-closed checks ==
        mod = task["mod"]
        if len(results) != len(seeds) or [r.get("seed") for r in results] != seeds:
            raise RuntimeError("score_survivors returned rows that do not match the chunk seeds")
        n_err = sum('error' in r for r in results)
        n_seq = sum('worker_gpu' not in r.get('metadata', {}) for r in results if 'error' not in r)
        n_unreduced = sum(r.get('features', {}).get('pred_max', 0) > mod - 1 for r in results)
        n_hq_empty = sum(not r.get('holdout_features') for r in results if 'error' not in r)
        flags = {
            "seed_errors": n_err,
            "sequential_fallback_rows": n_seq, "sequential_fallback_log": cap.hits["batch_fallback"],
            "cpu_generate_log": cap.hits["cpu_generate"], "unreduced_pred_max_rows": n_unreduced,
            "holdout_quality_empty_rows": n_hq_empty, "holdout_quality_failed_log": cap.hits["hq_failed"],
            "seed_failed_log": cap.hits["seed_failed"],
        }
        out["flags"] = flags
        problems = []
        if (n_err or cap.hits["seed_failed"]) and not task["allow"]["seed_errors"]:
            problems.append("seed errors")
        if (n_seq or cap.hits["batch_fallback"]) and not task["allow"]["sequential_fallback"]:
            problems.append("sequential fallback (6 of 18 metadata fields, Ch4 B6)")
        if (cap.hits["cpu_generate"] or n_unreduced) and not task["allow"]["cpu_generate_fallback"]:
            problems.append("CPU generate fallback (unreduced predictions, Ch4 B4)")
        if (n_hq_empty or cap.hits["hq_failed"]) and not task["allow"]["holdout_quality_failure"]:
            problems.append("holdout_quality failed and was written as 0.0")

        name = f"chunk_{task['chunk']:04d}.json"
        dest = Path(task["results_dir"]) / (name if not problems else FAILED_PREFIX + name)
        with open(dest, 'w') as f:
            json.dump(results, f, indent=2)          # as full_scoring_worker.py:673-674
        out.update({"rows": len(results), "output": dest.name, "output_sha256": sha256_file(dest)})
        out["status"] = "ok" if not problems else "failed"
        out["error"] = "; ".join(problems) or None
    except BaseException as e:
        out["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        out["status"] = "raised"
    finally:
        out["guards"] = dict(_GUARD)
        out["seconds"] = round(time.time() - t0, 3)
        out["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=f"{LABEL}: Step 3 full scoring, in-process, isolated.",
                                 allow_abbrev=False)   # no silent prefix matching (ADDENDUM_4 Fix 2)
    ap.add_argument("--run-copy", required=True, help="directory built by e2e_build_run_copy.py")
    ap.add_argument("--npz", default="bidirectional_survivors_binary.npz",
                    help="relative to --run-copy unless absolute; must be in its record")
    ap.add_argument("--train-history", default="train_history.json")
    ap.add_argument("--holdout-history", default="label_block_history.json",
                    help="must be the run copy's label block (the grade block is refused)")
    ap.add_argument("--config", default=None,
                    help="scorer config passed to generate_jobs (e.g. configs/optimal_scorer_config.json); "
                         "only prng_type/mod are read (Ch4 B5, TODO-E2E-2)")
    ap.add_argument("--out-dir", required=True, help="created; must lie under <worktree>/data/")
    ap.add_argument("--chunk-size", type=int, default=1000,
                    help="rows per chunk (launcher default 1000, run_step3_full_scoring.sh:70)")
    ap.add_argument("--chunks", default="all", help="'all' or a selection such as '0', '0-3', '0,5'")
    ap.add_argument("--prng-type", default="java_lcg", help="generate_jobs default; config overrides")
    ap.add_argument("--mod", type=int, default=1000, help="generate_jobs default; config overrides")
    ap.add_argument("--batch-size", type=int, default=100, help="score_survivors progress interval")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel chunk processes (one fresh spawned process per chunk)")
    ap.add_argument("--torch-threads", type=int, default=1, help="torch.set_num_threads per process")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--gpu-ids", default=None, help="for --device cuda: comma list, round-robin per chunk")
    ap.add_argument("--min-features", type=int, default=46, help="Phase 6 floor (launcher: 46)")
    ap.add_argument("--forbid-root", action="append", default=[])
    ap.add_argument("--allow-sequential-fallback", action="store_true")
    ap.add_argument("--allow-cpu-generate-fallback", action="store_true")
    ap.add_argument("--allow-holdout-quality-failure", action="store_true")
    ap.add_argument("--allow-seed-errors", action="store_true")
    ap.add_argument("--allow-global-feature-failure", action="store_true")
    # LL-2 settings (ch14-learn READY_LOOP_LEARN_W1A §6.2); defaults = today's behaviour
    ap.add_argument("--per-seed-sieve-features", choices=["off", "on"], default="off")
    ap.add_argument("--fallback-metadata-merge", choices=["legacy6", "shared"], default="legacy6")
    args = ap.parse_args(argv)

    t_start = time.time()
    try:
        if args.chunk_size < 1 or args.workers < 1 or args.torch_threads < 1 or args.batch_size < 1:
            raise Refused("--chunk-size, --workers, --torch-threads and --batch-size must be >= 1")
        gpu_ids = None
        if args.device == "cuda":
            if not args.gpu_ids:
                raise Refused("--device cuda needs --gpu-ids")
            gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
        elif args.gpu_ids:
            raise Refused("--gpu-ids given with --device cpu")
        # LL-2 asked for (a non-default value): the pipeline file must have the switch. Read as
        # source before anything is created (owner, TASK_LAND_WF_VOTE_20261125_ADDENDUM_1).
        ll2_requested = {k: getattr(args, k) for k, v in LL2_DEFAULTS.items() if getattr(args, k) != v}
        if ll2_requested:
            import importlib.util as _ilu
            _sp = _ilu.spec_from_file_location("e2e_pipeline_caps", HERE / "e2e_pipeline_caps.py")
            _caps = _ilu.module_from_spec(_sp)
            _sp.loader.exec_module(_caps)
            _ll2 = _caps.capabilities(REPO_ROOT)["LL-2"]
            if not _ll2["present"]:
                raise Refused(f"LL-2 switch needed by {ll2_requested} is missing, nothing run: "
                              + "; ".join(_ll2["lacks"]))
        out_dir = Path(os.path.expanduser(args.out_dir))
        try:
            out_res = _builder.check_out_dir(out_dir, args.forbid_root)
        except _builder.Refused as e:
            raise Refused(e.msg)
        inp = verify_inputs(args)
    except Refused as e:
        print(f"[{LABEL}] REFUSED: {e}", file=sys.stderr)
        return 2

    git = git_identity()
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    install_guards()
    os.makedirs(out_res, exist_ok=False)
    log_dir = out_res / "logs"
    log_dir.mkdir()
    root = logging.getLogger()
    fh = logging.FileHandler(log_dir / "harness.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(process)d %(name)s %(levelname)s %(message)s"))
    root.addHandler(fh)
    root.setLevel(logging.INFO)

    sys.path.insert(0, str(REPO_ROOT))
    try:
        sigs = check_signatures(bool(ll2_requested))
    except Refused as e:
        print(f"[{LABEL}] REFUSED: {e}", file=sys.stderr)
        return 2
    import generate_step3_scoring_jobs as gsj
    from utils.survivor_loader import load_survivors
    gsj.load_survivors = functools.partial(load_survivors, allow_fallback=False)

    npz, train, hold = inp["npz"][0], inp["train"][0], inp["holdout"][0]
    cfg = inp["config"][0] if inp["config"] else None

    # ---- Phase 1: the production chunker, cwd = out_dir ----------------------
    cwd0 = os.getcwd()
    os.chdir(out_res)
    try:
        jobs = gsj.generate_jobs(
            survivors_file=str(npz), train_history_file=str(train),
            holdout_history_file=str(hold), config_file=str(cfg) if cfg else None,
            chunk_size=args.chunk_size, prng_type=args.prng_type, mod=args.mod,
            remote_base_path=str(out_res), timeout=7200,
            forward_survivors_file=None, reverse_survivors_file=None)
    finally:
        os.chdir(cwd0)
    with open(out_res / "scoring_jobs.RECORD_ONLY.json", "w") as f:
        json.dump(jobs, f, indent=2)

    def argval(job, flag):
        a = job["args"]
        return a[a.index(flag) + 1]

    eff = {(argval(j, "--prng-type"), argval(j, "--mod")) for j in jobs}
    if len(eff) != 1:
        print(f"[{LABEL}] FAILED: jobs disagree on prng_type/mod: {eff}", file=sys.stderr)
        return 1
    prng_type, mod = next(iter(eff))
    mod = int(mod)

    import numpy as np
    with np.load(npz, allow_pickle=False) as z:
        npz_seeds = [int(x) for x in z["seeds"]]
    chunk_files = [out_res / j["chunk_file"] for j in jobs]
    chunk_seeds = []
    for cf in chunk_files:
        chunk_seeds.append([int(s["seed"]) for s in json.loads(cf.read_text())])
    if [s for c in chunk_seeds for s in c] != npz_seeds:
        print(f"[{LABEL}] FAILED: chunk files do not reproduce the NPZ seed order", file=sys.stderr)
        return 1
    try:
        selected = parse_chunks(args.chunks, len(jobs))
    except Refused as e:
        print(f"[{LABEL}] REFUSED: {e}", file=sys.stderr)
        return 2
    partial = len(selected) != len(jobs)

    # ---- Phase 3 replacement: one fresh process per chunk ---------------------
    run_name = f"full_scoring_results_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir = out_res / "full_scoring_results" / run_name
    results_dir.mkdir(parents=True)
    allow = {"seed_errors": args.allow_seed_errors,
             "sequential_fallback": args.allow_sequential_fallback,
             "cpu_generate_fallback": args.allow_cpu_generate_fallback,
             "holdout_quality_failure": args.allow_holdout_quality_failure}
    tasks = []
    for n, i in enumerate(selected):
        gid = gpu_ids[n % len(gpu_ids)] if gpu_ids else None
        tasks.append({"chunk": i, "chunk_file": str(chunk_files[i]), "train_history": str(train),
                      "holdout_history": str(hold), "prng_type": prng_type, "mod": mod,
                      "batch_size": args.batch_size, "results_dir": str(results_dir),
                      "log_dir": str(log_dir), "device": args.device,
                      "env": device_env(args.device, gid), "gpu_id": gid,
                      "torch_threads": args.torch_threads, "allow": allow,
                      "per_seed_sieve_features": args.per_seed_sieve_features,
                      "fallback_metadata_merge": args.fallback_metadata_merge,
                      "ll2_switch_present": sigs["ll2_switch_present"]})
    mutant = os.environ.get(MUTANT_ENV)
    if mutant and mutant not in CHUNK_MUTANTS + PARENT_MUTANTS:
        print(f"[{LABEL}] REFUSED: unknown {MUTANT_ENV}={mutant}", file=sys.stderr)
        return 2
    if mutant:
        print(f"[{LABEL}] SELFTEST MUTANT ACTIVE: {mutant}", flush=True)
    t_eval = time.time()
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        chunk_out = pool.map(run_chunk, tasks, chunksize=1)
    eval_seconds = time.time() - t_eval
    bad = [c for c in chunk_out if c["status"] != "ok"]

    # ---- Phase 5 reproduction + Phase 6 floor + harness checks ----------------
    agg = None
    checks = None
    phase6 = None
    agg_error = None
    out_name = PARTIAL_NAME if partial else FULL_NAME
    mutant_parent_applied = False
    if not bad:
        try:
            agg = aggregate_like_launcher(results_dir, train, out_res / out_name,
                                          args.allow_global_feature_failure,
                                          _sort_ascending=(mutant == "agg_sort_ascending"))
            mutant_parent_applied = mutant_parent_applied or mutant == "agg_sort_ascending"
            phase6 = validate_like_launcher(agg["survivors"], args.min_features)
            expected = [s for i in selected for s in chunk_seeds[i]]
            if mutant == "agg_drop_row":
                agg["survivors"].pop()
                mutant_parent_applied = True
            elif mutant == "agg_drop_feature":
                k0 = sorted(agg["survivors"][-1]["features"])[0]
                del agg["survivors"][-1]["features"][k0]
                mutant_parent_applied = True
            checks = harness_checks(agg["survivors"], expected)
        except Exception as e:
            agg_error = f"{type(e).__name__}: {e}"
            if (out_res / out_name).exists():
                os.replace(out_res / out_name, out_res / (FAILED_PREFIX + out_name))
    status = "ok" if (not bad and agg_error is None) else "failed"

    ru_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record = {
        "label": LABEL,
        "kind": "e2e_step3_harness",
        "harness_version": HARNESS_VERSION,
        "status": status,
        "selftest_mutant": mutant,
        "selftest_mutant_parent_applied": mutant_parent_applied,
        "partial": partial,
        "argv": sys.argv, "args": vars(args),
        "declared_parameters": {"per_seed_sieve_features": args.per_seed_sieve_features,
                                "fallback_metadata_merge": args.fallback_metadata_merge},
        # what score_survivors actually got: with the switch absent nothing is passed, which is
        # the default (off, legacy6); a non-default value on such a pipeline refused above
        "applied_parameters": {"ll2_switch_present": sigs["ll2_switch_present"],
                               "passed_to_score_survivors": ({k: getattr(args, k) for k in LL2_SIGNATURE_TAIL}
                                                             if sigs["ll2_switch_present"] else {}),
                               "effective": {k: getattr(args, k) for k in LL2_SIGNATURE_TAIL}},
        "host": os.uname().nodename, "cpu_count": os.cpu_count(),
        "cpu_affinity": len(os.sched_getaffinity(0)),
        "process_model": "multiprocessing spawn Pool, maxtasksperchild=1 (one fresh process per chunk)",
        "multi_core": args.workers > 1,
        "started_unix": t_start, "wall_seconds": round(time.time() - t_start, 3),
        "eval_seconds": round(eval_seconds, 3),
        "peak_rss_kib": {"parent": ru_self,
                         "chunk_process_max": max((c.get("peak_rss_kib", 0) for c in chunk_out), default=0)},
        "run_copy": {"dir": str(inp["run_copy"]), "record_sha256": inp["record_sha256"]},
        "inputs": {k: {"path": str(inp[k][0]), "rel": inp[k][1], "sha256": inp[k][2]}
                   for k in ("npz", "train", "holdout") if inp[k]},
        "config": ({"path": str(inp["config"][0]), "rel": inp["config"][1], "sha256": inp["config"][2]}
                   if inp["config"] else None),
        "effective": {"prng_type": prng_type, "mod": mod, "chunks_total": len(jobs),
                      "chunks_selected": selected, "rows_total": len(npz_seeds),
                      "rows_selected": sum(len(chunk_seeds[i]) for i in selected)},
        "code": git, "signatures": sigs, "python": sys.version.split()[0],
        "guards_parent": dict(_GUARD),
        "chunks": chunk_out,
        "aggregate": ({k: v for k, v in agg.items() if k != "survivors"} if agg else None),
        "aggregate_error": agg_error,
        "phase6": phase6, "checks": checks,
        "output": ({"name": out_name, "sha256": sha256_file(out_res / out_name)}
                   if status == "ok" else None),
        "results_dir": str(results_dir.relative_to(out_res)),
    }
    with open(out_res / "run_record.json", "w") as f:
        json.dump(record, f, indent=2)

    print(f"[{LABEL}] status {status}{' (PARTIAL)' if partial else ''}: "
          f"{len(selected)} of {len(jobs)} chunks, {record['effective']['rows_selected']} rows, "
          f"prng_type {prng_type} mod {mod}", flush=True)
    for c in bad:
        print(f"[{LABEL}] chunk {c['chunk']}: {c['status']}: {str(c['error']).splitlines()[0]}", flush=True)
    if agg_error:
        print(f"[{LABEL}] aggregation/validation: {agg_error}", flush=True)
    print(f"[{LABEL}] workers {args.workers} (multi-core: {args.workers > 1}) device {args.device}; "
          f"wall {record['wall_seconds']:.1f}s; peak RSS parent {ru_self} KiB, "
          f"chunk max {record['peak_rss_kib']['chunk_process_max']} KiB", flush=True)
    print(f"[{LABEL}] outputs in {out_res}", flush=True)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
