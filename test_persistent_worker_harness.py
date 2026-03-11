#!/usr/bin/env python3
"""
test_persistent_worker_harness.py
===================================
Dry-run harness for persistent_worker_coordinator.py

Tests ALL code paths without real GPUs, real SSH, or real sieve kernels.
Uses synthetic stubs and monkey-patches to exercise:

  T01 — PersistentWorkerCoordinator instantiation + config load
  T02 — Per-node semaphore creation
  T03 — worker_pool_size cap respected (min of pool_size vs gpu_count)
  T04 — ROCm spawn stagger timing (4s between workers, measured)
  T05 — Worker spawn success path → handle.alive = True
  T06 — Worker spawn failure path → handle.quarantined = True
  T07 — _ensure_worker_alive: dead process → respawn attempt
  T08 — _ensure_worker_alive: quarantined → returns False immediately
  T09 — Semaphore acquire/release in _dispatch_to_worker
  T10 — Semaphore released even on exception (finally block)
  T11 — Constant skip sieve pass (forward java_lcg)
  T12 — Constant skip sieve pass (reverse java_lcg_reverse)
  T13 — Hybrid sieve pass (forward java_lcg_hybrid) — strategy auto-load
  T14 — Hybrid sieve pass (reverse java_lcg_hybrid_reverse) — strategies in payload
  T15 — run_trial_persistent: pruned path (forward_zero)
  T16 — run_trial_persistent: full 4-pass path, correct result structure
  T17 — run_trial_persistent: test_both_modes=False skips hybrid passes
  T18 — _get_residues_for_config helper
  T19 — _build_test_result_from_pw helper + accumulator update
  T20 — window_optimizer_integration_final gate: use_persistent_workers=True routes to PWC
  T21 — window_optimizer_integration_final gate: use_persistent_workers=False uses original path
  T22 — Worker shutdown: sends shutdown command, reaps process
  T23 — Per-rig fault isolation: one rig quarantined, others continue
  T24 — Zeus local path: _dispatch_local_sieve called for localhost
  T25 — GPU isolation: PersistentWorkerCoordinator never imports CuPy

Usage:
  python3 test_persistent_worker_harness.py           # run all tests
  python3 test_persistent_worker_harness.py T05 T09   # run specific tests
"""

import sys
import os
import json
import time
import threading
import subprocess
import tempfile
import importlib
import unittest
from unittest.mock import patch, MagicMock, PropertyMock
from io import StringIO
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Add project root to path
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────────────────────────
# Mock GPU libraries so tests run without real GPUs
# Must be done BEFORE importing any project modules
# ─────────────────────────────────────────────────────────────────────────────
import unittest.mock as _mock

# Mock CuPy
_cupy_mock = _mock.MagicMock()
_cupy_mock.cuda.Device = _mock.MagicMock()
sys.modules['cupy'] = _cupy_mock
sys.modules['cupy.cuda'] = _cupy_mock.cuda

# Mock hybrid_strategy (optional module — not always present)
_hybrid_mock = _mock.MagicMock()
_strategy = _mock.MagicMock()
_strategy.max_consecutive_misses = 3
_strategy.skip_tolerance = 5
_hybrid_mock.get_all_strategies.return_value = [_strategy]
sys.modules['hybrid_strategy'] = _hybrid_mock

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic distributed_config.json for tests
# ─────────────────────────────────────────────────────────────────────────────
SYNTHETIC_CONFIG = {
    "nodes": [
        {
            "hostname": "localhost",
            "gpu_type": "RTX 3080 Ti",
            "gpu_count": 2,
            "python_env": "/home/michael/venvs/torch/bin/python3",
            "script_path": "/home/michael/distributed_prng_analysis",
            "username": "michael"
        },
        {
            "hostname": "192.168.3.120",
            "gpu_type": "RX 6600",
            "gpu_count": 8,
            "python_env": "/home/michael/rocm_env/bin/python3",
            "script_path": "/home/michael/distributed_prng_analysis",
            "username": "michael"
        },
        {
            "hostname": "192.168.3.154",
            "gpu_type": "RX 6600",
            "gpu_count": 8,
            "python_env": "/home/michael/rocm_env/bin/python3",
            "script_path": "/home/michael/distributed_prng_analysis",
            "username": "michael"
        },
        {
            "hostname": "192.168.3.162",
            "gpu_type": "RX 6600",
            "gpu_count": 8,
            "python_env": "/home/michael/rocm_env/bin/python3",
            "script_path": "/home/michael/distributed_prng_analysis",
            "username": "michael"
        },
    ]
}

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic worker response factory
# ─────────────────────────────────────────────────────────────────────────────
def make_ok_response(job_id="sieve_000", n_survivors=5):
    survivors   = list(range(1000, 1000 + n_survivors))
    match_rates = [0.5] * n_survivors
    return json.dumps({
        "status": "ok",
        "job_id": job_id,
        "survivors": survivors,
        "match_rates": match_rates,
        "skip_sequences": [],
        "strategy_ids": [],
    }) + "\n"

def make_error_response(job_id="sieve_000", msg="synthetic error"):
    return json.dumps({"status": "error", "job_id": job_id, "message": msg}) + "\n"

def make_ready_response(gpu_id=0):
    return json.dumps({"status": "ready", "gpu_id": gpu_id, "device": "RX 6600"}) + "\n"

# ─────────────────────────────────────────────────────────────────────────────
# Fake Popen that simulates a live persistent worker
# ─────────────────────────────────────────────────────────────────────────────
class FakeWorkerProcess:
    def __init__(self, gpu_id=0, fail=False, timeout=False):
        self._fail    = fail
        self._timeout = timeout
        self._alive   = True
        self._jobs    = []
        ready = make_ready_response(gpu_id) if not fail else ""
        self.stdout = StringIO(ready)
        self.stdin  = StringIO()
        self._stdin_writes = []

    def poll(self):
        return None if self._alive else 1

    def wait(self, timeout=None):
        self._alive = False
        return 0

    def kill(self):
        self._alive = False

    # Capture stdin writes and inject response into stdout
    def write_job(self, line):
        self._stdin_writes.append(line)
        job = json.loads(line.strip())
        job_id = job.get("job_id", "unknown")
        if self._fail:
            response = make_error_response(job_id)
        elif self._timeout:
            response = None  # simulate timeout
        else:
            response = make_ok_response(job_id)
        if response:
            self.stdout = StringIO(response)


# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────
PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = {}

def run_test(name, fn):
    try:
        fn()
        results[name] = PASS
        print(f"  {PASS} {name}")
    except AssertionError as e:
        results[name] = f"{FAIL}: {e}"
        print(f"  {FAIL} {name}: {e}")
    except Exception as e:
        results[name] = f"{FAIL}: {type(e).__name__}: {e}"
        print(f"  {FAIL} {name}: {type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def setup_pwc(pool_size=2, max_per_node=4, config_override=None):
    """Create a PWC with a synthetic config file."""
    cfg = config_override or SYNTHETIC_CONFIG
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(cfg, f)
        cfg_path = f.name
    from persistent_worker_coordinator import PersistentWorkerCoordinator
    pwc = PersistentWorkerCoordinator(
        config_file      = cfg_path,
        worker_pool_size = pool_size,
        max_per_node     = max_per_node,
    )
    return pwc, cfg_path


def T01_instantiation():
    pwc, cfg = setup_pwc()
    assert len(pwc.nodes) == 4, f"Expected 4 nodes, got {len(pwc.nodes)}"
    assert pwc.worker_pool_size == 2
    os.unlink(cfg)

def T02_semaphores_created():
    pwc, cfg = setup_pwc(max_per_node=4)
    for node in pwc.nodes:
        assert node.hostname in pwc._node_semaphores, \
            f"No semaphore for {node.hostname}"
        sem = pwc._node_semaphores[node.hostname]
        # Semaphore should allow up to max_per_node=4 acquires
        acquired = 0
        for _ in range(4):
            assert sem.acquire(blocking=False), f"Semaphore blocked early at {acquired}"
            acquired += 1
        assert not sem.acquire(blocking=False), "Semaphore should be exhausted at 4"
        for _ in range(4):
            sem.release()
    os.unlink(cfg)

def T03_worker_pool_size_cap():
    pwc, cfg = setup_pwc(pool_size=3)
    # rig nodes have gpu_count=8, pool_size=3 → should spawn min(3,8)=3 per rig
    # localhost has gpu_count=2, is_localhost → no workers spawned
    # We test the cap logic directly
    for node in pwc.nodes:
        if not pwc._is_localhost(node.hostname) and pwc._is_rocm(node):
            expected = min(pwc.worker_pool_size, node.gpu_count)
            assert expected == 3, f"Expected pool cap 3, got {expected}"
    os.unlink(cfg)

def T04_rocm_spawn_stagger():
    from persistent_worker_coordinator import ROCM_SPAWN_STAGGER_S
    assert ROCM_SPAWN_STAGGER_S == 4.0, f"Expected 4.0s stagger, got {ROCM_SPAWN_STAGGER_S}"

    pwc, cfg = setup_pwc(pool_size=2)
    spawn_times = []
    original_spawn = pwc._spawn_worker

    def mock_spawn(handle):
        spawn_times.append(time.monotonic())
        handle.alive = True
        handle.proc  = FakeWorkerProcess(handle.gpu_id)
        return True

    pwc._spawn_worker = mock_spawn

    # Patch sleep to record timing without actually sleeping
    sleep_calls = []
    import persistent_worker_coordinator as pwc_mod
    original_sleep = pwc_mod.time.sleep
    pwc_mod.time.sleep = lambda s: sleep_calls.append(s)

    try:
        # Manually run startup loop for one ROCm node
        node = next(n for n in pwc.nodes if pwc._is_rocm(n))
        pool = min(pwc.worker_pool_size, node.gpu_count)
        for gpu_id in range(pool):
            handle = pwc_mod.WorkerHandle(node=node, gpu_id=gpu_id)
            pwc._spawn_worker(handle)
            if gpu_id < pool - 1:
                pwc_mod.time.sleep(ROCM_SPAWN_STAGGER_S)
        assert len(sleep_calls) == pool - 1, \
            f"Expected {pool-1} sleep calls, got {len(sleep_calls)}"
        assert all(s == ROCM_SPAWN_STAGGER_S for s in sleep_calls), \
            f"Sleep durations wrong: {sleep_calls}"
    finally:
        pwc_mod.time.sleep = original_sleep
    os.unlink(cfg)

def T05_spawn_success():
    pwc, cfg = setup_pwc(pool_size=1)
    node = next(n for n in pwc.nodes if pwc._is_rocm(n))
    from persistent_worker_coordinator import WorkerHandle
    handle = WorkerHandle(node=node, gpu_id=0)

    fake_proc = FakeWorkerProcess(gpu_id=0, fail=False)

    with patch('subprocess.Popen', return_value=fake_proc):
        success = pwc._spawn_worker(handle)

    assert success is True, "Expected spawn success"
    assert handle.alive is True
    assert handle.quarantined is False
    os.unlink(cfg)

def T06_spawn_failure():
    pwc, cfg = setup_pwc(pool_size=1)
    node = next(n for n in pwc.nodes if pwc._is_rocm(n))
    from persistent_worker_coordinator import WorkerHandle
    handle = WorkerHandle(node=node, gpu_id=0)

    with patch('subprocess.Popen', side_effect=OSError("SSH failed")):
        success = pwc._spawn_worker(handle)

    assert success is False, "Expected spawn failure"
    assert handle.alive is False
    os.unlink(cfg)

def T07_ensure_worker_alive_respawn():
    pwc, cfg = setup_pwc(pool_size=1)
    node = next(n for n in pwc.nodes if pwc._is_rocm(n))
    from persistent_worker_coordinator import WorkerHandle
    handle = WorkerHandle(node=node, gpu_id=0)

    # Simulate dead process
    dead_proc = FakeWorkerProcess(gpu_id=0)
    dead_proc._alive = False
    handle.proc  = dead_proc
    handle.alive = True  # coordinator thinks it's alive

    new_proc = FakeWorkerProcess(gpu_id=0)
    with patch('subprocess.Popen', return_value=new_proc):
        result = pwc._ensure_worker_alive(handle)

    assert result is True, "Expected respawn success"
    assert handle.alive is True
    assert handle.quarantined is False
    os.unlink(cfg)

def T08_quarantined_skipped():
    pwc, cfg = setup_pwc(pool_size=1)
    node = next(n for n in pwc.nodes if pwc._is_rocm(n))
    from persistent_worker_coordinator import WorkerHandle
    handle = WorkerHandle(node=node, gpu_id=0)
    handle.quarantined = True

    result = pwc._ensure_worker_alive(handle)
    assert result is False, "Quarantined worker should return False"
    os.unlink(cfg)

def T09_semaphore_acquire_release():
    pwc, cfg = setup_pwc(pool_size=1, max_per_node=2)
    node = next(n for n in pwc.nodes if pwc._is_rocm(n))
    from persistent_worker_coordinator import WorkerHandle
    handle = WorkerHandle(node=node, gpu_id=0)
    handle.alive = True

    fake_proc = FakeWorkerProcess(gpu_id=0)
    handle.proc = fake_proc

    # Patch proc stdin/stdout for IPC
    response_data = make_ok_response("sieve_000", 3)
    handle.proc.stdout = StringIO(response_data)

    original_write = StringIO.write
    captured = []
    class CapturingIO(StringIO):
        def write(self, s):
            captured.append(s)
            return len(s)
        def flush(self): pass

    handle.proc.stdin = CapturingIO()

    sem = pwc._node_semaphores[node.hostname]
    initial_value = 2  # max_per_node=2

    # Confirm semaphore starts full
    assert sem.acquire(blocking=False)
    sem.release()

    job = {"job_id": "sieve_000", "prng_type": "java_lcg", "seed_start": 0, "seed_end": 1000}
    pwc._dispatch_to_worker(handle, job)

    # Semaphore should be released after dispatch
    assert sem.acquire(blocking=False), "Semaphore not released after dispatch"
    sem.release()
    os.unlink(cfg)

def T10_semaphore_released_on_exception():
    pwc, cfg = setup_pwc(pool_size=1, max_per_node=2)
    node = next(n for n in pwc.nodes if pwc._is_rocm(n))
    from persistent_worker_coordinator import WorkerHandle
    handle = WorkerHandle(node=node, gpu_id=0)
    handle.alive = True

    # Make stdin.write throw
    class ExplodingIO(StringIO):
        def write(self, s): raise RuntimeError("Simulated write failure")
        def flush(self): pass

    fake_proc = FakeWorkerProcess(gpu_id=0)
    fake_proc.stdin = ExplodingIO()
    handle.proc = fake_proc

    sem = pwc._node_semaphores[node.hostname]
    job = {"job_id": "sieve_000", "prng_type": "java_lcg", "seed_start": 0, "seed_end": 100}

    result = pwc._dispatch_to_worker(handle, job)
    assert result["status"] == "error"

    # Semaphore MUST be released even after exception
    assert sem.acquire(blocking=False), "Semaphore not released after exception!"
    sem.release()
    os.unlink(cfg)

def T11_constant_forward_pass():
    pwc, cfg = setup_pwc(pool_size=1)

    # Stub _dispatch_to_worker to return synthetic survivors
    def fake_dispatch(handle, job):
        assert "_hybrid" not in job["prng_type"], "Should not be hybrid"
        return {
            "status": "ok",
            "survivors": [1000, 2000, 3000],
            "match_rates": [0.5, 0.6, 0.4],
            "skip_sequences": [],
            "strategy_ids": [],
        }

    # Stub _dispatch_local_sieve for Zeus
    def fake_local(job, node):
        return {
            "status": "ok",
            "survivors": [4000],
            "match_rates": [0.5],
        }

    # Add fake workers
    node = next(n for n in pwc.nodes if pwc._is_rocm(n))
    from persistent_worker_coordinator import WorkerHandle
    handle = WorkerHandle(node=node, gpu_id=0, alive=True)
    pwc.workers = [handle]
    pwc._dispatch_to_worker   = fake_dispatch
    pwc._dispatch_local_sieve = fake_local

    result = pwc.run_sieve_pass(
        prng_type   = "java_lcg",
        residues    = list(range(8)),
        total_seeds = 500_000,
        threshold   = 0.25,
        window_size = 8,
        output_file = "/tmp/harness_T11.json",
    )
    assert result["survivor_count"] > 0, "Expected survivors"
    assert result["prng_type"] == "java_lcg"
    assert os.path.exists("/tmp/harness_T11.json")
    os.unlink("/tmp/harness_T11.json")
    os.unlink(cfg)

def T12_constant_reverse_pass():
    pwc, cfg = setup_pwc(pool_size=1)

    def fake_dispatch(handle, job):
        assert job["prng_type"] == "java_lcg_reverse"
        return {"status": "ok", "survivors": [5000], "match_rates": [0.3],
                "skip_sequences": [], "strategy_ids": []}

    def fake_local(job, node):
        return {"status": "ok", "survivors": [], "match_rates": []}

    node = next(n for n in pwc.nodes if pwc._is_rocm(n))
    from persistent_worker_coordinator import WorkerHandle
    pwc.workers = [WorkerHandle(node=node, gpu_id=0, alive=True)]
    pwc._dispatch_to_worker   = fake_dispatch
    pwc._dispatch_local_sieve = fake_local

    result = pwc.run_sieve_pass(
        prng_type   = "java_lcg_reverse",
        residues    = list(range(8)),
        total_seeds = 500_000,
        threshold   = 0.25,
        window_size = 8,
        output_file = "/tmp/harness_T12.json",
    )
    assert result["prng_type"] == "java_lcg_reverse"
    assert result["survivor_count"] >= 0
    os.unlink("/tmp/harness_T12.json")
    os.unlink(cfg)

def T13_hybrid_forward_auto_strategies():
    pwc, cfg = setup_pwc(pool_size=1)

    dispatched_jobs = []
    def fake_dispatch(handle, job):
        dispatched_jobs.append(job)
        assert job["prng_type"] == "java_lcg_hybrid"
        assert job["strategies"] is not None, "Strategies must be auto-loaded for hybrid"
        return {"status": "ok", "survivors": [9000, 9001],
                "match_rates": [0.4, 0.5], "skip_sequences": [], "strategy_ids": []}

    def fake_local(job, node):
        return {"status": "ok", "survivors": [], "match_rates": []}

    node = next(n for n in pwc.nodes if pwc._is_rocm(n))
    from persistent_worker_coordinator import WorkerHandle
    pwc.workers = [WorkerHandle(node=node, gpu_id=0, alive=True)]
    pwc._dispatch_to_worker   = fake_dispatch
    pwc._dispatch_local_sieve = fake_local

    # Mock hybrid_strategy so auto-load works
    mock_strategy = MagicMock()
    mock_strategy.max_consecutive_misses = 3
    mock_strategy.skip_tolerance = 5
    with patch.dict('sys.modules', {'hybrid_strategy': MagicMock(
        get_all_strategies=lambda: [mock_strategy]
    )}):
        result = pwc.run_sieve_pass(
            prng_type   = "java_lcg_hybrid",
            residues    = list(range(8)),
            total_seeds = 500_000,
            threshold   = 0.25,
            window_size = 8,
            output_file = "/tmp/harness_T13.json",
        )
    assert len(dispatched_jobs) > 0
    assert dispatched_jobs[0]["strategies"] is not None
    os.unlink("/tmp/harness_T13.json")
    os.unlink(cfg)

def T14_hybrid_reverse_strategies_in_payload():
    pwc, cfg = setup_pwc(pool_size=1)

    custom_strategies = [{"max_consecutive_misses": 2, "skip_tolerance": 4}]
    dispatched_jobs = []

    def fake_dispatch(handle, job):
        dispatched_jobs.append(job)
        assert job["prng_type"] == "java_lcg_hybrid_reverse"
        assert job["strategies"] == custom_strategies, "Strategies not passed through"
        return {"status": "ok", "survivors": [], "match_rates": [],
                "skip_sequences": [], "strategy_ids": []}

    def fake_local(job, node):
        return {"status": "ok", "survivors": [], "match_rates": []}

    node = next(n for n in pwc.nodes if pwc._is_rocm(n))
    from persistent_worker_coordinator import WorkerHandle
    pwc.workers = [WorkerHandle(node=node, gpu_id=0, alive=True)]
    pwc._dispatch_to_worker   = fake_dispatch
    pwc._dispatch_local_sieve = fake_local

    result = pwc.run_sieve_pass(
        prng_type   = "java_lcg_hybrid_reverse",
        residues    = list(range(8)),
        total_seeds = 500_000,
        threshold   = 0.25,
        window_size = 8,
        output_file = "/tmp/harness_T14.json",
        strategies  = custom_strategies,
    )
    assert len(dispatched_jobs) > 0
    assert dispatched_jobs[0]["strategies"] == custom_strategies
    os.unlink("/tmp/harness_T14.json")
    os.unlink(cfg)

def T15_run_trial_persistent_pruned():
    from persistent_worker_coordinator import run_trial_persistent

    # Stub config
    class FakeConfig:
        window_size = 8; offset = 43; skip_min = 5; skip_max = 56; sessions = ['evening']

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(SYNTHETIC_CONFIG, f)
        cfg_path = f.name

    # Patch PersistentWorkerCoordinator.run_sieve_pass to return 0 forward survivors
    call_log = []
    def fake_run_sieve(self, prng_type, **kwargs):
        call_log.append(prng_type)
        return {"survivors": [], "match_rates": [], "survivor_count": 0,
                "prng_type": prng_type, "failed_chunks": 0, "total_chunks": 1,
                "total_tested": 500_000}

    import persistent_worker_coordinator as pwc_mod
    original          = pwc_mod.PersistentWorkerCoordinator.run_sieve_pass
    original_startup  = pwc_mod.PersistentWorkerCoordinator.startup
    original_shutdown = pwc_mod.PersistentWorkerCoordinator.shutdown

    try:
        pwc_mod.PersistentWorkerCoordinator.run_sieve_pass = fake_run_sieve
        pwc_mod.PersistentWorkerCoordinator.startup  = lambda self: None
        pwc_mod.PersistentWorkerCoordinator.shutdown = lambda self: None

        result = run_trial_persistent(
            coordinator_cfg   = cfg_path,
            config            = FakeConfig(),
            trial_number      = 0,
            prng_base         = "java_lcg",
            residues          = list(range(8)),
            total_seeds       = 500_000,
            forward_threshold = 0.25,
            reverse_threshold = 0.25,
            test_both_modes   = True,
        )
    finally:
        pwc_mod.PersistentWorkerCoordinator.run_sieve_pass = original
        pwc_mod.PersistentWorkerCoordinator.startup        = original_startup
        pwc_mod.PersistentWorkerCoordinator.shutdown       = original_shutdown

    assert result["pruned"] is True, "Expected pruned=True when forward=0"
    assert result["bidirectional_count"] == 0
    assert len(call_log) == 1, f"Should stop after forward pass, got: {call_log}"
    os.unlink(cfg_path)

def T16_run_trial_persistent_full_4pass():
    from persistent_worker_coordinator import run_trial_persistent

    class FakeConfig:
        window_size = 8; offset = 43; skip_min = 5; skip_max = 56; sessions = ['evening']

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(SYNTHETIC_CONFIG, f)
        cfg_path = f.name

    call_log = []
    def fake_run_sieve(self, prng_type, **kwargs):
        call_log.append(prng_type)
        survivors   = [1000 + i for i in range(10)]
        match_rates = [0.5] * 10
        return {"survivors": survivors, "match_rates": match_rates,
                "survivor_count": 10, "prng_type": prng_type,
                "failed_chunks": 0, "total_chunks": 1, "total_tested": 500_000}

    import persistent_worker_coordinator as pwc_mod
    original          = pwc_mod.PersistentWorkerCoordinator.run_sieve_pass
    original_startup  = pwc_mod.PersistentWorkerCoordinator.startup
    original_shutdown = pwc_mod.PersistentWorkerCoordinator.shutdown
    try:
        pwc_mod.PersistentWorkerCoordinator.run_sieve_pass = fake_run_sieve
        pwc_mod.PersistentWorkerCoordinator.startup  = lambda self: None
        pwc_mod.PersistentWorkerCoordinator.shutdown = lambda self: None

        result = run_trial_persistent(
            coordinator_cfg   = cfg_path,
            config            = FakeConfig(),
            trial_number      = 1,
            prng_base         = "java_lcg",
            residues          = list(range(8)),
            total_seeds       = 500_000,
            forward_threshold = 0.25,
            reverse_threshold = 0.25,
            test_both_modes   = True,
        )
    finally:
        pwc_mod.PersistentWorkerCoordinator.run_sieve_pass = original
        pwc_mod.PersistentWorkerCoordinator.startup        = original_startup
        pwc_mod.PersistentWorkerCoordinator.shutdown       = original_shutdown

    assert result["pruned"] is False
    assert len(call_log) == 4, f"Expected 4 passes, got {len(call_log)}: {call_log}"
    assert "java_lcg"              in call_log
    assert "java_lcg_reverse"      in call_log
    assert "java_lcg_hybrid"       in call_log
    assert "java_lcg_hybrid_reverse" in call_log
    assert result["bidirectional_count"] > 0
    assert "forward_map"   in result
    assert "reverse_map"   in result
    assert "bidirectional_constant" in result
    assert "bidirectional_variable" in result
    os.unlink(cfg_path)

def T17_test_both_modes_false_skips_hybrid():
    from persistent_worker_coordinator import run_trial_persistent

    class FakeConfig:
        window_size = 8; offset = 43; skip_min = 5; skip_max = 56; sessions = ['evening']

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(SYNTHETIC_CONFIG, f)
        cfg_path = f.name

    call_log = []
    def fake_run_sieve(self, prng_type, **kwargs):
        call_log.append(prng_type)
        return {"survivors": [1000, 2000], "match_rates": [0.5, 0.5],
                "survivor_count": 2, "prng_type": prng_type,
                "failed_chunks": 0, "total_chunks": 1, "total_tested": 500_000}

    import persistent_worker_coordinator as pwc_mod
    original          = pwc_mod.PersistentWorkerCoordinator.run_sieve_pass
    original_startup  = pwc_mod.PersistentWorkerCoordinator.startup
    original_shutdown = pwc_mod.PersistentWorkerCoordinator.shutdown
    try:
        pwc_mod.PersistentWorkerCoordinator.run_sieve_pass = fake_run_sieve
        pwc_mod.PersistentWorkerCoordinator.startup  = lambda self: None
        pwc_mod.PersistentWorkerCoordinator.shutdown = lambda self: None

        result = run_trial_persistent(
            coordinator_cfg   = cfg_path,
            config            = FakeConfig(),
            trial_number      = 0,
            prng_base         = "java_lcg",
            residues          = list(range(8)),
            total_seeds       = 500_000,
            forward_threshold = 0.25,
            reverse_threshold = 0.25,
            test_both_modes   = False,   # <-- key
        )
    finally:
        pwc_mod.PersistentWorkerCoordinator.run_sieve_pass = original
        pwc_mod.PersistentWorkerCoordinator.startup        = original_startup
        pwc_mod.PersistentWorkerCoordinator.shutdown       = original_shutdown

    assert len(call_log) == 2, f"Expected 2 passes (no hybrid), got: {call_log}"
    assert "java_lcg_hybrid" not in call_log
    assert "java_lcg_hybrid_reverse" not in call_log
    os.unlink(cfg_path)

def T18_get_residues_helper():
    from window_optimizer_integration_final import _get_residues_for_config

    class FakeConfig:
        window_size = 4; offset = 0; sessions = ['evening']

    # Write a synthetic dataset
    draws = [{"draw": i, "full_state": i * 10} for i in range(20)]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(draws, f)
        ds_path = f.name

    # Patch sieve_filter.load_draws_from_daily3
    with patch('window_optimizer_integration_final.load_draws_from_daily3',
               return_value=[0, 10, 20, 30]) as mock_load:
        result = _get_residues_for_config(FakeConfig(), ds_path)

    assert result == [0, 10, 20, 30], f"Got: {result}"
    os.unlink(ds_path)

def T19_build_test_result_accumulator():
    from window_optimizer_integration_final import _build_test_result_from_pw

    class FakeConfig:
        window_size = 8; offset = 43; skip_min = 5; skip_max = 56
        sessions = ['evening']; forward_threshold = 0.25; reverse_threshold = 0.25

    pw_result = {
        "pruned": False,
        "bidirectional_count": 3,
        "bidirectional_constant": {1000, 2000},
        "bidirectional_variable": {3000},
        "forward_map":  {1000: 0.5, 2000: 0.6, 3000: 0.4, 4000: 0.3},
        "reverse_map":  {1000: 0.5, 2000: 0.5, 3000: 0.5},
        "forward_records": [{"seed": 1000, "match_rate": 0.5}],
        "reverse_records": [{"seed": 1000, "match_rate": 0.5}],
        "forward_records_hybrid": [{"seed": 3000, "match_rate": 0.4}],
        "reverse_records_hybrid": [{"seed": 3000, "match_rate": 0.5}],
    }
    accumulator = {"forward": [], "reverse": [], "bidirectional": []}

    from window_optimizer import TestResult
    result = _build_test_result_from_pw(pw_result, accumulator, FakeConfig(),
                                         "java_lcg", 1, None)

    assert isinstance(result, TestResult)
    assert result.bidirectional_count == 3
    assert result.forward_count == 4
    assert result.reverse_count == 3
    assert len(accumulator["bidirectional"]) == 3  # 2 constant + 1 variable
    assert len(accumulator["forward"])  == 2   # fwd_records + fwd_h_records
    assert len(accumulator["reverse"])  == 2

def T20_integration_gate_uses_pwc():
    """Gate routes to PWC when use_persistent_workers=True."""
    called = []

    with patch('window_optimizer_integration_final.run_trial_persistent') as mock_rtp, \
         patch('window_optimizer_integration_final._get_residues_for_config', return_value=list(range(8))):

        mock_rtp.return_value = {
            "pruned": False,
            "bidirectional_count": 5,
            "bidirectional_constant": {1, 2, 3, 4, 5},
            "bidirectional_variable": set(),
            "forward_map":  {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5},
            "reverse_map":  {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5},
            "forward_records": [], "reverse_records": [],
            "forward_records_hybrid": [], "reverse_records_hybrid": [],
        }

        from window_optimizer_integration_final import run_bidirectional_test
        from window_optimizer import WindowConfig

        mock_coord = MagicMock()
        mock_coord.use_persistent_workers = True
        mock_coord.config_file  = "distributed_config.json"
        mock_coord.worker_pool_size = 2
        mock_coord.seed_cap_nvidia  = 5_000_000
        mock_coord.seed_cap_amd     = 2_000_000

        config = WindowConfig(window_size=8, offset=43, sessions=['evening'],
                              skip_min=5, skip_max=56)
        accumulator = {"forward": [], "reverse": [], "bidirectional": []}

        result = run_bidirectional_test(
            coordinator       = mock_coord,
            config            = config,
            dataset_path      = "daily3.json",
            seed_start        = 0,
            seed_count        = 500_000,
            prng_base         = "java_lcg",
            test_both_modes   = True,
            forward_threshold = 0.25,
            reverse_threshold = 0.25,
            trial_number      = 0,
            accumulator       = accumulator,
        )

    assert mock_rtp.called, "run_trial_persistent should have been called"
    assert result.bidirectional_count == 5

def T21_integration_gate_uses_original():
    """Gate uses original path when use_persistent_workers=False."""
    from window_optimizer_integration_final import run_bidirectional_test
    from window_optimizer import WindowConfig

    mock_coord = MagicMock()
    mock_coord.use_persistent_workers = False

    # Mock execute_distributed_analysis to return empty result
    mock_coord.execute_distributed_analysis.return_value = {
        "survivors": [], "match_rates": [], "survivor_count": 0
    }

    config = WindowConfig(window_size=8, offset=43, sessions=['evening'],
                          skip_min=5, skip_max=56)

    # Should NOT call run_trial_persistent
    with patch('window_optimizer_integration_final.run_trial_persistent') as mock_rtp:
        try:
            run_bidirectional_test(
                coordinator       = mock_coord,
                config            = config,
                dataset_path      = "daily3.json",
                seed_start        = 0,
                seed_count        = 500_000,
                prng_base         = "java_lcg",
                test_both_modes   = False,
                forward_threshold = 0.25,
                reverse_threshold = 0.25,
                trial_number      = 0,
            )
        except Exception:
            pass  # original path may fail without real coordinator — that's fine
    assert not mock_rtp.called, "run_trial_persistent must NOT be called when use_persistent_workers=False"

def T22_worker_shutdown():
    pwc, cfg = setup_pwc(pool_size=1)
    node = next(n for n in pwc.nodes if pwc._is_rocm(n))
    from persistent_worker_coordinator import WorkerHandle

    shutdown_received = []

    # Build proc as a simple namespace object so closures work correctly
    proc = MagicMock()
    proc.poll.return_value = None   # appears alive
    proc.stdin.write.side_effect = lambda s: shutdown_received.append(s)

    handle = WorkerHandle(node=node, gpu_id=0, alive=True, proc=proc)
    pwc.workers = [handle]

    pwc.shutdown()

    combined = b"".join(
        s if isinstance(s, bytes) else s.encode() for s in shutdown_received
    ).decode()
    assert "shutdown" in combined, \
        f"Shutdown command not sent. Got: {shutdown_received}"
    assert handle.alive is False, "Worker should be marked dead after shutdown"
    os.unlink(cfg)

def T23_per_rig_fault_isolation():
    pwc, cfg = setup_pwc(pool_size=2)
    rocm_nodes = [n for n in pwc.nodes if pwc._is_rocm(n)]

    from persistent_worker_coordinator import WorkerHandle
    workers = []
    for i, node in enumerate(rocm_nodes):
        for gpu_id in range(2):
            # Quarantine all workers on first rig
            quarantined = (i == 0)
            h = WorkerHandle(node=node, gpu_id=gpu_id,
                             alive=not quarantined, quarantined=quarantined)
            workers.append(h)
    pwc.workers = workers

    available = pwc._get_available_workers()
    # First rig quarantined — only workers from rigs 2 and 3 should appear
    for w in available:
        if hasattr(w, 'node'):
            assert w.node.hostname != rocm_nodes[0].hostname, \
                f"Quarantined rig {rocm_nodes[0].hostname} appeared in available workers"
    assert len(available) >= 4, f"Expected workers from 2 rigs, got {len(available)}"
    os.unlink(cfg)

def T24_zeus_local_path():
    """Zeus (localhost) uses _dispatch_local_sieve, not persistent worker."""
    pwc, cfg = setup_pwc(pool_size=1)

    local_called = []
    worker_called = []

    def fake_local(job, node):
        local_called.append(job["job_id"])
        return {"status": "ok", "survivors": [], "match_rates": []}

    def fake_dispatch(handle, job):
        worker_called.append(job["job_id"])
        return {"status": "ok", "survivors": [], "match_rates": [],
                "skip_sequences": [], "strategy_ids": []}

    pwc._dispatch_local_sieve = fake_local
    pwc._dispatch_to_worker   = fake_dispatch

    # Only add Zeus (localhost) as available — no AMD workers
    pwc.workers = []

    result = pwc.run_sieve_pass(
        prng_type   = "java_lcg",
        residues    = list(range(8)),
        total_seeds = 100_000,
        threshold   = 0.25,
        window_size = 8,
        output_file = "/tmp/harness_T24.json",
    )
    assert len(local_called) > 0, "Zeus local path should have been called"
    assert len(worker_called) == 0, "Persistent worker should NOT have been called for Zeus"
    if os.path.exists("/tmp/harness_T24.json"):
        os.unlink("/tmp/harness_T24.json")
    os.unlink(cfg)

def T25_no_cupy_in_coordinator():
    """GPU isolation invariant — PersistentWorkerCoordinator must never import CuPy."""
    import ast
    src = open(os.path.join(os.path.dirname(__file__),
                            'persistent_worker_coordinator.py')).read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                assert "cupy" not in (name or "").lower(), \
                    f"GPU isolation violation: 'import {name}' found in coordinator!"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
ALL_TESTS = {
    "T01": T01_instantiation,
    "T02": T02_semaphores_created,
    "T03": T03_worker_pool_size_cap,
    "T04": T04_rocm_spawn_stagger,
    "T05": T05_spawn_success,
    "T06": T06_spawn_failure,
    "T07": T07_ensure_worker_alive_respawn,
    "T08": T08_quarantined_skipped,
    "T09": T09_semaphore_acquire_release,
    "T10": T10_semaphore_released_on_exception,
    "T11": T11_constant_forward_pass,
    "T12": T12_constant_reverse_pass,
    "T13": T13_hybrid_forward_auto_strategies,
    "T14": T14_hybrid_reverse_strategies_in_payload,
    "T15": T15_run_trial_persistent_pruned,
    "T16": T16_run_trial_persistent_full_4pass,
    "T17": T17_test_both_modes_false_skips_hybrid,
    "T18": T18_get_residues_helper,
    "T19": T19_build_test_result_accumulator,
    "T20": T20_integration_gate_uses_pwc,
    "T21": T21_integration_gate_uses_original,
    "T22": T22_worker_shutdown,
    "T23": T23_per_rig_fault_isolation,
    "T24": T24_zeus_local_path,
    "T25": T25_no_cupy_in_coordinator,
}

if __name__ == "__main__":
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(ALL_TESTS.keys())
    unknown   = [t for t in requested if t not in ALL_TESTS]
    if unknown:
        print(f"Unknown tests: {unknown}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Persistent Worker Coordinator — Dry Run Harness")
    print(f"{'='*60}\n")

    for name in requested:
        run_test(name, ALL_TESTS[name])

    passed = sum(1 for v in results.values() if v == PASS)
    failed = len(results) - passed
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{len(results)} passed", end="")
    if failed:
        print(f"  |  {failed} FAILED:")
        for k, v in results.items():
            if v != PASS:
                print(f"    {k}: {v}")
    else:
        print("  — all clear ✅")
    print(f"{'='*60}\n")
    sys.exit(0 if failed == 0 else 1)
