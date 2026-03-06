#!/usr/bin/env python3
"""
S115 Dry-Run Harness
====================
Validates all proposed changes from RESPONSE_TeamBeta_S115_v3 before implementation.

Tests:
  T1  — M4: node_allowlist filter correctness (IPs, zero-node guard, print)
  T2  — M4: zero-node guard fires on bad allowlist
  T3  — M3: analysis_id collision — confirmed dead code path (execute_truly_parallel_dynamic)
  T4  — M3: output path uniqueness with _t{trial_number} suffix
  T5  — M1: partition coordinator construction + node isolation
  T6  — M1/M5: test_config partition routing via trial.number % n_parallel
  T7  — M2: forward_count == 0 pruning hook raises TrialPruned
  T8  — M2: forward_count > 0 does NOT prune
  T9  — N2: _OPTUNA_AVAILABLE guard — prune hook graceful when optuna missing
  T10 — N3: SSHConnectionPool cleanup_all exists and is callable
  T11 — End-to-end: 20-trial Optuna study with mock sieve, pruning + parallel routing

Results printed with PASS/FAIL per test.
"""

import sys
import os
import json
import time
import threading
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

results = []

def PASS(name, note=""):
    msg = f"{GREEN}✅ PASS{RESET}  {name}" + (f"  ({note})" if note else "")
    print(msg)
    results.append(("PASS", name))

def FAIL(name, reason):
    msg = f"{RED}❌ FAIL{RESET}  {name}  →  {reason}"
    print(msg)
    results.append(("FAIL", name, reason))

def INFO(msg):
    print(f"   {YELLOW}→{RESET} {msg}")

print(f"\n{BOLD}{'='*65}{RESET}")
print(f"{BOLD}S115 DRY-RUN HARNESS{RESET}")
print(f"{BOLD}{'='*65}{RESET}\n")

# ─────────────────────────────────────────────────────────────────────────────
# APPLY PATCHES TO LIVE CODE (in-memory, no file writes)
# ─────────────────────────────────────────────────────────────────────────────
# Patch coordinator to add node_allowlist before importing

import coordinator as _coord_module

# Store original load_configuration
_orig_load_configuration = _coord_module.MultiGPUCoordinator.load_configuration
_orig_init = _coord_module.MultiGPUCoordinator.__init__

def _patched_init(self, config_file="distributed_config.json",
                  seed_cap_nvidia=40000, seed_cap_amd=19000, seed_cap_default=19000,
                  max_concurrent=8, max_per_node=4, max_local_concurrent=None,
                  job_timeout=600, resume_policy='prompt',
                  node_allowlist=None):          # NEW param
    # CRITICAL: set node_allowlist BEFORE calling _orig_init because
    # _orig_init calls self.load_configuration() internally, which runs
    # _patched_load_configuration, which checks self.node_allowlist.
    self.node_allowlist = node_allowlist
    _orig_init(self, config_file, seed_cap_nvidia, seed_cap_amd, seed_cap_default,
               max_concurrent, max_per_node, max_local_concurrent, job_timeout, resume_policy)

def _patched_load_configuration(self):
    _orig_load_configuration(self)
    # Apply allowlist filter AFTER nodes are populated
    if self.node_allowlist is not None:
        original_nodes = list(self.nodes)
        before = len(self.nodes)
        self.nodes = [n for n in self.nodes if n.hostname in self.node_allowlist]
        print(f"   Node allowlist active: {len(self.nodes)}/{before} nodes "
              f"({[n.hostname for n in self.nodes]})")
        if not self.nodes:
            raise ValueError(
                f"node_allowlist {self.node_allowlist} matched no nodes in config.\n"
                f"Available hostnames: {[n.hostname for n in original_nodes]}\n"
                f"Check for hostname vs IP mismatch in distributed_config.json."
            )

_coord_module.MultiGPUCoordinator.__init__ = _patched_init
_coord_module.MultiGPUCoordinator.load_configuration = _patched_load_configuration

from coordinator import MultiGPUCoordinator

# ─────────────────────────────────────────────────────────────────────────────
# T1 — M4: node_allowlist filter — correct IPs, expected nodes retained, print emitted
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}── T1: M4 node_allowlist filter (valid IPs){RESET}")
try:
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        coord_a = MultiGPUCoordinator(
            config_file="distributed_config.json",
            resume_policy="restart",
            node_allowlist=['localhost', '192.168.3.120']
        )
        coord_a.load_configuration()

    output = buf.getvalue()
    hostnames_a = [n.hostname for n in coord_a.nodes]
    INFO(f"Partition 0 nodes: {hostnames_a}")
    INFO(f"Print output: {output.strip()!r}")

    assert hostnames_a == ['localhost', '192.168.3.120'], f"Wrong nodes: {hostnames_a}"
    assert len(coord_a.nodes) == 2
    assert "Node allowlist active: 2/4 nodes" in output
    PASS("T1", f"2/4 nodes retained, print emitted")
except Exception as e:
    FAIL("T1", str(e))

# ─────────────────────────────────────────────────────────────────────────────
# T2 — M4: zero-node guard fires on bad allowlist
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}── T2: M4 zero-node guard (bad allowlist){RESET}")
try:
    coord_bad = MultiGPUCoordinator(
        config_file="distributed_config.json",
        resume_policy="restart",
        node_allowlist=['rig-6600', 'rig-6600b']   # hostnames not in config (IPs are)
    )
    coord_bad.load_configuration()
    FAIL("T2", "Should have raised ValueError but did not")
except ValueError as e:
    INFO(f"ValueError raised: {str(e)[:120]}")
    assert "matched no nodes" in str(e)
    assert "Available hostnames" in str(e)
    PASS("T2", "ValueError raised with diagnostic message")
except Exception as e:
    FAIL("T2", f"Wrong exception type: {type(e).__name__}: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# T3 — M3: analysis_id collision confirmed DEAD CODE (execute_truly_parallel_dynamic)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}── T3: M3 analysis_id collision — dead code path confirmation{RESET}")
try:
    with open('coordinator.py') as f:
        src = f.read()

    # Find execute_distributed_analysis routing
    idx = src.find('def execute_distributed_analysis(self, target_file')
    chunk = src[idx:idx+1500]

    # Confirm all paths lead to execute_truly_parallel_dynamic
    routes_to_dynamic = 'return self.execute_truly_parallel_dynamic(' in chunk
    # Confirm execute_truly_parallel_dynamic does NOT use analysis_id
    idx2 = src.find('def execute_truly_parallel_dynamic')
    chunk2 = src[idx2:idx2+4000]
    dynamic_uses_analysis_id = 'analysis_id' in chunk2
    dynamic_uses_recovery = 'recovery_manager' in chunk2

    INFO(f"execute_distributed_analysis routes to dynamic: {routes_to_dynamic}")
    INFO(f"execute_truly_parallel_dynamic uses analysis_id: {dynamic_uses_analysis_id}")
    INFO(f"execute_truly_parallel_dynamic uses recovery_manager: {dynamic_uses_recovery}")

    assert routes_to_dynamic, "Expected routing to execute_truly_parallel_dynamic"
    assert not dynamic_uses_analysis_id, "Dynamic path unexpectedly uses analysis_id"
    assert not dynamic_uses_recovery, "Dynamic path unexpectedly uses recovery_manager"
    PASS("T3", "analysis_id collision is dead code — no fix needed in execute_truly_parallel_dynamic")
except Exception as e:
    FAIL("T3", str(e))

# ─────────────────────────────────────────────────────────────────────────────
# T4 — M3: output path uniqueness with _t{trial_number} suffix
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}── T4: M3 output path uniqueness per trial{RESET}")
try:
    # Simulate the 4 output path templates for two concurrent trials
    # with identical window_size=512, offset=0 but different trial_numbers
    def make_paths(window_size, offset, trial_number):
        return [
            f'results/window_opt_forward_{window_size}_{offset}_t{trial_number}.json',
            f'results/window_opt_reverse_{window_size}_{offset}_t{trial_number}.json',
            f'results/window_opt_forward_hybrid_{window_size}_{offset}_t{trial_number}.json',
            f'results/window_opt_reverse_hybrid_{window_size}_{offset}_t{trial_number}.json',
        ]

    paths_t1 = make_paths(512, 0, 1)
    paths_t2 = make_paths(512, 0, 2)

    INFO(f"Trial 1 paths: {paths_t1[:2]}")
    INFO(f"Trial 2 paths: {paths_t2[:2]}")

    # No collisions
    all_paths = paths_t1 + paths_t2
    assert len(set(all_paths)) == len(all_paths), "Path collision detected!"

    # Same window/offset without suffix would collide
    old_paths_t1 = [f'results/window_opt_forward_512_0.json', f'results/window_opt_reverse_512_0.json']
    old_paths_t2 = [f'results/window_opt_forward_512_0.json', f'results/window_opt_reverse_512_0.json']
    assert old_paths_t1 == old_paths_t2, "Old paths should collide (confirming the bug)"

    PASS("T4", "4 output paths unique per trial; old paths confirmed to collide")
except Exception as e:
    FAIL("T4", str(e))

# ─────────────────────────────────────────────────────────────────────────────
# T5 — M1: partition coordinator construction + node isolation
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}── T5: M1 partition coordinator construction + isolation{RESET}")
try:
    PARALLEL_PARTITIONS = {
        0: ['localhost', '192.168.3.120'],
        1: ['192.168.3.154', '192.168.3.162'],
    }

    _partition_coordinators = {}

    def get_partition_coordinator(partition_idx):
        if partition_idx not in _partition_coordinators:
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                allowlist = PARALLEL_PARTITIONS[partition_idx % len(PARALLEL_PARTITIONS)]
                coord = MultiGPUCoordinator(
                    config_file='distributed_config.json',
                    node_allowlist=allowlist
                )
                coord.load_configuration()
                coord.create_gpu_workers()
            _partition_coordinators[partition_idx] = coord
            output = buf.getvalue()
            INFO(f"Partition {partition_idx}: {[n.hostname for n in coord.nodes]} "
                 f"→ {len(coord.gpu_workers)} GPUs")
        return _partition_coordinators[partition_idx]

    coord0 = get_partition_coordinator(0)
    coord1 = get_partition_coordinator(1)

    nodes0 = {n.hostname for n in coord0.nodes}
    nodes1 = {n.hostname for n in coord1.nodes}

    INFO(f"Partition 0 nodes: {nodes0}  GPUs: {len(coord0.gpu_workers)}")
    INFO(f"Partition 1 nodes: {nodes1}  GPUs: {len(coord1.gpu_workers)}")

    # Check disjoint
    assert nodes0 & nodes1 == set(), f"Partitions share nodes: {nodes0 & nodes1}"
    assert nodes0 | nodes1 == {'localhost','192.168.3.120','192.168.3.154','192.168.3.162'}
    assert len(coord0.gpu_workers) == 10  # 2 + 8
    assert len(coord1.gpu_workers) == 16  # 8 + 8

    # Check SSH pools are separate instances
    assert coord0.ssh_pool is not coord1.ssh_pool, "SSH pools must be separate instances"

    # Check cache works (second call returns same object)
    coord0_again = get_partition_coordinator(0)
    assert coord0_again is coord0, "Cache should return same coordinator instance"

    PASS("T5", f"Partition 0: {len(coord0.gpu_workers)} GPUs, Partition 1: {len(coord1.gpu_workers)} GPUs, disjoint, SSH pools separate")

    # Cleanup
    for c in _partition_coordinators.values():
        try:
            c.ssh_pool.cleanup_all()
        except Exception:
            pass
    _partition_coordinators.clear()

except Exception as e:
    FAIL("T5", str(e))
    import traceback; traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# T6 — M1/M5: test_config partition routing via trial.number % n_parallel
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}── T6: M1/M5 partition routing + per-trial log{RESET}")
try:
    PARALLEL_PARTITIONS = {
        0: ['localhost', '192.168.3.120'],
        1: ['192.168.3.154', '192.168.3.162'],
    }
    _partition_coordinators = {}

    def get_partition_coordinator(partition_idx):
        if partition_idx not in _partition_coordinators:
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                allowlist = PARALLEL_PARTITIONS[partition_idx]
                coord = MultiGPUCoordinator(config_file='distributed_config.json',
                                            node_allowlist=allowlist)
                coord.load_configuration()
                coord.create_gpu_workers()
            _partition_coordinators[partition_idx] = coord
        return _partition_coordinators[partition_idx]

    # Mock coordinator (the 'self' inside optimize_window)
    full_coordinator = MagicMock()
    full_coordinator.hostname = 'localhost'

    log = []

    def test_config_sim(config_stub, optuna_trial=None, n_parallel=2):
        """Simulates the patched test_config closure"""
        if optuna_trial is not None and n_parallel > 1:
            partition_idx = optuna_trial.number % n_parallel
            active_coordinator = get_partition_coordinator(partition_idx)
            log.append({
                'trial': optuna_trial.number,
                'partition': partition_idx,
                'nodes': [n.hostname for n in active_coordinator.nodes]
            })
            print(f"   🔀 Trial {optuna_trial.number} → Partition {partition_idx} "
                  f"({PARALLEL_PARTITIONS[partition_idx]})")
        else:
            active_coordinator = full_coordinator
            log.append({'trial': getattr(optuna_trial, 'number', None), 'partition': None})
        return active_coordinator

    # Simulate 6 trials
    for trial_num in range(6):
        mock_trial = MagicMock()
        mock_trial.number = trial_num
        test_config_sim("config_stub", optuna_trial=mock_trial, n_parallel=2)

    # n_parallel=1: should always use full_coordinator
    mock_trial_single = MagicMock(); mock_trial_single.number = 0
    result_single = test_config_sim("config_stub", optuna_trial=mock_trial_single, n_parallel=1)
    assert result_single is full_coordinator, "n_parallel=1 should use full coordinator"

    # Verify alternating partition assignment
    for entry in log[:6]:
        expected = entry['trial'] % 2
        assert entry['partition'] == expected, \
            f"Trial {entry['trial']} → partition {entry['partition']}, expected {expected}"

    INFO(f"Routing log: {[(e['trial'], e['partition']) for e in log[:6]]}")
    PASS("T6", "Trials alternate partitions correctly; n_parallel=1 uses full coordinator")

    for c in _partition_coordinators.values():
        try: c.ssh_pool.cleanup_all()
        except: pass
    _partition_coordinators.clear()

except Exception as e:
    FAIL("T6", str(e))
    import traceback; traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# T7 — M2: forward_count == 0 raises TrialPruned
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}── T7: M2 forward_count == 0 raises TrialPruned{RESET}")
try:
    import optuna
    _OPTUNA_AVAILABLE = True

    prune_log = []

    def pruning_hook(forward_records, optuna_trial, trial_number):
        """Simulates the M2 pruning hook from run_bidirectional_test"""
        if optuna_trial is not None:
            if not _OPTUNA_AVAILABLE:
                print("⚠️  optuna_trial passed but Optuna not installed — pruning disabled.")
                return False  # no prune
            forward_count = len(forward_records)
            if forward_count == 0:
                prune_log.append(trial_number)
                print(f"      ✂️  PRUNED  trial={optuna_trial.number}  "
                      f"forward_count=0")
                raise optuna.exceptions.TrialPruned()
        return False  # no prune

    # Dead trial (forward_count=0) → should prune
    mock_trial = MagicMock(); mock_trial.number = 5
    pruned = False
    try:
        pruning_hook([], mock_trial, trial_number=5)
    except optuna.exceptions.TrialPruned:
        pruned = True

    assert pruned, "Should have raised TrialPruned for forward_count=0"
    assert 5 in prune_log
    PASS("T7", "forward_count=0 → TrialPruned raised")

except Exception as e:
    FAIL("T7", str(e))

# ─────────────────────────────────────────────────────────────────────────────
# T8 — M2: forward_count > 0 does NOT prune
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}── T8: M2 forward_count > 0 does NOT prune{RESET}")
try:
    import optuna
    _OPTUNA_AVAILABLE = True

    mock_trial = MagicMock(); mock_trial.number = 3
    pruned = False
    try:
        # 5 forward survivors — should NOT prune
        forward_records = [{'seed': i, 'match_rate': 0.5} for i in range(5)]
        forward_count = len(forward_records)
        if forward_count == 0:
            raise optuna.exceptions.TrialPruned()
        # reaches here — no prune
    except optuna.exceptions.TrialPruned:
        pruned = True

    assert not pruned, "Should NOT prune when forward_count > 0"
    PASS("T8", f"forward_count=5 → proceeds to reverse sieve (no prune)")

except Exception as e:
    FAIL("T8", str(e))

# ─────────────────────────────────────────────────────────────────────────────
# T9 — N2: _OPTUNA_AVAILABLE guard — graceful when optuna unavailable
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}── T9: N2 _OPTUNA_AVAILABLE=False — graceful fallback{RESET}")
try:
    import io, contextlib
    _OPTUNA_AVAILABLE_sim = False

    buf = io.StringIO()
    pruned = False
    with contextlib.redirect_stdout(buf):
        mock_trial = MagicMock(); mock_trial.number = 7
        forward_records = []   # would prune if optuna available

        if mock_trial is not None:
            if not _OPTUNA_AVAILABLE_sim:
                print("⚠️  optuna_trial passed but Optuna not installed — pruning disabled.")
                # fall through, no prune
            else:
                if len(forward_records) == 0:
                    pruned = True

    output = buf.getvalue()
    INFO(f"Output: {output.strip()!r}")
    assert not pruned, "Should not prune when _OPTUNA_AVAILABLE=False"
    assert "pruning disabled" in output
    PASS("T9", "Graceful fallback when Optuna unavailable")

except Exception as e:
    FAIL("T9", str(e))

# ─────────────────────────────────────────────────────────────────────────────
# T10 — N3: SSHConnectionPool cleanup_all exists and is callable
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}── T10: N3 SSHConnectionPool.cleanup_all exists{RESET}")
try:
    from connection_manager import SSHConnectionPool
    pool = SSHConnectionPool(max_concurrent_per_node=4, max_connections_per_node=4)
    assert hasattr(pool, 'cleanup_all'), "cleanup_all method missing"
    assert callable(pool.cleanup_all), "cleanup_all not callable"

    # Call it on an empty pool — should not raise
    pool.cleanup_all()
    PASS("T10", "cleanup_all exists, callable, safe on empty pool")
except Exception as e:
    FAIL("T10", str(e))

# ─────────────────────────────────────────────────────────────────────────────
# T11 — End-to-end: 20-trial Optuna study with mock sieve + pruning + parallel routing
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}── T11: End-to-end 20-trial study (mock sieve, pruning, parallel routing){RESET}")
try:
    import optuna
    import time
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    PARALLEL_PARTITIONS = {
        0: ['localhost', '192.168.3.120'],
        1: ['192.168.3.154', '192.168.3.162'],
    }
    _partition_coordinators = {}

    def get_partition_coordinator(partition_idx):
        if partition_idx not in _partition_coordinators:
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                coord = MultiGPUCoordinator(
                    config_file='distributed_config.json',
                    node_allowlist=PARALLEL_PARTITIONS[partition_idx]
                )
                coord.load_configuration()
                coord.create_gpu_workers()
            _partition_coordinators[partition_idx] = coord
        return _partition_coordinators[partition_idx]

    # Tracking
    trial_log = []
    _lock = threading.Lock()

    def mock_run_bidirectional_test(coordinator, window_size, trial_number,
                                    optuna_trial=None):
        """
        Mock sieve: window_size=3 → 5000 survivors (live trial)
                    all others    → 0 survivors (pruned)
        """
        _OPTUNA_AVAILABLE = True
        # Mock forward sieve
        forward_records = [{'seed': i} for i in range(5000)] if window_size == 3 else []

        # M2 pruning hook
        if optuna_trial is not None and _OPTUNA_AVAILABLE:
            if len(forward_records) == 0:
                with _lock:
                    trial_log.append({
                        'trial': trial_number,
                        'optuna_num': optuna_trial.number,
                        'state': 'PRUNED',
                        'coordinator_nodes': [n.hostname for n in coordinator.nodes]
                            if hasattr(coordinator, 'nodes') else ['mock'],
                    })
                raise optuna.exceptions.TrialPruned()

        # Mock reverse sieve (only runs if not pruned)
        reverse_records = [{'seed': i} for i in range(4800)] if window_size == 3 else []
        bidirectional = len(set(r['seed'] for r in forward_records) &
                           set(r['seed'] for r in reverse_records))

        with _lock:
            trial_log.append({
                'trial': trial_number,
                'optuna_num': optuna_trial.number if optuna_trial else None,
                'state': 'COMPLETE',
                'forward': len(forward_records),
                'reverse': len(reverse_records),
                'bidirectional': bidirectional,
                'coordinator_nodes': [n.hostname for n in coordinator.nodes]
                    if hasattr(coordinator, 'nodes') else ['mock'],
            })
        return bidirectional

    # Simulate optimize_window with n_parallel=2
    n_parallel = 2
    trial_counter = {'count': 0}

    # Mock full coordinator (n_parallel=1 fallback)
    full_coord = MagicMock()
    full_coord.nodes = []

    def test_config(window_size, optuna_trial=None):
        trial_counter['count'] += 1

        if optuna_trial is not None and n_parallel > 1:
            partition_idx = optuna_trial.number % n_parallel
            active_coordinator = get_partition_coordinator(partition_idx)
        else:
            active_coordinator = full_coord

        return mock_run_bidirectional_test(
            coordinator=active_coordinator,
            window_size=window_size,
            trial_number=trial_counter['count'],
            optuna_trial=optuna_trial
        )

    # Create study with ThresholdPruner + JournalFileBackend
    import os
    studies_dir = "/tmp/optuna_studies_s115_test"
    os.makedirs(studies_dir, exist_ok=True)
    study_name = f"s115_dryrun_{int(time.time())}"
    journal_path = f"{studies_dir}/{study_name}.jsonl"

    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(journal_path)
    )
    pruner = optuna.pruners.ThresholdPruner(lower=1.0)
    sampler = optuna.samplers.TPESampler(n_startup_trials=3, seed=42)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        load_if_exists=False
    )

    # Enqueue warm-start trial with window_size=3
    study.enqueue_trial({'window_size': 3})

    # Objective
    def optuna_objective(trial):
        window_size = trial.suggest_int('window_size', 1, 10)
        result = test_config(window_size, optuna_trial=trial)
        return float(result)

    study.optimize(optuna_objective, n_trials=20, n_jobs=2)

    # Analyze results
    all_trials = study.trials
    n_pruned   = sum(1 for t in all_trials if t.state.name == 'PRUNED')
    n_complete = sum(1 for t in all_trials if t.state.name == 'COMPLETE')
    n_total    = len(all_trials)
    prune_rate = n_pruned / n_total * 100

    INFO(f"Total trials: {n_total}  Pruned: {n_pruned}  Complete: {n_complete}  "
         f"Prune rate: {prune_rate:.0f}%")
    INFO(f"Best trial: window_size={study.best_params.get('window_size')}  "
         f"value={study.best_value:.0f}")

    # Verify partition routing
    partition_0_trials = [e for e in trial_log
                          if e.get('coordinator_nodes') and
                          'localhost' in e.get('coordinator_nodes', [])]
    partition_1_trials = [e for e in trial_log
                          if e.get('coordinator_nodes') and
                          '192.168.3.154' in e.get('coordinator_nodes', [])]

    INFO(f"Partition 0 trial count: {len(partition_0_trials)}")
    INFO(f"Partition 1 trial count: {len(partition_1_trials)}")

    # Assertions
    assert n_pruned >= 14, f"Expected >= 14 pruned trials, got {n_pruned}"
    assert n_complete >= 1, "Expected at least 1 complete trial (window_size=3)"
    assert study.best_value > 0, "Best value should be > 0"
    assert study.best_params['window_size'] == 3, \
        f"Best should be window_size=3, got {study.best_params['window_size']}"
    assert len(partition_0_trials) > 0, "No trials routed to partition 0"
    assert len(partition_1_trials) > 0, "No trials routed to partition 1"
    assert os.path.exists(journal_path), "Journal file not created"

    # Prune telemetry summary (R1)
    print(f"\n   {'='*45}")
    print(f"   PRUNING SUMMARY (T11)")
    print(f"     Total trials:    {n_total}")
    print(f"     Pruned (fwd=0):  {n_pruned}  ({prune_rate:.0f}%)")
    print(f"     Completed:       {n_complete}")
    print(f"     Expected rate:   80-90% for window_size in [1..10] with live=1/10")
    print(f"   {'='*45}\n")

    PASS("T11", f"{n_pruned}/{n_total} pruned ({prune_rate:.0f}%), "
                f"best=window_size=3, both partitions used, journal created")

    # Cleanup
    for c in _partition_coordinators.values():
        try: c.ssh_pool.cleanup_all()
        except: pass
    _partition_coordinators.clear()
    import shutil
    shutil.rmtree(studies_dir, ignore_errors=True)

except Exception as e:
    FAIL("T11", str(e))
    import traceback; traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}{'='*65}{RESET}")
print(f"{BOLD}RESULTS SUMMARY{RESET}")
print(f"{BOLD}{'='*65}{RESET}")

passed = [r for r in results if r[0] == 'PASS']
failed = [r for r in results if r[0] == 'FAIL']

for r in results:
    if r[0] == 'PASS':
        print(f"  {GREEN}✅ PASS{RESET}  {r[1]}")
    else:
        print(f"  {RED}❌ FAIL{RESET}  {r[1]}  →  {r[2]}")

print(f"\n  {BOLD}Total: {len(results)}  Passed: {len(passed)}  Failed: {len(failed)}{RESET}")

if failed:
    print(f"\n  {RED}⚠️  {len(failed)} test(s) failed — fix before implementation{RESET}")
    sys.exit(1)
else:
    print(f"\n  {GREEN}All tests passed — proposal is implementation-ready{RESET}")
    sys.exit(0)
