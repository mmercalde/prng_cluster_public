#!/usr/bin/env python3
"""
scripts_coordinator.py v1.4.0 - Universal ML Script Job Orchestrator

Executes script-based jobs across 26-GPU cluster for Steps 3, 4, and 5.
ML-agnostic: works with Neural Networks, XGBoost, Optuna, K-Fold CV.

Conforms to established environment protocols:
- distributed_config.json for node/GPU configuration
- ROCm env vars (HSA_OVERRIDE_GFX_VERSION, HIP_VISIBLE_DEVICES) for AMD
- CUDA env vars (CUDA_VISIBLE_DEVICES) for NVIDIA
- distributed_worker.py as execution wrapper
- File-based success detection (no stdout parsing)

Team Beta Recommendations (v1.3.0):
- Run ID scoping: {output_dir}/{run_id}/
- Manifest file: scripts_run_manifest.json for auditability
- Explicit failure modes: MISSING | EMPTY | INVALID_JSON | TIMEOUT

New in v1.4.0:
- --output-dir: Flexible output directory (default: full_scoring_results)
- --preserve-paths: Don't rewrite job output paths (for Step 5 compatibility)
- --dry-run: Print job distribution without execution
- Dynamic run ID based on output directory name
- manifest_version field in manifest

SCRIPT JOB SPEC v1.0:
{
    "job_id": "string",
    "script": "worker_script.py",
    "args": ["--arg1", "value1", ...],
    "expected_output": "path/to/output.json",
    "timeout": 7200
}

Usage:
    # Step 3 (default)
    python3 scripts_coordinator.py --jobs-file scoring_jobs.json
    
    # Step 5 (custom output)
    python3 scripts_coordinator.py --jobs-file anti_overfit_jobs.json \\
        --output-dir anti_overfit_results --preserve-paths

Author: Distributed PRNG Analysis System
Date: December 18, 2025
Version: 1.4.0
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
# CONSTANTS (Established Environment Protocols)
# =============================================================================

ROCM_ENV_VARS = [
    "HSA_OVERRIDE_GFX_VERSION=10.3.0",
    "HSA_ENABLE_SDMA=0",
    "ROCM_PATH=/opt/rocm",
    "HIP_PATH=/opt/rocm/hip",
    "LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/hip/lib:${LD_LIBRARY_PATH}",
    "PATH=/opt/rocm/bin:${PATH}",
]

STAGGER_LOCALHOST = 3.0   # seconds - CUDA init collision prevention
STAGGER_REMOTE = 2.0      # seconds - ROCm needs less separation

ROCM_HOSTNAMES = ['192.168.3.120', '192.168.3.154', '192.168.3.162', 'rig-6600', 'rig-6600b', 'rig-6600c']
# =============================================================================
# JOB BATCHING (Team Beta Approved - 2026-01-22)
# =============================================================================
# Prevents rig overload when dispatching large trial counts.
# Step-aware batching: Step 3 is 6-7x heavier than Step 2.5.
#
# TEAM BETA RULING (2026-01-22):
#   Step 3 exceeded validated envelope with cap=6 (GPU unknown states, crashes)
#   Step 3: cap=2, cooldown=10s
#   Step 2.5: cap=6, cooldown=5s (unchanged)

MAX_JOBS_PER_BATCH = 20          # Validated stable limit (benchmark: 100% success)
ENABLE_ALLOCATOR_RESET = True    # Reset memory between batches (drop_caches)

# HIP kernel cache clear (Team Beta ruling: Jan 23, 2026)
# Clears stale compiled kernels that cause "invalid device function" errors
# Set CLEAR_HIP_CACHE=0 to disable, CLEAR_HIP_CACHE=force to clear even localhost
CLEAR_HIP_CACHE = os.environ.get('CLEAR_HIP_CACHE', '1')  # Default ON for ROCm nodes

# ROCm warm-up barrier (Team Beta: Jan 23, 2026)
# Warms each GPU sequentially ONCE per node per run_id to reduce first-wave init/compile races
# Set WARMUP_ROCM=0 to disable, WARMUP_ROCM=force to include localhost
WARMUP_ROCM = os.environ.get('WARMUP_ROCM', '1')  # Default ON for ROCm nodes  # Default ON for ROCm nodes

# Step 2.5 and other steps (default)
DEFAULT_MAX_JOBS_PER_NODE_PER_BATCH = 6   # ROCm nodes crash with >6 simultaneous HIP inits
DEFAULT_INTER_BATCH_COOLDOWN = 5.0        # Seconds between batches

# Step 3 (Full Scoring) - heavier workload requires conservative limits
STEP3_MAX_JOBS_PER_NODE_PER_BATCH = 8     # Full GPU utilization (8 per node)
STEP3_INTER_BATCH_COOLDOWN = 5.0         # Standard cooldown (5s)

# Legacy aliases (for backward compatibility with any direct references)
MAX_JOBS_PER_NODE_PER_BATCH = DEFAULT_MAX_JOBS_PER_NODE_PER_BATCH
INTER_BATCH_COOLDOWN = DEFAULT_INTER_BATCH_COOLDOWN

# =============================================================================
# STEP DETECTION (Team Beta - 2026-01-22)
# =============================================================================

def detect_job_step(jobs_file: str, jobs: list = None) -> str:
    """
    Detect which pipeline step these jobs belong to.
    
    Returns:
        'step3' for Full Scoring jobs
        'step2.5' for Scorer Meta-Optimizer jobs
        'unknown' for other job types
    
    Detection methods (in priority order):
    1. Explicit 'job_type' field in job spec
    2. Jobs file name pattern
    3. Script name in job command
    """
    jobs_file_lower = jobs_file.lower() if jobs_file else ''
    
    # Method 1: Check job specs for explicit job_type
    if jobs:
        for job in jobs[:5]:  # Check first 5 jobs
            job_type = getattr(job, 'job_type', '')
            if job_type == 'full_scoring':
                return 'step3'
            elif job_type == 'scorer_trial':
                return 'step2.5'
    
    # Method 2: Jobs file name pattern
    if 'scoring_jobs' in jobs_file_lower and 'scorer' not in jobs_file_lower:
        return 'step3'
    if 'full_scoring' in jobs_file_lower:
        return 'step3'
    if 'scorer_jobs' in jobs_file_lower or 'scorer_trial' in jobs_file_lower:
        return 'step2.5'
    
    # Method 3: Check script names in jobs
    if jobs:
        for job in jobs[:5]:
            cmd = job.get('command', '') or job.get('script', '')
            if 'full_scoring_worker' in cmd:
                return 'step3'
            if 'scorer_trial_worker' in cmd:
                return 'step2.5'
    
    return 'unknown'


def get_step_aware_limits(step: str) -> tuple:
    """
    Get batching limits for the detected step.
    
    Returns:
        (max_jobs_per_node, cooldown_seconds, step_name)
    """
    if step == 'step3':
        return (STEP3_MAX_JOBS_PER_NODE_PER_BATCH, 
                STEP3_INTER_BATCH_COOLDOWN,
                'Step 3 (Full Scoring)')
    else:
        return (DEFAULT_MAX_JOBS_PER_NODE_PER_BATCH,
                DEFAULT_INTER_BATCH_COOLDOWN,
                f'Step 2.5/Other ({step})')



MANIFEST_VERSION = "1.0.0"
SCRIPT_JOB_SPEC_VERSION = "1.0"


# =============================================================================
# FAILURE MODES (Team Beta Recommendation 3)
# =============================================================================

class FailureMode:
    """Explicit failure modes - no ambiguity"""
    MISSING = "FILE_MISSING"
    EMPTY = "FILE_EMPTY"
    INVALID_JSON = "INVALID_JSON"
    TIMEOUT = "TIMEOUT"
    SSH_ERROR = "SSH_ERROR"
    EXECUTION_ERROR = "EXECUTION_ERROR"


# =============================================================================
# DATA CLASSES (ML/AI Friendly)
# =============================================================================

@dataclass
class NodeConfig:
    """Node configuration from distributed_config.json"""
    hostname: str
    username: str
    gpu_count: int
    gpu_type: str
    script_path: str
    python_env: str
    max_concurrent: int = 12  # Max parallel jobs on this node
    
    @property
    def is_rocm(self) -> bool:
        return self.hostname in ROCM_HOSTNAMES
    
    @property
    def is_localhost(self) -> bool:
        return self.hostname in ['localhost', '127.0.0.1']
    
    @property
    def stagger_delay(self) -> float:
        return STAGGER_LOCALHOST if self.is_localhost else STAGGER_REMOTE


@dataclass
class Job:
    """Job specification from jobs file (SCRIPT JOB SPEC v1.0)"""
    job_id: str
    script: str
    args: List[str]
    expected_output: str
    chunk_file: str = ""
    seed_count: int = 0
    timeout: int = 7200


def clear_hip_cache_on_nodes(nodes: List[NodeConfig], run_id: str) -> None:
    """
    Clear HIP kernel cache once per node per run_id.
    
    Team Beta ruling (Jan 23, 2026):
    - B1: Once per node per run_id (NOT per batch/job)
    - B2: Non-fatal/idempotent (ignore failures)
    - B3: Configurable via CLEAR_HIP_CACHE env var
    - B4: Synchronous (wait before first wave)
    
    Fixes: "HIP error: invalid device function" from stale compiled kernels
    """
    if CLEAR_HIP_CACHE == '0':
        print("[PRE-FLIGHT] HIP cache clear disabled (CLEAR_HIP_CACHE=0)")
        return
    
    print(f"[PRE-FLIGHT] HIP cache clear for run_id={run_id}")
    
    # Select nodes to clear
    nodes_to_clear = []
    for node in nodes:
        # Skip localhost unless force mode
        if node.hostname == 'localhost' and CLEAR_HIP_CACHE != 'force':
            continue
        # ROCm detection: simple and stable (Team Beta required)
        is_rocm = 'rocm' in node.python_env.lower()
        if is_rocm or CLEAR_HIP_CACHE == 'force':
            nodes_to_clear.append(node)
    
    if not nodes_to_clear:
        print("[PRE-FLIGHT] No ROCm nodes to clear HIP cache")
        return
    
    node_names = [n.hostname for n in nodes_to_clear]
    print(f"[PRE-FLIGHT] Clearing HIP cache (once per run) on: {', '.join(node_names)}")
    
    cache_clear_cmd = "rm -rf ~/.cache/hip ~/.cache/amd_comgr ~/.cache/kernels 2>/dev/null || true"
    
    for node in nodes_to_clear:
        try:
            result = subprocess.run(
                ['ssh', '-o', 'ConnectTimeout=10', '-o', 'BatchMode=yes',
                 f'{node.username}@{node.hostname}', cache_clear_cmd],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                print(f"[PRE-FLIGHT] {node.hostname}: cleared (ok)")
            else:
                print(f"[PRE-FLIGHT] {node.hostname}: cleared (warning: exit {result.returncode})")
        except subprocess.TimeoutExpired:
            print(f"[PRE-FLIGHT] {node.hostname}: timeout (continuing anyway)")
        except Exception as e:
            print(f"[PRE-FLIGHT] {node.hostname}: error {e} (continuing anyway)")


def warmup_rocm_on_nodes(nodes: List[NodeConfig], run_id: str) -> None:
    """
    Warm ROCm GPUs sequentially once per node per run_id.

    Goal: mitigate + diagnose first-wave transient failures caused by parallel HIP init
    and/or concurrent kernel compilation.

    Guardrails:
    - W1: Once per node per run_id (called once in run() pre-flight)
    - W2: Non-fatal/idempotent (ignore failures, log warning)
    - W3: Configurable via WARMUP_ROCM env var
    - W4: Synchronous barrier before first dispatch
    """
    if WARMUP_ROCM == '0':
        print("[PRE-FLIGHT] ROCm warm-up disabled (WARMUP_ROCM=0)")
        return

    # Select ROCm nodes (same stable heuristic as HIP cache clear)
    nodes_to_warm = []
    for node in nodes:
        if node.hostname == 'localhost' and WARMUP_ROCM != 'force':
            continue
        is_rocm = 'rocm' in node.python_env.lower()
        if is_rocm or WARMUP_ROCM == 'force':
            nodes_to_warm.append(node)

    if not nodes_to_warm:
        print("[PRE-FLIGHT] No ROCm nodes selected for warm-up")
        return

    node_names = [n.hostname for n in nodes_to_warm]
    print(f"[PRE-FLIGHT] ROCm warm-up barrier for run_id={run_id}")
    print(f"[PRE-FLIGHT] Warming GPUs sequentially on: {', '.join(node_names)}")

    # Tiny matmul per GPU; sequential loop prevents init storm.
    # Keep workload small but non-trivial to trigger kernel compile + synchronize.
    remote_py = r"""
import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
import torch
n = torch.cuda.device_count()
print(f'WARMUP gpu_count={n}')
for i in range(n):
    torch.cuda.set_device(i)
    x = torch.randn(512, 512, device='cuda')
    y = x @ x
    torch.cuda.synchronize()
print('WARMUP_OK')
"""

    for node in nodes_to_warm:
        try:
            # Run under node.python_env (venv python path from distributed_config.json)
            cmd = (
                f"{node.python_env} - << 'PY'\n"
                f"{remote_py}\n"
                f"PY"
            )
            result = subprocess.run(
                ['ssh', '-o', 'ConnectTimeout=10', '-o', 'BatchMode=yes',
                 f'{node.username}@{node.hostname}', cmd],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                # Print a compact success line; keep stdout trimmed for logs
                print(f"[PRE-FLIGHT] {node.hostname}: warmup ok")
            else:
                print(f"[PRE-FLIGHT] {node.hostname}: warmup warning (exit {result.returncode})")
                if result.stderr:
                    print(f"[PRE-FLIGHT] {node.hostname}: stderr: {result.stderr.strip()[:300]}")
        except subprocess.TimeoutExpired:
            print(f"[PRE-FLIGHT] {node.hostname}: warmup timeout (continuing anyway)")
        except Exception as e:
            print(f"[PRE-FLIGHT] {node.hostname}: warmup error {e} (continuing anyway)")


@dataclass
class JobResult:
    """Execution result with explicit failure reason"""
    job_id: str
    success: bool
    node: str
    gpu_id: int
    runtime: float
    output_file: Optional[str] = None
    error: Optional[str] = None
    failure_mode: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# SCRIPTS COORDINATOR
# =============================================================================

class ScriptsCoordinator:
    """
    Universal ML Script Job Orchestrator.
    
    Conforms to established environment protocols.
    Uses file-based success detection (no stdout parsing).
    ML-agnostic: works with any script-based worker.
    """
    
    def __init__(self, config_file: str, jobs_file: str,
                 output_dir: str = 'full_scoring_results',
                 preserve_paths: bool = False,
                 max_retries: int = 3, verbose: bool = False,
                 dry_run: bool = False):
        self.config_file = config_file
        self.jobs_file = jobs_file
        self.output_base_dir = output_dir
        self.preserve_paths = preserve_paths
        self.max_retries = max_retries
        self.verbose = verbose
        self.dry_run = dry_run
        
        # Dynamic run ID based on output directory (v1.4.0)
        output_stem = Path(output_dir).stem
        self.run_id = f"{output_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = f"{output_dir}/{self.run_id}"
        
        self.nodes: List[NodeConfig] = []
        self.jobs: List[Job] = []
        self.results: List[JobResult] = []
        self.results_lock = threading.Lock()
        self.failed_jobs: List[Job] = []
        
        # Log warning for preserve_paths (Team Beta recommendation)
        if preserve_paths:
            logging.warning(
                "⚠️  --preserve-paths enabled: Coordinator will NOT rewrite output paths. "
                "Worker is responsible for path uniqueness."
            )
    
    def load_config(self) -> List[NodeConfig]:
        """Load cluster configuration from distributed_config.json"""
        with open(self.config_file) as f:
            config = json.load(f)
        
        self.nodes = []
        for node_data in config.get('nodes', []):
            node = NodeConfig(
                hostname=node_data['hostname'],
                username=node_data.get('username', 'michael'),
                gpu_count=node_data['gpu_count'],
                gpu_type=node_data.get('gpu_type', 'unknown'),
                script_path=node_data['script_path'],
                python_env=node_data['python_env'],
                max_concurrent=node_data.get('max_concurrent_script_jobs', node_data['gpu_count'])
            )
            self.nodes.append(node)
        
        return self.nodes
    
    def load_jobs(self) -> List[Job]:
        """Load jobs from jobs file (dynamic count)"""
        with open(self.jobs_file) as f:
            jobs_data = json.load(f)
        
        self.jobs = []
        for job_data in jobs_data:
            original_output = job_data['expected_output']
            
            if self.preserve_paths:
                # Step 5 mode: Use original paths as-is
                scoped_output = original_output
                updated_args = job_data.get('args', [])
            else:
                # Step 3 mode: Rewrite to run-scoped directory
                filename = Path(original_output).name
                scoped_output = f"{self.output_dir}/{filename}"
                updated_args = self._update_args_output_path(
                    job_data.get('args', []), 
                    scoped_output
                )
            
            job = Job(
                job_id=job_data['job_id'],
                script=job_data['script'],
                args=updated_args,
                expected_output=scoped_output,
                chunk_file=job_data.get('chunk_file', ''),
                seed_count=job_data.get('seed_count', 0),
                timeout=job_data.get('timeout', 7200)
            )
            self.jobs.append(job)
        
        return self.jobs
    
    def _update_args_output_path(self, args: List[str], new_output: str) -> List[str]:
        """Update --output-file argument to use run-scoped path"""
        updated_args = []
        skip_next = False
        found_output = False
        
        for i, arg in enumerate(args):
            if skip_next:
                skip_next = False
                continue
            
            if arg == '--output-file' and i + 1 < len(args):
                updated_args.append(arg)
                # Use absolute path for clarity
                script_path = "/home/michael/distributed_prng_analysis"
                updated_args.append(f"{script_path}/{new_output}")
                skip_next = True
                found_output = True
            else:
                updated_args.append(arg)
        
        return updated_args
    
    def _create_output_dirs(self):
        """Create run-scoped output directory on all nodes"""
        if self.preserve_paths:
            # For Step 5, create the base output directory only
            base_dir = Path(self.output_base_dir)
            print(f"  Creating base output directory: {base_dir}")
        else:
            print(f"  Creating output directories: {self.output_dir}")
        
        for node in self.nodes:
            if self.preserve_paths:
                path = f"{node.script_path}/{self.output_base_dir}"
            else:
                path = f"{node.script_path}/{self.output_dir}"
            
            if node.is_localhost:
                Path(path).mkdir(parents=True, exist_ok=True)
            else:
                try:
                    subprocess.run(
                        ['ssh', f'{node.username}@{node.hostname}', f'mkdir -p "{path}"'],
                        capture_output=True, timeout=10, check=True
                    )
                except Exception as e:
                    print(f"    WARNING: Failed to create dir on {node.hostname}: {e}")
    
    def assign_jobs_to_nodes(self) -> Dict[str, List[Job]]:
        """Distribute jobs proportionally by GPU count"""
        total_gpus = sum(n.gpu_count for n in self.nodes)
        assignments = {n.hostname: [] for n in self.nodes}
        
        # Round-robin assignment weighted by GPU capacity
        node_index = 0
        gpu_cursors = [0] * len(self.nodes)
        
        for job in self.jobs:
            # Find next node with available GPU slot
            assigned = False
            for _ in range(len(self.nodes) * 2):  # Prevent infinite loop
                node = self.nodes[node_index]
                assignments[node.hostname].append(job)
                gpu_cursors[node_index] = (gpu_cursors[node_index] + 1) % node.gpu_count
                node_index = (node_index + 1) % len(self.nodes)
                assigned = True
                break
            
            if not assigned:
                # Fallback: assign to first node
                assignments[self.nodes[0].hostname].append(job)
        
        return assignments
    
    def _reset_allocator_state(self):
        """
        Reset memory allocator on remote ROCm nodes.
        
        Mirrors benchmark_sample_sizes_v2.sh behavior:
        - sync filesystems
        - drop page cache (echo 3 > drop_caches)
        
        Note: localhost reset intentionally skipped (CUDA manages its own memory).
        """
        if not ENABLE_ALLOCATOR_RESET:
            return
            
        print("  Resetting allocator state on ROCm nodes...")
        for node in self.nodes:
            if not node.is_localhost:
                try:
                    subprocess.run(
                        ['ssh', f'{node.username}@{node.hostname}',
                         'sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true'],
                        capture_output=True, timeout=10
                    )
                except Exception as e:
                    print(f"    Warning: Failed to reset {node.hostname}: {e}")
        time.sleep(2)  # Brief pause for stability
        print("  Allocator reset complete")

    def run(self) -> Dict[str, Any]:
        """Execute all jobs and return results"""
        start_time = time.time()
        
        # Load configuration and jobs
        self.load_config()
        self.load_jobs()
        
        print("=" * 60)
        print(f"SCRIPTS COORDINATOR v1.4.0 - Universal ML Orchestrator")
        print("=" * 60)
        print(f"  Run ID: {self.run_id}")
        print(f"  Output: {self.output_dir if not self.preserve_paths else self.output_base_dir}")
        print(f"  Preserve Paths: {self.preserve_paths}")
        print(f"  Nodes: {len(self.nodes)}")
        print(f"  Total GPUs: {sum(n.gpu_count for n in self.nodes)}")
        print(f"  Jobs: {len(self.jobs)}")
        if self.dry_run:
            print(f"  Mode: DRY RUN (no execution)")
        print("=" * 60)
        
        # Create run-scoped output directories
        self._create_output_dirs()
        
        # Assign jobs to nodes
        assignments = self.assign_jobs_to_nodes()
        
        print("Job Distribution:")
        for hostname, jobs in assignments.items():
            node = next(n for n in self.nodes if n.hostname == hostname)
            print(f"  {hostname} ({node.gpu_count} GPUs): {len(jobs)} jobs")
        
        # Dry run - stop here
        if self.dry_run:
            print("-" * 60)
            print("DRY RUN - Job assignments:")
            for hostname, jobs in assignments.items():
                print(f"\n  {hostname}:")
                for j in jobs[:5]:  # Show first 5
                    print(f"    - {j.job_id}: {j.script}")
                if len(jobs) > 5:
                    print(f"    ... and {len(jobs) - 5} more")
            print("-" * 60)
            print("DRY RUN complete. No jobs executed.")
            return {
                "status": "dry_run",
                "run_id": self.run_id,
                "total_jobs": len(self.jobs),
                "assignments": {h: len(j) for h, j in assignments.items()}
            }
        
        print("-" * 60)
        # Clear HIP cache once per run (Team Beta: Jan 23, 2026)
        clear_hip_cache_on_nodes(self.nodes, self.run_id)
        warmup_rocm_on_nodes(self.nodes, self.run_id)

        
        print("Executing jobs...")
        print("-" * 60)
        
        # Check if batching needed
        total_jobs = len(self.jobs)
        if total_jobs > MAX_JOBS_PER_BATCH:
            # Step-aware batching (Team Beta 2026-01-22)
            detected_step = detect_job_step(self.jobs_file, self.jobs)
            effective_max_per_node, effective_cooldown, step_name = get_step_aware_limits(detected_step)
            
            # Batched execution
            num_batches = (total_jobs + MAX_JOBS_PER_BATCH - 1) // MAX_JOBS_PER_BATCH
            print(f"  [BATCH MODE] {total_jobs} jobs → {num_batches} batches of ≤{MAX_JOBS_PER_BATCH}")
            print(f"  [STEP-AWARE] Detected: {step_name}")
            print(f"  [STEP-AWARE] Limits: max_per_node={effective_max_per_node}, cooldown={effective_cooldown}s")
            
            # Build job queues per node
            node_queues = {}
            for node in self.nodes:
                node_queues[node.hostname] = list(assignments[node.hostname])
            
            # Form batches respecting per-node limits (using step-aware value)
            all_batches = []
            while any(node_queues.values()):
                batch = []
                for node in self.nodes:
                    queue = node_queues[node.hostname]
                    take = min(len(queue), effective_max_per_node)
                    for _ in range(take):
                        if len(batch) < MAX_JOBS_PER_BATCH:
                            batch.append((node, queue.pop(0)))
                if batch:
                    all_batches.append(batch)
            
            num_batches = len(all_batches)
            print(f"  [BATCH MODE] {total_jobs} jobs → {num_batches} batches (max {effective_max_per_node}/node)")
            
            # Process batches
            for batch_num, batch_jobs in enumerate(all_batches):
                
                print(f"\n  {'='*50}")
                print(f"  [BATCH {batch_num + 1}/{num_batches}] {len(batch_jobs)} jobs")
                print(f"  {'='*50}")
                
                # Reset allocator before batch (except first)
                if batch_num > 0:
                    self._reset_allocator_state()
                    print(f"  Cooling down {effective_cooldown}s...")
                    time.sleep(effective_cooldown)
                
                # Group batch jobs by node
                batch_by_node = {}
                for node, job in batch_jobs:
                    if node.hostname not in batch_by_node:
                        batch_by_node[node.hostname] = []
                    batch_by_node[node.hostname].append(job)
                
                # Launch node executor threads for this batch
                threads = []
                for node in self.nodes:
                    node_jobs = batch_by_node.get(node.hostname, [])
                    if node_jobs:
                        t = threading.Thread(
                            target=self._node_executor,
                            args=(node, node_jobs),
                            name=f"executor-{node.hostname}-batch{batch_num}"
                        )
                        threads.append(t)
                        t.start()
                
                # Wait for batch to complete
                for t in threads:
                    t.join()
                
                # Batch summary
                batch_successful = sum(1 for r in self.results[-len(batch_jobs):] if r.success)
                print(f"  [BATCH {batch_num + 1}] Complete: {batch_successful}/{len(batch_jobs)} successful")
        else:
            # Original non-batched execution for small job counts
            threads = []
            for node in self.nodes:
                node_jobs = assignments[node.hostname]
                if node_jobs:
                    t = threading.Thread(
                        target=self._node_executor,
                        args=(node, node_jobs),
                        name=f"executor-{node.hostname}"
                    )
                    threads.append(t)
                    t.start()
            
            # Wait for all threads
            for t in threads:
                t.join()
        
        # Retry failures on localhost
        if self.failed_jobs and self.max_retries > 0:
            self._retry_on_localhost()
        
        # Compile results
        runtime = time.time() - start_time
        successful = sum(1 for r in self.results if r.success)
        failed = len(self.results) - successful
        
        # Build results dict
        results = {
            "status": "complete" if failed == 0 else "partial",
            "run_id": self.run_id,
            "total_jobs": len(self.jobs),
            "successful": successful,
            "failed": failed,
            "runtime_seconds": round(runtime, 1),
            "timestamp": datetime.now().isoformat(),
            "output_dir": self.output_dir if not self.preserve_paths else self.output_base_dir,
            "preserve_paths": self.preserve_paths,
            "nodes": {
                n.hostname: {
                    "jobs": len(assignments.get(n.hostname, [])),
                    "successful": sum(1 for r in self.results if r.node == n.hostname and r.success),
                    "failed": sum(1 for r in self.results if r.node == n.hostname and not r.success),
                    "gpu_count": n.gpu_count,
                    "gpu_type": n.gpu_type
                }
                for n in self.nodes
            },
            "jobs": [asdict(r) for r in self.results]
        }
        
        # Write manifest (Team Beta Recommendation 2)
        if not self.preserve_paths:
            self._write_manifest(results)
        
        print("=" * 60)
        print("EXECUTION COMPLETE")
        print("=" * 60)
        print(f"  Run ID: {self.run_id}")
        print(f"  Total jobs: {len(self.jobs)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Runtime: {runtime:.1f}s")
        print(f"  Output: {self.output_dir if not self.preserve_paths else self.output_base_dir}")
        print("=" * 60)
        
        return results
    
    def _write_manifest(self, results: Dict[str, Any]):
        """Write run manifest for auditability (Team Beta Recommendation 2)"""
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "script_job_spec_version": SCRIPT_JOB_SPEC_VERSION,
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "config_file": self.config_file,
            "jobs_file": self.jobs_file,
            "jobs_expected": len(self.jobs),
            "jobs_completed": results['successful'],
            "jobs_failed": results['failed'],
            "runtime_seconds": results['runtime_seconds'],
            "output_dir": self.output_dir,
            "outputs": [
                r['output_file'] for r in results['jobs'] if r['success']
            ],
            "failures": [
                {
                    "job_id": r['job_id'],
                    "failure_mode": r.get('failure_mode'),
                    "error": r.get('error')
                }
                for r in results['jobs'] if not r['success']
            ],
            "nodes": results['nodes']
        }
        
        # Write to run-scoped directory
        manifest_path = Path(self.nodes[0].script_path) / self.output_dir / "scripts_run_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"  Manifest: {manifest_path}")
    
    def _node_executor(self, node: NodeConfig, jobs: List[Job]):
        """Execute jobs in PARALLEL on a node (one worker per GPU)"""
        
        # Pre-assign jobs to GPUs (round-robin)
        num_active_gpus = min(node.gpu_count, node.max_concurrent)
        gpu_jobs = {i: [] for i in range(num_active_gpus)}
        for i, job in enumerate(jobs):
            gpu_jobs[i % num_active_gpus].append(job)
        
        def gpu_worker(gpu_id: int, job_list: List[Job]):
            """Execute jobs sequentially on one GPU"""
            for i, job in enumerate(job_list):
                # Stagger delay between jobs on same GPU
                if i > 0:
                    time.sleep(node.stagger_delay)
                
                # Execute job
                result = self._execute_job(node, job, gpu_id)
                
                with self.results_lock:
                    self.results.append(result)
                    if not result.success:
                        self.failed_jobs.append(job)
                
                # Progress output
                status = "✓" if result.success else "✗"
                print(f"  {status} {job.job_id} → {node.hostname}:GPU{gpu_id} ({result.runtime:.1f}s)")
                
                if not result.success and self.verbose:
                    print(f"      FAIL: {result.failure_mode}: {result.error}")
        
        # Run GPUs in parallel, limited by max_concurrent
        active_gpus = [gid for gid, jlist in gpu_jobs.items() if jlist][:node.max_concurrent]
        max_workers = min(len(active_gpus), node.max_concurrent)
        
        # Debug: Show parallel execution mode
        jobs_per_gpu = {gid: len(gpu_jobs[gid]) for gid in active_gpus}
        print(f"  🔀 PARALLEL: {node.hostname} | {max_workers} GPU workers | {len(jobs)} jobs | distribution: {jobs_per_gpu}")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, gpu_id in enumerate(active_gpus):
                futures.append(executor.submit(gpu_worker, gpu_id, gpu_jobs[gpu_id]))
                # Stagger GPU worker startup to prevent HIP init collision
                # Uses existing node.stagger_delay (STAGGER_REMOTE=2.0s)
                if i < len(active_gpus) - 1:
                    time.sleep(node.stagger_delay)

            for future in as_completed(futures):
                try:
                    future.result()  # Raises if worker had exception
                except Exception as e:
                    print(f"  ✗ Worker exception on {node.hostname}: {e}")
    
    def _execute_job(self, node: NodeConfig, job: Job, gpu_id: int) -> JobResult:
        """Execute a single job and verify output file exists"""
        start_time = time.time()
        
        try:
            cmd = self._build_command(node, job, gpu_id)
            
            if node.is_localhost:
                # Local execution
                result = subprocess.run(
                    cmd, shell=True,
                    capture_output=True, text=True,
                    timeout=job.timeout,
                    cwd=node.script_path
                )
            else:
                # Remote execution via SSH
                result = subprocess.run(
                    ['ssh', f'{node.username}@{node.hostname}', cmd],
                    capture_output=True, text=True,
                    timeout=job.timeout
                )
            
            runtime = time.time() - start_time
            
            # DEBUG: Print SSH result for remote jobs
            if not node.is_localhost and result.returncode != 0:
                print(f"  [DEBUG] {job.job_id} SSH returncode: {result.returncode}")
                print(f"  [DEBUG] STDOUT: {result.stdout[:500]}")
                print(f"  [DEBUG] STDERR: {result.stderr[:500]}")
            
            # DEBUG: Also capture stderr when file check fails
            job_stderr = result.stderr if result else ""
            job_stdout = result.stdout if result else ""

            # File-based success detection (Team Beta Recommendation 3)
            success, error, failure_mode = self._check_output(node, job.expected_output)
            
            if success:
                return JobResult(
                    job_id=job.job_id,
                    success=True,
                    node=node.hostname,
                    gpu_id=gpu_id,
                    runtime=runtime,
                    output_file=job.expected_output
                )
            else:
                # DEBUG: Print stderr on job failure (Team Beta diagnostic)
                if job_stderr:
                    print(f"  [DEBUG] {job.job_id} failed - STDERR: {job_stderr[:500]}")
                if job_stdout:
                    print(f"  [DEBUG] {job.job_id} failed - STDOUT: {job_stdout[:500]}")
                return JobResult(
                    job_id=job.job_id,
                    success=False,
                    node=node.hostname,
                    gpu_id=gpu_id,
                    runtime=runtime,
                    error=error + (f" | stderr: {job_stderr[:200]}" if job_stderr else ""),
                    failure_mode=failure_mode
                )
                
        except subprocess.TimeoutExpired:
            return JobResult(
                job_id=job.job_id,
                success=False,
                node=node.hostname,
                gpu_id=gpu_id,
                runtime=job.timeout,
                error=f"Timeout after {job.timeout}s",
                failure_mode=FailureMode.TIMEOUT
            )
        except Exception as e:
            return JobResult(
                job_id=job.job_id,
                success=False,
                node=node.hostname,
                gpu_id=gpu_id,
                runtime=time.time() - start_time,
                error=str(e),
                failure_mode=FailureMode.EXECUTION_ERROR
            )
    
    def _build_command(self, node: NodeConfig, job: Job, gpu_id: int) -> str:
        """Build command with proper environment (ROCm/CUDA)"""
        
        # GPU environment variables
        if node.is_rocm:
            env_vars = ROCM_ENV_VARS + [
                f"CUDA_VISIBLE_DEVICES={gpu_id}",
                f"HIP_VISIBLE_DEVICES={gpu_id}"
            ]
        else:
            env_vars = [f"CUDA_VISIBLE_DEVICES={gpu_id}"]
        
        env_prefix = "env " + " ".join(env_vars) + " "
        
        # Job payload for distributed_worker.py
        job_payload = {
            'job_id': job.job_id,
            'script': job.script,
            'args': job.args,
            'expected_output': job.expected_output,
            'timeout': job.timeout,
            'analysis_type': 'script'
        }
        job_json = json.dumps(job_payload)
        job_filename = f"job_{job.job_id}.json"
        
        # Build command: write job JSON, execute, cleanup
        cmd = (
            f"cd '{node.script_path}' && "
            f"cat > {job_filename} <<'JSON'\n{job_json}\nJSON\n"
            f"{env_prefix}{node.python_env} -u distributed_worker.py "
            f"{job_filename} --gpu-id {gpu_id} ; "
            f"rm -f {job_filename} || true"
        )
        
        return cmd
    
    def _check_output(self, node: NodeConfig, filepath: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Strict file-based success detection (Team Beta Recommendation 3).
        
        Returns (success, error_message, failure_mode)
        
        Failure modes (explicit, no fallback):
        - File missing = FAIL
        - File empty (size=0) = FAIL
        - File not valid JSON = FAIL
        """
        # Handle absolute vs relative paths
        if filepath.startswith('/'):
            full_path = filepath
        else:
            full_path = f"{node.script_path}/{filepath}"
        
        if node.is_localhost:
            p = Path(full_path)
            
            # Check 1: File exists
            if not p.exists():
                return False, f"File missing: {filepath}", FailureMode.MISSING
            
            # Check 2: File not empty
            if p.stat().st_size == 0:
                return False, f"File empty (0 bytes): {filepath}", FailureMode.EMPTY
            
            # Check 3: Valid JSON
            try:
                with open(p) as f:
                    data = json.load(f)
                if not isinstance(data, (list, dict)):
                    return False, f"Invalid JSON structure (expected list or dict): {filepath}", FailureMode.INVALID_JSON
                if isinstance(data, list) and len(data) == 0:
                    return False, f"Empty JSON array: {filepath}", FailureMode.INVALID_JSON
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON: {e}", FailureMode.INVALID_JSON
            
            return True, None, None
        
        else:
            # Remote validation
            try:
                # Check file exists and has size > 0
                result = subprocess.run(
                    ['ssh', f'{node.username}@{node.hostname}',
                     f'test -s "{full_path}" && head -c 1 "{full_path}"'],
                    capture_output=True, text=True, timeout=10
                )
                
                # Check 1 & 2: File exists and not empty (test -s)
                if result.returncode != 0:
                    return False, f"File missing or empty: {filepath}", FailureMode.MISSING
                
                # Check 3: Starts with valid JSON marker (object or array)
                first_char = result.stdout.strip()[:1]
                if first_char not in ['{', '[']:
                    return False, f"Invalid JSON (doesn't start with '{{' or '['): {filepath}", FailureMode.INVALID_JSON
                
                return True, None, None
                
            except subprocess.TimeoutExpired:
                return False, f"Timeout checking file: {filepath}", FailureMode.TIMEOUT
            except Exception as e:
                return False, f"SSH error checking file: {e}", FailureMode.SSH_ERROR
    
    def _retry_on_localhost(self):
        """Retry failed jobs on localhost (guaranteed to work)"""
        if not self.failed_jobs:
            return
        
        print("")
        print(f"⚠️  Retrying {len(self.failed_jobs)} failed jobs on localhost...")
        print("-" * 60)
        
        localhost = next((n for n in self.nodes if n.is_localhost), None)
        if not localhost:
            print("  ERROR: No localhost node configured")
            return
        
        # Copy of failed jobs to retry
        jobs_to_retry = list(self.failed_jobs)
        self.failed_jobs.clear()
        
        gpu_id = 0
        for job in jobs_to_retry:
            # Apply stagger delay
            if gpu_id > 0:
                time.sleep(STAGGER_LOCALHOST)
            
            result = self._execute_job(localhost, job, gpu_id)
            
            with self.results_lock:
                # Remove original failed result
                self.results = [r for r in self.results if r.job_id != job.job_id]
                # Add retry result
                self.results.append(result)
                
                if not result.success:
                    self.failed_jobs.append(job)
            
            status = "✓" if result.success else "✗"
            print(f"  {status} RETRY {job.job_id} → localhost:GPU{gpu_id} ({result.runtime:.1f}s)")
            
            if not result.success and self.verbose:
                print(f"      FAIL: {result.failure_mode}: {result.error}")
            
            gpu_id = (gpu_id + 1) % localhost.gpu_count


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Universal ML Script Job Orchestrator (Steps 3, 5)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
SCRIPT JOB SPEC v1.0:
  {
    "job_id": "string",
    "script": "worker_script.py",
    "args": ["--arg1", "value1", ...],
    "expected_output": "path/to/output.json",
    "timeout": 7200
  }

Examples:
  # Step 3 - Full Scoring (default)
  python3 scripts_coordinator.py --jobs-file scoring_jobs.json

  # Step 5 - Anti-Overfit Trials
  python3 scripts_coordinator.py --jobs-file anti_overfit_jobs.json \\
      --output-dir anti_overfit_results --preserve-paths

  # Dry run - see job distribution without execution
  python3 scripts_coordinator.py --jobs-file scoring_jobs.json --dry-run
        """
    )
    parser.add_argument('--jobs-file', required=True,
                        help='Job specifications JSON (SCRIPT JOB SPEC v1.0)')
    parser.add_argument('--config', default='distributed_config.json',
                        help='Cluster configuration file (default: distributed_config.json)')
    parser.add_argument('--output-dir', default='full_scoring_results',
                        help='Base output directory (default: full_scoring_results)')
    parser.add_argument('--preserve-paths', action='store_true',
                        help='Do not rewrite output paths in job args (for Step 5)')
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Max retry attempts for failed jobs (default: 3)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print job distribution without execution')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output (show failure details)')
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not Path(args.jobs_file).exists():
        print(f"ERROR: Jobs file not found: {args.jobs_file}")
        return 1
    
    if not Path(args.config).exists():
        print(f"ERROR: Config file not found: {args.config}")
        return 1
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    coordinator = ScriptsCoordinator(
        config_file=args.config,
        jobs_file=args.jobs_file,
        output_dir=args.output_dir,
        preserve_paths=args.preserve_paths,
        max_retries=args.max_retries,
        verbose=args.verbose,
        dry_run=args.dry_run
    )
    
    results = coordinator.run()
    
    # Output results JSON for pipeline integration
    if not args.dry_run:
        print("")
        print("Results JSON:")
        print(json.dumps(results, indent=2))
    
    return 0 if results.get('failed', 0) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
