#!/usr/bin/env python3
"""
Chunk Size Configuration for Memory-Constrained Nodes
======================================================
Version: 1.0.0
Date: 2026-01-24

PURPOSE:
  Provides chunk size configuration to prevent OOM kills on mining rigs.
  Used by generate_step3_scoring_jobs.py to create appropriately-sized jobs.

INTEGRATION:
  Add to generate_step3_scoring_jobs.py:
  
  ```python
  from chunk_size_config import get_chunk_size_for_node, calculate_optimal_chunks
  
  # Get node-specific chunk size
  chunk_size = get_chunk_size_for_node(hostname)
  
  # Or calculate based on memory benchmark
  chunks = calculate_optimal_chunks(
      total_survivors=98172,
      chunk_size=chunk_size
  )
  ```

CONFIGURATION:
  1. Run benchmark on each rig:
     python3 benchmark_worker_memory.py --survivors bidirectional_survivors.json
  
  2. Update MEMORY_PROFILES below with results
  
  3. Jobs will automatically use safe chunk sizes
"""

import os
import json
import socket
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# ============================================================================
# MEMORY PROFILES BY NODE
# ============================================================================
# Update these values after running benchmark_worker_memory.py on each node
#
# TARGET: 12 GPUs active on each mining rig (full utilization)
#
# MATH FOR 12-GPU OPERATION:
#   Available RAM: 7,700 MB × 0.80 safety = 6,160 MB usable
#   RAM per worker: 6,160 ÷ 12 = 513 MB max
#   
#   RAM_per_worker = base_overhead + (chunk_size ÷ 1000) × mb_per_1k
#   513 = 200 + (chunk_size ÷ 1000) × 150
#   chunk_size = ((513 - 200) / 150) × 1000 = 2,086
#   
#   Round down for safety: chunk_size = 2,000
#
# After running benchmark, actual values may differ. Update accordingly.

MEMORY_PROFILES = {
    # Zeus (localhost) - 64GB RAM, no constraints
    "zeus": {
        "total_ram_mb": 64000,
        "available_ram_mb": 50000,
        "max_concurrent_workers": 2,  # Only 2 GPUs
        "chunk_size": None,  # No limit needed
        "mb_per_1k_survivors": 150,
        "base_overhead_mb": 200
    },
    
    # rig-6600 - 7.7GB RAM, TARGET: ALL 12 GPUs
    "rig-6600": {
        "total_ram_mb": 7700,
        "available_ram_mb": 6160,  # 80% of 7700 (safety margin)
        "max_concurrent_workers": 12,  # ALL 12 GPUs - FULL UTILIZATION
        "chunk_size": 2000,  # Sized for 12 workers @ ~500MB each
        "mb_per_1k_survivors": 150,  # UPDATE AFTER BENCHMARK
        "base_overhead_mb": 200  # UPDATE AFTER BENCHMARK
    },
    
    # rig-6600b - 7.7GB RAM, TARGET: ALL 12 GPUs
    "rig-6600b": {
        "total_ram_mb": 7700,
        "available_ram_mb": 6160,  # 80% of 7700 (safety margin)
        "max_concurrent_workers": 12,  # ALL 12 GPUs - FULL UTILIZATION
        "chunk_size": 2000,  # Sized for 12 workers @ ~500MB each
        "mb_per_1k_survivors": 150,  # UPDATE AFTER BENCHMARK
        "base_overhead_mb": 200  # UPDATE AFTER BENCHMARK
    }
}

# Hostname to profile mapping (for IP addresses)
HOSTNAME_MAP = {
    "localhost": "zeus",
    "127.0.0.1": "zeus",
    "192.168.3.120": "rig-6600",
    "192.168.3.154": "rig-6600b",
    "rig-6600": "rig-6600",
    "rig-6600b": "rig-6600b",
    "192.168.3.162": "rig-6600c",
    "rig-6600c": "rig-6600c",
    "zeus": "zeus"
}


@dataclass
class ChunkConfig:
    """Configuration for chunked job generation."""
    chunk_size: int
    total_chunks: int
    survivors_per_node: Dict[str, int]
    estimated_ram_per_worker_mb: float


def get_profile_for_host(hostname: str) -> Dict:
    """Get memory profile for a hostname or IP."""
    profile_name = HOSTNAME_MAP.get(hostname, "rig-6600")  # Default to constrained
    return MEMORY_PROFILES.get(profile_name, MEMORY_PROFILES["rig-6600"])


def calculate_chunk_size_for_12_gpus(
    available_ram_mb: float = 6160,
    mb_per_1k_survivors: float = 150,
    base_overhead_mb: float = 200,
    safety_factor: float = 1.0  # Already applied to available_ram
) -> int:
    """
    Calculate chunk size specifically for 12-GPU operation.
    
    This is the key function for maximizing GPU utilization on mining rigs.
    
    Args:
        available_ram_mb: Usable RAM (already with safety margin applied)
        mb_per_1k_survivors: Memory per 1000 survivors (from benchmark)
        base_overhead_mb: Fixed overhead per worker (from benchmark)
        safety_factor: Additional safety margin (default 1.0 = none)
    
    Returns:
        Maximum safe chunk size for 12 concurrent workers
    """
    target_workers = 12
    
    ram_per_worker = (available_ram_mb * safety_factor) / target_workers
    
    if ram_per_worker <= base_overhead_mb:
        print(f"⚠️  Warning: RAM per worker ({ram_per_worker:.0f} MB) <= base overhead ({base_overhead_mb:.0f} MB)")
        return 500  # Minimum viable
    
    # Solve: ram_per_worker = base_overhead + (chunk_size / 1000) * mb_per_1k
    chunk_size = int(((ram_per_worker - base_overhead_mb) / mb_per_1k_survivors) * 1000)
    
    # Round down to nearest 500 for cleaner job boundaries
    chunk_size = (chunk_size // 500) * 500
    
    # Enforce minimum
    chunk_size = max(500, chunk_size)
    
    return chunk_size


def get_chunk_size_for_node(hostname: str) -> Optional[int]:
    """
    Get the recommended chunk size for a specific node.
    
    Returns None if no limit needed (e.g., Zeus with lots of RAM).
    """
    profile = get_profile_for_host(hostname)
    return profile.get("chunk_size")


def calculate_safe_chunk_size(
    available_ram_mb: float,
    num_workers: int,
    mb_per_1k_survivors: float = 150,
    base_overhead_mb: float = 200,
    safety_factor: float = 0.80
) -> int:
    """
    Calculate safe chunk size based on available RAM.
    
    Formula:
      safe_ram = available_ram * safety_factor
      ram_per_worker = safe_ram / num_workers
      chunk_size = (ram_per_worker - base_overhead) / (mb_per_1k / 1000)
    """
    safe_ram = available_ram_mb * safety_factor
    ram_per_worker = safe_ram / num_workers
    
    # Solve for chunk size
    if ram_per_worker <= base_overhead_mb:
        return 100  # Minimum viable chunk
    
    chunk_size = int(((ram_per_worker - base_overhead_mb) / mb_per_1k_survivors) * 1000)
    
    # Enforce bounds
    chunk_size = max(100, min(chunk_size, 50000))
    
    return chunk_size


def calculate_optimal_chunks(
    total_survivors: int,
    chunk_size: Optional[int] = None,
    min_chunk_size: int = 500,
    max_chunk_size: int = 10000
) -> List[Tuple[int, int]]:
    """
    Calculate optimal chunk boundaries for job generation.
    
    Args:
        total_survivors: Total number of survivors to process
        chunk_size: Fixed chunk size (if None, uses max_chunk_size)
        min_chunk_size: Minimum survivors per chunk
        max_chunk_size: Maximum survivors per chunk (memory safety)
    
    Returns:
        List of (start_idx, end_idx) tuples
    """
    if chunk_size is None:
        chunk_size = max_chunk_size
    
    # Enforce bounds
    chunk_size = max(min_chunk_size, min(chunk_size, max_chunk_size))
    
    chunks = []
    start = 0
    
    while start < total_survivors:
        end = min(start + chunk_size, total_survivors)
        chunks.append((start, end))
        start = end
    
    return chunks


def get_cluster_chunk_config(
    total_survivors: int,
    nodes: List[str] = None
) -> ChunkConfig:
    """
    Get chunk configuration for the entire cluster.
    
    Uses the most constrained node's chunk size to ensure all jobs can run anywhere.
    """
    if nodes is None:
        nodes = ["zeus", "rig-6600", "rig-6600b", "rig-6600c"]
    
    # Find minimum chunk size across all nodes
    min_chunk_size = None
    for node in nodes:
        profile = get_profile_for_host(node)
        node_chunk = profile.get("chunk_size")
        if node_chunk is not None:
            if min_chunk_size is None or node_chunk < min_chunk_size:
                min_chunk_size = node_chunk
    
    # Use conservative default if no limits found
    if min_chunk_size is None:
        min_chunk_size = 5000
    
    # Calculate chunks
    chunks = calculate_optimal_chunks(total_survivors, min_chunk_size)
    
    # Estimate RAM usage
    profile = MEMORY_PROFILES["rig-6600"]  # Most constrained
    ram_per_worker = (
        profile["base_overhead_mb"] + 
        (min_chunk_size / 1000) * profile["mb_per_1k_survivors"]
    )
    
    return ChunkConfig(
        chunk_size=min_chunk_size,
        total_chunks=len(chunks),
        survivors_per_node={},  # Distribution happens at runtime
    # rig-6600c - 8GB RAM, 8 GPUs (same specs as rig-6600b)
    "rig-6600c": {
        "chunk_size": 5000,
        "max_concurrent": 8,
        "ram_gb": 8,
        "gpu_count": 8
    },
        estimated_ram_per_worker_mb=ram_per_worker
    )


def update_profile_from_benchmark(hostname: str, benchmark_file: str):
    """
    Update memory profile from benchmark results.
    
    Call this after running benchmark_worker_memory.py on a node.
    Automatically recalculates chunk_size for 12-GPU operation.
    """
    profile_name = HOSTNAME_MAP.get(hostname, hostname)
    
    with open(benchmark_file, 'r') as f:
        results = json.load(f)
    
    if profile_name not in MEMORY_PROFILES:
        print(f"❌ Unknown profile: {profile_name}")
        return
    
    profile = MEMORY_PROFILES[profile_name]
    
    # Update from benchmark
    old_chunk = profile.get("chunk_size", "N/A")
    
    profile["total_ram_mb"] = results["system"]["total_ram_mb"]
    # Apply 80% safety margin to available RAM
    profile["available_ram_mb"] = results["system"]["available_ram_mb"] * 0.80
    profile["mb_per_1k_survivors"] = results["memory_model"]["mb_per_1k_survivors"]
    profile["base_overhead_mb"] = results["memory_model"]["base_overhead_mb"]
    
    # Recalculate chunk size for 12-GPU operation
    new_chunk = calculate_chunk_size_for_12_gpus(
        available_ram_mb=profile["available_ram_mb"],
        mb_per_1k_survivors=profile["mb_per_1k_survivors"],
        base_overhead_mb=profile["base_overhead_mb"]
    )
    profile["chunk_size"] = new_chunk
    
    print(f"\n{'=' * 60}")
    print(f"✅ UPDATED PROFILE: {profile_name} (12-GPU TARGET)")
    print(f"{'=' * 60}")
    print(f"\n   From benchmark: {benchmark_file}")
    print(f"\n   Memory Model (measured):")
    print(f"   ├─ mb_per_1k_survivors: {profile['mb_per_1k_survivors']:.1f} MB")
    print(f"   └─ base_overhead_mb:    {profile['base_overhead_mb']:.1f} MB")
    print(f"\n   Chunk Size (for 12 GPUs):")
    print(f"   ├─ Old: {old_chunk}")
    print(f"   └─ New: {new_chunk:,}")
    
    # Verify the math
    ram_per_worker = (
        profile["base_overhead_mb"] + 
        (new_chunk / 1000) * profile["mb_per_1k_survivors"]
    )
    total_ram = ram_per_worker * 12
    
    print(f"\n   Verification:")
    print(f"   ├─ RAM per worker: {ram_per_worker:.0f} MB")
    print(f"   ├─ Total (12 GPU): {total_ram:.0f} MB")
    print(f"   └─ Available:      {profile['available_ram_mb']:.0f} MB")
    
    if total_ram <= profile['available_ram_mb']:
        print(f"\n   ✅ SAFE: {total_ram:.0f} MB < {profile['available_ram_mb']:.0f} MB")
    else:
        print(f"\n   ⚠️  WARNING: May exceed available RAM!")
    
    # Show how to apply
    print(f"\n{'─' * 60}")
    print(f"📝 TO APPLY THIS PERMANENTLY:")
    print(f"   Edit chunk_size_config.py and update {profile_name} profile:")
    print(f'   "chunk_size": {new_chunk},')
    print(f'   "mb_per_1k_survivors": {profile["mb_per_1k_survivors"]:.1f},')
    print(f'   "base_overhead_mb": {profile["base_overhead_mb"]:.1f}')


def print_cluster_status():
    """Print current memory configuration for all nodes."""
    print("=" * 70)
    print("CLUSTER MEMORY CONFIGURATION - 12-GPU TARGET")
    print("=" * 70)
    
    total_gpus = 0
    
    for name, profile in MEMORY_PROFILES.items():
        chunk = profile.get("chunk_size", "unlimited")
        if isinstance(chunk, int):
            chunk_str = f"{chunk:,}"
        else:
            chunk_str = "unlimited"
        
        workers = profile['max_concurrent_workers']
        total_gpus += workers
        
        print(f"\n{'─' * 50}")
        print(f"📊 {name.upper()}")
        print(f"{'─' * 50}")
        print(f"   RAM:        {profile['total_ram_mb']:,.0f} MB total")
        print(f"   Usable:     {profile['available_ram_mb']:,.0f} MB (80% safety)")
        print(f"   Workers:    {workers} GPUs {'← FULL UTILIZATION' if workers == 12 else ''}")
        print(f"   Chunk size: {chunk_str} survivors/job")
        
        if profile.get("chunk_size"):
            # Calculate expected RAM usage
            ram = (
                profile["base_overhead_mb"] + 
                (profile["chunk_size"] / 1000) * profile["mb_per_1k_survivors"]
            )
            total_ram = ram * workers
            utilization = (total_ram / profile['available_ram_mb']) * 100
            
            print(f"\n   Memory Estimates:")
            print(f"   ├─ Per worker:  {ram:.0f} MB")
            print(f"   ├─ Total used:  {total_ram:.0f} MB")
            print(f"   └─ Utilization: {utilization:.0f}% of available")
            
            if utilization > 95:
                print(f"   ⚠️  HIGH UTILIZATION - consider smaller chunk_size")
            elif utilization < 70:
                print(f"   💡 Could increase chunk_size for fewer jobs")
    
    print(f"\n{'=' * 70}")
    print(f"CLUSTER TOTAL: {total_gpus} GPUs")
    print(f"=" * 70)
    
    # Show what a typical Step 3 run looks like
    typical_survivors = 98172
    min_chunk = min(
        p.get("chunk_size", 99999) 
        for p in MEMORY_PROFILES.values() 
        if p.get("chunk_size")
    )
    num_jobs = (typical_survivors + min_chunk - 1) // min_chunk
    
    print(f"\n📋 STEP 3 ESTIMATE ({typical_survivors:,} survivors):")
    print(f"   Chunk size: {min_chunk:,}")
    print(f"   Total jobs: {num_jobs}")
    print(f"   Job waves:  ~{(num_jobs + total_gpus - 1) // total_gpus}")


# ============================================================================
# EXAMPLE: Patched generate_step3_scoring_jobs.py
# ============================================================================
"""
INTEGRATION EXAMPLE:

In generate_step3_scoring_jobs.py, replace fixed chunk logic with:

```python
from chunk_size_config import get_cluster_chunk_config, calculate_optimal_chunks

def generate_jobs(survivors_file, output_dir):
    # Load survivors
    with open(survivors_file) as f:
        survivors = json.load(f)
    
    total = len(survivors)
    
    # Get memory-safe chunk configuration
    config = get_cluster_chunk_config(total)
    
    print(f"Using chunk_size={config.chunk_size:,} ({config.total_chunks} jobs)")
    print(f"Estimated RAM per worker: {config.estimated_ram_per_worker_mb:.0f} MB")
    
    # Generate chunks
    chunks = calculate_optimal_chunks(total, config.chunk_size)
    
    jobs = []
    for i, (start, end) in enumerate(chunks):
        jobs.append({
            "job_id": f"step3_chunk_{i:04d}",
            "script": "full_scoring_worker.py",
            "args": [
                "--survivors", survivors_file,
                "--start-idx", str(start),
                "--end-idx", str(end),
                "--output", f"{output_dir}/chunk_{i:04d}.json"
            ]
        })
    
    return jobs
```
"""


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        print_cluster_status()
    elif len(sys.argv) > 2 and sys.argv[1] == "--update":
        # Usage: python3 chunk_size_config.py --update hostname benchmark_results.json
        hostname = sys.argv[2]
        benchmark_file = sys.argv[3] if len(sys.argv) > 3 else "memory_benchmark_results.json"
        update_profile_from_benchmark(hostname, benchmark_file)
    else:
        print("Usage:")
        print("  python3 chunk_size_config.py --status           # Show cluster config")
        print("  python3 chunk_size_config.py --update HOSTNAME [BENCHMARK_FILE]")
        print("")
        print("Example workflow:")
        print("  1. On rig-6600:  python3 benchmark_worker_memory.py -s survivors.json")
        print("  2. Copy results: scp rig-6600:memory_benchmark_results.json .")
        print("  3. Update:       python3 chunk_size_config.py --update rig-6600")
        print("")
        print_cluster_status()
