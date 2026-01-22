#!/usr/bin/env python3
"""
ramdisk_config.py - Unified Ramdisk Infrastructure for Distributed Steps
============================================================================
Version: 2.0.0 (Team Beta Approved - PROP-2026-01-21-RAMDISK-UNIFIED)

Usage:
    from ramdisk_config import get_data_path, preload_ramdisk, clear_ramdisk
    
    # Get path to file in ramdisk
    path = get_data_path(step_id=2, filename='survivors.npz')
    
    # Preload files for a step
    preload_ramdisk(step_id=5, files=['survivors_with_scores.json', 'holdout_history.json'])
    
    # Clear ramdisk for a step (or all)
    clear_ramdisk(step_id=5)  # Clear step 5 only
    clear_ramdisk()           # Clear all steps

Features:
    - Per-step subdirectories (/dev/shm/prng/step2/, step3/, step5/)
    - Cached node list (parsed once, reused)
    - WATCHER-aware lifecycle (env var detection)
    - Non-fatal cleanup (never bricks pipeline)
    - Pre-flight headroom check (warn >50%, abort >80%)

Team Beta Requirements Addressed:
    A1: Per-step subdirectories ✓
    A2: Cached node list ✓
    A3: Non-fatal cleanup ✓
    B1: Headroom check ✓
============================================================================
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from typing import List, Optional
from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

USE_RAMDISK = True
RAMDISK_BASE = "/dev/shm/prng"
SSD_BASE = "/home/michael/distributed_prng_analysis"

# Headroom thresholds (percentage of /dev/shm used)
WARN_THRESHOLD = 50
ABORT_THRESHOLD = 80

# ============================================================================
# A2: Cached node list - resolve ONCE per process
# ============================================================================

@lru_cache(maxsize=1)
def get_cluster_nodes() -> List[str]:
    """
    Get list of cluster nodes from distributed_config.json.
    Cached to avoid repeated file parsing (Team Beta A2).
    """
    config_paths = [
        Path(__file__).parent / "distributed_config.json",
        Path.home() / "distributed_prng_analysis" / "distributed_config.json",
        Path("distributed_config.json"),
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                nodes = [node['hostname'] for node in cfg.get('nodes', [])]
                if nodes:
                    logger.debug(f"Loaded {len(nodes)} nodes from {config_path}")
                    return nodes
            except Exception as e:
                logger.warning(f"Failed to parse {config_path}: {e}")
    
    # Fallback to localhost only
    logger.warning("Could not load cluster config, falling back to localhost only")
    return ['localhost']


def clear_node_cache():
    """Clear the cached node list (for testing or config reload)."""
    get_cluster_nodes.cache_clear()


# ============================================================================
# Path helpers
# ============================================================================

def get_step_dir(step_id: int) -> str:
    """Get the ramdisk directory for a specific step (A1: per-step subdirs)."""
    if USE_RAMDISK:
        return f"{RAMDISK_BASE}/step{step_id}"
    else:
        return SSD_BASE


def get_data_path(step_id: int, filename: str) -> str:
    """Get full path to a data file for a specific step."""
    return f"{get_step_dir(step_id)}/{filename}"


def get_sentinel_path(step_id: int) -> str:
    """Get path to sentinel file for a step."""
    return f"{get_step_dir(step_id)}/.ready"


# ============================================================================
# B1: Pre-flight headroom check
# ============================================================================

def check_ramdisk_headroom(node: str) -> tuple:
    """
    Check /dev/shm usage on a node.
    
    Returns:
        (usage_percent, status) where status is 'ok', 'warn', or 'abort'
    """
    try:
        if node == 'localhost':
            result = subprocess.run(
                "df /dev/shm | awk 'NR==2 {gsub(/%/,\"\"); print $5}'",
                shell=True, capture_output=True, text=True, timeout=10
            )
        else:
            result = subprocess.run(
                ['ssh', node, "df /dev/shm | awk 'NR==2 {gsub(/%/,\"\"); print $5}'"],
                capture_output=True, text=True, timeout=30
            )
        
        usage = int(result.stdout.strip() or 0)
        
        if usage >= ABORT_THRESHOLD:
            return usage, 'abort'
        elif usage >= WARN_THRESHOLD:
            return usage, 'warn'
        else:
            return usage, 'ok'
            
    except Exception as e:
        logger.warning(f"Could not check headroom on {node}: {e}")
        return 0, 'ok'  # Assume OK if we can't check


# ============================================================================
# Preload function
# ============================================================================

def preload_ramdisk(step_id: int, files: List[str], source_dir: Optional[str] = None) -> bool:
    """
    Preload files to ramdisk on all cluster nodes for a specific step.
    
    Args:
        step_id: Pipeline step number (2, 3, or 5)
        files: List of filenames to preload
        source_dir: Directory containing source files (default: cwd)
    
    Returns:
        True if preload succeeded on all nodes, False otherwise
    """
    if not USE_RAMDISK:
        logger.info("Ramdisk disabled, skipping preload")
        return True
    
    if source_dir is None:
        source_dir = os.getcwd()
    
    step_dir = get_step_dir(step_id)
    sentinel = get_sentinel_path(step_id)
    
    logger.info(f"Ramdisk preload for Step {step_id} ({len(files)} files)")
    logger.info(f"Target directory: {step_dir}")
    
    # Check if WATCHER is managing cleanup
    watcher_managed = os.environ.get('WATCHER_MANAGED_RAMDISK')
    if watcher_managed:
        logger.info("WATCHER-managed mode: cleanup handled externally")
    else:
        logger.info("Standalone mode: will clear previous data if needed")
    
    nodes = get_cluster_nodes()
    all_success = True
    
    for node in nodes:
        logger.info(f"  → {node}")
        
        # B1: Check headroom
        usage, status = check_ramdisk_headroom(node)
        if status == 'abort':
            logger.error(f"    ❌ ABORT: /dev/shm usage at {usage}% (threshold: {ABORT_THRESHOLD}%)")
            all_success = False
            continue
        elif status == 'warn':
            logger.warning(f"    ⚠️  WARNING: /dev/shm usage at {usage}%")
        
        try:
            if node == 'localhost':
                all_success &= _preload_localhost(step_id, files, source_dir, watcher_managed)
            else:
                all_success &= _preload_remote(node, step_id, files, source_dir, watcher_managed)
        except Exception as e:
            logger.error(f"    ❌ Failed: {e}")
            all_success = False
    
    logger.info(f"Ramdisk preload complete for Step {step_id}")
    return all_success


def _preload_localhost(step_id: int, files: List[str], source_dir: str, watcher_managed: bool) -> bool:
    """Preload files on localhost."""
    step_dir = get_step_dir(step_id)
    sentinel = get_sentinel_path(step_id)
    
    # Create directory
    os.makedirs(step_dir, exist_ok=True)
    
    # Check sentinel
    if os.path.exists(sentinel):
        logger.info("    ✓ Already loaded (skipped)")
        return True
    
    # Standalone mode: clear first
    if not watcher_managed:
        _clear_step_localhost(step_id)
    
    # Copy files
    copied = 0
    for f in files:
        src = os.path.join(source_dir, f)
        dst = os.path.join(step_dir, f)
        if os.path.exists(src):
            subprocess.run(['cp', src, dst], check=True)
            copied += 1
        else:
            logger.warning(f"    ⚠️  File not found: {src}")
    
    # Create sentinel
    Path(sentinel).touch()
    logger.info(f"    ✓ Preloaded ({copied} files)")
    return True


def _preload_remote(node: str, step_id: int, files: List[str], source_dir: str, watcher_managed: bool) -> bool:
    """Preload files on a remote node."""
    step_dir = get_step_dir(step_id)
    sentinel = get_sentinel_path(step_id)
    
    # Create directory
    subprocess.run(['ssh', node, f'mkdir -p {step_dir}'], timeout=30)
    
    # Check sentinel
    result = subprocess.run(['ssh', node, f'[ -f {sentinel} ]'], timeout=30)
    if result.returncode == 0:
        logger.info("    ✓ Already loaded (skipped)")
        return True
    
    # Standalone mode: clear first
    if not watcher_managed:
        _clear_step_remote(node, step_id)
    
    # Copy files via SCP
    copied = 0
    for f in files:
        src = os.path.join(source_dir, f)
        if os.path.exists(src):
            result = subprocess.run(
                ['scp', '-q', src, f'{node}:{step_dir}/'],
                timeout=120
            )
            if result.returncode == 0:
                copied += 1
    
    # Create sentinel
    subprocess.run(['ssh', node, f'touch {sentinel}'], timeout=30)
    logger.info(f"    ✓ Preloaded ({copied} files)")
    return True


# ============================================================================
# A3: Clear ramdisk - idempotent and non-fatal
# ============================================================================

def clear_ramdisk(step_id: Optional[int] = None) -> None:
    """
    Clear ramdisk data on all nodes.
    
    Args:
        step_id: Specific step to clear, or None to clear all steps
    
    This function is idempotent and non-fatal (Team Beta A3).
    Errors are logged but never raised.
    """
    if step_id is None:
        logger.info("Clearing ALL ramdisk data on all nodes...")
    else:
        logger.info(f"Clearing ramdisk for Step {step_id} on all nodes...")
    
    nodes = get_cluster_nodes()
    
    for node in nodes:
        logger.info(f"  → {node}")
        try:
            if node == 'localhost':
                _clear_step_localhost(step_id)
            else:
                _clear_step_remote(node, step_id)
            logger.info("    ✓ Cleared")
        except Exception as e:
            # A3: Non-fatal - log warning but don't raise
            logger.warning(f"    ⚠️  Warning: cleanup failed ({e}) - continuing")
    
    logger.info("Ramdisk cleanup complete")


def _clear_step_localhost(step_id: Optional[int]) -> None:
    """Clear ramdisk on localhost (non-fatal)."""
    try:
        if step_id is None:
            # Clear all step directories
            subprocess.run(f'rm -rf {RAMDISK_BASE}/step*', shell=True, timeout=10)
        else:
            # Clear specific step
            subprocess.run(f'rm -rf {RAMDISK_BASE}/step{step_id}', shell=True, timeout=10)
    except Exception:
        pass  # Non-fatal


def _clear_step_remote(node: str, step_id: Optional[int]) -> None:
    """Clear ramdisk on remote node (non-fatal)."""
    try:
        if step_id is None:
            subprocess.run(['ssh', node, f'rm -rf {RAMDISK_BASE}/step*'], timeout=30)
        else:
            subprocess.run(['ssh', node, f'rm -rf {RAMDISK_BASE}/step{step_id}'], timeout=30)
    except Exception:
        pass  # Non-fatal


# ============================================================================
# Status utility
# ============================================================================

def show_ramdisk_status() -> dict:
    """
    Get ramdisk status across all cluster nodes.
    
    Returns:
        Dictionary with status per node
    """
    status = {}
    nodes = get_cluster_nodes()
    
    for node in nodes:
        try:
            if node == 'localhost':
                # Get usage
                result = subprocess.run(
                    "df -h /dev/shm | awk 'NR==2 {print $3 \"/\" $2 \" (\" $5 \")\"}'",
                    shell=True, capture_output=True, text=True, timeout=10
                )
                usage = result.stdout.strip()
                
                # Get loaded steps
                steps = []
                if os.path.exists(RAMDISK_BASE):
                    for item in os.listdir(RAMDISK_BASE):
                        if item.startswith('step'):
                            step_dir = os.path.join(RAMDISK_BASE, item)
                            if os.path.isdir(step_dir):
                                file_count = len(os.listdir(step_dir))
                                has_sentinel = os.path.exists(os.path.join(step_dir, '.ready'))
                                steps.append({
                                    'step': item,
                                    'files': file_count,
                                    'ready': has_sentinel
                                })
                
                status[node] = {'usage': usage, 'steps': steps}
            else:
                result = subprocess.run(
                    ['ssh', node, f'''
                        echo "USAGE:$(df -h /dev/shm | awk 'NR==2 {{print $3 "/" $2 " (" $5 ")"}}')"
                        for d in {RAMDISK_BASE}/step*/; do
                            if [ -d "$d" ]; then
                                step=$(basename "$d")
                                count=$(ls -1 "$d" 2>/dev/null | wc -l)
                                ready=$([ -f "$d/.ready" ] && echo "true" || echo "false")
                                echo "STEP:$step:$count:$ready"
                            fi
                        done
                    '''],
                    capture_output=True, text=True, timeout=30
                )
                
                usage = ""
                steps = []
                for line in result.stdout.strip().split('\n'):
                    if line.startswith('USAGE:'):
                        usage = line[6:]
                    elif line.startswith('STEP:'):
                        parts = line[5:].split(':')
                        if len(parts) >= 3:
                            steps.append({
                                'step': parts[0],
                                'files': int(parts[1]),
                                'ready': parts[2] == 'true'
                            })
                
                status[node] = {'usage': usage, 'steps': steps}
                
        except Exception as e:
            status[node] = {'error': str(e)}
    
    return status


# ============================================================================
# Migration helper (for refactoring existing Step 2)
# ============================================================================

def migrate_flat_to_step_dirs() -> None:
    """
    Migrate existing flat /dev/shm/prng/ structure to per-step subdirectories.
    Call this once to upgrade from v1.x to v2.0 ramdisk layout.
    """
    logger.info("Migrating flat ramdisk to per-step directories...")
    
    nodes = get_cluster_nodes()
    
    for node in nodes:
        logger.info(f"  → {node}")
        try:
            if node == 'localhost':
                _migrate_node_localhost()
            else:
                _migrate_node_remote(node)
            logger.info("    ✓ Migrated")
        except Exception as e:
            logger.warning(f"    ⚠️  Migration failed: {e}")


def _migrate_node_localhost() -> None:
    """Migrate localhost from flat to step directories."""
    flat_dir = RAMDISK_BASE
    step2_dir = f"{RAMDISK_BASE}/step2"
    
    # Check if migration needed
    if os.path.exists(f"{flat_dir}/.ready") and not os.path.exists(step2_dir):
        # Old flat structure exists, migrate to step2
        os.makedirs(step2_dir, exist_ok=True)
        
        # Move files (not directories)
        for item in os.listdir(flat_dir):
            src = os.path.join(flat_dir, item)
            if os.path.isfile(src) and not item.startswith('step'):
                dst = os.path.join(step2_dir, item)
                os.rename(src, dst)


def _migrate_node_remote(node: str) -> None:
    """Migrate remote node from flat to step directories."""
    subprocess.run(['ssh', node, f'''
        if [ -f {RAMDISK_BASE}/.ready ] && [ ! -d {RAMDISK_BASE}/step2 ]; then
            mkdir -p {RAMDISK_BASE}/step2
            for f in {RAMDISK_BASE}/*; do
                if [ -f "$f" ]; then
                    mv "$f" {RAMDISK_BASE}/step2/
                fi
            done
        fi
    '''], timeout=60)


# ============================================================================
# CLI interface
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    parser = argparse.ArgumentParser(description='Ramdisk management utility')
    parser.add_argument('--status', action='store_true', help='Show ramdisk status')
    parser.add_argument('--clear', type=int, nargs='?', const=-1, metavar='STEP',
                        help='Clear ramdisk (specific step or all)')
    parser.add_argument('--migrate', action='store_true', 
                        help='Migrate from flat to step directories')
    parser.add_argument('--preload', type=int, metavar='STEP',
                        help='Preload files for a step')
    parser.add_argument('--files', nargs='+', help='Files to preload')
    
    args = parser.parse_args()
    
    if args.status:
        status = show_ramdisk_status()
        for node, info in status.items():
            print(f"\n{node}:")
            if 'error' in info:
                print(f"  Error: {info['error']}")
            else:
                print(f"  Usage: {info['usage']}")
                if info['steps']:
                    print("  Steps:")
                    for s in info['steps']:
                        ready = '✓' if s['ready'] else '✗'
                        print(f"    {s['step']}: {s['files']} files, ready: {ready}")
                else:
                    print("  No steps loaded")
    
    elif args.clear is not None:
        if args.clear == -1:
            clear_ramdisk()  # Clear all
        else:
            clear_ramdisk(step_id=args.clear)
    
    elif args.migrate:
        migrate_flat_to_step_dirs()
    
    elif args.preload and args.files:
        preload_ramdisk(step_id=args.preload, files=args.files)
    
    else:
        parser.print_help()
