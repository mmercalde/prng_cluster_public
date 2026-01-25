#!/usr/bin/env python3
"""
GPU Cleanup Module
==================
Version: 1.0.1
Date: January 24, 2026
Team Beta Approved: Item C (Conditional - bugs fixed)

Changes in v1.0.1:
- Use bash -lc for SSH commands (safer venv activation)
- Hardened GPU count parsing
- Explicit: Cleanup failures are WARNINGS, never block the run

Purpose: Clean up GPU state after job batches to prevent:
- HIP context accumulation
- SMU polling issues
- Memory fragmentation

Design Notes (Team Beta approved):
- Cleanup failures → warnings, never fail the run
- No forced resets by default (conservative)
- Only touches caches + temp files

Integration with scripts_coordinator.py:
    from gpu_cleanup import post_batch_cleanup
    
    # After batch completes:
    post_batch_cleanup(nodes=["192.168.3.120", "192.168.3.154"])

Standalone Usage:
    python3 gpu_cleanup.py --all
    python3 gpu_cleanup.py --node 192.168.3.120
"""

import json
import subprocess
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION CONSTANTS (Team Beta: tunable timeouts)
# ════════════════════════════════════════════════════════════════════════════════

SSH_TIMEOUT_SECONDS = 5
CLEANUP_TIMEOUT_SECONDS = 30
RESET_TIMEOUT_SECONDS = 60
GPU_CHECK_TIMEOUT_SECONDS = 15


def cleanup_single_node(hostname: str, timeout: int = CLEANUP_TIMEOUT_SECONDS) -> Dict:
    """
    Run GPU cleanup on a single remote node.
    
    Actions performed:
    1. Clear HIP cache files
    2. Clear ROCm temp files
    3. Sync filesystem
    
    Note: Cleanup failures are WARNINGS, not errors.
    """
    result = {
        "hostname": hostname,
        "success": False,
        "actions": [],
        "errors": []
    }
    
    # Conservative cleanup - no clock resets by default
    cleanup_commands = [
        "rm -rf ~/.cache/hip_* 2>/dev/null || true",
        "rm -rf /tmp/rocm_* 2>/dev/null || true",
        "sync"
    ]
    
    combined_cmd = " && ".join(cleanup_commands)
    
    try:
        # Team Beta FIX: Use bash -lc for safe venv activation
        cmd = [
            "ssh", "-o", f"ConnectTimeout={SSH_TIMEOUT_SECONDS}", "-o", "BatchMode=yes",
            hostname,
            "bash", "-lc",
            f"source ~/rocm_env/bin/activate 2>/dev/null; {combined_cmd}"
        ]
        
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
        
        if proc.returncode == 0:
            result["success"] = True
            result["actions"] = ["hip_cache_clear", "rocm_temp_clear", "sync"]
            logger.debug(f"[CLEANUP] {hostname}: ✅ Completed")
        else:
            # Cleanup failures are warnings, not fatal errors
            result["errors"].append(f"Exit code {proc.returncode}")
            stderr = proc.stderr.decode().strip()
            if stderr:
                result["errors"].append(stderr[:200])  # Truncate long errors
            logger.warning(f"[CLEANUP] {hostname}: ⚠️ rc={proc.returncode}")
            
    except subprocess.TimeoutExpired:
        result["errors"].append("Timeout")
        logger.warning(f"[CLEANUP] {hostname}: ⚠️ Timeout")
    except Exception as e:
        result["errors"].append(str(e))
        logger.warning(f"[CLEANUP] {hostname}: ⚠️ {e}")
    
    return result


def cleanup_all_nodes(
    nodes: Optional[List[str]] = None,
    config_file: str = "distributed_config.json"
) -> Dict:
    """
    Run GPU cleanup on all remote nodes.
    
    Args:
        nodes: List of hostnames. If None, reads from config.
        config_file: Path to distributed_config.json
        
    Returns:
        Dict with results per node
        
    Note: Overall success is True even if some nodes had issues.
          Cleanup failures are informational only.
    """
    # Get nodes from config if not provided
    if nodes is None:
        try:
            with open(config_file) as f:
                config = json.load(f)
            nodes = [
                n["hostname"] for n in config.get("nodes", [])
                if n.get("hostname") and n["hostname"] != "localhost"
            ]
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {"success": True, "error": str(e), "nodes": {}}  # Don't fail on config error
    
    if not nodes:
        logger.info("[CLEANUP] No remote nodes to clean")
        return {"success": True, "nodes": {}}
    
    logger.info(f"[CLEANUP] Cleaning {len(nodes)} nodes...")
    
    results = {"success": True, "nodes": {}}
    
    for hostname in nodes:
        node_result = cleanup_single_node(hostname)
        results["nodes"][hostname] = node_result
        # Note: We don't set success=False even if node cleanup fails
        # Cleanup failures are warnings, never blockers
    
    success_count = sum(1 for r in results["nodes"].values() if r["success"])
    logger.info(f"[CLEANUP] Complete: {success_count}/{len(nodes)} nodes cleaned")
    
    return results


def cleanup_with_gpu_reset(hostname: str, timeout: int = RESET_TIMEOUT_SECONDS) -> Dict:
    """
    More aggressive cleanup including GPU clock reset.
    
    Use sparingly - clock reset can take time and may disrupt running jobs.
    """
    result = {
        "hostname": hostname,
        "success": False,
        "actions": [],
        "errors": []
    }
    
    commands = [
        "rm -rf ~/.cache/hip_* 2>/dev/null || true",
        "rm -rf /tmp/rocm_* 2>/dev/null || true",
        "rocm-smi --resetclocks 2>/dev/null || true",
        "sync",
        "sleep 2"
    ]
    
    combined_cmd = " && ".join(commands)
    
    try:
        # Team Beta FIX: Use bash -lc for safe venv activation
        cmd = [
            "ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
            hostname,
            "bash", "-lc",
            f"source ~/rocm_env/bin/activate && {combined_cmd}"
        ]
        
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
        
        if proc.returncode == 0:
            result["success"] = True
            result["actions"] = ["hip_cache_clear", "rocm_temp_clear", "clock_reset", "sync"]
            logger.info(f"[CLEANUP+RESET] {hostname}: ✅ Completed")
        else:
            result["errors"].append(f"Exit code {proc.returncode}")
            logger.warning(f"[CLEANUP+RESET] {hostname}: ⚠️ rc={proc.returncode}")
            
    except subprocess.TimeoutExpired:
        result["errors"].append("Timeout (clock reset may be slow)")
        logger.warning(f"[CLEANUP+RESET] {hostname}: ⚠️ Timeout")
    except Exception as e:
        result["errors"].append(str(e))
        logger.warning(f"[CLEANUP+RESET] {hostname}: ⚠️ {e}")
    
    return result


def verify_gpu_health_after_cleanup(hostname: str) -> Dict:
    """Verify GPUs are healthy after cleanup."""
    result = {
        "hostname": hostname,
        "healthy": False,
        "gpu_count": 0,
        "errors": []
    }
    
    try:
        # Team Beta FIX: Use bash -lc for safe venv activation
        cmd = [
            "ssh", "-o", f"ConnectTimeout={SSH_TIMEOUT_SECONDS}", hostname,
            "bash", "-lc",
            "source ~/rocm_env/bin/activate && rocm-smi --showuse 2>/dev/null | grep -c 'GPU\\[' || echo 0"
        ]
        
        proc = subprocess.run(cmd, capture_output=True, timeout=GPU_CHECK_TIMEOUT_SECONDS)
        
        if proc.returncode == 0:
            # Team Beta FIX: Hardened GPU count parsing
            output = proc.stdout.decode()
            lines = [l.strip() for l in output.splitlines() if l.strip().isdigit()]
            gpu_count = int(lines[-1]) if lines else 0
            result["gpu_count"] = gpu_count
            result["healthy"] = gpu_count > 0
        else:
            result["errors"].append("rocm-smi failed")
            
    except Exception as e:
        result["errors"].append(str(e))
    
    return result


# ════════════════════════════════════════════════════════════════════════════════
# Integration helper for scripts_coordinator.py
# ════════════════════════════════════════════════════════════════════════════════

def post_batch_cleanup(nodes: List[str], verify: bool = False) -> bool:
    """
    Convenience function for scripts_coordinator.py integration.
    
    Call after a batch of jobs completes:
    
        from gpu_cleanup import post_batch_cleanup
        
        # After batch:
        post_batch_cleanup(["192.168.3.120", "192.168.3.154"])
    
    Args:
        nodes: List of node hostnames
        verify: If True, verify GPU health after cleanup
        
    Returns:
        True always - cleanup failures are warnings, never blockers
    """
    results = cleanup_all_nodes(nodes=nodes)
    
    if verify:
        logger.info("[CLEANUP] Verifying GPU health...")
        for hostname in nodes:
            health = verify_gpu_health_after_cleanup(hostname)
            if not health["healthy"]:
                logger.warning(f"[CLEANUP] {hostname}: GPU health check found issues (non-blocking)")
    
    # Always return True - cleanup failures never block the pipeline
    return True


# ════════════════════════════════════════════════════════════════════════════════
# CLI Interface
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU cleanup for remote nodes")
    parser.add_argument("--all", action="store_true", help="Clean all nodes from config")
    parser.add_argument("--node", type=str, help="Clean specific node")
    parser.add_argument("--reset", action="store_true", help="Include GPU clock reset (slower)")
    parser.add_argument("--verify", action="store_true", help="Verify GPU health after cleanup")
    parser.add_argument("--config", default="distributed_config.json")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    if args.node:
        # Single node
        if args.reset:
            result = cleanup_with_gpu_reset(args.node)
        else:
            result = cleanup_single_node(args.node)
        
        if args.verify:
            health = verify_gpu_health_after_cleanup(args.node)
            result["health_check"] = health
        
        print(f"\nResult: {'✅ Success' if result['success'] else '⚠️ Issues (non-blocking)'}")
        print(f"Actions: {result.get('actions', [])}")
        if result.get("errors"):
            print(f"Warnings: {result['errors']}")
            
    elif args.all:
        # All nodes
        results = cleanup_all_nodes(config_file=args.config)
        
        print(f"\nCleanup Complete:")
        for hostname, r in results.get("nodes", {}).items():
            status = "✅" if r["success"] else "⚠️"
            print(f"  {status} {hostname}")
            
        if args.verify:
            print("\nGPU Health Check:")
            with open(args.config) as f:
                config = json.load(f)
            nodes = [n["hostname"] for n in config.get("nodes", []) 
                    if n.get("hostname") and n["hostname"] != "localhost"]
            for hostname in nodes:
                health = verify_gpu_health_after_cleanup(hostname)
                status = "✅" if health["healthy"] else "⚠️"
                print(f"  {status} {hostname}: {health['gpu_count']} GPUs")
    else:
        parser.print_help()
