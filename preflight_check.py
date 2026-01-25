#!/usr/bin/env python3
"""
Preflight Check Module
======================
Version: 1.0.1
Date: January 24, 2026
Team Beta Approved: Item A (Conditional - bugs fixed)

Changes in v1.0.1:
- Fixed sys.exit() bug in CLI
- Use bash -lc for SSH commands (safer venv activation)
- Hardened GPU count parsing

Purpose: Verify cluster health BEFORE pipeline steps execute.
Called by watcher_agent.py to fail fast on infrastructure issues.

Integration with watcher_agent.py:
    from preflight_check import PreflightChecker, PreflightResult
    
    checker = PreflightChecker()
    result = checker.check_all(step=step_num)
    if not result.passed:
        return StepResult(success=False, reason=f"Preflight: {result.failures}")

Standalone Testing:
    python3 preflight_check.py --step 1 --verbose
    python3 preflight_check.py --step 3 --save
"""

import json
import subprocess
import time
import sys
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION CONSTANTS (Team Beta: tunable timeouts)
# ════════════════════════════════════════════════════════════════════════════════

SSH_TIMEOUT_SECONDS = 5
GPU_CHECK_TIMEOUT_SECONDS = 15
RAMDISK_CHECK_TIMEOUT_SECONDS = 10


@dataclass
class PreflightResult:
    """Result of preflight checks."""
    passed: bool = True
    checks_run: int = 0
    checks_passed: int = 0
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    
    def add_failure(self, message: str):
        self.passed = False
        self.failures.append(message)
    
    def add_warning(self, message: str):
        self.warnings.append(message)
    
    def summary(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        lines = [
            f"Preflight Check: {status}",
            f"Checks: {self.checks_passed}/{self.checks_run} passed",
            f"Duration: {self.duration_seconds:.1f}s"
        ]
        if self.failures:
            lines.append("Failures:")
            for f in self.failures:
                lines.append(f"  ✗ {f}")
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "checks_run": self.checks_run,
            "checks_passed": self.checks_passed,
            "failures": self.failures,
            "warnings": self.warnings,
            "details": self.details,
            "duration_seconds": self.duration_seconds,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }


class PreflightChecker:
    """
    Preflight checker for PRNG Analysis Pipeline.
    Reads node config from distributed_config.json.
    """
    
    # Steps requiring ramdisk on remote nodes
    RAMDISK_REQUIRED_STEPS = {3}
    
    # Required ramdisk files per step
    RAMDISK_FILES = {
        3: ["train_history.json", "holdout_history.json"]
    }
    
    # Required input files per step (on Zeus)
    STEP_INPUTS = {
        1: ["synthetic_lottery.json"],
        2: ["bidirectional_survivors.json", "optimal_window_config.json"],
        3: ["bidirectional_survivors_binary.npz", "optimal_scorer_config.json"],
        4: ["survivors_with_scores.json"],
        5: ["survivors_with_scores.json"],
        6: []
    }
    
    def __init__(self, config_file: str = "distributed_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        self.nodes = self._parse_nodes()
    
    def _load_config(self) -> Dict:
        try:
            with open(self.config_file) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Config load error: {e}")
            return {"nodes": []}
    
    def _parse_nodes(self) -> List[Dict]:
        """Extract remote nodes (skip localhost)."""
        nodes = []
        for node in self.config.get("nodes", []):
            hostname = node.get("hostname", "")
            if hostname and hostname != "localhost":
                nodes.append({
                    "hostname": hostname,
                    "gpu_count": node.get("gpu_count", 12),
                    "ramdisk_path": node.get("ramdisk_path", "/dev/shm/prng")
                })
        return nodes
    
    def check_all(self, step: int, params: Optional[Dict] = None, 
                  run_id: Optional[str] = None) -> PreflightResult:
        """
        Run all preflight checks for a given step.
        
        Args:
            step: Pipeline step number (1-6)
            params: Optional parameters (e.g., lottery_file override)
            run_id: Optional run ID for result file naming
        """
        start_time = time.time()
        result = PreflightResult()
        params = params or {}
        
        logger.info(f"[PREFLIGHT] Running checks for Step {step}...")
        
        # 1. SSH connectivity
        result.checks_run += 1
        ssh_result = self.check_ssh_connectivity()
        result.details["ssh"] = ssh_result
        if ssh_result["all_reachable"]:
            result.checks_passed += 1
            logger.info(f"[PREFLIGHT] SSH: ✅ {len(ssh_result['reachable'])} nodes reachable")
        else:
            result.add_failure(f"SSH unreachable: {ssh_result['unreachable']}")
        
        # 2. GPU health (warning only, not blocking)
        result.checks_run += 1
        gpu_result = self.check_gpu_health()
        result.details["gpu"] = gpu_result
        if gpu_result["all_healthy"]:
            result.checks_passed += 1
            logger.info("[PREFLIGHT] GPUs: ✅ All responding")
        else:
            for issue in gpu_result.get("issues", []):
                # Team Beta: Use structured warning format
                if isinstance(issue, dict):
                    result.add_warning(f"GPU: {issue['node']} - {issue['type']}: {issue.get('observed')}/{issue.get('expected')}")
                else:
                    result.add_warning(f"GPU: {issue}")
            result.checks_passed += 1  # Don't block on GPU warnings
        
        # 3. Ramdisk (only for steps that need it)
        if step in self.RAMDISK_REQUIRED_STEPS:
            result.checks_run += 1
            ramdisk_result = self.check_ramdisk(step)
            result.details["ramdisk"] = ramdisk_result
            if ramdisk_result["populated"]:
                result.checks_passed += 1
                logger.info("[PREFLIGHT] Ramdisk: ✅ Populated")
            else:
                result.add_failure(f"Ramdisk missing: {ramdisk_result['missing']}")
        
        # 4. Input files on Zeus
        result.checks_run += 1
        inputs_result = self.check_step_inputs(step, params)
        result.details["inputs"] = inputs_result
        if inputs_result["all_present"]:
            result.checks_passed += 1
            logger.info("[PREFLIGHT] Inputs: ✅ Present")
        else:
            result.add_failure(f"Missing inputs: {inputs_result['missing']}")
        
        result.duration_seconds = time.time() - start_time
        
        status = "✅ PASSED" if result.passed else "❌ FAILED"
        logger.info(f"[PREFLIGHT] {status} ({result.checks_passed}/{result.checks_run}, {result.duration_seconds:.1f}s)")
        
        # Team Beta recommendation: Auto-persist for forensic traceability
        if run_id:
            self.save_result(result, f"preflight_result_{run_id}_step{step}.json")
        
        return result
    
    def check_ssh_connectivity(self) -> Dict:
        """Check SSH to all remote nodes."""
        reachable, unreachable = [], []
        
        for node in self.nodes:
            hostname = node["hostname"]
            try:
                cmd = [
                    "ssh", 
                    "-o", f"ConnectTimeout={SSH_TIMEOUT_SECONDS}",
                    "-o", "BatchMode=yes",
                    hostname, 
                    "echo OK"
                ]
                proc = subprocess.run(cmd, capture_output=True, timeout=SSH_TIMEOUT_SECONDS + 2)
                if proc.returncode == 0 and b"OK" in proc.stdout:
                    reachable.append(hostname)
                else:
                    unreachable.append(hostname)
            except Exception:
                unreachable.append(hostname)
        
        return {
            "all_reachable": len(unreachable) == 0,
            "reachable": reachable,
            "unreachable": unreachable
        }
    
    def check_gpu_health(self) -> Dict:
        """Check GPU availability via rocm-smi."""
        results = {"all_healthy": True, "nodes": {}, "issues": []}
        
        for node in self.nodes:
            hostname = node["hostname"]
            expected = node.get("gpu_count", 12)
            
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
                    
                    results["nodes"][hostname] = {"gpu_count": gpu_count, "expected": expected}
                    if gpu_count < expected:
                        # Team Beta recommendation: Structured issue tags
                        results["issues"].append({
                            "node": hostname,
                            "type": "GPU_COUNT_MISMATCH",
                            "observed": gpu_count,
                            "expected": expected
                        })
                        results["all_healthy"] = False
                else:
                    results["nodes"][hostname] = {"error": "rocm-smi failed"}
                    results["issues"].append({
                        "node": hostname,
                        "type": "ROCM_SMI_FAILED",
                        "observed": None,
                        "expected": expected
                    })
                    results["all_healthy"] = False
            except subprocess.TimeoutExpired:
                results["nodes"][hostname] = {"error": "timeout"}
                results["issues"].append({
                    "node": hostname,
                    "type": "TIMEOUT",
                    "observed": None,
                    "expected": expected
                })
                results["all_healthy"] = False
            except Exception as e:
                results["nodes"][hostname] = {"error": str(e)}
                results["issues"].append({
                    "node": hostname,
                    "type": "ERROR",
                    "observed": str(e),
                    "expected": expected
                })
                results["all_healthy"] = False
        
        return results
    
    def check_ramdisk(self, step: int) -> Dict:
        """Check ramdisk files on remote nodes."""
        required_files = self.RAMDISK_FILES.get(step, [])
        if not required_files:
            return {"populated": True, "missing": []}
        
        results = {"populated": True, "missing": [], "nodes": {}}
        
        for node in self.nodes:
            hostname = node["hostname"]
            ramdisk_path = f"{node['ramdisk_path']}/step{step}"
            
            try:
                file_checks = " && ".join([f"test -f {ramdisk_path}/{f}" for f in required_files])
                cmd = [
                    "ssh", "-o", f"ConnectTimeout={SSH_TIMEOUT_SECONDS}", hostname,
                    "bash", "-c",
                    f"({file_checks}) && echo OK || echo MISSING"
                ]
                proc = subprocess.run(cmd, capture_output=True, timeout=RAMDISK_CHECK_TIMEOUT_SECONDS)
                
                output = proc.stdout.decode().strip()
                if "OK" in output:
                    results["nodes"][hostname] = {"populated": True}
                else:
                    results["nodes"][hostname] = {"populated": False}
                    for f in required_files:
                        results["missing"].append(f"{hostname}:{ramdisk_path}/{f}")
                    results["populated"] = False
            except Exception as e:
                results["missing"].append(f"{hostname}: {e}")
                results["populated"] = False
        
        return results
    
    def check_step_inputs(self, step: int, params: Optional[Dict] = None) -> Dict:
        """Check input files exist on Zeus."""
        params = params or {}
        required = list(self.STEP_INPUTS.get(step, []))
        
        # Override for Step 1 lottery file
        if step == 1 and "lottery_file" in params:
            required = [params["lottery_file"]]
        
        present = [f for f in required if Path(f).exists()]
        missing = [f for f in required if not Path(f).exists()]
        
        return {"all_present": len(missing) == 0, "present": present, "missing": missing}
    
    def save_result(self, result: PreflightResult, output_file: str = "preflight_result.json"):
        """Save result to JSON."""
        with open(output_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"[PREFLIGHT] Saved to {output_file}")


# ════════════════════════════════════════════════════════════════════════════════
# CLI Interface
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preflight checks for pipeline steps")
    parser.add_argument("--step", type=int, default=1, help="Pipeline step (1-6)")
    parser.add_argument("--config", default="distributed_config.json")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--save", action="store_true", help="Save result to JSON")
    parser.add_argument("--run-id", help="Run ID for result file naming")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    checker = PreflightChecker(args.config)
    result = checker.check_all(args.step, run_id=args.run_id)
    
    print("\n" + "=" * 50)
    print(result.summary())
    print("=" * 50)
    
    if args.save:
        checker.save_result(result)
    
    # Team Beta FIX: Proper sys.exit()
    sys.exit(0 if result.passed else 1)
