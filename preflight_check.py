#!/usr/bin/env python3
"""
Preflight Check Module
======================
Version: 1.1.0
Date: January 25, 2026
Team Beta Approved: Item A

Changes in v1.1.0:
- AUTO-REMEDIATE: Ramdisk now auto-populates if missing (calls ramdisk_preload.sh)
- FIXED: GPU count detection now uses correct rocm-smi parsing
- Added remediate_ramdisk() method
- Added --no-remediate flag for check-only mode

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
    python3 preflight_check.py --step 3 --verbose
    python3 preflight_check.py --step 3 --no-remediate  # Check only, don't fix
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
RAMDISK_REMEDIATION_TIMEOUT_SECONDS = 60


# ════════════════════════════════════════════════════════════════════════════════
# GPU PROBE — THREE OUTCOMES, NEVER TWO
# ════════════════════════════════════════════════════════════════════════════════
# [2026-08-09] The previous probe ran
#
#     ssh <host> bash -lc "rocm-smi 2>/dev/null | grep -cE '^[0-9]+[[:space:]]' || echo 0"
#
# and reported all three rigs as 0/8 while they each had 8 healthy GPUs. The
# PARSING was correct; the OBSERVATION was not. Two constructs conspired:
# `2>/dev/null` swallowed the "command not found" diagnostic, and `|| echo 0`
# converted an unobservable surface into a definite count of zero.
#
# ROOT CAUSE, measured live on all three CTs 2026-08-09 (not inferred):
#   * /opt/rocm/bin is placed on PATH by ~/.bashrc:120 and by nothing else —
#     no /etc/profile.d script, no /etc/profile, no /etc/environment entry
#     mentions rocm.
#   * ~/.bashrc:5-8 is Ubuntu's stock non-interactive guard
#     (`case $- in *i*) ;; *) return;; esac`), which returns ~112 lines BEFORE
#     the PATH export.
#   * `bash -l` sources /etc/profile and ~/.profile, and ~/.profile does source
#     ~/.bashrc — but .bashrc returns at the guard. So the login shell is NOT a
#     remedy here: `bash -lc` and a bare non-interactive command observe the
#     IDENTICAL PATH, neither containing /opt/rocm/bin. Only `bash -lic` sees it,
#     and forcing an interactive shell over SSH is not a probe contract.
#
# The remedy is therefore to LOCATE the binary rather than assume a PATH: prefer
# whatever `command -v` resolves (so a rig that installs it elsewhere still
# works), and fall back to the verified absolute path. Diagnostics are captured,
# never discarded.
#
# VIR-5: an inaccessible surface is not a clean one. UNAVAILABLE is not zero.
GPU_PROBE_OK = "OK"                    # probe ran; count is an observation
GPU_PROBE_UNAVAILABLE = "UNAVAILABLE"  # probe could not run — NOT zero
GPU_PROBE_ERROR = "ERROR"              # probe ran; output could not be parsed

# Fallback absolute path, live-verified on 192.168.3.122/.156/.164 (2026-08-09):
#   /opt/rocm/bin/rocm-smi -> ../libexec/rocm_smi/rocm_smi.py  (/opt/rocm -> /opt/rocm-6.4.3)
# PATH resolution takes precedence; this is only consulted when `command -v`
# finds nothing, and its absence is reported as UNAVAILABLE rather than as 0.
ROCM_SMI_FALLBACK_PATHS = ("/opt/rocm/bin/rocm-smi",)

_PROBE_BIN = "TFM_PROBE_BIN="
_PROBE_STATUS = "TFM_PROBE_STATUS="
_PROBE_COUNT = "TFM_PROBE_COUNT="


def _build_gpu_probe_script(fallbacks=None) -> str:
    """The remote command, as ONE ssh argument.

    Passed as a single argv element deliberately: ssh joins multiple trailing
    arguments with spaces and does NOT re-quote them, so the old
    `["bash", "-lc", "<pipeline>"]` form was re-parsed by the remote login shell
    with its quoting already flattened. One string is parsed exactly once.

    No `2>/dev/null` and no `|| echo 0`: stderr flows back over the SSH channel
    and is captured by the caller, and a probe that could not run says so.
    """
    # Resolved at CALL time, not bound as a default: the fallback list is a
    # module global so a test can repoint it without patching this function.
    if fallbacks is None:
        fallbacks = ROCM_SMI_FALLBACK_PATHS
    lines = ['RS=$(command -v rocm-smi 2>/dev/null)']
    for path in fallbacks:
        lines.append(f'if [ -z "$RS" ] && [ -x {path} ]; then RS={path}; fi')
    lines += [
        f'if [ -z "$RS" ]; then echo "{_PROBE_STATUS}NO_BINARY"; exit 0; fi',
        f'echo "{_PROBE_BIN}$RS"',
        'OUT=$("$RS"); RC=$?',
        f'if [ "$RC" -ne 0 ]; then echo "{_PROBE_STATUS}EXIT_$RC"; exit 0; fi',
        f'echo "{_PROBE_STATUS}OK"',
        "printf '%s\\n' \"$OUT\" | grep -cE '^[0-9]+[[:space:]]' "
        f"| sed 's/^/{_PROBE_COUNT}/'",
    ]
    return "; ".join(lines)


def _parse_gpu_probe(stdout: str) -> Dict[str, Any]:
    """Classify probe stdout into exactly one of the three outcomes.

    Kept separate from the SSH call so the classification is testable without a
    rig, and so a future transport change cannot quietly alter the semantics.
    """
    binary = None
    status_token = None
    count = None
    saw_count_line = False

    for raw in stdout.splitlines():
        line = raw.strip()
        if line.startswith(_PROBE_BIN):
            binary = line[len(_PROBE_BIN):] or None
        elif line.startswith(_PROBE_STATUS):
            status_token = line[len(_PROBE_STATUS):]
        elif line.startswith(_PROBE_COUNT):
            saw_count_line = True
            value = line[len(_PROBE_COUNT):].strip()
            count = int(value) if value.isdigit() else None

    if status_token is None:
        # The probe never announced itself: we cannot say it saw zero devices.
        return {"status": GPU_PROBE_ERROR, "gpu_count": None, "binary": binary,
                "reason": "probe_emitted_no_status"}
    if status_token == "NO_BINARY":
        return {"status": GPU_PROBE_UNAVAILABLE, "gpu_count": None, "binary": None,
                "reason": "binary_not_found"}
    if status_token.startswith("EXIT_"):
        return {"status": GPU_PROBE_UNAVAILABLE, "gpu_count": None, "binary": binary,
                "reason": f"rocm_smi_exit_{status_token[len('EXIT_'):]}"}
    if status_token != "OK":
        return {"status": GPU_PROBE_ERROR, "gpu_count": None, "binary": binary,
                "reason": f"unrecognized_status:{status_token}"}
    if not saw_count_line or count is None:
        return {"status": GPU_PROBE_ERROR, "gpu_count": None, "binary": binary,
                "reason": "unparseable_device_count"}
    return {"status": GPU_PROBE_OK, "gpu_count": count, "binary": binary,
            "reason": None}


# ─────────────────────────────────────────────────────────────────────────────
# [RESOLVED EXECUTION SET] consumer seam — see coordinator.py for the rationale.
# Lazy, defensive, and a no-op when no set is frozen.
# ─────────────────────────────────────────────────────────────────────────────
def _execution_set_nodes(node_dicts, *, consumer: str):
    try:
        from execution_set import filter_config_nodes
    except ImportError:
        return list(node_dicts)
    return filter_config_nodes(node_dicts, consumer=consumer)


def _render_gpu_issue(issue: Dict[str, Any]) -> str:
    """Warning text for one GPU finding.

    An UNAVAILABLE node must never render as `0/8` or `None/8` — the whole point
    of the three-outcome probe is lost the moment the operator-facing string
    puts an un-observed surface into a count-shaped slot.
    """
    node = issue.get("node")
    expected = issue.get("expected")
    status = issue.get("status")
    reason = issue.get("reason")
    stderr = (issue.get("stderr") or "").strip()

    if status in (GPU_PROBE_UNAVAILABLE, GPU_PROBE_ERROR):
        text = (f"GPU: {node} - {issue.get('type')}: device count {status} "
                f"(expected {expected}) — NOT observed as zero; reason={reason}")
        if stderr:
            text += f"; stderr={stderr.splitlines()[0][:200]}"
        return text
    return (f"GPU: {node} - {issue.get('type')}: "
            f"{issue.get('observed')}/{expected}")


@dataclass
class PreflightResult:
    """Result of preflight checks."""
    passed: bool = True
    checks_run: int = 0
    checks_passed: int = 0
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    remediations: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    
    def add_failure(self, message: str):
        self.passed = False
        self.failures.append(message)
    
    def add_warning(self, message: str):
        self.warnings.append(message)
    
    def add_remediation(self, message: str):
        self.remediations.append(message)
    
    def summary(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        lines = [
            f"Preflight Check: {status}",
            f"Checks: {self.checks_passed}/{self.checks_run} passed",
            f"Duration: {self.duration_seconds:.1f}s"
        ]
        if self.remediations:
            lines.append("Remediations Applied:")
            for r in self.remediations:
                lines.append(f"  🔧 {r}")
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
            "remediations": self.remediations,
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
        """Extract remote nodes (skip localhost).

        [RESOLVED EXECUTION SET] `check_gpu_health` compared the live rocm-smi
        count against `distributed_config.json`'s `gpu_count` at the BARE-METAL
        addresses — a comparison that cannot complete while the rigs are in
        Proxmox. The node list and the addresses now come from the run's frozen
        set instead. Deliberately unchanged: this checker stays NON-BLOCKING
        (`check_all` records GPU issues via `add_warning` and still counts the
        check as passed, :192-206), and localhost is still excluded, because
        neither property is a defect. Only the target list moves.
        """
        nodes = []
        for node in _execution_set_nodes(self.config.get("nodes", []),
                                         consumer="PreflightChecker"):
            hostname = node.get("hostname", "")
            if hostname and hostname != "localhost":
                nodes.append({
                    "hostname": hostname,
                    "gpu_count": node.get("gpu_count", 12),
                    "ramdisk_path": node.get("ramdisk_path", "/dev/shm/prng")
                })
        return nodes
    
    def check_all(self, step: int, params: Optional[Dict] = None, 
                  run_id: Optional[str] = None,
                  auto_remediate: bool = True) -> PreflightResult:
        """
        Run all preflight checks for a given step.
        
        Args:
            step: Pipeline step number (1-6)
            params: Optional parameters (e.g., lottery_file override)
            run_id: Optional run ID for result file naming
            auto_remediate: If True, attempt to fix issues (e.g., populate ramdisk)
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
                    result.add_warning(_render_gpu_issue(issue))
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
            elif auto_remediate:
                # AUTO-REMEDIATE: Try to populate ramdisk
                logger.info("[PREFLIGHT] Ramdisk: ⚠️ Missing - attempting auto-remediation...")
                remediate_success = self.remediate_ramdisk(step)
                
                if remediate_success:
                    # Re-check after remediation
                    ramdisk_result = self.check_ramdisk(step)
                    result.details["ramdisk"] = ramdisk_result
                    
                    if ramdisk_result["populated"]:
                        result.checks_passed += 1
                        result.add_remediation(f"Ramdisk populated for Step {step}")
                        logger.info("[PREFLIGHT] Ramdisk: ✅ Auto-remediated")
                    else:
                        # Steps with script-level preload: informational, not failure
                        PRELOAD_STEPS = {3}
                        if step in PRELOAD_STEPS:
                            result.add_warning(f"Ramdisk not yet populated — preload scheduled: {ramdisk_result['missing']}")
                            result.checks_passed += 1  # Preload will handle it
                        else:
                            result.add_failure(f"Ramdisk remediation failed: {ramdisk_result['missing']}")
                else:
                    result.add_failure(f"Ramdisk remediation script failed for Step {step}")
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
        """Check GPU availability via rocm-smi.

        Reports exactly one of three outcomes per node, which are never
        conflated (see the GPU PROBE block at the top of this module):

          * OK          — the probe ran; `gpu_count` is a real observation, and
                          a genuine zero is reported as a genuine zero.
          * UNAVAILABLE — the probe could not run (binary absent, non-zero exit,
                          SSH failure, timeout). THIS IS NOT ZERO. `gpu_count`
                          is None and no count is invented.
          * ERROR       — the probe ran but produced output we cannot parse.

        Gating is deliberately UNCHANGED: `check_all` records every finding here
        via `add_warning` and still counts the check as passed (:229). This
        method tells the truth about what was observed; it does not decide
        whether the run proceeds.
        """
        results = {"all_healthy": True, "nodes": {}, "issues": []}
        script = _build_gpu_probe_script()

        for node in self.nodes:
            hostname = node["hostname"]
            expected = node.get("gpu_count", 12)

            def _unavailable(reason: str, stderr: str = "", binary=None):
                """An unobservable surface is not a clean one (VIR-5)."""
                results["nodes"][hostname] = {
                    "status": GPU_PROBE_UNAVAILABLE, "gpu_count": None,
                    "expected": expected, "reason": reason,
                    "binary": binary, "stderr": stderr,
                }
                results["issues"].append({
                    "node": hostname,
                    "type": "GPU_PROBE_UNAVAILABLE",
                    "status": GPU_PROBE_UNAVAILABLE,
                    "observed": None,
                    "expected": expected,
                    "reason": reason,
                    "stderr": stderr,
                })
                results["all_healthy"] = False

            try:
                cmd = [
                    "ssh",
                    "-o", f"ConnectTimeout={SSH_TIMEOUT_SECONDS}",
                    "-o", "BatchMode=yes",
                    hostname,
                    script,          # ONE argument — see _build_gpu_probe_script
                ]
                proc = subprocess.run(cmd, capture_output=True,
                                      timeout=GPU_CHECK_TIMEOUT_SECONDS)
                stderr = proc.stderr.decode(errors="replace").strip()

                if proc.returncode != 0:
                    _unavailable(f"ssh_exit_{proc.returncode}", stderr)
                    continue

                parsed = _parse_gpu_probe(proc.stdout.decode(errors="replace"))

                if parsed["status"] == GPU_PROBE_UNAVAILABLE:
                    _unavailable(parsed["reason"], stderr, parsed["binary"])
                    continue

                if parsed["status"] == GPU_PROBE_ERROR:
                    results["nodes"][hostname] = {
                        "status": GPU_PROBE_ERROR, "gpu_count": None,
                        "expected": expected, "reason": parsed["reason"],
                        "binary": parsed["binary"], "stderr": stderr,
                    }
                    results["issues"].append({
                        "node": hostname,
                        "type": "GPU_PROBE_ERROR",
                        "status": GPU_PROBE_ERROR,
                        "observed": None,
                        "expected": expected,
                        "reason": parsed["reason"],
                        "stderr": stderr,
                    })
                    results["all_healthy"] = False
                    continue

                gpu_count = parsed["gpu_count"]
                results["nodes"][hostname] = {
                    "status": GPU_PROBE_OK, "gpu_count": gpu_count,
                    "expected": expected, "reason": None,
                    "binary": parsed["binary"], "stderr": stderr,
                }
                if gpu_count < expected:
                    results["issues"].append({
                        "node": hostname,
                        "type": "GPU_COUNT_MISMATCH",
                        "status": GPU_PROBE_OK,
                        "observed": gpu_count,
                        "expected": expected,
                        "reason": None,
                        "stderr": stderr,
                    })
                    results["all_healthy"] = False
                else:
                    logger.debug(
                        f"[PREFLIGHT] {hostname}: {gpu_count}/{expected} GPUs "
                        f"via {parsed['binary']}")
            except subprocess.TimeoutExpired:
                _unavailable("timeout")
            except Exception as e:
                _unavailable(f"probe_exception:{type(e).__name__}: {e}")

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
    
    def remediate_ramdisk(self, step: int) -> bool:
        """
        Attempt to populate ramdisk by calling ramdisk_preload.sh
        
        Returns:
            True if remediation script ran successfully, False otherwise
        """
        # Look for ramdisk preload script
        script_paths = [
            "ramdisk_preload.sh",
            "./ramdisk_preload.sh",
            "scripts/ramdisk_preload.sh"
        ]
        
        script_path = None
        for path in script_paths:
            if Path(path).exists():
                script_path = path
                break
        
        if not script_path:
            logger.error("[PREFLIGHT] Ramdisk preload script not found")
            return False
        
        try:
            logger.info(f"[PREFLIGHT] Running: bash {script_path} {step}")
            proc = subprocess.run(
                ["bash", script_path, str(step)],
                capture_output=True,
                timeout=RAMDISK_REMEDIATION_TIMEOUT_SECONDS
            )
            
            if proc.returncode == 0:
                logger.info("[PREFLIGHT] Ramdisk preload completed successfully")
                return True
            else:
                stderr = proc.stderr.decode()[:200]
                logger.error(f"[PREFLIGHT] Ramdisk preload failed: {stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"[PREFLIGHT] Ramdisk preload timed out after {RAMDISK_REMEDIATION_TIMEOUT_SECONDS}s")
            return False
        except Exception as e:
            logger.error(f"[PREFLIGHT] Ramdisk preload error: {e}")
            return False
    
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
    parser.add_argument("--no-remediate", action="store_true", 
                        help="Check only - don't attempt to fix issues")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    checker = PreflightChecker(args.config)
    result = checker.check_all(
        args.step, 
        run_id=args.run_id,
        auto_remediate=not args.no_remediate
    )
    
    print("\n" + "=" * 50)
    print(result.summary())
    print("=" * 50)
    
    if args.save:
        checker.save_result(result)
    
    sys.exit(0 if result.passed else 1)
