#!/usr/bin/env python3
"""
Step Executor - Execute pipeline steps as subprocesses
======================================================
Version: 1.2.0
Date: 2026-01-02

Runs step commands, captures output, handles timeouts,
and returns structured StepResult objects.

v1.2.0: Added distributed action handler via coordinator.py
v1.1.0: Added multi-action step support
v1.0.0: Initial release
"""

import subprocess
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from .models import StepResult, StepStatus, ActionResult, ActionConfig, StepManifest, get_step_display_name
from .command_builder import build_command_for_action

logger = logging.getLogger(__name__)

# How much stdout/stderr to capture (last N chars)
OUTPUT_TAIL_SIZE = 2000


# =============================================================================
# STEP EXECUTION
# =============================================================================

def execute_step(
    command: List[str],
    step: int,
    timeout_minutes: int = 240,
    work_dir: Optional[Path] = None,
    params: Optional[Dict[str, Any]] = None,
    env_vars: Optional[Dict[str, str]] = None,
    stream_output: bool = True
) -> StepResult:
    """
    Execute a pipeline step and capture results.
    
    Args:
        command: Command list to execute
        step: Pipeline step number
        timeout_minutes: Maximum execution time
        work_dir: Working directory for execution
        params: Parameters used (for logging/tracking)
        env_vars: Additional environment variables
        stream_output: If True, stream stdout in real-time
    
    Returns:
        StepResult with execution status and details
    """
    work_dir = Path(work_dir) if work_dir else Path.cwd()
    params = params or {}
    step_name = get_step_display_name(step)
    
    # Build environment
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # Ensure real-time output
    if env_vars:
        env.update(env_vars)
    
    # Initialize result
    started_at = datetime.now()
    
    logger.info("=" * 70)
    logger.info(f"EXECUTING STEP {step}: {step_name}")
    logger.info("=" * 70)
    logger.info(f"Command: {' '.join(command[:6])}...")  # Truncate for display
    logger.info(f"Work dir: {work_dir}")
    logger.info(f"Timeout: {timeout_minutes} minutes")
    logger.info("")
    
    try:
        if stream_output:
            result = _execute_streaming(command, work_dir, env, timeout_minutes)
        else:
            result = _execute_capture(command, work_dir, env, timeout_minutes)
        
        exit_code = result["exit_code"]
        stdout = result["stdout"]
        stderr = result["stderr"]
        timed_out = result["timed_out"]
        
    except Exception as e:
        logger.error(f"Execution error: {e}")
        return StepResult(
            step=step,
            step_name=step_name,
            status=StepStatus.FAILED,
            exit_code=-1,
            duration_seconds=int((datetime.now() - started_at).total_seconds()),
            command=command,
            params=params,
            error_message=str(e),
            started_at=started_at,
            completed_at=datetime.now()
        )
    
    # Calculate duration
    completed_at = datetime.now()
    duration_seconds = int((completed_at - started_at).total_seconds())
    
    # Determine status
    if timed_out:
        status = StepStatus.TIMEOUT
        error_message = f"Timed out after {timeout_minutes} minutes"
    elif exit_code == 0:
        status = StepStatus.SUCCESS
        error_message = None
    elif exit_code == 2:
        # Special exit code for degenerate signal (not a failure)
        status = StepStatus.SUCCESS  # Completed successfully, just no signal
        error_message = "Degenerate signal detected"
    else:
        status = StepStatus.FAILED
        error_message = f"Exit code: {exit_code}"
    
    # Build result
    step_result = StepResult(
        step=step,
        step_name=step_name,
        status=status,
        exit_code=exit_code,
        duration_seconds=duration_seconds,
        command=command,
        params=params,
        stdout_tail=stdout[-OUTPUT_TAIL_SIZE:] if stdout else "",
        stderr_tail=stderr[-OUTPUT_TAIL_SIZE:] if stderr else "",
        error_message=error_message,
        started_at=started_at,
        completed_at=completed_at
    )
    
    # Log result
    _log_result(step_result)
    
    return step_result


def _execute_streaming(
    command: List[str],
    work_dir: Path,
    env: Dict[str, str],
    timeout_minutes: int
) -> Dict[str, Any]:
    """
    Execute with real-time output streaming.
    
    Prints stdout as it comes, captures for result.
    """
    stdout_lines = []
    stderr_lines = []
    timed_out = False
    
    try:
        process = subprocess.Popen(
            command,
            cwd=work_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Stream stdout in real-time
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                process.kill()
                timed_out = True
                break
            
            # Read output (non-blocking would be better but this works)
            line = process.stdout.readline()
            if line:
                print(line.rstrip())  # Real-time display
                stdout_lines.append(line)
            
            # Check if process finished
            if process.poll() is not None:
                break
        
        # Capture any remaining output
        remaining_stdout, remaining_stderr = process.communicate(timeout=5)
        if remaining_stdout:
            stdout_lines.append(remaining_stdout)
        if remaining_stderr:
            stderr_lines.append(remaining_stderr)
        
        exit_code = process.returncode
        
    except subprocess.TimeoutExpired:
        process.kill()
        exit_code = -1
        timed_out = True
    
    return {
        "exit_code": exit_code,
        "stdout": "".join(stdout_lines),
        "stderr": "".join(stderr_lines),
        "timed_out": timed_out
    }


def _execute_capture(
    command: List[str],
    work_dir: Path,
    env: Dict[str, str],
    timeout_minutes: int
) -> Dict[str, Any]:
    """
    Execute and capture all output (no streaming).
    
    Simpler but no real-time feedback.
    """
    try:
        result = subprocess.run(
            command,
            cwd=work_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_minutes * 60
        )
        
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": False
        }
        
    except subprocess.TimeoutExpired as e:
        return {
            "exit_code": -1,
            "stdout": e.stdout or "",
            "stderr": e.stderr or "",
            "timed_out": True
        }


def _log_result(result: StepResult) -> None:
    """Log execution result summary."""
    logger.info("")
    logger.info("=" * 70)
    
    if result.status == StepStatus.SUCCESS:
        logger.info(f"✅ STEP {result.step} COMPLETE: {result.step_name}")
    elif result.status == StepStatus.TIMEOUT:
        logger.error(f"⏱️  STEP {result.step} TIMEOUT: {result.step_name}")
    else:
        logger.error(f"❌ STEP {result.step} FAILED: {result.step_name}")
    
    logger.info("=" * 70)
    logger.info(f"Duration: {_format_duration(result.duration_seconds)}")
    logger.info(f"Exit code: {result.exit_code}")
    
    if result.error_message:
        logger.info(f"Error: {result.error_message}")
    
    logger.info("")


def _format_duration(seconds: int) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}m {secs}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"


# =============================================================================
# VALIDATION
# =============================================================================

def check_script_exists(script: str, work_dir: Optional[Path] = None) -> bool:
    """Check if the script file exists."""
    work_dir = Path(work_dir) if work_dir else Path.cwd()
    return (work_dir / script).exists()


def check_python_available(python_executable: str = "python3") -> bool:
    """Check if Python interpreter is available."""
    try:
        result = subprocess.run(
            [python_executable, "--version"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


# =============================================================================
# DISTRIBUTED ACTION EXECUTION
# =============================================================================

def _execute_distributed_action(
    action: ActionConfig,
    params: Dict[str, Any],
    work_dir: Path,
    action_index: int,
    stream_output: bool = True
) -> ActionResult:
    """
    Execute a distributed action via coordinator.py.
    
    Distributed actions run across multiple GPUs (26-GPU cluster).
    The coordinator handles job distribution and result collection.
    
    Args:
        action: ActionConfig with distributed=True
        params: Runtime parameters
        work_dir: Working directory
        action_index: Index of this action in the step
        stream_output: Stream stdout in real-time
    
    Returns:
        ActionResult with job completion stats
    """
    action_start = datetime.now()
    
    # Build coordinator command
    jobs_file = params.get('jobs_file', 'scorer_jobs.json')
    coordinator_config = params.get('coordinator_config', 'ml_coordinator_config.json')
    max_concurrent = params.get('max_concurrent', 26)
    resume_policy = params.get('resume_policy', 'restart')
    
    cmd = [
        'python3', 'coordinator.py',
        '--jobs-file', jobs_file,
        '--config', coordinator_config,
        '--max-concurrent', str(max_concurrent),
        '--resume-policy', resume_policy
    ]
    
    logger.info(f"Executing distributed action via coordinator")
    logger.info(f"Jobs file: {jobs_file}")
    logger.info(f"Max concurrent: {max_concurrent}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Execute coordinator
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    stdout_lines = []
    stderr_lines = []
    jobs_completed = 0
    jobs_failed = 0
    timed_out = False
    
    try:
        process = subprocess.Popen(
            cmd,
            cwd=work_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Stream and parse output
        timeout_seconds = action.timeout_minutes * 60
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                process.kill()
                timed_out = True
                logger.error(f"Distributed action timed out after {action.timeout_minutes}m")
                break
            
            line = process.stdout.readline()
            if line:
                if stream_output:
                    print(line.rstrip())
                stdout_lines.append(line)
                
                # Parse job completion stats from coordinator output
                if "Successful jobs:" in line:
                    try:
                        jobs_completed = int(line.split(":")[-1].strip())
                    except ValueError:
                        pass
                elif "Failed jobs:" in line:
                    try:
                        jobs_failed = int(line.split(":")[-1].strip())
                    except ValueError:
                        pass
                elif "Total jobs executed:" in line:
                    try:
                        total = int(line.split(":")[-1].strip())
                        if jobs_completed == 0:
                            jobs_completed = total
                    except ValueError:
                        pass
            
            if process.poll() is not None:
                break
        
        # Capture remaining output
        remaining_stdout, remaining_stderr = process.communicate(timeout=10)
        if remaining_stdout:
            stdout_lines.append(remaining_stdout)
            if stream_output:
                print(remaining_stdout)
        if remaining_stderr:
            stderr_lines.append(remaining_stderr)
        
        exit_code = process.returncode
        
    except subprocess.TimeoutExpired:
        process.kill()
        exit_code = -1
        timed_out = True
    except Exception as e:
        logger.error(f"Distributed execution error: {e}")
        exit_code = -1
        stderr_lines.append(str(e))
    
    # Calculate duration
    action_duration = int((datetime.now() - action_start).total_seconds())
    
    # Determine success
    success = (exit_code == 0) and (jobs_failed == 0)
    
    error_message = None
    if timed_out:
        error_message = f"Timed out after {action.timeout_minutes}m"
    elif exit_code != 0:
        error_message = f"Coordinator exit code: {exit_code}"
    elif jobs_failed > 0:
        error_message = f"{jobs_failed} jobs failed out of {jobs_completed + jobs_failed}"
    
    logger.info(f"Distributed action complete: {jobs_completed} succeeded, {jobs_failed} failed")
    
    return ActionResult(
        action_index=action_index,
        action_type="run_distributed",
        script=action.script,
        success=success,
        exit_code=exit_code,
        duration_seconds=action_duration,
        command=cmd,
        stdout_tail="".join(stdout_lines)[-OUTPUT_TAIL_SIZE:],
        stderr_tail="".join(stderr_lines)[-OUTPUT_TAIL_SIZE:],
        error_message=error_message,
        distributed=True,
        jobs_completed=jobs_completed,
        jobs_failed=jobs_failed
    )


# =============================================================================
# MULTI-ACTION STEP EXECUTION
# =============================================================================

def execute_multi_action_step(
    manifest: StepManifest,
    params: Dict[str, Any],
    timeout_minutes: Optional[int] = None,
    work_dir: Optional[Path] = None,
    stream_output: bool = True
) -> StepResult:
    """
    Execute all actions in a manifest sequentially.
    
    WATCHER gets full visibility into each action's status.
    Stops on first failure, records which action failed.
    
    Args:
        manifest: StepManifest with actions list
        params: Runtime parameters for all actions
        timeout_minutes: Override timeout (uses action timeout if None)
        work_dir: Working directory
        stream_output: Stream stdout in real-time
    
    Returns:
        StepResult with per-action results in action_results
    """
    work_dir = Path(work_dir) if work_dir else Path.cwd()
    step = manifest.pipeline_step
    step_name = get_step_display_name(step)
    
    # Initialize step result
    started_at = datetime.now()
    action_results = []
    total_actions = len(manifest.actions) if manifest.actions else 1
    
    logger.info("=" * 70)
    logger.info(f"EXECUTING MULTI-ACTION STEP {step}: {step_name}")
    logger.info(f"Total actions: {total_actions}")
    logger.info("=" * 70)
    
    # No actions defined - fall back to single action using script
    if not manifest.actions:
        from .command_builder import build_command
        command = build_command(manifest, params)
        single_result = execute_step(
            command=command,
            step=step,
            timeout_minutes=timeout_minutes or manifest.timeout_minutes,
            work_dir=work_dir,
            params=params,
            stream_output=stream_output
        )
        single_result.total_actions = 1
        return single_result
    
    # Execute each action sequentially
    overall_exit_code = 0
    failed_action_index = None
    total_duration = 0
    
    for i, action in enumerate(manifest.actions):
        action_start = datetime.now()
        
        logger.info("")
        logger.info(f"--- Action {i + 1}/{total_actions}: {action.script} ---")
        logger.info(f"Type: {action.type}")
        logger.info(f"Distributed: {action.distributed}")
        
        # Route distributed actions to coordinator
        if action.distributed:
            logger.info("Routing to distributed handler (coordinator.py)")
            action_result = _execute_distributed_action(
                action=action,
                params=params,
                work_dir=work_dir,
                action_index=i,
                stream_output=stream_output
            )
            action_results.append(action_result)
            total_duration += action_result.duration_seconds
            
            if not action_result.success:
                overall_exit_code = action_result.exit_code
                failed_action_index = i
                break
            continue
        
        # Non-distributed: build command and execute directly
        command = build_command_for_action(action, params)
        logger.info(f"Command: {' '.join(command[:6])}...")
        
        # Use action-specific timeout or override
        action_timeout = timeout_minutes or action.timeout_minutes
        
        # Execute action
        try:
            if stream_output:
                result = _execute_streaming(command, work_dir, os.environ.copy(), action_timeout)
            else:
                result = _execute_capture(command, work_dir, os.environ.copy(), action_timeout)
            
            exit_code = result["exit_code"]
            stdout = result["stdout"]
            stderr = result["stderr"]
            timed_out = result["timed_out"]
            
        except Exception as e:
            exit_code = -1
            stdout = ""
            stderr = str(e)
            timed_out = False
        
        # Calculate action duration
        action_duration = int((datetime.now() - action_start).total_seconds())
        total_duration += action_duration
        
        # Create action result
        action_result = ActionResult(
            action_index=i,
            action_type=action.type,
            script=action.script,
            success=(exit_code == 0),
            exit_code=exit_code,
            duration_seconds=action_duration,
            command=command,
            stdout_tail=stdout[-OUTPUT_TAIL_SIZE:] if stdout else "",
            stderr_tail=stderr[-OUTPUT_TAIL_SIZE:] if stderr else "",
            error_message=f"Timed out after {action_timeout}m" if timed_out else None,
            distributed=action.distributed
        )
        action_results.append(action_result)
        
        # Log action result
        if action_result.success:
            logger.info(f"✅ Action {i + 1} complete ({_format_duration(action_duration)})")
        else:
            logger.error(f"❌ Action {i + 1} failed (exit code: {exit_code})")
            overall_exit_code = exit_code
            failed_action_index = i
            break  # Stop on first failure
    
    # Determine overall status
    completed_at = datetime.now()
    
    if failed_action_index is not None:
        status = StepStatus.FAILED
        error_message = f"Action {failed_action_index + 1} ({manifest.actions[failed_action_index].script}) failed"
    else:
        status = StepStatus.SUCCESS
        error_message = None
    
    # Build final result
    step_result = StepResult(
        step=step,
        step_name=step_name,
        status=status,
        exit_code=overall_exit_code,
        duration_seconds=total_duration,
        params=params,
        error_message=error_message,
        action_results=action_results,
        failed_action_index=failed_action_index,
        total_actions=total_actions,
        started_at=started_at,
        completed_at=completed_at
    )
    
    # Log final summary
    logger.info("")
    logger.info("=" * 70)
    if status == StepStatus.SUCCESS:
        logger.info(f"✅ ALL {total_actions} ACTIONS COMPLETE: {step_name}")
    else:
        logger.error(f"❌ STEP FAILED at action {failed_action_index + 1}/{total_actions}")
    logger.info(f"Total duration: {_format_duration(total_duration)}")
    logger.info("=" * 70)
    
    return step_result


def print_result_summary(result: StepResult) -> None:
    """Print a formatted summary of step execution."""
    print()
    print("=" * 60)
    
    if result.success:
        print(f"✅ STEP {result.step}: {result.step_name}")
    else:
        print(f"❌ STEP {result.step}: {result.step_name}")
    
    print("=" * 60)
    print(f"  Status:   {result.status.value}")
    print(f"  Duration: {_format_duration(result.duration_seconds)}")
    print(f"  Exit Code: {result.exit_code}")
    
    if result.error_message:
        print(f"  Error: {result.error_message}")
    
    # Multi-action summary
    if result.is_multi_action:
        print()
        print(f"  Actions: {result.actions_completed}/{result.total_actions} completed")
        for ar in result.action_results:
            icon = "✅" if ar.success else "❌"
            dist_info = ""
            if ar.distributed:
                dist_info = f" [distributed: {ar.jobs_completed} ok, {ar.jobs_failed} failed]"
            print(f"    {icon} [{ar.action_index + 1}] {ar.script} ({_format_duration(ar.duration_seconds)}){dist_info}")
    
    # Metrics
    if result.metrics:
        print()
        print("  Metrics:")
        for key, value in result.metrics.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            elif isinstance(value, int) and value > 1000:
                print(f"    {key}: {value:,}")
            else:
                print(f"    {key}: {value}")
    
    # Outputs
    if result.outputs_found:
        print()
        print("  Outputs:")
        for output, found in result.outputs_found.items():
            icon = "✅" if found else "❌"
            print(f"    {icon} {output}")
    
    print("=" * 60)
    print()
