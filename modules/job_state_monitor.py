# job_state_monitor.py
# Version: 1.0.0
# Created: December 16, 2025
# Purpose: Stateless job state observation via file inspection
#
# DESIGN CONTRACT:
# - This class is STATELESS (no instance variables tracking job history)
# - This class is READ-ONLY (never writes files, never kills processes)
# - This class is PASSIVE (only observes, never decides)
#
# FORBIDDEN in this class:
# - Retry decisions
# - Node selection
# - Heuristics or "smart" behavior
# - Any method that modifies external state
#
# If you need those behaviors, they belong in:
# - Coordinator (interim)
# - Watcher Agent (future)
# - NOT HERE

"""
Job State Monitor

This module observes job state by reading status/result files.
It is used by:
- Coordinator (interim self-monitoring)
- Watcher Agent (future intelligent monitoring)
- Web Dashboard (status display)
- CLI tools (debugging)
- AI/ML systems (autonomous operation)

All methods return data, none take action.
"""

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Callable, Any

from modules.job_file_contract import JobFileContract


class JobState(Enum):
    """Possible job states derived from file inspection."""
    UNKNOWN = "unknown"           # No files found yet
    PENDING = "pending"           # Input file exists, no status yet
    STARTING = "starting"         # Status file exists, state=pending
    RUNNING = "running"           # Status file exists, state=running, heartbeat fresh
    STALLED = "stalled"           # Status file exists, heartbeat stale
    COMPLETED = "completed"       # Result file exists, success=true
    FAILED = "failed"             # Result file exists, success=false
    STARTUP_FAILED = "startup_failed"  # No status file after timeout
    TIMED_OUT = "timed_out"       # Job did not complete within timeout (classification, not action)


@dataclass
class JobSnapshot:
    """Point-in-time snapshot of a job's state."""
    job_id: str
    state: JobState
    progress: float
    message: str
    heartbeat_age_seconds: float
    elapsed_seconds: float
    worker_hostname: str
    gpu_id: int
    error: Optional[str]
    output_file: Optional[str]
    stats: Optional[Dict[str, Any]]
    raw_status: Optional[Dict[str, Any]]
    raw_result: Optional[Dict[str, Any]]


class JobStateMonitor:
    """
    Monitors job state via file-based inspection.
    
    This module is intentionally stateless and side-effect free.
    It only READS files and REPORTS state. It does NOT:
    - Make retry decisions (that's the caller's job)
    - Kill processes (that's the caller's job)
    - Modify any files (that's the worker's job)
    
    This design allows ANY component to use it:
    - Coordinator (interim self-monitoring)
    - Watcher Agent (intelligent monitoring)
    - CLI tools (debugging)
    - Web dashboard (status display)
    - AI/ML systems (autonomous operation)
    """
    
    def __init__(self,
                 ssh_pool=None,
                 startup_timeout: int = 60,
                 heartbeat_stale_threshold: int = 30,
                 poll_timeout: int = 5,
                 logger=None):
        """
        Initialize monitor.
        
        Args:
            ssh_pool: SSH connection pool for remote file access (None for local-only)
            startup_timeout: Seconds to wait for status file before declaring startup failure
            heartbeat_stale_threshold: Seconds of no heartbeat before declaring stall
            poll_timeout: Timeout for individual file read operations
            logger: Optional logger instance
        """
        self.ssh_pool = ssh_pool
        self.startup_timeout = startup_timeout
        self.heartbeat_stale_threshold = heartbeat_stale_threshold
        self.poll_timeout = poll_timeout
        self.logger = logger
        
        # Validate timing discipline (Team Beta requirement)
        # heartbeat_stale_threshold should be >= 2x heartbeat interval (10s)
        assert heartbeat_stale_threshold >= 20, \
            f"heartbeat_stale_threshold ({heartbeat_stale_threshold}) must be >= 20s (2x heartbeat interval)"
    
    def _log(self, level: str, message: str):
        """Log message if logger available."""
        if self.logger:
            getattr(self.logger, level)(message)
    
    def _read_remote_file(self, node: Dict, filename: str) -> Optional[str]:
        """Read a file from a remote node via SSH."""
        if not self.ssh_pool:
            return None
        
        hostname = node.get('hostname', '')
        username = node.get('username', 'michael')
        password = node.get('password')
        script_path = node.get('script_path', '')
        
        # Handle localhost
        if hostname in ('localhost', '127.0.0.1', ''):
            filepath = os.path.join(script_path, filename)
            try:
                with open(filepath, 'r') as f:
                    return f.read()
            except FileNotFoundError:
                return None
            except Exception:
                return None
        
        # Remote node
        try:
            ssh = self.ssh_pool.get_connection(hostname, username, password)
            filepath = os.path.join(script_path, filename)
            
            _, stdout, _ = ssh.exec_command(f"cat '{filepath}' 2>/dev/null")
            stdout.channel.settimeout(self.poll_timeout)
            content = stdout.read().decode(errors='ignore').strip()
            
            self.ssh_pool.return_connection(hostname, ssh)
            
            return content if content else None
        except Exception as e:
            self._log('debug', f"Failed to read {filename} from {hostname}: {e}")
            return None
    
    def _read_local_file(self, filepath: str) -> Optional[str]:
        """Read a local file."""
        try:
            with open(filepath, 'r') as f:
                return f.read()
        except FileNotFoundError:
            return None
        except Exception:
            return None
    
    def _parse_heartbeat_age(self, heartbeat_str: Optional[str]) -> float:
        """Parse heartbeat timestamp and return age in seconds."""
        if not heartbeat_str:
            return float('inf')
        
        try:
            # Parse ISO format timestamp
            heartbeat_str = heartbeat_str.replace('Z', '+00:00')
            heartbeat_dt = datetime.fromisoformat(heartbeat_str)
            
            # Ensure timezone aware
            if heartbeat_dt.tzinfo is None:
                heartbeat_dt = heartbeat_dt.replace(tzinfo=timezone.utc)
            
            now = datetime.now(timezone.utc)
            age = (now - heartbeat_dt).total_seconds()
            return max(0, age)
        except Exception:
            return float('inf')
    
    def get_job_state(self, job_id: str, node: Dict, launched_at: float = 0) -> JobSnapshot:
        """
        Get current state of a single job.
        
        Args:
            job_id: The job identifier
            node: Dict with 'hostname', 'username', 'script_path' (and optionally 'password')
            launched_at: Unix timestamp when job was launched (for elapsed time calculation)
            
        Returns:
            JobSnapshot with current state
        """
        hostname = node.get('hostname', 'localhost')
        script_path = node.get('script_path', '.')
        
        # Read status and result files
        status_filename = JobFileContract.status_file(job_id)
        result_filename = JobFileContract.result_file(job_id)
        
        if hostname in ('localhost', '127.0.0.1', ''):
            status_content = self._read_local_file(os.path.join(script_path, status_filename))
            result_content = self._read_local_file(os.path.join(script_path, result_filename))
        else:
            status_content = self._read_remote_file(node, status_filename)
            result_content = self._read_remote_file(node, result_filename)
        
        # Parse JSON
        status_data = None
        result_data = None
        
        if status_content:
            try:
                status_data = json.loads(status_content)
            except json.JSONDecodeError:
                pass
        
        if result_content:
            try:
                result_data = json.loads(result_content)
            except json.JSONDecodeError:
                pass
        
        # Calculate elapsed time
        elapsed = time.time() - launched_at if launched_at > 0 else 0
        
        # Determine state based on files
        state = JobState.UNKNOWN
        progress = 0.0
        message = ""
        heartbeat_age = float('inf')
        error = None
        output_file = None
        stats = None
        worker_hostname = hostname
        gpu_id = 0
        
        # Check result file first (terminal state)
        if result_data:
            success = result_data.get('success', False)
            state = JobState.COMPLETED if success else JobState.FAILED
            progress = 1.0 if success else result_data.get('progress', 0.0)
            message = result_data.get('message', '')
            error = result_data.get('error')
            output_file = result_data.get('output_file')
            stats = result_data.get('stats')
            heartbeat_age = 0  # Job is done
        
        # Check status file
        elif status_data:
            heartbeat_str = status_data.get('heartbeat')
            heartbeat_age = self._parse_heartbeat_age(heartbeat_str)
            
            progress = status_data.get('progress', 0.0)
            message = status_data.get('message', '')
            worker_hostname = status_data.get('hostname', hostname)
            gpu_id = status_data.get('gpu_id', 0)
            
            file_state = status_data.get('state', 'unknown')
            
            if file_state == 'completed':
                state = JobState.COMPLETED
            elif file_state == 'failed':
                state = JobState.FAILED
                error = message
            elif heartbeat_age > self.heartbeat_stale_threshold:
                state = JobState.STALLED
                self._log('debug', f"[state-change] {job_id}: running → stalled (heartbeat {heartbeat_age:.0f}s old)")
            elif file_state == 'running':
                state = JobState.RUNNING
            elif file_state == 'pending':
                state = JobState.STARTING
            else:
                state = JobState.RUNNING  # Default to running if status file exists
        
        # No status file yet
        else:
            if elapsed > self.startup_timeout:
                state = JobState.STARTUP_FAILED
                error = f"No status file after {elapsed:.0f}s (threshold: {self.startup_timeout}s)"
                self._log('warning', f"[startup-fail] {job_id}: {error}")
            else:
                state = JobState.PENDING
        
        return JobSnapshot(
            job_id=job_id,
            state=state,
            progress=progress,
            message=message,
            heartbeat_age_seconds=heartbeat_age,
            elapsed_seconds=elapsed,
            worker_hostname=worker_hostname,
            gpu_id=gpu_id,
            error=error,
            output_file=output_file,
            stats=stats,
            raw_status=status_data,
            raw_result=result_data
        )
    
    def poll_jobs(self, jobs: List[Dict]) -> Dict[str, JobSnapshot]:
        """
        Poll state of multiple jobs.
        
        Args:
            jobs: List of dicts, each with 'job_id', 'node', 'launched_at'
            
        Returns:
            Dict mapping job_id to JobSnapshot
        """
        snapshots = {}
        
        for job_info in jobs:
            job_id = job_info.get('job_id', '')
            node = job_info.get('node', {})
            launched_at = job_info.get('launched_at', 0)
            
            snapshot = self.get_job_state(job_id, node, launched_at)
            snapshots[job_id] = snapshot
        
        return snapshots
    
    def wait_for_completion(self,
                            jobs: List[Dict],
                            timeout: int = 7200,
                            poll_interval: int = 10,
                            callback: Optional[Callable[[Dict[str, JobSnapshot]], None]] = None
                            ) -> Dict[str, JobSnapshot]:
        """
        Poll jobs until all complete or timeout.
        
        DESIGN NOTE: This method is intentionally THIN.
        - It is just a loop over poll_jobs()
        - Timeout is returned as a classification (TIMED_OUT state), NOT an action
        - No retry logic, no decisions, no side effects
        - Watcher can implement different polling cadence by calling poll_jobs() directly
        
        Args:
            jobs: List of job definitions with 'job_id', 'node', 'launched_at'
            timeout: Maximum wait time in seconds
            poll_interval: Seconds between polls
            callback: Optional function called after each poll with current snapshots
            
        Returns:
            Final snapshots for all jobs (caller decides what to do with them)
        """
        start_time = time.time()
        terminal_states = {JobState.COMPLETED, JobState.FAILED, JobState.STARTUP_FAILED}
        
        while (time.time() - start_time) < timeout:
            # Pure poll - no decisions
            snapshots = self.poll_jobs(jobs)
            
            # Callback for external progress tracking (optional)
            if callback:
                callback(snapshots)
            
            # Check if all jobs reached terminal state
            all_terminal = all(s.state in terminal_states for s in snapshots.values())
            
            if all_terminal:
                return snapshots
            
            time.sleep(poll_interval)
        
        # Timeout reached - mark remaining non-terminal jobs as TIMED_OUT
        # This is a CLASSIFICATION, not an action (we don't kill anything)
        final_snapshots = self.poll_jobs(jobs)
        
        for job_id, snapshot in list(final_snapshots.items()):
            if snapshot.state not in terminal_states:
                # Find launched_at for this job
                launched_at = 0
                for job_info in jobs:
                    if job_info.get('job_id') == job_id:
                        launched_at = job_info.get('launched_at', 0)
                        break
                
                # Create new snapshot with TIMED_OUT state
                final_snapshots[job_id] = JobSnapshot(
                    job_id=snapshot.job_id,
                    state=JobState.TIMED_OUT,
                    progress=snapshot.progress,
                    message=f"Timed out after {timeout}s",
                    heartbeat_age_seconds=snapshot.heartbeat_age_seconds,
                    elapsed_seconds=time.time() - launched_at if launched_at > 0 else timeout,
                    worker_hostname=snapshot.worker_hostname,
                    gpu_id=snapshot.gpu_id,
                    error=f"Job did not complete within {timeout}s timeout",
                    output_file=None,
                    stats=None,
                    raw_status=snapshot.raw_status,
                    raw_result=None
                )
        
        return final_snapshots
    
    def classify_outcomes(self, snapshots: Dict[str, JobSnapshot]) -> Dict[str, List[str]]:
        """
        Classify job outcomes for reporting.
        
        Args:
            snapshots: Dict mapping job_id to JobSnapshot
            
        Returns:
            Dict with lists of job_ids by outcome category:
            {
                'completed': [...],
                'failed': [...],
                'stalled': [...],
                'startup_failed': [...],
                'timed_out': [...],
                'still_running': [...]
            }
        """
        outcomes = {
            'completed': [],
            'failed': [],
            'stalled': [],
            'startup_failed': [],
            'timed_out': [],
            'still_running': []
        }
        
        for job_id, snapshot in snapshots.items():
            if snapshot.state == JobState.COMPLETED:
                outcomes['completed'].append(job_id)
            elif snapshot.state == JobState.FAILED:
                outcomes['failed'].append(job_id)
            elif snapshot.state == JobState.STALLED:
                outcomes['stalled'].append(job_id)
            elif snapshot.state == JobState.STARTUP_FAILED:
                outcomes['startup_failed'].append(job_id)
            elif snapshot.state == JobState.TIMED_OUT:
                outcomes['timed_out'].append(job_id)
            elif snapshot.state in (JobState.RUNNING, JobState.STARTING, JobState.PENDING):
                outcomes['still_running'].append(job_id)
        
        return outcomes


# Self-test when run directly
if __name__ == "__main__":
    import tempfile
    import shutil
    
    print("Testing JobStateMonitor...")
    
    # Create temp directory
    test_dir = tempfile.mkdtemp()
    
    try:
        # Create a mock status file
        status_file = os.path.join(test_dir, JobFileContract.status_file("test_001"))
        status_data = {
            "job_id": "test_001",
            "state": "running",
            "progress": 0.5,
            "heartbeat": datetime.utcnow().isoformat() + "Z",
            "pid": 12345,
            "gpu_id": 0,
            "hostname": "test-host",
            "started_at": datetime.utcnow().isoformat() + "Z",
            "message": "Processing..."
        }
        with open(status_file, 'w') as f:
            json.dump(status_data, f)
        
        # Create monitor (local only, no SSH)
        monitor = JobStateMonitor(
            ssh_pool=None,
            startup_timeout=60,
            heartbeat_stale_threshold=30
        )
        
        # Test get_job_state
        snapshot = monitor.get_job_state(
            "test_001",
            {"hostname": "localhost", "script_path": test_dir},
            launched_at=time.time() - 10
        )
        
        print(f"  Job: {snapshot.job_id}")
        print(f"  State: {snapshot.state.value}")
        print(f"  Progress: {snapshot.progress:.0%}")
        print(f"  Heartbeat age: {snapshot.heartbeat_age_seconds:.1f}s")
        
        assert snapshot.state == JobState.RUNNING
        assert snapshot.progress == 0.5
        
        # Test with result file (completed)
        result_file = os.path.join(test_dir, JobFileContract.result_file("test_002"))
        result_data = {
            "job_id": "test_002",
            "success": True,
            "output_file": "output.json",
            "stats": {"items": 100}
        }
        with open(result_file, 'w') as f:
            json.dump(result_data, f)
        
        snapshot2 = monitor.get_job_state(
            "test_002",
            {"hostname": "localhost", "script_path": test_dir},
            launched_at=time.time() - 60
        )
        
        print(f"\n  Job: {snapshot2.job_id}")
        print(f"  State: {snapshot2.state.value}")
        print(f"  Output: {snapshot2.output_file}")
        
        assert snapshot2.state == JobState.COMPLETED
        assert snapshot2.output_file == "output.json"
        
        # Test classify_outcomes
        snapshots = {"test_001": snapshot, "test_002": snapshot2}
        outcomes = monitor.classify_outcomes(snapshots)
        
        print(f"\n  Outcomes: {outcomes}")
        
        assert "test_001" in outcomes['still_running']
        assert "test_002" in outcomes['completed']
        
        print("\n✓ JobStateMonitor OK")
        
    finally:
        shutil.rmtree(test_dir)
