# job_status_writer.py
# Version: 1.0.0
# Created: December 16, 2025
# Purpose: Worker-side job status and result file writing
#
# This module is used by workers to report their state.
# JobStateMonitor reads these files to track job progress.

"""
Job Status Writer

Workers use this to:
1. Signal job started (creates status.json)
2. Update progress periodically (updates status.json with heartbeat)
3. Report completion (creates result.json, final status.json update)

The heartbeat mechanism allows external monitoring (JobStateMonitor, Watcher)
to detect stalled or crashed jobs.
"""

import json
import os
import socket
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any

from modules.job_file_contract import JobFileContract


class JobStatusWriter:
    """
    Writes job status and result files for external monitoring.
    
    Used by any worker script:
    - full_scoring_worker.py
    - scorer_trial_worker.py
    - distributed_worker.py (for script jobs)
    
    Usage:
        writer = JobStatusWriter(job_id, work_dir="/path/to/work")
        writer.start(gpu_id=0)
        
        for i, item in enumerate(items):
            if i % 100 == 0:
                writer.update_progress(i / len(items), f"Processing {i}/{len(items)}")
            process(item)
        
        writer.complete(success=True, output_file="results.json", 
                       stats={"items_processed": len(items)})
    """
    
    # Heartbeat interval in seconds
    HEARTBEAT_INTERVAL = 10
    
    def __init__(self, job_id: str, work_dir: str = "."):
        """
        Initialize status writer.
        
        Args:
            job_id: Unique job identifier
            work_dir: Directory to write status/result files
        """
        self.job_id = job_id
        self.work_dir = work_dir
        
        # File paths (using canonical contract)
        self.status_file = os.path.join(work_dir, JobFileContract.status_file(job_id))
        self.result_file = os.path.join(work_dir, JobFileContract.result_file(job_id))
        
        # State
        self.state = "pending"
        self.progress = 0.0
        self.message = ""
        self.gpu_id = 0
        self.extra_metadata: Dict[str, Any] = {}
        self._started_at: Optional[str] = None
        
        # Background heartbeat thread
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def start(self, gpu_id: int = 0, extra_metadata: Optional[Dict[str, Any]] = None):
        """
        Signal job started, begin heartbeat thread.
        
        Args:
            gpu_id: GPU device ID this job is running on
            extra_metadata: Optional additional metadata to include in status
        """
        self.gpu_id = gpu_id
        self.extra_metadata = extra_metadata or {}
        self._started_at = datetime.utcnow().isoformat() + "Z"
        self.state = "running"
        
        # Write initial status
        self._write_status()
        
        # Start heartbeat thread
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
    
    def update_progress(self, progress: float, message: str = ""):
        """
        Update progress (called by main worker loop).
        
        Args:
            progress: Progress value 0.0 - 1.0
            message: Optional status message
        """
        with self._lock:
            self.progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]
            self.message = message
    
    def complete(self, success: bool, output_file: str = "", 
                 stats: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        """
        Signal job completed, write result file, stop heartbeat.
        
        Args:
            success: True if job succeeded
            output_file: Path to output file (if any)
            stats: Optional statistics dictionary
            error: Error message (if failed)
        """
        # Stop heartbeat thread
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        
        # Update state
        with self._lock:
            self.state = "completed" if success else "failed"
            self.progress = 1.0 if success else self.progress
            if error:
                self.message = error
        
        # Write final status and result
        self._write_status()
        self._write_result(success, output_file, stats or {}, error)
    
    def fail(self, error: str, stats: Optional[Dict[str, Any]] = None):
        """
        Convenience method for failure.
        
        Args:
            error: Error message
            stats: Optional statistics gathered before failure
        """
        self.complete(success=False, output_file="", stats=stats, error=error)
    
    def _heartbeat_loop(self):
        """Background thread that updates status file periodically."""
        while not self._stop_event.wait(timeout=self.HEARTBEAT_INTERVAL):
            self._write_status()
    
    def _write_status(self):
        """Write current status to file (thread-safe)."""
        with self._lock:
            status = {
                "job_id": self.job_id,
                "state": self.state,
                "progress": self.progress,
                "heartbeat": datetime.utcnow().isoformat() + "Z",
                "pid": os.getpid(),
                "gpu_id": self.gpu_id,
                "hostname": socket.gethostname(),
                "started_at": self._started_at,
                "message": self.message,
                "metadata": self.extra_metadata
            }
        
        try:
            # Write atomically (write to temp, then rename)
            temp_file = self.status_file + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(status, f, indent=2)
            os.replace(temp_file, self.status_file)
        except Exception as e:
            # Don't crash the worker if status write fails
            print(f"[JobStatusWriter] Warning: Failed to write status: {e}")
    
    def _write_result(self, success: bool, output_file: str, 
                      stats: Dict[str, Any], error: Optional[str]):
        """Write final result to file."""
        result = {
            "job_id": self.job_id,
            "success": success,
            "state": "completed" if success else "failed",
            "output_file": output_file,
            "error": error,
            "stats": stats,
            "started_at": self._started_at,
            "completed_at": datetime.utcnow().isoformat() + "Z"
        }
        
        try:
            # Write atomically
            temp_file = self.result_file + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(result, f, indent=2)
            os.replace(temp_file, self.result_file)
        except Exception as e:
            print(f"[JobStatusWriter] Warning: Failed to write result: {e}")


# Self-test when run directly
if __name__ == "__main__":
    import tempfile
    import shutil
    
    print("Testing JobStatusWriter...")
    
    # Create temp directory
    test_dir = tempfile.mkdtemp()
    
    try:
        # Create writer
        writer = JobStatusWriter("test_job_001", test_dir)
        
        # Start
        writer.start(gpu_id=0, extra_metadata={"test": True})
        print(f"  Started, status file: {writer.status_file}")
        
        # Verify status file exists
        assert os.path.exists(writer.status_file), "Status file not created"
        
        # Check initial status
        with open(writer.status_file) as f:
            status = json.load(f)
        assert status["state"] == "running", f"Expected 'running', got '{status['state']}'"
        assert status["job_id"] == "test_job_001"
        print(f"  Initial status: state={status['state']}, progress={status['progress']}")
        
        # Update progress
        writer.update_progress(0.5, "Halfway there")
        time.sleep(0.5)  # Let heartbeat write
        
        # Wait for heartbeat
        time.sleep(writer.HEARTBEAT_INTERVAL + 1)
        
        # Check heartbeat updated
        with open(writer.status_file) as f:
            status = json.load(f)
        assert status["progress"] == 0.5
        assert status["message"] == "Halfway there"
        print(f"  After update: progress={status['progress']}, message='{status['message']}'")
        
        # Complete
        writer.complete(True, "output.json", {"items": 100})
        
        # Verify result file
        assert os.path.exists(writer.result_file), "Result file not created"
        
        with open(writer.result_file) as f:
            result = json.load(f)
        assert result["success"] == True
        assert result["output_file"] == "output.json"
        assert result["stats"]["items"] == 100
        print(f"  Result: success={result['success']}, output={result['output_file']}")
        
        print("\n✓ JobStatusWriter OK")
        
    finally:
        # Cleanup
        shutil.rmtree(test_dir)
