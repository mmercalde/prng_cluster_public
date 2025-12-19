# job_file_contract.py
# Version: 1.0.0
# Created: December 16, 2025
# Purpose: Canonical file naming contract for job state files
#
# This module is the SINGLE SOURCE OF TRUTH for job file naming.
# All components (coordinator, workers, monitor) MUST import this.
# DO NOT hardcode job file names anywhere else.

"""
Canonical Job File Contract

All job state is communicated via three files per job:
- input.json:  Job definition (written by coordinator)
- status.json: Heartbeat/progress (written by worker)  
- result.json: Final outcome (written by worker)

This contract ensures:
- Coordinator knows where to write input
- Workers know where to write status/results
- Monitor knows where to read all three
- Watcher (future) uses the same contract
"""

from typing import List


class JobFileContract:
    """
    Canonical file naming - import this, don't hardcode names.
    
    Usage:
        from modules.job_file_contract import JobFileContract
        
        input_file = JobFileContract.input_file(job_id)
        status_file = JobFileContract.status_file(job_id)
        result_file = JobFileContract.result_file(job_id)
    """
    
    @staticmethod
    def input_file(job_id: str) -> str:
        """
        Job input/definition file (written by coordinator).
        Contains: script, args, timeout, expected_output, etc.
        """
        return f"job_{job_id}.input.json"
    
    @staticmethod
    def status_file(job_id: str) -> str:
        """
        Job status/heartbeat file (written by worker).
        Contains: state, progress, heartbeat timestamp, pid, gpu_id, message.
        Updated every 10 seconds while job is running.
        """
        return f"job_{job_id}.status.json"
    
    @staticmethod
    def result_file(job_id: str) -> str:
        """
        Job result file (written by worker on completion).
        Contains: success, output_file, stats, error, timestamps.
        Written once when job completes (success or failure).
        """
        return f"job_{job_id}.result.json"
    
    @staticmethod
    def all_files(job_id: str) -> List[str]:
        """
        All job files (for cleanup operations).
        
        Usage:
            for f in JobFileContract.all_files(job_id):
                os.remove(os.path.join(work_dir, f))
        """
        return [
            JobFileContract.input_file(job_id),
            JobFileContract.status_file(job_id),
            JobFileContract.result_file(job_id),
        ]
    
    @staticmethod
    def parse_job_id_from_filename(filename: str) -> str:
        """
        Extract job_id from a job file name.
        
        Args:
            filename: e.g., "job_full_scoring_0003.status.json"
            
        Returns:
            job_id: e.g., "full_scoring_0003"
        """
        # Remove "job_" prefix and file extension
        if filename.startswith("job_"):
            filename = filename[4:]  # Remove "job_"
        
        # Remove known suffixes
        for suffix in [".input.json", ".status.json", ".result.json"]:
            if filename.endswith(suffix):
                return filename[:-len(suffix)]
        
        return filename


# Self-test when run directly
if __name__ == "__main__":
    job_id = "full_scoring_0003"
    
    print(f"Job ID: {job_id}")
    print(f"  Input:  {JobFileContract.input_file(job_id)}")
    print(f"  Status: {JobFileContract.status_file(job_id)}")
    print(f"  Result: {JobFileContract.result_file(job_id)}")
    print(f"  All:    {JobFileContract.all_files(job_id)}")
    
    # Test parsing
    test_files = [
        "job_full_scoring_0003.input.json",
        "job_full_scoring_0003.status.json", 
        "job_full_scoring_0003.result.json",
    ]
    for f in test_files:
        parsed = JobFileContract.parse_job_id_from_filename(f)
        print(f"  Parse '{f}' → '{parsed}'")
    
    print("\n✓ JobFileContract OK")
