#!/usr/bin/env python3
"""
Recovery and Fault Tolerance Module
Manages automatic job recovery, progress persistence, and retry logic
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Optional
from dataclasses import asdict
from cluster_models import ProgressState

class AutoRecoveryManager:
    """Manages automatic job recovery and progress persistence"""
    def __init__(self, recovery_dir: str = ".recovery"):
        self.recovery_dir = Path(recovery_dir)
        self.recovery_dir.mkdir(exist_ok=True)
        self.max_retries = 3
        self.retry_delay = 5 # seconds
        self.base_retry_delay = 5
        
    def calculate_retry_delay(self, attempt: int) -> int:
        """Exponential backoff, capped at 5 minutes."""
        try:
            return min(300, self.base_retry_delay * (2 ** max(0, attempt - 1)))
        except Exception:
            return 30
            
    def generate_analysis_id(self, params: Dict) -> str:
        """Generate unique analysis ID based on parameters"""
        params_str = json.dumps(params, sort_keys=True)
        hash_input = params_str.encode()
        analysis_hash = hashlib.md5(hash_input).hexdigest()[:12]
        return f"analysis_{analysis_hash}"
        
    def get_progress_file(self, analysis_id: str) -> Path:
        """Get progress file path for analysis"""
        return self.recovery_dir / f"{analysis_id}.progress"
        
    def save_progress(self, state: ProgressState) -> bool:
        """Save current progress state"""
        try:
            progress_file = self.get_progress_file(state.analysis_id)
            import time
            state.last_update = time.time()
            # Convert to dict for JSON serialization
            state_dict = asdict(state)
            # Atomic write using temporary file
            temp_file = progress_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(state_dict, f, indent=2)
            temp_file.rename(progress_file)
            return True
        except Exception as e:
            print(f"Warning: Failed to save progress: {e}")
            return False
            
    def load_progress(self, analysis_id: str) -> Optional[ProgressState]:
        """Load previous progress state"""
        try:
            progress_file = self.get_progress_file(analysis_id)
            if not progress_file.exists():
                return None
            with open(progress_file, 'r') as f:
                state_dict = json.load(f)
            return ProgressState(**state_dict)
        except Exception as e:
            print(f"Warning: Failed to load progress: {e}")
            return None
            
    def cleanup_progress(self, analysis_id: str) -> bool:
        """Remove progress file after successful completion"""
        try:
            progress_file = self.get_progress_file(analysis_id)
            if progress_file.exists():
                progress_file.unlink()
                return True
        except Exception:
            pass
        return False
        
    def should_retry_job(self, job_id: str, retry_count: Dict[str, int]) -> bool:
        """Determine if job should be retried"""
        current_retries = retry_count.get(job_id, 0)
        return current_retries < self.max_retries
