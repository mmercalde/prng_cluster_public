#!/usr/bin/env python3
"""
Step 2 Execution Manager - Distributed Job Lifecycle Controller
================================================================
Version: 1.1.0
Date: 2026-01-03

FIX v1.1.0: Run coordinator ONCE, use collect_scorer_results() for collection,
            retry ONLY missing jobs (not full re-dispatch).

Owns the COMPLETE Step 2 lifecycle:
  1. Generate jobs
  2. Dispatch via coordinator (ONCE)
  3. Collect results via coordinator.collect_scorer_results()
  4. Identify missing jobs
  5. Retry ONLY missing jobs (generate new job file)
  6. Emit step2_results.json
"""

import json
import subprocess
import sys
import os
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 30
RESULTS_DIR = Path("scorer_trial_results")


@dataclass
class JobState:
    job_id: str
    trial_id: int
    status: str = "PENDING"
    retry_count: int = 0
    result_file: Optional[str] = None


@dataclass 
class Step2Result:
    step: int = 2
    total_jobs: int = 0
    completed: int = 0
    failed: int = 0
    retried: int = 0
    results_path: str = "scorer_trial_results/"
    worker_stats: Dict[str, int] = field(default_factory=dict)
    job_states: List[Dict] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: int = 0
    success: bool = False


class Step2ExecutionManager:
    def __init__(self, coordinator_config="ml_coordinator_config.json", max_concurrent=26):
        self.coordinator_config = coordinator_config
        self.max_concurrent = max_concurrent
        self.job_states: Dict[str, JobState] = {}
        self.result = Step2Result()
        self.study_name = ""
        self.study_db = ""
        
    def run(self, survivors_file, train_history, holdout_history, study_name, study_db, num_trials, sample_size=25000):
        self.result.started_at = datetime.now().isoformat()
        self.result.total_jobs = num_trials
        self.study_name = study_name
        self.study_db = study_db
        
        print("=" * 70)
        print("STEP 2 EXECUTION MANAGER v1.1")
        print("=" * 70)
        print(f"Trials: {num_trials}, Study: {study_name}")
        
        try:
            # Phase 1: Generate ALL jobs
            print("\n--- Phase 1: Generate Jobs ---")
            jobs_file = "scorer_jobs.json"
            if not self._generate_jobs(survivors_file, train_history, holdout_history,
                                       study_name, study_db, num_trials, sample_size, jobs_file):
                raise RuntimeError("Job generation failed")
            
            # Phase 2: Push code to remotes
            print("\n--- Phase 2: Push Code to Remotes ---")
            self._push_code_to_remotes()
            
            # Phase 3: Push data to remotes
            print("\n--- Phase 3: Push Data to Remotes ---")
            self._push_data_to_remotes(survivors_file, train_history, holdout_history, jobs_file)
            
            # Phase 4: Dispatch jobs (ONCE)
            print("\n--- Phase 4: Dispatch Jobs ---")
            self._dispatch_jobs(jobs_file)
            
            # Phase 5: Collect results using coordinator's method
            print("\n--- Phase 5: Collect Results ---")
            self._collect_results_via_coordinator()
            
            # Phase 6: Check for missing, retry if needed
            retry_round = 0
            while retry_round < MAX_RETRIES:
                missing = self._get_missing_job_ids()
                if not missing:
                    print("\n✅ All jobs completed!")
                    break
                    
                retry_round += 1
                print(f"\n--- Retry Round {retry_round}/{MAX_RETRIES}: {len(missing)} missing jobs ---")
                self.result.retried += len(missing)
                
                # Generate retry jobs file with ONLY missing jobs
                retry_file = f"scorer_jobs_retry_{retry_round}.json"
                self._generate_retry_jobs(missing, survivors_file, train_history, holdout_history,
                                          study_name, study_db, sample_size, retry_file)
                
                # Push retry file to remotes
                self._push_retry_file(retry_file)
                
                # Wait before retry
                print(f"Waiting {RETRY_DELAY_SECONDS}s before retry...")
                time.sleep(RETRY_DELAY_SECONDS)
                
                # Dispatch retry jobs
                self._dispatch_jobs(retry_file)
                
                # Collect again
                self._collect_results_via_coordinator()
            
            # Final validation
            print("\n--- Phase 7: Validate Completion ---")
            self._validate_completion()
            
            self.result.success = (self.result.completed == self.result.total_jobs)
            
        except Exception as e:
            print(f"\n❌ Step 2 failed: {e}")
            import traceback
            traceback.print_exc()
            self.result.success = False
            
        finally:
            self.result.completed_at = datetime.now().isoformat()
            started = datetime.fromisoformat(self.result.started_at)
            completed = datetime.fromisoformat(self.result.completed_at)
            self.result.duration_seconds = int((completed - started).total_seconds())
            self._save_result_manifest()
            
        return self.result
    
    def _generate_jobs(self, survivors_file, train_history, holdout_history,
                       study_name, study_db, num_trials, sample_size, output_file):
        cmd = ["python3", "generate_scorer_jobs.py",
               "--trials", str(num_trials),
               "--survivors", survivors_file,
               "--train-history", train_history,
               "--holdout-history", holdout_history,
               "--study-name", study_name,
               "--study-db", study_db,
               "--sample-size", str(sample_size),
               "--output", output_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Job generation failed: {result.stderr}")
            return False
        print(result.stdout)
        
        # Initialize job states
        with open(output_file) as f:
            jobs = json.load(f)
        for job in jobs:
            job_id = job.get("job_id")
            trial_id = int(job_id.split("_")[-1]) if job_id else len(self.job_states)
            self.job_states[job_id] = JobState(job_id=job_id, trial_id=trial_id)
        print(f"✅ Generated {len(self.job_states)} jobs")
        return True
    
    def _generate_retry_jobs(self, missing_ids: Set[str], survivors_file, train_history,
                             holdout_history, study_name, study_db, sample_size, output_file):
        """Generate a jobs file containing ONLY the missing jobs."""
        # Load original jobs
        with open("scorer_jobs.json") as f:
            all_jobs = json.load(f)
        
        # Filter to missing only
        retry_jobs = [j for j in all_jobs if j.get("job_id") in missing_ids]
        
        with open(output_file, "w") as f:
            json.dump(retry_jobs, f, indent=2)
        
        print(f"✅ Generated retry file with {len(retry_jobs)} jobs: {output_file}")
    
    def _push_code_to_remotes(self):
        files = ["survivor_scorer.py", "reinforcement_engine.py", "scorer_trial_worker.py"]
        for node in ["192.168.3.120", "192.168.3.154"]:
            print(f"  → {node}")
            for f in files:
                if Path(f).exists():
                    subprocess.run(["scp", f, f"{node}:~/distributed_prng_analysis/"], capture_output=True)
            print(f"    ✅ Code pushed")
    
    def _push_data_to_remotes(self, survivors_file, train_history, holdout_history, jobs_file):
        files = [survivors_file, train_history, holdout_history, jobs_file]
        for node in ["192.168.3.120", "192.168.3.154"]:
            print(f"  → {node}")
            subprocess.run(["ssh", node, "mkdir -p ~/distributed_prng_analysis/scorer_trial_results"], capture_output=True)
            for f in files:
                if Path(f).exists():
                    subprocess.run(["scp", f, f"{node}:~/distributed_prng_analysis/"], capture_output=True)
            print(f"    ✅ Data pushed")
    
    def _push_retry_file(self, retry_file):
        for node in ["192.168.3.120", "192.168.3.154"]:
            subprocess.run(["scp", retry_file, f"{node}:~/distributed_prng_analysis/"], capture_output=True)
    
    def _dispatch_jobs(self, jobs_file):
        """Run coordinator.py ONCE for the given jobs file."""
        cmd = ["python3", "coordinator.py",
               "--jobs-file", jobs_file,
               "--config", self.coordinator_config,
               "--max-concurrent", str(self.max_concurrent),
               "--resume-policy", "restart"]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        
        for line in process.stdout:
            print(line.rstrip())
        
        process.wait()
        print(f"Coordinator exit code: {process.returncode}")
    
    def _collect_results_via_coordinator(self):
        """Use coordinator's collect_scorer_results() - matches shell script."""
        print("Collecting results via coordinator.collect_scorer_results()...")
        
        collect_script = f"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
coord = MultiGPUCoordinator('{self.coordinator_config}')
results = coord.collect_scorer_results({self.result.total_jobs})
print(f'Collected {{len(results)}} results')
"""
        result = subprocess.run(["python3", "-c", collect_script], 
                               capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Stderr: {result.stderr}")
        
        # Count local results
        RESULTS_DIR.mkdir(exist_ok=True)
        local_count = len(list(RESULTS_DIR.glob("trial_*.json")))
        print(f"Local results after collection: {local_count}")
        self.result.completed = local_count
    
    def _get_missing_job_ids(self) -> Set[str]:
        """Return job IDs that don't have result files."""
        collected = set()
        for f in RESULTS_DIR.glob("trial_*.json"):
            try:
                trial_num = int(f.stem.split("_")[-1])
                collected.add(f"scorer_trial_{trial_num}")
            except:
                pass
        
        all_ids = set(self.job_states.keys())
        missing = all_ids - collected
        
        if missing:
            print(f"Missing results for: {sorted(missing)}")
        
        return missing
    
    def _validate_completion(self):
        collected = set()
        for f in RESULTS_DIR.glob("trial_*.json"):
            try:
                trial_num = int(f.stem.split("_")[-1])
                job_id = f"scorer_trial_{trial_num}"
                collected.add(job_id)
                if job_id in self.job_states:
                    self.job_states[job_id].status = "COMPLETED"
                    self.job_states[job_id].result_file = str(f)
            except:
                pass
        
        for job_id, js in self.job_states.items():
            if js.status != "COMPLETED":
                js.status = "FAILED"
                self.result.failed += 1
        
        self.result.completed = len(collected)
        self.result.job_states = [asdict(js) for js in self.job_states.values()]
        
        print(f"Final: {self.result.completed}/{self.result.total_jobs} completed")
    
    def _save_result_manifest(self):
        with open("step2_results.json", "w") as f:
            json.dump(asdict(self.result), f, indent=2)
        print(f"\n✅ Manifest: step2_results.json")
        print("\n" + "=" * 70)
        print("STEP 2 COMPLETE")
        print("=" * 70)
        print(f"  Completed: {self.result.completed}/{self.result.total_jobs}")
        print(f"  Failed: {self.result.failed}, Retried: {self.result.retried}")
        print(f"  Duration: {self.result.duration_seconds}s")
        print(f"  Success: {'✅' if self.result.success else '❌'}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Step 2 Execution Manager v1.1")
    parser.add_argument("--survivors", required=True)
    parser.add_argument("--train-history", required=True)
    parser.add_argument("--holdout-history", required=True)
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--study-db", required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=25000)
    parser.add_argument("--coordinator-config", default="ml_coordinator_config.json")
    parser.add_argument("--max-concurrent", type=int, default=26)
    args = parser.parse_args()
    
    mgr = Step2ExecutionManager(args.coordinator_config, args.max_concurrent)
    result = mgr.run(args.survivors, args.train_history, args.holdout_history,
                     args.study_name, args.study_db, args.trials, args.sample_size)
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
