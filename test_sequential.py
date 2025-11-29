#!/usr/bin/env python3
import subprocess, os, time, json

def run_job(gpu_id):
    job_file = f"job_seq_gpu{gpu_id}.json"
    with open(job_file, 'w') as f:
        json.dump({"job_id": f"seq_gpu{gpu_id}", "analysis_type": "script", "script": "scorer_trial_worker.py",
            "args": ["bidirectional_survivors.json", "train_history.json", "holdout_history.json", str(gpu_id), '{"sample_size": 1000}'], "timeout": 300}, f)
    
    cmd = f"source /home/michael/venvs/torch/bin/activate && CUDA_VISIBLE_DEVICES={gpu_id} python -u distributed_worker.py {job_file} --gpu-id 0"
    print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu_id}: Starting...")
    proc = subprocess.Popen(cmd, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, 'CUPY_CUDA_MEMORY_POOL_TYPE': 'none', 'OMP_NUM_THREADS': '1'})
    stdout, _ = proc.communicate(timeout=180)
    print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu_id}: {'SUCCESS' if proc.returncode == 0 else 'FAILED'}")
    return proc.returncode

print("=== SEQUENTIAL (GPU 0 finishes, then GPU 1 starts) ===")
rc0 = run_job(0)
print(f"GPU 0 done, now starting GPU 1...")
rc1 = run_job(1)
print(f"\nResults: GPU 0 = {rc0}, GPU 1 = {rc1}")
