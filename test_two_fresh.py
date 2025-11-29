#!/usr/bin/env python3
"""Test running two localhost jobs in parallel with fresh output capture"""
import subprocess
import os
import threading
import time
import json

results = {}

def run_job(gpu_id):
    job_file = f"job_test_gpu{gpu_id}.json"
    
    # Create fresh job file
    job_data = {
        "job_id": f"test_gpu{gpu_id}",
        "analysis_type": "script",
        "script": "scorer_trial_worker.py",
        "args": [
            "bidirectional_survivors.json",
            "train_history.json", 
            "holdout_history.json",
            str(gpu_id),
            '{"sample_size": 1000}'
        ],
        "timeout": 300
    }
    with open(job_file, 'w') as f:
        json.dump(job_data, f)
    
    activate_path = "/home/michael/venvs/torch/bin/activate"
    cmd_str = f"source {activate_path} && CUDA_VISIBLE_DEVICES={gpu_id} python -u distributed_worker.py {job_file} --gpu-id 0"
    
    env = os.environ.copy()
    env.update({
        'CUPY_CUDA_MEMORY_POOL_TYPE': 'none',
        'OPENBLAS_NUM_THREADS': '1', 
        'OMP_NUM_THREADS': '1',
        'CUDA_DEVICE_MAX_CONNECTIONS': '1'
    })
    
    print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu_id}: Starting...")
    start = time.time()
    
    proc = subprocess.Popen(
        cmd_str,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.getcwd(),
        env=env,
        preexec_fn=os.setsid,
        text=True
    )
    
    stdout, stderr = proc.communicate(timeout=180)
    elapsed = time.time() - start
    
    results[gpu_id] = {
        'returncode': proc.returncode,
        'stdout': stdout,
        'stderr': stderr,
        'elapsed': elapsed
    }
    
    print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu_id}: Return code {proc.returncode} ({elapsed:.1f}s)")

print("=== Testing SIMULTANEOUS launch ===\n")

t0 = threading.Thread(target=run_job, args=(0,))
t1 = threading.Thread(target=run_job, args=(1,))

t0.start()
t1.start()

t0.join()
t1.join()

print("\n=== Results ===")
for gpu_id in [0, 1]:
    r = results[gpu_id]
    print(f"\nGPU {gpu_id}: {'SUCCESS' if r['returncode'] == 0 else 'FAILED'}")
    if r['returncode'] != 0:
        print(f"  STDOUT (last 800 chars):\n{r['stdout'][-800:]}")
        if r['stderr']:
            print(f"  STDERR: {r['stderr'][-300:]}")
