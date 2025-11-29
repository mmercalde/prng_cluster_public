#!/usr/bin/env python3
"""Test running two localhost jobs with stagger"""
import subprocess
import os
import threading
import time

def run_job(gpu_id):
    job_file = f"job_debug_gpu{gpu_id}.json"
    
    import json
    job_data = {
        "job_id": f"debug_gpu{gpu_id}",
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
    
    print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu_id}: Starting subprocess...")
    
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
    
    stdout, stderr = proc.communicate(timeout=120)
    
    print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu_id}: Return code {proc.returncode}")
    if proc.returncode != 0:
        print(f"  STDERR (last 300 chars): {stderr[-300:]}")
    return proc.returncode

print("=== Testing WITH 5-second stagger ===")

# Start GPU 0 first
t0 = threading.Thread(target=run_job, args=(0,))
t0.start()

# Wait 5 seconds
print(f"[{time.strftime('%H:%M:%S')}] Waiting 5 seconds before GPU 1...")
time.sleep(5)

# Then start GPU 1
t1 = threading.Thread(target=run_job, args=(1,))
t1.start()

t0.join()
t1.join()

print("\n=== Done ===")
