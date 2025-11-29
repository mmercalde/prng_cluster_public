#!/usr/bin/env python3
"""Test running two localhost jobs in parallel - like coordinator does"""
import subprocess
import os
import threading
import time

def run_job(gpu_id, job_name):
    job_file = f"job_debug_gpu{gpu_id}.json"
    
    # Create job file
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
            '{"residue_mod_1": 15, "residue_mod_2": 100, "residue_mod_3": 500, "max_offset": 5, "temporal_window_size": 80, "temporal_num_windows": 8, "min_confidence_threshold": 0.1, "hidden_layers": "128_64", "dropout": 0.3, "learning_rate": 0.001, "batch_size": 64, "sample_size": 1000}'
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
        print(f"  STDERR (last 500 chars): {stderr[-500:]}")
    else:
        # Find the JSON result
        for line in stdout.strip().split('\n'):
            if '"status"' in line:
                print(f"  Result: {line[:100]}...")
    
    return proc.returncode

# Launch both jobs at EXACTLY the same time
print("=== Testing SIMULTANEOUS launch (no stagger) ===")
t0 = threading.Thread(target=run_job, args=(0, "gpu0"))
t1 = threading.Thread(target=run_job, args=(1, "gpu1"))

t0.start()
t1.start()

t0.join()
t1.join()

print("\n=== Done ===")
