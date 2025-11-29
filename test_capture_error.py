#!/usr/bin/env python3
"""Capture the actual error from GPU 1"""
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
    
    stdout, stderr = proc.communicate(timeout=180)
    
    # Save full output to files
    with open(f'/tmp/gpu{gpu_id}_stdout.txt', 'w') as f:
        f.write(stdout)
    with open(f'/tmp/gpu{gpu_id}_stderr.txt', 'w') as f:
        f.write(stderr)
    
    print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu_id}: Return code {proc.returncode}")
    print(f"  Output saved to /tmp/gpu{gpu_id}_stdout.txt and /tmp/gpu{gpu_id}_stderr.txt")
    return proc.returncode

t0 = threading.Thread(target=run_job, args=(0,))
t1 = threading.Thread(target=run_job, args=(1,))

t0.start()
time.sleep(5)  # Stagger
t1.start()

t0.join()
t1.join()

print("\n=== GPU 1 STDERR (the one that failed): ===")
with open('/tmp/gpu1_stderr.txt', 'r') as f:
    print(f.read())
