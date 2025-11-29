#!/usr/bin/env python3
"""Mimic exactly what coordinator does to spawn localhost jobs"""
import subprocess
import os
import sys

job_file = "job_debug_gpu1.json"
gpu_id = 1  # Test GPU 1

activate_path = "/home/michael/venvs/torch/bin/activate"
worker_script = "distributed_worker.py"
worker_args = f"{job_file} --gpu-id 0"  # Note: --gpu-id 0 because CUDA_VISIBLE_DEVICES handles isolation

cmd_str = (
    f"source {activate_path} && "
    f"CUDA_VISIBLE_DEVICES={gpu_id} "
    f"python -u {worker_script} {worker_args}"
).strip()

print(f"Command: {cmd_str}")
print(f"CWD: {os.getcwd()}")
print("="*60)

# Enhanced environment (same as coordinator lines 649-655)
env = os.environ.copy()
env.update({
    'CUPY_CUDA_MEMORY_POOL_TYPE': 'none',
    'OPENBLAS_NUM_THREADS': '1',
    'OMP_NUM_THREADS': '1',
    'CUDA_DEVICE_MAX_CONNECTIONS': '1'
})

proc = subprocess.Popen(
    cmd_str,
    shell=True,
    executable="/bin/bash",
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=os.getcwd(),
    env=env,
    preexec_fn=os.setsid,
    text=True,
    encoding='utf-8',
    errors='ignore'
)

stdout, stderr = proc.communicate(timeout=120)
print("STDOUT:")
print(stdout)
print("="*60)
print("STDERR:")
print(stderr[-2000:] if len(stderr) > 2000 else stderr)
print("="*60)
print(f"Return code: {proc.returncode}")
