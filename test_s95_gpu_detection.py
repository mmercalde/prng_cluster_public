#!/usr/bin/env python3
"""
S95 GPU Detection Standalone Test
Run this on Zeus to verify dual-GPU detection works correctly.
Does NOT import torch or touch CUDA (matches S72 invariant).
"""
import os
import subprocess
import queue

def _s95_detect_cuda_gpus_no_torch() -> int:
    """Exact copy of the detection function from meta_prediction_optimizer_anti_overfit.py"""
    # 1) Respect CUDA_VISIBLE_DEVICES if set
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        cvd = cvd.strip()
        if cvd == "" or cvd.lower() in ("none", "-1"):
            print(f"  [PATH 1a] CVD='{cvd}' -> returning 1")
            return 1
        parts = [p.strip() for p in cvd.split(",") if p.strip()]
        result = max(1, len(parts))
        print(f"  [PATH 1b] CVD='{cvd}' -> {len(parts)} parts -> returning {result}")
        return result

    print("  [PATH 2] CVD not set, trying nvidia-smi -L...")
    # 2) Try nvidia-smi -L (subprocess, no CUDA init)
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True, text=True, timeout=3
        )
        print(f"  nvidia-smi returncode: {proc.returncode}")
        print(f"  nvidia-smi stdout: {proc.stdout.strip()}")
        if proc.returncode == 0:
            lines = [ln for ln in (proc.stdout or "").splitlines()
                     if ln.strip().startswith("GPU ")]
            if lines:
                result = max(1, len(lines))
                print(f"  Found {len(lines)} GPU lines -> returning {result}")
                return result
            else:
                print(f"  No 'GPU ' lines found in output!")
        else:
            print(f"  nvidia-smi failed with rc={proc.returncode}")
            print(f"  stderr: {proc.stderr}")
    except Exception as e:
        print(f"  nvidia-smi exception: {e}")

    # 3) Fallback
    print("  [PATH 3] Fallback -> returning 1")
    return 1


def test_gpu_queue(n_gpus):
    """Test the queue mechanism"""
    q = queue.Queue()
    for i in range(n_gpus):
        q.put(str(i))
    
    print(f"\n  Queue built with {n_gpus} GPUs")
    print(f"  Queue size: {q.qsize()}")
    
    # Simulate lease/return
    gpu = q.get()
    print(f"  Leased GPU: {gpu}")
    print(f"  Queue size after lease: {q.qsize()}")
    q.put(gpu)
    print(f"  Returned GPU: {gpu}")
    print(f"  Queue size after return: {q.qsize()}")


if __name__ == "__main__":
    print("=" * 60)
    print("S95 GPU Detection Test")
    print("=" * 60)
    
    print(f"\nEnvironment:")
    print(f"  CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"  S88_COMPARE_MODELS_CHILD = {os.environ.get('S88_COMPARE_MODELS_CHILD', '<unset>')}")
    
    print(f"\nRunning detection (normal):")
    n = _s95_detect_cuda_gpus_no_torch()
    print(f"\n  >>> RESULT: n_jobs = {n}")
    
    test_gpu_queue(n)
    
    # Also test what happens when S88 child env is set
    print(f"\n{'=' * 60}")
    print("Simulating S88 child environment...")
    os.environ["S88_COMPARE_MODELS_CHILD"] = "1"
    print(f"  S88_COMPARE_MODELS_CHILD = {os.environ.get('S88_COMPARE_MODELS_CHILD')}")
    
    print(f"\nRunning detection (as S88 child):")
    n2 = _s95_detect_cuda_gpus_no_torch()
    print(f"\n  >>> RESULT: n_jobs = {n2}")
    
    del os.environ["S88_COMPARE_MODELS_CHILD"]
    
    # Test with CVD set to single GPU (simulating lease)
    print(f"\n{'=' * 60}")
    print("Simulating CUDA_VISIBLE_DEVICES='0':")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    n3 = _s95_detect_cuda_gpus_no_torch()
    print(f"\n  >>> RESULT: n_jobs = {n3}")
    
    del os.environ["CUDA_VISIBLE_DEVICES"]
    
    print(f"\n{'=' * 60}")
    print(f"SUMMARY:")
    print(f"  Normal:     n_jobs = {n}  {'✅' if n == 2 else '❌ EXPECTED 2'}")
    print(f"  S88 child:  n_jobs = {n2} {'✅' if n2 == 2 else '❌ EXPECTED 2'}")
    print(f"  CVD='0':    n_jobs = {n3} {'✅' if n3 == 1 else '❌ EXPECTED 1'}")
    print(f"{'=' * 60}")
