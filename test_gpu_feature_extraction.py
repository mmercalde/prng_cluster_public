#!/usr/bin/env python3
"""
GPU Feature Extraction Benchmark - Standalone test script
"""
import json, time, argparse, numpy as np
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available - GPU tests skipped")

def benchmark_data_loading(survivors_file, num_samples=25000):
    print("\n" + "="*60)
    print("PHASE 1: DATA LOADING BENCHMARK")
    print("="*60)
    
    t0 = time.time()
    with open(survivors_file) as f:
        survivors = json.load(f)
    json_time = time.time() - t0
    print(f"JSON load: {json_time:.2f}s ({len(survivors):,} entries)")
    
    t0 = time.time()
    seeds = np.array([s['seed'] for s in survivors[:num_samples]], dtype=np.int64)
    extract_time = time.time() - t0
    print(f"Seed extraction: {extract_time:.2f}s")
    
    binary_path = Path("/tmp/test_seeds.npy")
    np.save(binary_path, seeds)
    
    t0 = time.time()
    seeds_loaded = np.load(binary_path)
    binary_time = time.time() - t0
    print(f"Binary load: {binary_time:.4f}s")
    print(f"Speedup: {(json_time + extract_time) / binary_time:.1f}x")
    
    return seeds, survivors[:num_samples]

def extract_features_cpu(seeds, params):
    features = []
    for seed in seeds:
        f = [
            seed % params['mod1'], seed % params['mod2'], seed % params['mod3'],
            bin(seed).count('1'), seed & 0xFFFF, (seed >> 16) & 0xFFFF,
            seed / (2**32), (seed % 1000) / 1000,
        ]
        features.append(f)
    return np.array(features, dtype=np.float32)

def extract_features_gpu(seeds_tensor, params, device):
    n = seeds_tensor.shape[0]
    mods = torch.tensor([params['mod1'], params['mod2'], params['mod3']], device=device)
    residues = seeds_tensor.unsqueeze(1) % mods
    
    popcount = torch.zeros(n, device=device)
    temp = seeds_tensor.clone()
    for _ in range(32):
        popcount += (temp & 1).float()
        temp = temp >> 1
    
    low_bits = (seeds_tensor & 0xFFFF).float()
    high_bits = ((seeds_tensor >> 16) & 0xFFFF).float()
    normalized = seeds_tensor.float() / (2**32)
    fractional = (seeds_tensor % 1000).float() / 1000
    
    return torch.cat([
        residues.float(), popcount.unsqueeze(1), low_bits.unsqueeze(1),
        high_bits.unsqueeze(1), normalized.unsqueeze(1), fractional.unsqueeze(1)
    ], dim=1)

def benchmark_feature_extraction(seeds, params, device='cuda'):
    print("\n" + "="*60)
    print("PHASE 2: FEATURE EXTRACTION BENCHMARK")
    print("="*60)
    
    n = len(seeds)
    
    t0 = time.time()
    features_cpu = extract_features_cpu(seeds, params)
    cpu_time = time.time() - t0
    print(f"CPU: {cpu_time:.3f}s ({n/cpu_time:,.0f} samples/sec)")
    
    if not HAS_TORCH or not torch.cuda.is_available():
        print("GPU not available")
        return cpu_time, None
    
    t0 = time.time()
    seeds_tensor = torch.from_numpy(seeds).to(device)
    transfer_time = time.time() - t0
    
    _ = extract_features_gpu(seeds_tensor[:100], params, device)
    torch.cuda.synchronize()
    
    t0 = time.time()
    features_gpu = extract_features_gpu(seeds_tensor, params, device)
    torch.cuda.synchronize()
    gpu_time = time.time() - t0
    
    print(f"GPU: {gpu_time:.3f}s ({n/gpu_time:,.0f} samples/sec)")
    print(f"Transfer: {transfer_time:.3f}s")
    print(f"Speedup: {cpu_time/(gpu_time+transfer_time):.1f}x")
    print(f"Memory: {torch.cuda.memory_allocated()/1e6:.1f} MB")
    
    return cpu_time, gpu_time

def benchmark_concurrent(seeds, params, device, num_concurrent=12):
    print("\n" + "="*60)
    print(f"PHASE 3: CONCURRENT SIMULATION ({num_concurrent} jobs)")
    print("="*60)
    
    if not HAS_TORCH or not torch.cuda.is_available():
        return
    
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory: {total_mem:.1f} GB")
    
    tensors = []
    try:
        for i in range(num_concurrent):
            t = torch.from_numpy(seeds).to(device)
            f = extract_features_gpu(t, params, device)
            tensors.append((t, f))
            print(f"Job {i+1}: {torch.cuda.memory_allocated()/1e6:.1f} MB")
        print(f"\n✅ {num_concurrent} concurrent jobs fit in memory!")
    except RuntimeError as e:
        print(f"\n❌ OOM at job {len(tensors)+1}")
    finally:
        del tensors
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--survivors", default="bidirectional_survivors.json")
    parser.add_argument("--samples", type=int, default=25000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--concurrent", type=int, default=12)
    args = parser.parse_args()
    
    params = {'mod1': 19, 'mod2': 108, 'mod3': 1299}
    
    print("="*60)
    print("GPU FEATURE EXTRACTION BENCHMARK")
    print("="*60)
    print(f"Samples: {args.samples:,}, Device: {args.device}")
    
    if HAS_TORCH and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    seeds, _ = benchmark_data_loading(args.survivors, args.samples)
    benchmark_feature_extraction(seeds, params, args.device)
    benchmark_concurrent(seeds, params, args.device, args.concurrent)
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
