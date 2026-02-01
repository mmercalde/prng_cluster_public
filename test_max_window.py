#!/usr/bin/env python3
"""
GPU Memory Stress Test - Find Maximum Safe Window Size
Tests hybrid sieve memory allocations to find safe limits
"""

import argparse
import json
import sys
import os

# ROCm environment setup
import socket
HOST = socket.gethostname()
if HOST in ["rig-6600", "rig-6600b", "rig-6600c"]:
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    os.environ.setdefault("HSA_ENABLE_SDMA", "0")
os.environ.setdefault("ROCM_PATH", "/opt/rocm")
os.environ.setdefault("HIP_PATH", "/opt/rocm")

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    print("ERROR: CuPy not available", file=sys.stderr)
    sys.exit(1)


def get_gpu_memory_info(device):
    """Get GPU memory information"""
    with device:
        mempool = cp.get_default_memory_pool()
        total_bytes = device.mem_info[1]  # Total memory
        used_bytes = mempool.used_bytes()
        free_bytes = total_bytes - used_bytes
        
        return {
            'total_gb': total_bytes / (1024**3),
            'used_gb': used_bytes / (1024**3),
            'free_gb': free_bytes / (1024**3),
            'used_bytes': used_bytes,
            'free_bytes': free_bytes,
            'total_bytes': total_bytes
        }


def test_allocation(gpu_id, n_seeds, window_size, chunk_size):
    """Test if we can allocate memory for given parameters"""
    device = cp.cuda.Device(gpu_id)
    
    try:
        with device:
            # Clear memory pool
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            
            # Get initial memory state
            initial_mem = get_gpu_memory_info(device)
            
            # Simulate hybrid sieve allocations
            # These are the arrays allocated in run_hybrid_sieve()
            
            # 1. Seeds array
            seeds_gpu = cp.arange(0, n_seeds, dtype=cp.uint32)
            
            # 2. Residues array (window_size values)
            residues_gpu = cp.zeros(window_size, dtype=cp.uint32)
            
            # 3. Survivors array
            survivors_gpu = cp.zeros(n_seeds, dtype=cp.uint32)
            
            # 4. Match rates
            match_rates_gpu = cp.zeros(n_seeds, dtype=cp.float32)
            
            # 5. Strategy IDs
            strategy_ids_gpu = cp.zeros(n_seeds, dtype=cp.uint32)
            
            # 6. Skip sequences - THE BIG ONE
            # This is n_seeds * window_size uint32 values
            skip_sequences_gpu = cp.zeros(n_seeds * window_size, dtype=cp.uint32)
            
            # 7. Survivor count
            survivor_count_gpu = cp.zeros(1, dtype=cp.uint32)
            
            # 8. Strategy arrays (assume 3 strategies)
            strategy_max_misses = cp.zeros(3, dtype=cp.int32)
            strategy_tolerances = cp.zeros(3, dtype=cp.int32)
            
            # Force GPU to actually allocate
            cp.cuda.Device().synchronize()
            
            # Get memory after allocation
            allocated_mem = get_gpu_memory_info(device)
            
            # Calculate actual usage
            bytes_used = allocated_mem['used_bytes'] - initial_mem['used_bytes']
            
            # Clean up
            del seeds_gpu, residues_gpu, survivors_gpu, match_rates_gpu
            del strategy_ids_gpu, skip_sequences_gpu, survivor_count_gpu
            del strategy_max_misses, strategy_tolerances
            
            mempool.free_all_blocks()
            
            return {
                'success': True,
                'bytes_allocated': bytes_used,
                'mb_allocated': bytes_used / (1024**2),
                'gb_allocated': bytes_used / (1024**3),
                'initial_free_gb': initial_mem['free_gb'],
                'final_free_gb': allocated_mem['free_gb'],
                'memory_percentage': (bytes_used / initial_mem['total_bytes']) * 100
            }
            
    except cp.cuda.memory.OutOfMemoryError as e:
        return {
            'success': False,
            'error': 'OutOfMemoryError',
            'message': str(e)
        }
    except Exception as e:
        return {
            'success': False,
            'error': type(e).__name__,
            'message': str(e)
        }


def find_max_window_size(gpu_id, chunk_size, max_window=2048):
    """Binary search to find maximum safe window size"""
    device = cp.cuda.Device(gpu_id)
    
    # Get GPU info
    with device:
        props = cp.cuda.runtime.getDeviceProperties(gpu_id)
        gpu_name = props['name'].decode('utf-8')
        mem_info = get_gpu_memory_info(device)
    
    print(f"\n{'='*70}")
    print(f"GPU {gpu_id}: {gpu_name}")
    print(f"Total Memory: {mem_info['total_gb']:.2f} GB")
    print(f"Free Memory:  {mem_info['free_gb']:.2f} GB")
    print(f"Testing chunk_size: {chunk_size:,} seeds")
    print(f"{'='*70}\n")
    
    # Binary search for max window size
    low = 10
    high = max_window
    max_working = None
    
    while low <= high:
        mid = (low + high) // 2
        
        print(f"Testing window_size={mid}...", end=" ", flush=True)
        
        result = test_allocation(gpu_id, chunk_size, mid, chunk_size)
        
        if result['success']:
            print(f"✅ SUCCESS - {result['mb_allocated']:.1f} MB allocated ({result['memory_percentage']:.1f}% of GPU memory)")
            max_working = mid
            low = mid + 1
        else:
            print(f"❌ FAILED - {result['error']}")
            high = mid - 1
    
    return max_working


def test_specific_configs(gpu_id):
    """Test specific common configurations"""
    device = cp.cuda.Device(gpu_id)
    
    # Get GPU info
    with device:
        props = cp.cuda.runtime.getDeviceProperties(gpu_id)
        gpu_name = props['name'].decode('utf-8')
        mem_info = get_gpu_memory_info(device)
    
    print(f"\n{'='*70}")
    print(f"GPU {gpu_id}: {gpu_name}")
    print(f"Total Memory: {mem_info['total_gb']:.2f} GB")
    print(f"{'='*70}\n")
    
    # Common configurations to test
    configs = [
        # (chunk_size, window_size, description)
        (100_000, 512, "Default hybrid config"),
        (100_000, 256, "Conservative hybrid config"),
        (100_000, 128, "Very conservative hybrid config"),
        (50_000, 512, "Half chunk, default window"),
        (10_000, 512, "Small chunk, default window"),
        (10_000, 1024, "Small chunk, large window"),
        (1_000, 2048, "Tiny chunk, huge window"),
    ]
    
    results = []
    
    for chunk_size, window_size, description in configs:
        print(f"\n{description}")
        print(f"  chunk_size={chunk_size:,}, window_size={window_size}")
        print(f"  ", end="", flush=True)
        
        result = test_allocation(gpu_id, chunk_size, window_size, chunk_size)
        
        if result['success']:
            print(f"✅ SUCCESS - {result['mb_allocated']:.1f} MB ({result['memory_percentage']:.1f}% of GPU)")
            results.append({
                'config': description,
                'chunk_size': chunk_size,
                'window_size': window_size,
                'success': True,
                'mb_allocated': result['mb_allocated'],
                'memory_percentage': result['memory_percentage']
            })
        else:
            print(f"❌ FAILED - {result['error']}: {result['message']}")
            results.append({
                'config': description,
                'chunk_size': chunk_size,
                'window_size': window_size,
                'success': False,
                'error': result['error']
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Test maximum safe window size for GPU hybrid sieve',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU device ID to test')
    parser.add_argument('--chunk-size', type=int, default=100_000, 
                       help='Number of seeds per chunk (default: 100,000)')
    parser.add_argument('--max-window', type=int, default=2048,
                       help='Maximum window size to test (default: 2048)')
    parser.add_argument('--mode', choices=['binary', 'configs'], default='binary',
                       help='Test mode: binary search or specific configs')
    parser.add_argument('--output', type=str, help='Save results to JSON file')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'binary':
            max_window = find_max_window_size(args.gpu_id, args.chunk_size, args.max_window)
            
            if max_window:
                print(f"\n{'='*70}")
                print(f"✅ RESULT: Maximum safe window_size = {max_window}")
                print(f"   For chunk_size = {args.chunk_size:,}")
                print(f"{'='*70}\n")
                
                result = {
                    'gpu_id': args.gpu_id,
                    'chunk_size': args.chunk_size,
                    'max_window_size': max_window,
                    'success': True
                }
            else:
                print(f"\n❌ Could not find any working configuration")
                result = {
                    'gpu_id': args.gpu_id,
                    'chunk_size': args.chunk_size,
                    'success': False
                }
        
        else:  # configs mode
            test_results = test_specific_configs(args.gpu_id)
            
            print(f"\n{'='*70}")
            print("SUMMARY OF CONFIGURATIONS:")
            print(f"{'='*70}")
            
            successful = [r for r in test_results if r['success']]
            failed = [r for r in test_results if not r['success']]
            
            if successful:
                print(f"\n✅ Successful configurations ({len(successful)}):")
                for r in successful:
                    print(f"  • {r['config']}")
                    print(f"    chunk={r['chunk_size']:,}, window={r['window_size']}, "
                          f"mem={r['mb_allocated']:.1f}MB ({r['memory_percentage']:.1f}%)")
            
            if failed:
                print(f"\n❌ Failed configurations ({len(failed)}):")
                for r in failed:
                    print(f"  • {r['config']}")
                    print(f"    chunk={r['chunk_size']:,}, window={r['window_size']}, "
                          f"error={r['error']}")
            
            print(f"\n{'='*70}\n")
            
            result = {
                'gpu_id': args.gpu_id,
                'test_results': test_results,
                'successful_count': len(successful),
                'failed_count': len(failed)
            }
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to: {args.output}")
        
        # Print as JSON to stdout
        print(json.dumps(result, indent=2))
        
        return 0
        
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
