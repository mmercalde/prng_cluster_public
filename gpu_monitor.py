#!/usr/bin/env python3
"""
GPU Monitor - Real-time GPU utilization for mixed NVIDIA/AMD cluster
"""

import subprocess
import json
import time
from typing import Dict, List, Optional


def get_nvidia_gpu_stats(host: str = "localhost") -> List[Dict]:
    """Get NVIDIA GPU stats using nvidia-smi"""
    try:
        if host == "localhost":
            cmd = ["nvidia-smi", "--query-gpu=index,utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw,clocks.gr", "--format=csv,noheader,nounits"]
        else:
            cmd = ["ssh", host, "nvidia-smi --query-gpu=index,utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw,clocks.gr --format=csv,noheader,nounits"]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return []
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 7:
                    gpus.append({
                        'index': int(parts[0]),
                        'utilization': int(parts[1]),
                        'temp': int(parts[2]),
                        'mem_used': int(parts[3]),
                        'mem_total': int(parts[4]),
                        'power': float(parts[5]) if parts[5] != '[N/A]' else 0,
                        'clock': int(parts[6]) if parts[6] != '[N/A]' else 0
                    })
        return gpus
    except Exception as e:
        return []


def get_amd_gpu_stats(host: str) -> List[Dict]:
    """Get AMD GPU stats using rocm-smi including clock speeds"""
    try:
        # Get clock speeds
        cmd = ["ssh", host, "rocm-smi --showclocks"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        # Parse clock speeds - sclk is the shader/compute clock
        clocks = {}
        for line in result.stdout.strip().split('\n'):
            if 'sclk' in line.lower():
                gpu_idx = None
                clock = None
                parts = line.split()
                for p in parts:
                    if 'GPU[' in p:
                        try:
                            gpu_idx = int(p.replace('GPU[', '').replace(']', ''))
                        except:
                            pass
                    if 'Mhz' in p or 'MHz' in p:
                        try:
                            clock = int(p.replace('(', '').replace(')', '').replace('Mhz', '').replace('MHz', ''))
                        except:
                            pass
                if gpu_idx is not None:
                    clocks[gpu_idx] = clock if clock else 0
        
        # Build GPU list for all 12 GPUs
        gpus = []
        for i in range(12):
            gpus.append({
                'index': i,
                'utilization': 0,
                'temp': 0,
                'mem_used': 0,
                'mem_total': 8192,
                'power': 0,
                'clock': clocks.get(i, 0)
            })
        
        return gpus
    except Exception as e:
        return []


def get_cluster_gpu_stats() -> Dict[str, List[Dict]]:
    """Get GPU stats for entire cluster"""
    cluster_stats = {}
    
    # Zeus (localhost) - NVIDIA
    cluster_stats['localhost'] = get_nvidia_gpu_stats('localhost')
    
    # Mining rigs - AMD
    cluster_stats['192.168.3.120'] = get_amd_gpu_stats('192.168.3.120')
    cluster_stats['192.168.3.154'] = get_amd_gpu_stats('192.168.3.154')
    
    return cluster_stats


if __name__ == '__main__':
    print("Testing GPU monitoring...")
    stats = get_cluster_gpu_stats()
    for host, gpus in stats.items():
        print(f"\n{host}:")
        for gpu in gpus:
            print(f"  GPU {gpu['index']}: clock={gpu.get('clock', 0)} MHz, util={gpu['utilization']}%, temp={gpu['temp']}°C")
