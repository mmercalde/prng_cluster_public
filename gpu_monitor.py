#!/usr/bin/env python3
"""
GPU Monitor - Real-time GPU utilization for mixed NVIDIA/AMD cluster
Updated Session 6: Fixed AMD rocm-smi parsing for temp, util, power, clock
"""
import subprocess
import re
from typing import Dict, List


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
    """
    Get AMD GPU stats using rocm-smi
    
    Parses the concise info table format:
    Device  Node  IDs              Temp    Power  Partitions          SCLK  MCLK   Fan     Perf  PwrCap  VRAM%  GPU%
                  (DID,     GUID)  (Edge)  (Avg)  (Mem, Compute, ID)
    0       1     0x73ff,   54645  27.0°C  4.0W   N/A, N/A, 0         0Mhz  96Mhz  27.84%  auto  135.0W  1%     0%
    """
    try:
        # Get main stats from rocm-smi
        cmd = ["ssh", host, "rocm-smi"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            return []
        
        gpus = {}
        
        # Parse each line of output
        for line in result.stdout.strip().split('\n'):
            # Skip header/separator lines
            if line.startswith('=') or 'Device' in line or 'DID' in line or not line.strip():
                continue
            
            # Try to parse GPU data line
            # Format: "0       1     0x73ff,   54645  27.0°C  4.0W   N/A, N/A, 0         0Mhz  96Mhz  27.84%  auto  135.0W  1%     0%"
            parts = line.split()
            if len(parts) >= 10:
                try:
                    gpu_idx = int(parts[0])
                    
                    # Find temperature (contains °C)
                    temp = 0
                    for p in parts:
                        if '°C' in p:
                            temp = int(float(p.replace('°C', '')))
                            break
                    
                    # Find power (contains W but not PwrCap which has larger value)
                    power = 0
                    for p in parts:
                        if p.endswith('W') and '.' in p:
                            val = float(p.replace('W', ''))
                            if val < 50:  # Actual power draw is usually < 50W idle
                                power = val
                                break
                    
                    # Find SCLK (shader clock) - format: "0Mhz" or "1800Mhz"
                    clock = 0
                    for p in parts:
                        if p.endswith('Mhz') or p.endswith('MHz'):
                            try:
                                clk = int(p.replace('Mhz', '').replace('MHz', ''))
                                if clk < 3000:  # SCLK is usually first, MCLK second
                                    clock = clk
                                    break
                            except:
                                pass
                    
                    # Find GPU utilization (last percentage in line)
                    util = 0
                    for p in reversed(parts):
                        if p.endswith('%') and not p.startswith('N'):
                            try:
                                util = int(float(p.replace('%', '')))
                                break
                            except:
                                pass
                    
                    # Find VRAM% (second to last percentage)
                    vram_pct = 0
                    pct_count = 0
                    for p in reversed(parts):
                        if p.endswith('%') and not p.startswith('N'):
                            pct_count += 1
                            if pct_count == 2:  # Second percentage from end is VRAM%
                                try:
                                    vram_pct = int(float(p.replace('%', '')))
                                except:
                                    pass
                                break
                    
                    gpus[gpu_idx] = {
                        'index': gpu_idx,
                        'utilization': util,
                        'temp': temp,
                        'mem_used': int(8192 * vram_pct / 100),  # Estimate from percentage
                        'mem_total': 8192,
                        'power': power,
                        'clock': clock
                    }
                except (ValueError, IndexError):
                    continue
        
        # Return as sorted list, fill in missing GPUs
        result_list = []
        for i in range(12):
            if i in gpus:
                result_list.append(gpus[i])
            else:
                result_list.append({
                    'index': i,
                    'utilization': 0,
                    'temp': 0,
                    'mem_used': 0,
                    'mem_total': 8192,
                    'power': 0,
                    'clock': 0
                })
        
        return result_list
        
    except Exception as e:
        # Return empty stats for all 12 GPUs on error
        return [{'index': i, 'utilization': 0, 'temp': 0, 'mem_used': 0, 
                 'mem_total': 8192, 'power': 0, 'clock': 0} for i in range(12)]


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
    print("=" * 70)
    
    stats = get_cluster_gpu_stats()
    
    for host, gpus in stats.items():
        print(f"\n{host}:")
        print("-" * 50)
        for gpu in gpus:
            print(f"  GPU {gpu['index']:2d}: {gpu['clock']:4d} MHz | {gpu['temp']:2d}°C | {gpu['utilization']:3d}% util | {gpu['power']:.1f}W")
    
    print("\n" + "=" * 70)
    print("Done!")
