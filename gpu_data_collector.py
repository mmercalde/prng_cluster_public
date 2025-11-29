#!/usr/bin/env python3
"""
GPU Data Collector - Fixed AMD GPU parsing
Collects actual GPU performance data from nvidia-smi and rocm-smi
"""

import subprocess
import json
import time
import paramiko
from typing import Dict, List, Any

class GPUDataCollector:
    def __init__(self):
        self.local_gpus = []
        self.remote_nodes = {
            '192.168.3.120': {'gpus': 12, 'type': 'RX 6600'},
            '192.168.3.154': {'gpus': 12, 'type': 'RX 6600'}
        }
    
    def get_local_nvidia_data(self) -> List[Dict]:
        """Get real NVIDIA GPU data from local machine"""
        try:
            cmd = [
                'nvidia-smi', 
                '--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                gpu_data = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 7:
                            gpu_data.append({
                                'id': f"zeus-{parts[0]}",
                                'name': parts[1],
                                'temperature': int(parts[2]) if parts[2] != '[Not Supported]' else 0,
                                'utilization': int(parts[3]) if parts[3] != '[Not Supported]' else 0,
                                'memory_used': int(parts[4]) if parts[4] != '[Not Supported]' else 0,
                                'memory_total': int(parts[5]) if parts[5] != '[Not Supported]' else 0,
                                'power': float(parts[6]) if parts[6] != '[Not Supported]' else 0,
                                'status': 'active' if int(parts[3]) > 0 else 'idle',
                                'node': 'localhost'
                            })
                return gpu_data
            else:
                print(f"nvidia-smi failed: {result.stderr}")
                return []
                
        except Exception as e:
            print(f"Error getting NVIDIA data: {e}")
            return []
    
    def get_remote_amd_data(self, host: str) -> List[Dict]:
        """Get real AMD GPU data from remote nodes using rocm-smi"""
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(host, username='michael', timeout=10)
            
            # Use rocm-smi to get GPU data
            stdin, stdout, stderr = ssh.exec_command('rocm-smi --showtemp --showuse --showmemuse --json', timeout=15)
            
            gpu_data = []
            node_info = self.remote_nodes.get(host, {})
            
            stdout_data = stdout.read().decode().strip()
            stderr_data = stderr.read().decode().strip()
            
            if stdout_data and not stderr_data:
                try:
                    # Parse the rocm-smi JSON output
                    rocm_data = json.loads(stdout_data)
                    
                    # rocm-smi returns data with keys like "card0", "card1", etc.
                    for card_key, card_data in rocm_data.items():
                        if card_key.startswith('card'):
                            card_number = card_key.replace('card', '')
                            
                            # Extract temperature (use edge temperature as primary)
                            temp_edge = card_data.get('Temperature (Sensor edge) (C)', '0')
                            temperature = int(float(temp_edge))
                            
                            # Extract utilization
                            gpu_use = card_data.get('GPU use (%)', '0')
                            utilization = int(gpu_use)
                            
                            # Extract memory usage (VRAM%)
                            vram_pct = card_data.get('GPU Memory Allocated (VRAM%)', '0')
                            memory_pct = int(vram_pct)
                            
                            # RX 6600 has 8GB VRAM
                            memory_total_mb = 8192
                            memory_used_mb = int((memory_pct / 100) * memory_total_mb)
                            
                            # Estimate power based on utilization (RX 6600 max ~132W)
                            estimated_power = 30 + (utilization * 1.0)  # Base 30W + scaling
                            
                            gpu_data.append({
                                'id': f"{host.replace('.', '_')}-{card_number}",
                                'name': f"{node_info.get('type', 'RX 6600')} {card_number}",
                                'temperature': temperature,
                                'utilization': utilization,
                                'memory_used': memory_used_mb,
                                'memory_total': memory_total_mb,
                                'power': estimated_power,
                                'status': 'active' if utilization > 5 else 'idle',
                                'node': host
                            })
                    
                    print(f"Successfully parsed {len(gpu_data)} GPUs from {host}")
                    
                except json.JSONDecodeError as e:
                    print(f"JSON decode error for {host}: {e}")
                    print(f"Raw output: {stdout_data[:200]}...")
                    gpu_data = self._create_fallback_gpus(host, node_info)
                    
            else:
                print(f"rocm-smi failed for {host}: {stderr_data}")
                gpu_data = self._create_fallback_gpus(host, node_info)
            
            ssh.close()
            return gpu_data
            
        except Exception as e:
            print(f"Error connecting to {host}: {e}")
            # Return offline placeholder GPUs instead of empty list
            node_info = self.remote_nodes.get(host, {})
            return self._create_fallback_gpus(host, node_info, status='offline')
    
    def _create_fallback_gpus(self, host: str, node_info: Dict, status: str = 'unknown') -> List[Dict]:
        """Create fallback GPU data when rocm-smi fails"""
        gpu_count = node_info.get('gpus', 12)
        gpu_type = node_info.get('type', 'RX 6600')
        
        fallback_gpus = []
        for i in range(gpu_count):
            fallback_gpus.append({
                'id': f"{host.replace('.', '_')}-{i}",
                'name': f"{gpu_type} {i}",
                'temperature': 0 if status == 'offline' else 35,
                'utilization': 0,
                'memory_used': 0,
                'memory_total': 8192,
                'power': 0 if status == 'offline' else 30,
                'status': status,
                'node': host
            })
        
        return fallback_gpus
    
    def get_all_gpu_data(self) -> List[Dict]:
        """Collect real GPU data from all nodes"""
        all_gpus = []
        
        # Get local NVIDIA data
        print("Collecting local NVIDIA GPU data...")
        local_data = self.get_local_nvidia_data()
        all_gpus.extend(local_data)
        print(f"Found {len(local_data)} local GPUs")
        
        # Get remote AMD data
        for host in self.remote_nodes.keys():
            print(f"Collecting remote AMD GPU data from {host}...")
            remote_data = self.get_remote_amd_data(host)
            all_gpus.extend(remote_data)
            print(f"Found {len(remote_data)} GPUs on {host}")
        
        print(f"Total GPUs collected: {len(all_gpus)}")
        return all_gpus
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get cluster-wide performance summary"""
        gpu_data = self.get_all_gpu_data()
        
        if not gpu_data:
            return {
                'total_gpus': 26,
                'active_gpus': 0,
                'avg_utilization': 0,
                'avg_temperature': 0,
                'total_power': 0,
                'total_memory_used': 0,
                'total_memory': 0,
                'memory_utilization': 0
            }
        
        active_gpus = sum(1 for gpu in gpu_data if gpu['status'] == 'active')
        avg_util = sum(gpu['utilization'] for gpu in gpu_data) / len(gpu_data)
        avg_temp = sum(gpu['temperature'] for gpu in gpu_data) / len(gpu_data)
        total_power = sum(gpu['power'] for gpu in gpu_data)
        total_mem_used = sum(gpu['memory_used'] for gpu in gpu_data)
        total_mem = sum(gpu['memory_total'] for gpu in gpu_data)
        
        return {
            'total_gpus': len(gpu_data),
            'active_gpus': active_gpus,
            'avg_utilization': round(avg_util, 1),
            'avg_temperature': round(avg_temp, 1),
            'total_power': round(total_power, 1),
            'total_memory_used': total_mem_used,
            'total_memory': total_mem,
            'memory_utilization': round((total_mem_used / total_mem) * 100, 1) if total_mem > 0 else 0
        }
