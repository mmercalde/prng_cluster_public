#!/usr/bin/env python3
"""
System Monitor Module - Hardware monitoring and diagnostics
"""

import os
import json
import subprocess
import time
from typing import Dict, Any

class SystemMonitor:
    """System monitoring and diagnostics functionality"""

    def __init__(self, core):
        """Initialize with core system reference"""
        self.core = core
        self.module_name = "SystemMonitor"

    def status_menu(self):
        """System status menu"""
        while True:
            self.core.clear_screen()
            self.core.print_header()
            print(f"\n{self.module_name.upper()} - STATUS")
            print("-" * 20)
            print("  1. GPU Status")
            print("  2. Network Status")
            print("  3. File System Status")
            print("  4. Process Monitor")
            print("  5. Configuration Check")
            print("  6. Back to Main Menu")
            print("-" * 20)

            choice = input("Select option (1-6): ").strip()

            if choice == '1':
                self.gpu_status()
            elif choice == '2':
                self.network_status()
            elif choice == '3':
                self.file_system_status()
            elif choice == '4':
                self.process_monitor()
            elif choice == '5':
                self.configuration_check()
            elif choice == '6':
                break
            else:
                print("Invalid choice")
                input("Press Enter to continue...")

    def diagnostics_menu(self):
        """System diagnostics menu"""
        while True:
            self.core.clear_screen()
            self.core.print_header()
            print(f"\n{self.module_name.upper()} - DIAGNOSTICS")
            print("-" * 25)
            print("  1. Full System Check")
            print("  2. Network Latency Test")
            print("  3. GPU Memory Test")
            print("  4. Performance Test")
            print("  5. Troubleshooting Guide")
            print("  6. Back to Main Menu")
            print("-" * 25)

            choice = input("Select option (1-6): ").strip()

            if choice == '1':
                self.full_system_check()
            elif choice == '2':
                self.network_latency_test()
            elif choice == '3':
                self.gpu_memory_test()
            elif choice == '4':
                self.performance_test()
            elif choice == '5':
                self.troubleshooting_guide()
            elif choice == '6':
                break
            else:
                print("Invalid choice")
                input("Press Enter to continue...")

    def gpu_status(self):
        """Check GPU status - LOCAL AND REMOTE"""
        print("\nGPU Status")
        print("=" * 60)
        
        # LOCAL NVIDIA GPUs
        print("\nLOCAL NVIDIA GPUs:")
        print("-" * 60)
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = line.split(', ')
                    if len(parts) >= 5:
                        idx, name, mem_used, mem_total, util = parts
                        print(f"GPU {idx}: {name}")
                        print(f"  Memory: {mem_used}MB / {mem_total}MB")
                        print(f"  Utilization: {util}%")
            else:
                print("No NVIDIA GPUs detected")
        except FileNotFoundError:
            print("nvidia-smi not found")
        except Exception as e:
            print(f"Error: {e}")
        
        # LOCAL AMD GPUs
        print("\nLOCAL AMD GPUs (ROCm):")
        print("-" * 60)
        try:
            result = subprocess.run(['rocm-smi', '--showid'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(result.stdout)
            else:
                print("No AMD GPUs detected")
        except FileNotFoundError:
            print("rocm-smi not found")
        except Exception as e:
            print(f"Error: {e}")
        
        # REMOTE CLUSTER GPUs
        print("\nREMOTE CLUSTER GPUs:")
        print("-" * 60)
        
        try:
            with open('distributed_config.json', 'r') as f:
                config = json.load(f)
            
            nodes = config.get('nodes', [])
            if nodes:
                for node in nodes:
                    hostname = node.get('hostname', 'unknown')
                    gpu_type = node.get('gpu_type', 'unknown')
                    gpu_count = node.get('gpu_count', 0)
                    
                    print(f"\n{hostname} ({gpu_count}x {gpu_type}):")
                    
                    if 'nvidia' in gpu_type.lower() or 'rtx' in gpu_type.lower():
                        try:
                            ssh_cmd = ['ssh', '-o', 'ConnectTimeout=5', '-o', 'BatchMode=yes',
                                     hostname, 'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits']
                            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=10)
                            
                            if result.returncode == 0:
                                for line in result.stdout.strip().split('\n'):
                                    parts = line.split(', ')
                                    if len(parts) >= 5:
                                        idx, name, mem_used, mem_total, util = parts
                                        print(f"  GPU {idx}: {name} - {mem_used}/{mem_total}MB ({util}% util)")
                            else:
                                print(f"  Unreachable or no GPUs")
                        except Exception as e:
                            print(f"  Error: {e}")
                    
                    elif 'amd' in gpu_type.lower() or 'rx' in gpu_type.lower():
                        try:
                            rocm_setup = 'export HSA_OVERRIDE_GFX_VERSION=10.3.0'
                            ssh_cmd = ['ssh', '-o', 'ConnectTimeout=5', '-o', 'BatchMode=yes',
                                     hostname, f'{rocm_setup} && rocm-smi --showid --showuse']
                            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=10)
                            
                            if result.returncode == 0:
                                for line in result.stdout.split('\n'):
                                    if 'GPU' in line or '%' in line:
                                        print(f"  {line.strip()}")
                            else:
                                print(f"  Unreachable or ROCm not available")
                        except Exception as e:
                            print(f"  Error: {e}")
            else:
                print("No remote nodes configured")
                
        except FileNotFoundError:
            print("No distributed_config.json found")
        except Exception as e:
            print(f"Error: {e}")
        
        input("\nPress Enter to continue...")

    def network_status(self):
        """Check network status"""
        print("\nNetwork Status")
        print("=" * 60)
        
        # Check network interfaces
        print("\nNetwork Interfaces:")
        print("-" * 60)
        try:
            result = subprocess.run(['ip', 'addr', 'show'], capture_output=True, text=True)
            # Parse for active interfaces
            for line in result.stdout.split('\n'):
                if ': ' in line and 'state UP' in line:
                    interface = line.split(':')[1].strip().split('@')[0]
                    print(f"Active: {interface} (UP)")
                elif 'inet ' in line:
                    ip = line.strip().split()[1]
                    print(f"  IP: {ip}")
        except Exception as e:
            print(f"Error: {e}")
        
        # Check connectivity to cluster nodes
        print("\nCluster Connectivity:")
        print("-" * 60)
        
        # Load distributed config if available
        try:
            with open('distributed_config.json', 'r') as f:
                config = json.load(f)
            
            nodes = config.get('nodes', [])
            for node in nodes:
                hostname = node.get('hostname', 'unknown')
                print(f"\nTesting {hostname}...", end=' ')
                
                try:
                    result = subprocess.run(['ping', '-c', '1', '-W', '2', hostname], 
                                          capture_output=True, text=True, timeout=3)
                    if result.returncode == 0:
                        # Extract ping time
                        for line in result.stdout.split('\n'):
                            if 'time=' in line:
                                time_ms = line.split('time=')[1].split()[0]
                                print(f"Reachable ({time_ms})")
                                break
                        else:
                            print("Reachable")
                    else:
                        print("Unreachable")
                except subprocess.TimeoutExpired:
                    print("Timeout")
                except Exception as e:
                    print(f"Error: {e}")
        
        except FileNotFoundError:
            print("No distributed_config.json found")
        except Exception as e:
            print(f"Error loading config: {e}")
        
        # Check open connections
        print("\nActive Connections:")
        print("-" * 60)
        try:
            result = subprocess.run(['ss', '-tunap'], capture_output=True, text=True)
            connections = [line for line in result.stdout.split('\n')[:10] if 'ESTAB' in line]
            if connections:
                for conn in connections[:5]:
                    print(conn[:80])
                if len(connections) > 5:
                    print(f"... and {len(connections)-5} more")
            else:
                print("No established connections")
        except Exception as e:
            print(f"Error: {e}")
        
        input("\nPress Enter to continue...")

    def gpu_memory_test(self):
        """Test GPU memory - LOCAL AND REMOTE"""
        print("\nGPU Memory Test")
        print("-" * 25)
        print("Testing GPU memory allocation and operations...")

        # Test local GPUs (NVIDIA on zeus)
        nvidia_tested = self.test_local_nvidia_memory()

        # Test remote GPUs (both NVIDIA and AMD on cluster nodes)
        remote_tested = self.test_remote_gpu_memory()

        if not nvidia_tested and not remote_tested:
            print("No GPU memory testing completed")

        input("Press Enter to continue...")

    def test_local_nvidia_memory(self):
        """Test local NVIDIA GPU memory"""
        try:
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
            if result.returncode != 0:
                return False

            print("\nLocal NVIDIA GPU Memory Test:")
            print("Testing local NVIDIA GPUs...")

            try:
                import cupy as cp
                print("  Using CuPy for detailed memory testing...")

                gpu_count = cp.cuda.runtime.getDeviceCount()
                print(f"  Found {gpu_count} local NVIDIA GPU(s)")

                for gpu_id in range(min(gpu_count, 3)):
                    try:
                        with cp.cuda.Device(gpu_id):
                            mempool = cp.get_default_memory_pool()
                            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()

                            print(f"\n  GPU {gpu_id}:")
                            print(f"    Total memory: {total_bytes / 1024**3:.1f} GB")
                            print(f"    Free memory: {free_bytes / 1024**3:.1f} GB")

                            test_size_mb = min(100, free_bytes // (1024**2) // 10)
                            if test_size_mb > 0:
                                print(f"    Testing {test_size_mb}MB allocation...")
                                test_array = cp.zeros((test_size_mb * 1024 * 256,), dtype=cp.float32)
                                test_array += 1.0
                                test_array *= 2.0
                                cp.cuda.Stream.null.synchronize()

                                result_sample = float(test_array[0])
                                if result_sample == 2.0:
                                    print("    Memory test: SUCCESS")
                                else:
                                    print(f"    Memory test: FAILED")

                                del test_array
                                mempool.free_all_blocks()
                            else:
                                print("    Insufficient free memory for testing")

                    except Exception as e:
                        print(f"  GPU {gpu_id}: ERROR - {e}")

                return True

            except ImportError:
                print("  CuPy not available, using nvidia-smi...")

                result = subprocess.run([
                    'nvidia-smi',
                    '--query-gpu=index,name,memory.total,memory.free,memory.used',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True)

                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            parts = [p.strip() for p in line.split(',')]
                            if len(parts) >= 5:
                                idx, name, total, free, used = parts
                                print(f"  GPU {idx} ({name}):")
                                print(f"    Memory: {used}/{total} MB used ({free} MB free)")
                                print("    Basic memory query: SUCCESS")

                return True

        except Exception as e:
            print(f"Local NVIDIA GPU test failed: {e}")
            return False

    def test_remote_gpu_memory(self):
        """Test GPU memory on remote cluster nodes"""
        if not os.path.exists(self.core.config_file):
            print("\nRemote GPU Test: Config file not available")
            return False

        try:
            with open(self.core.config_file, 'r') as f:
                config = json.load(f)

            nodes = config.get('nodes', [])
            if not nodes:
                print("\nRemote GPU Test: No nodes configured")
                return False

            print(f"\nRemote GPU Memory Test:")
            print(f"Testing {len(nodes)} cluster nodes...")

            any_success = False

            for i, node in enumerate(nodes, 1):
                hostname = node.get('hostname', 'unknown')
                gpu_type = node.get('gpu_type', 'unknown')
                gpu_count = node.get('gpu_count', 0)

                print(f"\n{i}. Testing {hostname} ({gpu_count}x {gpu_type})...")

                if 'nvidia' in gpu_type.lower() or 'rtx' in gpu_type.lower():
                    # Test NVIDIA GPU on remote node
                    success = self.test_remote_nvidia(hostname)
                    if success:
                        any_success = True

                elif 'amd' in gpu_type.lower() or 'rx' in gpu_type.lower():
                    # Test AMD GPU with ROCm environment
                    success = self.test_remote_amd(hostname)
                    if success:
                        any_success = True

                else:
                    print(f"    Unknown GPU type: {gpu_type}")

            return any_success

        except Exception as e:
            print(f"Remote GPU test error: {e}")
            return False

    def test_remote_nvidia(self, hostname: str):
        """Test NVIDIA GPU on remote node"""
        try:
            # Test basic NVIDIA GPU detection
            ssh_cmd = ['ssh', '-o', 'ConnectTimeout=10', '-o', 'BatchMode=yes',
                      hostname, 'nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv,noheader,nounits']

            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                print(f"    Found {len(lines)} NVIDIA GPU(s)")

                for line in lines:
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 5:
                            idx, name, total, free, used = parts
                            print(f"      GPU {idx}: {name}")
                            print(f"        Memory: {used}/{total} MB ({free} MB free)")

                print(f"    NVIDIA GPU test: SUCCESS")
                return True
            else:
                print(f"    NVIDIA GPU test failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"    Remote NVIDIA test error: {e}")
            return False

    def test_remote_amd(self, hostname: str):
        """Test AMD GPU on remote node with ROCm environment"""
        try:
            print(f"    Setting up ROCm environment for AMD GPUs...")

            # Create the command with proper ROCm environment setup
            rocm_setup = 'export HSA_OVERRIDE_GFX_VERSION=10.3.0 && export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11'

            # Test ROCm availability first
            test_cmd = f'{rocm_setup} && rocm-smi --showid'
            ssh_cmd = ['ssh', '-o', 'ConnectTimeout=10', '-o', 'BatchMode=yes',
                      hostname, test_cmd]

            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                print(f"    ROCm tools not available on {hostname}")
                print(f"    Error: {result.stderr}")
                return False

            print(f"    ROCm environment configured successfully")

            # Get detailed GPU memory information
            memory_cmd = f'{rocm_setup} && rocm-smi --showmeminfo vram'
            ssh_cmd = ['ssh', '-o', 'ConnectTimeout=10', '-o', 'BatchMode=yes',
                      hostname, memory_cmd]

            memory_result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)

            if memory_result.returncode == 0:
                print(f"    VRAM Information:")
                lines = memory_result.stdout.strip().split('\n')
                gpu_count = 0
                for line in lines:
                    if line.strip():
                        print(f"      {line}")
                        if 'GPU' in line:
                            gpu_count += 1

                if gpu_count > 0:
                    print(f"    Detected {gpu_count} AMD GPU(s)")

            # Get utilization information
            util_cmd = f'{rocm_setup} && rocm-smi --showuse'
            ssh_cmd = ['ssh', '-o', 'ConnectTimeout=10', '-o', 'BatchMode=yes',
                      hostname, util_cmd]

            util_result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)

            if util_result.returncode == 0:
                print(f"    GPU Utilization:")
                lines = util_result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip() and ('GPU' in line or '%' in line):
                        print(f"      {line}")

            # Test temperature monitoring
            temp_cmd = f'{rocm_setup} && rocm-smi --showtemp'
            ssh_cmd = ['ssh', '-o', 'ConnectTimeout=10', '-o', 'BatchMode=yes',
                      hostname, temp_cmd]

            temp_result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)

            if temp_result.returncode == 0:
                print(f"    Temperature Information:")
                lines = temp_result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip() and ('GPU' in line or 'C' in line or 'temp' in line.lower()):
                        print(f"      {line}")

            print(f"    AMD GPU test: SUCCESS")
            return True

        except subprocess.TimeoutExpired:
            print(f"    AMD GPU test timed out on {hostname}")
            return False
        except Exception as e:
            print(f"    Remote AMD test error: {e}")
            return False

    def performance_test(self):
        """Run actual performance test"""
        print("\nPerformance Test")
        print("-" * 20)
        print("Running system performance benchmarks...")

        # CPU benchmark
        print("\n1. CPU Performance Test")
        start_time = time.time()
        result = sum(i * i for i in range(1000000))
        cpu_time = time.time() - start_time
        print(f"   CPU test: {cpu_time:.3f} seconds")

        # Memory benchmark
        print("\n2. Memory Performance Test")
        start_time = time.time()
        large_list = [list(range(100)) for _ in range(100000)]
        memory_time = time.time() - start_time
        del large_list
        print(f"   Memory test: {memory_time:.3f} seconds")

        # Overall rating
        total_time = cpu_time + memory_time
        if total_time < 1.0:
            rating = "EXCELLENT"
        elif total_time < 2.0:
            rating = "GOOD"
        else:
            rating = "AVERAGE"

        print(f"\n3. Performance Summary: {rating}")
        input("Press Enter to continue...")

    def network_latency_test(self):
        """Test network latency to cluster nodes"""
        print("\nNetwork Latency Test")
        print("-" * 25)

        if not os.path.exists(self.core.config_file):
            print("Config file not available")
            input("Press Enter to continue...")
            return

        try:
            with open(self.core.config_file, 'r') as f:
                config = json.load(f)

            nodes = config.get('nodes', [])
            print(f"Testing latency to {len(nodes)} nodes...")

            for i, node in enumerate(nodes, 1):
                hostname = node.get('hostname', 'unknown')
                print(f"\n{i}. Testing {hostname}...")

                try:
                    result = subprocess.run(['ping', '-c', '3', hostname],
                                          capture_output=True, text=True, timeout=15)

                    if result.returncode == 0:
                        for line in result.stdout.split('\n'):
                            if 'min/avg/max' in line:
                                print(f"   {line.strip()}")
                                break
                        else:
                            print("   Ping successful")
                    else:
                        print(f"   FAILED: Unreachable")

                except subprocess.TimeoutExpired:
                    print(f"   TIMEOUT")
                except Exception as e:
                    print(f"   ERROR: {e}")

        except Exception as e:
            print(f"Network test error: {e}")

        input("Press Enter to continue...")

    def file_system_status(self):
        """Check file system status"""
        print("\nFile System Status")
        try:
            statvfs = os.statvfs('.')
            total_gb = statvfs.f_blocks * statvfs.f_frsize / (1024**3)
            free_gb = statvfs.f_bfree * statvfs.f_frsize / (1024**3)
            used_pct = (1 - free_gb / total_gb) * 100
            print(f"Disk usage: {used_pct:.1f}% ({free_gb:.1f}GB free)")
        except Exception as e:
            print(f"Error: {e}")
        input("Press Enter to continue...")

    def process_monitor(self):
        """Monitor processes"""
        print("\nProcess Monitor")
        try:
            result = subprocess.run(['pgrep', '-f', 'coordinator'], capture_output=True, text=True)
            pids = result.stdout.strip().split('\n') if result.stdout.strip() else []
            print(f"Coordinator processes: {len(pids)}")
        except:
            print("Process monitoring unavailable")
        input("Press Enter to continue...")

    def configuration_check(self):
        """Check configuration"""
        print("\nConfiguration Check")
        files = [self.core.config_file, self.core.data_file]
        for file in files:
            if os.path.exists(file):
                size = os.path.getsize(file)
                print(f"  {file}: OK ({size} bytes)")
            else:
                print(f"  {file}: MISSING")
        input("Press Enter to continue...")

    def full_system_check(self):
        """Full system check"""
        print("\nFull System Check")
        print("Running comprehensive diagnostics...")
        self.test_local_nvidia_memory()
        self.test_remote_gpu_memory()
        input("Press Enter to continue...")

    def troubleshooting_guide(self):
        """Troubleshooting guide"""
        print("\nTroubleshooting Guide")
        print("=" * 25)
        print("Common Issues:")
        print("1. AMD GPU not detected: Set HSA_OVERRIDE_GFX_VERSION=10.3.0")
        print("2. Remote connection failed: Check SSH keys")
        print("3. ROCm not working: Install rocm-smi-lib package")
        print("4. NVIDIA issues: Check CUDA installation")
        input("Press Enter to continue...")

    def shutdown(self):
        """Module shutdown"""
        print(f"Shutting down {self.module_name}...")
