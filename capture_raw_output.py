#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator

# Monkey-patch to capture raw SSH output
original_execute_remote = MultiGPUCoordinator.execute_remote_job

def debug_execute_remote(self, job, worker):
    import time
    start_time = time.time()
    node = worker.node
    
    # Execute normally
    ssh = self.ssh_pool.get_connection(node.hostname, node.username, node.password)
    eff_timeout = self._effective_timeout(self.job_timeout, job.attempt, node.hostname)
    cmd = self._build_sh_safe_cmd(node, f"job_{job.job_id}.json", job.payload, worker.gpu_id, timeout_s=eff_timeout)
    
    stdin, stdout, stderr = ssh.exec_command(cmd)
    stdout.channel.settimeout(eff_timeout)
    stderr.channel.settimeout(eff_timeout)
    
    output = stdout.read().decode()
    errout = stderr.read().decode()
    
    self.ssh_pool.return_connection(node.hostname, ssh)
    
    print(f"\n{'='*70}")
    print(f"RAW SSH OUTPUT from {node.hostname}")
    print(f"{'='*70}")
    print("STDOUT:")
    print(output)
    print("\nSTDERR:")
    print(errout)
    print(f"{'='*70}\n")
    
    # Parse
    runtime = time.time() - start_time
    json_result = self.parse_json_result(output)
    
    if json_result:
        print(f"Parsed JSON survivors: {json_result.get('survivors', 'NOT FOUND')}")
        import gc
        gc.collect()
        from coordinator import JobResult
        return JobResult(job.job_id, worker.node.hostname, True, json_result, None, runtime)
    else:
        print("Failed to parse JSON!")
        import gc
        gc.collect()
        from coordinator import JobResult
        return JobResult(job.job_id, worker.node.hostname, False, None, "No JSON", runtime)

MultiGPUCoordinator.execute_remote_job = debug_execute_remote

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()

class Args:
    target_file = 'test_26gpu_large.json'
    window_size = 512
    seeds = 20000
    seed_start = 0
    threshold = 0.01
    skip_min = 0
    skip_max = 10
    offset = 0
    prng_type = 'mt19937'
    hybrid = False
    phase2_threshold = 0.50
    gpu_id = 0

args = Args()
coordinator.current_target_file = args.target_file

forward_jobs = coordinator._create_sieve_jobs(args)

for job, worker in forward_jobs:
    if worker.node.hostname == '192.168.3.154' and worker.gpu_id == 2:
        seed_start, seed_end = job.seeds
        if seed_start <= 12345 < seed_end:
            result = coordinator.execute_remote_job(job, worker)
            break

