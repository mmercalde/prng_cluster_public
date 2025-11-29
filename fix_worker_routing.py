with open('coordinator.py', 'r') as f:
    content = f.read()

# Fix the worker routing to include reverse_sieve
old_routing = """# Choose worker script based on job type
        if payload_json.get('search_type') == 'residue_sieve':
            worker = "sieve_filter.py"
            worker_args = f"--job-file {payload_filename} --gpu-id 0"
        else:
            worker = "distributed_worker.py"
            mining_flag = "--mining-mode" if payload_json.get('mining_mode', False) else ""
            worker_args = f"{payload_filename} --gpu-id {gpu_id} {mining_flag}\""""

new_routing = """# Choose worker script based on job type
        if payload_json.get('search_type') in ['residue_sieve', 'reverse_sieve']:
            worker = "sieve_filter.py"
            worker_args = f"--job-file {payload_filename} --gpu-id 0"
        else:
            worker = "distributed_worker.py"
            mining_flag = "--mining-mode" if payload_json.get('mining_mode', False) else ""
            worker_args = f"{payload_filename} --gpu-id {gpu_id} {mining_flag}\""""

content = content.replace(old_routing, new_routing)

with open('coordinator.py', 'w') as f:
    f.write(content)

print("✅ Fixed worker routing to send reverse_sieve to sieve_filter.py")
