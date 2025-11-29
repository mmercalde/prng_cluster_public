import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

mismatches = 0
original_execute = coordinator.execute_gpu_job

def _get_skips(job):
    payload = getattr(job, "payload", {}) or {}
    sieve_cfg = payload.get("sieve_config", {}) or {}
    args = getattr(job, "args", None)

    skip_min = payload.get("skip_min")
    skip_max = payload.get("skip_max")
    if skip_min is None or skip_max is None:
        skip_min = sieve_cfg.get("skip_min", skip_min)
        skip_max = sieve_cfg.get("skip_max", skip_max)
    if (skip_min is None or skip_max is None) and args is not None:
        skip_min = getattr(args, "skip_min", skip_min)
        skip_max = getattr(args, "skip_max", skip_max)

    chunk_skip = payload.get("chunk_skip")
    if chunk_skip is None:
        chunk_skip = sieve_cfg.get("chunk_skip")

    return int(skip_min) if skip_min is not None else None, \
           int(skip_max) if skip_max is not None else None, \
           int(chunk_skip) if chunk_skip is not None else None, \
           payload

def check_execute(job, worker):
    global mismatches
    if getattr(job, "search_type", "") == "reverse_sieve":
        skip_min, skip_max, chunk_skip, payload = _get_skips(job)
        cands = (payload or {}).get("candidate_seeds", []) or []

        for c in cands:
            cskip = int(c.get("skip", c.get("best_skip", 0)))
            bad = False
            detail = ""
            if chunk_skip is not None:
                bad = (cskip != chunk_skip)
                detail = f"chunk_skip={chunk_skip}"
            elif (skip_min is not None and skip_max is not None):
                bad = not (skip_min <= cskip <= skip_max)
                detail = f"range=[{skip_min},{skip_max}]"
            else:
                bad = True
                detail = "no_skip_info"

            if bad:
                mismatches += 1
                print(
                    f"⚠️  MISMATCH: node={worker.node.hostname} gpu={worker.gpu_id} "
                    f"job_id={getattr(job,'job_id','?')} {detail} "
                    f"candidate_skip={cskip} seed={c.get('seed')}"
                )

    return original_execute(job, worker)

coordinator.execute_gpu_job = check_execute

optimizer = WindowOptimizer(coordinator, test_seeds=1_000_000)
_ = optimizer.evaluate_window('lcg32', 768, use_all_gpus=True)
print(f"Total mixed-skip candidates detected: {mismatches}")
