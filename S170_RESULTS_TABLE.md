# S170 Stability Curve Results

**Branch:** s167-clean  
**HEAD:** a6cd55e  
**Stable baseline:** a6bc546  
**S168 jitter:** PRNG_PWC_FIRST_ASSIGN_JITTER_SEC=3  
**S169 pacing:** PRNG_PWC_PER_WORKER_MIN_GAP_SEC=0.02  

---

| cap | trials | result | elapsed | avg seeds/s | netconsole | script_write_failed | crashed_rig | notes |
|---:|---:|:---:|:---:|---:|:---:|:---:|:---:|:---|
| 100k | 5 | ? | ? | ? | ? | ? | none | |
| 150k | 5 | ? | ? | ? | ? | ? | ? | |
| 200k | 5 | ? | ? | ? | ? | ? | ? | |

---

## Pass/Fail Criteria

| Grade | Conditions |
|:---:|:---|
| PASS | All trials complete, netconsole clean, no script write failures |
| WARN | Completes but netconsole has amdgpu/KFD faults **OR** script write failures |
| FAIL | Any rig crash / reset / unreachable |

---

## Failure Modes — Track Separately

**1. GPU/kernel faults (netconsole)**
- `GCVM_L2_PROTECTION_FAULT`
- `qcm fence timeout`
- `amdgpu reset`
- `KFD fault`
- `gfxhub / SQC (inst)` — rrig6600c signature

**2. Transport/I/O faults**
- `[PWC-TCP] 192.168.3.120:GPU* script write failed`
- Worker disconnect/reconnect storm
- Startup timeout / min_workers not reached

---

## Rig Notes at Test Start

| Rig | IP | GPU params | State |
|:---|:---|:---|:---|
| rrig6600 | 192.168.3.120 | cwsr_enable=0 mcbp=0 | ✅ |
| rrig6600b | 192.168.3.154 | stock | ✅ |
| rrig6600c | 192.168.3.162 | stock | ✅ (cwsr effect uncertain — under investigation) |

---

## Execution Sequence

```bash
./s170_cluster_health.sh

./s170_run_stability_curve.sh 100000 5
# wait for completion
./s170_check_latest.sh
./s170_cluster_health.sh
# fill 100k row

./s170_run_stability_curve.sh 150000 5
# wait for completion
./s170_check_latest.sh
./s170_cluster_health.sh
# fill 150k row

./s170_run_stability_curve.sh 200000 5
# wait for completion
./s170_check_latest.sh
./s170_cluster_health.sh
# fill 200k row

./s170_extract_results_table.sh
```

**Do not move to DPM harness until all three rows are classified PASS/WARN/FAIL.**
