# S164 Chat Prompt
**Date:** 2026-04-20+
**HEAD:** `884b424`
**Focus:** Clean production run + Step 2

---

## Cluster State

- Zeus: 2× RTX 3080Ti, `45.32.131.224` / `192.168.3.127`, SSH alias `rzeus`
- rrig6600 (192.168.3.120): 8× RX 6600 — **card0 PCIe riser needs reseat** (`Unknown 63` link)
- rrig6600b (192.168.3.154): 8× RX 6600 — healthy
- rrig6600c (192.168.3.162): 8× RX 6600 — healthy
- All rigs: `amdgpu-dkms 6.12.12` pinned, DPM pin service active, GDM disabled
- Zeus venv: `~/venvs/torch/bin/activate`
- Rig venv: `~/rocm_env/bin/activate`

---

## Immediate Actions (in order)

### 1. Reseat rrig6600 card0 PCIe riser (PHYSICAL)
Before any run — verify after reseat:
```bash
ssh rrig6600 "cat /sys/bus/pci/devices/0000:03:00.0/current_link_speed"
# Should show: 5.0 GT/s PCIe (not "Unknown 63")
```

### 2. Verify cluster health
```bash
for rig in rrig6600 rrig6600b rrig6600c; do
  echo "=== $rig ==="
  ssh $rig "uptime && cat /sys/class/drm/card0/device/power_dpm_force_performance_level"
done
ssh rzeus "ss -tlnp | grep -E '5600|5601' || echo 'ports clear'"
ssh rzeus "ps aux | grep -E 'watcher|window_opt' | grep -v grep || echo 'no stale processes'"
```

### 3. Reset seed coverage and launch production run
```bash
ssh rzeus "cd ~/distributed_prng_analysis && \
  source ~/venvs/torch/bin/activate && \
  python3 reset_seed_coverage.py java_lcg"

ssh rzeus "cd ~/distributed_prng_analysis && \
  source ~/venvs/torch/bin/activate && \
  PYTHONPATH=. nohup python3 agents/watcher_agent.py \
  --run-pipeline --start-step 1 --end-step 1 --force-step 1 \
  --params '{\"min_workers\": 24, \"seed_cap_amd\": 100000, \"window_trials\": 5}' \
  > logs/s164_production_$(date +%H%M).log 2>&1 &
echo PID: \$!"
```

### 4. Monitor
```bash
bash ~/monitor_all.sh
python3 ~/crash_forensic_daemon.py --log-pattern 's164_production'
```

### 5. After run completes — Run Step 2
```bash
ssh rzeus "cd ~/distributed_prng_analysis && \
  source ~/venvs/torch/bin/activate && \
  PYTHONPATH=. nohup python3 agents/watcher_agent.py \
  --run-pipeline --start-step 2 --end-step 2 \
  > logs/s164_step2_$(date +%H%M).log 2>&1 &"
```

---

## Accumulated State

- `bidirectional_survivors_all.npz`: ~23,765 seeds
- `optimal_window_config.json`: W5_O15 (best from last session)
- Coverage: 0 → 2,147,483,648 (2 billion seeds covered)
- Next seed_start will be 2,147,483,648

---

## Known Issues

1. **rrig6600 card0**: `Unknown 63` PCIe link — physical reseat needed
2. **crash_forensic_daemon.py**: 3 bugs (startup false-DOWN, log-caching, dmesg empty) — low priority
3. **Web dashboard**: doesn't auto-start with watcher — manual `python3 web_dashboard.py` needed
4. **Zeus 3080Ti**: slow (local sieve path, not TCP) — TB proposal needed

---

## TB Proposals Needed

1. Zeus TCP worker path — remove `_is_localhost` bypass
2. TCP-PWC job pre-fetch in `pwc_worker_service.py`

---

## Key Commands Reference

```bash
# Kill everything
ssh rzeus "pkill -9 -f watcher_agent; pkill -9 -f window_optimizer"
for rig in rrig6600 rrig6600b rrig6600c; do
  ssh $rig "pkill -9 -f pwc_worker_service 2>/dev/null"
done
ssh rzeus "fuser -k 5600/tcp 2>/dev/null"

# Check DPM
for rig in rrig6600 rrig6600b rrig6600c; do
  echo "=== $rig ==="
  ssh $rig "for card in /sys/class/drm/card[0-7]/device; do
    echo \"\$(basename \$(dirname \$card)): \$(cat \$card/power_dpm_force_performance_level 2>/dev/null)\"
  done"
done

# Reset coverage
ssh rzeus "cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate && python3 reset_seed_coverage.py java_lcg"

# Restart dashboard
ssh rzeus "cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate && nohup python3 web_dashboard.py > logs/dashboard.log 2>&1 &"
```

---

## Repo

- Private: `git@github.com:mmercalde/prng_cluster_project.git`
- Public: `https://github.com/mmercalde/prng_cluster_public`
- Clone: `git clone https://github.com/mmercalde/prng_cluster_public.git`
