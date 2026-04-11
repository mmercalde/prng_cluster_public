# S163 Session Changelog
**Date:** 2026-04-07 through 2026-04-10
**Author:** Team Alpha (Claude)
**Status:** OPEN — cluster stabilization in progress
**Git HEAD at session start:** `aa48ee4` (S162 close)
**Git HEAD at session end:** `0a316b8`

---

## Summary

S163 opened with the TCP transport benchmark as the primary goal. The session was derailed by a package removal disaster that destabilized rrig6600 and rrig6600c, leading to multiple days of crash investigation, failed recovery attempts, and ultimately a decision to restore from a full disk clone of rrig6600b.

---

## Completed Work

### [DONE] Dashboard Trial Number Fix
**File:** `persistent_worker_coordinator.py`
**Commit:** `dddad10`
`update_trial_stats()` now called at the START of each trial with `trial_num` set, so the Live Trial Data card on the dashboard always shows the correct running trial number.

### [DONE] torch.cuda.synchronize Performance Fix — `sieve_filter.py`
**Commit:** `0a316b8`
`torch.cuda.synchronize()` and `torch.cuda.empty_cache()` were firing unconditionally on every Zeus chunk, reducing Zeus throughput from ~466K s/s to ~13K s/s. Gated both calls behind `S163_MEM_DEBUG=1` env var. Zeus throughput restored.

### [DONE] S163_MEM_DEBUG Propagation Fix — `persistent_worker_coordinator.py`
**Status:** Patched and tested locally, NOT YET DEPLOYED
The `S163_MEM_DEBUG` env var was not propagating to remote AMD worker SSH spawn commands. Fix adds the var to the `rocm_env` string in `_spawn_worker()` when set in parent:
```python
] + ([f"S163_MEM_DEBUG={os.environ['S163_MEM_DEBUG']}"]
     if os.environ.get('S163_MEM_DEBUG') else []))
```
Verified internally. Deploy after cluster stability confirmed.

### [DONE] NPZ Accumulator Recovery
`bidirectional_survivors_all.npz.flush.tmp.npz` had 4,424 seeds from Apr 5. Restored as canonical accumulator. Committed to git.

### [DONE] Kernel/Package Holds
All three rigs hold: `amdgpu-dkms`, `amdgpu-install`, `linux-firmware`, `linux-generic-hwe-22.04`, `linux-headers-generic`, `linux-headers-generic-hwe-22.04`, `linux-image-generic`, `linux-image-generic-hwe-22.04`.

### [DONE] GRUB Kernel Parameters Added (2026-04-10)
Added to all three rigs before existing "quiet" in `GRUB_CMDLINE_LINUX_DEFAULT`:
```
amdgpu.gpu_recovery=1 amdgpu.sched_jobs=256
```
**Status:** Not yet proven to help or hurt. May need reverting.

---

## Package Removal Disaster

### What Happened
1. Attempted to remove multimedia packages from rrig6600 and rrig6600c to match rrig6600b's minimal install
2. rrig6600 package removal broke NIC — rig crash-looped on boot
3. Emergency reinstall of `amdgpu-lib amdgpu-lib32 amdgpu-multimedia` on rrig6600 via SSH during boot window — restored NIC
4. rrig6600c — multimedia packages removed, `autoremove` also removed 24 i386 orphan libs
5. Both rigs became unstable — crashing during 200K seed cap runs that were previously stable

### Current Package State (post-reinstall)
- rrig6600b: 11 packages (stable reference, never touched)
- rrig6600: ~45 packages (multimedia reinstalled)
- rrig6600c: ~45 packages (multimedia reinstalled)

### Crash Pattern
`GCVM_L2_PROTECTION_FAULT_STATUS:0xFFFFFFFF` on rrig6600 and rrig6600c during reverse sieve. `ih ring buffer overflow` cascade leading to `device lost from bus`. Occurring consistently during 200K seed cap runs that were stable before the package changes.

---

## Crash Investigation Findings

### GCVM_L2_PROTECTION_FAULT_STATUS:0xFFFFFFFF
- Indicates GPU register itself is unreadable — device physically dropped off PCIe bus
- NOT an OOM error (OOM would show `HIP_ERROR_OUT_OF_MEMORY`)
- `ih ring buffer overflow` is the precursor — interrupt handler ring saturates under concurrent 8-GPU load
- Known ROCm/amdgpu issue with multiple concurrent GPU workloads on RX 6600 (gfx1030)

### What Was Tried
- Multimedia package reinstall on both rigs ❌ — did not fix
- Multiple reboots and power cycles ❌ — did not fix
- `amdgpu.gpu_recovery=1 amdgpu.sched_jobs=256` grub params ❌ — did not fix (run crashed with 3362/5369 chunks failed)

### Root Cause Assessment
The repeated hard crashes and power cycles today corrupted the GPU driver/firmware state on rrig6600 and rrig6600c in a way that persists across reboots. The systems were stable at 200K before the package changes.

---

## Clone Restore — COMPLETED 2026-04-10

### rrig6600 Restored
- Full disk image created on rrig6600b: `~/rig6600b_clone_20260410.img.gz` (126G compressed)
- Clone verified (MBR/GPT intact, gzip valid)
- Restored to rrig6600 via USB-NVME adapter — full 238.5G write at ~70 MB/s
- Network config restored from saved files: IP 192.168.3.120, MAC f4:b5:20:3f:6e:86, hostname rig-6600
- SSH host keys restored — no host key warning after boot
- Netconsole rc.local fixed: `enp10s0` → `lan0`, local_ip set to `192.168.3.120`
- Result: 8/8 GPUs healthy, netconsole working, correct IP ✅

### rrig6600c Restored
- Same clone used (rrig6600b is golden reference)
- Network config read from mounted disk before restore: IP 192.168.3.162, MAC f4:b5:20:42:4d:63, hostname rig-6600c
- All SSH host keys, sshd_config, netplan, hostname, machine-id saved to Zeus then restored post-clone
- Netconsole rc.local fixed: `enp10s0` → `lan0`, local_ip set to `192.168.3.162`
- Result: 8/8 GPUs healthy, netconsole working, correct IP ✅

### Post-Restore State
- All three rigs: ROCm 6.4.3-128, identical grub params, identical packages
- Project files rsync'd from Zeus to rrig6600 and rrig6600c
- Clone image preserved on USB SATA drive connected to SER8 (`/dev/sda2`, raw gzip)
- Clone image deleted from rrig6600b to free 126G disk space

## Pending Items

### [PENDING — IMMEDIATE] 200K validation run
Full disk image being created on rrig6600b:
```
~/rig6600b_clone_20260410.img.gz
```
PID: 289905, running via nohup. ~90 minute process.
Restore command (on target with USB-SATA adapter):
```bash
gunzip -c rig6600b_clone_20260410.img.gz | sudo dd of=/dev/sda bs=4M status=progress
```
After restore: `sudo hostnamectl set-hostname rig-6600`

### [PENDING] Restore rrig6600c from same clone
After rrig6600 verified stable, repeat for rrig6600c.
After restore: `sudo hostnamectl set-hostname rig-6600c`

### [PENDING] Deploy S163_MEM_DEBUG propagation fix
```bash
scp ~/Downloads/persistent_worker_coordinator.py rzeus:~/distributed_prng_analysis/persistent_worker_coordinator.py
ssh rzeus "cd ~/distributed_prng_analysis && git add persistent_worker_coordinator.py && git commit -m 'fix(s163): propagate S163_MEM_DEBUG env var to remote AMD workers via SSH spawn command' && git push origin main && git push public main"
```
Deploy AFTER cluster stability confirmed.

### [PENDING] Seed cap ladder test
Once 200K stable for 3 full clean trials: step to 350K → 500K → 750K → 1M.

### [PENDING] TCP transport benchmark
Original S163 goal — never started due to cluster instability.

### [PENDING] Session changelog commit to git
This file needs committing after cluster restored.

### [PENDING] TB Postmortem on package removal disaster
Formal TB proposal required before any future package changes on AMD rigs.

---

## Key Invariants Reinforced This Session
- **Never touch packages on AMD rigs without TB proposal and rollback plan**
- **rrig6600b is the stable reference — never touch its packages**
- **Always verify grub changes on one rig before applying to all three**
- **Fix forward, restore from clone when fix-forward fails**
- **Dual push rule: `git push origin main && git push public main`**

---

## Performance Reference
- 200K seed cap: ~2.8M seeds/sec cluster throughput (when stable)
- Trial 1 of last run completed successfully — 300 bidirectional survivors
- Trials 2 and 3 failed due to crash — 3362/5369 chunks failed, 0 survivors

---

## Git Commits This Session
| Commit | Description |
|--------|-------------|
| `dddad10` | fix(dashboard): update trial_num at trial start |
| `0a316b8` | fix(s163): gate torch.cuda.synchronize behind S163_MEM_DEBUG |
| Earlier | NPZ accumulator recovery |
