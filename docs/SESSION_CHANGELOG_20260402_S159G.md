# Session Changelog — S159G
**Date:** 2026-04-02
**Team:** Alpha (Lead Dev/Implementation)
**Focus:** GPU crash root cause investigation, netconsole deployment, rig standardization

---

## Summary

Extended debugging session focused on identifying and addressing the root cause of rrig6600 and rrig6600c GPU crashes during ZMQ multi-rig runs. Netconsole infrastructure deployed across all nodes. GRUB parameters standardized. Key findings documented for TB review.

---

## S159G-1 — Netconsole Infrastructure Deployment

### Problem
No kernel-level crash visibility on rrig6600 and rrig6600c. Previous crashes left no useful logs because the rig went fully offline before anything could be written.

### Solution
Deployed netconsole UDP streaming to Zeus on all 3 rigs + persistent systemd listener on Zeus.

### Files Created
- `netconsole_listener.py` — UDP listener on Zeus port 6667, writes to `logs/netconsole_all_rigs.log`
- `install_netconsole_zeus.sh` — Installs listener as systemd service on Zeus
- `install_netconsole_rig.sh` — Auto-detects interface, writes `/etc/rc.local`, applies immediately

### Deployment
- Zeus: systemd service `netconsole-listener` — enabled, running, auto-starts on reboot
- rrig6600: `lan0`, 192.168.3.120 → Zeus 192.168.3.127:6667 ✅
- rrig6600b: `enp10s0`, 192.168.3.154 → Zeus 192.168.3.127:6667 ✅
- rrig6600c: `lan0`, 192.168.3.162 → Zeus 192.168.3.127:6667 ✅
- All 3 rigs persist netconsole across reboots via `/etc/rc.local`

---

## S159G-2 — Crash Root Cause Analysis

### Netconsole Capture
First ever real-time kernel crash capture on rrig6600c. Full sequence recorded:

```
gmc_v10_0_process_interrupt: 16 callbacks suppressed
[gfxhub] page fault on 0000:19:00.0 — python pid 3408
[gfxhub] page fault on 0000:06:00.0 — python pid 3003
GCVM_L2_PROTECTION_FAULT_STATUS: 0xFFFFFFFF
Faulty UTCL2 client ID: unknown (0x1ff)
WALKER_ERROR: 0x7
PERMISSION_FAULTS: 0xf
MAPPING_ERROR: 0x1
RW: 0x1
snd_hda_intel: Unable to change power state from D3hot to D0
qcm fence wait loop timeout expired (both GPUs)
The cp might be in an unrecoverable state due to an unsuccessful queues preemption
sq_intr: error (multiple shader engines)
GPU reset begin! → system goes down
```

### Team Beta Analysis
- Fault class: GPU virtual memory / bad address failure
- Initial hypothesis: 5M seeds above validated AMD operating point (2M)
- Updated after 2M also crashed: deeper GPU memory correctness issue

### Team Alpha Analysis
- `GCVM_L2_PROTECTION_FAULT_STATUS: 0xFFFFFFFF` — all protection fault bits set simultaneously
- Address `0x0000000000002000` — near-NULL, consistent with use-after-free
- Two separate Python PIDs faulting on two GPUs simultaneously
- Multiple fault addresses — not solely near-NULL
- Proposed: CuPy `free_all_blocks()` cross-worker race (TB rejected as unproven lead)

### Web Research Findings
- ROCm GitHub issues confirm identical crash pattern on RX 6600 (gfx1032)
- Gentoo ROCm wiki: `iommu=pt` is documented fix for multi-GPU AMD page fault crashes
- `GCVM_L2_PROTECTION_FAULT_STATUS: 0xFFFFFFFF` is known AMD multi-GPU symptom

### Seed Cap Testing
- Run at 5M seeds → crash (both rigs)
- Run at 2M seeds → also crashed (TB hypothesis partially disproven)
- 5M is a trigger/amplifier but not sole cause
- 2M remains the validated operating point

---

## S159G-3 — Package Difference Investigation

### Finding
rrig6600b (stable) has minimal compute-only amdgpu install.
rrig6600 and rrig6600c have full desktop/multimedia amdgpu stack:

**Extra packages on crashing rigs (not on rrig6600b):**
- `amdgpu-lib`, `amdgpu-lib32`, `amdgpu-multimedia`
- `mesa-amdgpu-*` (full mesa stack — va, vdpau, gallium, libgallium)
- `libva-amdgpu-*`, `libwayland-amdgpu-*`
- `libllvm19.1-amdgpu` (full LLVM)
- `xserver-xorg-amdgpu-video-amdgpu`

### History (Claude Error Log)
1. rrig6600c was originally cloned from rrig6600b (stable, minimal install)
2. rrig6600c started crashing → Claude advised installing extra amdgpu packages to match rrig6600 (**wrong**)
3. rrig6600c still crashed → Claude advised cloning rrig6600 onto rrig6600c (**wrong again**)
4. Both decisions made rrig6600c match the crashing configuration

### Correct Path Forward
Remove extra packages from rrig6600 and rrig6600c OR clone rrig6600b's clean state.
Decision: **Clone rrig6600b** (pending — after stability test with iommu=pt).

---

## S159G-4 — GRUB Standardization

### Problem
All 3 rigs had different GRUB configurations — inconsistent parameters, some in `CMDLINE_LINUX_DEFAULT`, some in `CMDLINE_LINUX`.

### Solution
Standardized all 3 rigs to identical GRUB config based on rrig6600b (stable reference) plus `iommu=pt`:

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash pcie_aspm=off pci=noaer amdgpu.ppfeaturemask=0xffff7fff amdgpu.gfxoff=0 amdgpu.runpm=0 amdgpu.aspm=0 pci=assign-busses,hpbussize=0x33 iommu=pt"
GRUB_CMDLINE_LINUX=""
```

### Verification
All 3 rigs confirmed booted with identical kernel cmdline including `iommu=pt vt.handoff=7`.

---

## S159G-5 — /etc/environment Standardization

### Problem
Only rrig6600 had `HSA_OVERRIDE_GFX_VERSION=10.3.0` in `/etc/environment`.
rrig6600b and rrig6600c were missing it (set elsewhere or not at all).

### Solution
Added `HSA_OVERRIDE_GFX_VERSION=10.3.0` to `/etc/environment` on all 3 rigs.

### Verification
```
PATH="/usr/local/sbin:..."
HSA_OVERRIDE_GFX_VERSION=10.3.0
```
Confirmed identical on all 3 rigs.

---

## S159G-6 — Stability Test (PENDING)

### Configuration at Test Time
- seed_cap_amd: 2,000,000
- seed_cap_nvidia: 5,000,000
- ZMQ+SQLite coordinator (--use-zmq-sqlite)
- iommu=pt active on all 3 rigs
- Netconsole monitoring all 3 rigs in real time
- GRUB and /etc/environment identical across all 3

### Log File
`logs/zmq_iommu_test_v1.log`

### Success Criteria
- All 3 rigs stay online for full run completion
- No gfxhub page fault messages in netconsole log
- 537/537 chunks completed

### If Test Fails
Clone rrig6600b's ROCm environment to rrig6600 and rrig6600c:
- Use USB-SATA drive to copy tarball
- Remove extra amdgpu packages from crashing rigs
- Re-run stability test

---

## Key Invariants Confirmed This Session
- rrig6600b is the gold standard reference rig — never crashed
- rrig6600b interface: `enp10s0` (differs from rrig6600/rrig6600c which use `lan0`)
- Zeus MAC: `0c:9d:92:c3:e5:a2`, Zeus LAN IP: `192.168.3.127`
- Netconsole Zeus listener port: 6667, rig local port: 6665
- All 3 rigs: kernel `6.8.0-106-generic`, ROCm `6.4.3`
- AppArmor: active on all 3 rigs in unconfined mode — identical, not a variable

---

## TODO Carried Forward
1. **Stability test result** — pass/fail determines next action
2. **If fail: Clone rrig6600b** ROCm packages to rrig6600 and rrig6600c
3. **apt update (kernel held)** on all 3 rigs after clone
4. **PWC TCP Gate 1** — 1-rig, 1-GPU validation (deferred pending stability)
5. **Selfplay NN fix** — remove forbidden guard in `inner_episode_trainer.py`
6. **S110 root cleanup** — 884 stray files
7. **Write netconsole setup to rc.local** — make persistent ✅ (done this session)

---

## Commits This Session
- None yet — pending stability test result before committing

---

*Session S159G — 2026-04-02 — Team Alpha*
