# Team Beta Submission — S159G
## rrig6600 GPU Crash Root Cause Investigation
**Date:** 2026-04-02  
**Submitted by:** Team Alpha  
**Priority:** P0 — Blocking production runs  

---

## Executive Summary

rrig6600 (192.168.3.120) crashes consistently during ZMQ multi-rig runs. rrig6600b (192.168.3.154) has never crashed. rrig6600c (192.168.3.162) survived this session's run. The crash is a GPU virtual memory fault leading to unrecoverable amdgpu queue state. Multiple fixes have been attempted without resolving the root cause.

**Positive finding this session:** ZMQ SQLite lease expiry mechanism successfully recovered the run when rrig6600 rebooted mid-run. rrig6600 rejoined automatically and the trial completed. This confirms the coordinator architecture is crash-resilient — but does not fix the underlying crash.

---

## Crash Sequence (Netconsole Capture — rrig6600, 2026-04-02 17:49)

```
[ 6511.764025] amdgpu 0000:03:00.0: [gfxhub] page fault
  address: 0x0000723cae6bd000
  GCVM_L2_PROTECTION_FAULT_STATUS: 0x00801231
  Faulty UTCL2 client ID: SQC (inst) (0x9)
  WALKER_ERROR: 0x0   ← partial mapping exists
  PERMISSION_FAULTS: 0x3
  MAPPING_ERROR: 0x0

[ 6511.860299] amdgpu 0000:03:00.0: [gfxhub] page fault  ← escalation
  GCVM_L2_PROTECTION_FAULT_STATUS: 0xFFFFFFFF  ← all bits set
  Faulty UTCL2 client ID: unknown (0x1ff)
  WALKER_ERROR: 0x7   ← complete failure
  PERMISSION_FAULTS: 0xf
  MAPPING_ERROR: 0x1
  RW: 0x1

[repeats every ~100ms on same and neighboring addresses]

[ 6513.063839] snd_hda_intel 0000:03:00.1: Unable to change power state D3hot→D0
[ 6520.767838] amdgpu 0000:03:00.0: qcm fence wait loop timeout expired
[ 6520.767845] amdgpu 0000:03:00.0: The cp might be in an unrecoverable state
[ 6520.767891] amdgpu: sq_intr: error, se 0, data 0x80000, err_type 1
[system goes offline]
```

**Post-reboot netconsole (18:04):**
```
[  63.188303] pci_bus 0000:03: busn_res: [bus 03] end is updated to 03
[multiple PCIe bus reassignments across all slots]
[  64.645764] r8169 0000:0a:00.0 lan0: NETDEV WATCHDOG: CPU:1: transmit queue 0 timed out 5644ms
[  73.666921] r8169 0000:0a:00.0 lan0: NETDEV WATCHDOG: CPU:1: transmit queue 0 timed out 5304ms
```

**Additional netconsole findings (rrig6600b and rrig6600c):**
```
[18:29] 192.168.3.162: perf: interrupt took too long (2503 > 2500), lowering perf_event_max_sample_rate to 79000
[18:29] 192.168.3.154: perf: interrupt took too long (2502 > 2500), lowering perf_event_max_sample_rate to 79000
```
Both surviving rigs showing perf interrupt pressure — not crashing but under load.

---

## Key Observations

### Crash Characteristics
- **Single GPU only:** `0000:03:00.0` — only one physical GPU involved
- **Single PID:** python pid 5153 — one worker process
- **NOT a multi-GPU race:** previous crashes appeared multi-GPU but were same pattern
- **WALKER_ERROR escalation:** 0x0 → 0x7 in ~100ms — partial mapping exists then completely lost
- **Fault address repeats:** GPU keeps retrying the same dead address
- **Audio device power failure:** `0000:03:00.1` on same PCIe slot loses power state

### Post-Crash PCIe Behavior (NEW — Critical)
After reboot, `pci=assign-busses,hpbussize=0x33` GRUB parameter triggers PCIe bus reassignment across ALL slots (03, 06, 09, 0a, 0d, 10, 13, 16, 19). Immediately after, the NIC (`lan0` on `0000:0a:00.0`) has transmit queue timeouts twice. This indicates PCIe bus instability on rrig6600 specifically during/after the reassignment cycle. The same GRUB parameter is on all 3 rigs but only rrig6600 shows this NIC watchdog behavior post-crash.

---

## Environment Analysis

### Worker Process Environment (What workers actually see)
```
HSA_OVERRIDE_GFX_VERSION=10.3.0   ← from /etc/environment ✅
```

### What workers SHOULD see but DON'T
```
HSA_ENABLE_SDMA=0        ← in .bashrc ONLY — not reaching systemd-run workers
HSA_ENABLE_RUNTIME_POWER_MGMT=0  ← in rocm_env/bin/activate ONLY
```

**Root issue:** systemd-run workers do not source `.bashrc` or activate the venv. Only `/etc/environment` is inherited. `HSA_ENABLE_SDMA=0` — which disables the SDMA engine known to cause ROCm memory faults — is never applied to the worker processes.

### Per-Rig Environment Differences
| Variable | rrig6600 | rrig6600b | rrig6600c |
|----------|----------|-----------|-----------|
| `/etc/environment` HSA_OVERRIDE_GFX_VERSION | ✅ | ✅ | ✅ |
| `/etc/environment` HSA_ENABLE_SDMA | ❌ | ❌ | ❌ |
| `.bashrc` HSA_ENABLE_SDMA=0 | ✅ | ✅ | ✅ |
| `rocm_env/bin/activate` HSA_ENABLE_RUNTIME_POWER_MGMT=0 | ✅ | ❌ | ✅ |
| `.bashrc` HIP_PATH=/opt/rocm/hip | ❌ | ✅ | ❌ |

---

## Fixes Attempted This Session (All Failed to Prevent Crash)

| Fix | Applied | Result |
|-----|---------|--------|
| seed_cap_amd 5M → 2M | ✅ | Still crashed at 2M |
| iommu=pt kernel parameter | ✅ all rigs | Still crashed |
| GRUB standardization | ✅ all rigs | Still crashed |
| /etc/environment standardization | ✅ all rigs | Still crashed |
| HSA_ENABLE_SDMA=0 in coordinator env | ✅ (partial) | Still crashed |
| HSA_ENABLE_RUNTIME_POWER_MGMT=0 | ✅ coordinator | Still crashed |

---

## Internet Research Findings

1. **`HSA_ENABLE_SDMA=0`** — Widely recommended fix for ROCm GPU memory faults. Disables SDMA DMA engine which has known bugs in memory isolation. **Not currently reaching worker processes.**

2. **linux-firmware update** — Medium article (Feb 2026) identifies MES component firmware bugs as primary cause of ROCm memory faults. rrig6600 has `linux-firmware 3.40` upgradable to `3.41`. We held the firmware update pending TB guidance.

3. **RX 6600 specific issue** — ROCm GitHub issue #5238 shows identical crash pattern on gfx1032 (RX 6600 chip). This is a known-affected GPU.

4. **`GCVM_L2_PROTECTION_FAULT_STATUS: 0xFFFFFFFF`** — Multiple ROCm issues confirm this "all bits set" pattern is a total GPU VM failure, distinct from normal page fault handling.

---

## Surviving Rigs Analysis

| Rig | Status | Notable |
|-----|--------|---------|
| rrig6600b (192.168.3.154) | ✅ Never crashed | HIP_PATH set, no HSA_ENABLE_RUNTIME_POWER_MGMT=0 in activate |
| rrig6600c (192.168.3.162) | ✅ Survived this run | Previously crashed in earlier sessions |
| rrig6600 (192.168.3.120) | ❌ Crashed again | Was originally stable rig |

---

## ZMQ Recovery Finding (Positive)

When rrig6600 crashed and rebooted mid-run, the ZMQ SQLite coordinator automatically:
1. Detected expired leases on chunks claimed by rrig6600 workers
2. Redistributed those chunks to surviving rrig6600b and rrig6600c workers
3. Accepted rrig6600 back when it rejoined after reboot
4. Completed the full trial without data loss

**Forward sieve completed:** 126,610,730 survivors from 1,073,741,824 seeds.
**Reverse sieve:** Running at time of report submission.

This confirms ZMQ architecture is crash-resilient but does not resolve the crash itself.

---

## Questions for Team Beta

1. Should `HSA_ENABLE_SDMA=0` be added to `/etc/environment` on all rigs as the immediate next fix?
2. Should `linux-firmware` be updated on rrig6600 and rrig6600c (3.40 → 3.41) while holding the kernel?
3. Is the post-crash PCIe bus reassignment (`pci=assign-busses,hpbussize=0x33`) a suspect? Should it be removed from rrig6600's GRUB to test?
4. Should `HIP_PATH=/opt/rocm/hip` be added to rrig6600 and rrig6600c `.bashrc` to match rrig6600b?
5. Is a full clone of rrig6600b to rrig6600 still warranted given the persistent crash?

---

## Proposed Immediate Actions (Pending TB Ruling)

**P0 — Environment fix (low risk, high impact):**
```bash
# Add to /etc/environment on all rigs
HSA_ENABLE_SDMA=0
```

**P1 — firmware update (medium risk):**
```bash
sudo apt-mark hold linux-image-generic linux-headers-generic linux-generic
sudo apt update && sudo apt upgrade linux-firmware -y
sudo apt-mark unhold linux-image-generic linux-headers-generic linux-generic
sudo reboot
```

**P2 — GRUB investigation:**
Remove `pci=assign-busses,hpbussize=0x33` from rrig6600 GRUB only, test if PCIe stability improves.

**P3 — Full clone rrig6600b → rrig6600 (if all else fails)**

---

*Submitted S159G — 2026-04-02 — Team Alpha*
