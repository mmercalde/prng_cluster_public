# TEAM BETA — TECHNICAL INCIDENT REPORT
## rrig6600c Fatal GPU Crash Analysis
**Session:** S163-KARG | **Date:** 2026-04-17/18 | **Prepared by:** Team Alpha (Claude)
**Classification:** BLOCKING — Must resolve before next 300K+ run

---

## 1. Executive Summary

During the S163-KARG validation run (100K seeds x 5 trials, launched 2026-04-17 20:44), rrig6600c experienced a fatal GPU crash at 21:16:45 UTC during **Trial 3** of the Optuna forward sieve (window_size=10, skip_range=(1,243), config W10_O61_midday_S1-243). The rig became completely unreachable and required a manual reboot.

This crash is DISTINCT from the prior S163 FINAL crash mechanism (kernel arg int32/int64 mismatch — KARG patch). The new failure is a GPU virtual memory PTE invalidation occurring while active kernels are mid-execution across multiple GPU devices on the same rig.

A secondary silent crash also occurred on rrig6600b GPU1 at 20:56:11 during **Trial 2** (window_size=17, skip_range=(3,174), config W17_O25_midday+evening_S3-174). rrig6600b GPU1 recovered autonomously without operator intervention.

---

## 2. Run Context & Configuration

| Parameter | Value |
|---|---|
| Session ID | S163-KARG |
| Run launched | 2026-04-17 20:44 (log suffix: 2044) |
| seed_cap_amd | 100,000 |
| window_trials | 5 |
| min_workers | 24 |
| Patches active | [S163-KARG] cp.int32() typing fix + JSON guard (100K limit) |
| Zeus compute mode | DEFAULT (manually set prior to run) |
| Cluster at launch | rrig6600: 8 workers / rrig6600b: 8 workers / rrig6600c: 8 workers |
| Log file | logs/s163_karg_json_100k_5trials_2044.log |

---

## 3. Full Crash Timeline

### 3.1 Optuna Trial Map — Parameter Identification

Trial numbers confirmed from watcher pipeline log. Optuna trial names encode window_size (W) and skip_range (S) directly.

| Trial | Config Name | window / skip_range | Score | Incident |
|---|---|---|---|---|
| Trial 1 | W25_O9_evening_S8-89 | W25 / S8-89 | NEW BEST | None |
| Trial 2 | W17_O25_midday+evening_S3-174 | W17 / S3-174 | 0.00 | rrig6600b GPU1 silent crash @ 20:56 |
| Trial 3 | W10_O61_midday_S1-243 | W10 / S1-243 | 0.00 | **rrig6600c FATAL @ 21:16** |
| Trial 4 | W23_O6_evening_S6-137 | W23 / S6-137 | 0.00 | None |
| Trial 5 | W29_O42_evening_S9-85 | W29 / S9-85 | 0.00 | None |

Evidence chain: rrig6600c worker snapshots at 21:11 show window_size=10, skip_range=(1,243) — matching Trial 3. rrig6600b GPU1 crash at 20:56 shows window_size=17, skip_range=(3,174) — matching Trial 2. Trial numbers confirmed from pipeline log, not inferred.

### 3.2 Primary Event — rrig6600c (Trial 3)

| Wall Time (UTC) | Kernel Time | Event |
|---|---|---|
| 21:11:42–21:11:51 | ~14175s | All 8 GPU workers complete last jobs cleanly (sieve_10433–10734). Workers enter idle wait. |
| 21:11:55 | ~14175s | Snapshot daemon heartbeat — rig alive, all 8 workers healthy. |
| 21:11:51 – 21:16:45 | 14175–14179s | ~4m54s gap. Workers idle, awaiting next chunk dispatch. |
| 21:16:45.860 | 14179.927s | FIRST FAULT: GPU 0000:16:00.0, STATUS=0x00801231, SQC(inst), pid 145822, vmid:8. WALKER_ERROR=0x0. PERMISSION_FAULTS=0x3. |
| 21:16:45.860 | 14179.927s | CONCURRENT FAULT: GPU 0000:06:00.0, STATUS=0x80081071, TCP, pid 145352, vmid:8. WALKER_ERROR=0x0. PERMISSION_FAULTS=0x7. |
| 21:16:46.002 | 14180.070s | ESCALATION: GPU 0000:19:00.0, STATUS=0xFFFFFFFF, unknown(0x1ff), pid 145917, vmid:8. WALKER_ERROR=0x7, PERMISSION_FAULTS=0xf, MAPPING_ERROR=0x1. Terminal MMU state. |
| 21:16:46.052 | 14180.120s | CASCADE: GPU 0000:16:00.0 reaches STATUS=0xFFFFFFFF. Same address 0x0000790de5615000 faulting again — kernel retry loop. |
| 21:16:46 – 21:16:51 | 14180–14185s | Continued cascade across GPUs 06:00.0, 16:00.0, 19:00.0 at 100–200ms intervals. MORE_FAULTS=0x1 throughout. |
| 21:16:54.860 | 14188.928s | GPU 0000:16:00.0: qcm fence wait loop timeout expired. |
| 21:16:56.035 | 14190.103s | GPU 0000:06:00.0: qcm fence wait loop timeout expired. |
| 21:16:56.176 | 14190.243s | FATAL: GPU 0000:06:00.0 device lost from bus! GPU 0000:16:00.0 device lost from bus! Simultaneous. |
| 21:16:26 – 21:16:58 | — | Last snapshot heartbeat to rig unreachable. rrig6600 and rrig6600b remain at 8 workers throughout. |
| ~21:17 | — | Manual reboot by operator. rrig6600c returns online with 8 workers reconnected. |

### 3.3 Secondary Event — rrig6600b GPU1 (Trial 2, Silent Crash)

| Wall Time (UTC) | Event |
|---|---|
| 20:56:11 | GPU1 completes sieve_1443, starts sieve_1447. No DONE entry — worker crashes silently mid-job. No GCVM fault in netconsole. |
| 20:56:11 – 21:06:20 | ~10 minute gap. GPU1 process dead. |
| 21:06:20 | GPU1 reconnects: handshake complete, connected to 192.168.3.127:5600. |
| 21:06:27 | received init — importing sieve_filter. |
| 21:06:40 | sieve_filter imported — gpu=1 rocm=True. Worker operational. |
| 21:06:42 | START job=sieve_579 — back in production. |

Note: rocm_smi captured at 21:16 shows rrig6600b GPU1 with fan=0%, 19W, 0% utilization — consistent with a freshly-recovered worker not yet under load.

---

## 4. Kernel Fault Evidence — Detailed

### 4.1 Involved GPU Devices on rrig6600c

| PCI Device | Worker PID | VMID | First Fault | Terminal State |
|---|---|---|---|---|
| 0000:06:00.0 | 145352 | 8 | 0x80081071 (TCP) | 0xFFFFFFFF → bus loss |
| 0000:16:00.0 | 145822 | 8 | 0x00801231 (SQC) | 0xFFFFFFFF → bus loss |
| 0000:19:00.0 | 145917 | 8 | 0xFFFFFFFF direct | 0xFFFFFFFF (no bus loss logged) |

### 4.2 Stage 1 — Real Page Fault Signatures (NOT Terminal)

The first two faults at 21:16:45.860 are genuine, readable fault codes — not the all-ones register corruption. The GPU MMU was still functional when the fault began.

> **⚠️ DECODE CORRECTION (per TB review):** Earlier versions of this report used the wording "R+X denied" and "page mapped without execute+read permission." This was incorrect. PERMISSION_FAULTS is a bit field: **bit0=PTE not valid, bit1=read not set, bit2=write not set, bit3=execute not set.** This materially changes the interpretation — the PTE became invalid, not that permissions were explicitly stripped from a valid mapping.

| Field | GPU 16:00.0 (pid 145822) | GPU 06:00.0 (pid 145352) |
|---|---|---|
| STATUS | 0x00801231 | 0x80081071 |
| Faulty Client | SQC(inst) — Shader instruction cache | TCP — Texture Cache Processor |
| WALKER_ERROR | 0x0 — Walk SUCCEEDED | 0x0 — Walk SUCCEEDED |
| PERMISSION_FAULTS | **0x3 = bit0(PTE not valid) + bit1(read not set)** | **0x7 = bit0(PTE not valid) + bit1(read) + bit2(write not set)** |
| MAPPING_ERROR | 0x0 | 0x0 |
| Fault address | 0x0000790de5615000 | 0x000000007d322000 |
| Interpretation (TB-corrected) | Page table walk succeeded but resulting PTE was invalid with read bit not set. PTE became invalid while kernel mid-execution — compatible with GPUVM resource lifetime issue, not uniquely characteristic of DRM/KMS reclaim. | Same pattern — PTE invalid, read+write bits not set. |

### 4.3 VMID=8 Across All Three Faulting GPUs

Every fault across all three crashing GPU devices shows vmid:8. Per amdgpu kernel docs: non-zero VMID faults link to user application processes (PASID/process info links VMID to a system process). vmid=0 points to kernel/driver/firmware paths. All three worker PIDs (145352, 145822, 145917) are user-space processes. This observation is consistent with a compute-side GPUVM state issue.

### 4.4 Repeated Fault on Same Address

GPU 0000:16:00.0 faults on address 0x0000790de5615000 and adjacent page 0x0000790de5616000 repeatedly across multiple events spanning several seconds. This is a HIP/ROCm kernel retry loop — the GPU kernel hits the invalid PTE, the runtime attempts recovery, retries the memory access, hits the same dead mapping again, and loops until the qcm fence timeout fires at kernel time 14188.928s.

---

## 5. GPU Power/Thermal State at Crash Moment

### 5.1 rrig6600 — rocm_smi at 21:16

All 8 GPUs: Performance Level=high, SCLK 2630–2740MHz. GPUs 2, 3, 7 active at 57%, 85%, 99% utilization. Temps 51–66°C edge, 52–91°C junction.

### 5.2 rrig6600b — rocm_smi at 21:16

All 8 GPUs: Performance Level=high, SCLK 2590–2735MHz. GPU1 shows 19W, 0% utilization, fan=0% — consistent with newly-recovered worker.

### 5.3 Power Management Hypothesis — Formally Ruled Out

All GPUs on all rigs were Performance Level=high with full SCLK at crash time. Crash occurred mid-active-compute during Trial 3 forward sieve, not during an inter-trial idle window.

---

## 6. Coordinator State at Crash Moment

Coordinator log shows chunks completing at ~30/sec across the cluster at 21:16. Zeus was actively dispatching:

```
[S163-ZEUS] chunk=524 gpu_slot=0 CUDA_VISIBLE_DEVICES=0   (21:16:56.835)
[S163-ZEUS] chunk=549 gpu_slot=1 CUDA_VISIBLE_DEVICES=1   (21:16:56.857)
```

Pipeline was mid Trial 3, forward sieve phase (W10_O61_midday_S1-243). The crash was entirely isolated to rrig6600c.

---

## 7. Root Cause Analysis

### 7.1 KARG Patch Status (TB Position)

The [S163-KARG] patch (cp.int32() wrapping) addressed a known arg-typing bug. This crash **appears distinct** from the prior KARG hypothesis — the fault character is different (prior KARG faults hit unmapped ranges; this crash shows valid PTEs becoming invalid). However, "different fault character" is not the same as "patch proven correct." Formal closure of KARG requires either a controlled A/B or direct evidence that kernel arguments are correctly typed under crash conditions.

**TB wording:** "This crash appears distinct from the earlier KARG hypothesis" — KARG is not yet formally closed.

**Operational decision (Michael, Project Owner):** KARG fix is retained. At 200K–300K seed cap more rigs would crash without it. The fix stays in place while the new root cause is investigated.

### 7.2 Current Root Cause Assessment — "Narrowed, Not Identified" (TB)

Evidence supports: a GPU virtual memory PTE became invalid while kernels were mid-execution across multiple GPU devices on the same rig. Leading hypothesis: CuPy memory lifecycle interference in a shared HIP context. This is a serious lead but not yet demonstrated — still a hypothesis.

Key evidence: Stage 1 faults originate from shader clients (TCP/SQC) on non-zero VMIDs with compute worker PIDs. This points more naturally to the compute/user-space side of GPUVM state than to display-stack or kernel-driver paths.

### 7.3 Open Unknowns

1. **Worker process isolation model** — are 8 workers on rrig6600c separate OS processes or sharing a single HIP context (same VMID)? PIDs 145352, 145822, 145917 are three crashed workers but process topology not yet confirmed.
2. **rrig6600b GPU1 silent crash cause** — died mid-job (sieve_1447) with no kernel MMU fault visible. Python-level exception vs hardware fault unknown.
3. **VMID=8 shared across three devices** — ROCm driver VMID reuse across separate processes, or true shared context?
4. **CuPy pool configuration** — does the default pool act as a cross-device invalidator?

---

## 8. Hypotheses Formally Ruled Out

| Hypothesis | Ruled Out By |
|---|---|
| KARG int32/int64 mismatch (prior cause) | Different fault character — Stage 1 shows valid PTEs, not unmapped ranges. KARG not formally closed but not primary cause here. |
| Idle power management / GFXOFF / D3hot | rocm_smi shows all GPUs perf=high, full SCLK. Crash is mid-active-compute. |
| Thermal throttling | Junction temps within envelope. No throttling flags. |
| ROCm/DKMS regression | amdgpu-dkms 6.12.12 pinned, unchanged since April 5. |
| Kernel version change | 6.8.0-107 on all rigs, unchanged from April 12-13 clean runs. |
| CuPy free_all_blocks() concurrent call | Removed in S162 Option B. Current crash is a different memory path. |
| Inter-trial idle window as trigger | Coordinator log shows active chunk dispatch through crash window. Trial 3 mid-compute. |
| Zeus compute mode (Exclusive_Process) | Fixed prior to run. Zeus dispatching normally throughout. |

---

## 9. Outstanding Unknowns Requiring Investigation

1. **CRITICAL:** Worker process isolation model on rrig6600c — separate OS processes vs shared HIP context. PIDs 145352, 145822, 145917 are three crashed workers — parent/child relationship unknown.
2. Root cause of rrig6600b GPU1 silent crash at 20:56 — worker log ends mid-job, no error output.
3. Whether vmid=8 across three devices is driver VMID reuse or true shared context.
4. CuPy memory pool configuration — cross-device invalidation possible?

---

## 10. Recommended Diagnostic Steps

### 10.1 Immediate — On ser8
```bash
grep -n "sieve_1447|ERROR|Exception|Traceback|DONE|START" \
  ~/crash_dumps/.../rrig6600b/snapshots/pwc_tcp_worker_192_168_3_154_gpu1.log | tail -20
```
Purpose: determine what killed rrig6600b GPU1 at 20:56.

### 10.2 Crash Forensic Daemon — Improvement Deployed (S163-KARG-HB)

Gap identified: daemon did not pull the watcher/pipeline log, making trial identification require post-session archaeology.

Fix deployed: `pipeline_tail.log` added to every Zeus capture. Also added: `capture_context.json`, worker heartbeat JSON/JSONL pulls, process topology (`ps_full.log`, `pstree.log`), per-worker `/proc` details (`worker_proc_details.json`). TB-approved, deployed to all 4 nodes, committed `ad4aa59`.

### 10.3 Immediate — On rrig6600c
```bash
# Confirm worker process isolation model
ssh rrig6600c 'ps auxf | grep pwc_tcp_worker | grep -v grep'
```
Purpose: determine if workers share a Python process or are fully isolated.

### 10.4 Before Next 100K Run
- Confirm all 3 rigs rebooted with snd-power.conf + amdgpu.conf active (DONE — see §11)
- Fix Zeus rc.local for compute mode persistence across reboots
- Commit uncommitted KARG + JSON guard patches on Zeus

### 10.5 Next Validation Run Parameters
- Keep seed_cap_amd=100K until process isolation question resolved
- Run 3 trials before advancing to 5
- After 100K clean pass: advance to 300K → 500K → 1M

---

## 11. Post-Review Finding — GDM/Xorg on Compute Rigs

### 11.1 Discovery

During post-crash infrastructure review on 2026-04-18, GDM3 was found to be the configured default display manager on all three AMD rigs (`/etc/X11/default-display-manager = /usr/sbin/gdm3`). Xorg was confirmed actively running on rrig6600b during the S163-KARG run.

| Evidence Item | Detail |
|---|---|
| Xorg crash dump | `/var/crash/_usr_lib_xorg_Xorg.128.crash` on rrig6600b — timestamped Apr 17 21:19, 3 min after rrig6600c fault cascade began at 21:16 |
| GPU1 silent crash | rrig6600b GPU1 silent crash at 20:56 (Trial 2). Xorg typically claims first available GPU — GPU1 competition is the leading explanation. |
| SDMA timeouts | Post-reboot SDMA ring timeouts on rrig6600b at 82s, 172s, 278s — consistent with GPU state corrupted by Xorg before shutdown. |
| gfxoff kernel param | `amdgpu.gfxoff=0` kernel cmdline parameter silently ignored by amdgpu-dkms 6.12.12 (`unknown parameter 'gfxoff' ignored`). modprobe.d is the only effective method — confirmed active on all 3 rigs post-reboot. |

### 11.2 Project Owner Hypothesis

GDM seat management makes periodic GPU resource probes independent of active Xorg rendering. If GDM or Xorg's DRM subsystem asynchronously reclaimed or remapped GPU virtual address space during compute kernel execution, it could pull PTE entries out from under active sieve kernels — producing the valid-then-invalid PTE cascade observed in Stage 1 faults.

### 11.3 TB Verdict on GDM Hypothesis

**rrig6600b:** GDM/Xorg is a valid first-tier suspect. Xorg crash artifact exists during the run window. The only documented silent worker loss was on this rig. Treat display stack as credible causal or co-causal factor.

**rrig6600c:** GDM is a confounder to test away but NOT the primary explanation per fault signature analysis:

- Non-zero VMID with worker PIDs — per amdgpu kernel docs, non-zero VMID faults link to user application processes. Shader clients (TCP/SQC) are compute-side, not display-controller IP blocks.
- PERMISSION_FAULTS decode — 0x3 = PTE-invalid + read-not-set, not a clean match for DRM/KMS explicit permission-stripping.
- TB conclusion: Stage 1 signature is more consistent with user-space GPUVM/resource-lifetime problem on the compute side than display-stack interference.

### 11.4 Remediation Applied (2026-04-18)

- `sudo systemctl set-default multi-user.target` applied to all 3 rigs — permanent, survives package updates
- gdm3 disabled on all 3 rigs. All confirmed: NO XORG, no SDMA timeouts post-reboot
- `update-initramfs -u` + reboot applied to all 3 rigs with amdgpu.conf and snd-power.conf active

### 11.5 Next Discriminator

100K × 3 trial run with GDM fully removed:
- **If rrig6600c runs clean:** display-stack interference was a sufficient trigger in practice.
- **If rrig6600c crashes again:** GDM ruled out for that rig. Return to compute-side hypotheses (Unknown 1).

---

## 12. Evidence Artefacts

| Artefact | Location on ser8 |
|---|---|
| Crash forensic capture (primary) | `~/crash_dumps/20260417_211658_GCVM_L2_PROTECTION_FAULT_rrig6600c-unrea/` |
| Netconsole log (Zeus) | `.../zeus/netconsole_since_launch.log` |
| Coordinator tail at crash | `.../zeus/coordinator_tail.log` |
| rrig6600 rocm_smi at crash | `.../rrig6600/rocm_smi.log` |
| rrig6600b rocm_smi at crash | `.../rrig6600b/rocm_smi.log` |
| rrig6600c pre-crash worker snapshots | `~/crash_dumps/rrig6600c_snapshots_2116/` |
| rrig6600b GPU1 worker log (silent crash) | `.../rrig6600b/snapshots/pwc_tcp_worker_192_168_3_154_gpu1.log` |
| Session summary (S163-KARG) | `docs/SESSION_CHANGELOG_20260418_S163KARG_FORENSIC.md` |

---

## 13. Sign-Off & Blocking Status

| Item | Status |
|---|---|
| KARG patch (cp.int32) | Appears distinct from this crash. Retained operationally. Not yet formally closed. |
| JSON guard (100K limit) | Confirmed working — no serialization hang this run. |
| Both patches committed? | YES — committed `ad4aa59`, dual-pushed. |
| crash_forensic_daemon.py v2 | TB-approved, deployed to ser8. |
| pwc_worker_service.py heartbeat | TB-approved, deployed all 4 nodes, committed `ad4aa59`. |
| GDM disabled all 3 rigs | DONE — multi-user.target permanent. |
| initramfs updated all 3 rigs | DONE — amdgpu.conf + snd-power.conf active. |
| rrig6600c crash root cause resolved? | NO — narrowed to compute-side GPUVM/resource-lifetime. Unknown 1 still open. |
| Next run cleared? | **CONDITIONALLY CLEARED** — 100K × 3 trial staged validation. GDM removed as confounder. |

---

*Prepared by: Team Alpha (Claude) — Session 2026-04-18*
*TB sign-off: Approved for deployment on next staged validation run*
*GDM/Xorg finding: TB reviewed, verdict incorporated — 2026-04-18*
