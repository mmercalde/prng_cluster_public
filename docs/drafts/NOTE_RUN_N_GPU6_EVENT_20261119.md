# Run N, rrig6600b gpu6 (0000:16:00.0), 2026-11-19 17:14: event order and comparison with the record (read-only, no verdict)

## Prior record

**Searches run** (2026-11-19, worktree `rig-gpu-fault`, main `7444aea`):
- `git show main:docs/REGISTER_GPU_FAULT_PRIOR_WORK.md` (absent in this checkout), searched for `PROTECTION_FAULT_STATUS|0xFFFFFFFF|lost from bus` and read at `:81`, `:109-115`, `:880-1045`, `:1505-1520`, `:2612-2640`;
- `docs/LEADS.md` L-6/L-7; `docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md` T1.1-T1.2; `docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT_DECODE_20261115.md`; `docs/CLAUDE_CODE_REPORT_GCVM_LOCATED_20261116.md` §1, §9; `docs/TFM_SESSION_HANDOFF_20261118_ADDENDUM.md`; `docs/RIG_POWER_CONTROL_TUYA_REFERENCE.md`;
- `find /home/michael -maxdepth 6 -name '*RESULT_RUN*'`: **no `RESULT_RUN_*` document exists** on VM101.

**Netconsole logs parsed** (main tree, read only):
- `logs/netconsole_all_rigs.log` (2026-05-11 → 11-19);
- `logs/netconsole_listener.log` (2026-04-17 → 11-19);
- `logs/rrig6600c_netconsole.log` (2026-04-01).

**Run evidence:**
- `/home/michael/runN_evidence_20261119` (all files);
- `/home/michael/runI_evidence_20261118` (kernel and netconsole files).

**Labels:**
- The Run N event and its timeline: **NEW**.
- The earlier faults: **ALREADY RECORDED** (cited per row).
- One correction to an earlier answer of Alpha's, in §3.
- No ruling is changed. The owner's position (multiple clean per_run runs: F, G, H on 11-17; L, M on 11-18) is stated, not tested here.

**Not searched:** ser8, the private repository, the claude.ai project, the rigs (not contacted). Nothing was run on any rig.

**Working files** (ignored by git): `logs/runN_note/`.
- `netconsole_bursts.py`: groups every `PROTECTION_FAULT_STATUS` line into per-card bursts.
- `*.tsv`: its output.
- `addr_check.py` and `addr_lowbits.py`: fault addresses against the trace's map events.
- `table_by_pci.md`: the §3 table.

---

## 1. Order of events on 16:00.0 (Q1)

**Clocks.** Three clocks are lined up:
- the kernel log (printk, µs, via netconsole);
- the recorder trace (ftrace, µs, plus the GPU's own interrupt timestamp);
- the worker's `mono`.

The trace clock runs about +0.2019 s ahead of printk. That offset is measured by matching trace interrupt entries #1, #3 and #4 to the first three printed faults (971.298348 ↔ [971.096394], 971.396684 ↔ [971.194657], 971.442796 ↔ [971.240770]). Worker `mono` agrees with the trace clock: its STRIPE_BEGIN at 971.2569 precedes that worker's own trace events at 971.29.

The GPU interrupt timestamp advances at about 100 MHz. That is measured: 224,995 ticks against 2.26 ms of host time. The rate is not documented here.

Times below are on the trace clock, with printk in brackets.

| t (trace s) | Source | Event |
|---|---|---|
| 901.6 → 905.5, 913.2 → 914.5 | worker log | Two earlier stripes on this card compute and send normally (3.88 s and 1.33 s) |
| **971.2569** | worker log | STRIPE_BEGIN for the third stripe, seed_start 3758096384 (phase 1) |
| 971.2936-971.2937 | trace | pid 9399 unmaps a 16 MiB buffer (VA pages `0x7db6c1400-0x7db6c23ff`) |
| 971.2937-971.2961 | trace | sdma0 page-table jobs (`amdgpu_vm_flush ring=sdma0`). SDMA trap interrupts `client_id:8 src_id:224`, the last at **971.296088**, GPU ts 99795901604 |
| 971.2973-971.2975 | trace | pid 9399 maps three new buffers (16, 16 and 8 MiB), the first at the same VA range. `kfd_map_memory_to_gpu_end` at **971.297481** |
| **971.298348** [971.0964] | trace + kernel | **First fault interrupt.** UTCL2 (`client_id:27 src_id:0`), ring 24, VMID 8, pasid 32830, GPU ts **99796126599**: **2.25 ms after the last SDMA trap and 0.87 ms after the last map**. Address `0x7db7d5001000` |
| [971.096418] | kernel | First status read: **`0x00801231`**, SQC (inst) 0x9, WALKER 0x0, PERMISSION 0x3, MAPPING 0x0, RW 0, MORE_FAULTS 1. Preceded by `gmc_v10_0_process_interrupt: 3 callbacks suppressed` |
| GPU ts +0 … +11,024 ticks (~110 µs) | trace | The card writes **134 fault interrupt entries for 121 distinct pages** into the host interrupt ring. SQ interrupts (`client_id:10/11 src_id:239`, SE0/SE1) start at GPU ts **+2,092 ticks (~21 µs) after the first fault** |
| 971.2983 → 971.4435 | trace | The host drains those entries over 145 ms. The gap between entry #2 and #3 is 98 ms |
| **[971.194673]** | kernel | Next status read: **`0xFFFFFFFF`**, client 0x1ff, WALKER 0x7, PERMISSION 0xf, MAPPING 0x1, RW 1. Every later read is the same |
| [971.698145] | kernel | `snd_hda_intel 0000:16:00.1: Unable to change power state from D3hot to D0, device inaccessible` (+0.60 s) |
| [971.864051] | kernel | snd_hda 16:00.1 `CORB reset timeout#2, CORBRP = 65535` |
| [980.195] | kernel | `qcm fence wait loop timeout`; `GPU reset begin!`; **`device lost from bus!`** (+9.1 s); `reset end ret = -19`; `sq_intr: error … err_type 1` lines (printed here, raised at ~971.443) |
| [989.2], [998.2] | kernel | The same reset, -19 cycle. At [998.2]: `SMU: response:0xFFFFFFFF … SetWorkloadMask` |
| 17:15:08, 17:15:39 | kernel | `ring sdma0 timeout, signaled seq=506, emitted seq=510`; sdma0 reset fails; `GPU Recovery Failed: -19` |
| 17:18:29 | kernel | Hung tasks (`ttm_bo_delayed_delete`, `dma_fence_default_wait`) |
| 17:22:50 → 17:24:47 | kernel | `device lost from bus!` every 9 s; 17:23:07 `Failed to quiesce KFD` |
| 17:20:52 | coordinator | `compute_lease_expiry` (owner's statement; console not re-read for this line) |

The worker's HSA runtime reports `Memory access fault by GPU node-7 … on address 0x7d96c145a000. Reason: Page not present or supervisor privilege.` That is one of the 121 fault pages.

**Did anything show the card or its link going away BEFORE the page fault?** **No line in any of the three captures does.** In detail:
- The card was DMA-writing into host memory 2.25 ms before the first fault: SDMA trap interrupts arrived with GPU timestamps that stay consistent with host time.
- It kept writing interrupt entries for ~110 µs after it.
- A register read about 22 µs after the first fault (by printk) returned a valid status (`0x00801231`).
- The first all-ones read came at [971.194673], **between 0.1 and 98 ms after the first fault**. No capture narrows that window.
- **The audio function (16:00.1)** is first reported inaccessible 0.60 s after the fault. It is printed when it tried to leave D3hot. Nothing records its state before then.
- **Neighbours:** from boot to the end of the log, the kernel log has **no line from any other PCI device** on this host after 17:00:10. It shows nothing for the upstream port 00:1d.0, the card's own switch ports 14:00.0/15:00.0, or the other seven cards. The two other hosts' captures (`dmesg_host_121.txt`, `dmesg_host_163.txt`; netconsole) have no fault, reset or link lines tonight.
- **Power state:** `amdgpu.runpm=0` ("runtime pm is manually disabled", boot log). No power-state change of 16:00.0 is logged.
- **Topology** (`lspci_rig6600b_gpu6_hung.txt`): `00:1d.0-[14-16]--00.0-[15]--00.0-[16]--00.0`. 14:00.0 and 15:00.0 are the card's own on-board switch ("Navi 10 XL Downstream Port"), so the chain is root port → card switch → GPU.
- The hung-state lspci shows only the header for 16:00.0 and 15:00.0, with no capability lines. The neighbour 13:00.0 still reads `LnkSta 16GT/s x16`.

**What the logs cannot show:**
- With `pci=noaer`, nothing reports AER. Any latched error status on 00:1d.0, 14:00.0, 15:00.0 or 16:00.0 (Surprise Down, Completion Timeout, receiver or replay errors) was never read, before or after.
- There is no link-status, link-speed or link-width history. No configuration-space read of 16:00.0 or its ports was taken before the fault.
- There is no sub-second power, voltage, current, temperature or PSU telemetry for this card around 17:14. It was not in the evidence folder; whether the 1 Hz `scripts/rig_gpu_recorder.py` ran in Run N is **[UNVERIFIED]**.
- The trace does not record which instructions or kernel arguments the card executed. It also does not record PTE contents as the card saw them, or the TLB/HDP flush steps (the same limit as `CLAUDE_CODE_REPORT_GCVM_LOCATED_20261116.md` §1).
- Whether the first fault interrupt was the card's first abnormal act, or a consequence of an earlier one, cannot be told from these captures.

**Observation, no inference.**
- Of the 121 distinct fault pages, **1** lies inside a VA range pid 9399 mapped in the trace. That is the buffer mapped at 971.297319.
- The first fault page (`0x7db7d5001000`) is not inside any mapped range.
- 42 of the 121 pages share their low 24 bits with that just-mapped buffer's range.

## 2. Comparison with earlier events (Q2)

Boot targets: rrig6600 = `.120` bare / `.121` Proxmox; rrig6600b = `.154` / `.155`; rrig6600c = `.162` / `.163`.

"First status" is the first printed status of that card's burst. "FF" means a `0xFFFFFFFF` read followed in the burst. "Lost" means `device lost from bus` was logged within 600 s.

A host that froze cannot log "lost", so **"no" in that column means "not logged", not "did not happen"**.

| Event | Rig, card, PCI | Phase | First status | Faulty client | FF followed | Lost from bus logged | Source |
|---|---|---|---|---|---|---|---|
| **Run N, 11-19 17:14:12** | rrig6600b gpu6 `16:00.0` (pve) | **1** | **`0x00801231`** | **SQC (inst)** | yes, ≤98 ms | **yes, +9.1 s; host kept logging** | §1 |
| Run 4 attempt 2, 11-15 13:21:29 | rrig6600c gpu6 `16:00.0` (pve) | 4 | `0x00801231` | SQC (inst) | yes, +196 ms | not logged (host hung) | DECODE `:72`; RIG_GPU_FAULT T1.1 |
| same second | rrig6600c gpu1 `06:00.0` | 4 | `0xFFFFFFFF` at first print | unknown 0x1ff | yes | not logged | DECODE `:73` |
| same second | rrig6600 gpu7 `19:00.0` | 4 | `0x00841011` | TCP (0x8), write | yes, +97 ms | not logged | DECODE `:71` |
| Confirm 1, 11-15 18:28:29 | rrig6600c gpu6 `16:00.0` | 4 | `0x00801231` | SQC (inst) | yes | **yes** (18:28:39) | DECODE `:75`; netconsole `:1277` |
| same second | rrig6600c gpu1 `06:00.0` | 4 | `0x80001281` (bit 31, VMID 0) | SQC (inst) | yes | not logged | DECODE `:76` |
| same second | rrig6600 gpu7 `19:00.0` | 4 | `0x00841051` | TCP (0x8) | yes | not logged | DECODE `:74` |
| **Run B**, 11-16 16:56:20 | rrig6600c gpu6 `16:00.0` | 4 | `0x00801031` (TCP read), then SQC (inst) `0x00801231` +50 ms | TCP (0x8) | yes | not logged | GCVM_LOCATED §1 |
| same | rrig6600c gpu1 `06:00.0` | 4 | `0xFFFFFFFF` at first print | 0x1ff | yes | **yes** | GCVM_LOCATED §1 |
| same | rrig6600 gpu7 `19:00.0` | 4 | `0x00841051` | TCP (0x8), write | yes | not logged | GCVM_LOCATED §1 |
| **Run C**, 11-16 17:55:48 | rrig6600c gpu6 `16:00.0` | 4 | `0x00841051` | TCP (0x8), write | yes | not logged | GCVM_LOCATED §9 |
| same | rrig6600c gpu7 `19:00.0` | 4 | `0x00841051` | TCP (0x8), write | yes | not logged | GCVM_LOCATED §9 |
| same | rrig6600 gpu7 `19:00.0` | 4 | `0xFFFFFFFF` at first print | 0x1ff | yes | not logged | GCVM_LOCATED §9 |
| Run 3, 08-22 | rrig6600 + rrig6600c | 4 start | UNOBSERVED | UNOBSERVED | UNOBSERVED | UNOBSERVED | RIG_GPU_FAULT T1.1 |
| **Run I**, 11-18 (killed 16:24) | rrig6600 gpu6 `16:00.0` (pve) | 3 | **no page fault** | — | **no** | **no** | §2.1 below |
| April-May 2026 (bare metal, PWC/ZMQ) | 84 bursts, all three rigs | n/r | see §3 | see §3 | see §3 | see §3 | netconsole_listener |

**2.1 Run I.**
- `dmesg_pve-rig6600_after_stop.txt` and `netconsole_1500_1659.log` have **no page fault, no `0xFFFFFFFF` and no `lost from bus`** for 16:00.0.
- They do have repeated `timeout waiting for kiq fence` on 16:00.0, from 16:25:19 to 16:26:37, after the 16:24 SIGTERM.
- The worker had sat at about 80 % CPU with no system calls while its heartbeats continued (addendum `:26`).
- There was no recorder trace.

**Field by field, Run N against the originals:**

| Field | Run N | Original phase-4 faults (11-15, Runs B and C) | Same? |
|---|---|---|---|
| PCI address | `16:00.0` | `16:00.0` (rrig6600c gpu6, all four events); also `06:00.0`, `19:00.0` | **same address**, other rig |
| Rig | rrig6600b | rrig6600c and rrig6600. rrig6600b never faulted in Proxmox-era captures; it was the survivor at 11-15 and in Runs B and C | **different** |
| Cards and hosts in the event | one card, one host | two or three cards on two hosts, the same second | **different** |
| Phase | 1, the third stripe on that card | 4, first launch after dispatch | **different** |
| First status | `0x00801231` | `0x00801231` at 11-15 13:21 and 18:28; `0x00801031`/`0x00841051` (TCP) in Runs B and C | **same as 11-15**; differs from B/C |
| Faulty client | SQC (inst) | SQC (inst) at 11-15; TCP first in B/C (B's second fault SQC (inst) `0x00801231`) | same as 11-15 |
| Bit fields | WALKER 0 / PERM 0x3 / MAPPING 0 / RW 0 | the same at 11-15 for 16:00.0 | same |
| VMID, pasid, ring | 8, **32830**, 24 | 8, 32830 (11-15), ring 24 | same. The pasid number follows launch order (gpu6's worker), so its equality is expected |
| `N callbacks suppressed` | 3 | 3 (11-15 13:21) | same |
| Time to `0xFFFFFFFF` | ≤98 ms (next printed read) | +97 to +196 ms | same order |
| Timing in the work | 0.87 ms after the stripe's buffer maps, 41 ms after STRIPE_BEGIN | Run C: fault 17-20 ms after the code map; 11-15: ~73 ms after phase-4 stripe creation | similar position (at a launch); different phase |
| Audio function | 16:00.1 inaccessible +0.60 s | 19:00.1 +1.15 s, 06:00.1 +2.29 s (11-15) | same kind |
| Lost from bus | logged +9.1 s | logged once (11-15 18:28, +10 s); otherwise hosts froze | — |
| Host after event | kept logging (hung tasks, periodic resets) | froze (11-15, B, C) | **different** |
| Cache policy | per_run, fresh nonce dirs, cache-join 24/24 (owner) | warm or pre-filled (L-7; Run C pre-filled) | **different** |

## 3. All recorded GCVM fault bursts, by rig and PCI address (Q3)

**Source:** every `PROTECTION_FAULT_STATUS` line in the three netconsole logs: **86 bursts** (a gap of more than 60 s starts a new burst). The 08-22 Run 3 event is UNOBSERVED and Run I has no GCVM line, so neither is counted.

**Caveats:**
- The Proxmox cmdline carries `pci=assign-busses,hpbussize=0x33`, and the bare-metal cmdline is not re-read here. **That the same PCI address is the same physical card across boot targets is [UNVERIFIED].**
- "Lost ≤600 s" is only what was logged.

| Rig | PCI | Bursts (bare / Proxmox) | First | Last | First status (count) | First client (count) | 0xFFFFFFFF in burst | lost-from-bus logged ≤600 s |
|---|---|---|---|---|---|---|---|---|
| rrig6600 | `03:00.0` | 6 (6 / 0) | 2026-04-17 | 2026-05-01 | `0xFFFFFFFF`×2, `0x008012A1`×1, `0x00801031`×1, `0x00801431`×1, `0x008012B1`×1 | SQC×3, unknown×2, TCP×1 | 6/6 | 4/6 |
| rrig6600 | `06:00.0` | 1 (1 / 0) | 2026-04-20 | 2026-04-20 | `0x00841051`×1 | TCP×1 | 0/1 | 0/1 |
| rrig6600 | `09:00.0` | 2 (2 / 0) | 2026-04-20 | 2026-04-20 | `0x00841051`×2 | TCP×2 | 0/2 | 0/2 |
| rrig6600 | `0d:00.0` | 1 (1 / 0) | 2026-04-20 | 2026-04-20 | `0x00841051`×1 | TCP×1 | 0/1 | 0/1 |
| rrig6600 | `10:00.0` | 1 (1 / 0) | 2026-04-20 | 2026-04-20 | `0x00841051`×1 | TCP×1 | 0/1 | 0/1 |
| rrig6600 | `13:00.0` | 2 (2 / 0) | 2026-04-20 | 2026-04-20 | `0x00841051`×2 | TCP×2 | 0/2 | 0/2 |
| rrig6600 | `16:00.0` | 2 (2 / 0) | 2026-04-20 | 2026-04-20 | `0x00841051`×2 | TCP×2 | 0/2 | 0/2 |
| rrig6600 | `19:00.0` | 5 (1 / 4) | 2026-04-20 | 2026-11-16 | `0x00841051`×3, `0x00841011`×1, `0xFFFFFFFF`×1 | TCP×4, unknown×1 | 4/5 | 0/5 |
| rrig6600b | `03:00.0` | 1 (1 / 0) | 2026-04-20 | 2026-04-20 | `0x00841051`×1 | TCP×1 | 0/1 | 0/1 |
| rrig6600b | `06:00.0` | 2 (2 / 0) | 2026-04-20 | 2026-04-20 | `0x00841051`×2 | TCP×2 | 0/2 | 0/2 |
| rrig6600b | `09:00.0` | 1 (1 / 0) | 2026-04-20 | 2026-04-20 | `0x00841051`×1 | TCP×1 | 0/1 | 0/1 |
| rrig6600b | `0d:00.0` | 3 (3 / 0) | 2026-04-17 | 2026-04-20 | `0x00841051`×2, `0x002012B1`×1 | TCP×2, SQC×1 | 0/3 | 0/3 |
| rrig6600b | `10:00.0` | 1 (1 / 0) | 2026-04-17 | 2026-04-17 | `0x006012B1`×1 | SQC×1 | 0/1 | 0/1 |
| rrig6600b | `13:00.0` | 1 (1 / 0) | 2026-04-20 | 2026-04-20 | `0x00841051`×1 | TCP×1 | 0/1 | 0/1 |
| rrig6600b | `16:00.0` | 2 (1 / 1) | 2026-04-20 | **2026-11-19** | `0x00041A40`×1 (SDMA0, one line), `0x00801231`×1 (**Run N**) | SDMA0×1, SQC×1 | 1/2 | 1/2 |
| rrig6600c | `06:00.0` | 17 (14 / 3) | 2026-04-01 | 2026-11-16 | `0xFFFFFFFF`×5, `0x808C1071`×3, `0x008C1071`×3, `0x80081071`×2, `0x00841051`×1, `0x00801231`×1, `0x80001431`×1, `0x80001281`×1 | TCP×9, unknown×5, SQC×3 | 16/17 | 7/17 |
| rrig6600c | `0a:00.0` | 1 (1 / 0) | 2026-04-20 | 2026-04-20 | `0x00841051`×1 | TCP×1 | 0/1 | 0/1 |
| rrig6600c | `0d:00.0` | 1 (1 / 0) | 2026-04-20 | 2026-04-20 | `0x00841051`×1 | TCP×1 | 0/1 | 0/1 |
| rrig6600c | `16:00.0` | 27 (23 / 4) | 2026-04-17 | 2026-11-16 | `0x00801231`×20, `0x00801431`×3, `0x00801281`×2, `0x00801031`×1, `0x00841051`×1 | SQC×25, TCP×2 | 27/27 | 15/27 |
| rrig6600c | `19:00.0` | 9 (8 / 1) | 2026-04-01 | 2026-11-16 | `0x00801431`×3, `0xFFFFFFFF`×2, `0x00841051`×2, `0x00801381`×1, `0x00801231`×1 | SQC×5, unknown×2, TCP×2 | 9/9 | 1/9 |

**Non-GCVM events on the same addresses:** Run I, rrig6600 `16:00.0`, kiq fence timeouts only (§2.1).

**Correction to Alpha's earlier answer.** The Run I lookup said card 16:00.0 on rrig6600 had no prior fault record. The bare-metal netconsole has **two single-line bursts on rrig6600 `16:00.0` on 2026-04-20**. Both are `0x00841051` TCP, in the all-three-rigs same-second events. Neither is followed by `0xFFFFFFFF`. The physical-card caveat above applies.

**Counts:**
- `0x00801231` SQC (inst) was the first status in **23 of 86 bursts**.
- **20** of those 23 were on rrig6600c `16:00.0` (18 bare metal, 2 Proxmox); the others were rrig6600c `06:00.0` and `19:00.0`, and Run N.
- Run N's is the first on rrig6600b, and the first with a per_run cache.
- **No Proxmox-era burst on rrig6600b appears before tonight.**

## 4. Measurements to separate a link or power cause from a software cause (Q4)

Both are read-only, put no context on the GPU, and can run while the cards work.

1. **Latched PCIe error and link status on the whole chain, read before the run and right after a fault.**
   - Chain: `00:1d.0` (and each card's root port), the card switch ports `14:00.0`/`15:00.0`, and `16:00.0`, plus one healthy neighbour as a control.
   - What to read with `setpci`/`lspci -vvv` (config reads only):
     - PCIe Device Status error-detected bits;
     - Link Status (speed, width, DLL Link Active);
     - if the AER capability is present, the AER uncorrectable and correctable status registers.
   - Per the PCIe specification these bits latch when the error occurs even with reporting disabled. `pci=noaer` stops the kernel servicing AER; it does not stop the hardware latching. Whether these root ports and the on-card switch implement the AER capability is **[UNVERIFIED]**; one lspci of a healthy rig settles it.
   - What the readings would show:
     - Surprise Down, DL_Down or receiver/replay errors latched on the root port or the card switch would point at the link.
     - A clean chain with only a later completion timeout would leave the link uninvolved before the fault.
   - Because the bits latch, sampling speed does not matter.
2. **Per-card power and voltage at sub-second rate up to the fault.**
   - Read `power1_average`, `in0_input` (vddgfx) and, if present, `gpu_metrics` (current power, voltages, throttle status), recorded on VM101 with host timestamps.
   - `scripts/rig_gpu_recorder.py` already reads these via sysfs at 1 Hz (`:123-125`) without opening `/dev/kfd`. A faster period for the faulting host would give the last values before the 98 ms window.
   - The cards share one separate server PSU per rig (`RIG_POWER_CONTROL_TUYA_REFERENCE.md:15`), so all eight cards' values in the same sample matter.
   - The owner's Tuya outlet reading does not cover the GPU PSU.
   - What the readings would show: a power cause predicts a sag, spike or throttle change in the last samples, possibly on several cards. A software cause predicts normal values to the last sample.
   - Whether reading these at more than 1 Hz is harmless on this stack is **[UNVERIFIED]**. The SMU has a recorded failure history (register `:400`, `:1961`).
