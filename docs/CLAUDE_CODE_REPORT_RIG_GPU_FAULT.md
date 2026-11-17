# CLAUDE CODE REPORT — rig GPU fault (Run 4 attempt 2, 2026-11-15)

**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_RIG_GPU_FAULT_20261115.md` (saved verbatim).
**Base:** worktree `rig-gpu-fault` at `3579088`, clean apart from this report, the brief,
`scripts/rig_gpu_recorder.py` (new) and `docs/LEADS.md` (L-6 added, L-3 correction appended).
**Status (R2):** T1, T2 done. T3 built, tested on idle rigs, revised per Beta R1 Q5. T4 redesigned
per Beta R1 Q6/Q7/Q9 and Alpha's R2 direction, **nothing run**. T5: R1 reviewed (four LIVE
objections), R2 sent. **No root cause is claimed.**

**R2 direction (Alpha, 2026-11-15):** make the R1 fixes as listed and send R2. Add **(g) memory
handling at the phase change** with a serialised-launch diagnostic and a repeated single-card
phase-3→phase-4 hand-over, citing S162 Option B. Drop every arm already tried with a recorded
result, using the prior-work register when it exists. Order software-only and least invasive
first. Put the owner's manual steps in one list. Stop after Beta R2 with the first test in two
sentences. To keep Alpha's letter, the cache candidate is now **(h)**.

**Gate-1 decisions (Alpha, 2026-11-15):** (1) L-3 correction adopted. (2) T4 states the single-rig
false-pass risk up front and includes a fleet-scale variant, and discriminates (a) DPM level,
(b) staggered start, (c) worker HSA env, (d) the two repeat rrig6600c cards held out — cheapest
first, one change at a time. (3) The recorder also listens on UDP 6668. (4) T3 then T5. Alpha
withdrew the clock-profile lead after checking S165. **Owner addition, 14:43 PST:** include what
mining operators report for RX 6600-class rigs, verified, and add (e) the riser path of the three
faulting cards with a hand swap test, and (f) PCIe atomics support and what ROCm does without
them. Done in T4.4–T4.7.

**Actions taken.** Read-only SSH only (**one exception, R3: an empty `/dev/shm/.none` created and deleted on the `.164` CT by a leftover in a command; disclosed in T4.11 (1)**): root to `.121/.155/.163`, michael to `.122/.156/.164`.
Commands used: `cat` of sysfs/procfs/configs, `lspci -vv`, `systemctl cat/is-active`, `journalctl`,
`dpkg-query`, `modinfo`, `stat`, a KFD-topology read and a package-metadata read. Nothing was
written on any host or CT. No GPU was opened: no HIP/HSA init, no `rocminfo`, no `rocm-smi`, no
cupy import. The bundle ledger *copy* was read with `immutable=1`. The live ledger was not touched.
Raw captures are in `logs/rig_gpu_fault_20261115/` (ignored).

---

## Null results and corrections first

1. **Same recorded trial inputs, same host boot, different outcomes** (narrowed per Beta R1 Q3). This morning's
   `distributed_config_t1_57dc50fa` and this afternoon's `distributed_config_t1_ab6318d4` ran in
   the **same host boot session**. Both hosts booted at 08:27 (`journalctl --list-boots`, boot −1).
   The two runs had **identical trial parameters**: W12, anchor 25, `generator_phase` 0, midday,
   skip 6–99, thresholds 0.71/0.47 (bundle ledger `trial_context`). The idle time before phase 4
   was also the same: about 218 s in the morning against 202–221 s in the afternoon (§T1.3).
   The morning run passed and the afternoon run failed. Per-run effective host settings and
   process environments were **not** snapshotted. That no static setting explains the difference
   by itself therefore rests on the assumption that none changed within the boot. Any of them
   can still raise the per-launch failure probability.
2. **Large phase-3 survivor volume is not necessary for the observed phase-4 host hang** (H3 of
   `RIG_CRASH_FORENSIC_20260822.md` §7). Today phase 3 delivered **60 survivors**; 2026-08-22
   delivered **13,146,485**. The same two hosts failed at the same phase-4 seam both times. This
   does not show the two hangs share a cause, nor exclude volume as an August contributor.
3. **Recurrence despite the configured cap mitigation; the effective cap at the fault is
   UNOBSERVED** (narrowed per Beta R1 Q3).
   - `amdgpu-powercap-min` was installed 2026-08-22 19:51, after that day's freeze.
   - Its "applied" log line appears in every later boot, including today's 08:27:34
     (`logs/rig_gpu_fault_20261115/aug22_arming_and_powercap_*.txt`). The script swallows write
     errors and logs unconditionally, so that line is not a readback.
   - No pre-fault `power1_cap` readback exists.
   - At most this concerns the deployed cap, not every cap and not the sag hypothesis.
4. **CORRECTION to LEADS L-3 and the Aug-22 forensic addendum (two positions for Michael, see
   L-3 and T5).** Alpha adopted at gate 1: *netconsole was NOT armed during the freeze.* Beta R1
   Q2 holds that the evidence supports only *no pre-freeze arming proof is retained; event status
   UNOBSERVED*, because an ad hoc `modprobe` in the journal's lost tail (the incident boot's
   journal ends 18:19 / 18:27) would leave neither trace. The session agrees with Beta's
   narrower reading. Evidence as first written: `/etc/modprobe.d/netconsole.conf` and
   `/etc/modules-load.d/netconsole.conf` were **created 2026-08-22 19:25:07–08** on all three
   hosts (`stat`, birth time). The incident boot on `.121` and `.163` (boot −3, 10:40 → 18:27 /
   18:19) contains **zero** netconsole kernel lines (`journalctl -b -3 -k | grep -c netconsole`
   → 0). The first `netpoll: netconsole` lines are in boot −2 at 19:25. So the 19:26 `NCPROOF`
   packets prove the senders were armed **after** the reboot, not during the freeze. The record
   currently reads *"not active — CLOSED / no event STANDS"*. That is contradicted. **Ruled
   (Alpha, after R3): the record's wording is Beta's: *"no pre-freeze arming proof retained;
   incident kernel-event status UNOBSERVED."*** The Aug-22 freeze is therefore compatible with
   today's GCVM signature; it is not evidence against it. Appended to L-3; L-3's text is not
   altered.
5. **Correction to the brief's live read:** `power1_cap` is 94 W only on `.155` and `.163`
   (range 94–120 W, default 100 W). On `.121` the "min" cap is **126 W** (default 135 W) for the
   `R66XTG` cards and **136 W** (default 145 W) for the `123XT145W`/`R66XTGOP` cards. That is a
   6–7 % reduction from default, not a fleet-wide 94 W.
6. ~~`TODO_CONSOLIDATED.md:81` is stale~~ **WITHDRAWN (Beta R1 Q4, verified).** The persistence
   files exist, but **boot-time arming failed on every retained boot inspected** (boots −1 and 0
   on all three hosts):
   `netpoll: netconsole: vmbr0 doesn't exist, aborting` at module load. Netconsole was armed
   later by hand (08:56 all hosts; 13:55 `.121/.163`)
   (`logs/rig_gpu_fault_20261115/atomics_netconsole_*.txt`; bundle journals `.121:2104`,
   `.163:2102`). TODO:81 is right.
7. **The miner reuses April's prime memory suspect.** `_best_effort_gpu_cleanup()` calls
   `free_all_blocks()` on both CuPy pools after **every** sub-stripe
   (`miner/range_miner_worker.py:1051-1070`, `:1331`), copied from `sieve_gpu_worker.py:78-94`.
   Its removal from the PWC worker was approved (`PROPOSAL_FREE_ALL_BLOCKS_REPLACEMENT_v1_0.md:275`)
   and is recorded as still open (`TODO_CONSOLIDATED.md:317`). → arm g2.
8. **S162's recorded root cause (DKMS) is contradicted by recurrence.** amdgpu-dkms 6.12.12 was
   S162's "complete" fix (`SESSION_CHANGELOG_S162_FINAL.md:10-14`). It was present at S163KARG
   and is present today. The 2026-04-04 S162 Option B run also had **both** repeat cards first:
   `16:00.0` and `06:00.0`, SQC, 48 ms apart (`TB_UPDATE_S162_OPTION_B_RESULTS.md:30-37`).

---

## Prior record

### Searches run (worktree at `3579088`)

```
/bin/grep -lE 'GCVM|PROTECTION_FAULT|0x00801231|0x00841011|device lost from bus|rrig6600c crash|rig crash|host hang|voltage sag|sag|power cap|powercap|dpm-pin|DPM harness|dpm_peak|Kaspa|netconsole|L03|VBIOS|riser|pci=noaer|AER' \
   docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_* docs/TB_RULING_REQUEST_* docs/PROPOSAL_* docs/CHAPTER_* \
   docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md docs/exchange/ docs/exchange/*/*
   -> hits 1 / 6 / 3 / 35 / 27 / 11 / 1 / 1 / 1 / 1 / 13 files
/bin/grep -rniE 'powercap|power1_cap|power cap|dpm harness|dpm-pin|gfxoff|ppfeaturemask|lockup_timeout|HSA_ENABLE_DEBUG_TRAP|audio=0|amdgpu\.dc' docs/*.md docs/exchange
/bin/grep -rlnE 'powercap|rig-power-telemetry|6668|sag mitigation|power1_cap_min' . ~/distributed_prng_analysis/{docs,scripts}
   -> no record of the amdgpu-powercap-min or rig-power-telemetry installation (only numeric false positives)
ls docs | grep -iE 'S159G|S162|S163KARG|S164|S165|S170|FORENSIC'
```

The keyword hits in `TB_RULING_*`, `CHAPTER_*` and `*_AUDIT.md` were read at each hit. They
concern netconsole as a soak/launch prerequisite and Chapter 9's GFXOFF claims. None of them
decides the fault.

### Documents that record this subject

| Document | What it records |
|---|---|
| `docs/PROPOSAL_S162_RRIG6600C_CRASH_ROOT_CAUSE_v1_0.md` §1–2 (2026-04-04) | rrig6600c `16:00.0` `0x00801231` SQC, PERM 0x3, then `0xFFFFFFFF`, snd_hda D3hot→D0, rig offline. **Isolation: rrig6600c alone + Zeus clean; only all-3-rig load crashes.** *Not in Alpha's prior record.* |
| `docs/TB_Incident_Report_rrig6600c_S163KARG.md` §3.2, §4, §8, §11 | 2026-04-17 rrig6600c: 16:00.0 `0x00801231` SQC + 06:00.0 `0x80081071` TCP, cascade to 19:00.0, device lost from bus. Preceded by a 4 m 54 s worker-idle gap. amdgpu-dkms 6.12.12, kernel 6.8.0-107. `amdgpu.gfxoff=0` on the cmdline is ignored by the dkms module |
| `docs/SESSION_CHANGELOG_20260418_S163KARG_FORENSIC.md:17-41` | Same event, stage decode |
| `docs/SESSION_CHANGELOG_20260419_S163KARG_P2.md` "DPM Pin Service", "rrig6600 card0 PCIe Issue" | dpm-pin (manual, sclk state 1, mclk state 3). Rig links recorded at 5.0 GT/s; rrig6600 card0 `Unknown 63` |
| `docs/SESSION_CHANGELOG_20260419_S164.md` | rrig6600 19:00.0 (card7) KFD preemption failure / MODE1 reset (**not a GCVM fault**). Deployed audio=0, dc=0, lockup_timeout=30000, gpu_recovery=1, Kaspa 2250 MHz/−150 mV. All on the **Ubuntu** installs |
| `docs/SESSION_CHANGELOG_20260420_S165.md` §1–2, §4 | **The Kaspa profile was found to cause crashes and was reverted; dpm-pin DISABLED on all three rigs; stock `high`.** HSA_ENABLE_DEBUG_TRAP=0 added after a KFD debug-trap NULL dereference. A 900 mV / 2100–2200 MHz harness was proposed. *Not in Alpha's prior record.* |
| `docs/SESSION_CHANGELOG_20260425_S170.md:274-286` | `s170-dpm-peak-lock.service` (manual, peak sclk, COMPUTE profile). rrig6600c "card1" is the VBIOS **L03 outlier**; reflash deferred. *Not in Alpha's prior record.* |
| `docs/TB_FINAL_UPDATE_S159G_ROOT_CAUSE_CONFIRMED.md:11-40` | rrig6600 crashes traced to workers missing HSA_ENABLE_SDMA=0 / HSA_ENABLE_RUNTIME_POWER_MGMT=0 / AMDGPU_NO_POWER_PROFILE=1. *Not in Alpha's prior record.* |
| `docs/TFM_PROJECT_FACTS_SKILL.md` §2.50 | PWC's GCVM fault "tracked the launch storm … avoided, not eliminated". The miner's ~52 launches/trial is the design target |
| `docs/TODO_CONSOLIDATED.md:80-82, :281, :323` | Power-cap/telemetry check (open). Netconsole persistence (stale, see correction 6). Circuit check (OWNER, H2). DPM harness (open). rrig6600c GCVM root cause "avoided, not eliminated" |
| `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:21` | Owner ruling 3: power caps / circuit check / power telemetry **dropped** as Run-4 prerequisites; netconsole stays |
| `docs/RIG_CRASH_FORENSIC_20260822.md` §4–7 + addendum | Aug-22 phase-4 hang of rrig6600 + rrig6600c. H2 (sag) and H3 (volume) co-leading. Addendum's netconsole reading is corrected above |
| `docs/LEADS.md` L-3 | Same correction |
| `docs/RUNBOOK_NETCONSOLE_REARM.md` | Receiver 6667 on VM101; sender steps marked `[RECONSTRUCTED]` |
| `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:24`, `:69`, `:183`, `:203` | **Added at R2 (Beta R1 Finding 1).** Freeze reproduction is not delegated; delegated env/system changes are forbidden; a silent host or kernel event ends a session → T4 is owner-operated |
| `docs/TB_UPDATE_S162_OPTION_B_RESULTS.md`, `docs/SESSION_CHANGELOG_S162_FINAL.md` | **Added at R2.** Serialised execution did not prevent the fault; the list of S162 diagnostics; DKMS as root cause |
| `docs/PROPOSAL_FREE_ALL_BLOCKS_REPLACEMENT_v1_0.md`; `docs/TODO_CONSOLIDATED.md:317` | **Added at R2.** April's memory-release theory; removal approved for PWC; still open |
| `docs/ROCm_Saturation_Report_S172.md` | **Added at R2.** 6 workers per rig stable, 8 crash; KFD/MES saturation; untested module-parameter proposal |
| `docs/SESSION_CHANGELOG_20260422_S166.md`, `docs/SESSION_CHANGELOG_20260424_S167_S168_S169.md`, `docs/SESSION_CHANGELOG_20260509_S174_D1.md` | **Added at R2.** cwsr_enable history; startup-hammer jitter and pacing; 2026-05-09 rrig6600 + rrig6600c crash with rrig6600b surviving |
| Prior-work register `/home/michael/tfm_worktrees/gpu-fault-history/docs/REGISTER_GPU_FAULT_PRIOR_WORK.md` | **Did not exist** at R2 writing; to be screened against when it does |
| `RUNBOOK_Proxmox_amdgpu_Rig_Install_v1_0.md` step 21 | **Not in the repository; not read** (per Alpha: Proxmox hosts got only gfxoff=0 ppfeaturemask) |

### Labels for each item this report raises

| Item | Label |
|---|---|
| rrig6600c 16:00.0 `0x00801231` SQC first fault | **ALREADY RECORDED** (S162 §2.2, S163KARG §4.1). Today is the 3rd observed occurrence → L-6 |
| Fault only under multi-rig load | ALREADY RECORDED (S162 §1) |
| Phase-4 seam, rrig6600 + rrig6600c, rrig6600b survives | ALREADY RECORDED (`RIG_CRASH_FORENSIC_20260822.md` §5–6) |
| Netconsole arming on 2026-08-22 | **CHANGES A RECORDED FINDING** (L-3, forensic addendum). Two positions (Alpha: not armed; Beta: no proof retained, UNOBSERVED) → Michael |
| Boot-time netconsole arming failed on every retained boot inspected (`vmbr0`) | NEW (Beta R1 Q4; verified live) |
| Miner calls `free_all_blocks()` after every sub-stripe | ALREADY RECORDED for PWC (TODO:317); NEW for the miner |
| Arms dropped as already tried (T4.6) | ALREADY RECORDED (cited per row) |
| KFD/MES concurrency saturation, 6 vs 8 workers | ALREADY RECORDED (`ROCm_Saturation_Report_S172.md`) |
| Large phase-3 volume not necessary; recurrence despite configured cap; same recorded inputs pass/fail | NEW |
| `amdgpu-powercap-min` + `rig-power-telemetry` live on hosts since 2026-08-22, unrecorded in repo | NEW (state). Owner ruling 3 dropped caps as a *requirement*; it did not forbid them. Not a conflict, but undocumented |
| Telemetry UDP has no listener on VM101 (only 6667 bound) | NEW |
| L03 VBIOS card = rrig6600c 06:00.0 = gpu1, faulted Apr-17 and today | Card ALREADY RECORDED (S170). Link to the fault is NEW |
| audio=-1, no snd-power.conf, no DPM unit, level=auto on Proxmox | NEW as a recorded diff (S164/S165/S170 recorded the Ubuntu state only) |
| HSA_* not explicitly set by the miner launcher; simulated non-interactive shell UNSET; live inheritance UNOBSERVED | NEW. Relates to S159G and S165 (ALREADY RECORDED for PWC) |
| `pci=noaer` on all hosts | NEW |
| S164's "16.0 GT/s healthy" reseat check read the GPU endpoint, which always reports its internal switch link | NEW |
| `.121` SATA `ata5` BadCRC / I/O errors on `sda` at 09:06–09:07 | NEW, separate from the GPU fault |
| PCIe atomics absent on 7 of 8 slots per rig (PCH root ports), on all three rigs | NEW (measured 2026-11-15, T4.4). Owner addition |
| Operator reports (HiveOS, AMD Radeon mGPU doc, Ray forum) | NEW, external; verified per line in T4.3 |
| Cold vs warm kernel cache as a candidate (arm h) | NEW (proposed by this session) |

### Not searched

The Zeus host and the ser8 archives (`~/crash_dumps/…` holding the April captures). The private
repository. `RUNBOOK_Proxmox_amdgpu_Rig_Install_v1_0.md` and any project file outside the repo.
The bare-metal Ubuntu boot targets, which are not booted, so their live state is UNAVAILABLE.
Switch and PDU logs. `monitor_all.sh`, `gpu_health_check.sh`, `soak_monitor.sh` and
`gpu_monitor.py` exist at the repo root but were **not read yet**; they are T3 input.

---

## T1 — Evidence table

### T1.1 Events

| | S162 (2026-04-04) | S163KARG (2026-04-17) | S164 (2026-04-19) | 2026-08-22 run 3 | **2026-11-15 attempt 2** |
|---|---|---|---|---|---|
| OS / transport | Ubuntu bare metal / PWC-TCP | Ubuntu / PWC-TCP | Ubuntu / PWC | Proxmox CT / miner | Proxmox CT / miner |
| Rig(s) hit | rrig6600c | rrig6600c (+ rrig6600b GPU1 silent, attributed to Xorg) | rrig6600 | rrig6600 + rrig6600c | **rrig6600 + rrig6600c** |
| Survivor rig | n/a | rrig6600, rrig6600b | rrig6600b, c | rrig6600b | rrig6600b |
| Cards (first → cascade) | 16:00.0 only | 16:00.0 + 06:00.0 → 19:00.0 | 19:00.0 (card7) | UNOBSERVED (no pre-freeze arming proof retained) | `.121` 19:00.0; `.163` 16:00.0 → 06:00.0 |
| First status | `0x00801231` SQC(inst), PERM 0x3 | 16:00.0 `0x00801231` SQC PERM 0x3; 06:00.0 `0x80081071` TCP PERM 0x7 | not GCVM: KFD preemption fail, SMU `0xFFFFFFFF`, MODE1 reset −121 | UNOBSERVED | `.163` 16:00.0 `0x00801231` SQC PERM 0x3 RW 0; `.121` 19:00.0 `0x00841011` TCP PERM 0x1 RW 1 |
| Escalation | `0xFFFFFFFF` ~100 ms | `0xFFFFFFFF` +142 ms (19:00.0), +192 ms (16:00.0) | — | — | `.121` +97 ms; `.163` 06:00.0 first logged at +146 ms already `0xFFFFFFFF`; 16:00.0 +196 ms |
| Terminal | snd_hda D3hot→D0, rig offline | fence timeout +9 s, **device lost from bus** +10.3 s | recovered in 3.5 min | host silent, sockets held open | snd_hda 19:00.1 (+1.15 s) and 06:00.1 (+2.29 s) **inaccessible**; last packet 13:21:39.15; hosts hung, powered |
| VMID | — | 8 | — | — | 8 (all) |
| Timing | uptime 3036 s; launch time UNAVAILABLE | after a 4 m 54 s worker-idle gap; launch time UNAVAILABLE | mid Trial 1 | 1–6 s after phase-4 stripes created 18:32:01; 7 + 8 shards recorded first | **~73 ms from phase-4 stripe creation to the first received fault output** (launch time not measured); **zero recorded** phase-4 shards from either host (bundle main file; its 4 MB WAL not read) |
| Clock/DPM | perf `high` (udev) | perf `high`, 2590–2740 MHz | 2750 MHz stock V (pre-Kaspa) | stock (Proxmox), no cap | `auto`, stock OD, 0 mV, powercap-min |
| Sources | S162 §1–2 | Incident §3.2, §4 | S164 "Crash Analysis" | Forensic §5–6 + stat finding | netconsole lines 359–508; ledger |

### T1.2 Today, exact sequence (one clock: VM101 ledger and VM101 netconsole receipt)

```
13:13:37.140  trial context created                       (ledger trials.created_at)
13:16:50.7    phase 3 starts on all rigs; last rig shards 13:17:45–13:18:07
13:21:28.989  phase-4 stripes created                     (ledger stripes, phase=4)
13:21:29.024  coordinator logs phase-4 derived bound       (gate12 log :537)
13:21:29.0627 .121 0000:19:00.0 TCP 0x00841011 PERM 0x1 RW 1, pasid 32839, "3 callbacks suppressed"  (nc :359-369)
13:21:29.0633 .163 0000:16:00.0 SQC 0x00801231 PERM 0x3 RW 0, pasid 32830, "3 callbacks suppressed"  (nc :370-380)
13:21:29.137  zeus first phase-4 shard; 13:21:29.253 rrig6600b first phase-4 shard
13:21:29.159  .121 19:00.0 0xFFFFFFFF WALKER 0x7 PERM 0xf MAPPING 0x1        (nc :384)
13:21:29.209  .163 06:00.0 first line, already 0xFFFFFFFF, pasid 32785        (nc :391-400)
13:21:29.259  .163 16:00.0 0xFFFFFFFF                                          (nc :404)
13:21:30.212  .121 snd_hda_intel 0000:19:00.1 D3hot->D0 device inaccessible   (tail)
13:21:31.354  .163 snd_hda_intel 0000:06:00.1 D3hot->D0 device inaccessible
13:21:39.152  last netconsole packet (.163, "24 callbacks suppressed")
13:31:29.123  TRIAL TERMINAL compute_lease_expiry, rrig6600:gpu0 st3_s1      (gate12 log :702)
```

The two hosts' first fault lines were **received 0.63 ms apart** at VM101. That is not a
measurement of host-to-host fault simultaneity. Equal StripeAssign arrival on both hosts is
**not established** by the coordinator timestamp. Common dispatch would explain
near-simultaneity, and it would not separate an electrical cause from a software one. The "3 callbacks suppressed" on the first printed line
comes from a ratelimit state, possibly the global `printk_ratelimit`. Whether earlier faults went
unprinted is **[UNVERIFIED]**. The host journals end at 13:20:32 (`.163`) and 13:18:56 (`.121`),
before the event (journald lost about the last minute). The host-local telemetry TSV ends
13:21:15.6 / 13:21:10.9. The CT `/tmp/minerlogs` worker logs were lost on the CT reboot:
**UNAVAILABLE**.

### T1.3 Idle before phase 4 (last phase-3 shard → first phase-4 shard, per host)

| Run | rrig6600 | rrig6600c | Outcome |
|---|---|---|---|
| 57dc50fa (this morning) | 09:11:38 → 09:15:16 = **218 s** | 09:11:37 → 09:15:16 = **219 s** | clean, 544 shards each |
| ab6318d4 (afternoon) | 13:18:07 → 13:21:29 = **202 s** | 13:17:48 → 13:21:29 = **221 s** | both hung, 0 shards |
| 8cb116da (Aug-22) | 18:30:03 → 18:32:02 = 119 s | 18:27:51 → 18:32:05 = 254 s | both hung after 7 / 8 shards |

The idle→resume shape is common to pass and fail. It is not sufficient for a failure.

### T1.4 Power at 1 Hz (package power `power1_average`, which under-counts board power)

| | phase-3 max total | phase-4 first seconds |
|---|---|---|
| rrig6600 `.121` | 558 W (21:17:20Z) | idle 48 W at 21:21:10.9Z; then file lost |
| rrig6600c `.163` | 619 W (21:16:52Z) | idle 47 W at 21:21:15.6Z; then file lost |
| rrig6600b `.155` (survived) | 546 W | 45 W at 21:21:28.7Z → **339 W at 21:21:29.7Z** → 156–313 W |

On the **surviving** rig, phase 3 drew more than phase 4. The failed rigs' phase-4 power is
missing, and phase 3 ran clean on every rig. **A 1 Hz sampler cannot see the
sub-second inrush**, so a transient sag is neither supported nor refuted here (H2 stays open).

### T1.5 Bus address → GPU index → card identity

GPU index is the miner's `ROCR_VISIBLE_DEVICES=N` (`scripts/launch_fleet_manual.sh:316`). This
maps to the Nth GPU node of the CT's KFD topology, which is in bus order on all three CTs
(`/sys/class/kfd/kfd/topology/nodes/*/properties`, `location_id`). That ROCr enumerates agents in
KFD node order is taken from ROCm behaviour and was not executed: **[UNVERIFIED by live HIP
init]**. The faulting PIDs are in launch order, which is consistent with the mapping (`.163`
891759 < 892773). `unique_id` and `serial_number` are empty on every card, so physical identity
rests on bus + VBIOS only.

| Host | gpuN | BDF | host DRM card | VBIOS | OD_SCLK max | cap/default W | Root port | Riser link |
|---|---|---|---|---|---|---|---|---|
| rrig6600 `.121` | 0 | 03:00.0 | card0 | 113-123XT145W201222 | 2669 | 136/145 | 00:01.0 (CPU) | 5 GT/s x1 |
| | 1 | 06:00.0 | card1 | 113-D53201-R66XTG | 2689 | 126/135 | 00:1b.0 | 5 GT/s x1 |
| | 2 | 09:00.0 | card2 | R66XTG | 2649 | 126/135 | 00:1b.5 | 5 GT/s x1 |
| | 3 | 0d:00.0 | card3 | 123XT145W201222 | 2599 | 136/145 | 00:1c.5 | 5 GT/s x1 |
| | 4 | 10:00.0 | card4 | R66XTG | 2604 | 126/135 | 00:1c.6 | 5 GT/s x1 |
| | 5 | 13:00.0 | card5 | R66XTGOP | 2634 | 136/145 | 00:1c.7 | 5 GT/s x1 |
| | 6 | 16:00.0 | card6 | R66XTG | 2709 | 126/135 | 00:1d.0 | 5 GT/s x1 |
| | **7** | **19:00.0** | card7 | **R66XTG** | 2709 | 126/135 | **00:1d.1** | 5 GT/s x1 |
| rrig6600b `.155` | 0–7 | 03,06,09,0d,10,13,16,19 | card0–7 in order | all 113-EXT800296-L04 | 2549–2704 | 94/100 | 01.0,1b.0,1b.5,1c.5,1c.6,1c.7,1d.0,1d.1 | 8 GT/s x1 |
| rrig6600c `.163` | 0 | 03:00.0 | card1 | L04 | 2664 | 94/100 | 00:01.0 | 8 GT/s x1 |
| | **1** | **06:00.0** | **card0** | **113-EXT800291-L03** (only L03 in fleet) | **2750** | 94/100 | 00:1b.0 | 8 GT/s x1 |
| | 2 | 0a:00.0 | card2 | L04 | 2639 | 94/100 | 00:1c.5 | 8 GT/s x1 |
| | 3 | 0d:00.0 | card3 | L04 | 2564 | 94/100 | 00:1c.6 | 8 GT/s x1 |
| | 4 | 10:00.0 | card4 | L04 | 2564 | 94/100 | 00:1c.7 | 8 GT/s x1 |
| | 5 | 13:00.0 | card5 | L04 | 2679 | 94/100 | 00:1d.0 | 8 GT/s x1 |
| | **6** | **16:00.0** | card6 | L04 | 2574 | 94/100 | **00:1d.1** | 8 GT/s x1 |
| | 7 | 19:00.0 | card7 | L04 | 2729 | 94/100 | 00:1d.3 | 8 GT/s x1 |

Observations:
- On both failing hosts one faulting card sits on root port **00:1d.1**. The rigs' bus
  numbering differs: rrig6600c has `0a:` and `1d.3`. So **the same BDF on two rigs is not the
  same slot.**
- The April addresses (Ubuntu) and today's (Proxmox, `pci=assign-busses`) may number buses
  differently. "Same card as April" is **[UNVERIFIED]** for 16:00.0. For 06:00.0 it holds by
  VBIOS: the only L03 card is today's 06:00.0.
- rrig6600 is a different card population from b/c: higher power limits, MCLK 1000 MHz against
  875 MHz, and risers at 5 GT/s against 8 GT/s.

---

## T2 — Configuration: Proxmox hosts today against the S164 record

Today's column is from live reads at 14:17 PST (`logs/rig_gpu_fault_20261115/host_*.txt`,
`host2_*.txt`). It is identical on all three hosts unless noted. The S164 column is the Ubuntu
bare-metal state recorded on 2026-04-19, as amended by S165 and S170 where they changed it. The
live Ubuntu targets are **UNAVAILABLE** (not booted).

| Setting | S164 record (Ubuntu) | Later record | Proxmox today | Source (today) |
|---|---|---|---|---|
| Kernel | 6.8.0-107 (Incident §8) | — | **6.8.12-9-pve** | `uname -r` |
| amdgpu driver | amdgpu-dkms 6.12.12 (Incident §8) | — | amdgpu-dkms `1:6.12.12.60403-2194681.22.04`, module `/lib/modules/6.8.12-9-pve/updates/dkms/amdgpu.ko`, srcversion `AC5C22E22EEDC97831DD74B`; dpkg `hold` on `.163` only, `install` on `.121/.155` | `modinfo`, `dkms status`, `dpkg-query` |
| Firmware | UNAVAILABLE | — | pve-firmware 3.15-3; `dimgrey_cavefish_*.bin` dated 2025-04-02; loaded ME 0x40, PFP 0x66, RLC 0x56, MEC 0x7a (first card per host; others not compared) | `ls -l /lib/firmware/amdgpu`, debugfs `amdgpu_firmware_info` |
| ROCm userspace (CT) | UNAVAILABLE | — | 6.4.3-128; cupy 13.5.1 (all CTs) | `/opt/rocm/.info/version`, `importlib.metadata` |
| Kernel cmdline | + `amdgpu.dc=0 lockup_timeout=30000 gpu_recovery=1`, − `sched_jobs` (S164 GRUB) | — | `pcie_aspm=off pci=noaer amdgpu.ppfeaturemask=0xffff7fff amdgpu.gfxoff=0 amdgpu.runpm=0 amdgpu.aspm=0 pci=assign-busses,hpbussize=0x33 iommu=pt amdgpu.dc=0 amdgpu.lockup_timeout=30000 amdgpu.gpu_recovery=1` | `/proc/cmdline`; `/etc/default/grub:9` |
| dc / lockup_timeout / gpu_recovery | 0 / 30000 / 1 | — | 0 / 30000 / 1 (same) | `/sys/module/amdgpu/parameters` |
| GFXOFF | `gfxoff=0` via modprobe.d (cmdline form ignored) | — | module has **no `gfxoff` parameter** (absent from sysfs listing). GFXOFF is off via `ppfeaturemask=0xffff7fff` (live) | sysfs params |
| **HDMI audio** | **`audio=0`** (S164: stops the PSP-suspend reset cascade) | — | **`audio=-1`**; snd_hda_intel binds every GPU `.1` function; both terminal lines today are snd_hda on 19:00.1 / 06:00.1 | sysfs `audio`; nc tail |
| `amdgpu.conf` modprobe | `options amdgpu gfxoff=0 audio=0` | — | `.163`: `options amdgpu gfxoff=0 ppfeaturemask=0xffff7fff` (`/etc/modprobe.d/amdgpu.conf:1`, 2026-07-08); `.121/.155`: **none** | `/etc/modprobe.d/*` |
| snd-power.conf | present (Incident §11.4) | — | **absent on all hosts** | `/etc/modprobe.d/*` |
| **DPM level** | manual, 2250 MHz, −150 mV (Kaspa) | **S165: reverted, dpm-pin DISABLED, stock `high`**; S170: `s170-dpm-peak-lock` (manual, peak, COMPUTE) | **`auto`**, no dpm-pin, no s170 unit, no udev perf rule | sysfs; `systemctl list-unit-files`; `/etc/udev/rules.d` |
| OD_SCLK / VDDGFX offset | 2250 / −150 mV | S165 proposed 2100–2200 / 900 mV absolute (never built, TODO:281) | stock: 2549–2750 MHz max per card, **0 mV** | `pp_od_clk_voltage` |
| Power profile | — | S170: COMPUTE | `BOOTUP_DEFAULT` | `pp_power_profile_mode` |
| MCLK | 1000 MHz | S165: 1000 (a), 875 (b/c) | 1000 (`.121`), 875 (`.155/.163`) | `pp_dpm_mclk` |
| Power cap | stock (none recorded) | owner ruling 3: caps dropped as a requirement | **`amdgpu-powercap-min.service`** active since 2026-08-22 19:51: 94 W (b/c); 126/136 W (a) | `systemctl cat`; script below |
| Telemetry | — | — | `rig-power-telemetry.service` 1 Hz UDP → VM101:6668 + local TSV. **No 6668 listener on VM101** (`ss -lunp`) | `systemctl cat` |
| netconsole | ad hoc | — | persistence files present (`/etc/modules-load.d/netconsole.conf` + `modprobe.d`, created 2026-08-22 19:25) → VM101:6667. **Boot arming failed on every retained boot inspected** (boots −1 and 0, all hosts; `vmbr0 doesn't exist, aborting`); armed later by hand (08:56; 13:55) | `stat`; `journalctl -b 0/-1 -k` (R2) |
| Not established either way | — | — | Older display-stack state (GDM/Xorg, S163KARG §11) against today's other GPU clients at event time; firmware equivalence across all cards (only the first card per host was read) | UNAVAILABLE |
| PCIe ASPM / runpm | — | — | `pcie_aspm=off`, `amdgpu.aspm=0`, `runpm=0` | cmdline |
| **AER** | — | — | **`pci=noaer`**: link errors are not reported. Latched CE bits (Timeout, BadTLP, BadDLLP, RxErr, Rollover) are present on all three hosts, **including the stable rrig6600b** (`.155` up since 08:26). `.121/.163` latches were reset by the 13:52 reboot | `lspci -vv` DevSta/CESta |
| PCIe AtomicOps | — | — | Completer support only on CPU root port `00:01.0`; all PCH root ports `AtomicOpsCap 32bit- 64bit- 128bitCAS-`. Only GPU `03:00.0` has `ReqEn+`; amdgpu logs `PCIE atomic ops is not supported` for the other 7 per rig. Same on all three hosts | `lspci -vv`; `journalctl -b 0 -k` |
| **Riser link (root port)** | S163KARG_P2: 5.0 GT/s; card0 `Unknown 63`. S164 "16.0 GT/s Gen4 healthy" after reseat | — | `.121` **5 GT/s x1** all 8; `.155/.163` **8 GT/s x1** all 8. USP "downgraded" from cap 16 GT/s x8. The GPU endpoint always reports 16 GT/s x16: that is the Navi-internal switch link, which is what S164's reseat check read | `lspci -vv` LnkSta on root port + USP |
| Worker HSA env | PWC: `HSA_OVERRIDE_GFX_VERSION=10.3.0`, `HSA_ENABLE_SDMA=0`, `HSA_ENABLE_DEBUG_TRAP=0` (`persistent_worker_coordinator.py:140-141`, `:145`; S165, S159G) | — | miner launch sets only `ROCR_VISIBLE_DEVICES`, `CUPY_CACHE_DIR`, `PYTHONPATH` (`scripts/launch_fleet_manual.sh:314-323`). `.bashrc:122-123` exports are unreachable non-interactively (`.bashrc:5-8` returns). Simulated non-interactive source → `UNSET`. The same on all three rigs. **Not explicitly set by the miner launcher; simulated shell UNSET; live inheritance (PAM/SSH) UNOBSERVED** | read + `env -i bash -c` |
| journald | — | — | persistent (`/var/log/journal`), default sync. Lost the last ~1 min before both hangs (today and Aug-22) | `journalctl --list-boots` |

**`/usr/local/sbin/amdgpu-powercap-min.sh`** (sha256 `f506b1fafc2a93e7…`, identical purpose and
text on all hosts, installed 2026-08-22 19:51):

```
 1 #!/bin/bash
 2 # S172 sag-mitigation: pin every amdgpu GPU to its power1_cap_min.
 3 # Purpose is blunting the phase-4 INRUSH TRANSIENT, not average heat.
 4 # Reversal: write power1_cap_max (or the prior value) to each power1_cap.
 5 for h in /sys/class/hwmon/hwmon*/; do
 6   [ -f "$h/power1_cap" ] || continue
 7   [ "$(cat "$h/name" 2>/dev/null)" = "amdgpu" ] || continue
 8   cat "$h/power1_cap_min" > "$h/power1_cap" 2>/dev/null || true
 9 done
10 logger -t amdgpu-powercap "applied power1_cap_min to all amdgpu GPUs"
```

`power1_cap` is the PPT limit that the SMU enforces over its averaging window. Whether it limits
a sub-millisecond inrush on these cards is **[UNVERIFIED]**. Today's recurrence is consistent with
it not doing so.

**Incidental (separate from the GPU fault):** `.121` logged `ata5` interface-fatal errors
(`UnrecovData 10B8B BadCRC`), link downshift to 3.0 Gbps and READ I/O errors on `sda` at 09:06–09:07
(netconsole :233–358). A SATA cable or port fault is a candidate explanation, not a diagnosis.

---

## What the T1/T2 facts predict, and what would settle each

Settling any of these takes GPU load or a host write, so none can be done in this brief.

| Concern | Prediction | Measurement that settles it |
|---|---|---|
| H2 transient sag at the synchronised launch (OPEN) | Fault probability rises with the number of cards hitting the first heavy launch together, across rigs on a shared circuit. Single-rig runs pass (S162 isolation table) | Sub-second supply/rail logging across a phase-4 launch; a staggered launch or launch-rate A/B at fleet scale |
| Card or slot susceptibility (rrig6600c 16:00.0 three times, the L03 card twice, root port 1d.1 on both failing hosts) | Faults keep landing on the same cards/slots | Swap cards between slots on rrig6600c; record whether the fault follows the card or the slot |
| Clock/voltage headroom (S165 lesson, auto DPM at stock 2.55–2.75 GHz, 0 mV) | A pinned lower clock lowers the per-launch failure rate | **EXCLUDED from T4** (T4.6: Kaspa profile recorded as crash-causing; Alpha withdrew the clock lead). Owner may reopen |
| Lost protections (audio=0, snd-power, HSA env) | If they matter, they change the *terminal outcome* (hang vs recover) more than the first fault **[UNVERIFIED]** | Repeat a fault with audio=0 / lockup settings and compare terminal outcome |

---

## T3 — `scripts/rig_gpu_recorder.py` (built, tested on idle rigs)

sha256 (R3) `a232a36d61e9c09a0bf5770341fe4466f405e091e009a3a25b9ee554d4d9593e` (R2 `8a4e7042…`, R1 `47a9a36c…`). Stdlib only; runs with
any `python3` on VM101.

**How it works.**
- One persistent `ssh -T root@host 'python3 -B -I -u - 1.0'` per host (`:206-209`). A read-only
  sampler is streamed on stdin (`REMOTE_SAMPLER`, `:77`). It opens sysfs for reading only and
  writes to stdout only. **`-B` disables bytecode writes**; R1's claim that `-I` did so was wrong
  (Beta R1 Q5). `-I` ignores cwd, user site and `PYTHON*`. It never opens `/dev/dri` or
  `/dev/kfd`.
- Per card, about once a second: `gpu_index` (bus order, which matched CT KFD order on
  2026-11-15), BDF, DRM card, `gpu_busy_percent`, `mem_busy_percent`, perf level, current SCLK and
  MCLK level, and from hwmon: `power1_average`, `power1_cap`, `in0_input` (VDDGFX mV),
  `freq1/2_input`, edge/junction/mem temperatures, fan rpm, pwm. A field that cannot be read is
  `null` and named in `missing`, never 0 (`:118-154`).
- VM101 stamps `recv_ts` and `recv_lag_s`. Lag above ~1 s marks a late, backed-up line. **OBSERVED
  means a sample was received, possibly late; it is not a statement that the host is alive now**
  (Beta R1 Q5). `host_ts` is the sample time.
- **R2 additions (Beta R1 Q5):**
  - single-instance guard: the PID file is claimed atomically (`O_CREAT|O_EXCL`; R3, Beta R2 Q5),
    so simultaneous starts cannot both run; a stale file is replaced. Tested in R3 (test5): two
    simultaneous starts with a stale PID file present → one rc 0 with `unresolved=[]`, one
    **REFUSED rc 2**;
  - `unresolved()` includes the stderr-drain thread (R3). It certifies the **local** side only.
    The remote sampler exits at its next write after sshd closes the session: a check about
    0.5 s after stop still saw it on `.155`, and checks a few seconds later saw none. So the
    remote check is made **≥ 2 s after stop**;
  - verified stop: the final `stopped` event lists any thread or ssh process still alive as
    `unresolved`, and the exit code is 3 (`:452`);
  - bounded reads: sampler lines capped at 1 MiB (`MAX_LINE`, `:67`); stderr drained in a thread
    keeping the last 4 KiB;
  - every `udp6668` record carries `producer_zero_fill_possible: true`, because
    `rig-power-telemetry.sh:20-21` writes 0 on read errors.
- **UNOBSERVED:** a watchdog writes one `status:"UNOBSERVED"` record per host per second once no
  sample has arrived for `--grace` (2.5 s), with `since_last_sample_s`, the ssh state and
  `cards:null` (`:315-329`). SSH sessions reconnect with backoff (`:193-313`).
- **UDP 6668:** binds the port the hosts' `rig-power-telemetry.service` already sends to. Each
  datagram is stored verbatim plus parsed fields (`:331-364`).
- Output: `logs/rig_gpu_recorder/rig_gpu_<ts>.jsonl` (ignored), written and flushed line by line,
  so memory is bounded. PID file `logs/rig_gpu_recorder/recorder.pid`. Stop with
  `kill $(cat …/recorder.pid)`, never pkill. Also `--duration`.
- Hosts are derived from `rig_profiles_config.json`: the CT100 `proxmox` endpoint minus 1
  (CLAUDE.md §3) (`:382-390`). Each host's own hostname is in every sample. `--host` overrides.

**Tests run (idle rigs; all three hosts and all 24 cards; raw output in `logs/rig_gpu_recorder/test{1,2,3,4}/`):**

| Test | Result |
|---|---|
| 1. 15 s, three hosts + unroutable `192.0.2.1` | 45 OBSERVED (15 per host), host-side period 1.000/1.000 s min/max, sampling ≤16 ms, 24/24 cards, `missing` empty, `busy_pct` 0.0 throughout. The silent host gave 13 UNOBSERVED with `cards:null` and no zero values. 44 UDP-6668 datagrams captured from all three hosts |
| 2. Default host derivation, ssh to `.163` frozen (SIGSTOP) for 6 s, stop via PID file | Derived hosts `.121/.155/.163` → `rrig6600/b/c`. UNOBSERVED every second at 2.71 → 6.71 s since last sample; then the six buffered samples arrived with `seq` 5–10 and no gap. SIGTERM via PID file: rc 0, PID file removed |
| 3. After edits | `recv_lag_s` 0.008 s, UNOBSERVED reason `no_sample_within_2.5s` |
| 4. R2 build: three hosts + `192.0.2.1`, 8 s, second instance started at +1.5 s | Second instance **REFUSED, rc 2**. First: 24 OBSERVED, 6 UNOBSERVED (silent host), 25 UDP records all flagged; `stopped` event `unresolved=[]`, rc 0 |
| After all tests | No recorder process on VM101; no sampler process on any host (`pgrep -af "[p]ython3 -B -I"` → none); no `.pyc` found on the hosts in R1 |

**Limits.**
- Each SSH login is itself logged by the host's sshd (journal/wtmp). This is the same side effect
  as any read-only SSH the brief allows.
- `power1_average` is package power, not board power, and is averaged by the SMU. **1 Hz cannot
  see a sub-second inrush.** Neither this tool nor the existing telemetry measures H2 directly.
- A sysfs read on a card that has dropped off the bus may block in the kernel. The host is
  already hung in that case; whether the read can make anything worse is **[UNVERIFIED]**.
- **Do not run the recorder while `beta_relay.py` runs.** The relay snapshots ignored files
  under `logs/`, and a growing JSONL would read as a BOUNDARY VIOLATION.

---

## T4 — Reproduction and discrimination plan, R2 (DESIGN ONLY; nothing here has been run)

**Governing record (Beta R1 Finding 1).** `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md`
excludes freeze-reproduction classes from delegated execution (`:24`, `:203`). It forbids
delegated environment or system changes (`:69`) and ends a session on a silent host or a kernel
event (`:183`). **This plan is therefore owner-operated.** Michael starts every series, performs
every recovery and every host or hardware change. No review of the probe and no existing Run-4
authorisation extends delegated authority to it.

### T4.0 Constraints and how results will be read (revised per Beta R1 Q6 and R2 Q6)

1. **Single-rig false-pass risk.** S162 §1 recorded rrig6600c *alone* + Zeus clean for 3 trials,
   while all three rigs together crashed it every time. The S172 saturation report records 6
   workers per rig stable and 8 crashing (`docs/ROCm_Saturation_Report_S172.md:29-41`). A
   single-card or single-rig series with no fault is a **non-reproduction**, pre-declared as such.
   It is never a pass.
2. **Base rate, descriptive only.** Six miner phase-4 starts are on record (three ledger main
   files, `immutable=1`; WAL contents not read): four passed and two failed (`8cb116da`,
   `ab6318d4`). 2/6 is a historical proportion. It is not used to size a stopping rule.
3. **Statistics (R3, per Beta R2 Q6).**
   - **Unit of observation:** one *attempt* = one fresh probe process taken through the declared
     history (T4.1) to the end of its hand-over window. Each arm gets a **fixed attempted budget
     B, declared before it starts** (8 unless stated). There is no stopping on a clean streak,
     no discarding, and no retry that replaces an attempt.
   - **Every attempt is reported in one of three classes**, B = F + C + I:
     - **F (fault):** a demonstrated E1. It counts as F even if another observation failed.
     - **C (clean):** all three observation channels were proven live across the whole window
       (netconsole proof before and a packet after, recorder OBSERVED throughout, probe trace
       complete), and there was no E1.
     - **I (invalid):** anything else, including an unknown E1 outcome.
   - **Per-endpoint observation coverage** (E1, E2, E3 separately) is reported for every series.
   - **E1 rate is reported as the descriptive range F/B … (F+I)/B.** An exact one-sided 95 %
     binomial bound is quoted **only over the C + F fully observed attempts**, and only with the
     stated assumption of independent, comparable attempts. Examples: 0 F in 8 C → 31.2 %; 0 F in
     4 C → **52.7 %**. Excluding I attempts is a stated assumption, not a repair: the bound is
     reported beside the range, never instead of it.
   - **Comparisons:** arm against **interleaved A0 attempts** (A0, arm, A0, arm …) inside the
     same host boot where possible. Fisher's exact p is computed over fully observed attempts,
     with the boot and recovery history of every attempt listed. Attempts across a recovery
     reboot are flagged as a different stratum, not pooled silently.
   - No arm is said to "prevent" anything. The licensed words are "supports a rate reduction
     under this configuration". Many arms make a screen, and any positive is followed by a
     replication series before it is reported as a finding.
4. **Endpoints, recorded separately (Q9):**
   - **E1:** first GCVM fault line on netconsole;
   - **E2:** card loss: `0xFFFFFFFF` status, function inaccessible, or `device lost from bus`;
   - **E3:** host silence: netconsole silent, recorder UNOBSERVED, sockets silent.

   An arm that changes only E2/E3 (recovery) is not reported as changing E1.
5. **Every E3 costs a hung host** and the owner's recovery (T4.8). The plan's cost is counted in
   hangs, not in minutes.

### T4.1 Probe: a REDUCED probe (to be built and reviewed as worker/kernel class before use; not built)

`scripts/rig_phase4_seam_probe.py`, revised per Beta R1 Q6/Q1 and R2 Q6/Q10.

**It is a reduced probe.** It omits production phases 1 and 2 and the coordinator, network and
staging. Its non-reproduction means non-reproduction only.

- **Declared history, one fresh process per attempt (R3).** Production launches fresh workers
  for every run (`scripts/launch_fleet_manual.sh:314-323`), and keeps kernel objects per family
  in-process (`sieve_gpu_worker.py:124`, `:127`, `:131`). So in a run, the reverse-hybrid module
  is first loaded at the phase-4 seam, after the phase-3 module is resident. Each attempt
  reproduces exactly that **first seam**: a new process, ~60 s of phase-3 sub-stripes, 200 s
  idle, then the phase-4 first launch and ~25 s of phase-4. Disk cache state is part of the
  declared history (warm, as in the failing afternoon run, unless the arm is h). A **later-seam**
  variant (the same process going round again with reverse-hybrid already loaded) is a separate,
  labelled arm, never pooled with first-seam attempts.
- **Real assignment, not a `trial_context` row.**
  - Build each `StripeAssign` with the production payload builder (`miner/range_miner_coordinator.py:9331-9358`)
    from the failed run's preflight, residues and digests, thresholds and strategies.
  - Execute via `SieveExecutor.execute` (`miner/range_miner_worker.py:1125`).
  - The payload carries no `skip_range`; the executor default is 0–16 (`:1164`). The hybrid
    path uses strategy arrays (`:1221`, `:276`). That is already recorded
    (X-VOLUME-LIMIT R1 `:79`) and is reproduced as production does it.
  - Production cleanup after every sub-stripe, `_best_effort_gpu_cleanup` incl. `free_all_blocks()`
    (`:1051-1070`, called at `:1331`), is kept unchanged in A0.
- **Seed ranges follow worker identity.** Each GPU gets the sub-stripe ranges its namesake
  worker had in `ab6318d4`, also after a hold-out or swap; ranges are never redistributed.
- **Device identity check before any load.** The probe reads its visible device's PCI BDF (HSA
  agent / sysfs) and refuses to run unless it equals the planned BDF. The KFD-order mapping
  (`logs/rig_gpu_fault_20261115/ct_kfd_topology_*.txt`) stays provisional until this check.
- **Operation trace (R3, per Beta R2 Q10).** Every GPU-touching operation emits an **entry** and
  an **exit** record: allocation, H2D copy, module load, launch call, sync, D2H `.get()`, torch
  `is_available`/`synchronize`/`empty_cache`, each pool free. Each record carries a per-process
  **monotonic sequence ID**, wall µs, pid, BDF, operation, and return or error status. Records
  go to VM101 by UDP as they happen, with a local copy in the CT's `/dev/shm` (a probe-run
  write, under the same authorisation as the GPU load). The controller detects gaps by
  sequence ID.
  - **Attribution rule:** at E1 the report states the **last confirmed completion** (last exit)
    and the **last entered operation** (last entry without exit), with any sequence gaps. An
    operation is attributed only if the trace is complete up to the E1 receipt. Otherwise the
    operation-at-fault is **UNOBSERVED**.
  - Receipt order between UDP trace records and netconsole lines is not a guarantee of event
    order. Both carry clocks; only the CT wall clock (shared with the host kernel) and the
    kernel stamp converted to wall time are compared.
- **Runtime-identity capture (R3, two-runtime lead).** After each process's first cleanup, the
  probe records the libraries mapped in `/proc/self/maps` and its open `/dev/kfd`/`/dev/dri`
  file descriptors. This reads only its own process and adds no GPU work.
- **Fidelity controls, three separate ones:**
  1. kernel argument tuple identical to production (dtype/order/values);
  2. runtime history matching the declared reduced-probe history (fresh process; phase-3
     family loaded, then reverse-hybrid at the seam; idle; cache state). This is not identical
     to a full production history (phases 1/2 omitted);
  3. survivor counts equal to the ledger's for the same ranges.

  Counts alone do not establish fidelity.

### T4.2 Stop controller (new, per Beta R1 Q9; design only)

`scripts/rig_seam_controller.py` on VM101 owns every cycle.

- **Launch slots.** Probes block at a barrier before every first-launch slot and launch only on
  the controller's per-slot "go". Staggers are therefore issued by the controller, not by
  timers in the probes.
- **Triggers.** The controller tails `logs/netconsole_all_rigs.log`, the recorder JSONL and the
  probe event stream. On the first E1/E2/E3 line, a probe error, a probe silence over 5 s, or
  the cycle's **wall deadline** (warm + idle + phase-4 + 60 s), it does the following:
  1. **withholds every not-yet-issued go, fleet-wide**;
  2. sends stop to all probes;
  3. after 10 s, SIGTERM and then SIGKILL via SSH to the recorded CT PIDs;
  4. reports any process still unresolved.

  It does not assume a signal can recover a wedged driver.
- **No automatic retry, GPU reset, reboot or next cycle.** The next cycle starts only on the
  owner's command, after recovery, re-reading the BDF→index→VBIOS map, and NCPROOF from every
  host.

### T4.3 Stages

| Stage | Scope | Outcome rule |
|---|---|---|
| S0 | **One card**, rrig6600c gpu6 `16:00.0`; other seven idle (arm g2) | An E1 **inside the hand-over window** (from the phase-4 module-load entry to 30 s after the first phase-4 launch exit) → one-card seam reproduction. An E1 during phase-3 warm-up or idle → one-card **warm-up reproduction**, recorded separately. 0 F over the budget → single-card non-reproduction; go to S1 |
| S1 | rrig6600c, 8 cards (A0) | Any E1 → arms run at S1. 0/B → single-rig non-reproduction; go to S2 |
| S2 | All three rigs, 24 cards (A0) | Any E1 → arms run at S2. 0/B → fleet-probe non-reproduction. That motivates S3; it does **not** show what the probe lacks |
| S3 | Production Run-4 shape, owner-launched, recorder + netconsole + controller-equivalent stop armed | Arms needing coordinator changes go through BEFORE-COMMIT review |

### T4.4 What operators report (owner addition 14:43 PST)

| Claim (relayed by Alpha) | What the source says | Status |
|---|---|---|
| HiveOS "GPU detected dead (06:00.0)", "driver error, no temps" on 6600-class rigs | Thread 68383 (Jan–Feb 2022): reports of a dead-GPU line at `06:00.0` and of "No Temps" needing a power cycle. **Advice** to swap the riser and check connections; default-first then per-card OC/UV staging. A riser-power question is unanswered ([68383](https://hiveon.com/forum/t/6600-6600xt-gpu-driver-error-no-temps/68383/34)). Beta could not retrieve the exact #34/#46/#48 posts; I read them through a fetch summary | Reports and advice. **No verified successful riser repair** |
| Riser or riser USB cable swaps resolve it | Thread 52441: one user replaced all risers and it was "not a solution" ([52441](https://hiveon.com/forum/t/6600xt-detected-dead/52441)). No USB-cable fix found | **Not supported** as a resolution; counter-report exists |
| Powering the riser properly | Not found in posts read directly | **UNVERIFIED** |
| One weaker card gets its own settings | Per-card OC/UV staging reported (68383) | Reports. Our cards run stock (0 mV, stock OD) |
| Dropping the BIOS PCIe link speed | One post (Batfink, 2021-11-02): crashes stopped after setting slots to Gen1. The same page has unsuccessful cable/PSU/riser changes ([54378 p2](https://hiveon.com/forum/t/gpu-driver-errors-and-6600xt/54378?page=2)) | One uncontrolled report |
| AMD: CPU slots, identical lane width, PCIe atomics; Radeon limits on simultaneous workloads | ROCm Radeon mGPU page (docs 6.4.2), quoted: identical lane width and PCIe 3.0 atomics; CPU-connected slots, avoid chipset; "Some consumer grade motherboards may only support the first PCIe slot"; "not recommended to exceed 2 simultaneous compute workloads, with the assumption that workloads are running alongside a graphics environment" ([mgpu 6.4.2](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-6.4.2/docs/install/native_linux/mgpu.html)) | Verified quotation. Applicability to a headless one-process-per-card RX 6600 XT fleet **UNVERIFIED** |
| Ray: delay between starting each GPU because start transients exceed steady state | 2025-04-05 request about NVIDIA GPUs; no replies; workaround was a frequency limit ([ray 22227](https://discuss.ray.io/t/mitigating-simultaneous-gpu-voltage-spikes-during-initialization/22227)) | A candidate explanation, not a measurement |
| What ROCm does without atomics | AMD's page: atomics required "for optimum compatibility"; unrouted atomics "will fail" (CA/UR) ([pcie-atomics](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/conceptual/pcie-atomics.html)). **Beta's lead:** upstream KFD v6.8 admits gfx10.3 parts without PCIe atomics when MEC firmware ≥ 92 ([kfd_device.c](https://github.com/torvalds/linux/blob/v6.8/drivers/gpu/drm/amd/amdkfd/kfd_device.c#L185)). Installed MEC `0x7a` = 122. Inference only; the installed module is 6.12.12 DKMS, not that source | Fallback mechanism **UNVERIFIED** |

### T4.5 Measured here (atomics, riser path)

Raw captures: `logs/rig_gpu_fault_20261115/atomics_netconsole_192.168.3.{121,155,163}.txt`
(`lspci -vv` DevCap2/DevCtl2 for every bridge and GPU, plus the boot-0 kernel journal's atomic
and netconsole lines).

- **PCIe AtomicOp requests from the GPUs to the root are not enabled on 7 of 8 cards per rig, on
  all three rigs.**
  - The CPU root port `00:01.0` advertises AtomicOps completion (`32bit+ 64bit+ 128bitCAS+`).
  - Every PCH root port advertises none (`32bit- 64bit- 128bitCAS-`).
  - Only GPU `03:00.0` has `AtomicOpsCtl: ReqEn+`.
  - amdgpu logs `PCIE atomic ops is not supported` for 7 cards per host.

  This says nothing about on-device atomics. It is identical on the stable rrig6600b, so it
  cannot by itself separate faulting from clean cards. **R3 correction:** the card position with
  root atomics *has* been a first-fault card. rrig6600 `03:00.0` SQC, 2026-04-02, Ubuntu era
  (`REG:249`); the bus-to-slot correspondence across OS is [UNVERIFIED].
- Riser path of the three faulting cards:
  - **rrig6600 `19:00.0`:** root port `00:1d.1`, link 5 GT/s x1.
  - **rrig6600c `16:00.0`:** root port `00:1d.1`, link 8 GT/s x1.
  - **rrig6600c `06:00.0` (L03):** root port `00:1b.0`, link 8 GT/s x1.

  All are PCH ports without atomics. The physical riser, USB cable and riser power are
  **UNAVAILABLE** to software (owner).

### T4.6 Prior-work screen (R3: against the register)

**Register:** `/home/michael/tfm_worktrees/gpu-fault-history/docs/REGISTER_GPU_FAULT_PRIOR_WORK.md`
(2,654 lines). I read its front section: Null results, Context, the four consolidated lists
(`:130-284`) and Cross-era contradictions (`:288-308`). Cited below as `REG:<line>`. Parts A–E
were not read in full. The R2 screen is confirmed, and the register adds the rows marked
**(REG)**. Conditions often differed (Ubuntu, PWC, stock driver). A historical fault rules out
absolute prevention in that treatment, not a present-day rate benefit (Beta R2 Q11). The owner
may reopen any row.

| Dropped arm | Recorded result |
|---|---|
| a1 perf `high` | S163KARG faulted with every card at `high` (Incident §5.1). **(REG)** "stock `high` DPM unstable in longer runs" (`REG:195`); udev perf=high in S80 (`REG:145`) preceded every later GCVM era |
| a2 S170 peak-lock | Deployed from boot in S170 (`SESSION_CHANGELOG_20260425_S170.md:24`, `:274-278`); pool 8 still crashed (`:14`, `:123`) |
| **Kaspa profile / any OD or voltage pin** | 2250 MHz / −150 mV "Was Causing All Crashes", ASIC reset −22 (`REG:194`). Excluded |
| c1 HSA env (incl. **`HSA_ENABLE_SDMA=0`**, `HSA_ENABLE_RUNTIME_POWER_MGMT=0`) | **(REG)** S159G: each "Still crashed" (`REG:181`, `TB_SUBMISSION_S159G_RIG6600_CRASHES.md:105-112`); `/etc/environment` vars "insufficient" (`REG:182`) |
| c2 `HSA_OVERRIDE_GFX_VERSION` | Always in the PWC env (`persistent_worker_coordinator.py:140`, `:385`), which faulted throughout S162–S170. **(REG)** Jan-23 "HSA_OVERRIDE fix … didn't help" (`REG:176`) |
| Serialisation **as a fix** (`AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3`) | **(REG)** "crashed ~1–2 min in; RULED no" (`REG:184`; `TB_UPDATE_S162_OPTION_B_RESULTS.md:21-25`, `:66-71`). Kept only as an **instrument** (g2s) |
| `cwsr_enable=0` | Mixed/adverse (`REG:190`, `:196`, contradiction 4 `REG:293`). Today all hosts run `cwsr_enable=1` and the same pair failed. That addresses necessity of cwsr=0, not efficacy; the adverse history is the reason it stays dropped |
| Install amdgpu-dkms | "ROOT CAUSE", contradicted: 2M crashed at ~1400 s (`REG:155`); present today |
| First-assignment jitter / pacing / per-chunk cooldown | S168 jitter alone: trial-2 fault (`REG:197`); jitter + pacing contested (`REG:161`); 50 ms cooldown crashed (`REG:198`) |
| **Spawn stagger (S158C 10→30 s inter-node)** | **(REG)** "did NOT fix"; "caused spawn crashes", reverted (`REG:180`). Spawn timing, not first-launch-of-family timing |
| **Warm-up** (GPU warm-up barrier; kernel pre-compile at start; reverse-sieve warm-up) | **(REG)** Jan-23 warm-up barrier "didn't help" (`REG:176`); Apr-05 pre-compile `22bb145` and reverse-sieve warm-up `8d91311` "made crashes worse" (`REG:188`). **This bears on arm h**, which is therefore narrowed (below) |
| Scratch-reclaim vars; CuPy pool off / raw hipMalloc; seed-cap cuts | `REG:185-187`, `:189` |
| Explicit `del` + gc only | "Did not resolve the crash" (`REG:179`) |

**Register items that stay in the plan and were proposed, never tried:**
- **STAGGERED-PHASE4-DISPATCH-1**, a 1–2 s per-host phase-4 dispatch offset at the stage-
  assignment seam, "NOT IMPLEMENTED", parked (`REG:230`). That is the production form of **b2**.
- **S157 riser/slot permutation and GPU/riser swap**, never done (`REG:215`). That is **e1/e2**.
- **S162 Option A gc-only cleanup** (`apply_s162_safe_cleanup.py` exists, never run) and free_all_blocks Option A
  (synchronize before free) and Option C (`MemoryAsyncPool`) (`REG:218-219`). These relate to **g2b/g2c**.
- **amdgpu scheduler parameters** (`REG:225`). That is **k1**.
- **"netconsole needs the host console level raised with `dmesg -n 7`"** (`REG:114`). This is now
  in the owner's step 1.

**Register correction to this report.** R1/R2 said the one card with root atomics (`03:00.0`)
"has never been a recorded first-fault card". **That is wrong:** on 2026-04-02 rrig6600 crashed on
`03:00.0` with an SQC first fault (`REG:249`; `TB_SUBMISSION_S159G_RIG6600_CRASHES.md:17-43`).
That was in the Ubuntu era, so the bus-to-slot correspondence is **[UNVERIFIED]**. Corrected in
T4.5.

### T4.7 Candidates that remain: software only first, least invasive first

Wording per Beta R1/R2 Q7: "supports" means a rate reduction under that configuration. A fault
refutes only absolute prevention, never a contributing role. Every series is read under T4.0
item 3. Every g2 attempt is a first seam (T4.1).

| # | Arm | Write? | What a result supports | Limits | Prior record |
|---|---|---|---|---|---|
| **g2a** | **Single-card hand-over, production cleanup.** rrig6600c gpu6 `16:00.0` alone, fresh process per attempt, history per T4.1 (production cleanup incl. torch sync and both pool frees), B = 8, other seven idle | No | An E1 **inside the hand-over window** shows that one card can fail in this seam workload with eight active cards unnecessary. An E1 in warm-up/idle is a one-card warm-up reproduction, recorded separately. 0 F is a non-reproduction | Card, slot, power, runtime and kernel stay open. Without g2n it shows "one card faulted in this seam workload", not a hand-over mechanism | **(g) April's memory-release theory** (S162 §5; `PROPOSAL_FREE_ALL_BLOCKS_REPLACEMENT_v1_0.md`). `free_all_blocks()` was removed, then restored (`REG:291`); it is present at `miner/range_miner_worker.py:1069-1070` |
| **g2n** | **No-hand-over control (R3; wording per Beta R3 Q13, ruled by Alpha).** As g2a, but after the 200 s idle the process resumes a full **phase-3** stripe (module already loaded) instead of the phase-4 stripe. Its window opens at the resume `phase_begin` and closes 30 s after the first resumed launch exit, so its exposure matches g2a's | No | An excess of in-window E1 in g2a over g2n supports **the phase-4 transition/workload bundle relative to a phase-3 resume**. It does **not** distinguish module loading or first launch from a fault specific to reverse-hybrid computation. **E1 in both arms: both resumed workloads can fault, and their shared cause is unresolved.** A finite "none in g2n" is not a zero rate; compare under T4.0. Run interleaved with g2a, with the same F/C/I reporting | Equal budgets; same card, boot and cache state. A no-idle control or a preloaded-phase-4 contrast could test narrower hypotheses later | — |
| **g2t** | **Superseded by the P/P1/M/N arm set (T6).** Removing the whole torch cleanup block tests removal of **that whole block**: the import, the availability query, the synchronise and the allocator cache release. It is not a test of co-resident runtimes alone (Beta R3 Q13, ruled by Alpha) | No | Fewer E1 supports that combined intervention; it cannot establish that runtime coexistence is causal | The trace records the operations, but does not by itself separate these causal effects. A missing torch library in the maps is a manipulation check, not proof of mechanism | T4.11; T6 |
| g2s | g2a re-run **only if g2a faulted**, with `AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3 HIP_LAUNCH_BLOCKING=1` | No (process env) | **An instrument, not a fix.** With serialised launches and copies, the trace **may bracket** the operation if it is complete: the last confirmed completion and the last entered operation (T4.1). If the trace is incomplete, operation-at-fault is UNOBSERVED. A fault after a completed free does not alone show free/kernel overlap. Serialisation cannot exclude damage left by an earlier operation | Changes timing; a non-fault under g2s is uninterpretable. Runtime honouring of the flags is verified from the trace | Option B non-prevention (`REG:184`); [AMD HIP debugging](https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.2/how_to_guides/debugging.html#general-debugging-tips) |
| g2b | g2a with **both** CuPy pool frees removed (`get_default_memory_pool().free_all_blocks()` and `get_default_pinned_memory_pool().free_all_blocks()`); the rest of cleanup unchanged | No | Fewer E1 than interleaved g2a supports involvement of **that combined release policy** | Retained allocation and timing changes are part of the intervention; not one pool uniquely | S162 Option A / free_all_blocks Option B (`REG:218-219`) |
| g2c | **Corrected premise (Beta R2 Q10).** Production already calls `torch.cuda.synchronize()` before the frees (`miner/range_miner_worker.py:1061-1069`), and torch **is** installed on every rig venv. The result `.get()` before cleanup is host-blocking (`:1270`). g2c adds an explicit **CuPy-runtime** barrier, `cp.cuda.Device(0).synchronize()`, immediately before the frees, and records whether the baseline torch sync ran and returned | No | Fewer E1 supports this cleanup difference. It is **not** a repair of an established unsynchronised free | Under the two-runtime finding (T4.11), torch's synchronise acts on torch's HIP runtime, so whether it orders CuPy's work at all is **[UNVERIFIED]**. g2c tests the explicit CuPy barrier, nothing more | free_all_blocks Option A (`REG:219`) |
| **A0** | rrig6600c, all 8 cards, production seam (S1) | No | Reproduction rate at rig scale | S162: may not reproduce on one rig | L-6 |
| **b1** | Phase-4 first-launch stagger within the rig, controller-issued 250 ms slots | No | Fewer E1 than interleaved A0 supports first-launch coincidence | No neighbouring launch within ±100 ms refutes only a trigger needing that exact coincidence | Spawn stagger and jitter priors (T4.6) are different seams |
| **h** | **Narrowed in R3.** Disk-cache state only, process history fixed: fresh process, reverse-hybrid never loaded in-process, empty vs populated `CUPY_CACHE_DIR`. **No warm-up or pre-compile** | No (run's own cache) | Fewer E1 with one state supports cache involvement | Must not bundle a process restart or a warm-up | The register records warm-up and pre-compile as worse (`REG:176`, `:188`). 2/2 first-in-boot passes vs 2/2 later failures (n = 4; cold state [UNVERIFIED]) |
| **d1 / d2** | Hold out gpu1 + gpu6 vs gpu3 + gpu4 (6 cards each) | No | d2 faulting while d1 does not supports card/slot specificity. Both clean is **compatible with** a count/load effect, not evidence establishing it | S172's boundary depends on chunk size as well as count (`REG:264`). Compare both against contemporary A0. **At S2 the d-arm endpoint is E1 on `.163`**; other hosts are recorded separately | `ROCm_Saturation_Report_S172.md:29-35` |
| d3 | Only gpu1 + gpu6 | No | E1 with two cards shows eight-card load unnecessary | Not a card/slot proof; 0 is ambiguous | — |
| **S2 A0, b2** | Fleet scale; b2 = rigs 1–2 s apart at phase-4 dispatch | No (probe); production form is a coordinator change (BEFORE-COMMIT) | b2 fewer E1 supports cross-rig start coincidence | Workloads still overlap; a b2 fault does not exclude a shared resource | **STAGGERED-PHASE4-DISPATCH-1**, parked, never built (`REG:230`) |
| **e1** | Owner: swap the `16:00.0` card with the `0d:00.0` card (labelled), paths unchanged; then **swap back** | Hardware | Fault follows the card, then returns with it → card | Record aux-power leads and connectors | **S157 card/riser swap plan, never done** (`REG:215`) |
| **e2** | Owner, only after e1 is restored: **crossed** riser + USB cable swap between `1d.1` and `1c.6` with a non-suspect card in `1d.1` | Hardware | Fault follows the riser bundle → riser/cable | Riser + cable stay a bundle | `REG:215`; HiveOS advice; counter-report |
| e3 | Owner: own PSU lead for the `1d.1` riser (record connector type) | Hardware | Lower rate supports that riser-power intervention | Unchanged finite data do not exclude power | — |
| f1 | Owner: move the `16:00.0` card into the CPU slot `00:01.0` | Hardware | Tests the **CPU-slot path class** (atomics + routing + path) | Cannot isolate atomics. Note the 04-02 `03:00.0` SQC fault on rrig6600 (`REG:249`) | AMD mGPU doc |
| k1 | `hws_max_conc_proc=4`; then separately `sched_policy=1`; then `queue_preemption_timeout_ms=30000`, one per series, with baseline restored between | **Yes** (modprobe.d + reboot) | Fewer E1 supports **the tested scheduling configuration**, not a named driver or firmware mechanism | Verify each setting is effective (`/sys/module/amdgpu/parameters`) | `REG:225`; no recorded result |
| e4 | BIOS: Gen1 on PCH slots | **Yes** (BIOS) | Fewer E1 supports the Gen1 configuration | Changes timing/throughput/power as well as margin | One HiveOS report; Feb-01 GPU[4] reseat + Gen1 "sensors healthy" (`REG:144`) |

**Order:** g2a ⇄ g2n (interleaved) → g2t → (g2s, g2b, g2c only if g2a faulted) → A0 → b1 → h →
d1/d2 (→ d3) → S2 A0 → b2 → e1 → e2 → e3 → f1 → k1 → e4.
- **Software before hardware:** g2 through b2 need no host write; e1–f1 need the owner's
  hands; k1 and e4 write host configuration.
- **Least invasive first:** one card before eight, one rig before three.
- **Hardware arms stay available** after a software non-reproduction too.

### T4.8 The owner's manual steps

1. **Arm netconsole on each host before every session**, raise the console level
   (`dmesg -n 7`, per `REG:114`) and confirm NCPROOF. Boot-time arming failed on every retained
   boot inspected (`vmbr0 doesn't exist, aborting`), so this is a host write and it is his.
2. **Start every series.** Every GPU load is owner-launched.
3. **If a host hangs:** tap its outlet in the plug app to power-cycle it, select Proxmox at boot
   (the rigs default-boot Ubuntu), re-arm netconsole, then tell the session. The session re-reads
   the card map read-only before anything else runs.
4. **Hand swaps (e1, e2, e3, f1):** clean shutdown, outlet off, label the cards physically before
   moving them, one swap per series, outlet on, boot Proxmox, re-arm netconsole.
5. **If reached:** module parameters for k1 (modprobe.d + reboot) and BIOS Gen1 for e4 at the
   console.
6. **Optional:** the rrig6600 host disk's SATA symptoms (cable or port is a candidate, not a
   diagnosis).

### T4.9 What could hang a host

1. Any cycle that reproduces: E3 is the failure itself. S2 puts rrig6600b at the same risk.
2. With `audio=-1`, snd_hda_intel binds each GPU's `.1` function. Both faulting cards ended in
   `D3hot→D0 … inaccessible`. S164 removed that component (`audio=0`) from the reset path.
   Today's inaccessible function does not show audio caused the hang, but it is a plausible
   terminal-path factor.
3. Host writes in k1/e4 (the only ones proposed). OD clock/voltage writes are excluded.
4. A worker re-initialising during a GPU reset (S165 KFD debug-trap NULL dereference): the
   controller never retries.
5. `pci=noaer`: no AER reporting or recovery. Latched AER bits are cleared by the power cycle.
6. Blocking kernel launch, sync or cleanup calls (`miner/range_miner_worker.py:1270`, `:1331`,
   `:1063`) cannot be bounded by the probe. That is why the external controller deadline and
   fleet-wide withholding of pending launches exist (T4.2).
7. The recorder's sysfs reads on a card that has left the bus may block (T3 limits).
8. Hand swaps can return a rig with a card missing or re-enumerated (`pci=assign-busses`). The
   map is re-read after every swap.
9. A wrong BIOS link setting (e4) can cost cards at boot.

### T4.10 Proposed first test

Once the reduced probe and the stop controller (T4.1–T4.2) are built and reviewed, the first test is g2a interleaved with its no-hand-over control g2n, owner-launched, on rrig6600c gpu6 (`16:00.0`) alone: one fresh process per attempt doing production phase-3 sub-stripes, 200 s idle, then either the phase-4 first launch with production cleanup (g2a) or a phase-3 resume (g2n), eight attempts each, reported as fault/clean/invalid, with the other seven cards idle, recorder, netconsole and controller armed, each process's mapped HIP runtimes captured, and no host write. An E1 inside g2a's hand-over window with none in g2n shows one card can fail at the family change without rig-wide concurrency, and a serialised re-run may bracket the operation if its trace is complete; no fault across the fully observed attempts is a recorded single-card non-reproduction that moves the plan to g2t and then A0 on all eight cards.

### T4.11 Measured in R3 (read-only, no GPU opened)

**(1) Two HIP runtimes load into one worker-environment process: CONFIRMED at the library level;
initialisation on the GPU is NOT shown.**

On the rrig6600c CT (`.164`), as michael, in the worker's own venv (`~/rocm_env/bin/python -B -I`).
The environment had `ROCR_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES=` (all
empty). The imports followed the worker's order: CuPy first, as in the executor and startup
probe (`miner/range_miner_worker.py:1149`, `:1575`); then torch, as in cleanup (`:1061`).
Nothing was initialised. The `/proc/self/maps` readout:

| After | Mapped HIP/HSA libraries | `/dev/kfd`, `/dev/dri` fds |
|---|---|---|
| start | none | none |
| `import cupy` (+ `cupy.cuda.runtime`) | `/opt/rocm-6.4.3/lib/libamdhip64.so.6.4.60403`, `/opt/rocm-6.4.3/lib/libhsa-runtime64.so.1.15.60403` (system 6.4.3) | none |
| `import torch` | **both of the above plus** `…/site-packages/torch/lib/libamdhip64.so` and `…/torch/lib/libhsa-runtime64.so` (torch wheel, HIP `6.3.42131`) | none |

- No `libhsakmt` file is mapped in either case; it is presumably built into the HSA runtime
  [UNVERIFIED].
- Versions: torch `2.7.0+rocm6.3`, cupy `13.5.1`.
- A before/after listing of `~/.cache`, `~/.config`, `~/.cupy`, `/tmp` and `/dev/shm` was
  unchanged.
- **What this shows:** a worker process holds two HIP and two HSA runtimes once cleanup has
  imported torch.
- **What it does not show:** whether both initialise on the GPU. Production's
  `torch.cuda.is_available()` and `synchronize()` would initialise torch's runtime
  [inference]. **Left for the first test:** the probe's own `/proc/self/maps` and device-fd
  capture (T4.1).
- **Disclosure:** the first attempt of this check also created and immediately deleted an empty
  file `/dev/shm/.none` on the `.164` CT. It was a leftover in my command, a **CT write
  outside the brief**. The second, clean run is the one reported.

**(2) rrig6600b SMU `0xFFFFFFFF` at 13:47:28: what it was.**
- `logs/netconsole_listener.log:10098-10101` and the host journal: `.155` card `19:00.0` logged
  `SMU: response:0xFFFFFFFF for index:18 param:0x00000005 message:TransferTableSmu2Dram?` and
  `Failed to export SMU metrics table!`, twice, 1 ms apart, at 13:47:28.123–.124. Those are its
  only SMU error lines this boot; there was no page fault, reset or other system activity in
  13:46:30–13:48:30.
- The host's own 1 Hz telemetry at that second (`/var/log/rig-power-telemetry.tsv`,
  `21:47:28.081Z`) read **0 W and 0 °C for that card only**; the samples before and after
  read 6 W and 43 °C.
- **So:** two hwmon reads by the host's telemetry script got no SMU answer once, and the script
  zero-filled them. The card was idle (the trial ended 13:31:29), and it answers normally now
  (vendor `0x1002`, 6 W, 42 °C, busy 0).
- A transient SMU non-response on the *stable* rig. The register records that SMU polling cuts
  once helped (`REG:140`). **Design consequence:** the T3 recorder would add a second 1 Hz SMU
  poller beside `rig-power-telemetry`. For g2 I propose the recorder run at `--period 2`, or that
  its SMU-backed fields be reviewed. That is an owner/Beta choice; not changed in code.

---

## T5 — Beta

Request: `docs/exchange/to_beta/X-RIG-GPU-FAULT_R1_ALPHA_REVIEW_REQUEST_fault_evidence_and_plan.md`.
Relay: **REVIEW RECEIVED (exit 0)**, 668.7 s. Log: `logs/beta_relay/X-RIG-GPU-FAULT_R1.log`.
Review: **`docs/exchange/from_beta/X-RIG-GPU-FAULT_R1_BETA_REVIEW.md`**; read that file, not
this line, for Beta's reasoning.

| Q | Beta's R1 position | What R2 did |
|---|---|---|
| Q1 | NEEDS EVIDENCE | Agree with the narrowing: "stripe creation → first received fault", not "dispatch/launch"; "zero recorded shards" (bundle WAL not read); phase-3 > phase-4 power only on the surviving rig; SATA symptoms, not a diagnosis |
| Q2 | NEEDS EVIDENCE | Agree to soften L-3 to "no pre-freeze arming proof; event status UNOBSERVED". Alpha's adopted stronger wording and Beta's go to Michael side by side |
| Q3 | NEEDS EVIDENCE | Agree: "same recorded trial inputs and boot"; "large phase-3 volume"; "effective cap at fault UNOBSERVED" |
| Q4 | **OBJECTION · LIVE** | **Agree, verified.** `vmbr0 doesn't exist, aborting` at module load on **every current boot of all three hosts**: `.155` 08:27:28, `.121` 08:27:40 and 13:53:14, `.163` 08:27:25 and 13:53:02 (`journalctl -b 0/-1 -k`). Armed later by hand (08:56; 13:55). **Correction 6 above is withdrawn**: `TODO_CONSOLIDATED.md:81` is right that boot persistence does not work. HSA wording to be narrowed; PWC anchor is `:140-141, :145` |
| Q5 | SUGGESTION | Agree: use `-B` (the `-I` no-`.pyc` claim was wrong; no `.pyc` was found on the hosts); single-instance PID guard; verify threads/ssh stopped |
| Q6 | **OBJECTION · LIVE** | Agree: drop "prevents at ~5 %". Use a fixed cycle budget, report every failure with a 95 % upper bound (0/8 → 31 %), and interleave A0 controls. S2 0/8 is "non-reproduction" only. The probe must keep persistent-process and module history, and build the real phase-4 assignment, not a `trial_context` row |
| Q7 | **OBJECTION · LIVE** | Agree: rewrite confirm/refute as "supports / rate reduction" with the per-arm limits Beta lists; mark c1/a2 as bundles; redesign e2 as a crossed swap after e1 |
| Q8 | NEEDS EVIDENCE | Supply the raw `lspci -vv` AtomicOps capture (it was printed in-session, not saved to file). Word it as "AtomicOp requests to the root not enabled". Beta's KFD v6.8 MEC ≥ 92 lead is noted (installed MEC 0x7a = 122) |
| Q9 | **OBJECTION · LIVE** | Agree: specify a VM101 stop controller (wall deadline, cancel pending launches on first fault or silence, no retry/reset/next cycle until Michael recovers), and record first-GCVM / bus-loss / host-silence as separate endpoints |
| F1 | Finding | Agree: add `PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md` (`:24`, `:69`, `:183`, `:203`) to the prior record. This plan is owner-operated, not delegated |

R2 reply: `docs/exchange/to_beta/X-RIG-GPU-FAULT_R2_ALPHA_REPLY.md`. Relay: **REVIEW RECEIVED
(exit 0)**, 346.5 s, log `logs/beta_relay/X-RIG-GPU-FAULT_R2.log`. Review:
**`docs/exchange/from_beta/X-RIG-GPU-FAULT_R2_BETA_REVIEW.md`**; read that file for the
reasoning.

| Q | Beta R2 | Open for R3 (not done; stop gate) |
|---|---|---|
| Q1 | SUGGESTION | Carry the "received earlier" and provisional-gpuN wording into LEADS L-6 |
| Q2 | NEEDS EVIDENCE | Beta keeps "no pre-freeze proof; UNOBSERVED". Attribute report item 4's "correct reading" and the T1.1 Aug cell to Alpha. Two positions stay for Michael |
| Q3 | SUGGESTION | Carry the narrowed null-result wording into LEADS L-6 |
| Q4 | NO OBJECTION | Scope "every boot" to the retained boots inspected |
| Q5 | SUGGESTION | Atomic lock for single-instance; include the stderr thread in `unresolved()` |
| Q6 | **OBJECTION · LIVE** | Report attempted = fault + clean + invalid. Exact bounds only over fully observed cycles (0/4 → 52.7 %); give the range F/B to (F+I)/B for unknown E1. Define g2 history: the first seam vs later seams with reverse-hybrid already loaded. Label the probe a reduced probe (phases 1/2 omitted) |
| Q7 | SUGGESTION | d1 + d2 both clean is "compatible with" a count effect; define the S2 d-endpoint host; k1 supports the scheduling configuration, not a named mechanism |
| Q8 | NO OBJECTION | — |
| Q9 | NO OBJECTION | Controller implementation to be exercised with fake probes at its review |
| Q10 | **OBJECTION · LIVE** | Define the g2 hand-over window (a phase-3 E1 is not a seam reproduction) and add a no-hand-over control. g2s reports last confirmed completion **and** last entered operation, with sequence IDs and gaps; otherwise operation-at-fault is UNOBSERVED. **g2c premise: production already runs `torch.cuda.synchronize()` before the frees when torch is available** (`miner/range_miner_worker.py:1061-1069`); specify the exact CuPy barrier and record whether the baseline sync ran |
| Q11 | NO OBJECTION | Screen accepted as historical prioritisation |
| Q12 | SUGGESTION | "may bracket the operation if the trace is complete" |
| — | Finding | `LEADS.md:432` still lists a clock/voltage follow-up; align it with T4's exclusion |

**Found after Beta R2, not yet reviewed (LEADS L-6):** the rig venvs carry torch
`2.7.0+rocm6.3` with its own bundled HIP/HSA, while CuPy links the system ROCm 6.4.3. The miner's
cleanup imports torch and synchronises through it after every sub-stripe. By loader rules this
probably puts **two HIP runtimes in one worker process** **[UNVERIFIED; needs a live worker's
`/proc/PID/maps`]**. It bears directly on g2 and on Beta's R2 Q10 point about the existing
`torch.cuda.synchronize()`.

R3 reply: `docs/exchange/to_beta/X-RIG-GPU-FAULT_R3_ALPHA_REPLY.md`. Relay: **REVIEW RECEIVED
(exit 0)**, 345.4 s, log `logs/beta_relay/X-RIG-GPU-FAULT_R3.log`. Review:
**`docs/exchange/from_beta/X-RIG-GPU-FAULT_R3_BETA_REVIEW.md`**. **Round 3 of 3: the thread is
closed for rounds; what remains goes to Michael.**

| Q | Beta R3 (final) | Status |
|---|---|---|
| Q1, Q3, Q6–Q12 | NO OBJECTION | Resolved (Q6 and Q10 LIVE objections from R2 are resolved) |
| Q2 | NEEDS EVIDENCE | **Unresolved by evidence.** Two positions for Michael: Alpha "not armed"; Beta "no pre-freeze arming proof retained; UNOBSERVED" |
| Q4 | SUGGESTION | T2 netconsole cell scoped to retained boots (applied after R3; unreviewed) |
| Q5 | SUGGESTION | **The atomic PID claim still races**: Beta's local AST counterexample had two starters both claim between `os.open` and the PID write, or via stale takeover. Suggested fix: hold an `flock` for the recorder's lifetime. **Not applied**; until then the recorder doc's guarantee is too strong, and startup must be serialised by the operator |
| **Q13** | **OBJECTION · LIVE** | **Unresolved, holds the plan's reading of g2n/g2t.** g2a > g2n supports only "the phase-4 transition/workload bundle relative to phase-3 resume", not module-load/first-launch specifically; "E1 in both" means both resumed workloads can fault, with the shared cause unresolved; g2n needs an explicit resume-entry window and matched exposure. g2t tests removal of the **whole torch cleanup block** (import, availability query, sync, `empty_cache`), not co-resident runtimes alone. **The session agrees.** It is recorded here for Michael, and the T4.7 wording is not yet revised |
| Q14 | SUGGESTION | Claim strength confirmed. Transcript preserved at `logs/rig_gpu_fault_20261115/two_runtime_import_check_192.168.3.164.txt`. Maps/fds in the first test will **not** settle which runtime initialised the GPU; that stays UNOBSERVED unless attributed by the bound implementation |
| Q15 | SUGGESTION | **Drop** the recorder's extra SMU-backed reads (incl. busy % and DPM levels) for g2; keep UDP-6668 + netconsole + probe trace + a no-sensor host heartbeat. That needs a new recorder mode (not built) |
| Findings | — | Report summaries now disclose the CT write; prediction-table clock row marked EXCLUDED (applied after R3; unreviewed). The `.155` 13:47:28 explanation stays labelled inference |

## T6 — Next thread: X-RIG-GPU-PROBE (after Alpha's post-R3 rulings)

Rulings applied here: Q13 wording (T4.7 g2n/g2t), Q2 wording (item 4; L-3), Q5 flock and Q15 passive
recorder mode. The torch arm set (P/P1/M/N), the escalation ladder, the fleet P-vs-N design with its
budget, the torch inventory, and the probe + stop controller (built, off-GPU tested) are in
**`docs/CLAUDE_CODE_PLAN_RIG_GPU_PROBE.md`**. T4.7's g2t is superseded by that arm set.

## RESULT 2026-11-15 18:28 — two-runtime hypothesis refuted as the sole cause

With `RIG_CLEANUP_TORCH=off` deployed (`048fd47`, parity 30/30), all 24 rig workers logged
`torch_imported=False` and exactly one HIP/HSA runtime. Run 4 attempt 3 credited clean at 17:59.
The 18:18 confirmation repeat faulted the same three cards at phase-4 dispatch + 46 ms:
- `.121 19:00.0`: `0x00841051`, TCP;
- `.163 16:00.0`: `0x00801231`, SQC (inst);
- `.163 06:00.0`: `0x80001281`, SQC (inst).

Both hosts hung. **The fault occurs without a second runtime in the process.** The rate effect
is unmeasured. Decode: `docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT_DECODE_20261115.md`; `docs/LEADS.md`
L-6, L-7.

## STOP GATE 5 (after Beta R3)

Nothing has run on a GPU. Nothing has been written to a host. **One CT write occurred** (empty
`/dev/shm/.none` created and deleted on `.164`, disclosed). Nothing is committed.
