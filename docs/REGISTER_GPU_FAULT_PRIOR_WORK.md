# REGISTER — GPU fault prior work (rig GPU faults, page faults, resets, device lost, hard resets, hangs)

**Written:** 2026-11-15, Claude Code (Team Alpha implementer), worktree `tfm_worktrees/gpu-fault-history` at HEAD `3579088`.
**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_GPU_FAULT_PRIOR_WORK_20261115.md` (owner direction 2026-11-15 15:01 PST).
**Nature:** a register of what the record says. **It does not conclude a cause** and it closes no lead.
Every statement in the era parts carries `file:line`, a commit sha, or `git show <sha>:<path>`.
"MEM" means a Claude auto-memory note, which is not repo record. `[pre-scope]` marks context an era extractor included from the previous era.
Today's Run 4 retry (`distributed_config_t1_ab6318d4`) appears in this tree only in the brief. Its evidence comparison belongs to the separate `rig-gpu-fault` worktree and is not duplicated here.

---

## Prior record

### 1. Searches run (all read-only, from this worktree unless noted)

| # | Command / scope | Purpose |
|---|---|---|
| S1 | `/bin/grep -rliE "GCVM\|PROTECTION_FAULT\|page fault\|device lost\|ring timeout\|gpu reset\|riser\|noaer\|ppfeaturemask\|gfxoff\|lockup_timeout\|gpu_recovery\|free_all_blocks\|AMD_SERIALIZE_KERNEL\|HIP_LAUNCH_BLOCKING\|seed_cap\|launch storm\|undervolt\|powercap\|sclk\|amdgpu" docs` | 124 files; dated by `git log --diff-filter=A` |
| S2 | `/bin/grep -rliE "\bAER\b\|iommu\|SDMA\|HSA_\|\bDPM\b\|Gen ?1\b\|Gen ?2\b\|PCIe\|launch storm\|semaphore\|cooldown\|jitter\|warm-?up\|\bhang\|freez\|hard reset\|rig (crash\|reboot\|went down\|died)" docs` | 182 more files; ranked by fault and rig term density, and read where relevant |
| S3 | Per-era greps over every `docs/` file named for the period (Jan–Mar: 168 files; S159–S163; S163KARG–S174; Jul–Nov: ~350 files + `docs/exchange/**`), for all brief terms plus SQC, D3hot, ASPM, runpm, scratch, cwsr, SMU, MaxSessions, BIOS, M.2, stagger, gpu_count, netconsole, dmesg, debug trap, karg, FAULT_KEYWORDS, amdgpu-dkms, pve, boot-proxmox | era extraction |
| S4 | `/bin/grep -nE -i "<S1 terms>\|register\|crash\|fault" docs/PROJECT_FILE_CATALOG.md` | catalog §3.12 and rows 128–132, 482 |
| S5 | `ls docs/TB_RULING_* docs/TB_RULING_REQUEST_*` (41 files) + `/bin/grep -liE "<S1 terms>"` over them | only `TB_RULING_REQUEST_ROCR_ISOLATION_S149.md` hit; Run-4 rulings were read by the Jul–Nov pass |
| S6 | `/bin/grep -liE "<S1 terms>" docs/PROPOSAL_* docs/CHAPTER_* docs/*_AUDIT.md` | 8 proposals, 6 chapters/audits |
| S7 | `/bin/grep -nE -i "<S1 terms>\|crash\|rig.*(down\|hang)" docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md` | TODO_CONSOLIDATED :83, :193, :280-283, :323; LEADS L-3 |
| S8 | `/bin/grep -rliE "<S1 terms>" docs/exchange/` + the Jul–Nov keyword sweep of `docs/exchange/**` | no fault-subject hit outside the PWC-backup threads |
| S9 | `ls docs \| /bin/grep -iE "register\|fault\|crash"` | is there an existing register? none found |
| S10 | `git log --all --format='%h %as %s%n%b'` grepped for fault, crash, riser, pcie, gen1/2, aer, iommu, sdma, hsa_, gfxoff, ppfeature, dpm, sclk, undervolt, powercap, lockup, gpu_recovery, free_all_blocks, serialize, launch_blocking, seed_cap, storm, semaphore, cooldown, jitter, rig6600, reboot, hard reset, freez | 77 matching subjects plus bodies; deleted docs read with `git show <sha>:<path>` |
| S11 | Module docstrings of the 127 `apply_s*.py` / `verify_s*.py` in the repo root | Part E |
| S12 | Gitignored files in the main tree `/home/michael/distributed_prng_analysis` (`git status --ignored --porcelain`, 455 entries), read-only | Part E |
| S13 | `/home/michael/dashboard_work` inventory and fault-related files, read-only | Part E |
| S14 | Memory notes `rig-ct100-four-cores-eight-gpus`, `rig-ct100-kernel-log-access-gap`, `rig-boot-selector-proxmox-recovery`, `rig-throughput-heterogeneity`, `rig-gpu-sku-is-rx-6600-xt` | cited as MEM |

### 2. Documents found that already decide, authorise, schedule or record the subject

These are the anchor documents. The full per-era file lists are in Parts A–E.
- **No existing register** of GPU-fault prior work (S9). This document is `NEW`.
- `docs/PROJECT_FILE_CATALOG.md` §3.12 "Infrastructure, GPU faults and cluster stability" (:590-610) and rows :128 (S159G crashes), :131 (S162 Option B, RULED "No"), :132 (S163KARG incident, RECORD), :482 (S151 RCA "RESOLVED").
- **The project's own prior consolidation:** `docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STAGE_A.md:322-373`, a PWC-era stability table and envelope. It states: "The record does not support a rig-count limit. It supports a per-rig concurrency × chunk-size limit" (:372-373). This register re-uses it and does not re-derive it.
- **Rulings:**
  - S162 Option B ruled ineffective (`docs/TB_UPDATE_S162_OPTION_B_RESULTS.md`).
  - TB provisional ruling on free_all_blocks (`docs/PROPOSAL_FREE_ALL_BLOCKS_REPLACEMENT_v1_0.md:268`, `:275`).
  - KARG fix retained (`docs/TB_Incident_Report_rrig6600c_S163KARG.md:150-156`).
  - Owner, 2026-05-11/12: stop debugging the cliff and switch backends to RANGE-MINER (`docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:94`; Part C E52).
  - Owner ruling 3, 2026-11-11: power caps, circuit check and power telemetry dropped; netconsole stays (`docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:21`). No stress or freeze-reproduction runs (`docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:15-26`, `:259-260`).
- **Open items:**
  - `docs/LEADS.md` L-3, the 2026-08-22 freeze, OPEN/UNDETERMINED (:179-182).
  - `docs/TODO_CONSOLIDATED.md:323`: rrig6600c GCVM root cause "avoided, not eliminated", GCVM criterion UNAVAILABLE.
  - `:281` DPM harness; `:282` crash_forensic_daemon; `:283` rig crash monitors.
- **Forensics:** `docs/RIG_CRASH_FORENSIC_20260822.md`, as corrected by `e75a6eb`.

### 3. Labels for what this document raises
| Item | Label |
|---|---|
| The register itself (one date-ordered list across Jan–Nov 2026) | `NEW` |
| Each entry in Parts A–E | `ALREADY RECORDED`; each cites its source document |
| PWC-era isolation table | `ALREADY RECORDED`: `CLAUDE_CODE_REPORT_PWC_BACKUP_STAGE_A.md:322-373`. Parts A–D extend it with rows that table does not carry |
| Contradictions between documents (§ "Cross-era contradictions" below) | `NEW` as a collected list; each side is `ALREADY RECORDED` |
| Git-only facts no document states (e.g. `ad4aa59` re-inserting `free_all_blocks()` at `sieve_gpu_worker.py:91-92`; the S170 cooldown "DEPLOYED" but no implementing commit) | `NEW` as observations; anchored to commits |
| Fault classification and Tables 1/2 (owner direction 2026-11-15 15:39 PST) | `NEW` as a classification; every evidence line it cites is `ALREADY RECORDED` in a doc or a main-tree log |
| "The pre-PWC search that did not crash" (owner direction 15:44 PST) | `NEW` as a section. The torch history (41165dd, b425171, 20b0e7c, 0a316b8) is `ALREADY RECORDED` in commits but was not collected anywhere |
| "How to record the next event" | `NEW`, a **proposal**. It changes no gate or ruling; adoption is the owner's |
| Any item that would change a ruling | **none raised.** Nothing here proposes reopening owner ruling 3 or the backend-switch ruling |

### 4. Not searched / not available
- The Proxmox hosts and rigs: no SSH by brief, so no dmesg, kern.log, GRUB, modprobe.d, sshd_config, BIOS or `/sys/module/amdgpu/parameters`.
- ser8 archives (`docs/TODO_CONSOLIDATED.md:277`: ~20 changelogs are ser8-only, so pre-July coverage is incomplete); the private repository; Claude project files outside the repo.
- The external repos `proxmox-gpu-passthrough` (the runbook moved there at `2cf23cf`) and `rx6600-llm-inference`.
- `/home/michael/tfm_worktrees/rig-gpu-fault` (excluded by brief); `/home/michael/run4_attempt1_20261115_090553/`.
- Missing referenced files:
  - `PROPOSAL_ROCm_Monitoring_Desync_Mitigation_v1_0.md`, `PROPOSAL_S157_rrig6600c_4worker_cliff.md`;
  - the S159A–F changelogs;
  - `crash_forensic_daemon.py`, `verify_s172_final.py`;
  - MEM `rig-crash-2026-08-22-hang-not-power`.
- No session changelogs exist for September or October 2026.

---

## Null results first

Denominators are given where the record gives them.
- **No GCVM_L2 or gfxhub page fault is recorded before 2026-04-02.** The Jan–Mar pass searched 168 period docs plus the deleted changelogs in history. The first captures are `docs/TB_SUBMISSION_S159G_RIG6600_CRASHES.md:3` and `bb65aa9:docs/SESSION_CHANGELOG_20260402_S159G.md`. Jan–Mar faults are recorded as freezes, SMU `0xFFFFFFFF`, OOM, segfaults, D-state and kernel panics (Part A).
- **No root cause has survived.** It was declared "confirmed" or "RESOLVED" at least five times and each claim was later superseded:
  - S151 sshd MaxSessions (`ROOT_CAUSE_ANALYSIS_RRIG6600C_S151.md:4`);
  - S159G env propagation (`TB_FINAL_UPDATE_S159G_ROOT_CAUSE_CONFIRMED.md:2`);
  - S160 inter-chunk cleanup (`SESSION_CHANGELOG_S160.md:10`);
  - S162 amdgpu-dkms (`SESSION_CHANGELOG_S162_FINAL.md:35`, "ROOT CAUSE");
  - S165 Kaspa OC "Was Causing All Crashes" (`SESSION_CHANGELOG_20260420_S165.md:22-34`).

  The standing record says "avoided, not eliminated" (`docs/TODO_CONSOLIDATED.md:323`).
- **No standing rig kernel log existed for any RANGE-MINER fleet run, 2026-07 → 2026-11.** CT100 is an unprivileged LXC (`docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md:251-276`), and the GCVM criterion always reports UNAVAILABLE (`docs/PHASE6_PREREQS.md:415-432`). The one exception is a single host-dmesg read for Phase 6.0 (`docs/TEAM_ALPHA_PHASE_6_0_RETURN_TO_BETA.md:69-80`).
- **The two Jul–Nov host losses that preceded today have no fault signature recorded:** 2026-08-19 (PIECE-MATCHER, MEM only) and 2026-08-22 run 3 (`docs/RIG_CRASH_FORENSIC_20260822.md`).
- **Experiments that were set up, or called decisive, but have no outcome recorded:**
  - S170 pool8 + 500 ms cooldown, called "the decisive experiment" (`SESSION_CHANGELOG_20260425_S170.md:104-110`);
  - S171 pool7;
  - S171 pool8 + 100 ms coordinator throttle;
  - the 100k/150k/200k stability curve;
  - S128 seed-cap probes, tables TBD (`SESSION_CHANGELOG_20260307_S128.md:83-111`);
  - the Step-1 GPU benchmark suite (`STEP1_GPU_BENCHMARK_SUITE.md:640-648`);
  - the S157 4-worker-cliff phases (`S157_CHAT_PROMPT.md:45-63`);
  - D1.1 configuration (`PROPOSAL_S172_RANGE_MINER_v1_4_4.md:94`).
- **No clean run at pool 8 × 3 rigs at the 2,000,000-seed AMD cap was found in the PWC era** (`docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STAGE_A.md:366-367`).
- **Main-tree netconsole capture.** `logs/netconsole_listener.log` covers 2026-04-17 → 2026-11-15 and has 504 `GCVM_L2_PROTECTION_FAULT` lines. From 2026-05-15 to 2026-08-22 it holds **no amdgpu datagrams from any rig**. Correction made in the classification pass: Part E's "0 kernel datagrams" was too strong. Bare-metal .162/.120 sent i915 and snd_hda lines on 06-14 and 06-15 (`:9295-9307`). Nothing then arrives from 06-15 20:39 until 08-22 19:25:50, and no Proxmox host sent anything before 08-22 19:25:50 (`:9453`). So Phase 6/7, Gate-12 attempts 1–9 and the 08-19 incident have no netconsole evidence either way (Part E). All 23 S174 `final_dmesg_tail.txt` files are 0 bytes (Part E).
- **`/home/michael/dashboard_work`: 0 of 131 files record a GPU fault, reset or mitigation** (Part E).

---

## Context entry: today (2026-11-15)
- **Run 4 attempt 1 (`distributed_config_t1_57dc50fa`, 09:07–09:19) was clean.** 25 workers were eligible every phase with 0 disconnects; survivors were 60 / 23,568; rigs were 8/8 afterwards (`docs/CLAUDE_CODE_INSTRUCTIONS_STEP1_CREDIT_WAL_20261115.md:50-63`).
- **Run 4 retry (`distributed_config_t1_ab6318d4`) lost two hosts.** pve-rig6600 and pve-rig6600c were lost at phase-4 start: amdgpu `GCVM_L2_PROTECTION_FAULT`, then status `0xFFFFFFFF`, then device inaccessible, then host hang (`docs/CLAUDE_CODE_INSTRUCTIONS_GPU_FAULT_PRIOR_WORK_20261115.md:1`, the brief; no other record in this tree).
- **Recorded fact, not a conclusion:** run 3 on 2026-08-22 also lost the rrig6600 and rrig6600c hosts 1–6 s after the phase-4 dispatch at 18:32:01, when all 25 workers resumed together (`docs/RIG_CRASH_FORENSIC_20260822.md:303-312`).
- **The main-tree netconsole log holds today's fault lines.** This register lists their location; the comparison belongs to the `rig-gpu-fault` worktree:
  - `/home/michael/distributed_prng_analysis/logs/netconsole_listener.log:9639` shows `192.168.3.121 amdgpu 0000:19:00.0 GCVM_L2_PROTECTION_FAULT_STATUS:0x00841011` at 13:21:29.062.
  - `:9650` shows `192.168.3.163 amdgpu 0000:16:00.0 ...:0x00801231` at 13:21:29.063, the same second.
  - `0xFFFFFFFF` follows on .121 19:00.0 and .163 06:00.0 / 16:00.0 (`:9660-9730`).
  - Also: `.121 ata5: SError { UnrecovData 10B8B BadCRC }` at 09:06–09:07 (`:9511-9590`, during attempt 1); `.155 amdgpu 0000:19:00.0 SMU response:0xFFFFFFFF ... TransferTableSmu2Dram` at 13:47:28 (`:10098-10100`); `.121 "Overdrive is enabled"` at boot 13:55:46 (`:10102`).
  - None of these lines is in `docs/`. All were read by this register.
- Also recorded today: "root key auth to the Proxmox hosts works from VM101; netconsole needs the host console level raised with `dmesg -n 7`" (`docs/CLAUDE_CODE_INSTRUCTIONS_STEP1_CREDIT_WAL_20261115.md:110`).

---

## Timeline index (Parts A–E hold the entries)

| Part | Window | Entries | Headline of what the record holds |
|---|---|---|---|
| A | 2026-01-01 → 03-31 (S70s–S159) | 40 | Freezes, SMU `0xFFFFFFFF`, OOM, HIP init storms. Fixes: ppfeaturemask / gfxoff, batching caps, stagger, MaxSessions, per-worker CUPY cache, BIOS reflash, worker caps on rrig6600c, S158 bounded dispatch |
| B | 2026-03-30 → 04-17 (S159–S163) | 49 | First GCVM_L2 / SQC captures (netconsole). Env vars, iommu, GRUB, cleanup, AMD_SERIALIZE, scratch-reclaim, cwsr, amdgpu-dkms, free_all_blocks, seed-cap ladders, the package-removal disaster and clone restore |
| C | 2026-04-17 → 05-12 (S163KARG–S174) | 52 | Device lost from bus, KFD faults, DPM pin, Kaspa OC (reverted), debug trap, cwsr/mcbp, jitter/pacing, pool 6/7/8 × chunk boundary map, first-fault instrumentation, backend switch |
| D | 2026-07-01 → 11-15 (Proxmox / RANGE-MINER) | 37 | Proxmox + dkms runbook and cmdline, CT100, Phase 6.0 clean, Gate-12 attempts 1–9 (no host loss), 08-19 and 08-22 host losses, owner ruling 3, Run 4 attempt 1 clean, Run 4 retry |
| E | non-docs surfaces | — | apply/verify docstrings, git-only facts, main-tree gitignored files and logs, `dashboard_work` |

---

## The four lists (consolidated across Parts A–E; the per-era lists, with full anchors, are at the end of each Part)

The labels below are **what the record says**, not this register's judgement. Several "helped" items were later contradicted; that is noted inline.

### (1) Recorded as helping

| # | Date / session | Item (exact setting) | Recorded effect | Anchor |
|---|---|---|---|---|
| 1 | 01-17 | `amdgpu.ppfeaturemask=0xffff7fff` (GFXOFF off) | gfxoff=00, rocm-smi populated; part of the 100-trial 100% envelope | `f614736`; `docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md:896-906` |
| 2 | 01-18/20 | sample_size ≤ 450–1000 at 12 concurrent; per-node batch cap 6 | 100% / 25/25 | `CHAPTER_9:910-917`; `49e8ad4` |
| 3 | 01-20 | SMU polling cut (kill `watch -n 0.5 rocm-smi`, fan LOOP_SECS 3→20) | "No SMU errors in dmesg post-fix" | `docs/SESSION_CHANGELOG_20260120.md:34-37` |
| 4 | 01-24/25 | chunk_size 1000 | 99/99 at 12 GPUs | `docs/SESSION_CHANGELOG_20260124_OOM_FIX.md:233-241` |
| 5 | 01-26, then 04-03 | Inter-chunk `_best_effort_gpu_cleanup()` (sieve_filter.py, then the ZMQ worker `7e1a62e`) | 20/20 with 0 hangs; S160 full trial with zero crashes and zero netconsole | `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:1034`; `docs/SESSION_CHANGELOG_S160.md:88-93` |
| 6 | 04-04 | Same cleanup in the TCP worker (`b3c3207` / `64d0307`) | crash delayed from ~18 s to ~50 min; **not prevented** | `PROPOSAL_S162_RRIG6600C_CRASH_ROOT_CAUSE_v1_0.md:176-177` |
| 7 | ~02-01 | rig-6600b GPU[4] reseat + PCIe set to Gen 1 | "Sensors now reading healthy" | `docs/SESSION_CHANGELOG_20260201.md:152` |
| 8 | 02-11 S80 | udev perf=high + `amdgpu.gfxoff=0` + reboot | all 26 GPUs available | `docs/SESSION_CHANGELOG_20260211_S80.md:54-56` |
| 9 | 03-09 S133 | semaphore throttle restored (`288de2a`) + hybrid strategies param | workers survive all passes | `docs/SESSION_CHANGELOG_20260309_S133.md:26, 64` |
| 10 | 03-16 S146 | pool 4 per rig + kernel-tail split / int32 casts / clamp | 3 trials, zero crashes; pool 4 "validated stable (not 8)" | `docs/SESSION_CHANGELOG_20260316_S146.md:92-103, 140` |
| 11 | 03-20 S149-B | direct `Device(gpu_id)`, masking removed | 8/rig, preprod 5/5 (Run 1 later crashed) | `docs/SESSION_CHANGELOG_20260321_S149.md:36-42, 63-66` |
| 12 | 03-21 S151 | rrig6600c sshd `MaxSessions 5→50`, `ClientAliveInterval 300→0` | declared root cause; 26/26 next session. **Crashes resumed later** | `docs/ROOT_CAUSE_ANALYSIS_RRIG6600C_S151.md:4, 114-128`; `docs/SESSION_CHANGELOG_20260322_S152.md:72-77` |
| 13 | 03-22 S152 | per-worker `CUPY_CACHE_DIR` | "eliminated rrig6600c crash" | `docs/SESSION_CHANGELOG_20260322_S152.md:48` |
| 14 | 03-24/26 | rrig6600c capped at 4, then 3 workers; BIOS reflash B36FF608.BSS | 3-trial runs complete; hard freeze became a soft CuPy OOM | `SESSION_CHANGELOG_20260324_S155.md:80-85`; `SESSION_CHANGELOG_20260326_S156.md:73-90, 96-101` |
| 15 | 03-29 S158B | bounded dispatch (537 threads removed) | first clean 2-rig run, 537/537 | `docs/SESSION_CHANGELOG_20260329_S158.md:33-40` |
| 16 | 04-02 | ZMQ SQLite lease expiry | run completed after a rig reboot (resilience only) | `docs/TB_SUBMISSION_S159G_RIG6600_CRASHES.md:138-149` |
| 17 | 04-02 → | netconsole | captured the faults (instrumentation) | `bb65aa9:docs/SESSION_CHANGELOG_20260402_S159G.md`; Part B E4–E5 |
| 18 | 04-05 S162 | amdgpu-dkms 6.12.12 | "Extended stability to 25 min" (c only); "ROOT CAUSE" (all rigs). **Contradicted**: 2M still crashed at ~1400 s | `SESSION_CHANGELOG_S162_FINAL.md:35-42`; `PROPOSAL_FREE_ALL_BLOCKS_REPLACEMENT_v1_0.md:264` |
| 19 | 04-05 S162 VICTORY | `seed_cap_amd=100000`, 26 GPUs, DKMS | 42:36, 3 trials, netconsole silent | `docs/SESSION_CHANGELOG_S162_VICTORY.md:13-24` |
| 20 | 04-12/13 S163 | clone restore from rrig6600b disk; `set_limit()` 256 MB in the TCP path; fresh reboot between runs | 8/8 GPUs; ladder stable to 300K | repo-root `CHANGELOG_S163.md:97, 104`; `SESSION_CHANGELOG_S163_FINAL.md:66-81` |
| 21 | 04-18 | GDM/Xorg disabled, multi-user.target | "no SDMA timeouts post-reboot" | `docs/TB_Incident_Report_rrig6600c_S163KARG.md:260-261` |
| 22 | 04-20 S165 | Kaspa OC reverted to stock `high` DPM + `HSA_ENABLE_DEBUG_TRAP=0` | 2 trials, ~14 min, netconsole silent | `docs/SESSION_CHANGELOG_20260420_S165.md:166-178` |
| 23 | 04-22 S166 | `amdgpu.cwsr_enable=0 mcbp=0` on rrig6600 | 75+ min clean against ~40 min before, at ~22K vs 95K+ s/s | `docs/SESSION_CHANGELOG_20260422_S166.md:92-97` |
| 24 | 04-24 S168/S169 | first-assignment jitter + per-worker gap pacing | 200k × 2 trials clean. **Contested** by the `b6cabe9` body ("200k remains unstable") | `docs/SESSION_CHANGELOG_20260424_S167_S168_S169.md:130` |
| 25 | 04-25 S170 | `worker_pool_size=6` | 5/5, no GCVM (2 rigs per log name) | `docs/SESSION_CHANGELOG_20260425_S170.md:66-74` |
| 26 | 05-01 S172 | pool 6 / 50K; pool 8 / 25K (26 GPUs) | 29:30 clean; 1:30:43 with one recovered eviction | `docs/ROCm_Saturation_Report_S172.md:31, 33` |
| 27 | 05-05 | reboot of all 3 rigs before FT=0.73 Run 3 | clean, 1:55 | Part C E46 |
| 28 | 07-09 | removing `nomodeset`, `amdgpu.runpm=0` (Proxmox host cmdline) | restored SMU telemetry (canary, not prevention) | `99f962e` |
| 29 | 07 → 11 | RANGE-MINER persistent per-GPU daemons (~52 launches/trial vs PWC ~17K); per-worker CUPY cache; `ROCR_VISIBLE_DEVICES=N` + `--device-index 0`; 3 s launcher stagger | Phase 6.0 host dmesg clean; Gate-12 attempts 3 and 9, 08-22 run 1 and Run 4 attempt 1 finished four stages with no host loss. **Contradicted** by 08-22 run 3 and today's retry | `docs/TFM_PROJECT_FACTS_SKILL.md:2533-2537`; `scripts/launch_fleet_manual.sh:36-43, 150`; Part D |
| 30a | 05-11 | S174 D1.2: pool 6 (per log name `pre_D1_2_pool6`), forced FT=0.73, 50K | `classification: VALID_CLEAN` (D1.1 at pool 8 the same day: `VALID_ACTIVE_FAULT`). Not in any docs changelog | main tree `logs/S174_D1_FT073_50K_425M_20260511_192244_summary.txt:60`; `..._175133_summary.txt:60` |
| 30 | 08-28 → 11-15 | Run 4 pinned to the attempt-9 low-volume geometry + survivor-volume cap | attempt 1 clean | `docs/CLAUDE_CODE_INSTRUCTIONS_STEP1_CREDIT_WAL_20261115.md:50-63` |

### (2) Recorded as not helping, or making it worse

| # | Date / session | Item (exact setting) | Recorded result | Anchor |
|---|---|---|---|---|
| 1 | 01-16 | scripts_coordinator stagger fix | reverted; cause unknown | `a47ba88` |
| 2 | 01-17 | kernel cmdline `amdgpu.gfxoff=0` | "ignored on 6.8.0-90". Re-used in S80; ignored again by dkms 6.12.12 (04-18); v1.2 runbook puts it back on the cmdline | `f614736`; `TB_Incident_Report_rrig6600c_S163KARG.md:242`; `99f962e` |
| 3 | 01-23 | HIP kernel cache clear; GPU warm-up barrier; HSA_OVERRIDE fix | failures continued / "didn't help" | `docs/SESSION_CHANGELOG_20260123_PART2.md:15-17` |
| 4 | 03-21 | rrig6600c 6-worker cap | reverted within 46 min | `38393b5` → `cc113b5` |
| 5 | 03-21 → 03-25 | `CUPY_CUDA_MEMORY_POOL_TYPE=none`; 256 MB pool cap; `CUPY_GPU_MEMORY_LIMIT`; `ROCR_VISIBLE_DEVICES` restored; M.2 SATA replaced; CPU i5-8400T → i5-9400 | "OOM persists" (each) | `docs/SESSION_CHANGELOG_20260324_S155.md:18-31, 41, 46` |
| 6 | 03-23 | explicit GPU `del` + gc | "Did not resolve the crash" | `docs/SESSION_CHANGELOG_20260323_S153_S154.md:134-136` |
| 7 | 03-29 | Zeus ulimit raise; S158C inter-node spawn stagger 10 → 30 s | "did NOT fix"; stagger "caused spawn crashes", reverted | `docs/SESSION_CHANGELOG_20260329_S158.md:69-81`; `d1301e1` |
| 8 | 04-02 S159G | seed cap 5M → 2M; `iommu=pt`; GRUB standardisation; `/etc/environment` standardisation; `HSA_ENABLE_SDMA=0`; `HSA_ENABLE_RUNTIME_POWER_MGMT=0` | "Still crashed" (each) | `docs/TB_SUBMISSION_S159G_RIG6600_CRASHES.md:105-112` |
| 9 | 04-03 | `/etc/environment` protective vars | "insufficient — GPU crashes continued" | `docs/SESSION_CHANGELOG_S160.md:10` |
| 10 | 04-04 | Zeus TCP sysctl + dispatch `Semaphore(8)` | "no effect on crash" / "unnecessary" | `TB_UPDATE_S162_OPTION_B_RESULTS.md:125-126`; `SESSION_CHANGELOG_S162_FINAL.md:32` |
| 11 | 04-04 | `AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3` on rrig6600c | crashed ~1–2 min in; RULED no | `docs/TB_UPDATE_S162_OPTION_B_RESULTS.md:21-25, 66-71` |
| 12 | 04-05 | `HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0` (rrig6600c) | "No effect" / "partial, not a fix" | `SESSION_CHANGELOG_S162_FINAL.md:29` |
| 13 | 04-05 | `HSA_NO_SCRATCH_RECLAIM=1`, all AMD (`606f94f`) | **worse**, crash at 219 s; reverted `a854647` | `SESSION_CHANGELOG_S162_FINAL.md:27` |
| 14 | 04-05 | CuPy pool disabled, raw hipMalloc (`d79e558`) | **worse**, crash at 358 s; reverted `070199d` | `SESSION_CHANGELOG_S162_FINAL.md:28` |
| 15 | 04-05 | kernel pre-compile at worker start (`22bb145`); reverse-sieve warm-up (`8d91311`) | "made crashes worse"; reverted `ac32cd9` | `ac32cd9`; Part B E27, E30 |
| 16 | 04-05 | seed cap 1M / 500K / 2M under 3 rigs with DKMS | delayed crashes; 2M crash at ~1400 s | `PROPOSAL_FREE_ALL_BLOCKS_REPLACEMENT_v1_0.md:264-265`; `SESSION_CHANGELOG_S162_FINAL.md:33` |
| 17 | 04-05 | `amdgpu.cwsr_enable=0` (S162) | "No effect" / "unnecessary" / KIQ fence timeouts on rrig6600b (three verdicts) | `SESSION_CHANGELOG_S162.md:82-92`; `SESSION_CHANGELOG_S162_FINAL.md:30` |
| 18 | 04-07..10 | multimedia package removal; reinstall; reboots; GRUB `amdgpu.gpu_recovery=1 amdgpu.sched_jobs=256` | broke the NIC and destabilised rigs; did not fix; 3362/5369 chunks failed | repo-root `CHANGELOG_S163.md:51-81` |
| 19 | 04-12/13 | 500K cap; ≥300K without a reboot between runs | rrig6600 crash; "triggers GPU page faults" | repo-root `SESSION_CHANGELOG_S163_FINAL.md:74, 78-81` |
| 20 | 04-17 | KARG `cp.int32` fix | the rrig6600c fatal happened with it active ("appears distinct"); kept | `TB_Incident_Report_rrig6600c_S163KARG.md:150-156` |
| 21 | 04-19/20 | Kaspa profile 2250 MHz / −150 mV (+ `audio=0 dc=0 lockup_timeout=30000 gpu_recovery=1`) | "Was Causing All Crashes"; ASIC reset −22 on rrig6600b 09:00.0; reverted | `docs/SESSION_CHANGELOG_20260420_S165.md:22-34` |
| 22 | 04-20 | stock `high` DPM | "unstable in longer runs (5-50 trials)"; no evidence recorded | `SESSION_CHANGELOG_20260420_S165.md:111-113` |
| 23 | 04-24 / 05-09 | `cwsr_enable=0` on rrig6600c; cwsr=0 rigs rebooted in S174 | "faster crashes" (no experiment recorded); post-reboot crashes, cwsr named the common factor | `SESSION_CHANGELOG_20260424_S167_S168_S169.md:180`; `SESSION_CHANGELOG_20260509_S174_D1.md:40-44` |
| 24 | 04-24 | S168 jitter alone | Trial-2 rrig6600c page fault at t+30 s | `SESSION_CHANGELOG_20260424_S167_S168_S169.md:107-108, 129` |
| 25 | 04-25 S170 | 512 → 2048 skip buffers; LIVE-GPU-PROBE; removing rrig6600b (2 rigs); pool 8 + 50 ms cooldown | still crashes (each); cooldown run crashed at ~10 min | `docs/SESSION_CHANGELOG_20260425_S170.md:41, 52, 58-64, 98-102` |
| 26 | 05-01 S172 | pool 8 at 50K / 100K, 26 GPUs | GCVM_L2 at ~23 / ~12 min | `docs/ROCm_Saturation_Report_S172.md:34-35` |
| 27 | 07-08 | in-kernel amdgpu at 8 procs/GPU; amdgpu-dkms host RSS leak ~5.4 GB/hr | "can't handle"; leak open | `ea372c6` runbook :150-152 |
| 28 | 08-19 (MEM only) | sustained multi-minute 8-GPU load on rrig6600 CT (PIECE-MATCHER) | rrig6600 + host .121 down; 5–16 s bursts fine | MEM `rig-ct100-four-cores-eight-gpus.md:35-41` |
| 29 | 07 → 11 | in-CT dmesg; empty netconsole; RAS counters on RDNA2; boot-notify | cannot tell no-event from not-observed | `S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md:260-264`; `CLAUDE_CODE_REPORT_DEFECT_A_TRANSPORT_RECOVERY.md:270-274`; `FLEET_STATE_REQUIREMENTS_v1.md:244-246` |

### (3) Proposed and never tried (or no result ever recorded)

| # | Proposed | Item | Anchor |
|---|---|---|---|
| 1 | 01-23 | add RAM (16 GB) to rigs; cap 5 workers/node | `SESSION_CHANGELOG_20260123_PART2.md:44-52` |
| 2 | 01-20 | job-batching Option B (WATCHER trial cap 20) and C (`INTER_NODE_STAGGER=5.0`); drop_caches between batches; STAGGER_REMOTE 0.5 → 1.0 s | `PROPOSAL_Job_Batching_Pipeline_Stability.md:106-143`; `_v2.md:153-167`; `PROPOSAL_FINAL_ROCm_HIP_Init_Fix_v2_1.md:140` |
| 3 | 01-22/24 | GPU[4] power/riser investigation and replacement; GPU[10] exclusion; `rocm-smi --resetclocks` post-batch | `SESSION_CHANGELOG_20260122.md:116, 124`; `PROPOSAL_Infrastructure_Improvements_v1_0.md:126-127, 154-185` |
| 4 | 03-07 | Phase A/B seed-cap probes (single card vs full rig, "multi-worker ROCm tax"); Step-1 GPU benchmark suite | `GPU_THROUGHPUT_INVESTIGATION_PLAN_v1_0.md:64-131`; `STEP1_GPU_BENCHMARK_SUITE.md:640-681` |
| 5 | S120 → | node watchdog for D-state / auto-reboot; rig crash monitors; crash_forensic_daemon fixes (`journalctl -k`, stale-log Option B) | `TODO_MASTER_S120.md:70`; `docs/TODO_CONSOLIDATED.md:190, 282-283` |
| 6 | 03-22 | `#define MAX_SURVIVORS` bounds guard in hybrid kernels | `SESSION_CHANGELOG_20260322_S152.md:44, 52-54` |
| 7 | S154 | disable syncthing, 8 GB swap, `vm.overcommit_memory` on rigs | `TODO_MASTER_S154.md:39-54`; `docs/TODO_CONSOLIDATED.md:280` |
| 8 | 03-27 | S157 rrig6600c 4-worker cliff: serialized 12–15 s start + warm-up, idle-only test, GPU-set permutation for riser/slot, GPU/riser swap, VSZ/RSS | `docs/S157_CHAT_PROMPT.md:45-63` |
| 9 | 03-26 | rrig6600c at 8 workers post-BIOS (Test D) | `SESSION_CHANGELOG_20260326_S156.md:102, 169` |
| 10 | 04-02 | linux-firmware 3.40 → 3.41 (rrig6600, c); remove `pci=assign-busses,hpbussize=0x33` from rrig6600 GRUB; `HIP_PATH` in `.bashrc`; remove the rrig6600 `.bashrc` `$PS1` guard | `TB_SUBMISSION_S159G_RIG6600_CRASHES.md:120, 156-180`; `TB_FINAL_UPDATE_S159G_ROOT_CAUSE_CONFIRMED.md:128-129` |
| 11 | 04-04 | PROPOSAL_S162 Option A (gc-only cleanup; `apply_s162_safe_cleanup.py` exists, never run), Option C (rrig6600c 4-worker cap), Option D (30–60 s rrig6600c start stagger); research KFD queue limits / ROCm 6.4.3 KFD multi-process bugs | `PROPOSAL_S162_RRIG6600C_CRASH_ROOT_CAUSE_v1_0.md:285-362`; `TB_UPDATE_S162_OPTION_B_RESULTS.md:142-145, 164-186` |
| 12 | 04-05 | free_all_blocks Option A (synchronize before free) and C (`MemoryAsyncPool`); staged validation at 1M / 2M after removal; external `rocm-smi --showmeminfo vram` every 30 s | `PROPOSAL_FREE_ALL_BLOCKS_REPLACEMENT_v1_0.md:80-158, 277-281, 295` |
| 13 | 04-05 | persistent `perf=high` systemd service; run at the "Known Working" 5M cap | `SESSION_CHANGELOG_S162.md:169`; `SESSION_CHANGELOG_S162_FINAL.md:89-99` |
| 14 | 04-18 | KARG controlled A/B; `ps auxf` isolation check; 100K × 3 "GDM removed" discriminator; staged ladder 100K → 1M | `TB_Incident_Report_rrig6600c_S163KARG.md:152, 212-227, 264-268` |
| 15 | 04-20 | DPM harness: 900 mV absolute, 2100–2200 MHz, per-rig mem | `SESSION_CHANGELOG_20260420_S165.md:115-147`; `docs/TODO_CONSOLIDATED.md:281` |
| 16 | 04-22 | cwsr two-sequential-run test on rrig6600; cwsr on rrig6600c with dkms; revert cwsr after D1 | `SESSION_CHANGELOG_20260422_S166.md:178-183`; `SESSION_CHANGELOG_20260509_S174_D1.md:45, 76` |
| 17 | 04-25 | stability curve 100k / 150k / 200k × 5; pool 8 + 500 ms cooldown; pool 7; coordinator 100 ms throttle; backpressure / token-bucket scheduler; worker-lifecycle hardening; VBIOS L03 → L04 on rrig6600c card1 (vetoed) | `SESSION_CHANGELOG_20260425_S170.md:104-110, 283, 291-302, 341-345`; Part C E29, E41 |
| 18 | 05-02 | amdgpu `hws_max_conc_proc=4`, `sched_policy=1`, `queue_preemption_timeout_ms=30000`, `max_num_of_queues_per_device=256`, `no_queue_eviction_on_vm_fault`, `enforce_isolation=1`; in-kernel driver instead of DKMS; ROCm upgrade; B-2 HIP stream pooling; B-3 per-rig semaphore; journalctl watchdog | `docs/ROCm_Saturation_Report_S172.md:117-180` |
| 19 | 07-07 | warm-up / power caps before 8-GPU cold load; READY warm-up "sequentially or in pairs"; VM-passthrough gate with FLR reset test | `PROPOSAL_Proxmox_LXC_Rig_Migration_v1_0.md:108-109`; `S172_INFRASTRUCTURE_INTERFACE_v1_0.md:71-75`; `PROPOSAL_Infrastructure_Reconciliation_S172_v1_0.md:107-116, 126-127` |
| 20 | 08-22 / 08-28 | per-rig supply logging + circuit map; power-cap + off-host power telemetry verification. **Both DROPPED by owner ruling 3 (2026-11-11)** | `RIG_CRASH_FORENSIC_20260822.md:356`; `TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:149`; `PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:21` |
| 21 | 08-22 | ≥1 Hz per-host resource series across the phase-3/4 boundary; CT MemAvailable/nproc; switch counters; explain the 15 staging failures; volume reproduction (reproduction removed by owner) | `RIG_CRASH_FORENSIC_20260822.md:357-381`; `PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:24` |
| 22 | 08-28 → 11 | netconsole reboot persistence; pip-freeze diff across CTs (Python 3.10.12 vs 3.10.4); baseline `/sys/module/amdgpu/parameters`; decide the dead `HSA_ENABLE_SDMA` prelude (hostname guard `rig-6600*` vs `rrig6600*`) | `RUNBOOK_NETCONSOLE_REARM.md:81-83`; `TFM_PROJECT_FACTS_SKILL.md:2552-2558`; `CLAUDE_CODE_REPORT_DELEGATED_RUN_GATES.md:491-494`; `CHAPTER_2_BIDIRECTIONAL_SIEVE.md:1046-1080` |
| 22a | 08-23 (gitignored only) | **STAGGERED-PHASE4-DISPATCH-1**: a 1–2 s per-host phase-4 dispatch offset at the coordinator's stage-assignment seam, "NOT IMPLEMENTED". It is parked out of `docs/BACKLOG.md`. The same text says "The cap applied on 2026-08-23 is an INVESTIGATION instrument". Whether that power cap is still in force is `[UNVERIFIED]` | main tree `logs/brief_i_evidence/parked_for_post_run_commit/BACKLOG.md.with_staggered:495-530` |
| 23 | 11-14 | PWC keep-workers-alive across trials (launch-storm removal), design note only | `CLAUDE_CODE_INSTRUCTIONS_PWC_BACKUP_STEP1_20261114.md:83` |

### (4) Isolation results — which combinations ran clean and which crashed

> **Superseded for fault-class purposes.** This first-version table does not separate GCVM from overload or other failures. Use **Tables 1 and 2 under "Fault classification"** below: 12 of this table's 22 failure rows move out of the GCVM picture, 2 split, and 8 stay. The table is kept unchanged as the record of what version 1 said.

The full per-era tables are in Parts A §LIST 4, B §LIST 4, C §(4) and D §LIST 4. The PWC-era consolidated table is already recorded at `docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STAGE_A.md:322-351`. The rows that discriminate between configurations:

| Date | Rigs | Workers / rig | Seed cap / chunk | Transport / stack | Result | Anchor |
|---|---|---|---|---|---|---|
| 01-20 | 2 rigs | 120 jobs at once (40/rig) | — | SSH, Step 2 | both crash < 60 s | `PROPOSAL_Job_Batching_Pipeline_Stability.md:21-23` |
| 01-20 | 2 rigs | batch 6 / node | — | SSH | 25/25 | `49e8ad4` |
| 03-16 | 3 rigs | 4 | 10M seeds, 3 trials | PWC SSH | zero crashes | `SESSION_CHANGELOG_20260316_S146.md:94-103` |
| 03-20 | 3 rigs | 8 | preprod 5 trials | PWC SSH | 5/5; Run 1 crashed later | `SESSION_CHANGELOG_20260321_S149.md:41-42, 63-66` |
| 03-25/26 | rrig6600c alone | 2 / 3 / 4 | — | PWC SSH | pass / pass / **hard freeze pre-BIOS, OOM post-BIOS** | `SESSION_CHANGELOG_20260326_S156.md:96-101` |
| 03-26 | rrig6600, rrig6600b | 8 each | — | PWC SSH | stable | `SESSION_CHANGELOG_20260326_S156.md:111-115` |
| 03-29 | rrig6600 + Zeus | 8 | 537 chunks, 2M | PWC SSH | clean | `SESSION_CHANGELOG_20260329_S158.md:30` |
| 03-29 | rrig6600c + Zeus | 8 | 537 chunks | PWC SSH | forward clean; reverse stalled 88/537 | `:31-32` |
| 03-29 | rrig6600 + rrig6600c + Zeus | 8 | 537 chunks | PWC SSH | 537/537, 34:15, zero crashes | `:33` |
| 03-29 | all 3 rigs + Zeus | 8 | 537 chunks | PWC SSH | **hard kernel panic on rrig6600c** | `:34` |
| 04-02 | 3 rigs | n/r | 2M | ZMQ | rrig6600 crash on 03:00.0 (SQC) | `TB_SUBMISSION_S159G_RIG6600_CRASHES.md:17-43` |
| 04-03 | 3 rigs + Zeus | n/r | 2M | ZMQ + per-chunk cleanup | zero crashes | `SESSION_CHANGELOG_S160.md:88-93` |
| 04-03/04 S161 | 3 rigs (24) / 26 | 8 | 25 × 2M; 50M total | TCP-PWC | zero crashes (short gates, 53 s) | `docs/SESSION_CHANGELOG_S161.md:37-46` |
| 04-04 | 3 rigs + Zeus | 8 | n/r | TCP-PWC, no cleanup | rrig6600c crash at ~18 s | `SESSION_CHANGELOG_S162.md:19-29` |
| 04-04 | **rrig6600c + Zeus only** | 8 | 3 trials, 1.07B | TCP-PWC + cleanup | **clean**, 18:52 | `SESSION_CHANGELOG_S162.md:36-40` |
| 04-04 | 3 rigs + Zeus | 8 | n/r | TCP-PWC + cleanup + semaphore | rrig6600c SQC cascade at ~50 min | `PROPOSAL_S162_RRIG6600C_CRASH_ROOT_CAUSE_v1_0.md:44-67` |
| 04-05 | 3 rigs + Zeus | 8 | **100K** | TCP-PWC, DKMS | clean, 42:36 | `SESSION_CHANGELOG_S162_VICTORY.md:13-24` |
| 04-05 | 3 rigs | 8 | 2M / 1M / 500K | TCP-PWC, DKMS | crash (~1400 s at 2M) / crash / delayed | `PROPOSAL_FREE_ALL_BLOCKS_REPLACEMENT_v1_0.md:264-265` |
| 04-12/13 | 3 rigs + Zeus | 8 | 10K, 100K, 250K, 300K / **500K** | TCP-PWC, DKMS, set_limit, reboot between | stable / **rrig6600 crash** | repo-root `SESSION_CHANGELOG_S163_FINAL.md:66-81` |
| 04-17 | 3 rigs + Zeus | 8 | 100K, 5 trials | TCP-PWC, KARG | rrig6600b GPU1 silent death; rrig6600c fatal (06/16/19:00.0) | `TB_Incident_Report_rrig6600c_S163KARG.md:20-46` |
| 04-20 | 26 GPUs | 8 | 100K, 2 trials | stock DPM, DEBUG_TRAP=0 | clean, ~14 min | `SESSION_CHANGELOG_20260420_S165.md:166-178` |
| 04-22 | rrig6600 with cwsr=0 mcbp=0 | 8 | n/r | TCP-PWC | 75+ min without a GPU crash on that rig | `SESSION_CHANGELOG_20260422_S166.md:92-97` |
| 04-25 | 2 rigs (no rrig6600b) | 8 | n/r | TCP-PWC | still crashes | `SESSION_CHANGELOG_20260425_S170.md:58-64` |
| 04-25 | 2 rigs (per log name) | **6** | 100k, 5 trials | TCP-PWC, jitter 15 / gap 0.20 | **clean, no GCVM** | `SESSION_CHANGELOG_20260425_S170.md:66-74` |
| 04-25 | n/r | 8 | 100k | + 50 ms cooldown | crash at ~10 min | `SESSION_CHANGELOG_20260425_S170.md:98-102` |
| 05-01 | 26 GPUs | 6 / 6 / 8 / 8 / 8 | 50K / 100K / 25K / 50K / 100K | TCP-PWC | clean / "per operator history" / clean (1 eviction) / **GCVM ~23 min** / **GCVM ~12 min** | `docs/ROCm_Saturation_Report_S172.md:29-35` |
| 05-02/03 S173 | 26 | 8 | 50k: 5 trials vs 1 trial | TCP-PWC | crashed twice on rrig6600c (1m41s, 13 min) vs clean | Part C E45 |
| 05-05 | 26 | 8 | 50k, forced FT=0.73 | TCP-PWC | compute done; GCVM on rrig6600c gpu0+gpu6 **4 min after completion** | Part C E46 |
| 05-09 D1 | 26 | 8 | 50k, 17,000 chunks | TCP-PWC, cwsr=0 on two rigs | compute clean (10:40); SMU breakdown on rrig6600 ~1 s after completion | `SESSION_CHANGELOG_20260509_S174_D1.md:20-31` |
| 05-11 D1.1 / D1.2 | 3 rigs | 8 / 6 | 50K, 425M, forced FT=0.73 | TCP-PWC | `VALID_ACTIVE_FAULT` (.120 and .162 unreachable after) / `VALID_CLEAN` | main tree `logs/S174_D1_FT073_50K_425M_20260511_{175133,192244}_summary.txt:60`; Part E |
| 07-30 | rrig6600 CT | 1 GPU / 1 worker | 8M seeds | RANGE-MINER, Proxmox | clean, host dmesg clean | `S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md:67-68` |
| 08-10 → 08-17 | 3 CTs + Zeus | 8 / CT (25 workers) | 2M const / 1M hybrid | RANGE-MINER, Proxmox | Gate-12 attempts 1–9: **no host loss recorded**; attempt 5 dropped 2 workers (cause unresolved); attempt 9 PASSED | `docs/TFM_PROJECT_FACTS_SKILL.md:1873-2456`; `TB_GATE12_ATTEMPT2_FORENSIC_REPORT.md:74-77` |
| 08-19 (MEM; not TFM) | rrig6600 CT, b, c | 1/2/4/8 GPUs | PIECE-MATCHER | Proxmox | bursts fine; first multi-minute 8-GPU load took rrig6600 + host down; b, c survived | MEM `rig-ct100-four-cores-eight-gpus.md:8-41` |
| 08-22 run 1 | 3 CTs + Zeus | 8 / CT | k=20, 1 phase-3 survivor | RANGE-MINER | clean compute, 713 s | `S172_BRIEF_I_PRODUCTION_SHAPE_CLASSIFICATION_REPORT.md:21-33` |
| 08-22 run 3 | 3 CTs + Zeus | 8 / CT | k=6, **13,146,485 phase-3 survivors** | RANGE-MINER | rrig6600 + rrig6600c **hosts silent 1–6 s into phase 4**; rrig6600b + Zeus OK | `RIG_CRASH_FORENSIC_20260822.md:303-349` |
| 11-15 Run 4 att 1 | 3 CTs + Zeus | 8 / CT | pinned attempt-9 geometry, 60 / 23,568 | RANGE-MINER | clean, ~12 min | `CLAUDE_CODE_INSTRUCTIONS_STEP1_CREDIT_WAL_20261115.md:50-63` |
| 11-15 Run 4 retry | pve-rig6600, pve-rig6600c | n/r | n/r (geometry not stated) | RANGE-MINER | **GCVM_L2 → 0xFFFFFFFF → host hang at phase-4 start** | brief :1 |

**Isolation facts the record states directly (not this register's inference):**
- S170: "Reducing active GPUs per rig from 8 to 6 stabilizes the system completely" (`SESSION_CHANGELOG_20260425_S170.md:74`).
- S172 report: "The empirical boundary is concurrency-driven, modulated by chunk size" (`ROCm_Saturation_Report_S172.md:19`).
- Stage A: "no clean run at the manifest's current 2,000,000-seed cap with pool 8 on three rigs was found" (`CLAUDE_CODE_REPORT_PWC_BACKUP_STAGE_A.md:366-367`).
- Every RANGE-MINER fleet run used Proxmox CT100, 8 workers per CT, TCP, java_lcg, and 2M / 1M caps (Part D).
- **Raw netconsole, recorded fact with no inference:** several GCVM bursts begin in the same second on two or three rigs. Onsets: 04-17; 04-20 at 17:06:40, 18:51:19 and 19:38:40 (all three rigs, status `0x00841051`); 04-28; 05-01; 11-15 13:21:29 (Part E; main tree `logs/netconsole_listener.log`). Most of these codes and times have 0 hits in `docs/`.
- **The code supplies no `HSA_*` to RANGE-MINER workers.** The fleet launcher sets only `ROCR_VISIBLE_DEVICES`, `CUPY_CACHE_DIR` and `PYTHONPATH` (`scripts/launch_fleet_manual.sh:316-318`), and `miner/range_miner_worker.py` sets no `HSA_*`. So the PWC-era variables (`HSA_ENABLE_SDMA=0`, `HSA_ENABLE_DEBUG_TRAP=0`, `HSA_ENABLE_RUNTIME_POWER_MGMT=0`, `AMDGPU_NO_POWER_PROFILE=1`) do not reach miner workers from code. Whether the CT shell exports them is `[UNVERIFIED]` (Part E). `free_all_blocks()` is present at HEAD in `miner/range_miner_worker.py:1069-1070` and `sieve_gpu_worker.py:91-92`.
- **No run in the whole record used any PRNG family other than java_lcg** (Part C §(4) note; Part D §LIST 4 header).

---

## Cross-era contradictions (each side is already recorded; collected here so the next reader does not resolve them by accident)

1. **Root cause.** S151 MaxSessions, S159G env propagation, S160 cleanup, S162 DKMS, S163 CuPy pool growth, S163KARG "narrowed, not identified", S165 Kaspa OC, S167 startup hammer, S170 concurrency × chunk, Phase 7 "PWC launch storm". Each was later superseded (Part B contradiction 1; Part C contradictions 6, 8, 12). Standing: "avoided, not eliminated" (`docs/TODO_CONSOLIDATED.md:323`).
2. **`free_all_blocks()`.** Removed in S163 (`03d9013`, `68ed075`). Put back at import (`d9971d7`). Put back in `_best_effort_gpu_cleanup()` by `ad4aa59` (2026-04-18), where it still is at `sieve_gpu_worker.py:91-92`. The incident report states it was "removed in S162 Option B" (`TB_Incident_Report_rrig6600c_S163KARG.md:182`); Option B was `AMD_SERIALIZE_KERNEL` (Part C contradiction 4).
3. **`amdgpu.gfxoff=0` on the cmdline.** "Ignored" (`f614736`; `TB_Incident_Report_rrig6600c_S163KARG.md:242`; `ea372c6` runbook), yet set on the cmdline in S80 and in runbook v1.2 (`99f962e`), which calls modprobe.d redundant.
4. **`cwsr_enable=0`.** Verdicts: "no effect", "required only on rrig6600c", "unnecessary", "75+ min clean on rrig6600", "faster crashes on rrig6600c", "common factor in post-reboot crashes". Which rigs carried it, and from when, is inconsistent (Part B contradiction 3; Part C contradictions 13–14).
5. **First SQC (inst) fault called "new" in S162.** It had already been captured on rrig6600 03:00.0 on 04-02 (Part B contradiction 2).
6. **Idle vs mid-compute at the 04-17 rrig6600c fault.** `TB_Incident_Report_rrig6600c_S163KARG.md:54-56` against `:131` (verified in this register).
7. **Reboot.** "Fully clears the issue" (`SESSION_CHANGELOG_20260425_S170.md:133-135`) against reboots of rrig6600 and rrig6600c leading to crashes (`SESSION_CHANGELOG_20260509_S174_D1.md:6, 40-41`).
8. **pool 8 / 100K.** S162 VICTORY clean against S172 crash at ~12 min (`CLAUDE_CODE_REPORT_PWC_BACKUP_STAGE_A.md:363-364`).
9. **HSA env vars on CT100.** Runbook v1.4 calls `HSA_ENABLE_SDMA=0` "load-bearing". Phase 6.0, Chapter 9 and the skill say no overrides are needed. The RANGE-MINER prelude's hostname guard is dead, so rig workers run without it (Part D E7, E31).
10. **Kernel-log observability.** Phase 6.0 says "closed from the host"; the evidence record and Phase 7 say not established. Gate-12 attempt 2 claims "zero dmesg GPU faults" with no source recorded (Part D contradictions 1–2).
11. **The e75a6eb netconsole correction.** "Not active" was marked CLOSED, but the arming proof came from boot sessions that started after the freeze. The freeze bound "18:42–18:52" sits against the hosts' last bytes at 18:32:07 (Part D contradiction 5).
12. **Card0 PCIe link on rrig6600.** 16.0 GT/s "Gen4 healthy" against an expected 5.0 GT/s "Gen1 riser" (Part C contradiction 7).
13. **S170 cooldown and S171 throttle.** S170 says "DEPLOYED in production", but no commit carries the implementation (Part C contradiction 18).
15. **Two different `RIG_CRASH_FORENSIC_20260822.md`.**
    - The gitignored copy at main tree `logs/brief_i_evidence/RIG_CRASH_FORENSIC_20260822.md` (24,044 bytes, Aug 22 20:04) differs from `docs/`.
    - It records host-journal reads: unclean-shutdown markers on the two lost hosts and none on rrig6600b (:96-104), and "no runtime amdgpu reset, ring timeout, GPU hang, MCE, thermal event or OOM on any host" (:121-122).
    - It records the host cmdline strings `pci=noaer`, `amdgpu.lockup_timeout=30000`, `amdgpu.gpu_recovery=1` (:113-115) and `power1_cap` ceilings of 1110 W on pve-rig6600 against 800 W on b and c (:441-462).
    - It names a prior "2026-08-19 `.121`/`.122`" signature (:122). That date has no hit in `docs/`; only MEM records it.
16. **Survivor volume as a cause.** The forensic doc lists H3 OPEN and co-leading; the capacity doc says "not a crash hypothesis" (`S172_PHASE3_SURVIVOR_CAPACITY_CHARACTERIZATION.md:10`) (Part D contradiction 4).

---

# Parts A–E — per-era entries (extracted read-only; anchors as given by each extractor, spot-checked by the register author)

Each Part was extracted by a read-only pass over the files it lists in its own "Files read" section. Abbreviated file keys (e.g. `SUB`, `CL162`, `IR`, `ROCm`) are defined in each Part's own *File key* section. Entry numbers restart in each Part (cite as `Part B E23`). A Part's own "Prior record" sub-section, where present, is that extractor's search log and is subordinate to the register-level Prior record above.

---

## Part A — 2026-01-01 → 2026-03-31 (pre-GCVM era: freezes, SMU, OOM, init storms)


Worktree: /home/michael/tfm_worktrees/gpu-fault-history (HEAD 3579088). All anchors are relative to that tree.
"docs/X:n" = file:line. "commit <sha>" = commit message only (diffs not read, per brief).
Files deleted from the tree but read from git history are cited as `git show <sha>:<path>:<line>`.

### NULL RESULTS FIRST

- **GCVM_L2_PROTECTION_FAULT / SQC page fault / gfxhub page fault: 0 mentions in any Jan–Mar 2026 document.** I searched
  all 168 docs whose names match `S1[0-5][0-9]|S[7-9][0-9]|2026(01|02|03)`, plus the listed chapters and manuals. The only
  period-named hits are `docs/SESSION_CHANGELOG_20260402_S159G.md` and `docs/TB_SUBMISSION_S159G_RIG6600_CRASHES.md`,
  both dated 2026-04-02 (S159G header `:2`; TB submission `:3`), which is **outside this window**. Every place that
  ties GCVM_L2 to the PWC era is a later retrospective, not a contemporaneous record: CHAPTER_9:1467,
  CHAPTER_2:894-896, :1039-1040, CHAPTER_1:2019, CHAPTER_9:641-648 (all written 2026-11-11 and citing
  TFM_PROJECT_FACTS_SKILL §0.7).
- **No Jan–Mar record of** AER, iommu, amdgpu.lockup_timeout, amdgpu.gpu_recovery, AMD_SERIALIZE_KERNEL,
  HIP_LAUNCH_BLOCKING, undervolt, power cap, or sclk/DPM pinning. The `perf=high` udev rule (S80) is the only
  DPM-adjacent setting found. I also checked the deleted Jan 3/18/27/29/30 changelogs and the deleted S70/S71/S88 changelogs.
- **"Ring timeout", "device lost from bus": 0 hits** in this period.
- **The GPU throughput probes (Phase A/B) were never filled in.** All result cells in
  `docs/SESSION_CHANGELOG_20260307_S128.md:83-111` read `TBD`, and the S131 `apply_caps.py` "was not run" (`docs/SESSION_CHANGELOG_20260308_S131.md:107-111`).
  The Step-1 benchmark suite result tables are blank (`docs/STEP1_GPU_BENCHMARK_SUITE.md:495-500, 640-648`).
- **The rrig6600c 4-worker-cliff test plan (S157 Phases 1–5) has no recorded results.** It is at `docs/S157_CHAT_PROMPT.md:45-63`.
  The proposal it cites, `PROPOSAL_S157_rrig6600c_4worker_cliff.md` (`:35`), is not in the repo (`git ls-files` search).
- **Why rrig6600/rrig6600b ran 8 workers while rrig6600c did not was "genuinely unknown" at the end of S155**
  (`docs/SESSION_CHANGELOG_20260324_S155.md:92-94`).

### Hardware-count note (as recorded)
- In January 2026 the rigs are recorded as **12-GPU** nodes: rig-6600 RAM 7.7 GB / "Workers: 12 GPUs" (`docs/SESSION_CHANGELOG_20260124_OOM_FIX.md:127-131`), GPU indices up to 11 (`docs/SESSION_CHANGELOG_20260122.md:22`; `docs/CHAPTER_9_ADDENDUM_v2_2_0.md:68`).
- On 2026-01-31 the counts were set to 8 per rig and rig-6600c (.162, 8 GPUs) was added (commit 4d979e0; commit 94342a4 "GPU 12→8 correction"). **The reason for the change (cards physically removed, or a doc correction) is not recorded.**
- In this era the cards are called "RX 6600" throughout. The XT SKU is a later correction (CHAPTER_9:641).

---

### ENTRIES (date order)

**E1. 2026-01-03 (pre-S70, Jan 3 changelog, deleted file).** Rigs: rig-6600, rig-6600b (i3 CPUs).
- Belief: a 258 MB JSON survivor load (4.2 s per job) was thrashing the rig CPUs at 10+ concurrent jobs. A second belief: `coordinator.py` for script jobs caused an "SSH thundering herd".
- Action: NPZ binary format; script jobs routed through `scripts_coordinator.py`; SSH stagger extended from localhost-only to all nodes (`coordinator.py` ~line 1488).
- Result: NPZ "88x faster loading". No stability measurement recorded.
- Kept: yes. "Script-based jobs MUST use scripts_coordinator.py" became a rule.
- Anchors: `git show 53a3829:docs/SESSION_CHANGELOG_20260103.md:11-15, 24-30, 47-54`. The file was deleted in commit 191d216 (2026-01-29).

**E2. 2026-01-16 (debug session).** Rigs: both ROCm rigs (rig-6600, rig-6600b), 12 GPUs each.
- Symptom: 50 Step-2 trials "crashed both ROCm rigs (froze, required reboot twice)". 5 and 12 trials worked; 26 untested.
- Belief: HIP init collision. A stagger fix was tried and then reverted. Also suspected: "SSH connection overload or simultaneous HIP init"; "MaxSessions in sshd_config may need increase to 50".
- Action: added `_best_effort_gpu_cleanup` to scorer_trial_worker.py; restored scripts_coordinator.py from backup (stagger reverted).
- Result: "Root Cause Still Unknown". Proposed isolation: run 26 trials (1 per GPU).
- Kept/reverted: stagger reverted; cleanup kept.
- Anchors: commit a47ba88.

**E3. 2026-01-17 (ROCm HIP Init Fix v2.1, Joint Alpha+Beta, "APPROVED FOR IMPLEMENTATION").** Rigs: rig-6600 (9 workers), rig-6600b (8 workers).
- Evidence: a 26-trial test. localhost 2/2 workers OK; rig-6600b 7/8 passed ("1 failed due to manual reboot"); **rig-6600: all 9 jobs timed out at 3600 s, "FROZEN"** (`docs/PROPOSAL_FINAL_ROCm_HIP_Init_Fix_v2_1.md:27-33`).
- Belief: "Parallel HIP initialization storm". The existing stagger was bypassed by a list comprehension in `_node_executor` (`:21-23, :59-69`).
- Action: replace the list comprehension with a loop that sleeps `node.stagger_delay` (STAGGER_REMOTE=0.5 s, STAGGER_LOCALHOST=3.0 s) in scripts_coordinator.py lines 524-531 (`:73-107, :133-140`). The proposal also says to raise it to 1.0 s if 0.5 s proves insufficient (`:140`).
- Result: none recorded in the proposal. The S70-era v2 batching proposal (E7) says the stagger "Implemented Jan 17 … working correctly" (`docs/PROPOSAL_Job_Batching_Pipeline_Stability_v2.md:25, 31-41`).
- Kept: yes (`docs/PROPOSAL_Job_Batching_Pipeline_Stability_v2.md:23-29`).
- Anchors: as above. Committed with commit f614736 (docs file added).

**E4. 2026-01-17 (same commit as E3).** Rigs: rig-6600, rig-6600b.
- Belief: a "GFXOFF race condition" (SMU race during GPU state transitions), "not concurrency or weak CPUs". `amdgpu.gfxoff=0` was "ignored on 6.8.0-90".
- Action: kernel parameter `amdgpu.ppfeaturemask=0xffff7fff` (disables PP_GFXOFF_MASK). GPU cleanup registered with atexit in scorer_trial_worker / anti_overfit_trial_worker / full_scoring_worker (gc.collect → CuPy pools → PyTorch sync/cache/ipc).
- Result: "Both rigs … show gfxoff=00"; rocm-smi fields populated. "Testing: 100 trials @ concurrency=4 in progress". References `PROPOSAL_ROCm_Monitoring_Desync_Mitigation_v1_0.md` (not in repo, not in git history; searched). "Team Beta approved 2026-01-17".
- Kept: yes. Listed as a Required Setting in CHAPTER_9:904, :922-925.
- Anchors: commit f614736.

**E5. 2026-01-18 (ROCm Stability Envelope).** Rigs: rig-6600 (.120), rig-6600b (.154), 12 concurrent.
- Belief: the "true constraint is host memory pressure during data loading", **not** HIP init collisions or GPU concurrency. A table records the retired assumptions: "HIP init collision is dominant failure → Memory pressure during load is the cause"; "Reduce GPU count → Reduce sample_size".
- Action: `max_concurrent_script_jobs=12` on both rigs; `--sample-size 450`; `amdgpu.ppfeaturemask=0xffff7fff`.
- Result: 100 trials, 100% success. Sample size 350–1000 all stable; **2000 = "Freeze risk"**. 15.41 trials/min at 450 (4.5x the old 5000@4 concurrent).
- Troubleshooting rule recorded: on N/A SCLK/MCLK or "unknown" Perf, reduce sample_size, verify GFXOFF off, verify cleanup, reboot on persistent N/A. "Do NOT reduce max_concurrent_script_jobs as first response". The root cause given there is "allocator thrashing".
- Kept: yes at the time. Later corrections: the value 12 → 8 (CHAPTER_9:935-937); S107 retired the 450 operating point (CHAPTER_3:1346).
- Anchors: `git show a490ea0:docs/SESSION_CHANGELOG_20260118.md:5-15, 17-25, 34-54`; commit a490ea0; `docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md:881-894, 896-918, 962-977`.

**E6. 2026-01-20 (SMU exhaustion).** Rigs: both rigs ("12-GPU rigs").
- Belief: Step-2 crash root cause = "SMU (System Management Unit) exhaustion from excessive polling, NOT SDMA timeout or memory pressure". The polling came from `watch -n 0.5 rocm-smi` ×2 (~240-360 reads/min) and a fan service loop at 3 s (~240 writes/min), ~500+ SMU ops/min combined.
- Kernel-log signature: `SMU: response:0xFFFFFFFF for message:TransferTableSmu2Dram`; `Failed to export SMU metrics table`; `failed to write reg 2890 wait reg 28a2`.
- Action: killed the watch processes; set `LOOP_SECS` 3→20 in `/usr/local/bin/rocm-fan-curve.sh` on both rigs (~36 writes/min). Rule recorded: fan loop 15–30 s minimum, rocm-smi 10+ s apart.
- Result: "No SMU errors in dmesg post-fix"; fan control healthy (Level 71, 1400-1700 RPM).
- Kept: superseded. On 2026-01-22 the fan service on rig-6600b was masked (E8), and on ~2026-02-01 fan services were disabled on all rigs (E13).
- Anchors: `docs/SESSION_CHANGELOG_20260120.md:3-43`; commit a82860f.

**E7. 2026-01-20 (Job Batching proposal v1 and v2).** Rigs: both rigs.
- Symptom: a WATCHER pipeline of 120 trials "crashes both rigs within 60 seconds"; rigs unresponsive ("HDD light solid, SSH hung"); "Requires HiveOS reboot". The benchmark with 20 jobs per batch passed 100%.
- Belief: "Simultaneous job initialization overload" (40 jobs per rig → 40 SSH sessions → data loading → NVMe I/O saturation). Listed as eliminated: sample size; GPU concurrency ("12 concurrent validated"); SMU polling ("Fixed with LOOP_SECS=20"); HIP init collision ("Stagger already in place"); memory pressure.
- Action proposed: `MAX_JOBS_PER_BATCH=20`, `INTER_BATCH_COOLDOWN=5`, allocator reset via `sync && echo 3 | sudo tee /proc/sys/vm/drop_caches` between batches (v2). Also floated: Option B, a WATCHER trial cap of 20; Option C, `INTER_NODE_STAGGER=5.0`. Rejected in v2: reducing max_concurrent_script_jobs, increasing stagger, limiting WATCHER trials.
- Result (implemented form): commit 49e8ad4 (2026-01-20 21:28) added `MAX_JOBS_PER_NODE_PER_BATCH=6` "to prevent ROCm HIP init storms" and recorded "25/25 trials now complete at 100% success rate". It also added the ramdisk preload.
- Kept: yes. The per-node cap of 6 and the 5 s cooldown are later described as current (CHAPTER_9:388-390).
- Anchors: `docs/PROPOSAL_Job_Batching_Pipeline_Stability.md:12, 20-32, 54-70, 76-143`; `docs/PROPOSAL_Job_Batching_Pipeline_Stability_v2.md:13-41, 53-81, 113-181, 297-309`; commit 49e8ad4.

**E8. 2026-01-22 (Team Beta Step-3 stability ruling).** Rigs: rig-6600b (GPU[4]), rig-6600 (GPU 11).
- Recorded failure modes: **rig-6600b "Kernel deadlock/crash", "SMU-faulty GPU[4] amplifies overload"**; **rig-6600 "GPU 11 'unknown' state", "Driver reset failure under I/O + HIP storms"**. TB called these "two failure modes of the same overload event". The stated cause was "Step 3 exceeds the validated operational envelope of RX 6600 nodes" (6–7× heavier per job than Step 2.5).
- Action: Step-3 `MAX_JOBS_PER_NODE_PER_BATCH` 6→**2**, `INTER_BATCH_COOLDOWN` 5→**10 s** (Step 2.5 unchanged at 6/5 s); `detect_job_step()` and `get_step_aware_limits()`; tarball chunk distribution (`step3_tarball_helpers.sh`) against "SSD + kernel I/O storms"; `rocm-fan-curve.service` disabled and masked on rig-6600b.
- Result: only expected outcomes are recorded ("No further GPU 'unknown' states…"). No measured result.
- Not taken: persistent GPU workers (deferred), hardware replacement ("Unless GPU[4] required for capacity"). Pending: "GPU[4] SMU investigation — Power/riser/ppfeaturemask?".
- Kept: the step-aware limits persist, but CHAPTER_9:389 (2026-11-11) records the Step-3 cap as **8**, not 2. When it changed is not recorded in this era.
- Anchors: `docs/SESSION_CHANGELOG_20260122.md:9-24, 30-52, 54-75, 110-126, 130-137`.

**E9. 2026-01-23 (GPU failure investigation, Part 2).** Rigs: rig-6600, rig-6600b.
- Symptom: 1–3 Step-3 job failures per run, first parallel wave only. They succeeded on retry (on Zeus). No HIP error in stderr.
- Hypotheses tested, in order: (1) HIP kernel cache corruption → fixed, failures continued; (2) parallel HIP init race → a **warmup barrier "didn't help"**; (3) missing `HSA_OVERRIDE_GFX_VERSION` → fixed, failures continued; (4) **OOM killer** → confirmed via `journalctl -k` (`Out of memory: Killed process 13994 (python) total-vm:13611472kB, anon-rss:1491948kB`). Arithmetic given: 7.7 GB RAM, 7 workers × ~1.5 GB.
- Action: deployed HIP cache clear (PRE-FLIGHT, ~2 s; "kept"), GPU warmup barrier (~15 s; "Remove"), HSA_OVERRIDE in warmup ("Remove if warmup removed"), debug stderr capture ("Keep"). Proposed: Option A, add RAM (16 GB); Option B, cap workers at 5 per node.
- Anchors: `docs/SESSION_CHANGELOG_20260123_PART2.md:4-31, 33-52, 58-59`; commit 3e6b224.

**E10. 2026-01-24 (OOM resolution).** Rigs: rig-6600 (.120), rig-6600b (.154), 12 GPUs.
- Belief: "~5 GB available after system/ROCm overhead"; 7 workers × ~700 MB → OOM. Memory benchmark: data loading only ~1.5 MB per 1K survivors. The "actual OOM cause is worker execution (PyTorch, ROCm init, scorer buffers)".
- Action: `chunk_size_config.py`, `benchmark_worker_memory.py`; fixed `max_concurrent_script_jobs` being ignored (Bugs 1–3); config `max_concurrent_script_jobs` localhost 2, .120 12, .154 12; recommended chunk_size 1000.
- Result: test matrix **12 GPUs 99/99 pass**; 4 GPUs 61/99 (orphaned-job bug) → 99/99 after the fix; final 12-GPU 99/99, 313 s (chunk_size=1000).
- Follow-up: Jan 25 manifest `full_scoring.json` chunk_size 5000→1000, "0 failures (was 2 OOM failures before fix)".
- Kept: yes.
- Anchors: `docs/SESSION_CHANGELOG_20260124_OOM_FIX.md:6-16, 22-40, 93 (key finding), 167-209 (bugs), 233-241 (test matrix), 265-286 (config)`; `docs/SESSION_CHANGELOG_20260125.md:14, 29`.

**E11. 2026-01-24 (Infrastructure Improvements proposal v1.0, TB review).** Rigs: rig-6600 GPU[10] (PCI 0000:22:00.0), rig-6600b GPU[4] (0F:00.0).
- Record: "rig-6600 crash mid-run with no preflight detection" and "GPU[10] SMU hardware degradation". "GPU[10] on rig-6600 experienced SMU failures (0xFFFFFFFF) that cascaded into system crash". GPU[10] shows the "same SMU 0xFFFFFFFF pattern as rig-6600b GPU[4]. This is a hardware defect".
- Action proposed: Item C, post-job `rocm-smi --resetclocks`, clear `~/.cache/hip_*`, N/A check, dmesg 0xFFFFFFFF grep. Item D, D1 monitor-only (recommended) or D2 `excluded_gpus: [10]`.
- Result: Jan 24 changelog says Items A and C were deployed (`preflight_check.py` v1.0.1, `gpu_cleanup.py` v1.0.1). Integrating gpu_cleanup into scripts_coordinator was left optional. **Item D's disposition is not recorded.**
- Anchors: `docs/PROPOSAL_Infrastructure_Improvements_v1_0.md:8, 114-149, 154-185`; `docs/SESSION_CHANGELOG_20260124.md:4, 22-23, 88, 96`.
- Related: CHAPTER_9_ADDENDUM v2.1.0 (2026-01-24) sample "rig-6600b … Device 10 Temp: N/A … Perf: unknown ← DEAD GPU … Reboot rig-6600b to recover GPU[10]", and the action table "N/A → Reboot node; unknown → SMU communication failure → Reboot node or reseat GPU" (`docs/CHAPTER_9_ADDENDUM_v2_2_0.md:43-45, 66-70, 150`).

**E12. 2026-01-26 (Inter-chunk GPU cleanup, Team Beta).** Rigs: AMD rigs (cluster 26 GPUs). The node index in the error is node-11.
- Symptom: Step-1 forward sieves (~26 chunks of ~19K seeds per invocation at 500K seeds) gave intermittent `Error: HW Exception by GPU node-11... reason: GPU Hang`.
- Belief: VRAM fragmentation accumulated because cleanup ran only once at exit.
- Action: `_best_effort_gpu_cleanup()` (with `gc.collect()`) between chunks in both forward-sieve loops of `sieve_filter.py` (`if chunk_start + chunk_size < seed_end:`).
- Result: "20/20 benchmark trials with 0 GPU hangs; all 26 GPUs healthy post-run; <5% overhead".
- Kept: yes. CHAPTER_2 scopes it to the legacy engine.
- Anchors: `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:1004-1034, 1036-1042`; commit 20b0e7c (2026-01-26, adds `benchmark_step1_stability_v2.sh`).

**E13. ~2026-02-01 (Session 56 end note).** Rig: rig-6600b GPU[4] (slot 5, PCI 0F:00.0); all rigs.
- Action: GPU[4] **reseated**; "**PCIe set to Gen 1 for stability**"; "Fan services disabled on all rigs — using built-in auto fan control".
- Result: "Sensors now reading healthy".
- Kept: the record does not say. Gen1 is cited again as context in S80 (E15).
- Anchors: `docs/SESSION_CHANGELOG_20260201.md:152`.

**E14. 2026-02-01 (Step 1 GPU Benchmark Suite v1.0.0).** All rigs.
- Belief: Step 1 lacks the Step-2/3 protections: no stagger (Step-2/3 value given as 2.0 s), unbounded `--max-concurrent 26`, no inter-trial cooldown, no 80% VRAM limit in the sieve path. "If 20 trials fails (matching your observed error), that confirms the need for inter-trial cleanup". Note: "rig-6600b GPU[4] … always shows anomalies due to the broken SMU".
- Action proposed: 4 benchmarks (seed scaling, concurrency, consecutive-trial stress, memory profile); then `--inter-trial-delay`, `--max-seed-count-per-trial`, inter-trial cleanup and health check.
- Result: **tables blank; not recorded as run.**
- Anchors: `docs/STEP1_GPU_BENCHMARK_SUITE.md:4-7, 15-24, 40-49, 377, 495-502, 640-681`.

**E15. 2026-02-11 (S80).** Rigs: rig-6600 (.120), rig-6600b (.154), rig-6600c (.162). All rebooted.
- Belief: the **rig-6600b crash was a kernel soft lockup** (`CPU#2 stuck for 26s` in `systemd-udevd`) during GPU enumeration on PCIe Gen1 with 8 GPUs. GFXOFF disable "prevents soft lockup crashes from GPU sleep/wake storms".
- Action: udev rule `/etc/udev/rules.d/99-amdgpu-perf.rules` (auto `perf=high` when GPUs detected); `amdgpu.gfxoff=0` in GRUB on all 3 rigs; reboot.
- Result: "GFXOFF confirmed active"; "all 26 GPUs available".
- Kept: yes ("permanent GPU stability fixes"; Operating Guide v2.0 "GPU Stability: udev rule (perf=high auto-set), GFXOFF disabled via kernel parameter").
- Note: E4 recorded `gfxoff=0` as ignored on kernel 6.8.0-90 and used ppfeaturemask instead. S80 uses `amdgpu.gfxoff=0`. Whether both parameters were present is not recorded.
- Anchors: `docs/SESSION_CHANGELOG_20260211_S80.md:10, 50-58, 185`; `docs/SESSION_81_HANDOFF.md:7, 14`; `docs/COMPLETE_OPERATING_GUIDE_v2_0.md:77`; commit d2306f5.

**E16. 2026-02-04 (S62) / 2026-02-18 (S96A): incidental.**
- "rig-6600c rebooted during Soak A (no impact — separate machine)". Cause not recorded (`docs/SESSION_CHANGELOG_20260204_S62.md:80`).
- "SSH to 192.168.3.154 (rig-6600b) failed intermittently from Zeus despite rig being operational. Network-layer issue, not rig issue" (`docs/SESSION_CHANGELOG_20260218_S96A.md:199`).

**E17. 2026-02-22 (S107).** All three rigs.
- Finding: `ml_coordinator_config.json` had .120/.154 at `gpu_count=12 / max_concurrent_script_jobs=12` "despite all remote rigs having 8 GPUs" → "potentially over-dispatching". .162 was missing.
- Action: set to 8/8/8 and added .162.
- Kept: yes.
- Anchors: `docs/S107_session_log.md:2, 30-41, 69`.

**E18. 2026-02-23 (S109).** Rig: rig-6600b (.154).
- Record: "D-state zombie processes on rig-6600b … from prior HIP initialization storm"; "ROCm driver deadlock during concurrent initialization. Requires reboot to clear. Stagger timing in scripts_coordinator.py mitigates but doesn't prevent edge cases".
- Action: reboot rig-6600b.
- Result: clean re-run 20/20 trials, all 4 nodes.
- Follow-on proposal: a "Node watchdog / auto-restart layer" to detect D-state zombies and SSH-reboot. Carried in TODO S120–S127. **No record of implementation in this era.**
- Anchors: `docs/SESSION_CHANGELOG_20260223_S109.md:40-56, 128-132, 198`; `docs/TODO_MASTER_S120.md:70`; `docs/TODO_MASTER_S127.md:89`.

**E19. 2026-02-26 (S112).** Rig: rig 120 (rig-6600), GPU 2.
- Record: first real-data Step 1 (10M seeds × 50 trials, 26 GPUs): "GPU hangs | 1 (rig 120, GPU 2 — recovered via retry)".
- Action: none beyond the retry. Root cause not recorded.
- Anchors: `docs/SESSION_CHANGELOG_20260226_S112.md:100-116`.

**E20. 2026-03-07 (S125b). Zeus (CUDA), not a rig.** Two RTX 3080 Ti in `Exclusive_Process` → `cudaErrorDevicesUnavailable` with n_parallel=2.
- Action: `nvidia-smi -i 0/1 -c DEFAULT`, persisted in `/etc/rc.local`.
- Result: "all 26 GPUs operational".
- Kept: yes, as an invariant.
- Anchors: `docs/SESSION_CHANGELOG_20260307_S125b.md:34-86, 110`; commit 19cd3b3.

**E21. 2026-03-07 (S125b TODO / S126 GPU Throughput Investigation Plan).** AMD rigs.
- Belief: `seed_cap_amd=19,000` "set conservatively during ROCm stability work S96B". **I could find no S96B record that sets it**: the S96B proposal is Zeus-only (`docs/PROPOSAL_S96B_PERSISTENT_GPU_WORKERS_v1_0.md:260`); S129B calls 19k a "holdover from early capacity probing" (`docs/SESSION_CHANGELOG_20260308_S129B.md:26`). The plan also cites ROCm documentation: "not recommended to exceed 2 simultaneous compute workloads" while 8 workers per rig were running. "Current stability is maintained by PYTORCH_HIP_ALLOC_CONF + 80% VRAM limiting from S96B".
- Action proposed: Phase A single-card step-ladder (RX 6600 100k→2M, `ROCR_VISIBLE_DEVICES=0`) and Phase B full-rig concurrent ladder (stop on OOM / "ROCm HangDetected" / HIP error); seed caps = Phase B ceiling × 0.85. Risk row: "ROCm HangDetected on concurrent overload … requires rig reboot".
- Result: **not recorded (S128 tables TBD).**
- Anchors: `docs/TODO_MASTER_S125b.md:19-20`; `docs/GPU_THROUGHPUT_INVESTIGATION_PLAN_v1_0.md:3, 19-21, 38-40, 64-120, 198-199, 209-214`; `docs/TODO_MASTER_S126.md:33-46, 188`; `docs/SESSION_CHANGELOG_20260307_S128.md:82-111`.

**E22. 2026-03-08 (S129B-A / S131): seed caps raised.** AMD rigs and Zeus.
- Action: `--seed-cap-amd` 19,000→**2,000,000**, `--seed-cap-nvidia` 40,000→**5,000,000** (CLI, S129B-A); constructor defaults and manifest v1.5.0 (S131, commit d168f83). RX 6600 sps 5,000→787,950 (the value labelled "actual").
- Result: no stability result recorded for the cap change itself.
- Kept: 2M is the production value through S158 ("seed_cap_amd Restored to 2,000,000 (production value)") and S159.
- Anchors: `docs/SESSION_CHANGELOG_20260308_S129B.md:12, 20-42`; `docs/SESSION_CHANGELOG_20260308_S131.md:64-99`; commit d168f83; `docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md:1210-1232`; `docs/SESSION_CHANGELOG_20260329_S158.md:83-84`; `docs/S159_CHAT_PROMPT.md:14, 67`.

**E23. 2026-03-08 (S130 persistent workers + spawn stagger).** All 3 rigs, 24 AMD GPUs.
- Action: persistent workers in `coordinator.py` (default off), `worker_pool_size=8`. Commit 7288e5e added a spawn stagger of `gpu_id * 2.0s` "prevents parallel HIP init collision on RX 6600 rigs causing libamdhip64 segfaults. GPU 7 max wait = 14s per rig".
- Result: S130 17/17 checks, 10 jobs × 20M seeds; "48 workers confirmed each run"; 2,082,140 sps (+150%). Gate 1 kill test "did not land".
- Kept: superseded by the S134 rewrite (E26).
- Anchors: `docs/SESSION_CHANGELOG_20260308_S130.md:12-39, 55-72, 85-97`; commit 7288e5e.

**E24. 2026-03-09 (S132): rig-6600c libamdhip64 segfaults.** Rig: rig-6600c (.162). Other rigs healthy.
- Symptom: `libamdhip64` segfaults "under persistent worker HIP concurrency", ~30–60 s into the sieve pass.
- Hypothesis: "i5-8400T CPU + 8-way HIP concurrency = memory pressure".
- Anchors: `docs/SESSION_CHANGELOG_20260309_S132.md:44-50, 55`.

**E25. 2026-03-09/10 (S133-A / S133-B).** Rig: rig-6600c; also 120 and 154 in pass 3.
- S133-A belief: the S130 integration re-introduced a semaphore on the Zeus local path that was never released → deadlock → "worker death timeouts on rig-6600c". Fix: the persistent path bypasses the semaphore. **Record conflict:** the commit message says "restore semaphore throttle to persistent worker dispatch". Result: workers survived pass 1.
- S133-B belief: on hybrid passes `sieve_gpu_worker.run_sieve()` lacked a `strategies` parameter → `TypeError` → pipe closed → "all 12 workers dead". Fix: strategies parameter, deployed to all 3 rigs. Result: "workers survive all 4 sieve passes"; hybrid not clean-verified in production.
- The record says "all 4 rig-6600c workers alive at pool startup", so 4 workers per rig at that time.
- Anchors: `docs/SESSION_CHANGELOG_20260309_S133.md:11, 15-26, 30-41, 62-67`; commit 288de2a.

**E26. 2026-03-10 (S134 → S135): standalone PWC.** All rigs.
- Action: coordinator.py reverted to pre-S130 (d5219a0); new `persistent_worker_coordinator.py` (commit b5b6532). ROCm guards: **4 s stagger between workers on the same rig**; `HSA_OVERRIDE_GFX_VERSION=10.3.0` in all SSH worker launches; `source ~/rocm_env/bin/activate`; "**Max 4 persistent workers per rig (not 8) — memory pressure validated at 4-way**". S135: per-WorkerHandle dispatch lock (Bug 9), SSH -q banner drain (Bug 8).
- Kept: yes. `ROCM_SPAWN_STAGGER_S = 4.0` persists (CHAPTER_9:1169, :1174).
- Anchors: `docs/SESSION_CHANGELOG_20260310_S134.md:2, 32-41, 49-52, 65-70`; `docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md:1166-1172, 1182-1185`; commits d5219a0, 0996582, b5b6532.

**E27. 2026-03-15 (S145): PWC by default; 162 hang.** Rig: "162 rig" (rig-6600c).
- Record: Phase 2 (100K seeds, 2 trials) was "Terminated — 162 rig hung on 3,846-seed chunks (SSH overhead > job time). Not a code issue."
- Action: commit 4d6801a made persistent workers the manifest default.
- Anchors: `docs/SESSION_CHANGELOG_20260315_S145.md:144-146`; commit 4d6801a.

**E28. 2026-03-15/16 (S146): worker_pool_size=4, JOB_TIMEOUT_S=600, kernel-tail fix.** All rigs.
- Action: commit 4b1c975 set `worker_pool_size=4` ("validated"), `JOB_TIMEOUT_S=600`. Bug 4: forward/reverse hybrid kernel arg tails split (sending `a, c` to the reverse hybrid → "crash on reverse hybrid"). Bug 8: `cp.int32()` casts for scalar kernel args ("ROCm/CuPy can be pickier"). Bug 9: survivor count clamp ("bad kernel write could corrupt count"). Bug 1 (Zeus): `_localhost_semaphore = Semaphore(2)` against `cudaErrorDevicesUnavailable`.
- Result: 3 trials, 10M seeds, all 4 passes, "Zero ❌ errors. Zero crashes. Zero tracebacks".
- Invariant: "`worker_pool_size=4` (not 8) — validated stable envelope".
- Later claim (S149 TB request): the 4-cap was "established in S146 because any value above 4 caused rig crashes".
- Anchors: `docs/SESSION_CHANGELOG_20260316_S146.md:16-20, 34-39, 60-70, 92-103, 111, 139, 177`; `docs/TB_RULING_REQUEST_ROCR_ISOLATION_S149.md:19-21`; commits 4b1c975, 723721a; `docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md:1304-1324`.

**E29. 2026-03-19 (S148 → S149-A ROCR ruling request).** All rigs, GPUs 4–7.
- Observation: AMD rigs at ~1,050 seeds/s per rig; `rocm-smi --showuse` 0% on all 8 GPUs with 4 workers alive; worker CPU 1.3%.
- Belief: `_spawn_worker()` set `CUDA_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` but **not** `ROCR_VISIBLE_DEVICES`, while `sieve_gpu_worker` hardcodes `Device(0)`. "HIP_VISIBLE_DEVICES … works reliably for GPUs 0–3 … For GPUs 4–7 … inconsistent … HSA context conflict and crashing the worker", which would explain the 4-worker ceiling. Source harness 15/15 checks.
- Action proposed: add `ROCR_VISIBLE_DEVICES={gpu_id}`. Questions for TB: interaction with `HSA_ENABLE_SDMA=0` / `HSA_OVERRIDE_GFX_VERSION=10.3.0`, raising the pool to 8, reducing the 4 s stagger.
- TB ruling as relayed: "Approve one-line fix. Keep stagger 4.0s. Validate via single-rig …". A later step was "Reduce spawn stagger 4.0s → 1.0s (only after 8-worker stability proven)".
- Deployment of S149-A: **ambiguous in the record.** S149 deployed S149-B instead (E30). PROJECT_FILE_CATALOG (written later) says `apply_s149a_rocr_isolation.py` "is the evidence that the S149 ROCR ruling was implemented". That script was committed only on 2026-04-03 (commit ba833a4).
- Anchors: `docs/TB_RULING_REQUEST_ROCR_ISOLATION_S149.md:2-4, 11-21, 33-75, 77-98, 118-145`; `docs/S149_CHAT_PROMPT.md:25-41, 63-79, 133-135, 156-164`; `docs/PROJECT_FILE_CATALOG.md:121, 826`.

**E30. 2026-03-20/21 (S149-B): direct Device(gpu_id), no masking.** All 3 rigs, GPUs 0–7.
- Belief: "HIP masking is unreliable for GPUs 4-7 — workers crashed on startup, causing a 4-worker ceiling".
- Action: remove `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` from `_spawn_worker()`; workers see all GPUs; `cp.cuda.Device(gpu_id)`; `run_sieve_job(job, gpu_id)` with a mismatch assertion. Invariant: "no HIP/CUDA/ROCR visibility masking in spawner".
- Result: all 3 rigs at 8 workers. ~31,000 → ~1,990,000 s/s burst (46x). "GPU 0 and GPU 7 spawn test passed on rrig6600. Preprod soak 5/5 trials clean, 24/24 workers alive". Worker pool 8 per rig × 3.
- But: "Run 1 crashed during Trial 7's hybrid reverse pass — 22,145 bidirectional survivors were lost" (rig not named). That led to the per-trial NPZ checkpoint.
- Kept: until S155 (E36), which "restored" ROCR isolation.
- Anchors: `docs/SESSION_CHANGELOG_20260321_S149.md:2, 21-42, 63-66, 129-133, 163-164`; commit c1c698d.

**E31. 2026-03-21 (S150): slim_v1 IPC.** All rigs.
- Action: slim_v1 parallel-array results (commit 65461c9).
- Result: pass 3/4 rig throughput ~3,774 → ~84,000 s/s.
- S151 timeline: "S150 ~10:30 slim_v1 deployed — crashes begin" on rrig6600c. Before that, "S149/S150 overnight rrig6600c ran 8 workers for hours — STABLE".
- Anchors: `docs/SESSION_CHANGELOG_20260321_S150.md:17-27, 63-69`; `docs/ROOT_CAUSE_ANALYSIS_RRIG6600C_S151.md:21-24`.

**E32. 2026-03-21 (S151): rrig6600c crash day — sequence of config changes.** Rig: rrig6600c (.162), 8 × RX 6600, i5-8400T. The "PCIe 1x" link is named in commits.
1. 14:24, commit 38393b5: **limit rrig6600c to 6 workers — "weaker CPU ROCm queue limit"**.
2. 15:10, commit cc113b5: **revert to 8 — "GPU7 fault was hardware not software"**. That is the only record of a GPU7 fault; no detail.
3. 16:54, commit f3fdbf1: **`CUPY_CUDA_MEMORY_POOL_TYPE=none`** added to the worker env ("prevents CuPy pool race on RX 6600").
4. 17:12, commit e4905c2: **gpu_count=4 — "PCIe 1x stability limit"**.
5. 18:19, commit 12dbeaf: **per-node respawn lock + 4 s stagger** — "prevents SSH hammer on PCIe 1x rigs". `_ensure_worker_alive()` serializes respawns.
6. 20:00 crash monitor `rig_crash_monitor.sh` (per-second worker count, load, RAM, VRAM, SSH conns) on all rigs. 20:36: stable at 4 workers.
7. 21:00–21:10 sshd comparison: rrig6600c had **MaxSessions 5** (others 50), ClientAliveInterval 300 (others 0), ClientAliveCountMax 2 (others 3), MaxStartups missing (others 50:30:100). Changed to 50 / 0 / 3 / 50:30:100, `systemctl reload sshd`, backup `/etc/ssh/sshd_config.bak_s151`.
8. 21:03, commit fd38d17: root cause declared MaxSessions 5; gpu_count restored to 8.
- Recorded cascade: workers 5-7 rejected → respawn storm → ClientAlive drop of 0-4 → all 8 dead → "rig overloaded → system crash → physical reboot required".
- Theories marked wrong: hardware fault, kernel upgrade 6.8.0-106 ("All 3 rigs same kernel"), slim_v1, CuPy pool race ("Real issue but not the crash cause"), weak CPU, PCIe 1x overload, ROCm queue eviction.
- Also: slim_v1 was reverted on all rigs during debugging (pre-slim worker restored).
- End of session: "Workers at end 20/20 alive (rrig6600c at 4 workers)" — which contradicts the restored gpu_count=8. The process died before the NPZ checkpoint.
- Status: "RESOLVED" (RCA). **Later sessions show crashes continued** (E33–E39).
- Anchors: `docs/ROOT_CAUSE_ANALYSIS_RRIG6600C_S151.md:4, 9-15, 19-29, 35-46, 50-71, 75-96, 100-110, 114-128, 151-159, 165-175`; `docs/SESSION_CHANGELOG_20260321_S151_FINAL.md:3, 10-16, 39-63, 67-76, 80-94, 107-120, 137-156`; commits 38393b5, cc113b5, f3fdbf1, e4905c2, 12dbeaf, fd38d17.

**E33. 2026-03-22 (S152).** Rig: rrig6600c mostly; Zeus.
- Belief A: slim_v1 + no per-worker masking → 8 workers race on the shared `~/.cupy/kernel_cache/` during the first `cp.RawKernel()` compile. rrig6600c's i5-8400T (1.70 GHz vs 2.80/2.90) makes it "deterministic crash". Fix: per-worker `CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_N` (commit 636abc0; TB "first-choice fix"; `CUPY_CACHE_IN_MEMORY=1` "not reliable on ROCm/hipcc"). Result: "eliminated rrig6600c crash but exposed this second crash on Zeus" (window_optimizer `-9` kill, suspected RSS spike from a 340k-survivor JSON at window_size=2; "Not confirmed").
- Recorded mismatch: S152 says S149-B "removed `ROCR_VISIBLE_DEVICES`" masking; S149-B's own record says it removed HIP/CUDA masking (E30).
- Belief B (TB confirmed, latent): hybrid kernels guard `pos < n_seeds` instead of `_max_surv` → out-of-bounds write to `skip_sequences` → "GPU memory corruption" at degenerate configs. Fix identified (`#define MAX_SURVIVORS`), **not implemented**.
- Other actions: commit ae6afb2 capped `skip_sequences_gpu` at `_max_surv*k` (slim_v1 VRAM fix); commit 12a8b6b SSH `ServerAliveCountMax=10`, `ConnectTimeout=10`; commit 701c501 `WORKER_HEARTBEAT_TIMEOUT_S` 30→**90** ("allows ROCm init on freshly rebooted rigs"). S153 records an "incomplete amdgpu library installation (fixed S152 via apt-get install)" on rrig6600c.
- Hardware: i5-9400 ordered for rrig6600c.
- Result: 3-trial pre-slim_v1 verification, **26/26 GPUs, rrig6600c stable throughout**. slim_v1 "PARKED". TODO S152 also records "rrig6600b workers died at ~22:09 after ~2 hours" (worker session drops).
- Anchors: `docs/SESSION_CHANGELOG_20260322_S152.md:10, 28-33, 37-56, 59-66, 70-77, 83-102`; `docs/SESSION_CHANGELOG_20260323_S153.md:27-29`; `docs/TODO_MASTER_S152.md:47-51`; commits ae6afb2, 12a8b6b, 701c501, 6424575.

**E34. 2026-03-23 (S153): diagnostic.** Rig: rrig6600c, GPUs 0–7.
- Hypothesis (S152): `java_lcg_reverse` was being sent extra `a, c` args.
- Test, Stage 8: 8A correct args passed on all 8 GPUs; 8B wrong args "DID NOT CRASH (ROCm tolerated it)". Stage 9: 8 real persistent workers, 1 reverse job each, all done in 3.1-4.1 s.
- Conclusion: not the crash cause. The crash "only occurs in production under sustained multi-hundred-chunk load" (~21 reverse jobs per worker over 5+ min) → resource-accumulation hypothesis. The crash monitor must log to `~/rig_crash_monitor_persistent.log` (`/tmp` is wiped on reboot).
- Anchors: `docs/SESSION_CHANGELOG_20260323_S153.md:2-4, 21-49, 51-66, 70-78, 93-94`.

**E35. 2026-03-23 (S154): OOM via kern.log.** Rig: rrig6600c.
- Evidence: `python invoked oom-killer … Out of memory: Killed process 3008 (python) total-vm:41452828kB, anon-rss:172748kB` at 12:56. A second event at 13:35 (after the fix below) showed the identical total-vm, pid 3379. A separate syncthing OOM at 10:09.
- Hypothesis 1: per-job GPU arrays accumulating. Fix: explicit `del` of 8 arrays + `gc.collect()` (commit 7182ab5, all nodes). Result: **did not resolve**; identical total-vm.
- Hypothesis 2: `CUPY_CUDA_MEMORY_POOL_TYPE=none` maps 8 GB of VRAM per worker → 8 × 8 GB = 64 GB VA on 7.7 GB RAM + 2 GB swap. Proposed fix: remove it and add `set_limit(256 MB)` (+64 MB pinned); disable syncthing; swap 2→8 GB.
- The S155 proposal states "All three AMD rigs … crash with OOM" with "3 separate OOM events on rrig6600c". The record does not show OOM evidence for rrig6600 or rrig6600b.
- Respawn storm noted: "537 dispatch threads simultaneously detect dead workers"; the S151 lock queue grows "hundreds deep".
- Run status: Run 1 Trial 1 OOM ~12:56; Trial 2 OOM ~13:35.
- Anchors: `docs/SESSION_CHANGELOG_20260323_S154.md:20-75`; `docs/SESSION_CHANGELOG_20260323_S153_S154.md:32-46, 50-99, 105-138, 145-160, 169-176, 180-186`; `docs/TODO_MASTER_S154.md:7-54, 94, 130-135`; `docs/PROPOSAL_S155_CUPY_POOL_FIX.md:3-23, 27-54, 58-91, 129-146`; commit 7182ab5.

**E36. 2026-03-23 → 03-25 (S155): every software and hardware fix for the OOM failed.** Rig: rrig6600c.
| Attempt | Action | Recorded result | Anchor |
|---|---|---|---|
| CuPy pool cap | removed `CUPY_CUDA_MEMORY_POOL_TYPE=none`; `cp.get_default_memory_pool().set_limit(256 MB)` after warmup; 3 TB rounds (commits c5af655, 5d8d10d) | "OOM persists — no change to crash signature" | S155:37-46 |
| Env cap | `CUPY_GPU_MEMORY_LIMIT=268435456`, `PRNG_CUPY_POOL_LIMIT_MB=256` in ROCM_ENV_VARS | OOM persists | S155:26, :43 |
| ROCR isolation | `ROCR_VISIBLE_DEVICES={gpu_id}` "restored" in `_spawn_worker()` (theory: otherwise each worker maps all 8 GPU apertures at HSA init); commit 5d8d10d also says "pool cap" | OOM persists | S155:27, :48-54 |
| M.2 SATA | axGear M.2 SATA adapter → Samsung 860 EVO; the e820 hole at `0x37700000-0x37700fff` went away | OOM persists | S155:28, :64-68 |
| CPU | i5-8400T → i5-9400 | throughput improved; OOM persists | S155:29, :70-74 |
- Config sequence: commit 52f16bc (03-23 18:42) gpu_count=4 "pending M.2 SATA drive replacement"; commit 8b04050 (03-24 17:00) "restore rrig6600c to 8 workers — new M.2 SATA installed, axGear adapter removed". The changelog then says rrig6600c was set to gpu_count=4 and "8 workers always OOMs". It cites commit "89c7597" for that, **but in git 89c7597 is the S130 commit**, so the hash in the changelog is wrong.
- Result: 3-trial run on 22 GPUs (rrig6600c at 4), all complete. "VA bloat is at the ROCm/amdgpu driver level, below Python/CuPy control" (the record's claim).
- Anchors: `docs/SESSION_CHANGELOG_20260324_S155.md:2-4, 10, 16-31, 37-74, 80-94, 100-127`; commits 52f16bc, 8b04050, c5af655, 5d8d10d.

**E37. 2026-03-25/26 (S156): PWC lifecycle + BIOS NVRAM.** Rig: rrig6600c.
- Monitor: `remote_crash_monitor.py`, polling every 2 s for PageTables and py_procs.
- Signature: py_procs 6 → PageTables 15,000 kB; 9 → 12,844 kB **CRASH**; 12 → 19,124 kB **CRASH** ("kernel freeze / D-state").
- Root cause 1 (TB confirmed): `run_trial_persistent()` creates a new PWC per trial. After a mid-run reboot, a new trial stacks 8 workers on survivors. Fix, Phase A bandaid: pre-spawn `pgrep -af "sieve_gpu_worker.*--persistent"` → SIGTERM, then SIGKILL, 2 s ROCm release delay. v1 (broad `pkill -9`, commit 1f03d24) → v2 (targeted) had a syntax error → restored v1 (commit c1cde8f) → v2 fixed (commit 0577c53, 03-27). Phase B (session-scoped PWC, `reset_for_new_trial()`) deferred.
- Root cause 2: "BIOS NVRAM corruption". Evidence: power-on failure stopped working; hard freeze at ~13,000 kB PageTables; e820 hole at `0x37700000` (M.2 remnant); crashes even with 0 stale workers at py_procs=6. Fix: reflash **B36FF608.BSS** (same version). Settings: Above 4G MMIO Enabled, Resize BAR Disabled, RC6 Render Standby Disabled, VT-d Enabled.
- Result: post-reflash stable at PageTables 13,500-13,900 kB, py_procs 5, 73+ samples. The failure mode changed from hard kernel freeze to a soft CuPy OOM exception.
- TB A→D test matrix: A 2 workers PASS (PageTables 35,008); B 3 workers post-BIOS PASS (some chunk CuPy OOM); C 4 workers pre-BIOS **hard freeze** (10,508); C 4 workers post-BIOS **CuPy OOM, machine up**; D 8 workers not tested.
- PageTables at 8 workers: rrig6600 38,684 kB and rrig6600b 39,640 kB, both stable. "PageTables are NOT the crash cause — they are a symptom".
- Config: commit d4eb55d set rrig6600c to **3 workers**. Run 1 (seeds 0→1,073,741,824) 3 trials complete — "first time in project history".
- Anchors: `docs/SESSION_CHANGELOG_20260326_S156.md:2-4, 10-16, 20-61, 65-90, 94-105, 109-131, 135-139, 153-160, 165-172`; `docs/PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md:2-6, 14-35, 49-61, 71-121, 130-189, 197-250, 306-330`; commits 1f03d24, c1cde8f, 0577c53, d4eb55d.

**E38. 2026-03-27/28 (S157).** Rig: rrig6600c (3 workers max); cluster 21 GPUs.
- Observation (TB): 3 workers → py_procs 5, PageTables ~13,000 → STABLE; **4 workers → py_procs 6, PageTables ~9,260 → HARD CRASH**. Because PageTables at the crash were lower than in the stable state, TB read this as "concurrent ROCm init timing, not memory pressure".
- Plan: Phase 1, serialized startup with a 12–15 s stagger + warmup job; Phase 2, idle-only 4 workers for 2–3 min (crash → hardware/platform); Phase 3, GPU set permutation {0,1,2,3}, {0,1,2,4}, {0,1,3,4} ("faulty riser/slot"); Phase 4, swap GPU/riser; Phase 5, VSZ/RSS per worker. **No results recorded.**
- Action deployed: commit c06e4e4, per-worker `CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_{gpu_id}` "prevents 8-worker cache race".
- Anchors: `docs/S157_CHAT_PROMPT.md:2-4, 11-22, 34-63, 98-100`; `docs/SESSION_CHANGELOG_20260323_S157.md:2-5, 11-22, 30, 37-50` (header date 2026-03-23 conflicts with the commit date 2026-03-28 and the prompt date 03-27; the file says "reconstructed"); commit c06e4e4.

**E39. 2026-03-29 (S158): multi-rig isolation.** Rigs: rrig6600, rrig6600c (8 workers each in isolation), rrig6600b.
- Belief: PWC launches 537 threads at once (one per chunk) → GIL thrashing → SSH pipe read timeouts → workers marked dead → respawn storm → "rig kernel panic". Ruled out: fd exhaustion (Zeus peak 81 fds; ulimit raised 1,024→1,048,576 in `/etc/security/limits.conf` — "did NOT fix crashes"); hardware ("both rigs stable in isolation, crash only together"); PageTables (the low value at crash is a symptom).
- Isolation table:
  - rrig6600 alone (8 AMD + Zeus): 537/537 pass.
  - rrig6600c alone, forward: pass.
  - rrig6600c alone, reverse: "Stalled chunk 88/537 — hang evidence".
  - rrig6600 + rrig6600c (16 AMD + Zeus): 537/537, 14,046 bidi, 34:15 elapsed, zero crashes.
  - All 3 rigs (24 AMD + Zeus): **hard kernel panic on rrig6600c**.
- Fixes: S158B-v3 bounded dispatch, `ThreadPoolExecutor(max_workers=min(num_workers, len(chunks)))`, progress-aware wait, `shutdown(wait=False, cancel_futures=True)` (commit b0235e7, TB approved after 3 cycles). Kept.
- S158C: `ROCM_CLIFF_NODES` inter-node spawn stagger (commit 8d62491), increased 10 s→30 s (commit c9e7221). **REVERTED** (commit d1301e1): it "caused spawn crashes on rrig6600 and rrig6600b"; "solving the wrong problem (crash during spawn vs crash during dispatch)".
- seed_cap_amd restored to 2,000,000.
- **Not recorded:** how rrig6600c went from the 3-worker cap (d4eb55d) back to the 8 workers in this table. No distributed_config.json commit between 03-26 and 03-30 (git log). rrig6600b was "never" tested in a 2-rig pair (TODO).
- Anchors: `docs/SESSION_CHANGELOG_20260329_S158.md:2, 11-13, 19-40, 46-52, 58-72, 78-87, 136-142, 150-152`; commits b0235e7, 8d62491, c9e7221, d1301e1.

**E40. 2026-03-29 → 03-31 (S158D → S159 start): transport change to ZMQ+SQLite.** All rigs.
- Belief: PWC holds 8 persistent SSH pipes per rig. The proposal records "Kernel panics during worker spawn (rrig6600c repeatedly)", "GIL thrashing from 537 simultaneous blocked SSH pipe threads", and "Respawn storms when pipe timeouts cascade".
- Action: ZMQ PUSH/PULL (ports 5557/5558) plus a SQLite lease ledger. Workers launched once per rig, first via SSH and later via `systemd-run --user` (`loginctl enable-linger` needed; done on rrig6600 only). Zeus `CUDA_VISIBLE_DEVICES` set at Popen per worker, per-GPU `CUPY_CACHE_DIR`. Commit a30f5dc (03-30) enabled ZMQ by default and disabled PWC. TB: "approved with mandatory guardrails", "Risk rated Medium", ladder 1-rig → 2-rig → 3-rig.
- Result at the end of this era: "rrig6600 + Zeus running clean via ZMQ+SQLite+systemd-run" (2-rig validation in progress); rrig6600c and rrig6600b not yet added. GPU utilization "~0% between chunks" (2M-seed chunk ~50-100 ms vs ~600 ms ZMQ round trip). Proposed: seed_cap_amd 2M → 10M after 3-rig stable.
- 03-31, commit 6681d4b (S159F): "startup import, env vars, Zeus worker stagger". The April S159G record continues from here (out of scope).
- Anchors: `docs/PROPOSAL_ZMQ_SQLITE_COORDINATOR_S158D_v1_0.md:13-22, 29-43, 85, 97-112, 119-134`; `docs/SESSION_CHANGELOG_20260329_S158.md:91-126`; `docs/S159_CHAT_PROMPT.md:5-14, 19-39, 54-67, 78-81, 131-136`; commits a30f5dc, 14e7406, 074ab62, 6681d4b.

---

### LIST 1 — Recorded as helping
1. Wiring the scripts_coordinator worker-start stagger (0.5 s ROCm). Described as working (`docs/PROPOSAL_Job_Batching_Pipeline_Stability_v2.md:25, 31`); no direct measurement.
2. `amdgpu.ppfeaturemask=0xffff7fff` (GFXOFF off): gfxoff=00, rocm-smi fields populated (commit f614736). It is part of the 100-trial 100% envelope (commit a490ea0; CHAPTER_9:896-906).
3. sample_size 450 at 12 concurrent: 100 trials, 100% (`git show a490ea0:docs/SESSION_CHANGELOG_20260118.md:17-25`).
4. SMU polling reduction (kill `watch -n 0.5 rocm-smi`, fan LOOP_SECS 20): "No SMU errors in dmesg post-fix" (`docs/SESSION_CHANGELOG_20260120.md:34-37`).
5. Per-node batch cap 6: "25/25 trials … 100% success" (commit 49e8ad4).
6. chunk_size 1000 + the max_concurrent bug fixes: 99/99 at 12 GPUs (`docs/SESSION_CHANGELOG_20260124_OOM_FIX.md:233-241`); Step-3 manifest 5000→1000, "0 failures (was 2 OOM)" (`docs/SESSION_CHANGELOG_20260125.md:14, 29`).
7. Inter-chunk `_best_effort_gpu_cleanup()` in sieve_filter.py: 20/20 trials, 0 GPU hangs (`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:1034`).
8. GPU[4] reseat on rig-6600b + PCIe Gen1: "Sensors now reading healthy" (`docs/SESSION_CHANGELOG_20260201.md:152`).
9. udev perf=high + gfxoff=0 + reboot: all 26 GPUs available (`docs/SESSION_CHANGELOG_20260211_S80.md:54-56, 185`).
10. Rebooting rig-6600b to clear D-state zombies: 20/20 clean re-run (`docs/SESSION_CHANGELOG_20260223_S109.md:54-59`).
11. S133-A semaphore fix: workers survive pass 1. S133-B strategies param: workers survive all 4 passes (`docs/SESSION_CHANGELOG_20260309_S133.md:26, 64`).
12. S146 bundle (kernel-tail split, int32 casts, count clamp, pool 4, timeout 600): 3 trials, zero crashes (`docs/SESSION_CHANGELOG_20260316_S146.md:92-103`).
13. S149-B direct Device(gpu_id), masking removed: 8 workers per rig, preprod 5/5, 24/24 alive (`docs/SESSION_CHANGELOG_20260321_S149.md:36-42`).
14. rrig6600c sshd fix (MaxSessions 50 etc.): declared root cause / RESOLVED (`docs/ROOT_CAUSE_ANALYSIS_RRIG6600C_S151.md:4, 114-128`). The next session's 3-trial pre-slim run was 26/26 with rrig6600c stable (`docs/SESSION_CHANGELOG_20260322_S152.md:72-77`). Crashes later resumed (E35+).
15. Per-worker CUPY_CACHE_DIR: "eliminated rrig6600c crash" (`docs/SESSION_CHANGELOG_20260322_S152.md:48`).
16. rrig6600c at 4 workers: 22-GPU 3-trial run complete (`docs/SESSION_CHANGELOG_20260324_S155.md:80-85`).
17. BIOS reflash B36FF608.BSS: hard freeze → soft CuPy OOM; stable at 3 workers, 3-trial Run 1 complete (`docs/SESSION_CHANGELOG_20260326_S156.md:73-90, 104-105, 122-131`).
18. PWC pre-spawn cleanup bandaid (v1/v2): deployed alongside the stable 3-worker run (S156:41-55). Its separate contribution is not isolated.
19. S158B bounded dispatch: first clean 2-rig run, 537/537 (`docs/SESSION_CHANGELOG_20260329_S158.md:33-40, 62-67`).
20. ZMQ+SQLite+systemd-run: rrig6600 + Zeus "running clean" (`docs/S159_CHAT_PROMPT.md:5-6`). This is a partial validation.

### LIST 2 — Recorded as not helping, or worse
1. Jan 16 scripts_coordinator stagger fix: reverted; root cause unknown (commit a47ba88).
2. `amdgpu.gfxoff=0` kernel param: "ignored on 6.8.0-90" (commit f614736). It was used again in S80.
3. HIP kernel cache clear: failures continued. GPU warmup barrier: "didn't help". HSA_OVERRIDE_GFX_VERSION fix: failures continued (`docs/SESSION_CHANGELOG_20260123_PART2.md:15-17`; commit 3e6b224).
4. A 4-GPU concurrency cap orphaned jobs (61/99) before the bug fix (`docs/SESSION_CHANGELOG_20260124_OOM_FIX.md:236`).
5. S151 6-worker cap on rrig6600c: reverted within 46 min (commits 38393b5 → cc113b5).
6. `CUPY_CUDA_MEMORY_POOL_TYPE=none`: later blamed for the 41 GB VA OOM (`docs/SESSION_CHANGELOG_20260323_S154.md:63-75`), then removed with "OOM persists" (S155:41, :46).
7. S154 explicit GPU `del` + gc.collect: "Did not resolve the crash" (`docs/SESSION_CHANGELOG_20260323_S153_S154.md:134-136`).
8. CuPy pool cap 256 MB, CUPY_GPU_MEMORY_LIMIT, ROCR_VISIBLE_DEVICES restoration, M.2 SATA replacement, CPU upgrade: all "OOM persists" (`docs/SESSION_CHANGELOG_20260324_S155.md:18-31`).
9. MaxSessions fix as the final root cause: rrig6600c crashes continued afterward (S153–S158).
10. S152 java_lcg_reverse kernel-arg theory and the IPC path: ruled out by Stages 8B and 9 (`docs/SESSION_CHANGELOG_20260323_S153.md:40-66`).
11. Zeus ulimit raise: "did NOT fix crashes" (`docs/SESSION_CHANGELOG_20260329_S158.md:78-81`).
12. S158C inter-node spawn stagger (10 s, then 30 s): "caused spawn crashes on rrig6600 and rrig6600b"; reverted (`docs/SESSION_CHANGELOG_20260329_S158.md:69-72`; commit d1301e1).
13. 4 workers on rrig6600c post-BIOS: CuPy OOM. 4 workers pre-BIOS: hard freeze (`docs/SESSION_CHANGELOG_20260326_S156.md:100-101`; `docs/S157_CHAT_PROMPT.md:39-40`).
14. slim_v1: crashes "began" after deployment, then it was ruled coincidental and parked (`docs/ROOT_CAUSE_ANALYSIS_RRIG6600C_S151.md:24, 84-89`; `docs/SESSION_CHANGELOG_20260322_S152.md:37-49`).

### LIST 3 — Proposed but never tried (or no result recorded) within these files
1. Add RAM to rigs (16 GB) — `docs/SESSION_CHANGELOG_20260123_PART2.md:44-47`. Cap 5 workers per node — `:49-52`.
2. Job-batching Option B (WATCHER trial cap 20) and Option C (`INTER_NODE_STAGGER=5.0`) — `docs/PROPOSAL_Job_Batching_Pipeline_Stability.md:106-143`. drop_caches allocator reset between batches — v2:153-167 (implementation not recorded; 49e8ad4 mentions only batching).
3. Raise STAGGER_REMOTE 0.5 → 1.0 s "if insufficient" — `docs/PROPOSAL_FINAL_ROCm_HIP_Init_Fix_v2_1.md:140`.
4. GPU[4] SMU investigation "Power/riser/ppfeaturemask?" — `docs/SESSION_CHANGELOG_20260122.md:124` (the reseat in E13 is the only follow-up recorded). Hardware replacement of GPU[4] — `:116`.
5. GPU[10] exclusion (`excluded_gpus: [10]`) or watch list — `docs/PROPOSAL_Infrastructure_Improvements_v1_0.md:154-185`. `rocm-smi --resetclocks` post-batch — `:126-127` (Item C was "deployed" as gpu_cleanup.py; its content vs. this script is not verified).
6. Step-1 GPU benchmark suite (4 benchmarks, inter-trial delay, max-seed-count cap) — `docs/STEP1_GPU_BENCHMARK_SUITE.md:640-681`.
7. Phase A / Phase B seed-cap probes (single card with `ROCR_VISIBLE_DEVICES=0`; full-rig concurrent; "multi-worker ROCm tax") — `docs/GPU_THROUGHPUT_INVESTIGATION_PLAN_v1_0.md:64-131`; S128 tables TBD.
8. Node watchdog that detects D-state zombies and SSH-reboots — `docs/TODO_MASTER_S120.md:70` and onward.
9. Node-failure resilience for an Optuna study when a rig drops — `docs/SESSION_CHANGELOG_20260307_S123.md:167`; `docs/TODO_MASTER_S139.md:55`.
10. Reduce spawn stagger 4.0 → 1.0 s after 8-worker stability — `docs/S149_CHAT_PROMPT.md:135`.
11. sshd MaxSessions check in PWC preflight / REMOTE_NODE_SETUP_CHECKLIST — `docs/ROOT_CAUSE_ANALYSIS_RRIG6600C_S151.md:146-147, 155-156`.
12. Kernel bounds guard `#define MAX_SURVIVORS` in hybrid kernels; minimum window_size floor — `docs/SESSION_CHANGELOG_20260322_S152.md:44, 52-54`.
13. Disable syncthing on all rigs; swap 2 → 8 GB — `docs/TODO_MASTER_S154.md:39-54`. Still open at S155 close (`docs/SESSION_CHANGELOG_20260324_S155.md:127`).
14. Respawn-storm fix (537 threads) — `docs/TODO_MASTER_S154.md:94`. Partly addressed by S158B bounded dispatch.
15. PWC Phase B session-scoped lifecycle (`reset_for_new_trial()`, one-worker-per-slot invariant, post-shutdown reap) — `docs/PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md:130-290`; deferred (S156:57-61). Superseded by ZMQ.
16. S157 rrig6600c 4-worker cliff Phases 1–5 (12–15 s serialized startup + warmup, idle-only test, GPU-set permutation for riser/slot, GPU/riser swap, VSZ/RSS) — `docs/S157_CHAT_PROMPT.md:45-63`.
17. Test 8 workers on rrig6600c post-BIOS (Test D) — `docs/SESSION_CHANGELOG_20260326_S156.md:102, 169`.
18. rrig6600b in a 2-rig pair, and a 3-rig run with S158B — `docs/SESSION_CHANGELOG_20260329_S158.md:138-139`.
19. seed_cap_amd 2M → 10M after 3-rig stable — `docs/S159_CHAT_PROMPT.md:14, 81, 136`.
20. Revert `CUPY_CUDA_MEMORY_POOL_TYPE=none` / revert the S151 respawn lock "if desired" — `docs/ROOT_CAUSE_ANALYSIS_RRIG6600C_S151.md:158-159`.

### LIST 4 — Isolation results (every recorded combination)
| Date/session | Rig(s) | Cards / workers | Seed cap / chunk / sample | Kernel/PRNG / step | Transport | Outcome | Anchor |
|---|---|---|---|---|---|---|---|
| 01-16 | both rigs | all 12/rig (50 trials) | — | Step 2 scorer | scripts_coordinator SSH | froze, rebooted twice | commit a47ba88 |
| 01-16 | both rigs | 2–4/rig (5, 12 trials) | — | Step 2 | SSH | worked | commit a47ba88 |
| 01-17 | localhost / rig-6600b / rig-6600 | 2 / 8 / 9 workers, 26 trials | — | Step 2 | SSH | ok / 7/8 (1 manual reboot) / ALL 9 timed out, FROZEN | PROPOSAL_FINAL_ROCm_HIP_Init_Fix_v2_1.md:27-31 |
| 01-17 | both rigs | concurrency 4, 100 trials | — | Step 2 | SSH | "in progress" (result not in the commit) | commit f614736 |
| 01-18 | both rigs | 12 concurrent | sample 350/450/550/650/750/1000 | Step 2.5 | SSH | 100% (100 trials) | CHAPTER_9:910-917 |
| 01-18 | both rigs | 12 concurrent | sample 2000 | Step 2.5 | SSH | "Freeze risk" | CHAPTER_9:918 |
| 01-20 | both rigs | 20 jobs (7/rig) per batch | sample 350–750 | Step 2 benchmark | SSH | 100% | PROPOSAL_Job_Batching_v2:78-81, 273-282 |
| 01-20 | both rigs | 120 jobs (40/rig) at once | — | Step 2 WATCHER | SSH | both rigs crash <60 s | PROPOSAL_Job_Batching_v1:21-23, 40-43 |
| 01-20 | cluster | per-node batch 6 | — | Step 2 | SSH | 25/25 | commit 49e8ad4 |
| 01-22 | rig-6600b, rig-6600 | Step 3 at cap 6 | — | Step 3 | SSH | 6600b kernel deadlock/crash (GPU[4]); 6600 GPU 11 "unknown" | SESSION_CHANGELOG_20260122:19-22 |
| 01-23 | rig-6600 | 7 parallel workers | — | Step 3 | SSH | OOM-killed (1–3 jobs/run, first wave) | SESSION_CHANGELOG_20260123_PART2:9-31 |
| 01-24 | rig-6600, rig-6600b (+Zeus 2) | 12/rig | chunk_size 1000 | Step 3 | SSH | 99/99, 313 s | OOM_FIX:233-249 |
| 01-24 | same | 4/rig | chunk 1000 | Step 3 | SSH | 61/99 (bug) → 99/99 after fix | OOM_FIX:236-237 |
| 01-26 | cluster 26 GPUs | — | ~19K seeds/chunk, 500K seeds (~26 chunks) | Step 1 forward sieve | coordinator.py | GPU Hang (node-11) before; 20/20, 0 hangs after the cleanup | CHAPTER_2:1006-1034 |
| 02-11 | rig-6600b | 8 GPUs on PCIe Gen1 | — | GPU enumeration at boot | — | soft lockup (systemd-udevd CPU#2 26 s) | S80:58 |
| 02-23 | rig-6600b | — | — | Step 2 | scripts_coordinator | D-state zombies (HIP init storm) → reboot → 20/20 | S109:41-59, 128-132 |
| 02-26 | rig-6600 GPU 2 | 26-GPU cluster | 10M seeds × 50 trials | Step 1 | coordinator.py | 1 GPU hang, recovered by retry | S112:104-116 |
| 03-08 | 3 rigs | persistent workers (48 workers confirmed) | 20M-seed jobs | Step 1 sieve | coordinator.py persistent SSH | 17/17 checks, 2.08M sps | S130:85-92 |
| 03-09 | rig-6600c | 4 workers (pool) | 2M AMD cap | Step 1 passes 1-2 | PWC-in-coordinator SSH | libamdhip64 segfaults ~30-60 s → fixed (S133-A) | S132:46-48; S133:18 |
| 03-09 | 3 rigs | — | — | java_lcg_hybrid passes 3-4 | SSH persistent | all workers die (TypeError) → fixed (S133-B) | S133:33-41 |
| 03-15 | rig-6600c (162) | — | 100K seeds, 3,846-seed chunks | Step 1 | PWC SSH | "hung" (SSH overhead > job time) | S145:144-146 |
| 03-16 | 3 rigs | pool 4/rig | 10M seeds, 3 trials | java_lcg fwd/rev/hybrid fwd/hybrid rev | PWC SSH | zero crashes | S146:94-103 |
| 03-19 | 3 rigs | 4 workers/rig, HIP+CUDA mask | — | Step 1 Run 1 | PWC SSH | 0% GPU util, ~1,050 s/s per rig; >4 workers said to crash | TB_RULING_REQUEST_ROCR_ISOLATION_S149:11-21, 67-75 |
| 03-20 | rrig6600 | GPU 0 and GPU 7 spawn test, no masking | — | — | PWC SSH | passed | S149:41 |
| 03-20 | 3 rigs | 8/rig, no masking | — | preprod 5 trials | PWC SSH | 5/5, 24/24 alive | S149:41-42 |
| 03-20/21 | (rig not named) | 8/rig | — | Run 1 trial 7, hybrid reverse | PWC SSH | crash, 22,145 bidi lost | S149:63-66 |
| 03-20/21 overnight | rrig6600c | 8 workers | — | — | PWC SSH, pre-slim | "STABLE for hours" | RCA_S151:23 |
| 03-21 | rrig6600c | 8 workers, MaxSessions 5 | — | slim_v1 | PWC SSH | repeated crashes → physical reboot | RCA_S151:24-25, 52-71 |
| 03-21 | rrig6600c | 4 workers | — | — | PWC SSH | stable (monitor 20:36) | RCA_S151:27, 165-168 |
| 03-22 | all, 26/26 GPUs incl. rrig6600c 8 | 8/rig | — | pre-slim_v1, 3 trials | PWC SSH | stable | S152:72-77 |
| 03-22 | rrig6600c | 8 | window_size=2 (340k surv/chunk) | slim_v1 | PWC SSH | rrig6600c crash eliminated by per-worker cache; Zeus -9 kill | S152:28-33, 46-48 |
| 03-22 | rrig6600b | 8 | — | — | PWC SSH | workers died ~22:09 after ~2 h | TODO_MASTER_S152:49 |
| 03-23 | rrig6600c | 8 GPUs, 1 kernel each | — | java_lcg_reverse correct args / wrong args | direct | pass / pass (tolerated) | S153:31-49 |
| 03-23 | rrig6600c | 8 persistent workers, 1 reverse job each | — | java_lcg_reverse slim_v1 | persistent IPC | pass, 3.1–4.1 s | S153:51-66 |
| 03-23 | rrig6600c | 8 (production) | ~21 reverse jobs/worker over 5+ min | Run 1 T1/T2 | PWC SSH | OOM, total-vm 41,452,828 kB (×2) | S154:25-42 |
| 03-23→25 | rrig6600c | 8 | 256 MB pool cap / env cap / ROCR isolation / new M.2 / new CPU | — | PWC SSH | OOM persists (each) | S155:18-31 |
| 03-25 | Zeus2 + rrig6600 8 + rrig6600b 8 + rrig6600c 4 | 22 GPUs | — | 3-trial production | PWC SSH | all 3 trials complete | S155:80-85 |
| 03-25/26 | rrig6600c | 2 | — | 3 trials | PWC SSH | PASS (PageTables 35,008) | S156:98 |
| 03-26 | rrig6600c | 3 post-BIOS | — | 2+ trials | PWC SSH | PASS (CuPy OOM on some chunks) | S156:99 |
| 03-25 | rrig6600c | 4 pre-BIOS | — | — | PWC SSH | HARD kernel freeze (10,508 kB) | S156:100 |
| 03-26 | rrig6600c | 4 post-BIOS | — | — | PWC SSH | CuPy OOM, machine up | S156:101 |
| pre-BIOS | rrig6600c | py_procs 6–12 (stale + new) | — | — | PWC SSH | crash at py_procs 9 and 12 | S156:23-28 |
| 03-26 | rrig6600 / rrig6600b | 8 each | — | — | PWC SSH | stable (PageTables 38,684 / 39,640) | S156:111-115 |
| 03-26 | Run 1, rrig6600c at 3 | — | seeds 0→1,073,741,824 | 3 trials | PWC SSH | complete | S156:122-131 |
| 03-27 | rrig6600c | 3 vs 4 workers | — | — | PWC SSH | 3 stable; 4 HARD CRASH (PageTables 9,260) | S157_CHAT_PROMPT:37-43 |
| 03-27 | 21 GPUs (rrig6600c 3) | — | seed_start 2,147,483,648 | 3-trial sweep | PWC SSH | "in progress" (result not recorded) | S157_CHAT_PROMPT:4, 24-28 |
| 03-29 | rrig6600 + Zeus | 8 AMD | 537 chunks (2M AMD cap) | Step 1 full sweep | PWC SSH (S158B) | pass | S158:30 |
| 03-29 | rrig6600c + Zeus | 8 AMD | 537 chunks | forward | PWC SSH | pass | S158:31 |
| 03-29 | rrig6600c + Zeus | 8 AMD | 537 chunks | reverse | PWC SSH | stalled at chunk 88/537 | S158:32 |
| 03-29 | rrig6600 + rrig6600c + Zeus | 16 AMD | 537 chunks | full | PWC SSH (S158B) | 537/537, 34:15, zero crashes | S158:33, 36-40 |
| 03-29 | all 3 rigs + Zeus | 24 AMD | 537 chunks | full | PWC SSH | hard kernel panic on rrig6600c | S158:34 |
| 03-29 | rrig6600, rrig6600b | — | — | spawn with S158C inter-node stagger 10/30 s | PWC SSH | spawn crashes → reverted | S158:69-72 |
| 03-30/31 | rrig6600 + Zeus | 8 AMD | 2M chunk | Step 1 | ZMQ+SQLite, systemd-run | "running clean" (in progress) | S159_CHAT_PROMPT:5-9, 66-67 |

---

### Files read
docs/: CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md (keyword hits + §5.3/6.2/6.3/8/8.4/PWC/seed-cap/rig-6600c sections), CHAPTER_9_ADDENDUM_v2_2_0.md, CHAPTER_1_WINDOW_OPTIMIZER.md (hits only), CHAPTER_2_BIDIRECTIONAL_SIEVE.md (§8.1, §9), CHAPTER_3_SCORER_META_OPTIMIZER.md (§9.4 grep only), PROPOSAL_Job_Batching_Pipeline_Stability.md, PROPOSAL_Job_Batching_Pipeline_Stability_v2.md, PROPOSAL_FINAL_ROCm_HIP_Init_Fix_v2_1.md, PROPOSAL_Infrastructure_Improvements_v1_0.md (items C/D + summary), SESSION_CHANGELOG_20260120.md, _20260122.md, _20260123_PART2.md, _20260124_OOM_FIX.md, _20260124.md (grep), _20260125.md (grep), _20260201.md (end note), _20260204_S62.md (grep), _20260211_S80.md, SESSION_81_HANDOFF.md, S107_session_log.md, SESSION_CHANGELOG_20260218_S96A.md (grep), _20260223_S109.md, _20260226_S112.md, SESSION_NOTES_20260118_PIPELINE_TEST.md, GPU_THROUGHPUT_INVESTIGATION_PLAN_v1_0.md, TODO_MASTER_S120/S122/S125b/S126/S127/S132/S135/S139/S143/S145/S148/S150/S152/S154.md (grep), SESSION_CHANGELOG_20260307_S125b.md, _S127.md, _S128.md (both 0307 and 0308), _20260308_S129B.md, _S130.md, _S131.md, _20260309_S132.md, _S133.md, _20260310_S134.md, _S135.md (grep), _S136.md (grep), _S137.md and _S137_FINAL.md (grep), _20260312_S138.md, _20260313_S138_S139.md, _20260314_S141.md, _S142.md (grep), S142_CHAT_PROMPT.md (grep), _20260315_S145.md, PROPOSAL_S145_R1 (grep), _20260316_S146.md, DOCUMENTATION_UPDATES_S146.md (grep), S147_CHAT_PROMPT.md, _20260319_S148.md (grep), TB_RULING_REQUEST_ROCR_ISOLATION_S149.md, S149_CHAT_PROMPT.md, _20260321_S149.md, _20260321_S150.md, TB_RULING_REQUEST_IPC_SERIALIZATION_S150.md (grep), IPC_SERIALIZATION_FIX_IMPLEMENTATION_GUIDE_S150.md (grep), ROOT_CAUSE_ANALYSIS_RRIG6600C_S151.md, _20260321_S151_FINAL.md, S151_CHAT_PROMPT.md (grep), S152_CHAT_PROMPT.md (grep), _20260322_S152.md, _20260323_S153.md, _20260323_S153_S154.md, _20260323_S154.md (diffed against S153_S154 + kern.log section), PROPOSAL_S155_CUPY_POOL_FIX.md, S155_CHAT_PROMPT.md (grep), _20260324_S155.md, _20260326_S156.md, PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md (head + grep), S157_CHAT_PROMPT.md, _20260323_S157.md, _20260329_S158.md, PROPOSAL_ZMQ_SQLITE_COORDINATOR_S158D_v1_0.md (grep), S159_CHAT_PROMPT.md, STEP1_GPU_BENCHMARK_SUITE.md, COMPLETE_OPERATING_GUIDE_v1_1.md and v2_0.md (grep), Cluster_operating_manual.txt and instructions.txt (grep), DOCUMENTATION_AUDIT_20260131.md (grep), PROJECT_FILE_CATALOG.md:121, :826; plus keyword greps over all 168 period-named docs.
Git history (read-only): deleted docs/SESSION_CHANGELOG_20260118.md (a490ea0) and _20260103.md (53a3829); grep of deleted _20260127, _20260129, _20260130, _S70, _S71, _S88_FINAL (no relevant hits). Commit messages: a47ba88 f614736 b457b6f a82860f d2306f5 19cd3b3 d168f83 288de2a 723721a 38393b5 cc113b5 e4905c2 12dbeaf fd38d17 6424575 701c501 52f16bc 8b04050 d4eb55d d1301e1 a30f5dc, plus a490ea0 3e6b224 20b0e7c 94342a4 4d979e0 7288e5e d5219a0 0996582 b5b6532 4b1c975 4d6801a c1c698d f3fdbf1 ae6afb2 12a8b6b 7182ab5 c5af655 5d8d10d 1f03d24 c1cde8f 0577c53 c06e4e4 b0235e7 8d62491 c9e7221 14e7406 6681d4b 49e8ad4 4afcdc7 8ac4047, and git log 2026-01-01..2026-04-01.

### Surfaces not read / not available
- Commit diffs (brief said message only). So the exact `distributed_config.json` values per commit, the per-node stagger values in code, and the ROCM_ENV_VARS contents at each date are taken from the docs, not the diffs.
- `PROPOSAL_ROCm_Monitoring_Desync_Mitigation_v1_0.md` (cited by f614736): not in tree or history.
- `PROPOSAL_S157_rrig6600c_4worker_cliff.md`: not in repo.
- `PROPOSAL_RAM_Disk_Data_Preloading_v1_0.md`: not read.
- Root-level apply_s1xx / fix_s15x scripts, `rig_crash_monitor.sh`, `remote_crash_monitor.py`, `test_rrig6600c_stress.py`, `slim_v1_diag_rrig6600c_v2.py`: not read (code, not record).
- Benchmark CSVs (benchmark_results_20260118_*.csv) and the deleted benchmark_step1_20260125_* rocm pre/post files: not read.
- Rig-side state (kern.log, dmesg, sshd_config, GRUB, BIOS settings, udev rules, fan script), Zeus logs, ser8 archives, the private repo, the Claude project files: not accessible (no SSH by brief).
- April-onward S159G/S160/S162/S163 crash records: out of the Jan–Mar window. They are where the GCVM_L2 / gfxhub page-fault record begins.
- TB_RULING_* files in docs: all are dated after March 2026 except the two TB_RULING_REQUEST files (ROCR S149, IPC S150), which were read. No other Jan–Mar ruling file was found. Team Beta rulings from Jan–Mar exist only as relayed text inside Alpha changelogs and proposals.

---

## Part B — 2026-03-30 → 2026-04-17 (S159–S163: first GCVM_L2 / SQC captures)


### Prior record

Searches run (read-only, worktree `/home/michael/tfm_worktrees/gpu-fault-history`, HEAD 3579088):
- `ls docs | /bin/grep -iE "S159|S160|S161|S162|S163"` and `ls | /bin/grep -iE "S159|S160|S161|S162|S163"` (repo root).
- `/bin/grep -rlE "power_save=0|snd-power|amdgpu.conf|sched_jobs|cwsr_enable|HSA_NO_SCRATCH|SCRATCH_ASYNC|AMD_SERIALIZE_KERNEL|amdgpu-dkms|ban-amdgpu"` over *.md/*.py/*.sh/*.txt.
- `/bin/grep -lE "S159|S160|S161|S162|S163[^K]"` over `docs/TB_RULING_*`, `docs/TB_RULING_REQUEST_*`, `docs/LEADS.md`, `docs/BACKLOG.md`, `docs/TODO_CONSOLIDATED.md`, `docs/*_AUDIT.md`, `docs/CHAPTER_*`. Also grepped `docs/exchange/` for S159G/S162/amdgpu-dkms/free_all_blocks, and `docs/PROJECT_FILE_CATALOG.md` for S159 to S163 rows.
- `git log --since=2026-03-29 --until=2026-04-18`, plus `git show` of the 17 commits I was given and about 30 neighbouring commits.

Documents that already record the subject:
- `docs/PROJECT_FILE_CATALOG.md:128-133` lists TB_SUBMISSION_S159G, TB_UPDATE_S159G, TB_FINAL_S159G, TB_UPDATE_S162_OPTION_B and TB_SUMMARY_S163 as RECORD/RULED, historical.
- `docs/TODO_CONSOLIDATED.md:315` (raise seed_cap 2M→10M, retired) and `:317` (free_all_blocks removal in `sieve_gpu_worker.py`, "still present there", `sieve_gpu_worker.py:91-92`).
- No hits in `docs/LEADS.md`, `docs/BACKLOG.md` or `docs/exchange/*.md`. The only TB_RULING_REQUEST hit (`TB_RULING_REQUEST_CPU_PARALLELISM_S175.md`) is outside this era.

Labels: this file is a historical extraction. Every entry is ALREADY RECORDED in the source it cites. The entries are not new findings. The cross-document contradictions flagged as **CONTRADICTION** are NEW only in the sense that they are collated here; each side is already in the record.

Not searched: the host or rigs (no SSH), ser8 archives, the private repository, netconsole logs (`logs/netconsole_all_rigs.log` and similar are not in the repo), and S163KARG files (another agent covers them).

### File key (anchors)
- P159 = docs/S159_CHAT_PROMPT.md
- TM159 = TODO_MASTER_S159.md (repo root)
- SUB = docs/TB_SUBMISSION_S159G_RIG6600_CRASHES.md
- UPD = docs/TB_UPDATE_S159G_ENV_PROPAGATION.md
- FIN = docs/TB_FINAL_UPDATE_S159G_ROOT_CAUSE_CONFIRMED.md
- CL159 = docs/SESSION_CHANGELOG_20260402_S159G.md (current version, rewritten in 0653f3f)
- CL159orig = `git show bb65aa9:docs/SESSION_CHANGELOG_20260402_S159G.md` (original 196-line version, replaced in 0653f3f)
- P160 = docs/S160_CHAT_PROMPT.md
- CL160 = docs/SESSION_CHANGELOG_S160.md
- CL161 = docs/SESSION_CHANGELOG_S161.md
- P162 = S162_CHAT_PROMPT.md (repo root)
- PR162 = docs/PROPOSAL_S162_RRIG6600C_CRASH_ROOT_CAUSE_v1_0.md
- CL162a = `git show 135a8ba:docs/SESSION_CHANGELOG_20260404_S162.md` (deleted in b7f5e1a)
- CL162b = `git show 89c1512:docs/SESSION_CHANGELOG_20260405_S162.md` (deleted in b7f5e1a)
- CL162 = docs/SESSION_CHANGELOG_S162.md (consolidation, b7f5e1a)
- OPTB = docs/TB_UPDATE_S162_OPTION_B_RESULTS.md
- FIN162 = docs/SESSION_CHANGELOG_S162_FINAL.md
- VIC = docs/SESSION_CHANGELOG_S162_VICTORY.md
- PRFAB = docs/PROPOSAL_FREE_ALL_BLOCKS_REPLACEMENT_v1_0.md
- TM163 = docs/TODO_MASTER_S163.md
- P163 = docs/S163_CHAT_PROMPT.md
- TBS163 = docs/TB_SUMMARY_S163.md
- CH163 = CHANGELOG_S163.md (root; 2026-04-07..10)
- CL163 = SESSION_CHANGELOG_S163.md (root; 2026-04-12)
- CL163F = SESSION_CHANGELOG_S163_FINAL.md (root; 2026-04-12..13)
- CL163R = SESSION_CHANGELOG_S163_RIG_FIXES.md (root; 2026-04-16)
- "docstring X" = module docstring of the repo-root script X

---

### ENTRIES (date-ordered)

#### E1. 2026-03-30, S159 (ZMQ 2-rig validation)
- **Rigs/cards:** Zeus and rrig6600 (8 GPUs). rrig6600b and rrig6600c not yet added (P159:5-6, 58-60, 66).
- **Belief:** the 2-rig run over ZMQ+SQLite+systemd-run was "running clean" (P159:6). Low GPU utilisation was put down to 2M-seed chunks (~50-100 ms GPU burst against a ~600 ms ZMQ round trip) (P159:78-81).
- **Action:** `seed_cap_amd=2,000,000` (P159:67). Planned to raise it to 10M after 3 rigs were stable (P159:14, 136; TM159:68-71).
- **Result:** the 2-rig run outcome is not recorded in these files. TM159:11-15 still says "Confirm rrig6600 2-rig run result".
- **Kept/reverted:** 2M remained the invariant through S160 (P160:100; CL160:126). The 10M raise is never recorded as done; `docs/TODO_CONSOLIDATED.md:315` retires it.
- **Anchors:** P159:5-6, 14, 58-67, 78-81, 136; TM159:11-15, 68-71.

#### E2. 2026-03-31, S159F, commit 6681d4b (Zeus only)
- **Rigs/cards:** Zeus 3080 Ti GPU0/GPU1.
- **Belief:** simultaneous CuPy initialisation on the two Zeus workers collides (6681d4b diff comment "stagger Zeus workers to prevent simultaneous CuPy init collision").
- **Action:** in the Zeus CUDA launch loop of `zmq_sqlite_coordinator.py`, `time.sleep(3)` after launching gpu_id 0. Commit subject: "startup import, env vars, Zeus worker stagger, chunk_id fix".
- **Result:** not recorded.
- **Kept/reverted:** not recorded.
- **Anchors:** 6681d4b (message and diff).

#### E3. 2026-04-01, S159G, commit 561428b (all AMD rigs, ZMQ launch path)
- **Belief:** commit subject "add power mgmt env vars to rig worker launch — prevent D3hot crash".
- **Action:** in the systemd-run command in `zmq_sqlite_coordinator.py` (~line 514), added `--setenv=HSA_ENABLE_SDMA=0`, `--setenv=HSA_ENABLE_RUNTIME_POWER_MGMT=0` and `--setenv=AMDGPU_NO_POWER_PROFILE=1`.
- **Result:** "HSA_ENABLE_SDMA=0 in coordinator env ✅ (partial) — Still crashed" and "HSA_ENABLE_RUNTIME_POWER_MGMT=0 ✅ coordinator — Still crashed" (SUB:111-112). UPD:82-90 later suggests that rrig6600's workers predated this commit.
- **Kept:** yes, superseded by /etc/environment (E12).
- **Anchors:** 561428b diff; SUB:111-112; UPD:84-90, 94-97.

#### E4. 2026-04-02 (earlier S159 sub-sessions), S159G-1: netconsole deployed
- **Rigs:** all three rigs, streaming to a Zeus listener. rrig6600 on `lan0`, rrig6600b on `enp10s0`, rrig6600c on `lan0`, all to Zeus 192.168.3.127:6667. Persisted via `/etc/rc.local`.
- **Belief:** there was no kernel-level crash visibility, because the rigs went offline before writing logs.
- **Action:** `netconsole_listener.py`, `install_netconsole_zeus.sh` and `install_netconsole_rig.sh`, plus a systemd `netconsole-listener` service on Zeus.
- **Result:** "First ever real-time kernel crash capture on rrig6600c".
- **Kept:** yes. Netconsole was used in every later session. On 2026-04-10 rc.local was fixed after the clone (`enp10s0`→`lan0`) (CH163:96, 103).
- **Anchors:** CL159orig:14-32, 39; CL159:64-67.

#### E5. 2026-04-02, S159G-2: first netconsole capture (rrig6600c, ZMQ)
- **Rig/cards:** rrig6600c, PCI `0000:19:00.0` (python pid 3408) and `0000:06:00.0` (pid 3003), faulting at the same time.
- **Recorded sequence:** `gmc_v10_0_process_interrupt: 16 callbacks suppressed` → `[gfxhub] page fault` on both GPUs → `GCVM_L2_PROTECTION_FAULT_STATUS: 0xFFFFFFFF`, `Faulty UTCL2 client ID: unknown (0x1ff)`, `WALKER_ERROR 0x7`, `PERMISSION_FAULTS 0xf`, `MAPPING_ERROR 0x1`, `RW 0x1` → `snd_hda_intel: Unable to change power state from D3hot to D0` → `qcm fence wait loop timeout expired` (both GPUs) → "cp might be in an unrecoverable state…" → `sq_intr: error` (multiple SEs) → `GPU reset begin!` → system goes down.
- **Beliefs at the time:**
  - TB: "GPU virtual memory / bad address failure". TB's initial hypothesis was that 5M seeds was above the validated AMD operating point (2M).
  - Alpha: address `0x0000000000002000` was "near-NULL, consistent with use-after-free" (multiple addresses, though). Alpha proposed a CuPy `free_all_blocks()` cross-worker race; **TB rejected it as an "unproven lead"**.
  - Web research: `iommu=pt` is cited from the Gentoo ROCm wiki as a fix for multi-GPU AMD page faults.
- **Action:** led to E6 to E9.
- **Anchors:** CL159orig:36-73.

#### E6. 2026-04-02, S159G: seed-cap testing (ZMQ)
- **Rigs:** "both rigs" (CL159orig:76; which two is not stated beyond rrig6600 and rrig6600c being the crashing rigs, CL159orig:10).
- **Belief:** 5M is above the validated 2M point (TB).
- **Action:** ran at 5M, then reduced `seed_cap_amd` from 5M to 2M.
- **Result:** "Run at 5M seeds → crash (both rigs)"; "Run at 2M seeds → also crashed". SUB:107 says "seed_cap_amd 5M → 2M … Still crashed at 2M". TB ruling: 5M "downgraded to 'amplifier at most'" (CL159:90).
- **Kept:** 2M stayed the operating point (CL159orig:79; P160:100).
- **Anchors:** CL159orig:58-61, 75-79; SUB:107; CL159:88-90.

#### E7. 2026-04-02, S159G-3: package-difference finding
- **Rigs:** rrig6600b has a minimal compute-only amdgpu install. rrig6600 and rrig6600c have the full desktop/multimedia stack: `amdgpu-lib`, `amdgpu-lib32`, `amdgpu-multimedia`, `mesa-amdgpu-*`, `libva-amdgpu-*`, `libwayland-amdgpu-*`, `libllvm19.1-amdgpu` and `xserver-xorg-amdgpu-video-amdgpu`.
- **History recorded ("Claude Error Log"):** rrig6600c was originally cloned from rrig6600b and then started crashing. Claude advised installing the extra packages to match rrig6600 ("wrong"), then advised cloning rrig6600 onto rrig6600c ("wrong again").
- **Belief:** the extra packages are suspect.
- **Action:** decision to "Clone rrig6600b (pending — after stability test with iommu=pt)".
- **Result:** not done in S159G. Later events are in E35 (package removal) and E36 (clone restore).
- **Anchors:** CL159orig:83-104, 162-166; SUB:182 (P3 full clone).

#### E8. 2026-04-02, S159G-4: GRUB standardisation and iommu=pt
- **Rigs:** all three.
- **Belief:** inconsistent GRUB across rigs. `iommu=pt` is a documented fix (E5).
- **Action:** every rig set to `GRUB_CMDLINE_LINUX_DEFAULT="quiet splash pcie_aspm=off pci=noaer amdgpu.ppfeaturemask=0xffff7fff amdgpu.gfxoff=0 amdgpu.runpm=0 amdgpu.aspm=0 pci=assign-busses,hpbussize=0x33 iommu=pt"` and `GRUB_CMDLINE_LINUX=""`. The base is rrig6600b's config plus `iommu=pt`. Verified booted on all three.
- **Result:** "iommu=pt kernel parameter ✅ all rigs — Still crashed"; "GRUB standardization ✅ all rigs — Still crashed". CL160:10 still calls the S159G work the "ROCm env var + iommu=pt fix".
- **Kept:** not recorded as reverted. CH163:107 says "identical grub params" after the 2026-04-10 clone.
- **Anchors:** CL159orig:108-122, 150; SUB:108-109.

#### E9. 2026-04-02 17:49, rrig6600 crash (ZMQ 3-rig run, netconsole)
- **Rig/card:** rrig6600, single GPU `0000:03:00.0`, single PID 5153 (python).
- **Recorded sequence:**
  - First: `GCVM_L2_PROTECTION_FAULT_STATUS: 0x00801231`, `Faulty UTCL2 client ID: SQC (inst) (0x9)`, `WALKER_ERROR 0x0`, `PERMISSION_FAULTS 0x3`, `MAPPING_ERROR 0x0`, address `0x0000723cae6bd000`.
  - About 100 ms later it escalated to `0xFFFFFFFF`, `unknown (0x1ff)`, `WALKER_ERROR 0x7`, `PERMISSION_FAULTS 0xf`, `MAPPING_ERROR 0x1`, `RW 0x1`, repeating every ~100 ms.
  - Then `snd_hda_intel 0000:03:00.1: Unable to change power state D3hot→D0`, `qcm fence wait loop timeout expired`, "cp might be in an unrecoverable state", and `sq_intr: error, se 0, data 0x80000, err_type 1`. The system went offline.
- **Post-reboot (18:04):** `pci=assign-busses,hpbussize=0x33` triggers PCIe bus reassignment across slots 03, 06, 09, 0a, 0d, 10, 13, 16 and 19. Then NIC `lan0` (`0000:0a:00.0`) shows `NETDEV WATCHDOG … transmit queue 0 timed out` twice (5644 ms and 5304 ms). This is recorded as seen only on rrig6600.
- **Also:** rrig6600b and rrig6600c logged `perf: interrupt took too long (2503/2502 > 2500)`.
- **Belief:** "NOT a multi-GPU race". It is a GPU VM fault leading to unrecoverable queue state, with suspected PCIe instability after reassignment.
- **Status of rigs in this doc:** rrig6600b "never crashed"; rrig6600c "survived this run" but "previously crashed in earlier sessions"; rrig6600 was "originally stable rig".
- **Anchors:** SUB:9-11, 17-58, 62-73, 128-134.
- **CONTRADICTION:** later documents describe the SQC (inst)-first signature as *new* after the S162 cleanup patch (PR162:44-62, 174-183; CL162:50-56; CL162a:142-155; OPTB:115-117). It is already recorded here, on rrig6600, on 2026-04-02, under ZMQ, before any TCP cleanup patch existed.

#### E10. 2026-04-02, S159G: ZMQ lease recovery (resilience, not a fix)
- **Result:** when rrig6600 crashed and rebooted mid-run, expired leases were redistributed to rrig6600b and rrig6600c, and rrig6600 rejoined. The forward sieve completed with 126,610,730 survivors from 1,073,741,824 seeds.
- **Note:** CL159:122 says the last run was "Killed mid-reverse-sieve (hybrid pass failed — rrig6600 crash)". FIN:81-82 says the forward and reverse sieves completed. **CONTRADICTION** on whether the reverse sieve or the hybrid pass completed.
- **Anchors:** SUB:13, 138-149; FIN:77-86; CL159:51-58, 122.

#### E11. 2026-04-02, S159G: hypotheses and proposals submitted to TB (SUB)
- **Proposals:**
  - P0: `HSA_ENABLE_SDMA=0` in /etc/environment (SUB:165-169).
  - P1: linux-firmware 3.40 → 3.41 on rrig6600 and rrig6600c with the kernel held (SUB:120, 156, 171-177). The firmware update was "held pending TB guidance" (SUB:120).
  - P2: remove `pci=assign-busses,hpbussize=0x33` from rrig6600 GRUB only (SUB:157, 179-180).
  - Q4: add `HIP_PATH=/opt/rocm/hip` to `.bashrc` (SUB:158).
  - P3: full clone of rrig6600b to rrig6600 (SUB:182).
- **Research cited:** ROCm issue #5238 (gfx1032, identical pattern), and a Medium article citing MES firmware bugs (SUB:116-124).
- **Env table:** `.bashrc HSA_ENABLE_SDMA=0` present on all rigs. `/etc/environment HSA_ENABLE_SDMA` absent on all. `rocm_env/bin/activate HSA_ENABLE_RUNTIME_POWER_MGMT=0` present on rrig6600 and rrig6600c, absent on rrig6600b (SUB:92-99).
- **Results:** P0 was done as E12. P1 is not recorded as applied in-era; linux-firmware is listed as an apt **hold** on all rigs on 2026-04-07..10 (CH163:40). P2 is not recorded as done; the GRUB string with `pci=assign-busses` is unchanged in the record. HIP_PATH is not recorded.
- **Anchors:** as cited.

#### E12. 2026-04-02, S159G: env-propagation investigation, then the /etc/environment fix
- **Rigs:** rrig6600 (live worker PID) showed only `HSA_OVERRIDE_GFX_VERSION=10.3.0`. rrig6600b and rrig6600c showed `ROCR_VISIBLE_DEVICES=0`, `HSA_ENABLE_SDMA=0`, `AMDGPU_NO_POWER_PROFILE=1`, `HSA_ENABLE_RUNTIME_POWER_MGMT=0` and `HSA_OVERRIDE_GFX_VERSION=10.3.0`. The transient units on b and c carried the full `Environment=` line (UPD:56-66).
- **Beliefs:**
  - UPD: Hypothesis A, a stale launch predating 561428b, "Most Likely"; or Hypothesis B, a coordinator code-path difference. rrig6600's unit file could not be checked because it had no active workers (UPD:72-97).
  - FIN: the defect is "consistent across EVERY run we tested — including a fresh launch today". The "most likely explanation" is rrig6600's `.bashrc` `[ -n "$PS1" ]` guard before sourcing `rocm_env/bin/activate`, or the `bash -lc` login shell "clearing or not preserving" the systemd vars (FIN:44, 55-73).
  - Missing `HSA_ENABLE_SDMA=0` is called "the critical variable" (CL159:37-38). TB final ruling: "Primary root-cause candidate: worker env propagation defect" (provisional accept) (CL159:92-101).
- **Action:** appended `HSA_ENABLE_SDMA=0`, `HSA_ENABLE_RUNTIME_POWER_MGMT=0` and `AMDGPU_NO_POWER_PROFILE=1` to `/etc/environment` on **all three** rigs. `ROCR_VISIBLE_DEVICES` was deliberately left coordinator-managed. All three rigs were rebooted and verified identical (CL159:69-81; FIN:111-118). Earlier the same day `HSA_OVERRIDE_GFX_VERSION=10.3.0` had been added to /etc/environment on all three, and that "Still crashed" (CL159orig:126-140; SUB:110).
- **Result:** acceptance pending at session end (CL159:15-16, 97-100). See E13 and E14.
- **Kept:** yes. It is listed as completed (TM163:12) and as a required-vars invariant (TM163:130; CL160:129).
- **CONTRADICTION:** UPD says a stale launch was most likely and rrig6600's unit could not be checked. FIN says the defect reproduces on a fresh launch. The PS1-guard mechanism is recorded as "most likely", not demonstrated. CL159:45-49 states it as "Investigation revealed".
- **Anchors:** UPD:15-97, 101-119; FIN:9-73, 90-133; CL159:20-49, 69-101.

#### E13. 2026-04-03, S160: worker env-invariant check (v1 to v4)
- **Action:** commits 55b2645, 4fcfa51, ecc2a88 and 815cd06. v4 replaced `systemctl --user` (no D-Bus in SSH exec) with `/proc/*/cgroup` PID discovery. `apply_s160_worker_env_check.py` (docstring) adds `_verify_worker_env()`, run 45 s after launch, which raises RuntimeError if a var is missing.
- **Result:** `[EnvCheck] PASS` on 192.168.3.120 (MainPID 100024), .154 (8938) and .162 (7221).
- **Kept:** yes (TM163:13).
- **Anchors:** CL160:14-47; docstring apply_s160_worker_env_check.py.

#### E14. 2026-04-03, S160: crashes continue with env vars confirmed
- **Rigs:** rrig6600 (.120) and rrig6600c (.162) "crashed repeatedly under load". rrig6600b (.154) "never crashed across entire session". The crash "moved between rigs depending on which was under active load"; it was "workload-triggered and time-delayed".
- **Signature:** `0xFFFFFFFF` / `WALKER_ERROR 0x7` / `PERMISSION_FAULTS 0xf` / `MAPPING_ERROR 0x1`.
- **Belief:** 0xFFFFFFFF means "PCIe read returned all ones … device stopped responding on the bus". Recovery required a "physical riser power cycle … (fan at 100%, OS-invisible until power cycled)".
- **Result:** "env var fix was confirmed operational but insufficient — GPU crashes continued".
- **Anchors:** CL160:8-10, 51-64.
- **CONTRADICTION:** SUB:11, 133 says rrig6600c survived the 2026-04-02 run. CL160 says it crashed repeatedly in S160. Later (CL163F:109-112) the same 0xFFFFFFFF/0x1ff/WALKER 0x7 signature is attributed to "unbounded CuPy memory pool growth across trials", not to a PCIe dropout.

#### E15. 2026-04-03, S160 v5, commit 7e1a62e: inter-chunk cleanup in the ZMQ worker
- **Belief:** the ephemeral coordinator's fresh process per chunk gave implicit GPU cleanup. Persistent ZMQ workers never called `_best_effort_gpu_cleanup()` (`sieve_filter.py`, TB-approved 2026-01-26). Allocator, HIP context and CuPy pool state accumulated until the "GPU memory controller hit an unrecoverable fault state". TB language: "Leading Runtime Hypothesis".
- **Action:** in `zmq_sqlite_worker.py`, after `_send_result(result)`, `from sieve_filter import _best_effort_gpu_cleanup; _best_effort_gpu_cleanup()` in try/except. The function then did gc.collect, `torch.cuda.synchronize`/`empty_cache`, and `free_all_blocks()` on the default and pinned pools (PR162:161-172).
- **Result:** a full bidirectional trial (java_lcg forward + reverse + java_lcg_hybrid forward) finished with zero GPU crashes, zero netconsole faults and all 3 AMD rigs active. rrig6600c "ran past all previous crash thresholds". Cost: a ~26 min cross-pass stall at the forward→reverse transition (CL160:111). All 537 hybrid chunks failed via lease exhaustion in trial 2, "not a crash" (CL160:109).
- **Kept:** yes for ZMQ (TM163:14). Recommended for PWC (CL160:97-101, 117).
- **Anchors:** CL160:66-93, 97-101, 105-111; 7e1a62e (zmq_sqlite_worker.py +11).

#### E16. 2026-04-03/04, S161: TCP-PWC gates (no faults)
- **Gate 1:** 500K seeds, 1 rig, 1 GPU, passed (CL161:33-35).
- **Gate 2:** 50M seeds, 3 rigs, 24 AMD GPUs, 25 chunks × 2M. "Zero crashes, zero netconsole faults" (CL161:37-40).
- **26-GPU run:** 24 AMD + 2 Zeus, 50M seeds in 53 s, 2,240,701 sps, all 4 nodes active (CL161:42-46).
- **Related launch changes:** sequential per-GPU launch, "Prevents simultaneous ROCm context competition" (docstring apply_s161_sequential_launch.py). TCP stagger cut from 4 s to 1 s (docstring apply_s161_min_workers_v2.py). Heartbeat 90 s → 120 s because "Each GPU on rrig6600 takes ~90s to initialize ROCm sequentially" (docstring apply_s161_heartbeat_timeout.py). The v2 two-phase online→init→ready protocol has workers importing sieve_filter (ROCm init) **in parallel**, ~90 s (CL161:70-80).
- **Note:** the ~2.2M sps runs were short (53 s). Cleanup was not yet in the TCP worker (CL160:117; CL162:31-33).
- **Anchors:** as cited.

#### E17. 2026-04-04, S162-1: first WATCHER-driven 26-GPU TCP-PWC run
- **Rigs:** all three plus Zeus, 26/26 GPUs, 7,162,151 sps.
- **Result:** rrig6600c crashed at ~18 s. Signature: `0xFFFFFFFF`, `unknown (0x1ff)`, `WALKER_ERROR 0x7`, `PERMISSION_FAULTS 0xf`, `MAPPING_ERROR 0x1`, "Multiple GPUs + PIDs simultaneously → watchdog hard LOCKUP" (PR162:35-42 adds "watchdog: hard LOCKUP on cpu 4").
- **Transport commits:** 716e641 (manifest default TCP-PWC) and 6432a48 (parser, fallback, helptext).
- **Anchors:** CL162:19-29; CL162a:35-47; PR162:35-42.
- **CONTRADICTION:** FIN162:22 records the S162-1 crash as "SQC (inst) page fault signature".

#### E18. 2026-04-04, S162-2: TCP worker cleanup (commits b3c3207 / 2fa4851 / 64d0307)
- **Belief:** the S160 cleanup was never in `persistent/pwc_worker_service.py`. All three rigs lacked it (`grep` returned nothing on rrig6600).
- **Action:** `_best_effort_gpu_cleanup()` called after `_send_result()` in the TCP worker, then scp'd to all three rigs (CL162a:51-68).
- **Commit oddity:** 2fa4851 (subject "add _best_effort_gpu_cleanup … prevents page fault crash") only *deletes* 18 lines, the S161 clean-exit / ECONNREFUSED logic. 64d0307 (same subject, 19:00) adds the cleanup call and restores the clean-exit block.
- **Result:** see E21. The crash was delayed from ~18 s to ~50 min (PR162:176-177).
- **Kept:** yes, until S163 removed free_all_blocks from within the function (E31).
- **Anchors:** CL162:31-34; CL162a:51-72; PR162:154-159; 2fa4851 and 64d0307 diffs.

#### E19. 2026-04-04, S162-3: isolation test, rrig6600c + Zeus
- **Action:** `gpu_count=0` for rrig6600 and rrig6600b in `distributed_config.json`. 10 GPUs, full 3-trial run.
- **Result:** "Complete success — 18m 52s, zero crashes, zero netconsole faults". 1.07B seeds at ~948K sps (CL162a:246).
- **Belief recorded:** "rrig6600c crash caused exclusively by simultaneous 3-rig load".
- **Kept:** config restored afterwards (CL162a:234).
- **Anchors:** CL162:36-40; CL162a:76-86, 234, 246; PR162:18-28.

#### E20. 2026-04-04, S162-4: Zeus network tuning and dispatch semaphore (Zeus side)
- **Belief:** 24 handler threads calling `conn.send_obj()` at the same moment flood Zeus's single 1GbE I210 NIC (enp2s0). Per the apply_s162_dispatch_semaphore.py docstring, "rrig6600c workers (last rig spawned) receive delayed/stalled job payloads → GPU page fault → cascade crash". The proposal itself says it "does not explain the GPU crash mechanism" (PR162:135-139). Zeus CPU oversubscription: ~31 threads on 24 logical (PR162:109-127).
- **Action:**
  - sysctl `net.core.wmem_max=16777216`, `wmem_default=1048576`, `rmem_max=16777216`, `rmem_default=1048576`, `tcp_wmem/tcp_rmem=4096 1048576 16777216`, persisted to `/etc/sysctl.conf`.
  - `threading.Semaphore(8)` around job `send_obj()` in `TCPWorkerTransport._handle_client()` (commit 135a8ba).
- **Result:** "TCP send buffer contention — eliminated by semaphore + buffer tuning (no change)" and "Zeus NIC bandwidth — TCP tuning applied, no effect on crash" (OPTB:125-126). Later called "unnecessary" (FIN162:32); "review for removal" (CL162b:57, 110-112); "Revert unnecessary workarounds from coordinator (stagger, semaphore, cleanup patches)" (FIN162:92).
- **Kept/reverted:** the semaphore is not removed by 7278db6, whose diff touches only coordinator, reverse sieve and config. The sysctl tuning is "benign and can stay" (CL162b:61-64). No removal is recorded in-era.
- **Anchors:** PR162:74-139, 273-279; CL162:42-46; CL162a:90-133; 135a8ba diff; docstring apply_s162_dispatch_semaphore.py.

#### E21. 2026-04-04, S162-5: second 26-GPU run (cleanup + semaphore)
- **Rig/card:** rrig6600c, `0000:16:00.0`, PID 5141 (single GPU and PID).
- **Recorded sequence:** `0x00801231` SQC (inst) (0x9), WALKER 0x0, PERMISSION 0x3, MAPPING 0x0, address `0x00007cbb41419000` → `0x00801031` TCP (0x8), "milliseconds later" → `0xFFFFFFFF` about 100 ms later → qcm fence timeout → `snd_hda_intel … D3hot to D0` → rig offline.
- **Timing:** "kernel uptime 3036s (~50 minutes into session)".
- **Belief:** the cleanup delayed the crash ("cleanup patch working") but may have introduced a new failure mode.
- **Anchors:** PR162:44-67, 174-183, 206; CL162:48-56; CL162a:137-156.
- **CONTRADICTION / caveat:** OPTB:17-24 shows that for its own run, kernel uptime (4078 s) was not trial runtime (~1-2 min after the rig's reboot). No document re-examines whether 3036 s uptime equals ~50 min of trial runtime. The SQC (inst) first fault is called new here, but see E9.

#### E22. 2026-04-04, PROPOSAL_S162: hypotheses and options A to D
- **Hypothesis (Alpha, MEDIUM confidence):** `free_all_blocks()` releases VA mappings that include kernel instruction pages, so the next launch faults on instruction fetch (PR162:187-213).
- **Counter-evidence recorded:** RawKernel binaries are not in the memory pool, and rrig6600 and rrig6600b run identical code without crashing (PR162:202-209, 245-254).
- **Alternative:** VRAM pressure with 8 workers per rig, or driver page eviction (PR162:256-269).
- **Research:** ROCm issue #5616 (SQC inst → MES REMOVE_QUEUE failure → evict failure → GPU reset) (PR162:219-233).
- **Options:**
  - A: gc-only cleanup. Script `apply_s162_safe_cleanup.py`, whose docstring states "Root cause confirmed via netconsole … rrig6600c's ROCm driver aggressively unmaps".
  - B: `AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3` on rrig6600c.
  - C: cap rrig6600c at `gpu_count=4` ("S155 stable operating point").
  - D: delay rrig6600c worker start by 30-60 s.
- **Result:** B was run (E23). A, C and D are not recorded as run in S162 (see list 3).
- **Anchors:** PR162:283-373; docstring apply_s162_safe_cleanup.py; 135a8ba stat (script added).

#### E23. 2026-04-04 ~21:52, S162 Option B: AMD_SERIALIZE_KERNEL=3, AMD_SERIALIZE_COPY=3 on rrig6600c only
- **TB ruling quoted:** "approves … on rrig6600c only as the next targeted diagnostic"; "not a permanent production setting" (docstring apply_s162_option_b.py; 7278db6 removed comment).
- **Action:** env added to rocm_env when `host == '192.168.3.162'`. Cleanup with free_all_blocks still active. All 3 rigs, 26 GPUs.
- **Result:** crash at 21:53:25, about 1-2 min into the trial (rig rebooted 20:45:52; uptime 4078 s).
  - `0000:16:00.0` (GPU6, PID 2986) SQC (inst) `0x00801231`; 48 ms later `0000:06:00.0` (GPU1, PID 2693) SQC (inst) `0x80801231`. Both went to `0xFFFFFFFF`.
  - 21:53:34: qcm timeouts on 16:00.0 and 06:00.0, and "Trap debug id already reserved" on `0000:03:00.0`.
  - 21:53:44: `0000:19:00.0` qcm timeout and "Failed to evict process queues"; "Failed to restore queues" on 06 and 16; "Failed to suspend process 0x8010 through 0x8017" (all 8 workers).
  - 21:53:55: `watchdog: BUG: soft lockup - CPU#3 stuck for 23s! [python:2997]`.
- **Belief after:** async overlap ruled out. A KFD resource limit or race under 3-rig load, framed by TB as "a 3-rig concurrency-triggered system bug whose first visible casualty is rrig6600c".
- **Kept/reverted:** replaced by Exp1 (docstring apply_s162_exp1_scratch_reclaim.py). The comments were removed in 7278db6. CL162b:31 says "No effect"; FIN162:26 says "no effect".
- **Anchors:** OPTB:9-145; docstring apply_s162_option_b.py; 7278db6 diff.

#### E24. 2026-04-05, S162 Exp1: HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0 on rrig6600c only (TB-directed)
- **Belief:** ROCr async scratch reclaim under full concurrency invalidates VA mappings, including instruction pages. TB rationale is quoted in the docstring. Plan: if it still crashes, go to Exp 2 (`HSA_NO_SCRATCH_RECLAIM=1`).
- **Action:** `_exp1_vars=["HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0"]` for host 192.168.3.162 in `persistent_worker_coordinator.py`, exported in the launch script.
- **Result:** CL162b:32 says "❌ No effect". FIN162:29 says "partial, not a fix".
- **Kept/reverted:** CL162b:50-55, 109 say it must be reverted. Code removed in 606f94f; export lines removed in 7278db6.
- **CONTRADICTION:** "No effect" (CL162b) against "partial" (FIN162).
- **Anchors:** docstring apply_s162_exp1_scratch_reclaim.py; CL162b:32, 50-55, 109; FIN162:29; 606f94f and 7278db6 diffs.

#### E25. 2026-04-05, S162: amdgpu.cwsr_enable=0 (GRUB)
- **Rig:** rrig6600c (diagnostic). Per CL162 it was also tried on rrig6600b and rrig6600.
- **Recorded results, which disagree:**
  - CL162b:33 says "cwsr_enable=0 GRUB param on rrig6600c — ❌ No effect"; CL162b:49 says "added to GRUB (diagnostic — can be reverted)"; CL162b:114 says "Remove cwsr_enable=0 from rrig6600c GRUB".
  - CL162:82-92 says "required ONLY on rrig6600c". "Adding to rrig6600b caused KIQ fence timeouts (broke stable rig)"; "Adding to rrig6600 caused instability". The final table: rrig6600 No, rrig6600b No, rrig6600c Yes. Invariant CL162:148 and end state CL162:158 ("✅ rrig6600c only").
  - FIN162:30 says "cwsr_enable=0 — unnecessary".
- **Kept/reverted:** CL162 records it as kept on rrig6600c. Removal is not recorded in-era. After the 2026-04-10 clone all rigs carry rrig6600b's "identical grub params" (CH163:107), and rrig6600b had No per CL162:90. Whether cwsr survived the clone is not recorded.
- **CONTRADICTION:** three-way (no effect / required on c only / unnecessary).

#### E26. 2026-04-05, S162-7/8: amdgpu-dkms driver
- **Belief:** all rigs were on Ubuntu's stock kernel amdgpu (`/lib/modules/6.8.0-106-generic/kernel/drivers/gpu/drm/amd/amdgpu/amdgpu.ko`), which "cannot handle 8 concurrent GPU compute workers under full 3-rig concurrent load". The ROCm 6.4.3 userspace was not matched with AMD's kernel driver (CL162b:95-99). A `ban-amdgpu-dkms` apt pin on rrig6600 had blocked it (CL162b:101-103).
- **Action:**
  - `sudo apt-get install -y amdgpu-dkms && sudo reboot` (FIN162:39-41). Version `1:6.12.12.60403-2194681.22.04` plus `amdgpu-dkms-firmware`, with pin `/etc/apt/preferences.d/pin-amdgpu-dkms` (CL162b:42-45).
  - Verified via `modinfo amdgpu` → `/lib/modules/6.8.0-106-generic/updates/dkms/amdgpu.ko` (CL162:71-79).
  - Sequence: rrig6600c first, then rrig6600 and rrig6600b (FIN162:36-49). Holds on `amdgpu-dkms`, `amdgpu-install`, `linux-image-generic` and `linux-headers-generic` (FIN162:62-68), extended in CH163:40.
- **Recorded results, which disagree:**
  - CL162b:34 says "DKMS on rrig6600c only — ✅ Extended stability to 25 min". CL162b:35 says "DKMS on ALL 3 rigs — ✅ Full clean run confirmed". CL162b:22-23 and 70-80 say "All 3 trials completed clean"; peak 35,632,151 sps; Trial 1 and Trial 2 clean; "Pipeline Step 1 ✅ PASSED, proceeded to Step 2".
  - FIN162:14 and 42 say that after DKMS on rrig6600c alone came the "first clean 26-GPU run at 35.6M seeds/sec, all 3 trials clean".
  - CL162:94-118 says Trial 1 completed (31:31; 9,921 bidirectional). "rrig6600c faulted during reverse sieve but DKMS driver recovered workers". **Trials 2 and 3 did not complete**: rrig6600c went offline, the cluster fell below `--min-workers 24`, and Step 1 was ESCALATED.
  - PRFAB:26-27 and 263-264 say "crashes continued at 2M seed cap post-DKMS". 2M crashed at "~1400 seconds (DKMS)".
  - Root cause stated as "CONFIRMED" in CL162b:8, CL162:67, FIN162:10-14, VIC:28-35 and TM163:15. PRFAB:268 and 275 say "Root-cause status remains provisional", per the TB ruling.
- **Why rrig6600c went first:** CL162b:84-91 says rrig6600c "received jobs last". FIN162:37 says "rrig6600b never crashed because job dispatch order — always got jobs first".
- **Kept:** yes. Pinned on all three (VIC:140-142; TM163:16, 121; P163:14).
- **CONTRADICTION:** the clean/incomplete trial outcome, which rig was installed first and whether the run was single-rig or all-rig, and confirmed versus provisional.

#### E27. 2026-04-05, S162-10: reverse-sieve warm-up and VRAM instrumentation, commit 8d91311
- **Belief:** crashes "always trigger at forward → reverse sieve transition". Cause: "cold kernel compilation on 8 workers simultaneously, no GPU context warmup". The forward worker has `cp.zeros(1)` warm-up; the reverse sieve has none.
- **Action:** in `reverse_sieve_filter.py` `GPUReverseSieve`, `cp.zeros(1)` + `Stream.null.synchronize()` before allocations, plus `[VRAM]` memGetInfo logging before and after alloc.
- **Result:** FIN162:51-52 says "All patches made things worse or had no effect post-DKMS". No specific timing is recorded for this patch.
- **Reverted:** yes, in 7278db6 (diff removes 18 lines from reverse_sieve_filter.py; VIC:169 "REVERTED").
- **Anchors:** CL162:104-112; 8d91311 diff; FIN162:51-58, 75; 7278db6.

#### E28. 2026-04-05 13:22, S162: no-pool diagnostic, commit d79e558
- **Belief:** a stale TLB, or CuPy pool recycling, causes the SQC (inst) faults (diff comment).
- **Action:** in `sieve_gpu_worker.py` `run_worker()`, `cp.cuda.set_allocator(None)` and `cp.cuda.set_pinned_memory_allocator(None)`, i.e. raw hipMalloc. Checkpoint 58b2155 was taken before it.
- **Result:** "made things worse (crashed at 358s)". 070199d: "pool NOT the cause, made things worse".
- **Reverted:** 070199d (13:55).
- **Caveat recorded later:** TBS163:38-40 states `sieve_gpu_worker.py` is the "SSH-PWC path, not used in TCP-PWC runs". CL163F:48-51 says TCP-PWC workers import `sieve_filter` directly. No document reconciles this with the effects recorded for patches made to `sieve_gpu_worker.py` during TCP-PWC runs (see also E30).
- **Anchors:** FIN162:28, 77-79; d79e558 and 070199d.

#### E29. 2026-04-05 14:11, S162: HSA_NO_SCRATCH_RECLAIM=1, all AMD rigs, commit 606f94f
- **Belief:** the diff comment says it "Prevents ROCr async scratch reclaim from invalidating GPU VA mappings (including instruction cache pages)… Root cause of SQC (inst) PERMISSION_FAULTS:0x3 RW:0x0 … Stronger than HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0".
- **Action:** added to `ROCM_ENV_VARS` in `persistent_worker_coordinator.py`. The same commit removed the Exp1 rrig6600c-only block.
- **Result:** "made things worse (crashed at 219s)".
- **Reverted:** a854647 (15:05) "made things worse, restore distributed_config".
- **Anchors:** FIN162:27, 80-81; 606f94f and a854647 diffs.

#### E30. 2026-04-05 16:23, S162: pre-compile all PRNG kernels at worker start-up, commit 22bb145
- **Belief:** the diff comment says "Prevents SQC (inst) page fault on rrig6600c caused by 8 workers simultaneously cold-compiling java_lcg_hybrid_reverse kernel when the first reverse sieve job arrives".
- **Action:** in `sieve_gpu_worker.py` `run_worker()`, a loop of `_get_kernel(fam)` over `list_available_prngs()` plus synchronize.
- **Result:** "made crashes worse" (ac32cd9 subject; FIN162:82-83).
- **Reverted:** ac32cd9 (16:51).
- **Caveat:** same file-path caveat as E28 (TBS163:38-40).
- **Anchors:** 22bb145 and ac32cd9.

#### E31. 2026-04-05, S162: seed-cap reduction under 3-rig load
- **Results recorded:**
  - "Seed cap reduction (1M, 500K) — delayed crashes, did not fix" (FIN162:33).
  - 2M: crashed at ~18 s (stock driver) and ~1400 s (DKMS) (PRFAB:264).
  - 1M: "crashed mid-run under 3-rig load" (PRFAB:265).
  - 100K: "stable, 95+ minutes, 3 rigs, zero crashes" (PRFAB:266).
- **Kept:** `seed_cap_amd=100000` became "current stable config — do not raise without S163 free_all_blocks validation" (TM163:122; P163:29, 81).
- **CONTRADICTION:** FIN162:89, 96-99 "Known Working Configuration" uses `seed_cap_amd: 5000000` ("test 5M seed cap throughput"), and CL162b:115 says "Increase --seed-cap-amd 2M → 5M". This conflicts with 100K as the stable cap and with 1M and 500K crashing.
- **Anchors:** as cited.

#### E32. 2026-04-05, S162-12: late-session forward-sieve crashes, perf level
- **Rigs:** rrig6600 **and** rrig6600c crashed simultaneously in the forward sieve "after multiple crash/reboot cycles".
- **Observation:** `perf=high` (power_dpm_force_performance_level) "resets to auto on reboot". "load imbalance: rrig6600c at 2750MHz, others at 700MHz".
- **Belief:** "Likely cumulative GPU state degradation from day of repeated crashes". Decision: isolation tests after a full clean reboot.
- **Action proposed:** "Make perf=high permanent — systemd startup service on all 3 rigs". Not recorded as done.
- **Anchors:** CL162:120-124, 159, 169-170.

#### E33. 2026-04-05 18:17, S162: restore to the 89c1512 baseline (commit 7278db6)
- **Action:** removed Option B/Exp1 comments and the Exp1 export loop from `persistent_worker_coordinator.py`, removed the reverse-sieve warm-up, and in `distributed_config.json` changed `max_concurrent_script_jobs` from 0 to 8 on one node. The stated end state is all 3 rigs at gpu_count=8, a clean coordinator and a clean reverse sieve.
- **Kept:** DKMS on all rigs, pinned.
- **Anchors:** FIN162:46-68, 84; 7278db6 diff.

#### E34. 2026-04-05 18:51 to ~20:23+, S162 "VICTORY" run (TCP-PWC, 26 GPUs)
- **Config:** 100K seed cap (PRFAB:224-226; TM163:122; P163:29). 26/26 GPUs, DKMS on all, baseline code (free_all_blocks still in cleanup).
- **Result:**
  - Dashboard at 25:46 elapsed showed 3,525,963 sps. rrig6600 at 1,318,582 sps (8,821 jobs), rrig6600b at 1,017,929 (8,804), rrig6600c at 1,186,336 (7,700). "Netconsole SILENT since 18:49 (95+ minutes)" (PRFAB:226-239).
  - Final: elapsed 42:36, 1,073,741,824 seeds, 887 bidirectional, "26/26 GPUs active throughout", "Netconsole: SILENT for entire run" (VIC:13-24).
- **Zeus:** `cudaErrorDevicesUnavailable` at `sieve_filter.py:291` in `Device.__exit__` on the Zeus local path. 250/10,738 chunks failed in Trial 3 `java_lcg_hybrid_reverse`. nvidia-smi showed both 3080 Ti at P8 and 0% utilisation. Candidates: P8 idle, `_localhost_semaphore=Semaphore(2)`, CuPy context (VIC:41-75).
- **Kept:** 100K became the invariant.
- **Anchors:** PRFAB:222-269; VIC:11-35, 41-75.

#### E35. 2026-04-05, PROPOSAL_FREE_ALL_BLOCKS and TB ruling
- **Hypothesis:** CuPy issue #4866. Concurrent `free_all_blocks()` from 8 workers per rig (24 globally) races when 2M chunks (~500 ms) finish together; 100K chunks (~25 ms) are staggered. The S155 256 MB pool cap (`PRNG_CUPY_POOL_LIMIT_MB=256`, `CUPY_GPU_MEMORY_LIMIT=268435456`) makes it redundant (PRFAB:31-74).
- **Options:** A, `cp.cuda.Device(0).synchronize()` before free_all_blocks; B, remove it (recommended); C, `MemoryAsyncPool` (deferred) (PRFAB:78-158).
- **TB ruling:** "Option B approved with conditions. Root-cause status remains provisional until staged 3-rig validation passes". 500K → 1M → 2M, each needing netconsole silence, no worker loss and no VM/OOM growth. Instrumentation behind `S163_MEM_DEBUG=1`, sampled every 25 chunks. `rocm-smi` external only, on a 30 s timer (PRFAB:273-406).
- **History relevant to this hypothesis:**
  - The same race was proposed on 2026-04-02 and TB rejected it as unproven (CL159orig:68).
  - PR162 Option A (gc-only, 2026-04-04) targeted free_all_blocks for a different mechanism.
  - CL162b:59 and 113 (2026-04-05 morning) say "Keep `_best_effort_gpu_cleanup()` / `free_all_blocks()` — KEEP, good hygiene".
- **Anchors:** as cited; TM163:43-51; P163:41-56.

#### E36. 2026-04-06, S163: free_all_blocks removal (03d9013 to 68ed075) and a 500K run
- **Action:**
  - 03d9013 patched `sieve_gpu_worker.py`, the "WRONG FILE".
  - 68ed075 removed free_all_blocks (default and pinned) from `_best_effort_gpu_cleanup()` in `sieve_filter.py`, added `S163_MEM_DEBUG` instrumentation and reverted `sieve_gpu_worker.py`. Deployed to Zeus and all 3 rigs.
- **Result at 500K seed cap:**
  - Trial 1: 3,140 bidirectional, 21:08 elapsed, "Netconsole: SILENT throughout", AMD rigs stable, ~20M sps.
  - Trials 2/3: "Run hung after Trial 1 completion". `window_optimizer.py` at 139% CPU / 2 GB RAM for 48+ min, then killed.
  - Zeus: `cudaErrorDevicesUnavailable` on "EVERY local chunk systematically (chunks 43, 87, 131, 175… every 44 chunks, in pairs)". Persistence mode was "already Enabled".
  - `S163_MEM_DEBUG=1` did not propagate, so there was no instrumentation output.
- **Status:** "PARTIALLY VALIDATED".
- **Caveat:** commit 526ae93 (2026-04-11) says `run_trial_persistent` "was always defaulting to ssh" regardless of `--pwc-transport tcp` (CL163:20-25). No in-era document re-examines which transport, and which worker file, the 2026-04-04 to 2026-04-06 "TCP-PWC" runs actually used. CL163 attributes the defect to the post-clone public-repo state (CL163:12-14). The diff shows the missing kwarg in `window_optimizer_integration_final.py`. **[UNVERIFIED]** when that omission was introduced.
- **Anchors:** TBS163:29-68, 138-153.

#### E37. 2026-04-06 to 04-08, S163: Zeus-side changes (not AMD faults)
- 3715b42: slot-based Zeus local GPU binding ("chunk index not safe proxy for GPU ownership under concurrent dispatch").
- 0a316b8: gated `torch.cuda.synchronize()`/`empty_cache()` in `_best_effort_gpu_cleanup()` behind `S163_MEM_DEBUG=1`. "Zeus throughput from ~466K s/s to ~13K s/s" was restored.
- **Anchors:** 3715b42 subject; CH163:23-25; 0a316b8 diff.

#### E38. 2026-04-07 to 04-10, S163: package-removal disaster
- **Rigs:** rrig6600 and rrig6600c.
- **Action:** removed multimedia packages to match rrig6600b's minimal install (cf. E7).
  - rrig6600: the removal "broke NIC — rig crash-looped on boot". Emergency reinstall of `amdgpu-lib amdgpu-lib32 amdgpu-multimedia`.
  - rrig6600c: multimedia removed, and `autoremove` also removed 24 i386 orphan libs.
- **Result:** both became "unstable — crashing during 200K seed cap runs that were previously stable". `GCVM_L2_PROTECTION_FAULT_STATUS:0xFFFFFFFF` on rrig6600 and rrig6600c during the **reverse sieve**. "`ih ring buffer overflow` cascade leading to `device lost from bus`".
- **Belief:** 0xFFFFFFFF means the "device physically dropped off PCIe bus"; not OOM. `ih ring buffer overflow` is "the precursor — interrupt handler ring saturates under concurrent 8-GPU load" (CH163 calls the GPU gfx1030).
- **Anchors:** CH163:51-76; CH163:160-163 (200K ~2.8M sps; last run Trial 1 completed with 300 bidirectional; Trials 2-3 failed with 3362/5369 chunks failed).

#### E39. 2026-04-07 to 04-10, S163: recovery attempts that failed
- Multimedia reinstall on both rigs: "did not fix".
- "Multiple reboots and power cycles": "did not fix".
- GRUB `amdgpu.gpu_recovery=1 amdgpu.sched_jobs=256`, added to all three rigs before "quiet" (2026-04-10): "did not fix (run crashed with 3362/5369 chunks failed)". Also "Not yet proven to help or hurt. May need reverting."
- Apt holds on all three: `amdgpu-dkms`, `amdgpu-install`, `linux-firmware`, `linux-generic-hwe-22.04`, `linux-headers-generic`, `linux-headers-generic-hwe-22.04`, `linux-image-generic`, `linux-image-generic-hwe-22.04`.
- **Assessment recorded:** "repeated hard crashes and power cycles today corrupted the GPU driver/firmware state … in a way that persists across reboots".
- **Kept/reverted:** reverting gpu_recovery/sched_jobs is not recorded. They were added "to all three rigs" including rrig6600b, the clone source, so whether the clone carried them is not recorded.
- **Anchors:** CH163:39-47, 78-84.

#### E40. 2026-04-10, S163: clone restore from rrig6600b's disk
- **Action:** full disk image `~/rig6600b_clone_20260410.img.gz` (126G), restored via USB-NVMe to rrig6600 and then to rrig6600c, using `gunzip -c … | sudo dd of=/dev/sda bs=4M`. Network, SSH keys, hostname and machine-id restored. Netconsole `enp10s0`→`lan0`.
- **Result:** "8/8 GPUs healthy, netconsole working" on each. All three on ROCm 6.4.3-128 with "identical grub params, identical packages". The image is kept on a USB SATA drive on SER8.
- **Recorded invariant:** "Fix forward, restore from clone when fix-forward fails" (CH163:155).
- **CONTRADICTION:** CL163:12-14 describes the cluster as destabilised "by cloning rrig6600b from the public repo (which lacked several critical private-repo changes)". CH163:88-110 describes a block-level disk clone.
- **Anchors:** CH163:88-110, 152-155; CL163:12-14, 70-94.

#### E41. 2026-04-11/12, S163: TCP-PWC restoration (transport state; affects run validity)
- 526ae93: `pwc_transport` passed to `run_trial_persistent()`, which "was always defaulting to ssh".
- a42e71c and c0615f3: v2 worker restored, plus `sock.settimeout(None)`.
- aacb0b1: `HEARTBEAT_TIMEOUT_S` 60 → 300.
- b632a26: `PWC_GPU_ID`/`PWC_WORKER_ID` exported. Before this, "All 8 workers per rig defaulted to gpu_id=0 … Only 1 worker per rig (3 total) survived".
- `daily3.json` scp'd to rrig6600b.
- **Result:** 26/26 GPUs active at 537,408 sps (rrig6600 146K, b 264K, c 125K, Zeus 1,071 sps).
- **Anchors:** CL163:18-66, 128-141; CL163F:120-121.

#### E42. 2026-04-12, S163: set_limit() in the TCP path (commit 5dd3d34)
- **Belief:** the 256 MB pool cap was enforced only in `sieve_gpu_worker.py` (SSH path). TCP-PWC workers import `sieve_filter` and never applied it, so "pool to grow unbounded across trials and triggering GPU page faults".
- **Action:** at module import in `sieve_filter.py`, `cp.get_default_memory_pool().set_limit(int(PRNG_CUPY_POOL_LIMIT_MB or 256)*1024*1024)`.
- **Result:** part of the 250K and 300K stable results (E44).
- **Kept:** yes (CL163F:178).
- **Anchors:** 5dd3d34 diff; CL163F:46-51, 97-98.
- **CONTRADICTION (hash):** CL163F:47 attributes set_limit to commit `9b3c443`. In git, 9b3c443 is the JSON-write skip and 5dd3d34 is set_limit.

#### E43. 2026-04-13, S163: JSON serialisation hang (9b3c443 / 0bfc391), not a GPU fault
- **Result:** `json.dump()` of 2.6M survivors took 2h44m and hung the process. Writes are now skipped when survivors exceed 100K. 0bfc391 also changes `distributed_config.json` `max_concurrent_script_jobs` from 0 to 8 on one node.
- **Anchors:** CL163F:53-58; 9b3c443 and 0bfc391 diffs.

#### E44. 2026-04-12/13, S163: seed-cap ladder (TCP-PWC, 26 GPUs)
- **Results:**
  - 10K: stable (~537K sps with 3/24 workers, then ~1.7M with 24/24).
  - 100K: stable (~1.7M).
  - 250K: stable "After fresh reboot" (~7-15M).
  - 300K: stable "After fresh reboot, all 26 GPUs held" (~12M).
  - 500K: "⚠️ Crashes — rrig6600 crashed — memory accumulation".
- **Belief:** "300K requires fresh rig reboot between runs. Without reboot, CuPy memory pool accumulates state across trials and triggers GPU page faults at 300K+". Learning 8 says the 0xFFFFFFFF/0x1ff/WALKER 0x7 signature indicates "complete GPU virtual address space corruption — caused by unbounded CuPy memory pool growth across trials".
- **Kept:** "Confirmed production ceiling: 300K".
- **CONTRADICTION:** 500K Trial 1 ran clean on 2026-04-06 (E36) but 500K crashes here. The attribution of the 0xFFFFFFFF signature differs between CL160:58 (PCIe all-ones / bus dropout), CH163:73 (device dropped off bus), VIC:30-33 (stock driver) and CL163F:109-112 (pool growth).
- **Anchors:** CL163F:17, 66-81, 106-112, 116-124, 174.

#### E45. 2026-04-14, S163: start-up pool flush (commit d9971d7)
- **Belief:** "CuPy pool grows across consecutive runs and triggers GPU page faults at 300K+ seed cap". A single call at import has "no concurrent workers yet, no race condition possible".
- **Action:** `cp.get_default_memory_pool().free_all_blocks()` at module import in `sieve_filter.py`, after set_limit.
- **Result:** not recorded in in-era docs.
- **Note:** this reintroduces a free_all_blocks call, once per process, after its per-chunk removal in E36.
- **Anchors:** d9971d7 diff and message; CL163F:146-148.

#### E46. 2026-04-15, S163: between-pass flush broadcast and Zeus as TCP workers (7c83861, 5bb04f6)
- **Action:**
  - 7c83861: Zeus GPUs launched as local TCP persistent workers, with auto-cleanup on shutdown.
  - 5bb04f6: `flush_workers()` (`[S163-FLUSH] Broadcast flush_pool command to all TCP workers … release accumulated CuPy pool state`) is called between Pass 1/2, 2/3 and 3/4, plus `S163_MEM_DEBUG` in `ROCM_ENV_VARS`.
  - 57eb048 and ba86d77: MEM_DEBUG propagation ("hardcoded 0 was overriding coordinator env").
- **Recorded fact (git grep, not a judgement):** at trees 5bb04f6 and 49ded74, `git grep "flush_pool\|broadcast_flush"` finds only the coordinator lines (`persistent_worker_coordinator.py:692, 698`). No definition of `broadcast_flush` or a worker-side `flush_pool` handler exists in any .py, and the call sits in a try/except that logs "flush broadcast failed".
- **Result:** not recorded. 39ff260 (2026-04-15 20:01) committed Step 1 survivors, "1069 bidirectional, W5_O53". The run's config and fault status are not recorded in the commit.
- **Anchors:** 7c83861, 5bb04f6, 57eb048, ba86d77 and 39ff260 messages and diffs.

#### E47. 2026-04-15/16, S163: snd_hda_intel power_save and amdgpu gfxoff via modprobe.d and initramfs
- **Rigs:** all three.
- **Action:**
  - `/etc/modprobe.d/snd-power.conf`: `power_save=0 power_save_controller=N`.
  - `/etc/modprobe.d/amdgpu.conf`: `gfxoff=0`.
  - `update-initramfs -u`. initrd timestamps: rrig6600 Apr 15 18:10, rrig6600b Apr 16 12:02, rrig6600c Apr 15 18:10.
  - Verified live: `snd_hda_intel power_save=0`; amdgpu gfxoff sysfs absent ("correct — disabled means no sysfs node").
- **Belief:** the stated hypothesis is not written down beyond "snd_hda_intel + amdgpu fixes". The recorded crash chains include `snd_hda_intel … Unable to change power state from D3hot to D0` on the GPU's audio function (SUB:38; CL159orig:51; CL162a:156).
- **Result:** "first time all 3 rigs have been in a verified clean state simultaneously". No run result is recorded in-era with the fixes confirmed. The next action was a 100K run with `S163_MEM_DEBUG=1` (CL163R:124-134).
- **Note:** `amdgpu.gfxoff=0` was already on the standardised GRUB cmdline from 2026-04-02 (CL159orig:117).
- **Anchors:** CL163R:50-81, 107-120; fd534de message.

#### E48. 2026-04-16 18:01, S163: 100K run, all three rigs crash
- **Config:** HEAD at start `3c20d2d` (free_all_blocks removed from `sieve_gpu_worker.py` plus MEM_DEBUG). `seed_cap_amd=100000`, 3 trials, 24 workers, TCP-PWC.
- **Result:** "All 3 rigs crashed and rebooted during the run". Workers went offline ~18:13, **rrig6600b first** (PCI enumeration in netconsole at 18:13). rrig6600c NETCONSOLE READY at 18:18, rrig6600 at 18:45. Last log entry was chunk 10729 at 18:31. Only Trial 1 completed (`W21_O68_midday+evening_S2-72_FT0.37_RT0.48`). The Zeus coordinator (PID 88761) spun at 94% CPU for 1.5+ h and was killed ~20:10. "No GPU fault signatures in kern.log (crashes happened BEFORE this boot)".
- **Belief:** "NOT a valid test", because the fixes in E47 had not been baked into initramfs on some rigs.
- **CONTRADICTIONS:**
  - (a) rrig6600b, recorded as "never crashed" (SUB:11, 132; CL160:62), crashed first here.
  - (b) CL163R's own initrd timestamps (Apr 15 18:10, Apr 16 12:02, Apr 15 18:10) are all earlier than an Apr 16 18:01 launch. The launch date is not stated explicitly, but the run followed deploying 3c20d2d (committed 2026-04-16 17:59). That is inconsistent with "update-initramfs had not yet been run on some rigs".
  - (c) 3c20d2d's MEM_DEBUG and free_all_blocks change was in `sieve_gpu_worker.py`. 49ded74 (2026-04-17) states the TCP-PWC worker `pwc_worker_service.py` "had zero memory sampling code" and that "S163_MEM_DEBUG=1 was exported … but no code read it".
- **Anchors:** CL163R:6-46, 107-120; 3c20d2d and 49ded74 messages.

#### E49. 2026-04-17, S163: MEM_DEBUG in the TCP worker (49ded74); end-of-era boundary
- **Action:** sampling every 25 jobs plus an always-on >200 MB breach log, added to `persistent/pwc_worker_service.py` (apply_s163_tcp_worker_mem_debug.py).
- **Result:** not recorded in-era.
- **Status note:** at the current worktree HEAD, `sieve_gpu_worker.py:91-92` again contains `free_all_blocks()`. The last commit touching that file is ad4aa59 [S163-KARG-HB], outside this era. `sieve_filter.py:148-150` still carries the removal comment. `docs/TODO_CONSOLIDATED.md:317` records "PWC worker only (still present there)".
- **Boundary:** `apply_s163_kernel_arg_typing.py` (docstring dated by content to April 17: "S163 FINAL and HEAD both fail today") and the other S163KARG items belong to the other agent.
- **Anchors:** 49ded74; docstring apply_s163_tcp_worker_mem_debug.py; `git log -3 -- sieve_gpu_worker.py`.

---

### LIST 1: Recorded as helping
1. **Netconsole** (instrumentation). It produced the crash captures (CL159orig:36-40; CL159:64-67).
2. **ZMQ lease expiry** (resilience). The run completed after rrig6600 rebooted (SUB:138-149).
3. **/etc/environment protective vars** on all rigs. TB gave a provisional accept, and the env check PASSed on all three (CL159:69-101; CL160:41-47). But CL160:10 says it was "insufficient", so it is not a crash fix on its own.
4. **Inter-chunk `_best_effort_gpu_cleanup()` in the ZMQ worker** (7e1a62e). A full bidirectional trial ran with zero crashes and zero netconsole events (CL160:88-93).
5. **The same cleanup in the TCP worker** (b3c3207/64d0307). The crash was delayed from ~18 s to ~50 min (PR162:176-177; CL162:55). It did not prevent the crash.
6. **amdgpu-dkms 6.12.12.60403.** On rrig6600c only it "Extended stability to 25 min" (CL162b:34). On all rigs it is recorded as a clean run (CL162b:35; FIN162:42; VIC:28-35). See the CONTRADICTION in E26.
7. **seed_cap_amd=100K** with DKMS: 95+ min clean, 3 rigs (PRFAB:266), and the 42:36 VICTORY run (VIC:13-24).
8. **free_all_blocks removal** (68ed075, sieve_filter.py) at 500K: Trial 1 clean, netconsole silent, 21:08 (TBS163:43-52). Partial; trials 2 and 3 hung.
9. **Clone restore from the rrig6600b disk image** to rrig6600 and rrig6600c: 8/8 GPUs healthy each (CH163:97, 104).
10. **set_limit() in the TCP path, the JSON-write skip, and a fresh reboot between runs**: 250K and 300K stable (CL163F:66-81, 123-124).
11. **Isolation** (fewer rigs): rrig6600c + Zeus ran clean (E19). See list 4.

### LIST 2: Recorded as not helping or making things worse
1. seed_cap 5M → 2M: still crashed (CL159orig:76-77; SUB:107).
2. `iommu=pt`: still crashed (SUB:108).
3. GRUB standardisation (full cmdline in E8): still crashed (SUB:109).
4. /etc/environment `HSA_OVERRIDE_GFX_VERSION` standardisation: still crashed (SUB:110).
5. `HSA_ENABLE_SDMA=0` and `HSA_ENABLE_RUNTIME_POWER_MGMT=0` via coordinator `--setenv` (561428b): still crashed (SUB:111-112).
6. /etc/environment SDMA/power vars: "insufficient — GPU crashes continued" (CL160:10).
7. Zeus TCP buffer sysctl and dispatch `Semaphore(8)`: "no change" / "no effect on crash" (OPTB:125-126); "unnecessary" (FIN162:32).
8. `AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3` on rrig6600c: crashed ~1-2 min into the trial, "No meaningful improvement" (OPTB:21-25, 68-71; CL162b:31; FIN162:26).
9. `HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0` on rrig6600c: "No effect" (CL162b:32) against "partial, not a fix" (FIN162:29).
10. `amdgpu.cwsr_enable=0`: "No effect" (CL162b:33) and "unnecessary" (FIN162:30). On rrig6600b it caused KIQ fence timeouts and on rrig6600 instability (CL162:84-86). CL162 also calls it "required" on rrig6600c. Contradictory.
11. Worker spawn stagger: "unnecessary" (FIN162:31). Which stagger was tested, and its run result, are not recorded.
12. `HSA_NO_SCRATCH_RECLAIM=1` on all AMD rigs: worse, crashed at 219 s (FIN162:27; a854647).
13. CuPy pool disabled (raw hipMalloc, d79e558): worse, crashed at 358 s (FIN162:28; 070199d).
14. Reverse-sieve warm-up plus VRAM instrumentation (8d91311): "worse or no effect" (FIN162:51-52); reverted.
15. Kernel pre-compile at start-up (22bb145): "made crashes worse" (ac32cd9).
16. Seed cap 1M and 500K (S162, under 3 rigs): "delayed crashes, did not fix" (FIN162:33). 1M crashed mid-run (PRFAB:265). 2M with DKMS crashed at ~1400 s (PRFAB:264).
17. Multimedia package removal: broke the rrig6600 NIC and destabilised rrig6600 and rrig6600c (CH163:51-58). The reinstall did not fix it (CH163:79).
18. Repeated reboots and power cycles: did not fix (CH163:80).
19. GRUB `amdgpu.gpu_recovery=1 amdgpu.sched_jobs=256`: did not fix; run had 3362/5369 chunks failed (CH163:81).
20. 500K seed cap (S163 ladder, 2026-04-12/13): rrig6600 crashed (CL163F:74).
21. A 100K run after 3c20d2d (2026-04-16): all three rigs crashed. Ruled "not a valid test" (CL163R:12-20); see the contradictions in E48.

### LIST 3: Proposed but not recorded as tried in-era
1. Raise seed_cap_amd from 2M to 10M (P159:14; TM159:68-71). Retired (`docs/TODO_CONSOLIDATED.md:315`).
2. linux-firmware 3.40 → 3.41 on rrig6600 and rrig6600c (SUB:120, 156, 171-177). Later held (CH163:40).
3. Remove `pci=assign-busses,hpbussize=0x33` from rrig6600 GRUB (SUB:157, 179-180).
4. `HIP_PATH=/opt/rocm/hip` in `.bashrc` on rrig6600 and rrig6600c (SUB:158).
5. Normalise rrig6600 `.bashrc` by removing the `$PS1` guard (FIN:128-129; CL159:171; P160:88; CL160:120). Pending at the end of S160.
6. PR162 Option A, gc-only cleanup in the TCP worker (PR162:285-309). `apply_s162_safe_cleanup.py` exists (135a8ba) but no run of it is recorded.
7. PR162 Option C, cap rrig6600c at 4 workers (PR162:334-346; OPTB:169-183). No run recorded.
8. PR162 Option D, a 30-60 s rrig6600c start stagger (PR162:350-362). No specific run is recorded (FIN162:31 mentions a "Worker spawn stagger" as unnecessary, without detail).
9. TB_UPDATE Q4: Option A on rrig6600c only with the 4-worker cap (OPTB:164-167, 181-183).
10. Research KFD queue limits and ROCm 6.4.3 KFD multi-process bugs (OPTB:142-145, 185-186).
11. PRFAB Option A (synchronize before free_all_blocks) and Option C (`MemoryAsyncPool`) (PRFAB:80-158).
12. PRFAB staged validation at 1M and 2M after the free_all_blocks removal (PRFAB:277-281). Only 500K is recorded (TBS163); 1M/2M not recorded in-era.
13. External `rocm-smi --showmeminfo vram` sampling every 30 s (PRFAB:295, 403-406). No result recorded.
14. Make `perf=high` persistent with a systemd service (CL162:169).
15. "Revert unnecessary workarounds from coordinator (stagger, semaphore, cleanup patches)" (FIN162:92). Not recorded as done.
16. Run at `seed_cap_amd=5000000` ("Known Working Configuration", FIN162:89, 96-99). No result recorded.
17. Zeus `cudaErrorDevicesUnavailable` Options A (`nvidia-smi -pm 1`; TBS163:63-64 found it already enabled), B (tie `_localhost_semaphore` to gpu_count) and C (context keep-alive) (VIC:67-72; TM163:53-61; P163:57-67).
18. TB Q1 test: re-run at 500K without `S163_MEM_DEBUG=1` (TBS163:98-100, 129-131).
19. A timeout on the inter-trial transition (TBS163:115-123).
20. Seed-cap ladder 350K → 500K → 750K → 1M after 200K was stable (CH163:137-138). Superseded by the CL163F ladder.
21. A TB postmortem on the package removal (CH163:146-147).
22. Pre-warm the CuPy kernel cache on the rigs (CL161:142; P162:31-34; TM163:88-90). Not done as a rig-wide cache; the 22bb145 in-process pre-compile was tried and reverted.
23. Worker-registration env-invariant check: done in S160 (E13), so it is not in this list.
24. Start-up pool flush (d9971d7) and between-pass flush (5bb04f6): code was committed, but no run result is recorded in-era. For 5bb04f6, see the git-grep fact in E46.

### LIST 4: Isolation and configuration results (every recorded combination)

| # | Date / session | Rigs (cards) | Workers | Seed cap / chunk | Kernel / pass | Transport | Driver / notable env | Outcome | Duration | Anchor |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-03-30 S159 | Zeus + rrig6600 | 8 AMD + Zeus | 2M | java_lcg Step 1 | ZMQ + systemd-run | stock | "running clean" (in progress); final not recorded | n/r | P159:5-6, 66-67; TM159:11-15 |
| 2 | 2026-04-02 S159G | "both rigs" (rrig6600, rrig6600c implied) | n/r | 5M | Step 1 | ZMQ | stock | crash, both rigs | n/r | CL159orig:76 |
| 3 | 2026-04-02 S159G | same | n/r | 2M | Step 1 | ZMQ | stock | crash | n/r | CL159orig:77; SUB:107 |
| 4 | 2026-04-02 S159G (rrig6600c capture) | rrig6600c GPUs 19:00.0, 06:00.0 | n/r | n/r (2M or 5M) | n/r | ZMQ | stock | 0xFFFFFFFF on 2 GPUs, GPU reset, down | n/r | CL159orig:39-56 |
| 5 | 2026-04-02 17:49 S159G | 3 rigs (rrig6600 crashed on 03:00.0; b, c survived) | n/r | 2M (iommu=pt test config) | java_lcg fwd done; reverse/hybrid disputed | ZMQ | stock; rrig6600 lacking SDMA vars | rrig6600 crash at uptime ~6511 s; fwd sieve completed via leases | n/r | SUB:17-43, 138-149; CL159orig:146-155; CL159:122 |
| 6 | 2026-04-03 S160 (pre-v5) | 3 rigs (+Zeus) | n/r | 2M | Step 1 | ZMQ | stock; env vars PASS | rrig6600 and rrig6600c crashed repeatedly; b never | n/r | CL160:51-64 |
| 7 | 2026-04-03 S160 v5 | all 3 AMD rigs (+Zeus) | n/r | 2M | java_lcg fwd + reverse + hybrid fwd | ZMQ | stock; cleanup per chunk | zero crashes, zero netconsole; trial 2 hybrid lease exhaustion (not a crash) | n/r (26-min cross-pass stall) | CL160:88-93, 109-111 |
| 8 | 2026-04-03 S161 Gate 1 | 1 rig, 1 GPU | 1 | 500K seeds total | sieve | TCP-PWC | stock | passed | n/r | CL161:33-35 |
| 9 | 2026-04-03/04 S161 Gate 2 | 3 rigs, 24 AMD | 24 | 50M total, 25 × 2M | sieve | TCP-PWC | stock; no TCP cleanup | zero crashes, zero netconsole | n/r | CL161:37-40 |
| 10 | 2026-04-04 S161 | 3 rigs + Zeus, 26 | 24 + 2 | 50M total | sieve | TCP-PWC | stock | clean, 2.24M sps | 53 s | CL161:42-46, 52-56 |
| 11 | S161 benchmark | 8 AMD (1 rig) | 8 | 16M total | sieve | SSH-PWC | stock | completed, ~133K sps | 2m 1.5s | CL161:55 |
| 12 | S161 benchmark | 24 AMD | 24 | 50M total | sieve | ZMQ | stock | completed, ~800K sps | ~60 s | CL161:54 |
| 13 | 2026-04-04 S162-1 | 3 rigs + Zeus, 26 | 24 + 2 | n/r (2M implied by later PRFAB:264) | Step 1 | TCP-PWC (see E36 caveat) | stock; no cleanup on rigs | rrig6600c crash, multi-GPU, hard LOCKUP | ~18 s | CL162:19-29; PRFAB:264 |
| 14 | 2026-04-04 S162-3 | rrig6600c + Zeus, 10 | 8 + 2 | n/r | full 3-trial Step 1 | TCP-PWC | stock; cleanup | clean, zero netconsole | 18m 52s (~948K sps, 1.07B seeds) | CL162:36-40; CL162a:246 |
| 15 | "historical" | rrig6600 + Zeus | n/r | n/r | n/r | n/r | n/r | "Stable (historical)" | n/r | PR162:22 |
| 16 | "historical" | rrig6600b + Zeus | n/r | n/r | n/r | n/r | n/r | "Stable (historical)" | n/r | PR162:23 |
| 17 | "S158 historical" | rrig6600 + rrig6600c + Zeus, 18 | n/r | n/r | n/r | n/r (S158 era) | n/r | "Clean" | n/r | PR162:24 |
| 18 | S155 (cited) | rrig6600c at 4 workers | 4 | n/r | n/r | n/r | n/r | "historically proven stable" (cited, not re-run) | n/r | PR162:339-346; OPTB:179 |
| 19 | 2026-04-04 S162-5 | 3 rigs + Zeus, 26 | 24 + 2 | n/r | Step 1 | TCP-PWC | stock; cleanup + semaphore + sysctl | rrig6600c 16:00.0 SQC (inst) → cascade | uptime 3036 s ("~50 min") | PR162:44-67; CL162:48-56 |
| 20 | 2026-04-04 21:52 Option B | 3 rigs + Zeus, 26 | 24 + 2 | n/r | Step 1 | TCP-PWC | stock; AMD_SERIALIZE_* on .162 | rrig6600c GPU6 + GPU1 SQC → collapse, soft lockup | ~1-2 min | OPTB:11-62 |
| 21 | 2026-04-05 Exp1 | 3 rigs | n/r | n/r | n/r | TCP-PWC | stock; SCRATCH_ASYNC_RECLAIM=0 on .162 | "No effect" / "partial" | n/r | CL162b:32; FIN162:29 |
| 22 | 2026-04-05 cwsr | rrig6600c, then rrig6600b, rrig6600 | n/r | n/r | n/r | TCP-PWC | cwsr_enable=0 | c: "no effect" / "required"; b: KIQ fence timeouts; rrig6600: instability | n/r | CL162b:33; CL162:82-92; FIN162:30 |
| 23 | 2026-04-05 | 3 rigs, DKMS on rrig6600c only | n/r | n/r (2M implied) | Step 1 | TCP-PWC | DKMS on c | "Extended stability to 25 min" / "first clean 26-GPU run … all 3 trials" | 25 min / n/r | CL162b:34; FIN162:35-42 |
| 24 | 2026-04-05 | 3 rigs, DKMS all, 26 | 24 + 2 | n/r | Step 1, 3 trials | TCP-PWC | DKMS all | Trial 1 clean (31:31, 9,921 bidir); rrig6600c faulted in reverse but recovered; trials 2-3 incomplete. Versus "All 3 trials clean, 35.6M sps" | 31:31 (T1) | CL162:94-118 vs CL162b:22-23, 70-80 |
| 25 | 2026-04-05 | 3 rigs | n/r | 2M | Step 1 | TCP-PWC | DKMS | crash | ~1400 s | PRFAB:264 |
| 26 | 2026-04-05 | 3 rigs | n/r | 1M | Step 1 | TCP-PWC | DKMS | "crashed mid-run" | n/r | PRFAB:265; FIN162:33 |
| 27 | 2026-04-05 | 3 rigs | n/r | 500K | Step 1 | TCP-PWC | DKMS (per FIN162:33 context) | "delayed crashes, did not fix" | n/r | FIN162:33 |
| 28 | 2026-04-05 | 3 rigs | n/r | n/r | n/r | TCP-PWC | DKMS; no-pool (d79e558, sieve_gpu_worker.py) | crash | 358 s | FIN162:28 |
| 29 | 2026-04-05 | 3 rigs | n/r | n/r | n/r | TCP-PWC | DKMS; HSA_NO_SCRATCH_RECLAIM=1 all | crash | 219 s | FIN162:27 |
| 30 | 2026-04-05 | 3 rigs (rrig6600c named in commit) | 8/rig | n/r | reverse transition | TCP-PWC | DKMS; kernel pre-compile (22bb145) | "made crashes worse" | n/r | ac32cd9 |
| 31 | 2026-04-05 late | rrig6600 + rrig6600c crashed (3-rig run) | n/r | n/r | forward sieve | TCP-PWC | DKMS; perf auto (.162 at 2750 MHz, others 700 MHz) | simultaneous forward crashes | n/r | CL162:120-124 |
| 32 | 2026-04-05 18:51 VICTORY | 3 rigs + Zeus, 26 | 24 + 2 | 100K | Step 1, 3 trials incl. hybrid | TCP-PWC | DKMS all, baseline (free_all_blocks present) | netconsole silent; 887 bidir; Zeus 250/10,738 chunk failures (cudaErrorDevicesUnavailable) | 42:36 | VIC:13-24, 41-64; PRFAB:224-266 |
| 33 | 2026-04-06 S163 | 3 rigs + Zeus | 24 + 2 | 500K | Step 1 | "TCP-PWC" (see E36 caveat) | DKMS; free_all_blocks removed in sieve_filter.py | T1 clean (3,140 bidir, ~20M sps, netconsole silent); T2/3 hung; Zeus near-100% local chunk failure | 21:08 (T1); hang 48+ min | TBS163:43-64 |
| 34 | 2026-04-07..10 S163 | 3 rigs (rrig6600 and rrig6600c after package removal) | n/r | 200K | reverse sieve | n/r ("TCP-PWC" default; ssh per 526ae93) | DKMS; ± gpu_recovery/sched_jobs | 0xFFFFFFFF, ih ring overflow, device lost from bus; T1 OK (300 bidir), T2-3 3362/5369 chunks failed | n/r | CH163:51-81, 160-163 |
| 35 | 2026-04-12 S163 | 3 rigs + Zeus | 3/24 AMD (worker_id collision), then 24/24 | 10K | Step 1 | TCP-PWC (fixed 526ae93) | DKMS; post-clone | stable; 537K → 1.7M sps | n/r | CL163:52-59; CL163F:70, 120-121 |
| 36 | 2026-04-12/13 | 3 rigs + Zeus, 26 | 24 + 2 | 100K | Step 1 | TCP-PWC | DKMS | stable, ~1.7M sps | n/r | CL163F:71, 122 |
| 37 | 2026-04-12/13 | 3 rigs + Zeus, 26 | 24 + 2 | 250K | Step 1 | TCP-PWC | DKMS; set_limit; fresh reboot | stable, ~7-15M sps (one 250K run had a JSON hang of 2h44m) | n/r | CL163F:56-58, 72, 123 |
| 38 | 2026-04-12/13 | 3 rigs + Zeus, 26 | 24 + 2 | 300K | Step 1 | TCP-PWC | DKMS; set_limit; fresh reboot | stable, ~12M sps, "all 26 GPUs held" | n/r | CL163F:73, 124 |
| 39 | 2026-04-12/13 | 3 rigs + Zeus | 24 + 2 | 500K | Step 1 | TCP-PWC | DKMS; set_limit | rrig6600 crashed ("memory accumulation") | n/r | CL163F:74 |
| 40 | 2026-04-12/13 (stated) | 3 rigs | n/r | ≥300K without reboot between runs | Step 1 | TCP-PWC | DKMS | "triggers GPU page faults" | n/r | CL163F:78-81, 106-107 |
| 41 | 2026-04-15 | n/r | n/r | n/r | Step 1 | n/r | n/r | 1069 bidir committed (W5_O53); faults n/r | n/r | 39ff260 |
| 42 | 2026-04-16 18:01 | 3 rigs + Zeus, 26 online | 24 | 100K, 3 trials | Step 1 | TCP-PWC | DKMS; 3c20d2d; snd/gfxoff modprobe (initramfs state disputed) | all 3 rigs crashed and rebooted (rrig6600b first ~18:13); only T1 completed; coordinator spun 1.5 h | ~12 min to first loss; last chunk at 18:31 | CL163R:12-46 |

n/r = not recorded.

### Files read
- **Read in full:** docs/S159_CHAT_PROMPT.md; docs/TB_SUBMISSION_S159G_RIG6600_CRASHES.md; docs/TB_UPDATE_S159G_ENV_PROPAGATION.md; docs/TB_FINAL_UPDATE_S159G_ROOT_CAUSE_CONFIRMED.md; docs/SESSION_CHANGELOG_20260402_S159G.md; docs/S160_CHAT_PROMPT.md; docs/SESSION_CHANGELOG_S160.md; docs/SESSION_CHANGELOG_S161.md; docs/PROPOSAL_S162_RRIG6600C_CRASH_ROOT_CAUSE_v1_0.md; docs/SESSION_CHANGELOG_S162.md; docs/SESSION_CHANGELOG_S162_VICTORY.md; docs/SESSION_CHANGELOG_S162_FINAL.md; docs/TB_UPDATE_S162_OPTION_B_RESULTS.md; docs/PROPOSAL_FREE_ALL_BLOCKS_REPLACEMENT_v1_0.md; docs/TODO_MASTER_S163.md; docs/S163_CHAT_PROMPT.md; docs/TB_SUMMARY_S163.md.
- **Repo root, read in full:** S162_CHAT_PROMPT.md; TODO_MASTER_S159.md; CHANGELOG_S163.md; SESSION_CHANGELOG_S163.md; SESSION_CHANGELOG_S163_FINAL.md; SESSION_CHANGELOG_S163_RIG_FIXES.md.
- **From git history, read in full (deleted or rewritten docs):** `bb65aa9:docs/SESSION_CHANGELOG_20260402_S159G.md`; `135a8ba:docs/SESSION_CHANGELOG_20260404_S162.md`; `89c1512:docs/SESSION_CHANGELOG_20260405_S162.md`.
- **Commits, message + `--stat` (+ .py diff where relevant):** 561428b bb65aa9 9e09af7 135a8ba 716e641 6432a48 2fa4851 606f94f 22bb145 a854647 ac32cd9 a5e26cc 68ed075 03d9013 526ae93 d9971d7 3c20d2d. Additional: 64d0307 d79e558 070199d 8d91311 7278db6 89c1512 47ab865 f837f21 b7f5e1a 7e1a62e 6681d4b 5c61e39 0653f3f 59bfbe6 aa48ee4 0b1b516 0250e06 a26ce45 fd534de 0bfc391 9b3c443 5dd3d34 5bb04f6 7c83861 57eb048 ba86d77 49ded74 0a316b8 3715b42 39ff260 c4d4e9a. Also the full `git log` 2026-03-29..2026-04-18.
- **Module docstrings only:** apply_s160_worker_env_check.py; all 21 apply_s161_*.py; apply_s162_dispatch_semaphore.py, apply_s162_exp1_scratch_reclaim.py, apply_s162_option_b.py, apply_s162_safe_cleanup.py; apply_s163_integration_json_guard.py, apply_s163_kernel_arg_typing.py (boundary), apply_s163_mem_debug_worker.py, apply_s163_tcp_worker_mem_debug.py. No apply_s159* or verify_s159*/verify_s16[0-3]* files exist in the repo root.
- **Grep/line checks only:** docs/PROJECT_FILE_CATALOG.md:128-133; docs/TODO_CONSOLIDATED.md:310-320; current `sieve_gpu_worker.py:91-92` and `sieve_filter.py:92,148-150`.

### Could not read or did not read
- Netconsole logs, kern.log, the WATCHER/run logs cited (`logs/zmq_*.log`, `logs/s16*_*.log`, `logs/netconsole_all_rigs.log`, `netconsole_listener.log`). They are not in the repo, and reading them would need host access.
- Rig-side files (GRUB, /etc/environment, modprobe.d, apt pins, .bashrc). No SSH by instruction.
- ser8 archives and the private repo. TODO_CONSOLIDATED:277 notes ~20 changelogs are ser8-only, so this era's record is likely incomplete. S159A to S159F changelogs and a SESSION_CHANGELOG_20260330_S159 are not in the repo (TM159:99).
- S163KARG files (SESSION_CHANGELOG_20260418_S163KARG_FORENSIC.md, SESSION_CHANGELOG_20260419_S163KARG_P2.md, TB_Incident_Report_rrig6600c_S163KARG.md, TODO_MASTER_S163KARG.md): skipped by instruction.
- Full diffs of 57eb048, ba86d77, 7c83861 and 49ded74 beyond their messages and the grep checks noted.

---

## Part C — 2026-04-17 → 2026-05-12 (S163KARG–S174: device lost, DPM/clock, cwsr, pool × chunk map, backend switch)


This extract is read-only. Every statement carries an anchor. "Not recorded" means that no file I read records it. I draw no root-cause conclusions.

Null results come first. The rigs never ran clean at pool=8 with chunks above 25k: 0 of 6 recorded runs did (entries 46, 48 and 50). The lead-handling skill was not loaded: no lead was closed and no next run was chosen.

### File key (all paths relative to /home/michael/tfm_worktrees/gpu-fault-history)
| Key | Path |
|---|---|
| IR | docs/TB_Incident_Report_rrig6600c_S163KARG.md |
| FOR | docs/SESSION_CHANGELOG_20260418_S163KARG_FORENSIC.md |
| P2 | docs/SESSION_CHANGELOG_20260419_S163KARG_P2.md |
| TK | docs/TODO_MASTER_S163KARG.md |
| S164 | docs/SESSION_CHANGELOG_20260419_S164.md |
| S164P | docs/S164_CHAT_PROMPT.md |
| S165 | docs/SESSION_CHANGELOG_20260420_S165.md |
| S166 | docs/SESSION_CHANGELOG_20260422_S166.md |
| S167 | docs/SESSION_CHANGELOG_20260424_S167_S168_S169.md |
| S170 | docs/SESSION_CHANGELOG_20260425_S170.md |
| S170P | docs/S170_CHAT_PROMPT.md |
| T170 | docs/TODO_MASTER_S170.md |
| RT | S170_RESULTS_TABLE.md (repo root) |
| ROCm | docs/ROCm_Saturation_Report_S172.md |
| S174a | SESSION_CHANGELOG_20260507_S174.md (repo root) |
| D1 | docs/SESSION_CHANGELOG_20260509_S174_D1.md |
| PROP | docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md |
| RF | SESSION_CHANGELOG_S163_RIG_FIXES.md (repo root) |
| S163F | SESSION_CHANGELOG_S163_FINAL.md (repo root) |
| C163 | CHANGELOG_S163.md (repo root) |
| KARGPY | apply_s163_kernel_arg_typing.py (module docstring) |
| L174 | launch_s174.py |
| REPRO | launch_tb_step1_ft073_repro.sh |
| T:<file> | file as it exists in commit 23919ad (TEMP, 2026-05-03), read with `git show 23919ad:<file>`. These are the S170–S173 launch scripts, which are not in the working tree. |

---

### Part 0 — Pre-scope context (S162/S163, 2026-04-05 to 2026-04-16). Included because later documents in scope cite it.

**P0-1. 2026-04-07..10, S163 (rrig6600, rrig6600c).**
- **Belief:** removing packages corrupted driver or firmware state (C163:83-84).
- **Event:** `GCVM_L2_PROTECTION_FAULT_STATUS:0xFFFFFFFF` during the reverse sieve, then `ih ring buffer overflow` and `device lost from bus` at 200K seed cap (C163:65-66, 72-76).
- **Tried:**
  - Reinstalled the multimedia packages: did not fix (C163:79).
  - Reboots and power cycles: did not fix (C163:80).
  - Added GRUB `amdgpu.gpu_recovery=1 amdgpu.sched_jobs=256` on 2026-04-10, labelled "Not yet proven to help or hurt" (C163:42-47). It is also recorded as "did not fix (run crashed with 3362/5369 chunks failed)" (C163:81).
- **Fix:** restored a full disk clone of rrig6600b onto rrig6600 and rrig6600c. Result: 8/8 GPUs on each (C163:88-104).
- **Package state:** packages were held on all rigs (C163:39-40).

**P0-2. 2026-04-12/13, S163 FINAL (all rigs).**
- **Action:** added `set_limit()` (the 256MB `PRNG_CUPY_POOL_LIMIT_MB` cap) to `sieve_filter.py` for the TCP-PWC path, commit 9b3c443. The cause was recorded as "pool to grow unbounded across trials and triggering GPU page faults" (S163F:46-51).
- **Seed-cap ladder:** 10K, 100K, 250K and 300K stable; 500K crashed rrig6600 with "memory accumulation"; 300K needs a fresh reboot (S163F:66-81).
- **Signature belief:** 0xFFFFFFFF + client 0x1ff + WALKER_ERROR 0x7 was attributed to unbounded CuPy pool growth (S163F:108-112).
- **Proposed:** a startup pool flush (S163F:146-148).

**P0-3. 2026-04-14 and 2026-04-15, git only.**
- d9971d7 added a startup `free_all_blocks()` to `sieve_filter.py` ("eliminates rig reboot requirement").
- 5bb04f6 added a between-pass flush, Zeus persistent workers, auto-cleanup, and `S163_MEM_DEBUG` in `ROCM_ENV_VARS` (`git log -S"S163-FLUSH"`).

**P0-4. 2026-04-16 (all rigs).**
- **Removal:** 3c20d2d removed `free_all_blocks()` from `_best_effort_gpu_cleanup()` in `sieve_gpu_worker.py`. This is the TB "Option B": "S155 256MB pool cap makes it redundant; race condition source" (apply_s163_mem_debug_worker.py docstring :10-11; commit 3c20d2d).
- **Run, 18:01:** 100K seed cap, 3 trials. All 3 rigs crashed and rebooted (RF:12-16). The run was ruled not a valid test because initramfs had not been rebuilt (RF:19-20, 42-46).
- **Then verified on all 3 rigs:**
  - `/etc/modprobe.d/snd-power.conf` contains `power_save=0 power_save_controller=N` (RF:52-81).
  - `/etc/modprobe.d/amdgpu.conf` contains `gfxoff=0` (RF:52-81).
  - initramfs was rebuilt (RF:52-81).
  - Commit fd534de.

**P0-5. KARG hypothesis (patch committed in d32c1d1 on 2026-04-18 10:08; the 2026-04-17 run already had it "active", IR:27).**
- **Belief:** bare Python ints marshal as int64 against an int32 kernel signature. That misaligns the arguments and could cause out-of-bounds reads producing `0xFFFFFFFF`, client 0x1ff and `WALKER_ERROR 0x7` (KARGPY docstring :7-17). The docstring calls it a "latent bug with buffer-placement-dependent expression" (KARGPY :19-31).
- **Action:** wrap `cp.int32()` around `n_seeds`, `k`, `skip_min` and `skip_max` in the non-hybrid `run_sieve()` in `sieve_filter.py` (KARGPY :33-49; git show d32c1d1).

**P0-6. S162 (pre-scope) items cited later.**
- **Option B:** "Option B" was `AMD_SERIALIZE_KERNEL=3` (docs/PROPOSAL_S162_RRIG6600C_CRASH_ROOT_CAUSE_v1_0.md:313). It was ruled out as a fix (docs/TB_UPDATE_S162_OPTION_B_RESULTS.md:66-70, 112-113, 124). During that run `free_all_blocks()` stayed active (same file :15).
- **Kernel pre-compile warmup:** added in 22bb145 and reverted in ac32cd9 as "made crashes worse" (git log of sieve_gpu_worker.py).
- **No-pool raw hipMalloc diagnostic:** d79e558, reverted in 070199d with "pool NOT the cause, made things worse" (git log).

---

### Part 1 — ENTRIES (in date order)

#### E1. 2026-04-17 20:44 — S163-KARG validation run launched
- **Rigs and cards:** 8 workers each on rrig6600, rrig6600b and rrig6600c, plus Zeus (IR:29).
- **Belief:** KARG plus the JSON guard would hold at 100K (IR:27).
- **Action:**
  - Config: `seed_cap_amd=100,000`, `window_trials=5`, `min_workers=24` (IR:24-26).
  - Zeus compute mode was set to DEFAULT manually (IR:28).
  - Log: `logs/s163_karg_json_100k_5trials_2044.log` (IR:30).
- **Result:** Step 1 completed with score 1.0000, but two GPU crashes happened before completion (FOR:12). See E2 and E3.
- **Kept or reverted:** n/a.

#### E2. 2026-04-17 20:56:11 — rrig6600b GPU1 silent worker death
- **Rig and card:** rrig6600b GPU1 (IR:14, 68-77).
- **Trial:** 2, W17_O25 midday+evening, S3-174 (IR:43; FOR:16).
- **Belief at the time:** cause unknown (IR:167; FOR:19). Later, the GDM/Xorg competition for GPU1 became "the leading explanation" (IR:240).
- **Event:**
  - The worker died mid-job on sieve_1447. Netconsole shows no GCVM fault (IR:72; FOR:17).
  - It recovered on its own at 21:06:20 and resumed at 21:06:42 (IR:74-77).
  - rocm_smi at 21:16 read fan 0%, 19W, 0% utilisation (IR:79).
- **Diagnostic proposed:** grep the worker log (IR:199-204; FOR:237-239). Its result is not recorded.
- **Kept or reverted:** n/a.

#### E3. 2026-04-17 21:16:45 — rrig6600c fatal GCVM cascade and device lost from bus
- **Rig and cards:** rrig6600c, PCI 0000:06:00.0 (pid 145352), 0000:16:00.0 (pid 145822) and 0000:19:00.0 (pid 145917), all vmid 8 (IR:87-91).
- **Trial:** 3, W10_O61 midday, S1-243 (IR:44, 48).
- **Belief:** a GPUVM PTE was invalidated mid-kernel. The leading hypothesis is CuPy memory lifecycle in a shared HIP context, and the cause is "narrowed, not identified" (IR:158-162; FOR:45-50).
- **Event, Stage 1:**
  - 16:00.0: STATUS=0x00801231, SQC(inst), WALKER_ERROR=0x0, PERMISSION_FAULTS=0x3, address 0x0000790de5615000.
  - 06:00.0: STATUS=0x80081071, TCP, PERMISSION_FAULTS=0x7, address 0x7d322000.
  - Sources: IR:57-58, 99-106.
- **Event, Stage 2:** 19:00.0 went to STATUS=0xFFFFFFFF with client 0x1ff, WALKER_ERROR=0x7, PERMISSION_FAULTS=0xf and MAPPING_ERROR=0x1 (IR:59). The same address faulted repeatedly, read as a "kernel retry loop" (IR:60, 113-115).
- **Event, end:**
  - qcm fence timeouts at 21:16:54 and 21:16:56 (IR:62-63).
  - "device lost from bus" on 06:00.0 and 16:00.0 at 21:16:56.176 (IR:64).
  - Manual reboot (IR:66).
- **Kept or reverted:** n/a.
- **Contradictions:**
  1. Idle or active at the fault:
     - IR:54-56 says all 8 workers finished their last jobs at 21:11:51 and "enter idle wait". It records a "~4m54s gap. Workers idle, awaiting next chunk dispatch" before the 21:16:45 fault.
     - IR:131 says the opposite: "mid-active-compute … not during an inter-trial idle window". IR:183 and IR:137 agree with IR:131, and so does FOR:38 ("MID-COMPUTE").
  2. PERMISSION_FAULTS decode:
     - FOR:30, 32 and 40 read it as "R+X denied", "R+W+X denied" and "permissions stripped".
     - IR:97 is the TB correction: bit0 means PTE not valid, so the PTE became invalid.
  3. VMID:
     - FOR:39 says the GPUs "share vmid:8 — shared virtual address space context".
     - IR:111, 166-168 and 192 leave this as an open unknown.

#### E4. 2026-04-17/18 — Hypotheses recorded as ruled out (S163-KARG)
- **Rig:** rrig6600c, the incident in E3.
- **Ruled out:** see IR:173-184:
  - KARG mismatch: different fault character, not formally closed.
  - Idle power management, GFXOFF and D3hot: "perf=high, full SCLK".
  - Thermal.
  - ROCm or DKMS regression: amdgpu-dkms 6.12.12 pinned since April 5.
  - Kernel version change: 6.8.0-107 on all rigs.
  - CuPy `free_all_blocks()` concurrent call: "Removed in S162 Option B".
  - Inter-trial idle trigger.
  - Zeus Exclusive_Process.
- **Recorded limit:** the power and thermal evidence in §5 contains rocm_smi only for rrig6600 and rrig6600b. No rrig6600c reading is listed (IR:119-127).
- **Contradictions:**
  1. IR:182 calls the `free_all_blocks()` removal "S162 Option B". S162 Option B was `AMD_SERIALIZE_KERNEL=3`, with `free_all_blocks()` still active (P0-6). The removal was the 2026-04-16 S163 Option B (P0-4).
  2. TK:73-76, written 2026-04-19, lists "remove `free_all_blocks()` from `_best_effort_gpu_cleanup()`" as not yet implemented. T170:114 lists it as deferred.
  3. Git state: commit ad4aa59 (2026-04-18 09:14) put `cp.get_default_memory_pool().free_all_blocks()` and the pinned-pool `free_all_blocks()` back into `sieve_gpu_worker._best_effort_gpu_cleanup()`, and they are in the current tree (sieve_gpu_worker.py:91-92). `sieve_filter.py:148-150`, the TCP path import, records it removed. See E8.
  4. The "power management ruled out" conclusion (IR:129-131) was drawn at SCLK 2630–2740 MHz. S164 later lists "2750MHz clock above mining-proven stable point" as a contributing factor (S164:47-48), and S165 attributes the crashes to clock and voltage settings (E16, E18).
- **Kept or reverted:** n/a.

#### E5. 2026-04-18 — Ruling: KARG fix retained
- **Rigs:** all.
- **Belief:** KARG "appears distinct" from the new crash and is "not yet formally closed". Formal closure needs a controlled A/B test (IR:150-154).
- **Action:** the owner ruled to keep the KARG fix ("At 200K–300K seed cap more rigs would crash without it") (IR:156, 291).
- **Result:** the controlled A/B was not run; its result is not recorded.
- **Kept or reverted:** kept. It was committed in d32c1d1 and is present in the current tree (`git show d32c1d1 -- sieve_filter.py`).

#### E6. 2026-04-18 — GDM/Xorg finding and display-stack remediation (all 3 rigs)
- **Findings:**
  - GDM3 was the default display manager on all rigs (IR:235).
  - An Xorg crash dump on rrig6600b is timestamped Apr 17 21:19 (IR:239).
  - rrig6600b showed SDMA ring timeouts after reboot at 82s, 172s and 278s (IR:241).
  - The kernel cmdline `amdgpu.gfxoff=0` is ignored by amdgpu-dkms 6.12.12 ("unknown parameter 'gfxoff' ignored"). Only modprobe.d takes effect (IR:242).
- **Owner hypothesis:** GDM seat-management probes remap GPU virtual address space (IR:244-246).
- **TB verdict:**
  - rrig6600b: the display stack is a "first-tier suspect" (IR:250).
  - rrig6600c: GDM is a confounder, not the primary cause, because the faults come from compute-side clients on non-zero VMIDs (IR:252-256).
- **Action:**
  - `sudo systemctl set-default multi-user.target` and gdm3 disabled on all rigs (IR:260-261).
  - `update-initramfs -u` and reboot with amdgpu.conf and snd-power.conf active (IR:262).
  - Recorded result: "NO XORG, no SDMA timeouts post-reboot" (IR:261).
- **Planned discriminator:** a 100K × 3-trial run without GDM (IR:264-268). That run's result is not recorded under that name. See E10.
- **Kept or reverted:** kept. "GDM disabled + multi-user.target permanent" is restated at P2:93, S164P:14, S170P:16 and S166:154-156.

#### E7. 2026-04-18 — Forensic tooling (crash_forensic_daemon v2 on ser8; worker heartbeats)
- **Action:**
  - Daemon captures added: pipeline_tail, capture_context.json, worker heartbeats, ps/pstree topology and /proc details (FOR:79-107; IR:206-210).
  - `pwc_worker_service.py` heartbeat at 10 states (FOR:109-143).
- **Belief:** the topology capture would resolve the process-isolation and VMID unknowns (FOR:94-96).
- **Result:** the isolation-model check (IR:212-217) has no recorded result.
- **Later bugs recorded:**
  - "3 bugs (startup false-DOWN, log-caching, dmesg empty)" (P2:153; TK:44-47).
  - The crash monitor window was removed in S166 because of false-DOWN (S166:72-77).
  - A stale-log-pointer bug was "confirmed in production" in S174 (D1:36).
- **Kept or reverted:** the heartbeat was committed in ad4aa59. The daemon is not in the repo (FOR:194).
- **Contradiction:** IR:210 and IR:293 say "committed `ad4aa59`, dual-pushed". FOR:157-163 says "NOT YET COMMITTED". Git shows ad4aa59 dated 2026-04-18 09:14.

#### E8. 2026-04-18 09:14 — Git-only observation: commit ad4aa59 diff (no document records this)
- **What the diff shows** (`git show ad4aa59`):
  - `sieve_filter.py`: removed the `[S163-FLUSH]` startup `free_all_blocks()` that d9971d7 added.
  - `sieve_gpu_worker.py`: removed the "[S163] free_all_blocks() removed — TB-approved Option B" stub and restored the `free_all_blocks()` calls on both the default and pinned pools. It also removed the S163 MEM_DEBUG sampling block.
  - `persistent_worker_coordinator.py`: removed `[S163-FLUSH]` between-pass flushes, `[S163-ZEUS]` local TCP workers, `[S163-AUTOCLEAN]`, and `S163_MEM_DEBUG` propagation.
- **Later commits:** d32c1d1, 54 minutes later, re-added only the KARG typing in `sieve_filter.py` and the JSON guard (`git show d32c1d1 --stat`; `git log -- sieve_gpu_worker.py` shows no later commit to that file).
- **Belief, result, and kept or reverted:** not recorded. No changelog that I read mentions this diff.

#### E9. 2026-04-19 — DPM pin service (all 3 rigs, all 24 GPUs)
- **Action:** `amdgpu-dpm-pin.service` set `manual` DPM, sclk state 1 and mclk state 3 at boot, with `ExecStartPre=/bin/sleep 10`. `gpu-enum-heal.service` was removed, and the `rocm-perf-auto` cron was removed because it "was overriding DPM pin" (P2:87-93; TK:12-14).
- **Belief:** "DPM pin service + GFXOFF=0 + snd-power.conf = stable 26 GPU operation when hardware is healthy" (P2:132).
- **Result:** not separately recorded.
- **Kept or reverted:** in S164 it was repurposed to apply the Kaspa profile (S164:77-82). In S165 it was disabled on all 3 rigs (S165:36-37, 196-198; c5832bf).
- **Contradiction:** TK:11 records the "Root cause S163 GPU crashes" as the missing amdgpu.conf (gfxoff=0) and snd-power.conf on rrig6600 and rrig6600c. That differs from the "narrowed, not identified" assessment in IR:158 and the CuPy-pool attribution in S163F:109-112.

#### E10. 2026-04-19 — S163-KARG Part 2 production runs
- **Runs:**
  - `s163_karg_production_1433`: seed 0, 5 trials, 26 GPUs, 0 bidirectional survivors, wrong seed range (P2:119).
  - `s163_karg_production2_1508`: seed start 1,073,741,824, 5 trials, "GPUs 10-26". "rrig6600/c crashed Trial 3" and 2,851 survivors (P2:120).
- **Not recorded:** crash signature, card indices, and the seed cap (TK:127 says "seed_cap_amd 100K (stable)").
- **Kept or reverted:** n/a.

#### E11. 2026-04-19 — rrig6600 card0 PCIe link anomaly
- **Rig and card:** rrig6600 card0, 0000:03:00.0.
- **Observation:** `current_link_speed: Unknown 63`. The other GPUs show "5.0 GT/s PCIe (correct for Gen1 riser)" (P2:102-104).
- **Belief:** a bad riser cable or slot. It is called an "intermittent crash source" (P2:105, 131).
- **Action:** a physical reseat, pending (P2:106, 150; S164P:22-27).
- **Result in S164:** "rrig6600 card0 PCIe riser reseat — 16.0 GT/s Gen4 confirmed healthy" (S164:14).
- **Kept or reverted:** n/a (physical).
- **Contradictions:**
  1. S164P:26 expected "5.0 GT/s PCIe (not 'Unknown 63')", and P2:104 says 5.0 GT/s is correct for the risers. S164:14 instead reports 16.0 GT/s as healthy.
  2. P2:104 labels 5.0 GT/s as "Gen1". (Labels quoted as recorded; I did not re-verify them.)
  3. S170:298 later still lists "PCIe Gen1 interconnect saturation" as a candidate.
  4. S170:11 and S170:329 say "Hardware is not the root cause" and "hypothesis is exhausted", against P2:131.

#### E12. 2026-04-19 — Worker reconnect window (workaround for rig reboots)
- **Action:** in `persistent/pwc_worker_service.py`, `RECONNECT_DELAY_S` went from 2.0 to 5.0 and `_MAX_REFUSED` from 5 to 60, so the retry window grew from 10s to about 5 min (P2:71-76; commit 884b424).
- **Belief:** workers gave up before a rig finished rebooting (~60-90s) (P2:130; 884b424 body).
- **Kept or reverted:** kept.
- **Side effect recorded later:** after a coordinator SIGKILL, 26 orphan workers survived about 2 h. "60-retry × 5-second cycle = ~5 minute timeout" (S170:181-190).

#### E13. 2026-04-19 — S164 crash: KFD queue preemption failure
- **Rig and card:** rrig6600 card7, 0000:19:00.0 (S164:31).
- **Trial:** 1 of the S164 production run, at 17:39:22 (S164:31-32).
- **Timeline:**
  - KFD preemption failed, doorbell 0x80004008, ×2. 14 vital buffers were evicted.
  - SMU 0xFFFFFFFF; ring buffer out of bounds; PSP suspend -22; MODE1 reset failed (-121).
  - SDMA0 timeout; sdma0 reset test -110.
  - qcm fence timeout, "cp might be in unrecoverable state".
  - 10 kworkers blocked for more than 122s, with a dm_suspend hang.
  - Recovered in 3.5 min with no reboot.
  - Source: S164:35-43, 53-54.
- **Belief recorded as root cause:** a KFD-initiated reset. Contributing factors: 2750 MHz is above the "mining-proven stable" clock, Display Core is active during reset, HDMI audio causes the PSP suspend failure, and `lockup_timeout` is -1 (S164:46-50, 207-209).
- **Seed cap and config:** not recorded. The S164P:49 template uses `seed_cap_amd 100000`, 5 trials and `min_workers` 24.
- **Kept or reverted:** n/a.

#### E14. 2026-04-19 — S164 driver and boot parameters (all 3 rigs)
- **Action:**
  - `/etc/modprobe.d/amdgpu.conf` = `options amdgpu gfxoff=0 audio=0` (S164:62-67).
  - GRUB added `amdgpu.dc=0`, `amdgpu.lockup_timeout=30000` and `amdgpu.gpu_recovery=1`, and removed `amdgpu.sched_jobs=256` ("kernel showed 32") (S164:69-75).
  - All verified (S164:84-95).
- **Belief:**
  - `audio=0` removes the "failed to suspend display audio" cascade.
  - `dc=0` removes the dm_suspend hang.
  - 30s for long HIP kernels.
  - Source: S164:66-67, 71-73.
- **Result:** not isolated. The next session attributes crashes to Kaspa instead (E16).
- **Kept or reverted:** no revert is recorded in era docs. lockup_timeout=30000 is still referenced: the miner sub-stripe count is "sized to fit lockup_timeout=30000" (agent_manifests/window_optimizer.json:233).
- **Note:** `gpu_recovery=1` was recorded in S163 as "did not fix" (C163:81).

#### E15. 2026-04-19 — Zeus compute mode fix (Zeus 3080 Ti)
- **Action:** `nvidia-gpu-policy.service` changed from EXCLUSIVE_PROCESS to DEFAULT. It had been overriding `nvidia-compute-mode.service` (S164:99-114).
- **Related:** IR:184 lists Zeus Exclusive_Process as fixed before the 2026-04-17 run.
- **Kept or reverted:** kept. A later host read shows this unit failing on VM 101 for lack of GPU 1 (docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md:1664-1669; outside the era).

#### E16. 2026-04-19 — Kaspa kHeavyHash OC profile (all 24 AMD GPUs)
- **Action:** through the DPM pin service at boot: SCLK 2250 MHz (state 1 custom OD), VDD offset -150 mV (~850 mV), MCLK 1000 MHz, DPM manual (S164:77-82, 91-92).
- **Belief:** the sieve resembles kHeavyHash (compute-bound), and the earlier crash at 2750 MHz on stock voltage was about 500 MHz above a stable clock (S164:128-140).
- **Result (S165):** "Kaspa DPM Profile (2250MHz/-150mV) Was Causing All Crashes". The 100W PwrCap card variants on rrig6600b failed first. Card 0000:09:00.0 on rrig6600b had repeated unrecoverable ASIC reset failures (-22) needing several power cycles (S165:22-34). c5832bf: "caused ASIC reset failures".
- **Kept or reverted:** reverted in S165 to stock `high`, with the dpm-pin service disabled (S165:36-37; c5832bf).

#### E17. 2026-04-20 — S165 KFD debug-trap NULL dereference (all 3 rigs at once)
- **Event:** `kq_acquire_packet_buffer` → `pm_send_set_resources` → `kfd_dbg_trap_activate` → NULL dereference at 0x30, during a concurrent GPU reset plus worker init. Workers blocked for more than 368s, and the rigs were force-rebooted (S165:41-56).
- **Action:** `HSA_ENABLE_DEBUG_TRAP=0` added to `ROCM_ENV_VARS` (S165:58-60; c5832bf). It is in the current code at persistent_worker_coordinator.py:142-145 and in the TCP launch export at :387.
- **Result:** the clean 2-trial run in E18 had this set (S165:168).
- **Kept or reverted:** kept.

#### E18. 2026-04-20 — S165 stock-DPM validation run
- **Config:** 2 trials, `seed_cap_amd=100000`, stock `high` DPM, `HSA_ENABLE_DEBUG_TRAP=0` (S165:168).
- **Result:** 17,721 bidirectional survivors, ~14 min, 26/26 GPUs, netconsole silent (S165:170-178).
- **Live clocks under load:**
  - rrig6600: SCLK 2639–2755, MCLK 1000, PwrCap 135–145W.
  - rrig6600b: 2585–2740 / 875 / 100W.
  - rrig6600c: 2595–2750 / 875 / 100W.
  - Source: S165:128-134.
- **Recorded belief:** stock `high` is stable for short runs, "but dynamic frequency scaling between trials causes instability in longer runs (5-50 trials). `manual` DPM … is required for production" (S165:111-113). The evidence for the long-run claim is not recorded.
- **Kept or reverted:** stock `high` stayed until S170 (E31).

#### E19. 2026-04-20 — Proposed DPM harness and absolute-voltage profile (never built)
- **Proposal:**
  - rrig6600: 2200 MHz, 900 mV absolute, 950 MHz mem.
  - rrig6600b and rrig6600c: 2100 MHz, 900 mV, 875 MHz.
  - All manual.
  - Harness: 30 min whole-rig, netconsole zero faults, rrig6600b first.
  - Source: S165:115-147.
- **Status:** never built.
  - Carried as pending at S166:180, S167:209 and S170P:105-138.
  - T170:48-58 marks it P1, "requires TB proposal".
  - S170:351: "deferred — DPM lock service deployed at default peak".

#### E20. 2026-04-20 — Sudoers lost on hard crash reboots (all 4 nodes)
- **Finding:** `/etc/sudoers.d/michael-cluster` was wiped by hard crash-forced reboots because the filesystem did not sync (S165:149-156).
- **Workaround:** always run `sudo sync` after writing (S165:155).
- **Kept or reverted:** kept as a practice.

#### E21. 2026-04-22 — S166: cwsr_enable=0 mcbp=0 on rrig6600 only
- **Action:** added `cwsr_enable=0 mcbp=0` to `/etc/modprobe.d/amdgpu.conf` on rrig6600, rebuilt initramfs, and rebooted (S166:79-85).
- **Belief:** it is the "community-validated fix for `queue preemption failed` → `SMU 0xFFFFFFFF` → ring buffer corruption". It was tried in S162 with mixed results on the stock driver and is being retried on dkms 6.12.12 (S166:87-90).
- **Result:**
  - rrig6600 ran 75+ min with no GPU crash; the previous longest was about 40 min (S166:92-93).
  - Throughput on rrig6600 was about 22K s/s, against 95K+ on the other rigs (S166:95-97).
- **History notes:** in S162, cwsr_enable=0 broke rrig6600b with KIQ fence timeouts; rrig6600c "had it previously on stock driver" (S166:99-102).
- **Planned test:** two sequential runs without a reboot (S166:178-179). Its result is not recorded.
- **Kept or reverted:**
  - Kept on rrig6600 (S167:178; S170P:12; RT:49; T170:121).
  - In S174, rrig6600 and rrig6600c, both with cwsr_enable=0, crashed on reboot. The proposal was to revert (D1:43-45, 76). Whether it was reverted is not recorded.

#### E22. 2026-04-22 — S166 run table (crashes not attributed to a rig)
- **Runs:** s166_run1 to run5. run2 (W17_O36) "Complete, card1 crash" (S166:136-142). The rig, signature and seed cap are not recorded.
- **Zeus OOM:** run5 W2_O30 produced 21.9M survivors and was SIGKILLed at -9 after using 60GB of RAM on the Zeus host. This was a host OOM, not a GPU fault (S166:126-130). The fix was to clear `accumulator['bidirectional']` after each flush (S166:60-70).

#### E23. 2026-04-24 — S167: warm-start leak named as a rig-hammering contributor
- **Belief:** DB-selected historical configs forced a first-trial workload that was "hammering AMD rigs on trial 0" (S167:24-26). The S166 warm-start mixing "caused rig crashes by changing first-trial workload character" (S167:195).
- **Action:** removed the S140b `get_best_step1_params()` injection from `agents/watcher_agent.py` (S167:32-37). TB rejected the S166 commits and approved S167 (S167:48-51).
- **Kept or reverted:** kept (T170:123-125).

#### E24. 2026-04-24 — S168A-DIAG: startup-hammer telemetry (rrig6600c)
- **Context:** rrig6600c kept crashing with a "gfxhub page fault + SQC (inst)" signature. "Initial theories (hardware, cwsr, initramfs) were all incorrect" (S167:57-58).
- **Action:** `PRNG_PWC_STARTUP_DIAG=1` passive logging in `persistent/pwc_transport_tcp.py`, written to `logs/pwc_startup_diag_simple.jsonl` (S167:60-64).
- **Result:**
  - All 26 GPUs got their first job within 129 ms. rrig6600c offsets: gpu2 +0.001s, gpu7 +0.007, gpu5 +0.023, gpu3 +0.023, gpu4 +0.096, gpu0 +0.110, gpu1 +0.115, gpu6 +0.129 (S167:66-79).
  - Run `s168a_diag`, 200k cap, 2 trials: Trial 1 passed, Trial 2 failed with an rrig6600c page fault at t+30s (S167:129).
- **Kept or reverted:** kept as env-gated diagnostics.

#### E25. 2026-04-24 — S168 first-assignment jitter
- **Action:** a CRC32-deterministic delay on each worker's first job, `delay = crc32(worker_id) % (slots+1) / 1000`. Env `PRNG_PWC_FIRST_ASSIGN_JITTER_SEC` (default off). It resets on reconnect (S167:87-101; b6cabe9; code at persistent/pwc_transport_tcp.py:472).
- **Belief:** a 129 ms synchronized first wave causes an ROCm init spike (S167:89-90).
- **Result:** "S168 reduced startup hammer but did not eliminate crashes" (b6cabe9 body). Trial 2 crashes continued (S167:107-108).
- **Kept or reverted:** kept, default off.
- **Values used later:**
  - 3 in S169, S174 and REPRO (S167:145; L174:378; REPRO:63).
  - 15 in every S170–S173 launch script (T:launch_pool6.sh, T:launch_pool8_*.sh, T:launch_s173_*.sh).

#### E26. 2026-04-24 — S169 per-worker minimum gap pacing
- **Action:** `PRNG_PWC_PER_WORKER_MIN_GAP_SEC` (default off). 0.02 s was used (S167:105-121; b6cabe9; code at persistent/pwc_transport_tcp.py:497).
- **Belief:** about 20+ assignments reached rrig6600c within 0.5 s during sustained operation (S167:107-108).
- **Result:** `s169_test`, 200k, 2 trials, jitter plus pacing: both trials clean. Netconsole showed "rrig6600 SMU warnings (non-fatal)" (S167:130). T170:18 records "2 consecutive trials completed cleanly".
- **Kept or reverted:** kept, default off. Values later: 0.02 (L174:379; REPRO:64) and 0.20 in the S170–S173 scripts (T:launch_*.sh).
- **Contradiction:** the b6cabe9 commit body (same day) says "200k cap remains unstable under multi-trial load", "S169 improved pacing but has not yet certified stability", and "GCVM_L2_PROTECTION_FAULT and KFD-related faults still observed". S167:130 says the 200k/2-trial run was the "First clean 2-trial run". S167:191 calls startup hammering "always the root cause", and S167:197-200 lists three independent failure modes.

#### E27. 2026-04-24 — cwsr_enable=0 recorded as worse on rrig6600c
- **Recorded statements:**
  - "rrig6600c | stock (cwsr causes faster crashes on this rig)" (S167:180; S170P:14).
  - "cwsr_enable=0 behaves differently per rig — … causes faster crashes on rrig6600c. Do not apply universally" (S167:202).
  - Architecture invariant: "`cwsr_enable=0 mcbp=0` on rrig6600 ONLY" (T170:121).
  - "cwsr_enable=0 does NOT help rrig6600c" (T170:82).
- **Not recorded:** the experiment itself (date, run, duration).
- **Contradictions:**
  1. RT:51 calls the effect on rrig6600c "uncertain — under investigation".
  2. ROCm:123 (2026-05-02) says cwsr_enable=0 is "Already set to `0` on rrig6600 and rrig6600c per project configuration".
  3. D1:43 (2026-05-09) names cwsr_enable=0 as the "common factor" on rrig6600 and rrig6600c.
  4. No document records when cwsr_enable=0 was applied to rrig6600c.

#### E28. 2026-04-24 → 2026-04-25 — rrig6600 "script write failed" transport errors
- **Rig:** rrig6600 (.120).
- **Event:** `[PWC-TCP] 192.168.3.120:GPU* script write failed` during the Trial 2+ worker restart. Non-fatal (S170P:213; T170:74-77; b6cabe9 body).
- **Belief:** transport, I/O or lifecycle contention "independent of GPU faults" (b6cabe9 body).
- **Result of the investigation:** not recorded.

#### E29. 2026-04-25 — Stability-curve plan (100k / 150k / 200k × 5 trials)
- **Plan:** `min_workers=24`, jitter 3, gap 0.02, diag on. Grades: PASS, WARN or FAIL (S167:134-169; S170P:44-101; T170:25-46).
- **Result:** the results table was never filled in. Every cell is "?" (RT:11-15). The session instead ran the experiments in E30–E39.

#### E30. 2026-04-25 — S170 Run 1 (100k cap, S168/S169 jitter)
- **Rigs:** rrig6600c and rrig6600 GPU0.
- **Result:**
  - rrig6600c: GCVM_L2_PROTECTION_FAULT cascade at 08:53:56.
  - rrig6600 GPU0: SMU response 0xFFFFFFFF, then device lost from bus at 09:08:33.
  - Source: S170:166-169.
- **Belief:** later, "the same dispatch-pressure crash signature" (S170:169).

#### E31. 2026-04-25 — DPM peak-lock systemd service (all 3 rigs)
- **Action:** `/usr/local/sbin/s170_dpm_peak_lock.sh` and `/etc/systemd/system/s170-dpm-peak-lock.service` set `manual` perf level, the DPM peak sclk index and the COMPUTE profile at boot. Verified after every reboot (S170:274-278, 24).
- **Run 2, post-reboot with peak-lock, 100k / 2-trial Optuna:**
  - Ran 53 min and completed all 10,737 trial-1 chunks.
  - 292,671,006 survivors, then a SIGKILL from result-write blowup.
  - Netconsole clean throughout, attributed to "the survivor explosion throttled GPU dispatch".
  - Recorded as "Hardware PASS, pipeline FAIL" (S170:171-179).
- **Kept or reverted:** the service is listed as active at the end of S170 (S170:24). No later revert is recorded in era docs.
- **Contradictions:**
  1. S170P:111, the session prompt, says "Current rigs are running stock DPM settings".
  2. The S165 proposal was manual 2100–2200 MHz / 900 mV (E19). The peak-lock uses the default peak index instead (S170:351).

#### E32. 2026-04-25 — S170 Patch A: 512→2048 local skip buffers in prng_registry.py
- **Belief:** unsafe local array sizing created a risk of illegal memory access (S170:35-37).
- **Action:** buffers went from 512 to 2048 (S170:39). Script `patch_prng_registry_512_buffers.py`, labelled "S170-K512" (docstring). Committed in 7313a43 (prng_registry.py 22 +-).
- **Result:** "System still crashes → NOT root cause" (S170:41).
- **Kept or reverted:** kept, "DEPLOYED in production" (S170:316).

#### E33. 2026-04-25 — S170 Patch B: LIVE-GPU-PROBE
- **Action:** the coordinator probes the live GPU count, and each worker validates `Device(0)` before READY. If the probe fails, the coordinator fails closed and skips that node (S170:43-50; patch_s170_live_gpu_probe_v2.py docstring; 7313a43).
- **Result:** GPU_COUNT_MISMATCH was eliminated, but "System still crashes → NOT root cause" (S170:52).
- **Kept or reverted:** kept (S170:317).

#### E34. 2026-04-25 — S170 Test 1: remove rrig6600b (2-rig configuration)
- **Belief:** cross-rig contention (S170:60).
- **Result:** "Still crashes". Network and inter-rig coordinator pressure were ruled out (S170:62-64).
- **Not recorded:** pool size, seed cap, which rig crashed, time to crash.
- **Topology record:** 3fdf434 (2026-04-30) "rrig6600b (192.168.3.154) node restored" to `distributed_config.json`. "Reduced topology was a diagnostic tool for S170" (commit body).

#### E35. 2026-04-25 — S170 Test 2: worker_pool_size=6
- **Result:** 5/5 trials, no hipError, no GCVM, netconsole clean (S170:66-74).
- **Launch record:** T:launch_pool6.sh:
  - RUNLOG `s170_2rig_no6600b_k512_probe_pool6_100k_t5`
  - `min_workers 12`, `seed_cap_amd 100000`, `window_trials 5`, `worker_pool_size 6`
  - jitter 15, gap 0.20
- **Note:** the log name records that the run was 2-rig (no rrig6600b), with K512 and the probe. S170:66-74 does not say it was 2-rig.
- **Duration:** not recorded.

#### E36. 2026-04-25 — S170 throughput anomaly
- **Observation:** "pool8 throughput ≈ pool6 throughput (per-rig seeds/sec roughly equal)". Read as a "saturating shared resource" (S170:76-84, 137-139).
- **Numbers:** not recorded.

#### E37. 2026-04-25 — S170 Patch C: worker cooldown, and Test 3 (cooldown 50 ms)
- **Action:** env `PRNG_PWC_PER_WORKER_COOLDOWN_MS` inserts a sleep after each chunk completion (S170:90-96).
- **Belief:** to test whether the saturating resource is dispatch pressure, kernel-launch rate, or ROCm context churn (S170:96).
- **Test 3, pool8 + cooldown 50 ms:** "Still crashes at ~10 minutes" (S170:98-102).
- **Launch record:** T:launch_pool8_cooldown50.sh. RUNLOG `s170_pool8_cooldown50_100k_t5`, `min_workers 16`, 100k, 5 trials, pool 8, jitter 15, gap 0.20, cooldown 50.
- **Kept or reverted:** S170:318 says "DEPLOYED in production".
- **Git observation:** `git log --all -S"PRNG_PWC_PER_WORKER_COOLDOWN_MS"` finds only 7313a43 and 23919ad, and in those it appears only in launch scripts. `git grep -i cooldown 7313a43 -- '*.py'` finds no implementation in `persistent/` or `persistent_worker_coordinator.py`. The current code has no COOLDOWN_MS.

#### E38. 2026-04-25 — S170 Test 4: pool8 + cooldown 500 ms ("decisive experiment")
- **Launch record:** T:launch_pool8_cooldown500.sh. RUNLOG `s170_pool8_cooldown500_100k_t5`, `min_workers 16`, 100k, 5 trials, pool 8, jitter 15, gap 0.20, cooldown 500.
- **Status:** running at the end of the session (S170:104-110, 28).
- **Result:** not recorded in any document I read.
- **Decision tree:**
  - If it passes: design backpressure (S170:289-292).
  - If it fails: suspect memory footprint, ROCm queue or KFD limits, or "PCIe Gen1 interconnect saturation", then run pool7 and larger chunks (S170:294-302).

#### E39. 2026-04-25 — S170 recorded "known" facts and model
- **Recorded as known (S170:116-139):**
  - "Hardware is not fundamentally broken". Basis: stable while mining, stable at pool6, recovers after reboot.
  - "pool8 → crash within ~10 minutes".
  - "~10 minute runtime is the consistent failure threshold".
  - Failure chain: SMU errors → GPU reset → device lost from bus → probe failure.
  - "Reboot fully clears the issue".
- **Model:** "too much dispatch + concurrency + churn", not compute (S170:143-160).
- **Lessons:** "hardware is broken" hypothesis exhausted (S170:329); TB repeatedly corrected toward a code-path framing (S170:329).
- **Contradictions:**
  1. "Reboot fully clears" (S170:133-135) conflicts with S174 D1. There, rebooting rrig6600 and rrig6600c led to crashes (D1:6, 40-41).
  2. P2:131 (card0 PCIe) and S165:22-34 (Kaspa clock/voltage) recorded hardware or tuning causes.

#### E40. 2026-04-25 — VBIOS archive
- **Rig and card:** rrig6600c card1.
- **Action:** all 24 ROMs were saved to `~/vbios_backups/` on each rig. rrig6600c card1 is an L03 outlier, 0.052% different from L04, metadata only (S170:280-282).
- **Ruling:** TB deferred the L03→L04 flash because STRICT_DEVMEM requires `iomem=relaxed`, which was vetoed (S170:283).
- **Status:** not done.

#### E41. ~2026-04-26 to 2026-04-30 — S171 pool7 and coordinator dispatch-throttle runs (launch scripts only)
- **Pool7 scripts:** T:launch_pool7_100k_clean.sh and T:launch_pool7_100k_t5.sh. RUNLOGs `s171_pool7_100k_*`, `min_workers 14`, 100k, 5 trials, pool 7, jitter 15, gap 0.20.
- **Throttle patch:** `PRNG_PWC_COORD_DISPATCH_THROTTLE_MS` (default 0), a coordinator-side sleep before TCP submit. Its purpose: "Test whether Zeus-side dispatch burst pressure contributes to pool8 ROCm/SMU crashes" (patch_s171_coord_dispatch_throttle.py docstring :1-13).
- **Throttle run:** T:launch_pool8_100k_coord_throttle100.sh, throttle 100 ms, pool 8, 100k, 5 trials, `min_workers 16`.
- **Results:** not recorded for pool7 or for throttle100.
- **Git observation:** the throttle insertion is not in any committed coordinator. `git grep "S171\|THROTTLE"` returns nothing at 7313a43 or 23919ad. Only the patch script and a `.s171_coord_dispatch_throttle_bak` backup are in 23919ad.
- **Dates:** not recorded. The "S171" prefix and the 23919ad bundling are the only anchors.

#### E42. 2026-04-30 — Full topology restored for S172 (ruling)
- **Ruling:** TB said S172 validation requires the full 26-GPU load, so rrig6600b was restored to `distributed_config.json`. It was a "pre-condition for next pool8 run" (3fdf434 body).
- **Verification:** s172_prelaunch_check.py checks 4 nodes and 26 GPUs (docstring and :46-56).
- **Kept or reverted:** kept.

#### E43. 2026-04-30 to 2026-05-01 — S172 boundary map (all 3 rigs, 26 GPUs)
- **Runs** (ROCm:29-35; HEAD 3fdf434, "full 26-GPU topology", ROCm:186):

  | Pool | Chunk | Result | Time | Failure |
  |---|---|---|---|---|
  | 6 | 50k | stable | 29:30, 5 trials | — |
  | 6 | 100k | stable "Per operator history" | — | — |
  | 8 | 25k | stable, "1 transient eviction (recovered)" | 1:30:43, 5 trials | — |
  | 8 | 50k | crash | ~23 min | GCVM_L2 |
  | 8 | 100k | crash | ~12 min | GCVM_L2 |

- **Signatures:**
  - Both pool-8 crashes: SQC clients, WALKER_ERROR 0x7, MAPPING_ERROR 0x1, then 0xFFFFFFFF within ~95 ms (ROCm:41).
  - VRAM flat at 2% (ROCm:42).
  - The pool8/25k eviction was at 23:20, fully recovered (ROCm:87). The crash dump is named `20260501_232038_rrig6600-workers_8→6` (ROCm:194).
  - The pool8/50k crash dump is `20260501_220948_GCVM_L2_PROTECTION_FAULT` (ROCm:193).
- **Launch records:**
  - T:launch_pool8_50k_pool8.sh:2-10 prior data: pool6/50k = 29:30 clean; pool6/100k = stable (per operator); "pool8 / 100k = crash at 12:00 (rrig6600c device loss)".
  - T:launch_pool8_25k.sh, `min_workers 16`, 25k: "pool=8/50k crashed at ~23min; pool=8/100k crashed at 12min".
  - All used jitter 15 and gap 0.20.
- **Contradictions and limits:**
  1. ROCm:17 says "Six runs" and ROCm:27 says "Six runs … produced the following", but the table has five rows.
  2. The only recorded pool6/100k run is the 2-rig S170 run (E35). ROCm:32 cites "operator history".
  3. T:launch_pool8_50k_clean.sh is named "pool8" but runs pool 6, `min_workers 12`, 50k ("Per TB ruling: pool=6/min=12/seed_cap=50k").
  4. "Pool size 6 is stable at all chunk sizes tested" (ROCm:19) rests on two data points.

#### E44. 2026-05-02 — ROCm saturation report: hypotheses and proposed mitigations (none recorded as tried)
- **Belief:**
  - "Driver-level saturation" of the KFD queue manager, MES firmware and the UTCL2 walker (ROCm:91-105).
  - Concurrency is the primary trigger, and chunk size changes time-to-failure (ROCm:39-40).
  - Three mechanisms explain why 25k survives (ROCm:68-87).
- **External ROCm issues cited:** #5443, #5724, #6012, #5151, #5915 (ROCm:52-64). They are external and were not re-verified here [UNVERIFIED].
- **Proposed Category A** (ROCm:113-134):
  - `hws_max_conc_proc=4`
  - `sched_policy=1`
  - `queue_preemption_timeout_ms=30000`
  - `max_num_of_queues_per_device=256`
  - `no_queue_eviction_on_vm_fault=1`, labelled "not recommended"
  - recommended test line: `options amdgpu hws_max_conc_proc=4 sched_policy=1 queue_preemption_timeout_ms=30000`, then rerun pool8/50k
- **Proposed Category B** (ROCm:136-152):
  - B-1: one persistent worker per GPU.
  - B-2: HIP stream pooling.
  - B-3: a per-rig semaphore.
- **Proposed Category C** (ROCm:154-166):
  - C-1: in-kernel amdgpu instead of DKMS.
  - C-2: ROCm upgrade.
  - C-3: `amdgpu.enforce_isolation=1`.
- **Proposed Step 2 and Step 4:**
  - Step 2: accept pool=6 (~1.86M s/s vs a 2.07M ceiling) (ROCm:176).
  - Step 4: a journalctl watchdog (ROCm:180).
- **Result:** no document I read records any Category A or C parameter being applied. The ruling the report leads to is in E52 (switch backends).
- **Contradiction:** ROCm:123, the cwsr_enable=0 rig list. See E27.

#### E45. 2026-05-02/03 — S173 first-fault instrumentation and runs
- **Action:**
  - Per-worker active-job state (atomic JSON).
  - A post-dispatch coordinator ledger. v1 logged worker_host=localhost, so v2 moved the hook after `_run_once()`.
  - A daemon with inode and line-count truncate detection.
  - Sources: 7313a43 body; docstrings of s173_patch_coordinator.py, s173_patch_coordinator_v2.py and s173_patch_worker_service.py.
- **Recorded runs** (comments in T: scripts):
  - pool8/50k crashed "twice on rrig6600c (1m41s + 13min recurrence)" the day before (T:launch_s173_v2_crash_pool8_50k_t5.sh:7).
  - "pool=8/50k/t1 just ran clean — confirms trial count drives the crash" (T:launch_s173_validation_t5.sh:4).
  - The pool8/25k healthy baseline was saved to `~/baselines/pool8_25k_20260503_0938/` (T:launch_s173_v2_crash_pool8_50k_t5.sh:11).
- **Result:** "2 first-fault manifests captured 2026-05-03 (Trial 2 W21_O66_FT0.73 — rrig6600c gpu6 + rrig6600 gpu0)" (7313a43 body).
- **Kept or reverted:** committed in 7313a43.

#### E46. 2026-05-04/05 — TB Step 1 forced FT=0.73 reproduction (three runs)
- **Config:** W21 O66 FT0.73 RT0.31, skip 10–209, pool 8, chunk 50k, 1 trial, 425M seeds (about 354 chunks per worker), jitter 3, gap 0.02 (REPRO:2-19, 61-85).
- **Belief:** "Failure deterministic at this config (~325-350 chunks per worker)" (REPRO:6).
- **Results:**
  - Run 1, 2026-05-04: rrig6600b was operator-rebooted mid-trial, so the run is CONTAMINATED (S174a:23).
  - Run 2, 2026-05-05 17:43: all 8,500 chunks completed. Four minutes after trial finalization, a GCVM_L2 fault hit rrig6600c gpu0 and gpu6 at the same time. An SMU cascade followed on rrig6600 gpu0, matching manifests #1 and #2 from 2026-05-03. The crash phase was post-completion teardown, not chunk_active (S174a:25).
  - Run 3, 2026-05-05 18:31, after rebooting all 3 rigs: clean in 1:55 (S174a:27).
- **Follow-up:** the operator ordered all S173 instrumentation outputs deleted. That included ser8 `~/crash_dumps/2026*` ("~2400 dirs, daemon manifests back to 2026-04-17") (S174a:29-41).
- **Evidence-loss note:** the 2026-04-17 and 2026-05-01 dump directories cited in IR:276 and ROCm:193-194 fall inside that deletion range (S174a:36).

#### E47. 2026-05-06 — S174 baseline, pool8/25k (control)
- **Config:** pool 8, chunk 25k, 213M seeds, open Optuna (organic W27_O52_evening_S1-132_FT0.74_RT0.31), `min_workers 24` (S174a:57-59).
- **Result:**
  - 8,520 of 8,520 chunks, 0 faults, 0 manifests, 8/8/8 GPUs after the run (S174a:60).
  - Dispatch began with 2 of 26 workers ready, 0.5 s after the init broadcast, so the run is INVALID as a controlled baseline (S174a:62-69).
- **Kept or reverted:** n/a.

#### E48. 2026-05-07/08 — Hard ready gate (fix to the dispatch path)
- **Root cause found:** `min_workers` never reached `PersistentWorkerCoordinator`, which defaulted to 1 (S174a:83-104).
- **Action:** `apply_s174_ready_gate_fix.py`, which raises when ready is below `min_workers`, plus the wiring at window_optimizer_integration_final.py:352 (S174a:108-126; docstring; commit ca06f8c).
- **Gate validation, pool8/25k/5M:** 26/26 READY GATE PASSED. Clean, no netconsole entries (S174a:130-148).
- **Negative test:** READY_GATE_FAILED, exit 5 (D1:19).
- **Kept or reverted:** kept (ca06f8c).

#### E49. 2026-05-08/09 — S174 D1 forced crash reproduction
- **Config:** `d1-forced` mode: W21 O66 skip 10–209 FT0.73 RT0.31, pool 8, chunk 50k, 425M seeds, 1 trial, `min_workers 24` (L174:585-604). Env: jitter 3, gap 0.02, diag 1 (L174:377-379).
- **Result:**
  - The child exited 0. 425M seeds, 10:40 of compute, 8,500 forward and 8,500 reverse chunks, ~1.4M s/s (D1:20, 25).
  - About 1 s after the optimizer finished, rrig6600 had an SMU breakdown: `response:0xFFFFFFFF`, `TransferTableSmu2Dram`, `Failed to retrieve enabled ppfeatures` (D1:26-28).
  - Cascade: rrig6600 became unreachable, rrig6600b workers went 8→0, and rrig6600c 8→1 (D1:29).
  - The launcher classified the run VALID_CLEAN because of a FAULT_KEYWORDS gap. TB reclassified it VALID_POST_COMPLETION_FAULT (D1:5, 30-31).
  - After the run, rrig6600 and rrig6600c were rebooted and crashed. rrig6600b was not verified (D1:6, 38-41).
- **Belief at session close:** "Common factor in failed rigs: cwsr_enable=0 mcbp=0 … Hypothesis for tomorrow: revert cwsr_enable=0 on rrig6600 + rrig6600c" (D1:43-45, 72-77).
- **Ruling:** D2 is blocked until the rigs recover, the cwsr hypothesis is tested, D1 is reclassified, and TB clears it (D1:57-63).
- **Recorded outcome of the cwsr revert:** not recorded.

#### E50. 2026-05-10/11 — FAULT_KEYWORDS extension (classification fix)
- **Action:** added `response:0xFFFFFFFF`, `Failed to retrieve enabled ppfeatures`, `TransferTableSmu2Dram`, `GetEnabledSmuFeaturesHigh` and `GetEnabledSmuFeaturesLow` (L174:705-738; commit b78d76c 2026-05-11; the code comment says "2026-05-10").
- **Commit message:** "Required before D1.1 hypothesis test" (b78d76c).
- **Not recorded:** whether the other two queued patches were done, namely the stale-log fix and `--reclassify` (D1:47-55).

#### E51. 2026-05-11 — D1.1 (third multi-rig fault at 8 GPUs per rig)
- **Recorded:** "D1.1 (2026-05-11) was the third multi-rig fault under PWC at 8 active GPUs per rig" (PROP:94).
- **Not recorded:** the config, whether cwsr was reverted beforehand, the rigs and cards, the signature, and the duration. No document I read has them.

#### E52. 2026-05-11/12 — Ruling: stop debugging the cliff, switch backends (RANGE-MINER spec frozen 1f6c0c5)
- **Ruling:** "Owner directive: stop debugging the cliff, switch backends" (PROP:94). The spec was frozen at 1f6c0c5 on 2026-05-12 (git log).
- **Fault-relevant design choices:**
  - Sub-stripes "sized to fit VRAM and the watchdog ceiling" (PROP:92).
  - Per-family VRAM caps: constant phases use `seed_cap_amd` 2M and hybrid phases 1M. RX 6600 has 8 GB VRAM with a 4 GB miner reserve (PROP:402, 708-710).
  - One retry, then fail the trial (PROP:698-703, 722).
  - Acceptance §11.K, "Pool=8 cliff regression": a full 50-trial run on 3 rigs × 8 GPUs with no `launch_s174` FAULT_KEYWORD triggered (PROP:594-596).
  - The manifest sub-stripe default is 8, "sized to fit lockup_timeout=30000" (agent_manifests/window_optimizer.json:233).
- **Later status (outside the era):** the rrig6600c GCVM root cause is "attributed to the PWC launch storm; avoided, not eliminated" (docs/TODO_CONSOLIDATED.md:323). The facts file says "traced to launch-storm behaviour" (docs/TFM_PROJECT_FACTS_SKILL.md:181-188).
- **Gap:** no in-era document uses the phrase "launch storm". The era terms are "startup hammer" (S167:81), "dispatch pressure" (S170:5) and "driver-level saturation" (ROCm:17).

---

### Part 2 — Four lists

#### (1) Recorded as helping
1. Clone restore from rrig6600b onto rrig6600 and rrig6600c: 8/8 GPUs (C163:88-104) [pre-scope].
2. `set_limit()` 256MB in the TCP path: ladder stable to 300K (S163F:46-51, 66-76) [pre-scope].
3. GDM disabled and multi-user.target: "NO XORG, no SDMA timeouts post-reboot" (IR:260-261).
4. DPM pin + GFXOFF=0 + snd-power: "stable 26 GPU operation when hardware is healthy" (P2:132). This is a claim; no run is cited.
5. rrig6600 card0 riser reseat: "16.0 GT/s Gen4 confirmed healthy" (S164:14). Link state only.
6. Reverting Kaspa to stock `high` DPM, plus `HSA_ENABLE_DEBUG_TRAP=0`: a clean 2-trial, 14 min run with silent netconsole (S165:166-178).
7. cwsr_enable=0 mcbp=0 on rrig6600: 75+ min clean against a ~40 min prior best, at about 22K s/s (S166:92-97).
8. S168 jitter and S169 pacing: s169_test at 200k, 2 trials, both clean (S167:130). Contested by b6cabe9 (E26).
9. Worker reconnect window: workers survive rig reboot (P2:71-76).
10. pool=6, 100k: 5/5 trials clean on 2 rigs (S170:66-74; T:launch_pool6.sh).
11. pool=6, 50k: 29:30, 5 trials clean (ROCm:31).
12. pool=8, 25k: 1:30:43, 5 trials, one recovered eviction (ROCm:33).
13. DPM peak-lock run at 100k, 2 trials: 53 min, netconsole clean. This is attributed to survivor-explosion throttling, not to the lock (S170:171-179).
14. Reboot of all 3 rigs before FT=0.73 Run 3: clean in 1:55 (S174a:27).
15. Ready gate: correct 26/26 dispatch, gate run clean (S174a:130-148). This is not a fault fix.
16. FAULT_KEYWORDS extension: correct classification of SMU faults (b78d76c). This is not a fault fix.

#### (2) Recorded as not helping, or worse
1. `amdgpu.gpu_recovery=1 amdgpu.sched_jobs=256`: did not fix (C163:81) [pre-scope]. sched_jobs was later removed as not taking effect (S164:74).
2. Package reinstall and repeated reboots or power cycles: did not fix (C163:79-80) [pre-scope].
3. S162 Option B, `AMD_SERIALIZE_KERNEL=3`: ruled out (docs/TB_UPDATE_S162_OPTION_B_RESULTS.md:66-70) [pre-scope].
4. S162 kernel pre-compile warmup: "made crashes worse", reverted ac32cd9 (git) [pre-scope].
5. S162 CuPy pool disabled (raw hipMalloc): "made things worse", reverted 070199d (git) [pre-scope].
6. Kernel cmdline `amdgpu.gfxoff=0`: silently ignored by dkms 6.12.12 (IR:242).
7. KARG cp.int32 fix: the 2026-04-17 rrig6600c crash happened with it active. Classified "appears distinct", and the fix was kept (IR:150-156).
8. Kaspa 2250 MHz / -150 mV: "Was Causing All Crashes", with ASIC reset failures (-22) on rrig6600b 0000:09:00.0 (S165:22-34).
9. Stock `high` DPM: "unstable in longer runs (5-50 trials)" (S165:111-113). The evidence is not recorded.
10. cwsr_enable=0 on rrig6600c: "causes faster crashes" (S167:180, 202; T170:82, 121). Contested (E27).
11. cwsr_enable=0 on rrig6600b (S162): KIQ fence timeouts (S166:99-100; T170:122).
12. cwsr_enable=0 rigs (rrig6600 and rrig6600c) in S174: crashed after reboot. Named as the common factor (D1:40-44).
13. S168 jitter alone: Trial 2 crash on s168a_diag (S167:107-108, 129); "did not eliminate crashes" (b6cabe9).
14. 512→2048 buffer fix: still crashes (S170:41).
15. LIVE-GPU-PROBE: still crashes (S170:52).
16. Removing rrig6600b (2 rigs): still crashes (S170:58-64).
17. pool8 + cooldown 50 ms: crash at ~10 min (S170:98-102).
18. pool8 at 50k and 100k chunks on the full 26 GPUs: crashes at ~23 and ~12 min (ROCm:34-35).
19. Rig reboot as a fix: after S174 D1, rebooting rrig6600 and rrig6600c led to crashes (D1:6, 40-41).

#### (3) Proposed but never tried (or no result recorded)
1. Isolation-model check `ps auxf` on rrig6600c (IR:212-217). No result recorded.
2. Controlled A/B to close KARG (IR:152). Not recorded.
3. 100K × 3-trial "GDM removed" discriminator (IR:264-268). Not recorded under that name.
4. Staged ladder 100K → 300K → 500K → 1M (IR:224-227; FOR:236). Not recorded.
5. TK "free_all_blocks() removal, staged 500K → 1M → 2M" (TK:73-76). Deferred (T170:114).
6. DPM harness, 900 mV absolute, 2100–2200 MHz, with per-rig mem (S165:115-147; T170:48-58). Deferred (S170:351).
7. cwsr_enable=0 two-sequential-run test on rrig6600 (S166:178-179). Not recorded.
8. cwsr_enable=0 on rrig6600c with dkms 6.12.12 (S166:182-183). The outcome is asserted (S167:180), but the experiment is not recorded.
9. Stability curve 100k/150k/200k × 5 (RT:11-15). Never filled.
10. pool8 + cooldown 500 ms (S170:104-110). Outcome not recorded.
11. pool7 boundary test (S170:300; T:launch_pool7_*.sh). Outcome not recorded.
12. Coordinator dispatch throttle at 100 ms (T:launch_pool8_100k_coord_throttle100.sh). Outcome not recorded, and the code was not found in any commit.
13. Larger chunk sizes and kernel memory-pressure profiling (S170:301-302). Not recorded.
14. Backpressure, token-bucket or adaptive scheduler (S170:291-292, 341). Not built in era. RANGE-MINER was chosen instead (PROP:94).
15. Runtime pressure instrumentation (S170:344). Partly covered by S173 (E45).
16. Worker lifecycle hardening (exit within 30s of coordinator death) (S170:190, 345). Not recorded.
17. VBIOS L03→L04 flash on rrig6600c card1 (S170:283). Deferred and vetoed.
18. Kernel parameters `hws_max_conc_proc=4`, `sched_policy=1`, `queue_preemption_timeout_ms=30000`, `max_num_of_queues_per_device=256`, `no_queue_eviction_on_vm_fault` (ROCm:117-134). Not recorded as applied.
19. `amdgpu.enforce_isolation=1` (ROCm:166), in-kernel amdgpu instead of DKMS (ROCm:156-158), ROCm upgrade (ROCm:160-162). Not recorded.
20. B-1 single persistent worker per GPU, B-2 HIP stream pooling, B-3 per-rig semaphore (ROCm:140-152). B-1 is effectively what RANGE-MINER does (PROP:92), but the era docs do not name it as such.
21. journalctl watchdog for "Freeing queue vital buffer" and GCVM (ROCm:180). Not recorded.
22. Revert cwsr_enable=0 on rrig6600 and rrig6600c (D1:45, 76). Outcome not recorded.
23. crash_forensic_daemon stale-log fix (Option B, `/tmp/active_run.json`) and `launch_s174.py --reclassify` (D1:52-55). Not recorded in era.
24. D2 (D1:57-63). Blocked, and its run is not recorded.
25. Fix the `dmesg` capture in crash_forensic_daemon with `journalctl -k` (TK:47). Not recorded.

#### (4) Isolation results: every recorded combination
Columns: date | rigs (topology) | pool (workers per rig) | seed cap or chunk | trials | pacing env | clock/power and modparams | kernel/PRNG | transport | outcome | duration | anchor.

The kernel/PRNG column is java_lcg wherever it is recorded. TFM targets java_lcg (S164P:43 template `reset_seed_coverage.py java_lcg`; S170:248). No run in the era is recorded with another PRNG.

- **2026-04-07..10** | rrig6600, rrig6600c (post package removal) | 8 | 200K | 3 trials | none | `gpu_recovery=1 sched_jobs=256` | java_lcg | not recorded | CRASH, GCVM 0xFFFFFFFF on reverse sieve, ih ring overflow, device lost; 3362/5369 chunks failed | duration not recorded | C163:65-66, 81, 163. [pre-scope]
- **2026-04-12/13** | 3 rigs + Zeus, 26 GPUs | 8 | 10K, 100K, 250K, 300K | n/r | none | stock, dkms 6.12.12 | java_lcg | TCP-PWC | stable (250K and 300K after fresh reboot) | n/r | S163F:66-76. [pre-scope]
- **2026-04-12/13** | 3 rigs | 8 | 500K | n/r | none | stock | java_lcg | TCP-PWC | CRASH on rrig6600, "memory accumulation" | n/r | S163F:74. [pre-scope]
- **2026-04-16 18:01** | 3 rigs + Zeus, 26 | 8 (24 workers) | 100K | 3 | none | initramfs not rebuilt on some rigs | java_lcg | TCP-PWC | all 3 rigs crashed or rebooted; rrig6600b first at ~18:13; ruled invalid | ~12 min to first loss | RF:12-20, 29-35. [pre-scope]
- **2026-04-17 20:44** | 3 rigs + Zeus | 8 | 100K | 5 | none | gfxoff=0 + snd-power; stock DPM; GDM active; KARG active | java_lcg | TCP-PWC | rrig6600b GPU1 silent death at 20:56 (Trial 2); rrig6600c 06/16/19:00.0 fatal at 21:16 (Trial 3); Step 1 completed | ~12 min and ~32 min after launch | IR:20-30, 40-46.
- **2026-04-19** | 26 GPUs | n/r (8 assumed by TK:126, "26/26") | 100K (TK:127) | 5 | none | DPM pin, manual sclk1/mclk3 | java_lcg | TCP-PWC | production_1433 completed; production2_1508 had "rrig6600/c crashed Trial 3", GPUs 10–26 | n/r | P2:117-120.
- **2026-04-19 S164** | 3 rigs + Zeus | 8 | n/r | Trial 1 | none | stock clock ~2750 MHz (per S164:47); lockup_timeout -1 | java_lcg | TCP-PWC | rrig6600 card7 0000:19:00.0 KFD preemption failure, MODE1 reset; recovered in 3.5 min, no reboot | n/r | S164:30-54.
- **2026-04-19/20** | 3 rigs | 8 | n/r | n/r | none | Kaspa 2250 MHz / -150 mV, audio=0, dc=0, lockup 30000, gpu_recovery=1 | java_lcg | TCP-PWC | "multiple crash cycles"; rrig6600b 100W cards failed first; 09:00.0 ASIC reset -22, multiple power cycles; KFD debug-trap NULL dereference on all 3 rigs | n/r | S165:12-14, 22-34, 41-56.
- **2026-04-20** | 3 rigs + Zeus, 26 | 8 | 100K | 2 | none | stock `high` DPM, dpm-pin disabled, `HSA_ENABLE_DEBUG_TRAP=0` | java_lcg | TCP-PWC (Zeus partly on subprocess) | CLEAN, netconsole silent | ~14 min | S165:166-178.
- **2026-04-22** | rrig6600 (other rigs stock) | 8 | n/r | runs 1–5 | none | rrig6600 cwsr_enable=0 mcbp=0 | java_lcg | TCP-PWC | rrig6600: no GPU crash for 75+ min (prior best ~40 min), ~22K s/s; run2 "card1 crash" (rig n/r); run5 Zeus host OOM | 75+ min | S166:92-97, 136-142.
- **2026-04-24 s168a_diag** | 3 rigs + Zeus, 26 | 8 | 200k | 2 | DIAG=1 only | rrig6600 cwsr=0; others stock | java_lcg | TCP-PWC | T1 ok; T2 FAIL with rrig6600c page fault at t+30s | n/r | S167:129.
- **2026-04-24 s169_test** | 26 | 8 | 200k | 2 | jitter (value n/r; template 3) + gap (template 0.02) | same | java_lcg | TCP-PWC | both trials clean; rrig6600 SMU warnings, non-fatal | n/r | S167:130. Contested: "200k cap remains unstable"; "100k cap appears stable across repeated trials" (b6cabe9 body).
- **2026-04-25 S170 Run 1** | 3 rigs (implied) | n/r (8) | 100k | n/r | S168/S169 jitter | stock DPM (peak-lock was installed later) | java_lcg | TCP-PWC | rrig6600c GCVM cascade at 08:53:56; rrig6600 GPU0 SMU 0xFFFFFFFF, device lost at 09:08:33 | n/r | S170:166-169.
- **2026-04-25 S170 Run 2** | 26 workers | n/r | 100k | 2 (Optuna) | jitter | DPM peak-lock (manual, peak sclk, COMPUTE profile) | java_lcg | TCP-PWC | netconsole clean; pipeline SIGKILLed on 292M survivors | 53 min | S170:171-179.
- **2026-04-25 S170 Runs 3–4 and final** | 26 | — | 2M chunks by mistake (Run 4), then 100k | — | — | peak-lock | java_lcg | Run 3 fell back to legacy SSH by mistake; later TCP-PWC | config-mode bugs; 0 survivors; no fault recorded | Run 4: 5 s | S170:192-207.
- **2026-04-25 Test 1** | 2 rigs (no rrig6600b) | n/r | n/r | n/r | n/r | peak-lock (assumed by session state, S170:24) | java_lcg | TCP-PWC | still crashes | n/r | S170:58-64.
- **2026-04-25 Test 2** | 2 rigs (no rrig6600b, per log name) | 6 | 100k | 5 | jitter 15, gap 0.20 | K512 + probe patches | java_lcg | TCP-PWC | CLEAN, 5/5, no GCVM | n/r | S170:66-74; T:launch_pool6.sh.
- **2026-04-25 Test 3** | n/r (`min_workers` 16) | 8 | 100k | 5 | jitter 15, gap 0.20, cooldown 50 ms | — | java_lcg | TCP-PWC | CRASH | ~10 min | S170:98-102; T:launch_pool8_cooldown50.sh.
- **2026-04-25 Test 4** | n/r (`min_workers` 16) | 8 | 100k | 5 | jitter 15, gap 0.20, cooldown 500 ms | — | java_lcg | TCP-PWC | outcome not recorded | — | S170:104-110; T:launch_pool8_cooldown500.sh.
- **S170 general claim** | — | pool8 vs pool6 | — | — | — | — | java_lcg | TCP-PWC | pool8 crashes within ~10 min; pool6 stable; throughput roughly equal | ~10 min | S170:121-127, 137-139.
- **S171 (date n/r)** | `min_workers` 14 | 7 | 100k | 5 | jitter 15, gap 0.20 | — | java_lcg | TCP-PWC | outcome not recorded | — | T:launch_pool7_100k_clean.sh; T:launch_pool7_100k_t5.sh.
- **S171 (date n/r)** | `min_workers` 16 | 8 | 100k | 5 | jitter 15, gap 0.20, coordinator throttle 100 ms | — | java_lcg | TCP-PWC | outcome not recorded | — | T:launch_pool8_100k_coord_throttle100.sh.
- **S172, ~2026-04-30 to 05-01** | 3 rigs + Zeus, 26 GPUs | 8 | 100k | 5 | jitter 15, gap 0.20 | peak-lock (n/r in-run); cwsr=0 per ROCm:123 | java_lcg | TCP-PWC | CRASH, GCVM_L2, rrig6600c device loss | ~12 min | ROCm:35; T:launch_pool8_50k_pool8.sh:9; T:launch_pool8_100k_clean.sh.
- **S172** | 26 | 6 | 50k | 5 | jitter 15, gap 0.20 | same | java_lcg | TCP-PWC | CLEAN | 29:30 | ROCm:31; T:launch_pool8_50k_clean.sh (actually pool 6).
- **S172** | 26 | 6 | 100k | — | — | — | java_lcg | TCP-PWC | "STABLE — Per operator history" | — | ROCm:32.
- **S172, 2026-05-01** | 26 | 8 | 50k | 5 | jitter 15, gap 0.20 | same | java_lcg | TCP-PWC | CRASH, GCVM_L2 (SQC, WALKER 0x7, MAPPING 0x1, 0xFFFFFFFF within ~95 ms) | ~23 min (in trial 1) | ROCm:34, 41, 191, 193.
- **S172, 2026-05-01/02** | 26 | 8 | 25k | 5 | jitter 15, gap 0.20 | same | java_lcg | TCP-PWC | CLEAN except one transient eviction at 23:20, recovered (dump "rrig6600-workers_8→6") | 1:30:43 | ROCm:33, 87, 192, 194.
- **S173, ~2026-05-02** | `min_workers` 16 | 8 | 50k | n/r | jitter 15, gap 0.20 | — | java_lcg | TCP-PWC | crashed twice on rrig6600c | 1m41s and 13 min | T:launch_s173_v2_crash_pool8_50k_t5.sh:7.
- **S173** | `min_workers` 16 | 8 | 50k | 1 | jitter 15, gap 0.20 | — | java_lcg | TCP-PWC | CLEAN ("t1 just ran clean") | n/r | T:launch_s173_validation_t5.sh:4.
- **S173, 2026-05-03** | `min_workers` 16 | 8 | 25k | 1 | jitter 15, gap 0.20 | — | java_lcg | TCP-PWC | baseline saved, presumed clean (healthy signature saved) | n/r | T:launch_s173_v2_baseline_pool8_25k.sh; T:launch_s173_v2_crash_pool8_50k_t5.sh:11.
- **S173, 2026-05-03** | 26 | 8 | 50k (script) | 5 (Trial 2 config W21_O66_FT0.73) | jitter 15, gap 0.20 | — | java_lcg | TCP-PWC | two first-fault manifests: rrig6600c gpu6 and rrig6600 gpu0 | n/r | 7313a43 body; T:launch_s173_v2_crash_pool8_50k_t5.sh.
- **2026-05-04 FT073 Run 1** | 26 | 8 | 50k | 1 (forced) | jitter 3, gap 0.02 | — | java_lcg | TCP-PWC | CONTAMINATED (rrig6600b operator reboot) | — | S174a:23; REPRO:61-85.
- **2026-05-05 FT073 Run 2** | 26 | 8 | 50k, 425M seeds | 1 | jitter 3, gap 0.02 | — | java_lcg | TCP-PWC | 8,500 chunks done; 4 min after finalization GCVM on rrig6600c gpu0+gpu6, then rrig6600 gpu0 SMU cascade | compute n/r; fault at +4 min post-completion | S174a:25.
- **2026-05-05 FT073 Run 3** (after reboot of all 3 rigs) | 26 | 8 | 50k, 425M | 1 | jitter 3, gap 0.02 | — | java_lcg | TCP-PWC | CLEAN | 1:55 | S174a:27.
- **2026-05-06 S174 baseline** | 26 (dispatch began at ready=2) | 8 | 25k, 213M | 1 (open Optuna) | jitter 3, gap 0.02 | — | java_lcg | TCP-PWC | CLEAN, 0 faults, 8/8/8 after; INVALID as control | n/r | S174a:53-69; launch_s174_baseline_pool8_25k.sh:23-27, 85-87.
- **2026-05-07 gate validation** | 26/26 | 8 | 25k, 5M | 1 | jitter 3 (per L174 env) | — | java_lcg | TCP-PWC | CLEAN, no netconsole | ~1.5 s compute | S174a:130-148.
- **2026-05-08/09 D1** | 26 | 8 | 50k, 425M | 1 (forced W21_O66_FT0.73_RT0.31) | jitter 3, gap 0.02 | rrig6600 and rrig6600c cwsr=0 mcbp=0; rrig6600b stock | java_lcg | TCP-PWC | compute clean (17,000 chunks, ~1.4M s/s); ~1 s after completion rrig6600 SMU breakdown, then rrig6600 unreachable, rrig6600b 8→0, rrig6600c 8→1; reboots of rrig6600 and rrig6600c crashed | 10:40 compute | D1:20-31, 38-44; L174:585-604, 377-379.
- **2026-05-11 D1.1** | n/r | 8 per rig | n/r | n/r | n/r | n/r | n/r | PWC | "third multi-rig fault" | n/r | PROP:94.
- **Zeus 3080 Ti:** no GPU fault is recorded for Zeus in the era. Recorded Zeus items:
  - compute mode EXCLUSIVE_PROCESS → DEFAULT (IR:184; S164:99-114)
  - host OOM at 60 GB (S166:126-130)
  - throughput: 291K s/s per GPU alone vs ~4K when paired on the subprocess path (S165:80-94); 1,060,595 s/s after the TCP routing fix (S166:28-29)

---

### Part 3 — Contradictions found (summary; detail in entries)
1. **Idle versus mid-compute at the 2026-04-17 rrig6600c fault:** IR:54-56 against IR:131, 137, 183 and FOR:38 (E3).
2. **PERMISSION_FAULTS decode:** FOR:30-32, 40 against IR:97 (E3).
3. **VMID 8 "shared context":** FOR:39 against IR:111, 166-168 (E3).
4. **"free_all_blocks removed in S162 Option B":** IR:182, against S162 Option B being `AMD_SERIALIZE_KERNEL` with free_all_blocks active (TB_UPDATE_S162:15), against TK:73-76 (pending), T170:114 (deferred), and git ad4aa59 putting it back in sieve_gpu_worker.py:91-92 (E4, E8).
5. **ad4aa59 committed?** IR:210, 293 say committed; FOR:157-163 says not committed (E7).
6. **"S163 root cause":** TK:11 (amdgpu.conf/snd-power), IR:158 ("narrowed, not identified") and S163F:109-112 (CuPy pool) all differ (E9).
7. **rrig6600 card0 link speed:** S164:14 (16.0 GT/s Gen4 = healthy) against S164P:26 and P2:104 (5.0 GT/s expected, "Gen1 riser") (E11).
8. **Power management:** ruled out at 2630–2740 MHz (IR:129-131), but 2750 MHz is named as contributing (S164:47-48) and Kaspa clocks as causal (S165:22-34) (E4).
9. **Hardware:** "not root cause / exhausted" (S170:11, 329) against "card0 bad PCIe link — intermittent crash source" (P2:131) (E11, E39).
10. **Reboot:** "fully clears the issue" (S170:133-135) against reboots of rrig6600 and rrig6600c leading to crashes (D1:6, 40-41) (E39).
11. **S169 at 200k:** "first clean 2-trial run" (S167:130) against "200k remains unstable; GCVM still observed" (b6cabe9) (E26).
12. **Startup hammer:** "always the root cause" (S167:191) against three failure modes (S167:197-200) and "not sole root cause" (b6cabe9) (E26).
13. **cwsr_enable=0 rig set:** rrig6600 only (T170:121; RT:49-51) against rrig6600 + rrig6600c (ROCm:123; D1:43). When rrig6600c got it is not recorded (E27).
14. **cwsr on rrig6600c:** "faster crashes" (S167:180) against "uncertain — under investigation" (RT:51) (E27).
15. **DPM state at S170 start:** "stock DPM settings" (S170P:111) against the peak-lock service deployed in-session (S170:274-278) (E31).
16. **ROCm report run count:** "Six runs" (ROCm:17, 27) against five table rows (ROCm:29-35). pool6/100k is "per operator history" (ROCm:32); the only recorded pool6/100k run was 2-rig (E35, E43).
17. **Launch-script naming:** launch_pool8_50k_clean.sh actually runs pool 6 / 50k (T:) (E43).
18. **Cooldown and throttle:** S170-WORKER-COOLDOWN is "DEPLOYED in production" (S170:318), but no implementation is in any commit (git grep). The S171 throttle is likewise absent (E37, E41).
19. **jitter/gap values:** S169 and S174 used 3 / 0.02 (S167:145; L174:378-379); all S170–S173 launch scripts used 15 / 0.20 (T:). This is a recorded difference, not a contradiction, but no changelog mentions the change.

---

### Part 4 — Files read

#### Read in full
- docs/TB_Incident_Report_rrig6600c_S163KARG.md
- docs/SESSION_CHANGELOG_20260418_S163KARG_FORENSIC.md
- docs/SESSION_CHANGELOG_20260419_S163KARG_P2.md
- docs/TODO_MASTER_S163KARG.md
- docs/SESSION_CHANGELOG_20260419_S164.md
- docs/S164_CHAT_PROMPT.md
- docs/SESSION_CHANGELOG_20260420_S165.md
- docs/SESSION_CHANGELOG_20260422_S166.md
- docs/SESSION_CHANGELOG_20260424_S167_S168_S169.md
- docs/SESSION_CHANGELOG_20260425_S170.md
- docs/S170_CHAT_PROMPT.md
- docs/TODO_MASTER_S170.md
- docs/ROCm_Saturation_Report_S172.md
- docs/SESSION_CHANGELOG_20260509_S174_D1.md
- SESSION_CHANGELOG_20260507_S174.md
- SESSION_CHANGELOG_S163_FINAL.md
- SESSION_CHANGELOG_S163_RIG_FIXES.md
- S170_RESULTS_TABLE.md

#### Read in part
- docs/window_optimizer_integration_final.py: grep for fault terms; header :1-60; kill/port section :1130-1175. There are no fault-specific comments beyond the S163-KARG-KILL/PORT cleanup and the `seed_cap_amd` default of 2,000,000 at :365.
- docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md: grep for fault terms; read :85-100, :590-600 and :698-725.
- CHANGELOG_S163.md: grep and :30-110 (pre-scope).
- README_S170_PARITY.md: grep only.
- docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md: :636-652, :895-945, :1460-1472, :1664-1672.
- docs/TFM_PROJECT_FACTS_SKILL.md: :175-190.
- docs/TODO_CONSOLIDATED.md and docs/PROJECT_FILE_CATALOG.md: grep hits only.
- docs/PROPOSAL_S162_RRIG6600C_CRASH_ROOT_CAUSE_v1_0.md and docs/TB_UPDATE_S162_OPTION_B_RESULTS.md: grep hits only (pre-scope).
- launch_s174.py: :536-630, :705-738 and grep.
- launch_tb_step1_ft073_repro.sh: grep of header and command.
- launch_s174_baseline_pool8_25k.sh and launch_s174_gate_validation.sh: first 40 lines plus grep.
- s172_prelaunch_check.py: grep.
- persistent_worker_coordinator.py: :115-150 and grep.
- persistent/pwc_transport_tcp.py and persistent/pwc_worker_service.py: grep only.
- agent_manifests/window_optimizer.json: grep.

#### Module docstrings only
apply_s163_integration_json_guard.py, apply_s163karg_dedup_timing_diag.py, apply_s163karg_npz_sort_fix.py, apply_s163_kernel_arg_typing.py, apply_s163_mem_debug_worker.py, apply_s163_tcp_worker_mem_debug.py, apply_s165_zeus_tcp_worker.py, apply_s174_ready_gate_fix.py, launch_s174.py, patch_s171_coord_dispatch_throttle.py, patch_prng_registry_512_buffers.py, patch_s170_live_gpu_probe_v2.py, patch_config_mode_exec_parity.py, s173_patch_coordinator.py, s173_patch_coordinator_v2.py, s173_patch_worker_service.py, s172_prelaunch_check.py.

There are no `apply_s16[4-9]*`, `apply_s17[0-3]*`, `verify_s16[3-9]*`, `verify_s17*` or `launch_s170-173*` files in the repo root. `verify_s172_final.py` is cited in 3fdf434 but is not in the root.

#### Git, read-only
- Message and `--stat`: 884b424, c5832bf, b6cabe9, 7313a43, d00e075, b78d76c, 3c20d2d, fd534de, d32c1d1, ad4aa59, 3fdf434, 742a255, 23919ad, ca06f8c.
- Diffs: `git show ad4aa59 -- sieve_gpu_worker.py sieve_filter.py` and part of `persistent_worker_coordinator.py`; `git show d32c1d1 -- sieve_filter.py`.
- `git log` for the era and per file; `git log -S` for S163-FLUSH, S163-AUTOCLEAN, COOLDOWN_MS and COORD_DISPATCH_THROTTLE.
- From 23919ad: launch_pool6.sh, launch_pool7_100k_clean.sh, launch_pool7_100k_t5.sh, launch_pool8_100k_clean.sh, launch_pool8_100k_coord_throttle100.sh, launch_pool8_25k.sh, launch_pool8_50k_clean.sh, launch_pool8_50k_pool8.sh, launch_pool8_cooldown50.sh, launch_pool8_cooldown500.sh, launch_s173_v2_baseline_pool8_25k.sh, launch_s173_v2_crash_pool8_50k_t5.sh, launch_s173_v2_smoke.sh, launch_s173_validation.sh, launch_s173_validation_t5.sh.

#### Not read or not available
- Everything off-repo:
  - ser8 crash_dumps (deleted 2026-05-05 per S174a:36)
  - crash_forensic_daemon.py (ser8-only)
  - netconsole logs, run logs and tarballs on Zeus or ser8
  - rig `/etc/modprobe.d` and GRUB state
  - the S170 dpm peak-lock script
  - the private repository
  - host state
- D1.1 run records: no file found.
- The S170 cooldown-500 outcome, the pool7 outcome and the throttle100 outcome: no file found.
- The docx version of the incident report.
- Not read in full: docs/CHAPTER_9 (1,803 lines), TODO_CONSOLIDATED, PROJECT_FILE_CATALOG.
- docs/exchange/: not searched.
- docs/LEADS.md and docs/BACKLOG.md: grep returned no hits for the era terms.

---

## Part D — 2026-07-01 → 2026-11-15 (Proxmox / RANGE-MINER era)


Worktree `/home/michael/tfm_worktrees/gpu-fault-history` at HEAD `3579088` (2026-11-15). All `docs/...` anchors are relative to that worktree. "MEM:" = Claude auto-memory note under `/home/michael/.claude/projects/-home-michael-distributed-prng-analysis/memory/` (memory, NOT repo record). No root cause is concluded here.

### Null results / limits first
- No in-container kernel log was ever obtainable in this era: CT100 is an unprivileged LXC; `dmesg` denied, `/dev/kmsg` absent, `journalctl -k` empty, no root key auth VM101→`.121` (docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md:251-276; MEM rig-ct100-kernel-log-access-gap.md:13-25). Every Gate-12/Run report carries the GCVM criterion as `UNAVAILABLE`, never PASS (docs/PHASE6_PREREQS.md:415-432; docs/TFM_PROJECT_FACTS_SKILL.md:885-893).
- Exception: one manual host-side `dmesg` read by Michael for Phase 6.0 on 2026-07-30 (docs/TEAM_ALPHA_PHASE_6_0_RETURN_TO_BETA.md:69-80). And on 2026-11-15 Alpha's brief records "root key auth to the Proxmox hosts works from VM101" (docs/CLAUDE_CODE_INSTRUCTIONS_STEP1_CREDIT_WAL_20261115.md:110) — this reverses the standing access-gap fact; not otherwise recorded.
- The only GCVM_L2 fault *recorded as observed* in this era is the 2026-11-15 Run 4 retry, and that is recorded ONLY in Alpha's untracked brief (docs/CLAUDE_CODE_INSTRUCTIONS_GPU_FAULT_PRIOR_WORK_20261115.md:1). Every other era-D host loss (2026-08-19, 2026-08-22) has NO fault signature captured.
- The 2026-08-22 run-3 freeze is OPEN/UNDETERMINED (docs/LEADS.md:179-182, :222-225; docs/RIG_CRASH_FORENSIC_20260822.md:10-11, :456-461).

---

### ENTRIES (date order)

#### E1 — 2026-07-02/07 · Proxmox LXC rig-migration proposal v1.0 (commit 75ec134, 2026-07-07)
- Rigs/cards: rrig6600, rrig6600b, rrig6600c, "8× RX 6600" each (docs/PROPOSAL_Proxmox_LXC_Rig_Migration_v1_0.md:5).
- Belief: an 8-GPU simultaneous cold load is a "known power-spike crash risk" (docs/PROPOSAL_Proxmox_LXC_Rig_Migration_v1_0.md:108-109); "Cold-start power spike (historical crash)" (:134).
- Action: proposed — "apply existing warm-up/power-cap practice before any 8-GPU simultaneous load" (:108-109); "never cold-load 8 GPUs" (:134). CT100 "privileged or with systemd support" (:85-86). Ramdisk bind `/dev/shm/prng` (:91). 8 GB rig RAM risk (:132).
- Result: none recorded in this doc (proposal). Acceptance criteria superseded by the reconciliation doc (:7-11).
- Kept/reverted: superseded on acceptance criteria (:7-11).

#### E2 — 2026-07-07 · Infrastructure Reconciliation (S172), Beta-approved (commit 43a10c9)
- Rigs: Biostar TB360-BTC Pro 2.0, 8× RX 6600 "on x1 risers" (docs/PROPOSAL_Infrastructure_Reconciliation_S172_v1_0.md:82-85).
- Belief: "PCIe fragility on record. TB_SUBMISSION_S159G documents rrig6600 post-crash PCIe bus reassignment across all slots and NIC TX-queue timeouts. VFIO bind/reset operates at exactly this layer; LXC device binding does not touch it." (:86-89). IOMMU groups "typically fragmented" on mining boards (:84-85).
- Action: decide VM-vs-LXC by rrig6600c trial; Gate #1 requires "device/FLR resets work" for VM (:107-116); Gate 5 "Cold-start caution: Warm-up / power-cap practice applied before any 8-GPU simultaneous load (documented crash risk, ROCm issue #5238)" (:126-127). Host IP scheme "host = rig + 10" (.130/.164/.172) (:149-150).
- Result: not recorded here; LXC was adopted in practice (E3/E4).
- Contradiction: IP scheme "+10" here vs "+1" in CLAUDE.md §3 (.121/.155/.163) and in the runbook's own text (E3).

#### E3 — 2026-07-08 · Runbook v1.0 Proxmox + amdgpu-dkms install, verified on rrig6600c (commit ea372c6; file removed from TFM at 2cf23cf, 2026-07-11)
- Rig: rrig6600c first conversion; "applies to rrig6600, rrig6600b (identical hardware)" (ea372c6:docs/RUNBOOK_Proxmox_amdgpu_Rig_Install_v1_0.md:4-5).
- Belief: Proxmox VE 8.4 (kernel 6.8, inside ROCm 6.4.3 range) not 9.x (kernel 7.0 "unverified with our gfx1032-spoofed stack") (:12-15). "Driver = amdgpu-dkms 6.12.12 (matches the pinned rig version; fixes the 8-concurrent-procs-per-GPU crash)" (:16-18).
- Action: `apt-get install -y amdgpu-dkms amdgpu-dkms-firmware` → `amdgpu/6.12.12 … 6.8.12-9-pve` (:103-107); `/etc/modprobe.d/amdgpu.conf`: `options amdgpu gfxoff=0 ppfeaturemask=0xffff7fff` ("gfxoff must be modprobe.d — kernel cmdline is silently ignored per TB incident report") (:121-128); `apt-mark hold amdgpu-dkms amdgpu-install` (:131-134). Ext4 not ZFS "protects 8 GB RAM" (:45).
- Known open issues carried: "amdgpu-dkms has a host RSS leak (~5.4 GB/hr under load, ROCm #5915). In-kernel driver lacks the leak but can't handle 8 concurrent procs/GPU. Unresolved driver dilemma" (:150-152); "Cold-start 8-GPU power spike (ROCm #5238): keep warm-up/power-cap practice" (:153). Host owns driver/modprobe; container owns ROCm userspace + `HSA_OVERRIDE_GFX_VERSION=10.3.0 / HSA_ENABLE_SDMA=0` (:157-161).
- Result: "amdgpu-dkms 6.12.12 built + loaded on Proxmox VE 8.4 / kernel 6.8.12-9-pve; all 8 GPUs enumerated" (:6).
- Kept/reverted: superseded by v1.1–v1.4 same week; runbook then removed from TFM repo ("belongs in proxmox-gpu-passthrough") (commit 2cf23cf). Current state of that external repo: not read.
- Contradiction: IP scheme line says "host = rig IP + 1 … rrig6600 .120→.130 · rrig6600b .154→.164" (:19-20) — internally inconsistent and conflicts with CLAUDE.md (.121/.155/.163).

#### E4 — 2026-07-08 · Runbook v1.1: CT100 build (commit 0afb88e)
- Action: `pct create 100 … --cores 4 --memory 4096 --swap 512 … --unprivileged 0` ("privileged = simplest reliable GPU passthrough") (commit 0afb88e diff, §9.1). Gotchas: render GID 104 shows as `ssl-cert` in the CT; `/dev/kfd` major floats 236/237 across boots (0afb88e §10).
- Result: "a privileged LXC saw all 8 GPUs and rocminfo opened all 8 gfx1030 compute agents" (0afb88e §9).
- Contradiction (later record): CT100 is described as an UNPRIVILEGED LXC from 2026-07-30 on (docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md:252; MEM rig-ct100-kernel-log-access-gap.md:13). MEM rig-ct100-four-cores-eight-gpus.md:8 records CT RAM as 6 GB vs `--memory 4096` here. Which was deployed is not reconciled in the repo.

#### E5 — 2026-07-09 · Runbook v1.2: kernel cmdline + rocm-smi telemetry canary (commit 99f962e)
- Belief: installer-persisted `nomodeset` disables KMS → SMU hwmon `temp1_input` "Operation not permitted" for ALL cards, killing the "rocm-smi failure canary"; `amdgpu.runpm=0` (BACO off) "is the key telemetry fix" (99f962e step 21). The rocm-smi `Expected integer value from monitor, but got ""` error is "a validated dual-purpose signal: (1) a per-card hardware/SMU failure, and (2) — more commonly — feedback that the running code pushed the cards too hard" (99f962e §12).
- Action (exact): `mv /etc/default/grub.d/installer.cfg /root/installer.cfg.disabled`; `GRUB_CMDLINE_LINUX_DEFAULT="quiet pcie_aspm=off pci=noaer amdgpu.ppfeaturemask=0xffff7fff amdgpu.gfxoff=0 amdgpu.runpm=0 amdgpu.aspm=0 pci=assign-busses,hpbussize=0x33 iommu=pt amdgpu.dc=0 amdgpu.lockup_timeout=30000 amdgpu.gpu_recovery=1"`, "copied from a working bare-metal rig's /proc/cmdline"; `update-grub` (99f962e step 21). modprobe.d tuning demoted to "Optional/legacy … redundant" (step 22).
- Result: 8 real temps after fix (verification step 24). Three ruled-out theories: idle power state, "SMU driver if version not matched" dmesg line (present on working rig too), kernel lockdown (99f962e §12 debugging note).
- Kept: kept through v1.4 (2cf23cf removal diff shows the same step 21). Whether these params are on the hosts today: NOT recorded in TFM after 2026-07-11; Chapter 9 marks GFXOFF/ppfeaturemask on rigs `[UNVERIFIED]` (docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md:936-944; docs/CHAPTER_9_AUDIT.md:237). Module parameters are NOT baselined by the env-freeze gate (docs/CLAUDE_CODE_REPORT_DELEGATED_RUN_GATES.md:491-494).

#### E6 — 2026-07-09 · Runbook v1.3: `dev0:` path-based passthrough (commit 6e047bd)
- Action: `dev0: /dev/kfd,gid=104` + `lxc.cgroup2.devices.allow: c 226:* rwm` (6e047bd §9.2). Result: survives host reboot with floating kfd major; CT telemetry works via default /sys mount (6e047bd §10, §12).

#### E7 — 2026-07-09 · Runbook v1.4: ROCm userspace restore into CT100 (commit 89b546c)
- Action: ROCm 6.4.3 apt-pinned (`Pin-Priority: 1000`), `rocm-device-libs` required ("without it EVERY kernel segfaults"), `usermod -aG 104 michael`; `~/.bashrc`: `HSA_OVERRIDE_GFX_VERSION=10.3.0`, `HSA_ENABLE_SDMA=0  # RX 6600 stability`; "Both HSA vars are load-bearing on RX 6600"; note `su - michael -c` does not reliably source ~/.bashrc (89b546c §11.6).
- Result: CuPy matmul, PyTorch 8 devices, TF 8 GPUs on rrig6600c (89b546c §11).
- Contradiction (later): Phase 6.0 records "env overrides NONE required or set — empty in both non-interactive and login environments" (docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_6_0_ROCM_PARITY.md:33; docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md:49, :127). See E16.

#### E8 — 2026-07-07/18 · Miner design: cold-start warm-up, VRAM seed caps
- Belief/plan: "Cold-start power-cap practice … historical 8-GPU cold-load crash risk (ROCm issue #5238). The miner's Phase 3 READY handshake will warm up GPUs sequentially or in pairs, not all 8 simultaneously. Implementation detail — not exposed as a deployment knob." (docs/S172_INFRASTRUCTURE_INTERFACE_v1_0.md:71-75).
- As built: `warm_gpu()` warms only its own device; "Cross-GPU 'sequential/pairs, never all at once' sequencing is the SPAWNER's job" (miner/range_miner_worker.py:1589-1598). Implemented as a launcher stagger, default 3 s (scripts/launch_fleet_manual.sh:150, :340; brief docs/CLAUDE_CODE_INSTRUCTIONS_MANUAL_FLEET_LAUNCH.md:55-58).
- VRAM caps (TB Q2): AMD constant 2,000,000 / hybrid 1,000,000; NVIDIA 5,000,000 / 2,500,000 ("RX 6600 (8 GB VRAM, 4 GB miner reserve)") (docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:710; miner/range_miner_worker.py:648-659, :2815-2817). Launch shape: one launch per sub-stripe, 256 threads, default stream (docs/RIG_CRASH_FORENSIC_20260822.md:81-91).
- Rationale recorded: miner replaces "the ~17K-launch GCVM_L2 crash under PWC" (docs/S172_SIEVE_PATH_VERIFICATION_SCOPE.md:14-17).
- Kept: yes (current code).

#### E9 — 2026-07-22 · Topology: boot-selector model (commit 341ad09; CLAUDE.md §3)
- Fact: all rigs booted into Proxmox; bare-metal .120/.154/.162 DOWN, CT .122/.156/.164 UP; CT key auth verified (CLAUDE.md §3 as given in session context; commit 341ad09). Not a fault entry; baseline for later "unreachable" readings.

#### E10 — 2026-07-29 · Phase 6.0 proposal (commit 9e0bfe9)
- Belief: "the entire RANGE-MINER rearchitecture exists to solve a ROCm-side failure: PWC's silent hard resets / L2 protection faults on the rx 6600 rigs at full-fleet saturation, traced to launch-storm behaviour" (docs/TEAM_ALPHA_PROPOSAL_PHASE_6_0_ROCM_SMOKE.md:29-33). Rig stack stated as "ROCm 6.4.3 / kernel 6.8.0-107 / amdgpu-dkms 6.12.12 pinned" (:50).
- Contradiction: the run recorded kernel `6.8.12-9-pve` (docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md:122); 6.8.0-107 is not the Proxmox kernel.

#### E11 — 2026-07-30 · Phase 6.0 ROCm parity smoke, rrig6600 (commits 8e2f5bf brief, 23fa413 evidence, d3b3cd8 return)
- Rig/cards: CT100 rrig6600 `.122`, device 0, RX 6600 XT `0x73ff` rev `0xc1` gfx1032, VBIOS `113-123XT145W201222`, BUS 0000:03:00.0; host kernel `6.8.12-9-pve`, amdgpu `6.12.12`, ROCm `6.4.3-128`, cupy 13.5.1, env overrides none (docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md:115-127).
- Action: 1 worker, `--seed-start 0 --seed-count 8000000 --stripe-size 4000000 --seed-cap 1000000 --window-size 3 --offset 0` (:67-68); java_lcg; required "no GPU reset, no L2 protection fault, no VM fault … capture dmesg before and after" (docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_6_0_ROCM_PARITY.md:161-163).
- Result: byte-identical artifact to CUDA (sha 0e0092fe…) (:232-234); trial wall 15.9 s ROCm vs 24.7 s CUDA (:230). In-CT dmesg impossible (:251-264). Root-free surfaces unchanged: PCIe replay 0→0, KFD nodes 9→9, DRI nodes same, VRAM 0%→0%, post-run kernel sum 999000 = expected, 14-pattern worker-log scan 0 matches (:280-307). RAS "Not supported" on RDNA2 → unavailable, not clean (:309-312). Then Michael read host dmesg: boot init 11:46:39, three `Freeing queue vital buffer … queue evicted` pairs at 12:57:10, 12:59:48, 13:02:32; no GPU reset / gpu_recover / GCVM_L2 / VM_L2 / ring timeout / SDMA error / soft recovery / MES failure (docs/TEAM_ALPHA_PHASE_6_0_RETURN_TO_BETA.md:69-80).
- Kept: accepted as Phase 6.0 evidence (docs/TFM_PROJECT_FACTS_SKILL.md:476-478). Recorded later as "no GPU reset, no GCVM_L2 in the host kernel log" (:478).
- Contradiction: return-to-Beta says "The §4 kernel-log gap is closed from the host" (docs/TEAM_ALPHA_PHASE_6_0_RETURN_TO_BETA.md:153) while the evidence record ends "reported as an access-limited gap … rather than claimed" (docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md:485-486), and Phase 7 later states kernel-log observability "is NOT established" (docs/TFM_PROJECT_FACTS_SKILL.md:885). The closure was a one-off manual read, not standing access.
- Memory: MEM rig-ct100-kernel-log-access-gap.md:11-39 (same facts; root-free surfaces list).

#### E12 — 2026-08-02 · Phase 7 launch notice / PHASE6_PREREQS REV5 / skill v9 (commits 8781af8, 727362b, 23b5055)
- Belief (owner basis, Alpha shares): "GCVM_L2_PROTECTION_FAULT was a PWC launch-storm defect — ~17K kernel launches per trial — which followed that workload across every transport, a code revert and a package rollback. RANGE-MINER's persistent per-GPU daemons eliminate the workload" (docs/TEAM_ALPHA_PHASE7_LAUNCH_NOTICE.md:97-100). Qualification: "PWC was also stable at reduced scale and failed only at full saturation. This soak is the first time RANGE-MINER meets that condition" (:102-104).
- Action/ruling: launch without kernel-log observability by owner order; GCVM criterion "reported UNAVAILABLE, never PASS"; substitute detection polled at 60 s: `rocm-smi` device count/state, worker liveness, repeated lease expiries (:82-95). Binding TB language: "Phase 7 executed under an owner-authorized host-log observability exception. Absence of GCVM_L2_PROTECTION_FAULT or GPU-reset events was not independently established from the Proxmox host logs." (docs/PHASE6_PREREQS.md:415-432). Soak abort criteria include "any GCVM_L2_PROTECTION_FAULT or GPU reset in a host kernel log" (docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_7_SOAK.md:138-140).
- Result: the 50-trial Phase-7 soak "has not happened" (docs/TFM_PROJECT_FACTS_SKILL.md:2801-2803); the launch became the Gate-12 sequence.
- Kept: the UNAVAILABLE-never-PASS rule stands throughout (docs/TFM_PROJECT_FACTS_SKILL.md:890-893).

#### E13 — 2026-08-04/05 · First 25-daemon hand launch (brief 8f0ca7b; note 7c4f11b)
- Belief: RANGE-MINER "has never run more than 3 daemons"; "S157 JIT-cache race and the S155 VA-space/OOM risk at 8 workers/rig are untested on this path" (docs/CLAUDE_CODE_INSTRUCTIONS_MANUAL_FLEET_LAUNCH.md:14-26).
- Action: per worker `ROCR_VISIBLE_DEVICES=N`, `--gpu-id N --device-index 0`, per-worker `CUPY_CACHE_DIR`, ~3 s stagger, `--host 192.168.3.177` (:39-58, :101-102). Persisted in scripts/launch_fleet_manual.sh:36-43, :150, :316-321.
- Result: 2026-08-05 run, 25 daemons, died on `staging_deferred_max=64` (coordinator), not GPU (docs/TEAM_ALPHA_DEFERRED_QUEUE_NOTE.md:12-22). GPU/host fault: none recorded; kernel logs unavailable (docs/S172_STAGING_PART_B_REPORT.md:420). Duration: not recorded in documents read.

#### E14 — 2026-08-09 · Gate-12 attempt 1 `distributed_config_t1_689f3cd9`
- Rig/cards: cohort only on rrig6600 (`.122` "only .122 hosted the cohort") gpu0–gpu6 + Zeus; gpu7 log empty (docs/CLAUDE_CODE_REPORT_GATE12_FAILURE_FORENSICS.md:24-25, :121-154, :717-720). 8 workers, 32 stripes (:171).
- Result: 12:37:09→12:47:17; lease expiry `rrig6600:gpu4` (bulk lease aging) (:4, :47-55); rig worker logs 90 B, identical, two kernel-compile lines, no error text (:104-105). GPU fault: not claimed; rig kernel logs UNAVAILABLE (:717-718).

#### E15 — 2026-08-10 · Gate-12 attempt 2 `distributed_config_t1_abc63f71` (commit 4c76f42)
- Rigs: 24× RX 6600 XT + 1× 3080 Ti = 25 (docs/TB_GATE12_ATTEMPT2_FORENSIC_REPORT.md:33-34). Stages 1–3 complete, 6.44B seed-evals, 44,331 hybrid survivors (:17-27).
- Result: terminal 10:44:16 worker_admission_timeout 23/25 (:5-7). "Not hardware / not GPU lockup. Post-run, all three rigs report 8/8 GPUs responding (rocm-smi), zero dmesg GPU faults, no GCVM_L2 / ring-reset signatures." (:74-77). Defect-A report: VM101 dmesg silent; netconsole log ends at 06:53:34 LISTENER STARTED, zero entries 09:23–10:44, "equally consistent with no rig kernel event and with netconsole was not active" (docs/CLAUDE_CODE_REPORT_DEFECT_A_TRANSPORT_RECOVERY.md:268-274). Two lost sessions: cause UNRESOLVED, Beta accepted (docs/TFM_PROJECT_FACTS_SKILL.md:1743-1748).
- Fix: Defect A transport-session recovery (commits acd6f13, 2532803) — transport, not GPU.
- Contradiction: "zero dmesg GPU faults" (docs/TB_GATE12_ATTEMPT2_FORENSIC_REPORT.md:76; repeated docs/TFM_PROJECT_FACTS_SKILL.md:1724-1725) conflicts with the recorded impossibility of in-CT dmesg and absence of host key auth (docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md:251-258). Source of that dmesg read: not recorded.

#### E16 — 2026-08-10 · GPU probe/gate (commits 49ff9b4, 4643a11)
- Finding: `/opt/rocm/bin` on PATH only via `~/.bashrc:120` behind Ubuntu's non-interactive guard; only `bash -lic` reaches it; three outcomes (OK/count, UNAVAILABLE) (commit 49ff9b4 message). Gate: OK and count==expected on ALL three rigs → PROCEED; else REFUSE (scripts/gate12_gpu_gate.py:24-34). Also "rocm-smi is not on the rigs' non-interactive PATH" (docs/S172_STAGING_PART_B_REPORT.md:324).
- Relevance: explains how a non-interactive shell would also miss any `~/.bashrc` HSA exports (E7) — recorded as two separate facts, not linked in the record.

#### E17 — 2026-08-10 · attempt 3 `d606edbe`; 2026-08-11 attempt 4 `c8939b64`
- Attempt 3: 17:28:31→19:48:14, GPU preflight 8/8×3, 25-worker admission held ESTAB min=25 over 3,541 samples, 4 stages 128/128, zero disconnects; failed at publication (dirty tree) (docs/TFM_PROJECT_FACTS_SKILL.md:1873-1890). Phase-3 survivors 774 (docs/S172_PHASE3_SURVIVOR_CAPACITY_CHARACTERIZATION.md:52).
- Attempt 4: 19:04:14→19:22:53, stage 2, lease-origin defect (docs/TFM_PROJECT_FACTS_SKILL.md:1967-1970). No host/GPU fault recorded for either.

#### E18 — 2026-08-12 · Gate-12 attempt 5 `7e0d020b`
- Result: 17:38:44→18:59:53; GPU 8/8×3 prelaunch; 23/25 at stage 3→4; `rrig6600c:gpu2` and `rrig6600b:gpu3` WORKER_DISCONNECTED 18:27:10.537/.539; 940.856 s coordinator drain stall ended 1.46 s before; coordinator performed final close, "antecedent cause UNRESOLVED"; rig logs byte-identical, stopped 18:02:12 (docs/TFM_PROJECT_FACTS_SKILL.md:2086-2127). Phase-3 survivors 276,439 (:2093-2095).
- GPU fault: not recorded; not claimed. Kept open: "attempt-5 initiating reader-exit cause remains UNRESOLVED" (docs/TFM_PROJECT_FACTS_SKILL.md:2578).

#### E19 — 2026-08-14/15 · D6 dry runs; rigs power-cycled
- "the rigs rebooted 50 minutes earlier … that reboot destroyed the D6 dry run #2 rig logs … collected before the power-down" (docs/SESSION_CHANGELOG_20260815_S183.md:96-102). Who/why: "power-down" only; no fault recorded.

#### E20 — 2026-08-15/16 · attempts 6, 7, 8 (`db0393b0`, `36bf30e3`, `5c010902`)
- All died stage 2 on compute_lease_expiry (coordinator drain); attempt 6: "nobody crashed, nobody lost transport" (commit 087b7f4 message; docs/TFM_PROJECT_FACTS_SKILL.md:2319-2331). Host GPU kernel logs UNAVAILABLE (docs/ATTEMPT6_RIG_LOG_FORENSIC_v1_0.md:401-413; docs/GATE12_ATTEMPT7_H1H2_FORENSIC.md:247; docs/GATE12_MP1_RUN_FORENSIC.md:292). Attempt-6 release barrier: all 25 unparked from one token, 3.00 s ladder (commit 087b7f4 message).

#### E21 — 2026-08-17 · Gate-12 attempt 9 `554463d3` PASSED (launch e9ca800)
- Result: 128/128 over 4 stages, 738 s loop, peak 25 compute-active, zero lease expiries, fleet exited clean 0 workers on all three rigs; GPU 24/24, parity 30/30 (docs/TFM_PROJECT_FACTS_SKILL.md:2444-2456). Geometry k=12, anchor 25, τ 0.71, phase-3 59 / phase-4 23,515 survivors (docs/S172_PHASE3_SURVIVOR_CAPACITY_CHARACTERIZATION.md:58; docs/RIG_CRASH_FORENSIC_20260822.md:170). Trailing int32 (generator phase) = 25 (docs/RIG_CRASH_FORENSIC_20260822.md:143).
- Recorded "what is proven": "no transport collapse … no GCVM_L2 fault. The compute path works." (docs/TFM_PROJECT_FACTS_SKILL.md:2076-2080) — note the GCVM criterion itself was UNAVAILABLE (:890-893).

#### E22 — 2026-08-18 · Per-rig throughput heterogeneity (MEM only)
- MEM rig-throughput-heterogeneity.md:11-24: GPU0, identical work: .122 4.04/4.24 Gout/s; .156 3.55/3.65; .164 3.58/3.68, tracking `fclk` 1228/1006/942 MHz (`rocm-smi --showclocks`); "runtime clock state, not hardware". Not in repo.

#### E23 — 2026-08-19 · rrig6600 + host pve-rig6600 went down under sustained 8-GPU PIECE-MATCHER load (MEM only; not in TFM repo — searched `2026-08-19` in docs: 0 hits)
- MEM rig-ct100-four-cores-eight-gpus.md:8-41: CT100 nproc 4 (i5-9400, 6 GB RAM) for 8 GPUs; one-process-per-GPU with 2^22 chunks (128 launches/stripe): 1 GPU 155, 2 GPU 297, 4 GPU 552, 8 GPU 446 Mseed/s; coarsening chunk to 2^25/2^27/2^29 → 613/742/803 Mseed/s at 8 GPUs; "host-side launch overhead, not PCIe or thermal". "rrig6600 (.122) AND its Proxmox host pve-rig6600 (.121) stopped answering during the first multi-minute 8-GPU PIECE-MATCHER load; earlier 5-16 second bursts at the same concurrency were fine. Host-level … rrig6600b / rrig6600c completed the identical workload. Cause undetermined … no host dmesg and .121 was unreachable." "consider whether sustained 8-way concurrency is safe on these chassis at all." Kept/reverted: n/a.

#### E24 — 2026-08-22 · Brief I run 1 `eed23c7f` (clean compute) and run 2 `35ba80aa`
- Run 1: full 25-GPU fleet, 128/128, compute_lease_expiry 0, WORKER_DISCONNECTED 0, serve loop 713.663 s; failed at Phase-5 commit (Brief-I defect) (docs/S172_BRIEF_I_PRODUCTION_SHAPE_CLASSIFICATION_REPORT.md:3-4, :21-33). k=20, τ 0.64, phase-3 survivors 1 (docs/S172_PHASE3_SURVIVOR_CAPACITY_CHARACTERIZATION.md:59).
- Run 2: 17:40:58, "phase 1 only" (docs/RIG_CRASH_FORENSIC_20260822.md:6); max phase 1 (docs/S172_PHASE3_SURVIVOR_CAPACITY_CHARACTERIZATION.md:60). Why it stopped at phase 1: not recorded in documents read.
- Contradiction: forensic calls run 1 "same binary" `69ca910` (docs/RIG_CRASH_FORENSIC_20260822.md:8, :18, :144) but the run-1 report gives commit under proof `48a8705` (docs/S172_BRIEF_I_PRODUCTION_SHAPE_CLASSIFICATION_REPORT.md:3); `69ca910` was committed 08-22 17:02, after run 1 (git log).

#### E25 — 2026-08-22 · RUN 3 `8cb116da` — two rig hosts froze at phase-4 start (RIG_CRASH_FORENSIC_20260822, committed e75a6eb)
- Rigs: rrig6600 and rrig6600c stopped; rrig6600b and Zeus survived (docs/RIG_CRASH_FORENSIC_20260822.md:348-349).
- Facts: launch 18:21:19, terminal 18:42:01 (:3-4); phase 3 (`java_lcg_hybrid`) 18:26:09–18:30:03, 13,146,485 survivors, 478.96 MiB inline (:304-306); phase 4 dispatched 18:32:01, "all 25 workers resume GPU work simultaneously" (:307); rrig6600 7 shards, rrig6600c 8 shards 18:32:02–07, ALL 15 fail staging, then nothing (:310-311); WORKER_DISCONNECTED = 0, sockets ESTABLISHED for 10 minutes, conn9–12/14 exited TRANSPORT_ERROR at teardown (:324-332); trial terminal compute_lease_expiry `rrig6600:gpu4` attempt 1 (:313-315). k=6, τ 0.35, M=3 (docs/S172_PHASE3_SURVIVOR_CAPACITY_CHARACTERIZATION.md:61).
- Eliminations recorded: launch mechanics/kernel bodies/arg tuple identical e9ca800→69ca910 (:43-131); only device difference trailing int32 25→0 (≈0.6% less arithmetic) (:133-153); rig disk/tmpfs I/O zero (0 spooled ever) (:211-227); worker RSS ≈10.6 MiB/CT (:233-246); run 3 used LESS VRAM (24 MB vs 48 MB skip buffer) (:248-250); per-host volume inverted vs survival (:280-283); "long idle before phase 4" refuted (117.2 s vs 189.9 s) (:284-286). H4 worker death ELIMINATED (:358).
- Hypotheses open: H2 shared-circuit voltage sag under synchronised 24-GPU phase-4 resume — "OPEN, co-leading", not demoted (:288-299, :356); H3 host resource exhaustion under 13.1M-survivor staging — OPEN (:357); H5 network/switch — OPEN (:359); H6 boot-selector N/A (:360). CT100 memory/CPU/disk UNAVAILABLE ("No route to host" 2026-08-27) (:22-26, :252-258).
- Proposed next experiments (not run): reproduce the volume deliberately with host instrumentation; ≥1 Hz per-host series (`MemAvailable`, load, sockets, `rocm-smi`) armed before dispatch; instrument the phase transition; explain the 15 staging failures; consider a survivor-volume bound (:362-384).
- Kept: OPEN (:10-11).

#### E26 — 2026-08-22 evening · post-incident operator actions; .155 reboot
- 11 netconsole packets: NC-TEST3 and NCPROOF from .121/.155/.163 at 19:25–19:26; `watchdog: watchdog0: watchdog did not stop!` ×2 and `systemd-shutdown … Failed to finalize DM devices` on .155 19:52:59; same shutdown line .121 20:33:53, .163 20:33:57 (docs/RIG_CRASH_FORENSIC_20260822.md:415-427). Operator: "he always shuts the rigs down after a crash" (:431-435). Implied boots 18:52:41 (.155), 18:58:18 (.163), 18:58:55 (.121) (:466-470).
- MEM rig-boot-selector-proxmox-recovery.md:19-26: "On 2026-08-22 I rebooted `.155` to test netconsole boot-persistence, it returned as Ubuntu, and I began treating a healthy machine as a possible wedge." Rule added: never reboot a host unasked; rebooted rig returns as Ubuntu; `boot-proxmox` required (MEM :11-16; docs/TFM_PROJECT_FACTS_SKILL.md:2943-2956, skill v27.1 dated 2026-08-23 :8).
- MEM references a note `rig-crash-2026-08-22-hang-not-power` (MEM rig-boot-selector-proxmox-recovery.md:27) — that file does NOT exist in the memory directory.

#### E27 — 2026-08-27 · Phase-3 survivor capacity characterization (committed e75a6eb)
- Belief: "Survivor volume is **not** a crash hypothesis. RIG_CRASH_FORENSIC_20260822.md refuted the rig-side mechanisms" (docs/S172_PHASE3_SURVIVOR_CAPACITY_CHARACTERIZATION.md:10-16). Corrects forensic §3.2 "82% of file ceiling" framing (:18-21).
- Result: run table k/τ/M/survivors for attempts 1–9 and runs 1–3 (:48-62); last known-good points (:350-360); "No crash-seeking ladder is designed or proposed here, and none should be run" (:410).
- Contradiction: forensic lists H3 (volume/host resource exhaustion) as "OPEN, co-leading" (docs/RIG_CRASH_FORENSIC_20260822.md:357); the capacity doc says volume is not a crash hypothesis (:10); the Run-4 proposal calls the forensic the doc for "why run 3 is not a volume crash" (docs/S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md:7); the delegated-execution proposal says the volume cap "guards against one of its two candidate causes" (docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:259-260).
- LEADS L-2 (ingress queue bounded by count not bytes; VM101 15.9 GiB, swap 0) — computed, never observed (docs/LEADS.md:92-173).

#### E28 — 2026-08-28 · Run-4 routing patch cycle: pinned geometry; TB R2 post-commit hardening
- Ruling: Run 4 pinned to attempt-9 geometry (k=12, anchor 25, midday, 0.71/0.47, skip 6–99, generator_phase 0) (docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:16); reason: 55% of reachable cells predict phase-3 volume above 23,515 (docs/S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md:166-171). TB R2 post-commit step 3: "verify the Run-4 infrastructure hardening state—actual GPU power caps, off-host power telemetry, and persistent netconsole" (docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:149).
- Result: fleet powered off — all nine rig endpoints "No route to host"; power caps/telemetry "Unverifiable fleet-down"; physical circuit check "OWED", H2 "OPEN, co-leading" (docs/SESSION_CHANGELOG_20260828_RUN4_ROUTING_PATCH.md:229-233, :274-275).

#### E29 — 2026-08-28 · Crash-forensic correction / LEADS L-3 / netconsole runbook (commit e75a6eb)
- Correction: "netconsole = EMPTY" was wrong; 11 packets (E26) are post-incident; "not active" branch CLOSED, "no event during the freeze" STANDS; freeze "bounded to 18:42-18:52" (docs/RIG_CRASH_FORENSIC_20260822.md:403-475; docs/LEADS.md:177-277; docs/NOTE_FOR_BETA_20260828_NETCONSOLE_EVIDENCE.md:13-34).
- Runbook (stub, sender `[RECONSTRUCTED]`): receiver UDP 6667 (`netconsole_listener.py:16`), senders are the Proxmox hosts not the CTs; `modprobe netconsole netconsole=6665@${SRCIP}/${IFACE},6667@192.168.3.177/${DSTMAC}`; proof `echo "NCPROOF-$(hostname)" > /dev/kmsg`; "Persistence across reboot is NOT configured and is NOT assumed" (docs/RUNBOOK_NETCONSOLE_REARM.md:15-29, :55-83, :87-115).
- Contradictions within the correction (recorded, not resolved): (a) the NCPROOF packets were sent at 19:26 from boot sessions that started 18:52–18:58 (docs/LEADS.md:232-235), and the runbook says arming does not persist across reboot (docs/RUNBOOK_NETCONSOLE_REARM.md:81-83, :138); so the 19:26 proof is from a post-reboot session and the record does not show the senders were armed during the 18:32 event — yet the correction says "senders were provably armed and delivering" and "not active — CLOSED" (docs/RIG_CRASH_FORENSIC_20260822.md:447, :450-452). (b) The forensic places the two hosts' last bytes at 18:32:07 (:310-311, :325-326) but L-3 labels the run log's last line at 18:42:04 "the freeze" and bounds it 18:42–18:52 (docs/LEADS.md:231, :259-260); 18:42:04 is VM101-side teardown (docs/RIG_CRASH_FORENSIC_20260822.md:316). (c) .155 (rrig6600b, the surviving host) also shows a post-incident boot at 18:52:41 (docs/LEADS.md:232), consistent with MEM's agent reboot of .155 (E26), not with a freeze.
- Kept: L-3 OPEN; follow-ups: re-arm before any run and confirm NCPROOF; freeze cause uninvestigated (docs/LEADS.md:268-274).

#### E30 — 2026-11-11 · Owner rulings: power caps / circuit check / power telemetry DROPPED; no stress runs
- "Power caps, circuit check, power telemetry before Run 4 → Dropped. The crash recorder (netconsole) stays" (owner ruling 3) (docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:21); "Freeze-reproduction and 13M-survivor stress runs → Not planned. The owner does not want runs that try to overwhelm the rigs" (:26). Delegated execution: circuit check/power caps removed; power cutoff removed from stop conditions; freeze-reproduction class removed; survivor-volume limit added (docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:15-26). Stop conditions include "a host or worker goes silent, or the visible GPU count drops", "any kernel message on netconsole" (:187-189). No agent reboot/power-off (:77-78). Supply sag "not addressed, by owner ruling" (:259-260).
- Contradiction: TODO still lists power caps/telemetry and circuit check as open (docs/TODO_CONSOLIDATED.md:80, :82); flagged as "ALREADY RECORDED (TODO_CONSOLIDATED.md:80 still lists it; the ruling drops it)" (docs/CLAUDE_CODE_REPORT_RUN4_LAUNCH_PREP.md:120).
- TODO items carried (not done): DPM harness "900 mV / 2100–2200 MHz" `amdgpu-dpm-pin.service` status unknown (docs/TODO_CONSOLIDATED.md:281); `crash_forensic_daemon.py` not in clone (:282); `rig_crash_monitor.sh` not in clone (:283); rrig6600c GCVM root cause marked SUPERSEDED "avoided, not eliminated" (:323); node watchdog/auto-restart owed (:190).

#### E31 — 2026-11-11 · Chapter 9 corrections (ROCm env)
- "SUPERSEDED (2026-11-11) for the current rigs … no override is required or set … The variables are still exported by scripts_coordinator.py:67-74 and by PWC's launch command (persistent_worker_coordinator.py:140-141, :385-386) … RANGE-MINER sets none of them. Kept as the historical reason (the GCVM_L2 fault chase)" (docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md:640-648). Hostname-guarded prelude (`rig-6600*`) never fires on `rrig6600*` (docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:1046-1067, :1173; code sieve_gpu_worker.py:26-30, sieve_filter.py:23-27); the miner imports sieve_gpu_worker (miner/range_miner_worker.py:1150). Disposition: do NOT rename hosts (would "activate obsolete ROCm environment overrides") (docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:1069-1080).
- Consequence as a fact: RANGE-MINER rig workers run without `HSA_ENABLE_SDMA=0`, which the 2026-07-09 runbook called "RX 6600 stability" / "load-bearing" (E7). No A/B recorded.

#### E32 — 2026-11-13 · Delegated-run gates G2/G3 (commit 0066b6e)
- Env-freeze gate baselines kernel release, `/proc/cmdline`, pip freeze, dpkg, amdgpu version, ROCm version; CT `uname -r`/`/sys/module/amdgpu/version` show host values (docs/CLAUDE_CODE_REPORT_DELEGATED_RUN_GATES.md:113-125). Not baselined: `/sys/module/amdgpu/parameters/*` e.g. `ppfeaturemask` (:491-494). Fleet 0/3 reachable (:14-16). Python 3.10.12 on rrig6600c vs 3.10.4 on the other two — "STILL NOT DIAGNOSED" (docs/TFM_PROJECT_FACTS_SKILL.md:2552-2558; docs/CLAUDE_CODE_REPORT_DELEGATED_RUN_GATES.md:114).

#### E33 — 2026-11-14 · PWC backup Stage A (owner direction) — PWC-era stability envelope restated
- Owner 2026-11-14: PWC to become a working backup "at the settings that were stable" (docs/CLAUDE_CODE_INSTRUCTIONS_PWC_BACKUP_STEP1_20261114.md:27). Stage A tabulates PWC history (pre-July sessions; source changelogs cited there, not re-read here) (docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STAGE_A.md:318-351); repeatable envelope "pool 6 per rig at ≤50K … and pool 8 per rig at ≤25K" (:357-358); conflict at pool 8/100K (S162 VICTORY clean vs S172 crash ~12 min) not adjudicated (:360-362); "The record does not support a rig-count limit. It supports a per-rig concurrency × chunk-size limit." (:372-373); manifest defaults pool 8 / seed_cap_amd 2,000,000 lie outside every recorded survival (:110, :310); PWC still injects `HSA_OVERRIDE_GFX_VERSION=10.3.0` (:314); launch-per-trial = launch-storm shape (:311).
- Action: step-1 encodes envelope table 6/50K and 8/25K with pre-spawn refusal (Beta X-PWC-BACKUP R1 Q4 "NO OBJECTION", anchors `persistent_worker_coordinator.py:362-383` in the pwc-step1 worktree) (docs/exchange/from_beta/X-PWC-BACKUP_R1_BETA_REVIEW.md:52, :82-84). Not present in this worktree's persistent_worker_coordinator.py (grep for ENVELOPE/25_000/50_000: 0 hits) — uncommitted elsewhere.

#### E34 — 2026-11-14 · Survivor-volume cap raised 23,515 → 25,000 by owner (commit 35f5503)
- (docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md:5-6); Run 4 upper-half phase-4 exact count 23,568 (:16-21); Beta position: retain 23,515, no completed-run evidence for raising (:31-40). Original cap and rationale "Run 3 froze after 13,146,485" (docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:163-165).

#### E35 — 2026-11-14 · Run 4 launch checklist / prep
- Steps: `boot-proxmox`; GPU gate `python3 -u scripts/gate12_gpu_gate.py … PIPESTATUS[0]` (docs/RUN4_LAUNCH_CHECKLIST.md:61-67, :125-130); netconsole re-arm by owner as root on hosts (:215-247); netconsole receiver already a systemd service on UDP 6667, PID 1102 (docs/CLAUDE_CODE_REPORT_RUN4_LAUNCH_PREP.md:42-43, :288-291). "A rig drops … Do not reboot the host" (docs/RUN4_LAUNCH_CHECKLIST.md:572). Rig state last measured 2026-08-22 (docs/CLAUDE_CODE_REPORT_RUN4_LAUNCH_PREP.md:52-55).

#### E36 — 2026-11-15 09:05 · Run 4 attempt 1 `distributed_config_t1_57dc50fa` — CLEAN compute
- Owner-launched 09:05:53 from `/home/michael/run4_launch.sh`; HEAD 0e13dd6; parity 30/30; seeds [2^31, 2^32); phase starts 09:07:39 · 09:09:05 · 09:10:44 · 09:15:16, 25 eligible each; summary 09:19:49; survivors phase 3: 60, phase 4: 23,568; WORKER_DISCONNECTED 0; saturation SATISFIED; credit refused (non-empty ledger -wal); after: "rigs up, 8/8 GPUs each" (docs/CLAUDE_CODE_INSTRUCTIONS_STEP1_CREDIT_WAL_20261115.md:50-63). Evidence bundle `/home/michael/run4_attempt1_20261115_090553/` (:65).
- Same-day record corrections: "root key auth to the Proxmox hosts works from VM101; netconsole needs the host console level raised with `dmesg -n 7`" (:110) — recorded only as out-of-scope notes.
- Fix (not GPU): Step-1 credit reads a stable copy when -wal non-empty (commit 3579088).

#### E37 — 2026-11-15 (after E36) · Run 4 retry `distributed_config_t1_ab6318d4` — GCVM_L2 + device lost + host hang (recorded ONLY in Alpha's untracked brief)
- "Run 4 retry distributed_config_t1_ab6318d4 lost pve-rig6600 and pve-rig6600c at phase-4 start with amdgpu GCVM_L2_PROTECTION_FAULT, then status 0xFFFFFFFF and device inaccessible, then host hang; this is the same fault as the PWC era." (docs/CLAUDE_CODE_INSTRUCTIONS_GPU_FAULT_PRIOR_WORK_20261115.md:1). Card indices, timestamps, log source: not recorded in this worktree. Evidence work delegated to `/home/michael/tfm_worktrees/rig-gpu-fault` (not read, by instruction).
- Recorded juxtaposition (no conclusion): the same two hosts (rrig6600, rrig6600c) at the same point (phase-4 start) as run 3 (E25); the same geometry ran clean three hours earlier in E36.

---

### LIST 1 — recorded as helping
1. Removing installer `nomodeset` + `amdgpu.runpm=0` (+ full bare-metal cmdline) restored SMU hwmon telemetry / rocm-smi canary on Proxmox host (commit 99f962e step 21, step 24). [helps telemetry, not recorded as preventing faults]
2. `dev0: /dev/kfd,gid=104` path passthrough survives floating kfd major and host reboot (commit 6e047bd §9.2, §10).
3. amdgpu-dkms 6.12.12 "fixes the 8-concurrent-procs-per-GPU crash" (ea372c6 runbook :16-18) — stated, no test result in TFM.
4. RANGE-MINER persistent per-GPU daemons (~52 launches/trial vs PWC ~17K) — recorded as removing the launch-storm workload (docs/TFM_PROJECT_FACTS_SKILL.md:2533-2537; docs/TEAM_ALPHA_PHASE7_LAUNCH_NOTICE.md:97-100); evidence: Phase 6.0 clean host dmesg (docs/TEAM_ALPHA_PHASE_6_0_RETURN_TO_BETA.md:69-80); attempts 3/9, run 1, Run 4 attempt 1 completed four stages without host loss (E17, E21, E24, E36). Qualification recorded at the time: first full-saturation test (docs/TEAM_ALPHA_PHASE7_LAUNCH_NOTICE.md:102-104). Contradicted by E37 per Alpha's brief.
5. Coarsening kernel launches (chunk 2^22→2^29) raised 8-GPU throughput 446→803 Mseed/s on a 4-core CT (MEM rig-ct100-four-cores-eight-gpus.md:15-24) — throughput, not stability.
6. Per-worker `CUPY_CACHE_DIR`, `ROCR_VISIBLE_DEVICES=N` + `--device-index 0`, 3 s launch stagger (scripts/launch_fleet_manual.sh:36-43, :150) — in every 25-worker run since 08-05; no failure attributed to their absence.
7. Pinned low-volume geometry (attempt-9) for Run 4 + survivor-volume cap (E28, E34); Run 4 attempt 1 clean (E36).
8. PWC era (as restated 2026-11-14): pool 6/≤50K and pool 8/≤25K clean; pacing/jitter (S169), pool 6 (S170 Test 2: 5/5, no GCVM) (docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STAGE_A.md:343-345, :347-349, :357-358).
9. Root-free fault surfaces (PCIe replay count, KFD topology nodes, DRI inventory, VRAM, post-run kernel launch, worker-log scan) as substitute detection (docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md:280-307).
10. Netconsole re-arm + NCPROOF gate (docs/RUNBOOK_NETCONSOLE_REARM.md:87-115) — helps observability; captured nothing during the 08-22 event (docs/RIG_CRASH_FORENSIC_20260822.md:472).

### LIST 2 — recorded as not helping / worse
1. In-kernel amdgpu driver "can't handle 8 concurrent procs/GPU"; amdgpu-dkms has ~5.4 GB/hr host RSS leak (ROCm #5915) (ea372c6 runbook :150-152).
2. Kernel-cmdline gfxoff "silently ignored per TB incident report" (ea372c6 runbook :121-122) — later superseded: v1.2 puts it on the cmdline and calls modprobe.d redundant (commit 99f962e step 21-22). Contradiction in the record.
3. `nomodeset` (installer) breaks SMU telemetry (99f962e step 21).
4. 8 one-process-per-GPU workers with fine-grained launches on a 4-core CT: 8 GPUs slower than 4 (MEM rig-ct100-four-cores-eight-gpus.md:13-18).
5. Sustained multi-minute 8-GPU load took rrig6600 + host .121 down (2026-08-19); 5–16 s bursts fine (MEM rig-ct100-four-cores-eight-gpus.md:35-41).
6. In-container `dmesg` = vacuous pass (docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md:260-264); empty netconsole could not distinguish no-event from not-active (docs/CLAUDE_CODE_REPORT_DEFECT_A_TRANSPORT_RECOVERY.md:270-274); RAS counters unsupported on RDNA2 (MEM rig-ct100-kernel-log-access-gap.md:37-39).
7. Boot-notify fires once at boot only — "A GPU that drops off the bus after boot is invisible to it until the next reboot" (docs/FLEET_STATE_REQUIREMENTS_v1.md:244-246).
8. Rebooting a host = it returns as Ubuntu, indistinguishable from a hang (MEM rig-boot-selector-proxmox-recovery.md:8-26; docs/TFM_PROJECT_FACTS_SKILL.md:2948-2953).
9. PWC era (restated 2026-11-14): pool 8 + 50 ms cooldown crashed ~10 min; pool 8 at 50K/100K GCVM_L2 at ~23/~12 min; removing rrig6600b (2 rigs) "Still crashes"; S162–S170 "band-aided dispatch cadence (jitter, pacing, cooldown, transport swaps)" (docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STAGE_A.md:344, :346, :350; docs/TFM_PROJECT_FACTS_SKILL.md:2544-2546); `AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3` on rrig6600c did not delay the crash (S162 Option B, as catalogued: docs/PROJECT_FILE_CATALOG.md:131).
10. Brief-I generator-phase change ruled incapable of altering device dispatch (docs/RIG_CRASH_FORENSIC_20260822.md:130-131, :355) — "eliminated" as a cause of run 3.

### LIST 3 — proposed but never tried (or never recorded as done)
1. Warm-up / power-cap practice before 8-GPU simultaneous load (docs/PROPOSAL_Proxmox_LXC_Rig_Migration_v1_0.md:108-109; docs/PROPOSAL_Infrastructure_Reconciliation_S172_v1_0.md:126-127) — miner implements only launch stagger; no power cap recorded as applied in era D; then DROPPED by owner ruling 3 (docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:21).
2. READY handshake warming "sequentially or in pairs" (docs/S172_INFRASTRUCTURE_INTERFACE_v1_0.md:71-75) — delegated to the spawner (miner/range_miner_worker.py:1589-1592); the synchronous phase-boundary resume of all workers (docs/RIG_CRASH_FORENSIC_20260822.md:307) has no stagger recorded.
3. VM-passthrough Gate #1 incl. FLR reset test (docs/PROPOSAL_Infrastructure_Reconciliation_S172_v1_0.md:107-116) — no result recorded.
4. Root key auth VM101→Proxmox hosts for kernel logs (docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md:444-446) — never recorded as set up; noted working on 2026-11-15 (docs/CLAUDE_CODE_INSTRUCTIONS_STEP1_CREDIT_WAL_20261115.md:110).
5. Per-rig instrumented supply logging + circuit map (H2) (docs/RIG_CRASH_FORENSIC_20260822.md:356; docs/TODO_CONSOLIDATED.md:82) — OWED then dropped (E30).
6. GPU power caps + off-host power telemetry verification (docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:149) — "Unverifiable fleet-down" (docs/SESSION_CHANGELOG_20260828_RUN4_ROUTING_PATCH.md:274), then dropped.
7. ≥1 Hz per-host resource series (MemAvailable, load, sockets, rocm-smi) armed before dispatch across phase boundary; instrument the transition; explain the 15 staging failures; deliberate volume reproduction (docs/RIG_CRASH_FORENSIC_20260822.md:371-381) — reproduction removed by owner (docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:24).
8. CT100 MemTotal/MemAvailable/nproc/df for H3 (docs/RIG_CRASH_FORENSIC_20260822.md:357) — UNAVAILABLE; coordinator RSS sampler (docs/LEADS.md:161-164).
9. Switch counters / port topology for H5 (docs/RIG_CRASH_FORENSIC_20260822.md:359).
10. Netconsole reboot persistence (docs/RUNBOOK_NETCONSOLE_REARM.md:81-83, :151).
11. pip-freeze diff across the three CTs (Python 3.10.12 vs 3.10.4) (docs/TFM_PROJECT_FACTS_SKILL.md:2552-2558).
12. Baselining `/sys/module/amdgpu/parameters/*` (docs/CLAUDE_CODE_REPORT_DELEGATED_RUN_GATES.md:491-494).
13. Decide whether the dead ROCm prelude (HSA_ENABLE_SDMA) is supported; key it from a profile property (docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:1074-1080).
14. Node watchdog / auto-restart; rig crash monitors; `crash_forensic_daemon.py` fixes (docs/TODO_CONSOLIDATED.md:190, :282-283).
15. DPM harness 900 mV / 2100–2200 MHz persistence check (docs/TODO_CONSOLIDATED.md:281).
16. "consider whether sustained 8-way concurrency is safe on these chassis at all" (MEM rig-ct100-four-cores-eight-gpus.md:40-41).
17. PWC keep-workers-alive across trials (launch-storm removal) — design note only (docs/CLAUDE_CODE_INSTRUCTIONS_PWC_BACKUP_STEP1_20261114.md:83).
18. Supply-sag hypothesis: "not addressed, by owner ruling" (docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:259-260).

### LIST 4 — isolation results (era D; all RANGE-MINER over TCP unless noted; all rigs Proxmox host + unprivileged CT100 unless noted; AMD seed caps 2,000,000 constant / 1,000,000 hybrid; kernels java_lcg family)
| date · run | rigs/host type | cards/workers | geometry / kernels | duration | outcome | anchor |
|---|---|---|---|---|---|---|
| 07-08/09 rrig6600c install | Proxmox 8.4, privileged CT (v1.1) | 8 GPUs, framework smoke | CuPy matmul / torch / TF | not recorded | all 8 enumerate | commits ea372c6, 89b546c |
| 07-30 Phase 6.0 | rrig6600 CT .122 | 1 GPU (dev 0), 1 worker | java_lcg const fwd/rev, 8M seeds, cap 1M | 15.9 s trial; 7 smoke runs | CLEAN (root-free + host dmesg) | docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md:67-68, :230, :336-340; docs/TEAM_ALPHA_PHASE_6_0_RETURN_TO_BETA.md:69-80 |
| 08-05 hand launch | 3 rigs + Zeus | 25 workers (8/CT) | const | not recorded | coordinator staging failure; no host loss recorded | docs/TEAM_ALPHA_DEFERRED_QUEUE_NOTE.md:12-22 |
| 08-09 att 1 `689f3cd9` | rrig6600 only + Zeus | 7 rig + 1 Zeus = 8 | k=20 τ.45, phases 1–2 | ~10 min | lease defect; no GPU fault claimed | docs/CLAUDE_CODE_REPORT_GATE12_FAILURE_FORENSICS.md:4, :121-154; docs/S172_PHASE3_SURVIVOR_CAPACITY_CHARACTERIZATION.md:50 |
| 08-10 att 2 `abc63f71` | 3 rigs + Zeus | 24+1 | k=13 τ.47, phases 1–3, 44,331 p3 | ~81 min (09:23–10:44) | 23/25 admission; post-run 8/8×3 | docs/TB_GATE12_ATTEMPT2_FORENSIC_REPORT.md:5-34, :74-77 |
| 08-10 att 3 `d606edbe` | 3 rigs + Zeus | 25 | k=22 τ.47, 4 phases, 774 p3 | 17:28:31–19:48:14 | 128/128, 0 disconnects (publication refused) | docs/TFM_PROJECT_FACTS_SKILL.md:1873-1890 |
| 08-11 att 4 `c8939b64` | same | 25 | k=15 τ.31, to phase 2 | ~18 min | lease-origin defect | docs/TFM_PROJECT_FACTS_SKILL.md:1967-1970 |
| 08-12 att 5 `7e0d020b` | same | 25 | k=12 τ.46, 276,439 p3 | 17:38:44–18:59:53 | 2 workers (6600c:gpu2, 6600b:gpu3) dropped 18:27:10, cause unresolved | docs/TFM_PROJECT_FACTS_SKILL.md:2086-2127 |
| 08-15/16 att 6/7/8 | same | 25 | phase 2 | not tabulated here | coordinator lease expiry; no crash | commit 087b7f4; docs/TFM_PROJECT_FACTS_SKILL.md:2319-2331 |
| 08-17 att 9 `554463d3` | same | 25 (GPU 24/24) | k=12 τ.71, gen-phase 25; 59 p3 / 23,515 p4 | 738 s loop | PASSED, fleet exit clean | docs/TFM_PROJECT_FACTS_SKILL.md:2444-2456 |
| 08-18 throughput | each CT, GPU0 | 1 GPU | two families | best-of-2 | .122 12–15% faster | MEM rig-throughput-heterogeneity.md:11-14 |
| 08-19 PIECE-MATCHER (not TFM) | rrig6600 CT + host; b, c | 1/2/4/8 GPUs, 1 proc/GPU | chunk 2^22…2^29 | bursts 5–16 s OK; first multi-minute 8-GPU load | rrig6600 + .121 DOWN; b, c completed | MEM rig-ct100-four-cores-eight-gpus.md:8-41 |
| 08-22 run 1 `eed23c7f` | 3 rigs + Zeus | 25 | k=20 τ.64, gen-phase 0; 1 p3 / 0 p4 | serve loop 713.663 s | 128/128, 0 disconnects (Phase-5 defect) | docs/S172_BRIEF_I_PRODUCTION_SHAPE_CLASSIFICATION_REPORT.md:21-33 |
| 08-22 run 2 `35ba80aa` | same | 25 | k=22 τ.69, phase 1 only | not recorded | not recorded | docs/RIG_CRASH_FORENSIC_20260822.md:6 |
| 08-22 run 3 `8cb116da` | same | 25 | k=6 τ.35 M=3, gen-phase 0; 13,146,485 p3 | p3 234.2 s; p4 dies 1–6 s after 18:32:01 | rrig6600 + rrig6600c hosts silent; rrig6600b + Zeus OK; no netconsole packet | docs/RIG_CRASH_FORENSIC_20260822.md:303-349 |
| 11-15 Run 4 att 1 `57dc50fa` | same | 25 eligible each phase | k=12 anchor 25 τ.71/.47 gen-phase 0, seeds [2^31,2^32); 60 p3 / 23,568 p4 | 09:07:39–09:19:49 | CLEAN compute, 0 disconnects, rigs 8/8 after | docs/CLAUDE_CODE_INSTRUCTIONS_STEP1_CREDIT_WAL_20261115.md:50-63 |
| 11-15 Run 4 retry `ab6318d4` | same | not recorded | same pinned geometry (implied, not stated) | lost at phase-4 start | pve-rig6600 + pve-rig6600c: GCVM_L2 → 0xFFFFFFFF → device inaccessible → host hang | docs/CLAUDE_CODE_INSTRUCTIONS_GPU_FAULT_PRIOR_WORK_20261115.md:1 (untracked brief only) |
| PWC era (pre-July, restated 11-14) | bare-metal Ubuntu rigs | see table | SSH/TCP/ZMQ PWC | see table | pool 6/≤50K and 8/≤25K clean; 8/50K–100K GCVM; 2-rig still crashed; S158 3-rig kernel panic on 6600c | docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STAGE_A.md:322-378 |

Fleet-unreachable observations (no cause recorded, not counted as faults): all rig endpoints DOWN at 07-31 audit (docs/HYBRID_SKIP_BOUND_AUDIT.md:48); "No route to host" 08-09 sessions (docs/CLAUDE_CODE_REPORT_SEED_DOMAIN_CURSOR_R1.md:433); 08-27 CTs unreachable (docs/RIG_CRASH_FORENSIC_20260822.md:22-23); 08-28 fleet powered off (docs/SESSION_CHANGELOG_20260828_RUN4_ROUTING_PATCH.md:229-233); 11-13 rrig6600 No route (docs/CLAUDE_CODE_REPORT_DELEGATED_RUN_GATES.md:14-15).

### Other contradictions noted (not already above)
- LLM_INFRASTRUCTURE.md gives rig-6600c CT100 IP `192.168.3.192` and "only 7.7 GB host RAM", "one recorded crash was host-RAM exhaustion" (LLM stack) (docs/LLM_INFRASTRUCTURE.md:15-16, :26-28) vs CT .164 in CLAUDE.md §3 and MEM's 6 GB CT RAM.
- Chapter 9 "Required Settings: GFXOFF disabled … amdgpu.ppfeaturemask=0xffff7fff" and troubleshooting "Reboot rig if GPU shows persistent N/A" (docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md:922-924, :961-977) are legacy and `[UNVERIFIED]` on current rigs (:936-944).

### Files read
Repo (worktree): docs/RIG_CRASH_FORENSIC_20260822.md (full); docs/LEADS.md (full); docs/PROPOSAL_Proxmox_LXC_Rig_Migration_v1_0.md (full); docs/PROPOSAL_Infrastructure_Reconciliation_S172_v1_0.md (full); runbook at commits ea372c6/0afb88e/99f962e/6e047bd/89b546c/2cf23cf (full v1.0 + diffs); docs/RUNBOOK_NETCONSOLE_REARM.md (full); docs/NOTE_FOR_BETA_20260828_NETCONSOLE_EVIDENCE.md (full); docs/TEAM_ALPHA_PHASE_6_0_RETURN_TO_BETA.md (full); docs/TEAM_ALPHA_PROPOSAL_PHASE_6_0_ROCM_SMOKE.md (full); docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md (§§1-4, 9-14); docs/LLM_INFRASTRUCTURE.md (full). Read in part (at hits, in context): docs/TFM_PROJECT_FACTS_SKILL.md; docs/TB_GATE12_ATTEMPT2_FORENSIC_REPORT.md; docs/CLAUDE_CODE_REPORT_DEFECT_A_TRANSPORT_RECOVERY.md; docs/CLAUDE_CODE_REPORT_GATE12_FAILURE_FORENSICS.md; docs/S172_PHASE3_SURVIVOR_CAPACITY_CHARACTERIZATION.md; docs/S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md; docs/S172_RUN4_ROUTING_PATCH_BRIEF_ROUTE_A.md; docs/SESSION_CHANGELOG_20260828_RUN4_ROUTING_PATCH.md; docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md; docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md; docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md; docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md; docs/TODO_CONSOLIDATED.md; docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STAGE_A.md; docs/CLAUDE_CODE_INSTRUCTIONS_PWC_BACKUP_STEP1_20261114.md; docs/exchange/from_beta/X-PWC-BACKUP_R1/R2; docs/exchange/to_beta/X-PWC-BACKUP_R1/R2; docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md; docs/RUN4_LAUNCH_CHECKLIST.md; docs/CLAUDE_CODE_REPORT_RUN4_LAUNCH_PREP.md; docs/CLAUDE_CODE_INSTRUCTIONS_RUN4_LAUNCH_PREP_20261114.md; docs/CLAUDE_CODE_INSTRUCTIONS_STEP1_CREDIT_WAL_20261115.md; docs/CLAUDE_CODE_REPORT_STEP1_CREDIT_WAL.md; docs/SESSION_CHANGELOG_20261115_STEP1_CREDIT_WAL.md; docs/CLAUDE_CODE_INSTRUCTIONS_GPU_FAULT_PRIOR_WORK_20261115.md; docs/TEAM_ALPHA_PHASE7_LAUNCH_NOTICE.md; docs/PHASE6_PREREQS.md; docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_7_SOAK.md; docs/S172_PHASE_7_PREREQ_REPORT.md; docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_6_0_ROCM_PARITY.md; docs/CLAUDE_CODE_INSTRUCTIONS_MANUAL_FLEET_LAUNCH.md; docs/TEAM_ALPHA_DEFERRED_QUEUE_NOTE.md; docs/S172_STAGING_PART_B_REPORT.md; docs/S172_BRIEF_I_PRODUCTION_SHAPE_CLASSIFICATION_REPORT.md; docs/SESSION_CHANGELOG_20260815_S183.md; docs/SESSION_CHANGELOG_20260814_S182.md; docs/SESSION_CHANGELOG_20260822_*; docs/SESSION_CHANGELOG_20260817_GATE12_ATTEMPT9.md; docs/SESSION_CHANGELOG_20260820_SKILL_V27.md; docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md; docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md; docs/CHAPTER_9_AUDIT.md; docs/CLAUDE_CODE_REPORT_DELEGATED_RUN_GATES.md; docs/CLAUDE_CODE_INSTRUCTIONS_DELEGATED_RUN_GATES_20261111.md; docs/FLEET_STATE_REQUIREMENTS_v1.md; docs/S172_INFRASTRUCTURE_INTERFACE_v1_0.md; docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md (§12.4); docs/S172_SIEVE_PATH_VERIFICATION_SCOPE.md; docs/S172_PHASE4_BRIEF.md, docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE4.md, docs/BACKLOG.md, docs/CHAPTER_CORRECTIONS_REPORT_20261111.md (keyword hits only — no fault content); docs/TFM_SESSION_HANDOFF_20261113.md, docs/TFM_SESSION_HANDOFF_20261114.md (hits); docs/exchange/** (keyword sweep). Code: miner/range_miner_worker.py (:648-659, :1150, :1266-1268, :1589-1598, :2805-2891), sieve_gpu_worker.py:22-34, sieve_filter.py:20-30, scripts/launch_fleet_manual.sh:25-60, :300-345, scripts/gate12_gpu_gate.py:14-34. Git bodies: ea372c6 75ec134 0afb88e 99f962e 6e047bd 89b546c 2cf23cf 23fa413 d3b3cd8 8781af8 23b5055 4c76f42 acd6f13 2532803 e75a6eb 0066b6e 087b7f4 6bf89ef 8f0ca7b 49ff9b4 4643a11 + `git log --all --since=2026-07-01` grep. Keyword sweep over 350 Jul–Nov docs + all SESSION_CHANGELOG_202607*/08*/11* + docs/exchange (scratchpad lines3.txt).
Memory (read-only): rig-ct100-four-cores-eight-gpus.md, rig-ct100-kernel-log-access-gap.md, rig-boot-selector-proxmox-recovery.md, rig-throughput-heterogeneity.md, rig-gpu-sku-is-rx-6600-xt.md, MEMORY.md.

### Could not read / not searched
- External repo `proxmox-gpu-passthrough` (where the runbook moved, 2cf23cf) and `rx6600-llm-inference` RUNBOOK (docs/LLM_INFRASTRUCTURE.md:7-12): not in this tree.
- MEM note `rig-crash-2026-08-22-hang-not-power` (referenced, file absent).
- `~/dashboard_work/` (D6 reports, e.g. D6_I1_I2_REPAIR.md), `/home/michael/run4_attempt1_20261115_090553/`, logs/ (netconsole_all_rigs.log, gate12_*.log), rig `/tmp/minerlogs`, ser8 archives, private repo: not read (outside brief / no SSH).
- `/home/michael/tfm_worktrees/rig-gpu-fault`: not touched, by instruction. The Run 4 retry (ab6318d4) evidence is there or on hosts, not in this worktree.
- No SESSION_CHANGELOG exists for 2026-09 or 2026-10 (ls: none); no session changelog for the 08-22 run-3 incident itself.
- Pre-July PWC/S159G/S162/S163KARG documents were not re-read; cited only via the Nov-14 Stage A report and PROJECT_FILE_CATALOG.

---

## Part E — non-docs surfaces (apply/verify docstrings, git log, main-tree gitignored files and logs, dashboard_work)


No root cause is concluded anywhere below. "Belief" = what the source said at the time.
Hosts: .120/.122/.121 = rrig6600 (bare/CT/host), .154/.156/.155 = rrig6600b, .162/.164/.163 = rrig6600c.

### NULL RESULTS FIRST
- Surface 4 (~/dashboard_work, 131 files): **0 of 131** files record a GPU fault, GPU reset, amdgpu message or env/kernel mitigation. Only 2 files touch the subject at all (a rig reboot side-effect and an interpreter divergence). The ATTEMPT4/5/6, GATE12_MP1, D6_DRYRUN forensics are about admission, lease and drain defects. They are not about GPU faults.
- S174 bundles: **23 of 23** `final_dmesg_tail.txt` captures are 0 bytes. The bundles hold no kernel-log evidence.
- Netconsole: from 2026-05-15 to 2026-08-22, **0** kernel datagrams were received from any rig. The listener restarted daily ("LISTENER STARTED" x135), but no sender was armed. Runs in that window have no kernel-log evidence: Phase 6/7, Gate-12 attempts 1-9, and the 2026-08-19 `.121`/`.122` incident.
- Telemetry `logs/rig_telemetry/rig_power.tsv` (6024 rows) covers only 2026-08-23 02:52-03:33Z, with all rigs idle (38-49 W per host). It holds **no load transient**.

### SURFACE 1 — apply_s*/verify_s* (127 files; 38 concern the subject, 89 do not)
| file:line | session | applied / verified (exact) |
|---|---|---|
| apply_stagger_fix.py:3-6,37-49,140-141 | 2026-01-16 (TB) | scripts_coordinator staggered GPU worker start: localhost 3.0 s (STAGGER_LOCALHOST), remote 0.5 s (STAGGER_REMOTE). Belief: HIP init collision. (a47ba88 says this was reverted. f614736 later set STAGGER_REMOTE 0.5->2.0) |
| apply_sieve_cleanup_patch.py:3-8 | 2026-01-26 (TB) | gc.collect() in _best_effort_gpu_cleanup + inter-chunk cleanup in both forward sieve loops (commit 20b0e7c "prevents Step 1 hangs") |
| apply_s136_doc_updates.py:249-259 | S136 | docs: spawn stagger 4.0 s per gpu_id; pool cap worker_pool_size (stable at 4); HSA_OVERRIDE_GFX_VERSION=10.3.0, HSA_ENABLE_SDMA=0 "reduce HIP init crashes" |
| apply_s137_seed_cap_argparse.py:3-13 | S137 | --seed-cap-nvidia/--seed-cap-amd argparse wiring (seed-cap plumbing) |
| apply_s149a_rocr_isolation.py:5-18 / verify_s149a_rocr_fix.py:11-19 | S149-A | ROCR_VISIBLE_DEVICES={gpu_id} in _spawn_worker; verify keeps spawn stagger 4.0 s |
| apply_s149b_device_gpu_id.py:7-33 / verify_s149b_device_fix.py:7-20 | S149-B | Removes per-worker HIP/CUDA_VISIBLE_DEVICES and uses cp.cuda.Device(gpu_id). Keeps HSA_OVERRIDE_GFX_VERSION and HSA_ENABLE_SDMA. "ROCR_VISIBLE_DEVICES=1 -> hipErrorNoDevice" |
| apply_s151_respawn_stagger.py:5-13 | S151 | per-node respawn Lock + ROCM_SPAWN_STAGGER_S between respawns. Belief: "SSH hammer that crashes rrig6600c" |
| apply_s152_cupy_cache_dir.py:7-37 | S152 (TB) | CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_{gpu_id}. Belief: concurrent kernel-cache write race "crashes ... bringing down the entire rig" |
| apply_s152_heartbeat_timeout.py:5-30 | S152 | WORKER_HEARTBEAT_TIMEOUT_S 30->90 (8-GPU ROCm init on reboot >30 s; 4 s stagger) |
| apply_s152_slim_v1_vram_fix.py:7-39 | S152 | skip_sequences alloc n_seeds*k (14.5 GB) -> PRNG_MAX_HYBRID_SURVIVORS (default 5000)*k = 361 MB. "8 workers x 14.5 GB = instant hard crash" |
| apply_s152_ssh_keepalive.py:7-24 | S152 | ServerAliveCountMax=10, ConnectTimeout=10 (all 8 rig workers dying at once) |
| apply_s154_gpu_del_fix.py:3-16 | S154 | explicit del of GPU arrays + gc.collect(). kern.log "Out of memory: Killed process 3008 (python) total-vm:41452828kB" with CUPY_CUDA_MEMORY_POOL_TYPE=none, 7.7 GB RAM rig |
| apply_s156_pwc_prestartup_cleanup.py:5-22, apply_s156_pwc_cleanup_v2.py:5-10 | S156 | pre-spawn kill of stale sieve_gpu_worker on rigs (v1 pkill -9; v2 SIGTERM first). Belief: zombie workers after a rig reboot mid-run -> py_procs 12+ -> "PageTable overflow crashes"; "TEMPORARY SAFETY BANDAID" |
| apply_s158d_b_systemd_run_launcher.py:5-22,98-100 | S158D-B | systemd-run --user per worker, `--setenv ROCR_VISIBLE_DEVICES`, `CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_N`, `HSA_OVERRIDE_GFX_VERSION=10.3.0`, Restart=always RestartSec=2; loginctl enable-linger on all 3 rigs |
| apply_s158d_e_cuda_mask_hardening.py:5-11 | S158D-E | Zeus CUDA mask setdefault; logs CUDA/ROCR_VISIBLE_DEVICES |
| apply_s160_worker_env_check.py:5-18,36-42 | S160 (TB) | /proc/<pid>/environ check 45 s after launch; halts the run unless the worker has HSA_ENABLE_SDMA=0, HSA_ENABLE_RUNTIME_POWER_MGMT=0, AMDGPU_NO_POWER_PROFILE=1, HSA_OVERRIDE_GFX_VERSION=10.3.0, ROCR_VISIBLE_DEVICES |
| apply_s161_tcp_launch_workers.py:5-12,76-80,108-110; apply_s161_tcp_launch_v2.py:5-9 | S161 | TCP-PWC nohup launch: ROCM_ENV_VARS + ROCR_VISIBLE_DEVICES + per-GPU CUPY_CACHE_DIR, ROCM_SPAWN_STAGGER_S between GPUs |
| apply_s161_min_workers_v2.py:3-13,25-39 | S161 | TCP stagger 4 s -> 1 s (TCP_SPAWN_STAGGER_S=1.0), ROCM_READY_TIMEOUT_S=110, --min-workers gate |
| apply_s161_sequential_launch.py:3-6 | S161 | launch one GPU, wait for it to connect, then the next ("prevents simultaneous ROCm context competition") |
| apply_s161_heartbeat_timeout.py:3-9,21-22 | S161 | WORKER_HEARTBEAT_TIMEOUT_S 90->120 (~90 s per GPU sequential ROCm init on rrig6600) |
| apply_s161_v2_coordinator.py:3-20 | S161 | two-phase online/init/ready startup |
| apply_s162_dispatch_semaphore.py:3-24 | S162 | threading.Semaphore(8) on job sends. Belief: 24 concurrent sends saturate Zeus 1GbE I210 -> rrig6600c stalled payloads -> "GPU page fault -> cascade crash". Zeus sysctl wmem/rmem max 16777216, default 1048576, tcp_wmem/rmem "4096 1048576 16777216" |
| apply_s162_safe_cleanup.py:3-26 | S162 | replace TCP-worker _best_effort_gpu_cleanup with gc.collect() only. netconsole "GCVM_L2_PROTECTION_FAULT_STATUS:0x00801231, SQC (inst)". Belief: free_all_blocks unmaps live instruction pages on rrig6600c only |
| apply_s162_option_b.py:3-33 | S162 (TB) | AMD_SERIALIZE_KERNEL=3, AMD_SERIALIZE_COPY=3 on host 192.168.3.162 only; "DIAGNOSTIC, not permanent" |
| apply_s162_exp1_scratch_reclaim.py:3-38 | S162 (TB Exp 1) | HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0 on .162 only. Records: Option B "did not prevent the crash"; crash starts with SQC(inst) faults on two GPUs at once -> KFD queue collapse -> CPU soft lockup. Next step: Exp 2 HSA_NO_SCRATCH_RECLAIM=1 |
| apply_s163_kernel_arg_typing.py:3-45 | S163-KARG | cp.int32() for n_seeds,k,skip_min,skip_max in the non-hybrid run_sieve. Fault signature quoted: 0xFFFFFFFF, client 0x1ff, WALKER_ERROR 0x7. "S163 FINAL validated Apr 12-13 ... fails Apr 17" |
| apply_s163_mem_debug_worker.py:3-13 | S163 (TB Opt B) | removes free_all_blocks() from sieve_gpu_worker _best_effort_gpu_cleanup (the S155 256 MB pool cap makes it redundant; "race condition source"); S163_MEM_DEBUG pool sampling |
| apply_s163_tcp_worker_mem_debug.py:3-18 | S163 | same MEM_DEBUG sampling in pwc_worker_service (TCP path) |
| apply_s165_zeus_tcp_worker.py:3-18 | S165 | Zeus persistent CUDA workers (context; not an AMD mitigation) |
| apply_s174_ready_gate_fix.py:3-28 | S174 | hard ready gate (ready<min_workers raises) |
| apply_s142_worker_retry.py:12-15 | S142 | 0.1-0.5 s random-jitter retry (SQLite only; not GPU) |
| apply_s152_incremental_npz_flush.py, apply_s149_npz_checkpoint.py | S152/S149 | crash-loss protection for survivors (context only) |
Not related to the subject (89 files): the scorer, feature, doc, dashboard, NPZ, Optuna and threshold patches. Their headers were read.

### SURFACE 2 — git log (942 commits; date-ordered relevant set; values from `git show`)
| sha | date | what | kept/reverted |
|---|---|---|---|
| deb8788 | 2025-11-29 | GPU isolation, ROCm HIP binding; python_env tf->torch | — |
| 41165dd | 2025-12-24 | lazy torch import prevents ROCm/CuPy kernel-compile failure | — |
| 90b6789 | 2026-01-03 | stagger SSH connections on all nodes | — |
| a47ba88 | 2026-01-16 | 50 trials crashed both ROCm rigs (froze, rebooted twice). Stagger fix tried then reverted (scripts_coordinator restored from backup). _best_effort_gpu_cleanup added to scorer worker. "MaxSessions may need 50" | stagger reverted in-commit |
| f614736 | 2026-01-17 | GFXOFF off via amdgpu.ppfeaturemask=0xffff7fff on rig-6600/6600b ("gfxoff=0 kernel param ignored on 6.8.0-90"). atexit cleanup in 3 workers. STAGGER_REMOTE 0.5->2.0 s. max_concurrent_script_jobs 12->4 (.120,.154). Belief: GFXOFF SMU race | concurrency undone by a490ea0. ppfeaturemask still on the host cmdline 2026-08-22 (S3) |
| a490ea0 | 2026-01-18 | max_concurrent_script_jobs 4->12; sample_size 450; belief "host memory pressure, NOT HIP init" | — |
| a82860f | 2026-01-20 | "SMU exhaustion crash": removed `watch -n 0.5 rocm-smi`; rocm-fan-curve.sh LOOP_SECS 3->20 (500+ -> 36 SMU ops/min) | [UNVERIFIED] current |
| 3e6b224 | 2026-01-23 | Step 3 GPU failures = Linux OOM on rig-6600 (7.7 GB RAM, 7x1.5 GB). HIP cache clear kept. Warm-up barrier "didn't fix". setdefault HSA_OVERRIDE | — |
| 20b0e7c | 2026-01-26 | inter-chunk GPU cleanup prevents Step 1 hangs | — |
| 4d979e0 | 2026-01-31 | gpu_count 12->8 (.120,.154); .162 added gpu_count 8 | kept |
| 66620be | 2026-02-11 | fix_gpu_rigs.sh: udev perf=high on GPU detect; GFXOFF disable via GRUB; all 3 rigs | — |
| 89c7597 | 2026-03-08 | persistent workers get HSA_ENABLE_SDMA=0, HSA_OVERRIDE_GFX_VERSION=10.3.0 | kept (persistent_worker_coordinator.py:140-141) |
| 7288e5e | 2026-03-08 | persistent spawn stagger gpu_id*2.0 s (libamdhip64 segfaults) | later ROCM_SPAWN_STAGGER_S=4.0 (:116) |
| d168f83 | 2026-03-08 | seed caps 40k/19k -> 5M/2M | — |
| 288de2a | 2026-03-09 | restore semaphore throttle to persistent dispatch: "fixes rig-6600c crash" | — |
| d5219a0 | 2026-03-09 | revert S130 persistent workers (S134) | reverted feature |
| c1c698d | 2026-03-20 | S149b Device(gpu_id), no HIP/CUDA masking | — |
| 38393b5 | 2026-03-21 | .162 gpu_count 8->6 "weaker CPU ROCm queue limit" | reverted by cc113b5 |
| cc113b5 | 2026-03-21 | .162 6->8 "GPU7 fault was hardware" | superseded by e4905c2 |
| e4905c2 | 2026-03-21 | .162 8->4 "PCIe 1x stability limit" | reverted by fd38d17 |
| 12dbeaf | 2026-03-21 | per-node respawn lock + stagger ("PCIe 1x rigs") | — |
| fd38d17 | 2026-03-21 | "root cause rrig6600c — MaxSessions 5 in sshd_config", .162 4->8; docs/ROOT_CAUSE_ANALYSIS_RRIG6600C_S151.md | — |
| 701c501 | 2026-03-22 | heartbeat 30->90 s | 120 s at HEAD (:153) |
| ae6afb2 | 2026-03-22 | slim_v1 VRAM fix (PRNG_MAX_HYBRID_SURVIVORS) | — |
| 7182ab5 | 2026-03-23 | S154 explicit del + gc | — |
| 5d8d10d / c5af655 | 2026-03-23 | S155: removed CUPY_CUDA_MEMORY_POOL_TYPE=none (41 GB VM mmap OOM). set_limit pool cap; PRNG_CUPY_POOL_LIMIT_MB=256; CUPY_GPU_MEMORY_LIMIT=268435456 | kept (:130,:137) |
| 52f16bc | 2026-03-23 | .162 gpu_count+max_concurrent 8->4 "pending M.2 SATA replacement" | reverted by 8b04050 |
| 8b04050 | 2026-03-24 | .162 4->8 "new M.2 SATA installed, axGear adapter removed" | superseded |
| d4eb55d | 2026-03-26 | .162 8->3 "stable post-BIOS-reflash" | reverted by 7ea1783 |
| 0577c53 | 2026-03-27 | S156 SIGTERM-first targeted cleanup | — |
| 8d62491 / c9e7221 / d1301e1 | 2026-03-29 | ROCM_CLIFF_NODES=["192.168.3.162"], ROCM_INTER_NODE_STAGGER_S 10 -> 30 s | **reverted** d1301e1 "caused spawn crashes" |
| 4afcdc7 / e454fbc / 2d5db89 | 2026-03-29/30 | ZMQ+SQLite coordinator; systemd-run launcher --setenv HSA_OVERRIDE_GFX_VERSION=10.3.0 | ZMQ default a30f5dc; later superseded by PWC-TCP (6432a48) |
| 6681d4b | 2026-03-31 | worker setdefault HSA_ENABLE_SDMA=0; Zeus worker stagger | — |
| 561428b | 2026-04-01 | ZMQ launcher +--setenv HSA_ENABLE_SDMA=0, HSA_ENABLE_RUNTIME_POWER_MGMT=0, AMDGPU_NO_POWER_PROFILE=1 ("prevent D3hot crash") | in zmq_sqlite_coordinator.py:524-527 only. **Absent** from PWC ROCM_ENV_VARS (:119-150) and from the miner launcher |
| bb65aa9 / 9e09af7 | 2026-04-02 | netconsole deployed; rrig6600 crash TB submission | — |
| 55b2645 / 4fcfa51 | 2026-04-03 | env-invariant check, auto-halt | — |
| 050581b | 2026-04-04 | TCP_SPAWN_STAGGER_S=1.0, ROCM_READY_TIMEOUT_S=110, min-workers | kept (:117-118) |
| 2fa4851 / 64d0307 | 2026-04-04 | _best_effort_gpu_cleanup in TCP worker between chunks ("page fault -> qcm timeout -> rig reboot") | kept (persistent/pwc_worker_service.py:462-463) |
| 135a8ba | 2026-04-04 | dispatch Semaphore(8) in pwc_transport_tcp; S162 proposal | kept (pwc_transport_tcp.py:160,540) |
| 58b2155 | 2026-04-05 | checkpoint: AMD_SERIALIZE_KERNEL/COPY=3 and HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0 (.162 only); .120 max_concurrent 8->0 | reverted 7278db6 |
| d79e558 / 070199d | 2026-04-05 | disable CuPy pool (raw hipMalloc) | **reverted** "pool NOT the cause, made things worse" |
| 606f94f / a854647 | 2026-04-05 | HSA_NO_SCRATCH_RECLAIM=1 all AMD rigs | **reverted** "made things worse" |
| 22bb145 / ac32cd9 | 2026-04-05 | pre-compile all PRNG kernels at worker start | **reverted** "made crashes worse" |
| 7278db6 | 2026-04-05 | restore to 89c1512: all diagnostic patches reverted, "DKMS baseline"; .120 0->8 | — |
| 89c1512 / 47ab865 / f837f21 | 2026-04-05 | belief: root cause amdgpu-dkms 6.12.12; "26-GPU stable", 887 survivors; rigs pinned | — |
| 03d9013 -> 68ed075 -> 3c20d2d -> ad4aa59 | 04-06 -> 04-18 | free_all_blocks: removed from sieve_gpu_worker, restored there and removed from sieve_filter, removed from sieve_gpu_worker again (cudaErrorIllegalAddress race), **restored in sieve_gpu_worker by ad4aa59** | HEAD: absent from sieve_filter.py:148-150; **present** in sieve_gpu_worker.py:91-92 and miner/range_miner_worker.py:1069-1070 (dbe3d0e) |
| d9971d7 | 2026-04-14 | startup pool flush ("eliminates rig reboot between runs") | — |
| 0bfc391 | 2026-04-13 | 300K seed cap "validated stable"; .154 max_concurrent 0->8 | — |
| fd534de | 2026-04-16 | snd_hda_intel power_save=0 + amdgpu gfxoff=0 live on all 3 rigs, initramfs rebuilt | [UNVERIFIED] current |
| d32c1d1 | 2026-04-18 | cp.int32 kernel-arg typing; TB incident report; GDM/Xorg finding | kept |
| ad4aa59 | 2026-04-18 | worker heartbeat instrumentation; crash_forensic_daemon v2 "ser8-local, not in repo" | — |
| c4698c8 / c7f00a6 | 2026-04-19 | DPM pin service all 3 rigs (manual perf level); gpu-enum-heal.service and rocm-perf-auto cron removed | DPM pin disabled c5832bf |
| 884b424 | 2026-04-19 | reconnect 2->5 s x60 (5 min) to survive a rig reboot | — |
| b851511 | 2026-04-19 | amdgpu audio=0, dc=0, lockup_timeout=30000, gpu_recovery=1 all 3 rigs; Kaspa OC 2250 MHz/-150 mV on all 24 AMD GPUs; "KFD preemption failure root cause" | OC reverted c5832bf; lockup_timeout and gpu_recovery still on the host cmdline 2026-08-22 (S3) |
| c5832bf | 2026-04-20 | HSA_ENABLE_DEBUG_TRAP=0 ("KFD NULL ptr crash under concurrent reset+init"); Kaspa DPM reverted ("ASIC reset failures"); amdgpu-dpm-pin.service disabled; stock high DPM "zero netconsole errors" | DEBUG_TRAP kept (:145,:387) |
| b6cabe9 | 2026-04-24 | S168 PRNG_PWC_FIRST_ASSIGN_JITTER_SEC, S169 PRNG_PWC_PER_WORKER_MIN_GAP_SEC (default off). 100k cap "appears stable", 200k unstable; GCVM still observed | code kept (pwc_transport_tcp.py:466-497) |
| 7313a43 | 2026-05-03 | S170 live GPU probe, S171 coord throttle patch script, S173 first-fault manifests (2026-05-03: rrig6600c gpu6 + rrig6600 gpu0) | throttle and cooldown code **not** at HEAD (only patch_s171_coord_dispatch_throttle.py) |
| 742a255 | 2026-05-02 | ROCm saturation report | — |
| ca06f8c | 2026-05-08 | hard ready gate; FT0.73 crash repro launcher | — |
| d00e075 | 2026-05-09 | VALID_POST_COMPLETION_FAULT + cwsr hypothesis | — |
| b78d76c | 2026-05-11 | FAULT_KEYWORDS += SMU strings (response:0xFFFFFFFF, TransferTableSmu2Dram, GetEnabledSmuFeaturesHigh/Low, ppfeatures) | — |
| 3fdf434 | 2026-04-30 | "rrig6600b node restored", 26-GPU (the distributed_config diff touches only window_size.min 2->6) | — |
| ea372c6 / 0afb88e / 89b546c | 2026-07-08/09 | Proxmox + amdgpu-dkms rig runbook; CT100 build; ROCm userspace restore | — |
| dbe3d0e | 2026-07-18 | RANGE-MINER daemon (~52 launches/trial vs PWC "~17K/trial GCVM_L2 crash") | — |
| 23fa413 | 2026-07-30 | ROCm parity smoke, "no GPU faults" | — |
| f255912 | 2026-08-02 | localhost gpu_count 2->1 | — |
| 23b5055 / 8781af8 / 727362b | 2026-08-02 | kernel-log observability not established -> GCVM_L2 criterion "UNAVAILABLE" | — |
| 8f0ca7b | 2026-08-04 | brief: first 8-workers-per-rig miner launch (S157 JIT race, S155 VA space) | — |
| e75a6eb / a15b524 | 2026-08-28 | crash-forensic correction; netconsole runbook; L-3 OPEN; 18:42 freeze UNDETERMINED | — |

### SURFACE 3 — gitignored files, main tree (455 entries -> 81,677 files; 79,907 text files scanned; 106 >5 MB skipped; binaries excluded; 48 hit files)
1. **logs/brief_i_evidence/RIG_CRASH_FORENSIC_20260822.md (472 lines, mtime 2026-08-22 20:04) is a DIFFERENT document from docs/RIG_CRASH_FORENSIC_20260822.md** (diff at line 1). It is not in docs. It records:
   - :9-21 Michael found rrig6600 and rrig6600c POWERED ON but UNRESPONSIVE. "Power loss" was withdrawn and the verdict is UNDETERMINED.
   - :23-48 Hypothesis: PSU/12 V rail sag on one shared 240 V circuit at the phase-4 ramp. Confirm/refute criteria are stated.
   - :64-123 Host journals, previous boot: pve-rig6600 ends 18:27:00, pve-rig6600c ends 18:19:00, both on routine auth lines; 0 shutdown lines. Next boot shows "system.journal corrupted or uncleanly shut down" on both dead hosts and not on the survivor. **0 runtime amdgpu/thermal/MCE/AER/OOM events**. The host cmdline contains `pci=noaer`, `amdgpu.ppfeaturemask`, `amdgpu.lockup_timeout=30000`, `amdgpu.gpu_recovery=1`.
   - :85-94,393-397 journald SyncIntervalSec unset (5 min default), so the journal ends are not the death times (JOURNAL-SYNC-INTERVAL-1 OPEN).
   - :127-151 Death at 18:32:06/07, 5-6 s after the phase-4 stripes (18:32:01). rrig6600 delivered 7 of 544 phase-4 shards; rrig6600c claimed 0 phase-4 stripes; rrig6600b took 1020 shards and survived.
   - :122,280 cite a prior "2026-08-19 .121/.122 signature" from sustained 8-GPU load. That date appears nowhere in docs/.
   - :365-386 netconsole re-armed 19:25:50 (NC-TEST3). /etc/modules-load.d and /etc/modprobe.d/netconsole.conf written. **console_loglevel=3 on every host, so only KERN_ERR and above is transmitted.**
   - :401-435 boot-selector rule; Alpha's unauthorised reboot of .155 on 2026-08-23.
   - :441-462 measured power1_cap ceilings: pve-rig6600 145 W x3 + 135 W x5 = 1110 W (cap_max 174/162, cap_min 136/126); pve-rig6600b and -c 100 W x8 = 800 W (cap_max 120, cap_min 94). Recorded as a "different power table (VBIOS)".
   - :464-470 four breaks in the "identical clones" claim (cwsr modparams; Python 3.10.12 vs 3.10.4; mixed-vintage source; power tables).
2. logs/brief_i_evidence/S172_BRIEF_I_RUN3_REPORT.md:13,67-75. Run 3 verdict "power-class suspected" (superseded by item 1).
3. logs/brief_i_evidence/parked_for_post_run_commit/BACKLOG.md.with_staggered:495-530 and BACKLOG_staggered_phase4.patch. **STAGGERED-PHASE4-DISPATCH-1** (1-2 s per-host phase-4 dispatch offset, NOT IMPLEMENTED, conditional on sag). It says "the cap applied on 2026-08-23 is an INVESTIGATION instrument". This item is not in docs/BACKLOG.md.
4. logs/brief_i_evidence/parked_for_post_run_commit/rig_power_collector.py (UDP :6668 collector -> logs/rig_telemetry/rig_power.tsv). The tsv covers 2026-08-23 02:52:14-03:33:43Z, idle only: .121 46-49 W, .155 38-45 W, .163 45-48 W total.
5. **Netconsole captures**: logs/netconsole_listener.log (10,127 lines, 2026-04-17 -> **2026-11-15 14:03**), logs/netconsole_all_rigs.log (851 lines, 2026-05-11 -> 2026-11-15), logs/rrig6600c_netconsole.log (402 lines, 2026-04-01), logs/netconsole_all_rigs.pre_D1_1_20260511_164057.log and .pre_D1_2_pool6_20260511_192238.log, plus S174 bundle copies. GCVM burst onsets (gap >60 s), first status in each burst:
   - 04-01 20:06:49 .162; GPU reset begin 20:06:58; D3hot->D0 "device inaccessible" 20:06:57
   - 04-17 10:36:53 .120+.162 at the same second (SMU GetEnabledSmuFeaturesHigh 0xFFFFFFFF); 14:34:25 .162 0x00801231; 16:21:02 .154 0x006012B1 (77 sdma0/sdma1/gfx ring timeouts 11:02-17:23); 16:33:45 .120; 17:21:58 .154 0x002012B1; 21:16:45 .162
   - 04-18 19:04:32 .120+.162 SMU TransferTableSmu2Dram 0xFFFFFFFF (no GCVM)
   - 04-19 15:12:52 .120 0x008012A1; 17:39:30 .120 Mode1Reset 0xFFFFFFFF, "GPU mode1 reset failed", "ASIC reset failed -121", kworker hung 122 s
   - 04-20 **17:06:40 .120+.154+.162 at the same second, all 0x00841051**; 17:07:18-20 NULL deref Oops on .162 and .120, "GPU reset(1) succeeded"; **18:51:19 all three again** (.162 MODE1 reset x11 "VRAM is lost"; .154 "ASIC reset failed -22" x8); 19:17:07 .154; **19:38:40 all three again**; .154 hung tasks 122/245/368 s
   - 04-21 17:33:37 .120 Mode1 reset failed -121; 18:13:59 NULL deref
   - 04-23 15:52:32 .162 0x80081071 then hard LOCKUP cpu5 15:52:58; 16:42, 17:13, 18:11 .162
   - 04-24 17:53, 21:02, 21:39 .162; 21:25:53 .120 0x00801031 (kfd_evict_process_device trace)
   - 04-25 08:53, 17:19, 22:17 (0x00801431), 22:36 .162
   - 04-26 07:47, 15:42, 19:11 .162
   - 04-28 16:33:03 .162 0x008C1071; **20:26:42 .120+.162 at the same second** 0x00801431
   - 05-01 **15:55:07 .120 (0x008012B1) + .162 (0x00801431) at the same second**; .162 hard LOCKUP cpu4 15:55:16; .120 hung tasks; 22:04:32 .162
   - 05-02 22:55, 23:11 .162; 05-03 10:24 .162; 05-04 18:17 .154 SMU only; 05-05 18:18:55 .162
   - 05-11 17:52:20 .162 0x80001431, then SMU failures on .120+.162 at 17:52:34
   - **11-15 (today)**: 09:06:08-09:07:44 .121 ata5 BadCRC/link resets and sda I/O errors (link limited to 3.0 Gbps). **13:21:29 .121 (0000:19:00.0, 0x00841011, TCP) + .163 (0000:16:00.0, 0x00801231, SQC inst; 0000:06:00.0) at the same second**, then 0xFFFFFFFF/0x1ff WALKER_ERROR 0x7. 13:21:30/31 snd_hda D3hot->D0 "device inaccessible" on .121 0000:19:00.1 and .163 0000:06:00.1. **13:47:28 .155 0000:19:00.0 SMU TransferTableSmu2Dram 0xFFFFFFFF "Failed to export SMU metrics table"**. .121 reboot about 13:52; "amdgpu: Overdrive is enabled" at boot (13:55:46); NCPROOF-afternoon. Counts today: .121 20 GCVM lines (1 device), .163 26 lines (2 devices).
6. logs/S174_D1_FT073_50K_425M_20260511_175133_summary.txt (pool 8, min_workers 24, b78d76c): **VALID_ACTIVE_FAULT**, .120 and .162 unreachable after the run. logs/S174_D1_FT073_50K_425M_20260511_192244_summary.txt (**pool 6, min_workers 20**): **VALID_CLEAN**, all reachable. ..._20260509_222453_summary.txt: VALID_CLEAN with .120 unreachable (BETA reclassified it as VALID_POST_COMPLETION_FAULT in the docs changelog). The 05-11 runs are absent from docs/SESSION_CHANGELOG_20260509_S174_D1.md.
7. Gitignored launch scripts (2026-05-03): launch_pool6/7/8_*, launch_pool8_cooldown50.sh and cooldown500.sh (PRNG_PWC_PER_WORKER_COOLDOWN_MS=50/500), launch_pool8_100k_coord_throttle100.sh (PRNG_PWC_COORD_DISPATCH_THROTTLE_MS=100), launch_s173_*. All of them set PRNG_PWC_FIRST_ASSIGN_JITTER_SEC=15 and PRNG_PWC_PER_WORKER_MIN_GAP_SEC=0.20. Recorded outcomes in their headers: pool6/50k stable (29:30, 5 trials), pool6/100k stable "per operator", pool8/100k crash at 12:00 (rrig6600c device loss), pool8/50k crash about 23 min, pool8/50k/t1 clean, pool8/25k clean 1:30:43 (launch_pool8_50k_pool8.sh:5-10, launch_s173_v2_baseline_pool8_25k.sh:7-12, launch_s173_validation_t5.sh). The matching logs (s170_pool8_cooldown*_*.log, s171_pool8_100k_coord_throttle100_t5_1632.log, s171_pool7_*) contain no "Bayesian optimization complete". Netconsole GCVM fell inside the throttle100 run (04-28 16:33:03 .162) and inside the pool7_clean run (04-28 20:26:42).
8. Other: logs/crash_forensic_daemon.log (the daemon is not present on VM101; it ran on ser8); logs/s112_steps1_2_real_data.log:6902 (worker "Aborted" with the HSA env, 2026-02-26); logs/zmq_iso_rrig6600_v2/v3, zmq_2rig_6600_6600c (2026-03-30 launch-quoting errors); persistent_worker_coordinator.py.s170/s171/s173 backups (env identical to HEAD ROCM_ENV_VARS); logs/beta_relay/* (quotes of docs); data/prod_shape_scoping/* ("page faults" = /usr/bin/time output, not GPU).

### SURFACE 4 — /home/michael/dashboard_work (131 files, 3.1 MB)
- D6_I1_I2_REPAIR.md:8-10,341-352,400. The rigs had been rebooted/power-cycled about 50 min before (2026-08-15, D6-I1/I2 session). The reboot destroyed /tmp/minerlogs on all three CTs. Proxmox-host kernel logs were unavailable (no root key).
- MP1_FLEET_DEPLOY_RECORD.md:38-53. rrig6600c ~/rocm_env runs Python 3.10.12; rrig6600 and rrig6600b run 3.10.4; cupy 13.5.1 on all three. It cites "the GCVM_L2 fault chase". pip-freeze diff not done.
- ATTEMPT5_FORENSIC_P0_P3.md:83-120,359-360. Lost workers rrig6600c:gpu2 and rrig6600b:gpu3 (WORKER_DISCONNECTED 18:27:10). Cause UNRESOLVED; host kernel logs unavailable. No GPU-fault evidence either way.

### ONLY ON THESE SURFACES (checked with /bin/grep -rn in docs/; 0 docs hits unless noted)
1. The host-journal forensic version of RIG_CRASH_FORENSIC_20260822 (logs/brief_i_evidence): powered-on-but-unresponsive observation, unclean-journal markers, host cmdline `pci=noaer`, SyncIntervalSec gap, console_loglevel=3, power1_cap ceilings (1110 W vs 800 W), shared 240 V circuit, "2026-08-19 .121/.122" prior incident, and RIG_SAG_INSTRUMENTATION_20260823.md (the file is not found anywhere under /home/michael to depth 6).
2. STAGGERED-PHASE4-DISPATCH-1 and "cap applied 2026-08-23" (parked backlog). Whether the 2026-08-23 power cap is still in force is [UNVERIFIED]; sysfs caps do not normally persist across reboot.
3. rig_power.tsv 1 Hz telemetry and rig_power_collector.py (port 6668).
4. The raw netconsole burst timeline above. These timestamps and codes have 0 docs hits: 0x00841051, 0x008C1071/0x808C1071, 0x80001431, 0x006012B1/0x002012B1, 0x008012A1, and every HH:MM:SS listed (0x00801031 and 0x00801231 do appear in docs). The repeated same-second onsets on two or three rigs (04-17, 04-20 x3, 04-28, 05-01, 11-15) appear only here.
5. 2026-11-15 events outside the GCVM burst: .121 ata5 SATA BadCRC at 09:06-09:07, .155 SMU failure at 13:47:28, "Overdrive is enabled" on .121 at boot. Owner context: the brief records today's .121/.163 GCVM -> 0xFFFFFFFF -> inaccessible -> hang. The ata and .155 lines are not in docs.
6. S174 D1.1 (05-11 17:51, pool 8, VALID_ACTIVE_FAULT, two rigs unreachable) vs D1.2 (05-11 19:22, pool 6, VALID_CLEAN); all 23 S174 dmesg tails empty.
7. S170 per-worker cooldown and S171 coordinator dispatch throttle experiments (launch scripts and logs). The cooldown env var has 1 docs hit (S170 changelog). The throttle has 0 docs hits; its code is not at HEAD.
8. Source fact, not stated in docs per grep. The RANGE-MINER fleet launcher sets only ROCR_VISIBLE_DEVICES, CUPY_CACHE_DIR and PYTHONPATH (scripts/launch_fleet_manual.sh:316-318, stagger default 3 s :150), and miner/range_miner_worker.py sets no HSA_*. So none of these reach miner workers from code: HSA_ENABLE_SDMA=0, HSA_OVERRIDE_GFX_VERSION, HSA_ENABLE_DEBUG_TRAP=0, HSA_ENABLE_RUNTIME_POWER_MGMT=0, AMDGPU_NO_POWER_PROFILE=1, PRNG_CUPY_POOL_LIMIT_MB, CUPY_GPU_MEMORY_LIMIT. Whether rig ~/rocm_env or ~/.bashrc exports them is [UNVERIFIED] (no SSH). Also, free_all_blocks() is present in miner/range_miner_worker.py:1069-1070 and sieve_gpu_worker.py:91-92 at HEAD.

### SEARCHED / NOT READ
Searched: the headers (first 40 lines) of all 127 apply/verify scripts, plus targeted body greps; `git log --all` (942 commits) with keyword filters; `git show` on 18 distributed_config.json commits and on the env/free_all_blocks commits (`-G`/`-S`); the HEAD source of persistent_worker_coordinator.py, pwc_transport_tcp.py, pwc_worker_service.py, sieve_*.py, zmq_sqlite_coordinator.py, miner/range_miner_worker.py and scripts/launch_fleet_manual.sh; all ignored text files at most 5 MB in the main tree; all of dashboard_work.
Not read: 106 ignored files >5 MB (listed in ignored_skipped.txt, mostly logs/s16x-s172 run logs and full_scoring_results). Only the experiment-log outcomes were grepped. Also not read: binaries (npz, db, tar.gz, including S174_D1_1_*_ZEUS_bundle.tar.gz and ~/Downloads/ser8_crash_forensics_D1_*.tar.gz), the rig hosts and CTs (no SSH), ser8 (crash_forensic_daemon v2), ~/baselines, other worktrees, the private repo. No permission errors occurred.

---

# Fault classification (added 2026-11-15, owner direction 15:39 PST)

**Why this section exists.** The owner observed that many recorded crashes were the rigs being overwhelmed, not the GCVM fault, and that the record does not separate the two. This section gives every crash, hang, reset and failed run in the register one fault class, with the evidence for that class. It concludes no cause and closes no lead.

## Rubric (applied identically to every row)

| Class | Requires |
|---|---|
| **GCVM** | A `GCVM_L2_PROTECTION_FAULT` line, or the `[gfxhub] page fault` line that heads it, is recorded **for that event**: in a raw log, or quoted for that event in a document. The row cites that line's file:line. |
| **OVERLOAD** | An OOM kill, RAM exhaustion, an SSH or session limit, a load or job-count limit, a kernel soft lockup, a launch/respawn/HIP-init storm, or a cause **the record itself** attributes to overwhelming the rig. The row cites the record's words. |
| **OTHER-GPU** | A ring timeout, SMU `0xFFFFFFFF`, a reset or reset failure, a KFD preemption failure, an amdgpu NULL dereference, device lost or inaccessible, or a missing card, **with no GCVM line** for that event. |
| **OTHER** | A recorded non-GPU, non-overload cause: a coordinator, lease, staging, config, transport, interpreter or JSON defect, an operator action, a Zeus-only failure. |
| **UNKNOWN** | No kernel-level or attributing evidence is recorded. The row says what is missing. **No evidence means UNKNOWN; a hypothesis is not evidence.** |
| **CLEAN** | The run completed with no recorded host, card or worker loss. Listed so that the clean runs sit beside the failures. |

Where a kernel line and the record's prose disagree, the **kernel line decides** and the row notes the prose. Example: the record says "memory accumulation" or "3-rig load", but a GCVM line exists for the event.

**Evidence sources.**
- The cited documents in this worktree.
- `git show` of deleted documents.
- The main tree's gitignored logs (read-only), abbreviated as follows:
  - `NL` / `NCL`: `/home/michael/distributed_prng_analysis/logs/netconsole_listener.log`, 2026-04-17 → 2026-11-15.
  - `RCN`: `logs/rrig6600c_netconsole.log`, 2026-04-01.
  - `ML/` and `L/`: the main tree's `logs/`.
- The netconsole lines were indexed with file:line in `kernel_evidence_summary.txt` and checked by reading.
- Each part's extraction notes and full abbreviation keys are kept in the scratchpad (`classify/partA.md` … `partD.md`).

## Null results first

- **OVERLOAD.** 27 of 199 failure events. 22 of the 27 fall in January–March, and only 10 of those carry a kernel or OOM line: the OOM kills in A-12, A-16, A-44 to A-51, and the boot-time soft lockup in A-21. The other OVERLOAD rows rest on the record's own attribution.
  - From 04-17 to 05-12, `NL` contains **0** rig OOM-kill, RAM-exhaustion or soft-lockup lines. The one OVERLOAD row in that window is the Zeus host OOM (C-27), which is not a rig.
  - **July–November: 0 OVERLOAD.** The two host losses without a GCVM line (08-19, 08-22 run 3) are UNKNOWN. The only overwhelm-type explanations recorded for them are OPEN hypotheses (H2 supply sag, H3 host resource exhaustion).
- **GCVM.** 44 of 199 failure events. None falls before 2026-04-01.
- **UNKNOWN.** 43 of 199. Most have no kernel capture at all. The causes:
  - no netconsole before 04-01;
  - no raw log 04-02 → 04-16;
  - the listener was down 04-17 17:25 → 21:16 (`NCL:1185`);
  - no rig amdgpu datagram arrived 05-15 → 08-22.
- **A Step-1 "PASSED" does not mean the kernel was clean.** B-53 (04-17 10:36) and B-56 (04-17 14:34) passed with GCVM lines on the rigs. Part C lists three more runs the record calls clean or complete with a rig kernel line inside their window.

## Counts by Part

| Part | GCVM | OVERLOAD | OTHER-GPU | OTHER | UNKNOWN | CLEAN | rows |
|---|---|---|---|---|---|---|---|
| A (Jan–Mar) | 0 | 22 | 4 | 10 | 8 | 23 | 67 |
| B (03-30 → 04-17 20:00) | 13 | 4 | 3 | 9 | 21 | 14 | 64 |
| C (04-17 20:00 → 05-12) | 30 | 1 | 33 | 11 | 10 | 19 | 104 |
| D (07-01 → 11-15) | 1 | 0 | 1 | 14 | 4 | 6 | 26 |
| **all** | **44** | **27** | **41** | **44** | **43** | **62** | **261** |

The 261 rows are **199 failure events + 62 clean runs**. Two rows were dropped: C-104 (the compute half of C-105) and C-71 (a roll-up of C-05/48/50/57).

## How many rows moved out of the GCVM picture

**Whole register: 155 of 199 failure events, 78%, are not GCVM-class.** That is 27 OVERLOAD, 41 OTHER-GPU, 44 OTHER and 43 UNKNOWN. 44 events, 22%, carry a recorded GCVM line.

**The consolidated isolation list (4) of the first version.** It held 22 failure rows. Reclassified:
- **8 stay GCVM:** 04-02 rrig6600 (B-05); 04-04 ~18 s (B-16); 04-04 ~50 min (B-18); 05-01 pool 8 50K/100K (C-82, C-79); S173 ×2 (C-85, C-86); 05-05 post-completion (C-92); 05-11 D1.1 (C-105); 11-15 retry (D-25).
- **2 split:**
  - 04-17: rrig6600b GPU1 silent death is UNKNOWN (C-01); the rrig6600c fatal is GCVM (C-02).
  - 04-25/26 "2 rigs still crashes": Test 1 is OTHER-GPU, `GPU_COUNT_MISMATCH 7/8` (C-62); the probe run is GCVM (C-63).
- **12 move out of the GCVM picture:**

  | Row | New class | Ref |
  |---|---|---|
  | 01-20 120 jobs at once | OVERLOAD | A-06 |
  | 03-20 Run 1 crash | UNKNOWN | A-34 |
  | 03-25/26 rrig6600c 4 workers | OTHER (BIOS NVRAM) + OVERLOAD (CuPy OOM) | A-55, A-56 |
  | 03-29 reverse stall | UNKNOWN | A-63 |
  | 03-29 3-rig kernel panic | OVERLOAD, by the record's attribution; no panic text | A-65 |
  | 04-05 2M / 1M / 500K | UNKNOWN | B-32, B-29, B-31 |
  | 04-13 500K | OVERLOAD, by attribution; no kernel line; the `_v2` log shows PASSED | B-48 |
  | 04-26 pool 8 + 50 ms cooldown | OTHER-GPU, SMU `0xFFFFFFFF` on rrig6600 03:00.0 | C-66, `NL:6878` |
  | 05-09 D1 | OTHER-GPU, SMU | C-101, `NL:8989` |
  | 08-12 attempt-5 drops | UNKNOWN | D-11 |
  | 08-19 | UNKNOWN | D-18 |
  | 08-22 run 3 | UNKNOWN | D-22 |

## Recorded facts the classification surfaced (no cause concluded)

- **The first raw GCVM line is 2026-04-01 20:06:49 on rrig6600c 0000:19:00.0 (`RCN:9`).** The register's earlier "first GCVM record 2026-04-02" refers to the first *document* (`TB_SUBMISSION_S159G_RIG6600_CRASHES.md:3`). Part B E5's date is corrected to 04-01 by the raw log.
- **OTHER-GPU on rrig6600.** 25 of Part C's 33 OTHER-GPU events are on rrig6600 03:00.0 or 19:00.0: the SMU `0xFFFFFFFF`, queue-preemption and mode1-reset family, with no GCVM line.
- **Several GCVM events begin in the same second on two or three rigs.**
  - 04-17 10:36:53: .120 + .162 (`NCL:4`, `:14`).
  - 04-20 at 17:06:40, 18:51:19 and 19:38:40: all three rigs, status `0x00841051`.
  - 04-28 20:26:42.
  - 05-01 15:55:07 (`NL:7600`, `:7610`).
  - 11-15 13:21:29: .121 + .163 (`NL:9639`, `:9650`).
- **Timing of GCVM onsets in Part C.** Of the 26 Part-C GCVM events whose run log survives, 10 begin 3–9 s after a `workers launched` line, and 15 within 30 s of one. This is a timing fact, not a cause.
- **Two GCVM lines on rrig6600b with no TFM run computing,** 105–150 s after boot (B-59 `NCL:689`; B-61 `NCL:1160`).
- **The Run 4 retry's configuration is in the run log, not in the record.** It used the same pinned geometry and seed start as attempt 1 (`ML/gate12_20261115_131154.log:7, :44`). Phase 4 was dispatched at 13:21:29.024 (`:537`), 38 ms before the first GCVM line (`NL:9639`).
- **Register corrections from this pass.**
  - 08-22 run 2's cause *is* recorded, in the main tree only: `staging_capacity_timeout`, "STARVED BY CONSTRUCTION" (`ML/brief_i_evidence/S172_BRIEF_I_RUN2_CAPACITY_FAILURE_REPORT.md:13`).
  - Part E's "0 kernel datagrams 05-15 → 08-22" is corrected in Null results above.
  - S112's "1 GPU hang, recovered" (`SESSION_CHANGELOG_20260226_S112.md:116`) is 4 jobs on .120 gpu2 that "failed permanently after 3 retries" (`ML/s112_steps1_2_real_data.log:6896-6900`).

## Isolation results, reclassified — Table 1: GCVM-class events and the clean runs beside them

Date order within each Part. GCVM rows are the failures; CLEAN rows are the configurations that completed.

| ID | Register ref | Date/time | Rig(s)/cards | Config | Event | Class | Evidence |
|---|---|---|---|---|---|---|---|
| A-02 | E2 / LIST4 | 01-16 | rig-6600, rig-6600b | 2–4 GPUs/rig, 5 and 12 trials, Step 2, SSH | 5 and 12 trials worked | **CLEAN** | commit a47ba88: "5 trials worked, 12 trials worked, 26 trials untested" |
| A-07 | E7 | 01-20 | both rigs | 20 trials/batch (7 jobs/rig), sample 350–750, Step 2 benchmark, SSH | 5 × 20 = 100 trials, 100% | **CLEAN** | `docs/PROPOSAL_Job_Batching_Pipeline_Stability_v2.md:282` "5 batches × 20 trials = 100 trials total, 100% success rate" |
| A-08 | E7 | 01-20 | cluster | MAX_JOBS_PER_NODE_PER_BATCH=6, Step 2, SSH | 25/25 trials | **CLEAN** | commit 49e8ad4: "25/25 trials now complete at 100% success rate" |
| A-09 | E5 | 01-18 | rig-6600, rig-6600b | 12 concurrent, sample 350/450/550/650/750, Step 2.5, SSH | 100 trials, 100% | **CLEAN** | `git show a490ea0:docs/SESSION_CHANGELOG_20260118.md:17` "Benchmark Results (100 trials total, 100% success rate)" |
| A-14 | E10 | 01-24 | rig-6600, rig-6600b (+Zeus) | 12/rig, chunk 1000, Step 3, SSH | 99/99, 313 s (also post-fix #2 at 4 GPUs 99/99) | **CLEAN** | `docs/SESSION_CHANGELOG_20260124_OOM_FIX.md:241` "Final validation \| 12 \| 12 \| 99/99 \| ✅ Pass" |
| A-17 | E10 | 01-25 | rig-6600/rig-6600b | Step 3, chunk 1000 | 100/100 jobs, 5:20 | **CLEAN** | `docs/SESSION_CHANGELOG_20260125.md:28` "100/100 jobs completed" |
| A-19 | E12 | 01-26 | 26-GPU cluster | same, plus inter-chunk cleanup | 20/20 trials, 0 hangs | **CLEAN** | `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:1034` "20/20 benchmark trials with 0 GPU hangs" |
| A-25 | E18 | 02-23 | Zeus + 3 rigs | Step 2, 20 trials, after rig-6600b reboot | 20/20 trials | **CLEAN** | `docs/SESSION_CHANGELOG_20260223_S109.md:55` "20/20 trials completed, 56s trials + 2s collection" |
| A-27 | E23 | 03-08 | 3 rigs (24 AMD) + Zeus | persistent workers in coordinator.py, 10 jobs × 20M seeds, Step 1 | 17/17 checks; 48 workers each run | **CLEAN** | `docs/SESSION_CHANGELOG_20260308_S130.md:93` "Total: 17/17 checks passed." |
| A-31 | E28 | 03-16 | 3 rigs | pool 4/rig, 10M seeds, 3 trials, fwd/rev/hybrid fwd/hybrid rev, PWC SSH | all 4 passes | **CLEAN** | `docs/SESSION_CHANGELOG_20260316_S146.md:103` "Zero ❌ errors. Zero crashes. Zero tracebacks across all chunks." |
| A-33 | E30 | 03-20 | rrig6600; 3 rigs | GPU0/GPU7 spawn test; then 8/rig, no masking, preprod 5 trials, PWC SSH | spawn test pass; 5/5 trials, 24/24 alive | **CLEAN** | `docs/SESSION_CHANGELOG_20260321_S149.md:41-42` "Preprod soak 5/5 trials clean, 24/24 workers alive" |
| A-35 | E31 | 03-20/21 overnight | rrig6600c | 8 workers, pre-slim_v1, PWC SSH | ran for hours | **CLEAN** | `docs/ROOT_CAUSE_ANALYSIS_RRIG6600C_S151.md:23` "rrig6600c ran 8 workers for hours — STABLE" |
| A-38 | E32 | 03-21 20:36 | rrig6600c | 4 workers, PWC SSH | stable on monitor | **CLEAN** | `docs/ROOT_CAUSE_ANALYSIS_RRIG6600C_S151.md:167` "20:36:02 workers=4 load=0.12 … ssh_conns=1" |
| A-41 | E33 | 03-22 | 26/26 GPUs incl. rrig6600c 8 | 8/rig, pre-slim_v1, 3 trials, PWC SSH | 3 trials; rrig6600c stable | **CLEAN** | `docs/SESSION_CHANGELOG_20260322_S152.md:77` "rrig6600c stable throughout all 3 trials" |
| A-43 | E34 | 03-23 | rrig6600c GPUs 0–7 | Stage 8A/8B (1 kernel per GPU) and Stage 9 (8 persistent workers, 1 reverse job each) | all stages pass | **CLEAN** | `docs/SESSION_CHANGELOG_20260323_S153.md:61` "Stage 9 PASSED — All 8 reverse jobs completed successfully" |
| A-52 | E36 | 03-25 | Zeus 2 + rrig6600 8 + rrig6600b 8 + rrig6600c 4 (22 GPUs) | 3-trial production, PWC SSH | all 3 trials complete | **CLEAN** | `docs/SESSION_CHANGELOG_20260324_S155.md:84-85` "All 3 trials completed … rrig6600c stable throughout at 4 workers" |
| A-57 | E37 / iso 03-25/26 | 03-25/26 | rrig6600c | Test A 2 workers (3 trials); Test B 3 workers post-BIOS (2+ trials) | pass / pass | **CLEAN** | `docs/SESSION_CHANGELOG_20260326_S156.md:98-99` "A — 2 workers … ✅ PASSED"; "B … CuPy OOM on some chunks" |
| A-58 | E37 / iso 03-26 | 03-26 | rrig6600, rrig6600b | 8 workers each, PWC SSH | stable | **CLEAN** | `docs/SESSION_CHANGELOG_20260326_S156.md:112-113` "rrig6600: 38,684 kB (8 workers, stable)" |
| A-59 | E37 | 03-26 | cluster, rrig6600c at 3 | Run 1, seeds 0→1,073,741,824, 3 trials, PWC SSH | 3 trials complete | **CLEAN** | `docs/SESSION_CHANGELOG_20260326_S156.md:125-126` "3 trials completed ✅ / 3 workers on rrig6600c" |
| A-61 | E39 | 03-29 | rrig6600 + Zeus | 8 AMD, 537 chunks, 2M AMD cap, Step 1 full sweep, PWC SSH | 537-chunk sweep | **CLEAN** | `docs/SESSION_CHANGELOG_20260329_S158.md:30` "rrig6600 isolated \| 8 AMD + Zeus \| ✅ Full 537-chunk sweep" |
| A-62 | E39 | 03-29 | rrig6600c + Zeus | 8 AMD, 537 chunks, forward, PWC SSH | full sweep | **CLEAN** | `docs/SESSION_CHANGELOG_20260329_S158.md:31` "rrig6600c isolated (forward) … ✅ Full 537-chunk sweep" |
| A-64 | E39 | 03-29 | rrig6600 + rrig6600c + Zeus | 16 AMD, 537 chunks, S158B bounded dispatch, PWC SSH | 537/537, 34:15, 14,046 bidi | **CLEAN** | `docs/SESSION_CHANGELOG_20260329_S158.md:39` "34:15 elapsed, zero crashes" |
| A-67 | E40 | 03-30/31 | rrig6600 + Zeus | 8 AMD, 2M chunk, Step 1, ZMQ+SQLite, systemd-run | "running clean" (in progress) | **CLEAN** | `docs/S159_CHAT_PROMPT.md:6` "rrig6600 + Zeus running clean via ZMQ+SQLite+systemd-run" |
| B-02 | E5; L4 #4 | **2026-04-01 20:06:49** per log (register says 04-02) | rrig6600c 19:00.0 (pid 3408), 06:00.0 (pid 3003) | ZMQ, stock, seed cap n/r (TB's "5M" hypothesis at CL159orig:60 implies 5M [UNVERIFIED]) | Two GPUs page-fault together, then GPU reset and rig down | **GCVM** | RCN:9 "GCVM_L2_PROTECTION_FAULT_STATUS:0xFFFFFFFF"; RCN:5 "[gfxhub] page fault"; RCN:399 reset |
| B-05 | E9, E10; L4 #5; iso 04-02 | 2026-04-02 17:49 (run `L/zmq_iommu_test_v1.log` 17:33→21:01) | rrig6600 03:00.0, pid 5153 | 3 rigs, workers n/r, 2M, ZMQ, stock, iommu=pt, rrig6600 workers lacking SDMA vars | SQC(inst) fault escalates to 0xFFFFFFFF; rig reboots; forward completes via leases; hybrid failed=537 (`L/zmq_iommu_test_v1.log:8641`) | **GCVM** | `TB_SUBMISSION_S159G_RIG6600_CRASHES.md:22` "GCVM_L2_PROTECTION_FAULT_STATUS: 0x00801231"; :23 "SQC (inst)" |
| B-07 | E14; L4 #6 | 2026-04-03 pre-v5 (series; `L/zmq_s160v4b_stability.log:1064` code 1 [UNVERIFIED link]) | rrig6600 (.120), rrig6600c (.162), repeatedly; b never | 3 rigs + Zeus, ZMQ, 2M, stock, env vars PASS | Repeated crashes under load; riser power cycle needed | **GCVM** | `SESSION_CHANGELOG_S160.md:55` "GCVM_L2_PROTECTION_FAULT_STATUS:0xFFFFFFFF" (session crash signature) |
| B-08 | E15; L4 #7; iso 04-03 | 2026-04-03 (v5, `L/zmq_s160v5_cleanup.log` 12:02) | all 3 AMD + Zeus | ZMQ, 2M, per-chunk `_best_effort_gpu_cleanup`, stock | Full bidirectional trial | **CLEAN** | `SESSION_CHANGELOG_S160.md:90-91` "Zero GPU crashes / Zero netconsole fault events" |
| B-11 | E16; L4 #8 | 2026-04-03 S161 Gate 1 | 1 rig, 1 GPU | 1 worker, 500K seeds, TCP-PWC, stock | Passed | **CLEAN** | `SESSION_CHANGELOG_S161.md:33` "Gate 1 (500K seeds, 1 rig, 1 GPU) — PASSED" |
| B-12 | E16; L4 #9 | 2026-04-03/04 Gate 2 | 3 rigs, 24 AMD | 8/rig, 25 × 2M = 50M, TCP-PWC, stock, no TCP cleanup | Passed | **CLEAN** | `SESSION_CHANGELOG_S161.md:40` "Zero crashes, zero netconsole faults" |
| B-13 | E16; L4 #10; iso 04-03/04 | 2026-04-04 S161 | 3 rigs + Zeus, 26 | 24+2, 50M in 53 s, TCP-PWC, stock | Passed, 2.24M sps | **CLEAN** | `SESSION_CHANGELOG_S161.md:44-46` "50M seeds in 53 seconds" |
| B-14 | L4 #11 | S161 benchmark | 1 rig, 8 AMD | 8 workers, 16M, SSH-PWC | Completed in 2m 1.5s | **CLEAN** | `SESSION_CHANGELOG_S161.md:55` "SSH-PWC / 8 AMD / 16M / 2m 1.5s" |
| B-15 | L4 #12 | S161 benchmark | 3 rigs, 24 AMD | 50M, ZMQ | Completed in ~60 s | **CLEAN** | `SESSION_CHANGELOG_S161.md:54` "ZMQ+SQLite / 24 AMD / 50M / ~60s" |
| B-16 | E17; L4 #13; iso 04-04 | 2026-04-04 S162-1 | rrig6600c, multiple GPUs and PIDs | 3 rigs + Zeus, 8/rig (24+2), cap n/r (2M implied, PRFAB:264), TCP-PWC, stock, no cleanup on rigs | Crash at ~18 s; watchdog hard LOCKUP | **GCVM** | `docs/SESSION_CHANGELOG_S162.md:25` "GCVM_L2_PROTECTION_FAULT_STATUS: 0xFFFFFFFF"; CL162a:43 |
| B-17 | E19; L4 #14; iso 04-04 | 2026-04-04 S162-3 | rrig6600c + Zeus (10) | 8+2, 3 trials, 1.07B, TCP-PWC + cleanup, stock | 18m 52s | **CLEAN** | `SESSION_CHANGELOG_S162.md:39` "zero crashes, zero netconsole faults" |
| B-18 | E21; L4 #19 | 2026-04-04 S162-5 (rig uptime 3036 s) | rrig6600c 16:00.0, pid 5141 | 26 (24+2), cap n/r, TCP-PWC + cleanup + Semaphore(8) + sysctl, stock | SQC(inst), then TCP, then 0xFFFFFFFF; rig offline | **GCVM** | `docs/PROPOSAL_S162_RRIG6600C_CRASH_ROOT_CAUSE_v1_0.md:47` "GCVM_L2_PROTECTION_FAULT_STATUS: 0x00801231"; CL162a:145 |
| B-19 | E23; L4 #20; iso 04-04 | 2026-04-04 21:53:25 | rrig6600c 16:00.0 (pid 2986), 06:00.0 (pid 2693); all 8 workers fail suspend | 26, `AMD_SERIALIZE_KERNEL/COPY=3` on .162, cleanup active, stock | Crash ~1-2 min into trial; CPU soft lockup at 21:53:55 | **GCVM** | `docs/TB_UPDATE_S162_OPTION_B_RESULTS.md:35` "GCVM_L2_PROTECTION_FAULT_STATUS: 0x00801231" |
| B-34 | E34; L4 #32; iso 04-05 | 2026-04-05 18:51 → 42:36 (`L/s162_100k_run1.log`) | 3 rigs + Zeus, 26 | 24+2, 100K, 3 trials, TCP-PWC, DKMS all, free_all_blocks present | VICTORY run | **CLEAN** | `docs/SESSION_CHANGELOG_S162_VICTORY.md:23` "Netconsole: SILENT for entire run" |
| B-36 | E36; L4 #33 | 2026-04-06 (`L/s163_500k_sieve_filter_fix.log` 18:07 [UNVERIFIED link]) | 3 rigs + Zeus | 24+2, 500K, TCP-PWC (see E36 caveat), DKMS, free_all_blocks removed (68ed075) | Trial 1, 3,140 bidirectional, 21:08 | **CLEAN** | `docs/TB_SUMMARY_S163.md:48-49` "Netconsole: SILENT throughout / AMD rigs: stable, no faults" |
| B-40 | E38; L4 #34 | 2026-04-07..10 (series; `L/s163_200k_validation*.log`, `L/s163_step1_200k*.log`) | rrig6600, rrig6600c | 200K, reverse sieve, DKMS, after package removal | Crashes, then ih ring overflow, then device lost from bus | **GCVM** | `CHANGELOG_S163.md:66` "GCVM_L2_PROTECTION_FAULT_STATUS:0xFFFFFFFF on rrig6600 and rrig6600c" |
| B-41 | E39; L4 #34 | 2026-04-10 (gpu_recovery run) | rrig6600 / rrig6600c | 200K, `amdgpu.gpu_recovery=1 amdgpu.sched_jobs=256` | T1 OK (300 bidirectional); T2-3 failed, 3362/5369 chunks failed | **GCVM** | `CHANGELOG_S163.md:81` "run crashed with 3362/5369 chunks failed"; GCVM per series line :66 |
| B-43 | L4 #35 | 2026-04-12 | 3 rigs + Zeus | 24+2 (after fix), 10K, TCP-PWC, DKMS, post-clone | Stable | **CLEAN** | `SESSION_CHANGELOG_S163_FINAL.md:70` "10K ✅ Stable" |
| B-44 | L4 #36 | 2026-04-12 14:49 (`L/s163_ladder_100k.log`) | 26 | 100K | Stable | **CLEAN** | `SESSION_CHANGELOG_S163_FINAL.md:71` "100K ✅ Stable" |
| B-46 | L4 #37 | 2026-04-13 16:01 (`L/s163_ladder_250k_v3.log`) | 26 | 250K, set_limit, JSON skip, fresh reboot | Stable | **CLEAN** | `SESSION_CHANGELOG_S163_FINAL.md:72` "250K ✅ Stable / After fresh reboot" |
| B-47 | L4 #38 | 2026-04-13 18:49 (`L/s163_ladder_300k_v2.log`) | 26 | 300K, set_limit, fresh reboot | Stable, "all 26 GPUs held" | **CLEAN** | `SESSION_CHANGELOG_S163_FINAL.md:73` "300K ✅ Stable / all 26 GPUs held" |
| B-52 | **not in register** | 2026-04-17 08:58-10:08 (`L/s163_100k_fixes_confirmed.log`) | 3 rigs + Zeus | min_workers 24, 100K, 3 trials, TCP-PWC | Step 1 PASSED | **CLEAN** | `L/s163_100k_fixes_confirmed.log:107874` "Step 1 PASSED" |
| B-53 | **not in register** | 2026-04-17 10:36:53 (run `L/s163_100k_json_guard.log`, launched 10:36:28) | rrig6600 03:00.0 + rrig6600c 06:00.0 in the same second | 100K, 3 trials, TCP-PWC, JSON guard | GCVM on 2 rigs, then SMU 0xFFFFFFFF, then device lost 10:37:02 (NCL:239, 242); Step 1 still PASSED (json_guard.log:35682) | **GCVM** | NCL:4 ".120 … GCVM_L2_PROTECTION_FAULT_STATUS:0xFFFFFFFF"; NCL:14 ".162 … 0xFFFFFFFF" |
| B-56 | **not in register** | 2026-04-17 14:34:25 (run `L/s163_100k_mem_deployed.log` 14:23-14:40) | rrig6600c 16:00.0, then device lost on 16/06/19:00.0 | 100K, 2 trials | GCVM, then device lost (NCL:523-542); Step 1 still PASSED (mem_deployed.log:43368) | **GCVM** | NCL:324 ".162 … GCVM_L2_PROTECTION_FAULT_STATUS:0x00801231" |
| B-59 | **not in register** | 2026-04-17 16:20:49-16:36:28 (booted ~16:18, NCL:614) | rrig6600b 10:00.0 | GCVM at 16:21:02, before the 1622 run start (16:23:00); continues into it | sdma1/gfx ring timeouts, gfx ring reset failure (NCL:710), qcm timeout (NCL:1061), "Failed to quiesce KFD" (NCL:1064) | **GCVM** | NCL:689 ".154 … GCVM_L2_PROTECTION_FAULT_STATUS:0x006012B1" |
| B-60 | **not in register** | 2026-04-17 16:33:45 (run `L/s163_revert_test_1622.log`, 16:23-16:36) | rrig6600 03:00.0 | 300K, 2 trials, "S163 FINAL" revert test | GCVM; Step 1 code -9 then ESCALATED (log:110186); Zeus logs 7,158 `cudaErrorDevicesUnavailable` lines | **GCVM** | NCL:826 ".120 … GCVM_L2_PROTECTION_FAULT_STATUS:0xFFFFFFFF" |
| B-61 | **not in register** | 2026-04-17 17:21:58 (.154 booted ~17:20, NCL:1070) | rrig6600b 0d:00.0 | no run active (none between 16:36 and 17:55) | sdma1 timeout at 17:21:45 (NCL:1153), then GCVM at uptime 105 s (vmid 2) | **GCVM** | NCL:1160 ".154 … GCVM_L2_PROTECTION_FAULT_STATUS:0x002012B1" |
| C-02 | E3; ISO 04-17 | 04-17 21:16:45 | rrig6600c 16:00.0, 06:00.0, 19:00.0 | same as C-01 | GCVM cascade, device lost, manual reboot. Onset 0.6 s after reverse-pass dispatch to 24 workers (log:68799 21:16:45.273) | **GCVM** | NL:1190 "GCVM_L2_PROTECTION_FAULT_STATUS:0x00801231"; NL:1404 "device lost from bus!" |
| C-04 | LOG-ONLY (E6 planned "GDM removed" discriminator) | 04-18 10:29–14:00 | 3 rigs+Zeus | 24 workers, 100K, 5 trials, GDM disabled | completed; post-processing IndexError only (not GPU) | **CLEAN** | ML/s163_karg_gdm_clean_100k_5trials_1029.log:380852 "Step 1 PASSED"; listener live NL:1591 (09:35) → NL:1592 (19:04), no amdgpu line |
| C-06 | LOG-ONLY (E10 "26/26" runs) | 04-19 11:19–14:23 | 26 (min_workers 1) | smoke_nparallel2_tcp ×5 (100K, 2 trials, n_parallel 2); tb_disc_test4 (rrig6600 only) | no worker loss in logs; smokes stop mid-chunk at ~37k survivors/chunk (stop cause n/r) | **CLEAN** | ML/smoke_nparallel2_tcp_1401.log last line "100,000 seeds → 36,801 survivors"; ML/tb_disc_test4_rrig6600_only_1349.log:179 PASSED |
| C-07 | E10 (production_1433); ISO 04-19 | 04-19 14:33–15:04 | 26 | 24 workers+Zeus, 100K, 5 trials, DPM pin manual sclk1/mclk3 | completed; NameError in post-processing only | **CLEAN** | ML/s163_karg_production_1433.log:179714 "Step 1 PASSED"; P2:119 |
| C-08 | E10 (production2_1508) — fault not recorded in docs | 04-19 15:12:52 | rrig6600 03:00.0 | s163_karg_production2_1508: 24 workers, 100K, 5 trials, DPM pin | GCVM at +4:04, device lost 15:13:01; rig rebooted (READY 15:19:34); ready 24→22 | **GCVM** | NL:1811 "GCVM_L2_PROTECTION_FAULT_STATUS:0x008012A1"; NL:2010 "device lost from bus!" |
| C-11 | E16/E17; ISO 04-19/20 Kaspa | 04-20 17:06:40 | all 3 rigs: .120 06:00.0, .162 06:00.0, .154 03:00.0 | s165_zeus_tcp_smoke_1706: 26 workers, **seed_cap_amd 10,000**, 2 trials, Kaspa 2250 MHz/-150 mV (S165:12-14; revert time n/r) | GCVM in the same second on 3 rigs at launch+4 s (+0:24); queue preemption; NULL deref .162 17:07:18, .120 17:07:20 | **GCVM** | NL:2192, NL:2257, NL:2294 "STATUS:0x00841051"; NL:2402 "BUG: kernel NULL pointer dereference" |
| C-13 | E16/E17 | 04-20 18:51:19 | all 3 rigs: .120 16:00.0, .162 0d:00.0, .154 06:00.0 | s165_production2_1850: 26 workers, 100K, 2 trials | GCVM in the same second on 3 rigs at launch+3 s; NULL deref .120/.162; MODE1 reset .162; ready 26→22→10→9 | **GCVM** | NL:2621, NL:2683, NL:2723 "STATUS:0x00841051"; NL:2800, NL:2882 NULL deref |
| C-15 | E16/E17 | 04-20 19:17:07 | rrig6600b 13:00.0 (uptime 81 s after reboot) | same run | GCVM at worker relaunch+3 s; python task blocked 122/245/368 s (S165:55 "368+ seconds") | **GCVM** | NL:3248 "STATUS:0x00841051"; NL:3365 "blocked for more than 368 seconds" |
| C-17 | E16/E17 | 04-20 19:38:40 | all 3 rigs: .162 0a:00.0, .120 16:00.0, .154 06:00.0 | s165_smoke2_1938: 26 workers, 50K, 2 trials | GCVM in the same second on 3 rigs at launch+3 s; NULL deref .162; .154 09:00.0 ASIC reset -22 | **GCVM** | NL:3401, NL:3438, NL:3496 "STATUS:0x00841051"; NL:3564; NL:3876 "ASIC reset failed with error, -22" |
| C-18 | E18; ISO 04-20 | 04-20 19:56:55–20:15:12 | 26 | s165_stock_dpm_test_1956: 100K, 2 trials, stock `high` DPM, HSA_ENABLE_DEBUG_TRAP=0 | 17,721 survivors, ~14 min. Step 1 later exited -9 at 04:27 04-21 (compute had ended 20:15; cause n/r) | **CLEAN** | S165:177 "Netconsole: SILENT throughout"; listener live NL:3879-3881 (19:46 READY ×3) → NL:3882 (21:02); ML/…_1956.log:111809 "code -9" |
| C-25 | E22 run4 | 04-21 19:59–20:06 | 26 | s166_run4_1959: 100K, 2 trials | completed, no warnings | **CLEAN** | ML/s166_run4_1959.log:107946 "Step 1 PASSED"; listener live NL:4065 (19:57) → NL:4066 (20:29) |
| C-26 | E21; ISO 04-22 | 04-22 16:15–21:04 | rrig6600 (cwsr_enable=0 mcbp=0) | runs 5–7 window | rrig6600 ran 75+ min with no GPU crash | **CLEAN** | S166:92 "ran for 75+ minutes without a single GPU crash"; listener live NL:4076 (16:15) → NL:4077 (21:04), no amdgpu line |
| C-31 | LOG-ONLY | 04-23 15:52:32 | rrig6600c 06:00.0 | s166_run10_1552: 26 workers, 100K, 2 trials | GCVM at launch+9 s; hard LOCKUP cpu5; device lost | **GCVM** | NL:4086 "STATUS:0x80081071"; NL:4189 "Watchdog detected hard LOCKUP"; NL:4185 |
| C-33 | LOG-ONLY | 04-23 16:42:01 | rrig6600c 16:00.0 | s166_run11_1640: 26 workers, 100K, 2 trials | GCVM at launch+4 s; device lost; ready 26→15 | **GCVM** | NL:4272 "STATUS:0x00801231"; NL:4371 "device lost from bus!" |
| C-34 | LOG-ONLY | 04-23 17:13:54 | rrig6600c 16:00.0 (uptime 166 s) | s166_run12_1713: 100K, 2 trials | GCVM at launch+6 s; device lost; run abandoned at 17:16:47 | **GCVM** | NL:4405 "STATUS:0x00801231"; NL:4504 |
| C-36 | LOG-ONLY | 04-23 18:11:50 | rrig6600c 16:00.0, 06:00.0 | s166_run13_1810: 100K, 2 trials | GCVM at +1:08 (launch+48 s); device lost on both cards; SMU | **GCVM** | NL:4586 "STATUS:0x00801281"; NL:4685, NL:4688 |
| C-41 | LOG-ONLY (S167:58 "rrig6600c … crashing repeatedly") | 04-24 17:53:36 | rrig6600c 16:00.0 | s167_fresh_test_1753: 26 workers, 200K, 2 trials, no jitter/gap | GCVM at launch+5 s (+0:26) | **GCVM** | NL:4806 "STATUS:0x00801231" |
| C-45 | E25 (S168 jitter) | 04-24 21:02:58 | rrig6600c 16:00.0 | s168_jitter_test_2101: 26 workers, 200K, 2 trials, first-assign jitter | GCVM at +1:36 (Trial-2 launch+26 s), 0xFFFFFFFF at 21:03:03 | **GCVM** | NL:4983 "STATUS:0x00801231"; NL:5080 "STATUS:0xFFFFFFFF" |
| C-46 | E26 (S169 pacing) | 04-24 21:25:53 | rrig6600 03:00.0; rrig6600c 16:00.0 page fault | s169_pacing_test_2125: 26 workers, 200K, 2 trials | GCVM .120 at launch+29 s; .162 page fault in the same second; .120 "Shutting down cpus with NMI" at 21:26:53 | **GCVM** | NL:5181 "STATUS:0x00801031"; NL:5176 "[gfxhub] page fault"; NL:5282 |
| C-47 | E26 s169_test; ISO 04-24 s169_test | 04-24 21:39:36 | rrig6600c 16:00.0 (uptime 161 s) | s169_test_2138: 26 workers, 200K, 2 trials, jitter+pacing | GCVM at Trial-2 worker launch+5 s; run completed (PASSED) | **GCVM** | NL:5430 "STATUS:0x00801231"; ML/s169_test_2138.log:11145 startup 21:39:32 |
| C-49 | E30 S170 Run 1; ISO 04-25 Run 1 | 04-25 08:53:56 | rrig6600c 16:00.0 | stability_cap_100000_t5_0853: 26 workers, 100K, 5 trials, S168/S169 jitter | GCVM at launch+7 s (+0:28) → 0xFFFFFFFF | **GCVM** | NL:5660 "STATUS:0x00801231"; NL:5760 "STATUS:0xFFFFFFFF" |
| C-51 | LOG-ONLY | 04-25 17:19:48 | rrig6600c 16:00.0, 19:00.0 | stability_cap_50000_t25_dpm_peak_jitter_1716: 26 workers, 50K, 25 trials, DPM peak-lock | GCVM at +3:33; device lost ×2; .120 03:00.0 KFD quiesce failure at 17:23:37 | **GCVM** | NL:5927 "STATUS:0x00801231"; NL:6026, NL:6030; NL:6076 |
| C-54 | ISO S170 final | 04-25 21:18 / 21:47 | 26 | parity2_2118 / both_modes_2147: 100K | full 1.07B scan, 0 survivors | **CLEAN** | S170:205-207 "Full 1.07B seed scan completed" |
| C-56 | LOG-ONLY | 04-25 22:17:50 | rrig6600c 16:00.0 | s170_clean_100k_t5_2203: 26 workers, 100K, 5 trials, peak-lock | GCVM at +14:34 | **GCVM** | NL:6088 "STATUS:0x00801431" |
| C-58 | LOG-ONLY | 04-25 22:36:08 | rrig6600c 16:00.0, 06:00.0 | same run (+32:52, trial launch+47 s) | GCVM, device lost ×2; run PASSED at 22:46 | **GCVM** | NL:6347; NL:6449, NL:6472 |
| C-60 | LOG-ONLY | 04-26 07:47:46 | rrig6600c 16:00.0 | same run (+3:25) | GCVM, device lost | **GCVM** | NL:6534 "STATUS:0x00801231"; NL:6633 |
| C-63 | E34/E33 (2-rig + probe) | 04-26 15:42:55 | rrig6600c 16:00.0 | s170_2rig_no6600b_k512_probe_100k_t5_1539: 18 workers, pool 8, 100K, 5 trials, K512+probe | GCVM at +3:20 (trial launch+17 s); device lost | **GCVM** | NL:6759 "STATUS:0x00801231"; NL:6858 |
| C-65 | E35 Test 2; ISO 04-25 Test 2 (and the ROCm:32 pool6/100k "operator history") | 04-26 16:39:23–17:06:01 | rrig6600 + rrig6600c (6 each) + Zeus | s170_2rig_no6600b_k512_probe_pool6_100k_t5_1639: 14 workers, pool 6, min 12, 100K, 5 trials, jitter 15/gap 0.20 | 5/5 trials in 26:38 (register: duration n/r) | **CLEAN** | ML/…_1639.log:172612 "Step 1 PASSED"; S170:69-72 "Netconsole clean" |
| C-69 | E38 Test 4 attempt 3 (register: result n/r) | 04-26 19:11:27 | rrig6600c 16:00.0 | cooldown500_1910: 18 workers, pool 8, 100K, 5 trials, cooldown 500 | GCVM at +1:21; device lost; then .120 03:00.0 device lost at 19:15:03; run abandoned 19:13:58 | **GCVM** | NL:6901 "STATUS:0x00801231"; NL:7000; NL:7022 |
| C-72 | E41 S171 throttle100 (register: result n/r) | 04-28 16:33:03 | rrig6600c 06:00.0 | s171_pool8_100k_coord_throttle100_t5_1632: 17 workers (live probe), min 16, pool 8, 100K, 5 trials, jitter 15/gap 0.20, coordinator throttle 100 ms (not committed) | GCVM at +0:39 (launch+21 s) | **GCVM** | NL:7054 "STATUS:0x008C1071" |
| C-76 | E41 pool7_clean (register: result n/r) | 04-28 20:26:42 | rrig6600 03:00.0 + rrig6600c 16:00.0 (same second) | s171_pool7_100k_clean_t5_2000: 16 workers, pool 7, 100K, 5 trials | GCVM on 2 rigs at +26:08 (trial launch+3:45); device lost .162; ready 16→14 | **GCVM** | NL:7293 & NL:7303 "STATUS:0x00801431"; NL:7555 |
| C-79 | E43 pool8/100k; ISO 05-01 | 05-01 15:55:07 | rrig6600 03:00.0 + rrig6600c 16:00.0 (same second) | s172_pool8_100k_clean_t5_1543: 26 workers, min 16, pool 8, 100K, 5 trials, jitter 15/gap 0.20, full topology | GCVM on 2 rigs at +11:39 (ROCm "~12 min"); .162 hard LOCKUP; .120 device lost; kworker hung 122 s | **GCVM** | NL:7600 "0x008012B1"; NL:7610 "0x00801431"; NL:7897 "hard LOCKUP"; NL:7900 |
| C-81 | E43 pool6/50k; ISO 05-01 | 05-01 20:39–21:08 | 26 (pool 6 → 20 workers) | s172_pool8_50k_clean_t5_2039 (actually pool 6, min 12), 50K, 5 trials | 29:36, clean | **CLEAN** | ML/…_2039.log:215691 "Step 1 PASSED"; ROCm:31; listener live NL:8078 (19:06) → NL:8079 (22:04) |
| C-82 | E43 pool8/50k; ISO 05-01 | 05-01 22:04:32 | rrig6600c 16:00.0 | s172_pool8_50k_clean_t5_2201: 26 workers, min 16, pool 8, 50K, 5 trials | GCVM at +2:38; .120 SMU 1 s later; log stops at 22:08:00 | **GCVM** | NL:8082 "STATUS:0x00801281"; NL:8179 |
| C-83 | E43 pool8/25k; ISO 05-01/02 | 05-01 22:26–23:57 | 26 | s172_pool8_25k_clean_t5_2226: pool 8, min 16, 25K, 5 trials | 1:30:48, PASSED | **CLEAN** | ML/…_2226.log:430561 "Step 1 PASSED"; ROCm:33; listener live NL:8278 (22:12) → NL:8279 (00:03) |
| C-85 | E45 / ISO S173 "1m41s"; MT/launch_s173_v2_crash_pool8_50k_t5.sh:7 | 05-02 22:55:41 | rrig6600c 16:00.0 | S173 pool8/50k: min 16, pool 8, 50K, 5 trials, jitter 15/gap 0.20 | GCVM → device lost 22:55:50 → SMU | **GCVM** | NL:8295 "STATUS:0x00801231"; NL:8400 "device lost from bus!" |
| C-86 | E45 / ISO S173 "13min recurrence" | 05-02 23:11:25 | rrig6600c 16:00.0 (uptime 275 s) | same | GCVM → 0xFFFFFFFF, WALKER_ERROR 0x7 | **GCVM** | NL:8432 "STATUS:0x00801231"; NL:8442 "STATUS:0xFFFFFFFF" |
| C-87 | E45 / ISO S173 t1 | 05-02 (time n/r) | 26 | pool 8, 50K, 1 trial, jitter 15/gap 0.20 | "pool=8/50k/t1 just ran clean" | **CLEAN** | MT/launch_s173_validation_t5.sh:4 |
| C-88 | E45 / ISO S173 baseline | 05-03 before 09:38 | 26 | pool 8, 25K, 1 trial | baseline saved | **CLEAN** | MT/launch_s173_v2_crash_pool8_50k_t5.sh:11 "baseline saved" |
| C-89 | E45 manifest #1 | 05-03 10:24:15 | rrig6600c 16:00.0 (gpu6 per 7313a43) | S173 v2 crash run, pool8/50k/t5 (Trial 2 W21_O66_FT0.73) | GCVM SQC (inst) → device lost | **GCVM** | NL:8514 "STATUS:0x00801231"; NL:8613 |
| C-92 | E46 Run 2; ISO 05-05 | 05-05 18:18:55 | rrig6600c 16:00.0 + 06:00.0 | FT073 Run 2 (17:43), same config | post-completion GCVM ~4 min after finalisation; 06:00.0 0xFFFFFFFF at 18:19:00 | **GCVM** | NL:8742 "STATUS:0x00801231"; NL:8839 "STATUS:0xFFFFFFFF"; S174a:25 |
| C-94 | E46 Run 3 | 05-05 18:31 | 26 (all rigs rebooted) | same config | clean in 1:55 | **CLEAN** | S174a:27; listener live NL:8936-8937 (READY 18:26/18:30) → NL:8938 (20:03) |
| C-95 | E47 S174 baseline | 05-06 17:46:50 | 26 (dispatch at ready=2) | pool 8, 25K, 213M, open Optuna, min 24 | 8520/8520, 0 faults; INVALID as a control | **CLEAN** | S174a:60; ML/s174_baseline_pool8_25k_20260506_174650_summary.txt:20 "netconsole_fault_count: 0" |
| C-96 | E48 gate validation | 05-07 19:25:21 | 26/26 | pool 8, 25K, 5M, 1 trial, min 24 | ~1.5 s compute | **CLEAN** | S174a:148 "8/8/8 GPUs all rigs, no faults, no netconsole entries" |
| C-98 | E48 | 05-09 21:02:14 | 26/26 | pool 8, 25K, 5M, min 24 | VALID_CLEAN | **CLEAN** | ML/S174_GATE_VALIDATION_20260509_210214_summary.txt:60 |
| C-100 | ISO 05-09 D1 compute | 05-09 22:24:54–22:35:34 | 26 (ready 25) | S174 D1: pool 8, min 24, 50K, 425M, 1 forced trial FT0.73, jitter 3/gap 0.02/diag 1; cwsr=0 mcbp=0 on .120/.162 | compute clean, 17,000 chunks, 10:40 | **CLEAN** | D1:25; ML/S174_D1_FT073_50K_425M_20260509_222453_summary.txt:26-27 child exit 0 |
| C-103 | LOG-ONLY | 05-11 17:48:00 | 25 ready | gate validation: pool 8, 25K, 5M | VALID_CLEAN | **CLEAN** | ML/S174_GATE_VALIDATION_20260511_174800_summary.txt:60 |
| C-105 | E51 D1.1; ISO 05-11 | 05-11 17:52:20 | rrig6600c 06:00.0; rrig6600 03:00.0 SMU 17:52:34 | same as C-104 (cwsr revert status n/r) | GCVM 46 s after child start (launch+26 s); device lost 17:52:30; SMU on .120/.162; .120 and .162 unreachable afterwards | **GCVM** | NL:9033 "STATUS:0x80001431"; NL:9102; summary:54 faults; :60 "VALID_ACTIVE_FAULT" |
| C-106 | ISO 05-11 D1.2 | 05-11 19:22:46–19:24:24 | 3 rigs pool 6 (ready 20) | D1.2: pool 6, min 20, 50K, 425M, 1 forced trial FT0.73 | VALID_CLEAN, 1:38 compute | **CLEAN** | ML/S174_D1_FT073_50K_425M_20260511_192244_summary.txt:60; listener live NL:9199 (19:08) → NL:9200 (19:51) |
| D-01 | Part D E7; LIST 4 row 07-08/09 | 2026-07-08/09 | rrig6600c, CT100 (privileged per v1.1), 8 GPUs | Proxmox 8.4, kernel 6.8.12-9-pve, amdgpu-dkms 6.12.12, ROCm 6.4.3; framework smoke only (not TFM compute) | CuPy, PyTorch and TF smoke on 8 GPUs | **CLEAN** | commit 89b546c runbook §11: "CuPy 2000x2000 matmul, PyTorch (cuda available, 8 devices), TensorFlow (8 GPUs) ALL working" |
| D-02 | E11; isolation row 07-30; LIST 4 | 2026-07-30, 12:57–13:02 (3 runs in host dmesg) | rrig6600 CT100 .122, 1 GPU (dev 0, 03:00.0) | 1 worker, java_lcg const fwd/rev, seeds 8M, stripe 4M, cap 1M, k=3; RANGE-MINER Phase 6.0 | ROCm parity smoke | **CLEAN** | docs/TEAM_ALPHA_PHASE_6_0_RETURN_TO_BETA.md:77-78 "no `GCVM_L2_PROTECTION_FAULT`; no `VM_L2` / vm fault; no ring timeout" (one-off manual host dmesg); config docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md:66-67 |
| D-08 | E17; LIST 4 att 3 (compute) | 2026-08-10 17:28–19:48 | as D-07 | as D-07 | Att 3 compute: 128/128, 0 disconnects | **CLEAN** | docs/TFM_PROJECT_FACTS_SKILL.md:1885-1886 "All four stages, 128/128 stripes … Zero disconnects, zero reconnects" |
| D-17 | E21; isolation row; LIST 4 att 9 | 2026-08-17, 738 s loop | 3 CTs + Zeus, 25 (GPU 24/24) | 8/CT, k=12, anchor 25, τ 0.71, skip 6–99, gen-phase 25, 59 p3 / 23,515 p4, `554463d3`, code e9ca800 | Gate-12 att 9 PASSED | **CLEAN** | docs/TFM_PROJECT_FACTS_SKILL.md:2448,2454 "Step 1 PASSED … ZERO lease expiries · fleet exited clean" |
| D-20 | E24; isolation row 08-22 run 1 (compute) | 2026-08-22, serve loop 713.663 s | as D-19 | as D-19 | Run 1 compute: 128/128, 0 expiries, 0 disconnects | **CLEAN** | docs/S172_BRIEF_I_PRODUCTION_SHAPE_CLASSIFICATION_REPORT.md:25-28 "128/128 stripes DONE … compute_lease_expiry 0 · WORKER_DISCONNECTED 0" |
| D-24 | E36; context entry; isolation row 11-15 att 1 | 2026-11-15 09:07:39–09:19:49 | 3 Proxmox hosts/CTs + Zeus; rigs 8/8 after | 8/CT, 25 eligible every phase; k=12, anchor 25, τ 0.71/0.47, gen-phase 0, seeds [2^31, 2^32), 60 p3 / 23,568 p4, `57dc50fa`, HEAD 0e13dd6, TCP | Run 4 att 1 compute clean, 0 disconnects | **CLEAN** | docs/CLAUDE_CODE_INSTRUCTIONS_STEP1_CREDIT_WAL_20261115.md:56,58-59,62 "25 eligible workers at each" · "Worker disconnects 0" · "8/8 GPUs each". Netconsole was receiving: NCPROOF ×3 at 08:58:37–38 (MT/logs/netconsole_listener.log:9506-9508). Only kernel lines in the window are .121 SATA `ata5: SError: { UnrecovData 10B8B BadCRC }` (:9511, through :9634); no amdgpu line |
| D-25 | E37; context entry; isolation row 11-15 retry | 2026-11-15; phase 4 dispatched 13:21:29.024; first GCVM 13:21:29.062; terminal 13:31:29 | pve-rig6600 .121 card 0000:19:00.0; pve-rig6600c .163 cards 0000:16:00.0 and 0000:06:00.0; both hosts hung (.121 booted again 13:55:46) | 25 eligible at phase 4; same pinned geometry as D-24 (measured here, not stated in the record): k=12, offset 25, τ 0.71/0.47, skip 6–99, seeds from 2^31, stripe 67,108,864; `ab6318d4`; phase-3 idle (active_count 0) from 13:20:37 → all 25 resume at phase 4 | Run 4 retry: GCVM_L2 on 2 hosts in the same second at phase-4 start, then 0xFFFFFFFF, device inaccessible, host hang | **GCVM** | MT/logs/netconsole_listener.log:9639 ".121 amdgpu 0000:19:00.0 … GCVM_L2_PROTECTION_FAULT_STATUS:0x00841011"; :9650 ".163 amdgpu 0000:16:00.0 … GCVM_L2_PROTECTION_FAULT_STATUS:0x00801231"; :9660/:9730 "0xFFFFFFFF"; :9837-9838 "device inaccessible"; same lines in MT/logs/netconsole_all_rigs.log:363, :374, :561-562; timing MT/logs/gate12_20261115_131154.log:537 "derived_bound stage phase=4", :702 "TRIAL TERMINAL … compute_lease_expiry … rrig6600:gpu0" |

## Isolation results, reclassified — Table 2: everything else (OVERLOAD, OTHER-GPU, OTHER, UNKNOWN)

These rows are **not** GCVM-class. They should not be used as evidence about the GCVM fault without a kernel line.

| ID | Register ref | Date/time | Rig(s)/cards | Config | Event | Class | Evidence | Missing evidence |
|---|---|---|---|---|---|---|---|---|
| A-01 | E2 | 01-16 | rig-6600, rig-6600b (12 GPUs each) | 2 rigs, all 12/rig, 50 Step-2 trials, scripts_coordinator SSH | 50 trials froze both rigs; rebooted twice | **UNKNOWN** | commit a47ba88: "Root Cause Still Unknown … Likely SSH connection overload or simultaneous HIP init" | No dmesg or kern.log recorded. The record's lean is OVERLOAD, but it explicitly says unknown |
| A-03 | E3 | 01-17 | rig-6600 | 9 workers, 26-trial test, Step 2, scripts_coordinator SSH | all 9 jobs timed out at 3600 s; rig FROZEN | **OVERLOAD** | `docs/PROPOSAL_FINAL_ROCm_HIP_Init_Fix_v2_1.md:23` "Parallel HIP initialization storm"; `:31` "ALL 9 TIMED OUT (3600s) — FROZEN" | — (same-day commit f614736 instead blames a "GFXOFF race condition") |
| A-04 | E3 | 01-17 | rig-6600b | 8 workers, same 26-trial test | 1 of 8 jobs failed | **OTHER** | `docs/PROPOSAL_FINAL_ROCm_HIP_Init_Fix_v2_1.md:30` "7/8 passed (1 failed due to manual reboot)" | — |
| A-05 | E6 | 01-20 | both rigs (12 GPUs) | Step 2; `watch -n 0.5 rocm-smi` ×2 + fan loop at 3 s | Step-2 crash | **OTHER-GPU** | `docs/SESSION_CHANGELOG_20260120.md:29` "SMU: response:0xFFFFFFFF for message:TransferTableSmu2Dram" | — The record says "SMU exhaustion from excessive polling" (`:6`), meaning monitoring overwhelmed the SMU, not job load. The line is given as a "signature for future reference" (`:27`), not as an excerpt with a timestamp |
| A-06 | E7 | 01-20 | both rigs | 120 trials at once (40 jobs/rig), Step 2 WATCHER, SSH | both rigs crash within 60 s; SSH hung; HiveOS reboot | **OVERLOAD** | `docs/PROPOSAL_Job_Batching_Pipeline_Stability.md:66` "Simultaneous job initialization overload."; `:21` "crashes both rigs within 60 seconds" | — (no kernel line) |
| A-10 | E8 | 01-22 | rig-6600b (GPU[4]) | Step 3 at batch cap 6, SSH | kernel deadlock/crash | **OVERLOAD** | `docs/SESSION_CHANGELOG_20260122.md:21` "Kernel deadlock/crash … SMU-faulty GPU[4] amplifies overload"; `:24` "same overload event" | — (no kernel line. Secondary: GPU[4] SMU fault, no line quoted) |
| A-11 | E8 | 01-22 | rig-6600 (GPU 11) | same Step-3 run | GPU 11 "unknown" state; driver reset failure | **OVERLOAD** | `docs/SESSION_CHANGELOG_20260122.md:22` "Driver reset failure under I/O + HIP storms"; `:24` "same overload event" | — (secondary: OTHER-GPU reset failure, no kernel line) |
| A-12 | E9 | 01-23 | rig-6600 (+rig-6600b) | 7 parallel workers, Step 3, SSH | 1–3 first-wave job failures per run | **OVERLOAD** | `docs/SESSION_CHANGELOG_20260123_PART2.md:22` "rig-6600 kernel: Out of memory: Killed process 13994 (python)" | — |
| A-13 | E10 | 01-24 | rig-6600, rig-6600b (+Zeus) | 4 GPUs (post-fix #1), chunk 1000, Step 3, SSH | 61/99 jobs, rest orphaned | **OTHER** | `docs/SESSION_CHANGELOG_20260124_OOM_FIX.md:239` "Post-fix #1 \| 4 \| 4 \| 61/99 \| ❌ Jobs orphaned" | — (coordinator bug) |
| A-15 | E11 | 01-24 | rig-6600 GPU[10] (22:00.0) | 2-hour WATCHER pipeline run | rig crashed mid-run; the run also failed for zero survivors | **OTHER-GPU** | `docs/PROPOSAL_Infrastructure_Improvements_v1_0.md:117` "SMU failures (0xFFFFFFFF) that cascaded into system crash" | — (the SMU code is paraphrased, not a verbatim log line. The run failure also has an OTHER cause, "insufficient Optuna trials finding zero survivors", `:8`) |
| A-16 | E10 | 01-25 (pre-fix run) | rig-6600/rig-6600b | Step 3, full_scoring chunk_size 5000 | 2 OOM job failures | **OVERLOAD** | `docs/SESSION_CHANGELOG_20260125.md:29` "0 failures (was 2 OOM failures before fix)" | — (probably the same OOM regime as A-12. No new kernel line) |
| A-18 | E12 | ≤01-26 | AMD rig ("node-11") | Step 1 forward sieve, ~26 chunks × ~19K seeds (500K), coordinator.py | intermittent GPU Hang | **OTHER-GPU** | `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:1010` "Error: HW Exception by GPU node-11... reason: GPU Hang" | — The record blames VRAM fragmentation (`:1007`). CH2 is a 2026-11 retrospective. The rig is not named |
| A-20 | E16 | 02-04 | rig-6600c | Soak A (WATCHER soak) | rig rebooted during soak | **UNKNOWN** | `docs/SESSION_CHANGELOG_20260204_S62.md:80` "rig-6600c rebooted during Soak A (no impact — separate machine)" | Cause not recorded. No dmesg; it may have been an operator reboot |
| A-21 | E15 | ≤02-11 | rig-6600b | 8 GPUs on PCIe Gen1, at GPU enumeration (boot) | rig-6600b crash | **OVERLOAD** | `docs/SESSION_CHANGELOG_20260211_S80.md:58` "Kernel soft lockup (`CPU#2 stuck for 26s` in `systemd-udevd`)" | — It happened during enumeration, not compute load. The record's mechanism is "GPU sleep/wake storms" (`:55`) |
| A-22 | E16 | 02-18 | rig-6600b (.154) | WATCHER preflight from Zeus | intermittent SSH failure | **OTHER** | `docs/SESSION_CHANGELOG_20260218_S96A.md:199` "Network-layer issue, not rig issue. Non-blocking" | — (this is not a crash; listed for completeness) |
| A-23 | E18 | ≤02-23 | rig-6600b | prior Step-2 run, scripts_coordinator | D-state zombie processes; reboot required | **OVERLOAD** | `docs/SESSION_CHANGELOG_20260223_S109.md:130` "D-state zombies on rig-6600b from ROCm driver deadlock during concurrent initialization" | — (date of the originating run not recorded) |
| A-24 | E18 | 02-23 | 4 nodes | Step 2, 20 trials | first run exit −15 | **OTHER** | `docs/SESSION_CHANGELOG_20260223_S109.md:48` "User killed collection process during SSH/SFTP pull-back phase" | — |
| A-26 | E19 | 02-26 | rig-6600 (.120) GPU 2 | 26 GPUs, 10M seeds × 50 trials, Step 1 (hybrid pass), coordinator.py SSH | GPU Hang (HSA), in 4 jobs | **OTHER-GPU** | `logs/s112_steps1_2_real_data.log:6900` "HW Exception by GPU node-3 … reason :GPU Hang"; doc `docs/SESSION_CHANGELOG_20260226_S112.md:116` | — The doc says 1 hang "recovered via retry". The log shows 4 jobs, each "failed permanently after 3 retries" (`log:6896, 8112, 11697, 24761` region). The line is user-space HSA, not kernel |
| A-28 | E24/E25 | 03-09 | rig-6600c (.162) | 4 workers (pool), 2M AMD cap, Step 1 pass 1, PWC-in-coordinator SSH | libamdhip64 segfaults ~30–60 s in; workers dead | **OTHER** | `docs/SESSION_CHANGELOG_20260309_S133.md:21` "semaphore that was never released … causing a deadlock" | — S132 hypothesis was "8-way HIP concurrency = memory pressure" (`docs/SESSION_CHANGELOG_20260309_S132.md:49`). No dmesg |
| A-29 | E25 | 03-09 | 3 rigs (all 12 workers) | java_lcg_hybrid passes 3–4, persistent SSH | all workers dead | **OTHER** | `docs/SESSION_CHANGELOG_20260309_S133.md:36` "TypeError: run_sieve() got an unexpected keyword argument 'strategies'" | — |
| A-30 | E27 | 03-15 | rig-6600c ("162 rig") | 100K seeds, 2 trials, 3,846-seed chunks, Step 1, PWC SSH | rig hung; Phase 2 terminated | **OVERLOAD** | `docs/SESSION_CHANGELOG_20260315_S145.md:145` "162 rig hung on 3,846-seed chunks (SSH overhead > job time)" | — Weak: the record blames SSH dispatch overhead and does not use the word overload. No dmesg |
| A-32 | E28/E29/E30 | ≤03-16 (S146 era) | AMD rigs, GPUs 4–7 | pool >4/rig, HIP_VISIBLE_DEVICES masking, PWC SSH | "any value above 4 caused rig crashes" | **OTHER** | `docs/SESSION_CHANGELOG_20260321_S149.md:27-28` "HIP masking is unreliable for GPUs 4-7 — workers crashed on startup" | — The individual crash runs are not recorded, only the later claim (`docs/TB_RULING_REQUEST_ROCR_ISOLATION_S149.md:20-21`). S134 instead framed it as "memory pressure validated at 4-way" (`docs/SESSION_CHANGELOG_20260310_S134.md:70`) |
| A-34 | E30 | 03-20/21 | rig not named | 8/rig, Run 1 trial 7, hybrid reverse pass, PWC SSH | Run 1 crashed; 22,145 bidi lost | **UNKNOWN** | `docs/SESSION_CHANGELOG_20260321_S149.md:65-66` "Run 1 crashed during Trial 7's hybrid reverse pass" | Rig, crash type and cause are not recorded. No dmesg |
| A-36 | E31/E32 | 03-21 (from ~10:30 S150 through S151) | rrig6600c (i5-8400T) | 8 workers, slim_v1, sshd MaxSessions 5, PWC SSH | repeated rig crashes; physical reboots | **OVERLOAD** | `docs/ROOT_CAUSE_ANALYSIS_RRIG6600C_S151.md:70` "rig overloaded → system crash → physical reboot required" | — The cascade is MaxSessions 5 → respawn storm (`:52-71`). No kernel line. S152 later blames the same slim_v1 crash on a CuPy cache race (A-40) |
| A-37 | E32 | 03-21 ~15:10 | rrig6600c GPU7 | 6 workers (38393b5) → 8 | "GPU7 fault" | **UNKNOWN** | commit cc113b5: "GPU7 fault was hardware not software" | No fault type, log or line recorded. This is the only mention |
| A-39 | E32 | 03-21 | cluster (rrig6600c at 4) | Run 1, trial 1 done, PWC SSH | process died before NPZ checkpoint; survivors lost | **UNKNOWN** | `docs/SESSION_CHANGELOG_20260321_S151_FINAL.md:119` "process killed before checkpoint fired. Survivors lost." | The killed process and the cause are not recorded. It may be Zeus-side, not a rig |
| A-40 | E33 | 03-22 | rrig6600c | 8 workers, slim_v1, no per-worker masking, PWC SSH | "deterministic crash" at first RawKernel compile | **OTHER** | `docs/SESSION_CHANGELOG_20260322_S152.md:30` "they race on the shared `~/.cupy/kernel_cache/` directory" | — (software race, TB-confirmed fix. No rig-side log) |
| A-42 | E33 | 03-22 ~22:09 | rrig6600b | 8 workers, ~2 h run, PWC SSH | workers died | **UNKNOWN** | `docs/TODO_MASTER_S152.md:49` "rrig6600b workers died at ~22:09 after ~2 hours" | Cause not recorded. The record suspects SSH session drops/keepalive (`:47-50`). No dmesg |
| A-44 | E35 | 03-23 ~12:56 | rrig6600c | 8 workers, Run 1 Trial 1, ~21 reverse jobs/worker, PWC SSH | crash + reboot; OOM kill | **OVERLOAD** | `docs/SESSION_CHANGELOG_20260323_S154.md:28` "Out of memory: Killed process 3008 (python) total-vm:41452828kB" | — |
| A-45 | E35 | 03-23 ~13:35 | rrig6600c | same, after `del`+gc fix | OOM kill again | **OVERLOAD** | `docs/SESSION_CHANGELOG_20260323_S154.md:35` "Out of memory: Killed process 3379 (python) total-vm:41452828kB" | — |
| A-46 | E35 | 03-23 ~10:09 | rrig6600c | — (syncthing) | syncthing invoked the OOM killer | **OVERLOAD** | `docs/SESSION_CHANGELOG_20260323_S154.md:44-45` "a 10:09 crash where **syncthing** invoked the OOM killer" | — (the doc reports it from kern.log but does not quote the line) |
| A-47 | E36 | 03-23→25 | rrig6600c | 8 workers + CuPy pool cap 256 MB, PWC SSH | OOM persists | **OVERLOAD** | `docs/SESSION_CHANGELOG_20260324_S155.md:25` "CuPy memory pool cap (256MB) … OOM persists"; `:31` identical total-vm signature | — |
| A-48 | E36 | 03-24 | rrig6600c | 8 + CUPY_GPU_MEMORY_LIMIT env | OOM persists | **OVERLOAD** | `docs/SESSION_CHANGELOG_20260324_S155.md:26` "CUPY_GPU_MEMORY_LIMIT env var … OOM persists" | — |
| A-49 | E36 | 03-24 | rrig6600c | 8 + ROCR_VISIBLE_DEVICES restored | OOM persists | **OVERLOAD** | `docs/SESSION_CHANGELOG_20260324_S155.md:27` "ROCR_VISIBLE_DEVICES isolation … OOM persists" | — |
| A-50 | E36 | 03-24 | rrig6600c | 8 + new M.2 SATA | OOM persists | **OVERLOAD** | `docs/SESSION_CHANGELOG_20260324_S155.md:28` "Replaced with Samsung 860 EVO — e820 cleaner, OOM persists" | — |
| A-51 | E36 | 03-24/25 | rrig6600c | 8 + i5-9400 CPU | OOM persists | **OVERLOAD** | `docs/SESSION_CHANGELOG_20260324_S155.md:29` "CPU upgrade (i5-8400T → i5-9400) … OOM persists" | — |
| A-53 | E37 | 03-25 (pre-BIOS) | rrig6600c | stale + new workers, py_procs 9, PWC SSH | kernel freeze | **OVERLOAD** | `docs/SESSION_CHANGELOG_20260326_S156.md:12` "zombie worker accumulation … py_procs to climb to 12+ → … kernel freeze"; `:26` "py_procs=9 … CRASH" | — (no kernel line; mechanism = stacked workers) |
| A-54 | E37 | 03-25 (pre-BIOS) | rrig6600c | py_procs 12, PWC SSH | kernel freeze | **OVERLOAD** | `docs/SESSION_CHANGELOG_20260326_S156.md:27` "py_procs=12 → PageTables=19,124 kB   → CRASH" | — |
| A-55 | E37 / iso 03-25/26 | 03-25 (pre-BIOS) | rrig6600c | Test C, 4 workers (also clean first spawn at py_procs 6), PWC SSH | hard kernel freeze | **OTHER** | `docs/SESSION_CHANGELOG_20260326_S156.md:14` "BIOS NVRAM corruption — hard kernel freeze during ROCm context initialization"; `:100` | — The record says BIOS NVRAM (`:65-71`). TB in S157 instead reads the 4-worker crash as "concurrent ROCm init timing" (`docs/S157_CHAT_PROMPT.md:43`). No kernel line |
| A-56 | E37 / iso 03-25/26 | 03-26 (post-BIOS) | rrig6600c | Test C, 4 workers, PWC SSH | CuPy OOM; machine stayed up | **OVERLOAD** | `docs/SESSION_CHANGELOG_20260326_S156.md:101` "4 workers (post-BIOS) \| 4 \| ❌ CuPy OOM … Machine stayed up" | — (Python CuPy exception, not a kernel OOM kill) |
| A-60 | E38 | ≤03-27 | rrig6600c | 3 vs 4 workers, PWC SSH | 4 workers: HARD CRASH (3 stable) | **UNKNOWN** | `docs/S157_CHAT_PROMPT.md:40` "4 workers → py_procs=6, PageTables=~9,260 kB  → HARD CRASH" | No kernel line. Pre- or post-BIOS is not stated. TB's reading is only a lean, "points to concurrent ROCm init timing" (`:43`), and its test plan has no results |
| A-63 | E39 | 03-29 | rrig6600c + Zeus | 8 AMD, 537 chunks, reverse, PWC SSH | stalled at chunk 88/537 | **UNKNOWN** | `docs/SESSION_CHANGELOG_20260329_S158.md:32` "Stalled chunk 88/537 — hang evidence" | No cause or kernel line. `logs/isolation_test_rrig6600c.log` has 0 fault-pattern hits |
| A-65 | E39 | 03-29 | all 3 rigs + Zeus (24 AMD) | 8/rig, 537 chunks, PWC SSH (S158B status not stated) | hard kernel panic on rrig6600c | **OVERLOAD** | `docs/SESSION_CHANGELOG_20260329_S158.md:22` "Workers marked dead → respawn storm → rig kernel panic"; `:34` | — No panic text was recorded. The ZMQ proposal says "Kernel panics during worker spawn (rrig6600c repeatedly)" (`docs/PROPOSAL_ZMQ_SQLITE_COORDINATOR_S158D_v1_0.md:17`) |
| A-66 | E39 | 03-29 | rrig6600, rrig6600b | S158C inter-node spawn stagger 10 s / 30 s, PWC SSH | spawn crashes; reverted | **OTHER** | `docs/SESSION_CHANGELOG_20260329_S158.md:70` "Inter-node stagger caused spawn crashes on rrig6600 and rrig6600b" | — Weak: the record blames the patch, but no crash mechanism or log is recorded. `logs/s158c_validation_run.log` has 0 fault-pattern hits |
| B-01 | E1; L4 #1 | 2026-03-30 | Zeus + rrig6600 (8) | 1 AMD rig + Zeus, 8 AMD workers, 2M, java_lcg Step 1, ZMQ+systemd-run, stock driver | 2-rig validation run; final result never recorded | **UNKNOWN** | `TODO_MASTER_S159.md:10` "Confirm rrig6600 2-rig run result" | Run outcome. Netconsole not yet deployed, so there is no kernel capture |
| B-03 | E6; L4 #2 | 2026-04-02 (time n/r) | "both rigs" (rrig6600, rrig6600c implied) | ZMQ, 5M, stock | Crash on both rigs | **UNKNOWN** | CL159orig:76 "Run at 5M seeds → crash (both rigs)" | No per-run kernel line. B-02 may be this run's rrig6600c capture, but the record does not say so. Nothing for rrig6600 |
| B-04 | E6; L4 #3 | 2026-04-02 (time n/r) | n/r | ZMQ, 2M, stock | Crashed again at 2M | **UNKNOWN** | CL159orig:77 "Run at 2M seeds → also crashed"; `docs/TB_SUBMISSION_S159G_RIG6600_CRASHES.md:107` | Rig and kernel line. It may be the same event as B-05 [UNVERIFIED] |
| B-06 | not an E-entry (E13 context) | 2026-04-03 08:34-10:14 (`L/zmq_s160v2/v3/v4_envcheck.log`) | coordinator | ZMQ, env-check v1–v3 | Env-invariant check failed across iterations (no D-Bus in SSH exec); Step 1 exit code 1 | **OTHER** | `docs/SESSION_CHANGELOG_S160.md:29` "failed across 4 iterations"; :34 "Failed to connect to bus" | — |
| B-09 | E15 | 2026-04-03, trial 2 of the v5 run | coordinator (ZMQ leases) | as B-08 | All 537 hybrid chunks failed via lease exhaustion; run ends code -9 (`L/zmq_s160v5_cleanup.log:5801`) | **OTHER** | `SESSION_CHANGELOG_S160.md:109` "not a crash — workers were alive" | — |
| B-10 | not an E-entry (E16 context) | 2026-04-03 S161 | AMD workers | TCP-PWC v1 worker (ROCm import before connect), sequential launch | Only 1/8 workers connected within timeout | **OTHER** | `docs/SESSION_CHANGELOG_S161.md:65` "only 1/8 workers connected within timeout" | — |
| B-20 | E24; L4 #21 | 2026-04-05 Exp1 (before DKMS, per CL162b:29-35 order) | rrig6600c | 3 rigs, `HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0` on .162, stock | "No effect": crash persisted | **OVERLOAD** | CL162b:15-16 "cannot handle 8 concurrent GPU compute workers under full 3-rig concurrent load"; CL162b:32 | No kernel line for this run. FIN162:29 says "partial" |
| B-21 | E25; L4 #22 (c) | 2026-04-05 | rrig6600c | `amdgpu.cwsr_enable=0` on c, stock | "No effect": crash persisted | **OVERLOAD** | CL162b:33 "cwsr_enable=0 GRUB param on rrig6600c ❌ No effect"; attribution CL162b:15-16 | No kernel line. CL162:83 instead says cwsr is "required" on c |
| B-22 | E25; L4 #22 (b) | 2026-04-05 | rrig6600b | cwsr_enable=0 on b | KIQ fence timeouts; "broke stable rig" | **OTHER-GPU** | `SESSION_CHANGELOG_S162.md:84` "Adding to rrig6600b caused KIQ fence timeouts" | — (no GCVM recorded) |
| B-23 | E25; L4 #22 (rrig6600) | 2026-04-05 | rrig6600 | cwsr_enable=0 on rrig6600 | "instability" | **UNKNOWN** | `SESSION_CHANGELOG_S162.md:85` "Adding to rrig6600 caused instability" | What failed, and any kernel line |
| B-24 | E26; L4 #23 | 2026-04-05 | 3 rigs, DKMS on rrig6600c only | cap n/r (2M implied), TCP-PWC | "Extended stability to 25 min", i.e. failed at ~25 min | **UNKNOWN** | CL162b:34 "amdgpu-dkms 6.12.12 on rrig6600c only ✅ Extended stability to 25 min" | Rig, kernel line. Contradicted by FIN162:14, 42 ("all 3 trials clean") |
| B-25 | E26; L4 #24 | 2026-04-05 (Trial 1 31:31) | 3 rigs DKMS, 26 | 24+2, 3 trials, cap n/r, TCP-PWC | T1 done; c "faulted during reverse sieve"; c then offline, <24 workers, Step 1 ESCALATED | **UNKNOWN** | `SESSION_CHANGELOG_S162.md:102` "rrig6600c faulted during reverse sieve"; :116 | Kernel line for the c fault and outage. CL162b:22, 78-80 claim all trials clean |
| B-26 | E27 | 2026-04-05 ~11:57 (8d91311) | 3 rigs (DKMS) | reverse-sieve `cp.zeros(1)` warm-up + VRAM logging | "made things worse or had no effect" | **UNKNOWN** | `docs/SESSION_CHANGELOG_S162_FINAL.md:52` "All patches made things worse or had no effect post-DKMS" | Run, rig, time-to-fault, kernel line |
| B-27 | E28; L4 #28 | 2026-04-05 13:22-13:55 (`L/s162_nopool_run1.log` 13:38) | 3 rigs | DKMS [UNVERIFIED], CuPy pool disabled (d79e558) | Crash at 358 s | **UNKNOWN** | `SESSION_CHANGELOG_S162_FINAL.md:28` "made things worse (crashed at 358s)" | Rig, kernel line |
| B-28 | E29; L4 #29 | 2026-04-05 14:11-15:05 (`L/s162_scratch_reclaim_run1.log` 14:12) | 3 rigs | `HSA_NO_SCRATCH_RECLAIM=1` all AMD (606f94f) | Crash at 219 s | **UNKNOWN** | `SESSION_CHANGELOG_S162_FINAL.md:27` "made things worse (crashed at 219s)" | Rig, kernel line |
| B-29 | E31; L4 #26; iso 04-05 | 2026-04-05 15:30 (`L/s162_1m_seeds_run1.log`) | 3 rigs | 1M, DKMS, TCP-PWC | "crashed mid-run under 3-rig load" | **UNKNOWN** | `docs/PROPOSAL_FREE_ALL_BLOCKS_REPLACEMENT_v1_0.md:265` "1M seeds — crashed mid-run under 3-rig load" | Rig, kernel line. The record's hypothesis is a concurrent free_all_blocks race (PRFAB:268-269), not overload |
| B-30 | E30; L4 #30 | 2026-04-05 16:23-16:51 (`L/s162_warmup_patch_run1/2.log`, 1M [UNVERIFIED link]) | 3 rigs (c named) | 8/rig, DKMS, kernel pre-compile (22bb145) | "made crashes worse" | **UNKNOWN** | ac32cd9 subject "revert kernel pre-compile warmup patch — made crashes worse" | Rig, timing, kernel line |
| B-31 | E31; L4 #27; iso 04-05 | 2026-04-05 16:56 (`L/s162_500k_run1.log`) | 3 rigs | 500K, DKMS | "delayed crashes, did not fix" | **UNKNOWN** | `SESSION_CHANGELOG_S162_FINAL.md:33` "Seed cap reduction (1M, 500K) — delayed crashes" | Rig, kernel line |
| B-32 | E31; L4 #25; iso 04-05 | 2026-04-05 (time n/r) | 3 rigs | 2M, DKMS, TCP-PWC | Crash at ~1400 s | **UNKNOWN** | `PROPOSAL_FREE_ALL_BLOCKS_REPLACEMENT_v1_0.md:264` "~1400 seconds (DKMS)" | Rig, run log, kernel line |
| B-33 | E32; L4 #31 | 2026-04-05 late | rrig6600 + rrig6600c at the same time | DKMS, perf auto (c 2750 MHz, others 700 MHz) | Forward-sieve crashes on both | **UNKNOWN** | `SESSION_CHANGELOG_S162.md:121` "forward sieve started crashing on rrig6600 AND rrig6600c" | Kernel line. The record says "Likely cumulative GPU state degradation" (:123), a hypothesis |
| B-35 | E34 | 2026-04-05, Trial 3 of B-34 | Zeus 2× 3080 Ti | Zeus local path, `_localhost_semaphore=2` | 250/10,738 chunks failed, `cudaErrorDevicesUnavailable` | **OTHER** | `SESSION_CHANGELOG_S162_VICTORY.md:46` "cudaErrorDevicesUnavailable"; :49 "250/10,738 chunks failed" | — |
| B-37 | E36 | 2026-04-06, after T1 of B-36 | Zeus `window_optimizer.py` | as B-36 | Run hung 48+ min at 139% CPU / 2 GB; killed | **UNKNOWN** | `TB_SUMMARY_S163.md:54-55` "Run hung after Trial 1 completion" | Cause. TBS163:121-123 asks TB about it. No rig fault recorded |
| B-38 | E36 | 2026-04-06 | Zeus local GPUs | Zeus local path | `cudaErrorDevicesUnavailable` on every local chunk | **OTHER** | `TB_SUMMARY_S163.md:59-60` "Zeus local GPU path failed on EVERY local chunk" | — |
| B-39 | E38 | 2026-04-07..10 | rrig6600 | multimedia package removal | NIC broke; rig crash-looped on boot | **OTHER** | `CHANGELOG_S163.md:55` "rrig6600 package removal broke NIC — rig crash-looped on boot" | — |
| B-42 | E41 | 2026-04-11 18:45 → 04-12 08:50 (`L/s163_step1_10k_2trial*.log` v1–v11) | 3 rigs + Zeus | 10K, 2 trials; transport fell back to SSH; v1 worker; 60 s heartbeat; worker_id collision | Repeated failed or degraded attempts (workers not connecting; 3/24 workers) | **OTHER** | `SESSION_CHANGELOG_S163.md:23` "always defaulted to `"ssh"`"; :59 "Only 1 worker per rig (3 total) survived" | — |
| B-45 | E43 | 2026-04-12/13 (`L/s163_ladder_250k_v2.log:38470` code -9 [UNVERIFIED link]) | Zeus coordinator | 250K, 2.6M survivors | `json.dump` took 2h44m and hung the process | **OTHER** | `SESSION_CHANGELOG_S163_FINAL.md:56-57` "json.dump() … took 2h44m and hung the process" | — |
| B-48 | E44; L4 #39; iso 04-12/13 | 2026-04-13 (`L/s163_ladder_500k.log` 16:56, no result; `_v2` PASSED at :9607) | rrig6600 | 24+2, 500K, TCP-PWC, DKMS, set_limit | rrig6600 crashed | **OVERLOAD** | `SESSION_CHANGELOG_S163_FINAL.md:74` "rrig6600 crashed — memory accumulation" | No kernel line for this crash; no 04-13 netconsole in the main tree. :109-112 quotes the GCVM signature only generically ("caused by unbounded CuPy memory pool growth"), not for this run. Which log is the crash is not recorded |
| B-49 | L4 #40 | 2026-04-12/13 (generalisation) | 3 rigs | ≥300K without a reboot between runs | "triggers GPU page faults" | **OVERLOAD** | `SESSION_CHANGELOG_S163_FINAL.md:79-80` "CuPy memory pool accumulates state across trials and triggers GPU page faults" | No individual run or kernel line |
| B-50 | E46; L4 #41 | 2026-04-15 (commit 39ff260 20:01) | n/r | n/r | Step 1 survivors committed (1069 bidirectional) | **UNKNOWN** | 39ff260 subject "Step 1 survivors — 1069 bidirectional, W5_O53" | Config, run log and fault status |
| B-51 | E48; L4 #42 | 2026-04-16 18:01 (`L/s163_100k_no_audio.log:107830` code -15) | all 3 rigs (rrig6600b first ~18:13) | 24 workers, 100K, 3 trials, TCP-PWC, DKMS, 3c20d2d, snd/gfxoff modprobe | All 3 rigs crashed and rebooted; coordinator spun 1.5 h; only T1 done | **UNKNOWN** | `SESSION_CHANGELOG_S163_RIG_FIXES.md:14` "All 3 rigs crashed and rebooted"; :39 "No GPU fault signatures in kern.log" | Kernel lines. The netconsole timeline the doc cites (:26) is not in the main tree: NCL starts 04-17 10:36 (NCL:1). The record's "initramfs not baked" cause (:43-46) conflicts with its own initrd timestamps (:57, :66, :75) |
| B-54 | **not in register** | 2026-04-17 10:58 reboot; 11:02:16 | rrig6600b 09:00.0 (uptime 258 s) | no run active (all rigs NETCONSOLE READY 10:58-10:59, NCL:245, 317, 318) | sdma0 ring timeout and ring reset | **OTHER-GPU** | NCL:319 "ring sdma0 timeout, signaled seq=5954, emitted seq=5957" | — (no GCVM on .154; reboot cause n/r) |
| B-55 | **not in register** | 2026-04-17 14:01-14:19 (`L/s163_100k_mem_instrumented.log`) | n/r | 100K, 2 trials, MEM instrumentation | Step 1 failed with code -9; ESCALATED | **UNKNOWN** | `L/s163_100k_mem_instrumented.log:86118` "Step 1 failed with code -9" | Source of the SIGKILL. NCL was up but has no rig line between 11:02 and 14:34 |
| B-57 | **not in register** | 2026-04-17 14:40:16 (9 s after B-56's Step 1 PASSED) | rrig6600 03:00.0 | after the 100K run | qcm fence timeout, then evict failure, then device lost from bus, repeating | **OTHER-GPU** | NCL:544 "qcm fence wait loop timeout expired"; NCL:547 "device lost from bus!" | — (no .120 GCVM in that window) |
| B-58 | **not in register** | 2026-04-17 16:21 (`L/s163_revert_test_1621.log`) | Zeus | watcher launch | Watcher failed at import | **OTHER** | `L/s163_revert_test_1621.log:4` "ModuleNotFoundError: No module named 'agents'" | — |
| B-62 | **not in register** | 2026-04-17 17:55-18:02 (`L/s163_karg_ladder_step1_1755.log`) | n/r | 10K, 5 trials, KARG ladder | Step 1 code -9; ESCALATED | **UNKNOWN** | `L/s163_karg_ladder_step1_1755.log:4576` "Step 1 failed with code -9" | **Netconsole listener was down**: NCL:1185 "Shutting down listener" after 17:25:26; restart line NCL:1186 has no timestamp; next line 21:16 |
| B-63 | **not in register** | 2026-04-17 18:45-19:14 (`L/s163_karg_ladder_step1_retry_1845.log`) | n/r | 10K, 5 trials | Log ends mid-run (last line 19:14:23), no Step result | **UNKNOWN** | `L/s163_karg_ladder_step1_retry_1845.log:23563` (last line, chunk 8747 dispatch) | Outcome; listener down (NCL:1185) |
| B-64 | **not in register** | 2026-04-17 19:44 → 20:17 (`L/s163_karg_100k_5trials_1944.log`) | n/r | 100K, 5 trials | Log ends mid-run at 20:17:49, then relaunched at 20:44 (Part C E1) | **UNKNOWN** | `L/s163_karg_100k_5trials_1944.log:112848` (last line) | Outcome; listener down (NCL:1185) |
| C-01 | E1/E2; ISO 04-17 | 04-17 20:56:11 | rrig6600b GPU1 | 3 rigs+Zeus, 8/rig (24), seed_cap_amd 100K, 5 trials, TCP-PWC, KARG+JSON guard, stock DPM, GDM active | GPU1 worker died silently mid-job; self-recovered 21:06:20 | **UNKNOWN** | IR:72 "No GCVM fault in netconsole"; ML/s163_karg_json_100k_5trials_2044.log:22976 "transient failure (TCP worker timeout)" 20:55:11 | No .154 kernel line between NL:1182 (17:23:55) and NL:1579 (04-18). Listener restarted (NL:1185-1186) at an unrecorded time, so liveness at 20:56 is unproven; the first post-restart line is NL:1187 at 21:16:45. Worker-log grep result n/r (IR:199-204). IR:240 "GDM/Xorg leading explanation" is a hypothesis. |
| C-03 | E6 | 04-18 09:05–09:08 | rrig6600b 09:00.0 | idle after reboot, no run log active | SDMA ring timeouts at 82/172/278 s uptime | **OTHER-GPU** | NL:1579 "[82.458543] … ring sdma0 timeout"; IR:241 | — |
| C-05 | LOG-ONLY | 04-18 19:04:32–19:14:09 | rrig6600 03:00.0; rrig6600c 06:00.0 | s163_karg_gdm_clean_…_1737: 24 workers, 100K, 5 trials, GDM disabled | SMU 0xFFFFFFFF on both rigs every ~2 min, launch+85 s; .162 evict failure; workers 24→8 at 19:14:04; both rebooted (READY 19:42) | **OTHER-GPU** | NL:1592 ".120 SMU: response:0xFFFFFFFF … TransferTableSmu2Dram"; NL:1594 (.162); NL:1776 "Failed to evict process queues"; ML/…_1737.log:134778 "8 ready" | — |
| C-09 | E10 "rrig6600/c crashed Trial 3" | 04-19 15:36:38–15:36:54 | rrig6600 03:00.0; rrig6600c 16:00.0 | same as C-08 | SMU 0xFFFFFFFF and evict failure on both rigs; ready 24→8 at 15:36:50 | **OTHER-GPU** | NL:2015 ".120 SMU: response:0xFFFFFFFF"; NL:2062 (.162); NL:2106 "Failed to evict process queues" | — |
| C-10 | E13; ISO 04-19 S164 | 04-19 17:39:22 | rrig6600 19:00.0 (card7) | s164_production_1723: 24 workers, **seed_cap_amd 200,000** (log CLI; register says n/r), 3 trials, stock ~2750 MHz | KFD queue preemption failed → SMU → MODE1 reset failed → sdma timeout → kworkers blocked 122 s; ready 24→16 | **OTHER-GPU** | NL:2114 "Queue preemption failed"; NL:2127 "GPU mode1 reset failed"; NL:2135 "blocked for more than 122 seconds"; ML/s164_production_1723.log:7 CLI 'seed_cap_amd': 200000 | — (S164:46-50 attributes it to 2750 MHz, DC, HDMI audio and lockup_timeout.) |
| C-12 | LOG-ONLY | 04-20 17:57 / 18:04 | — | s165_production_1757 / _1804: 100K, 5 trials | aborted at start: pre-spawn cleanup SSH failed / preflight SSH unreachable (rigs down after C-11) | **OTHER** | ML/s165_production_1804.log:21 "Preflight BLOCKED: SSH unreachable" | — |
| C-14 | E16 (09:00.0 ASIC reset) | 04-20 19:12:51–19:13:36 | rrig6600b 09:00.0 | same run (+22 min) | sdma1 timeout, MODE1 reset failed; no GCVM in this burst | **OTHER-GPU** | NL:3222 "ring sdma1 timeout"; NL:3230 "GPU mode1 reset failed" | — |
| C-16 | LOG-ONLY | 04-20 19:28–19:29 | rrig6600b | no run log active | ring timeouts and resets | **OTHER-GPU** | NL:3375 ring timeout; NL:3383 reset | — |
| C-19 | LOG-ONLY | 04-21 17:06–17:27 | rrig6600 (7/8 GPUs at launch) | s166_accum_test_1706: 26 workers, 100K, 5 trials | card missing at launch; 1361 chunks hipErrorNoDevice from 17:18:08 | **OTHER-GPU** | ML/s166_accum_test_1706.log:35 "GPU_COUNT_MISMATCH: 7/8"; :22521 "hipErrorNoDevice" | Run-log evidence only. Listener restarted (NL:3886-3887) at an unrecorded time. |
| C-20 | LOG-ONLY | 04-21 17:33:29 | rrig6600 19:00.0 | no run log active (6 min after C-19 ended; orphan workers possible, S170:182-188) | queue preemption failed → SMU → MODE1 reset failed -121 | **OTHER-GPU** | NL:3888 "Queue preemption failed"; NL:3899 "ASIC reset failed with error, -121" | — |
| C-21 | E22 run1 | 04-21 17:42–18:06 | rrig6600 (7/8) | s166_run1_1742: 26 workers, 100K, 2 trials, warm-start params | 2036 chunks hipErrorNoDevice from 17:46:12. Card 19:00.0 had failed in C-20 and was not rebooted until 18:17 | **OTHER-GPU** | ML/s166_run1_1742.log:35 "GPU_COUNT_MISMATCH: 7/8"; :22519 "hipErrorNoDevice" | — (S166:138 records "✅ Complete".) |
| C-22 | LOG-ONLY | 04-21 18:11:50–18:13:59 | rrig6600 19:00.0 | 5 min after run1 ended (18:06:53) | queue preemption → SMU → GPU recovery failed → NULL deref in amdgpu_device_gpu_recover | **OTHER-GPU** | NL:3905 "Queue preemption failed"; NL:3929 "GPU Recovery Failed: -110"; NL:3945 NULL deref | — |
| C-23 | E22 run2 | 04-21 18:45:55–18:53:23 | rig n/r, "card1" | s166_run2_1845: 26 workers, 100K, 2 trials (DB override W17_O36) | "✅ Complete, card1 crash" | **UNKNOWN** | S166:139 "Complete, card1 crash" | Listener live NL:3997 (18:20) → NL:3998 (19:27) with no line. Run log shows 26 ready and no warnings. Rig, card, time and signature n/r. |
| C-24 | E22 run3 — fault not recorded in docs | 04-21 19:27:59 | rrig6600 03:00.0 | s166_run3_1917: 26 workers, 100K, 2 trials, warm-start | SMU 0xFFFFFFFF, evict failure; ready 26→18 | **OTHER-GPU** | NL:3998 "SMU: response:0xFFFFFFFF"; NL:4042 "Failed to evict"; ML/s166_run3_1917.log:21720 timeout 19:27:53 | — (S166:140 records "✅ Complete".) |
| C-27 | E22 run5 | 04-22 16:26–17:42 | Zeus host | s166_run5_1626: 26 workers, 100K, 2 trials, W2_O30 | 23,387,198 survivors; Zeus OOM SIGKILL after 60 GB RAM | **OVERLOAD** | S166:129 "Zeus OOM killed the process (code -9) after consuming 60GB"; ML/s166_run5_1626.log:86109 "code -9" | — (No rig fault: C-26.) |
| C-28 | LOG-ONLY | 04-22 18:25–19:18 | — | s166_run6_1825: 100K, 2 trials | compute done 18:38:33; Step 1 exit -15 at 19:18:29 | **UNKNOWN** | ML/s166_run6_1825.log:172226 "failed with code -15" | Source of the SIGTERM n/r. No rig kernel line (C-26 window). |
| C-29 | LOG-ONLY | 04-22 20:01–20:55 | Zeus | s166_run7_2001: 100K, warm-start | 291,112,579 forward survivors; Step 1 exit -9 at 20:55:17 | **OTHER** | ML/s166_run7_2001.log:21684 "code -9"; log "291,112,579 survivors — full JSON skipped" | — (Survivor-explosion SIGKILL. It matches the C-27 OOM pattern, but no OOM is recorded for this run.) |
| C-30 | LOG-ONLY | 04-22 20:10; 04-23 15:32, 16:55, 19:52, 19:55, 19:58, 20:01 | — | s166_run8, run9, run12_1655, run14 ×2, run15 ×2 | exit 1/2 within seconds; run12_1655 preflight SSH unreachable | **OTHER** | ML/s166_run12_1655.log:21 "Preflight BLOCKED: SSH unreachable"; ML/s166_run14_1952.log:111 "File does not exist" | — |
| C-32 | LOG-ONLY | 04-23 16:02:39 | rrig6600 03:00.0 | same run (+10:36) | SMU, evict failure; ready 26→11 at 16:03:11; 1372 chunks hipErrorNoDevice | **OTHER-GPU** | NL:4208 "SMU: response:0xFFFFFFFF"; NL:4252 "Failed to evict" | — |
| C-35 | LOG-ONLY | 04-23 17:33:22 | rrig6600 03:00.0 | s166_run12_1722 (+10:37) | SMU, evict failure; ready 26→18 | **OTHER-GPU** | NL:4512; NL:4556 "Failed to evict" | — |
| C-37 | LOG-ONLY | 04-23 18:22:00 | rrig6600 03:00.0 | same run (+11:18) | SMU, evict failure | **OTHER-GPU** | NL:4704; NL:4748 | — |
| C-38 | LOG-ONLY | 04-23 19:19:53 | 2 workers, host n/r | s166_test_1909: 100K, 1 trial | 2 workers lost (ready 26→24 at 19:30:38); run PASSED | **UNKNOWN** | ML/s166_test_1909.log:21720 timeout; :43222 "24 ready" | Listener live NL:4767 (19:03) → NL:4768 (19:33:48) with no amdgpu line in the run window. Host and cause n/r. |
| C-39 | LOG-ONLY | 04-23 19:33:48 | rrig6600 03:00.0 | 1:46 after C-38 ended | SMU, evict failure (post-run) | **OTHER-GPU** | NL:4768 "TransferTableSmu2Dram"; NL:4786 | — |
| C-40 | LOG-ONLY | 04-23 20:04–21:16 | 26 | s166_run15_2004: 100K, 2 trials | log stops mid-chunk at 21:16:30 (3,146 survivors/chunk); no exit code | **UNKNOWN** | ML/s166_run15_2004.log last line 21:16:30 | Listener restarted (NL:4801) at an unrecorded time. No amdgpu line until 04-24 17:53. Stop cause n/r. |
| C-42 | LOG-ONLY | 04-24 18:15:24 | rrig6600 03:00.0 | same run (+22 min) | SMU, evict failure; ready 26→18 | **OTHER-GPU** | NL:4904; NL:4948 "Failed to evict" | — |
| C-43 | E24 s168a_diag; ISO 04-24 | 04-24 19:04–20:05 | rrig6600c per record | s168a_diag_test_1904: 26 workers, 200K, 2 trials, DIAG=1 only | record: "Trial 2 ❌ rrig6600c page fault at t+30s". Log: 26 ready throughout, Trial 2 dispatched 19:37:49, Step 1 exit -9 at 20:05:07 | **UNKNOWN** | S167:129 "rrig6600c page fault at t+30s"; ML/s168a_diag_test_1904.log:19765 "code -9" | No page-fault or GCVM line in NL between 4976 (18:40:50) and 4983 (21:02:58). The listener ran continuously and .162 sent READY at NL:4974, so the record's page fault is absent from netconsole. The -9 cause is n/r. |
| C-44 | LOG-ONLY (S168 code defect) | 04-24 20:45–20:54 | — | s168_jitter_test_2045: 200K | 0/26 workers online; coordinator UnboundLocalError | **OTHER** | ML/s168_jitter_test_2045.log:3701 "online timeout: 0/26"; :132 "UnboundLocalError: local variable 'time'" | — |
| C-48 | LOG-ONLY | 04-24 21:50:49 | rrig6600 03:00.0 | s169_test_2150: 100K, 2 trials, 18 ready at start | SMU at +1 s; rig rebooted (READY 21:53); .120 script-write failures | **OTHER-GPU** | NL:5578 "SMU: response:0xFFFFFFFF"; NL:5634; ML/s169_test_2150.log:21769 ".120:GPU0 script write failed" | — |
| C-50 | E30 | 04-25 09:08:33 | rrig6600 03:00.0 | same run (+15:05) | SMU → device lost; .120 script-write failures 09:11; later 2818 chunks cudaErrorNoDevice (Zeus CUDA) from 09:19:17 | **OTHER-GPU** | NL:5858 "SMU: response:0xFFFFFFFF"; NL:5903 "device lost from bus!"; ML/stability_cap_100000_t5_0853.log:151096 cudaErrorNoDevice | — |
| C-52 | E31 S170 Run 2; ISO 04-25 Run 2 | 04-25 17:34–18:28 | Zeus pipeline | stability_cap_100000_t2_dpm_peak_jitter_1734: 26 workers, 100K, 2 trials, peak-lock | 53 min; 292,671,006 survivors; SIGKILL -9 | **OTHER** | S170:176 "SIGKILL'd (code -9) … due to result-write blowup"; ML/…_1734.log:21697 | — (Second listener start at NL:6081, time unrecorded; S170:178 "Netconsole clean" cannot be proven live.) |
| C-53 | E31/ISO S170 Runs 3–4 | 04-25 20:13 / 20:39 | — | s170_victory_config_…_2013 (config-file mode → legacy SSH); _2039 (2M chunks) | config-mode defects; no GPU fault | **OTHER** | S170:196 "silently downgraded to legacy SSH path"; S170:200 "chunks were 2,000,000"; ML/…_2013.log:99 "Oops, unhandled type 3" | — |
| C-55 | LOG-ONLY | 04-25 21:57 | — | s170_official_100k_t5_dpm_jitter_2157 | escalated at once: optimal_window_config.json missing | **OTHER** | ML/s170_official_100k_t5_dpm_jitter_2157.log tail "File does not exist" | — |
| C-57 | LOG-ONLY (E28 instance) | 04-25 22:26:04 | rrig6600 03:00.0 | same run (+22:48) | SMU, evict failure; .120 script-write failures from 22:28:28; ready →18 | **OTHER-GPU** | NL:6283; NL:6327; ML/s170_clean_100k_t5_2203.log:129528 ".120:GPU0 script write failed" | — |
| C-59 | LOG-ONLY | 04-26 07:44–07:46 | rrig6600 (7/8 at launch) | s170_jitter15_gap020_100k_t5_0744: 26 workers, 100K, 5 trials, jitter 15/gap 0.20 | card missing at launch; 2172 chunks hipErrorNoDevice | **OTHER-GPU** | ML/…_0744.log:34 "GPU_COUNT_MISMATCH: 7/8"; :21724 | Run-log evidence only. |
| C-61 | LOG-ONLY | 04-26 07:58:55 | rrig6600 03:00.0 | same run (+14:34) | SMU, evict failure, sdma timeout | **OTHER-GPU** | NL:6676; NL:6731 "ring sdma0 timeout" | — |
| C-62 | E34 Test 1 (2-rig), mapped by config | 04-26 09:13–09:44 | rrig6600 (7/8), rrig6600c | s170_2rig_no6600b_k512_100k_t5_0913: 18 workers, min 16, pool 8, 100K, 5 trials, K512 | card missing at launch; 7584 chunks hipErrorNoDevice from 09:15:54; log stops mid-chunk | **OTHER-GPU** | ML/…_0913.log:34 "GPU_COUNT_MISMATCH: 7/8"; :21682 | No kernel line. Listener restarted (NL:6752-6755) at an unrecorded time. The S170:58-64 Test 1 run is not named; it could be this run or C-63. |
| C-64 | E34/E33 | 04-26 15:53:03 | rrig6600 03:00.0 | same run (+13:28) | SMU; probe fails closed; ready 18→2 | **OTHER-GPU** | NL:6866; ML/…_1539.log:43338 "no live ROCm GPUs available; skipping node" | — |
| C-66 | E37 Test 3; ISO 04-25 Test 3 | 04-26 17:39:46 | rrig6600 03:00.0 | s170_pool8_cooldown50_100k_t5_1729: 18 workers, min 16, pool 8, 100K, 5 trials, jitter 15/gap 0.20, COOLDOWN_MS 50 (no code, E37) | SMU 0xFFFFFFFF at +10:45 ("~10 min"); no GCVM | **OTHER-GPU** | NL:6878 "Failed to retrieve enabled ppfeatures"; S170:100 "Still crashes at ~10 minutes" | — (S170:160 dispatch/concurrency model.) |
| C-67 | E37 | 04-26 17:40:31 | rrig6600c 06:00.0 | same run (+11:30) | SMU; probe skips both rigs; ready 18→2 at 17:41:17 | **OTHER-GPU** | NL:6886 "SMU: response:0xFFFFFFFF"; ML/…_1729.log:21754 | — |
| C-68 | E38 Test 4 attempts 1–2 | 04-26 18:07 / 18:12 | — | cooldown500_1807 / _1812 | preflight SSH unreachable (rigs down after C-66/67) | **OTHER** | ML/s170_pool8_cooldown500_100k_t5_1807.log:21 "Preflight BLOCKED" | — |
| C-70 | E38 attempt 4 | 04-26 19:41:14 | rrig6600 03:00.0 | cooldown500_1924 (+16:27) | SMU; probe skips node; ready 18→8 | **OTHER-GPU** | NL:7026; ML/…_1924.log:64705 "skipping node" | — |
| C-73 | E41 | 04-28 16:51:31 | rrig6600 03:00.0 | same run (+19:07) | SMU, evict failure, device lost 16:52:00 | **OTHER-GPU** | NL:7248; NL:7269 "device lost from bus!" | — |
| C-74 | LOG-ONLY | 04-28 17:20:57 | rrig6600 03:00.0 | 1:39 after throttle run ended | SMU (post-run) | **OTHER-GPU** | NL:7281 "SMU: response:0xFFFFFFFF" | — |
| C-75 | E41 pool7_t5 (register: result n/r) | 04-28 17:54–18:58 | 16 workers | s171_pool7_100k_t5_1754: min 14, pool 7, 100K, 5 trials, jitter 15/gap 0.20 | ~24k survivors/chunk at the end; Step 1 exit -9 at 18:58:34; no rig fault | **UNKNOWN** | ML/s171_pool7_100k_t5_1754.log:27769 "code -9"; last chunk "24,120 survivors" | Cause of -9 n/r (the survivor-explosion pattern of C-52, not recorded). Listener live NL:7289 (17:27) → NL:7290 (20:26) with no line. |
| C-77 | LOG-ONLY | 04-29 08:24 | — | s172_pool8_100k_clean_t5_0824 | preflight SSH unreachable | **OTHER** | ML/…_0824.log:21 "Preflight BLOCKED" | — |
| C-78 | LOG-ONLY | 04-29 08:52–09:01 | 18 workers | s172_pool8_100k_clean_t5_0852: pool 8, 100K | log stops mid-chunk at 09:01:53 (~32.6k survivors/chunk); no exit code | **UNKNOWN** | ML/…_0852.log last line 09:01:53 | No kernel line. Listener restarted (NL:7589-7590) at an unrecorded time. Stop cause n/r. |
| C-80 | LOG-ONLY | 05-01 16:33:10–16:47:14 | rrig6600 03:00.0 | no run log active (after 16:05:18) | SMU; device lost 16:46:43; GPU Recovery Failed -19 | **OTHER-GPU** | NL:8040; NL:8069 "device lost from bus!" | — |
| C-84 | E43 pool8/25k eviction | 05-01 23:20 | rrig6600 (dump "workers_8→6") | same run | 2 workers evicted, recovered | **UNKNOWN** | ROCm:87 "one observed at 23:20"; ROCm:194 | Listener live 22:12 → 00:03 with no amdgpu line. Run log has no timeout or ready-count change. The dump was deleted (S174a:36). Cause n/r. |
| C-90 | E45 manifest #2 | 05-03 10:34:25 | rrig6600 03:00.0 (gpu0) | same (+10 min) | SMU 0xFFFFFFFF | **OTHER-GPU** | NL:8617 "GetEnabledSmuFeaturesHigh" | Run log deleted. |
| C-91 | E46 FT073 Run 1; ISO 05-04 | 05-04 18:17:30–18:30:25 | rrig6600b 10:00.0 | FT073 repro: pool 8, 50K, 1 forced trial W21_O66_FT0.73, 425M, jitter 3/gap 0.02 | SMU 0xFFFFFFFF bursts over 13 min; rig rebooted (READY 19:18:13) = the record's "operator-rebooted mid-trial" | **OTHER-GPU** | NL:8637 "SMU: response:0xFFFFFFFF … TransferTableSmu2Dram"; NL:8721; S174a:23 | Run start time n/r; matched by date and host. **Disagrees**: S174a:23 records only the operator reboot. The kernel shows an SMU failure on rrig6600b about 47 min before it. |
| C-93 | E46 Run 2 cascade | 05-05 18:19:08 | rrig6600 03:00.0 | same | SMU cascade | **OTHER-GPU** | NL:8880 "SMU: response:0xFFFFFFFF" | — |
| C-97 | LOG-ONLY | 05-09 19:58 / 20:39 / 20:43 | — | 3 S174 gate-validation launches | child exit 1 in <1 s: wrong interpreter | **OTHER** | ML/S174_GATE_VALIDATION_20260509_195822.log tail "ERROR: CuPy not available"; summary:60 INVALID_MISSING_EVIDENCE | — |
| C-99 | E48 negative test | 05-09 22:02:58 | — | min_workers 27 | READY_GATE_FAILED, as designed | **OTHER** | ML/S174_NEGATIVE_TEST_20260509_220258_summary.txt:60 | — |
| C-101 | E49 D1; ISO 05-09 | 05-09 22:35:35 | rrig6600 03:00.0 | same | post-completion SMU breakdown 1 s after child exit; rrig6600 unreachable; .154 workers 8→0, .162 8→1 | **OTHER-GPU** | NL:8989 "SMU: response:0xFFFFFFFF … GetEnabledSmuFeaturesHigh"; D1:26-29 | No kernel line on .154/.162 for their worker losses (listener live NL:8988 → NL:9009). Launcher VALID_CLEAN (summary:60) is a recorded FAULT_KEYWORDS gap (D1:30). |
| C-102 | E49 reboot crashes | 05-09 ~23:09–23:30 | rrig6600, rrig6600c | idle after reboot | "rebooted, crashed" | **UNKNOWN** | D1:40-41 "rrig6600: rebooted, crashed" | NL:9009-9010 READY 23:09. NL:9011 shows .120 alive at 23:49:54 (uptime 2489 s) with no amdgpu line. No crash signature is recorded anywhere. |
| D-03 | E13; LIST 4 row 08-05 | 2026-08-05 | 3 CTs + Zeus, 25 daemons | 8 workers/CT, hand-started, const phase, `staging_deferred_max=64` | Run died in the coordinator staging deferred queue | **OTHER** | docs/TEAM_ALPHA_DEFERRED_QUEUE_NOTE.md:14,20 "staging deferred queue full — dispatch back-pressure" | No host loss recorded. Host kernel logs unavailable; netconsole not receiving (no datagram 06-15 20:39 → 08-22 19:25, MT/logs/netconsole_listener.log:9307→9453) |
| D-04 | E14; LIST 4 att 1 | 2026-08-09 12:37:09–12:47:17 | rrig6600 CT .122 gpu0–6 + Zeus (cohort on .122 only) | 8 workers, k=20, anchor 44, τ 0.45, phases 1–2, 2M/1M caps, TCP, `689f3cd9` | Gate-12 att 1: lease expiry | **OTHER** | docs/CLAUDE_CODE_REPORT_GATE12_FAILURE_FORENSICS.md:47,54 "compute-lease expiry in a constant workflow phase" · `rrig6600:gpu4` | Rig kernel logs UNAVAILABLE (:717-718); netconsole not receiving |
| D-05 | E15; isolation row 08-10→17; LIST 4 att 2 | 2026-08-10 09:23–10:44:16 | 3 CTs (24× RX 6600 XT) + Zeus 3080 Ti = 25 | 8/CT, k=13, anchor 69, τ 0.47, phases 1–3, 44,331 p3 survivors, TCP, `abc63f71` | Gate-12 att 2: admission timeout at stage 3→4 | **OTHER** | docs/TB_GATE12_ATTEMPT2_FORENSIC_REPORT.md:6-7 "TRIAL TERMINAL class=worker_admission_timeout" | — |
| D-06 | E15 (antecedent of D-05) | 2026-08-10, before 10:44 | 2 of 25 worker sessions (identities not named in the text read) | as D-05 | Two worker TCP sessions lost across stage boundary | **UNKNOWN** | docs/TFM_PROJECT_FACTS_SKILL.md:1744 "initiating cause of the two lost sessions is UNRESOLVED" | Netconsole empty; "cannot distinguish 'no event' from 'not active'" (docs/CLAUDE_CODE_REPORT_DEFECT_A_TRANSPORT_RECOVERY.md:270-274), so netconsole was not receiving. Rig logs unexamined. Disagreement: TB report :75-76 claims "zero `dmesg` GPU faults" but does not record where that dmesg read came from |
| D-07 | E17; LIST 4 att 3 | 2026-08-10 17:28:31–19:48:14 | 3 CTs + Zeus, 25 | 8/CT, k=22, anchor 0, τ 0.47, 4 phases, 774 p3, `d606edbe` | Gate-12 att 3: publication refused | **OTHER** | docs/TFM_PROJECT_FACTS_SKILL.md:1905 "PRE-LAUNCH CLEAN-TREE ADMISSION DEFECT" | — |
| D-09 | E17; LIST 4 att 4 | 2026-08-11 19:04:14–19:22:53 | 3 CTs + Zeus, 25 | 8/CT, k=15, anchor 71, τ 0.31, stage 2, `c8939b64` | Gate-12 att 4: stale lease origin | **OTHER** | docs/TFM_PROJECT_FACTS_SKILL.md:1986 "PRIMARY DEFECT — STALE LOOP-TIME COMPUTE LEASE ORIGIN" | — |
| D-10 | E18; LIST 4 att 5 | 2026-08-12 17:38:44–18:59:53 | 3 CTs + Zeus, 25 | 8/CT, k=12, anchor 49, τ 0.46, skip 10–55, 276,439 p3, `7e0d020b` | Gate-12 att 5: admission timeout at phase 4 (23/25) | **OTHER** | docs/TFM_PROJECT_FACTS_SKILL.md:2091-2092 "expected 25 eligible worker(s); 23 admitted after 180.0s" | — |
| D-11 | E18; isolation row 08-10→17 ("attempt 5 dropped 2 workers") | 2026-08-12 18:27:10.537/.539 | rrig6600c:gpu2, rrig6600b:gpu3 | as D-10; drops came 1.46 s after a 940.856 s coordinator drain stall | 2 workers dropped (WORKER_DISCONNECTED) | **UNKNOWN** | docs/TFM_PROJECT_FACTS_SKILL.md:2122-2123 "coordinator performed the final close … antecedent cause is UNRESOLVED"; :2127 | Rig logs stopped at 18:02:12 (:2124-2125); host kernel logs unavailable; netconsole not receiving. No GPU-fault evidence either way |
| D-12 | not in Part D (S182; Part E surface 4 cites D6 docs) | 2026-08-14 | 3 CTs | D6 dry run #1; worker `range_miner_worker.py` last deployed 08-02 (stale) | 24 of 25 workers died at argparse | **OTHER** | docs/SESSION_CHANGELOG_20260814_S182.md:21-23 "24 of 25 workers died at argparse" | — |
| D-13 | E19 (context); not tabulated in Part D | 2026-08-14 | 3 CTs + local worker | D6 dry run #2; launcher `&`-binding defect; 400 s wrapper timeout | Local worker killed by the wrapper timeout; showed up as a fleet finding | **OTHER** | docs/SESSION_CHANGELOG_20260815_S183.md:109-110 "400 s wrapper timeout signalled the process group and killed the local worker" | Rig logs destroyed by the 08-15 reboot (:101-102) |
| D-14 | E20; LIST 4 att 6 | 2026-08-15, terminal 13:32:15 | 3 CTs + Zeus, 25; expiring worker rrig6600b:gpu1 | 8/CT, k=8, anchor 13, τ 0.39, stage 2, `db0393b0` | Gate-12 att 6: compute_lease_expiry | **OTHER** | commit 087b7f4 msg :32-33 "nobody crashed, nobody lost transport"; docs/TFM_PROJECT_FACTS_SKILL.md:2324-2326 "BOUNDED-UNRESOLVED" | Antecedent unresolved for att 6 itself. H2 (coordinator acceptance backlog) was confirmed for the same wall in att 7 (D-15). Host GPU kernel logs UNAVAILABLE |
| D-15 | E20; LIST 4 att 7 | 2026-08-16 | 3 CTs + Zeus, 25 | k=29, anchor 49, τ 0.39, stage 2, `36bf30e3` | Gate-12 att 7: lease expiry with coordinator drain taking no frames | **OTHER** | docs/TFM_PROJECT_FACTS_SKILL.md:2348-2350 "H2 CONFIRMED … 45 frames arrived and the drain took NONE" | — |
| D-16 | E20; LIST 4 att 8 (MP-1) | 2026-08-16 | 3 CTs + Zeus, 25 | k=15, anchor 82, τ 0.60, stage 2, `5c010902` | Gate-12 att 8: coordinator (VM101) rate starvation in staging | **OTHER** | docs/TFM_PROJECT_FACTS_SKILL.md:2372,2378 "staging executors x4 ~3,640 s exclusive in _pump_deferred"; "RATE starvation" | — (this overload is on the VM101 coordinator, not on a rig) |
| D-18 | E23; isolation row 08-19 (MEM; not TFM) | 2026-08-19 | rrig6600 CT .122 + host pve-rig6600 .121 down; rrig6600b, rrig6600c survived | PIECE-MATCHER (not TFM); 1 process/GPU, 8 GPUs on a 4-core, 6 GB CT; chunk 2^22…2^29; first multi-minute 8-GPU load | Host + CT stopped answering | **UNKNOWN** | MEM rig-ct100-four-cores-eight-gpus.md:35-39 "stopped answering during the first multi-minute 8-GPU … load"; "Cause undetermined" | No host dmesg; .121 unreachable; netconsole not receiving; not in the repo record. MEM ties the loss to sustained load but gives no mechanism, so this is not OVERLOAD evidence. The CPU-starvation measurement (:13-18) is throughput only and is not tied to the loss |
| D-19 | E24; isolation row 08-22 run 1 | 2026-08-22 14:33–~14:47 | 3 CTs + Zeus, 25 | 8/CT, k=20, anchor 58, τ 0.64, gen-phase 0, 1 p3 / 0 p4, `eed23c7f`, code 48a8705 | Run 1 failed at Phase-5 commit | **OTHER** | docs/S172_BRIEF_I_PRODUCTION_SHAPE_CLASSIFICATION_REPORT.md:12,14 "FAILED at Phase-5 commit" · "BRIEF-I DEFECT" | — |
| D-21 | E24 run 2 (register: "why it stopped … not recorded") | 2026-08-22 17:39:17 (stamp)–17:52:26 | 3 CTs + Zeus, 25 | k=22, anchor 26, τ 0.69, phase 1 only, `35ba80aa`, code 69ca910 | Run 2: staging capacity timeout in phase 1 | **OTHER** | MT/logs/brief_i_evidence/S172_BRIEF_I_RUN2_CAPACITY_FAILURE_REPORT.md:13,15 "`staging_capacity_timeout` — the trial was STARVED BY CONSTRUCTION"; "fleet healthy throughout" | — (the register's "not recorded" is answered by this gitignored main-tree report, which is not in docs/) |
| D-22 | E25; isolation row 08-22 run 3; context entry | 2026-08-22, phase 4 dispatched 18:32:01; hosts silent 18:32:06/07 | rrig6600 (.121/.122) and rrig6600c (.163/.164) hosts hung; rrig6600b + Zeus OK | 8/CT, k=6, anchor 75, τ 0.35, M=3, gen-phase 0, 13,146,485 p3 (478.96 MiB inline), all 25 resume together at phase 4, `8cb116da`, code 69ca910 | Two hosts hard-hung (found powered on but unresponsive) | **UNKNOWN** | MT/logs/brief_i_evidence/RIG_CRASH_FORENSIC_20260822.md:9 "POWERED ON but UNRESPONSIVE"; :121 "no runtime amdgpu reset, ring timeout, GPU hang, MCE, thermal event or OOM" (host journal); docs/RIG_CRASH_FORENSIC_20260822.md:356-359 H2/H3/H5 OPEN, H4 ELIMINATED | No kernel line. The host journal lost the last 5–13 min (SyncIntervalSec default; MT forensic :75-94). Netconsole was armed only afterwards (first NC-TEST3 at 19:25:50, MT/logs/netconsole_listener.log:9453) at console_loglevel 3 (MT forensic :381-383). CT memory/CPU UNAVAILABLE. Overwhelm-type explanations are hypotheses only: H2 supply sag, H3 host resource exhaustion. Disagreement: MT forensic :34, :283 says rrig6600c claimed ZERO phase-4 stripes; docs forensic :341-347 corrects this to 8 shards at 18:32:05–07 |
| D-23 | E36; context entry; isolation row 11-15 att 1 | 2026-11-15 09:05:53–09:19:49 | 3 CTs + Zeus, 25 | as D-24 | Run 4 att 1: Step-1 credit refused (non-empty ledger -wal) | **OTHER** | docs/CLAUDE_CODE_INSTRUCTIONS_STEP1_CREDIT_WAL_20261115.md:61 "`miner_ledger.db-wal` is non-empty (4148872 bytes) … no credit" | — |
| D-26 | Part E surface 3 item 5 / "only on these surfaces" item 5 | 2026-11-15 13:47:28 | pve-rig6600b .155, card 0000:19:00.0 (the host that survived) | After the retry's terminal (13:31:32 summary, gate12 log :730); no run attributed | SMU TransferTableSmu2Dram failure | **OTHER-GPU** | MT/logs/netconsole_listener.log:10098-10099 "SMU: response:0xFFFFFFFF … TransferTableSmu2Dram" · "Failed to export SMU metrics table!" (also netconsole_all_rigs.log:822) | No GCVM line on .155. The record does not say what workload or action was running at 13:47 |

---

## The pre-PWC search that did not crash

Worktree `/home/michael/tfm_worktrees/gpu-fault-history`, HEAD `3579088`. This is a read-only pass. Anchors are `file:line` at HEAD, `<sha>:<path>:<line>`, or a commit. Rows reused from the Jan–Mar classification are cited as `partA A-xx` (`.../scratchpad/classify/partA.md`). No cause is concluded here.

### Null results first

- **GCVM records for the job-per-process era: 0.** That covers 2025-11-29 to 2026-03-07, with 44 Jan–Mar failure events (partA, "Null results first") plus the Nov–Dec 2025 items in (5) below. **No kernel-log capture existed in that era.** The first netconsole entry is 2026-04-01 (partA, null-results list). So "no GCVM" means no record exists. It does not mean the fault was observed to be absent.
- **The record does not support "never crashed" for the era as a whole.** From 2026-01-16 on, the same job-per-process transport froze rigs repeatedly (partA A-01, A-03, A-06, A-10, A-11, and others). Every one of those was a **Step 2 or Step 3 script workload**, not the Step 1 sieve. The Step 1 sieve on this transport has 2 recorded GPU-Hang events (A-18, A-26) and no recorded freeze.
- **For Nov–Dec 2025 alone, I found no crash, freeze, hang or reboot record.** What is recorded is software failures and timeouts (rows N-01 to N-05 below).
- **torch was in the sieve worker process before the first GCVM record (2026-04-02), and during the zero-GCVM era too.** It sat at module level from 2025-11-27 to 2025-12-24. It came back as a cleanup call from 2026-01-14. Details in (2).

### (1) How a job reached a GPU in that era

- **There was a new OS process per job, started over SSH.** The coordinator builds `env HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_ENABLE_SDMA=0 … CUDA_VISIBLE_DEVICES={gpu_id} HIP_VISIBLE_DEVICES={gpu_id} {python_env} -u sieve_filter.py --job-file … --gpu-id 0`.
  - Sources: `0101306:coordinator.py:196-230`, where the `residue_sieve` branch is `:223-225`. It is executed with paramiko `ssh.exec_command(cmd)` (`0101306:coordinator.py:798-805`).
  - The same command shape is still there just before S130 (`24a7d94:coordinator.py:362-376`, `:390`).
  - Step 2.5 and other script jobs went through `distributed_worker.py`, the same way (`0101306:coordinator.py:216-221`).
  - After 2026-01-03, script jobs went through `scripts_coordinator.py` instead (commit 90b6789 body).
- **One job per GPU at a time.** There is one `GPUWorker` per `gpu_id` in `range(node.gpu_count)` (`0101306:coordinator.py:271-272`). There is one thread per worker (`0101306:coordinator.py:1389-1392`). Each thread runs its jobs one after another from a shared queue (`:1233-1241`). The per-node cap applies only to script jobs: "Only applies to script-based jobs; seed jobs are unlimited" (`0101306:coordinator.py:839-848`).
- **Concurrent jobs per rig = `gpu_count`:**
  - **12 per rig, 2 rigs**, from 2025-11-29: `0101306:distributed_config.json:14,22`. Zeus `gpu_count` was 2 (`:6`). The same 12/12 is still set at `6490771` (2025-12-13) and `a490ea0` (2026-01-18), with `max_concurrent_script_jobs` 12 (`6490771:distributed_config.json:15,19,24,28`).
  - **8 per rig, 3 rigs**, from 2026-01-31 (commit 4d979e0; `4d979e0:distributed_config.json:15,26,37`). The reason for 12→8 is not recorded (`docs/REGISTER_GPU_FAULT_PRIOR_WORK.md:348`).
  - CLI defaults: `--max-concurrent 26`, `--max-per-node 12` (`0101306:coordinator.py:2307-2310`).
- **Before 2025-12-05, Step 1 did not reach the rigs.** `window_optimizer.py` built `MultiGPUCoordinator(resume_policy="restart")`, so it "only used localhost (2 GPUs)" (`b45c8cc:CURRENT_Status.txt:5`, `:157-161`). After the fix, "All 26 GPUs now receiving jobs (14 on rig-6600, 8 on rig-6600b, 2 on Zeus)" (`:161`).
  - Direct coordinator runs had used 26 GPUs earlier ("VERIFIED WORKING (26/26 GPUs each)", `0101306:ACHIEVEMENT_SUMMARY_20251020_193302.txt:17`, 2025-10-20).
- **Job size:**
  - Seed cap per job: AMD 19,000, NVIDIA 40,000 (`0101306:coordinator.py:2300-2303`). Chunk size was `max(19000, total_seeds // 100)` from ~2025-12-05 (`b45c8cc:CURRENT_Status.txt:426-428`; before that, 500-seed chunks timed out).
  - Recorded speed: "~13s per 50K-seed sieve" (`b45c8cc:CURRENT_Status.txt:501`).
  - Step 1 defaults: java_lcg, 10M seeds, 20 trials, `--test-both-modes` (`b45c8cc:window_optimizer.py:763`; `b45c8cc:agent_manifests/window_optimizer.json:22,28,30,31`).
  - The caps were raised to 2M AMD and 5M NVIDIA on 2026-03-08 (`docs/SESSION_CHANGELOG_20260308_S129B.md:40-41`; commit d168f83). The old RX 6600 figure was `seeds_per_second = 5,000` against a measured ~787,950 (`:22`).
- **Stagger:**
  - Sieve and seed jobs had **no stagger**. At 2025-11-29 the word "stagger" does not appear in `coordinator.py`.
  - Commit 90b6789 (2026-01-03) widened the per-job stagger from localhost to all nodes, `gpu_id * 2.0` s (`90b6789:coordinator.py:1488-1491`). That stagger is gated on `analysis_type == 'script'`, and the gate is still there at `24a7d94:coordinator.py:1508-1509`. It is applied before **every** script job, not only at startup.
  - Thread launch has no delay (`0101306:coordinator.py:1389-1392`).
  - The first sieve-relevant spawn stagger is 7288e5e (2026-03-08, persistent workers, `gpu_id*2.0 s`) (`docs/REGISTER_GPU_FAULT_PRIOR_WORK.md`, row E23).
- **Workloads on the rigs in this era:**
  - Step 1 sieve (from 2025-12-05).
  - Step 2.5 scorer meta-optimizer (`scorer_trial_worker.py`).
  - Step 3 full scoring (`full_scoring_worker.py`; "36/36 jobs successful … 395,211 survivors", `0e82155:CURRENT_Status.txt:251`).
  - Step 5 distributed (commit c9f7f07, 2025-11-30).
  - All the Jan–Feb rig freezes are Step 2 or Step 3 runs (see (5)).

**Register author's verification.** The following anchors were re-read and confirmed: 41165dd (date and body), b425171 `sieve_filter.py:59-75`, 20b0e7c (date and subject), `0101306:prng_registry.py:70-78`, and `miner/range_miner_worker.py:1051-1072`.

### (2) GPU libraries in the worker process

| Path | torch? | When | Anchor |
|---|---|---|---|
| Sieve (`sieve_filter.py` → `prng_registry`) | **Yes, at module level, before CuPy.** `sieve_filter` imports `prng_registry` (`:46`) before `import cupy` (`:52`). `prng_registry` runs `try: import torch` twice at load | Added in prng_registry **v2.4, 2025-11-27** (`0101306:prng_registry.py:9`, `:70-90`). Absent from `prng_registry.py.working_20251022_210323` (0 hits) | `0101306:sieve_filter.py:46,52`; `0101306:prng_registry.py:70-90` |
| Same | **Removed** (lazy `_get_torch()`) | **41165dd, 2025-12-24 18:57 −0800**. Body: "Importing torch at module level … corrupted the HIP/ROCm environment, causing CuPy kernel compilation to fail"; "Fixes remote sieve failures" | commit 41165dd; HEAD `prng_registry.py:71-89` |
| Sieve cleanup `_best_effort_gpu_cleanup` in `sieve_filter.py` | **Yes:** `import torch` + `torch.cuda.synchronize()` + `empty_cache()`, then CuPy `free_all_blocks` | **b425171, 2026-01-14 20:48 −0800**. Called once at end of job (`b425171:sieve_filter.py:59-75`, call `:679`). Called **between chunks** from **20b0e7c, 2026-01-26** (`20b0e7c:sieve_filter.py:232,387,693`) | as cited |
| Same | Gated behind `S163_MEM_DEBUG=1` | **0a316b8, 2026-04-08** | HEAD `sieve_filter.py:136-145` |
| `distributed_worker.py` (script/standard jobs) | **Yes.** `enforce_gpu_backend` tries torch first: `torch.cuda.set_device`, a tensor, `torch.cuda.synchronize` | Already in the initial commit | `0101306:distributed_worker.py:128-160` |
| Step 2/3 workers | **Yes**: `scorer_trial_worker.py` `import torch` (`0101306:…:214,381,401`); `survivor_scorer.py:54`; `anti_overfit_trial_worker.py:40,179`. atexit torch cleanup added f614736 (2026-01-17) | from 2025-11-29 | as cited |
| `sieve_gpu_worker.py` (PWC SSH worker) | **Yes**, the same torch cleanup | First tracked in **f300396, 2026-03-09**. Still at HEAD `sieve_gpu_worker.py:78-95` (torch `:84-87`) | as cited |
| ZMQ worker (the path running at the first GCVM record) | **Yes.** `zmq_sqlite_worker.py` imports `sieve_filter.execute_sieve_job` (`561428b:zmq_sqlite_worker.py:215`), whose cleanup is still unconditional torch | `561428b` (2026-04-01): `561428b:sieve_filter.py:59-79`, calls `:232,387,693` | as cited |
| RANGE-MINER worker (HEAD) | **Yes.** `_best_effort_gpu_cleanup` does `import torch` + `synchronize` + `empty_cache`, then CuPy `free_all_blocks`, after **every** sub-stripe | dbe3d0e (2026-07-18) | `miner/range_miner_worker.py:1051-1072`, call `:1331`. Import chain: `prng_registry` (lazy torch only, `:634`), `sieve_gpu_worker._get_kernel` (`:1150`; module-level code has no torch). `adaptive_thresholds.py` and `hybrid_strategy.py` have 0 torch hits at HEAD and at 0101306 |

- **Was torch installed and working on the rigs?**
  - In Dec 2025 the rig venv had "PyTorch: 2.7.0+rocm6.3 … WORKS - Full GPU acceleration" (`1ddeb75:PROPOSAL_Unified_Agent_Context_Framework_v3_2_5.md:2487`). The system python had 1.12.1+rocm5.0 (`:2481`).
  - Whether torch is installed in today's CT100 `~/rocm_env` is **UNKNOWN**. The Phase 6.0 rig record lists cupy, not torch (`docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md:115-128`). If it is not installed, the miner's `import torch` falls through to `except` (`miner/range_miner_worker.py:1060-1066`).
- **Timeline against the first GCVM record.** The first document is 2026-04-02 (`docs/TB_SUBMISSION_S159G_RIG6600_CRASHES.md:3,22`). The first raw line is 2026-04-01 20:06:49 (main tree `logs/rrig6600c_netconsole.log:9`; see Fault classification). Torch entered the sieve worker process **before** both:
  - at module level from 2025-11-27 to 2025-12-24;
  - as per-job cleanup from 2026-01-14, and per-chunk from 2026-01-26;
  - and it was present in the ZMQ worker path on 04-01 and 04-02.
- **Torch was also present through all of Jan–Mar, which has 0 GCVM records.** Torch presence alone does not separate the eras. The difference lies in *how often* torch sync and empty_cache run (per job → per chunk → per sub-stripe). That frequency has not been counted for any era (**UNKNOWN**).

### (3) Cleanup between jobs

- **Era (2025-11-29 to 2026-01-13): process exit only.**
  - `0101306:sieve_filter.py` contains no `free_all_blocks`, `empty_cache`, `gc.collect` or `del` (grep returns nothing).
  - Each job's python process ends after the SSH command (`0101306:coordinator.py:796-805`).
  - The coordinator side does only `gc.collect()` ("no CUDA calls", `:816-818`).
- **2026-01-14 to 2026-03-07, still a process per job:**
  - end-of-job `_best_effort_gpu_cleanup` (torch sync + empty_cache, CuPy default and pinned `free_all_blocks`) (b425171);
  - plus the same cleanup between chunks inside one job (20b0e7c; partA A-19 "20/20 benchmark trials with 0 GPU hangs").
  - Process exit still follows.
- **PWC workers (long-lived):**
  - `sieve_gpu_worker.py:78-95`: gc, then torch sync/empty_cache, then CuPy `free_all_blocks` after each job (`:348`).
  - CuPy pool `set_limit(PRNG_CUPY_POOL_LIMIT_MB=256)` is set once at worker start (`sieve_gpu_worker.py:416-425`).
  - On the TCP-PWC path, `pwc_worker_service.py` calls `sieve_filter._best_effort_gpu_cleanup` after each result (`persistent/pwc_worker_service.py:462-463`). At HEAD that is gc only: torch is gated (`sieve_filter.py:136-145`) and `free_all_blocks` was removed (`:147-150`). `set_limit` comes from `sieve_filter.py:59-67`.
- **RANGE-MINER worker (HEAD):** a long-lived daemon.
  - After every sub-stripe, success or exception, it runs `del` of the GPU arrays and then `_best_effort_gpu_cleanup()`: gc, torch sync/empty_cache, CuPy default and pinned `free_all_blocks` (`miner/range_miner_worker.py:1051-1072`, `:1317-1331`).
  - **No `set_limit`** (`git grep set_limit HEAD -- miner/` returns no hit). `sieve_gpu_worker.run_worker` is not on the miner path.

### (4) Versions then vs now

| Item | Era record | Now | Anchors |
|---|---|---|---|
| ROCm | Conflicting. "ROCm 5.x with CuPy-ROCm" (`0101306:instructions_10-16.txt:1030,1038`, a doc dated 10-16); "ROCm 5.7" (`53a3829:docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md:66`); "ROCm 5.7+" corrected to "ROCm 6.4.3" by the 2026-01-31 audit (`docs/DOCUMENTATION_AUDIT_20260131.md:68,85`); "ROCm 6.4.3" (`5ab5411:docs/SELFPLAY_ARCHITECTURE_PROPOSAL_v1_0.md:28-29`, 2026-01-28). **The actual Nov–Dec 2025 host ROCm is UNKNOWN** | `6.4.3-128` (`rocm-core 6.4.3.60403-128~22.04`) on CT100 rrig6600 | `docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md:124` |
| CuPy (rig) | **UNKNOWN.** Only an install hint, `pip install cupy-rocm-5-0` (`0101306:instructions_10-16.txt:740`) | 13.5.1 / HIP `60443484` on CT100 rrig6600 | `docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md:125`. Also cupy 13.5.1 on all three CTs, with python 3.10.4 / 3.10.4 / 3.10.12 (`/home/michael/dashboard_work/MP1_FLEET_DEPLOY_RECORD.md:25-28`; the register author verified this, the extractor had looked under the main tree). Whether torch is installed in the CT `~/rocm_env` is not in that record (**UNKNOWN**). VM 101 cupy 13.5.1: `docs/FALLBACK_PARITY_PASS1_20260815_pipfreeze.txt:45` |
| torch (rig venv) | 2.7.0+rocm6.3 (`1ddeb75:PROPOSAL_…_v3_2_5.md:2487`; `5ab5411:docs/SELFPLAY_ARCHITECTURE_PROPOSAL_v1_0.md:39`) | **UNKNOWN** on CT100 | — |
| Kernel | 6.8.0-90 ("gfxoff=0 kernel param ignored on 6.8.0-90", commit f614736 body, 2026-01-17) | `6.8.12-9-pve` (Proxmox host). The Phase 6.0 *proposal* says "kernel 6.8.0-107" (`docs/TEAM_ALPHA_PROPOSAL_PHASE_6_0_ROCM_SMOKE.md:50`), which conflicts | `docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md:122,260` |
| amdgpu driver | In-kernel (stock Ubuntu) until S162. "Stock Ubuntu kernel amdgpu driver cannot handle…" (`docs/SESSION_CHANGELOG_S162_FINAL.md:12`); DKMS installed 2026-04-04/05 (`:14`, `:49`) | amdgpu module `6.12.12` in the PVE host kernel | `docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md:123`; `docs/SESSION_CHANGELOG_S162_VICTORY.md:140-142` |
| HSA_OVERRIDE_GFX_VERSION | `10.3.0` and `HSA_ENABLE_SDMA=0` on every rig job (`0101306:coordinator.py:200-201`; also `0101306:remote_env.txt:3` in the login env) | Miner launcher sets **no HSA_*** (`scripts/launch_fleet_manual.sh:316-322`); Phase 6.0 records "env overrides **none set**" (`docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md:127`) | as cited |

### (5) Fault record for the era

**Summary: no kernel capture existed in this era. The owner testifies there were no crashes of the Step 1 sieve (below).** This is testimony, not a document finding. The document record for the era is in the tables that follow.

> **OWNER TESTIMONY, 2026-11-15 16:24 PST** (relayed by Alpha; not a document finding):
> *"There is no record of it because it never was an issue. I only decided to move to a faster method when I realized how slow Step 1 was."*
>
> Register notes beside it:
> - The rigs are headless. A host freeze needs a manual power cycle by the owner, so a freeze in this era would have been noticed by him even with no kernel capture.
> - The reason the owner gives for leaving this engine is **speed, not stability**.

Classes: GCVM / OVERLOAD / OTHER-GPU / OTHER / UNKNOWN / CLEAN. For 2026-01-01 to 2026-03-07, rows are reused from partA: A-01 to A-26 are this transport. A-27 onward are persistent or PWC and outside the era.

**Nov–Dec 2025 and early Jan (new rows):**

| ID | Date | Workload / rigs | Event | Class | Evidence |
|---|---|---|---|---|---|
| N-00 | undated, in the 2025-11-29 tree | Step 2.5 scorer on .120 | troubleshooting entry: "ROCm initialization takes ~120 seconds on first run … If persistent, reboot the remote node" | UNKNOWN | `0101306:complete_workflow_guide_v2_PULL_UPDATED.md:2495-2508`. This is guidance, not a dated event. No log |
| N-01 | ≤2025-12-05 | Step 1, 500-seed chunks | jobs timed out: "SSH setup overhead (~16-22s) exceeded actual work time" | OTHER | `b45c8cc:CURRENT_Status.txt:426-427` |
| N-02 | ≤2025-12-13 | rig jobs (Step 3) | "2-4 failures per run that required retries", from stdout JSON parsing. Same session: MaxSessions 10→50, and an SSH pool that allowed 12 concurrent but pooled 6 | OTHER (the session-limit fix is a weak OVERLOAD signal, but no symptom is tied to it) | `0e82155:CURRENT_Status.txt:37-39`, `:48-50` |
| N-03 | ≤2025-12-18 (S13/S14) | Step 3 on rigs | ~75% job success: "Numpy is not available" (system torch 1.12.1), `hipErrorInvalidDevice`, "No valid JSON … under 26-way concurrency" | OTHER | `1ddeb75:PROPOSAL_Unified_Agent_Context_Framework_v3_2_5.md:2461-2469` |
| N-04 | 2025-11-27 → 12-24 | Step 1 sieve on rigs | "remote sieve failures": CuPy compile fails ("stddef.h not found") after torch import | OTHER | commit 41165dd body. **Conflicts** with "26-GPU Cluster ✅ WORKING … ~13s per 50K-seed sieve" (`b45c8cc:CURRENT_Status.txt:501`). How much of Dec the rig sieve actually worked is UNKNOWN |
| N-05 | 2026-01-03 | Step 2.5 on 26 GPUs | "failing … JSON survivor loading (258MB) thrashing i3 CPUs … 10+ concurrent jobs"; "SSH thundering herd" | OVERLOAD (record's attribution, no crash or kernel line). partA excluded it as E1 | commit 90b6789 body; `53a3829:docs/SESSION_CHANGELOG_20260103.md:13,26,49` |
| N-C1 | 2025-12-13 | Step 3, 36 jobs | 36/36, zero failures | CLEAN | `0e82155:CURRENT_Status.txt:251-255` |

**Jan 1 to Mar 7, 2026, same transport (partA rows reused):**

- GCVM: **0**.
- OVERLOAD:
  - A-03, A-06, A-10, A-11, A-12, A-16, A-23 are all Step 2 or Step 3 script runs. The freezes are A-01 (UNKNOWN), A-03, A-06, A-10 and A-11.
  - A-21 is a soft lockup at boot enumeration.
- OTHER-GPU: A-05 (SMU, monitoring polling), A-15 (SMU), and **A-18 and A-26, which are the only Step 1 sieve events: HSA "GPU Hang", not kernel lines.**
- OTHER: A-04, A-13, A-22, A-24.
- UNKNOWN: A-01, A-20.
- CLEAN: A-02, A-07, A-08, A-09, A-14, A-17, A-19 (Step 1, 20/20) and A-25.

So the record splits by workload:

- Step 1 sieve in this era: 0 rig freezes recorded and 2 OTHER-GPU hang events.
- Step 2/3 script jobs: every recorded freeze.

What is missing for every row: no dmesg, kern.log or netconsole for any Nov–Dec item. The OOM kernel lines exist only for A-12 (partA).

### (6) Differences: that era's sieve worker vs today's miner worker

1. **Process lifetime.**
   - Era: one python process per job (19K seeds), exiting after the job (`0101306:coordinator.py:223-225,798-805`).
   - Now: one long-lived daemon per GPU running many sub-stripes (`miner/range_miner_worker.py`; launched once, `scripts/launch_fleet_manual.sh:318-322`).
2. **Jobs per GPU:** 1 at a time in both, by thread (`0101306:coordinator.py:271,1388-1392`) and by daemon respectively.
3. **Concurrency per rig.**
   - Era: 12, from 2025-11-29 (`0101306:distributed_config.json:14,22`); 8 from 2026-01-31 (`4d979e0`).
   - Now: 8 (`distributed_config.json:15,26,37`).
4. **GPU count per rig, and rigs.**
   - Era: 12 GPUs × 2 rigs, then 8 × 3 rigs from 2026-01-31.
   - Now: 8 × 3.
   - Step 1 reached the rigs only from 2025-12-05 (`b45c8cc:CURRENT_Status.txt:5,161`).
5. **Device selection.**
   - Era: `CUDA_VISIBLE_DEVICES=N HIP_VISIBLE_DEVICES=N`, then `cp.cuda.Device(0)` (`0101306:coordinator.py:207-208`; `0101306:sieve_filter.py:86`).
   - Now: `ROCR_VISIBLE_DEVICES=N`, `--device-index 0` (`scripts/launch_fleet_manual.sh:316,321`), `cp.cuda.Device(self.device_index)` (`miner/range_miner_worker.py:1199`).
6. **HSA env.**
   - Era: `HSA_OVERRIDE_GFX_VERSION=10.3.0`, `HSA_ENABLE_SDMA=0` on every job (`0101306:coordinator.py:200-201`).
   - Now: none from the launcher (`scripts/launch_fleet_manual.sh:316-322`; "none set", `docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md:127`). The `sieve_gpu_worker` prelude is gated on hostnames `rig-6600*` (`sieve_gpu_worker.py:27-30`), while CT100 hostnames are `rrig6600*` (CLAUDE.md §3). The miner also imports it after `import cupy` (`miner/range_miner_worker.py:1149-1150`).
7. **CuPy kernel cache.**
   - Era: one shared `CUPY_CACHE_DIR=${HOME}/.cache/cupy` for all 12 processes (`0101306:coordinator.py:206`).
   - Now: per worker, `/tmp/cupy_cache_gpu$N` (`scripts/launch_fleet_manual.sh:317`).
8. **Libraries.**
   - Era: CuPy + torch. torch was a module-level import from 11-27 to 12-24, then absent until it returned as cleanup on 01-14, per job then per chunk.
   - Now: CuPy + torch sync/empty_cache after every sub-stripe (see (2)).
9. **Cleanup.**
   - Era: process exit only until 01-14. Then torch+CuPy `free_all_blocks` at job end, and per chunk from 01-26, followed by exit.
   - Now: `del` + gc + torch + CuPy `free_all_blocks` per sub-stripe, no process exit, no pool `set_limit` (see (3)).
10. **Chunk size / seed cap.**
    - Era: 19K AMD / 40K NVIDIA per job (`0101306:coordinator.py:2300-2303`).
    - Now: defaults of 2M AMD and 1M AMD-hybrid (`miner/range_miner_worker.py:2814-2817`). The Run-4 pinned geometry is not read here (**UNKNOWN**).
11. **Kernels / families.**
    - Era: registry kernels through `sieve_filter`, any family (Oct-2025 summary lists 14 variants, `0101306:ACHIEVEMENT_SUMMARY_20251020_193302.txt:17-25`). The Step 1 default was java_lcg + `--test-both-modes` (`b45c8cc:window_optimizer.py:763`).
    - Now: 7 branches / 6 families, and 5 families raise `NotImplementedError` (`miner/range_miner_worker.py:109,476-480`; CLAUDE.md §7).
12. **Transport.**
    - Era: paramiko SSH `exec_command` per job, result parsed from stdout and later from files (`0e82155:CURRENT_Status.txt:48-50`).
    - Now: a TCP length-prefixed protocol to the miner coordinator (`miner/range_miner_protocol.py`; CLAUDE.md §6).
13. **Stagger.**
    - Era: none for sieve jobs. The 2 s/GPU stagger applied only to script jobs (`24a7d94:coordinator.py:1508-1509`).
    - Now: 3 s between daemon launches, once (`scripts/launch_fleet_manual.sh:150`).
14. **Host OS.**
    - Era: bare Ubuntu, kernel 6.8.0-90 (f614736 body). One record says "Requires HiveOS reboot" (`docs/PROPOSAL_Job_Batching_Pipeline_Stability.md:23`), so the exact OS is not settled.
    - Now: Proxmox host plus an unprivileged CT100 (CLAUDE.md §3; `docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md:260`).
15. **Driver.**
    - Era: in-kernel amdgpu (`docs/SESSION_CHANGELOG_S162_FINAL.md:12`).
    - Now: amdgpu 6.12.12 (DKMS) in kernel 6.8.12-9-pve (`docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md:122-123`).
16. **ROCm / CuPy userspace.**
    - Era: UNKNOWN (5.x, 5.7 and 6.4.3 are all recorded).
    - Now: 6.4.3-128 / cupy 13.5.1 (`docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md:124-125`).

### Searches run

- `git log --reverse --format='%h %ad %s' --date=short | head -120`, then `git show` of `0101306:` coordinator.py, distributed_config.json, sieve_filter.py, distributed_worker.py, prng_registry.py, reverse_sieve_filter.py, instruction_rocm.txt, remote_env.txt, instructions_10-16.txt, ACHIEVEMENT_SUMMARY_20251020_193302.txt and complete_workflow_guide_v2_PULL_UPDATED.md.
- `git log -S'import torch'` and `git log -G'torch'` on sieve_filter.py, distributed_worker.py, sieve_gpu_worker.py, prng_registry.py, persistent_worker_coordinator.py, gpu_cleanup.py, miner/range_miner_worker.py and coordinator.py. Then `git show` of 41165dd, b425171, 20b0e7c, 0a316b8, f300396, f614736, a47ba88, 90b6789, deb8788, 4d979e0, 94342a4 and 561428b (sieve_filter.py, zmq_sqlite_worker.py).
- `git grep -n torch HEAD -- miner/ sieve_gpu_worker.py sieve_filter.py prng_registry.py`, plus adaptive_thresholds.py, hybrid_strategy.py and miner/range_miner_protocol.py at HEAD and 0101306.
- `git log --until=2026-01-16` message bodies, grepped for crash, hang, freeze, oom, reboot, thrash, stagger, rocm and hip. `git log --diff-filter=A --until=2026-01-02 --name-only` for era docs. `git show` of CURRENT_Status.txt at b45c8cc, c77c08b and 0e82155, `PROPOSAL_…_v3_2_5.md` at 1ddeb75, `docs/SESSION_CHANGELOG_20260103.md` at 53a3829, and `CHAPTER_9` at 53a3829.
- Version-string `git grep` (ROCm 5/6, cupy-rocm, +rocm, HiveOS, 6.8.0-*, torch 2.x) at 0101306, 4ffd5ac, a490ea0 and 4d979e0. `/bin/grep` over HEAD docs for cupy version, 6.12.12, 6.8.12-9-pve and ROCm 6.4.3.
- `/bin/grep -rn` over HEAD `docs/*.md` for "never crash / zero crash / Nov|Dec 2025". Read `docs/REGISTER_GPU_FAULT_PRIOR_WORK.md:344-350,438-510,2500-2560` and partA.md in full.

**Not searched:**

- the host, ser8 archives, the private repository, rig filesystems or logs (no SSH);
- main-tree gitignored `logs/` for Nov–Dec 2025 (partA scanned pre-04-01 logs for fault patterns);
- `dashboard_work/MP1_FLEET_DEPLOY_RECORD.md`, which is absent from this worktree and from `/home/michael/distributed_prng_analysis/dashboard_work/`.

---

## Run-since-boot pattern (added 2026-11-16)

**The claim tested.** The owner recalls this, and Alpha and Beta confirmed instances in the repo. The first run or trial after a rig reboot is clean. A later run or trial on the same boot faults. A rebooted rig rejoins and finishes. This section states, for every GCVM-class event and every clean run in Table 1, where the rig stood on its current boot. **It concludes no cause and closes no lead.**

### Method

- **Boot times are measured, not inferred.** Every netconsole line carries the rig's kernel uptime (`[ 6511.76]`). Boot = receive time − uptime. A fault line therefore dates **its own** boot exactly.
  - For clean runs, the boot is the latest one observed before the run. Most are dated by the `NETCONSOLE READY` line each rig sends about 40–55 s after boot.
  - 197 boots were reconstructed from main tree `logs/netconsole_listener.log`, `logs/netconsole_all_rigs.log` and `logs/rrig6600c_netconsole.log`, read-only.
- **Runs and trials since boot are counted from run logs.** These are the PWC `N workers launched` lines in the main tree's surviving `logs/*.log`; PWC relaunches workers at each trial, so one line ≈ one trial start. For RANGE-MINER, one `gate12_*.log` is one run.
  - **"Position" = runs completed on this boot before this one / trial launch number within the run.**
  - **First** = 0 prior runs and trial 1, or no run since boot. **Later** = ≥1 prior run, or trial ≥2.
- **Limits** (a "0 prior" can be an undercount):
  - The S173 and FT073 run logs of 05-02 → 05-05 were deleted (main tree `SESSION_CHANGELOG_20260507_S174.md:31-39`).
  - No netconsole exists before 04-01, or for 04-02 → 04-16, or for July–August runs.
  - GPU work done outside TFM run logs is not counted: manual tests, OC tuning, probes, `rocm-smi` polling.
  - An outlet reboot restarts the motherboard only (owner testimony below), so "boot" here means OS boot, not a card power cycle.

### GCVM-class events (44 in Table 1, plus 1 new event from 2026-11-15)

| Event | Fault time · rig | Boot (anchor) | Uptime at fault | Position | Bucket |
|---|---|---|---|---|---|
| B-02 | 04-01 20:06:49 · rrig6600c | 19:19:02 (`RCN:9`, uptime 2866 s) | 47 min | 0 prior runs; `zmq_s159g_v1.log` started 20:05:56, the only run log in the window | **first** |
| B-05 | 04-02 17:49 · rrig6600 | ≈15:57, from uptime 6511 s (`TB_SUBMISSION_S159G_RIG6600_CRASHES.md:20`) | 108 min | 0 prior runs; `zmq_iommu_test_v1.log` started 17:33:52. The rig rebooted, rejoined, and the forward pass finished through leases (`:138-149`) | **first** |
| B-07 | 04-03 · rrig6600, rrig6600c | not recorded | — | series-level; no per-event time | UNKNOWN: no boot or time |
| B-16 | 04-04 · rrig6600c | not recorded | — | "~18 s into the run" (`SESSION_CHANGELOG_S162.md:25`); no 04-04 run logs survive | UNKNOWN: boot and prior runs |
| B-18 | 04-04 · rrig6600c 16:00.0 | uptime 3036 s; wall clock not recorded | 51 min | the doc says "~50 minutes into session" (`PROPOSAL_S162_RRIG6600C_CRASH_ROOT_CAUSE_v1_0.md:65`), but does not say no run preceded | UNKNOWN: prior runs |
| B-19 | 04-04 21:53:25 · rrig6600c | 20:45:52 (`TB_UPDATE_S162_OPTION_B_RESULTS.md:19`) | 68 min | trial started ~21:52 (`:20`); whether a run used 20:46 → 21:52 is not recorded; no 04-04 logs | UNKNOWN: prior runs |
| B-40, B-41 | 04-07..10 · rrig6600, rrig6600c | not recorded | — | series | UNKNOWN |
| B-53 | 04-17 10:36:53 · rrig6600, rrig6600c | 08:18:41 / 08:18:54 (`NCL:1`, `:11`) | 138 min | 1 prior run (`s163_100k_fixes_confirmed.log`, CLEAN B-52); trial 1 of the 2nd run | **later** (run) |
| B-56 | 04-17 14:34:25 · rrig6600c | 10:58:31 (`NCL:318`) | 216 min | 1 prior run; trial 2 | **later** |
| B-59 | 04-17 16:21:02 · rrig6600b | 16:18:32 (`NCL:614`) | **150 s** | no worker launch since boot; no run computing | **first** (before any run) |
| B-60 | 04-17 16:33:45 · rrig6600 | 14:52:20 (`NCL:613`) | 101 min | 0 prior runs; trial 1 of `s163_revert_test_1622.log` | **first** |
| B-61 | 04-17 17:21:58 · rrig6600b | 17:20:13 (`NCL:1070`) | **105 s** | no run since boot | **first** (before any run) |
| C-02 | 04-17 21:16:45 · rrig6600c | 17:20:25 (`NCL:1152`) | 236 min | 3 prior runs (1755, 1845, 1944); trial 3 of the 2044 run | **later** |
| C-08 | 04-19 15:12:52 · rrig6600 | 09:50:25 (`NCL:1802`) | 322 min | 6 prior runs (5 smoke runs + production_1433); trial 1 | **later** |
| C-11 | 04-20 17:06:40 · all three | 15:38:28–33 (`NCL:2189`, `:2254`, `:2291`) | 88 min | 0 prior runs; trial 1, 4 s after launch | **first** |
| C-13 | 04-20 18:51:19 · all three | 17:59:45–18:10:57 (`NCL:2601`, `:2613`, `:2614`) | 40–51 min | 0 prior runs; trial 1, 3 s after launch | **first** |
| C-15 | 04-20 19:17:07 · rrig6600b | 19:15:45 (`NCL:3244`) | **81 s** | first worker relaunch onto the rebooted rig (`s165_production2_1850.log:86352`) | **first** |
| C-17 | 04-20 19:38:40 · all three | 19:24:10–19:31:08 (`NCL:3371`, `:3372`, `:3397`) | 7–14 min | 0 prior runs; trial 1, 3 s after launch | **first** |
| C-31 | 04-23 15:52:32 · rrig6600c | 14:44:23 (from `NCL:4083`) | 68 min | 0 prior; trial 1 | **first** |
| C-33 | 04-23 16:42:01 · rrig6600c | 16:30:21 (`NCL:4268`) | 11.6 min | 0 prior runs; trial 2 | **later** (trial) |
| C-34 | 04-23 17:13:54 · rrig6600c | 17:11:08 (`NCL:4401`) | **165 s** | 0 prior; trial 1 | **first** |
| C-36 | 04-23 18:11:50 · rrig6600c | 17:46:06 (from `NCL:4583`) | 26 min | 0 prior; trial 1 | **first** |
| C-41 | 04-24 17:53:36 · rrig6600c | 16:17:27 (from `NCL:4803`) | 96 min | 0 prior; trial 1 | **first** |
| C-45 | 04-24 21:02:58 · rrig6600c | 18:34:12 (`NCL:4974`) | 149 min | 2 prior runs; trial 2 | **later** |
| C-46 | 04-24 21:25:53 · rrig6600 (+ rrig6600c) | 21:23:12 (`NCL:5175`); c 21:11:06 (`NCL:5127`) | **161 s** / 15 min | 0 prior; trial 1 | **first** |
| C-47 | 04-24 21:39:36 · rrig6600c | 21:36:55 (`NCL:5426`) | **160 s** | 0 prior runs; trial 2 (the rig rebooted mid-run, trial 2 launched 21:39:31) | **later** (trial) |
| C-49 | 04-25 08:53:56 · rrig6600c | 06:46:10 (from `NCL:5657`) | 128 min | 0 prior; trial 1 | **first** |
| C-51 | 04-25 17:19:48 · rrig6600c | 16:57:08 (from `NCL:5924`) | 23 min | 0 prior; trial 1 | **first** |
| C-56 | 04-25 22:17:50 · rrig6600c | 17:24:49 (`NCL:6079`) | 293 min | 4 prior runs; trial 2 | **later** |
| C-58 | 04-25 22:36:08 · rrig6600c | 22:20:43 (`NCL:6282`) | 15 min | 0 prior runs; trial 3 (rebooted mid-run) | **later** (trial) |
| C-60 | 04-26 07:47:46 · rrig6600c | 06:35:32 (from `NCL:6531`) | 72 min | 0 prior runs; trial 2 | **later** (trial) |
| C-63 | 04-26 15:42:55 · rrig6600c | 14:25:08 (from `NCL:6756`) | 78 min | 0 prior runs; trial 2 | **later** (trial) |
| C-69 | 04-26 19:11:27 · rrig6600c | 18:11:48 (`NCL:6896`) | 60 min | 0 prior; trial 1 | **first** |
| C-72 | 04-28 16:33:03 · rrig6600c | 15:41:16 (from `NCL:7048`) | 52 min | 0 prior; trial 1 | **first** |
| C-76 | 04-28 20:26:42 · rrig6600, rrig6600c | 17:26:49 / 16:58:43 (`NCL:7289`, `:7279`) | 3.0 h / 3.5 h | 1 / 2 prior runs; trial 3 | **later** |
| C-79 | 05-01 15:55:07 · rrig6600, rrig6600c | 14:48:29 / 14:48:27 (from `NCL:7597`, `:7601`) | 67 min | 0 prior runs; **trial 5** | **later** (trial) |
| C-82 | 05-01 22:04:32 · rrig6600c | 16:41:52 (`NCL:8052`) | 5.4 h | 1 prior run (the pool-6 CLEAN run C-81); trial 1 | **later** |
| C-85 | 05-02 22:55:41 · rrig6600c | 19:36:00 (`NCL:8285`) | 3.3 h | S173 run log deleted | UNKNOWN: prior runs (log deleted) |
| C-86 | 05-02 23:11:25 · rrig6600c | 23:06:50 (`NCL:8428`) | **274 s** | the rig rejoined the running S173 run 4.6 min after boot; no run can have completed on this boot. Log deleted | **first** (by uptime) |
| C-89 | 05-03 10:24:15 · rrig6600c | 07:57:18 (from `NCL:8511`) | 2.5 h | S173 v2 log deleted | UNKNOWN: prior runs |
| C-92 | 05-05 18:18:55 · rrig6600c | 15:45:10 (from `NCL:8736`) | 2.6 h | FT073 Run 2 started 17:43 (main tree `SESSION_CHANGELOG_20260507_S174.md:24`); nothing recorded for 15:45 → 17:43; log deleted. **If Run 2 was the first run on this boot, this is a first-run fault** (post-completion) | UNKNOWN: prior runs (log deleted) |
| C-105 | 05-11 17:52:20 · rrig6600c | 16:11:26 (from `NCL:9030`) | 101 min | 1 prior run (`S174_GATE_VALIDATION_20260511_174800.log`, 5M seeds, ~1.5 s of compute) | **later** |
| D-25 | 11-15 13:21:29 · pve-rig6600, pve-rig6600c | 08:27:06 / 08:26:56 (from `NCL:9506`, `:9508`) | 4.9 h | 1 prior run (attempt 1 `gate12_20261115_090553`, CLEAN) | **later** |
| **NEW** | 11-15 18:28:29 · pve-rig6600, pve-rig6600c (`NCL:10135` `0x00841051` .121 19:00.0; `:10149` `0x00801231` .163 16:00.0) | 13:55:42 / 13:52:32 (`NCL:10102`, `:10104`) | 4.5 h | 1 prior run (attempt 3 `gate12_20261115_174941`, CLEAN, credited); this is confirmation 1 `gate12_20261115_181853` (`distributed_config_t1_c8413236`, terminal `compute_lease_expiry` 18:38:29). **Both attempt 3 and confirmation 1 ran with `cleanup-torch: rigs=off`** (main tree `logs/gate12_20261115_174941_fleet.log:4`; `_181853_fleet.log:4`) | **later** |

### Clean runs beside them (62 in Table 1, plus 1 new)

| Clean run | Rig positions (boot anchor) | Bucket |
|---|---|---|
| B-46, B-47 (250K, 300K) | "After fresh reboot" (repo-root `SESSION_CHANGELOG_S163_FINAL.md:72-73`) | **first** |
| B-52 (04-17 08:59, 100K × 3 trials) | rrig6600 and rrig6600c 0 prior (`NCL:1`, `:11`); rrig6600b boot not observed | **first** (trials 2–3 clean) |
| C-04 (04-18 10:29, 5 trials) | all 0 prior (`NCL:1585`, `:1590`, `:1591`) | **first** (trials 2–5 clean) |
| C-18 (04-20 19:57, 2 trials) | all 0 prior (`NCL:3879-3881`) | **first** |
| C-65 (04-26 16:39, 2 rigs, pool 6, 5 trials) | rrig6600 and rrig6600c 0 prior (`NCL:6874-6875`) | **first** (trials 2–5 clean) |
| C-94 (05-05 18:31, FT073 Run 3) | rrig6600 and rrig6600c rebooted 18:29:47 / 18:25:39 (`NCL:8937`, `:8936`). The record says "post-reboot of all 3 rigs" (`S174.md:26`), **but no rrig6600b boot on 05-05 appears in netconsole**; its last observed boot is 05-04 19:17:33 (`NCL:8727`). Conflict recorded | **first** (rrig6600 and rrig6600c) |
| C-98 (05-09 21:02 gate validation) | all 0 prior (`NCL:8986-8988`) | **first** |
| D-24 (11-15 09:07, Run 4 attempt 1) | all 0 prior (`NCL:9506-9508`) | **first** |
| C-06 (04-19, 5 smoke runs + 1) | all three on one boot from 09:50 (`NCL:1802-1804`); runs 2–5 had 1–4 prior | **later** |
| C-07 (04-19 14:33, 5 trials) | all three 5 prior runs | **later** |
| C-25 (04-21 19:59) | rrig6600 0 prior (`NCL:4060`); rrig6600b and rrig6600c 2 prior (`NCL:3996`, `:3997`) | **later** |
| C-26 (04-22, rrig6600 cwsr=0) | rrig6600 runs 5, 6 and 7 all on boot 16:14:10 (`NCL:4076`, last line 21:04) without a GPU fault | **later** |
| C-54 (04-25 21:18) | all three 2 prior runs (`NCL:6078-6080`) | **later** |
| C-95 (05-06 17:46) | rrig6600 0 prior. rrig6600c: FT073 Run 3 was earlier on its 05-05 18:25 boot. rrig6600b on its 05-04 boot ran FT073 Runs 2 and 3 (logs deleted; `S174.md:24-26`) | **later** |
| C-96 (05-07 19:25) | rrig6600b and rrig6600c ≥1 prior | **later** |
| C-100 (05-09 22:24, D1 compute) | all three 2 prior (`NCL:8986-8988`) | **later** |
| C-81, C-83, C-103, C-106 | rrig6600 and rrig6600c **0 prior**; only rrig6600b on a later run (2, 4, 3 and 5 prior) | **later** (rrig6600b only) |
| NEW: Run 4 attempt 3 (11-15 17:51, `176ebfb6`, credited) | pve-rig6600 and pve-rig6600c 0 prior (`NCL:10102`, `:10104`); pve-rig6600b 2 prior (boot 08:26:58, `NCL:9507`) | **later** (rrig6600b only) |
| 41 others: all of Part A (23); B-08, 11, 12, 13, 14, 15, 17, 34, 36, 43, 44; C-87, C-88; D-01, 02, 08, 17, 20 | no netconsole before 04-01 or for 04-02 → 04-16; S173 logs deleted; July–August hosts not sending | UNKNOWN |

### Counts

| | first since boot | later (prior run, or trial ≥2) | UNKNOWN |
|---|---|---|---|
| **GCVM, 44 register events** | **19**, of which 2 had no run at all (B-59, B-61) | **16**: 10 after ≥1 prior run (B-53, B-56, C-02, C-08, C-45, C-56, C-76, C-82, C-105, D-25); 6 on trial ≥2 of the first run (C-33, C-47, C-58, C-60, C-63, C-79) | **9** (B-07, B-16, B-18, B-19, B-40, B-41, C-85, C-89, C-92) |
| **GCVM incl. the 11-15 18:28 event** (45) | 19 | **17** | 9 |
| **Clean, 62 register runs** | **9** (B-46, B-47, B-52, C-04, C-18, C-65, C-94, C-98, D-24); 4 of them multi-trial with later trials clean | **12**: 8 with rrig6600 or rrig6600c on a later run (C-06, C-07, C-25, C-26, C-54, C-95, C-96, C-100); 4 with only rrig6600b later (C-81, C-83, C-103, C-106) | **41** |
| **Clean incl. 11-15 attempt 3** (63) | 9 | **13** (attempt 3: rrig6600b only) | 41 |

### Counter-examples, each checked

1. **S164 trial 1 (C-10, 04-19 17:39:22, rrig6600 0000:19:00.0).**
   - Boot 16:52:45 (from `NCL:2114`, uptime 2796 s). 0 prior runs; trial 1 of `s164_production_1723.log`, launched 17:23:33.
   - It is a fault on the first trial since boot. **It is OTHER-GPU, not GCVM:** KFD preemption failure, then SMU, then MODE1 reset failed (`NCL:2114`, `:2127`, `:2135`).
   - **Verdict: a counter-example only if the pattern is meant to cover all GPU faults. Not one for GCVM.**
2. **Events 81–275 s after boot.**
   - **B-61 (105 s) and B-59 (150 s) on rrig6600b.** No worker had been launched on that boot; the GCVM came with no TFM run computing (B-59 vmid 6, B-61 vmid 2).
     - **Verdict: real counter-examples to "first run clean."** They are faults before any run. They are also counter-examples to a fault needing TFM work at all. Unrecorded host-side GPU activity at boot (e.g. a `rocm-smi` poll or the boot-time GPU probe) is not excluded.
   - **C-15 (81 s, rrig6600b).** GCVM 3 s after the first worker relaunch onto the rebooted rig. **Verdict: real** (first launch since boot).
   - **C-34 (165 s) and C-46 (161 s on rrig6600).** Trial 1, first launch since boot. **Verdict: real.**
   - **C-47 (160 s).** This was trial 2 since boot, about 1 min after trial 1. **Verdict: not a counter-example as defined** (later trial), but the boot was under 3 min old.
   - **C-86 (274 s).** The rig rejoined a running run. **Verdict: real by uptime;** the run log is deleted.
3. **rrig6600b running four runs on one boot without fault, 2026-11-15.**
   - pve-rig6600b boot 08:26:58 (`NCL:9507`). The next boot is 11-16 15:20:03. It carried attempt 1, attempt 2, attempt 3 and confirmation 1 on that boot.
   - It has no GCVM line that day. It has one SMU `0xFFFFFFFF` burst at 13:47:28 (`NCL:10098-10101`), 16 min after attempt 2's terminal and outside any run.
   - **Verdict: real, for rrig6600b.** The same holds earlier: rrig6600b's later runs in C-81, C-83, C-103 and C-106 were clean. rrig6600b has 6 GCVM events in the whole record (B-59, B-61, C-11, C-13, C-15, C-17), and **all 6** were at the first launch since boot or before any run.
4. **Faults on the first trial of the first run since boot (found in this pass): 17 GCVM events,** plus B-59 and B-61, which came before any run. B-02, B-05, B-60, C-11, C-13, C-15, C-17, C-31, C-34, C-36, C-41, C-46, C-49, C-51, C-69, C-72 and C-86, the 81–275 s cases included.
   - In C-11, C-13 and C-17 all three rigs faulted in the same second, 3–4 s after the first launch since boot.
   - **Verdict: these are counter-examples to "first run clean" as stated,** with two caveats:
     - In April the rigs were rebooted after nearly every fault, so the boots are short. Several were motherboard-only restarts, which by owner testimony leave the cards powered.
     - GPU work outside TFM run logs, e.g. OC tuning on 04-20 (`SESSION_CHANGELOG_20260420_S165.md:12-14`), is not counted.
5. **Clean later runs with rrig6600 or rrig6600c on a later run: 8.**
   - C-06, C-07, C-25, C-26, C-54, C-95, C-96 and C-100.
   - C-07 was run 6 on rrig6600's boot. The next run on that boot, C-08, faulted.
   - **Verdict: real counter-examples to "later run faults" as a rule.** They are consistent with "later runs are where faults *can* occur".

**Where the pattern holds exactly as stated:** RANGE-MINER on 2026-11-15, on pve-rig6600 and pve-rig6600c.
- attempt 1: first on its boot → clean.
- attempt 2: second → GCVM.
- reboot.
- attempt 3: first → clean.
- confirmation 1: second → GCVM.

Confirmation 1 also ran **without the rig torch cleanup** (`cleanup-torch: rigs=off`). That makes 2 of 2 first runs clean and 2 of 2 second runs faulting, on one host pair, in one day. Also consistent with it: B-53, B-56, C-02, C-08, C-45, C-56, C-76, C-82 and C-105. **The PWC-era record (04-17 → 05-11) does not show the pattern as a rule:** 17 GCVM events came at the first launch since boot or before any run (B-59, B-60, B-61, C-11, C-13, C-15, C-17, C-31, C-34, C-36, C-41, C-46, C-49, C-51, C-69, C-72, C-86), against 15 at a later run or trial.

### Owner testimony, 2026-11-16 (relayed by Alpha; not a document finding)

> - An **outlet power cycle restarts the motherboard only**. The cards stay powered from a separate server PSU.
> - Garbage `rocm-smi` readings in the PWC era led him to **cold-start the cards**. Those readings were **provoked by polling several times a second**.
> - A **cold start behaved about the same as a motherboard-only restart** for later faults.

Register notes beside it, record facts only:
- **Polling has a record.** Polling several times a second matches the January SMU-exhaustion entry: `watch -n 0.5 rocm-smi`, SMU `0xFFFFFFFF` `TransferTableSmu2Dram` (`docs/SESSION_CHANGELOG_20260120.md:3-36`; Table 2 A-05).
- **A "boot" may not reset the cards.** Every boot time in this section is an OS boot. By this testimony it is not necessarily a card power cycle.
- **No card-power state is recorded for any event.**

---

## How to record the next event

**Proposal, not a ruling.** It is offered so that this register can be kept as the living record. It changes no gate, no launch path and no prior ruling. The owner decides whether to adopt it.

Every fleet attempt gets **one entry, appended in date order** to a section `## Living record` at the end of this file. That includes clean runs, aborted runs and runs that never reached the GPU. Clean runs matter as much as failures, because the isolation tables are useless without them. Nothing is edited after the fact. A later correction is a new line, `CORRECTS <entry id>`, carrying its own evidence.

### Entry format (fixed; every field required; `n/r` = not recorded, never left blank)

```
### LR-<NNN> · <YYYY-MM-DD HH:MM–HH:MM TZ> · <run id>
- Run id / config:      distributed_config_t1_<hash>  ·  commit <sha>  ·  launch command file <path>
- Rigs and cards:       <host>/<CT or bare> : <GPU indices in use> (<n> workers)  — one line per rig, incl. Zeus
- Change from previous: LR-<NNN-1> → this: <exact diff: setting=old→new, env var, kernel arg, cap,
                        workers/rig, geometry, rig set>.  "none" if a repeat.
- Workload:             step / phase reached · seed range · seed caps · k/τ/geometry · phase-3/4 survivors
- Result:               CLEAN | FAILED at <phase/time>  ·  <what was lost: host, card, worker>  ·  rigs n/8 after
- Fault class:          GCVM | OVERLOAD | OTHER-GPU | OTHER | UNKNOWN | CLEAN   (rubric below)
- Evidence path:        <file:line> of the line that decides the class (e.g. netconsole_listener.log:<n>
                        "GCVM_L2_PROTECTION_FAULT_STATUS:…"); for CLEAN, the run summary path
- Observability armed:  netconsole receiving from <hosts> (proof line file:line) · console level · host
                        journal reachable y/n · telemetry series path
- Missing evidence:     what was not captured (required when class is UNKNOWN; "—" otherwise)
- Recorded by / where:  <session> · changelog or report path
```

### Classing rule (the same rubric this register used)

- **GCVM.** A `GCVM_L2_PROTECTION_FAULT` line is recorded for that event, on the failed host, inside the run window.
- **OVERLOAD.** An OOM kill, RAM exhaustion, an SSH or session limit, a load or job-count limit, a soft lockup, or a cause the record itself attributes to overwhelming the rig.
- **OTHER-GPU.** A ring timeout, SMU `0xFFFFFFFF`, a reset failure or device inaccessible, with no GCVM line.
- **OTHER.** A recorded non-GPU cause: a coordinator, lease, staging or config defect.
- **UNKNOWN.** No kernel-level or attributing line. Name what is missing.
- **No evidence means UNKNOWN.** A hypothesis is not evidence. A kernel line outranks prose attribution.

### Two standing requirements, so future entries are not UNKNOWN by construction

These are proposals.
1. **Before an attempt counts, prove the evidence path is live.** Record a netconsole proof line from each host before dispatch, using the existing NCPROOF practice (`docs/RUNBOOK_NETCONSOLE_REARM.md:87-115`). Without that, an empty capture cannot be told apart from no event: 05-15 → 08-22 had zero datagrams, and the hosts run `console_loglevel=3`.
2. **State the exact change from the previous attempt.** The record now has dozens of runs where several things changed at once, or where the change is unrecorded. Examples: the S170–S173 jitter values appear only in launch scripts; Run 4 attempt 1 and the retry are not distinguished in the record. One change per attempt, written down, is what makes the isolation tables usable.

## Living record

*(empty — first entry LR-001 is the next fleet attempt)*

---

*End of register. No cause is concluded; no lead is closed (docs/LEADS.md L-3 remains OPEN).*
