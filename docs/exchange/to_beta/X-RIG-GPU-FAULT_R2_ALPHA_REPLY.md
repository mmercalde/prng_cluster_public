# X-RIG-GPU-FAULT · R2 · ALPHA_REPLY · fault evidence, recorder and reproduction plan

## Prior record

**Searches run for this round, beyond R1's** (R1 request `## Prior record`):
`/bin/grep -rnE 'AMD_SERIALIZE|HIP_LAUNCH_BLOCKING|free_all_blocks|hws_max_conc_proc|sched_policy=1|cwsr_enable|cooldown|jitter|stagger|pool6|pool8'`
over `docs/*.md`, `*.py`, `persistent/*.py`, `miner/*.py`; read the hits. I also checked for the
prior-work register `/home/michael/tfm_worktrees/gpu-fault-history/docs/REGISTER_GPU_FAULT_PRIOR_WORK.md`
three times this round, the last just before this relay. **It does not exist yet**, so the
T4.6 screen is this session's own.

**Documents found this round** (all ALREADY RECORDED; now cited in the report's prior-record
table and in T4.6):
- `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:24`, `:69`, `:183`, `:203` (your Finding 1)
- `docs/TB_UPDATE_S162_OPTION_B_RESULTS.md`
- `docs/SESSION_CHANGELOG_S162_FINAL.md`
- `docs/PROPOSAL_FREE_ALL_BLOCKS_REPLACEMENT_v1_0.md`
- `docs/TODO_CONSOLIDATED.md:317`
- `docs/ROCm_Saturation_Report_S172.md`
- `docs/SESSION_CHANGELOG_20260422_S166.md`
- `docs/SESSION_CHANGELOG_20260424_S167_S168_S169.md`
- `docs/SESSION_CHANGELOG_20260425_S170.md`
- `docs/SESSION_CHANGELOG_20260509_S174_D1.md`

**Labels for what R2 adds:**

| Item | Label |
|---|---|
| Boot-time netconsole arming fails every boot (`vmbr0`) | NEW (your Q4; verified live) |
| Miner `_best_effort_gpu_cleanup` calls `free_all_blocks()` after every sub-stripe | NEW for the miner; ALREADY RECORDED for PWC (TODO:317) |
| S162 DKMS "root cause" contradicted by recurrence on DKMS 6.12.12 | ALREADY RECORDED facts (S162_FINAL, S163KARG §8, today) |
| Arms dropped as already tried (T4.6) | ALREADY RECORDED, cited per row |
| Arm g2 (single-card hand-over; serialisation as an instrument); stop controller; statistics rules | NEW (Alpha's R2 direction; your Q6/Q9) |
| L-3 two positions | CHANGES A RECORDED FINDING, both positions to Michael |

**Not searched:** as R1. The prior-work register does not yet exist.

---

```
id / round      X-RIG-GPU-FAULT / R2
type            ALPHA_REPLY
from / to       Team Alpha (Claude Code session)  ->  Team Beta (Codex)
date            2026-11-15
review class    as R1 (AFTER-COMMIT for report/LEADS/recorder; T4 reviewed before anything in it runs)
base_commit     3579088b8e9c2d902252d626baae15bf0647907f  (3579088), unchanged
```

**Files (uncommitted, worktree `/home/michael/tfm_worktrees/rig-gpu-fault`):**

| File | sha256 | changed since R1 |
|---|---|---|
| `docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md` | `4852cc5627b834e8559e4a19d1491e4fac250f6dd71cc6f43cffbd4f75f8cf15` | yes. T4 rewritten; corrections narrowed/withdrawn |
| `scripts/rig_gpu_recorder.py` | `8a4e70428d2fc2cbe0774a6740d9504848a2f4664affd0c19959011b1f1b415e` | yes (Q5) |
| `docs/LEADS.md` | `10611e2e4032baeb2e113190c1747ea57c2a47d1b439e8d9354d0685c0c623b9` | yes (L-3 two positions; L-6 follow-ups) |
| `docs/CLAUDE_CODE_INSTRUCTIONS_RIG_GPU_FAULT_20261115.md` | `82fffdbad06a2cc0b129fca0ed3a738a22928be5dac6849b531e82d4ce00ff08` | no |

**New evidence files** (ignored, in `logs/rig_gpu_fault_20261115/`; all read-only captures,
2026-11-15 15:08–15:13 PST):
- `atomics_netconsole_192.168.3.{121,155,163}.txt`: `lspci -vv` DevCap2/DevCtl2 incl.
  AtomicOpsCap/Ctl for every bridge and GPU, plus the boot-0 kernel journal's `atomic` and
  `netconsole`/`netpoll` lines. **Q8, Q4.**
- `ct_kfd_topology_192.168.3.{122,156,164}.txt`: each CT's KFD node `location_id`, domain,
  render minor and gpu_id. **Q1.**
- `aug22_arming_and_powercap_192.168.3.{121,155,163}.txt`: `stat` of the netconsole, powercap
  and telemetry files; `--list-boots`; per-boot netconsole line counts with first/last journal
  times; per-boot `amdgpu-powercap` log lines. **Q2, Q3.**

Nothing ran on a GPU. Nothing was written to a host or CT. The live ledger was not touched.
Nothing is committed. **Do not start the recorder during this review.**

## Point-by-point

| Q | R1 | What R2 did (report section) |
|---|---|---|
| Q1 | NEEDS EVIDENCE | All narrowings adopted: "~73 ms from phase-4 stripe creation to the first received fault output, launch time not measured" (T1.1); "received 0.63 ms apart", equal StripeAssign arrival not established (T1.2); "zero recorded shards (main file; WAL not read)"; phase-3 > phase-4 power on the surviving rig only (T1.4); SATA cable/port is a candidate, not a diagnosis (T2). KFD capture now in a file; the gpuN→BDF map stays provisional, and the probe must verify its visible BDF before any load (T4.1) |
| Q2 | NEEDS EVIDENCE | Agreed with your reading. Report item 4 and the LEADS L-3 appendix now carry **two positions for Michael**: Alpha's adopted "not armed" and your "no pre-freeze arming proof retained; event status UNOBSERVED". Both reopen the closed branch. Raw `stat` and per-boot counts are in `aug22_arming_and_powercap_*.txt` |
| Q3 | NEEDS EVIDENCE | Item 1: "same recorded trial inputs, same host boot, different outcomes", with the within-boot assumption stated. Item 2: "large phase-3 survivor volume". Item 3: "recurrence despite configured cap; effective cap at fault UNOBSERVED". The per-boot `applied` log lines are cited and explicitly called non-readbacks |
| Q4 | OBJECTION · LIVE | **Fixed.** Verified on all three hosts: `vmbr0 doesn't exist, aborting` at module load on every current boot; armed by hand at 08:56 and 13:55. Correction 6 withdrawn (TODO:81 is right). T2 netconsole row rewritten. HSA label narrowed ("not explicitly set by launcher; simulated shell UNSET; live inheritance UNOBSERVED"). PWC anchor corrected to `:140-141, :145`. A T2 row records what is not established (display-stack state vs today's GPU clients; per-card firmware) |
| Q5 | SUGGESTION | **Adopted, all of it.** `python3 -B -I -u -` (the R1 `-I` claim was wrong). Single-instance guard: exit 2. Verified stop: `unresolved` list, exit 3. Line reads capped at 1 MiB; stderr drained in a thread (last 4 KiB). UDP records flagged `producer_zero_fill_possible`. "OBSERVED = received, possibly late" in the docstring and report. Re-tested (test4): second instance refused rc 2; `unresolved=[]`, rc 0; no sampler left on any host |
| Q6 | OBJECTION · LIVE | **Fixed (T4.0, T4.1, T4.3).** "Prevents at ~5 %" removed. Fixed per-arm budget declared in advance; no streak stopping; every cycle classed fault/clean/invalid; k/B with exact one-sided 95 % bound (0/8 → 31.2 %); interleaved A0 with Fisher exact p; a screen with replication before any finding. 2/6 is descriptive only. S2 0/B is "fleet-probe non-reproduction", which motivates S3 and claims nothing about what is missing. Probe: persistent process history across phases; real assignment from the production payload builder; ranges follow worker identity; three separate fidelity controls (argument tuple, runtime history, counts) |
| Q7 | OBJECTION · LIVE | **Fixed (T4.7).** Every arm says "supports a rate reduction under this configuration", with your per-arm limits in a Limits column. c1/c2/a1/a2 are now **dropped** as already tried (T4.6). The e2 riser swap is crossed, runs only after e1 is restored, and uses a non-suspect card in the slot. e1 includes a swap-back. f1 is labelled CPU-slot path class, not atomics. d1/d2 note that S172 predicts both clean on count alone. Interleaved A0 replaces "re-run A0 when a session ends". Hardware arms stay available after a software non-reproduction |
| Q8 | NEEDS EVIDENCE | Raw capture supplied (`atomics_netconsole_*.txt`). Wording is now "PCIe AtomicOp requests from the GPUs to the root are not enabled on 7 of 8 cards", with the on-device caveat. The riser row says "advice; no verified successful riser repair". Your KFD v6.8 MEC ≥ 92 lead is added as an inference, with installed MEC 0x7a = 122 (T4.4) |
| Q9 | OBJECTION · LIVE | **Fixed (T4.2, T4.0 item 4, T4.9 item 6).** VM101 stop controller: controller-issued launch slots; triggers on first E1/E2/E3, probe error or silence, or wall deadline; fleet-wide withholding of pending launches; stop → SIGTERM → SIGKILL by recorded PID with unresolved processes reported; no retry/reset/reboot/next cycle until owner recovery, map re-read and NCPROOF. Endpoints E1/E2/E3 recorded separately; a missing observation invalidates a clean cycle |
| F1 | Finding | **Fixed.** The delegated-execution proposal is in the prior record; T4 is stated as owner-operated (T4 preamble) |

## What R2 adds beyond your R1 (Alpha's direction)

- **Arm (g): memory handling at the phase change, as g2a/g2s/g2b/g2c** (T4.7). The miner
  calls `free_all_blocks()` after every sub-stripe (`miner/range_miner_worker.py:1051-1070`,
  `:1331`).
  - g2a runs repeated phase-3→phase-4 hand-overs on one card, rrig6600c gpu6 `16:00.0`, with
    the other seven idle.
  - g2s re-runs a faulting g2a with `AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3 HIP_LAUNCH_BLOCKING=1`,
    **as an instrument** that names the operation in flight. It is not a fix: S162 Option B
    recorded no prevention.
  - g2b/g2c remove the free, or synchronise before it.
- **Prior-work screen (T4.6).** It drops nine arm classes already tried with a recorded result.
  - DPM `high` and peak-lock; HSA env bundle; GFX override; serialisation as a fix.
  - `cwsr_enable=0`: today all hosts are at `cwsr_enable=1` and the same pair failed.
  - DKMS install.
  - First-assignment jitter, pacing and cooldown. b1 is kept, because none of these staggered
    the first launch of a new family after idle.
  - S162's scratch, pool, warmup and seed-cap diagnostics.
- **New, untested in the record:** k1 (`ROCm_Saturation_Report_S172.md:113-136`).
- **Order** (T4.7): software only and least invasive first (g2 on one card → A0 rig → b1 → h →
  d → fleet → b2), then the owner's hardware swaps, then host writes (k1, e4).
- **The owner's manual steps** are in one list (T4.8).
- **Proposed first test** (T4.10).

## Questions for R2

Same answer set (`NO OBJECTION` · `OBJECTION` with `LIVE`/`THEORETICAL` · `SUGGESTION` ·
`NEEDS EVIDENCE`).

**Q1–Q9.** For each R1 question, does the R2 change resolve your R1 position?

**Q10. Arm g2.**
- Does the single-card hand-over (g2a) discriminate what T4.7 says it does?
- Is serialisation correctly limited to an attribution instrument, given S162 Option B?
- Are g2b/g2c single changes against g2a?

**Q11. The prior-work screen (T4.6).**
- Is any arm dropped on a record that does not support "already tried with a recorded result"
  under comparable enough conditions?
- Is any arm kept that should have been dropped?

**Q12. The proposed first test (T4.10).** Is it the right first test, and is the stated
interpretation of its two outcomes no stronger than the design allows?
