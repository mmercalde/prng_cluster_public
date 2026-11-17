# X-RIG-GPU-FAULT · R1 · REVIEW_REQUEST · rig GPU fault: evidence, config diff, recorder, reproduction plan

## Prior record

**Searches run** (worktree `/home/michael/tfm_worktrees/rig-gpu-fault` at `3579088`, 2026-11-15).
The full commands and hit counts are in the report's `## Prior record`:

- `/bin/grep -lE 'GCVM|PROTECTION_FAULT|0x00801231|0x00841011|device lost from bus|rig crash|host hang|voltage sag|sag|power cap|powercap|dpm-pin|DPM harness|dpm_peak|Kaspa|netconsole|L03|VBIOS|riser|pci=noaer|AER'`
  over `docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`, `docs/TB_RULING_REQUEST_*`,
  `docs/PROPOSAL_*`, `docs/CHAPTER_*`, `docs/*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`,
  `docs/BACKLOG.md`, `docs/LEADS.md`, `docs/exchange/`.
- `/bin/grep -rniE 'powercap|power1_cap|dpm harness|dpm-pin|gfxoff|ppfeaturemask|lockup_timeout|HSA_ENABLE_DEBUG_TRAP|audio=0|amdgpu\.dc'` over `docs/`.
- `/bin/grep -rlnE 'powercap|rig-power-telemetry|6668|power1_cap_min'` over the repo, the live
  tree's `docs/` and `scripts/`. No record of the host power-cap or telemetry installation.

**Documents that record the subject:**

| Document | What it records |
|---|---|
| `docs/PROPOSAL_S162_RRIG6600C_CRASH_ROOT_CAUSE_v1_0.md` §1–2 | Same rrig6600c `16:00.0` `0x00801231` SQC signature, 2026-04-04. **rrig6600c alone ran clean; the 3-rig load crashed** |
| `docs/TB_Incident_Report_rrig6600c_S163KARG.md` §3–5, §8, §11 | 2026-04-17, same signature; perf `high` at the time |
| `docs/SESSION_CHANGELOG_20260419_S164.md` | audio=0, dc=0, lockup/recovery, Kaspa pin (Ubuntu installs) |
| `docs/SESSION_CHANGELOG_20260420_S165.md` | Kaspa pin reverted as crash-causing; dpm-pin disabled; HSA_ENABLE_DEBUG_TRAP=0 |
| `docs/SESSION_CHANGELOG_20260425_S170.md:274-286` | DPM peak-lock service; rrig6600c L03 VBIOS outlier |
| `docs/SESSION_CHANGELOG_20260402_S159G.md:69-81`, `docs/TB_FINAL_UPDATE_S159G_ROOT_CAUSE_CONFIRMED.md` | HSA env standardised via `/etc/environment` |
| `docs/TFM_PROJECT_FACTS_SKILL.md` §2.50 | GCVM fault "avoided, not eliminated" |
| `docs/RIG_CRASH_FORENSIC_20260822.md`, `docs/LEADS.md` L-3 | Aug-22 phase-4 hang; H2 sag / H3 volume |
| `docs/TODO_CONSOLIDATED.md:80-82, :281, :323` | Power-cap/telemetry check, netconsole persistence, circuit check (OWNER), DPM harness, GCVM root cause open |
| `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:21` | Owner ruling 3: power caps etc. dropped as Run-4 prerequisites |

**Labels.** The report's "Labels for each item" table classifies every item. It holds:
- one `CHANGES A RECORDED FINDING`: L-3 netconsole arm state, **adopted by Alpha at gate 1**;
- `ALREADY RECORDED` for the recurring signature, the multi-rig dependence and the phase-4 seam;
- `NEW` for the rest.

None changes an owner ruling.

**Not searched:** Zeus, the ser8 archives (April captures), the private repository,
`RUNBOOK_Proxmox_amdgpu_Rig_Install_v1_0.md` (not in the repo), bare-metal Ubuntu boot targets
(not booted), switch and PDU logs.

---

```
id / round      X-RIG-GPU-FAULT / R1
type            REVIEW_REQUEST
from / to       Team Alpha (Claude Code session)  ->  Team Beta (Codex)
date            2026-11-15
review class    AFTER-COMMIT for the report, LEADS edits and the recorder (documents, tools,
                monitoring: owner ruling 2). The T4 plan is reviewed BEFORE anything in it runs;
                the probe it describes is not built and will come as worker/kernel class
base_commit     3579088b8e9c2d902252d626baae15bf0647907f  (3579088)
brief           docs/CLAUDE_CODE_INSTRUCTIONS_RIG_GPU_FAULT_20261115.md
```

**Files under review** (uncommitted, in the worktree above):

| File | Status | sha256 |
|---|---|---|
| `docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md` | new | `c753e3d69e154110378876a671f706f1b9c2c7f22b8cf7b0c3f5d5dd61899ced` |
| `scripts/rig_gpu_recorder.py` | new | `47a9a36ca168bcd8f0c2b81e8f3442b3606f78f8793bd3bc9d01f2514318b355` |
| `docs/LEADS.md` | modified (L-3 correction appended, L-6 added) | `6924e33db21a0cf90e78c44ab4c6a88887973f64b2615118f006aac7d3fc0737` |
| `docs/CLAUDE_CODE_INSTRUCTIONS_RIG_GPU_FAULT_20261115.md` | new (brief, verbatim) | `82fffdbad06a2cc0b129fca0ed3a738a22928be5dac6849b531e82d4ce00ff08` |

**Evidence you can read** (outside the repo, read-only):
- `/home/michael/run4_attempt2_20261115_131154/`:
  - `logs/netconsole_all_rigs.log` lines 229–508 and the tail;
  - `logs/gate12_20261115_131154.log:535-543, :702`;
  - `ledger/miner_ledger.db`, a copy; open with `?mode=ro&immutable=1`.
- `logs/rig_gpu_fault_20261115/` in the worktree (ignored): the raw host and CT captures behind
  T2 and T1.5.
- `logs/rig_gpu_recorder/test{1,2,3}/` (ignored): recorder test output.

**What was done.**
- Read-only SSH only. Nothing was written on any host or CT. No GPU was opened.
- The live miner ledger was not touched.
- Nothing was committed.
- Nothing in T4 has run.

**Do not start `scripts/rig_gpu_recorder.py` during this review.** It appends to `logs/`, which
the relay snapshots.

## Questions

Answer each with `NO OBJECTION` · `OBJECTION` (with `LIVE` or `THEORETICAL`) · `SUGGESTION` ·
`NEEDS EVIDENCE`.

**Q1. Evidence (report T1.1–T1.5).**
- Are the event table, today's sequence and the bus → `gpuN` → VBIOS map supported by the
  anchors given?
- In particular: the 74 ms dispatch-to-fault interval; zero phase-4 shards from rrig6600 and
  rrig6600c; the KFD-node-order index mapping.
- Is any claim stronger than its anchor?

**Q2. The L-3 correction (report "Null results and corrections" item 4; LEADS L-3 appendix).**
- The claim: netconsole was not armed during the 2026-08-22 freeze.
- It rests on file birth times (19:25 on all hosts) and on zero netconsole kernel lines in the
  incident boot.
- Is that sufficient? Is there an arming route that would leave neither trace?

**Q3. Null results 1–3.**
- 1: identical config and trial between a pass and a fail.
- 2: volume is not necessary.
- 3: the power cap is not sufficient.
- Does each follow from the measurement cited, and is any overstated?

**Q4. Configuration difference (T2).**
- Is anything material missing from the Proxmox-today vs S164/S165/S170 comparison?
- Is any row mis-anchored?
- The HSA-env row is by reading plus a simulated non-interactive shell, not a live
  `/proc/PID/environ`. Is it stated at the right strength?

**Q5. Recorder (T3, `scripts/rig_gpu_recorder.py`).**
- Is it read-only on the hosts as claimed? See `REMOTE_SAMPLER` `:67-155` and the ssh command
  `:197-204`.
- Does it ever emit a zero, or an OBSERVED record, for a silent host? See the watchdog
  `:273-286`.
- Is memory bounded, does it stop cleanly via the PID file, and is the host derivation
  (`:323-331`) acceptable under the no-hardcoding rule?

**Q6. Plan design (T4.0–T4.2).**
- The single-rig false-pass rule.
- The 2/6 base rate from all six phase-4 starts on record, with its stated assumption.
- The ≥8-clean-cycles criterion.
- The S1 → S2 → S3 escalation.
- Are these sound? Is the probe in T4.1 faithful enough to the production phase-4 seam that a
  probe non-reproduction means what the plan says?

**Q7. Candidates (T4.5).**
- For each of b1, b2, d1/d2/d3, g, c1, c2, a1, a2, e1–e4, f1: do the confirm and refute
  criteria actually discriminate?
- Is each a single change?
- Are the controls adequate: d2 for load count, the c1/c2 split, e1 vs e2 for card vs riser vs
  slot?
- Is the recommended order right?

**Q8. External reports (T4.3).**
- Is each operator or vendor claim stated no stronger than the source supports?
- Is the atomics measurement (T4.4) read correctly from `lspci -vv` `AtomicOpsCap`/`AtomicOpsCtl`?

**Q9. Hang risk (T4.6).** Is the list of ways the plan could hang a host, or lose a rig,
complete? Is anything proposed that should be excluded?
