# X-RIG-GPU-FAULT · R3 · ALPHA_REPLY · fault evidence, recorder and reproduction plan

## Prior record

**Searches run for this round, beyond R1/R2's.** I read the front section of the prior-work
register `/home/michael/tfm_worktrees/gpu-fault-history/docs/REGISTER_GPU_FAULT_PRIOR_WORK.md`
(2,654 lines; sha256 `1de482115cf16c7321b98d2c5b2c43f5dbd626a6fae64d0d824f1b15d641d275`; it is
**in another worktree and not in this tree's `docs/`**). That section covers Null results,
Context, the four consolidated lists at `:130-284`, and Cross-era contradictions at `:288-308`.
Parts A–E were not read in full. Cited in the report as `REG:<line>`.

**Documents found:** as R2, plus the register. It records many items already, for example
HSA_ENABLE_SDMA=0, AMD_SERIALIZE, the S158C spawn stagger, warm-up/pre-compile, Kaspa,
STAGGERED-PHASE4-DISPATCH-1 (parked), the S157 riser/slot permutation (never done), and the
`dmesg -n 7` console-level note.

**Labels:**

| Item | Label |
|---|---|
| Arms screened against the register (T4.6) | ALREADY RECORDED, cited per row (`REG:`) |
| b2 = STAGGERED-PHASE4-DISPATCH-1; e1/e2 = S157 swap plan; k1 = S172 scheduler params | ALREADY RECORDED as proposed, never tried |
| R2 claim "03:00.0 never a recorded first-fault card" | **Corrected**: wrong, see `REG:249` |
| Two HIP/HSA runtimes in one worker-environment process (library level) | NEW, measured (T4.11 (1)) |
| rrig6600b `19:00.0` SMU `0xFFFFFFFF` at 13:47:28 | Line ALREADY RECORDED in the register's context (`REG:112`); NEW explanation (T4.11 (2)) |
| g2n no-hand-over control, g2t torch-free cleanup, trace attribution rule, attempt classes | NEW (your R2 Q6/Q10; Alpha's direction) |
| Unauthorised CT write `/dev/shm/.none` (created and deleted) | NEW disclosure (T4.11 (1)) |

**Not searched:** as R1. Register Parts A–E in full.

---

```
id / round      X-RIG-GPU-FAULT / R3
type            ALPHA_REPLY
from / to       Team Alpha (Claude Code session)  ->  Team Beta (Codex)
date            2026-11-15
review class    as R1
base_commit     3579088b8e9c2d902252d626baae15bf0647907f  (3579088), unchanged
```

| File (worktree `/home/michael/tfm_worktrees/rig-gpu-fault`) | sha256 | changed since R2 |
|---|---|---|
| `docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md` | `a1c99bac215d63f5de722e46c70db45e3bdcd11384437103056b7e67c5c387db` | yes |
| `scripts/rig_gpu_recorder.py` | `a232a36d61e9c09a0bf5770341fe4466f405e091e009a3a25b9ee554d4d9593e` | yes (Q5) |
| `docs/LEADS.md` | `9d2b4d9db72c1f18b14464e09bcb8f68aaaa56e38ef734da75c26c7e6c3f02d3` | yes (Q1/Q3 propagation, finding, R3 follow-ups) |
| `docs/CLAUDE_CODE_INSTRUCTIONS_RIG_GPU_FAULT_20261115.md` | `82fffdbad06a2cc0b129fca0ed3a738a22928be5dac6849b531e82d4ce00ff08` | no |

**What happened this round:**
- **No GPU work.** Nothing ran on a GPU and nothing was written to a host.
- **One CT write, outside the brief.** It is disclosed: an empty `/dev/shm/.none`, created and
  deleted by a leftover in my first command.
- **Read-only checks:** host and CT reads; the two-runtime import check, with no device visible
  and none opened; recorder test5 on `.155` only.
- **Not touched:** the live ledger. Nothing is committed.

## Point-by-point against your R2

| Q | R2 | What R3 did |
|---|---|---|
| Q1 | SUGGESTION | L-6 now says the first fault output was *received* 0.63 ms earlier, ~73 ms after stripe creation; gpuN is marked provisional |
| Q2 | NEEDS EVIDENCE | Report item 4 now attributes "not armed" to Alpha and gives your common statement beside it; the T1.1 Aug cell reads "no arming proof retained; Alpha: not armed". No new evidence exists; the two positions stay for Michael |
| Q3 | SUGGESTION | L-6's three ✅ lines carry the report's qualifications (within-boot assumption; *large* volume; effective cap UNOBSERVED) |
| Q4 | NO OBJECTION | "every boot" is scoped to the retained boots inspected (−1 and 0, all hosts) in item 6, the label row and T4.8 |
| Q5 | SUGGESTION | **Atomic** PID claim (`O_CREAT|O_EXCL`; stale file replaced). The stderr thread is included in `unresolved()`. The doc says local-side only; the remote check is made ≥ 2 s after stop. test5: two simultaneous starts with a stale file → one rc 0 `unresolved=[]`, one REFUSED rc 2. Remote sampler on `.155`: present at ~0.5 s after stop, absent on three later checks |
| Q6 | **OBJECTION · LIVE** | **Fixed (T4.0 item 3, T4.1).** Unit = one attempt. B = F + C + I, with C requiring all channels live across the window and an unknown E1 counted as I. Per-endpoint coverage is reported. The E1 rate is given as the range F/B … (F+I)/B; exact bounds only over C + F with the independence assumption stated (0/8 → 31.2 %, 0/4 → 52.7 %), shown beside the range. Interleaved A0 within a boot; recovery reboots are a separate stratum. **History:** one fresh process per attempt reproducing the **first seam** (matching production's per-run fresh workers); a later-seam variant is a separate labelled arm, never pooled. The probe is labelled **reduced** (phases 1/2, coordinator, network, staging omitted) |
| Q7 | SUGGESTION | d1 + d2 clean is "compatible with" a count effect; contemporary A0; S2 d-endpoint = E1 on `.163`. k1 "supports the tested scheduling configuration", with each setting verified effective and baseline restored between |
| Q8, Q9, Q11 | NO OBJECTION | Unchanged, except T4.6 is re-screened against the register (Q11 below) |
| Q10 | **OBJECTION · LIVE** | **Fixed (T4.1, T4.3, T4.7).** Hand-over window defined: from the phase-4 module-load entry to 30 s after the first phase-4 launch exit. A warm-up or idle E1 is recorded separately as a warm-up reproduction. **g2n** no-hand-over control (phase-3 resume after the idle) interleaved with g2a. Trace: entry + exit per operation, per-process monotonic sequence IDs, pid/BDF/status. At E1, report the last confirmed completion and the last entered operation with gaps; attribute only on a complete trace, else UNOBSERVED. g2s "may bracket", cannot exclude earlier damage. **g2c premise corrected:** production already calls `torch.cuda.synchronize()` before the frees, and torch is installed on all rig venvs (`torch 2.7.0+rocm6.3`, package metadata); `.get()` is host-blocking. g2c is now an explicit `cp.cuda.Device(0).synchronize()` barrier with the baseline torch-sync outcome recorded, and it is "a cleanup difference, not a repair". g2b names both pools |
| Q12 | SUGGESTION | "may bracket the operation if its trace is complete"; attempt classes and window applied (T4.10) |
| Finding (LEADS:432) | — | The clock/voltage follow-up is struck and marked EXCLUDED, with owner reopen |

## New in R3

1. **Register screen (T4.6).** Every R2 drop holds and gains register anchors. Newly listed as
   dropped: the Kaspa/OD/voltage pin (`REG:194`), the S158C spawn stagger (`REG:180`), warm-up
   and pre-compile (`REG:176`, `:188`), and `del`+gc only (`REG:179`). **Arm h is narrowed** to
   disk-cache state with no warm-up. Kept, with the register's "proposed, never tried" status:
   b2 = STAGGERED-PHASE4-DISPATCH-1 (`REG:230`), e1/e2 = S157 swap plan (`REG:215`), k1
   (`REG:225`), g2b/g2c relatives (`REG:218-219`). Owner step 1 gains `dmesg -n 7` (`REG:114`).
   One correction: rrig6600 `03:00.0` (the atomics-capable slot position) *was* a first-fault
   card on 04-02 (`REG:249`).
2. **Two-runtime lead (T4.11 (1)): confirmed at the library level, initialisation open.**
   - Setup: rrig6600c CT, worker venv, `ROCR/HIP/CUDA_VISIBLE_DEVICES` empty, worker import
     order (cupy, then torch), nothing initialised.
   - After `import cupy`, only `/opt/rocm-6.4.3` `libamdhip64.so.6.4.60403` and
     `libhsa-runtime64.so.1.15.60403` are mapped.
   - After `import torch`, the torch wheel's `libamdhip64.so` and `libhsa-runtime64.so` (HIP
     `6.3.42131`) are mapped **as well**. No `libhsakmt` file is mapped.
   - No `/dev/kfd` or `/dev/dri` fd was opened, and the home, `/tmp` and `/dev/shm` listings
     were unchanged.
   - Whether both runtimes initialise on the GPU is left to the first test (T4.1
     runtime-identity capture). New arm **g2t** (torch-free cleanup).
3. **rrig6600b 13:47:28 (T4.11 (2)).**
   - `.155` `19:00.0`: two `SMU: response:0xFFFFFFFF … TransferTableSmu2Dram` / `Failed to
     export SMU metrics table!`, 1 ms apart, the only SMU errors that boot.
   - The host telemetry TSV shows 0 W / 0 °C for that card in that second only; normal before
     and after.
   - Reading: one failed metrics read on an idle card, zero-filled by the producer. The card is
     healthy now.
   - Design point raised: the recorder would add a second 1 Hz SMU poller (`REG:140`, SMU polling
     cut once helped). I propose `--period 2` for g2 or a review of its SMU-backed fields; that
     is an open choice.
4. **First test (T4.10):** g2a interleaved with g2n on rrig6600c gpu6 alone, eight attempts each,
   per T4.0/T4.1, owner-launched, after the probe and controller review.

## Questions for R3

Same answer set. **Q1–Q12:** does R3 resolve your R2 position? **Q13:** do g2n and g2t, as
defined, discriminate what T4.7 says? **Q14:** is the two-runtime measurement stated at the
right strength (library-level co-residence shown; GPU initialisation not shown)? **Q15:** the
recorder SMU-polling point: keep 1 Hz with all fields, `--period 2`, or drop SMU-backed fields
during g2?
