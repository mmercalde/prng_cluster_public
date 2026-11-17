# PLAN — rig GPU fault: torch-runtime arms, escalation ladder, probe and stop controller

**Thread:** X-RIG-GPU-PROBE (new; X-RIG-GPU-FAULT is out of rounds).
**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_RIG_GPU_FAULT_20261115.md`, plus Alpha's rulings after
X-RIG-GPU-FAULT R3 and the owner additions of 15:59 PST.
**Companion report:** `docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md`.
**Status:** design and code. **Nothing has run on a GPU. Nothing has been written to a host. No
host or CT write was made for this plan.** Only off-GPU tests have run. No root cause is claimed.

---

## Prior record

**Searches run (2026-11-15, worktree `rig-gpu-fault` at `3579088`):**
- I read everything X-RIG-GPU-FAULT R1–R3 lists.
- I read the prior-work register `/home/michael/tfm_worktrees/gpu-fault-history/docs/REGISTER_GPU_FAULT_PRIOR_WORK.md`
  (3,275 lines; another worktree). From it I read the front section (`:75-308`), the Fault
  classification (`:2659-2867`), the pre-PWC section (`:3032-3226`) and the recording format
  (`:3229-3268`).
- `git show -s` of `41165dd`, `b425171`, `20b0e7c`, `0a316b8`; `git show b425171:sieve_filter.py`.
- `git log -S'S163_MEM_DEBUG'` over `persistent_worker_coordinator.py persistent/ sieve_filter.py scripts/ agents/`.
- `git grep -n S163_MEM_DEBUG HEAD`.
- `/bin/grep -rnE '\btorch\b'` over `miner/*.py sieve_gpu_worker.py prng_registry.py persistent/*.py sieve_filter.py reverse_sieve_filter.py adaptive_thresholds.py hybrid_strategy.py`.
- `/bin/grep -rn 'get_pytorch_gpu_reference|_get_torch('` over all `.py`.
- Main tree `logs/rrig6600c_netconsole.log:1-12` (first raw GCVM line).
- The run bundle `logs/gate12_20261115_131154.log:537`.
- The bundle ledger copy (`immutable=1`), for the gpu6 stripes.

**Documents that record the subject:**

| Document | What it settles or records |
|---|---|
| Register, Fault classification (`:2659-2867`) | 44 GCVM-class events in 199 failures, with 62 clean runs beside them (Table 1). The first raw GCVM line is 04-01 20:06:49 (`:2739`). Onset timing: 10 of 26 Part-C GCVM events begin 3–9 s after `workers launched`, and 15 within 30 s (`:2747`). Same-second multi-rig onsets (`:2741-2746`) |
| Register, pre-PWC section (`:3032-3226`) | torch was in the sieve worker during the zero-GCVM era, and that era had **no kernel capture** (`:3038`, `:3041`, `:3080-3097`). Era differences (`:3162-3210`) |
| Register, list (3) (`:204-231`) | STAGGERED-PHASE4-DISPATCH-1 parked (`:230`); S157 swap plan never done (`:215`); scheduler params never tried (`:225`) |
| `ROCm_Saturation_Report_S172.md:29-35` | 6 workers per rig stable, 8 crash (PWC) |
| `PROPOSAL_S162_RRIG6600C_CRASH_ROOT_CAUSE_v1_0.md` §1 | One rig alone ran clean |
| `PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:24`, `:69`, `:183`, `:203` | Freeze reproduction is not delegated. **This plan is owner-operated** |
| X-RIG-GPU-FAULT R1–R3 reviews | Statistics, windows, trace attribution and control readings; adopted here |

**Labels:**

| Item | Label |
|---|---|
| torch history in the worker (presence/frequency by era) | ALREADY RECORDED (register `:3080-3097`; anchors re-verified here) |
| torch call state per PWC-era GCVM event (`S163_MEM_DEBUG` gating) | NEW: the register does not separate it (§1.3) |
| Arms P / P1 / M / N; the escalation ladder; the budget; the worker flag design | NEW |
| Probe, stop controller, gates, mutants; recorder passive mode and flock | NEW |
| STAGGERED-PHASE4-DISPATCH-1, S157 swaps, k1 | ALREADY RECORDED, never tried |

**Not searched:** register Parts A–E in full; Zeus; the ser8 archives; the private repository;
rig filesystems beyond the read-only reads recorded in the report.

---

## 1. The lead, restated with the register's history

### 1.1 Measured today (report T4.11 (1))

In the worker's own environment, `import cupy` maps the system ROCm 6.4.3 HIP/HSA. A later
`import torch` maps **a second** HIP/HSA pair, the torch wheel's HIP 6.3. No device handle was
opened. Whether both runtimes initialise on the GPU in a live worker is **not shown**.

### 1.2 torch's presence alone does not separate the eras

Register anchors, re-verified here:

| When | torch in the sieve worker process | Anchor (verified) |
|---|---|---|
| 2025-11-27 → 12-24 | module-level `import torch` in `prng_registry` (imported by `sieve_filter`) | register `:3080`; removed by **`41165dd` (2025-12-24 18:57)**. Its body says the import "corrupted the HIP/ROCm environment, causing CuPy kernel compilation to fail with 'stddef.h not found' or 'hip/hip_version.h not found'", and that "PyTorch ROCm modifies HIP include paths when imported, breaking subsequent CuPy HIPRTC compilation". It kept the order cupy → compile → torch |
| 2026-01-14 → | cleanup `import torch; torch.cuda.synchronize(); empty_cache()` + CuPy frees, **once per job** | **`b425171`** `sieve_filter.py:59-75` (verified text) |
| 2026-01-26 → | the same cleanup **between chunks** | **`20b0e7c`** "Inter-chunk GPU cleanup prevents Step 1 hangs" |
| 2026-03-09 → | PWC SSH worker `sieve_gpu_worker.py:84-87`, per job (`:348`), long-lived process | register `:3086`; HEAD verified |
| 2026-04-08 → | TCP-PWC cleanup: torch **gated** behind `S163_MEM_DEBUG=1` | **`0a316b8`**; HEAD `sieve_filter.py:139-145` |
| 2026-07-18 → | miner: torch cleanup after **every sub-stripe**, long-lived daemon | `miner/range_miner_worker.py:1051-1072`, `:1331` |

Statement for the lead (added to L-6):
- torch was in the worker throughout an era with **no GCVM records**, and that era had **no
  kernel capture**: the first netconsole entry is 2026-04-01 (register `:3038`), so "zero GCVM"
  there is **absence of record**.
- torch's presence alone does not separate the eras. What changed is **how often the torch
  cleanup runs** (per job → per chunk → per sub-stripe) and that **the process no longer exits
  afterwards** (job-per-process → PWC → long-lived daemon).
- The record has not counted that frequency for any era (register `:3097`).

### 1.3 torch-call state of the PWC-era GCVM events is UNKNOWN per event (new)

On the TCP-PWC path, the torch cleanup ran only when `S163_MEM_DEBUG=1` from 04-08 (`0a316b8`).
- It was hardcoded 0 in the workers until `ba86d77` (04-15, "hardcoded 0 was overriding
  coordinator env"), then propagated (`57eb048`).
- It was forced to 1 by `launch_s174.py:380` and `launch_s174_baseline_pool8_25k.sh:88`.
- The main-tree logs carry 18 `"S163_MEM_DEBUG": "1"` and 2 `S163_MEM_DEBUG=0` occurrences,
  not yet attributed to runs.
- The reverse path `reverse_sieve_filter.py:73-76` and the SSH-PWC `sieve_gpu_worker.py:84-87`
  are unconditional.

**So for each Table-1 GCVM event, whether torch had been called at all in the faulting process
is not established.** A GCVM event in a process that provably never called torch would weigh
against a torch call being necessary; it would not close the lead. **Proposed record check
(off-GPU, cheap):** for each of the 44 GCVM rows, record the worker path (TCP-PWC / SSH-PWC /
ZMQ / miner), the run's `S163_MEM_DEBUG` value, and torch called yes / no / UNKNOWN. That fits
the history agent's format (register `:3229-3268`).

### 1.4 Onset window: what the worker is doing (register `:2747`; today `gate12…log:537`, `NL:9639`)

| Engine | Window | What the worker is doing then | torch called yet in that process? |
|---|---|---|---|
| **Miner, today** | 13:21:29.024 phase-4 bound logged → 13:21:29.062 first GCVM (38 ms) | It receives StripeAssign. The reverse-hybrid `_get_kernel` is first resolved in this process: a compile or disk-cache load and a module load. The phase-4 buffers are allocated and copied host→device, and the **first launch** of the reverse kernel is made. The first phase-4 cleanup runs only after the first phase-4 sub-stripe completes (≥ 260 ms on rrig6600b, ledger) | **Yes, many times earlier**: after every sub-stripe of phases 1–3. Its last call was ~200 s before, at the end of phase 3. **Not** inside the window |
| **TCP-PWC, Part C** (10 of 26 events at launch +3–9 s) | after `workers launched` | The worker connects and imports `sieve_filter` (CuPy import and HIP initialisation), then starts its first job: kernel compile/load, first allocations, first launch. The only per-worker timeline on record (S163KARG §3.3) shows handshake → init +7 s → import done +13 s → first job +2 s, so +3–9 s falls in HIP initialisation or the first job | **Almost certainly not.** Cleanup runs after a job completes, and only with `S163_MEM_DEBUG=1` from 04-08 [inference; per-event job timing not recorded] |
| ZMQ / SSH-PWC (04-01 → 04-05) | per job or chunk | each job: import, compile, alloc, launch, then torch cleanup per chunk | Depends on the chunk; per event UNKNOWN |

**Recorded pattern (not a cause):** in both engines the onset sits at **the first launch of a
kernel module newly loaded in that process, with peer processes doing the same at the same
moment**. In the PWC launch-window events that is before any torch call in the process
[inference]. In today's miner event it comes after hundreds of torch calls. This weighs against
"a torch call inside the window" as the trigger. It leaves open "a second runtime that is
initialised and repeatedly used in a long-lived process", and it leaves open causes that have
nothing to do with torch. **No conclusion is drawn; the lead stays OPEN.**

### 1.5 First raw GCVM

`logs/rrig6600c_netconsole.log:9`, **2026-04-01 20:06:49**, rrig6600c `0000:19:00.0`, status
`0xFFFFFFFF`. It arrived 109 s after that capture's first NETCONSOLE TEST line (20:05:00), and
it is preceded by "16 callbacks suppressed". The date marks the start of capture, not of the
fault.

---

## 2. Escalation ladder (owner additions (b), (c))

**Plain statement:** a clean single-card or single-rig result is a **non-reproduction**. It says
nothing about the lead. The record already shows one rig alone running clean (S162 §1) and today's
faults came with all 25 workers entering phase 4 together.

**Phase-4 work is the real production stripe at real size and survivor volume, at every rung**
(owner addition (a)):
- The plan carries the failing run's own stripe for that worker, built with the production
  payload builder, and the full sub-stripe list from production `partition_stripe` at production
  VramCaps. For gpu6: `st3_s20`, seeds 3,489,660,928 + 2²⁶, 68 sub-stripes of 1,000,000, last
  108,864; `residue_sha256` `c761272d…` = trial context.
- The production `SieveExecutor` executes it. Nothing is shrunk (`scripts/rig_seam_probe.py`
  docstring; `scripts/rig_seam_controller.py plan`).

| Rung | Scope | Arm | Budget | Stop rule |
|---|---|---|---|---|
| R1 | rrig6600c gpu6 `16:00.0`, one card; probe | P | B = 8 attempts | **Any F → stop**, owner recovery, then the arm comparison runs at R1. **0 F in all fully observed attempts → non-reproduction; climb to R2.** An I attempt is re-planned, never retried silently |
| R2 | rrig6600c, all 8 cards entering phase 4 together; probe | P | B = 8 | As R1 |
| R3 | **all three rigs plus Zeus entering phase 4 together, as in production** | production worker | per §4 | As R1; this is the rung where the fault is already recorded (2 in 6 entries, 1 in 2 today) |

Arms other than P run **only at the lowest rung where P reproduces** (owner addition (d)). R1/R2
are cheap screens that might make the decisive comparison cheaper. They cannot replace it.

---

## 3. The torch arm set (same card, same hand-over, differing ONLY in torch)

| Arm | Per sub-stripe | Once per phase | torch in the process |
|---|---|---|---|
| **P** | production cleanup: gc, `import torch`, `is_available`, `synchronize`, `empty_cache`, both CuPy pool frees (logic pinned to production by source sha `d756acc3…`; the probe refuses P if production changed) | — | mapped and called ~per sub-stripe |
| **P1** | gc + both CuPy frees; `import torch` on the first cleanup, no call | `is_available`, `synchronize`, `empty_cache` once at the end of each phase | mapped, called once per phase |
| **M** | gc + both CuPy frees; `import torch` on the first cleanup, never called | — | mapped, never initialised by torch |
| **N** | gc + both CuPy frees | — | never imported (meta-path blocker; self-checked by `find_spec`, recorded as `torch_blocked`) |

**What each contrast shows.** Read only at a rung where P reproduces. "Fewer" means fewer
in-window E1 over equal budgets of interleaved attempts, under the T4.0 statistics.

| Contrast | Fewer E1 in the second arm supports | Equal E1 shows | Never shows |
|---|---|---|---|
| P vs P1 | the **frequency** of torch's cleanup calls inside a long-lived process matters | per-sub-stripe frequency is not needed to reproduce at this rate | that the torch runtime, rather than its blocking synchronise, is involved |
| P1 vs M | torch's **calls** (its initialisation + sync + `empty_cache`, a bundle) matter at all | the calls are not needed at this rate | which of init, sync and `empty_cache` matters |
| M vs N | the second runtime's **mapping alone** matters (load-time effects, e.g. 41165dd's include-path effect) | mapping alone is not needed | that mapping is sufficient |
| P vs N | the whole torch presence matters (the decisive fleet contrast, §4) | torch is not needed at this rate | a mechanism inside torch |

**A clean budget in any arm** is "no fault in n fully observed attempts", with its upper bound.
It is never "this arm prevents".

**Trace needed to say which runtime initialised the device** (in the probe now):
- **Differential identity snapshots.** Mapped `libamdhip64` / `libhsa-runtime64` / `libhsakmt`
  paths with their origin (system vs torch wheel), plus the open `/dev/kfd` and
  `/dev/dri/*` handles. They are taken before and after `import cupy`, `detect_gpu`,
  `warm_gpu`, `import torch`, `torch.cuda.is_available()` and `synchronize`/`empty_cache`.
  - Each libhsakmt instance opens `/dev/kfd` itself, so **a new `/dev/kfd` handle appearing
    inside a torch call attributes that open to torch's runtime** [inference from the library
    structure; it is to be confirmed in the first real attempt].
  - A handle does not say which runtime later *uses* the device. Per Beta R3 Q14, use beyond
    initialisation stays **UNOBSERVED** unless attributed by the bound implementation.
- **Documented debug logs (diagnostic variant only; they change timing):**
  - `AMD_LOG_LEVEL` (0–5) and `AMD_LOG_MASK` for HIP/ROCclr;
  - `HSAKMT_DEBUG_LEVEL` (3–7 in the 5.x guides; 3–4 in 7.2.4) for libhsakmt.
  - Both runtime copies read the same variables, so a log line is attributed to a copy only if
    its content or its timing between differential snapshots places it. Otherwise it is
    corroborative only.
  - The controller passes them with `--extra-env` on separate, labelled attempts.

### 3.1 Every torch import or call on the worker paths (size of the real fix)

**RANGE-MINER worker** (what runs on the rigs now):

| Site | What |
|---|---|
| `miner/range_miner_worker.py:1059-1066` | `import torch`; `torch.cuda.is_available()`; `synchronize()`; `empty_cache()` inside `_best_effort_gpu_cleanup` (`:1051-1072`) |
| `miner/range_miner_worker.py:1331` | the only caller: `SieveExecutor.execute`'s `finally`, after every sub-stripe |
| imports reached | `prng_registry` (`:634`): torch only via lazy `_get_torch()` (`prng_registry.py:84-89`), whose callers (`get_pytorch_gpu_reference` etc.) are used only by `survivor_scorer.py` and a test, never by the miner. `sieve_gpu_worker` (`:1150`, for `_get_kernel`): torch only inside its own cleanup (`:84-87`), which the miner does not call. `hybrid_strategy`, `adaptive_thresholds`: no torch |

**Size of a no-torch miner worker:** delete one 8-line `try` block (`:1059-1066`). Nothing else
on the miner path imports or calls torch. The behaviour change is that the per-sub-stripe
`torch.cuda.synchronize()` and `empty_cache()` disappear. `.get()` at `:1270` is already
host-blocking for CuPy's own work.

**PWC workers:**

| Path | Site | Gating |
|---|---|---|
| TCP-PWC `persistent/pwc_worker_service.py:462-463` → `sieve_filter._best_effort_gpu_cleanup` | `sieve_filter.py:139-145` | only when `S163_MEM_DEBUG=1` (`:139`) |
| SSH-PWC `sieve_gpu_worker.py:348` → `:78-95` | `:84-87` | unconditional, per job |
| reverse script path | `reverse_sieve_filter.py:73-76` | unconditional |
| step-2/3 and script workers | `distributed_worker.py`, `scorer_trial_worker.py`, `survivor_scorer.py:47-49, :498, :572` (register `:3084-3085`) | not on the sieve path |

---

## 4. The decisive comparison: production worker vs no-torch worker at full fleet (owner addition (d), (e))

**Design.**
- Production Run-4 shape, owner-launched, at R3: all three rigs plus Zeus entering phase 4
  together.
- **P = today's worker. N = the same worker with the torch block disabled.**
- Interleaved P, N, P, N … inside one host boot where possible; strata are recorded.
- Recorder in passive mode, netconsole armed and proven before and after each entry.

**How N is produced.** Preferred: a **one-flag worker change**, environment
`RANGE_MINER_CLEANUP_TORCH = per_substripe (default, = today) | per_phase | import_only | off`,
read once at worker start, giving P / P1 / M / N. It is **design only, not built**: it is worker
code, so BEFORE-COMMIT review. The fallback is the fleet probe (25 probe processes) with P vs N;
it lacks the coordinator, network and staging, so the production form is preferred.

**Cycle budget**, exact power, interleaved, one-sided Fisher α = 0.05, 80 % power, computed with
`scripts/rig_seam_controller.py` functions:

| P rate per full-fleet phase-4 entry | Source | N rate | Entries per arm | Power at 8 / 12 / 20 per arm |
|---|---|---|---|---|
| 1/2 | today, 1 in 2 (`57dc50fa` clean, `ab6318d4` GCVM) | 0 | **10** | 0.64 / 0.93 / 0.99 |
| 1/2 | | 1/6 (two-thirds reduction) | 29 | 0.21 / 0.43 / 0.64 |
| 1/3 | miner record, 2 in 6 (register Table 1: D-24, D-17, D-20, D-08 CLEAN; D-25 GCVM; 08-22 run 3 UNKNOWN host loss) | 0 | **19** | 0.26 / 0.61 / 0.85 |
| 1/3 | | 1/9 | 49 | 0.11 / 0.23 / 0.39 |
| 1/6 | GCVM-class only, 1 in 6 (the UNKNOWN host loss excluded) | 0 | 39 | 0.03 / 0.13 / 0.23 |

- **Measured:** the counts. **Assumed:** a stationary per-entry rate, and independence across
  entries and boots.
- The PWC-era per-run GCVM rates (Part C: 30 GCVM events beside 19 clean rows) come from a
  different engine with different launch counts. They are not pooled into this budget.
- **Recommendation:** a fixed budget of **20 entries per arm (40 total)**. It has power 0.85 if
  N removes the fault at today's record rate of 1/3, and 0.99 at 1/2. It is reported as F/C/I
  per arm with ranges and bounds, never as "prevents".

**Duration per full-fleet phase-4 entry (shortest honest route):**
- **Clean entry ≈ 16 min:**
  - fleet launch (25 workers, 3 s stagger) ~2 min;
  - trial start to phase-4 entry **~8 min** (13:13:37 → 13:21:29 today);
  - phase 4 ~4–5 min;
  - teardown plus NCPROOF ~2 min.
- **Faulted entry ≈ 40 min:** a hung host, outlet power-cycle, boot Proxmox, re-arm (today
  13:21 → 13:55).
- **Expected total for 20 + 20:**
  - P rate 1/3, N 0: about 20 × 16 + 7 × 24 + 20 × 16 ≈ **13.5 h**;
  - P rate 1/2: about 15 h;
  - spread over sessions, owner present at each fault.

**Can phases 1–3 be shortened without changing the worker's module and memory history? No, not
for this comparison.**
- **Module history is preserved** by any geometry, because every family loads once.
- **The cleanup history is not.** The number of cleanup calls before phase 4 equals the number
  of sub-stripes in phases 1–3: per stripe 34 + 34 + 68 = **136** calls, each with torch's sync
  and `empty_cache` in P. That count is **the variable under test** (§3: P vs P1).
- The ~200 s idle before phase 4 is part of the seam.
- So the route is the production geometry. The only time that can be cut is launch overhead
  outside the GPU history (pre-run gates), and that is the owner's call.

---

## 5. Probe, stop controller, recorder (built; off-GPU tested only)

| File | sha256 | What |
|---|---|---|
| `scripts/rig_seam_probe.py` | see the R1 request | One attempt per process. Real backend = production executor, `detect_gpu`/`warm_gpu`, BDF check before load, arm cleanup at the module global `:1331` calls, traced `_gpu_launch` and `_get_kernel`. Fake backend: stub CuPy pools and **stub torch** (fake mode never imports the real torch) |
| `scripts/rig_seam_probe.cleanup.sha256` | — | pins production `_best_effort_gpu_cleanup` (`:1051-1072`) |
| `scripts/rig_seam_controller.py` | see the R1 request | `plan` (production payload builder, dataset sha checked), `run` (one attempt), `summary`. One VM101 clock. Withholding, stop, then TERM/KILL escalation, unresolved list. No retry; refuses the next attempt until the owner clears a recovery-needing one. F/FW/C/I classes, hand-over window, E1/E2/E3 |
| `scripts/rig_gpu_recorder.py` | see the R1 request | adds `--mode passive` (no SMU-backed reads; reason: rrig6600b `19:00.0` SMU `TransferTableSmu2Dram` failure at 13:47:28) and a **lifetime flock** (Beta R3 Q5) |
| `tests/test_rig_seam_probe_controller.py` | see the R1 request | 12 gates + 9 mutants |

**Off-GPU results (VM101, CPU only, localhost, fake probes):**
- `python3 -B -m pytest -q tests/test_rig_seam_probe_controller.py` → **12 passed**.
- Mutants: **9 of 9 DETECTED**, each with a passing clean control:
  - M1 invalid counted clean;
  - M2 no withholding;
  - M3 window ignored;
  - M4 deadline ignored;
  - M5 bound over B;
  - M6 N without blocker (after a fix: it first SURVIVED, the gate gap was fixed with the
    `torch_blocked` self-check);
  - M7 P without torch sync (first INVALID: a non-unique pattern, fixed);
  - M8 next attempt not refused;
  - M9 P1 torch per sub-stripe.
- **Plan built (CPU only)** from the bundle ledger and the live `daily3.json`, whose sha matches
  the trial: `logs/rig_seam/plans/rrig6600c_gpu6.json`.
- **Recorder:** passive mode on the idle hosts, 18 OBSERVED with no sensor fields. Three
  simultaneous starts gave one run and two refused. No remote sampler was left.

**Not verified (needs the owner's first real attempt):**
- The real backend has never run.
- `getDeviceProperties` keys under HIP (`pciBusID`, `pciDeviceID`, `pciDomainID`) are
  **[UNVERIFIED]**.
- The CT repo must be at the same commit as the plan's payload contract (checked by the owner's
  pre-flight).
- `/dev/kfd` handle attribution is not yet confirmed in a live process.

---

## 6. First test (R1, arm P), for the owner

**What it is.** The probe at R1 on rrig6600c gpu6 (`16:00.0`), arm P, eight attempts. Each
attempt is:
1. bring-up;
2. the full phase-3 stripe `st2_s20` (68 sub-stripes);
3. 200 s idle;
4. the full phase-4 stripe `st3_s20` (68 sub-stripes);
5. the 30 s window;
6. exit and NCPROOF.

**Duration:** about 6–7 min per attempt, plus the owner's NCPROOF. Eight attempts take
**≈ 1 h** if clean.

**Exact commands and owner preparation** are in the stop message and in the R1 request §Commands.
