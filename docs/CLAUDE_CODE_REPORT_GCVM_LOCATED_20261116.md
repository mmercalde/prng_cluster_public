# CLAUDE CODE REPORT — GCVM fault located: fault addresses against the host's recorded memory picture

**Revision R2 (2026-11-16 ~17:45 PST).** Beta's three LIVE objections in X-GCVM-LOCATED R1 are accepted in full by
Alpha and applied:
- **host removal, eviction or pressure is excluded** for the recorded events;
- **a host translation or visibility error on a real page is NOT excluded**;
- the 12 MiB read is stated as "not an in-bounds normal access";
- **H3 is corrected**: kernel arguments on gfx1032 are host memory by default.

The verdict labels are narrowed per Q2, and the mapper is fixed per Q1/Q6 (`scripts/rig_fault_map.py`; output
`fault_map_v2.txt`).

**Direction:** Alpha (owner-delegated tests), 2026-11-16, 17:0x PST.
**Basis:** offline analysis of the outside recorder's capture, two runs on one boot:
- Run A launched 16:23:05 and credited clean;
- Run B launched 16:46:41 and faulted at 16:56:20.382.

Nothing ran on a GPU and nothing changed on a host for this analysis.

**Evidence (VM101):** `logs/rig_fault_rec/20261116_AB_secured_165728/` (`SHA256SUMS.txt`): trace stream, Run A T0–T7 and
Run B T0–T5+FAULT snapshots, netconsole copy, `fault_map.txt`. Tools: `scripts/rig_fault_recorder.py`,
`scripts/rig_fault_map.py`.

**This is a reading of driver events.** The trace records what the host driver mapped and wrote into the
page tables (`amdgpu_vm_bo_map/unmap`, `amdgpu_vm_update_ptes`, KFD map/evict/restore, `amdgpu_bo_create`).
It does **not** show the PTE bytes in VRAM, the card's TLB, or the instructions and kernel arguments the card
actually read. No root cause is claimed.

## Prior record

**Searched** (`/bin/grep -ciE`):
- the register `…/gpu-fault-history/docs/REGISTER_GPU_FAULT_PRIOR_WORK.md`;
- `docs/PROJECT_FILE_CATALOG.md`, `TB_RULING_*`, `PROPOSAL_*`, `CHAPTER_*`, `*_AUDIT.md`, `TODO_CONSOLIDATED.md`,
  `BACKLOG.md`, `LEADS.md`, `docs/exchange/`;
- for `kernarg`, `HDP`, `code object|InvalidateCodeCaches`, `stale TLB|TLB|never mapped`,
  `vm_info|tracepoint|ftrace|tracefs`.

**Hits:**
- register E28 (`:1089`): "a stale TLB, or CuPy pool recycling, causes the SQC (inst) faults". The no-pool
  test "made things worse". **ALREADY RECORDED** as a belief; never measured.
- `docs/LEADS.md` L-6/L-7; `docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT_DECODE_20261115.md` §2 (code object in VRAM
  per ROCr 6.4.3 source); `docs/CLAUDE_CODE_RUNBOOK_RIG_FAULT_RECORDER_AB.md`.
- **No hit** for kernarg, HDP, or any tracepoint or debugfs capture.

**Labels:**
- everything measured below: **NEW**;
- the L-7 run-#2 pattern: ALREADY RECORDED; Run B is its third same-boot repeat (run #2 → fault);
- no ruling changed.

**Not searched:** ser8, the private repository.

## 1. Verdict per card (Run B)

| Card (worker) | First fault | Covered by any map or PTE update, any process, whole run? | In the process's CPU maps (T5)? | Verdict |
|---|---|---|---|---|
| rrig6600 `19:00.0` (gpu7, pid 212162) | TCP **write** `0x77e7d5826000`, then `0x77e7d17e6000`; `0x00841051` | no; nothing this process mapped within 256 MB | no | **No covering mapping/PTE event in the retained capture** |
| rrig6600c `16:00.0` (gpu6, pid 211495) | TCP **read** `0x76a2cd9fb000`; `0x00801031` | no. It is 4 MiB past the end of the phase-4 **seeds** buffer (8 MiB, base `0x76a2cce00000`, mapped 18 ms after dispatch; identified as seeds by creation order and size) + 11.98 MiB | yes: an anonymous 8 MiB CPU mapping `76a2cd800000–76a2ce000000` (host memory, with no GPU mapping in the capture) | **No covering mapping/PTE event in the retained capture** |
| same card, +50 ms | SQC (inst) `0x76a3e0662000`; `0x00801231` | **yes**: inside the phase-3 code's **system-memory staging block** `0x76a3e0660000`–`0664` (5 pages, R/W, **not executable**, mapped at phase 3, never unmapped) | yes (`76a3e0660000-76a3e0665000 rw-p`: the CPU view of the staging copy) | **Recorded R/W, non-executable mapping, with no recorded subsequent revocation.** An instruction request into a recorded data range, consistent with PERMISSION 0x3 and MAPPING_ERROR 0. It came 49.5 ms after the card's TCP read, so it is **not necessarily the initiating fault**. Its PTEs point at four scattered system-memory pages (`dst 0x1e2ab8000, 0x1e21de000, 0x1e577b000, 0x1e2bc9000`) |
| rrig6600c `06:00.0` (gpu1, pid 211254) | first printed line already `0xFFFFFFFF`; IV address `0x76ab421aa000` | no; nothing within 256 MB | no | **No covering mapping/PTE event in the retained capture** (low confidence: the card was not answering when its first line printed; not a reliable initiating access) |

**Not observed before any fault, on any host, in either run:**
- any KFD evict or restore event;
- any unmap in the last 2 s;
- any map or PTE update that failed (every `kfd_map_memory_to_gpu_end` reads `Success`).

**The fault addresses are not inside any other process's mapping either.**

**Scope (Beta R1 Q3, accepted).** These are recorded driver events. What they establish:
- **Excluded for the recorded events:** host *removal*, *eviction* or *memory pressure* acting on these pages or
  on the phase-4 buffers. There is no unmap, no evict/restore, no failed map, no reclaim or compaction, and no
  swap.
- **Not excluded:** a host *translation or visibility* error on a **real** page (code, kernel arguments,
  descriptor, data) that corrupts what the card consumes and then produces a wild access elsewhere. The final
  fault address need not be the page that was wrong.
- **Not shown:** PTE bytes as the card saw them, TLB state, or the instructions and arguments executed. "No
  ring-buffer drop recorded" is bounded: per-CPU loss counters were not captured, and a tail that never left a
  freezing host is unobservable.
- **Mechanics the trace does show:**
  - The `amdgpu_vm_flush` events (17,254 on `.163`) are all `ring=sdma0, id=0`: kernel VMID 0 jobs, not worker
    VMs. For the worker VMs, the PTE writes carry their target physical addresses (the `dst` lines), but
    neither the KFD TLB invalidations nor the HDP flush is traced. That visibility step is part of what
    remains open.
  - That compute VMs are CPU-updated on large-BAR cards (the upstream default for `vm_update_mode=-1` with
    fully visible VRAM) is **[UNVERIFIED]** for the installed DKMS.

## 2. Phase-4 allocation, Run A against Run B

**Sizes and kinds.** The first sub-stripe is **1,000,000 seeds** (`--seed-cap-amd-hybrid` default 1,000,000,
`miner/range_miner_worker.py:655, :2948`; `partition_stripe :680` → 68 sub-stripes per 2²⁶ stripe). Creation
order is `SieveExecutor.execute :1329-1360`. Kernel signature and indexing: `prng_registry.py:3170-3240`.

| # | Buffer (code) | Requested for n = 1e6, k = 12 | Mapped BO | Same in A, B, healthy, rrig6600b? |
|---|---|---|---|---|
| 1 | `seeds` = `cp.arange(n, uint64)` | 8,000,000 B | 2048 pages (8 MiB) | yes |
| 2 | `survivors` uint64 | 8,000,000 B | 8 MiB | yes |
| 3 | `match_rates` float32 | 4,000,000 B | 4 MiB | yes |
| 4 | `strategy_ids` uint32 | 4,000,000 B | 4 MiB | yes |
| 5 | `skip_sequences` uint32, n·k | 48,000,000 B | 11776 pages (46 MiB) | yes |
| — | `residues` (48 B), `survivor_count` (4 B), strategy arrays (n_strategies·4 B) | small | no new BO in the phase-4 window | yes |
| C1 | code, executable | — | 5 pages, VRAM (type2 pref4), flags **0x70** (R/W/X) | yes, 5 pages |
| C2 | code staging | — | 5 pages, system memory (type0 pref1, moved to placement 6), flags **0x60** (R/W) | yes, 5 pages |

Rows checked: the three faulting workers, rrig6600c gpu0 (healthy, same rig) and rrig6600b gpu6, in Run A and
Run B.

- **The Run-A buffer matching B's faulted 8 MiB seeds buffer was also 8 MiB.** No phase-4 buffer differs in
  number or size between A and B, between faulting and healthy cards, or against rrig6600b.
- **Code-pair layout is identical in both runs.** Each phase's pair (C2 directly below C1) is allocated just
  below the previous phase's pair, and both stay mapped until the process exits:
  - phase 3: C1 at +0x10, C2 at +0x08 pages;
  - phase 4: C1 at +0x08, C2 at +0x00;
  - healthy cards show the same layout.
- **Phases 1–2** use smaller code blocks (< 5 pages; not listed). At start-up an 8-page executable GTT block
  is mapped once into all 8 of the process's VMs.

## 3. Arguments, stripe and offsets

**Stripes:**

| Worker | Run A | Run B |
|---|---|---|
| rrig6600c:gpu6 | st3_s23 | **st3_s23** (fault) |
| rrig6600c:gpu1 | st3_s21 | st3_s16 (fault) |
| rrig6600:gpu7 | st3_s3 | st3_s5 (fault) |

All at sub-stripe 0, n = 1,000,000, k = 12, generator_phase 0 (offset argument 0), threshold 0.47.

- **rrig6600c gpu6 ran the same stripe, sub-stripe and arguments in both runs: clean in A, fault in B.**
- **The read 11.98 MiB above the seeds base is not an in-bounds normal access** (Beta R1 Q5, accepted).
  - If it were an eight-byte seeds load, it would be index 1,570,304–1,570,815. But 3907 blocks × 256
    threads cap the normal thread index at 1,000,191 **even without the `n_candidates` guard**, so a wrong
    `n_candidates` alone cannot produce it.
  - A page near the seeds BO is not evidence that the operand was `candidate_seeds`. The kernel also reads
    residues, strategy arrays and private (scratch) storage, and CuPy kernels run in the same window.
  - Neither the operand nor the issuing kernel is determined.
- **The 19:00.0 writes** lie 1.7 GB and more above the process's lowest mapping. They match no argument or
  buffer base.
- **The SQC fetch** is at phase-4 C1 + 10 pages (code is 5 pages), and phase-3 C2 + 2 pages: inside the host
  copy of the *previous* phase's code.
- **Which kernel issued the TCP accesses is not determined.** `cp.arange` (seeds) and the `cp.zeros` fills
  launch CuPy kernels from modules loaded in earlier phases at +18–20 ms, before the sieve kernel's code load
  at +21.6 ms.

## 4. How the code reaches the card

- The trace shows, in this order: C1 created in VRAM and mapped executable, then C2 created in system memory
  and mapped R/W, all within 0.6 ms (B) or 0.05 ms (A).
- Then two completion interrupts on the same queue (`src_id 181`, `src_data 0x1b` then `0x03`) follow 0.1–0.9 ms
  later, in both runs.
- There is **no event for the copy itself**. ROCr's `Freeze()` either does a CPU memcpy over the large BAR or
  a DMA on a user-mode queue. Neither emits a tracepoint, so the method is **not observable** here. Large BAR
  is on for all 24 cards, and 20 KiB is likely under `co_dmacopy_size` [default UNVERIFIED].
- The instruction-cache invalidate (`InvalidateCodeCaches`) is also a user-queue packet with no tracepoint.
  The two interrupts *may* be its completion [inference].

## 5. What differs between Run A and Run B for these workers before the fault

| | Run A (first run after boot; cold cache, compile) | Run B (second run; warm cache) |
|---|---|---|
| Phase-4 buffers mapped | dispatch + 13.6–24.1 ms | + 13.2–30.3 ms |
| Code pair mapped | **+585–629 ms** (after hiprtc compile) | **+15.4–30.9 ms** (cache hit) |
| Buffers → code | ≈ 570 ms | ≈ 1–2 ms |
| Code map → next event | IVs +0.1/+0.3 ms, then more IVs, first sub-stripe done ≈ +534 ms | IVs +0.7/+0.9 ms, **fault ≈ +24 ms** |
| VMID / PASID | vmid 8; PASID 32776 + 9·gpu (gpu6 32830 in both) | same |
| Addresses | different bases (ASLR), same relative layout | same |
| Sizes, order of creation | identical | identical |
| Carried driver state (from the Run-A report) | none at T0 | `amdgpu_sa_info` 1,435–1,648 entries per card and VRAM free-list fragmentation, on **all** cards including rrig6600b |

## 6. Hypotheses left, each with the one observation that kills it

| | Hypothesis | Killing observation |
|---|---|---|
| H1 | **Host translation or visibility error on a real page** (PTEs written but not yet visible, or a TLB not invalidated, for code, kernarg, descriptor or data pages). Host removal/eviction/pressure is excluded for the recorded events | Open. Bounded tests only: record the KFD TLB-flush and HDP-flush completions relative to the first dispatch. A PTE read-back tests stored translations at one instant, not hardware-visible translation throughout |
| H2 | **Program contents on the card wrong or stale at the first launch** (code not yet visible after the BAR copy, or instruction cache not invalidated). This would explain wild data addresses and a jump into a data page | Read the C1 block back from the card just before the first launch and compare it byte-for-byte with the cached `.hsaco`'s loaded segment. **Equal and still faults → killed.** Possible routes: the worker process's CPU view of C1 via host `/proc/PID/mem` (ROCr CPU-maps C1 for the large-BAR memcpy; TTM's `access` handler reads VRAM), or `amdgpu_vram` debugfs at the BO's physical offset (not in the trace: `vm_update_ptes` `dst` is empty). Both are reads of card memory and need a go |
| H3 | **Kernel arguments wrong or stale on a warm start.** **Corrected premise:** on gfx1032 the kernel arguments are in **host (system) memory** by default. In ROCclr 6.4.3 `rocsettings.cpp` the constructor sets `kernel_arg_impl_ = HostKernelArgs` ("coarse grain system memory … by default"); device placement is copied in only `if (isGfx94x)` or `if (!flagIsDefault(HIP_FORCE_DEV_KERNARG))`. The worker environment has no `HIP_FORCE_DEV_KERNARG` (T5 snapshot), so `=0` would change nothing | Bounded test: record the kernarg bytes the dispatch actually used and compare A vs B. `HIP_FORCE_DEV_KERNARG=1` is a *treatment* that moves kernargs to device memory; recurrence under it excludes host placement as a necessary condition, nothing more |
| H4 | Buffer sizes, number or layout differ on a warm start | **Killed** by §2 |
| H5 | Card-specific defect, triggered by the fast load→launch sequence | Permute card ↔ worker identity; the fault follows the identity or launch position instead of the card → killed |
| H6 | Driver state left by Run A (suballocator growth, VRAM fragmentation) is the trigger | Run B with the kernel cache **emptied** (same boot, carried state kept, cold compile): it faults → H2/H3 timing explanations weaken and H6 or H5 strengthen; it is clean → H6 killed. rrig6600b already carries the same state without faulting |

## 7. Recorder status

- `auto` for both runs stopped by PID.
- Tracing is armed on `.155` only. `.121` and `.163` rebooted (not re-armed, as instructed).
- The stream reader (pid 8181) was stopped before the Beta relay: the relay boundary forbids writes under
  `logs/` while it runs. Restarting it needs only `stream` again.
- Run B's FAULT snapshot holds `.155` only; `.121` and `.163` timed out (frozen).

## 8. Load-to-launch gaps, all 24 rig workers (R2, Alpha's request)

Source: `scripts/rig_fault_map.py` ownership rules; host trace clock (`mono`) for gaps within a host. Kernel
**launch is not traced** (user-mode AQL queue). The nearest observable is the **first completion interrupt**
(`amdgpu_iv`, the worker's PASID) after the code pair's last `kfd_map_memory_to_gpu_end`. Tables:
`/tmp/…/scratchpad/gaps.json`, to be copied into the evidence freeze.

| Run / phase | buffer map → code map | code map → code-pair map done | pair done → first IV | PTE updates ≤ 2 ms after first IV | fault cards' rank (buf→code, of 24) |
|---|---|---|---|---|---|
| B ph1 (warm) | 0.77–6.84 ms | 0.06–0.12 ms (one 9.1) | 0.13–0.30 ms | 0–9 (all workers) | 2, 17, 22 |
| B ph2 (warm) | 0.72–0.99 ms | 0.05–0.10 ms (one 0.96) | 0.11–0.21 ms | 0 | 10, 4, 15 |
| B ph3 (warm) | 0.76–7.13 ms | 0.06–0.09 ms (one 6.1) | 0.02–3.19 ms | 0, or 11–39 (scratch mapped at first dispatch) | 13, 7, 20 |
| **B ph4 (warm; FAULT)** | **0.705–0.904 ms** | 0.05–0.10 ms | 0.09–0.21 ms | **0** | **2 (gpu7), 9 (c:gpu1), 17 (c:gpu6)** |
| A ph4 (cold compile; clean) | **278.9–670.9 ms** | 0.07–0.13 ms | 0.15–0.23 ms | 0 | 5, 10, 17 |

**Findings:**
- **The faulting cards did not have the shortest gaps**, and showed no ordering difference. In B ph4 no
  worker had a PTE update for its code or argument pages after its first completion interrupt. Code-pair
  mapping completed 0.05–0.10 ms after the code map, on faulting and healthy cards alike.
- **Phases 1–3 of B had the same ~1 ms fast path, with no fault.**
- **What is different about phase 4**, observed and equal on all 24 workers, so not a discriminator by
  itself:
  - (a) It follows the **longest GPU idle**: 219–237 s against 22–100 s before phases 1–3. Run A ph4 had the
    same idle.
  - (b) It is the first phase whose kernel **reuses scratch** allocated by the previous phase (phase 3 maps
    449 MB, or 513 MB on rrig6600, at its first dispatch; phase 4 maps none).
  - (c) It stacks a second code pair below phase 3's.
  - In Run A, (a)–(c) also held, but the sieve kernel ran ~280–670 ms after the arange/fill kernels had woken
    the card. In Run B it ran ~1 ms after them.
- **Physical placement is not a discriminator.** The phase-4 code pages (PTE `dst`) of rrig6600c gpu6
  (faulting) are the same card-local addresses as rrig6600b gpu6/gpu7 and rrig6600c gpu5 (healthy):
  `0x1fec64000–67000` + `0x1fec6f000`.
- The fault-time rows: rrig6600 gpu7 fault rx 16.0 ms after its code map; rrig6600c gpu6 25.7 ms; rrig6600c
  gpu1 224.9 ms, where the first *printed* line is late with a dead status.

## 9. Run C (V2): first run after boot, warm caches pre-filled. FAULT

**Run:**
- Launched 17:46:06; phase-4 dispatch 17:55:48.651; first fault line received 17:55:48.692.
- rrig6600 and rrig6600c: run #1 of the boot (boots 17:01), caches copied from rrig6600b.
- rrig6600b: run #3 of its boot. Clean.
- Evidence: `logs/rig_fault_rec/20261116_C_secured_175651/` (`MANIFEST.sha256`, `fault_map.txt`,
  coordinator log `gate12_20261116_174606.log`).
- Both faulting hosts froze. The FAULT snapshot holds `.155` only.

**Cards:**
- rrig6600c `16:00.0` (gpu6): faulted again.
- **rrig6600c `19:00.0` (gpu7): new.**
- rrig6600 `19:00.0` (gpu7): faulted again.
- **rrig6600c `06:00.0` (gpu1), which faulted in B, printed no lines at all in Run C.**

| Card (worker, pid) | First fault | Verdict (retained capture) |
|---|---|---|
| rrig6600c `16:00.0` (gpu6, 19146) | TCP write `0x738d53600000`, `0x00841051` | no covering mapping/PTE event; no CPU mapping |
| rrig6600c `19:00.0` (gpu7, 19196) | TCP write `0x77f115470000`, `0x00841051` | no covering mapping/PTE event; no CPU mapping |
| rrig6600 `19:00.0` (gpu7, 15979) | first printed status `0xFFFFFFFF`; IV address `0x7ad60665f000` | no covering mapping/PTE event (low confidence) |
| rrig6600c `16:00.0`, second fault (+46 ms) | TCP read `0x738d72000000`, `0x00801011` | **recorded mapping, then a recorded unmap of it.** The page lies inside a 16 MiB buffer (`0x738d71c00000`, 4096 pages, flags 0x60) that was mapped and unmapped 66× in phase 1 and 33× in phase 2; the last unmap was 17:49:35 (phase 2). Fault at base + 4 MiB. No phase-4 BO covers it (the nearest begins 10 MiB higher) |

- **First data access to a formerly valid, since-freed buffer address.** It is a second fault, so not
  necessarily the initiating one.
- It is consistent with the card using an **earlier phase's address**. One example would be a stale argument
  or code reaching the card; other mechanisms are possible (H1, H2, H3).

**Load-to-launch gaps, phase 4** (23 of 24 rig workers; rrig6600c gpu0 was assigned st3_s23 but recorded no
phase-4 activity before its host froze):
- buffer→code 0.695–1.027 ms (outliers 4.06 rrig6600c gpu1, 6.86 rrig6600b gpu7);
- dispatch→code 11–30 ms; code→pair-done 0.04–0.10 ms; pair-done→first IV 0.10–0.18 ms;
- no PTE update within 2 ms after the first IV.

**The pre-filled caches did give the warm fast path on `.122`/`.164`.** Phase 1 had already loaded in the
~1 ms pattern, so nothing was compiled.

Faulting cards: buffer→code 0.724, 0.730, 0.756 ms, ranks 4, 7, 9 of 23. Fault receipt after the code map:
17.1 ms (c:gpu6), 19.9 ms (c:gpu7), 121 ms (rrig6600 gpu7, dead status).

**Against A and B:**

| | Run A ph4 (cold, run #1) | Run B ph4 (warm, run #2) | Run C ph4 (warm, run #1) |
|---|---|---|---|
| buffer→code | 279–671 ms | 0.705–0.904 ms | 0.695–1.027 ms |
| Outcome | clean | fault: rrig6600 19, rrig6600c 16, 06 | fault: rrig6600 19, rrig6600c 16, 19 |
| Idle before phase 4 | 227–236 s | 219–237 s | (not computed) |

**Reading** (single events per arm): on these two hosts every warm phase-4 fast path has faulted (B, C, and
11-15's two warm runs), and every cold one was clean (A and 11-15's two). The earlier run's carried state
was absent in C, so it is not necessary. The cold-cache second run (V1, now Run E) is the remaining cell.

## 10. Runs D and E (cold), and why phases 1–3 survive a warm load

**Run D** (18:10:03; run #1 after the 18:00 boot, caches emptied) and **Run E** (18:28:47; run #2 on the same
boot, caches emptied again): **both clean**, no GCVM lines. Stream `logs/rig_fault_rec/20261116_D/stream/`;
snapshots `…/20261116_D/runD/`, `…/20261116_E/runE/`.

Run D's automatic T7 was lost to an ordering bug in `auto` (fixed). It is replaced by `T7_manual`, about
25 min after the workers exited.

**Phase-4 buffer→code, all 24 rig workers:**

| | A (cold, #1) | B (warm, #2) | C (warm, #1) | D (cold, #1) | E (cold, #2) |
|---|---|---|---|---|---|
| range | 279–671 ms | 0.705–0.904 ms | 0.695–1.027 ms (+4.1, 6.9) | 280–879 ms | 276–1184 ms |
| fault-history cards | 562, 576, 612 | 0.71, 0.76, 0.82 | 0.72, 0.73, 0.76 | 573, 680, 879 | 587, 701, 754 |
| outcome | clean | **fault** | **fault** | clean | clean |

**Run E really compiled in every phase.** Dispatch→code: phase 1 370–1036 ms, phase 2 281–780, phase 3
287–743, phase 4 286–1216 ms (all 24 workers). The warm runs show 11–100 ms.

**Why phases 1–3 survive a warm load and phase 4 does not.** These are measured differences, offered as
candidates, not established causes. Each is uniform across all 24 workers.

1. **Scratch.**
   - Phases 1–2 (constant-skip kernels) map no scratch.
   - Phase 3 (first hybrid kernel) maps a fresh 449 MB (rrig6600: 513 MB) scratch block **at its first
     dispatch**, about 6 ms after the code load, with PTE updates after the first completion interrupt. So
     its waves cannot start until scratch is mapped.
   - **Phase 4 maps no scratch: it reuses phase 3's**, mapped 4.5 min earlier. On a warm load it is the only
     phase whose scratch-using kernel can start executing about 1 ms after its code is mapped.
   - A cold phase 4 (A, D, E) reuses scratch too, but starts 280–1180 ms later.
2. **Idle.** Phase 4 follows the longest GPU idle: 219–237 s, against 22–100 s for phases 1–3.
3. **Layout.** Phase 4 stacks a second code pair below phase 3's. The instruction fetch in B landed in phase
   3's staging block.
4. **Stale address.** In Run C, a second fault read a since-freed phase-1/2 buffer address.

**Killing observations:**
- for (1): a warm phase 4 forced onto **fresh** scratch (e.g. scratch released at the end of phase 3) that
  still faults;
- for (2): a warm phase 4 dispatched without the long idle that still faults.

Register E29 (`HSA_NO_SCRATCH_RECLAIM=1` "made things worse") concerns scratch reclaim and should be read
again against (1).

**Draft mitigation (not landed):** `docs/drafts/DRAFT_launcher_cupy_cache_reset.diff`;
`docs/exchange/to_beta/X-CUPY-CACHE-RESET_R1_…` (not sent).

## 11. 2026-11-17: Runs F, G, H (cold, one boot); all runs compared; scratch; the installed KFD map path

**Runs F (17:00:45), G (17:20:33) and H (17:40:57)**, one boot (hosts 16:43–16:48), caches emptied before each:
**all clean**, no page-fault line, T1–T7 + POST recorded.

**Trace loss**, per run (from cumulative per-CPU stats, clock `mono`): overrun 0, commit overrun 0, dropped 0
on every host for F, G and H. Events read:

| Run | rrig6600 | rrig6600b | rrig6600c |
|---|---|---|---|
| F | 185,242 | 152,500 | 144,237 |
| G | 157,215 | 153,993 | 150,394 |
| H | 157,371 | 159,090 | 146,397 |

Files: `20261117_{F,G,H}/run*/trace_loss_counters_post.txt`.

**Phase-4 buffer→code, all 24 rig workers** (`logs/rig_fault_rec/allgaps.py`, `allgaps.json`; C has 23
workers, one with no phase-4 activity before its host froze):

| Run | cache, boot position | min / median / max (ms) | rrig6600 g7, rrig6600c g6, g1, g7 (ms) | outcome |
|---|---|---|---|---|
| A | cold #1 | 278.9 / 579.2 / 670.9 | 561.5, 612.2, 576.4, 588.2 | clean |
| B | warm #2 | 0.705 / 0.792 / 0.904 | 0.713, 0.816, 0.764, 0.834 | **FAULT** (g7, c-g6, c-g1) |
| C | warm #1 | 0.695 / 0.771 / 6.855 | 0.724, 0.730, 4.064, 0.756 | **FAULT** (g7, c-g6, c-g7) |
| D | cold #1 | 280.0 / 616.4 / 879.4 | 573.1, 679.8, 568.0, 879.4 | clean |
| E | cold #2 | 275.7 / 683.4 / 1183.9 | 754.4, 701.3, 693.9, 586.6 | clean |
| F | cold #1 | 287.7 / 572.9 / 823.6 | 564.9, 679.6, 576.5, 576.8 | clean |
| G | cold #2 | 278.7 / 586.2 / 881.0 | 865.7, 738.0, 302.9, 587.9 | clean |
| H | cold #3 | 280.0 / 582.7 / 1407.4 | 566.4, 581.0, 581.6, 577.6 | clean |

On rrig6600 and rrig6600c, single events per run: **cold 8/8 clean, warm 4/4 faulted** (adding 11-15's two
cold and two warm runs).

### Scratch (Beta lead): code reading and fault placement

- **The miner and sieve code never allocate scratch.** There is no `scratch`/`private_segment` handling in
  `miner/range_miner_worker.py` or `sieve_gpu_worker.py`. ROCr sizes scratch at dispatch from the kernel's
  per-lane private segment.
- **Phase 3** (`JAVA_LCG_HYBRID_KERNEL`, `prng_registry.py:1005-1081`) keeps two per-lane arrays,
  `unsigned int best_skip_seq[2048]` (`:1020`) and `current_skip_seq[2048]` (`:1028`): ≥16 KiB per lane.
- **Phase 4** (`JAVA_LCG_HYBRID_REVERSE_KERNEL`, `:3170-3244`) keeps one, `skip_seq[2048]` (`:3196`): ≥8 KiB
  per lane.
- **Phases 1–2** (constant kernels) keep none.
- So phase 3's scratch is large enough for phase 4. That is consistent with the trace: 449 MiB per worker
  (513 MiB on rrig6600) mapped at phase 3's first dispatch, **no** new scratch mapping in phase 4.

**No Run B or Run C fault address is inside its process's scratch block.**

| Run | Card | Fault | Position relative to scratch |
|---|---|---|---|
| B | rrig6600 g7 | write `0x77e7d5826000` | 2340 MiB above the end |
| B | rrig6600 g7 | write `0x77e7d17e6000` | 2276 MiB above |
| B | rrig6600c g6 | read `0x76a2cd9fb000` | 3935 MiB below the start |
| B | rrig6600c g6 | SQC `0x76a3e0662000` | **12.4 MiB above the end** |
| B | rrig6600c g1 | `0x76ab421aa000` | 1802 MiB above |
| C | rrig6600c g6 | write `0x738d53600000` | 4535 MiB below |
| C | rrig6600c g6 | stale-buffer read `0x738d72000000` | 4045 MiB below |
| C | rrig6600c g7 | `0x77f115470000` | 786 MiB above |
| C | rrig6600 g7 | `0x7ad60665f000` | 1566 MiB above |

The scratch-reuse candidate (§10) is therefore not "an access inside scratch". It would have to act
indirectly, for example through the dispatch's scratch setup when no new allocation precedes it. That remains
a hypothesis.

### Installed driver: KFD map path and flush ordering

Source: `/usr/src/amdgpu-6.12.12-2194681.22.04` on `.155`, read-only. File sha256 prefixes:

| File | sha256 prefix |
|---|---|
| `kfd_chardev.c` | `7cd9f0c7` |
| `amdgpu_amdkfd_gpuvm.c` | `85b09294` |
| `amdgpu_vm.c` | `3633dcdb` |
| `amdgpu_vm_cpu.c` | `2b607bda` |

1. **Map** (`kfd_ioctl_map_memory_to_gpu`, `kfd_chardev.c:1286`):
   - `amdgpu_amdkfd_gpuvm_map_memory_to_gpu` per device (`:1356`);
   - then `amdgpu_amdkfd_gpuvm_sync_memory(..., true)`, which waits for the page-table update fences
     (`:1375`);
   - then `kfd_flush_tlb(peer_pdd, TLB_FLUSH_LEGACY)` per device (`:1388`);
   - **then** the `kfd_map_memory_to_gpu_end` tracepoint.

   So every recorded `…_end Success` came after the TLB flush was **issued**. Whether the flush had
   **completed** on the card is not established by this reading.
2. **The flush** (`kfd_flush_tlb`, `kfd_priv.h:1651`) calls `amdgpu_vm_flush_compute_tlb` (`amdgpu_vm.c:1632`).
   That **skips the flush entirely** when the VM's `tlb_seq` has not changed since the last KFD flush
   (`atomic64_xchg(&vm->kfd_last_flushed_seq, tlb_seq) == tlb_seq`). Otherwise it calls
   `amdgpu_gmc_flush_gpu_tlb_pasid`.
3. **PTE writes are CPU writes.** `vm_update_mode == -1` with fully visible VRAM →
   `AMDGPU_VM_USE_CPU_FOR_COMPUTE` (`amdgpu_vm.c:2759-2766`); large BAR is measured on all 24 cards. The
   commit (`amdgpu_vm_cpu_commit`, `amdgpu_vm_cpu.c`) does `mb(); amdgpu_device_flush_hdp()`. It bumps
   `tlb_seq` **only if `needs_flush`**.
4. **Unmap** (`kfd_chardev.c:1462-1492`): a TLB flush and the wait happen only if
   `kfd_flush_tlb_after_unmap()` is true. That is **only for GC 9.4.x** (`kfd_priv.h:1660-1664`), so **not for
   gfx1032**. On these cards an unmap returns, and the DMA mapping is removed, without a TLB flush. The pending
   `tlb_seq` change is flushed at the next map's `kfd_flush_tlb`.

**What this means for the hypotheses** (reading only; nothing measured):
- **(i)** A freshly mapped range whose map did not bump `tlb_seq` gets **no** TLB flush. That is safe only if
  the card never cached an invalid translation for that VA.
- **(ii)** Between an unmap and the next map, the card can hold stale translations for freed VAs. Phase 4's
  buffers reuse VAs freed in earlier phases: Run B's seeds BO reused the VA mapped 274 s earlier.
- **(iii)** Run C's second fault read a freed phase-1/2 VA **and faulted**. A stale translation would not
  fault, so for that page the TLB was not stale. The access itself used an old address, which points back to
  the arguments or code the card consumed (H2/H3), not to the TLB.
- **To test:** whether gfx10.3's UTCL2 caches invalid (faulting) translations with `noretry` in effect
  (`amdgpu.noretry=-1`, auto). Also whether warm phase 4's maps skip the flush because `tlb_seq` is unchanged.
  Both are observable, offline, by adding the `tlb_seq`/flush decision to the record (driver-side only;
  needs a go).

## 12. Host hardware notes (2026-11-17 evening; Alpha's live read, offline comparison)

**(a) CPU model does not separate the faulting rigs from rrig6600b.**
- Alpha's live read, confirmed by `/home/michael/runFGH_evidence_20261117/amdgpu_params_bios_<host>.txt`:
  - `.121` rrig6600: **i5-9400**;
  - `.155` rrig6600b: **i5-9400**;
  - `.163` rrig6600c: **i5-8400** (2.80 GHz, not the "T" part).
- The faulting rigs are rrig6600 (i5-9400) and rrig6600c (i5-8400), while rrig6600b (i5-9400) is clean, so
  the CPU model does not separate them.
- **None of this session's documents stated a CPU model for rrig6600b**
  (`/bin/grep -nE 'i5-|8400|9400'` over the `CLAUDE_CODE_*` reports, `LEADS.md`, the 11-15 and 11-16
  changelogs and the X-* requests).
- **Other repository documents conflicted with the live read.** The current-state tables were **corrected on
  2026-11-17** (inline "corrected 2026-11-17, live read"):
  - `TRIANGULATED_FUNCTIONAL_MIMICRY_VERIFIED_v1_0.md:33-34`;
  - `CHAPTER_13_IMPLEMENTATION_PROGRESS_v1_6.md:183`;
  - `SELFPLAY_ARCHITECTURE_PROPOSAL_v1_0.md:29, :170`.

  The historical session changelogs (`SESSION_CHANGELOG_20260322_S152.md`, `SESSION_CHANGELOG_20260324_S155.md`,
  which record an rrig6600c i5-8400T → i5-9400 upgrade) are **left as written**. They record what was believed
  then; today rrig6600c reads i5-8400. The statements originally found were:
  - "rig-6600b … i5-8400": `TRIANGULATED_FUNCTIONAL_MIMICRY_VERIFIED_v1_0.md:33`,
    `CHAPTER_13_IMPLEMENTATION_PROGRESS_v1_6.md:183`, `SELFPLAY_ARCHITECTURE_PROPOSAL_v1_0.md:29, :170`.
  - "rrig6600c upgraded i5-8400T → i5-9400": `SESSION_CHANGELOG_20260324_S155.md:29, :71`;
    `TRIANGULATED_FUNCTIONAL_MIMICRY_VERIFIED_v1_0.md:34` lists rig-6600c as i5-9400. Today rrig6600c reads
    i5-8400.
- The driver parameters (`/sys/module/amdgpu/parameters`) and BIOS (5.13 06/08/2021, TB360-BTC PRO 2.0) are
  identical on all three hosts: `diff` of the files shows only the CPU-model line.

**(b) Slot-level PCIe links** (`lspci_amd_<host>.txt`, AMD devices only, cards idle after Runs F–H). Each card
is an internal switch (upstream port → downstream port → GPU). The **slot link** is the upstream port's link.

| Rig | Slot link, all 8 cards (upstream-port LnkSta) | Capability | GPU-internal link |
|---|---|---|---|
| rrig6600 | **5 GT/s ×1** (downgraded) | 16 GT/s ×8 | 16 GT/s ×16 |
| rrig6600b | **8 GT/s ×1** (downgraded) | 16 GT/s ×8 | 16 GT/s ×16 |
| rrig6600c | **8 GT/s ×1** (downgraded) | 16 GT/s ×8 | 16 GT/s ×16 |

- **Nothing in the link lines separates the fault-history cards** (rrig6600 `19:00.0`; rrig6600c `16:00.0`,
  `06:00.0`, `19:00.0`) from the other cards on their own rig, or rrig6600c from the clean rrig6600b.
  - Speed and width are rig-uniform: rrig6600 runs every slot at Gen2 ×1; rrig6600b and rrig6600c both at
    Gen3 ×1.
  - MaxPayload 256 / MaxReadReq 512 on all 24.
  - DevSta `CorrErr+ UnsupReq+` on all 24.
- **Correctable-error status bits** (CESta, sticky since boot; the kernel runs `pci=noaer`, so nothing
  reports them):
  - `Timeout+` on 5/8 rrig6600b cards, 6/8 rrig6600c cards (incl. all three rrig6600c fault-history cards)
    and 2/8 rrig6600 cards (`06:00.0`, `10:00.0`, not the fault card);
  - `Rollover+` on rrig6600 `06:00.0`; `BadTLP+` on rrig6600c `0a:00.0`.
  - The clean rig carries `Timeout+` on most cards too, so these bits do not separate the faulting cards.
- **Correction to the record:** the "link=16.0 GT/s PCIe x16" in the 11-15 `gpu_config_<host>.txt` captures,
  and the "Link: all x16 16 GT/s" row in the decode report §1.3, are the **GPU-internal** link. The slot
  links are ×1 at 5 or 8 GT/s.


## 13. Launcher draft v3 and the Run C manifest supplement (2026-11-17, offline, nothing landed)

**Draft v3:** `docs/drafts/DRAFT_launcher_cupy_cache_reset.diff`, sha256 `6e136180…`, 140 lines, against the
committed launcher `048fd47` (`21ff9399…`). `scripts/launch_fleet_manual.sh` was not touched. It answers every
Beta X-GCVM-LOCATED R2 Q2 point:

| Beta R2 Q2 requirement | Draft v3 |
|---|---|
| Check the listing's exit status, not only its output | `ENTS=$(find "$WC" -mindepth 1 -maxdepth 1 -printf .) \|\| cache_refuse enumeration_failed` |
| Count hidden entries explicitly; newline-safe | `find … -printf .` emits one dot per entry, hidden included; the count is `${#ENTS}`; refusal reason `nonempty_<n>` |
| Exclusive creation | `[ -e ] \|\| [ -L ]` → `exists`; then plain `mkdir` → `mkdir_failed` (also catches a racing creator) |
| Keep the per-run directory out of the later generic `mkdir -p` | per_run: `SHARED_MKDIR=""`; shared: `SHARED_MKDIR=" $WCACHE"` (today's line, byte-identical) |
| Refusal visible as rc 14 with a reason | `[cache] REFUSE reason=<…> tag host gpu path` on stderr, exit 14, before the worker starts |
| The same PID received the intended `CUPY_CACHE_DIR` | after the settle check, the CT reads `/proc/$worker_pid/environ`; a mismatch kills that worker and exits 15 |
| CT-side record per worker | `[cache] OK tag host gpu policy=per_run path created=exclusive entries=0 t_prep pid env_CUPY_CACHE_DIR`, printed before the launch ACK |
| Join the record to the PID; one worker refusing stops the fleet | `verify_cache_records` requires exactly one OK record per rig dispatch log for this tag, with matching gpu, path and the ACK's pid. It writes `$LOGDIR/cache_records_<tag>.tsv`. Any defect → `exit 1` before "all … daemons dispatched", hence before Gate 12 creates a coordinator |
| Policy on by default; opt-out; Zeus untouched | `RIG_CUPY_CACHE_POLICY=per_run` (default) or `shared`; `RIG_CUPY_CACHE_ROOT` (default `/tmp`, validated); the tag is `RUN_NONCE` or `launch<stamp>_<pid>`, validated; Zeus keeps `/tmp/cupy_cache_local_gpu<N>` |

- **Not in the draft:** the join from the TSV to the sentinel and liveness identity. That belongs to the
  existing gate scripts, which the draft does not modify, and is a landing item for Beta.
- **Old per-run directories** accumulate under `/tmp` until the CT restarts, a few tens of KB per worker per
  run.

**Gates and mutants:** `docs/drafts/test_launcher_cupy_cache_policy.py`, sha256 `8242b09d…`. It applies the
diff to a temporary copy and runs the CT fragments locally against fixtures, with an inert `sleep` stand-in.
- **Gates G-POLICY, G-PREP, G-RECORDS, G-WIRING: 4/4 pass, 5 of 5 repeated runs.**
- **14/14 mutants detected**, each by its intended gate. They covered:
  - default `shared`; validation bypass; tag dropped from the path; GPU dropped from the path;
  - `mkdir -p` (killed by an injected racing creator); exists check removed; non-empty check removed;
    enumeration status ignored;
  - env check removed; record count ignored; record pid ignored;
  - cache put back into `mkdir -p`; verify not called; ACK moved before the check.
- **A harness defect found and fixed during this work:** reading `/proc/<pid>/environ` immediately after the
  stand-in forked raced the `exec`. The fixture now settles 0.5 s, as the launcher does with
  `REMOTE_LAUNCH_SETTLE`.

**Run C manifest supplement:** `logs/rig_fault_rec/20261116_C_secured_175651/MANIFEST_SUPPLEMENT_20261117.md`
(+ `.sha256`); the originals are not altered.
- 39/40 manifest entries match.
- The coordinator log entry `e67a3d10…` is **the first 1,016,856 bytes** of the main-tree log. The file now in
  the folder (`777abb49…`) is **the first 1,035,885 bytes**.
- Cause: a second copy taken ~9 s after the manifest overwrote the first while the coordinator was still
  writing.
- The fault evidence (netconsole, traces) matches the manifest.
