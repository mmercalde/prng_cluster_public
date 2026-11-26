# LEADS — the tracked register

**What this file is.** The lead register required by `~/.claude/skills/lead-handling/SKILL.md` §5
and by `CLAUDE.md`. Every surviving candidate, anomaly or open finding gets an entry carrying:
**ID · what it is · where found · follow-ups RUN and their results · follow-ups OPEN · status**.

**Status vocabulary is fixed:** `OPEN` · `CLOSED-BY-EXPERIMENT` · `DEFERRED-BY-OPERATOR`.
**A lead may not be closed in a report while its entry has unrun follow-ups**, and only a *failed
follow-up experiment* may close one — never a background model, never a probability, never
"noted as future work". Carry this register forward between sessions; an open lead from three
phases ago is still open.

**Created 2026-08-27.** This file did not exist before that date; `CLAUDE.md` required it and it
was never made. Entries predating this file are **not** back-filled here — absence of an entry is
not evidence that a lead was closed.

---

## L-1 — float32/float64 survivor-filter seam: effective M is set by the HOST, not the kernel

**Status: OPEN.** **Repair is NOT authorized and is NOT proposed.** Filed as a defect, not a
footnote.

**What it is.** The GPU kernel and the host post-filter apply the same survivor threshold at two
different precisions, and they disagree at exact boundaries.

```c
// prng_registry.py, JAVA_LCG_HYBRID_KERNEL — float32 throughout
float match_rate = (float)matches / k;
if (best_match_rate >= threshold)                 // float32 >= float32
```
```python
# miner/range_miner_worker.py:1281 — the host re-applies it
if rate >= hybrid_threshold                        # float64(float32 rate) >= float64 threshold
```

`rate` arrives from the device as float32 and is widened to float64; `hybrid_threshold` is a
Python float64. When `k·τ` is an exact integer `m` **and** `m/k` is not exactly representable in
float32, the widened value lands just *below* the float64 threshold, so **the host rejects seeds
the kernel admitted** and the effective match requirement silently becomes `m+1`.

**Where found.** 2026-08-27, while validating the phase-3 capacity grid
(`S172_PHASE3_SURVIVOR_CAPACITY_CHARACTERIZATION.md` §2d). Found because two grid cells with
different M returned identical counts, which should be impossible.

**Measurement (MEASURED, production kernel, local GPU, 2²⁴-seed sample, midday/anchor 75).**
At `k=20, τ=0.35`: `7/20 = 0.34999999403953552` in float32 versus `0.35` in float64.

```
match-count histogram, k=20, threshold 0.30, 16,777,216 seeds
   matches= 6/20   exactly=42,436   >=M: 52,011
   matches= 7/20   exactly= 8,062   >=M:  9,575     <- kernel admits these
   matches= 8/20   exactly= 1,327   >=M:  1,513     <- host delivers only these
```

**9,575 kernel survivors become 1,513 host survivors — a 6.3× population change from a
floating-point representability accident.** Confirmed in three cells of the 64-cell grid:
`k=20/τ=0.35` (M 7→8), `k=20/τ=0.45` (M 9→10), `k=10/τ=0.70` (M 7→8). Every remaining exact-`kτ`
cell was checked and behaves as the naive formula predicts, so the residue is fully explained.

**What it means for correctness — state it plainly.** The survivor population that reaches Step 2
and the 22-array NPZ **depends on whether `m/k` happens to be representable in binary32**. That
makes populations produced at these boundaries **non-reproducible across any change to that
comparison** — a dtype change, a threshold quantization, a refactor that compares in float32 on
both sides, or a move of the post-filter. The affected geometries are exactly the ones an
optimizer is most likely to land on, because `τ = 0.35, 0.40, 0.45, 0.50, 0.70, 0.75` are round
decimals and Optuna's canonical thresholds are `round(…, 2)` (§2.36).

**Follow-ups RUN**
- ✅ Histogrammed the match-count distribution at `k=20` and `k=6` to prove the kernel/host
  disagreement is real and to size it (above). **Result: confirmed, 6.3× at the measured cell.**
- ✅ Enumerated every exact-`kτ` cell in the 8×8 grid and checked each against the naive M.
  **Result: three cells affected, all explained; no unexplained residue.**
- ✅ Confirmed the same double-application exists on the constant-skip path
  (`rate >= threshold` at the same host site) — the seam is not hybrid-specific.

**Follow-ups OPEN**
- ⬜ Determine whether the historical certified generations sit on an affected boundary. Attempt
  9 (`k=12, τ=0.71`) does not; the full corpus has not been swept.
- ⬜ Determine which side is the *intended* semantics. `CHAPTER_2` and the whitepaper describe a
  match-rate threshold without specifying precision; no governance artifact has been found that
  rules on it.
- ⬜ Relationship to the open §2.36 Optuna raw→canonical quantization item. That one is about
  threshold *provenance* (`optuna_suggested` vs `canonical_requested`); this one is about the
  *survivor filter*. They touch the same value at different seams and may want one ruling.

**Do not repair without authorization.** Either side could be made to match the other, and the
two choices produce different survivor populations. That is a semantics decision, not a bug fix.

---

## L-2 — coordinator ingress is bounded by COUNT and unbounded in BYTES

**Status: OPEN — flagged for Beta as a design observation. No fix proposed.**

**What it is.** `miner/range_miner_coordinator.py:9602`:

```python
inbound: "_queue.Queue" = _queue.Queue(maxsize=1024)
```

The connection reader decodes each frame and puts the **decoded message object** on this queue
(`:11284`). The bound is 1,024 *entries*. The byte weight of an entry is linear in the survivor
rate and has no bound anywhere on the path. Every other staging bound in the system
(`staging_high_water_files`, `staging_high_water_bytes`, the derived deferred bound) is applied
*after* this queue.

**Where found.** 2026-08-27, computing which resource binds first under high survivor volume
(`S172_PHASE3_SURVIVOR_CAPACITY_CHARACTERIZATION.md` §4.4). Reached by elimination: the
file-count retention bound is survivor-independent and the 16 GiB byte bound peaks at ~61%
utilisation from inside the Optuna-reachable space.

**⚠ THIS IS AN ALLOCATION LIMIT, NOT AN ARCHITECTURAL ONE.** VM101 is a Proxmox VM. Its memory is
configurable; the Zeus host has **64 GB**. The numbers below are properties of the *current
allocation*, and **raising the allocation moves the ceiling above the reachable maximum.**

**Arithmetic (COMPUTED — never observed).**

```
decoded bytes, full queue = 1024 x seed_cap x r x B_per_survivor
current allocation: MemTotal 15,924.8 MiB, SwapTotal 0        (live read, VM101, 2026-08-27)
B_per_survivor      = 237 - 271   (ledger wire 38.17 B x measured 6.2x decode expansion;
                                   271 B from a tracemalloc decode of a real payload)

                        r          full queue        at run-3 occupancy (550/1024)
  run 3        0.00612    1.4 - 1.6 GiB       0.74 - 0.85 GiB
  M=2 corner   0.06425   14.5 - 16.6 GiB      7.8  - 8.9  GiB

  exhaustion point at the CURRENT allocation:  r_crit = 0.0602 - 0.0688
  search space's provable reachable maximum:   r_max  = 0.0642
```

**At the current allocation the reachable maximum STRADDLES the exhaustion point** — above it at
271 B/survivor, at 93% of it at 237 B. An allocation of roughly **17 GiB clears a full queue at
the reachable maximum**, and ~32 GiB would leave 2× headroom; the host has 64 GB, so the headroom
exists. *Stated as arithmetic, not as a recommendation.*

**Two operational notes, both consequential:**
- **Raising the allocation before run 4 changes the acceptance run's environment.** Run 4 is
  meant to be an acceptance run for Brief I; changing VM101's memory beforehand introduces a
  second variable into it. **The allocation change should wait until after Brief I closes.**
- **Adding swap converts a hard OOM into degradation.** `SwapTotal` is currently 0, so the
  failure mode at the ceiling is an immediate kill rather than a slowdown that telemetry could
  catch. That is a separate change from the allocation size and has its own trade-off.

**The design observation, which stands regardless of box size and is the part for Beta.**
A queue bounded by count and unbounded in bytes is worth naming even when the box can be made
bigger: it means **ingress admission control cannot see volume at all**, so the safe operating
region is a function of an unrelated resource (host RAM) rather than of any governed bound. Every
volume-aware bound in the system sits downstream of the one place that has none. Raising the
allocation moves the ceiling; it does not give the system a way to know where the ceiling is.

**Follow-ups RUN**
- ✅ Measured decode expansion on a real payload shape: **6.2×, 271 B/survivor**.
- ✅ Confirmed by source that `inbound` carries decoded objects, not raw frames (`:11284`).
- ✅ Eliminated the two candidate retention bounds by measurement (§4.1, §4.2 of the capacity
  doc) — this is what left ingress as the binding resource.
- ✅ Read VM101 `MemTotal`/`SwapTotal` live rather than assuming.

**Follow-ups OPEN**
- ⬜ Establish the actual peak RSS of the coordinator process during run 3. `[S172-BP]` records
  queue *occupancy* (550) but no byte or RSS series exists, so the 0.74–0.85 GiB figure for run 3
  is computed, not measured. **This is the cheapest thing that would turn L-2 from computed into
  observed, and it needs no fleet — it needs an RSS sampler on the next run.**
- ⬜ Confirm whether staging throughput degrades with payload size in a way that raises queue
  occupancy at higher `r` (assumed in §4.4, not measured).
- ⬜ Beta ruling on whether ingress needs a volume-aware bound. **Not proposed here.**

**Four assumptions, stated as assumptions** (from capacity doc §4.4): (i) the queue reaches high
occupancy — run 3 reached 53.7% at 1/10th the volume; (ii) `seed_cap` 1×10⁶ (AMD) — the Zeus
worker's 2.5×10⁶ cap makes its frames 2.5× heavier; (iii) 237–271 B/survivor covers the survivor
list only, not the enclosing message; (iv) `MemTotal`, not available memory. **This ceiling has
never been observed.**

---

## L-3 — the 2026-08-22 netconsole packets are post-incident operator activity, not the incident

**Status: OPEN.** Two follow-ups are unrun, so this entry may not be closed — the register's rule
is that only a *failed follow-up experiment* closes a lead, and an operator explanation is not
one. What the operator explanation settles is the **11 packets**, recorded below as an explained
sub-question inside an open lead. **The 2026-08-22 mid-run freeze itself remains UNDETERMINED.**

**Filed 2026-08-28.** Recorded because a finding surfaced during the Run-4 post-commit runway
(TB R2 ruling step 3) that appeared to contradict a committed forensic document. It does not.

**What it is.** `logs/netconsole_all_rigs.log` contains **11 packets dated 2026-08-22** from the
three Proxmox hosts. `docs/RIG_CRASH_FORENSIC_20260822.md:24` states *"netconsole = EMPTY, and an
empty netconsole cannot distinguish 'no event' from 'not active'."* The file is not empty.

```
19:25:50  .121/.155/.163   NC-TEST3 $(hostname) $(date +%T)
19:26:09  .121             NCPROOF-pve-rig6600
19:26:11  .155             NCPROOF-pve-rig6600b
19:26:14  .163             NCPROOF-pve-rig6600c
19:52:59  .155             watchdog: watchdog0: watchdog did not stop!   (x2)
19:52:59  .155             systemd-shutdown[1]: Failed to finalize DM devices, ignoring.
20:33:53  .121             systemd-shutdown[1]: Failed to finalize DM devices, ignoring.
20:33:57  .163             systemd-shutdown[1]: Failed to finalize DM devices, ignoring.
```

**Where found.** 2026-08-28, checking netconsole sender arm-state for TB R2 post-commit step 3.
The live probe was UNAVAILABLE (all nine rig endpoints "No route to host"), so the archived
capture was read instead.

**OPERATOR EXPLANATION (Michael, 2026-08-28), which is the resolution.** He always shuts the rigs
down after a crash and did so on the evening of 2026-08-22. The `NC-TEST3`/`NCPROOF` pair is his
arm-verification test; the `systemd-shutdown` lines are that cleanup. **All 11 packets are
post-incident operator activity.**

**WHAT THIS EXPLAINS — the packets, completely.** No unexplained kernel event remains in the
2026-08-22 capture. The `watchdog did not stop!` line on `.155` is part of an orderly systemd
shutdown path, not a fault.

**WHAT THIS CLOSES — one branch of the forensic doc's disjunction, and only that one.** The doc
could not distinguish *"no event"* from *"not active"*. The `NCPROOF` packets prove the senders
were **active and armed** on all three hosts. Therefore:

- ❌ *"not active"* — **CLOSED.** netconsole was armed and demonstrably delivering.
- ✅ *"no event during the freeze itself"* — **STANDS**, and is now the correct reading.

**WHAT THIS DOES NOT CHANGE — the freeze.** The 2026-08-22 mid-run freeze remains **UNDETERMINED**.
netconsole captured **nothing** during the freeze window, and an armed-but-silent netconsole is
evidence of *no kernel-level message reaching the wire*, not evidence of a healthy host. The
forensic doc's audit-claim scope ("NOT a claim about why the hosts stopped") is unaffected.

**Timeline assembled from the evidence (all 2026-08-22):**

```
18:22:55   run log logs/gate12_20260822_182119.log begins
18:42:04   LAST line written to that log; mtime confirms  <-- the freeze
18:52:41   .155 boots  |
18:58:18   .163 boots  |  implied boot times, post-incident restart
18:58:55   .121 boots  |
19:25-26   netconsole armed; NC-TEST3 + NCPROOF from all three
19:52:59   .155 operator shutdown
20:33:53/57 .121 / .163 operator shutdown
```

**RESIDUAL DETAIL — RESOLVED, recorded because it was asked as open.** The question was whether
the frozen rigs accepted the 20:33 shutdown *directly*, or whether these packets come from a
boot-check-then-shutdown cycle. **It is the second.** Two independent proofs from the kernel
monotonic uptime stamps in the capture:

1. **Implied boot times all post-date the 18:42 freeze** (18:52:41 / 18:58:18 / 18:58:55). These
   are post-incident boot sessions; the frozen session is not the one that emitted these packets.
2. **Uptime is continuous across the observed window on all three hosts** — wall-clock delta and
   kernel-uptime delta agree to <0.07 s, so no *second* reboot occurred between the 19:25 arming
   and each shutdown:

```
   .121   wall 4082.733s   uptime 4082.799s   drift 0.067s   CONTINUOUS
   .155   wall 1629.276s   uptime 1629.261s   drift 0.015s   CONTINUOUS
   .163   wall 4086.597s   uptime 4086.625s   drift 0.028s   CONTINUOUS
```

**Minor, non-blocking.** Does not bear on Run 4 and requires no action. It does, however, sharpen
the freeze window: the hosts were alive enough at ~18:52-18:59 to boot, and alive enough at 19:25
to be armed and emit — so whatever happened is bounded to **18:42-18:52** and left no netconsole
trace. Stated as a bound, not as a hypothesis.

**Follow-ups RUN.**
- ✅ Live sender arm-state probe, all three CTs + hosts, 2026-08-28 — **UNAVAILABLE**, fleet down.
- ✅ Archived-capture read, per-sender attribution and last-packet times — 11 packets, as above.
- ✅ Kernel-uptime continuity test — resolved the residual detail (above).
- ✅ Run-log bound — last write 18:42:04.

**Follow-ups OPEN.**
- ⬜ On next power-on, re-arm netconsole **before** any run and confirm with `NCPROOF`
  (`docs/RUNBOOK_NETCONSOLE_REARM.md`). An unarmed sender during a future freeze would recreate
  exactly the ambiguity this entry just resolved.
- ⬜ The 18:42-18:52 freeze cause. **Not investigated here, not closed, no hypothesis offered.**
  Requires the fleet up and the root-free fault surfaces; host kernel ring is UNAVAILABLE from
  inside unprivileged LXC.

**What this entry must NOT be read as.** It does not explain the freeze, does not exonerate any
component, and does not make the 2026-08-22 run's environment healthy. It explains **11 packets**.

**CORRECTION APPENDED 2026-11-15 (rig-GPU-fault session; the text above is left as written).** The
"not active — CLOSED" branch is **contradicted by measurement**. `/etc/modprobe.d/netconsole.conf`
and `/etc/modules-load.d/netconsole.conf` were *created* 2026-08-22 19:25:07–08 on all three hosts
(`stat` birth time), and the incident boot on `.121`/`.163` (`journalctl -b -3 -k`, 10:40→18:27 /
18:19) has **zero** netconsole lines; the first `netpoll: netconsole` lines are in boot −2 at
19:25. The 19:26 `NCPROOF` packets prove arming *after* the post-incident reboot. Correct reading:
**netconsole NOT armed during the freeze; kernel-event status UNOBSERVED.** The freeze time is
also 18:32:02–07 per the ledger (forensic doc §5), not 18:42–18:52. **ADOPTED** at gate 1 by
Alpha, 2026-11-15.
**RULED (Alpha, after X-RIG-GPU-FAULT R3, 2026-11-15): the recorded wording is Beta's — "no
pre-freeze arming proof retained; incident kernel-event status UNOBSERVED".** History of the two
positions follows. **TWO POSITIONS (for Michael), 2026-11-15.** Alpha (adopted at gate 1): *not armed during the
freeze.* Beta (X-RIG-GPU-FAULT R1 Q2): the record supports only *no pre-freeze arming proof
retained; kernel-event status UNOBSERVED*. An ad hoc `modprobe netconsole …` in the incident
boot's lost journal tail (it ends 18:19 / 18:27) would leave neither the persistence-file birth
time nor a retained journal line. The session agrees with Beta's narrower reading. **Both
readings reopen the "not active — CLOSED" branch.** Separately measured (Beta R1 Q4, verified
2026-11-15): the persistence files never arm at boot (`vmbr0 doesn't exist, aborting` on every
boot of all three hosts); arming has been by hand. Detail: `docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md` "Null results and corrections".
Related: L-6.

---

## L-4 — staging admission can stay held by a DONE stripe, and then nothing pumps

**Status: OPEN — REPAIRED ON CPU, NOT YET COMMITTED; fleet incidence still OPEN.** Found
2026-11-14 by the miner seed-start work (`docs/CLAUDE_CODE_REPORT_MINER_SEED_START.md` §8).
Repaired the same day under `docs/CLAUDE_CODE_INSTRUCTIONS_MINER_ADMISSION_STALL_20261114.md`
(report `docs/CLAUDE_CODE_REPORT_MINER_ADMISSION_STALL.md`): the StripeComplete branch hands
the pump to the staging executor after its finalize. Not observed on the fleet; observed and
repaired on CPU with fake workers. Stays OPEN until a real fleet run shows no stall (only a
failed follow-up closes a lead).

**What it is.** The coordinator stages ONE attempt at a time and defers other attempts'
sub-stripe frames (`_try_admit_locked`, `miner/range_miner_coordinator.py:4832`: a non-empty
`_admitted` refuses). An admitted key is removed only by `_prune_admitted_locked` (`:4783`),
which runs only inside `_pump_deferred`, once the stripe is `done`. The staging-job path
finalizes then pumps (`_run_staging_job`, `:8230`, `:8234`). The StripeComplete path
finalizes and does not pump (`_serve_dispatch`, `:11897`). `enqueue_staging` does not prune.
If a stripe's last staging job finishes **before** its StripeComplete is processed, the
stripe reaches `done` on the non-pumping path, its key stays in `_admitted`, every later
attempt is deferred, and no further pump occurs. With `serve_timeout=None` (production) the
trial would wait indefinitely; no capacity timeout covers deferred frames that are not paused.

**Where found.** CPU, real `run_trial_miner` → `serve_trial`, fake workers over loopback
(`tests/fixtures/miner_seed_start_driver.py`), worktree at base `b11b6fd` + this brief's diff
(the defect surfaces are untouched by it).

**Follow-ups RUN.**
- *Mechanism, observed directly:* stalled run, live coordinator object: `_admitted` =
  {`…__st0_s5`} with s5 `done`; `_deferred` = [s6, s7]; trial aborted at serve_timeout.
- *Rate, subject vs control* (`data/miner_seed_start/leads/l4_result.json` in the worktree):
  2 workers, 4×250-seed stripes, no nudger: **1 of 10 stalled**; 1 worker (nothing can be
  deferred): **0 of 10**. 4 workers × 2²⁶-seed stripes: **4 of 4** attempts stalled before the
  harness nudger was added.
- *Recurrence in the record:* the stage-1 report's demo
  (`docs/CLAUDE_CODE_REPORT_STEP1_SEARCH_RECORD_STAGE1.md` §6.2) had its 4th stripe
  "cancelled by the fixture's 30 s serve_timeout" — the same shape, read then as a fake-worker
  limit.

- *Beta, X-MINER-START R1 Q7:* mechanism confirmed independently by injecting a DONE admission
  holder on a real coordinator over a throwaway ledger (next admission stays false; an
  explicit prune restores it). Beta rates it an **OBJECTION · LIVE** to launching Run 4 until
  repaired.

**Follow-ups RUN, 2026-11-14 (admission-stall brief; report §2–§3).**
- *M, base `7b9184a`, no nudger* (`data/miner_stall/M/base/` in worktree
  `/home/michael/tfm_worktrees/miner-stall`): 2 workers × 4 stripes **6 of 10 stalled**;
  1 worker **0 of 10**; 4 workers × 2²⁶ **4 of 4**. Every stall showed the L-4 state
  (`_admitted` = one `done` key; `_deferred` non-empty, unchanged ≥ 3 s). M ran while the
  baseline suites loaded the CPU, so rates are timing-dependent.
- *Correction to the earlier "1 worker: nothing can be deferred":* false. One worker's next
  stripe's frames defer behind its own previous `done` stripe. Observed: 1 of 3 one-worker,
  both-modes runs on base stalled while capturing the R4 control.
- *Deterministic reproduction on the real `_serve_dispatch`* (gate G-UNIT): at base,
  s0 `done`, `_admitted` = {s0}, `_deferred` = [s1], s1's shard `pending`.
- *Repair and tests, no nudger anywhere* (`tests/test_s172_admission_stall.py`): forced
  staging-before-StripeComplete and the opposite ordering, 2/4/25 workers, 20 runs each,
  120/120 committed; Run 4's shape in three orderings committed; one-worker run byte-equal
  to base; 4/4 mutants red.
- *Attempt 9 (read-only, `logs/gate12_20260817_181819.log`):* `deferred_high_water=1679`,
  5,364 of 5,896 frames deferred, staging 7.7 jobs/s. At 107 of 128 StripeCompletes the stripe
  still had deferred frames none of which had been released, so its staging could not have
  finished first (the safe order). Staging lag is the likely reason it did not stall
  (inference).

**Follow-ups OPEN.**
- Beta review of the repair (thread X-MINER-STALL).
- Does it occur with real workers? Shown only that attempt 9 did not stall. The first fleet
  run after the repair (Run 4) is the follow-up: it should commit with the coordinator
  sha256 in the report's landing list. Read-only on its logs.
- What latency between a final staging job and its StripeComplete makes it reachable with real
  payload sizes (fsync, hash)? Measurable on CPU with a real-size inline payload.
- *Mechanism chosen (pending Beta and owner):* pump after the StripeComplete finalize, handed
  to the staging executor. Prune in `enqueue_staging` is ruled out by the record (R-1 G1a pins
  `enqueue_staging` byte-identical, and it would not fire when no further frame arrives). A
  periodic pump is recovery, not prevention, and was not added.

---

## L-5 — WATCHER loses its own INFO records once Step 1's search comparison imports window_optimizer

**Status: OPEN — REPAIRED ON CPU, NOT YET COMMITTED; real-log confirmation OPEN.** Found
2026-11-14 by the miner seed-start work (report §8).
Introduced by stage 1 (`915d382` / `b1cd2fa`) by import order, not by any line it wrote.

**What it is.** `step1_presented_configuration` (`agents/watcher_agent.py:769`) imports
`window_optimizer`. Its import chain (sklearn → `scipy.stats` → `numpy.testing` →
`numpy.distutils.log` → setuptools `_distutils/log.py:54`) sets the **root** logger to
WARNING. `run_step` calls it in the S4 comparison, before it builds the command, so every
later INFO record in that WATCHER process is dropped — including `EXEC CMD` and the
`[S172-PIN] step1_pin_source` record at `:2424`. The structured provenance in the step
result is unaffected.

**Follow-ups RUN** (`data/miner_seed_start/leads/l5_run.log` in the worktree). Fresh process,
`window_optimizer` not yet imported, real `run_step` (dispatch intercepted), open search:
root level **20 → 30** across the first Step 1 dispatch; `EXEC CMD` records captured **0**.
Control in the same process with the level restored: `EXEC CMD` **1**.

**Repair, 2026-11-14 (admission-stall brief Part 3; not committed).**
`step1_presented_configuration` restores the root level after the `window_optimizer` import
(`agents/watcher_agent.py:776-787`). Gate G-L5 (`tests/test_s172_small_repairs_part3.py`):
fresh process, real `run_step` on the pinned path, dispatch intercepted. `window_optimizer` is
imported inside `run_step`, `EXEC CMD` **1**, `[S172-PIN]` **1**, root level 20 → 20. Mutant
(base function): `EXEC CMD` 0, `[S172-PIN]` 0, root 20 → 30.

**Follow-ups OPEN.** Confirm on a real WATCHER log file: the next Step 1 dispatch's log should
now CONTAIN `EXEC CMD` and `[S172-PIN]`.

---

## L-6 — rig GPU fault at the phase-4 first launch: rrig6600c 16:00.0 `0x00801231` SQC recurs

**Status: OPEN.** Filed 2026-11-15 under `docs/CLAUDE_CODE_INSTRUCTIONS_RIG_GPU_FAULT_20261115.md`;
report `docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md`.

**What it is.** The same first-fault signature — rrig6600c `0000:16:00.0`,
`GCVM_L2_PROTECTION_FAULT_STATUS 0x00801231`, SQC(inst), PERMISSION_FAULTS 0x3 — observed on
2026-04-04 (S162), 2026-04-17 (S163KARG) and 2026-11-15 (Run 4 attempt 2), followed within
~100–200 ms by `0xFFFFFFFF` on it and sibling cards and loss of the card's PCIe functions. On
2026-11-15 the first fault output from rrig6600 `19:00.0` (TCP `0x00841011`) was **received**
0.63 ms before rrig6600c's, ~73 ms after phase-4 stripes were created, and before either host
had a recorded phase-4 shard. The VBIOS-L03 card (rrig6600c `06:00.0`; gpuN mapping
**provisional**, KFD order, gpu1) faulted on 2026-04-04, 2026-04-17 and 2026-11-15.

**Follow-ups RUN.**
- ✅ Recur across the record: S162, S163KARG, S164 (different signature), Aug-22 (unobserved —
  no arming proof retained, two positions in L-3), today. Pattern as above.
- ✅ Same recorded trial inputs, same host boot, morning pass / afternoon fail. That no static
  setting explains it alone **assumes no setting changed within the boot** (per-run settings
  were not snapshotted).
- ✅ Large phase-3 survivor volume is not necessary for today's phase-4 host hang (60 vs 13.1M).
  This does not show the two hangs share a cause.
- ✅ Recurrence despite the configured cap mitigation (installed after Aug-22); the effective cap
  at the fault is UNOBSERVED (the script logs unconditionally).

**Follow-ups OPEN.**
- ⬜ Sub-second power/rail record across a phase-4 first launch (H2).
- ⬜ Card-vs-slot swap on rrig6600c (does the fault follow 16:00.0's card or root port 1d.1?).
- ~~Clock-pin / voltage A/B~~ — **EXCLUDED from T4** (Kaspa profile recorded as crash-causing,
  S165; Alpha withdrew the clock-profile lead 2026-11-15). Owner may reopen.
- ✅ (R3) Two HIP/HSA runtimes in one worker-environment process **confirmed at the library
  level**, read-only with no device visible or opened. After `import cupy`, only
  `/opt/rocm-6.4.3` `libamdhip64`/`libhsa-runtime64` are mapped; after `import torch`, the
  torch wheel's HIP 6.3 copies are mapped too. Whether both **initialise on the GPU** in a live
  worker is OPEN (first test).
- ✅ (R3) rrig6600b `19:00.0` SMU `TransferTableSmu2Dram` 0xFFFFFFFF ×2 at 13:47:28.123: a single
  failed metrics read while idle (the telemetry TSV shows 0 W / 0 °C for that card in that
  second only); the card is healthy since. Observation on the stable rig, recorded; not a GCVM.
- ⬜ Live worker `/proc/PID/environ` on a rig (HSA_* absent by reading, not observed).
- ⬜ Whether earlier faults were rate-limited away ("3 callbacks suppressed").
- ✅ (owner addition) PCIe atomics read on all hosts: absent on all 7 PCH slots per rig, including
  the stable rrig6600b → not a rig discriminator by itself.
- ⬜ Riser path of the three faulting cards (riser, USB cable, riser power) — owner hand swaps
  e1/e2/e3, f1 (CPU-slot move) per `CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md` T4.5.
- ⬜ Cold vs warm kernel cache (arm h): first-run-in-CT-boot passed 2/2, later runs failed 2/2.
- ✅ (R2) Recurrence extended through the record: 2026-04-04 S162 Option B also had **both**
  repeat cards first (`16:00.0` and `06:00.0`, SQC, 48 ms apart,
  `TB_UPDATE_S162_OPTION_B_RESULTS.md:30-37`). 2026-05-09 S174_D1: rrig6600 + rrig6600c
  crashed; rrig6600b "not verified at session close, last known healthy 22:55"
  (`SESSION_CHANGELOG_20260509_S174_D1.md:40-42`).
- ✅ (R2) `cwsr_enable=0` is not necessary: all three hosts run `cwsr_enable=1` today, and the
  same pair failed.
- ⬜ **(R2, after Beta R2) Possible two HIP runtimes in one worker process.** Measured read-only
  on rrig6600c CT (2026-11-15 ~15:40):
  - CuPy's extensions link `/opt/rocm/lib/libamdhip64.so.6` (6.4.3) via `ldd`.
  - torch `2.7.0+rocm6.3`, installed in all three rig venvs, bundles `torch/lib/libamdhip64.so` and
    `libhsa-runtime64.so`; both have soname `libamdhip64.so.6` / `libhsa-runtime64.so.1`.
  - `libtorch_hip.so` has `NEEDED libamdhip64.so` with `RPATH $ORIGIN`.
  - The miner's per-sub-stripe cleanup imports torch and calls `torch.cuda.synchronize()` and
    `empty_cache()` after CuPy is loaded (`miner/range_miner_worker.py:1059-1066`).

  **Inference [UNVERIFIED]:** glibc resolves `NEEDED libamdhip64.so` by name, not by the loaded
  object's soname, so a second (6.3) HIP/HSA runtime would load beside CuPy's 6.4.3 in the same
  process, on the same GPU. The PWC worker has the same cleanup (`sieve_gpu_worker.py:78-94`).
  **Follow-up:** read `/proc/PID/maps` of a live worker in the first owner-run test (no extra
  load), then a torch-free cleanup arm.
- ✅ (after R3, register-verified) **torch's presence alone does not separate the eras.** torch was
  in the sieve worker 2025-11-27 → 12-24 (module import; removed in `41165dd`, which records it
  broke CuPy HIPRTC compilation by altering HIP include paths). It returned as a per-job cleanup
  (`b425171`, 2026-01-14), then per chunk (`20b0e7c`, 2026-01-26), and now runs per sub-stripe in
  the long-lived miner (`miner/range_miner_worker.py:1051-1072`). Jan–Mar has zero GCVM records,
  but **no kernel capture existed before 2026-04-01**, so that is absence of record. What changed
  is the **frequency** of the torch cleanup and that **the process no longer exits** afterwards.
  The torch-called state of each PWC-era GCVM event is UNKNOWN (gated by `S163_MEM_DEBUG` from
  04-08): record check proposed in `docs/CLAUDE_CODE_PLAN_RIG_GPU_PROBE.md` §1.3.
- ✅ (after R3; wording per Beta X-RIG-GPU-PROBE R1 Q1) Onset timing: in 10 of 26 Part-C GCVM events
  the onset is **near worker startup**, with module/load/launch and torch-call state **UNKNOWN per
  event**; today's onset is near phase-4 dispatch, in processes that had run the torch cleanup after
  every earlier sub-stripe. Timing fact only (plan §1.4).
- ⬜ Arms P / P1 / M / N and the fleet P-vs-N comparison: `docs/CLAUDE_CODE_PLAN_RIG_GPU_PROBE.md`
  §2–§4; thread X-RIG-GPU-PROBE.
- ⬜ Memory handling at the hand-over (arm g2): the miner calls `free_all_blocks()` after every
  sub-stripe (`miner/range_miner_worker.py:1051-1070`, `:1331`); single-card hand-over series
  with a serialised re-run.
- ❌ **2026-11-15 18:28 — the two-runtime hypothesis is REFUTED as the sole cause (arm N, fleet).**
  The no-torch worker (`048fd47`, parity 30/30) ran Run 4 with `RIG_CLEANUP_TORCH=off`. Attempt 3
  credited clean at 17:59. The 18:18 confirmation repeat, with all 24 rig workers logging
  `torch_imported=False` and one HIP/HSA runtime, **faulted the same three cards**: phase-4
  dispatch 18:28:29.403; first received GCVM 18:28:29.449 on `.121 19:00.0`
  (`0x00841051`, TCP 0x8, RW 1) and in the same millisecond on `.163 16:00.0` (`0x00801231`,
  SQC (inst) 0x9, RW 0) and `.163 06:00.0` (`0x80001281`, SQC (inst), RW 0). Both hosts hung;
  rrig6600b silent. A second runtime is **not required** for this fault. Whether it changes the
  rate is not measured (n=2 per arm). Evidence: `/home/michael/run4_confirm1_fault_20261115_182829`,
  main-tree `logs/netconsole_all_rigs.log:855-890`. Decode and follow-ups:
  `docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT_DECODE_20261115.md`.
- ⬜ Card hold-out run and card↔identity permutation run: designed in the decode report §4.

## L-7 — today's faults came only on the second run after a host boot (warm CuPy kernel cache)

**Status: OPEN.** Filed 2026-11-15 19:00 PST. Report:
`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT_DECODE_20261115.md` §1.6, §3.

**What it is.** On `.121`/`.163` (boots 08:27 and 13:53 PST, `journalctl --list-boots`), the
first miner run of each boot ran clean (09:05 attempt 1; 17:49 attempt 3), and the second run of
each boot faulted at phase-4 dispatch + 39–46 ms (13:11 attempt 2; 18:18 confirm 1). The CT's
`/tmp/cupy_cache_gpuN` is cleared at CT start, so run 1 compiles every kernel and run 2 loads them
from the cache. rrig6600b (`.155`, one boot 08:27 → all four runs, warm from run 2 on) never faulted.

**Against it, from the record:** register Table 1 has GCVM events with no warm cache possible:
C-15 (uptime 81 s), C-34 (166 s), C-47 (161 s), C-86 (275 s), and B-59, B-61 with no run active.
So a warm cache is **not necessary** for GCVM across the record. The other ~37 GCVM rows carry no
boot time in the register.

- ⬜ Recur: run-since-boot for every GCVM row with a run log (boot markers from netconsole
  `[uptime]` stamps and journals; main-tree netconsole starts 2026-05-11).
- ⬜ Mechanism: a warm load removes the per-worker compile (≈0.3–0.7 s phase-4 compile on `.156`,
  attempt 1 cache mtimes), so all 24 rig workers reach the first phase-4 launch within a few ms of
  each other. Test: a run-#2 with `CUPY_CACHE_DIR` emptied per worker (cold), and a run-#1 with the
  cache pre-seeded from the copied tars (warm). Register precedent: in-process pre-compile
  `22bb145` "made crashes worse" (reverted `ac32cd9`); a warm-up barrier "didn't help" (01-23).

- ✅ **2026-11-16: third same-boot repeat. Run A (run #1 after boot, cold cache) clean and credited
  16:23–16:4x; Run B (run #2, same boot, warm cache) FAULTED at phase-4 dispatch + ~45 ms, 16:56:20.382, on the
  same three cards** (rrig6600 `19:00.0`, rrig6600c `16:00.0` and `06:00.0`). rrig6600b (run #2 of its boot,
  warm) was clean.
  - Recorded with the outside recorder: `logs/rig_fault_rec/20261116_AB_secured_165728/`.
  - Tally for these two hosts: run #1 clean 3/3 (11-15 09:05, 11-15 17:49, 11-16 16:23); run #2 faulted 3/3.
  - The register's counter-examples (C-15, C-34, C-47, C-86, B-59, B-61) still stand. The pattern is
    today-specific evidence, not a general rule.
- ✅ Fault addresses located (`docs/CLAUDE_CODE_REPORT_GCVM_LOCATED_20261116.md`, Beta X-GCVM-LOCATED R1).
  - No covering mapping/PTE event in the retained capture for the data faults.
  - The instruction fetch landed in the recorded non-executable phase-3 code-staging block.
  - Host removal/eviction/pressure is excluded for the recorded events. A host translation or visibility
    error on a real page is **not** excluded.
- ✅ Load-to-launch gaps (report §8). Run A ph4 buffer→code was 279–671 ms; Run B ph4 was 0.7–0.9 ms on all 24
  workers. The faulting cards were not the shortest. Phases 1–3 of B had the same fast path, no fault.
- ✅ **V2 = Run C (2026-11-16, run #1 after boot on rrig6600 and rrig6600c, warm caches pre-filled from
  rrig6600b): FAULTED** at phase-4 dispatch 17:55:48.651 + 41 ms. Cards: rrig6600c `16:00.0` (gpu6) and
  rrig6600c **`19:00.0` (gpu7, new)**, both TCP write `0x00841051`, and rrig6600 `19:00.0` (gpu7, first printed
  status dead). rrig6600c `06:00.0` (gpu1), which faulted in B, was clean. rrig6600b (run #3 of its boot) was
  clean.
  - The warm path was confirmed on all 24 workers: phase-4 buffer→code 0.70–1.03 ms (one 4.06, one 6.86).
  - Reading, as a single event: **carried state from an earlier run on the same boot is NOT necessary**; a
    first run with warm caches is sufficient for the fault on these hosts.
  - New: rrig6600c gpu6's second fault (TCP read `0x00801011`, +46 ms) hit `0x738d72000000`. That page was
    inside a 16 MiB buffer mapped and unmapped 99 times in phases 1–2 (last unmap 17:49:35), at
    base + 4 MiB, with no mapping in phase 4.
  - Evidence: `logs/rig_fault_rec/20261116_C_secured_175651/` (`MANIFEST.sha256`, `fault_map.txt`).
- ⬜ V1 (Beta's): run #2 on a boot with the per-worker caches emptied first.
- ✅ **Run D (2026-11-16 18:10, run #1 after the 18:00 boot on rrig6600 and rrig6600c, all caches emptied:
  cold): CLEAN.** T1–T7 recorded; no page-fault line after 18:00. Phase-4 dispatch 18:19:30.391.
  - Cold path confirmed on all 24 rig workers: buffer→code 280–879 ms, dispatch→code 290–899 ms.
  - Faulting-history cards: rrig6600 gpu7 573 ms, rrig6600c gpu6 680 ms, rrig6600c gpu7 879 ms (the longest).
  - rrig6600b: run #4 of its boot, cold, clean.
  - Evidence: `logs/rig_fault_rec/20261116_D/`.
- ✅ **Run E = V1 (2026-11-16 18:28, run #2 on the same boot as Run D, caches emptied again: cold): CLEAN.**
  T1–T7 recorded; no page-fault line. Phase-4 dispatch 18:38:12.542.
  - Cold path confirmed on all 24 rig workers: buffer→code 276–1184 ms.
  - Faulting-history cards: rrig6600c gpu7 587 ms, rrig6600c gpu6 701 ms, rrig6600 gpu7 754 ms.
  - Stream: `logs/rig_fault_rec/20261116_D/stream/`; snapshots `logs/rig_fault_rec/20261116_E/`.
  - **2×2 tally on rrig6600 and rrig6600c**, single events per run, phase 4:

    | | cold | warm |
    |---|---|---|
    | run #1 | clean 4/4 | fault 1/1 |
    | run #2+ | clean 1/1 | fault 3/3 |

    The fault tracks the warm (~1 ms load→launch) phase-4 path, not the run's position on the boot.
    Carried state is neither necessary (C) nor sufficient (E).
  - Remaining confounder: in every warm run, the CuPy arange/fill kernels and the sieve kernel run within
    ~1 ms of each other after the ~220–237 s idle. In cold runs a 280–1180 ms compile separates them.
    Whether the fast load or the closeness of the dispatches matters is not separated.
  - The register's cold-start GCVM counter-examples (C-15, C-34, C-47, C-86) still stand: they were
    PWC-era, at worker launch, on a different code path.
- ✅ **2026-11-17, one boot (hosts 16:43–16:48), caches emptied before each run: Runs F (17:00:45), G (17:20:33)
  and H (17:40:57): all CLEAN.** T1–T7 recorded for each; no page-fault line all day.
  - Cold path confirmed on all 24 rig workers in each run: phase-4 dispatch→code F 288–824 ms (buf→code),
    G 288–900, H 293–1428 ms.
  - Trace loss counters (overrun / commit overrun / dropped) were 0 on all three hosts through G.
  - Evidence: `logs/rig_fault_rec/20261117_{F,G,H}/`; stream `20261117_F/stream/`.
  - **Running tally on rrig6600 and rrig6600c, phase 4, single events per run:**
    - cold: clean 8/8 (first runs: 11-15 ×2, 11-16 A, D, 11-17 F; later runs on the same boot: E, G, H);
    - warm: fault 4/4.
- ✅ 2026-11-17 hardware notes (report §12):
  - **CPU model does not separate the rigs:** i5-9400 on rrig6600 and rrig6600b, i5-8400 on rrig6600c. Older
    repository documents that say rig-6600b is i5-8400, or rig-6600c i5-9400, conflict with the live read
    (listed in §12, not edited).
  - Driver parameters and BIOS are identical on all three hosts.
  - Slot links are ×1: 5 GT/s on every rrig6600 card, 8 GT/s on every rrig6600b and rrig6600c card. Nothing
    in the link lines or correctable-error bits separates the fault-history cards.
- ✅ 2026-11-17 (offline, nothing landed):
  - Launcher draft v3 (`docs/drafts/DRAFT_launcher_cupy_cache_reset.diff`) meets every Beta R2 Q2 point. Gates
    4/4 and mutants 14/14 in `docs/drafts/test_launcher_cupy_cache_policy.py`. It is **not** sent to Beta (a
    round 3 would be the thread's last).
  - The Run C manifest supplement records the coordinator-log mismatch: both versions are prefixes of the
    main-tree log.
  - CPU corrected in the current-state repository tables.
- ✅ 2026-11-17 Beta X-GCVM-LOCATED R3 (final round): **draft v3 not landable as is — OBJECTION LIVE.**
  - The R2 enumeration objection is resolved.
  - Two open defects in `verify_cache_records`:
    - it returns success even when the TSV cannot be written;
    - its regex join accepts mismatched dotted tags/roots, missing host/t_prep, a wrong-GPU ACK and a
      diagnostic line in place of the ACK.
  - **Required to land:**
    - literal-field parsing with complete-write failure;
    - an executable launcher-path test with stubs, including Gate 12 stopping;
    - `per_run` mandatory on the counted path, with Route A included.
  - **Required before the first counted run:**
    - the sentinel/liveness join plus a second `CUPY_CACHE_DIR` read at the parked probe;
    - evidence freeze and launcher provenance;
    - live start-up time and footprint.
  - The thread is at its three-round limit (`docs/exchange/from_beta/X-GCVM-LOCATED_R3_BETA_REVIEW.md`).
- ✅ 2026-11-17 Beta X-CACHE-POLICY R1 on draft v4 (`docs/exchange/from_beta/X-CACHE-POLICY_R1_BETA_REVIEW.md`):
  **Q1, Q2 OBJECTION LIVE; Q3 SUGGESTION.**
  - Beta confirms as satisfied: the literal parser, the dispatch/entry-point evidence, mandatory `per_run` on
    both counted paths, the join keys and ordering, provenance/freeze placement, and L2 being equivalent.
  - **Open before landing:**
    - git HEAD and status reads must be status-checked (a failed read currently passes with empty/zero
      provenance);
    - the environment checks must keep NUL boundaries and compare exactly one complete value (newline inside
      a value and non-UTF-8 cases);
    - a failure after rename (directory `fsync`) must not leave a success-looking TSV;
    - the test must also run in the landed layout.
  - **Open before the first counted run:**
    - freeze the nonce/PID-matched sentinel line per rig slot for the start-up timing;
    - measure footprint from the TSV paths (apparent vs allocated bytes stated).
- 2026-11-17 draft v5 (`docs/drafts/cache_policy_v5/`, nothing landed) addresses every R1 item: status-checked git and hash
  reads; NUL-exact environment checks at startup and at the parked re-read; post-rename invalidation of the TSV; a
  landed-layout harness with a local Route A fixture and a fixed stub; the timed sentinel per slot frozen; the footprint
  script (apparent and allocated bytes). Tests: 109/109 three times pre-landing, 108/108 on a simulated landed
  tree, 19/19 mutants. Sent as X-CACHE-POLICY R2. No rig run: the warm/cold lead itself is unchanged and still OPEN.
- ✅ 2026-11-17 Beta X-CACHE-POLICY R2 on v5: R1 items satisfied. Q1 LIVE (the test's scratch path depends on the draft
  layout); Q2 LIVE (a footprint SSH timeout escapes without INCOMPLETE); theoretical: the encoder status. First run:
  Route A parameters checked separately; join/freeze time in the timing report.
- 2026-11-18 draft v6 (`docs/drafts/cache_policy_v6/`, nothing landed): fixes R2 items 1–4 (README corrected for item
  3; the diffs are unchanged); items 5 and 6 are in the README's first-counted-run checklist. Tests: 112/112 three
  times pre-landing, 111/111 installed under `tests/` in a simulated checkout, 22/22 mutants. Sent as
  X-CACHE-POLICY R3 (final). The lead remains OPEN; no rig run.
- ✅ 2026-11-18 Beta X-CACHE-POLICY R3 on v6: **NOT LANDABLE**. One blocker: the package broke the existing D6 suite
  (`tests/test_s172_d6_integration_repair.py`); R2 items 2–4 hold; a leftover (draft paths at import); three
  first-run checklist corrections.
- 2026-11-18 draft v7 (`docs/drafts/cache_policy_v7/`, nothing landed): D6 repaired as a draft diff (Part B on `shared`,
  WS-7 per-worker path, new WS-2f). On a simulated landed tree: D6 84/84 ×3 (base 82/82; v6 unrepaired 71/82); cache
  suite 111/111 landed, 112/112 ×3 pre-landing, 22/22 mutants. Sent as new thread X-CACHE-POLICY-D6 R1. Lead still
  OPEN; no rig run.

---

## L-8 — RANGE-MINER two-way (forward ∩ reverse) count exceeds the independence figure

**Status: OPEN.** Filed 2026-11-22. Note: `docs/drafts/NOTE_LEAD_BIDIRECTIONAL_EXCESS_20261122.md` — **not on
main**; on branch `pwc-offset` (WIP snapshot `9809d7e`, worktree `~/tfm_worktrees/pwc-offset`). Its instruments and
outputs are in that worktree's `logs/lead_bidir_20261122/` (sha256s in the note's "Files" table). Brought into the
register 2026-11-25 from the uncommitted draft entry at `pwc-offset:docs/LEADS.md:656`.

**What it is.** In runs E and F (window 12, anchor 25, midday, skip [0,16], switch on, τ_f 0.47 / τ_r 0.38), the
variable-skip two-way count was 17 and 13. The independence figures fwd × rev / 2^28 are 6.26 and 6.21. Independence
was **assumed**.

**Where found.** All runs were lead-run on 2026-11-22 at commit `00e2ee7` (present in all four bundle manifests).
The draft entry says "owner approved each" (`pwc-offset:docs/LEADS.md:660`). Direct owner approvals were located for
G (`logs/skipsw_session/OWNER_OVERRIDE_runG_thr3838.txt:1`) and H (`logs/skipsw_session/OWNER_OVERRIDE_runH_thr3030.txt:1`);
approval records for E and F were not located.

| Run | Thresholds (fwd/rev) | Slice | Tree | Evidence |
|---|---|---|---|---|
| E | 0.47 / 0.38 | [2147483648, 2415919104) | `~/skipsw_run_thr4738` | `logs/skipsw_session/bundle_thr4738` |
| F | 0.47 / 0.38 | [2415919104, 2684354560) | `~/skipsw_run_thr4738n` | `bundle_thr4738n` |
| G | 0.38 / 0.38 | E slice | | `bundle_thr3838` |
| H | 0.30 / 0.30 | E slice | | `bundle_thr3030` |

The blocks for E and F are in `~/skipsw_blocks/{e,f}`.

**Measured rows, no verdict.** Phases 3 and 4 are the variable modes. Every published row is variable (`skip_mode` 1).

| Run | Phase 3 (fwd) | Phase 4 (rev) | Two-way rows | Independence figure | Ratio |
|---|---|---|---|---|---|
| E | 34,454 | 48,791 | 17 | 6.26 | 2.72 |
| F | 34,419 | 48,449 | 13 | 6.21 | 2.09 |
| G | 326,627 | 48,791 | 167 | 59.37 | 2.81 |
| H | 2,368,585 | 595,773 | 12,156 | 5,256.9 | 2.31 |

Run H also has phase 1 = 1 and phase 2 = 2 (constant mode). The lead's "2,368,586 / 595,775" are phase 1 + 3 and
phase 2 + 4.

**Follow-ups RUN:**
- ✅ **Recur on the neighbouring slice = run F:** 13 two-way against 6.21.
- ✅ **CPU reproduction** (numba port of `tests/phase6/known_answer_reference.py`, validated against it). It gives
  E, F, G and H exactly, and keeps the per-phase sets the rig run deletes.
- ✅ **Test the mechanism** (2^24 seeds per window). The question: is the overlap sieve arithmetic alone, or does it
  depend on the draws?

  | Group | Windows | F≥6 R≥5 | F≥4 R≥4 |
  |---|---|---|---|
  | other real midday windows | 100 | 3.47 | 2.40 |
  | shuffled run window | 100 | 3.91 | 2.35 |
  | iid random values | 100 | 3.32 | 2.37 |
  | **cross** (forward and reverse on unrelated windows) | 50 | **1.13** | **1.05** |

  - **Measured, for the windows and thresholds tested:** the excess appears whenever both passes compare against the
    same values, and does not depend on which of the tested values. It rises with repeated or mirror-equal values in
    a window, and the run window has neither.
  - The group comparisons are pooled by group (Σ both / Σ independence figure, note `:114`). The 2^24 sample of the
    run window itself gave zero two-way rows at the tight threshold (F≥6 R≥5: 0 both, independence figure 0.39,
    note `:122`).
  - **Mechanism read from the kernels:** one output stream per seed, wide skip windows. In 26 of the 30 rows, one
    output counts as a hit for the same draw in both passes.
  - The anchor-25 window's excess at the run thresholds (2.41 [1.62, 3.43]) is not above the control groups'.
- ✅ **Real-window hit-rate comparison for this frozen configuration** = L-9's counted grade (run H, A1, midday grade
  block, k ≤ 20, 904 draws: 22 hits against 17.907 expected, p 0.1925). *(Recorded 2026-11-25, Beta X-LEADS-TIDY R1
  Q2.)*
- ✅ **Neighbourhood** (population description only):
  - the 30 rows are spread across both slices (gap CV 1.09);
  - they share no residue class mod 1000, 2^16 or 2^19;
  - none follows from another within 2^20 steps;
  - survivor density around them is at the slice average.

**Limits** (note §2.2, §4). Measured inputs: the marginals (this window, rig and CPU) and the ratio (iid controls).
Assumed: "that the iid ratio transfers to this window. The F≥4 R≥4 per-window spread supports that, but the
assumption is not proven at F≥6 R≥5." No GPU and no rig for the follow-ups; everything is CPU on VM101. The evening
session was not used. Recur over the whole record as a rig run was not run.

**For hit rate against `k/1000`** (note §6). The measured outcome "predicts **nothing about pool hit rate.**" Because
the same excess appears for shuffled and random windows, "the excess itself cannot be read as evidence for or against
the draws containing structure." "Population size is load, not quality." Had the excess depended on the draws, it
would have predicted that pools ranked from real-window rows beat matched pools from control-window populations on
unseen draws; "The controls did not show that difference at the thresholds tested. **This is not a result about hit
rate either way.**"

**Follow-ups OPEN:**
- ⬜ **Shuffled-draw rig control:** scoped in the note §5, not built. It needs the owner's go. It measures overlap
  counts and is a separate follow-up from the hit-rate measurement.
- ⬜ **Hit-rate measurement**, the only settling one: pools from a real-window population (run H's 12,156 rows),
  optionally against matched shuffled-draw pools, scored on unseen draws against `k/1000`. This is the main line
  (Step 2 on run H), not gated by this lead. *(2026-11-25: the real-window part is done for this frozen
  configuration — L-9's counted grade, above. Still OPEN: the matched shuffled-pool comparison, and further
  unseen-draw grading.)*
- ⬜ Recur on other sessions and window sizes (not run; CPU-cheap).

**Licensed phrasing:** its rate is consistent with the measured same-window control background. No follow-up has
tested a hit-rate effect. It remains OPEN. Downgrading it is the operator's decision.

---

## L-9 — run H prediction pools against k/1000 on draws the sieve never saw

**Status: OPEN.** Filed 2026-11-23. Brought into the register 2026-11-25 from the uncommitted draft entry at
`e2e-grader:docs/LEADS.md:658` (branch `e2e-grader`, WIP snapshot `d52e84b`), which the grader landing left out
(`docs/drafts/e2e_v1/READY_GRADER_LANDING_20261124.md:115-121`).
- Reports (on main): `docs/drafts/e2e_v1/RESULT_E2E_GRADE_RUNH_20261123.md` (counted) and
  `docs/drafts/e2e_v1/RESULT_E2E_FOLLOWUP_RUNH_20261123.md` (follow-ups).

**What it is.** The yardstick measurement (CLAUDE.md): on draws the pipeline never saw, is the hit count of TFM's run H
prediction pools above Σk_t/1000 (equivalently, is the hit rate per graded draw above Σk_t/1000 divided by the number of
graded draws)?
- **Run H:**
  - 12,156 survivors;
  - sieve window of 12 midday draws at offset 25, dated 2002;
  - thresholds 0.30/0.30;
  - 1/16 of the seed space;
  - one trial.
- **Step 6 vote as built:** combined index, skip 0.
- **Arms:**
  - A1 = the frozen Step 5 CatBoost ranking (cross-validation score +0.0019);
  - A0 = forward_matches + reverse_matches, seed_asc tie-break.
- **Instrument:** `scripts/diagnostics/e2e_grader.py` (sha256 `a4d96fe7…`, draft, not
  Beta-reviewed), a non-certifying route. This description applies to the frozen instrument used for this grade.

**Where found.** Dataset `daily3.json` (sha256 `513648160d35…`).
- Grade block: combined [16261, 18068), 2023-09-07..2026-02-26, sealed until grading.
- Label block: [14454, 16261).
- Pools: `~/tfm_worktrees/e2e-vote/data/e2e_runH_vote/` and `…/e2e_runH_vote_followup/`.
- Grades: `~/tfm_worktrees/e2e-grader/data/e2e_runH_grade/` and `…/e2e_runH_grade_followup/` — **untracked, worktree
  only** (`READY_GRADER_LANDING_20261124.md:127`).

**Follow-ups RUN:**
- ✅ **Counted test, 2026-11-23 13:50 (owner go 13:44).** A1 vs chance, 904 midday grade draws,
  one-sided 5 %, m = 1.
  - **22 hits of 904**, against **17.907** expected; null sd 4.19; p **0.1925**.
  - Lift 1.23, two-sided 95 % interval (descriptive) [0.77, 1.85]; one-sided lower bound 0.835.
  - The first count reaching p ≤ 0.05 was 26.
  - Controls passed.
  - Not above the modelled background at this sample size.
- ✅ **D1–D3, descriptive** (same session). None is above the modelled background.

  | Run | Hits | Expected | p |
  |---|---|---|---|
  | D1: A0, midday | 21/904 | 17.91 | 0.2607 |
  | D2: A1, midday, m = 3 | 22/904 | 17.91 | 0.5774 (adjusted) |
  | D2: A0, midday, m = 3 | 21/904 | 17.91 | 0.7821 (adjusted) |
  | D3: A1, evening | 16/903 | 17.89 | 0.7072 |
  | D3: A0, evening | 22/903 | 17.89 | 0.1917 |

  - D2's A1 vs A0 comparison was "not tested: unequal budget".
- ✅ **Follow-ups F1 and F2, descriptive, 14 runs at m = 1 each (owner go 14:40).**
  - F1 (extend): pools of 100 and 300 on the grade block, A1 and A0, both sessions.
  - F2 (recur): A0 on the label block at pools of 20, 100 and 300, both sessions.
  - All controls passed. None is above the modelled background at this sample size.
  - Arithmetic: 10 of 14 above expected and 4 below.
  - Smallest p 0.1416: A0, F2, midday, pool 100, 96 against 86.02.
  - The runs are dependent (same survivors; nested pools, where within the same arm and block each smaller raw pool
    was measured as the exact prefix of the larger; same draws), so there is no combined p.

**Limits** (counted report §5; follow-up report §6). One generation (run H); the Step 6 vote as built (combined
index, skip 0); features and target as built; a frozen CatBoost ranking for A1; a non-certifying harness route; a
draft grader, not Beta-reviewed (the frozen instrument used for this grade). **Power:** "With 904 draws at k about 20, this test needs 26 hits, about 1.45×
chance, to reach 5 %. A smaller real lift could exist and not be detected here."

Every background figure here assumes each draw is uniform on 000–999, independent of the other draws and of the pools
(follow-up report `:182-185`). The pool sizes, the draw counts and the hits are measured.

**For hit rate against `k/1000`.** This lead *is* the hit-rate measurement. The counted result is "a measured null for
this declared configuration: run H, A1, midday grade block, k ≤ 20, 904 draws. **It closes no lead.**" "No verdict on
TFM either way." (counted report §6.)

**Follow-ups OPEN:**
- ⬜ A second generation with the sieve window placed just before the holdout. Run H's window is
  dated 2002. *(Added 2026-11-25 from the record: run I, "the open follow-up of lead L-9"
  (`logs/e2e_session/TASK_RUN_I_PREP.txt:3`): run H with the window anchor 6695, the 12 midday draws 2021-03-05 to
  2021-03-16 (`docs/drafts/anchor_v1/RUNBOOK_RUN_I_20261123.md:10-16`). Owner go 2026-11-23 18:28, then 18:29 "Lets
  wait for all the agents to finish -then run I" (`logs/e2e_session/OWNER_GO_run_I_20261123.txt`). Cap: owner
  2026-11-24, "YES (one-off, run I only)" past the 50,000 survivor cap
  (`logs/e2e_session/OWNER_RULINGS_pending_list_20261124.txt:5`). No completion result was found in the records
  checked.)*
- ⬜ A later untouched period, once the dataset is updated.
- ⬜ Pools from a shuffled-draw population, as a control.
- ⬜ The vote-position arms of map Q2.

**Status: OPEN.** No follow-up listed as open has been run (as of the source, 2026-11-23). Downgrading this lead is
the owner's decision.

---

## L-10 — block-0 why-report: three per-seed battery columns separated top-N voters of the drawn number from top-N non-voters (pools 100 and 300)

**Status: OPEN.** Filed 2026-11-24 as proposed ID **L-why-0** in
`docs/drafts/loop_v1/READY_WHY_BLOCK0_20261124.md` §4 (`:233-245`) — **not on main**; on branch `ch14-learn` (WIP
snapshot `7368f34`, worktree `~/tfm_worktrees/ch14-learn`, outputs under its `data/ch14_learn/why_block0/`).
DEVELOPMENT evidence only: one block, already graded and looked at.

**What it is.** On walk-forward block 0, in the contrast C1b — top-N voters for the drawn number against top-N
non-voters (`docs/drafts/loop_v1/READY_LOOP_MISS_20261123.md:180`) — three per-seed battery columns had a signed-
attribution interval excluding 0 with the same sign in both sessions, as measured in the table below (`:233-239`).

The source's proposed register text (draft wording), verbatim (`:245`):
> on walk-forward block 0, among each tree family's top 100, the survivors that voted the drawn number differed in
> signed attribution on `batt_ac_lag_01` (−, XGBoost both sessions; CatBoost and LightGBM midday),
> `batt_cs_zero_crossings` (−, XGBoost both sessions) and `batt_fft_diff_peak` (+, LightGBM both sessions).

Against the measured table: `batt_fft_diff_peak` was found for LightGBM at midday 100 **and 300** and evening 100, so
"top 100" does not cover all of it. For `batt_ac_lag_01`, the CatBoost and LightGBM midday results are same-block F1
results; LightGBM's midday interval excluding 0 is F1 at pool 300.

**Where found.** Walk-forward block 0 = draws [3808, 4402), 2006-08-20 to 2007-06-12; run H's 12,156 survivors; LL-8
(`scripts/diagnostics/attribution_cycle.py` `a30cc50f…`) run by hand per family, pools 20/100/300, B = 2000, seed
20261124. For C1b and C2, counts of per-seed columns whose interval excludes 0 were mostly near the rough multiplicity
reference ("about 2.5 of 48–52"); the two higher cells were CatBoost evening and LightGBM evening at pool 100, 10 each.
That reference applies to C1b/C2 under a hypothetical null in which no column differed and the intervals were
independent (`:222`); they are not independent. The source reports the different C1 contrast separately: 14–45 of
48–52 in every cell, "mostly selection" (`:231`). Pool 20 abstains everywhere (4 to 9 hit draws per session).

| Column | Where found | F1, RUN (all four families, block 0) | F2, RUN (run H miss report, raw values, different period and model) |
|---|---|---|---|
| `batt_fft_diff_peak` +, LightGBM | midday 100 and 300; evening 100 | CatBoost midday 100 + (interval excludes 0); XGBoost and NN mixed, intervals include 0 | 15 cells: 5 +, 10 −, none excluding 0 |
| `batt_ac_lag_01` −, XGBoost | midday 100; evening 100 | CatBoost midday 100 and 300 − (excluding 0); LightGBM midday 300 − (excluding 0), other cells −; NN mixed | 15 cells: 9 +, 6 −, none excluding 0 |
| `batt_cs_zero_crossings` −, XGBoost | midday 100; evening 100 | LightGBM midday 300 − (excluding 0); CatBoost and NN mixed | 15 cells: 5 +, 10 −, 1 excluding 0 |

**Follow-ups RUN.**
- ✅ F1 (other families, same block): "partial, dependent".
- ✅ F2 (run H miss report, raw values): "mixed".

**Limits** (`:241-242`, `:270-272`).
- "families share survivors (top-300 overlap 62–187) and the same 594 draws, so agreement across families on block 0
  is not independent recurrence."
- F2 "measures a raw feature, not an attribution, under another model and period. It does not test the object
  directly."
- One block only; every number is DEVELOPMENT; cells are dependent. The NN integrated-gradients residual (1.2e-5)
  makes its attributions approximate.

**For hit rate against `k/1000`.** The source gives no directional hit-rate prediction and no graded settling
measurement for these three columns. It records that "The effect on hits is not measured" for the variants it lists
(`:271`). TODO-WHY-1's later-block contrast (below) tests recurrence of the attribution pattern, not hit rate. Until
TODO-WHY-1's trigger and queue are built, the lead runs the attribution cycle by hand after each block
(`logs/e2e_session/OWNER_RULINGS_pending_list_20261124.txt:27-28`).

**Follow-ups OPEN.**
- ⬜ "the same contrast on the next walk-forward blocks, with each block's own models. This happens automatically when
  the lead runs LL-8 after each block (TODO-WHY-1). Nothing here closes or downgrades these leads." (`:243`)

The later what-moves note did not test it: "The why-report leads (L-why-0) are unchanged; nothing here tested them."
(`READY_WHAT_MOVES_20261124.md:208`, branch `ch14-learn`.)

---

## L-11 — block-0 label windows: the last 900 / 1,200 label draws gave fewer hits

**Status: OPEN.** Filed 2026-11-24 as proposed ID **L-move-1** in `docs/drafts/loop_v1/READY_WHAT_MOVES_20261124.md`
§5 (`:204`) — **not on main**; on branch `ch14-learn` (WIP snapshot `7368f34`; harness and results under that
worktree's `data/ch14_learn/what_moves/`, every run in `what_moves_final.json` `613fb959…`). DEVELOPMENT; "a map of
what moves block 0's numbers. It is **not a pick of a winner**".

**What it is.** "Labels from the last 900 / 1,200 label draws gave fewer hits at pools 100 and 300 and a worse mean
rank. The windows overlap; z about −2." LightGBM, target = Step 6 vote hit rate over the window, 3 splits each
(42, 1, 2), scored on block 0 against the split-seed floor (9 runs; pool 300 mean 154.2, SD 13.1):

| Window | Pool 100 | Pool 300 | Mean rank | Δ pool 100 (SD units, z) | Δ pool 300 | Δ rank | Overlap |
|---|---|---|---|---|---|---|---|
| last 900 [2908, 3808) | 48.3 | 139.0 | 514.3 | −7.6 (−1.49, z −2.2) | −15.2 (−1.16, z −1.7) | +14.1 (+1.34, z +2.0) | 9 |
| last 1,200 [2608, 3808) | 48.7 | 137.3 | 515.1 | −7.2 (−1.43, z −2.1) | −16.9 (−1.29, z −1.9) | +14.9 (+1.42, z +2.1) | 9 |

Pool-300 hits per run: last 900 — 138 / 144 / 135; last 1,200 — 122 / 146 / 144 (splits 42 / 1 / 2). Chance at pool
300 is 153.9 expected hits (SD 10.7).

**Where found.** Block 0 = draws [3808, 4402), LightGBM, 3 splits each.

**Follow-ups RUN.**
- ✅ The other windows: last 300 and last 600 near 0, first 900 −0.5 SD.

**Limits** (`:24-25`, `:143`, `:200`, `:213`).
- "The two windows share 900 draws, so they are not two findings."
- "Of about 84 comparisons in table B, 6 reach |z| ≥ 2; about 4 would be expected by chance alone, and the comparisons
  are dependent."
- In each z, "the floor mean and SD are measured on this block; independence between runs and the normal
  approximation are assumed."
- "One run per variant cannot show an effect smaller than about 25 hits at pool 300, roughly 2 split-seed SDs." This
  resolution is for pool 300 on block 0.
- Chance SDs are binomial with the pools' distinct counts measured and the draws assumed independent and uniform
  (`:118`).

**For hit rate against `k/1000`.** The source states no prediction beyond the block-0 measurement above; it calls
these "open leads to test on other blocks, not effects" (`:25`). The settling comparison is the named variant against
the current baseline on later unseen blocks, with equal pool budgets, hits against `k/1000`. More splits on block 0
only quantify run-to-run variation there. *(Beta X-LEADS-TIDY R1 Q2, adopted by the lead 2026-11-25.)*

**Follow-ups OPEN.**
- ⬜ The same windows on walk-forward blocks 1+.
- ⬜ More splits on block 0.

---

## L-12 — block-0 capacity: fewer Optuna trials (5) gave more pool-300 hits; depth 4 a smaller rise

**Status: OPEN.** Filed 2026-11-24 as proposed ID **L-move-2** in `READY_WHAT_MOVES_20261124.md` §5 (`:205`), branch
`ch14-learn` (as L-11). DEVELOPMENT.

**What it is.** "Fewer Optuna trials (5) gave more pool-300 hits; depth 4 gave a smaller rise." The trials comparison is
5 trials against the 20-trial baseline. LightGBM, 3 splits (42, 1, 2), against the split-seed floor:

| Variant | Pool 20 | Pool 100 | Pool 300 | Mean rank | Δ pool 20 (SD units, z) | Δ pool 100 | Δ pool 300 | Δ rank | Overlap |
|---|---|---|---|---|---|---|---|---|---|
| `--trials 5` | 12.7 | 58.7 | 170.7 | 487.6 | +2.1 (+0.75, z +1.1) | +2.8 (+0.55, z +0.8) | +16.4 (+1.26, z +1.9) | −12.6 (−1.20, z −1.8) | 176 |
| `--trials 50` | 13.0 | 58.7 | 159.0 | 496.7 | +2.4 (+0.86, z +1.3) | +2.8 (+0.55, z +0.8) | +4.8 (+0.36, z +0.5) | −3.4 (−0.33, z −0.5) | 172 |
| depth at most 4 (`--depth-max 4`) | 14.7 | 56.0 | 162.0 | 493.8 | +4.1 (+1.45, z +2.2) | +0.1 (+0.02, z +0.0) | +7.8 (+0.59, z +0.9) | −6.3 (−0.60, z −0.9) | 171 |

Pool-300 hits per run: `--trials 5` — 184 / 170 / 158; depth 4 — 159 / 164 / 163 (splits 42 / 1 / 2). Chance at
pool 300 is 153.9 (SD 10.7). `--depth-max` is an in-process patch of the harness, not a production setting (`:73`).

**Where found.** Block 0, LightGBM, 3 splits.

**Follow-ups RUN.**
- ✅ trials 50: +0.4 SD at pool 300; depth 4: +0.6 SD at pool 300.

**Limits.** As L-11 (`:25`, `:200`, `:213`): about 84 dependent comparisons, 6 at |z| ≥ 2 against about 4 by chance;
independence between runs and the normal approximation assumed in each z; one run per variant cannot show an effect
smaller than about 25 hits at pool 300 (pool 300, block 0).

**For hit rate against `k/1000`.** The source states no prediction beyond the block-0 measurement above ("open leads to
test on other blocks, not effects", `:25`). The settling comparison is the named variant against the current baseline
on later unseen blocks, with equal pool budgets, hits against `k/1000`. More splits on block 0 only quantify run-to-run
variation there. *(Beta X-LEADS-TIDY R1 Q2, adopted by the lead 2026-11-25.)*

**Follow-ups OPEN.**
- ⬜ More splits.
- ⬜ Other blocks.
- ⬜ XGBoost.

---

## L-13 — XGBoost and CatBoost give different results on CPU and GPU at the same split

**Status: OPEN (device determinism, not a hit lead).** Filed 2026-11-24 as proposed ID **L-move-3** in
`READY_WHAT_MOVES_20261124.md` §5 (`:206`), branch `ch14-learn` (as L-11). DEVELOPMENT.

**What it is.** "XGBoost and CatBoost give different results on CPU and GPU at the same split (166→170, 150→158)."
The frozen block-0 models were trained on the card, the retrains on CPU at the same split 42: XGBoost 166 on the card,
170 on CPU; CatBoost 150 on the card, 158 on CPU (pool-300 hits). "LightGBM was CPU in both and matches exactly. So the
device is itself a change for XGBoost and CatBoost." (`:130-134`)

**Where found.** Block 0, split 42.

**Follow-ups RUN.** None.

**Limits.** At split 42, one card run (the frozen model) and one CPU run per family; CatBoost has one CPU run only, no floor (`:33`, `:116`).

**For hit rate against `k/1000`.** The source labels this "device determinism, not a hit lead". It gives no
prediction and no settling measurement for unseen-draw hit rate. The proposed repeat (below) is a check on GPU
repeatability for this device lead.

**Follow-ups OPEN.**
- ⬜ "repeat the GPU training once, to see whether the card itself is deterministic".

---

## L-14 — run H miss report: `residual_*` separated top-N voters of the drawn number in 2 of 4 draw sets

**Status: OPEN.** Filed 2026-11-23 in `docs/drafts/loop_v1/READY_LOOP_MISS_20261123.md` §4 (`:297-325`, on main),
proposed there as an "L-9 sub-item" for the lead "to merge into L-9 or file as its own ID" (`:322`). The lead's
answer: "follow-up F-a (residual_* ranking variant) is registered OPEN and will be run as a development variant on the
walk-forward once the driver is ready; lead L-9 unchanged" (`logs/e2e_session/TASK_LOOP_ADDENDUM_3.txt:6`). Given its
own ID 2026-11-25 (lead, `TASK_LEADS_TIDY_ADDENDUM_1_20261125.txt`). **Related: L-9** (the same run H pools and
grades). DEVELOPMENT / DESCRIPTIVE evidence, NOT COUNTED (`:3`).

**What it is.** "`residual_std` and `residual_max_abs` have a remainder interval excluding 0, with the same sign, in two
separate draw sets" (`:299`):
- A1, grade midday, pool 20: +0.544 and +0.386;
- A0, label midday, pool 300: +0.100 and +0.121.

The remainder is H − top-N mean, ≈ C1b (top-N voters against top-N non-voters) (`:178-180`, `:270`). The source's proposed
register text, verbatim (`:323`):
> `residual_*` (Step 3 per-seed) separated top-N voters of the drawn number from the rest of the top N in 2 of 4 draw
> sets (A1 grade midday N = 20; A0 label midday N = 300). The recurrence check RAN (read-only, run3): mixed,
> `residual_max_abs` positive in 10 of 15 dependent cells. Graded follow-ups F-a and F-b are OPEN.

**Where found.** The run H miss report (window h6), over run H's A1 and A0 pools on the grade and label blocks; data in
`~/tfm_worktrees/e2e-vote/data/e2e_loop_miss/` (untracked).

**Follow-ups RUN.**
- ✅ **Recurrence** (`data/e2e_loop_miss/followup_residual_recurrence.txt`, `5f8dc771…`; "the same features in every
  other cell, read from run3", `:303`):

  | Feature | Remainder sign over non-abstained cells | Rank-free C2 |
  |---|---|---|
  | `residual_std` | 7 positive, 8 negative of 15 | interval includes 0 in all 4 draw sets |
  | `residual_abs_mean` | 7 positive, 8 negative of 15 | interval includes 0 in all 4 draw sets |
  | `residual_max_abs` | 10 positive, 5 negative of 15 | grade midday +0.018 [−0.001, +0.036]; others include 0 |

  "The recurrence check did not show a consistent sign for `residual_std` or `residual_abs_mean`. For
  `residual_max_abs` it is mixed." (`:314-315`)

**Limits.** "The cells share draws, and the pools are nested, so these counts are not independent." (`:311`) The miss
report's overall measured fact: "at this sample size, nothing in the 51 per-seed Step 3 features separates the top-N
survivors that voted the drawn number from those that did not" (`:280`).

**For hit rate against `k/1000`.** "This does not measure whether a different ranking would hit more often. That needs
a graded variant" (`:280`). "No graded follow-up has tested whether ranking on this group changes pool hits." (`:316`)
The settling measurement is F-a.

**Follow-ups OPEN** (`:318-320`).
- ⬜ **(F-a)** "A declared ranking variant that adds or weights the `residual_*` group. Compare it with A1 at equal k,
  through an LL-4-style setting comparison, on draws after 2026-02-26 or a new generation (run I)." A graded ranking
  comparison.
- ⬜ **(F-b)** "The same recurrence check on run I's pools, once they exist."

Downgrading it is the owner's decision (`:316`).

---

## L-15 — block-0 feature groups: Step 5 without `battery` gave fewer pool-100 hits (z −2.0)

**Status: OPEN.** From `docs/drafts/loop_v1/READY_WHAT_MOVES_20261124.md` table B (`:156-159`) — **not on main**; on
branch `ch14-learn` (WIP snapshot `7368f34`; harness and results under that worktree's `data/ch14_learn/what_moves/`).
DEVELOPMENT. **The source proposed no lead for it** (its lead table, `:202-206`, has L-move-1 to L-move-3 only). The
lead first decided not to register it (`logs/e2e_session/TASK_LEADS_TIDY_ADDENDUM_1_20261125.txt`), withdrew that
decision because it rested on a background figure, and registered it on 2026-11-25
(`logs/e2e_session/TASK_LEADS_TIDY_ADDENDUM_2_20261125.txt`).

**What it is.** One row of what-moves table B, quoted exactly with its header and group label (`:147`, `:156`, `:158`):

| Variant | Runs | Pool 20 | Pool 100 | Pool 300 | Mean rank | Δ pool 20 (SD units, z) | Δ pool 100 | Δ pool 300 | Δ rank | Overlap |
|---|---|---|---|---|---|---|---|---|---|---|
| **B2 features** | | | | | | | | | | |
| without `battery` | 5 | 9.6 | 50.2 | 148.6 | 506.2 | −1.0 (−0.34, z −0.6) | −5.7 (−1.12, z −2.0) | −5.6 (−0.43, z −0.8) | +6.0 (+0.57, z +1.0) | 127 |

Against the split-seed floor (9 runs, splits 42, 2–9): pool 100 55.9 (SD 5.1), pool 300 154.2 (SD 13.1) (`:149`).
Chance at pool 100 is 56.5 expected hits (SD 7.2), at pool 300 153.9 (SD 10.7) (`:118`). Δ is the variant's mean minus
the floor mean; SD units are Δ over the floor SD; z = Δ / (SD·√(1/n + 1/9)) (`:141-143`). Overlap 127 is the mean
top-300 survivors shared with the same-split base, split 1 left out (`:144`). The source's summary: "without `battery`,
127 [of 300 kept] … The changes are within about 0.5 SD at pool 300: … −5.6 hits" (`:20`).

**Where found.** Walk-forward block 0 = draws [3808, 4402), LightGBM, split seeds 42, 1, 2, 3, 4 (`:138`).

**Follow-ups RUN.** None. The source records none for this row.

**Limits** (the source's own).
- "Of about 84 comparisons in table B, 6 reach |z| ≥ 2; about 4 would be expected by chance alone, and the comparisons
  are dependent." (`:25`) The source does not state the inputs of the "about 4"; it states that the comparisons are
  dependent.
- Measured on this block: the floor mean and SD (`:200`, `:149`) and the pools' distinct counts behind the chance SDs
  (`:118`). Assumed: "independence between runs and the normal approximation" in each z (`:200`); draws independent and
  uniform for the chance SDs (`:118`). The z "treats runs as independent … Runs with high overlap are positively
  dependent, so their z is conservative" (`:143`).
- One block; DEVELOPMENT (`:3`).
- "One run per variant cannot show an effect smaller than about 25 hits at pool 300, roughly 2 split-seed SDs" (`:213`),
  for pool 300 on block 0.

**For hit rate against `k/1000`.** The source gives no prediction for this row. It calls table B's |z| ≥ 2 results "open
leads to test on other blocks, not effects" (`:25`).

**Follow-ups OPEN.**
- ⬜ The same variant (`--exclude-feature-groups battery`) against the current baseline on later walk-forward blocks,
  with equal pool budgets, hits against `k/1000`.
- ⬜ More splits on block 0, to size run-to-run variation there.

**Cross-reference only (no inference):** L-10's three columns (`batt_fft_diff_peak`, `batt_ac_lag_01`,
`batt_cs_zero_crossings`) are `battery` columns.

**Licensed phrasing:** its rate is consistent with the modelled background. No follow-up has yet tested whether it
extends or recurs. It remains an open lead. Downgrading it is the owner's decision.

---

**Older ID ranges.** `docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:90` (WP-6) and
`docs/TFM_SESSION_HANDOFF_20261113.md:183` name "L-4 to L-9" from a v1.0 proposal that is not in the repository; their
mapping to this register's entries of the same numbers is not recorded, so those references are historical only.
