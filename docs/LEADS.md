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
