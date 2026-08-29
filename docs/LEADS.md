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
