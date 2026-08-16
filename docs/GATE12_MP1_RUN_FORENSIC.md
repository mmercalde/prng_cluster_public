# GATE-12 MP-1 RUN — FORENSIC

**Run:** `distributed_config_t1_5c010902` · **nonce** `gate12-20260816_160503-4630` (fresh; all prior
nonces stay burned) · **launch tree** `168b6f18be9e343912e79ade317c5c312a15ea06` · 2026-08-16.

**Result: FAILED at stage 2 on a compute-lease expiry — the same wall as attempt 7. THE CAUSE IS NOW
NAMED AND MEASURED.** Gate 12 is NOT passed; nothing composes from this attempt.

**No remedy is proposed.** MP-1 was authorized as measurement; the choice of mechanism is Beta's.

---

## 1. What the run achieved

| Beta §21 completion condition | result |
|---|---|
| truthful GPU preflight PASS | ✅ 3/3 rigs at 8/8 |
| 25-worker frozen admission | ✅ `expected_workers=25`, frozen set `bea580e76490` |
| `GATE-12 SATURATION VERDICT : SATISFIED` | ✅ **both** verdicts satisfied |
| all four stages complete | ❌ stage 2 terminal; stages 3-4 never assigned |
| D3.5 canonical publication | ❌ not reached |
| S145 publication-bound coverage | ❌ not reached |
| certified cursor == 2,147,483,648 | ❌ still **0** |

Seven pre-launch gates passed in order: clean-tree admission · GPU 3/3 at 8/8 · **rig parity 30
MATCH · 0 MISMATCH · 0 UNAVAILABLE** · clean-tree pre-dispatch · sentinel 25/25 · liveness 25/25
ALIVE+PARKED · release written on 4 hosts.

**Stage ledger:** phase 1 **32/32 done**; phase 2 **9 done, 23 cancelled**. Per the binding forensic
frame the 23 cancellations are the abort-cleanup footprint of **one** terminal event.

**Fail-closed cascade held end to end:** trial terminal → `provenance_validated` false →
`MinerIngressError` → no `optimal_window_config.json` → certified cursor still **0** →
`git status --porcelain` still empty.

---

## 2. The terminal event

```
16:17:25.914  [S172-BP] burst_exact stage=1 family=java_lcg_reverse phase=2
              exact=830 conservative=1088 claimed=25 queued=7
16:22:53.813  [F1/F2] TRIAL TERMINAL class=compute_lease_expiry
              stripe='…__st1_s30' worker='rrig6600c:gpu0' attempt=0
              workflow phase 2 is CONSTANT-MODE — fails immediately, never retries
```

5 min 28 s after the stage was assigned, against a 300 s compute lease. Correct behaviour for a
constant phase, exactly as in attempt 7. **Same stripe index (`st1_s30`), different worker**
(`rrig6600c:gpu0` here, `rrig6600b:gpu2` in attempt 7) — the failure follows the geometry, not a
machine.

---

## 3. THE CAUSE — measured, at three levels, and it sums

Over the whole trial: **969.349 s of serve loop, 1,771 iterations.**

```
iteration   969.349 s
  drain     818.090   84.4% of the loop
    msg     689.204   84.2% of the drain
      staging 681.153  98.8% of msg      <-- THE SERVE LOOP'S TURN INSIDE enqueue_staging
      pump      0.000  (on the serve thread)
      msg_remainder     8.051
    drain_remainder   128.886
  loop_remainder        4.209   0.43% of the loop
```

**The attribution reconciles.** `loop_remainder_total = 4.209` and the independently-computed
certified `unattributed_total = 4.209` agree exactly, and
`remainder_negative_loop = remainder_negative_drain = remainder_negative_msg = 0` — no level's
children ever exceeded their parent. **Only 0.43 % of the run is outside a named phase.** The
question "where did the time go" has a complete answer for the first time in this arc.

### 3.1 Both halves of Beta's conjunction are large — the chain is SUPPORTED, not refuted

The certification correction required refutation to be a conjunction: the chain is refuted **only if
both** the serve thread's staging/lock-wait **and** the staging-thread pump attribution stay small
and flat. Measured (run totals, thread-keyed):

| side | thread | exclusive | calls |
|---|---|---|---|
| staging executor | `miner-staging_2` | **913.3 s** | 592 |
| staging executor | `miner-staging_1` | **910.6 s** | 654 |
| staging executor | `miner-staging_3` | **908.6 s** | 588 |
| staging executor | `miner-staging_0` | **908.0 s** | 658 |
| **serve loop** | `MainThread` (`staging`) | **681.2 s** | 1,862 |
| abort cleanup | `miner-cleanup_0` | 0.0 s | 1 |

**≈3,640 s of `pump` across four threads over a 969 s wall clock — i.e. ~3.76 of the 4 staging
threads were inside `_pump_deferred` at essentially all times**, holding `_admission_lock`; and the
serve loop spent **681.2 s of 969.3 s** blocked inside `enqueue_staging`, which takes that same lock.
**0.366 s of lock wait per sub-result.**

Neither side is small. Neither is flat. **The §7.3 chain is confirmed on both halves.**

### 3.2 BUILD-UP — what grew, and by how much

The window series is the derivative the totals cannot show. Per-frame message cost:

| window | staging (s) | msg (s) | frames | **s/frame** | inbound_qsize |
|---|---|---|---|---|---|
| 1 (early) | 0.07 | 0.23 | 49 | **0.005** | 0 |
| 2 | 6.96 | 8.32 | 417 | 0.020 | 479 |
| 88 (late) | 10.66 | 10.69 | 5 | **2.138** | 350 |
| 89 (last full) | 11.71 | 11.75 | 6 | **1.958** | 369 |

**Per-frame cost grew ~400×, from 5 ms to ~2 s, and the phase that grew is `staging`** — from 0.07 s
per 11 s window to 11.71 s, i.e. from 0.6 % of the window to **100 % of it**. In the final windows
the serve loop does nothing but wait for `_admission_lock`.

Run-total derived rates: `drain_seconds_per_frame = 0.331`, `msg_seconds_per_frame = 0.279` —
averages over a run whose late behaviour is 7× worse than its mean, which is precisely why the
series and not the total is the evidence.

### 3.3 LATE-INDEX — and it is RATE starvation, not ORDER starvation

```
drain_passes            1771
drain_passes_partial    1762      99.5% of passes reached fewer connections than were live
drain_pass_conns_max      25      (it CAN reach all 25)
drain_pass_conns_min       0
drain_pass_live_max       25
drain_frames_total      2471      = 1.4 frames per pass
```

Per-connection census, last full window — **25 live, 3 serviced, 22 measured zero**:

```
conn13  frames=2 passes=2 pos_min=1 pos_max=1   rrig6600b:gpu6
conn19  frames=2 passes=2 pos_min=1 pos_max=1   rrig6600c:gpu4
conn23  frames=2 passes=2 pos_min=1 pos_max=1   rrig6600c:gpu2
22 unserviced: status OK, frames_window 0, positions None (UNOBSERVED, never 0)
```

**This is the discrimination MP-1 was built to make, and it comes out on the side the report listed
third, not first.** The serviced connections sit at `pos_min = pos_max = 1` — they are *first* in
their pass, not late. Nothing is being skipped over. Each pass simply services **one frame, at
position 1, and then the ~2 s that frame cost has consumed the pass**. Twenty-two connections wait
because the drain cannot afford a second frame, not because they are behind others in the queue
order.

Order starvation is therefore **refuted**: it predicts starved connections at uniformly *high*
positions, and the measurement shows the opposite. **The mechanism is rate starvation, order-neutral**
— §7.1's third row — and the remedy space for the two is not the same, which is why the distinction
was worth instrumenting.

### 3.4 The instrument did not cause what it measured

| | attempt 7 (pre-MP-1) | attempt 8 (MP-1) |
|---|---|---|
| `staging_jobs_completed` | 1,250 | 1,246 |
| `staging_jobs_per_sec` | 1.225 | **1.285** |
| `inbound_qsize_high_water` | 547 | 553 |
| `pause_events` | 0 | 0 |
| `deferred_high_water` | 247 | **739** |

Throughput is **marginally better** with the instrumentation live, and every back-pressure counter is
in the same place. The instrument is not perturbing the system it measures; the pre-MP-1 attempt 7
produced the identical failure. `deferred_high_water` is 3× larger, consistent with a longer stage-2
before the terminal.

Back-pressure is exonerated again on its own zeroed counters: `pause_events=0`,
`capacity_timeout_terminations=0`, `capacity_invariant_terminations=0`,
`inbound_saturation_events=0`, `emergency_events_total=0`, queue peak 553 against
`bound_in_force=1113`.

---

## 4. The heartbeat question — one branch REFUTED, the rest NARROWED

MP-1's drain message-class census, taken at dequeue and independent of stripe identity:

```
frame_classes_run : sub_stripe_result 1862 · heartbeat 550 · stripe_complete 59
                    heartbeat_with_stripe 550 · heartbeat_without_stripe 0
```

**550 heartbeats reached the drain, and every one carried a `current_stripe_id`.**

> **§2.6's second branch is REFUTED.** Attempt 7's `heartbeats_accepted = 0` run-wide is **not**
> explained by stripeless heartbeats being invisible to the stripe-keyed inventory. That was the
> reading the source made plausible, and it is now measured false.

And yet `heartbeats_accepted` is **0 across all 59 `STRIPE_RX_SUMMARY` records again** — the same
zero, on a run where the heartbeats demonstrably arrived and were dequeued.

What is observed, and what is not:

| probe | level | result |
|---|---|---|
| `identity mismatch (Decision A)` | WARNING | **0 — genuinely observed** |
| `L1 fence dropped` | WARNING | **0 — genuinely observed** |
| `[F1] lease renewal REFUSED` | DEBUG | **UNOBSERVED** — the run logged 0 DEBUG lines, so this is *not* a zero |

**The remaining candidates, stated as candidates and not as findings** — MP-1 did not build the
discriminating measurement for them:

1. **The silent `wconn is None` return** in `_serve_dispatch` — the only early exit in the heartbeat
   path that emits no log line at any level.
2. **The heartbeat lands on an already-reported stripe's slot.** The worker sets `current_stripe_id`
   at `handle_stripe` (`miner/range_miner_worker.py:1922`) and clears it only on a transport-loss
   reset (`:2280`), so a heartbeat sent *between* stripes carries the **previous** stripe id — whose
   `STRIPE_RX_SUMMARY` was already emitted at its `stripe_complete`. The count would be real and
   land where nothing reports it.
3. `note_stripe_claimed` zeroing `heartbeats_accepted` at each claim.

Closing this needs one more cheap counter — heartbeat dispositions inside `_serve_dispatch`, keyed by
exit path. **Beta's question is advanced, not closed:** one branch eliminated by measurement, three
named and separable.

---

## 5. A second instrumentation gap, found by using it

**`st1_s30` — the stripe whose lease expired — has NO `STRIPE_RX_SUMMARY` record.** The only
disposition emitted all run is `stripe_complete` (59 of them). The
`lease_expired_offered_to_matrix` emission never fired, on a run whose terminal *is* a lease expiry.
Attempt 7 recorded the same absence and called it "itself the finding"; MP-1 shows it is systematic
rather than incidental.

The likely mechanism is visible in source and is **reported, not fixed**: the serve loop's expiry
report calls `expired_claimed_stripes_for_report(run_id, now)` using the **iteration's shared
`now`**, while `process_lease_expiry` reads its own fresher clock. With iterations up to
**2.983 s** long, a lease can cross its deadline inside the gap — so the report sees nothing expired
in the same pass where the matrix terminates the trial. The consequence is that the one stripe a
forensic reader most wants a lifecycle record for is the one stripe that never gets one.

*(The worst iteration's own profile: `iteration 2.983 s = drain 1.919 (staging 1.912) + expiry 1.052
+ 0.012 everything else`. Note `expiry` at 1.052 s — the lease scan is the second-largest phase in
that iteration, which is its own datum.)*

---

## 6. Statement of the finding

> The serve loop spent **681 s of a 969 s trial** blocked inside `enqueue_staging`, waiting on
> `_admission_lock`, while **four staging-executor threads held that lock for ~3,640 thread-seconds**
> inside `_pump_deferred`. The cost per message grew **~400×** over the run, from 5 ms to ~2 s. By
> the final windows each drain pass serviced **one frame, at position 1**, and **22 of 25 connections
> were measured at zero**. Both lease-renewal paths are downstream of that drain, so the compute
> lease on `st1_s30` had no renewal signal available and expired on schedule, correctly terminating
> a constant-mode trial.
>
> **H2 is not merely confirmed — it is attributed.** The drain starved because a single lock,
> contended between the serve loop and the staging pump, serialized the entire ingest path, and the
> contention grew with the deferred backlog. Servicing-order starvation is **refuted** by the
> position data. Back-pressure is exonerated by its own zeroed counters, again.

**Not claimed here:** any remedy. The candidates differ materially in their concurrency properties —
the §2.26 precedent — and the choice is Beta's. Also not claimed: why `heartbeats_accepted` is zero
(§4, narrowed to three separable candidates), and why the lease-expiry lifecycle record never emits
(§5, mechanism identified in source, unmeasured).

---

## 7. Evidence preserved

`logs/gate12_mp1_forensics/` (the `logs/` tree is gitignored as a whole directory, so none of this
dirties the worktree):

| artifact | file |
|---|---|
| coordinator + WATCHER log | `gate12_20260816_160503.log` |
| **`[S172-SL] window` series (91)** | `sl_window_series.txt` |
| **`[S172-SL] iteration_profile`** | `sl_iteration_profile.txt` |
| **`[S172-SL] summary`** | `sl_summary.txt` |
| `STRIPE_RX_SUMMARY` (59) | `stripe_rx_summary_all.txt` |
| `ACTIVE_STRIPES` (90) | `active_stripes_all.txt` |
| `[S172-BP] summary` | `bp_summary.txt` |
| saturation verdict / samples | `gate12_20260816_160503_{verdict.txt,concurrency.tsv,sampler.log}` |
| harness + gate evidence | `gate12_20260816_160503_evidence.txt`, `..._source_digests.json` |

**Verification-integrity controls (VIR-1…6)**
- execution proof: 91 window records, 90 `ACTIVE_STRIPES`, 59 `STRIPE_RX_SUMMARY`, 1,771 accounted
  iterations, sampler verdict written, watcher and sampler both confirmed exited.
- clean control: attempt 7 (same shape, no MP-1 instrumentation) as the throughput control — 1.225
  vs 1.285 jobs/s, i.e. the instrument is non-perturbing.
- fault-injection control: not applicable — this is observation of a production terminal.
- completion sentinel: `GATE-12 SATURATION VERDICT : SATISFIED`; WATCHER hard output-validation
  failure recorded; both processes exited.
- unavailable-observer behavior: per-connection positions on unserviced rows report `None`, never 0;
  `drain_pass_conns_min=0` is a measured zero from a successful read. **`[F1] lease renewal REFUSED`
  is reported as UNOBSERVED, not as zero**, because the run emitted no DEBUG records.
- audit claim scope: this run only. No claim about attempts 1-7 beyond the published attempt-7
  figures used as the throughput control.
- searched surfaces: coordinator log, MP-1 window/profile/summary series, `STRIPE_RX_SUMMARY`,
  `ACTIVE_STRIPES`, `[S172-BP] summary`, miner ledger (`mode=ro`), sampler TSV/verdict, certified
  cursor, `git status`; `miner/range_miner_coordinator.py` and `miner/range_miner_worker.py` for the
  §4/§5 mechanisms.
- unavailable surfaces: host GPU kernel log (CT100 unprivileged LXC — `UNAVAILABLE` per §2.17, never
  `PASS`); DEBUG-level coordinator records (not emitted at this log level); rig-side worker logs (not
  pulled this session).
- governance trail searched: §2.19, §2.26 (no remedy without cause), §2.27, §2.29 (frozen shape),
  §2.33 (§19/§21/§22), and Beta's MP-1 certification corrections of 2026-08-16 (the conjunction
  refuter, and the arrival-side-evidence boundary — both applied above).
