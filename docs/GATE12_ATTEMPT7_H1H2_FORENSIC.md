# GATE-12 ATTEMPT 7 — H1/H2 DISCRIMINATION FORENSIC

**Run:** `distributed_config_t1_36bf30e3` · **date:** 2026-08-16 · **launch tree:** `ae4bf85`
**Result: FAILED** at stage 2 (workflow phase 2, constant-mode reverse) on a compute-lease expiry.
**Verdict: H1 REFUTED · H2 CONFIRMED**, by direct measurement, on the first run the H1/H2
instrumentation was live for.

This report is **read-only forensics**. No remedy is proposed — per the §2.26 precedent the
choice of mechanism is Beta's.

---

## 1. What the run achieved before it died

| Beta §21 completion condition | result |
|---|---|
| truthful GPU preflight PASS | ✅ 3/3 rigs OK at 8/8 |
| 25-worker frozen admission | ✅ 25 distinct workers, admission 25 |
| `GATE-12 SATURATION VERDICT : SATISFIED` | ✅ **both** verdicts satisfied |
| all four stages complete | ❌ stage 2 terminal; stages 3-4 never assigned |
| D3.5 canonical publication | ❌ not reached |
| S145 publication-bound coverage | ❌ not reached |
| certified cursor == 2,147,483,648 | ❌ still **0** |

**Gate 12 is NOT passed.** Nothing composes from this attempt.

**The saturation verdict is nonetheless a first-in-sequence result and was earned on this run's
own samples:** peak **25** simultaneous compute-active workers with **7** queued at the same
instant, **5** qualifying windows, **4** of them showing turnover, witness = window 1
(`08:33:58 → 08:34:05`, drained 3, transitions 3). Sample census **513 OBSERVED / 0 UNOBSERVED**.
The sampler was armed before the coordinator process existed.

**Stage ledger:** phase 1 **32/32 done**; phase 2 **9 done, 23 cancelled**. Per the binding
forensic frame the 23 cancellations are the abort-cleanup footprint of **one** terminal event,
not 23 failures.

**Fail-closed cascade held end to end:** trial terminal → `provenance_validated` false →
`MinerIngressError` → Optuna trial 0 failed → no `optimal_window_config.json` → WATCHER hard
output-validation failure, confidence 0.00, human escalation. Post-run: **cursor still 0**, no
config file, `git status --porcelain` still empty.

---

## 2. The terminal event

```
08:50:53,465 [F1/F2] TRIAL TERMINAL run=distributed_config_t1_36bf30e3
  class=compute_lease_expiry stripe='…__st1_s30' worker='rrig6600b:gpu2' attempt=0
  reason=… exceeded its compute lease with no valid heartbeat or active-stripe progress;
  workflow phase 2 is CONSTANT-MODE, which fails the trial immediately and never retries
```

Per the binding frame this is **correct behaviour for a constant phase**, not a retry defect.
The question is why that worker had no accepted progress. It is answered below.

*(Box is UTC−7; worker logs are UTC, coordinator log local. `15:45:52.395Z` = `08:45:52` local.)*

---

## 3. H1 — REFUTED. The worker finished the stripe in 2.6 seconds.

`rrig6600b:gpu2`, stripe `…__st1_s30`, from the preserved rig log:

| event | UTC | key fields |
|---|---|---|
| `STRIPE_BEGIN` | 15:45:52.395 | `seed_start=2013265920`, `sub_count=34` |
| `STRIPE_COMPUTE_DONE` | 15:45:55.031 | `compute_s=2.628941`, `subs_computed=34` |
| `STRIPE_SEND_DONE` | 15:45:55.032 | `send_s=0.006782`, `substripes_sent=34` |
| `STRIPE_END` | 15:45:55.032 | **`outcome="complete"`**, `subs_sent=34`, `total_s=2.637158` |

`STRIPE_END` send-attribution, in full:

```
send_s                     0.006782      stripe_send_calls          35
stripe_send_syscall_s      0.001056      stripe_send_syscall_max_s  0.000066
stripe_send_lock_wait_s    0.000022      stripe_send_lock_wait_max_s 0.000001
stripe_send_stall_s        0.001078      heartbeat_send_syscall_s   0.0
unattributed_s             0.007139      compute_done true · send_done true
```

**Every one of the 34 sub-results plus the terminal frame left the worker inside 6.8 ms, with
1.1 ms of syscall, 22 µs of lock wait and 1.1 ms of stall.** There is no send-side stall, no
lock contention, no unattributed time of any consequence. The worker was done and idle
**298.4 seconds before** the coordinator declared its lease expired.

`SESSION_END` for that worker records `classification="explicit_shutdown"`,
`assignment_active_at_loss=false`, `reconnect_attempted=false`, and its `last` stripe accounting
is `…st1_s30 / outcome complete`. **The worker never knew anything was wrong.**

Run-wide: **`WORKER_DISCONNECTED` = 0, `WORKER_RECONNECTED` = 0.** The transport never dropped;
the Defect-A recovery path was never invoked.

---

## 4. H2 — CONFIRMED. 45 frames arrived and not one was ever drained.

Coordinator `ACTIVE_STRIPES` snapshot at **08:50:48.869**, 4.6 s before the expiry:

```json
{"stripe_id": "…__st1_s30", "worker_id": "rrig6600b:gpu2", "attempt": 0,
 "claim_precision": "exact",
 "frames_enqueued": 45, "frames_dequeued": 0, "frames_pending": 45,
 "frames_received": 0,  "frames_deferred": 0, "subresults_accepted": 0,
 "heartbeats_accepted": 0,
 "age_since_claim_s": 296.477, "oldest_pending_age_s": 296.396,
 "age_since_last_accepted_frame_s": null,
 "age_since_last_accepted_progress_s": null,
 "age_since_last_subresult_s": null,
 "residency_max_s": null, "lease_remaining_s": 3.522}
```

**Counter semantics, read from source, not inferred from the names**
(`miner/range_miner_coordinator.py:5722-5818`):

- `frames_enqueued` — *"A frame for this stripe was **DECODED AND ACCEPTED ONTO `inbound`**…
  This is ARRIVAL, a different event from acceptance, and **the gap against `frames_dequeued`
  IS the coordinator-side backlog**."*
- `frames_dequeued` — *"left `inbound` and reached the drain."*

So: **45 frames were decoded off the socket and placed on `inbound`, and the drain took none of
them for the entire 296-second lease.** `oldest_pending_age_s` 296.396 against
`age_since_claim_s` 296.477 means the first frame landed ~80 ms after the claim and sat there
until the lease died. `age_since_last_accepted_frame_s` is **`null`** — not "stale", *never*.

**This signature cannot be a phantom of the enqueue/dequeue race.** That exact false positive —
`enqueued=1 · dequeued=0 · pending=1` for an already-processed frame — is named in the R2-1
docstring as *"the exact H2b signature this instrument exists to prove"*, and was repaired by
per-frame token reconciliation that converges under either interleaving. `frames_untokened` is
**0** across the run, so every frame was reconcilable.

---

## 5. Why no renewal saved it — both renewal paths run through the starved queue

F1/F2's contract is *heartbeat **OR** accepted active-stripe progress renews the active attempt*.

- **Progress renewal:** requires acceptance; acceptance requires dequeue; dequeue was 0. Dead.
- **Heartbeat renewal:** **`heartbeats_accepted` totals `0` across all 60 `STRIPE_RX_SUMMARY`
  records in the run.** Not merely for s30 — **no heartbeat was accepted for any stripe, at any
  point.** Heartbeats are enqueued onto the same `inbound` under the same stripe id (the 45
  enqueued frames exceed the worker's 35 sent frames by ~10, consistent with heartbeats also
  arriving and also never draining).

**Both renewal routes are downstream of the same drain.** When the drain starves, the lease has
no surviving renewal path, and the 300 s clock from claim runs unopposed. The lease behaved
exactly as designed; what failed is that the thing it measures had become unobservable to it.

---

## 6. It is not the S172-BP back-pressure path

Terminal summary:

```
inbound_qsize_high_water=547   derived_bound=1113   bound_in_force=1113
paused_high_water=0            pause_events=0       pause_seconds_total=0.000
capacity_timeout_terminations=0  capacity_invariant_terminations=0
inbound_saturation_seconds_total=0.000  inbound_saturation_events=0
emergency_events_total=0       staging_jobs_completed=1250  staging_jobs_per_sec=1.225
```

The queue peaked at **547 against a bound of 1113**; nothing was ever paused, deferred to
timeout, or refused on capacity. `frames_deferred=0` on s30 confirms it locally — those frames
were never dequeued far enough to *be* deferred. **The pause/resume credit machinery is
exonerated; it never engaged.**

The one anomalous throughput figure is **`staging_jobs_per_sec=1.225`** (1,250 jobs). Attempt 1
sustained 3.055/sec and attempt 2 completed 3,948 jobs. This run's drain was running roughly
**2.5× slower** than attempt 1's while the inbound queue sat half-full and unpaused.

---

## 7. The failure is a tail effect, and it was building all run

Stripes reaching `frames_dequeued == 0` with `age_since_claim_s > 60` at any snapshot — **7 of
64**, and every one is a late-index stripe:

| stripe | worker | last seen | age_since_claim | enqueued |
|---|---|---|---|---|
| `st0_s31` | `rrig6600:gpu2` | 08:39:08 | 281.5 s | 44 |
| `st1_s26` | `zeus-ubuntu-vm:gpu0` | 08:47:15 | 114.3 s | 19 |
| `st1_s27` | `rrig6600c:gpu7` | 08:47:47 | 129.6 s | 39 |
| `st1_s28` | `rrig6600c:gpu2` | 08:49:14 | 214.4 s | 42 |
| `st1_s29` | `rrig6600c:gpu0` | 08:49:14 | 214.4 s | 42 |
| **`st1_s30`** | **`rrig6600b:gpu2`** | 08:50:48 | **296.5 s** | 45 |
| `st1_s31` | `rrig6600c:gpu4` | 08:50:48 | 276.8 s | 44 |

**`st0_s31` is the near-miss that proves the mechanism was already live in stage 1**: same
zero-dequeue state at 281.5 s, ~18 s of lease left, and it recovered — stage 1 completed 32/32.
Stage 2 ran the same shape and one stripe crossed the line.

The final snapshot shows the split cleanly. `s28`/`s29` were draining slowly (20/19 dequeued,
20/19 accepted) and therefore held **297 s / 295 s of lease remaining** — renewal working.
`s30`/`s31` had drained nothing and held **3.5 s / 23.2 s**. Same instant, same stage, same
fleet: **the stripes that were being drained kept their leases; the stripes that were not, died.**

Healthy stripes were not comfortable either — a completed stage-1 stripe shows
`age_since_claim_s=248.8` with `residency_max_s=233.5` and `frames_deferred=34` against a 300 s
lease. **The run was operating inside ~50 s of margin throughout.**

---

## 8. Statement of the finding

> The worker delivered a complete, correct stripe into the coordinator's `inbound` queue in
> 2.64 seconds with 6.8 ms of send time and full attribution. The coordinator's drain removed
> none of those 45 frames for 296 seconds. Because both lease-renewal paths — accepted progress
> and heartbeat — are downstream of that drain, the compute lease had no renewal signal
> available, expired on schedule, and correctly terminated a constant-mode trial that had
> nothing wrong with it.
>
> **H1 (worker-side send stall) is refuted by direct measurement. H2 (coordinator-side ingest
> backlog) is confirmed.** The back-pressure/pause path is exonerated by its own zeroed
> counters; the queue never approached its bound. What starved was the drain, not the queue.

**Not claimed here:** the cause of the drain starvation, why `heartbeats_accepted` is zero
run-wide, and whether that zero is a second defect or a consequence of the first. Those need
the drain's own scheduling read, which this report did not do.

---

## 9. Evidence preserved

| artifact | path |
|---|---|
| coordinator + WATCHER log | `logs/gate12_20260816_083211.log` |
| harness/gate evidence | `logs/gate12_20260816_083211_evidence.txt` |
| saturation verdict | `logs/gate12_20260816_083211_verdict.txt` |
| concurrency samples (513) | `logs/gate12_20260816_083211_concurrency.tsv` |
| **24 rig worker logs** | `logs/gate12_a7_forensics/rig_worker_logs/{rrig6600,rrig6600b,rrig6600c}/` |
| `ACTIVE_STRIPES` extract (95) | `logs/gate12_a7_forensics/active_stripes_all.txt` |
| `STRIPE_RX_SUMMARY` extract (60) | `logs/gate12_a7_forensics/stripe_rx_summary_all.txt` |
| s30 coordinator record | `logs/gate12_a7_forensics/s30_rx_summary.txt` (empty — **s30 never reached a lifecycle summary**, which is itself the finding) |
| source digests / GPU / liveness | `logs/gate12_20260816_083211_{source_digests,liveness}.json` |

Rig logs were pulled before any subsequent clean slate could remove them. `logs/` is ignored as
a whole directory, so none of this dirties the tree.

**Verification-integrity controls (VIR-1…6)**
- execution proof: 513 sampler rows, 95 `ACTIVE_STRIPES`, 60 `STRIPE_RX_SUMMARY`, 24 rig logs retrieved with `scp` exit 0 per host
- clean control: `st0_s31` (same zero-dequeue state, recovered) and `s28`/`s29` (draining, leases renewed) in the same snapshot
- fault-injection control: not applicable — this is observation of a production terminal, not a detector under test
- completion sentinel: WATCHER pipeline summary + sampler verdict file both written; both processes confirmed exited
- unavailable-observer behavior: sampler census reports 0 UNOBSERVED of 513; ESTAB 0 UNAVAILABLE; rig hostnames self-reported per probe
- audit claim scope: this run only; no claim about attempts 1-6
- searched surfaces: coordinator log, 24 rig worker logs, miner ledger (`mode=ro`), sampler TSV/verdict, `miner/range_miner_coordinator.py` counter definitions
- unavailable surfaces: host GPU kernel log (CT100 unprivileged LXC — `UNAVAILABLE`, per §2.17, never `PASS`); drain-thread scheduling internals (not instrumented)
- governance trail searched: §2.19 (S172-BP), §2.26 (F-1 lease forensics), §2.27 (F1/F2 renewal contract), §2.32 (Defect A/B), §2.33 (Beta §19/§21/§22)
- chapters searched: n/a — no design-intent claim is made
