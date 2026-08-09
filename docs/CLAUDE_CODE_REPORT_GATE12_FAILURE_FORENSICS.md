# CLAUDE CODE REPORT — GATE-12 FAILURE FORENSICS (READ-ONLY)

**Host:** VM101 · repo `~/distributed_prng_analysis` · HEAD `a3bb4da`
**Run:** `distributed_config_t1_689f3cd9` · 2026-08-09 12:37:09 → 12:47:17
**Authority:** Team Beta *"GATE-12 FAILED EXECUTION"* (2026-08-09) — read-only forensics only.
**Nothing was launched, no fleet work was performed, no production file was edited, no commit, no push.**

---

## 0. Forensic-copy identity

Copies made under the session scratchpad; originals left in place and re-hashed after copying to
prove they were not mutated.

| artifact | original path | sha256 | size / mtime |
|---|---|---|---|
| ledger | `/home/michael/miner_staging/miner_ledger.db` | `875145b9bda3db309e5fd87db212d3ef9aed5342cb437e846a845353ac2d039a` | 2,330,624 B · 12:47:17.388 |
| ledger WAL | `…/miner_ledger.db-wal` | `e3b0c44298fc…7852b855` (**empty** — db fully checkpointed, copy is complete) | 0 B · 12:47:28.408 |
| ledger SHM | `…/miner_ledger.db-shm` | `fd4c9fda9cd3f9ae7c962b0ddf37232294d55580e1aa165aa06129b8549389eb` | 32,768 B · 13:40:20 |
| coordinator log | `logs/gate12_20260809_123705.log` | `36b52254891c72bfcc0fcbf2c9d06e12828a1df2cee01cccfbe983f83df85861` | 21,734 B · 12:47:17.818 |
| evidence file | `logs/gate12_20260809_123705_evidence.txt` | `338e60a6493022ac776817c05b530a907b6ab978c22cb4fa1d77748fbcaa0ea7` | 2,192 B |
| concurrency TSV | `logs/gate12_20260809_123705_concurrency.tsv` | `252a844b…` at first read → `08b15440…` at copy (**still growing**, see below) | — |
| launch script | `gate12_launch.sh` | `398fa233e6e106827508b3bb08bcf31316a32485800babae24863f744fdda0b0` | 3,586 B |
| rig worker logs | `rrig6600:/tmp/minerlogs/gpu{0..6}.log` | `c46728c7908d28dd179f68a275a1a0ded45e6b1e4fbae7455422ea737d1aaedf` (all 7 identical) | 90 B · 12:42:05 |
| rig worker log | `rrig6600:/tmp/minerlogs/gpu7.log` | `e3b0c442…7852b855` (empty) | 0 B |

Copies: `<scratchpad>/forensic/{miner_ledger.db.copy, gate12.log.copy, gate12_evidence.txt.copy,
gate12_concurrency.tsv.copy, gate12_launch.sh.copy}`. All ledger queries below were run against
`file:…/miner_ledger.db.copy?mode=ro`. The ledger original was re-hashed after copying and is
byte-identical.

> **⚠ LIVE PROCESS TOUCHING EVIDENCE — reported, not acted on.** `gate12_launch.sh` is **still
> running** (PID 42253, started 12:47:27, reparented to init). It is the step-4 sampler subshell
> (`gate12_launch.sh:60-77`), looping `for i in $(seq 1 1440)` at 5 s — it will run until ≈14:51 and
> is **still appending rows to the concurrency TSV**, which is why that file hashed differently
> between two consecutive commands. It opens the ledger `mode=ro` only
> (`gate12_launch.sh:66`), so the ledger content is untouched (hash stable across two reads); it
> does update the `-shm` mtime. **I did not kill it** — that is an action, not forensics. Its rows
> are post-mortem only (§E). Killing it before further analysis is your call.

---

## A. STAGE-2 CAUSAL RECONSTRUCTION

### A.1 The initiating event

> **FIRST AUTHORITATIVE TERMINAL EVENT — compute-lease expiry in a constant workflow phase.**
>
> | field | value |
> |---|---|
> | **classification** | **lease expiry** (from the brief's list) |
> | **exact timestamp** | **12:47:13.143** (`trials.finalized_at = 1786304833.142975`) |
> | **stripe** | `distributed_config_t1_689f3cd9__st1_s5` |
> | **worker / attempt** | `rrig6600:gpu4` / attempt **0** |
> | **lease deadline breached** | 12:47:05.487 (`lease_expires_at = 1786304825.487252`) |
> | **detecting branch** | `miner/range_miner_coordinator.py:6367` → `:5186` → `:5205` |
> | **deciding branch** | `miner/range_miner_coordinator.py:5106-5107` — `if phase in (1, 2): self.fail_trial(...)` |
> | **abort branch** | `:5348` `fail_trial` → `:5405` `cancel_active_stripes` |

**Per Beta §8 this is CORRECT BEHAVIOUR, not a retry defect.** Phase 2 is constant-mode; the
matrix at `:5106` is specified to fail the trial immediately, and the reassign-to-another-worker
path at `:5111-5136` is reachable only for phases 3/4. **No retry was permitted and none was
attempted. I do not report "retry failed."** The remaining question is therefore *why s5's lease
expired*, which is §A.4.

### A.2 Why it is lease expiry and not something else

1. **The stripe rows carry the signature of this path and no other.** `cancel_active_stripes`
   (`:1546-1556`) updates **`state` only** — it preserves `claimed_by` and `lease_expires_at`.
   `_handle_stripe_failure_locked` likewise mutates nothing on the stripe before calling
   `fail_trial`. So a stripe killed by the constant-phase lease-expiry path ends as
   `state='cancelled'` **with `claimed_by` still set and `lease_expires_at` still populated** —
   which is exactly what s5, s7 and s9 show. (Contrast `reclaim_expired_leases` at `:1663-1697`,
   which *does* NULL both; it is a different method and it did not run.)
2. **The expired set at the abort instant was exactly {s5, s7, s9}, and s5 sorts first.**
   `expired_claimed_stripes` (`:1533-1544`) selects `state='claimed' AND lease_expires_at < now
   ORDER BY stripe_id`. At 12:47:13.143 the other three cancelled stripes had already left
   `claimed`: s3, s6 and s26 all carry `stripe_complete_seen=1` and `lease_expires_at IS NULL`,
   which only `record_stripe_complete` (`:1793-1806`) produces, and it moves the stripe to
   `staging`. Lexicographic order over `…__st1_s5` < `…__st1_s7` < `…__st1_s9` makes **s5** the one
   stripe that reached `handle_stripe_failure`; s7 and s9 followed in the same loop and returned
   `{"action": "noop", "reason": "trial already terminal"}` (`:5093-5094`).
3. **Every other terminal path in `serve_trial` is excluded by positive evidence:**

| candidate path | anchor | excluded because |
|---|---|---|
| `serve_trial timeout` | `:6017-6019` | `trial_timeout` is `serve_timeout=None` by design (skill §2.6) |
| staging capacity timeout | `:6027-6034` | run log: `capacity_timeout_terminations=0`; also logs `logger.error` — absent |
| capacity invariant | §2.19 | run log: `capacity_invariant_terminations=0` |
| worker admission timeout | `:6193-6203` | stage 2 *was* assigned (`[S172-BP] derived_bound … phase=2` at 12:42:05.645) |
| no eligible worker for variant | `:6220-6223` | same — assignment succeeded, 32 stripes created |
| retention sizing / provenance | `:6252-6275` | preflight logged `mode=derived required=6528 resolved=6528` and admitted |
| staging sizing failed closed | `:6329-6349` | logs `logger.exception` — absent; `burst_exact` logged normally |
| threshold-provenance abort | `:6409` | unreachable — requires `stage_idx >= len(workflow_stages)`; stage 2 never completed |
| explicit trial abort | — | no operator action; §5 of the evidence package records zero intervention |
| coordinator exception | — | no traceback in the coordinator log; the only traceback is the downstream `MinerIngressError` |
| **StripeErrorMessage** | `:6987-6990` | **see the caveat below** |

4. **`StripeErrorMessage` — not excluded with certainty, but no positive evidence for it, and the
   classification is unchanged either way.** A `stripe_error` is dispatched in the same serve-loop
   iteration's drain, *above* `process_lease_expiry`, so one arriving in that batch would have
   fired first. Against that: (a) all three incomplete streams truncate together at 12:47:11.338 /
   12:47:12.056 / 12:47:12.607, ~0.5–1.8 s before the abort — the shape of a socket teardown, not
   of one worker erroring while two others carry on; (b) the seven rig worker logs are **90 bytes,
   byte-identical, and contain only two kernel-compile lines** — no error text on any GPU;
   (c) `_fail_stripe` (`range_miner_worker.py:1351-1364`) sets `state="idle"` and stops that
   stripe, which would have truncated one stream noticeably earlier. **What would settle it:** a
   coordinator-side log line on the `stripe_error` branch (there is none today), or worker-side
   logging in `_fail_stripe` (there is none today). **Cannot be determined from the available
   evidence** — but note that a `stripe_error` in phase 2 lands on the *same* line `:5106-5107` and
   produces the same immediate constant-phase failure, so the disposition in §F does not turn on it.

### A.3 All 32 stage-2 stripes

`created_at` is identical (`1786304525.487252` = **12:42:05.487**) for all 32 rows — see §A.4.
`elapsed_s` is the **worker-reported service time**; `ingest_span` is the coordinator-side interval
between this stripe's first and last recorded sub-stripe result. The ledger has **no per-transition
timestamp columns** (no `claimed_at`, no `cancelled_at`), so claim time and cancellation time are
derived as stated, not read.

| stripe | seed_start | worker | att | state | exp/done | SC seen | elapsed_s (worker) | first shard | last shard | ingest span | lease_expires_at |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s0 | 0 | zeus:gpu0 | 0 | done | 14/14 | 1 | 0.96 | 12:42:05.767 | 12:42:06.618 | 0.9 s | NULL |
| s1 | 67108864 | rrig6600:gpu0 | 0 | done | 34/34 | 1 | 6.42 | 12:42:06.529 | 12:42:32.429 | 25.9 s | NULL |
| s2 | 134217728 | rrig6600:gpu1 | 0 | done | 34/34 | 1 | 5.47 | 12:44:15.735 | 12:45:58.841 | 103.1 s | NULL |
| **s3** | 201326592 | rrig6600:gpu2 | 0 | **cancelled** | 34/34 | 1 | 5.30 | 12:45:58.732 | 12:47:03.256 | 64.5 s | NULL |
| s4 | 268435456 | rrig6600:gpu3 | 0 | done | 34/34 | 1 | 5.72 | 12:45:47.421 | 12:47:03.103 | 75.7 s | NULL |
| **s5** | 335544320 | **rrig6600:gpu4** | 0 | **cancelled** | 34/**31** | **0** | — | 12:46:14.178 | 12:47:12.056 | 57.9 s | **12:47:05.487** |
| **s6** | 402653184 | rrig6600:gpu5 | 0 | **cancelled** | 34/34 | 1 | 5.23 | 12:45:59.333 | 12:47:09.496 | 70.2 s | NULL |
| **s7** | 469762048 | **rrig6600:gpu6** | 0 | **cancelled** | 34/**28** | **0** | — | 12:46:25.617 | 12:47:12.607 | 47.0 s | **12:47:05.487** |
| s8 | 536870912 | zeus:gpu0 | 0 | done | 14/14 | 1 | 0.90 | 12:42:09.064 | 12:42:10.568 | 1.5 s | NULL |
| **s9** | 603979776 | **rrig6600:gpu0** | 0 | **cancelled** | 34/**30** | **0** | — | 12:46:05.895 | 12:47:11.338 | 65.4 s | **12:47:05.487** |
| s10 | 671088640 | rrig6600:gpu1 | 0 | done | 34/34 | 1 | 5.60 | 12:42:06.110 | 12:42:17.073 | 11.0 s | NULL |
| s11 | 738197504 | rrig6600:gpu2 | 0 | done | 34/34 | 1 | 5.16 | 12:42:06.479 | 12:42:15.620 | 9.1 s | NULL |
| s12 | 805306368 | rrig6600:gpu3 | 0 | done | 34/34 | 1 | 5.35 | 12:42:06.116 | 12:42:16.021 | 9.9 s | NULL |
| s13 | 872415232 | rrig6600:gpu4 | 0 | done | 34/34 | 1 | 6.66 | 12:42:06.556 | 12:42:41.164 | 34.6 s | NULL |
| s14 | 939524096 | rrig6600:gpu5 | 0 | done | 34/34 | 1 | 6.10 | 12:42:06.443 | 12:42:18.620 | 12.2 s | NULL |
| s15 | 1006632960 | rrig6600:gpu6 | 0 | done | 34/34 | 1 | 5.49 | 12:42:06.313 | 12:42:16.536 | 10.2 s | NULL |
| s16 | 1073741824 | zeus:gpu0 | 0 | done | 14/14 | 1 | 0.91 | 12:42:06.730 | 12:42:07.711 | 1.0 s | NULL |
| s17 | 1140850688 | rrig6600:gpu0 | 0 | done | 34/34 | 1 | 5.43 | 12:42:36.856 | 12:44:16.455 | 99.6 s | NULL |
| s18 | 1207959552 | rrig6600:gpu1 | 0 | done | 34/34 | 1 | 5.38 | 12:42:17.145 | 12:44:14.825 | 117.7 s | NULL |
| s19 | 1275068416 | rrig6600:gpu2 | 0 | done | 34/34 | 1 | 6.18 | 12:42:15.992 | 12:44:15.777 | 119.8 s | NULL |
| s20 | 1342177280 | rrig6600:gpu3 | 0 | done | 34/34 | 1 | 5.34 | 12:42:17.058 | 12:44:14.336 | 117.3 s | NULL |
| s21 | 1409286144 | rrig6600:gpu4 | 0 | done | 34/34 | 1 | 5.32 | 12:42:47.155 | 12:44:22.260 | 95.1 s | NULL |
| s22 | 1476395008 | rrig6600:gpu5 | 0 | done | 34/34 | 1 | 4.66 | 12:42:27.992 | 12:44:14.367 | 106.4 s | NULL |
| s23 | 1543503872 | rrig6600:gpu6 | 0 | done | 34/34 | 1 | 6.28 | 12:42:17.109 | 12:44:15.925 | 118.8 s | NULL |
| s24 | 1610612736 | zeus:gpu0 | 0 | done | 14/14 | 1 | 0.92 | 12:42:07.738 | 12:42:08.905 | 1.2 s | NULL |
| s25 | 1677721600 | rrig6600:gpu0 | 0 | done | 34/34 | 1 | 5.32 | 12:44:22.982 | 12:45:59.620 | 96.6 s | NULL |
| **s26** | 1744830464 | rrig6600:gpu1 | 0 | **cancelled** | 34/34 | 1 | 4.46 | 12:45:58.873 | 12:47:02.353 | 63.5 s | NULL |
| s27 | 1811939328 | rrig6600:gpu2 | 0 | done | 34/34 | 1 | 4.92 | 12:44:15.815 | 12:45:56.561 | 100.7 s | NULL |
| s28 | 1879048192 | rrig6600:gpu3 | 0 | done | 34/34 | 1 | 4.77 | 12:44:14.383 | 12:45:47.365 | 93.0 s | NULL |
| s29 | 1946157056 | rrig6600:gpu4 | 0 | done | 34/34 | 1 | 5.58 | 12:44:28.622 | 12:46:10.998 | 102.4 s | NULL |
| s30 | 2013265920 | rrig6600:gpu5 | 0 | done | 34/34 | 1 | 5.82 | 12:44:14.798 | 12:45:59.063 | 104.3 s | NULL |
| s31 | 2080374784 | rrig6600:gpu6 | 0 | done | 34/34 | 1 | 6.15 | 12:44:19.856 | 12:46:20.956 | 121.1 s | NULL |

**Not available from the ledger for any row:** a claim timestamp, a heartbeat series, a
StripeComplete timestamp, a cancellation timestamp, or a per-stripe connection-loss record. The
ledger stores no state-transition history. Everything above is either a stored column or the
coordinator's own write time on the `shards` rows.

### A.4 Why s5/s7/s9 had an expired lease — the mechanism

**All 32 stage-2 stripes were claimed at ONE instant, each with the same 300 s deadline.**
`assign_stripes` (`:2680-2705`) loops over the whole macro-stripe partition and, for each stripe,
calls `claim_stripe(..., now + self.config.compute_lease_timeout)` (`:2695`) with a **single `now`
for the entire loop**, round-robin over `compatible[i % len(compatible)]` (`:2688`).
`compute_lease_timeout` is `300.0` (`:245`). The ledger confirms it exactly: all 32 rows carry
`created_at = 1786304525.487252` and the three survivors of the batch carry
`lease_expires_at = 1786304825.487252` — `created_at + 300.000000`, identical to the microsecond.

**With 8 workers and 32 stripes, each worker holds 4 stripes but can work only one at a time.**
`_dispatch_pending` (`:7004-7014`) sends a `StripeAssign` for **every** claimed-and-undispatched
stripe immediately, and `range_miner_worker.serve_forever` (`:1425-1431`) processes messages
serially — `handle_stripe` runs inline on the control loop. So each worker's four assignments queue
on the wire and are executed one after another. The per-worker execution order is visible in the
table (e.g. `rrig6600:gpu0` → s1, s17, s25, **s9**).

**The lease clock therefore runs against the whole stage, not against the stripe's own service.**
A worker's 4th stripe does not begin until ~230–260 s after its lease started, and the three that
had not finished by 12:47:05.487 were **actively streaming results at that moment** — s9's last
shard landed at 12:47:11.338, s5's at 12:47:12.056, s7's at 12:47:12.607. **These were not dead
workers.** The lease, whose stated purpose is "reclaim expired COMPUTE leases" from workers that
have stopped (`:1663-1667`), expired on three healthy, productive workers.

**The stage could not fit inside the lease at the throughput actually achieved:**

| | phase 1 (`java_lcg`) | phase 2 (`java_lcg_reverse`) |
|---|---|---|
| stage assigned | 12:37:37.462 | 12:42:05.487 |
| lease deadline (assign + 300 s) | 12:42:37.462 | **12:47:05.487** |
| sub-stripe results expected | 1,008 | 1,008 |
| delivered before the deadline | **1,008** (last 12:41:33.412) | **995** (last 12:47:12.607) |
| delivery rate | **4.31 shards/s** | **3.24 shards/s** |
| margin against the lease | **+64 s** | **−11 s** (1,008 ÷ 3.24 ≈ 311 s) |

Phase 1 passed with ~21 % margin at 4.31 shards/s. Phase 2 ran 25 % slower and went over budget by
about eleven seconds. **The two phases are geometrically identical** — same 32 stripes, same 1,008
sub-stripes, same cohort. Only the delivery rate differed. *Why it differed is not established by
this evidence;* the natural candidate is that nothing is released before commit (§2.19 / the
staging-capacity amendment), so phase 2's staging executor works against every phase-1 file still
retained, but **I have no measurement isolating that and do not assert it.**

**Worker compute is not the bottleneck and is nowhere near the lease.** Worker-reported `elapsed_s`
is **4.46–6.66 s per stripe** on every rig stripe in both phases. The coordinator-side ingest span
for the same stripes reaches **95–121 s**. The 300 s lease was consumed almost entirely by result
delivery and staging, not by GPU work.

**No heartbeat ever renewed s5, s7 or s9 — this is proven, its cause is not.** `renew_lease`
(`:1648-1661`) is reached only from the heartbeat branch (`:6894-6901`) and renews **only
`msg.current_stripe_id`**. The worker heartbeats every 30 s with its current stripe
(`range_miner_worker.py:1266-1278`), so s9 — worked from ~12:46:05 — should have been renewed at
least twice before 12:47:05.487. It was not: its `lease_expires_at` is still the original
assignment value to the microsecond. Two mechanisms are consistent with the evidence and **I cannot
separate them**:

- **TCP ordering.** Results and heartbeats share one ordered stream. The reader's own design note
  (`:6549-6552`) states that frames behind a held result "stay on the wire — that is the point of
  the design, and it is precisely why the §1.4 lease exemption has to exist." At 3.24 shards/s a
  heartbeat queued behind ~30 result frames is delivered tens of seconds late.
- **Send-lock contention on the worker.** `_send` holds `self._send_guard`
  (`range_miner_worker.py:1260-1263`) and `send_msg` holds `_send_lock` around a blocking
  `_sendall` with no socket timeout (`:1120-1126`). While the mining thread is parked on a full TCP
  buffer, the heartbeat thread waits.

**Either way the §1.4/F2 lease exemption did not apply, and could not have.** It keys strictly on
membership in `_paused_connections` (`:5185-5196`), and the run log records
`paused_high_water=0 pause_events=0` — **the coordinator never paused a connection.** The exemption
covers *coordinator-initiated* silence; this was ordinary back-pressure silence, which the
exemption is not written to cover.

### A.5 Proof that the six cancellations were abort-cleanup consequences

`cancel_active_stripes` (`:1546-1556`) is called exactly once, from `abort_trial:5405`, under
`if first:`. It sets `state='cancelled'` for every stripe in `pending`, `claimed` or `staging` and
leaves `done`/`failed`/`cancelled` alone. The six divide cleanly into the two states that existed
at 12:47:13.143, and the shard table confirms each:

| group | stripes | state at abort | evidence |
|---|---|---|---|
| **A — in `staging`** | s3, s6, s26 | `staging` | `stripe_complete_seen=1`, `lease_expires_at IS NULL` (only `record_stripe_complete:1793-1806` produces this, and it transitions `claimed`→`staging`). They were not `done` because `finalize_stripe:2748-2751` promotes to `done` only when every shard is verified — s3 had **28/34 verified, 6 pending**; s6 and s26 had **0/34 verified**. |
| **B — in `claimed`** | s5, s7, s9 | `claimed` | `stripe_complete_seen=0`, `claimed_by` set, `lease_expires_at` populated and in the past. Substripes received 31/34, 28/34, 30/34. |

**None of the six was independently caused.** Group B is the initiating condition itself (s5) plus
the two that the same scan swept up as `noop`. Group A was complete work that had not finished
staging — at the abort the pipeline was **163 sub-stripe files behind** (`staging_status='pending'`
on 163 phase-2 shards). All **1,844 reservations are `released`** and
`trials.abort_cleanup_status='done'`, so the L7 discharge completed normally.

**The 26 `done` stripes are real completed work** and were untouched by the abort.

### A.6 Timeline, forward

```
12:37:13.867  [ADMISSION] expected_workers=8 (source=execution_set(d89834f1bf26))
12:37:15-37   8 cohort workers register (zeus:gpu0, rrig6600:gpu0-6)
12:37:37.311  cohort FROZEN, 4 stages x 8 identities
12:37:37.312  retention preflight: mode=derived required=6528 resolved=6528
12:37:37.462  STAGE 1 assigned — 32 stripes claimed at one instant, lease deadline 12:42:37.462
12:37:39.305  first phase-1 sub-stripe result
12:41:08.054  Defect-6 drop of an UNREGISTERED connection            <- see §B
12:41:33.412  last phase-1 sub-stripe result (1008/1008, +64 s margin)
12:41:33.6-7  12 further workers register (pool -> 22); correctly EXCLUDED by the freeze
12:42:05.487  STAGE 2 assigned — 32 stripes claimed at one instant, lease deadline 12:47:05.487
12:42:05.645  [S172-BP] derived_bound phase=2 eligible_workers=22 bound=1110
12:42:05.767  first phase-2 sub-stripe result
...           delivery proceeds at 3.24 shards/s; each worker works its 4 stripes serially
12:47:02.353  s26 completes (34/34)      -> staging
12:47:03.103  s4  completes (34/34)      -> staging -> done
12:47:03.256  s3  completes (34/34)      -> staging
12:47:05.487  *** leases of s5, s7, s9 expire while all three are actively streaming ***
12:47:09.496  s6  completes (34/34)      -> staging
12:47:11.338  s9  sub_stripe_result #29  <- event immediately before the terminal event
12:47:12.056  s5  sub_stripe_result #30
12:47:12.607  s7  sub_stripe_result #27  <- LAST ledger write before the abort
12:47:13.143  *** process_lease_expiry:5186 -> handle_stripe_failure:5205 ->
                  _handle_stripe_failure_locked:5106 (phase in (1,2)) -> fail_trial:5107
                  -> abort_trial -> mark_trial_aborted + cancel_active_stripes:5405  ***
              trials.state='aborted', finalized_at=1786304833.142975
              1,844 reservations released, abort_cleanup_status='done'
12:47:1x      serve_trial `finally`: MinerShutdown to each worker, sockets shut down
12:47:17.448  [S172-BP] summary (the first and only log line after 12:42:05)
12:47:17.458  MinerIngressError raised — validated=False, because stage_idx never
              reached len(workflow_stages) and the gate at :6409 never ran
```

### A.7 Observability finding (secondary, but it is why this took a ledger to answer)

**A constant-phase trial abort is completely silent in the coordinator log.**
`_handle_stripe_failure_locked:5106-5107` emits no log record; neither does `fail_trial:5342-5348`,
`abort_trial:5350-5423`, nor `cancel_active_stripes`. `process_lease_expiry` logs **only** the two
*skip* cases (`lease_exempt:5188`, `lease_grace:5197`) — the path that actually terminates a trial
logs nothing. That is why the run log has **nothing at all between 12:42:05.645 and 12:47:17.448**
and why the operator saw only the downstream `MinerIngressError`. The `reason` string
(`"…__st1_s5: constant-phase failure"`) is constructed, passed to `abort_trial`, placed in the event
dict at `:5406-5407` — and then dropped: `trials` persists `abort_event_id` but has **no reason
column**, and no sink logged it. Compare the capacity-timeout path at `:6031-6032`, which does
`logger.error` before failing. **This is an asymmetry in the same function.**

---

## B. THE 15-SECOND "DEFECT 6" CONNECTION DROP

**Classification: UNRELATED.** Two independent proofs, neither of which requires identifying the
connection.

1. **It was an unregistered connection, by construction.** The branch that emits that exact string
   is `:6127-6141`, and its loop body opens with `if meta["registered"]: continue`. A connection
   that has never completed a `register` frame has no `worker_id` in `worker_by_sock`, is not in
   `wconn_by_worker`, and therefore **cannot hold any stripe claim**: `assign_stripes` claims only
   to members of `eligible` (`= wconn_by_worker.values()`, `:6003-6004`).
2. **No stage-2 stripe existed at 12:41:08.** The 32 phase-2 stripe rows were created at
   **12:42:05.487**, 57 seconds later. There was nothing for it to hold.

Answering the seven questions as asked:

| # | question | answer |
|---|---|---|
| 1 | which `worker_id` owned that TCP connection? | **Cannot be determined from the available evidence.** The connection never registered, so no `worker_id` was ever bound to it; the log line carries no peer address (`:6136-6138` formats only the deadline); the ledger stores nothing for unregistered sockets. *What would be needed:* the peer address in that log line, or a packet/`ss` capture from 12:40:53–12:41:08. |
| 2 | was it part of the frozen 8-worker cohort? | **No.** All 8 cohort members (`zeus-ubuntu-vm:gpu0`, `rrig6600:gpu0-6`) registered between **12:37:15.301 and 12:37:37.305**, so their `conn_meta` entries had `registered=True` and were skipped by the `continue`. All 8 continued delivering sub-stripe results without interruption through 12:47:12.607. |
| 3 | did it hold a stage-2 stripe? | **No** — see proof 2 above. No stage-2 stripe existed. |
| 4 | did the disconnect classify a stripe as failed, or did a later lease expiry? | **Neither, for this connection.** `_drop_conn` evicts the socket from the eligible pool; it does not touch stripe state. The trial's terminal event was the lease expiry at 12:47:13.143 on a *different*, fully-registered worker (`rrig6600:gpu4`). |
| 5 | exact source branch | `miner/range_miner_coordinator.py:6127-6141` (the `read_deadline` sweep over `conn_meta`, `read_deadline = 15.0`). |
| 6 | was `TrialAbort` emitted from that branch? | **No.** That branch calls only `_drop_conn` + dict cleanup. It contains no `fail_trial`, no `abort_trial`, no `submit_abort`. |
| 7 | did the six cancellations occur after that abort? | The six cancellations occurred at **12:47:13.143**, 6 min 5 s after the 12:41:08.054 drop, and were produced by `cancel_active_stripes` under the lease-expiry abort (§A.5) — **not** by this branch. |

**Context, not causation.** Between 12:37:46.838 and 12:41:33.645 the registry shows a 3¾-minute gap,
then **12 workers registering inside 100 ms**. The dropped connection sits in that gap. The
plausible reading is that a burst of late fleet connections exceeded the 15 s registration deadline
for one socket while the coordinator was saturated draining phase-1 results — and that **22 of 25**
workers ever registered (`rrig6600b:gpu2`, `rrig6600c:gpu5`, `rrig6600c:gpu7` never appear). Whether
the dropped connection was one of those three is **not determinable**. **None of this bears on the
terminal event**, because the frozen cohort was already sealed at 12:37:37.311 and the run log
confirms the late arrivals were correctly excluded.

---

## C. `GPU_COUNT_MISMATCH: 0/8`

### Disposition

```
C. environment/probe defect identified
```

### Evidence

**The exact function:** `PreflightChecker.check_gpu_health`, `preflight_check.py:320-383`; the
warning string is assembled at `:346-353` and rendered at `:227`.

**The exact command it runs** (`preflight_check.py:328-333`):

```
ssh -o ConnectTimeout=<n> <host> bash -lc "rocm-smi 2>/dev/null | grep -cE '^[0-9]+[[:space:]]' || echo 0"
```

**Re-run read-only against `192.168.3.122` (a query, per the brief's §C allowance):**

```
$ ssh 192.168.3.122 bash -lc "rocm-smi 2>/dev/null | grep -cE '^[0-9]+[[:space:]]' || echo 0"
0
0                       <- return code 0
$ ssh 192.168.3.122 'rocm-smi 2>&1 | head'
bash: line 1: rocm-smi: command not found
$ ssh 192.168.3.122 'echo $PATH'
/home/michael/rocm_env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:...
$ ssh 192.168.3.122 'ls /opt/rocm*/bin/rocm-smi'
/opt/rocm-6.4.3/bin/rocm-smi
/opt/rocm/bin/rocm-smi
$ ssh 192.168.3.122 '/opt/rocm/bin/rocm-smi 2>/dev/null | grep -cE "^[0-9]+[[:space:]]"'
8                       <- return code 0
```

- **stdout:** `0\n0` · **stderr:** suppressed by `2>/dev/null` · **return code:** 0.
- **Environment visible to the probe:** a non-interactive `bash -lc` login shell whose `PATH` is
  `/home/michael/rocm_env/bin:/usr/local/sbin:…` and **does not contain `/opt/rocm/bin`**.
  `rocm-smi` is not in `~/rocm_env/bin` either.
- **What it counts:** `rocm-smi` device table rows — i.e. **ROCm-visible devices**. Not worker
  daemons, not physical PCI devices.
- **The parsing logic is CORRECT.** The identical `grep -cE` expression returns **8** the moment the
  binary is found by absolute path. The detector is not miscounting; it is counting the output of a
  command that did not run.

**Why the two probes disagree:** the cluster bot's 8/8 at 12:36 and this probe are not measuring the
same thing under the same conditions — the bot reaches a working `rocm-smi`, the preflight probe
reaches none. Beta is right that the bot's 8/8 is not proof the detector is wrong; the direct
comparison above shows **the detector is right and its execution environment is wrong.**

**Why `0/8` did not fail the 3/3 preflight:** `preflight_check.py:215-229` — the GPU check
increments `checks_run`, and on failure emits warnings and then executes
`result.checks_passed += 1  # Don't block on GPU warnings` at **`:229`**. The section header at
`:215` reads `# 2. GPU health (warning only, not blocking)`. **The non-blocking behaviour is
deliberate and documented in the source.** So the "3/3 passed" is honest about its own contract —
it is the *observation* that was wrong, not the gate arithmetic.

**Aggravating secondary observation** (reported, not the disposition): `2>/dev/null … || echo 0`
converts *"the tool could not be run"* into *"there are zero GPUs."* Under
`VERIFICATION_INTEGRITY_STANDARD` VIR-3/VIR-5 an inability to observe must terminate in
`UNAVAILABLE`, never in a numeric observation. The probe as written cannot distinguish a rig with
no GPUs from a rig where `rocm-smi` is off `PATH`, and it reports the second as the first. **Note
the frozen-rig doctrine** (`CLAUDE.md` §4): the rigs are pinned and must not be modified, so any
remedy belongs in the probe (absolute path or discovery, plus an `UNAVAILABLE` terminal), not in the
rig `PATH`. **No change is proposed or made here.**

**This warning had no bearing on the failure.** All 7 `rrig6600` GPUs in the frozen cohort did real
work in both phases (§A.3).

---

## D. FUTURE LAUNCH SHAPE — DOCUMENTED ONLY, NOT LAUNCHED

### D.1 Prospective parameter set

```
worker_pool_size      = 25            <- the correction; manifest default 8 was never overridden
seed_start            = 0
seed_count            = 2147483648    (2^31)
miner_stripe_size     = 67108864      (2^26)  -> 32 stripes/stage
test_both_modes       = true
prng_type             = java_lcg      ·  window_trials = 1  ·  n_parallel = 1
use_range_miner       = true          ·  use_persistent_workers = false
```

### D.2 The `worker_pool_size` route is live — confirmed end to end

| hop | anchor | state |
|---|---|---|
| 1 — manifest `default_params` | `agent_manifests/window_optimizer.json:262` → `"worker_pool_size": 8` | present, so it survives the step-scoped filter |
| 1b — WATCHER filter | `agents/watcher_agent.py:1290-1314`, `_step1_declared_params`, `if key in declared:` at `:1313` | a `--params` key **is** applied because `worker_pool_size` is declared |
| 1c — `args_map` | `agent_manifests/window_optimizer.json:38` → `"worker-pool-size": "worker_pool_size"` | maps to the CLI flag |
| 2 — CLI | `window_optimizer.py:1504` → `parser.add_argument('--worker-pool-size', type=int, default=8)` | accepted |
| proof from this run | run log `EXEC CMD: … --worker-pool-size 8 …` | the flag was emitted with the **default**, confirming the route carries whatever value hop 1 holds |

**Passing `"worker_pool_size": 25` in `--params` will reach `--worker-pool-size 25`.** The route is
intact; only the value was never supplied.

### D.3 Does raising the pool to 25 change the derived retention requirement?

> **No. It stays 6,528 — computed, not guessed.**

Computed by importing the production function `trial_retention_files_required`
(`miner/range_miner_coordinator.py:613`) with the **caps read out of this run's own ledger**
(`workers.seed_caps_json` = `{"amd":2000000,"amd_hybrid":1000000,"nvidia":5000000,
"nvidia_hybrid":2500000}`, identical on every registered worker) and the real macro-stripe
partition:

| cohort | phase 1 | phase 2 | phase 3 | phase 4 | **required_files** |
|---|---|---|---|---|---|
| **as run — 7 AMD + 1 NVIDIA (8)** | 1088 | 1088 | 2176 | 2176 | **6528** |
| **proposed — 24 AMD + 1 NVIDIA (25)** | 1088 | 1088 | 2176 | 2176 | **6528** |

The 8-worker row **reproduces the run's own logged line exactly** —
`required=6528 resolved=6528 stages=4 stripes=32 per_stage=[('java_lcg',1,8,1088),
('java_lcg_reverse',2,8,1088),('java_lcg_hybrid',3,8,2176),('java_lcg_hybrid_reverse',4,8,2176)]` —
which validates the reconstruction.

**Why worker count does not enter:** the derivation is documented at `:620-624` as

```
sum over every planned phase
  sum over every planned stripe
    max( expected_substripes ) over workers eligible for that stripe/phase
```

— a **max over eligible workers**, not a sum over them. Adding 17 more RX 6600 XT workers does not
change the maximum, because the tightest cap in the pool is already the AMD cap:
`ceil(67108864 / 2000000) = 34` constant and `ceil(67108864 / 1000000) = 68` hybrid, giving
`32 × 34 = 1088` and `32 × 68 = 2176` per stage regardless of pool size. It would change only if the
25-worker set introduced a worker advertising a *smaller* cap than 2,000,000 / 1,000,000; none of
the 22 that registered does.

> **⚠ This corrects `docs/TB_GATE12_EVIDENCE_PACKAGE_FAIL.md` §3**, which states *"The retention
> bound sized correctly for 8 (`eligible_workers=8`, `burst_conservative=1088`); at 25 it would
> derive a different number. The 6,528 figure is therefore not the 25-worker requirement."* The
> first clause is right and the last two are wrong: **6,528 IS the 25-worker requirement.** (The
> separate per-stage `[S172-BP] derived_bound` *burst* figure does move with pool size — it carries
> `resume_margin = live connections`, which is why it logged `bound=1096` at 8 eligible and
> `bound=1110` at 22. That is a different quantity from the retention requirement, and conflating
> the two is what produced the error.)

### D.4 Two things the launch shape should account for — stated, not implemented

1. **The §A.4 lease arithmetic is not fixed by raising the pool to 25 — but it is relieved by it.**
   At 25 workers, 32 stripes give **1–2 stripes per worker** instead of 4, so the queue-wait
   component of the lease largely disappears. The underlying coupling (one 300 s deadline stamped
   on every stripe at bulk-claim time, regardless of when the worker can start it) **remains**, and
   would reappear at any geometry where stripes-per-worker × per-stripe delivery time approaches
   300 s. It is not a hypothetical: phase 1 cleared it by 64 s and phase 2 missed it by 11 s in the
   same run.
2. **Aggregate delivery rate is the binding constraint, and it is unmeasured at 25.** Both phases
   here were delivery-bound at 3.24–4.31 sub-stripe results/s against worker compute of ~5 s per
   34-sub-stripe stripe. Whether 25 workers produce a proportionally higher staging rate or simply
   contend on the same 4 staging workers (`--staging-workers 4`) is **not determinable from this
   run**, and it is the number a saturation attempt most needs to observe.

---

## E. CONCURRENCY SAMPLER ORDERING

### E.1 What happened

Confirmed and slightly corrected. The sampler is step 4 of `gate12_launch.sh:58-79`, placed after
step 3 (`gate12_launch.sh:52-56`), which runs `./scripts/launch_fleet_manual.sh … | tail -4` in the
foreground. That pipeline did not return until the run was already dead:

- sampler subshell PID 42253 start time: **12:47:27** (`ps -o lstart`);
- first TSV row: **`12:47:28  0  0  {'cancelled': 6, 'done': 58}`**.

The run failed at 12:47:17. **Zero in-run rows.** (The evidence package's §6.2 says sampling began
at 12:51; the actual first row is 12:47:28 — still entirely post-mortem, so the conclusion is
unchanged.) This is Alpha's tooling error, as stated.

### E.2 The sampler's query cannot demonstrate what Beta requires — a second, independent defect

Beta requires an observation window with **≥25 distinct in-flight workers AND queued stripes still
available**. The current query (`gate12_launch.sh:66-70`) cannot show either:

- **"in-flight workers"** is `count(distinct claimed_by) where state in ('claimed','staging')`. But
  `assign_stripes:2680-2705` claims **every stripe of the stage at once**, so this counter jumps to
  the full cohort size the instant a stage is assigned and stays there — it reports *assignment*,
  not *occupancy*. In this run it would have read 8 from 12:42:05 onward while, in truth, each
  worker was computing one stripe and holding three idle.
- **"queued stripes still available"** is read from `state_counts`. Under the same bulk-claim
  design **no stripe is ever `pending`** after assignment, so `pending` will never appear in the
  counts. A run can be maximally backlogged and the sampler will show zero queue.

**Any future saturation evidence built on this query will be unfalsifiable in the direction Beta
cares about.** Concurrency must be measured from *work in progress*, not from claims.

### E.3 Corrected block

Two changes: (a) the sampler starts **before the fleet can register**, therefore before the
coordinator can issue the first `StripeAssign`; (b) the metrics measure occupancy and backlog rather
than claims. **Provided as documentation per §D "document only" — not applied to
`gate12_launch.sh`, which is untracked and unchanged.**

```bash
# ---------- 3. CONCURRENCY SAMPLER (Beta §6) — BEFORE the fleet, not after ----------
# The coordinator cannot issue a StripeAssign until at least one worker REGISTERS,
# and no worker can register until the fleet launch in step 4. Starting the sampler
# here makes it provably active before the first assignment.
( printf 'ts\testab\tassigned_workers\tactive_workers\tundelivered_stripes\tstate_counts\n' > "$CONC"
  while :; do
    EST=$(ss -tn 2>/dev/null | grep -c '5700.*ESTAB')
    ROW=$(python3 - <<'PY' 2>/dev/null
import sqlite3
p='/home/michael/miner_staging/miner_ledger.db'
try:
    c=sqlite3.connect(f'file:{p}?mode=ro',uri=True)
    # assigned: workers holding >=1 non-terminal stripe (what the old query measured)
    a=c.execute("""select count(distinct claimed_by) from stripes
                   where state in ('claimed','staging') and claimed_by is not null""").fetchone()[0]
    # ACTIVE: workers with a claimed stripe that has actually produced a shard and
    # is not yet complete -> real in-flight occupancy, not assignment.
    v=c.execute("""select count(distinct st.claimed_by) from stripes st
                   where st.state='claimed' and st.claimed_by is not null
                     and exists (select 1 from shards sh
                                 where sh.run_id=st.run_id and sh.stripe_id=st.stripe_id)
                     and st.stripe_complete_seen=0""").fetchone()[0]
    # BACKLOG: claimed stripes with no shard yet = queued work still available,
    # the quantity 'pending' can never express under bulk claim.
    q=c.execute("""select count(*) from stripes st
                   where st.state='claimed'
                     and not exists (select 1 from shards sh
                                     where sh.run_id=st.run_id and sh.stripe_id=st.stripe_id)"""
                ).fetchone()[0]
    s=dict(c.execute("select state,count(*) from stripes group by state").fetchall())
    print(f"{a}\t{v}\t{q}\t{s}")
except Exception as e:
    print(f"-\t-\t-\t{e}")
PY
)
    printf '%s\t%s\t%s\n' "$(date +%H:%M:%S)" "$EST" "$ROW" >> "$CONC"
    sleep 5
  done ) &
SAMPLER=$!
echo "concurrency sampler pid=$SAMPLER -> $CONC" | tee -a "$EVID"
trap 'kill '"$SAMPLER"' 2>/dev/null' EXIT   # bounded by the script, not by seq 1 1440

# ---------- 4. WAIT FOR BIND, THEN LAUNCH THE FLEET ----------
for i in $(seq 1 40); do ss -ltn | grep -q 5700 && break; sleep 1; done
if ss -ltn | grep -q 5700; then
  ./scripts/launch_fleet_manual.sh 192.168.3.177 5700 2>&1 | tail -4
else
  echo "COORDINATOR NEVER BOUND — aborting fleet launch"; tail -30 "$LOG"; exit 1
fi
```

The Beta criterion is then read off `active_workers >= 25 AND undelivered_stripes > 0` **in the same
row**, sustained across consecutive samples — which is the "≥25 distinct in-flight workers AND
queued stripes still available" test, and is not satisfiable by "distinct workers eventually used
= 25". Note the `trap`/`EXIT` line also removes the current script's second-order problem: the
`seq 1 1440` sampler **outlives the run by two hours** and is still running now (§0).

---

## F. PRODUCTION-CHANGE CLASSIFICATION

```
PRODUCTION DEFECT FOUND — amendment submitted
```

**Not implemented. Described only, for separate Beta review before any rerun.** Two findings; the
first is the one that terminated the trial.

### F-1 (primary) — the compute lease is stamped at bulk-claim time, so it measures queue wait, not worker liveness

`assign_stripes` (`:2680-2705`) claims **every** stripe of a stage in one loop with one `now`, and
stamps each with `now + compute_lease_timeout` (`:2695`, `:245` = 300 s). `_dispatch_pending`
(`:7004-7014`) then dispatches all of them, and the worker executes them **serially**
(`range_miner_worker.py:1425-1431`). With `stripes_per_worker = 32/8 = 4`, a worker's last stripe
does not begin until ~230–260 s of its own 300 s lease has already been consumed by the other three.

The lease's stated purpose (`:1663-1667`) is to reclaim leases from workers that have **stopped**.
Here it expired on three workers that were **actively streaming results at that instant** — s9 at
12:47:11.338, s5 at 12:47:12.056, s7 at 12:47:12.607, against a 12:47:05.487 deadline.

Renewal cannot compensate: `renew_lease` (`:1648-1661`, driven from `:6894-6901`) renews **only
`msg.current_stripe_id`**, so a queued stripe's lease burns down untouched; and once the stripe *is*
current, the heartbeat competes with the result stream on one ordered TCP connection
(`:6549-6552` states this explicitly) — no heartbeat renewal landed on s5, s7 or s9 at any point.
The §1.4/F2 lease exemption does not cover this: it keys on `_paused_connections` membership
(`:5185-5196`) and the run recorded `pause_events=0`.

**Blast radius:** phase 1 and phase 2 are constant-mode, so **any** stage whose per-worker stripe
queue takes longer than 300 s to deliver terminates the whole trial with no retry. Phase 1 cleared
it by 64 s; phase 2 missed it by 11 s. **This is a live latent failure at any geometry where
`stripes_per_worker × per-stripe delivery time → 300 s`, and it is a fail-closed cliff, not a
degradation.** Raising `worker_pool_size` to 25 reduces stripes-per-worker to 1–2 and would very
likely have avoided it in this run, but does not remove the coupling.

**Deliberately not proposing a remedy here.** Beta's own §1.4 note shows the lease/back-pressure
interaction has already cost three review rounds (§2.19 F1), and the candidate fixes — stamp the
lease at dispatch rather than at claim, renew on any accepted frame from the bound worker rather
than on heartbeat alone, or claim only what a worker can start — differ materially in their
concurrency properties. That is a design decision for Beta, and the owner rule on choosing the
structurally stronger mechanism (skill §7, 2026-08-06) applies to it.

### F-2 (secondary, observability) — the constant-phase terminal path is silent

`_handle_stripe_failure_locked:5106-5107` constructs a precise reason string and emits **no log
record**; `fail_trial:5342-5348`, `abort_trial:5350-5423` and `cancel_active_stripes:1546-1556`
emit none either; and `trials` has no column for the reason, so it is discarded at `:5406-5407`.
`process_lease_expiry` logs only its two *skip* branches (`:5188`, `:5197`), never its terminal one.
The neighbouring capacity-timeout path does `logger.error` before failing (`:6031-6032`), so this is
an inconsistency inside one file rather than a global convention.

**Consequence, observed:** the coordinator log contains nothing whatsoever between 12:42:05.645 and
12:47:17.448, and the operator was left with a downstream `MinerIngressError` describing a
threshold-provenance gate that never ran. Every fact in section A had to be recovered from ledger
row *shapes*. Under VIR-1 a terminal decision that leaves no execution record is not observable, and
the fail-closed design that Beta correctly credits (§4 of the evidence package) is only auditable
because `cancel_active_stripes` happens not to overwrite `claimed_by`.

### F-3 (secondary, and independently disposed in §C) — the preflight GPU probe

Reported under §C with disposition **C**. It did not contribute to the failure and is listed here
only so it is not lost: the probe cannot distinguish `UNAVAILABLE` from `0`, and its
`|| echo 0` reports an unobservable surface as a definite count of zero.

### Explicitly NOT defects

- **The constant-phase immediate failure itself.** Beta §8: phase 2 is constant-mode; `:5106`
  behaved exactly as specified. **No retry was permitted.**
- **The `MinerIngressError`.** The D6 gate at `:6409` correctly never ran, so `validated` correctly
  stayed False and ingress correctly refused. Fail-closed worked.
- **The cohort freeze.** 12 workers registered at 12:41:33 and were correctly excluded.
- **The 6,528 retention derivation.** Reproduced exactly (§D.3).
- **`paused_high_water=0` / `pause_events=0`.** Correct — the back-pressure amendment's pause path
  genuinely was not needed. **But it is also why the lease exemption could not apply**, which is
  F-1, not a defect in the amendment.

---

## Disagreements with the brief

None on scope, constraints or method. Three corrections to inherited statements, each anchored
above rather than asserted:

1. **§D's premise** — *"at 8 workers it derived 6,528; at 25 the per-stage conservative bound will
   differ"*. The **retention requirement** does not differ: it is 6,528 at both (§D.3, computed with
   the production function). The per-stage **burst bound** does differ (1096 → 1110 in this run),
   because it adds `resume_margin = live connections`. Two different quantities.
2. **Evidence-package §6.2** — the sampler's first row is **12:47:28**, not 12:51. Conclusion
   unchanged (still post-failure).
3. **Evidence-package §4** — *"the open question is why stage 2 cancelled 6 of 32 stripes"*. Beta's
   correction is upheld by the evidence: the six are two different states under one abort (§A.5),
   and only **one** stripe initiated anything.

## Verification-integrity controls (VIR-1…6)

- **execution proof:** every ledger figure from named queries against a hashed copy; every source
  claim from `file:line` read this session; §C re-run live against `192.168.3.122` with stdout,
  stderr and return code recorded; §D.3 computed by importing the production function.
- **clean control:** phase 1 — geometrically identical to phase 2, same cohort — completed all
  1,008 sub-stripes with 64 s of lease margin. It is the negative control for F-1.
- **fault-injection control:** **none run.** No fault may be injected under a read-only,
  no-launch brief. F-1's mechanism is therefore established by ledger reconstruction and source
  reading, **not** by reproduction. *What would settle it:* an offline gate driving
  `assign_stripes` → serial worker execution past `compute_lease_timeout` with no fault present.
- **completion sentinel:** ledger `trials.finalized_at`, `abort_cleanup_status='done'`, 1,844/1,844
  reservations `released`; log `[S172-BP] summary` at 12:47:17.448.
- **unavailable-observer behaviour:** stated as `UNAVAILABLE`/"cannot be determined", never as
  clean — §B.1 (identity of the dropped connection), §A.2.4 (`stripe_error` exclusion), §A.4 (which
  of the two heartbeat-starvation mechanisms), §A.3 (claim/cancel timestamps absent from schema).
- **audit claim scope:** this run only — `distributed_config_t1_689f3cd9`, 2026-08-09
  12:37:09–12:47:17, coordinator side. **No claim is made about any other run, or about the run's
  behaviour at `worker_pool_size=25`.**
- **searched surfaces:** ledger copy (all 7 tables); coordinator log; evidence file; concurrency
  TSV; `gate12_launch.sh`; live source `miner/range_miner_coordinator.py`,
  `miner/range_miner_worker.py`, `preflight_check.py`, `window_optimizer.py`,
  `agents/watcher_agent.py`; `agent_manifests/window_optimizer.json`; live host processes (`ps`,
  `fuser`); live rig `192.168.3.122` (probe re-run, `/tmp/minerlogs/*`, `PATH`, `/opt/rocm*`);
  `docs/` — `TB_GATE12_EVIDENCE_PACKAGE_FAIL.md`, `TB_NOTE_AMENDMENTS_COMMITTED_GATE12_REQUEST.md`,
  `CLAUDE_CODE_INSTRUCTIONS_GATE12_FAILURE_FORENSICS.md`; `CLAUDE.md`; the tfm-project-facts skill
  (v19).
- **unavailable surfaces:** worker-side heartbeat/lease telemetry (**does not exist** — no logging
  in `_heartbeat_loop` or `_fail_stripe`); rig kernel logs (CT100 is an unprivileged LXC, no root
  key auth to `.121` — skill §2.17, unchanged); rigs `.156`/`.164` (not probed — only `.122` hosted
  the cohort); network capture for 12:40:53–12:41:08 (not retained); `rrig6600b`/`rrig6600c` worker
  logs (not read — no cohort member ran there).
- **governance trail searched:** `TB_GATE12_EVIDENCE_PACKAGE_FAIL.md`,
  `TB_NOTE_AMENDMENTS_COMMITTED_GATE12_REQUEST.md`, the S172-BP/staging-capacity/S145 rulings as
  carried in the commit messages of `a3bb4da` / `4dd5535` and in the S172-BP amendment blocks quoted
  inline in `range_miner_coordinator.py`.
- **chapters searched:** none — no claim in this report concerns sieve mathematics, feature
  semantics or pipeline intent.
