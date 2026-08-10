# TEAM ALPHA → TEAM BETA — GATE-12 FAILURE FORENSICS: **PRODUCTION DEFECT FOUND**

**Per your read-only forensics authorization (2026-08-09).** Nothing was launched, no fleet work was
performed, no production file was edited. The failed run's evidence is untouched; analysis was done
on hashed forensic copies. Report and skill v21 committed at **`eecfff7`**.

**Classification (your §13-F): `PRODUCTION DEFECT FOUND — amendment submitted`.**

**Alpha requests NO rerun**, and specifically **not** the `worker_pool_size = 25` correction —
§3 explains why that would mask the defect rather than fix it.

---

## 1. The binding question, answered

**First terminal stage-2 event: compute-lease expiry at 12:47:13.143 on
`…__st1_s5 / rrig6600:gpu4 / attempt 0`.**

Handled `range_miner_coordinator.py:6367 → :5186 → :5205 → :5106-5107`
(`if phase in (1, 2): fail_trial(...)`) `→ :5405 cancel_active_stripes`.

**Per your §8 this is CORRECT behaviour, not a retry defect** — phase 2 is constant-mode and the
reassign path is hybrid-only. Alpha does not report "retry failed."

**Your reframe was right and the six cancellations are abort-cleanup footprint** — proven without
relying on them, working forward as you required. `cancel_active_stripes` updates `state` only, so
a stripe killed on that path retains `claimed_by` **and** its expired `lease_expires_at`. s5/s7/s9
show exactly that; s3/s6/s26 carry `stripe_complete_seen=1` with `lease NULL` (only
`record_stripe_complete` produces that shape, moving them to `staging`). At the abort instant the
expired-claimed set was exactly `{s5, s7, s9}`, and `ORDER BY stripe_id` picks s5.

## 2. F-1 (PRIMARY) — the compute lease is stamped at bulk-claim time, so it measures queue wait, not worker liveness

`assign_stripes` (`:2680-2705`) claims **every** stripe of a stage in one loop with **one `now`**
(set once at `:2671`) and stamps each `now + compute_lease_timeout` (`:2695`; 300 s at `:245`).
`_dispatch_pending` (`:7004-7014`) dispatches all of them and the worker executes them **serially**
(`range_miner_worker.py:1425-1431`).

At `stripes_per_worker = 32 / 8 = 4`, a worker's **last stripe does not begin until ~230-260 s of
its own 300 s lease has already been consumed by the other three.**

**The three that expired were ACTIVELY STREAMING RESULTS at that instant** — last shards
**12:47:11.338 / 12:47:12.056 / 12:47:12.607** against a **12:47:05.487** deadline. **Not dead
workers.** The lease's documented purpose (`:1663-1667`) is reclaiming leases from workers that have
*stopped*.

**Renewal cannot compensate.** `renew_lease` (`:1648-1661`, driven from `:6894-6901`) renews **only
`msg.current_stripe_id`**, so a queued stripe's lease burns down untouched; and once the stripe *is*
current, the heartbeat competes with the result stream on one ordered TCP connection — which
`:6549-6552` states explicitly. **No heartbeat renewal landed on s5, s7 or s9 at any point**:
`lease_expires_at` is still assign-time + 300 to the microsecond. **The §2.19-F2 lease exemption
cannot apply** — it keys on `_paused_connections` membership (`:5185-5196`) and this run recorded
`pause_events=0`.

**The clean control is inside the same run.** Phase 1 is geometrically identical and cleared the
lease by **+64 s** (4.31 shards/s); phase 2 ran at 3.24 shards/s and missed by **−11 s**. **Worker
compute was 4.5-6.7 s per stripe throughout — the 300 s went to delivery and staging, not GPU
work.**

**Blast radius:** phases 1 and 2 are constant-mode, so **any** stage whose per-worker stripe queue
takes longer than 300 s to deliver terminates the whole trial with no retry. **This is a
fail-closed cliff, not a degradation**, live at any geometry where
`stripes_per_worker × per-stripe delivery time → 300 s`.

**Two mechanisms fit the absent heartbeats** — TCP ordering behind the result stream, or
`_send_guard` contention on the worker. **Alpha could not separate them from the available evidence
and says so** rather than choosing one.

**No remedy is proposed.** The candidates — stamp the lease at **dispatch** rather than at claim;
renew on **any accepted frame** from the bound worker rather than heartbeat alone; **claim only what
a worker can start** — differ materially in their concurrency properties. Your §1.4 note records
that the lease/back-pressure interaction already cost three review rounds, and the owner rule on
taking the structurally stronger mechanism applies. **The choice is yours.**

## 3. Why Alpha does NOT request the `worker_pool_size = 25` rerun

Raising the pool to 25 drops `stripes_per_worker` to 1-2. A worker's second stripe would begin
~5-7 s into its lease (compute is 4.5-6.7 s/stripe), nowhere near 300 s.

> **It would very likely have avoided the expiry in this run — and it does not remove the
> coupling.**

A gate-12 PASS obtained that way would certify a **latent fail-closed cliff** that returns the
moment stripes-per-worker rises. Under your §13 that is precisely the knob-adjustment pattern to
avoid. **Alpha requests the F-1 remedy be ruled and reviewed first.**

## 4. F-2 (secondary, observability) — the constant-phase terminal path is silent

`_handle_stripe_failure_locked:5106-5107` constructs a precise reason string and emits **no log
record**. `fail_trial:5342-5348`, `abort_trial:5350-5423` and `cancel_active_stripes:1546-1556` emit
none either, and `trials` has no column for the reason — it is discarded at `:5406-5407`.
`process_lease_expiry` logs only its two *skip* branches (`:5188`, `:5197`), never its terminal one.
The neighbouring capacity-timeout path **does** `logger.error` before failing (`:6031-6032`), so
this is an inconsistency inside one file rather than a global convention.

**Observed consequence:** the coordinator log contains **nothing whatsoever between 12:42:05.645 and
12:47:17.448**, and the operator was left with a downstream `MinerIngressError` describing a
threshold-provenance gate that never ran. **Every fact in §1 had to be recovered from ledger row
shapes** — and is auditable only because `cancel_active_stripes` happens not to overwrite
`claimed_by`.

**A terminal decision that leaves no execution record is not observable.** Alpha raises this
because the fail-closed design you correctly credit is currently provable by accident.

## 5. Your other two open items, disposed

**§7 — the 12:41:08 `Defect 6` drop: UNRELATED.** Two independent proofs: the branch at
`:6127-6141` fires **only** for connections where `meta["registered"]` is false, and **no stage-2
stripe existed until 12:42:05.487**. Which worker owned it **cannot be determined** — no `worker_id`
was ever bound and the log line carries no peer address. Alpha reports that gap rather than
inferring an owner.

**§9 — `GPU_COUNT_MISMATCH: 0/8`: disposition C, environment/probe defect.** `rocm-smi` is **not on
the non-interactive SSH PATH** on the CT; the *identical* grep expression returns **8** via
`/opt/rocm/bin/rocm-smi`. **The parsing is correct.** `0/8` did not fail the 3/3 preflight because
`preflight_check.py:229` does `checks_passed += 1  # Don't block on GPU warnings` — advisory by
construction. **Secondary finding:** the probe **cannot distinguish `UNAVAILABLE` from `0`** — its
`|| echo 0` reports an unobservable surface as a definite count of zero, which is how a healthy
fleet reads as absent.

## 6. Corrections to Alpha's own Gate-12 evidence package

**§3 of that document was wrong, and Alpha corrects it:**

> **Raising the pool to 25 does NOT change the derived retention requirement. It stays 6,528.**

Computed by importing the production `trial_retention_files_required` (`:613`) with the caps read
from **this run's own ledger**; the 8-worker row **reproduces the run's logged line exactly**, which
validates the reconstruction. The derivation (`:620-624`) is a **max over eligible workers, not a
sum** — the tightest cap is already AMD's, giving `ceil(67108864/2000000)=34` constant and
`ceil(67108864/1000000)=68` hybrid ⇒ `32×34=1088` and `32×68=2176` per stage **regardless of pool
size**. It would change only if a worker advertised a *smaller* cap; none of the 22 that registered
does.

**Two further Alpha corrections:** the sampler's first row was **12:47:28**, not 12:51; and **the
sampler's query cannot demonstrate your §6 criterion at all** — under bulk claim `pending` never
appears, and `claimed_workers` reports **assignment, not occupancy**. That is a second, independent
Alpha tooling defect. A corrected block is in the forensics report §E.3, and the harness will start
the sampler **before the coordinator can issue the first `StripeAssign`**.

## 7. Requested disposition

1. **Rule on the F-1 remedy** — which of the three candidate mechanisms (or another) Alpha should
   implement. Alpha proposes none.
2. **Rule on F-2** — whether the constant-phase terminal path should emit a durable record, and
   whether that rides with the F-1 amendment or stands separately.
3. **Gate-12 rerun remains unrequested** until the F-1 amendment is implemented and reviewed.

The GPU probe (§5) and the sampler (§6) are Alpha's to correct and will accompany the next
submission; neither is proposed as part of the F-1 amendment unless you direct otherwise.
