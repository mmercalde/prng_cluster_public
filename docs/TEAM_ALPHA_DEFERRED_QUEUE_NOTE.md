# TEAM ALPHA → TEAM BETA — the deferred-queue bound, and a ruling Beta once left to Alpha

**Ruling requested.** The `staging_deferred_max = 64` bound was Alpha's number, not Beta's, and it
just terminated the first production-shape trial that ever got past staging. **The sizing is the
smaller question; the classification is the larger one.**

---

## 1. Staging is repaired. The next defect is now reached.

Beta's §0 held: *"the staging defect is NOT YET PROVEN to be the only production-path defect."*
**It was not.** It was the first.

The 2026-08-05 run, 25 daemons hand-started, `--staging-dir` carried from the manifest:

```
:118  [S172 Part B] coordinator staging VALIDATED: /home/michael/miner_staging
      (fstype=ext4, disk-backed, avail=426.86 GiB, high_water=16.00 GiB,
       headroom=1.60 GiB, atomic-rename proven)
:121  staging failed …__st0_s0 (retryable=True): staging deferred queue full — dispatch back-pressure
:122-129  L1 fence dropped every later sub_stripe_result and stripe_complete:
      "state 'cancelled' does not permit this message"
:130  MinerIngressError … payload={1:[0.65]} phase_direction={1:'forward'} validated=False
```

**Part B works.** `MinerIngressError` is again the symptom, not the defect — the L1 fence dropped
`st0_s3`'s `stripe_complete` **whose 32 shards had verified**, so only phase 1 was ever recorded.

## 2. Measured from the ledger the run left behind — no inference

`/home/michael/miner_staging/miner_ledger.db`:

| stripe | worker | phase | expected_substripes | shards recorded | staging_status |
|---|---|---|---|---|---|
| `__st0_s0` | `rrig6600:gpu0` | 1 | 34 | 33 | **pending** |
| `__st0_s1` | `zeus-ubuntu-vm:gpu0` | 1 | 14 | 14 | verified |
| `__st0_s2` | `rrig6600:gpu1` | 1 | 34 | 32 | **pending** |
| `__st0_s3` | `rrig6600:gpu2` | 1 | 34 | 32 | verified |

**Pending shards: 33 + 32 = 65. The cap is 64. The 65th staging request is the one that
back-pressured.** To the unit.

### 2.1 The premise needed correcting — `--miner-substripes 8` does not size sub-stripes

The worker partitions its stripe by the **VRAM seed cap**
(`partition_stripe` + `select_seed_cap`, `range_miner_worker.py:472-504`), and the coordinator
agrees via `expected_substripes_for` (`range_miner_coordinator.py:364`):

```
ceil(67,108,864 / 2,000,000) = 34   per ROCm stripe   (--seed-cap-amd)
ceil(67,108,864 / 5,000,000) = 14   per CUDA stripe   (--seed-cap-nvidia)
```

**Stage 0 generated 3 × 34 + 14 = 116 staging requests against 6 concurrent slots and a 64-deep
queue.** 111 arrived before cancellation.

**The queue must buffer `stripes_in_flight × ceil(stripe_size / seed_cap)` — here 4 × 34 = 136
against 64.** Undersized by ~2×, **and the parameter that drives it is the VRAM seed cap — one
nobody would look at when sizing a staging queue.**

### 2.2 ⚠ This happened at 4 workers. Production is worse, not better.

`max_seeds 268,435,456 ÷ miner_stripe_size 67,108,864 = 4 stripes`, so **21 of the 25 admitted
workers were idle.** At full utilisation the arrival count is **25 × 34 = 850 against the same 64.**

### 2.3 The two bounds are mismatched by six orders of magnitude

`_defer_locked` (`:2642-2656`) enforces **a count cap of 64** and **a retained-bytes cap reusing
`staging_high_water_bytes` (16 GiB)**. These shards are inline (`remote_spool_path` NULL) at
**~160 bytes** each — 64 of them retain about **10 KB**. **The byte cap exists to bound coordinator
RAM and cannot fire. The bound that actually fires is the count, which is not a statement about
memory at all.**

### 2.4 Not configurable

`staging_deferred_max`, `staging_workers` and `staging_queue_depth` appear **only** in the
coordinator's dataclass. `build_coordinator` accepts `staging_high_water_bytes` and
`staging_high_water_files` (`:4691`, `:4709`) **but not these three**. They are absent from
`window_optimizer.py`, from `window_optimizer_integration_final.py`, and from the manifest.

**That is the three-hop route missing at hops 1 and 2 — the identical shape to the `staging_dir`
dead read Part B just closed.** Only an in-source edit can change these values.

## 3. ⚠ `retryable=True` is inert on every path TFM actually runs

`enqueue_staging:2729` reports `retryable=True`. But `_handle_stripe_failure_locked` consults
`retryable` **only in the non-retryable branch** (`:3059`), and the very next test is:

```python
if phase in (1, 2):                     # :3064
    self.fail_trial(...); return {"action": "fail_trial", "reason": "constant_phase"}
```

**For a constant phase, `retryable=True` and `retryable=False` produce the identical outcome.** The
Q3 one-retry exists only for hybrid phases 3/4. **The flag bought nothing here.**

**And it would buy nothing on a hybrid phase either: the deferred queue is GLOBAL to the
coordinator, not per-worker, so a reassignment lands on the same full queue.**

## 4. The governance finding — Beta offered two options and the second was taken

`docs/CLAUDE_CODE_CORRECTION4_S172_PHASE4_OVERLOAD.md` §1c, Beta's correction that created this
bound:

> *Fix: enforce a bound on `_deferred` … **The correct back-pressure for a miner is to NOT read the
> next result off that worker's socket until capacity frees** (so the payload stays on the wire / at
> the worker, not in coordinator RAM), **or** to reject with a retryable error that the matrix
> handles. **Pick one.**

**Beta named the socket-level option as correct and permitted either. The implementation took the
second.**

**There is also a tension with CORRECTION2:128** — *"Back-pressure (reserve returns None) must
POSTPONE and resume (re-queue), not abandon"* — which governs the reserve path **while the
deferred-overflow path abandons.**

**No Beta ruling on the deferred queue exists.** `TB_BINDING_RULINGS_S172_PHASE4.md` contains no
back-pressure, deferred or queue language. **The 64 came from Alpha's implementation of C4**,
recorded in `TEAM_ALPHA_REVIEW_S172_PHASE4_REV5.md:51-53`.

## 5. The question Alpha is actually asking

> **A coordinator-side transient capacity condition is currently charged to a worker's stripe as a
> fault.** The worker did nothing wrong; the coordinator was momentarily full; the trial died.

**Is that the intended contract?** Alpha does not think it is, but the bound is Beta's correction
and the choice between its two options was Alpha's, so the disposition is Beta's.

## 6. Options, with costs

| | option | cost | notes |
|---|---|---|---|
| **A** | Raise `staging_deferred_max` **and make it reachable** (manifest + call site + `build_coordinator`) | small | Retained RAM is trivial (~160 B/entry). Honest rule: **`stripes_in_flight × ceil(stripe_size / seed_cap)`**. **Does not fix the classification.** |
| **B** | Implement the **socket-level** back-pressure Beta named correct — stop reading that worker's socket until capacity frees | largest | **Structurally right; removes the condition from the failure matrix entirely. Beta's stated preference.** |
| **C** | Raise staging throughput (`staging_workers` / `staging_queue_depth`, also unreachable) | small | **6 concurrent staging jobs for a 25-worker fleet is the underlying imbalance**; A alone just makes the queue deeper |
| **D** | Reclassify: coordinator-side back-pressure **postpones/re-queues** rather than entering the stripe matrix | medium | Resolves the C2/C4 tension. **Needs a Beta ruling.** |
| **E** | Parameter-only: raise seed caps or shrink `miner_stripe_size` | none (code) | **Seed caps are VRAM-governed — not an Alpha call.** |

**Alpha's reading:** **A + C are the minimum to get past this. B is what Beta actually asked for.
D is the question worth putting**, because retryable-vs-non-retryable is currently *a distinction
without a difference on the constant phases every TFM run uses*.

**Nothing has been changed. No fix applied.**

## 7. Rulings requested

1. **A, B, C, D or a combination** — and if not B, Alpha asks Beta to record why the option it named
   correct is being set aside.
2. **The sizing rule**, if A: is `stripes_in_flight × ceil(stripe_size / seed_cap)` the right
   formula, and should it be **derived** rather than a constant, given the driver is the VRAM cap?
3. **§5's contract question** — is a coordinator-side capacity condition a worker fault?
4. **The C2/C4 tension** — reserve postpones, deferred-overflow abandons. Which governs?

## 8. VIR declaration

**Audit scope:** the live run of 2026-08-05 18:37–18:39 plus the repo at HEAD.
**Execution proof:** the ledger table in §2 is measured from `miner_ledger.db`, not inferred;
every anchor re-read this session.
**Searched surfaces:** `logs/partb_prodshape.log` · `miner_ledger.db` ·
`miner/range_miner_coordinator.py` · `miner/range_miner_worker.py` ·
`CLAUDE_CODE_CORRECTION4_S172_PHASE4_OVERLOAD.md` · `TB_BINDING_RULINGS_S172_PHASE4.md` ·
`TEAM_ALPHA_REVIEW_S172_PHASE4_REV5.md` · `window_optimizer.py` ·
`window_optimizer_integration_final.py` · `agent_manifests/window_optimizer.json`.
**Unavailable surfaces:** Proxmox host kernel logs on `.121/.155/.163`; `dmesg` on all four hosts.
**Not established:** whether anything downstream of the staging queue works — **the trial still has
not completed, so §0 continues to stand.**
