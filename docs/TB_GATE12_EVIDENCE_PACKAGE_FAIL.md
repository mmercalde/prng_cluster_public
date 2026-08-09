# TEAM ALPHA → TEAM BETA — GATE-12 EVIDENCE PACKAGE: **FAIL**

**Run:** `distributed_config_t1_689f3cd9` · 2026-08-09 12:37:09 → 12:47:17 (≈10 min)
**Alpha's determination: GATE 12 FAILED.** Two distinct findings, one of them an Alpha run-shape
error. Per your §13 nothing was adjusted and re-run under the same label; the run was left exactly
as it failed.

**No 25-GPU saturation claim is made.**

---

## 1. Authority evidence (your §12)

| item | value |
|---|---|
| HEAD | `a3bb4da` (S145 certified); prerequisite `4dd5535` (S172 certified) |
| tree | untracked docs + runtime residue only; no unreviewed production edits |
| pre-run certified cursor | `CursorResult(status='OPEN', next_seed_start=0, domain_start=0, domain_end_exclusive=4294967296, covered_seed_count=0, certified_interval_count=0, intervals=())` — **0, namespace empty** |
| dataset | `daily3-20260801T145551443433Z-513648160d35`, sha256 `513648160d35…`, 1,380,711 B / 18,068 records, **verified on all four nodes** |
| execution set | `d89834f1bf26`, backend=miner, profile=proxmox, **gpus=25**, remote=True, **admission=8** |
| seed interval | `[0, 2,147,483,648)` — entirely inside `[0, 2^32)`, `--seed-start 0` supplied **explicitly** |
| legacy tracker | not used: *"no certified coverage for java_lcg — using seed_start=0 (the legacy exhaustive_progress tracker … carries zero certified authority)"* |
| cursor after | unchanged — **no publication, so no coverage advance.** Correct. |

## 2. What the certified machinery did — all of it worked

**Retention preflight, at your authorized geometry:**

```
[S172-CAP] retention preflight run=distributed_config_t1_689f3cd9 mode=derived
required=6528 resolved=6528 stages=4 stripes=32
per_stage=[('java_lcg',1,8,1088), ('java_lcg_reverse',2,8,1088),
           ('java_lcg_hybrid',3,8,2176), ('java_lcg_hybrid_reverse',4,8,2176)]
```

**32 macro-stripes per stage, four stages, 6,528 files derived and admitted** — computed from real
geometry, never hardcoded, exactly as the amendment requires.

**Cohort frozen at preflight**, all four stages recorded with their 8 identities.

**Staging back-pressure exercised under genuine load and held:**

```
[S172-BP] summary inbound_qsize_high_water=690 deferred_high_water=247
derived_bound=1110 bound_in_force=1110 paused_high_water=0 pause_events=0
staging_jobs_completed=1844 staging_jobs_per_sec=3.055
capacity_timeout_terminations=0 capacity_invariant_terminations=0
```

247 deferred against a 1,110 bound with **zero pauses and zero capacity terminations**. The
2026-08-05 deadlock class did not recur, and unlike the 2026-08-07 attempt it was **actually put
under pressure** — 1,844 staging jobs completed.

**Work completed:**

```
stripes by phase/state:  (1,'done',32) · (2,'done',26) · (2,'cancelled',6)
distinct workers used:   8
```

**Stage 1 completed all 32 stripes.** This is by a wide margin the furthest this pipeline has ever
run: ~10 minutes of real fleet work versus ~3 on 2026-08-07.

## 3. FINDING 1 — Alpha run-shape error: `worker_pool_size = 8` (saturation claim void)

**This is Alpha's error, not a system defect.**

`admission_count = min(requested, count of selected worker identities)`, and for the miner
`requested` is **`worker_pool_size`** (`execution_set.py:170-176`). The manifest default is **8**
(`agent_manifests/window_optimizer.json:262`), and Alpha's `--params` set the seed geometry but
**never overrode the pool size.** The command line therefore carried `--worker-pool-size 8`, and
the chain was internally consistent throughout:

```
[EXEC-SET] gpus=25 … admission=8
[EXEC-SET] miner: expected_workers=8, bound to the frozen set (agrees with the requested pool size)
[ADMISSION] expected_workers=8 (source=execution_set(d89834f1bf26))
frozen cohort: rrig6600:gpu0-6 + zeus-ubuntu-vm:gpu0   (8 identities)
```

**Consequences, stated rather than minimised:**

- **No 25-GPU saturation claim is made or implied.** Your §3 forbids reinterpreting a smaller
  successful run as a saturation pass; Alpha does not.
- The 32-stripe geometry *was* generated and stage 1 fully executed — but across **8** GPUs, so it
  demonstrates stripe-count correctness, not full-fleet occupancy.
- The retention bound sized correctly **for 8** (`eligible_workers=8`, `burst_conservative=1088`);
  at 25 it would derive a different number. The 6,528 figure is therefore not the 25-worker
  requirement.
- **The cohort freeze behaved exactly as certified:** stage 2 logged `eligible_workers=22` — more
  workers connected after preflight — and they were correctly **excluded** from the frozen trial.
  That is your §5 law working under real conditions, and Alpha reports it as positive evidence.

**Correction for a future attempt: `"worker_pool_size": 25` in `--params`.** Alpha is not
proposing to apply it and re-run, because Finding 2 is independent of it.

## 4. FINDING 2 — the trial did not complete its four-stage workflow (independent defect)

The terminal error is `MinerIngressError … validated=False`. **That is the symptom, not the
defect** — for the fourth time in this project's history, and Alpha states it plainly rather than
reporting the wall as the cause.

**Verified from source, not inferred:** `validate_threshold_provenance` is called **only** under
`if stage_idx >= len(workflow_stages)` (`miner/range_miner_coordinator.py:6375-6385`) — i.e. after
**all four** planned stages complete. The ledger shows **stage 2 incomplete (26 done, 6 cancelled)
and stages 3 and 4 never started.** The gate therefore never fired, `validated` remained False, and
ingress correctly refused a trial whose filter was unproven. **The fail-closed design worked.**

**Notable progress versus 2026-08-07:** the provenance record is now **populated and internally
consistent** —

```
requested={'forward':0.45,'reverse':0.45}
payload={1:[0.45], 2:[0.45]}   effective={1:[0.45], 2:[0.45]}
phase_direction={1:'forward', 2:'reverse'}   validated=False
```

versus `payload={}`/`effective={}` on 2026-08-07. Thresholds propagated correctly for both stages
that ran; only the terminal validation was never reached.

**The open question is therefore why stage 2 cancelled 6 of 32 stripes and the workflow stopped
there.** One correlated event is logged at 12:41:08 —
`dropping connection that never completed a frame within 15.0s read deadline (Defect 6)` — but
Alpha does **not** assert causation from a single log line and has not diagnosed it.

## 5. Deviations from the frozen run shape

1. **`worker_pool_size` left at the manifest default of 8** (§3). Every other frozen dimension was
   honoured: `seed_start=0`, `seed_count=2^31`, `miner_stripe_size=2^26`, 32 stripes/stage,
   `java_lcg` + `{constant, variable}`, range-miner path, one active trial.
2. **No mid-run intervention of any kind.** No limit was altered, no knob twiddled, nothing
   restarted. The run was allowed to fail and left as it failed.

## 6. Two items requiring explanation before any saturation claim

1. **`GPU_COUNT_MISMATCH: 0/8`** warned for all three rigs during preflight (which nonetheless
   passed 3/3), while the cluster bot independently reported **8/8 GPUs OK on every rig at
   12:36**, one minute before. A detection-path discrepancy, unexplained.
2. **The concurrency sampler produced no in-run rows.** It is started in step 4 of Alpha's launch
   script, after the fleet-launch step returns; the fleet script ran long enough that sampling
   began at 12:51 — after the 12:47 failure. Its rows are post-mortem only. **Alpha's tooling
   error**, and it means this run carries **no live concurrency evidence** even for the 8 workers
   it did use. The sampler will be started before the coordinator on any future attempt.

## 7. Requested disposition

Alpha requests **no re-run authorization at this time.**

Correcting `worker_pool_size` and re-running would leave Finding 2 undiagnosed, and your §13 is
explicit that multiple knobs must not be adjusted and the run continued under the same Gate-12
label. Alpha proposes instead:

1. a **read-only** diagnosis of why stage 2 cancelled 6 stripes and the workflow terminated before
   stages 3-4, using the ledger and logs already on disk — **no fleet run required**;
2. on the basis of that diagnosis, a return to Beta with either a defect submission or a request to
   re-run with `worker_pool_size=25`;
3. the sampler-ordering and preflight GPU-count items addressed as part of the same package.

**Gate 12 is recorded as FAILED. Nothing has been changed since the failure.**
