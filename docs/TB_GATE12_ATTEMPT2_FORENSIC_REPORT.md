# TEAM ALPHA → TEAM BETA — GATE-12 ATTEMPT-2 FORENSIC REPORT (§13)

**Date:** 2026-08-10
**Run:** `distributed_config_t1_abc63f71` · tree `4643a11` (authorized) · one production-shape execution
**Disposition of the run:** FAILED at Step 1, stage 3→4 transition. **Preserved, not repaired. No relaunch.**
**First authoritative terminal event:** `2026-08-10 10:44:16,113` — `TRIAL TERMINAL
class=worker_admission_timeout` (worked forward from this, not from the cleanup tail, per §13).

**One-line finding:** Not hardware, not the certified F1/F2 within-stage scheduler, not staging —
**the coordinator's per-stage admission gate requires `expected_workers` live registered connections
at the start of every stage, and by stage 4 two of the 25 non-persistent workers' TCP connections
were no longer in the registered set; 23 were eligible, the 180 s admission window elapsed, the trial
aborted.** The whole fail-closed chain below it behaved exactly as designed.

---

## 1. WHAT COMPLETED (measured — ledger + coordinator summary)

- **Stages 1, 2, 3 each completed all 32 stripes**, `state=done`, `COUNT(DISTINCT claimed_by)=25`,
  each covering the full seed range **[0, 2,147,483,648)**. Verified against the live ledger at
  `/home/michael/miner_staging/miner_ledger.db`.
- **6,442,450,944 seed-evaluations completed** (3 of 4 modes × 2^31). The 4th mode
  (`java_lcg_hybrid_reverse`, phase 4) never dispatched — **no phase-4 row exists in the ledger**,
  and there is **no phase-4 `derived_bound` log line**; phases 1–3 each have one.
- **Survivors banked (real, on disk):** java_lcg 0 · java_lcg_reverse 0 · **java_lcg_hybrid 44,331**.
  Constant modes empty, hybrid non-empty — consistent with the design thesis that structure is a
  variable-skip phenomenon.
- **Coordinator end-of-run summary:** `staging_jobs_completed=3948` at `0.818 jobs/s` ·
  `capacity_timeout_terminations=0` · `capacity_invariant_terminations=0` · `pause_events=0` ·
  `inbound_qsize_high_water=975` · `deferred_high_water=1597` (bound 2201). **The staging /
  back-pressure subsystem never faulted, never wedged, never paused.**
- **Provisioning note:** the fleet was **25 of a planned 26 GPUs** — Zeus's second RTX 3080Ti was not
  provisioned. The above was achieved 24×RX6600 + 1×3080Ti.

## 2. THE TERMINAL EVENT AND ITS MECHANISM (read from source at `4643a11`)

**Log:** `class=worker_admission_timeout reason=stage 3 (family 'java_lcg_hybrid_reverse', phase 4)
expected 25 eligible worker(s); 23 admitted after 180.1s (worker_admission_timeout=180.0s)`.

**The gate** (`miner/range_miner_coordinator.py:6732-6781`, the §4.3 admission-liveness repair):
at each new stage, `admission_started_at` is armed; while `len(eligible) < expected_workers` the
stage is **not assigned** and the loop keeps accepting registrations until
`worker_admission_timeout` elapses, then `fail_trial(... TC_WORKER_ADMISSION_TIMEOUT)`.

**The 180 s bound is anchored** (not a magic number): the run set no `worker_admission_timeout`, so
the coordinator resolved it to `DEFAULT_WORKER_ADMISSION_TIMEOUT = 180.0`
(`range_miner_coordinator.py:237`, via `:6410` `context.get(..., DEFAULT_...)` and `:8234-8235`).
The default is deliberately **180 s to match the PWC backend's `_tcp_wait_ready(timeout_s=180.0)`
fleet-startup contract** (`:234`), so the miner and PWC transports agree on how long a fleet may
take to come up. The log confirms the wait actually ran **180.1 s** (`waited > worker_admission_timeout`
at `:6747`) before aborting at 23/25.

**Eligibility** (`:_eligible()`): `[w for w in wconn_by_worker.values() if not w.quarantined]` —
i.e. **workers whose TCP connection is currently bound in the coordinator's registered set.** A
worker whose connection has closed is removed from `wconn_by_worker` and is no longer eligible,
independent of whether its host process or GPU is healthy.

**The worker is one-shot by design** (`miner/range_miner_worker.py:1418-1512`): `main()` does
`connect() → register() → run()`; `run()` loops until it receives a `shutdown` message, then exits.
With **`use_persistent_workers=false`** and a single fleet launch at run start
(`gate12_launch.sh:152 → scripts/launch_fleet_manual.sh`, one invocation), **nothing re-establishes
a worker whose connection drops between stages.**

**Therefore:** across three stage transitions and stage 3's ~57-minute duration (dispatch 09:47:29 →
last shard ~10:41:21), two of the 25 worker connections left the registered set. At the stage-4
admission window only 23 remained eligible; 180 s passed without the pool reaching 25; abort.

**Why "23" is the diagnostic tell:** a clean worker-process death would show **0** eligible; a no-op
reuse of the existing fleet would show **25**. **23** is a partial live-connection count — most of the
fleet was still connected and requalified; two connections were gone. This is a **connection-liveness
gap across stage boundaries**, not a mass exit and not a hardware loss.

## 3. WHAT IT IS *NOT* (each excluded on evidence, not assertion)

- **Not hardware / not GPU lockup.** Post-run, all three rigs report **8/8 GPUs responding**
  (`rocm-smi`), **zero** `dmesg` GPU faults, **no** GCVM_L2 / ring-reset signatures. Worker logs show
  clean kernel compiles, no tracebacks.
- **Not the certified F1/F2 within-stage scheduler.** It did exactly what it was certified to do:
  it refused to assign stage 4 on fewer than the frozen `expected_workers` and failed closed — Beta's
  own §9 condition (*fewer than 25 admitted ⇒ no saturation claim*) enforced by construction.
- **Not staging / back-pressure.** 3,948 jobs, zero capacity terminations, zero pauses, high-waters
  well under bound.
- **Not the `ps=0`-after-the-fact.** All worker processes were absent when checked at ~10:48, **but
  that is 4 minutes AFTER the 10:44 abort**, whose cleanup tears down the fleet. The post-mortem `ps`
  cannot distinguish "died before stage 4" from "torn down by the abort," and is **not** used as
  evidence of the cause. The cause is established from the eligibility mechanism and the
  connection-count, not from the post-abort process table.

## 4. THE FAIL-CLOSED CASCADE WORKED END TO END

admission timeout (10:44:16) → `fail_trial` marks the trial **aborted** → threshold provenance stays
`validated: False` → the integration adapter raises `MinerIngressError` refusing to ingest candidates
or certify a generation (D6 doing precisely its job) → Optuna trial 0 fails with value None → no
`optimal_window_config.json` is written → **WATCHER validates the missing primary output, scores
confidence 0.00, and ESCALATES to human review** (the certified `file_exists` HARD-failure contract,
`CHAPTER_12_WATCHER_AGENT.md` §3.3, `PATCH_watcher_agent_file_validation_v1_1.py`).

**The system refused to fabricate a Step-1 result from an incomplete run. That is the designed
behaviour, and every link executed correctly.** This is a materially better failure than attempt 1:
a named terminal event, full observability (the F-2 repair delivered — the terminal reason is IN the
log this time), and banked live evidence.

## 5. THE SATURATION SUB-RESULT (banked live, valid independent of the abort)

The concurrency sampler ran clean: **2,363 samples, 0 UNOBSERVED.** Verdict file written.

- **VERDICT 1 — SUSTAINED SIMULTANEITY: SATISFIED.** Peak **25** compute-active with **7** queued at
  09:25:10; **6 qualifying windows**; longest **19 samples / 36.2 s** at min-active 25, min-queued 5.
  The frozen 25-worker cohort was demonstrably co-resident with a non-empty queue.
- **VERDICT 2 — TURNOVER UNDER FULL OCCUPANCY: NOT SATISFIED.** In the longest qualifying window,
  occupancy held at 25 but `pending` stayed at 5 and no stripe entered done/staging within that
  window — the queue was full but did not move *during that specific interval*.
- **GATE-12 SATURATION VERDICT: NOT SATISFIED** (the authoritative conjunction; exit 3).

**For Beta's judgement, not asserted by Alpha:** verdict 1 is banked, real, and independent of the
later abort. Whether verdict 2's negative is a genuine turnover gap or an artifact of *which* windows
qualified before stage 4 died — the qualifying windows fall early in each stage, at the claim burst,
before the drain that would move `pending` — is a real question. The S4 identity column is intact for
audit. Alpha takes no position on whether any saturation credit survives a run that did not complete.

## 6. THE DEFECT CLASS, IN BETA'S OWN FRAME

This is the correlated-blind-spot pattern (§2.30): **the gate encoded the same assumption the
implementation did.** The F1/F2 fixtures and every phase-4 acceptance test used small worker counts and
did not exercise **per-stage re-admission of a full 25-worker fleet across three stage transitions in a
single `test_both_modes` trial.** Attempt 1 died in stage 2, before any hybrid transition. **Stage 4 is
the first time in the project's history that a real full fleet had to survive to a fourth admission
round** — and the admission gate's dependence on continuous connection-liveness (rather than
re-registration or the frozen cohort's *identity*) had never been reached. It lies on the queued
adversarial-fixture dimension list Beta already specified (§8): *late joins · disconnect/reconnect ·
`W < N` at a stage boundary.*

## 7. OPEN QUESTIONS ALPHA WILL NOT ANSWER BY GUESSING

1. **Why did those two connections drop?** TCP idle-timeout across stage 3's long staging drain, a
   coordinator-side reaper, or worker-side exit — **not yet isolated.** The definitive evidence is the
   coordinator's per-connection drop/heartbeat trace, which the current log level did not emit for the
   two workers, and rig syslog (needs `sudo`, not yet pulled). Alpha did not manufacture a cause.
2. **Which two.** The TSV `active_workers_json` at the stage-4 admission window shows occupancy flat at
   0 (the fleet was between stages, not compute-active), so the identity column does not name the two;
   isolating them needs the coordinator registration/drop trace, above.

## 8. NO REMEDY PROPOSED — the candidates differ materially and the choice is Beta's (§7 owner rule)

Alpha has NOT changed code and proposes no fix in this report. The candidate directions, for Beta to
rule among, are structurally different:

- **(a) Persist workers across stages** (`use_persistent_workers=true` path, or a worker that
  re-registers on connection loss) — removes the per-stage re-admission requirement entirely, but
  changes the transport model this run deliberately used, and PWC is the path with its own history.
- **(b) Admit on the FROZEN COHORT'S IDENTITY, not live connection count** — a worker in the frozen
  set whose connection dropped is re-admitted when it reconnects, rather than the stage waiting on a
  count of currently-open sockets. Changes the meaning of `_eligible()` at a stage boundary and must
  not weaken the mid-run F1/F2 loss handling the §4.3 repair installed.
- **(c) Relaunch the fleet at each stage boundary** — matches the one-shot worker model, but adds a
  cold-start (kernel recompile) cost inside whatever admission window is set.
- **(d) Widen `worker_admission_timeout`** — the weakest, and now doubly so. It masks the drop rather
  than removing the dependence on continuous liveness, and would let a genuine loss hide behind a
  longer wait (matching the "raising the pool would MASK F-1" reasoning from the attempt-1 forensics).
  Additionally, the 180 s default is **tied to the PWC `_tcp_wait_ready` startup contract**
  (`range_miner_coordinator.py:234-237`), so raising it desynchronizes the two transports'
  fleet-liveness assumptions rather than being a free knob-turn. Alpha flags it as insufficient alone.

These differ in concurrency and correctness properties, so per the §7 owner rule on taking the
structurally stronger mechanism, **the choice is Beta's.** Whatever is chosen must be exercised by an
adversarial fixture with disconnect/reconnect across ≥3 stage transitions at full fleet size — the
dimension that was never generated.

## 9. PRESERVED FOR FORENSIC REVIEW (§13 — nothing repaired, nothing relaunched)

Gate-12 evidence block · WATCHER/coordinator log `logs/gate12_20260810_092341.log` · concurrency TSV
`…_concurrency.tsv` · sampler log · verdict file `…_verdict.txt` · miner ledger
`/home/michael/miner_staging/miner_ledger.db` (+ shm/wal) · the 44,331-survivor staging artifacts ·
S145 coverage state · post-run cursor. **The offered resume command
(`watcher_agent.py --clear-halt --run-pipeline --start-step 1`) was NOT run** — a relaunch inside a
dead authorization is out of scope; this report is the forensic return §13 requires.
