# CLAUDE CODE INSTRUCTIONS — GATE-12 FAILURE FORENSICS (READ-ONLY)

**Host:** VM101, repo `~/distributed_prng_analysis`, HEAD `a3bb4da`.
**Run under investigation:** `distributed_config_t1_689f3cd9`, 2026-08-09 12:37:09 → 12:47:17.
**Authority:** Team Beta ruling *"GATE-12 FAILED EXECUTION"* (2026-08-09) — FAIL accepted,
**read-only forensics AUTHORIZED**, rerun NOT authorized, production changes NOT authorized.

## CONSTRAINTS

**No fleet run. No GPU work. No launching anything. No production edits. No commit, no push.**
Do not start `watcher_agent.py`, `window_optimizer.py`, the fleet script, any worker, or bind 5700.

**Do not mutate the failed run's evidence.** Beta permits a **forensic copy** of databases/logs for
analysis **provided you record hashes** (or equivalent identity) tying each copy back to the
original. Work on copies; leave `/home/michael/miner_staging/miner_ledger.db` and
`logs/gate12_20260809_123705.log` untouched.

## THE ONE THING THAT MATTERS

Alpha framed the question as *"why did stage 2 cancel 6 of 32 stripes?"* **Beta says that is the
wrong question and Alpha accepts the correction.**

Under the frozen Phase-4 contract, **workflow phases 1 and 2 are constant-mode: any stripe failure
or lease expiry fails the trial IMMEDIATELY — no retry.** Whole-trial abort cleanup then marks
every still-pending/active stripe `cancelled`. **One real failure therefore produces exactly a
`26 done / 6 cancelled` tail.**

> **BINDING FORENSIC QUESTION: what was the FIRST authoritative event that put the stage-2 trial
> into terminal-failure / TrialAbort state?**

**Work forward from the earliest terminal event in time. Do NOT start from the six `cancelled`
rows and reason backward** — Beta prohibits that explicitly. Interpret the cancellations only
*after* the initiating event is identified.

**Beta §8, important:** if a genuine stage-2 stripe failure or lease expiry is found, **immediate
terminal failure may be CORRECT behaviour, not a retry defect** — the retry-to-another-worker path
exists for retryable failures in hybrid phases 3/4 only. Do not report "retry failed" unless the
phase policy actually permitted one. If the policy behaved correctly, the remaining question
becomes *why did that worker/stripe fail*, which is a different finding.

---

## A. STAGE-2 CAUSAL RECONSTRUCTION (the actual work)

For **all 32 stage-2 stripes**, reconstruct from the ledger and logs:

```
stripe_id · seed interval · worker_id · attempt · claimed_at ·
last heartbeat / lease state · StripeComplete time (if any) · error event (if any) ·
connection-loss event (if any) · state immediately before trial abort ·
final state · cancellation timestamp/reason
```

Then identify the earliest event that made continued stage-2 execution impossible, and classify
the initiating condition as one of:

```
StripeErrorMessage · retryable=False error · lease expiry ·
connection loss leading to lease expiry · staging failure · staging timeout ·
capacity invariant failure · coordinator exception · protocol/frame error ·
explicit trial abort · other terminal path
```

Deliver: the first terminal event with **exact timestamp**, the **exact source branch** that
handled it (`file:line`), worker/stripe/attempt identity, the event immediately before and after,
the `TrialAbort` timestamp, and **proof** that the six cancellations were either abort-cleanup
consequences **or** independently caused.

## B. THE 15-SECOND "DEFECT 6" CONNECTION DROP

At 12:41:08 the coordinator logged
`dropping connection that never completed a frame within 15.0s read deadline (Defect 6)`.

**Status per Beta: CORRELATED, causation NOT established.** One timestamp is not proof. Answer:

1. which `worker_id` owned that TCP connection?
2. was that worker part of the frozen 8-worker cohort?
3. did it hold a stage-2 stripe at the time — if so, which stripe and attempt?
4. did the disconnect itself classify the stripe as failed, or did a later lease expiry?
5. what exact source branch handled the condition?
6. was `TrialAbort` emitted from that branch?
7. did the six cancellations occur **after** that abort?

**Classify explicitly as: CAUSAL · CONTRIBUTORY · UNRELATED · UNRESOLVED**, with evidence. If the
connection belonged to a late non-cohort worker, or to a connection with no active stage-2 claim,
it can be **excluded** as root cause — say so plainly.

## C. `GPU_COUNT_MISMATCH: 0/8`

Preflight warned `0/8` for all three rigs (and still passed 3/3) while the cluster bot
independently reported **8/8 GPUs OK on every rig at 12:36**, one minute earlier.

**Beta: the bot's 8/8 is NOT sufficient proof that the preflight detector is wrong. Compare the
two detection paths directly.** Determine: the exact function emitting the warning · the exact
command/API it uses to count GPUs · its stdout, stderr and return code · the environment visible
to the probe · whether it counts physical GPUs, worker daemons, ROCm-visible devices, or something
else · why `0/8` did not fail the 3/3 preflight.

**Return exactly one disposition:**

```
A. warning is advisory and accurately named/documented
B. detector defect identified
C. environment/probe defect identified
D. the two probes measure different things
```

**Do not leave it at "probably cosmetic."** Re-running the probe read-only on the rigs is permitted
(it is a query, not fleet work).

## D. FUTURE LAUNCH SHAPE — DOCUMENT ONLY, DO NOT LAUNCH

State the prospective parameter set. It is already known; just record it accurately:

```
worker_pool_size = 25      ← the correction; manifest default 8 was never overridden
seed_start       = 0
seed_count       = 2147483648      (2^31)
miner_stripe_size = 67108864       (2^26)   → 32 stripes/stage
test_both_modes  = true
prng_type        = java_lcg   ·  window_trials = 1  ·  n_parallel = 1
use_range_miner  = true       ·  use_persistent_workers = false
```

Confirm from source that `worker_pool_size` has a live CLI route
(`agent_manifests/window_optimizer.json:38`, default at `:262`) and that passing it in `--params`
reaches `--worker-pool-size`. **Note whether raising it to 25 changes the derived retention
requirement** (at 8 workers it derived 6,528; at 25 the per-stage conservative bound will differ) —
report the number the derivation would produce, computed, not guessed.

## E. CONCURRENCY SAMPLER ORDERING

The sampler in `gate12_launch.sh` starts in step 4, **after** the fleet-launch step returns — so it
began at 12:51 for a run that died at 12:47, and produced **no in-run rows**. Alpha's tooling error.

Show the corrected ordering: the sampler must be **active before the coordinator can issue the
first `StripeAssign`**. Provide the revised block. Beta requires future saturation evidence to
contain an observation window with **≥25 distinct in-flight workers AND queued stripes still
available** — "distinct workers eventually used = 25" is explicitly insufficient.

## F. PRODUCTION-CHANGE CLASSIFICATION

Conclude with exactly one:

```
NO PRODUCTION DEFECT FOUND — rerun requested
PRODUCTION DEFECT FOUND — amendment submitted
```

If a defect exists, describe it — **do not implement it**; it needs separate Beta review before any
rerun.

---

## REPORT

`docs/CLAUDE_CODE_REPORT_GATE12_FAILURE_FORENSICS.md`, sections A–F in that order. Every claim
anchored with `file:line`, a ledger query, or a log line + timestamp. **"Cannot be determined from
the available evidence" is a valid and preferred answer** over inference — say what would be needed.
Record the forensic-copy hashes. Report any disagreement with this brief rather than working around
it.
