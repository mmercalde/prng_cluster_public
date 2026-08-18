# TB RULING — GATE-12 ATTEMPT 9 ACCEPTANCE

**Received:** 2026-08-17
**Applies to:** run `distributed_config_t1_554463d3`, nonce `gate12-20260817_181819-46500`,
launch HEAD `e9ca800ae65b44bc555f1402a9102932ba2e72ca`
**Recorded by:** Team Alpha, verbatim below. No Alpha edits to the ruling body.
**Companion evidence:** `docs/SESSION_CHANGELOG_20260817_GATE12_ATTEMPT9.md` and the
attempt-9 artifact set `logs/gate12_20260817_181819*` (preserved off-git per this ruling's
sequencing item 1).

**Binding dispositions:**

| item | disposition |
|---|---|
| Gate-12 Attempt 9 | **PASSED** |
| Coverage/cursor certification | **ACCEPTED** |
| MP-1 drain-starvation defect (R-1..R-4) | **CLOSED** — do not reopen R-1 |
| Field 1 (pump exclusive `<100 s`) | **MISSED as written**; per-call collapse is corroborating evidence only, not a rewritten pass |
| Field 2 (serve-thread staging `<60 s`) | **MISSED as written**; `staging/msg` ratio ruled non-binding; causal question closed |
| Field 6 (`deferred_distinct_attempts_high_water`, `pump_liveness_probes_high_water`) | **UNOBSERVED — instrumentation-output defect.** Repair required (bounded, observability-only, own gate). **No Gate-12 rerun.** |
| Publication symlinks / accumulator | No Gate-12 blocker. NPZ payloads stay OUT of git. `.s172_accumulator/generations/` is durable data plane — needs backup/recovery policy (git is not that backup). `.gitignore` negation removal = hygiene, when convenient |
| Window-anchor production merge | **UNBLOCKED** |
| Phase 7 | **UNBLOCKED** |

**Ruled sequence:**
1. Anchor `e9ca800` as the Gate-12-passing production commit; preserve the Attempt-9 forensic bundle.
2. Window-anchor production merge against that certified anchor.
3. Field-6 logging repair — separate, observability-only commit with its own small gate. Must land **before the next production-class Step-1 run** (i.e., before the Phase 7 soak). No admission, lease, pump, queue, cursor, publication, or scheduler logic changes; no new performance optimization. Do not fold heartbeat-disposition or missing-expiry-summary work into it.
4. Phase 7.

**Mandated phrasing for the complexity result until the falsifiers are first observed in
production:**

> R-3's scaling model is gate- and benchmark-certified and strongly corroborated by Attempt 9's
> per-call cost, but its two dedicated production falsifier fields were not persisted and
> therefore were not observed in Attempt 9.

---

## Ruling body (verbatim)

## BETA → ALPHA — RULING: **GATE-12 ATTEMPT 9 PASSED. DRAIN-STARVATION DEFECT CLOSED. FIELD-6 OBSERVABILITY REPAIR REQUIRED, NO RERUN.**

I accept Attempt 9 as the first successful Gate-12 production-class run.

The important state transition is now complete: fresh zero-credit cursor → four completed stages / 128 stripes → saturation satisfied at 25 compute-active → **zero lease expiries** → certified `[0, 2^31)` coverage for both modes → cursor advanced to one certified interval. The `--end-step 1` boundary also remains clean: an evaluator saying "Triggering Step 2" is not execution of Step 2.

### Field 1 — **the literal prediction did not pass; do not renormalize it into a pass**

The predeclared acceptance table said roughly:

> staging-thread `pump` exclusive: ~3,640 s → **<100 s**

and, separately, predicted pump calls would remain approximately unchanged, ±20%. The report explicitly said a large call-count change means something else moved.

Attempt 9 gives 2,394.1 s. Therefore:

**Field 1 = NOT MET AS WRITTEN.**

I will not substitute "−85.6% per call" for the original `<100 s` criterion after seeing the result.

But I also **do not interpret the miss as refuting the remedy**. The original cumulative-total comparison depended on roughly stable pump-call population, and that precondition failed spectacularly: Attempt 9 executed about **4.6× more pump calls** while actually completing the workload that MP-1 died partway through.

That makes the cumulative thread-second total a poor cross-run causal comparator here.

The per-call result is therefore admissible as **diagnostic corroboration**, not as a rewritten gate:

`1.463 s/call → 0.210 s/call`, an ~85.6% reduction.

More importantly, the 0.210 s observed cost at a deferred HWM of 1,679 is close to the R-3 pathological-shape prediction of 0.1364 s at 1,700 entries. That is precisely the scaling mechanism we changed. The MP-1 baseline was ~3,640 pump-thread seconds with ~3.76/4 staging threads effectively consumed by `_pump_deferred`.

So the ruling is:

> **Prediction miss, causal mechanism supported. Do not reopen R-1.**

The absolute pump total remains worth watching as a future efficiency opportunity, but it is no longer producing the lock-starvation failure.

### Field 2 — **also not met literally; the staging/msg ratio is not the criterion**

The declared prediction was:

> serve-thread staging: 681.2 s → **<60 s**

Attempt 9 produced **159.9 s**.

Therefore:

**Field 2 = NOT MET AS WRITTEN.**

Again, no post-hoc conversion.

But the `staging/msg` ratio of `0.988 → 0.880` is **not** a useful failure signal and was never the binding threshold. Structurally, most sub-result message handling legitimately passes through staging; that ratio can remain high even after the staging operation becomes much cheaper.

The meaningful movement is:

`681.2 s → 159.9 s`, **−76.5% absolute**, and approximately `70.3% → 21.7%` of the serve-loop wall time.

That is a major reduction in the lock-bearing path, even though it missed the aggressive `<60 s` forecast. The original MP-1 forensic identified those 681.2 seconds as the serve thread's side of the same lock-contention conjunction.

And unlike MP-1, Attempt 9 ended with **25/25 connections serviced, queue zero, zero lease expiry**. That operational result outranks whether our pre-run magnitude forecast was too aggressive.

So:

> **Prediction miss, remedy not refuted. No additional drain remedy is authorized.**

### Why I am closing the defect despite those two misses

The direct failure signature is gone.

MP-1 had the pathological growth in per-frame message cost, partial drain behavior, late connection starvation, a persistent queue, and a valid 300-second lease expiry. Its attribution showed the pump/serve-lock chain directly.

Attempt 9, by Alpha's calibrated extraction, instead shows:

* per-frame message cost finishes **below its starting value**;
* the ~400× buildup is gone;
* drain throughput rises;
* **no work-bearing window is starved**;
* terminal service is **25/25**;
* terminal queue is **0**;
* **zero lease expiries**;
* all 128 stripes complete.

That is the mechanism-level acceptance result.

The original report's failure criteria explicitly treated continued per-frame buildup, continued ~99% partial draining, or renewed capacity/pause terminations as evidence that the remedy had failed. Those signatures did not survive.

**MP-1 drain starvation is CLOSED.**

---

## Field 6 — **real observability defect; repair it, but do not rerun Gate 12**

This one is different.

The two fields were added for one explicit reason:

> make the complexity guarantee falsifiable on the production acceptance run.

The report even says that a guarantee a production run cannot refute is not an acceptance criterion, and identifies `deferred_distinct_attempts_high_water` and `pump_liveness_probes_high_water` as the production falsifiers.

They were computed but never persisted.

Therefore:

**Field 6 = UNOBSERVED, caused by an instrumentation-output defect.**

Not zero. Not passed. Not failed.

And yes, it warrants its own **small observability repair pass**.

Do **not** rerun Attempt 9 merely to obtain them. Gate 12 already passed, and nothing about absence of those two records invalidates the completed workflow or the direct drain-starvation evidence.

The repair should be narrowly bounded:

1. Add both values to the existing staging-backpressure summary/output that is already persisted.
2. Add a gate that drives two different pump populations and proves the emitted values vary appropriately—mere key presence is insufficient.
3. Pin `UNOBSERVED` behavior if no pump pass occurred.
4. No admission, lease, pump, queue, cursor, publication, or scheduler logic changes.
5. No new performance optimization.

Then let the **next naturally required production Step-1 run** provide the first production observation of those fields.

Until then, phrase the complexity result precisely:

> **R-3's scaling model is gate- and benchmark-certified and strongly corroborated by Attempt 9's per-call cost, but its two dedicated production falsifier fields were not persisted and therefore were not observed in Attempt 9.**

That is the complete truth.

---

## Publication symlinks — no Gate-12 blocker

I do not see a reason to reopen Gate 12 over the three post-run paths as described.

The relevant facts are that the admission/pre-dispatch clean-tree gates passed, publication and cursor advancement completed first, and these paths are now publication symlinks into a per-generation accumulator rather than tracked survivor blobs.

Do **not** put the NPZ payloads back into git merely to satisfy the historical rule that applied when `binary.npz` was a regular tracked file.

But the operational consequence Alpha identified is real:

> `.s172_accumulator/generations/` is now part of the durable data plane.

It needs backup/recovery policy appropriate to an authenticated generation store. Git is no longer that backup.

The stale `.gitignore` negation can be removed as hygiene when convenient; it is not a Gate-12 issue.

---

# Sequencing ruling

**Gate 12 is no longer blocking Phase 7 or the window-anchor production merge.**

I want the sequence to preserve the exact passing artifact:

1. **Record/anchor `e9ca800` as the Gate-12-passing production commit and preserve the Attempt-9 forensic bundle.**
2. **Perform the window-anchor production merge against that certified anchor.**
3. **Do the field-6 logging repair as a separate, observability-only commit with its own small gate. No Gate-12 rerun required.**
4. **Proceed into Phase 7.**

The logging repair need not delay Phase-7 design/work if those can occur independently, but it must land **before the next production-class Step-1 run**, otherwise we knowingly throw away the same falsifiers twice.

Do not fold heartbeat-disposition or missing-expiry-summary work into that repair.

---

## Final status

**Gate-12 Attempt 9: PASSED.**

**Gate-12 coverage/cursor certification: ACCEPTED.**

**MP-1 / drain-starvation production defect: CLOSED.**

**Field 1:** literal `<100 s` prediction **MISSED**; call-count precondition invalidated; per-call collapse is corroborating evidence, not a rewritten pass.

**Field 2:** literal `<60 s` prediction **MISSED**; `staging/msg` ratio is non-binding; large absolute/share reduction plus restored service closes the causal question.

**Field 6:** **UNOBSERVED — logging defect. Repair required, rerun not required.**

**Window-anchor production merge: UNBLOCKED.**

**Phase 7: UNBLOCKED.**
