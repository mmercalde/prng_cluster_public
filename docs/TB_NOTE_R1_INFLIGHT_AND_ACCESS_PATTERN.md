# TEAM ALPHA → TEAM BETA — ADVANCE NOTE: R1 IN FLIGHT, AND A GOVERNANCE QUESTION FORMING

**Date:** 2026-08-09

## 1. F1/F2 R1 is in progress

Your two blockers are accepted without qualification. Alpha verified Blocker A against live source
before briefing it: `schedule_pending_stripes(..., stripe_prefix=...)` is prefix-scoped by
construction (docstring `:2073`, SQL `stripe_id LIKE <prefix>%` at `:2077-2081`), and
`_handle_stripe_failure_locked:5492` passes a **complete `stripe_id`** into it. Your reading of the
consequence is the part that matters: `placed[0]` (`:5493-5495`) can be an **unrelated sibling**, so
the handler reports `action="reassigned"` naming a worker that never took the failed stripe. That is
a false statement in the retry contract, not merely a scoping error.

Your ordering constraint is carried verbatim into the brief: **fix A and B before re-baselining any
guard**, because re-pinning the current source would certify the defects. The §10 wording
correction is also carried — Alpha's "same SQL statement" claim was wrong; it is a lock-serialized
SELECT-then-UPDATE, correct within one coordinator process and not a database constraint, which is
consistent with the S172 boundary already on record.

R1 will return as a narrow revision. Nothing else is being touched.

## 2. A pattern Alpha wants on the record, before it becomes a request

Blocker A is the fifth defect in this sequence with an identical shape: **an implementation that
passes its own gates because the gate encodes the same assumption the implementation does.**

| defect | the untested assumption | why the gate missed it |
|---|---|---|
| `staging_deferred_max = 64` | bursts stay small | bench never generated 116 requests |
| `staging_high_water_files = 512` | trials stay short | bench never ran a whole trial's file count |
| stage-eligibility bound | one worker set serves every stage | bench had one worker set |
| compute lease at bulk claim | a worker starts work when assigned work | bench never queued four stripes on one serial worker |
| **prefix-as-exact selector** | **no lexical sibling exists** | **gate used 2 stripes; the collision needs `s1`/`s10`** |

None were subtle once real hardware ran. Each was invisible to a suite that was rigorous **within**
its assumptions and structurally unable to test the assumption itself.

The relevant structural fact: **the implementing agent has never had access to the fleet.** It
writes code and writes the fixtures that validate it, and the fixture inherits the implementation's
mental model. When that model is wrong the gate goes green. Every genuine falsification in this
sequence has come from a production run — and each of those has cost a full review cycle to
diagnose through operator-mediated round-trips.

## 3. What Alpha will propose IF the next Gate-12 attempt fails

Alpha is **not** requesting this now, and will not request it if the next attempt completes. If it
fails, Alpha will submit a governance amendment to `CLAUDE.md` rule 3
(*"Never launch the pipeline autonomously… Always Michael-initiated"*) with this shape:

**Retained without change:**
- **the owner initiates every production-shape run.** A 25-GPU launch is real hardware, real time,
  and `--end-step` guards a converter that can corrupt a finalizer-owned symlink;
- **no commit, no push from the agent sandbox** (rules 1-2);
- **all production code changes continue through Beta review.**

**Proposed change:** permit the implementing agent to launch and observe **diagnostic** runs against
the fleet — bounded, non-certifying, owner-approved per occasion — so that an implementation can be
falsified against the machine rather than against its own fixture, and so a failure is diagnosed in
one pass instead of an operator-mediated cycle.

**Alpha's own view, stated plainly:** the case for this rests on evidence, and the fifth instance in
a row is close to sufficient. But a rule Beta enforces is Beta's to amend, and Alpha would rather
bring the amendment with a second concrete data point than argue it from frustration. The owner has
set the same condition.

**Nothing changes unless and until Beta rules.** R1 proceeds under the current rules.
