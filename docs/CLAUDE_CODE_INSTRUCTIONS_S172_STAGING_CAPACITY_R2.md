# CLAUDE CODE INSTRUCTIONS — S172 STAGING-CAPACITY AMENDMENT, REVISION 2

**Host:** VM101, repo `~/distributed_prng_analysis`. The amendment + R1 are **uncommitted in the
working tree** at base `c7058d8`. `source ~/venvs/torch/bin/activate` before every test. Long
suites: `python3 -u <suite> | tee /tmp/<name>.log` — never `| tail`.

**Authority:** Team Beta ruling *"S172 STAGING-CAPACITY R1"* (2026-08-08). **R1 ACCEPTED IN
SUBSTANCE — two narrow corrections required before commit.**

**CLOSED — do not reopen, do not re-argue, do not "improve":** crash-resumable commit cleanup ·
stage-specific eligibility implementation · the 16-stripe geometry and the 3,264 derivation ·
Gate 37 supersession · `elapsed_s` (R4) · preflight-plan *content* · the anti-drift single-derivation
design. Beta: *"Do not reopen the staging amendment generally."*

**Beta withdrew one of its own claims.** Its earlier ruling said the submitted code reused the
*first stage's filtered eligible set* and could understate later stages. **Withdrawn** — Alpha
showed `serve_trial._eligible()` returns all connected non-quarantined workers, so the old
calculation reused a **superset** and could only over-estimate. This matters for correction 4 below.

**Beta §9 — make ONLY these five changes** (2 production, 3 gates). *"No `.gitignore` cleanup. No
new telemetry. No seed-domain/cursor changes. No byte-model work. No Gate-12 production run. No
Phase-7 soak."*

**Hard constraints:** no commit, no push, **no pipeline launch, no fleet launch, no port 5700
bind.** Gate 12 and the Phase-7 soak are HELD. If a fix appears to need a file outside the existing
change set, STOP and report.

**Base verification:** working tree still carries amendment + R1 — `git log --oneline -1` =
`c7058d8`, the amendment diffstat intact, `test_s172_staging_backpressure.py` **48/48**,
`test_s172_elapsed_roundtrip.py` **6/6**. Untracked runtime residue is expected and is not a stop
condition. Stop only on unexpected tracked drift.

---

## PRODUCTION CHANGE 1 — FREEZE THE TRIAL'S ASSIGNABLE COHORT AT PREFLIGHT (Beta §4)

**The seam:** stage-specific eligible sets are resolved once, before first dispatch, but worker
membership can change afterward. A worker joining later was never counted in the derived
requirement.

**Beta's correction to Alpha's history, accepted:** the old all-connected calculation did **not**
universally make late joiners safe — it covered a late worker only when that worker could not
introduce a **tighter applicable cap** than the population present at preflight. (E.g. a CUDA-only
population at preflight, then a tighter-cap ROCm worker joins.) So this is **not** a hazard R1
created; R1 makes the contract precise enough to close it.

**The ruling — admission freeze, not connection refusal.** Beta: *"freeze eligibility for the
trial, not worker connectivity globally. A new daemon can come online and register normally; it
simply cannot alter the execution geometry of a trial whose safety bound has already been certified
and persisted."*

Implement exactly:

1. At **successful** retention preflight, the stage-specific worker identities in
   `preflight_plans.per_stage` become **frozen** for that run.
2. Every later `assign_stripes` eligibility calculation **for that trial** must intersect its live
   workers with the frozen stage set.
3. A completely new worker identity may register with the coordinator, but is eligible only for a
   **subsequent** trial.
4. A frozen identity that **reconnects** may re-enter the same trial **only if its relevant
   advertised capabilities/caps still match the frozen preflight contract.** Mismatch ⇒ excluded
   from that trial.
5. **Losing frozen workers does not enlarge the cohort.** Existing retry / no-alternate behaviour
   governs the reduced live set.

Target invariant, and the point of the whole preflight:

```
actual worker used by trial  ⊆  worker population used to derive the trial retention ceiling
```

**Do NOT re-preflight mid-trial** (Beta rejected it — undermines the one-time whole-trial admission
decision, the immutable persisted provenance, and the claim that the ceiling was sufficient before
work began; plus concurrency and partial-retention questions). **Do NOT add a conservative margin
for hypothetical fleet members** (Beta rejected it — it needs an authoritative capability
description for non-participating workers; freeze the concrete verified population instead).

## PRODUCTION CHANGE 2 — FAIL CLOSED WHEN THE PREFLIGHT PLAN CANNOT BE PERSISTED (Beta §5)

The current code catches a `record_preflight_plan` failure, logs it, and **admits the trial
anyway** — and a gate explicitly requires that behaviour. Beta: that contradicts the ruling. *"The
durable plan was not optional telemetry."* A trial cannot satisfy both *"must be durably persisted
before dispatch"* and *"if persistence fails, dispatch anyway."*

**Two cases, both required:**

**A. Trial would otherwise be ADMITTED** — provenance write failure is **fail-closed before the
first `StripeAssign`**. Suggested classification:

```
coordinator_staging_preflight_provenance: unable to durably persist retention plan
```

It is a **coordinator/infrastructure** failure, not a worker failure: no retry-matrix charge, no
stripe assignment, no result traffic, no partial execution.

**B. Trial is ALREADY REFUSED by retention sizing** — the sizing refusal remains **primary**. A
failure to persist the refusal record must **not** turn the refusal into admission and must not
mask its root classification. Attach the provenance failure as secondary evidence; the terminal
cause stays:

```
coordinator_staging_retention_sizing
```

Beta's framing to preserve: *"failure to write the audit record may not override a safety refusal,
but inability to create the mandatory audit record prevents a would-be admission."*

---

## GATE CORRECTION 3 — LATE-WORKER EXCLUSION (Beta §4, new arm)

One focused arm:

1. preflight with workers A/B;
2. persist the stage-specific execution set;
3. introduce/register worker **C after preflight**, where C **would materially alter the
   conservative bound** (i.e. a tighter applicable cap — make this real, not cosmetic);
4. prove **C cannot receive a `StripeAssign` for that trial**;
5. prove **C remains usable by a later/new trial**;
6. optionally, reconnect A under the same worker identity and prove it is admissible **only when
   its relevant capability signature matches the frozen record.**

**No re-derivation should occur** — assert that too.

## GATE CORRECTION 4 — FIX `G-MUT-STAGE-ELIGIBILITY` (Beta §3)

The current mutant claims to restore *"the SUBMITTED behaviour — resolve ONE eligible collection
and reuse it for every planned stage,"* but it obtains that collection by taking the **first
stage's resolved eligible population** and copying it across. **That is Beta's withdrawn
hypothesis, not what the submitted code did.**

Change the mutant to reproduce the **real** previous behaviour:

```
for every planned stage:
    eligible[stage] = ALL candidate workers passed to the old calculation
                      (the all-connected, non-quarantined collection)
```

Then assert the actual result. Under the asymmetric fixture this is expected to be **different
from, and generally more conservative than**, the exact stage-resolved result.

**Do NOT require it to understate. Do not manufacture a safety failure the previous code did not
have.** The gate's purpose is now:

> exact-variant stage semantics are preserved, **and** the old all-connected-population calculation
> is **detectably different**.

This is a verification-integrity correction, not an architectural blocker.

## GATE CORRECTION 5 — REPLACE THE PROVENANCE-FAILURE ARM (Beta §5)

Delete the current *"a provenance-write failure must not change the decision … the trial is still
admitted"* arm. Replace with **two**:

- **admission + provenance failure ⇒ zero `StripeAssign`, fail closed** with the §5-A
  classification;
- **sizing refusal + provenance failure ⇒ still sizing-refused, never admitted**, terminal cause
  remains `coordinator_staging_retention_sizing`.

---

## VERIFICATION BEFORE RESUBMISSION

- staging-backpressure suite: all prior gates green **plus** the late-worker arm and the two
  replacement provenance arms; the corrected `G-MUT-STAGE-ELIGIBILITY` green under its new
  purpose;
- `test_s172_elapsed_roundtrip.py` **6/6**;
- `test_s172_staging_partb.py` **24/24**;
- phase-4 **63/63** clean/committed, Gate 22 and Gate 37 green;
- **red-first evidence** for the two new production behaviours (cohort freeze, fail-closed
  provenance) against the current R1 tree;
- pre-amendment gate assertions unchanged except the already-authorized Gate-37 supersession —
  prove by AST, the method already used;
- no gate-12 production run; no Phase-7 soak.

## REPORT

`docs/CLAUDE_CODE_REPORT_S172_STAGING_CAPACITY_R2.md`:

1. Per-change implementation notes with `file:line`.
2. How the cohort freeze is enforced at every later eligibility calculation for the run, and how
   capability-signature matching on reconnect is decided (what exactly is compared).
3. The provenance failure policy as implemented, both cases, with the terminal classifications.
4. Red-first evidence for both production changes; the corrected mutant's before/after result under
   the asymmetric fixture.
5. Full verification results.
6. Files changed — expect the same set plus test additions. Anything else justified.
7. Any disagreement with this brief **reported, not worked around.**
