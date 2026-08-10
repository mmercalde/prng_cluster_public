# CLAUDE CODE INSTRUCTIONS — PRE-RERUN ITEMS, REVISION 1 (SAMPLER ONLY)

**Host:** VM101, repo `~/distributed_prng_analysis`, HEAD **`c4e0037`**. The two pre-rerun items
are **uncommitted in the working tree**. `source ~/venvs/torch/bin/activate` before every test.

**Authority:** Team Beta ruling *"PRE-RERUN ITEMS REVIEW"* (2026-08-09) — **GPU probe CERTIFIED;
sampler returned; Gate-12 rerun NOT authorized from this tree.**

**CERTIFIED AND CLOSED — do not touch:** the GPU probe repair in `preflight_check.py`
(three-outcome status, `UNAVAILABLE` ≠ 0 with `gpu_count None`, located binary, stderr surfaced,
advisory gating preserved at `:370`, the render guard at `:190-200`). **Do not modify it, do not
re-verify it, do not let sampler work touch it.**

**Also closed:** the sampler's post-F1 **state meanings** — `state='claimed'` for occupancy with
staging excluded, `pending` as real backlog, run-scoping, `mode=ro`, production-DB refusal by name,
arming before the coordinator, terminating with the run. **Those are right. Only the measurement
implementation and the verdict change.**

**Hard constraints:** no commit, no push, **no launch, no fleet, no port 5700 bind**; Gate 12
HELD. Do not modify the coordinator, miner, ledger, seed-domain/coverage surface, or any certified
suite.

**Base verification:** `git log --oneline -1` = `c4e0037`; `tests/test_preflight_gpu_probe.py`
**12/12**; `tests/test_gate12_concurrency_sampler.py` **14/14**.

---

## A. BLOCKER — each sample must be ONE atomic snapshot (Beta §3)

**Alpha verified this in source before briefing it.** `sample_run` issues **two separate
`conn.execute` calls in autocommit mode** (`scripts/gate12_concurrency_sampler.py:176` and
`:184`), with no explicit transaction:

```
read 1 → DISTINCT claimed_by WHERE state='claimed'      (the occupancy set)
read 2 → SELECT state, COUNT(*) GROUP BY state          (the queue depth)
```

Under WAL these are two **independent read transactions**. A stripe transitioning between them
yields a sample whose occupancy and queue depth **describe different instants**.

**Why this matters more than it sounds:** the verdict asks whether ≥25 workers were compute-active
**while** stripes remained queued. **The single sample that decides the verdict is the one most
likely to be internally inconsistent**, because the interesting window is precisely when
transitions are occurring — 32 stripes, 25 workers, turnover in progress. **Both a false positive
and a false negative are reachable.**

### Required

Wrap the reads of one sample in an **explicit read transaction** so both observe the same WAL
snapshot — e.g. `BEGIN` / `BEGIN DEFERRED` before the reads and `COMMIT` (or equivalent
`isolation_level` handling) after, on the read-only connection. Prefer the smallest change that
provably gives one snapshot; **state in the report which mechanism you used and why it is
sufficient under WAL.**

### Gate

Prove the sample is atomic — e.g. an interleaving fixture that mutates stripe state **between**
what would have been the two reads, and assert the emitted sample is self-consistent (the occupancy
set and the counts agree with one another). A mutant restoring two autocommit reads must red.

## B. BLOCKER — the ESTAB count must be honest or absent (Beta §4)

`estab_count` returns **`0` when `ss` is unavailable or fails**, which is the identical
"unobservable rendered as a definite zero" defect the GPU probe was just certified for fixing —
**reproduced inside the evidence tool for the same run.**

### Required

**Apply the same discipline the probe now uses.** `ss` unavailable, non-zero exit, or unparseable
⇒ a distinguishable **`UNAVAILABLE` / `None`**, never `0`. Render it as unavailable in the TSV and
summary — never as `0`.

**ESTAB remains context only, not occupancy.** Beta: it *"is not part of the saturation criterion
and must not silently degrade the evidence file."*

### Gate

`ss` missing ⇒ recorded unavailable, **not 0**; `ss` non-zero exit ⇒ unavailable; unparseable ⇒
unavailable; and in every case the **saturation verdict is unaffected**, since ESTAB is not a
criterion term.

## C. THE VERDICT OMITS TURNOVER (Beta §5)

The current predicate — ≥25 compute-active **and** `pending > 0`, sustained — proves the queue was
**non-empty**. It does **not** prove the queue was **consumed**.

**You chose 32 stripes over the 25-stripe minimum precisely so seven queued stripes would exercise
scheduler turnover, completion, reassignment, staging and back-pressure under full occupancy.** A
run that holds 25 claimed and 7 pending without ever draining would satisfy the present predicate
while demonstrating none of that.

### Required — report BOTH, and keep them distinct

1. **Sustained simultaneity** (existing) — the qualifying window, unchanged.
2. **Turnover under full occupancy** (new) — during the qualifying window, evidence that pending
   work was **actually consumed while occupancy remained at the threshold**. At minimum:
   - `pending` **strictly decreased** across the window, and/or
   - stripes transitioned into `done`/`staging` while `compute_active` stayed ≥ the threshold;
   - report the **number of such transitions** and the **pending delta** across the window.

**Do not collapse the two into a single pass/fail.** A run may satisfy simultaneity and fail
turnover; that distinction is the point, and the summary must make it legible. State each verdict
separately and label clearly which criterion each satisfies.

### Gate

A fixture with 25 claimed / 7 pending held **static** across the window ⇒ simultaneity SATISFIED,
turnover **NOT** satisfied. A fixture where pending drains while occupancy holds ⇒ **both**
satisfied. A fixture reaching 25 only across different instants ⇒ neither (existing arm, retained).

## D. A TEST THAT WILL RED IMMEDIATELY AFTER COMMIT (Beta §6)

`test_preflight_gpu_probe.py` anchors its mutation-authenticity check to **`HEAD`** —
`git show HEAD:preflight_check.py`. That is true **now**, while the change is uncommitted. **The
moment it is committed, `HEAD` becomes the mutated source and the check inverts.**

This is the same class as `G-MATRIX-DIFF-a`, which red on its own success when `4b1aad6` became
HEAD.

### Required

Re-anchor to a **stable, explicit baseline** — the commit that predates the change (`c4e0037`),
pinned by hash, in the manner already used elsewhere in this repo (`git show <hash>:path`). **Do
not anchor to `HEAD`.** Verify by proving the check still behaves correctly when the working tree
is committed — state in the report how you demonstrated that without committing.

**This file is otherwise CERTIFIED — change only the anchor, nothing else.**

---

## VERIFICATION

- `tests/test_gate12_concurrency_sampler.py` — all existing arms plus the new atomicity, ESTAB-
  honesty and turnover arms;
- `tests/test_preflight_gpu_probe.py` — **12/12 unchanged in substance**, anchor corrected;
- red-first and mutation evidence for each new arm;
- **no fleet execution, no launch.**

`gate12_launch.sh` may need the sampler invocation updated if its interface changes — **deliver it,
do not run it**, and confirm again that port 5700 stayed unbound and ledger mtime unchanged.

## REPORT

`docs/CLAUDE_CODE_REPORT_PRERUN_R1.md`:

1. The snapshot mechanism, and why it is sufficient under WAL.
2. The ESTAB unavailable path, and proof the verdict is unaffected by it.
3. The two separate verdicts, with the exact predicate for each.
4. The re-anchored mutation check, and how post-commit correctness was demonstrated.
5. Red-first / mutation evidence per new arm.
6. Confirmation the GPU probe's certified logic is byte-unchanged apart from the anchor.
7. Files changed. **Any disagreement reported, not worked around.**
