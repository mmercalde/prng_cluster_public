# CLAUDE CODE INSTRUCTIONS — PRE-RERUN ITEMS, REVISION 2 (CLOSE EVERYTHING)

**Host:** VM101, repo `~/distributed_prng_analysis`, HEAD **`c4e0037`**. Pre-rerun work is
**uncommitted in the working tree**. `source ~/venvs/torch/bin/activate` before every test.

**Authority:** Team Beta ruling *"PRE-RERUN R1 REVIEW"* (2026-08-10). **This is the last blocking
revision before Gate-12 authorization. Close every item below — nothing may be deferred, and
nothing beyond them may be touched.**

**CERTIFIED AND CLOSED — do not modify, do not re-verify:** the GPU probe repair in
`preflight_check.py` · the `read_snapshot` atomicity fix and its A1/A2/A3 arms · the ESTAB
observability work and its B7 invariance arm · the `_PRE_FIX_REV` anchor.

**Hard constraints:** no commit, no push, **no launch, no fleet, no port 5700 bind**; Gate 12
HELD; do not modify the coordinator, miner, ledger, seed-domain/coverage surface, or any certified
suite.

**Base verification:** HEAD `c4e0037`; `test_preflight_gpu_probe.py` **12/12**;
`test_gate12_concurrency_sampler.py` **29/29**.

---

## 1. BLOCKER — VIR-5 ON THE LEDGER READ (the one that actually matters)

**Alpha applied VIR-5 to ESTAB — a context field — and never applied it to the ledger read, which
is where the criterion lives. That substitution is Alpha's error, and this item is the correction.**

**Alpha verified the failure path in source; it is worse than "sampling stops":**

```python
row = {... "compute_active": 0, "queued_pending": 0 ...}     # :687-691
try:
    ... row.update(sample_run(conn, run_id))
except sqlite3.Error as e:
    # "record it and keep the sample out of the verdict"      ← the comment
    print(...)                                                # :700-703
row.update(estab_observation(...))
satisfies = (row["compute_active"] >= threshold and row["queued_pending"] >= 1)
out.write(...)
if run_id: samples.append(row)                                # ← enters the verdict anyway
```

**A failed ledger read is written as a definite `compute_active=0, queued_pending=0` observation
and appended to `samples`.** The comment states the correct intent; the code does the opposite.
This is the *exact* defect the GPU probe was certified for fixing, reproduced inside the tool
measuring the same run.

Beta: **"A saturation verdict computed from an unknown number of missing samples is not
evidence."**

### Required

1. **A failed or unavailable ledger read must be recorded as UNOBSERVED — never as zero
   occupancy.** Distinct status in the row, distinct rendering in the TSV, and **excluded from the
   verdict**, exactly as the comment already claims.
2. **Count and surface them.** The summary must report **how many samples were unobserved**, and
   the verdict must be computed over a **known** denominator.
3. **A window containing unobserved samples cannot silently qualify.** Either the window is broken
   at the gap, or the verdict is explicitly annotated with the gap — **state which rule you
   implemented and why.** Do not let an unknown interior gap pass as continuous.
4. Apply the same treatment to any other read that can fail mid-run (`discover_run_id`, the
   connection itself).

### Gate

Inject a ledger read failure mid-run: the affected sample is **UNOBSERVED, not `0`** · it does
**not** enter the verdict as a zero · the unobserved count appears in the summary · a window
spanning the gap does not qualify silently. **A mutant restoring the fall-through must red.**

## 2. TURNOVER — remaining prerequisites from the previous ruling

The direction is right; these were omitted. Close them:

1. **Define the turnover window precisely** — is it the qualifying simultaneity window, or the
   whole run? State it in the summary output, not only in the report.
2. **Report the exact predicate for each verdict** in the summary itself, so the evidence file is
   self-describing without the report beside it.
3. **`pending_delta`, `transitions`, `done_delta` must be reported over the SAME window the
   verdict uses** — not over the run — or the numbers describe a different interval than the claim.
4. **A monotonic `pending` decrease alone is not turnover under full occupancy.** The evidence must
   pair consumption **with** sustained threshold occupancy across the same samples.

**Do not collapse the two verdicts.** Exit codes `0 / 2 / 3` stay as implemented.

## 3. THE RESIDUAL COMMENT-STRIPPING ASSERTION — Beta authorized it; add it

Beta ruled the one-line addition **authorized**. Add the comment-stripping assertion to `M1A` so
the mutation-authenticity check cannot be satisfied by the legacy string quoted in commentary at
`preflight_check.py:62`.

**One line. Nothing else in that file changes.** Demonstrate it still behaves correctly post-commit
using the throwaway-clone method already used, and confirm repo HEAD, reflog and refs unchanged.

## 4. SELF-DESCRIBING EVIDENCE

The TSV and summary must stand alone as evidence. At minimum the summary must state: the threshold
used · the sample interval · total samples · **unobserved samples** · the qualifying window(s) with
their sample counts · both verdicts with their exact predicates · the turnover window definition ·
and the exit code's meaning.

**Beta will read this file without the report next to it. Write it for that reader.**

---

## VERIFICATION

- `tests/test_gate12_concurrency_sampler.py` — all existing arms **plus** the unobserved-read arms
  and the turnover-window arms;
- `tests/test_preflight_gpu_probe.py` — **12/12**, plus the M1A comment-stripping assertion;
- red-first and mutation evidence for every new arm;
- `gate12_launch.sh` — deliver updated **only if** the sampler interface changed; **do not run it**;
  confirm port 5700 unbound and ledger mtimes unchanged.

**No fleet execution. No launch.**

## REPORT

`docs/CLAUDE_CODE_REPORT_PRERUN_R2.md`:

1. The unobserved-read status, how it is excluded from the verdict, and the window rule you chose
   for gaps — with the reasoning.
2. The turnover window definition and the four prerequisite closures.
3. The M1A assertion and its post-commit demonstration.
4. A **verbatim sample** of the summary output, so Beta can judge whether it is self-describing.
5. Red-first / mutation evidence per new arm.
6. Confirmation that every CERTIFIED item above is byte-unchanged.
7. Files changed. **Any disagreement reported, not worked around.**

**Nothing in this brief is optional. Beta has stated this is the last blocking revision before
Gate-12 authorization — close every item.**
