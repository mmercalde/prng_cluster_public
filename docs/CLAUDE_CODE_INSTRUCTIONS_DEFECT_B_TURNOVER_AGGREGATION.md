# CLAUDE CODE INSTRUCTIONS — DEFECT B: SAMPLER ALL-WINDOW TURNOVER AGGREGATION

**Host:** VM101, repo `~/distributed_prng_analysis`, HEAD **`4c76f42`**. `source
~/venvs/torch/bin/activate` before every test.

**Authority:** Team Beta ruling *"GATE-12 ATTEMPT-2 FORENSIC RULING"* (2026-08-10), §§17–20. This
brief implements **Defect B only** — a sampler instrumentation fix. Defect A (worker transport
recovery) is a separate brief.

**Hard constraints — no commit, no push, no launch.** Gate 12 and attempt 3 HELD. **Criterion 1
(simultaneity) must NOT be weakened** (Beta §18). No change to the atomic-snapshot mechanism, the
UNOBSERVED semantics, the S4 identity column, or the exit-code contract — all certified.

---

## THE DEFECT, READ FROM SOURCE

`scripts/gate12_concurrency_sampler.py`, `evaluate()`: after finding all qualifying windows it sets
`qualifying = best if satisfied else None`, where `best = max(windows, key=len …)` — the **single
longest** window — and calls `_turnover(qualifying)`. **Turnover is therefore evaluated ONLY in the
longest qualifying window.**

The Gate-12 turnover question is **existential, not longest-window** (Beta §17): *did **any** valid
qualifying full-occupancy window contain turnover?* The current code asks *did the **longest** one?* —
a **false negative** whenever a shorter qualifying window holds real turnover and the longest is static.
Attempt 2 exhibited exactly the risk shape: 6 qualifying windows, turnover judged only on the 19-sample
longest (which was static). Attempt 2 failed regardless (stage 4 never ran), but the evaluator must be
correct before attempt 3.

## WHAT TO BUILD — Beta §18

**Criterion 1 unchanged.** Continue to identify **all** qualifying simultaneity windows exactly as now
(≥ `min_window_samples` consecutive satisfying observed samples; gaps still break windows).

**For each qualifying window independently**, compute the full `_turnover` field set: `pending_drained`,
`transitions`, `done_delta`, `min_active`, `min_queued`, `start`, `end`, `sample_count`.

**Aggregate existentially:**

```
TURNOVER_SATISFIED = EXISTS qualifying window WHERE pending_drained > 0 OR transitions > 0
```

**Overall unchanged in form:**

```
GATE12_SATURATION_SATISFIED = SIMULTANEITY_SATISFIED AND TURNOVER_SATISFIED
```

`overall_satisfied()` and `exit_code()` keep their current shape — only the turnover input becomes the
existential aggregate instead of the longest-window result.

**Witness reporting (Beta §18 + §19-B5):** the summary must identify the **actual turnover witness
window** — the qualifying window that satisfied turnover — chosen **deterministically**. Document the
rule (Beta's example: *earliest qualifying turnover window*; any documented deterministic rule is
acceptable — state which). **Do NOT silently substitute the longest window.** The longest window may
still be printed as context, clearly labelled as context, not as the turnover basis.

## REQUIRED SAMPLER GATES — Beta §19, all five

- **B1 — longest static, shorter turnover (the defect reproduction):** longest qualifying window has no
  turnover; a shorter qualifying window has turnover → **SIMULTANEITY SATISFIED · TURNOVER SATISFIED ·
  OVERALL SATISFIED.** This is the arm that reds under the current longest-only code.
- **B2 — turnover outside occupancy:** pending drains while occupancy drops below threshold → **not
  credited** (the drain is not inside a qualifying window; the step-wise-inside-window rule already
  handles this — prove it survives the multi-window change).
- **B3 — UNOBSERVED gap:** a gap separates qualifying observations → no turnover inferred across it (a
  window broken by a gap is two windows; neither may borrow the other's movement).
- **B4 — no qualifying turnover anywhere:** several qualifying static windows → **TURNOVER NOT
  SATISFIED · OVERALL NOT SATISFIED.** (Attempt-2's actual shape, if §20 reanalysis confirms it.)
- **B5 — deterministic witness:** multiple windows contain turnover → the witness is chosen and labelled
  by the documented deterministic rule, and is stable across runs of the same input.

Red-first + mutation evidence per arm. **B1 is the load-bearing red-first:** it must FAIL against the
current longest-only implementation and PASS after the fix; a mutant reverting to longest-only must red
B1. B4 is the clean control (the fix must not turn a genuine all-static run into a false positive).

## §20 FORENSIC REANALYSIS — authorized, read-only, non-certifying

After the evaluator is correct, run it **read-only against a COPY of the attempt-2 TSV**
(`logs/gate12_20260810_092341_concurrency.tsv` → copy first; never mutate the preserved evidence) to
answer: **did any of the six qualifying windows other than the longest contain real turnover?**

State the result plainly. It is **forensic only** — Beta §20/§21: it does **not** retroactively convert
attempt 2 into a Gate-12 pass (the trial aborted before stage 4, no publication, no coverage advance).
Report it as: "the corrected evaluator finds turnover in window N / finds no turnover in any qualifying
window" — a fact about the instrumentation, not a certification.

## VERIFICATION

`test_gate12_concurrency_sampler.py` — all existing arms (currently 44/44) **must stay green** (the S3
authority line, S4 identities, atomic snapshot, UNOBSERVED semantics are certified and untouched) **plus
B1–B5.** If any existing arm changes behaviour, that is a regression in something certified — STOP and
report it rather than editing the arm. Long suites `python3 -u <suite> | tee /tmp/<n>.log`, never `tail`.

## REPORT — `docs/CLAUDE_CODE_REPORT_DEFECT_B_TURNOVER_AGGREGATION.md`

1. The per-window computation and the existential aggregate, with source lines.
2. The deterministic witness rule chosen, and where the witness (vs the longest-as-context) is rendered
   in the summary — with a **verbatim summary sample**.
3. B1–B5 with red-first/mutation evidence; B1 shown failing pre-fix and passing post-fix; B4 the control.
4. Confirmation the 44 certified arms are unchanged and green; confirmation criterion 1 is untouched.
5. §20 reanalysis result against the attempt-2 TSV copy — labelled forensic-only, non-certifying.
6. Files changed from `git status`. Any conflict with a certified surface **returned, not worked
   around.**
