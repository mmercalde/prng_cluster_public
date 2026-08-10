# TEAM ALPHA → TEAM BETA — DEFECT B: SAMPLER TURNOVER AGGREGATION — DELTA REVIEW

**Per your Gate-12 attempt-2 forensic ruling §§17–20.** Defect B (all-qualifying-window turnover
aggregation) is implemented. **Defect A (worker transport recovery) is separate and in progress —
this submission is Defect B only.**

**State:** uncommitted in the VM101 working tree at HEAD `4c76f42` (accepted attempt-2 forensic
lineage). **Nothing committed, pushed or launched** — per your §24, commit awaits Michael's
direction. Gate 12 and attempt 3 remain HELD. Report: `docs/CLAUDE_CODE_REPORT_DEFECT_B_TURNOVER_AGGREGATION.md`.

**Suite: `test_gate12_concurrency_sampler.py` 49/49** — the **44 certified arms unchanged and green**
plus 5 new turnover-aggregation arms. Verified independently by Alpha: the sampler diff read in full,
and the §20 finding reproduced from the preserved TSV's window-1 rows by an independent calculation
(not the replay harness).

---

## 1. THE DEFECT (§17) AND THE FIX (§18)

`evaluate()` fed `_turnover` the single **longest** qualifying window (`best if satisfied else None`),
so turnover was judged only there — a false negative whenever a shorter qualifying window holds real
turnover and the longest is static.

The fix splits **measurement** from **verdict**:

- **`_window_turnover(window)`** — per-window measurement, the step-wise `pending_drained` /
  `transitions` arithmetic carried **verbatim** from the certified `_turnover`. It also now produces
  the `windows_detail` census, so the per-window census and the verdict are **one computation** and
  cannot disagree about what a window held.
- **`_turnover(measurements)`** — the existential aggregate, exactly §18:
  `TURNOVER_SATISFIED = EXISTS qualifying window WHERE pending_drained > 0 OR transitions > 0`
  (`satisfied = bool(hits)`).
- **`overall_satisfied` / `exit_code` unchanged in form** — only the turnover input became the
  existential aggregate. `GATE12_SATURATION = SIMULTANEITY AND TURNOVER` holds.

**Criterion 1 is untouched** — the window-cutting and `satisfied` logic are byte-unchanged; the
longest window `best` survives only as `longest_window_*`, explicitly rendered `CONTEXT ONLY … NOT
the turnover basis`.

**The measured interval did NOT widen with the aggregation** — each `_window_turnover` is confined to
one window's consecutive pairs, so a drain during an occupancy dip (B2) or across a gap (B3) still lies
between windows, inside none, and remains uncreditable. This is the correctness point that keeps
widening-to-all-windows from re-introducing a run-wide-drain false positive.

## 2. WITNESS (§18 + §19-B5)

Deterministic: `TURNOVER_WITNESS_RULE` = **the earliest qualifying window, by start epoch, that shows
turnover** (`witness_index = hits[0] + 1`; `measurements` is temporally ordered, so earliest = first
hit, stable across identical input). The summary marks it `<-- TURNOVER WITNESS`, prints **every**
qualifying window's own `turnover: drained/transitions -> YES/no` (so the existential claim is
checkable against named intervals), and labels the longest window CONTEXT ONLY. The longest is never
silently substituted.

## 3. GATES (§19) — all five, red-first + mutation

- **B1 — longest static, shorter turnover (the reproduction):** longest qualifying window static, a
  shorter one has turnover → SIMULTANEITY · TURNOVER · OVERALL all SATISFIED. **Red-first proven
  whole-file:** `git show HEAD:scripts/...` (longest-only) gives NOT SATISFIED / exit 3 on the fixture;
  the working tree gives SATISFIED / exit 0. An in-arm mutant reconstructing longest-only reds it.
- **B2 — turnover outside occupancy:** drain while occupancy < threshold → not credited (movement is
  between windows).
- **B3 — UNOBSERVED gap:** a gap splits the observations → no turnover inferred across it.
- **B4 — no qualifying turnover anywhere (the control):** several static qualifying windows → TURNOVER
  NOT SATISFIED · OVERALL NOT SATISFIED. Guards against the fix turning genuine all-static into a false
  positive.
- **B5 — deterministic witness:** multiple turnover windows → the earliest is chosen and labelled,
  stable across runs.

**Criterion-1 non-weakening proven by differential:** 20 criterion-1/census keys + 6 `windows_detail`
subkeys compared pre-vs-post across 13 fixtures — **zero differential** except `turnover_satisfied` on
the B1 reproduction fixture.

## 4. §20 FORENSIC REANALYSIS — read-only, non-certifying

Ran the corrected evaluator against a **copy** of the preserved attempt-2 TSV (original byte-unchanged,
`sha256` and mtime verified). **Finding: turnover WAS present — in qualifying window 1** (09:25:10–:14):
three consecutive samples at `compute_active=25`, `pending 7→6→3` (**drained 4**), `done 0→1→4`
(**transitions 4**).

**Alpha confirmed this independently** — not by trusting the replay harness, but by recomputing the
step-wise aggregate from the window-1 rows Michael read live during the run (same values): drained 4,
transitions 4, min_active 25 → `turnover=True`.

**Therefore attempt-2's banked `VERDICT 2: NOT SATISFIED` was a longest-window-selection artifact, not
a fact about the fleet.** The cluster *was* demonstrating scheduler turnover under full saturation; the
old evaluator judged only the 19-sample static longest window and missed it.

**This is forensic only and does not rescue attempt 2** (your §20/§21): the trial aborted at stage
3→4, no publication, no coverage advance, no Gate-12 pass. What it establishes for attempt 3 is that the
saturation machinery and the fleet behaviour are sound — the corrected evaluator is what was missing,
and attempt 3 should be expected to clear verdict 2 on real evidence.

## 5. TWO DISCLOSURES

**(a) Gate naming — DB1–DB5, not B1–B5.** Your ruling names the arms B1–B5, but the suite already has
**certified ESTAB arms B1–B7**. To avoid shadowing certified names, the new arms are **DB1–DB5** with
a one-to-one mapping to your B1–B5 documented in the suite docstring. The certified B1–B7 are untouched.
Trivially renamed if you prefer different letters.

**(b) `windows_detail` shape gained turnover fields.** `windows_detail` entries are now the full
`_window_turnover` dict (they previously carried only samples/seconds/start/end/min_active/min_queued).
This is additive — every prior key is present and unchanged — and it is what makes the census and the
verdict one computation. The certified F1 self-describing arm's required-element list still passes;
the additions are supersets. Offered for you to reverse if the coupling is unwanted.

## 6. FILES CHANGED

`scripts/gate12_concurrency_sampler.py` · `tests/test_gate12_concurrency_sampler.py` — two files, the
scope §18/§19 defines. No other sampler surface touched; no coordinator, miner, ledger or certified
non-sampler file touched. `git status` is the authority for the commit's `git add` list.
