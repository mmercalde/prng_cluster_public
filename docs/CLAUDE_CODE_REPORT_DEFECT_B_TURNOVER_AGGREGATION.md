# REPORT — DEFECT B: SAMPLER ALL-WINDOW TURNOVER AGGREGATION

**Host:** VM101, repo `~/distributed_prng_analysis`, HEAD **`4c76f42`** (unchanged, not reverted).
**Authority:** Team Beta, *"GATE-12 ATTEMPT-2 FORENSIC RULING"* (2026-08-10) §§17–20.
**Scope:** Defect B only. Defect A (worker transport recovery) untouched.
**State:** nothing committed, nothing pushed, nothing launched. Gate 12 and attempt 3 remain HELD.

---

## 0. HEADLINE — read this before §5

**The §20 forensic reanalysis contradicts the brief's expectation, and this is the most important
result in the report.** The brief anticipated (§19-B4) that attempt 2 was an all-static run. It was
not. **The corrected evaluator finds real turnover in attempt-2 qualifying window 1** — pending
7→6→3 and done 0→1→4 across three consecutive samples all at compute_active=25.

**The defect did not merely risk a false negative in attempt 2. It produced one.** The banked
sub-verdict `VERDICT 2 — TURNOVER UNDER FULL OCCUPANCY : NOT SATISFIED` is an artefact of
longest-only aggregation, not a fact about the fleet. Details and the non-certifying framing: §5.

---

## 1. THE PER-WINDOW COMPUTATION AND THE EXISTENTIAL AGGREGATE

### 1.1 The defect, confirmed against live source before editing

The brief's cited site was verified in the working tree at `4c76f42` before any edit. Pre-fix
`scripts/gate12_concurrency_sampler.py`:

| Line (pre-fix) | Code | Effect |
|---|---|---|
| 636 | `best = max(windows, key=lambda w: (len(w), …), default=None)` | the single longest window |
| 650 | `qualifying_windows = [w for w in windows if len(w) >= min_window_samples]` | all qualifying windows — computed, and used only for the census |
| 660 | `qualifying = best if satisfied else None` | **the defect**: all-but-one discarded |
| 664 | `**_turnover(qualifying)` | verdict 2 decided on the longest window alone |

No conflict with the brief. The line numbers and the code shape matched exactly.

### 1.2 The split: measurement vs. verdict

The fix separates a per-window **measurement** from the **aggregation** that turns measurements
into verdict 2. This is what makes the existential quantifier expressible without touching the
interval any single measurement covers.

**`_window_turnover(window)` — `scripts/gate12_concurrency_sampler.py:494`.** Pure measurement of
one window, no verdict. Carries the step-wise arithmetic **verbatim from the certified pre-fix
`_turnover`** — the `zip(window, window[1:])` loop, the `max(0, …)` drain term, the
`(done+staging)` transition term. Returns the full field set Beta §18 requires — `pending_drained`,
`transitions`, `done_delta`, `min_active`, `min_queued`, `start`, `end`, `sample_count` (as
`samples`) — plus `seconds`, `pending_first/last/delta`, and a per-window `turnover` boolean.

It is also now the **single source of truth for `windows_detail`** (`:735`). The summary's
per-window census and the turnover verdict are one computation, so the two can never disagree about
what a window held.

**`_turnover(measurements)` — `:553`.** Takes the list of per-window measurements and aggregates:

```python
hits = [i for i, m in enumerate(measurements) if m["turnover"]]   # :583
satisfied = bool(hits)                                            # :634
```

which is exactly Beta §18:

```
TURNOVER_SATISFIED = EXISTS qualifying window WHERE pending_drained > 0 OR transitions > 0
```

**Call site — `evaluate()`, `:735–739`:**

```python
windows_detail = [_window_turnover(w) for w in qualifying_windows]
return {
    "satisfied": satisfied,
    **_turnover(windows_detail),
```

`qualifying = best if satisfied else None` is **deleted**. `best` survives only as the
`longest_window_*` context fields.

### 1.3 Why widening the aggregation does not widen any measured interval

Each measurement is confined to one window by `_window_turnover`, whose pairs are drawn from a
single window's sample list. No step ever spans the stretch between two windows — the dip or gap
that separated them is precisely the unproven interval, and nothing crossing it is counted. This is
why DB2 and DB3 still hold. Documented at `:509–513` and `:571–579`.

### 1.4 Unchanged surfaces

`overall_satisfied()` and `exit_code()` keep their exact prior bodies — only their
`turnover_satisfied` input changed meaning. The atomic-snapshot mechanism, UNOBSERVED semantics,
the S4 identity column, and the exit-code contract are untouched.

---

## 2. THE WITNESS RULE, AND WHERE IT IS RENDERED

### 2.1 The rule

`TURNOVER_WITNESS_RULE` — `scripts/gate12_concurrency_sampler.py:490`:

> **the EARLIEST qualifying window, by start epoch, that shows turnover**

Beta's own example, adopted. It is deterministic: windows are cut from the sample sequence in
temporal order, so `measurements` arrives ordered and the earliest hit is `hits[0]`. No set
iteration, no `max()` over a tie-prone key. Stability on identical input is asserted by DB5.

**When no window shows turnover** there is no witness (`turnover_witness_window = None`). The
`turnover_*` scalar fields then describe the **earliest qualifying window**, reported explicitly as
*"no witness exists; shown as the earliest qualifying window"* — never as a basis that satisfied
anything. The verdict is NOT SATISFIED either way.

**The longest window is never substituted.** It is printed under its own heading and explicitly
demoted (`:931`).

### 2.2 New fields

`turnover_witness_rule`, `turnover_witness_window`, `turnover_basis_window`,
`turnover_qualifying_windows`, `turnover_windows_with_turnover`. All pre-existing `turnover_*` field
names and meanings are retained.

### 2.3 Verbatim summary sample

From DB5's fixture (3 qualifying windows: window 1 static, window 2 drains 3, window 3 is the
**longest** and also drains 3). The witness must be window 2, not window 3:

```
    a window SHOWS TURNOVER  <=>  pending_drained > 0  OR  transitions > 0
    VERDICT 2 satisfied  <=>  THERE EXISTS a qualifying window that shows
      turnover. The test is EXISTENTIAL over all qualifying windows, not
      a property of the longest one: a shorter window in which the queue
      genuinely moved is evidence that it was consumed under full
      occupancy, and the longest window being static is not evidence
      against it. Each measurement stays confined to a single window, so
      widening the aggregation does not widen any measured interval.
      Every counted step is bracketed by two samples that are themselves
      at or above the threshold, so consumption is paired with sustained
      occupancy across the SAME samples. A run-wide monotonic decrease
      does not qualify on its own.
    WITNESS RULE: when satisfied, the window credited is
      the EARLIEST qualifying window, by start epoch, that shows turnover.
      A deterministic choice, stable across runs of identical input,
      and named below so the claim can be checked against an interval
      rather than taken on trust.

runs of satisfying samples               : 3
  of those, QUALIFYING (>= 2 samples)         : 3
    window 1: 2 samples, 2.0s, T+0s -> T+2s, min_active=25 min_queued=7
      turnover: drained=0 transitions=0 done_delta=0 -> no
    window 2: 3 samples, 4.0s, T+6s -> T+10s, min_active=25 min_queued=4
      turnover: drained=3 transitions=0 done_delta=0 -> YES   <-- TURNOVER WITNESS
    window 3: 4 samples, 6.0s, T+14s -> T+20s, min_active=25 min_queued=1
      turnover: drained=3 transitions=0 done_delta=0 -> YES
longest window                           : 4 samples, 6.0s
  from / to                              : T+14s -> T+20s
  min compute-active within window       : 25
  min queued within window               : 1
  CONTEXT ONLY. The longest window is NOT the turnover basis and never
  substitutes for the witness below — verdict 2 quantifies over every
  qualifying window, and the longest holds no special standing in it.

-- verdict 2 evidence: turnover WITHIN a qualifying window --
qualifying windows examined              : 3
  of those, showing turnover             : 2
TURNOVER WITNESS                         : window 2
  witness rule                           : the EARLIEST qualifying window, by start epoch, that shows turnover
turnover basis window                    : window 2 (THE WITNESS — this window's own movement is what satisfies verdict 2)
  measured over                          : 3 samples, T+6s -> T+10s
  (a qualifying simultaneity window, not the whole run)
  min compute-active across it           : 25
pending at window start / end            : 7 -> 4
  pending DRAINED step-wise in window    : 3
  endpoint delta (context only)          : 3
stripes transitioned into done/staging   : 0
  of which reached done                  : 0
finding                                  : queued work was consumed under full occupancy in qualifying window 2 of 3 (3 drained, 0 transitions)
```

Every qualifying window's turnover is printed, not only the credited one: an existential verdict is
checkable only if a reader can see each window it quantifies over.

---

## 3. THE FIVE GATES — DB1–DB5

### 3.0 Naming — please note

Beta numbers these gates **B1–B5**. They are named **DB1–DB5** in the suite because the certified
ESTAB arms already own the labels `B1-SS-MISSING-UNAVAIL` … `B5-OBSERVED-ZERO-IS-ZERO` (and B6, B7).
Two arms answering to "B1" in one evidence log is itself a defect in the log. **DBn is Beta's Bn,
one for one**, and the mapping is stated in the suite docstring. The certified B1–B7 names were not
touched. Raise it if you want different letters.

| Gate | Arm (`tests/test_gate12_concurrency_sampler.py`) | Result |
|---|---|---|
| B1 | `DB1-SHORTER-WINDOW-TURNOVER-CREDITED` (`:972`) | PASS |
| B2 | `DB2-DIP-DRAIN-NOT-CREDITED-MULTIWINDOW` (`:1018`) | PASS |
| B3 | `DB3-GAP-FORBIDS-BORROWED-TURNOVER` (`:1048`) | PASS |
| B4 | `DB4-NO-TURNOVER-ANYWHERE-CONTROL` (`:1080`) | PASS |
| B5 | `DB5-WITNESS-DETERMINISTIC-AND-LABELLED` (`:1113`) | PASS |

### 3.1 DB1 — the load-bearing red-first

**Fixture.** Five samples at 25-active with pending flat at 7 (window 1, the **longest**), a dip to
3 active, then three samples at 25-active draining 7→5→3 (window 2, **shorter**, drains 4).

**Red-first, whole-file.** The same fixture through the pre-fix evaluator loaded from
`git show HEAD:scripts/gate12_concurrency_sampler.py` and through the working tree, side by side:

```
B1 fixture: 2 qualifying windows — window 1 = 5 samples static (LONGEST),
            window 2 = 3 samples, pending 7 -> 3 under full occupancy

PRE-FIX  (HEAD 4c76f42, longest-only)
    criterion 1 (simultaneity) : SATISFIED
    criterion 2 (turnover)     : NOT SATISFIED
    GATE-12 SATURATION         : NOT SATISFIED
    exit code                  : 3
    turnover measured over     : 5 samples  (witness: n/a — no witness concept)
    pending drained            : 0

POST-FIX (working tree, existential)
    criterion 1 (simultaneity) : SATISFIED
    criterion 2 (turnover)     : SATISFIED
    GATE-12 SATURATION         : SATISFIED
    exit code                  : 0
    turnover measured over     : 3 samples  (witness: 2)
    pending drained            : 4
```

**B1 fails against the current longest-only code and passes after the fix.** Criterion 1 reports
SATISFIED in both — the change is isolated to verdict 2.

**In-arm mutation.** `_longest_only(v)` (`:957`) reconstructs the pre-fix aggregation from the
verdict's own `windows_detail` — the same aggregator, differing only in the set it quantifies over —
and DB1 asserts `mutant["turnover_satisfied"] is False`. Built from live data rather than a saved
copy of the old source so the mutant cannot drift from the code under test. A revert to longest-only
reds DB1 twice over: once on the verdict, once on the mutant differential.

### 3.2 DB2 — turnover outside occupancy, re-proved multi-window

Pending falls 10→2 precisely while occupancy dips to 3, splitting the run into two qualifying
windows. Run-wide the queue drained 8. Both windows report `pending_drained == 0`; the aggregate is
NOT SATISFIED, exit 3, no witness. Widening the aggregation is the one change that could plausibly
leak movement in from outside occupancy, so the property is re-proved against a **multi-window**
fixture rather than inherited from certified E2.

### 3.3 DB3 — UNOBSERVED gap

Five samples at 25-active, pending 10,10,**GAP**,4,4. The gap splits them into two windows; pending
is 10 before and 4 after, and nothing in the evidence file can say whether that drain happened under
full occupancy or while the fleet stood empty. Neither window may borrow the other's endpoint —
both report `pending_drained == 0`, aggregate NOT SATISFIED. **Clean control in the same arm:** the
identical fixture with that sample OBSERVED is one window draining 6, and is SATISFIED. The gap is
doing the work, not the fixture.

### 3.4 DB4 — the control

Three qualifying windows separated by dips, every one static at 25-active/7-pending.
`turnover_windows_with_turnover == 0`, no witness, `turnover_basis_window == 1` (earliest, stated as
such), authority line NOT SATISFIED, exit 3, `WHY NOT (verdict 2)` present, and the summary renders
`TURNOVER WITNESS : NONE`. The fix does not turn a genuinely static run into a false positive.

### 3.5 DB5 — deterministic witness

Three qualifying windows: window 1 static, window 2 drains 3, window 3 drains 3 **and is the
longest**. Asserted: per-window turnover is `[False, True, True]`; the longest is window 3; the
witness is **window 2**; `turnover_basis_window == 2`; `turnover_window_samples == 3` (window 2's
geometry, not window 3's 4); `"EARLIEST"` appears in the rule; the summary carries
`TURNOVER WITNESS : window 2`; the `<-- TURNOVER WITNESS` marker appears **exactly once**; the rule
text is self-described in the file; the longest is labelled `CONTEXT ONLY`. **Stability:** the same
input evaluated and rendered twice produces byte-identical output.

---

## 4. THE 44 CERTIFIED ARMS, AND CRITERION 1

### 4.1 Suite results

```
baseline (before any edit) : 44/44 checks green
final    (after the fix)   : 49/49 checks green
```

Arm-by-arm diff of the 44 certified arm names and results, baseline vs. final, excluding only the
five new DB arms:

```
IDENTICAL: same 44 arms, same order, same results
```

**No certified arm was edited.** The test-file diff is purely additive: the docstring header, one
helper (`_longest_only`), five new arm functions, and five new `main()` calls. In particular
**F1 was not touched** — its 19-element self-describing list is unchanged, so the new witness-rule
text is asserted by DB5 rather than by extending a certified arm.

Also run: `tests/test_gate12_gpu_gate.py` — **9/9 green** (it reads the sampler's invocation line out
of `gate12_launch.sh`, which was not modified).

### 4.2 Criterion 1 is untouched — differential, not assertion

Criterion 1 code is byte-identical: the `satisfies` predicate, the window-cutting loop, the
`satisfies is True` gap rule, `best`, and `satisfied = best is not None and len(best) >= …` were not
edited. That was verified by differential rather than by inspection alone.

For each of 13 fixtures — every turnover-relevant fixture in the certified suite plus all five new
ones — the pre-fix and post-fix evaluators were run on identical input and **20 criterion-1 / census
keys** (`satisfied`, `window_count`, `qualifying_window_count`, `longest_window_*`, `peak_*`,
`samples_*`, `distinct_workers_union`, …) plus **6 pre-existing `windows_detail` subkeys per
qualifying window** were compared:

```
  OK   C1 static            crit1 identical   | turnover: same
  OK   C2 drain             crit1 identical   | turnover: same
  OK   C3 union             crit1 identical   | turnover: same
  OK   C4 transitions       crit1 identical   | turnover: same
  OK   E1 not-the-run       crit1 identical   | turnover: same
  OK   E2 dip drain         crit1 identical   | turnover: same
  OK   E3 refill            crit1 identical   | turnover: same
  OK   D2 gap               crit1 identical   | turnover: same
  OK   B1 defect repro      crit1 identical   | turnover: PRE=False -> POST=True
  OK   B2 multiwindow dip   crit1 identical   | turnover: same
  OK   B3 gap borrow        crit1 identical   | turnover: same
  OK   B4 all static        crit1 identical   | turnover: same
  OK   B5 two witnesses     crit1 identical   | turnover: same

RESULT: criterion 1 IDENTICAL across every fixture — zero differential
```

**The only behavioural change anywhere is `turnover_satisfied` on the defect-reproduction fixture**,
which is the entire point of the change. Criterion 1 is not weakened; it is not altered at all.

---

## 5. §20 FORENSIC REANALYSIS — READ-ONLY, NON-CERTIFYING

### 5.1 Evidence handling

Run against a **copy**. The preserved file was never opened for write and is byte-identical after
the work:

```
sha256  4f69dba7c44e35eb44c78ee44981855f6c4f79f36897e46839643af362c874b2   (before and after)
mtime   2026-08-10 10:44:18.149888926 -0700   size 376933                  (before and after)
```

2404 rows; 2363 replayed (rows with a latched `run_id`, reproducing exactly which rows `main()`
appended to `samples`). Parameters `threshold=25`, `min_window_samples=2`, read from the run's own
`logs/gate12_20260810_092341_verdict.txt`.

### 5.2 Replay fidelity

Before trusting the corrected reading, the **pre-fix** evaluator was replayed over the same
reconstructed samples and reproduces the banked verdict exactly:

| | replay | banked verdict |
|---|---|---|
| qualifying windows | 6 | 6 |
| longest window | 19 samples | 19 samples |
| verdict 1 | SATISFIED | SATISFIED |
| verdict 2 | NOT SATISFIED | NOT SATISFIED |
| overall | NOT SATISFIED | NOT SATISFIED |

The reconstruction is faithful, so what the corrected evaluator says about the same samples is about
the aggregation and nothing else.

### 5.3 The answer

**The corrected evaluator finds turnover in attempt-2 qualifying window 1.**

```
  window 1:  3 samples,   4.0s, 09:25:10 -> 09:25:14
            pending 7 -> 3, drained=4 transitions=4 done_delta=4  ->  turnover YES
  window 2:  6 samples,  10.1s, 09:25:20 -> 09:25:30   pending 1 -> 1, drained=0 transitions=0  ->  no
  window 3:  7 samples,  12.1s, 09:34:19 -> 09:34:31   pending 6 -> 6, drained=0 transitions=0  ->  no
  window 4:  4 samples,   6.0s, 09:47:29 -> 09:47:35   pending 7 -> 7, drained=0 transitions=0  ->  no
  window 5:  5 samples,   8.1s, 09:47:45 -> 09:47:53   pending 6 -> 6, drained=0 transitions=0  ->  no
  window 6: 19 samples,  36.2s, 09:48:06 -> 09:48:42   pending 5 -> 5, drained=0 transitions=0  ->  no
            <-- LONGEST (the only window the pre-fix code measured)
```

Confirmed directly against the raw preserved TSV rows, independent of the replay harness:

```
44:2026-08-10T09:25:10 | obs=OBSERVED active=25 pending=7 claimed=25 staging=0 done=0 sat=1
45:2026-08-10T09:25:12 | obs=OBSERVED active=25 pending=6 claimed=25 staging=0 done=1 sat=1
46:2026-08-10T09:25:14 | obs=OBSERVED active=25 pending=3 claimed=25 staging=0 done=4 sat=1
47:2026-08-10T09:25:16 | obs=OBSERVED active=24 pending=2 claimed=24 staging=0 done=6 sat=0
```

Three consecutive `sat=1` samples, all at `active=25`. Pending 7→6→3: step drains 1 and 3, total 4.
Done 0→1→4 with staging flat at 0: step transitions 1 and 3, total 4. Every counted step is bracketed
by two at-threshold samples. Row 47 drops to 24 active and correctly ends the window. This is real
turnover under full occupancy, and the pre-fix evaluator never looked at it.

Corrected: verdict 2 **SATISFIED** (pre-fix: NOT SATISFIED), witness **window 1**, overall
**SATISFIED** (pre-fix: NOT SATISFIED).

### 5.4 What this does and does not mean — Beta §20/§21

**FORENSIC ONLY. NON-CERTIFYING.** Attempt 2 aborted at the stage 3→4 transition on
`worker_admission_timeout`. Stage 4 never ran, nothing was published, no coverage advanced. **This
does not retroactively convert attempt 2 into a Gate-12 pass**, and no Gate-12 result is claimed
here. The finding is a fact about the instrumentation: the evaluator that produced the banked
sub-verdict was asking the wrong question, and the answer it gave was wrong for this run.

**Correction to the brief.** §19-B4 records attempt 2's shape as "several qualifying static windows…
if §20 reanalysis confirms it". **The reanalysis does not confirm it.** Attempt 2's actual shape is
**B1** — a static longest window (19 samples) concealing a shorter window with genuine turnover.
Reported, not resolved silently, per the brief's §6. It changes nothing about the fix (DB4 remains a
valid and necessary control fixture; it is simply synthetic rather than a replay of attempt 2) but it
does change what the banked attempt-2 saturation sub-result means, and that is Beta's call, not mine.

Nothing was re-banked, and the attempt-2 verdict/evidence files were not modified.

---

## 6. FILES CHANGED — from `git status`

```
 M scripts/gate12_concurrency_sampler.py
 M tests/test_gate12_concurrency_sampler.py

 scripts/gate12_concurrency_sampler.py    | 251 +++++++++++++++++++++++--------
 tests/test_gate12_concurrency_sampler.py | 232 ++++++++++++++++++++++++++++
 2 files changed, 418 insertions(+), 65 deletions(-)
```

Two files, as scoped. This report adds `docs/CLAUDE_CODE_REPORT_DEFECT_B_TURNOVER_AGGREGATION.md`.
The other untracked entries (`docs/CLAUDE_CODE_INSTRUCTIONS_DEFECT_A_…`,
`docs/CLAUDE_CODE_INSTRUCTIONS_DEFECT_B_…`, `docs/TB_SUBMISSION_PRERUN_R3_…`, the two
`miner_ledger.db-shm/-wal` files, `optimal_window_config.json.stale_1786149572`) were present at
session start and were not created or touched by this work.

Reanalysis and differential harnesses were written to the session scratchpad, not the repo:
`b1_redfirst.py`, `crit1_differential.py`, `s20_reanalysis.py`. Say the word if any should be
committed as a durable artifact.

**HEAD is still `4c76f42`. No commit, no push, no launch. Port 5700 was never bound.**

---

## 7. CONFLICTS RETURNED, NOT WORKED AROUND

1. **§20 vs §19-B4 (material).** The brief expects attempt 2 to be an all-static B4 run. It is a B1
   run: the defect produced an actual false negative in attempt 2, not merely a latent risk. §5.4.
2. **Gate naming (procedural).** Beta's B1–B5 collide with the suite's certified ESTAB arms B1–B7.
   Named DB1–DB5 with the mapping documented in the suite docstring; certified names untouched. §3.0.

No certified surface required modification. Criterion 1, the atomic snapshot, UNOBSERVED semantics,
the S4 identity column, `overall_satisfied()`, and the exit-code contract are all unchanged.
