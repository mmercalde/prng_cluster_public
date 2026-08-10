# CLAUDE CODE REPORT — PRE-RERUN ITEMS, REVISION 2 (CLOSE EVERYTHING)

**Host:** VM101 (`192.168.3.177`), repo `~/distributed_prng_analysis`, HEAD **`c4e0037`**, venv
`~/venvs/torch`. **Authority:** Team Beta ruling *"PRE-RERUN R1 REVIEW"* (2026-08-10), as
transcribed in `docs/CLAUDE_CODE_INSTRUCTIONS_PRERUN_R2.md`.

**All four items are closed. Nothing is deferred.** No commit, no push, no launch, no fleet, no
port 5700 bind; Gate 12 HELD; coordinator, miner, ledger, seed-domain/coverage surface and every
certified suite untouched.

## BASE VERIFICATION (before any edit)

```
git log --oneline -1                     c4e0037
tests/test_preflight_gpu_probe.py        12/12 checks green
tests/test_gate12_concurrency_sampler.py 29/29 checks green
```

## FINAL STATE (after the last change — the evidence below describes this run)

```
tests/test_preflight_gpu_probe.py        12/12 checks green   (M1A assertion strengthened)
tests/test_gate12_concurrency_sampler.py 38/38 checks green   (29 retained + 9 new)
R1 mutation harness                      ALL MUTANTS BEHAVED AS REQUIRED  (no regression)
R2 mutation harness                      ALL MUTANTS BEHAVED AS REQUIRED
port 5700                                UNBOUND
/home/michael/miner_staging/miner_ledger.db  mtime 2026-08-09 12:47:17.388348534  (unchanged)
./miner_ledger.db                            mtime 2026-08-04 18:09:56.049862973  (unchanged)
repo HEAD c4e0037 · reflog head unchanged · 0 stashes · 18 refs (unchanged)
```

---

## 1. THE UNOBSERVED-READ STATUS, ITS EXCLUSION, AND THE GAP RULE

**Beta's finding is confirmed in source and the correction is accepted without qualification.** I
applied VIR-5 to ESTAB — a context field — and left the same defect standing on the ledger read,
which is the term the criterion is actually made of. That was my error, not an oversight of
emphasis: the handler's own comment said "keep the sample out of the verdict" while the next line
appended it.

**The status.** A new vocabulary parallel to the ESTAB one:

```python
OBS_OBSERVED   = "OBSERVED"
OBS_UNOBSERVED = "UNOBSERVED"
LEDGER_FIELDS  = ("compute_active", "queued_pending", "claimed_rows",
                  "staging", "done", "cancelled", "failed")
```

`unobserved_row()` sets **every** field in `LEDGER_FIELDS` to `None` — enumerated in one place so it
cannot drift out of step with `sample_run` and leave a stale zero behind.

**The ordering is the fix, not the handler.** The row is now **born UNOBSERVED** and only becomes an
observation if a read actually succeeds:

```python
row = unobserved_row(now, ts_iso, "not_yet_read")   # born unobserved
try:
    with connect_ro(ledger) as conn:
        ...
        row.update(sample_run(conn, run_id))        # the ONLY path to OBSERVED
except Exception as e:
    row = unobserved_row(now, ts_iso, f"{type(e).__name__}:{e}")
```

The pre-R2 shape pre-seeded zeros and let a failure fall through. Born-unobserved makes the
fall-through **structurally impossible** rather than correctly handled — there is no zero to fall
through to. Per §1.4 the `except` is now broad (`Exception`, not `sqlite3.Error`) and covers
`connect_ro`, `discover_run_id` and `sample_run` alike; an escaping exception would previously have
killed the loop **and the summary with it**.

**One case deliberately stays a genuine zero:** the ledger was read successfully and holds no run of
ours yet (pre-latch). That is a known state, not a failed read, and is marked OBSERVED.

**Exclusion from the verdict** — four places, all in `evaluate()`:

| | |
|---|---|
| criterion | an unobserved sample carries `satisfies=None`; window building tests `is True`, not truthiness |
| peak occupancy | computed over observed samples only |
| worker union | accumulated over observed samples only |
| denominator | `samples_total` / `samples_observed` / `samples_unobserved` all reported; satisfying is stated as `N/M observed` |

**Rendering:** `render_ledger_value()` emits the literal `UNOBSERVED` for every ledger column, and
`satisfies` renders `-` rather than `0` — a `0` there would claim the criterion was evaluated and
failed, which is the same lie one column to the left. Two new TSV columns, `obs_status` and
`obs_reason`, carry the status and the exception text.

### The gap rule I implemented: **BREAK the window**, and why

Beta allowed either breaking or annotating. **This implementation breaks, and also annotates.**

A window is a claim of **sustained** simultaneity, and "sustained" is exactly the property an
unknown interior instant destroys — across a gap the fleet may have emptied and refilled, and
nothing in the evidence file can distinguish that from continuity. Breaking makes the claim true
**by construction** rather than true by assumption, and it **fails closed**: the worst a gap can do
is understate the verdict, never inflate it. Annotation alone would leave the reader to decide
whether to believe a window the tool itself could not vouch for — which is the same "unobservable
rendered as clean" move in a politer form (VIR-5). The gap count and reasons are reported either
way, so choosing to break hides nothing.

*This is the owner rule on taking the structurally stronger mechanism, applied to a choice Beta left
open.*

### A second consequence of the same fall-through, not named in the brief

`runnable = pending + claimed + staging` summed an unobserved sample to `0 + 0 + 0`, so **a spell of
failed ledger reads started the quiescence timer and could stop the sampler with "run is over"
while the run was alive.** The fall-through did not merely corrupt the verdict — it could end the
observation. Quiescence is now decided only on observed samples: a gap neither starts nor clears the
timer. Gated as **D4**, which drives `main()` with every ledger read failing and a 0.1 s quiescence
against a 0.5 s run, and asserts the sampler ran to `max-seconds` instead of declaring the run over.

## 2. THE TURNOVER WINDOW DEFINITION AND THE FOUR CLOSURES

**Definition.** The turnover window **is the qualifying simultaneity window** — the single longest
run of consecutive satisfying observed samples, the same interval verdict 1 is decided on. **Not the
whole run.** By construction it contains no unobserved samples.

| Beta §2 | closure |
|---|---|
| **1.** define the window, in the summary output | The summary prints `measured over THE QUALIFYING SIMULTANEITY WINDOW — the same interval criterion 1 was decided on, NOT the whole run`, plus the window's sample count and timestamps. Gated by **F1**. |
| **2.** report the exact predicate for each verdict, in the summary | Both predicates are printed as formulas — see the verbatim sample in §4. Gated by **F1** on the literal predicate strings. |
| **3.** the three numbers over the SAME window | `pending_drained`, `transitions`, `done_delta` are all computed inside `_turnover(window)`, which only ever receives the qualifying window. Gated by **E1**: a fixture that drains 20→10 *before* the fleet fills reports `turnover_pending_first=10` (window start, not run start) and `drained=0`. Mutant: turnover recomputed run-wide → E1 and E2 red. |
| **4.** consumption paired with sustained occupancy across the same samples | Both terms are summed **step-wise over consecutive pairs inside the window**, so every counted step is bracketed by two samples that are themselves at or above the threshold. A run-wide monotonic decrease cannot qualify. Gated by **E2**: the queue drains 7→1 precisely while occupancy dips to 3, which splits the run into two qualifying windows, neither containing the drain → NOT satisfied. |

**A strengthening beyond the four, with its own gate.** Summing step-wise rather than reading the
endpoints also credits consumption that a stage boundary later masks: pending 7→3 (real consumption
under full occupancy) then →10 as the next stage's stripes are created gives an **endpoint delta of
−3** but a **step-wise drain of 4**. **E3** is that case; its mutant is an endpoint-only `_turnover`.
Both numbers are printed, the endpoint delta labelled "context only".

**Not collapsed.** Exit codes `0 / 2 / 3` stay exactly as implemented, now produced by a single
`exit_code()` used by both `main()` and the summary. **F2** asserts the returned code matches the
legend the summary prints; **C5** (retained) independently restates the mapping rather than importing
it, so a bug in `exit_code()` cannot hide behind its own definition.

## 3. THE M1A ASSERTION AND ITS POST-COMMIT DEMONSTRATION

Beta authorized the residual assertion; it is added. The `found` expression now strips comment lines
before testing:

```python
found = LEGACY_SHELL in "\n".join(
    l for l in committed.stdout.decode(errors="replace").splitlines()
    if not l.lstrip().startswith("#"))
```

**Scope, stated exactly:** one assertion expression changed, its detail string now says
`EXECUTABLE code`, and a five-line rationale comment sits above it. **No other line of
`test_preflight_gpu_probe.py` changed**, and `preflight_check.py` was not opened.

**Post-commit demonstration**, throwaway `--depth 1` clone into the scratchpad, both files copied in
and committed **inside the clone** so its HEAD is the mutated source:

```
clone HEAD after throwaway commit : eaa3c05
(1) pinned anchor  c4e0037  ->  12/12 green
    M1A: `|| echo 0` construct located in EXECUTABLE code at c4e0037:preflight_check.py
(2) anchor repointed at the post-fix HEAD  ->  M1A FAILS  (10/11)
```

**(2) is the point.** In R1 the same experiment with the bare substring test returned **12/12 —
green on the repair's own commentary**. The assertion now reds there. The residual weakness I flagged
and declined to fix last round is closed, and the closure is demonstrated by the failure mode
changing, not merely by the pinned case still passing.

Clone deleted. **Repo HEAD `c4e0037`, reflog head unchanged, 0 stashes, 18 refs — unchanged.**

## 4. VERBATIM SUMMARY OUTPUT

From an end-to-end run of `main()` against a synthetic WAL ledger at gate-12 geometry (32 stripes,
25 workers), with a writer thread draining the backlog under full occupancy and **one injected
unobserved ledger read**. No fleet, no coordinator, no port bind.

Note what the gap did: it **split the run into two qualifying windows** (3 samples and 7 samples),
and turnover was measured over the second — visible in the file without explanation.

```
==========================================================================
GATE-12 CONCURRENCY VERDICT — Beta's saturation criteria
==========================================================================
run_id            : runE2E
ledger            : .../scratchpad/e2e2/miner_ledger.db
sampling          : 2026-08-10T08:01:55 -> 2026-08-10T08:01:57
sample interval   : 0.15s
occupancy threshold : 25 distinct compute-active workers
window minimum    : 2 consecutive satisfying samples

-- sample census (the verdict's denominator) --
samples emitted   : 11
  OBSERVED        : 10   (ledger read succeeded)
  UNOBSERVED      : 1   (ledger read FAILED — not evidence of an idle fleet)  reasons: OperationalError
  An UNOBSERVED sample is excluded from both criteria and BREAKS any
  window it falls inside: a window is a claim of SUSTAINED occupancy,
  and an unknown interior instant is exactly what makes 'sustained'
  unprovable. Gaps can only understate this verdict, never inflate it.

EXACT PREDICATES — this file is self-describing; nothing below depends
on a companion document.

CRITERION 1 (sustained simultaneity), per sample:
      satisfies  <=>  compute_active >= 25  AND  queued_pending >= 1
    VERDICT 1 satisfied  <=>  there exists a run of >= 2 CONSECUTIVE
      satisfying OBSERVED samples (that run is 'the qualifying window').
      compute_active = COUNT(DISTINCT claimed_by) WHERE state='claimed';
      staging is deliberately excluded — StripeComplete has already
      released that worker's compute slot, so counting it overstates.

CRITERION 2 (turnover under full occupancy), measured over THE
  QUALIFYING SIMULTANEITY WINDOW — the same interval criterion 1 was
  decided on, NOT the whole run:
      pending_drained = SUM over consecutive pairs (a,b) in the window
                        of max(0, a.pending - b.pending)
      transitions     = SUM over the same pairs
                        of max(0, (b.done+b.staging) - (a.done+a.staging))
    VERDICT 2 satisfied  <=>  pending_drained > 0  OR  transitions > 0
      Every counted step is bracketed by two samples that are themselves
      at or above the threshold, so consumption is paired with sustained
      occupancy across the SAME samples. A run-wide monotonic decrease
      does not qualify on its own.

THE TWO ARE SEPARATE AND ARE NOT COLLAPSED. Criterion 1 proves the queue
was non-empty; only criterion 2 proves it was consumed.

VERDICT 1 — SUSTAINED SIMULTANEITY        : SATISFIED
VERDICT 2 — TURNOVER UNDER FULL OCCUPANCY : SATISFIED
EXIT CODE                                 : 0
  0 = both criteria satisfied · 2 = criterion 1 (simultaneity) NOT satisfied · 3 = criterion 1 satisfied, criterion 2 (turnover) NOT satisfied

-- verdict 1 evidence: simultaneity --
peak simultaneous compute-active workers : 25
  observed at                            : 2026-08-10T08:01:55
  queued (pending) at that same instant  : 7
satisfying samples                       : 10/10 observed
runs of satisfying samples               : 2
  of those, QUALIFYING (>= 2 samples)         : 2
    window 1: 3 samples, 0.3s, 2026-08-10T08:01:55 -> 2026-08-10T08:01:55, min_active=25 min_queued=7
    window 2: 7 samples, 0.9s, 2026-08-10T08:01:56 -> 2026-08-10T08:01:57, min_active=25 min_queued=2
longest window                           : 7 samples, 0.9s
  from / to                              : 2026-08-10T08:01:56 -> 2026-08-10T08:01:57
  min compute-active within window       : 25
  min queued within window               : 2

-- verdict 2 evidence: turnover WITHIN the qualifying window --
turnover window                          : 7 samples, 2026-08-10T08:01:56 -> 2026-08-10T08:01:57
  (this is the qualifying simultaneity window, not the whole run)
  min compute-active across it           : 25
pending at window start / end            : 5 -> 2
  pending DRAINED step-wise in window    : 3
  endpoint delta (context only)          : 3
stripes transitioned into done/staging   : 3
  of which reached done                  : 3
finding                                  : queued work was consumed under full occupancy

-- NOT evidence of saturation (recorded so it is not mistaken for it) --
distinct workers ever seen active (union across instants) : 25
  A union over time is not simultaneity: 25 workers running strictly one
  after another produce the same number as 25 running together. This
  figure CANNOT satisfy the criterion and is printed only for context.

-- ESTAB (context only — NOT a term in either criterion) --
established connections            : max=0 min=0 over 11 sample(s) where ss succeeded
samples where ss was UNAVAILABLE   : 0
  An unobservable ss is recorded UNAVAILABLE, never as 0 — a
  connection count nobody could take is not a count of zero.
  A connected worker is not an occupied worker. ESTAB is not an input to
  either verdict above and cannot change one.
==========================================================================
```

The corresponding TSV row for the gap, showing that no column offers a number to misread:

```
ts_iso               obs_status  obs_reason                           compute_active  queued_pending  ...  satisfies
2026-08-10T08:01:56  UNOBSERVED  OperationalError:database is locked  UNOBSERVED      UNOBSERVED      ...  -
```

## 5. RED-FIRST / MUTATION EVIDENCE, PER NEW ARM

| arm | mutant installed | result |
|---|---|---|
| D1-READ-FAILURE-IS-UNOBSERVED | pre-R2 fall-through: failed read → zeros, marked OBSERVED | **RED** |
| D2-GAP-BREAKS-THE-WINDOW | same | **RED** |
| D3-UNOBSERVED-NOT-A-ZERO-SAMPLE | same | **RED** |
| D4-GAP-DOES-NOT-END-THE-RUN | same | **RED** |
| E1-TURNOVER-WINDOW-IS-NOT-THE-RUN | turnover recomputed over the whole run | **RED** |
| E2-DIP-DRAIN-NOT-CREDITED | same | **RED** |
| C2-DRAINING-QUEUE-BOTH | same (control) | **green — correctly**, a genuine in-window drain is unaffected |
| E3-STEPWISE-DRAIN-COUNTED | `_turnover` uses the endpoint delta only | **RED** |
| F1-SUMMARY-IS-SELF-DESCRIBING | summary drops the exit-code legend | **RED** |
| F2-EXIT-CODE-MATCHES-LEGEND | `exit_code` collapses 3 into 2 | **RED** |

**Clean controls (VIR-2), so no negative above is vacuous:** D2 runs the identical six-sample fixture
*without* the gap and requires SATISFIED; C2/G6 remain the both-criteria-met control; E3 is a
positive turnover case carried solely by the step-wise term; B5 (retained) keeps a genuine zero a
zero.

**D1 and D4 are end-to-end through the real `main()`** — real loop, real TSV writer, real
`evaluate`, real summary — with `connect_ro` made to raise on chosen call indices. D1 asserts on the
emitted TSV and summary files, not on internal state.

**No regression in the R1 arms.** The R1 mutation harness was re-run in full against the R2 tree:
`ALL MUTANTS BEHAVED AS REQUIRED`. All 29 previously-certified arms remain green.

## 6. CONFIRMATION ON THE CERTIFIED ITEMS

| certified item | status |
|---|---|
| GPU probe repair in `preflight_check.py` | **byte-unchanged.** `sha256 cfbde94c71b66d07a613b4ef49dbc38088efdb4005d28899e5846c2f2c346730`, identical to the R1 report; mtime `2026-08-09 19:57:33`, predating this session. Not opened. |
| `read_snapshot` atomicity fix | **byte-unchanged.** Verified by `inspect.getsource` hash and by asserting `sample_run` still opens `with read_snapshot(conn):` with both reads inside it. |
| A1 / A2 / A3 arms | **unchanged and green.** |
| ESTAB observability work | semantics unchanged; `estab_observation`, `render_estab`, `summarize_estab` untouched. **One display-string change — disclosed below.** |
| B7 invariance arm | **unchanged and green.** |
| `_PRE_FIX_REV` anchor | unchanged; only the `found` expression it feeds was strengthened, as authorized by §3. |

### ⚠ TWO DISCLOSURES, NOT WORKED AROUND

**(a) `sample_run`'s return dict gained two keys.** It now also returns `obs_status: OBSERVED` and
`obs_reason: None`. The certified *mechanism* — the `read_snapshot` wrapper and the two queries
inside it — is byte-unchanged, and A1/A2/A3 still pass. But the function itself is not byte-identical,
and since its atomicity fix is certified I am naming that rather than letting "byte-unchanged" cover
it. Marking the successful path OBSERVED at its source is what makes born-unobserved work; the
alternative was inferring success in the caller, which is the weaker construction.

**(b) One display string inside the certified ESTAB block changed.** It read
`over N observed sample(s)`. R2 gives OBSERVED/UNOBSERVED a specific ledger meaning **elsewhere in
the same file**, so that word became ambiguous to exactly the reader item §4 exists for. It now reads
`over N sample(s) where ss succeeded`. **Display only** — no semantics, no verdict, no gate behaviour;
B7 unaffected and green. Item §4 is mandatory and §1's vocabulary is what created the collision, so
leaving it would have traded a certified word against a mandatory requirement. Flagged for Beta to
reverse if that reading is wrong.

## 7. FILES CHANGED

| file | status | change |
|---|---|---|
| `scripts/gate12_concurrency_sampler.py` | untracked, modified | `OBS_*` vocabulary + `LEDGER_FIELDS` + `unobserved_row` + `is_observed`; born-unobserved loop ordering with a broad `except`; quiescence gated on observed samples; `render_ledger_value`; two new TSV columns; `evaluate` excludes gaps and breaks windows at them, reports the census and per-window detail; `_turnover` step-wise `pending_drained` + window identity fields; `exit_code()` + `EXIT_CODE_LEGEND`; self-describing summary with both predicates; one ESTAB display string |
| `tests/test_gate12_concurrency_sampler.py` | untracked, modified | 29 arms retained unchanged; **9 new** — D1–D4, E1–E3, F1–F2; helpers `gap`, `failing_connect`, `run_main`, `_static_saturated_ledger` |
| `tests/test_preflight_gpu_probe.py` | untracked, modified | **the authorized M1A assertion only** — comment-stripping in the `found` expression, its detail string, and a five-line rationale comment |
| `docs/CLAUDE_CODE_REPORT_PRERUN_R2.md` | new | this report |

**Not changed:** `preflight_check.py` · `gate12_launch.sh` · every coordinator, miner, ledger,
seed-domain and certified-suite file.

```
sha256  e6467641…  scripts/gate12_concurrency_sampler.py
sha256  54c81264…  tests/test_gate12_concurrency_sampler.py
sha256  62849d9f…  tests/test_preflight_gpu_probe.py
sha256  cfbde94c…  preflight_check.py            (unchanged from R1)
```

### `gate12_launch.sh` — DELIVERED UNCHANGED; the interface did not change

The **TSV schema** changed (two new columns), but the launcher never parses the TSV — it names the
file, passes the path, and echoes it (`gate12_launch.sh:44, 85, 94, 131`). The **CLI** is unchanged:
verified behaviourally, the launcher's exact flag set (`--out --summary --interval 2 --threshold 25
--min-window-samples 2 --port 5700 --max-seconds 7200`) was accepted and the process died at the
**ledger guard**, not at argparse. No new required argument. Exit codes `0/2/3` are unchanged from
R1 and the launcher does not inspect them. **Nothing was run from it.**

### Standing-item note

No new untracked `.py` file was created, so the phase-4 Gate 22 untracked-`.py` sensitivity is
unchanged by this work.

## VERIFICATION-INTEGRITY CONTROLS (VIR-1…6)

- **execution proof:** every figure is from a run in this session on VM101 under `~/venvs/torch`;
  both suites print a terminal `N/N checks green`, both mutation harnesses print a terminal verdict.
- **clean control:** D2 without the gap · C2/G6 both-criteria-met · E3 positive turnover · B5 genuine
  zero · G1B/G6 retained.
- **fault-injection control:** the ten-row mutant table in §5; every new arm reds under its defect,
  and the R1 table was re-run unchanged.
- **completion sentinel:** `12/12`, `38/38`, `ALL MUTANTS BEHAVED AS REQUIRED` ×2.
- **unavailable-observer behaviour:** ledger read → OBSERVED/UNOBSERVED, excluded from the verdict
  and breaking windows; ESTAB → OK/UNAVAILABLE; neither is ever rendered as a number it did not
  measure.
- **audit claim scope:** the sampler, its suite, and the single authorized M1A assertion. **No claim
  is made about the certified probe logic beyond "not touched"**; it was deliberately not re-verified.
- **searched surfaces:** live working tree on VM101; live source of the pre-R2 failure path before
  briefing it; `git show`/`status`/`log`/`reflog`/`for-each-ref`; a throwaway post-commit clone;
  `docs/CLAUDE_CODE_INSTRUCTIONS_PRERUN_R2.md` and the R1 instruction/report pair; live `ss`,
  `sqlite3` and WAL behaviour on the box.
- **unavailable surfaces:** no committed baseline exists for the sampler (still untracked), so the
  pre-R2 mutant is transcribed from the working tree rather than anchored to a hash — the same
  disclosure as R1, and it resolves the moment the file is committed. The fleet, coordinator and
  ledger were **deliberately** not exercised.
- **governance trail searched:** `CLAUDE_CODE_INSTRUCTIONS_PRERUN_R1/R2`,
  `CLAUDE_CODE_REPORT_PRERUN_PROBE_AND_SAMPLER`, `CLAUDE_CODE_REPORT_PRERUN_R1`,
  `TB_SUBMISSION_PRERUN_ITEMS_AND_RERUN_REQUEST`, `TB_NOTE_R1_INFLIGHT_AND_ACCESS_PATTERN`.
- **chapters searched:** none — this item touches no pipeline stage.
