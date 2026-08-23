# BRIEF I — BOUNDED DEFECT REPAIR, PATCH REPORT TO TEAM BETA

**Base commit:** `48a87059f5200e00727556f05c1462df07ba4614` (unchanged; nothing committed)
**Authority:** `docs/TB_RULING_BRIEF_I_PRODUCTION_SHAPE_FAILURE.md` §3, §4
**Fixture migration authorized by:** Michael, 2026-08-22, under the field-6 precedent —
**submitted here for Beta's ratification.**

## CHANGE SET

```
utils/canonical_records.py                     +19 -1     PRODUCTION — 2 executable lines
tests/test_s172_window_anchor_brief_i.py      +125       new gate G-PHASE5-ASSEMBLY
tests/test_s172_phase5_d3_25_candidate_ingress.py  +6    FIXTURE ONLY
tests/test_s172_phase5_d1_engine.py            +7 -1     FIXTURE ONLY
docs/BACKLOG.md                                +42       DEP-ABI-V2-NPZ-SEMANTICS
docs/TB_RULING_BRIEF_I_PRODUCTION_SHAPE_FAILURE.md  (new, verbatim ruling)
```

## 1. PRODUCTION — and the repair was larger than first reported

**Alpha's first diagnosis named ONE site. There were TWO, in the same file.** Both feed
`build_mode_records`; both are inside Beta's authorized surface.

| site | what it is |
|---|---|
| `utils/canonical_records.py:217` | the miner assembly path — the site that failed in production |
| `utils/canonical_records.py:369` | **`normalize_trial_populations`, the PWC/ZMQ wrapper**, called from `window_optimizer_integration_final.py:1239`. It builds its OWN context and fed `"offset"` into it |

The second site was **found by the regression sweep, not by the original analysis** — it is why
`test_s172_phase5_d3_25_candidate_ingress` went 13/13 → 1/13 after the first patch. **Alpha
initially reported that red as stale test fixtures. That was wrong: it was a second unmigrated
production consumer**, the same defect class as the first, one call away.

*The lesson is the brief's own, twice over: a fix that migrates one consumer is not evidence
that every consumer migrated.*

Both now read `ctx["window_anchor"]`. **The caller's keyword stays `offset`. The emitted record
field stays `offset`.** Beta's formulation is in the code, not Alpha's:

> Canonical array 4 `offset` is a LEGACY WIRE NAME with exactly ONE post-F-4 meaning: it IS the
> window anchor. It is NEVER the generator phase, **at any phase value**. Generator phase remains
> independently represented in versioned generation metadata and never enters this array.

**Alpha's overruled rationale** — "coherent only because `generator_phase == 0`" — was removed
from every artifact. `docs/BACKLOG.md` records the correction explicitly rather than deleting it
quietly, and carries Beta's consequence: **ABI-v2 does not automatically resurrect F-4; it would
do so only if a consumer again reads array 4 as both anchor and phase.**

**Frozen contract untouched, verified:**

```
CANONICAL_RECORD_FIELDS   utils/canonical_records.py:117   unchanged
                          utils/canonical_arrays.py:145    unchanged
canonical array contract  utils/canonical_arrays.py:104    unchanged
                          utils/run_finalizer.py  index 4 == "offset", 22 arrays
"window_anchor" in any array-name list : NO      23rd array : NO
"generator_phase" in canonical_records.py : NO
```

## 2. `G-PHASE5-ASSEMBLY` — the new gate

Distinct from `G-PHASE5-SEAM`, which is **unchanged and unwidened** (asserted: the seam gate's
body is byte-identical to HEAD). The new gate publishes a real forward+reverse constant pair
through the real `AssemblingPhase5Sink`, commits, and reaches the true
`assemble_trial` → `build_mode_records` surface the suite never touched.

```
anchor=58 phase=0 -> canonical 'offset'=58 over 1 record(s);
mutants A(offset)/B(generator_phase) both DETECTED
```

**Fixture values are load-bearing and deliberately unequal**, per Beta: the assertion is `== 58`,
never `== 0`. It also asserts `"offset" not in ctx`, and that no 23rd field leaked into the record.

**Non-vacuity is two-directional**, by AST reconstruction of the live source with an
exactly-once-hit guard:

* **mutant A** — `ctx["offset"]` restored → the production `KeyError` returns;
* **mutant B** — `"offset"` sourced from `ctx["generator_phase"]` → records carry `0`, caught by
  the `== 58` assertion.

## 3. FIXTURE MIGRATION — two certified suites, ZERO assertion changes (PROVEN)

```
                                              assert lines HEAD   live   ADDED   REMOVED/CHANGED
test_s172_phase5_d3_25_candidate_ingress            114           114      0            0
test_s172_phase5_d1_engine                          186           186      0            0
```

* **`d3_25`** — `base` is splatted into the oracle's signature, so it was left byte-identical; the
  **trial context** is now a separate object (`ctx["window_anchor"] = ctx.pop("offset")`).
  *Alpha's first attempt added keys to `base` and broke the oracle — corrected.*
* **`d1_engine`** — `CTX` gains `window_anchor=2`, `generator_phase=0`. `offset=2` is **retained**
  because the dict doubles as the expected-record source at `:386`, where the record field
  legitimately keeps the name `offset`.

**Disclosed limitation:** because both fixtures keep anchor and legacy value equal, neither can
*discriminate* anchor from phase. **`G-PHASE5-ASSEMBLY` is the gate that carries that
discrimination (58 vs 0)**, which is exactly why Beta required unequal values there.

## 4. REGRESSION EVIDENCE

```
test_s172_window_anchor_brief_i            26/26 checks green            PASS
test_s172_window_anchor_brief_i_mutants    all mutants detected          PASS
test_s172_phase5_d3_25_candidate_ingress   13/13   (== baseline 13/13)   PASS
test_s172_phase5_d3_columnizer             10/10                         PASS
test_s172_phase5_d3_5_finalizer            60/60                         PASS
test_s172_d6_2_checkpoint_reconciliation   31/31                         PASS
test_chapter2_content_gate                 12/12                         PASS
test_s172_phase3_worker                    18/18                         PASS
```

### 4.1 `test_s172_phase5_d1_engine` — same pre-existing red population, proven by worktree

```
git worktree @ 48a8705 (PRE-repair, same environment)   18 FAIL   EXIT=1
live tree (POST-repair + fixture)                       18 FAIL   EXIT=1
```

**Beta's binding characterization (§5) — used verbatim, not paraphrased:**

> **same pre-existing red population / changed failure depth / no demonstrated new production
> regression**

Explicitly **not** "identical behavior" and **not** "zero differential": some failures moved from
the mandatory-context guard into the already-known RC-1 path, so the depth changed even though the
count did not. Its reds are pre-existing at `48a8705` — Brief I's *coordinator* mandatory-field
guard meeting stale fixtures, plus the RC-1 `expected_substripes: None` class. **No RC-1 repair is
authorized here.**

### 4.2 ⚠ A BASELINE FIGURE IN `logs/ac7_final/SUMMARY.tsv` IS A NESTED-TALLY LEAK

```
test_s172_phase4_coordinator    62/63 checks green
test_s172_phase5_d1_engine      62/63 checks green      <-- identical string
```

`d1_engine`'s recorded baseline was **never its own tally**; it captured the neighbouring suite's.
Measured directly at `48a8705`, d1_engine has 18 failures, not 1. **This is the same nested-tally
artifact Michael corrected Alpha on earlier in this programme, recurring in the final battery's
summary.** The affected row is corrected here; the rest of `ac7_final` is not re-derived and
should be treated with the same suspicion where a tally looks borrowed.

## 5. NOT TOUCHED, per the ruling

`OBSERVABILITY_GAP_1` · `B7` · `HARNESS-LEDGER-ORDER-1` · the 22-array contract ·
`WindowConfig.offset` (Brief II) · `G-PHASE5-SEAM` · **all retained artifacts** — 6151 staging
files, C2's 512, the archived ledger, ledger sha unchanged.

## 6. WHAT ALPHA ASKS

1. **Ratify the second production site** (`normalize_trial_populations`) as inside the authorized
   bounded scope — it is the same defect, in the named file, and the repair is incomplete without it.
2. **Ratify the fixture-only migration** of the two certified suites under the field-6 precedent,
   with the zero-assertion-change proof and the disclosed discrimination limitation.
3. **Note the nested-tally leak** in `ac7_final`, which affects how earlier battery figures should
   be read.

Per the ruling's sequence, the next steps after ratification are Michael's: commit the repair,
then a fresh Michael-authorized production-shape run that must reach successful Phase-5
publication before B5/B6/B8 reclassification, B7 acceptance classification, or Brief-I acceptance.

**Nothing committed. Nothing pushed. HEAD remains `48a8705`.**
