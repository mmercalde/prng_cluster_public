# TEAM_ALPHA_REVIEW_S172_PHASE5_D3.md

**Subject:** Team Alpha code-level review of the D3 implementation (shared
backend-neutral 24→22 columnizer + independent structural validator)
**Spec:** `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3.md` REV3
**Base:** HEAD `66f0425`
**Artifacts:** `utils/canonical_arrays.py` (470 lines),
`tests/test_s172_phase5_d3_columnizer.py` (780 lines), diff (gate-22
registration, 9 lines), status, numpy version capture.
**Verdict: APPROVED — production module correct and unchanged (`md5 e3033e1e…`);
the MC2 harness gap was corrected in a test-only round and re-verified with
correct attribution (§5). All Team Beta §3 commit conditions are satisfied. The
residual LB/FWD gap family was closed in a second test-only round and
independently re-verified (§5). All Team Beta final-ruling conditions are
satisfied; commit is authorized.**

## 1. Scope — clean, and independently verified

`git status` shows exactly the two new files plus the gate-22 registration; the
diff is 9 added lines, entirely inside gate 22's `allowed` whitelist comment
block. Team Alpha independently confirmed **no production call site was
rewired**: `grep -rn "canonical_arrays|records_to_arrays|validate_array_bundle"
--include=*.py` returns **zero** hits outside the new module, its own harness,
and the whitelist entry. The legacy paths remain intact and in use — the inline
`_survivors_to_arrays` closure still exists and is still called at
`window_optimizer_integration_final.py:1786`, and the
`convert_survivors_to_binary` array block is untouched, exactly as §0/§6
require.

## 2. The four REV3 amendments — all correctly implemented

- **[A1] base-family restriction.** `BASE_PRNG_FAMILIES` is **derived** from
  `KERNEL_REGISTRY` by stripping `_hybrid_reverse` / `_hybrid` / `_reverse`
  rather than hardcoded — 11 base families, and a registry change cannot leave
  a stale literal behind. The semantically-invalid-but-equality-consistent case
  Beta named (`prng_base = "java_lcg_reverse"` with constant mode) is rejected.
- **[A2] float32 representability.** `_check_float` implements all five checks
  in the frozen order, with check 4 as `np.isfinite(np.float32(value))` and an
  inline comment explaining that Python-level finiteness does not prove
  `float32` representability. The six count arrays additionally require
  nonnegative **and** `value == math.floor(value)`, with the rationale (they are
  `float32` only because the schema demands it) recorded in the raise message.
- **[A3] `Iterable`.** Signature and docstring both say `Iterable`; the body
  uses a single `for index, record in enumerate(records)` with no `len()`, no
  indexing, and no re-traversal — a generator is a first-class input, and the
  harness exercises one.
- **[A4] bound wording.** `_UNIT_INTERVAL_ARRAYS` / `_COUNT_ARRAYS` /
  `_NONNEGATIVE_ARRAYS` partition the 13 float fields; the nonnegative branch's
  raise message explicitly records that no generic `<= 1` ceiling applies and
  that `bidirectional_selectivity` may legitimately exceed 1.

Also verified: identity consistency (`constant → prng_type == prng_base`,
`variable → prng_type == prng_base + "_hybrid"`); strict exact-24-key set with
both missing and extra keys failing; `sessions` and `prng_base` validated
despite not becoming arrays; encoder `ValueError` propagating unwrapped;
`CANONICAL_ARRAY_CONTRACT` dtypes normalized through `np.dtype`; the validator
enforcing 22 keys / names / order / dtype / equal lengths / 1-D; and
`records_to_arrays` self-validating before returning.

**Order preservation is structural, not asserted** — the single `enumerate`
traversal appends into per-array lists in arrival order and materializes once.
There is no `sorted()`, no `len()`, no indexing anywhere in the module. That is
the strongest available form of the [C1] guarantee.

**Int range checks are numpy-version-independent** (§4 below): `_INT_FIELD_RANGE`
is derived from `np.iinfo` and enforced in Python space before any
materialization.

## 3. Mechanical verification (Team Alpha sandbox, pristine `66f0425`)

- Full gate: **10/10 green** — independent reproduction. Claude Code
  additionally captured the pre-edit baseline green at `66f0425` (D3.0 10/10,
  D2 7/7, D1.1 18/18, D1.0 8/8, D0 12/12, Phase 4 63/63, Phase 3 17/17) and
  reported all suites green post-edit, with **21/21** of its own C10 mutants
  killed.
- Its mutation methodology is sound: each mutant is a textual edit `exec`'d
  into a fresh namespace so the on-disk file is never modified, and it
  explicitly scoped the M09-M17 missing-key relaxations to the *targeted* field
  after observing that a blanket relaxation killed them incidentally via a raw
  `KeyError` on `seed` — i.e. it noticed its own evidence was about to prove
  the wrong thing and corrected it.

**Team Alpha independent mutants (four, targeting seams its 21 did not):**

| mutant | result |
|---|---|
| MA — invert the constant/variable identity rule | **killed** (C1/C2/C3) |
| MB — drop `_hybrid` from the suffix list (partial base restriction) | **killed surgically by C5 alone**, 9/10 |
| MC — remove `score` from `_UNIT_INTERVAL_ARRAYS` | killed, but **for the wrong reason** — `score` then falls out of `_FLOAT_ARRAYS` entirely and trips the "unclassified array" assertion |
| MC2 — move `score` to `_NONNEGATIVE_ARRAYS` (stays classified, loses its `<= 1` ceiling) | **SURVIVED — 10/10 green** |

MB is the valuable positive result: a *partial* weakening of the [A1]
restriction is caught precisely by the gate that owns it.

## 4. numpy version split — recorded, and it does not affect the evidence

Three versions are in play: **VM101 venv `1.22.0`** (where the gate actually
runs), **VM101 system `2.2.6`** (what a non-venv `ssh python3` reports — the
capture in `d3_numpy.txt`), and **Team Alpha sandbox `2.4.4`**. This matters
because `np.array([-1], dtype=np.uint32)` **wraps silently** on 1.22.0 and
**raises** on 2.x, so a range-check mutant could be killed by numpy rather than
by the check under test.

**It does not compromise the evidence here:** `_check_int` validates in Python
space against `np.iinfo`-derived bounds *before* any array is constructed, so
the check is load-bearing and version-independent. Claude Code identified the
hazard itself and noted M14's red confirms the Python-space check is the only
thing between a bad producer and a corrupted column on 1.22.0. Recorded so a
future reader does not "simplify" the range check on the assumption that numpy
will raise.

## 5. GAP (found, corrected, re-verified) — plus a confirmed residual family

**Finding.** REV3 §4.5 freezes `score in [0.0, 1.0]`. The production code
enforces it correctly (Team Alpha verified directly: `score` of `1.5`,
`1.0000001` and `-0.1` are all rejected with the frozen-bound message, while
`0.8` is accepted). But no gate case exercised it, so MC2 — moving `score` from
`_UNIT_INTERVAL_ARRAYS` to `_NONNEGATIVE_ARRAYS`, keeping it classified while
removing its ceiling — survived at 10/10.

**Correction (test-only).** One row added,
`("score above 1.0", dict(score=1.5), "score")`, plus a tightening Team Alpha
did not request but endorses: the field-specificity assertion now matches the
**quoted** form (`repr(field)`, i.e. `'score'`) so a bare substring collision
cannot satisfy it, and Claude Code demonstrated that assertion bites by showing
a rejection that raises correctly but fails to name the field goes red.

**Team Alpha re-verification (pristine `66f0425`):** production module
byte-identical to the reviewed delivery (`md5 e3033e1e…`); gate **10/10**; MC2
re-injected → **killed by C5 with correct attribution** —
`C5 red -> AssertionError: score above 1.0: expected ValueError, but the call
SUCCEEDED (fail-closed lost)`, i.e. the failure is attributable to the score
upper-bound case and not to an incidental classification or unhandled-key
failure (Beta §3 condition satisfied).

**Residual gap family — found, corrected, re-verified.** Claude Code flagged,
without acting, that no case drove any unit-interval field below `0.0`. Team
Alpha confirmed and extended it: **LB** (weaken the shared branch
`0.0 <= v <= 1.0` → `v <= 1.0`) and **FWD** (reclassify `forward_matches` to
nonnegative-only) both survived at 10/10. Coverage was asymmetric — the table
had upper-bound cases for `reverse_match_rate` and `score` but none for
`forward_match_rate`, and no below-zero case for any unit-interval field.

Team Beta ruled both blocking and required exactly **two** rows (correcting
Team Alpha's "three": three per-field upper-bound cases plus one shared
lower-bound case, of which two were new):

```python
("forward_match_rate above 1.0", dict(forward_match_rate=1.5), "forward_match_rate"),
("score below 0.0",              dict(score=-0.1),             "score"),
```

**Team Alpha independent re-verification (pristine `66f0425`):** production
module byte-identical, `md5 e3033e1ee523a188a7b631f572157b24`; clean gate
**10/10**. Both mutants re-injected independently and attributed by
differential comparison — the LB run's C5 reds contain `score below 0.0` and
**not** `forward_match_rate above 1.0`; the FWD run's contain
`forward_match_rate above 1.0` and **not** `score below 0.0`. Each mutant
produces its own targeted attribution and only its own; neither is killed by an
unclassified-field assertion, a missing-key failure, an unrelated rate, or an
unnamed exception. MC2 remains killed and attributed. Closed.

**Observation (non-blocking):** on numpy 2.x the deliberate float32-overflow
case emits `RuntimeWarning: overflow encountered in cast` from the check-4 line.
The check functions correctly; the warning is cosmetic and does not appear on
VM101's venv numpy 1.22.0.

## 6. Team Beta §3 conditions — all satisfied

| condition | status |
|---|---|
| unmodified D3 gate 10/10 green | YES — reproduced in the Team Alpha sandbox |
| MC2 mutant red | YES — killed by C5 |
| MC2 failure attributable to the score upper-bound case | YES — `score above 1.0: expected ValueError, but the call SUCCEEDED` |
| all 21 original mutants remain killed | YES — C10 reports 21/21 |
| blocking non-regression green | YES — D3.0 10/10, D2 7/7, D1.1 18/18, D1.0 8/8, D0 12/12, Phase 4 63/63, Phase 3 17/17, Phase 0 8/8 |
| production module diff unchanged | YES — byte-identical, `md5 e3033e1e…` |

Commit scope per Beta §6: `utils/canonical_arrays.py`,
`tests/test_s172_phase5_d3_columnizer.py`, the gate-22 registration, the REV3
brief, this memo, and a session changelog. The §5 residual family is a separate
ruling and does not block.

— Team Alpha (Claude), 2026-07-25
