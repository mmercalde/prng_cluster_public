# CLAUDE CODE INSTRUCTIONS — S145 SEED-DOMAIN / CURSOR AMENDMENT, REVISION 2 (TEST-ONLY)

**Host:** VM101, repo `~/distributed_prng_analysis`. The amendment + R1 are **uncommitted in the
working tree** at base `4dd5535`. `source ~/venvs/torch/bin/activate` before every test.

**Authority:** Team Beta ruling *"S145 SEED-DOMAIN / CURSOR AMENDMENT R1"* (2026-08-09).
**R1 PRODUCTION CHANGES ACCEPTED — one test-only verification correction required before commit.**

## THIS IS TEST-ONLY. DO NOT MODIFY PRODUCTION.

Beta: *"Do **not** modify production for this finding. Correct only the acceptance harness."*
**Production changes expected: NONE.** If you believe a production change is required, **STOP and
report** rather than making it.

**CLOSED — do not reopen:** Blocker A (one certification door) · Blocker B (canonical
`prng_base` + skip-mode containment) · Blocker C (config-mode whole-plan wall, coercion removal,
the 10M WATCHER default correction) · the terminus · legacy deauthorization · first-gap cursor ·
`COMPLETE`/`None` · publication producer wiring · raise-on-coverage-write-failure · WATCHER
fail-closed cursor lookup · `dataset_sha256` as provenance only · live-test DB isolation.

**Hard constraints:** no commit, no push, **no pipeline launch, no fleet launch, no port 5700
bind.** Gate 12 HELD. Do not touch staging-capacity files or suites.

**Base verification:** amendment + R1 intact; `tests/test_seed_domain_cursor_amendment.py` →
**39/39**. Untracked runtime residue expected, not a stop condition.

---

## THE DEFECT — a false green in `G-NO-REPLACE-CLOBBER`

R1 changed the certified table schema from `prng_type` / `mapping_mode` to `prng_base` /
`skip_modes_executed`. But `arm_append_only_is_enforced_by_the_database()` still builds its
`INSERT OR REPLACE` with the **old** columns:

```python
cols = ("coverage_id,run_id,study_identity,"
        "prng_type,mapping_mode,"          # ← columns that no longer exist
        ...)
```

…and then catches **any** `sqlite3.Error` as success. So the observed path is:

```
INSERT OR REPLACE → "column prng_type does not exist" → sqlite3.Error → caught → PASS
```

instead of the path it claims to prove:

```
INSERT OR REPLACE → same coverage_id collision → implicit DELETE
                  → BEFORE DELETE trigger → ABORT
```

**Those are completely different observations.** This matters because the production ledger
explicitly states that `PRAGMA recursive_triggers = ON` is load-bearing *specifically so an
`INSERT OR REPLACE` conflict cannot evade the delete trigger* — and that claim is **currently
uncertified**. Beta classifies this as a **verification-integrity blocker, not a demonstrated
production defect**: the bare INSERT, BEFORE UPDATE rejection, BEFORE DELETE rejection and
recursive triggers all remain correct as implemented.

---

## REQUIRED R2 CHANGES

### A. Build the REPLACE statement against the CURRENT schema

Use the live columns, with an otherwise **valid** row:

```
coverage_id · run_id · study_identity · prng_base · skip_modes_executed ·
seed_domain_contract · seed_start · seed_end_exclusive · dataset_sha256 ·
repository_commit · artifact_sha256 · generation_id · publication_status · recorded_at
```

Use the **existing `coverage_id`** so a **genuine uniqueness conflict** occurs. Every other value
must be valid, so the only thing that can stop the statement is the append-only enforcement itself.

### B. Assert the REASON, not merely "some SQLite error"

For **UPDATE, DELETE and REPLACE**, require the append-only trigger classification/message.

```python
except sqlite3.Error:      # NOT sufficient evidence
    pass
```

is explicitly rejected. The REPLACE arm in particular must prove the error came from
`certified_coverage is append-only` (or the specific delete-trigger diagnostic). Beta's reason:
*"That prevents another bad-column, malformed-value, CHECK-constraint, or syntax error from
masquerading as append-only enforcement."*

### C. Add the load-bearing `recursive_triggers` mutant

The production code claims `recursive_triggers=ON` is **necessary** for REPLACE protection. Prove
it. Same valid table, same existing `coverage_id`, same valid REPLACE:

```
recursive_triggers = OFF → REPLACE SUCCEEDS / the old record is replaced
recursive_triggers = ON  → REPLACE ABORTS through the no-delete trigger
```

Beta: *"the strongest proof of the exact invariant, and directly tests the rationale embedded in
production."* This is the arm that turns a comment into a certified claim.

### D. Correct the publication-witness gate WORDING (no behaviour change)

`_require_publication_witness()` permits a non-`RunArtifactResult` object **if it satisfies the
complete frozen dataclass contract** — which is consistent with Beta's own wording (*"reject an
object that is not the canonical result type **or** cannot satisfy the complete frozen result
contract"*), so **the implementation is correct and stays as-is**.

But the gate title *"only a canonical `RunArtifactResult` may certify"* is **stronger than what is
enforced**. The real guarantee is **the production call path plus complete frozen-contract
validation**, not an unforgeable Python object. Rename to something accurate, e.g.
*"canonical publication witness / complete frozen contract required."*

---

## VERIFICATION

- `tests/test_seed_domain_cursor_amendment.py` re-run **after the last test edit** — 39/39 (or the
  revised count, all green) **×3**;
- **REPLACE mutant/control evidence**: the ON/OFF pair from §C, with the OFF arm proven to actually
  replace the row;
- the corrected REPLACE arm proven to fail **for the append-only reason**, and proven **red** if the
  reason assertion is removed (i.e. it must no longer pass on a bad-column error);
- **staging regression suites do NOT need re-running** — Beta: *"the already accepted staging
  regression results do not need to be rerun solely for this correction unless production files
  change."* If any production file changes, stop and report instead.

## REPORT

`docs/CLAUDE_CODE_REPORT_SEED_DOMAIN_CURSOR_R2.md`:

1. The corrected REPLACE statement, and proof the conflict is a genuine `coverage_id` collision
   rather than a schema error.
2. The reason-assertion for each of UPDATE / DELETE / REPLACE, quoted.
3. The `recursive_triggers` ON/OFF evidence, both directions.
4. Confirmation that **no production file changed** — `git diff --stat` limited to the suite.
5. The renamed witness gate.
6. Any disagreement with this brief **reported, not worked around.**

---

## ONE ITEM FOR THE RECORD — NOT A TASK

Beta recorded a **certification boundary** on cursor-zero, which Alpha will carry into the gate-12
run shape and which requires **no code change**: WATCHER only auto-overwrites `seed_start` when
`next_seed_start > 0`, so an explicitly supplied nonzero operator start can remain in force while
the certified cursor still reports the first uncovered seed as zero. Beta declined to make this a
blocker (the ledger still does not falsely certify a skipped range; the first-gap cursor stays
zero on the next query; intentional non-contiguous sweeps have historically been operator-
controlled). **Do not describe R1 as proving WATCHER forcibly rewrites every requested run to the
first gap.** For gate 12, the certified first-gap value is to be supplied **explicitly**, and under
the empty certified `{constant, variable}` namespace that value is **0**.
