# CLAUDE CODE REPORT — S145 SEED-DOMAIN / CURSOR AMENDMENT, REVISION 2 (TEST-ONLY)

**Host:** VM101 (`zeus-ubuntu-vm`, 192.168.3.177), repo `~/distributed_prng_analysis`,
venv `~/venvs/torch`. Base `4dd5535`; amendment + R1 + R2 **uncommitted in the working tree**.
**Authority:** Team Beta ruling *"S145 SEED-DOMAIN / CURSOR AMENDMENT R1"* (2026-08-09),
per `docs/CLAUDE_CODE_INSTRUCTIONS_SEED_DOMAIN_CURSOR_R2.md`.
**Status:** verification correction complete. **40/40 gates green ×3** after the last edit.
**ZERO production files changed. Not committed, not pushed, nothing launched. Gate 12 not run.**

## 0. BASE VERIFICATION

| check | result |
|---|---|
| HEAD | `4dd5535` |
| amendment + R1 present | yes — 4 modified production files + 2 untracked `.py` |
| suite before any R2 edit | **39/39 green** |
| untracked runtime residue | WAL sidecars + `*.stale_*` — expected |

**Beta's finding is correct, and I should have caught it.** It was my own R1 schema change that
invalidated the arm's column list, and the broad `except sqlite3.Error` is what let the resulting
schema error stand in for append-only enforcement. Reproduced verbatim before touching anything:

```
current arm observes: OperationalError -> table certified_coverage has no column named prng_type
```

---

## 1. §A — THE CORRECTED REPLACE STATEMENT, AND PROOF THE CONFLICT IS GENUINE

The statement is no longer written out by hand. `_replace_statement()` reads the **live** column
list from `PRAGMA table_info(certified_coverage)` and binds values from a
`{column: value}` mapping, so a schema change cannot silently turn into a bad-column error again:

```sql
INSERT OR REPLACE INTO certified_coverage
  (coverage_id, run_id, study_identity, prng_base, skip_modes_executed,
   seed_domain_contract, seed_start, seed_end_exclusive, dataset_sha256,
   repository_commit, artifact_sha256, generation_id, publication_status,
   recorded_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
```

**Proof the conflict is a genuine `coverage_id` collision, not a schema error:**

* the bound `coverage_id` is read back off the stored record —
  `param0 = 79f4d02c4c14030f10e37ea3f6ab145d93b440d019893910ba4805566c8316ea`, and
  `param0 == kept.coverage_id` → **True**;
* every other value is **valid**: `prng_base='java_lcg'`, `skip_modes_executed='constant'`,
  `seed_domain_contract='v1.1-stratum'`, `[0, 1000)` (satisfying all three CHECK constraints),
  well-formed 64-hex digests, `publication_status='CERTIFIED'`, ISO-8601 `recorded_at`. So the
  only thing that can stop the statement is the append-only enforcement itself;
* the fixture is **reconciled against the live schema** before the statement is built, and reds
  with a named diagnosis if it drifts (see §4, M-COLS);
* the asserted failure explicitly excludes `no column named`, `no such column`, `has no column`
  and `syntax error`.

The R2 fixture is the **1,000-seed smoke row** colliding with the **billion-seed production row** —
the incident this gate exists for, now expressed as a real uniqueness conflict.

---

## 2. §B — THE REASON ASSERTION, QUOTED

`_assert_append_only_refusal()` requires three things of every refusal: the append-only
classification, the **specific trigger verb**, and the absence of any schema/syntax masquerade.
`except sqlite3.Error: pass` is gone.

Measured live, verbatim:

| statement | asserted verb | message |
|---|---|---|
| **DELETE** | `DELETE is forbidden` | `certified_coverage is append-only: DELETE is forbidden - this also blocks INSERT OR REPLACE, which satisfies a conflict by deleting` |
| **UPDATE** | `UPDATE is forbidden` | `certified_coverage is append-only: UPDATE is forbidden - a certified interval is immutable evidence bound to a published artifact` |
| **REPLACE** | `DELETE is forbidden` | `certified_coverage is append-only: DELETE is forbidden - this also blocks INSERT OR REPLACE, which satisfies a conflict by deleting` |

REPLACE is required to be refused by the **DELETE** trigger specifically, because REPLACE satisfies
the conflict by deleting the losing row — that is the exact mechanism the production
`recursive_triggers` pragma exists to make observable, so asserting the generic classification
alone would not have proved it.

The arm additionally now checks the survivor is **unchanged**, not merely present:
`run_id == 'production'` and `seed_end_exclusive == 1,000,000,000`.

---

## 3. §C — `recursive_triggers` ON/OFF, BOTH DIRECTIONS

New arm `arm_recursive_triggers_is_load_bearing`. Same table, same existing `coverage_id`, same
fully valid REPLACE; **the pragma is the only difference between the two halves.**

```
--- recursive_triggers = OFF ---
  REPLACE SUCCEEDED
  stored: run_id='smoke-test'  end=1,000        rows=1
  (original was: run_id='production'  end=1,000,000,000)

--- recursive_triggers = ON ---
  aborted: IntegrityError -> certified_coverage is append-only: DELETE is forbidden
           - this also blocks INSERT OR REPLACE, which satisfies a conflict by deleting
  stored: run_id='production'  end=1,000,000,000  rows=1
```

**The OFF arm is proven to actually replace the row**, as required — the arm fails if REPLACE
succeeds without the record changing, because a "successful" REPLACE that changed nothing would not
demonstrate the clobber the pragma prevents. This is the billion-seed production interval being
destroyed by a 1,000-seed smoke row: the precise incident that damaged the legacy tracker twice.

**The production rationale is therefore now certified rather than asserted.** The comment at
`utils/seed_coverage_ledger.py:537` claiming the pragma is load-bearing was, before this arm, an
uncertified claim about a pragma.

The arm also AST-checks `CoverageLedger._connect` and reds if it stops setting
`recursive_triggers = ON` — so the guarantee cannot be silently removed from production while the
gate stays green.

*(Environment note: `sqlite3` library version **3.37.2** on VM101. The ON/OFF asymmetry is SQLite
behaviour, so this arm is also a regression detector if that behaviour ever changes under a
different runtime.)*

---

## 4. RED-FIRST / MUTATION EVIDENCE FOR THE CORRECTION ITSELF

Test-only mutations, in a throwaway `4dd5535` worktree carrying the R1/R2 files (plus the
gitignored dataset files the WATCHER arms need — VIR-6).

```
CONTROL (R2 suite)          40/40 checks green   failures=[]

M-COLS — R2 fixture pointed at the PRE-R1 columns          38/40
  [RED] G-NO-REPLACE-CLOBBER: UPDATE/DELETE/REPLACE abort for the append-only reason
  [RED] G-NO-REPLACE-CLOBBER: recursive_triggers=ON is load-bearing (ON/OFF)
  reason: AssertionError: the append-only fixture no longer matches the live schema
          (missing=['prng_base', 'skip_modes_executed'],
           extra=['mapping_mode', 'prng_type']). Fix the fixture — do NOT let a
           column error stand in for append-only enforcement; that is the R2 false green.

M-REASON — pre-R2 arm restored verbatim (bad columns + bare except)    40/40
  [FALSE GREEN REPRODUCED] the pre-R2 arm PASSES on a bad-column error,
  exactly as Beta found

RESTORED                    40/40 checks green   failures=[]
```

This satisfies both halves of Beta's requirement:

* **the corrected arm no longer passes on a bad-column error** — M-COLS reds it with a named
  diagnosis rather than letting the schema error through (and reds the new pragma arm too, since
  both share the reconciled fixture);
* **the pre-R2 form is demonstrably a false green** — M-REASON reproduces it at 40/40, which is
  what makes the reason assertion the load-bearing change rather than a stylistic one.

---

## 5. §D — THE RENAMED WITNESS GATE

**No behaviour change.** `_require_publication_witness` accepts an `isinstance` match **or** an
object satisfying the complete frozen dataclass contract, which is Beta's own wording, so the
implementation stays exactly as R1 shipped it.

| | |
|---|---|
| **was** | `R1-A: only a canonical RunArtifactResult may certify` |
| **now** | `R1-A: canonical publication witness / complete frozen contract required` |

The function is renamed to match (`arm_witness_requires_complete_frozen_contract`), and its
docstring now states the accurate guarantee: **the production call path plus complete
frozen-contract validation — not an unforgeable Python object.** A stand-in that genuinely
reproduces all 28 fields with correct types *is* the contract; what is refused is the pre-R1
bypass, an object whose entire claim to be a publication is a well-formed digest.

---

## 6. CONFIRMATION THAT NO PRODUCTION FILE CHANGED

`git diff --stat 4dd5535`, byte-identical to the figures accepted at R1:

```
 agents/watcher_agent.py               | 163 ++++++++++++++++++++++++++--------
 database_system.py                    | 150 ++++++++++++++++++++++++-------
 window_optimizer.py                   | 100 +++++++++++++++++++++
 window_optimizer_integration_final.py |  42 +++++++++
 4 files changed, 386 insertions(+), 69 deletions(-)
```

**R2 changed exactly one file: `tests/test_seed_domain_cursor_amendment.py`** (untracked, so it
carries no diff line). `utils/seed_coverage_ledger.py` is untouched by R2.

Per Beta, **the staging regression suites were NOT re-run** — no production file changed, so the
already-accepted results stand (50/50 · 24/24 · 6/6 · 60/60, and phase-4 63/63 in the committed
model, all recorded in the R1 report).

**Live database re-verified after three full suite runs:** `certified_coverage` absent,
`exhaustive_progress` unchanged at 15 rows. The R1 test-DB isolation continues to hold.

---

## 7. VERIFICATION

Last edit `tests/test_seed_domain_cursor_amendment.py` at **09:33:29**; all three runs and this
report followed it.

```
RUN 1: 40/40 checks green | COMPLETION SENTINEL: PASS
RUN 2: 40/40 checks green | COMPLETION SENTINEL: PASS
RUN 3: 40/40 checks green | COMPLETION SENTINEL: PASS
```

39 → 40 checks: one added (`recursive_triggers` ON/OFF), one corrected in place, one renamed.

---

## 8. DISAGREEMENTS

**None.** The finding is correct, the classification is correct — it is a verification-integrity
defect, not a production defect; the bare INSERT, both triggers and the pragma were all right as
implemented, and the ON/OFF arm now proves the last of those rather than asserting it. The §D
wording correction is also right: my gate title claimed more than the code enforces, and the
narrower statement is the true one.

**No production change was needed and none was made.** The one moment it might have been tempting —
proving the pragma is load-bearing — is satisfied by toggling the pragma **on the test's own
connection**, which is what makes it a clean A/B against unchanged production code.

### Two items recorded, neither a task

1. **Carried forward as instructed, and NOT restated as a stronger claim:** WATCHER only
   auto-overwrites `seed_start` when `next_seed_start > 0`, so an explicitly supplied nonzero
   operator start can remain in force while the certified cursor still reports the first uncovered
   seed as zero. **Nothing in R1 proves WATCHER forcibly rewrites every requested run to the first
   gap, and this report does not claim it.** For gate 12 the certified first-gap value will be
   supplied **explicitly**, and under the empty certified `{constant, variable}` namespace that
   value is **0**.
2. **A generalizable lesson from this defect, offered for the backlog, not actioned:** the R2 false
   green existed because a *fixture* was pinned to a schema by hand while production moved. Any
   gate that hand-writes a column list against a live table has the same exposure. The fix used
   here — build the statement from `PRAGMA table_info` and reconcile a keyed mapping against it —
   is mechanical and would be worth applying wherever else a suite hardcodes a schema.

---

## 9. VERIFICATION-INTEGRITY CONTROLS (VIR-1…6)

* **execution proof:** per-arm PASS/FAIL, explicit `COMPLETION SENTINEL`, matching exit codes;
  control / M-COLS / M-REASON / restored runs all captured.
* **clean control:** control asserted green before any mutation; the `recursive_triggers=OFF` half
  is itself the control that makes the ON half meaningful, and it must demonstrate a **real**
  replacement to count.
* **fault-injection control:** M-COLS (corrected arm must red) and M-REASON (pre-R2 arm must
  reproduce the false green) — both directions, since proving only that the new arm passes would
  not have shown the old one was wrong.
* **completion sentinel:** `COMPLETION SENTINEL: PASS — seed-domain cursor amendment green`.
* **unavailable-observer behavior:** the arms fail closed on schema drift with a named diagnosis
  rather than accepting any SQLite error as evidence.
* **audit claim scope:** the append-only enforcement of `certified_coverage` and the witness gate's
  wording, on this VM101 tree at `4dd5535` plus the working-tree diff, under sqlite3 3.37.2. No
  claim about rig-side state, a live fleet run, Gate 12, or `n_parallel > 1`.
* **searched surfaces:** the R2 brief and the R1/R0 governance trail · the live suite and ledger
  source · live SQLite schema, trigger and row introspection · AST of
  `CoverageLedger._connect` · a `4dd5535` worktree for mutation runs · the production database
  (read-only verification).
* **unavailable surfaces:** the three rigs · Proxmox host kernel logs · any live-fleet or GPU
  path. **No claim here depends on them**, and no staging suite was re-run because no production
  file changed.
* **governance trail searched:** the R2 ruling as carried by the brief, the R1 ruling and report,
  and D3.5's frozen result contract.
* **chapters searched:** not required — no claim here concerns sieve mathematics, kernel semantics
  or feature provenance.
