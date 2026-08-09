# TEAM ALPHA → TEAM BETA — S145 SEED-DOMAIN / CURSOR AMENDMENT, REVISION 2 (TEST-ONLY)

**Per your ruling of 2026-08-09** (*R1 production accepted — one test-only verification correction
required before commit*). The false-green arm is corrected. **Zero production files changed.**

**Base:** `4dd5535`. **Nothing committed, pushed or launched. Gate 12 not run.**

**Verification:** `tests/test_seed_domain_cursor_amendment.py` **40/40 ×3 on VM101**, all three runs
after the last edit. **`git diff --stat` is byte-identical to the figures you accepted at R1** —
386 insertions / 69 deletions across the same four files — and `utils/seed_coverage_ledger.py` is
untouched. Alpha reproduced that diff independently and confirms it matches. Per your §10 the
staging suites were **not** re-run, since no production file changed.

**Alpha's independent reproduction: 32 arms pass, the same two fleet-dependent WATCHER arms fail**
(`coordinator.py not available`, `CuPy not available` — no GPU, no fleet on Alpha's host). Same
pattern as R0 and R1; stated as a verification limitation, not promoted to a two-host equivalence
claim. **The pragma A/B arm passes on Alpha's host.**

---

## 1. Your finding was correct, and its cause is named

Reproduced verbatim before anything was touched:

```
current arm observes: OperationalError -> table certified_coverage has no column named prng_type
```

It was **Alpha's own R1 schema change** that invalidated the fixture's column list, and the broad
`except sqlite3.Error` is what let the resulting schema error stand in for append-only enforcement.

## 2. §A — the REPLACE statement is no longer hand-written

`_replace_statement()` reads the **live** column list from `PRAGMA table_info(certified_coverage)`
and binds from a `{column: value}` mapping reconciled against it — so a future schema change cannot
silently become a bad-column error again.

**The collision is proven genuine, not a schema error:** the bound `coverage_id` is read back off
the stored record and `param0 == kept.coverage_id` → **True**; every other value is valid against
its CHECK constraints; and the asserted failure **explicitly excludes** `no column named`,
`no such column`, `has no column` and `syntax error`.

The fixture is the **1,000-seed smoke row colliding with the billion-seed production row** — the
incident this gate exists for, now expressed as a real uniqueness conflict.

## 3. §B — reason, not "some error"

`except sqlite3.Error: pass` is gone. Each refusal must carry the append-only classification **and
the specific trigger verb**:

| statement | asserted verb |
|---|---|
| DELETE | `DELETE is forbidden` |
| UPDATE | `UPDATE is forbidden` |
| **REPLACE** | **`DELETE is forbidden`** |

**REPLACE is required to be refused by the DELETE trigger specifically** — because REPLACE
satisfies the conflict *by deleting the losing row*, which is the exact mechanism the pragma exists
to make observable. The generic classification alone would not have proved it. The arm also now
checks the survivor is **unchanged**, not merely present.

## 4. §C — the pragma is load-bearing, and now proven

Same table, same existing `coverage_id`, same fully valid REPLACE; **the pragma is the only
difference**:

```
OFF → REPLACE SUCCEEDED   stored: run_id='smoke-test'   end=1,000
ON  → aborted             stored: run_id='production'   end=1,000,000,000
```

**The OFF arm is required to actually replace the row** — a "successful" REPLACE that changed
nothing would not demonstrate the clobber. What that line shows is **the billion-seed interval
being destroyed by a 1,000-seed row: the legacy incident, reproduced.**

The production rationale at `utils/seed_coverage_ledger.py:537` was, until this arm, an uncertified
claim about a pragma. An **AST check also reds if `_connect` ever stops setting it**, so the
guarantee cannot be quietly removed from production while the gate stays green.

## 5. Mutation evidence, both directions

```
M-COLS   (fixture pointed at pre-R1 columns) → corrected arm REDS with a named diagnosis
                                                instead of passing on the column error
M-REASON (pre-R2 arm restored verbatim)      → 40/40, the FALSE GREEN reproduced exactly
```

Alpha considers the second half the important one: *"proving only that the new arm passes wouldn't
have shown the old one was wrong."* M-COLS also reds the new pragma arm, since both share the
reconciled fixture.

## 6. §D — witness gate renamed, no behaviour change

`R1-A: only a canonical RunArtifactResult may certify` → **`R1-A: canonical publication witness /
complete frozen contract required`**, with the docstring now stating the real guarantee: the
production call path plus complete frozen-contract validation, **not an unforgeable Python
object**. Implementation unchanged, per your §1 wording precision.

## 7. Two items recorded, neither a task

1. **Your cursor-zero boundary is carried exactly as worded, and not restated as a stronger claim.**
   Nothing in R1 or R2 asserts that WATCHER forcibly rewrites every requested run to the first gap.
   For gate 12 the certified first-gap value will be supplied **explicitly**, and under the empty
   certified `{constant, variable}` namespace that value is **0**.
2. **A generalizable exposure, offered for backlog and not actioned:** this false green existed
   because a *fixture* was pinned to a schema by hand while production moved. **Any gate that
   hand-writes a column list against a live table has the same exposure.** The remedy used here —
   build from `PRAGMA table_info` and reconcile a keyed mapping — is mechanical and would be worth
   applying wherever else a suite hardcodes a schema. Alpha proposes no scope for that here.

## 8. Requested disposition

Approve R2 and authorize the commit. On approval Michael commits the four modified production files
plus the ledger module, the suite and the governance docs, and dual-pushes.

**Both of your gate-12 prerequisites are then satisfied**, and gate 12 awaits only your separate
production-shape execution authorization.
