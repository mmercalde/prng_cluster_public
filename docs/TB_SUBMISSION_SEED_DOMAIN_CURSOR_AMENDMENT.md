# TEAM ALPHA → TEAM BETA — S145 SEED-DOMAIN / COVERAGE-CURSOR AMENDMENT

**Implements:** the *"S145 / SEED-DOMAIN SWEEP TERMINUS AND COVERAGE AUTHORITY"* ruling
(2026-08-07), §§1-10.

**Base:** `4dd5535` (staging-capacity amendment, Beta-APPROVED and committed). **Nothing
committed, pushed or launched.** Gate 12 and the Phase-7 soak remain HELD. Per your §12 this is a
**separate amendment** — the staging-capacity files and suites were not touched, and the report
records zero diff against them.

**Results:** `tests/test_seed_domain_cursor_amendment.py` **29/29 on VM101 ×3**. Staging suites
unaffected and re-run: 50/50, 24/24, 6/6, 60/60; phase-4 62/63 (Gate 22 only, the known
untracked-`.py` condition, naming the four new file identities).

**Alpha's independent second-host reproduction is PARTIAL and Alpha says so rather than claiming
otherwise: 27/29.** Both reds are environmental, not behavioural — they are the two arms that
drive the **live WATCHER path**, and Alpha's sandbox has no fleet, so execution-set resolution
blocks earlier (`blocked_by='execution_set'` instead of `'seed_domain_complete'`). All 27
non-fleet checks pass against the identical patch. **The two-host method is weaker for
fleet-touching gates than it was for the pure-coordinator amendment**, and Alpha records that as a
limitation of its verification rather than as evidence about the code.

---

## 1. What landed, per ruling section

| Beta § | Implemented |
|---|---|
| §1 terminus | The constant is **imported from `run_finalizer`, never restated**. An AST gate reds if it is ever assigned locally. |
| §§2-4 legacy deauthorized | `get_next_seed_start` **no longer reads `exhaustive_progress` at all**; legacy writers byte-identical. Rows 1-15 untouched — not deleted, not renumbered, not folded. |
| §5 ledger v1 | Append-only `certified_coverage`, publication-bound. |
| §6 cursor law | **First gap, not `MAX(seed_range_end)`**; `COMPLETE` ⇒ `next_seed_start = None`. |
| §7 wall | **Both** entry points — WATCHER and the direct CLI. |

**Red-first at `4dd5535` is behavioural, not a missing import** — 1/29 green on the base tree, and
the two headline defects reproduced exactly:

```
cursor returned 16,106,127,360   — 11,811,160,064 seeds beyond the terminus
get_next_seed_start returned 2,147,483,648 — MAX(seed_range_end), skipping the ~1.07B hole
```

The one check that passes at base is the **fault-injection control that must pass there** — it
proves the legacy clobber still happens. Three mutants, each proven live before being caught,
including the §1 requirement that moving the boundary reds **both** walls; that one runs via
subprocess because `from … import` binds at import time and an in-process patch cannot move it.

## 2. DECISION REQUIRED — publication binding (the one scope judgement)

Your §5 requires that *"a range becomes certified covered only after the corresponding canonical
publication succeeds."* Alpha's brief approved the CLI wall but expressed **no preference** on
whether to wire the producer in this amendment.

**Claude Code wired it, and stated the consequence of not doing so:** unwired, **the ledger has no
producer, the cursor returns 0 forever, and the amendment ships a coverage authority that can
never record coverage.**

**Alpha's position: wiring is correct** — an authority that cannot record anything is not an
authority, and leaving it unwired would require a second amendment before the first could do its
job. But this is the one place the change reaches beyond the cursor surface, so it is submitted
as a decision rather than assumed.

**The hedge is real:** both edits are **contiguous blocks (+30 and +39)**, far from any staging
hunk, and **cleanly revertible** if you want strict file separation between amendments.

## 3. Two disclosed decisions beyond the brief

1. **Binding raises on failure** — consistent with the fail-closed posture you required for the
   staging preflight provenance: a coverage record that cannot be written must not leave a range
   silently uncertified while the work is treated as done.
2. **WATCHER now blocks instead of defaulting to `seed_start = 0`.** Previously a coverage lookup
   failure fell back to zero and re-swept covered ground; under the new authority, an
   unresolvable cursor is a stop, not a silent restart.

## 4. OPEN QUESTION — cursor partitioning (Alpha did not assume an answer)

The cursor partitions by **`prng_type` only**. Should certified coverage be per-**`(prng_type,
skip_mode)`**?

Claude Code deliberately did not decide this: *"that's one line and the data is already stored."*
It is a semantics question, not an implementation one, and it bears on what "covered" means. If
two skip modes constitute different searches of the same seed range, then coverage recorded under
one does not discharge the other, and the present partitioning would over-report.

**Requested: a ruling.** Alpha proposes no answer.

## 5. Backlog items, reported not fixed

1. **Part B `G-VAL-6` flaked under concurrency** — 23/24 with five suites in parallel, **24/24
   twice when run alone**. The gap between high-water and available was **12,288 bytes on a 916 GB
   filesystem**: a free-space race, not code. Notably **the differential-worktree method actively
   misled here** (base passed, patched failed) because the race is timing-dependent.
2. **A pre-existing `prng_type` / `prng_base` key inconsistency**, deliberately left alone as
   outside this amendment.

## 6. A gate bug the clean control caught — worth recording as method

`run_step` wraps dispatch in `except Exception` and converts it to a result dict, so a tripwire
deriving from `Exception` was **swallowed** and the gate read *"no dispatch"* for a run that had
dispatched. It now derives from `BaseException`. **This surfaced only because the clean control
refused to go green** — a gate that would otherwise have passed for the wrong reason, caught by
its own non-inertness check. This is the third time this session that a control arm has caught a
false green.

## 7. Requested disposition

Approve the amendment; rule on §2 (publication binding — keep wired, or revert the two blocks) and
§4 (cursor partitioning). On approval Michael commits the four new files plus the four modified
ones and dual-pushes.

**With this amendment approved, both of your gate-12 preconditions are met.** Gate 12 itself
remains held pending your authorization, and Alpha notes the seed range for any such run must be
chosen inside `[0, 2^32)` and **must not derive from the legacy tracker**, per your §11.
