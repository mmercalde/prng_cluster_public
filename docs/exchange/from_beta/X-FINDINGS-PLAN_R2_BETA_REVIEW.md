# X-FINDINGS-PLAN · R2 · BETA REVIEW

- **Thread / round:** X-FINDINGS-PLAN / R2
- **Date:** 2026-11-13, America/Los_Angeles
- **Reviewer:** Team Beta (Codex)
- **Class:** AFTER-COMMIT, documents. Review position, not approval.
- **Base read:** `0066b6e9a2d701a41d26a22e7e338f9073104718` (HEAD at initial read and pre-write recheck).
- **`git status --porcelain` at initial read and pre-write recheck:** empty output.

| Question | Position | Severity |
|---|---|---|
| Q1 — SQLite procedure | NO OBJECTION | — |
| Q2 — Isolation additions | NO OBJECTION | — |
| Q3 — RANGE-MINER reads | NO OBJECTION | — |
| Q4 — Adoption of suggestions | NO OBJECTION | — |
| Q5 — Measurement rules against R2 §4 | OBJECTION | THEORETICAL — The omissions affect a future WP-3 performance comparison, while the addendum expressly starts no WP-3 session today. |

**Disposition:** close both R1 LIVE objections under the superseding addendum. The literal comparison with the supplied reverse-sieve R2 is now possible; it reveals remaining measurement-plan omissions detailed under Q5. No LIVE objection remains in this review. The THEORETICAL objection holds no commit.

## Scope and evidence

I read the request, addendum, proposal, original three briefs, prior findings-plan review, supplied reverse-sieve R2, and the source anchors below. The supplied R2's SHA-256 is `8e03153ab1ddfe05eb30d96323a4a442cc2799e2284d266d73c2da15d47b7da4`, matching the request. This verifies the local bytes against the supplied digest; it does not independently establish their hand-carried provenance. The unavailable reverse-sieve R1 is not needed to compare the now-present R2 §4.

I ran synthetic SQLite checks in `/tmp` with the activated `~/venvs/torch` environment and bytecode writes disabled. I opened no live ledger, ran no pipeline or worker, made no rig connection, and created no worktree, branch, tag or commit. Source-routing conclusions below are inspection, not production execution.

## Q1 — SQLite procedure

**Yes: §A correctly implements the procedure requested in R1 for the checkpointed and nonempty-WAL cases. No remaining SQLite sidecar-write route is prescribed against the original.** The addendum overrides the old instructions (`docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md:6`), permits only an immutable URI against an eligible original, refuses a nonempty original WAL, confines recovery/query work for that case to a scratch copy, and prohibits source repair (`docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md:26`, `:31`, `:32`, `:37`). Its stopped-writer and before/after stability checks are substantive preconditions, not permission to query a changing source (`:28`, `:35`, `:38`).

The prohibition on production helpers closes the second route: `CoverageLedger.__init__` calls schema initialization (`utils/seed_coverage_ledger.py:704`), `_connect` uses ordinary `sqlite3.connect` (`utils/seed_coverage_ledger.py:709`), and initialization executes DDL (`utils/seed_coverage_ledger.py:719`). The wrapper constructs that helper lazily (`database_system.py:350`, `:353`). The existing verifier also refuses a nonempty WAL and otherwise uses immutable mode (`tests/gate_s172_prod_shape.py:152`, `:158`).

**Executed, SQLite 3.37.2 in this environment:**

- On a closed, checkpointed WAL fixture, immutable reading returned `7` and left source filenames, sizes, mtimes and hashes unchanged. A subsequent `mode=ro` control created both `-wal` and `-shm`.
- A subprocess committed `9` into a WAL and exited without closing SQLite. After that writer had stopped, a deliberately forbidden main-file-only immutable read of this synthetic fixture returned `7`. Copying the database and all its sidecars to scratch and querying the copy returned `9`. Source names, sizes, mtimes and hashes remained unchanged across the checks.

These results support both branches, including the need to preserve the WAL during an offline copy. They are fixtures, not evidence that any actual ledger is checkpointed, stable or readable. “Cannot write beside the original” here describes the compliant reading operations; it is not a guarantee that another process cannot start writing. Section B explicitly recognizes that limit (`docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md:55`).

## Q2 — Isolation additions

**R1 Q4 no longer stands under §§A–B.** Section B names the root launch harness, prohibits shared refs and commits, keeps both gates scoped to the main tree, and requires renewed process checks (`docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md:48`, `:50`, `:53`, `:55`). These address the recorded isolation qualifications:

- The relay hashes all refs separately from HEAD and main-tree status (`scripts/beta_relay.py:248`, `:249`, `:250`). A detached worktree avoids adding a branch; it does not isolate absolute live paths, which §A now governs.
- The clean-tree predicate queries the specified repository (`window_optimizer_integration_final.py:115`, `:119`), and the launch gate passes its repository root to that predicate (`scripts/gate12_cleantree_gate.py:117`). Gate 22 reads porcelain at its own `_ROOT` (`tests/test_s172_phase4_coordinator.py:1626`). Running them from the main tree avoids accepting scratch-tree results as main-tree evidence.
- The root harness contains the actual launch and its parameter block (`gate12_launch.sh:377`, `:379`); explicitly naming it closes the no-touch enumeration gap.

Section B also corrects the characterization of G2/G3 as read-only (`docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md:57`). Its account of what that session actually wrote or probed is **reported by Alpha, not independently verified here**. This review closes the prospective instruction objection; it does not certify that session's historical compliance or authorize further probes.

## Q3 — RANGE-MINER reads

**R1 Q5 no longer stands under §§A and C. No substantive source misstatement found in §C.**

The cursor additions correctly distinguish count from first gap and scope qualifying records by base and mode containment. Those are separate output fields (`utils/seed_coverage_ledger.py:449`, `:453`); records are filtered by requested modes (`:1018`), and the cursor applies `first_uncovered_seed` to their intervals (`:1049`). WATCHER refuses COMPLETE (`agents/watcher_agent.py:2069`) and replaces the requested start when the cursor is positive (`:2087`, `:2089`). Section C appropriately says it **can** replace the start, rather than claiming an unconditional overwrite.

The command distinction is also correct: the root harness's block lacks warm-start keys (`gate12_launch.sh:379`), whereas the retained Run-4 command contains all seven (`docs/SESSION_CHANGELOG_20260828_RUN4_ROUTING_PATCH.md:221`, `:222`). **Clerical anchor correction:** extend §C's `:211–218` citation through `:222`; its current range stops before the warm-start flags. The claim itself is supported.

The count distinctions are supported by the implementation: canonical records iterate only the intersection, with directional populations in aggregate metadata (`utils/canonical_records.py:211`, `:234`, `:239`, `:246`). Publication selects L2 winners, merges the prior arrays and then calculates final rows (`utils/run_finalizer.py:1613`, `:1640`, `:1659`); parent identity and the four requested counts are retained (`:1665`, `:1696`). Section C's “where available” qualification is necessary for per-mode attempt totals; row membership cannot supply them.

The required “Not launch readiness” section correctly leaves later fleet, authorization, environment and volume gates outside these reads (`docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md:90`; `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:76`, `:79`, `:91`). Actual cursor values, attempt-9 counts and the effective intended Run-4 interval remain **UNAVAILABLE in this review**, not zero and not passed.

## Q4 — Adoption of suggestions

**No misstatement found. §§D–E adopt the R1 suggestions.**

Section D names the batch entry (`full_scoring_worker.py:375`), vectorized `skip=0` route (`survivor_scorer.py:574`), CPU fallback both after failure and when the vectorized implementation is absent (`:578`, `:581`), CPU fallback's `skip=0` call (`:807`), and outer sequential fallback (`full_scoring_worker.py:438`, `:445`). It separates stored dictionary features from fresh extraction for integer survivors (`prediction_generator.py:782`, `:787`) and asks for exact compatibility controls and artifact-regeneration consequences (`docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md:104`, `:108`). Those are the requested distinctions. No GPU behavior was executed here.

Section E preserves the distinction between existing `exact_hits` and one forecast membership hit, and requires consumer tracing (`docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md:117`). The code increments once per winning entry (`chapter_13_diagnostics.py:242`, `:248`), while coverage uses distinct values (`:262`). Its eligibility requirements and controls, including pending outcomes and valid `000`, match R1 (`docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md:123`, `:126`). This is acceptance of the investigation instructions, not a finding that the existing grading integration is repaired.

## Q5 — Measurement rules against the supplied R2 §4

**Yes, several explicit requirements remain missing or weaker in the three specified sections taken together.** The earlier NEEDS EVIDENCE limitation is resolved by the supplied text; full adoption is not established.

Already retained: distinct `k`, one membership hit, summed null expectation, advance pool-size/period/abstention choices, uncertainty, per-draw records and unavailable-input handling (`docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:80`). Section F adds value validation, frozen target/cutoff/configuration binding, duplicate-target resolution, abstention denominators, varying-`k` uncertainty and a tolerance for random controls (`docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md:132`).

The remaining differences are:

1. **Explicit isolation of feature fitting and configuration selection.** R2 names both and states that training holdout labels are not automatically final test data (`docs/exchange/from_beta/X-REVERSE-SIEVE_R2_BETA_REALIGNMENT.md:114`). Proposal §1 names training, target construction, Optuna and threshold tuning but omits those explicit qualifications (`docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:38`). Binding a configuration identity in §F is useful provenance; it does not establish how that configuration was selected.
2. **Paired comparison and accounting.** R2 requires the same target draws and comparable coverage/budget, or explicit accounting for differences (`docs/exchange/from_beta/X-REVERSE-SIEVE_R2_BETA_REALIGNMENT.md:122`). Neither WP-3 nor §F states this. Expected hits alone do not make two differently covered periods a fair configuration comparison.
3. **Advance comparison rule.** R2 requires declaring it before inspecting results (`docs/exchange/from_beta/X-REVERSE-SIEVE_R2_BETA_REALIGNMENT.md:124`). WP-3 declares pool size, period and abstention but does not name the comparison rule (`docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:83`).
4. **A further untouched period after result-guided choice.** R2 requires this explicitly (`docs/exchange/from_beta/X-REVERSE-SIEVE_R2_BETA_REALIGNMENT.md:126`). It is absent from the specified sections. “Never saw” for a forecast does not by itself prevent choosing the reported configuration after inspecting that period's results.
5. **Interpretation of uncertainty and repeated comparisons.** R2 says a point estimate above baseline is insufficient by itself, and retains per-draw records so dependence and repeated comparisons can be assessed (`docs/exchange/from_beta/X-REVERSE-SIEVE_R2_BETA_REALIGNMENT.md:125`). WP-3 retains uncertainty and records, but drops these explicit assessment purposes. A varying-`k` uncertainty formula alone does not specify how either issue is handled.

**Suggested addition to §F for Alpha to apply:**

> Final evaluation outcomes must be unseen by feature fitting and configuration selection as well as the stages listed in proposal §1; training holdout labels are not automatically final test data. Compare configurations on the same target draws with comparable coverage and budget, or account explicitly for differences. Declare the comparison rule before inspecting results. A point estimate above baseline alone is insufficient evidence of lift. If results guide configuration choice, evaluate the chosen configuration on a further untouched period. Preserve ordered per-draw records and configuration identities so dependence and repeated comparisons can be assessed, and specify how the uncertainty/comparison procedure handles them.

This objection concerns completeness of the future measurement specification. Section F expressly starts no session now (`docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md:130`); no WP-3 implementation or performance result is supplied for acceptance here. That is why severity is THEORETICAL. It does not reopen either SQLite objection, establish predictive loss, or require changing the sieve. The settling evidence is the completed instrument specification and fixture checks, followed by an eligible frozen-output evaluation under those declared rules.

## Checks and additional findings

**Executed:** `git status --porcelain`, HEAD checks, source/document reads and targeted `rg` searches; SHA-256 verification of the supplied R2; the SQLite fixture script `/tmp/X-FINDINGS-PLAN_R2_beta.fww253zo/checks.py` (exit 0), with output in `checks.log` and structured evidence in `results.json`. SQLite here is **3.37.2**; Alpha's reported 3.53.1 execution was not repeated in that interpreter. No full production suite or launch gate was run for this document review.

**Status search:** searched `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md` and the relevant exchange threads for immutable/WAL/journal and the omitted measurement-rule phrases. The SQLite hazard and measurement requirements are already recorded; this review reports their adoption status, not newly discovered production defects.

1. **No additional finding outside Q1–Q5.** Historical G2/G3 execution, live-ledger contents, GPU/fleet health and prospective TFM hit rate were not established. Original source/document content and HEAD remained unchanged at the final check; the only repository write was this response. The response's five positions were accepted by the existing relay parser, and all cited local anchors were checked for existence and range.
