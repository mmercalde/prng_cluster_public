# CLAUDE_CODE_INSTRUCTIONS_CHAPTER_1_P1_P2_CORRECTION.md — REV1

**Chapter 1 remediation, second tranche: P1 and P2 — audit items 6–17.**

**DOCUMENTATION ONLY. No production code changes.** Tranche 1 (`ddd2ac8`) carried the three
behavioural fixes; this tranche is the remaining correction list from
`docs/CHAPTER_1_AUDIT_v1.md` §6. Two items touch a *script*, not the pipeline — see §3.

**Base:** current `main` on VM 101. Claude Code as `michael`, venv `~/venvs/torch`. Implement
and iterate; do **NOT** commit, push, or run WATCHER. STOP at the gate for Team Alpha review.

**⚠️ Concurrency:** other read-only audit sessions may be running. This tranche edits
`docs/CHAPTER_1_WINDOW_OPTIMIZER.md` and one script. If another process appears to be editing a
file you are working on, stop and report rather than racing it.

---

## 0. Authority and standing constraints

- **Findings:** `docs/CHAPTER_1_AUDIT_v1.md` (`db9782a`), correction list §6 items 6–17.
- **Tranche 1:** `ddd2ac8` — already landed; do not redo it. The chapter's skip-defect callout,
  bounds snapshot, `resolve_directional_threshold()` invariant, dead-dimension record and
  output-file contract are **done**.
- **Foundations:** the `tfm-project-facts` skill §0. **§0.4 governs every judgement here.**
- **`CHAPTER_1_PATCH_S114.md` is superseded** — see item 13. Do not merge it verbatim.

**A correction to a claim tranche 1 committed.** The audit
`docs/STRATEGY_ORIGIN_AUDIT.md` established that the chapter's statement at ~`:965-985` —
*"Root cause is code rot, not design"* — is **misleading**, and the same claim in the
`ddd2ac8` remedy comment carries the same defect. Two committed documents
(`SESSION_CHANGELOG_20260207_S63.md:9`, `PROPOSAL_SEARCH_STRATEGY_VISIBILITY_FIX_v1_0.md:20`)
state the design was **four Optuna samplers**. `RandomSearch` was always hand-rolled;
`GridSearch`/`EvolutionarySearch` were **working code deleted to `return {}`** in Nov 2025.
**Correcting the chapter's wording is item 8 below.** The in-code comment is corrected
separately — do not edit `window_optimizer.py` in this tranche.

---

## 1. P1 — corrections that prevent wasted debugging (items 6–10)

**Item 6 — fix the header.** Declared `Version: 3.1` has no source counterpart (the live module
docstring says `Version: 2.0`). Declared `~868 + ~595` lines vs actual **1306 + 2679**. Add the
third load-bearing module the header omits: **`window_optimizer_bayesian.py` (984 lines)**,
which owns the Optuna search space, study storage and warm-start. Note that
`docs/window_optimizer_integration_final.py` (1877 lines) and `modules/window_optimizer.py`
(327 lines) are **stale duplicates ruled to be left in place** — so the next reader neither
edits them nor re-proposes deleting them. **Live modules are the repo-root ones.**

**Item 7 — document the backend cascade** (§2.1 / §11): miner → PWC → ZMQ → legacy, and the
argparse mutex at `window_optimizer.py:1143-1154`. Without it, §2.1 describes a path most
production runs do not take.

**Item 8 — mark `--strategy random|grid|evolutionary` as gated, and correct the cause.**
State the `TypeError` mechanism (`optimize()` forwards four kwargs; only
`BayesianOptimization.search` accepts them) and the stale `SearchStrategy` ABC
(`window_optimizer.py:299-303`) as the reason it went unnoticed. **Then correct the "code rot,
not design" claim** per §0: the documented design was four Optuna samplers; Grid and
Evolutionary were deleted working code. Cite `docs/STRATEGY_ORIGIN_AUDIT.md`.

**Do not state a remedy.** The prescription is under Beta ruling
(`TEAM_ALPHA_AUTONOMY_CONTROL_SURFACE_SUBMISSION.md` Q3). Say what is broken and why; say the
repair is pending a ruling. **Specifically do not repeat "bring the signatures up to the calling
convention"** — for Grid and Evolutionary that would turn a signature-derived gate green on a
function returning `{}`.

**Item 9 — bring §10.1 to all 31 CLI flags**, correcting the three inconsistent
threshold-bound figures to the single live pair. Reference the §4.1 snapshot rather than
restating numbers.

**Item 10 — correct the §8.3 governance picture:** per-session ruling; combined-session
sequential sieve **non-certifying, prohibited by default**; hybrid certification **blocked**
pending the skip wire-in; PWC hybrid **quarantined**
(`persistent_worker_coordinator.py:176`).

## 2. P2 — completeness and mechanical (items 11–17)

**Item 11** — refresh §8.1/§8.2/§9.1/§9.2 signatures and flows: coverage write-back, the
incremental merge, TRSE feedback, the NPZ-conversion gate.

**Item 12** — add `validate_baseline_in_bounds()` (TB mandate) to §3.2 and §14.3.

**Item 13** — fold in the **surviving** parts of `CHAPTER_1_PATCH_S114.md`: the
`--resume-study` CLI semantics only. Rewrite warm-start from
`window_optimizer_bayesian.py:627-650` (context-driven from `step1_trial_history`, gated on all
six params non-`None` at `:639`, requires `session_idx` at `:643`). **Drop the hardcoded
`W8_O43` enqueue block — that code was deleted by S144.** Re-frame the "discrete regime
structure" discovery against the S172 ruling that raised the window floor to 6 and reinterprets
W3's 143,959 survivors as **noise, not signal**. Then **mark the patch file superseded** so it
stops being read as current.

**Item 14 — fix Appendix A:** `run_trial_persistent` `:669` → `:1612`; delete the non-existent
`execute_local_sieve_job()`; retract the "zero changes" invariant; remove the missing study DB;
correct **JournalStorage → SQLite**.

**Item 15 — delete the duplicated S146 section** (chapter lines 1099–1133) **and fix
`apply_s146_doc_updates.py:48`**, whose idempotency guard tests for a `label` string that is
never written to the file — otherwise the next run re-duplicates it. **This is the one script
edit in this tranche**; it is a doc-generator, not pipeline code.

**Item 16 — repair the stray code fence at chapter line 885.** Everything after it currently
renders inverted.

**Item 17 — update the §16 line-count table** and re-point "Next Chapter" at the RANGE-MINER
replacement of Step 2. Note Chapter 2 is currently a 128-line fragment pending
restore-and-audit (`docs/CHAPTER_2_SOURCE_MAP_v1.md`) — **do not** describe it as complete.

## 3. Scope

**In scope:** `docs/CHAPTER_1_WINDOW_OPTIMIZER.md`, `apply_s146_doc_updates.py:48` (item 15),
and marking `docs/CHAPTER_1_PATCH_S114.md` superseded (item 13).

**Explicitly NOT in scope:**
- **Any production code.** Not `window_optimizer.py`, not
  `window_optimizer_integration_final.py`, not the miner, PWC, ZMQ or kernels.
- The `ddd2ac8` in-code remedy comment — corrected separately.
- The hybrid skip wire-in; sampler work; anything under Beta ruling.
- The four behavioural defects flagged as separate tickets (`run_with_config` writing `[]`;
  `window_optimizer.py:798` calling `logger.warning` with no `logging` import — **unverified at
  runtime**; combined-session sampling).
- The two stale duplicates — **ruled to be left in place.** Do not edit or delete.
- `distributed_config.json` bare-metal addresses — deliberate (`CLAUDE.md` §3).

## 4. Constraints on the writing itself

- **Numbers reference the snapshot, never restate it.** Tranche 1 established
  `scripts/extract_search_bounds_snapshot.py` with `repository_commit` and
  `configuration_digest`. Any new numeric claim either cites live source with `file:line` or
  points at the snapshot. **Do not hand-copy values** — that is the "~62 features" failure class.
- **Cite, do not duplicate:** `THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md`,
  `HYBRID_SKIP_BOUND_AUDIT.md`, `STRATEGY_ORIGIN_AUDIT.md`, `DAILY3_CONSUMER_CONTRACT_v1.md`,
  `TRSE_STEP0_AUDIT_v1.md`, `CHAPTER_2_SOURCE_MAP_v1.md`.
- **Per §0.4, describe defects without prescribing removal.** Where a remedy is under ruling,
  say so.

## 5. Verification-integrity controls (VIR-1…6)

- **execution proof** — every corrected claim carries a `file:line` anchor read this session, or
  cites a named audit.
- **clean control (VIR-2)** — the audit listed ~24 sections **verified correct and must-preserve**
  (`CHAPTER_1_AUDIT_v1.md` §9). **Confirm those are unchanged.** A correction pass that silently
  rewrites a correct section is a regression.
- **fault-injection control** — for item 15, prove the `apply_s146_doc_updates.py` guard fix
  actually prevents re-duplication: run it twice against a scratch copy and show the section
  appears once. Without that, the fix is unverified.
- **completion sentinel (VIR-3)** — explicit `PASS | FAIL | UNAVAILABLE | INCOMPLETE` and a
  per-item coverage table. An item not reached is `INCOMPLETE`, never silently absent.
- **unavailable-observer (VIR-5)** — anything unverifiable is `UNAVAILABLE`, not assumed correct.
- **audit claim scope (VIR-6)** — declare searched and unavailable surfaces.

## 6. Non-regression

No production code changes, so the full suite is not required. **Do run:**
`tests/test_chapter1_p0_corrections.py` (12/12 — tranche 1 must not regress) and
`tests/test_s172_phase4_coordinator.py` (63/63 — gate 22 sees changed `.py` files, and item 15
edits one). If gate 22 flags `apply_s146_doc_updates.py`, register it in the allowlist **with
rationale**, per the established pattern.

## 7. Report

Per item 6–17: what changed, with `file:line` or the audit citation it derives from. The item-15
double-run proof. Confirmation the §9 must-preserve sections are untouched. Explicit statement
that no production code was modified. Then STOP. **Do not commit.**
