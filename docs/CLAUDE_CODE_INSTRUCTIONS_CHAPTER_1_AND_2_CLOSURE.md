# CLAUDE_CODE_INSTRUCTIONS_CHAPTER_1_AND_2_CLOSURE.md — REV1

**Close Chapter 1 and Chapter 2.**

Both have been corrected. This pass **verifies they are still accurate against HEAD** and adds a
closure statement to each. **Documentation only — no code, tests, config or manifests.**

**Base:** current `main` on VM 101 (`a87725b` or later). Claude Code as `michael`, venv
`~/venvs/torch`. You do **NOT** commit, push, or run WATCHER. STOP at the gate.

---

## 0. Why a verification pass rather than "we stopped editing"

Chapter 1's tranche-2 session found that **tranche 1's own code edits shifted
`window_optimizer.py` by ~286 lines and invalidated ~20 of its own anchors.** A correction pass
that edits code invalidates its own citations.

**Since Chapter 1 was corrected, more has moved:**
- bounded Phase 6 (`d98298c`) changed `window_optimizer_bayesian.py` by **+233/−90** — the
  neutral `run_optimization(..., sampler, sampler_metadata)` extraction, the random entrypoint,
  `describe_sampler`, and `SAMPLER_ENTRYPOINTS`;
- the Resolved Execution Set (`63e627f`) and admission binding (`eff6616`) changed
  `window_optimizer.py` again.

**So: re-verify, then close.** Not a rewrite.

## 1. Chapter 1 — verify and close

1. **Re-verify every `file:line` anchor** against HEAD. Where a line has moved, correct it. Where
   a citation is stable by function or symbol name, prefer that over a line number — the
   tranche-2 session used *"cite by function name only"* for a file dirty under a concurrent
   session, and that convention survives edits better.
2. **The §4.1 bounds snapshot** — regenerate it with `scripts/extract_search_bounds_snapshot.py`
   so `repository_commit` and `configuration_digest` are current. **Do not hand-edit it**; that
   was the whole point of making it machine-generated.
3. **The sampler entrypoints** — Chapter 1 predates the neutral extraction. State that
   `run_optimization(..., sampler, sampler_metadata)` is the core, that both arguments are
   **required and keyword-only with no default** (so a caller cannot get TPE by omission and
   report the run as something else), and that TPE and Random are thin wrappers.
   **`SAMPLER_ENTRYPOINTS` is deliberately not wired to any advisor, WATCHER policy or
   `strategy_recommendation.json`** — autonomous sampler selection is reserved authority (TB).
4. **The gated strategies** — confirm the chapter still describes `random`/`grid`/`evolutionary`
   correctly: gated at the CLI, **not deleted**, documented design was four Optuna samplers, and
   `GridSampler` is **unconstructible** here (7.649 × 10¹⁰ points ≈ 7.2 TiB at construction).
5. **Absorb Chapter 2's F-4** into the C-2 entry: `offset` drives **both** the host residue slice
   and the device pre-advance from one payload scalar, coherent only at `skip=0`. **Beta ruled
   this settles C-2 as an observed inconsistency, NOT the repair** — no single `offset*(skip+1)`
   multiplier exists for variable skip, and it belongs inside the future hybrid input-semantics
   design, **not a standalone arithmetic patch.**

## 2. Chapter 2 — settle §6.2 and close

**The one open item.** §6.2 claims *"39 occurrences of the lane test across the live registry."*
The corrections session could not reproduce 39: **31** by strict-format pattern on each of the
three lanes, **43** counting by residue index (`residues[i] 30 + residues[draw_idx] 13`). It
declined to assert 39 as verified and asked for a one-line settlement.

**Settle it:** count it yourself, state the method used, and give the number. If the three
plausible counts measure genuinely different things, say which one §6.2 should carry and why.
**A number in a chapter must be reproducible by the method the chapter names.**

Then close.

## 3. The closure statement — what "closed" must mean

Add to each chapter, as a short final section:

- **Verified against** — commit and date.
- **What is verified** — which sections were checked against live source this pass.
- **What remains open, and where it is tracked** — do not silently drop open items. Chapter 1
  carries P1/P2 residue and the four behavioural tickets; Chapter 2 carries its §12.1 inherited
  items and the F-4/F-5 dispositions.
- **What the chapter is NOT** — Chapter 1 is not an operator runbook for the gated strategies;
  Chapter 2 documents the sieve as built, with **hybrid worker semantics covered by the Phase-6
  transfer gate, not by a four-phase Wall-A consumer run.**

**"Closed" means verified-and-bounded, not finished.** A closure statement that hides an open
item is worse than no closure statement.

## 4. Out of scope

- **No code, tests, config or manifests.** Not `window_optimizer.py`, not
  `window_optimizer_bayesian.py`, not `execution_set.py`.
- Do not repair F-4, F-5, the dead dimensions, or anything in §2.7 of the skill.
- Do not touch other chapters, delivered Beta submissions, or the historical changelogs — the
  corrections session correctly left those alone; *retroactively editing a sent record falsifies
  the audit trail.*
- Do not edit `~/.claude/skills/…` — outside the repo. If the skill needs anything, report it.
- Do not propose removing anything (skill §0.4).

## 5. Verification-integrity controls (VIR-1…6)

- **execution proof** — every anchor confirmed against HEAD this session; the bounds snapshot
  machine-regenerated, not transcribed.
- **clean control (VIR-2)** — state which sections you verified **correct and unchanged.** A
  closure pass that reports only edits gives no evidence the rest was checked, and "closed" would
  then mean nothing.
- **fault-injection control** — n/a for documentation; **say so** rather than omitting it. If a
  chapter edit touches a file under an executable gate, run that gate (the corrections session
  did exactly this and got 12/12 including the M6 mutant).
- **completion sentinel** — explicit `PASS | FAIL | UNAVAILABLE | INCOMPLETE` **per chapter.**
- **unavailable-observer** — anything unverifiable is `UNAVAILABLE`, not assumed correct.
- **audit claim scope** — list every file changed.

## 6. Report

Per chapter: anchors corrected, sections verified unchanged, the closure statement text, and the
sentinel. For Chapter 2: the §6.2 count, the method, and why that method is the right one.
Anything that must stay open, and where it is tracked. Then STOP. **Do not commit.**
