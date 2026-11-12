# CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md — REV1

**Bring Chapters 3 to 14 in line with the code, using one subagent per chapter.**

**Documentation only.** No code, tests, config, manifests, skill file or `CLAUDE.md`. No runs. No
rig access. You do **NOT** commit or push. STOP at the gate in §8.

**Host:** VM 101 (`192.168.3.177`) as `michael`, in `/home/michael/distributed_prng_analysis`.
**Base:** current `main` on VM 101. Record `git rev-parse HEAD` first and use it everywhere.
**Authorized by:** Michael, 2026-11-11.

---

## 0. Owner rulings of 2026-11-11 that override what the repo's documents say

Read these before the skill or `CLAUDE.md`, because both predate them.

1. **Authority.** Michael is the final authority. Team Alpha (Claude) is lead designer and coder.
   **Team Beta is a reviewer, not an approver.** Where a document says a Beta ruling is "binding"
   or "final", read it as a recorded review position that the owner may keep or lift.
2. **Review timing.** Documentation work is committed first and reviewed after. Only coordinator,
   worker and kernel code, and runs that count, are reviewed before commit. This brief is
   documentation, so nothing here waits on Beta.
3. **Disagreements** between Alpha and Beta go to Michael with both positions.
4. **Subagent model.** Run the per-chapter subagents on **Opus**. If this session cannot select a
   subagent model, say so in the report and continue.

**Known-stale statements in the fact file (`docs/TFM_PROJECT_FACTS_SKILL.md` v27.1). Do not
repeat them:**

- "Implementation Brief I does not exist yet." It exists, is implemented and committed. Final
  acceptance is pending a completed production-shape run.
- "`G-PROD-SHAPE` has never run." It ran on 2026-08-22 and failed
  (`docs/S172_BRIEF_I_PRODUCTION_SHAPE_CLASSIFICATION_REPORT.md`).
- `CLAUDE.md` §6 shows Phase 3 as next. Phases 0 to 6 are done and Gate 12 passed.

## 1. The framing every chapter must carry

**TFM is functional mimicry of PRNG surface output. It is NOT seed recovery and NOT state
reconstruction.** Readers keep concluding the opposite, because the older chapters open in
seed-recovery language.

Every corrected chapter gets, directly under its title, this block unchanged:

> **What TFM is.** TFM filters candidate generators against observed draws in two directions,
> keeps a population of survivors on purpose, ranks them, and lets the top candidates vote their
> next value into a prediction pool. A survivor is a scored candidate, not a verdict. The
> generator family is a substrate for mimicry, not a claim about the machine. The target is a
> pool hit rate. TFM does not recover a seed or reconstruct state, and an objection that assumes
> it does is out of scope.

"Identify candidate seeds" stays where it is accurate: the sieve does filter seeds. What must go
is any sentence implying the goal is the one true seed.

## 2. Inputs

Three reports produced on 2026-11-11 must be in `docs/` before you start. If any is missing,
stop and say which.

```
docs/CHAPTER_13_AUDIT.md
docs/CHAPTER_14_AUDIT.md
docs/TODO_CONSOLIDATED.md
```

**They were made from the public mirror at `a15b524`, with no access to host state, gitignored
files or git history.** Every anchor in them is a claim with an expiry date. Re-verify each
against the live tree before you rely on it, and resolve their `[UNVERIFIED]` items where the
live host lets you.

Existing audit for Chapter 3: `docs/CHAPTER_3_ALIGNMENT_AUDIT.md` (17 of 55 claims accurate).

Accepted model for a finished chapter: `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md`. Match its shape:
status header, how the chapter was produced, anchored claims, an open-items section, a VIR-1…6
declaration, a closure statement.

## 3. The work, per chapter

One subagent per chapter. Each subagent does three things, in this order.

### 3.1 Audit (skip for 3, 13 and 14, which have one; re-verify it instead)

For every substantive claim in the chapter, decide whether the code at HEAD does it. Label each
disagreement with exactly one of:

- **BROKEN** — the code does not do it and no governance document says that was intended.
- **SWITCHED-OFF-BY-RULING** — it exists or was designed, and a Beta ruling or approved proposal
  holds it back. Cite the ruling.
- **DESCRIPTION-ONLY** — a wording difference, not a code path.

Write `docs/CHAPTER_<N>_AUDIT.md` in the same eight-section layout as `CHAPTER_13_AUDIT.md`.

### 3.2 Correct the chapter in place

- Make every statement match what the code does at HEAD, with a `file:line` or symbol anchor read
  in this session. Prefer symbol names where line numbers are likely to move.
- Add the §1 framing block.
- Add a section **"Known gaps at <HEAD>"** with three lists: BROKEN, SWITCHED-OFF-BY-RULING,
  DESCRIPTION-ONLY. For each switched-off item, state what lifting it would enable. Do not
  recommend for or against.
- Where RANGE-MINER, the 22-array NPZ contract or the window-anchor change altered what the
  chapter describes, say so. **Chapters 9 and 12 do not mention RANGE-MINER at all and need a
  full section each.**
- Add the status header, VIR-1…6 declaration and closure statement, as in Chapter 2.

### 3.3 Preserve history

- **Never shrink a chapter by overwrite.** Chapter 2 was once destroyed by a 34-line file copied
  over 743 lines. Edit in place.
- Do not delete a stale section. Mark it **SUPERSEDED** with the date and a pointer to what
  replaced it, and leave the text.
- Corrections leave a visible trail. Do not silently rewrite.
- A chapter that ends more than 10% shorter than it started needs each removal listed in its
  closure statement.

### 3.4 What a chapter subagent must not do

- Fix code. A broken code path is **recorded**, in the chapter and in the audit. The repair is a
  separate brief.
- Propose removing or simplifying any component.
- Raise a seed-recovery objection (§1).
- Claim something is absent without listing the surfaces searched: tracked repo, gitignored
  files, git history, host state. `docs/` and the governance trail are mandatory surfaces.
- Report a known, governed item as a new finding. Search the governance trail first
  (`docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/TEAM_ALPHA_*`), then chapters, then code.

## 4. Order and batching

Run in batches of four so the token cost stays visible. Finish and check a batch before starting
the next.

| Batch | Chapters | Why this order |
|---|---|---|
| 1 | **9, 12, 13, 14** | 9 (cluster) and 12 (WATCHER) are the two RANGE-MINER changed. 13 and 14 already have audits |
| 2 | 3, 4, 5, 6 | Step 2.5 through Step 5. Chapter 3 already has an audit |
| 3 | 7, 8, 10, 11 | Step 6, the registry, the agent framework, feature importance |

Chapter files:

```
docs/CHAPTER_3_SCORER_META_OPTIMIZER.md            docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md
docs/CHAPTER_4_FULL_SCORING.md                     docs/CHAPTER_10_AUTONOMOUS_AGENT_FRAMEWORK_v3.md
docs/CHAPTER_5_ML_ARCHITECTURE_OPTIMIZER_v2.md     docs/CHAPTER_11_FEATURE_IMPORTANCE_VISUALIZATION.md
docs/CHAPTER_6_ANTI_OVERFIT_TRAINING.md            docs/CHAPTER_12_WATCHER_AGENT.md
docs/CHAPTER_7_PREDICTION_GENERATOR.md             docs/CHAPTER_13_LIVE_FEEDBACK_LOOP.md
docs/CHAPTER_8_PRNG_REGISTRY.md                    docs/CHAPTER_14_TRAINING_DIAGNOSTICS.md
```

**Chapter numbers are not step numbers.** WATCHER's step 2 is the scorer meta-optimizer; the
sieve runs inside Step 1. Every chapter states which numbering scheme it uses
(`docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1).

## 5. Things the live host can settle that the 2026-11-11 audits could not

Give these to the relevant subagent as explicit questions.

- **Chapter 13.** Is there a host-side bridge that writes `prediction_pool.json` or raises
  `new_draw.flag`? Check systemd units, cron and untracked files. The audit found no writer in
  the tracked tree while Step 6 writes `predictions/next_draw_prediction.json`.
- **Chapter 14.** Does a real `training_diagnostics` artifact on VM 101 carry `best_round` or
  `best_val_round`? The health check reads the first (`training_health_check.py`), the producer
  writes the second (`training_diagnostics.py`). Is `neural_net` currently in the skip registry?
- **Chapter 12.** `agent_manifests/window_optimizer.json` still defaults `use_range_miner` to
  `false`. Record it; do not change it.
- **All chapters.** `agent_manifests/definitions.json` is gitignored and was invisible to the
  audits. Read it on the host.

Host checks are **read-only**. Do not start, stop or edit a service.

## 6. Main-session duties

1. Brief each subagent with §0, §1, §3 and its chapter's lines from §5.
2. When a batch returns, check every chapter before accepting it:
   - `git diff --stat` shows only the chapter and its audit file;
   - line count before and after, and the §3.3 rule;
   - every section heading that existed before still exists or is marked SUPERSEDED;
   - spot-check at least five anchors per chapter by opening the file at that line;
   - the §1 block is present and unchanged.
3. Reject and re-run a chapter that fails any check. Do not repair it yourself silently.
4. Keep a running total of BROKEN items across chapters. Two chapters reporting the same break
   count once.

## 7. Deliverables

```
docs/CHAPTER_<N>_AUDIT.md                         for 4, 5, 6, 7, 8, 9, 10, 11, 12 (new)
docs/CHAPTER_3_ALIGNMENT_AUDIT.md                 re-verified, addendum appended
docs/CHAPTER_13_AUDIT.md, CHAPTER_14_AUDIT.md     re-verified against the live host, addendum appended
the twelve chapter files                          corrected in place
docs/CHAPTER_CORRECTIONS_REPORT_20261111.md       the summary below
docs/SESSION_CHANGELOG_20261111_CHAPTER_CORRECTIONS.md
```

`CHAPTER_CORRECTIONS_REPORT_20261111.md` contains:

1. Per chapter: claims checked, accurate, BROKEN, SWITCHED-OFF, DESCRIPTION-ONLY, UNVERIFIED;
   lines before and after.
2. **Repair list:** every BROKEN code item, deduplicated, with anchor and a one-line statement of
   what a repair must do. No code. This becomes the input to the repair briefs.
3. **Owner decision list:** every SWITCHED-OFF-BY-RULING item, with the ruling and what lifting
   it would enable.
4. Updates needed elsewhere and not made here: `docs/BACKLOG.md` §1,
   `docs/PROJECT_FILE_CATALOG.md` §7 gap 6, the skill, `CLAUDE.md`.
5. The §5 host questions and their answers.
6. Tokens or cost used per batch, if the session can report it.

## 8. The gate

Stop here. Do not commit.

- Print `git status --porcelain` and an explicit `git add` line naming every file, built from the
  report's file list, not from memory. Never `git add -A`.
- State the completion sentinel for the whole pass: `PASS | FAIL | UNAVAILABLE | INCOMPLETE`. A
  pass with unfinished chapters is `INCOMPLETE`.
- Michael commits and dual-pushes: `git push origin main && git push public main`.

## Verification-integrity controls (VIR-1…6)

- **execution proof:** each chapter's audit lists the files it opened; the main session's
  five-anchor spot-check is recorded per chapter.
- **clean control:** Chapter 2, already closed, is not touched. Its content gate
  `tests/test_chapter2_content_gate.py` must still pass after the session; run it once at the end.
- **fault-injection control:** not applicable to a documentation pass. Say so; do not invent one.
- **completion sentinel:** per chapter and for the pass, as in §8.
- **unavailable-observer behavior:** a host surface that cannot be read is `UNAVAILABLE`, never
  clean.
- **audit claim scope:** what each chapter says against what the code at HEAD does. Not whether
  the design is right.
- **searched surfaces / unavailable surfaces:** listed in every audit.
- **governance trail searched; chapters searched:** listed in every audit.
