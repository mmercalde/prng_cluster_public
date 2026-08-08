# CLAUDE CODE INSTRUCTIONS — IS `bidirectional_selectivity` A PER-SEED QUANTITY? (READ-ONLY)

**Host:** VM101, repo `~/distributed_prng_analysis`.

## CONSTRAINT — READ-ONLY. NO LAUNCHING.

Pipeline runs are MICHAEL-INITIATED ONLY; Beta holds gate 12 and the Phase-7 soak. Do not start
`watcher_agent.py`, `window_optimizer.py`, the fleet script, any worker, or bind 5700. No commits,
no production edits. Permitted: reading, `git log`/`log -S`/`show`, read-only NPZ/DB reads, and
writing your report. **Do not fix anything you find.**

## The claim to verify or refute

**[CODE, found by Alpha this session]** `window_optimizer_integration_final.py:1783`:

```python
'bidirectional_selectivity': len(forward_set) / max(len(reverse_set), 1),
```

and the hybrid path at `:1887`:

```python
'bidirectional_selectivity': len(forward_set_hybrid) / max(len(reverse_set_hybrid), 1),
```

**Alpha's reading:** this is the ratio of two **set cardinalities** for the whole trial — a
property of the trial, not of any individual seed. If so, **every survivor produced by one trial
carries an identical value**, and the field cannot discriminate between seeds by construction.

**Why it matters.** The field is used as a **per-seed** quantity in at least three places:
- stored per-seed in the 22-array NPZ contract;
- listed in the ML feature merge (`survivor_scorer.py:777`, and the field list at `:460`, `:790`);
- used as Step 2's quality signal, where `TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md:30-50` (S107,
  2026-02-22) reports **98.8% of 6,739 survivors at one value (1.0099)**, `sel_score = 0.0000` on
  every trial, and concludes *"`bidirectional_selectivity` cannot serve as the primary quality
  signal for this dataset."*

**If Alpha's reading is right, that conclusion is correct but its stated root cause is wrong** —
it is not that the dataset lacks signal, it is that a trial-level constant was named, stored and
consumed as a per-seed discriminator.

## What to determine

1. **Is it trial-level or per-seed?** Read the full assignment context at `:1783` and `:1887` —
   what `forward_set` / `reverse_set` are, and how the resulting dict is attached to survivors.
   Is one value computed once per trial and copied to every survivor record, or is it recomputed
   per seed? Quote the loop or comprehension that does the attaching, with `file:line`.
2. **Prove it empirically against a held artifact.** Find an NPZ on disk (the last release-grade
   generation at `/home/michael/d6_release_grade_20260729/generation_root/` is one candidate;
   locate others). For each, report: number of distinct `bidirectional_selectivity` values, the
   value counts, and whether the distinct-value count matches the number of trials or windows
   that contributed. **A single NPZ with exactly one distinct value across thousands of seeds
   proves the claim; several distinct values matching a trial count proves it equally.**
   Read-only — do not modify any NPZ.
3. **Is the S107 measurement consistent with this?** 98.8% at 1.0099 with ~81 above it, over
   6,739 survivors. Under the trial-level reading, what would produce that shape? State whether
   the numbers are consistent with an accumulated multi-trial NPZ, and if the accumulator's merge
   (`run_finalizer.py` L3 merge, strict `>`) affects it.
4. **How far does it reach?** Enumerate every consumer of the field —
   NPZ arrays, feature vector, Step 2 objective, any dashboard or report — with `file:line`. For
   each, say whether that consumer requires per-seed variance to be meaningful.
5. **Is there a governance record?** Search the trail for any document that defines what
   `bidirectional_selectivity` was *intended* to measure. **The binding search order applies:
   governance trail → chapters → code.** If the intended definition differs from
   `len(forward_set)/len(reverse_set)`, quote both. If no definition exists anywhere, say
   **"no evidence found"** — that is a finding in itself.
6. **Did S107 or any later work identify this?** `TB_RULING_REQUEST_STEP2_v4_1_OBJECTIVE.md:112`
   has a section *"Why `bidirectional_selectivity`"* — read it and report what rationale was
   given. Also check whether Beta ever **ruled** on either v4.1 or v4.2 request, or whether both
   remain open requests. State plainly which.
7. **Are there siblings?** The same file computes `intersection_ratio`,
   `survivor_overlap_ratio`, `intersection_weight` nearby. Are any of those also set-cardinality
   ratios attached per-seed? Chapter 2 **F-1** already records that `intersection_count`
   duplicates `bidirectional_count` and is **not** a defect — do not re-report that. Report only
   fields whose per-seed use is inconsistent with a trial-level computation.

## Report

`docs/CLAUDE_CODE_REPORT_SELECTIVITY_PER_SEED_AUDIT.md`. Lead with a one-line verdict:
**per-seed**, **trial-level**, or **cannot determine**. Then the empirical evidence from held
artifacts, then the consumer list, then §5/§6 governance findings. Every claim anchored with
`file:line` or query output. **"No evidence found" is preferred over inference.** If Alpha's
reading is wrong, say so plainly and show why — that outcome is as useful as confirmation.
**Propose nothing and change nothing.**
