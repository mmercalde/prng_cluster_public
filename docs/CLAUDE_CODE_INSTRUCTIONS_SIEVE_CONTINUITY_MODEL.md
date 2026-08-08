# CLAUDE CODE INSTRUCTIONS — WHAT CONTINUITY DOES THE SIEVE ASSUME? (READ-ONLY)

**Host:** VM101, repo `~/distributed_prng_analysis`.

## CONSTRAINT — READ-ONLY. NO LAUNCHING.

Pipeline runs are MICHAEL-INITIATED ONLY, and Beta has NOT authorized any further Gate-12 run.
Do not start `watcher_agent.py`, `window_optimizer.py`, the fleet script, any worker, or bind
5700. No commits, no production edits. Permitted: reading files, git history, read-only DB
reads, writing your report only.

**Search order is binding: governance trail → chapters → code.**

## Why this is being asked

The owner is re-establishing Step 1's design intent before any seed-geometry decision. Alpha's
prior report (`docs/CLAUDE_CODE_REPORT_STEP1_PURPOSE_LINEAGE.md`) established that Step 1 is a
**meta-tool that finds the best window parameters** — the deliverable is an alignment
(`optimal_window_config.json`), and survivors are evidence that an alignment resolves.

The open question is what physical model that search rests on. **Alpha has already been wrong
here once** — it asserted "evening is one machine, midday is another," which the CA *Daily &
SuperLotto Plus Draw Procedures* (effective 2021-06-09) contradicts. Do not inherit Alpha's
framing.

## What the official draw procedures actually say

These are facts from the source document, given so the code can be checked against them — not
conclusions:

- **The machine is selected per draw, at random, from a pool.** §II: *"A random number
  generation (RNG) program is used to select the primary and alternate draw equipment which
  will be used for the draw."* The observable results do **not** record which machine was used.
- **Each automatic draw machine exposes TWO RNGs.** §V.4: *"Verify both A and B RNG icons
  displayed are green."*
- **Machines are powered down after each draw session** (§VII.8 `[Shut Down]`) and powered on
  fresh for the next (§V.2-3).
- **A pre-test draw is run before every official draw** (§V.14) — i.e. at least one additional
  generator output is consumed before the live draw, on the same session.
- **Evening draws four games in one session**: Daily 3, Daily 4, Fantasy 5, Daily Derby, with
  specs `03:00-09r`, `04:00-09r`, `05:01-39u`, `03:01-12u 03:00-09r`. **Midday draws Daily 3
  alone.**
- Draw order within the evening session is not stated in the procedures.

## The questions

**Q1 — What continuity does the bidirectional sieve assume between two consecutive observed
Daily 3 values?** Does it model them as consecutive outputs of one continuous generator stream,
as outputs separated by an unknown skip (`skip_min`/`skip_max`), or something else? Quote the
model from `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md`, the whitepaper, and the kernel with
`file:line`.

**Q2 — Does any part of the system model a per-draw machine change, an A/B RNG choice, or a
per-session power cycle / reseed?** Search the governance trail and code for any concept of
machine identity, reseeding, session boundary, or stream discontinuity. If none exists, that is
a legitimate answer — say **"no evidence found"** — but search the trail first, because the
project's rule is that an unexplained component means the explanation has not been found yet.

**Q3 — What do `skip_min` / `skip_max` actually represent, in the model's own words?** Bounds
are `skip_min ∈ [0,10]`, `skip_max ∈ [10,250]`. Establish whether a "skip" is: unobserved draws
from other games in the same session, generator outputs consumed internally, an abstraction over
both, or something else. Also state whether skip is applied **uniformly** across a window or
**per-gap** (the skill refers to "variable skip"), and quote the source.

**Q4 — Can a window legitimately span a session boundary?** `window_size ∈ [6,50]` and the
`sessions` filter is an Optuna dimension. When `sessions` selects both midday and evening, a
window of 24 draws spans roughly 12 calendar days and therefore ~24 separate machine power-ups.
Does any document or code address whether one seed is expected to explain draws across those
boundaries — or is the session filter itself the mechanism that avoids the question? Report what
the sources say, not what would be reasonable.

**Q5 — Is the model documented anywhere as an explicit physical hypothesis?** i.e. a statement
of the form "we assume the draw sequence is produced by X, seeded at Y, advancing by Z." If such
a statement exists, quote it in full. If the model exists only implicitly in kernel behaviour,
say that plainly — it matters, because the owner is deciding whether the current bounds are
principled or scaffolding.

## Report

`docs/CLAUDE_CODE_REPORT_SIEVE_CONTINUITY_MODEL.md`. For each question: the answer, verbatim
quotes with `file:line`, and an explicit **"no evidence found"** where that is the truth.
Distinguish documented intent from implemented behaviour wherever they differ. Where the
official draw procedures describe a real-world condition the model does not represent, note the
gap as an **observation**, not a defect — whether it matters is the owner's and Beta's call, not
yours. Do not propose or implement anything.
