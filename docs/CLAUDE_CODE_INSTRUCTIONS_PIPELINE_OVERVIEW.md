# CLAUDE CODE INSTRUCTIONS — WHAT THE PIPELINE DOES, END TO END (READ-ONLY)

**Host:** VM101, repo `~/distributed_prng_analysis`. HEAD `8bbe79e`.

## CONSTRAINT — READ-ONLY. NO LAUNCHING.

Pipeline runs are MICHAEL-INITIATED ONLY, and Beta has NOT authorized any further Gate-12 run.
Do not start `watcher_agent.py`, `window_optimizer.py`, the fleet script, any worker, or bind
5700. No commits, no production edits, no changes to the skill file. Permitted: reading files,
git history, read-only DB reads. Write ONE file: your report.

**Search order is binding: governance trail → chapters → code.** Load the `tfm-project-facts`
skill first. `docs/PROJECT_FILE_CATALOG.md` is intent-indexed — read it before any absence claim.

## Purpose of this task

The owner needs a single authoritative account of **what this system does and how the parts
connect**, to be folded into the project skill. Prior sessions have repeatedly reasoned from
half-remembered fragments and gotten it wrong — including Alpha, twice this week. This report
becomes the reference that prevents that.

**Write it for someone who knows the code exists but does not know why.** Mechanism over
inventory: not "there are six steps" but what each one consumes, produces, decides, and hands
forward.

## The framing to check, not to assume

The owner's description, offered as the thing to verify against sources:

> "Step 1 sweeps the seed space and the sieve filters candidates — it is NOT reversing or
> recovering generator state. Those initial steps find the most likely seeds, nothing more.
> The ML then learns from them. The windows and offsets are used later in ML."

Confirm, refute, or refine each clause with evidence. In particular, resolve **whether the
alignment parameters (`window_size`, `offset`, `skip_min`/`skip_max`, `sessions`) travel
forward as ML features, or whether the ML sees only survivor-derived metrics.** Trace the
actual feature vector to its construction site and list its members with `file:line`.

## What the report must cover

1. **The one-paragraph answer:** what is this system for, in the sources' own words. Quote the
   most authoritative statement you can find.
2. **Step by step (0 through 6), and for each:** what it consumes, what it produces, what
   decision it makes, what it hands to the next step, and which file owns it. Note both
   numbering schemes where they differ (the skill documents two).
3. **The data spine.** Follow one unit of work end to end — a seed that survives the sieve —
   through every transformation until it influences a prediction. Name every artifact it
   becomes (survivor record → NPZ arrays → features → model input → prediction pool), with the
   contract or schema at each hop.
4. **What the sieve actually decides,** stated precisely: what makes a seed a survivor, what
   bidirectional means operationally, and what the survivor is taken to be evidence *of*.
   Cite the whitepaper and Chapter 2.
5. **What the ML learns from** — the feature vector, its target, and how it is validated
   (holdout? K-fold?). This is the clause the owner most needs verified.
6. **Where the outputs go** — prediction pools, and what "relevance" of a seed means downstream.
7. **The control layer:** WATCHER's role, what it evaluates and how, Chapter 13 triggers, and
   what is human-gated versus automatic.
8. **The five to ten load-bearing facts a new session must know** to avoid the errors this
   project keeps repeating — each with a one-line justification and a `file:line`.
9. **Contradictions between documented intent and implementation** that you encounter along the
   way. Report them; do not resolve or fix them.

## Discipline

- Every substantive claim carries `file:line` or a doc path. **"No evidence found" is a valid
  and preferred answer** where that is the truth.
- Distinguish (i) documented intent, (ii) what was implemented, (iii) what runs today.
- Do not repeat findings already covered in
  `CLAUDE_CODE_REPORT_STEP1_PURPOSE_LINEAGE.md` and `CLAUDE_CODE_REPORT_SIEVE_CONTINUITY_MODEL.md`
  — cite them and move on. This is the *whole pipeline*, not Step 1 again.
- Do not propose, recommend, or implement anything.

## Report

`docs/CLAUDE_CODE_REPORT_PIPELINE_OVERVIEW.md`. Length is whatever the material requires;
completeness matters more than brevity, but every paragraph must earn its place with evidence.
