# How to read the first walk-forward table (DEVELOPMENT, run H)

*One page for the owner and the lead. Written 2026-11-23 before any run. The commands are in `runH_commands/`. Every number in the table is DEVELOPMENT evidence: it is never a counted test and never a verdict on TFM.*

**What runs first** (lead answer Q2):
- **Smoke:** block 0 alone, 2006-08-20 to 2007-06-12, 594 draws that nobody has graded before. Its job is to measure cost.
- **Then every 4th block:** blocks 0, 4, 8, 12, 16 and 20. Then the rest.
- **The four variants** (`00_base`, `01_sieve_on`, `02_target_hits`, `03_rankers`) each write their own table.

**What one row is.** One block, one arm, one pool size, one session view: midday, evening, or both pooled. In each block:
- the model was trained only on draws before the block: 1,807 label draws, with features from everything earlier;
- it then voted on every draw of the block;
- and the grader counted how often the drawn number was in the pool.

**The arms:**

| Arm id starts with | What it is |
|---|---|
| `A1` | the learned arm: the family the loop chose for this block |
| `A0` | the unlearned arm: survivors ordered by their sieve match counts only, with no learning. This is the comparison at equal pool size |
| `F<family>` | every family on its own, at pool 300. These rows are what the loop uses to choose the next block's model |
| `R<objective>` | the ranker arms (variant 03 only), each on its own |

**The hit columns, read left to right:**
- `graded`: draws in that view, about 297 per session or 594 pooled.
- `hits`: draws whose number was in the pool.
- `expected`: what uniform guessing would give with the same pool sizes (each draw's distinct pool size / 1000, added up).
- `null sd`: the usual spread of that count by chance.
- `p`: the one-sided chance probability, **unadjusted**.
- `lift`: hits / expected.

**The scale of chance on one block** (about 297 draws per session, approximate distinct pool sizes):

| Pool | Expected hits | Chance spread (sd) | Roughly needed to stand out (p ≈ 0.05) |
|---|---|---|---|
| 20 | 5.9 | 2.4 | 10 |
| 100 | 29 | 5.1 | 37 |
| 300 | 82 | 7.7 | 95 |

So one block can only show a large effect. A real lift of 1.2× is invisible on one block. It needs many blocks, and even then the rows are **not independent**: the same survivors vote in every block, the pools are nested (20 ⊂ 100 ⊂ 300), and "pooled" contains the midday and evening rows. The table therefore gives sums across blocks as plain arithmetic, with no combined p-value.

**The rank columns** score every draw, not just the hits.
- They show where the drawn number sat in the vote: 1 is the top, 1000 the bottom. Lower is better, and pure chance averages 500.5.
- Until the full-vote ordering is pinned, rank is taken from the pool-300 vote. Every number outside that pool shares one middle rank of about 640.
- So for now the rank mostly repeats the pool-300 hit count. With the full ordering on, it uses all 1,000 values. `p lower (MC)` is the chance probability of a mean rank this low.

**How the loop shows up.** The `winner_rule` field (in `table.json` and in each block's `WINNER.json`; `TABLE.md` shows the chosen family) says how the model was picked:
- **`cv`:** the first block of a run has nothing before it, so it falls back to today's rule, the best cross-validation score.
- **`prev_block_rank`:** every later block takes the family with the best mean rank on the previous block of the same run. In the every-4th run, "previous" means four blocks earlier.
- Each block's `WINNER.json` also records the cross-validation ranking, so you can read off what today's rule would have picked and compare (lead Q5).

**Comparing variants.** Hold the same block, arm role, pool and session fixed, and compare `hits` and `mean rank` across the four tables. A difference on one block is mostly noise at these sizes (the table above). A variant counts as interesting only when it is ahead in most blocks. Even then it is a development lead for a later untouched period (draws after 2026-02-26, or a new generation). It is not a finding.

**Blocks already looked at.** Block 20 (2022-11 to 2023-09) overlaps the run H label and grade blocks that were graded today. Its rows carry `previously_inspected_overlaps`; read them as re-looks. Blocks 0 to 16 had never been graded.

**What the table cannot say.** It cannot say whether TFM works, whether a seed is "the right one", or that anything is chance. A null row is a measured null for that block and setting only, and it closes no lead.
