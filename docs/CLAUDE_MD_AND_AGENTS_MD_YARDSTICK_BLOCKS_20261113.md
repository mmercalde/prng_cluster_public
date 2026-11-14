# YARDSTICK BLOCKS FOR `CLAUDE.md` AND `AGENTS.md`

**Purpose:** put the pitfall rule where both agents read it at the start of every session.
**Apply on VM101 against the live files.** Insert the block; do not rewrite either file.
The public mirror at `b10e1ef` was used to choose the insertion points, so confirm them live.
Michael commits and dual-pushes. Documents are reviewed by Beta after commit.

---

## Block A — insert into `CLAUDE.md`

**Where:** after the opening paragraph that ends "Claude is never Team Beta and never speaks
for it." and its `---` line, before `## 0. Run-as and identity (READ FIRST)`.

```markdown
## The yardstick (READ BEFORE EVALUATING ANY PART OF TFM)

**TFM is judged by one thing: whether its prediction pools hit more often than `k/1000` on
draws the sieve never saw.** It is a black box. It does not care what produced the draws.

**Never judge TFM, a sieve or a survivor set by whether "the true seed" or "the generating
candidate" is found, survives a filter or ranks highly.** That is a state-recovery standard.
The project rejected it in November 2025 and state recovery is not sought.

This mistake has been made twice (2026-08-08 and 2026-11-13) and cost a working day each
time. If you are about to write "the true seed" as a pass/fail test, "this filter excludes
the real stream", "cannot succeed" or "does not work", stop and do this instead:

1. State the measured fact, with its limits, and no verdict.
2. Turn the concern into a prediction about hit rate.
3. Name the measurement that would settle it.
4. Propose any alternative as a flag-selectable configuration. The current algorithm stays
   as the baseline.
5. Search the repo before saying anything is untested, missing or wrong.

The whitepaper formulas, the chapters and the fact file are AI-written formalisations of
Michael's design. Where one conflicts with his stated intent, his intent governs. Ask him.

This rule does not excuse softening findings. Measured problems are reported with full
force, as facts tied to hit rate. Full record: `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md`.
```

---

## Block B — insert into `AGENTS.md`

**Where:** at the end of `## 1. What this project is`, after the line "The project is
called TFM. Do not describe it as a lottery project."

```markdown
### The yardstick for every review

**TFM is judged by one thing: whether its prediction pools hit more often than `k/1000` on
draws the sieve never saw.** It is a black box and does not care what produced the draws.

**Do not review a sieve, a survivor set or a proposal against whether "the true seed" or
"the generating candidate" is found or survives a filter.** That is a state-recovery
standard. Do not set an acceptance test of the form "plant a seed and require it to
survive". On 2026-11-13 Alpha raised the reverse sieve on that standard and Beta adopted the
same acceptance test; both were withdrawn.

When a review finds something about a filter: state the measured fact with its limits and
no verdict; say what it predicts for hit rate; name the measurement that would settle it.
An alternative is a flag-selectable configuration judged by hit rate against the current
baseline. An objection framed on the state-recovery standard is THEORETICAL by definition.

This does not excuse softening findings. Full record:
`docs/PITFALL_STATE_RECOVERY_YARDSTICK.md`.
```

---

## Delivery

```
# run on ser8
scp ~/Downloads/PITFALL_STATE_RECOVERY_YARDSTICK.md michael@192.168.3.177:/home/michael/distributed_prng_analysis/docs/
scp ~/Downloads/CLAUDE_MD_AND_AGENTS_MD_YARDSTICK_BLOCKS_20261113.md michael@192.168.3.177:/home/michael/distributed_prng_analysis/docs/
```

Then a Claude Code session on VM101 inserts Block A and Block B at the points above and
shows the diff. No other line in either file changes.

The same rule also belongs in the fact file (`docs/TFM_PROJECT_FACTS_SKILL.md`, §0 and the
§9 self-check). That file lives in three places, so it is on the documentation fix list
and is not part of this delivery.
