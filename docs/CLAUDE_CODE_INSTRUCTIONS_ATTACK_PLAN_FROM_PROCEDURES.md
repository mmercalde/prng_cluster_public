# CLAUDE CODE INSTRUCTIONS — ATTACK PLAN FROM THE OFFICIAL DRAW PROCEDURES

**Host:** VM101, repo `~/distributed_prng_analysis`.

## PREREQUISITE — the source document

The official *California State Lottery — Daily & SuperLotto Plus Draw Procedures* (MODIFIED for
Release for Solicitation, effective 2021-06-09) is **not in the repo**. It will be placed at:

```
docs/reference/CA_DAILY_SLP_DRAW_PROCEDURES_20210609.pdf
```

**If that file is absent, STOP and say so** — do not substitute recollection, a web source, or
another document. This task is worthless without the primary source.

## CONSTRAINT — NO LAUNCHING, NO IMPLEMENTATION

Pipeline runs are MICHAEL-INITIATED ONLY; Beta has not authorized any run. Do not start
`watcher_agent.py`, `window_optimizer.py`, the fleet script, any worker, or bind 5700. **Write
no production code and change no configuration.** This task produces a *proposal document* for
owner and Team Beta review. Permitted: reading (including the PDF), git history, read-only DB
reads, and writing your report.

## The task

Read the procedures as an engineer reading a specification of the thing being modelled, then
answer: **given what this document says about how draws are physically produced, what is the
best attack, and how does it differ from what TFM does today?**

This is analysis of publicly published draw results against a publicly released procedures
document. Nothing here involves access to lottery systems.

## Part A — What the document actually specifies

Extract, with section citations, everything that constrains the generator's observable output:

- Which games are drawn in which session, and in what order if stated (evening: D3, D4, F5, DD;
  midday: D3 alone).
- The draw specs (`03:00-09r`, `04:00-09r`, `05:01-39u`, `03:01-12u 03:00-09r`) — what each
  means about digits, ranges, and replacement, and therefore **how many generator outputs each
  game plausibly consumes**.
- Equipment selection: an RNG program selects primary and alternate equipment per draw
  (§II). What is *not* recorded in public results as a consequence.
- The two RNGs per machine ("A and B icons green", §V.4) — what that implies about which
  stream produced a given value.
- The pre-test draw before every official draw (§V.14-16) and the `Run Draw as Test` flag —
  how many outputs a pre-test consumes.
- Power-on / `[Shut Down]` per session (§V.2-3, §VII.8) — what the machine does at start-up.
- Anything about the vendor stack ("Welcome to [machine name], RNS on line") that identifies
  the generator family. **Do not guess a vendor or algorithm from outside the document** — if
  it does not name one, say so.
- Anomaly paths that inject extra draws (invalid draws, re-tests, malfunctions) — how often the
  document expects the stream to be perturbed.

## Part B — What that implies about observable structure

From Part A only, derive the structure a Daily 3 series should have if it comes from this
process:

1. Between two consecutive **midday** D3 values, how many generator outputs are plausibly
   consumed? Give a range and show the arithmetic.
2. Same for two consecutive **evening** D3 values, accounting for the other three games and the
   pre-test.
3. What discontinuities exist and where — per draw, per session, per equipment change.
4. Which of these are *fixed* (always the same count) versus *variable* (depend on a choice not
   visible in the data)?

State each as a testable proposition, not a conclusion.

## Part C — Where TFM's current model matches and where it does not

Cross-reference against `docs/CLAUDE_CODE_REPORT_SIEVE_CONTINUITY_MODEL.md` (cite it, do not
re-derive it). Its established findings: the kernel models **one unbroken 48-bit LCG
trajectory**, gaps as `skip` burns, with no machine identity, no A/B branch, no reseed, and no
session/date input in the draw loop; `skip_min ∈ [0,10]`, `skip_max ∈ [10,250]`,
`window_size ∈ [6,50]`, `offset ∈ [0,100]`; S112 found real data optimises at **W8** and
concluded *"short-lived regimes, not one continuous seed stream."*

For each element of Part B, say whether the current model **represents it, abstracts it, or
ignores it** — and whether that matters given that the sieve is a **candidate filter**, not a
state-recovery attack (owner's framing; the survivor is evidence that an alignment resolves,
and the ML learns from *how* it survived). Be precise about when an unmodelled physical detail
is harmless because selectivity carries the weight, versus when it makes a search
mathematically unable to succeed.

## Part D — The plan of attack

Propose **two to four distinct approaches**, ranked, each with:

- the hypothesis it tests, stated so it can be falsified;
- what it would consume (data already held? new data? which);
- the search geometry it implies (window, offset, skip, session scoping, seed domain — note
  the governed domain is `[0, 2^32)` per the Seed-Domain v1.1 ruling);
- what a positive result would look like, and what a null result would rule out;
- cost in cluster terms, in units the fleet uses (stripes, sub-stripes) rather than wall-clock —
  **there is no trustworthy throughput baseline; see the standing finding that `elapsed_s` is
  computed by workers and dropped at the ledger boundary. Do not invent a seeds/sec figure.**
- what would have to change in the current implementation, described only.

At least one approach must be executable **within the current governed constraints** (2^32
domain, existing bounds, no new production code) so there is something runnable if the others
are rejected.

## Part E — What the data cannot answer

State plainly what the published Daily 3 series can never reveal, no matter the method — e.g.
which machine or which RNG produced a given value, if the document implies that is unrecorded.
This section prevents an approach from being proposed that depends on unavailable information.

## Report

`docs/CLAUDE_CODE_REPORT_ATTACK_PLAN_FROM_PROCEDURES.md`. Cite the PDF by section for every
Part A claim, and `file:line` for every claim about the implementation. Mark clearly which
statements are **derived from the document**, which are **inference**, and which are
**speculation** — the three must not blur. Where the document is silent, say "the document does
not specify." Propose; do not implement.
