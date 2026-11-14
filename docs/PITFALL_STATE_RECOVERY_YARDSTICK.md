# PITFALL: JUDGING TFM BY A STATE-RECOVERY YARDSTICK

**Destination:** `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md`
**Written:** 2026-11-13 by Team Alpha, at Michael's direction.
**Read by:** Team Alpha (Claude), Team Beta (Codex), and any agent, before evaluating any part of TFM.

---

## 1. The rule

**TFM is judged by one thing: whether its prediction pools hit more often than `k/1000` on
draws the sieve never saw.**

It is **not** judged by whether "the true seed", "the real stream" or "the generating
candidate" is found, survives a filter, or ranks highly. That is a state-recovery standard.
TFM is a black box: it does not care what produced the draws. It mimics the output.

## 2. Why (Michael's design, in his terms)

- Seeds are vehicles. The sieves brute-force every seed against every draw and keep the
  ones that pass. Passing both filters by chance is very rare at correct thresholds.
- What passes is either the seed or something that shares a property with it. Which, and
  what property, is for the heuristic extraction and the ML to work out.
- The lottery's own design (unseen test draws, other games between draws, a machine and a
  method chosen at random each session, a power-down after every session) means state
  recovery is not available. It is not sought and no time is to be spent on it.
- The whitepaper formulas, the chapters and the fact file are **AI-written formalisations**
  of this design. Where one conflicts with Michael's stated intent, his intent governs. Ask
  him. Do not quote the formula back at him.

## 3. The pitfall

An agent reads the code, reasons about what a correct seed would do, finds that a filter
rejects it, and concludes the design is broken. Every step feels rigorous. The conclusion
rests on a standard the project rejected in November 2025.

**It has happened twice:**

| Date | Who | What | Outcome |
|---|---|---|---|
| 2026-08-08 | Claude Code | Attack-plan report: "trajectory identity", "the sieve can never succeed" | Two of three negative conclusions VOID. Its own words: "a state-recovery critique wearing a candidate-filter caveat" (`CLAUDE_CODE_REPORT_ATTACK_PLAN_BLACKBOX_REEVAL.md` §7) |
| 2026-11-13 | Team Alpha, then adopted by Team Beta | "The reverse sieve excludes the true seed, so the sieves do not work"; acceptance test "the generating candidate must pass both passes" | Withdrawn by Alpha the same day. Cost most of a working day |

**Tell-tale phrases.** If you are about to write any of these, stop:

- "the true seed" / "the real seed" / "the generating candidate" **as a pass/fail test**
- "this filter excludes the real stream"
- "cannot succeed" / "does not work" / "is broken" about a sieve or a survivor set
- "a correct seed would..." used to reach a verdict
- an acceptance test of the form "plant a seed and require it to survive"

## 4. What to do instead

1. **State measured facts as facts, with their limits, and no verdict.** Example: "A
   candidate that generated a 12-draw window scored 12/12 in the forward pass and 0 to 1 of
   12 in the reverse pass (CPU copies of the kernels, 2026-11-13, Alpha and Beta)."
2. **Turn the concern into a prediction about hit rate.** "I expect configuration B to give
   a higher pool hit rate than configuration A."
3. **Name the measurement that would settle it.** Same data, same settings, hit rate on
   held-out draws against `k/1000`.
4. **Treat every alternative as a configuration of the box,** selectable by a flag, never
   as a correction to the premise. The current algorithm stays as the baseline.
5. **Search the repo before asserting anything is untested, missing or wrong.** On
   2026-11-13 Alpha said no test put one planted seed through both passes; the repo had one.

## 5. What this rule does not excuse

Taken from the 2026-08-08 brief, §6: **do not soften findings to fit the frame.** Measured
problems stay real and are reported with full force, as facts tied to hit rate:

- the analysis window reaches 150 of 18,068 records;
- three tuned dimensions never reach their kernels;
- single-direction survivor counts on real data equal the simulated false-seed rate;
- hit rate on real draws has never been measured.

Mimicry is a claim that survivorship plus ranking beats `k/1000`. Whether it does is a
measurement. Until it is made, nobody may say TFM works and nobody may say it does not.

## 6. The doctrine trail (read these first)

| Date | Document |
|---|---|
| 2025-11-27 | `Functional_Mimicry_via_Reinforcement_v1.1.pdf` (project file): state recovery "may be infeasible"; emulate the generator's statistical surface; success is accuracy above random baselines |
| 2026-01-19 | `docs/PROPOSAL_Documentation_Paradigm_Correction_v1_2.md`: seeds are vehicles, not discovery targets; "true seed in top-K" removed as an acceptance criterion; a planted seed is for a reproducible sequence, not to check the system finds it |
| 2026-02-10 | `docs/TRIANGULATED_FUNCTIONAL_MIMICRY_VERIFIED_v1_0.md`: "Black-Box PRNG Analysis" |
| 2026-08 | `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` §5.6; `docs/PIPELINE_BEHAVIOUR_MODEL.md` §15.1: "The goal was never to reverse state. It is to extract a fingerprint." |
| 2026-08-08 | `docs/CLAUDE_CODE_INSTRUCTIONS_ATTACK_PLAN_BLACKBOX_REEVAL.md` and the matching report |

## 7. Facts recorded on 2026-11-13 (status, not verdicts)

- The reverse pass is the forward kernel run against the window in newest-first order
  (`miner/range_miner_worker.py:1201`). This is Michael's method, implemented as he
  describes it.
- A candidate that generated a window passes forward and fails reverse (Alpha C and Python
  copies; Beta CPU wrapper around the unchanged kernel bodies, constant and hybrid). Not run
  on GPU or VM101.
- For constant skip, forward matches plus reverse matches at one skip value cannot exceed
  the window size. Does not apply to the hybrid kernels in that form (Beta).
- Chapter 2 open item F-7 stays "confirmed, unresolved".
- Alpha's prediction, unmeasured: a reverse pass that keeps each draw at its position would
  score a higher hit rate. To be tested as a configuration against the current baseline.
