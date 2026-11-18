# Bidirectional Sieve: Mathematical Whitepaper
> **Header note (added 2026-11-17, doc fix list item B1, `docs/DOC_FIX_LIST_20261113.md:25`; written as one line so that no line number below it moves — Chapter 2, the fact file and several reports cite this file by line).** **This document is an AI-written formalisation of Michael's design. Where it conflicts with his stated intent, his intent governs** (`docs/PITFALL_STATE_RECOVERY_YARDSTICK.md:27-29`; `CLAUDE.md`, "The yardstick"). The mathematics below has **not** been rewritten; changes to it wait for Michael. **The yardstick for TFM is hit rate:** whether its prediction pools hit more often than `k/1000` on draws the sieve never saw (`docs/PITFALL_STATE_RECOVERY_YARDSTICK.md:11-12`), not whether a particular seed is found, survives or ranks highly; seeds are vehicles for heuristic extraction, not discovery targets (`docs/PROPOSAL_Documentation_Paradigm_Correction_v1_2.md` §2). **§4, §6 and §8 do not describe the implemented method.** §4's `G(s,−i)` is a backward generator index; the implementation runs the forward kernel against the window in newest-first order (`miner/range_miner_worker.py:1330`, `residues[::-1] if reverse`; `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` §3.2, §3.5, open item F-7). §6's "only the true seed survives" and §8's `P(s = s*)` are seed-recovery language, which the project does not use as a standard (`docs/PITFALL_STATE_RECOVERY_YARDSTICK.md` §1, §3). §7 and §10 also use `s*`, and §10 ends "That is why this system works"; whether TFM beats `k/1000` is a measurement that has not been made on real draws (`docs/PITFALL_STATE_RECOVERY_YARDSTICK.md:74-77`).
## 1. Problem Statement
We consider a discrete pseudo-random number generator (PRNG) producing values in \( \mathbb{Z}_{1000} \).
Given an observed sequence of draws, the objective is to identify candidate seeds whose generated
sequences are consistent with the observed data.

This document analyzes the *mathematical filtering power* of forward and reverse sieves,
and the resulting implications for machine learning and autonomy.

---

## 2. Model Assumptions

- PRNG output space: \( \{0,1,\dots,999\} \)
- Observed draws: \( D = (d_1, d_2, \dots, d_n) \)
- Candidate seeds: \( s \in S \), where \(|S| = 2^{32}\) (conceptually)
- For a random seed, each output is assumed uniform in \( \mathbb{Z}_{1000} \)

---

## 3. Forward Sieve

Define a forward sieve predicate:

\[
F(s) = \frac{1}{n} \sum_{i=1}^n \mathbf{1}[G(s,i) = d_i] \ge \tau_f
\]

where:
- \(G(s,i)\) is the PRNG output of seed \(s\) at position \(i\)
- \(\tau_f \in (0,1]\) is the forward match threshold

### Survival Probability

For a random seed:

\[
P(G(s,i) = d_i) = \frac{1}{1000}
\]

Let \(X \sim \text{Binomial}(n, 1/1000)\).
Then:

\[
P(F(s)=1) = P\left(X \ge \tau_f n\right)
\]

This probability decays exponentially in \(n\).

---

## 4. Reverse Sieve — *not the implemented method (note added 2026-11-17, fix list B1): the predicate below evaluates the generator at a negative index; the implemented reverse pass runs the forward kernel against the window newest first (`miner/range_miner_worker.py:1330`; `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` §3.2, §3.5, F-7). The independence premise stated here is about the backward construction. Kept unedited pending Michael.*

Reverse sieve applies the same criterion but on reversed indices:

\[
R(s) = \frac{1}{n} \sum_{i=1}^n \mathbf{1}[G(s,-i) = d_{n+1-i}] \ge \tau_r
\]

Under independence assumptions, forward and reverse matches are *approximately independent*
for incorrect seeds.

---

## 5. Bidirectional Sieve

A seed survives if:

\[
B(s) = F(s) \land R(s)
\]

### Combined Survival Probability

For random seeds:

\[
P(B(s)=1) \approx P(F(s)=1)^2
\]

Thus, if:

\[
P(F(s)=1) \approx e^{-c n}
\]

then:

\[
P(B(s)=1) \approx e^{-2 c n}
\]

This **squares the exponent** — a catastrophic collapse of noise.

---

## 6. Exact-Match Limit

If \(\tau_f = \tau_r = 1\):

\[
P(B(s)=1) = (1/1000)^{2n}
\]

For \(n=50\):

\[
P \approx 10^{-300}
\]

Meaning: **only the true seed survives**. *(Note added 2026-11-17, fix list B1: this is a state-recovery statement. It is not the project's standard and is not a pass/fail test for any sieve or survivor set (`docs/PITFALL_STATE_RECOVERY_YARDSTICK.md` §1, §3). It also rests on §4's backward construction, which is not the implemented method. Kept unedited pending Michael.)*

---

## 7. Why Loose Thresholds Are Required for ML

Exact sieves eliminate *all variance*:

- Survivors = \( \{s^*\} \)
- No ranking
- No gradients
- No learning signal

Looser thresholds produce a *manifold* of near-consistent seeds:

\[
\mathcal{S}_{\text{near}} = \{ s : d(s,s^*) \le \epsilon \}
\]

These seeds share structured deviations that ML can learn to rank.

---

## 8. ML After Sieving Is Statistically Sound

Conditioned on survival:

\[
P(s = s^* \mid B(s)=1) \gg P(s \ne s^*)
\]

Thus ML operates in a **high signal-to-noise posterior**, not raw PRNG space. *(Note added 2026-11-17, fix list B1: `P(s = s* | B(s)=1)` measures the system by recovery of one seed `s*`, which is seed-recovery language. The project measures it by prediction-pool hit rate against `k/1000` on unseen draws; a survivor is a vehicle for heuristic extraction (`docs/PITFALL_STATE_RECOVERY_YARDSTICK.md` §1, §2; `docs/PROPOSAL_Documentation_Paradigm_Correction_v1_2.md` §2). Kept unedited pending Michael.)*

---

## 9. Autonomy Implications

- Sieves eliminate entropy
- ML ranks residual structure
- Feedback loop tightens thresholds over time
- Autonomy adjusts *parameters*, never structure

This is **constraint satisfaction → statistical refinement**, not brute-force prediction.

---

## 10. Conclusion

Bidirectional sieving provides exponential noise suppression.
Loose thresholds are not a weakness — they are a mathematical necessity
to expose a learnable structure.

ML does not guess.
It refines a space already reduced from \(2^{32}\) to \(10^4\).

That is why this system works.
