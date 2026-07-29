# Team Beta Update — Material Context Before REV2.1

**Date:** 2026-07-28
**From:** Team Alpha (Claude)
**To:** Team Beta
**Re:** Self-play learning proposal — the picture has changed since your last review
**Reference:** `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_0.md` (full detail + `file:line` anchors)

---

## Why this note

Whatever version Team Beta last reviewed (proposal v1.0 / REV2), the design picture it
rests on is now **incomplete**. A full read of the autonomous-loop source and Chapter 14
— done *after* the proposal work — materially changes what the self-play proposal should
say. Team Alpha is flagging this **before** drafting REV2.1, so the addendum is written
against an accurate picture rather than perpetuating the original framing. No ruling is
requested yet beyond acknowledgment and three confirmations at the end.

The architecture Team Beta approved is not wrong. Its *framing* is.

---

## Correction 1 — Self-play is a discovery front-end to an ALREADY-BUILT loop

The proposal treated self-play as *the* learning system. It is not. The
grade → attribute → concentrate → reinforce machinery already exists and runs:

- **Ch13 feedback daemon** — ingests each new draw, scores coverage/hit-rate with
  lift-over-random (`evaluate_pools.py`), runs **per-survivor attribution** on
  regressions, evaluates triggers, consults the LLM advisor, validates via the
  acceptance engine, executes under a human gate.
- **WATCHER** — ~85% pipeline autonomy.
- **reinforcement_engine** — reinforces high-quality survivors (GlobalStateTracker).
- **prediction_generator** — concentrates the pool on strength; **Signal Quality Gate**
  abstains when signal is weak.

So the self-play engine is not a from-scratch learning system — it is the **missing
DISCOVERY stage** feeding a loop that already exists. This changes REV2.1's scope: the
engine must plug into the existing loop, not duplicate it.

## Correction 2 — Chapter 14 attribution IS the "which-heuristics-are-viable" mechanism

Per-survivor attribution (Chapter 14 §4, authored by Team Beta) already answers "for
THIS survivor, which heuristics drove its ranking." The discovery engine should
**consume Ch14 attribution** as its per-survivor signal, not invent one.

Corollary that refines Team Beta's own NN ruling: the NN's value is **per-survivor
gradient attribution** (`grad_x_input`, Ch14 §4.2) — a capability trees only approximate.
The NN earns its place through *attribution*, not R². The "opt-in challenger, judged on
fitness" framing undersells why it is worth reinstating.

## Correction 3 — What the proposal omitted entirely

For the record: the proposal (through REV2) contains **no reference** to Chapter 14,
per-survivor attribution, `evaluate_pools`, `reinforcement_engine`, or
`prediction_generator`. Its individual claims were largely accurate; its picture of "the
learning system" was not. (The target-field question Team Beta already caught in
Correction B stands corrected — the live target is `holdout_hits`,
`selfplay_orchestrator.py:933`.)

---

## RANGE-MINER (S172) interaction — for the record

The learning/attribution layer consumes the **22-array NPZ survivor contract**
(`[S172 Phase-5 D3.0]` frozen). S172 **Phase 6's four-path `np.array_equal` over all 22
arrays** is the guarantee that the PWC → RANGE-MINER cutover cannot starve attribution:
if the miner reproduces the PWC arrays byte-for-byte, the downstream learning layer sees
exactly what it always saw. RANGE-MINER therefore introduces **no** missing-parameter
risk to the learning layer, provided Phase 6 asserts identity over **all 22 named
arrays** (fail on any missing/renamed/reordered), not a subset.

---

## Requested from Team Beta

1. **Acknowledge the reframing** so REV2.1 is written as "discovery front-end to the
   existing loop," not "new learning system."
2. **Confirm** Phase-6 acceptance = identity over all 22 named arrays (explicit criterion).
3. **Direct** whether REV2.1 should absorb the front-end reframing inline, or whether the
   discovery-loop architecture (attribution-driven, strength-seeking, generator-confirmed)
   warrants its **own companion proposal** — Team Alpha's lean is a companion, since it is
   an architectural layer above the search-engine contracts REV2.1 covers.

Full as-built map, with anchors and an honesty log of Team Alpha's own corrected errors,
is in `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_0.md`.
