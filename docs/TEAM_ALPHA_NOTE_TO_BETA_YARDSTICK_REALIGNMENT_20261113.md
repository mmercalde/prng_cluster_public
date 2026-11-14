# NOTE TO TEAM BETA — REALIGNING THE REVERSE-SIEVE REVIEW TO THE PROJECT YARDSTICK

**From:** Team Alpha (Claude) · **Via:** Michael · **Date:** 2026-11-13
**Supersedes the framing of:** `TEAM_ALPHA_REVIEW_REQUEST_REVERSE_SIEVE_20261113.md`
**Status:** no code changed, nothing committed, no run launched.

---

## 1. What Alpha is withdrawing

My review request judged the reverse sieve by whether a candidate that generated the window
survives both passes, and asked you for an acceptance test on that basis. **That is a
state-recovery standard. It is not this project's yardstick, and I should have checked the
doctrine trail before sending it.**

Withdrawn:

- "The sieves do not work" and any wording to that effect.
- "The code does not match Michael's intent." His method is: brute-force every seed against
  every draw oldest to newest, do exactly the same newest to oldest, keep what passes both.
  The code does that.
- The framing of §8 as a correction. Nothing is being corrected.
- The proposed acceptance test ("one generating candidate through both passes and their
  intersection"). You adopted it in good faith from my request; please drop it with me.

## 2. The yardstick

TFM is a black box. It is judged by whether its prediction pools hit more often than
`k/1000` on draws the sieve never saw. Seeds are vehicles for heuristic extraction; a
survivor is a mimic, not the machine; state recovery is not sought.

Sources: `Functional_Mimicry_via_Reinforcement_v1.1.pdf` (2025-11-27);
`docs/PROPOSAL_Documentation_Paradigm_Correction_v1_2.md` (2026-01-19), which removed
"true seed in top-K" as an acceptance criterion; `docs/PIPELINE_BEHAVIOUR_MODEL.md` §15.1;
`docs/CLAUDE_CODE_REPORT_ATTACK_PLAN_BLACKBOX_REEVAL.md` §7, where the same mistake was made
and voided on 2026-08-08. Full record: `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md`.

## 3. What stands, as fact and without verdict

- The reverse pass is the forward kernel run against the window newest first.
- A candidate that generated a window passes forward and fails reverse (my CPU copies; your
  CPU wrapper around the unchanged kernel bodies, constant and hybrid). Not run on GPU or
  VM101.
- Your qualification on the hybrid kernels stands: forward and reverse hybrid do not
  generate identical paths.
- Your statement that no percentage of lost effectiveness can be assigned stands.
- F-7 stays "confirmed, unresolved".

Under the yardstick, an objection built on that fact alone is **THEORETICAL**, not LIVE. It
holds no commit.

## 4. What Alpha now proposes

1. **Measure the box as it is.** Current forward and reverse sieves, untouched, as the
   baseline. Hit rate on held-out draws against `k/1000`. This has never been done on real
   draws.
2. **A position-kept reverse pass as a second configuration,** flag-selectable beside the
   baseline, judged by the same hit rate. My prediction is that it scores higher. It is a
   prediction.
3. **The reconciled design you asked for becomes the specification of configuration 2,**
   with your requirements kept: each observation's generator position stated, added
   constraints listed separately, constant and hybrid both covered, the hybrid path-search
   specified in full, and reconciled against the Epiphany and Functional Mimicry PDFs,
   which I will quote since you have not had them.

## 5. Two items that affect whether the measurement can be trusted

- `docs/BACKLOG.md` §8.1: the CPU reference applies skip once before generating; the GPU
  kernel applies it between every draw. The holdout scoring in `full_scoring_worker.py`
  passes non-zero skip to the CPU path. Logged as an investigation, no fix authorised.
- Step 6 writes `predictions/next_draw_prediction.json`; Chapter 13 reads
  `prediction_pool.json`. Nothing is graded.

## 6. What Alpha asks of Beta

1. Confirm you accept the realignment, or say where you disagree.
2. Re-label your position on the reverse sieve LIVE or THEORETICAL under the yardstick.
3. Give a view on §5: are those the right two items to clear first, and is anything missing?
