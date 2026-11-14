# TFM DOCUMENTATION FIX LIST — 2026-11-13

**From:** Team Alpha · **For:** Michael
**Base:** public mirror at `b10e1ef`. Line numbers must be confirmed on VM101 before editing.
**Rule:** documentation only unless marked CODE. Edit in place; never shrink a document by
overwrite. Beta reviews documents after commit.

---

## A. Wording that puts every new session in the wrong frame (do first)

| # | Where | Problem | Fix |
|---|---|---|---|
| A1 | `CLAUDE.md`, `AGENTS.md` | No statement of the yardstick or the pitfall | Insert Block A and Block B from `CLAUDE_MD_AND_AGENTS_MD_YARDSTICK_BLOCKS_20261113.md` |
| A2 | `docs/` | No record of the pitfall | Add `PITFALL_STATE_RECOVERY_YARDSTICK.md` |
| A3 | `README.md:4`, `docs/README.md:11` | Open with "Reverse-engineer PRNG behavior" | Replace with the functional-mimicry statement. Already flagged as D16 in `PIPELINE_BEHAVIOUR_MODEL.md` §15.1 |
| A4 | `docs/proposals/README.md:1` | Titled "Distributed PRNG Analysis & Seed Reconstruction System" | Retitle |
| A5 | The claude.ai Project description and instructions | Same "Seed Reconstruction" and "Reverse-engineer PRNG behavior" text; this is what Claude reads first in chat | Michael edits in claude.ai; Alpha can draft the replacement text |
| A6 | `docs/TFM_PROJECT_FACTS_SKILL.md` (fact file) | §0.1 says "find candidate seeds whose generated sequences are consistent with what was drawn"; no yardstick rule; §9 self-check has no item for it. Also predates the 2026-11-11 rulings | Add the yardstick to §0 and §9; apply the 2026-11-11 corrections. Three copies must be updated together: tracked file, `~/.claude/skills/`, Settings upload |

## B. AI-written formalisations that do not match Michael's design

| # | Where | Problem | Fix |
|---|---|---|---|
| B1 | `docs/BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md` §4, §6, §8 | §4 defines reverse with a backward generator index, which the code does not do. §6 "only the true seed survives" and §8 `P(s = s*)` are seed-recovery language | Add a header note: this is an AI formalisation; Michael's intent governs; the yardstick is hit rate. Mark §4/§6/§8 as not describing the implemented method. Do not rewrite the maths without Michael |
| B2 | `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` §3.3, §3.5, F-7 | §3.3 table implies reverse re-checks the same stream from the other end. F-7 names the mismatch but records no measurement | Record the 2026-11-13 fact (generating candidate passes forward, fails reverse, CPU-level) as status, not verdict. F-7 stays "confirmed, unresolved". Cross-reference the pitfall document |
| B3 | `docs/KNOWN_ANSWER_VALIDATION_INVENTORY.md:97` | Says the PA harness "proves the bidirectional sieve method recovers a planted seed". The S143 changelog records forward recovery only | Correct to what S143 records; mark the bidirectional claim unverified |
| B4 | `docs/PROPOSAL_Documentation_Paradigm_Correction_v1_2.md` header | Still says "PENDING APPROVAL" although it is the cited origin of the doctrine | Mark adopted, with the date Michael gives |

## C. Descriptions that contradict the code (found in the fact file and audits)

| # | Where | Problem | Fix |
|---|---|---|---|
| C1 | `prng_registry.py:3911`, `:3917` | Registry descriptions say "fixed skip backward validation" for kernels that step forward | Correct the description strings. CODE file: Beta reviews before commit |
| C2 | `config_manifests/feature_registry.json` vs `parameter_registry.json:160,166` | Disagree on what `skip_min`/`skip_max` mean (found during vs for search) | Settle and make both say the same |
| C3 | `config_manifests/feature_registry.json` | Files `bidirectional_selectivity` under per-seed features; it is trial-level | Move or relabel |
| C4 | `parameter_registry.json:38-43`; `agent_manifests/window_optimizer.json` offset entry | Offset descriptions match no loader ("time offset from current draw position") | Correct once Brief I/II wording for anchor and phase is final |
| C5 | `docs/CHAPTER_6_ANTI_OVERFIT_TRAINING.md` | Says the training target is `holdout_hits`; it is `holdout_quality` | Part of chapter-corrections batch 2 |
| C6 | `docs/CHAPTER_13_LIVE_FEEDBACK_LOOP.md` | Describes a loop that reads `prediction_pool.json`; Step 6 writes `predictions/next_draw_prediction.json` | Document the gap now; the repair is CODE and needs its own brief |

## D. Housekeeping already in flight

| # | Item | State |
|---|---|---|
| D1 | Chapter corrections, batches 2 (Ch. 3-6) and 3 (Ch. 7, 8, 10, 11) | Batch 1 done 2026-11-11 |
| D2 | `CLAUDE.md` §6 phase plan, the behaviour model and the file catalog | Out of date against the 2026-11-11 handoff |
| D3 | `docs/PROJECT_FILE_CATALOG.md` | Add today's documents |
| D4 | `SESSION_CHANGELOG_20261113_YARDSTICK.md` | To write |
| D5 | Alpha's "TFM Effectiveness Assessment" report | Corrected 2026-11-13 |
| D6 | `TEAM_ALPHA_REVIEW_REQUEST_REVERSE_SIEVE_20261113.md` | Superseded by the realignment note; keep both |

## E. Not documentation, but it decides whether the hit-rate measurement can be trusted

| # | Item | Where |
|---|---|---|
| E1 | CPU reference applies skip once; GPU kernel applies it between every draw. Holdout scoring uses the CPU path with non-zero skip | `docs/BACKLOG.md` §8.1; `full_scoring_worker.py:305`; `survivor_scorer.py:124` |
| E2 | Step 6 output and Chapter 13 input are different files; nothing is graded | `docs/CHAPTER_13_AUDIT.md` |
| E3 | Forward and reverse match counts are not in the Step-3 merge list; skip statistics are dropped by the host | fact file §2.2, §2.3 |
