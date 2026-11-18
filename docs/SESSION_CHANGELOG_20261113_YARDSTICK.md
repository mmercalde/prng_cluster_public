# SESSION CHANGELOG — 2026-11-13 — THE YARDSTICK REALIGNMENT

**Changelog without an S-number.** Documentation only.
**Written:** 2026-11-17, by a Claude Code session (Agent A, worktree `docs-a`, HEAD `03f3712`),
under `/home/michael/doc_fix_brief_20261117.md`, item D4 of `docs/DOC_FIX_LIST_20261113.md:48`.
**This is a reconstruction from the committed record, not a first-hand log.** The session that
wrote it was not present on 2026-11-13. Every line below is anchored to a commit or a committed
file; nothing is taken from memory.
**Scope:** the yardstick thread of 2026-11-13 only. The day's other threads (findings proposal,
SQLite reads, run gates, landing script, Run 4 range, Step 1 rerun policy) are recorded in
`docs/TFM_SESSION_HANDOFF_20261113.md` §3-§4 and are not repeated here.

## Prior record

**Searches run (2026-11-17, in this worktree at `03f3712`):**

```
git log --format="%h %ad %s" --date=iso --since=2026-11-13T00:00 --until=2026-11-14T00:00 --reverse
ls docs | /bin/grep "20261113"            # documents dated 2026-11-13
ls docs | /bin/grep -i "SESSION_CHANGELOG_2026111"
cd docs; /bin/grep -rl -i "SESSION_CHANGELOG_20261113" PROJECT_FILE_CATALOG.md TB_RULING_* PROPOSAL_* \
    CHAPTER_* *_AUDIT.md TODO_CONSOLIDATED.md BACKLOG.md LEADS.md exchange
/bin/grep -n -i "yardstick|pitfall|reverse sieve|realign|DOC_FIX|changelog" docs/TFM_SESSION_HANDOFF_20261113.md
```

**Documents found that record the subject:** `docs/TFM_SESSION_HANDOFF_20261113.md` §1, §2.1,
§3, §6.1 and §8 (`:180`, which lists this changelog as owed); `docs/DOC_FIX_LIST_20261113.md`
(D4 at `:48`); `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md`;
`docs/TEAM_ALPHA_NOTE_TO_BETA_YARDSTICK_REALIGNMENT_20261113.md`;
`docs/CLAUDE_MD_AND_AGENTS_MD_YARDSTICK_BLOCKS_20261113.md`;
`docs/exchange/from_beta/X-REVERSE-SIEVE_R2_BETA_REALIGNMENT.md`. No existing
`SESSION_CHANGELOG_20261113*` file; no hit for the name in the governed scopes.

**Labels:** every item below is `ALREADY RECORDED` in the documents above; this file collects
them. Nothing in it is `NEW` or `CHANGES A RULING`.

**Not searched:** the host, ser8 archives, the private repository, the claude.ai project files
(including `TEAM_ALPHA_REVIEW_REQUEST_REVERSE_SIEVE_20261113.md` and Beta's X-REVERSE-SIEVE R1,
neither of which is in the repository at `03f3712`), and the 2026-11-13 chat itself.

---

## 1. What happened

1. Team Alpha raised the reverse sieve on a state-recovery standard: a candidate that generated
   a window passes the forward pass and fails the reverse pass, so "the sieves do not work",
   with a proposed acceptance test "the generating candidate must pass both passes". Beta
   adopted the acceptance test (`docs/PITFALL_STATE_RECOVERY_YARDSTICK.md:42`;
   `docs/TEAM_ALPHA_NOTE_TO_BETA_YARDSTICK_REALIGNMENT_20261113.md:11-24`).
2. Michael restated the yardstick: TFM is judged by whether its prediction pools hit more often
   than `k/1000` on draws the sieve never saw; state recovery is not sought
   (`docs/PITFALL_STATE_RECOVERY_YARDSTICK.md` §1-§2; `docs/TFM_SESSION_HANDOFF_20261113.md:26-32`,
   `:43-44`).
3. Alpha withdrew the framing the same day. This was the second occurrence; the first was the
   2026-08-08 attack-plan report (`docs/PITFALL_STATE_RECOVERY_YARDSTICK.md:37-42`).

## 2. Commits

| Commit | Time (PST) | What |
|---|---|---|
| `6267c7b` | 17:13 | Yardstick Block A in `CLAUDE.md` and Block B in `AGENTS.md`; `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md` (101 lines); `docs/TEAM_ALPHA_NOTE_TO_BETA_YARDSTICK_REALIGNMENT_20261113.md`; `docs/CLAUDE_MD_AND_AGENTS_MD_YARDSTICK_BLOCKS_20261113.md`; `docs/DOC_FIX_LIST_20261113.md`. 6 files, +368 |
| `4b1e81e` | 18:48 | Among other files: `docs/exchange/from_beta/X-REVERSE-SIEVE_R2_BETA_REALIGNMENT.md`, hand-carried from the ChatGPT project, unchanged |
| `d60a714` | 20:58 | `docs/TFM_SESSION_HANDOFF_20261113.md`, which records the yardstick as owner ruling 1 and lists this changelog as owed |

## 3. Facts recorded that day (status, not verdicts)

As listed in `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md` §7 (`:89-101`) and the realignment note §3:

- The reverse pass is the forward kernel run against the window newest first. At `03f3712` this
  is `miner/range_miner_worker.py:1330`; the pitfall document's `:1201` was correct at
  `b10e1ef` and moved with `048fd47`.
- A candidate that generated a window passed forward and failed reverse, in Alpha's C and Python
  copies and in Beta's CPU wrapper around the unchanged kernel bodies, constant and hybrid. Not
  run on GPU or VM101.
- Beta's R2 limits: measured facts about the tested streams and settings, not a universal
  statement, not evidence of predictive loss; no percentage loss is known; the objection is
  THEORETICAL and holds no commit (`X-REVERSE-SIEVE_R2_BETA_REALIGNMENT.md:34`, `:38-46`).
- Chapter 2 F-7 stays confirmed, unresolved.
- Alpha's unmeasured prediction: a position-kept reverse pass would score a higher hit rate; to
  be tested as a flag-selectable configuration against the current baseline.

## 4. Open after the day (from the committed record)

- Beta's R2 §5 states that the bound at `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md:97-98`
  ("forward plus reverse matches cannot exceed the window size") is false as stated; the
  handoff lists it as owed (`docs/TFM_SESSION_HANDOFF_20261113.md:181`). Not corrected as of
  `03f3712`.
- The documentation fix list, `docs/DOC_FIX_LIST_20261113.md`. A1 and A2 landed in `6267c7b`.
  The remaining framing items are taken up by the 2026-11-17 Agent A pass
  (`docs/CLAUDE_CODE_REPORT_DOC_FIX_LIST_A_20261117.md`); the fact file, registries and
  catalog items by Agent B.
- The baseline hit-rate measurement on real draws has not been made
  (`docs/PITFALL_STATE_RECOVERY_YARDSTICK.md:74-77`).

## 5. What this changelog did not do

No code, config or JSON touched. No run launched. Nothing committed. It does not decide or
re-open anything; it collects what the committed files already say.
