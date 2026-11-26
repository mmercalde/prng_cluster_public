# X-6P2-SCRAPER · R1 · REVIEW_REQUEST · 6-P2 safe dataset scraper, brief REV5 (design)


## Prior record

**Searches run** (worktree at base `aefdd74`, 2026-11-26):
- `/bin/grep -rlnE -i "unseen draw|draws after 2026-02-26|post-cutoff|counted holdout|CONTAINS_UNSEEN" docs`.
- `/bin/grep -nE -i "scraper|6-P2|6P2|lottery\.net|lotteryextreme|calottery"` over `docs/LEADS.md`,
  `docs/BACKLOG.md`, `docs/TODO_CONSOLIDATED.md`, `docs/exchange/LEDGER.md`, `docs/exchange/DEFERRED.md`.
- `ls docs/TB_RULING*` and `docs/exchange/to_beta`, `from_beta`, filtered for scraper, 6P2 and dataset.
  **No 6-P2 thread exists; no Beta review of REV4 exists.**
- `git log -- daily3_scraper.py` (one commit, `334dacf`).
- Carried by reference: the full prior-record section of the brief under review, and of
  `~/tfm_worktrees/_staging/scraper_check/PLAN_6P2_SAFE_SCRAPER_20261126.md`.

**Documents that already decide this subject:**

| Document | What it settles |
|---|---|
| `docs/DATASET_PUBLICATION_SCHEMA_v1.md` | FROZEN layout, grammar, pointer manifest, lineage and correction protocol. Wins over the brief. |
| `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_6_P2_SCRAPER.md` REV4 (`7004c08`) | The prior brief; REV5 keeps all of its rules (REV5 §0). |
| REV4 §2; `docs/TFM_PROJECT_FACTS_SKILL.md:3216` | Transition option (a), binding as relayed. No Beta file holds it. |
| `docs/TEAM_ALPHA_APPEND_ONLY_SIMPLIFICATION.md` | Rewrite mode will not exist (RULED). |
| `docs/TEAM_ALPHA_SCRAPER_RECENT_SAFETY_NOTICE.md` | The overwrite hazard; the service stays off. |
| `docs/BACKLOG.md` §14 | `2019-01-25` is not 6-P2's work. |

**Labels for what this request raises:** as in the brief's "Labels" table. In short: the REV4 design is
ALREADY RECORDED and kept; sources and independence, quarantine, draw-number continuity, the
pointer-is-the-release finding, unseen-draw handling, fail-closed fetching, dry-run default, verifier A1
scoping and the historical key list are NEW. **Nothing in REV5 changes a ruling.** REV5 §5.4 option (d)
would change the frozen schema and is listed for the owner only.

**Not searched:** the live tree's gitignored dataset files, ser8, the private repository, chat-only
rulings, `~/piece_matcher_project/`, the source websites.

---

```
id / round      X-6P2-SCRAPER / R1
type            REVIEW_REQUEST
from / to       Team Alpha (Claude Code)  ->  Team Beta (Codex)
date            2026-11-26
review class    DESIGN (brief, no code). The later U3 durable writer is advised for a
                before-commit review because it writes every counted input.
base_commit     aefdd74 (main)
answers         (opening packet)
```

**Nothing here is code, and nothing is committed.** Read the files at these paths:

| File | Status |
|---|---|
| `~/tfm_worktrees/_staging/scraper_check/rev5/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_6_P2_SCRAPER_REV5_DRAFT.md` (staged outside the repo; not in docs/ yet) | new; read first |
| `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_6_P2_SCRAPER.md` | REV4, unchanged, for comparison |
| `daily3_scraper.py` | unchanged; the code REV5 replaces |
| `miner/dataset_authority.py:381-600`, `agents/watcher_agent.py:910-930, 2080-2095`, `window_optimizer.py:2030-2060`, `scripts/verify_dataset_publication.py:235-252` | unchanged; anchors for REV5 §5.1, §9.3, §9.4 |

The agent reports REV5 cites (`r4_scrape1.txt`, `r5_alt.txt`, `r6_hist.txt`, `r7_deep.txt`) are
outside the repo, in `~/tfm_worktrees/_review/agents_20261126/`. **Do not open any file whose name
contains `CONTAINS_UNSEEN`, and do not print any draw value dated after `2026-02-26 midday`.**

## 1. What changed

- REV4's design is kept whole (REV5 §0 lists it).
- The single source becomes three independent sources: lottery.net; lotterycorner and lotteryextreme
  counted as one (they appear to share an upstream); the calottery API. A draw is accepted only when two
  independent sources agree and none disagrees. Disagreements and single-source draws are quarantined,
  and publication stops at the first quarantined key.
- Draw-number continuity is a completeness check; the last published number is derived live.
- Moving the pointer releases new draws to every governed run (REV5 §5.1, anchored). The release rule
  is left to the owner; until then no publication moves the pointer past `2026-02-26 midday`.
- Fetching is polite and fails closed; logs carry keys and counts only; dry run is the default.
- Verifier A1 is scoped to L001's first version; the authority-check location is put to the owner.
- Historical corrections are a separate, owner-approved lineage step, keys listed, none decided.
- WATCHER scheduling is named as a later step (6-P3) with its prerequisites.

## 2. Questions

Allowed answers: `NO OBJECTION` · `OBJECTION` (with Severity `LIVE` or `THEORETICAL`) ·
`SUGGESTION` · `NEEDS EVIDENCE`.

1. Lineage (REV5 §2): is there any path by which an existing version file, or `daily3.json`, could be
   changed, or a new lineage created by the program?
2. Independence (§3.1-§3.2): is counting lotterycorner and lotteryextreme as one vote, and requiring
   two independent agreeing sources with no dissent, sufficient and not over-strict for routine
   appends?
3. Quarantine (§3.3-§3.4): does stopping publication at the first quarantined key prevent every case
   where a later resolution would become a backfill under REV4 §3.2?
4. Continuity (§4): is the draw-number rule (consecutive from the live-derived last number; midday
   lower on a date; O over lotteryextreme) sound, and is `DRAW_NUMBER_UNAVAILABLE` the right outcome
   beyond O's depth?
5. Precedence (§9.2): is the extended order right, in particular where quarantine and continuity (7)
   sit relative to `CORRECTION_REQUIRED` (5) and `NON_APPEND_INSERTION_REQUIRED` (6)?
6. Release (§5.1, §5.3): is the finding correct that moving the pointer releases the new version to
   WATCHER and window-optimizer runs, and does the interim rule keep unseen draws out of every
   pipeline path? Name any reader of the bare alias or the pointer that REV5 missed.
7. Unseen handling (§5.2): can a draw value reach a log, an exception, a test output or a report by
   any route the listed rules and `G-NO-VALUE-IN-LOG` do not cover?
8. Fail closed (§7): does the run-level rule for a (year, session) failure, together with the per-source
   rule, close the partial-session hazard (plan A15)?
9. Verifier and authority (§9.3, §9.4): is the A1 scoping correct, and which of options (i) and (ii)
   avoids a weaker duplicate of the authority check?
10. Gates and mutants (§9.6): is any REV5 rule without a gate, or any gate without a mutant that kills
    it? Is the fresh-copy proof (§9.7) sufficient?
11. Historical corrections (§6): is keeping them out of the scraper, as a separate owner-approved
    lineage step, consistent with the schema's correction protocol?
12. Any governing record this request missed.
