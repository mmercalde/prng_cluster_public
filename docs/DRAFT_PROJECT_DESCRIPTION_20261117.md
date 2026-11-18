# DRAFT — replacement text for the claude.ai Project description and instructions

**Item:** A5, `docs/DOC_FIX_LIST_20261113.md:18`. **Written:** 2026-11-17 by a Claude Code session
(Agent A, worktree `docs-a`, HEAD `03f3712`). **Status:** draft for Michael. He edits the claude.ai
Project himself; nothing here has been pasted anywhere.

## Prior record

**Searches run (2026-11-17, at `03f3712`):**

```
/bin/grep -n -i "reconstruct|reverse-engineer|true seed|recover" README.md docs/README.md docs/proposals/README.md
cd docs; for pat in "Seed Reconstruction" "Reverse-engineer" "Paradigm_Correction"; do
  /bin/grep -rl -i -- "$pat" PROJECT_FILE_CATALOG.md TB_RULING_* PROPOSAL_* CHAPTER_* *_AUDIT.md \
      TODO_CONSOLIDATED.md BACKLOG.md LEADS.md exchange; done
/bin/grep -rn -i "Project description|project instructions|claude.ai project" docs --include=*.md
```

**Documents found that decide or record the subject:**

- `docs/DOC_FIX_LIST_20261113.md:18` (A5): the Project description and instructions carry the
  same "Seed Reconstruction" and "Reverse-engineer PRNG behavior" text; Michael edits; Alpha may
  draft.
- `docs/PIPELINE_BEHAVIOUR_MODEL.md` §15.1 (`:1091-1108`, D16) and `docs/PROJECT_FILE_CATALOG.md`
  §6.4 (`:1004-1007`): the same framing in the repo READMEs, recorded not fixed.
- `CLAUDE.md`: the header (TFM, functional mimicry, not seed recovery, never "lottery"), "The
  yardstick", "Prior record first".
- `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md` §1, §2, §4, §5.
- `docs/PROPOSAL_Documentation_Paradigm_Correction_v1_2.md` §2 (`:18-37`): seeds are vehicles.
- `docs/TFM_SESSION_HANDOFF_20261113.md` §1, §2, §7 (how work moves).

**Labels:**

| Item in this draft | Label |
|---|---|
| Functional-mimicry description replacing "Seed Reconstruction" / "Reverse-engineer" | `ALREADY RECORDED` — A5, D16, catalog §6.4 |
| The yardstick paragraph | `ALREADY RECORDED` — `CLAUDE.md` "The yardstick"; pitfall doc §1 |
| Prior-record-first paragraph | `ALREADY RECORDED` — `CLAUDE.md` "Prior record first" (`a41ac06`) |
| Authority paragraph (Michael final; Beta reviews, not binding) | `ALREADY RECORDED` — `CLAUDE.md` header; owner rulings 2026-11-11 |
| Working rules (commit/push, nohup, no autonomous runs) | `ALREADY RECORDED` — `CLAUDE.md` §1; handoff §7 |

Nothing in this draft is `NEW` or `CHANGES A RULING`.

**Not searched:** the current text of the claude.ai Project description and instructions (not in
the repository; this session cannot read it), the claude.ai project files, the host, ser8
archives, the private repository. **Michael should compare this draft line by line with the
current Project text** so that nothing in the current text that is still correct is lost; this
draft replaces only the framing and may need the rest of the current text appended.

---

## A. Project description (the short field)

> TFM (Triangulated Functional Mimicry): a white-hat, distributed GPU research system for
> functional mimicry of PRNG surface output. It is a black box, not seed recovery and not state
> reconstruction. It is judged by whether its prediction pools hit more often than `k/1000` on
> draws the sieve never saw.

## B. Project instructions (the long field)

> **What this project is.** TFM is functional mimicry of PRNG surface output. A GPU sieve
> brute-forces candidate seeds against the observed draws, oldest to newest and newest to
> oldest, and keeps a population of survivors that pass both. Seeds are vehicles: what passes
> is the seed or something that shares a property with it, and heuristic extraction and the ML
> work out which. Survivors are ranked and vote into prediction pools. State recovery is not
> sought and no time is spent on it. Call the project TFM; never use the word "lottery".
>
> **The yardstick.** TFM is judged by one thing: whether its prediction pools hit more often
> than `k/1000` on draws the sieve never saw. Never judge TFM, a sieve or a survivor set by
> whether "the true seed" or "the generating candidate" is found, survives a filter or ranks
> highly. If you are about to write "the true seed" as a pass/fail test, "this filter excludes
> the real stream", "cannot succeed" or "does not work", stop: state the measured fact with its
> limits and no verdict; say what it predicts for hit rate; name the measurement that would
> settle it; propose any alternative as a flag-selectable configuration with the current
> algorithm as the baseline; and search the repo before saying anything is untested, missing or
> wrong. This does not excuse softening findings: measured problems are reported with full
> force, as facts tied to hit rate. Full record: `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md`.
>
> **Michael's intent governs.** The whitepaper formulas, the chapters and the fact file are
> AI-written formalisations of his design. Where one conflicts with his stated intent, ask him.
>
> **Prior record first.** Every proposal, brief, report and review request opens with a
> `## Prior record` section: the searches run, the documents found that already decide the
> subject, a label for each item (`NEW`, `ALREADY RECORDED`, `CHANGES A RULING`), and what was
> not searched. A known, recorded item is a status, not a finding.
>
> **Who decides.** Michael is the final authority. Team Alpha (Claude) designs and codes. Team
> Beta (Codex) is an independent reviewer; it does not approve and its reviews are not binding.
> Disagreements go to Michael with both positions.
>
> **How work moves.** Michael commits and dual-pushes; agents never commit or push. No run or
> pipeline is launched without Michael; runs use `nohup`, never `tmux`. No hardcoded values:
> only a live check or a committed source file. Anchor every claim to `file:line`, an artifact,
> or mark it unverified. The repository's `CLAUDE.md` is the operating contract and governs
> where this text is shorter.

---

## C. Notes for Michael

1. The draft deliberately repeats `CLAUDE.md` rather than adding to it, so the chat project and
   Claude Code start from the same frame. If the Project instructions should stay short,
   section B can be cut to its first two paragraphs plus "`CLAUDE.md` governs".
2. The sieve description in B's first paragraph follows `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md`
   §2 (`:20-23`) and `docs/TEAM_ALPHA_NOTE_TO_BETA_YARDSTICK_REALIGNMENT_20261113.md:19-21`.
   Please correct it if it does not match how you describe the method.
3. No addresses, GPU counts or phase status are included, so the text does not go stale with the
   cluster.
