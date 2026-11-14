# PRIOR-RECORD RULE — BLOCKS FOR `CLAUDE.md` AND `AGENTS.md` — 2026-11-13

**From:** Team Alpha · **Owner:** Michael · **Base commit:** `ea85436`

## Prior record

Searches run on the public mirror at `ea85436`:

- `grep -n "^## " CLAUDE.md AGENTS.md` — where existing rules sit.
- `grep -n "THE RULE\|SEARCH ORDER" docs/TFM_PROJECT_FACTS_SKILL.md` — the existing search-first rule.
- `grep -rn "known, recorded item is a status" AGENTS.md` — Beta's existing standard.

Found, already recorded:

| Document | What it already says | Status |
|---|---|---|
| `docs/TFM_PROJECT_FACTS_SKILL.md` §1, §1.1 | No absence claim without an anchor read this session; search the governance trail, then chapters, then code | ALREADY RECORDED |
| `AGENTS.md` §5, "Standards for what you write" | Before reporting a new defect, search `docs/`; a known item is a status, not a finding | ALREADY RECORDED |
| `CLAUDE.md`, "The yardstick" | Search the repo before any claim that something is missing, untested or broken | ALREADY RECORDED |

What is NEW here: the rule is made checkable. A document must **show** its search in a fixed
section, and a document without that section is refused. Nothing above required that.

Not searched: VM101, ser8, the private repository.

**Why this exists.** On 2026-11-13 Team Alpha wrote a proposal and three briefs without
searching the governance trail. The hit-rate measurement, the Step 6 pool format and the
session rule were already decided in `TB_RULING_S176` to `S179` and the rulings of
2026-07-30/31. One brief prescribed a SQLite read that `tests/gate_s172_prod_shape.py:137-162`
already documents as unsafe. The owner and Beta caught each one. The rule already existed in
three places and was not followed, so it is now enforced by the shape of the document.

---

## Block A — insert into `CLAUDE.md`, immediately before `## 0. Run-as and identity (READ FIRST)`

```markdown
## Prior record first (READ BEFORE WRITING ANY PROPOSAL, BRIEF, REPORT OR REVIEW REQUEST)

Owner rule, 2026-11-13. It binds Team Alpha in chat, every Claude Code session, and Beta.

Every proposal, brief, report and review request **opens with a `## Prior record` section**,
written before anything else in the document. It holds four things:

1. **The searches run**, as commands with their scope, covering at least
   `docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`, `docs/TB_RULING_REQUEST_*`,
   `docs/PROPOSAL_*`, `docs/CHAPTER_*` and `docs/*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`,
   `docs/BACKLOG.md`, `docs/LEADS.md` and `docs/exchange/`.
2. **Every document found** that already decides, authorises, schedules or records the
   subject, with file and section.
3. **For each item the new document raises,** one label: `NEW`, `ALREADY RECORDED` (cite it),
   or `CHANGES A RULING` (cite it; this needs the owner).
4. **What was not searched:** the host, ser8 archives, the private repository, project files
   that are not in the repository.

A document without this section is not accepted. A session given one stops and reports it.
Beta answers `NEEDS EVIDENCE`. The owner rejects it.

A known, recorded item is a **status**, not a finding. Reporting it as new is an error.
Read the hits; a keyword match is not a finding until the surrounding text has been read.
```

## Block B — insert into `AGENTS.md`, immediately before `## 2. Who decides (owner rulings, 2026-11-11)`

```markdown
### Prior record (owner rule, 2026-11-13)

Every request you receive must open with a `## Prior record` section: the searches the author
ran, the governing documents found, a label for each item (`NEW`, `ALREADY RECORDED`,
`CHANGES A RULING`), and what was not searched.

- If the section is missing, or lists no searches, answer `NEEDS EVIDENCE` on every question,
  say why in one line, and do not review the substance.
- If the section is present, check it. A governing document it missed is your finding 1,
  ahead of anything else.
- Your own reviews follow the same rule you already hold: a known, recorded item is a status,
  not a finding.
```
