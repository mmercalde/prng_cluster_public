# CLAUDE_CODE_INSTRUCTIONS_PROJECT_CATALOG_REGENERATION.md — REV1

**Regenerate `docs/PROJECT_FILE_CATALOG.md`. Read-only except for that one file.**

**Base:** current HEAD. Claude Code on **VM101** as `michael`, venv `~/venvs/torch`.

**Why this exists:** the current catalog was compiled **2026-02-04** and is six months stale. In that
window RANGE-MINER replaced the Step-2 engine, Phase 6 certified, D6.2 certified, and Chapters 1
and 2 were audited and closed. **More importantly, a name-only catalog does not solve the problem
this one is being rebuilt to solve.**

---

## 0. The problem this must actually solve

**Tonight, Team Alpha made a false absence claim** — that the three-lane CRT test was undocumented —
**while the answer sat in `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` §6, committed.** Separately, Alpha
nearly submitted a "finding" to Team Beta that **Beta had already ruled on** in
`docs/TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md`, also committed.

**Neither failure was caused by missing information. Both were caused by information being
unfindable.** `ls docs/` was available and did not help — because a **filename does not say what
question a document answers.**

> **THEREFORE: this catalog is INTENT-INDEXED, not name-indexed.** Every entry answers *"what
> question does this document settle?"* — not *"what is this file called?"*
>
> **A catalog that only lists names will have failed its purpose even if it is complete and
> accurate.**

---

## 1. What to produce

`docs/PROJECT_FILE_CATALOG.md`, replacing the current one. **Sections, in this order:**

### 1.1 THE GOVERNANCE TRAIL — put this FIRST

**The single most valuable section**, because it is the one that was missed.

Every `TB_RULING_*`, `TB_RULING_REQUEST_*`, and `TEAM_ALPHA_*` submission, with:

| file | what it rules on / asks | disposition | still binding? |
|---|---|---|---|

**Disposition** means: RULED · AWAITING RULING · SUPERSEDED BY *(name it)*. Where a ruling request
has a corresponding ruling or an implementation commit, **link them** — the pairing is what makes
the trail traversable.

**Read enough of each file to state its subject in one line.** A row saying *"a TB ruling request
about Step 2"* is worthless; *"establishes bidirectional_selectivity is 98.8% at floor and cannot
serve as the Step-2 quality signal"* is the whole point.

### 1.2 CHAPTERS — status and currency

| chapter | file | lines | audited? | currency | known-stale sections |
|---|---|---|---|---|---|

**Facts to carry, verified against the repo:** Chapter 1 audited (**9 of 41 claims accurate**);
Chapter 2 **destroyed at `248e48c`, restored from `d14dcdd`, audited, extended to 1,463 lines,
closed `ef4b1c6` with content gate `09bbfbf`**; Chapter 3 audited
(`docs/CHAPTER_3_ALIGNMENT_AUDIT.md`, **55 claims: 17 accurate / 9 stale / 24 false / 5
unverifiable**, and **§8/§9/§14.2 describe code deleted at v4.0**). Chapters **5, 6, 8, 13 are
UNAUDITED.**

### 1.3 THE INTENT INDEX — the core deliverable

**Every document in `docs/`**, one line each, stating **what question it answers**. Group by theme,
not alphabetically. Suggested themes — adjust to what you actually find:

- *why the design is what it is* (foundations, whitepapers, physical model, skip semantics)
- *what was decided and by whom* (governance trail — cross-reference §1.1)
- *how the system is operated*
- *what happened when* (session changelogs — these may collapse to one grouped line)
- *implementation instructions* (`CLAUDE_CODE_INSTRUCTIONS_*`)
- *audits and reports*

**Test each line against this question: if someone asked "where is X documented", would this line
let them find it?** If not, rewrite it.

**Session changelogs may be summarised as a group** with a date range and a note on what they are
useful for — do not write 150 individual lines for them.

### 1.4 CODE INVENTORY

Current, by role: pipeline steps · miner · utils (the frozen authorities) · tests · agents ·
scripts. **Note which are frozen/reuse-never-reimplement** (`_l2_sort_key`, `_select_l2_winners`,
`CANONICAL_ARRAY_CONTRACT`, `utils/prng_encoding`, `canonical_map_hash`, the three finalizer
validators).

### 1.5 SUPERSEDED / DO-NOT-CITE

Anything a future reader could mistake for current. **The v1 catalog's own "Runtime Data" table is a
candidate** — verify its figures before carrying them forward.

### 1.6 KNOWN GAPS

What is genuinely not documented, **each with the search that establishes it.** An absence claim
here needs the same anchor discipline as anywhere else. **Do not populate this section by
assumption** — if you did not search for it in this pass, it does not go here.

---

## 2. Method

- **`ls docs/` and count first.** State the total in the report so coverage is checkable.
- **Open every file you index.** A one-line summary written from a filename is a guess and will
  reproduce exactly the failure this catalog exists to prevent.
- For long documents, the heading structure plus the opening section is usually enough to state the
  subject. **You do not need to read 900 lines to say what a document is for.**
- **Where a document is dated and its subject has since changed, say so in the line.** A submission
  arguing for a fix is the least reliable source on whether the fix happened.

## 3. Scope

**Do not fix anything. Do not update any chapter. Do not resolve any gap you find.** Findings go in
§1.6 or into `docs/BACKLOG.md`'s existing register — **note them for Alpha, do not action them.**

Do NOT commit, do NOT push, do NOT run WATCHER or the pipeline.

## 4. ⚠ Operational constraint

**Your output file will be untracked and dirties the tree.** The Phase-7 soak's clean-tree preflight
rejects a dirty tree at finalization. **Write only `docs/PROJECT_FILE_CATALOG.md`, create no other
file, and end your report by telling Michael it must be committed before the soak launches** — or,
if the soak is already running, that it must be committed before the run reaches publication.

## 5. Report

In your final message, not a second file: the document count indexed vs the total in `docs/`; which
themes you used; anything you could not classify; and the §1.6 gaps with their searches. Then STOP.

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** state the total file count and the number indexed. **A catalog claiming
  completeness without those two numbers is unverifiable.**
- **clean control / fault injection:** `NOT_APPLICABLE` — this produces an index, not a detector.
  **Write `NOT_APPLICABLE`, never `PASS`.**
- **completion sentinel:** `PASS | FAIL | UNAVAILABLE | INCOMPLETE`. **Partial coverage is
  `INCOMPLETE`, and that is an acceptable outcome** — say what you did not reach.
- **audit claim scope:** repo-scoped. **Say so.** The catalog indexes what is committed; host state
  and any ser8 archive are **out of scope and must not be implied.**
- **searched surfaces:** `docs/`, the repo root, and every directory you index — **name them.**
- **unavailable surfaces:** anything gitignored (`git check-ignore` before calling a config absent),
  host state, and any pre-repository material.
