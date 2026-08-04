# CLAUDE_CODE_INSTRUCTIONS_SER8_ARCHIVE_INVENTORY.md — REV1

**Inventory the ser8 pre-repository archive. READ-ONLY. Nothing is imported in this pass.**

**Runs from:** VM101, `/home/michael/distributed_prng_analysis`, venv `~/venvs/torch`.
**Reads:** ser8 (`192.168.1.229`) over the restricted read-only key.
**Deliverable:** `docs/SER8_ARCHIVE_INVENTORY.md`. **The ONLY file you create.**

**Run this AFTER `docs/PIPELINE_BEHAVIOUR_MODEL.md` exists** — its `INCOMPLETE` list (behaviours
with a WHAT but no WHY) is the priority list for §3.

---

## 0. Why this exists, and what it is NOT

The project predates its repository. Foundational material — proposals, operating guides,
implementation summaries — lives on ser8 and **has never been searchable by any session.** The
skill has listed *"pre-repository archives on ser8"* as a surface for months; **it has never once
been searched, because it could not be.**

**This is an INVENTORY, not an import.** Nothing is copied into the repo in this pass.

**Michael's own reasoning, which governs the scope:** *"all of the info on ser8 is old — everything
on the git was supposed to fix this issue."* Bulk-importing ~30+ stale documents into a repository
whose problem is that current documents go unread would **add noise to a reading failure**, and
create a second class of citable-but-superseded material. **Import happens later, per document,
with a reason.**

**Both documentation failures of 2026-08-02 were files already in git** — `CHAPTER_2_BIDIRECTIONAL_SIEVE.md`
§6 and `TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md`. **ser8 access would have prevented neither.** Its
value is narrow and real: answering *"does a WHY exist for this, anywhere?"* when the repo has none.

## 1. ⚠ PRIVACY SCOPE — this is the most important section

**`~/Downloads/` on ser8 is MIXED.** It contains this project's material **and** unrelated
engineering work and personal files: ESP32 solar-display firmware, Panasonic IGBT files,
process-tube-adapter CAD, thermal-imaging captures, **voicemails**, personal photographs, PCB
images, antenna documentation, 3MF models, and more.

> **DO NOT read, open, transfer, summarise, or index any file outside the project patterns in §2.**
> **DO NOT recurse into unrelated project folders.**
> **DO NOT open any media file — no `.jpg`, `.jpeg`, `.png`, `.mp4`, `.3mf`, `.pdb`, voicemail or
> audio — under ANY circumstances, including to identify it.**
> **If a filename's relevance is ambiguous, SKIP IT and list the name in §5 for Michael to rule on.
> Do not open it to decide.**

**`~/Downloads/PRNG/` is a clean project directory** — approximately 30 files, all
project-related. That one may be enumerated in full.

**Access is read-only by construction** (restricted key, chrooted SFTP). **Do not attempt to write,
move, delete or rename anything on ser8.** If a write is refused, that is the guard working — do
not work around it.

## 2. Scope patterns — `~/Downloads/` root only

Consider **only** files and directories matching project vocabulary:

**Directories:** `PRNG/` · `agent_contexts/` · `code_staging/` · `step6_restoration/` ·
`step_runner/` · `tb_loop/` · `out4/` · `CONCISE_OPERATING_GUIDE*/`

**File patterns:** `PROPOSAL_*` · `ADDENDUM_*` · `IMPLEMENTATION_*` · `CHAPTER_*` · `*OPERATING_GUIDE*`
· `*workflow_guide*` · `apply_s*.py` · `verify_s*.py` · `*_patch*.py` · `instructions.txt` ·
`CURRENT_Status.txt` · `agent_manifest*` · `active_job_state*` · `job_*` · `scripts_coordinator*` ·
`coordinator_adapter*` · `run_step*` · `generate_step*` · `label_leakage*` · `multi_model*` ·
`SUBPROCESS_ISOLATION*` · `Functional_Mimicry*` · `*DAILY*SLP*` (see §4)

**Anything not matching: skip silently. Do not list it, do not count it, do not describe it.**

## 3. What to produce

### 3.1 Inventory — per file

| filename | size | mtime | one-line subject | **in git?** | verdict |
|---|---|---|---|---|---|

- **Subject** from the **first ~40 lines only** — enough to say what it is. **Do not read whole
  documents in this pass.**
- **In git?** — check `docs/PROJECT_FILE_CATALOG.md` first (it is intent-indexed and faster), then
  the repo by name and by subject. **A different filename covering the same subject counts as
  present** — say which file supersedes it.
- **Verdict**, exactly one of:
  - **SUPERSEDED** — the repo has this, current
  - **ABSENT-RELEVANT** — not in the repo, and it answers a question the repo leaves open
  - **ABSENT-HISTORICAL** — not in the repo, and it is of its time only
  - **UNCLEAR** — cannot tell without reading further. **Leave it UNCLEAR; do not read further.**

### 3.2 Priority list — driven by the behaviour model

**If `docs/PIPELINE_BEHAVIOUR_MODEL.md` exists, read its `INCOMPLETE` list first** — behaviours
with a code anchor and no WHY.

**For each, state whether ser8 holds a candidate explanation, and name the file.** That mapping is
this brief's highest-value output: it converts *"we don't know why this exists"* into *"the answer
is in this specific document."*

### 3.3 The `apply_s*.py` patch corpus

There are **many** of these, session-numbered. **Do not inventory them individually.** Report:
count, session-number range, and whether the repo's own one-shot patch corpus (catalog §4.8) covers
the same range. **Note any session number present on ser8 and absent from the repo** — that is a
gap in the change record, not a document to import.

## 4. ⚠ The CA draw-procedures PDF — a specific, known gap

The repo **cites** the CA Lottery *Daily & SuperLotto Plus Draw Procedures* (Chapter 2 §5.1,
Chapter 1 §3.1.2, the skip-semantics work) but **does not contain it.** Catalog §7 records it as
**citation-only**; the skill has carried it as `UNAVAILABLE` for months.

A file resembling it appears in `~/Downloads/` (name beginning `178546540…`, containing
`DAILY` and `SLP` and `06-20`).

**Confirm identity from filename and file size only.** Report the exact filename and size.
**Do not extract, transfer, or summarise its contents in this pass** — it is a candidate for
deliberate import with a README stating its two caveats (effective **2021-06-09**, covering under a
quarter of the dataset; and it is the *"MODIFIED for Release for Solicitation"* public version).

## 5. Ambiguous names — for Michael, not for you

List any filename that **might** be project-related but that you **did not open** under §1.
**Michael rules on these. Do not resolve them yourself.**

## 6. What NOT to do

- **No import.** Copy nothing into the repo.
- **No audit.** No claim that repo content is wrong because ser8 differs. **A ser8 document is
  presumed SUPERSEDED by anything current in the repo** — it predates the repository by design.
- **No fixes**, no commits, no pushes.
- Do not run the pipeline, WATCHER, or any scraper. Do not touch `miner/`.
- **Do not write to ser8.**

## 7. Report

In your final message: files matched vs skipped-by-pattern (counts only — **do not name skipped
files**); the §3.2 priority mapping; the §4 PDF confirmation; the §5 ambiguous list; and the
`apply_s*` corpus summary. Then **STOP**.

⚠ Your output file dirties the tree — **tell Michael it must be committed before the Phase-7 soak
launches, or before a running soak reaches publication.**

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** state files matched, files inventoried, and files skipped by pattern —
  **three separate counts.** An inventory without them cannot be checked for coverage.
- **clean control / fault-injection:** `NOT_APPLICABLE` — inventory, not a detector. **Write
  `NOT_APPLICABLE`, never `PASS`.**
- **completion sentinel:** `PASS | FAIL | UNAVAILABLE | INCOMPLETE`. **If the ser8 key does not
  work, the whole report is `UNAVAILABLE` — say so and stop. Do not substitute guesses from
  filenames seen in a screenshot.**
- **audit claim scope:** ser8 `~/Downloads/` (pattern-scoped) and `~/Downloads/PRNG/`, plus the repo
  at the stated HEAD.
- **searched surfaces:** name both ser8 paths **and** `docs/PROJECT_FILE_CATALOG.md`, which is how
  "in git?" was determined.
- **unavailable surfaces:** everything on ser8 outside §2's patterns — **deliberately not searched,
  and that is correct.** Also any file listed in §5.
