# CLAUDE_CODE_INSTRUCTIONS_BACKLOG_MAINTENANCE.md — REV1

**Repair and update `docs/BACKLOG.md`. Nothing else.**

**Base:** HEAD `c4917a8` or later. Claude Code on **VM101** as `michael`, venv `~/venvs/torch`,
**run from `/home/michael/distributed_prng_analysis`.**

**Deliverable:** `docs/BACKLOG.md`, edited in place. **The ONLY file you change.**

---

## 0. What this file is

The **tracked register of known, deliberately-deferred items** — things that are not blocking, are
understood, and must not be rediscovered as surprise findings later. It is 274 lines, 14 sections,
and actively used.

**Preservation applies.** Every entry cost something to learn. **Do not delete, condense, merge or
"tidy" any existing item.** This pass repairs structure, corrects one stale fact, and adds four
new items. Nothing else.

## 1. Repairs

### 1.1 Section order is broken

`## 10. Standing reminders that keep costing us` currently sits **after** `## 14`. It was displaced
when §§11–14 were inserted.

**Move §10 back into numeric position**, so the file reads 1 … 14 in order. **Renumber nothing** —
move the section, keep its number and its content byte-identical.

### 1.2 §12 is stale — Beta ruled on D3.0-B

§12 reads *"D3.0-B — awaiting Beta's disposition."* **Beta has ruled**, 2026-08-02.

**Replace the disposition with the ruling.** Verify against
`docs/TEAM_ALPHA_D3_0_B_AND_ITEM1_NOTICE.md` and the skill's §2.18 before writing — **do not
transcribe from this brief.** The ruling in substance:

- **D3.0-B is OPEN and REQUIRES COMPLETION.** *Waived* and *superseded* were both **rejected**.
- **Beta recorded its own Phase-6 certification as a governance error** for omitting the
  prerequisite.
- **Phase 6's certification scope is narrowed** — certified for the demonstrated miner/finalizer
  path; **legacy conversion and dormant legacy-writer surfaces are UNCERTIFIED.**
- **The legacy converter must not be invoked until D3.0-B closes.**
- **Does not block the miner-backed Phase-7 soak.**
- Bounded scope when done: canonical fail-closed resolver replacing missing-identity defaults ·
  preserve valid `prng_type` precedence and `prng_base` fallback · reject records carrying neither ·
  **retire divergent executable encoding tables including rerunnable patch scripts** · behavioural
  gates and mutants for missing identity, unknown identity, and reintroduced `java_lcg` defaulting.

### 1.3 The currency stamp is commit-pinned and 23 commits stale

Line 7 pins **HEAD `6892661`**. HEAD is now `c4917a8`.

**This is the third instance of the same defect today** — the skill's currency line was pinned and
went stale within minutes of loading; `PROJECT_FILE_CATALOG.md`'s is **12 commits** behind. **A
commit-pinned currency stamp goes stale the moment anything else lands, and it reads as noise on
the first line a reader sees.**

**Change to a date**, matching the skill's v14 form:

```
**Currency:** 2026-08-03. Every anchor in this file was re-read at source when the file was written.
An anchor is a claim with an expiry date — re-verify before acting.
```

Keep the existing anchor-expiry sentence; **only the pin changes.**

*(`PROJECT_FILE_CATALOG.md`'s stamp is out of scope here — note it in your report, do not edit it.)*

## 2. Four new items to add

Number them **15–18**, after §14, **before** the relocated §10 if §10 is last by design — **decide
from how the file actually reads and say what you chose.**

### §15 — Step 3's output validation floor is stale

`run_step3_full_scoring.sh:475-478` tests `if feature_count < 46`, against a comment reading
`# Check feature count (should be 50)`.

**The live contract is 91 extracted / 89 trained.** A run emitting **46** features passes this
validation. **The wall is set three contracts behind the code it guards.**

Source: the Step-3 script read, 2026-08-03. **Verify the line numbers yourself before writing
them.** Not blocking; not fixed.

### §16 — `full_scoring.json` declares 26 GPUs; the frozen set is 25

The manifest declares `parallel_workers: 26` with the note *"Use full cluster (26 GPUs)"*. The
frozen Phase-7 execution set is **25** (`bea580e76490…`, owner-ruled, Beta-ratified).

**The script does not consume it** — the manifest's own `_note_default_params` says so — **so this
is documentation drift, not a wiring path.** No execution consequence.

**Record it as the fourth place carrying a stale 25-vs-26 figure**, alongside
`ml_coordinator_config.json` (which is a **live** 26 and correctly stays), and the two already
corrected in the catalog and behaviour model. Not blocking; not fixed.

### §17 — a skill revision lives in three places

`docs/TFM_PROJECT_FACTS_SKILL.md` (tracked) · `~/.claude/skills/tfm-project-facts/SKILL.md`
(Claude Code, on invocation) · the Settings upload (new chat sessions, at session start).

**Committing updates ONE. Nothing warns when they diverge.**

On **2026-08-03** the tracked copy reached **v13** while the installed copy still held **v6** — last
touched 00:22 that day, **before the entire day's work** — and Settings held **v11**. **Thirteen
revisions, none of which had reached a runtime copy.** Every correction made that day protected
nothing until the copies were fixed by hand.

**A running chat session cannot be updated at all** — its copy is fixed at session start.

The four-step completion rule is in the skill's §7 working agreements. **This entry exists so the
failure is findable from the backlog too.**

### §18 — `chapter_13_triggers.py` reaches Step 3 outside `--end-step`

Two routes, **both requiring a human action**, so neither is a live risk:

- `chapter_13_triggers.py:647` — `execute_standalone` runs `run_step3_full_scoring.sh` via
  `subprocess.run`; sole caller is the CLI at `:932`;
- `execute_learning_loop` (`:578`) defaults to steps `[3,5,6]` and calls
  `run_pipeline(start=3, end=6)` at `:617` — **a fresh `run_pipeline` with its own bounds, so a
  soak's `--end-step 1` does not constrain it.**

**Standing operational rule: no Chapter-13 approval or learning-loop invocation while a soak is
running.** Verify the line numbers yourself.

## 3. What NOT to do

- **Do not touch any other file.**
- **Do not delete, condense or merge any existing entry.**
- Do not fix anything the backlog describes — **it is a register, not a worklist.**
- Do not run the pipeline, WATCHER or any scraper. Do not touch `miner/`.
- **Do not commit or push.**

## 4. Report

Final message: the section order before → after · the §12 text you wrote and the source you
verified it against · the four new sections' numbers and where you placed them, with your reasoning ·
line count before → after · confirmation **no existing entry was removed or condensed** · and the
`PROJECT_FILE_CATALOG.md` currency observation, **reported not fixed**.

Then **STOP**.

⚠ This dirties the tree — **tell Michael it must be committed.**

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** quote the `file:line` you verified for §15 and §18 rather than copying this
  brief's numbers. **A brief is a snapshot and its anchors expire** — re-read them.
- **clean control:** after editing, `grep -n "^## "` and show the full section list, in order.
- **fault-injection control:** `NOT_APPLICABLE` — a document edit, not a detector. **Write
  `NOT_APPLICABLE`, never `PASS`.**
- **completion sentinel:** `PASS | FAIL | UNAVAILABLE | INCOMPLETE`.
- **audit claim scope:** repo-scoped at the stated HEAD.
- **searched surfaces:** `docs/BACKLOG.md`, `docs/TEAM_ALPHA_D3_0_B_AND_ITEM1_NOTICE.md`, the
  skill's §2.18 and §7, `run_step3_full_scoring.sh`, `agent_manifests/full_scoring.json`,
  `chapter_13_triggers.py`.
- **unavailable surfaces:** name anything you could not open.
