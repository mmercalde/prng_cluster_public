# CLAUDE_CODE_INSTRUCTIONS_CATALOG_CORRECTIONS.md — REV1

**Two small corrections and one coverage report. No restructure, no rewrite.**

**Base:** HEAD `c1e9205` or later. Claude Code on **VM101** as `michael`, venv `~/venvs/torch`,
**run from `/home/michael/distributed_prng_analysis`.**

**Files you may edit — ONLY these two:**
- `docs/PROJECT_FILE_CATALOG.md`
- `docs/PIPELINE_BEHAVIOUR_MODEL.md`

**Do NOT touch `docs/TFM_PROJECT_FACTS_SKILL.md`** — v13 is correct and is the source of correction 1.

---

## Part A — the corrections

### A1. Chapters 1 and 2 closure commit — `PROJECT_FILE_CATALOG.md`

**The catalog attributes both chapters' closure to `81ef3f1`. That is the commit that COMMISSIONED
the closure, not the one that performed it.**

Verified at source:
- **`81ef3f1`** — *"docs: admission binding submission (both Phase-7 closure repairs) + Chapter 1/2
  closure **brief**"* — adds the brief, **touches neither chapter file**
- **`ef4b1c6`** — *"docs: Chapters 1 and 2 closed — anchors re-verified against HEAD, §4.1 snapshot
  regenerated, §6.2 settled at 43…"* — **edits both chapter files (+269/+698)**

**Re-verify both with `git show --stat` yourself before editing.** Then:

- Chapter 1 row (`:175`) and Chapter 2 row (`:176`): closure commit → **`ef4b1c6`**.
- **⚠ Row 175 is internally inconsistent** — it says *"CLOSED at `81ef3f1`"* and then, in the same
  cell, *"closure `ef4b1c6`"*. **Resolve the contradiction, do not leave both.**
- **Search the whole file for other `81ef3f1` occurrences** and correct any that assert closure.
  Where `81ef3f1` is correctly described as the commissioning brief, **leave it**.

`docs/TFM_PROJECT_FACTS_SKILL.md` §2.17b and its correction note already carry the right answer —
use it as a cross-check, **not** as the sole source.

### A2. "26-GPU saturation" → 25 — both files

The owner ruled the Phase-7 soak runs at **exactly 25 GPUs** (24 AMD RX 6600 XT + one VM101 RTX
3080 Ti; the second 3080 Ti stays on VM100). **Team Beta ratified the waiver.** The frozen execution
set is `bea580e764905a0d9485d2688be5841cc95f16e16837c23aced1f634d97f67a8` — 25 identities, 25
requested, 25 admitted, unclamped, non-partial.

Both files predate the ruling and still say 26:
- `PROJECT_FILE_CATALOG.md` **§5.2**
- `PIPELINE_BEHAVIOUR_MODEL.md` **§1.1**

**Grep both files for `26-GPU`, `26 GPU` and `26 GPUs` and fix every occurrence that describes the
S172 Phase-7 soak.**

**⚠ Do NOT blanket-replace 26 with 25.** Some 26s are correct and must survive:
- **historical** references to the 26-GPU fleet as it was before the second card moved;
- **`ml_coordinator_config.json` names a 26-GPU fleet** — that is a live fact about a tracked file
  (behaviour model §17 I-5) and **stays 26**;
- `eff6616`'s **clamp worked example** (`min(25, 26) = 25`) — the 26 is load-bearing arithmetic.

**Judge each occurrence. Report every one you changed and every one you deliberately left.**

### A3. `PIPELINE_BEHAVIOUR_MODEL.md` §17 preamble — add the missing surface

§17's preamble lists the surfaces where an explanation might live: the 168-file changelog corpus,
the eleven unaudited chapters, `instructions.txt`, `Cluster_operating_manual.txt`, the PDFs and
`.docx`, and ser8.

**It does not list the `apply_s*.py` patch corpus — and that is where I-6's answer was found**
(§17.1), in docstrings quoting TB rulings verbatim.

**Add it to the preamble list**, noting: 123 `apply_s*.py` + 4 `verify_s*.py` at repo root ·
**their docstrings quote TB rulings verbatim** · indexed at catalog §4.8 · **forensic only, never
re-execute**. **Say plainly that this surface was missing from §17's own list and held I-6's answer
anyway** — that is the lesson worth carrying.

---

## Part B — the coverage report

**Michael wants to know what each document actually contains**, so he can tell at a glance which to
reach for.

Write it **in your final message only — create NO new file.**

For **each** of the three:
`docs/TFM_PROJECT_FACTS_SKILL.md` · `docs/PROJECT_FILE_CATALOG.md` ·
`docs/PIPELINE_BEHAVIOUR_MODEL.md`

report:

1. **line count** and **top-level section list** (headings only, one line each);
2. **what it is the authority on** — one sentence;
3. **what question a reader should bring to it** — one sentence, plain language;
4. **what it deliberately does NOT cover**, and which of the other two does.

Then **one comparison table**, three columns, answering: *"where do I look for X?"* across the
subjects that actually recur —

pipeline structure · why a design decision was made · what a document contains · what Beta ruled ·
current fleet/topology state · a specific `file:line` anchor · what is frozen and must be reused ·
what is superseded · verification rules (VIR) · past mistakes and their cost.

**Then name any subject covered by two or more of them**, and say **which should be authoritative**
— overlap is where the next contradiction comes from.

**Plain language. Michael reads this to decide where to look, not to audit it.**

---

## What NOT to do

- **Do not edit the skill.** Do not create any new file.
- Do not restructure, reorganise or "improve" either file beyond A1–A3.
- Do not fix anything else you notice — **report it in Part B instead.**
- Do not run the pipeline, WATCHER or any scraper. Do not touch `miner/`.
- **Do not commit or push.**

⚠ Both edits dirty the tree. **Tell Michael they must be committed before the Phase-7 soak
launches.**

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** quote the `git show --stat` output for **both** `81ef3f1` and `ef4b1c6`. The
  correction rests on it.
- **clean control:** after editing, re-grep both files for `81ef3f1` and for `26 GPU` and show the
  remaining hits with a one-line justification each. **A remaining hit is fine if it is correct —
  silence is not.**
- **fault-injection control:** `NOT_APPLICABLE` — document edits, not a detector. **Write
  `NOT_APPLICABLE`, never `PASS`.**
- **completion sentinel:** `PASS | FAIL | UNAVAILABLE | INCOMPLETE`.
- **audit claim scope:** repo-scoped at the stated HEAD.
- **searched surfaces / unavailable surfaces:** name them.
