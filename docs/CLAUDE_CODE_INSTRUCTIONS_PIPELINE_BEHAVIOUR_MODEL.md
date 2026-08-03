# CLAUDE_CODE_INSTRUCTIONS_PIPELINE_BEHAVIOUR_MODEL.md — REV1

**Build a verified model of how the pipeline actually works, sourced from the documentation and
checked against the code.**

**Base:** `1fc05bb` or later. Claude Code on **VM101** as `michael`, venv `~/venvs/torch`,
**run from `/home/michael/distributed_prng_analysis`**.

**Deliverable:** `docs/PIPELINE_BEHAVIOUR_MODEL.md`. **This is the ONLY file you create.**

---

## 0. What this is, and what it is NOT

**This is NOT an audit.** Audits ask *"what is broken?"* and this project has now learned, twice,
that an auditor without the governance trail generates false findings at a high rate.

**This asks: "how does it work, and why is it built that way?"**

The output is the **explanation** a new session needs in order to reason about this system without
re-deriving it — and without concluding that something is dead because its purpose is not visible
from the code.

> **The governing fact, and it has survived every test put to it:**
> **EVERY LINE OF THIS CODEBASE HAS BEEN DOCUMENTED.**
> **A component whose purpose you cannot see means the explanation has not been found yet.**

**Read `docs/PROJECT_FILE_CATALOG.md` FIRST.** It is intent-indexed — it tells you which document
answers which question. It exists because two prior efforts failed by searching code and treating
documentation as commentary.

## 1. Method — binding

**Search order: governance trail → chapters → code. Never code-first.**

For every behaviour you describe, give **two anchors**:

| anchor | what it establishes |
|---|---|
| **WHY** — a chapter, whitepaper, proposal or TB ruling, with `file:§` | the intent |
| **WHAT** — a source location, with `file:line`, read this session | the implementation |

**A behaviour with only a code anchor is INCOMPLETE — mark it so.** That gap is itself the finding:
it means either the explanation exists somewhere you did not look, or it genuinely is not written
down. **Do not resolve the ambiguity by assuming the second.**

**When documentation and code disagree: record BOTH and mark it DIVERGENT. Do not decide which is
right, and do not change either.** Chapter 3's audit found the chapter describing v4.2 while the
worker ran v4.3 — the divergence was the finding; adjudicating it was not the auditor's job.

## 2. Scope — the whole pipeline, step by step

For **each of steps 0–6**, and for the **RANGE-MINER engine** inside Step 1:

- **Purpose** — what it is for, in the project's own terms
- **Inputs** — artifacts and parameters, and where they come from
- **Outputs** — artifacts, and who consumes them
- **The seam** — what contract binds it to the next step *(e.g. the 22-array NPZ, which exists so
  downstream cannot tell which engine produced the survivors)*
- **Why it is built this way** — the design decision and its source
- **Known governed issues** — anything the trail shows as diagnosed, escalated, or mid-remediation.
  **Cite the ruling. This is what prevents a later reader re-reporting it as new.**

**Also cover, as first-class subjects rather than footnotes:**

- **the bidirectional sieve** — why bidirectional, why the intersection matters
- **skip** — why it exists at all *(the physical model: pre-test draws, per-session equipment
  selection; source `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` §5.1 and the CA draw-procedures
  document — note that the PDF is **citation-only**, not in the repo)*
- **the survivor manifold → ML → prediction path**
- **WATCHER's autonomy model** — what it decides alone and what it escalates
- **the frozen authorities** — `_l2_sort_key`, `_select_l2_winners`, `CANONICAL_ARRAY_CONTRACT`,
  `utils/prng_encoding`, `canonical_map_hash`, the three finalizer validators. **State what each
  guarantees and why reimplementing it is forbidden.**

## 3. The whitepaper and foundational documents

Find and read them — catalog §3.1 indexes them under *"why the design is what it is."*

**Record what the project CLAIMS, precisely**, because it is repeatedly restated imprecisely:

- **TFM is functional mimicry of a deterministic PRNG surface** — **black-box mimicry, NOT state
  recovery, NOT seed reconstruction.** *(Catalog finding: `docs/README.md` and
  `docs/proposals/README.md` open with "Seed Reconstruction" / "reverse-engineer PRNG behavior",
  against this rule. **Record the divergence; do not fix it.**)*
- **`holdout_hits` is the designated falsification criterion.** State what would falsify the thesis.
- **Seed-Domain v1.1** — uint32 storage, the `high16=0` stratum, 1 part in 65,536, and **why honest
  stratum labelling was the approved resolution rather than a uint64 migration.**
- **Sieve selectivity** — why the bidirectional intersection squaring the exponent is what makes
  survivor validity rest on mathematics rather than on search extent.

## 4. What NOT to do

- **Do not audit.** No claim that anything is broken, unused, dead, or unwired. If you notice
  something, put it in a short §Observations list **with the searches you performed**, and label it
  an observation, not a finding.
- **Do not fix anything.** No code, no config, no chapter, no manifest.
- **Do not run the pipeline, WATCHER, or any scraper.** Do not invoke
  `convert_survivors_to_binary.py` — D3.0-B is open and Beta prohibits it.
- **Do not touch `miner/`** — S172 work is in flight.
- **Do not dispatch to CT100 workers or rigs.** VM101 only. If an invocation path would distribute,
  stop and report.
- **Do not commit or push.**

## 5. Length and shape

**Aim for something a new session can read in full.** Depth belongs in the chapters — this points
at them and explains how the pieces fit.

Prefer: a **one-page pipeline overview** · then **one section per step** · then the cross-cutting
subjects in §2 · then §Observations · then a short **"what a new session most needs to know"** list.

**If a step's documentation is thin, say so plainly and name what is missing.** A short honest
section beats a long inferred one.

## 6. Report

In your final message, not a second file: which documents you read for each step; where you found
only a code anchor and no WHY; every DIVERGENT pair; and anything in §Observations. Then **STOP**.

⚠ **Your output file will be untracked and dirties the tree.** The Phase-7 soak's clean-tree
preflight rejects a dirty tree at finalization — **tell Michael it must be committed before the
soak launches, or before a running soak reaches publication.**

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** count the documents read per step and state it. **A model written from
  filenames is worthless and indistinguishable from one written from reading unless you show the
  count.**
- **clean control / fault-injection:** `NOT_APPLICABLE` — this produces an explanation, not a
  detector. **Write `NOT_APPLICABLE`, never `PASS`.**
- **completion sentinel:** `PASS | FAIL | UNAVAILABLE | INCOMPLETE`. **A step you could not
  document from sources is `INCOMPLETE` for that step** — an acceptable and useful outcome.
- **audit claim scope:** repo-scoped at the stated HEAD. **`agent_manifests/definitions.json` is
  untracked and absent from a fresh clone — if you read it, say that it is host-only.**
- **searched surfaces:** `docs/` **and** the governance trail **must appear** (VIR-6 addendum), plus
  every code path you anchored.
- **unavailable surfaces:** the CA draw-procedures PDF (citation-only, not in repo) · host state
  beyond VM101 · anything gitignored — **run `git check-ignore` before calling any config absent.**
