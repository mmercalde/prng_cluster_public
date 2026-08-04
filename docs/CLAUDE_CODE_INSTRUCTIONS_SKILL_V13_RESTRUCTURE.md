# CLAUDE_CODE_INSTRUCTIONS_SKILL_V13_RESTRUCTURE.md — REV1

**Restructure `docs/TFM_PROJECT_FACTS_SKILL.md` from v12 (1,047 lines) to v13.**

**Base:** HEAD `3b9286c` or later. Claude Code on **VM101** as `michael`, venv `~/venvs/torch`,
**run from `/home/michael/distributed_prng_analysis`.**

**Deliverable:** `docs/TFM_PROJECT_FACTS_SKILL.md`, edited in place. **The ONLY file you change.**

---

## 0. What the skill is FOR, and what it must stop being

The skill loads into every session at start. It is the only thing a fresh session knows before it
reads anything.

**Two new artifacts now exist that did not when v6 was written:**

| artifact | authority on |
|---|---|
| `docs/PROJECT_FILE_CATALOG.md` (803L) | **what documents exist and what question each answers** |
| `docs/PIPELINE_BEHAVIOUR_MODEL.md` (1,603L) | **how the pipeline works and why**, every claim with a WHY anchor and a WHAT anchor |

> **THEREFORE: skill = judgement, rules, and pointers. Those two = the facts.**
> Where the skill restates a fact those documents carry with anchors, **the skill should point
> instead** — it has no anchors, so its version is strictly weaker.

**The skill grew ~400 lines in one day across six revisions, by accretion.** This pass makes it
smaller and sharper. **Target: under ~900 lines. Do not treat that as a quota** — if a section
earns its place, keep it.

## 1. ⚠ PRESERVATION-FIRST — the binding constraint

**Every fact in v12 was put there because it cost something to learn.** Several are corrections to
errors that were made, submitted, and returned.

> **DELETE NOTHING BECAUSE IT LOOKS REDUNDANT.**
> **Replace a fact with a pointer ONLY when you have verified, by opening the target this session,
> that the target states the same fact AND carries at least as much anchoring.**
> **If the target is weaker, vaguer, or missing the fact — KEEP THE SKILL'S VERSION.**

**Before removing or condensing anything, state in your report: what it was, where it now lives,
and the `file:§` you verified.** A removal without that line is not permitted.

**Never remove:** anything marked ⚠ · anything phrased as a correction to a prior claim · anything
naming a specific past failure with its cost · §1 THE RULE · §5 VIR · §9 SELF-CHECK.

## 2. Required changes

### 2.1 §0.6 — the highest-priority fix in this brief

The pipeline diagram at `:124-134` is the **first** pipeline description a session reads, and it is
ambiguous in exactly the way that produced a launch-day hazard.

Problems, all verifiable at source:
- it uses the **conceptual** numbering (`Step 2 = Bidirectional Sieve`) **without saying so**;
- **`PIPELINE_BEHAVIOUR_MODEL.md` §1.1 states this is *"the single most common error a new session
  makes"***;
- it **omits Step 0 (TRSE)** and **Step 4 (ML Meta-Optimizer)** entirely;
- it says **26 GPUs**; the owner has ruled **25** for the Phase-7 soak.

**Required:** show **both** schemes, side by side, labelled. Include **all** of steps 0–6 in the
executable scheme. State that **bare "Step N" in this skill means the EXECUTABLE scheme.** Point to
`PIPELINE_BEHAVIOUR_MODEL.md` §1.1 and §1.3 for the full map. Fix the GPU count.

**Use `PIPELINE_BEHAVIOUR_MODEL.md` §1.3 as the source** — it was parsed live from
`agents/watcher_agent.py:386-416` and all seven manifests. **Do not retype it from memory.**

### 2.2 §1.1 taxonomy — add the patch corpus

The taxonomy table lists document categories. **It is missing the surface that answered I-6.**

Add a row: **`apply_s*.py` / `verify_s*.py` (repo root)** — *"the one-shot patch corpus.
**Their docstrings quote TB rulings verbatim.** `apply_s142_partition_runid.py` records the
TB-confirmed root cause of the partition `run_id` collision; `apply_s142c_remove_worker_writes.py`
records TB Option A superseding it. **Governance lives in code here, and no taxonomy named it until
2026-08-03.**"*

**State plainly that this surface was missing from `PIPELINE_BEHAVIOUR_MODEL.md` §17's own
"where to look next" list, and that I-6's answer was in it anyway.** That is the lesson, not the
row.

### 2.3 §1.1 — the catalog and behaviour model as mandatory first reads

Both already appear. **Make the behaviour model's status explicit:** it is where a session goes to
learn **how the pipeline works**, and every claim in it carries a WHY and a WHAT anchor.

### 2.4 §2.17 — move it out

*"Fleet state as launched — measured 2026-08-02"* is **operational state, not a durable fact.**
It will be stale within a week and will read as authoritative.

`docs/PHASE6_PREREQS.md` REV5 already carries it. **Verify that before removing** — open REV5,
confirm the frozen `set_id`, the DHCP reservation and the rig parity are all there, and **cite the
sections in your report.** If anything is in the skill and NOT in REV5, **keep it in the skill**.

**Leave a one-line pointer** so a session knows where the fleet state lives.

### 2.5 Sections that may become pointers — each requires the §1 check

Candidates only. **For each, open the target and verify before touching:**

- **§0.6** — see §2.1 (fix in place, then point)
- **§2.13 control chains** — much of this is in the behaviour model with anchors
- **§2.8 RANGE-MINER Phase 5 as-built** — check against behaviour model §3.7
- **§2.11 fleet authority (six mechanisms)** — check `FLEET_STATE_REQUIREMENTS_v1.md`
- **§3 SUPERSEDED** — check catalog §6, which covers the same ground

**Any candidate that fails the check stays exactly as it is.**

### 2.6 Currency

Update the currency line to the HEAD you actually work at.

## 3. What NOT to do

- **Do not touch any other file.**
- Do not fix, audit, or "improve" anything the skill describes. **This is an edit of one document.**
- Do not run the pipeline, WATCHER, or any scraper. Do not touch `miner/`.
- **Do not commit or push.**
- **Do not soften a warning to save lines.** ⚠ blocks exist because the thing happened.

## 4. Report

In your final message:
- line count before → after;
- **every removal or condensation, with what it was, where it now lives, and the `file:§` you
  verified this session** — §1 requires this and a report without it is incomplete;
- anything you were told to condense but **kept**, and why;
- the §0.6 replacement in full, so it can be reviewed without opening the file;
- anything you found stale that this brief did not name.

Then **STOP** for Team Alpha review.

⚠ The file is tracked, so this dirties the tree as a modification. **Tell Michael it must be
committed before the Phase-7 soak launches.**

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** name every target document you opened to justify a pointer. **A pointer
  written without opening its target is exactly the failure this skill exists to prevent** — see
  §1.2, *a report is a snapshot*.
- **clean control:** after editing, re-read the file end to end and confirm no section references a
  section that no longer exists, and no ⚠ block was weakened.
- **fault-injection control:** `NOT_APPLICABLE` — a document edit, not a detector. **Write
  `NOT_APPLICABLE`, never `PASS`.**
- **completion sentinel:** `PASS | FAIL | UNAVAILABLE | INCOMPLETE`.
- **audit claim scope:** repo-scoped at the stated HEAD.
- **searched surfaces:** the skill, the catalog, the behaviour model, `PHASE6_PREREQS.md`, and every
  target you verified.
- **unavailable surfaces:** anything you could not open — **say so rather than assuming the pointer
  is safe.**
