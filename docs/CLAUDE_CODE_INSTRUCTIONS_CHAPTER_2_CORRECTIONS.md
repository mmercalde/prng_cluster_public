# CLAUDE_CODE_INSTRUCTIONS_CHAPTER_2_CORRECTIONS.md — REV1

**Chapter 2 — three factual corrections required for documentation closure.**

Team Beta **accepted** the restoration and audit. Closure requires three corrections, and **the
first is an error Alpha introduced and propagated.**

**Documentation only.** No code, tests or config. Small — three edits and their downstream
copies.

**Base:** current `main` on VM 101. Claude Code as `michael`, venv `~/venvs/torch`. You do **NOT**
commit, push, or run WATCHER. STOP at the gate.

**Concurrency:** an admission-binding session may be running against `miner/`, `execution_set.py`
and `agents/`. This brief touches documentation only.

---

## 1. Correction 1 — "two pre-test draws" is wrong, and Alpha put it there

**Alpha's claim, repeated in Chapter 2 §5.1, the skill §0.4, and three submissions:**

> *Two pre-test draws run before every live draw.*

**Beta's finding:**

> **Unsupported and appears incorrect for automatic Daily draws.** The official procedure
> describes **one automatic pre-test session**, with additional pre-tests only for anomalies.
> The "two test draws" language applies to **manual SuperLotto Plus equipment.**

Source: *California Lottery Daily & SuperLotto Plus Draw Procedures*
`https://static.www.calottery.com/-/media/Project/calottery/PWS/PDFs/RFP-Documents/DAILY--SLP--06-2021_for-RFP-ij.pdf`

**Alpha misread the document** — the two-test language belongs to a different draw type — and it
propagated into the chapter, the skill and three Beta submissions.

**Correct in `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` §5.1**: one automatic pre-test session for
automatic Daily draws, additional pre-tests only on anomalies, and note that the two-test
procedure applies to manual SuperLotto Plus equipment.

**Then find and correct every downstream copy.** Grep for `pre-test`, `pretest`, `two pre-test`,
`two test draws` across `docs/`. **Do not edit the skill** (`~/.claude/skills/…`) — it is outside
the repo; report that it needs the same fix and Michael will apply it.

**The correction does not weaken the rationale.** One unpublished pre-test session still produces
unpublished outputs, and per-session equipment selection and co-drawn evening games are
unaffected. **Skip is still physically motivated. Only the count was wrong.**

## 2. Correction 2 — qualify the physical-gap inference

**Beta:**

> The procedures establish equipment selection, an unpublished pre-test, and co-drawn evening
> games. **They do not establish that every omitted output belongs to one uninterrupted PRNG
> state stream.** Describe these as **physically motivated candidate gaps supporting skip as a
> detector** — not proven state advances.

That distinction matters and Chapter 2 currently overstates. The procedures show outputs are
consumed and not published; they do **not** show that the published values and the unpublished
ones form one continuous advance sequence from a single generator.

**This is consistent with §5.6's own framing** — variable skip is a **detector** looking for
windows where coherent structure surfaces, not a reconstruction of state. Correct §5.1 to match
§5.6's epistemics: **candidate gaps, physically motivated, not proven advances.**

## 3. Correction 3 — `full_state`, again

Chapter 2 calls it a *"forward-compatibility seam."* **Wrong, and it is the second time this
field has been mischaracterised.**

**Beta:** it is the **deliberate synthetic known-answer / multi-modulo validation hook**
identified in the Wall C ruling — the fixture generator's
`"full_state": int(state)  # Critical for multi-modulo validation`. **Inert in the live dataset,
but its purpose is known.**

**Also in the same area:** do **not** say the sieve compares full 32-bit values. **The active
predicate still reduces them modulo 1000 / 8 / 125** (§6's three lanes). Check §6 does not
contradict this.

## 4. One framing note Beta added

The fingerprint framing (§5.6) is **accepted as Michael's governing design intent** and now has
an appropriate permanent home. Present it as **design doctrine corroborated by earlier
artifacts** — *not* as a historically discovered repository statement. Its current NOT-FOUND
table is honest and should stay; what changes is the framing of *why it belongs here*: it is
doctrine being recorded, not evidence being reported.

## 5. F-4 and F-5 — both confirmed, neither to be repaired here

**F-4 (offset coupling) — CONFIRMED.** The host slices records using `offset` while constant and
reverse kernels also pre-advance state by the same flat `offset`; with skip N, one record is N+1
advances, so the two align only at skip zero. **It settles Chapter 1 C-2 as an observed
inconsistency — it does NOT settle the repair**, especially for variable skip where no single
`offset*(skip+1)` multiplier exists. **F-4 belongs inside the future hybrid input-semantics
design, not a standalone arithmetic patch.** Ensure the chapter says that.

**F-5 (ROCm hostname guard) — CONFIRMED dead legacy branch**, and **not** a Phase-7 blocker.
**Do not rename the hosts** — that would activate obsolete ROCm environment overrides the current
rigs reportedly do not need. A later repair must first decide whether the prelude is still
supported: if obsolete, remove it with its historical explanation preserved; if retained, key it
from an **explicit platform/profile property** and test it — **never from another handwritten
hostname tuple.** Record that disposition in the chapter.

## 6. Out of scope

- **No code, tests, config or manifests.**
- Do not repair F-4 or F-5.
- Do not touch other chapters, the skill file, or the execution-set work.
- Do not re-audit anything — this is three corrections and their downstream copies.

## 7. Verification-integrity controls (VIR-1…6)

- **execution proof** — the corrected pre-test claim cites the procedures document by section;
  the `full_state` correction cites the fixture generator.
- **clean control (VIR-2)** — state which parts of §5.1 and §6 you verified as **correct and
  unchanged**. This is a correction pass; a report listing only edits gives no evidence the rest
  was checked.
- **fault-injection control** — n/a for documentation; **say so** rather than omitting it.
- **completion sentinel** — explicit `PASS | FAIL | UNAVAILABLE | INCOMPLETE`.
- **unavailable-observer** — the PDF is **not in the repo**; if you cannot read it, say so and
  correct from Beta's ruling text, marking the citation `UNAVAILABLE` rather than implying you
  verified it at source.
- **audit claim scope** — list every file changed and every downstream copy found.

## 8. Report

The three corrections with before/after text. **Every downstream copy found and corrected**, and
an explicit list of any that are outside the repo and need Michael to apply (the skill). The F-4
and F-5 dispositions as recorded. Confirmation that no code was touched. Then STOP.
**Do not commit.**
