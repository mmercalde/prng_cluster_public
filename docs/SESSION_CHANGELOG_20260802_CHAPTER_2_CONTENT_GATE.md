# SESSION CHANGELOG — 2026-08-02 (Chapter 2 content gate)

**Focus:** Build the executable gate for Chapter 2 that the closure pass found missing.
**Outcome:** `tests/test_chapter2_content_gate.py` — 6 gates + 6 mutants, **12/12, SENTINEL: PASS**.
All six gates additionally proven red against the **actual** `248e48c` fragment.

> **Follow-up to `ef4b1c6`** (*"docs: Chapters 1 and 2 closed…"*) and its changelog
> `docs/SESSION_CHANGELOG_20260802_CHAPTER_1_AND_2_CLOSURE.md`. That record is **left
> byte-intact**: it was committed before this work existed, and retroactively editing a
> committed record falsifies the audit trail. This is a separate deliverable changelog, per the
> established convention (2026-08-01 carries six).

---

## Summary

Chapter 1's closure statement recorded, as a governance finding, that **no executable gate
covered Chapter 2** — Chapter 1 had `tests/test_chapter1_p0_corrections.py` and Chapter 2 had
nothing, so its claims were protected by review only. This deliverable closes that gap.

The gate is scoped to **content presence and source agreement, deliberately not prose style**:
every assertion is either "this load-bearing statement is still here" or "this number still
matches the source it claims to describe". Rewording a paragraph must not red it. Deleting the
paragraph must.

Three defects surfaced — **all three in the harness, none in the chapter** — and each was
repaired by strengthening the detector rather than loosening the assertion.

---

## The failure mode this defends against

`248e48c` (*"chore: move CHAPTER docs to docs/ folder"*, 2026-01-29) copied a **34-line
fragment** over `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md`, deleting **709 lines**. Nobody read
either file. In the interim, Team Alpha, Team Beta and Claude Code **independently** recommended
deleting `skip_min`/`skip_max` — a cornerstone of the design — because the document explaining
why skip exists no longer existed to be read (§5.1.1). Michael stopped the removal.

**A gate is a message to whoever does not open the file.** That framing set the scope.

---

## Work Completed

| Item | Status | Notes |
|------|--------|-------|
| `G-SKIP-RATIONALE` | ✅ | §5.1 + §5.1.1: the **WIRE-IN rule** verbatim, the three-party near-removal record, *"Michael stopped the removal"*, the procedural facts, the *candidate gaps ≠ proven state advances* qualification, the `UNAVAILABLE` citation marker, and a stub-detector on §5.1's body length. **The most important gate in the file** |
| `G-LANE-COUNT` | ✅ | §6's CRT construction present; the declared count **re-derived from live `prng_registry.py`** by the §6.2.1 method; the single-lane exception resolved to its real owning kernel; the verbatim block matched against source |
| `G-HYBRID-DEFECT` | ✅ | §5.4's callout **and** its *non-certifying* consequence; then **AST-parses** `_hybrid_prefix` / `_constant_prefix` to confirm the defect is still true of source. If the defect is ever repaired, the gate reds to say the chapter is stale |
| `G-SOURCE-ANCHORS` | ✅ | every `file.py:line` resolves in range; `a`/`c` hardcoded in exactly **two** kernels, both reverse; the `default_params` anchor really points at `default_params`; §7.2's two `offset` consumers still exist |
| `G-CLOSURE-INTEGRITY` | ✅ | §14, the sentinel, **F-1…F-8**, F-4's *"NOT the repair"*, F-5's *"do not rename the hosts"*, the VIR-6 declaration, §12.1's inherited items |
| `G-CHAPTER-SUBSTANCE` | ✅ | **the 248e48c gate.** Line/byte floors, 12 required sections, plus per-section body floors so gutting bodies under surviving headings still reds |
| Six mutants (M1–M6) | ✅ | one per gate, each asserting its gate reds **for the right reason**; written to tempfile copies, never into the repo |
| Proof against the real historical artifact | ✅ | see below — stronger than M6's synthetic truncation |
| `gate22_coexistence` registration | ✅ | the new `.py` reddened it; registered in the whitelist per house style |
| Non-regression | ✅ | Phase-4 **63/63** (pinned tally unmoved), Chapter 1 gate **12/12**, Chapter 2 gate **12/12** |

---

## Files Created/Modified

| File | Action | Notes |
|------|--------|-------|
| `tests/test_chapter2_content_gate.py` | **Created** | 6 gates + 6 mutants. Read-only: imports no pipeline module, executes no CLI, touches no GPU/rig/fleet/pipeline |
| `tests/test_s172_phase4_coordinator.py` | Modified | **Whitelist registration only** — one entry + its rationale comment in `gate22_coexistence`'s `allowed` set. No gate added, no gate changed; the 63/63 tally is unmoved |
| `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` | Modified | **+22 lines.** §6.2.1 gains a machine-readable `<!-- BEGIN LANE TEST COUNT -->` block, mirroring Chapter 1's bounds-snapshot convention, so the count is checkable without gating prose |
| `docs/SESSION_CHANGELOG_20260802_CHAPTER_2_CONTENT_GATE.md` | Created | this file |

**Untouched:** no kernel, sampler, threshold, protocol, dataset-authority, execution-set, PWC or
ZMQ path. No production module of any kind.

---

## Verification performed

**The decisive proof — the gate run against the actual destroying commit,** recovered with
`git show 248e48c:docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` (34 lines; a fragment of the old §14
GPU-cleanup section):

```
RED  G-SKIP-RATIONALE     §5.1 heading is gone — the physical model section was removed
RED  G-LANE-COUNT         §6 heading is gone
RED  G-HYBRID-DEFECT      §5.4's defect heading is gone
RED  G-SOURCE-ANCHORS     only 0 anchors checked — the chapter lost most of its citations
RED  G-CLOSURE-INTEGRITY  the §14 closure statement is gone
RED  G-CHAPTER-SUBSTANCE  CHAPTER TRUNCATED: 35 lines, expected >= 1000
6 of 6 gates red against the historical fragment
```

This is why the proof is not left at M6: M6 *synthesises* a truncation, whereas the above is the
real payload that actually destroyed the chapter.

```bash
cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate

python3 tests/test_chapter2_content_gate.py       # 12/12, SENTINEL : PASS
python3 tests/test_chapter1_p0_corrections.py     # 12/12, SENTINEL : PASS (unchanged)
python3 tests/test_s172_phase4_coordinator.py     # 63/63 (pinned tally unmoved)
```

Fleet at time of running: `.122` / `.156` / `.164` **UP**. Noted only because the Chapter 1
harness needs it — **this harness does not** (see Decisions).

---

## Issues / Incidents

All three were defects in the **harness**, not the chapter. Each was fixed by making the detector
stronger, never by weakening the assertion.

| Issue | Resolution |
|-------|------------|
| `G-LANE-COUNT` red: *"§6.2's verbatim block lost a lane"* | **False positive.** The chapter pads the quoted block for column alignment (`%    8`) while source does not (`% 8`), so a literal comparison could never match. Now compared on **whitespace-normalised** text — the same comparison with layout removed, not a looser one |
| `G-HYBRID-DEFECT` `UNAVAILABLE`: *"cannot import range_miner_worker"* | Importing it pulls in `sieve_gpu_worker`, whose import **replaces `sys.stdout`**. Replaced the import with an **AST parse** of the returned list literals — no side effects, and a *stronger* detector: a skip bound appearing in a comment or docstring can no longer satisfy it |
| `G-LANE-COUNT` red on the chapter's own **withdrawal** of "39" | The chapter legitimately quotes "39 occurrences" in §6.2.1, in the VIR-6 amendment and in the version history, in order to withdraw it. A bare substring ban would have forbidden the historical record. The check now requires **withdrawal vocabulary in context**, so quoting is permitted and re-assertion is not |
| New `.py` reddened `gate22_coexistence` | Confirmed empirically (`unexpected changed .py files: {'tests/test_chapter2_content_gate.py'}`), then registered in the `allowed` set with a rationale comment in the established house style. Re-verified gate22 **GREEN** and the full harness **63/63** |

---

## Decisions Made

- **Scoped to content presence and source agreement, not prose style.** Every assertion is either
  a presence check on a load-bearing statement or a live re-derivation from source. This is a
  deliberate ceiling: a gate that pins phrasing would red on every legitimate edit and be
  disabled within a month, which would leave the chapter exactly as unprotected as it was.

- **No fleet dependency, by design — and this is a deliberate divergence from the sibling
  harness.** Four of Chapter 1's twelve arms shell out to the real CLI; during the closure pass
  they went red because the rigs were powered off, which is a **fleet** fact and not a
  **documentation** fact (recorded in Chapter 1 §17.2). This harness imports no pipeline module
  and runs no subprocess, so **a red here always means the chapter or its source agreement
  actually changed.** The rationale is written into the module docstring and the whitelist entry
  so the difference is legible to whoever reads either file next.

- **Gates re-derive rather than trust.** `G-LANE-COUNT` recounts the registry, `G-HYBRID-DEFECT`
  AST-parses the prefix builders, `G-SOURCE-ANCHORS` resolves every citation against disk and
  re-derives the two-kernel `a`/`c` fact. A chapter gate that only read the chapter would confirm
  the document is self-consistent while it drifts away from the code — the exact failure the
  anchor drift in both chapters has already produced twice.

- **The gates red when the SOURCE changes too, not only the prose.** If `_hybrid_prefix` ever
  emits a skip bound, `G-HYBRID-DEFECT` reds with *"SOURCE CHANGED … §5.4 is stale"*. That is
  intended: a repaired defect makes the chapter wrong, and the gate should say so rather than
  silently pass.

- **A machine-readable block was added to §6.2.1** rather than regexing prose for the number.
  Mirrors Chapter 1's `<!-- BEGIN EXTRACTED BOUNDS SNAPSHOT -->` convention. It carries an
  in-document warning: *do not hand-edit it to make a gate pass; re-run the method and fix the
  prose with it.*

- **The committed closure changelog was NOT edited.** `ef4b1c6` landed mid-session; amending its
  changelog to describe work that did not exist when it was written would falsify the record.
  Hence this separate file.

---

## Git Commands

> **Not run by the agent.** Per CLAUDE.md §1.1/§1.2 Michael commits and dual-pushes.

```bash
cd ~/distributed_prng_analysis
git add tests/test_chapter2_content_gate.py \
        tests/test_s172_phase4_coordinator.py \
        docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md \
        docs/SESSION_CHANGELOG_20260802_CHAPTER_2_CONTENT_GATE.md

git commit -m "test: Chapter 2 content gate — the executable gate the closure pass found missing

Chapter 1 is covered by tests/test_chapter1_p0_corrections.py; Chapter 2 was
covered by nothing, so its claims were protected by review only. Chapter 1 §17.3
records that as an open governance item; this closes it.

Six gates + six mutants (12/12 PASS), scoped to content presence and source
agreement rather than prose style:
  G-SKIP-RATIONALE     §5.1/§5.1.1 incl. the WIRE-IN rule and the three-party
                       near-removal record — the section that did not exist to
                       be read when three parties reasoned toward deleting
                       skip_min/skip_max
  G-LANE-COUNT         the declared count RE-DERIVED from live prng_registry.py
                       by the published §6.2.1 method (currently 43 of 44)
  G-HYBRID-DEFECT      §5.4's callout, plus an AST re-derivation of the prefix
                       builders — reds if the defect is repaired and the chapter
                       goes stale
  G-SOURCE-ANCHORS     every file:line resolves; the closure-corrected §1.1 and
                       §7.2 facts re-derived from source
  G-CLOSURE-INTEGRITY  §14, the sentinel, F-1..F-8, VIR-6, §12.1
  G-CHAPTER-SUBSTANCE  the 248e48c gate — line/byte floors and per-section body
                       floors

Proven, not asserted: all six gates go RED against the ACTUAL post-destruction
file recovered by 'git show 248e48c:docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md'.

Unlike its Chapter 1 sibling this harness has NO fleet dependency — it imports no
pipeline module and runs no CLI, so a red always means the chapter or its source
agreement changed, never that the rigs are off.

Also: gate22_coexistence whitelist registration for the new test path (no gate
added or changed; Phase-4 stays 63/63), and a machine-readable lane-count block
in §6.2.1 so the number is checkable without gating prose.

No production code touched."

git push origin main && git push public main
```

---

## Hot State (Next Session Pickup)

**Where we left off:** Gate built, green, and proven against the historical artifact. Working
tree has one new test, one whitelist registration, one +22-line chapter edit and this changelog.
Nothing committed by the agent, nothing pushed, no pipeline run.

**Next action:** Michael reviews and dual-pushes (commands above).

**Blockers:** None.

**Worth knowing:**
- **Chapter 2's closure sentinel says its fault-injection control is `NOT_APPLICABLE` because no
  gate existed.** That is now false in the good direction. §14.3 lists *"No executable gate covers
  this chapter"* as an open governance item, and **it can be marked discharged** — a small
  documentation follow-up, deliberately not done here so the gate lands as its own reviewable
  unit.
- Chapter 1 §17.3 carries the same item in its open-items table and would be updated in the same
  pass.
- The gate is **hand-run**, like every other harness in `tests/`. Nothing schedules it. If it
  should run automatically on a docs change, that is a separate decision for the gate owner.
- Everything else from `ef4b1c6` remains open exactly as its changelog records — the closure pass
  repaired nothing, and this deliverable repairs nothing either.

---

*End of deliverable — Chapter 2 content gate, 2026-08-02*
