# CLAUDE CODE INSTRUCTIONS — PROJECT CATALOG REGENERATION, 2026-08-20 PASS

**Authority:** `docs/CLAUDE_CODE_INSTRUCTIONS_PROJECT_CATALOG_REGENERATION.md` (REV2).
**REV2 governs in full — read it first and follow it exactly.** This file adds only what is
specific to this pass: the anchor, the currency delta, and the items that must not be missed.
Where anything here appears to conflict with REV2, REV2 wins.

**Host:** VM101, user `michael`, `/home/michael/distributed_prng_analysis`,
`source ~/venvs/torch/bin/activate`.
**Deliverable:** `docs/PROJECT_FILE_CATALOG.md`, replacing the current one. **One file. No
others.** Do not commit, do not push (deny-rules block it anyway).

---

## 1. Why now

The current catalog was regenerated **2026-08-03 at HEAD `9e79a26`** and last corrected
`f8cb1c5` the same day. HEAD is now **`1bf49a5`** — seventeen days and roughly forty commits
later, including the entire Gate-12 pass, the drain-remedy series, the field-6 repair, and
four Beta rulings.

The catalog's own header states why staleness matters here: it exists because Alpha twice
claimed something was undocumented when the answer sat committed, and **once nearly
submitted a finding to Beta that Beta had already ruled on**. §1.1 (the governance trail) is
the section that failure would have used. Since 2026-08-03 the governance trail has grown by
more than it contained for months. A stale §1.1 reproduces exactly the failure the catalog
was built to prevent.

## 2. Anchor and coverage

- Regenerate **at current HEAD**, and state that HEAD (`git rev-parse --short HEAD`) plus the
  regeneration date in the replacement header, exactly as the current one does.
- Re-count `docs/` (`ls docs/ | wc -l` plus subdirectories) and state indexed-vs-total in the
  report per REV2 §2 and §5. The previous pass indexed 562; do not carry that number forward —
  measure it.
- **Open every file you index.** REV2 §2 is explicit and this pass is where it matters most:
  many of the new documents are rulings whose *dispositions* are the content, and a filename
  cannot tell you whether a ruling accepted, rejected, or superseded something.

## 3. Must not be missed in §1.1 (the governance trail)

These landed after the last catalog and each is binding today. Pair each ruling with the
request or report it answers, per REV2 §1.1:

| file | pair with |
|---|---|
| `TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md` | `SESSION_CHANGELOG_20260817_GATE12_ATTEMPT9.md` |
| `TB_RULING_REQUEST_WINDOW_ANCHOR_SEQUENCING.md` | → `TB_RULING_WINDOW_ANCHOR_SEQUENCING.md` |
| `TB_RULING_REQUEST_CHANGELOG_NUMBERING.md` | → `TB_RULING_CHANGELOG_NUMBERING.md` |
| `PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_0.md` | → `TB_RULING_WINDOW_ANCHOR_PROPOSAL_V1_0.md` → **superseded by `_v1_1.md`** |
| `CLAUDE_CODE_INSTRUCTIONS_FIELD6_OBSERVABILITY_REPAIR.md` | → `CLAUDE_CODE_REPORT_FIELD6_OBSERVABILITY_REPAIR.md` → `TB_RULING_FIELD6_IMPLEMENTATION.md` → implemented `d8b21e3` |
| the R-1..R-4 / MP-1 drain series | its rulings and `e9ca800` |

Also index: `SESSION_CHANGELOG_20260817_GOVERNANCE_RULINGS.md`,
`SESSION_CHANGELOG_20260820_FIELD6_AND_WINDOW_ANCHOR_DESIGN.md`, and BACKLOG **§19**
(accumulator backup/recovery — ruled real, non-blocking, unowned).

**Two STANDING RULES now exist and currently live only inside ruling documents. They must be
findable from the catalog** — a future session that misses either will red a certified gate
or violate a naming ruling without knowing why:

1. **`DECLARED_CHANGED` maintenance** (`TB_RULING_FIELD6_IMPLEMENTATION.md` §2) — any
   authorized commit changing a `miner/range_miner_coordinator.py` definition covered by a
   historical exact live-vs-anchor scope gate MUST update every affected `DECLARED_CHANGED`
   set before acceptance; anchor never moves; `changed == DECLARED_CHANGED` never relaxed;
   only actually-changed definitions added; each entry carries provenance.
2. **Changelog naming** (`TB_RULING_CHANGELOG_NUMBERING.md`) — `SESSION_CHANGELOG_YYYYMMDD_<TOPIC>.md`
   is canonical; **no new S-numbers**; no retro-numbering; one reconciliation ruling at
   SER8-backlog import.

## 4. Two specific currency corrections to make

- **`gate12-passed-attempt9` = `e9ca800`** is the certified pre-change reference for all
  window-anchor work. The catalog predates the Gate-12 pass entirely; anywhere it describes
  Gate 12 as pending or Phase 7 as blocked is now wrong. **Say what changed, per REV2 §2's
  rule about dated documents whose subject has moved.**
- **`G-PROD-SHAPE`** (`tests/gate_s172_prod_shape.py`) is Phase 7's certifying verifier —
  built, proven red against the failed 2026-08-04 soak log (9 pass / 5 fail), **NOT RUN**,
  Michael-initiated only, requires a live 25-daemon fleet. It belongs in §1.4 (code
  inventory) and should be reachable from whatever §1.6/§1.7 says about Phase 7.

## 5. Scope — REV2 §3, restated because it is the easiest rule to drift from

**Do not fix anything. Do not update any chapter. Do not resolve any gap you find.** Findings
go in §1.6 or into `BACKLOG.md`'s register as a NOTE for Alpha — **note them, do not action
them.** No commits, no pushes, no WATCHER, no pipeline.

If you find a document that contradicts the current state, that is a §1.6 gap entry with the
evidence, not an edit.

## 6. Known untracked paths — index NOTHING from them

Present in the tree and **out of scope**: `PIECE_MATCHER_*.md` and `piece_matcher/` (a
separate project), and `docs/SESSION_CHANGELOG_20260819_S1.md` (unattributed, violates the
naming ruling, pending Michael's disposition — **mention it in §1.6 as an unresolved
tree-state item, do not index it as governance**).

## 7. Operational constraint — REV2 §4

Your output file is untracked and dirties the tree. The Phase-7 soak's clean-tree preflight
rejects a dirty tree at finalization, and the soak is currently UNBLOCKED and may launch.
**End your report by telling Michael the catalog must be committed before the soak launches.**

## 8. Report

Per REV2 §5: in your final message, not a second file — indexed-vs-total count, themes used,
anything unclassifiable, §1.6 gaps with the searches that found them. Then STOP.
