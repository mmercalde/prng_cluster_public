# SESSION CHANGELOG — 2026-08-20 — SKILL v27

**Session type:** documentation / skill revision (no code changes)
**Naming per:** `docs/TB_RULING_CHANGELOG_NUMBERING.md` — SR-2, date + topic, no S-number.
**Repo at open:** `17a51e5` · **working tree at close:** `17a51e5` + this session's uncommitted
`docs/` changes (Claude does not commit — CLAUDE.md rule 1).
**Certified baseline throughout:** `gate12-passed-attempt9` = `e9ca800`.

---

## 1. What this session was

The v27 skill pass that `SESSION_CHANGELOG_20260820_FIELD6_AND_WINDOW_ANCHOR_DESIGN.md` §5 and
`PROJECT_FILE_CATALOG.md` §7 gap 10 both record as owed. v26 (2026-08-17) predates three days of
rulings, the field-6 repair, both standing rules, the window-anchor design, and the regenerated
catalog.

## 2. ⚠ Drift measured at session open — the three-copy hazard, again

| copy | state found |
|---|---|
| `docs/TFM_PROJECT_FACTS_SKILL.md` (tracked) | **v26**, 2,810 lines |
| `~/.claude/skills/tfm-project-facts/SKILL.md` (what Claude Code loads) | **v23, 2026-08-10**, 2,124 lines |

**Three revisions and ten days of Gate-12 history — attempts 3 through 9, the pass itself, the
clean-tree admission repair, the stale-rig discovery — had never reached the runtime copy.** A
session invoking the skill that morning read a world in which Gate 12 had failed twice and Phase 7
was held. Recorded in §7 of the skill as a second dated instance of the 2026-08-03 failure.

## 3. Sources read (all at HEAD `17a51e5`, verified against a fresh public clone)

`PROJECT_FILE_CATALOG.md` §1.0/§1.1/§1.3/§4.6/§5.2/§5.3/§6.5/§7/§8 · `TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md` ·
`TB_RULING_WINDOW_ANCHOR_SEQUENCING.md` · `TB_RULING_WINDOW_ANCHOR_PROPOSAL_V1_0.md` ·
`TB_RULING_CHANGELOG_NUMBERING.md` · `TB_RULING_FIELD6_IMPLEMENTATION.md` ·
`PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md` ·
`SESSION_CHANGELOG_20260817_GOVERNANCE_RULINGS.md` ·
`SESSION_CHANGELOG_20260820_FIELD6_AND_WINDOW_ANCHOR_DESIGN.md`.

**Live-source verification (not carried from documents):** `git show --stat d8b21e3` (6 files,
967 insertions) · `miner/range_miner_coordinator.py:3731-3732` (`_bp` `None` seeds), `:7315-7343`
(emitter + `UNOBSERVED` render + the grep-collision comment), `:8034-8040` (None-aware update) ·
`tests/gate_s172_prod_shape.py` present, 14,524 bytes, mtime 2026-08-04 · fresh clone of
`mmercalde/prng_cluster_public` at `17a51e5`, tracked skill byte-identical to the working tree.

## 4. What changed in the skill (v26 → v27)

**New sections**
- **§1.3 ⚑ STANDING RULES** — SR-1 (`DECLARED_CHANGED` maintenance, four binding constraints,
  reverse protection, the recorded-but-unauthorised future housekeeping) and SR-2 (changelog
  naming, and why guessing `S186` converts unknown history into asserted governance state).
- **§2.52 FIELD-6 — CLOSED at `d8b21e3`** — the narrower diagnosis (the metrics dict was never the
  defect; the emitter was), the three changed definitions with live anchors, the `TypeError`-into-
  blanket-`except` trap, `G-FIELD6`/`G-MUT-FIELD6`, the three forced certified-suite edits, the
  **FAIR-4/0 ratification and why an exact tally was never an authorization proof**, Gate 22 as
  transitional, and the claim boundary (observable ≠ observed).
- **§2.53 WINDOW-ANCHOR / GENERATOR-PHASE** — the absent-design finding and Beta's correction of
  its own sequencing wording; the binding semantic contract; **the terminology law: 100 = anchor
  ceiling, 149 = record-envelope ceiling**, with the v1.0 `[0,149]` rejection recorded as the
  category error the design exists to eliminate; capability matrix; frozen ABI / DEP-ABI-V2;
  `generator_phase = 0`; derived anchor domain and its machine representation; `offset` key removed
  outright; 22-array wall closed; `anchor_era` provenance-not-authority; the `continuation_phase`
  firewall; executable legacy-engine closure; sequential Brief I → Brief II.
- **§2.54 PHASE 7 / `G-PROD-SHAPE`** — Phase 7 unblocked, Gate 12 ≠ the soak, the verifier's three
  anti-fabrication checks and why every previously-certified miner run failed them, status **built
  / proven red / NOT RUN / Michael-initiated**, what the soak delivers, and its **non-certifying**
  classification for anchor semantics.

**Revised**
- Currency header rewritten for the post-acceptance state.
- **§2.49** — Beta's acceptance ruling appended: dispositions table, Fields 1 and 2 **MISSED AS
  WRITTEN** with the reasoning for refusing to renormalise them, and the **mandated phrasing** for
  the R-3 complexity result, to be used verbatim.
- **§2.51** — items 2 and 3 marked **DISCHARGED / CLOSED at `d8b21e3`** (observation still owed);
  two items added (accumulator backup policy; WATCHER failure-authority); **the catalog's thirteen
  open gaps added as a pointer table**.
- **§3** — pointer to catalog **§6.5** (statements true on 2026-08-03, false now).
- **§8** — Gate-12 / field-6 / design-gate lines added; the stale *"Beta HOLDS gate 12 and the
  soak"* wording replaced with the two open parallel tracks; the *"soak is HELD"* paragraph
  corrected, flagged as a live example of §1.2; the unaudited-chapter count corrected from five to
  **eleven** per catalog §7 gap 6.
- **§7** — SR-2 folded into the changelog bullet, SR-1 added as an operational pre-commit step, and
  the measured 2026-08-20 drift recorded in the three-copy block.
- **§9** — five new self-check items: SR-1, SR-2, anchor-vs-envelope, soak non-certification, and
  the mandated R-3 phrasing.
- Catalog anchors updated `1fc05bb`/803L/562 files → **`0a4cef1`/1,085L/699 files** in all three
  places that carried them.

Result: **2,810 → 3,213 lines.**

## 5. Three-copy state at close

| copy | state |
|---|---|
| `docs/TFM_PROJECT_FACTS_SKILL.md` | **v27 written, UNCOMMITTED** — Michael commits and dual-pushes |
| `~/.claude/skills/tfm-project-facts/SKILL.md` | **v27 installed**; prior copy backed up as `SKILL.md.bak-v23` |
| Settings upload | **STILL v26 or older — owed. Manual re-upload required** |

All copies verified `sha256[:12] = 27a35a6b5133`. **The revision is not done until the Settings
upload and a fresh-session currency check are also done** (skill §7).

## 6. Files changed

| file | change |
|---|---|
| `docs/TFM_PROJECT_FACTS_SKILL.md` | v26 → v27 |
| `docs/SESSION_CHANGELOG_20260820_SKILL_V27.md` | this file (new) |

No code, config, manifest, gate or test was touched. Nothing was committed or pushed. WATCHER and
the pipeline were not run. `tests/gate_s172_prod_shape.py` was not invoked.
