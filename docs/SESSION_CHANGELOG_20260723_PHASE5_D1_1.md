# SESSION_CHANGELOG_20260723_PHASE5_D1_1.md

**Session scope:** S172 Phase 5 — D1.1 implementation (assembly engine +
`AssemblingPhase5Sink`), Team Alpha adversarial review, G9 correction round,
Team Beta approval.
**Base:** HEAD `b7d2d09` (D1.0). Same-day continuation of
`SESSION_CHANGELOG_20260723_PHASE5_D1.md`.

## Outcome

**D1.1 APPROVED FOR COMMIT by Team Beta.** After this commit is dual-pushed,
**D1 is complete** and D2 begins against the new HEAD.

## Delivered

- `miner/range_miner_npz_writer.py` (769 lines): frozen 24-field
  `CANONICAL_RECORD_FIELDS`; backend-independent `assemble_trial` (§5.1-§5.5
  validation: identity matrix incl. run_id + lifted provenance, imported
  `workflow_stages_for` family authority, 11-field consistency via the
  coordinator's canonicalizer, `{1,2}`/`{1,2,3,4}` phase-set, full
  [TB-D1-PV] container/semantic battery, `DirectionalDuplicateError` with 13
  structured attributes, live-frozen enrichment formulas);
  `AssemblingPhase5Sink` (RLock, canonical-deep-copy ownership, no publish-time
  spool I/O via the single `_read_spool_bytes` seam, exact replay/slot rules,
  atomic commit with marker-after-install + retained manifests on failure,
  tombstoned abort, frozen `get_assembly`); six exception types.
- `tests/test_s172_phase5_d1_engine.py` (1285+ lines): G1-G16 + blocking NR
  over the real coordinator/ledger/publish lifecycle; asymmetric
  opposite-direction populations per mode so direction/mode cross-wires
  cannot pass; five labeled direct-sink invariant-break probes; per-case
  failure-reason assertions.
- Gate-22 whitelist: both new paths registered (6 lines).

## Review arc — the G9 finding

Claude Code's self-testing: NR baseline captured green at `b7d2d09` BEFORE
any edit; 10 injected mutants, 9 killed by their intended gates, the 10th
shown equivalent and the genuinely dangerous variant killed by G5.

**Team Alpha's independent mutation pass found the one structural gap:** G9
imported `CANONICAL_RECORD_FIELDS` from the module under test — circular for
the constant itself. A reordered frozen constant left the ENTIRE harness
18/18 green (demonstrated empirically). None of the ten behavior-targeted
mutants could see it: mutating the imported oracle is exactly the blind spot
circularity creates.

**Correction round (harness-only):** `_G9_RECORD_FIELDS_ORACLE`, hand-
transcribed from REV5 §6 with a never-derive-from-production covenant; G9
asserts production constant == oracle AND every record == oracle directly.
Claude Code's non-circularity evidence: four constant mutations (two pure
reorders — old G9 passed, new G9 reds, G9-only, the load-bearing cases;
rename + 25th field red on G9+12); harness-wide circularity audit confirmed
G9 was the only instance; writer verified byte-identical to the reviewed
module. Team Alpha re-verified with a third independent reorder pair
(killed, exact drift diff) and a clean 18/18.

## Team Beta rulings (binding)

- **Whitelist standing rule EXTENDED:** a deliverable's exact new Python
  file paths (not just harness) may be registered in an established
  coexistence whitelist that checks all changed `.py`, under the same
  conditions (new files of the approved deliverable; registration-only; no
  gate-logic change; local format; explicitly reported).
- **`utils/prng_encoding.ValueError` propagates un-wrapped** — the canonical
  module's hard-fail is the validation decision; wrapping would obscure the
  single source of truth. It must never be converted to a warning, default,
  skipped record, or partial result.
- **`sessions` shared reference accepted** (matches legacy); D3/D6 must
  treat `MinerTrialAssembly` and its records as immutable inputs — never
  mutate a record's `sessions` in place.
- **Commit condition:** the committed files must be exactly the
  Alpha-re-verified versions; any post-verification change to the writer or
  G9 requires a full harness + NR rerun.

## Verification record

D1.1 18/18 · Phase 4 63/63 · Phase 3 17/17 · D0 12/12 · D1.0 W1-W6 8/8 —
independently reproduced in the Team Alpha sandbox on pristine `b7d2d09`.
Mutants: formula-denominator → G1 kill; tombstone removal → G8+G16b kill;
schema reorder → survived old G9, killed by corrected G9.

## Committed in this change

`miner/range_miner_npz_writer.py`, `tests/test_s172_phase5_d1_engine.py`,
`tests/test_s172_phase4_coordinator.py` (whitelist only),
`docs/TEAM_ALPHA_REVIEW_S172_PHASE5_D1_1.md`, this changelog.
Excluded: pre-existing untracked briefs and `tmp/`.

## Next

**D2** — the adversarial producer-level `DirectionalDuplicateError` fixture,
per REV5 §5.4/G14 forward-reference, against the post-D1.1 HEAD. Follow-up
(non-blocking): Claude Code's queued circularity audit of the D0/D1.0
harnesses; any finding routes through the normal review path. D6 reminder
stands: seam-level adversarial treatment of `serve_trial`'s return path.
