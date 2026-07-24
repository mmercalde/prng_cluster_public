# TEAM_ALPHA_REVIEW_S172_PHASE5_D1_1.md

**Subject:** Team Alpha code-level review of the D1.1 implementation
(assembly engine + `AssemblingPhase5Sink`)
**Spec:** `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D1.md` REV5 §4-§9
**Base:** HEAD `b7d2d09`
**Artifacts:** `miner/range_miner_npz_writer.py` (769 lines),
`tests/test_s172_phase5_d1_engine.py` (1285 lines), diff (whitelist, 6 lines),
status.
**Verdict: APPROVED — production module unchanged and correct; the single
harness blocker (G9 circularity) was corrected in a Claude Code correction
round and re-verified by Team Alpha (§4). One scope item and one design note
flagged (§5). Recommend Team Beta review for commit authorization.**

## 1. Scope

Diff touches only gate-22's whitelist, registering the two new D1.1 paths.
Note for Beta: the standing rule pre-authorizes "the deliverable's exact new
**harness** path"; this registration also includes the production module
`miner/range_miner_npz_writer.py`, which is outside the rule's literal
wording though mechanically forced identically (gate 22 checks all changed
`.py`). Requested: extend the standing rule to "the deliverable's new file
paths" under the same four conditions. No coordinator code changed. Status
clean (briefs/tmp pre-existing, excluded).

## 2. Production module — verified faithful to REV5

- **§6 record:** the frozen 24-field constant matches the REV5 sequence
  exactly; records rebuilt explicitly against it (not insertion-order trust);
  ascending seed; `threshold_used` absent; `sessions` normalized `None → []`.
- **§5.5 formulas:** field-by-field identical to the live blocks (:652-694,
  :756-796): `len()` counts, the deliberate `bidirectional_count` /
  `intersection_count` twin, all `max(..., 1)` denominators,
  `skip_range = skip_max - skip_min`, `score = (f+r)/2`.
- **§5.1-§5.4:** full identity matrix incl. run_id + lifted provenance;
  family authority via imported `workflow_stages_for(base, True)` inverted
  (deliberate superset — phase-set completeness stays §5.2's job, correctly);
  11-field consistency through the coordinator's own canonicalizer;
  phase-set `{1,2}`/`{1,2,3,4}`; the complete [TB-D1-PV] container/semantic
  battery with bool-excluded-before-equality; `DirectionalDuplicateError`
  with all 13 structured attributes as real attributes, deterministic
  first-insertion ordering.
- **§4 sink:** §4.0 wording verbatim; one RLock over all four methods;
  deep-copy-as-canonical-form; no spool I/O at publish (single `_read_spool_bytes`
  seam — measured, not assumed, and a D5 swap point); replay/slot matrix
  exact incl. identical-bytes slot conflict; atomic commit with
  marker-after-install and manifests retained on failure; tombstoned abort;
  frozen `get_assembly`. Two subtleties deliberately audited: the commit
  check ordering makes it impossible for the failure path to clobber a
  previously stored result, and post-commit new-slot publication correctly
  raises `AssemblyStateError` (state precedence over replay conflict).
- The un-wrapped `utils/prng_encoding.ValueError` (flagged by Claude Code):
  Team Alpha endorses — §5.4 designates the canonical module's own hard-fail;
  inventing a D1 wrapper would shadow the single source of truth. The engine
  docstring documents the propagation. Beta to bless.

## 3. Mechanical verification (Team Alpha sandbox, pristine `b7d2d09`)

- Full harness: **18/18 green** (G1-G16 + NR: Phase 4 63/63, Phase 3 17/17,
  D0 12/12, D1.0 W1-W6 8/8) — independent reproduction of the reported
  result. Claude Code additionally captured the NR baseline green at
  `b7d2d09` BEFORE any edit and reported 9/10 injected mutants killed by the
  intended gates, with the 10th shown equivalent and the genuinely dangerous
  variant killed by G5.
- Team Alpha injected THREE independent mutants:
  - **M1** formula drift (`survivor_overlap_ratio` denominator |F|→|R|):
    **killed** by G1 with the exact field and both values (the asymmetric
    |F|=4/|R|=3 fixture working as designed).
  - **M3** tombstone removed from `abort_trial`: **killed** by G8 + G16b.
  - **M2** frozen-constant reorder ("score"/"window_size" swapped in
    `CANONICAL_RECORD_FIELDS`): **SURVIVED — 18/18 green.** See §4.

## 4. BLOCKER (found, corrected, re-verified) — G9 was circular for the frozen constant

**Finding:** the original G9 imported `CANONICAL_RECORD_FIELDS` from the
writer and asserted every record's key tuple against that import. This proved
records conform to the module's constant — length 24, no duplicates,
ascending seeds all real — but NOT that the constant matches the REV5-frozen
sequence: reordering the constant reordered the records identically and the
WHOLE harness stayed 18/18 green (demonstrated empirically by Team Alpha's
M2). Rule 2 was violated for exactly the drift class the gate names. Notably,
none of Claude Code's ten behavior-targeted mutants could expose this — the
blind spot circularity creates is precisely that mutating the imported oracle
is invisible.

**Correction (harness-only, one gate):** `_G9_RECORD_FIELDS_ORACLE` — an
independent 24-tuple transcribed BY HAND from REV5 §6 at test-module level,
with a covenant comment (never derive from / sort against / regenerate out of
the module under test; disagreements resolve against REV5 §6, never by
editing the oracle). G9 now asserts three separable things: production
constant == oracle (membership AND order); every record's key tuple == oracle
DIRECTLY (holds even if production bypassed its own constant); length-24 /
no-`threshold_used` / ascending seeds retained.

**Claude Code's own non-circularity evidence:** four production-constant
mutations run against the full gate set — two pure reorders (old G9: passed;
new G9: RED, G9 only — the load-bearing cases, invisible to every other gate
since record VALUES stay correct), a field rename and a 25th-field addition
(RED on G9 + 12 others). It additionally audited the rest of the harness for
the same circularity shape (other gates assert against hand-computed
literals, fixture-derived values, or independently transcribed failure
fragments) and confirmed G9 was the only instance, and verified the writer
byte-identical to the reviewed module against a pre-correction copy.

**Team Alpha re-verification (pristine `b7d2d09` sandbox):** writer confirmed
byte-identical to the reviewed copy; M2 (score/window_size swap — a THIRD
reorder pair, independent of Claude Code's two) re-injected → **killed**, G9
red with the exact "drifted from REV5 §6" production-vs-oracle diff, 17/18;
writer restored → **18/18 green** including full NR. Closed.

## 5. Notes (non-blocking)

- `sessions` is stored by reference into every record's `metadata_base`
  (one shared list per mode). Identical to legacy behavior (legacy reuses
  `config.sessions`); consumers are read-only. Recorded for D3/D6 awareness.
- Claude Code's report and gate design (per-case failure-reason assertions;
  asymmetric opposite-direction populations so a mode/direction cross-wire
  cannot pass) are of notably high quality; the G9 miss is the single gap.

## 6. Recommendation

Submit to Team Beta for commit authorization: production module approved
unchanged; harness approved after the G9 correction (re-verified, §4). Two
rulings requested of Beta: (a) the whitelist standing-rule extension to "the
deliverable's new file paths" (§1); (b) blessing of the un-wrapped
`utils/prng_encoding.ValueError` propagation (§2). Claude Code's self-directed
circularity audit of the D1.0 and D0 harnesses (queued by Michael) is a
separate follow-up; any finding there is not a D1.1 blocker.

— Team Alpha (Claude), 2026-07-23
