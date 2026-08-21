# SESSION CHANGELOG — 2026-08-20 — FIELD-6 REPAIR AND WINDOW-ANCHOR DESIGN

**Session type:** implementation (field-6) + design (window-anchor) + governance
**Naming per:** `docs/TB_RULING_CHANGELOG_NUMBERING.md` — date + topic canonical until
SER8-backlog reconciliation; no S-numbers assigned.
**Repo at open:** `24ed568` · **at close:** `d8b21e3` + governance commit
**Certified baseline throughout:** `gate12-passed-attempt9` = `e9ca800`

---

## 1. Field-6 observability repair — IMPLEMENTED, ACCEPTED, COMMITTED

TB sequencing item 3 of the Gate-12 acceptance ruling. Claude Code executed the committed
brief on VM101 (59m31s); Alpha reviewed against source; Beta ruled; committed `d8b21e3`,
dual-pushed.

**What shipped** (`miner/range_miner_coordinator.py`, 3 definitions changed, **zero added** —
AST proof 289/289):

- `_bp` seeds for both R-1 falsifier fields changed `0` → **`None`** — the UNOBSERVED
  sentinel. `0` conflated "never measured" with "measured a maximum of zero".
- `_pump_deferred` update made **None-aware** (`obs if prev is None else max(int(prev), obs)`),
  inline, under `_bp_lock`, still wrapped. This closes the trap the brief predicted: the old
  `int()` cast over a `None` seed raises `TypeError` into the blanket `except`, which would
  have left both fields `None` forever while looking like a working feature.
- `log_staging_backpressure_summary` — **the actual defect** — appends both keys to the END
  of the grep-stable `[S172-BP] summary` line; integer when observed, literal `UNOBSERVED`
  when `None` (`staging_jobs_per_sec=n/a` precedent). Dict keeps `None` (JSON-safe).
  Verified non-colliding: `deferred_high_water=` does not match
  `deferred_distinct_attempts_high_water=`.
- **Rider (pre-authorized debt):** `_pump_deferred` docstring corrected to the R-1..R-4
  report's recommended wording. Rode this commit per the debt rule.

**Diagnosis correction found at source:** `staging_backpressure_metrics()` already exported
both fields via `dict(self._bp)` since `e9ca800`. The metrics dict was never the defect; the
emitter was. Gate-12 attempt 9 ran the instrument and threw the reading away.

**Brief correction from the implementer:** Scope-B's edge case "a pump pass over an empty
`_deferred` legitimately records 0" is **wrong** — `_pump_deferred` early-returns under
`_admission_lock` before the instrument, so a recorded `0` is unreachable from that call
site. Verified, not reasoned. The None-aware form still does not special-case zero away.

**New gates** (`tests/test_s172_staging_backpressure.py`, suite 50 → **52**):

- `G-FIELD6` — three arms, all values parsed off the **emitted line**, never the dict.
  Arm 1: two runs at K=3 and K=6 distinct attempts × 4 frames, asserting **exactly**
  `distinct == K` and `probes == frames + 2·(K−1)` — a relation **derived from
  `_pump_deferred` as read** (R-2 discard-at-grant + R-3 end-of-pass sweep), then
  corroborated against `gate_g8e`'s existing measurements. Emitted `3/8` and `6/14`.
  Non-vacuity asserted (frames really took the defer branch); `distinct != probes` asserted
  so M2's detectability is a fixture fact. Arm 2: no-pump run emits literal `UNOBSERVED` on
  both, dict `None`, adjacency at line end asserted. Arm 3: dict↔line coherence.
- `G-MUT-FIELD6` — M1 hardcoded zeros, M2 transposed arguments, M3 restored `int()` cast.
  All APPLIED (anchor-moved assertions), EXECUTED (M3 carries an execution proof independent
  of the gate's verdict), DETECTED. Mutants rebound against **production module globals**
  (the A8-B2 escape closed).
- Completeness list extended with both keys — key presence explicitly insufficient as
  evidence; `G-FIELD6` is the gate that proves the values are measurements.

**Three certified suites edited, outside the brief's enumerated scope** — all forced by the
mandated emitter change, all ratified by Beta:

| suite | change | result |
|---|---|---|
| `test_s172_r1_drain_remedy.py` | `DECLARED_CHANGED` += `log_staging_backpressure_summary` (provenance: FIELD-6, not R-1); M9 fixture made None-aware so the mutant stops dying on `TypeError` before exercising its mutation | 42/44 → **44/44** |
| `test_s172_mp1_drain_attribution.py` | same declaration addition, same provenance comment | 37/38 → **38/38** |
| `test_s172_attempt6_remediation.py` | FAIR-4/0's transcribed `50/50` pin replaced with parse + `passed == total` + floor `>= 50` + the suite's pass-only COMPLETION SENTINEL | 77/78 → **78/78** |

**Clean-tree battery after commit: 52/52 green, `COMPLETION SENTINEL: PASS`.** Transient
Gate-22 red self-cleared exactly as Beta predicted; allowlist unchanged.

## 2. TB ruling — field-6 implementation review

`docs/TB_RULING_FIELD6_IMPLEMENTATION.md`. Accepted. Two rulings of lasting weight:

- **FAIR-4/0 replacement RATIFIED, do not revert.** Beta's reasoning goes further than
  Alpha's: an exact tally was never a sound authorization proof — an unauthorized gate swap
  still reads 50/50, two legitimate additions falsely red at 52/52. It conflated suite health
  with suite governance. FAIR-4/0's ruled job is completion + all-passed + no gross deletion
  below the floor; it is **not** the authority on whether new checks were authorized.
- **⚑ STANDING RULE ADOPTED:** any authorized commit changing a coordinator definition
  covered by a historical exact live-vs-anchor scope gate MUST update every affected
  `DECLARED_CHANGED` set before acceptance — without moving the anchor, without relaxing
  `changed == DECLARED_CHANGED`, adding only definitions actually changed, each with
  provenance. Reverse protection (declared-but-unchanged on revert) retained deliberately.
  **No per-commit ruling needed henceforth.** Beta's preferred eventual housekeeping —
  immutable original owned set vs. provenance-tagged post-anchor delta set, exact over the
  union — is recorded but explicitly NOT to be built yet.

## 3. Window-anchor / generator-phase separation — DESIGN GATE CLOSED IN TWO ROUNDS

- **v1.0** (`docs/PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_0.md`) drafted from
  source reads: per-variant capability matrix verified at the worker builders, fused-key hard
  reject, silent-clamp removal, derived anchor domain, phase pinned 0, ABI-v2 dependency
  recorded, consumer enumeration, five scoped questions.
- **Beta ruling** (`docs/TB_RULING_WINDOW_ANCHOR_PROPOSAL_V1_0.md`): **architecture
  ACCEPTED**; all five questions ruled; one genuine semantic error caught —
  **`[0,149]` as an anchor range REJECTED. 100 is the historical ANCHOR ceiling; 149 is the
  historical RECORD-ENVELOPE ceiling** (anchor 149 + window 50 reaches record 198, outside
  history). Alpha committed the exact anchor/extent category error this design exists to
  eliminate. Corrected control domain: `[0, min(100, N_filtered − window_size)]`.
- **v1.1** (`docs/PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md`) — bounded
  corrections only, no new semantics: Q4 fix everywhere + a terminology law in §2 (anchor =
  start index, envelope = reachable records, never interchanged) + the error encoded as a
  **permanent regression test** (AC3 asserts anchor 149 with window 50 is NOT in
  `control_era`); `search_bounds.offset` key REMOVED outright (no tombstone — JSON has no
  comments); exact machine representation for the derived max
  (`{min, max_cap}` with `effective_max = min(max_cap, N_filtered − window_size)` — widening
  impossible by construction); `anchor_era` demoted to **provenance, not authority**; new
  §4.8 legacy-engine closure (route determination, hard-disable or align, fail-loud entry
  guard, call-graph evidence, tested by AC5); AC1 strengthened to prove independence with a
  **synthetic nonzero phase on a supported ABI** via arg-capture while the public schema
  stays fail-closed; sequential Brief-I → Brief-II lineage.
- Per Beta's disposition, if v1.1 is bounded-corrections-only the **design gate is closed**
  and implementation proceeds directly to **Brief I**.

## 4. Governance housekeeping

- **Changelog naming ruled** (`docs/TB_RULING_CHANGELOG_NUMBERING.md`): date + topic
  canonical until SER8-backlog reconciliation; no new S-numbers; no retro-numbering of the
  three existing topic-named sessions; one deliberate Beta ruling at import time to restore
  or formally retire the S-sequence. The trigger was that `S185` is only the highest
  *visible* number while ~20 SER8-only changelogs await backfill — guessing `S186` would
  convert unknown history into asserted governance state.
- The owed 2026-08-17 governance changelog was written and committed under that convention.

## 5. State at close

| track | state |
|---|---|
| Field-6 repair | **CLOSED** — `d8b21e3`, accepted, 52/52 on the clean tree |
| Phase-7 soak | **UNBLOCKED.** Provides the first production observation of both falsifiers. `G-PROD-SHAPE` (`tests/gate_s172_prod_shape.py`) is its certifying verifier — built, proven red against the failed 2026-08-04 soak log, **NOT RUN**, Michael-initiated only |
| Window-anchor | design gate closing at v1.1; **Implementation Brief I** is the next artifact |
| WATCHER failure-authority defect | still open, untouched |
| Skill currency | **v26 (2026-08-17) is now stale** — three days of rulings, the field-6 repair, two standing rules, and the window-anchor design are unrecorded; `G-PROD-SHAPE` was never in it. v27 pass is owed |
| Untracked, unattributed | `docs/SESSION_CHANGELOG_20260819_S1.md` — carries an S-number the naming ruling forbids; owner unidentified; excluded from all commits pending disposition. `PIECE_MATCHER_*` / `piece_matcher/` are a separate project, out of scope here |

## 6. Commits this session

| commit | content |
|---|---|
| `d8b21e3` | Field-6 observability repair — coordinator + 4 suites + implementer report |
| (governance) | Beta's field-6 ruling, proposals v1.0 + v1.1, v1.0 review ruling, this changelog |

All dual-pushed; `d8b21e3` verified on the public remote from an independent clone.
