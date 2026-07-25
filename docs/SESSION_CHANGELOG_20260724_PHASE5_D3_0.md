# SESSION_CHANGELOG_20260724_PHASE5_D3_0.md

**Session scope:** S172 Phase 5 — D3 architecture rulings (D3.0 / D3 / D3.25 /
D3.5 split) and the D3.0 legacy seam correction: brief, implementation, review,
correction round, approval.
**Base:** HEAD `2d37b77` (D2 complete).

## Outcome

**D3.0 APPROVED FOR COMMIT** — all Team Beta §9 conditions satisfied. Phase 5's
remaining sequence was substantially restructured by this session's rulings
(§Architecture below).

## Why D3.0 existed

Phase 0 replaced inline PRNG encoding tables with a registry-derived canonical
module that hard-fails on unknown identities. **That fix never reached either
NPZ writer.** `utils/prng_encoding.py` was imported by exactly two files
(`tests/test_prng_encoding.py`, `miner/range_miner_npz_writer.py`); both writers
carried a local 12-entry table with a silent `.get(..., 0)` fallback —
`convert_survivors_to_binary.py:30-38` and
`window_optimizer_integration_final.py:1715` (the **live** Step-1 producer).
Measured: 7 shared keys disagreed on value, `java_lcg_hybrid` was absent so
every hybrid survivor was relabelled `java_lcg`, and two ids
(`randu`/`randu_reverse`) were not in KERNEL_REGISTRY at all. Second defect: the
two writers disagreed on the empty case — one array (`seeds=[]`) vs 22
rectangular.

## Delivered

- `convert_survivors_to_binary.py` — local tables deleted for
  `utils.prng_encoding` (canonical names re-exported to preserve the module's
  public surface); resolution chain `prng_type → prng_base → 'java_lcg'` kept
  verbatim, only the encode step changed; empty path now writes all 22
  zero-length arrays with frozen dtypes in the frozen order.
- `window_optimizer_integration_final.py` — `_PRNG_ENC`/`_SKIP_ENC` deleted,
  two seam-local canonical imports, two encode call sites updated. **The
  accumulator (merge, supersede, backfill, sort, dual write) is outside the
  diff entirely** — verified structurally, the second hunk terminates
  immediately after the encode lines.
- `tests/test_s172_phase5_d3_0_encoding_contract.py` — E1-E10, independent
  hand-transcribed oracle; the non-importable inline closure is extracted from
  live source by AST line-range and `exec`'d, so editing the seam changes what
  the gate runs.
- Gate-22 whitelist registration (production-file change explicitly authorized
  by Beta for this deliverable).

## Review arc

Team Alpha found **one harness gap**: M4 (swap two adjacent keys in the
empty-path order) left the full gate **10/10 green**, while the harness header
claimed its oracle covered *"names / order / dtypes."* Same class as D1.1's
circular G9 — stated scope exceeding actual assertion. Beta made it a blocking
correction. Fixed test-only: `E8_EXPECTED_KEY_ORDER` as a literal tuple
(verified not derived from `_EMPTY_NPZ_DTYPES`, `NPZ_CONTRACT`, `NPZ_KEYS`, or
any production constant) asserted against `tuple(z.files)`. Re-verified with an
**independent** key pair → killed; production byte-identical; 10/10 clean.

Three other mutants killed on the first pass (restored `.get(...,0)` → E4/E6;
dropped array → E8; wrong dtype → E8). M1's profile confirmed gate precision:
restoring the *fallback* while keeping the *canonical table* correctly leaves E2
green.

**Team Alpha process note:** the first re-verification reported E8 red against a
clean source file — cause was a stale `__pycache__` in the Alpha sandbox holding
bytecode from the earlier mutant. Recorded because the discipline applies to the
reviewer's own tooling: purge bytecode on restore, and trace a surprising red
before reporting it.

## Findings escalated (Team Beta ruled)

1. **The write/read asymmetry was already live** — `utils/survivor_loader.py`
   re-exports the canonical `PRNG_TYPE_DECODING`, so Phase 0 reached the reader
   but not the writers. **9 of 12 legacy ids misread**; only 0, 7, 8 round-trip.
2. **RESOLVED — no migration needed for TFM's data.** Inspection of the live
   accumulators: `bidirectional_survivors_all.npz` and `_binary.npz`, 20,949
   rows each, `prng_type={0}`, `skip_mode={0}` — all java_lcg **constant**, zero
   hybrid records, so the collapse never bit this data and id 0 round-trips.
   Beta accepted a fourth disposition, *"verified canonical-compatible; no data
   migration required,"* conditional on a one-time provenance snapshot (paths,
   SHA-256, row/schema/value sets, **source identity from an authoritative run
   record**, timestamp, commit) — because `prng_type=0` alone is ambiguous
   between `java_lcg` and an unknown identity collapsed by the old fallback.
   No operational hold on post-D3.0 runs of these files.
3. **Metadata sidecar becomes the prospective provenance mechanism** for
   post-D3.0 artifacts, with Beta's expanded required field list; it does not
   retroactively certify older NPZs.
4. **D3.0-B authorized (non-blocking)** — residual copied tables:
   `window_optimizer_bayesian.py:231-235` (6-entry, writes both canonical
   filenames, dormant but a regression vector), `apply_s145r1_npz_accumulator.py`
   and `apply_s149_npz_checkpoint.py` (bake the legacy table into patch text),
   stale `docs/` copy, `harness_npz` 5-entry oracle. Must complete before Phase
   6 certification.
5. **Residual `'java_lcg'` default** accepted as D3.0 compatibility only —
   D3/D3.25 must reject a new record carrying neither identity field.
6. **No historical hybrid baseline exists** (`skip_mode={0}` throughout), so
   Phase 6's both-mode surface is a first establishment, not a regression.
   Requires synthetic adversarial both-mode fixtures first.

## Architecture rulings this session (Phase 5 sequence restructured)

Team Alpha objected to two items in the original D3 plan; **both sustained**:
L2/L3 are run-level services shared by every backend (the miner rejoins at
`_build_test_result_from_pw:426` exactly as PWC does at `:472`), so a miner-only
accumulator would put different finalizers between compared sieve engines and
confound Phase 6; and `binary_npz_path`/`all_npz_path` are run-level artifact
claims wrongly attached to the per-trial `MinerTrialAssembly`. Beta's resolution:
one **shared** finalizer for all backends, Phase 6 compares at **two**
boundaries (normalized per-trial candidates AND final 22-array artifacts), and
canonical paths move to a new run-level `RunArtifactResult` while the per-trial
fields stay deprecated and permanently `None`.

Beta then found a further defect: `_build_test_result_from_pw`'s
`for seed in bidi_constant | bidi_variable` **union** collapses a cross-mode
seed into one variable-labelled record, destroying the constant candidate before
L2. Team Alpha traced it deeper — the adapter has no per-mode maps at all, its
aggregates are hardcoded to constant-mode, and `skip_range`/`sessions` diverge
in format from the canonical contract. Both producers already compute all four
maps and discard two on return (PWC `:1690`/`:1717`/`:1720`, return `:1747`; ZMQ
`:1136`/`:1151`/`:1155`, return `:1186`), and their hybrid *record lists* are
not a safe substitute (different provenance per backend). Hence **D3.25**.

Revised sequence: **D3.0** (this commit) → **D3** (shared 24→22 columnizer, no
wiring) → **D3.25** (v2 four-map producer contract + shared canonical
normalizer + adapter ingress) → **D3.5** (shared run finalizer,
`RunArtifactResult`) → **D4/D5** → **D6**. `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3_25.md`
is written (REV2) and awaits rebase onto this HEAD.

## Verification record

D3.0 gate **10/10** · D2 7/7 · D1.1 18/18 · D1.0 8/8 · D0 12/12 · Phase 4
63/63 · Phase 3 17/17 · Phase 0 8/8 · Phase 1/2 6/6 6/6. Baseline captured green
at `2d37b77` before any edit; pre-fix reds captured (E2/E8/E10 plus
E3/E4/E5/E6, 3/10); independently reproduced in the Team Alpha sandbox.

## Committed in this change

`convert_survivors_to_binary.py`, `window_optimizer_integration_final.py`,
`tests/test_s172_phase5_d3_0_encoding_contract.py`,
`tests/test_s172_phase4_coordinator.py` (whitelist only),
`docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3_0.md`,
`docs/TEAM_ALPHA_REVIEW_S172_PHASE5_D3_0.md`, this changelog.
Excluded: `docs/PHASE6_PREREQS.md` (still awaiting its own Beta review), the
D3.25 brief (pending rebase), pre-existing untracked briefs and `tmp/`.

## Next

**D3** — the shared backend-neutral 24→22 columnizer plus its independent
contract validator, no production wiring, per Beta's revised scope. In parallel
and non-blocking: the Ruling-F provenance snapshot (needs source identity from
Step-1 config / study DB / run log), D3.0-B hygiene, and
`docs/PHASE6_PREREQS.md` review.
