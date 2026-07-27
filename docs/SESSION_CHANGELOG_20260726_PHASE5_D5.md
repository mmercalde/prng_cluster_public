# SESSION CHANGELOG — 2026-07-26 — S172 Phase 5 D5 (process_sharded backend)

**Status:** COMMITTED, not yet pushed at time of writing. Team Beta approved D5
REV3 for commit; `serial_reference` remains production default, `process_sharded`
is committed but unpromoted pending Phase 6's performance/≤50%-RAM decision.

**Commits (VM 101, user michael):**
- **C1 `e69d667`** — S172 Phase5 D5 C1: canonical extraction + serial preservation
  + lossless dual-encoding projection codec (TB REV3 approved). One file,
  `miner/range_miner_npz_writer.py`, +604/−77.
- **C2 `bdd36b4`** — S172 Phase5 D5 C2: process_sharded backend + D5 gates 24/24
  (18 mutants), unpromoted pending Phase 6 (TB REV3 approved). Five files,
  +4702/−8: `miner/assembly_shard_worker.py` (new),
  `miner/assembly_backends.py`, `tests/test_s172_phase5_d5_process_sharded.py`
  (new), `tests/fixtures/pre_d5_range_miner_npz_writer.py.frozen` (new),
  `tests/test_s172_phase4_coordinator.py` (gate-22 whitelist, +27/−0).

Base before commit: `3e8580a`. HEAD after: `bdd36b4`. (C2 was `--amend`ed once,
pre-push, to apply Beta's final terminology correction "minimal-signed-length" →
"deterministic signed-byte" in the D5 gate label + M17 description — display text
only, no code path, formula, or assertion changed.)

---

## What D5 delivers

`process_sharded` parallelizes **only** spool-local validation. D1.1 remains the
sole authority for validation semantics and global assembly semantics; workers
produce ordered, lossless validated-spool projections, and the parent alone
performs deterministic global merge, duplicate attribution, intersection,
enrichment, and final assembly. Field-for-field equivalent to `serial_reference`.

**Two-commit split (deliberate):** C1 is the semantics-preserving extraction
(`prepare_trial_assembly` / `read_and_validate_spool` / `merge_validated_spools`,
with `assemble_trial` a thin serial wrapper) — **one file, independently
checkoutable**, no process machinery. C2 is the backend + worker + gates + the
pre-D5 oracle fixture + the gate-22 whitelist. The `.py.frozen` oracle fixture
lives in C2 because the only consumer is the C2 acceptance harness.

---

## The three-revision arc

- **REV1 → Option B rework:** the initial read-all-then-merge structure changed
  observable exception precedence (earlier-dup + later-malformed raised
  `SpoolIdentityError` instead of `DirectionalDuplicateError`). Team Beta ruled
  **Option B**: preserve deterministic read/merge precedence — workers return
  canonical read errors as typed data, parent replays in manifest order.
- **REV2:** Option B implemented — lazy interleaved serial, `CapturedSpoolReadError`
  descriptors (never pickled exceptions), `as_completed`-fills / replay-in-order
  split, backend failures distinct from producer defects, cleanup preserving the
  primary canonical exception. Approved.
- **REV3 → int64 seed blocker:** Beta held D5 because the `int64` seed projection
  raised `OverflowError` for seeds ≥ 2⁶³, a valid-input divergence from the
  pre-D5 arbitrary-precision domain. Resolved with a lossless **dual-encoding
  projection**: fast `int64` for the signed-64 domain, `signed_bytes` fallback
  (concatenated uint8 runs + uint64 offsets, `int.from_bytes(..., signed=True)`)
  the moment any seed leaves that range. `allow_pickle=False`, no object arrays.
  A single decoder returns Python ints on both paths, so merge keys match pre-D5
  exactly. **Approved for commit.**

Two blockers were caught here (precedence, then seed-overflow), both genuine
accepted-input divergences that D1.1's test domain did not exercise — the same
evidentiary blind spot each time. Both are now gated against a frozen pre-D5
oracle.

---

## Gate + non-regression evidence (committed tree `7137c95`, venv `~/venvs/torch`)

- **D5: 24/24 green**, 18 mutants all RED with four-part attribution
  (applies-once / mutated-path / detector-clean / injected-defect). Includes
  M8 (RUSAGE_CHILDREN vs sampled concurrent tree-sum: 133 MiB single-child max
  < 173 MiB needed for two concurrent 96 MiB children) and the REV3 seed mutants
  M15/M16/M17.
- **Internal non-regression (same run):** D1.1 18/18, D2 7/7, D3 10/10, D3.0
  10/10, D3.25 13/13, D3.5 60/60, D4 8/8 (D1.1 nests Phase 4 63/63, Phase 3
  17/17, D0, D1.0).
- **D4 8/8** with all nine original mutants intact (verified live at `3e8580a`
  and on the committed tree). **gate-22:** `1 file changed, 27 insertions(+)`,
  zero deletions, no logic change; changed-`.py` set is exactly the five
  whitelisted paths.

**Commit-1 no-op proof:** D1.1 18/18 with zero test edits, in the working tree
and over a pristine `3e8580a` archive with only the writer overlaid (no worker,
no D5 harness, no fixture). The wider downstream set ran in the same construction
at writer digest `440080f4…`, AST-identical to the submitted bytes.

**Artifact digests (final submitted state):**
- `miner/range_miner_npz_writer.py` (C1 re-freeze): `2d829938a9df3901542471a6fb78fecaece629fd120a2cc5f60651de28eefebe`
- `miner/assembly_shard_worker.py`: `577a71f0d38ab569713c2eebeb67ce708a3998142033440f33be9235bd471979`
- `tests/test_s172_phase5_d5_process_sharded.py`: `d3cb74d669e1465ce826bf32284ccd75bc67aed17abb55be2aa6ab26b94b251d` (post-rename; was `10875d2a…` before Beta's terminology correction)

The writer moved from `440080f4…` to `2d829938…` for the REV3 doc-wording
correction only; proven docs-only by an `ast.dump`-with-docstrings-stripped
identity check.

---

## Benchmark (§4.5 sweep; measurement only, §17 promotion is Phase 6's)

peak_rss = `sampled_sum_of_parent_and_recursive_children_rss`, 25 ms interval
(conservative — RSS-sum double-counts shared pages).

High-survivor (32 manifests, 6.5 MiB): serial 3.209 s; process pool 1→8 =
3.151 → 1.910 s (peaks ~1.63–1.65× at pool 6–8), peak RSS 290 → 799 MiB.
Low-survivor (16 manifests): serial 0.002 s; process ~0.33–0.41 s (pool startup
dominates). serial_reference stays default; process buys ~1.6× on high-survivor
at ~2–3× RAM — the ≤50% RAM clause reads as the binding §17 constraint.

---

## Notes for Team Beta

1. **psutil is a hard import** in the process backend (the concurrent-tree RSS
   sampler). On a shell without the venv active (no psutil), the D5 suite fails
   to construct the backend and the mutation harness correctly refuses to green
   (positive control reds). Not a D5 blocker — serial_reference is the default
   and needs no psutil — but whether psutil should be a **lazy/optional import**
   (loaded only when the sampler runs) rather than a hard module-level import is
   an open question worth a ruling, since only the sampler consumes it and the
   process backend is unimportable without it.
2. **Pre-existing scope note (unchanged):** D3.5's columnizer bounds `seeds` to
   uint32 (`utils/canonical_arrays.py:204`). That wall predates D5, applies
   identically to both backends and the pre-D5 engine; a seed that clears
   assembly then fails D3.5's uint32 contract fails the same way it always did.
   REV3's obligation was the engine's accepted domain, discharged.
3. **Judgment calls on the record:** `.py.frozen` oracle fixture (avoids a second
   importable engine copy), `__post_init__` shape validator (beyond REV3's
   literal dataclass), `-128` (and analogous negative signed-width boundaries)
   encode non-minimally but losslessly under the deterministic signed-byte length
   formula, and the corrected mutant count of **18** (the inherited REV2 "16" was
   itself one over the 15 that actually ran; REV3 adds three).

---

## Post-commit checklist

- [x] C1 `e69d667`, C2 `bdd36b4` committed on VM 101
- [x] D5 24/24 + full internal non-regression green on committed tree (venv)
- [x] Beta terminology correction ("minimal-signed-length" → "deterministic
      signed-byte") applied to the committed D5 gate label + M17 description
- [ ] this changelog + D5 session docs committed to `docs/`
- [ ] dual-push: `git push origin main && git push public main`
