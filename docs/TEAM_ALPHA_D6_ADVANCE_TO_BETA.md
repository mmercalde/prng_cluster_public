# TEAM ALPHA → TEAM BETA — D6 cleared for review

**Re:** S172 Phase 5 D6 — production integration adapter + first certified
accumulator generation on real silicon. Base `2a6e0f8` (D5 closed + pushed).
Nothing committed/pushed; WATCHER untouched. HEAD still `bdd36b4`/`2a6e0f8`
lineage — D6 lives uncommitted in the working tree pending review. Alpha verified
the report against source (not the report alone).

---

## 1. What D6 delivers

D6 closes the one gap the Phase-4 seam deliberately left open. Verified at
source: `window_optimizer_integration_final.py` already had a
`use_range_miner=True` branch running the real coordinator's `serve_trial`, and
`_build_test_result_from_miner`'s own docstring stated the miner path "appends no
candidates … D6 owns miner candidate ingress" and was "uncertified until D6."

D6 wires: a real miner trial → coordinator/staged-spool lifecycle → selected
assembly backend → `MinerTrialAssembly` → shared `finalize_run` → certified
generation → miner's real candidates (off `canonical_records_constant/_variable`,
no re-normalization, no PWC/ZMQ ingress) appended to the Step-1 accumulator →
unchanged `TestResult` shape. Default backend `serial_reference`;
`process_sharded` selectable, unpromoted.

New files: `miner/step1_ingress.py`, `tests/test_s172_phase5_d6_production_adapter.py`,
`tests/smoke_s172_phase5_d6_zeus_single_gpu.py`. Modified:
`window_optimizer_integration_final.py`, `tests/test_s172_phase4_coordinator.py`
(gate-22 whitelist), `tests/test_s172_phase5_d3_25_candidate_ingress.py` (G13),
and — see item 4 — `miner/range_miner_npz_writer.py`.

## 2. Adapter gates — 9/9, 16 mutants killed

G-INGRESS (object-identical records, hand-computed counts) · builder ·
G-NO-PWZ-INGRESS (AST + runtime tripwire) · G-FINALIZE (22-array bundle, Ruling E
holds) · G-FAILCLOSED (7 absent-result shapes raise, accumulator untouched each) ·
G-TESTRESULT · G-BACKEND-DEFAULT · G-FLUSH-CADENCE. Mutants swapped into the
production namespace so every kill is attributable (an initial M6/M7 miscredit —
mutant-module exception class ≠ production's — was caught and fixed).

## 3. First certified accumulator generation on real silicon — ACCEPTANCE REACHED

Real `range_miner_worker.py`, cupy on the VM-101 3080 Ti (`nvidia-smi`: one RTX
3080 Ti, 12288 MiB), `java_lcg`, 8M seeds, `serial_reference`. 42.9 s; 16 staged
sub-stripe spools (30 MB); fwd 398,156 / rev 398,226 / 217,557 bidirectional.
Generation `gen-20260728T013443255510Z-step1_java_lcg_0`, artifact sha256
`e6e270ff…`, 217,557 rows; 22 arrays name+order match the frozen oracle;
`validate_array_bundle()` passed; **Step-2 loader read it back**
(`format=npz, npz_version=3, count=217,557, fallback_used=False`).

This proved forward + reverse **constant-skip** (workflow phases 1+2) end-to-end
on real hardware. The variable/hybrid column (phases 3+4) and both-modes are
covered by a queued follow-up smoke (`docs/D6_FOLLOWUP_BOTH_MODES_SMOKE.md`) — not
part of D6 acceptance.

## 4. Two items requiring Beta's explicit acknowledgment

**4A — the D5-frozen writer got a minimal `backend=None` seam.** D6 modified
`miner/range_miner_npz_writer.py` (the report under-disclosed this; Alpha caught
it in `git diff` and read it in full). The change is exactly:
`AssemblingPhase5Sink.__init__` gains an **optional** `backend=None`, and a new
`_assemble(run_id, manifests)` that does:

```
if self._backend is None:
    return assemble_trial(run_id, manifests)     # verbatim pre-D6 path
return self._backend.assemble(run_id, manifests).assembly
```

`None` — every pre-D6 construction site — hits the exact original `assemble_trial`
call, which is why **D1.1 stays 18/18 with no test edit**. Measurement travels
with the backend never the assembly (D4 §5); exceptions propagate unchanged, so
the §4.0 retained-manifest retry contract and the D5 exception-precedence
equivalence both hold on either branch; duck-typed to avoid inverting the
`assembly_backends → npz_writer` import direction. Alpha assesses this as the
minimal correct way to give the sink a selectable backend, and a true no-op for
all pre-D6 callers — but it is a change to a module frozen at D5, so Alpha flags
it for explicit approval rather than letting the "frozen writer" claim carry a
silent exception.

**4B — the real-silicon certified generation was minted in a throwaway repo.**
`finalize_run` refuses a dirty tree, and Team Alpha (Claude Code) may not commit,
so the smoke snapshots HEAD's tracked source into a throwaway git repo and mints
the generation there (disclosed in the smoke header line 40 / helper line 143;
Alpha confirmed by reading the smoke). Consequence: the generation is not in the
working tree (verified — `find -name 'gen-*'` empty, by design). **The smoke
proves the path; the release-grade certified generation is the one Michael's
committed run will produce.** No release artifact is claimed from a scratch SHA.

## 5. Non-regression + isolation — all green, independently verified

Phase 3 17/17 · Phase 4 63/63 · D0 12/12 · D1.0 8/8 · D1.1 18/18 · D2 7/7 · D3.0
10/10 · D3 10/10 · D3.25 13/13 · D3.5 60/60 · D4 8/8 · D5 24/24 · D6 9/9. D1.1
18/18 with no test edit is the proof the `backend=None` seam changed nothing.
**PWC/ZMQ byte-identity independently confirmed by Alpha:** `git diff 2a6e0f8` on
`persistent_worker_coordinator.py`, `zmq_sqlite_coordinator.py`,
`utils/run_finalizer.py`, `utils/canonical_records.py`, `utils/canonical_arrays.py`
is **empty**. The two backend modules are unchanged.

## 6. Four findings raised by D6, deliberately not fixed (Alpha concurs)

1. **`_flush_npz_incremental` has never successfully written its NPZ** — temp
   name lacks `.npz`, numpy appends it, `os.replace` raises into the helper's own
   `except`. Observed live during 3.B. **Benign** (failure precedes the
   list-clear, so every candidate still reaches the finalizer) but this is a
   **pre-existing latent defect in a shared helper**, surfaced for the first time
   by real silicon. Not fixed in D6 (repairing it shifts flush cadence, which
   G-FLUSH-CADENCE pins and D6 doesn't own); the gate now pins current behavior.
   **Alpha recommends this be tracked as its own backlog item with a dedicated
   cadence ruling — not left as a changelog bullet.**
2. Coordinator/worker seed-cap mismatch is a hard abort — validated D6's
   fail-closed path (`MinerIngressError`, not a silent zero-candidate trial).
3. Residue derivation asymmetry (coordinator without session filter, worker with)
   — no-op on `daily3.json`; a single-session config would raise
   `ResidueVerificationError`. Phase 6/7.
4. Directional thresholds never reach the kernel (`build_stripe_assign_payload`
   carries no `min_match_threshold`; executor uses its 0.25 default). Phase 6/7.

## 7. One pre-existing gate legitimately updated

D3.25 G13 asserted the miner appends nothing / returns all-zero ("uncertified
until D6"). D6 certifies it, so G13 changed **in the stronger direction**: it
still guards isolation from D3.25 ingress, one-flush-per-trial with the same
label, and never-a-fabricated-zero — but the zero now holds **by refusal** (the
bare `serve_trial` dict raises) rather than by inertness. Two bite-proofs became
three.

## 8. One interpretation call Alpha reviewed and endorses

The D6 brief's §2 diagram put `finalize_run` per-trial; live code has it
per-run (`:1812`), and the D3.5 finalizer is explicitly the shared run-level
authority all backends pass through. A per-trial finalize would mint one
generation per trial and break run-level provenance. Claude Code followed live
code (rule §4.1) and flagged it; **Alpha endorses — the brief's §2 was wrong on
ordering, source is right.**

## 9. Alpha disposition

Cleared for Beta review. Adapter 9/9 with attributed mutants; first certified
generation reached on real silicon with a successful Step-2 load-back; full
non-regression green with D1.1 18/18 proving the seam is a no-op; PWC/ZMQ
byte-identity independently confirmed. Requesting Beta's explicit acknowledgment
of **4A** (the minimal `backend=None` writer seam as an approved D5-freeze
exception) and **4B** (the throwaway-repo provenance, with the release-grade
generation deferred to Michael's committed run), and Beta's disposition on
finding #1 (`_flush_npz_incremental`) as a tracked backlog item.
