# D6 FOLLOW-UP — both-modes (variable / hybrid) single-GPU smoke

**Purpose:** D6's 3.B smoke proved the constant-skip column (workflow phases 1+2,
forward/reverse constant) end-to-end on real silicon. This follow-up proves the
**variable-skip / hybrid column** (phases 3+4) — a *different kernel* and
*different seed caps* — and the full four-phase both-modes assembly, on the same
3080 Ti. It is scoped exactly like 3.B: a real-hardware smoke, not a throughput
benchmark, no s/s target.

**Frozen against the same base as D6** (HEAD `a22216c`, code `2a6e0f8`). Claude
Code on VM 101 as `michael`, venv `~/venvs/torch`,
`--optimizer-python ~/distributed_prng_analysis/python3_with_venv.sh`. Implement
and iterate; do NOT commit/push/run WATCHER. STOP at the gate.

---

## Why this is not just "run 3.B again with a flag"

Verified at source (`range_miner_coordinator.py:1482–1523`): the §6.8 workflow
table is `1→forward/constant, 2→reverse/constant, 3→forward/variable,
4→reverse/variable`, and **variable skip IS the hybrid family** — `prng_type` is
`java_lcg` for constant but `java_lcg_hybrid` for variable, which runs the hybrid
kernel under the tighter `*_hybrid` seed caps. So phases 3/4 exercise a different
ABI and different caps than 3.B's phases 1/2. Both-modes also produces a SECOND
population pair (`bidirectional_variable` alongside `bidirectional_constant`) that
3.B never populated. This is genuinely new coverage on real hardware.

## The two smokes to run

### T1 — variable/hybrid-only single-GPU smoke (phases 3+4)
Same shape as D6 3.B but constant OFF, variable ON: real `range_miner_worker.py`,
cupy on the 3080 Ti, `java_lcg` base resolving to the **`java_lcg_hybrid`**
kernel, `serial_reference` backend, a small-but-real seed window.

**Acceptance:**
- the run drives phases **3 and 4** (assert the published manifests carry
  `skip_mode="variable"` / `prng_type="java_lcg_hybrid"`, not constant);
- a certified generation is produced whose 22-array bundle passes
  `validate_array_bundle()`;
- the **Step-2 loader reads it back** (`fallback_used=False`);
- capture `nvidia-smi` during the run, the generation dir + sidecar, the
  hybrid `prng_type` on the manifests, the variable-population counts, and the
  Step-2 load-back.

This is the first certified generation from the **hybrid kernel** on real
silicon — the variable-skip counterpart to D6's constant-skip milestone.

### T2 — both-modes single-GPU smoke (phases 1+2+3+4 in one trial)
`test_both_modes = true`: one trial that runs all four phases and assembles BOTH
populations.

**Acceptance:**
- all four workflow phases execute (manifests for constant AND variable present);
- the assembly carries non-empty `bidirectional_constant` **and**
  `bidirectional_variable`, with `canonical_records_constant` and
  `canonical_records_variable` both populated (ascending seed);
- one certified generation for the run (per-run finalize, matching D6's
  run-level provenance — NOT one generation per phase);
- 22-array bundle validates; Step-2 loader reads it back `fallback_used=False`;
- the D6 adapter appends candidates from **both** record lists into the
  accumulator with real counts (the both-modes analogue of D6 G-INGRESS);
- capture the same evidence set as T1, plus the two separate population counts.

## Guardrails (unchanged from D6)

- `serial_reference` default; `process_sharded` not exercised here.
- Fail closed on any absent publication result — never a fabricated zero.
- Do NOT touch the PWC/ZMQ ingress, the D3.25 four-map contract, or `TestResult`.
- `MinerTrialAssembly.binary_npz_path` / `all_npz_path` stay `None`; certified
  paths come from `RunArtifactResult`.
- These are smokes — no s/s target, no promotion decision (Phase 6 owns those).
- Reuse the D6 adapter and the existing coordinator; do not reimplement assembly,
  finalization, or ingress. If a both-modes path needs a config field that 3.B
  didn't set (e.g. the hybrid caps, `test_both_modes`), wire it through the same
  `getattr`-on-coordinator pattern the D6 `_use_miner` gate already uses — do not
  hardcode.

## Watch for (real-silicon-only surprises, like 3.B found the flush bug)

- **Hybrid seed caps actually apply.** Phases 3/4 must use `seed_cap_*_hybrid`,
  not the constant caps — confirm the tighter cap is in force, and that a
  cap/family mismatch still hard-aborts with `MinerIngressError` (the fail-closed
  path 3.B validated).
- **Residue derivation asymmetry** (D6 finding #3): coordinator computes residues
  without a session filter, worker re-derives with one — a no-op on
  `daily3.json` (both sessions) but a `ResidueVerificationError` on a
  single-session config. Run T1/T2 on the both-sessions dataset so this doesn't
  confound the smoke; note it if it surfaces.
- **Directional thresholds** (D6 finding #4): `build_stripe_assign_payload`
  carries no `min_match_threshold`, so the kernel uses its 0.25 default. Same
  behavior as 3.B; just record what threshold the variable run actually used so
  the counts are interpretable.

## Non-regression

No production-code change is expected — these are new smoke gates, not a rework.
If any wiring change is needed to enable both-modes, re-run the full suite
(D1.1 18/18 … D6 9/9) to confirm nothing shifted, and confirm PWC/ZMQ byte-
identical to `2a6e0f8`. If no production code changes (config/flags only), state
that explicitly.

## Report

For T1 and T2 separately: the manifest `skip_mode`/`prng_type` proof, the
population counts, the certified generation dir + sidecar, the 22-array
validation, and the Step-2 load-back. Confirm which caps were in force and which
threshold the kernel used. Then STOP for Team Alpha review.
