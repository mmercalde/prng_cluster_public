# CLAUDE_CODE_INSTRUCTIONS_BOUNDED_PHASE_6.md — REV1

**Bounded Phase 6: certify RANGE-MINER.**

This is the certification the whole PWC → RANGE-MINER pivot was built toward. Everything since
D0 has been making it possible.

**Base:** current `main` on VM 101. Claude Code as `michael`, venv `~/venvs/torch`. Implement
and iterate; you do **NOT** commit, push, or run WATCHER. STOP at the gate.

**The rigs are up and provisioned** — all three CT100s hold the frozen dataset, verified on
target. Multi-rig work is in scope.

---

## 0. What is being certified, and what is already done

Team Beta scoped Phase 6 as three walls. **Wall C's broad 44-PRNG requirement was struck**
(Beta, Wall C ruling): known-answer validation was established practice before the repository
existed, the method is valid, the references are genuine, and **Michael's account is accepted as
the historical project record.** Nothing is being repeated.

What replaces it is **one bounded transfer gate on four `java_lcg` variants** — §3. Its subject
is **RANGE-MINER**, which did not exist when the original validation was done. The question is
whether that result *transfers* to the new engine, not whether the original work happened.

**Already established, cite rather than re-derive:**
- **CUDA/ROCm byte-identity** — Phase 6.0 (`23fa413`): identical `artifact_sha256` across the D6
  release-grade generation and both 6.0 runs, 22/22 arrays equal, no GPU reset, no
  `GCVM_L2_PROTECTION_FAULT`.
- **Dataset authority** — P0.5 (`d4ff1e4`) + Q2 closure (`8600e75`).
- **Admission liveness** — (`ee0db06`). A worker loss now reaches the failure matrix.
- **Currency** — `miner/range_miner_worker.py:837` imports `sieve_gpu_worker._get_kernel`, which
  compiles the registry's own `kernel_source`. **The miner runs the same kernel source the
  original campaign exercised.**

## 1. Wall A — interface and consumer

**"Steps 2 onward cannot tell which engine produced the data"** is the pivot's founding rule. It
is broader than opening the NPZ.

Prove, on a miner-produced certified generation:
- frozen **22-array** contract — names, order, shapes, dtypes;
- `validate_array_bundle()` passes;
- **Step-2 loader succeeds with `fallback_used=False`**;
- NPZ → dict conversion preserves every field;
- **Step-3 chunk generation** preserves the contract;
- `survivors_with_scores` smoke completes **without metadata loss**.

Beta was explicit that the Step-3 leg stays: *"'Steps 2 onward cannot tell' is broader than
merely opening the NPZ in Step 2."*

## 2. Wall B — determinism and platform

With **identical frozen inputs and configuration**:
- **CUDA vs ROCm** — all 22 arrays field-for-field equal *(Phase 6.0 evidence may be cited; state
  what is cited vs re-run)*;
- **repeated run vs repeated run** — identical semantic artifact;
- **`serial_reference` vs `process_sharded`** — identical semantic artifact;
- **multi-rig** — identical results **independent of node assignment**. This is the one with no
  prior evidence: Phase 6.0 was single-rig.

Provenance must bind: repository commit and tree state · dataset lineage, size and SHA-256 ·
canonical run-input-manifest digest · seed-domain contract and exact range · PRNG family and
variant · window/session selection · **requested / payload / effective thresholds** · skip mode
and effective skip semantics · assembly backend.

## 3. The Miner Known-Answer Transfer Gate (replaces Wall C)

**Four variants only:** `java_lcg` · `java_lcg_reverse` · `java_lcg_hybrid` ·
`java_lcg_hybrid_reverse`.

Beta's six requirements, verbatim in substance:

1. **Expectations generated independently** of the registry, miner, coordinator, backend and
   finalizer.
2. **Production semantics** — raw seed state, `>>16`, **inter-draw** skips, and **forward
   iteration over reversed residues** for reverse mode.
3. **Exercise the actual miner worker path** — residue resolution, payload interpretation,
   argument builder, kernel launch, result extraction. **Not merely a direct `RawKernel` call.**
4. **Compare the complete bounded survivor set** per population against the independent result.
   **Planted-seed recovery alone is insufficient — it would not detect extra false survivors.**
5. **Clean control and fault-injection control** that the comparator demonstrably rejects.
6. **Terminate with an unambiguous nonzero failure status.**

**Starting material:** `pa_sieve_validation_harness.py` (352 lines, stdlib only, its own CPU
brute-force bidirectional sieve). It needs alignment to production — it uses the Java
seed-scramble and `>>17` where production uses raw seed and `>>16`, and its reverse uses a
modular inverse where production reverses residues.

**Beta did not ratify Alpha's "~20 lines, an afternoon" estimate**, and was right not to:
aligning the reference addresses the reference only, not the four miner ABI paths, the exact-set
comparison, or the VIR controls. **Scope it honestly and report the real shape.**

## 4. RandomSampler control arm

Beta approved this as an **operator-selected** TPE control. **Autonomous `search_strategy`
selection is NOT approved** — sampler choice is reserved authority.

**Implementation direction (Beta):** do **not** route all samplers through a permanently named
`run_bayesian_optimization()`. Extract a neutral `run_optimization(..., sampler,
sampler_metadata)` with thin TPE and random entrypoints. This closes the signature mismatch and
the vacuous-body problem **without making a RandomSampler run semantically "Bayesian."**

**The comparison must bind:** sampler class and Optuna version · sampler seed · trial budget ·
warm-start mode · objective definition · effective search-space digest · **effective skip
semantics** · repository commit and tree state · dataset/input-manifest digest · study identity.

**A single TPE run versus a single random run is not sufficient.** Matched budgets across
**multiple deterministic sampler seeds**, reporting **distributions, not only the best trial**.

**Sequencing caution:** `skip_min`/`skip_max` remain dead on the hybrid path (skill §2.7 #4). A
sampler comparison over a **falsely seven-dimensional space** measures noise in two of its
dimensions. **State this in the report as a known limitation of any comparison run now**, and
flag whether Beta should sequence the comparison after the skip work.

## 5. Out of scope

- **The skip-output work** (§0.4 of the skill) — Beta sequenced it after bounded Phase 6.
- **The Resolved Execution Set** and Beta's Q1 local-run refinement — after Phase 6.
- **Autonomous sampler selection** — not approved.
- **`grid` / `evolutionary` samplers** — `GridSampler` is unconstructible here
  (7.649 × 10¹⁰ points ≈ 7.2 TiB); CMA-ES deferred.
- **The `java_lcg_cpu` non-zero-skip mismatch** in `survivor_scorer.py:124` /
  `full_scoring_worker.py:305` — Beta ruled this a **separate bounded reachability/consequence
  audit before Phase 7**, and **no fix is authorized**. Do not touch it.
- D6.2 · D6.3 · the scraper · session-split dataset authority.

## 6. Two corrections Beta ordered, small and in scope

- **`reverse_kernel_test_results.txt`** — mark **superseded in place**, preserving the original
  rows. Header must state: `SUPERSEDED — NOT VALIDATION EVIDENCE` · all results were `BOTH
  ZERO` · no positive control established detector sensitivity · **prohibited from citation
  under VIR-2** · and identify the replacement gate once §3 exists. Correct any document citing
  it as correctness evidence.
- **`test_ALL_46_prngs_10M.sh`** — the filename and two "12" comments are stale; **its array
  contains 44, eleven in each of four categories**. Correct the header and category counts. **Do
  not claim it contains two invalid registry names** — Beta checked, it does not. Do not cite it
  as a known-answer test.

## 7. Verification-integrity controls (VIR-1…6)

- **execution proof** — every certification claim carries an artifact digest or a `file:line`
  read this session. Effective values read **off the executor**, not recomputed from config.
- **clean control** — a healthy run passes each wall.
- **fault-injection control** — §3.5 is mandatory. For Walls A and B, state what would make each
  fail and show it does.
- **completion sentinel** — explicit `PASS | FAIL | UNAVAILABLE | INCOMPLETE` per wall.
- **unavailable-observer** — anything not exercised is `UNAVAILABLE`, never assumed. **Do not
  cite Phase 6.0 evidence as though re-run** — say which is cited and which is fresh.
- **audit claim scope** — declare searched and unavailable surfaces; state which claims are
  VM-101-only and which were observed on a rig.

## 8. Non-regression

D1.1 · D4 · D5 · D6 3.A · **D6-threshold 17/17** · D6.1 · **threshold-propagation 5/5** ·
Chapter1-P0 12/12 · **P0.5 dataset authority 38/38 with `--fleet`** · **admission liveness
16/16** · Phase 3 · **Phase 4 63/63**.

Gate 22 and `G-MINER-UNCHANGED` will see changed files — register with rationale, **append
rather than rewrite**, and **keep P0.5's strengthening of `G-MINER-UNCHANGED` intact** (it greps
registered diffs for threshold tokens).

## 9. Report

Per wall: what was proven, with digests and anchors; what was **cited** from prior evidence
versus **re-run**; the completion sentinel. For §3: the independent reference's provenance, the
four ABI paths exercised, the exact-set comparison result, and both controls. For §4: the
neutral entrypoint, the bound comparison metadata, the seed-matched distributions, and the
dead-dimension caveat. Then STOP. **Do not commit.**
