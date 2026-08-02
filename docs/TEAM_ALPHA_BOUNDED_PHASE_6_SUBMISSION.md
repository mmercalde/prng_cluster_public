# TEAM ALPHA — BOUNDED PHASE 6 SUBMISSION

**Authority:** `docs/CLAUDE_CODE_INSTRUCTIONS_BOUNDED_PHASE_6.md` (REV1)
**Base:** VM 101 (`zeus-ubuntu`, 192.168.3.177), `main` at **`76e8eaf`**, venv `~/venvs/torch`, run as `michael`.
**Session:** S184, 2026-08-01/02.
**Nothing committed, nothing pushed, WATCHER not run, the pipeline not launched.** STOP at the gate.

---

## 0. Sentinels

| Item | Sentinel | Evidence |
|---|---|---|
| **Wall A** — interface and consumer (§1) | **PASS** | `docs/phase6_evidence/wall_ab.json` · `tests/phase6/wall_ab_gate.py` |
| **Wall B** — determinism and platform (§2) | **PASS** | same, four fresh comparisons + one cited leg |
| **§3** — Miner Known-Answer Transfer Gate | **PASS** | `docs/phase6_evidence/known_answer_gate.json` · `tests/phase6/known_answer_gate.py` |
| **§4** — RandomSampler control arm | **PASS (NON-CERTIFYING)** | `docs/phase6_evidence/sampler_control_arm.json` — see §4 and the dead-dimension caveat |
| **§6** — the two ordered corrections | **DONE** | in place, in the repository |
| **§8** — non-regression | **PASS — 22/22 suites exit 0** | every pinned tally at its figure: Phase 4 **63/63**, Phase 3 **17/17**, D5 **24/24**, D6 3.A **9/9**, D6-threshold **17/17**, threshold-propagation **5/5**, Chapter1-P0 **12/12**, P0.5 `--fleet` **38/38**, admission liveness **16/16** |

Two items are flagged for Beta's decision rather than claimed as complete: the §4
**sequencing** question, and one **operational cleanup on the rigs** that Alpha created and
could not remove (§10).

---

## 1. WALL A — interface and consumer · **PASS**

**Subject:** a miner-produced **certified generation**, built this session on the RTX 3080 Ti
through the production coordinator, the Phase-5 assembly and `utils.run_finalizer.finalize_run`.

```
generation_id    gen-20260802T011719898562Z-step1_java_lcg_0
artifact_sha256  0e0092feeb02e22d28557ddf4d8e421941d6117bcc0448d7f7323ec402c1c4b0
rows             319        forward 398,156   reverse 383   bidirectional 319
dataset          daily3-20260801T145551443433Z-513648160d35.json
                 sha256 513648160d356617c22a1e543ae1c9c65f4921ec21718989308b1f70c00768f6
                 18,068 records, 1,380,711 bytes, resolved through the P0.5 pointer manifest
seed domain      contiguous uint64, half-open [0, 8,000,000)
window           window_size=3, offset=0, sessions=[midday, evening]
thresholds       requested forward 0.31 / reverse 0.47 — requested == payload == effective,
                 read off the executor, per direction
skip             constant, test_both_modes=False; kernel skip range [0,16] from the payload
```

### Legs

| Leg | Result | What would make it fail |
|---|---|---|
| **A1** frozen 22-array contract | **PASS** — 22 arrays, order identical to the frozen oracle, all shaped `(319,)` | an array added, removed, renamed, reordered, or not per-row |
| **A2** `validate_array_bundle()` | **PASS** | dtype / length / NaN / contract violation in any of the 22 |
| **A3** Step-2 loader | **PASS** — `format=npz`, `npz_version=3`, `count=319`, **`fallback_used=False`** | the loader cannot read the NPZ and silently falls back to JSON |
| **A4** NPZ → dict preserves every field | **PASS** — 319 records × **22 fields**, missing `[]`, extra `[]` | `extract_survivors_full` drops a field — the exact regression that once left 14/47 ML features at zero |
| **A4b** NPZ → dict value round-trip | **PASS** — **7,018 values** compared, 0 mismatched | a value coerced, truncated or reordered in the conversion |
| **A5** Step-3 chunk generation | **PASS** — 3 chunks, 319 records read back, field set preserved in every chunk, byte round-trip equal | chunking drops records, drops fields, or reorders them |
| **A6** `survivors_with_scores` smoke | **PASS** — 107 scored, **91 features/record**, sieve metadata present **with the miner's values** (0 value mismatches) | the sieve metadata does not survive the chunk → scorer boundary, or arrives with scorer defaults |

A6 runs the real `full_scoring_worker.py` on a real chunk on the real GPU. It is a *metadata-loss*
test, not a key-presence test: `forward_count`, `reverse_count`, `bidirectional_count`,
`skip_min`, `skip_max`, `skip_range` are compared **value-by-value** against the miner's own
NPZ rows. The 91-feature count matches the settled contract (91 extracted / 89 trained).

### Fault-injection control (VIR-2) — 5/5 rejected

| Injection | Rejected by |
|---|---|
| FA1 remove `reverse_matches` from the bundle | `ArrayBundleError` |
| FA2 widen `score` float32 → float64 | `ArrayBundleError` |
| FA3 truncate `seeds` by one row | `ArrayBundleError` |
| FA4 hand `extract_survivors_full` a seeds-only bundle (the 14/47 shape) | `ValueError` |
| FA5 point the Step-2 loader at a missing NPZ | `FileNotFoundError` — a silent `fallback_used=False` would have been undetectable |

**Beta's Step-3 leg is discharged.** "Steps 2 onward cannot tell which engine produced the data"
is exercised as a chain — NPZ → loader → dict → chunk files → scorer — not merely as an NPZ open.

---

## 2. WALL B — determinism and platform · **PASS**

**Five arms, identical frozen inputs and configuration; one variable changed per comparison.**
Every arm advertises all four seed caps equal, so `select_seed_cap` resolves identically on CUDA
and ROCm and the sub-stripe boundaries — hence the canonical record order — match by construction.

| Arm | Workers | Backend | artifact_sha256 | rows | f / r / b |
|---|---|---|---|---|---|
| `cuda_run1` | zeus-ubuntu-vm (CUDA) | serial_reference | `0e0092fe…c4b0` | 319 | 398,156 / 383 / 319 |
| `cuda_run2` | zeus-ubuntu-vm (CUDA) | serial_reference | `0e0092fe…c4b0` | 319 | 398,156 / 383 / 319 |
| `cuda_sharded` | zeus-ubuntu-vm (CUDA) | **process_sharded**, pool_size 4 | `0e0092fe…c4b0` | 319 | 398,156 / 383 / 319 |
| `rigs_ab` | **rrig6600 + rrig6600b** (ROCm) | serial_reference | `0e0092fe…c4b0` | 319 | 398,156 / 383 / 319 |
| `rigs_bc` | **rrig6600b + rrig6600c** (ROCm) | serial_reference | `0e0092fe…c4b0` | 319 | 398,156 / 383 / 319 |

### What is CITED and what is FRESH

| Leg | Status |
|---|---|
| **B1 CUDA vs ROCm** | **CITED — NOT RE-RUN.** Source: `docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md`, commit `23fa413`. Claim as recorded there: identical `artifact_sha256 0e0092fe…c4b0` across the D6 release-grade generation and both Phase 6.0 runs; 22/22 arrays field-for-field equal; 398,156 / 383 / 319; no GPU reset and no `GCVM_L2_PROTECTION_FAULT` in the host kernel log. **Caveat:** the Phase 6.0 ROCm run executed the **pre-P0.5** worker (`miner/range_miner_worker.py` at `8e2f5bf`). That citation is evidence about the kernel and the platform, not about the current worker source. |
| **B2 repeated run vs repeated run** | **FRESH — IDENTICAL SEMANTIC ARTIFACT** (22/22 arrays equal, order equal, `artifact_sha256` equal) |
| **B3 `serial_reference` vs `process_sharded`** | **FRESH — IDENTICAL SEMANTIC ARTIFACT** |
| **B4a multi-rig {a,b} vs {b,c}** | **FRESH — IDENTICAL SEMANTIC ARTIFACT.** *The leg with no prior evidence.* |
| **B4b single CUDA GPU vs two ROCm rigs** | **FRESH — IDENTICAL SEMANTIC ARTIFACT** |

**B4 is the new evidence Beta asked for.** Phase 6.0 was single-rig, so "identical results
independent of node assignment" had never been shown. Two arms ran the same trial across two
*different pairs of physical machines* — `{rrig6600, rrig6600b}` and `{rrig6600b, rrig6600c}` —
with stripes dispatched dynamically to whichever worker was free. Both produced the same 22
arrays and the same `artifact_sha256`. B4b then shows the same artifact from one NVIDIA GPU,
so the output is independent of **fleet shape and platform simultaneously**, not just of one.

Rig identity captured on target: all three report `backend=rocm`, `gcnArchName=gfx1032`,
`miner/range_miner_worker.py sha256 0b9a7b86…c55c` — byte-identical to VM 101's working tree.

### Fault-injection control (VIR-2)

**FB1** — one `forward_matches` value bumped in a copy of the baseline bundle, and one hex
nibble flipped in the digest. Comparator verdict **DIVERGENT**, localised to
`forward_matches`, first differing index 0 (`0.33333334` vs `1.3333334`, 1 of 319).
`all_arrays_equal=False`, `artifact_sha256_equal=False`. **Rejected.**

### Provenance binding (§2's list)

| Required | Bound value |
|---|---|
| repository commit and tree state | `76e8eafcae335b9afa81399ada09bc5a93b76dd7`; tracked-clean **False** at report time (this session's edits are uncommitted by design), 4 untracked paths recorded; each arm's `finalize_run` used a throwaway **source-snapshot** commit — see the honesty note below |
| dataset lineage, size and SHA-256 | `daily3-20260801T145551443433Z-513648160d35`, 1,380,711 bytes, `513648160d35…68f6`, 18,068 records; pointer manifest sha256 `ef327526…250a` |
| canonical run-input-manifest digest | `residue_sha256` per stripe, computed by the production coordinator and re-verified by every worker |
| seed-domain contract and exact range | contiguous uint64, half-open `[0, 8,000,000)` |
| PRNG family and variant | `java_lcg`; variants executed `java_lcg` (phase 1, forward constant) and `java_lcg_reverse` (phase 2, reverse constant) |
| window / session selection | window_size 3, offset 0, sessions `[midday, evening]` |
| requested / payload / effective thresholds | forward 0.31, reverse 0.47 — equal on all three legs, per direction, **effective read off the executor**, `validated=True` on the parent-side fail-closed gate for every arm |
| skip mode and effective skip semantics | constant (`test_both_modes=False`); kernel skip range from the payload, default `[0,16]`; skip burned **before the first draw and between every subsequent pair** (inter-draw) |
| assembly backend | `serial_reference` (4 arms) and `process_sharded` pool_size 4 (1 arm) |

**Honesty note on repository identity.** Each arm finalizes against a **throwaway
source-snapshot repo** (HEAD's tracked files with the working-tree `.py` overlaid, committed
there), exactly as the D6 3.B smoke does in its default mode, because `finalize_run` refuses a
dirty tree and an agent may not commit. The recorded SHA identifies a tree byte-identical to the
source that ran; **it is not the project's own commit.** `--release-grade` is the mode that
certifies against the real commit, and it is Michael's to run after committing.

---

## 3. THE MINER KNOWN-ANSWER TRANSFER GATE · **PASS**

### What this is, and what it is not

It is **not** a re-validation of the registry. Beta struck the broad 44-PRNG Wall-C requirement:
known-answer validation was established practice before this repository existed, the method is
valid, the references are genuine, and Michael's account is accepted as the historical project
record. **Nothing here repeats it and nothing here is offered as a substitute for it.**

The subject is **RANGE-MINER**, which did not exist when that work was done. The question is
whether the result **transfers** to the new engine.

### The independent reference — Beta requirement 1

`tests/phase6/known_answer_reference.py`, sha256
**`369bf5c4e5e81523f4d4a396cbcce076a7ee66c08f4e1876a89ce42ceb97f837`**.

Imports `json`, `hashlib`, `struct` — **standard library only**. No `prng_registry`, no
`sieve_gpu_worker`, no `miner.*`, no `utils.*`, no cupy, no numpy. It reads no `kernel_source`
string. It cannot inherit a defect from the code it checks.

It is **not** independent of the kernel *specification*: the four algorithms were transcribed by
hand, this session, from the four live CUDA kernels read out of `KERNEL_REGISTRY` on VM 101. The
gate prints the reference's sha256 next to each live kernel's identity so the transcription can
be re-audited:

| variant | kernel_name | `kernel_source` sha256 (16) | bytes | seed_type |
|---|---|---|---|---|
| `java_lcg` | `java_lcg_flexible_sieve` | `dbad586c55c2d6f7` | 1,775 | uint64 |
| `java_lcg_reverse` | `java_lcg_reverse_sieve` | `47b4e7de5d239686` | 2,075 | uint64 |
| `java_lcg_hybrid` | `java_lcg_hybrid_multi_strategy_sieve` | `b237b8757f0e4c9e` | 3,318 | uint64 |
| `java_lcg_hybrid_reverse` | `java_lcg_hybrid_reverse_sieve` | `6936e6c47b2ffa2c` | 3,027 | uint64 |

### Production semantics — Beta requirement 2

The starting material, `pa_sieve_validation_harness.py` (S143), is misaligned with production in
three ways, all confirmed against the live kernel text:

1. **Seed scramble.** The old harness applies `state = (seed ^ 0x5DEECE66D) & MASK`.
   **Production does not** — every one of the four kernels begins `state = seed & m`, the RAW
   seed masked to 48 bits. The sieve searches raw LCG states, not `java.util.Random` constructor
   arguments.
2. **Output shift.** The old harness uses `>> 17`. **Production uses `>> 16`**, then `& 0xFFFFFFFF`.
3. **Reverse direction.** The old harness steps backwards with a modular inverse. **Production
   has no inverse LCG** — both reverse kernels iterate FORWARD and the direction comes from the
   HOST reversing the residue window (`range_miner_worker.py:887-889`).

Two further production facts, neither in the old harness: **inter-draw skips** (burned before the
first draw *and between every subsequent pair*, not once up front), and the **multi-modulo match**
(mod 1000 AND mod 8 AND mod 125 — redundant, since 1000 = 8 × 125, and transcribed verbatim
anyway, because a reference mirrors the specification rather than improving it).

These are not merely *claimed*. Fault injections F5/F6/F7 re-derive expectations under each wrong
rule and require the comparator to reject them — see below.

Float semantics are exact: every rate and threshold passes through IEEE-754 **binary32** via
`struct`, because the kernels compute `((float)matches)/((float)k)` and compare in float32.
Tie-breaking is transcribed, not assumed: constant kernels keep the first strictly-greater rate
(ties → lowest skip); the forward hybrid keeps the first strictly-greater rate (ties → lowest
strategy id); the reverse hybrid does not maximise at all — it emits on the **first qualifying**
strategy and returns.

### The actual miner worker path — Beta requirement 3

Every population runs `miner.range_miner_worker.SieveExecutor.execute` on a real
`StripeAssignMessage` whose payload was built by the production
`RangeMinerCoordinator.build_stripe_assign_payload`. **The gate constructs no kernel, never calls
`_get_kernel`, and never touches cupy directly.** `_InstrumentedExecutor` overrides only the
executor's own documented single mockable kernel entry `_gpu_launch` — to **record** the
materialised argument vector and then call the real launch. Nothing is stubbed.

**Argument builder — the materialised vector recorded at the real launch site:**

| variant | args | tail after the shared prefix |
|---|---|---|
| `java_lcg` | **14** | `uint64 a=25214903917`, `uint64 c=11`, `int32 offset` |
| `java_lcg_reverse` | **12** | `int32 offset` only (a/c hardcoded in-kernel) |
| `java_lcg_hybrid` | **15** | `uint64 a`, `uint64 c` — **no offset** |
| `java_lcg_hybrid_reverse` | **14** | `int32 offset` |

These match the audited ABI in the worker's own header, observed rather than restated.

**Residue resolution and payload interpretation — five negative controls.** Each corrupts ONE
payload field on the same production `execute` call and requires the documented production
exception. A leg that cannot be made to fail was not doing work.

| control | raised |
|---|---|
| `residue_sha256` mismatch | `ResidueVerificationError` |
| `dataset_sha256` mismatch | `ResidueVerificationError` |
| `dataset_sha256` absent | `ResidueResolutionError` |
| window fields absent | `ResidueResolutionError` |
| contradictory `min_match_threshold` vs `phase2_threshold` | `ThresholdContractError` |

Residue resolution gets a second, structural proof: the gate derives the residue window with the
**independent** reference and stamps *its* sha256 into the payload. The worker then recomputes
from its own canonical derivation and would raise on any disagreement. The two independent
derivations are cross-checked **by production code**, not by assertion.

### Exact-set comparison — Beta requirement 4

> "Planted-seed recovery alone is insufficient because it would not detect extra false survivors."

So the acceptance criterion is the **complete bounded survivor set per population**, compared
three ways — *missing*, **extra**, and per-member value disagreement (match rate, best skip,
strategy id, skip sequence). Planted-seed recovery is reported but is never the criterion.

Two populations per variant: a **DENSE** one (low threshold, hundreds-to-thousands of survivors —
the population that can *see* one extra false survivor) and a **TIGHT** one (high threshold, where
essentially only the plant clears — the population that can see a missed true survivor).

| population | k | thr | seeds | reference | miner | missing | **extra** | value mismatch | plant | payload |
|---|---|---|---|---|---|---|---|---|---|---|
| `java_lcg_dense` | 4 | 0.25 | 50,000 | 3,377 | 3,377 | 0 | **0** | 0 | ✅ | **coordinator payload VERBATIM**, worker default skip `[0,16]` |
| `java_lcg_tight` | 8 | 0.625 | 50,000 | 1 | 1 | 0 | **0** | 0 | ✅ | + `skip_range [0,6]` |
| `java_lcg_reverse_dense` | 4 | 0.25 | 50,000 | 3,363 | 3,363 | 0 | **0** | 0 | ✅ | **VERBATIM** |
| `java_lcg_reverse_tight` | 8 | 0.625 | 50,000 | 1 | 1 | 0 | **0** | 0 | ✅ | + `skip_range [0,6]` |
| `java_lcg_hybrid_dense` | 4 | 0.25 | 25,000 | 1,196 | 1,196 | 0 | **0** | 0 | ✅ | + `strategies` |
| `java_lcg_hybrid_tight` | 8 | 0.625 | 25,000 | 1 | 1 | 0 | **0** | 0 | ✅ | + `strategies` |
| `java_lcg_hybrid_reverse_dense` | 4 | 0.25 | 25,000 | 572 | 572 | 0 | **0** | 0 | ✅ | + `strategies`, offset 3 |
| `java_lcg_hybrid_reverse_tight` | 8 | 0.625 | 25,000 | 1 | 1 | 0 | **0** | 0 | ✅ | + `strategies`, offset 3 |

**8/8 exact-set equal. 8/8 plants recovered at match rate 1.0**, including the exact skip
sequences `(5,6,4,7)` and `(0,2,1,3)` for the two hybrids and the correct `best_skip` for the two
constant variants. Two populations run the coordinator's payload **verbatim** — no key added, no
key changed — so at least one population per constant variant exercises the exact dict production
emits, `[0,16]` default included.

**Anti-vacuity guard, added because this gate failed it once.** The first full-scale run reported
the two reverse populations as "exact-set equal" with the plant absent from *both* sides — a true
statement that certified nothing, and in the TIGHT case an empty-vs-empty comparison. The planted
seed `987_654_321` sat just outside `[987_600_000, 987_650_000)`. `_assert_plant_in_scope` now
makes that arrangement impossible: the plant must lie inside the bounded range, the *independent
reference* must recover it, and the reference set must be non-empty — any failure is
**INCOMPLETE**, never PASS. Reported here because a gate that can go green on nothing is exactly
the VIR-2 failure this project adopted VIR to stop.

**Skip-sequence comparison — a stated bound.** Both hybrid kernels write their per-draw skip
sequence into an *uninitialised* per-thread stack array and emit all `k` entries even when an
early break left the tail unwritten. Every strategy this gate uses pins
`max_consecutive_misses = 999` (far above `k`), so no early break can occur and all `k` entries
are written; the reference reports `n_defined` per seed and the gate asserts `n_defined == k`
before comparing. Comparing further would be comparing uninitialised device memory.

### Controls — Beta requirement 5 · **8/8 rejected**

| # | Injection | Comparator verdict |
|---|---|---|
| F1 | remove one true survivor from the observation | missing=1 |
| F2 | **inject one EXTRA false survivor** — the class a planted-seed test cannot see | extra=1 |
| F3 | perturb one match_rate by **1 binary32 ULP** | mismatch=1 |
| F4 | perturb one hybrid skip-sequence entry | mismatch=1 |
| F5 | expectations under the **java.util.Random constructor scramble** (`pa_sieve_validation_harness` semantics) | missing=3,063 extra=3,182 mismatch=188 |
| F6 | expectations under **`output = state >> 17`** (same harness's semantics) | missing=3,071 extra=3,153 mismatch=212 |
| F7 | expectations with **skip applied ONCE before generating** instead of between draws — the `java_lcg_cpu` semantics, **Beta's own Wall-C caution** | missing=1,433 extra=1,436 mismatch=930 |
| F8 | drive the **production worker** off-spec (`min_match_threshold` 0.25 → 0.50) | missing=3,372 — engine returned 5 survivors vs 3,377 expected |

F5/F6/F7 are what make this a control rather than a self-test: they are the three misalignments
that were *actually* present in the starting material or in `java_lcg_cpu`. **F7 in particular
demonstrates that Beta's Wall-C caution was correct** — a reference built on `java_lcg_cpu`
would have validated the wrong semantics, and the comparator sees the difference at 3,799 seeds.
F8 injects into the miner path itself: the worker really receives, transmits and filters at the
wrong value. The threshold is *raised* rather than lowered because with k=4 the attainable rates
are {0, .25, .5, .75, 1}, so lowering below 0.25 would have to reach 0.0 — a degenerate
"everything survives" case rather than a real signal.

### Failure status — Beta requirement 6

Exit 0 only on PASS. `FAIL → 1`, unhandled exception `→ 2`, `UNAVAILABLE → 3`, `INCOMPLETE → 4`.
`--no-faults` forces INCOMPLETE, because VIR-2 requires the fault-injection control.
The sentinel is printed on every path. **Observed: `SENTINEL: PASS`, exit 0, 124 s.**

### The scope Beta did not ratify — the real shape

Beta declined to ratify Alpha's "~20 lines, an afternoon" estimate for aligning
`pa_sieve_validation_harness.py`, and was right. The real work, measured:

| Component | Actual |
|---|---|
| The independent reference (4 algorithms, float32 semantics, planting helpers, residue derivation + fingerprint reimplemented) | **~430 lines, new file.** Not a 20-line edit of the S143 harness: three semantic misalignments, two undocumented production behaviours (inter-draw skips, multi-modulo), and two *structurally different* hybrid algorithms — the reverse hybrid has an offset the forward one lacks, restores state on a miss where the forward one does not, uses `>` where the forward uses `>=` for its miss budget, and does not maximise at all. |
| The four miner ABI paths | Not covered by the estimate at all. Required a real coordinator-built payload per variant, a synthetic conforming dataset per population (including the discovery that `entry.get("full_state", entry["draw"])` evaluates `entry["draw"]` **eagerly**), and an instrumented executor that records the launch without stubbing it. |
| The exact-set comparison | Not covered. Required a dense/tight population design, per-member value comparison across four different result shapes, the `n_defined` bound on uninitialised skip-sequence memory, and the anti-vacuity guard. |
| The VIR controls | Not covered. Five worker-path negative controls plus eight fault injections, three of which are re-derivations under wrong semantics and one of which drives production off-spec. |
| **Total** | **1,382 lines across two new files** (`known_answer_reference.py` 450, `known_answer_gate.py` 932), plus a full-scale run of ~2 minutes and one defect found in the gate's own first version. Not an afternoon's ~20 lines. |

---

## 4. THE RandomSampler CONTROL ARM · **PASS (NON-CERTIFYING)**

### The neutral entrypoint

Beta's direction, followed exactly: **do not route all samplers through a permanently named
`run_bayesian_optimization()`.**

`window_optimizer_bayesian.OptunaBayesianSearch.run_optimization(..., sampler, sampler_metadata)`
is now sampler-agnostic. Both new arguments are **required and keyword-only**, so a caller cannot
get TPE by omission and then report the run as something else. `search()` is the thin TPE
entrypoint (unchanged signature, unchanged behaviour, `multivariate=True` preserved);
`OptunaRandomSearch.search()` is the equally thin RandomSampler entrypoint. Search space,
objective wrapper, warm start, pruner, storage, incremental save, telemetry and result shape are
shared **by construction** — which is what makes the two arms comparable.

The result dict's `strategy` key now reports the sampler that **actually chose the points**
instead of a hardcoded `'optuna_bayesian'`, and carries a `sampler` block. A RandomSampler run
that self-reported as Bayesian was the mislabelling Beta ruled out; a **labelling control** in the
harness checks all 10 arms and passed 10/10.

This also closes, without touching them, the two defects that made the old `--strategy random`
path unusable as a control: `window_optimizer.RandomSearch.search` cannot accept the kwargs
`WindowOptimizer.optimize` forwards, and `GridSearch`/`EvolutionarySearch` have vacuous
`return {}` bodies. Per the standing rule those classes stay **GATED, not deleted** — verified
still gated this session (`bayesian` callable; `random`, `grid`, `evolutionary` raise
`StrategyContractError`). The control arm no longer needs them: it is an Optuna study with a
different sampler, not a different code path.

**Autonomous sampler selection is NOT approved and is NOT built.** Nothing in the harness or in
`window_optimizer_bayesian.py` reads an advisor recommendation, `strategy_recommendation.json`,
or a WATCHER policy to pick a sampler. Both arms are named on the command line.

### The comparison — matched budgets, 5 deterministic seeds, distributions

Objective: a **real** bidirectional sieve per trial (forward + reverse) on the RTX 3080 Ti through
the same production `SieveExecutor` path §3 certifies; score = bidirectional intersection count
via `window_optimizer.BidirectionalCountScorer`.

Calibrated for responsiveness **before** the comparison, so the objective is known non-degenerate
rather than assumed: over 8,000,000 seeds the bidirectional count spans **161,371** (window 2,
skip `[0,10]`, thresholds 0.40/0.40) down to **0** (window 8, same skip and thresholds). An
all-zero objective would have made the whole comparison vacuous.

| Bound | Value |
|---|---|
| sampler classes | `optuna.samplers.TPESampler` (multivariate) vs `optuna.samplers.RandomSampler` |
| Optuna version | 4.4.0 |
| sampler seeds | 0, 1, 2, 3, 4 |
| trial budget | **24 per arm**, matched |
| warm-start mode | none (`trial_history_context=None`, `resume_study=False`) — identical in both arms |
| objective | bidirectional intersection over `[0, 8,000,000)`, phases 1 and 2, constant skip |
| effective search-space digest | `34e53e66d9a89fca9e7a1aaf5eab7cd2200ddadc992f600593e32fc1544b59f1` |
| effective search space | `window_size int[2,5]`, `offset int[0,100]`, `session_idx int[0,2]`, `skip_min int[0,4]`, `skip_max int[max(skip_min,5),20]`, `forward_threshold float[0.4,0.75]`, `reverse_threshold float[0.4,0.75]` |
| effective skip semantics | **constant only.** Sampled `skip_min`/`skip_max` are placed in the payload's `skip_range` and reach `for (int skip = skip_min; skip <= skip_max; skip++)`. |
| repository commit / tree | `76e8eaf…`, tracked tree not clean at report time |
| dataset | `daily3-20260801T145551443433Z-513648160d35`, sha256 `513648160d35…68f6`, 18,068 records |
| study identity | one fresh Optuna study per arm, name recorded per arm in the JSON |

**Results — distributions, not the best trial alone:**

| | best-trial objective across 5 seeds | pooled per-trial (120 trials/arm) |
|---|---|---|
| **TPE** | values `[107, 126, 162216, 146378, 145186]` · min 107 · **median 145,186** · mean 90,803 · max 162,216 · sd 83,057 | min 0 · median 2.0 · mean 13,095 · max 162,216 · **73/120 non-zero** |
| **RandomSampler** | values `[76, 53, 113351, 92, 1]` · min 1 · **median 76** · mean 22,715 · max 113,351 · sd 50,667 | min 0 · median 0.0 · mean 1,354 · max 113,351 · **20/120 non-zero** |

Index *i* of each list is sampler seed *i*. On that alignment **TPE is ahead at all five seeds**
(107 > 76 · 126 > 53 · 162,216 > 113,351 · 146,378 > 92 · 145,186 > 1). Stated with the caveat it
deserves: the same integer seed does **not** produce the same draws in two different sampler
classes, so this is a nominal alignment, not a matched-pairs design, and 5 seeds cannot carry a
significance claim either way.

**Reading this honestly.** TPE is ahead on every summary statistic, and the exploitation signal is
clear — 73/120 of its trials landed on a non-zero objective against random's 20/120, and its
pooled median is 2.0 against 0.0. But the best-trial distributions **overlap heavily** and both
have a standard deviation on the order of their mean: 2 of 5 TPE seeds and 4 of 5 random seeds
never found the high-value basin at all. **With n = 5 this is a direction, not a significance
claim, and Alpha does not present it as one.**

### THE DEAD-DIMENSION CAVEAT — and a sequencing question for Beta

`skip_min`/`skip_max` remain **dead on the hybrid path** (skill 2.7 #4): the sampled values
survive eight hops and die at `_hybrid_prefix`, because no hybrid kernel declares skip bounds and
`expected_skip` is hardcoded to 5. A sampler comparison that includes hybrid phases therefore
searches a **falsely seven-dimensional space** and measures noise in two of its seven dimensions.

**This run is constant-skip only, so all seven of its dimensions are live** — the constant kernels
really do iterate the sampled skip range, and §3 demonstrates the survivor set changing with it.
That is a *narrower* claim than it may look, and the narrowing is the point:

* the comparison is valid **for a constant-skip run**;
* it does **not** represent a production `--test-both-modes` run, whose hybrid legs carry two
  dead dimensions;
* so this TPE-vs-random result **must not be generalised** to the full four-phase workflow.

Two further bounds, stated rather than hidden: the search space was **narrowed** (production
allows `window_size ≤ 50` and `skip_max ≤ 250`, which is ~1.6 M LCG steps per seed and cannot be
swept across a matched multi-seed budget on one GPU — threshold bounds are unchanged), and the
harness drives one local GPU, not the fleet, because an agent may not launch the pipeline.

> **ALPHA'S RECOMMENDATION TO BETA — a decision Alpha does not take.**
> **Sequence the certifying sampler comparison AFTER the skip-output work.** The neutral
> entrypoint is done and is not blocked by anything; it can land now. But any comparison intended
> to *govern* sampler choice for production should run over the four-phase workflow, and today
> two of that workflow's seven dimensions are connected to nothing on the hybrid legs. Running the
> certifying comparison now would either (a) restrict it to constant skip, which does not govern
> the production workflow, or (b) include hybrid phases and let an autonomous-facing decision be
> influenced by two knobs wired to a void — the exact failure mode skill §0.5 names. Alpha
> therefore submits this run as the **neutral-entrypoint proof plus a constant-skip datapoint**,
> explicitly **non-certifying**, and asks Beta to schedule the governing comparison after the skip
> work.

---

## 5. Out of scope — confirmed untouched

Per §5: the skip-output work, the Resolved Execution Set and Beta's Q1 local-run refinement,
autonomous sampler selection, `grid`/`evolutionary` samplers, the `java_lcg_cpu` non-zero-skip
mismatch (`survivor_scorer.py:124` / `full_scoring_worker.py:305` — Beta ruled this a separate
bounded audit, no fix authorized), D6.2, D6.3, the scraper, session-split dataset authority.
**None was modified.** `java_lcg_cpu` is *referenced* in §3 only as the source of fault injection
F7's wrong semantics; the file itself is untouched and no fix was attempted.

---

## 6. The two ordered corrections — DONE

### `reverse_kernel_test_results.txt` — marked SUPERSEDED IN PLACE

Original 20 rows **preserved verbatim**; the header is the only addition. It states:
`SUPERSEDED — NOT VALIDATION EVIDENCE` · all results were `BOTH ZERO` · **no positive control was
established**, so the run cannot distinguish "the kernels are correct and there were no survivors"
from "something returns zero unconditionally" · **prohibited from citation under VIR-2** · and it
names the replacement gate (`tests/phase6/known_answer_gate.py`) with a list of exactly what the
replacement supplies that this file lacks. The scope difference (20 variants → 4) is stated as
deliberate, not a regression: twenty rows of zeros is not broader coverage than four exact-set
comparisons, it is no coverage at all.

**One in-scope addition Alpha judged necessary and flags for review.** `quick_test_all_22.sh`
**generates** that file and truncated it (`> $RESULTS_FILE`), so the next run of that script would
have silently undone Beta's order. Its output now goes to a timestamped file and the superseded
record is never touched. No other behaviour changed.

**Documents citing it:** `docs/KNOWN_ANSWER_VALIDATION_INVENTORY.md` and
`docs/TEAM_ALPHA_WALL_C_SUBMISSION.md` already describe it correctly as vacuous and
non-citable; no correction was needed there.

### `test_ALL_46_prngs_10M.sh` — header and category counts corrected

Verified this session against the live `KERNEL_REGISTRY` on VM 101:

* the `PRNGS` array contains **44** entries, not 46;
* they are **11 + 11 + 11 + 11** — both "Reverse" comments said 12 and both had 11;
* all 44 are unique, **all 44 are valid registry names**, and they cover `KERNEL_REGISTRY`
  **exactly** (`len(KERNEL_REGISTRY) == 44`, set difference empty in both directions).

Corrected in place: the echoed header (`46 → 44`), both `(12) → (11)` comments, and the
`$SUCCESS/46` / `$FAIL/46` summary lines. A header block records what was verified and states
that the script is a liveness sweep with no expected answer and **must not be cited as
correctness evidence**. The **filename is deliberately left alone** — renaming it would break
every existing reference for no gain.

**Beta was right and Alpha was wrong.** The claim that this script "contains two invalid registry
names" and "would hard-fail" is **FALSE**. It appeared in two documents and both are now corrected
in place with a dated `[S184 CORRECTION]` note rather than silently rewritten:

* `docs/KNOWN_ANSWER_VALIDATION_INVENTORY.md` §1.5, the registry-count section, and the
  runnability table (three separate places);
* `docs/TEAM_ALPHA_WALL_C_SUBMISSION.md` §6 item 5.

---

## 7. Verification-integrity controls (VIR-1…6)

```
execution proof:      every claim above carries an artifact digest, a JSON evidence record, or a
                      file:line read in this session. Effective thresholds are read OFF THE
                      EXECUTOR (SubStripeOutcome.effective_threshold), never recomputed from
                      config. Kernel argument vectors are recorded AT the real launch site.
clean control:        Wall A on a healthy certified generation (7/7 legs). Wall B: five healthy
                      arms, four comparisons. §3: eight healthy populations, exact-set equal.
                      §4: an objective calibrated non-degenerate before use.
fault-injection
control:              Wall A 5/5 rejected · Wall B FB1 rejected · §3 8/8 rejected (including
                      three re-derivations under genuinely wrong semantics and one that drives
                      the production worker off-spec) · §3 path evidence 5/5 production
                      exceptions raised · §4 labelling control 10/10.
completion sentinel:  PASS | FAIL | UNAVAILABLE | INCOMPLETE printed per gate; non-zero exit on
                      anything but PASS.
unavailable-observer
behaviour:            B1 (CUDA vs ROCm) is CITED and explicitly NOT re-run, with the pre-P0.5
                      worker caveat attached. `--no-faults` forces INCOMPLETE. A population whose
                      plant is out of scope, whose reference set is empty, or whose hybrid skip
                      sequence has undefined entries is INCOMPLETE, never PASS.
audit claim scope:    SEARCHED — the live repository at 76e8eaf on VM 101; the live
                      KERNEL_REGISTRY; the live source deployed on all three CT100s (per-file
                      sha256 on target); the RTX 3080 Ti; RX 6600 XT device 0 on rrig6600,
                      rrig6600b and rrig6600c.
                      OBSERVED ON A RIG: source identity, backend=rocm, gcnArchName=gfx1032,
                      dataset digest, and the B4a/B4b comparison results.
                      VM-101-ONLY: Wall A in full, B2, B3, §3 in full, §4 in full.
                      UNAVAILABLE surfaces: the CT100 in-container kernel ring buffer (dmesg is
                      not permitted in an unprivileged LXC and amdgpu lives in the Proxmox host
                      kernel), so no fresh host-side GPU-fault scan was performed this session —
                      the Phase 6.0 "no GCVM_L2_PROTECTION_FAULT" claim is CITED, not re-run.
                      Also not exercised: hybrid phases in any Wall-A/Wall-B run (constant skip
                      only), `--release-grade` finalization, and the 26-GPU fleet.
```

---

## 8. Non-regression

Run on VM 101, `~/venvs/torch`, at `76e8eaf` plus this session's uncommitted edits.

**22 suites, 22 exit 0, every §8 tally at its pinned figure.**

| Suite | Exit | Elapsed | Tally reported by the suite |
|---|---|---|---|
| `tests/test_prng_encoding.py` | **0** | 0 s | — |
| `tests/test_s172_phase1_scaffolding.py` | **0** | 3 s | — |
| `tests/test_s172_phase2_protocol.py` | **0** | 3 s | — |
| `tests/test_s172_phase3_worker.py` — **Phase 3** | **0** | 41 s | **17/17 gates**, "all gates green" |
| `tests/test_s172_phase4_coordinator.py` — **Phase 4, incl. Gate 22** | **0** | 58 s | **63/63 checks**, "all checks green" |
| `tests/test_s172_phase5_d0.py` | **0** | 3 s | — |
| `tests/test_s172_phase5_d1_engine.py` — **D1.1** | **0** | 215 s | — |
| `tests/test_s172_phase5_d1_workflow.py` | **0** | 104 s | — |
| `tests/test_s172_phase5_d2_directional_uniqueness.py` | **0** | 434 s | — |
| `tests/test_s172_phase5_d3_0_encoding_contract.py` | **0** | 1 s | — |
| `tests/test_s172_phase5_d3_columnizer.py` | **0** | 0 s | — |
| `tests/test_s172_phase5_d3_25_candidate_ingress.py` | **0** | 2 s | — |
| `tests/test_s172_phase5_d3_5_finalizer.py` | **0** | 8 s | — |
| `tests/test_s172_phase5_d4_serial_backend.py` — **D4** | **0** | 1 s | — |
| `tests/test_s172_phase5_d5_process_sharded.py` — **D5** | **0** | 746 s | **24/24**, "field-for-field equivalent to `serial_reference`" |
| `tests/test_s172_phase5_d6_production_adapter.py` — **D6 3.A** | **0** | 16 s | **9/9**, "all D6 3.A gate checks green" |
| `tests/test_s172_phase5_d6_threshold_path.py` — **D6-threshold** | **0** | 7 s | **17/17** |
| `tests/test_s172_d6_1_flush_durability.py` — **D6.1** | **0** | 1 s | — |
| `tests/test_s172_threshold_propagation.py` — incl. **G-MINER-UNCHANGED** | **0** | 12 s | **5/5** |
| `tests/test_chapter1_p0_corrections.py` — **Chapter1-P0** | **0** | 19 s | **12/12 checks** |
| `tests/test_s172_phase6_p05_dataset_authority.py --fleet` — **P0.5** | **0** | 14 s | **38/38 checks**, "RESULT: PASS" |
| `tests/test_s172_admission_liveness.py` — **admission liveness** | **0** | 123 s | **16/16**, "RESULT: PASS" |

Note that **P0.5 `--fleet` 38/38 passed**, which independently re-confirms the dataset is present
and digest-correct on all three CT100s at the time of this submission — the same fleet the
Wall-B multi-rig arms ran on.

### Gate 22 and G-MINER-UNCHANGED

**`G-MINER-UNCHANGED` needs no new registration.** This deliverable does not touch `miner/`,
`sieve_gpu_worker.py`, `prng_registry.py` or `persistent/pwc_protocol.py` at all — those files are
byte-unchanged. **P0.5's strengthening of that gate (which greps registered diffs for threshold
tokens) is left exactly as it was.**

**Gate 22** was extended by **appending** a registration block to its `allowed` set — nothing
earlier was rewritten. Registered, with rationale in the file:

* `window_optimizer_bayesian.py` — the one production file, §4's sampler-neutral extraction. Not
  in this gate's protected surface. No search-space, objective, threshold or warm-start change,
  and no autonomous sampler selection.
* `tests/phase6/known_answer_reference.py`, `known_answer_gate.py`, `wall_ab_gate.py`,
  `sampler_control_arm.py` — four new test paths, no production code.

No gate was added to `test_s172_phase4_coordinator.py`, so **its 63/63 tally is unchanged** —
the file's own note warns that growing it would silently move a number other documents cite.

---

## 9. Findings

### 9.1 `process_sharded` and the §6.7.A GPU-context guard — a HARNESS defect, not a production one

The first Wall-B run failed the `process_sharded` arm with `BrokenProcessPool` → "no committed
assembly". Root cause, established by reproduction rather than inference:

D5 builds its pool with the **spawn** start method, and a spawn child re-imports the parent's
`__main__` module before running its task. `assembly_shard_worker` enforces §6.7.A — *assembly is
CPU-only work and a worker must never hold a GPU context* — by refusing any shard worker that
finds a GPU module in `sys.modules` (`ShardArtifactError`). `wall_ab_gate.py` imported
`window_optimizer_integration_final` at module level, **and WOI imports cupy at module level**, so
every spawn child inherited cupy through the `__main__` re-import and the guard correctly killed
the arm.

**Measured both ways, not assumed:**

| module, imported alone | `cupy` in `sys.modules` afterwards |
|---|---|
| `window_optimizer_integration_final` | **True** |
| `window_optimizer` (the real Step-1 `__main__`) | **False** |
| `utils.run_finalizer`, `utils.survivor_loader`, `miner.range_miner_coordinator`, `miner.step1_ingress` | False |

`window_optimizer.py` imports WOI **lazily**, inside `run_bayesian_optimization`, so a production
spawn child re-imports a cupy-free `__main__` and the guard passes. **`process_sharded` is not
broken in production.** The gate now uses lazy `_woi()` / `_d6()` accessors and stays cupy-free at
import time; with that fix B3 passes and produces a byte-identical artifact.

Two things worth Beta's attention: the guard is **load-bearing and it worked**, catching a real
GPU-context leak into a CPU-only worker; and the property that makes `process_sharded` usable —
*the Step-1 `__main__` must not import cupy* — is currently **implicit and untested**. Alpha
recommends (does not implement) a small gate asserting `cupy not in sys.modules` after importing
`window_optimizer` as `__main__`. That is a one-line invariant guarding a whole backend.

### 9.2 Rig source deployment was stale and had to be brought current

Before this session, `rrig6600` (.122) carried the repository at **`8e2f5bf`** and
`rrig6600b`/`rrig6600c` carried **no source at all** — only the frozen dataset. Digest comparison
showed `miner/range_miner_worker.py` differing between VM 101 and .122 (the P0.5
`DatasetProvisioningError` classification); the other four key files were identical.

The brief's claim is nonetheless **correct**: all three CT100s did hold the frozen dataset at
`513648160d35…68f6`, verified on target. What they did not hold is the *code*. `daily3.json` and
`dataset_provisioning.json` are gitignored and the rigs had never been given the current tree.

Alpha deployed `git archive HEAD` + the working-tree `.py` overlay to
`/home/michael/distributed_prng_analysis` on all three rigs and verified **per-file sha256 on
target**; all five key files now match VM 101 byte-for-byte, and
`import miner.range_miner_worker` succeeds on each rig reporting 24 validated variants.

### 9.3 The Phase 6.0 artifact digest reproduced today

Every one of the five Wall-B arms produced `artifact_sha256`
**`0e0092feeb02e22d28557ddf4d8e421941d6117bcc0448d7f7323ec402c1c4b0`** — the same digest as the
authoritative pre-dataset-provenance generation (`b08c2c5`) and both Phase 6.0 runs. That is an
independent reproduction, today, at `76e8eaf`, with the **current post-P0.5 worker**, on both
platforms and across three physical machines. It is reported as an observation, not as a
re-run of Phase 6.0.

---

## 10. Open item Alpha could not close — needs Michael

**Stray repository extraction in `$HOME` on all three rigs.** Alpha's first deployment command
omitted a `cd` and extracted the source tarball into `/home/michael/` instead of
`/home/michael/distributed_prng_analysis/`, creating **837 stray top-level entries** in each
rig's home directory. The correct deployment was then done separately and verified on target, so
**nothing in the run depended on the stray copy** and the multi-rig evidence is unaffected.

Alpha's cleanup command was **denied by the sandbox** (recursive removal over SSH), so the stray
files are still there. Verified before proposing removal: on each rig, every one of the 837 names
is *newly created* (mtime ≥ the deploy moment) and none of them is `distributed_prng_analysis`,
`rocm_env`, or any pre-existing entry.

Suggested cleanup, for Michael to run from VM 101 (per host):

```bash
# on VM101 — repeat for 192.168.3.156 and 192.168.3.164
ssh michael@192.168.3.122 '
  cd /home/michael
  find . -maxdepth 1 -mindepth 1 -newermt "2026-08-01 17:40" -printf "%f\n" | LC_ALL=C sort > /tmp/new_top.txt
  # keep the repo, the venv and the spool dir; everything else new is the stray copy
  grep -vxE "distributed_prng_analysis|rocm_env|s172_wallb_spool_gpu0" /tmp/new_top.txt > /tmp/rm_list.txt
  wc -l /tmp/rm_list.txt          # expect ~837 — REVIEW THIS LIST BEFORE THE NEXT LINE
  while IFS= read -r e; do rm -rf -- "/home/michael/$e"; done < /tmp/rm_list.txt'
```

**One caveat Alpha cannot undo:** the stray extraction overwrote any same-named pre-existing
file in `$HOME`, including dotfiles the repository happens to carry at top level
(`.gitignore`, `.tmux.conf`, `.hash_local.txt`, `.hash_remote.txt`, `.recovery`). If a rig had its
own `.tmux.conf`, it is now the repository's version. Flagged explicitly rather than left to be
discovered.

---

## 11. What Alpha asks Beta to rule on

1. **Wall A, Wall B and the Miner Known-Answer Transfer Gate** — accept or return, on the evidence
   above. All three carry PASS sentinels, clean controls and fault-injection controls.
2. **§4 sequencing** — Alpha recommends landing the neutral `run_optimization` entrypoint now and
   **sequencing the certifying sampler comparison after the skip-output work** (§4). The run
   submitted here is explicitly non-certifying.
3. **The implicit `process_sharded` precondition** (§9.1) — whether the "Step-1 `__main__` must not
   import cupy" invariant should be gated. Alpha recommends yes and has not implemented it.
4. **The `quick_test_all_22.sh` output-path change** (§6) — an Alpha judgment call made to keep
   Beta's supersession order durable; confirm or reverse.

---

## 12. Artifacts produced

**New (all untracked; Michael commits and dual-pushes):**

```
tests/phase6/known_answer_reference.py     independent stdlib reference, 4 java_lcg variants
tests/phase6/known_answer_gate.py          §3 transfer gate
tests/phase6/wall_ab_gate.py               §1 + §2
tests/phase6/sampler_control_arm.py        §4
docs/phase6_evidence/known_answer_gate.json
docs/phase6_evidence/wall_ab.json
docs/phase6_evidence/sampler_control_arm.json
docs/TEAM_ALPHA_BOUNDED_PHASE_6_SUBMISSION.md   (this file)
```

**Modified:**

```
window_optimizer_bayesian.py               §4 sampler-neutral extraction
tests/test_s172_phase4_coordinator.py      Gate 22 registration (appended)
reverse_kernel_test_results.txt            §6 superseded in place, rows preserved
test_ALL_46_prngs_10M.sh                   §6 header + category counts
quick_test_all_22.sh                       §6 durability of the supersession
docs/KNOWN_ANSWER_VALIDATION_INVENTORY.md  §6 three [S184 CORRECTION] notes
docs/TEAM_ALPHA_WALL_C_SUBMISSION.md       §6 one [S184 CORRECTION] note
```

**Not committed, not pushed. WATCHER not run. The pipeline not launched. STOP at the gate.**
