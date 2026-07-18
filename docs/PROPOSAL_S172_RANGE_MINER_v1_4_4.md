> **SUPERSEDED — retained for audit trail.**
> This document is superseded by PROPOSAL_S172_RANGE_MINER_v1_4_5.md, which absorbs
> the binding Team Beta S175 ruling (remote spool staging, staged A+C parallel
> assembly, high-survivor acceptance, three-way verification). Where v1.4.4 and
> v1.4.5 conflict, **v1.4.5 governs.** v1.4.4 remains authoritative only for sections
> v1.4.5 explicitly marks PRESERVED / carried forward unchanged.

---

# PROPOSAL_S172_RANGE_MINER_v1_4_4.md
**End-to-end miner-style architecture to replace Step 1 window_optimizer chunked dispatch.**

| Field | Value |
| --- | --- |
| Version | **v1.4.4** (TB final-check cleanup — SPEC FROZEN) |
| Author | Team Alpha (Michael / Claude) |
| Reviewer | Team Beta |
| Status | **APPROVED — spec frozen, ready for Phase 0 implementation** |
| Prior versions | v1.0 · v1.1 · v1.2 · v1.3 · v1.4 · v1.4.1 · v1.4.2 · v1.4.3 (approved with cleanup pending) |
| Verification HEAD | `b78d76c` on `mmercalde/prng_cluster_public` |
| Scope | Step 1 **replacement** (mutually exclusive with PWC; opt-in via `--use-range-miner`) |
| Out of scope | Steps 2–6 (Step 1 output contract is frozen — see §4) |

---

## Version diff: v1.4.3 → v1.4.4

TB final check on v1.4.3 ruled **APPROVED with one tiny cleanup before commit**. v1.4.4 applies that cleanup; no substantive changes.

**The cleanup.** Two textual leftovers from v1.4.2 that the v1.4.3 erratum patch did not propagate:

1. **§10 Phase 3 description** still said "7 family kernel-args branches + `NotImplementedError` for the 4 uncovered" — the "4 uncovered" miscount, identical to the §5.3 error TB caught in v1.4.3. Now reads: "7 hardcoded branch patterns covering 6 base families + `NotImplementedError` for the 5 uncovered base families."
2. **§14 header** said "commands updated for v1.4.2." Bumped to "v1.4.4."

After these, **spec is frozen**. Implementation begins with Phase 0.

---

## Version diff: v1.4.2 → v1.4.3 (preserved for audit trail)

TB re-audited v1.4.2 and ruled **APPROVED WITH ONE REQUIRED ERRATUM**. v1.4.3 applies only that erratum; all other sections are preserved verbatim from v1.4.2.

**The erratum.** Two related inconsistencies in v1.4.2:

1. **§5.3 count error.** v1.4.2 said "4 registered base families" (mt19937, xorshift64, xoshiro256pp, philox4x32, sfc64) — but the list contains **5 names**. Correct count: **5 uncovered base families**.
2. **§11.I covered-family error.** v1.4.2 named `mt19937` as an acceptance PRNG ("java_lcg, lcg32, mt19937 — three of the 7 covered-by-worker families"). **`mt19937` is in the uncovered set**, not the covered set. The current worker (`sieve_gpu_worker.py:208-306`) has 7 hardcoded family branches covering **6 distinct base families** (`xorshift32`, `pcg32`, `lcg32`, `java_lcg`, `minstd`, `xorshift128`) plus 3 variants of `java_lcg` (`_reverse`, `_hybrid`, `_hybrid_reverse`). The 7-branch count includes the 3 java_lcg variants; the base-family-coverage count is 6.

Both errors corrected below. TB §11.I rewording also adopted: uncovered families **must fail fast with `NotImplementedError`** until a dedicated kernel-args builder is implemented and tested.

No other changes vs v1.4.2.

---

## Version diff: v1.4.1 → v1.4.2 (preserved for audit trail)

TB audited v1.4.1 against the live repo and ruled **NOT APPROVED AS WRITTEN, APPROVED DIRECTIONALLY**. v1.4.2 absorbs every TB correction. The corrections fall in five categories:

1. **§4.1 file-count language** — "7 files" replaced with **"8 mandatory + 1 optional fallback"** throughout. v1.4.1's own enumeration listed 8; some prose still said 7. TB ruling §2.
2. **§4.4 NPZ field classification** — v1.4.1's "Tier 3: 15 fields written but never read" claim is **wrong** for Step 3. The active Step 3 driver `generate_step3_scoring_jobs.py:62-102` iterates **every** NPZ array into per-survivor chunk dicts with a hard `METADATA LOSS DETECTED` guardrail at line 95-100. Per-seed metadata is passed into feature extraction at `full_scoring_worker.py:321, 379-380, 411, 451-452, 472, 582-584`. TB ruling §3, §6. **All 22 NPZ arrays are part of the live consumer surface.** No "warning tier" in §11.
3. **§5.4 PRNG_TYPE_ENCODING** — downgrade to "cosmetic" is reverted. `utils/survivor_loader.py:42-49, 356-357` decodes NPZ uint8 → string on `return_format="dict"`, then `generate_step3_scoring_jobs.py:81-93` preserves the string into chunk metadata. The encode→decode round-trip silently destroys `*_hybrid` provenance (encode collapses to `0`, decode emits `'java_lcg'`). **Fix is required, not deferred.** TB ruling §4, Q1.
4. **§11 acceptance criteria** — rewritten per TB §10 — A through G. Includes new **NPZ→dict identity test** that v1.4.1 missed, mandatory Step 2 + Step 3 output identity, and Step 4 partial-compatibility ruling.
5. **§13 outstanding questions** — TB Q1-Q4 rulings absorbed. Q1: fix encoding. Q2: per-family caps approved. Q3: one retry, then fail trial. Q4: preserve drift in S172.

v1.4.2 also introduces a new **§12.1 — EXPECTED_NPZ_KEYS contract wall** that every miner-emitted NPZ must pass before being written to disk. TB §11 mandates this as the implementation rule.

---

## 0. Why this proposal exists

v1.3 was written against documentation. v1.4 corrected to code-verified producer-side language. v1.4.1 added the consumer-surface audit. **v1.4.2 corrects v1.4.1's errors** caught by TB:

| v1.4.1 claim | Truth in code (HEAD b78d76c, audited by TB) |
| --- | --- |
| "15 NPZ fields written but never read" | **Wrong for Step 3.** `generate_step3_scoring_jobs.py:81-93` reads every NPZ key into per-survivor metadata dicts. All 22 arrays are part of the live consumer surface. |
| "PRNG_TYPE_ENCODING silent collapse is cosmetic" | **Wrong.** Round-trip Step 1 encode → NPZ → `survivor_loader._array_to_dict` decode (line 356-357) → Step 3 chunk metadata → `full_scoring_worker.survivor_metadata` (line 321) silently destroys `*_hybrid` provenance. |
| "7 mandatory output files" (in some prose) | **Inconsistent.** v1.4.1's own §4.1 table lists 8. Standardised to **8 mandatory + 1 optional fallback metadata**. |
| Active Step 3 driver = `generate_full_scoring_jobs.py` | **Wrong.** Active driver is `generate_step3_scoring_jobs.py`, invoked from `run_step3_full_scoring.sh:160`. The older `generate_full_scoring_jobs.py` is dead code per Jan 23 TB ruling (referenced only by superseded `run_full_scoring.sh:31`). |
| "§11.1 array-by-array binary identity" | **Insufficient.** TB §6 requires NPZ→dict identity as a separate, stronger criterion. |

Original v1.4 corrections (preserved):

| v1.3 claim | Truth |
| --- | --- |
| "9 output files (6 JSON + 3 NPZ)" | **8 mandatory + 1 optional fallback metadata** — see §4.1. |
| "22-field schema gate on the survivor record" | The gate is **22 NPZ arrays** enforced at the writer (`convert_survivors_to_binary.py:152-176`, `window_optimizer_integration_final.py:1783, 1786`). |
| "46-PRNG registry" | **44 PRNG entries** = 11 base × 4 variants (`prng_registry.py:3729-4135`). |

---

## 1. Executive summary

**Goal.** Replace the chunk-dispatch backend used by Step 1 (`coordinator.execute_distributed_analysis` over PWC-SSH or PWC-TCP) with a miner-style, range-based engine: every GPU runs a long-lived daemon that pulls 60 M-seed stripes, executes them in sub-stripes sized to fit VRAM and the watchdog ceiling, and streams survivors back through the existing 22-array NPZ contract.

**Why miner.** D1.1 (2026-05-11) was the third multi-rig fault under PWC at 8 active GPUs per rig. Owner directive: stop debugging the cliff, switch backends.

**Mutual exclusion.** `--use-range-miner` and `--use-persistent-workers` are mutually exclusive at the CLI level. The current code has an additive-gate pattern at `window_optimizer_integration_final.py:346` (PWC) and `:392` (ZMQ-SQLite); the miner uses the same pattern as a third top-level gate that runs before PWC.

**TB-ruled invariant (v1.4.2).** Per TB §1:

> Range-Miner is allowed to change *how* Step 1 computes, but not *what* Step 1 emits. Every downstream path must see the same filenames, schemas, data types, key names, value semantics, and fallback behavior as current Step 1.

**Step-1 output contract.** Frozen by TB ruling and now verified at the byte level + semantic level on both producer and consumer sides (§4 + §4.4). The miner is a backend swap — not a contract change.

---

## 2. Verification methodology

Producer side (v1.4): 15 priority files. Consumer side (v1.4.1): 5 additional files. **v1.4.2 adds 2 more files** that v1.4.1 missed:

| File | Lines | Verified | Why missed in v1.4.1 |
| --- | --- | --- | --- |
| `generate_step3_scoring_jobs.py` | 414 | Step 3 driver — full NPZ-to-dict metadata preservation at `:62-102`; guardrail at `:95-100` | v1.4.1 cited the older `generate_full_scoring_jobs.py` (dead code per Jan 23 TB ruling) |
| `utils/survivor_loader.py` (lines 321-389) | (re-read) | `_array_to_dict` decode of all 22 fields including `PRNG_TYPE_DECODING` round-trip at `:42-49, 356-357` | v1.4.1 read the version-detect block only, not the conversion block |

Combined producer + consumer + v1.4.2 additions: **22 478 + 800 ≈ 23 278 LoC inspected**. Every numerical claim and naming convention is followed by a `file:line` citation.

---

## 3. Architecture overview

```
                            ┌────────────────────────────────────────┐
                            │ window_optimizer.py main()             │
                            │   argparse → run_bayesian_optimization │
                            └───────────────┬────────────────────────┘
                                            │ (CLI flags wired onto coordinator
                                            │  at window_optimizer.py:614-625)
                                            ▼
                            ┌────────────────────────────────────────┐
                            │ MultiGPUCoordinator(...)               │
                            │ (coordinator.py:232)                   │
                            │                                        │
                            │ ────── attribute gates ──────          │
                            │ coordinator.use_persistent_workers     │
                            │ coordinator.use_zmq_sqlite             │
                            │ coordinator.use_range_miner   ◄── NEW  │
                            └───────────────┬────────────────────────┘
                                            │
                                            ▼
              ┌─────────────────────────────────────────────────────────┐
              │ run_bidirectional_test(coordinator, ...)                │
              │ (window_optimizer_integration_final.py:318)             │
              │                                                          │
              │  if coord.use_range_miner:                ◄── NEW gate  │
              │      return _build_test_result_from_miner(...)          │
              │  elif coord.use_persistent_workers:   (existing :346)   │
              │      return _build_test_result_from_pw(...)             │
              │  elif coord.use_zmq_sqlite:           (existing :392)   │
              │      return _build_test_result_from_pw(...)             │
              │  else:                                                   │
              │      # legacy chunked dispatch via                       │
              │      # coordinator.execute_distributed_analysis          │
              └─────────────────────────────────────────────────────────┘
                                            │
                                            ▼
                            ┌────────────────────────────────────────┐
                            │ NEW: miner/                            │
                            │   range_miner_coordinator.py           │
                            │   range_miner_worker.py                │
                            │   range_miner_protocol.py              │
                            │   range_miner_npz_writer.py  ◄── NEW   │
                            │     (EXPECTED_NPZ_KEYS contract wall;  │
                            │      see §12.1)                        │
                            └────────────────────────────────────────┘
```

---

## 4. Step 1 output contract (frozen)

### 4.1 Files written — 8 mandatory + 1 optional fallback metadata

| # | File | Source | Mode | Notes |
| - | --- | --- | --- | --- |
| 1 | `forward_survivors.json` | `window_optimizer_integration_final.py:1559-1571` | always summary-only (S166) | `{"survivor_count": N, "note": "..."}` shape |
| 2 | `reverse_survivors.json` | `window_optimizer_integration_final.py:1573-1585` | always summary-only (S166) | same shape |
| 3 | `bidirectional_survivors.json` | `window_optimizer_integration_final.py:1587-1600` | full ≤ `_JSON_WRITE_LIMIT` (100 000), summary otherwise | listed in `agent_manifests/scorer_meta.json:13` |
| 4 | `bidirectional_survivors_all.npz` | `window_optimizer_integration_final.py:1783` | always — multi-run accumulator | Merge policy: highest score per seed (TB ruling S145-R1) |
| 5 | `bidirectional_survivors_binary.npz` | `window_optimizer_integration_final.py:1786` | always — canonical per-run | **22 arrays, identical schema to accumulator** |
| 6 | `optimal_window_config.json` | `window_optimizer.py:692-749` | Bayesian-only | Exact written keys per §4.4.3 |
| 7 | `train_history.json` | `window_optimizer.py:798-799` and `:991-992` | always | 80 % split |
| 8 | `holdout_history.json` | `window_optimizer.py:800-801` and `:994-995` | always | 20 % split |

**Optional file 9 (fallback only)**: `bidirectional_survivors_binary.meta.json` — written **only** when the primary NPZ accumulator (`window_optimizer_integration_final.py:1782-1793`) raises an exception and falls through to the subprocess path at `:1810-1814`. Miner must replicate both paths.

**Per-trial intermediates** (not part of final contract): `results/window_opt_{forward|reverse}[_hybrid]_{W}_{O}_t{T}.json`.

### 4.2 NPZ schema gate — 22 arrays

Enforced upstream of the normalizer, at the NPZ writer:

* `convert_survivors_to_binary.py:152-176`
* `window_optimizer_integration_final.py:1783, 1786`

Both emit the same 22 arrays:

```
seeds                    uint32
forward_matches          float32
reverse_matches          float32
window_size              int32
offset                   int32
trial_number             int32
skip_min                 int32
skip_max                 int32
skip_range               int32
forward_count            float32
reverse_count            float32
bidirectional_count      float32
intersection_count       float32
intersection_ratio       float32
intersection_weight      float32
bidirectional_selectivity float32
forward_only_count       float32
reverse_only_count       float32
survivor_overlap_ratio   float32
score                    float32
skip_mode                uint8
prng_type                uint8
```

### 4.3 Dedup contract — highest score per seed

Confirmed at:

* `window_optimizer.py:935-942` (`run_with_config`)
* `window_optimizer_integration_final.py:1697-1717` (vectorised — strict `>` at line 1706)

Greater-than (not ≥) is intentional: equal-score collisions keep the prior record (stable across runs).

---

### 4.4 Downstream consumption proof (REWRITTEN in v1.4.2 to absorb TB §3, §6)

This section corrects v1.4.1's tiered classification. Per TB audit, **all 22 NPZ arrays are functionally consumed** because the Step 3 driver iterates every array into per-survivor chunk metadata.

#### 4.4.1 Consumer map — every Step-1 file × every downstream reader

| Step-1 file | Consumer (script:line) | Pipeline step | Read mode | Fields/keys accessed |
| --- | --- | --- | --- | --- |
| `bidirectional_survivors_binary.npz` | `scorer_trial_worker.py:151-246` | Step 2 (scorer trial) | `survivor_loader.load_survivors(return_format='array')` | **hard**: `seeds` (line 179, ValueError if missing at 173-177), `forward_matches`+`reverse_matches` (183-196, RuntimeError if missing at 192). **soft**: `bidirectional_count` (201-211), `intersection_ratio` (213-222), `trial_number` (224-233), `skip_mode` (237-246) — all have explicit ones/zeros fallback |
| `bidirectional_survivors_binary.npz` | `generate_step3_scoring_jobs.py:62-102` (**active** Step 3 driver, invoked from `run_step3_full_scoring.sh:160`) | Step 3 (chunk generator) | `survivor_loader.load_survivors(return_format='array')` then `extract_survivors_full` iterates **every NPZ key** into per-survivor dict (line 83-93) | **ALL 22 arrays** become per-seed dict fields in chunk JSON. Hard guardrail at line 95-100 raises `ValueError("METADATA LOSS DETECTED")` if fewer than 3 fields preserved; error message specifies "Expected 20+" |
| `bidirectional_survivors_binary.npz` (via chunk JSON) | `full_scoring_worker.py:321, 411, 451-452, 472, 582-584` | Step 3 (worker) | Reads chunk JSON written by `generate_step3_scoring_jobs.py`; line 582-584 builds `survivor_metadata = {s['seed']: s for s in survivors_full}`; line 411/451/472 look up per-seed metadata during feature extraction | **All per-seed dict fields** flow into feature extraction. Comment at line 591-592: "Removed loading of forward/reverse survivor files — the metadata is already in the chunk file via survivor_metadata" |
| `bidirectional_survivors.json` | `agent_manifests/scorer_meta.json:13` lists in `inputs` (not `required_inputs`) | Step 2 (manifest declaration) | not directly read by `scorer_trial_worker.py` | — |
| `optimal_window_config.json` | `adaptive_meta_optimizer.py:226-227, 244, 252, 263` | Step 4 (ml_meta) | `json.load` | reads `best_result.bidirectional_count`, `all_results[].bidirectional_count`, `best_result.precision` — **none written by current Step 1** (see §4.4.3); `.get()` fallback at line 242 returns hardcoded defaults `{'min': 100, 'optimal': 500, 'max': 2000}` |
| `optimal_window_config.json` | `prediction_generator.py:152-158` | Step 6 (prediction) | `json.load` | `prng_type` (156), `mod` (157) — `mod` not written by current Step 1, fallback to default `1000` at line 165 |
| `train_history.json` | `scorer_trial_worker.py:151`, `full_scoring_worker.py:580` (via `load_lottery_history` at 201-225), `adaptive_meta_optimizer.py:321-322` | Steps 2, 3, 4 | `json.load` | flexible: flat `[int]` or `[{draw:int}]` per `full_scoring_worker.py:213-218` shape rule |
| `holdout_history.json` | `scorer_trial_worker.py:151` arg 3, `full_scoring_worker.py` | Steps 2, 3 | same | same shape |
| `forward_survivors.json` | `prediction.json` manifest line 4 (`required_inputs`); consumed via `prediction_generator.py:868` `forward_survivors` parameter | Step 6 | manifest declaration; runtime tolerates summary stub | parameter is a list, not a file read at the call site |
| `reverse_survivors.json` | `survivor_scorer.py:141, 150` accepts `reverse_survivors` parameter | indirect | list of ints | — |

#### 4.4.2 NPZ field consumption — REVISED CLASSIFICATION (TB §3, §6)

v1.4.1 split the 22 arrays into 3 hard / 4 soft / 15 unread. **TB audit overrules the "unread" tier.** Revised classification:

**Active Step 2 signals (7 fields).** Used at scoring time, with severity:

| Field | Step 2 severity | Site |
| --- | --- | --- |
| `seeds` | Hard — ValueError on absence | `scorer_trial_worker.py:173-177` |
| `forward_matches` | Hard — RuntimeError on absence | `:183-196` |
| `reverse_matches` | Hard — RuntimeError on absence | `:183-196` |
| `bidirectional_count` | Soft — fallback `np.ones(N)` | `:201-211` |
| `intersection_ratio` | Soft — fallback `np.zeros(N)` | `:213-222` |
| `trial_number` | Soft — fallback `np.zeros(N)` | `:224-233` |
| `skip_mode` | Soft — fallback `np.zeros(N)` | `:237-246` |

**Step 3 metadata-surface fields (15 fields).** Carried into chunk JSON by `generate_step3_scoring_jobs.py:62-102` and made available per-seed inside `full_scoring_worker.survivor_metadata` (`:321, 411, 451, 472`):

```
forward_count, reverse_count, intersection_count, intersection_weight,
bidirectional_selectivity, forward_only_count, reverse_only_count,
survivor_overlap_ratio, score, window_size, offset, skip_min, skip_max,
skip_range, prng_type
```

Whether `full_scoring_worker.py` *currently* reads each of these for feature extraction is implementation-dependent — but the **contract** is that the chunk JSON carries them, and the feature extractor is permitted to read any of them at any time without Step 1 modification. Per TB §3: *all 22 are consumer surface*. Miner must emit all 22 correctly.

**There is no "unread" tier in v1.4.2.** v1.4.1's claim was based on grep'ing for `survivors['FIELD']` in production code, which missed the chunk-JSON metadata pipeline. v1.4.2 corrects.

#### 4.4.3 `optimal_window_config.json` exact written shape

`window_optimizer.py:692-749` writes these top-level keys (verbatim):

```
window_size, offset, skip_min, skip_max, sessions,        ← line 692-697
prng_type, test_both_modes, seed_count,                   ← 698-700
optimization_score,                                       ← 701
forward_count, reverse_count, bidirectional_count,        ← 703-705
                                                          (merge block 710-725:)
[ status, completed_trials, total_trials,
  best_trial_number, best_value, best_bidirectional_count,
  last_updated, last_trial_number, last_trial_value,
  completed_at ]                                          ← only if existing incremental file merged
agent_metadata,                                           ← injected at 730
run_id                                                    ← line 746
```

**Consumer expectations vs. reality (preserve, do not fix in S172 — TB §5, Q4):**

* **Step 4** reads `best_result.bidirectional_count`, `best_result.precision`, `all_results[].bidirectional_count`. None written. Step 4 silently uses defaults `{'min':100, 'optimal':500, 'max':2000}` (`adaptive_meta_optimizer.py:242`) and `{'speed':0.5, 'stability':0.5}` (`:269, 272`).
* **Step 6** reads `prng_type` (written) and `mod` (not written; falls back to `1000` at `prediction_generator.py:165`).

**TB Q4 ruling**: do NOT add `best_result`, `all_results`, or `mod` to `optimal_window_config.json` inside S172. That is a separate compatibility-changing patch. Miner inherits the silent-fallback behavior unchanged.

#### 4.4.4 Step 5 is fully insulated from Step 1

`meta_prediction_optimizer_anti_overfit.py:610-680` reads `survivors_with_scores.json` (Step 3 output). `adaptive_meta_optimizer.py:9, 32, 99` explicitly: *"Intentionally does NOT consume survivors_with_scores.json"*. Step 1 → Step 5 is fully mediated by Step 3. Miner has zero compatibility risk against Step 5, provided Steps 2 + 3 outputs are preserved.

#### 4.4.5 The single-channel argument (revised — TB §6)

Across all six downstream steps, the data-carrying channels from Step 1 are:

| Channel | Carrier | Bytes flowing | Direction |
| --- | --- | --- | --- |
| Channel A | `bidirectional_survivors_binary.npz` | 22 arrays × N seeds | Step 1 → Step 2 → Step 3 |
| Channel A′ | `bidirectional_survivors_binary.npz` → chunk JSON | **every** array decoded per-seed into chunk dicts (`generate_step3_scoring_jobs.py:62-102`) | Step 1 → Step 3 metadata surface |
| Channel B | `optimal_window_config.json` | flattened key set per §4.4.3 | Step 1 → Step 2, Step 4, Step 6 |
| Channel C | `train_history.json` / `holdout_history.json` | lottery draws | Step 1 → Step 2, Step 3, Step 4 |
| Channel D | `forward_survivors.json` | summary stub | Step 1 → Step 6 (via parameter passing) |
| Channel E | `bidirectional_survivors.json` | summary or full | Step 1 → Step 2 manifest declaration only |

**Compatibility proof.** Byte-identity on Channels A, B, C, D, E ⇒ identical Step 2 input ⇒ identical Step 2 output (Optuna-seeded determinism) ⇒ identical Channel A′ chunk JSON ⇒ identical Step 3 output ⇒ identical downstream pipeline. **QED.**

#### 4.4.6 What this proves vs. what it does not

**Proven.** If miner emits §4.1 files with §4.2–§4.3 schemas, Steps 2–6 cannot distinguish miner-produced data from PWC-produced data.

**Not proven (validated by §11).** Miner *correctness* — the miner producing the right bytes for a given seed range. §11.A + §11.B + §11.C + §11.E together pin this down.

**Not safe to defer.** The 15 "Step 3 metadata-surface" fields and the PRNG_TYPE_ENCODING round-trip are part of the live contract today. Miner must hit them all from day one. v1.4.1's "future-fragile, fix later" framing is **withdrawn**.

---

## 5. PRNG-agnostic design

### 5.1 Registry inventory

`KERNEL_REGISTRY` at `prng_registry.py:3729` enumerates **44 entries** = 11 base × 4 variants. All 11 base families have complete 4-variant coverage with consistent suffix convention: `{base}` / `{base}_reverse` / `{base}_hybrid` / `{base}_hybrid_reverse`.

### 5.2 `resolve_kernel_families()`

```python
def resolve_kernel_families(prng_type: str, test_both_modes: bool) -> List[str]:
    families = [prng_type, prng_type + "_reverse"]
    if test_both_modes:
        families += [prng_type + "_hybrid", prng_type + "_hybrid_reverse"]
    for f in families:
        if f not in KERNEL_REGISTRY:
            raise ValueError(f"PRNG '{prng_type}' incomplete: missing {f} in registry")
    return families
```

### 5.3 Kernel-arg layout — S146 hybrid invariant

`sieve_gpu_worker.py:208-306` has **7 hardcoded `family_name ==` branches** covering **6 distinct base families** (the 7th branch is a `java_lcg` variant):

| Branch | Line | Covers |
| --- | --- | --- |
| `family_name == 'xorshift32'` | 217 | base: xorshift32 |
| `family_name == 'pcg32'` | 221 | base: pcg32 |
| `family_name == 'lcg32'` | 223 | base: lcg32 |
| `family_name in ('java_lcg', 'java_lcg_reverse')` | 227 | base: java_lcg (+ its `_reverse` variant) |
| `family_name in ('java_lcg_hybrid', 'java_lcg_hybrid_reverse')` | 232 | **variants** of java_lcg (different kernel signature) |
| `family_name == 'minstd'` | 299 | base: minstd |
| `family_name == 'xorshift128'` | 302 | base: xorshift128 |

Hybrid kernels (line 232-280) have a completely different signature. Forward hybrid takes `seeds, residues, survivors, match_rates, skip_sequences, strategy_ids, survivor_count, n_seeds, k, strategy_max_misses, strategy_tolerances, n_strategies, threshold, a, c`. Reverse hybrid replaces trailing `a, c` with a single `offset`. Miner must replicate verbatim.

**Coverage gap.** Of the 11 registered base families in `prng_registry.py`, **5 have no dedicated kernel-args branch** in the current worker: `mt19937`, `xorshift64`, `xoshiro256pp`, `philox4x32`, `sfc64`. This is a pre-existing PWC-mode bug; the worker's default-arg fallthrough at line 304 would launch any of these with potentially-wrong arity.

Miner introduces a `kernel_args_builders` registry keyed by base-family name. The 6 covered base families get builders today; the 5 uncovered families **must raise `NotImplementedError`** at dispatch time. No silent fallthrough. Coverage for each uncovered family is a separate per-family session, prioritised by owner.

### 5.4 PRNG_TYPE_ENCODING — REQUIRED FIX (TB §4, Q1)

**v1.4.2 reverts v1.4.1's "cosmetic" downgrade.** TB audit established:

* Encode side: `convert_survivors_to_binary.py:31-38` and `window_optimizer_integration_final.py:1627-1634` use a 12-key dict. Unknown strings collapse to `0` via `.get(..., 0)`. **All `*_hybrid` variants silently collapse to `0` (java_lcg)** because no `*_hybrid` keys exist in the dict.
* Decode side: `utils/survivor_loader.py:42-49` defines `PRNG_TYPE_DECODING` (the reverse mapping, also 12 entries), and `_array_to_dict:356-357` decodes via `.get(int_val, 'java_lcg')` — same silent fallback.
* Round-trip: a survivor written by the variable-skip kernel with `prng_type='java_lcg_hybrid'` arrives at Step 3 chunk metadata tagged as `prng_type='java_lcg'`. The hybrid provenance is **destroyed** between Step 1 and Step 3.
* Downstream impact: `generate_step3_scoring_jobs.py:81-93` propagates the (corrupted) string into every chunk dict. `full_scoring_worker.survivor_metadata` (`:321, 411, 451, 472`) makes the field available to any feature extractor — and even where current code doesn't read it, the **contract** says it's part of per-seed metadata.

**Required behavior in v1.4.2** (TB Q1):

1. **Hard-fail on unknown `prng_type`** — both encode and decode sites must raise `ValueError`, not fall back to `0`/`'java_lcg'`.
2. **Stable encoding for all `KERNEL_REGISTRY` keys** — every key in the 44-entry registry receives a deterministic uint8 ID. Legacy `0=java_lcg`, `1=java_lcg_reverse` are preserved for back-compat with existing NPZs on disk.
3. **`randu` / `randu_reverse` handling** — these strings exist in the encoding dict but are absent from `KERNEL_REGISTRY`. Treated as legacy decode-only aliases (decode emits `'randu'` → ID `10` for back-compat with older NPZs; encode rejects new `prng_type='randu'` because no kernel can be launched).
4. **Pre-S172 patch**. The encoding fix should ship as a **focused patch** (`convert_survivors_to_binary.py v3.2` + matching integration writer + matching `survivor_loader.py` decode dict) **before** the miner lands, so both PWC and miner share the corrected encoding. This guarantees: (a) §11.B array equality is meaningful, (b) no silent corruption of historical data when miner runs alongside PWC fallback.

**Encoding-fix patch is in S172's critical path** (not optional, not deferred).

---

## 6. CLI, modes, dedup, PRNG resolution, config precedence

### 6.1–6.2 CLI flags

(Unchanged from v1.4.1.) 27 existing flags. 3 new miner flags. Argparse-level mutex with `--use-persistent-workers` and `--use-zmq-sqlite`.

### 6.3–6.7 Architecture details

Sub-stripe partitioning (default 8 per stripe). **TB Q2 ruling: per-family / per-variant VRAM caps approved.** Hybrid kernels (with `skip_sequences_gpu = cp.zeros(n_seeds * k, dtype=cp.uint32)` at `sieve_gpu_worker.py:254`) use a tighter cap than constant kernels; default `seed_cap_amd` applies to constant phases only. Per-family caps configured via WATCHER manifest extension (see §7).

### 6.8 Test-both-modes — 4-phase workflow

`run_bidirectional_test` at `window_optimizer_integration_final.py:318-723` executes (with Q0 gate at 633-635 and idempotency guard at 610):

| Phase | Kernel (`prng_base = java_lcg`) | Line | Skip mode |
| - | --- | --- | --- |
| 1 | `java_lcg` | 477 | constant |
| 2 | `java_lcg_reverse` | 506 | constant |
| 3 | `java_lcg_hybrid` | 617 | variable |
| 4 | `java_lcg_hybrid_reverse` | 641 | variable |

### 6.9–6.11

Dedup (§4.3). PRNG resolution (§5). 6-level config precedence: CLI > JSON config > WATCHER `default_params` > WATCHER `parameter_bounds.default` > coordinator constructor defaults > argparse defaults.

---

## 7. WATCHER manifest

`agent_manifests/window_optimizer.json` v1.7.0 → v1.8.0. Miner adds to `actions[0].args_map`:

```
"use-range-miner":     "use_range_miner",
"miner-stripe-size":   "miner_stripe_size",
"miner-substripes":    "miner_substripes",
"seed-cap-amd-hybrid": "seed_cap_amd_hybrid",      ← TB Q2 (new in v1.4.2)
"seed-cap-nvidia-hybrid": "seed_cap_nvidia_hybrid" ← TB Q2 (new in v1.4.2)
```

And to `default_params`:

```
"use_range_miner":          true,   ── once TB approves
"miner_stripe_size":        67108864,
"miner_substripes":         8,
"seed_cap_amd_hybrid":      1000000,  ← TB Q2: hybrid VRAM is tighter
"seed_cap_nvidia_hybrid":   2500000   ← TB Q2: hybrid VRAM is tighter
```

`success_condition` unchanged.

**Mandatory dry-run test** (per TB §8) — replicates S137 failure mode (manifest args present, argparse missing):

```bash
PYTHONPATH=. python3 agents/watcher_agent.py \
    --run-pipeline --start-step 1 --end-step 1 --dry-run
```

Acceptance:
* `--use-range-miner` appears when manifest sets `use_range_miner: true`.
* `--use-persistent-workers` does NOT appear simultaneously.
* `--pwc-transport` is absent or ignored in miner mode.
* All new miner flags resolve from manifest → CLI without argparse failure.

---

## 8. Coexistence with PWC backends

Single-active-backend enforced at integration via cascading gates at `window_optimizer_integration_final.py:346, 392`. Miner adds:

```python
_use_miner = getattr(coordinator, 'use_range_miner', False)
if _use_miner:
    if run_trial_miner is None:
        raise ImportError("miner/range_miner_coordinator.py not found — cannot use --use-range-miner")
    _miner_result = run_trial_miner(...)
    return _build_test_result_from_miner(_miner_result, accumulator, ...)
```

Argparse-level mutex prevents two backends in the same process. OS-level: miner default port 5700 (PWC default 5600 at `:368`).

No `.service` files in repo (verified `find . -name "*.service"` = zero matches). Miner ships systemd unit `/etc/systemd/system/range-miner-worker@.service` (template per v1.4.1 §8.2) plus `miner/install_systemd.sh`.

---

## 9. Persistent/ files audit — April TB blockers re-checked

All four fixed in code (unchanged from v1.4.1):

| TB blocker | Status | Evidence |
| --- | --- | --- |
| `inspect.fields` bug | FIXED | `persistent/pwc_protocol.py:159-167` |
| Spool disabled | NOT TRUE — spool is live | `persistent/pwc_protocol.py:93-102`, `pwc_result_normalizer.py:33-34, 65-124` |
| Missing lease / reclaim | FIXED | `persistent/pwc_transport_tcp.py:143-148, 287-340, 342-358` |
| Missing heartbeat | FIXED | `persistent/pwc_transport_tcp.py:32, 148-153, 342-358` |

---

## 10. Effort estimate

| Phase | Sessions | Scope |
| --- | --- | --- |
| Phase 0 (NEW v1.4.2): PRNG_TYPE_ENCODING patch | 1 | Pre-S172 focused patch — both PWC and miner share corrected encoding before Phase 1 starts |
| Phase 1: scaffolding | 1 | `miner/` dir, argparse wiring, mutex, integration gate, WATCHER manifest update |
| Phase 2: protocol + transport | 1 | `miner/range_miner_protocol.py`, TCP framing reusing `persistent/pwc_transport_tcp.py` patterns |
| Phase 3: worker daemon | 1 | `miner/range_miner_worker.py` — READY handshake, sub-stripe loop, 7 hardcoded branch patterns covering 6 base families + `NotImplementedError` for the 5 uncovered base families |
| Phase 4: coordinator | 1 | `miner/range_miner_coordinator.py` — stripe assignment, per-family VRAM caps (TB Q2), fail-closed Phase 1/2, **one-retry-then-fail-trial Phase 3/4 (TB Q3)** |
| Phase 5: dedup + NPZ write-back with **EXPECTED_NPZ_KEYS contract wall** (§12.1) | 1 | Output contract layer with fail-hard validation |
| Phase 6: hold-out validation | 1 | Acceptance per §11.A–§11.E |
| Phase 7: WATCHER soak + production | 1 | 50-trial soak + §11.F–§11.G acceptance |
| **Total** | **8 sessions** | (was 7 in v1.4.1; +1 for Phase 0 encoding patch) |

---

## 11. Acceptance criteria (REWRITTEN per TB §10)

All criteria are block-on-failure. v1.4.1's "Tier 3 = warning" is withdrawn; **all 22 NPZ arrays are blockers** because the Step 3 metadata surface preserves every one.

### 11.A Producer file-presence + identity

Run PWC and Range-Miner on identical inputs (same lottery file, same `seed_start`, same `seed_count`, same window config, same thresholds, same `prng_type`, same `test_both_modes`). Then SHA-256 or content-compare:

* `forward_survivors.json`
* `reverse_survivors.json`
* `bidirectional_survivors.json`
* `bidirectional_survivors_all.npz`
* `bidirectional_survivors_binary.npz`
* `optimal_window_config.json` — ignore `run_id` and any timestamp fields only
* `train_history.json`
* `holdout_history.json`

### 11.B NPZ raw array identity (TB §10.B)

For `bidirectional_survivors_binary.npz`, for **all 22 arrays** (no warning tier):

```python
np.array_equal(pwc_npz[k], miner_npz[k])
```

Failure on any array is a **release blocker**.

### 11.C NPZ → dict identity (NEW per TB §6, §10.C)

```python
pwc_dict   = load_survivors("pwc/bidirectional_survivors_binary.npz",   return_format="dict").data
miner_dict = load_survivors("miner/bidirectional_survivors_binary.npz", return_format="dict").data
assert sorted(pwc_dict, key=lambda x: x['seed']) == sorted(miner_dict, key=lambda x: x['seed'])
```

This catches the round-trip failure mode that the raw-array test would miss only if encoded `prng_type` differed semantically (e.g. if the encoding fix exposes a bug in the round-trip).

### 11.D Step 2 output identity (TB §10.D)

Compare:

* `optimal_scorer_config.json` — JSON key-by-key equality.
* `scorer_trial_results/*.json` — per-trial result equality (Optuna trial sequence determinism must be held by seeding the study identically).

### 11.E Step 3 output identity (TB §10.E)

Compare:

* `scoring_chunks/chunk_*.json` — per-survivor dict equality (this is where metadata-loss bugs surface).
* `survivors_with_scores.json` — record-level equality, sort by seed.

This is the **stronger** test — it catches metadata corruption that array-identity (11.B) cannot.

### 11.F Step 4 output compatibility (TB §10.F)

Compare:

* `reinforcement_engine_config.json`

Step 4 silently falls back to defaults (see §4.4.3). The miner inherits this behavior; output should be identical because both backends provide the same (missing) `best_result`/`all_results` keys.

### 11.G Step 5/6 operational compatibility (TB §10.G)

Step 5 and Step 6 do not require byte-identical output (model training has non-deterministic GPU behavior). They must:

* Load all required files without schema errors.
* Validate sidecar feature schemas (`models/reinforcement/best_model.meta.json` per `prediction.json` manifest).
* Produce expected output files (`predictions/next_draw_prediction.json` for Step 6).
* Not branch on backend identity (no `if backend == 'miner'` anywhere in Step 5/6 code).

### 11.H Dedup correctness

Across 50 trials of stripe-overlapping seed ranges, no duplicate seeds in final `bidirectional_survivors_binary.npz`; highest-score-per-seed rule satisfied for all collisions.

### 11.I PRNG agnosticism

Miner runs successfully on at least 3 **explicitly supported base PRNGs** from the current worker arg-builder set — for example: `java_lcg`, `lcg32`, `minstd` — with `test_both_modes=True`, passing §11.A–§11.E against PWC counterparts.

**Uncovered registry families** — `mt19937`, `xorshift64`, `xoshiro256pp`, `philox4x32`, and `sfc64` — must fail fast with `NotImplementedError` until a dedicated kernel-args builder is implemented and tested. No silent fallthrough; no degraded execution path. The miner's `kernel_args_builders` registry (§5.3) is the enforcement point: missing-builder lookup raises at dispatch time, before any GPU work begins.

Acceptance test for the negative case: invoking the miner with `--prng-type mt19937` (or any of the other 4 uncovered families) must produce a clean `NotImplementedError` with a message naming the missing family, and must NOT launch any kernel.

### 11.J WATCHER manifest dry-run

Passes for `agent_manifests/window_optimizer.json:1.8.0` per §7.

### 11.K Pool=8 cliff regression

Full 50-trial run at default settings (3 rigs × 8 GPUs = 24 active GPUs) completes without any `launch_s174` FAULT_KEYWORD triggering.

### 11.L Persistent/ coexistence

PWC-TCP run after a miner run on the same Zeus succeeds without manual cleanup.

### 11.M No raw-string PRNG names

`grep -rE "'(java_lcg|mt19937|lcg32|xorshift|minstd|pcg32|sfc64|philox|xoshiro)'" miner/` returns zero matches outside test scaffolding.

---

## 12. Implementation rules

### 12.1 EXPECTED_NPZ_KEYS contract wall (NEW per TB §11)

Every miner-emitted NPZ must pass through a single validator before being written to disk:

```python
EXPECTED_NPZ_KEYS = [
    "seeds",                    # uint32
    "forward_matches",          # float32
    "reverse_matches",          # float32
    "window_size",              # int32
    "offset",                   # int32
    "trial_number",             # int32
    "skip_min",                 # int32
    "skip_max",                 # int32
    "skip_range",               # int32
    "forward_count",            # float32
    "reverse_count",            # float32
    "bidirectional_count",      # float32
    "intersection_count",       # float32
    "intersection_ratio",       # float32
    "intersection_weight",      # float32
    "bidirectional_selectivity",# float32
    "forward_only_count",       # float32
    "reverse_only_count",       # float32
    "survivor_overlap_ratio",   # float32
    "score",                    # float32
    "skip_mode",                # uint8
    "prng_type",                # uint8
]

EXPECTED_DTYPES = {
    "seeds": np.uint32,
    "skip_mode": np.uint8,
    "prng_type": np.uint8,
    **{k: np.int32 for k in ["window_size","offset","trial_number","skip_min","skip_max","skip_range"]},
    **{k: np.float32 for k in [
        "forward_matches","reverse_matches","forward_count","reverse_count",
        "bidirectional_count","intersection_count","intersection_ratio","intersection_weight",
        "bidirectional_selectivity","forward_only_count","reverse_only_count",
        "survivor_overlap_ratio","score",
    ]},
}

def validate_and_write_npz(path: str, arrays: dict) -> None:
    """Contract wall — TB ruling §11. Fail hard on any deviation."""
    # 1. Key set
    actual_keys = set(arrays.keys())
    expected_keys = set(EXPECTED_NPZ_KEYS)
    if actual_keys != expected_keys:
        missing = expected_keys - actual_keys
        extra   = actual_keys - expected_keys
        raise ValueError(
            f"NPZ schema violation: missing keys {missing!r}, extra keys {extra!r}"
        )
    # 2. Dtypes
    for k in EXPECTED_NPZ_KEYS:
        if arrays[k].dtype != EXPECTED_DTYPES[k]:
            raise TypeError(
                f"NPZ dtype violation: {k} is {arrays[k].dtype}, expected {EXPECTED_DTYPES[k]}"
            )
    # 3. Length equality
    n = len(arrays["seeds"])
    for k in EXPECTED_NPZ_KEYS:
        if len(arrays[k]) != n:
            raise ValueError(
                f"NPZ length violation: {k} has len={len(arrays[k])}, expected {n}"
            )
    # 4. Seeds sorted ascending (dedup post-condition)
    if not np.all(np.diff(arrays["seeds"]) > 0):
        raise ValueError("NPZ ordering violation: seeds not strictly ascending after dedup")
    # 5. prng_type and skip_mode in known encoding range
    valid_prng_ids = set(PRNG_TYPE_ENCODING.values())  # after §5.4 fix, this is full registry coverage
    if not set(arrays["prng_type"].tolist()).issubset(valid_prng_ids):
        unknown = set(arrays["prng_type"].tolist()) - valid_prng_ids
        raise ValueError(f"NPZ prng_type violation: unknown encoded IDs {unknown!r}")
    if not set(arrays["skip_mode"].tolist()).issubset({0, 1}):
        unknown = set(arrays["skip_mode"].tolist()) - {0, 1}
        raise ValueError(f"NPZ skip_mode violation: unknown encoded IDs {unknown!r}")
    # 6. Write
    np.savez_compressed(path, **arrays)
```

This validator runs on every miner-produced NPZ, on every flush. Failure aborts the run rather than writing corrupted data.

### 12.2 Dedup post-condition (NEW)

After dedup, NPZ `seeds` array must be strictly ascending. Enforced by validator step 4. This guarantees `np.array_equal` ordering for §11.B.

### 12.3 Reassign + degraded policy (TB Q3 ruling)

For Phase 3 (hybrid forward) and Phase 4 (hybrid reverse):

* First sub-stripe failure → reassign to another GPU once, mark `phase_degraded=True` in `agent_metadata`.
* Second failure → **fail the whole trial**.
* No reduced-sample variable-skip objective. No partial dataset.

Optuna comparability is preserved.

### 12.4 Per-family VRAM caps (TB Q2 ruling)

Hybrid kernels require additional `skip_sequences_gpu = cp.zeros(n_seeds * k, dtype=cp.uint32)` (`sieve_gpu_worker.py:254`). With k=12, 2 M-seed sub-stripe ⇒ 96 MB extra. On RX 6600 (8 GB VRAM, 4 GB miner reserve), constant-phase cap = 2 M (existing `seed_cap_amd`), hybrid-phase cap = 1 M (new `seed_cap_amd_hybrid`). Same ratio applied to NVIDIA (5 M / 2.5 M). New WATCHER manifest entries in §7.

---

## 13. TB rulings on outstanding questions

All four resolved by TB v1.4.1 audit:

| Q | Question | TB Ruling | v1.4.2 absorbs in |
| --- | --- | --- | --- |
| Q1 | PRNG_TYPE_ENCODING fix vs. defer | **Fix it.** Pre-S172 patch, share encoding between PWC and miner. Unknown `prng_type` is hard error, not fallback. | §5.4, Phase 0 of §10, §12.1 validator step 5 |
| Q2 | Per-family vs. global VRAM caps | **Per-family approved.** Hybrid cap tighter than constant cap. | §6.3–§6.7, §7 manifest, §12.4 |
| Q3 | Reassign + degraded semantics | **One retry, then fail trial.** No partial dataset. | §6.7, §12.3 |
| Q4 | Step 1 → Step 4 schema drift | **Preserve drift.** Do not add `best_result`/`all_results`/`mod` in S172. | §4.4.3, §11.F |

---

## 14. Audit trail for TB (commands updated for v1.4.4)

| Claim | Verify by running |
| --- | --- |
| 22-array NPZ contract | `python3 -c "import ast, convert_survivors_to_binary as c; t=ast.parse(open(c.__file__).read()); [print(f'{len(n.keywords)} kwargs:', [k.arg for k in n.keywords]) for n in ast.walk(t) if isinstance(n, ast.Call) and getattr(n.func,'attr','')=='savez_compressed']"` — prints `22 kwargs: ['seeds', 'forward_matches', ...]` |
| 44-entry kernel registry | `python3 -c "from prng_registry import KERNEL_REGISTRY; print(len(KERNEL_REGISTRY))"` — prints `44` |
| 11 base families | `python3 -c "from prng_registry import KERNEL_REGISTRY; keys=set(KERNEL_REGISTRY.keys()); bases=sorted(k for k in keys if not k.endswith(('_reverse','_hybrid','_hybrid_reverse'))); print(len(bases), bases)"` — prints `11 [...]` |
| **Active Step 3 driver is `generate_step3_scoring_jobs.py`** (NEW v1.4.2) | `grep -n "generate_step3_scoring_jobs\|generate_full_scoring_jobs" run_step3_full_scoring.sh` — single match at line 160, invokes `generate_step3_scoring_jobs.py` only |
| **Step 3 preserves all 22 NPZ arrays into chunk metadata** (NEW v1.4.2) | `sed -n '62,102p' generate_step3_scoring_jobs.py` — function `extract_survivors_full` iterates every key in NPZ dict (line 83-93); `METADATA LOSS DETECTED` guardrail at 96-100 |
| **PRNG_TYPE_DECODING in survivor_loader** (NEW v1.4.2) | `sed -n '42,49p' utils/survivor_loader.py` — 12-entry decode dict |
| **survivor_loader silently falls back on unknown encoded ID** (NEW v1.4.2) | `sed -n '356,358p' utils/survivor_loader.py` — `PRNG_TYPE_DECODING.get(int(data['prng_type'][i]), 'java_lcg')` |
| **full_scoring_worker uses survivor_metadata** (NEW v1.4.2) | `grep -n "survivor_metadata" full_scoring_worker.py` — 6+ matches at lines 321, 379-380, 411, 451-452, 472, 582-584 |
| 3 hard-required NPZ fields | `grep -n "if 'forward_matches' in survivors\|'seeds' not in survivors" scorer_trial_worker.py` — lines 173 (ValueError) and 183 (RuntimeError block) |
| 4 soft-required NPZ fields | `grep -n "if 'bidirectional_count' in survivors\|if 'intersection_ratio' in survivors\|if 'trial_number' in survivors\|if 'skip_mode' in survivors" scorer_trial_worker.py` — exactly 4 lines (201, 213, 224, 237) |
| `best_result`/`all_results` absent from writer | `sed -n '692,706p' window_optimizer.py` |
| Step 4 reads with fallback | `sed -n '241,265p' adaptive_meta_optimizer.py` |
| 4 April TB blockers fixed | See §9 |

---

**End of v1.4.4. SPEC FROZEN. Implementation begins with Phase 0 (PRNG_TYPE_ENCODING v3.2 shared patch).**
