# Chapter 8: PRNG Registry

> **What TFM is.** TFM filters candidate generators against observed draws in two directions,
> keeps a population of survivors on purpose, ranks them, and lets the top candidates vote their
> next value into a prediction pool. A survivor is a scored candidate, not a verdict. The
> generator family is a substrate for mimicry, not a claim about the machine. The target is a
> pool hit rate. TFM does not recover a seed or reconstruct state, and an objection that assumes
> it does is out of scope.

## PRNG Analysis Pipeline — Complete Operating Guide

**Version:** 2.4 (this is the version string of `prng_registry.py`'s own docstring, `prng_registry.py:9`; the module's `__main__` banner still prints "v2.1", `:4199`)  
**File:** `prng_registry.py`  
**Lines:** 4,323 — *Corrected 2026-11-11: previously said 4,323, which is the length of the file at the initial commit `0101306` (`git show 0101306:prng_registry.py | wc -l` → 4322). At `a15b524` the file is **4,235** lines (`wc -l`).*  
**Purpose:** Complete PRNG kernel library with GPU sieve implementations

**Chapter status (2026-11-11):** **CORRECTED against code at `a15b524`** (`a15b5247336255801de8cf67d52653701b1e8c80`).
**Authority for this pass:** `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md` (REV1), §0, §1, §3, §5; batch-3 common brief.
**Audit:** `docs/CHAPTER_8_AUDIT.md` (claim-by-claim labels and the count reconciliation).
**Code base:** `a15b524`. Current HEAD is `b6cb789`, which changed only `agents/watcher_agent.py`, `agents/watcher_dispatch.py` and `tests/test_watcher_safety_repairs.py` (`git diff --name-only a15b524 HEAD`). This chapter cites none of those three files. Every code file it cites (`prng_registry.py`, `utils/prng_encoding.py`, `sieve_gpu_worker.py`, `sieve_filter.py`, `miner/range_miner_worker.py`, `hybrid_strategy.py`, `survivor_scorer.py`, `full_scoring_worker.py`, `scorer_trial_worker.py`, `tests/test_prng_encoding.py`) is identical at `a15b524` and in the working tree (`git diff --quiet a15b524 -- <file>`, each checked), so the line anchors below hold for both.
**Numbering scheme:** **neither.** This is a component chapter, not a pipeline step. It does not use the conceptual step numbers or the executable `STEP_SCRIPTS` numbers (`docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1). Where a step is named, it is named in the **executable** scheme and labelled so (the sieve runs inside executable Step 1; executable Step 2 is the scorer meta-optimizer; executable Step 3 is full scoring).
**Closed chapter this one defers to:** `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` (CLOSED). It owns the three-lane count (§6.2.1, re-derived by `tests/test_chapter2_content_gate.py` G-LANE-COUNT), the hybrid skip-bound defect (§5.4), the `offset` disposition (§7) and the hostname-guard finding F-5 (§9.5). This chapter cross-references those and does not restate or re-count them.

### How this chapter was produced

1. The chapter was read in full at `a15b524` (1,058 lines; the working-tree copy was byte-identical before this pass).
2. The governance trail was searched first: `docs/TB_RULING_*`, `docs/PROPOSAL_*` and `docs/TEAM_ALPHA_*` (140 files) for `prng_registry`, `KERNEL_REGISTRY`, family counts and "uncovered"; then `docs/BACKLOG.md`, `docs/TODO_CONSOLIDATED.md`, `docs/PIPELINE_BEHAVIOUR_MODEL.md`, `docs/TFM_PROJECT_FACTS_SKILL.md` §2.6–2.7; then the closed Chapter 2 and the committed audits of Chapters 3, 4, 5, 6, 9, 12, 13, 14.
3. The code was then read: `prng_registry.py` (docstring, every CPU reference, the java_lcg / xorshift32 / xorshift128 / xorshift64 kernels, the java_lcg forward-hybrid and reverse kernels, the registry dict and every helper), a parse of all 44 kernel signatures, `utils/prng_encoding.py`, `sieve_gpu_worker.py:180-306`, `miner/range_miner_worker.py:105-200`, `:259-300`, `:464-485`, `:628-640`, `:1098-1109`, `:1195-1205`, `sieve_filter.py:230-233`, `:393-396`, `hybrid_strategy.py`, and the registry's consumers in `survivor_scorer.py` and `full_scoring_worker.py`.
4. The family count was **re-derived live**, not taken from any document: `python3 -I -c` importing `prng_registry` (read-only introspection under `~/venvs/torch`; no kernel compiled, no GPU touched) lists **44** keys over **11** base families, 4 variants each.
5. Each claim was labelled (Accurate / BROKEN / SWITCHED-OFF-BY-RULING / DESCRIPTION-ONLY / UNVERIFIED / absorbed into an item another chapter already holds). The chapter was then corrected **in place with the Edit tool**: wrong statements carry a visible "Corrected 2026-11-11" note; no section was deleted; every original heading survives.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Registry Structure](#2-registry-structure)
3. [PRNG Categories](#3-prng-categories)
4. [CPU Reference Implementations](#4-cpu-reference-implementations)
5. [GPU Kernel Architecture](#5-gpu-kernel-architecture)
6. [Kernel Memory Layout](#6-kernel-memory-layout)
7. [Standard Sieve Kernels](#7-standard-sieve-kernels)
8. [Hybrid Sieve Kernels](#8-hybrid-sieve-kernels)
9. [Reverse Sieve Kernels](#9-reverse-sieve-kernels)
10. [PyTorch GPU Implementations](#10-pytorch-gpu-implementations)
11. [Adding New PRNGs](#11-adding-new-prngs)
12. [Helper Functions](#12-helper-functions)
13. [Complete PRNG Reference](#13-complete-prng-reference)
14. [Chapter Summary](#14-chapter-summary)
15. [The shared registry-derived encoding (S172 Phase 0)](#15-the-shared-registry-derived-encoding-s172-phase-0) — *added 2026-11-11*
16. [Which registry entries actually run: PWC branches and RANGE-MINER coverage](#16-which-registry-entries-actually-run-pwc-branches-and-range-miner-coverage) — *added 2026-11-11*
17. [Known gaps at a15b524](#17-known-gaps-at-a15b524) — *added 2026-11-11*
18. [Open items](#18-open-items) — *added 2026-11-11*
19. [Verification declaration (VIR-1…6)](#19-verification-declaration-vir-16) — *added 2026-11-11*
20. [Closure statement](#20-closure-statement) — *added 2026-11-11*

---

## 1. Overview

### 1.1 What the PRNG Registry Does

The PRNG Registry is the **kernel library** for the entire system. It contains:

- **46 PRNG implementations** across 14 families
- **GPU kernels** for forward and reverse sieves (CuPy/CUDA/ROCm)
- **CPU reference implementations** for verification
- **PyTorch GPU implementations** for ML scoring (v2.4)

> **Corrected 2026-11-11.** Previously said "46 PRNG implementations across 14 families". The
> registry at `a15b524` holds **44 entries over 11 base families**, each in exactly four variants
> (`{base}`, `{base}_hybrid`, `{base}_reverse`, `{base}_hybrid_reverse`): `KERNEL_REGISTRY`,
> `prng_registry.py:3729-4132`; live `len(KERNEL_REGISTRY)` = 44. The 11 bases are `java_lcg`,
> `lcg32`, `minstd`, `mt19937`, `pcg32`, `philox4x32`, `sfc64`, `xorshift128`, `xorshift32`,
> `xorshift64`, `xoshiro256pp`. The 44 is load-bearing: the Phase-0 encoding pins it
> (§15). This divergence was already recorded as `docs/PIPELINE_BEHAVIOUR_MODEL.md` D18 and in
> `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md` §5.1 (`:86`, `:339`); it is corrected here, not newly found.
>
> **Being in the registry is not the same as being run.** At `a15b524` the certifying engine
> (RANGE-MINER) builds kernel arguments for 6 of the 11 bases and raises `NotImplementedError`
> for the other 5. TFM sieves `java_lcg` only. See §16.
>
> A family in this registry is a **substrate for mimicry**: a generator whose surface output the
> sieve scores against observed draws. No family here is "the" generator behind the draws, and
> nothing in this chapter claims one is.

### 1.2 Version History

```
Version 2.4 - November 27, 2025
- ENHANCEMENT: PyTorch GPU implementations for Step 2.5 ML scoring
  * Works on both CUDA (NVIDIA) and ROCm (AMD)
  * Phase 1: java_lcg, java_lcg_hybrid implemented

Version 2.3 - October 29, 2025
- CRITICAL FIX: Replaced hardcoded 512 buffer sizes with dynamic sizing
  * Changed 71 instances across all hybrid kernels
  * Arrays now: best_skip_seq[2048], current_skip_seq[2048]
  * Window sizes up to 2048 draws now supported

- CRITICAL FIX: Changed skip_sequences stride from hardcoded to dynamic
  * Changed: skip_sequences[pos * k + i] (was pos * 512 + i)
  * Eliminated illegal memory access causing GPU crashes
```

> **Corrected 2026-11-11 (history).** The current state is as described: every hybrid kernel at
> `a15b524` uses 2048-element local skip arrays and stores with stride `pos * k`
> (e.g. `java_lcg_hybrid_multi_strategy_sieve`, `prng_registry.py:1027-1030`, `:1077`). **The v2.3
> fix was not complete when v2.3 shipped.** Ten `[512]` local arrays survived in nine
> **reverse-hybrid** kernel sources — two in `SFC64_HYBRID_REVERSE_KERNEL` (whose entry point is
> named `sfc64_hybrid_multi_strategy_sieve`, `:1867`, `:1881`, `:1898`) and one each in the
> lcg32, xorshift64, xorshift128, pcg32, java_lcg, minstd, philox4x32 and mt19937 reverse
> hybrids — until commit `7313a43` (2026-05-03, tag `[S170-K512] was [512]`;
> `git show 7313a43 -- prng_registry.py` removes 10 `[512]` lines; the 10 `S170-K512` markers are at
> `:1881`, `:1898`, `:2474`, `:2780`, `:2923`, `:3063`, `:3196`, `:3330`, `:3489`, `:3678`). The same commit changed the docstring line to read "(was pos \* k + i)"
> (`prng_registry.py:32`), which no longer says what the stride was before. The original `512`
> stride is in the ancestor files (`git log -S 'pos * 512' -- prng_registry.py` → `0101306`,
> `7313a43`). The claim that the fix "eliminated illegal memory access" is a runtime claim that
> this documentation pass did not re-test: **[UNVERIFIED]** (audit U1).

### 1.3 Key Statistics

| Metric | Count |
|--------|-------|
| Total PRNGs | 46 |
| Fixed Skip | 14 |
| Hybrid (Variable Skip) | 14 |
| Reverse Sieve | 18 |
| Max Window Size | 2048 draws |
| Seed Types | uint32, uint64 |

> **Corrected 2026-11-11.** Previously: 46 total = 14 fixed + 14 hybrid + 18 reverse. Live at
> `a15b524` (`KERNEL_REGISTRY`, `prng_registry.py:3729-4132`):
>
> | Metric | Count |
> |--------|-------|
> | Total registry entries | **44** |
> | Base families | **11** |
> | Forward, fixed skip (`{base}`) | **11** |
> | Forward, hybrid (`{base}_hybrid`) | **11** |
> | Reverse, fixed skip (`{base}_reverse`) | **11** |
> | Reverse, hybrid (`{base}_hybrid_reverse`) | **11** |
> | Max window size | 2048 draws — the hybrid kernels' local-array bound (`draw_idx < 2048`, `:1030`); the constant kernels keep no per-draw array |
> | Seed types | uint32, uint64 (unchanged) |
>
> The old "18 reverse" counted the reverse entries of 9 families and missed `xoshiro256pp` and
> `sfc64`, which also have both reverse variants (`:4094-4131`). There are 22 reverse-direction entries.

## 2. Registry Structure

### 2.1 KERNEL_REGISTRY Dictionary

```python
KERNEL_REGISTRY = {
    'prng_name': {
        'kernel_source': KERNEL_SOURCE_STRING,  # CUDA/HIP kernel code
        'kernel_name': 'kernel_function_name',  # Entry point name
        'cpu_reference': cpu_function,          # Python CPU implementation
        'pytorch_gpu': pytorch_function,        # PyTorch GPU (if available)
        'default_params': {                     # PRNG-specific parameters
            'a': 25214903917,
            'c': 11,
        },
        'description': 'Human-readable description',
        'seed_type': 'uint32' | 'uint64',       # Seed data type
        'state_size': 4,                        # State size in bytes
        'variable_skip': True | False,          # Supports variable skip?
        'multi_strategy': True | False,         # Multi-strategy hybrid?
    },
    ...
}
```

> **Corrected 2026-11-11 (which keys each entry carries).** The ten keys above are exactly the union
> of keys across the live registry. **Not every entry carries all of them.** The 18 reverse entries of
> nine families (all but `xoshiro256pp` and `sfc64`) carry only `kernel_source`, `kernel_name`,
> `description`, sometimes `seed_type`, and `variable_skip`/`multi_strategy` on the hybrids.
> They have **no `cpu_reference` and no `default_params`** (e.g. `mt19937_reverse`, `:3850-3854`;
> `java_lcg_reverse`, `:3908-3913`). Live key-absence counts over the 44 entries: `cpu_reference` 18,
> `default_params` 18, `seed_type` 12 (all reverse entries of uint32 families), `state_size` 19
> (the 18 plus `java_lcg`, `:3957-3968`). `pytorch_gpu` is on 2 entries only (`java_lcg`, `java_lcg_hybrid`; the key is
> written twice in each literal, `:3961-3962`, `:3974-3975`, harmlessly). Callers that need
> `seed_type` read it with a default of `uint32` (`sieve_gpu_worker.py:180`). For every reverse entry
> without the key, that default equals the base family's seed type. The registry does not say whether
> reverse entries were meant to carry their own `cpu_reference`; see §12.3.

### 2.2 Naming Convention

| Pattern | Meaning | Example |
|---------|---------|---------|
| `{prng}` | Standard forward sieve | `java_lcg` |
| `{prng}_hybrid` | Variable skip forward | `java_lcg_hybrid` |
| `{prng}_reverse` | Standard reverse sieve | `java_lcg_reverse` |
| `{prng}_hybrid_reverse` | Variable skip reverse | `java_lcg_hybrid_reverse` |

---

## 3. PRNG Categories

### 3.1 Fixed Skip PRNGs (14 total)

> **Corrected 2026-11-11:** **11** forward fixed-skip entries, not 14. The table below always listed
> 11. One row is also corrected: `xorshift64` is **xorshift64\*** (shift 12/25/27, then multiply by
> `0x2545F4914F6CDD1D`). It has **no Weyl sequence** (`xorshift64_cpu`, `prng_registry.py:217-232`;
> kernel `xorshift64_flexible_sieve`, `:679-700`). Registry description: "Xorshift64 with correct
> skip/gap logic" (`:3835`).

Standard sieves with constant skip between outputs.

| PRNG | Seed Type | State Size | Description |
|------|-----------|------------|-------------|
| `xorshift32` | uint32 | 4 bytes | Xorshift32 with configurable shifts |
| `xorshift64` | uint64 | 8 bytes | Xorshift64 with Weyl sequence |
| `xorshift128` | uint32 | 16 bytes | 128-bit state xorshift |
| `pcg32` | uint32 | 8 bytes | PCG-XSH-RR 32-bit |
| `lcg32` | uint32 | 4 bytes | Generic 32-bit LCG |
| `java_lcg` | uint64 | 8 bytes | Java Random (48-bit) |
| `minstd` | uint32 | 4 bytes | Park & Miller minimal standard |
| `mt19937` | uint32 | 2496 bytes | Mersenne Twister (624-word state) |
| `xoshiro256pp` | uint64 | 32 bytes | Modern high-quality PRNG |
| `philox4x32` | uint64 | 8 bytes | Counter-based (GPU-optimized) |
| `sfc64` | uint64 | 32 bytes | Small Fast Counting |

### 3.2 Hybrid PRNGs (14 total)

> **Corrected 2026-11-11:** **11** forward hybrid entries, not 14. The table lists all 11
> correctly. What a hybrid kernel does with skip is set out in Chapter 2: it runs a greedy per-draw
> adaptive search seeded at `expected_skip = 5` (Chapter 2 §5.3). It does **not** read the
> sampled `skip_min`/`skip_max` (Chapter 2 §5.4, OPEN).

Variable skip pattern detection with multi-strategy search.

| PRNG | Base | Description |
|------|------|-------------|
| `xorshift32_hybrid` | xorshift32 | Variable skip detection |
| `xorshift64_hybrid` | xorshift64 | Variable skip detection |
| `xorshift128_hybrid` | xorshift128 | Variable skip detection |
| `pcg32_hybrid` | pcg32 | Variable skip detection |
| `lcg32_hybrid` | lcg32 | Variable skip detection |
| `java_lcg_hybrid` | java_lcg | Variable skip detection |
| `minstd_hybrid` | minstd | Variable skip detection |
| `mt19937_hybrid` | mt19937 | Variable skip detection |
| `xoshiro256pp_hybrid` | xoshiro256pp | Variable skip detection |
| `philox4x32_hybrid` | philox4x32 | Variable skip detection |
| `sfc64_hybrid` | sfc64 | Variable skip detection |

### 3.3 Reverse Sieve PRNGs (18 total)

> **Corrected 2026-11-11:** **22** reverse-direction entries (11 `_reverse` + 11
> `_hybrid_reverse`), not 18. The table below omits `xoshiro256pp` and `sfc64`. Their four entries
> are `xoshiro256pp_reverse`, `xoshiro256pp_hybrid_reverse`, `sfc64_reverse` and
> `sfc64_hybrid_reverse` (`prng_registry.py:4094-4131`). The two hybrid-reverse kernel sources name
> their entry points `*_hybrid_multi_strategy_sieve`, the same name as the forward hybrid
> (`:1726`, `:1867`, registry `kernel_name` at `:4103-4131`). They are compiled from separate
> sources, so the names do not collide in a module. Both families are uncovered by RANGE-MINER
> (§16).
>
> "Backward validation from candidate seeds" is also corrected. The reverse pass does not receive
> the forward survivors. The two passes score the same seed range on their own, and their seed sets
> are then intersected (Chapter 2 §4.2). "Reverse" means the order of the target residues, not an
> inverted generator (Chapter 2 §3.2; §9 below).

Backward validation from candidate seeds.

| Forward | Reverse | Hybrid Reverse |
|---------|---------|----------------|
| `xorshift32` | `xorshift32_reverse` | `xorshift32_hybrid_reverse` |
| `xorshift64` | `xorshift64_reverse` | `xorshift64_hybrid_reverse` |
| `xorshift128` | `xorshift128_reverse` | `xorshift128_hybrid_reverse` |
| `pcg32` | `pcg32_reverse` | `pcg32_hybrid_reverse` |
| `lcg32` | `lcg32_reverse` | `lcg32_hybrid_reverse` |
| `java_lcg` | `java_lcg_reverse` | `java_lcg_hybrid_reverse` |
| `minstd` | `minstd_reverse` | `minstd_hybrid_reverse` |
| `mt19937` | `mt19937_reverse` | `mt19937_hybrid_reverse` |
| `philox4x32` | `philox4x32_reverse` | `philox4x32_hybrid_reverse` |

---

## 4. CPU Reference Implementations

### 4.1 Purpose

CPU implementations serve as:
1. **Verification** — Validate GPU results
2. **Fallback** — When GPU unavailable
3. **Documentation** — Reference for algorithm behavior

> **Corrected 2026-11-11: two limits other chapters already hold.** They are cross-referenced
> here, not re-reported.
>
> - **Verification holds only at `skip = 0`.** Every CPU reference applies `skip` **once**, as a
>   burn-in before the first output (`java_lcg_cpu`, `prng_registry.py:176-177`). The sieve kernel
>   burns `skip` states before the first draw **and between every pair of draws**
>   (`java_lcg_flexible_sieve`, `:974-989`). The two agree only at `skip = 0`. This is held by
>   ruling as **Chapter 4 S1** (`docs/CHAPTER_4_AUDIT.md` §5), with `docs/BACKLOG.md` §8.1 saying
>   "No fix is authorised", and Chapter 2 §12.1 inherits it.
> - **As a fallback, the outputs are raw.** The references return unreduced 32-bit (for
>   `xoshiro256pp`/`sfc64`, 64-bit) values (e.g. `:181`). The scorer's batched CPU fallback stores
>   them without `% mod` (`survivor_scorer.py:802-809`). Every family without a PyTorch
>   implementation, 42 of 44, reaches that path. This is **Chapter 4 B4** (`docs/CHAPTER_4_AUDIT.md`
>   §4). The single-sequence path does reduce (`survivor_scorer.py:123-124`).
> - The 18 reverse entries named in §2.1 have **no** CPU reference at all.

### 4.2 Common Signature

```python
def prng_cpu(seed: int, n: int, skip: int = 0, **kwargs) -> List[int]:
    """
    Generate n outputs from PRNG starting at seed.
    
    Args:
        seed: Initial seed value
        n: Number of outputs to generate
        skip: Number of outputs to skip before first
        **kwargs: PRNG-specific parameters
        
    Returns:
        List of n integer outputs
    """
```

> **Checked 2026-11-11.** 12 of the 13 CPU functions have this signature (`prng_registry.py:153-404`).
> `mt19937_cpu` (`:335`), the reference registered for `mt19937_hybrid` (`:3779`), is
> `mt19937_cpu(seed, n, skip=0)` with no `**kwargs`. The forward `mt19937` entry uses
> `mt19937_cpu_simple` (`:3770`). The live callers pass only `seed`, `n` and `skip`
> (`survivor_scorer.py:123`, `:807`), so nothing fails today. The docstring's "skip: number of
> outputs to skip before first" correctly describes the CPU code. Its difference from the kernel is
> the §4.1 note above.

### 4.3 Java LCG Example

*Verified 2026-11-11 against `java_lcg_cpu`, `prng_registry.py:170-183` (same logic; comments added in the chapter). The state is 48-bit; the 32-bit value is the extracted output (Chapter 2 F-6).*

```python
def java_lcg_cpu(seed: int, n: int, skip: int = 0, **kwargs) -> List[int]:
    """Java LCG (java.util.Random) CPU reference"""
    a = kwargs.get('a', 25214903917)
    c = kwargs.get('c', 11)
    m = (1 << 48)  # 2^48
    
    state = seed & (m - 1)
    
    # Skip initial outputs
    for _ in range(skip):
        state = (a * state + c) & (m - 1)
    
    outputs = []
    for _ in range(n):
        state = (a * state + c) & (m - 1)
        output = (state >> 16) & 0xFFFFFFFF  # Extract bits 16-47
        outputs.append(output)
    
    return outputs
```

### 4.4 MT19937 Example

*Verified 2026-11-11 against `mt19937_cpu_simple`, `prng_registry.py:371-404`. `mt19937` is one of the five base families RANGE-MINER does not cover (§16).*

```python
def mt19937_cpu_simple(seed: int, n: int, skip: int = 0, **kwargs) -> List[int]:
    """MT19937 with init_genrand (matches GPU kernel)"""
    
    # Initialize 624-word state array
    state = [0] * 624
    state[0] = seed & 0xFFFFFFFF
    for i in range(1, 624):
        state[i] = (1812433253 * (state[i-1] ^ (state[i-1] >> 30)) + i) & 0xFFFFFFFF
    
    index = 624  # Force twist on first extract
    
    def mt19937_extract():
        nonlocal index
        if index >= 624:
            # Twist operation
            for i in range(624):
                y = (state[i] & 0x80000000) + (state[(i+1) % 624] & 0x7FFFFFFF)
                state[i] = state[(i + 397) % 624] ^ (y >> 1)
                if y % 2 != 0:
                    state[i] ^= 0x9908B0DF
            index = 0
        
        # Extract and temper
        y = state[index]
        index += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & 0xFFFFFFFF
    
    # Skip outputs
    for _ in range(skip):
        mt19937_extract()
    
    # Generate outputs
    return [mt19937_extract() for _ in range(n)]
```

---

## 5. GPU Kernel Architecture

### 5.1 Common Kernel Structure

> **Corrected 2026-11-11: "all GPU kernels" means the 22 constant-skip kernels.** The pattern below
> is the forward and reverse **constant-skip** pattern. A parse of all 44 kernel signatures at
> `a15b524` confirms it for all 22 of them: they all take `best_skips`, `skip_min`, `skip_max` and
> a trailing `int offset`. Chapter 2 §2.2 verifies the loop body against `java_lcg_flexible_sieve`,
> `prng_registry.py:958-1004`. The 22 **hybrid** kernels follow §8 instead. They take
> `skip_sequences`/`strategy_ids` in place of `best_skips` and declare no skip bounds (Chapter 2
> §5.4).
>
> "`offset` (always LAST)" does not hold for the hybrids. **6 of the 11 forward hybrid kernels
> take no `offset` at all**: xorshift32, java_lcg, minstd, xorshift128, philox4x32 and sfc64
> (signatures at `:862`, `:1005`, `:1136`, `:1274`, `:1558`, `:2011`). The other five, mt19937,
> xoshiro256pp, pcg32, lcg32 and xorshift64, end in `int offset`. All 11 reverse hybrids end in
> `int offset`. This is the open item Chapter 2 §7.3 records (skill §2.7 #5). RANGE-MINER lists the
> four covered variants that cannot take a phase as `PHASE_INCAPABLE_VARIANTS`
> (`miner/range_miner_worker.py:159-161`).
>
> **Window-anchor change.** Under Window-Anchor Brief I, the host value that used to be called
> `offset` is split. `window_anchor` selects the residue window on the host and reaches no kernel
> argument. `generator_phase` is delivered through this same trailing `int offset`, in an unchanged
> position and dtype (`_generator_phase_tail`, `miner/range_miner_worker.py:295`; Chapter 2
> §7.2.1). The kernel ABI is unchanged for all 44 entries. Only the host-side name of the value
> moved. `generator_phase` is pinned to 0 in v1.

All GPU kernels follow this pattern:

```c
extern "C" __global__
void prng_flexible_sieve(
    // Input arrays
    TYPE* seeds,              // Candidate seeds
    unsigned int* residues,   // Target draws to match
    
    // Output arrays
    TYPE* survivors,          // Seeds that passed threshold
    float* match_rates,       // Match rate for each survivor
    unsigned char* best_skips,// Best skip value found
    unsigned int* survivor_count,  // Atomic counter
    
    // Parameters
    int n_seeds,              // Number of seeds to test
    int k,                    // Number of residues (window size)
    int skip_min,             // Minimum skip to try
    int skip_max,             // Maximum skip to try
    float threshold,          // Minimum match rate
    
    // PRNG-specific parameters...
    int offset                // Window offset (always LAST)
) {
    // Thread index
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_seeds) return;
    
    // Get this thread's seed
    TYPE seed = seeds[idx];
    
    float best_rate = 0.0f;
    int best_skip_val = 0;
    
    // Try each skip value
    for (int skip = skip_min; skip <= skip_max; skip++) {
        // Initialize PRNG state
        TYPE state = seed;
        
        // Pre-advance by offset
        for (int o = 0; o < offset; o++) {
            state = prng_step(state);
        }
        
        // Burn skip values before first draw
        for (int s = 0; s < skip; s++) {
            state = prng_step(state);
        }
        
        int matches = 0;
        
        // Generate and compare
        for (int i = 0; i < k; i++) {
            state = prng_step(state);
            unsigned int output = extract_output(state);
            
            // 3-lane modulo validation
            if (((output % 1000) == (residues[i] % 1000)) &&
                ((output % 8) == (residues[i] % 8)) &&
                ((output % 125) == (residues[i] % 125))) {
                matches++;
            }
            
            // Skip between draws
            for (int s = 0; s < skip; s++) {
                state = prng_step(state);
            }
        }
        
        float rate = (float)matches / (float)k;
        if (rate > best_rate) {
            best_rate = rate;
            best_skip_val = skip;
        }
    }
    
    // Store survivor if above threshold
    if (best_rate >= threshold) {
        unsigned int pos = atomicAdd(survivor_count, 1);
        survivors[pos] = seed;
        match_rates[pos] = best_rate;
        best_skips[pos] = (unsigned char)best_skip_val;
    }
}
```

### 5.2 3-Lane Modulo Validation

All kernels use a 3-lane modulo check for robustness:

```c
if (((output % 1000) == (residues[i] % 1000)) &&
    ((output % 8) == (residues[i] % 8)) &&
    ((output % 125) == (residues[i] % 125))) {
    matches++;
}
```

**Why 3 lanes?**
- `% 1000`: Matches the 3-digit draw directly
- `% 8`: Fast power-of-2 check
- `% 125`: Coprime to 8, covers different bit patterns

> **Corrected 2026-11-11 — deferred to Chapter 2, which owns this subject.** "All kernels" and
> "Why 3 lanes?" are both settled in Chapter 2 §6 and are not re-derived here:
>
> - **The count.** The three-lane conjunction appears in 43 of the registry's 44 kernels.
>   `mt19937_hybrid_multi_strategy_sieve` reduces once and compares once (Chapter 2 §6.2.1, the
>   machine-checked block that `tests/test_chapter2_content_gate.py` G-LANE-COUNT re-derives).
> - **The rationale.** Because 1000 = 8 × 125 and gcd(8, 125) = 1, the mod-8 and mod-125 conjuncts
>   follow from the mod-1000 conjunct. They add **no** filtering power and do not "cover different
>   bit patterns" (Chapter 2 §6.3–§6.4, F-1). Lane agreement as a **graded ML feature** is a
>   separate and legitimate use (Chapter 2 §6.6).
> - **Do not remove the lanes** on the strength of this note (Chapter 2 §6.5).

---

## 6. Kernel Memory Layout

### 6.1 Standard Sieve Memory

| Array | Size | Type | Purpose |
|-------|------|------|---------|
| `seeds` | n_seeds | uint32/uint64 | Input seeds |
| `residues` | k | uint32 | Target draws |
| `survivors` | n_seeds | uint32/uint64 | Output seeds |
| `match_rates` | n_seeds | float32 | Output rates |
| `best_skips` | n_seeds | uint8 | Output skip values |
| `survivor_count` | 1 | uint32 | Atomic counter |

### 6.2 Hybrid Sieve Memory

Additional arrays for variable skip:

| Array | Size | Type | Purpose |
|-------|------|------|---------|
| `skip_sequences` | n_seeds × k | uint32 | Skip pattern per survivor |
| `strategy_ids` | n_seeds | uint32 | Winning strategy ID |

> **Corrected 2026-11-11.** The two shapes above are right (`sieve_gpu_worker.py:251-252`;
> RANGE-MINER `_hybrid_prefix`, `miner/range_miner_worker.py:276-292`). The arrays are not purely
> "additional", though. A hybrid kernel has **no `best_skips`**, and it also takes two
> per-strategy **input** arrays, `strategy_max_misses` and `strategy_tolerances` (`int32`, length
> `n_strategies`), plus the `n_strategies` scalar. That makes the 13-element common prefix in
> `_hybrid_prefix`. The emitted `skip_sequences` reaches no later step (Chapter 2 §8.6).

### 6.3 Local Arrays (Per Thread)

```c
// Version 2.3+: Dynamic sizing up to 2048
unsigned int best_skip_seq[2048];     // Best skip sequence found
unsigned int current_skip_seq[2048];  // Current trial sequence
```

**CRITICAL FIX (v2.3):** Changed from `[512]` to `[2048]` to support larger windows.

### 6.4 Skip Sequence Stride

```c
// FIXED in v2.3: Dynamic stride based on k (window size)
skip_sequences[pos * k + i] = best_skip_seq[i];

// BROKEN (before v2.3): Hardcoded 512 stride
skip_sequences[pos * 512 + i] = best_skip_seq[i];  // WRONG!
```

---

## 7. Standard Sieve Kernels

*Verified 2026-11-11: §7.1 matches `xorshift32_flexible_sieve` (`prng_registry.py:408-464`) and §7.2 matches `java_lcg_flexible_sieve` (`:958-1004`). The only differences are comments, and the `(unsigned int)` cast on the residue side, which C integer promotion makes inert (Chapter 2 §6.2.1).*

### 7.1 Xorshift32 Kernel

```c
extern "C" __global__
void xorshift32_flexible_sieve(
    unsigned int* seeds, unsigned int* residues, unsigned int* survivors,
    float* match_rates, unsigned char* best_skips, unsigned int* survivor_count,
    int n_seeds, int k, int skip_min, int skip_max, float threshold,
    int shift_a, int shift_b, int shift_c, int offset
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_seeds) return;
    
    unsigned int seed = seeds[idx];
    float best_rate = 0.0f;
    int best_skip_val = 0;
    
    for (int skip = skip_min; skip <= skip_max; skip++) {
        unsigned int state = seed;
        
        // Pre-advance by offset
        for (int o = 0; o < offset; o++) {
            state ^= state << shift_a;
            state ^= state >> shift_b;
            state ^= state << shift_c;
        }
        
        // Burn skip values
        for (int s = 0; s < skip; s++) {
            state ^= state << shift_a;
            state ^= state >> shift_b;
            state ^= state << shift_c;
        }
        
        int matches = 0;
        for (int i = 0; i < k; i++) {
            // Generate output
            state ^= state << shift_a;
            state ^= state >> shift_b;
            state ^= state << shift_c;
            
            // 3-lane check
            if (((state % 1000) == (residues[i] % 1000)) &&
                ((state % 8) == (residues[i] % 8)) &&
                ((state % 125) == (residues[i] % 125))) {
                matches++;
            }
            
            // Skip between draws
            for (int s = 0; s < skip; s++) {
                state ^= state << shift_a;
                state ^= state >> shift_b;
                state ^= state << shift_c;
            }
        }
        
        float rate = (float)matches / (float)k;
        if (rate > best_rate) {
            best_rate = rate;
            best_skip_val = skip;
        }
    }
    
    if (best_rate >= threshold) {
        unsigned int pos = atomicAdd(survivor_count, 1);
        survivors[pos] = seed;
        match_rates[pos] = best_rate;
        best_skips[pos] = (unsigned char)best_skip_val;
    }
}
```

### 7.2 Java LCG Kernel

```c
extern "C" __global__
void java_lcg_flexible_sieve(
    unsigned long long* seeds, unsigned int* residues, unsigned long long* survivors,
    float* match_rates, unsigned char* best_skips, unsigned int* survivor_count,
    int n_seeds, int k, int skip_min, int skip_max, float threshold,
    unsigned long long a, unsigned long long c, int offset
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_seeds) return;
    
    unsigned long long seed = seeds[idx];
    const unsigned long long m = 0xFFFFFFFFFFFFULL;  // 2^48 - 1
    
    float best_rate = 0.0f;
    int best_skip_val = 0;
    
    for (int skip = skip_min; skip <= skip_max; skip++) {
        unsigned long long state = seed & m;
        
        // Pre-advance and burn
        for (int o = 0; o < offset; o++) state = (a * state + c) & m;
        for (int s = 0; s < skip; s++) state = (a * state + c) & m;
        
        int matches = 0;
        for (int i = 0; i < k; i++) {
            state = (a * state + c) & m;
            unsigned int output = (state >> 16) & 0xFFFFFFFF;
            
            if (((output % 1000) == (residues[i] % 1000)) &&
                ((output % 8) == (residues[i] % 8)) &&
                ((output % 125) == (residues[i] % 125))) {
                matches++;
            }
            
            for (int s = 0; s < skip; s++) state = (a * state + c) & m;
        }
        
        float rate = (float)matches / (float)k;
        if (rate > best_rate) {
            best_rate = rate;
            best_skip_val = skip;
        }
    }
    
    if (best_rate >= threshold) {
        unsigned int pos = atomicAdd(survivor_count, 1);
        survivors[pos] = seed;
        match_rates[pos] = best_rate;
        best_skips[pos] = (unsigned char)best_skip_val;
    }
}
```

---

## 8. Hybrid Sieve Kernels

### 8.1 Hybrid Kernel Architecture

> **Checked 2026-11-11 against `java_lcg_hybrid_multi_strategy_sieve` (`prng_registry.py:1005-1084`).**
> The body below matches: `expected_skip = 5` (`:1027`), the `±skip_tolerance` window, the re-centre
> on a hit, the consecutive-miss abort, the best strategy kept, the `pos < n_seeds` guard and the
> `pos * k` stride (`:1077`). The chapter's `seq_size` is written inline in the code as
> `i < k && i < 2048`, which is equivalent. Three corrections, each already held by Chapter 2:
>
> - The template's trailing `int offset` is **not** in the java_lcg forward hybrid. Its signature
>   ends `float threshold, unsigned long long a, unsigned long long c` (`:1007-1012`). Six of the
>   eleven forward hybrids take no `offset` (§5.1 note; Chapter 2 §7.3).
> - Neither this template nor any of the 22 hybrid kernels declares `skip_min`/`skip_max`. The
>   sampled skip bounds therefore never constrain a hybrid pass (Chapter 2 §5.4, OPEN).
> - "Test each strategy … keep the best" describes the **forward** hybrid. The **reverse** hybrid
>   returns on the first strategy that clears the threshold (Chapter 2 §5.3; e.g. the `return;` in
>   `mt19937_hybrid_reverse_sieve`, `:3720`).

Hybrid kernels search for variable skip patterns using multiple strategies:

```c
extern "C" __global__
void prng_hybrid_multi_strategy_sieve(
    TYPE* seeds, unsigned int* residues, TYPE* survivors,
    float* match_rates, unsigned int* skip_sequences, unsigned int* strategy_ids,
    unsigned int* survivor_count, int n_seeds, int k,
    int* strategy_max_misses, int* strategy_tolerances, int n_strategies,
    float threshold, /* PRNG params... */ int offset
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_seeds) return;
    
    TYPE seed = seeds[idx];
    float best_match_rate = 0.0f;
    int best_strategy_id = 0;
    unsigned int best_skip_seq[2048];
    
    // Test each strategy
    for (int strat_id = 0; strat_id < n_strategies; strat_id++) {
        int max_misses = strategy_max_misses[strat_id];
        int skip_tolerance = strategy_tolerances[strat_id];
        
        TYPE state = seed;
        int matches = 0;
        int consecutive_misses = 0;
        int expected_skip = 5;  // Initial guess
        unsigned int current_skip_seq[2048];
        
        for (int draw_idx = 0; draw_idx < k && draw_idx < 2048; draw_idx++) {
            TYPE state_backup = state;
            bool found = false;
            int actual_skip = expected_skip;
            
            // Search window around expected skip
            int search_min = (expected_skip > skip_tolerance) ? 
                             (expected_skip - skip_tolerance) : 0;
            int search_max = expected_skip + skip_tolerance;
            
            for (int test_skip = search_min; test_skip <= search_max; test_skip++) {
                state = state_backup;
                
                // Advance by test_skip
                for (int j = 0; j < test_skip; j++) {
                    state = prng_step(state);
                }
                
                // Generate output
                TYPE temp_state = prng_step(state);
                unsigned int output = extract_output(temp_state);
                
                // Check match
                if (((output % 1000) == (residues[draw_idx] % 1000)) &&
                    ((output % 8) == (residues[draw_idx] % 8)) &&
                    ((output % 125) == (residues[draw_idx] % 125))) {
                    matches++;
                    consecutive_misses = 0;
                    actual_skip = test_skip;
                    expected_skip = test_skip;  // Adapt expectation
                    found = true;
                    state = temp_state;
                    break;
                }
            }
            
            if (draw_idx < 2048) current_skip_seq[draw_idx] = actual_skip;
            
            if (!found) {
                consecutive_misses++;
                if (consecutive_misses >= max_misses) break;  // Abandon strategy
            }
        }
        
        float match_rate = (float)matches / k;
        if (match_rate > best_match_rate) {
            best_match_rate = match_rate;
            best_strategy_id = strat_id;
            for (int i = 0; i < k && i < 2048; i++) {
                best_skip_seq[i] = current_skip_seq[i];
            }
        }
    }
    
    // Store survivor
    if (best_match_rate >= threshold) {
        int pos = atomicAdd(survivor_count, 1);
        if (pos < n_seeds) {
            survivors[pos] = seed;
            match_rates[pos] = best_match_rate;
            strategy_ids[pos] = best_strategy_id;
            
            // Store skip sequence with dynamic stride
            int seq_size = (k < 2048) ? k : 2048;
            for (int i = 0; i < seq_size; i++) {
                skip_sequences[pos * k + i] = best_skip_seq[i];
            }
        }
    }
}
```

### 8.2 Strategy Parameters

```c
// Typical strategy configurations
strategies = [
    {'max_consecutive_misses': 3, 'skip_tolerance': 5},   // Tight
    {'max_consecutive_misses': 5, 'skip_tolerance': 10},  // Medium
    {'max_consecutive_misses': 8, 'skip_tolerance': 15}   // Loose
]
```

| Parameter | Meaning |
|-----------|---------|
| `max_consecutive_misses` | Abort strategy after this many consecutive failures |
| `skip_tolerance` | Search window: `expected_skip ± tolerance` |

> **Corrected 2026-11-11.** The parameter meanings in the table are right (`hybrid_strategy.py:19-20`).
> The three "typical" configurations above (3/5, 5/10, 8/15) do **not** exist in the tree. Only
> 3/5 appears, as a fallback. The strategy set actually loaded is the five presets in
> `STRATEGY_PRESETS` (`hybrid_strategy.py:33-79`):
>
> | Preset | `max_consecutive_misses` | `skip_tolerance` |
> |---|---|---|
> | Strict Continuous (No Reseed) | 3 | 5 |
> | Lenient Continuous (Large Gaps OK) | 10 | 20 |
> | Aggressive Reseed Detection | 5 | 5 |
> | Balanced Hybrid (Recommended) | 7 | 10 |
> | Extreme Tolerance (Catch Everything) | 20 | 50 |
>
> RANGE-MINER takes the payload's `strategies` if one is given. Otherwise it takes
> `get_all_strategies()`, which returns all five, and falls back to the single `{3, 5}` only if
> `hybrid_strategy` cannot be imported (`_load_strategies`, `miner/range_miner_worker.py:1098-1109`).
> PWC does the same (`sieve_gpu_worker.py:239-249`). The kernel re-centres fully on each hit
> (`expected_skip = test_skip`, `prng_registry.py:1048`). A configured `skip_learning_rate` has no
> kernel counterpart; that is skill §2.7 #6, OPEN.

---

## 9. Reverse Sieve Kernels

### 9.1 Purpose

Reverse sieve kernels validate candidates from forward sieves by testing them against a **reversed** temporal sequence (looking backward in time).

> **Corrected 2026-11-11.** Reverse kernels do not take the forward survivors as input. Both
> passes score the same seed range on their own, with `seeds = arange(seed_start, …)` in both
> directions (`miner/range_miner_worker.py:1203-1205`). The two seed sets are then intersected,
> with no joint gate (Chapter 2 §4.2). "Reverse" is the order of the **target residues**. The
> generator still steps forward with the same recurrence as the forward kernel. There is no modular
> inverse anywhere in the tree (Chapter 2 §3.2; skill §2.6, "looks like a bug, isn't"). The kernel
> parameter is still named `candidate_seeds` (§9.3), which is only a name.

### 9.2 Temporal Reversal

**CRITICAL:** The residue array is reversed BEFORE being passed to the kernel (handled in `sieve_filter.py`):

```python
# In sieve_filter.py
if '_reverse' in prng_family:
    residues_reversed = residues[::-1]
    residues_gpu = cp.array(residues_reversed, dtype=residue_dtype)
```

> **Updated 2026-11-11 (RANGE-MINER).** The snippet above is verified at `sieve_filter.py:231-233`
> and again at `:394-396`. That is the **legacy** engine. PWC does the same at
> `sieve_gpu_worker.py:188-191`. The **certifying** engine is RANGE-MINER. It reverses at
> `residues[::-1] if reverse else residues` (`miner/range_miner_worker.py:1201`) and decides
> direction with `family_name.endswith("_reverse")` (`is_reverse_family`, `:215-216`). The legacy
> engine uses the substring test `'_reverse' in`. The two tests agree on every one of the 44
> registry names.

### 9.3 Reverse Kernel Example

```c
extern "C" __global__
void java_lcg_reverse_sieve(
    unsigned long long* candidate_seeds, unsigned int* residues,
    unsigned long long* survivors, float* match_rates,
    unsigned char* best_skips, unsigned int* survivor_count,
    int n_candidates, int k, int skip_min, int skip_max,
    float threshold, int offset
) {
    // Identical logic to forward kernel
    // Residues are already reversed by caller
    // ...
}
```

*Verified 2026-11-11: signature at `prng_registry.py:3115-3121`. `a` and `c` are hardcoded in the body (`:3125-3126`), so the reverse-constant ABI has no family tail. That gives 12 arguments (`EXPECTED_KERNEL_ARITY`, `miner/range_miner_worker.py:171-185`; Chapter 2 §3.4). Two registry descriptions call these kernels "backward validation" (`:3911`, `:3917`). The kernel bodies are forward recurrences, and the kernel text governs (Chapter 2 §3.5).*

---

## 10. PyTorch GPU Implementations

### 10.1 Purpose (v2.4)

PyTorch GPU implementations enable:
- **ML scoring** in Step 2.5 without CuPy
- **Cross-platform** support (CUDA + ROCm)
- **Batch processing** with native PyTorch tensors

> **Corrected 2026-11-11 (consumer and step).** The PyTorch path no longer serves "Step 2.5". That
> name is the conceptual one, and the executable scheme calls it Step 2, the scorer
> meta-optimizer. Executable Step 2 removed `SurvivorScorer` at v4.0 (`scorer_trial_worker.py:8`,
> `:103`), and nothing in it imports `prng_registry`. At `a15b524` the live consumer is **executable
> Step 3** (full scoring). `full_scoring_worker.py:375` calls `SurvivorScorer.extract_ml_features_batch`,
> which takes `get_pytorch_gpu_reference` when `has_pytorch_gpu` is true (`survivor_scorer.py:570-572`)
> and otherwise falls back to the unreduced CPU path (Chapter 4 B4). Only 2 of 44 entries have a
> PyTorch implementation, so for 42 the fallback is the normal path (§4.1 note).
>
> "Cross-platform (CUDA + ROCm)" is true on reading: the code is device-string agnostic. It was not
> run in this pass: **[UNVERIFIED]** as a runtime claim (audit U2).

### 10.2 Lazy Import Pattern

**CRITICAL:** PyTorch must not be imported at module load to avoid breaking CuPy on ROCm:

```python
# DO NOT: import torch  # Breaks CuPy kernel compilation on ROCm

# DO: Lazy import
def _torch_available() -> bool:
    """Check if PyTorch is installed without importing it."""
    return importlib.util.find_spec("torch") is not None

def _get_torch():
    """Lazy import torch - only when actually needed."""
    if not _torch_available():
        raise RuntimeError("PyTorch not available")
    import torch
    return torch
```

### 10.3 PyTorch GPU Function Signature

```python
def java_lcg_pytorch_gpu(
    seeds: 'torch.Tensor',    # [N] tensor of seeds
    n: int,                   # Number of outputs
    mod: int,                 # Modulo for output
    device: str = 'cuda',     # 'cuda' or 'cuda:0', etc.
    skip: int = 0,            # Initial skip
    **kwargs                  # PRNG-specific params
) -> 'torch.Tensor':          # [N, n] output tensor
```

### 10.4 Java LCG PyTorch Implementation

```python
def java_lcg_pytorch_gpu(
    seeds: 'torch.Tensor',
    n: int,
    mod: int,
    device: str = 'cuda',
    skip: int = 0,
    **kwargs
) -> 'torch.Tensor':
    """PyTorch GPU implementation of Java LCG"""
    torch = _get_torch()
    
    a = kwargs.get('a', 25214903917)
    c = kwargs.get('c', 11)
    mask = (1 << 48) - 1
    
    seeds = seeds.to(device).long()
    N = seeds.shape[0]
    
    # Initialize state (no XOR - matches CPU reference)
    state = seeds & mask
    
    # Apply initial skip
    for _ in range(skip):
        state = (a * state + c) & mask
    
    # Preallocate output
    output = torch.zeros((N, n), dtype=torch.int64, device=device)
    
    # Generate sequence (vectorized over seeds)
    for i in range(n):
        state = (a * state + c) & mask
        output[:, i] = ((state >> 16) & 0xFFFFFFFF) % mod
    
    return output
```

### 10.5 Helper Functions

```python
def get_pytorch_gpu_reference(prng_family: str) -> Callable:
    """Get PyTorch GPU implementation for PRNG family."""
    info = get_kernel_info(prng_family)
    if 'pytorch_gpu' not in info:
        raise ValueError(f"PyTorch GPU not implemented for '{prng_family}'")
    return info['pytorch_gpu']

def has_pytorch_gpu(prng_family: str) -> bool:
    """Check if PRNG has PyTorch GPU implementation."""
    return 'pytorch_gpu' in get_kernel_info(prng_family)

def list_pytorch_gpu_prngs() -> List[str]:
    """List all PRNGs with PyTorch GPU support."""
    return [name for name, info in KERNEL_REGISTRY.items() 
            if 'pytorch_gpu' in info]
```

> **Corrected 2026-11-11.** The live helpers (`prng_registry.py:4141-4180`) differ from the sketch
> above in three ways that change what a caller sees:
>
> - `get_pytorch_gpu_reference` first raises `RuntimeError("PyTorch not available")` when torch is not
>   installed (`:4143-4144`). Its `ValueError` names the families that do have an implementation and
>   points to `get_cpu_reference`.
> - `has_pytorch_gpu` returns **`False`** when torch is not installed, and **`False`** for an
>   unknown family, because it catches `ValueError` (`:4160-4169`). The sketch would raise.
> - `list_pytorch_gpu_prngs` returns `[]` when torch is not installed (`:4174-4175`). With torch
>   present it returns `['java_lcg', 'java_lcg_hybrid']` (live).

---

## 11. Adding New PRNGs

### 11.1 Step-by-Step Guide

1. **CPU Reference Implementation**
2. **GPU Kernel (Standard)**
3. **GPU Kernel (Hybrid)** — if variable skip needed
4. **GPU Kernel (Reverse)** — for bidirectional validation
5. **Registry Entry**
6. **Testing**

> **Corrected 2026-11-11: the procedure is incomplete at `a15b524`.** Phase 0 and RANGE-MINER
> changed what adding an entry involves. The six steps above give an entry that is **registered
> but not runnable** by the certifying engine. They also break a deliberate tripwire. Two more
> things are now part of the procedure:
>
> 7. **Encoding decision (S172 Phase 0).** The NPZ `prng_type` id comes from the sorted registry
>    keys (§15). Adding or removing a key shifts the ids. `tests/test_prng_encoding.py` pins
>    `len(PRNG_TYPE_ENCODING) == 44` (`:98-102`, `:236-243`) and fails **on purpose**. The fix is
>    an explicit decision to regenerate NPZs and/or bump `ENCODING_VERSION`
>    (`utils/prng_encoding.py:21-24`, `:34`). Do not just edit the assertion.
> 8. **RANGE-MINER coverage.** A new base family is **uncovered** until it has a
>    kernel-args builder in `kernel_args_builders` (`miner/range_miner_worker.py:464`) and appears in
>    `COVERED_FAMILIES` (`:106-108`), `EXPECTED_KERNEL_ARITY` (`:171`) and exactly one of
>    `PHASE_CAPABLE_VARIANTS`/`PHASE_INCAPABLE_VARIANTS` (`:142`, `:159`). Import-time assertions
>    enforce that partition (`:189-200`). Until then, `resolve_builder` raises
>    `NotImplementedError` before any GPU work (`:474-485`). Under the approved RANGE-MINER spec,
>    each such family is "a separate per-family session" (§16).

### 11.2 Template: CPU Reference

```python
def my_prng_cpu(seed: int, n: int, skip: int = 0, **kwargs) -> List[int]:
    """My PRNG CPU reference implementation"""
    # Get parameters
    param1 = kwargs.get('param1', DEFAULT_VALUE)
    
    # Initialize state
    state = seed
    
    # Skip initial outputs
    for _ in range(skip):
        state = step_function(state, param1)
    
    # Generate outputs
    outputs = []
    for _ in range(n):
        state = step_function(state, param1)
        outputs.append(extract_output(state))
    
    return outputs
```

### 11.3 Template: GPU Kernel

```c
extern "C" __global__
void my_prng_flexible_sieve(
    TYPE* seeds, unsigned int* residues, TYPE* survivors,
    float* match_rates, unsigned char* best_skips, unsigned int* survivor_count,
    int n_seeds, int k, int skip_min, int skip_max, float threshold,
    /* PRNG params */ int offset
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_seeds) return;
    
    TYPE seed = seeds[idx];
    float best_rate = 0.0f;
    int best_skip_val = 0;
    
    for (int skip = skip_min; skip <= skip_max; skip++) {
        TYPE state = seed;
        
        // Pre-advance by offset
        for (int o = 0; o < offset; o++) {
            // PRNG step function here
        }
        
        // Burn skip
        for (int s = 0; s < skip; s++) {
            // PRNG step function here
        }
        
        int matches = 0;
        for (int i = 0; i < k; i++) {
            // Generate output
            // PRNG step function here
            unsigned int output = /* extract */;
            
            // 3-lane check
            if (((output % 1000) == (residues[i] % 1000)) &&
                ((output % 8) == (residues[i] % 8)) &&
                ((output % 125) == (residues[i] % 125))) {
                matches++;
            }
            
            // Skip between draws
            for (int s = 0; s < skip; s++) {
                // PRNG step function here
            }
        }
        
        float rate = (float)matches / (float)k;
        if (rate > best_rate) {
            best_rate = rate;
            best_skip_val = skip;
        }
    }
    
    if (best_rate >= threshold) {
        unsigned int pos = atomicAdd(survivor_count, 1);
        survivors[pos] = seed;
        match_rates[pos] = best_rate;
        best_skips[pos] = (unsigned char)best_skip_val;
    }
}
```

### 11.4 Registry Entry

```python
KERNEL_REGISTRY['my_prng'] = {
    'kernel_source': MY_PRNG_KERNEL,
    'kernel_name': 'my_prng_flexible_sieve',
    'cpu_reference': my_prng_cpu,
    'default_params': {
        'param1': DEFAULT_VALUE,
    },
    'description': 'My PRNG description',
    'seed_type': 'uint32',  # or 'uint64'
    'state_size': 4,        # bytes
}
```

---

## 12. Helper Functions

### 12.1 get_kernel_info()

```python
def get_kernel_info(prng_family: str) -> Dict[str, Any]:
    """Get kernel configuration for PRNG family"""
    if prng_family not in KERNEL_REGISTRY:
        raise ValueError(
            f"Unknown PRNG family: {prng_family}. "
            f"Available: {list_available_prngs()}"
        )
    return KERNEL_REGISTRY[prng_family]
```

### 12.2 list_available_prngs()

```python
def list_available_prngs() -> List[str]:
    """List all available PRNG families"""
    return list(KERNEL_REGISTRY.keys())
```

### 12.3 get_cpu_reference()

```python
def get_cpu_reference(prng_family: str) -> Callable:
    """Get CPU reference implementation for PRNG"""
    return get_kernel_info(prng_family)['cpu_reference']
```

> **Checked 2026-11-11.** §12.1–§12.3 match `prng_registry.py:4187-4197`. One consequence follows
> from §2.1: `get_cpu_reference` raises **`KeyError`** for the 18 reverse entries that carry no
> `cpu_reference`. Every live caller found passes a base or forward name
> (`full_scoring_worker.py:254`, `:291`; `survivor_scorer.py:46` import; `synthetic_draw_injector.py:227`).
> One tool already works around the gap by falling back to the base entry
> (`generate_fingerprint_library.py:418-426`). A reverse entry's CPU sequence is the base family's
> sequence, because a reverse kernel steps the generator forward (§9).

---

## 13. Complete PRNG Reference

### 13.1 All 46 PRNGs

> **Corrected 2026-11-11:** the table has always had **44** rows, and 44 is the live registry size.
> The heading's "46" was wrong. Names, direction and the variable-skip column match
> `KERNEL_REGISTRY` row for row. The Seed and State columns for reverse entries come from the base
> family. Many reverse entries do not carry those keys themselves (§2.1).

| # | Name | Type | Seed | State | Variable Skip |
|---|------|------|------|-------|---------------|
| 1 | `xorshift32` | Forward | uint32 | 4B | No |
| 2 | `xorshift32_hybrid` | Forward | uint32 | 4B | **Yes** |
| 3 | `xorshift32_reverse` | Reverse | uint32 | 4B | No |
| 4 | `xorshift32_hybrid_reverse` | Reverse | uint32 | 4B | **Yes** |
| 5 | `xorshift64` | Forward | uint64 | 8B | No |
| 6 | `xorshift64_hybrid` | Forward | uint64 | 8B | **Yes** |
| 7 | `xorshift64_reverse` | Reverse | uint64 | 8B | No |
| 8 | `xorshift64_hybrid_reverse` | Reverse | uint64 | 8B | **Yes** |
| 9 | `xorshift128` | Forward | uint32 | 16B | No |
| 10 | `xorshift128_hybrid` | Forward | uint32 | 16B | **Yes** |
| 11 | `xorshift128_reverse` | Reverse | uint32 | 16B | No |
| 12 | `xorshift128_hybrid_reverse` | Reverse | uint32 | 16B | **Yes** |
| 13 | `pcg32` | Forward | uint32 | 8B | No |
| 14 | `pcg32_hybrid` | Forward | uint32 | 8B | **Yes** |
| 15 | `pcg32_reverse` | Reverse | uint32 | 8B | No |
| 16 | `pcg32_hybrid_reverse` | Reverse | uint32 | 8B | **Yes** |
| 17 | `lcg32` | Forward | uint32 | 4B | No |
| 18 | `lcg32_hybrid` | Forward | uint32 | 4B | **Yes** |
| 19 | `lcg32_reverse` | Reverse | uint32 | 4B | No |
| 20 | `lcg32_hybrid_reverse` | Reverse | uint32 | 4B | **Yes** |
| 21 | `java_lcg` | Forward | uint64 | 8B | No |
| 22 | `java_lcg_hybrid` | Forward | uint64 | 8B | **Yes** |
| 23 | `java_lcg_reverse` | Reverse | uint64 | 8B | No |
| 24 | `java_lcg_hybrid_reverse` | Reverse | uint64 | 8B | **Yes** |
| 25 | `minstd` | Forward | uint32 | 4B | No |
| 26 | `minstd_hybrid` | Forward | uint32 | 4B | **Yes** |
| 27 | `minstd_reverse` | Reverse | uint32 | 4B | No |
| 28 | `minstd_hybrid_reverse` | Reverse | uint32 | 4B | **Yes** |
| 29 | `mt19937` | Forward | uint32 | 2496B | No |
| 30 | `mt19937_hybrid` | Forward | uint32 | 2496B | **Yes** |
| 31 | `mt19937_reverse` | Reverse | uint32 | 2496B | No |
| 32 | `mt19937_hybrid_reverse` | Reverse | uint32 | 2496B | **Yes** |
| 33 | `xoshiro256pp` | Forward | uint64 | 32B | No |
| 34 | `xoshiro256pp_hybrid` | Forward | uint64 | 32B | **Yes** |
| 35 | `xoshiro256pp_reverse` | Reverse | uint64 | 32B | No |
| 36 | `xoshiro256pp_hybrid_reverse` | Reverse | uint64 | 32B | **Yes** |
| 37 | `philox4x32` | Forward | uint64 | 8B | No |
| 38 | `philox4x32_hybrid` | Forward | uint64 | 8B | **Yes** |
| 39 | `philox4x32_reverse` | Reverse | uint64 | 8B | No |
| 40 | `philox4x32_hybrid_reverse` | Reverse | uint64 | 8B | **Yes** |
| 41 | `sfc64` | Forward | uint64 | 32B | No |
| 42 | `sfc64_hybrid` | Forward | uint64 | 32B | **Yes** |
| 43 | `sfc64_reverse` | Reverse | uint64 | 32B | No |
| 44 | `sfc64_hybrid_reverse` | Reverse | uint64 | 32B | **Yes** |

### 13.2 PRNG Default Parameters

| PRNG | Parameters |
|------|------------|
| `xorshift32` | `shift_a=13, shift_b=17, shift_c=5` |
| `pcg32` | `increment=1442695040888963407` |
| `lcg32` | `a=1103515245, c=12345, m=0x7FFFFFFF` |
| `java_lcg` | `a=25214903917, c=11` |
| `minstd` | `a=48271, m=2147483647` |
| `mt19937` | (none - uses standard initialization) |
| `xoshiro256pp` | (none - uses standard constants) |
| `philox4x32` | (none - uses standard constants) |
| `sfc64` | (none - uses standard constants) |

---

## 14. Chapter Summary

**Chapter 3: PRNG Registry** covers the complete kernel library:

| Category | Count | Purpose |
|----------|-------|---------|
| **Fixed Skip** | 14 | Standard forward sieves |
| **Hybrid** | 14 | Variable skip detection |
| **Reverse** | 18 | Backward validation |
| **Total** | 46 | Complete PRNG coverage |

> **Corrected 2026-11-11.** This is **Chapter 8**, not Chapter 3. Chapter 3 is the scorer
> meta-optimizer. The counts are **11 / 11 / 22 (11 reverse + 11 hybrid-reverse) / 44** (§1.3), and
> "reverse" means forward stepping against time-reversed residues (§9), not backward validation.
> "Complete PRNG coverage" holds for the **registry**: all 11 bases have all 4 variants. It does
> not hold for **execution**. RANGE-MINER runs 24 of the 44 entries (6 bases × 4) and holds the
> other 20 back by the approved spec (§16, S1 in §17). TFM sieves `java_lcg` only.

**Key Features:**
- GPU kernels (CUDA/ROCm via CuPy)
- CPU reference implementations
- PyTorch GPU implementations (v2.4)
- Support for window sizes up to 2048 draws
- 3-lane modulo validation
- Multi-strategy hybrid search

*(2026-11-11: "3-lane modulo validation" is present in 43 of 44 kernels and is CRT-redundant with the mod-1000 test; see the §5.2 note and Chapter 2 §6.)*

---

## 15. The shared registry-derived encoding (S172 Phase 0)

*Added 2026-11-11. This did not exist when the chapter was written.*

The registry is also the **source of the NPZ `prng_type` id**. S172 Phase 0 (`2389b61`) replaced three
diverging hardcoded dicts with one module (`utils/prng_encoding.py:1-29`). Each of the old dicts
silently collapsed an unknown or hybrid type to `0` (`java_lcg`) and so destroyed provenance.

| Property | As built at `a15b524` | Anchor |
|---|---|---|
| Source of the mapping | `sorted(KERNEL_REGISTRY.keys())`; ids `0..N-1` in that order | `utils/prng_encoding.py:42-44` |
| Size | 44 ids (live). `java_lcg` = 0, `java_lcg_hybrid` = 1 (live) | live import |
| uint8 ceiling | raises at import if the registry ever exceeds 256 keys | `:47-51` |
| Unknown type | `encode_prng_type` raises `ValueError`; **no silent 0** | `:54-73` |
| Unknown id | `decode_prng_type` raises `ValueError`; **no silent `java_lcg`** | `:76-92` |
| Version | `ENCODING_VERSION = "3.2.0"` | `:34` |
| Tripwire | `tests/test_prng_encoding.py` pins the size at 44 | `:98-102`, `:236-243` |
| `skip_mode` | fixed `{'constant': 0, 'variable': 1}`, **not** registry-derived | `:36-38` |

**What follows for this chapter.** The registry's **key set** is an interface. Renaming, adding or
removing an entry changes every later id. The module's docstring says that NPZs are commit-local
artifacts and not a durable ABI (`:18-19`). That is why §11 now has a step 7. Chapter 2 §12.1
records that the registry's size is load-bearing for this encoding even though production compiles
only the `java_lcg` variants.

## 16. Which registry entries actually run: PWC branches and RANGE-MINER coverage

*Added 2026-11-11. RANGE-MINER changed what this chapter describes. The registry is complete, but
execution covers only part of it.*

**Counts re-derived live** (not taken from any document): 44 entries, 11 base families, 4 variants
each. The committed `CLAUDE.md` §7 figures were checked against code and **confirmed**:

| Engine | What it covers | Anchor |
|---|---|---|
| **PWC** `sieve_gpu_worker.py` (legacy, non-certifying) | **7** hardcoded `family_name ==` / `family_name in` branches covering **6 base families**: `xorshift32` (`:217`), `pcg32` (`:221`), `lcg32` (`:223`), `java_lcg` + `java_lcg_reverse` (`:227`), `java_lcg_hybrid` + `java_lcg_hybrid_reverse` (`:232`, with the forward/reverse split at `:259`), `minstd` (`:299`), `xorshift128` (`:302`). Any other name falls through to a generic launch with only the trailing `offset` (`:304`) | `sieve_gpu_worker.py:208-306` |
| **RANGE-MINER** `miner/range_miner_worker.py` (certifying) | `COVERED_FAMILIES` = {java_lcg, lcg32, minstd, pcg32, xorshift128, xorshift32}. `SUPPORTED_VARIANTS` = all 4 variants of each, **24 entries** | `:105-108`, `:119-124` |
| **RANGE-MINER, uncovered** | `UNCOVERED_FAMILIES` = {mt19937, philox4x32, sfc64, xorshift64, xoshiro256pp}, **20 entries**. `resolve_builder` raises `NotImplementedError` naming the family, **before any GPU work, launching no kernel** | `:109-111`, `:474-485` |

The PWC hybrid branch has its own signature: extra `skip_sequences`/`strategy_ids` buffers. The
forward branch takes `a, c` and the reverse branch takes `offset` (`sieve_gpu_worker.py:232-284`).
RANGE-MINER reproduces this per variant and records each arity in `EXPECTED_KERNEL_ARITY`
(`miner/range_miner_worker.py:171-185`). Chapter 2 §3.4 gives the arities.

**What is actually compiled in production** is narrower still. TFM sieves `java_lcg` only (`CLAUDE.md`
§7), so only the four `java_lcg` variants are compiled (Chapter 2 §12.1). The other 40 entries are
a substrate on hand for mimicry experiments. None of them is a claim about which generator lies behind
the draws.

**Why the five are held back.** The approved RANGE-MINER spec (`docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md`,
status APPROVED, `:18`) requires that uncovered families "must fail fast with `NotImplementedError` until a
dedicated kernel-args builder is implemented and tested. No silent fallthrough; no degraded execution path"
(§11.I, `:586`). It also requires that "Coverage for each uncovered family is a separate per-family session,
prioritised by owner" (§5.3, `:370-372`). The same section records the PWC fallthrough at `:304` as "a
pre-existing PWC-mode bug" that could launch these families with the wrong arity. This is item **S1** in
§17.

## 17. Known gaps at a15b524

*Added 2026-11-11. Labels and evidence are in `docs/CHAPTER_8_AUDIT.md`. Items that another
chapter already holds are cross-referenced in the sections above and are **not** listed again here.*

### 17.1 BROKEN

**None.** No claim in this chapter describes a code path that the code at `a15b524` fails to perform
without a governing document saying so. The registry-related breaks that do exist are held elsewhere
and are not re-reported. They are Chapter 4 B4 (the unreduced CPU fallback), Chapter 2 §5.4 (hybrid
skip bounds) and Chapter 2 §7.3 / skill §2.7 #5 (no `offset` on six forward hybrids).

### 17.2 SWITCHED-OFF-BY-RULING

| # | What | Held by | What lifting it would enable |
|---|---|---|---|
| **S1** | 5 of the 11 registered base families (20 of the 44 entries: mt19937, philox4x32, sfc64, xorshift64, xoshiro256pp) have kernels in the registry but **no RANGE-MINER kernel-args builder**, and they raise `NotImplementedError` at dispatch (`miner/range_miner_worker.py:109-111`, `:474-485`). The chapter's "Complete PRNG coverage" (§14) holds only for the registry | `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md` (APPROVED) §11.I `:586-588`, §5.3 `:370-372` | One builder per family, each implemented and tested in its own session, would let the certifying engine sieve that family and emit survivors under the 22-array NPZ contract. The family would then be on hand as a further mimicry substrate. Each builder also needs an `EXPECTED_KERNEL_ARITY` row and a phase-capability classification (§11 step 8) |

### 17.3 DESCRIPTION-ONLY

| # | Was | Is | Where corrected |
|---|---|---|---|
| D1 | 46 PRNGs over 14 families; 14 fixed / 14 hybrid / 18 reverse (repeated in §1.3, §3.1–§3.3, §13.1, §14) | 44 entries, 11 bases, 11/11/11/11; 22 reverse-direction. Already recorded as `PIPELINE_BEHAVIOUR_MODEL.md` D18 | §1.1, §1.3, §3.1–§3.3, §13.1, §14 |
| D2 | `prng_registry.py` is 4,323 lines | 4,235 at `a15b524` (4,322 at `0101306`) | header |
| D3 | v2.3 fixed all `[512]` hybrid buffers | 10 remained in 9 reverse-hybrid sources until `7313a43` (S170-K512). The docstring's stride line was garbled by the same commit | §1.2 |
| D4 | xorshift64 "with Weyl sequence" | xorshift64\*, with no Weyl term | §3.1 |
| D5 | every entry carries the §2.1 keys, and `get_cpu_reference` works for every family | 18 reverse entries have no `cpu_reference`/`default_params` (`KeyError`). `mt19937_cpu` has no `**kwargs` | §2.1, §4.2, §12.3 |
| D6 | hybrid memory = standard + two "additional" arrays | hybrids drop `best_skips` and add two per-strategy input arrays | §6.2 |
| D7 | three "typical" strategies 3/5, 5/10, 8/15 | five presets, 3/5, 10/20, 5/5, 7/10, 20/50; fallback 3/5 | §8.2 |
| D8 | PyTorch path serves "Step 2.5" ML scoring | executable Step 2 no longer uses it. Its live consumer is executable Step 3 | §10.1 |
| D9 | helper sketches | live helpers guard on torch presence, and `has_pytorch_gpu` swallows unknown families | §10.5 |
| D10 | 6-step "Adding New PRNGs" | needs the Phase-0 encoding decision and RANGE-MINER builder coverage as well | §11.1 |
| D11 | "Chapter 3: PRNG Registry"; next chapter "Chapter 4: Meta-Optimizer (Step 2.5)" | this is Chapter 8; the next is Chapter 9 (GPU cluster infrastructure) | §14, Next Chapter, end line |

## 18. Open items

Nothing here is repaired. Each item is carried, not closed.

| Item | Status | Where tracked |
|---|---|---|
| S1: five uncovered base families | switched off by the approved spec; owner-prioritised, one session per family | §17.2; RANGE-MINER spec §5.3 |
| CPU reference vs kernel skip placement | held, "No fix is authorised" | Chapter 4 S1; `docs/BACKLOG.md` §8.1; Chapter 2 §12.1 |
| Unreduced CPU fallback for 42 of 44 families | BROKEN, owned by Chapter 4 | Chapter 4 B4 |
| Hybrid kernels ignore `skip_min`/`skip_max` | OPEN | Chapter 2 §5.4; skill §2.7 #4 |
| Six forward hybrids take no `offset`/`generator_phase` | OPEN (DEP-ABI-V2 recorded, not built, `miner/range_miner_worker.py:139`, `:596`, `:604`) | Chapter 2 §7.3; skill §2.7 #5 |
| `skip_learning_rate` has no kernel counterpart | OPEN | skill §2.7 #6 |
| PyTorch implementations exist for 2 of 44 entries | the docstring calls the rest "placeholders ready" (`prng_registry.py:14`), but there are no placeholder functions in the file | not tracked as a work item; recorded here only |
| Reverse entries without `cpu_reference` | DESCRIPTION-ONLY; no live caller affected | §12.3 |
| Runtime claims U1 (512→2048 removed the illegal access) and U2 (PyTorch on ROCm) | UNVERIFIED; this pass did no runs | audit §9 |

## 19. Verification declaration (VIR-1…6)

- **VIR-1, execution proof.** Every anchor above was opened in this session. The full list of files
  opened is in `docs/CHAPTER_8_AUDIT.md` §8. The family count came from a live, read-only
  `python3 -I -c` import of `prng_registry` and `utils.prng_encoding`. No kernel was compiled and
  no GPU was used.
- **VIR-2, clean control.** `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` was not edited
  (`git diff --quiet`). `tests/test_chapter2_content_gate.py` was run once after the edits. Its result
  is recorded in the audit §10a.
- **VIR-3, fault-injection control.** **Not applicable** to a documentation pass. None is invented.
- **VIR-4, completion sentinel.** §20.
- **VIR-5, unavailable-observer behaviour.** Any surface that could not be read is recorded as
  `UNAVAILABLE`, never as clean (audit §8). The rigs' deployed copies of `prng_registry.py` were
  not compared against VM 101 (no rig access in this pass): **UNAVAILABLE**.
- **VIR-6, audit claim scope.** The chapter's statements against what the code at `a15b524` does.
  The pass did not judge whether the design is right, and it does not propose removing or
  simplifying any component.
  - **Searched surfaces:** the tracked repo; gitignored files (`git ls-files -o -i --exclude-standard`;
    this found `prng_registry.py.s170_k512_bak`, which corroborates D3, and
    `agent_manifests/definitions.json`); git history (`git log -S`, `git show 7313a43`, `0101306`);
    and host state limited to read-only `ls`/`cat`.
  - **Governance trail searched:** all 140 `docs/TB_RULING_*`, `docs/PROPOSAL_*` and
    `docs/TEAM_ALPHA_*` files, plus `docs/BACKLOG.md`, `docs/TODO_CONSOLIDATED.md`,
    `docs/PIPELINE_BEHAVIOUR_MODEL.md` and `docs/TFM_PROJECT_FACTS_SKILL.md` §2.6–§2.7.
  - **Chapters searched:** Chapter 2 (closed), and the audits of Chapters 1, 3, 4, 5, 6, 9, 12, 13 and 14.

## 20. Closure statement

**Verified against:** code at `a15b524`. The working tree is identical for every file cited.

**What this pass did.** It audited the chapter claim by claim. There were 66 claims: 35 accurate, 0 BROKEN,
1 SWITCHED-OFF-BY-RULING, 11 DESCRIPTION-ONLY, 2 UNVERIFIED and 17 absorbed into items another
chapter or this chapter's own D1/D5/D11 already holds. The reconciliation is in the audit §10. It corrected every wrong statement in place with a visible note, and it added §15–§20. No
section was removed. Every pre-existing heading survives, including the misnumbered §14 and
"Next Chapter", which keep their original text with correction notes. The chapter got longer, not
shorter, so §3.3's >10% removal rule does not apply.

**What it did not do.** It did not change or repair any code, and it did not edit Chapter 2. It did
not run or compile any kernel or GPU, and it had no rig access.

**Completion sentinel:**

```
CHAPTER 8 CORRECTION:  PASS
```

`PASS` covers the declared scope: every substantive claim is labelled and the counts reconcile.
Every correction is anchored to code read in this session. The two UNVERIFIED items are runtime
claims, which a documentation pass cannot settle. They are declared as UNVERIFIED, not as
verified.

---

## Next Chapter

**Chapter 4: Meta-Optimizer (Step 2.5)** will cover:
- `meta_prediction_optimizer_anti_overfit.py` — ML model training
- Bayesian hyperparameter optimization
- 4 ML models: PyTorch, XGBoost, LightGBM, CatBoost
- Feature importance analysis

> **Corrected 2026-11-11.** The chapter after this one is **Chapter 9: GPU Cluster Infrastructure**
> (`docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md`). The pointer above was carried over from when
> this text was numbered Chapter 3. `meta_prediction_optimizer_anti_overfit.py` is covered by
> Chapter 6 (`docs/CHAPTER_6_ANTI_OVERFIT_TRAINING.md`).

---

*End of Chapter 3: PRNG Registry* — *Corrected 2026-11-11: end of **Chapter 8**: PRNG Registry.*
