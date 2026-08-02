#!/usr/bin/env python3
"""
tests/phase6/known_answer_reference.py — the INDEPENDENT known-answer reference
for the S172 bounded Phase-6 Miner Known-Answer Transfer Gate.

Authority: docs/CLAUDE_CODE_INSTRUCTIONS_BOUNDED_PHASE_6.md §3, requirement 1
("expectations generated independently of the registry, miner, coordinator,
backend and finalizer") and requirement 2 ("production semantics — raw seed
state, >>16, inter-draw skips, and forward iteration over reversed residues for
reverse mode").

WHAT INDEPENDENCE MEANS HERE, PRECISELY
---------------------------------------
This module imports NOTHING from the project. Its only imports are `json`,
`hashlib` and `struct` from the standard library. It does not import
`prng_registry`, `sieve_gpu_worker`, `miner.*`, `utils.*`, cupy or numpy. It
never reads a `kernel_source` string. It therefore cannot inherit a defect from
the code it is used to check — which is the entire point of a known-answer
reference and the reason a "reference" built by importing the registry would be
worthless.

It is NOT independent of the kernel *specification*. The four algorithms below
were transcribed BY HAND, this session, from the four live CUDA kernel sources
read out of `prng_registry.KERNEL_REGISTRY` on VM 101:

    java_lcg                -> java_lcg_flexible_sieve
    java_lcg_reverse        -> java_lcg_reverse_sieve
    java_lcg_hybrid         -> java_lcg_hybrid_multi_strategy_sieve
    java_lcg_hybrid_reverse -> java_lcg_hybrid_reverse_sieve

That is the correct relationship: a known-answer reference must encode what the
production kernel is SUPPOSED to compute. Transcription is the audited step;
`known_answer_gate.py` prints the four `kernel_name`s and the sha256 of each
live `kernel_source` alongside this file's own sha256, so a future reader can
tell whether the kernel the gate ran is still the kernel this file was written
against.

DERIVATION NOTES — THE SEMANTICS THAT DIFFER FROM `pa_sieve_validation_harness.py`
---------------------------------------------------------------------------------
The starting material named in the brief (`pa_sieve_validation_harness.py`,
S143) is misaligned with production in three ways, all confirmed against the
live kernel text:

  1. SEED SCRAMBLE. The old harness applies the java.util.Random constructor
     scramble `state = (seed ^ 0x5DEECE66D) & MASK`. **Production does not.**
     Every one of the four kernels begins `unsigned long long state = seed & m;`
     — the RAW seed, masked to 48 bits. The sieve searches raw LCG states, not
     `java.util.Random` constructor arguments.
  2. OUTPUT SHIFT. The old harness uses `state >> (48 - 31)`, i.e. `>> 17`.
     **Production uses `>> 16`**, then `& 0xFFFFFFFF`.
  3. REVERSE DIRECTION. The old harness steps the LCG backwards with a modular
     inverse. **Production has no inverse LCG.** Both reverse kernels iterate
     the generator FORWARD; the direction comes from the HOST reversing the
     residue window (`range_miner_worker.py:887-889`,
     `residues[::-1] if reverse else residues`). This module reproduces that: it
     takes the forward-order window and reverses it internally for reverse
     variants, so the reversal is part of the reference, not of its caller.

Two further production facts, neither of them in the old harness:

  4. INTER-DRAW SKIPS. The constant kernels burn `skip` states BEFORE the first
     draw AND between every subsequent pair of draws — not once up front. (This
     is the same distinction that makes `prng_registry.java_lcg_cpu` disagree
     with the kernel at non-zero skip; that mismatch is explicitly out of scope
     for this gate and this file does not use `java_lcg_cpu`.)
  5. MULTI-MODULO MATCH. A match requires agreement mod 1000 AND mod 8 AND mod
     125. Because 1000 = 8 x 125 with gcd(8,125) = 1, the mod-1000 test already
     implies the other two; the redundancy is transcribed verbatim anyway rather
     than "simplified", because the job of a reference is to mirror the
     specification, not to improve it.

FLOAT SEMANTICS
---------------
The kernels compute `((float)matches)/((float)k)` and compare in IEEE-754
BINARY32, against a `float32` threshold. Doing that arithmetic in Python
doubles would let a survivor sit on the wrong side of a `>=` at the boundary. So
every rate and every threshold here is passed through `_f32()`, an exact
round-to-nearest binary32 via `struct` — standard library, no numpy.

TIE-BREAKING, transcribed rather than assumed:
  * constant kernels keep the FIRST skip achieving a strictly greater rate
    (`if (rate > best_rate)`), so ties resolve to the LOWEST skip;
  * the forward hybrid keeps the FIRST strategy achieving a strictly greater
    rate, so ties resolve to the LOWEST strategy id;
  * the reverse hybrid does not maximise at all — it emits on the FIRST strategy
    whose rate clears the threshold and returns immediately.

SKIP-SEQUENCE BUFFERS. Both hybrid kernels write their per-draw skip sequence
into an UNINITIALISED per-thread stack array and, on an early break/`failed`,
leave the tail of that array unwritten before copying/emitting all `k` entries.
Any comparison of the emitted `skip_sequences` past the break index would be
comparing uninitialised device memory. This reference therefore reports, per
seed, how many leading entries are DEFINED, and the gate compares only those —
and pins its strategies so that no early break occurs, making all `k` defined.
"""
from __future__ import annotations

import hashlib
import json
import struct
from typing import Dict, List, Optional, Sequence, Tuple

# --- Java LCG constants, transcribed from the kernel text -------------------
# java_lcg_flexible_sieve takes a/c as kernel arguments; the registry's
# default_params supply {'a': 25214903917, 'c': 11}. The three other kernels
# hardcode `const unsigned long long a = 25214903917ULL; c = 11ULL;`.
JAVA_A = 25214903917          # 0x5DEECE66D
JAVA_C = 11                   # 0xB
MASK48 = 0xFFFFFFFFFFFF       # `const unsigned long long m = 0xFFFFFFFFFFFFULL`
OUT_MASK32 = 0xFFFFFFFF

# The forward hybrid kernel's hardcoded initial skip estimate. This is the
# `expected_skip = 5` of skill 2.7 #4 — the reason the sampled skip_min/skip_max
# never reach a hybrid kernel. It is transcribed here because the reference must
# reproduce what production DOES, not what the search space claims it does.
HYBRID_INITIAL_EXPECTED_SKIP = 5

# Both hybrid kernels declare `unsigned int ...[2048]` and guard `i < 2048`.
HYBRID_SEQ_CAP = 2048


def _f32(x: float) -> float:
    """Exact IEEE-754 binary32 round-to-nearest, standard library only."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _matches(output: int, residue: int) -> bool:
    """The kernels' three-way multi-modulo test, transcribed verbatim."""
    r = residue & OUT_MASK32
    return (
        (output % 1000) == (r % 1000)
        and (output % 8) == (r % 8)
        and (output % 125) == (r % 125)
    )


# ===========================================================================
# Residue-window derivation and fingerprint — reimplemented, not imported
# ===========================================================================

def load_residue_window(path: str, window_size: int,
                        sessions: Optional[Sequence[str]], offset: int) -> List[int]:
    """Independent reimplementation of the miner's canonical window derivation.

    Deliberately NOT `from miner.range_miner_worker import load_residue_window`.
    The gate stamps the sha256 of THIS function's output into the assignment
    payload as `residue_sha256`; the worker then recomputes it from its own
    derivation and raises `ResidueVerificationError` on a mismatch. So the two
    independent derivations are cross-checked by the production code path
    itself — an agreement that an import would have manufactured.
    """
    with open(path, "r") as f:
        data = json.load(f)
    if sessions:
        wanted = set(sessions)
        data = [e for e in data if e.get("session") in wanted]
    n = len(data)
    if n < window_size:
        raise ValueError(
            f"dataset {path!r} has only {n} entries, need window_size {window_size}")
    start = max(0, min(int(offset), n - window_size))
    window = data[start:start + window_size]
    return [int(e.get("full_state", e["draw"])) for e in window]


def sha256_residues(residues: Sequence[int]) -> str:
    """Independent reimplementation of the residue fingerprint contract:
    sha256 over compact JSON of the int list."""
    body = json.dumps([int(x) for x in residues], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


# ===========================================================================
# 1/2. CONSTANT-SKIP variants — java_lcg and java_lcg_reverse
# ===========================================================================

def constant_sieve(
    residues_forward: Sequence[int],
    seed_start: int,
    seed_count: int,
    *,
    reverse: bool,
    skip_min: int,
    skip_max: int,
    offset: int,
    threshold: float,
    a: int = JAVA_A,
    c: int = JAVA_C,
) -> Dict[int, Tuple[float, int]]:
    """Transcription of `java_lcg_flexible_sieve` / `java_lcg_reverse_sieve`.

    The two kernel bodies are IDENTICAL apart from a/c arriving as arguments in
    the forward kernel and being hardcoded in the reverse one — the reverse
    kernel does not step backwards. Direction is applied here, on the host side,
    exactly as `range_miner_worker.SieveExecutor.execute` applies it.

    Returns {seed: (match_rate_binary32, best_skip)} for every SURVIVOR.
    """
    residues = list(residues_forward)[::-1] if reverse else list(residues_forward)
    k = len(residues)
    thr = _f32(threshold)
    out: Dict[int, Tuple[float, int]] = {}

    for seed in range(seed_start, seed_start + seed_count):
        best_rate = _f32(0.0)
        best_skip = 0
        for skip in range(skip_min, skip_max + 1):
            state = seed & MASK48
            for _ in range(offset):
                state = (a * state + c) & MASK48
            for _ in range(skip):
                state = (a * state + c) & MASK48
            matches = 0
            for i in range(k):
                state = (a * state + c) & MASK48
                output = (state >> 16) & OUT_MASK32
                if _matches(output, residues[i]):
                    matches += 1
                for _ in range(skip):
                    state = (a * state + c) & MASK48
            rate = _f32(_f32(matches) / _f32(k))
            if rate > best_rate:          # strict: ties keep the LOWEST skip
                best_rate = rate
                best_skip = skip
        if best_rate >= thr:
            out[seed] = (best_rate, best_skip)
    return out


# ===========================================================================
# 3. FORWARD HYBRID — java_lcg_hybrid_multi_strategy_sieve
# ===========================================================================

def hybrid_forward_sieve(
    residues_forward: Sequence[int],
    seed_start: int,
    seed_count: int,
    *,
    strategies: Sequence[dict],
    threshold: float,
    a: int = JAVA_A,
    c: int = JAVA_C,
) -> Dict[int, Tuple[float, int, List[int], int]]:
    """Transcription of `java_lcg_hybrid_multi_strategy_sieve`.

    Note three production facts reproduced verbatim and NOT tidied:
      * there is NO `offset` argument on this kernel (skill 2.7 #5 — the sampled
        forward-hybrid offset dies here, in the ABI);
      * `expected_skip` starts at a hardcoded 5 and is re-centred on every hit,
        which is why the sampled skip_min/skip_max never reach this kernel
        (skill 2.7 #4);
      * on a MISS the kernel does NOT restore `state` to `state_backup` — it
        leaves it advanced by `search_max` steps from the backup. That is
        load-bearing for every subsequent draw and is replicated exactly.

    Returns {seed: (rate_binary32, strategy_id, skip_sequence, n_defined)}.
    `n_defined` is how many leading skip-sequence entries the winning strategy
    actually WROTE; entries past it are uninitialised device memory in the real
    kernel and must not be compared.
    """
    residues = list(residues_forward)
    k = len(residues)
    thr = _f32(threshold)
    out: Dict[int, Tuple[float, int, List[int], int]] = {}
    n_strategies = len(strategies)

    for seed in range(seed_start, seed_start + seed_count):
        best_rate = _f32(0.0)
        best_sid = 0
        best_seq: List[int] = [0] * k
        best_defined = 0
        for sid in range(n_strategies):
            max_misses = int(strategies[sid]["max_consecutive_misses"])
            tol = int(strategies[sid]["skip_tolerance"])
            state = seed & MASK48
            matches = 0
            consecutive_misses = 0
            expected_skip = HYBRID_INITIAL_EXPECTED_SKIP
            cur_seq = [0] * k
            n_written = 0
            for draw_idx in range(min(k, HYBRID_SEQ_CAP)):
                state_backup = state
                found = False
                actual_skip = expected_skip
                search_min = (expected_skip - tol) if expected_skip > tol else 0
                search_max = expected_skip + tol
                for test_skip in range(search_min, search_max + 1):
                    state = state_backup
                    for _ in range(test_skip):
                        state = (a * state + c) & MASK48
                    temp_state = (a * state + c) & MASK48
                    output = (temp_state >> 16) & OUT_MASK32
                    if _matches(output, residues[draw_idx]):
                        matches += 1
                        consecutive_misses = 0
                        actual_skip = test_skip
                        expected_skip = test_skip
                        found = True
                        state = temp_state
                        break
                    # NOT restored on a miss — see the docstring.
                cur_seq[draw_idx] = actual_skip
                n_written = draw_idx + 1
                if not found:
                    consecutive_misses += 1
                    if consecutive_misses >= max_misses:
                        break
            rate = _f32(_f32(matches) / _f32(k))
            if rate > best_rate:          # strict: ties keep the LOWEST sid
                best_rate = rate
                best_sid = sid
                best_seq = list(cur_seq)
                best_defined = n_written
        if best_rate >= thr:
            out[seed] = (best_rate, best_sid, best_seq, best_defined)
    return out


# ===========================================================================
# 4. REVERSE HYBRID — java_lcg_hybrid_reverse_sieve
# ===========================================================================

def hybrid_reverse_sieve(
    residues_forward: Sequence[int],
    seed_start: int,
    seed_count: int,
    *,
    strategies: Sequence[dict],
    threshold: float,
    offset: int,
    a: int = JAVA_A,
    c: int = JAVA_C,
) -> Dict[int, Tuple[float, int, List[int], int]]:
    """Transcription of `java_lcg_hybrid_reverse_sieve`.

    A DIFFERENT algorithm from the forward hybrid, not a mirrored one. The
    differences are transcribed, not harmonised:
      * it HAS an `offset` pre-advance (the forward hybrid does not);
      * its skip search runs `try_skip` from 0 upward to `skip_tolerance`, with
        no `expected_skip` re-centring;
      * on a miss it DOES restore `state = state_save`;
      * its miss budget is `consecutive_misses > max_consecutive_misses`
        (strictly greater), against the forward hybrid's `>=`;
      * it does not maximise over strategies. It emits on the FIRST strategy
        that both completes without failing and clears the threshold, and
        returns. So the reported strategy id is "first qualifying", never "best".

    Residues are reversed here, host-side, as the worker does.
    Returns {seed: (rate_binary32, strategy_id, skip_sequence, n_defined)}.
    """
    residues = list(residues_forward)[::-1]
    k = len(residues)
    thr = _f32(threshold)
    out: Dict[int, Tuple[float, int, List[int], int]] = {}

    for seed in range(seed_start, seed_start + seed_count):
        for sid in range(len(strategies)):
            max_cm = int(strategies[sid]["max_consecutive_misses"])
            tol = int(strategies[sid]["skip_tolerance"])
            state = seed & MASK48
            for _ in range(offset):
                state = (a * state + c) & MASK48
            matches = 0
            consecutive_misses = 0
            skip_seq = [0] * k
            n_written = 0
            failed = False
            i = 0
            while i < k and not failed:
                found = False
                try_skip = 0
                while try_skip <= tol and not found:
                    state_save = state
                    for _ in range(try_skip):
                        state = (a * state + c) & MASK48
                    state = (a * state + c) & MASK48
                    output = (state >> 16) & OUT_MASK32
                    if _matches(output, residues[i]):
                        found = True
                        matches += 1
                        consecutive_misses = 0
                        skip_seq[i] = try_skip
                    else:
                        state = state_save
                    try_skip += 1
                if not found:
                    consecutive_misses += 1
                    if consecutive_misses > max_cm:
                        failed = True
                    skip_seq[i] = 0
                n_written = i + 1
                i += 1
            if not failed:
                rate = _f32(_f32(matches) / _f32(k))
                if rate >= thr:
                    out[seed] = (rate, sid, skip_seq, n_written)
                    break          # the kernel's `return` — first qualifying wins
    return out


# ===========================================================================
# Planting helper — generate a residue stream a chosen seed WILL match
# ===========================================================================

def generate_constant_stream(seed: int, k: int, *, skip: int, offset: int = 0,
                             a: int = JAVA_A, c: int = JAVA_C) -> List[int]:
    """The exact 32-bit outputs a constant-skip kernel produces for `seed`.

    Same iteration order as `constant_sieve`: offset pre-advance, skip burn,
    then per draw (advance, emit, skip burn). Planting these as `full_state`
    guarantees the seed matches at rate 1.0 at that skip.
    """
    state = seed & MASK48
    for _ in range(offset):
        state = (a * state + c) & MASK48
    for _ in range(skip):
        state = (a * state + c) & MASK48
    out: List[int] = []
    for _ in range(k):
        state = (a * state + c) & MASK48
        out.append((state >> 16) & OUT_MASK32)
        for _ in range(skip):
            state = (a * state + c) & MASK48
    return out


def generate_variable_stream(seed: int, skips: Sequence[int], *, offset: int = 0,
                             a: int = JAVA_A, c: int = JAVA_C) -> List[int]:
    """The 32-bit outputs produced when the gap BEFORE each draw is `skips[i]`.

    This is the shape both hybrid kernels search for: burn `skips[i]` states,
    then emit. `generate_constant_stream(seed, k, skip=s)` is the special case
    `skips = [s] * k`.
    """
    state = seed & MASK48
    for _ in range(offset):
        state = (a * state + c) & MASK48
    out: List[int] = []
    for s in skips:
        for _ in range(int(s)):
            state = (a * state + c) & MASK48
        state = (a * state + c) & MASK48
        out.append((state >> 16) & OUT_MASK32)
    return out


def self_sha256() -> str:
    """sha256 of THIS file, so the report can pin which reference text ran."""
    with open(__file__, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
