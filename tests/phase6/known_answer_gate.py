#!/usr/bin/env python3
"""
tests/phase6/known_answer_gate.py — the S172 bounded Phase-6
MINER KNOWN-ANSWER TRANSFER GATE.

Authority: docs/CLAUDE_CODE_INSTRUCTIONS_BOUNDED_PHASE_6.md §3.

WHAT THIS GATE IS, AND WHAT IT IS NOT
-------------------------------------
It is NOT a re-validation of the PRNG registry. Team Beta struck the broad
44-PRNG Wall-C requirement: known-answer validation of the registry was
established practice before this repository existed, the method is valid, the
references are genuine, and Michael's account is accepted as the historical
project record. None of that is repeated here and nothing here is offered as a
substitute for it.

What this gate asks is a different question, about a different subject.
RANGE-MINER did not exist when that validation was done. The question is
whether the known-answer result TRANSFERS to the new engine — whether the miner
worker, driven exactly as the coordinator drives it, computes the same bounded
survivor set that an independent reference computes from the same inputs.

Bounded to FOUR variants, per the ruling:
    java_lcg  ·  java_lcg_reverse  ·  java_lcg_hybrid  ·  java_lcg_hybrid_reverse

BETA'S SIX REQUIREMENTS AND WHERE EACH IS DISCHARGED
----------------------------------------------------
1. Expectations generated independently of the registry, miner, coordinator,
   backend and finalizer.
       -> `known_answer_reference.py`. Imports only json/hashlib/struct. See its
          header for what "independent" does and does not mean. This gate prints
          that file's sha256 and the sha256 of each live `kernel_source` it ran
          against, so the transcription can be re-audited.
2. Production semantics — raw seed state, >>16, inter-draw skips, forward
   iteration over reversed residues for reverse mode.
       -> transcribed in the reference; ACTIVELY PROVEN by fault injections
          F5/F6/F7, which re-derive expectations under the three WRONG semantics
          (Java constructor scramble, >>17, skip-applied-once) and require the
          comparator to reject each. A gate that merely claimed the right
          semantics would be indistinguishable from one that got them wrong.
3. Exercise the ACTUAL miner worker path — residue resolution, payload
   interpretation, argument builder, kernel launch, result extraction. Not
   merely a direct RawKernel call.
       -> every population runs `miner.range_miner_worker.SieveExecutor.execute`
          on a real `StripeAssignMessage` whose payload was built by the
          production `RangeMinerCoordinator.build_stripe_assign_payload`. The
          gate never constructs a kernel, never calls `_get_kernel`, and never
          touches cupy directly. `_InstrumentedExecutor` overrides only the
          documented single mockable kernel entry `_gpu_launch` — to RECORD the
          materialised argument vector and then call the real launch. Nothing is
          stubbed. §PATH-EVIDENCE prints what each of the five legs did.
4. Compare the COMPLETE BOUNDED SURVIVOR SET per population.
       -> exact-set comparison over the whole seed range: missing members, EXTRA
          members, and per-member value disagreement (match rate, best skip,
          strategy id, skip sequence). Planted-seed recovery is reported too, but
          it is never the acceptance criterion — a planted-seed test alone cannot
          see an extra false survivor, which is precisely the failure this
          engine could plausibly have.
5. Clean control and fault-injection control that the comparator demonstrably
   rejects.
       -> the eight populations are the clean control. `--faults` then runs EIGHT
          injections and requires the comparator to reject ALL of them. An
          undetected injection fails the gate.
6. Terminate with an unambiguous nonzero failure status.
       -> exit 0 only on PASS. FAIL -> 1, UNAVAILABLE -> 3, INCOMPLETE -> 4, and
          any unhandled exception -> 2. The sentinel is printed on every path.

SKIP-SEQUENCE COMPARISON — A DELIBERATE, STATED BOUND
------------------------------------------------------
Both hybrid kernels write their per-draw skip sequence into an UNINITIALISED
per-thread stack array and emit all `k` entries even when an early break left
the tail unwritten. Comparing those entries would be comparing uninitialised
device memory. Every strategy this gate uses therefore pins
`max_consecutive_misses` far above `k`, so no early break can occur and all `k`
entries are written; the reference reports `n_defined` per seed and the gate
asserts `n_defined == k` before comparing. If that assertion ever fails the
population is reported INCOMPLETE rather than silently narrowed.

Run:
    source ~/venvs/torch/bin/activate
    python tests/phase6/known_answer_gate.py            # clean + faults
    python tests/phase6/known_answer_gate.py --no-faults
    python tests/phase6/known_answer_gate.py --json out.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import known_answer_reference as REF                                  # noqa: E402

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"

VARIANTS = ("java_lcg", "java_lcg_reverse", "java_lcg_hybrid",
            "java_lcg_hybrid_reverse")

# Pinned far above every population's k so no early break can occur — see the
# skip-sequence note in the header. The two entries differ in `skip_tolerance`
# so the multi-strategy loop and the two DIFFERENT strategy-selection rules
# (forward hybrid = best rate; reverse hybrid = first qualifying) are exercised.
STRATEGIES = [
    {"max_consecutive_misses": 999, "skip_tolerance": 5},
    {"max_consecutive_misses": 999, "skip_tolerance": 3},
]


# ===========================================================================
# Instrumented executor — records the real launch, stubs nothing
# ===========================================================================

class _InstrumentedExecutor:
    """Wraps the production `SieveExecutor` and records what the five legs did.

    `_gpu_launch` is the executor's own documented single mockable kernel entry
    (`range_miner_worker.py:824`). This subclass does not mock it: it records the
    materialised argument vector, then calls the real implementation. So the
    kernel that runs is the production kernel, launched by production code, with
    production arguments.
    """

    def __init__(self, device_index: int = 0):
        from miner.range_miner_worker import SieveExecutor, ResidueResolver

        gate = self

        class _Exec(SieveExecutor):
            def _gpu_launch(self, kernel, blocks, threads, kernel_args):
                gate.launches.append({
                    "kernel_repr": type(kernel).__name__,
                    "blocks": int(blocks),
                    "threads": int(threads),
                    "n_args": len(kernel_args),
                    "arg_signature": [gate._describe(a) for a in kernel_args],
                })
                return super()._gpu_launch(kernel, blocks, threads, kernel_args)

        self.resolver = ResidueResolver()
        self._exec = _Exec(resolver=self.resolver, device_index=device_index)
        self.launches: List[dict] = []

    @staticmethod
    def _describe(a: Any) -> str:
        dt = getattr(a, "dtype", None)
        if dt is None:
            return type(a).__name__
        if getattr(a, "shape", ()) == ():
            try:
                return f"scalar:{dt}={a.item()}"
            except Exception:
                return f"scalar:{dt}"
        return f"buffer:{dt}[{a.shape[0]}]"

    def run(self, assign) -> Tuple[Any, dict]:
        self.launches = []
        outcome = self._exec.execute(assign, assign.seed_start, assign.seed_count)
        return outcome, (self.launches[-1] if self.launches else {})


# ===========================================================================
# Population definition + execution
# ===========================================================================

class Population:
    """One bounded seed range, one variant, one fully specified configuration."""

    def __init__(self, *, name, variant, phase, k, threshold, seed_start,
                 seed_count, plant_seed, plant_skips, offset=0,
                 skip_range=None, verbatim_payload=False, workdir=None):
        self.name = name
        self.variant = variant
        self.phase = phase
        self.k = k
        self.threshold = threshold
        self.seed_start = seed_start
        self.seed_count = seed_count
        self.plant_seed = plant_seed
        self.plant_skips = list(plant_skips)
        self.offset = offset
        self.skip_range = skip_range
        self.verbatim_payload = verbatim_payload
        self.workdir = workdir
        self.reverse = variant.endswith("_reverse")
        self.hybrid = "_hybrid" in variant
        self.dataset_path: Optional[str] = None
        self.residues: List[int] = []
        self.payload: Dict[str, Any] = {}

    # ---- planting --------------------------------------------------------
    def build_dataset(self) -> str:
        """Write a synthetic dataset whose window the planted seed generates.

        The residues go on disk in FORWARD (file) order. For the two reverse
        variants the worker reverses the window host-side, so the file order is
        the reverse of the generated stream — which is exactly how a reverse
        sieve is supposed to see a real dataset, and it means the reverse
        residue path is exercised rather than sidestepped.

        Both `draw` and `full_state` are written. `full_state` carries the exact
        32-bit kernel output so the plant matches unambiguously; `draw` is
        present because the canonical derivation
        (`entry.get("full_state", entry["draw"])`) evaluates `entry["draw"]`
        EAGERLY and raises KeyError without it — a real production constraint
        found by running it, not assumed.
        """
        stream = REF.generate_variable_stream(
            self.plant_seed, self.plant_skips, offset=self.offset)
        on_disk = stream[::-1] if self.reverse else stream
        recs = [{"date": f"2020-{1 + i // 28:02d}-{1 + i % 28:02d}",
                 "session": "midday", "draw": v % 1000, "full_state": v}
                for i, v in enumerate(on_disk)]
        path = os.path.join(self.workdir, f"ds_{self.name}.json")
        with open(path, "w") as f:
            json.dump(recs, f)
        self.dataset_path = path
        return path

    # ---- payload ---------------------------------------------------------
    def build_payload(self) -> Dict[str, Any]:
        """Built by the PRODUCTION coordinator method, then annotated.

        `build_stripe_assign_payload` is called unbound on an uninitialised
        instance: it touches no instance state except through
        `resolve_dataset_sha256`, which we pre-empt by passing `dataset_sha256`
        explicitly. That keeps the payload the coordinator's own construction —
        including its threshold contract, its direction resolution through the
        §6.8 phase table, and its `residue_sha256` — without standing up a
        coordinator, a fleet or a socket.
        """
        from miner.range_miner_coordinator import RangeMinerCoordinator

        # The residues are derived by the INDEPENDENT reference, and their
        # sha256 goes into the payload. The worker recomputes both from its own
        # canonical derivation and raises ResidueVerificationError on a
        # mismatch, so the two independent derivations are cross-checked by
        # production code rather than by assertion.
        self.residues = REF.load_residue_window(
            self.dataset_path, self.k, ["midday"], 0)
        with open(self.dataset_path, "rb") as f:
            ds_sha = hashlib.sha256(f.read()).hexdigest()

        payload = RangeMinerCoordinator.build_stripe_assign_payload(
            RangeMinerCoordinator.__new__(RangeMinerCoordinator),
            self.dataset_path, self.k, ["midday"], self.offset, self.residues,
            dataset_sha256=ds_sha, phase=self.phase,
            forward_threshold=self.threshold, reverse_threshold=self.threshold)
        if not self.verbatim_payload:
            if self.skip_range is not None:
                payload["skip_range"] = list(self.skip_range)
            if self.hybrid:
                payload["strategies"] = [dict(s) for s in STRATEGIES]
        self.payload = payload
        return payload

    def assign(self, **overrides):
        from miner.range_miner_protocol import StripeAssignMessage
        payload = dict(self.payload)
        payload.update(overrides.pop("payload_overrides", {}) or {})
        return StripeAssignMessage(
            stripe_id=f"ka_{self.name}", trial_number=1,
            seed_start=self.seed_start, seed_count=self.seed_count,
            prng_type="java_lcg", family_name=self.variant, phase=self.phase,
            payload=payload, **overrides)

    # ---- expectation -----------------------------------------------------
    def effective_skip_range(self) -> Tuple[int, int]:
        """What the WORKER will use — read the way the worker reads it, so a
        verbatim coordinator payload (which carries no skip_range) resolves to
        the same production default [0, 16] on both sides."""
        lo, hi = tuple(self.payload.get("skip_range", [0, 16]))
        return int(lo), int(hi)

    def expected(self) -> Dict[int, tuple]:
        if not self.hybrid:
            lo, hi = self.effective_skip_range()
            exp = REF.constant_sieve(
                self.residues, self.seed_start, self.seed_count,
                reverse=self.reverse, skip_min=lo, skip_max=hi,
                offset=self.offset, threshold=self.threshold)
            return {s: (r, sk) for s, (r, sk) in exp.items()}
        strategies = self.payload.get("strategies", STRATEGIES)
        if self.reverse:
            raw = REF.hybrid_reverse_sieve(
                self.residues, self.seed_start, self.seed_count,
                strategies=strategies, threshold=self.threshold,
                offset=self.offset)
        else:
            raw = REF.hybrid_forward_sieve(
                self.residues, self.seed_start, self.seed_count,
                strategies=strategies, threshold=self.threshold)
        out = {}
        for seed, (rate, sid, seq, n_def) in raw.items():
            if n_def != self.k:
                raise _Incomplete(
                    f"population {self.name}: seed {seed} has only {n_def}/"
                    f"{self.k} DEFINED skip-sequence entries — the kernel broke "
                    f"early and the tail is uninitialised device memory. This "
                    f"population's strategies were supposed to make that "
                    f"impossible; refusing to compare uninitialised memory.")
            out[seed] = (rate, sid, tuple(seq))
        return out

    # ---- observation -----------------------------------------------------
    def observe(self, executor: _InstrumentedExecutor, **assign_kw
                ) -> Tuple[Dict[int, tuple], Any, dict]:
        outcome, launch = executor.run(self.assign(**assign_kw))
        got: Dict[int, tuple] = {}
        for seed, rate, sid, seq in outcome.survivors:
            if self.hybrid:
                got[int(seed)] = (float(rate), int(sid),
                                  tuple(int(x) for x in seq))
            else:
                got[int(seed)] = (float(rate), int(seq[0]))
        return got, outcome, launch


class _Incomplete(Exception):
    """A population could not be compared soundly -> INCOMPLETE, never PASS."""


def _assert_plant_in_scope(pop: "Population", expected: Dict[int, tuple]) -> None:
    """VIR-2 anti-vacuity guard, added because this gate failed it once.

    An exact-set comparison of an EMPTY expected set against an EMPTY observed
    set is `equal == True` and proves nothing. The first full-scale run of this
    gate produced exactly that on the two reverse populations, because the
    planted seed sat just outside the bounded seed range — so the TIGHT
    population compared nothing against nothing and reported PASS.

    Two conditions, both hard:
      * the planted seed must lie INSIDE the bounded range, so the population
        actually contains the answer it is named for;
      * the INDEPENDENT reference must find it, so a comparison that agrees is
        agreeing about a real survivor.
    Either failure is INCOMPLETE, never PASS.
    """
    lo, hi = pop.seed_start, pop.seed_start + pop.seed_count
    if not (lo <= pop.plant_seed < hi):
        raise _Incomplete(
            f"population {pop.name}: planted seed {pop.plant_seed} is OUTSIDE "
            f"the bounded range [{lo}, {hi}). The comparison would be vacuous.")
    if pop.plant_seed not in expected:
        raise _Incomplete(
            f"population {pop.name}: the INDEPENDENT reference does not recover "
            f"the planted seed {pop.plant_seed}. Either the plant or the "
            f"configuration is wrong; an agreement here would certify nothing.")
    if not expected:
        raise _Incomplete(
            f"population {pop.name}: the reference survivor set is EMPTY. "
            f"Empty-vs-empty is not evidence.")


# ===========================================================================
# The comparator — exact set, per population
# ===========================================================================

def compare(expected: Dict[int, tuple], observed: Dict[int, tuple]) -> dict:
    """Exact-set comparison of the COMPLETE bounded survivor set.

    Three independent failure modes, reported separately because they mean
    different things:
      missing   — the engine did not find a seed the reference says survives;
      extra     — the engine returned a seed the reference says does NOT survive
                  (the false-survivor class a planted-seed test cannot see);
      mismatch  — both agree the seed survives but disagree on its values.
    """
    exp_keys, obs_keys = set(expected), set(observed)
    missing = sorted(exp_keys - obs_keys)
    extra = sorted(obs_keys - exp_keys)
    mismatch = [(s, expected[s], observed[s])
                for s in sorted(exp_keys & obs_keys) if expected[s] != observed[s]]
    return {
        "expected_count": len(expected),
        "observed_count": len(observed),
        "missing": missing,
        "extra": extra,
        "mismatch": mismatch,
        "equal": not missing and not extra and not mismatch,
    }


# ===========================================================================
# Fault-injection oracles (F5/F6/F7) — the WRONG semantics, on purpose
# ===========================================================================

def _faulted_constant_sieve(residues_forward, seed_start, seed_count, *,
                            reverse, skip_min, skip_max, offset, threshold,
                            defect: str):
    """A deliberately MISALIGNED constant sieve, used only as a fault oracle.

    Kept here rather than in `known_answer_reference.py` so the reference stays
    a single-purpose, defect-free transcription. Each defect is a real,
    historically observed misalignment, not an invented one:

      'java_scramble'  state = (seed ^ 0x5DEECE66D) & MASK before iterating.
                       This is `pa_sieve_validation_harness.py`'s semantics, and
                       the reason that harness could not be used as-is.
      'shift_17'       output = (state >> 17) & 0xFFFFFFFF, again the S143
                       harness's semantics.
      'skip_once'      the skip is applied ONCE before generating instead of
                       between every draw. This is the `java_lcg_cpu`
                       (prng_registry.py:170-183) semantics, and it is exactly
                       the defect class Beta's Wall-C caution warned would make
                       a reference "validate the wrong semantics". Building the
                       gate on it would have produced a green light for a broken
                       engine; F7 proves the comparator sees it.
    """
    a, c, M = REF.JAVA_A, REF.JAVA_C, REF.MASK48
    residues = list(residues_forward)[::-1] if reverse else list(residues_forward)
    k = len(residues)
    thr = REF._f32(threshold)
    shift = 17 if defect == "shift_17" else 16
    out = {}
    for seed in range(seed_start, seed_start + seed_count):
        best_rate, best_skip = REF._f32(0.0), 0
        for skip in range(skip_min, skip_max + 1):
            state = ((seed ^ a) & M) if defect == "java_scramble" else (seed & M)
            for _ in range(offset):
                state = (a * state + c) & M
            for _ in range(skip):
                state = (a * state + c) & M
            matches = 0
            for i in range(k):
                state = (a * state + c) & M
                output = (state >> shift) & REF.OUT_MASK32
                if REF._matches(output, residues[i]):
                    matches += 1
                if defect != "skip_once":
                    for _ in range(skip):
                        state = (a * state + c) & M
            rate = REF._f32(REF._f32(matches) / REF._f32(k))
            if rate > best_rate:
                best_rate, best_skip = rate, skip
        if best_rate >= thr:
            out[seed] = (best_rate, best_skip)
    return out


def _ulp_bump(x: float) -> float:
    """Next binary32 upward — the smallest perturbation the comparator must see."""
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    return struct.unpack("<f", struct.pack("<I", bits + 1))[0]


# ===========================================================================
# Populations
# ===========================================================================

def build_populations(workdir: str, scale: float) -> List[Population]:
    """Two populations per variant, eight in all.

    DENSE  — a low threshold, so the survivor set is large. This is the
             population that can SEE an extra false survivor: with hundreds of
             legitimate near-miss survivors, an engine that admits one seed too
             many is detectable, which it would not be in a set of size one.
    TIGHT  — a high threshold, where essentially only the planted seed clears.
             This is the population that can see a MISSED true survivor.
    Neither alone is sufficient; the gate requires both.

    The two DENSE constant populations deliberately carry the coordinator's
    payload VERBATIM — no key added, no key changed — so at least one population
    per constant variant runs the exact dict production emits, skip range
    included (the worker's [0, 16] default).
    """
    def n(x):
        return max(200, int(x * scale))

    P = []
    # --- java_lcg (forward constant) -------------------------------------
    P.append(Population(
        name="java_lcg_dense", variant="java_lcg", phase=1, k=4, threshold=0.25,
        seed_start=123_400_000, seed_count=n(50_000), plant_seed=123_412_345,
        plant_skips=[3, 3, 3, 3], verbatim_payload=True, workdir=workdir))
    P.append(Population(
        name="java_lcg_tight", variant="java_lcg", phase=1, k=8, threshold=0.625,
        seed_start=123_400_000, seed_count=n(50_000), plant_seed=123_412_345,
        plant_skips=[2] * 8, skip_range=[0, 6], workdir=workdir))
    # --- java_lcg_reverse -------------------------------------------------
    # NOTE the planted seed here is 987_6*1*2_345, not the 987_654_321 that
    # reads more naturally: 987_654_321 falls OUTSIDE [987_600_000,
    # 987_650_000). The first full-scale run of this gate reported those two
    # populations as exact-set equal with the plant absent from BOTH sides —
    # a true statement that certified nothing, and in the TIGHT case an
    # empty-vs-empty comparison. `_assert_plant_in_scope` below now makes that
    # arrangement impossible rather than relying on the reader noticing.
    P.append(Population(
        name="java_lcg_reverse_dense", variant="java_lcg_reverse", phase=2, k=4,
        threshold=0.25, seed_start=987_600_000, seed_count=n(50_000),
        plant_seed=987_612_345, plant_skips=[2, 2, 2, 2],
        verbatim_payload=True, workdir=workdir))
    P.append(Population(
        name="java_lcg_reverse_tight", variant="java_lcg_reverse", phase=2, k=8,
        threshold=0.625, seed_start=987_600_000, seed_count=n(50_000),
        plant_seed=987_612_345, plant_skips=[4] * 8, skip_range=[0, 6],
        workdir=workdir))
    # --- java_lcg_hybrid (forward) ---------------------------------------
    # No offset: this kernel has no offset argument at all (skill 2.7 #5).
    P.append(Population(
        name="java_lcg_hybrid_dense", variant="java_lcg_hybrid", phase=3, k=4,
        threshold=0.25, seed_start=555_000_000, seed_count=n(25_000),
        plant_seed=555_012_345, plant_skips=[5, 6, 4, 7], workdir=workdir))
    P.append(Population(
        name="java_lcg_hybrid_tight", variant="java_lcg_hybrid", phase=3, k=8,
        threshold=0.625, seed_start=555_000_000, seed_count=n(25_000),
        plant_seed=555_012_345, plant_skips=[5, 6, 4, 7, 5, 3, 8, 5],
        workdir=workdir))
    # --- java_lcg_hybrid_reverse -----------------------------------------
    # Non-zero offset: this kernel DOES take one, and the planted stream is
    # generated with the same pre-advance, so the offset leg is load-bearing.
    P.append(Population(
        name="java_lcg_hybrid_reverse_dense", variant="java_lcg_hybrid_reverse",
        phase=4, k=4, threshold=0.25, seed_start=777_000_000,
        seed_count=n(25_000), plant_seed=777_012_345, plant_skips=[0, 2, 1, 3],
        offset=3, workdir=workdir))
    P.append(Population(
        name="java_lcg_hybrid_reverse_tight", variant="java_lcg_hybrid_reverse",
        phase=4, k=8, threshold=0.625, seed_start=777_000_000,
        seed_count=n(25_000), plant_seed=777_012_345,
        plant_skips=[0, 2, 1, 3, 5, 4, 2, 1], offset=3, workdir=workdir))
    return P


# ===========================================================================
# Provenance
# ===========================================================================

def print_provenance(workdir: str) -> dict:
    from prng_registry import KERNEL_REGISTRY

    ref_sha = REF.self_sha256()
    with open(os.path.abspath(__file__), "rb") as f:
        gate_sha = hashlib.sha256(f.read()).hexdigest()
    kernels = {}
    for v in VARIANTS:
        cfg = KERNEL_REGISTRY[v]
        src = cfg["kernel_source"]
        kernels[v] = {
            "kernel_name": cfg.get("kernel_name"),
            "seed_type": cfg.get("seed_type"),
            "default_params": cfg.get("default_params"),
            "kernel_source_sha256": hashlib.sha256(src.encode()).hexdigest(),
            "kernel_source_bytes": len(src),
        }
    print("=" * 78)
    print("MINER KNOWN-ANSWER TRANSFER GATE — PROVENANCE")
    print("=" * 78)
    print(f"  independent reference : tests/phase6/known_answer_reference.py")
    print(f"    sha256              : {ref_sha}")
    print(f"    imports             : json, hashlib, struct (stdlib only) — no "
          f"registry, miner,\n                          coordinator, backend, "
          f"finalizer, cupy or numpy")
    print(f"  gate                  : tests/phase6/known_answer_gate.py")
    print(f"    sha256              : {gate_sha}")
    print(f"  workdir               : {workdir}")
    print("\n  LIVE KERNEL SOURCES THE TRANSCRIPTION WAS AUDITED AGAINST")
    for v, info in kernels.items():
        print(f"    {v:<26} {info['kernel_name']}")
        print(f"      seed_type={info['seed_type']}  "
              f"default_params={info['default_params']}")
        print(f"      kernel_source sha256={info['kernel_source_sha256']} "
              f"({info['kernel_source_bytes']} bytes)")
    return {"reference_sha256": ref_sha, "gate_sha256": gate_sha,
            "kernels": kernels}


# ===========================================================================
# Path evidence — the five legs of requirement 3
# ===========================================================================

def path_evidence(executor: _InstrumentedExecutor, pop: Population) -> dict:
    """Negative controls proving each leg is real and load-bearing.

    Each of these drives the SAME production `SieveExecutor.execute` with one
    field corrupted, and requires the documented production exception. A leg
    that cannot be made to fail is a leg that was not doing anything.
    """
    from miner.range_miner_worker import (ResidueVerificationError,
                                          ResidueResolutionError,
                                          ThresholdContractError)
    out = {}

    def _expect(label, overrides, exc_types):
        try:
            executor.run(pop.assign(payload_overrides=overrides))
        except exc_types as e:
            out[label] = {"raised": type(e).__name__, "detail": str(e)[:160],
                          "detected": True}
            return
        except Exception as e:      # wrong exception is still a failure
            out[label] = {"raised": type(e).__name__, "detail": str(e)[:160],
                          "detected": False}
            return
        out[label] = {"raised": None, "detected": False}

    bad_res = "0" * 64
    _expect("residue_resolution.residue_sha256_mismatch",
            {"residue_sha256": bad_res}, ResidueVerificationError)
    _expect("residue_resolution.dataset_sha256_mismatch",
            {"dataset_sha256": bad_res}, ResidueVerificationError)
    _expect("residue_resolution.dataset_sha256_absent",
            {"dataset_sha256": None}, ResidueResolutionError)
    _expect("residue_resolution.window_fields_absent",
            {"window_size": None}, ResidueResolutionError)
    _expect("payload_interpretation.contradictory_thresholds",
            {"phase2_threshold": pop.threshold + 0.1}, ThresholdContractError)
    return out


# ===========================================================================
# Main
# ===========================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply every population's seed_count (smoke: 0.02)")
    ap.add_argument("--no-faults", action="store_true",
                    help="skip the fault-injection control (NOT acceptable for "
                         "certification — VIR-2 requires it)")
    ap.add_argument("--json", default=None, help="write the machine record here")
    args = ap.parse_args(argv)

    workdir = args.workdir or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "s172_phase6_known_answer")
    os.makedirs(workdir, exist_ok=True)

    record: Dict[str, Any] = {"gate": "miner_known_answer_transfer",
                              "authority": "docs/CLAUDE_CODE_INSTRUCTIONS_"
                                           "BOUNDED_PHASE_6.md §3"}
    record["provenance"] = print_provenance(workdir)

    try:
        executor = _InstrumentedExecutor(device_index=args.device_index)
    except Exception as e:
        print(f"\n[{_FAIL}] cannot construct the miner executor: {e}")
        print("\nSENTINEL: UNAVAILABLE")
        return 3

    pops = build_populations(workdir, args.scale)
    results = []
    abi = {}
    all_equal = True
    incomplete = False

    print("\n" + "=" * 78)
    print("CLEAN CONTROL — eight bounded populations, exact-set comparison")
    print("=" * 78)
    for pop in pops:
        pop.build_dataset()
        pop.build_payload()
        t0 = time.time()
        try:
            got, outcome, launch = pop.observe(executor)
            exp = pop.expected()
            _assert_plant_in_scope(pop, exp)
        except _Incomplete as e:
            incomplete = True
            print(f"\n  [{pop.name}] INCOMPLETE: {e}")
            results.append({"population": pop.name, "status": "INCOMPLETE",
                            "detail": str(e)})
            continue
        cmp = compare(exp, got)
        elapsed = time.time() - t0
        all_equal = all_equal and cmp["equal"]
        plant_in_exp = pop.plant_seed in exp
        plant_in_got = pop.plant_seed in got
        abi.setdefault(pop.variant, launch)

        lo, hi = pop.effective_skip_range()
        print(f"\n  --- {pop.name} " + "-" * (60 - len(pop.name)))
        print(f"      variant={pop.variant} phase={pop.phase} k={pop.k} "
              f"threshold={pop.threshold}")
        print(f"      seeds [{pop.seed_start:,}, "
              f"{pop.seed_start + pop.seed_count:,})  ({pop.seed_count:,} seeds)")
        if pop.hybrid:
            print(f"      strategies={pop.payload.get('strategies')}  "
                  f"offset={pop.offset}")
        else:
            print(f"      effective skip_range=[{lo},{hi}]"
                  f"{'  (coordinator payload VERBATIM — worker default)' if pop.verbatim_payload else ''}"
                  f"  offset={pop.offset}")
        print(f"      payload keys: {sorted(pop.payload)}")
        print(f"      kernel launch: {launch.get('n_args')} args, "
              f"blocks={launch.get('blocks')} threads={launch.get('threads')}")
        print(f"      reference survivors : {cmp['expected_count']:,}")
        print(f"      miner survivors     : {cmp['observed_count']:,}")
        print(f"      missing (reference-only) : {len(cmp['missing'])}"
              f"{'  ' + str(cmp['missing'][:5]) if cmp['missing'] else ''}")
        print(f"      EXTRA  (miner-only)      : {len(cmp['extra'])}"
              f"{'  ' + str(cmp['extra'][:5]) if cmp['extra'] else ''}")
        print(f"      value mismatches         : {len(cmp['mismatch'])}"
              f"{'  ' + str(cmp['mismatch'][:3]) if cmp['mismatch'] else ''}")
        print(f"      planted seed {pop.plant_seed}: reference={plant_in_exp} "
              f"miner={plant_in_got} "
              f"values={got.get(pop.plant_seed)}")
        print(f"      effective_threshold off the executor: "
              f"{outcome.effective_threshold}")
        print(f"      EXACT-SET EQUAL: {cmp['equal']}   ({elapsed:.1f}s)")

        results.append({
            "population": pop.name, "variant": pop.variant, "phase": pop.phase,
            "k": pop.k, "threshold": pop.threshold,
            "seed_start": pop.seed_start, "seed_count": pop.seed_count,
            "effective_skip_range": [lo, hi],
            "verbatim_coordinator_payload": pop.verbatim_payload,
            "payload_keys": sorted(pop.payload),
            "dataset_sha256": pop.payload.get("dataset_sha256"),
            "residue_sha256": pop.payload.get("residue_sha256"),
            "kernel_launch": launch,
            "effective_threshold": outcome.effective_threshold,
            "planted_seed": pop.plant_seed,
            "planted_recovered": bool(plant_in_got and plant_in_exp),
            "expected_count": cmp["expected_count"],
            "observed_count": cmp["observed_count"],
            "missing": cmp["missing"][:50], "extra": cmp["extra"][:50],
            "mismatch_count": len(cmp["mismatch"]),
            "exact_set_equal": cmp["equal"],
            "elapsed_s": round(elapsed, 2),
            "status": "PASS" if cmp["equal"] else "FAIL",
        })
    record["clean_control"] = results

    # ---- requirement 3: the five legs ------------------------------------
    print("\n" + "=" * 78)
    print("PATH EVIDENCE — the five legs of §3 requirement 3")
    print("=" * 78)
    print("  Every population above ran miner.range_miner_worker.SieveExecutor")
    print("  .execute() on a StripeAssignMessage whose payload came from the")
    print("  production RangeMinerCoordinator.build_stripe_assign_payload().")
    print("  The gate constructs no kernel and calls no RawKernel directly.\n")
    print("  ARGUMENT BUILDER — materialised vector per variant (recorded at the")
    print("  real launch site, not re-derived):")
    for v, launch in abi.items():
        print(f"    {v:<26} {launch.get('n_args')} args")
        for i, s in enumerate(launch.get("arg_signature", [])):
            print(f"        [{i:>2}] {s}")
    record["abi"] = abi

    print("\n  RESIDUE RESOLUTION and PAYLOAD INTERPRETATION — negative controls")
    print("  (each corrupts ONE payload field and requires the documented")
    print("  production exception; a leg that cannot fail was not doing work):")
    legs = path_evidence(executor, pops[0])
    record["path_evidence"] = legs
    legs_ok = True
    for label, info in legs.items():
        ok = info["detected"]
        legs_ok = legs_ok and ok
        print(f"    {'OK ' if ok else 'BAD'}  {label:<52} -> "
              f"{info['raised']}")
        if not ok:
            print(f"           expected a specific production exception, got "
                  f"{info['raised']!r}")
    record["path_evidence_ok"] = legs_ok

    # ---- requirement 5: fault injection ----------------------------------
    faults: List[dict] = []
    if args.no_faults:
        print("\n[WARN] fault injection SKIPPED (--no-faults). VIR-2 requires a "
              "fault-injection control; this run cannot certify.")
        record["fault_injection"] = "SKIPPED"
    else:
        print("\n" + "=" * 78)
        print("FAULT-INJECTION CONTROL — the comparator must REJECT all eight")
        print("=" * 78)
        faults = run_faults(executor, pops, workdir)
        record["fault_injection"] = faults
        for f in faults:
            print(f"    {'OK ' if f['rejected'] else 'BAD'}  {f['id']:<16} "
                  f"{f['description']}")
            print(f"           -> {f['evidence']}")

    faults_ok = all(f["rejected"] for f in faults) if faults else args.no_faults

    # ---- sentinel ---------------------------------------------------------
    print("\n" + "=" * 78)
    if incomplete:
        sentinel, code = "INCOMPLETE", 4
    elif args.no_faults:
        sentinel, code = ("INCOMPLETE", 4)
    elif all_equal and legs_ok and faults_ok:
        sentinel, code = "PASS", 0
    else:
        sentinel, code = "FAIL", 1
    print(f"MINER KNOWN-ANSWER TRANSFER GATE")
    print(f"  populations exact-set equal : {all_equal} "
          f"({sum(1 for r in results if r.get('exact_set_equal'))}/{len(results)})")
    print(f"  worker-path legs proven     : {legs_ok}")
    print(f"  fault injections rejected   : "
          f"{sum(1 for f in faults if f['rejected'])}/{len(faults)}"
          f"{' (SKIPPED)' if args.no_faults else ''}")
    print(f"  scope                       : java_lcg, java_lcg_reverse, "
          f"java_lcg_hybrid,\n                                java_lcg_hybrid_reverse "
          f"— four variants, as ruled")
    print(f"\nSENTINEL: {sentinel}")
    print("=" * 78)
    record["sentinel"] = sentinel

    if args.json:
        with open(args.json, "w") as f:
            json.dump(record, f, indent=2, sort_keys=True, default=str)
        print(f"[RECORD] {args.json}")
    return code


def run_faults(executor, pops, workdir) -> List[dict]:
    """Eight injections. Each must make `compare()` report NOT equal.

    Four are comparator-level (a corrupted observation) and four are
    semantic (expectations re-derived under a genuinely wrong rule, or the
    engine driven with a genuinely wrong configuration). The semantic ones are
    what make this a control rather than a self-test: F5/F6/F7 are the three
    misalignments that were actually present in the starting material or in
    `java_lcg_cpu`, and F8 drives the production path itself off-spec.
    """
    out = []
    const_pop = pops[0]        # java_lcg_dense
    hyb_pop = pops[4]          # java_lcg_hybrid_dense

    base_got, _outcome, _l = const_pop.observe(executor)
    base_exp = const_pop.expected()
    assert compare(base_exp, base_got)["equal"], (
        "fault injection needs a clean baseline and did not get one")
    hyb_got, _o2, _l2 = hyb_pop.observe(executor)
    hyb_exp = hyb_pop.expected()

    def rec(fid, desc, cmp_result, extra=""):
        rejected = not cmp_result["equal"]
        out.append({
            "id": fid, "description": desc, "rejected": rejected,
            "evidence": (f"missing={len(cmp_result['missing'])} "
                         f"extra={len(cmp_result['extra'])} "
                         f"mismatch={len(cmp_result['mismatch'])}"
                         f"{'; ' + extra if extra else ''}"),
        })

    # F1 — a true survivor silently dropped
    g = dict(base_got)
    dropped = sorted(g)[0]
    del g[dropped]
    rec("F1_drop_one", f"remove one true survivor ({dropped}) from the observation",
        compare(base_exp, g))

    # F2 — an EXTRA false survivor. The class a planted-seed test cannot see.
    g = dict(base_got)
    spurious = const_pop.seed_start + const_pop.seed_count - 1
    while spurious in base_exp:
        spurious -= 1
    g[spurious] = (1.0, 0)
    rec("F2_add_spurious", f"inject one EXTRA false survivor ({spurious})",
        compare(base_exp, g))

    # F3 — one match rate off by a single binary32 ULP
    g = dict(base_got)
    s = sorted(g)[0]
    g[s] = (_ulp_bump(g[s][0]), g[s][1])
    rec("F3_rate_1ulp", f"perturb one match_rate by 1 binary32 ULP (seed {s})",
        compare(base_exp, g))

    # F4 — one hybrid skip-sequence entry changed
    g = dict(hyb_got)
    s = sorted(g)[0]
    rate, sid, seq = g[s]
    g[s] = (rate, sid, tuple([seq[0] + 1] + list(seq[1:])))
    rec("F4_skip_seq", f"perturb one hybrid skip-sequence entry (seed {s})",
        compare(hyb_exp, g))

    # F5/F6/F7 — expectations re-derived under the WRONG production semantics
    lo, hi = const_pop.effective_skip_range()
    for fid, defect, desc in (
            ("F5_java_scramble", "java_scramble",
             "expectations under the java.util.Random constructor scramble "
             "(pa_sieve_validation_harness semantics)"),
            ("F6_shift_17", "shift_17",
             "expectations under output = state>>17 "
             "(pa_sieve_validation_harness semantics)"),
            ("F7_skip_once", "skip_once",
             "expectations with skip applied ONCE before generating instead of "
             "between draws (java_lcg_cpu semantics — Beta's Wall-C caution)")):
        bad_exp = _faulted_constant_sieve(
            const_pop.residues, const_pop.seed_start, const_pop.seed_count,
            reverse=const_pop.reverse, skip_min=lo, skip_max=hi,
            offset=const_pop.offset, threshold=const_pop.threshold,
            defect=defect)
        rec(fid, desc, compare(bad_exp, base_got),
            f"faulted reference produced {len(bad_exp)} survivors vs "
            f"{len(base_exp)} correct")

    # F8 — drive the PRODUCTION path off-spec and compare against the CORRECT
    # expectations. This injects into the miner path itself, not into the
    # observation record: the worker really does receive, transmit and filter at
    # the wrong value, so what the comparator rejects is a genuinely wrong run.
    #
    # The threshold is RAISED (0.25 -> 0.50) rather than lowered. With k=4 the
    # attainable rates are {0, .25, .5, .75, 1}, so lowering below 0.25 would
    # have to go all the way to 0.0 to change the set at all — a degenerate
    # "everything survives" case. Raising to 0.50 keeps only seeds with >= 2 of 4
    # matches, a strict and non-trivial subset, so the rejection is a real
    # missing-member signal rather than an artefact of a degenerate threshold.
    raised = round(const_pop.threshold + 0.25, 6)
    off_got, _o3, _l3 = const_pop.observe(
        executor, payload_overrides={"min_match_threshold": raised,
                                     "phase2_threshold": raised})
    rec("F8_engine_offspec",
        f"drive the production worker with min_match_threshold={raised} "
        f"instead of {const_pop.threshold}",
        compare(base_exp, off_got),
        f"engine returned {len(off_got)} survivors vs {len(base_exp)} expected")
    return out


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        print("\nSENTINEL: FAIL (unhandled exception)")
        sys.exit(2)
