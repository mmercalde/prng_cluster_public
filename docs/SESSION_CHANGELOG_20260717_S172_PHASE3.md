# SESSION_CHANGELOG — 2026-07-17 — S172 RANGE-MINER Phase 3 (worker daemon)

**Team Alpha (Claude) implementation. NOT committed/pushed — Michael reviews,
commits, and dual-pushes.**

## Scope
Phase 3 of PROPOSAL_S172_RANGE_MINER_v1_4_4 per `docs/S172_PHASE3_BRIEF.md`:
the per-GPU worker daemon + acceptance harness. Coordinator (Phase 4), NPZ
contract wall (Phase 5), and ROCm/multi-rig acceptance (Phase 6) remain out of
scope.

## Files added
- `miner/range_miner_worker.py` — per-GPU daemon.
- `tests/test_s172_phase3_worker.py` — 8-gate acceptance harness.

## What shipped
- **Kernel-arg builder registry (§5.3)** keyed by base family, no default entry.
  6 covered base families (java_lcg, lcg32, minstd, pcg32, xorshift128,
  xorshift32) → builders; 7 branches (java_lcg splits constant/hybrid). The 5
  uncovered families (mt19937, philox4x32, sfc64, xorshift64, xoshiro256pp)
  raise `NotImplementedError` at dispatch, naming the family, **before any GPU
  allocation or launch** (§11.I enforcement point).
- Builders emit a **declarative, dtype-tagged** arg list (`BufferArg`/`ScalarArg`),
  materialized to exact cupy dtypes at launch. This keeps the arg LAYOUT
  CPU-introspectable while preserving the ABI — notably the **uint64 a,c** on the
  java_lcg hybrid kernel. Layouts replicate the LIVE `sieve_gpu_worker.py:208-306`
  verbatim (forward hybrid 15 args ending uint64 a,c; reverse hybrid 14 args
  ending int32 offset; constant prefix 11 + tail + int32 offset).
- **Per-family VRAM caps (§12.4, TB Q2):** constant amd/nvidia 2M/5M, hybrid
  1M/2.5M, backend-selected. Sourced from argparse defaults (spec §7).
- **Sub-stripe partitioning:** ceil(seed_count/cap), contiguous, gap/overlap-free.
- **Protocol/socket layer:** `MinerFramedSocket` wraps the Phase 2 framing
  helpers around a real socket (wire-identical to pwc_transport_tcp.FramedSocket).
  READY handshake, stripe flow (sub_stripe_result / stripe_complete /
  stripe_error), heartbeat, status, shutdown.
- **Retry semantics (TB Q3):** worker only reports; a sub-stripe failure emits a
  well-formed `stripe_error` (retryable set, traceback populated) and the daemon
  stays alive. Retry orchestration is the coordinator's job (Phase 4).
- **Cold-start GPU warm (interface §2.5):** per-daemon single-GPU warm hook;
  cross-GPU sequencing noted as the spawner's job.
- GPU execution injected (`executor=`) so the socket/control-flow is fully
  CPU-testable with a stub.

## Harness result
`8/8 gates green`, exit 0, run under `~/venvs/torch` on VM 101 (cupy 13.5.1,
1 CUDA device — 3080 Ti). Gate 7 GPU smoke ran for real (java_lcg constant,
256 seeds, 0 survivors, no error). Gates 1–6 + gate 7-skip also verified green
under system `python3` with cupy absent (CPU-only box parity).

## Decisions carried from plan review (Michael-approved)
- Hybrid caps come from argparse flags at spec values; a code comment flags that
  `seed_cap_amd_hybrid` / `seed_cap_nvidia_hybrid` are **NOT yet in the WATCHER
  manifest v1.8.0** and Phase 4/7 must wire them. (Brief's "already defines them"
  provenance was inaccurate — confirmed by repo-wide grep; they exist only in the
  proposal doc §7.)
- Declarative arg list retained, but every element carries its ABI dtype and
  launch materializes exactly that dtype (uint64 a,c preserved).
- Did not touch the committed Phase 2 test's singular `seed_cap_hybrid`; new code
  uses the amd/nvidia split per spec.

## Open Phase-3 prerequisites (infra, not code — unchanged)
- michael→CT100 SSH key auth for rig smoke tests (not yet set up).
- Pin VM 101 `.177` static.

## Fallback parity
code=current, env=ok (VM 101 venv cupy 13.5.1; no new deps added — worker uses
existing cupy + `sieve_gpu_worker`/`prng_registry` already present).
