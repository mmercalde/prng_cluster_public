# Note: run scope for the first v7-launcher run (DRAFT, 2026-11-18)

**Prior record.**
- Read: `docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md`, `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md` §7.3, `miner/range_miner_coordinator.py`, `persistent_worker_coordinator.py`, and the live `prng_analysis.db` (read-only, `immutable=1`).
- Labels: the scope decision is NEW (owner, 2026-11-18). Everything else is ALREADY RECORDED as cited.
- Not searched: ser8, the private repository, the rigs (off).

**Owner decision, 2026-11-18.**
- The first run with the v7 launcher covers the lower half, [0, 2³¹), from seed 0, with the Run 4 pins.
- It is recorded as a NEW search: no `--search-repeat-of`.
- The intention is to run the full [0, 2³²) in one run later.
- In normal operation WATCHER will choose ranges from the learning loop.

**Context.** The lower half's only coverage is the 2026-08-18 Gate 12 record. It has no search record, and its parameters are not the Run 4 pins.

**1. Why 25,000 per phase.**
- The cap enforces "do not overwhelm the rigs". Run 3 froze after 13,146,485 survivors (`PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:155-165`).
- No document I read ties it to the 3080 Ti's memory or to Step 2/3 size (UNVERIFIED that none exists elsewhere).
- History: the starting cap was 23,515, attempt 9's proven phase-4 count. The owner raised it to 25,000 on 2026-11-14 (`CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md:5`).
- The lower half's exact phase-4 count is 23,410 (`:26`), so this run passes.
- To raise the cap to about 50,000: the owner commits a new `config_manifests/survivor_volume.cap`. The gate accepts only a file tracked and identical to HEAD (`scripts/survivor_volume_gate.py:21-23`).
- The record asks for "evidence from completed runs" first (`PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:164-165`). Beta asks for completed runs at the new operating point plus resource observations (`X-VOLUME-LIMIT_R2_BETA_REVIEW.md:116-118`).

**2. Staging for 64 slots.**
- VM101 `/` has 309 GB free (`df`, 2026-11-18). `~/miner_staging` holds 91 MB, mostly the ledger.
- The file retention requirement is derived, not fixed. The half range needed 6,528 files over 4 stages (`logs/gate12_20261115_174941.log`, `[S172-CAP]`), so the full range should need about 13,056.
- The byte cap is a fixed 16 GiB (`miner/range_miner_coordinator.py:604`).
- Per-run peak staging bytes were never measured, because staging empties after each run (UNVERIFIED).
- No launch-path gate hard-codes 32 stripes or the half range. The "32 stripes" in `gate12_launch.sh:4` and `scripts/gate12_concurrency_sampler.py:44` are comments. Only the parameter blocks (`gate12_launch.sh:470`, the Route A wrapper, the volume-gate parameter file) carry 2³¹ and would change.

**3. PWC cache.**
- PWC sets a fixed per-GPU cache, `CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_<N>`, in three places: `persistent_worker_coordinator.py:606` and `:750`, and `persistent/pwc_worker_service.py:305`.
- It is never cleared per run, so it reuses a warm cache exactly like the miner did before v7.
- Whether warm PWC kernels hit the phase-4 fault is UNVERIFIED. L-7's 4/4 warm faults are all on the miner path, and the PWC-era register faults were cold-start.
- If PWC is used as the backup, the same fresh-per-run cache rule should apply until that is tested.
