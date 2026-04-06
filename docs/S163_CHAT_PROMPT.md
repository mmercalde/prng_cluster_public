# S163 SESSION CHAT PROMPT
**Date:** 2026-04-06
**Previous Session:** S162 (4-day debug marathon — VICTORY)
**HEAD:** `aa48ee4`
**Author:** Claude (Team Alpha Lead Dev)

---
## System State
**Cluster:**
- Zeus: `45.32.131.224` / `rzeus`, 2×RTX 3080Ti, venv: `~/venvs/torch/bin/activate`
- rrig6600: `192.168.3.120`, 8×RX 6600, venv: `~/rocm_env/bin/activate`
- rrig6600b: `192.168.3.154`, 8×RX 6600, venv: `~/rocm_env/bin/activate`
- rrig6600c: `192.168.3.162`, 8×RX 6600, venv: `~/rocm_env/bin/activate`
- All 3 rigs on `amdgpu-dkms 6.12.12` — **pinned, do not update**
- Transport: TCP-PWC (default)
**Git:**
- Private: `git@github.com:mmercalde/prng_cluster_project.git`
- Public: `git@github.com:mmercalde/prng_cluster_public.git`
- HEAD: `aa48ee4` on both remotes
- **Always clone public repo BEFORE asking for live code lines** — never ask for SSH code greps when repo can be cloned first
- **Never commit from Claude sandbox — dual push rule: `git push origin main && git push public main`**
**Step 1 Results (S162 Victory Run):**
- Best config: `W6_O64_evening_S3-37_FT0.68_RT0.7`
- 887 bidirectional survivors
- `optimal_window_config.json` ✅ written
- `bidirectional_survivors.json` ✅ 887 seeds
- `bidirectional_survivors_binary.npz` ⚠️ NOT updated correctly (NPZ bug)
- Seed coverage: 1,073,741,824 → 2,147,483,648 logged
- Working seed cap: `seed_cap_amd=100000` (stable), `seed_cap_nvidia=5000000`
## Priority 2 — HIGH — Fix `convert_survivors_to_binary.py` (blocks Step 2)
**Bug:** `UnboundLocalError: local variable 'np' referenced before assignment`
**Root cause:** Duplicate `import numpy as np` — once at module level (line 24),
once inside a function (line 62). Python treats `np` as local to that function,
so line 77's `np.array(...)` fails before the local import is reached.
**Fix:** Remove the duplicate `import numpy as np` at line 62 inside the function.
One-line fix. Verify NPZ writes correctly after fix.
**Verify fix works:**
```bash
ssh rzeus "cd ~/distributed_prng_analysis && python3 convert_survivors_to_binary.py bidirectional_survivors.json && echo OK"
```
## Priority 1 — MEDIUM — Implement S163 (`free_all_blocks()` removal)
**TB-approved proposal:** `docs/PROPOSAL_FREE_ALL_BLOCKS_REPLACEMENT_v1_0.md`
**Change:** Remove `free_all_blocks()` from `_best_effort_gpu_cleanup()` in
`sieve_gpu_worker.py`. Add sampling-gated memory instrumentation behind
`S163_MEM_DEBUG=1` env var.
**Rationale:** `free_all_blocks()` called concurrently from 8+ workers is a known
CuPy race condition (CuPy issue #4866). S155's 256MB pool cap makes it redundant.
This is likely a contributing factor to crashes at 2M seed cap post-DKMS.
**Validation:** Staged 3-rig testing: 500K → 1M → 2M seeds with memory telemetry.
If 2M passes clean, throughput increases dramatically from current ~3.5M seeds/sec.
**Key instrumentation requirements (TB-mandated):**
- Sample every 25 chunks (not every chunk)
- Log before AND after cleanup
- Log `pool_used`, `pool_total`, `n_free_blocks`, `VmRSS`, `VmSize`
- Only active when `S163_MEM_DEBUG=1`
- Threshold breach logging always active
## Priority 3 — MEDIUM — Fix Zeus 3080Ti `cudaErrorDevicesUnavailable` (same fix pattern as PWC SSH GPU path)
**Symptom:** 250/10,738 chunks failed in Trial 3 reverse hybrid sieve.
Error: `cudaErrorDevicesUnavailable` in `sieve_filter.py:291` — Zeus local GPU path.
**Root cause candidates:**
1. Zeus GPUs entering P8 idle state between 100K chunks (~25ms jobs)
2. `_localhost_semaphore = threading.Semaphore(2)` hardcoded — starves Zeus of work
3. CuPy device context not maintained between chunks on Zeus local path
**Proposed fixes (TB proposal needed — same pattern as S156 PWC SSH GPU stale worker fix):**
- Option A: `sudo nvidia-smi -pm 1` persistence mode before launch
- Option B: Tie `_localhost_semaphore` to `gpu_count` or `max_per_node`
- Option C: CuPy device context keepalive on Zeus local sieve path
## Priority 4 — Run Step 2
Once NPZ bug fixed and S163 implemented:
ssh rzeus "cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate && \
  PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 2 --end-step 2 \
  2>&1 | tee logs/s163_step2_run1.log"
## Key Principles (Never Forget)
- **Verify live code via SSH** before writing patches — never assume repo matches Zeus
- **Dry-run before apply** on all patch scripts
- **Fix forward, never restore from backup**
- **TB proposal before architectural changes**
- **Dual push always:** `git push origin main && git push public main`
- **Never commit from Claude sandbox**
- **kern.log first** for hard crashes
- **Seed cap `100000` is current stable config** — do not increase without S163 validation
- **All 3 rigs pinned** — never run `apt upgrade` without checking holds first
