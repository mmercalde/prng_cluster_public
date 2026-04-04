# S162 Chat Prompt — PWC TCP Production Integration

## Session Context
Team Alpha (Claude) continuing distributed PRNG analysis system development.
Previous session: S161 — PWC TCP transport fully implemented, validated, and dashboard integrated.

## System State
- **Cluster:** Zeus (2×RTX 3080Ti, rzeus/45.32.131.224/192.168.3.127) + 3 AMD rigs (rrig6600=.120, rrig6600b=.154, rrig6600c=.162), 8×RX 6600 each = **26 GPUs total**
- **Repo:** `git clone https://github.com/mmercalde/prng_cluster_public.git` (clone at session start)
- **Latest commit:** `0c084f2` — all S161 work committed to both remotes

## S161 Accomplishments
- PWC TCP transport fully implemented with two-phase startup (`online → init → ready`)
- 26-GPU validated: 50M seeds in 53 seconds, 2,240,701 aggregate sps
- Web dashboard fully integrated — all 4 nodes showing active with correct per-GPU stats
- Benchmark: TCP-PWC is **10x faster** than SSH-PWC wall-clock, **2.8x faster** than ZMQ aggregate
- All 4 transport modes coexist: ephemeral, SSH-PWC, ZMQ, TCP-PWC

## S162 Primary Goals (in priority order)

### 1. Wire TCP-PWC into WATCHER pipeline
- Add `--pwc-transport tcp` to WATCHER manifest `default_params`
- WATCHER should use TCP-PWC as the default persistent worker transport
- Test via: `PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 1 --end-step 1`

### 2. Run original ephemeral coordinator benchmark
- Complete the comparison table — we have ZMQ, SSH-PWC, TCP-PWC numbers but not original coordinator
- Use WATCHER to invoke Step 1 with default (ephemeral) coordinator
- Record aggregate sps for fair comparison

### 3. Pre-warm CuPy kernel cache on all rigs
- Cold-start ROCm init takes ~90s per worker
- Pre-warming cache reduces this to ~10s
- Run a warmup script on each rig once so subsequent sessions start fast

### 4. Selfplay NN fix in `inner_episode_trainer.py`
- Remove hardcoded forbidden guard blocking NN in selfplay
- Add y-normalization to selfplay path (same fix as S121 for `train_single_trial.py`)

### 5. S110 root cleanup
- 884 stray files in project root need organizing

## Key Architecture Notes
- TCP-PWC invocation: `--pwc-transport tcp --pool-size 8 --min-workers 24`
- Workers persist across trials (session-scoped) — launch once, reuse
- Zeus runs local sieve path (no TCP worker) — participates via `_dispatch_local_sieve()`
- Dashboard: `http://45.32.131.224:5002`
- Worker logs: `/tmp/pwc_tcp_worker_<ip>_gpu<n>.log` on each rig

## Hard Rules (unchanged)
1. NEVER restore from backup — fix forward only
2. SESSION_CHANGELOG every chat — committed to `docs/`
3. Clone public repo at session start before any code work
4. VERIFY bugs internally first before presenting patches
5. Dual-push every commit: `git push origin main && git push public main`
6. DB rotation before every run (mandatory)

## File Delivery
Claude saves to `/mnt/user-data/outputs/` → Michael downloads on ser8 → `scp ~/Downloads/<file> rzeus:~/distributed_prng_analysis/<path>/`
