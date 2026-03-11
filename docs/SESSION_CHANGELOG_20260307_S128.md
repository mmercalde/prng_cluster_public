# SESSION CHANGELOG — S128
**Date:** 2026-03-07
**Session:** S128
**Engineer:** Team Alpha (Michael + Claude)
**Status:** IN PROGRESS — probe scripts delivered, awaiting execution results

---

## Session Objectives
1. Confirm manifest fix dual-push (S127 carry-forward) ✅
2. Phase A — RTX 3080 Ti isolated single-card ceiling probe
3. Phase A — RX 6600 isolated single-card ceiling probe
4. Phase B — Full concurrent rig ceiling (both RTX / all 8 AMD simultaneously)
5. Phase C — Stability test (50 consecutive jobs at Phase B × 0.85)
6. Update `coordinator.py` seed caps and `gpu_optimizer.py` profiles
7. Commit both files to both repos

---

## Completed This Session

### 1. Manifest Fix Dual-Push Verification ✅

Cloned public repo at S128 open. Confirmed `006623c` is HEAD on both repos.
`agent_manifests/window_optimizer.json` inspected in-clone — S127 fix is live:

**Confirmed present in `default_params`:**
- `"enable_pruning": false`
- `"n_parallel": 1`
- `"resume_study": false`
- `"study_name": ""`

**Confirmed present in `actions[0].args_map`:**
- `"enable-pruning": "enable_pruning"`
- `"n-parallel": "n_parallel"`
- `"resume-study": "resume_study"`
- `"study-name": "study_name"`

**Status:** Both repos at `006623c`. Manifest fix live. No further action needed.

---

### 2. GPU Throughput Probe Scripts — Delivered ✅

Four files prepared for deployment:

| File | Purpose | Run On |
|------|---------|--------|
| `probe_phase_A_rtx.sh` | Single RTX 3080 Ti ceiling — 100k→500k→1M→2M→5M | Zeus |
| `probe_phase_A_amd.sh` | Single RX 6600 ceiling — 100k→250k→500k→1M→2M | rrig6600 |
| `probe_phase_B.sh` | Full concurrent ceiling — both GPU types | Zeus (rtx) / rrig6600 (amd) |
| `probe_phase_C_stability.sh` | 50-job stability gate at Phase B × 0.85 | Zeus |
| `apply_caps.py` | Patches coordinator.py + gpu_optimizer.py from measured values | Zeus |

**Deploy commands (from ser8):**
```bash
scp ~/Downloads/probe_phase_A_rtx.sh rzeus:~/distributed_prng_analysis/scripts/
scp ~/Downloads/probe_phase_A_amd.sh rzeus:~/distributed_prng_analysis/scripts/
scp ~/Downloads/probe_phase_B.sh rzeus:~/distributed_prng_analysis/scripts/
scp ~/Downloads/probe_phase_C_stability.sh rzeus:~/distributed_prng_analysis/scripts/
scp ~/Downloads/apply_caps.py rzeus:~/distributed_prng_analysis/scripts/
```

**Then copy AMD scripts to rig:**
```bash
ssh rzeus "scp ~/distributed_prng_analysis/scripts/probe_phase_A_amd.sh rrig6600:~/"
ssh rzeus "scp ~/distributed_prng_analysis/scripts/probe_phase_B.sh rrig6600:~/"
```

---

## Execution Protocol (Pending — Fill In Results)

### Phase A — RTX Isolated (Zeus)
```bash
ssh rzeus
cd ~/distributed_prng_analysis
chmod +x scripts/probe_phase_A_rtx.sh
bash scripts/probe_phase_A_rtx.sh 2>&1 | tee /tmp/probe_A_rtx.log
```

**Results table (fill after run):**
| Step | Seeds | seeds/sec | Peak VRAM (MB) | Duration (s) | Status |
|------|-------|-----------|----------------|--------------|--------|
| A1 | 100,000 | TBD | TBD | TBD | TBD |
| A2 | 500,000 | TBD | TBD | TBD | TBD |
| A3 | 1,000,000 | TBD | TBD | TBD | TBD |
| A4 | 2,000,000 | TBD | TBD | TBD | TBD |
| A5 | 5,000,000 | TBD | TBD | TBD | TBD |

**RTX Phase A ceiling: TBD seeds**

---

### Phase A — AMD Isolated (rrig6600)
```bash
ssh rrig6600
chmod +x probe_phase_A_amd.sh
bash probe_phase_A_amd.sh 2>&1 | tee /tmp/probe_A_amd.log
```

**Results table (fill after run):**
| Step | Seeds | seeds/sec | Peak VRAM (MB) | Duration (s) | Status |
|------|-------|-----------|----------------|--------------|--------|
| A1 | 100,000 | TBD | TBD | TBD | TBD |
| A2 | 250,000 | TBD | TBD | TBD | TBD |
| A3 | 500,000 | TBD | TBD | TBD | TBD |
| A4 | 1,000,000 | TBD | TBD | TBD | TBD |
| A5 | 2,000,000 | TBD | TBD | TBD | TBD |

**AMD Phase A ceiling: TBD seeds**

---

### Phase B — Full Concurrent (edit CEILING values in script first)
```bash
# Edit probe_phase_B.sh: set PHASE_A_RTX_CEILING and PHASE_A_AMD_CEILING

# RTX (Zeus):
bash scripts/probe_phase_B.sh rtx 2>&1 | tee /tmp/probe_B_rtx.log

# AMD (rrig6600):
bash ~/probe_phase_B.sh amd 2>&1 | tee /tmp/probe_B_amd.log
```

**RTX Phase B ceiling: TBD seeds/card**
**AMD Phase B ceiling: TBD seeds/card**
**ROCm multi-worker tax: TBD% (Phase A - Phase B gap)**

---

### Phase C — Stability Gate
```bash
# Edit probe_phase_C_stability.sh: set RTX_CAP and AMD_CAP (Phase B × 0.85)
bash scripts/probe_phase_C_stability.sh 2>&1 | tee /tmp/probe_C.log
```

**Result: TBD (PASS / FAIL)**

---

### Apply Caps
```bash
python3 scripts/apply_caps.py \
    --rtx-phaseA-sps <TBD> \
    --amd-phaseA-sps <TBD> \
    --rtx-phaseB-ceiling <TBD> \
    --amd-phaseB-ceiling <TBD> \
    --safety-factor 0.85
```

---

### Commit Both Files
```bash
cd ~/distributed_prng_analysis
git diff coordinator.py gpu_optimizer.py  # review changes
git add coordinator.py gpu_optimizer.py
git commit -m "feat(S128): update GPU seed caps from throughput ceiling probe"
git push origin main

cd ~/prng_cluster_public  # or wherever public repo is checked out
# cherry-pick or copy same changes
git add coordinator.py gpu_optimizer.py
git commit -m "feat(S128): update GPU seed caps from throughput ceiling probe"
git push origin main
```

---

## Key Numbers (End of S127 — Carry Into S128)
- Real draws: 18,068
- Bidirectional survivors: 85 (W8_O43)
- Best NN R²: +0.020538
- TRSE: regime=short_persistence, conf=0.828, Rule A active (window ceiling=32)
- Primary Optuna study: `window_opt_1772507547.db` (24 COMPLETE, 26 PRUNED, 1 FAIL)
- Active studies on Zeus: 1 (clean)
- Zeus GPU compute mode: DEFAULT ✅
- n_parallel=2: OPERATIONAL ✅
- Current seed caps (stale): `seed_cap_nvidia=40,000` / `seed_cap_amd=19,000`
- Current gpu_optimizer.py: RTX sps=29,000 / RX6600 sps=5,000 (2.3x understated) / scaling_factor=6.0 (wrong)
- Expected gain from cap update: ~5–6x cluster throughput

---

## Carry Forward to S129
- [ ] Execute Phase A RTX probe, record results
- [ ] Execute Phase A AMD probe, record results
- [ ] Execute Phase B concurrent probe (both GPU types)
- [ ] Execute Phase C stability test
- [ ] Run apply_caps.py with measured values
- [ ] Commit coordinator.py + gpu_optimizer.py to both repos
- [ ] Validate full WATCHER pipeline run with new caps
