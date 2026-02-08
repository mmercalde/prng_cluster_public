# Remote Worker Node Setup Checklist

**Document Version:** 1.0.0  
**Date:** February 7, 2026  
**Purpose:** Complete checklist for setting up a new remote worker node (rig-6600, rig-6600b, rig-6600c)

---

## Gap Identified

Existing documentation covers ramdisk files and core GPU scripts, but does NOT document:
- Python package directories required on remote workers
- Step-specific worker scripts
- Complete sync procedure for new nodes

This document fills that gap.

---

## 1. Prerequisites

### 1.1 ROCm Environment

```bash
# Verify ROCm venv exists and has required packages
ssh <new_rig> "source ~/rocm_env/bin/activate && python3 -c 'import torch; import numpy; import cupy; print(\"OK\")'"
```

Required packages in `rocm_env`:
- torch (ROCm version)
- numpy
- cupy-rocm
- optuna
- scikit-learn
- ijson (optional but recommended)

### 1.2 SSH Connectivity

```bash
# From Zeus, verify passwordless SSH
ssh michael@<new_rig_ip> "hostname"
```

### 1.3 Directory Structure

```bash
# Create base directory if missing
ssh <new_rig> "mkdir -p ~/distributed_prng_analysis"
```

---

## 2. Required Files & Directories

### 2.1 Python Package Directories (CRITICAL)

These directories contain Python modules with `__init__.py` files:

| Directory | Purpose | Required For |
|-----------|---------|--------------|
| `utils/` | Utility modules (survivor_loader.py) | Steps 2, 3, 5 |
| `models/` | ML model wrappers and factories | Steps 3, 5, 6 |
| `models/wrappers/` | XGBoost, LightGBM, CatBoost, Neural Net | Step 5 |
| `schemas/` | JSON schemas for results | Steps 3, 5 |
| `modules/` | Analysis modules | Various |

### 2.2 Core Worker Scripts

| File | Purpose | Required For |
|------|---------|--------------|
| `distributed_worker.py` | Base GPU worker | All steps |
| `scorer_trial_worker.py` | Scorer optimization trials | Step 2.5 |
| `full_scoring_worker.py` | Feature extraction | Step 3 |
| `anti_overfit_trial_worker.py` | ML training trials | Step 5 |
| `survivor_scorer.py` | 50-feature scoring engine | Step 3 |
| `reinforcement_engine.py` | Neural network model | Steps 5, 6 |

### 2.3 GPU/PRNG Core Files

| File | Purpose | Required For |
|------|---------|--------------|
| `sieve_filter.py` | GPU residue sieve | Step 1, 2 |
| `prng_registry.py` | 46 PRNG implementations | All sieve steps |
| `enhanced_gpu_model_id.py` | GPU analysis engine | Various |
| `hybrid_strategy.py` | Variable skip patterns | Step 1 |

### 2.4 Configuration Files

| File | Purpose | Required For |
|------|---------|--------------|
| `distributed_config.json` | Cluster topology | All steps |
| `ml_coordinator_config.json` | ML coordinator settings | Steps 2.5, 3, 5 |

---

## 3. Sync Procedure for New Node

### 3.1 Full Sync from Working Rig (Recommended)

```bash
# From Zeus - sync from rig-6600 to new rig
# Exclude large/unnecessary files
ssh michael@192.168.3.120 "rsync -avz \
    --exclude='*.log' \
    --exclude='*.db' \
    --exclude='__pycache__' \
    --exclude='scorer_trial_results/' \
    --exclude='results/' \
    --exclude='*.gguf' \
    --exclude='*.pth' \
    --exclude='backups/' \
    --exclude='archives/' \
    ~/distributed_prng_analysis/*.py \
    ~/distributed_prng_analysis/utils/ \
    ~/distributed_prng_analysis/models/ \
    ~/distributed_prng_analysis/modules/ \
    ~/distributed_prng_analysis/schemas/ \
    ~/distributed_prng_analysis/*.json \
    michael@<NEW_RIG_IP>:~/distributed_prng_analysis/"
```

### 3.2 Minimal Sync (Step-Specific)

#### For Step 2.5 (Scorer Meta-Optimizer):
```bash
scp scorer_trial_worker.py survivor_scorer.py reinforcement_engine.py <rig>:~/distributed_prng_analysis/
scp -r utils/ models/ <rig>:~/distributed_prng_analysis/
```

#### For Step 3 (Full Scoring):
```bash
scp full_scoring_worker.py survivor_scorer.py <rig>:~/distributed_prng_analysis/
scp -r utils/ models/ schemas/ <rig>:~/distributed_prng_analysis/
```

#### For Step 5 (Anti-Overfit):
```bash
scp anti_overfit_trial_worker.py reinforcement_engine.py <rig>:~/distributed_prng_analysis/
scp -r utils/ models/ <rig>:~/distributed_prng_analysis/
```

---

## 4. Verification Commands

### 4.1 Test Each Worker Script

```bash
# Test scorer_trial_worker (Step 2.5)
ssh <rig> "source ~/rocm_env/bin/activate && cd ~/distributed_prng_analysis && python3 scorer_trial_worker.py --help"

# Test full_scoring_worker (Step 3)
ssh <rig> "source ~/rocm_env/bin/activate && cd ~/distributed_prng_analysis && python3 full_scoring_worker.py --help"

# Test anti_overfit_trial_worker (Step 5)
ssh <rig> "source ~/rocm_env/bin/activate && cd ~/distributed_prng_analysis && python3 anti_overfit_trial_worker.py --help"
```

### 4.2 Test GPU Access

```bash
ssh <rig> "source ~/rocm_env/bin/activate && python3 -c 'import torch; print(f\"GPUs: {torch.cuda.device_count()}\")'"
# Expected: GPUs: 8
```

### 4.3 Verify Directory Structure

```bash
ssh <rig> "ls -d ~/distributed_prng_analysis/{utils,models,schemas,modules} 2>/dev/null | wc -l"
# Expected: 4
```

---

## 5. Common Errors and Fixes

### Error: `ModuleNotFoundError: No module named 'utils'`
**Cause:** `utils/` directory missing  
**Fix:** `scp -r utils/ <rig>:~/distributed_prng_analysis/`

### Error: `ModuleNotFoundError: No module named 'models.global_state_tracker'`
**Cause:** `models/` exists but Python files missing (only model checkpoints)  
**Fix:** Sync Python files: `scp models/*.py models/__init__.py <rig>:~/distributed_prng_analysis/models/`

### Error: `No module named 'numpy'` (via SSH)
**Cause:** Non-interactive SSH doesn't activate venv  
**Fix:** Ensure `distributed_config.json` has correct `python_env` path

### Error: Jobs fail in 0.4s with no output
**Cause:** Import error in worker script  
**Fix:** Test script directly with `--help` to see traceback

---

## 6. Node-Specific Configuration

### 6.1 distributed_config.json Entry

Each rig needs an entry in `distributed_config.json`:

```json
{
    "hostname": "192.168.3.162",
    "username": "michael",
    "gpu_count": 8,
    "gpu_type": "RX 6600",
    "script_path": "/home/michael/distributed_prng_analysis",
    "python_env": "/home/michael/rocm_env/bin/python",
    "max_concurrent_script_jobs": 8
}
```

### 6.2 ROCm Hostname Check

Ensure the new rig's hostname is in the ROCm prelude list in GPU scripts:

```python
if HOST in ["rig-6600", "rig-6600b", "rig-6600c"]:  # Add new hostname here
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
```

---

## 7. Quick Setup Script

Save this as `setup_new_rig.sh` on Zeus:

```bash
#!/bin/bash
# Usage: ./setup_new_rig.sh <rig_ip>

RIG_IP=$1
SOURCE_RIG="192.168.3.120"  # rig-6600 as template

if [ -z "$RIG_IP" ]; then
    echo "Usage: $0 <new_rig_ip>"
    exit 1
fi

echo "Syncing from $SOURCE_RIG to $RIG_IP..."

ssh michael@$SOURCE_RIG "rsync -avz \
    --exclude='*.log' \
    --exclude='*.db' \
    --exclude='__pycache__' \
    --exclude='scorer_trial_results/' \
    --exclude='results/' \
    --exclude='*.gguf' \
    --exclude='backups/' \
    ~/distributed_prng_analysis/*.py \
    ~/distributed_prng_analysis/utils/ \
    ~/distributed_prng_analysis/models/ \
    ~/distributed_prng_analysis/modules/ \
    ~/distributed_prng_analysis/schemas/ \
    ~/distributed_prng_analysis/*.json \
    michael@$RIG_IP:~/distributed_prng_analysis/"

echo "Verifying..."
ssh michael@$RIG_IP "source ~/rocm_env/bin/activate && cd ~/distributed_prng_analysis && python3 scorer_trial_worker.py --help | head -5"

echo "Done!"
```

---

## 8. Integration with Existing Docs

This document should be added as:
- **Chapter 9 Addendum** in `CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md`
- Or as a standalone file in `/docs/REMOTE_NODE_SETUP_CHECKLIST.md`

Reference from:
- `Cluster_operating_manual.txt` - Hardware Architecture section
- `COMPLETE_OPERATING_GUIDE_v1_1.md` - Part 4: Distributed Workers

---

*Document created after rig-6600c setup issues discovered February 7, 2026.*
