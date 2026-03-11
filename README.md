# Distributed PRNG Analysis System
**Updated:** S135 (2026-03-10)

A distributed system for reverse-engineering PRNG behavioral patterns via bidirectional
sieve validation, ML ensemble scoring, and autonomous agent orchestration.
Modular and configurable -- not tied to any specific PRNG type or data source.

## Pipeline (Steps 0->6)

| Step | Module | Description |
|------|--------|-------------|
| Step 0 | trse_step0.py | Temporal Regime Segmentation -- skip_on_fail |
| Step 1 | window_optimizer.py | Bayesian TPE window search, 200 trials |
| Step 2 | scorer meta-optimizer | Distributed scoring config optimization |
| Step 3 | survivor_scorer.py | 91-feature extraction per survivor |
| Step 4 | anti-overfit training | PRNG-agnostic feature training |
| Step 5 | meta_prediction_optimizer_anti_overfit.py | 4-model ML comparison |
| Step 6 | prediction_generator.py | Final prediction pool generation |

## Cluster Configuration

| Node | GPUs | Backend | IP |
|------|------|---------|-----|
| Zeus (primary) | 2x RTX 3080 Ti | CUDA | localhost |
| rig-6600 | 8x RX 6600 | ROCm | 192.168.3.120 |
| rig-6600b | 8x RX 6600 | ROCm | 192.168.3.154 |
| rig-6600c | 8x RX 6600 | ROCm | 192.168.3.162 |

**Total: 26 GPUs**
Note: rig-6600c has i5-8400T CPU (~50% throughput deficit vs other rigs)

## Key Metrics

- Aggregate throughput: 2,082,140 sps (persistent workers active, S130)
- Best survivor run: 17,247 bidirectional survivors
- Real data: 18,068 draws (CA Daily 3, 2000-2026)
- Primary Optuna study: window_opt_1772507547.db (resume target)
- Best ML model: CatBoost, R2~+0.005-+0.010 (holdout_quality target)

## WATCHER CLI

```bash
# Full 0->6 pipeline
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline \
  --start-step 0 --end-step 6 \
  --params '{"lottery_file":"daily3.json","trials":200,
             "resume_study":true,"study_name":"window_opt_1772507547"}'

# With persistent workers (direct window_optimizer)
PYTHONPATH=. python3 window_optimizer.py --lottery-file daily3.json \
  --strategy bayesian --max-seeds 5000000 --prng-type java_lcg \
  --trials 200 --use-persistent-workers --worker-pool-size 4 \
  --resume-study --study-name window_opt_1772507547
```

## Environment Setup

```bash
# Zeus (CUDA)
source ~/venvs/torch/bin/activate
cd ~/distributed_prng_analysis

# AMD rigs (ROCm)
export HSA_OVERRIDE_GFX_VERSION=10.3.0
source ~/rocm_env/bin/activate
```

## Repository

- Private: git@github.com:mmercalde/prng_cluster_project.git  (remote: origin)
- Public:  git@github.com:mmercalde/prng_cluster_public.git   (remote: public)
- Push:    git push origin main && git push public main

## Key Documentation (docs/)

| File | Contents |
|------|----------|
| COMPLETE_OPERATING_GUIDE_v2_0.md | Full system reference (v2.1.0) |
| CHAPTER_1_WINDOW_OPTIMIZER.md | Step 1, persistent worker call chain |
| CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md | GPU specs, seed caps, throughput |
| CHAPTER_10_AUTONOMOUS_AGENT_FRAMEWORK_v3.md | WATCHER, TRSE, autonomy |
| CHAPTER_12_WATCHER_AGENT.md | CLI params S114-S131, manifest v1.5.0 |
| CHAPTER_14_TRAINING_DIAGNOSTICS.md | ML targets, feature signal |
