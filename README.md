Complete README.md (Markdown Format)

# 🎲 Distributed PRNG Analysis & Seed Reconstruction System  
**Multi-GPU Cluster • Survivor Filtering • ML Scoring • Optuna Meta-Optimization • Reinforcement Engine**

---

## 📌 Project Overview

This repository contains a fully distributed analysis system designed to:

🧠 Reverse-engineer PRNG behavior used in lottery draw simulations  
⚙️ Brute-force and sieve candidate seeds using GPU-accelerated forward/reverse filtering  
📊 Score survivors using statistical and ML-based probability matching  
🧪 Optimize parameters using Optuna (Bayesian TPE Meta-Optimizer, Step 2.5)  
🧬 Reinforce high-confidence candidates using pattern feedback learning  
🚀 Scale across **26 GPUs** using a pull-based distributed architecture  

---

## 🔗 High-Level Pipeline Flow

```text
Stage 1: Window Optimizer (Optuna, Bayesian)
        ↓
Stage 2: Forward/Reverse Seed Sieve (Filter-based elimination)
        ↓
Stage 2.5: Distributed ML Scorer Meta-Optimizer (26 GPUs, PULL model)
        ↓
Stage 3: Survivor Fusion & Confidence Ranking
        ↓
Stage 4: Reinforcement Engine & Long-Window Validation


📄 See PROJECT_MAP.md for complete system logic and module organization.

🧠 Logical Code Structure — AI/NAV-Friendly Summary
Component	File(s)
Cluster Controller	coordinator.py, distributed_worker.py
Window Optimization	window_optimizer.py, modules/window_optimizer.py
Sieving / Filtering	sieve_filter.py
ML Scoring / Probability Matching	survivor_scorer.py, modules/mt_engine_exact.py
Reinforcement Engine	reinforcement_engine.py
Job Generation (Step 2.5)	generate_scorer_jobs.py, scorer_trial_worker.py
Central Analytics	modules/performance_analytics.py, system_monitor.py
Data Storage	modules/database_manager.py, file_manager.py
Visualization	modules/visualization_manager.py, web_visualizer.py

💡 Complete expanded map: PROJECT_MAP.md

🖥 Multi-Node Cluster Deployment
Node	GPUs	Status	Purpose
Zeus (Primary)	2× RTX 3080 Ti	CUDA	Orchestration, job generation
rig-6600	12× RX 6600	ROCm	Worker Node 1
rig-6600b	12× RX 6600	ROCm	Worker Node 2
rig-6600xt (planned)	RX 6600 XT	ROCm	Worker Node 3

ROCm Activation (AMD rigs):

export HSA_OVERRIDE_GFX_VERSION=10.3.0
source ~/tf/bin/activate

🔐 Git SSH Auto-Sync Enabled

Git on Zeus now pushes without password/token using SSH authentication:

ssh -T git@github.com
git remote -v
git push   # No credentials required


🚀 One-command auto-push script:

./automation/push_to_claude.sh

📁 Conceptual Folder Layout (No Physical File Moves)
distributed_prng_analysis/
│
├── modules/               # Orchestration, analytics, visualization, UI
├── docs/                  # Whitepapers, workflow guides, strategy research
├── configs/               # JSON/YAML configs, seed maps, job templates
├── scripts/               # Cluster tools, diagnostics, maintenance scripts
├── results_archive/       # Final validated results (NOT raw logs)
├── automation/            # Git sync, archiving, Claude integration
├── PROJECT_MAP.md         # Master logical navigation map
└── README.md

🚀 Execution Examples
Run full distributed cluster controller:
python3 coordinator.py --config configs/distributed_config.json

Generate distributed jobs for 26 GPUs:
python3 generate_scorer_jobs.py --study meta_opt --num-trials 120

Run scoring worker (one GPU):
python3 scorer_trial_worker.py --job-file scorer_jobs.json

📄 Key Documentation Included
File	Purpose
PROJECT_MAP.md	🌟 Logical, AI-friendly navigation map
WORKFLOW_GUIDE_v2.0.pdf	Full process: Seed → PRNG Match → ML Fusion
OPTUNA_MetaOptimizer_Design.md	Deep dive into Step 2.5
PRNG_Strategy_Whitepaper.md	PRNG theory, timestamp, skip, entropy patterns
README.md	Main onboarding and repo guide
🧭 Future Development Roadmap

🟩 Archive large backup (.bak, backup_*) files—exclude from Git
🟩 Enable automated Git pull on worker rigs
🟨 Convert modules to structured Python package (cluster_core/)
🟨 Integrate visual analysis (helix, temporal heatmaps)
🟪 Build WebUI for interactive seed visualization and filtering
🟦 Claude RAG-ready documentation linking with GitHub Codespaces

💡 Goals

✔ Reverse-engineer PRNG seed behavior using ML, windowing, timestamp inference
✔ Use multi-GPU cluster to accelerate probability elimination
✔ Support full seed tracking, persistence, and survivor evolution
✔ Provide explainable visual pattern analysis
✔ Support AI-assisted search and inference with GPU-powered optimizers

🤝 Questions, Ideas, or Improvements?

Open an issue, fork the repo, or propose modular architecture improvements.

🔬 This system is an evolving experimentation platform — targeting PRNG inference, timestamp reconstruction, and ML-enhanced pattern detection.# prng_cluster_project
PRNG analysis, GPU cluster automation, Optuna, Meta-optimizer, and ML fusion
