# 🗂 Distributed PRNG Analysis — Logical Project Map  
📌 *AI-Friendly, Developer-Friendly — No Files Actually Moved*

> This structure explains **how your system is organized logically**,  
> without physically changing file locations on disk.  
> It is designed for Claude, ChatGPT, GitHub navigation, and future refactoring.

──────────────────────────────────────────────
🚀 CORE PIPELINE — MAIN EXECUTION FLOW
──────────────────────────────────────────────
These are the foundation of your multi-node PRNG analysis system.

coordinator.py
│   • Master controller for distributed execution
│   • Handles SSH, GPU job scheduling, ROCm/CUDA activation
│   • Collects outputs, retries failed jobs, orchestrates execution flow

distributed_worker.py
│   • Runs jobs on individual GPUs (remote or local)
│   • Loads survivors, scoring configs, window settings
│   • Writes local job results (JSON)

window_optimizer.py
│   • Optuna-based Bayesian PRNG window optimization
│   • Handles variable skip, timestamp variants, and threshold tuning

sieve_filter.py
│   • Forward/Reverse filtering of impossible seed candidates
│   • Implements survivor elimination logic, skip-based sieving

survivor_scorer.py
│   • Machine learning and statistical scoring of survivor seeds
│   • Probability matching, weighted scoring, pattern alignment

reinforcement_engine.py
│   • Feedback-based survivor improvement for long-window runs
│   • Reinforcement scoring, pattern convergence tracking

unified_system_working.py
│   • Full pipeline execution for v1.6+ integrated system
│   • Combines optimization, sieving, scoring, and ML steps

──────────────────────────────────────────────
🧪 META-OPTIMIZER & JOB GENERATION (Step 2.5 — 26 GPU Distributed Mode)
──────────────────────────────────────────────

generate_scorer_jobs.py
│   • Creates distributed job specs across all rigs/GPUs
│   • Config sampling, job splitting, JSON job distribution

scorer_trial_worker.py
│   • Executes a single trial (one GPU → one parameter config)
│   • Saves trial JSON results locally before collection

collect_scorer_results.py (if present)
│   • Pull-based collection of distributed trial outputs

──────────────────────────────────────────────
📁 MODULES — HIGH-LEVEL ORCHESTRATION, RESEARCH, ANALYTICS & UI
──────────────────────────────────────────────

modules/
│
├── mt_pipeline.py
│   • High-level orchestrator: connects engine, analysis, and scoring
│   • Likely main entry point for end-to-end workflow execution

├── mt_engine_exact.py
│   • PRNG engine logic for exact seed reconstruction & scoring
│
├── direct_analysis.py
│   • Lightweight / local-only execution path without cluster

├── advanced_research.py
│   • Experimental scripting: timestamp hypothesis, pattern isolation

├── database_manager.py
│   • Manages persistence storage/logging of survivors, runs, ML output

├── file_manager.py
│   • Central utility for safe read/write of JSON, CSV, configs

├── performance_analytics.py
│   • Tracks GPU usage, run efficiency, time-per-trial, job throughput

├── system_monitor.py
│   • Monitors live GPU temps, worker status, resource utilization

├── result_viewer.py
│   • CLI/GUI interface for visualizing survivors, seed matches

├── visualization_manager.py
│   • Generates 2D/3D visual views (helix, heat maps, anomaly plots)

├── web_visualizer.py
│   • Web-based interface for visualizing and browsing results

├── window_optimizer.py
│   • Secondary / experimental window optimizer (legacy or test)

└── **Legacy/Backup Files — Suggested Archiving**
    ├─ *_backup_2025*.py
    ├─ *.bak
    ├─ *.backup_before_new_format
    └─ Safe to move to /archive or exclude via .gitignore

──────────────────────────────────────────────
⚙️ CONFIGURATION (.json / .yaml)
──────────────────────────────────────────────

distributed_config.json     ← Node IPs, GPU mappings, SSH runtime config  
agent_config.yaml           ← Meta-optimizer parameters  
optimal_window_config.json  ← Best window sizes after Optuna selection  
device_mapping_*.json       ← GPU layouts per rig  
survivor_job_template.json  ← Used for building job specs  
ml_config.yaml              ← ML fusion strategy configuration  

──────────────────────────────────────────────
🛠 DIAGNOSTIC / MAINTENANCE TOOLS
──────────────────────────────────────────────

gpu_diag.py                 ← Confirms GPU visibility and ROCm/CUDA status  
restart_cluster.sh          ← Safely restarts all worker nodes  
watch_jobs.sh               ← Live status monitoring of active GPU jobs  
merge_results.py            ← Merges JSON run outputs for global scoring  
fix_incomplete_results.py   ← Optional script for job result recovery  

──────────────────────────────────────────────
🤖 AUTOMATION / SYNC / GITHUB / AI INTEGRATION
──────────────────────────────────────────────

push_to_claude.sh           ← One-command commit + push script  
prepare_archive.sh          ← Compress/organize result files  
sync_status.log             ← Automatic log of sync/push events  

──────────────────────────────────────────────
📘 DOCUMENTATION, WHITEPAPERS & RESEARCH NOTES
──────────────────────────────────────────────

docs/
│   WHITEPAPER_v1.5.pdf  
│   WORKFLOW_GUIDE_v2.0.pdf  
│   PRNG_Strategy_Whitepaper.md  
│   OPTUNA_MetaOptimizer_Design.md  
│   README_Structure_Overview.md  

──────────────────────────────────────────────
📊 RESULTS (Large Data — Suggest Move to /mnt/data or Archive)
──────────────────────────────────────────────

results/
│   final_results.json  
│   bidirectional_survivors.json  
│   meta_optimizer_results_*.json  
│   verification_report_*.json  

🔎 Suggested: Move bulk results to `/mnt/data/archive/` (excluded from Git)

──────────────────────────────────────────────
🌐 MULTI-NODE CLUSTER INFO
──────────────────────────────────────────────

Known Hosts:
  • Zeus (primary):       192.168.3.127  
  • rig-6600:             192.168.3.120  
  • rig-6600b:            192.168.3.154  
  

ROCm Activation:
  export HSA_OVERRIDE_GFX_VERSION=10.3.0

Virtual Environment:
  source ~/tf/bin/activate  

──────────────────────────────────────────────
📌 NOTES & FUTURE REFACTOR PLAN
──────────────────────────────────────────────
• This map is **logical, not physical** (no files moved)
• Ideal for:
   🧠 Claude AI comprehension
   📝 Auto-README documentation generation
   📦 Future modular refactor (cluster_core/, utils/, ui/, analytics/)
• Backup and .bak files should eventually be archived externally

