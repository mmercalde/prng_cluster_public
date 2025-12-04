# 🎲 Distributed PRNG Analysis & Seed Reconstruction System

**Multi-GPU Cluster • AI Agent Architecture • ML Scoring • Optuna Meta-Optimization • Reinforcement Engine**

---

## 📌 Project Overview

A fully distributed, AI-driven analysis system designed to:

- 🧠 Reverse-engineer PRNG behavior through functional mimicry
- ⚙️ Brute-force and sieve candidate seeds using GPU-accelerated forward/reverse filtering
- 📊 Score survivors using statistical and ML-based probability matching
- 🧪 Optimize parameters using Optuna (Bayesian TPE Meta-Optimizer)
- 🧬 Reinforce high-confidence candidates using pattern feedback learning
- 🤖 Automate pipeline execution via AI Agent Architecture
- 🚀 Scale across **26 GPUs** using a pull-based distributed architecture

---

## 🔗 6-Step Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Step 1          Step 2.5        Step 3         Step 4         Step 5       │
│  Window ───────► Scorer ───────► Full ────────► ML Meta ─────► Anti-        │
│  Optimizer       Meta-Opt        Scoring        Optimizer      Overfit      │
│                                                                    │        │
│  Bayesian        Distributed     26-GPU         Adaptive          │        │
│  TPE             Optuna          Scoring        Architecture      │        │
│                                                                    ▼        │
│                                                              Step 6         │
│                                                              Prediction     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Step | Name | Script | Output |
|------|------|--------|--------|
| 1 | Window Optimizer | `window_optimizer.py` | `bidirectional_survivors.json` |
| 2.5 | Scorer Meta-Optimizer | `generate_scorer_jobs.py` | `optimal_scorer_config.json` |
| 3 | Full Scoring | `generate_full_scoring_jobs.py` | `survivors_with_scores.json` |
| 4 | ML Meta-Optimizer | `adaptive_meta_optimizer.py` | `reinforcement_engine_config.json` |
| 5 | Anti-Overfit Training | `meta_prediction_optimizer_anti_overfit.py` | `best_model.pth` |
| 6 | Prediction | `reinforcement_engine.py` | `prediction_pool.json` |

📄 See `PROJECT_MAP.md` for complete system logic and module organization.

---

## 🤖 AI Agent Architecture

The system includes a complete AI agent framework for autonomous pipeline execution:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ZEUS DUAL-LLM LAYER                                      │
│  ┌─────────────────────────┐    ┌─────────────────────────┐                 │
│  │   GPU0: ORCHESTRATOR    │    │   GPU1: MATH SPECIALIST │                 │
│  │   Qwen2.5-Coder-14B     │    │   Qwen2.5-Math-7B       │                 │
│  │   Port: 8080            │    │   Port: 8081            │                 │
│  └───────────┬─────────────┘    └───────────┬─────────────┘                 │
│              └──────────────┬───────────────┘                                │
│                             ▼                                                │
│                  ┌─────────────────────┐                                     │
│                  │    LLM Router       │                                     │
│                  └──────────┬──────────┘                                     │
└─────────────────────────────┼────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AI AGENT LAYER                                        │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │ WindowOptAgent │  │ ScorerMetaAgent│  │ PredictionAgent│  ...            │
│  └────────────────┘  └────────────────┘  └────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Agent Components

| Directory | Purpose |
|-----------|---------|
| `agents/` | BaseAgent class, agent implementations |
| `agent_manifests/` | JSON configuration for each pipeline step |
| `llm_services/` | Dual-LLM router and server management |

### 6 Pipeline Agents

| Agent | Step | Manifest |
|-------|------|----------|
| WindowOptimizerAgent | 1 | `window_optimizer.json` |
| ScorerMetaAgent | 2.5 | `scorer_meta.json` |
| FullScoringAgent | 3 | `full_scoring.json` |
| MLMetaAgent | 4 | `ml_meta.json` |
| ReinforcementAgent | 5 | `reinforcement.json` |
| PredictionAgent | 6 | `prediction.json` |

📄 See `docs/proposals/` for complete architecture documentation.

---

## 🖥 Multi-Node Cluster

| Node | GPUs | Type | Purpose |
|------|------|------|---------|
| Zeus (Primary) | 2× RTX 3080 Ti | CUDA | Orchestration, LLM hosting, job generation |
| rig-6600 | 12× RX 6600 | ROCm | Worker Node 1 |
| rig-6600b | 12× RX 6600 | ROCm | Worker Node 2 |
| **Total** | **26 GPUs** | | **~285 TFLOPS** |

### ROCm Activation (AMD rigs)

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
source ~/rocm_env/bin/activate
```

---

## 🧬 PRNG Support

**44 PRNG Algorithms** across 11 families with 4 variants each:

| Family | Base | Hybrid | Reverse | Hybrid+Reverse |
|--------|------|--------|---------|----------------|
| java_lcg | ✅ | ✅ | ✅ | ✅ |
| mt19937 | ✅ | ✅ | ✅ | ✅ |
| xorshift32 | ✅ | ✅ | ✅ | ✅ |
| xorshift64 | ✅ | ✅ | ✅ | ✅ |
| xorshift128 | ✅ | ✅ | ✅ | ✅ |
| pcg32 | ✅ | ✅ | ✅ | ✅ |
| lcg32 | ✅ | ✅ | ✅ | ✅ |
| minstd | ✅ | ✅ | ✅ | ✅ |
| xoshiro256pp | ✅ | ✅ | ✅ | ✅ |
| philox4x32 | ✅ | ✅ | ✅ | ✅ |
| sfc64 | ✅ | ✅ | ✅ | ✅ |

All kernels in `prng_registry.py` (~174KB, 4000+ lines).

---

## 📁 Project Structure

```
distributed_prng_analysis/
│
├── agents/                    # AI Agent implementations
│   ├── agent_core.py          # BaseAgent class
│   └── __init__.py
│
├── agent_manifests/           # JSON configs for 6 pipeline agents
│   ├── window_optimizer.json
│   ├── scorer_meta.json
│   ├── full_scoring.json
│   ├── ml_meta.json
│   ├── reinforcement.json
│   └── prediction.json
│
├── llm_services/              # Dual-LLM infrastructure
│   ├── llm_router.py
│   ├── llm_server_config.json
│   └── start_llm_servers.sh
│
├── core/                      # Results management
│   └── results_manager.py
│
├── integration/               # Adapters and bridges
│   ├── metadata_writer.py
│   └── sieve_integration.py
│
├── schemas/                   # Data schemas (v1.0.4)
│   ├── results_schema_v1.json
│   └── output_templates.json
│
├── modules/                   # Analytics, visualization, UI
├── docs/                      # Proposals, whitepapers
├── optuna_studies/            # Persistent Optuna DBs
├── results/                   # Output files
│
├── coordinator.py             # 26-GPU distributed controller
├── distributed_worker.py      # GPU worker script
├── prng_registry.py           # 44 PRNG kernels
├── reinforcement_engine.py    # ML training engine
├── window_optimizer.py        # Step 1
├── generate_scorer_jobs.py    # Step 2.5
│
├── PROJECT_MAP.md             # System navigation
└── README.md                  # This file
```

---

## 🚀 Quick Start

### Run Complete Pipeline

```bash
python3 complete_whitepaper_workflow_with_meta_optimizer.py \
    --lottery-file synthetic_lottery.json \
    --window-opt-trials 10 \
    --seed-count 10000000 \
    --scorer-trials 20 \
    --anti-overfit-trials 10 \
    --k-folds 5 \
    --prng-type java_lcg \
    --test-both-modes
```

### Run Individual Steps

```bash
# Step 1: Window Optimization
python3 window_optimizer.py --lottery-file data.json --trials 50

# Step 2.5: Scorer Meta-Optimization  
python3 generate_scorer_jobs.py --trials 100 --study scorer_meta

# Run coordinator for distributed execution
python3 coordinator.py --jobs-file scorer_jobs.json
```

### Start LLM Servers (for AI agents)

```bash
cd llm_services
./start_llm_servers.sh
```

---

## 📊 Progress Display

The system includes a rich terminal progress display via tmux:

```bash
# Auto-launches with workflow script
# Or manually:
tmux new-session -d -s prng
tmux split-window -h "python3 progress_monitor.py"
tmux attach -t prng
```

Shows: Progress bar, ETA, seeds/sec, per-node GPU stats.

---

## 🔐 Git SSH Auto-Sync

```bash
ssh -T git@github.com
git push   # No credentials required
```

---

## 📄 Key Documentation

| File | Purpose |
|------|---------|
| `PROJECT_MAP.md` | 🌟 Logical, AI-friendly navigation map |
| `docs/proposals/README.md` | Agent architecture proposals |
| `complete_workflow_guide_v2.md` | Full pipeline execution guide |
| `instructions.txt` | Development instructions |

---

## 🧭 Roadmap

- [x] 26-GPU distributed architecture
- [x] 44 PRNG algorithms (forward + reverse)
- [x] 6-step pipeline
- [x] Dual-LLM infrastructure
- [x] Agent manifests
- [x] Schema v1.0.4 with agent_metadata
- [ ] Watcher Agent (autonomous pipeline)
- [ ] optuna_agent_bridge.py (cross-run learning)
- [ ] WebUI for visualization

---

## 💡 Design Principles

- **Modular**: All components JSON-configurable
- **Distributed**: 26-GPU pull-based architecture
- **AI-Native**: Designed for agent automation from day one
- **Backward Compatible**: All changes are additive
- **ML-Ready**: Structured outputs for ML training

---

## 🤝 Contributing

Open an issue, fork the repo, or propose improvements.

---

*Distributed PRNG Analysis System — Functional mimicry through ML-enhanced pattern detection*
