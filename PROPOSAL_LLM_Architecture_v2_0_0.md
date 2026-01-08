# PROPOSAL: LLM Architecture v2.0.0

## DeepSeek-R1 Primary + Claude Opus Backup

**Version:** 2.0.0  
**Date:** January 7, 2026  
**Status:** Validated via A/B Testing  
**Replaces:** PROPOSAL_Schema_v1_0_4_Dual_LLM_Architecture.md

---

## Executive Summary

This proposal replaces the dual-LLM architecture (Qwen Coder + Math) with a single primary reasoning model (DeepSeek-R1-14B) backed by Claude Opus 4.5 for escalation.

### Key Changes

| Aspect | v1.0.4 (Old) | v2.0.0 (New) |
|--------|--------------|--------------|
| **Primary** | Qwen2.5-Coder-14B | DeepSeek-R1-14B |
| **Secondary** | Qwen2.5-Math-7B | None (R1 handles both) |
| **Backup** | None | Claude Opus 4.5 |
| **Ports** | 8080 + 8081 | 8080 only |
| **Routing** | Keyword-based math routing | Escalation-based |
| **Speed** | ~25 tok/s | 51 tok/s (2x faster) |

### Why This Change?

1. **R1 handles reasoning natively** - No need for separate math model
2. **2x faster** - 51 tok/s vs 25 tok/s
3. **Simpler architecture** - One model, one port
4. **Better backup** - Claude Opus for deep analysis when needed

---

## Part 1: Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│                    WATCHER AGENT                           │
│                                                            │
│    ┌──────────────────────────────────────────────────┐   │
│    │           LLM Router v2.0.0                      │   │
│    │                                                  │   │
│    │   ┌─────────────────┐    ┌──────────────────┐   │   │
│    │   │    PRIMARY      │    │     BACKUP       │   │   │
│    │   │                 │    │                  │   │   │
│    │   │ DeepSeek-R1-14B │───►│ Claude Opus 4.5  │   │   │
│    │   │    (Local)      │    │  (Claude Code)   │   │   │
│    │   │                 │    │                  │   │   │
│    │   │  Port: 8080     │    │  CLI invocation  │   │   │
│    │   │  51 tok/s       │    │  38 tok/s        │   │   │
│    │   │  FREE           │    │  FREE (Max sub)  │   │   │
│    │   └─────────────────┘    └──────────────────┘   │   │
│    │          │                        ▲              │   │
│    │          │    ESCALATION          │              │   │
│    │          │    TRIGGERS:           │              │   │
│    │          │    - UNCERTAIN         │              │   │
│    │          │    - LOW_CONFIDENCE    │              │   │
│    │          │    - ESCALATE_TO_BACKUP│              │   │
│    │          └────────────────────────┘              │   │
│    └──────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

---

## Part 2: Model Specifications

### 2.1 Primary: DeepSeek-R1-Distill-Qwen-14B

| Property | Value |
|----------|-------|
| **Model File** | `DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf` |
| **Size** | 8.4 GB |
| **Parameters** | 14.77B |
| **Quantization** | Q4_K_M |
| **Context Length** | 8192 (expandable to 16K) |
| **Port** | 8080 |
| **Backend** | Vulkan (dual GPU auto-split) |

**Benchmark Results (Zeus - 2× RTX 3080 Ti):**

| Metric | Value |
|--------|-------|
| Prompt Processing | 1,472 tok/s |
| Text Generation | 63.7 tok/s |
| Effective Speed | 51.4 tok/s |
| Sections Addressed | 5/5 ✅ |

**Capabilities:**
- Reasoning / Chain-of-thought
- Code generation
- JSON manipulation
- Statistical analysis
- Threshold evaluation
- Signal quality assessment
- Experiment design

### 2.2 Backup: Claude Opus 4.5

| Property | Value |
|----------|-------|
| **Access Method** | Claude Code CLI |
| **Command** | `claude --print -p "<prompt>"` |
| **Cost** | Free (Claude Max subscription) |
| **Speed** | ~38 tok/s |

**Benchmark Results:**

| Metric | Value |
|--------|-------|
| Time for Test | 59.7s |
| Sections Addressed | 5/5 ✅ |
| Novel Derivations | Yes (bias calculations) |
| Depth | PhD Professor level |

**Use Cases:**
- Deep analysis when primary is uncertain
- Edge case discovery
- Novel derivations (found 0.00002% MOD bias)
- Comprehensive debugging

### 2.3 Model Comparison (A/B Test Results)

| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| DeepSeek-R1-14B (Local) | 51 tok/s | PhD Candidate | Production WATCHER |
| DeepSeek-R1-32B (Local) | 27 tok/s | PhD Candidate+ | Not needed |
| Claude Opus 4.5 | 38 tok/s | PhD Professor | Deep analysis backup |
| DeepSeek API | 29 tok/s | Truncated | Not recommended |

---

## Part 3: Configuration

### 3.1 llm_server_config.json

```json
{
    "primary": {
        "model": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
        "model_path": "models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
        "port": 8080,
        "context_length": 8192,
        "max_tokens": 4096,
        "n_gpu_layers": 99,
        "backend": "vulkan",
        "expected_speed_tps": 51,
        "capabilities": [
            "reasoning", "code_generation", "json_manipulation",
            "statistical_analysis", "threshold_evaluation"
        ]
    },
    "backup": {
        "provider": "claude_code",
        "model": "claude-opus-4.5",
        "invocation": "claude --print -p",
        "expected_speed_tps": 38,
        "use_when": [
            "primary returns uncertain/low_confidence",
            "complex debugging required"
        ]
    },
    "routing": {
        "default": "primary",
        "escalation_triggers": [
            "UNCERTAIN", "LOW_CONFIDENCE",
            "ESCALATE_TO_BACKUP", "REQUIRES_DEEP_ANALYSIS"
        ]
    }
}
```

### 3.2 Server Startup

```bash
# Single command - start primary
~/llama.cpp/llama-server \
    --model ~/distributed_prng_analysis/models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf \
    --port 8080 \
    --ctx-size 8192 \
    --n-gpu-layers 99

# Or use the startup script
./llm_services/start_llm_servers.sh
```

### 3.3 LLM Router Usage

```python
from llm_services.llm_router import get_router

router = get_router()

# Normal request - goes to primary
response = router.route("Analyze this bidirectional_rate: 0.16")

# Force backup for deep analysis
response = router.route("Debug this edge case...", force_backup=True)

# Check health
status = router.health_check()
# {"primary": True, "backup": True}
```

---

## Part 4: Escalation Flow

```
WATCHER Agent Request
        │
        ▼
┌───────────────────────┐
│  DeepSeek-R1-14B      │
│  (Primary)            │
│                       │
│  51 tok/s, local      │
└───────────┬───────────┘
            │
            ▼
    ┌───────────────┐
    │ Response      │
    │ contains      │───── NO ────► Return response
    │ escalation    │
    │ trigger?      │
    └───────┬───────┘
            │
           YES
            │
            ▼
┌───────────────────────┐
│  Claude Opus 4.5      │
│  (Backup)             │
│                       │
│  38 tok/s, Claude Code│
└───────────┬───────────┘
            │
            ▼
    Return deep analysis
```

**Escalation Triggers:**
- `UNCERTAIN`
- `LOW_CONFIDENCE`
- `ESCALATE_TO_BACKUP`
- `REQUIRES_DEEP_ANALYSIS`

---

## Part 5: Migration from v1.0.4

### 5.1 Files to Update

| File | Action |
|------|--------|
| `llm_server_config.json` | Replace with v2.0.0 |
| `llm_router.py` | Replace with v2.0.0 |
| `start_llm_servers.sh` | Replace with v2.0.0 |
| Agent contexts | Update LLM references |

### 5.2 Removed Components

- `Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf` - No longer needed
- `Qwen2.5-Math-7B-Instruct-Q5_K_M.gguf` - No longer needed
- Port 8081 - No longer used
- Math keyword routing - Replaced by escalation

### 5.3 New Model Download

```bash
cd ~/distributed_prng_analysis/models

huggingface-cli download bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF \
    DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf \
    --local-dir .
```

---

## Part 6: Validation

### 6.1 A/B Test Methodology

1. Created standardized prompt with mission context + technical documentation
2. Asked 5 evaluation questions about sieve architecture
3. Ran identical prompts on all 4 models
4. Measured time, tokens, section coverage, quality

### 6.2 Results Summary

```
============================================================
FULL COMPARISON: Local vs API Models
============================================================
Metric                   14b            32b            claude_api     deepseek_api   
-------------------------------------------------------------------------------------
elapsed_seconds          24.09          53.36          59.66          142.87         
tokens_predicted         1238           1450           2263           4096           
tokens_per_second        51.39          27.17          37.93          28.67          
-------------------------------------------------------------------------------------
sections_addressed       5              5              5              5              
total_length             6144           7045           9052           1774           
============================================================
Section Coverage:
-------------------------------------------------------------------------------------
state_variables          ✅              ✅              ✅              ✅              
information_loss         ✅              ✅              ✅              ✅              
leverage_params          ✅              ✅              ✅              ✅              
experiment_plan          ✅              ✅              ✅              ✅              
assumptions              ✅              ✅              ✅              ✅
```

### 6.3 Quality Assessment

| Model | Assessment |
|-------|------------|
| **14B Local** | PhD Candidate - Correct answers, applies knowledge properly |
| **32B Local** | Senior PhD Candidate - Same correctness, slightly more thorough |
| **Claude Opus** | PhD Professor - Derives novel insights (bias calculation) |
| **DeepSeek API** | Truncated - Not recommended |

---

## Part 7: Operational Notes

### 7.1 Server Management

```bash
# Start
./llm_services/start_llm_servers.sh

# Health check
curl http://localhost:8080/health

# Stop
pkill -f 'llama-server.*port 8080'

# View logs
tail -f logs/llm/primary.log
```

### 7.2 Resource Usage

| Resource | Primary | Backup |
|----------|---------|--------|
| VRAM | ~8.5 GB (split across 2 GPUs) | N/A |
| RAM | ~2 GB | ~500 MB (CLI) |
| Network | None | Claude API |

### 7.3 Troubleshooting

| Issue | Solution |
|-------|----------|
| Primary not responding | Check `logs/llm/primary.log`, restart server |
| Backup failing | Verify `claude login` completed, check `~/claude_test` exists |
| Slow responses | Check GPU utilization with `nvidia-smi` |

---

## Appendix A: Chat Template

DeepSeek-R1-Distill uses ChatML format:

```
<|im_start|>system
You are an expert AI assistant for PRNG analysis.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

Stop tokens: `</s>`, `<|im_end|>`, `<|endoftext|>`

---

## Appendix B: Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.4 | 2025-12-01 | Dual-LLM (Qwen Coder + Math) |
| 2.0.0 | 2026-01-07 | DeepSeek-R1 primary + Claude Opus backup |

---

**END OF PROPOSAL**
