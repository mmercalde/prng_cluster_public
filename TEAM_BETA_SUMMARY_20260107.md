# Team Beta Summary: LLM Architecture v2.0.0

**Date:** January 7, 2026  
**Session:** LLM Model Selection & WATCHER Integration

---

## Executive Summary

Completed A/B testing of 4 LLM options and established production architecture for autonomous WATCHER agent. Created step-specific mission context templates based on updated Chapter documentation.

---

## 1. LLM Architecture Changes

### Previous (v1.0.5)
```
Dual-LLM Architecture:
├── Qwen2.5-Coder-14B (port 8080) - Orchestration
└── Qwen2.5-Math-7B (port 8081) - Math routing
```

### New (v2.0.0)
```
Primary + Backup Architecture:
├── DeepSeek-R1-14B (port 8080) - All WATCHER decisions
└── Claude Opus 4.5 (Claude Code CLI) - Deep analysis backup
```

### Why the Change

| Model | Speed | Quality | Role |
|-------|-------|---------|------|
| DeepSeek-R1-14B | 51 tok/s | PhD Candidate | Production WATCHER |
| DeepSeek-R1-32B | 27 tok/s | PhD Candidate+ | Not needed |
| Claude Opus 4.5 | 38 tok/s | PhD Professor | Backup for deep analysis |
| DeepSeek API | 29 tok/s | Truncated | Not recommended |

**Key finding:** R1-14B handles both code AND math reasoning natively - no need for separate math model.

---

## 2. A/B Test Results

### Test Methodology
- Standardized prompt with mission context + technical documentation
- 5 evaluation questions about bidirectional sieve architecture
- Identical prompts across all 4 models

### Results
```
Model              Time      Speed       Sections    Quality
─────────────────────────────────────────────────────────────
DeepSeek-R1-14B    24.1s     51 tok/s    5/5         Correct
DeepSeek-R1-32B    53.4s     27 tok/s    5/5         Correct
Claude Opus 4.5    59.7s     38 tok/s    5/5         Novel insights
DeepSeek API       142.9s    29 tok/s    5/5         Truncated
```

### Quality Assessment

**14B Local:** Correct answers, proper application of concepts  
**Claude Opus:** Found 0.00002% MOD 1000 bias unprompted (novel derivation)

**Analogy:** 14B = PhD Candidate (correct), Opus = PhD Professor (discovers new things)

---

## 3. Files Updated

### On Zeus (Code)

| File | Change |
|------|--------|
| `llm_services/llm_server_config.json` | Replaced v1.0.5 → v2.0.0 |
| `llm_services/llm_router.py` | 96% rewrite - single primary + escalation |
| `llm_services/start_llm_servers.sh` | 91% rewrite - single server |
| `PROPOSAL_LLM_Architecture_v2_0_0.md` | New architecture documentation |

### New Files (Mission Contexts)

| File | Purpose |
|------|---------|
| `agent_contexts/step1_window_optimizer.md` | Step 1 LLM prompt template |
| `agent_contexts/step2_bidirectional_sieve.md` | Step 2 LLM prompt template |
| `agent_contexts/step2_5_scorer_meta.md` | Step 2.5 LLM prompt template |
| `agent_contexts/step3_full_scoring.md` | Step 3 LLM prompt template |
| `agent_contexts/step4_adaptive_meta.md` | Step 4 LLM prompt template |
| `agent_contexts/step5_anti_overfit.md` | Step 5 LLM prompt template |
| `agent_contexts/step6_prediction.md` | Step 6 LLM prompt template |
| `agent_contexts/context_loader.py` | Python loader utility |

---

## 4. Mission Context Template Architecture

### Layered Prompt Structure
```
Layer 1: MISSION STATEMENT (constant)
Layer 2: STEP-SPECIFIC ROLE (varies)
Layer 3: MATHEMATICAL CONTEXT (step formulas)
Layer 4: CURRENT DATA (runtime injection)
Layer 5: DECISION REQUEST (JSON output)
```

### Key Design Decisions

1. **Step 4 is NOT data-aware** - Derives capacity from optimizer metrics only
2. **Step 5 is FIRST data-aware** - Consumes survivors_with_scores.json
3. **Decision thresholds from Chapters** - bidirectional_rate < 0.02 = PROCEED
4. **Feature schema hash** - FATAL if mismatch in Step 6

### Example Usage
```python
from agent_contexts.context_loader import build_prompt

step1_results = {
    "bidirectional_count": 847,
    "bidirectional_rate": 0.000017,
    "trials_completed": 50
}

prompt = build_prompt(step=1, current_data=step1_results)
response = llm_router.route(prompt)
# {"decision": "PROCEED", "confidence": 0.85, ...}
```

---

## 5. Escalation Flow

```
WATCHER Request
      │
      ▼
┌─────────────────┐
│ DeepSeek-R1-14B │
│ (51 tok/s)      │
└────────┬────────┘
         │
    Contains trigger?
    ├── NO → Return response
    │
    YES (UNCERTAIN, LOW_CONFIDENCE)
         │
         ▼
┌─────────────────┐
│ Claude Opus 4.5 │
│ (38 tok/s)      │
└────────┬────────┘
         │
    Return deep analysis
```

---

## 6. Server Commands

### Start LLM Server
```bash
./llm_services/start_llm_servers.sh
```

### Health Check
```bash
curl http://localhost:8080/health
```

### Stop Server
```bash
pkill -f 'llama-server.*port 8080'
```

---

## 7. Next Steps

1. ✅ LLM Architecture v2.0.0 deployed
2. ✅ Mission context templates created
3. ⬜ Wire templates into WATCHER agent
4. ⬜ Test end-to-end autonomous pipeline
5. ⬜ Implement --save-all-models flag (from memory TODO)

---

## 8. Git Commits This Session

```
86a71ea - Add LLM A/B test harness
07bfd79 - LLM Architecture v2.0.0: DeepSeek-R1-14B + Claude Opus backup
[pending] - Add WATCHER mission context templates
```

---

**Status:** Production ready for WATCHER integration

**Models on Zeus:**
- `models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf` (8.4GB) ← **Primary**
- `models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf` (19GB) ← Available
