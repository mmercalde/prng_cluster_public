# Chapter 10: Autonomous Agent Framework

**Version:** 2.0.0 (LLM Architecture Update)  
**Date:** January 7, 2026  
**Components:** Primary LLM • Claude Backup • Watcher Agent • Agent Manifests • Pydantic • GBNF  
**Status:** ~85% Autonomous  

---

## 1. Executive Summary

The Autonomous Agent Framework enables the PRNG analysis system to execute the 6-step pipeline with minimal human intervention. The architecture comprises:

| Component | Purpose |
|-----------|---------|
| **Primary LLM (DeepSeek-R1-14B)** | All WATCHER decisions at 51 tok/s |
| **Backup LLM (Claude Opus 4.5)** | Deep analysis escalation via CLI |
| **Watcher Agent** | Pipeline orchestration and decision-making |
| **Agent Manifests** | Configuration files defining each step's behavior |
| **Pydantic Models** | Type-safe data validation for agent communication |
| **GBNF Grammar** | Constrained LLM output to prevent hallucination |

**Goal:** 85%+ autonomous operation with human oversight only for edge cases.

---

## 2. LLM Architecture v2.0.0

### 2.1 Architecture Change Summary

| Aspect | v1.0.5 (Previous) | v2.0.0 (Current) |
|--------|-------------------|------------------|
| **Primary** | Qwen2.5-Coder-14B | DeepSeek-R1-14B |
| **Secondary** | Qwen2.5-Math-7B | Claude Opus 4.5 (backup) |
| **Ports** | 8080 + 8081 | 8080 only |
| **Routing** | Keyword-based | Single primary + escalation |
| **Math handling** | Separate model | R1-14B handles both |

### 2.2 Why the Change?

A/B testing on January 7, 2026 revealed:

| Model | Speed | Quality | Role |
|-------|-------|---------|------|
| DeepSeek-R1-14B | **51 tok/s** | PhD Candidate | ✅ Production WATCHER |
| DeepSeek-R1-32B | 27 tok/s | PhD Candidate+ | Available but not needed |
| Claude Opus 4.5 | 38 tok/s | **PhD Professor** | Backup for deep analysis |
| DeepSeek API | 29 tok/s | Truncated | Not recommended |

**Key finding:** DeepSeek-R1-14B handles both code AND math reasoning natively — no need for separate math model.

### 2.3 Hardware Configuration

**KEY DESIGN: ON-DEMAND LLM**

The LLM server is NOT persistent. It starts only when WATCHER needs a decision, then shuts down to free GPU resources for sieve/ML work.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ZEUS LLM CONFIGURATION v2.0.0                           │
│                        (ON-DEMAND, NOT PERSISTENT)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   WHEN LLM ACTIVE (WATCHER decision needed):                               │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │              DeepSeek-R1-14B (PARTITIONED)                  │          │
│   │         GPU0 (RTX 3080 Ti)  +  GPU1 (RTX 3080 Ti)          │          │
│   │              12GB VRAM      +      12GB VRAM                │          │
│   │                         = 24GB total                        │          │
│   │                                                             │          │
│   │   Model: ~8.4GB split across both GPUs                     │          │
│   │   KV Cache: ~6GB split across both GPUs                    │          │
│   │   Port: 8080                                                │          │
│   │   Context: 16K tokens                                       │          │
│   │   Speed: 51 tok/s                                           │          │
│   └─────────────────────────────────────────────────────────────┘          │
│                                                                             │
│   WHEN LLM INACTIVE (normal pipeline operation):                           │
│   ┌─────────────────────────┐       ┌─────────────────────────┐            │
│   │   GPU0 - AVAILABLE      │       │   GPU1 - AVAILABLE      │            │
│   │   Full 12GB VRAM        │       │   Full 12GB VRAM        │            │
│   │   Sieve jobs, ML work   │       │   Sieve jobs, ML work   │            │
│   └─────────────────────────┘       └─────────────────────────┘            │
│                                                                             │
│   BACKUP: Claude Opus 4.5 via Claude Code CLI (external API)               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 On-Demand LLM Lifecycle

```
Pipeline Step Completes
        │
        ▼
┌─────────────────┐
│ WATCHER detects │
│ output file     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Start LLM Server│◄── Both GPUs allocated
│ (takes ~15-20s) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Query LLM for   │
│ decision        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Stop LLM Server │◄── Both GPUs freed
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Execute decision│
│ (PROCEED/RETRY) │
└─────────────────┘
```

**Why On-Demand?**
- Pipeline steps take 10-45+ minutes
- LLM decisions take ~30 seconds
- Keeping LLM resident wastes 24GB VRAM for 99%+ of runtime
- On-demand frees both GPUs for actual compute work

### 2.4 Model Selection Rationale

#### Primary: DeepSeek-R1-14B

| Capability | Use Case |
|------------|----------|
| Code generation | Script parameters, JSON configs |
| Math reasoning | Residue calculations, PRNG computations |
| Planning | Pipeline step sequencing |
| Result interpretation | Analyzing trial outputs |
| JSON manipulation | Creating/modifying configs |

**Why DeepSeek-R1?**
- **Native reasoning:** Built-in chain-of-thought without prompting
- **Math + Code:** Handles both domains in single model
- **Speed:** 51 tok/s on RTX 3080 Ti (fastest tested)
- **Quality:** Correct answers on all 5 A/B test questions

#### Backup: Claude Opus 4.5

| Capability | Use Case |
|------------|----------|
| Deep analysis | Complex failure diagnosis |
| Novel insights | Discovered 0.00002% MOD 1000 bias unprompted |
| Semantic validation | Feature importance interpretation |
| Escalation handling | When R1-14B returns UNCERTAIN |

**When to escalate:**
- R1-14B returns `"decision": "UNCERTAIN"`
- R1-14B confidence < 0.50
- Complex multi-step reasoning required
- Novel pattern discovery needed

### 2.5 LLM Server Configuration

**File:** `llm_services/llm_server_config.json`

```json
{
    "schema_version": "2.0.0",
    "architecture": "on_demand_primary_plus_backup",
    
    "primary": {
        "name": "DeepSeek-R1-14B",
        "model_file": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
        "model_path": "models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
        "port": 8080,
        "gpu_partition": {
            "gpus": [0, 1],
            "tensor_split": "0.5,0.5",
            "total_vram_gb": 24,
            "description": "Model partitioned across both RTX 3080 Ti GPUs"
        },
        "context_length": 16384,
        "batch_size": 512,
        "threads": 8,
        "n_gpu_layers": 99,
        "lifecycle": "on_demand",
        "startup_timeout_seconds": 45,
        "description": "All WATCHER decisions - code, math, planning",
        "expected_speed_tps": 51,
        "capabilities": ["code", "math", "reasoning", "json"]
    },
    
    "backup": {
        "name": "Claude Opus 4.5",
        "type": "external_api",
        "access_method": "claude_code_cli",
        "lifecycle": "always_available",
        "description": "Deep analysis escalation - novel insights, complex reasoning",
        "expected_speed_tps": 38,
        "capabilities": ["deep_analysis", "novel_discovery", "semantic_validation"]
    },
    
    "escalation": {
        "triggers": [
            "UNCERTAIN",
            "LOW_CONFIDENCE",
            "COMPLEX_FAILURE",
            "NOVEL_PATTERN"
        ],
        "confidence_threshold": 0.50,
        "max_escalations_per_run": 3
    },
    
    "defaults": {
        "temperature": 0.7,
        "max_tokens": 2048,
        "request_timeout_seconds": 120
    },
    
    "resource_management": {
        "free_gpus_after_query": true,
        "max_idle_seconds": 0,
        "description": "Server stops immediately after query to free both GPUs for pipeline work"
    }
}
```

### 2.6 LLM Router v2.0 (On-Demand)

**File:** `llm_services/llm_router.py` (~120 LOC)

```python
"""
LLM Router v2.0.0 - On-Demand Primary + Backup Architecture

- Starts LLM server only when needed
- Model partitioned across both GPUs
- Stops server after query to free GPU memory
- Escalates to Claude Opus 4.5 on trigger conditions
"""

import json
import subprocess
import requests
import time
from pathlib import Path
from typing import Optional, Dict, Any


class LLMRouter:
    """On-demand LLM routing with automatic server lifecycle."""
    
    def __init__(self, config_path: str = "llm_services/llm_server_config.json"):
        self.config = self._load_config(config_path)
        self.primary_url = f"http://localhost:{self.config['primary']['port']}/completion"
        self.escalation_triggers = set(self.config['escalation']['triggers'])
        self.confidence_threshold = self.config['escalation']['confidence_threshold']
        self.project_dir = Path(__file__).parent.parent
        
    def _load_config(self, path: str) -> dict:
        with open(path) as f:
            return json.load(f)
    
    def route(self, prompt: str, require_json: bool = True) -> Dict[str, Any]:
        """
        Route request to primary LLM (on-demand), escalate if needed.
        
        Lifecycle:
        1. Start LLM server (partitioned across both GPUs)
        2. Make query
        3. Stop LLM server (free GPUs)
        4. Return response (or escalate to Claude)
        """
        try:
            # Start server on-demand
            self._start_server()
            
            # Try primary
            response = self._call_primary(prompt)
            
            # Check for escalation triggers
            if self._should_escalate(response):
                backup_response = self._call_backup(prompt)
                return {
                    'response': backup_response,
                    'model_used': 'claude_opus_4.5',
                    'escalated': True,
                    'primary_response': response
                }
            
            return {
                'response': response,
                'model_used': 'deepseek_r1_14b',
                'escalated': False
            }
            
        finally:
            # Always stop server to free GPU memory
            self._stop_server()
    
    def _start_server(self, timeout: int = 60) -> None:
        """Start LLM server (partitioned across both GPUs)."""
        # Check if already running
        try:
            r = requests.get(f"http://localhost:{self.config['primary']['port']}/health", timeout=2)
            if r.status_code == 200:
                return  # Already running
        except:
            pass
        
        # Start server
        script = self.project_dir / "llm_services" / "start_llm_server.sh"
        result = subprocess.run(
            ["bash", str(script)],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start LLM server: {result.stderr}")
        
        # Wait for health
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = requests.get(f"http://localhost:{self.config['primary']['port']}/health", timeout=2)
                if r.status_code == 200:
                    return
            except:
                pass
            time.sleep(1)
        
        raise RuntimeError("LLM server failed to become healthy")
    
    def _stop_server(self) -> None:
        """Stop LLM server to free GPU memory."""
        script = self.project_dir / "llm_services" / "stop_llm_server.sh"
        subprocess.run(["bash", str(script)], capture_output=True, timeout=10)
    
    def _call_primary(self, prompt: str) -> str:
        """Call DeepSeek-R1-14B via llama.cpp server."""
        try:
            response = requests.post(
                self.primary_url,
                json={
                    'prompt': prompt,
                    'temperature': self.config['defaults']['temperature'],
                    'n_predict': self.config['defaults']['max_tokens'],
                    'stop': ['```', '</response>']
                },
                timeout=self.config['defaults']['request_timeout_seconds']
            )
            response.raise_for_status()
            return response.json()['content']
        except Exception as e:
            return f'{{"error": "{str(e)}", "decision": "UNCERTAIN"}}'
    
    def _call_backup(self, prompt: str) -> str:
        """Call Claude Opus 4.5 via Claude Code CLI."""
        try:
            result = subprocess.run(
                ['claude', '-p', prompt],
                capture_output=True,
                text=True,
                timeout=300
            )
            return result.stdout
        except Exception as e:
            return f'{{"error": "Backup failed: {str(e)}"}}'
    
    def _should_escalate(self, response: str) -> bool:
        """Check if response contains escalation triggers."""
        response_upper = response.upper()
        
        # Check for trigger keywords
        for trigger in self.escalation_triggers:
            if trigger in response_upper:
                return True
        
        # Check for low confidence in JSON response
        try:
            data = json.loads(response)
            confidence = data.get('confidence', 1.0)
            if confidence < self.confidence_threshold:
                return True
        except json.JSONDecodeError:
            pass
        
        return False
    
    def query(self, prompt: str) -> str:
        """Simple query interface - returns just the response text."""
        result = self.route(prompt)
        return result['response']
    
    def health_check(self) -> Dict[str, bool]:
        """Check availability of LLM services (without starting server)."""
        health = {'primary_model_exists': False, 'backup': False}
        
        # Check primary model file exists
        model_path = self.project_dir / "models" / self.config['primary']['model_file']
        health['primary_model_exists'] = model_path.exists()
        
        # Check backup (Claude CLI)
        try:
            result = subprocess.run(['claude', '--version'], capture_output=True, timeout=5)
            health['backup'] = result.returncode == 0
        except:
            pass
        
        return health
```

### 2.7 Server Startup (On-Demand) - VALIDATED

**CRITICAL LESSONS LEARNED (January 8, 2026):**

| Issue | Symptom | Solution |
|-------|---------|----------|
| `--flash-attn` | Server crashes silently | Remove flag OR use `--flash-attn on` |
| `stdout=DEVNULL` | Server dies immediately | Log to file instead |
| `--tensor-split` | Optional, works without | Remove for simplicity |

**File:** `llm_services/llm_on_demand.py` (Validated Working Code)

```python
import subprocess
import time
import requests

LLAMA_SERVER = "/home/michael/llama.cpp/build/bin/llama-server"
MODEL_PATH = "/home/michael/distributed_prng_analysis/models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf"
PORT = 8080

def start_llm():
    """Start LLM server - returns process handle."""
    
    # CRITICAL: Log to file, NOT DEVNULL (causes silent crash)
    log = open("/tmp/llm_server.log", "w")
    
    proc = subprocess.Popen([
        LLAMA_SERVER,
        "--model", MODEL_PATH,
        "--port", str(PORT),
        "--ctx-size", "16384",
        "--n-gpu-layers", "99"
        # NOTE: No --flash-attn (requires value), no --tensor-split (optional)
    ], stdout=log, stderr=subprocess.STDOUT)
    
    # Wait for ready (typically 3-6 seconds)
    for i in range(60):
        try:
            r = requests.get(f"http://localhost:{PORT}/health", timeout=2)
            if r.status_code == 200:
                print(f"✅ LLM ready in {i}s")
                return proc
        except:
            pass
        time.sleep(1)
    
    proc.terminate()
    return None


def stop_llm(proc):
    """Stop LLM server, free GPU memory."""
    if proc:
        proc.terminate()
        proc.wait(timeout=10)
    subprocess.run(["pkill", "-f", f"llama-server.*{PORT}"], capture_output=True)
```

### 2.8 Memory Usage (16K Context)

```
CUDA0 (RTX 3080 Ti): 5.8GB used, 5.9GB free ✅
CUDA1 (RTX 3080 Ti): 6.1GB used, 5.6GB free ✅
```

### 2.9 Server Management Commands

```bash
# Manual test (see full output)
/home/michael/llama.cpp/build/bin/llama-server \
    --model models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf \
    --port 8080 \
    --ctx-size 16384 \
    --n-gpu-layers 99

# Check health
curl http://localhost:8080/health

# View logs (if started via Python)
tail -f /tmp/llm_server.log

# Kill server
pkill -f 'llama-server.*8080'
```

### 2.10 Escalation Flow

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
    YES (UNCERTAIN, LOW_CONFIDENCE, etc.)
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

## 3. Watcher Agent

### 3.1 Purpose

The Watcher Agent is the central orchestrator that:
- Monitors pipeline step outputs
- Reads agent manifests for step configuration
- Queries LLM for decisions (PROCEED/RETRY/ESCALATE)
- Executes the appropriate action
- Injects `agent_metadata` into all results

**File:** `agents/watcher_agent.py`

### 3.2 Decision Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    WATCHER AGENT DECISION FLOW v2.0             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Step Completes → Output JSON written                        │
│                ↓                                                 │
│  2. Watcher reads output + manifest                             │
│                ↓                                                 │
│  3. Extract evaluation_metrics from output                      │
│                ↓                                                 │
│  4. Query DeepSeek-R1-14B: "Given these metrics, what action?"  │
│                ↓                                                 │
│  5. Check response for escalation triggers                      │
│     ├── No triggers → Use R1-14B response                       │
│     └── Triggers found → Escalate to Claude Opus 4.5            │
│                ↓                                                 │
│  6. Execute decision:                                           │
│     ├── PROCEED → Start next step                               │
│     ├── RETRY   → Re-run with modified params                   │
│     └── ESCALATE → Alert human, pause pipeline                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Watcher Agent Configuration

```python
@dataclass
class WatcherConfig:
    """Configuration for the Watcher Agent v2.0."""
    
    manifest_dir: str = "agent_manifests"
    output_dir: str = "pipeline_outputs"
    llm_config: str = "llm_services/llm_server_config.json"
    
    # Decision thresholds
    min_confidence_proceed: float = 0.7
    max_retries: int = 3
    escalation_timeout_hours: float = 24.0
    
    # LLM settings (v2.0)
    use_llm_for_decisions: bool = True
    primary_model: str = "deepseek_r1_14b"
    backup_model: str = "claude_opus_4.5"
    escalation_confidence: float = 0.50
    
    # GBNF grammar
    use_grammar: bool = True
    grammar_file: str = "llm_services/grammars/decision.gbnf"
```

### 3.4 LLM Integration

```python
class WatcherAgent:
    """Pipeline orchestrator with LLM-based decision making."""
    
    def __init__(self, config: WatcherConfig):
        self.config = config
        self.router = LLMRouter(config.llm_config)
        self.manifests = self._load_manifests()
        
    def evaluate_step(self, step: int, results: dict) -> AgentDecision:
        """Evaluate step results and decide next action."""
        
        manifest = self.manifests[step]
        
        # Build prompt with mission context
        prompt = self._build_evaluation_prompt(step, results, manifest)
        
        # Query LLM (auto-escalates if needed)
        llm_result = self.router.route(prompt, require_json=True)
        
        # Parse decision
        decision = self._parse_decision(llm_result['response'])
        decision.model_used = llm_result['model_used']
        decision.escalated = llm_result.get('escalated', False)
        
        # Log decision
        self._log_decision(step, results, decision)
        
        return decision
    
    def _build_evaluation_prompt(self, step: int, results: dict, manifest: dict) -> str:
        """Build step-specific evaluation prompt."""
        
        context = load_mission_context(step)
        
        return f"""
{context}

## Current Step Results

Step: {step} ({manifest['description']})
Results:
```json
{json.dumps(results, indent=2)}
```

## Decision Required

Based on these results, provide your decision in JSON format:
```json
{{
    "decision": "PROCEED" | "RETRY" | "ESCALATE",
    "confidence": 0.0-1.0,
    "reasoning": "explanation",
    "suggested_params": {{}}  // if RETRY
}}
```
"""
```

### 3.5 Decision Actions

| Action | Confidence | Behavior |
|--------|------------|----------|
| PROCEED | ≥ 0.70 | Trigger next step automatically |
| RETRY | 0.50 - 0.70 | Re-run current step with adjustments |
| ESCALATE | < 0.50 | Alert human for review |

### 3.6 Validation Results

All 6 pipeline steps validated:

```
✅ Step 1 (Window Optimizer): proceed (conf=0.79)
✅ Step 2 (Scorer Meta-Optimizer): proceed (conf=0.85)
✅ Step 3 (Full Scoring): proceed (conf=0.93)
✅ Step 4 (ML Meta-Optimizer): proceed (conf=0.93)
✅ Step 5 (Anti-Overfit Training): proceed (conf=0.85)
✅ Step 6 (Prediction Generator): proceed (conf=0.93)
ALL TESTS PASSED ✅
```

---

## 4. Agent Manifests

### 4.1 Location

`agent_manifests/` directory contains one JSON file per step:

| Manifest | Step | Purpose |
|----------|------|---------|
| `window_optimizer.json` | 1 | Bayesian parameter optimization |
| `scorer_meta.json` | 2.5 | Scorer hyperparameter tuning |
| `full_scoring.json` | 3 | Distributed feature extraction |
| `ml_meta.json` | 4 | Capacity and architecture planning |
| `reinforcement.json` | 5 | Anti-overfit model training |
| `prediction.json` | 6 | Final prediction generation |

### 4.2 Manifest Structure

```json
{
  "schema_version": "1.1",
  "agent_name": "window_optimizer_agent",
  "description": "Bayesian window optimization via Optuna TPE",
  "pipeline_step": 1,
  
  "inputs": [
    {"name": "lottery_file", "type": "file", "required": true},
    {"name": "seed_count", "type": "int", "default": 10000000}
  ],
  
  "outputs": [
    "optimal_window_config.json",
    "bidirectional_survivors.json",
    "train_history.json",
    "holdout_history.json"
  ],
  
  "actions": [{
    "type": "run_script",
    "script": "window_optimizer.py",
    "args_map": {
      "--lottery-file": "lottery_file",
      "--trials": "window_trials",
      "--strategy": "bayesian"
    }
  }],
  
  "success_criteria": {
    "min_survivors": 1000,
    "bidirectional_rate": {"max": 0.02}
  },
  
  "follow_up_agents": ["scorer_meta_agent"],
  "retry": 2,
  "timeout_minutes": 120
}
```

### 4.3 Complete Manifest Set (58 Parameters)

| Step | Key Parameters | Bounds |
|------|----------------|--------|
| 1 | window_size, offset, skip_min, skip_max, thresholds | See distributed_config.json |
| 2.5 | residue_mod_1/2/3, max_offset, temporal_window | 5-20, 50-150, 500-1500 |
| 3 | batch_size, holdout_offset | Derived from train_history |
| 4 | survivor_count, network_architecture, epochs | Derived, not data-aware |
| 5 | model_type, k_folds, n_trials, learning_rate | 4 model types, 5 folds |
| 6 | pool_sizes, confidence_threshold | 20/100/300, 0.6 |

---

## 5. Pydantic Models

### 5.1 Purpose

Type-safe data validation for agent communication:

```python
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union
from enum import Enum


class DecisionAction(str, Enum):
    PROCEED = "proceed"
    RETRY = "retry"
    ESCALATE = "escalate"


class AgentDecision(BaseModel):
    """LLM decision output with validation."""
    
    decision: DecisionAction
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=10)
    suggested_params: Optional[Dict] = None
    model_used: Optional[str] = None
    escalated: bool = False
    
    @validator('suggested_params')
    def validate_params(cls, v, values):
        if values.get('decision') == DecisionAction.RETRY and not v:
            raise ValueError("RETRY decision requires suggested_params")
        return v


class StepResult(BaseModel):
    """Validated step output."""
    
    step: int = Field(ge=1, le=6)
    status: str
    metrics: Dict[str, float]
    outputs: List[str]
    agent_metadata: Optional[Dict] = None
    
    
class AgentManifest(BaseModel):
    """Agent configuration manifest."""
    
    schema_version: str = "1.1"
    agent_name: str
    description: str
    pipeline_step: int
    inputs: List[Dict]
    outputs: List[Union[str, Dict]]
    actions: List[Dict]
    success_criteria: Dict
    follow_up_agents: List[str]
    retry: int = 2
    timeout_minutes: int = 120
    
    @validator('outputs', pre=True)
    def normalize_outputs(cls, v):
        """Convert v1.1 rich outputs to v1.0 string list."""
        result = []
        for item in v:
            if isinstance(item, dict):
                result.append(item.get('name', item.get('file_pattern', str(item))))
            else:
                result.append(str(item))
        return result
```

### 5.2 Context Models

```python
class MissionContext(BaseModel):
    """Step-specific mission context for LLM."""
    
    step: int
    step_name: str
    role: str
    mathematical_context: str
    decision_criteria: Dict[str, str]
    

class DataContext(BaseModel):
    """Data fingerprint for loop prevention."""
    
    fingerprint_hash: str = Field(min_length=8, max_length=8)
    fingerprint_version: str = "v2_data_only"
    training_window: Dict
    holdout_window: Dict
    survivor_source: Dict
    prng_hypothesis: Dict
```

---

## 6. GBNF Grammar

### 6.1 Purpose

GBNF grammar **forces** the LLM to output valid JSON - no prose, no thinking tags, just structured decisions.

**CRITICAL:** Without GBNF, DeepSeek-R1 outputs `<think>` reasoning blocks that break JSON parsing.

**File:** `llm_services/grammars/decision.gbnf`

```gbnf
root ::= "{" ws "\"decision\"" ws ":" ws decision-value ws "," ws "\"confidence\"" ws ":" ws number ws "," ws "\"reasoning\"" ws ":" ws string ws "}"
decision-value ::= "\"PROCEED\"" | "\"RETRY\"" | "\"ESCALATE\""
string ::= "\"" ([^"\\] | "\\" .)* "\""
number ::= "0." [0-9]+
ws ::= [ \t\n]*
```

### 6.2 Grammar Breakdown

| Rule | Purpose |
|------|---------|
| `root` | Forces exact JSON structure with 3 required fields |
| `decision-value` | Only allows PROCEED, RETRY, or ESCALATE |
| `number` | Confidence as 0.XX (e.g., 0.85) |
| `string` | Quoted reasoning text |
| `ws` | Optional whitespace |

### 6.3 Using Grammar in Requests

```python
DECISION_GRAMMAR = r'''
root ::= "{" ws "\"decision\"" ws ":" ws decision-value ws "," ws "\"confidence\"" ws ":" ws number ws "," ws "\"reasoning\"" ws ":" ws string ws "}"
decision-value ::= "\"PROCEED\"" | "\"RETRY\"" | "\"ESCALATE\""
string ::= "\"" ([^"\\] | "\\" .)* "\""
number ::= "0." [0-9]+
ws ::= [ \t\n]*
'''

def query_with_grammar(prompt: str) -> dict:
    """Query LLM with GBNF grammar constraint."""
    
    response = requests.post(
        "http://localhost:8080/completion",
        json={
            "prompt": prompt,
            "grammar": DECISION_GRAMMAR,
            "temperature": 0.3,  # Lower temp for deterministic JSON
            "n_predict": 200    # JSON is short
        },
        timeout=120
    )
    
    # Grammar guarantees valid JSON
    return json.loads(response.json()["content"])
```

### 6.4 Without vs With Grammar

| Without Grammar | With Grammar |
|-----------------|--------------|
| `"Okay, so I'm trying to figure out..."` | `{"decision": "PROCEED", ...}` |
| 500+ tokens of reasoning | 50 tokens of JSON |
| Parse failures, fallbacks needed | 100% reliable parsing |
| DeepSeek `<think>` tags | Clean JSON only |

---

## 7. Mission Context Templates

### 7.1 Prompt Structure (Validated)

The working prompt structure is simpler than originally designed:

```
Layer 1: ROLE STATEMENT (what you're evaluating)
Layer 2: CURRENT DATA (the metrics)
Layer 3: DECISION RULES (thresholds)
Layer 4: OUTPUT REQUEST (triggers GBNF)
```

### 7.2 Step 1: Window Optimizer Context

**File:** `agent_contexts/step1_window_optimizer.py`

```python
def build_step1_prompt(results: dict) -> str:
    """Build Step 1 evaluation prompt - works with GBNF grammar."""
    
    return f"""You are evaluating PRNG window optimization results.

RESULTS:
- bidirectional_count: {results.get('bidirectional_count', 0)}
- forward_count: {results.get('forward_count', 0)}
- prng_type: {results.get('prng_type', 'unknown')}

DECISION RULES:
- PROCEED if bidirectional_count > 1000 (confidence 0.85-0.95)
- PROCEED if bidirectional_count 100-1000 (confidence 0.70-0.85)
- RETRY if bidirectional_count 10-100 (confidence 0.50-0.70)
- ESCALATE if bidirectional_count < 10 (confidence 0.30-0.50)

Output your decision as JSON:
"""
```

### 7.3 Validated Test Results (January 8, 2026)

```
Scenario          Bidirectional   Decision    Confidence   Model
─────────────────────────────────────────────────────────────────────
EXCELLENT         15,847          PROCEED     0.92         deepseek_r1_14b
GOOD              2,847           PROCEED     0.92         deepseek_r1_14b
MARGINAL          47              RETRY       0.55         deepseek_r1_14b
POOR              3               ESCALATE    0.35         deepseek_r1_14b

REAL DATA         801             PROCEED     0.85         deepseek_r1_14b
(optimal_window_config.json)
```

**All decisions correct. All reasoning accurate.**

### 7.4 Step-Specific Context Templates (Validated)

**File:** `agent_contexts/mission_contexts.py`

```python
"""
Step-specific mission contexts for WATCHER Agent.
Each prompt is designed to work with GBNF grammar constraints.

Validated: January 8, 2026
"""

# =============================================================================
# MAIN MISSION STATEMENT (Shared across all steps)
# =============================================================================

MAIN_MISSION = """
You are the WATCHER Agent for a distributed PRNG analysis system.

SYSTEM GOAL: Reverse-engineer unknown PRNG behavior through:
1. Bidirectional sieve filtering (Forward + Reverse intersection)
2. ML-based probability scoring
3. Multi-model architecture comparison
4. Anti-overfit training validation
5. High-confidence prediction generation

CORE PRINCIPLE: "Learning steps declare signal quality; execution steps act 
only on declared usable signals; control agents decide recovery."

Your job: Evaluate pipeline step results and decide PROCEED / RETRY / ESCALATE.
"""


# =============================================================================
# STEP 1: Window Optimizer
# =============================================================================

def build_step1_prompt(results: dict) -> str:
    """
    Step 1: Bayesian window optimization via Optuna TPE.
    Key output: bidirectional_survivors.json
    """
    return f"""{MAIN_MISSION}

STEP 1: WINDOW OPTIMIZER (Bayesian Parameter Optimization)

YOUR ROLE: Evaluate if Optuna found sufficient bidirectional survivors.

MATHEMATICAL CONTEXT:
- Bidirectional survivors pass BOTH forward AND reverse sieves
- P(false positive) ≈ 10⁻¹¹⁹¹ for bidirectional match
- Every survivor is mathematically significant
- Threshold philosophy: LOW thresholds (0.001-0.10) for DISCOVERY
- Intersection performs actual filtering

CURRENT RESULTS:
- bidirectional_count: {results.get('bidirectional_count', 0)}
- forward_count: {results.get('forward_count', 0)}
- reverse_count: {results.get('reverse_count', 0)}
- prng_type: {results.get('prng_type', 'unknown')}
- optimization_score: {results.get('optimization_score', 0)}

DECISION RULES:
- PROCEED if bidirectional_count > 1000 (confidence 0.85-0.95)
- PROCEED if bidirectional_count 100-1000 (confidence 0.70-0.85)
- RETRY if bidirectional_count 10-100 (confidence 0.50-0.70)
- ESCALATE if bidirectional_count < 10 (confidence 0.30-0.50)

Output your decision as JSON:
"""


# =============================================================================
# STEP 2.5: Scorer Meta-Optimizer
# =============================================================================

def build_step2_prompt(results: dict) -> str:
    """
    Step 2.5: Distributed scorer hyperparameter tuning across 26 GPUs.
    Key output: optimal_scorer_config.json
    """
    return f"""{MAIN_MISSION}

STEP 2.5: SCORER META-OPTIMIZER (Hyperparameter Tuning)

YOUR ROLE: Evaluate if distributed Optuna found optimal scorer parameters.

MATHEMATICAL CONTEXT:
- Scorer evaluates seed candidates against historical lottery data
- Best accuracy indicates prediction potential
- Trial completion rate indicates cluster health
- Score variance indicates parameter stability

CURRENT RESULTS:
- best_accuracy: {results.get('best_accuracy', 0)}
- completed_trials: {results.get('completed_trials', 0)}
- total_trials: {results.get('total_trials', 0)}
- best_params: {results.get('best_params', {{}})}
- trial_success_rate: {results.get('trial_success_rate', 0)}

DECISION RULES:
- PROCEED if best_accuracy > 0.70 AND completed_trials > 50 (confidence 0.85-0.95)
- PROCEED if best_accuracy > 0.50 (confidence 0.70-0.85)
- RETRY if best_accuracy 0.30-0.50 (confidence 0.50-0.70)
- ESCALATE if best_accuracy < 0.30 OR completed_trials < 20 (confidence 0.30-0.50)

Output your decision as JSON:
"""


# =============================================================================
# STEP 3: Full Scoring
# =============================================================================

def build_step3_prompt(results: dict) -> str:
    """
    Step 3: Distributed feature extraction across all survivors.
    Key output: scored_survivors.json
    """
    return f"""{MAIN_MISSION}

STEP 3: FULL SCORING (Feature Extraction)

YOUR ROLE: Evaluate if all survivors were successfully scored.

MATHEMATICAL CONTEXT:
- Each survivor receives ML feature vector
- Scoring extracts statistical patterns from PRNG sequences
- Coverage percentage indicates data completeness
- Mean score indicates overall signal quality

CURRENT RESULTS:
- scored_count: {results.get('scored_count', 0)}
- total_survivors: {results.get('total_survivors', 0)}
- coverage_pct: {results.get('coverage_pct', 0)}
- mean_score: {results.get('mean_score', 0)}
- score_std: {results.get('score_std', 0)}

DECISION RULES:
- PROCEED if coverage_pct > 95% AND mean_score > 0.5 (confidence 0.90-0.95)
- PROCEED if coverage_pct > 90% (confidence 0.75-0.90)
- RETRY if coverage_pct 70-90% (confidence 0.50-0.70)
- ESCALATE if coverage_pct < 70% (confidence 0.30-0.50)

Output your decision as JSON:
"""


# =============================================================================
# STEP 4: ML Meta-Optimizer
# =============================================================================

def build_step4_prompt(results: dict) -> str:
    """
    Step 4: Multi-model architecture comparison (PyTorch, XGBoost, LightGBM, CatBoost).
    Key output: ml_meta_results.json
    """
    return f"""{MAIN_MISSION}

STEP 4: ML META-OPTIMIZER (Architecture Comparison)

YOUR ROLE: Evaluate if ML models show learnable signal.

MATHEMATICAL CONTEXT:
- Compares 4 model types: PyTorch neural net, XGBoost, LightGBM, CatBoost
- R² > 0.5 indicates learnable signal exists
- Validation R² close to training R² indicates generalization
- Best model advances to anti-overfit training

CURRENT RESULTS:
- best_model_type: {results.get('best_model_type', 'unknown')}
- best_train_r2: {results.get('best_train_r2', 0)}
- best_val_r2: {results.get('best_val_r2', 0)}
- pytorch_r2: {results.get('pytorch_r2', 0)}
- xgboost_r2: {results.get('xgboost_r2', 0)}
- lightgbm_r2: {results.get('lightgbm_r2', 0)}
- catboost_r2: {results.get('catboost_r2', 0)}

DECISION RULES:
- PROCEED if best_val_r2 > 0.5 (confidence 0.85-0.95)
- PROCEED if best_val_r2 > 0.3 (confidence 0.70-0.85)
- RETRY if best_val_r2 0.1-0.3, try different architecture (confidence 0.50-0.70)
- ESCALATE if best_val_r2 < 0.1, signal may be degenerate (confidence 0.30-0.50)

Output your decision as JSON:
"""


# =============================================================================
# STEP 5: Anti-Overfit Training
# =============================================================================

def build_step5_prompt(results: dict) -> str:
    """
    Step 5: Final model training with overfit prevention.
    Key output: model.pth, model.meta.json
    """
    return f"""{MAIN_MISSION}

STEP 5: ANTI-OVERFIT TRAINING (Final Model)

YOUR ROLE: Evaluate if trained model generalizes without overfitting.

MATHEMATICAL CONTEXT:
- Overfit ratio = train_r2 / test_r2 (should be < 1.5)
- Test R² on holdout data indicates true generalization
- Signal quality gate: test_r2 > 0.1 required for prediction
- Early stopping prevents memorization

CURRENT RESULTS:
- train_r2: {results.get('train_r2', 0)}
- test_r2: {results.get('test_r2', 0)}
- overfit_ratio: {results.get('overfit_ratio', 0)}
- signal_quality: {results.get('signal_quality', 'unknown')}
- epochs_completed: {results.get('epochs_completed', 0)}

DECISION RULES:
- PROCEED if test_r2 > 0.3 AND overfit_ratio < 1.5 (confidence 0.85-0.95)
- PROCEED if test_r2 > 0.1 AND overfit_ratio < 2.0 (confidence 0.70-0.85)
- RETRY if overfit_ratio > 2.0, needs more regularization (confidence 0.50-0.70)
- ESCALATE if test_r2 < 0.1, signal insufficient for prediction (confidence 0.30-0.50)

CRITICAL: If signal_quality == "DEGENERATE", must ESCALATE regardless of R².

Output your decision as JSON:
"""


# =============================================================================
# STEP 6: Prediction Generator
# =============================================================================

def build_step6_prompt(results: dict) -> str:
    """
    Step 6: Generate final prediction pools.
    Key output: prediction_pool.json
    """
    return f"""{MAIN_MISSION}

STEP 6: PREDICTION GENERATOR (Final Output)

YOUR ROLE: Evaluate prediction quality and coverage.

MATHEMATICAL CONTEXT:
- Prediction confidence based on model certainty
- Pool coverage = predicted numbers / total possible
- Higher confidence = narrower, more useful predictions
- Historical validation compares to past draws

CURRENT RESULTS:
- prediction_confidence: {results.get('prediction_confidence', 0)}
- pool_size: {results.get('pool_size', 0)}
- pool_coverage_pct: {results.get('pool_coverage_pct', 0)}
- historical_hit_rate: {results.get('historical_hit_rate', 0)}
- top_predictions: {results.get('top_predictions', [])}

DECISION RULES:
- PROCEED if prediction_confidence > 0.6 AND pool_coverage < 30% (confidence 0.85-0.95)
- PROCEED if prediction_confidence > 0.4 (confidence 0.70-0.85)
- RETRY if prediction_confidence 0.2-0.4, model may need retraining (confidence 0.50-0.70)
- ESCALATE if prediction_confidence < 0.2, predictions unreliable (confidence 0.30-0.50)

Output your decision as JSON:
"""


# =============================================================================
# CONTEXT LOADER
# =============================================================================

STEP_PROMPTS = {
    1: build_step1_prompt,
    2: build_step2_prompt,      # Step 2.5 uses same key
    2.5: build_step2_prompt,
    3: build_step3_prompt,
    4: build_step4_prompt,
    5: build_step5_prompt,
    6: build_step6_prompt,
}


def get_step_prompt(step: int, results: dict) -> str:
    """Get the appropriate prompt builder for a pipeline step."""
    builder = STEP_PROMPTS.get(step)
    if builder:
        return builder(results)
    raise ValueError(f"Unknown pipeline step: {step}")
```

### 7.5 Integration with WATCHER

```python
from agent_contexts.mission_contexts import get_step_prompt, DECISION_GRAMMAR

def evaluate_step(step: int, results: dict) -> dict:
    """Evaluate any pipeline step with LLM."""
    
    # Build step-specific prompt
    prompt = get_step_prompt(step, results)
    
    # Query LLM with GBNF grammar
    response = requests.post(
        "http://localhost:8080/completion",
        json={
            "prompt": prompt,
            "grammar": DECISION_GRAMMAR,
            "temperature": 0.3,
            "n_predict": 200
        }
    )
    
    return json.loads(response.json()["content"])
```

**File:** `agent_contexts/context_loader.py`

```python
from pathlib import Path
from typing import Dict, Any
import json


CONTEXT_FILES = {
    1: "step1_window_optimizer.md",
    2: "step2_bidirectional_sieve.md",
    2.5: "step2_5_scorer_meta.md",
    3: "step3_full_scoring.md",
    4: "step4_adaptive_meta.md",
    5: "step5_anti_overfit.md",
    6: "step6_prediction.md"
}


def load_mission_context(step: int) -> str:
    """Load mission context for a pipeline step."""
    context_dir = Path(__file__).parent
    filename = CONTEXT_FILES.get(step)
    
    if not filename:
        raise ValueError(f"No context defined for step {step}")
    
    context_path = context_dir / filename
    return context_path.read_text()


def build_prompt(step: int, current_data: Dict[str, Any]) -> str:
    """Build complete prompt with context and current data."""
    
    context = load_mission_context(step)
    
    return f"""
{context}

## CURRENT DATA

```json
{json.dumps(current_data, indent=2)}
```

## YOUR DECISION

Based on the above context and data, provide your decision:
"""
```

---

## 8. Fingerprint Registry

### 8.1 Purpose

Track dataset + PRNG combinations to prevent redundant work:

- **Fingerprint:** SHA256 hash of dataset characteristics (DATA ONLY, no PRNG)
- **Attempts:** List of PRNG types tried on this fingerprint
- **Outcomes:** Success/failure history per combination

### 8.2 Key Design: Data-Only Fingerprint

```python
def compute_data_context(train_draws, holdout_draws, survivors_file, 
                         survivor_count, prng_type, mod, train_start=1):
    """
    Compute data context with fingerprint for WATCHER loop prevention.
    
    CRITICAL (v3.3): Fingerprint is DATA-ONLY (no prng_type in hash).
    """
    train_end = train_start + train_draws - 1
    holdout_start = train_end + 1
    holdout_end = holdout_start + holdout_draws - 1
    
    survivors_filename = Path(survivors_file).name
    context_str = (
        f"{train_start}:{train_end}|"
        f"{holdout_start}:{holdout_end}|"
        f"{survivor_count}|"
        f"{survivors_filename}"
    )
    fingerprint = hashlib.sha256(context_str.encode()).hexdigest()[:8]
    
    return {
        "fingerprint_hash": fingerprint,
        "fingerprint_version": "v2_data_only",
        "training_window": {...},
        "holdout_window": {...},
        "prng_hypothesis": {
            "prng_type": prng_type,  # Tracked separately, NOT in hash
            "mod": mod
        }
    }
```

### 8.3 Registry Methods

```python
from agents.fingerprint_registry import FingerprintRegistry

registry = FingerprintRegistry()

# Check if PRNG already tried
if registry.is_combination_tried(fingerprint, prng_type):
    print("Skip - already tried this combination")

# Get untried PRNGs in priority order
untried = registry.get_untried_prngs(fingerprint, ALL_PRNGS)

# Record attempt outcome
registry.record_attempt(fingerprint, prng_type, outcome, sidecar)

# Higher-level decisions
entry = registry.get_entry(fingerprint)
if entry.total_failures >= 5:
    print("Window failed too many PRNGs → reject window")
```

### 8.4 PRNG Priority Order

```python
PRNG_PRIORITY_ORDER = [
    # Forward fixed (most common)
    "java_lcg", "mt19937", "xorshift32", "pcg32", "lcg32",
    "minstd", "xorshift64", "xorshift128", "xoshiro256pp", 
    "philox4x32", "sfc64",
    
    # Forward hybrid (variable skip)
    "java_lcg_hybrid", "mt19937_hybrid", "xorshift32_hybrid",
    
    # Reverse fixed
    "java_lcg_reverse", "mt19937_reverse",
    
    # Reverse hybrid
    "java_lcg_hybrid_reverse", "mt19937_hybrid_reverse"
]
```

### 8.5 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    FINGERPRINT REGISTRY v1.0                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  data_fingerprint = hash(window + survivors)  ← NO PRNG!        │
│         ↓                                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    SQLite Database                          ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         ││
│  │  │   attempts  │  │    runs     │  │  summaries  │         ││
│  │  │ fingerprint │  │   run_id    │  │ fingerprint │         ││
│  │  │ prng_type   │  │  started_at │  │   attempts  │         ││
│  │  │  outcome    │  │policy_ver   │  │  failures   │         ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘         ││
│  └─────────────────────────────────────────────────────────────┘│
│         ↓                                                        │
│  WATCHER Decisions:                                              │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Condition                    │ Action                       ││
│  │─────────────────────────────│────────────────────────────────││
│  │ Same fingerprint fails 5+   │ REJECT_WINDOW                 ││
│  │ Only linear PRNGs fail      │ TRY_CHAOTIC_FAMILY            ││
│  │ All PRNGs exhausted         │ SHIFT_WINDOW or ESCALATE      ││
│  │ Small holdout fails         │ EXPAND_HOLDOUT                ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| **LLM Architecture v2.0.0** | ✅ Complete | DeepSeek-R1-14B + Claude backup |
| LLM Router v2.0 | ✅ Complete | Single primary + escalation |
| Server startup | ✅ Complete | Single server script |
| Watcher Agent | ✅ ~70% | Needs escalation wiring |
| Agent Manifests | ✅ Complete | All 6 steps defined |
| Pydantic Models | ✅ Complete | Full type coverage |
| GBNF Grammars | ✅ Complete | Decision schema enforced |
| Fingerprint Registry | ✅ Complete | SQLite backend deployed |
| Mission Contexts | ✅ Complete | All 6 steps |
| Full Autonomous Loop | ❌ ~15% TODO | Wiring + testing |

**Overall Autonomy: ~85%**

---

## 10. TODO Items

1. **Complete escalation wiring** — Connect Claude backup to decision flow
2. **`--save-all-models` flag** — Save all 4 ML models for AI analysis
3. **End-to-end testing** — Full pipeline autonomous run
4. **Phase 2 Feature Importance** — Semantic validation with Claude API

---

## 11. A/B Test Results Reference

### Test Methodology (January 7, 2026)

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

- **14B Local:** Correct answers, proper application of concepts
- **Claude Opus:** Found 0.00002% MOD 1000 bias unprompted (novel derivation)

**Analogy:** 14B = PhD Candidate (correct), Opus = PhD Professor (discovers new things)

---

## 12. References

| Document | Purpose |
|----------|---------|
| `PROPOSAL_LLM_Architecture_v2_0_0.md` | Architecture proposal |
| `TEAM_BETA_SUMMARY_20260107.md` | A/B test results |
| `agents/watcher_agent.py` | Watcher implementation |
| `agents/fingerprint_registry.py` | SQLite registry |
| `llm_services/llm_router.py` | LLM routing logic |
| `llm_services/llm_server_config.json` | Server configuration |
| `agent_manifests/*.json` | Step configurations |
| `agent_contexts/*.md` | Mission context templates |

---

## Version History

```
Version 2.0.0 - January 7, 2026
- MAJOR: Replaced Dual-LLM with Primary + Backup architecture
- MAJOR: DeepSeek-R1-14B as primary (51 tok/s, handles code+math)
- MAJOR: Claude Opus 4.5 as backup for escalation
- MAJOR: Simplified LLM router (no keyword routing needed)
- Updated server startup script (single server)
- Added escalation trigger system
- Added A/B test results reference

Version 1.0.0 - December 2025
- Initial Dual-LLM architecture (Qwen2.5-Coder-14B + Qwen2.5-Math-7B)
- Keyword-based routing
- Dual server configuration
```

---

**END OF CHAPTER 10**
