# PROPOSAL: LLM Infrastructure Optimization & Grammar Completion

**Version:** 1.1.1  
**Date:** 2026-02-01 (reviewed 2026-02-02 after Part A validation)  
**Status:** PROPOSED — Pending Team Beta Approval  
**Author:** Claude (Team Beta session)  
**Scope:** LLM subsystem on Zeus (llm_services/ + agent_grammars/)  
**Impact:** Chapters 10, 12, 13 + Selfplay Integration — Enables full autonomous pipeline operation  
**Supersedes:** v1.0.0 (added selfplay integration context per user feedback)  
**Prerequisite Validated:** Selfplay system confirmed operational (8 episodes, 6/6 checks pass — see below)

---

## Executive Summary

Three targeted improvements to the LLM subsystem that collectively remove the remaining barriers to full pipeline autonomy. No new dependencies. No architectural changes. All work stays within existing file boundaries.

| Item | What | Effort | Impact |
|------|------|--------|--------|
| **A** | Context window 8192 → 32768 | 10 min | 4× richer LLM evaluation context |
| **B** | On-demand LLM server lifecycle | 1-2 hours | Reclaims BOTH 3080 Ti GPUs during compute |
| **C** | Grammar file completion (4 files) | 1-2 hours | Eliminates HTTP/heuristic fallbacks |
| **Total** | | **~3-4 hours** | **LLM subsystem production-ready** |

### Why Selfplay Makes This Urgent

With Phase 9B.2 complete (Jan 30), the system now has a **learning cycle** that repeatedly invokes the LLM subsystem:

```
[LLM ON]   Chapter 13 detects drift → LLM advisor proposes RETRAIN
                  (grammar: chapter_13.gbnf, needs rich context)
[LLM OFF]  WATCHER dispatches selfplay → episodes run on GPU cluster
                  (CPU ML + coordinator-mediated GPU sieving)
[LLM OFF]  Selfplay emits learned_policy_candidate.json
                  (fitness, lineage, fingerprint, val_r2, transforms)
[LLM ON]   Chapter 13 evaluates candidate vs REAL draws
                  (needs: candidate payload + diagnostics + policy history)
[LLM OFF]  If ACCEPT → WATCHER promotes to learned_policy_active.json
[LLM OFF]  WATCHER dispatches learning loop (Steps 3→5→6, heavy GPU compute)
[LLM ON]   WATCHER evaluates each step output
                  (grammar: agent_decision.gbnf)
[LLM ON]   Chapter 13 post-cycle assessment
                  (grammar: chapter_13.gbnf, full diagnostics)
```

Every `[LLM ON]` ↔ `[LLM OFF]` transition is a GPU resource conflict. The current infrastructure has no mechanism to manage this — the LLM holds ~4.25GB on EACH 3080 Ti whether or not it's being used.

**Without this proposal:** Phase 7 wiring (~180 lines) cannot use the GPUs efficiently during learning loops.  
**With this proposal + Phase 7:** Full autonomous learning cycle with dynamic GPU allocation.

---

## Part A: Context Window Increase (8192 → 32768)

### Problem

`llm_server_config.json` sets `context_length: 8192`. DeepSeek-R1-14B runs **partitioned across BOTH RTX 3080 Ti GPUs** via `n_gpu_layers: 99`. At 8192 tokens, the system uses ~40% of available VRAM — leaving over 14GB combined on the table.

### Why Selfplay Makes 32K Essential

When Chapter 13's LLM advisor evaluates a selfplay candidate, it needs to see:

| Context Item | Approx. Tokens | Source |
|-------------|---------------|--------|
| System prompt + evaluation instructions | ~800 | chapter_13_llm_advisor.py |
| Current diagnostics (multi-metric) | ~600-1,200 | post_draw_diagnostics.json |
| Selfplay health block | ~400-800 | learning_health_latest.json |
| Candidate payload (fitness, val_r2, transforms, lineage) | ~500-1,000 | learned_policy_candidate.json |
| Current active policy (for comparison) | ~300-500 | learned_policy_active.json |
| Recent policy history (last 3-5 promotions) | ~400-800 | policy_history/*.json |
| Multi-step pipeline history (last run) | ~500-1,000 | step output summaries |
| Parameter proposals + rationale | ~200-400 | generated in-context |
| **Total input before generation** | **~3,700-6,700** | |

At **8K context**, the advisor frequently truncates policy history and selfplay health — flying blind on exactly the signals that determine whether to ACCEPT or REJECT a candidate.

At **32K context**, the advisor sees the complete picture: candidate payload + current diagnostics + selfplay history + active policy + recent promotions. This is the difference between an informed decision and a coin flip.

### VRAM Budget (Dual-GPU Partition)

Model weights: ~8.5GB total → ~4.25GB per GPU  
Remaining per GPU: ~7.75GB for KV cache + overhead

| Context Length | KV Cache/GPU | Total/GPU | Headroom/GPU | Assessment |
|---------------|-------------|-----------|-------------|-----------|
| 8,192 (current) | ~0.65GB | ~4.9GB | **~7.1GB free** | Wasteful |
| 16,384 | ~1.3GB | ~5.55GB | **~6.45GB free** | Comfortable |
| **32,768** | **~2.6GB** | **~6.85GB** | **~5.15GB free** | **✅ RECOMMENDED** |
| 65,536 | ~5.2GB | ~9.45GB | ~2.55GB free | Tight |
| 131,072 | ~10.4GB | ~14.65GB | ❌ exceeds 12GB | Not viable |

### Changes Required

**File 1: `llm_services/llm_server_config.json`**
```json
{
    "primary": {
        "context_length": 32768    // was 8192
    }
}
```

**File 2: `llm_services/start_llm_servers.sh`**
```bash
--ctx-size 32768    # was 8192
```

### Speed Impact

Generation speed (51 tok/s) is **unchanged** — context length does not affect token generation rate. Prompt processing time increases linearly with input length (a 16K prompt takes ~2× as long as 8K to prefill). For WATCHER evaluations that happen once per pipeline step, this is negligible.

### Verification Step (REQUIRED before committing)

```bash
# Start server with new context size
./llm_services/start_llm_servers.sh

# Check ACTUAL VRAM on both cards
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# Expected: each GPU ~6-7GB used out of 12GB
# If tighter than expected: fall back to 16384 (still 2× improvement, completely safe)
```

---

## Part B: On-Demand LLM Server Lifecycle

### Problem

The LLM server (llama.cpp) runs 24/7, holding ~4.25GB on **each** 3080 Ti GPU. During Steps 2/3 heavy GPU compute, Zeus's GPUs need VRAM for sieving and scoring. During Chapter 13 LLM evaluation, both GPUs hold model weights. These two modes never run simultaneously, but currently the model weights sit in VRAM even during compute phases.

Chapter 10 §10 lists this as a known gap: *"On-demand LLM lifecycle — Server stays running vs start/stop per query."*

### Why Selfplay Makes Lifecycle Management Critical

The selfplay learning cycle creates a **repeating pattern** of LLM-on → LLM-off transitions:

```
SELFPLAY LEARNING CYCLE — GPU TIMELINE
═══════════════════════════════════════

Phase 1: TRIGGER DETECTION
    LLM: ON  (Chapter 13 advisor evaluates drift)
    GPU: Model weights (~4.25GB each card)
         ↓
Phase 2: SELFPLAY EXPLORATION
    LLM: OFF ← lifecycle.stop() reclaims 8.5GB total
    GPU: Outer episodes use coordinator.py for GPU sieving
         Inner episodes use CPU ML (tree models)
         Duration: minutes to hours depending on episode count
         ↓
Phase 3: CANDIDATE EVALUATION
    LLM: ON  ← lifecycle.ensure_running() loads model
    GPU: Model weights (~4.25GB each card)
         Chapter 13 evaluates candidate vs REAL draws
         ↓
Phase 4: PROMOTION + RETRAIN
    LLM: OFF ← lifecycle.stop() reclaims 8.5GB total
    GPU: dispatch_learning_loop() runs Steps 3→5→6
         Step 3 (full scoring) = heavy GPU compute on cluster
         Step 5 (anti-overfit training) = heavy GPU compute on Zeus
         Duration: 20-60 minutes depending on scope
         ↓
Phase 5: POST-CYCLE ASSESSMENT
    LLM: ON  ← lifecycle.ensure_running()
    GPU: WATCHER evaluates each step output
         Chapter 13 runs post-cycle diagnostics
         ↓
(repeat from Phase 1 on next trigger)
```

Without lifecycle management, **8.5GB of VRAM sits idle during Phases 2 and 4** — exactly when the GPUs are under maximum compute load. With the learning cycle running regularly (drift → explore → retrain → assess), this waste compounds.

### Design

Create `llm_services/llm_lifecycle.py` — a lightweight manager that the WATCHER and Chapter 13 LLM advisor call to ensure the server is running before evaluation, and optionally stop it after.

**Principle:** The WATCHER already has the three-tier fallback hierarchy (grammar → HTTP → heuristic). The lifecycle manager only needs to handle server startup/shutdown and health checks. If the server fails to start, the existing fallback chain handles it gracefully.

```python
# llm_services/llm_lifecycle.py — Proposed API

class LLMLifecycleManager:
    """On-demand LLM server management.
    
    Usage:
        mgr = LLMLifecycleManager()
        
        # Before LLM evaluation
        mgr.ensure_running()
        response = llm_router.evaluate(prompt)
        
        # After evaluation (optional — keeps server for rapid re-use)
        mgr.stop()
    
    Context manager support:
        with mgr.session():
            response = llm_router.evaluate(prompt)
        # Server auto-stops after context exit
    """
    
    def __init__(self, config_path="llm_services/llm_server_config.json"):
        self.config = load_config(config_path)
        self.process = None
    
    def is_healthy(self) -> bool:
        """Check if server responds on health endpoint."""
        # GET http://localhost:8080/health with 2s timeout
    
    def ensure_running(self, timeout_sec=30) -> bool:
        """Start server if not already running. Block until healthy."""
        if self.is_healthy():
            return True
        return self._start_server(timeout_sec)
    
    def stop(self, timeout_sec=10):
        """Gracefully stop the server, freeing GPU VRAM."""
        # pkill -f 'llama-server' or process.terminate()
    
    @contextmanager
    def session(self):
        """Context manager: start → yield → stop."""
        self.ensure_running()
        try:
            yield
        finally:
            self.stop()
    
    def _start_server(self, timeout_sec) -> bool:
        """Launch start_llm_servers.sh, poll health until ready."""
        # subprocess.Popen with startup script
        # Poll health endpoint every 1s up to timeout
        # Log startup time for monitoring
```

### Integration Points — Selfplay Cycle

**1. Chapter 13 LLM Advisor — Candidate Evaluation**

When Chapter 13 evaluates a selfplay candidate (Phase 3 above), the advisor wraps the call in a lifecycle session:

```python
# chapter_13_llm_advisor.py — candidate evaluation flow
def evaluate_selfplay_candidate(self, candidate, diagnostics, active_policy):
    """Evaluate selfplay candidate against real draws.
    
    Lifecycle: start LLM → generate proposal → stop LLM
    VRAM freed after evaluation for compute phases.
    """
    with self.llm_lifecycle.session():
        # Build rich context (benefits from 32K window — Part A)
        context = self._build_candidate_context(
            candidate=candidate,          # fitness, val_r2, transforms, lineage
            diagnostics=diagnostics,      # current pipeline health
            active_policy=active_policy,  # comparison baseline
            policy_history=self._load_recent_promotions(n=5)
        )
        proposal = self._generate_proposal(context)
    # Server stops here — VRAM freed for dispatch_learning_loop()
    return proposal
```

**2. WATCHER Agent — Selfplay Dispatch (Phase 7 integration)**

```python
# agents/watcher_agent.py — dispatch_selfplay()
def dispatch_selfplay(self, request: dict) -> bool:
    """Execute selfplay_orchestrator.py with policy conditioning.
    
    LLM is NOT needed during selfplay episodes.
    Stop server before dispatch to free GPU VRAM.
    """
    # STOP LLM — selfplay uses CPU ML + coordinator-mediated GPU sieving
    if self.llm_lifecycle:
        self.llm_lifecycle.stop()
    
    cmd = [
        "python3", "selfplay_orchestrator.py",
        "--survivors", "survivors_with_scores.json",
        "--episodes", str(request.get("episodes", 5)),
        "--policy-conditioned",
        "--project-root", self.project_root,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=7200)
    
    # LLM will restart on next evaluation via ensure_running()
    return result.returncode == 0
```

**3. WATCHER Agent — Learning Loop Dispatch (Phase 7 integration)**

```python
# agents/watcher_agent.py — dispatch_learning_loop()
def dispatch_learning_loop(self, scope: str = "steps_3_5_6") -> bool:
    """Execute partial pipeline rerun (Steps 3→5→6).
    
    Heavy GPU compute — LLM must be stopped.
    WATCHER evaluates each step output with LLM after completion.
    """
    # STOP LLM before heavy GPU compute
    if self.llm_lifecycle:
        self.llm_lifecycle.stop()
    
    if scope == "steps_3_5_6":
        for step in [3, 5, 6]:
            self._run_step(step)
            
            # BRIEF LLM session to evaluate step output
            if self.llm_lifecycle:
                self.llm_lifecycle.ensure_running()
            evaluation = self._evaluate_step_output(step)
            if self.llm_lifecycle:
                self.llm_lifecycle.stop()
            
            if evaluation.decision == "escalate":
                return False  # abort learning loop
    
    return True
```

**4. Chapter 13 — Drift Detection (Trigger Phase)**

```python
# chapter_13_orchestrator.py — on new draw
def on_new_draw(self, draw_result):
    """Process new draw result and trigger learning if needed.
    
    Uses LLM session for analysis, then frees GPU.
    """
    diagnostics = self.diagnostics_engine.generate(draw_result)
    triggers = self.trigger_engine.evaluate(diagnostics)
    
    if triggers.should_retrain:
        with self.llm_lifecycle.session():
            proposal = self.llm_advisor.analyze_diagnostics(diagnostics)
        
        # LLM stopped — proposal written to watcher_requests/
        if self.acceptance_engine.validate(proposal):
            self.request_selfplay(proposal)
    # GPU fully free for WATCHER to execute the request
```

### Behavior Modes — Complete Selfplay Cycle

| Cycle Phase | LLM State | GPU Use | Trigger |
|-------------|-----------|---------|---------|
| Drift detection | ON (session) | Model weights | on_new_draw() |
| Selfplay exploration | **OFF** | Coordinator sieving + CPU ML | dispatch_selfplay() → stop() |
| Candidate evaluation | ON (session) | Model weights | evaluate_selfplay_candidate() |
| Policy promotion | N/A (no LLM) | Minimal | Chapter 13 file write |
| Learning loop (Steps 3→5→6) | **OFF** (with brief ON between steps) | Full GPU compute | dispatch_learning_loop() → stop() |
| Post-cycle assessment | ON (session) | Model weights | Post-dispatch evaluation |
| Idle (between draws) | **OFF** | GPUs fully free | No activity |

### Keep-Alive Option

For rapid-fire evaluations (e.g., WATCHER evaluating Steps 3, 5, 6 in quick succession during a learning loop), the lifecycle manager supports an optional cooldown:

```python
def __init__(self, ..., idle_timeout_sec=60):
    """If set, server stays alive for N seconds after last use."""
```

This avoids the ~5-10 second startup penalty when evaluations happen in quick succession. Particularly useful during dispatch_learning_loop() where 3 step evaluations happen within minutes.

### Estimated Size

~120-150 lines for `llm_lifecycle.py`. ~15 lines of integration into `watcher_agent.py`. ~10 lines of integration into `chapter_13_llm_advisor.py`.

---

## Part C: Grammar File Completion

### Problem

Chapter 10 §7 defines four grammar types in `grammar_loader.py`:

```python
class GrammarType(str, Enum):
    AGENT_DECISION = "agent_decision"
    SIEVE_ANALYSIS = "sieve_analysis"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    JSON_GENERIC = "json_generic"
```

The `agent_grammars/` directory exists on Zeus (created Jan 30, commit 22abd7b) with `chapter_13.gbnf` moved there. But the four grammar files referenced by the loader **do not exist**. This means:

- WATCHER step evaluations fall through to HTTP JSON extraction or heuristic fallback
- Non-Chapter-13 LLM calls cannot use grammar-constrained decoding
- The LLM can return malformed JSON that requires error-prone regex parsing

### Where Grammars Meet Selfplay

The selfplay learning cycle uses two distinct grammar contexts:

| Grammar | Used By | When in Cycle | Purpose |
|---------|---------|---------------|---------|
| `chapter_13.gbnf` | Chapter 13 LLM advisor | Drift detection + candidate evaluation | Structured proposal: RETRAIN/WAIT/ESCALATE/FULL_RESET + parameter proposals |
| `agent_decision.gbnf` | WATCHER | Step output evaluation during learning loop | Structured decision: proceed/retry/escalate |
| `sieve_analysis.gbnf` | WATCHER | Step 2 evaluation (if sieve re-runs during learning loop) | Survivor assessment + forward/reverse balance |
| `parameter_adjustment.gbnf` | WATCHER + Chapter 13 | Post-selfplay parameter changes | Structured adjustments with delta format |
| `json_generic.gbnf` | Any | Fallback when specific grammar fails to load | Syntactically valid JSON |

**Without these grammars:** The WATCHER evaluations during `dispatch_learning_loop()` (Steps 3, 5, 6) use HTTP JSON extraction — which can return malformed JSON that crashes the loop. A single parse failure during an automated learning cycle can halt the entire process and require human intervention.

**With these grammars:** Every LLM call in the cycle is grammar-constrained. The LLM physically cannot produce invalid output. The learning loop runs to completion without parse failures.

### Grammar File Specifications

#### C1: `agent_grammars/agent_decision.gbnf`

**Used by:** WATCHER `_evaluate_with_router()` for all Steps 1-6  
**Selfplay role:** Step evaluations during `dispatch_learning_loop()` (Steps 3→5→6)  
**Purpose:** Force the LLM to output exactly `{decision, confidence, reasoning}`

Chapter 10 §7.3 already specifies this grammar:

```gbnf
# Agent Decision Grammar (Chapter 10 §7.3)
# WATCHER step evaluation: proceed / retry / escalate
#
# VERSION: 1.0.0
# DATE: 2026-02-01

root ::= "{" ws
    "\"decision\"" ws ":" ws decision-value ws "," ws
    "\"confidence\"" ws ":" ws confidence-value ws "," ws
    "\"reasoning\"" ws ":" ws string
    ws "}"

decision-value ::= "\"proceed\"" | "\"retry\"" | "\"escalate\""

confidence-value ::= "0" ("." [0-9]+)? | "1" (".0")? | "0." [0-9]+

string ::= "\"" string-content "\""
string-content ::= ([^"\\] | "\\" ["\\/bfnrt])*

ws ::= [ \t\n\r]*
```

**Note:** This is the grammar that Chapter 10 already documented but never created as a file. Direct transcription from the spec.

#### C2: `agent_grammars/sieve_analysis.gbnf`

**Used by:** WATCHER for Step 2 (Bidirectional Sieve) evaluation  
**Selfplay role:** If selfplay outer episodes trigger re-sieving via coordinator, WATCHER evaluates the result  
**Purpose:** Force the LLM to interpret sieve results with structured assessment

```gbnf
# Sieve Analysis Grammar
# Step 2: Bidirectional sieve result interpretation
#
# VERSION: 1.0.0
# DATE: 2026-02-01

root ::= "{" ws
    "\"decision\"" ws ":" ws decision-value ws "," ws
    "\"confidence\"" ws ":" ws confidence-value ws "," ws
    "\"survivor_assessment\"" ws ":" ws survivor-assessment ws "," ws
    "\"forward_reverse_balance\"" ws ":" ws balance-value ws "," ws
    "\"recommended_seed_action\"" ws ":" ws seed-action ws "," ws
    "\"reasoning\"" ws ":" ws string
    ws "}"

decision-value ::= "\"proceed\"" | "\"retry\"" | "\"escalate\""

survivor-assessment ::= "\"excellent\"" | "\"adequate\"" | "\"marginal\"" | "\"insufficient\""

balance-value ::= "\"balanced\"" | "\"forward_heavy\"" | "\"reverse_heavy\"" | "\"severe_imbalance\""

seed-action ::= "\"keep_current\"" | "\"increase_seeds\"" | "\"decrease_seeds\"" | "\"adjust_thresholds\""

confidence-value ::= "0" ("." [0-9]+)? | "1" (".0")? | "0." [0-9]+

string ::= "\"" string-content "\""
string-content ::= ([^"\\] | "\\" ["\\/bfnrt])*

ws ::= [ \t\n\r]*
```

**Design rationale:**
- `survivor_assessment` maps to sieve survivor count quality (excellent ≥100K, adequate ≥50K, marginal ≥10K, insufficient <10K)
- `forward_reverse_balance` captures bidirectional intersection health
- `seed_action` gives WATCHER an actionable recommendation (validated against manifest bounds)

#### C3: `agent_grammars/parameter_adjustment.gbnf`

**Used by:** WATCHER and Chapter 13 for parameter change proposals  
**Selfplay role:** After selfplay produces a learned policy candidate, Chapter 13 may propose parameter adjustments as part of the promotion decision  
**Purpose:** Structured parameter recommendations with bounds checking

```gbnf
# Parameter Adjustment Grammar
# Steps 1-6: Parameter change recommendations
#
# VERSION: 1.0.0
# DATE: 2026-02-01

root ::= "{" ws
    "\"decision\"" ws ":" ws decision-value ws "," ws
    "\"confidence\"" ws ":" ws confidence-value ws "," ws
    "\"adjustments\"" ws ":" ws adjustments-array ws "," ws
    "\"risk_level\"" ws ":" ws risk-level ws "," ws
    "\"reasoning\"" ws ":" ws string
    ws "}"

decision-value ::= "\"apply\"" | "\"defer\"" | "\"escalate\""

adjustments-array ::= "[]" |
    "[" ws adjustment (ws "," ws adjustment)* ws "]"

adjustment ::= "{" ws
    "\"parameter\"" ws ":" ws string ws "," ws
    "\"current_value\"" ws ":" ws (number | "null") ws "," ws
    "\"proposed_value\"" ws ":" ws number ws "," ws
    "\"delta\"" ws ":" ws delta-string ws "," ws
    "\"rationale\"" ws ":" ws string
    ws "}"

delta-string ::= "\"" delta-content "\""
delta-content ::= ("+" | "-" | "*")? [0-9]+ ("." [0-9]+)?

risk-level ::= "\"low\"" | "\"medium\"" | "\"high\""

confidence-value ::= "0" ("." [0-9]+)? | "1" (".0")? | "0." [0-9]+

number ::= "-"? [0-9]+ ("." [0-9]+)?

string ::= "\"" string-content "\""
string-content ::= ([^"\\] | "\\" ["\\/bfnrt])*

ws ::= [ \t\n\r]*
```

**Design rationale:**
- `adjustments` array reuses the same parameter-proposal structure from `chapter_13.gbnf` (consistent schema)
- `delta-string` uses the same format as Chapter 13 (`+0.05`, `-10`, `*1.2`)
- `risk_level` matches Chapter 13's vocabulary
- Shared primitives are identical across all grammars

#### C4: `agent_grammars/json_generic.gbnf`

**Used by:** Fallback for any LLM call that needs valid JSON but no specific schema  
**Purpose:** Ensures the LLM always outputs syntactically valid JSON

```gbnf
# Generic JSON Grammar
# Fallback: Ensures any LLM output is valid JSON
#
# VERSION: 1.0.0
# DATE: 2026-02-01
#
# Use when no specific grammar is available.
# Constrains output to valid JSON object (not array/scalar).

root ::= object

object ::= "{" ws (pair (ws "," ws pair)*)? ws "}"

pair ::= string ws ":" ws value

value ::= string | number | object | array | "true" | "false" | "null"

array ::= "[" ws (value (ws "," ws value)*)? ws "]"

string ::= "\"" string-content "\""
string-content ::= ([^"\\] | "\\" ["\\/bfnrt] | "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])*

number ::= "-"? [0-9]+ ("." [0-9]+)? ([eE] [+-]? [0-9]+)?

ws ::= [ \t\n\r]*
```

### Grammar Loader Update

`llm_services/grammar_loader.py` needs a minor update to resolve the directory path:

```python
# Current (v1.0.0):
GRAMMAR_DIR = "agent_grammars"  # hardcoded relative path

# Updated (v1.1.0):
GRAMMAR_DIR = os.path.join(os.path.dirname(__file__), "..", "agent_grammars")
# Resolves to distributed_prng_analysis/agent_grammars/ regardless of CWD
```

And the `get_grammar_path()` method needs to map enum values to filenames:

```python
GRAMMAR_FILES = {
    GrammarType.AGENT_DECISION: "agent_decision.gbnf",
    GrammarType.SIEVE_ANALYSIS: "sieve_analysis.gbnf",
    GrammarType.PARAMETER_ADJUSTMENT: "parameter_adjustment.gbnf",
    GrammarType.JSON_GENERIC: "json_generic.gbnf",
}
```

The existing `chapter_13.gbnf` is NOT mapped through the GrammarType enum — it's loaded directly by the Chapter 13 LLM advisor via filename. No change needed there.

### Post-Grammar Directory Structure

```
agent_grammars/
├── agent_decision.gbnf           # WATCHER Steps 1-6 (NEW)
├── sieve_analysis.gbnf           # Step 2 specific (NEW)
├── parameter_adjustment.gbnf     # Parameter changes (NEW)
├── json_generic.gbnf             # Fallback (NEW)
└── chapter_13.gbnf               # Ch.13 proposals (EXISTS — moved Jan 30)
```

---

## Implementation Plan

### Dependencies

```
Part A (Context Window) ── no dependencies, can go first
       ↓
Part B (LLM Lifecycle) ── depends on A (config loaded by lifecycle manager)
       ↓
Part C (Grammar Files) ── independent, can parallel with B
       ↓
Phase 7 (WATCHER Dispatch) ── depends on B for stop/start around compute
```

### Session Plan

**Session 1: Quick Wins + Lifecycle (estimated 2-3 hours)**

| Order | Item | Time | Action |
|-------|------|------|--------|
| 1 | Part A: Context window | 10 min | Edit 2 files, verify with nvidia-smi |
| 2 | Part C: Grammar files | 60-90 min | Create 4 .gbnf files, update grammar_loader.py |
| 3 | Part B: LLM lifecycle | 60-90 min | Create llm_lifecycle.py, integrate with WATCHER |
| 4 | Verification | 30 min | Test grammar loading, lifecycle start/stop, context size |

**Session 2: Phase 7 WATCHER Integration (estimated 2-2.5 hours)**

Per existing `TODO_PHASE7_WATCHER_INTEGRATION_REVISED.md`:
- ~~Part A: Selfplay validation testing~~ — ✅ COMPLETE (2026-02-01)
- Part B: WATCHER dispatch functions — 180 lines (60-90 min)
  - `dispatch_selfplay()` with `llm_lifecycle.stop()` before dispatch
  - `dispatch_learning_loop()` with lifecycle transitions between steps
  - `process_chapter_13_request()` handling watcher_requests/*.json
- Part D: Integration testing (60 min)
  - End-to-end: Chapter 13 → WATCHER → Selfplay → Evaluate → Learning Loop

#### Selfplay Validation Results (Part A — Completed 2026-02-01)

The selfplay system was validated end-to-end on Zeus before this proposal was finalized. This confirms the selfplay cycle assumptions throughout this document are grounded in tested behavior.

| Check | Result | Evidence |
|-------|--------|---------|
| Multi-episode run (5+3 episodes) | ✅ | Zero crashes, all models trained |
| Config loading | ✅ | `configs/selfplay_config.json` created and loaded |
| Policy transforms (filter + weight) | ✅ | 75,396 → 47,614 survivors; 46,715/47,614 weights adjusted |
| Candidate emission | ✅ | `learned_policy_candidate.json` written (schema v1.1.0) |
| Policy history archive | ✅ | 3 files accumulated in `policy_history/` |
| Telemetry tracking | ✅ | 38 models tracked, JSON valid |

Git commit: `c0f5d32` — "docs: Part A selfplay testing COMPLETE, selfplay config created"

### Git Commit Plan

```bash
# Commit 1: Context window
git commit -m "config: Increase LLM context window 8192 → 32768

Dual-GPU partition leaves 5.15GB headroom per card at 32K.
Enables richer context for Chapter 13 LLM advisor.
Selfplay candidate evaluation now sees full diagnostic payload."

# Commit 2: Grammar files
git commit -m "feat: Create 4 missing GBNF grammar files (Ch.10 §7)

- agent_decision.gbnf (WATCHER step evaluation)
- sieve_analysis.gbnf (Step 2 specific)
- parameter_adjustment.gbnf (parameter changes)
- json_generic.gbnf (fallback)
- Update grammar_loader.py v1.0.0 → v1.1.0
Closes grammar gap that caused HTTP fallback in learning loops."

# Commit 3: LLM lifecycle
git commit -m "feat: On-demand LLM server lifecycle manager

New: llm_services/llm_lifecycle.py
- ensure_running() / stop() / session() context manager
- Frees both 3080 Ti GPUs during selfplay + learning loops
- Integrates with WATCHER and Ch.13 LLM advisor
Closes Ch.10 §10 known gap: on-demand LLM lifecycle"
```

---

## Complete Autonomy Loop — Post-Implementation

After this proposal + Phase 7 dispatch wiring:

```
NEW DRAW
    ↓
Chapter 13 Ingestion → Diagnostics → Trigger Evaluation
    ↓
[LLM ON — session]
    Chapter 13 LLM Advisor analyzes diagnostics
    (32K context: full diagnostic payload + selfplay health + policy history)
    (grammar: chapter_13.gbnf — structured RETRAIN/WAIT/ESCALATE)
    → Proposal emitted
[LLM OFF — session ends, VRAM freed]
    ↓
WATCHER Acceptance Engine validates proposal
    ↓ (if RETRAIN approved)
[LLM OFF — stop() called]
    dispatch_selfplay() → selfplay_orchestrator.py
    (CPU ML + coordinator-mediated GPU sieving)
    (policy-conditioned episodes using learned_policy_active.json)
    → learned_policy_candidate.json emitted
    ↓
[LLM ON — session]
    Chapter 13 evaluates candidate vs REAL draws
    (32K context: candidate + active policy + recent promotions)
    (grammar: chapter_13.gbnf)
    → ACCEPT or REJECT
[LLM OFF — session ends]
    ↓ (if ACCEPT)
Chapter 13 promotes to learned_policy_active.json
Chapter 13 writes retrain request → watcher_requests/
    ↓
[LLM OFF — stop() called]
    dispatch_learning_loop(scope="steps_3_5_6")
    ↓
    Step 3: Full scoring (distributed GPU compute on cluster)
        → [brief LLM ON] evaluate output (grammar: agent_decision.gbnf) → [LLM OFF]
    ↓
    Step 5: Anti-overfit training (GPU compute on Zeus)
        → [brief LLM ON] evaluate output (grammar: agent_decision.gbnf) → [LLM OFF]
    ↓
    Step 6: Prediction generation (CPU on Zeus)
        → [brief LLM ON] evaluate output (grammar: agent_decision.gbnf) → [LLM OFF]
    ↓
[LLM ON — session]
    Chapter 13 post-cycle assessment
    → Ready for next draw
[LLM OFF]
    ↓
IDLE (GPUs fully free until next draw)
```

**No human in the loop for routine decisions.**

---

## Authority Contract Compliance

This proposal respects all invariants from `CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md`:

| Invariant | How This Proposal Complies |
|-----------|---------------------------|
| 1. Promotion Authority | LLM lifecycle does not affect who writes learned_policy_active.json — still Chapter 13 only |
| 2. Ground Truth Isolation | Selfplay never receives draw outcomes — lifecycle transitions don't change data access |
| 3. Selfplay Output = Hypotheses | Grammar constraints don't change selfplay's advisory status |
| 4. Coordinator Requirement | Lifecycle manager never touches GPU sieving — coordinators still handle all distributed work |
| 5. Telemetry ≠ Sole Input | Grammar-constrained LLM proposals still go through acceptance engine validation |
| 6. Safe Fallback | Lifecycle failures trigger existing 3-tier fallback (grammar → HTTP → heuristic), baseline policy always recoverable |

---

## Risk Analysis

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| 32K context exceeds VRAM | Low | Medium | Verify with nvidia-smi before committing; fallback to 16K |
| LLM server fails to restart mid-cycle | Low | Low | Existing 3-tier fallback handles it; learning loop pauses, doesn't crash |
| Grammar file syntax error | Medium | Low | Test each grammar with llama.cpp `--grammar-file` before integration |
| Lifecycle manager adds latency | Low | Low | 5-10s startup; negligible for once-per-step evaluations |
| Grammar too restrictive for LLM | Medium | Medium | Test with real diagnostic payloads; json_generic.gbnf as fallback |
| Stop/start thrashing during learning loop | Low | Low | idle_timeout_sec=60 prevents rapid restart cycles |

---

## Success Criteria

### Part A Complete When:
- [ ] `nvidia-smi` shows both GPUs with acceptable VRAM at 32768 context
- [ ] `curl http://localhost:8080/health` returns healthy
- [ ] LLM advisor can process a full Chapter 13 diagnostic payload + selfplay candidate without truncation

### Part B Complete When:
- [ ] `llm_lifecycle.ensure_running()` starts server from cold state in <15 seconds
- [ ] `llm_lifecycle.stop()` releases VRAM on both GPUs (confirmed via nvidia-smi)
- [ ] `llm_lifecycle.session()` context manager handles start/stop cleanly
- [ ] dispatch_selfplay() successfully stops LLM before selfplay episodes
- [ ] dispatch_learning_loop() manages LLM lifecycle across Steps 3→5→6

### Part C Complete When:
- [ ] All 4 grammar files pass llama.cpp syntax validation
- [ ] `grammar_loader.py` resolves all GrammarType enums to existing files
- [ ] WATCHER `_evaluate_with_router()` uses grammar-constrained decoding for Steps 1-6
- [ ] Chapter 13 LLM advisor uses chapter_13.gbnf for proposal generation
- [ ] Fallback chain still works when grammar loading fails

### Full Autonomy Achieved When:
- [ ] End-to-end cycle: Draw → Chapter 13 → Selfplay → Evaluate → Retrain → Predict
- [ ] No HTTP fallback needed for standard evaluations
- [ ] GPU VRAM fully available between pipeline phases
- [ ] No human intervention required for routine learning cycles

---

## Frozen Components (No Changes)

| Component | Reason |
|-----------|--------|
| `llm_router.py` | Routing logic unchanged; lifecycle is orthogonal |
| `chapter_13.gbnf` | Already complete and in correct location |
| `doctrine.py` | Decision framework unchanged |
| Step contexts (6 files) | No schema changes needed |
| `agent_manifests/*.json` | Manifest format unchanged |
| `distributed_config.json` | Runtime bounds unchanged |
| `selfplay_orchestrator.py` | Selfplay code unchanged; lifecycle managed by callers |
| `policy_transform.py` | Pure functional, no LLM dependency |
| `policy_conditioned_episode.py` | Episode logic unchanged |
| `configs/selfplay_config.json` | Created during Part A validation (production-ready) |
| `CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md` | All invariants preserved |

---

## Changelog from v1.0.0

| Section | Change |
|---------|--------|
| Executive Summary | Added selfplay cycle urgency and LLM ON/OFF transition diagram |
| Part A | Added context budget table showing selfplay candidate evaluation needs |
| Part B | Replaced generic integration examples with selfplay-specific code: dispatch_selfplay(), dispatch_learning_loop(), evaluate_selfplay_candidate(), on_new_draw() |
| Part B | Added "Behavior Modes — Complete Selfplay Cycle" table |
| Part B | Added GPU timeline diagram showing 5-phase learning cycle |
| Part C | Added "Where Grammars Meet Selfplay" mapping table |
| Part C | Added selfplay role annotations to each grammar spec |
| New section | "Complete Autonomy Loop — Post-Implementation" full flow diagram |
| New section | "Authority Contract Compliance" verification table |
| Success Criteria | Added selfplay-specific criteria (dispatch stops LLM, learning loop lifecycle) |
| Frozen Components | Added selfplay files to frozen list (no changes needed) |

### v1.1.1 Updates (2026-02-02)

| Section | Change |
|---------|--------|
| Header | Added prerequisite validation status |
| Session Plan | Part A marked COMPLETE with validation results table and git commit reference |
| Frozen Components | Added `configs/selfplay_config.json` (created during Part A validation) |
| Session 2 estimate | Reduced from 3 hours to 2-2.5 hours (Part A done) |

---

**END OF PROPOSAL**
