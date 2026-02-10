# PRNG Analysis System — Soak Test Handoff Prompt
## Paste this entire prompt into a new Claude chat to resume work

---

# CONTEXT: Distributed PRNG Analysis & Functional Mimicry System

You are helping me operate and test a **distributed PRNG (Pseudo-Random Number Generator) analysis system** that performs functional mimicry — learning surface patterns and heuristics from PRNG outputs to generate high-value prediction pools. This is a white-hat research project. The system is modular, configurable, ML & AI compatible, targeting full autonomous operation.

**This is NOT specifically a lottery system.** The code is PRNG-agnostic by design — no step assumes a specific generator. All generator behavior is abstracted via `prng_registry.py` (46 PRNG algorithms).

---

## HARDWARE INFRASTRUCTURE

| Node | GPUs | Backend | IP | Role |
|------|------|---------|----|------|
| **Zeus** (coordinator) | 2× RTX 3080 Ti (12GB each) | CUDA/PyTorch | localhost | Orchestration, job dispatch, LLM host |
| **rig-6600** | 12× RX 6600 (8GB each) | ROCm/HIP | 192.168.3.120 | Worker Node 1 |
| **rig-6600b** | 12× RX 6600 (8GB each) | ROCm/HIP | 192.168.3.154 | Worker Node 2 |
| **rig-6600c** | (pending GPU install — 8 cards planned) | ROCm/HIP | 192.168.3.162 | Worker Node 3 |

- **Total active:** 26 GPUs, ~285 TFLOPS
- **Project directory on Zeus:** `~/distributed_prng_analysis`
- **Documentation on ser8:** `~/Downloads/CONCISE_OPERATING_GUIDE_v1.0/`
- **SSH from ser8 to Zeus:** use `rzeus` (not `zeus`)
- **ROCm activation:** `export HSA_OVERRIDE_GFX_VERSION=10.3.0 && source ~/tf/bin/activate`
- **Known hardware issue:** rig-6600b GPU[4] (slot 5, PCI 0F:00.0) has chronic loose connection — recently reseated Feb 1, PCIe set to Gen 1 for stability
- **Fan services:** Disabled on all rigs, using built-in auto fan control

---

## 6-STEP PIPELINE ARCHITECTURE

```
Step 1 → Step 2.5 → Step 3 → Step 4 → Step 5 → Step 6
Window    Scorer     Full      ML Meta   Anti-     Prediction
Optimizer Meta-Opt   Scoring   Optimizer  Overfit   Generator
(Optuna)  (26-GPU)   (26-GPU)  (Adaptive) (4 ML)   (Read-only)
```

| Step | Script | Output | GPU? |
|------|--------|--------|------|
| 1 | `window_optimizer.py` | `bidirectional_survivors.json` | Yes |
| 2.5 | `generate_scorer_jobs.py` → `scripts_coordinator.py` | `optimal_scorer_config.json` | Yes (26 GPU) |
| 3 | `run_step3_full_scoring.sh` (v2.0.0) | `survivors_with_scores.json` | Yes (26 GPU) |
| 4 | `adaptive_meta_optimizer.py` | `reinforcement_engine_config.json` | No |
| 5 | `meta_prediction_optimizer_anti_overfit.py` | `best_model.*` + `.meta.json` | Yes |
| 6 | `reinforcement_engine.py` | `prediction_pool.json` | No |

**Two coordinators:**
- `coordinator.py` — Seed-based operations (Steps 1-2)
- `scripts_coordinator.py` — Script-based jobs (Steps 3-6)

**4 ML models in Step 5:** neural_net, xgboost, lightgbm, catboost

---

## AI AGENT ARCHITECTURE

### WATCHER Agent (Chapter 12)
The autonomous pipeline orchestrator (`agents/watcher_agent.py`):
- Monitors pipeline step outputs
- Evaluates results via heuristic OR grammar-constrained LLM (DeepSeek-R1-14B)
- Decides next action: PROCEED / RETRY / ESCALATE
- Tracks PRNG attempts via Fingerprint Registry
- **~85% autonomy achieved**

### Chapter 13 — Live Feedback Loop
Post-prediction learning system:
- Detects new draws, compares predictions vs reality
- Generates diagnostics, triggers selective retraining (Steps 3→5→6)
- Proposes parameter changes via LLM
- Dispatches via WATCHER (autonomous, grammar-constrained)

### Selfplay (Phase 9B)
Reinforcement learning via tree models + bandit algorithms:
- `selfplay_orchestrator.py` (v1.1.0) — main selfplay loop
- `policy_transform.py` — stateless, deterministic, pure functional transforms
- `policy_conditioned_episode.py` — filter, weight, mask episode conditioning

### Authority Separation (CRITICAL INVARIANT)
- **Chapter 13 decides** (ground truth, promotion/rejection)
- **WATCHER enforces** (dispatch, safety, halt flag)
- **Selfplay explores** (never self-promotes)
- **LLM advises** (never executes code, never alters weights)

---

## DISPATCH MODULE (Phase 7 — Just Completed)

The dispatch module (`agents/watcher_dispatch.py`) wires Chapter 13 → WATCHER → Selfplay:

| Function | Purpose |
|----------|---------|
| `dispatch_selfplay()` | Spawns `selfplay_orchestrator.py` |
| `dispatch_learning_loop()` | Runs Steps 3→5→6 sequence |
| `process_chapter_13_request()` | Handles `watcher_requests/*.json` |
| `build_step_awareness_bundle()` | Unified LLM context assembly (7 bundle types) |
| LLM lifecycle management | Stop LLM before GPU phase, restart after |

**Evaluation path:**
```
_evaluate_step_via_bundle(prompt, grammar_name)
  ├─ Try 1: LLM Router (public API) — ONLY for watcher_decision.gbnf
  ├─ Try 2: HTTP Direct (generic) — POST localhost:8080/completion with inline grammar
  └─ Try 3: Heuristic Fallback — proceed, confidence=0.50 (should NEVER be reached)
```

**Guardrails:**
- Guardrail #1: Single context entry point — dispatch calls `build_llm_context()`, nothing else
- Guardrail #2: No baked-in token assumptions — bundle_factory owns prompt structure

---

## WATCHER CLI COMMANDS

```bash
cd ~/distributed_prng_analysis

# Pipeline Steps 1-6
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 1 --end-step 6

# Dispatch selfplay
PYTHONPATH=. python3 agents/watcher_agent.py --dispatch-selfplay

# Dispatch learning loop (Steps 3→5→6)
PYTHONPATH=. python3 agents/watcher_agent.py --dispatch-learning-loop steps_3_5_6

# Process pending Chapter 13 requests
PYTHONPATH=. python3 agents/watcher_agent.py --process-requests

# DAEMON MODE (monitor + auto-dispatch — this is what soak testing targets)
PYTHONPATH=. python3 agents/watcher_agent.py --daemon

# Selfplay standalone
python3 selfplay_orchestrator.py --survivors survivors_with_scores.json --episodes 5 --policy-conditioned

# Halt / Resume
python3 -m agents.watcher_agent --halt "Reason"
python3 -m agents.watcher_agent --run-pipeline --resume
```

---

## CURRENT COMPLETION STATUS (as of 2026-02-03)

### ALL PHASES COMPLETE ✅

| Phase | Status | Completed |
|-------|--------|-----------|
| 1. Draw Ingestion | ✅ Complete | 2026-01-12 |
| 2. Diagnostics Engine | ✅ Complete | 2026-01-12 |
| 3. Retrain Triggers | ✅ Complete | 2026-01-12 |
| 4. LLM Integration | ✅ Complete | 2026-01-12 |
| 5. Acceptance Engine | ✅ Complete | 2026-01-12 |
| 6. Chapter 13 Orchestration | ✅ Complete | 2026-01-12 |
| **7. WATCHER Integration** | **✅ Complete** | **2026-02-03** |
| 8. Selfplay Integration | ✅ Complete | 2026-01-30 |
| 9A. Chapter 13 ↔ Selfplay Hooks | ✅ Complete | 2026-01-30 |
| 9B.1 Policy Transform Module | ✅ Complete | 2026-01-30 |
| 9B.2 Policy-Conditioned Mode | ✅ Complete | 2026-01-30 |
| 9B.3 Policy Proposal Heuristics | 🔲 Deferred | — |

### Phase 7 Completion Details (Sessions 57-59)

**5 integration bugs found and fixed during D5 testing:**

| # | Bug | Root Cause | Fix |
|---|-----|-----------|-----|
| 1 | Lifecycle dead code | `self.llm_lifecycle` never set in `__init__` | Added initialization block |
| 2 | API mismatch | `.start()` / `.stop(string)` not real methods | → `.ensure_running()` / `.stop()` |
| 3 | Router always None | `GrammarType` import poisoned entire import | Removed dead import |
| 4 | Grammar 400 errors | `agent_grammars/` had broken v1.0 GBNF | Copied fixed v1.1 from `grammars/` |
| 5 | Try 1 private API | `_call_primary_with_grammar()` missing config | Gate to public API for `watcher_decision.gbnf` only |

**D5 End-to-End Test — CLEAN PASS (Session 59):**
```
Pre-validation: real LLM (4s response, not instant heuristic)
LLM stop: "confirmed stopped — GPU VRAM freed"
Selfplay: rc=0, candidate emitted (58s)
LLM restart: "healthy after 3.2s"
Post-eval: grammar-constrained JSON — real structured output
Archive: COMPLETED — zero warnings, zero heuristic fallbacks
```

This was a **single-cycle integration proof**, NOT sustained validation.

---

## TESTING COMPLETED vs TESTING REMAINING

### ✅ Testing DONE

| Test | Status | What It Proved |
|------|--------|----------------|
| D1-D4: Component dry-runs | ✅ | Individual modules work |
| D5: Single-cycle end-to-end | ✅ | Full wiring works (Chapter 13 → WATCHER → Selfplay → LLM eval → Archive) |
| Part A: Selfplay validation | ✅ | 8 episodes, 3 candidates, policy-conditioned mode, zero crashes |

### 🔲 Testing REMAINING — SOAK TESTS (This is what we need to do NOW)

| Test | Status | Description |
|------|--------|-------------|
| **Soak Test A: Multi-hour daemon run** | 🔲 Not started | Start `--daemon` mode, let it run 2-4+ hours continuously. Monitor for memory leaks, LLM connection drops, file handle exhaustion, stale lock files |
| **Soak Test B: 5-10 back-to-back requests** | 🔲 Not started | Inject 5-10 sequential Chapter 13 requests into `watcher_requests/`. Verify each is picked up, evaluated, dispatched, and archived correctly without race conditions or queue corruption |
| **Soak Test C: Stability under sustained load** | 🔲 Not started | Run daemon with synthetic draw injection enabled — continuous cycle of: new draw → diagnostics → trigger → dispatch → selfplay → evaluation → archive. Verify the full autonomous loop sustains without degradation |

### Why Soak Testing Was Deferred Until Now

Team Beta recommended (Session 53) building Phase 9B (policy-conditioned learning) BEFORE soak testing, with rationale: "Phase 7 testing is more meaningful after policy conditioning exists — you want to test sequential learning, not just static replay."

That has now been done:
- Sessions 53-55: Built policy transforms (Phase 9B)
- Sessions 57-59: Completed WATCHER dispatch wiring (Phase 7)
- Session 60: Documentation fully synchronized

**Soak testing is now the natural next step.** Team Beta's prerequisite — policy conditioning — is in place.

---

## KEY FILES ON ZEUS

### Chapter 13 Core (~226KB)
```
chapter_13_diagnostics.py (39KB), chapter_13_llm_advisor.py (23KB),
chapter_13_triggers.py (36KB), chapter_13_acceptance.py (41KB),
chapter_13_orchestrator.py (23KB), llm_proposal_schema.py (14KB),
chapter_13.gbnf (2.9KB), draw_ingestion_daemon.py (22KB),
synthetic_draw_injector.py (20KB), watcher_policies.json (4.7KB)
```

### Phase 7 Dispatch
```
agents/watcher_dispatch.py (~30KB) — dispatch functions
agents/contexts/bundle_factory.py (~32KB) — step awareness bundles
llm_services/llm_lifecycle.py (~8KB) — LLM lifecycle management
agent_grammars/*.gbnf (~6KB) — 4 fixed v1.1 grammar files
```

### Phase 9B Selfplay
```
selfplay_orchestrator.py (43KB), policy_transform.py (36KB),
policy_conditioned_episode.py (25KB), inner_episode_trainer.py,
modules/learning_telemetry.py
```

### Monitoring / Logs
```
watcher_decisions.jsonl — detailed decision audit
watcher_history.json — run history summary
watcher_requests/ — Chapter 13 request queue + archive
logs/watcher_agent.log — application logs
telemetry/learning_health_latest.json — selfplay telemetry
policy_history/ — policy candidate archive
```

---

## CRITICAL INVARIANTS & RULES

1. **Immutable structure, configurable parameters** — code is frozen, only parameters change
2. **LLMs never execute code** — they propose, WATCHER disposes
3. **NEVER restore from backup after modifying code** — fix mistakes by removing/editing the bad additions
4. **Step 3 uses `run_step3_full_scoring.sh` (v2.0.0)** — NOT `run_full_scoring.sh` (v1.2)
5. **Step 2 uses `.sh` (PULL architecture)** — NOT the broken `.py` (has hardcoded /shared/ml/ paths)
6. **Selfplay invariant:** GPU sieving MUST use coordinator.py / scripts_coordinator.py — direct SSH to rigs for GPU work is FORBIDDEN
7. **Policy transform invariant:** `apply_policy()` is pure functional — stateless, deterministic, never fabricates data
8. **Documentation sync invariant:** When code completes, update BOTH progress tracker AND original chapter checklist in same session

---

## KNOWN ISSUES & OPERATIONAL NOTES

- **Ramdisk preload is standalone-only:** When WATCHER runs Step 3, only Zeus gets ramdisk populated. Remote nodes need manual SCP before distributed Step 3 execution
- **System crashes stem from HIP initialization storms and I/O saturation** — not CPU weakness. Requires careful concurrency management and stagger timing
- **Memory pressure during data loading** (not GPU-side) is the primary constraint — requires sample size optimization and batching
- **NPZ v3.0 format** — preserves all 22 metadata fields (v2.0 silently dropped 19 fields causing 14/47 ML features to be zero)
- **LLM:** DeepSeek-R1-14B (primary) with Claude API backup, 32K context window, Pydantic + GBNF grammar constraints

---

## TEST MODE SETUP (if needed for soak testing)

```bash
cd ~/distributed_prng_analysis

# Enable test mode
python3 -c "
import json
with open('watcher_policies.json') as f: p = json.load(f)
p['test_mode'] = True
p['synthetic_injection']['enabled'] = True
with open('watcher_policies.json','w') as f: json.dump(p,f,indent=2)
print('Test mode enabled. True seed:', p['synthetic_injection']['true_seed'])
"
```

If `lottery_history.json` doesn't exist, bootstrap synthetic history first (see CANONICAL_PIPELINE Section 6.2).

---

## DEFERRED ITEMS (NOT in scope for soak testing)

- **Phase 9B.3:** Automatic policy proposal heuristics — deferred until 9B.2 validated
- **Parameter advisor (Item B):** LLM-advised parameter recommendations — deferred until pipeline reaches Steps 4-6 in production
- **`--save-all-models` flag:** Save all 4 models (not just winner) for post-hoc AI analysis
- **GPU2 failure on rig-6600:** Add debug logging to capture error before retry masks it

---

## YOUR TASK: Design and execute the Phase 7 soak tests

The three soak tests above (A: daemon endurance, B: sequential request handling, C: sustained autonomous loop) are the validation gate. All prerequisite code and infrastructure is in place.

Please propose a soak test plan and we'll execute on Zeus.
