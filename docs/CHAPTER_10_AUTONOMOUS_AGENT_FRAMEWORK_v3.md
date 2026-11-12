# Chapter 10: Autonomous Agent Framework

> **What TFM is.** TFM filters candidate generators against observed draws in two directions,
> keeps a population of survivors on purpose, ranks them, and lets the top candidates vote their
> next value into a prediction pool. A survivor is a scored candidate, not a verdict. The
> generator family is a substrate for mimicry, not a claim about the machine. The target is a
> pool hit rate. TFM does not recover a seed or reconstruct state, and an objection that assumes
> it does is out of scope.

**Version:** 3.2.0 — corrected 2026-11-11 (was 3.1.0)
**Status:** **CORRECTED at `a15b524`, 2026-11-11** — audited against the code, corrected in place,
history preserved. Not closed: see §17 for the closure statement and sentinel, §15 for open items.
**Authority for this pass:** `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md` (REV1).
Audit of record: `docs/CHAPTER_10_AUDIT.md`.
**Base:** `a15b5247336255801de8cf67d52653701b1e8c80` (`a15b524`), VM 101, working tree. **Code moved during
this pass:** `b6cb789` and then `cf9edaa` changed only `agents/watcher_agent.py`, `agents/watcher_dispatch.py`
and `tests/test_watcher_*` (`git diff --name-only a15b524 HEAD`). Anchors into those two files are taken
from `git show a15b524:<path>` and say "at a15b524"; every other file cited is identical at `a15b524` and
HEAD (`git diff --quiet`). Behaviour repaired at `b6cb789` is labelled against `a15b524` and carries a
"repaired at `b6cb789`" note.
**Canonical version:** this file (`_v3.md`). `docs/CHAPTER_10_AUTONOMOUS_AGENT_FRAMEWORK_v2.md` is superseded
(`docs/PROJECT_FILE_CATALOG.md:970`) and `_v3.md.bak` is a gitignored stale duplicate (`:652`, `.gitignore:106`);
neither was edited. Evidence in `docs/CHAPTER_10_AUDIT.md` header.
**Numbering scheme:** **EXECUTABLE** (`docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1). Bare "Step N" is the step
WATCHER runs (`STEP_SCRIPTS` / `STEP_MANIFESTS` / `STEP_NAMES`, `agents/watcher_agent.py:387-417` at a15b524):
Step 0 = TRSE, **Step 1 = Window Optimizer, inside which the bidirectional sieve runs**, Step 2 = Scorer
Meta-Optimizer (the "Step 2.5" label below is the conceptual name; `scorer_meta.json` `pipeline_step` is `2`),
Steps 3–6 as in §5.2. Chapter numbers are not step numbers.

**Original header (retained):**
**Version:** 3.1.0 (Full Autonomous Operation)  
**Date:** February 3, 2026  
**Status:** ✅ Full Autonomous Operation — Phase 7 Complete  
**Autonomy:** ~85%  

> **Corrected 2026-11-11 — header.** "Phase 7 Complete" is the February 2026 Phase 7 (selfplay / Chapter 13
> integration), not S172 RANGE-MINER Phase 7, whose soak is unblocked and not run
> (`docs/PROJECT_FILE_CATALOG.md:290`; `docs/CHAPTER_12_AUDIT.md` §7). DESCRIPTION-ONLY (D1). "~85%" has no
> definition or measurement in code or docs: **[UNVERIFIED]** (U1). "Full autonomous operation" is not what
> the code does at `a15b524`: every step's verdict is decided by file existence (§1, Known gaps).

---

## 0. How this chapter was produced

**Added 2026-11-11.** The 2026-02-03 text (v3.1.0, 586 lines) was audited claim by claim against the code
at `a15b524` on VM 101; the audit is `docs/CHAPTER_10_AUDIT.md` (66 claims). Method:

1. Governance trail first (`docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/TEAM_ALPHA_*`), then the fact file,
   `docs/TODO_CONSOLIDATED.md`, `docs/PIPELINE_BEHAVIOUR_MODEL.md`, the committed chapter audits (Ch5, Ch12,
   Ch13, Ch14) and `docs/TRSE_STEP0_AUDIT_v1.md`, then code.
2. Code read this session: `agents/watcher_agent.py` and `agents/watcher_dispatch.py` (at a15b524, in the
   parts named in the anchors), `agents/doctrine.py`, `agents/full_agent_context.py`, `agents/contexts/`,
   `agents/manifest/agent_manifest.py`, `agents/pipeline/pipeline_step_context.py`,
   `agents/registry/parameter_registry.py`, `agents/parameters/parameter_context.py`, `agents/step_runner/`,
   `agents/prompts/`, `llm_services/` (router, grammar loader, lifecycle, config, start script),
   `agent_grammars/`, `grammars/`, `agent_contexts/`, all eight `agent_manifests/*.json` (including the
   gitignored `definitions.json`), `watcher_policies.json`, `trse_step0.py`, `window_optimizer.py` (bounds
   loader, TRSE write-back, argparse), `adaptive_meta_optimizer.py` argparse, `chapter_13_orchestrator.py`.
3. Host state, read-only: LLM server health (`curl`), model and binary presence, `nvidia-smi`,
   `systemctl list-units --all`, `crontab -l`.
4. **Nothing was executed.** No WATCHER, agent, LLM server or pipeline step was run; argparse was read.
   "Works" below means the code path is complete on reading.

**How to read the corrections.** Original text is kept. Each disagreement is followed by a
`Corrected 2026-11-11` note naming what the code does and its label (BROKEN / SWITCHED-OFF-BY-RULING /
DESCRIPTION-ONLY / absorbed into an existing item); superseded sections carry **SUPERSEDED (2026-11-11)**
and a pointer. All labels are collected in §14.

**What changed underneath this chapter since v3.1.0.**
- **RANGE-MINER** is now the certifying Step-1 sieve engine. The agent framework does not choose a sieve
  backend: WATCHER routes Step 1 by manifest parameter, and `use_range_miner` defaults to `false`
  (`docs/CHAPTER_12_WATCHER_AGENT.md` §13; Ch12 S1). Nothing in `llm_services/`, `agents/contexts/`, the
  doctrine or the parameter registry names RANGE-MINER.
- **The 22-array NPZ contract** is Step 1's real output. No agent-framework file reads it; Step 1's WATCHER
  verdict is the existence of `optimal_window_config.json`.
- **The window-anchor change** (Brief I: `window_anchor` + `generator_phase` split) reaches this framework
  only at the parameter registry's `offset` entry, which Brief I deferred to Brief II (§14, S1). Brief I is
  implemented; **final acceptance is pending a completed production-shape run**
  (`docs/S172_BRIEF_I_PRODUCTION_SHAPE_CLASSIFICATION_REPORT.md`).

---

## 1. Executive Summary

The Autonomous Agent Framework is **fully implemented and working**. This chapter documents the existing codebase, validated through live testing on January 8, 2026.

> **Corrected 2026-11-11 — "fully implemented and working".** The modules exist. Whether they work was not
> confirmed by a run this pass: **[UNVERIFIED]** (U2). On reading, the LLM evaluation path this chapter
> describes is **not reached for any step at `a15b524`**: all seven step manifests declare
> `evaluation_type: file_exists`, and `evaluate_results` returns the file-existence verdict before the
> router, doctrine or heuristic is consulted (`agents/watcher_agent.py:946-1004` at a15b524). Absorbed into
> `docs/CHAPTER_12_AUDIT.md` D3 and B1. **B1 repaired at `b6cb789`**: an explicit step failure
> (`success: False`) now escalates before the file-existence verdict
> (`docs/CLAUDE_CODE_REPORT_WATCHER_SAFETY_REPAIRS.md` R-1); the LLM path is still not reached. The LLM server
> was not running when checked (2026-11-11, `curl localhost:8080/health` → connection refused).

### 1.1 Verified Working Components

| Component | File | Status |
|-----------|------|--------|
| Watcher Agent | `agents/watcher_agent.py` | ✅ v1.4.0 Working |
| Watcher Dispatch | `agents/watcher_dispatch.py` | ✅ v1.0.0 Working (Session 58) |
| Bundle Factory | `agents/contexts/bundle_factory.py` | ✅ v1.0.0 Working (Session 58) |
| LLM Lifecycle | `llm_services/llm_lifecycle.py` | ✅ v1.0.0 Working (Session 57) |
| LLM Router | `llm_services/llm_router.py` | ✅ v2.0.0 Working |
| Grammar Loader | `llm_services/grammar_loader.py` | ✅ v1.0.0 Working |
| Server Startup | `llm_services/start_llm_servers.sh` | ✅ v2.1.0 Working |
| Step Contexts | `agents/contexts/*.py` | ✅ All 6 steps + Chapter 13 |
| Doctrine | `agents/doctrine.py` | ✅ v3.2.0 Working |
| Prompt Builder | `agents/prompt_builder.py` | ✅ v3.2.0 Working |
| GBNF Grammars | `agent_grammars/*.gbnf` | ✅ v1.1 (4 files, fixed Session 59) |

> **Corrected 2026-11-11 — versions in §1.1** (DESCRIPTION-ONLY unless stated):
> - Watcher Agent is **v2.0.0** (`agents/watcher_agent.py:12`, `:3467` at a15b524), not v1.4.0 (D2).
> - Bundle Factory is **v1.1.0** (`agents/contexts/bundle_factory.py:5`, `bundle_version` `:432`) (D3).
> - LLM Router is **v2.1.0** (`llm_services/llm_router.py:3`) (D4).
> - Grammar Loader is **v1.1.0** (`llm_services/grammar_loader.py:5`) (D5).
> - GBNF grammars: `agent_grammars/` holds **5** files (`chapter_13.gbnf` added). A second tracked directory,
>   **`grammars/`** (8 files, including `watcher_decision.gbnf`), is the one the router and the Chapter-13
>   advisor actually load, CWD-relative (`llm_router.py:235`, `:252`; `chapter_13_llm_advisor.py:309-312`;
>   cf. `docs/CHAPTER_13_AUDIT.md:262`). `grammar_loader.py` serves `agent_grammars/` (`:41-54`) (D6).
> - Accurate as written: Watcher Dispatch v1.0.0 (`agents/watcher_dispatch.py:5`), LLM Lifecycle v1.0.0
>   (`llm_services/llm_lifecycle.py:5`), Server Startup v2.1.0 (`start_llm_servers.sh:4`), Doctrine v3.2.0
>   (`agents/doctrine.py:8`), Prompt Builder v3.2.0 (`agents/prompt_builder.py:8`), six step contexts plus a
>   Chapter-13 bundle (`agents/contexts/*_context.py`; `bundle_factory.py:595`).
> - "Working": see the note above (U2).

### 1.2 Live Test Results (January 8, 2026)

```
$ PYTHONPATH=. python3 agents/watcher_agent.py --evaluate optimal_window_config.json

Step: 1 - Window Optimizer
Success: True
Confidence: 0.95
Action: proceed
Reasoning: The evaluation shows bidirectional_count, forward_count, and reverse_count 
           all at 801, meeting the success condition.
Parse method: llm_http_extracted
```

> **SUPERSEDED (2026-11-11) — historical record, not current behaviour.** Three days after this test,
> `6326257` (2026-01-11, "Step 1 file_exists evaluation") set `window_optimizer.json` to
> `evaluation_type: file_exists`. The same command at `a15b524` returns `Parse method: file_exists` with
> confidence 1.00 (file valid) or 0.00, without calling the LLM (`agents/watcher_agent.py:946-1004`,
> `--evaluate` branch `:3782-3803`, at a15b524). Absorbed into `docs/CHAPTER_12_AUDIT.md` D3. Replaced by:
> §4.3 note and §14.

---

## 2. Architecture Overview

### 2.1 System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AUTONOMOUS AGENT FRAMEWORK v3.1.0                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │  Pipeline Step  │───►│  Watcher Agent  │───►│   LLM Router    │        │
│  │   Completes     │    │  (Orchestrator) │    │   (v2.0.0)      │        │
│  └─────────────────┘    └────────┬────────┘    └────────┬────────┘        │
│                                  │                      │                  │
│                                  ▼                      ▼                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │  Step Context   │    │    Doctrine     │    │ DeepSeek-R1-14B │        │
│  │ (Metrics Only)  │    │ (Decision Rules)│    │   (Primary)     │        │
│  └─────────────────┘    └─────────────────┘    └────────┬────────┘        │
│                                                         │                  │
│                                  ┌──────────────────────┘                  │
│                                  ▼                                          │
│                         ┌─────────────────┐                                │
│                         │ Claude Opus 4.5 │                                │
│                         │    (Backup)     │                                │
│                         └─────────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

> **Corrected 2026-11-11 — diagram.** Primary DeepSeek-R1-14B on llama.cpp and backup Claude via the Claude
> Code CLI match `llm_services/llm_server_config.json:4-20` and `llm_router.py:112-215` (accurate). The flow
> "Pipeline Step Completes → Watcher → LLM Router" is not taken for any step at `a15b524`: the file-existence
> verdict comes first (absorbed into `docs/CHAPTER_12_AUDIT.md` D3). The doctrine reaches the LLM only as
> prompt text (`agents/full_agent_context.py:155`, `:219`); WATCHER imports
> `validate_decision_against_doctrine` (`agents/watcher_agent.py:151` at a15b524) and does not call it.
> Step-evaluation prompts for Step 1 come from `agents/prompts/step1_prompt.py`; dispatch prompts come from
> `agents/contexts/bundle_factory.py` (`agents/watcher_dispatch.py:75` at a15b524).

### 2.2 File Locations

```
distributed_prng_analysis/
├── agents/
│   ├── watcher_agent.py           # Main orchestrator (v1.1.0)
│   ├── doctrine.py                # Decision framework rules
│   ├── prompt_builder.py          # Prompt assembly
│   ├── agent_decision.py          # Decision models
│   ├── agent_core.py              # Base agent class
│   ├── fingerprint_registry.py    # Dataset+PRNG tracking
│   ├── full_agent_context.py      # Context builder
│   │
│   ├── contexts/                  # Step-specific contexts
│   │   ├── base_agent_context.py
│   │   ├── window_optimizer_context.py      # Step 1
│   │   ├── scorer_meta_context.py           # Step 2.5
│   │   ├── full_scoring_context.py          # Step 3
│   │   ├── ml_meta_context.py               # Step 4
│   │   ├── anti_overfit_context.py          # Step 5
│   │   └── prediction_context.py            # Step 6
│   │
│   ├── manifest/                  # Manifest loading
│   ├── parameters/                # Parameter context
│   ├── history/                   # Analysis history
│   ├── runtime/                   # Runtime context
│   ├── safety/                    # Kill switch
│   └── step_runner/               # Step execution
│
├── llm_services/
│   ├── llm_router.py              # Primary + Backup routing (v2.0.0)
│   ├── grammar_loader.py          # GBNF grammar management
│   ├── llm_server_config.json     # Server configuration
│   └── start_llm_servers.sh       # Server startup script
│
└── agent_manifests/               # Step configurations
    ├── window_optimizer.json
    ├── scorer_meta.json
    ├── full_scoring.json
    ├── ml_meta.json
    ├── reinforcement.json
    └── prediction.json
```

> **Corrected 2026-11-11 — file locations.** Every file listed exists (accurate). DESCRIPTION-ONLY (D7):
> `watcher_agent.py` is v2.0.0, not v1.1.0; `scorer_meta_context.py` serves executable Step 2 (Ch12 D14);
> there are **7** manifests, including `trse.json` (Ch12 D5), plus the gitignored, unread `definitions.json`
> (§13.4). Not listed but part of the framework: `agents/watcher_dispatch.py`, `watcher_registry_hooks.py`,
> `registry_inspector.py`, `threshold_guardrail.py`, `progress_display.py`, `agents/registry/` (parameter
> registry, §13.1), `agents/prompts/`, `agents/pipeline/`, `agents/contexts/bundle_factory.py` and
> `advisor_bundle.py`, the second grammar directory `grammars/` (§7 note), and `agent_contexts/` (§13.3).
> - `fingerprint_registry.py # Dataset+PRNG tracking`: the module exists; **no WATCHER code path calls it or
>   its hooks** (only importer `agents/registry_inspector.py:30`; DB `agents/data/fingerprint_registry.db`
>   last written 2026-01-02). Absorbed into `docs/CHAPTER_12_AUDIT.md` B2.
> - `step_runner/ # Step execution` (D8): the package exists (`agents/step_runner/__init__.py:1-30`, v1.0.0)
>   and **nothing imports it**; WATCHER executes steps through its own `run_step`
>   (`agents/watcher_agent.py:1705` at a15b524). Its "separate dispatch surface is not part of the certified
>   Run-4 route" (`docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:133`; `docs/BACKLOG.md:590-592`).

---

## 3. LLM Architecture

### 3.1 Primary + Backup Design

| Role | Model | Speed | Access |
|------|-------|-------|--------|
| **Primary** | DeepSeek-R1-14B | 51 tok/s | llama.cpp port 8080 |
| **Backup** | Claude Opus 4.5 | 38 tok/s | Claude Code CLI |

> **Corrected 2026-11-11.** Port 8080 on llama.cpp and the Claude Code CLI backup
> (`subprocess.run(['claude', '--print', '-p', prompt])`, `llm_router.py:186-187`) are accurate. The speeds
> appear only in a docstring (`llm_router.py:9-10`) and no benchmark log was found: **[UNVERIFIED]** (U3).
> "Claude Opus 4.5" is the name in the config (`llm_server_config.json:16`); the CLI uses whatever model the
> installed `claude` is configured for.

### 3.2 Server Configuration

**File:** `llm_services/llm_server_config.json`

```json
{
    "schema_version": "2.0.0",
    "architecture": "primary_plus_backup",
    
    "primary": {
        "name": "DeepSeek-R1-14B",
        "model_file": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
        "port": 8080,
        "context_length": 8192,
        "n_gpu_layers": 99
    },
    
    "backup": {
        "name": "Claude Opus 4.5",
        "type": "external_api",
        "access_method": "claude_code_cli"
    },
    
    "routing": {
        "escalation_triggers": [
            "UNCERTAIN", "LOW_CONFIDENCE", 
            "ESCALATE_TO_BACKUP", "REQUIRES_DEEP_ANALYSIS"
        ]
    }
}
```

> **Corrected 2026-11-11 (D9).** The live config is `schema_version` **2.1.0** with `context_length`
> **32768** (raised from 8192 in Session 57), plus `stop_tokens: ["<|im_end|>"]`, backup `working_dir`, and
> `routing.request_timeout_seconds: 180` (`llm_services/llm_server_config.json:2-28`). The router reads
> `stop_tokens` and the timeout (`llm_router.py:138-160`).

### 3.3 Server Startup

**File:** `llm_services/start_llm_servers.sh`

```bash
# Start server
cd ~/distributed_prng_analysis
./llm_services/start_llm_servers.sh

# Output:
# ✅ Primary model found: DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf
# ✅ Primary (DeepSeek-R1-14B) (port 8080): HEALTHY
# ✅ Claude Code CLI: AVAILABLE

# Stop server
pkill -f 'llama-server.*port 8080'
```

> **Corrected 2026-11-11 (D10).** The script prints `✅ Primary model found: …`, then
> `Waiting for health check... ready! (Ns)`, then `✅ Claude Code CLI: AVAILABLE` and a summary block
> (`llm_services/start_llm_servers.sh:41`, `:69-91`, `:95-106`); there is no "HEALTHY" line. The script
> **hardcodes** `PORT=8080`, `CTX_SIZE=32768`, `N_GPU_LAYERS=99` (`:28-30`) and never reads the
> `CONFIG_FILE` it defines (`:20` is the only occurrence), so the config file and the script are two separate
> sources of these values. The stop pattern matches the server command line (`:55-58`, `--port 8080`); the
> script's own advice is `pkill -f 'llama-server'` (`:106`). The script and config assume a dual-3080-Ti
> partition (`:9`, `:100`; config `:10`); on 2026-11-11 `nvidia-smi` on VM 101 showed one RTX 3080 Ti
> (point-in-time host state). The server was not running at that check.

### 3.4 LLM Router

**File:** `llm_services/llm_router.py` (v2.0.0)

```python
class LLMRouter:
    """Routes requests to primary LLM with backup escalation."""
    
    def route(self, prompt: str, force_backup: bool = False) -> str:
        """Route request to primary, escalate to backup if needed."""
        
        if force_backup:
            return self._call_backup(prompt)
        
        response = self._call_primary(prompt)
        
        if self._should_escalate(response):
            return self._call_backup(prompt)
        
        return response
```

> **Corrected 2026-11-11.** The routing logic is accurate (`llm_router.py:112-136`; `route` also takes
> `temperature` and `max_tokens`). The router is **v2.1.0** (`:3`), which restored the v1.0.5 method API
> and added `evaluate_watcher_decision` (grammar-constrained, `:375-430`) and `evaluate_with_grammar`
> (`:441-492`). A failed primary call also falls through to the backup (`:170-178`); if JSON parsing fails
> the router returns a safe default of `"decision": "escalate"` (`:358-363`).

---

## 4. Watcher Agent

### 4.1 CLI Reference

```bash
cd ~/distributed_prng_analysis
PYTHONPATH=. python3 agents/watcher_agent.py --help

Options:
  --daemon              Run as daemon, watching for results
  --run-pipeline        Run the full pipeline
  --start-step N        Starting step for pipeline (1-6)
  --end-step N          Ending step for pipeline (1-6)
  --evaluate FILE       Evaluate a single result file
  --status              Show watcher status
  --clear-halt          Clear the halt file to resume
  --halt                Create halt file to stop watcher
  --no-llm              Disable LLM evaluation, use heuristics only
  --no-grammar          Disable grammar-constrained decoding
  --threshold N         Auto-proceed confidence threshold (default: 0.70)
```

> **Corrected 2026-11-11 (D11).** The eleven options above exist as described (`agents/watcher_agent.py:3470-3529`
> at a15b524). The list is incomplete: argparse also defines `--params` (JSON overrides, step-scoped),
> `--force-step N` (repeatable), `--episodes`, `--dispatch-selfplay`, `--dispatch-learning-loop [SCOPE]`,
> `--process-requests`, `--dry-run`, `--stop`, `--explain [N]` and `--approve` (`:3530-3565` at a15b524).
> `--halt` takes an optional reason string; `--start-step` accepts `0` (TRSE). WATCHER is documented in full in
> `docs/CHAPTER_12_WATCHER_AGENT.md` §9. At `a15b524`, `--dry-run` was ignored by `--run-pipeline`
> (`docs/CHAPTER_12_AUDIT.md` B6) — **repaired at `b6cb789`**: `--dry-run` with `--run-pipeline`, `--approve`
> or `--daemon` now refuses with exit 2 (`docs/CLAUDE_CODE_REPORT_WATCHER_SAFETY_REPAIRS.md` R-2). `--halt`
> writes `/tmp/agent_halt`; at `a15b524` dispatch read a different file, `watcher_halt.flag`
> (`docs/CHAPTER_12_AUDIT.md` B3) — **repaired at `b6cb789`** (R-3: dispatch checks both through one helper).

### 4.2 Evaluation Modes

| Mode | Command | Description |
|------|---------|-------------|
| Single file | `--evaluate file.json` | Evaluate one output |
| Pipeline | `--run-pipeline` | Run full pipeline |
| Daemon | `--daemon` | Watch for new outputs |
| Heuristic only | `--no-llm` | Skip LLM, use rules |

### 4.3 Decision Flow

```python
# In watcher_agent.py
def _evaluate_with_llm(self, context):
    """Evaluation hierarchy:
    1. Try LLMRouter with grammar (if available)
    2. Try HTTP direct call
    3. Fall back to heuristic
    """
    
    # Try grammar-constrained evaluation
    if self.llm_router and self.config.use_grammar:
        return self._evaluate_with_router(context)
    
    # Try HTTP fallback
    return self._evaluate_with_http(context)
```

> **Corrected 2026-11-11 (D12).** At `a15b524` the order is: (0) `evaluate_results` returns the
> file-existence verdict for every manifest with `evaluation_type: file_exists` — all seven — before any of
> the following (`agents/watcher_agent.py:946-1004`; absorbed into `docs/CHAPTER_12_AUDIT.md` D3); (1) for
> Step 1, `_evaluate_step1_with_new_prompt` (`:1047-1058`, prompt from `agents/prompts/step1_prompt.py`,
> grammar `grammars/watcher_decision.gbnf`); (2) `_evaluate_with_router` (`:1059-1065`); (3)
> `_evaluate_with_http` (`:1068`); (4) if those return nothing, `evaluate_results` falls back to
> `_evaluate_heuristic` (`:1027-1037`, `:1269-1308`). All line numbers at a15b524.

### 4.4 Decision Actions

| Action | Confidence | Behavior |
|--------|------------|----------|
| PROCEED | ≥ 0.70 | Trigger next step automatically |
| RETRY | 0.50 - 0.70 | Re-run current step with adjustments |
| ESCALATE | < 0.50 | Alert human for review |

> **Corrected 2026-11-11.** The thresholds are accurate (`auto_proceed_threshold = 0.70`,
> `escalate_threshold = 0.50`, `agents/watcher_agent.py:328-329`; applied in the heuristic at `:1283-1291`,
> at a15b524). The heuristic's RETRY needs `success` false and retries remaining. "With adjustments" is not
> what happens: `_handle_retry` logs the suggested adjustments and re-runs with the same parameters —
> absorbed into `docs/CHAPTER_12_AUDIT.md` B5 (dormant: no RETRY is produced under `file_exists` manifests).

---

## 5. Step Contexts

### 5.1 Design Philosophy

> **"LLM does the reasoning, not this file."**
> — window_optimizer_context.py header

Step contexts are responsible for:
- ✅ Extract raw metrics from results
- ✅ Compute derived metrics (rates, ratios)
- ✅ Load threshold priors from config
- ✅ Package data for LLM evaluation

NOT responsible for:
- ❌ Semantic interpretation ("good", "bad")
- ❌ Decision making (proceed/retry/escalate)

### 5.2 Available Contexts

| Context | Step | Key Metrics |
|---------|------|-------------|
| `window_optimizer_context.py` | 1 | bidirectional_count, forward_count, rates |
| `scorer_meta_context.py` | 2.5 | best_accuracy, completed_trials |
| `full_scoring_context.py` | 3 | scored_count, coverage_pct |
| `ml_meta_context.py` | 4 | best_r2, validation_r2, model comparisons |
| `anti_overfit_context.py` | 5 | test_r2, overfit_ratio, signal_quality |
| `prediction_context.py` | 6 | prediction_confidence, pool_coverage |

> **Corrected 2026-11-11 (D13).** The design statement in §5.1 is accurate (`window_optimizer_context.py:8`).
> The key metrics in this table do not match what the contexts read (`.get(` keys in each file):
> Step 1 `bidirectional_count`, `forward_count`, rates (accurate); Step 2 (executable; "2.5" above)
> `best_validation_score`, `total_trials`, `best_k_folds`, `best_threshold`, `cv_std`; Step 3
> `survivors_scored`, `survivors_total`, `completion_rate`, `mean_score`; Step 4 `architecture_score`,
> `validation_loss`, `best_layers`, `best_neurons`; Step 5 `overfit_ratio`, `kfold_mean`, `kfold_std`,
> `best_epoch`, `early_stopped`; Step 6 `pool_size`, `coverage_pct`, `mean_confidence`, `diversity_score`.
> None reads `best_accuracy`, `best_r2`, `validation_r2`, `test_r2`, `signal_quality` or `pool_coverage`.
> Step 0 reuses the Step-1 context (`agents/contexts/__init__.py:31`, `:42`).

### 5.3 Context Structure

**File:** `agents/contexts/window_optimizer_context.py` (v4.0.0)

```python
class WindowOptimizerContext(BaseAgentContext):
    """
    Specialized context for Step 1: Window Optimizer.
    
    REFACTORED RESPONSIBILITIES:
    - Extract raw metrics from results
    - Compute derived metrics (rates, ratios)
    - Load threshold priors from config
    - Package for LLM evaluation
    """
    
    agent_name: str = "window_optimizer_agent"
    pipeline_step: int = 1
    step_id: str = "step_1_window_optimizer"
```

> **Verified 2026-11-11.** Accurate: `agents/contexts/window_optimizer_context.py:27`, `:43-45`, v4.0.0 (`:10`).

---

## 6. Doctrine (Decision Framework)

### 6.1 Purpose

The doctrine defines consistent decision-making rules across all steps.

**File:** `agents/doctrine.py` (v3.2.0)

```python
def get_doctrine() -> Dict[str, Any]:
    return {
        "version": "1.0.0",
        "purpose": "PRNG pattern analysis through systematic pipeline execution",
        
        "decision_framework": {
            "proceed": {
                "conditions": [
                    "success_condition_met = true",
                    "confidence >= 0.70",
                    "no critical anomalies detected"
                ]
            },
            "retry": {
                "conditions": [
                    "success_condition_met = false",
                    "retries_remaining > 0",
                    "adjustable parameters available"
                ]
            },
            "escalate": {
                "conditions": [
                    "confidence < 0.50",
                    "retries_exhausted",
                    "critical anomaly detected"
                ]
            }
        }
    }
```

> **Corrected 2026-11-11 (D14).** The snippet abbreviates `agents/doctrine.py:18-96`. In the code, RETRY
> also requires "no safety violations" (`:46`); ESCALATE's conditions are "confidence < 0.50", "anomalies
> detected", "retries exhausted", "safety concerns" (`:51-56`); the dict also carries
> `confidence_calibration`, `safety_priorities`, `parameter_adjustment_rules` and `output_requirements`
> (`:61-95`). Module v3.2.0, `DOCTRINE_VERSION = "1.0.0"` (`:8`, `:15`). **What the code enforces:** the
> doctrine is prompt text (`agents/full_agent_context.py:155`, `:219`). `validate_decision_against_doctrine`
> (`:113-174`) is imported by WATCHER (`agents/watcher_agent.py:151` at a15b524) and not called there; only
> `test_subphase4.py` calls it.

---

## 7. Grammar System

### 7.1 Grammar Loader

**File:** `llm_services/grammar_loader.py` (v1.0.0)

```python
class GrammarType(str, Enum):
    """Available grammar types for constrained decoding."""
    AGENT_DECISION = "agent_decision"
    SIEVE_ANALYSIS = "sieve_analysis"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    JSON_GENERIC = "json_generic"
```

### 7.2 Grammar File Location

Grammar files should be in: `agent_grammars/` (referenced but needs creation)

```
agent_grammars/
├── agent_decision.gbnf       # Proceed/retry/escalate decisions
├── sieve_analysis.gbnf       # Sieve result interpretation
├── parameter_adjustment.gbnf # Parameter change suggestions
└── json_generic.gbnf         # Fallback for any valid JSON
```

> **Corrected 2026-11-11 (D15, D6).** The `GrammarType` enum in §7.1 is accurate
> (`llm_services/grammar_loader.py:29-38`; the module is v1.1.0, `:5`). `agent_grammars/` **exists** and is
> tracked, with five files (the four above plus `chapter_13.gbnf`); `grammar_loader.py` resolves it
> independent of CWD (`:41-54`). **There is a second grammar directory, `grammars/`** (tracked, 8 files:
> `agent_decision`, `chapter_13`, `diagnostics_analysis`, `json_generic`, `parameter_adjustment`,
> `sieve_analysis`, `strategy_advisor`, `watcher_decision`). The router loads grammars from `grammars/`
> relative to the CWD (`llm_router.py:235`, `:252`, `:475`), and so does the Chapter-13 advisor
> (`chapter_13_llm_advisor.py:309-312`). The two `chapter_13.gbnf` copies differ
> (`docs/CHAPTER_13_AUDIT.md:262`).

### 7.3 Decision Grammar

```gbnf
root ::= "{" ws "\"decision\"" ws ":" ws decision-value ws "," ws 
         "\"confidence\"" ws ":" ws number ws "," ws 
         "\"reasoning\"" ws ":" ws string ws "}"

decision-value ::= "\"proceed\"" | "\"retry\"" | "\"escalate\""
string ::= "\"" ([^"\\] | "\\" .)* "\""
number ::= "0." [0-9]+
ws ::= [ \t\n]*
```

> **Corrected 2026-11-11 (D16).** Neither grammar file reads like this. `agent_grammars/agent_decision.gbnf:1-14`
> has `success_condition_met` (boolean), `confidence`, `reasoning`, `recommended_action`, with
> `number ::= [0-9] "." [0-9] [0-9]`. The grammar WATCHER's router actually applies to step decisions is
> **`grammars/watcher_decision.gbnf`** (`llm_router.py:410-413`), whose root object is `decision`,
> `retry_reason`, `confidence`, `reasoning`, `primary_signal`, `suggested_params`, `warnings`, and a `checks`
> object (`used_rates`, `mentioned_data_source`, `avoided_absolute_only`).

---

## 8. Usage Examples

### 8.1 Evaluate Single Step Output

```bash
cd ~/distributed_prng_analysis

# Start LLM server
./llm_services/start_llm_servers.sh

# Evaluate Step 1 output
PYTHONPATH=. python3 agents/watcher_agent.py --evaluate optimal_window_config.json

# Output:
# Step: 1 - Window Optimizer
# Success: True
# Confidence: 0.95
# Action: proceed
```

### 8.2 Run Without LLM (Heuristic Only)

```bash
PYTHONPATH=. python3 agents/watcher_agent.py --evaluate optimal_window_config.json --no-llm

# Output:
# Using heuristic evaluation (LLM unavailable)
# Action: proceed
# Confidence: 0.95
```

> **Corrected 2026-11-11 — §8.1–§8.2 expected output.** At `a15b524` both commands print
> `Parse method: file_exists` and `Confidence: 1.00` (or `0.00` with `Action: escalate` if the file is invalid);
> `--no-llm` changes nothing for Step 1, because the file-existence verdict comes first
> (`agents/watcher_agent.py:946-1004`, `:3782-3803` at a15b524). Absorbed into `docs/CHAPTER_12_AUDIT.md` D3/D4.

### 8.3 Run Full Pipeline

```bash
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 1 --end-step 6
```

> **Note 2026-11-11.** The syntax of §8.3 and §8.4 is valid (argparse at a15b524). `--run-pipeline` launches
> the fleet: it is **Michael-initiated only**, always under `nohup`. Steps 2–6 on RANGE-MINER output are held
> (`docs/CHAPTER_12_AUDIT.md` S2); the current launch rule for Step 1 is `--end-step 1`.

### 8.4 Run as Daemon

```bash
PYTHONPATH=. python3 agents/watcher_agent.py --daemon
```

---

## 9. Troubleshooting

### 9.1 "No module named 'agents'"

```bash
# Always run from project root with PYTHONPATH
cd ~/distributed_prng_analysis
PYTHONPATH=. python3 agents/watcher_agent.py --help
```

### 9.2 "Connection refused" on port 8080

```bash
# Start the LLM server first
./llm_services/start_llm_servers.sh

# Verify health
curl http://localhost:8080/health
```

### 9.3 LLM Returns Invalid JSON

The system has multiple fallbacks:
1. Grammar-constrained evaluation (if LLMRouter available)
2. HTTP direct call with JSON extraction
3. Heuristic fallback (always works)

> **Corrected 2026-11-11.** §9.1 is accurate: `agents/watcher_agent.py` imports `agents.*` at module level
> (`:75` at a15b524) with no root insertion into `sys.path` before it. §9.2 is accurate: the server exposes
> `/health` (`start_llm_servers.sh:44`, `:72`). §9.3's list is absorbed into the §4.3 note (D12); none of
> the three is reached for a step whose manifest is `file_exists`.

---

## 10. Known Gaps

| Gap | Impact | Priority |
|-----|--------|----------|
| `agent_grammars/` directory missing | Grammar files referenced but not created | Medium |
| On-demand LLM lifecycle | Server stays running vs start/stop per query | Low |
| MAIN_MISSION in doctrine.py | No global mission context | Low |

> **SUPERSEDED (2026-11-11) — all three gaps are closed; replaced by §14 "Known gaps at a15b524".**
> - `agent_grammars/` exists (5 files) (D17).
> - On-demand lifecycle exists: `llm_services/llm_lifecycle.py` v1.0.0 (`:1-28`); dispatch stops the server for
>   GPU-heavy work and restarts it (`agents/watcher_dispatch.py:131-133`, `:202-204`, `:309-332` at a15b524);
>   WATCHER uses `session()` for Step-5 diagnostics (`agents/watcher_agent.py:2405-2420` at a15b524) (D18).
> - A global mission exists, though not in `doctrine.py`: `MAIN_MISSION` in `agents/contexts/bundle_factory.py:76`,
>   rendered first in every bundle (`:756`); WATCHER's Step-1 prompt uses `GLOBAL_MISSION`
>   (`agents/prompts/global_mission.py:6`, via `agents/prompts/step1_prompt.py:14`, `:49`) (D19). The two texts
>   differ: `MAIN_MISSION` says functional mimicry, "rather than attempting to discover or reconstruct actual
>   seeds"; `GLOBAL_MISSION` says the system "reverse-engineers PRNG behavior by testing candidate seeds".
>   Code-side text, listed for the owner (§15), not edited.

---

## 11. Version History

```
Version 3.0.0 - January 8, 2026
- VALIDATED: Live testing confirmed all components working
- DOCUMENTED: Existing codebase structure
- REMOVED: Duplicate code proposals (test_watcher_clean.py)

Version 2.0.0 - January 7, 2026
- Updated to DeepSeek-R1-14B primary + Claude backup
- A/B testing results documented

Version 1.0.0 - December 2025
- Initial Dual-LLM architecture (Qwen models)
```

> **Added 2026-11-11 (D20).** The history had no 3.1.0 entry. Recorded from git:
> - **3.1.0** — 2026-02-03 header; edits in `6a3aefa` (2026-02-02), `e966e81` (2026-02-11), `2120b7d`
>   (2026-03-10, S136 TRSE appendix). Added §1.1 rows for dispatch, bundle factory, lifecycle.
> - **3.2.0** — 2026-11-11, audited and corrected against `a15b524` (`docs/CHAPTER_10_AUDIT.md`). Added §0,
>   §13–§17, framing block, status header; corrections inline; no original line removed.

---

## 12. References

| File | Purpose |
|------|---------|
| `agents/watcher_agent.py` | Main orchestrator |
| `agents/watcher_dispatch.py` | Dispatch: selfplay, learning loop, Ch13 requests |
| `agents/contexts/bundle_factory.py` | Unified LLM context assembly (7 bundle types) |
| `agents/doctrine.py` | Decision framework |
| `agents/contexts/*.py` | Step-specific contexts |
| `llm_services/llm_router.py` | LLM routing logic |
| `llm_services/llm_lifecycle.py` | LLM server lifecycle management |
| `llm_services/grammar_loader.py` | Grammar management |
| `llm_services/start_llm_servers.sh` | Server startup (v2.1.0) |
| `agent_manifests/*.json` | Step configurations |
| `agent_grammars/*.gbnf` | Grammar constraint files (v1.1) |

> **Corrected 2026-11-11.** All files listed exist. `bundle_factory.py` assembles **8** bundle types, not 7
> (steps 1–6, 13, 99; `agents/contexts/bundle_factory.py:587-596`) — absorbed into `docs/CHAPTER_12_AUDIT.md`
> D2. Add `grammars/*.gbnf` (the directory the router loads, §7 note) and `agents/registry/parameter_registry.py`
> (§13.1).

---

### Policy Configuration

WATCHER behavior is governed by `watcher_policies.json`. Key flags:

| Flag | Values | Purpose |
|------|--------|--------|
| `approval_route` | `"orchestrator"` / `"watcher"` | Who owns execution authority |
| `test_mode` | boolean | Master switch for test mode |
| `auto_approve_in_test_mode` | boolean | WATCHER auto-approves (requires test_mode) |

> **📋 Complete Policy Reference:** See `docs/WATCHER_POLICIES_REFERENCE.md` for all flags, safety invariants, and production vs test configurations.

> **Verified 2026-11-11 — what the code enforces.** The three flags exist in `watcher_policies.json` (live
> values: `approval_route: "orchestrator"`, `test_mode: false`, `auto_approve_in_test_mode: false`) and are
> read by `chapter_13_orchestrator.py:403-425` and `agents/watcher_agent.py:3314-3326` (at a15b524).
> `approval_route` selects which software path handles an accepted Chapter-13 proposal: `"watcher"` routes it
> to the WATCHER daemon's approval queue; `"orchestrator"` auto-executes only when `test_mode` and
> `auto_approve_in_test_mode` are both true, and otherwise raises a human approval request
> (`request_approval`, `:410-425`). With the live values every accepted proposal raises a human approval
> request. The code encodes no review body. `docs/WATCHER_POLICIES_REFERENCE.md` exists (not re-audited).


### Parameter Bounds Authority

**Critical:** Agent manifests (`agent_manifests/*.json`) define parameter metadata and documentation only.
Runtime bounds for Optuna search are defined exclusively in `distributed_config.json`.

| Source | Role |
|--------|------|
| `distributed_config.json` | **Authoritative** - Runtime search bounds |
| `agent_manifests/*.json` | Documentation only - NOT runtime bounds |
| `baselines/baseline_window_thresholds.json` | Recovery baseline |

See `docs/THRESHOLD_GOVERNANCE.md` for full governance model.

> **Corrected 2026-11-11.** Accurate on the WATCHER path: manifest `parameter_bounds` are read only by
> `agents/step_runner/command_builder.py:151`, and nothing imports `step_runner` (§2.2 note). The baseline
> file and `docs/THRESHOLD_GOVERNANCE.md` exist. DESCRIPTION-ONLY (D21): "exclusively" overstates it —
> `load_search_bounds_from_config` merges `distributed_config.json` `search_bounds` **over a hardcoded defaults
> dict** that applies when a key is absent or the file is unreadable (`window_optimizer.py:67-92`); retiring the
> `offset` default is Brief II item C-4 (`docs/S172_WINDOW_ANCHOR_BRIEF_I_REPORT.md:700-704`). The table also
> omits a third bounds source, the agent parameter registry, which is what the LLM is shown as "legal
> bounds" (§13.1, B1).

---

**END OF CHAPTER 10**

> **Note 2026-11-11.** This marker is retained from v3.1.0. The TRSE appendix below and the 2026-11-11
> sections §13–§17 follow it and are part of the chapter.

---

## Step 0 Integration: TRSE (S121/S122)

**Module:** `trse_step0.py` v1.15.1
**Invariant:** skip_on_fail=True -- TRSE failure NEVER halts pipeline.

> **Verified 2026-11-11.** `trse_step0.py` `__version__ = "1.15.1"` (`:107`; its docstring still says 1.15.0,
> `:5`). `agent_manifests/trse.json` sets `skip_on_fail: true` (tracked since `93918f5`); a missing Step-0
> output is recorded as `proceed` with confidence 0.0 (`agents/watcher_agent.py:966-987` at a15b524). The
> `b6cb789` repair keeps this precedence for `skip_on_fail` manifests even when the step exits non-zero, and
> adds a warning (`docs/CLAUDE_CODE_REPORT_WATCHER_SAFETY_REPAIRS.md` R-1, gate R1-E). Accurate.
> **But see `docs/TRSE_STEP0_AUDIT_v1.md` F1** (absorbed, not re-reported): WATCHER's Step-0 command is
> malformed, still at `a15b524` — `trse.json` `default_params` carries `window_size`, `stride` and
> `trse_context`, which `trse_step0.py` does not define (`:811-816`), so a Step-0 run that is not skipped as
> fresh exits 2 and is then passed over by `skip_on_fail`. The Pydantic `ge=0` change below is real; dispatch
> through WATCHER does not succeed while F1 stands.

### confirmed_windows Feedback Loop

After Step 1 produces survivors, their window configs are written back to
`trse_context.json` as `confirmed_windows`. TRSE uses these on subsequent runs
to bias regime boundary detection.

> **Corrected 2026-11-11 — absorbed into `docs/TRSE_STEP0_AUDIT_v1.md` F3.** The write-back exists
> (`window_optimizer.py:1055-1084`, last 50 entries kept, only when `bidirectional_count > 0`). **TRSE does not
> read `confirmed_windows`**: the string does not occur in `trse_step0.py`, and no other reader exists in the
> tree (`/bin/grep -rln confirmed_windows` → the writer, its backups, a test, a patch script and docs). The
> stated feedback loop is unrealised; `save_context` overwrites the file without merging, so a successful
> Step-0 run would drop the accumulated entries (F3). Fix proposed, not implemented:
> `docs/TEAM_ALPHA_TRSE_FIX_PROPOSAL.md`.

### regime_type Fix (v1.15.1, S122)

Previous versions used absolute normalization causing `regime_type=unknown` on most
segments. v1.15.1 switched to relative normalization -- regime classification functional.

> **Verified 2026-11-11.** Relative normalization is in `classify_regime_type` (`trse_step0.py:397`,
> `:422-455`; `__version__` comment `:107`). Accurate as code; "functional" was not checked by a run.

### Pydantic ge=1 -> ge=0 Fix (S122)

Five agent files had `step: int = Field(ge=1)` which rejected step=0 dispatch.
All five fixed to `ge=0` so TRSE can be dispatched through WATCHER.

> **Corrected 2026-11-11 (D22).** Commit `b2a89d8` (S122) changed **four fields in three files** to `ge=0`:
> `agents/full_agent_context.py:42` (`step`), `agents/manifest/agent_manifest.py:154-158` (`pipeline_step`),
> `agents/pipeline/pipeline_step_context.py:28` (`step_number`) and `:167` (`current_step`). It also added
> Step 0 to `agents/contexts/__init__.py` (`:31`, `:42`, reusing the Step-1 context) and to
> `agents/progress_display.py` — five agent files touched in all. `agents/contexts/base_agent_context.py:53`
> still has `pipeline_step … ge=1`; that is harmless because Step 0 builds a `WindowOptimizerContext`, whose
> `pipeline_step` is `1` (`window_optimizer_context.py:44`). "Can be dispatched": see the F1 note above.

### Survivor Threshold (S122)

WATCHER validation threshold lowered from >=100 to >=50 survivors to prevent
false ESCALATE with real data on Steps 1 and 3.

> **Verified / corrected 2026-11-11.** The minimum is 50 for `bidirectional_survivors*.json`,
> `forward_survivors*.json`, `reverse_survivors*.json` and `survivors_with_scores*.json`
> (`FILE_VALIDATION_CONFIG["json_array_minimums"]`, `agents/watcher_agent.py:169-172` at a15b524) — accurate.
> DESCRIPTION-ONLY (D23): Step 1's `success_condition` is `optimal_window_config.json` only (manifest parse), so
> at `a15b524` the survivor-array minimum is reached by Step 3's `survivors_with_scores.json`, not by Step 1.

### WATCHER CLI with TRSE

```bash
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline \
  --start-step 0 --end-step 6 \
  --params '{"lottery_file":"daily3.json","trials":200,
             "resume_study":true,"study_name":"window_opt_1772507547",
             "trse_context":"trse_context.json"}'
```

> **Corrected 2026-11-11 (D24).** `--params` is step-scoped: a key reaches a step only if that step's manifest
> declares it in `default_params` (`agents/watcher_agent.py:1811-1820` at a15b524; `docs/CHAPTER_12_AUDIT.md`
> D20). `lottery_file`, `resume_study`, `study_name` and `trse_context` are declared in `window_optimizer.json`;
> **`trials` is not** and is silently dropped — the declared key is `window_trials`. This is a fleet launch:
> **Michael-initiated only**, under `nohup`. Steps 2–6 on RANGE-MINER output are held
> (`docs/CHAPTER_12_AUDIT.md` S2), so the current launch rule is `--end-step 1`.

---

## 13. Agent-framework components not described above (added 2026-11-11)

### 13.1 Parameter registry — `agents/registry/parameter_registry.py`

Purpose (`:1-9`): "Enables LLM introspection of legal parameter bounds." `PARAMETER_REGISTRIES` (`:263`) holds
one `ScriptParameterRegistry` per script, 29 `ParameterSpec`s in all. `ParameterContext`
(`agents/parameters/parameter_context.py:177-195`) iterates the manifest's primary-action `args_map` and looks
each flag up here; the result goes into `FullAgentContext` and the LLM prompt.

**BROKEN (B1, dormant).** 20 of the 29 flags do not exist in the scripts they name (static comparison with
each script's `add_argument` calls): `window_optimizer.py` `--window-size`, `--offset`, `--skip-min`,
`--skip-max`; `scorer_meta_optimizer.py` and `full_scoring.py` (7 flags) — **neither script exists or ever
existed** (`git log --all --` empty); `adaptive_meta_optimizer.py` `--trials`, `--min-layers`, `--max-layers`,
and `--mode` choices `quick/full/deep` against the real `full/survivor/network/epochs`
(`:437-478` vs `adaptive_meta_optimizer.py:1101-1116`; the entry also calls Step 4 "Optimize ML model
architecture using Optuna"); `meta_prediction_optimizer_anti_overfit.py` `--epochs`, `--dropout-min`,
`--dropout-max`, `--distributed`; `prediction_generator.py` `--confidence-threshold`, `--diversity-weight`.
Effective reach today is the Step-4 `--mode` spec (the only stale entry whose flag is in a manifest
`args_map`), and only on the LLM path, which `file_exists` manifests bypass (§1 note). Repair must make each
entry match its script, or drop entries for scripts that do not exist.

**SWITCHED-OFF-BY-RULING (S1).** The Step-1 `offset` entry (`:282-289`, `max_value=500`) is designed to become
`window_anchor` with no static maximum (`docs/PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md` §4.6,
`:156-161`). Brief I left it for Brief II (`docs/S172_WINDOW_ANCHOR_BRIEF_I.md:360-361`, `:675`). Lifting that
hold would let the registry describe the anchor with a dynamic maximum delegated to the resolver. The entry is
not reached today (`--offset` is in no manifest `args_map` and is not a flag of `window_optimizer.py`).

### 13.2 `agents/step_runner/`

v1.0.0 "modular step automation framework" (`agents/step_runner/__init__.py:1-30`; added `15bf8c2`,
2026-01-02; design `docs/PROPOSAL_Modular_Step_Automation_Framework_v1_1.md`, status DRAFT). No importer outside
the package. It is the only reader of manifest `parameter_bounds` (`command_builder.py:151`). Recorded as a
separate dispatch surface outside the certified Run-4 route
(`docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:133`).

### 13.3 `agent_contexts/`

`context_loader.py` v1.0.0 and seven step templates (`89b6087`, 2026-01-07). No importer. It uses the
**conceptual** numbering (`STEP_TEMPLATES`: `2` = bidirectional sieve, `2.5` = scorer meta,
`agent_contexts/context_loader.py:15-23`), unlike the rest of the framework.

### 13.4 `agent_manifests/definitions.json` (gitignored)

6,635 B, mtime 2026-01-01, ignored by `.gitignore:41` (`*.json`), never tracked. Schema 1.1.0: `pipeline_steps`
(Step 5 emits `signal_quality`, `data_context`, `feature_schema`; Step 6 consumes them), `watcher_protocol`
(degenerate-fingerprint registry at `agent_state/degenerate_fingerprints.json`, a five-step recovery workflow,
fingerprint = sha256, truncated to 8, of `{train_start}:{train_end}|{holdout_start}:{holdout_end}|{survivor_count}|{prng_type}|{mod}`),
`sidecar_schema`. **No code reads it**; `agent_state/` does not exist on the host; the producer half exists
(`fingerprint_hash` in `meta_prediction_optimizer_anti_overfit.py` and `prediction_generator.py`). Same finding
as `docs/CHAPTER_12_WATCHER_AGENT.md` §6.4.

### 13.5 Fingerprint registry and hooks

`agents/fingerprint_registry.py`, `agents/watcher_registry_hooks.py`: exist, not called by WATCHER. See
`docs/CHAPTER_12_AUDIT.md` B2.

---

## 14. Known gaps at a15b524

Full detail and anchors: `docs/CHAPTER_10_AUDIT.md` §4–§6.

### 14.1 BROKEN

- **B1 — parameter registry advertises parameters the scripts do not accept** (§13.1).
  `agents/registry/parameter_registry.py:437`. Dormant.
- Cross-referenced, not new: Ch12 **B1** file-exists verdict (repaired at `b6cb789`), **B2** fingerprint
  registry unwired, **B3** two halt files (repaired at `b6cb789`), **B5** RETRY adjustments not applied,
  **B6** `--dry-run` with `--run-pipeline` (repaired at `b6cb789`) — `docs/CHAPTER_12_AUDIT.md` §4. TRSE
  **F1** (Step-0 command malformed) and **F3** (`confirmed_windows` write-only) — `docs/TRSE_STEP0_AUDIT_v1.md` §8.

### 14.2 SWITCHED-OFF-BY-RULING

- **S1 — registry `offset` entry not yet `window_anchor`** (§13.1). Held by
  `docs/S172_WINDOW_ANCHOR_BRIEF_I.md:360-361` (Brief II). Lifting it would let the registry describe the
  anchor with a dynamic maximum. No recommendation; per the 2026-11-11 owner ruling this is a review position
  the owner may keep or lift.

### 14.3 DESCRIPTION-ONLY

D1 header "Phase 7 Complete" · D2 WATCHER v2.0.0 · D3 bundle factory v1.1.0 · D4 router v2.1.0 · D5 grammar
loader v1.1.0 · D6 grammar file count and the second `grammars/` directory · D7 file tree omissions · D8
`step_runner/` not used for execution · D9 server config 2.1.0 / 32768 · D10 startup output; script ignores the
config file · D11 CLI list incomplete · D12 evaluation order · D13 context key metrics · D14 doctrine snippet
abbreviated · D15 `agent_grammars/` exists · D16 decision grammar fields · D17–D19 the three v3.1.0 known gaps
are closed · D20 no 3.1.0 history entry · D21 bounds not "exclusively" from config · D22 `ge=0` count · D23
survivor minimum reaches Step 3, not Step 1 · D24 `trials` dropped from the CLI example.

---

## 15. Open items

1. B1 repair (owner choice: fix the registry or narrow its claim).
2. S1 is Brief II scope.
3. Three claims stay **[UNVERIFIED]**: U1 "~85% autonomy" (no definition), U2 "fully working" (needs a run with
   a non-`file_exists` manifest and a live server), U3 51 / 38 tok/s (needs a benchmark log).
4. TRSE F1 and F3 are open at `a15b524` (`docs/TRSE_STEP0_AUDIT_v1.md`; fix proposal not implemented).
5. Code-side text listed for the owner, not edited: `agents/prompts/global_mission.py:6` (seed-recovery
   phrasing in a live Step-1 prompt); `trse_step0.py:5` version string; `start_llm_servers.sh:9`, `:100` and
   `llm_server_config.json:10` assume two 3080 Ti; `start_llm_servers.sh` does not read its config file.
6. Two grammar directories (`grammars/`, `agent_grammars/`) with diverging `chapter_13.gbnf` copies — which is
   canonical is undecided in code.
7. Not re-audited here: `docs/WATCHER_POLICIES_REFERENCE.md`, `docs/THRESHOLD_GOVERNANCE.md`; the WATCHER
   behaviour changed at `b6cb789` and `cf9edaa` (Chapter 12's scope).

---

## 16. Verification declaration (VIR-1…6)

**Execution proof (VIR-1).** Every anchor in the 2026-11-11 notes was opened on VM 101 in this session with
`Read`, `sed -n`, `git show a15b524:<path>` or `/bin/grep -n`. The file list is in `docs/CHAPTER_10_AUDIT.md`
§8. Nothing was executed except read-only host queries, JSON parses of the manifests and policies, and one
`python3 -I` regex pass over the registry source (no import of `agents`).

**Clean control (VIR-2).** `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` was read, not modified. Claims verified
accurate and left standing are listed in the audit §9 (28 items).

**Fault-injection control.** **Not applicable** to a documentation pass; none was invented.

**Completion sentinel (VIR-3).** §17.

**Unavailable-observer behaviour (VIR-5).** `/tmp` (halt files, LLM server log), rig/CT100 state, any result
that needs a run (live LLM evaluation, throughput, `test_subphase*.py`, `agents/step_runner/test_step_runner.py`)
and Beta texts held outside the repo are **UNAVAILABLE**, never clean.

**Audit claim scope (VIR-6).** What this chapter says against what the code at `a15b524` does — not whether
the agent framework's design is right. Searched surfaces: tracked tree; gitignored/untracked files
(`git ls-files -o -i --exclude-standard`, `agent_manifests/definitions.json`, `_v3.md.bak`); git history
(`git log --all -- <path>`, `git log -S`, commit inspection of `b2a89d8`, `6326257`); host (`curl`, `ls`,
`nvidia-smi`, `systemctl list-units --all`, `crontab -l`). Governance trail searched: `docs/TB_RULING_*`,
`docs/PROPOSAL_*`, `docs/TEAM_ALPHA_*`, the fact file, `docs/TODO_CONSOLIDATED.md`, `docs/TRSE_STEP0_AUDIT_v1.md`.
Chapters searched: Ch2 (structure), Ch5 audit, Ch12 chapter and audit, Ch13 audit, Ch14 audit, Ch1 chapter and
audit (grep), `docs/PIPELINE_BEHAVIOUR_MODEL.md`.

---

## 17. Closure statement

**Verified against** `a15b524` on VM 101, 2026-11-11. **Checked:** 66 claims — 28 accurate, 1 BROKEN,
1 SWITCHED-OFF-BY-RULING, 24 DESCRIPTION-ONLY, 3 UNVERIFIED, 9 absorbed into existing items
(28 + 1 + 1 + 24 + 3 + 9 = 66). **Added:** framing block, status header, §0, §13–§17, and inline
`Corrected` / `Verified` notes. **Superseded, kept:** §1.2 live test, §10 Known Gaps. **Removed:** nothing —
every pre-existing heading is present and no original line was deleted; the chapter grew.

**What this chapter is NOT:** not a certification that the agent framework works (no run was made), not an
operator runbook (nothing here is to be executed as written; `--run-pipeline` is Michael-initiated only), and
not a re-audit of WATCHER, which is Chapter 12's.

```
CHAPTER 10 CORRECTION:  PASS
```

`PASS` for the declared scope: every substantive claim audited and labelled, the chapter corrected in place
with a visible trail, RANGE-MINER / NPZ / window-anchor effects stated, gaps listed. It is not a claim that any
gap is repaired. Files changed: this chapter and `docs/CHAPTER_10_AUDIT.md` only.
