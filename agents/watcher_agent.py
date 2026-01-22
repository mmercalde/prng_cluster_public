#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules.*")
"""
Watcher Agent - Autonomous pipeline monitoring and orchestration.

The Watcher Agent monitors pipeline outputs, evaluates results using
the Pydantic Context Framework, and automatically triggers next steps
or escalates to human review.

Version: 1.1.0 (Grammar-Constrained LLM Integration)

Changes in v1.1.0:
- Integrated LLMRouter v1.1.0 with grammar-constrained decoding
- Uses router.evaluate_decision() for guaranteed JSON structure
- Fallback to direct HTTP if router unavailable
- Better error handling for LLM responses

Usage:
    # Start watcher daemon
    python3 watcher_agent.py --daemon

    # Evaluate a single result file
    python3 watcher_agent.py --evaluate /path/to/results.json

    # Run full pipeline from step 1
    python3 watcher_agent.py --run-pipeline --start-step 1

    # Check status
    python3 watcher_agent.py --status
"""

import argparse
import json
import os
import sys
import time
import subprocess
import signal
import fnmatch
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

# Try to import progress display (optional)
try:
    from progress_display import WatcherProgress, print_banner
    PROGRESS_DISPLAY_AVAILABLE = True
except ImportError:
    PROGRESS_DISPLAY_AVAILABLE = False
    WatcherProgress = None

# Try to import LLM Router v1.1.0 (preferred)
try:
    from llm_services.llm_router import LLMRouter, get_router, GrammarType
    LLM_ROUTER_AVAILABLE = True
except ImportError:
    LLM_ROUTER_AVAILABLE = False
    LLMRouter = None

# Pydantic Context Framework imports - use direct submodule imports to avoid circular dependency
from agents.full_agent_context import FullAgentContext, build_full_context
from agents.agent_decision import AgentDecision, parse_llm_response
from agents.history import AnalysisHistory, load_history
# Step-specific prompt builders (Team Beta Approved 2026-01-04)
try:
    from agents.prompts.step1_prompt import build_step1_prompt, build_step1_corrective_prompt
    from agents.threshold_guardrail import validate_decision, enforce_thresholds
    STEP_PROMPTS_AVAILABLE = True
except ImportError:
    STEP_PROMPTS_AVAILABLE = False
from agents.safety import KillSwitch, check_safety, create_halt, clear_halt
from agents.pipeline import get_step_info
from agents.doctrine import validate_decision_against_doctrine

# ════════════════════════════════════════════════════════════════════════════════
# FILE VALIDATION CONFIG (Team Beta Approved v1.1.0)
# ════════════════════════════════════════════════════════════════════════════════

FILE_VALIDATION_CONFIG = {
    "min_sizes": {
        ".json": 50,
        ".npz": 100,
        ".pth": 1000,
        ".pt": 1000,
        ".xgb": 1000,
        ".lgb": 1000,
        ".cbm": 1000,
        "default": 10
    },
    "json_array_minimums": {
        "bidirectional_survivors*.json": 100,
        "forward_survivors*.json": 100,
        "reverse_survivors*.json": 100,
        "survivors_with_scores*.json": 100,
        "train_history*.json": 10,
        "holdout_history*.json": 5,
    },
    "json_required_keys": {
        "optimal_window_config*.json": ["window_size", "offset"],
        "optimal_scorer_config*.json": ["residue_mod_1"],
        "best_model*.meta.json": ["model_type", "feature_schema"],
        "reinforcement_engine_config*.json": ["survivor_pool"],
        "predictions*.json": ["predictions"],
        "prediction_pool*.json": ["predictions"],
    }
}


def _get_min_file_size(filepath: str) -> int:
    """Get minimum expected file size based on extension."""
    ext = Path(filepath).suffix.lower()
    return FILE_VALIDATION_CONFIG["min_sizes"].get(
        ext, FILE_VALIDATION_CONFIG["min_sizes"]["default"]
    )


def _match_config_by_pattern(filename: str, table: dict):
    """Match filename against pattern-based config table using fnmatch."""
    for pattern, value in table.items():
        if fnmatch.fnmatch(filename, pattern):
            return value
    return None


def _validate_json_structure(filepath: str) -> tuple:
    """Validate JSON file has meaningful content. Returns (is_valid, reason)."""
    filename = Path(filepath).name
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Read error: {e}"

    if data is None:
        return False, "JSON contains null"

    if isinstance(data, list):
        if len(data) == 0:
            return False, "JSON array is empty"
        min_count = _match_config_by_pattern(
            filename, FILE_VALIDATION_CONFIG["json_array_minimums"]
        )
        if min_count and len(data) < min_count:
            return False, f"JSON array has {len(data)} items, expected >= {min_count}"

    elif isinstance(data, dict):
        if len(data) == 0:
            return False, "JSON object is empty"
        required_keys = _match_config_by_pattern(
            filename, FILE_VALIDATION_CONFIG["json_required_keys"]
        )
        if required_keys:
            missing_keys = [k for k in required_keys if k not in data]
            if missing_keys:
                return False, f"Missing required keys: {missing_keys}"

    return True, "Valid JSON structure"


def _validate_file_content(filepath: str) -> tuple:
    """Validate file content based on type. Returns (is_valid, reason)."""
    ext = Path(filepath).suffix.lower()

    if ext == ".json":
        return _validate_json_structure(filepath)

    elif ext == ".npz":
        try:
            import numpy as np
            with np.load(filepath, mmap_mode="r") as data:
                if len(data.files) == 0:
                    return False, "NPZ contains no arrays"
                for key in data.files:
                    if data[key].size > 0:
                        return True, f"NPZ valid with {len(data.files)} arrays"
                return False, "All NPZ arrays are empty"
        except Exception as e:
            return False, f"NPZ read error: {e}"

    elif ext in [".pth", ".pt"]:
        try:
            with open(filepath, "rb") as f:
                magic = f.read(16)
            if len(magic) < 16:
                return False, "PyTorch file too small"
            return True, "PyTorch checkpoint present (not deserialized)"
        except Exception as e:
            return False, f"PyTorch file read error: {e}"

    elif ext in [".xgb", ".lgb", ".cbm"]:
        try:
            with open(filepath, "rb") as f:
                magic = f.read(16)
            if len(magic) < 16:
                return False, f"{ext} model file too small"
            return True, f"{ext} model present (not deserialized)"
        except Exception as e:
            return False, f"{ext} file read error: {e}"

    return True, "Content validation skipped (unknown type)"


def evaluate_file_exists(filepath: str, validate_content: bool = True) -> tuple:
    """
    Evaluate if file exists AND has meaningful content.
    Returns (success, reason) tuple.
    """
    if not os.path.exists(filepath):
        return False, f"File does not exist: {filepath}"

    if not os.path.isfile(filepath):
        return False, f"Path is not a file: {filepath}"

    file_size = os.path.getsize(filepath)
    min_size = _get_min_file_size(filepath)

    if file_size < min_size:
        return False, f"File too small: {file_size} bytes (min: {min_size})"

    if validate_content:
        is_valid, content_reason = _validate_file_content(filepath)
        if not is_valid:
            return False, f"Content validation failed: {content_reason}"
        return True, f"Valid file: {file_size} bytes - {content_reason}"

    return True, f"Valid file: {file_size} bytes"



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("WatcherAgent")


# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class WatcherConfig:
    """Configuration for the Watcher Agent."""

    # Thresholds
    auto_proceed_threshold: float = 0.70  # Minimum confidence to auto-proceed
    escalate_threshold: float = 0.50      # Below this, always escalate

    # Retry limits
    max_retries_per_step: int = 3
    max_total_retries: int = 10

    # Timing
    poll_interval_seconds: int = 30
    step_timeout_minutes: int = 120

    # Paths
    results_dir: str = "results"
    manifests_dir: str = "agent_manifests"
    history_file: str = "watcher_history.json"
    log_file: str = "watcher_decisions.jsonl"

    # LLM settings
    llm_endpoint: str = "http://localhost:8080/completion"
    llm_timeout: int = 60
    use_llm: bool = True
    use_grammar: bool = True  # v1.1.0: Enable grammar-constrained decoding

    # Safety
    halt_file: str = "/tmp/agent_halt"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            "auto_proceed_threshold": self.auto_proceed_threshold,
            "escalate_threshold": self.escalate_threshold,
            "max_retries_per_step": self.max_retries_per_step,
            "max_total_retries": self.max_total_retries,
            "poll_interval_seconds": self.poll_interval_seconds,
            "step_timeout_minutes": self.step_timeout_minutes,
            "use_llm": self.use_llm,
            "use_grammar": self.use_grammar
        }


# Step to script mapping
STEP_SCRIPTS = {
    1: "window_optimizer.py",
    2: "run_scorer_meta_optimizer.sh",
    3: "run_step3_full_scoring.sh",
    4: "adaptive_meta_optimizer.py",
    5: "meta_prediction_optimizer_anti_overfit.py",
    6: "prediction_generator.py"
}

# Step to manifest mapping
STEP_MANIFESTS = {
    1: "window_optimizer.json",
    2: "scorer_meta.json",
    3: "full_scoring.json",
    4: "ml_meta.json",
    5: "reinforcement.json",
    6: "prediction.json"
}

# Step names
STEP_NAMES = {
    1: "Window Optimizer",
    2: "Scorer Meta-Optimizer",
    3: "Full Scoring",
    4: "ML Meta-Optimizer",
    5: "Anti-Overfit Training",
    6: "Prediction Generator"
}


# ════════════════════════════════════════════════════════════════════════════════
# WATCHER AGENT
# ════════════════════════════════════════════════════════════════════════════════

class WatcherAgent:
    """
    Autonomous pipeline monitoring and orchestration agent.

    The Watcher Agent:
    1. Monitors for completed pipeline steps
    2. Builds context using FullAgentContext
    3. Evaluates results (LLM or heuristic)
    4. Decides: proceed, retry, or escalate
    5. Triggers next step or alerts human

    v1.1.0: Now uses grammar-constrained LLM evaluation via LLMRouter
    """

    def __init__(self, config: WatcherConfig = None):
        self.config = config or WatcherConfig()
        self.history = self._load_history()
        self.kill_switch = KillSwitch(
            halt_file_path=self.config.halt_file,
            max_retries_per_step=self.config.max_retries_per_step
        )

        # Runtime state
        self.current_step = 0
        self.retry_counts: Dict[int, int] = {}
        self.total_retries = 0
        self.running = False

        # v1.1.0: Initialize LLM Router if available
        self.llm_router = None
        if LLM_ROUTER_AVAILABLE and self.config.use_llm:
            try:
                self.llm_router = get_router()
                logger.info("LLMRouter v1.1.0 initialized with grammar support")
            except Exception as e:
                logger.warning(f"Could not initialize LLMRouter: {e}")

        logger.info(f"WatcherAgent v1.1.0 initialized with config: {self.config.to_dict()}")

    def _load_history(self) -> AnalysisHistory:
        """Load or create analysis history."""
        if os.path.exists(self.config.history_file):
            return load_history(self.config.history_file)
        return AnalysisHistory()

    def _save_history(self):
        """Save analysis history."""
        self.history.save(self.config.history_file)

    def _log_decision(self, decision_record: Dict[str, Any]):
        """Append decision to JSONL log."""
        decision_record["timestamp"] = datetime.utcnow().isoformat()

        with open(self.config.log_file, 'a') as f:
            f.write(json.dumps(decision_record, default=str) + "\n")

    # ════════════════════════════════════════════════════════════════════════
    # CORE EVALUATION
    # ════════════════════════════════════════════════════════════════════════

    def evaluate_results(
        self,
        step: int,
        results: Dict[str, Any],
        run_number: int = 1
    ) -> Tuple[AgentDecision, FullAgentContext]:
        """
        Evaluate results for a pipeline step.

        Args:
            step: Pipeline step (1-6)
            results: Results dict to evaluate
            run_number: Current run number for this step

        Returns:
            (AgentDecision, FullAgentContext) tuple
        """
        logger.info(f"Evaluating Step {step} ({STEP_NAMES.get(step, 'Unknown')})")

        # Get manifest path
        manifest_name = STEP_MANIFESTS.get(step)
        manifest_path = None
        if manifest_name:
            manifest_path = os.path.join(self.config.manifests_dir, manifest_name)
            if not os.path.exists(manifest_path):
                manifest_path = None
                logger.warning(f"Manifest not found: {manifest_path}")

        # Check for file-based evaluation (Step 2.5 style)
        if manifest_path and os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest_data = json.load(f)
            
            if manifest_data.get('evaluation_type') == 'file_exists':
                required_files = manifest_data.get('success_condition', [])
                if isinstance(required_files, str):
                    required_files = [required_files]
                
                # Validate files exist AND have meaningful content (v1.1.0 fix)
                validation_results = []
                for p in required_files:
                    is_valid, reason = evaluate_file_exists(p)
                    validation_results.append({"file": p, "valid": is_valid, "reason": reason})
                
                failed = [r for r in validation_results if not r["valid"]]
                success = len(failed) == 0

                if success:
                    logger.info(f"File-based evaluation: all {len(required_files)} files valid")
                else:
                    for f in failed:
                        logger.warning(f"File validation failed: {f['file']} - {f['reason']}")
                
                decision = AgentDecision(
                    success_condition_met=success,
                    confidence=1.0 if success else 0.0,
                    reasoning="All required files valid" if success else f"File validation failed: {[r['file'] + ': ' + r['reason'] for r in failed]}",
                    recommended_action="proceed" if success else "escalate",
                    parse_method="file_exists"
                )
                # Build minimal context for return
                context = build_full_context(
                    step=step,
                    results=results,
                    run_number=run_number,
                    manifest_path=manifest_path
                )
                return decision, context

        # Build full context
        context = build_full_context(
            step=step,
            results=results,
            run_number=run_number,
            manifest_path=manifest_path
        )

        # Set history
        context.history = self.history

        # Check safety first
        if not context.is_safe():
            logger.warning("Safety check failed - forcing escalate")
            decision = AgentDecision(
                success_condition_met=False,
                confidence=0.0,
                reasoning="Safety halt triggered - human review required",
                recommended_action="escalate",
                warnings=["Kill switch activated"]
            )
            return decision, context

        # Try LLM evaluation
        if self.config.use_llm:
            decision = self._evaluate_with_llm(context)
            if decision:
                return decision, context

        # Fallback to heuristic evaluation
        decision = self._evaluate_heuristic(context)
        return decision, context

    def _evaluate_with_llm(self, context: FullAgentContext) -> Optional[AgentDecision]:
        """
        Evaluate using LLM with grammar-constrained decoding.
        v1.2.0: Step-specific prompts with mission context (Team Beta 2026-01-04)
        v1.1.0: Uses LLMRouter.evaluate_decision() for guaranteed JSON structure.
        Returns AgentDecision or None if LLM unavailable.
        """
        # v1.2.0: Use step-specific prompts for supported steps
        if self.llm_router and self.config.use_grammar:
            # Step 1: Use new prompt with mission context
            if context.step == 1 and STEP_PROMPTS_AVAILABLE:
                try:
                    decision = self._evaluate_step1_with_new_prompt(context)
                    if decision:
                        return decision
                    # Fall through to legacy if new method returns None
                    logger.info("Step 1 new prompt failed, trying legacy router")
                except Exception as e:
                    logger.warning(f"Step 1 new prompt failed: {e}, trying legacy")
            
            # Legacy router for other steps or fallback
            try:
                decision = self._evaluate_with_router(context)
                if decision:
                    return decision
            except Exception as e:
                logger.warning(f"Router evaluation failed: {e}, trying direct HTTP")
        
        # Fallback: Direct HTTP request (legacy method)
        return self._evaluate_with_http(context)


    def _evaluate_step1_with_new_prompt(self, context: FullAgentContext) -> Optional[AgentDecision]:
        """
        Evaluate Step 1 using the new prompt structure.
        Team Beta Approved: 2026-01-04
        
        Uses raw + derived metrics instead of legacy to_context_dict().
        Includes mission context so LLM understands PRNG domain.
        """
        if not STEP_PROMPTS_AVAILABLE:
            logger.warning("Step prompts not available, falling back to legacy")
            return None
        
        # Get the WindowOptimizerContext
        agent_ctx = context.agent_context
        if not hasattr(agent_ctx, 'get_raw_metrics'):
            logger.warning("Agent context missing get_raw_metrics, falling back to legacy")
            return None
        
        # Extract metrics using new methods
        raw_metrics = agent_ctx.get_raw_metrics()
        derived_metrics = agent_ctx.get_derived_metrics()
        threshold_priors = agent_ctx.get_threshold_priors()
        data_source_type = getattr(agent_ctx, 'data_source_type', 'synthetic')
        
        # Build new prompt
        prompt = build_step1_prompt(
            raw_metrics=raw_metrics,
            derived_metrics=derived_metrics,
            threshold_priors=threshold_priors,
            data_source_type=data_source_type
        )
        
        agent_name = "watcher_step1_v2"
        
        try:
            # Use grammar-constrained evaluation with NEW grammar
            decision_dict = self.llm_router.evaluate_watcher_decision(
                prompt,
                step_id="step1_window_optimizer",
                agent=agent_name
            )
            
            # Validate checks
            checks = decision_dict.get("checks", {})
            if not checks.get("used_rates", False) or not checks.get("mentioned_data_source", False):
                logger.warning("LLM violated checks, attempting corrective retry")
                # One corrective retry (Team Beta rule)
                corrective_prompt = build_step1_corrective_prompt(
                    raw_metrics=raw_metrics,
                    derived_metrics=derived_metrics,
                    threshold_priors=threshold_priors,
                    data_source_type=data_source_type,
                    violation="checks.used_rates or checks.mentioned_data_source was false"
                )
                decision_dict = self.llm_router.evaluate_watcher_decision(
                    corrective_prompt,
                    step_id="step1_window_optimizer",
                    agent=agent_name + "_retry"
                )
                # If still failing, return None to trigger heuristic fallback
                checks = decision_dict.get("checks", {})
                if not checks.get("used_rates", False):
                    logger.error("LLM failed checks after retry, using heuristic fallback")
                    return None
            
            # Apply deterministic threshold guardrail (Team Beta rule)
            # LLMs reason, deterministic code decides on thresholds
            decision_dict = validate_decision(
                decision_dict,
                derived_metrics,
                threshold_priors,
                data_source_type
            )
            
            # Map new schema to AgentDecision
            decision = decision_dict.get("decision", "escalate")
            action_map = {
                "proceed": "proceed",
                "retry": "retry",
                "escalate": "escalate"
            }
            
            agent_decision = AgentDecision(
                success_condition_met=(decision == "proceed"),
                confidence=float(decision_dict.get("confidence", 0.5)),
                reasoning=decision_dict.get("reasoning", "No reasoning provided"),
                recommended_action=action_map.get(decision, "escalate"),
                suggested_param_adjustments=decision_dict.get("suggested_params") or {},
                warnings=decision_dict.get("warnings", [])
            )
            agent_decision.parse_method = "llm_step1_new_prompt"
            
            logger.info(
                f"Step 1 LLM decision (new prompt): {agent_decision.recommended_action} "
                f"(confidence={agent_decision.confidence:.2f}, "
                f"primary_signal={decision_dict.get('primary_signal', 'unknown')})"
            )
            return agent_decision
            
        except Exception as e:
            logger.warning(f"Step 1 new prompt evaluation error: {e}")
            return None

    def _evaluate_with_router(self, context: FullAgentContext) -> Optional[AgentDecision]:
        """
        Evaluate using LLMRouter v1.1.0 with grammar-constrained decoding.

        This method guarantees valid JSON output structure via GBNF grammar.
        """
        prompt = context.to_llm_prompt()

        # Set agent identity for logging
        agent_name = f"watcher_step{context.step}"

        try:
            # Use grammar-constrained evaluation
            # Returns Dict with guaranteed structure:
            # {
            #   "success_condition_met": bool,
            #   "confidence": float,
            #   "reasoning": str,
            #   "recommended_action": "proceed"|"retry"|"escalate"
            # }
            decision_dict = self.llm_router.evaluate_decision(
                prompt,
                agent=agent_name,
                temperature=0.1
            )

            # Convert to AgentDecision
            decision = AgentDecision(
                success_condition_met=decision_dict.get("success_condition_met", False),
                confidence=float(decision_dict.get("confidence", 0.5)),
                reasoning=decision_dict.get("reasoning", "No reasoning provided"),
                recommended_action=decision_dict.get("recommended_action", "escalate"),
                suggested_param_adjustments=decision_dict.get("suggested_param_adjustments", {}),
                warnings=decision_dict.get("warnings", [])
            )
            decision.parse_method = "llm_router_grammar"

            logger.info(
                f"LLM Router decision: {decision.recommended_action} "
                f"(confidence={decision.confidence:.2f}, grammar-enforced)"
            )
            return decision

        except json.JSONDecodeError as e:
            # Should not happen with grammar, but handle gracefully
            logger.error(f"JSON decode error (grammar may have failed): {e}")
            return None
        except Exception as e:
            logger.warning(f"Router evaluation error: {e}")
            return None

    def _evaluate_with_http(self, context: FullAgentContext) -> Optional[AgentDecision]:
        """
        Evaluate using direct HTTP request (legacy fallback).

        Used when LLMRouter is not available or grammar evaluation fails.
        """
        try:
            import requests

            prompt = context.to_llm_prompt()

            payload = {
                "prompt": prompt,
                "n_predict": 500,
                "temperature": 0.1,
                "stop": ["</s>", "<|im_end|>", "<|endoftext|>"]
            }

            response = requests.post(
                self.config.llm_endpoint,
                json=payload,
                timeout=self.config.llm_timeout
            )

            if response.status_code == 200:
                result = response.json()
                llm_response = result.get("content", result.get("response", ""))

                decision = parse_llm_response(llm_response)
                decision.parse_method = f"llm_http_{decision.parse_method}"

                logger.info(
                    f"LLM HTTP decision: {decision.recommended_action} "
                    f"(confidence={decision.confidence:.2f})"
                )
                return decision
            else:
                logger.warning(f"LLM request failed: {response.status_code}")
                return None

        except Exception as e:
            logger.warning(f"LLM HTTP evaluation failed: {e}")
            return None

    def _evaluate_heuristic(self, context: FullAgentContext) -> AgentDecision:
        """
        Evaluate using heuristic rules (no LLM).

        Uses the specialized agent context's built-in evaluation.
        """
        logger.info("Using heuristic evaluation (LLM unavailable)")

        # Get evaluation from context
        summary = context.get_evaluation_summary()
        success = summary.get("success", False)
        confidence = summary.get("confidence", 0.5)
        interpretation = summary.get("interpretation", "No interpretation")

        # Determine action based on thresholds
        if success and confidence >= self.config.auto_proceed_threshold:
            action = "proceed"
        elif confidence < self.config.escalate_threshold:
            action = "escalate"
        elif not success and self.retry_counts.get(context.step, 0) < self.config.max_retries_per_step:
            action = "retry"
        else:
            action = "escalate"

        # Get retry suggestions if needed
        suggested_adjustments = {}
        if action == "retry" and context.agent_context:
            suggestions = context.agent_context.get_retry_suggestions()
            for s in suggestions[:3]:  # Take top 3
                suggested_adjustments[s["param"]] = s["suggestion"]

        decision = AgentDecision(
            success_condition_met=success,
            confidence=confidence,
            reasoning=interpretation[:500],
            recommended_action=action,
            suggested_param_adjustments=suggested_adjustments,
            warnings=[]
        )
        decision.parse_method = "heuristic"

        logger.info(f"Heuristic decision: {action} (confidence={confidence:.2f})")
        return decision

    # ════════════════════════════════════════════════════════════════════════
    # ACTION EXECUTION
    # ════════════════════════════════════════════════════════════════════════

    def execute_decision(
        self,
        decision: AgentDecision,
        context: FullAgentContext
    ) -> bool:
        """
        Execute the decision.

        Args:
            decision: The AgentDecision to execute
            context: The FullAgentContext used

        Returns:
            True if pipeline should continue, False to stop
        """
        step = context.step

        # Record to history
        context.record_to_history(decision.model_dump())
        self._save_history()

        # Log decision
        self._log_decision({
            "step": step,
            "step_name": STEP_NAMES.get(step, "Unknown"),
            "run_number": context.run_number,
            "action": decision.recommended_action,
            "confidence": decision.confidence,
            "success": decision.success_condition_met,
            "reasoning": decision.reasoning,
            "warnings": decision.warnings,
            "parse_method": decision.parse_method
        })

        # Display LLM feedback to user
        print(f"\n{'='*60}")
        print(f"📊 WATCHER EVALUATION - Step {step}: {STEP_NAMES.get(step, 'Unknown')}")
        print(f"{'='*60}")
        print(f"   Parse Method: {decision.parse_method}")
        print(f"   Confidence:   {decision.confidence:.2f}")
        print(f"   Action:       {decision.recommended_action.upper()}")
        print(f"   Reasoning:    {decision.reasoning}")
        if decision.suggested_param_adjustments:
            print(f"   Adjustments:  {decision.suggested_param_adjustments}")
        if decision.warnings:
            print(f"   ⚠️  Warnings:  {decision.warnings}")
        print(f"{'='*60}\n")

        # Execute based on action
        if decision.recommended_action == "proceed":
            return self._handle_proceed(step, context)

        elif decision.recommended_action == "retry":
            return self._handle_retry(step, decision, context)

        else:  # escalate
            return self._handle_escalate(step, decision, context)

    def _handle_proceed(self, step: int, context: FullAgentContext) -> bool:
        """Handle proceed action - advance to next step."""
        logger.info(f"✅ Step {step} PASSED - proceeding to next step")

        # Reset retry count for this step
        self.retry_counts[step] = 0

        # Check if pipeline complete
        if step >= 6:
            logger.info("🎉 PIPELINE COMPLETE - all 6 steps finished!")
            self._notify_complete(context)
            return False

        # Trigger next step
        next_step = step + 1
        self.current_step = next_step

        logger.info(f"Triggering Step {next_step}: {STEP_NAMES.get(next_step, 'Unknown')}")
        return True

    def _handle_retry(
        self,
        step: int,
        decision: AgentDecision,
        context: FullAgentContext
    ) -> bool:
        """Handle retry action - re-run step with adjustments."""
        self.retry_counts[step] = self.retry_counts.get(step, 0) + 1
        self.total_retries += 1

        logger.warning(
            f"⚠️ Step {step} needs RETRY ({self.retry_counts[step]}/{self.config.max_retries_per_step})"
        )

        # Check retry limits
        if self.retry_counts[step] >= self.config.max_retries_per_step:
            logger.error(f"Max retries reached for step {step} - escalating")
            return self._handle_escalate(step, decision, context)

        if self.total_retries >= self.config.max_total_retries:
            logger.error(f"Max total retries ({self.config.max_total_retries}) reached - escalating")
            return self._handle_escalate(step, decision, context)

        # Log suggested adjustments
        if decision.suggested_param_adjustments:
            logger.info(f"Suggested adjustments: {decision.suggested_param_adjustments}")

        # Will re-run the same step
        return True

    def _handle_escalate(
        self,
        step: int,
        decision: AgentDecision,
        context: FullAgentContext
    ) -> bool:
        """Handle escalate action - stop and alert human."""
        logger.error(f"❌ Step {step} ESCALATED - human review required")
        logger.error(f"   Reason: {decision.reasoning}")

        if decision.warnings:
            logger.error(f"   Warnings: {decision.warnings}")

        # Create halt file
        self.kill_switch.create_halt_file(
            f"Escalated at step {step}: {decision.reasoning[:100]}"
        )

        # Notify human
        self._notify_escalation(step, decision, context)

        return False

    def _notify_complete(self, context: FullAgentContext):
        """Notify that pipeline completed successfully."""
        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Total runs in history: {len(self.history.runs)}")
        print(f"Success rate: {self.history.get_success_rate():.1%}")
        print("=" * 60 + "\n")

    def _notify_escalation(
        self,
        step: int,
        decision: AgentDecision,
        context: FullAgentContext
    ):
        """Notify human of escalation."""
        print("\n" + "=" * 60)
        print("⚠️  HUMAN REVIEW REQUIRED")
        print("=" * 60)
        print(f"Step: {step} - {STEP_NAMES.get(step, 'Unknown')}")
        print(f"Confidence: {decision.confidence:.2f}")
        print(f"Reason: {decision.reasoning}")
        if decision.warnings:
            print(f"Warnings: {decision.warnings}")
        print("-" * 60)
        print("To resume after review:")
        print(f"  python3 watcher_agent.py --clear-halt --run-pipeline --start-step {step}")
        print("=" * 60 + "\n")

    # ════════════════════════════════════════════════════════════════════════
    # STEP EXECUTION
    # ════════════════════════════════════════════════════════════════════════

    def run_step(
        self,
        step: int,
        params: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Run a pipeline step and return results.

        Args:
            step: Step number (1-6)
            params: Optional parameters to pass (overrides defaults)

        Returns:
            Results dict or None if failed
        """
        script = STEP_SCRIPTS.get(step)
        if not script:
            logger.error(f"No script defined for step {step}")
            return None

        logger.info(f"Running Step {step}: {script}")

        # Try to load default params from manifest
        default_params = {}
        manifest_name = STEP_MANIFESTS.get(step)
        if manifest_name:
            manifest_path = os.path.join(self.config.manifests_dir, manifest_name)
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path) as f:
                        manifest_data = json.load(f)
                    default_params = manifest_data.get("default_params", {})
                    logger.debug(f"Loaded default params: {list(default_params.keys())}")
                except Exception as e:
                    logger.warning(f"Could not load manifest defaults: {e}")

        # Merge params: user params override defaults (STEP-SCOPED)
        final_params = {**default_params}
        if params:
            # Only merge params that this step's manifest declares (Team Beta: step-scoped filtering)
            allowed_params = set(default_params.keys())
            for key, value in params.items():
                if key in allowed_params:
                    final_params[key] = value
                else:
                    logger.debug(f"Skipping param '{key}' - not declared in step {step} manifest")

        # Remove output_file if present (use script default)
        final_params.pop("output_file", None)

        # Build command
        # Detect script type: .sh uses bash, .py uses python3
        if script.endswith(".sh"):
            cmd = ["bash", script]
        else:
            cmd = ["python3", script]
        # Build args based on manifest's declared arg_style
        arg_style = manifest_data.get("arg_style", "named") if manifest_data else "named"
        
        if arg_style == "positional":
            # Shell scripts with positional args - use manifest hints
            positional_args = manifest_data.get("positional_args", [])
            flag_args = manifest_data.get("flag_args", [])
            
            # Add positional args in order
            for arg_name in positional_args:
                if arg_name in final_params:
                    cmd.append(str(final_params[arg_name]))
            
            # Add flag args (--flag style, no value)
            for arg_name in flag_args:
                if final_params.get(arg_name):
                    cmd.append(f"--{arg_name.replace('_', '-')}")
        else:
            # Python scripts with named args (--key value)
            for key, value in final_params.items():
                cli_key = key.replace("_", "-")
                
                # Boolean flag-only behavior
                if isinstance(value, bool):
                    if value:
                        cmd.append(f"--{cli_key}")
                    # If False, omit the flag entirely
                    continue
                
                cmd.extend([f"--{cli_key}", str(value)])

        logger.info(f"EXEC CMD: {' '.join(cmd)}")
        try:
            # Run the script with live streaming (Team Beta approved)
            result = self._run_step_streaming(
                cmd,
                step,
                self.config.step_timeout_minutes * 60
            )
            if result.returncode != 0:
                logger.error(f"Step {step} failed with code {result.returncode}")
                logger.error(f"stderr: {result.stderr[:500]}")
                return {
                    "success": False,
                    "return_code": result.returncode,
                    "error": result.stderr[:500]
                }

            # Try to find and load results file
            results = self._find_results(step)
            if results:
                return results

            # Return basic success
            return {
                "success": True,
                "return_code": 0,
                "stdout": result.stdout[:500]
            }

        except subprocess.TimeoutExpired:
            logger.error(f"Step {step} timed out after {self.config.step_timeout_minutes} minutes")
            return {
                "success": False,
                "error": "Timeout"
            }
        except Exception as e:
            logger.error(f"Step {step} execution error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _find_results(self, step: int) -> Optional[Dict[str, Any]]:
        """Find and load results file for a step."""
        # Step-specific output files (check main directory first)
        step_files = {
            1: "optimal_window_config.json",
            2: "optimal_scorer_config.json",
            3: "full_scoring_results.json",
            4: "optimal_ml_config.json",
            5: "anti_overfit_results.json",
            6: "prediction_pool.json"
        }
        
        # First, check for step-specific file in main directory
        if step in step_files:
            main_file = Path(step_files[step])
            if main_file.exists():
                try:
                    with open(main_file) as f:
                        logger.debug(f"Loaded results from {main_file}")
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load {main_file}: {e}")
        
        # Fallback: check results directory with patterns
        patterns = [
            f"step{step}_*.json",
            f"*_results.json",
            "results.json"
        ]
        results_path = Path(self.config.results_dir)
        if not results_path.exists():
            return None
        
        # Find most recent result file
        for pattern in patterns:
            files = sorted(results_path.glob(pattern), key=os.path.getmtime, reverse=True)
            if files:
                try:
                    with open(files[0]) as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load {files[0]}: {e}")
        return None

    # ════════════════════════════════════════════════════════════════════════
    # PIPELINE EXECUTION
    # ════════════════════════════════════════════════════════════════════════

    def _run_step_streaming(self, cmd, step, timeout_seconds):
        """
        Run command with live output streaming.
        Adapted from complete_whitepaper_workflow_with_meta_optimizer.py
        
        Team Beta amendments applied:
        - PYTHONUNBUFFERED=1 for Python scripts
        - Token-aware progress parsing (not generic regex)
        - Atomic progress file writes
        - Process group kill on timeout
        """
        import re
        
        # Environment with unbuffered Python output
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        # Start process in new session for clean group kill
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True
        )
        
        # Guard against None stdout (Team Beta recommendation A)
        if process.stdout is None:
            raise RuntimeError("Process stdout not available")
        
        output_lines = []
        start_time = time.time()
        step_name = STEP_NAMES.get(step, f"Step {step}")
        
        # Token-aware progress patterns (Team Beta: not generic regex)
        PROGRESS_PATTERNS = [
            # Window optimizer: [5/50] or [Trial 5/50]
            re.compile(r'\[(?:Trial\s*)?(\d+)/(\d+)\]'),
            # Iteration format: --- Iteration 5/50 ---
            re.compile(r'Iteration\s+(\d+)/(\d+)'),
            # Coordinator jobs: Job 5/34 or similar
            re.compile(r'Job\s+(\d+)/(\d+)'),
        ]
        
        current_progress = {"current": 0, "total": 0}
        
        # Write initial progress (Team Beta recommendation B)
        self._write_progress_atomic(step, step_name, current_progress, 0.0)
        
        try:
            while True:
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    # Kill entire process group
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                        time.sleep(1)
                        if process.poll() is None:
                            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    raise subprocess.TimeoutExpired(cmd, timeout_seconds)
                
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    line = output.strip()
                    print(line)  # Stream to terminal
                    output_lines.append(output)
                    
                    # Token-aware progress parsing
                    for pattern in PROGRESS_PATTERNS:
                        match = pattern.search(line)
                        if match:
                            groups = match.groups()
                            if len(groups) >= 2:
                                current_progress["current"] = int(groups[0])
                                current_progress["total"] = int(groups[1])
                                self._write_progress_atomic(step, step_name, current_progress, elapsed)
                            break
            
            # Final progress update
            self._write_progress_atomic(step, step_name, current_progress, 
                                        time.time() - start_time, finished=True)
            
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=process.returncode,
                stdout=''.join(output_lines),
                stderr=''
            )
            
        except subprocess.TimeoutExpired:
            raise
        except Exception as e:
            logger.error(f"Streaming execution error: {e}")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except:
                pass
            raise

    def _write_progress_atomic(self, step, step_name, progress, elapsed, finished=False):
        """Atomic write to progress file (Team Beta: avoid partial JSON reads)."""
        progress_data = {
            "step": step,
            "step_name": step_name,
            "jobs_completed": progress.get("current", 0),
            "total_jobs": progress.get("total", 0),
            "elapsed_seconds": elapsed,
            "finished": finished,
            "timestamp": time.time()
        }
        
        progress_file = "/tmp/cluster_progress.json"
        temp_file = f"{progress_file}.tmp.{os.getpid()}"
        
        try:
            with open(temp_file, 'w') as f:
                json.dump(progress_data, f)
            os.rename(temp_file, progress_file)  # Atomic on POSIX
        except Exception as e:
            logger.debug(f"Progress file write failed: {e}")


    def run_pipeline(self, start_step: int = 1, end_step: int = 6, params: Dict[str, Any] = None):
        """
        Run the full pipeline from start_step to end_step.

        Args:
            start_step: First step to run (1-6)
            end_step: Last step to run (1-6)
        """
        logger.info(f"Starting pipeline from step {start_step} to {end_step}")

        
        # Check for halt file BEFORE starting
        if not check_safety():
            halt_file = "/tmp/agent_halt"
            print("\n" + "="*60)
            print(f"⛔ PIPELINE HALTED - Cannot start")
            print("="*60)
            try:
                with open(halt_file) as f:
                    reason = f.read().strip()
                print(f"Reason: {reason}")
            except:
                print("Reason: Unknown (halt file exists)")
            print(f"\nTo resume: python3 -m agents.watcher_agent --clear-halt")
            print("="*60 + "\n")
            return
        self.running = True
        self.current_step = start_step

        # Use progress display if available
        progress = None
        if PROGRESS_DISPLAY_AVAILABLE:
            progress = WatcherProgress()
            progress.__enter__()

        try:
            while self.running and self.current_step <= end_step:
                # Safety check
                if not check_safety():
                    logger.error("Safety halt detected - stopping pipeline")
                    break

                step = self.current_step
                run_number = self.retry_counts.get(step, 0) + 1

                # Update progress display
                if progress:
                    # Get total trials from manifest if available
                    total_trials = self._get_step_trials(step)
                    progress.start_step(step, total_trials=total_trials)

                logger.info(f"\n{'='*60}")
                logger.info(f"STEP {step}: {STEP_NAMES.get(step, 'Unknown')} (run #{run_number})")
                logger.info(f"{'='*60}")

                # Run the step
                results = self.run_step(step, params)

                if results is None:
                    results = {"success": False, "error": "No results returned"}

                # Update progress with results
                if progress:
                    best_score = results.get("best_score", results.get("confidence", 0))
                    progress.update_step(step, best_score=best_score)

                # Evaluate results
                decision, context = self.evaluate_results(step, results, run_number)

                # Update progress display with decision
                if progress:
                    success = decision.recommended_action == "proceed"
                    progress.complete_step(step, success=success, score=decision.confidence)

                # Execute decision
                should_continue = self.execute_decision(decision, context)

                if not should_continue:
                    break

                # If retry, stay on same step; if proceed, move to next
                # if decision.recommended_action == "proceed":  # REMOVED: double-increment bug
                #     self.current_step += 1  # Step advancement handled in _handle_proceed()

                # Small delay between steps
                time.sleep(1)

        finally:
            # Clean up progress display
            if progress:
                progress.__exit__(None, None, None)

        self.running = False
        logger.info("Pipeline execution finished")

    def _get_step_trials(self, step: int) -> int:
        """Get expected number of trials for a step from manifest."""
        manifest_name = STEP_MANIFESTS.get(step)
        if not manifest_name:
            return 0

        manifest_path = os.path.join(self.config.manifests_dir, manifest_name)
        if not os.path.exists(manifest_path):
            return 0

        try:
            with open(manifest_path) as f:
                data = json.load(f)
            return data.get("default_params", {}).get("trials", 20)
        except Exception:
            return 20  # Default

    # ════════════════════════════════════════════════════════════════════════
    # DAEMON MODE
    # ════════════════════════════════════════════════════════════════════════

    def run_daemon(self, watch_dir: str = None):
        """
        Run as a daemon, watching for new result files.

        Args:
            watch_dir: Directory to watch for results
        """
        watch_dir = watch_dir or self.config.results_dir
        logger.info(f"Starting daemon mode, watching: {watch_dir}")

        processed_files = set()
        self.running = True

        while self.running:
            # Safety check
            if not check_safety():
                logger.warning("Safety halt detected - pausing daemon")
                time.sleep(self.config.poll_interval_seconds)
                continue

            # Scan for new result files
            results_path = Path(watch_dir)
            if results_path.exists():
                for result_file in results_path.glob("*.json"):
                    if str(result_file) not in processed_files:
                        self._process_result_file(result_file)
                        processed_files.add(str(result_file))

            time.sleep(self.config.poll_interval_seconds)

    def _process_result_file(self, result_file: Path):
        """Process a new result file."""
        logger.info(f"Processing new result file: {result_file}")

        try:
            with open(result_file) as f:
                results = json.load(f)

            # Try to determine step from file or content
            step = results.get("pipeline_step",
                   results.get("agent_metadata", {}).get("pipeline_step", 1))

            run_number = self.retry_counts.get(step, 0) + 1

            # Evaluate
            decision, context = self.evaluate_results(step, results, run_number)

            # Execute
            self.execute_decision(decision, context)

        except Exception as e:
            logger.error(f"Error processing {result_file}: {e}")

    def stop(self):
        """Stop the watcher."""
        self.running = False
        logger.info("Watcher stopped")


# ════════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Watcher Agent v1.1.0 - Autonomous pipeline monitoring with grammar-constrained LLM"
    )

    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as daemon, watching for results"
    )
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="Run the full pipeline"
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=1,
        help="Starting step for pipeline (1-6)"
    )
    parser.add_argument(
        "--end-step",
        type=int,
        default=6,
        help="Ending step for pipeline (1-6)"
    )
    parser.add_argument(
        "--evaluate",
        type=str,
        help="Evaluate a single result file"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show watcher status"
    )
    parser.add_argument(
        "--clear-halt",
        action="store_true",
        help="Clear the halt file to resume"
    )
    parser.add_argument(
        "--halt",
        type=str,
        nargs="?",
        const="Manual halt",
        help="Create halt file to stop watcher"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM evaluation, use heuristics only"
    )
    parser.add_argument(
        "--no-grammar",
        action="store_true",
        help="Disable grammar-constrained decoding (v1.1.0)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Auto-proceed confidence threshold (default: 0.70)"
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="JSON string of params to override manifest defaults"
    )

    args = parser.parse_args()

    # Create config
    config = WatcherConfig(
        auto_proceed_threshold=args.threshold,
        use_llm=not args.no_llm,
        use_grammar=not args.no_grammar
    )

    # Handle halt commands
    if args.clear_halt:
        clear_halt()
        print("Halt file cleared")

    if args.halt:
        create_halt(args.halt)
        print(f"Halt file created: {args.halt}")
        return

    # Create watcher
    watcher = WatcherAgent(config)

    # Execute command
    if args.status:
        print("\n=== Watcher Agent Status (v1.1.0) ===")
        print(f"Safety: {'SAFE' if check_safety() else 'HALTED'}")
        print(f"History runs: {len(watcher.history.runs)}")
        print(f"Success rate: {watcher.history.get_success_rate():.1%}")
        print(f"LLM enabled: {config.use_llm}")
        print(f"Grammar enabled: {config.use_grammar}")
        print(f"LLM Router: {'Available' if watcher.llm_router else 'Not available'}")
        print(f"Threshold: {config.auto_proceed_threshold}")
        print("======================================\n")

    elif args.evaluate:
        # Evaluate single file
        with open(args.evaluate) as f:
            results = json.load(f)

        step = results.get("pipeline_step",
               results.get("agent_metadata", {}).get("pipeline_step", 1))

        decision, context = watcher.evaluate_results(step, results)

        print("\n=== Evaluation Result ===")
        print(f"Step: {step} - {STEP_NAMES.get(step, 'Unknown')}")
        print(f"Success: {decision.success_condition_met}")
        print(f"Confidence: {decision.confidence:.2f}")
        print(f"Action: {decision.recommended_action}")
        print(f"Reasoning: {decision.reasoning}")
        print(f"Parse method: {decision.parse_method}")
        if decision.suggested_param_adjustments:
            print(f"Adjustments: {decision.suggested_param_adjustments}")
        if decision.warnings:
            print(f"Warnings: {decision.warnings}")
        print("=========================\n")

    elif args.run_pipeline:
        # Parse params JSON if provided
        override_params = None
        if args.params:
            try:
                override_params = json.loads(args.params)
                logger.info(f"CLI param overrides: {override_params}")
            except json.JSONDecodeError as e:
                print(f"ERROR: Invalid JSON in --params: {e}")
                return
        watcher.run_pipeline(args.start_step, args.end_step, override_params)

    elif args.daemon:
        try:
            watcher.run_daemon()
        except KeyboardInterrupt:
            watcher.stop()
            print("\nDaemon stopped")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
