#!/usr/bin/env python3
"""
Chapter 13 Retrain Triggers — Phase 3
Evaluates diagnostic outputs and determines when to trigger learning loop

RESPONSIBILITIES:
1. Evaluate retrain trigger conditions from diagnostics
2. Enforce cooldown periods to prevent thrashing
3. Execute partial pipeline reruns (Steps 3→5→6)
4. Gate execution on human approval (v1 requirement)
5. Track trigger history for audit

VERSION: 1.0.0
DATE: 2026-01-12
DEPENDS ON: chapter_13_diagnostics.py, watcher_policies.json, watcher_agent.py

INTEGRATION:
    This module is designed to be imported by watcher_agent.py:
    
    from chapter_13_triggers import Chapter13TriggerManager
    
    # In WatcherAgent.__init__:
    self.trigger_manager = Chapter13TriggerManager(self)
    
    # In daemon loop after new draw:
    if self.trigger_manager.should_retrain():
        self.trigger_manager.request_learning_loop()
"""

import json
import os
import sys
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Chapter13Triggers")

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_DIAGNOSTICS = "post_draw_diagnostics.json"
DEFAULT_POLICIES = "watcher_policies.json"
TRIGGER_HISTORY_FILE = "retrain_history.json"
COOLDOWN_STATE_FILE = ".chapter13_cooldown_state.json"
APPROVAL_REQUEST_FILE = "pending_approval.json"


class TriggerType(str, Enum):
    """Types of retrain triggers."""
    N_DRAWS = "n_draws_accumulated"
    CONFIDENCE_DRIFT = "confidence_drift"
    CONSECUTIVE_MISSES = "consecutive_misses"
    HIT_RATE_COLLAPSE = "hit_rate_collapse"
    REGIME_SHIFT = "regime_shift"
    LLM_PROPOSED = "llm_proposed"
    MANUAL = "manual"
    SELFPLAY_RETRAIN = "selfplay_retrain"  # Phase 9A.3 (enum only, no auto-dispatch)


class TriggerAction(str, Enum):
    """Actions that can be triggered."""
    LEARNING_LOOP = "learning_loop"      # Steps 3→5→6
    FULL_PIPELINE = "full_pipeline"       # Steps 1→6
    STEP_6_ONLY = "step_6_only"          # Just Step 6
    SELFPLAY = "selfplay"                 # Phase 9A.3 (enum only, WATCHER dispatches)


@dataclass
class TriggerEvaluation:
    """Result of evaluating retrain triggers."""
    should_trigger: bool
    trigger_type: Optional[TriggerType]
    action: Optional[TriggerAction]
    confidence: float
    reasoning: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    requires_approval: bool = True  # v1: always True
    cooldown_remaining: int = 0  # Runs until cooldown expires
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_trigger": self.should_trigger,
            "trigger_type": self.trigger_type.value if self.trigger_type else None,
            "action": self.action.value if self.action else None,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "metrics": self.metrics,
            "requires_approval": self.requires_approval,
            "cooldown_remaining": self.cooldown_remaining
        }


@dataclass
class CooldownState:
    """Tracks cooldown state to prevent thrashing."""
    last_retrain_at: Optional[str] = None
    runs_since_retrain: int = 0
    last_parameter_change_at: Optional[str] = None
    runs_since_param_change: int = 0
    consecutive_triggers_blocked: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CooldownState':
        return cls(**data)
    
    @classmethod
    def load(cls) -> 'CooldownState':
        if os.path.exists(COOLDOWN_STATE_FILE):
            with open(COOLDOWN_STATE_FILE, 'r') as f:
                return cls.from_dict(json.load(f))
        return cls()
    
    def save(self) -> None:
        with open(COOLDOWN_STATE_FILE, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# TRIGGER MANAGER
# =============================================================================

class Chapter13TriggerManager:
    """
    Manages Chapter 13 retrain triggers and learning loop execution.
    
    This class evaluates diagnostics against policy thresholds and
    determines when the learning loop (Steps 3→5→6) should run.
    
    v1 Behavior:
    - Triggers are evaluated automatically
    - Execution requires human approval
    - Cooldowns are enforced between retrains
    """
    
    def __init__(
        self,
        watcher_agent=None,  # Reference to WatcherAgent for pipeline execution
        policies_path: str = DEFAULT_POLICIES,
        diagnostics_path: str = DEFAULT_DIAGNOSTICS
    ):
        self.watcher_agent = watcher_agent
        self.policies_path = policies_path
        self.diagnostics_path = diagnostics_path
        
        self.policies = self._load_policies()
        self.cooldown = CooldownState.load()
        self.trigger_history: List[Dict[str, Any]] = self._load_trigger_history()
        
        logger.info("Chapter13TriggerManager initialized")
    
    def _load_policies(self) -> Dict[str, Any]:
        """Load watcher policies."""
        if os.path.exists(self.policies_path):
            with open(self.policies_path, 'r') as f:
                return json.load(f)
        logger.warning(f"Policies not found at {self.policies_path}, using defaults")
        return {}
    
    def _load_diagnostics(self) -> Optional[Dict[str, Any]]:
        """Load current diagnostics."""
        if os.path.exists(self.diagnostics_path):
            with open(self.diagnostics_path, 'r') as f:
                return json.load(f)
        return None
    
    def _load_trigger_history(self) -> List[Dict[str, Any]]:
        """Load trigger history."""
        if os.path.exists(TRIGGER_HISTORY_FILE):
            with open(TRIGGER_HISTORY_FILE, 'r') as f:
                data = json.load(f)
                return data.get("triggers", [])
        return []
    
    def _save_trigger_history(self) -> None:
        """Save trigger history."""
        with open(TRIGGER_HISTORY_FILE, 'w') as f:
            json.dump({
                "triggers": self.trigger_history,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }, f, indent=2)
    
    # =========================================================================
    # TRIGGER EVALUATION
    # =========================================================================
    
    def evaluate_triggers(
        self,
        diagnostics: Optional[Dict[str, Any]] = None
    ) -> TriggerEvaluation:
        """
        Evaluate all retrain triggers against current diagnostics.
        
        Args:
            diagnostics: Diagnostics dict (loads from file if None)
        
        Returns:
            TriggerEvaluation with decision
        """
        # Load diagnostics if not provided
        if diagnostics is None:
            diagnostics = self._load_diagnostics()
        
        if diagnostics is None:
            return TriggerEvaluation(
                should_trigger=False,
                trigger_type=None,
                action=None,
                confidence=0.0,
                reasoning="No diagnostics available",
                requires_approval=True
            )
        
        # Check cooldown first
        cooldown_check = self._check_cooldown()
        if cooldown_check.cooldown_remaining > 0:
            return cooldown_check
        
        # Get thresholds from policies
        triggers = self.policies.get("retrain_triggers", {})
        regime_triggers = self.policies.get("regime_shift_triggers", {})
        
        # Extract metrics from diagnostics
        pipeline_health = diagnostics.get("pipeline_health", {})
        confidence_cal = diagnostics.get("confidence_calibration", {})
        prediction_val = diagnostics.get("prediction_validation", {})
        summary_flags = diagnostics.get("summary_flags", [])
        
        # Evaluate each trigger condition
        triggered = []
        
        # 1. Consecutive misses
        consecutive_misses = pipeline_health.get("consecutive_misses", 0)
        max_misses = triggers.get("max_consecutive_misses", 5)
        if consecutive_misses >= max_misses:
            triggered.append((
                TriggerType.CONSECUTIVE_MISSES,
                TriggerAction.LEARNING_LOOP,
                0.9,
                f"Consecutive misses ({consecutive_misses}) >= threshold ({max_misses})"
            ))
        
        # 2. Confidence drift
        correlation = confidence_cal.get("predicted_vs_actual_correlation", 1.0)
        drift_threshold = triggers.get("confidence_drift_threshold", 0.2)
        if correlation < drift_threshold:
            triggered.append((
                TriggerType.CONFIDENCE_DRIFT,
                TriggerAction.LEARNING_LOOP,
                0.85,
                f"Confidence correlation ({correlation:.2f}) < threshold ({drift_threshold})"
            ))
        
        # 3. Hit rate collapse
        hit_rate = pipeline_health.get("current_hit_rate", 1.0)
        collapse_threshold = triggers.get("hit_rate_collapse_threshold", 0.01)
        if hit_rate < collapse_threshold:
            triggered.append((
                TriggerType.HIT_RATE_COLLAPSE,
                TriggerAction.LEARNING_LOOP,
                0.95,
                f"Hit rate ({hit_rate:.4f}) < collapse threshold ({collapse_threshold})"
            ))
        
        # 4. N draws accumulated (periodic retrain)
        draws_since_retrain = self.cooldown.runs_since_retrain
        n_draws_threshold = triggers.get("retrain_after_n_draws", 10)
        if draws_since_retrain >= n_draws_threshold:
            triggered.append((
                TriggerType.N_DRAWS,
                TriggerAction.LEARNING_LOOP,
                0.7,
                f"Draws since retrain ({draws_since_retrain}) >= threshold ({n_draws_threshold})"
            ))
        
        # 5. Regime shift (full pipeline)
        window_decay = pipeline_health.get("window_decay", 0)
        survivor_churn = pipeline_health.get("survivor_churn", 0)
        decay_threshold = regime_triggers.get("window_decay_threshold", 0.5)
        churn_threshold = regime_triggers.get("survivor_churn_threshold", 0.4)
        
        if window_decay > decay_threshold and survivor_churn > churn_threshold:
            triggered.append((
                TriggerType.REGIME_SHIFT,
                TriggerAction.FULL_PIPELINE,
                0.8,
                f"Regime shift: decay={window_decay:.2f}, churn={survivor_churn:.2f}"
            ))
        
        # 6. Summary flags
        if "RETRAIN_RECOMMENDED" in summary_flags:
            triggered.append((
                TriggerType.LLM_PROPOSED,
                TriggerAction.LEARNING_LOOP,
                0.75,
                "Diagnostics flagged RETRAIN_RECOMMENDED"
            ))
        
        if "REGIME_SHIFT_POSSIBLE" in summary_flags:
            triggered.append((
                TriggerType.REGIME_SHIFT,
                TriggerAction.FULL_PIPELINE,
                0.7,
                "Diagnostics flagged REGIME_SHIFT_POSSIBLE"
            ))
        
        # No triggers fired
        if not triggered:
            return TriggerEvaluation(
                should_trigger=False,
                trigger_type=None,
                action=None,
                confidence=1.0,
                reasoning="No trigger conditions met - system healthy",
                metrics={
                    "consecutive_misses": consecutive_misses,
                    "correlation": correlation,
                    "hit_rate": hit_rate,
                    "draws_since_retrain": draws_since_retrain
                },
                requires_approval=False
            )
        
        # Select highest priority trigger
        # Priority: REGIME_SHIFT > HIT_RATE_COLLAPSE > CONSECUTIVE_MISSES > CONFIDENCE_DRIFT > N_DRAWS
        priority_order = [
            TriggerType.REGIME_SHIFT,
            TriggerType.HIT_RATE_COLLAPSE,
            TriggerType.CONSECUTIVE_MISSES,
            TriggerType.CONFIDENCE_DRIFT,
            TriggerType.N_DRAWS,
            TriggerType.LLM_PROPOSED
        ]
        
        triggered.sort(key=lambda x: (
            priority_order.index(x[0]) if x[0] in priority_order else 99,
            -x[2]  # Higher confidence first
        ))
        
        best = triggered[0]
        
        return TriggerEvaluation(
            should_trigger=True,
            trigger_type=best[0],
            action=best[1],
            confidence=best[2],
            reasoning=best[3],
            metrics={
                "consecutive_misses": consecutive_misses,
                "correlation": correlation,
                "hit_rate": hit_rate,
                "draws_since_retrain": draws_since_retrain,
                "all_triggers": [t[0].value for t in triggered]
            },
            requires_approval=True  # v1: always require approval
        )
    
    def _check_cooldown(self) -> TriggerEvaluation:
        """
        Check if cooldown period is active.
        
        Returns:
            TriggerEvaluation with cooldown info (should_trigger=False if cooling down)
        """
        acceptance = self.policies.get("acceptance_rules", {})
        cooldown_runs = acceptance.get("cooldown_runs", 3)
        
        runs_remaining = cooldown_runs - self.cooldown.runs_since_retrain
        
        if runs_remaining > 0 and self.cooldown.last_retrain_at is not None:
            return TriggerEvaluation(
                should_trigger=False,
                trigger_type=None,
                action=None,
                confidence=1.0,
                reasoning=f"Cooldown active: {runs_remaining} runs remaining",
                cooldown_remaining=runs_remaining,
                requires_approval=False
            )
        
        return TriggerEvaluation(
            should_trigger=False,  # Will be overwritten by caller
            trigger_type=None,
            action=None,
            confidence=0.0,
            reasoning="Cooldown check passed",
            cooldown_remaining=0,
            requires_approval=True
        )
    
    def should_retrain(self) -> bool:
        """
        Quick check if retrain should be triggered.
        
        Returns:
            True if any trigger condition is met (still requires approval in v1)
        """
        evaluation = self.evaluate_triggers()
        return evaluation.should_trigger
    
    # =========================================================================
    # APPROVAL HANDLING (v1)
    # =========================================================================
    
    def request_approval(self, evaluation: TriggerEvaluation) -> str:
        """
        Create approval request for human review.
        
        v1: All trigger executions require human approval.
        
        Args:
            evaluation: The trigger evaluation to approve
        
        Returns:
            Path to approval request file
        """
        request = {
            "request_id": f"ch13_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending",
            "evaluation": evaluation.to_dict(),
            "diagnostics_file": self.diagnostics_path,
            "action_required": evaluation.action.value if evaluation.action else None,
            "steps_to_run": self._get_steps_for_action(evaluation.action),
            "approval_instructions": self._get_approval_instructions(evaluation)
        }
        
        with open(APPROVAL_REQUEST_FILE, 'w') as f:
            json.dump(request, f, indent=2)
        
        logger.info(f"Approval request created: {APPROVAL_REQUEST_FILE}")
        self._display_approval_request(request)
        
        return APPROVAL_REQUEST_FILE
    
    def _get_steps_for_action(self, action: Optional[TriggerAction]) -> List[int]:
        """Get pipeline steps for an action."""
        if action == TriggerAction.LEARNING_LOOP:
            return [3, 5, 6]
        elif action == TriggerAction.FULL_PIPELINE:
            return [1, 2, 3, 4, 5, 6]
        elif action == TriggerAction.STEP_6_ONLY:
            return [6]
        return []
    
    def _get_approval_instructions(self, evaluation: TriggerEvaluation) -> str:
        """Generate human-readable approval instructions."""
        steps = self._get_steps_for_action(evaluation.action)
        step_names = {
            1: "Window Optimizer",
            2: "Scorer Meta-Optimizer", 
            3: "Full Scoring",
            4: "ML Meta-Optimizer",
            5: "Anti-Overfit Training",
            6: "Prediction Generator"
        }
        
        step_list = ", ".join([f"Step {s} ({step_names.get(s, '?')})" for s in steps])
        
        return f"""
CHAPTER 13 RETRAIN REQUEST
==========================

Trigger: {evaluation.trigger_type.value if evaluation.trigger_type else 'unknown'}
Confidence: {evaluation.confidence:.2f}
Reason: {evaluation.reasoning}

Proposed Action: Run {step_list}

To APPROVE and execute:
    python3 chapter_13_triggers.py --approve

To REJECT:
    python3 chapter_13_triggers.py --reject

To view full diagnostics:
    cat {self.diagnostics_path}
"""
    
    def _display_approval_request(self, request: Dict[str, Any]) -> None:
        """Display approval request to console."""
        print("\n" + "=" * 60)
        print("⚠️  CHAPTER 13 RETRAIN APPROVAL REQUIRED")
        print("=" * 60)
        print(request["approval_instructions"])
        print("=" * 60 + "\n")
    
    def check_approval(self) -> Optional[Dict[str, Any]]:
        """
        Check if there's a pending approval request.
        
        Returns:
            Approval request dict if pending, None otherwise
        """
        if os.path.exists(APPROVAL_REQUEST_FILE):
            with open(APPROVAL_REQUEST_FILE, 'r') as f:
                request = json.load(f)
            if request.get("status") == "pending":
                return request
        return None
    
    def approve_request(self) -> bool:
        """
        Approve pending request and execute learning loop.
        
        Returns:
            True if executed successfully
        """
        request = self.check_approval()
        if not request:
            print("No pending approval request")
            return False
        
        # Update request status
        request["status"] = "approved"
        request["approved_at"] = datetime.now(timezone.utc).isoformat()
        
        with open(APPROVAL_REQUEST_FILE, 'w') as f:
            json.dump(request, f, indent=2)
        
        # Execute the learning loop
        steps = request.get("steps_to_run", [3, 5, 6])
        success = self.execute_learning_loop(steps)
        
        # Update request with result
        request["status"] = "completed" if success else "failed"
        request["completed_at"] = datetime.now(timezone.utc).isoformat()
        
        with open(APPROVAL_REQUEST_FILE, 'w') as f:
            json.dump(request, f, indent=2)
        
        # Archive to history
        self._record_trigger(request)
        
        return success
    
    def reject_request(self, reason: str = "Manually rejected") -> None:
        """
        Reject pending approval request.
        
        Args:
            reason: Rejection reason
        """
        request = self.check_approval()
        if not request:
            print("No pending approval request")
            return
        
        request["status"] = "rejected"
        request["rejected_at"] = datetime.now(timezone.utc).isoformat()
        request["rejection_reason"] = reason
        
        with open(APPROVAL_REQUEST_FILE, 'w') as f:
            json.dump(request, f, indent=2)
        
        # Archive to history
        self._record_trigger(request)
        
        print(f"✅ Request rejected: {reason}")
    
    # =========================================================================
    # LEARNING LOOP EXECUTION
    # =========================================================================
    
    def execute_learning_loop(
        self,
        steps: List[int] = None,
        params: Dict[str, Any] = None
    ) -> bool:
        """
        Execute the learning loop (partial pipeline rerun).
        
        Default: Steps 3→5→6 (label refresh → retrain → predict)
        
        Args:
            steps: List of steps to run (default: [3, 5, 6])
            params: Optional parameter overrides
        
        Returns:
            True if all steps completed successfully
        """
        if steps is None:
            steps = [3, 5, 6]
        
        print(f"\n{'='*60}")
        print(f"🔄 CHAPTER 13 LEARNING LOOP — Steps {steps}")
        print(f"{'='*60}\n")
        
        if self.watcher_agent is None:
            logger.error("No WatcherAgent reference - cannot execute pipeline")
            print("❌ Error: WatcherAgent not connected")
            print("   Run learning loop via watcher_agent.py instead:")
            print(f"   python3 watcher_agent.py --run-pipeline --start-step {steps[0]} --end-step {steps[-1]}")
            return False
        
        # Execute via WatcherAgent
        try:
            start_step = min(steps)
            end_step = max(steps)
            
            logger.info(f"Executing learning loop: steps {start_step} to {end_step}")
            self.watcher_agent.run_pipeline(start_step, end_step, params)
            
            # Update cooldown state
            self.cooldown.last_retrain_at = datetime.now(timezone.utc).isoformat()
            self.cooldown.runs_since_retrain = 0
            self.cooldown.save()
            
            logger.info("Learning loop completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Learning loop failed: {e}")
            return False
    
    def execute_standalone(self, steps: List[int]) -> bool:
        """
        Execute learning loop without WatcherAgent (standalone mode).
        
        Uses subprocess to call pipeline scripts directly.
        
        Args:
            steps: List of steps to run
        
        Returns:
            True if all steps completed
        """
        import subprocess
        
        STEP_SCRIPTS = {
            1: "window_optimizer.py",
            2: "run_scorer_meta_optimizer.sh",
            3: "run_step3_full_scoring.sh",
            4: "adaptive_meta_optimizer.py",
            5: "meta_prediction_optimizer_anti_overfit.py",
            6: "prediction_generator.py"
        }
        
        print(f"\n{'='*60}")
        print(f"🔄 STANDALONE LEARNING LOOP — Steps {steps}")
        print(f"{'='*60}\n")
        
        for step in sorted(steps):
            script = STEP_SCRIPTS.get(step)
            if not script:
                logger.error(f"Unknown step: {step}")
                return False
            
            print(f"\n--- Step {step}: {script} ---")
            
            # Determine command
            if script.endswith(".sh"):
                cmd = ["bash", script]
            else:
                cmd = ["python3", script]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=False,  # Stream output
                    timeout=7200  # 2 hour timeout
                )
                
                if result.returncode != 0:
                    logger.error(f"Step {step} failed with code {result.returncode}")
                    return False
                    
            except subprocess.TimeoutExpired:
                logger.error(f"Step {step} timed out")
                return False
            except Exception as e:
                logger.error(f"Step {step} error: {e}")
                return False
        
        # Update cooldown
        self.cooldown.last_retrain_at = datetime.now(timezone.utc).isoformat()
        self.cooldown.runs_since_retrain = 0
        self.cooldown.save()
        
        print(f"\n{'='*60}")
        print("✅ Learning loop completed")
        print(f"{'='*60}\n")
        
        return True
    
    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================
    
    def increment_run_counter(self) -> None:
        """
        Increment the runs-since-retrain counter.
        
        Call this after each new draw is processed.
        """
        self.cooldown.runs_since_retrain += 1
        self.cooldown.runs_since_param_change += 1
        self.cooldown.save()
        
        logger.debug(f"Run counter: {self.cooldown.runs_since_retrain} since retrain")
    
    def _record_trigger(self, request: Dict[str, Any]) -> None:
        """Record trigger to history."""
        self.trigger_history.append({
            "request_id": request.get("request_id"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trigger_type": request.get("evaluation", {}).get("trigger_type"),
            "action": request.get("action_required"),
            "status": request.get("status"),
            "steps": request.get("steps_to_run", [])
        })
        
        # Keep last 100 triggers
        self.trigger_history = self.trigger_history[-100:]
        self._save_trigger_history()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current trigger manager status."""
        pending = self.check_approval()
        
        return {
            "cooldown": self.cooldown.to_dict(),
            "pending_approval": pending is not None,
            "pending_request_id": pending.get("request_id") if pending else None,
            "trigger_history_count": len(self.trigger_history),
            "last_trigger": self.trigger_history[-1] if self.trigger_history else None,
            "policies_loaded": bool(self.policies),
            "watcher_connected": self.watcher_agent is not None
        }


# =============================================================================
# CLI
# =============================================================================

    # =========================================================================
    # Phase 9A.3: Selfplay Retrain Trigger
    # =========================================================================
    #
    # PHASE 9A.3 ONLY:
    # This method creates a retrain request artifact.
    # It does NOT evaluate learning quality.
    # It does NOT trigger execution.
    # WATCHER remains the sole execution authority.
    #
    # =========================================================================
    
    def request_selfplay(
        self, 
        reason: str,
        source_policy: Optional[str] = None,
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """
        Request selfplay exploration via WATCHER.
        
        Phase 9A.3: Chapter 13 can request selfplay retraining, but:
        - Does NOT directly invoke selfplay_orchestrator.py
        - WATCHER is the gate that authorizes execution
        - Selfplay decides its own exploration strategy
        
        Args:
            reason: Why selfplay is being requested
            source_policy: Policy ID that triggered this (if any)
            priority: "normal" or "high"
            
        Returns:
            Request record for audit
        """
        request_id = f"selfplay_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        request = {
            "request_id": request_id,
            "request_type": TriggerType.SELFPLAY_RETRAIN.value,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending",
            "reason": reason,
            "source_policy": source_policy,
            "priority": priority,
            "requested_by": "chapter_13_triggers",
            "requires_watcher_approval": True  # WATCHER must authorize
        }
        
        # FIX #1: Unique request files (append-only audit)
        requests_dir = Path("watcher_requests")
        requests_dir.mkdir(exist_ok=True)
        request_file = requests_dir / f"{request_id}.json"
        
        with open(request_file, 'w') as f:
            json.dump(request, f, indent=2)
        
        logger.info(f"Selfplay retrain requested: {reason}")
        logger.info(f"   Request file: {request_file}")
        logger.info(f"   Source policy: {source_policy or 'none'}")
        logger.info(f"   ⚠️  Awaiting WATCHER authorization")
        
        # Record in trigger history with consistent type
        self._record_trigger({
            "type": TriggerType.SELFPLAY_RETRAIN.value,
            "request": request
        })
        
        return request
    
    def should_request_selfplay(self, diagnostics: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Evaluate whether to request selfplay retraining.
        
        Phase 9A.3: Only gate on EXPLICIT signals, not analysis.
        Policy interpretation belongs in Phase 9B or WATCHER.
        
        Returns:
            (should_request, reason)
        """
        # FIX #2: Only explicit triggers, no policy heuristics
        
        # Check for explicit flag from diagnostics
        summary_flags = diagnostics.get("summary_flags", [])
        if "SELFPLAY_RECOMMENDED" in summary_flags:
            return True, "Diagnostics flagged SELFPLAY_RECOMMENDED"
        
        # Check for explicit flag from recommended_actions
        recommended = diagnostics.get("recommended_actions", {})
        if recommended.get("request_selfplay", False):
            return True, "Recommended actions included request_selfplay"
        
        # No explicit trigger
        return False, "No explicit selfplay trigger"



def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Chapter 13 Retrain Triggers — Evaluate and execute learning loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  --evaluate         Evaluate triggers against current diagnostics (default)
  --approve          Approve pending request and execute learning loop
  --reject           Reject pending request
  --execute          Execute learning loop immediately (bypasses approval)
  --status           Show current trigger manager status

Examples:
  python3 chapter_13_triggers.py --evaluate
  python3 chapter_13_triggers.py --approve
  python3 chapter_13_triggers.py --reject --reason "Not ready for retrain"
  python3 chapter_13_triggers.py --execute --steps 3 5 6
  python3 chapter_13_triggers.py --status
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--evaluate", action="store_true", default=True,
                           help="Evaluate triggers (default)")
    mode_group.add_argument("--approve", action="store_true",
                           help="Approve pending request")
    mode_group.add_argument("--reject", action="store_true",
                           help="Reject pending request")
    mode_group.add_argument("--execute", action="store_true",
                           help="Execute learning loop immediately")
    mode_group.add_argument("--status", action="store_true",
                           help="Show status")
    
    # Options
    parser.add_argument("--reason", type=str, default="Manually rejected",
                       help="Rejection reason (for --reject)")
    parser.add_argument("--steps", type=int, nargs="+", default=[3, 5, 6],
                       help="Steps to run (for --execute, default: 3 5 6)")
    parser.add_argument("--diagnostics", type=str, default=DEFAULT_DIAGNOSTICS,
                       help=f"Diagnostics file (default: {DEFAULT_DIAGNOSTICS})")
    parser.add_argument("--policies", type=str, default=DEFAULT_POLICIES,
                       help=f"Policies file (default: {DEFAULT_POLICIES})")
    parser.add_argument("--request-approval", action="store_true",
                       help="Create approval request if trigger fires")
    
    args = parser.parse_args()
    
    # Create manager
    manager = Chapter13TriggerManager(
        watcher_agent=None,  # Standalone mode
        policies_path=args.policies,
        diagnostics_path=args.diagnostics
    )
    
    try:
        if args.status:
            status = manager.get_status()
            print(f"\n{'='*60}")
            print("CHAPTER 13 TRIGGER MANAGER — Status")
            print(f"{'='*60}")
            print(f"\n📊 Cooldown State:")
            print(f"   Runs since retrain: {status['cooldown']['runs_since_retrain']}")
            print(f"   Last retrain: {status['cooldown']['last_retrain_at'] or 'Never'}")
            print(f"\n📋 Approval:")
            print(f"   Pending: {status['pending_approval']}")
            if status['pending_request_id']:
                print(f"   Request ID: {status['pending_request_id']}")
            print(f"\n📜 History:")
            print(f"   Triggers recorded: {status['trigger_history_count']}")
            if status['last_trigger']:
                print(f"   Last trigger: {status['last_trigger']['trigger_type']} ({status['last_trigger']['status']})")
            print()
            return 0
        
        if args.approve:
            success = manager.approve_request()
            return 0 if success else 1
        
        if args.reject:
            manager.reject_request(args.reason)
            return 0
        
        if args.execute:
            success = manager.execute_standalone(args.steps)
            return 0 if success else 1
        
        # Default: evaluate triggers
        evaluation = manager.evaluate_triggers()
        
        print(f"\n{'='*60}")
        print("CHAPTER 13 TRIGGER EVALUATION")
        print(f"{'='*60}")
        print(f"\n📊 Result:")
        print(f"   Should trigger: {evaluation.should_trigger}")
        print(f"   Trigger type: {evaluation.trigger_type.value if evaluation.trigger_type else 'N/A'}")
        print(f"   Action: {evaluation.action.value if evaluation.action else 'N/A'}")
        print(f"   Confidence: {evaluation.confidence:.2f}")
        print(f"   Reasoning: {evaluation.reasoning}")
        
        if evaluation.metrics:
            print(f"\n📈 Metrics:")
            for key, value in evaluation.metrics.items():
                print(f"   {key}: {value}")
        
        if evaluation.cooldown_remaining > 0:
            print(f"\n⏳ Cooldown: {evaluation.cooldown_remaining} runs remaining")
        
        # Create approval request if triggered and requested
        if evaluation.should_trigger and args.request_approval:
            manager.request_approval(evaluation)
        elif evaluation.should_trigger:
            print(f"\n⚠️  Trigger fired! Run with --request-approval to create approval request")
        
        print()
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 99


if __name__ == "__main__":
    sys.exit(main())
