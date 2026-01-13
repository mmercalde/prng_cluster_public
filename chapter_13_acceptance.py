#!/usr/bin/env python3
"""
Chapter 13 Acceptance Engine — Phase 5
Validates LLM proposals against WATCHER policies

RESPONSIBILITIES:
1. Validate proposal confidence thresholds
2. Enforce parameter change magnitude limits
3. Check frozen parameter violations
4. Track parameter change history (reversal detection)
5. Enforce cooldown periods
6. Determine: ACCEPT / REJECT / ESCALATE

VERSION: 1.0.0
DATE: 2026-01-12

AUTHORITY:
The Acceptance Engine enforces WATCHER policies. LLM proposals are
ADVISORY ONLY — this engine is the gatekeeper for all changes.

Flow: LLM Proposal → Acceptance Engine → ACCEPT/REJECT/ESCALATE
"""

import json
import os
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

# Local imports
from llm_proposal_schema import (
    LLMProposal,
    ParameterProposal,
    RecommendedAction,
    RiskLevel
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Chapter13Acceptance")


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_POLICIES = "watcher_policies.json"
PARAMETER_HISTORY_FILE = ".parameter_change_history.json"
ACCEPTANCE_LOG_FILE = "acceptance_decisions.jsonl"

# Frozen parameters - these CANNOT be modified by LLM proposals
# Steps 1, 2, 4 are considered stable and should not be auto-modified
FROZEN_PARAMETERS = frozenset([
    # Step 1 parameters
    "window_start",
    "window_end", 
    "window_size",
    # Step 2 parameters
    "scorer_weights",
    "scorer_config",
    # Step 4 parameters
    "ml_architecture",
    "ml_hyperparameters",
    "feature_selection"
])


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class ValidationResult(str, Enum):
    """Possible validation outcomes."""
    ACCEPT = "accept"
    REJECT = "reject"
    ESCALATE = "escalate"


class RejectionReason(str, Enum):
    """Standardized rejection reasons."""
    LOW_CONFIDENCE = "low_confidence"
    HIGH_RISK = "high_risk"
    DELTA_TOO_LARGE = "delta_too_large"
    TOO_MANY_PARAMS = "too_many_params"
    FROZEN_PARAMETER = "frozen_parameter"
    COOLDOWN_ACTIVE = "cooldown_active"
    REVERSAL_DETECTED = "reversal_detected"
    POLICY_VIOLATION = "policy_violation"


@dataclass
class AcceptanceDecision:
    """Result of proposal validation."""
    result: ValidationResult
    reason: str
    violations: List[str] = field(default_factory=list)
    proposal_id: Optional[str] = None
    timestamp: Optional[str] = None
    
    # For accepted proposals
    approved_changes: List[Dict[str, Any]] = field(default_factory=list)
    
    # For escalated proposals
    escalation_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "result": self.result.value,
            "reason": self.reason,
            "violations": self.violations,
            "proposal_id": self.proposal_id,
            "timestamp": self.timestamp,
            "approved_changes": self.approved_changes,
            "escalation_reasons": self.escalation_reasons
        }


@dataclass
class ParameterChange:
    """Record of a parameter change."""
    parameter: str
    old_value: Optional[float]
    new_value: float
    delta: str
    timestamp: str
    proposal_id: str
    run_number: int


@dataclass
class ParameterHistory:
    """Tracks parameter changes for reversal detection."""
    changes: List[Dict[str, Any]] = field(default_factory=list)
    current_run: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterHistory':
        return cls(
            changes=data.get("changes", []),
            current_run=data.get("current_run", 0)
        )
    
    @classmethod
    def load(cls) -> 'ParameterHistory':
        if os.path.exists(PARAMETER_HISTORY_FILE):
            with open(PARAMETER_HISTORY_FILE, 'r') as f:
                return cls.from_dict(json.load(f))
        return cls()
    
    def save(self) -> None:
        with open(PARAMETER_HISTORY_FILE, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def add_change(self, change: ParameterChange) -> None:
        self.changes.append({
            "parameter": change.parameter,
            "old_value": change.old_value,
            "new_value": change.new_value,
            "delta": change.delta,
            "timestamp": change.timestamp,
            "proposal_id": change.proposal_id,
            "run_number": change.run_number
        })
        # Keep last 100 changes
        self.changes = self.changes[-100:]
        self.save()
    
    def get_recent_changes(self, parameter: str, n_runs: int = 3) -> List[Dict[str, Any]]:
        """Get recent changes for a parameter within last n runs."""
        min_run = self.current_run - n_runs
        return [
            c for c in self.changes
            if c["parameter"] == parameter and c["run_number"] >= min_run
        ]
    
    def would_reverse(self, parameter: str, new_value: float, n_runs: int = 3) -> bool:
        """Check if proposed change would reverse a recent change."""
        recent = self.get_recent_changes(parameter, n_runs)
        if not recent:
            return False
        
        # Check if we'd be going back to a previous value
        for change in recent:
            old_val = change.get("old_value")
            if old_val is not None and abs(new_value - old_val) < 0.001:
                return True
        
        return False
    
    def increment_run(self) -> None:
        self.current_run += 1
        self.save()


# =============================================================================
# ACCEPTANCE ENGINE
# =============================================================================

class Chapter13AcceptanceEngine:
    """
    Validates LLM proposals against WATCHER policies.
    
    This engine is the SOLE authority for accepting or rejecting
    proposed changes. LLM proposals are advisory only.
    
    Decision Flow:
    1. Check hard rejections (frozen params, delta too large, etc.)
    2. Check automatic escalation conditions
    3. Check automatic acceptance conditions
    4. Default: escalate to human review
    """
    
    def __init__(self, policies_path: str = DEFAULT_POLICIES):
        self.policies_path = policies_path
        self.policies = self._load_policies()
        self.history = ParameterHistory.load()
        
        logger.info("Chapter13AcceptanceEngine initialized")
    
    def _load_policies(self) -> Dict[str, Any]:
        """Load watcher policies."""
        if os.path.exists(self.policies_path):
            with open(self.policies_path, 'r') as f:
                return json.load(f)
        logger.warning(f"Policies not found at {self.policies_path}, using defaults")
        return {}
    
    # =========================================================================
    # MAIN VALIDATION
    # =========================================================================
    
    def validate_proposal(
        self,
        proposal: LLMProposal,
        diagnostics: Optional[Dict[str, Any]] = None
    ) -> AcceptanceDecision:
        """
        Validate an LLM proposal against all policies.
        
        Args:
            proposal: The LLM proposal to validate
            diagnostics: Optional current diagnostics for context
        
        Returns:
            AcceptanceDecision with result and reasoning
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        proposal_id = f"prop_{timestamp.replace(':', '').replace('-', '')[:15]}"
        
        violations = []
        escalation_reasons = []
        
        # Get policy thresholds
        acceptance_rules = self.policies.get("acceptance_rules", {})
        min_confidence = acceptance_rules.get("min_confidence", 0.60)
        max_delta = acceptance_rules.get("max_parameter_delta", 0.30)
        max_params = acceptance_rules.get("max_parameters_per_proposal", 3)
        cooldown_runs = acceptance_rules.get("cooldown_runs", 3)
        
        # =====================================================================
        # HARD REJECTIONS (Section 13.1)
        # =====================================================================
        
        # 1. Confidence too low
        if proposal.confidence < min_confidence:
            violations.append(
                f"Confidence {proposal.confidence:.2f} < minimum {min_confidence}"
            )
            return self._create_decision(
                ValidationResult.REJECT,
                RejectionReason.LOW_CONFIDENCE.value,
                violations,
                proposal_id,
                timestamp
            )
        
        # 2. Too many parameters
        if len(proposal.parameter_proposals) > max_params:
            violations.append(
                f"Too many parameters: {len(proposal.parameter_proposals)} > max {max_params}"
            )
            return self._create_decision(
                ValidationResult.REJECT,
                RejectionReason.TOO_MANY_PARAMS.value,
                violations,
                proposal_id,
                timestamp
            )
        
        # 3. Check each parameter proposal
        for param_prop in proposal.parameter_proposals:
            # Frozen parameter check
            if param_prop.parameter in FROZEN_PARAMETERS:
                violations.append(
                    f"Frozen parameter: {param_prop.parameter}"
                )
                return self._create_decision(
                    ValidationResult.REJECT,
                    RejectionReason.FROZEN_PARAMETER.value,
                    violations,
                    proposal_id,
                    timestamp
                )
            
            # Delta magnitude check
            delta_violation = self._check_delta_magnitude(param_prop, max_delta)
            if delta_violation:
                violations.append(delta_violation)
                return self._create_decision(
                    ValidationResult.REJECT,
                    RejectionReason.DELTA_TOO_LARGE.value,
                    violations,
                    proposal_id,
                    timestamp
                )
            
            # Reversal check
            if self.history.would_reverse(
                param_prop.parameter,
                param_prop.proposed_value,
                cooldown_runs
            ):
                violations.append(
                    f"Would reverse recent change to {param_prop.parameter}"
                )
                return self._create_decision(
                    ValidationResult.REJECT,
                    RejectionReason.REVERSAL_DETECTED.value,
                    violations,
                    proposal_id,
                    timestamp
                )
            
            # Cooldown check
            recent_changes = self.history.get_recent_changes(
                param_prop.parameter, 
                cooldown_runs
            )
            if recent_changes:
                violations.append(
                    f"Cooldown active for {param_prop.parameter}: "
                    f"changed {len(recent_changes)} time(s) in last {cooldown_runs} runs"
                )
                return self._create_decision(
                    ValidationResult.REJECT,
                    RejectionReason.COOLDOWN_ACTIVE.value,
                    violations,
                    proposal_id,
                    timestamp
                )
        
        # =====================================================================
        # MANDATORY ESCALATION (Section 13.3)
        # =====================================================================
        
        # High or medium risk
        if proposal.risk_level in [RiskLevel.HIGH, RiskLevel.MEDIUM]:
            escalation_reasons.append(f"Risk level: {proposal.risk_level.value}")
        
        # LLM requested human review
        if proposal.requires_human_review:
            escalation_reasons.append("LLM requested human review")
        
        # Action is ESCALATE
        if proposal.recommended_action == RecommendedAction.ESCALATE:
            escalation_reasons.append("LLM recommended escalation")
        
        # Check for regime shift flags in diagnostics
        if diagnostics:
            flags = diagnostics.get("summary_flags", [])
            if "REGIME_SHIFT_POSSIBLE" in flags or "CHAOTIC" in flags:
                escalation_reasons.append(f"Diagnostic flags: {[f for f in flags if 'REGIME' in f or 'CHAOTIC' in f]}")
            
            # 3+ consecutive failures
            consecutive = diagnostics.get("pipeline_health", {}).get("consecutive_misses", 0)
            if consecutive >= 3:
                escalation_reasons.append(f"Consecutive failures: {consecutive}")
        
        # If any escalation reasons, escalate
        if escalation_reasons:
            return self._create_decision(
                ValidationResult.ESCALATE,
                "Mandatory escalation",
                violations,
                proposal_id,
                timestamp,
                escalation_reasons=escalation_reasons
            )
        
        # =====================================================================
        # AUTOMATIC ACCEPTANCE (Section 13.2)
        # =====================================================================
        
        # Check all acceptance criteria
        acceptance_criteria = [
            proposal.risk_level == RiskLevel.LOW,
            proposal.confidence >= 0.75,
            len(proposal.parameter_proposals) <= 2,
            not proposal.requires_human_review
        ]
        
        if all(acceptance_criteria):
            # Build list of approved changes
            approved_changes = []
            for param_prop in proposal.parameter_proposals:
                approved_changes.append({
                    "parameter": param_prop.parameter,
                    "current_value": param_prop.current_value,
                    "new_value": param_prop.proposed_value,
                    "delta": param_prop.delta
                })
            
            return self._create_decision(
                ValidationResult.ACCEPT,
                "Passed all validation checks",
                violations,
                proposal_id,
                timestamp,
                approved_changes=approved_changes
            )
        
        # =====================================================================
        # DEFAULT: ESCALATE
        # =====================================================================
        
        # Doesn't meet auto-accept criteria, but no hard rejections
        escalation_reasons.append("Does not meet automatic acceptance criteria")
        
        return self._create_decision(
            ValidationResult.ESCALATE,
            "Manual review recommended",
            violations,
            proposal_id,
            timestamp,
            escalation_reasons=escalation_reasons
        )
    
    def _check_delta_magnitude(
        self,
        param_prop: ParameterProposal,
        max_delta: float
    ) -> Optional[str]:
        """Check if parameter delta exceeds maximum allowed."""
        if param_prop.current_value is None or param_prop.current_value == 0:
            # Can't compute relative delta without current value
            return None
        
        try:
            # Parse delta string
            delta_str = param_prop.delta.replace('+', '').replace('*', '')
            delta_val = abs(float(delta_str))
            
            # Compute relative change
            relative_delta = delta_val / abs(param_prop.current_value)
            
            if relative_delta > max_delta:
                return (
                    f"Delta for {param_prop.parameter}: "
                    f"{relative_delta:.1%} > max {max_delta:.0%}"
                )
        except (ValueError, ZeroDivisionError):
            pass
        
        return None
    
    def _create_decision(
        self,
        result: ValidationResult,
        reason: str,
        violations: List[str],
        proposal_id: str,
        timestamp: str,
        approved_changes: List[Dict[str, Any]] = None,
        escalation_reasons: List[str] = None
    ) -> AcceptanceDecision:
        """Create and log an acceptance decision."""
        decision = AcceptanceDecision(
            result=result,
            reason=reason,
            violations=violations,
            proposal_id=proposal_id,
            timestamp=timestamp,
            approved_changes=approved_changes or [],
            escalation_reasons=escalation_reasons or []
        )
        
        # Log decision
        self._log_decision(decision)
        
        logger.info(f"Validation result: {result.value} - {reason}")
        
        return decision
    
    def _log_decision(self, decision: AcceptanceDecision) -> None:
        """Append decision to log file."""
        with open(ACCEPTANCE_LOG_FILE, 'a') as f:
            f.write(json.dumps(decision.to_dict()) + "\n")
    
    # =========================================================================
    # CHANGE APPLICATION
    # =========================================================================
    
    def record_applied_changes(
        self,
        decision: AcceptanceDecision,
        proposal: LLMProposal
    ) -> None:
        """
        Record that changes from an accepted proposal were applied.
        
        Call this AFTER successfully applying the changes.
        """
        if decision.result != ValidationResult.ACCEPT:
            logger.warning("Cannot record changes for non-accepted decision")
            return
        
        for param_prop in proposal.parameter_proposals:
            change = ParameterChange(
                parameter=param_prop.parameter,
                old_value=param_prop.current_value,
                new_value=param_prop.proposed_value,
                delta=param_prop.delta,
                timestamp=decision.timestamp,
                proposal_id=decision.proposal_id,
                run_number=self.history.current_run
            )
            self.history.add_change(change)
        
        logger.info(f"Recorded {len(proposal.parameter_proposals)} parameter change(s)")
    
    def increment_run(self) -> None:
        """Increment run counter (call after each draw processed)."""
        self.history.increment_run()
        logger.debug(f"Run counter: {self.history.current_run}")
    
    # =========================================================================
    # STATUS & HISTORY
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status."""
        # Count recent decisions
        recent_decisions = {"accept": 0, "reject": 0, "escalate": 0}
        if os.path.exists(ACCEPTANCE_LOG_FILE):
            with open(ACCEPTANCE_LOG_FILE, 'r') as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        result = d.get("result", "")
                        if result in recent_decisions:
                            recent_decisions[result] += 1
                    except json.JSONDecodeError:
                        pass
        
        return {
            "current_run": self.history.current_run,
            "total_changes_tracked": len(self.history.changes),
            "recent_decisions": recent_decisions,
            "policies_loaded": bool(self.policies),
            "frozen_parameters": list(FROZEN_PARAMETERS)[:5] + ["..."]
        }
    
    def get_recent_decisions(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent acceptance decisions."""
        decisions = []
        if os.path.exists(ACCEPTANCE_LOG_FILE):
            with open(ACCEPTANCE_LOG_FILE, 'r') as f:
                for line in f:
                    try:
                        decisions.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return decisions[-n:]


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Chapter 13 Acceptance Engine — Validate LLM proposals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  --validate FILE    Validate a proposal JSON file
  --status           Show engine status
  --history          Show recent decisions
  --test             Run test validation with mock proposal

Examples:
  python3 chapter_13_acceptance.py --status
  python3 chapter_13_acceptance.py --validate proposal.json
  python3 chapter_13_acceptance.py --history --count 20
  python3 chapter_13_acceptance.py --test
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--validate", type=str, metavar="FILE",
                           help="Validate a proposal JSON file")
    mode_group.add_argument("--status", action="store_true",
                           help="Show engine status")
    mode_group.add_argument("--history", action="store_true",
                           help="Show recent decisions")
    mode_group.add_argument("--test", action="store_true",
                           help="Run test validation")
    
    # Options
    parser.add_argument("--policies", type=str, default=DEFAULT_POLICIES,
                       help=f"Policies file (default: {DEFAULT_POLICIES})")
    parser.add_argument("--count", type=int, default=10,
                       help="Number of decisions to show (for --history)")
    parser.add_argument("--json", action="store_true",
                       help="Output as JSON")
    
    args = parser.parse_args()
    
    # Create engine
    engine = Chapter13AcceptanceEngine(policies_path=args.policies)
    
    try:
        if args.status:
            status = engine.get_status()
            if args.json:
                print(json.dumps(status, indent=2))
            else:
                print(f"\n{'='*60}")
                print("CHAPTER 13 ACCEPTANCE ENGINE — Status")
                print(f"{'='*60}")
                print(f"\n📊 State:")
                print(f"   Current run: {status['current_run']}")
                print(f"   Changes tracked: {status['total_changes_tracked']}")
                print(f"\n📋 Decision History:")
                for result, count in status['recent_decisions'].items():
                    print(f"   {result}: {count}")
                print(f"\n🔒 Frozen Parameters (sample):")
                for p in status['frozen_parameters']:
                    print(f"   - {p}")
                print()
            return 0
        
        if args.history:
            decisions = engine.get_recent_decisions(args.count)
            if args.json:
                print(json.dumps(decisions, indent=2))
            else:
                print(f"\n{'='*60}")
                print(f"CHAPTER 13 ACCEPTANCE ENGINE — Recent Decisions")
                print(f"{'='*60}")
                for d in decisions:
                    print(f"\n📋 {d.get('timestamp', 'N/A')}")
                    print(f"   Result: {d.get('result')}")
                    print(f"   Reason: {d.get('reason')}")
                    if d.get('violations'):
                        print(f"   Violations: {d.get('violations')}")
                print()
            return 0
        
        if args.validate:
            with open(args.validate, 'r') as f:
                data = json.load(f)
            proposal = LLMProposal.from_dict(data)
            decision = engine.validate_proposal(proposal)
            
            if args.json:
                print(json.dumps(decision.to_dict(), indent=2))
            else:
                print(f"\n{'='*60}")
                print("CHAPTER 13 ACCEPTANCE ENGINE — Validation Result")
                print(f"{'='*60}")
                print(f"\n📋 Result: {decision.result.value.upper()}")
                print(f"   Reason: {decision.reason}")
                if decision.violations:
                    print(f"   Violations:")
                    for v in decision.violations:
                        print(f"      - {v}")
                if decision.escalation_reasons:
                    print(f"   Escalation reasons:")
                    for r in decision.escalation_reasons:
                        print(f"      - {r}")
                if decision.approved_changes:
                    print(f"   Approved changes:")
                    for c in decision.approved_changes:
                        print(f"      - {c['parameter']}: {c['delta']}")
                print()
            return 0
        
        if args.test:
            # Create test proposal
            from llm_proposal_schema import ParameterProposal, FailureMode
            
            test_proposal = LLMProposal(
                analysis_summary="Test proposal for validation",
                failure_mode=FailureMode.CALIBRATION_DRIFT,
                confidence=0.80,
                recommended_action=RecommendedAction.RETRAIN,
                risk_level=RiskLevel.LOW,
                requires_human_review=False,
                parameter_proposals=[
                    ParameterProposal(
                        parameter="confidence_threshold",
                        current_value=0.7,
                        proposed_value=0.65,
                        delta="-0.05",
                        confidence=0.85,
                        rationale="Test adjustment"
                    )
                ]
            )
            
            print(f"\n{'='*60}")
            print("CHAPTER 13 ACCEPTANCE ENGINE — Test Validation")
            print(f"{'='*60}")
            print(f"\n📝 Test Proposal:")
            print(f"   Confidence: {test_proposal.confidence}")
            print(f"   Risk: {test_proposal.risk_level.value}")
            print(f"   Parameters: {len(test_proposal.parameter_proposals)}")
            
            decision = engine.validate_proposal(test_proposal)
            
            print(f"\n📋 Result: {decision.result.value.upper()}")
            print(f"   Reason: {decision.reason}")
            if decision.violations:
                print(f"   Violations: {decision.violations}")
            if decision.approved_changes:
                print(f"   Approved: {decision.approved_changes}")
            print()
            return 0
        
        # Default: show help
        parser.print_help()
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
