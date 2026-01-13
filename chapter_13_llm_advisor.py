#!/usr/bin/env python3
"""
Chapter 13 LLM Advisor — Phase 4
LLM-based analysis and proposal generation for the feedback loop

ROLE: STRATEGIST (Advisory Only)
The LLM interprets diagnostics and proposes adjustments.
It does NOT execute actions — that is WATCHER's role.

VERSION: 1.0.0
DATE: 2026-01-12

CAPABILITIES:
- Interpret post-draw diagnostics
- Identify cross-run trends
- Propose parameter adjustments
- Flag regime shifts
- Explain performance changes

RESTRICTIONS:
- Cannot modify files
- Cannot execute code
- Cannot apply parameters directly
- Cannot override WATCHER
- Cannot bypass validation

INTEGRATION:
    from chapter_13_llm_advisor import Chapter13LLMAdvisor
    
    advisor = Chapter13LLMAdvisor()
    proposal = advisor.analyze_diagnostics(diagnostics)
    
    # WATCHER validates before execution
    if watcher.validate_proposal(proposal):
        watcher.execute_proposal(proposal)
"""

import json
import os
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

# Local imports
from llm_proposal_schema import (
    LLMProposal,
    DiagnosticsSummary,
    RunHistoryEntry,
    FailureMode,
    RecommendedAction,
    RetrainScope,
    RiskLevel,
    ParameterProposal,
    parse_llm_response_to_proposal,
    create_empty_proposal
)

# Try to import LLM Router
try:
    from llm_services.llm_router import LLMRouter, get_router
    LLM_ROUTER_AVAILABLE = True
except ImportError:
    LLM_ROUTER_AVAILABLE = False
    LLMRouter = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Chapter13LLMAdvisor")


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_DIAGNOSTICS = "post_draw_diagnostics.json"
DEFAULT_HISTORY = "watcher_history.json"
PROPOSALS_ARCHIVE = "llm_proposals"
GRAMMAR_FILE = "chapter_13.gbnf"

# LLM settings
DEFAULT_MODEL = "deepseek-r1-14b"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 1000


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

SYSTEM_PROMPT = """You are an analytical advisor for a probabilistic research system.

HARD CONSTRAINTS:
- You do NOT execute actions
- You do NOT modify parameters directly
- You do NOT assume stationarity
- You MUST express uncertainty

YOUR TASK:
Interpret diagnostic deltas from real-world outcomes and propose cautious, reversible adjustments.

If uncertainty is high, recommend NO CHANGE (action: "WAIT").

CONTEXT:
This system analyzes PRNG (Pseudo-Random Number Generator) patterns to predict lottery outcomes.
The ML model uses:
- Survivor-level features (pattern characteristics from seed+skip combinations)
- Global features (hot/cold numbers, residue distributions, regime indicators)

The system finds PATTERNS, not seeds. It learns which pattern+global combinations predict outcomes.

OUTPUT FORMAT:
You must respond with valid JSON matching this schema:
{
  "analysis_summary": "Brief summary of your analysis",
  "failure_mode": "calibration_drift|feature_relevance|window_misalignment|random_variance|regime_shift|model_overfit|data_quality|none_detected",
  "confidence": 0.0-1.0,
  "recommended_action": "RETRAIN|WAIT|ESCALATE|FULL_RESET",
  "retrain_scope": "steps_3_5_6|steps_5_6|step_6_only|full_pipeline" or null,
  "parameter_proposals": [...],
  "risk_level": "low|medium|high",
  "requires_human_review": true|false,
  "alternative_hypothesis": "string or null"
}"""


USER_PROMPT_TEMPLATE = """Given the following post-draw diagnostics:

CURRENT DRAW: {draw_id}
TIMESTAMP: {draw_timestamp}
FINGERPRINT: {data_fingerprint}

=== PREDICTION VALIDATION ===
- Exact hits: {exact_hits} / {pool_size}
- Hit rate: {hit_rate:.4f}
- Best rank: {best_rank}
- Pool coverage: {pool_coverage:.4f}

=== CONFIDENCE CALIBRATION ===
- Mean confidence: {mean_confidence:.4f}
- Confidence correlation: {confidence_correlation:.4f}
- Overconfident: {overconfident}
- Underconfident: {underconfident}

=== PIPELINE HEALTH ===
- Consecutive misses: {consecutive_misses}
- Model stability: {model_stability}
- Window decay: {window_decay:.4f}
- Survivor churn: {survivor_churn:.4f}

=== SUMMARY FLAGS ===
{summary_flags}

=== PREVIOUS RUN SUMMARIES (last 5) ===
{run_history}

TASKS:
1. Identify the most likely failure mode (if any)
2. Classify the issue:
   - calibration_drift: Model confidence doesn't match reality
   - feature_relevance: Features losing predictive power
   - window_misalignment: History window not optimal
   - random_variance: Normal statistical noise
   - regime_shift: Fundamental pattern change
   - model_overfit: Model too fitted to training data
   - data_quality: Issues with input data
   - none_detected: System operating normally
3. Propose parameter adjustments ONLY if justified
4. Assign confidence score (0.0-1.0) to the analysis
5. Recommend: RETRAIN / WAIT / ESCALATE / FULL_RESET

Respond with valid JSON only."""


# =============================================================================
# LLM ADVISOR CLASS
# =============================================================================

class Chapter13LLMAdvisor:
    """
    LLM-based advisor for Chapter 13 feedback loop.
    
    This class manages:
    - Building prompts from diagnostics
    - Calling LLM with grammar constraints
    - Parsing responses into validated proposals
    - Archiving proposals for audit
    
    The advisor is STATELESS — all context comes from inputs.
    """
    
    def __init__(
        self,
        use_grammar: bool = True,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE
    ):
        self.use_grammar = use_grammar
        self.model = model
        self.temperature = temperature
        
        # Initialize LLM router if available
        self.router = None
        if LLM_ROUTER_AVAILABLE:
            try:
                self.router = get_router()
                logger.info(f"LLMRouter initialized for Chapter 13 advisor")
            except Exception as e:
                logger.warning(f"Could not initialize LLMRouter: {e}")
        
        # Create proposals archive directory
        Path(PROPOSALS_ARCHIVE).mkdir(exist_ok=True)
        
        logger.info(f"Chapter13LLMAdvisor initialized (router={'available' if self.router else 'unavailable'})")
    
    def analyze_diagnostics(
        self,
        diagnostics: Dict[str, Any],
        run_history: List[Dict[str, Any]] = None
    ) -> LLMProposal:
        """
        Analyze diagnostics and generate proposal.
        
        Args:
            diagnostics: Post-draw diagnostics dict
            run_history: Optional list of recent run summaries
        
        Returns:
            LLMProposal with analysis and recommendations
        """
        logger.info(f"Analyzing diagnostics for draw {diagnostics.get('draw_id', 'unknown')}")
        
        # Build prompt
        prompt = self._build_prompt(diagnostics, run_history or [])
        
        # Get LLM response
        if self.router:
            proposal = self._analyze_with_router(prompt, diagnostics)
        else:
            proposal = self._analyze_with_heuristic(diagnostics)
        
        # Archive proposal
        self._archive_proposal(proposal, diagnostics)
        
        return proposal
    
    def _build_prompt(
        self,
        diagnostics: Dict[str, Any],
        run_history: List[Dict[str, Any]]
    ) -> str:
        """Build user prompt from diagnostics."""
        pv = diagnostics.get("prediction_validation", {})
        cc = diagnostics.get("confidence_calibration", {})
        ph = diagnostics.get("pipeline_health", {})
        
        # Format run history
        history_str = "No previous runs available"
        if run_history:
            history_lines = []
            for run in run_history[-5:]:
                history_lines.append(
                    f"  - {run.get('run_id', 'N/A')}: "
                    f"hit_rate={run.get('hit_rate', 'N/A')}, "
                    f"action={run.get('action_taken', 'N/A')}"
                )
            history_str = "\n".join(history_lines)
        
        # Format summary flags
        flags = diagnostics.get("summary_flags", [])
        flags_str = "\n".join(f"  - {f}" for f in flags) if flags else "  (none)"
        
        return USER_PROMPT_TEMPLATE.format(
            draw_id=diagnostics.get("draw_id", "unknown"),
            draw_timestamp=diagnostics.get("draw_timestamp", ""),
            data_fingerprint=diagnostics.get("data_fingerprint", ""),
            exact_hits=pv.get("exact_hits", 0),
            pool_size=pv.get("pool_size", 0),
            hit_rate=pv.get("exact_hits", 0) / max(pv.get("pool_size", 1), 1),
            best_rank=pv.get("best_rank", "N/A"),
            pool_coverage=pv.get("pool_coverage", 0),
            mean_confidence=cc.get("mean_confidence", 0),
            confidence_correlation=cc.get("predicted_vs_actual_correlation", 0),
            overconfident=cc.get("overconfidence_detected", False),
            underconfident=cc.get("underconfidence_detected", False),
            consecutive_misses=ph.get("consecutive_misses", 0),
            model_stability=ph.get("model_stability", "unknown"),
            window_decay=ph.get("window_decay", 0),
            survivor_churn=ph.get("survivor_churn", 0),
            summary_flags=flags_str,
            run_history=history_str
        )
    
    def _analyze_with_router(
        self,
        prompt: str,
        diagnostics: Dict[str, Any]
    ) -> LLMProposal:
        """Analyze using LLMRouter with grammar constraints."""
        try:
            # Build full prompt with system context
            full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
            
            # Grammar filename (must exist in grammars/ directory)
            grammar_filename = "chapter_13.gbnf"
            grammars_dir = Path("grammars")
            grammar_available = self.use_grammar and (grammars_dir / grammar_filename).exists()
            
            # Call LLM
            if grammar_available:
                logger.info("Using grammar-constrained decoding")
                response = self.router._call_primary_with_grammar(
                    prompt=full_prompt,
                    grammar=grammar_filename,
                    temperature=self.temperature,
                    max_tokens=DEFAULT_MAX_TOKENS
                )
            else:
                logger.info("Using standard completion (no grammar)")
                response = self.router.route(
                    prompt=full_prompt,
                    temperature=self.temperature,
                    max_tokens=DEFAULT_MAX_TOKENS
                )
            
            # Parse response
            proposal = parse_llm_response_to_proposal(
                response,
                diagnostics_fingerprint=diagnostics.get("data_fingerprint"),
                model_id=self.model
            )
            
            logger.info(f"LLM proposal: {proposal.recommended_action.value} "
                       f"(confidence={proposal.confidence:.2f})")
            return proposal
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._analyze_with_heuristic(diagnostics)
    
    def _analyze_with_heuristic(
        self,
        diagnostics: Dict[str, Any]
    ) -> LLMProposal:
        """
        Fallback heuristic analysis when LLM unavailable.
        
        This provides basic analysis based on diagnostic flags
        without requiring LLM inference.
        """
        logger.info("Using heuristic analysis (LLM unavailable)")
        
        flags = diagnostics.get("summary_flags", [])
        ph = diagnostics.get("pipeline_health", {})
        cc = diagnostics.get("confidence_calibration", {})
        
        # Determine failure mode
        failure_mode = FailureMode.NONE_DETECTED
        if "CONFIDENCE_DRIFT" in flags:
            failure_mode = FailureMode.CALIBRATION_DRIFT
        elif "HIGH_FEATURE_TURNOVER" in flags:
            failure_mode = FailureMode.FEATURE_RELEVANCE
        elif "HIGH_WINDOW_DECAY" in flags:
            failure_mode = FailureMode.WINDOW_MISALIGNMENT
        elif "REGIME_SHIFT_POSSIBLE" in flags:
            failure_mode = FailureMode.REGIME_SHIFT
        elif "WEAK_SIGNAL" in flags:
            failure_mode = FailureMode.RANDOM_VARIANCE
        
        # Determine action
        action = RecommendedAction.WAIT
        retrain_scope = None
        
        if "RETRAIN_RECOMMENDED" in flags:
            action = RecommendedAction.RETRAIN
            retrain_scope = RetrainScope.STEPS_3_5_6
        elif "REGIME_SHIFT_POSSIBLE" in flags:
            action = RecommendedAction.ESCALATE
            retrain_scope = RetrainScope.FULL_PIPELINE
        elif "MODEL_DEGRADED" in flags:
            action = RecommendedAction.RETRAIN
            retrain_scope = RetrainScope.STEPS_5_6
        
        # Determine risk
        risk = RiskLevel.LOW
        if "REGIME_SHIFT_POSSIBLE" in flags:
            risk = RiskLevel.HIGH
        elif len([f for f in flags if "HIGH" in f or "DEGRADED" in f]) > 0:
            risk = RiskLevel.MEDIUM
        
        # Build parameter proposals
        proposals = []
        if cc.get("overconfidence_detected"):
            proposals.append(ParameterProposal(
                parameter="confidence_threshold",
                current_value=None,
                proposed_value=0.6,
                delta="-0.1",
                confidence=0.7,
                rationale="Overconfidence detected - lower threshold"
            ))
        if cc.get("underconfidence_detected"):
            proposals.append(ParameterProposal(
                parameter="confidence_threshold",
                current_value=None,
                proposed_value=0.8,
                delta="+0.1",
                confidence=0.7,
                rationale="Underconfidence detected - raise threshold"
            ))
        
        # Build summary
        if flags:
            summary = f"Heuristic analysis detected: {', '.join(flags[:3])}"
        else:
            summary = "No significant issues detected via heuristic analysis"
        
        return LLMProposal(
            analysis_summary=summary,
            failure_mode=failure_mode,
            confidence=0.6,  # Lower confidence for heuristic
            recommended_action=action,
            retrain_scope=retrain_scope,
            parameter_proposals=proposals[:3],  # Max 3
            risk_level=risk,
            requires_human_review=(risk != RiskLevel.LOW),
            alternative_hypothesis="Heuristic analysis - LLM unavailable",
            generated_at=datetime.now(timezone.utc).isoformat(),
            diagnostics_fingerprint=diagnostics.get("data_fingerprint"),
            model_id="heuristic_v1"
        )
    
    def _archive_proposal(
        self,
        proposal: LLMProposal,
        diagnostics: Dict[str, Any]
    ) -> str:
        """Archive proposal for audit trail."""
        timestamp = datetime.now(timezone.utc)
        filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_proposal.json"
        filepath = os.path.join(PROPOSALS_ARCHIVE, filename)
        
        archive_data = {
            "proposal": proposal.model_dump(),
            "diagnostics_summary": {
                "draw_id": diagnostics.get("draw_id"),
                "fingerprint": diagnostics.get("data_fingerprint"),
                "flags": diagnostics.get("summary_flags", [])
            },
            "archived_at": timestamp.isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(archive_data, f, indent=2)
        
        logger.debug(f"Proposal archived: {filepath}")
        return filepath
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get advisor status."""
        proposal_files = list(Path(PROPOSALS_ARCHIVE).glob("*_proposal.json"))
        
        return {
            "router_available": self.router is not None,
            "grammar_available": os.path.exists(GRAMMAR_FILE),
            "use_grammar": self.use_grammar,
            "model": self.model,
            "temperature": self.temperature,
            "archived_proposals": len(proposal_files),
            "latest_proposal": proposal_files[-1].name if proposal_files else None
        }
    
    def load_recent_proposals(self, n: int = 5) -> List[Dict[str, Any]]:
        """Load recent archived proposals."""
        proposal_files = sorted(
            Path(PROPOSALS_ARCHIVE).glob("*_proposal.json"),
            reverse=True
        )[:n]
        
        proposals = []
        for f in proposal_files:
            with open(f, 'r') as fp:
                proposals.append(json.load(fp))
        
        return proposals


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Chapter 13 LLM Advisor — Diagnostic analysis and proposal generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  --analyze          Analyze current diagnostics and generate proposal (default)
  --status           Show advisor status
  --history          Show recent proposals

Examples:
  python3 chapter_13_llm_advisor.py --analyze
  python3 chapter_13_llm_advisor.py --analyze --diagnostics custom.json
  python3 chapter_13_llm_advisor.py --status
  python3 chapter_13_llm_advisor.py --history --count 10
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--analyze", action="store_true", default=True,
                           help="Analyze diagnostics (default)")
    mode_group.add_argument("--status", action="store_true",
                           help="Show advisor status")
    mode_group.add_argument("--history", action="store_true",
                           help="Show recent proposals")
    
    # Options
    parser.add_argument("--diagnostics", type=str, default=DEFAULT_DIAGNOSTICS,
                       help=f"Diagnostics file (default: {DEFAULT_DIAGNOSTICS})")
    parser.add_argument("--no-grammar", action="store_true",
                       help="Disable grammar-constrained decoding")
    parser.add_argument("--count", type=int, default=5,
                       help="Number of proposals to show (for --history)")
    parser.add_argument("--json", action="store_true",
                       help="Output as JSON")
    
    args = parser.parse_args()
    
    # Create advisor
    advisor = Chapter13LLMAdvisor(use_grammar=not args.no_grammar)
    
    try:
        if args.status:
            status = advisor.get_status()
            if args.json:
                print(json.dumps(status, indent=2))
            else:
                print(f"\n{'='*60}")
                print("CHAPTER 13 LLM ADVISOR — Status")
                print(f"{'='*60}")
                print(f"\n⚙️  Configuration:")
                print(f"   Router available: {status['router_available']}")
                print(f"   Grammar available: {status['grammar_available']}")
                print(f"   Use grammar: {status['use_grammar']}")
                print(f"   Model: {status['model']}")
                print(f"   Temperature: {status['temperature']}")
                print(f"\n📦 Archive:")
                print(f"   Proposals stored: {status['archived_proposals']}")
                print(f"   Latest: {status['latest_proposal'] or 'None'}")
                print()
            return 0
        
        if args.history:
            proposals = advisor.load_recent_proposals(args.count)
            if args.json:
                print(json.dumps(proposals, indent=2))
            else:
                print(f"\n{'='*60}")
                print(f"CHAPTER 13 LLM ADVISOR — Recent Proposals (last {args.count})")
                print(f"{'='*60}")
                for p in proposals:
                    prop = p.get("proposal", {})
                    print(f"\n📋 {p.get('archived_at', 'N/A')}")
                    print(f"   Action: {prop.get('recommended_action')}")
                    print(f"   Confidence: {prop.get('confidence')}")
                    print(f"   Failure mode: {prop.get('failure_mode')}")
                    print(f"   Summary: {prop.get('analysis_summary', 'N/A')[:80]}...")
                print()
            return 0
        
        # Default: analyze
        if not os.path.exists(args.diagnostics):
            print(f"❌ Diagnostics file not found: {args.diagnostics}")
            print("   Run chapter_13_diagnostics.py --generate first")
            return 1
        
        with open(args.diagnostics, 'r') as f:
            diagnostics = json.load(f)
        
        proposal = advisor.analyze_diagnostics(diagnostics)
        
        if args.json:
            print(json.dumps(proposal.model_dump(), indent=2))
        else:
            print(f"\n{'='*60}")
            print("CHAPTER 13 LLM ADVISOR — Analysis Result")
            print(f"{'='*60}")
            print(f"\n📊 Analysis:")
            print(f"   Summary: {proposal.analysis_summary}")
            print(f"   Failure mode: {proposal.failure_mode.value}")
            print(f"   Confidence: {proposal.confidence:.2f}")
            print(f"\n🎯 Recommendation:")
            print(f"   Action: {proposal.recommended_action.value}")
            print(f"   Scope: {proposal.retrain_scope.value if proposal.retrain_scope else 'N/A'}")
            print(f"   Risk level: {proposal.risk_level.value}")
            print(f"   Human review: {proposal.requires_human_review}")
            if proposal.parameter_proposals:
                print(f"\n📝 Parameter Proposals:")
                for p in proposal.parameter_proposals:
                    print(f"   - {p.parameter}: {p.delta} (confidence={p.confidence:.2f})")
                    print(f"     Rationale: {p.rationale}")
            if proposal.alternative_hypothesis:
                print(f"\n🔄 Alternative: {proposal.alternative_hypothesis}")
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
