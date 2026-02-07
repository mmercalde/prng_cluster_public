#!/usr/bin/env python3
"""
Chapter 13 Orchestrator — Phase 6
Main orchestration daemon for the live feedback loop

RESPONSIBILITIES:
1. Monitor for new draws (via flag file)
2. Run diagnostics on new data
3. Evaluate retrain triggers
4. Query LLM for analysis (optional)
5. Validate proposals through acceptance engine
6. Execute approved learning loop
7. Log all decisions for audit

VERSION: 1.0.0
DATE: 2026-01-12

AUTHORITY:
The Orchestrator coordinates all Chapter 13 modules but does NOT
override WATCHER policies. Human approval is required for v1.

USAGE:
    # Run daemon mode
    python3 chapter_13_orchestrator.py --daemon

    # Process single cycle (for testing)
    python3 chapter_13_orchestrator.py --once

    # Status check
    python3 chapter_13_orchestrator.py --status

INTEGRATION:
    Can be invoked by watcher_agent.py or run standalone.
"""

import argparse
import json
import os
import sys
import time
import signal
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

# Chapter 13 module imports
from chapter_13_diagnostics import generate_diagnostics, save_diagnostics, load_json_safe
from chapter_13_triggers import Chapter13TriggerManager, TriggerEvaluation, TriggerAction
from chapter_13_llm_advisor import Chapter13LLMAdvisor
from llm_proposal_schema import LLMProposal, RecommendedAction
from chapter_13_acceptance import Chapter13AcceptanceEngine, ValidationResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Chapter13Orchestrator")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Flag files
NEW_DRAW_FLAG = "new_draw.flag"
HALT_FLAG = "/tmp/agent_halt"
CHAPTER_13_HALT = ".chapter13_halt"

# Config files
DEFAULT_POLICIES = "watcher_policies.json"
DEFAULT_HISTORY = "lottery_history.json"

# Output files
ORCHESTRATOR_LOG = "chapter13_orchestrator.log"
CYCLE_HISTORY = "chapter13_cycle_history.jsonl"

# LLM server
LLM_SERVER_SCRIPT = "llm_services/start_llm_servers.sh"


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class Chapter13Orchestrator:
    """
    Main orchestration daemon for Chapter 13 feedback loop.
    
    Coordinates:
    - Draw ingestion (monitors new_draw.flag)
    - Diagnostics (chapter_13_diagnostics.py)
    - Trigger evaluation (chapter_13_triggers.py)
    - LLM analysis (chapter_13_llm_advisor.py)
    - Proposal validation (chapter_13_acceptance.py)
    - Learning loop execution (Steps 3→5→6)
    
    v1 Behavior:
    - Human approval required for all executions
    - LLM is advisory only
    - Cooldowns enforced between actions
    """
    
    def __init__(
        self,
        policies_path: str = DEFAULT_POLICIES,
        use_llm: bool = True,
        auto_start_llm: bool = False
    ):
        self.policies_path = policies_path
        self.use_llm = use_llm
        self.auto_start_llm = auto_start_llm
        
        # Load policies
        self.policies = self._load_policies()
        
        # Initialize sub-modules
        self.trigger_manager = Chapter13TriggerManager(
            watcher_agent=None,  # Standalone mode
            policies_path=policies_path
        )
        
        self.acceptance_engine = Chapter13AcceptanceEngine(
            policies_path=policies_path
        )
        
        self.llm_advisor = None
        if use_llm:
            self.llm_advisor = Chapter13LLMAdvisor()
        
        # Runtime state
        self.running = False
        self.cycles_completed = 0
        self.last_cycle_time = None
        
        logger.info("Chapter13Orchestrator initialized")
        logger.info(f"  LLM: {'enabled' if use_llm else 'disabled'}")
        logger.info(f"  Policies: {policies_path}")
    
    def _load_policies(self) -> Dict[str, Any]:
        """Load watcher policies."""
        if os.path.exists(self.policies_path):
            with open(self.policies_path, 'r') as f:
                return json.load(f)
        return {}
    
    # =========================================================================
    # SAFETY CHECKS
    # =========================================================================
    
    def _check_safety(self) -> bool:
        """Check if it's safe to run."""
        # Check global halt
        if os.path.exists(HALT_FLAG):
            logger.warning(f"Global halt file exists: {HALT_FLAG}")
            return False
        
        # Check chapter 13 specific halt
        if os.path.exists(CHAPTER_13_HALT):
            logger.warning(f"Chapter 13 halt file exists: {CHAPTER_13_HALT}")
            return False
        
        # Check test mode
        if not self.policies.get("test_mode", False):
            # Production mode - extra caution
            logger.info("Running in production mode")
        
        return True
    
    def _check_llm_server(self) -> bool:
        """Check if LLM server is available."""
        if not self.use_llm:
            return True
        
        try:
            import requests
            response = requests.get("http://localhost:8080/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _start_llm_server(self) -> bool:
        """Attempt to start LLM server."""
        if not os.path.exists(LLM_SERVER_SCRIPT):
            logger.warning(f"LLM server script not found: {LLM_SERVER_SCRIPT}")
            return False
        
        try:
            logger.info("Starting LLM server...")
            result = subprocess.run(
                ["bash", LLM_SERVER_SCRIPT],
                capture_output=True,
                timeout=60
            )
            
            # Wait for server to be ready
            time.sleep(5)
            return self._check_llm_server()
            
        except Exception as e:
            logger.error(f"Failed to start LLM server: {e}")
            return False
    
    # =========================================================================
    # FLAG MANAGEMENT
    # =========================================================================
    
    def _check_new_draw_flag(self) -> Optional[Dict[str, Any]]:
        """Check for new_draw.flag and return its contents."""
        if not os.path.exists(NEW_DRAW_FLAG):
            return None
        
        try:
            with open(NEW_DRAW_FLAG, 'r') as f:
                content = f.read().strip()
            
            # Try to parse as JSON
            if content.startswith('{'):
                return json.loads(content)
            
            # Plain text flag
            return {"source": "flag", "content": content}
            
        except Exception as e:
            logger.warning(f"Error reading flag: {e}")
            return {"source": "flag", "error": str(e)}
    
    def _clear_new_draw_flag(self) -> None:
        """Remove the new_draw.flag after processing."""
        if os.path.exists(NEW_DRAW_FLAG):
            os.remove(NEW_DRAW_FLAG)
            logger.debug("Cleared new_draw.flag")
    
    def _create_halt(self, reason: str) -> None:
        """Create Chapter 13 halt file."""
        with open(CHAPTER_13_HALT, 'w') as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()}: {reason}")
        logger.warning(f"Created halt file: {reason}")
    
    # =========================================================================
    # MAIN CYCLE
    # =========================================================================
    
    def run_cycle(self) -> Dict[str, Any]:
        """
        Run a single Chapter 13 feedback cycle.
        
        Flow:
        1. Generate diagnostics from current state
        2. Evaluate retrain triggers
        3. If triggered, get LLM analysis (optional)
        4. Validate any proposals
        5. Execute if approved OR request human approval
        
        Returns:
            Cycle result dict with all decisions
        """
        cycle_start = datetime.now(timezone.utc)
        cycle_id = f"cycle_{cycle_start.strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"CHAPTER 13 CYCLE — {cycle_id}")
        logger.info(f"{'='*60}")
        
        result = {
            "cycle_id": cycle_id,
            "started_at": cycle_start.isoformat(),
            "steps": {},
            "outcome": None
        }
        
        try:
            # Step 1: Generate diagnostics
            logger.info("\n📊 Step 1: Generating diagnostics...")
            diagnostics = generate_diagnostics()
            result["steps"]["diagnostics"] = {
                "success": True,
                "draw_id": diagnostics.get("draw_id"),
                "flags": diagnostics.get("summary_flags", [])
            }
            
            # Save diagnostics
            save_diagnostics(diagnostics)
            
            # Step 2: Evaluate triggers
            logger.info("\n🔍 Step 2: Evaluating triggers...")
            trigger_eval = self.trigger_manager.evaluate_triggers(diagnostics)
            result["steps"]["triggers"] = trigger_eval.to_dict()
            
            # Increment run counter
            self.trigger_manager.increment_run_counter()
            
            if not trigger_eval.should_trigger:
                logger.info("✅ No triggers fired - system healthy")
                result["outcome"] = "no_action_needed"
                self._log_cycle(result)
                return result
            
            logger.info(f"⚠️ Trigger fired: {trigger_eval.trigger_type.value}")
            
            # Step 3: LLM analysis (optional)
            proposal = None
            if self.use_llm and self.llm_advisor:
                logger.info("\n🤖 Step 3: Getting LLM analysis...")
                
                # Check LLM availability
                if not self._check_llm_server():
                    if self.auto_start_llm:
                        self._start_llm_server()
                    else:
                        logger.warning("LLM server not available, skipping analysis")
                
                if self._check_llm_server():
                    proposal = self.llm_advisor.analyze_diagnostics(diagnostics)
                    result["steps"]["llm_analysis"] = proposal.to_dict()
                    logger.info(f"   LLM recommendation: {proposal.recommended_action.value}")
            else:
                logger.info("\n🤖 Step 3: LLM analysis skipped (disabled)")
            
            # Step 4: Validate proposal (if we have one)
            if proposal:
                logger.info("\n✓ Step 4: Validating proposal...")
                validation = self.acceptance_engine.validate_proposal(proposal, diagnostics)
                result["steps"]["validation"] = validation.to_dict()
                
                if validation.result == ValidationResult.ACCEPT:
                    logger.info("✅ Proposal ACCEPTED")
                    result["outcome"] = "proposal_accepted"
                    
                    # SOAK C PATCH: Skip v1 gate in test mode
                    _test_mode = self.policies.get("test_mode", False)
                    _auto_approve = self.policies.get("auto_approve_in_test_mode", False)
                    
                    if _test_mode and _auto_approve:
                        logger.info("🔄 SOAK C: Auto-executing (test mode)")
                        result["outcome"] = "auto_executed_test_mode"
                    else:
                        # v1: Still require human approval even for accepted proposals
                        v1_approval = self.policies.get("v1_approval_required", {})
                        if v1_approval.get("retrain_execution", True):
                            logger.info("📋 v1 mode: Requesting human approval...")
                            self.trigger_manager.request_approval(trigger_eval)
                            result["outcome"] = "pending_approval"
                        else:
                            result["outcome"] = "auto_execute_disabled"
                        
                elif validation.result == ValidationResult.ESCALATE:
                    logger.info("⚠️ Proposal ESCALATED to human review")
                    self.trigger_manager.request_approval(trigger_eval)
                    result["outcome"] = "escalated"
                    
                else:  # REJECT
                    logger.info(f"❌ Proposal REJECTED: {validation.reason}")
                    result["outcome"] = "proposal_rejected"
            else:
                # No LLM proposal, use trigger-based action
                logger.info("\n✓ Step 4: Using trigger-based action (no LLM proposal)")
                
                # Request approval for trigger-based action
                self.trigger_manager.request_approval(trigger_eval)
                result["outcome"] = "pending_approval"
            
        except Exception as e:
            logger.error(f"Cycle error: {e}")
            result["outcome"] = "error"
            result["error"] = str(e)
            import traceback
            traceback.print_exc()
        
        # Finalize
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        result["duration_seconds"] = (
            datetime.now(timezone.utc) - cycle_start
        ).total_seconds()
        
        self._log_cycle(result)
        self.cycles_completed += 1
        self.last_cycle_time = cycle_start
        
        return result
    
    def _log_cycle(self, result: Dict[str, Any]) -> None:
        """Log cycle to history file."""
        with open(CYCLE_HISTORY, 'a') as f:
            f.write(json.dumps(result) + "\n")
    
    # =========================================================================
    # DAEMON MODE
    # =========================================================================
    
    def run_daemon(self, poll_interval: int = None):
        """
        Run as daemon, watching for new draws.
        
        Args:
            poll_interval: Seconds between flag checks (default from policies)
        """
        # Get poll interval from policies if not specified
        if poll_interval is None:
            daemon_settings = self.policies.get("daemon_settings", {})
            poll_interval = daemon_settings.get("poll_interval_seconds", 30)
        
        logger.info(f"\n{'='*60}")
        logger.info("CHAPTER 13 DAEMON STARTING")
        logger.info(f"{'='*60}")
        logger.info(f"Poll interval: {poll_interval}s")
        logger.info(f"LLM enabled: {self.use_llm}")
        logger.info(f"Press Ctrl+C to stop")
        logger.info(f"{'='*60}\n")
        
        self.running = True
        
        # Handle graceful shutdown
        def signal_handler(sig, frame):
            logger.info("\nShutdown signal received...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        while self.running:
            try:
                # Safety check
                if not self._check_safety():
                    logger.warning("Safety check failed, waiting...")
                    time.sleep(poll_interval)
                    continue
                
                # Check for new draw flag
                flag_data = self._check_new_draw_flag()
                
                if flag_data:
                    logger.info(f"🚨 New draw detected!")
                    
                    # Run cycle
                    result = self.run_cycle()
                    
                    # Clear flag after processing
                    self._clear_new_draw_flag()
                    
                    logger.info(f"Cycle complete: {result.get('outcome', 'unknown')}")
                
                # Sleep before next check
                time.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Daemon error: {e}")
                time.sleep(poll_interval)
        
        logger.info("Daemon stopped")
    
    # =========================================================================
    # STATUS & CONTROL
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        # Count recent cycles
        recent_cycles = []
        if os.path.exists(CYCLE_HISTORY):
            with open(CYCLE_HISTORY, 'r') as f:
                for line in f:
                    try:
                        recent_cycles.append(json.loads(line))
                    except:
                        pass
        
        # Outcome summary
        outcomes = {}
        for c in recent_cycles[-20:]:
            outcome = c.get("outcome", "unknown")
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        
        return {
            "running": self.running,
            "cycles_completed": self.cycles_completed,
            "last_cycle": self.last_cycle_time.isoformat() if self.last_cycle_time else None,
            "new_draw_flag_present": os.path.exists(NEW_DRAW_FLAG),
            "halt_active": os.path.exists(CHAPTER_13_HALT),
            "llm_enabled": self.use_llm,
            "llm_server_available": self._check_llm_server() if self.use_llm else None,
            "total_cycles_logged": len(recent_cycles),
            "recent_outcomes": outcomes,
            "pending_approval": self.trigger_manager.check_approval() is not None,
            "trigger_manager": self.trigger_manager.get_status(),
            "acceptance_engine": self.acceptance_engine.get_status()
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chapter 13 Orchestrator — Live feedback loop daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  --daemon           Run as daemon, watching for new draws
  --once             Run a single cycle (for testing)
  --status           Show orchestrator status
  --clear-halt       Clear Chapter 13 halt file
  --approve          Approve pending request
  --reject           Reject pending request

Examples:
  python3 chapter_13_orchestrator.py --daemon
  python3 chapter_13_orchestrator.py --once
  python3 chapter_13_orchestrator.py --status
  python3 chapter_13_orchestrator.py --approve
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--daemon", action="store_true",
                           help="Run as daemon")
    mode_group.add_argument("--once", action="store_true",
                           help="Run single cycle")
    mode_group.add_argument("--status", action="store_true",
                           help="Show status")
    mode_group.add_argument("--clear-halt", action="store_true",
                           help="Clear halt file")
    mode_group.add_argument("--approve", action="store_true",
                           help="Approve pending request")
    mode_group.add_argument("--reject", action="store_true",
                           help="Reject pending request")
    
    # Options
    parser.add_argument("--no-llm", action="store_true",
                       help="Disable LLM analysis")
    parser.add_argument("--auto-start-llm", action="store_true",
                       help="Auto-start LLM server if not running")
    parser.add_argument("--poll-interval", type=int, default=None,
                       help="Poll interval in seconds (default: from policies)")
    parser.add_argument("--policies", type=str, default=DEFAULT_POLICIES,
                       help=f"Policies file (default: {DEFAULT_POLICIES})")
    parser.add_argument("--reason", type=str, default="Manually rejected",
                       help="Rejection reason (for --reject)")
    parser.add_argument("--json", action="store_true",
                       help="Output as JSON")
    
    args = parser.parse_args()
    
    # Handle halt file clearing
    if args.clear_halt:
        if os.path.exists(CHAPTER_13_HALT):
            os.remove(CHAPTER_13_HALT)
            print("✅ Chapter 13 halt file cleared")
        else:
            print("No halt file present")
        return 0
    
    # Create orchestrator
    orchestrator = Chapter13Orchestrator(
        policies_path=args.policies,
        use_llm=not args.no_llm,
        auto_start_llm=args.auto_start_llm
    )
    
    try:
        if args.status:
            status = orchestrator.get_status()
            if args.json:
                print(json.dumps(status, indent=2))
            else:
                print(f"\n{'='*60}")
                print("CHAPTER 13 ORCHESTRATOR — Status")
                print(f"{'='*60}")
                print(f"\n🔄 Runtime:")
                print(f"   Running: {status['running']}")
                print(f"   Cycles completed: {status['cycles_completed']}")
                print(f"   Last cycle: {status['last_cycle'] or 'Never'}")
                print(f"\n📊 State:")
                print(f"   New draw flag: {status['new_draw_flag_present']}")
                print(f"   Halt active: {status['halt_active']}")
                print(f"   Pending approval: {status['pending_approval']}")
                print(f"\n🤖 LLM:")
                print(f"   Enabled: {status['llm_enabled']}")
                print(f"   Server available: {status['llm_server_available']}")
                print(f"\n📜 History:")
                print(f"   Total cycles: {status['total_cycles_logged']}")
                if status['recent_outcomes']:
                    print(f"   Recent outcomes:")
                    for outcome, count in status['recent_outcomes'].items():
                        print(f"      {outcome}: {count}")
                print()
            return 0
        
        if args.approve:
            success = orchestrator.trigger_manager.approve_request()
            return 0 if success else 1
        
        if args.reject:
            orchestrator.trigger_manager.reject_request(args.reason)
            return 0
        
        if args.once:
            result = orchestrator.run_cycle()
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(f"\n{'='*60}")
                print("CHAPTER 13 CYCLE RESULT")
                print(f"{'='*60}")
                print(f"Cycle ID: {result.get('cycle_id')}")
                print(f"Outcome: {result.get('outcome')}")
                print(f"Duration: {result.get('duration_seconds', 0):.2f}s")
                if result.get('steps'):
                    print(f"Steps executed: {list(result['steps'].keys())}")
                print()
            return 0
        
        if args.daemon:
            orchestrator.run_daemon(poll_interval=args.poll_interval)
            return 0
        
        # Default: show help
        parser.print_help()
        return 0
        
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 99


if __name__ == "__main__":
    sys.exit(main())
