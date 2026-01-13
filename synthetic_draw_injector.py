#!/usr/bin/env python3
"""
Synthetic Draw Injector — Chapter 13 Phase 1
Test mode draw generation for learning loop validation

CRITICAL DESIGN PRINCIPLES:
- No hardcoded PRNG type — reads from optimal_window_config.json
- Uses prng_registry.py (same as all pipeline steps)
- Dual safety flags: test_mode AND synthetic_injection.enabled
- All synthetic draws tagged with "draw_source": "synthetic"

VERSION: 1.0.0
DATE: 2026-01-12
DEPENDS ON: prng_registry.py, optimal_window_config.json, watcher_policies.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from prng_registry import get_cpu_reference, list_available_prngs, get_kernel_info

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_LOTTERY_HISTORY = "lottery_history.json"
DEFAULT_OPTIMAL_CONFIG = "optimal_window_config.json"
DEFAULT_POLICIES = "watcher_policies.json"
NEW_DRAW_FLAG = "new_draw.flag"


# =============================================================================
# SAFETY CHECKS
# =============================================================================

class InjectionSafetyError(Exception):
    """Raised when safety conditions for injection are not met."""
    pass


def validate_injection_enabled(policies: Dict[str, Any]) -> None:
    """
    Validate that BOTH test_mode AND synthetic_injection.enabled are true.
    
    This is the PRIMARY safety gate. Injection cannot occur without
    explicit dual-flag consent.
    
    Raises:
        InjectionSafetyError: If safety conditions not met
    """
    test_mode = policies.get("test_mode", False)
    injection_config = policies.get("synthetic_injection", {})
    injection_enabled = injection_config.get("enabled", False)
    
    if not test_mode:
        raise InjectionSafetyError(
            "test_mode is False in watcher_policies.json. "
            "Synthetic injection requires test_mode: true"
        )
    
    if not injection_enabled:
        raise InjectionSafetyError(
            "synthetic_injection.enabled is False in watcher_policies.json. "
            "Synthetic injection requires synthetic_injection.enabled: true"
        )
    
    print("✅ Safety check passed: test_mode=true, synthetic_injection.enabled=true")


# =============================================================================
# PRNG CONFIGURATION LOADING
# =============================================================================

def load_prng_config(config_path: str) -> Dict[str, Any]:
    """
    Load PRNG configuration from optimal_window_config.json.
    
    CRITICAL: PRNG type is NEVER hardcoded. It must come from config.
    
    Returns:
        Dict containing prng_type and related parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"optimal_window_config.json not found at {config_path}. "
            "Run Step 1 (Window Optimizer) first."
        )
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    prng_type = config.get("prng_type")
    if not prng_type:
        raise ValueError(
            "prng_type not found in optimal_window_config.json. "
            "Config must specify the PRNG family."
        )
    
    # Validate PRNG exists in registry
    available = list_available_prngs()
    if prng_type not in available:
        raise ValueError(
            f"PRNG '{prng_type}' not in registry. "
            f"Available PRNGs: {available}"
        )
    
    print(f"✅ PRNG type loaded from config: {prng_type}")
    return config


def load_policies(policies_path: str) -> Dict[str, Any]:
    """Load watcher_policies.json."""
    if not os.path.exists(policies_path):
        raise FileNotFoundError(
            f"watcher_policies.json not found at {policies_path}. "
            "Create policies file first."
        )
    
    with open(policies_path, 'r') as f:
        return json.load(f)


# =============================================================================
# DRAW STATE MANAGEMENT
# =============================================================================

class SyntheticDrawState:
    """
    Manages the state of synthetic draw generation.
    
    Tracks:
    - Current position in PRNG sequence
    - True seed for validation
    - Draw history for continuity
    """
    
    STATE_FILE = ".synthetic_draw_state.json"
    
    def __init__(self, true_seed: int, prng_type: str, mod_value: int = 1000):
        self.true_seed = true_seed
        self.prng_type = prng_type
        self.mod_value = mod_value
        self.current_position = 0
        self.draws_generated = 0
        
    @classmethod
    def load_or_create(cls, policies: Dict[str, Any], prng_config: Dict[str, Any]) -> 'SyntheticDrawState':
        """Load existing state or create new one."""
        injection_config = policies.get("synthetic_injection", {})
        true_seed = injection_config.get("true_seed", 12345)
        prng_type = prng_config.get("prng_type", "java_lcg")
        mod_value = prng_config.get("mod_value", 1000)
        
        state = cls(true_seed, prng_type, mod_value)
        
        if os.path.exists(cls.STATE_FILE):
            with open(cls.STATE_FILE, 'r') as f:
                saved = json.load(f)
                # Only restore if same seed and PRNG
                if (saved.get("true_seed") == true_seed and 
                    saved.get("prng_type") == prng_type):
                    state.current_position = saved.get("current_position", 0)
                    state.draws_generated = saved.get("draws_generated", 0)
                    print(f"📂 Restored state: position={state.current_position}, draws={state.draws_generated}")
                else:
                    print("🔄 State reset: seed or PRNG changed")
        
        return state
    
    def save(self) -> None:
        """Persist state to disk."""
        with open(self.STATE_FILE, 'w') as f:
            json.dump({
                "true_seed": self.true_seed,
                "prng_type": self.prng_type,
                "mod_value": self.mod_value,
                "current_position": self.current_position,
                "draws_generated": self.draws_generated,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }, f, indent=2)
    
    def reset(self) -> None:
        """Reset state to beginning."""
        self.current_position = 0
        self.draws_generated = 0
        if os.path.exists(self.STATE_FILE):
            os.remove(self.STATE_FILE)
        print("🔄 State reset to position 0")


# =============================================================================
# DRAW GENERATION
# =============================================================================

def generate_synthetic_draw(
    state: SyntheticDrawState,
    prng_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a single synthetic draw using the configured PRNG.
    
    CRITICAL: Uses prng_registry.py CPU reference (same as pipeline).
    
    Args:
        state: Current synthetic draw state
        prng_config: PRNG configuration from optimal_window_config.json
    
    Returns:
        Draw record with metadata and synthetic tag
    """
    prng_type = state.prng_type
    true_seed = state.true_seed
    mod_value = state.mod_value
    position = state.current_position
    
    # Get CPU reference from registry (same as all pipeline steps)
    prng_func = get_cpu_reference(prng_type)
    
    # Get PRNG-specific parameters from config
    prng_params = prng_config.get("prng_params", {})
    
    # Generate output at current position
    # Skip to position, then generate 1 value
    outputs = prng_func(true_seed, n=position + 1, skip=0, **prng_params)
    raw_value = outputs[-1]  # Get the value at current position
    
    # Apply modulo to get draw value
    draw_value = raw_value % mod_value
    
    # Format draw (Daily 3 format: 3 digits)
    # Different formats for different lottery types
    draw_digits = format_draw_value(draw_value, mod_value)
    
    # Create draw record with required metadata
    timestamp = datetime.now(timezone.utc)
    draw_record = {
        "draw_id": f"SYNTHETIC-{timestamp.strftime('%Y-%m-%d')}-{state.draws_generated + 1:03d}",
        "timestamp": timestamp.isoformat(),
        "draw": draw_digits,
        "raw_value": draw_value,
        
        # REQUIRED: Synthetic tagging for diagnostics
        "draw_source": "synthetic",
        "true_seed": true_seed,
        "prng_type": prng_type,
        "position": position,
        "generated_at": timestamp.isoformat(),
        
        # Audit fields
        "mod_value": mod_value,
        "prng_params": prng_params
    }
    
    # Update state
    state.current_position += 1
    state.draws_generated += 1
    state.save()
    
    return draw_record


def format_draw_value(value: int, mod_value: int) -> List[int]:
    """
    Format raw draw value into digit list.
    
    Examples:
        mod=1000, value=42  -> [0, 4, 2]
        mod=1000, value=978 -> [9, 7, 8]
        mod=100,  value=42  -> [4, 2]
    """
    if mod_value == 1000:
        # Daily 3 format: 3 digits
        return [
            (value // 100) % 10,
            (value // 10) % 10,
            value % 10
        ]
    elif mod_value == 100:
        # Daily 2 format: 2 digits
        return [
            (value // 10) % 10,
            value % 10
        ]
    elif mod_value == 10000:
        # Daily 4 format: 4 digits
        return [
            (value // 1000) % 10,
            (value // 100) % 10,
            (value // 10) % 10,
            value % 10
        ]
    else:
        # Generic: return as single value
        return [value]


# =============================================================================
# HISTORY MANAGEMENT
# =============================================================================

def append_draw_to_history(
    draw_record: Dict[str, Any],
    history_path: str
) -> None:
    """
    Append synthetic draw to lottery history (append-only).
    
    CRITICAL: Never overwrites existing draws.
    """
    # Load existing history
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
    else:
        history = {"draws": [], "metadata": {"source": "synthetic_test"}}
    
    # Ensure draws list exists
    if "draws" not in history:
        history["draws"] = []
    
    # Append new draw
    history["draws"].append(draw_record)
    
    # Update metadata
    history["metadata"] = history.get("metadata", {})
    history["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()
    history["metadata"]["total_draws"] = len(history["draws"])
    history["metadata"]["synthetic_count"] = sum(
        1 for d in history["draws"] if d.get("draw_source") == "synthetic"
    )
    
    # Write back
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"📝 Appended draw to {history_path} (total: {len(history['draws'])})")


def create_new_draw_flag(flag_path: str) -> None:
    """
    Create new_draw.flag to signal WATCHER that a new draw is available.
    
    The flag contains metadata for the WATCHER to process.
    """
    flag_data = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "synthetic_draw_injector",
        "trigger": "chapter_13"
    }
    
    with open(flag_path, 'w') as f:
        json.dump(flag_data, f, indent=2)
    
    print(f"🚩 Created {flag_path}")


# =============================================================================
# INJECTION MODES
# =============================================================================

def inject_one(
    prng_config: Dict[str, Any],
    policies: Dict[str, Any],
    history_path: str,
    create_flag: bool = True
) -> Dict[str, Any]:
    """
    Inject a single synthetic draw.
    
    Mode: Manual testing with --inject-one
    """
    print("\n" + "=" * 60)
    print("SYNTHETIC DRAW INJECTION — Single Draw Mode")
    print("=" * 60)
    
    # Load or create state
    state = SyntheticDrawState.load_or_create(policies, prng_config)
    
    print(f"\n📊 Configuration:")
    print(f"   PRNG Type: {state.prng_type}")
    print(f"   True Seed: {state.true_seed}")
    print(f"   Mod Value: {state.mod_value}")
    print(f"   Current Position: {state.current_position}")
    
    # Generate draw
    draw_record = generate_synthetic_draw(state, prng_config)
    
    print(f"\n✨ Generated Draw:")
    print(f"   Draw ID: {draw_record['draw_id']}")
    print(f"   Value: {draw_record['draw']} (raw: {draw_record['raw_value']})")
    print(f"   Position: {draw_record['position']}")
    
    # Append to history
    append_draw_to_history(draw_record, history_path)
    
    # Create flag for WATCHER
    if create_flag:
        create_new_draw_flag(NEW_DRAW_FLAG)
    
    print("\n✅ Injection complete")
    return draw_record


def run_daemon(
    prng_config: Dict[str, Any],
    policies: Dict[str, Any],
    history_path: str,
    interval_seconds: int
) -> None:
    """
    Run continuous synthetic draw injection.
    
    Mode: Automated testing with --daemon --interval N
    """
    print("\n" + "=" * 60)
    print("SYNTHETIC DRAW INJECTION — Daemon Mode")
    print("=" * 60)
    print(f"Interval: {interval_seconds} seconds")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    try:
        iteration = 0
        while True:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            
            # Re-validate safety on each iteration
            validate_injection_enabled(policies)
            
            # Inject one draw
            inject_one(prng_config, policies, history_path, create_flag=True)
            
            # Wait for next interval
            print(f"\n⏳ Waiting {interval_seconds}s until next injection...")
            time.sleep(interval_seconds)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Daemon stopped by user")
    except InjectionSafetyError as e:
        print(f"\n\n🛑 Daemon stopped: {e}")
        raise


def show_status(
    prng_config: Dict[str, Any],
    policies: Dict[str, Any],
    history_path: str
) -> None:
    """Show current synthetic injection status."""
    print("\n" + "=" * 60)
    print("SYNTHETIC DRAW INJECTION — Status")
    print("=" * 60)
    
    # Safety status
    test_mode = policies.get("test_mode", False)
    injection_config = policies.get("synthetic_injection", {})
    injection_enabled = injection_config.get("enabled", False)
    
    print(f"\n🔒 Safety Flags:")
    print(f"   test_mode: {test_mode}")
    print(f"   synthetic_injection.enabled: {injection_enabled}")
    print(f"   Injection allowed: {test_mode and injection_enabled}")
    
    # PRNG config
    print(f"\n⚙️  PRNG Configuration:")
    print(f"   Type: {prng_config.get('prng_type', 'N/A')}")
    print(f"   Mod Value: {prng_config.get('mod_value', 1000)}")
    
    # State
    state = SyntheticDrawState.load_or_create(policies, prng_config)
    print(f"\n📊 State:")
    print(f"   True Seed: {state.true_seed}")
    print(f"   Current Position: {state.current_position}")
    print(f"   Draws Generated: {state.draws_generated}")
    
    # History
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        total = len(history.get("draws", []))
        synthetic = sum(1 for d in history.get("draws", []) 
                       if d.get("draw_source") == "synthetic")
        print(f"\n📜 History ({history_path}):")
        print(f"   Total Draws: {total}")
        print(f"   Synthetic Draws: {synthetic}")
    else:
        print(f"\n📜 History: {history_path} not found")
    
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Synthetic Draw Injector — Chapter 13 Test Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  --inject-one     Inject a single synthetic draw (manual testing)
  --daemon         Run continuous injection (automated testing)
  --status         Show current state and configuration
  --reset          Reset state to position 0

Safety:
  Injection requires BOTH flags in watcher_policies.json:
    - test_mode: true
    - synthetic_injection.enabled: true
  
  PRNG type is read from optimal_window_config.json (never hardcoded).

Examples:
  python3 synthetic_draw_injector.py --inject-one
  python3 synthetic_draw_injector.py --daemon --interval 60
  python3 synthetic_draw_injector.py --status
  python3 synthetic_draw_injector.py --reset
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--inject-one", action="store_true",
                           help="Inject a single synthetic draw")
    mode_group.add_argument("--daemon", action="store_true",
                           help="Run continuous injection daemon")
    mode_group.add_argument("--status", action="store_true",
                           help="Show current status")
    mode_group.add_argument("--reset", action="store_true",
                           help="Reset state to position 0")
    
    # Options
    parser.add_argument("--interval", type=int, default=60,
                       help="Injection interval in seconds for daemon mode (default: 60, or from policies)")
    parser.add_argument("--history", type=str, default=DEFAULT_LOTTERY_HISTORY,
                       help=f"Lottery history file (default: {DEFAULT_LOTTERY_HISTORY})")
    parser.add_argument("--config", type=str, default=DEFAULT_OPTIMAL_CONFIG,
                       help=f"PRNG config file (default: {DEFAULT_OPTIMAL_CONFIG})")
    parser.add_argument("--policies", type=str, default=DEFAULT_POLICIES,
                       help=f"Policies file (default: {DEFAULT_POLICIES})")
    parser.add_argument("--no-flag", action="store_true",
                       help="Don't create new_draw.flag after injection (default: flag IS created)")
    
    args = parser.parse_args()
    
    try:
        # Load configurations
        prng_config = load_prng_config(args.config)
        policies = load_policies(args.policies)
        
        # Handle reset (doesn't require safety check)
        if args.reset:
            state = SyntheticDrawState.load_or_create(policies, prng_config)
            state.reset()
            print("✅ State reset complete")
            return 0
        
        # Handle status (doesn't require safety check)
        if args.status:
            show_status(prng_config, policies, args.history)
            return 0
        
        # Safety check for injection modes
        validate_injection_enabled(policies)
        
        # Execute requested mode
        if args.inject_one:
            inject_one(prng_config, policies, args.history, 
                      create_flag=not args.no_flag)
        
        elif args.daemon:
            interval = policies.get("synthetic_injection", {}).get(
                "interval_seconds", args.interval
            )
            run_daemon(prng_config, policies, args.history, interval)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        return 1
    except InjectionSafetyError as e:
        print(f"\n❌ Safety check failed: {e}")
        return 2
    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        return 3
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 99


if __name__ == "__main__":
    sys.exit(main())
