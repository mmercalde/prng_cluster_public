#!/usr/bin/env python3
"""
Draw Ingestion Daemon — Chapter 13 Phase 1
Monitors for new draws and maintains lottery history

RESPONSIBILITIES:
1. Monitor for new draw sources (file drop, API, synthetic injection)
2. Validate and normalize incoming draws
3. Append to lottery_history.json (append-only, never overwrite)
4. Compute fingerprint for change detection
5. Signal WATCHER via new_draw.flag

VERSION: 1.0.0
DATE: 2026-01-12
DEPENDS ON: watcher_policies.json, lottery_history.json
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

# Watchdog is OPTIONAL - only needed for --watch-dir mode
# Core functionality (--daemon, --ingest) works without it
WATCHDOG_AVAILABLE = False
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    # Watchdog not installed - --watch-dir will be disabled
    Observer = None
    FileSystemEventHandler = object  # Placeholder for class inheritance
    FileCreatedEvent = None

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_LOTTERY_HISTORY = "lottery_history.json"
DEFAULT_POLICIES = "watcher_policies.json"
DEFAULT_WATCH_DIR = "incoming_draws/"
NEW_DRAW_FLAG = "new_draw.flag"
FINGERPRINT_FILE = ".history_fingerprint"

# Supported draw file formats
SUPPORTED_EXTENSIONS = {".json", ".txt", ".csv"}


# =============================================================================
# FINGERPRINTING
# =============================================================================

def compute_history_fingerprint(history: Dict[str, Any]) -> str:
    """
    Compute a deterministic fingerprint of lottery history.
    
    Used for change detection — Chapter 13 only triggers when
    fingerprint differs from previous run.
    
    Returns:
        8-character hex fingerprint
    """
    draws = history.get("draws", [])
    if not draws:
        return "00000000"
    
    # Create deterministic representation
    fingerprint_data = []
    for draw in draws:
        # Use draw value and timestamp (or ID) for uniqueness
        draw_str = json.dumps({
            "draw": draw.get("draw", draw.get("value", [])),
            "id": draw.get("draw_id", draw.get("timestamp", ""))
        }, sort_keys=True)
        fingerprint_data.append(draw_str)
    
    combined = "|".join(fingerprint_data)
    return hashlib.sha256(combined.encode()).hexdigest()[:8]


def load_previous_fingerprint() -> Optional[str]:
    """Load the last saved fingerprint."""
    if os.path.exists(FINGERPRINT_FILE):
        with open(FINGERPRINT_FILE, 'r') as f:
            return f.read().strip()
    return None


def save_fingerprint(fingerprint: str) -> None:
    """Save current fingerprint for future comparison."""
    with open(FINGERPRINT_FILE, 'w') as f:
        f.write(fingerprint)


def fingerprint_changed(history: Dict[str, Any]) -> bool:
    """
    Check if history fingerprint has changed since last check.
    
    Returns:
        True if fingerprint changed (new draw detected)
    """
    current = compute_history_fingerprint(history)
    previous = load_previous_fingerprint()
    
    if previous is None:
        print(f"📊 First fingerprint: {current}")
        save_fingerprint(current)
        return True
    
    if current != previous:
        print(f"📊 Fingerprint changed: {previous} → {current}")
        save_fingerprint(current)
        return True
    
    return False


# =============================================================================
# DRAW VALIDATION
# =============================================================================

class DrawValidationError(Exception):
    """Raised when draw validation fails."""
    pass


def validate_draw(draw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize an incoming draw.
    
    Ensures:
    - Required fields present
    - Draw values in valid range
    - Proper metadata attached
    
    Returns:
        Normalized draw record
    
    Raises:
        DrawValidationError: If validation fails
    """
    # Accept multiple input formats
    draw_values = None
    
    # Format 1: {"draw": [d1, d2, d3]}
    if "draw" in draw_data:
        draw_values = draw_data["draw"]
    # Format 2: {"value": 123}
    elif "value" in draw_data:
        value = draw_data["value"]
        # Convert to digits
        draw_values = [
            (value // 100) % 10,
            (value // 10) % 10,
            value % 10
        ]
    # Format 3: {"digits": [d1, d2, d3]}
    elif "digits" in draw_data:
        draw_values = draw_data["digits"]
    else:
        raise DrawValidationError(
            "Draw data must contain 'draw', 'value', or 'digits' field"
        )
    
    # Validate draw values
    if not isinstance(draw_values, list):
        raise DrawValidationError(f"Draw values must be a list, got {type(draw_values)}")
    
    if not all(isinstance(d, int) and 0 <= d <= 9 for d in draw_values):
        raise DrawValidationError(
            f"All draw digits must be integers 0-9, got {draw_values}"
        )
    
    # Determine draw source
    draw_source = draw_data.get("draw_source", "external")
    
    # Build normalized record
    timestamp = draw_data.get("timestamp", datetime.now(timezone.utc).isoformat())
    
    normalized = {
        "draw": draw_values,
        "raw_value": sum(d * (10 ** (len(draw_values) - 1 - i)) 
                        for i, d in enumerate(draw_values)),
        "timestamp": timestamp,
        "draw_source": draw_source,
        "ingested_at": datetime.now(timezone.utc).isoformat()
    }
    
    # Preserve existing metadata
    for key in ["draw_id", "true_seed", "prng_type", "position"]:
        if key in draw_data:
            normalized[key] = draw_data[key]
    
    # Generate draw_id if not present
    if "draw_id" not in normalized:
        ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        normalized["draw_id"] = f"{draw_source.upper()}-{ts.strftime('%Y-%m-%d')}-{ts.strftime('%H%M%S')}"
    
    return normalized


# =============================================================================
# HISTORY MANAGEMENT
# =============================================================================

def load_history(history_path: str) -> Dict[str, Any]:
    """Load lottery history from file."""
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            return json.load(f)
    return {"draws": [], "metadata": {}}


def save_history(history: Dict[str, Any], history_path: str) -> None:
    """Save lottery history to file."""
    # Update metadata
    history["metadata"] = history.get("metadata", {})
    history["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()
    history["metadata"]["total_draws"] = len(history.get("draws", []))
    
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)


def append_draw(
    draw_record: Dict[str, Any],
    history_path: str
) -> bool:
    """
    Append a validated draw to history.
    
    CRITICAL: Append-only. Never overwrites existing draws.
    
    Returns:
        True if draw was appended (new), False if duplicate
    """
    history = load_history(history_path)
    
    # Check for duplicates
    existing_ids = {d.get("draw_id") for d in history.get("draws", [])}
    if draw_record.get("draw_id") in existing_ids:
        print(f"⚠️  Duplicate draw ID: {draw_record.get('draw_id')} - skipping")
        return False
    
    # Append draw
    history.setdefault("draws", []).append(draw_record)
    
    # Save
    save_history(history, history_path)
    
    print(f"✅ Appended draw: {draw_record['draw_id']} = {draw_record['draw']}")
    return True


# =============================================================================
# WATCHER SIGNALING
# =============================================================================

def create_new_draw_flag(draw_record: Dict[str, Any]) -> None:
    """
    Create new_draw.flag to trigger Chapter 13 processing.
    
    The flag contains metadata about the new draw for WATCHER.
    """
    flag_data = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "draw_ingestion_daemon",
        "draw_id": draw_record.get("draw_id"),
        "draw_source": draw_record.get("draw_source"),
        "trigger": "chapter_13"
    }
    
    with open(NEW_DRAW_FLAG, 'w') as f:
        json.dump(flag_data, f, indent=2)
    
    print(f"🚩 Created {NEW_DRAW_FLAG}")


def check_and_clear_flag() -> Optional[Dict[str, Any]]:
    """
    Check if new_draw.flag exists and read its contents.
    
    Returns:
        Flag data if present, None otherwise
    """
    if os.path.exists(NEW_DRAW_FLAG):
        with open(NEW_DRAW_FLAG, 'r') as f:
            data = json.load(f)
        return data
    return None


def clear_flag() -> None:
    """Remove new_draw.flag after processing."""
    if os.path.exists(NEW_DRAW_FLAG):
        os.remove(NEW_DRAW_FLAG)
        print(f"🗑️  Cleared {NEW_DRAW_FLAG}")


# =============================================================================
# FILE MONITORING
# =============================================================================

class DrawFileHandler(FileSystemEventHandler):
    """
    Watchdog handler for incoming draw files.
    
    Monitors a directory for new draw files and processes them.
    """
    
    def __init__(self, history_path: str, create_flag: bool = True):
        self.history_path = history_path
        self.create_flag = create_flag
        super().__init__()
    
    def on_created(self, event):
        """Handle new file creation."""
        if isinstance(event, FileCreatedEvent):
            self.process_file(event.src_path)
    
    def process_file(self, file_path: str) -> bool:
        """
        Process an incoming draw file.
        
        Returns:
            True if draw was successfully ingested
        """
        path = Path(file_path)
        
        # Check extension
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"⚠️  Ignoring unsupported file: {file_path}")
            return False
        
        print(f"\n📥 Processing: {file_path}")
        
        try:
            # Parse based on format
            if path.suffix.lower() == ".json":
                with open(file_path, 'r') as f:
                    draw_data = json.load(f)
            elif path.suffix.lower() == ".txt":
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                # Expect format: "d1 d2 d3" or "d1,d2,d3" or "123"
                if ',' in content:
                    parts = content.split(',')
                elif ' ' in content:
                    parts = content.split()
                else:
                    # Single number
                    draw_data = {"value": int(content)}
                    parts = None
                
                if parts:
                    draw_data = {"draw": [int(p.strip()) for p in parts]}
            elif path.suffix.lower() == ".csv":
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                # Take first non-header line
                for line in lines:
                    if line.strip() and not line.startswith('#'):
                        parts = line.strip().split(',')
                        draw_data = {"draw": [int(p.strip()) for p in parts[:3]]}
                        break
            else:
                print(f"❌ Unsupported format: {path.suffix}")
                return False
            
            # Validate and normalize
            normalized = validate_draw(draw_data)
            
            # Append to history
            if append_draw(normalized, self.history_path):
                # Signal WATCHER
                if self.create_flag:
                    create_new_draw_flag(normalized)
                
                # Archive processed file
                archive_dir = Path("processed_draws")
                archive_dir.mkdir(exist_ok=True)
                archive_path = archive_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{path.name}"
                os.rename(file_path, archive_path)
                print(f"📦 Archived to: {archive_path}")
                
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            # Move to error directory
            error_dir = Path("error_draws")
            error_dir.mkdir(exist_ok=True)
            error_path = error_dir / path.name
            os.rename(file_path, error_path)
            print(f"📦 Moved to error: {error_path}")
            return False


# =============================================================================
# DAEMON MODES
# =============================================================================

def ingest_one(
    draw_file: str,
    history_path: str,
    create_flag: bool = True
) -> bool:
    """
    Ingest a single draw file.
    
    Mode: Manual ingestion with --ingest FILE
    """
    print("\n" + "=" * 60)
    print("DRAW INGESTION — Single File Mode")
    print("=" * 60)
    
    handler = DrawFileHandler(history_path, create_flag)
    return handler.process_file(draw_file)


def run_directory_watch(
    watch_dir: str,
    history_path: str,
    create_flag: bool = True
) -> None:
    """
    Watch a directory for new draw files.
    
    Mode: Daemon with --watch-dir DIR
    
    REQUIRES: watchdog package (optional dependency)
    """
    if not WATCHDOG_AVAILABLE:
        print("\n❌ --watch-dir requires the 'watchdog' package.")
        print("   This is an OPTIONAL feature. Core functionality works without it.")
        print("\n   Alternatives:")
        print("   1. Use --daemon mode (watches for new_draw.flag)")
        print("   2. Use --ingest FILE mode (process single files)")
        print("   3. Install watchdog in a virtual environment:")
        print("      python3 -m venv ~/venv_watchdog")
        print("      source ~/venv_watchdog/bin/activate")
        print("      pip install watchdog")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("DRAW INGESTION — Directory Watch Mode")
    print("=" * 60)
    print(f"Watching: {watch_dir}")
    print(f"History: {history_path}")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    # Create watch directory if needed
    Path(watch_dir).mkdir(exist_ok=True)
    
    # Process any existing files first
    handler = DrawFileHandler(history_path, create_flag)
    for file_path in Path(watch_dir).glob("*"):
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            handler.process_file(str(file_path))
    
    # Set up watchdog observer
    observer = Observer()
    observer.schedule(handler, watch_dir, recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Daemon stopped by user")
        observer.stop()
    
    observer.join()


def run_flag_watch(
    history_path: str,
    poll_interval: int = 30
) -> None:
    """
    Watch for new_draw.flag and process when detected.
    
    Mode: Daemon with --daemon (works with synthetic_draw_injector)
    """
    print("\n" + "=" * 60)
    print("DRAW INGESTION — Flag Watch Mode")
    print("=" * 60)
    print(f"Polling interval: {poll_interval}s")
    print(f"History: {history_path}")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    try:
        while True:
            # Check for flag
            flag_data = check_and_clear_flag()
            
            if flag_data:
                print(f"\n🚩 Flag detected: {flag_data.get('draw_id', 'unknown')}")
                
                # Load and check history fingerprint
                history = load_history(history_path)
                
                if fingerprint_changed(history):
                    print("✅ New draw confirmed via fingerprint")
                    # Chapter 13 processing would be triggered here
                    # (WATCHER handles the actual processing)
                else:
                    print("⚠️  Flag present but no fingerprint change")
            
            time.sleep(poll_interval)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Daemon stopped by user")


def show_status(history_path: str) -> None:
    """Show current ingestion status."""
    print("\n" + "=" * 60)
    print("DRAW INGESTION — Status")
    print("=" * 60)
    
    # History status
    history = load_history(history_path)
    draws = history.get("draws", [])
    
    print(f"\n📜 History ({history_path}):")
    print(f"   Total Draws: {len(draws)}")
    
    if draws:
        # Count by source
        sources = {}
        for d in draws:
            src = d.get("draw_source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        
        print(f"   By Source:")
        for src, count in sources.items():
            print(f"      {src}: {count}")
        
        # Show last few draws
        print(f"\n   Last 5 Draws:")
        for d in draws[-5:]:
            print(f"      {d.get('draw_id', 'N/A')}: {d.get('draw')} ({d.get('draw_source', 'N/A')})")
    
    # Fingerprint status
    current_fp = compute_history_fingerprint(history)
    previous_fp = load_previous_fingerprint()
    
    print(f"\n🔒 Fingerprint:")
    print(f"   Current: {current_fp}")
    print(f"   Previous: {previous_fp or 'None'}")
    print(f"   Changed: {current_fp != previous_fp if previous_fp else 'N/A'}")
    
    # Flag status
    flag_data = check_and_clear_flag()
    if flag_data:
        print(f"\n🚩 Flag Present:")
        print(f"   Created: {flag_data.get('created_at')}")
        print(f"   Source: {flag_data.get('source')}")
        print(f"   Draw ID: {flag_data.get('draw_id')}")
    else:
        print(f"\n🚩 No flag present")
    
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Draw Ingestion Daemon — Chapter 13",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  --ingest FILE      Ingest a single draw file
  --watch-dir DIR    Watch directory for new draw files (requires watchdog)
  --daemon           Watch for new_draw.flag (works with synthetic_draw_injector)
  --status           Show current ingestion status

Note: --watch-dir requires the 'watchdog' package (optional).
      Core functionality (--daemon, --ingest) works without any extra packages.

Examples:
  python3 draw_ingestion_daemon.py --ingest new_draw.json
  python3 draw_ingestion_daemon.py --daemon
  python3 draw_ingestion_daemon.py --status
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--ingest", type=str, metavar="FILE",
                           help="Ingest a single draw file")
    mode_group.add_argument("--watch-dir", type=str, metavar="DIR",
                           help="Watch directory for new files (requires watchdog package)")
    mode_group.add_argument("--daemon", action="store_true",
                           help="Watch for new_draw.flag")
    mode_group.add_argument("--status", action="store_true",
                           help="Show current status")
    mode_group.add_argument("--check-fingerprint", action="store_true",
                           help="Check if history fingerprint changed")
    
    # Options
    parser.add_argument("--history", type=str, default=DEFAULT_LOTTERY_HISTORY,
                       help=f"Lottery history file (default: {DEFAULT_LOTTERY_HISTORY})")
    parser.add_argument("--poll-interval", type=int, default=30,
                       help="Poll interval in seconds for daemon mode (default: 30)")
    parser.add_argument("--no-flag", action="store_true",
                       help="Don't create new_draw.flag after ingestion (default: flag IS created)")
    
    args = parser.parse_args()
    
    try:
        if args.status:
            show_status(args.history)
            return 0
        
        if args.check_fingerprint:
            history = load_history(args.history)
            if fingerprint_changed(history):
                print("✅ Fingerprint CHANGED - new draw detected")
                return 0
            else:
                print("➖ Fingerprint unchanged")
                return 1
        
        if args.ingest:
            success = ingest_one(args.ingest, args.history, 
                                create_flag=not args.no_flag)
            return 0 if success else 1
        
        if args.watch_dir:
            run_directory_watch(args.watch_dir, args.history,
                              create_flag=not args.no_flag)
            return 0
        
        if args.daemon:
            run_flag_watch(args.history, args.poll_interval)
            return 0
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 99


if __name__ == "__main__":
    sys.exit(main())
