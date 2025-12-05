"""
Safety Module - Kill switch and safety controls.

Exports:
    KillSwitch - Safety override controls
    SafetyCheck - Individual safety check result
    SafetyLevel - Safety level enum
    check_safety - Quick safety check
    create_halt - Create halt file
    clear_halt - Clear halt file
"""

from .kill_switch import (
    KillSwitch,
    SafetyCheck,
    SafetyLevel,
    check_safety,
    create_halt,
    clear_halt
)
