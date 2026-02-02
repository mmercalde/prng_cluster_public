#!/usr/bin/env python3
"""
GBNF Grammar Loader for LLM Constrained Decoding.

Version: 1.1.0 (was 1.0.0)
Date: 2026-02-01 (Session 57)
Chapter: 10 §7

Changes from v1.0.0:
    - Fixed GRAMMAR_DIR to use os.path resolution instead of hardcoded relative path
      (resolves to distributed_prng_analysis/agent_grammars/ regardless of CWD)
    - Added explicit GRAMMAR_FILES mapping from GrammarType enum to filenames
    - Added get_grammar_content() for direct grammar text loading
    - Added list_available_grammars() for diagnostics

Note: chapter_13.gbnf is NOT mapped through GrammarType — it's loaded directly
by the Chapter 13 LLM advisor via filename. No change needed there.
"""

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class GrammarType(str, Enum):
    """Available grammar types for constrained decoding.

    Each type maps to a .gbnf file in agent_grammars/.
    """

    AGENT_DECISION = "agent_decision"
    SIEVE_ANALYSIS = "sieve_analysis"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    JSON_GENERIC = "json_generic"


# ── Path Resolution (v1.1.0 fix) ────────────────────────────────────
# Resolves to distributed_prng_analysis/agent_grammars/ regardless of CWD.
# This file lives at llm_services/grammar_loader.py, so parent.parent
# gives the project root.
GRAMMAR_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "agent_grammars"
)

# ── Grammar File Mapping (v1.1.0 addition) ──────────────────────────
GRAMMAR_FILES: Dict[GrammarType, str] = {
    GrammarType.AGENT_DECISION: "agent_decision.gbnf",
    GrammarType.SIEVE_ANALYSIS: "sieve_analysis.gbnf",
    GrammarType.PARAMETER_ADJUSTMENT: "parameter_adjustment.gbnf",
    GrammarType.JSON_GENERIC: "json_generic.gbnf",
}


def get_grammar_path(grammar_type: GrammarType) -> Optional[str]:
    """Resolve filesystem path for a grammar type.

    Args:
        grammar_type: The grammar type to resolve.

    Returns:
        Absolute path to the .gbnf file, or None if not found.
    """
    filename = GRAMMAR_FILES.get(grammar_type)
    if filename is None:
        logger.error("Unknown grammar type: %s", grammar_type)
        return None

    path = os.path.join(GRAMMAR_DIR, filename)
    abs_path = os.path.abspath(path)

    if not os.path.isfile(abs_path):
        logger.warning(
            "Grammar file not found: %s (type=%s)", abs_path, grammar_type.value
        )
        return None

    logger.debug("Resolved grammar %s → %s", grammar_type.value, abs_path)
    return abs_path


def get_grammar_content(grammar_type: GrammarType) -> Optional[str]:
    """Load grammar file content as a string.

    Args:
        grammar_type: The grammar type to load.

    Returns:
        Grammar content string, or None if file not found.
    """
    path = get_grammar_path(grammar_type)
    if path is None:
        return None

    try:
        with open(path, "r") as f:
            content = f.read()
        logger.debug("Loaded grammar %s (%d bytes)", grammar_type.value, len(content))
        return content
    except (IOError, OSError) as e:
        logger.error("Failed to read grammar file %s: %s", path, e)
        return None


def get_grammar_for_step(step_number: int) -> Optional[str]:
    """Get the appropriate grammar path for a pipeline step.

    Mapping:
        Step 1: agent_decision (Window Optimizer evaluation)
        Step 2: sieve_analysis (Bidirectional Sieve evaluation)
        Step 3: agent_decision (Full Scoring evaluation)
        Step 4: agent_decision (ML Meta evaluation)
        Step 5: agent_decision (Anti-Overfit Training evaluation)
        Step 6: agent_decision (Prediction Generation evaluation)

    Args:
        step_number: Pipeline step (1-6).

    Returns:
        Path to grammar file, or None if unavailable.
    """
    if step_number == 2:
        grammar_type = GrammarType.SIEVE_ANALYSIS
    elif 1 <= step_number <= 6:
        grammar_type = GrammarType.AGENT_DECISION
    else:
        logger.warning("No grammar mapping for step %d", step_number)
        return None

    return get_grammar_path(grammar_type)


def get_chapter13_grammar_path() -> Optional[str]:
    """Get path to chapter_13.gbnf (loaded directly, not via GrammarType enum).

    Returns:
        Absolute path to chapter_13.gbnf, or None if not found.
    """
    path = os.path.join(GRAMMAR_DIR, "chapter_13.gbnf")
    abs_path = os.path.abspath(path)

    if not os.path.isfile(abs_path):
        logger.warning("chapter_13.gbnf not found at %s", abs_path)
        return None

    return abs_path


def list_available_grammars() -> Dict[str, bool]:
    """List all expected grammar files and their availability.

    Returns:
        Dict mapping grammar name to exists (True/False).
    """
    result = {}

    # GrammarType-mapped grammars
    for gtype, filename in GRAMMAR_FILES.items():
        path = os.path.join(GRAMMAR_DIR, filename)
        result[f"{gtype.value} ({filename})"] = os.path.isfile(path)

    # Chapter 13 grammar (not in enum)
    ch13_path = os.path.join(GRAMMAR_DIR, "chapter_13.gbnf")
    result["chapter_13 (chapter_13.gbnf)"] = os.path.isfile(ch13_path)

    return result


# ------------------------------------------------------------------
# Self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Grammar Loader v1.1.0 — Self-Test")
    print("=" * 60)

    print(f"\nGrammar directory: {os.path.abspath(GRAMMAR_DIR)}")

    print("\nAvailable grammars:")
    for name, exists in list_available_grammars().items():
        status = "✅" if exists else "❌ MISSING"
        print(f"  {status}  {name}")

    print("\nStep-to-grammar mapping:")
    for step in range(1, 7):
        path = get_grammar_for_step(step)
        if path:
            print(f"  Step {step} → {os.path.basename(path)}")
        else:
            print(f"  Step {step} → ❌ No grammar found")

    # Test content loading
    print("\nContent loading test:")
    for gtype in GrammarType:
        content = get_grammar_content(gtype)
        if content:
            lines = len(content.strip().split("\n"))
            print(f"  ✅ {gtype.value}: {lines} lines loaded")
        else:
            print(f"  ❌ {gtype.value}: FAILED to load")

    print("\n" + "=" * 60)
    print("Self-test complete")
    print("=" * 60)
