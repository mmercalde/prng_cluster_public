#!/usr/bin/env python3
"""
Grammar Loader - GBNF Grammar Management for LLM Constrained Decoding
File: llm_services/grammar_loader.py
Version: 1.0.0
Date: December 6, 2025

Purpose:
    Loads and manages GBNF grammar files for constraining LLM output.
    Provides grammar selection based on request type.

Usage:
    from llm_services.grammar_loader import GrammarLoader
    
    loader = GrammarLoader()
    grammar = loader.get_grammar("agent_decision")
    
    # In LLM call
    response = llm.complete(prompt, grammar=grammar)

Grammar Files:
    - agent_decision.gbnf     : Agent evaluation responses (proceed/retry/escalate)
    - sieve_analysis.gbnf     : Sieve result interpretation
    - parameter_adjustment.gbnf: Parameter change suggestions
    - json_generic.gbnf       : Fallback for any valid JSON
"""

import os
from pathlib import Path
from typing import Optional, Dict
from enum import Enum


class GrammarType(str, Enum):
    """Available grammar types for constrained decoding."""
    AGENT_DECISION = "agent_decision"
    SIEVE_ANALYSIS = "sieve_analysis"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    JSON_GENERIC = "json_generic"


# Map request patterns to grammar types
REQUEST_GRAMMAR_MAP = {
    # Agent decision patterns
    "evaluate": GrammarType.AGENT_DECISION,
    "decision": GrammarType.AGENT_DECISION,
    "success_condition": GrammarType.AGENT_DECISION,
    "proceed": GrammarType.AGENT_DECISION,
    "retry": GrammarType.AGENT_DECISION,
    
    # Sieve analysis patterns
    "sieve": GrammarType.SIEVE_ANALYSIS,
    "survivors": GrammarType.SIEVE_ANALYSIS,
    "forward": GrammarType.SIEVE_ANALYSIS,
    "reverse": GrammarType.SIEVE_ANALYSIS,
    "bidirectional": GrammarType.SIEVE_ANALYSIS,
    
    # Parameter adjustment patterns
    "adjust": GrammarType.PARAMETER_ADJUSTMENT,
    "parameter": GrammarType.PARAMETER_ADJUSTMENT,
    "window_size": GrammarType.PARAMETER_ADJUSTMENT,
    "threshold": GrammarType.PARAMETER_ADJUSTMENT,
    "suggest": GrammarType.PARAMETER_ADJUSTMENT,
}


class GrammarLoader:
    """
    Loads and manages GBNF grammar files for constrained LLM output.
    
    Attributes:
        grammar_dir: Directory containing .gbnf files
        cache: In-memory cache of loaded grammars
    """
    
    def __init__(self, grammar_dir: Optional[str] = None):
        """
        Initialize grammar loader.
        
        Args:
            grammar_dir: Path to grammar files. Defaults to ./grammars/
        """
        if grammar_dir is None:
            # Default: look for grammars/ in same directory as this file
            base_dir = Path(__file__).parent.parent
            grammar_dir = base_dir / "grammars"
        
        self.grammar_dir = Path(grammar_dir)
        self.cache: Dict[str, str] = {}
        
        # Validate grammar directory exists
        if not self.grammar_dir.exists():
            print(f"⚠️  Grammar directory not found: {self.grammar_dir}")
            print("   Creating directory and expecting grammar files to be added.")
            self.grammar_dir.mkdir(parents=True, exist_ok=True)
    
    def get_grammar(self, grammar_type: str) -> Optional[str]:
        """
        Load grammar by type name.
        
        Args:
            grammar_type: One of: agent_decision, sieve_analysis, 
                         parameter_adjustment, json_generic
        
        Returns:
            Grammar string content, or None if not found
        """
        # Check cache first
        if grammar_type in self.cache:
            return self.cache[grammar_type]
        
        # Build file path
        grammar_file = self.grammar_dir / f"{grammar_type}.gbnf"
        
        if not grammar_file.exists():
            print(f"⚠️  Grammar file not found: {grammar_file}")
            return None
        
        # Load and cache
        try:
            grammar_content = grammar_file.read_text()
            self.cache[grammar_type] = grammar_content
            return grammar_content
        except Exception as e:
            print(f"❌ Error loading grammar {grammar_file}: {e}")
            return None
    
    def get_grammar_for_request(self, prompt: str) -> Optional[str]:
        """
        Auto-select grammar based on prompt content.
        
        Scans prompt for keywords and selects appropriate grammar.
        Falls back to json_generic if no specific match.
        
        Args:
            prompt: The prompt being sent to LLM
        
        Returns:
            Grammar string content
        """
        prompt_lower = prompt.lower()
        
        # Check for keyword matches
        for keyword, grammar_type in REQUEST_GRAMMAR_MAP.items():
            if keyword in prompt_lower:
                return self.get_grammar(grammar_type.value)
        
        # Default to generic JSON
        return self.get_grammar(GrammarType.JSON_GENERIC.value)
    
    def list_available_grammars(self) -> list:
        """List all available grammar files."""
        if not self.grammar_dir.exists():
            return []
        
        return [f.stem for f in self.grammar_dir.glob("*.gbnf")]
    
    def validate_grammar(self, grammar_type: str) -> bool:
        """
        Validate that a grammar file exists and is readable.
        
        Args:
            grammar_type: Grammar type name
        
        Returns:
            True if valid, False otherwise
        """
        grammar = self.get_grammar(grammar_type)
        return grammar is not None and len(grammar) > 0


# Singleton instance for easy import
_loader_instance: Optional[GrammarLoader] = None


def get_grammar_loader(grammar_dir: Optional[str] = None) -> GrammarLoader:
    """
    Get singleton GrammarLoader instance.
    
    Args:
        grammar_dir: Optional grammar directory override
    
    Returns:
        GrammarLoader instance
    """
    global _loader_instance
    
    if _loader_instance is None or grammar_dir is not None:
        _loader_instance = GrammarLoader(grammar_dir)
    
    return _loader_instance


def get_grammar(grammar_type: str) -> Optional[str]:
    """
    Convenience function to get grammar by type.
    
    Args:
        grammar_type: One of: agent_decision, sieve_analysis,
                     parameter_adjustment, json_generic
    
    Returns:
        Grammar string content
    """
    return get_grammar_loader().get_grammar(grammar_type)


def get_grammar_for_prompt(prompt: str) -> Optional[str]:
    """
    Convenience function to auto-select grammar based on prompt.
    
    Args:
        prompt: The prompt being sent to LLM
    
    Returns:
        Grammar string content
    """
    return get_grammar_loader().get_grammar_for_request(prompt)


# ============================================================================
# TEST / DEMO
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("GBNF Grammar Loader - Test")
    print("=" * 60)
    
    loader = GrammarLoader()
    
    print(f"\nGrammar directory: {loader.grammar_dir}")
    print(f"Available grammars: {loader.list_available_grammars()}")
    
    # Test loading each grammar
    for grammar_type in GrammarType:
        grammar = loader.get_grammar(grammar_type.value)
        if grammar:
            lines = len(grammar.split('\n'))
            print(f"✅ {grammar_type.value}: {lines} lines loaded")
        else:
            print(f"❌ {grammar_type.value}: NOT FOUND")
    
    # Test auto-selection
    print("\n--- Auto-selection test ---")
    test_prompts = [
        "Evaluate the sieve results and decide whether to proceed",
        "Analyze the bidirectional survivors from the forward pass",
        "Suggest parameter adjustments for window_size",
        "Generate a summary report",
    ]
    
    for prompt in test_prompts:
        grammar = loader.get_grammar_for_request(prompt)
        grammar_name = "unknown"
        for gt in GrammarType:
            if loader.get_grammar(gt.value) == grammar:
                grammar_name = gt.value
                break
        print(f"  '{prompt[:50]}...' → {grammar_name}")
