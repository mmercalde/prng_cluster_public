#!/usr/bin/env python3
"""
Direct WATCHER-LLM integration test.

This test would have caught the "method missing" and "parse errors" issues
without requiring a full pipeline run.

Tests:
1. Method existence - evaluate_watcher_decision exists
2. Valid decision JSON (proceed) - should work
3. Invalid JSON - fallback should escalate safely
4. Markdown-wrapped JSON - should be extracted

Version: 1.0.0
Date: January 9, 2026
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRouterMethodExists:
    """Test that required methods exist on LLMRouter."""
    
    def test_router_importable(self):
        """LLMRouter should be importable."""
        from llm_services.llm_router import LLMRouter, get_router
        assert LLMRouter is not None
        assert get_router is not None
    
    def test_evaluate_watcher_decision_exists(self):
        """evaluate_watcher_decision must exist on router."""
        from llm_services.llm_router import LLMRouter
        assert hasattr(LLMRouter, 'evaluate_watcher_decision')
        assert callable(getattr(LLMRouter, 'evaluate_watcher_decision'))
    
    def test_evaluate_decision_alias_exists(self):
        """evaluate_decision alias must exist for backward compatibility."""
        from llm_services.llm_router import LLMRouter
        assert hasattr(LLMRouter, 'evaluate_decision')
        assert callable(getattr(LLMRouter, 'evaluate_decision'))
    
    def test_generate_json_exists(self):
        """generate_json must exist for other agents."""
        from llm_services.llm_router import LLMRouter
        assert hasattr(LLMRouter, 'generate_json')
        assert callable(getattr(LLMRouter, 'generate_json'))
    
    def test_orchestrate_exists(self):
        """orchestrate must exist for planning tasks."""
        from llm_services.llm_router import LLMRouter
        assert hasattr(LLMRouter, 'orchestrate')
        assert callable(getattr(LLMRouter, 'orchestrate'))


class TestRouterReturnTypes:
    """Test that return types are locked as specified."""
    
    @patch('llm_services.llm_router.requests.post')
    def test_route_returns_string(self, mock_post):
        """route() must return str, not dict."""
        mock_response = Mock()
        mock_response.json.return_value = {"content": "test response"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        from llm_services.llm_router import LLMRouter
        
        # Mock config loading
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps({
                "primary": {"port": 8080, "model": "test", "stop_tokens": []},
                "backup": {"model": "test", "working_dir": "/tmp"},
                "routing": {"escalation_triggers": [], "request_timeout_seconds": 30}
            })
            router = LLMRouter.__new__(LLMRouter)
            router.config = {
                "primary": {"port": 8080, "model": "test", "stop_tokens": []},
                "backup": {"model": "test", "working_dir": "/tmp"},
                "routing": {"escalation_triggers": [], "request_timeout_seconds": 30}
            }
            router.primary_endpoint = "http://localhost:8080/completion"
            router.escalation_triggers = []
            router.metrics = Mock()
            router.metrics.trace = []
            
            result = router._call_primary("test", 0.7, 100)
            assert isinstance(result, str), f"route() should return str, got {type(result)}"


class TestDecisionParsing:
    """Test decision parsing with various inputs."""
    
    def get_mock_router(self):
        """Create a mock router for testing."""
        from llm_services.llm_router import LLMRouter
        
        router = LLMRouter.__new__(LLMRouter)
        router.config = {
            "primary": {"port": 8080, "model": "test", "stop_tokens": []},
            "backup": {"model": "test", "working_dir": "/tmp"},
            "routing": {"escalation_triggers": [], "request_timeout_seconds": 30}
        }
        router.primary_endpoint = "http://localhost:8080/completion"
        router.escalation_triggers = []
        router.metrics = Mock()
        router.metrics.trace = []
        router.metrics.primary_calls = 0
        router.metrics.backup_calls = 0
        router.metrics.total_tokens = 0
        return router
    
    def test_valid_proceed_decision(self):
        """Valid proceed decision should be parsed correctly."""
        from llm_services.llm_router import LLMRouter
        
        router = self.get_mock_router()
        
        # Mock _call_primary_with_grammar to return valid JSON
        valid_response = json.dumps({
            "decision": "proceed",
            "confidence": 0.85,
            "reasoning": "High survivor count indicates good signal",
            "retry_reason": None,
            "primary_signal": "bidirectional_count",
            "suggested_params": None,
            "warnings": [],
            "checks": {
                "used_rates": True,
                "mentioned_data_source": True,
                "avoided_absolute_only": True
            }
        })
        
        with patch.object(router, '_call_primary_with_grammar', return_value=valid_response):
            with patch.object(router, '_is_primary_available', return_value=True):
                decision = router.evaluate_watcher_decision(
                    "Test prompt",
                    step_id="test_step",
                    agent="test_agent"
                )
        
        assert decision["decision"] == "proceed"
        assert decision["confidence"] == 0.85
        assert "reasoning" in decision
    
    def test_valid_escalate_decision(self):
        """Valid escalate decision should be parsed correctly."""
        from llm_services.llm_router import LLMRouter
        
        router = self.get_mock_router()
        
        valid_response = json.dumps({
            "decision": "escalate",
            "confidence": 0.4,
            "reasoning": "Low confidence requires human review",
            "retry_reason": None,
            "primary_signal": "low_confidence",
            "suggested_params": None,
            "warnings": ["Manual review recommended"],
            "checks": {
                "used_rates": True,
                "mentioned_data_source": True,
                "avoided_absolute_only": True
            }
        })
        
        with patch.object(router, '_call_primary_with_grammar', return_value=valid_response):
            with patch.object(router, '_is_primary_available', return_value=True):
                decision = router.evaluate_watcher_decision(
                    "Test prompt",
                    step_id="test_step",
                    agent="test_agent"
                )
        
        assert decision["decision"] == "escalate"
        assert decision["confidence"] == 0.4
    
    def test_invalid_json_fallback_escalates(self):
        """Invalid JSON should trigger safe fallback (escalate)."""
        from llm_services.llm_router import LLMRouter
        
        router = self.get_mock_router()
        
        # Return invalid JSON
        with patch.object(router, '_call_primary_with_grammar', return_value="This is not valid JSON"):
            with patch.object(router, '_is_primary_available', return_value=True):
                decision = router.evaluate_watcher_decision(
                    "Test prompt",
                    step_id="test_step",
                    agent="test_agent"
                )
        
        # SAFETY RULE: If parsing fails, default to escalate
        assert decision["decision"] == "escalate"
        assert decision["confidence"] <= 0.5
    
    def test_markdown_wrapped_json_extracted(self):
        """JSON in markdown code block should be extracted."""
        from llm_services.llm_router import LLMRouter
        
        router = self.get_mock_router()
        
        markdown_response = '''```json
{
    "decision": "proceed",
    "confidence": 0.9,
    "reasoning": "All checks passed",
    "retry_reason": null,
    "primary_signal": "all_clear",
    "suggested_params": null,
    "warnings": [],
    "checks": {
        "used_rates": true,
        "mentioned_data_source": true,
        "avoided_absolute_only": true
    }
}
```'''
        
        # This tests the _parse_json_response fallback path
        with patch.object(router, '_call_primary_with_grammar', return_value=markdown_response):
            with patch.object(router, '_is_primary_available', return_value=True):
                decision = router.evaluate_watcher_decision(
                    "Test prompt",
                    step_id="test_step",
                    agent="test_agent"
                )
        
        assert decision["decision"] == "proceed"
        assert decision["confidence"] == 0.9


class TestWatcherIntegration:
    """Test WATCHER agent can use the router."""
    
    def test_watcher_can_import_router(self):
        """WATCHER should be able to import and use router."""
        try:
            from agents.watcher_agent import WatcherAgent, WatcherConfig
            from llm_services.llm_router import get_router
            
            # These imports should work
            assert WatcherAgent is not None
            assert get_router is not None
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
    
    def test_watcher_config_has_llm_options(self):
        """WatcherConfig should have LLM-related options."""
        from agents.watcher_agent import WatcherConfig
        
        config = WatcherConfig()
        assert hasattr(config, 'use_llm')
        assert hasattr(config, 'use_grammar')


class TestSafetyBehavior:
    """Test safety behaviors per Team Beta requirements."""
    
    def test_backup_path_escalates_on_ambiguity(self):
        """
        SAFETY RULE:
        If parsing fails or backup LLM output is ambiguous,
        default to {"decision": "escalate"}.
        """
        from llm_services.llm_router import LLMRouter
        
        router = LLMRouter.__new__(LLMRouter)
        router.config = {
            "primary": {"port": 8080, "model": "test", "stop_tokens": []},
            "backup": {"model": "test", "working_dir": "/tmp"},
            "routing": {"escalation_triggers": [], "request_timeout_seconds": 30}
        }
        router.metrics = Mock()
        router.metrics.trace = []
        
        # Simulate backup returning ambiguous response
        with patch.object(router, '_is_primary_available', return_value=False):
            with patch.object(router, '_call_backup', return_value="I'm not sure what to do"):
                decision = router.evaluate_watcher_decision(
                    "Test prompt",
                    step_id="test_step",
                    agent="test_agent"
                )
        
        # Must escalate on ambiguity
        assert decision["decision"] == "escalate"
        assert decision["confidence"] <= 0.5


if __name__ == "__main__":
    # Run with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
