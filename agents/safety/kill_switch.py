#!/usr/bin/env python3
"""
Kill Switch - Safety override controls for autonomous agents.

Provides emergency stop capability and safety checks for
autonomous pipeline execution.

Version: 3.2.0
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
import os
import json


class SafetyLevel(str, Enum):
    """Safety level for operations."""
    NORMAL = "normal"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"
    HALT = "halt"


class SafetyCheck(BaseModel):
    """Individual safety check result."""
    
    name: str
    passed: bool
    level: SafetyLevel = SafetyLevel.NORMAL
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            "name": self.name,
            "passed": self.passed,
            "level": self.level.value,
            "message": self.message
        }


class KillSwitch(BaseModel):
    """
    Safety kill switch for autonomous operations.
    
    Monitors for:
    1. Manual halt file (/tmp/agent_halt)
    2. Resource thresholds (memory, disk)
    3. Retry limits
    4. Time limits
    5. Anomaly escalation
    """
    
    enabled: bool = True
    halt_file_path: str = "/tmp/agent_halt"
    
    # Thresholds
    max_consecutive_failures: int = 3
    max_retries_per_step: int = 5
    max_runtime_minutes: int = 480  # 8 hours
    min_disk_free_gb: float = 10.0
    max_memory_usage_percent: float = 95.0
    
    # Current state
    consecutive_failures: int = 0
    retries_this_step: int = 0
    start_time: datetime = Field(default_factory=datetime.utcnow)
    
    # Safety checks
    checks: List[SafetyCheck] = Field(default_factory=list)
    current_level: SafetyLevel = SafetyLevel.NORMAL
    
    def check_all(self) -> bool:
        """
        Run all safety checks.
        
        Returns True if safe to proceed, False if halt required.
        """
        self.checks = []
        
        # 1. Check for manual halt file
        self.checks.append(self._check_halt_file())
        
        # 2. Check consecutive failures
        self.checks.append(self._check_failures())
        
        # 3. Check retry limit
        self.checks.append(self._check_retries())
        
        # 4. Check runtime
        self.checks.append(self._check_runtime())
        
        # 5. Check disk space
        self.checks.append(self._check_disk())
        
        # 6. Check memory
        self.checks.append(self._check_memory())
        
        # Determine overall level
        self._update_level()
        
        return self.is_safe()
    
    def _check_halt_file(self) -> SafetyCheck:
        """Check for manual halt file."""
        exists = os.path.exists(self.halt_file_path)
        
        if exists:
            # Try to read reason
            try:
                with open(self.halt_file_path) as f:
                    reason = f.read().strip() or "Manual halt requested"
            except:
                reason = "Manual halt file exists"
            
            return SafetyCheck(
                name="halt_file",
                passed=False,
                level=SafetyLevel.HALT,
                message=reason
            )
        
        return SafetyCheck(
            name="halt_file",
            passed=True,
            level=SafetyLevel.NORMAL,
            message="No halt file present"
        )
    
    def _check_failures(self) -> SafetyCheck:
        """Check consecutive failure count."""
        if self.consecutive_failures >= self.max_consecutive_failures:
            return SafetyCheck(
                name="consecutive_failures",
                passed=False,
                level=SafetyLevel.CRITICAL,
                message=f"Too many consecutive failures: {self.consecutive_failures}"
            )
        elif self.consecutive_failures >= self.max_consecutive_failures - 1:
            return SafetyCheck(
                name="consecutive_failures",
                passed=True,
                level=SafetyLevel.WARNING,
                message=f"Approaching failure limit: {self.consecutive_failures}/{self.max_consecutive_failures}"
            )
        
        return SafetyCheck(
            name="consecutive_failures",
            passed=True,
            level=SafetyLevel.NORMAL,
            message=f"Failures: {self.consecutive_failures}/{self.max_consecutive_failures}"
        )
    
    def _check_retries(self) -> SafetyCheck:
        """Check retry count for current step."""
        if self.retries_this_step >= self.max_retries_per_step:
            return SafetyCheck(
                name="retry_limit",
                passed=False,
                level=SafetyLevel.CRITICAL,
                message=f"Max retries reached: {self.retries_this_step}"
            )
        elif self.retries_this_step >= self.max_retries_per_step - 1:
            return SafetyCheck(
                name="retry_limit",
                passed=True,
                level=SafetyLevel.WARNING,
                message=f"Approaching retry limit: {self.retries_this_step}/{self.max_retries_per_step}"
            )
        
        return SafetyCheck(
            name="retry_limit",
            passed=True,
            level=SafetyLevel.NORMAL,
            message=f"Retries: {self.retries_this_step}/{self.max_retries_per_step}"
        )
    
    def _check_runtime(self) -> SafetyCheck:
        """Check total runtime."""
        elapsed = (datetime.utcnow() - self.start_time).total_seconds() / 60
        
        if elapsed >= self.max_runtime_minutes:
            return SafetyCheck(
                name="runtime",
                passed=False,
                level=SafetyLevel.CRITICAL,
                message=f"Max runtime exceeded: {elapsed:.0f} minutes"
            )
        elif elapsed >= self.max_runtime_minutes * 0.9:
            return SafetyCheck(
                name="runtime",
                passed=True,
                level=SafetyLevel.WARNING,
                message=f"Approaching max runtime: {elapsed:.0f}/{self.max_runtime_minutes} minutes"
            )
        
        return SafetyCheck(
            name="runtime",
            passed=True,
            level=SafetyLevel.NORMAL,
            message=f"Runtime: {elapsed:.0f}/{self.max_runtime_minutes} minutes"
        )
    
    def _check_disk(self) -> SafetyCheck:
        """Check disk space."""
        try:
            statvfs = os.statvfs('/')
            free_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024 ** 3)
            
            if free_gb < self.min_disk_free_gb:
                return SafetyCheck(
                    name="disk_space",
                    passed=False,
                    level=SafetyLevel.CRITICAL,
                    message=f"Low disk space: {free_gb:.1f}GB free"
                )
            elif free_gb < self.min_disk_free_gb * 2:
                return SafetyCheck(
                    name="disk_space",
                    passed=True,
                    level=SafetyLevel.CAUTION,
                    message=f"Disk space getting low: {free_gb:.1f}GB free"
                )
            
            return SafetyCheck(
                name="disk_space",
                passed=True,
                level=SafetyLevel.NORMAL,
                message=f"Disk space: {free_gb:.1f}GB free"
            )
        except Exception as e:
            return SafetyCheck(
                name="disk_space",
                passed=True,
                level=SafetyLevel.CAUTION,
                message=f"Could not check disk: {e}"
            )
    
    def _check_memory(self) -> SafetyCheck:
        """Check memory usage."""
        try:
            with open('/proc/meminfo') as f:
                meminfo = f.read()
            
            total = available = 0
            for line in meminfo.split('\n'):
                if line.startswith('MemTotal:'):
                    total = int(line.split()[1])
                elif line.startswith('MemAvailable:'):
                    available = int(line.split()[1])
            
            if total > 0:
                used_percent = ((total - available) / total) * 100
                
                if used_percent >= self.max_memory_usage_percent:
                    return SafetyCheck(
                        name="memory",
                        passed=False,
                        level=SafetyLevel.CRITICAL,
                        message=f"High memory usage: {used_percent:.1f}%"
                    )
                elif used_percent >= self.max_memory_usage_percent * 0.9:
                    return SafetyCheck(
                        name="memory",
                        passed=True,
                        level=SafetyLevel.WARNING,
                        message=f"Memory usage elevated: {used_percent:.1f}%"
                    )
                
                return SafetyCheck(
                    name="memory",
                    passed=True,
                    level=SafetyLevel.NORMAL,
                    message=f"Memory usage: {used_percent:.1f}%"
                )
        except Exception as e:
            pass
        
        return SafetyCheck(
            name="memory",
            passed=True,
            level=SafetyLevel.NORMAL,
            message="Memory check skipped"
        )
    
    def _update_level(self):
        """Update current safety level based on checks."""
        # Priority: HALT > CRITICAL > WARNING > CAUTION > NORMAL
        level_priority = {
            SafetyLevel.HALT: 4,
            SafetyLevel.CRITICAL: 3,
            SafetyLevel.WARNING: 2,
            SafetyLevel.CAUTION: 1,
            SafetyLevel.NORMAL: 0
        }
        
        max_level = SafetyLevel.NORMAL
        for check in self.checks:
            if level_priority[check.level] > level_priority[max_level]:
                max_level = check.level
        
        self.current_level = max_level
    
    def is_safe(self) -> bool:
        """Check if safe to proceed."""
        if not self.enabled:
            return True
        return self.current_level not in [SafetyLevel.HALT, SafetyLevel.CRITICAL]
    
    def record_success(self):
        """Record a successful operation."""
        self.consecutive_failures = 0
    
    def record_failure(self):
        """Record a failed operation."""
        self.consecutive_failures += 1
    
    def record_retry(self):
        """Record a retry attempt."""
        self.retries_this_step += 1
    
    def reset_step(self):
        """Reset counters for new step."""
        self.retries_this_step = 0
    
    def create_halt_file(self, reason: str = ""):
        """Create manual halt file."""
        with open(self.halt_file_path, 'w') as f:
            f.write(reason or f"Halt created at {datetime.utcnow().isoformat()}")
    
    def clear_halt_file(self):
        """Remove halt file."""
        if os.path.exists(self.halt_file_path):
            os.remove(self.halt_file_path)
    
    def to_context_dict(self) -> Dict[str, Any]:
        """
        Generate safety context as clean dict for LLM.
        
        Hybrid JSON approach - data only.
        """
        self.check_all()
        
        return {
            "enabled": self.enabled,
            "safe_to_proceed": self.is_safe(),
            "current_level": self.current_level.value,
            "consecutive_failures": self.consecutive_failures,
            "retries_this_step": self.retries_this_step,
            "runtime_minutes": round((datetime.utcnow() - self.start_time).total_seconds() / 60, 1),
            "checks": [c.to_dict() for c in self.checks if not c.passed or c.level != SafetyLevel.NORMAL]
        }


# Convenience functions
def check_safety() -> bool:
    """Quick safety check."""
    ks = KillSwitch()
    return ks.check_all()


def create_halt(reason: str = "Manual halt"):
    """Create halt file to stop agents."""
    ks = KillSwitch()
    ks.create_halt_file(reason)
    print(f"Halt file created: {ks.halt_file_path}")


def clear_halt():
    """Clear halt file to allow agents to continue."""
    ks = KillSwitch()
    ks.clear_halt_file()
    print(f"Halt file cleared: {ks.halt_file_path}")
