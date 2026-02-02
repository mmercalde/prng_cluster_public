#!/usr/bin/env python3
"""
LLM Lifecycle Manager — On-demand server start/stop for GPU resource fairness.

Version: 1.0.0
Date: 2026-02-01 (Session 57)
Proposal: PROPOSAL_LLM_Infrastructure_Optimization_v1_1.md (Part B)
Team Beta: APPROVED with required guardrails (stop guard + startup time log)

Purpose:
    Manages the llama.cpp server lifecycle so that GPU VRAM is freed during
    compute-intensive phases (selfplay, Steps 3/5/6) and restored for LLM
    evaluation phases (Chapter 13 advisor, WATCHER step evaluation).

Integration Points:
    - WATCHER agent: dispatch_selfplay() → stop(), dispatch_learning_loop() → stop()/ensure_running()
    - Chapter 13 LLM advisor: session() context manager for candidate evaluation
    - Standalone: ensure_running() before any llm_router.evaluate() call

Architecture:
    - Does NOT modify llm_router.py (routing logic is orthogonal)
    - Relies on existing 3-tier fallback (grammar → HTTP → heuristic) when server unavailable
    - Uses start_llm_servers.sh for startup (proven working)
    - Uses pkill for shutdown (standard llama.cpp management)

Team Beta Required Guardrails:
    1. stop() guard: `if self.process is None and not self._find_server_process(): return`
    2. Health-check startup time logged once per lifecycle session
"""

import json
import logging
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


class LLMLifecycleManager:
    """On-demand LLM server management.

    Manages start/stop of the llama.cpp server hosting DeepSeek-R1-14B
    on Zeus's dual RTX 3080 Ti GPUs. Frees ~4.25GB VRAM per GPU when stopped.

    Usage:
        mgr = LLMLifecycleManager()

        # Before LLM evaluation
        mgr.ensure_running()
        response = llm_router.evaluate(prompt)

        # After evaluation (optional — keeps server for rapid re-use)
        mgr.stop()

    Context manager support:
        with mgr.session():
            response = llm_router.evaluate(prompt)
        # Server auto-stops after context exit

    With idle timeout (for rapid-fire evaluations):
        mgr = LLMLifecycleManager(idle_timeout_sec=60)
        # Server stays alive for 60s after last use
    """

    # Default paths relative to project root
    DEFAULT_CONFIG_PATH = "llm_services/llm_server_config.json"
    DEFAULT_STARTUP_SCRIPT = "llm_services/start_llm_servers.sh"
    DEFAULT_HEALTH_URL = "http://localhost:8080/health"

    def __init__(
        self,
        project_root: Optional[str] = None,
        config_path: Optional[str] = None,
        idle_timeout_sec: int = 0,
    ):
        """Initialize lifecycle manager.

        Args:
            project_root: Path to distributed_prng_analysis/ directory.
                          Defaults to auto-detection from this file's location.
            config_path: Path to llm_server_config.json. Defaults to standard location.
            idle_timeout_sec: If > 0, server stays alive for N seconds after last use.
                             Useful during dispatch_learning_loop() where 3 step
                             evaluations happen within minutes.
        """
        # Resolve project root
        if project_root:
            self._project_root = Path(project_root)
        else:
            # llm_services/llm_lifecycle.py → project root is parent
            self._project_root = Path(__file__).resolve().parent.parent

        # Load config
        if config_path:
            self._config_path = Path(config_path)
        else:
            self._config_path = self._project_root / self.DEFAULT_CONFIG_PATH

        self._config = self._load_config()

        # Server state
        self.process: Optional[subprocess.Popen] = None
        self._idle_timeout_sec = idle_timeout_sec
        self._last_use_time: float = 0.0
        self._session_start_time: float = 0.0
        self._startup_count: int = 0

        # Derived from config
        self._port = self._config.get("primary", {}).get("port", 8080)
        self._health_url = f"http://localhost:{self._port}/health"
        self._startup_script = self._project_root / self.DEFAULT_STARTUP_SCRIPT
        self._ctx_size = self._config.get("primary", {}).get("context_length", 32768)

        logger.info(
            "LLMLifecycleManager initialized: port=%d, ctx_size=%d, idle_timeout=%ds",
            self._port,
            self._ctx_size,
            self._idle_timeout_sec,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_healthy(self, timeout_sec: float = 2.0) -> bool:
        """Check if LLM server responds on health endpoint.

        Args:
            timeout_sec: HTTP timeout for health check.

        Returns:
            True if server is running and healthy.
        """
        try:
            req = urllib.request.Request(self._health_url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError, TimeoutError):
            return False

    def ensure_running(self, timeout_sec: int = 30) -> bool:
        """Start server if not already running. Block until healthy.

        Args:
            timeout_sec: Maximum seconds to wait for server to become healthy.

        Returns:
            True if server is healthy (whether it was already running or just started).
        """
        if self.is_healthy():
            logger.debug("LLM server already healthy")
            self._last_use_time = time.time()
            return True

        logger.info("LLM server not healthy — starting...")
        return self._start_server(timeout_sec)

    def stop(self, timeout_sec: int = 10):
        """Gracefully stop the LLM server, freeing GPU VRAM.

        Team Beta Required Guardrail: Guard against double-stop crashes
        during fallback escalation, partial Phase 7 failures, or
        interrupted learning loops.

        Args:
            timeout_sec: Maximum seconds to wait for server to stop.
        """
        # ── Team Beta Required Guardrail #1 ──────────────────────────
        # Guard against double-stop when process is already gone
        server_pid = self._find_server_process()
        if self.process is None and server_pid is None:
            logger.debug("LLM server already stopped (no process found)")
            return
        # ─────────────────────────────────────────────────────────────

        logger.info("Stopping LLM server (freeing GPU VRAM)...")
        stop_start = time.time()

        # Try graceful shutdown via our tracked process first
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=timeout_sec)
                logger.info(
                    "LLM server stopped via tracked process (%.1fs)",
                    time.time() - stop_start,
                )
            except subprocess.TimeoutExpired:
                logger.warning("Graceful shutdown timed out — killing")
                self.process.kill()
                self.process.wait(timeout=5)
            except Exception as e:
                logger.warning("Error stopping tracked process: %s", e)
            finally:
                self.process = None

        # Also pkill any orphaned llama-server processes
        if server_pid is not None or self._find_server_process() is not None:
            try:
                subprocess.run(
                    ["pkill", "-f", "llama-server"],
                    timeout=timeout_sec,
                    capture_output=True,
                )
                logger.debug("pkill sent to any remaining llama-server processes")
            except Exception as e:
                logger.warning("pkill failed: %s", e)

        # Verify stopped
        time.sleep(0.5)
        if self.is_healthy(timeout_sec=1.0):
            logger.error("LLM server still healthy after stop attempt!")
        else:
            logger.info("LLM server confirmed stopped — GPU VRAM freed")

    @contextmanager
    def session(self):
        """Context manager: start → yield → stop.

        Usage:
            with lifecycle.session():
                response = llm_router.evaluate(prompt)
            # Server auto-stops here, freeing GPU VRAM
        """
        self._session_start_time = time.time()
        try:
            if not self.ensure_running():
                logger.warning(
                    "LLM session started but server unhealthy — "
                    "falling through to existing fallback chain"
                )
            yield
        finally:
            self._last_use_time = time.time()
            session_duration = time.time() - self._session_start_time

            if self._idle_timeout_sec > 0:
                logger.info(
                    "LLM session ended (%.1fs). Server kept alive (idle_timeout=%ds)",
                    session_duration,
                    self._idle_timeout_sec,
                )
                # Don't stop — let idle timeout handle it
            else:
                logger.info("LLM session ended (%.1fs). Stopping server.", session_duration)
                self.stop()

    def should_idle_stop(self) -> bool:
        """Check if idle timeout has elapsed (for external polling).

        Call this from a timer or event loop to implement idle shutdown.

        Returns:
            True if server should be stopped due to idle timeout.
        """
        if self._idle_timeout_sec <= 0:
            return False
        if self._last_use_time == 0.0:
            return False
        elapsed = time.time() - self._last_use_time
        return elapsed > self._idle_timeout_sec

    def idle_check_and_stop(self):
        """Stop server if idle timeout has elapsed.

        Convenience method for event loops / daemon timers.
        """
        if self.should_idle_stop():
            logger.info(
                "Idle timeout reached (%.0fs since last use). Stopping LLM server.",
                time.time() - self._last_use_time,
            )
            self.stop()

    @property
    def is_running(self) -> bool:
        """Quick check: is the server process alive (does not verify health)."""
        if self.process is not None and self.process.poll() is None:
            return True
        return self._find_server_process() is not None

    @property
    def startup_count(self) -> int:
        """Number of times server has been started during this manager's lifetime."""
        return self._startup_count

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _load_config(self) -> dict:
        """Load LLM server configuration."""
        try:
            with open(self._config_path, "r") as f:
                config = json.load(f)
            logger.debug("Loaded LLM config from %s", self._config_path)
            return config
        except FileNotFoundError:
            logger.warning(
                "LLM config not found at %s — using defaults", self._config_path
            )
            return {
                "primary": {
                    "port": 8080,
                    "context_length": 32768,
                }
            }
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in LLM config: %s", e)
            return {"primary": {"port": 8080, "context_length": 32768}}

    def _start_server(self, timeout_sec: int) -> bool:
        """Launch LLM server and poll health until ready.

        Args:
            timeout_sec: Maximum seconds to wait for healthy status.

        Returns:
            True if server became healthy within timeout.
        """
        start_time = time.time()
        self._startup_count += 1

        # Try startup script first (preferred — handles model path, GPU config)
        if self._startup_script.exists():
            logger.info(
                "Starting LLM server via %s (attempt #%d)",
                self._startup_script.name,
                self._startup_count,
            )
            try:
                self.process = subprocess.Popen(
                    ["bash", str(self._startup_script)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(self._project_root),
                )
                # The script starts the server in background (nohup),
                # so the script itself exits quickly. We poll health below.
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                logger.warning("Startup script timed out (15s) — polling health anyway")
            except Exception as e:
                logger.error("Failed to run startup script: %s", e)
                return False
        else:
            logger.warning(
                "Startup script not found at %s — attempting direct llama-server launch",
                self._startup_script,
            )
            return self._start_server_direct(timeout_sec)

        # Poll health endpoint
        healthy = self._poll_health(timeout_sec, start_time)

        if healthy:
            startup_duration = time.time() - start_time
            # ── Team Beta Required Guardrail #2 ──────────────────────
            # Log startup time once per lifecycle session for observability
            logger.info(
                "LLM server healthy after %.1fs (startup #%d). "
                "ctx_size=%d, kv_cache_est≈2.6GB/GPU",
                startup_duration,
                self._startup_count,
                self._ctx_size,
            )
            # ─────────────────────────────────────────────────────────
        else:
            logger.error(
                "LLM server failed to become healthy within %ds", timeout_sec
            )

        return healthy

    def _start_server_direct(self, timeout_sec: int) -> bool:
        """Fallback: start llama-server directly without the startup script.

        Used when start_llm_servers.sh is not found.
        """
        start_time = time.time()

        # Find model file
        model_name = self._config.get("primary", {}).get(
            "model_file", "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf"
        )
        model_path = Path.home() / "models" / model_name

        if not model_path.exists():
            logger.error("Model file not found: %s", model_path)
            return False

        n_gpu_layers = self._config.get("primary", {}).get("n_gpu_layers", 99)

        cmd = [
            "llama-server",
            "--model", str(model_path),
            "--port", str(self._port),
            "--ctx-size", str(self._ctx_size),
            "--n-gpu-layers", str(n_gpu_layers),
            "--threads", "4",
            "--log-disable",
        ]

        logger.info("Starting llama-server directly: %s", " ".join(cmd))

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=open("/tmp/llm_server_primary.log", "w"),
                cwd=str(self._project_root),
            )
        except FileNotFoundError:
            logger.error("llama-server binary not found in PATH")
            return False
        except Exception as e:
            logger.error("Failed to start llama-server: %s", e)
            return False

        healthy = self._poll_health(timeout_sec, start_time)

        if healthy:
            startup_duration = time.time() - start_time
            logger.info(
                "LLM server healthy after %.1fs (direct launch, startup #%d). "
                "ctx_size=%d, kv_cache_est≈2.6GB/GPU",
                startup_duration,
                self._startup_count,
                self._ctx_size,
            )

        return healthy

    def _poll_health(self, timeout_sec: int, start_time: float) -> bool:
        """Poll health endpoint until healthy or timeout.

        Args:
            timeout_sec: Maximum total wait time from start_time.
            start_time: When the startup sequence began.

        Returns:
            True if server became healthy.
        """
        poll_interval = 1.0
        while (time.time() - start_time) < timeout_sec:
            if self.is_healthy(timeout_sec=2.0):
                return True
            time.sleep(poll_interval)
        return False

    def _find_server_process(self) -> Optional[int]:
        """Find PID of any running llama-server process.

        Returns:
            PID if found, None otherwise.
        """
        try:
            result = subprocess.run(
                ["pgrep", "-f", "llama-server"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                pid = int(result.stdout.strip().split("\n")[0])
                return pid
        except (subprocess.TimeoutExpired, ValueError, Exception):
            pass
        return None


# ------------------------------------------------------------------
# Module-level convenience (singleton pattern for WATCHER integration)
# ------------------------------------------------------------------

_default_manager: Optional[LLMLifecycleManager] = None


def get_lifecycle_manager(**kwargs) -> LLMLifecycleManager:
    """Get or create the default lifecycle manager singleton.

    Usage from watcher_agent.py:
        from llm_services.llm_lifecycle import get_lifecycle_manager
        lifecycle = get_lifecycle_manager()
        lifecycle.ensure_running()
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = LLMLifecycleManager(**kwargs)
    return _default_manager


# ------------------------------------------------------------------
# Self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    print("=" * 60)
    print("LLM Lifecycle Manager — Self-Test")
    print("=" * 60)

    mgr = LLMLifecycleManager()

    # Test 1: Health check (should work whether server is running or not)
    print("\n[Test 1] Health check...")
    healthy = mgr.is_healthy()
    print(f"  Server healthy: {healthy}")

    # Test 2: Stop guard (Team Beta guardrail — should not crash)
    print("\n[Test 2] Stop guard (no-op when not running)...")
    mgr.stop()
    print("  ✅ Stop guard works (no crash)")

    # Test 3: Double stop (Team Beta guardrail)
    print("\n[Test 3] Double stop (no-op)...")
    mgr.stop()
    mgr.stop()
    print("  ✅ Double stop works (no crash)")

    # Test 4: Config loading
    print(f"\n[Test 4] Config loaded...")
    print(f"  Port: {mgr._port}")
    print(f"  Context: {mgr._ctx_size}")
    print(f"  ✅ Config loaded successfully")

    # Test 5: is_running property
    print(f"\n[Test 5] is_running: {mgr.is_running}")

    # Test 6: Startup count
    print(f"\n[Test 6] Startup count: {mgr.startup_count}")

    if "--start" in sys.argv:
        print("\n[Test 7] Starting server...")
        success = mgr.ensure_running(timeout_sec=30)
        print(f"  Server started: {success}")

        if success:
            print("\n[Test 8] Session context manager...")
            with mgr.session():
                print("  Inside session — server should be running")
                print(f"  Healthy: {mgr.is_healthy()}")
            print("  Outside session — server should be stopped")
            print(f"  Healthy: {mgr.is_healthy()}")

    print("\n" + "=" * 60)
    print("Self-test complete")
    print("=" * 60)
