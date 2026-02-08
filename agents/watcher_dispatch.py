#!/usr/bin/env python3
"""
WATCHER Dispatch Module — Phase 7 Part B
=========================================
Version: 1.0.0
Date: 2026-02-03
Session: 58

PURPOSE:
    Dispatch functions for WatcherAgent — bridges WATCHER to Selfplay
    and Learning Loop execution. Closes the autonomous operation gap.

    Functions are bound to WatcherAgent at import time via bind_to_watcher().
    This module does NOT modify any existing method behavior.

ARCHITECTURE:
    Chapter 13 Triggers → watcher_requests/ → WATCHER → Selfplay/Learning Loop
           ↑                                                        ↓
           └──────────────── Diagnostics ← Reality ←────────────────┘

GUARDRAILS:
    #1: All LLM context assembly goes through build_llm_context() ONLY.
        No inline prompt construction. No ad-hoc context dicts.
    #2: No baked-in token assumptions. bundle_factory owns prompt structure.

AUTHORITY CONTRACT (CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md):
    - Selfplay explores. Chapter 13 decides. WATCHER enforces.
    - WATCHER may start/stop selfplay jobs.
    - WATCHER must NOT decide promotion.
    - WATCHER must NOT trust selfplay output without Chapter 13 validation.
    - GPU sieving work MUST use coordinator.py / scripts_coordinator.py.

LLM LIFECYCLE:
    stop() before GPU-heavy dispatch → brief start() for evaluation → stop() again
    Frees BOTH 3080 Ti GPUs (~12GB VRAM) during selfplay and learning loops.

DEPENDENCIES:
    - agents/contexts/bundle_factory.py (build_llm_context)  — Part B0, commit ffe397a
    - llm_services/llm_lifecycle.py                          — Session 57
    - llm_services/grammar_loader.py                         — Session 57
    - selfplay_orchestrator.py                               — Phase 9B.2
    - agents/watcher_agent.py (WatcherAgent, STEP_MANIFESTS) — Chapter 12

BINDING:
    Called from watcher_agent.py:
        from agents.watcher_dispatch import bind_to_watcher
        bind_to_watcher(WatcherAgent)

    Or from patch_watcher_dispatch.py for automated installation.
"""

import json
import glob
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger("watcher_dispatch")

# ---------------------------------------------------------------------------
# Lazy import for bundle_factory — avoids circular imports at module load
# ---------------------------------------------------------------------------
_build_llm_context = None

def _get_build_llm_context():
    """Lazy-load build_llm_context from bundle_factory."""
    global _build_llm_context
    if _build_llm_context is None:
        try:
            from agents.contexts.bundle_factory import build_llm_context
            _build_llm_context = build_llm_context
        except ImportError as e:
            logger.error(f"Failed to import build_llm_context: {e}")
            logger.error("Ensure agents/contexts/bundle_factory.py exists (Part B0)")
            raise
    return _build_llm_context


# ===========================================================================
# B1: dispatch_selfplay()  (~70 lines)
# ===========================================================================

def dispatch_selfplay(self, request: dict, dry_run: bool = False) -> bool:
    """Execute selfplay_orchestrator.py with policy conditioning.

    Authority: WATCHER executes; Chapter 13 decides promotion.
    Contract:  Selfplay outputs are HYPOTHESES until Chapter 13 promotes.
    Invariant 4: GPU work via coordinators (selfplay handles this internally).

    Guardrail #1: Uses build_llm_context() for all LLM evaluation.
    Guardrail #2: No baked-in token assumptions.

    LLM Lifecycle:
        stop() → selfplay (GPU-heavy) → start() → evaluate candidate → done

    Args:
        request:  Dict with selfplay parameters.
                  Optional keys: episodes (int), survivors_file (str).
        dry_run:  If True, log intended actions without executing.

    Returns:
        True if selfplay completed successfully, False otherwise.
    """
    build_llm_context = _get_build_llm_context()
    run_id = f"selfplay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    episodes = request.get("episodes", 5)
    survivors_file = request.get("survivors_file", "survivors_with_scores.json")

    logger.info(f"[{run_id}] dispatch_selfplay: episodes={episodes}, "
                f"survivors={survivors_file}, dry_run={dry_run}")

    if dry_run:
        logger.info(f"[{run_id}] DRY RUN — would stop LLM, run "
                     f"selfplay_orchestrator.py --episodes {episodes} "
                     f"--policy-conditioned, then restart LLM for evaluation")
        _log_dispatch_decision(self, run_id, "selfplay", "DRY_RUN_OK",
                               {"episodes": episodes})
        return True

    # ── Safety gate ──────────────────────────────────────────────
    if os.path.exists("watcher_halt.flag"):
        logger.warning(f"[{run_id}] Halt flag present — aborting selfplay dispatch")
        return False

    # ── Pre-dispatch: free VRAM ──────────────────────────────────
    if hasattr(self, 'llm_lifecycle') and self.llm_lifecycle:
        logger.info(f"[{run_id}] Stopping LLM server (freeing VRAM for selfplay)")
        self.llm_lifecycle.stop()

    success = False
    candidate = None
    candidate_path = "learned_policy_candidate.json"

    try:
        # ── Spawn selfplay orchestrator ──────────────────────────
        cmd = [
            sys.executable, "selfplay_orchestrator.py",
            "--survivors", survivors_file,
            "--episodes", str(episodes),
            "--policy-conditioned",

        ]

        # ── Apply Strategy Advisor overrides to CLI args ─────────
        overrides = request.get("selfplay_overrides", {})
        if overrides.get("min_fitness_threshold"):
            cmd.extend(["--min-fitness", str(overrides["min_fitness_threshold"])])
        if overrides.get("max_episodes"):
            # Override episode count from advisor recommendation
            for i, arg in enumerate(cmd):
                if arg == "--episodes":
                    cmd[i + 1] = str(overrides["max_episodes"])
                    break

        logger.info(f"[{run_id}] Spawning: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1-hour safety timeout
            cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
        )

        if result.returncode != 0:
            logger.error(f"[{run_id}] Selfplay failed (rc={result.returncode})")
            if result.stderr:
                # Truncate to prevent log flooding
                logger.error(f"[{run_id}] stderr (first 500 chars): "
                             f"{result.stderr[:500]}")
            _log_dispatch_decision(self, run_id, "selfplay", "FAILED",
                                   {"returncode": result.returncode})
            return False

        logger.info(f"[{run_id}] Selfplay completed (rc=0)")
        success = True

        # ── Check for emitted candidate ──────────────────────────
        full_candidate_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", candidate_path
        )
        if os.path.exists(full_candidate_path):
            with open(full_candidate_path) as f:
                candidate = json.load(f)
            logger.info(f"[{run_id}] Candidate emitted — awaits Chapter 13 review")
        else:
            logger.info(f"[{run_id}] No candidate emitted (normal for early episodes)")

    except subprocess.TimeoutExpired:
        logger.error(f"[{run_id}] Selfplay timed out after 3600 s")
        _log_dispatch_decision(self, run_id, "selfplay", "TIMEOUT", {})
    except Exception as e:
        logger.error(f"[{run_id}] Selfplay exception: {e}")
        _log_dispatch_decision(self, run_id, "selfplay", "EXCEPTION",
                               {"error": str(e)})
    finally:
        # ── Post-dispatch: restart LLM for evaluation ────────────
        if hasattr(self, 'llm_lifecycle') and self.llm_lifecycle:
            logger.info(f"[{run_id}] Restarting LLM server")
            self.llm_lifecycle.ensure_running()

    # ── Post-dispatch: LLM evaluation of candidate (advisory) ────
    if success and candidate:
        try:
            prompt, grammar_name, bundle = build_llm_context(
                step_id=13,
                is_chapter_13=True,
                run_id=run_id,
                results={"candidate": candidate, "source": "selfplay"},
                state_paths=[candidate_path],
            )

            if getattr(self.config, 'use_llm', False):
                llm_response = _evaluate_step_via_bundle(self, prompt, grammar_name)
                logger.info(f"[{run_id}] LLM advisory on candidate: {llm_response}")
            else:
                logger.info(f"[{run_id}] LLM disabled — candidate awaits "
                            f"manual Chapter 13 review")

        except Exception as e:
            # Non-fatal: candidate still exists for Chapter 13
            logger.warning(f"[{run_id}] Post-selfplay LLM evaluation failed: {e}")

    _log_dispatch_decision(self, run_id, "selfplay",
                           "COMPLETED" if success else "FAILED",
                           {"episodes": episodes,
                            "candidate_emitted": candidate is not None})
    return success


# ===========================================================================
# B2: dispatch_learning_loop()  (~65 lines)
# ===========================================================================

def dispatch_learning_loop(self, scope: str = "steps_3_5_6",
                           request: Optional[dict] = None,
                           dry_run: bool = False) -> bool:
    """Execute partial or full pipeline rerun.

    Default scope is Steps 3→5→6 (scoring → anti-overfit → prediction).
    Full scope runs Steps 1→6.

    For each step:
      1. Stop LLM if step is GPU-heavy (3, 4, 5)
      2. Execute step via existing run_step()
      3. Restart LLM
      4. Evaluate via build_llm_context() — Guardrail #1
      5. If not "proceed", abort the loop

    Args:
        scope:    "steps_3_5_6" | "full" | "steps_X_Y_Z" (custom).
        request:  Optional dict with additional parameters from Chapter 13.
        dry_run:  If True, log intended actions without executing.

    Returns:
        True if all steps completed successfully, False otherwise.
    """
    # Import here to avoid circular reference at top level
    from agents.watcher_agent import STEP_MANIFESTS

    build_llm_context = _get_build_llm_context()
    run_id = f"learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # ── Parse step sequence ──────────────────────────────────────
    if scope == "steps_3_5_6":
        steps = [3, 5, 6]
    elif scope == "full":
        steps = [1, 2, 3, 4, 5, 6]
    elif scope.startswith("steps_"):
        try:
            steps = [int(s) for s in scope.replace("steps_", "").split("_")]
            if not all(1 <= s <= 6 for s in steps):
                raise ValueError("Steps must be 1-6")
        except ValueError as e:
            logger.error(f"[{run_id}] Invalid scope '{scope}': {e}")
            return False
    else:
        logger.error(f"[{run_id}] Unknown scope: {scope}")
        return False

    logger.info(f"[{run_id}] dispatch_learning_loop: steps={steps}, "
                f"dry_run={dry_run}")

    if dry_run:
        logger.info(f"[{run_id}] DRY RUN — would execute Steps {steps} "
                     f"sequentially with per-step LLM evaluation")
        _log_dispatch_decision(self, run_id, "learning_loop", "DRY_RUN_OK",
                               {"scope": scope, "steps": steps})
        return True

    # ── Safety gate ──────────────────────────────────────────────
    if os.path.exists("watcher_halt.flag"):
        logger.warning(f"[{run_id}] Halt flag present — aborting learning loop")
        return False

    completed_steps = []
    gpu_heavy_steps = {3, 4, 5}  # Steps that benefit from freed VRAM

    for step in steps:
        logger.info(f"[{run_id}] ═══ Step {step} starting ═══")

        is_gpu_heavy = step in gpu_heavy_steps

        # ── Free VRAM for GPU-heavy steps ────────────────────────
        if is_gpu_heavy and hasattr(self, 'llm_lifecycle') and self.llm_lifecycle:
            self.llm_lifecycle.stop()

        # ── Execute step ─────────────────────────────────────────
        try:
            results = self.run_step(step)
        except Exception as e:
            logger.error(f"[{run_id}] Step {step} exception: {e}")
            results = None

        if results is None:
            logger.error(f"[{run_id}] Step {step} returned no results — "
                         f"aborting loop")
            _log_dispatch_decision(self, run_id, "learning_loop",
                                   "STEP_FAILED",
                                   {"failed_step": step,
                                    "completed": completed_steps})
            # Ensure LLM is restarted before returning
            if is_gpu_heavy and hasattr(self, 'llm_lifecycle') and self.llm_lifecycle:
                self.llm_lifecycle.ensure_running()
            return False

        # ── Restart LLM for evaluation ───────────────────────────
        if is_gpu_heavy and hasattr(self, 'llm_lifecycle') and self.llm_lifecycle:
            self.llm_lifecycle.ensure_running()

        # ── Evaluate step via bundle factory (Guardrail #1) ──────
        try:
            manifest_name = STEP_MANIFESTS.get(step)
            manifest_path = None
            if manifest_name:
                manifests_dir = getattr(self.config, 'manifests_dir',
                                        'agent_manifests')
                manifest_path = os.path.join(manifests_dir, manifest_name)

            prompt, grammar_name, bundle = build_llm_context(
                step_id=step,
                run_id=run_id,
                results=results,
                manifest_path=manifest_path,
            )

            if getattr(self.config, 'use_llm', False):
                decision = _evaluate_step_via_bundle(self, prompt, grammar_name)
                action = decision.get("decision", "proceed")
                confidence = decision.get("confidence", 0.7)

                logger.info(f"[{run_id}] Step {step} LLM evaluation: "
                            f"{action} (conf={confidence:.2f})")

                if action != "proceed":
                    logger.warning(f"[{run_id}] Step {step}: {action} — "
                                   f"stopping learning loop")
                    _log_dispatch_decision(
                        self, run_id, "learning_loop", "STEP_HALTED",
                        {"halted_step": step, "action": action,
                         "confidence": confidence,
                         "completed": completed_steps})
                    return False
            else:
                # Heuristic fallback
                heuristic_decision, _ = self.evaluate_results(step, results)
                action = getattr(heuristic_decision, 'action', 'proceed')
                logger.info(f"[{run_id}] Step {step} heuristic: {action}")
                if action not in ("proceed",):
                    logger.warning(f"[{run_id}] Step {step} heuristic: "
                                   f"{action} — stopping")
                    return False

        except Exception as e:
            # Permissive: if evaluation fails but step succeeded, continue
            logger.warning(f"[{run_id}] Step {step} evaluation failed: {e} "
                           f"— continuing (step output exists)")

        completed_steps.append(step)

        # ── Check halt flag between steps ────────────────────────
        if os.path.exists("watcher_halt.flag"):
            logger.warning(f"[{run_id}] Halt flag detected between steps — "
                           f"stopping after Step {step}")
            _log_dispatch_decision(self, run_id, "learning_loop", "HALTED",
                                   {"completed": completed_steps})
            return False

    _log_dispatch_decision(self, run_id, "learning_loop", "COMPLETED",
                           {"scope": scope, "steps_completed": completed_steps})
    logger.info(f"[{run_id}] Learning loop completed: Steps {completed_steps}")
    return True


# ===========================================================================
# B3: process_chapter_13_request()  (~75 lines)
# ===========================================================================

def process_chapter_13_request(self, request_path: str,
                               dry_run: bool = False) -> str:
    """Process a request file from watcher_requests/ directory.

    Routes to dispatch_selfplay() or dispatch_learning_loop() based on
    the request_type field.  Optionally validates via build_llm_context()
    before execution (Guardrail #1).

    Authority: WATCHER validates and executes; Chapter 13 originates.

    Supported request_types:
        "selfplay_retrain"  → dispatch_selfplay()
        "learning_loop"     → dispatch_learning_loop()
        "pipeline_rerun"    → dispatch_learning_loop(scope="full")

    Args:
        request_path: Absolute or relative path to a JSON request file.
        dry_run:      If True, validate and log without executing.

    Returns:
        Status string: COMPLETED | FAILED | REJECTED | DRY_RUN_OK |
                        INVALID | UNKNOWN_TYPE
    """
    build_llm_context = _get_build_llm_context()
    run_id = f"ch13req_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"[{run_id}] Processing Chapter 13 request: {request_path}")

    # ── Load request ─────────────────────────────────────────────
    try:
        with open(request_path) as f:
            request = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"[{run_id}] Failed to load request: {e}")
        _archive_request(self, request_path, "INVALID")
        return "INVALID"

    request_type = request.get("request_type")
    source = request.get("source", "unknown")
    logger.info(f"[{run_id}] type={request_type}, source={source}")

    # ── Pre-execution LLM validation (Guardrail #1) ─────────────
    try:
        prompt, grammar_name, bundle = build_llm_context(
            step_id=13,
            is_chapter_13=True,
            run_id=run_id,
            results={"request": request, "request_type": request_type},
        )

        if getattr(self.config, 'use_llm', False) and not dry_run:
            validation = _evaluate_step_via_bundle(self, prompt, grammar_name)
            val_decision = validation.get("decision", "proceed")
            logger.info(f"[{run_id}] LLM pre-validation: {val_decision}")

            if val_decision == "escalate":
                logger.warning(f"[{run_id}] LLM recommends escalation — "
                               f"rejecting request")
                _archive_request(self, request_path, "REJECTED_BY_LLM")
                _log_dispatch_decision(self, run_id,
                                       f"ch13_request_{request_type}",
                                       "REJECTED",
                                       {"reason": "LLM escalation",
                                        "source": source})
                return "REJECTED"

    except Exception as e:
        logger.warning(f"[{run_id}] Pre-validation failed: {e} — "
                       f"proceeding with heuristic safety checks")

    # ── Strategy Advisor enrichment (before dispatch) ────────────
    if request_type == "selfplay_retrain":
        try:
            from parameter_advisor import StrategyAdvisor
            advisor = StrategyAdvisor(
                state_dir=os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), ".."
                )
            )
            rec = advisor.analyze()
            if rec and rec.selfplay_overrides:
                overrides = rec.selfplay_overrides.model_dump()
                logger.info(
                    f"[{run_id}] Strategy Advisor: focus={rec.focus_area.value}, "
                    f"action={rec.recommended_action.value}, "
                    f"mode={rec.advisor_model}"
                )
                # Merge overrides into request for dispatch_selfplay()
                request.setdefault("selfplay_overrides", {}).update(overrides)
            elif rec:
                logger.info(
                    f"[{run_id}] Strategy Advisor: action={rec.recommended_action.value} "
                    f"(no selfplay overrides)"
                )
            else:
                logger.info(f"[{run_id}] Strategy Advisor: gate not passed, skipping")
        except Exception as e:
            logger.warning(
                f"[{run_id}] Strategy Advisor failed: {e} — "
                f"proceeding without strategic guidance"
            )


    # ── Route by request type ────────────────────────────────────
    status = "UNKNOWN_TYPE"

    if request_type == "selfplay_retrain":
        if not _validate_selfplay_request(self, request):
            logger.warning(f"[{run_id}] Selfplay request validation failed")
            status = "REJECTED"
        elif dry_run:
            logger.info(f"[{run_id}] DRY RUN — would dispatch selfplay "
                        f"(episodes={request.get('episodes', 5)})")
            status = "DRY_RUN_OK"
        else:
            ok = dispatch_selfplay(self, request)
            status = "COMPLETED" if ok else "FAILED"

    elif request_type == "learning_loop":
        scope = request.get("scope", "steps_3_5_6")
        if dry_run:
            logger.info(f"[{run_id}] DRY RUN — would dispatch learning "
                        f"loop (scope={scope})")
            status = "DRY_RUN_OK"
        else:
            ok = dispatch_learning_loop(self, scope=scope, request=request)
            status = "COMPLETED" if ok else "FAILED"

    elif request_type == "pipeline_rerun":
        if dry_run:
            logger.info(f"[{run_id}] DRY RUN — would dispatch full pipeline")
            status = "DRY_RUN_OK"
        else:
            ok = dispatch_learning_loop(self, scope="full", request=request)
            status = "COMPLETED" if ok else "FAILED"

    else:
        logger.warning(f"[{run_id}] Unknown request_type: {request_type}")

    # ── Archive processed request ────────────────────────────────
    _archive_request(self, request_path, status)

    _log_dispatch_decision(self, run_id, f"ch13_request_{request_type}",
                           status, {"request_path": request_path,
                                    "source": source})
    logger.info(f"[{run_id}] Request result: {status}")
    return status


# ===========================================================================
# B4: _scan_watcher_requests()  (~25 lines)
# ===========================================================================

def _scan_watcher_requests(self, dry_run: bool = False) -> int:
    """Scan watcher_requests/ for pending Chapter 13 request files.

    Called by the daemon loop on each polling cycle.

    Args:
        dry_run: If True, validate without executing.

    Returns:
        Number of requests processed.
    """
    project_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".."
    )
    requests_dir = os.path.join(project_root, "watcher_requests")

    if not os.path.isdir(requests_dir):
        return 0

    # Sorted for deterministic processing order
    request_files = sorted(glob.glob(os.path.join(requests_dir, "*.json")))
    # Exclude archive subdirectory
    request_files = [f for f in request_files
                     if os.sep + "archive" + os.sep not in f]

    if not request_files:
        return 0

    logger.info(f"Found {len(request_files)} pending request(s)")
    processed = 0
    for request_file in request_files:
        status = process_chapter_13_request(self, request_file,
                                            dry_run=dry_run)
        processed += 1
        logger.info(f"Request {os.path.basename(request_file)}: {status}")

    return processed


# ===========================================================================
# Support Methods
# ===========================================================================

def _validate_selfplay_request(self, request: dict) -> bool:
    """Validate a selfplay request has required fields and sane values."""
    episodes = request.get("episodes", 5)
    if not isinstance(episodes, int) or not (1 <= episodes <= 50):
        logger.warning(f"Invalid episodes count: {episodes} (must be 1–50)")
        return False

    survivors_file = request.get("survivors_file",
                                 "survivors_with_scores.json")
    # Resolve relative to project root
    project_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".."
    )
    full_path = os.path.join(project_root, survivors_file)
    if not os.path.exists(full_path):
        logger.warning(f"Survivors file not found: {full_path}")
        return False

    return True


def _evaluate_step_via_bundle(self, prompt: str,
                              grammar_name: str) -> dict:
    """Evaluate using a bundle-factory-produced prompt and grammar.

    Bridges the bundle_factory output format with the existing
    LLM evaluation infrastructure.

    Fallback chain:
        1. LLM router with grammar constraint
        2. HTTP direct call with grammar
        3. Heuristic default (proceed, conf=0.50)
    """
    # ── Try 1: Grammar-constrained via LLM router ────────────────
    # Router public API only supports watcher_decision.gbnf.
    # For all other grammars, fall through to Try 2 (HTTP direct).
    if (hasattr(self, 'llm_router') and self.llm_router
            and getattr(self.config, 'use_grammar', True)
            and grammar_name == 'watcher_decision.gbnf'):
        try:
            result = self.llm_router.evaluate_watcher_decision(prompt)
            return result
        except Exception as e:
            logger.warning(f"LLM router evaluation failed: {e}")

    # ── Try 2: HTTP direct with grammar ──────────────────────────
    try:
        import requests as http_requests
        llm_port = getattr(self.config, 'llm_port', 8080)
        llm_url = f"http://localhost:{llm_port}/completion"
        payload = {
            "prompt": prompt,
            "n_predict": 512,
            "temperature": 0.1,
        }
        # Attach grammar if available
        grammar_path = _resolve_grammar_path(grammar_name)
        if grammar_path and os.path.exists(grammar_path):
            with open(grammar_path) as gf:
                payload["grammar"] = gf.read()

        resp = http_requests.post(llm_url, json=payload, timeout=60)
        if resp.status_code == 200:
            content = resp.json().get("content", "")
            return _parse_llm_response(content)
    except Exception as e:
        logger.warning(f"HTTP LLM evaluation failed: {e}")

    # ── Fallback: proceed with low confidence ────────────────────
    return {
        "decision": "proceed",
        "confidence": 0.50,
        "reasoning": "LLM evaluation unavailable — heuristic default",
    }


def _resolve_grammar_path(grammar_name: str) -> Optional[str]:
    """Resolve a grammar name to its .gbnf file path."""
    if not grammar_name:
        return None
    try:
        from llm_services.grammar_loader import GrammarLoader
        loader = GrammarLoader()
        return loader.get_grammar_path(grammar_name)
    except Exception:
        # Manual fallback
        project_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".."
        )
        candidates = [
            os.path.join(project_root, "agent_grammars",
                         f"{grammar_name}.gbnf"),
            os.path.join(project_root, "agent_grammars", grammar_name),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
    return None


def _parse_llm_response(response) -> dict:
    """Parse an LLM response (string or dict) into a decision dict."""
    if isinstance(response, dict):
        return response
    if isinstance(response, str):
        # Strip markdown fences if present
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
    return {
        "decision": "proceed",
        "confidence": 0.50,
        "reasoning": f"Could not parse LLM response: {str(response)[:100]}",
    }


def _archive_request(self, request_path: str, status: str):
    """Move a processed request file to watcher_requests/archive/."""
    archive_dir = os.path.join(os.path.dirname(request_path), "archive")
    os.makedirs(archive_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    basename = os.path.basename(request_path)
    archive_name = f"{status}_{timestamp}_{basename}"
    archive_path = os.path.join(archive_dir, archive_name)

    try:
        shutil.move(request_path, archive_path)
        logger.info(f"Archived request → {archive_path}")
    except Exception as e:
        logger.warning(f"Failed to archive request: {e}")


def _log_dispatch_decision(self, run_id: str, dispatch_type: str,
                           status: str, details: dict):
    """Append a dispatch decision to watcher_decisions.jsonl."""
    decision = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "dispatch_type": dispatch_type,
        "status": status,
        "details": details,
    }

    try:
        decisions_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..",
            "watcher_decisions.jsonl"
        )
        with open(decisions_path, "a") as f:
            f.write(json.dumps(decision) + "\n")
    except Exception as e:
        logger.warning(f"Failed to log dispatch decision: {e}")


# ===========================================================================
# Method Binding
# ===========================================================================

def bind_to_watcher(watcher_cls):
    """Bind all dispatch methods to WatcherAgent class.

    Called once at import time.  Does NOT override existing methods.

    Usage:
        from agents.watcher_dispatch import bind_to_watcher
        bind_to_watcher(WatcherAgent)
    """
    methods = {
        "dispatch_selfplay": dispatch_selfplay,
        "dispatch_learning_loop": dispatch_learning_loop,
        "process_chapter_13_request": process_chapter_13_request,
        "_scan_watcher_requests": _scan_watcher_requests,
        "_validate_selfplay_request": _validate_selfplay_request,
        "_evaluate_step_via_bundle": _evaluate_step_via_bundle,
        "_archive_request": _archive_request,
        "_log_dispatch_decision": _log_dispatch_decision,
    }

    bound = []
    skipped = []
    for name, func in methods.items():
        if hasattr(watcher_cls, name):
            skipped.append(name)
        else:
            setattr(watcher_cls, name, func)
            bound.append(name)

    if bound:
        logger.info(f"Dispatch methods bound to {watcher_cls.__name__}: "
                     f"{', '.join(bound)}")
    if skipped:
        logger.warning(f"Skipped (already exist): {', '.join(skipped)}")

    return bound


# ===========================================================================
# Standalone CLI (for independent testing)
# ===========================================================================

def _standalone_cli():
    """Standalone CLI for testing dispatch without modifying watcher_agent.py.

    Usage:
        cd ~/distributed_prng_analysis
        PYTHONPATH=. python3 agents/watcher_dispatch.py --help
        PYTHONPATH=. python3 agents/watcher_dispatch.py --dispatch-selfplay --dry-run
        PYTHONPATH=. python3 agents/watcher_dispatch.py --dispatch-learning-loop steps_3_5_6 --dry-run
        PYTHONPATH=. python3 agents/watcher_dispatch.py --process-requests --dry-run
        PYTHONPATH=. python3 agents/watcher_dispatch.py --self-test
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="WATCHER Dispatch — Phase 7 Part B (standalone)"
    )
    parser.add_argument("--dispatch-selfplay", action="store_true",
                        help="Dispatch selfplay orchestrator")
    parser.add_argument("--dispatch-learning-loop", type=str, nargs="?",
                        const="steps_3_5_6", metavar="SCOPE",
                        help="Dispatch learning loop "
                             "(default: steps_3_5_6)")
    parser.add_argument("--process-requests", action="store_true",
                        help="Process pending watcher_requests/*.json")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Selfplay episodes (default: 5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log actions without executing")
    parser.add_argument("--self-test", action="store_true",
                        help="Run self-test (verify imports + binding)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if args.self_test:
        _run_self_test()
        return

    # Import and bind to WatcherAgent
    try:
        from agents.watcher_agent import WatcherAgent, WatcherConfig
        bind_to_watcher(WatcherAgent)
    except ImportError as e:
        logger.error(f"Cannot import WatcherAgent: {e}")
        logger.error("Run from project root with PYTHONPATH=.")
        sys.exit(1)

    # Create a minimal watcher instance
    config = WatcherConfig(use_llm=False)
    watcher = WatcherAgent(config)

    if args.dispatch_selfplay:
        request = {"episodes": args.episodes}
        ok = watcher.dispatch_selfplay(request, dry_run=args.dry_run)
        print(f"dispatch_selfplay: {'OK' if ok else 'FAILED'}")

    elif args.dispatch_learning_loop:
        ok = watcher.dispatch_learning_loop(
            scope=args.dispatch_learning_loop, dry_run=args.dry_run
        )
        print(f"dispatch_learning_loop: {'OK' if ok else 'FAILED'}")

    elif args.process_requests:
        count = watcher._scan_watcher_requests(dry_run=args.dry_run)
        print(f"Processed {count} request(s)")

    else:
        parser.print_help()


def _run_self_test():
    """Verify imports and binding work correctly."""
    print("=" * 60)
    print("WATCHER Dispatch — Self-Test")
    print("=" * 60)

    # Test 1: Module imports
    print("\n[Test 1] Module imports... ", end="")
    try:
        _get_build_llm_context()
        print("✅ build_llm_context imported")
    except ImportError as e:
        print(f"⚠️  build_llm_context not available: {e}")
        print("         (Expected if bundle_factory not yet deployed)")

    # Test 2: Method binding
    print("[Test 2] Method binding... ", end="")
    try:
        from agents.watcher_agent import WatcherAgent
        bound = bind_to_watcher(WatcherAgent)
        print(f"✅ {len(bound)} methods bound")
    except ImportError:
        print("⚠️  WatcherAgent not importable (run with PYTHONPATH=.)")

    # Test 3: Request directory
    print("[Test 3] watcher_requests/ directory... ", end="")
    project_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".."
    )
    requests_dir = os.path.join(project_root, "watcher_requests")
    if os.path.isdir(requests_dir):
        pending = glob.glob(os.path.join(requests_dir, "*.json"))
        pending = [f for f in pending if "archive" not in f]
        print(f"✅ exists ({len(pending)} pending)")
    else:
        print(f"⚠️  not found — creating")
        os.makedirs(requests_dir, exist_ok=True)
        print(f"         Created: {requests_dir}")

    # Test 4: Grammar resolution
    print("[Test 4] Grammar resolution... ", end="")
    grammar = _resolve_grammar_path("agent_decision")
    if grammar and os.path.exists(grammar):
        print(f"✅ {grammar}")
    else:
        print(f"⚠️  agent_decision.gbnf not found")

    # Test 5: Standalone functions are callable
    print("[Test 5] Function signatures... ", end="")
    required = [dispatch_selfplay, dispatch_learning_loop,
                process_chapter_13_request, _scan_watcher_requests]
    all_ok = all(callable(f) for f in required)
    print(f"✅ {len(required)} dispatch functions verified" if all_ok
          else "❌ Some functions missing")

    print("\n" + "=" * 60)
    print("Self-test complete")
    print("=" * 60)


if __name__ == "__main__":
    _standalone_cli()
