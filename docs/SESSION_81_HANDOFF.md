# Session 81 Handoff — Chapter 14 Remaining Phases + Real Learning Cycle

## System State
- **Git:** Clean, all pushed. Latest commit `d2306f5` on main.
- **WATCHER:** v2.0.0 (2,795 lines), daemon validated via Soak C v2.0.0 (69 cycles, 46 min, 0 tracebacks)
- **Production mode:** test_mode=false, approval_route=orchestrator
- **All 26 GPUs available:** Zeus (2× 3080 Ti), rig-6600/6600b/6600c (8× RX 6600 each). All rigs rebooted with GFXOFF disabled + udev perf=high.
- **Documentation:** Fully synced across Zeus, ser8, Claude project. Operating Guide v2.0, Doc Index v1.1, Progress v3.7.

## Session 80 Completed
- `_pipeline_running` lifecycle fix (daemon survives pipeline completion)
- `approval_route` policy (orchestrator vs watcher execution authority)
- Soak C v2.0.0 PASSED — full daemon execution path validated
- GPU rig permanent fixes (udev + GFXOFF) deployed and rebooted
- Comprehensive documentation sweep (6 docs updated/created)

## Chapter 14 Status — Phases 1-6 COMPLETE

| Phase | Status | Key Files |
|---|---|---|
| 1. Core Diagnostics | ✅ S69 | `training_diagnostics.py` (~995 lines) |
| 2. GPU/CPU Collection | ✅ S70 | CUDA + ROCm metrics |
| 3. Engine Wiring | ✅ S70 | `reinforcement_engine.py` v1.7.0 (1168 lines) |
| 4. RETRY Param-Threading | ✅ S76 | WATCHER health check integration |
| 5. FIFO Pruning | ✅ S72 | Unbounded growth prevention |
| 6. Health Check | ✅ S72 | `check_training_health()` deployed |

## Session 81 Tasks — Three Remaining Ch14 Phases

### Priority 1: Phase 7 — LLM Diagnostics Integration (~2 hours)

**Goal:** When WATCHER hits RETRY/SKIP_MODEL during Step 5, ask DeepSeek WHY training degraded and get structured recommendations instead of retrying blindly.

**Build:**
1. `diagnostics_analysis.gbnf` — grammar constraint for DeepSeek diagnostic response
2. `diagnostics_analysis_schema.py` — Pydantic model for validation
3. `build_diagnostics_bundle()` in `agents/contexts/bundle_factory.py` — assembles training health JSON + history into LLM prompt
4. `request_llm_diagnostics_analysis()` — calls DeepSeek, returns grammar-constrained recommendation

**Pattern:** Identical to Strategy Advisor (parameter_advisor.py + advisor_bundle.py + strategy_advisor.gbnf). Follow that existing pattern.

**Integration point:** `_build_retry_params()` in watcher_agent.py currently uses hardcoded fallback logic. After Phase 7, it consults LLM analysis first.

### Priority 2: Real Learning Cycle

**Goal:** Run full pipeline through WATCHER with actual GPU training (not freshness-skip). This also serves as Phase 9 (first diagnostic investigation).

**Steps:**
1. Invalidate stale Step 3+ outputs (force recomputation)
2. Set approval_route=watcher, test_mode=true
3. Start daemon + orchestrator + injector
4. Watch actual GPU training execute on rigs
5. Examine `diagnostics_outputs/*.json` — calibrate health check thresholds
6. Restore production mode

### Priority 3: Phase 8 — Selfplay + Ch13 Wiring (~1.5 hours, if time)

**Goal:** Wire diagnostics into selfplay episodes (cross-episode trend detection) and Chapter 13 root cause analysis (correlate prediction degradation with training health).

**Dependency:** Selfplay not in active daily use yet, so this is lower priority.

## Key Paths on Zeus
- `~/distributed_prng_analysis/` — project root
- `agents/watcher_agent.py` — WATCHER v2.0.0
- `training_diagnostics.py` — Ch14 core module
- `reinforcement_engine.py` — v1.7.0 with diagnostics hooks
- `agents/contexts/bundle_factory.py` — LLM context assembly (7 bundle types, will be 8)
- `parameter_advisor.py` — Strategy Advisor (reference pattern for Phase 7)
- `agents/contexts/advisor_bundle.py` — Advisor bundle (reference pattern)
- `grammars/strategy_advisor.gbnf` — Advisor grammar (reference pattern)
- `watcher_policies.json` — production mode restored
- `docs/CHAPTER_14_TRAINING_DIAGNOSTICS.md` — full spec (Phases 7-9 in Sections 13.8-13.9)

## Other Deferred Items (NOT Session 81)
- Phase 9B.3: Auto policy heuristics
- Chunk 3: APScheduler + scraper subprocess
- P1: `_is_within_policy_bounds()` whitelist
- P2: Strategy Advisor GBNF/prompt expansion
- Autonomous Optuna trial type recommendation

## Rules
1. NEVER restore from backup — edit/remove bad additions
2. Sync progress tracker + chapter checklist in same session
3. Create SESSION_CHANGELOG every chat
4. VERIFY bugs before fixing
5. Documentation goes in `docs/` subdirectory on Zeus
6. SSH alias: `rzeus` (from ser8), NOT `zeus`
7. Files download to ser8 `~/Downloads/`, then `scp` to Zeus
