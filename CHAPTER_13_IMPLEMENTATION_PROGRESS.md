# Chapter 13 Implementation Progress

**Last Updated:** 2026-01-11  
**Document Version:** 1.1.0  
**Status:** Planning Complete → Implementation Phase 1

---

## Overall Progress

| Phase | Status | Owner | Target |
|-------|--------|-------|--------|
| 1. Draw Ingestion | 🔲 Not Started | TBD | Week 1 |
| 2. Diagnostics Engine | 🔲 Not Started | TBD | Week 1-2 |
| 3. Retrain Triggers | 🔲 Not Started | TBD | Week 2 |
| 4. LLM Integration | 🔲 Not Started | TBD | Week 3 |
| 5. Acceptance Engine | 🔲 Not Started | TBD | Week 3 |
| 6. WATCHER Orchestration | 🔲 Not Started | TBD | Week 4 |
| 7. Testing & Validation | 🔲 Not Started | TBD | Week 4 |

**Legend:** 🔲 Not Started | 🟡 In Progress | ✅ Complete | ❌ Blocked

---

## Phase 1: Draw Ingestion

| Task | Status | Notes |
|------|--------|-------|
| Create `draw_ingestion_daemon.py` | 🔲 | Monitors for new draws |
| Create `synthetic_draw_injector.py` | 🔲 | Test mode, reads PRNG from config |
| Create `watcher_policies.json` | 🔲 | Thresholds & test_mode settings |
| Append-only history update logic | 🔲 | Updates `lottery_history.json` |
| Fingerprint change detection | 🔲 | Triggers Chapter 13 |
| Test: Manual injection | 🔲 | `--inject-one` mode |
| Test: Daemon injection | 🔲 | `--daemon` mode |

**Blockers:** None  
**Notes:** 
- PRNG type inherited from `optimal_window_config.json` (no hardcoding)
- Uses `prng_registry.py` (same as Steps 1-6)

---

## Phase 2: Diagnostics Engine

| Task | Status | Notes |
|------|--------|-------|
| Create `chapter_13_diagnostics.py` | 🔲 | Core diagnostic generator |
| Prediction vs reality comparison | 🔲 | Hit rate, rank, distance |
| Confidence calibration metrics | 🔲 | Predicted vs actual correlation |
| Survivor performance tracking | 🔲 | Hit/decay/reinforce candidates |
| Feature drift detection | 🔲 | Entropy, turnover |
| Generate `post_draw_diagnostics.json` | 🔲 | Output artifact |
| Create `diagnostics_history/` archival | 🔲 | Historical storage |
| Test: Diagnostic accuracy | 🔲 | Validate metrics |

**Blockers:** None  
**Notes:**

---

## Phase 3: Retrain Trigger Logic

| Task | Status | Notes |
|------|--------|-------|
| Define thresholds in `watcher_policies.json` | 🔲 | N draws, drift, misses |
| Add `should_retrain()` to WATCHER | 🔲 | Evaluates triggers |
| Add `execute_learning_loop()` to WATCHER | 🔲 | Runs Steps 3→5→6 |
| Implement partial rerun logic | 🔲 | Selective step execution |
| Implement cooldown enforcement | 🔲 | Prevent thrashing |
| Human approval gate | 🔲 | Required for v1 |
| Test: Trigger conditions | 🔲 | Each threshold |

**Blockers:** None  
**Notes:** v1 requires human approval for trigger execution. Autonomous execution is deferred.

---

## Phase 4: LLM Integration

| Task | Status | Notes |
|------|--------|-------|
| Create `chapter_13_llm_advisor.py` | 🔲 | LLM analysis module |
| Create `llm_proposal_schema.py` | 🔲 | Pydantic model |
| Create `chapter_13.gbnf` | 🔲 | Grammar constraint |
| System prompt template | 🔲 | Strategist role |
| User prompt template | 🔲 | Diagnostic analysis |
| Integration with existing LLM infra | 🔲 | Qwen2.5 backend |
| Test: Proposal generation | 🔲 | Valid schema output |

**Blockers:** None  
**Notes:**

---

## Phase 5: Acceptance/Rejection Engine

| Task | Status | Notes |
|------|--------|-------|
| Implement `validate_proposal()` | 🔲 | In WATCHER |
| Enforce 30% max delta | 🔲 | Bounds checking |
| Enforce 3 param max | 🔲 | Limit scope |
| Enforce cooldown periods | 🔲 | Time-based gating |
| Escalation logic | 🔲 | Human review triggers |
| Create `watcher_decision_log.json` | 🔲 | Audit trail |
| Test: Rejection conditions | 🔲 | Bounds, cooldowns |
| Test: Acceptance conditions | 🔲 | Valid proposals |

**Blockers:** None  
**Notes:**

---

## Phase 6: WATCHER Orchestration

| Task | Status | Notes |
|------|--------|-------|
| Enhance `--daemon` mode | 🔲 | Event-driven loop |
| Wait for `new_draw.flag` | 🔲 | Trigger detection |
| Run diagnostics on trigger | 🔲 | Chapter 13 flow |
| Query LLM (optional) | 🔲 | Advisory analysis |
| Execute approved reruns | 🔲 | Steps 3→5→6 |
| Clear flag, repeat | 🔲 | Loop closure |
| Test: Full autonomous cycle | 🔲 | End-to-end |

**Blockers:** None  
**Notes:**

---

## Phase 7: Testing & Validation

| Task | Status | Notes |
|------|--------|-------|
| Synthetic draw convergence test | 🔲 | True seed rises in rankings |
| Forced retrain validation | 🔲 | Steps 3→5→6 execute |
| Proposal rejection test | 🔲 | Bounds enforced |
| Divergence detection test | 🔲 | Halt on instability |
| Cooldown enforcement test | 🔲 | Thrashing prevented |
| Full autonomy test (100 draws) | 🔲 | Extended run |

**Blockers:** Phase 1-6 complete  
**Notes:**

---

## Convergence Metrics (Test Mode)

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Hit Rate (Top-20) | > 5% | - | 🔲 |
| Confidence Calibration | Correlation > 0.3 | - | 🔲 |
| Confidence trend | Increasing | - | 🔲 |
| Hit rate | >0.05 | - | 🔲 |

---

## Files Created/Modified

| File | Status | Purpose |
|------|--------|---------|
| `draw_ingestion_daemon.py` | 🔲 | Draw monitoring |
| `synthetic_draw_injector.py` | 🔲 | Test mode |
| `chapter_13_diagnostics.py` | 🔲 | Diagnostic engine |
| `chapter_13_llm_advisor.py` | 🔲 | LLM integration |
| `llm_proposal_schema.py` | 🔲 | Pydantic models |
| `chapter_13.gbnf` | 🔲 | Grammar constraint |
| `watcher_policies.json` | 🔲 | Config |
| `agents/watcher_agent.py` | 🔲 Modified | Orchestration |

---

## Deferred Extensions (Future Work)

These are **not required** for v1. They will be implemented after core Chapter 13 stability is proven.

| Extension | Description | Status | Depends On |
|-----------|-------------|--------|------------|
| #1 | Step-6 Backtesting Hooks | 🔲 Deferred | v1 stable |
| #2 | Confidence Calibration Curves (rolling) | 🔲 Deferred | v1 stable |
| #3 | Autonomous Trigger Execution (no human approval) | 🔲 Deferred | v1 stable |
| #4 | Convergence Dashboards | 🔲 Deferred | v1 stable |

### v1 vs Deferred Execution Clarification

| Aspect | v1 (Current) | Extension #3 (Deferred) |
|--------|--------------|-------------------------|
| Trigger definition | ✅ Implemented | No change |
| Trigger evaluation | ✅ Implemented | No change |
| Trigger execution | ⚠️ **Requires human approval** | Fully autonomous |
| Human-in-the-loop | Required | Optional |

---

## Commits

| Date | Hash | Description |
|------|------|-------------|
| 2026-01-11 | 263ebec | docs: Add Chapter 13 - Live Feedback Loop & Implementation Progress |
| - | - | - |

---

## Blockers & Issues

| Issue | Status | Resolution |
|-------|--------|------------|
| None | - | - |

---

## Notes & Decisions

- **2026-01-11:** Chapter 13 spec finalized. Team Alpha + Beta aligned.
- **2026-01-11:** Synthetic injection uses config-based PRNG (no hardcoding).
- **2026-01-11:** Test mode requires dual flags: `test_mode` AND `synthetic_injection.enabled`.
- **2026-01-11:** v1 trigger execution requires human approval (Extension #3 deferred).
- **2026-01-11:** Added deferred extensions roadmap (Team Beta proposals).

---

## Next Actions

1. [ ] Begin Phase 1: Create `synthetic_draw_injector.py`
2. [ ] Begin Phase 1: Create `watcher_policies.json`
3. [ ] Begin Phase 1: Create `draw_ingestion_daemon.py`

---

*Update this document as implementation progresses.*
