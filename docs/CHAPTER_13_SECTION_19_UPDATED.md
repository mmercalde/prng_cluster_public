# CHAPTER 13 — Section 19 (UPDATED)

**Last Verified:** 2026-01-30  
**Status:** Phases 1-6 COMPLETE, Phase 7 Testing In Progress

---

## 19. Implementation Checklist

### Phase 1: Draw Ingestion ✅ COMPLETE (2026-01-12)

- [x] `draw_ingestion_daemon.py` — Monitors for new draws (22KB)
- [x] `synthetic_draw_injector.py` — Test mode draw generation (20KB)
  - Reads PRNG type from `optimal_window_config.json` (no hardcoding)
  - Uses `prng_registry.py` (same as Steps 1-6)
  - Modes: manual (`--inject-one`), daemon (`--daemon --interval 60`), flag-triggered
- [x] Append-only history updates
- [x] Fingerprint change detection
- [x] `watcher_policies.json` — Includes test_mode and synthetic_injection settings (4.7KB, updated Jan 29)

### Phase 2: Diagnostics Engine ✅ COMPLETE (2026-01-12, updated 2026-01-29)

- [x] `chapter_13_diagnostics.py` — Core diagnostic generator (39KB)
- [x] Prediction vs reality comparison
- [x] Confidence calibration metrics
- [x] Survivor performance tracking
- [x] Feature drift detection
- [x] Generate `post_draw_diagnostics.json`
- [x] Create `diagnostics_history/` archival

### Phase 3: LLM Integration ✅ COMPLETE (2026-01-12)

- [x] `chapter_13_llm_advisor.py` — LLM analysis module (23KB)
- [x] `llm_proposal_schema.py` — Pydantic model for proposals (14KB)
- [x] `chapter_13.gbnf` — Grammar constraint (2.9KB)
- [x] System/user prompt templates
- [x] Integration with existing LLM infrastructure

### Phase 4: WATCHER Policies ✅ COMPLETE (2026-01-12, updated 2026-01-29)

- [x] `chapter_13_acceptance.py` — Acceptance/rejection rules (41KB)
- [x] `chapter_13_triggers.py` — Retrain trigger thresholds (36KB)
- [x] Cooldown enforcement
- [x] Escalation handlers

### Phase 5: Orchestration ✅ COMPLETE (2026-01-12)

- [x] `chapter_13_orchestrator.py` — Main orchestrator (23KB)
- [x] Partial rerun logic (Steps 3→5→6)
- [x] Full rerun trigger (Steps 1→6)
- [x] Decision logging
- [x] Audit trail

### Phase 6: Testing 🟡 IN PROGRESS

- [x] Synthetic draw injection (module exists)
- [x] Proposal validation tests (in acceptance.py)
- [ ] End-to-end convergence monitoring
- [ ] Divergence detection tests
- [ ] Live integration testing

### Phase 7: WATCHER Integration ❌ NOT COMPLETE

**This is the actual gap preventing full autonomy.**

- [ ] `dispatch_selfplay()` in `watcher_agent.py`
- [ ] `dispatch_learning_loop()` in `watcher_agent.py`
- [ ] Wire Chapter 13 orchestrator into WATCHER daemon
- [ ] Move `chapter_13.gbnf` to `agent_grammars/` directory
- [ ] Integration tests: WATCHER → Chapter 13 → Selfplay

---

## Files Summary (Verified 2026-01-30)

| File | Size | Phase | Status |
|------|------|-------|--------|
| `draw_ingestion_daemon.py` | 22KB | 1 | ✅ |
| `synthetic_draw_injector.py` | 20KB | 1 | ✅ |
| `watcher_policies.json` | 4.7KB | 1,4 | ✅ |
| `chapter_13_diagnostics.py` | 39KB | 2 | ✅ |
| `chapter_13_llm_advisor.py` | 23KB | 3 | ✅ |
| `llm_proposal_schema.py` | 14KB | 3 | ✅ |
| `chapter_13.gbnf` | 2.9KB | 3 | ✅ |
| `chapter_13_acceptance.py` | 41KB | 4,5 | ✅ |
| `chapter_13_triggers.py` | 36KB | 4 | ✅ |
| `chapter_13_orchestrator.py` | 23KB | 5 | ✅ |

**Total Chapter 13 Code:** ~235KB across 10 files

---

## What Remains for Full Autonomy

```
Chapter 13 Components          WATCHER                      Selfplay
─────────────────────          ───────                      ────────
✅ diagnostics.py              ❌ dispatch_selfplay()       ✅ orchestrator.py
✅ llm_advisor.py              ❌ dispatch_learning_loop()  ✅ policy_transform.py
✅ triggers.py                 ❌ Chapter 13 daemon wire    ✅ policy_conditioned.py
✅ acceptance.py               ✅ Pipeline Steps 1-6        ✅ inner_episode_trainer.py
✅ orchestrator.py             ✅ Request validation        ✅ telemetry
```

**Gap:** WATCHER can run Steps 1-6, but cannot yet dispatch Chapter 13 or Selfplay.

---

*This section replaces the original Section 19 in CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md*
