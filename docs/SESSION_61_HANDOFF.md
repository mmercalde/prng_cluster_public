# Session 61 Handoff — Soak C + Strategy Advisor

## Project Context
Distributed PRNG analysis system using functional mimicry (learning output patterns, not seed discovery). 26-GPU cluster (Zeus 2× RTX 3080 Ti, rig-6600 8× RX 6600, rig-6600b 8× RX 6600, rig-6600c 8× RX 6600). 6-step pipeline with Chapter 13 live feedback loop and WATCHER autonomous agent.

## Session 60 Completed
- ✅ bundle_factory v1.1.0 deployed (MAIN_MISSION + step_id=99 selfplay)
- ✅ Soak C gaps documented (acceptance engine didn't honor test_mode flags)
- ✅ Patches v1.1.1 applied to `chapter_13_acceptance.py`
- ✅ Git commit `dfdffd9`

## Immediate Task: Run Soak C

**Terminal 1:**
```bash
cd ~/distributed_prng_analysis
PYTHONPATH=. python3 synthetic_draw_injector.py --daemon --interval 60
```

**Terminal 2:**
```bash
cd ~/distributed_prng_analysis
mkdir -p logs/soak
PYTHONPATH=. python3 chapter_13_orchestrator.py --daemon --auto-start-llm |& tee logs/soak/soakC_$(date +%Y%m%d_%H%M%S).log
```

**Let run 1-2 hours. Verify with:**
```bash
echo "Cycles: $(grep -c 'CHAPTER 13 CYCLE' logs/soak/soakC_*.log)"
echo "Auto-approved: $(grep -c 'SOAK C: Auto-approving' logs/soak/soakC_*.log)"
echo "Escalated: $(grep -c 'pending_approval' logs/soak/soakC_*.log)"
```

**Pass criteria:** Cycles > 10, Escalated = 0

## After Soak C Pass: Strategy Advisor Implementation

Per `CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md` and `PROGRESS_BUNDLE_FACTORY_AND_STRATEGY_ADVISOR_v1_0.md`:

1. Create `strategy_advisor.gbnf` (~80 lines)
2. Add `build_advisor_bundle()` to `bundle_factory.py` (~100 lines)
3. Create `parameter_advisor.py` (~400 lines)
4. Add `dispatch_strategy_advisor()` to WATCHER

**Activation gate:** ≥15 real draws, ≥10 selfplay episodes, ≥1 promoted policy, Soak A/B/C complete

## Key Files
- `chapter_13_acceptance.py` — patched with Soak C v1.1.1
- `patch_soak_c_integration_v1.py` — revert with `--revert` if needed
- `chapter_13_acceptance.py.pre_soakc_patch` — backup
- `CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md` — Strategy Advisor spec
- `CHAPTER_14_TRAINING_DIAGNOSTICS.md` — PyTorch dynamic graph diagnostics
- `SOAK_C_GAPS_AND_PATCHES_v1_0.md` — gap analysis

## Infrastructure Status

### ✅ EXISTS (Built Sessions 57-59)

| Component | File | Status |
|-----------|------|--------|
| LLM Router | `llm_router.py` | ✅ Exists |
| LLM Lifecycle | `llm_services/llm_lifecycle.py` | ✅ ~8KB |
| Bundle Factory | `agents/contexts/bundle_factory.py` | ✅ ~32KB, v1.1.0 |
| WATCHER Dispatch | `agents/watcher_dispatch.py` | ✅ ~30KB |
| Grammar Loader | `grammar_loader.py` | ✅ Exists |
| Base Grammars | `agent_grammars/*.gbnf` | ✅ agent_decision, sieve_analysis, chapter_13 |

### ❌ NEEDS CREATION — Strategy Advisor (Task 2)

**Proposal:** `CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md`

| File | Size Est. | Exists? |
|------|-----------|---------|
| `parameter_advisor.py` | ~400 lines | ❌ NO |
| `strategy_advisor.gbnf` | ~80 lines | ❌ NO |
| `build_advisor_bundle()` in bundle_factory | ~150 lines | ❌ NO |
| `dispatch_strategy_advisor()` in watcher_dispatch | ~50 lines | ❌ NO |

### ❌ NEEDS CREATION — Chapter 14 Training Diagnostics (Task 3)

**Proposal:** `CHAPTER_14_TRAINING_DIAGNOSTICS.md`

| File | Size Est. | Exists? |
|------|-----------|---------|
| `training_diagnostics.py` | ~400 lines | ❌ NO |
| `per_survivor_attribution.py` | ~200 lines | ❌ NO |
| `diagnostics_analysis.gbnf` | ~70 lines | ❌ NO |
| `build_diagnostics_bundle()` in bundle_factory | ~100 lines | ❌ NO |

**The plumbing is built. The proposals define new components that plug into it.**

## Soak Status
- Soak A: ✅ Complete
- Soak B: ✅ Complete + Certified  
- Soak C: ⏳ Ready to run (patches applied)

## Post-Soak C Cleanup
```bash
# Restore production configs
cp lottery_history.json.pre_soakC lottery_history.json
cp optimal_window_config.json.pre_soakC optimal_window_config.json
cp watcher_policies.json.pre_soakC watcher_policies.json

# Optionally revert patches (or keep for future testing)
python3 patch_soak_c_integration_v1.py --revert
```

## SSH Shortcuts
- `rzeus` — Zeus primary node
- `rr1` — rig-6600
- `rr2` — rig-6600b
- `rr3` — rig-6600c

---

**Resume:** 
1. Run Soak C first (validates Chapter 13 Live Feedback autonomy)
2. If pass → certify Phase 7, proceed to Strategy Advisor
3. After Strategy Advisor → Chapter 14 Training Diagnostics (PyTorch dynamic graphs)
