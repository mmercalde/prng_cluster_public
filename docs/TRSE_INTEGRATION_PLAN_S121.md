# TRSE Integration Plan — S121
**Date:** 2026-03-07  
**Status:** Step 0 standalone complete. Wiring into pipeline NOT yet done.

---

## 1. What Is Installed Right Now

### `trse_step0.py` (commit `c33b125`)
- Standalone script — runs independently, no pipeline integration yet
- Reads `daily3.json` → computes 354 windows (W=400, S=50) → K=5 KMeans clusters
- Writes `trse_context.json` with: `current_regime`, `regime_age`, `regime_stable`,
  `silhouette`, `switch_rate`, `regime_entropy_profile`, `current_window_features`
- Has freshness check — skips if `trse_context.json` newer than `daily3.json`
- CLI: `python3 trse_step0.py --lottery-data daily3.json --output trse_context.json`
- **Verified on real data:** regime=0, age=5, stable=True, silhouette=0.047, 354 windows

### `trse_context.json` (gitignored, present on Zeus)
- Baseline snapshot from first real run
- Regenerated automatically on each Step 0 run

---

## 2. What Needs To Be Installed

### A. Wire Step 0 into WATCHER (3 changes to `watcher_agent.py`)

WATCHER builds step commands entirely from two registries:
```python
STEP_SCRIPTS   = {1: "window_optimizer.py", 2: ..., ...}
STEP_NAMES     = {1: "Window Optimizer", ...}
STEP_MANIFESTS = {1: "window_optimizer.json", ...}
```
Step 0 doesn't exist in any of these. Three lines needed:

```python
STEP_SCRIPTS[0]   = "trse_step0.py"
STEP_NAMES[0]     = "Regime Segmentation (TRSE)"
STEP_MANIFESTS[0] = "trse.json"
```

Also add timeout override (TRSE is fast ~2s, but be explicit):
```python
step_timeout_overrides = {0: 5, 1: 480, 5: 360}
```

### B. Create `agent_manifests/trse.json`

WATCHER reads this to know: what params to pass, what output to check for freshness.

```json
{
  "agent_name": "trse_agent",
  "description": "Step 0: Temporal Regime Segmentation — characterises current draw regime before window optimization.",
  "pipeline_step": 0,
  "version": "1.0.0",
  "required_inputs": ["daily3.json"],
  "primary_output": "trse_context.json",
  "evaluation_type": "file_exists",
  "disable_llm_parsing": true,
  "disable_heuristic_parsing": true,
  "retry_policy": "none",
  "default_params": {
    "lottery_data": "daily3.json",
    "output": "trse_context.json",
    "window_size": 400,
    "stride": 50,
    "k_clusters": 5,
    "recommended_window_size": 8
  },
  "arg_style": "named"
}
```

### C. Wire `trse_context.json` → `window_optimizer_bayesian.py` SearchBounds

**Approach: PASSIVE** — Step 1 reads `trse_context.json` on its own if present.
WATCHER doesn't need to parse or inject anything. This is the clean separation:

- Step 0 writes `trse_context.json`
- Step 1 reads it if it exists → narrows `SearchBounds`
- If absent → Step 1 runs with full default bounds (backward compatible)

**Touch point in `window_optimizer_bayesian.py` — `OptunaBayesianSearch.search()`**
Before the study is created (~line 284), add:

```python
# [S121] TRSE: Load regime context and narrow search bounds if available
trse_ctx = _load_trse_context("trse_context.json")  # returns None if absent
if trse_ctx and trse_ctx.get("regime_stable"):
    rec_ws = trse_ctx.get("recommended_window_size", bounds.max_window_size)
    # Bias upper bound toward regime-optimal window size
    # Don't go below current min — just narrow the ceiling
    new_max = max(bounds.min_window_size + 1, min(rec_ws * 4, bounds.max_window_size))
    print(f"[TRSE] Regime {trse_ctx['current_regime']} stable "
          f"(age={trse_ctx['regime_age']}) → "
          f"window_size ceiling: {bounds.max_window_size} → {new_max}")
    bounds = bounds._replace(max_window_size=new_max)
```

**Touch point in `window_optimizer.py`** — add `--trse-context` CLI arg (optional,
defaults to `trse_context.json`):
```python
parser.add_argument('--trse-context', type=str, default='trse_context.json',
                    help='TRSE regime context file (optional, Step 0 output)')
```

And add `trse_context` to `window_optimizer.json` manifest `default_params`:
```json
"trse_context": "trse_context.json"
```

---

## 3. Do Steps 2–6 Need to Know About Step 0?

**Short answer: No — only Step 1 needs it.**

Here is the full analysis per step:

| Step | Needs TRSE? | Reason |
|---|---|---|
| Step 1: Window Optimizer | **YES** | TRSE narrows `SearchBounds` — regime-aware window size ceiling |
| Step 2: Scorer Meta-Optimizer | No | Optimises residue mods / thresholds — not window params |
| Step 3: Full Scoring | No | Runs fixed params from Step 1 output; features are per-seed |
| Step 4: ML Meta-Optimizer | No | Capacity planner — reads Step 1 output, not regime context |
| Step 5: Anti-Overfit Training | No | Trains on survivors; regime is already baked into survivor features |
| Step 6: Prediction Generator | No | Loads trained model; no regime dependency |

**Why Step 5 doesn't need it directly:**  
The regime is already *implicitly encoded* in the survivor features — `global_regime_age`,
`global_regime_change_detected`, `global_temporal_stability` are all in the 91-feature
vector. Step 5 learns from those. TRSE's job is upstream: ensure Step 1 generates survivors
*from the right window* for the current regime.

**Future exception — per-segment runs:**  
If/when we implement per-segment pipeline runs (run Steps 1-6 separately for each regime
cluster), then Steps 2-6 would each receive a `--segment` flag pointing at regime-specific
data slices. But that's a Phase 2 feature, not part of this wiring.

---

## 4. Complete Wiring Summary (Execution Order)

```
WATCHER --start-step 0 --end-step 6
    │
    ├─ Step 0: trse_step0.py
    │     reads: daily3.json
    │     writes: trse_context.json
    │     WATCHER checks: trse_context.json exists → PROCEED
    │
    ├─ Step 1: window_optimizer.py --trse-context trse_context.json
    │     reads: trse_context.json  ← NEW (passive, optional)
    │     narrowed bounds: max_window_size biased toward recommended_window_size=8
    │     writes: optimal_window_config.json, bidirectional_survivors.json
    │
    ├─ Step 2: scorer meta-optimizer  (no change)
    ├─ Step 3: full scoring           (no change)
    ├─ Step 4: ML meta-optimizer      (no change)
    ├─ Step 5: anti-overfit training  (no change)
    └─ Step 6: prediction generator   (no change)
```

---

## 5. Files To Create/Modify

| File | Action | Change |
|---|---|---|
| `agents/watcher_agent.py` | Modify | Add Step 0 to 3 registries + timeout |
| `agent_manifests/trse.json` | Create | New manifest for Step 0 |
| `window_optimizer_bayesian.py` | Modify | Read trse_context.json, narrow bounds |
| `window_optimizer.py` | Modify | Add `--trse-context` CLI arg |
| `agent_manifests/window_optimizer.json` | Modify | Add `trse_context` to default_params |

**Total: 3 modified files, 1 new file, 1 new manifest**

---

## 6. Implementation Order

1. `agent_manifests/trse.json` — create first (WATCHER needs it to register Step 0)
2. `agents/watcher_agent.py` — add Step 0 to registries
3. `window_optimizer.py` — add `--trse-context` CLI arg
4. `window_optimizer_bayesian.py` — add bounds narrowing logic
5. `agent_manifests/window_optimizer.json` — add `trse_context` to default_params
6. Smoke test: `--start-step 0 --end-step 1` with `--force` on TRSE

---

*Team Alpha S121*
