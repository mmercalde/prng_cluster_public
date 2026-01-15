# Threshold Governance Model

**Version:** 1.0  
**Created:** 2026-01-14  
**Status:** Authoritative

---

## 1. Purpose

This document establishes the single source of truth for threshold configuration, 
authority boundaries, and recovery procedures in the PRNG Analysis Pipeline.

---

## 2. Sources of Truth

| Source | Role | Authority |
|--------|------|-----------|
| `distributed_config.json` | **Optuna search bounds** | Human-only modification |
| `optimal_window_config.json` | **Selected values** (Step 1 output) | Optuna / LLM (bounded) |
| `baselines/baseline_window_thresholds.json` | **Recovery baseline** | Human-only, read-only at runtime |
| `sieve_filter.py` defaults | **Fallback behavior** | Must match baseline |
| `agent_manifests/*.json` | **Documentation only** | NOT runtime bounds |

---

## 3. Current Threshold Configuration

### Search Bounds (distributed_config.json)
```json
{
  "forward_threshold": { "min": 0.15, "max": 0.60, "default": 0.25 },
  "reverse_threshold": { "min": 0.15, "max": 0.60, "default": 0.25 },
  "skip_max": { "min": 10, "max": 250 }
}
```

### Baseline (baselines/baseline_window_thresholds.json)
```json
{
  "forward_threshold": 0.25,
  "reverse_threshold": 0.25,
  "skip_max": 200,
  "expected_survivor_band": [1000, 10000]
}
```

### Key Invariant
```
baseline ∈ [search_min, search_max]
```
The baseline must always be reachable within the search bounds.

---

## 4. What Thresholds Mean

Thresholds are **match rates** — what fraction of residues must match for a seed to survive:

| Threshold | Meaning | Survivors |
|-----------|---------|-----------|
| 0.15 | 15% must match | Many (loose) |
| 0.25 | 25% must match | Moderate (baseline) |
| 0.50 | 50% must match | Few (strict) |
| 0.60 | 60% must match | Very few (aggressive) |

### Target Survivor Band
- **Ideal:** 1,000 – 10,000 bidirectional survivors
- **Too loose (>100K):** No ML signal, R² ≈ 0
- **Too tight (<500):** Overfitting risk

---

## 5. Authority Model

### Who Can Modify What

| Component | Can Modify |
|-----------|------------|
| **Baseline file** | Human only (read-only at runtime) |
| **Search bounds** (distributed_config.json) | Human only |
| **Optimal config** (output) | Optuna, LLM (within bounds, max_delta=30%) |
| **Sieve defaults** | Human only (must match baseline) |

### LLM Constraints (Chapter 13)
- Max parameter change: 30% per proposal
- Max parameters per proposal: 3
- Cooldown between changes: 3 runs

---

## 6. Validation Rules

### Runtime Check (SearchBounds.validate_baseline_in_bounds)
```python
assert search_min <= baseline_threshold <= search_max
```
If violated → WATCHER halts with config error.

### Alignment Invariant
```
sieve_default == baseline == distributed_config.default
```

---

## 7. Recovery Procedures

### If ML Signal Collapses
1. Check survivor count (target: 1K-10K)
2. If >100K survivors → thresholds too loose
3. If <500 survivors → thresholds too tight

### If Config Corruption Suspected
```bash
# Revert to baseline
cp baselines/baseline_window_thresholds.json optimal_window_config.json
```

### If Search Bounds Compromised
```bash
# Restore from backup
cp distributed_config.json.bak_YYYYMMDD distributed_config.json
```

---

## 8. Failure Modes & Mitigations

| Failure Mode | Symptom | Mitigation |
|--------------|---------|------------|
| Thresholds too loose | 99%+ survival, R² ≈ 0 | Raise min threshold |
| Thresholds too strict | <100 survivors | Lower max threshold |
| Baseline unreachable | Validation error | Align bounds with baseline |
| Silent drift | Gradual performance loss | Track threshold history |

---

## 9. Change History

| Date | Change | Author |
|------|--------|--------|
| 2026-01-14 | Initial governance model | Team Alpha + Team Beta |
| 2026-01-14 | Bounds updated to [0.15, 0.60] | Team Beta recommendation |
| 2026-01-14 | Baseline created at 0.25 | Human-approved |

---

## 10. References

- `baselines/baseline_window_thresholds.json` - Recovery baseline
- `distributed_config.json` - Search bounds
- `window_optimizer.py` - SearchBounds class + validation
- `docs/CHAPTER_1_WINDOW_OPTIMIZER.md` - Step 1 documentation
- `docs/CHAPTER_13_*.md` - Live feedback loop

