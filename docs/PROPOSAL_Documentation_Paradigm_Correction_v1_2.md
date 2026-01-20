# PROPOSAL: Documentation Cleanup — Functional Mimicry Language

**Version:** 1.2  
**Date:** 2026-01-19  
**Status:** PENDING APPROVAL  
**Type:** Documentation Only (No Code Changes)

---

## 1. Summary

The Chapter 13 **code is correct** — it measures hit rate, confidence calibration, and pattern learning metrics. However, several documentation files contain misleading language implying the system tries to "find seeds."

This proposal corrects documentation to align with the actual code behavior.

---

## 2. The Correct Understanding: Seeds as Vehicles for Heuristic Extraction

The system does not attempt to reverse-engineer PRNG internal state or "discover" seeds. Seeds serve a fundamentally different purpose in this architecture.

**Seeds are vehicles for generating candidate output sequences.** The sieves (Steps 1-2) filter seeds based on whether their *outputs* match observed draw patterns — not because we want the seeds themselves, but because seeds that produce matching outputs allow us to extract meaningful heuristics from those output sequences.

The pipeline flow is:

```
Seeds → Generate Candidate Sequences → Filter via Sieve → EXTRACT HEURISTICS → ML Learns Patterns → Predict Future Outputs
       ↑                                                    ↑
       |                                                    |
  Means to an end                               The actual goal
```

Survivors are seeds whose **output statistics** align with observed lottery data. From these survivor outputs, we extract 47+ features: residue coherence, lane agreement, temporal stability, skip entropy, and others. The ML model then learns the mapping from these **surface patterns** to prediction quality.

The critical insight is that we never need to know the "true" seed to make predictions. The system achieves **functional mimicry** — learning the statistical surface of the PRNG's behavior well enough to predict future outputs, even though the internal state remains unknown. This is mathematically distinct from (and far more achievable than) seed recovery or state reconstruction.

When synthetic data uses a known `true_seed`, the purpose is to generate **consistent, reproducible test sequences** for measuring whether pattern learning improves over time — not to validate whether the system "finds" that seed.

---

## 3. Code Verification

**`chapter_13_diagnostics.py` already measures:**
```python
"prediction_validation": {
    "exact_hits": int,           # ✅ Hit rate
    "pool_coverage": float       # ✅ Coverage metric
},
"confidence_calibration": {
    "predicted_vs_actual_correlation": float  # ✅ Calibration
}
```

**Diagnostic thresholds (correct):**
| Metric | Threshold |
|--------|-----------|
| Exact Hit Rate | > 0.05 |
| Confidence Correlation | > 0.3 |
| Consecutive Misses | < 5 |

**No code changes required.**

---

## 4. Documentation Fixes

### 4.1 CHAPTER_13_IMPLEMENTATION_PROGRESS_v1_1.md

**REMOVE (Phase 7 table):**
```markdown
| True seed in top-100 | ≤20 draws | Feature construction issue |
| True seed in top-20 | ≤50 draws | Learning loop issue |
```

**REPLACE WITH:**
```markdown
| Hit Rate (Top-20) | > 5% | Pattern extraction working |
| Confidence Calibration | r > 0.3 | Confidence scores meaningful |
| Hit Rate Trend | Non-decreasing | Learning loop improving |
```

---

### 4.2 CANONICAL_PIPELINE_AND_CH13_WITH_STARTUP_COMPLETE.md (Section 9)

**REMOVE:**
```markdown
| True seed in top‑100 | ≤ 20 draws |
| True seed in top‑20 | ≤ 50 draws |
```

**REPLACE WITH:**
```markdown
| Hit Rate | > 5× random baseline |
| Confidence Calibration | Correlation > 0.3 |
| Pattern Learning | Improving over N draws |
```

---

### 4.3 CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md (Section 6)

**REMOVE:**
```markdown
> **With a correct PRNG hypothesis, the true seed should enter the top-K survivors within N synthetic draws.**
```

**REPLACE WITH:**
```markdown
> **With a correct PRNG hypothesis, learned patterns should produce measurable prediction lift over random baseline.**
```

---

### 4.4 watcher_policies.json (if applicable)

**Verify and remove if present:**
```json
"true_seed_top_100": 20,
"true_seed_top_20": 50
```

**Should contain:**
```json
"convergence_targets": {
    "hit_rate_threshold": 0.05,
    "confidence_correlation_threshold": 0.3,
    "max_consecutive_misses": 10
}
```

---

## 5. Add Paradigm Note

Add to Chapter 1, Chapter 6, Chapter 13 headers:

```markdown
> **Paradigm:** This system performs functional mimicry — learning output patterns to predict future draws. Seeds generate candidate sequences for heuristic extraction; they are not discovery targets. The ML model learns from output features, not seed values.
```

---

## 6. Implementation

```bash
# On Zeus
cd ~/distributed_prng_analysis

# Update docs
# [Apply changes above]

# Verify no seed-ranking language remains
grep -r "seed.*top-100\|seed.*top-20\|find.*seed" *.md docs/

# Commit
git add -A
git commit -m "docs: Remove misleading seed-ranking language

Code already measures hit rate & confidence calibration correctly.
Documentation now aligns with functional mimicry paradigm.

Seeds are vehicles for generating candidate sequences, enabling
heuristic extraction from outputs. The system learns surface
patterns to predict future draws - it does not discover seeds."

git push
```

---

## 7. Approval

- [ ] Proceed with documentation cleanup
- [ ] Modify proposal
- [ ] Decline

**Effort:** ~15 minutes

---

*End of Proposal*
