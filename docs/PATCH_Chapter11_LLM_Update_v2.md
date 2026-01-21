# PATCH: Chapter 11 LLM Configuration Update

**File:** `CHAPTER_11_FEATURE_IMPORTANCE_VISUALIZATION.md`  
**Date:** 2026-01-19  
**Purpose:** Update from Qwen2.5-Math to DeepSeek-R1-14B + Claude backup  
**Relates to:** `llm_server_config.json` v2.0.0 (Jan 7, 2026)

---

## Summary of Changes

| Line | Old | New |
|------|-----|-----|
| 56 | Qwen2.5-Math | DeepSeek-R1-14B + Claude backup |
| 763 | port 8081 | port 8080 |
| 766 | Qwen2.5-Math docstring | DeepSeek-R1-14B docstring |
| 809 | Example header Qwen | Example header DeepSeek |
| 1084 | Summary Qwen | Summary DeepSeek + Claude |

---

## Patch 1: Line 56 (Key Features Table)

**FIND:**
```markdown
| **AI Interpretation** | LLM-powered analysis via Qwen2.5-Math |
```

**REPLACE WITH:**
```markdown
| **AI Interpretation** | LLM-powered analysis via DeepSeek-R1-14B + Claude backup |
```

---

## Patch 2: Lines 757-835 (Section 10 - Complete Replacement)

**FIND (entire section 10.1 and 10.2):**
```markdown
### 10.1 LLM Integration

```python
def generate_ai_interpretation(
    importance: Dict[str, float],
    model_metrics: Dict[str, float],
    llm_endpoint: str = "http://localhost:8081/v1/completions"
) -> str:
    """
    Generate AI-powered interpretation using Qwen2.5-Math.
    
    Uses the Math LLM for statistical reasoning about features.
    """
    import requests
    
    # Prepare prompt
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:10]
    features_str = "\n".join([f"  {name}: {value:.4f}" for name, value in top_features])
    
    prompt = f"""Analyze this feature importance distribution from a PRNG seed prediction model:

Top 10 Features:
{features_str}

Model Metrics:
  Validation MAE: {model_metrics.get('val_mae', 'N/A')}
  Test MAE: {model_metrics.get('test_mae', 'N/A')}
  Overfit Ratio: {model_metrics.get('overfit_ratio', 'N/A')}

Questions to address:
1. Is the importance distribution healthy (spread across multiple features)?
2. Are there signs of circular/leaky features?
3. Which feature categories dominate (intersection, skip, lane, temporal)?
4. Recommendations for feature engineering?

Provide a concise statistical interpretation."""

    response = requests.post(
        llm_endpoint,
        json={
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.3
        }
    )
    
    return response.json()['choices'][0]['text']
```

### 10.2 Example AI Interpretation

```
AI Interpretation (Qwen2.5-Math-7B):
────────────────────────────────────
The feature importance distribution shows a HEALTHY pattern:

1. **Distribution Quality**: Top 10 features account for ~65% of importance,
   with remaining 35% spread across 52 features. This indicates the model
   is learning multiple predictive signals, not overfitting to one.

2. **No Circular Features Detected**: 'residue_1000_match_rate' is NOT in
   top 10, suggesting training target is correctly defined as holdout_hits
   rather than training score.

3. **Category Analysis**:
   - Intersection features (32%): Strongest predictors, indicating
     bidirectional sieve agreement is key
   - Skip features (24%): Second strongest, confirming skip hypothesis
     consistency matters
   - Lane features (18%): CRT validation adds predictive value
   - Temporal features (14%): Time stability provides moderate signal

4. **Recommendations**:
   - Consider combining intersection_weight and bidirectional_selectivity
     into a single "bidirectional_quality" meta-feature
   - skip_entropy and skip_range are highly correlated (r=0.87), may
     be redundant
   - Explore non-linear interactions between lane_agreement_* features
```
```

**REPLACE WITH:**
```markdown
### 10.1 LLM Integration

The AI interpretation system uses a fallback chain for reliability:

| Priority | Model | Endpoint | Use Case |
|----------|-------|----------|----------|
| Primary | DeepSeek-R1-14B | localhost:8080 | Fast local inference |
| Backup | Claude Opus 4.5 | Claude CLI | Deep analysis fallback |
| Final | Template | N/A | When both LLMs unavailable |

**Implementation:** `feature_importance_ai_interpreter.py` (v2.0.0)

```python
def query_llm(prompt: str, max_tokens: int = 1024) -> str:
    """
    Query LLM with automatic fallback chain.
    
    Fallback order:
    1. DeepSeek-R1-14B (localhost:8080) - primary
    2. Claude CLI (claude -p) - backup
    3. Template interpretation - final fallback
    """
    
    # Try DeepSeek first
    if check_deepseek_available():
        result = query_deepseek(prompt, max_tokens)
        if result:
            return result
    
    # Try Claude backup
    if check_claude_available():
        result = query_claude(prompt)
        if result:
            return result
    
    # Return None to signal template needed
    return None


def query_deepseek(prompt: str, max_tokens: int) -> Optional[str]:
    """Query DeepSeek-R1-14B on localhost:8080."""
    response = requests.post(
        "http://localhost:8080/completion",
        json={
            "prompt": f"{SYSTEM_PROMPT}\n\nUser: {prompt}\n\nAssistant:",
            "n_predict": max_tokens,
            "temperature": 0.7,
            "stop": ["</s>", "<|im_end|>", "<|endoftext|>", "User:"]
        },
        timeout=120
    )
    return response.json().get('content', '').strip()


def query_claude(prompt: str) -> Optional[str]:
    """Query Claude via CLI as fallback."""
    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True, text=True, timeout=120
    )
    return result.stdout.strip() if result.returncode == 0 else None
```

### 10.2 CLI Usage

```bash
# Run interpretation with automatic fallback
python3 feature_importance_ai_interpreter.py \
    --step5 feature_importance_step5.json \
    --drift feature_drift_step4_to_step5.json

# Test LLM connectivity
python3 feature_importance_ai_interpreter.py --test-llm
```

Expected output:
```
Testing LLM connectivity...

DeepSeek-R1-14B: ✅ Available
Claude CLI:      ✅ Available
```

### 10.3 Example AI Interpretation

```
============================================================
🤖 AI FEATURE IMPORTANCE INTERPRETATION
============================================================

LLM Priority:
  1. DeepSeek-R1-14B (localhost:8080)
  2. Claude CLI (fallback)
  3. Template (final fallback)

Querying LLM...

  Trying DeepSeek-R1-14B (primary)...
------------------------------------------------------------
# PRNG Pattern Detection: Feature Importance Analysis

## 1. What Top Features Reveal About Predictability

The feature importance hierarchy tells a clear story:

| Feature | Importance | Interpretation |
|---------|------------|----------------|
| **lane_agreement_8** | 14.2% | Multi-lane consensus signal |
| **skip_entropy** | 11.2% | Low entropy = predictable gaps |
| **temporal_stability_mean** | 9.5% | Consistent behavior over time |

**Key insight:** The PRNG is vulnerable through consensus-based 
multi-lane analysis combined with temporal consistency.

## 2. Concerning Patterns

- Low `reseed_probability` importance (0.8%) — weak reseeding
- `global_entropy` nearly useless (1.5%) — looks random but isn't
- Forward/reverse match rates similar — state function leaks both ways

## 3. Drift Analysis

| Metric | Value | Assessment |
|--------|-------|------------|
| Drift Score | 0.042 | **Low** — model stable |
| Status | stable | No retraining needed |

## 4. Recommendations

1. Drop bottom 3 features (<2% combined importance)
2. Engineer interaction features (lane × temporal)
3. Set drift alert threshold at 0.08
------------------------------------------------------------

✅ AI interpretation complete.
```

### 10.4 Server Configuration

See `llm_services/llm_server_config.json` (v2.0.0):

```json
{
  "primary": {
    "model": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
    "port": 8080,
    "context_length": 8192,
    "capabilities": ["reasoning", "statistical_analysis", "prng_calculations"]
  },
  "backup": {
    "provider": "claude_code",
    "model": "claude-opus-4.5",
    "invocation": "claude --print -p"
  }
}
```

Start server with:
```bash
./llm_services/start_llm_servers.sh
```
```

---

## Patch 3: Line 1084 (Chapter Summary)

**FIND:**
```markdown
- AI interpretation via Qwen2.5-Math
```

**REPLACE WITH:**
```markdown
- AI interpretation via DeepSeek-R1-14B + Claude backup (fallback chain)
```

---

## Verification Commands

After applying patches, verify no Qwen references remain:

```bash
grep -i "qwen" CHAPTER_11_FEATURE_IMPORTANCE_VISUALIZATION.md
# Should return nothing

grep -i "8081" CHAPTER_11_FEATURE_IMPORTANCE_VISUALIZATION.md
# Should return nothing (old math port)

grep -i "deepseek" CHAPTER_11_FEATURE_IMPORTANCE_VISUALIZATION.md
# Should return multiple matches
```

---

## Related Changes

These files were also updated as part of the LLM migration:

| File | Status | Notes |
|------|--------|-------|
| `llm_services/llm_server_config.json` | ✅ Updated Jan 7 | v2.0.0 |
| `llm_services/llm_router.py` | ✅ Updated Jan 9 | v2.0.0 |
| `llm_services/start_llm_servers.sh` | ✅ Updated Jan 7 | DeepSeek only |
| `feature_importance_ai_interpreter.py` | ✅ Updated Jan 19 | v2.0.0 |
| `CHAPTER_11_FEATURE_IMPORTANCE_VISUALIZATION.md` | 🔲 This patch | Pending |

---

## Migration History

```
2026-01-07: LLM infrastructure migrated (commit 07bfd79)
  - Removed: Qwen2.5-Coder-14B, Qwen2.5-Math-7B
  - Added: DeepSeek-R1-Distill-Qwen-14B
  - Added: Claude Opus 4.5 backup via CLI

2026-01-19: Documentation alignment
  - feature_importance_ai_interpreter.py patched
  - CHAPTER_11 patched (this document)
```

---

*End of Patch Document*
