# Chapter 11: Feature Importance & Visualization

## PRNG Analysis Pipeline — Complete Operating Guide

**Files:** `feature_importance.py`, `feature_visualizer.py`  
**Lines:** ~600 combined  
**Purpose:** ML interpretability, feature analysis, and 13-chart visualization system

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Position](#2-architecture-position)
3. [Feature System Recap](#3-feature-system-recap)
4. [Feature Importance Methods](#4-feature-importance-methods)
5. [Permutation Importance](#5-permutation-importance)
6. [Gradient-Based Importance](#6-gradient-based-importance)
7. [SHAP Analysis](#7-shap-analysis)
8. [Feature Visualizer](#8-feature-visualizer)
9. [13 Chart Types](#9-13-chart-types)
10. [AI-Powered Interpretation](#10-ai-powered-interpretation)
11. [Integration with Pipeline](#11-integration-with-pipeline)
12. [CLI Interface](#12-cli-interface)
13. [Output Formats](#13-output-formats)
14. [Troubleshooting](#14-troubleshooting)
15. [Complete Method Reference](#15-complete-method-reference)

---

## 1. Overview

### 1.1 Why Feature Importance?

After training ML models (Step 5), we need to understand:

- **Which features matter?** — Identify predictive features
- **Are we overfitting?** — Detect circular/leaky features
- **What drives predictions?** — Explain model decisions
- **How to improve?** — Guide feature engineering

### 1.2 The Two Files

| File | Purpose | Charts |
|------|---------|--------|
| `feature_importance.py` | Compute importance scores | N/A |
| `feature_visualizer.py` | Generate 13 chart types | 13 |

### 1.3 Key Features

| Feature | Description |
|---------|-------------|
| **3 Importance Methods** | Permutation, Gradient, SHAP |
| **4 Model Types** | Neural Net, XGBoost, LightGBM, CatBoost |
| **13 Chart Types** | Bar, radar, heatmap, waterfall, etc. |
| **AI Interpretation** | LLM-powered analysis via DeepSeek-R1-14B + Claude backup |
| **Export Formats** | PNG, HTML (interactive), JSON, CSV |

---

## 2. Architecture Position

### 2.1 In the Pipeline

```
Step 5: Anti-Overfit Training
    │
    ├── best_model.pth (or .xgb, .lgb, .cbm)
    ├── best_model.meta.json
    │
    └── Feature Importance Analysis ◄── THIS CHAPTER
            │
            ├── feature_importance.py
            │       ├── Permutation importance
            │       ├── Gradient importance (neural nets)
            │       └── SHAP values
            │
            └── feature_visualizer.py
                    └── 13 chart types
                            │
                            └── visualization_outputs/
                                    ├── importance_bar.png
                                    ├── correlation_matrix.png
                                    ├── shap_summary.html
                                    └── ... (10 more)
```

### 2.2 Why Post-Training?

```
Training gives us:
    - Model weights
    - Validation metrics
    - Best hyperparameters

Feature importance tells us:
    - WHY the model works
    - Which features to keep/remove
    - Whether training target is correct
```

---

## 3. Feature System Recap

### 3.1 62 Total Features

```
Combined Features (62 total):
│
├── Per-Seed Statistical (48)
│   ├── Residue Features (9)
│   │   ├── residue_8_match_rate
│   │   ├── residue_8_coherence
│   │   ├── residue_8_kl_divergence
│   │   ├── residue_125_match_rate
│   │   ├── residue_125_coherence
│   │   ├── residue_125_kl_divergence
│   │   ├── residue_1000_match_rate
│   │   ├── residue_1000_coherence
│   │   └── residue_1000_kl_divergence
│   │
│   ├── Temporal Features (5)
│   │   ├── temporal_stability_mean
│   │   ├── temporal_stability_std
│   │   ├── temporal_stability_min
│   │   ├── temporal_stability_max
│   │   └── temporal_stability_trend
│   │
│   ├── Lane Features (3)
│   │   ├── lane_agreement_8
│   │   ├── lane_agreement_125
│   │   └── lane_consistency
│   │
│   ├── Skip Features (6)
│   │   ├── skip_entropy
│   │   ├── skip_mean
│   │   ├── skip_std
│   │   ├── skip_range
│   │   ├── skip_min
│   │   └── skip_max
│   │
│   ├── Intersection Features (6)
│   │   ├── intersection_count
│   │   ├── intersection_ratio
│   │   ├── intersection_weight
│   │   ├── forward_only_count
│   │   ├── reverse_only_count
│   │   └── bidirectional_selectivity
│   │
│   └── Other Statistical (19)
│       ├── actual_mean, actual_std
│       ├── consecutive_matches
│       ├── exact_matches
│       └── ... (15 more)
│
└── Global State Features (14)
    ├── regime_change_detected
    ├── regime_stability
    ├── marker_390_variance
    ├── marker_567_variance
    ├── reseed_probability
    ├── entropy
    └── ... (8 more)
```

### 3.2 Feature Categories for Analysis

| Category | Count | Predictive Value |
|----------|-------|------------------|
| **Intersection** | 6 | HIGH — bidirectional robustness |
| **Skip** | 6 | HIGH — hypothesis consistency |
| **Lane** | 3 | HIGH — CRT coherence |
| **Temporal** | 5 | MEDIUM — time stability |
| **Residue** | 9 | MEDIUM — (beware circular features) |
| **Global** | 14 | LOW-MEDIUM — regime detection |
| **Statistical** | 19 | VARIES |

### 3.3 Circular Features (Exclude from X)

```python
CIRCULAR_FEATURES = {
    'score',                    # The training target!
    'confidence',               # Derived from score
    'exact_matches',            # Defines score
    'residue_1000_match_rate',  # Equivalent to score
}
```

**Why?** These features measure the training target itself, leading to:
- 100% importance on 1-2 features
- 0% importance on 60 other features
- No generalization

---

## 4. Feature Importance Methods

### 4.1 Three Methods

| Method | Works With | Strengths | Weaknesses |
|--------|------------|-----------|------------|
| **Permutation** | All models | Model-agnostic, reliable | Slow for large datasets |
| **Gradient** | Neural nets | Fast, captures interactions | Only for differentiable models |
| **SHAP** | All models | Explains individual predictions | Very slow, memory-intensive |

### 4.2 When to Use Each

```
Quick check (minutes):
    → Permutation importance

Deep analysis (hours):
    → SHAP values

Neural net debugging:
    → Gradient importance

Production monitoring:
    → Permutation (fast, reliable)
```

---

## 5. Permutation Importance

### 5.1 How It Works

```
For each feature f:
    1. Record baseline score S₀ on validation set
    2. Shuffle feature f (break its relationship with target)
    3. Record new score S₁
    4. Importance(f) = S₀ - S₁

If shuffling feature f hurts performance:
    → f is important
    
If shuffling has no effect:
    → f is not used by model
```

### 5.2 Implementation

```python
def compute_permutation_importance(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Compute permutation importance for any model.
    
    Args:
        model: Trained model with .predict() method
        X_val: Validation features [n_samples, n_features]
        y_val: Validation targets [n_samples]
        feature_names: List of feature names
        n_repeats: Number of shuffles per feature
        random_state: Random seed for reproducibility
    
    Returns:
        Dict mapping feature_name → importance_score
    """
    from sklearn.inspection import permutation_importance
    
    result = permutation_importance(
        model, X_val, y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring='neg_mean_absolute_error'
    )
    
    importance = {}
    for i, name in enumerate(feature_names):
        importance[name] = result.importances_mean[i]
    
    # Normalize to sum to 1
    total = sum(abs(v) for v in importance.values())
    if total > 0:
        importance = {k: abs(v)/total for k, v in importance.items()}
    
    return importance
```

### 5.3 Example Output

```
Permutation Importance (CORRECT training):
──────────────────────────────────────────
intersection_weight:       0.182  ████████████████████
skip_entropy:              0.156  █████████████████
lane_agreement_8:          0.134  ███████████████
temporal_stability_mean:   0.098  ███████████
bidirectional_selectivity: 0.087  ██████████
lane_consistency:          0.076  █████████
skip_range:                0.065  ███████
residue_8_coherence:       0.054  ██████
forward_only_count:        0.048  █████
... (distributed across 50+ features)
```

### 5.4 Red Flag: Circular Features

```
Permutation Importance (BROKEN training):
──────────────────────────────────────────
residue_1000_match_rate:   0.652  █████████████████████████████████████████
exact_matches:             0.348  ██████████████████████
[60 other features]:       0.000  

⚠️ WARNING: Only 2 features have importance!
   This indicates circular features in training.
   'residue_1000_match_rate' ≈ 'score' (the target)
```

---

## 6. Gradient-Based Importance

### 6.1 How It Works

```
For neural networks:
    1. Forward pass: compute predictions
    2. Backward pass: compute ∂prediction/∂input
    3. Importance(f) = mean(|gradient_f|) across samples

Large gradient → small change in feature causes big change in prediction
→ feature is important
```

### 6.2 Implementation

```python
def compute_gradient_importance(
    model: torch.nn.Module,
    X_val: torch.Tensor,
    feature_names: List[str],
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Compute gradient-based importance for PyTorch neural nets.
    
    Args:
        model: Trained PyTorch model
        X_val: Validation features as tensor
        feature_names: List of feature names
        device: 'cuda' or 'cpu'
    
    Returns:
        Dict mapping feature_name → importance_score
    """
    model.eval()
    X_val = X_val.to(device)
    X_val.requires_grad_(True)
    
    # Forward pass
    predictions = model(X_val)
    
    # Backward pass
    predictions.sum().backward()
    
    # Gradient magnitude per feature
    gradients = X_val.grad.abs().mean(dim=0)  # [n_features]
    
    importance = {}
    for i, name in enumerate(feature_names):
        importance[name] = gradients[i].item()
    
    # Normalize
    total = sum(importance.values())
    if total > 0:
        importance = {k: v/total for k, v in importance.items()}
    
    return importance
```

### 6.3 Integrated Gradients (Advanced)

```python
def compute_integrated_gradients(
    model: torch.nn.Module,
    X_val: torch.Tensor,
    baseline: torch.Tensor = None,
    steps: int = 50
) -> torch.Tensor:
    """
    Integrated gradients for more accurate attribution.
    
    Integrates gradients along path from baseline to input.
    """
    if baseline is None:
        baseline = torch.zeros_like(X_val)
    
    # Interpolate between baseline and input
    alphas = torch.linspace(0, 1, steps).to(X_val.device)
    
    gradients = []
    for alpha in alphas:
        interpolated = baseline + alpha * (X_val - baseline)
        interpolated.requires_grad_(True)
        
        output = model(interpolated)
        output.sum().backward()
        
        gradients.append(interpolated.grad.clone())
        interpolated.grad.zero_()
    
    # Average gradients and multiply by (input - baseline)
    avg_gradients = torch.stack(gradients).mean(dim=0)
    integrated_grads = (X_val - baseline) * avg_gradients
    
    return integrated_grads
```

---

## 7. SHAP Analysis

### 7.1 What is SHAP?

**SHAP** (SHapley Additive exPlanations) provides:
- **Local explanations** — Why did THIS prediction happen?
- **Global explanations** — Which features matter overall?
- **Interaction effects** — How do features combine?

### 7.2 Implementation

```python
def compute_shap_values(
    model: Any,
    X_val: np.ndarray,
    feature_names: List[str],
    model_type: str = 'tree',
    max_samples: int = 1000
) -> Dict[str, Any]:
    """
    Compute SHAP values for model interpretability.
    
    Args:
        model: Trained model
        X_val: Validation features
        feature_names: List of feature names
        model_type: 'tree' for XGBoost/LightGBM, 'kernel' for neural nets
        max_samples: Max samples for computation (SHAP is slow!)
    
    Returns:
        Dict with shap_values, feature_importance, and plots
    """
    import shap
    
    # Sample if too many
    if len(X_val) > max_samples:
        idx = np.random.choice(len(X_val), max_samples, replace=False)
        X_sample = X_val[idx]
    else:
        X_sample = X_val
    
    # Choose explainer based on model type
    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(
            model.predict, 
            shap.sample(X_sample, 100)
        )
    
    shap_values = explainer.shap_values(X_sample)
    
    # Global importance (mean absolute SHAP)
    importance = {}
    for i, name in enumerate(feature_names):
        importance[name] = np.abs(shap_values[:, i]).mean()
    
    # Normalize
    total = sum(importance.values())
    if total > 0:
        importance = {k: v/total for k, v in importance.items()}
    
    return {
        'shap_values': shap_values,
        'feature_importance': importance,
        'explainer': explainer
    }
```

### 7.3 SHAP Visualizations

```python
def generate_shap_plots(shap_result: Dict, feature_names: List[str], output_dir: str):
    """Generate SHAP visualization plots."""
    import shap
    import matplotlib.pyplot as plt
    
    shap_values = shap_result['shap_values']
    
    # Summary plot (beeswarm)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    plt.savefig(f"{output_dir}/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Bar plot (importance)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, feature_names=feature_names, 
                      plot_type='bar', show=False)
    plt.savefig(f"{output_dir}/shap_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Dependence plots for top features
    importance = shap_result['feature_importance']
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:5]
    
    for name, _ in top_features:
        idx = feature_names.index(name)
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(idx, shap_values, feature_names=feature_names, show=False)
        plt.savefig(f"{output_dir}/shap_dependence_{name}.png", dpi=150, bbox_inches='tight')
        plt.close()
```

---

## 8. Feature Visualizer

### 8.1 Overview

`feature_visualizer.py` generates 13 chart types for comprehensive feature analysis:

```python
class FeatureVisualizer:
    """
    Comprehensive feature visualization system.
    
    Generates 13 chart types using Plotly, Seaborn, and Matplotlib.
    """
    
    def __init__(
        self,
        feature_importance: Dict[str, float],
        feature_names: List[str],
        X_data: np.ndarray,
        y_data: np.ndarray,
        output_dir: str = "visualization_outputs"
    ):
        self.importance = feature_importance
        self.feature_names = feature_names
        self.X = X_data
        self.y = y_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all(self):
        """Generate all 13 chart types."""
        self.importance_bar_chart()
        self.importance_radar_chart()
        self.correlation_matrix()
        self.feature_distribution()
        self.pairwise_scatter()
        self.importance_heatmap()
        self.cumulative_importance()
        self.feature_violin()
        self.top_features_boxplot()
        self.shap_waterfall()
        self.shap_force_plot()
        self.feature_cluster_dendrogram()
        self.interactive_dashboard()
```

---

## 9. 13 Chart Types

### 9.1 Chart Catalog

| # | Chart Type | Library | Purpose |
|---|------------|---------|---------|
| 1 | **Importance Bar** | Plotly | Ranked feature importance |
| 2 | **Importance Radar** | Plotly | Radial importance view |
| 3 | **Correlation Matrix** | Seaborn | Feature correlations |
| 4 | **Feature Distribution** | Seaborn | Histograms per feature |
| 5 | **Pairwise Scatter** | Seaborn | Feature relationships |
| 6 | **Importance Heatmap** | Plotly | Category-grouped importance |
| 7 | **Cumulative Importance** | Matplotlib | Pareto analysis |
| 8 | **Feature Violin** | Seaborn | Distribution by target |
| 9 | **Top Features Boxplot** | Seaborn | Spread of top features |
| 10 | **SHAP Waterfall** | SHAP | Individual prediction explain |
| 11 | **SHAP Force Plot** | SHAP | Feature contribution flow |
| 12 | **Feature Cluster Dendrogram** | Scipy | Feature groupings |
| 13 | **Interactive Dashboard** | Plotly | Combined HTML dashboard |

### 9.2 Chart 1: Importance Bar Chart

```python
def importance_bar_chart(self, top_n: int = 20):
    """Horizontal bar chart of top N features by importance."""
    import plotly.express as px
    
    # Sort and select top N
    sorted_imp = sorted(self.importance.items(), key=lambda x: -x[1])[:top_n]
    names = [x[0] for x in sorted_imp]
    values = [x[1] for x in sorted_imp]
    
    fig = px.bar(
        x=values, y=names,
        orientation='h',
        title=f"Top {top_n} Feature Importance",
        labels={'x': 'Importance', 'y': 'Feature'}
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    fig.write_html(self.output_dir / "importance_bar.html")
    fig.write_image(self.output_dir / "importance_bar.png", scale=2)
```

### 9.3 Chart 2: Importance Radar Chart

```python
def importance_radar_chart(self, top_n: int = 10):
    """Radar/spider chart for top features."""
    import plotly.graph_objects as go
    
    sorted_imp = sorted(self.importance.items(), key=lambda x: -x[1])[:top_n]
    names = [x[0] for x in sorted_imp]
    values = [x[1] for x in sorted_imp]
    
    # Close the polygon
    names = names + [names[0]]
    values = values + [values[0]]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=names,
        fill='toself',
        name='Importance'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(values)*1.1])),
        title=f"Feature Importance Radar (Top {top_n})"
    )
    
    fig.write_html(self.output_dir / "importance_radar.html")
    fig.write_image(self.output_dir / "importance_radar.png", scale=2)
```

### 9.4 Chart 3: Correlation Matrix

```python
def correlation_matrix(self, method: str = 'pearson'):
    """Heatmap of feature correlations."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.DataFrame(self.X, columns=self.feature_names)
    corr = df.corr(method=method)
    
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr, mask=mask,
        cmap='RdBu_r', center=0,
        annot=False,  # Too many features for annotations
        square=True,
        linewidths=0.5
    )
    
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(self.output_dir / "correlation_matrix.png", dpi=150)
    plt.close()
```

### 9.5 Chart 7: Cumulative Importance (Pareto)

```python
def cumulative_importance(self):
    """Pareto chart showing cumulative importance."""
    import matplotlib.pyplot as plt
    
    sorted_imp = sorted(self.importance.items(), key=lambda x: -x[1])
    names = [x[0] for x in sorted_imp]
    values = [x[1] for x in sorted_imp]
    cumulative = np.cumsum(values)
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Bar chart
    ax1.bar(range(len(names)), values, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Feature Rank')
    ax1.set_ylabel('Importance', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    
    # Cumulative line
    ax2 = ax1.twinx()
    ax2.plot(range(len(names)), cumulative, 'r-', linewidth=2, marker='o', markersize=3)
    ax2.set_ylabel('Cumulative Importance', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # 80% line
    ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='80% threshold')
    
    # Find how many features for 80%
    n_80 = np.searchsorted(cumulative, 0.8) + 1
    ax1.axvline(x=n_80, color='green', linestyle='--', alpha=0.7)
    
    plt.title(f"Cumulative Feature Importance (80% at {n_80} features)")
    plt.tight_layout()
    plt.savefig(self.output_dir / "cumulative_importance.png", dpi=150)
    plt.close()
```

### 9.6 Chart 13: Interactive Dashboard

```python
def interactive_dashboard(self):
    """Combined interactive HTML dashboard."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Feature Importance (Bar)",
            "Feature Importance (Radar)",
            "Top Feature Distributions",
            "Cumulative Importance"
        ),
        specs=[
            [{"type": "bar"}, {"type": "polar"}],
            [{"type": "histogram"}, {"type": "scatter"}]
        ]
    )
    
    # Add all subplots...
    # (Implementation details for each subplot)
    
    fig.update_layout(
        height=900,
        width=1400,
        title_text="Feature Importance Dashboard"
    )
    
    fig.write_html(self.output_dir / "dashboard.html")
```

---

## 10. AI-Powered Interpretation

### 10.1 LLM Integration

```python
def generate_ai_interpretation(
    importance: Dict[str, float],
    model_metrics: Dict[str, float],
    llm_endpoint: str = "http://localhost:8080/completion"
) -> str:
    """
    Generate AI-powered interpretation using DeepSeek-R1-14B.
    
    Uses DeepSeek-R1-14B (primary) with Claude backup for statistical reasoning about features.
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
AI Interpretation (DeepSeek-R1-14B):
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

---

## 11. Integration with Pipeline

### 11.1 After Step 5 Training

```bash
# Train model (Step 5)
python3 meta_prediction_optimizer_anti_overfit.py \
    --survivors survivors_with_scores.json \
    --model-type neural_net \
    --n-trials 50

# Analyze feature importance
python3 feature_importance.py \
    --model-path models/reinforcement/best_model.pth \
    --survivors survivors_with_scores.json \
    --output-dir visualization_outputs

# Generate all visualizations
python3 feature_visualizer.py \
    --importance-json visualization_outputs/importance.json \
    --data-file survivors_with_scores.json \
    --output-dir visualization_outputs
```

### 11.2 With --save-all-models (TODO)

```bash
# Train all 4 models
python3 meta_prediction_optimizer_anti_overfit.py \
    --survivors survivors_with_scores.json \
    --compare-models \
    --save-all-models

# Compare importance across models
python3 feature_importance.py \
    --model-dir models/reinforcement/ \
    --compare-all \
    --output-dir visualization_outputs

# Generates:
#   importance_neural_net.json
#   importance_xgboost.json
#   importance_lightgbm.json
#   importance_catboost.json
#   importance_comparison.html
```

---

## 12. CLI Interface

### 12.1 feature_importance.py

```bash
python3 feature_importance.py [options]

Options:
  --model-path PATH       Path to trained model file
  --model-type TYPE       Model type: neural_net, xgboost, lightgbm, catboost
  --survivors PATH        Path to survivors JSON with features
  --method METHOD         Importance method: permutation, gradient, shap, all
  --n-repeats INT         Repeats for permutation (default: 10)
  --max-samples INT       Max samples for SHAP (default: 1000)
  --output-dir DIR        Output directory (default: visualization_outputs)
  --json-only             Only output JSON, skip plots
  -h, --help              Show help
```

### 12.2 feature_visualizer.py

```bash
python3 feature_visualizer.py [options]

Options:
  --importance-json PATH  Path to importance JSON from feature_importance.py
  --data-file PATH        Path to survivors JSON with features
  --charts CHARTS         Comma-separated chart types (default: all)
  --top-n INT             Number of top features to show (default: 20)
  --output-dir DIR        Output directory (default: visualization_outputs)
  --format FORMAT         Output format: png, html, both (default: both)
  --dpi INT               DPI for PNG output (default: 150)
  -h, --help              Show help
```

### 12.3 Example Workflows

**Quick importance check:**
```bash
python3 feature_importance.py \
    --model-path best_model.pth \
    --model-type neural_net \
    --survivors survivors.json \
    --method permutation \
    --json-only
```

**Full analysis with all charts:**
```bash
python3 feature_importance.py \
    --model-path best_model.pth \
    --model-type neural_net \
    --survivors survivors.json \
    --method all \
    --output-dir analysis_run_001

python3 feature_visualizer.py \
    --importance-json analysis_run_001/importance.json \
    --data-file survivors.json \
    --output-dir analysis_run_001 \
    --format both
```

---

## 13. Output Formats

### 13.1 importance.json

```json
{
    "method": "permutation",
    "model_type": "neural_net",
    "n_features": 62,
    "timestamp": "2025-12-30T14:23:45",
    "importance": {
        "intersection_weight": 0.182,
        "skip_entropy": 0.156,
        "lane_agreement_8": 0.134,
        "temporal_stability_mean": 0.098,
        "bidirectional_selectivity": 0.087,
        ...
    },
    "metadata": {
        "n_samples": 5000,
        "n_repeats": 10,
        "baseline_score": 0.0423,
        "computation_time_seconds": 127.4
    }
}
```

### 13.2 Generated Files

```
visualization_outputs/
├── importance.json              # Raw importance scores
├── importance_bar.html          # Interactive bar chart
├── importance_bar.png           # Static bar chart
├── importance_radar.html        # Interactive radar
├── importance_radar.png         # Static radar
├── correlation_matrix.png       # Feature correlations
├── feature_distributions.png    # Histograms
├── cumulative_importance.png    # Pareto chart
├── feature_violin.png           # Violin plots
├── top_features_boxplot.png     # Box plots
├── shap_summary.png             # SHAP beeswarm
├── shap_bar.png                 # SHAP importance
├── shap_dependence_*.png        # Per-feature SHAP
├── feature_dendrogram.png       # Hierarchical clustering
├── dashboard.html               # Combined interactive
└── ai_interpretation.txt        # LLM analysis
```

---

## 14. Troubleshooting

### 14.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "SHAP too slow" | Large dataset | Use `--max-samples 500` |
| "All importance = 0" | Model not trained | Verify model loaded correctly |
| "2 features = 100%" | Circular features | Check training target is holdout_hits |
| "Memory error" | Too many features | Use permutation instead of SHAP |
| "Plotly not found" | Missing dependency | `pip install plotly kaleido` |

### 14.2 Verify Correct Training

```python
# Check if importance is healthy
importance = load_json("importance.json")["importance"]

top_10 = sorted(importance.items(), key=lambda x: -x[1])[:10]
top_10_sum = sum(v for _, v in top_10)

if top_10_sum > 0.95:
    print("⚠️  WARNING: Top 10 features = 95%+ importance")
    print("   This suggests circular features in training")
    print("   Check that y = holdout_hits, not score")
else:
    print(f"✅ Healthy distribution: Top 10 = {top_10_sum:.1%}")
```

---

## 15. Complete Method Reference

### 15.1 feature_importance.py

| Function | Purpose |
|----------|---------|
| `compute_permutation_importance()` | Model-agnostic importance |
| `compute_gradient_importance()` | Neural net gradients |
| `compute_integrated_gradients()` | Advanced gradient attribution |
| `compute_shap_values()` | SHAP analysis |
| `generate_shap_plots()` | SHAP visualizations |
| `compare_model_importance()` | Cross-model comparison |

### 15.2 feature_visualizer.py

| Method | Chart Type |
|--------|------------|
| `importance_bar_chart()` | #1 Bar chart |
| `importance_radar_chart()` | #2 Radar chart |
| `correlation_matrix()` | #3 Heatmap |
| `feature_distribution()` | #4 Histograms |
| `pairwise_scatter()` | #5 Scatter matrix |
| `importance_heatmap()` | #6 Category heatmap |
| `cumulative_importance()` | #7 Pareto chart |
| `feature_violin()` | #8 Violin plots |
| `top_features_boxplot()` | #9 Box plots |
| `shap_waterfall()` | #10 SHAP waterfall |
| `shap_force_plot()` | #11 SHAP force |
| `feature_cluster_dendrogram()` | #12 Dendrogram |
| `interactive_dashboard()` | #13 HTML dashboard |

---

## 16. Chapter Summary

**Chapter 14: Feature Importance & Visualization** covers ML interpretability:

| Component | Purpose |
|-----------|---------|
| **Permutation** | Model-agnostic importance |
| **Gradient** | Neural net attributions |
| **SHAP** | Individual + global explanations |
| **13 Charts** | Comprehensive visualization |
| **AI Interpretation** | LLM-powered analysis |

**Key Points:**
- 3 importance methods for different use cases
- 13 chart types cover all visualization needs
- Circular feature detection prevents overfitting
- AI interpretation via DeepSeek-R1-14B + Claude backup
- Export to PNG, HTML (interactive), JSON

---

## Next Chapter

**Chapter 16: Prediction Generator** will cover:
- `prediction_generator.py` — Generate final predictions
- Model loading with schema validation
- Top-K survivor ranking
- Agent metadata for autonomous operation

---

*End of Chapter 14: Feature Importance & Visualization*
