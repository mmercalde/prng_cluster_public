#!/usr/bin/env python3
"""
Feature Importance Visualizer - Phase 5
========================================

Plotly-based visualization for ML model feature importance and drift tracking.
Designed to integrate with web_dashboard.py following existing patterns.

Usage:
    # Standalone
    python3 feature_visualizer.py
    
    # In dashboard
    from feature_visualizer import generate_feature_importance_chart, generate_drift_chart
    
Author: Distributed PRNG Analysis System
Date: December 10, 2025
Version: 1.0.0
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


def load_feature_importance(filepath: str) -> Optional[Dict]:
    """Load feature importance from JSON file."""
    try:
        path = Path(filepath)
        if not path.exists():
            return None
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def generate_feature_importance_chart(
    filepath: str = None,
    importance_dict: Dict[str, float] = None,
    top_n: int = 20,
    title: str = "Feature Importance",
    plot_height: int = 500
) -> Optional[str]:
    """
    Generate horizontal bar chart for feature importance using Plotly.
    
    Args:
        filepath: Path to feature_importance JSON file
        importance_dict: Direct dict of {feature_name: importance_value}
        top_n: Number of top features to show
        title: Chart title
        plot_height: Height in pixels
        
    Returns:
        HTML string with Plotly chart, or None if error
    """
    try:
        import plotly.graph_objects as go
        
        # Load data
        if importance_dict is None:
            if filepath is None:
                # Try default locations
                for default_path in [
                    'feature_importance_step5.json',
                    'feature_importance_step4.json',
                    'optimization_results/feature_importance.json'
                ]:
                    if Path(default_path).exists():
                        filepath = default_path
                        break
            
            if filepath is None:
                return None
                
            data = load_feature_importance(filepath)
            if data is None:
                return None
            
            # Extract importance dict from various formats
            if 'feature_importance' in data:
                importance_dict = data['feature_importance']
            elif 'importance_by_feature' in data:
                importance_dict = data['importance_by_feature']
            else:
                importance_dict = data
        
        if not importance_dict:
            return None
        
        # Sort by importance descending and take top N
        sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Reverse for horizontal bar chart (top at top)
        sorted_items = sorted_items[::-1]
        
        features = [item[0] for item in sorted_items]
        values = [item[1] * 100 for item in sorted_items]  # Convert to percentage
        
        # Color gradient based on importance
        colors = []
        max_val = max(values) if values else 1
        for v in values:
            intensity = v / max_val
            # Green gradient: higher importance = more saturated green
            r = int(59 * (1 - intensity * 0.5))
            g = int(130 + 50 * intensity)
            b = int(246 * (1 - intensity * 0.7))
            colors.append(f'rgb({r},{g},{b})')
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=features,
            x=values,
            orientation='h',
            marker=dict(color=colors),
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            autosize=True,
            height=plot_height,
            title=dict(
                text=title,
                font=dict(color="#e8e8e8", size=16)
            ),
            xaxis=dict(
                title="Importance %",
                color="#8a9099",
                gridcolor="#3a3f45",
                range=[0, max(values) * 1.1] if values else [0, 100]
            ),
            yaxis=dict(
                color="#8a9099",
                tickfont=dict(size=10)
            ),
            plot_bgcolor="#2a2e33",
            paper_bgcolor="#1a1d21",
            font=dict(color="#e8e8e8"),
            margin=dict(l=180, r=30, t=50, b=40)
        )
        
        return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})
        
    except Exception as e:
        print(f"Error generating feature importance chart: {e}")
        return None


def generate_drift_comparison_chart(
    step4_path: str = 'feature_importance_step4.json',
    step5_path: str = 'feature_importance_step5.json',
    top_n: int = 15,
    plot_height: int = 500
) -> Optional[str]:
    """
    Generate side-by-side comparison of Step 4 vs Step 5 feature importance.
    
    Shows which features gained/lost importance between training phases.
    """
    try:
        import plotly.graph_objects as go
        
        # Load both files
        step4_data = load_feature_importance(step4_path)
        step5_data = load_feature_importance(step5_path)
        
        if step4_data is None or step5_data is None:
            return None
        
        # Extract importance dicts
        step4_imp = step4_data.get('feature_importance', step4_data)
        step5_imp = step5_data.get('feature_importance', step5_data)
        
        # Get union of features, sorted by Step 5 importance
        all_features = set(step4_imp.keys()) | set(step5_imp.keys())
        
        # Calculate deltas and sort by absolute change
        feature_data = []
        for f in all_features:
            v4 = step4_imp.get(f, 0)
            v5 = step5_imp.get(f, 0)
            delta = v5 - v4
            feature_data.append((f, v4, v5, delta))
        
        # Sort by Step 5 importance, take top N
        feature_data.sort(key=lambda x: x[2], reverse=True)
        feature_data = feature_data[:top_n]
        
        # Reverse for chart display
        feature_data = feature_data[::-1]
        
        features = [d[0] for d in feature_data]
        step4_vals = [d[1] * 100 for d in feature_data]
        step5_vals = [d[2] * 100 for d in feature_data]
        
        fig = go.Figure()
        
        # Step 4 bars
        fig.add_trace(go.Bar(
            y=features,
            x=step4_vals,
            name='Step 4 (ML Opt)',
            orientation='h',
            marker=dict(color='#6366f1'),
            hovertemplate='<b>%{y}</b><br>Step 4: %{x:.2f}%<extra></extra>'
        ))
        
        # Step 5 bars
        fig.add_trace(go.Bar(
            y=features,
            x=step5_vals,
            name='Step 5 (Anti-Overfit)',
            orientation='h',
            marker=dict(color='#22c55e'),
            hovertemplate='<b>%{y}</b><br>Step 5: %{x:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            autosize=True,
            height=plot_height,
            title=dict(
                text="Feature Importance: Step 4 vs Step 5",
                font=dict(color="#e8e8e8", size=16)
            ),
            barmode='group',
            xaxis=dict(
                title="Importance %",
                color="#8a9099",
                gridcolor="#3a3f45"
            ),
            yaxis=dict(
                color="#8a9099",
                tickfont=dict(size=10)
            ),
            plot_bgcolor="#2a2e33",
            paper_bgcolor="#1a1d21",
            font=dict(color="#e8e8e8"),
            margin=dict(l=180, r=30, t=50, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color="#e8e8e8")
            )
        )
        
        return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})
        
    except Exception as e:
        print(f"Error generating drift comparison chart: {e}")
        return None


def generate_drift_indicator_chart(
    drift_path: str = 'feature_drift_step4_to_step5.json',
    plot_height: int = 300
) -> Optional[str]:
    """
    Generate drift indicator gauge showing overall drift score.
    """
    try:
        import plotly.graph_objects as go
        
        drift_data = load_feature_importance(drift_path)
        if drift_data is None:
            return None
        
        drift_score = drift_data.get('drift_score', 0)
        threshold = drift_data.get('alert_threshold', 0.15)
        status = drift_data.get('status', 'unknown')
        
        # Color based on status
        if status == 'stable':
            color = '#22c55e'  # Green
        elif status == 'drift_detected':
            color = '#f59e0b'  # Yellow/Orange
        else:
            color = '#ef4444'  # Red
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=drift_score * 100,
            title={'text': "Feature Drift Score", 'font': {'color': '#e8e8e8'}},
            number={'suffix': '%', 'font': {'color': '#e8e8e8'}},
            gauge={
                'axis': {'range': [0, 30], 'tickcolor': '#8a9099'},
                'bar': {'color': color},
                'bgcolor': '#2a2e33',
                'bordercolor': '#3a3f45',
                'steps': [
                    {'range': [0, threshold * 100], 'color': '#1a3d1a'},
                    {'range': [threshold * 100, 30], 'color': '#3d1a1a'}
                ],
                'threshold': {
                    'line': {'color': '#ef4444', 'width': 2},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        
        fig.update_layout(
            autosize=True,
            height=plot_height,
            plot_bgcolor="#1a1d21",
            paper_bgcolor="#1a1d21",
            font=dict(color="#e8e8e8"),
            margin=dict(l=30, r=30, t=60, b=30)
        )
        
        return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})
        
    except Exception as e:
        print(f"Error generating drift indicator: {e}")
        return None


def generate_top_movers_chart(
    drift_path: str = 'feature_drift_step4_to_step5.json',
    top_n: int = 10,
    plot_height: int = 400
) -> Optional[str]:
    """
    Generate chart showing features with biggest importance changes.
    """
    try:
        import plotly.graph_objects as go
        
        drift_data = load_feature_importance(drift_path)
        if drift_data is None:
            return None
        
        delta = drift_data.get('delta', {})
        if not delta:
            return None
        
        # Sort by absolute change
        sorted_items = sorted(delta.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        
        # Reverse for display
        sorted_items = sorted_items[::-1]
        
        features = [item[0] for item in sorted_items]
        changes = [item[1] * 100 for item in sorted_items]
        
        # Color based on direction
        colors = ['#22c55e' if c > 0 else '#ef4444' for c in changes]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=features,
            x=changes,
            orientation='h',
            marker=dict(color=colors),
            hovertemplate='<b>%{y}</b><br>Change: %{x:+.2f}%<extra></extra>'
        ))
        
        # Add zero line
        fig.add_vline(x=0, line_width=2, line_color="#8a9099")
        
        fig.update_layout(
            autosize=True,
            height=plot_height,
            title=dict(
                text="Top Feature Changes (Step 4 → Step 5)",
                font=dict(color="#e8e8e8", size=16)
            ),
            xaxis=dict(
                title="Change %",
                color="#8a9099",
                gridcolor="#3a3f45",
                zeroline=True,
                zerolinecolor="#8a9099"
            ),
            yaxis=dict(
                color="#8a9099",
                tickfont=dict(size=10)
            ),
            plot_bgcolor="#2a2e33",
            paper_bgcolor="#1a1d21",
            font=dict(color="#e8e8e8"),
            margin=dict(l=180, r=30, t=50, b=40)
        )
        
        return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})
        
    except Exception as e:
        print(f"Error generating top movers chart: {e}")
        return None


def generate_category_breakdown_chart(
    filepath: str = None,
    importance_dict: Dict[str, float] = None,
    plot_height: int = 350
) -> Optional[str]:
    """
    Generate pie chart showing statistical vs global feature weights.
    """
    try:
        import plotly.graph_objects as go
        from feature_importance import get_importance_summary_for_agent
        
        # Load data
        if importance_dict is None:
            if filepath is None:
                filepath = 'feature_importance_step5.json'
            data = load_feature_importance(filepath)
            if data is None:
                return None
            importance_dict = data.get('feature_importance', data)
        
        if not importance_dict:
            return None
        
        summary = get_importance_summary_for_agent(importance_dict)
        
        stat_weight = summary.get('statistical_weight', 0.5)
        global_weight = summary.get('global_weight', 0.5)
        
        fig = go.Figure(data=[go.Pie(
            labels=['Statistical Features', 'Global State Features'],
            values=[stat_weight * 100, global_weight * 100],
            marker=dict(colors=['#3b82f6', '#8b5cf6']),
            hole=0.4,
            hovertemplate='<b>%{label}</b><br>%{value:.1f}%<extra></extra>'
        )])
        
        fig.update_layout(
            autosize=True,
            height=plot_height,
            title=dict(
                text="Feature Category Breakdown",
                font=dict(color="#e8e8e8", size=16)
            ),
            plot_bgcolor="#1a1d21",
            paper_bgcolor="#1a1d21",
            font=dict(color="#e8e8e8"),
            margin=dict(l=30, r=30, t=50, b=30),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5,
                font=dict(color="#e8e8e8")
            )
        )
        
        return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})
        
    except Exception as e:
        print(f"Error generating category breakdown: {e}")
        return None


# =============================================================================
# Additional Chart Types
# =============================================================================

def generate_importance_scatter_chart(
    step4_path: str = 'feature_importance_step4.json',
    step5_path: str = 'feature_importance_step5.json',
    plot_height: int = 500
) -> Optional[str]:
    """
    Generate scatter plot: Step 4 importance (x) vs Step 5 importance (y).
    Points above diagonal gained importance, below lost importance.
    """
    try:
        import plotly.graph_objects as go
        
        step4_data = load_feature_importance(step4_path)
        step5_data = load_feature_importance(step5_path)
        
        if step4_data is None or step5_data is None:
            return None
        
        step4_imp = step4_data.get('feature_importance', step4_data)
        step5_imp = step5_data.get('feature_importance', step5_data)
        
        # Get common features
        common_features = set(step4_imp.keys()) & set(step5_imp.keys())
        
        features = list(common_features)
        x_vals = [step4_imp[f] * 100 for f in features]
        y_vals = [step5_imp[f] * 100 for f in features]
        
        # Color by change direction
        colors = []
        for f in features:
            delta = step5_imp[f] - step4_imp[f]
            if delta > 0.01:
                colors.append('#22c55e')  # Green - gained
            elif delta < -0.01:
                colors.append('#ef4444')  # Red - lost
            else:
                colors.append('#8a9099')  # Gray - stable
        
        fig = go.Figure()
        
        # Diagonal line (no change)
        max_val = max(max(x_vals), max(y_vals)) * 1.1
        fig.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            line=dict(color='#8a9099', dash='dash', width=1),
            name='No Change',
            hoverinfo='skip'
        ))
        
        # Scatter points
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers',
            marker=dict(
                size=10,
                color=colors,
                line=dict(width=1, color='#1a1d21')
            ),
            text=features,
            hovertemplate='<b>%{text}</b><br>Step 4: %{x:.2f}%<br>Step 5: %{y:.2f}%<extra></extra>',
            name='Features'
        ))
        
        fig.update_layout(
            autosize=True,
            height=plot_height,
            title=dict(
                text="Feature Importance: Step 4 vs Step 5 Scatter",
                font=dict(color="#e8e8e8", size=16)
            ),
            xaxis=dict(
                title="Step 4 Importance %",
                color="#8a9099",
                gridcolor="#3a3f45",
                range=[0, max_val]
            ),
            yaxis=dict(
                title="Step 5 Importance %",
                color="#8a9099",
                gridcolor="#3a3f45",
                range=[0, max_val]
            ),
            plot_bgcolor="#2a2e33",
            paper_bgcolor="#1a1d21",
            font=dict(color="#e8e8e8"),
            margin=dict(l=60, r=30, t=50, b=50),
            showlegend=False
        )
        
        # Add annotations
        fig.add_annotation(
            x=max_val * 0.2, y=max_val * 0.8,
            text="↑ Gained Importance",
            showarrow=False,
            font=dict(color="#22c55e", size=11)
        )
        fig.add_annotation(
            x=max_val * 0.8, y=max_val * 0.2,
            text="↓ Lost Importance",
            showarrow=False,
            font=dict(color="#ef4444", size=11)
        )
        
        return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})
        
    except Exception as e:
        print(f"Error generating scatter chart: {e}")
        return None


def generate_importance_treemap(
    filepath: str = None,
    importance_dict: Dict[str, float] = None,
    plot_height: int = 500
) -> Optional[str]:
    """
    Generate treemap visualization of feature importance by category.
    """
    try:
        import plotly.graph_objects as go
        
        # Load data
        if importance_dict is None:
            if filepath is None:
                filepath = 'feature_importance_step5.json'
            data = load_feature_importance(filepath)
            if data is None:
                return None
            importance_dict = data.get('feature_importance', data)
        
        if not importance_dict:
            return None
        
        # Categorize features
        statistical_features = []
        global_features = []
        
        global_prefixes = ['entropy_', 'bias_', 'regime_', 'marker_', 'reseed_', 'global_']
        
        for feature, importance in importance_dict.items():
            is_global = any(feature.startswith(p) or feature.endswith(p.rstrip('_')) for p in global_prefixes)
            if is_global or feature in ['entropy_mean', 'entropy_std', 'bias_ratio_mean', 'regime_age', 'reseed_probability']:
                global_features.append((feature, importance))
            else:
                statistical_features.append((feature, importance))
        
        # Build treemap data
        labels = ['All Features', 'Statistical', 'Global State']
        parents = ['', 'All Features', 'All Features']
        values = [0, sum(v for _, v in statistical_features), sum(v for _, v in global_features)]
        colors = ['#1a1d21', '#3b82f6', '#8b5cf6']
        
        # Add individual features
        for feature, importance in sorted(statistical_features, key=lambda x: x[1], reverse=True)[:15]:
            labels.append(feature)
            parents.append('Statistical')
            values.append(importance)
            colors.append('#3b82f6')
        
        for feature, importance in sorted(global_features, key=lambda x: x[1], reverse=True)[:10]:
            labels.append(feature)
            parents.append('Global State')
            values.append(importance)
            colors.append('#8b5cf6')
        
        fig = go.Figure(go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=colors),
            textinfo='label+percent parent',
            hovertemplate='<b>%{label}</b><br>Importance: %{value:.4f}<br>%{percentParent:.1%} of parent<extra></extra>'
        ))
        
        fig.update_layout(
            autosize=True,
            height=plot_height,
            title=dict(
                text="Feature Importance Treemap",
                font=dict(color="#e8e8e8", size=16)
            ),
            paper_bgcolor="#1a1d21",
            font=dict(color="#e8e8e8"),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})
        
    except Exception as e:
        print(f"Error generating treemap: {e}")
        return None


def generate_radar_chart(
    filepath: str = None,
    importance_dict: Dict[str, float] = None,
    top_n: int = 10,
    plot_height: int = 500
) -> Optional[str]:
    """
    Generate radar/spider chart for top N features.
    """
    try:
        import plotly.graph_objects as go
        
        # Load data
        if importance_dict is None:
            if filepath is None:
                filepath = 'feature_importance_step5.json'
            data = load_feature_importance(filepath)
            if data is None:
                return None
            importance_dict = data.get('feature_importance', data)
        
        if not importance_dict:
            return None
        
        # Get top N features
        sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        features = [item[0] for item in sorted_items]
        values = [item[1] * 100 for item in sorted_items]
        
        # Close the radar by repeating first value
        features_closed = features + [features[0]]
        values_closed = values + [values[0]]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=features_closed,
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.3)',
            line=dict(color='#3b82f6', width=2),
            hovertemplate='<b>%{theta}</b><br>Importance: %{r:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            autosize=True,
            height=plot_height,
            title=dict(
                text=f"Top {top_n} Features Radar",
                font=dict(color="#e8e8e8", size=16)
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    color="#8a9099",
                    gridcolor="#3a3f45",
                    range=[0, max(values) * 1.1]
                ),
                angularaxis=dict(
                    color="#8a9099",
                    gridcolor="#3a3f45"
                ),
                bgcolor="#2a2e33"
            ),
            paper_bgcolor="#1a1d21",
            font=dict(color="#e8e8e8", size=10),
            margin=dict(l=80, r=80, t=50, b=50),
            showlegend=False
        )
        
        return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})
        
    except Exception as e:
        print(f"Error generating radar chart: {e}")
        return None


def generate_dual_radar_chart(
    step4_path: str = 'feature_importance_step4.json',
    step5_path: str = 'feature_importance_step5.json',
    top_n: int = 10,
    plot_height: int = 500
) -> Optional[str]:
    """
    Generate overlaid radar chart comparing Step 4 vs Step 5.
    """
    try:
        import plotly.graph_objects as go
        
        step4_data = load_feature_importance(step4_path)
        step5_data = load_feature_importance(step5_path)
        
        if step4_data is None or step5_data is None:
            return None
        
        step4_imp = step4_data.get('feature_importance', step4_data)
        step5_imp = step5_data.get('feature_importance', step5_data)
        
        # Get top features from Step 5
        sorted_items = sorted(step5_imp.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features = [item[0] for item in sorted_items]
        
        step4_vals = [step4_imp.get(f, 0) * 100 for f in features]
        step5_vals = [step5_imp.get(f, 0) * 100 for f in features]
        
        # Close the radar
        features_closed = features + [features[0]]
        step4_closed = step4_vals + [step4_vals[0]]
        step5_closed = step5_vals + [step5_vals[0]]
        
        fig = go.Figure()
        
        # Step 4
        fig.add_trace(go.Scatterpolar(
            r=step4_closed,
            theta=features_closed,
            fill='toself',
            fillcolor='rgba(99, 102, 241, 0.2)',
            line=dict(color='#6366f1', width=2),
            name='Step 4',
            hovertemplate='<b>%{theta}</b><br>Step 4: %{r:.2f}%<extra></extra>'
        ))
        
        # Step 5
        fig.add_trace(go.Scatterpolar(
            r=step5_closed,
            theta=features_closed,
            fill='toself',
            fillcolor='rgba(34, 197, 94, 0.2)',
            line=dict(color='#22c55e', width=2),
            name='Step 5',
            hovertemplate='<b>%{theta}</b><br>Step 5: %{r:.2f}%<extra></extra>'
        ))
        
        max_val = max(max(step4_vals), max(step5_vals)) * 1.1
        
        fig.update_layout(
            autosize=True,
            height=plot_height,
            title=dict(
                text=f"Step 4 vs Step 5 Radar Comparison",
                font=dict(color="#e8e8e8", size=16)
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    color="#8a9099",
                    gridcolor="#3a3f45",
                    range=[0, max_val]
                ),
                angularaxis=dict(
                    color="#8a9099",
                    gridcolor="#3a3f45"
                ),
                bgcolor="#2a2e33"
            ),
            paper_bgcolor="#1a1d21",
            font=dict(color="#e8e8e8", size=10),
            margin=dict(l=80, r=80, t=50, b=50),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(color="#e8e8e8")
            )
        )
        
        return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})
        
    except Exception as e:
        print(f"Error generating dual radar chart: {e}")
        return None


def generate_importance_heatmap(
    history_dir: str = 'feature_importance_history',
    top_n: int = 20,
    plot_height: int = 600
) -> Optional[str]:
    """
    Generate heatmap of feature importance over multiple runs.
    Requires historical data stored in a directory.
    """
    try:
        import plotly.graph_objects as go
        from pathlib import Path
        
        history_path = Path(history_dir)
        
        # Try to find historical files
        if not history_path.exists():
            # Fall back to using step4 and step5 as two time points
            step4_data = load_feature_importance('feature_importance_step4.json')
            step5_data = load_feature_importance('feature_importance_step5.json')
            
            if step4_data is None and step5_data is None:
                return None
            
            runs = []
            if step4_data:
                runs.append(('Step 4', step4_data.get('feature_importance', step4_data)))
            if step5_data:
                runs.append(('Step 5', step5_data.get('feature_importance', step5_data)))
        else:
            # Load all historical files
            runs = []
            for f in sorted(history_path.glob('*.json')):
                data = load_feature_importance(str(f))
                if data:
                    timestamp = data.get('timestamp', f.stem)
                    importance = data.get('feature_importance', data)
                    runs.append((timestamp[:10], importance))
        
        if len(runs) < 1:
            return None
        
        # Get top features from most recent run
        latest_imp = runs[-1][1]
        top_features = sorted(latest_imp.items(), key=lambda x: x[1], reverse=True)[:top_n]
        feature_names = [f[0] for f in top_features]
        
        # Build heatmap matrix
        run_labels = [r[0] for r in runs]
        z_data = []
        
        for feature in feature_names:
            row = [r[1].get(feature, 0) * 100 for r in runs]
            z_data.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=run_labels,
            y=feature_names,
            colorscale='Viridis',
            hovertemplate='<b>%{y}</b><br>Run: %{x}<br>Importance: %{z:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            autosize=True,
            height=plot_height,
            title=dict(
                text="Feature Importance Over Time",
                font=dict(color="#e8e8e8", size=16)
            ),
            xaxis=dict(
                title="Run",
                color="#8a9099",
                tickangle=45
            ),
            yaxis=dict(
                title="Feature",
                color="#8a9099",
                tickfont=dict(size=9)
            ),
            plot_bgcolor="#2a2e33",
            paper_bgcolor="#1a1d21",
            font=dict(color="#e8e8e8"),
            margin=dict(l=180, r=30, t=50, b=80)
        )
        
        return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})
        
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        return None


def generate_distribution_histogram(
    filepath: str = None,
    importance_dict: Dict[str, float] = None,
    plot_height: int = 400
) -> Optional[str]:
    """
    Generate histogram showing distribution of feature importance values.
    """
    try:
        import plotly.graph_objects as go
        import numpy as np
        
        # Load data
        if importance_dict is None:
            if filepath is None:
                filepath = 'feature_importance_step5.json'
            data = load_feature_importance(filepath)
            if data is None:
                return None
            importance_dict = data.get('feature_importance', data)
        
        if not importance_dict:
            return None
        
        values = [v * 100 for v in importance_dict.values()]
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=values,
            nbinsx=20,
            marker=dict(
                color='#3b82f6',
                line=dict(color='#1a1d21', width=1)
            ),
            hovertemplate='Range: %{x:.1f}%<br>Count: %{y}<extra></extra>'
        ))
        
        # Add mean line
        mean_val = np.mean(values)
        fig.add_vline(x=mean_val, line_width=2, line_dash="dash", line_color="#22c55e",
                      annotation_text=f"Mean: {mean_val:.2f}%", annotation_position="top")
        
        fig.update_layout(
            autosize=True,
            height=plot_height,
            title=dict(
                text="Feature Importance Distribution",
                font=dict(color="#e8e8e8", size=16)
            ),
            xaxis=dict(
                title="Importance %",
                color="#8a9099",
                gridcolor="#3a3f45"
            ),
            yaxis=dict(
                title="Count",
                color="#8a9099",
                gridcolor="#3a3f45"
            ),
            plot_bgcolor="#2a2e33",
            paper_bgcolor="#1a1d21",
            font=dict(color="#e8e8e8"),
            margin=dict(l=60, r=30, t=50, b=50),
            bargap=0.1
        )
        
        return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})
        
    except Exception as e:
        print(f"Error generating histogram: {e}")
        return None


def generate_waterfall_chart(
    drift_path: str = 'feature_drift_step4_to_step5.json',
    top_n: int = 15,
    plot_height: int = 450
) -> Optional[str]:
    """
    Generate waterfall chart showing cumulative impact of feature changes.
    """
    try:
        import plotly.graph_objects as go
        
        drift_data = load_feature_importance(drift_path)
        if drift_data is None:
            return None
        
        delta = drift_data.get('delta', {})
        if not delta:
            return None
        
        # Sort by change value (not absolute)
        sorted_items = sorted(delta.items(), key=lambda x: x[1], reverse=True)
        
        # Take top gainers and top losers
        top_gainers = [(f, v) for f, v in sorted_items if v > 0][:top_n//2]
        top_losers = [(f, v) for f, v in sorted_items if v < 0][-top_n//2:]
        
        combined = top_gainers + top_losers
        
        features = [item[0] for item in combined]
        changes = [item[1] * 100 for item in combined]
        
        # Waterfall requires measure types
        measures = ['relative'] * len(changes)
        
        fig = go.Figure(go.Waterfall(
            orientation='v',
            x=features,
            y=changes,
            measure=measures,
            increasing=dict(marker=dict(color='#22c55e')),
            decreasing=dict(marker=dict(color='#ef4444')),
            connector=dict(line=dict(color='#8a9099', width=1)),
            hovertemplate='<b>%{x}</b><br>Change: %{y:+.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            autosize=True,
            height=plot_height,
            title=dict(
                text="Feature Change Waterfall",
                font=dict(color="#e8e8e8", size=16)
            ),
            xaxis=dict(
                color="#8a9099",
                tickangle=45,
                tickfont=dict(size=9)
            ),
            yaxis=dict(
                title="Change %",
                color="#8a9099",
                gridcolor="#3a3f45"
            ),
            plot_bgcolor="#2a2e33",
            paper_bgcolor="#1a1d21",
            font=dict(color="#e8e8e8"),
            margin=dict(l=60, r=30, t=50, b=100),
            showlegend=False
        )
        
        return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})
        
    except Exception as e:
        print(f"Error generating waterfall chart: {e}")
        return None


def generate_sunburst_chart(
    filepath: str = None,
    importance_dict: Dict[str, float] = None,
    plot_height: int = 500
) -> Optional[str]:
    """
    Generate sunburst chart for hierarchical feature importance visualization.
    """
    try:
        import plotly.graph_objects as go
        
        # Load data
        if importance_dict is None:
            if filepath is None:
                filepath = 'feature_importance_step5.json'
            data = load_feature_importance(filepath)
            if data is None:
                return None
            importance_dict = data.get('feature_importance', data)
        
        if not importance_dict:
            return None
        
        # Categorize features into groups
        categories = {
            'Lane Agreement': [],
            'Residue': [],
            'Skip/Temporal': [],
            'Score/Confidence': [],
            'Global State': [],
            'Other': []
        }
        
        for feature, importance in importance_dict.items():
            if 'lane_agreement' in feature:
                categories['Lane Agreement'].append((feature, importance))
            elif 'residue' in feature:
                categories['Residue'].append((feature, importance))
            elif any(x in feature for x in ['skip', 'temporal', 'window']):
                categories['Skip/Temporal'].append((feature, importance))
            elif any(x in feature for x in ['score', 'confidence', 'pred']):
                categories['Score/Confidence'].append((feature, importance))
            elif any(x in feature for x in ['entropy', 'bias', 'regime', 'reseed', 'marker']):
                categories['Global State'].append((feature, importance))
            else:
                categories['Other'].append((feature, importance))
        
        # Build sunburst data
        ids = ['Features']
        labels = ['All Features']
        parents = ['']
        values = [sum(importance_dict.values())]
        
        colors = {
            'Lane Agreement': '#3b82f6',
            'Residue': '#22c55e',
            'Skip/Temporal': '#f59e0b',
            'Score/Confidence': '#ef4444',
            'Global State': '#8b5cf6',
            'Other': '#6b7280'
        }
        
        for cat_name, features in categories.items():
            if features:
                cat_total = sum(v for _, v in features)
                ids.append(cat_name)
                labels.append(cat_name)
                parents.append('Features')
                values.append(cat_total)
                
                # Add top 5 features from each category
                for feature, importance in sorted(features, key=lambda x: x[1], reverse=True)[:5]:
                    ids.append(f"{cat_name}-{feature}")
                    labels.append(feature)
                    parents.append(cat_name)
                    values.append(importance)
        
        fig = go.Figure(go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues='total',
            hovertemplate='<b>%{label}</b><br>Importance: %{value:.4f}<extra></extra>',
            marker=dict(
                colors=[colors.get(p, '#3b82f6') if p in colors else '#2a2e33' for p in parents]
            )
        ))
        
        fig.update_layout(
            autosize=True,
            height=plot_height,
            title=dict(
                text="Feature Importance Sunburst",
                font=dict(color="#e8e8e8", size=16)
            ),
            paper_bgcolor="#1a1d21",
            font=dict(color="#e8e8e8"),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})
        
    except Exception as e:
        print(f"Error generating sunburst chart: {e}")
        return None


# =============================================================================
# Dashboard Integration Helpers
# =============================================================================

def get_feature_summary_html() -> str:
    """
    Generate HTML summary card for feature importance status.
    Used in dashboard overview.
    """
    try:
        # Try to load latest importance file
        step5_data = load_feature_importance('feature_importance_step5.json')
        drift_data = load_feature_importance('feature_drift_step4_to_step5.json')
        
        if step5_data is None:
            return """
            <div style="padding: 15px; background: #2a2e33; border-radius: 8px;">
                <h4 style="color: #8a9099; margin: 0;">Feature Importance</h4>
                <p style="color: #6b7280; margin-top: 8px;">No feature importance data available. Run Step 4 or Step 5 first.</p>
            </div>
            """
        
        importance = step5_data.get('feature_importance', {})
        top_features = list(importance.keys())[:5]
        
        drift_status = "N/A"
        drift_color = "#8a9099"
        if drift_data:
            drift_status = drift_data.get('status', 'unknown')
            if drift_status == 'stable':
                drift_color = "#22c55e"
            elif drift_status == 'drift_detected':
                drift_color = "#f59e0b"
        
        features_html = "".join([f"<li style='color: #e8e8e8;'>{f}</li>" for f in top_features])
        
        return f"""
        <div style="padding: 15px; background: #2a2e33; border-radius: 8px;">
            <h4 style="color: #e8e8e8; margin: 0 0 10px 0;">🎯 Feature Importance</h4>
            <div style="display: flex; gap: 20px;">
                <div>
                    <span style="color: #8a9099; font-size: 12px;">Top Features:</span>
                    <ol style="margin: 5px 0; padding-left: 20px; font-size: 13px;">
                        {features_html}
                    </ol>
                </div>
                <div>
                    <span style="color: #8a9099; font-size: 12px;">Drift Status:</span>
                    <div style="color: {drift_color}; font-size: 18px; font-weight: bold; margin-top: 5px;">
                        {drift_status.upper()}
                    </div>
                </div>
            </div>
        </div>
        """
    except Exception as e:
        return f"""
        <div style="padding: 15px; background: #2a2e33; border-radius: 8px;">
            <h4 style="color: #ef4444; margin: 0;">Feature Importance Error</h4>
            <p style="color: #6b7280; margin-top: 8px;">{str(e)}</p>
        </div>
        """


# =============================================================================
# CLI / Standalone Mode
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature Importance Visualizer')
    parser.add_argument('--step4', default='feature_importance_step4.json', help='Step 4 importance file')
    parser.add_argument('--step5', default='feature_importance_step5.json', help='Step 5 importance file')
    parser.add_argument('--drift', default='feature_drift_step4_to_step5.json', help='Drift analysis file')
    parser.add_argument('--output', default='feature_report.html', help='Output HTML file')
    parser.add_argument('--top-n', type=int, default=20, help='Number of top features to show')
    parser.add_argument('--charts', default='all', help='Charts to generate: all, basic, comparison, or comma-separated list')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Feature Importance Visualizer v1.0.0")
    print("=" * 60)
    
    # Define all available charts
    all_charts = {
        'importance_bar': ('Feature Importance (Bar)', lambda: generate_feature_importance_chart(args.step5, top_n=args.top_n)),
        'drift_comparison': ('Step 4 vs Step 5 Comparison', lambda: generate_drift_comparison_chart(args.step4, args.step5, top_n=15)),
        'drift_gauge': ('Drift Score Gauge', lambda: generate_drift_indicator_chart(args.drift)),
        'top_movers': ('Top Feature Changes', lambda: generate_top_movers_chart(args.drift)),
        'category_pie': ('Category Breakdown', lambda: generate_category_breakdown_chart(args.step5)),
        'scatter': ('Step 4 vs Step 5 Scatter', lambda: generate_importance_scatter_chart(args.step4, args.step5)),
        'treemap': ('Feature Treemap', lambda: generate_importance_treemap(args.step5)),
        'radar': ('Top Features Radar', lambda: generate_radar_chart(args.step5, top_n=10)),
        'dual_radar': ('Step 4 vs Step 5 Radar', lambda: generate_dual_radar_chart(args.step4, args.step5, top_n=10)),
        'heatmap': ('Importance Heatmap', lambda: generate_importance_heatmap()),
        'histogram': ('Importance Distribution', lambda: generate_distribution_histogram(args.step5)),
        'waterfall': ('Change Waterfall', lambda: generate_waterfall_chart(args.drift)),
        'sunburst': ('Feature Sunburst', lambda: generate_sunburst_chart(args.step5)),
    }
    
    # Determine which charts to generate
    if args.charts == 'all':
        charts_to_generate = list(all_charts.keys())
    elif args.charts == 'basic':
        charts_to_generate = ['importance_bar', 'category_pie', 'histogram', 'radar']
    elif args.charts == 'comparison':
        charts_to_generate = ['drift_comparison', 'scatter', 'dual_radar', 'top_movers', 'waterfall']
    else:
        charts_to_generate = [c.strip() for c in args.charts.split(',')]
    
    print(f"\nAvailable charts: {len(all_charts)}")
    print(f"Generating: {len(charts_to_generate)} charts")
    
    # Generate charts
    charts = []
    print("\nGenerating charts...")
    
    for chart_id in charts_to_generate:
        if chart_id not in all_charts:
            print(f"  ⚠️  Unknown chart: {chart_id}")
            continue
        
        title, generator = all_charts[chart_id]
        try:
            result = generator()
            if result:
                charts.append((title, result))
                print(f"  ✅ {title}")
            else:
                print(f"  ⚠️  {title} - no data")
        except Exception as e:
            print(f"  ❌ {title} - error: {e}")
    
    if not charts:
        print("\n❌ No charts generated. Make sure feature importance files exist.")
        print(f"   Expected: {args.step4}, {args.step5}, {args.drift}")
        exit(1)
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Feature Importance Report</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background-color: #1a1d21;
                color: #e8e8e8;
                margin: 0;
                padding: 20px;
            }}
            h1 {{
                color: #e8e8e8;
                border-bottom: 2px solid #3b82f6;
                padding-bottom: 10px;
            }}
            .chart-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                gap: 20px;
            }}
            .chart-container {{
                background: #2a2e33;
                border-radius: 8px;
                padding: 15px;
            }}
            .chart-container.full-width {{
                grid-column: 1 / -1;
            }}
            .chart-title {{
                color: #8a9099;
                font-size: 14px;
                margin-bottom: 10px;
            }}
            .timestamp {{
                color: #6b7280;
                font-size: 12px;
                margin-top: 20px;
            }}
            .summary {{
                background: #2a2e33;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>🎯 Feature Importance Report</h1>
        <div class="summary">
            <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p style="color: #8a9099;">Charts generated: {len(charts)} / {len(charts_to_generate)} requested</p>
        </div>
        <div class="chart-grid">
    """
    
    # Charts that should be full width
    full_width_charts = ['Importance Heatmap', 'Feature Treemap', 'Feature Sunburst', 'Step 4 vs Step 5 Comparison']
    
    for title, chart_html in charts:
        css_class = "chart-container full-width" if title in full_width_charts else "chart-container"
        html_content += f"""
            <div class="{css_class}">
                <div class="chart-title">{title}</div>
                {chart_html}
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Save report
    with open(args.output, 'w') as f:
        f.write(html_content)
    
    print(f"\n✅ Report saved to: {args.output}")
    print(f"   Open in browser: file://{os.path.abspath(args.output)}")
    print(f"\n📊 Available chart IDs for --charts flag:")
    for chart_id in all_charts.keys():
        print(f"   - {chart_id}")
