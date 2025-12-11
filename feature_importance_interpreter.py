#!/usr/bin/env python3
"""
Feature Importance Interpreter
Generates human-readable insights from feature importance data.
Can be used standalone or fed to an LLM for deeper analysis.
"""

import json
from pathlib import Path

def interpret_feature_importance(step5_path='feature_importance_step5.json', 
                                  drift_path='feature_drift_step4_to_step5.json'):
    """Generate human-readable interpretation of feature importance."""
    
    # Load data
    step5_data = json.load(open(step5_path)) if Path(step5_path).exists() else None
    drift_data = json.load(open(drift_path)) if Path(drift_path).exists() else None
    
    if not step5_data:
        return "No feature importance data available."
    
    importance = step5_data.get('feature_importance', {})
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    # Categorize features
    lane_features = [(f, v) for f, v in sorted_features if 'lane' in f]
    entropy_features = [(f, v) for f, v in sorted_features if 'entropy' in f]
    temporal_features = [(f, v) for f, v in sorted_features if 'temporal' in f or 'skip' in f]
    score_features = [(f, v) for f, v in sorted_features if 'score' in f or 'confidence' in f]
    
    # Calculate category weights
    total = sum(v for _, v in sorted_features)
    lane_weight = sum(v for _, v in lane_features) / total * 100
    entropy_weight = sum(v for _, v in entropy_features) / total * 100
    temporal_weight = sum(v for _, v in temporal_features) / total * 100
    
    report = []
    report.append("=" * 60)
    report.append("🎯 FEATURE IMPORTANCE INTERPRETATION")
    report.append("=" * 60)
    report.append("")
    
    # Top features explanation
    report.append("📊 TOP 5 MOST IMPORTANT FEATURES:")
    report.append("-" * 40)
    for i, (feature, value) in enumerate(sorted_features[:5], 1):
        pct = value * 100
        explanation = get_feature_explanation(feature)
        report.append(f"{i}. {feature}: {pct:.1f}%")
        report.append(f"   → {explanation}")
    report.append("")
    
    # Category insights
    report.append("📈 CATEGORY BREAKDOWN:")
    report.append("-" * 40)
    report.append(f"• Lane Agreement features: {lane_weight:.1f}% of model decisions")
    report.append(f"• Entropy/Randomness features: {entropy_weight:.1f}% of model decisions")
    report.append(f"• Temporal/Skip features: {temporal_weight:.1f}% of model decisions")
    report.append("")
    
    # Key insight
    top_feature = sorted_features[0][0]
    report.append("💡 KEY INSIGHT:")
    report.append("-" * 40)
    if 'lane' in top_feature:
        report.append("The model relies heavily on LANE AGREEMENT patterns.")
        report.append("This suggests the PRNG has detectable bit-level correlations")
        report.append("across different output positions.")
    elif 'entropy' in top_feature or 'skip' in top_feature:
        report.append("The model relies heavily on ENTROPY/SKIP patterns.")
        report.append("This suggests the PRNG has detectable patterns in its")
        report.append("output distribution or skip behavior.")
    elif 'temporal' in top_feature:
        report.append("The model relies heavily on TEMPORAL patterns.")
        report.append("This suggests the PRNG state evolves predictably over time.")
    report.append("")
    
    # Drift analysis
    if drift_data:
        report.append("🔄 DRIFT ANALYSIS (Step 4 → Step 5):")
        report.append("-" * 40)
        drift_score = drift_data.get('drift_score', 0)
        status = drift_data.get('status', 'unknown')
        
        if status == 'stable':
            report.append(f"✅ Status: STABLE (drift score: {drift_score:.3f})")
            report.append("   The model's feature priorities remained consistent")
            report.append("   during anti-overfit training. This is good!")
        else:
            report.append(f"⚠️ Status: DRIFT DETECTED (drift score: {drift_score:.3f})")
            report.append("   Feature priorities shifted during training.")
            report.append("   Consider investigating model stability.")
        
        report.append("")
        gainers = drift_data.get('top_gainers', [])[:3]
        losers = drift_data.get('top_losers', [])[:3]
        
        if gainers:
            report.append(f"📈 Features that GAINED importance: {', '.join(gainers)}")
        if losers:
            report.append(f"📉 Features that LOST importance: {', '.join(losers)}")
    
    report.append("")
    report.append("=" * 60)
    report.append("END OF INTERPRETATION")
    report.append("=" * 60)
    
    return "\n".join(report)


def get_feature_explanation(feature_name):
    """Get human-readable explanation for a feature."""
    explanations = {
        'lane_agreement': "Measures bit correlation across PRNG output positions",
        'skip_entropy': "Randomness in the skip pattern between outputs",
        'temporal_stability': "How consistent patterns are over time",
        'residue_coherence': "Modular arithmetic patterns in outputs",
        'confidence_score': "Model's self-assessed prediction confidence",
        'window_coverage': "How much of the analysis window is utilized",
        'prediction_variance': "Spread of model predictions",
        'survivor_overlap': "Agreement between forward/reverse survivors",
        'forward_match': "Success rate of forward seed prediction",
        'reverse_match': "Success rate of reverse seed prediction",
        'bidirectional': "Combined forward + reverse matching score",
        'entropy': "Overall randomness/unpredictability measure",
        'bias_ratio': "Deviation from expected uniform distribution",
        'regime': "Detected operational mode of the PRNG",
        'reseed': "Probability that PRNG was reseeded",
        'marker': "Detection of known PRNG state markers",
    }
    
    for key, explanation in explanations.items():
        if key in feature_name.lower():
            return explanation
    
    return "Statistical feature used for PRNG pattern detection"


if __name__ == "__main__":
    print(interpret_feature_importance())
