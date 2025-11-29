#!/usr/bin/env python3
"""
Model Health Check & Normalization Diagnostic
==============================================

Validates reinforcement engine model health and feature normalization.
Runs automatically before training or can be run manually.

Author: Distributed PRNG Analysis System
Date: November 7, 2025
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    from reinforcement_engine import ReinforcementEngine, ReinforcementConfig
except ImportError:
    print("ERROR: Cannot import reinforcement_engine")
    sys.exit(1)


class ModelHealthCheck:
    """Diagnostic tool for model and feature health"""
    
    def __init__(self, engine: ReinforcementEngine):
        self.engine = engine
        self.results = {}
        
    def run_full_diagnostic(self, test_survivors: Optional[List[int]] = None) -> Dict:
        """Run complete diagnostic suite"""
        
        print("="*70)
        print("MODEL HEALTH DIAGNOSTIC")
        print("="*70)
        print()
        
        # Generate test survivors if not provided
        if test_survivors is None:
            test_survivors = np.random.randint(1, 100000, 100).tolist()
        
        # Run all checks
        self.check_feature_normalization(test_survivors[:10])
        self.check_model_parameters()
        self.check_prediction_variance(test_survivors)
        self.check_gradient_flow()
        self.check_model_saturation(test_survivors)
        
        # Generate report
        return self.generate_report()
    
    def check_feature_normalization(self, test_survivors: List[int]) -> Dict:
        """Check if features are properly normalized"""
        
        print("🔍 Feature Normalization Check")
        print("-" * 70)
        
        features_list = []
        for seed in test_survivors:
            features = self.engine.extract_combined_features(seed)
            features_list.append(features)
        
        features_array = np.array(features_list)
        
        # Calculate statistics
        means = features_array.mean(axis=0)
        stds = features_array.std(axis=0)
        mins = features_array.min(axis=0)
        maxs = features_array.max(axis=0)
        
        # Check for normalization
        mean_of_means = np.mean(means)
        mean_of_stds = np.mean(stds)
        max_range = (maxs - mins).max()
        
        print(f"  Feature Statistics:")
        print(f"    Mean across features: {mean_of_means:.4f}")
        print(f"    Std across features: {mean_of_stds:.4f}")
        print(f"    Max feature range: {max_range:.2f}")
        print(f"    Min feature value: {mins.min():.2f}")
        print(f"    Max feature value: {maxs.max():.2f}")
        
        # Determine if normalized
        is_normalized = (
            abs(mean_of_means) < 10 and  # Mean close to 0
            max_range < 100  # Range reasonable
        )
        
        if is_normalized:
            status = "✅ PASS"
            recommendation = "Features appear normalized"
        else:
            status = "❌ FAIL"
            recommendation = "Features NOT normalized - model will not learn properly"
        
        print(f"\n  Status: {status}")
        print(f"  Recommendation: {recommendation}")
        print()
        
        self.results['normalization'] = {
            'status': 'pass' if is_normalized else 'fail',
            'mean_of_means': float(mean_of_means),
            'mean_of_stds': float(mean_of_stds),
            'max_range': float(max_range),
            'recommendation': recommendation
        }
        
        return self.results['normalization']
    
    def check_model_parameters(self) -> Dict:
        """Check model parameter health"""
        
        print("🔍 Model Parameter Check")
        print("-" * 70)
        
        total_params = sum(p.numel() for p in self.engine.model.parameters())
        
        # Check for NaN or Inf
        has_nan = False
        has_inf = False
        param_stats = []
        
        for name, param in self.engine.model.named_parameters():
            if torch.isnan(param).any():
                has_nan = True
            if torch.isinf(param).any():
                has_inf = True
            
            param_stats.append({
                'name': name,
                'shape': list(param.shape),
                'mean': param.mean().item(),
                'std': param.std().item(),
                'min': param.min().item(),
                'max': param.max().item()
            })
        
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Has NaN values: {'Yes ❌' if has_nan else 'No ✅'}")
        print(f"  Has Inf values: {'Yes ❌' if has_inf else 'No ✅'}")
        
        # Check first layer (most important)
        first_layer = param_stats[0]
        print(f"\n  First Layer ({first_layer['name']}):")
        print(f"    Shape: {first_layer['shape']}")
        print(f"    Mean: {first_layer['mean']:.6f}")
        print(f"    Std: {first_layer['std']:.6f}")
        
        is_healthy = not (has_nan or has_inf)
        status = "✅ PASS" if is_healthy else "❌ FAIL"
        
        print(f"\n  Status: {status}")
        print()
        
        self.results['parameters'] = {
            'status': 'pass' if is_healthy else 'fail',
            'total_params': total_params,
            'has_nan': has_nan,
            'has_inf': has_inf,
            'first_layer_stats': first_layer
        }
        
        return self.results['parameters']
    
    def check_prediction_variance(self, test_survivors: List[int]) -> Dict:
        """Check if model produces varied predictions"""
        
        print("🔍 Prediction Variance Check")
        print("-" * 70)
        
        qualities = self.engine.predict_quality_batch(test_survivors)
        qualities = np.array(qualities)
        
        unique_values = len(np.unique(qualities))
        mean_quality = qualities.mean()
        std_quality = qualities.std()
        min_quality = qualities.min()
        max_quality = qualities.max()
        
        print(f"  Prediction Statistics:")
        print(f"    Mean: {mean_quality:.6f}")
        print(f"    Std: {std_quality:.6f}")
        print(f"    Min: {min_quality:.6f}")
        print(f"    Max: {max_quality:.6f}")
        print(f"    Unique values: {unique_values} / {len(qualities)}")
        
        # Check for saturation
        saturated_at_zero = (qualities < 0.01).sum() / len(qualities)
        saturated_at_one = (qualities > 0.99).sum() / len(qualities)
        
        print(f"    Saturated at 0: {saturated_at_zero*100:.1f}%")
        print(f"    Saturated at 1: {saturated_at_one*100:.1f}%")
        
        # Determine health
        has_variance = std_quality > 0.01
        not_saturated = saturated_at_one < 0.9 and saturated_at_zero < 0.9
        
        is_healthy = has_variance and not_saturated
        status = "✅ PASS" if is_healthy else "❌ FAIL"
        
        if not has_variance:
            recommendation = "No prediction variance - model not learning"
        elif saturated_at_one > 0.9:
            recommendation = "Saturated at 1.0 - features need normalization"
        elif saturated_at_zero > 0.9:
            recommendation = "Saturated at 0.0 - check training data"
        else:
            recommendation = "Healthy prediction variance"
        
        print(f"\n  Status: {status}")
        print(f"  Recommendation: {recommendation}")
        print()
        
        self.results['variance'] = {
            'status': 'pass' if is_healthy else 'fail',
            'mean': float(mean_quality),
            'std': float(std_quality),
            'unique_values': int(unique_values),
            'saturated_at_one': float(saturated_at_one),
            'recommendation': recommendation
        }
        
        return self.results['variance']
    
    def check_gradient_flow(self) -> Dict:
        """Check if gradients can flow through model"""
        
        print("🔍 Gradient Flow Check")
        print("-" * 70)
        
        self.engine.model.train()
        
        # Create dummy batch
        dummy_input = torch.randn(10, 60).to(self.engine.device)
        dummy_target = torch.rand(10, 1).to(self.engine.device)
        
        # Forward pass
        output = self.engine.model(dummy_input)
        loss = torch.nn.functional.mse_loss(output, dummy_target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        has_gradients = False
        gradient_stats = []
        
        for name, param in self.engine.model.named_parameters():
            if param.grad is not None:
                has_gradients = True
                grad_norm = param.grad.norm().item()
                gradient_stats.append({
                    'name': name,
                    'grad_norm': grad_norm
                })
        
        print(f"  Gradients computed: {'Yes ✅' if has_gradients else 'No ❌'}")
        
        if gradient_stats:
            avg_grad_norm = np.mean([g['grad_norm'] for g in gradient_stats])
            print(f"  Average gradient norm: {avg_grad_norm:.6f}")
        
        is_healthy = has_gradients
        status = "✅ PASS" if is_healthy else "❌ FAIL"
        
        print(f"\n  Status: {status}")
        print()
        
        self.results['gradients'] = {
            'status': 'pass' if is_healthy else 'fail',
            'has_gradients': has_gradients
        }
        
        self.engine.model.eval()
        return self.results['gradients']
    
    def check_model_saturation(self, test_survivors: List[int]) -> Dict:
        """Check raw model outputs for saturation"""
        
        print("🔍 Model Saturation Check")
        print("-" * 70)
        
        self.engine.model.eval()
        
        raw_outputs = []
        with torch.no_grad():
            for seed in test_survivors[:20]:
                features = self.engine.extract_combined_features(seed)
                feat_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.engine.device)
                raw_out = self.engine.model(feat_tensor).item()
                raw_outputs.append(raw_out)
        
        raw_outputs = np.array(raw_outputs)
        
        print(f"  Raw Output Statistics (before sigmoid):")
        print(f"    Mean: {raw_outputs.mean():.6f}")
        print(f"    Std: {raw_outputs.std():.6f}")
        print(f"    Min: {raw_outputs.min():.6f}")
        print(f"    Max: {raw_outputs.max():.6f}")
        
        # Check for saturation
        saturated = abs(raw_outputs.mean()) > 10
        
        if saturated:
            status = "❌ FAIL"
            recommendation = "Raw outputs too extreme - sigmoid will saturate"
        else:
            status = "✅ PASS"
            recommendation = "Raw outputs in healthy range"
        
        print(f"\n  Status: {status}")
        print(f"  Recommendation: {recommendation}")
        print()
        
        self.results['saturation'] = {
            'status': 'pass' if not saturated else 'fail',
            'raw_mean': float(raw_outputs.mean()),
            'raw_std': float(raw_outputs.std()),
            'recommendation': recommendation
        }
        
        return self.results['saturation']
    
    def generate_report(self) -> Dict:
        """Generate final diagnostic report"""
        
        print("="*70)
        print("DIAGNOSTIC SUMMARY")
        print("="*70)
        print()
        
        all_passed = all(r.get('status') == 'pass' for r in self.results.values())
        
        for check_name, result in self.results.items():
            status_symbol = "✅" if result['status'] == 'pass' else "❌"
            print(f"{status_symbol} {check_name.upper()}: {result['status'].upper()}")
            if 'recommendation' in result:
                print(f"   → {result['recommendation']}")
        
        print()
        print("="*70)
        
        if all_passed:
            print("✅ ALL CHECKS PASSED - Model is healthy")
        else:
            print("❌ SOME CHECKS FAILED - Action required")
            print("\nRecommended Actions:")
            for check_name, result in self.results.items():
                if result['status'] == 'fail':
                    print(f"  • {check_name}: {result.get('recommendation', 'Review and fix')}")
        
        print("="*70)
        print()
        
        return {
            'overall_status': 'pass' if all_passed else 'fail',
            'checks': self.results
        }


def main():
    """Run model health check"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Health Check & Diagnostics')
    parser.add_argument('--lottery-data', type=str, default='daily3.json',
                       help='Lottery data file')
    parser.add_argument('--config', type=str, default='reinforcement_engine_config.json',
                       help='Config file')
    parser.add_argument('--output', type=str, default='model_health_report.json',
                       help='Output report file')
    parser.add_argument('--survivors', type=int, default=100,
                       help='Number of test survivors')
    
    args = parser.parse_args()
    
    # Load lottery data
    print(f"Loading lottery data from {args.lottery_data}...")
    with open(args.lottery_data) as f:
        lottery_data = json.load(f)
    lottery_history = [d['draw'] for d in lottery_data]
    
    # Initialize engine
    print(f"Initializing engine...")
    config = ReinforcementConfig.from_json(args.config)
    engine = ReinforcementEngine(config, lottery_history)
    
    # Run diagnostics
    checker = ModelHealthCheck(engine)
    
    # Generate test survivors
    test_survivors = np.random.randint(1, 100000, args.survivors).tolist()
    
    # Run full diagnostic
    report = checker.run_full_diagnostic(test_survivors)
    
    # Save report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to {args.output}")
    
    # Return exit code based on status
    sys.exit(0 if report['overall_status'] == 'pass' else 1)


if __name__ == '__main__':
    main()
