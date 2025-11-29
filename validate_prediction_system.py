#!/usr/bin/env python3
"""
Prediction System Validator
============================

Validates COMPLETE system compatibility:
✅ Config file format and structure
✅ Feature count matches survivor_scorer.py (46 features)
✅ Feature weights sum to 1.0
✅ GPU availability and setup
✅ survivor_scorer.py integration
✅ File system compatibility
✅ No hardcoded values
✅ No placeholder code

Author: PRNG Analysis System
Date: November 6, 2025
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

# ANSI colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_success(text: str):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text: str):
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_info(text: str):
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")


class SystemValidator:
    """Comprehensive system validation"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.successes = []
        
        # Expected 46 features from survivor_scorer.py
        self.EXPECTED_FEATURES = [
            # Basic (5)
            'score', 'exact_matches', 'total_predictions', 'confidence', 'best_offset',
            
            # Residue coherence (9)
            'residue_8_match_rate', 'residue_8_kl_divergence', 'residue_8_coherence',
            'residue_125_match_rate', 'residue_125_kl_divergence', 'residue_125_coherence',
            'residue_1000_match_rate', 'residue_1000_kl_divergence', 'residue_1000_coherence',
            
            # Skip entropy (4)
            'skip_entropy', 'skip_mean', 'skip_std', 'skip_range',
            
            # Temporal stability (5)
            'temporal_stability_mean', 'temporal_stability_std', 'temporal_stability_trend',
            'temporal_stability_min', 'temporal_stability_max',
            
            # Survivor velocity (2)
            'survivor_velocity', 'velocity_acceleration',
            
            # Intersection weights (8)
            'intersection_ratio', 'intersection_count', 'forward_count', 'reverse_count',
            'intersection_weight', 'survivor_overlap_ratio', 'forward_only_count', 'reverse_only_count',
            
            # Lane agreement (3)
            'lane_agreement_8', 'lane_agreement_125', 'lane_consistency',
            
            # Statistical features (10)
            'pred_mean', 'pred_std', 'pred_min', 'pred_max',
            'actual_mean', 'actual_std',
            'residual_mean', 'residual_std', 'residual_abs_mean', 'residual_max_abs'
        ]
    
    def validate_config_file(self, config_path: str) -> Tuple[bool, Dict]:
        """Validate config file structure"""
        print_header("VALIDATING CONFIG FILE")
        
        config_file = Path(config_path)
        
        # Check file exists
        if not config_file.exists():
            print_error(f"Config file not found: {config_path}")
            self.errors.append("Config file missing")
            return False, {}
        
        print_success(f"Config file exists: {config_path}")
        
        # Check file extension
        if config_file.suffix != '.json':
            print_error(f"Config must be JSON format (found: {config_file.suffix})")
            self.errors.append("Wrong config format")
            return False, {}
        
        print_success("Config format: JSON ✓")
        
        # Load and parse
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print_success("Config parsed successfully")
        except json.JSONDecodeError as e:
            print_error(f"Invalid JSON: {e}")
            self.errors.append("JSON parsing failed")
            return False, {}
        
        # Check structure
        if 'prediction' not in config:
            print_error("Missing 'prediction' section in config")
            self.errors.append("Invalid config structure")
            return False, {}
        
        print_success("Config structure valid")
        
        pred_config = config['prediction']
        
        # Check required sections
        required_sections = ['pool_size', 'feature_weights', 'gpu', 'prng', 'output']
        for section in required_sections:
            if section not in pred_config:
                print_error(f"Missing required section: {section}")
                self.errors.append(f"Missing {section}")
                return False, {}
        
        print_success(f"All required sections present: {len(required_sections)}")
        
        return True, config
    
    def validate_feature_weights(self, config: Dict) -> bool:
        """Validate feature weights"""
        print_header("VALIDATING FEATURE WEIGHTS")
        
        pred_config = config['prediction']
        feature_weights = pred_config.get('feature_weights', {})
        
        # Remove comment fields
        feature_weights = {k: v for k, v in feature_weights.items() if not k.startswith('_')}
        
        # Check feature count
        actual_count = len(feature_weights)
        expected_count = len(self.EXPECTED_FEATURES)
        
        if actual_count != expected_count:
            print_error(f"Feature count mismatch: {actual_count} (expected {expected_count})")
            self.errors.append("Wrong feature count")
            return False
        
        print_success(f"Feature count: {actual_count} ✓")
        
        # Check all expected features present
        missing_features = set(self.EXPECTED_FEATURES) - set(feature_weights.keys())
        if missing_features:
            print_error(f"Missing features: {missing_features}")
            self.errors.append("Missing features")
            return False
        
        print_success("All 46 features present ✓")
        
        # Check for unexpected features
        extra_features = set(feature_weights.keys()) - set(self.EXPECTED_FEATURES)
        if extra_features:
            print_warning(f"Extra features (will be ignored): {extra_features}")
            self.warnings.append("Extra features in config")
        
        # Check weight values
        weights_sum = sum(feature_weights.values())
        tolerance = pred_config.get('validation', {}).get('weights_tolerance', 0.001)
        
        if abs(weights_sum - 1.0) > tolerance:
            print_error(f"Feature weights sum to {weights_sum:.6f} (expected 1.0 ± {tolerance})")
            self.errors.append("Weights don't sum to 1.0")
            return False
        
        print_success(f"Feature weights sum to {weights_sum:.6f} ✓")
        
        # Check individual weight ranges
        invalid_weights = {k: v for k, v in feature_weights.items() if v < 0 or v > 1}
        if invalid_weights:
            print_error(f"Invalid weight values (must be 0-1): {invalid_weights}")
            self.errors.append("Invalid weight values")
            return False
        
        print_success("All weights in valid range [0, 1] ✓")
        
        # Display weight distribution
        print_info("\nFeature weight categories:")
        print(f"  Basic features (5): {sum(feature_weights[f] for f in self.EXPECTED_FEATURES[:5]):.4f}")
        print(f"  Residue coherence (9): {sum(feature_weights[f] for f in self.EXPECTED_FEATURES[5:14]):.4f}")
        print(f"  Skip entropy (4): {sum(feature_weights[f] for f in self.EXPECTED_FEATURES[14:18]):.4f}")
        print(f"  Temporal stability (5): {sum(feature_weights[f] for f in self.EXPECTED_FEATURES[18:23]):.4f}")
        print(f"  Survivor velocity (2): {sum(feature_weights[f] for f in self.EXPECTED_FEATURES[23:25]):.4f}")
        print(f"  Intersection weights (8): {sum(feature_weights[f] for f in self.EXPECTED_FEATURES[25:33]):.4f}")
        print(f"  Lane agreement (3): {sum(feature_weights[f] for f in self.EXPECTED_FEATURES[33:36]):.4f}")
        print(f"  Statistical features (10): {sum(feature_weights[f] for f in self.EXPECTED_FEATURES[36:46]):.4f}")
        
        return True
    
    def validate_gpu_config(self, config: Dict) -> bool:
        """Validate GPU configuration"""
        print_header("VALIDATING GPU CONFIGURATION")
        
        gpu_config = config['prediction'].get('gpu', {})
        
        # Check GPU availability
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count()
            
            if gpu_available:
                print_success(f"CUDA available: {gpu_count} GPU(s) detected")
                
                for i in range(gpu_count):
                    name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / 1e9
                    print_info(f"  GPU {i}: {name} ({memory_gb:.2f} GB)")
                
                # Validate device_ids
                device_ids = gpu_config.get('device_ids', [])
                for dev_id in device_ids:
                    if dev_id >= gpu_count:
                        print_error(f"Invalid device_id {dev_id} (only {gpu_count} GPUs available)")
                        self.errors.append("Invalid GPU device ID")
                        return False
                
                if device_ids:
                    print_success(f"Device IDs valid: {device_ids}")
                
            else:
                print_warning("CUDA not available - will use CPU")
                if gpu_config.get('use_gpu', False):
                    print_warning("Config requests GPU but none available")
                    self.warnings.append("GPU requested but unavailable")
        
        except ImportError:
            print_error("PyTorch not installed - cannot validate GPU")
            self.errors.append("PyTorch missing")
            return False
        
        return True
    
    def validate_survivor_scorer_integration(self) -> bool:
        """Validate survivor_scorer.py integration"""
        print_header("VALIDATING SURVIVOR_SCORER INTEGRATION")
        
        # Check if survivor_scorer.py exists
        scorer_path = Path('survivor_scorer.py')
        if not scorer_path.exists():
            scorer_path = Path('modules/survivor_scorer.py')
        
        if not scorer_path.exists():
            print_error("survivor_scorer.py not found")
            self.errors.append("survivor_scorer.py missing")
            return False
        
        print_success(f"survivor_scorer.py found: {scorer_path}")
        
        # Try to import
        try:
            sys.path.insert(0, str(scorer_path.parent))
            from survivor_scorer import SurvivorScorer
            print_success("SurvivorScorer imported successfully")
        except ImportError as e:
            print_error(f"Failed to import SurvivorScorer: {e}")
            self.errors.append("Import failed")
            return False
        
        # Validate required methods exist
        required_methods = [
            'score_survivor',
            'extract_ml_features',
            'build_prediction_pool',
            'rank_by_dual_confidence',
            'calculate_survivor_overlap_ratio',
            'compute_dual_sieve_intersection',
            'validate_bidirectional_consistency',
            'score_with_dual_sieve'
        ]
        
        scorer = SurvivorScorer()
        
        for method in required_methods:
            if not hasattr(scorer, method):
                print_error(f"Missing required method: {method}")
                self.errors.append(f"Missing {method}")
                return False
        
        print_success(f"All {len(required_methods)} required methods present ✓")
        
        # Test feature extraction
        try:
            test_history = [123, 456, 789, 234, 567]
            features = scorer.extract_ml_features(
                seed=42424242,
                lottery_history=test_history
            )
            
            feature_count = len(features)
            if feature_count != 46:
                print_error(f"Feature extraction returned {feature_count} features (expected 46)")
                self.errors.append("Wrong feature count from scorer")
                return False
            
            print_success(f"Feature extraction works: {feature_count} features ✓")
            
            # Check all expected features present
            missing = set(self.EXPECTED_FEATURES) - set(features.keys())
            if missing:
                print_error(f"Missing features from scorer: {missing}")
                self.errors.append("Scorer missing features")
                return False
            
            print_success("All features extracted correctly ✓")
            
        except Exception as e:
            print_error(f"Feature extraction test failed: {e}")
            self.errors.append("Feature extraction failed")
            return False
        
        return True
    
    def validate_file_system(self, config: Dict) -> bool:
        """Validate file system structure"""
        print_header("VALIDATING FILE SYSTEM")
        
        pred_config = config['prediction']
        
        # Check output directories
        output_config = pred_config.get('output', {})
        
        directories = {
            'predictions_dir': output_config.get('predictions_dir', 'results/predictions'),
            'features_dir': output_config.get('features_dir', 'results/features'),
            'log_dir': str(Path(output_config.get('log_file', 'logs/prediction_generator.log')).parent)
        }
        
        for name, dir_path in directories.items():
            dir_obj = Path(dir_path)
            
            if dir_obj.exists():
                if not dir_obj.is_dir():
                    print_error(f"{name} exists but is not a directory: {dir_path}")
                    self.errors.append(f"Invalid {name}")
                    return False
                print_success(f"{name} exists: {dir_path}")
            else:
                print_info(f"{name} will be created: {dir_path}")
                # Try to create it
                try:
                    dir_obj.mkdir(parents=True, exist_ok=True)
                    print_success(f"Created {name}: {dir_path}")
                except Exception as e:
                    print_error(f"Cannot create {name}: {e}")
                    self.errors.append(f"Cannot create {name}")
                    return False
        
        return True
    
    def validate_prediction_generator(self) -> bool:
        """Validate prediction_generator.py"""
        print_header("VALIDATING PREDICTION_GENERATOR.PY")
        
        # Check file exists
        gen_path = Path('prediction_generator.py')
        if not gen_path.exists():
            gen_path = Path('modules/prediction_generator.py')
        
        if not gen_path.exists():
            print_error("prediction_generator.py not found")
            self.errors.append("prediction_generator.py missing")
            return False
        
        print_success(f"prediction_generator.py found: {gen_path}")
        
        # Read and check for placeholders
        with open(gen_path, 'r') as f:
            content = f.read()
        
        # Check for placeholder patterns
        placeholder_patterns = [
            'pass  # TODO',
            'raise NotImplementedError',
            '# Placeholder',
            '# TO BE IMPLEMENTED',
            'return None  # TODO'
        ]
        
        found_placeholders = []
        for pattern in placeholder_patterns:
            if pattern in content:
                found_placeholders.append(pattern)
        
        if found_placeholders:
            print_error(f"Found placeholder code: {found_placeholders}")
            self.errors.append("Placeholder code detected")
            return False
        
        print_success("No placeholder code found ✓")
        
        # Check for hardcoded values (common patterns)
        hardcoded_patterns = [
            ('pool_size = 10', 'pool_size hardcoded'),
            ('mod = 1000', 'mod hardcoded (should come from config)'),
            ('k = 10', 'k value hardcoded'),
        ]
        
        found_hardcoded = []
        for pattern, desc in hardcoded_patterns:
            if pattern in content and 'config' not in content[max(0, content.find(pattern)-100):content.find(pattern)+100]:
                found_hardcoded.append(desc)
        
        if found_hardcoded:
            print_warning(f"Potentially hardcoded values: {found_hardcoded}")
            self.warnings.append("Check hardcoded values")
        else:
            print_success("No obvious hardcoded values detected ✓")
        
        # Check imports
        required_imports = ['torch', 'numpy', 'SurvivorScorer']
        for imp in required_imports:
            if f'import {imp}' not in content and f'from {imp}' not in content and f'from survivor_scorer import SurvivorScorer' not in content:
                if imp != 'SurvivorScorer' or 'from survivor_scorer import SurvivorScorer' not in content:
                    print_error(f"Missing required import: {imp}")
                    self.errors.append(f"Missing import: {imp}")
                    return False
        
        print_success("All required imports present ✓")
        
        return True
    
    def run_full_validation(self, config_path: str = 'prediction_generator_config.json') -> bool:
        """Run complete validation suite"""
        print_header("PREDICTION SYSTEM VALIDATION")
        print_info(f"Validating system at: {Path.cwd()}")
        print_info(f"Config file: {config_path}\n")
        
        # 1. Config file
        valid_config, config = self.validate_config_file(config_path)
        if not valid_config:
            return False
        
        # 2. Feature weights
        if not self.validate_feature_weights(config):
            return False
        
        # 3. GPU config
        if not self.validate_gpu_config(config):
            return False
        
        # 4. survivor_scorer integration
        if not self.validate_survivor_scorer_integration():
            return False
        
        # 5. File system
        if not self.validate_file_system(config):
            return False
        
        # 6. prediction_generator.py
        if not self.validate_prediction_generator():
            return False
        
        # Summary
        print_header("VALIDATION SUMMARY")
        
        print_success(f"Successes: {len(self.successes)}")
        if self.warnings:
            print_warning(f"Warnings: {len(self.warnings)}")
            for w in self.warnings:
                print(f"  - {w}")
        
        if self.errors:
            print_error(f"Errors: {len(self.errors)}")
            for e in self.errors:
                print(f"  - {e}")
            print("\n❌ VALIDATION FAILED")
            return False
        else:
            print("\n" + "="*70)
            print(f"{Colors.GREEN}{Colors.BOLD}✅ ALL VALIDATIONS PASSED! SYSTEM READY FOR PRODUCTION{Colors.END}")
            print("="*70 + "\n")
            return True


def main():
    """Main validation entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Prediction System')
    parser.add_argument('--config', type=str, default='prediction_generator_config.json',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    validator = SystemValidator()
    success = validator.run_full_validation(args.config)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
