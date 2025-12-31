#!/usr/bin/env python3
"""
PRNG Fingerprint Library Generator
===================================
Version: 1.0.0
Date: 2025-12-26

Generates reference fingerprint data for all PRNGs in the registry.
This creates the comparison library that Step 0 Classification uses.

Usage:
    # Generate full library (all 46 PRNGs, 20 sequences each)
    python3 generate_fingerprint_library.py --output-dir fingerprints/
    
    # Quick test (3 PRNGs, 5 sequences each)
    python3 generate_fingerprint_library.py --output-dir fingerprints/ --quick-test
    
    # Single PRNG
    python3 generate_fingerprint_library.py --output-dir fingerprints/ --prng java_lcg

Output Structure:
    fingerprints/
    ├── reference_sequences/{prng_id}/seed_{seed}_{n}draws.json
    ├── computed_fingerprints/{prng_id}_fingerprint.json
    ├── ml_training_set/fingerprint_features.csv
    ├── ml_training_set/fingerprint_labels.csv
    └── fingerprint_registry.json

Resource Estimates:
    - 46 PRNGs × 20 sequences × 20,000 draws = 18.4M values
    - Storage: ~500MB (compressed JSON)
    - Generation time: ~10-30 minutes (depends on CPU)
"""

import argparse
import json
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv

# Add current directory to path for imports
import sys
import os
sys.path.insert(0, os.getcwd())

# Import PRNG registry
try:
    from prng_registry import (
        list_available_prngs,
        get_kernel_info
    )
    PRNG_REGISTRY_AVAILABLE = True
    print(f"✓ Loaded prng_registry.py with {len(list_available_prngs())} PRNGs")
except ImportError as e:
    PRNG_REGISTRY_AVAILABLE = False
    print(f"Error importing prng_registry: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files here: {[f for f in os.listdir('.') if f.endswith('.py')][:10]}")
    print("Make sure prng_registry.py is in the current directory.")
    exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class FingerprintConfig:
    """Configuration for fingerprint generation."""
    sequences_per_prng: int = 20          # Number of different seeds to test
    draws_per_sequence: int = 20000       # Draws per sequence
    mod: int = 1000                       # Modulo for output values
    seed_strategy: str = "distributed"    # "distributed", "random", "sequential"
    seed_range: tuple = (1, 2**31 - 1)    # Seed value range
    
    # Feature extraction settings
    residue_mods: List[int] = field(default_factory=lambda: [8, 125, 1000])
    max_lag_autocorr: int = 10
    
    # Output settings
    compress_sequences: bool = True       # Store as compact arrays
    generate_ml_dataset: bool = True      # Create training CSV


# =============================================================================
# STATISTICAL FEATURE EXTRACTION (Team Beta Approved - v2.0)
# =============================================================================
# Guiding Principle: Temporal/relational structure, NOT marginal distribution
# Uniformity ≠ discriminative power under %1000

def extract_fingerprint_features(draws: List[int], config: FingerprintConfig) -> Dict[str, Any]:
    """
    Extract discriminative fingerprint features for PRNG classification.
    
    Team Beta approved feature set (~33 features):
    - Permutation structure (TestU01 adaptation)
    - Transition geometry (JS divergence)
    - CRT-lane dynamics
    - Minimal correlation profile
    - Run/texture features
    """
    arr = np.array(draws, dtype=np.int64)
    n = len(arr)
    
    features = {}
    
    # =========================================================================
    # 1. PERMUTATION TESTS (TestU01 adaptation) - HIGH VALUE
    # =========================================================================
    if n >= 3:
        # Permutation entropy for length 3 (6 possible orderings)
        perm_counts_3 = [0] * 6
        perm_map = {(0,1,2):0, (0,2,1):1, (1,0,2):2, (1,2,0):3, (2,0,1):4, (2,1,0):5}
        for i in range(n - 2):
            triple = arr[i:i+3]
            order = tuple(np.argsort(triple))
            perm_counts_3[perm_map[order]] += 1
        total_3 = sum(perm_counts_3)
        perm_probs_3 = [c / total_3 for c in perm_counts_3]
        
        # Entropy
        perm_entropy_3 = -sum(p * np.log2(p + 1e-10) for p in perm_probs_3)
        features["permutation_entropy_len3"] = float(perm_entropy_3)
        
        # Chi-squared against uniform (1/6 each)
        expected_3 = total_3 / 6
        chi2_3 = sum((c - expected_3)**2 / expected_3 for c in perm_counts_3)
        features["permutation_chi2_len3"] = float(chi2_3)
    
    if n >= 4:
        # Permutation entropy for length 4 (24 possible orderings)
        from itertools import permutations
        perm_list_4 = list(permutations(range(4)))
        perm_map_4 = {p: i for i, p in enumerate(perm_list_4)}
        perm_counts_4 = [0] * 24
        for i in range(n - 3):
            quad = arr[i:i+4]
            order = tuple(np.argsort(quad))
            perm_counts_4[perm_map_4[order]] += 1
        total_4 = sum(perm_counts_4)
        perm_probs_4 = [c / total_4 for c in perm_counts_4]
        perm_entropy_4 = -sum(p * np.log2(p + 1e-10) for p in perm_probs_4)
        features["permutation_entropy_len4"] = float(perm_entropy_4)
    
    # Delta permutation (permutation on differences)
    if n >= 4:
        diffs = arr[1:] - arr[:-1]
        perm_counts_d = [0] * 6
        for i in range(len(diffs) - 2):
            triple = diffs[i:i+3]
            order = tuple(np.argsort(triple))
            perm_counts_d[perm_map[order]] += 1
        total_d = sum(perm_counts_d)
        if total_d > 0:
            perm_probs_d = [c / total_d for c in perm_counts_d]
            perm_entropy_d = -sum(p * np.log2(p + 1e-10) for p in perm_probs_d)
            features["permutation_entropy_delta_mod8"] = float(perm_entropy_d)
        else:
            features["permutation_entropy_delta_mod8"] = 0.0
    
    # =========================================================================
    # 2. JS PAIR DIVERGENCE (Serial test adaptation) - HIGH VALUE
    # =========================================================================
    def transition_matrix(stream, m):
        """Build transition probability matrix with Laplace smoothing."""
        counts = np.ones((m, m), dtype=np.float64)  # Laplace smoothing
        for i in range(len(stream) - 1):
            a, b = stream[i] % m, stream[i+1] % m
            counts[a, b] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        return counts / row_sums
    
    def js_divergence_flat(P, Q):
        """Jensen-Shannon divergence between matrices (flattened)."""
        p = P.flatten()
        q = Q.flatten()
        p = p / p.sum()
        q = q / q.sum()
        m = (p + q) / 2
        
        def kl(a, b):
            mask = (a > 0) & (b > 0)
            return np.sum(a[mask] * np.log(a[mask] / b[mask]))
        
        return 0.5 * kl(p, m) + 0.5 * kl(q, m)
    
    if n > 2:
        # Mod 8 transition matrices at lag 1 and 2
        P8_lag1 = transition_matrix(arr, 8)
        features["lane_mod8_transition_entropy"] = float(-np.sum(
            P8_lag1 * np.log2(P8_lag1 + 1e-10)
        ) / 8)
        features["lane_mod8_diagonal_dominance"] = float(np.trace(P8_lag1) / 8)
        
        # For JS divergence, we store the transition matrix stats
        # (actual JS comparison happens at classification time against references)
        features["js_pair_mod8_row_entropy_mean"] = float(np.mean([
            -np.sum(row * np.log2(row + 1e-10)) for row in P8_lag1
        ]))
        features["js_pair_mod8_row_entropy_std"] = float(np.std([
            -np.sum(row * np.log2(row + 1e-10)) for row in P8_lag1
        ]))
        
        # Lag 2 transition
        if n > 3:
            arr_lag2 = arr[::2] if len(arr) > 4 else arr
            P8_lag2 = transition_matrix(arr, 8)  # Actually compute lag-2
            counts_lag2 = np.ones((8, 8), dtype=np.float64)
            for i in range(len(arr) - 2):
                a, b = arr[i] % 8, arr[i+2] % 8
                counts_lag2[a, b] += 1
            P8_lag2 = counts_lag2 / counts_lag2.sum(axis=1, keepdims=True)
            features["js_pair_mod8_lag2_diag"] = float(np.trace(P8_lag2) / 8)
        
        # Mod 125 (compressed - just key stats)
        P125 = transition_matrix(arr, 125)
        features["js_pair_mod125_entropy"] = float(-np.sum(
            P125 * np.log2(P125 + 1e-10)
        ) / 125)
    
    # =========================================================================
    # 3. SPACING FEATURES (Birthday spacing concept) - HIGH VALUE
    # =========================================================================
    if n > 10:
        # Spacing between same residue occurrences (mod 8)
        from collections import defaultdict
        last_seen = {}
        gaps = defaultdict(list)
        for i, d in enumerate(arr):
            r = d % 8
            if r in last_seen:
                gaps[r].append(i - last_seen[r])
            last_seen[r] = i
        
        all_gaps = [g for gs in gaps.values() for g in gs]
        if all_gaps:
            features["spacing_mean_mod8"] = float(np.mean(all_gaps))
            features["spacing_std_mod8"] = float(np.std(all_gaps))
            # Spacing entropy
            gap_hist, _ = np.histogram(all_gaps, bins=min(50, len(set(all_gaps))), density=True)
            gap_hist_nz = gap_hist[gap_hist > 0]
            features["spacing_entropy_mod8"] = float(-np.sum(gap_hist_nz * np.log2(gap_hist_nz)))
        else:
            features["spacing_mean_mod8"] = 8.0
            features["spacing_std_mod8"] = 0.0
            features["spacing_entropy_mod8"] = 0.0
    
    # =========================================================================
    # 4. CORRELATION FEATURES (trimmed to lag 1-3)
    # =========================================================================
    if n > 1:
        corr = np.corrcoef(arr[:-1], arr[1:])[0, 1]
        features["adjacent_correlation"] = float(corr) if not np.isnan(corr) else 0.0
    
    for lag in [1, 2, 3]:
        if n > lag:
            corr = np.corrcoef(arr[:-lag], arr[lag:])[0, 1]
            features[f"autocorr_lag{lag}"] = float(corr) if not np.isnan(corr) else 0.0
    
    # Lane-specific autocorrelation
    if n > 1:
        for m in [8, 125]:
            lane = (arr % m).astype(np.float64)
            corr = np.corrcoef(lane[:-1], lane[1:])[0, 1]
            features[f"lane_mod{m}_autocorr"] = float(corr) if not np.isnan(corr) else 0.0
    
    # =========================================================================
    # 5. DELTA TEXTURE (kept features only)
    # =========================================================================
    if n > 1:
        diffs = np.diff(arr)
        features["diff_std"] = float(np.std(diffs))
        
        # Diff entropy (mod 1000)
        diff_mod = np.abs(diffs) % 1000
        diff_hist, _ = np.histogram(diff_mod, bins=100, density=True)
        diff_hist_nz = diff_hist[diff_hist > 0]
        features["diff_entropy"] = float(-np.sum(diff_hist_nz * np.log2(diff_hist_nz + 1e-10)))
        
        # Small delta ratio
        small_delta_count = np.sum(np.abs(diffs) <= 100)
        features["small_delta_ratio"] = float(small_delta_count / len(diffs))
        
        # Diff mod 8 entropy (KEEP per Team Beta)
        diff_mod8 = diffs % 8
        hist8, _ = np.histogram(diff_mod8, bins=8, range=(0, 8), density=True)
        hist8_nz = hist8[hist8 > 0]
        features["diff_mod8_entropy"] = float(-np.sum(hist8_nz * np.log2(hist8_nz)))
    
    # =========================================================================
    # 6. RUN / MONOTONIC STRUCTURE
    # =========================================================================
    runs = compute_run_lengths(arr)
    features["run_mean"] = float(np.mean(runs))
    features["run_std"] = float(np.std(runs))
    features["run_max"] = int(np.max(runs))
    
    if n > 1:
        diffs = np.diff(arr)
        increasing = np.sum(diffs > 0)
        decreasing = np.sum(diffs < 0)
        features["monotonic_ratio"] = float(increasing / (increasing + decreasing + 1e-10))
    
    # =========================================================================
    # 7. POWER-OF-2 BIAS (mod 2, 4, 8 only)
    # =========================================================================
    for power in [2, 4, 8]:
        mod_vals = arr % power
        expected_uniform = n / power
        observed = np.bincount(mod_vals, minlength=power)
        bias = np.max(np.abs(observed - expected_uniform)) / expected_uniform
        features[f"mod{power}_bias"] = float(bias)
    
    return features


def entropy_of_values(arr: np.ndarray) -> float:
    """Calculate entropy of value distribution."""
    unique, counts = np.unique(arr, return_counts=True)
    probs = counts / len(arr)
    return float(-np.sum(probs * np.log2(probs + 1e-10)))


def compute_run_lengths(arr: np.ndarray) -> np.ndarray:
    """Compute lengths of monotonic runs."""
    diffs = np.diff(arr)
    signs = np.sign(diffs)
    
    # Find where sign changes
    sign_changes = np.where(signs[:-1] != signs[1:])[0]
    
    if len(sign_changes) == 0:
        return np.array([len(arr)])
    
    runs = np.diff(np.concatenate([[0], sign_changes + 1, [len(signs)]]))
    return runs


def skewness(arr: np.ndarray) -> float:
    """Calculate skewness."""
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 3))


def kurtosis(arr: np.ndarray) -> float:
    """Calculate excess kurtosis."""
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 4) - 3)


# =============================================================================
# FINGERPRINT GENERATION
# =============================================================================

def generate_seeds(n: int, strategy: str, seed_range: tuple) -> List[int]:
    """Generate seed values based on strategy."""
    min_seed, max_seed = seed_range
    
    if strategy == "distributed":
        # Evenly distributed across range
        step = (max_seed - min_seed) // (n + 1)
        return [min_seed + step * (i + 1) for i in range(n)]
    elif strategy == "random":
        np.random.seed(42)  # Reproducible
        return list(np.random.randint(min_seed, max_seed, size=n))
    elif strategy == "sequential":
        return list(range(min_seed, min_seed + n))
    else:
        raise ValueError(f"Unknown seed strategy: {strategy}")


def get_base_prng_name(prng_id: str) -> str:
    """
    Get the base PRNG name by stripping reverse/hybrid suffixes.
    
    Examples:
        mt19937_reverse -> mt19937
        mt19937_hybrid_reverse -> mt19937
        java_lcg_hybrid -> java_lcg
    """
    base = prng_id
    
    # Strip _reverse suffix first
    if base.endswith('_reverse'):
        base = base[:-8]  # len('_reverse') = 8
    
    # Strip _hybrid suffix
    if base.endswith('_hybrid'):
        base = base[:-7]  # len('_hybrid') = 7
    
    return base


def generate_prng_sequences(
    prng_id: str, 
    config: FingerprintConfig
) -> Dict[str, Any]:
    """Generate all reference sequences for a single PRNG."""
    
    try:
        prng_info = get_kernel_info(prng_id)
        
        # Check for cpu_reference, with fallback to base PRNG
        if 'cpu_reference' in prng_info:
            cpu_func = prng_info['cpu_reference']
        else:
            # Try base PRNG for reverse/hybrid variants
            base_prng = get_base_prng_name(prng_id)
            if base_prng != prng_id:
                try:
                    base_info = get_kernel_info(base_prng)
                    if 'cpu_reference' in base_info:
                        cpu_func = base_info['cpu_reference']
                        print(f"  ℹ️  Using base PRNG '{base_prng}' CPU reference")
                    else:
                        print(f"  ⚠️  No cpu_reference in base PRNG '{base_prng}'")
                        return None
                except:
                    print(f"  ⚠️  Base PRNG '{base_prng}' not found")
                    return None
            else:
                print(f"  ⚠️  No cpu_reference (GPU-only kernel)")
                return None
        
    except Exception as e:
        print(f"  ⚠️  Error: {e}")
        return None
    
    seeds = generate_seeds(
        config.sequences_per_prng, 
        config.seed_strategy, 
        config.seed_range
    )
    
    sequences = []
    all_features = []
    
    for seed in seeds:
        try:
            # Generate sequence using CPU reference
            # CPU reference signature: func(seed, n, skip=0, **kwargs) -> List[int]
            raw_draws = cpu_func(seed, config.draws_per_sequence, skip=0)
            
            # Apply modulo
            if isinstance(raw_draws, np.ndarray):
                draws = (raw_draws % config.mod).tolist()
            else:
                draws = [d % config.mod for d in raw_draws]
            
            # Extract features
            features = extract_fingerprint_features(draws, config)
            features["seed"] = seed
            
            sequences.append({
                "seed": seed,
                "draws": draws if not config.compress_sequences else None,
                "draw_count": len(draws),
                "features": features
            })
            
            all_features.append(features)
            
        except Exception as e:
            print(f"  ⚠️  Seed {seed} failed for {prng_id}: {e}")
            continue
    
    if not sequences:
        return None
    
    # Aggregate features across all sequences
    aggregated = aggregate_features(all_features)
    
    return {
        "prng_id": prng_id,
        "config": asdict(config),
        "sequences": sequences,
        "aggregated_fingerprint": aggregated,
        "description": prng_info.get('description', ''),
        "generated_at": datetime.now().isoformat()
    }


def aggregate_features(feature_list: List[Dict]) -> Dict[str, Any]:
    """Aggregate features across multiple sequences into a single fingerprint."""
    
    if not feature_list:
        return {}
    
    # Get all feature names (excluding 'seed')
    feature_names = [k for k in feature_list[0].keys() if k != 'seed']
    
    aggregated = {}
    feature_vector = []
    feature_vector_names = []
    
    for name in feature_names:
        values = [f[name] for f in feature_list if name in f]
        
        if not values:
            continue
        
        # Handle different types
        if isinstance(values[0], list):
            # Distribution - average across sequences
            arr = np.array(values)
            mean_dist = np.mean(arr, axis=0).tolist()
            std_dist = np.std(arr, axis=0).tolist()
            aggregated[name] = {
                "mean": mean_dist,
                "std": std_dist
            }
            # Add to feature vector (just the mean)
            feature_vector.extend(mean_dist)
            feature_vector_names.extend([f"{name}_{i}" for i in range(len(mean_dist))])
        else:
            # Scalar - compute statistics
            arr = np.array(values)
            aggregated[name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr))
            }
            # Add mean to feature vector
            feature_vector.append(float(np.mean(arr)))
            feature_vector_names.append(name)
    
    aggregated["_feature_vector"] = feature_vector
    aggregated["_feature_names"] = feature_vector_names
    
    return aggregated


# =============================================================================
# LIBRARY GENERATION
# =============================================================================

def generate_fingerprint_library(
    output_dir: Path,
    config: FingerprintConfig,
    prng_filter: Optional[List[str]] = None,
    parallel: bool = True,
    max_workers: int = 4,
    append_mode: bool = False
) -> Dict[str, Any]:
    """
    Generate complete fingerprint library for all PRNGs.
    
    Args:
        output_dir: Directory for output files
        config: Generation configuration
        prng_filter: Optional list of specific PRNGs to process
        parallel: (unused, for future)
        max_workers: (unused, for future)
        append_mode: If True, add to existing library instead of regenerating
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / "reference_sequences").mkdir(exist_ok=True)
    (output_dir / "computed_fingerprints").mkdir(exist_ok=True)
    (output_dir / "ml_training_set").mkdir(exist_ok=True)
    
    # Load existing registry if append mode
    existing_registry = {}
    existing_prngs = set()
    if append_mode:
        registry_file = output_dir / "fingerprint_registry.json"
        if registry_file.exists():
            with open(registry_file) as f:
                existing_registry = json.load(f)
                existing_prngs = set(existing_registry.get("prngs", {}).keys())
            print(f"📂 Append mode: Found {len(existing_prngs)} existing fingerprints")
        else:
            print(f"📂 Append mode: No existing registry, creating new")
    
    # Get PRNG list
    all_prng_ids = list_available_prngs()
    
    # Filter out _reverse entries - they're sieve strategies, not different algorithms
    # Reverse PRNGs produce identical statistical fingerprints as their forward counterparts
    reverse_entries = [p for p in all_prng_ids if '_reverse' in p]
    prng_ids = [p for p in all_prng_ids if '_reverse' not in p]
    
    if reverse_entries:
        print(f"Note: Skipping {len(reverse_entries)} reverse sieve entries (same fingerprint as forward)")
    
    # Apply filter
    if prng_filter:
        prng_ids = [p for p in prng_ids if p in prng_filter]
    
    # In append mode, skip already-existing PRNGs unless explicitly filtered
    skipped_existing = []
    if append_mode and not prng_filter:
        # No filter = process only NEW PRNGs
        new_prng_ids = [p for p in prng_ids if p not in existing_prngs]
        skipped_existing = [p for p in prng_ids if p in existing_prngs]
        prng_ids = new_prng_ids
        if skipped_existing:
            print(f"⏭️  Skipping {len(skipped_existing)} existing PRNGs (use --prng to regenerate specific ones)")
    elif append_mode and prng_filter:
        # With filter = regenerate specified PRNGs even if they exist
        will_overwrite = [p for p in prng_ids if p in existing_prngs]
        if will_overwrite:
            print(f"🔄 Will regenerate {len(will_overwrite)} existing PRNGs: {will_overwrite}")
    
    if not prng_ids:
        print(f"\n{'='*60}")
        print("NO NEW PRNGS TO PROCESS")
        print(f"{'='*60}")
        print(f"All {len(existing_prngs)} PRNGs already have fingerprints.")
        print(f"Use --prng <name> to regenerate specific PRNGs.")
        print(f"{'='*60}\n")
        return existing_registry
    
    print(f"\n{'='*60}")
    print(f"PRNG FINGERPRINT LIBRARY GENERATOR")
    print(f"{'='*60}")
    print(f"PRNGs to process: {len(prng_ids)}")
    print(f"Sequences per PRNG: {config.sequences_per_prng}")
    print(f"Draws per sequence: {config.draws_per_sequence}")
    print(f"Total draws: {len(prng_ids) * config.sequences_per_prng * config.draws_per_sequence:,}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    results = {}
    ml_rows = []
    failed = []
    
    # Process PRNGs
    for i, prng_id in enumerate(prng_ids):
        print(f"[{i+1}/{len(prng_ids)}] Processing {prng_id}...")
        
        result = generate_prng_sequences(prng_id, config)
        
        if result is None:
            print(f"  ❌ Failed")
            failed.append(prng_id)
            continue
        
        results[prng_id] = result
        
        # Save individual fingerprint
        fingerprint_file = output_dir / "computed_fingerprints" / f"{prng_id}_fingerprint.json"
        with open(fingerprint_file, 'w') as f:
            json.dump({
                "prng_id": prng_id,
                "version": "1.0.0",
                "source_sequences": len(result["sequences"]),
                "total_draws": sum(s["draw_count"] for s in result["sequences"]),
                "fingerprint": result["aggregated_fingerprint"],
                "generated_at": result["generated_at"]
            }, f, indent=2)
        
        # Save reference sequences (optional, large files)
        if not config.compress_sequences:
            seq_dir = output_dir / "reference_sequences" / prng_id
            seq_dir.mkdir(exist_ok=True)
            for seq in result["sequences"]:
                seq_file = seq_dir / f"seed_{seq['seed']}_{seq['draw_count']}draws.json"
                with open(seq_file, 'w') as f:
                    json.dump(seq, f)
        
        # Collect ML training data
        if config.generate_ml_dataset:
            agg = result["aggregated_fingerprint"]
            if "_feature_vector" in agg:
                for j, seq in enumerate(result["sequences"]):
                    row = {
                        "prng_id": prng_id,
                        "sequence_id": f"{prng_id}_{j:03d}",
                        "seed": seq["seed"]
                    }
                    # Add individual sequence features
                    for fname, fval in seq["features"].items():
                        if fname != "seed" and not isinstance(fval, list):
                            row[fname] = fval
                    ml_rows.append(row)
        
        print(f"  ✅ Generated {len(result['sequences'])} sequences")
    
    # Save ML training set
    if config.generate_ml_dataset and ml_rows:
        csv_file = output_dir / "ml_training_set" / "fingerprint_features.csv"
        
        # Get all feature names from new rows
        all_columns = set()
        for row in ml_rows:
            all_columns.update(row.keys())
        
        # In append mode, load existing CSV and merge
        existing_ml_rows = []
        if append_mode and csv_file.exists():
            import pandas as pd
            try:
                existing_df = pd.read_csv(csv_file)
                # Remove rows for PRNGs we're regenerating
                regenerated_prngs = set(results.keys())
                existing_df = existing_df[~existing_df['prng_id'].isin(regenerated_prngs)]
                existing_ml_rows = existing_df.to_dict('records')
                all_columns.update(existing_df.columns)
                print(f"📂 Merging with {len(existing_ml_rows)} existing ML rows")
            except Exception as e:
                print(f"⚠️  Could not load existing ML CSV: {e}")
        
        # Combine rows
        combined_rows = existing_ml_rows + ml_rows
        
        # Sort columns (prng_id first, then alphabetical)
        columns = ["prng_id", "sequence_id", "seed"] + sorted(
            [c for c in all_columns if c not in ["prng_id", "sequence_id", "seed"]]
        )
        
        # Write CSV
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(combined_rows)
        
        print(f"\n✅ ML training set: {csv_file}")
        print(f"   Rows: {len(combined_rows)} ({len(ml_rows)} new), Columns: {len(columns)}")
    
    # Build registry - merge with existing if append mode
    new_prngs = {
        prng_id: {
            "fingerprint_file": f"computed_fingerprints/{prng_id}_fingerprint.json",
            "sequences": len(results[prng_id]["sequences"]),
            "total_draws": sum(s["draw_count"] for s in results[prng_id]["sequences"])
        }
        for prng_id in results
    }
    
    if append_mode and existing_registry:
        # Merge: existing + new (new overwrites existing for same prng_id)
        merged_prngs = existing_registry.get("prngs", {})
        merged_prngs.update(new_prngs)
        all_prngs = merged_prngs
        
        # Combine failed lists
        all_failed = list(set(existing_registry.get("failed", []) + failed))
        # Remove from failed if we successfully regenerated it
        all_failed = [f for f in all_failed if f not in results]
    else:
        all_prngs = new_prngs
        all_failed = failed
    
    registry = {
        "version": "1.0.0",
        "generated_at": datetime.now().isoformat(),
        "config": asdict(config),
        "prngs": all_prngs,
        "skipped_reverse_variants": reverse_entries,
        "failed": all_failed,
        "summary": {
            "total_prngs": len(all_prngs),
            "skipped_reverse": len(reverse_entries),
            "failed": len(all_failed),
            "total_sequences": sum(p.get("sequences", 0) for p in all_prngs.values()),
            "total_draws": sum(p.get("total_draws", 0) for p in all_prngs.values())
        }
    }
    
    registry_file = output_dir / "fingerprint_registry.json"
    with open(registry_file, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    if append_mode:
        print(f"Mode: APPEND")
        print(f"New PRNGs added: {len(results)}")
        print(f"Total PRNGs in library: {len(all_prngs)}")
    else:
        print(f"PRNGs processed: {len(results)}")
    print(f"Reverse variants skipped: {len(reverse_entries)} (identical to forward)")
    if all_failed:
        print(f"Failed: {len(all_failed)} - {all_failed}")
    print(f"Total sequences: {registry['summary']['total_sequences']}")
    print(f"Total draws: {registry['summary']['total_draws']:,}")
    print(f"Registry: {registry_file}")
    print(f"{'='*60}\n")
    
    return registry


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate PRNG Fingerprint Library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full library generation
  python3 generate_fingerprint_library.py --output-dir fingerprints/
  
  # Quick test (3 PRNGs, 5 sequences)
  python3 generate_fingerprint_library.py --output-dir fingerprints/ --quick-test
  
  # Single PRNG
  python3 generate_fingerprint_library.py --output-dir fingerprints/ --prng java_lcg
  
  # Add new PRNG(s) to existing library
  python3 generate_fingerprint_library.py --output-dir fingerprints/ --prng new_prng_name --append
  
  # Custom configuration
  python3 generate_fingerprint_library.py --output-dir fingerprints/ \\
      --sequences 10 --draws 10000
        """
    )
    
    parser.add_argument("--output-dir", type=str, default="fingerprints/",
                        help="Output directory for fingerprint library")
    parser.add_argument("--sequences", type=int, default=20,
                        help="Number of sequences per PRNG (default: 20)")
    parser.add_argument("--draws", type=int, default=20000,
                        help="Draws per sequence (default: 20000)")
    parser.add_argument("--prng", type=str, nargs="+",
                        help="Specific PRNG(s) to process (default: all)")
    parser.add_argument("--quick-test", action="store_true",
                        help="Quick test mode (3 PRNGs, 5 sequences, 5000 draws)")
    parser.add_argument("--no-ml-dataset", action="store_true",
                        help="Skip ML training set generation")
    parser.add_argument("--save-sequences", action="store_true",
                        help="Save full draw sequences (large files)")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing library instead of regenerating all")
    
    args = parser.parse_args()
    
    # Configure
    if args.quick_test:
        config = FingerprintConfig(
            sequences_per_prng=5,
            draws_per_sequence=5000,
            generate_ml_dataset=not args.no_ml_dataset,
            compress_sequences=not args.save_sequences
        )
        prng_filter = ["java_lcg", "mt19937", "xorshift64"]
    else:
        config = FingerprintConfig(
            sequences_per_prng=args.sequences,
            draws_per_sequence=args.draws,
            generate_ml_dataset=not args.no_ml_dataset,
            compress_sequences=not args.save_sequences
        )
        prng_filter = args.prng
    
    # Generate
    generate_fingerprint_library(
        output_dir=Path(args.output_dir),
        config=config,
        prng_filter=prng_filter,
        append_mode=args.append
    )


if __name__ == "__main__":
    main()
