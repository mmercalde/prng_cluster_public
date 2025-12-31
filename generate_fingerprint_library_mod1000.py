#!/usr/bin/env python3
"""
PRNG Fingerprint Library Generator - Mod1000 Specific Features
===============================================================
Version: 2.0.0 (Experimental)
Date: 2025-12-27

Implements Team Beta's mod1000-specific discrimination strategies:
1. Digit analysis with correlations
2. Temporal walk patterns  
3. Fourier analysis
4. Transition matrix statistics
5. Modulo cascade analysis
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional
from collections import Counter
import csv
import sys
import os

sys.path.insert(0, os.getcwd())

from prng_registry import list_available_prngs, get_kernel_info


@dataclass
class FingerprintConfig:
    sequences_per_prng: int = 20
    draws_per_sequence: int = 20000
    mod: int = 1000
    seed_strategy: str = "distributed"
    seed_range: tuple = (1, 2**31 - 1)
    compress_sequences: bool = True
    generate_ml_dataset: bool = True


# =============================================================================
# MOD1000-SPECIFIC FEATURE EXTRACTION
# =============================================================================

def extract_fingerprint_features(draws: List[int], config: FingerprintConfig) -> Dict[str, Any]:
    """
    Extract mod1000-specific features designed to survive modular reduction.
    """
    arr = np.array(draws, dtype=np.int64) % 1000  # Ensure mod1000
    n = len(arr)
    features = {}
    
    # =========================================================================
    # 1. DIGIT DISTRIBUTION ANALYSIS
    # =========================================================================
    hundreds = (arr // 100) % 10
    tens = (arr // 10) % 10
    units = arr % 10
    
    # Entropies
    features["digit_hundreds_entropy"] = entropy_of_values(hundreds)
    features["digit_tens_entropy"] = entropy_of_values(tens)
    features["digit_units_entropy"] = entropy_of_values(units)
    
    # Chi-squared against uniform (10 bins)
    for name, digit_arr in [("hundreds", hundreds), ("tens", tens), ("units", units)]:
        hist = np.bincount(digit_arr, minlength=10) / n
        expected = 0.1
        chi2 = np.sum((hist - expected)**2 / expected) * n
        features[f"digit_{name}_chi2"] = float(chi2)
    
    # Inter-digit correlations
    features["digit_corr_hundreds_tens"] = safe_corr(hundreds, tens)
    features["digit_corr_tens_units"] = safe_corr(tens, units)
    features["digit_corr_hundreds_units"] = safe_corr(hundreds, units)
    
    # Digit transition patterns (how often does hundreds digit change?)
    features["digit_hundreds_change_rate"] = float(np.sum(hundreds[1:] != hundreds[:-1]) / (n-1))
    features["digit_tens_change_rate"] = float(np.sum(tens[1:] != tens[:-1]) / (n-1))
    features["digit_units_change_rate"] = float(np.sum(units[1:] != units[:-1]) / (n-1))
    
    # =========================================================================
    # 2. TEMPORAL WALK PATTERNS
    # =========================================================================
    if n > 1:
        # Step sizes (differences mod 1000)
        steps = (arr[1:] - arr[:-1]) % 1000
        
        features["step_mean"] = float(np.mean(steps))
        features["step_std"] = float(np.std(steps))
        features["step_median"] = float(np.median(steps))
        
        # Step size distribution entropy
        step_hist, _ = np.histogram(steps, bins=100, range=(0, 1000), density=True)
        step_hist_nz = step_hist[step_hist > 0]
        features["step_entropy"] = float(-np.sum(step_hist_nz * np.log2(step_hist_nz))) if len(step_hist_nz) > 0 else 0.0
        
        # Consecutive step patterns (lookahead=3)
        if n > 3:
            step_pairs = list(zip(steps[:-1], steps[1:]))
            pair_counts = Counter(step_pairs)
            features["step_pair_unique_ratio"] = float(len(pair_counts) / len(step_pairs))
            
            # How often do we see the same step twice in a row?
            same_step = np.sum(steps[1:] == steps[:-1])
            features["step_repeat_ratio"] = float(same_step / (len(steps) - 1))
        
        # Direction changes (increasing vs decreasing)
        direction = np.sign(arr[1:].astype(float) - arr[:-1].astype(float))
        direction_changes = np.sum(direction[1:] != direction[:-1])
        features["direction_change_rate"] = float(direction_changes / (n-2)) if n > 2 else 0.0
        
    # =========================================================================
    # 3. FOURIER ANALYSIS
    # =========================================================================
    if n >= 64:
        # Convert to unit circle representation
        complex_draws = np.exp(2j * np.pi * arr / 1000)
        
        # FFT
        fft_result = np.fft.fft(complex_draws)
        magnitudes = np.abs(fft_result[1:n//2])  # Skip DC, use positive frequencies
        
        if len(magnitudes) > 0 and np.mean(magnitudes) > 0:
            # Spectral flatness (1 = white noise, <1 = peaks present)
            log_mean = np.mean(np.log(magnitudes + 1e-10))
            features["spectral_flatness"] = float(np.exp(log_mean) / np.mean(magnitudes))
            
            # Peak-to-average ratio
            features["spectral_peak_ratio"] = float(np.max(magnitudes) / np.mean(magnitudes))
            
            # Number of significant peaks (> 2x mean)
            threshold = 2 * np.mean(magnitudes)
            features["spectral_peak_count"] = int(np.sum(magnitudes > threshold))
            
            # Energy concentration in low frequencies
            quarter = len(magnitudes) // 4
            if quarter > 0:
                low_energy = np.sum(magnitudes[:quarter]**2)
                total_energy = np.sum(magnitudes**2)
                features["spectral_low_freq_ratio"] = float(low_energy / total_energy) if total_energy > 0 else 0.0
        else:
            features["spectral_flatness"] = 1.0
            features["spectral_peak_ratio"] = 1.0
            features["spectral_peak_count"] = 0
            features["spectral_low_freq_ratio"] = 0.25
    
    # =========================================================================
    # 4. TRANSITION MATRIX STATISTICS (compressed - not full 1000x1000)
    # =========================================================================
    if n > 1:
        # Use binned transitions (100 bins = 10x compression)
        bins = 100
        binned = arr // 10  # 0-99
        
        trans_matrix = np.zeros((bins, bins), dtype=np.float64)
        for i in range(n-1):
            trans_matrix[binned[i], binned[i+1]] += 1
        
        # Normalize rows
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        trans_probs = trans_matrix / row_sums
        
        # Statistics on transition matrix
        features["trans_sparsity"] = float(np.sum(trans_matrix == 0) / (bins * bins))
        features["trans_diagonal_strength"] = float(np.trace(trans_probs) / bins)
        
        # Row entropy variance (how uniform are transitions from each state?)
        row_entropies = []
        for row in trans_probs:
            row_nz = row[row > 0]
            if len(row_nz) > 0:
                row_entropies.append(-np.sum(row_nz * np.log2(row_nz)))
            else:
                row_entropies.append(0)
        features["trans_row_entropy_mean"] = float(np.mean(row_entropies))
        features["trans_row_entropy_std"] = float(np.std(row_entropies))
        
        # Off-diagonal patterns (banded structure = LCG signature)
        diag_weights = []
        for offset in range(1, min(20, bins)):
            diag_sum = np.sum(np.diag(trans_probs, k=offset)) + np.sum(np.diag(trans_probs, k=-offset))
            diag_weights.append(diag_sum)
        features["trans_near_diag_weight"] = float(sum(diag_weights[:5])) if diag_weights else 0.0
        features["trans_far_diag_weight"] = float(sum(diag_weights[10:])) if len(diag_weights) > 10 else 0.0
    
    # =========================================================================
    # 5. MODULO CASCADE ANALYSIS
    # =========================================================================
    # How does the sequence look under different moduli?
    cascade_bases = [8, 16, 32, 64, 125, 128, 256, 500]
    
    for base in cascade_bases:
        mod_arr = arr % base
        
        # Autocorrelation at lag 1
        if n > 1:
            corr = safe_corr(mod_arr[:-1], mod_arr[1:])
            features[f"cascade_mod{base}_autocorr"] = corr
        
        # Entropy
        features[f"cascade_mod{base}_entropy"] = entropy_of_values(mod_arr)
    
    # =========================================================================
    # 6. RUNS AND PATTERNS
    # =========================================================================
    # Monotonic runs
    if n > 1:
        diffs = np.diff(arr.astype(np.int64))
        signs = np.sign(diffs)
        sign_changes = np.where(signs[:-1] != signs[1:])[0]
        
        if len(sign_changes) > 0:
            run_lengths = np.diff(np.concatenate([[0], sign_changes + 1, [len(signs)]]))
            features["run_mean"] = float(np.mean(run_lengths))
            features["run_std"] = float(np.std(run_lengths))
            features["run_max"] = int(np.max(run_lengths))
        else:
            features["run_mean"] = float(n)
            features["run_std"] = 0.0
            features["run_max"] = n
    
    # =========================================================================
    # 7. GAP PATTERNS (spacing between same values)
    # =========================================================================
    # How often do we see the same value again?
    from collections import defaultdict
    last_seen = {}
    gaps = []
    for i, v in enumerate(arr):
        if v in last_seen:
            gaps.append(i - last_seen[v])
        last_seen[v] = i
    
    if gaps:
        features["repeat_gap_mean"] = float(np.mean(gaps))
        features["repeat_gap_std"] = float(np.std(gaps))
        features["repeat_gap_min"] = int(np.min(gaps))
    else:
        features["repeat_gap_mean"] = float(n)
        features["repeat_gap_std"] = 0.0
        features["repeat_gap_min"] = n
    
    # Birthday paradox: expected gap for 1000 values is ~40 draws
    # Significant deviation might indicate PRNG structure
    features["repeat_rate"] = float(len(gaps) / n) if n > 0 else 0.0
    
    return features


def entropy_of_values(arr: np.ndarray) -> float:
    """Calculate entropy of value distribution."""
    unique, counts = np.unique(arr, return_counts=True)
    probs = counts / len(arr)
    return float(-np.sum(probs * np.log2(probs + 1e-10)))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Safe correlation that handles edge cases."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    try:
        corr = np.corrcoef(a.astype(float), b.astype(float))[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0
    except:
        return 0.0


# =============================================================================
# FINGERPRINT GENERATION (same structure as original)
# =============================================================================

def generate_seeds(n: int, strategy: str, seed_range: tuple) -> List[int]:
    min_seed, max_seed = seed_range
    if strategy == "distributed":
        step = (max_seed - min_seed) // (n + 1)
        return [min_seed + step * (i + 1) for i in range(n)]
    elif strategy == "random":
        np.random.seed(42)
        return list(np.random.randint(min_seed, max_seed, size=n))
    else:
        return list(range(min_seed, min_seed + n))


def get_base_prng_name(prng_id: str) -> str:
    base = prng_id
    if base.endswith('_reverse'):
        base = base[:-8]
    if base.endswith('_hybrid'):
        base = base[:-7]
    return base


def generate_prng_sequences(prng_id: str, config: FingerprintConfig) -> Dict[str, Any]:
    try:
        prng_info = get_kernel_info(prng_id)
        if 'cpu_reference' in prng_info:
            cpu_func = prng_info['cpu_reference']
        else:
            base_prng = get_base_prng_name(prng_id)
            if base_prng != prng_id:
                base_info = get_kernel_info(base_prng)
                if 'cpu_reference' in base_info:
                    cpu_func = base_info['cpu_reference']
                else:
                    return None
            else:
                return None
    except Exception as e:
        print(f"  ⚠️  Error: {e}")
        return None
    
    seeds = generate_seeds(config.sequences_per_prng, config.seed_strategy, config.seed_range)
    sequences = []
    all_features = []
    
    for seed in seeds:
        try:
            raw_draws = cpu_func(seed, config.draws_per_sequence, skip=0)
            if isinstance(raw_draws, np.ndarray):
                draws = (raw_draws % config.mod).tolist()
            else:
                draws = [d % config.mod for d in raw_draws]
            
            features = extract_fingerprint_features(draws, config)
            features["seed"] = seed
            sequences.append({
                "seed": seed,
                "draws": None,
                "draw_count": len(draws),
                "features": features
            })
            all_features.append(features)
        except Exception as e:
            print(f"  ⚠️  Seed {seed} failed: {e}")
            continue
    
    if not sequences:
        return None
    
    aggregated = aggregate_features(all_features)
    
    return {
        "prng_id": prng_id,
        "sequences": sequences,
        "aggregated_fingerprint": aggregated,
        "generated_at": datetime.now().isoformat()
    }


def aggregate_features(feature_list: List[Dict]) -> Dict[str, Any]:
    if not feature_list:
        return {}
    
    feature_names = [k for k in feature_list[0].keys() if k != 'seed']
    aggregated = {}
    feature_vector = []
    feature_vector_names = []
    
    for name in feature_names:
        values = [f[name] for f in feature_list if name in f]
        if not values:
            continue
        
        arr = np.array(values)
        aggregated[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr))
        }
        feature_vector.append(float(np.mean(arr)))
        feature_vector_names.append(name)
    
    aggregated["_feature_vector"] = feature_vector
    aggregated["_feature_names"] = feature_vector_names
    
    return aggregated


def generate_library(output_dir: Path, config: FingerprintConfig, prng_filter=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "computed_fingerprints").mkdir(exist_ok=True)
    (output_dir / "ml_training_set").mkdir(exist_ok=True)
    
    all_prngs = list_available_prngs()
    prng_ids = [p for p in all_prngs if '_reverse' not in p]
    
    if prng_filter:
        prng_ids = [p for p in prng_ids if p in prng_filter]
    
    print(f"\n{'='*60}")
    print(f"MOD1000-SPECIFIC FINGERPRINT GENERATOR")
    print(f"{'='*60}")
    print(f"PRNGs: {len(prng_ids)}")
    print(f"Sequences per PRNG: {config.sequences_per_prng}")
    print(f"Draws per sequence: {config.draws_per_sequence}")
    print(f"{'='*60}\n")
    
    results = {}
    ml_rows = []
    
    for i, prng_id in enumerate(prng_ids):
        print(f"[{i+1}/{len(prng_ids)}] {prng_id}...")
        result = generate_prng_sequences(prng_id, config)
        
        if result is None:
            print(f"  ❌ Failed")
            continue
        
        results[prng_id] = result
        
        # Save fingerprint
        fp_file = output_dir / "computed_fingerprints" / f"{prng_id}_fingerprint.json"
        with open(fp_file, 'w') as f:
            json.dump({
                "prng_id": prng_id,
                "version": "2.0.0-mod1000",
                "feature_count": len(result["aggregated_fingerprint"].get("_feature_vector", [])),
                "fingerprint": result["aggregated_fingerprint"],
                "generated_at": result["generated_at"]
            }, f, indent=2)
        
        # ML rows
        for j, seq in enumerate(result["sequences"]):
            row = {"prng_id": prng_id, "sequence_id": f"{prng_id}_{j:03d}", "seed": seq["seed"]}
            for fname, fval in seq["features"].items():
                if fname != "seed" and not isinstance(fval, (list, dict)):
                    row[fname] = fval
            ml_rows.append(row)
        
        print(f"  ✅ {len(result['aggregated_fingerprint'].get('_feature_vector', []))} features")
    
    # Save ML CSV
    if ml_rows:
        csv_file = output_dir / "ml_training_set" / "fingerprint_features.csv"
        all_cols = set()
        for row in ml_rows:
            all_cols.update(row.keys())
        cols = ["prng_id", "sequence_id", "seed"] + sorted([c for c in all_cols if c not in ["prng_id", "sequence_id", "seed"]])
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(ml_rows)
        print(f"\n✅ ML CSV: {csv_file} ({len(ml_rows)} rows, {len(cols)} cols)")
    
    # Save registry
    registry = {
        "version": "2.0.0-mod1000",
        "feature_set": "mod1000_specific",
        "generated_at": datetime.now().isoformat(),
        "prngs": {p: {"fingerprint_file": f"computed_fingerprints/{p}_fingerprint.json"} for p in results}
    }
    with open(output_dir / "fingerprint_registry.json", 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: {len(results)} PRNGs processed")
    print(f"{'='*60}\n")
    
    return registry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mod1000-specific fingerprints")
    parser.add_argument("--output-dir", default="fingerprints_mod1000/")
    parser.add_argument("--sequences", type=int, default=20)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--prng", nargs="+")
    parser.add_argument("--quick-test", action="store_true")
    
    args = parser.parse_args()
    
    if args.quick_test:
        config = FingerprintConfig(sequences_per_prng=5, draws_per_sequence=5000)
        prng_filter = ["java_lcg", "mt19937", "xorshift64"]
    else:
        config = FingerprintConfig(sequences_per_prng=args.sequences, draws_per_sequence=args.draws)
        prng_filter = args.prng
    
    generate_library(Path(args.output_dir), config, prng_filter)
