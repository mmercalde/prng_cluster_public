#!/usr/bin/env python3
"""
PRNG Classifier - Step 0 of Autonomous Pipeline
================================================
Version: 1.1.0
Date: 2025-12-26

Compares unknown data stream against pre-computed fingerprint library.
Outputs hypothesis weights for downstream pipeline scoping.

This is the operational core of the "Heuristic Fingerprint Machine":
- Loads 46 pre-computed PRNG behavioral signatures
- Extracts features from unknown lottery stream
- Computes similarity to all reference fingerprints
- Outputs probability distribution over PRNG hypotheses

Usage:
    # Standard classification
    python3 prng_classifier.py \\
        --lottery-file daily3.json \\
        --fingerprint-dir fingerprints/ \\
        --output hypothesis_weights.json

    # With trigger reason
    python3 prng_classifier.py \\
        --lottery-file daily3.json \\
        --fingerprint-dir fingerprints/ \\
        --trigger regime_change

    # Custom thresholds
    python3 prng_classifier.py \\
        --lottery-file daily3.json \\
        --fingerprint-dir fingerprints/ \\
        --draws 200 \\
        --threshold 0.05 \\
        --max-hypotheses 10

Prerequisites:
    Fingerprint library must be generated first:
    python3 generate_fingerprint_library.py --output-dir fingerprints/

Output:
    Schema-compliant hypothesis_weights.json following
    schemas/hypothesis_weights_schema.json
"""

import argparse
import json
import time
import hashlib
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field

# Optional: integration with existing metadata writer
try:
    from integration.metadata_writer import inject_agent_metadata
    METADATA_WRITER_AVAILABLE = True
except ImportError:
    METADATA_WRITER_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ClassificationConfig:
    """Configuration for PRNG classification."""
    
    # Analysis parameters
    draws_to_analyze: int = 100
    
    # Output filtering
    min_weight_threshold: float = 0.10
    max_hypotheses_output: int = 5
    
    # Similarity computation
    similarity_method: str = "euclidean"  # "cosine", "euclidean", "correlation" - euclidean works best
    
    # Feature extraction (MUST match fingerprint library exactly!)
    residue_mods: List[int] = field(default_factory=lambda: [8, 125, 1000])
    max_lag_autocorr: int = 10  # Library uses 10
    power_of_2_mods: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32])  # Library uses all 5


@dataclass
class PRNGHypothesis:
    """Single PRNG hypothesis with evidence."""
    prng_id: str
    weight: float
    rank: int
    evidence: Dict[str, float]
    execution_time_ms: int = 0


# =============================================================================
# FEATURE EXTRACTION (Team Beta Approved - v2.0)
# =============================================================================

class FeatureExtractor:
    """
    Extracts discriminative fingerprint features for PRNG classification.
    
    Team Beta approved feature set (~33 features):
    - Permutation structure (TestU01 adaptation)
    - Transition geometry (JS divergence)
    - CRT-lane dynamics
    - Minimal correlation profile
    - Run/texture features
    
    Must match generate_fingerprint_library.py exactly!
    """
    
    def __init__(self, config: ClassificationConfig):
        self.config = config
    
    def extract(self, draws: List[int]) -> Dict[str, Any]:
        """Extract fingerprint features from draw sequence."""
        arr = np.array(draws, dtype=np.int64)
        n = len(arr)
        
        features = {}
        feature_vector = []
        feature_names = []
        
        # =====================================================================
        # 1. PERMUTATION TESTS (TestU01 adaptation) - HIGH VALUE
        # =====================================================================
        perm_map_3 = {(0,1,2):0, (0,2,1):1, (1,0,2):2, (1,2,0):3, (2,0,1):4, (2,1,0):5}
        
        if n >= 3:
            perm_counts_3 = [0] * 6
            for i in range(n - 2):
                triple = arr[i:i+3]
                order = tuple(np.argsort(triple))
                perm_counts_3[perm_map_3[order]] += 1
            total_3 = sum(perm_counts_3)
            perm_probs_3 = [c / total_3 for c in perm_counts_3]
            
            perm_entropy_3 = -sum(p * np.log2(p + 1e-10) for p in perm_probs_3)
            features["permutation_entropy_len3"] = float(perm_entropy_3)
            
            expected_3 = total_3 / 6
            chi2_3 = sum((c - expected_3)**2 / expected_3 for c in perm_counts_3)
            features["permutation_chi2_len3"] = float(chi2_3)
        else:
            features["permutation_entropy_len3"] = 0.0
            features["permutation_chi2_len3"] = 0.0
        
        feature_vector.extend([features["permutation_entropy_len3"], features["permutation_chi2_len3"]])
        feature_names.extend(["permutation_entropy_len3", "permutation_chi2_len3"])
        
        if n >= 4:
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
        else:
            features["permutation_entropy_len4"] = 0.0
        
        feature_vector.append(features["permutation_entropy_len4"])
        feature_names.append("permutation_entropy_len4")
        
        # Delta permutation
        if n >= 4:
            diffs = arr[1:] - arr[:-1]
            perm_counts_d = [0] * 6
            for i in range(len(diffs) - 2):
                triple = diffs[i:i+3]
                order = tuple(np.argsort(triple))
                perm_counts_d[perm_map_3[order]] += 1
            total_d = sum(perm_counts_d)
            if total_d > 0:
                perm_probs_d = [c / total_d for c in perm_counts_d]
                perm_entropy_d = -sum(p * np.log2(p + 1e-10) for p in perm_probs_d)
                features["permutation_entropy_delta_mod8"] = float(perm_entropy_d)
            else:
                features["permutation_entropy_delta_mod8"] = 0.0
        else:
            features["permutation_entropy_delta_mod8"] = 0.0
        
        feature_vector.append(features["permutation_entropy_delta_mod8"])
        feature_names.append("permutation_entropy_delta_mod8")
        
        # =====================================================================
        # 2. TRANSITION MATRIX FEATURES (Serial test adaptation) - HIGH VALUE
        # =====================================================================
        if n > 2:
            # Mod 8 transition matrix
            counts = np.ones((8, 8), dtype=np.float64)  # Laplace smoothing
            for i in range(n - 1):
                a, b = arr[i] % 8, arr[i+1] % 8
                counts[a, b] += 1
            row_sums = counts.sum(axis=1, keepdims=True)
            P8 = counts / row_sums
            
            features["lane_mod8_transition_entropy"] = float(-np.sum(P8 * np.log2(P8 + 1e-10)) / 8)
            features["lane_mod8_diagonal_dominance"] = float(np.trace(P8) / 8)
            
            row_entropies = [-np.sum(row * np.log2(row + 1e-10)) for row in P8]
            features["js_pair_mod8_row_entropy_mean"] = float(np.mean(row_entropies))
            features["js_pair_mod8_row_entropy_std"] = float(np.std(row_entropies))
            
            # Lag 2 diagonal
            counts_lag2 = np.ones((8, 8), dtype=np.float64)
            for i in range(n - 2):
                a, b = arr[i] % 8, arr[i+2] % 8
                counts_lag2[a, b] += 1
            P8_lag2 = counts_lag2 / counts_lag2.sum(axis=1, keepdims=True)
            features["js_pair_mod8_lag2_diag"] = float(np.trace(P8_lag2) / 8)
            
            # Mod 125 (compressed)
            counts_125 = np.ones((125, 125), dtype=np.float64)
            for i in range(n - 1):
                a, b = arr[i] % 125, arr[i+1] % 125
                counts_125[a, b] += 1
            P125 = counts_125 / counts_125.sum(axis=1, keepdims=True)
            features["js_pair_mod125_entropy"] = float(-np.sum(P125 * np.log2(P125 + 1e-10)) / 125)
        else:
            features["lane_mod8_transition_entropy"] = 0.0
            features["lane_mod8_diagonal_dominance"] = 0.125
            features["js_pair_mod8_row_entropy_mean"] = 0.0
            features["js_pair_mod8_row_entropy_std"] = 0.0
            features["js_pair_mod8_lag2_diag"] = 0.125
            features["js_pair_mod125_entropy"] = 0.0
        
        feature_vector.extend([
            features["lane_mod8_transition_entropy"],
            features["lane_mod8_diagonal_dominance"],
            features["js_pair_mod8_row_entropy_mean"],
            features["js_pair_mod8_row_entropy_std"],
            features["js_pair_mod8_lag2_diag"],
            features["js_pair_mod125_entropy"]
        ])
        feature_names.extend([
            "lane_mod8_transition_entropy",
            "lane_mod8_diagonal_dominance",
            "js_pair_mod8_row_entropy_mean",
            "js_pair_mod8_row_entropy_std",
            "js_pair_mod8_lag2_diag",
            "js_pair_mod125_entropy"
        ])
        
        # =====================================================================
        # 3. SPACING FEATURES (Birthday spacing concept) - HIGH VALUE
        # =====================================================================
        if n > 10:
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
                gap_hist, _ = np.histogram(all_gaps, bins=min(50, max(1, len(set(all_gaps)))), density=True)
                gap_hist_nz = gap_hist[gap_hist > 0]
                features["spacing_entropy_mod8"] = float(-np.sum(gap_hist_nz * np.log2(gap_hist_nz))) if len(gap_hist_nz) > 0 else 0.0
            else:
                features["spacing_mean_mod8"] = 8.0
                features["spacing_std_mod8"] = 0.0
                features["spacing_entropy_mod8"] = 0.0
        else:
            features["spacing_mean_mod8"] = 8.0
            features["spacing_std_mod8"] = 0.0
            features["spacing_entropy_mod8"] = 0.0
        
        feature_vector.extend([features["spacing_mean_mod8"], features["spacing_std_mod8"], features["spacing_entropy_mod8"]])
        feature_names.extend(["spacing_mean_mod8", "spacing_std_mod8", "spacing_entropy_mod8"])
        
        # =====================================================================
        # 4. CORRELATION FEATURES (trimmed to lag 1-3)
        # =====================================================================
        if n > 1:
            corr = np.corrcoef(arr[:-1], arr[1:])[0, 1]
            features["adjacent_correlation"] = float(corr) if not np.isnan(corr) else 0.0
        else:
            features["adjacent_correlation"] = 0.0
        feature_vector.append(features["adjacent_correlation"])
        feature_names.append("adjacent_correlation")
        
        for lag in [1, 2, 3]:
            if n > lag:
                corr = np.corrcoef(arr[:-lag], arr[lag:])[0, 1]
                features[f"autocorr_lag{lag}"] = float(corr) if not np.isnan(corr) else 0.0
            else:
                features[f"autocorr_lag{lag}"] = 0.0
            feature_vector.append(features[f"autocorr_lag{lag}"])
            feature_names.append(f"autocorr_lag{lag}")
        
        # Lane-specific autocorrelation
        for m in [8, 125]:
            if n > 1:
                lane = (arr % m).astype(np.float64)
                corr = np.corrcoef(lane[:-1], lane[1:])[0, 1]
                features[f"lane_mod{m}_autocorr"] = float(corr) if not np.isnan(corr) else 0.0
            else:
                features[f"lane_mod{m}_autocorr"] = 0.0
            feature_vector.append(features[f"lane_mod{m}_autocorr"])
            feature_names.append(f"lane_mod{m}_autocorr")
        
        # =====================================================================
        # 5. DELTA TEXTURE
        # =====================================================================
        if n > 1:
            diffs = np.diff(arr)
            features["diff_std"] = float(np.std(diffs))
            
            diff_mod = np.abs(diffs) % 1000
            diff_hist, _ = np.histogram(diff_mod, bins=100, density=True)
            diff_hist_nz = diff_hist[diff_hist > 0]
            features["diff_entropy"] = float(-np.sum(diff_hist_nz * np.log2(diff_hist_nz + 1e-10)))
            
            small_delta_count = np.sum(np.abs(diffs) <= 100)
            features["small_delta_ratio"] = float(small_delta_count / len(diffs))
            
            diff_mod8 = diffs % 8
            hist8, _ = np.histogram(diff_mod8, bins=8, range=(0, 8), density=True)
            hist8_nz = hist8[hist8 > 0]
            features["diff_mod8_entropy"] = float(-np.sum(hist8_nz * np.log2(hist8_nz))) if len(hist8_nz) > 0 else 0.0
        else:
            features["diff_std"] = 0.0
            features["diff_entropy"] = 0.0
            features["small_delta_ratio"] = 0.0
            features["diff_mod8_entropy"] = 0.0
        
        feature_vector.extend([features["diff_std"], features["diff_entropy"], 
                              features["small_delta_ratio"], features["diff_mod8_entropy"]])
        feature_names.extend(["diff_std", "diff_entropy", "small_delta_ratio", "diff_mod8_entropy"])
        
        # =====================================================================
        # 6. RUN / MONOTONIC STRUCTURE
        # =====================================================================
        runs = self._compute_run_lengths(arr)
        features["run_mean"] = float(np.mean(runs))
        features["run_std"] = float(np.std(runs))
        features["run_max"] = int(np.max(runs))
        
        if n > 1:
            diffs = np.diff(arr)
            increasing = np.sum(diffs > 0)
            decreasing = np.sum(diffs < 0)
            features["monotonic_ratio"] = float(increasing / (increasing + decreasing + 1e-10))
        else:
            features["monotonic_ratio"] = 0.5
        
        feature_vector.extend([features["run_mean"], features["run_std"], 
                              float(features["run_max"]), features["monotonic_ratio"]])
        feature_names.extend(["run_mean", "run_std", "run_max", "monotonic_ratio"])
        
        # =====================================================================
        # 7. POWER-OF-2 BIAS (mod 2, 4, 8 only)
        # =====================================================================
        for power in [2, 4, 8]:
            mod_vals = arr % power
            expected_uniform = n / power
            observed = np.bincount(mod_vals, minlength=power)
            bias = float(np.max(np.abs(observed - expected_uniform)) / expected_uniform)
            features[f"mod{power}_bias"] = bias
            feature_vector.append(bias)
            feature_names.append(f"mod{power}_bias")
        
        features["_feature_vector"] = feature_vector
        features["_feature_names"] = feature_names
        
        return features
    
    def _compute_run_lengths(self, arr: np.ndarray) -> np.ndarray:
        """Compute lengths of monotonic runs."""
        if len(arr) < 2:
            return np.array([1])
        
        diffs = np.diff(arr)
        signs = np.sign(diffs)
        
        sign_changes = np.where(signs[:-1] != signs[1:])[0]
        
        if len(sign_changes) == 0:
            return np.array([len(arr)])
        
        runs = np.diff(np.concatenate([[0], sign_changes + 1, [len(signs)]]))
        return runs


# =============================================================================
# SIMILARITY COMPUTATION
# =============================================================================
# =============================================================================
# SIMILARITY COMPUTATION
# =============================================================================

class SimilarityComputer:
    """Computes similarity between feature vectors."""
    
    def __init__(self, method: str = "cosine"):
        self.method = method
    
    def compute(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute similarity between two feature vectors.
        
        Returns value in range [0, 1] where 1 = identical.
        """
        v1 = np.array(vec1, dtype=np.float64)
        v2 = np.array(vec2, dtype=np.float64)
        
        # Handle length mismatch
        min_len = min(len(v1), len(v2))
        v1 = v1[:min_len]
        v2 = v2[:min_len]
        
        # Handle NaN/Inf
        v1 = np.nan_to_num(v1, nan=0.0, posinf=0.0, neginf=0.0)
        v2 = np.nan_to_num(v2, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.method == "cosine":
            return self._cosine_similarity(v1, v2)
        elif self.method == "euclidean":
            return self._euclidean_similarity(v1, v2)
        elif self.method == "correlation":
            return self._correlation_similarity(v1, v2)
        else:
            raise ValueError(f"Unknown similarity method: {self.method}")
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Cosine similarity: dot(v1, v2) / (|v1| * |v2|)"""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(v1, v2) / (norm1 * norm2)
        
        # Ensure in [0, 1] range
        return float(max(0.0, min(1.0, (similarity + 1) / 2)))
    
    def _euclidean_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Inverse euclidean distance: 1 / (1 + distance)"""
        distance = np.linalg.norm(v1 - v2)
        return float(1.0 / (1.0 + distance))
    
    def _correlation_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Pearson correlation mapped to [0, 1]"""
        if len(v1) < 2:
            return 0.5
        
        try:
            corr = np.corrcoef(v1, v2)[0, 1]
            if np.isnan(corr):
                return 0.5
            # Map [-1, 1] to [0, 1]
            return float((corr + 1) / 2)
        except:
            return 0.5


# =============================================================================
# MAIN CLASSIFIER
# =============================================================================

class PRNGClassifier:
    """
    Compares unknown stream against pre-computed fingerprint library.
    
    This is the core of Step 0 in the autonomous pipeline.
    """
    
    def __init__(
        self, 
        fingerprint_dir: str = "fingerprints/",
        config: ClassificationConfig = None
    ):
        self.fingerprint_dir = Path(fingerprint_dir)
        self.config = config or ClassificationConfig()
        
        # Initialize components
        self.feature_extractor = FeatureExtractor(self.config)
        self.similarity_computer = SimilarityComputer(self.config.similarity_method)
        
        # Load fingerprint library
        self.reference_fingerprints = self._load_fingerprints()
        
    def _load_fingerprints(self) -> Dict[str, Dict]:
        """Load pre-computed fingerprints from library."""
        fingerprints = {}
        
        fp_dir = self.fingerprint_dir / "computed_fingerprints"
        registry_file = self.fingerprint_dir / "fingerprint_registry.json"
        
        # Check library exists
        if not fp_dir.exists():
            raise FileNotFoundError(
                f"\n{'='*60}\n"
                f"FINGERPRINT LIBRARY NOT FOUND\n"
                f"{'='*60}\n"
                f"Expected: {fp_dir}\n\n"
                f"Generate the library first:\n"
                f"  python3 generate_fingerprint_library.py --output-dir {self.fingerprint_dir}\n"
                f"{'='*60}\n"
            )
        
        # Load registry for metadata
        registry = {}
        if registry_file.exists():
            with open(registry_file) as f:
                registry = json.load(f)
        
        # Load individual fingerprints
        for fp_file in sorted(fp_dir.glob("*_fingerprint.json")):
            try:
                with open(fp_file) as f:
                    fp = json.load(f)
                
                prng_id = fp["prng_id"]
                fingerprint_data = fp.get("fingerprint", fp.get("aggregated_fingerprint", {}))
                
                # Extract feature vector
                if "_feature_vector" not in fingerprint_data:
                    print(f"  Warning: {prng_id} has no feature vector, skipping")
                    continue
                
                feature_vector = fingerprint_data["_feature_vector"]
                feature_names = fingerprint_data.get("_feature_names", [])
                
                # Extract std values for normalization
                # The library stores each feature as {"mean": x, "std": y}
                std_vector = []
                for name in feature_names:
                    if name in fingerprint_data and isinstance(fingerprint_data[name], dict):
                        std_vector.append(fingerprint_data[name].get("std", 1.0))
                    elif name.startswith("mod") and "_distribution_" in name:
                        # Distribution bins - get from parent
                        base_name = name.rsplit("_", 1)[0]  # e.g., "mod8_distribution"
                        if base_name in fingerprint_data and isinstance(fingerprint_data[base_name], dict):
                            idx = int(name.split("_")[-1])
                            std_list = fingerprint_data[base_name].get("std", [1.0])
                            if idx < len(std_list):
                                std_vector.append(std_list[idx])
                            else:
                                std_vector.append(1.0)
                        else:
                            std_vector.append(1.0)
                    else:
                        std_vector.append(1.0)
                
                # If std_vector is wrong length, fall back to ones
                if len(std_vector) != len(feature_vector):
                    std_vector = [1.0] * len(feature_vector)
                
                fingerprints[prng_id] = {
                    "feature_vector": feature_vector,
                    "std_vector": std_vector,
                    "feature_names": feature_names,
                    "source_sequences": fp.get("source_sequences", 0),
                    "total_draws": fp.get("total_draws", 0),
                    "file": str(fp_file)
                }
                
            except Exception as e:
                print(f"  Warning: Failed to load {fp_file}: {e}")
                continue
        
        if not fingerprints:
            raise ValueError(
                f"No valid fingerprints found in {fp_dir}\n"
                f"Re-generate the library with:\n"
                f"  python3 generate_fingerprint_library.py --output-dir {self.fingerprint_dir}"
            )
        
        print(f"Loaded {len(fingerprints)} reference fingerprints")
        return fingerprints
    
    def classify(
        self, 
        lottery_data: List[Dict], 
        trigger: str = "cold_start",
        lottery_file: str = None
    ) -> Dict:
        """
        Main classification entry point.
        
        Args:
            lottery_data: List of draw records with 'draw' field
            trigger: What triggered this classification
            lottery_file: Path to source file (for metadata)
            
        Returns:
            Complete hypothesis_weights.json structure
        """
        start_time = time.time()
        
        # Extract draws from lottery data
        draws = self._extract_draws(lottery_data)
        
        # Limit to configured window
        analysis_draws = draws[-self.config.draws_to_analyze:]
        
        print(f"Analyzing {len(analysis_draws)} draws against {len(self.reference_fingerprints)} PRNGs...")
        
        # Extract features from unknown stream
        unknown_features = self.feature_extractor.extract(analysis_draws)
        unknown_vector = unknown_features["_feature_vector"]
        
        # Compare against all reference fingerprints
        similarities = {}
        evidence_details = {}
        
        for prng_id, ref_fp in self.reference_fingerprints.items():
            ref_vector = ref_fp["feature_vector"]
            std_vector = ref_fp.get("std_vector", [1.0] * len(ref_vector))
            
            # Use z-score normalized distance
            similarity = self._compute_normalized_similarity(
                unknown_vector, ref_vector, std_vector
            )
            similarities[prng_id] = similarity
            
            # Store evidence (top contributing features)
            evidence_details[prng_id] = self._compute_evidence(
                unknown_vector, 
                ref_vector, 
                unknown_features.get("_feature_names", [])
            )
        
        # Normalize to probability weights
        weights = self._normalize_weights(similarities)
        
        # Build output
        execution_time = time.time() - start_time
        
        return self._build_output(
            weights=weights,
            evidence=evidence_details,
            trigger=trigger,
            draws_analyzed=len(analysis_draws),
            total_draws=len(draws),
            execution_time=execution_time,
            lottery_file=lottery_file
        )
    
    def _extract_draws(self, lottery_data: List[Dict]) -> List[int]:
        """Extract draw values from lottery data."""
        draws = []
        
        for entry in lottery_data:
            if isinstance(entry, dict):
                if "draw" in entry:
                    draws.append(int(entry["draw"]))
                elif "number" in entry:
                    draws.append(int(entry["number"]))
            elif isinstance(entry, (int, float)):
                draws.append(int(entry))
        
        if not draws:
            raise ValueError("No valid draw values found in lottery data")
        
        return draws
    
    def _compute_normalized_similarity(
        self, 
        unknown: List[float], 
        reference: List[float],
        std_vector: List[float]
    ) -> float:
        """
        Compute z-score normalized similarity.
        
        Each feature difference is divided by its std (from the library's 
        cross-sequence variance) so features with high variance don't dominate.
        
        Returns inverse distance (higher = more similar).
        """
        import numpy as np
        
        u = np.array(unknown)
        r = np.array(reference)
        s = np.array(std_vector)
        
        # Match lengths
        min_len = min(len(u), len(r), len(s))
        u = u[:min_len]
        r = r[:min_len]
        s = s[:min_len]
        
        # Replace zero/tiny std with 1.0 to avoid division issues
        s = np.where(s < 1e-6, 1.0, s)
        
        # Z-score normalized difference
        z_diff = (u - r) / s
        
        # Cap z-scores at ±3 to prevent outliers from dominating
        # Entropy features have artificially low std but high variance across seeds
        z_diff = np.clip(z_diff, -3.0, 3.0)
        
        # Normalized Euclidean distance
        distance = np.sqrt(np.mean(z_diff ** 2))
        
        # Convert to similarity (higher = more similar)
        # Using 1 / (1 + d) mapping
        similarity = 1.0 / (1.0 + distance)
        
        return float(similarity)
    
    def _compute_evidence(
        self, 
        unknown: List[float], 
        reference: List[float],
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Compute per-feature evidence scores."""
        evidence = {}
        
        min_len = min(len(unknown), len(reference), len(feature_names))
        
        for i in range(min_len):
            name = feature_names[i]
            u_val = unknown[i]
            r_val = reference[i]
            
            # Compute feature-level similarity
            if abs(r_val) > 1e-10:
                rel_diff = abs(u_val - r_val) / (abs(r_val) + 1e-10)
                feature_sim = max(0.0, 1.0 - rel_diff)
            else:
                feature_sim = 1.0 if abs(u_val) < 1e-10 else 0.0
            
            evidence[name] = round(feature_sim, 4)
        
        # Return top 5 most similar features
        sorted_evidence = sorted(evidence.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_evidence[:5])
    
    def _normalize_weights(self, similarities: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize similarities to probability weights using softmax with temperature.
        
        Temperature controls discrimination:
        - Low temperature (0.001) = amplify small differences (more discriminating)
        - High temperature (1.0) = more uniform distribution
        """
        import numpy as np
        
        scores = np.array(list(similarities.values()))
        keys = list(similarities.keys())
        
        # Z-score normalize first to handle different similarity scales
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        if std_score > 0:
            z_scores = (scores - mean_score) / std_score
        else:
            # All identical - return uniform
            n = len(similarities)
            return {k: 1.0 / n for k in similarities}
        
        # Softmax with temperature
        # Lower temperature = more discrimination
        temperature = 0.5  # Tuned for typical similarity spreads
        
        # Numerical stability: subtract max before exp
        z_shifted = z_scores - np.max(z_scores)
        exp_scores = np.exp(z_shifted / temperature)
        softmax_weights = exp_scores / np.sum(exp_scores)
        
        return {k: float(w) for k, w in zip(keys, softmax_weights)}
    
    def _build_output(
        self, 
        weights: Dict[str, float],
        evidence: Dict[str, Dict],
        trigger: str,
        draws_analyzed: int,
        total_draws: int,
        execution_time: float,
        lottery_file: str = None
    ) -> Dict:
        """Build schema-compliant output."""
        
        # Sort by weight descending
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        # Build hypotheses list
        hypotheses = []
        for rank, (prng_id, weight) in enumerate(sorted_weights, 1):
            if weight < self.config.min_weight_threshold:
                continue
            if len(hypotheses) >= self.config.max_hypotheses_output:
                break
            
            hypotheses.append({
                "prng_id": prng_id,
                "weight": round(weight, 4),
                "rank": rank,
                "evidence": evidence.get(prng_id, {}),
                "execution_time_ms": 0
            })
        
        # Calculate confidence
        if hypotheses:
            top_weight = hypotheses[0]["weight"]
            if len(hypotheses) > 1:
                gap = top_weight - hypotheses[1]["weight"]
                confidence = min(0.95, top_weight + gap * 0.5)
            else:
                confidence = min(0.95, top_weight * 1.2)
        else:
            confidence = 0.0
        
        # Build eliminated list
        eliminated = [
            {"prng_id": k, "weight": round(v, 4), "reason": "below_threshold"}
            for k, v in sorted_weights
            if v < self.config.min_weight_threshold
        ][:10]
        
        # Generate run ID
        run_id = f"step0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Build inputs metadata
        inputs = []
        if lottery_file:
            inputs.append({
                "path": lottery_file,
                "record_count": total_draws
            })
        
        return {
            "agent_metadata": {
                "pipeline_step": 0,
                "step_name": "prng_classification",
                "run_id": run_id,
                "parent_run_id": None,
                "timestamp": datetime.now().isoformat() + "Z",
                "inputs": inputs,
                "outputs": [{"path": "hypothesis_weights.json"}],
                "follow_up_agent": "window_optimizer_agent",
                "confidence": round(confidence, 4),
                "success_criteria_met": len(hypotheses) > 0 and confidence >= 0.15,
                "suggested_params": {
                    "prng_hypotheses": [h["prng_id"] for h in hypotheses],
                    "hypothesis_weights": [h["weight"] for h in hypotheses],
                    "recommended_primary": hypotheses[0]["prng_id"] if hypotheses else None
                },
                "cluster_resources": {
                    "nodes_used": ["local"],
                    "gpus_used": 0,
                    "execution_time_seconds": round(execution_time, 2)
                }
            },
            "classification_results": {
                "regime_id": "current",
                "trigger": trigger,
                "total_prngs_tested": len(weights),
                "prngs_above_threshold": len(hypotheses),
                "hypotheses": hypotheses,
                "eliminated": eliminated
            },
            "metadata": {
                "version": "1.1.0",
                "schema_ref": "schemas/hypothesis_weights_schema.json",
                "classification_config": {
                    "draws_to_analyze": self.config.draws_to_analyze,
                    "min_weight_threshold": self.config.min_weight_threshold,
                    "max_hypotheses_output": self.config.max_hypotheses_output,
                    "similarity_method": self.config.similarity_method
                },
                "draws_analyzed": draws_analyzed,
                "total_draws_available": total_draws,
                "execution_time_seconds": round(execution_time, 2),
                "fingerprint_library": str(self.fingerprint_dir)
            }
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRNG Classification - Step 0 of Autonomous Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic classification
  python3 prng_classifier.py --lottery-file daily3.json
  
  # With custom fingerprint directory
  python3 prng_classifier.py --lottery-file daily3.json --fingerprint-dir /path/to/fingerprints/
  
  # With trigger reason (for Watcher)
  python3 prng_classifier.py --lottery-file daily3.json --trigger regime_change
  
  # Custom thresholds
  python3 prng_classifier.py --lottery-file daily3.json --draws 200 --threshold 0.05
        """
    )
    
    parser.add_argument("--lottery-file", required=True,
                        help="Input lottery JSON file")
    parser.add_argument("--fingerprint-dir", default="fingerprints/",
                        help="Path to fingerprint library (default: fingerprints/)")
    parser.add_argument("--output", default="hypothesis_weights.json",
                        help="Output file (default: hypothesis_weights.json)")
    parser.add_argument("--trigger", default="cold_start",
                        choices=["cold_start", "regime_change", "confidence_decay", "overlap_collapse", "manual"],
                        help="Classification trigger reason")
    parser.add_argument("--draws", type=int, default=100,
                        help="Number of recent draws to analyze (default: 100)")
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="Minimum weight threshold for output (default: 0.10)")
    parser.add_argument("--max-hypotheses", type=int, default=5,
                        help="Maximum hypotheses in output (default: 5)")
    parser.add_argument("--similarity", default="euclidean",
                        choices=["cosine", "euclidean", "correlation"],
                        help="Similarity method (default: cosine)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    
    args = parser.parse_args()
    
    # Load lottery data
    lottery_file = Path(args.lottery_file)
    if not lottery_file.exists():
        print(f"Error: Lottery file not found: {lottery_file}")
        sys.exit(1)
    
    with open(lottery_file) as f:
        lottery_data = json.load(f)
    
    if not args.quiet:
        print(f"\n{'='*60}")
        print("PRNG CLASSIFICATION - STEP 0")
        print(f"{'='*60}")
        print(f"Lottery file: {lottery_file}")
        print(f"Fingerprint library: {args.fingerprint_dir}")
        print(f"Trigger: {args.trigger}")
        print(f"{'='*60}\n")
    
    # Configure
    config = ClassificationConfig(
        draws_to_analyze=args.draws,
        min_weight_threshold=args.threshold,
        max_hypotheses_output=args.max_hypotheses,
        similarity_method=args.similarity
    )
    
    # Run classification
    try:
        classifier = PRNGClassifier(
            fingerprint_dir=args.fingerprint_dir,
            config=config
        )
        
        results = classifier.classify(
            lottery_data=lottery_data,
            trigger=args.trigger,
            lottery_file=str(lottery_file)
        )
        
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)
    except Exception as e:
        print(f"Classification failed: {e}")
        sys.exit(1)
    
    # Write output
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    hypotheses = results["classification_results"]["hypotheses"]
    confidence = results["agent_metadata"]["confidence"]
    exec_time = results["metadata"]["execution_time_seconds"]
    
    if not args.quiet:
        print(f"\n{'='*60}")
        print("CLASSIFICATION COMPLETE")
        print(f"{'='*60}")
        print(f"\nTop hypotheses:")
        for h in hypotheses:
            print(f"  {h['rank']}. {h['prng_id']:20s} {h['weight']:6.2%}")
        print(f"\nConfidence: {confidence:.2%}")
        print(f"Execution time: {exec_time:.2f}s")
        print(f"Output: {output_path}")
        print(f"\nSuccess: {results['agent_metadata']['success_criteria_met']}")
        print(f"Next agent: {results['agent_metadata']['follow_up_agent']}")
        print(f"{'='*60}\n")
    
    # Exit code based on success
    sys.exit(0 if results['agent_metadata']['success_criteria_met'] else 1)


if __name__ == "__main__":
    main()
