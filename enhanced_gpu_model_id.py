#!/usr/bin/env python3
# ===== ROCm/NVIDIA environment prelude (MUST be first, before importing cupy) =====
import os as _os, socket as _socket
_HOST = _socket.gethostname()
if _HOST in ("rig-6600", "rig-6600b", "rig-6600c"):
    _os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    _os.environ.setdefault("HSA_ENABLE_SDMA", "0")
    _os.environ.setdefault("ROCM_PATH", "/opt/rocm")
    _os.environ.setdefault("HIP_PATH", "/opt/rocm/hip")
    _os.environ.setdefault("CUPY_CACHE_DIR", _os.path.expanduser("~/distributed_prng_analysis/.cache/cupy"))

# enhanced_gpu_model_id.py
# MERGED VERSION: Contains BOTH draw matching system AND vectorized correlation analysis
# PRNG analysis – GPU-accelerated, host-sync free hot path

import os
import math
import time
import json
from typing import Dict, Any, List, Tuple

import cupy as cp
import numpy as np


# ============================================================================
# DRAW MATCHING SYSTEM - All original functions preserved
# ============================================================================

def _inverse_error_function(x):
    """Approximate inverse error function for p-value to z-score conversion"""
    import math
    try:
        if abs(x) >= 1.0:
            return 0.0
        a = 0.147
        sign = 1 if x >= 0 else -1
        x = abs(x)
        ln_term = math.log(1 - x * x)
        sqrt_term = math.sqrt((2.0 / (math.pi * a)) + (ln_term / 2.0))
        result = sign * math.sqrt(sqrt_term - (ln_term / a))
        return result
    except:
        return 0.0


def _adaptive_lag_analysis(seq: cp.ndarray, max_lag: int = None) -> Dict[str, Any]:
    """Comprehensive lag correlation analysis with automatic pattern detection."""
    seqf = seq.astype(cp.float32)
    n = seqf.size

    if max_lag is None:
        if n < 100:
            max_lag = min(20, n // 4)
        elif n < 1000:
            max_lag = min(50, n // 10)
        elif n < 10000:
            max_lag = min(100, n // 20)
        else:
            max_lag = min(200, n // 50)

    short_lags = list(range(1, min(11, max_lag + 1)))
    medium_lags = list(range(12, min(31, max_lag + 1), 2))
    long_lags = []

    potential_lags = [7, 14, 21, 28, 30, 60, 90, 180, 365]
    for lag in potential_lags:
        if lag <= max_lag and lag not in short_lags and lag not in medium_lags:
            long_lags.append(lag)

    all_lags = sorted(short_lags + medium_lags + long_lags)

    mean = cp.mean(seqf)
    std = cp.std(seqf) + 1e-12

    correlations = {}
    significant_lags = []
    correlation_magnitudes = []

    for L in all_lags:
        if L >= n:
            continue

        a = seqf[: n - L] - mean
        b = seqf[L:] - mean
        num = cp.sum(a * b)
        den = (n - L) * (std ** 2)
        r = float((num / den).get())

        correlations[f"lag_{L}"] = r
        correlation_magnitudes.append((L, abs(r), r))

        significance_threshold = 2.0 / math.sqrt(n - L) if n - L > 4 else 0.1
        if abs(r) > significance_threshold:
            significant_lags.append((L, r, abs(r)))

    correlation_magnitudes.sort(key=lambda x: x[2], reverse=True)
    significant_lags.sort(key=lambda x: x[2], reverse=True)

    pattern_analysis = {
        "immediate_dependencies": {},
        "short_term_patterns": {},
        "medium_term_patterns": {},
        "long_term_patterns": {}
    }

    for lag_name, corr_val in correlations.items():
        lag_num = int(lag_name.split('_')[1])
        if lag_num <= 5:
            pattern_analysis["immediate_dependencies"][lag_name] = corr_val
        elif lag_num <= 20:
            pattern_analysis["short_term_patterns"][lag_name] = corr_val
        elif lag_num <= 60:
            pattern_analysis["medium_term_patterns"][lag_name] = corr_val
        else:
            pattern_analysis["long_term_patterns"][lag_name] = corr_val

    return {
        "all_correlations": correlations,
        "top_correlations": correlation_magnitudes[:10],
        "significant_lags": significant_lags[:5],
        "pattern_analysis": pattern_analysis,
        "total_lags_tested": len(all_lags),
        "max_correlation": correlation_magnitudes[0] if correlation_magnitudes else (0, 0.0, 0.0),
        "correlation_summary": {
            "strongest_lag": correlation_magnitudes[0][0] if correlation_magnitudes else 0,
            "strongest_correlation": correlation_magnitudes[0][2] if correlation_magnitudes else 0.0,
            "num_significant": len(significant_lags)
        }
    }


def _enhanced_z_score_calculation(results_detail: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate comprehensive z-scores from enhanced correlation analysis."""
    if not results_detail:
        return {
            "z_scores_array": [0.0, 0.0, 0.0, 0.0],
            "z_scores_detailed": {},
            "correlation_z_scores": {},
            "statistical_significance": {}
        }

    result = results_detail[0]
    z_scores_array = []
    z_scores_detailed = {}
    correlation_z_scores = {}

    try:
        chi2_p = result.get("chi2_p_value", 0.5)
        chi2_z = _p_value_to_z_score(chi2_p)
        z_scores_array.append(chi2_z)
        z_scores_detailed["chi2_uniformity"] = chi2_z

        enhanced_lags = result.get("enhanced_lag_analysis", {})
        top_correlations = enhanced_lags.get("top_correlations", [])

        for i in range(3):
            if i < len(top_correlations):
                lag_num, abs_corr, corr_val = top_correlations[i]
                n = result.get("length", 1000)
                corr_z = corr_val * math.sqrt(n - lag_num - 1) if n > lag_num + 1 else corr_val
                z_scores_array.append(corr_z)
                z_scores_detailed[f"lag_{lag_num}_correlation"] = corr_z
                correlation_z_scores[f"lag_{lag_num}"] = {
                    "correlation": corr_val,
                    "z_score": corr_z,
                    "magnitude": abs_corr
                }
            else:
                z_scores_array.append(0.0)
                z_scores_detailed[f"lag_none_{i}"] = 0.0

        runs_data = result.get("runs_test", {})
        runs_p = runs_data.get("p_value", 0.5)
        runs_z = _p_value_to_z_score(runs_p)

        while len(z_scores_array) < 4:
            z_scores_array.append(0.0)
        z_scores_array = z_scores_array[:4]

        z_scores_array[3] = runs_z
        z_scores_detailed["runs_randomness"] = runs_z

        significance_summary = {
            "highly_significant": sum(1 for z in z_scores_array if abs(z) > 2.5),
            "moderately_significant": sum(1 for z in z_scores_array if 1.5 < abs(z) <= 2.5),
            "weakly_significant": sum(1 for z in z_scores_array if 0.5 < abs(z) <= 1.5),
            "non_significant": sum(1 for z in z_scores_array if abs(z) <= 0.5),
        }

        return {
            "z_scores_array": z_scores_array,
            "z_scores_detailed": z_scores_detailed,
            "correlation_z_scores": correlation_z_scores,
            "statistical_significance": significance_summary
        }

    except Exception as e:
        print(f"Error in enhanced z-score calculation: {e}")
        return {
            "z_scores_array": [0.0, 0.0, 0.0, 0.0],
            "z_scores_detailed": {"error": str(e)},
            "correlation_z_scores": {},
            "statistical_significance": {}
        }


def _p_value_to_z_score(p_value: float) -> float:
    """Convert p-value to approximate z-score using inverse normal"""
    try:
        if p_value <= 0.0:
            return 5.0
        elif p_value >= 1.0:
            return 0.0

        if p_value < 0.5:
            return -math.sqrt(2) * _inverse_error_function(1 - 2 * p_value)
        else:
            return math.sqrt(2) * _inverse_error_function(2 * p_value - 1)
    except:
        return 0.0


_XORSHIFT32_SRC = r'''
extern "C" __global__
void xorshift32_fill(unsigned int seed, unsigned int *out, const int n) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned int s = seed ^ (0x9E3779B9u + tid * 0x7f4a7c15u);

    for (int i = tid; i < n; i += stride) {
        s ^= s << 13;  s ^= s >> 17;  s ^= s << 5;
        out[i] = s;
    }
}
'''.strip()

_LCG32_SRC = r'''
extern "C" __global__
void lcg32_fill(unsigned int seed, unsigned int *out, const int n,
                const unsigned int a, const unsigned int c) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned int s = seed ^ (0x9E3779B9u + tid * 0x632be59bu);

    for (int i = tid; i < n; i += stride) {
        s = a * s + c;
        out[i] = s;
    }
}
'''.strip()

_xorshift32_k = cp.RawKernel(_XORSHIFT32_SRC, "xorshift32_fill")
_lcg32_k = cp.RawKernel(_LCG32_SRC, "lcg32_fill")


def _dev_info() -> Dict[str, Any]:
    try:
        d = cp.cuda.Device()
        props = d.attributes
        try:
            gp = cp.cuda.runtime.getDeviceProperties(d.id)
            nm = gp["name"]
            name = nm.decode() if isinstance(nm, (bytes, bytearray)) else nm
        except Exception:
            name = "GPU"
        total_mem = cp.cuda.runtime.memGetInfo()[1] / (1024**3)
        return {
            "id": d.id,
            "name": name,
            "total_mem_GB": round(total_mem, 2),
            "multiProcessorCount": props.get(16) if isinstance(props, dict) else None
        }
    except Exception:
        return {"id": None, "name": "Unknown", "total_mem_GB": None, "multiProcessorCount": None}


def _grid_for(n: int, block: int = 256) -> Tuple[int, int]:
    max_blocks = 65535
    blocks = min(max_blocks, (n + block - 1) // block)
    return blocks, block


def _convert_lottery_to_32bit(draws: List[int], method: str = 'hash') -> List[int]:
    """Convert lottery draws to 32-bit space using different methods"""
    converted = []

    if method == 'hash':
        import hashlib
        for draw in draws:
            draw_str = str(draw)
            hash_obj = hashlib.md5(draw_str.encode())
            hash_int = int(hash_obj.hexdigest()[:8], 16)
            converted.append(hash_int)
    elif method == 'polynomial':
        for i, draw in enumerate(draws):
            if isinstance(draw, str):
                draw = int(draw)
            poly_val = (draw * draw * draw * 1009) + (draw * draw * 2017) + (draw * 4093) + i
            converted.append(poly_val & 0xFFFFFFFF)
    elif method == 'xor_rotate':
        for i, draw in enumerate(draws):
            if isinstance(draw, str):
                draw = int(draw)
            xor_val = draw ^ (i * 0x9E3779B9)
            rotated = ((xor_val << 13) | (xor_val >> 19)) & 0xFFFFFFFF
            converted.append(rotated)
    elif method == 'multiply':
        for draw in draws:
            if isinstance(draw, str):
                draw = int(draw)
            converted_val = (draw * 4294967) & 0xFFFFFFFF
            converted.append(converted_val)
    else:
        return _convert_lottery_to_32bit(draws, 'hash')

    return converted


def _basic_mt_reconstruction(outputs: List[int]) -> Dict[str, Any]:
    """Basic MT19937 reconstruction analysis"""
    try:
        if len(outputs) < 50:
            return {
                "success": False,
                "error": "Insufficient data for MT reconstruction",
                "confidence": 0.0
            }

        arr = np.array(outputs, dtype=np.uint32)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        min_val = int(np.min(arr))
        max_val = int(np.max(arr))

        expected_mean = 2**31 - 1
        expected_std = 2**31 / np.sqrt(3)

        mean_score = max(0.0, 1.0 - abs(mean_val - expected_mean) / expected_mean)
        std_score = max(0.0, 1.0 - abs(std_val - expected_std) / expected_std)
        range_score = 1.0 if (min_val >= 0 and max_val <= 0xFFFFFFFF) else 0.0

        unique_ratio = len(np.unique(arr)) / len(arr)
        period_score = min(1.0, unique_ratio * 1.2)

        confidence = max(0.0, min(1.0, (mean_score + std_score + range_score + period_score) / 4.0))
        likely_mt = confidence > 0.7 and len(outputs) >= 100

        return {
            "success": True,
            "method": "basic_statistical_analysis",
            "confidence": confidence,
            "likely_mt": likely_mt,
            "statistics": {
                "sample_size": len(outputs),
                "mean": mean_val,
                "std": std_val,
                "min": min_val,
                "max": max_val,
                "unique_ratio": unique_ratio
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"MT reconstruction failed: {str(e)}",
            "confidence": 0.0
        }


def _attempt_mt_reconstruction(seeds: List[int], conversion_methods: List[str] = None) -> Dict[str, Any]:
    """Attempt MT reconstruction with multiple conversion methods"""
    if conversion_methods is None:
        conversion_methods = ['hash', 'polynomial', 'xor_rotate', 'multiply']

    best_result = {"success": False, "confidence": 0.0, "method": "none"}

    try:
        import sys
        modules_path = os.path.join(os.path.dirname(__file__), 'modules')
        if os.path.exists(modules_path) and modules_path not in sys.path:
            sys.path.append(modules_path)

        from mt_engine_exact import AdvancedMT19937Reconstructor
        reconstructor = AdvancedMT19937Reconstructor()
        use_advanced = True
        print("Using advanced MT reconstruction")

    except ImportError as e:
        print(f"Advanced MT modules not available: {e}, using basic reconstruction")
        reconstructor = None
        use_advanced = False

    for method in conversion_methods:
        try:
            converted_outputs = _convert_lottery_to_32bit(seeds[:100], method)

            if use_advanced and reconstructor:
                result = reconstructor.reconstruct_mt_state(converted_outputs)
                if result.get('success', False):
                    confidence = result.get('confidence', 0.0)
                    if confidence > best_result.get('confidence', 0.0):
                        best_result = result.copy()
                        best_result['conversion_method'] = method
                        best_result['converted_sample'] = converted_outputs[:10]
            else:
                result = _basic_mt_reconstruction(converted_outputs)
                if result.get('success', False):
                    confidence = result.get('confidence', 0.0)
                    if confidence > best_result.get('confidence', 0.0):
                        best_result = result.copy()
                        best_result['conversion_method'] = method
                        best_result['converted_sample'] = converted_outputs[:10]

        except Exception as e:
            print(f"MT reconstruction failed with {method}: {e}")
            continue

    return best_result


def _xorshift_cupy(n: int, seed: int) -> cp.ndarray:
    """Return uint32 sequence of length n using a fast kernel (no host sync)."""
    out = cp.empty(n, dtype=cp.uint32)
    grid, block = _grid_for(n)
    _xorshift32_k((grid,), (block,), (np.uint32(seed), out, np.int32(n)))
    return out


def _lcg_cupy(n: int, seed: int, a: int = 1664525, c: int = 1013904223) -> cp.ndarray:
    """Return uint32 sequence of length n using a fast kernel (no host sync)."""
    out = cp.empty(n, dtype=cp.uint32)
    grid, block = _grid_for(n)
    _lcg32_k((grid,), (block,),
             (np.uint32(seed), out, np.int32(n), np.uint32(a), np.uint32(c)))
    return out


def _apply_mapping_uint32_to_range(u: cp.ndarray, mapping: str, rng: int) -> cp.ndarray:
    """Map uint32 -> [0, rng) either by mod or scale."""
    if mapping == "mod":
        return (u % rng).astype(cp.int32, copy=False)
    if mapping == "scale":
        f = (u.astype(cp.float32) * (1.0 / 4294967296.0)) * float(rng)
        return cp.floor(f).astype(cp.int32, copy=False)
    return (u % rng).astype(cp.int32, copy=False)


def _chi2_uniform_test(seq: cp.ndarray, k: int = 100) -> Tuple[float, float]:
    """GPU chi-square against uniform buckets. Returns (stat, pvalue)."""
    mn = float(cp.min(seq).get())
    mx = float(cp.max(seq).get())
    if mx - mn <= 0.0:
        return 0.0, 1.0

    bins = cp.linspace(mn, mx + 1.0, k + 1, dtype=cp.float32)
    counts, _ = cp.histogram(seq.astype(cp.float32), bins=bins)
    n = int(seq.size)
    expected = n / float(k)

    diff = counts.astype(cp.float32) - expected
    chi2 = cp.sum((diff * diff) / expected, dtype=cp.float32)

    chi2_host = float(chi2.get())
    dof = max(1, k - 1)
    try:
        import scipy.stats as sps
        p = float(1.0 - sps.chi2.cdf(chi2_host, dof))
    except Exception:
        nu = float(dof)
        z = ((chi2_host / nu) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * nu))) / math.sqrt(2.0 / (9.0 * nu))
        p = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0)))))

    return chi2_host, p


def _runs_test(seq: cp.ndarray) -> Tuple[float, float]:
    """Wald–Wolfowitz runs test around median (GPU). Returns (statistic, pvalue)."""
    x = seq.astype(cp.float32)
    med = cp.median(x)
    signs = cp.where(x >= med, 1, 0).astype(cp.int32)

    n1 = int(cp.sum(signs).get())
    n2 = int(x.size) - n1
    if n1 == 0 or n2 == 0:
        return float(x.size), 1.0

    diffs = cp.diff(signs)
    runs = int(cp.sum(diffs != 0).get()) + 1

    mu = 1.0 + (2.0 * n1 * n2) / float(n1 + n2)
    var = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n1 - n2)) / (float((n1 + n2) ** 2) * (n1 + n2 - 1.0) + 1e-9)
    z = (runs - mu) / math.sqrt(max(var, 1e-12))

    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return float(runs), float(p)


def _build_sequence(prng_type: str, mapping_type: str, n: int, seed: int, rng: int, job_data: Dict[str, Any] = None) -> cp.ndarray:
    """Build one PRNG sequence on GPU and map to target range."""
    prng = prng_type.lower()
    map_t = mapping_type.lower()

    if prng in ("xorshift", "xorshift32"):
        u = _xorshift_cupy(n, seed)
    elif prng in ("lcg", "lcg32"):
        u = _lcg_cupy(n, seed)
    elif prng in ("mt", "mt19937"):
        try:
            print(f"Attempting MT reconstruction for seed {seed}")

            if job_data and 'seeds' in job_data:
                seeds = job_data['seeds']
                if isinstance(seeds, list) and len(seeds) > 1:
                    mt_result = _attempt_mt_reconstruction(seeds)

                    if mt_result.get('success', False) and mt_result.get('confidence', 0.0) > 0.5:
                        confidence = mt_result.get('confidence', 0.0)
                        method = mt_result.get('conversion_method', 'unknown')
                        likely_mt = mt_result.get('likely_mt', False)

                        print(f"MT reconstruction: method={method}, confidence={confidence:.3f}, likely_mt={likely_mt}")

                        if 'converted_sample' in mt_result and len(mt_result['converted_sample']) > 0:
                            converted_data = mt_result['converted_sample']
                            if len(converted_data) >= n:
                                u = cp.array(converted_data[:n], dtype=cp.uint32)
                            else:
                                rs = cp.random.RandomState(seed)
                                u = rs.randint(0, 2**32, size=n, dtype=cp.uint32)
                        else:
                            rs = cp.random.RandomState(seed)
                            u = rs.randint(0, 2**32, size=n, dtype=cp.uint32)
                    else:
                        print("MT reconstruction confidence too low, using CuPy MT19937")
                        rs = cp.random.RandomState(seed)
                        u = rs.randint(0, 2**32, size=n, dtype=cp.uint32)
                else:
                    rs = cp.random.RandomState(seed)
                    u = rs.randint(0, 2**32, size=n, dtype=cp.uint32)
            else:
                rs = cp.random.RandomState(seed)
                u = rs.randint(0, 2**32, size=n, dtype=cp.uint32)

        except Exception as e:
            print(f"MT reconstruction error: {e}, falling back to CuPy MT19937")
            rs = cp.random.RandomState(seed)
            u = rs.randint(0, 2**32, size=n, dtype=cp.uint32)
    else:
        u = _xorshift_cupy(n, seed)

    seq = _apply_mapping_uint32_to_range(u, map_t, rng)
    return seq


def _analyze_sequence(seq: cp.ndarray, take_hist_k: int = 100) -> Dict[str, Any]:
    """Compute stats on a sequence without leaving the GPU except for tiny scalars."""
    chi2_stat, chi2_p = _chi2_uniform_test(seq, k=take_hist_k)
    enhanced_lags = _adaptive_lag_analysis(seq)
    lcorrs = enhanced_lags["all_correlations"]
    runs_stat, runs_p = _runs_test(seq)

    mn_val = float(cp.min(seq).get())
    mx_val = float(cp.max(seq).get())
    if mx_val <= mn_val:
        bins = cp.linspace(mn_val, mn_val + 1.0, 101, dtype=cp.float32)
    else:
        bins = cp.linspace(mn_val, mx_val + 1.0, 101, dtype=cp.float32)
    counts, edges = cp.histogram(seq.astype(cp.float32), bins=bins)

    detailed = {
        "length": int(seq.size),
        "mean": float(cp.mean(seq).get()),
        "std": float(cp.std(seq).get()),
        "min": float(mn_val),
        "max": float(mx_val),
        "histogram": {
            "counts": counts.get().astype(np.int64).tolist(),
            "bin_edges": edges.get().astype(np.float64).tolist()
        },
        "chi2_stat": float(chi2_stat),
        "chi2_p_value": float(chi2_p),
        "lag_correlations": lcorrs,
        "enhanced_lag_analysis": enhanced_lags,
        "runs_test": {
            "statistic": float(runs_stat),
            "p_value": float(runs_p)
        }
    }
    return detailed


def _calculate_z_scores(results_detail: List[Dict[str, Any]]) -> List[float]:
    """Calculate proper z-scores from statistical test results"""
    if not results_detail:
        return [0.0, 0.0, 0.0, 0.0]

    result = results_detail[0]
    z_scores = []

    try:
        chi2_p = result.get("chi2_p_value", 0.5)
        if chi2_p > 0.0 and chi2_p < 1.0:
            if chi2_p < 0.5:
                z_chi2 = -math.sqrt(2) * _inverse_error_function(2 * chi2_p - 1)
            else:
                z_chi2 = math.sqrt(2) * _inverse_error_function(2 * chi2_p - 1)
        else:
            z_chi2 = 0.0
        z_scores.append(float(z_chi2))

        lag_corrs = result.get("lag_correlations", {})
        z_lag_1 = float(lag_corrs.get("lag_1", 0.0)) * 10.0
        z_lag_5 = float(lag_corrs.get("lag_5", 0.0)) * 10.0
        z_scores.append(z_lag_1)
        z_scores.append(z_lag_5)

        runs_data = result.get("runs_test", {})
        runs_p = runs_data.get("p_value", 0.5)
        if runs_p > 0.0 and runs_p < 1.0:
            if runs_p < 0.5:
                z_runs = -math.sqrt(2) * _inverse_error_function(2 * runs_p - 1)
            else:
                z_runs = math.sqrt(2) * _inverse_error_function(2 * runs_p - 1)
        else:
            z_runs = 0.0
        z_scores.append(float(z_runs))

    except Exception as e:
        print(f"Error calculating z-scores: {e}")
        return [0.0, 0.0, 0.0, 0.0]

    return z_scores


# ============================================================================
# VECTORIZED CORRELATION ANALYSIS SYSTEM - New high-performance functions
# ============================================================================

def _fp32c(a):
    """Convert array to float32 contiguous format"""
    a = cp.asarray(a, dtype=cp.float32)
    return a if a.flags.c_contiguous else cp.ascontiguousarray(a)


def gpu_corr_two(x, y):
    """Compute correlation between two 1D arrays"""
    x = _fp32c(x)
    y = _fp32c(y)
    xm = x - x.mean()
    ym = y - y.mean()
    xs = cp.sqrt(cp.sum(xm*xm).clip(1e-12))
    ys = cp.sqrt(cp.sum(ym*ym).clip(1e-12))
    return float(cp.sum(xm*ym) / (xs*ys))


def gpu_corr_matrix_rows(M):
    """Compute correlation matrix from row vectors
    M shape: (k, n) where k=number of series, n=samples per series
    Returns: (k, k) correlation matrix
    """
    M = _fp32c(M)
    M = M - M.mean(axis=1, keepdims=True)
    denom = cp.sqrt(cp.sum(M*M, axis=1, keepdims=True).clip(1e-12))
    Mn = M / denom
    C = cp.einsum('in,jn->ij', Mn, Mn)
    C = cp.clip(C, -1.0, 1.0)
    return C


def get_prng_state(prng_type: str, seed: int):
    """Generate initial PRNG state from seed"""
    if prng_type == "mt":
        rng = np.random.Generator(np.random.MT19937(seed))
        return int(rng.integers(0, 2**32))
    elif prng_type == "lcg":
        return seed
    elif prng_type == "xorshift":
        return seed if seed != 0 else 1
    else:
        raise ValueError(f"Unknown PRNG type: {prng_type}")


def generate_xorshift32_batch_gpu(seeds_list: list, samples: int, gpu_id: int):
    """GPU-accelerated batch XORshift32 generator - processes ALL seeds in parallel"""
    num_seeds = len(seeds_list)

    # Batch XORshift32 CUDA kernel - processes all seeds in parallel
    xorshift32_batch_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void xorshift32_batch_fill(unsigned int* output, const unsigned int* seeds, int num_seeds, int samples_per_seed) {
        int seed_idx = blockIdx.y;
        if (seed_idx >= num_seeds) return;

        int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        unsigned int state = seeds[seed_idx];

        for (int i = 0; i < sample_idx; i++) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
        }

        int output_offset = seed_idx * samples_per_seed;
        for (int i = sample_idx; i < samples_per_seed; i += stride) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            output[output_offset + i] = state;
        }
    }
    ''', 'xorshift32_batch_fill')

    gpu_output = cp.zeros((num_seeds, samples), dtype=cp.uint32)
    gpu_seeds = cp.array(seeds_list, dtype=cp.uint32)

    block_size = 256
    grid_x = min((samples + block_size - 1) // block_size, 2048)
    grid_y = num_seeds

    try:
        xorshift32_batch_kernel(
            (grid_x, grid_y), (block_size,),
            (gpu_output, gpu_seeds, cp.int32(num_seeds), cp.int32(samples))
        )
        cp.cuda.Device().synchronize()
        return gpu_output
    except Exception as e:
        print(f"XORshift32 batch GPU kernel failed: {e}")
        raise


def compute_correlations_batch_gpu(samples_batch, lmax: int, grid_size: int):
    """Compute correlations for ALL seeds in parallel on GPU"""
    num_seeds, n = samples_batch.shape

    float_samples = samples_batch.astype(cp.float32)

    means = cp.mean(float_samples, axis=1, keepdims=True)
    stds = cp.std(float_samples, axis=1, keepdims=True)
    normalized_batch = (float_samples - means) / stds

    max_lag = min(lmax, n // 4)
    all_autocorrs = []

    if max_lag > 0:
        for lag in range(1, max_lag + 1):
            if lag < n:
                length = n - lag
                arr_pairs = cp.empty((num_seeds * 2, length), dtype=cp.float32)
                arr_pairs[0::2] = normalized_batch[:, :length]
                arr_pairs[1::2] = normalized_batch[:, lag:lag+length]

                corr_matrix = gpu_corr_matrix_rows(arr_pairs)

                correlations = cp.array([corr_matrix[i*2, i*2+1] for i in range(num_seeds)])
                correlations = cp.where(cp.isnan(correlations), 0.0, correlations)
                all_autocorrs.append(correlations)

    if all_autocorrs:
        autocorrs_batch = cp.stack(all_autocorrs, axis=1)
    else:
        autocorrs_batch = cp.zeros((num_seeds, 0), dtype=cp.float32)

    grid_samples_batch = samples_batch[:, :grid_size**2] if n >= grid_size**2 else samples_batch

    correlation_matrices = []
    for seed_idx in range(num_seeds):
        grid_samples = grid_samples_batch[seed_idx]
        reshaped = grid_samples.reshape(-1, 1)

        try:
            correlation_matrix = gpu_corr_matrix_rows(reshaped.T)
            if correlation_matrix.size == 1:
                correlation_matrix = cp.array([[1.0]], dtype=cp.float32)
        except Exception:
            correlation_matrix = cp.array([[1.0]], dtype=cp.float32)

        correlation_matrices.append(correlation_matrix)

    results = []
    for seed_idx in range(num_seeds):
        corr_matrix = correlation_matrices[seed_idx]
        autocorrs = autocorrs_batch[seed_idx].tolist()

        mask = ~cp.eye(corr_matrix.shape[0], dtype=bool)
        off_diag = corr_matrix[mask]

        results.append({
            "correlation_matrix": corr_matrix,
            "max_correlation": float(cp.max(cp.abs(off_diag))) if off_diag.size > 0 else 0.0,
            "mean_correlation": float(cp.mean(cp.abs(off_diag))) if off_diag.size > 0 else 0.0,
            "autocorrelations": autocorrs
        })

    return results


def analyze_correlation_gpu(seeds: List[int], prng_type: str, samples: int,
                          lmax: int = 20, grid_size: int = 8, gpu_id: int = 0) -> Dict[str, Any]:
    """GPU-accelerated correlation analysis - VECTORIZED to process all seeds in parallel"""

    try:
        with cp.cuda.Device(gpu_id):
            results = {
                "success": True,
                "analysis_type": "gpu_correlation_vectorized",
                "gpu_id": gpu_id,
                "samples": samples,
                "seeds_analyzed": len(seeds),
                "prng_type": prng_type,
                "lmax": lmax,
                "grid_size": grid_size,
                "correlations": [],
                "device_info": {
                    "name": cp.cuda.runtime.getDeviceProperties(gpu_id)["name"].decode(),
                    "memory": cp.cuda.runtime.getDeviceProperties(gpu_id)["totalGlobalMem"],
                    "compute_capability": cp.cuda.runtime.getDeviceProperties(gpu_id)["major"]
                }
            }

            start_time = time.time()

            initial_states = [get_prng_state(prng_type, seed) for seed in seeds]

            if prng_type == "xorshift":
                gpu_samples_batch = generate_xorshift32_batch_gpu(initial_states, samples, gpu_id)
            else:
                cpu_samples_list = [generate_samples_cpu(state, prng_type, samples)
                                   for state in initial_states]
                gpu_samples_batch = cp.array(cpu_samples_list, dtype=cp.uint32)

            correlation_results = compute_correlations_batch_gpu(
                gpu_samples_batch, lmax, grid_size
            )

            for seed, initial_state, corr_data in zip(seeds, initial_states, correlation_results):
                results["correlations"].append({
                    "seed": seed,
                    "initial_state": initial_state,
                    "correlation_matrix": corr_data["correlation_matrix"].tolist(),
                    "max_correlation": corr_data["max_correlation"],
                    "mean_correlation": corr_data["mean_correlation"],
                    "autocorrelations": corr_data["autocorrelations"]
                })

            results["execution_time"] = time.time() - start_time
            results["samples_per_second"] = (len(seeds) * samples) / results["execution_time"]

            return results

    except Exception as e:
        print(f"GPU analysis failed on device {gpu_id}: {e}")
        raise e


def generate_samples_cpu(initial_state: int, prng_type: str, samples: int) -> List[int]:
    """CPU-based sample generation for fallback"""

    if prng_type == "xorshift":
        state = initial_state
        results = []
        for _ in range(samples):
            state ^= state << 13
            state ^= state >> 17
            state ^= state << 5
            state &= 0xFFFFFFFF
            results.append(state)
        return results

    elif prng_type == "lcg":
        state = initial_state
        results = []
        for _ in range(samples):
            state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
            results.append(state)
        return results

    elif prng_type == "mt":
        rng = np.random.Generator(np.random.MT19937(initial_state))
        return rng.integers(0, 2**32, samples, dtype=np.uint32).tolist()

    else:
        raise ValueError(f"Unknown PRNG type: {prng_type}")


# ============================================================================
# PUBLIC API FUNCTIONS - Entry points for distributed_worker.py
# ============================================================================

def run_statistical_analysis(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point for draw matching statistical analysis (original system)
    Called by distributed_worker.py for draw matching jobs
    """
    t0 = time.time()

    prng_type = (job_data.get("prng_type") or "xorshift").lower()
    mapping_type = (job_data.get("mapping_type") or "mod").lower()
    seeds = job_data.get("seeds") or []
    samples = int(job_data.get("samples", 10_000))
    rng = int(job_data.get("target_range", job_data.get("range", 1000)))

    if not seeds:
        seeds = [job_data.get("seed", 12345)]

    results_detail: List[Dict[str, Any]] = []
    comp_acc = []

    for s in seeds[: min(4, len(seeds))]:
        st = time.time()
        seq = _build_sequence(prng_type, mapping_type, samples, int(s), rng, job_data)
        det = _analyze_sequence(seq)
        det["analysis_time"] = float(time.time() - st)
        results_detail.append(det)
        comp_acc.append(0.25 * det["chi2_p_value"]
                        + 0.25 * det["runs_test"]["p_value"]
                        + 0.25 * abs(det["lag_correlations"].get("lag_1", 0.0))
                        + 0.25 * abs(det["lag_correlations"].get("lag_5", 0.0)))

    composite_score = float(np.mean(comp_acc)) if comp_acc else 0.0
    dv = _dev_info()
    is_nvidia = "NVIDIA" in (dv.get("name") or "").upper()

    out: Dict[str, Any] = {
        "job_id": job_data.get("job_id", "unknown"),
        "prng_type": prng_type,
        "mapping_type": mapping_type,
        "n_seeds": len(seeds),
        "success": True,
        "runtime": float(time.time() - t0),
        "mining_mode": bool(job_data.get("mining_mode", False)),
        "gpu_id": job_data.get("gpu_id"),
        "composite_score": composite_score,
        "z_scores": _enhanced_z_score_calculation(results_detail)["z_scores_array"] if results_detail else [0.0, 0.0, 0.0, 0.0],
        "z_scores_enhanced": _enhanced_z_score_calculation(results_detail) if results_detail else {},
        "detailed_properties": results_detail,
        "node_info": {
            "hostname": os.uname().nodename,
            "gpus_available": "detected",
            "execution_mode": "enhanced_standard",
            "hardware_type": "NVIDIA" if is_nvidia else "AMD"
        },
        "execution_info": {
            "worker_hostname": os.uname().nodename,
            "gpu_id": job_data.get("gpu_id"),
            "hardware_type": "NVIDIA" if is_nvidia else "AMD",
            "mining_mode": bool(job_data.get("mining_mode", False))
        }
    }
    return out


def run_advanced_draw_matching(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point for advanced draw matching (original system)
    Called by distributed_worker.py for draw matching jobs
    """
    t0 = time.time()

    prng_type = (job_data.get("prng_type") or "xorshift").lower()
    mapping_type = (job_data.get("mapping_type") or "mod").lower()
    seed = int(job_data.get("seed", 12345))
    samples = int(job_data.get("samples", 10_000))
    rng = int(job_data.get("target_range", job_data.get("range", 1000)))

    seq = _build_sequence(prng_type, mapping_type, samples, seed, rng, job_data)

    target = job_data.get("target_sequence") or job_data.get("target_draw") or []
    match_rate = None
    if isinstance(target, (list, tuple)) and len(target) > 0:
        tgt = cp.asarray(np.asarray(target, dtype=np.int32))
        m = min(tgt.size, seq.size)
        if m > 0:
            eq = cp.mean((seq[:m] == tgt[:m]).astype(cp.float32))
            match_rate = float(eq.get())

    res = {
        "job_id": job_data.get("job_id", "unknown"),
        "success": True,
        "method": "advanced_draw_matching",
        "match_rate": match_rate,
        "elapsed": float(time.time() - t0),
        "engine": "cupy_rawkernel",
        "gpu": _dev_info(),
    }
    return res


def run_correlation_analysis(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point for vectorized correlation analysis (new high-performance system)
    Called by distributed_worker.py for correlation analysis jobs
    
    This is the NEW VECTORIZED function that processes all seeds in parallel.
    Use this for high-performance correlation analysis.
    """
    seeds = job_data.get('seeds', [])
    prng_type = job_data.get('prng_type', 'mt')
    samples = job_data.get('samples', 10000)
    lmax = job_data.get('lmax', 20)
    grid_size = job_data.get('grid_size', 8)
    gpu_id = job_data.get('gpu_id', 0)

    return analyze_correlation_gpu(seeds, prng_type, samples, lmax, grid_size, gpu_id)
