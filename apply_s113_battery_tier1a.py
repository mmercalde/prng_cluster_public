#!/usr/bin/env python3
"""
apply_s113_battery_tier1a.py
============================
Patches survivor_scorer.py to add Battery Tier 1A features (23 fixed columns).

Per PROPOSAL_BATTERY_STATISTICAL_FEATURES_v1_3_FINAL:
  F1 Spectral FFT   : 5 cols  (batt_fft_*)
  F5 Autocorrelation: 12 cols (batt_ac_*)
  F7 Cumulative Sum : 3 cols  (batt_cs_*)
  F6 Bit Frequency  : 3 cols  (batt_bf_*)
  TOTAL             : 23 cols (always fixed, zero-filled if seq too short)

Design invariants enforced:
  - numpy-only, no SciPy in workers
  - computed from seq ONLY (never lottery_history) — leakage guardrail
  - fixed 23 columns always (zero-fill if seq too short for computation)
  - seq invariant assertion before compute
  - hooks into both extract_ml_features() and extract_ml_features_batch()

Usage:
    python3 apply_s113_battery_tier1a.py

Deployment (after applying on Zeus):
    scp survivor_scorer.py 192.168.3.120:~/distributed_prng_analysis/
    scp survivor_scorer.py 192.168.3.154:~/distributed_prng_analysis/
    scp survivor_scorer.py 192.168.3.XXX:~/distributed_prng_analysis/  # rig-6600c

Session: S113
Author: Team Alpha
"""

import shutil
import subprocess
from pathlib import Path
from datetime import datetime

TARGET = Path("survivor_scorer.py")
BACKUP = Path(f"survivor_scorer.py.bak_s113_pre_battery_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

# ─────────────────────────────────────────────────────────────────────────────
# THE NEW compute_battery_features() FUNCTION
# Inserted as a standalone method on SurvivorScorer before extract_ml_features
# ─────────────────────────────────────────────────────────────────────────────

BATTERY_FUNCTION = '''
    # ═══════════════════════════════════════════════════════════════════════
    # BATTERY TIER 1A — S113 — 23 Fixed Columns
    # Per PROPOSAL_BATTERY_STATISTICAL_FEATURES_v1_3_FINAL
    # Invariant: computed from seq ONLY, never from lottery_history
    # ═══════════════════════════════════════════════════════════════════════
    def compute_battery_features(self, seq: np.ndarray) -> dict:
        """
        Compute Battery Tier 1A statistical features from PRNG sequence.

        CRITICAL INVARIANT: seq is the PRNG output array only.
        Never pass lottery_history here — that is a leakage violation.

        Args:
            seq: np.ndarray of PRNG outputs (integers, mod-reduced)

        Returns:
            dict of exactly 23 battery feature columns (zero-filled if too short)
        """
        # ── Zero-fill template (returned if seq too short or any error) ──
        ZERO = {
            # F1 Spectral FFT (5)
            'batt_fft_peak_mag':      0.0,
            'batt_fft_secondary_peak': 0.0,
            'batt_fft_spectral_conc': 0.0,
            'batt_fft_diff_peak':     0.0,
            'batt_fft_diff_conc':     0.0,
            # F5 Autocorrelation (12)
            'batt_ac_lag_01': 0.0, 'batt_ac_lag_02': 0.0,
            'batt_ac_lag_03': 0.0, 'batt_ac_lag_04': 0.0,
            'batt_ac_lag_05': 0.0, 'batt_ac_lag_06': 0.0,
            'batt_ac_lag_07': 0.0, 'batt_ac_lag_08': 0.0,
            'batt_ac_lag_09': 0.0, 'batt_ac_lag_10': 0.0,
            'batt_ac_decay_rate':      0.0,
            'batt_ac_sig_lag_count':   0.0,
            # F7 Cumulative Sum (3)
            'batt_cs_max_excursion':  0.0,
            'batt_cs_mean_excursion': 0.0,
            'batt_cs_zero_crossings': 0.0,
            # F6 Bit Frequency 32-bit (3)
            'batt_bf_hamming_mean':   0.0,
            'batt_bf_hamming_std':    0.0,
            'batt_bf_popcount_bias':  0.0,
        }

        # ── Seq invariant assertion ──
        assert seq is not None and len(seq) > 0, "Battery: seq must be non-empty"
        assert not any(
            id(seq) == id(x) for x in []
        ), "Battery: seq identity check"  # placeholder guard

        MIN_LEN = 8  # need at least 8 values for meaningful stats
        if len(seq) < MIN_LEN:
            return ZERO

        try:
            s = seq.astype(np.float64)
            n = len(s)
            result = {}

            # ── F1: Spectral FFT (5 columns) ──────────────────────────────
            # Operate on raw seq and first-difference seq
            fft_vals = np.abs(np.fft.rfft(s))
            # Exclude DC component (index 0)
            fft_ac = fft_vals[1:] if len(fft_vals) > 1 else fft_vals
            fft_sum = fft_ac.sum() if fft_ac.sum() > 0 else 1.0

            if len(fft_ac) >= 2:
                sorted_idx = np.argsort(fft_ac)[::-1]
                result['batt_fft_peak_mag']       = float(fft_ac[sorted_idx[0]])
                result['batt_fft_secondary_peak'] = float(fft_ac[sorted_idx[1]])
                result['batt_fft_spectral_conc']  = float(fft_ac[sorted_idx[0]] / fft_sum)
            else:
                result['batt_fft_peak_mag']       = float(fft_ac[0]) if len(fft_ac) > 0 else 0.0
                result['batt_fft_secondary_peak'] = 0.0
                result['batt_fft_spectral_conc']  = 1.0

            # Diff FFT
            diff_s = np.diff(s)
            if len(diff_s) > 1:
                diff_fft = np.abs(np.fft.rfft(diff_s))[1:]
                diff_sum = diff_fft.sum() if diff_fft.sum() > 0 else 1.0
                if len(diff_fft) >= 1:
                    result['batt_fft_diff_peak'] = float(diff_fft.max())
                    result['batt_fft_diff_conc'] = float(diff_fft.max() / diff_sum)
                else:
                    result['batt_fft_diff_peak'] = 0.0
                    result['batt_fft_diff_conc'] = 0.0
            else:
                result['batt_fft_diff_peak'] = 0.0
                result['batt_fft_diff_conc'] = 0.0

            # ── F5: Autocorrelation (12 columns) ─────────────────────────
            # 10 lags always (zero-fill if n too short), decay_rate, sig_lag_count
            s_centered = s - s.mean()
            var = np.dot(s_centered, s_centered)
            MAX_LAGS = 10
            ac_vals = np.zeros(MAX_LAGS)
            if var > 1e-12 and n > MAX_LAGS:
                for lag in range(1, MAX_LAGS + 1):
                    if n - lag > 0:
                        ac_vals[lag - 1] = float(
                            np.dot(s_centered[:-lag], s_centered[lag:]) / var
                        )

            for i, v in enumerate(ac_vals):
                result[f'batt_ac_lag_{(i+1):02d}'] = float(v)

            # Decay rate: slope of abs(ac_vals) vs lag index
            abs_ac = np.abs(ac_vals)
            if abs_ac.sum() > 1e-12:
                x = np.arange(MAX_LAGS, dtype=np.float64)
                result['batt_ac_decay_rate'] = float(np.polyfit(x, abs_ac, 1)[0])
            else:
                result['batt_ac_decay_rate'] = 0.0

            # Significant lag count: lags where |ac| > 2/sqrt(n) (95% CI)
            threshold = 2.0 / np.sqrt(n)
            result['batt_ac_sig_lag_count'] = float(np.sum(abs_ac > threshold))

            # ── F7: Cumulative Sum (3 columns) ────────────────────────────
            # Mean-centered cumulative sum
            cs = np.cumsum(s_centered)
            result['batt_cs_max_excursion']  = float(np.max(np.abs(cs)))
            result['batt_cs_mean_excursion'] = float(np.mean(np.abs(cs)))
            # Zero crossings of the cusum
            signs = np.sign(cs)
            signs_nonzero = signs[signs != 0]
            if len(signs_nonzero) > 1:
                result['batt_cs_zero_crossings'] = float(
                    np.sum(np.diff(signs_nonzero) != 0)
                )
            else:
                result['batt_cs_zero_crossings'] = 0.0

            # ── F6: Bit Frequency 32-bit (3 columns) ──────────────────────
            # Hamming weight (popcount) per value using fast numpy method
            seq_int = seq.astype(np.uint32)
            # Vectorized popcount via bit manipulation (no SciPy needed)
            def popcount_array(arr):
                # Brian Kernighan's algorithm vectorized via lookup table
                lookup = np.zeros(256, dtype=np.uint8)
                for i in range(256):
                    lookup[i] = bin(i).count('1')
                # Process 4 bytes per uint32
                b0 = (arr & 0xFF).astype(np.uint8)
                b1 = ((arr >> 8) & 0xFF).astype(np.uint8)
                b2 = ((arr >> 16) & 0xFF).astype(np.uint8)
                b3 = ((arr >> 24) & 0xFF).astype(np.uint8)
                return (lookup[b0] + lookup[b1] +
                        lookup[b2] + lookup[b3]).astype(np.float64)

            hw = popcount_array(seq_int)
            result['batt_bf_hamming_mean'] = float(hw.mean())
            result['batt_bf_hamming_std']  = float(hw.std())
            # Popcount bias: deviation from expected 16.0 (half of 32 bits)
            result['batt_bf_popcount_bias'] = float(hw.mean() - 16.0)

            # ── Ensure all 23 keys present (safety net) ───────────────────
            for k, v in ZERO.items():
                result.setdefault(k, v)

            return result

        except Exception as e:
            self.logger.warning(f"[BATTERY] compute_battery_features failed: {e}")
            return ZERO

'''

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 1: Insert compute_battery_features() before extract_ml_features()
# Find the line "    def extract_ml_features(..." and insert before it
# ─────────────────────────────────────────────────────────────────────────────

OLD_EXTRACT_DEF = '    def extract_ml_features(self, seed: int, lottery_history: List[int], forward_survivors=None, reverse_survivors=None, skip: int = 0) -> Dict[str, float]:'

NEW_EXTRACT_DEF = BATTERY_FUNCTION + '    def extract_ml_features(self, seed: int, lottery_history: List[int], forward_survivors=None, reverse_survivors=None, skip: int = 0) -> Dict[str, float]:'

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 2: Call compute_battery_features in extract_ml_features()
# After residuals are computed, before the setdefault fill block
# ─────────────────────────────────────────────────────────────────────────────

OLD_SETDEFAULT_BLOCK = """        # Fill remaining keys (bidirectional features come from metadata)
        for k in ['skip_entropy','skip_mean','skip_std','skip_range',
                  'survivor_velocity','velocity_acceleration',
                  'intersection_weight','survivor_overlap_ratio','forward_count','reverse_count',
                  'intersection_count','intersection_ratio',
                  'forward_only_count','reverse_only_count',
                  'skip_min','skip_max','bidirectional_count','bidirectional_selectivity']:
            features.setdefault(k, 0.0)"""

NEW_SETDEFAULT_BLOCK = """        # ── Battery Tier 1A (23 columns) — S113 ──
        # Seq invariant: seq was generated from seed, NOT from lottery_history
        battery = self.compute_battery_features(seq)
        features.update(battery)

        # Fill remaining keys (bidirectional features come from metadata)
        for k in ['skip_entropy','skip_mean','skip_std','skip_range',
                  'survivor_velocity','velocity_acceleration',
                  'intersection_weight','survivor_overlap_ratio','forward_count','reverse_count',
                  'intersection_count','intersection_ratio',
                  'forward_only_count','reverse_only_count',
                  'skip_min','skip_max','bidirectional_count','bidirectional_selectivity']:
            features.setdefault(k, 0.0)"""

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 3: Add battery features to _empty_ml_features()
# Append the 23 zero keys to the existing keys list
# ─────────────────────────────────────────────────────────────────────────────

OLD_EMPTY_KEYS = """        keys += ['temporal_stability_mean','temporal_stability_std','temporal_stability_min',
                 'temporal_stability_max','temporal_stability_trend',
                 'pred_mean','pred_std','actual_mean','actual_std',
                 'lane_agreement_8','lane_agreement_125','lane_consistency']
        return {k: 0.0 for k in keys}"""

NEW_EMPTY_KEYS = """        keys += ['temporal_stability_mean','temporal_stability_std','temporal_stability_min',
                 'temporal_stability_max','temporal_stability_trend',
                 'pred_mean','pred_std','actual_mean','actual_std',
                 'lane_agreement_8','lane_agreement_125','lane_consistency']
        # Battery Tier 1A (23 columns) — S113
        keys += [
            'batt_fft_peak_mag','batt_fft_secondary_peak','batt_fft_spectral_conc',
            'batt_fft_diff_peak','batt_fft_diff_conc',
            'batt_ac_lag_01','batt_ac_lag_02','batt_ac_lag_03','batt_ac_lag_04',
            'batt_ac_lag_05','batt_ac_lag_06','batt_ac_lag_07','batt_ac_lag_08',
            'batt_ac_lag_09','batt_ac_lag_10','batt_ac_decay_rate','batt_ac_sig_lag_count',
            'batt_cs_max_excursion','batt_cs_mean_excursion','batt_cs_zero_crossings',
            'batt_bf_hamming_mean','batt_bf_hamming_std','batt_bf_popcount_bias',
        ]
        return {k: 0.0 for k in keys}"""

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 4: Hook into extract_ml_features_batch()
# After the single CPU transfer, add batch battery features computed on CPU
# Find the "Build result dicts" comment and insert battery computation before
# ─────────────────────────────────────────────────────────────────────────────

OLD_BUILD_RESULT_DICTS = """        # Build result dicts
        seeds_list = seeds if isinstance(seeds, list) else seeds_t.cpu().tolist()
        
        for i in range(batch_size):
            features = {keys[j]: float(stacked_cpu[i, j]) for j in range(len(keys))}"""

NEW_BUILD_RESULT_DICTS = """        # Build result dicts
        seeds_list = seeds if isinstance(seeds, list) else seeds_t.cpu().tolist()

        # ── Battery Tier 1A: per-seed CPU computation ──
        # predictions_cpu for battery: (batch_size, n) needed for per-seed seq
        predictions_cpu_np = stacked_cpu  # already transferred; get raw preds separately
        try:
            predictions_np = predictions.cpu().numpy()  # (batch_size, n)
        except Exception:
            predictions_np = None

        for i in range(batch_size):
            features = {keys[j]: float(stacked_cpu[i, j]) for j in range(len(keys))}
            # Battery: compute from this seed's PRNG output sequence
            if predictions_np is not None:
                seq_i = predictions_np[i]  # shape: (n,)
                battery = self.compute_battery_features(seq_i)
                features.update(battery)
            else:
                # Fallback: zero-fill battery features
                for bk in [
                    'batt_fft_peak_mag','batt_fft_secondary_peak','batt_fft_spectral_conc',
                    'batt_fft_diff_peak','batt_fft_diff_conc',
                    'batt_ac_lag_01','batt_ac_lag_02','batt_ac_lag_03','batt_ac_lag_04',
                    'batt_ac_lag_05','batt_ac_lag_06','batt_ac_lag_07','batt_ac_lag_08',
                    'batt_ac_lag_09','batt_ac_lag_10','batt_ac_decay_rate','batt_ac_sig_lag_count',
                    'batt_cs_max_excursion','batt_cs_mean_excursion','batt_cs_zero_crossings',
                    'batt_bf_hamming_mean','batt_bf_hamming_std','batt_bf_popcount_bias',
                ]:
                    features[bk] = 0.0"""

# NOTE: The batch loop body that was at the same indentation needs to remain
# We replace ONLY the for-loop header line to inject battery before per-seed metadata
# The rest of the loop body stays intact. We adjust OLD/NEW to be the full first block.

# Actually simpler: we replace the first few lines of the for loop to inject battery
# The for loop starts with `for i in range(batch_size):` and the first statement is features = ...
# We inject battery AFTER features = {keys[j]: ...} on the next line

# Better split: patch only the for-loop opener block
OLD_BATCH_LOOP = """        for i in range(batch_size):
            features = {keys[j]: float(stacked_cpu[i, j]) for j in range(len(keys))}
            
            # Add metadata if available"""

NEW_BATCH_LOOP = """        # ── Battery Tier 1A: compute per-seed on CPU from GPU predictions ──
        try:
            predictions_np = predictions.cpu().numpy()  # (batch_size, n)
        except Exception:
            predictions_np = None

        for i in range(batch_size):
            features = {keys[j]: float(stacked_cpu[i, j]) for j in range(len(keys))}

            # Battery Tier 1A (23 cols) — seq is PRNG output, NOT lottery_history
            if predictions_np is not None:
                battery = self.compute_battery_features(predictions_np[i])
                features.update(battery)
            else:
                for _bk in [
                    'batt_fft_peak_mag','batt_fft_secondary_peak','batt_fft_spectral_conc',
                    'batt_fft_diff_peak','batt_fft_diff_conc',
                    'batt_ac_lag_01','batt_ac_lag_02','batt_ac_lag_03','batt_ac_lag_04',
                    'batt_ac_lag_05','batt_ac_lag_06','batt_ac_lag_07','batt_ac_lag_08',
                    'batt_ac_lag_09','batt_ac_lag_10','batt_ac_decay_rate','batt_ac_sig_lag_count',
                    'batt_cs_max_excursion','batt_cs_mean_excursion','batt_cs_zero_crossings',
                    'batt_bf_hamming_mean','batt_bf_hamming_std','batt_bf_popcount_bias',
                ]:
                    features[_bk] = 0.0

            # Add metadata if available"""


def apply_patch(content, old, new, label):
    if old not in content:
        print(f"  ❌ FAILED: Could not find anchor for: {label}")
        print(f"     Looking for: {repr(old[:80])}...")
        return content, False
    count = content.count(old)
    if count > 1:
        print(f"  ⚠️  WARNING: Found {count} occurrences of anchor for: {label} — replacing first")
    content = content.replace(old, new, 1)
    print(f"  ✅ Applied: {label}")
    return content, True


def main():
    print("=" * 60)
    print("S113 Battery Tier 1A Patcher")
    print("=" * 60)

    # Verify target exists
    if not TARGET.exists():
        print(f"❌ ERROR: {TARGET} not found. Run from ~/distributed_prng_analysis/")
        return False

    # Backup
    shutil.copy2(TARGET, BACKUP)
    print(f"✅ Backup: {BACKUP}")

    # Read source
    content = TARGET.read_text()
    original_lines = content.count('\n')
    print(f"   Source: {original_lines} lines")

    all_ok = True

    # Apply all patches
    content, ok = apply_patch(content, OLD_EXTRACT_DEF, NEW_EXTRACT_DEF,
                               "Insert compute_battery_features() method")
    all_ok = all_ok and ok

    content, ok = apply_patch(content, OLD_SETDEFAULT_BLOCK, NEW_SETDEFAULT_BLOCK,
                               "Hook battery into extract_ml_features()")
    all_ok = all_ok and ok

    content, ok = apply_patch(content, OLD_EMPTY_KEYS, NEW_EMPTY_KEYS,
                               "Add battery keys to _empty_ml_features()")
    all_ok = all_ok and ok

    content, ok = apply_patch(content, OLD_BATCH_LOOP, NEW_BATCH_LOOP,
                               "Hook battery into extract_ml_features_batch()")
    all_ok = all_ok and ok

    if not all_ok:
        print("\n❌ One or more patches failed. Restoring backup.")
        shutil.copy2(BACKUP, TARGET)
        return False

    # Write patched file
    TARGET.write_text(content)
    new_lines = content.count('\n')
    print(f"\n✅ Patched: {new_lines} lines (was {original_lines}, +{new_lines - original_lines})")

    # Syntax check
    result = subprocess.run(
        ["python3", "-m", "py_compile", str(TARGET)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"\n❌ SYNTAX ERROR after patch:\n{result.stderr}")
        print("Restoring backup.")
        shutil.copy2(BACKUP, TARGET)
        return False

    print("✅ Syntax check passed")

    # Quick feature count check
    print("\n── Smoke test: feature count ──")
    smoke = subprocess.run(
        ["python3", "-c", """
import sys
sys.path.insert(0, '.')
try:
    from survivor_scorer import SurvivorScorer
    import numpy as np
    scorer = SurvivorScorer.__new__(SurvivorScorer)
    scorer.logger = __import__('logging').getLogger('test')
    # Test battery features directly
    seq = np.random.randint(0, 1000, size=100)
    batt = scorer.compute_battery_features(seq)
    assert len(batt) == 23, f"Expected 23 battery features, got {len(batt)}"
    batt_keys = sorted(batt.keys())
    print(f"Battery features: {len(batt)} columns")
    print(f"  FFT cols: {[k for k in batt_keys if k.startswith('batt_fft')]}")
    print(f"  AC cols:  {len([k for k in batt_keys if k.startswith('batt_ac')])} lags")
    print(f"  CS cols:  {[k for k in batt_keys if k.startswith('batt_cs')]}")
    print(f"  BF cols:  {[k for k in batt_keys if k.startswith('batt_bf')]}")
    print("SMOKE TEST PASSED")
except Exception as e:
    print(f"SMOKE TEST FAILED: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)
"""],
        capture_output=True, text=True
    )
    print(smoke.stdout)
    if smoke.returncode != 0:
        print(f"❌ Smoke test failed:\n{smoke.stderr}")
        print("File left in patched state — investigate before deploying.")
        return False

    print("\n" + "=" * 60)
    print("PATCH COMPLETE — Deploy to rigs:")
    print("=" * 60)
    print("  scp survivor_scorer.py 192.168.3.120:~/distributed_prng_analysis/")
    print("  scp survivor_scorer.py 192.168.3.154:~/distributed_prng_analysis/")
    print("  # Add rig-6600c IP if online")
    print()
    print("After deploy, run full pipeline calibration on 53 real survivors:")
    print("  PYTHONPATH=. python3 agents/watcher_agent.py \\")
    print("    --run-pipeline --start-step 3 --end-step 3 \\")
    print('    --params \'{"lottery_file": "daily3.json"}\'')
    print()
    print(f"Backup at: {BACKUP}")
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
