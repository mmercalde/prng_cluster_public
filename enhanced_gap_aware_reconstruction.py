#!/usr/bin/env python3
"""
Enhanced Gap-Aware PRNG Reconstruction with comprehensive algorithm support
Integrates with existing MT19937 reconstruction and adds many more algorithms
"""

import time
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

class EnhancedGapAwarePRNGReconstructor:
    """Enhanced gap-aware PRNG reconstruction with comprehensive algorithm support"""

    def __init__(self):
        self.supported_algorithms = [
            'lcg', 'xorshift', 'xorshift32', 'xorshift64', 'xorshift128',
            'xorshift_plus', 'xorshift_star', 'xoshiro128', 'xoshiro256',
            'mt19937', 'mt19937_64', 'lfsr', 'fibonacci', 'combined_lcg',
            'well512', 'pcg32', 'splitmix64', 'lehmer64'
        ]
        self.debug = True

    def reconstruct_with_gaps(self, prng_type: str, sparse_outputs: List[Tuple[int, int]],
                             user_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main reconstruction interface with enhanced algorithm support"""
        prng_type = prng_type.lower().strip()

        if prng_type not in self.supported_algorithms:
            return {
                "success": False,
                "error": f"Unsupported PRNG type: {prng_type}. Supported: {', '.join(self.supported_algorithms)}"
            }

        positions = [pos for pos, val in sparse_outputs]
        values = [val for pos, val in sparse_outputs]
        gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]

        result = {
            "algorithm": prng_type,
            "gap_analysis": {
                "has_gaps": any(gap > 1 for gap in gaps),
                "total_outputs": len(values),
                "largest_gap": max(gaps) if gaps else 0,
                "gap_pattern": gaps[:10]  # First 10 gaps for analysis
            }
        }

        try:
            # Route to appropriate reconstruction method
            if prng_type in ['lcg', 'combined_lcg']:
                return self._reconstruct_lcg_variants(values, gaps, result, prng_type)
            elif prng_type.startswith('xorshift') or prng_type.startswith('xoshiro'):
                return self._reconstruct_xorshift_variants(values, gaps, result, prng_type)
            elif prng_type.startswith('mt'):
                return self._reconstruct_mt_variants(values, gaps, result, prng_type, sparse_outputs)
            elif prng_type in ['lfsr']:
                return self._reconstruct_lfsr(values, gaps, result)
            elif prng_type in ['fibonacci']:
                return self._reconstruct_fibonacci(values, gaps, result)
            elif prng_type in ['well512']:
                return self._reconstruct_well(values, gaps, result)
            elif prng_type in ['pcg32']:
                return self._reconstruct_pcg(values, gaps, result)
            elif prng_type in ['splitmix64', 'lehmer64']:
                return self._reconstruct_64bit_generators(values, gaps, result, prng_type)
            else:
                result.update({"success": False, "error": f"Reconstruction method not implemented for {prng_type}"})
                return result

        except Exception as e:
            if self.debug:
                import traceback
                traceback.print_exc()
            result.update({"success": False, "error": f"Reconstruction failed: {str(e)}"})
            return result

    def _reconstruct_lcg_variants(self, values: List[int], gaps: List[int], result: Dict[str, Any], variant: str) -> Dict[str, Any]:
        """Enhanced LCG reconstruction supporting multiple variants"""

        # Standard LCG parameters to test
        lcg_variants = {
            'lcg': [
                (1103515245, 12345, 2**32),      # glibc
                (1664525, 1013904223, 2**32),    # Numerical Recipes
                (16807, 0, 2**31-1),             # Park and Miller
                (48271, 0, 2**31-1),             # Park and Miller revised
                (69069, 1, 2**32),               # Super-duper
                (1812433253, 0, 2**32),          # Knuth MMIX
                (134775813, 1, 2**32),           # Borland C++
                (214013, 2531011, 2**32),        # Microsoft Visual C++
                (1140671485, 128201163, 2**24),  # RANDU (historical)
            ],
            'combined_lcg': [
                # L'Ecuyer's combined generators
                (40014, 0, 2147483563, 40692, 0, 2147483399),  # MRG32k3a components
            ]
        }

        if variant == 'combined_lcg':
            return self._test_combined_lcg(values, gaps, result, lcg_variants[variant])
        else:
            return self._test_standard_lcg(values, gaps, result, lcg_variants[variant])

    def _test_standard_lcg(self, values: List[int], gaps: List[int], result: Dict[str, Any], params_list: List[Tuple[int, int, int]]) -> Dict[str, Any]:
        """Test standard LCG parameters with corrected gap handling"""

        # Try consecutive reconstruction first if no gaps
        if all(gap == 1 for gap in gaps) and len(values) >= 3:
            consecutive_result = self._lcg_consecutive_corrected(values, result.copy())
            if consecutive_result.get('success'):
                return consecutive_result

        # Test known parameters with corrected gap verification
        for a, c, m in params_list:
            if self._verify_lcg_corrected(a, c, m, values, gaps):
                if self.debug:
                    print(f"Found gap-aware LCG: a={a}, c={c}, m={m}")

                result.update({
                    "success": True,
                    "method": "gap_aware_known_params",
                    "parameters": {"a": a, "c": c, "m": m},
                    "verification": {"accuracy": 1.0},
                    "next_predictions": self._predict_lcg(a, c, m, values[-1], gaps[-1] if gaps else 1, 5)
                })
                return result

        result.update({"success": False, "error": "No LCG parameters found"})
        return result

    def _lcg_consecutive_corrected(self, values: List[int], result: Dict[str, Any]) -> Dict[str, Any]:
        """Corrected consecutive LCG reconstruction from corrected version"""
        if len(values) < 3:
            result.update({"success": False, "error": "Need at least 3 consecutive values"})
            return result

        x1, x2, x3 = values[0], values[1], values[2]

        # Try common moduli
        moduli = [2**32, 2**31-1, 2**31, 2**24, 2**16]

        for m in moduli:
            if max(values) >= m:
                continue

            # Solve for a and c using consecutive relation
            delta1 = (x2 - x1) % m
            delta2 = (x3 - x2) % m

            if delta1 == 0:
                continue

            # Calculate a = delta2 / delta1 (mod m)
            try:
                a = (delta2 * pow(delta1, -1, m)) % m
                c = (x2 - a * x1) % m

                # Verify all values
                if self._verify_lcg_all_values(a, c, m, values, [1] * (len(values) - 1)):
                    if self.debug:
                        print(f"Found consecutive LCG: a={a}, c={c}, m={m}")

                    result.update({
                        "success": True,
                        "prng_type": "LCG",
                        "method": "consecutive_corrected",
                        "parameters": {"a": a, "c": c, "m": m},
                        "verification": {"accuracy": 1.0},
                        "next_predictions": self._predict_lcg(a, c, m, values[-1], 1, 5)
                    })
                    return result
            except (ValueError, ZeroDivisionError):
                continue

        result.update({"success": False, "error": "No consecutive LCG parameters found"})
        return result

    def _verify_lcg_corrected(self, a: int, c: int, m: int, values: List[int], gaps: List[int]) -> bool:
        """Corrected LCG verification with robust gap handling"""
        for i, gap in enumerate(gaps):
            x1, x2 = values[i], values[i + 1]

            # Use iterative approach for gap handling
            expected = x1
            for _ in range(gap):
                expected = (a * expected + c) % m

            if expected != x2:
                if self.debug:
                    print(f"Gap {gap}: {x1} -> expected {expected}, got {x2}")
                return False

        return True

    def _verify_lcg_all_values(self, a: int, c: int, m: int, values: List[int], gaps: List[int]) -> bool:
        """Verify LCG parameters against all values"""
        return self._verify_lcg_corrected(a, c, m, values, gaps)

    def _predict_lcg(self, a: int, c: int, m: int, last_value: int, gap: int, count: int) -> List[int]:
        """Predict next LCG values"""
        predictions = []
        current = last_value

        for _ in range(count):
            for _ in range(gap):
                current = (a * current + c) % m
            predictions.append(current)

        return predictions

    def _test_combined_lcg(self, values: List[int], gaps: List[int], result: Dict[str, Any], params_list: List[Tuple]) -> Dict[str, Any]:
        """Test combined LCG generators (L'Ecuyer style)"""
        # Implementation for combined generators would go here
        result.update({"success": False, "error": "Combined LCG reconstruction not yet implemented"})
        return result

    def _reconstruct_xorshift_variants(self, values: List[int], gaps: List[int], result: Dict[str, Any], variant: str) -> Dict[str, Any]:
        """Enhanced Xorshift reconstruction with corrected sequence testing"""

        xorshift_params = {
            'xorshift': [(13, 17, 5), (5, 14, 1), (6, 21, 7), (23, 18, 5)],
            'xorshift32': [(13, 17, 5), (1, 3, 10), (3, 5, 17), (7, 9, 13)],
            'xorshift64': [(12, 25, 27), (21, 35, 4), (7, 9, 8)],
            'xorshift128': [(11, 8, 19), (8, 14, 9), (13, 7, 17)],
            'xorshift_plus': [(23, 17, 26), (17, 45, 11)],
            'xorshift_star': [(12, 25, 27), (25, 27, 12)],
            'xoshiro128': [(9, 5, 7, 9), (13, 5, 10, 9)],
            'xoshiro256': [(17, 45, 13, 11), (49, 21, 28, 33)]
        }

        if variant not in xorshift_params:
            result.update({"success": False, "error": f"Unknown xorshift variant: {variant}"})
            return result

        if self.debug:
            print(f"Xorshift reconstruction: values={values}")

        # Test different initial states with corrected sequence testing
        init_states = [1, 12345, 0x12345678]

        for params in xorshift_params[variant]:
            for init_state in init_states:
                if self._test_xorshift_sequence_corrected(values, gaps, params, init_state, variant):
                    if self.debug:
                        print(f"Found {variant}: params={params}, init={init_state}")

                    result.update({
                        "success": True,
                        "prng_type": "Xorshift",
                        "method": f"{variant}_parameter_search_corrected",
                        "parameters": {
                            "variant": variant,
                            "params": params,
                            "initial_state": init_state
                        },
                        "verification": {"accuracy": 1.0}
                    })
                    return result

        result.update({"success": False, "error": f"No {variant} parameters found"})
        return result

    def _test_xorshift_sequence_corrected(self, values: List[int], gaps: List[int], params: Tuple, init_state: int, variant: str) -> bool:
        """Corrected test if Xorshift parameters can generate the sequence"""

        state = init_state
        generated = []

        # Generate enough values
        for i in range(50):
            state = self._xorshift_step_corrected(state, params, variant)
            generated.append(state)

        # Look for sequence match
        for start_offset in range(len(generated) - len(values) + 1):
            matches = 0
            for i, val in enumerate(values):
                if start_offset + i < len(generated) and generated[start_offset + i] == val:
                    matches += 1

            if matches == len(values):
                return True

        return False

    def _xorshift_step_corrected(self, state: int, params: Tuple, variant: str) -> int:
        """Corrected single Xorshift step from corrected version"""
        if state == 0:
            state = 1

        if variant in ['xorshift', 'xorshift32']:
            a, b, c = params[:3]
            state ^= state << a
            state ^= state >> b
            state ^= state << c
            return state & 0xFFFFFFFF
        elif variant == 'xorshift64':
            a, b, c = params[:3]
            state ^= state << a
            state ^= state >> b
            state ^= state << c
            return state & 0xFFFFFFFFFFFFFFFF
        elif variant in ['xorshift_plus', 'xorshift_star']:
            return self._xorshift_enhanced_step(state, params, variant)
        elif variant.startswith('xoshiro'):
            return self._xoshiro_step(state, params, variant)
        else:
            # Default to standard xorshift
            a, b, c = params[:3]
            state ^= state << a
            state ^= state >> b
            state ^= state << c
            return state & 0xFFFFFFFF

    def _reconstruct_mt_variants(self, values: List[int], gaps: List[int], result: Dict[str, Any], variant: str, sparse_outputs: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Enhanced MT reconstruction using existing MT19937 infrastructure"""

        try:
            # Import your existing MT reconstruction modules
            import sys
            import os

            modules_path = os.path.join(os.path.dirname(__file__), 'modules')
            if os.path.exists(modules_path) and modules_path not in sys.path:
                sys.path.append(modules_path)

            from mt_engine_exact import AdvancedMT19937Reconstructor
            from analyze_my_lottery_data import convert_lottery_to_32bit

            reconstructor = AdvancedMT19937Reconstructor()

            # For gap-aware MT reconstruction, we need to handle sparse data differently
            if variant == 'mt19937':
                # Try multiple conversion methods as in your existing code
                conversion_methods = ['hash', 'polynomial', 'xor_rotate', 'multiply']
                best_result = {"success": False, "confidence": 0.0}

                for method in conversion_methods:
                    try:
                        # Convert sparse values using your existing conversion
                        converted_values = convert_lottery_to_32bit(values, method)

                        # Use your existing reconstruction
                        mt_result = reconstructor.reconstruct_mt_state(converted_values)

                        if mt_result.get('success') and mt_result.get('confidence', 0) > best_result.get('confidence', 0):
                            best_result = mt_result.copy()
                            best_result['conversion_method'] = method
                            best_result['gap_aware'] = True

                    except Exception as e:
                        continue

                if best_result.get('success'):
                    result.update({
                        "success": True,
                        "method": "mt19937_gap_aware",
                        "parameters": best_result,
                        "confidence": best_result.get('confidence', 0)
                    })
                else:
                    result.update({
                        "success": False,
                        "error": "MT19937 reconstruction failed with all conversion methods"
                    })

            elif variant == 'mt19937_64':
                # Implement 64-bit MT reconstruction
                result.update({
                    "success": False,
                    "error": "MT19937-64 reconstruction not yet implemented"
                })

            return result

        except ImportError as e:
            result.update({
                "success": False,
                "error": f"MT reconstruction modules not available: {e}"
            })
            return result

    def _reconstruct_lfsr(self, values: List[int], gaps: List[int], result: Dict[str, Any]) -> Dict[str, Any]:
        """Linear Feedback Shift Register reconstruction"""

        # Common LFSR polynomials (Galois configuration)
        lfsr_polynomials = [
            0x80000057,  # 32-bit primitive polynomial
            0x80000062,  # Another 32-bit primitive
            0x8000006B,  # 32-bit primitive
            0x80000074,  # 32-bit primitive
        ]

        for poly in lfsr_polynomials:
            for init_state in [1, 0xACE1, 0xDEADBEEF]:
                if self._test_lfsr(values, gaps, poly, init_state):
                    result.update({
                        "success": True,
                        "method": "lfsr_reconstruction",
                        "parameters": {
                            "polynomial": hex(poly),
                            "initial_state": hex(init_state)
                        }
                    })
                    return result

        result.update({"success": False, "error": "No LFSR parameters found"})
        return result

    def _reconstruct_fibonacci(self, values: List[int], gaps: List[int], result: Dict[str, Any]) -> Dict[str, Any]:
        """Lagged Fibonacci generator reconstruction"""

        # Common lag pairs for Fibonacci generators
        lag_pairs = [(24, 55), (38, 89), (37, 100), (30, 127)]

        for j, k in lag_pairs:
            if len(values) > k:  # Need enough values to test
                if self._test_fibonacci(values, gaps, j, k):
                    result.update({
                        "success": True,
                        "method": "fibonacci_reconstruction",
                        "parameters": {"j": j, "k": k}
                    })
                    return result

        result.update({"success": False, "error": "No Fibonacci generator parameters found"})
        return result

    def _reconstruct_well(self, values: List[int], gaps: List[int], result: Dict[str, Any]) -> Dict[str, Any]:
        """WELL (Well Equidistributed Long-period Linear) reconstruction"""
        result.update({"success": False, "error": "WELL reconstruction not yet implemented"})
        return result

    def _reconstruct_pcg(self, values: List[int], gaps: List[int], result: Dict[str, Any]) -> Dict[str, Any]:
        """PCG (Permuted Congruential Generator) reconstruction"""
        result.update({"success": False, "error": "PCG reconstruction not yet implemented"})
        return result

    def _reconstruct_64bit_generators(self, values: List[int], gaps: List[int], result: Dict[str, Any], variant: str) -> Dict[str, Any]:
        """64-bit generator reconstruction (SplitMix64, Lehmer64)"""
        result.update({"success": False, "error": f"{variant} reconstruction not yet implemented"})
        return result

    # Helper methods for specific algorithm implementations
    def _verify_lcg(self, a: int, c: int, m: int, values: List[int], gaps: List[int]) -> bool:
        """Legacy method - redirects to corrected version"""
        return self._verify_lcg_corrected(a, c, m, values, gaps)

    def _xorshift32_step(self, state: int, params: Tuple[int, int, int]) -> int:
        """Single step of 32-bit xorshift - redirects to corrected version"""
        return self._xorshift_step_corrected(state, params, 'xorshift32')

    def _xorshift64_step(self, state: int, params: Tuple[int, int, int]) -> int:
        """Single step of 64-bit xorshift"""
        a, b, c = params
        if state == 0:
            state = 1
        state ^= (state << a) & 0xFFFFFFFFFFFFFFFF
        state ^= state >> b
        state ^= (state << c) & 0xFFFFFFFFFFFFFFFF
        return state & 0xFFFFFFFFFFFFFFFF

    def _xorshift_enhanced_step(self, state: int, params: Tuple[int, int], variant: str) -> int:
        """Enhanced xorshift variants (+ and *)"""
        # Placeholder implementation
        return self._xorshift32_step(state, params + (5,))

    def _xoshiro_step(self, state: int, params: Tuple, variant: str) -> int:
        """Xoshiro family step function"""
        # Placeholder implementation
        return state

    def _test_lfsr(self, values: List[int], gaps: List[int], poly: int, init_state: int) -> bool:
        """Test LFSR parameters"""
        state = init_state
        generated = []

        for _ in range(len(values) + sum(gaps)):
            # Galois LFSR step
            if state & 1:
                state = (state >> 1) ^ poly
            else:
                state = state >> 1
            generated.append(state & 0xFFFFFFFF)

        return self._match_with_gaps(generated, values, gaps)

    def _test_fibonacci(self, values: List[int], gaps: List[int], j: int, k: int) -> bool:
        """Test Fibonacci generator parameters"""
        if len(values) < k:
            return False

        # Initialize with first k values
        state = values[:k]
        generated = state.copy()

        # Generate more values
        for i in range(len(values) + sum(gaps)):
            next_val = (state[-j] + state[-k]) % (2**32)
            state.append(next_val)
            generated.append(next_val)

        return self._match_with_gaps(generated[k:], values[k:], gaps[k-1:] if len(gaps) >= k else [])

    def _match_with_gaps(self, generated: List[int], target: List[int], gaps: List[int]) -> bool:
        """Match generated sequence against target considering gaps"""
        if len(gaps) == 0:
            return generated[:len(target)] == target

        # For gapped comparison
        pos = 0
        for i, target_val in enumerate(target[:-1]):
            if pos >= len(generated):
                return False
            if generated[pos] != target_val:
                return False
            if i < len(gaps):
                pos += gaps[i]
            else:
                pos += 1

        # Check final value
        if pos < len(generated) and len(target) > 0:
            return generated[pos] == target[-1]

        return True


# Main interface function - enhanced version
def create_enhanced_gap_aware_analysis(prng_type: str, sparse_outputs: List[Tuple[int, int]],
                                      user_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Enhanced gap-aware reconstruction with comprehensive algorithm support"""

    reconstructor = EnhancedGapAwarePRNGReconstructor()
    result = reconstructor.reconstruct_with_gaps(prng_type, sparse_outputs, user_config)

    result.update({
        "timestamp": time.time(),
        "analysis_type": "enhanced_gap_aware_reconstruction",
        "supported_algorithms": reconstructor.supported_algorithms
    })

    return result


# Backward compatibility wrapper
def create_gap_aware_reconstruction_analysis(prng_type: str, sparse_outputs: List[Tuple[int, int]],
                                           user_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Backward compatibility wrapper"""
    return create_enhanced_gap_aware_analysis(prng_type, sparse_outputs, user_config)
