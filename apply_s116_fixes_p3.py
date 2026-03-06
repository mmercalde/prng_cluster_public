#!/usr/bin/env python3
"""
S116 Fix Script Part 3 - Full call chain fix
Trace: run_bayesian_optimization() -> coordinator.optimize_window() -> optimizer.optimize() -> strategy.search()

Fix 1: optimize_window() in window_optimizer_integration_final.py - add resume_study/study_name to signature
Fix 2: optimizer.optimize() call in window_optimizer_integration_final.py - pass resume_study/study_name
Fix 3: optimize() method in window_optimizer.py - add resume_study/study_name to signature
Fix 4: coordinator.optimize_window() call in window_optimizer.py - pass resume_study/study_name
"""
import sys

# ============================================================================
# FIX 1+2: window_optimizer_integration_final.py
# ============================================================================
print("Fix 1+2: window_optimizer_integration_final.py")

with open('window_optimizer_integration_final.py', 'r') as f:
    content = f.read()

# Fix 1: Add resume_study/study_name to optimize_window() signature
old_sig = """    def optimize_window(self,
                        dataset_path: str,
                        seed_start: int = 0,
                        seed_count: int = 10_000_000,
                        prng_base: str = 'java_lcg',
                        test_both_modes: bool = False,
                        strategy_name: str = 'bayesian',
                        max_iterations: int = 50,
                        output_file: str = 'window_optimization.json'):"""

new_sig = """    def optimize_window(self,
                        dataset_path: str,
                        seed_start: int = 0,
                        seed_count: int = 10_000_000,
                        prng_base: str = 'java_lcg',
                        test_both_modes: bool = False,
                        strategy_name: str = 'bayesian',
                        max_iterations: int = 50,
                        output_file: str = 'window_optimization.json',
                        resume_study: bool = False,
                        study_name: str = ''):"""

if old_sig in content:
    content = content.replace(old_sig, new_sig)
    print("  ✅ 1: Added resume_study/study_name to optimize_window() signature")
elif "study_name: str = ''):" in content:
    print("  ✅ 1: Already patched - skipping")
else:
    print("  ❌ 1: optimize_window() signature not found"); sys.exit(1)

# Fix 2: Pass resume_study/study_name to optimizer.optimize() call
old_call = """        results = optimizer.optimize(
            strategy=strategy,
            bounds=bounds,
            max_iterations=max_iterations,
            scorer=BidirectionalCountScorer(),
            seed_start=seed_start,
            seed_count=seed_count
        )"""

new_call = """        results = optimizer.optimize(
            strategy=strategy,
            bounds=bounds,
            max_iterations=max_iterations,
            scorer=BidirectionalCountScorer(),
            seed_start=seed_start,
            seed_count=seed_count,
            resume_study=resume_study,
            study_name=study_name
        )"""

if old_call in content:
    content = content.replace(old_call, new_call)
    print("  ✅ 2: Passed resume_study/study_name to optimizer.optimize()")
elif "resume_study=resume_study" in content:
    print("  ✅ 2: Already patched - skipping")
else:
    print("  ❌ 2: optimizer.optimize() call block not found"); sys.exit(1)

with open('window_optimizer_integration_final.py', 'w') as f:
    f.write(content)
print("✅ Fix 1+2 complete\n")

# ============================================================================
# FIX 3+4: window_optimizer.py
# ============================================================================
print("Fix 3+4: window_optimizer.py")

with open('window_optimizer.py', 'r') as f:
    content = f.read()

# Fix 3: Add resume_study/study_name to optimize() method signature
old_opt = """    def optimize(self, strategy: SearchStrategy, bounds: SearchBounds,
                max_iterations: int = 50, scorer: ScoringFunction = None,
                seed_start: int = 0, seed_count: int = 10_000_000) -> Dict[str, Any]:"""

new_opt = """    def optimize(self, strategy: SearchStrategy, bounds: SearchBounds,
                max_iterations: int = 50, scorer: ScoringFunction = None,
                seed_start: int = 0, seed_count: int = 10_000_000,
                resume_study: bool = False, study_name: str = '') -> Dict[str, Any]:"""

if old_opt in content:
    content = content.replace(old_opt, new_opt)
    print("  ✅ 3: Added resume_study/study_name to optimize() signature")
elif "resume_study: bool = False, study_name: str = '') -> Dict[str, Any]:" in content:
    print("  ✅ 3: Already patched - skipping")
else:
    print("  ❌ 3: optimize() signature not found"); sys.exit(1)

# Fix 4: Pass resume_study/study_name to coordinator.optimize_window() call
old_coord = """    results = coordinator.optimize_window(
        dataset_path=lottery_file,
        seed_start=0,
        seed_count=seed_count,
        prng_base=prng_type,
        test_both_modes=test_both_modes,  # NEW: Pass through to integration layer
        strategy_name=strategy_name,
        max_iterations=trials,
        output_file='window_optimization_results.json'
    )"""

new_coord = """    results = coordinator.optimize_window(
        dataset_path=lottery_file,
        seed_start=0,
        seed_count=seed_count,
        prng_base=prng_type,
        test_both_modes=test_both_modes,  # NEW: Pass through to integration layer
        strategy_name=strategy_name,
        max_iterations=trials,
        output_file='window_optimization_results.json',
        resume_study=resume_study,
        study_name=study_name
    )"""

if old_coord in content:
    content = content.replace(old_coord, new_coord)
    print("  ✅ 4: Passed resume_study/study_name to coordinator.optimize_window()")
elif "resume_study=resume_study,\n        study_name=study_name\n    )" in content:
    print("  ✅ 4: Already patched - skipping")
else:
    print("  ❌ 4: coordinator.optimize_window() call block not found"); sys.exit(1)

with open('window_optimizer.py', 'w') as f:
    f.write(content)
print("✅ Fix 3+4 complete\n")

print("=" * 60)
print("ALL FIXES COMPLETE")
print("\nVerify syntax:")
print("  python3 -c \"import py_compile; [py_compile.compile(f, doraise=True) for f in ['window_optimizer.py','window_optimizer_integration_final.py']]; print('Syntax OK')\"")
