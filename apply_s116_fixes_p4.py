#!/usr/bin/env python3
"""
S116 Fix Script Part 4
Fix: BayesianOptimization.search() wrapper in window_optimizer.py
     - Add resume_study/study_name to signature
     - Forward to self.optuna_search.search()
"""
import sys

print("Fix: BayesianOptimization.search() wrapper in window_optimizer.py")

with open('window_optimizer.py', 'r') as f:
    content = f.read()

old_search = """    def search(self, objective_function, bounds, max_iterations, scorer):
        \"\"\"Run Bayesian optimization\"\"\"
        if self.optuna_search:
            # Use real Optuna implementation
            return self.optuna_search.search(objective_function, bounds, max_iterations, scorer)
        else:
            # Fallback to random search
            print("⚠️  Optuna not available, using random search fallback")
            return RandomSearch().search(objective_function, bounds, max_iterations, scorer)"""

new_search = """    def search(self, objective_function, bounds, max_iterations, scorer,
               resume_study: bool = False, study_name: str = ''):
        \"\"\"Run Bayesian optimization\"\"\"
        if self.optuna_search:
            # Use real Optuna implementation
            return self.optuna_search.search(objective_function, bounds, max_iterations, scorer,
                                             resume_study=resume_study, study_name=study_name)
        else:
            # Fallback to random search
            print("⚠️  Optuna not available, using random search fallback")
            return RandomSearch().search(objective_function, bounds, max_iterations, scorer)"""

if old_search in content:
    content = content.replace(old_search, new_search)
    print("  ✅ Added resume_study/study_name to BayesianOptimization.search()")
else:
    print("  ❌ Pattern not found"); sys.exit(1)

with open('window_optimizer.py', 'w') as f:
    f.write(content)

print("✅ Fix complete")
print("\nVerify:")
print("  python3 -c \"import py_compile; py_compile.compile('window_optimizer.py', doraise=True); print('Syntax OK')\"")
