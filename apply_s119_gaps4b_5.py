#!/usr/bin/env python3
"""
S119 Pruning Fix: Gap 4b + Gap 5
==================================

Gap 4a: ALREADY PATCHED ON ZEUS (S118, not pushed to GitHub) — SKIP

Gap 4b (window_optimizer.py):
  test_configuration() accepts optuna_trial=None but doesn't forward it
  to test_configuration_func(). So even though objective() now passes
  optuna_trial correctly, it gets dropped here.

Gap 5 (window_optimizer_integration_final.py):
  test_config() accepts optuna_trial=None but never passes it to
  run_bidirectional_test(), so the pruning hook at line 176 never fires.
"""

import os
import shutil
from datetime import datetime

ZEUS_BASE = os.path.expanduser("~/distributed_prng_analysis")

WO_FILE  = os.path.join(ZEUS_BASE, "window_optimizer.py")
INT_FILE = os.path.join(ZEUS_BASE, "window_optimizer_integration_final.py")

TS = datetime.now().strftime("%Y%m%d_%H%M%S")

# ─── Gap 4b ────────────────────────────────────────────────────────────────────
# test_configuration() doesn't accept or forward optuna_trial

GAP4B_OLD = """\
    def test_configuration(self, config: WindowConfig, seed_start: int = 0,
                          seed_count: int = 10_000_000) -> TestResult:
        \"\"\"
        Test a configuration.
        This will be overridden by the integration layer to run real sieves.
        Thresholds are now taken from config.forward_threshold and config.reverse_threshold.
        \"\"\"
        if self.test_configuration_func:
            return self.test_configuration_func(config, seed_start, seed_count)"""

GAP4B_NEW = """\
    def test_configuration(self, config: WindowConfig, seed_start: int = 0,
                          seed_count: int = 10_000_000,
                          optuna_trial=None) -> TestResult:  # S119
        \"\"\"
        Test a configuration.
        This will be overridden by the integration layer to run real sieves.
        Thresholds are now taken from config.forward_threshold and config.reverse_threshold.
        \"\"\"
        if self.test_configuration_func:
            return self.test_configuration_func(config, seed_start, seed_count,
                                                optuna_trial=optuna_trial)  # S119"""

# ─── Gap 5 ─────────────────────────────────────────────────────────────────────
# test_config() calls run_bidirectional_test() without passing optuna_trial

GAP5_OLD = """\
            return run_bidirectional_test(
                coordinator=self,
                config=config,
                dataset_path=dataset_path,
                seed_start=ss,
                seed_count=sc,
                prng_base=prng_base,
                test_both_modes=test_both_modes,
                forward_threshold=ft,
                reverse_threshold=rt,
                trial_number=trial_counter['count'],
                accumulator=survivor_accumulator
            )"""

GAP5_NEW = """\
            return run_bidirectional_test(
                coordinator=self,
                config=config,
                dataset_path=dataset_path,
                seed_start=ss,
                seed_count=sc,
                prng_base=prng_base,
                test_both_modes=test_both_modes,
                forward_threshold=ft,
                reverse_threshold=rt,
                trial_number=trial_counter['count'],
                accumulator=survivor_accumulator,
                optuna_trial=optuna_trial          # S119 Gap5
            )"""


def apply_patch(filepath, old_text, new_text, label):
    with open(filepath, 'r') as f:
        content = f.read()

    count = content.count(old_text)
    if count == 0:
        print(f"  ❌ {label}: anchor NOT FOUND in {os.path.basename(filepath)}")
        return False
    if count > 1:
        print(f"  ⚠️  {label}: anchor appears {count} times — aborting (ambiguous)")
        return False

    bak = filepath + f".bak_s119_{TS}"
    shutil.copy2(filepath, bak)
    print(f"  📦 Backup: {os.path.basename(bak)}")

    with open(filepath, 'w') as f:
        f.write(content.replace(old_text, new_text))

    print(f"  ✅ {label}: applied")
    return True


def verify(filepath, text, label):
    with open(filepath) as f:
        found = text in f.read()
    print(f"  {'✅' if found else '❌'} VERIFY {label}")
    return found


def main():
    print("=" * 60)
    print("S119 Pruning Wire-Up: Gap 4b + Gap 5")
    print("(Gap 4a already patched on Zeus from S118)")
    print("=" * 60)

    all_ok = True

    print("\n[Gap 4b] test_configuration() signature + forwarding")
    ok = apply_patch(WO_FILE, GAP4B_OLD, GAP4B_NEW, "Gap4b")
    all_ok = all_ok and ok

    print("\n[Gap 5] run_bidirectional_test() missing optuna_trial")
    ok = apply_patch(INT_FILE, GAP5_OLD, GAP5_NEW, "Gap5")
    all_ok = all_ok and ok

    print("\n── Verification ──")
    verify(WO_FILE,  "optuna_trial=optuna_trial)  # S119", "Gap4b in window_optimizer.py")
    verify(INT_FILE, "optuna_trial=optuna_trial          # S119 Gap5", "Gap5 in integration_final.py")

    print("\n" + "=" * 60)
    if all_ok:
        print("✅ Both patches applied. Full pruning chain:")
        print()
        print("  objective(config, optuna_trial=trial)             [S118 ✅]")
        print("  → test_configuration(..., optuna_trial=trial)     [S119 Gap4b ✅]")
        print("  → test_config(..., optuna_trial=trial)")
        print("  → run_bidirectional_test(..., optuna_trial=trial) [S119 Gap5 ✅]")
        print("  → if forward==0: raise TrialPruned()              [S115 hook ✅]")
        print()
        print("Run verify test:")
        print("  python3 verify_pruning_s118.py --trials 6 --max-seeds 2000000")
    else:
        print("❌ Some patches failed — check output above")
    print("=" * 60)


if __name__ == '__main__':
    main()
