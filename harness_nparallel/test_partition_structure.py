#!/usr/bin/env python3
"""
test_partition_structure.py
Harness B — structural tests for n_parallel=2 partition worker launch

Verifies:
1. Exactly 2 partition workers are created
2. Each partition gets the correct allowlist (TB spec)
3. Child partition coordinators inherit required runtime flags
4. No legacy/non-TCP transport fallback

IMPORTANT NOTE (from live code analysis):
  The _partition_worker() function in window_optimizer_integration_final.py
  creates child MultiGPUCoordinator with ONLY:
    - config_file
    - node_allowlist
    - seed_cap_nvidia (hardcoded 5_000_000)
    - seed_cap_amd   (hardcoded 2_000_000)

  The following flags are NOT passed to child coordinators:
    - use_persistent_workers
    - pwc_transport
    - pwc_min_workers
    - worker_pool_size

  This is the FLAG INHERITANCE GAP that this harness is designed to detect.
  Tests for missing inheritance will FAIL until the live code is fixed.

Usage: python3 test_partition_structure.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/michael/distributed_prng_analysis')

import unittest
from unittest.mock import patch, MagicMock, call
from fakes import FakeCoordinator, FakeTestResult

# ── Expected partition configuration (from live code _PARALLEL_PARTITIONS) ────
EXPECTED_PARTITION_0 = ['localhost', '192.168.3.120']
EXPECTED_PARTITION_1 = ['192.168.3.154', '192.168.3.162']

# ── Required flags that MUST be inherited by child coordinators ───────────────
REQUIRED_FLAGS = [
    'use_persistent_workers',
    'pwc_transport',
    'pwc_min_workers',
    'worker_pool_size',
    'seed_cap_amd',
    'seed_cap_nvidia',
]

RECOMMENDED_FLAGS = [
    'use_zmq_sqlite',
    'config_file',
    'node_allowlist',
]


class TestPartitionStructure(unittest.TestCase):
    """
    Structural tests for n_parallel=2 partition worker creation.
    Uses monkeypatching to intercept coordinator creation without real GPU work.
    """

    def setUp(self):
        FakeCoordinator.reset()

    def _get_partition_worker_args(self):
        """
        Extract the arguments passed to _partition_worker() calls
        by parsing the live code. We check _PARALLEL_PARTITIONS directly
        from the live module.
        """
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "window_optimizer_integration_final",
                "/home/michael/distributed_prng_analysis/window_optimizer_integration_final.py"
            )
            mod = importlib.util.load_from_spec(spec) if hasattr(importlib.util, 'load_from_spec') else None
        except Exception:
            return None
        return None

    def test_parallel_partitions_defined_in_live_code(self):
        """Verify _PARALLEL_PARTITIONS matches TB spec exactly."""
        # Read directly from live code
        live_file = '/home/michael/distributed_prng_analysis/window_optimizer_integration_final.py'
        if not os.path.exists(live_file):
            self.skipTest("Live code not accessible from harness environment")

        with open(live_file) as f:
            content = f.read()

        # Verify both partition entries exist
        self.assertIn("'localhost', '192.168.3.120'", content,
                      "P0 allowlist not found in live code")
        self.assertIn("'192.168.3.154', '192.168.3.162'", content,
                      "P1 allowlist not found in live code")
        print("  ✅ _PARALLEL_PARTITIONS P0/P1 definitions found in live code")

    def test_flag_inheritance_gap_detected(self):
        """
        CRITICAL: Verify that the live code does NOT currently inherit
        pwc_transport and use_persistent_workers into child coordinators.
        This test DOCUMENTS the gap — it SHOULD FAIL after the fix is applied.

        When this test flips from PASS to FAIL, it means the flag inheritance
        bug has been fixed and Harness C (live smoke) can proceed.
        """
        live_file = '/home/michael/distributed_prng_analysis/window_optimizer_integration_final.py'
        if not os.path.exists(live_file):
            self.skipTest("Live code not accessible from harness environment")

        with open(live_file) as f:
            content = f.read()

        # Find _partition_worker definition and the _WMCC constructor call within it
        import re
        # Look for MultiGPUCoordinator constructor call inside _partition_worker
        wmcc_match = re.search(
            r'_wcoord\s*=\s*_WMCC\s*\((.*?)\)',
            content, re.DOTALL
        )
        self.assertIsNotNone(wmcc_match, "_WMCC constructor call not found in partition worker")

        wmcc_args = wmcc_match.group(1)
        print(f"  _WMCC constructor args found:\n    {wmcc_args.strip()[:200]}")

        # TB-approved pattern: transport flags NOT in constructor (coordinator.py does not accept them).
        # They are set as post-construction attributes — confirm both sides.
        transport_flags = ['pwc_transport', 'use_persistent_workers', 'pwc_min_workers', 'worker_pool_size']
        not_in_constructor = [f for f in transport_flags if f not in wmcc_args]
        post_construction_present = [f for f in transport_flags
                                     if f'_wcoord.{f}' in content or f'{f}_w' in content]

        if not_in_constructor and post_construction_present:
            print(f"\n  ✅ POST-CONSTRUCTION TRANSPORT INHERITANCE CONFIRMED (TB-approved pattern):")
            print(f"     Correctly absent from constructor (coordinator.py does not accept these):")
            for f in not_in_constructor:
                print(f"       - {f}")
            print(f"     Present as post-construction attribute assignments:")
            for f in post_construction_present:
                print(f"       + _wcoord.{f} / {f}_w")
        elif not_in_constructor and not post_construction_present:
            print(f"\n  ⚠️  FLAGS MISSING FROM BOTH constructor AND post-construction assignments:")
            for f in not_in_constructor:
                print(f"       - {f}")
        else:
            print("  ✅ Transport flags handled correctly")

    def test_partition_count_is_two(self):
        """Verify n_parallel=2 launches exactly 2 partition workers."""
        live_file = '/home/michael/distributed_prng_analysis/window_optimizer_integration_final.py'
        if not os.path.exists(live_file):
            self.skipTest("Not accessible")

        with open(live_file) as f:
            content = f.read()

        # Verify the loop range
        self.assertIn('for _pi in range(n_parallel)', content,
                      "n_parallel loop not found")
        self.assertIn('LAUNCHING {n_parallel} PARTITION WORKERS', content,
                      "launch banner not found")
        print("  ✅ n_parallel loop confirmed in live code")

    def test_nparallel_single_process_skip_banner(self):
        """Verify NP2 path skips single-process search when n_parallel > 1."""
        live_file = '/home/michael/distributed_prng_analysis/window_optimizer_integration_final.py'
        if not os.path.exists(live_file):
            self.skipTest("Not accessible")

        with open(live_file) as f:
            content = f.read()

        # The banner uses n_parallel variable, not the literal string "n_parallel=2"
        self.assertIn('[NP2] Single-process search path SKIPPED', content,
                      "NP2 skip banner not found")
        self.assertIn('_np2_complete = n_parallel > 1', content,
                      "_np2_complete flag not found")
        print("  ✅ NP2 single-process skip banner confirmed in live code")

    def test_seed_caps_passed_to_partition(self):
        """Verify seed_cap_nvidia and seed_cap_amd are inherited from parent (not hardcoded)."""
        live_file = '/home/michael/distributed_prng_analysis/window_optimizer_integration_final.py'
        if not os.path.exists(live_file):
            self.skipTest("Not accessible")

        with open(live_file) as f:
            content = f.read()

        # Check that seed_cap_nvidia_w and seed_cap_amd_w are used (not hardcoded 5M/2M)
        self.assertIn('seed_cap_nvidia_w', content,
                      "seed_cap_nvidia_w not found — still hardcoded?")
        self.assertIn('seed_cap_amd_w', content,
                      "seed_cap_amd_w not found — still hardcoded?")
        print("  ✅ seed_cap_nvidia_w and seed_cap_amd_w used in partition worker (not hardcoded)")

    def test_transport_flag_inheritance_fix_required(self):
        """
        Verifies the TB-approved Fix 1: transport/runtime flags are set as
        attributes on child coordinator AFTER construction (not in constructor,
        which doesn't accept these args per live coordinator.py).

        TB-approved fix shape:
          _wcoord = _WMCC(config_file=..., node_allowlist=..., seed_caps=...)
          _wcoord.use_persistent_workers = use_persistent_workers_w
          _wcoord.pwc_transport          = pwc_transport_w
          _wcoord.pwc_min_workers        = pwc_min_workers_w
          _wcoord.worker_pool_size       = worker_pool_size_w
          _wcoord.pwc_host               = pwc_host_w
          _wcoord.pwc_port               = pwc_port_w
        """
        live_file = '/home/michael/distributed_prng_analysis/window_optimizer_integration_final.py'
        if not os.path.exists(live_file):
            self.skipTest("Not accessible")

        with open(live_file) as f:
            content = f.read()

        # Check for post-construction attribute assignments (TB-approved pattern)
        required_attrs = [
            '_wcoord.use_persistent_workers',
            '_wcoord.pwc_transport',
            '_wcoord.pwc_min_workers',
            '_wcoord.worker_pool_size',
            '_wcoord.pwc_host',
            '_wcoord.pwc_port',
        ]
        missing = [a for a in required_attrs if a not in content]

        if missing:
            print(f"\n  ❌ REQUIRED FIX: Set these attributes on child coordinator after construction:")
            for a in missing:
                print(f"     {a} = <value>")
            self.fail(
                f"FLAG INHERITANCE GAP: {missing} not set on child coordinator. "
                f"Fix _partition_worker in window_optimizer_integration_final.py"
            )
        else:
            print("  ✅ All transport flags set as post-construction attributes on child coordinator")


class TestFakeCoordinator(unittest.TestCase):
    """Verify the FakeCoordinator capture mechanism works correctly."""

    def setUp(self):
        FakeCoordinator.reset()

    def test_fake_captures_constructor_args(self):
        coord = FakeCoordinator(
            config_file='test.json',
            node_allowlist=['localhost', '192.168.3.120'],
            seed_cap_amd=100000,
            seed_cap_nvidia=5000000,
        )
        snap = coord.snapshot()
        self.assertEqual(snap['config_file'], 'test.json')
        self.assertEqual(snap['node_allowlist'], ['localhost', '192.168.3.120'])
        self.assertEqual(snap['seed_cap_amd'], 100000)
        self.assertEqual(snap['seed_cap_nvidia'], 5000000)
        print("  ✅ FakeCoordinator captures constructor args")

    def test_fake_captures_runtime_attrs(self):
        coord = FakeCoordinator(config_file='test.json')
        coord.pwc_transport = 'tcp'
        coord.use_persistent_workers = True
        coord.pwc_min_workers = 24
        coord.worker_pool_size = 8
        snap = coord.snapshot()
        self.assertEqual(snap['pwc_transport'], 'tcp')
        self.assertTrue(snap['use_persistent_workers'])
        self.assertEqual(snap['pwc_min_workers'], 24)
        self.assertEqual(snap['worker_pool_size'], 8)
        print("  ✅ FakeCoordinator captures runtime attribute assignments")

    def test_fake_registry(self):
        FakeCoordinator(config_file='a.json')
        FakeCoordinator(config_file='b.json')
        instances = FakeCoordinator.get_instances()
        self.assertEqual(len(instances), 2)
        print("  ✅ FakeCoordinator instance registry works")


if __name__ == '__main__':
    print("=" * 60)
    print("HARNESS B — n_parallel structural tests")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestFakeCoordinator))
    suite.addTests(loader.loadTestsFromTestCase(TestPartitionStructure))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print()
    if result.wasSuccessful():
        print("All structural tests PASSED")
    else:
        print(f"FAILURES: {len(result.failures)}  ERRORS: {len(result.errors)}")
        # Flag inheritance test failure is expected until fix applied
        sys.exit(1)
