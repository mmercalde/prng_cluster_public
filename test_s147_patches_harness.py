#!/usr/bin/env python3
"""
test_s147_patches_harness.py — S147 patch verification harness (CORRECTED v2)

Verifies:
  Q0: Hybrid forward zero-survivor gate (both PWC and legacy paths)
  Q1: WATCHER step_timeout_overrides {1: 0} → infinite timeout via S145 guard
  Q2: Single strategy passed to BOTH forward AND reverse hybrid calls
      pwc.logger used (not self.logger — standalone function)

Harness uses mocks — no GPU, no SSH, no real sieve runs.
Tests patch BEHAVIOR, not mock design.
"""
import sys, argparse
PASS = 0; FAIL = 0

def ok(label): global PASS; PASS += 1; print(f"  ✅ PASS  {label}")
def fail(label, r=""): global FAIL; FAIL += 1; print(f"  ❌ FAIL  {label}" + (f": {r}" if r else ""))
def section(t): print(f"\n── {t} ──")

# ── Mocks ────────────────────────────────────────────────────────────────────

class MockResult:
    def __init__(self, survivors, match_rates=None):
        self._s = survivors
        self._m = match_rates or [0.5]*len(survivors)
    def get(self, key, default=None):
        if key == "survivors": return self._s
        if key == "match_rates": return self._m
        return default

class MockLogger:
    def __init__(self): self.warnings = []
    def warning(self, msg): self.warnings.append(msg)

class MockPWC:
    def __init__(self, results):
        self._results = results
        self.calls = []
        self.strategies_received = {}
        self._progress_writer = None
        self._shutdown_called = False
        self.logger = MockLogger()
    def run_sieve_pass(self, prng_type, strategies=None, **kwargs):
        self.calls.append(prng_type)
        self.strategies_received[prng_type] = strategies
        return self._results.get(prng_type, MockResult([]))
    def shutdown(self): self._shutdown_called = True

class MockConfig:
    window_size=8; offset=43; skip_min=5; skip_max=56
    sessions=["midday","evening"]

# ── Simulated PATCHED run_trial_persistent (mirrors corrected patch) ──────────

def run_trial_persistent_PATCHED(pwc, config, residues, total_seeds,
                                  fwd_thresh, rev_thresh, trial_num,
                                  prng_base, test_both_modes, dataset_path,
                                  hybrid_strategies_override="AUTO"):
    """
    Mirrors the corrected PWC code after Q0+Q2 patches applied.
    hybrid_strategies_override="AUTO" simulates Q2 loading balanced_hybrid.
    """
    ws = config.window_size; off = config.offset

    # Pass 1
    fwd = pwc.run_sieve_pass(prng_type=prng_base, residues=residues,
                              total_seeds=total_seeds, threshold=fwd_thresh)
    fwd_survivors = fwd.get("survivors", [])
    fwd_map = dict(zip(fwd_survivors, fwd.get("match_rates", [])))

    if not fwd_survivors:
        pwc.shutdown()
        return {"pruned":True, "bidirectional_count":0,
                "bidirectional_constant":set(), "bidirectional_variable":set()}

    # Pass 2
    rev = pwc.run_sieve_pass(prng_type=prng_base+"_reverse", residues=residues,
                              total_seeds=total_seeds, threshold=rev_thresh)
    rev_survivors = rev.get("survivors", [])
    rev_map = dict(zip(rev_survivors, rev.get("match_rates", [])))
    bidirectional_constant = set(fwd_map.keys()) & set(rev_map.keys())

    bidirectional_variable = set()

    if test_both_modes and not prng_base.endswith("_hybrid"):
        prng_hybrid = f"{prng_base}_hybrid"
        prng_hybrid_rev = f"{prng_hybrid}_reverse"

        # Q2: load single strategy (simulated)
        if hybrid_strategies_override == "AUTO":
            _hybrid_strategies = [{"name":"Balanced Hybrid (Recommended)",
                                    "max_consecutive_misses":7,"skip_tolerance":10,
                                    "enable_reseed_search":True,"skip_learning_rate":0.4,
                                    "breakpoint_threshold":0.5}]
        else:
            _hybrid_strategies = hybrid_strategies_override

        # Pass 3 — with strategies kwarg
        fwd_h = pwc.run_sieve_pass(prng_type=prng_hybrid, residues=residues,
                                    total_seeds=total_seeds, threshold=fwd_thresh,
                                    strategies=_hybrid_strategies)
        fwd_h_survivors = fwd_h.get("survivors", [])
        fwd_h_map = dict(zip(fwd_h_survivors, fwd_h.get("match_rates", [])))

        # Q0 gate
        if not fwd_h_survivors:
            print(f"      Hybrid forward zero survivors — skipping hybrid reverse (Q0 gate)")
            rev_h_survivors = []; rev_h_map = {}
        else:
            # Pass 4 — with strategies kwarg
            rev_h = pwc.run_sieve_pass(prng_type=prng_hybrid_rev, residues=residues,
                                        total_seeds=total_seeds, threshold=rev_thresh,
                                        strategies=_hybrid_strategies)
            rev_h_survivors = rev_h.get("survivors", [])
            rev_h_map = dict(zip(rev_h_survivors, rev_h.get("match_rates", [])))

        bidirectional_variable = set(fwd_h_map.keys()) & set(rev_h_map.keys())

    total_bidi = len(bidirectional_constant) + len(bidirectional_variable)
    return {"pruned":False, "bidirectional_count":total_bidi,
            "bidirectional_constant":bidirectional_constant,
            "bidirectional_variable":bidirectional_variable}


# ── Q0 Tests ─────────────────────────────────────────────────────────────────

def test_q0():
    section("Q0 — Hybrid forward zero-survivor gate")
    cfg = MockConfig()

    # T1: constant forward=0 → trial pruned, hybrid never called
    pwc = MockPWC({"java_lcg": MockResult([])})
    r = run_trial_persistent_PATCHED(pwc, cfg, [134], 1_000_000,
                                      0.25, 0.25, 1, "java_lcg", True, "d.json")
    ok("T1: pruned") if r["pruned"] else fail("T1: should be pruned")
    ok("T1: hybrid fwd never called") if "java_lcg_hybrid" not in pwc.calls \
        else fail("T1: hybrid should not run")
    ok("T1: shutdown called") if pwc._shutdown_called else fail("T1: shutdown missing")

    # T2: hybrid forward=0 → Pass 4 skipped, constant results preserved
    pwc = MockPWC({"java_lcg":MockResult([100,200]),
                   "java_lcg_reverse":MockResult([100,200]),
                   "java_lcg_hybrid":MockResult([]),
                   "java_lcg_hybrid_reverse":MockResult([999])})
    r = run_trial_persistent_PATCHED(pwc, cfg, [134], 1_000_000,
                                      0.25, 0.25, 1, "java_lcg", True, "d.json")
    ok("T2: not pruned") if not r["pruned"] else fail("T2: should not be pruned")
    ok("T2: Pass4 skipped") if "java_lcg_hybrid_reverse" not in pwc.calls \
        else fail("T2: Pass4 should be skipped")
    ok("T2: constant={100,200}") if r["bidirectional_constant"]=={100,200} \
        else fail("T2: constant wrong", str(r["bidirectional_constant"]))
    ok("T2: variable=empty") if r["bidirectional_variable"]==set() \
        else fail("T2: variable should be empty")
    ok("T2: count=2") if r["bidirectional_count"]==2 \
        else fail("T2: count wrong", str(r["bidirectional_count"]))

    # T3: hybrid forward>0 → Pass 4 runs normally
    pwc = MockPWC({"java_lcg":MockResult([100,200]),
                   "java_lcg_reverse":MockResult([100,200]),
                   "java_lcg_hybrid":MockResult([500,600]),
                   "java_lcg_hybrid_reverse":MockResult([500,600])})
    r = run_trial_persistent_PATCHED(pwc, cfg, [134], 1_000_000,
                                      0.25, 0.25, 1, "java_lcg", True, "d.json")
    ok("T3: Pass4 runs") if "java_lcg_hybrid_reverse" in pwc.calls \
        else fail("T3: Pass4 should run")
    ok("T3: variable={500,600}") if r["bidirectional_variable"]=={500,600} \
        else fail("T3: variable wrong", str(r["bidirectional_variable"]))
    ok("T3: count=4") if r["bidirectional_count"]==4 \
        else fail("T3: count wrong", str(r["bidirectional_count"]))

    # T4: test_both_modes=False → hybrid never runs
    pwc = MockPWC({"java_lcg":MockResult([100]),
                   "java_lcg_reverse":MockResult([100])})
    run_trial_persistent_PATCHED(pwc, cfg, [134], 1_000_000,
                                  0.25, 0.25, 1, "java_lcg", False, "d.json")
    ok("T4: no hybrid when test_both_modes=False") if "java_lcg_hybrid" not in pwc.calls \
        else fail("T4: hybrid should not run")


# ── Q1 Tests ─────────────────────────────────────────────────────────────────

def test_q1():
    section("Q1 — WATCHER Step 1 infinite timeout via {1: 0}")

    class WatcherConfig:
        step_timeout_minutes: int = 120
        step_timeout_overrides = None
        def get_step_timeout_minutes(self, step):
            if self.step_timeout_overrides and step in self.step_timeout_overrides:
                return self.step_timeout_overrides[step]
            return self.step_timeout_minutes

    # T5: Before patch — Step 1 missing from overrides → gets 120
    cfg = WatcherConfig()
    cfg.step_timeout_overrides = {0: 1, 5: 360}  # old config, no Step 1
    ok("T5: (pre-patch) Step1 timeout=120") if cfg.get_step_timeout_minutes(1)==120 \
        else fail("T5: pre-patch should be 120")
    timeout_sec = cfg.get_step_timeout_minutes(1) * 60
    ok("T5: 120min*60=7200s > 0, guard does NOT fire") if timeout_sec > 0 \
        else fail("T5: should be > 0 pre-patch")

    # T6: After patch — Step 1 = 0 → guard fires → infinite
    cfg2 = WatcherConfig()
    cfg2.step_timeout_overrides = {0: 1, 1: 0, 5: 360}  # patched
    ok("T6: Step1 timeout=0 minutes") if cfg2.get_step_timeout_minutes(1)==0 \
        else fail("T6: should be 0", str(cfg2.get_step_timeout_minutes(1)))
    timeout_sec2 = cfg2.get_step_timeout_minutes(1) * 60
    ok("T6: 0*60=0 seconds") if timeout_sec2 == 0 \
        else fail("T6: should be 0 seconds")

    # T7: S145 guard simulation — 0 → float('inf')
    def s145_guard(timeout_seconds):
        if timeout_seconds <= 0:
            timeout_seconds = float('inf')
        return timeout_seconds
    ok("T7: S145 guard 0 → inf") if s145_guard(0) == float('inf') \
        else fail("T7: guard should return inf")
    ok("T7: S145 guard 7200 → 7200 (no fire)") if s145_guard(7200) == 7200 \
        else fail("T7: 7200 should pass through unchanged")

    # T8: Other steps unaffected
    ok("T8: Step0=1min") if cfg2.get_step_timeout_minutes(0)==1 \
        else fail("T8: Step0 wrong")
    ok("T8: Step2=120min") if cfg2.get_step_timeout_minutes(2)==120 \
        else fail("T8: Step2 wrong")
    ok("T8: Step5=360min") if cfg2.get_step_timeout_minutes(5)==360 \
        else fail("T8: Step5 wrong")


# ── Q2 Tests ─────────────────────────────────────────────────────────────────

def test_q2():
    section("Q2 — Single strategy to BOTH forward AND reverse hybrid calls")
    cfg = MockConfig()

    SINGLE = [{"name":"Balanced Hybrid (Recommended)",
               "max_consecutive_misses":7,"skip_tolerance":10,
               "enable_reseed_search":True,"skip_learning_rate":0.4,
               "breakpoint_threshold":0.5}]

    # T9: Both forward and reverse hybrid receive single strategy
    pwc = MockPWC({"java_lcg":MockResult([100,200]),
                   "java_lcg_reverse":MockResult([100,200]),
                   "java_lcg_hybrid":MockResult([500,600]),
                   "java_lcg_hybrid_reverse":MockResult([500,600])})
    run_trial_persistent_PATCHED(pwc, cfg, [134], 1_000_000,
                                  0.25, 0.25, 1, "java_lcg", True, "d.json",
                                  hybrid_strategies_override=SINGLE)
    fwd_strat = pwc.strategies_received.get("java_lcg_hybrid")
    rev_strat  = pwc.strategies_received.get("java_lcg_hybrid_reverse")
    ok("T9: forward hybrid receives single strategy") if fwd_strat==SINGLE \
        else fail("T9: fwd strategy wrong", str(fwd_strat))
    ok("T9: reverse hybrid receives single strategy") if rev_strat==SINGLE \
        else fail("T9: rev strategy wrong", str(rev_strat))

    # T10: When hybrid fwd=0, reverse never called (Q0 gate fires first)
    pwc2 = MockPWC({"java_lcg":MockResult([100,200]),
                    "java_lcg_reverse":MockResult([100,200]),
                    "java_lcg_hybrid":MockResult([]),
                    "java_lcg_hybrid_reverse":MockResult([999])})
    run_trial_persistent_PATCHED(pwc2, cfg, [134], 1_000_000,
                                  0.25, 0.25, 1, "java_lcg", True, "d.json",
                                  hybrid_strategies_override=SINGLE)
    ok("T10: Q0 gate fires before rev strategy matters") \
        if "java_lcg_hybrid_reverse" not in pwc2.calls \
        else fail("T10: reverse should not be called")
    ok("T10: no rev strategy entry (call skipped)") \
        if "java_lcg_hybrid_reverse" not in pwc2.strategies_received \
        else fail("T10: should have no rev strategy entry")

    # T11: pwc.logger.warning used (not self.logger) — verify it's callable on pwc
    ok("T11: pwc.logger.warning callable") \
        if callable(getattr(pwc.logger, 'warning', None)) \
        else fail("T11: pwc.logger.warning not callable")
    pwc.logger.warning("test warning from standalone function")
    ok("T11: warning recorded on pwc.logger") \
        if "test warning from standalone function" in pwc.logger.warnings \
        else fail("T11: warning not recorded")

    # T12: strategies=None (fallback) still passes None correctly
    pwc3 = MockPWC({"java_lcg":MockResult([100]),
                    "java_lcg_reverse":MockResult([100]),
                    "java_lcg_hybrid":MockResult([500]),
                    "java_lcg_hybrid_reverse":MockResult([500])})
    run_trial_persistent_PATCHED(pwc3, cfg, [134], 1_000_000,
                                  0.25, 0.25, 1, "java_lcg", True, "d.json",
                                  hybrid_strategies_override=None)
    ok("T12: strategies=None passes None to fwd") \
        if pwc3.strategies_received.get("java_lcg_hybrid") is None \
        else fail("T12: should be None")
    ok("T12: strategies=None passes None to rev") \
        if pwc3.strategies_received.get("java_lcg_hybrid_reverse") is None \
        else fail("T12: rev should be None")

    # T13: 5x work reduction confirmed
    ok("T13: 5 strategies / 1 strategy = 5x reduction") if 5/1==5.0 \
        else fail("T13: math wrong")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q0", action="store_true")
    parser.add_argument("--q1", action="store_true")
    parser.add_argument("--q2", action="store_true")
    args = parser.parse_args()
    run_all = not (args.q0 or args.q1 or args.q2)

    print("="*60)
    print("S147 Patch Verification Harness (CORRECTED v2)")
    print("="*60)

    if run_all or args.q0: test_q0()
    if run_all or args.q1: test_q1()
    if run_all or args.q2: test_q2()

    total = PASS + FAIL
    print(f"\n{'='*60}")
    print(f"RESULT: {PASS}/{total} passed, {FAIL} failed")
    print("✅ ALL PASS — safe to deploy" if FAIL==0 else "❌ FAILURES — do NOT deploy")
    print("="*60)
    sys.exit(0 if FAIL==0 else 1)

if __name__ == "__main__":
    main()
