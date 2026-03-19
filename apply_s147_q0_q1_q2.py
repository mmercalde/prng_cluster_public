#!/usr/bin/env python3
"""
apply_s147_q0_q1_q2.py — Three patches from S147 TB rulings (CORRECTED v2)

Q0: Hybrid forward zero-survivor gate — both PWC and legacy paths
Q1: Step 1 infinite timeout — set 1:0 in step_timeout_overrides
Q2: Single strategy (balanced_hybrid) for full-range hybrid scan
    - strategies kwarg added to BOTH forward AND reverse hybrid calls
    - uses pwc.logger (not self.logger — standalone function)

Apply order: Q2 first (adds loader), then Q0A (adds gate + rev kwarg), then Q1

Usage:
    python3 apply_s147_q0_q1_q2.py --dry-run   # preview changes
    python3 apply_s147_q0_q1_q2.py              # apply all patches

TB ruling: FULL MODE ONLY — selective flags removed.
Q0 and Q2 are interdependent (Q0A references _hybrid_strategies from Q2).
"""
import os, shutil, sys, argparse

ROOT = os.path.expanduser("~/distributed_prng_analysis")
def path(p): return os.path.join(ROOT, p)
DRY_RUN = False
results = []

def read_file(fp):
    with open(fp, "r", encoding="utf-8") as f: return f.read()

def write_file(fp, content):
    with open(fp, "w", encoding="utf-8") as f: f.write(content)

def backup(fp):
    bak = fp + ".bak_s147"
    if not os.path.exists(bak):
        shutil.copy(fp, bak)
        print(f"  BAK  {os.path.basename(bak)}")

def apply_patch(fp, label, old, new):
    if not os.path.exists(fp):
        print(f"  SKIP {label}: file not found")
        results.append((label, False)); return False
    content = read_file(fp)
    count = content.count(old)
    if count == 0:
        print(f"  SKIP {label}: anchor not found")
        results.append((label, False)); return False
    if count > 1:
        print(f"  WARN {label}: {count} matches — ambiguous")
        results.append((label, False)); return False
    before = len(content.splitlines())
    new_content = content.replace(old, new, 1)
    after = len(new_content.splitlines())
    if DRY_RUN:
        print(f"  DRY  {label}: {before} → {after} lines")
        results.append((label, True)); return True
    backup(fp)
    write_file(fp, new_content)
    print(f"  OK   {label}: {before} → {after} lines")
    results.append((label, True)); return True


# ── Q2: strategy loader + forward strategies kwarg ───────────────────────────
Q2_OLD = '''        if test_both_modes and not prng_base.endswith("_hybrid"):
            prng_hybrid = f"{prng_base}_hybrid"
            prng_hybrid_rev = f"{prng_hybrid}_reverse"

            print(f"    Running FORWARD sieve ({prng_hybrid}) [VARIABLE SKIP] [PERSISTENT]...")
            fwd_h_result = pwc.run_sieve_pass(
                prng_type    = prng_hybrid,
                residues     = residues,
                total_seeds  = total_seeds,
                threshold    = forward_threshold,
                window_size  = ws,
                dataset_path = dataset_path,
                output_file  = f"results/window_opt_forward_hybrid_{ws}_{off}_t{trial_number}.json",
                offset       = config.offset,
                sessions     = list(config.sessions) if hasattr(config, 'sessions') else ["midday", "evening"],
                skip_range   = [config.skip_min, config.skip_max] if hasattr(config, 'skip_min') else [0, 147],
            )'''

Q2_NEW = '''        if test_both_modes and not prng_base.endswith("_hybrid"):
            prng_hybrid = f"{prng_base}_hybrid"
            prng_hybrid_rev = f"{prng_hybrid}_reverse"

            # [S147 Q2] Single strategy for full-range scan — 5x work reduction
            # TB ruling: balanced_hybrid for discovery, all 5 for refinement only
            # Uses pwc.logger — run_trial_persistent is a standalone function, not a method
            try:
                from hybrid_strategy import get_strategy as _get_strategy
                _s = _get_strategy("balanced_hybrid")
                _hybrid_strategies = [_s.to_dict() if hasattr(_s, "to_dict") else vars(_s)]
            except Exception as _e:
                pwc.logger.warning(f"Q2: could not load balanced_hybrid ({_e}) — using all strategies")
                _hybrid_strategies = None  # fallback: auto-load all in run_sieve_pass

            print(f"    Running FORWARD sieve ({prng_hybrid}) [VARIABLE SKIP] [PERSISTENT]...")
            fwd_h_result = pwc.run_sieve_pass(
                prng_type    = prng_hybrid,
                residues     = residues,
                total_seeds  = total_seeds,
                threshold    = forward_threshold,
                window_size  = ws,
                dataset_path = dataset_path,
                output_file  = f"results/window_opt_forward_hybrid_{ws}_{off}_t{trial_number}.json",
                offset       = config.offset,
                sessions     = list(config.sessions) if hasattr(config, 'sessions') else ["midday", "evening"],
                skip_range   = [config.skip_min, config.skip_max] if hasattr(config, 'skip_min') else [0, 147],
                strategies   = _hybrid_strategies,  # [S147 Q2] single strategy
            )'''


# ── Q0A: PWC hybrid gate + reverse strategies kwarg ──────────────────────────
# NOTE: Q2 must run before Q0A — Q0A_NEW references _hybrid_strategies
Q0A_OLD = '''            fwd_h_survivors   = fwd_h_result.get("survivors", [])
            fwd_h_match_rates = fwd_h_result.get("match_rates", [])
            fwd_h_map = dict(zip(fwd_h_survivors, fwd_h_match_rates))
            print(f"      Forward (variable): {len(fwd_h_survivors):,} survivors")

            print(f"    Running REVERSE sieve ({prng_hybrid_rev}) [VARIABLE SKIP] [PERSISTENT]...")
            rev_h_result = pwc.run_sieve_pass(
                prng_type    = prng_hybrid_rev,
                residues     = residues,
                total_seeds  = total_seeds,
                threshold    = reverse_threshold,
                window_size  = ws,
                dataset_path = dataset_path,
                output_file  = f"results/window_opt_reverse_hybrid_{ws}_{off}_t{trial_number}.json",
                offset       = config.offset,
                sessions     = list(config.sessions) if hasattr(config, 'sessions') else ["midday", "evening"],
                skip_range   = [config.skip_min, config.skip_max] if hasattr(config, 'skip_min') else [0, 147],
            )
            rev_h_survivors   = rev_h_result.get("survivors", [])
            rev_h_match_rates = rev_h_result.get("match_rates", [])
            rev_h_map = dict(zip(rev_h_survivors, rev_h_match_rates))
            print(f"      Reverse (variable): {len(rev_h_survivors):,} survivors")

            bidirectional_variable = set(fwd_h_map.keys()) & set(rev_h_map.keys())
            print(f"      ✨ Bidirectional (variable): {len(bidirectional_variable):,} survivors")

            fwd_records_hybrid = [{"seed": s, "match_rate": fwd_h_map[s]} for s in fwd_h_survivors]
            rev_records_hybrid = [{"seed": s, "match_rate": rev_h_map[s]} for s in rev_h_survivors]'''

Q0A_NEW = '''            fwd_h_survivors   = fwd_h_result.get("survivors", [])
            fwd_h_match_rates = fwd_h_result.get("match_rates", [])
            fwd_h_map = dict(zip(fwd_h_survivors, fwd_h_match_rates))
            print(f"      Forward (variable): {len(fwd_h_survivors):,} survivors")

            # [S147 Q0] Gate: skip hybrid reverse if hybrid forward = 0
            # Mirrors constant-skip B1 gate. SKIP not prune — constant results preserved.
            if not fwd_h_survivors:
                print(f"      Hybrid forward zero survivors — skipping hybrid reverse (Q0 gate)")
                rev_h_survivors   = []
                rev_h_match_rates = []
                rev_h_map         = {}
            else:
                print(f"    Running REVERSE sieve ({prng_hybrid_rev}) [VARIABLE SKIP] [PERSISTENT]...")
                rev_h_result = pwc.run_sieve_pass(
                    prng_type    = prng_hybrid_rev,
                    residues     = residues,
                    total_seeds  = total_seeds,
                    threshold    = reverse_threshold,
                    window_size  = ws,
                    dataset_path = dataset_path,
                    output_file  = f"results/window_opt_reverse_hybrid_{ws}_{off}_t{trial_number}.json",
                    offset       = config.offset,
                    sessions     = list(config.sessions) if hasattr(config, 'sessions') else ["midday", "evening"],
                    skip_range   = [config.skip_min, config.skip_max] if hasattr(config, 'skip_min') else [0, 147],
                    strategies   = _hybrid_strategies,  # [S147 Q2] single strategy
                )
                rev_h_survivors   = rev_h_result.get("survivors", [])
                rev_h_match_rates = rev_h_result.get("match_rates", [])
                rev_h_map = dict(zip(rev_h_survivors, rev_h_match_rates))
                print(f"      Reverse (variable): {len(rev_h_survivors):,} survivors")

            bidirectional_variable = set(fwd_h_map.keys()) & set(rev_h_map.keys())
            print(f"      ✨ Bidirectional (variable): {len(bidirectional_variable):,} survivors")

            fwd_records_hybrid = [{"seed": s, "match_rate": fwd_h_map[s]} for s in fwd_h_survivors]
            rev_records_hybrid = [{"seed": s, "match_rate": rev_h_map[s]} for s in rev_h_survivors]'''


# ── Q0B: legacy coordinator hybrid gate ──────────────────────────────────────
Q0B_OLD = '''        forward_records_hybrid = extract_survivor_records(forward_result_hybrid)
        print(f"      Forward (variable): {len(forward_records_hybrid):,} survivors")

        print(f"    Running REVERSE sieve ({prng_hybrid}_reverse) [VARIABLE SKIP]...")
        reverse_args_hybrid = Args()
        reverse_args_hybrid.threshold = reverse_threshold
        reverse_args_hybrid.step_name = f"Reverse Sieve ({prng_hybrid}) [VARIABLE]"
        reverse_args_hybrid.prng_type = prng_hybrid + "_reverse"  # e.g. java_lcg_hybrid_reverse

        reverse_result_hybrid = coordinator.execute_distributed_analysis(
            reverse_args_hybrid.target_file,
            f\'results/window_opt_reverse_hybrid_{config.window_size}_{config.offset}_t{trial_number}.json\',  # S115 M3
            reverse_args_hybrid,
            reverse_args_hybrid.seeds,
            1000, 8, 50
        )

        reverse_records_hybrid = extract_survivor_records(reverse_result_hybrid)
        print(f"      Reverse (variable): {len(reverse_records_hybrid):,} survivors")'''

Q0B_NEW = '''        forward_records_hybrid = extract_survivor_records(forward_result_hybrid)
        print(f"      Forward (variable): {len(forward_records_hybrid):,} survivors")

        # [S147 Q0] Gate: skip hybrid reverse if hybrid forward = 0
        # SKIP not prune — constant-skip results preserved.
        if not forward_records_hybrid:
            print(f"      Hybrid forward zero survivors — skipping hybrid reverse (Q0 gate)")
            reverse_records_hybrid = []
        else:
            print(f"    Running REVERSE sieve ({prng_hybrid}_reverse) [VARIABLE SKIP]...")
            reverse_args_hybrid = Args()
            reverse_args_hybrid.threshold = reverse_threshold
            reverse_args_hybrid.step_name = f"Reverse Sieve ({prng_hybrid}) [VARIABLE]"
            reverse_args_hybrid.prng_type = prng_hybrid + "_reverse"  # e.g. java_lcg_hybrid_reverse

            reverse_result_hybrid = coordinator.execute_distributed_analysis(
                reverse_args_hybrid.target_file,
                f\'results/window_opt_reverse_hybrid_{config.window_size}_{config.offset}_t{trial_number}.json\',  # S115 M3
                reverse_args_hybrid,
                reverse_args_hybrid.seeds,
                1000, 8, 50
            )

            reverse_records_hybrid = extract_survivor_records(reverse_result_hybrid)
            print(f"      Reverse (variable): {len(reverse_records_hybrid):,} survivors")'''


# ── Q1: watcher_agent.py — Step 1 infinite timeout ───────────────────────────
Q1_OLD = \
'        step_timeout_overrides={0: 1, 5: 360}  # Step 1 has no timeout — production runs are 13-18hrs'
Q1_NEW = \
'        step_timeout_overrides={0: 1, 1: 0, 5: 360}  # [S147 Q1] 1:0 → S145 guard fires (<=0 → inf)'


def main():
    global DRY_RUN
    parser = argparse.ArgumentParser(
        description="S147 Q0/Q1/Q2 patches — FULL MODE ONLY (TB ruling)",
        epilog="NOTE: Q0 and Q2 are interdependent (Q0A references _hybrid_strategies "
               "introduced by Q2). Selective patch flags are intentionally not supported. "
               "Always run all patches together."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change — no files modified")
    args = parser.parse_args()
    DRY_RUN = args.dry_run

    if DRY_RUN:
        print("=== DRY RUN — no files will be modified ===\n")

    print("Applying all patches (Q2 → Q0A → Q0B → Q1)")
    print("TB ruling: full mode only — Q0 and Q2 are interdependent\n")

    # Q2 MUST run before Q0A — Q0A_NEW references _hybrid_strategies set by Q2
    print("── Q2: PWC strategy loader + forward strategies kwarg ──")
    apply_patch(path("persistent_worker_coordinator.py"), "Q2", Q2_OLD, Q2_NEW)

    print("── Q0A: PWC hybrid gate + reverse strategies kwarg ──")
    apply_patch(path("persistent_worker_coordinator.py"), "Q0A", Q0A_OLD, Q0A_NEW)

    print("── Q0B: legacy coordinator hybrid gate ──")
    apply_patch(path("window_optimizer_integration_final.py"), "Q0B", Q0B_OLD, Q0B_NEW)

    print("── Q1: agents/watcher_agent.py Step 1 timeout ──")
    apply_patch(path("agents/watcher_agent.py"), "Q1", Q1_OLD, Q1_NEW)

    ok  = sum(1 for _, s in results if s)
    bad = sum(1 for _, s in results if not s)
    print(f"\n{'DRY RUN ' if DRY_RUN else ''}Summary: {ok} applied, {bad} skipped/failed")

    if not DRY_RUN and ok > 0:
        print("""
Next — commit and dual-push:
  cd ~/distributed_prng_analysis
  git add persistent_worker_coordinator.py \\
          window_optimizer_integration_final.py \\
          agents/watcher_agent.py
  git commit -m 'fix(S147): Q0 hybrid gate + Q1 timeout + Q2 single strategy

Q0: Skip hybrid reverse when hybrid forward=0 — PWC and legacy paths.
    Skip not prune — constant-skip results preserved.
Q1: Add 1:0 to step_timeout_overrides — S145 guard fires correctly.
    Step 1 now has infinite timeout for production sweeps.
Q2: Single strategy (balanced_hybrid) for full-range hybrid scan.
    Both forward and reverse hybrid calls receive strategies kwarg.
    5x work reduction. Uses pwc.logger (standalone function).'
  git push origin main && git push public main""")

    sys.exit(0 if bad == 0 else 1)

if __name__ == "__main__":
    main()
