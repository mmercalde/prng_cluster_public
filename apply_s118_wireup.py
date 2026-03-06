#!/usr/bin/env python3
"""
apply_s118_wireup.py
====================
Fixes 3 missing wire-ups that prevented --enable-pruning and --n-parallel
from ever reaching the pruning hook. S115 implemented the hook correctly
but the call chain had gaps that silently dropped the flags.

Gap 1: main() → run_bayesian_optimization() (window_optimizer.py)
Gap 2a: run_bayesian_optimization() → coordinator.optimize_window() (window_optimizer.py)
Gap 2b: optimize_window() → BayesianOptimization() (window_optimizer_integration_final.py)

Verified against live Zeus files at commit 6b444e1.

Usage:
    python3 apply_s118_wireup.py [--dry-run] [--repo-path PATH]
"""
import sys, shutil, argparse
from pathlib import Path
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--dry-run', action='store_true')
parser.add_argument('--repo-path', type=str, default='.')
args = parser.parse_args()
REPO    = Path(args.repo_path).resolve()
DRY_RUN = args.dry_run

G="\033[92m"; R="\033[91m"; Y="\033[93m"; B="\033[1m"; Z="\033[0m"
applied=[]; failed=[]
_backed_up = set()

def read(p):
    with open(p) as f: return f.read()

def write(p, c):
    if not DRY_RUN:
        with open(p,'w') as f: f.write(c)

def backup(path):
    if str(path) not in _backed_up and not DRY_RUN:
        ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
        bak = Path(str(path)+f'.bak_s118_{ts}')
        shutil.copy2(path, bak)
        print(f"   💾 Backup: {bak.name}")
        _backed_up.add(str(path))

def patch(pid, filepath, anchor, replacement, desc):
    path = REPO / filepath
    src  = read(path)
    if anchor not in src:
        print(f"{R}❌ FAIL{Z}  {pid}  anchor not found in {filepath}")
        print(f"         anchor[:80]: {repr(anchor[:80])}")
        failed.append((pid, desc, "anchor not found"))
        return False
    n = src.count(anchor)
    if n > 1:
        print(f"{R}❌ FAIL{Z}  {pid}  anchor appears {n} times (not unique)")
        failed.append((pid, desc, f"anchor not unique ({n}x)"))
        return False
    if replacement in src:
        print(f"{Y}⏭  SKIP{Z}  {pid}  already applied  [{desc}]")
        applied.append((pid,"SKIP")); return True
    backup(path)
    write(path, src.replace(anchor, replacement, 1))
    mode = "[DRY RUN] " if DRY_RUN else ""
    print(f"{G}✅ OK  {Z}  {mode}{pid}  [{desc}]")
    applied.append((pid,"OK")); return True

print(f"\n{B}{'='*65}{Z}")
print(f"{B}S118 WIRE-UP PATCH  (enable_pruning + n_parallel call chain){Z}")
print(f"{'  [DRY RUN]' if DRY_RUN else ''}")
print(f"{B}{'='*65}{Z}\n")

# ── G1: main() → run_bayesian_optimization() ─────────────────────────────────
patch("G1","window_optimizer.py",
"""        results = run_bayesian_optimization(
            lottery_file=args.lottery_file,
            trials=args.trials,
            output_config=args.output,
            seed_count=args.max_seeds if args.max_seeds else 10_000_000,
            prng_type=args.prng_type,
            test_both_modes=args.test_both_modes,
            resume_study=getattr(args, 'resume_study', False),
            study_name=getattr(args, 'study_name', '')  # NEW: Pass through
        )""",
"""        results = run_bayesian_optimization(
            lottery_file=args.lottery_file,
            trials=args.trials,
            output_config=args.output,
            seed_count=args.max_seeds if args.max_seeds else 10_000_000,
            prng_type=args.prng_type,
            test_both_modes=args.test_both_modes,
            resume_study=getattr(args, 'resume_study', False),
            study_name=getattr(args, 'study_name', ''),
            enable_pruning=getattr(args, 'enable_pruning', False),  # S115 wire-up
            n_parallel=getattr(args, 'n_parallel', 1)               # S115 wire-up
        )""",
"main(): forward enable_pruning + n_parallel to run_bayesian_optimization()")

# ── G2a: run_bayesian_optimization() → coordinator.optimize_window() ─────────
patch("G2a","window_optimizer.py",
"""    results = coordinator.optimize_window(
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
    )""",
"""    results = coordinator.optimize_window(
        dataset_path=lottery_file,
        seed_start=0,
        seed_count=seed_count,
        prng_base=prng_type,
        test_both_modes=test_both_modes,  # NEW: Pass through to integration layer
        strategy_name=strategy_name,
        max_iterations=trials,
        output_file='window_optimization_results.json',
        resume_study=resume_study,
        study_name=study_name,
        enable_pruning=enable_pruning,  # S115 wire-up
        n_parallel=n_parallel           # S115 wire-up
    )""",
"run_bayesian_optimization(): forward enable_pruning + n_parallel to optimize_window()")

# ── G2b: optimize_window() → BayesianOptimization() ─────────────────────────
patch("G2b","window_optimizer_integration_final.py",
"            'bayesian': BayesianOptimization(n_initial=3),",
"            'bayesian': BayesianOptimization(n_initial=3, enable_pruning=enable_pruning, n_parallel=n_parallel),  # S115 wire-up",
"optimize_window(): pass enable_pruning + n_parallel to BayesianOptimization()")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{B}{'='*65}{Z}")
ok=sum(1 for _,s in applied if s=="OK"); sk=sum(1 for _,s in applied if s=="SKIP")
for pid,s in applied:
    print(f"  {G if s=='OK' else Y}{'✅' if s=='OK' else '⏭ '}{Z}  {pid}  {s}")
for pid,desc,reason in failed:
    print(f"  {R}❌{Z}  {pid}  FAIL  →  {reason}")
print(f"\n  Applied: {ok}   Skipped: {sk}   Failed: {len(failed)}")
if failed:
    print(f"\n  {R}⚠️  {len(failed)} failure(s) — do NOT deploy{Z}")
    sys.exit(1)
elif DRY_RUN:
    print(f"\n  {Y}Dry run complete — no files written.{Z}")
    print(f"  Remove --dry-run to apply.")
else:
    print(f"\n  {G}All wire-ups applied.{Z}")
    print(f"  Test: python3 window_optimizer.py --help | grep pruning")
    sys.exit(0)
