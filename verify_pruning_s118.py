#!/usr/bin/env python3
"""
verify_pruning_s118.py  (v2 — fixed shared-study logic)
=========================================================
S118 Pruning Correctness Verification

Confirms --enable-pruning produces IDENTICAL survivors to baseline (no pruning).

FIX vs v1:
  v1 passed --study-name "verify_s118_<ts>" to BOTH runs, assuming the resume
  logic would use that name for the fresh study. It doesn't — fresh studies are
  always named "window_opt_<timestamp>" regardless of --study-name.

  v2 approach:
    Run 1 (baseline): NO --resume-study. Creates a fresh study.
                      Parse actual study name from log output.
    Run 2 (pruned):   --resume-study --study-name <actual_name_from_run1>
                      Resumes Run 1's completed study with pruning ON.
                      Optuna replays same hyperparameters -> results must match.

USAGE:
  python3 verify_pruning_s118.py --trials 6 --max-seeds 2000000
  python3 verify_pruning_s118.py --trials 10 --max-seeds 5000000
"""

import subprocess, sys, json, os, time, argparse, shutil, re

WORK_DIR  = os.path.expanduser("~/distributed_prng_analysis")
PYTHON    = os.path.expanduser("~/venvs/torch/bin/python3")
SCRIPT    = "window_optimizer.py"
SURVIVORS = "bidirectional_survivors.json"
OUT_DIR   = "/tmp/verify_pruning_s118"

# ── Helpers ──────────────────────────────────────────────────────────────────

def run_step1(label, trials, max_seeds, lottery,
              resume_study, study_name, enable_pruning, log_path):

    cmd = [
        PYTHON, "-u", SCRIPT,
        "--strategy",     "bayesian",
        "--lottery-file", lottery,
        "--trials",       str(trials),
        "--max-seeds",    str(max_seeds),
    ]
    if resume_study:
        cmd += ["--resume-study", "--study-name", study_name]
    if enable_pruning:
        cmd.append("--enable-pruning")

    env = os.environ.copy()
    env["PYTHONPATH"] = WORK_DIR

    flag = "PRUNING=ON " if enable_pruning else "PRUNING=OFF"
    resume_tag = f"resume={study_name}" if resume_study else "fresh"
    print(f"\n{'='*60}")
    print(f"  RUN: {label.upper()}  [{flag}]  [{resume_tag}]")
    print(f"  trials={trials}  max_seeds={max_seeds:,}")
    print(f"  log -> {log_path}")
    print(f"{'='*60}")
    sys.stdout.flush()

    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, cwd=WORK_DIR, env=env,
                              stdout=f, stderr=subprocess.STDOUT)
    elapsed = round(time.time() - t0, 1)

    result = {
        "label":          label,
        "enable_pruning": enable_pruning,
        "returncode":     proc.returncode,
        "elapsed_sec":    elapsed,
        "survivor_count": 0,
        "survivors":      [],
        "prune_lines":    [],
        "study_name":     None,
    }

    # Parse actual study name from log
    with open(log_path) as f:
        log_text = f.read()
    m = re.search(r'Optuna study[^:]*:\s*optuna_studies/(window_opt_\S+?)\.db', log_text)
    if m:
        result["study_name"] = m.group(1)
        print(f"  Actual study name: {result['study_name']}")
    else:
        # Fallback: look for RESUMING line
        m2 = re.search(r'RESUMING study:\s*(window_opt_\S+)', log_text)
        if m2:
            result["study_name"] = m2.group(1)
            print(f"  Resumed study: {result['study_name']}")

    # Copy survivors file
    src = os.path.join(WORK_DIR, SURVIVORS)
    dst = os.path.join(OUT_DIR, f"survivors_{label}.json")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        with open(dst) as f:
            data = json.load(f)
        result["survivors"]      = data
        result["survivor_count"] = len(data)
    else:
        print(f"  WARNING: {SURVIVORS} not found in {WORK_DIR}")

    # Pull prune-related log lines
    for line in log_text.splitlines():
        low = line.lower()
        if any(k in low for k in ("prun", "trial pruned", "forward_count=0",
                                   "trialstate.pruned", "% prune", "✂️")):
            result["prune_lines"].append(line.rstrip())

    print(f"  rc={proc.returncode}  elapsed={elapsed}s  "
          f"survivors={result['survivor_count']:,}")
    if result["prune_lines"]:
        print(f"  Prune activity ({len(result['prune_lines'])} lines):")
        for ln in result["prune_lines"][:5]:
            print(f"    {ln}")
    else:
        print(f"  No prune log lines (expected for baseline; may mean pruning not firing on pruned run)")

    return result


def compare(baseline, pruned):
    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")

    passes, fails = [], []

    def check(name, ok, detail=""):
        sym = "PASS" if ok else "FAIL"
        mark = "✅" if ok else "❌"
        print(f"  {mark} {sym}  {name}  {detail}")
        (passes if ok else fails).append(name)

    check("rc_baseline", baseline["returncode"] == 0,
          f"(rc={baseline['returncode']})")
    check("rc_pruned",   pruned["returncode"]   == 0,
          f"(rc={pruned['returncode']})")
    check("study_name_parsed", baseline["study_name"] is not None,
          f"(study={baseline['study_name']})")
    check("pruning_fired", len(pruned["prune_lines"]) > 0,
          f"({len(pruned['prune_lines'])} prune log lines)")

    bc = baseline["survivor_count"]
    pc = pruned["survivor_count"]
    check("survivor_count_match", bc == pc,
          f"(baseline={bc:,}  pruned={pc:,})")

    if bc > 0 and pc > 0:
        b_seeds = {r["seed"] for r in baseline["survivors"]}
        p_seeds = {r["seed"] for r in pruned["survivors"]}
        only_b  = b_seeds - p_seeds
        only_p  = p_seeds - b_seeds
        match   = (b_seeds == p_seeds)
        detail  = ("" if match else
                   f"(only_baseline={len(only_b)}, only_pruned={len(only_p)})")
        check("seed_sets_identical", match, detail)
    elif bc == 0 and pc == 0:
        print(f"  ⚠️  SKIP  seed_sets — both runs produced 0 survivors.")
        print(f"            Try --max-seeds 10000000 for more productive results.")
    else:
        check("seed_sets_identical", False,
              f"(mismatch: baseline={bc}, pruned={pc})")

    speedup = baseline["elapsed_sec"] / max(pruned["elapsed_sec"], 0.1)
    print(f"\n  Timing: baseline={baseline['elapsed_sec']}s  "
          f"pruned={pruned['elapsed_sec']}s  speedup={speedup:.2f}x")

    print(f"\n{'='*60}")
    total = len(passes) + len(fails)
    if fails:
        print(f"  ❌ RESULT: FAIL  ({len(passes)}/{total} passed)")
        print(f"     Failed: {', '.join(fails)}")
        print(f"\n  DO NOT use --enable-pruning in production until fixed.")
        print(f"  Logs: {OUT_DIR}/baseline.log  {OUT_DIR}/pruned.log")
    else:
        print(f"  ✅ RESULT: PASS  ({len(passes)}/{total} passed)")
        print(f"\n  Pruning is SAFE. Production command:")
        print(f"    python3 window_optimizer.py --lottery-file daily3.json \\")
        print(f"      --trials 50 --strategy bayesian --resume-study \\")
        print(f"      --study-name window_opt_1772507547 --enable-pruning")
        if speedup >= 1.3:
            print(f"\n  Speedup on this test: {speedup:.2f}x")
    print(f"{'='*60}\n")
    return len(fails) == 0


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials",    type=int, default=6)
    ap.add_argument("--max-seeds", type=int, default=2_000_000)
    ap.add_argument("--lottery",   default="daily3.json")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  S118 PRUNING CORRECTNESS VERIFICATION  (v2)")
    print(f"  trials={args.trials}  max_seeds={args.max_seeds:,}")
    print(f"  working_dir={WORK_DIR}")
    print(f"{'#'*60}")

    # Verify wire-up fix is applied
    target = os.path.join(WORK_DIR, SCRIPT)
    with open(target) as f:
        src = f.read()
    if "enable_pruning=getattr(args, 'enable_pruning'" not in src:
        print("\n❌ STOP: S118 wire-up not applied.")
        print("   Run: python3 apply_s118_wireup.py")
        sys.exit(1)
    print("  ✅ S118 wire-up confirmed in window_optimizer.py")

    # Run 1: baseline — fresh study, no pruning
    print(f"\n  Step 1: Run baseline (fresh study, no pruning)")
    baseline = run_step1(
        label="baseline", trials=args.trials, max_seeds=args.max_seeds,
        lottery=args.lottery, resume_study=False, study_name="",
        enable_pruning=False,
        log_path=os.path.join(OUT_DIR, "baseline.log"),
    )

    if baseline["study_name"] is None:
        print("\n❌ STOP: Could not parse study name from baseline log.")
        print(f"   Check {OUT_DIR}/baseline.log for errors.")
        sys.exit(1)

    print(f"\n  Step 2: Run pruned (resume Run 1's study, pruning ON)")
    pruned = run_step1(
        label="pruned", trials=args.trials, max_seeds=args.max_seeds,
        lottery=args.lottery, resume_study=True,
        study_name=baseline["study_name"],
        enable_pruning=True,
        log_path=os.path.join(OUT_DIR, "pruned.log"),
    )

    passed = compare(baseline, pruned)

    report = {
        "session":   "S118",
        "version":   "v2",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config":    {"trials": args.trials, "max_seeds": args.max_seeds,
                      "shared_study": baseline["study_name"]},
        "baseline":  {k: v for k, v in baseline.items() if k != "survivors"},
        "pruned":    {k: v for k, v in pruned.items()   if k != "survivors"},
        "verdict":   "PASS" if passed else "FAIL",
    }
    rp = os.path.join(OUT_DIR, "report.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report:       {rp}")
    print(f"  Baseline log: {OUT_DIR}/baseline.log")
    print(f"  Pruned log:   {OUT_DIR}/pruned.log")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
