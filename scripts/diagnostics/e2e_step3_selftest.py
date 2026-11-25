#!/usr/bin/env python3
"""
scripts/diagnostics/e2e_step3_selftest.py -- self-test for the e2e run-copy
builder and the Step 3 harness. CPU only. SYNTHETIC FIXTURE ONLY: no run H
row is read or scored (lead task H3 item 3).

Everything is written under <worktree>/data/e2e_step3_selftest_<stamp>/
(never /tmp, CLAUDE.md rule 7). Run with the torch venv:

    ~/venvs/torch/bin/python3 -B scripts/diagnostics/e2e_step3_selftest.py [--workers 3]

Gates (each prints PASS/FAIL; exit 0 only if all pass):
  B1  builder writes the copy; record holds every parameter and sha256; label
      block == holdout[:N]; grade records carry combined indices and dates
  B2  --grade-sessions midday lists only midday draws and counts the rest
  B3  refusals, nothing created: existing out-dir, out-dir outside data/, out-dir
      under the main tree, wrong digest, dataset != train+holdout, label/grade
      overlap, label-start != 0, empty grade block, duplicate seeds, a 21-array NPZ
  H1  harness, all chunks, --workers W: rc 0, every row through, no guard hit,
      torch saw no GPU, RSS and wall time recorded
  H2  --workers 1 gives the same aggregate as H1 (timestamps stripped)
  H3  PARITY: the launcher's own AGGREGATE_EOF heredoc, run on H1's chunk files,
      writes a byte-identical survivors_with_scores.json (TODO-E2E-3)
  H4  --chunks 0: PARTIAL output name, only chunk 0's rows
  H5  refusals: grade block as holdout, tampered input, out-dir in the main tree,
      --gpu-ids with --device cpu, out-dir exists
  H6  --min-features above the real count: rc 1, FAILED_ output, no clean output
  M1-M4  chunk mutants (production fallbacks forced): batch_raise,
      torch_generate_raise, holdout_quality_raise, seed_raise. Each: the mutant
      applied, the mutated function was called, the run failed, and the failure
      names the credited check. Then the same mutant with its --allow-* flag
      passes, which proves the credited check (not another) stopped it.
  M5  agg_drop_row and agg_drop_feature: harness_checks raise
  M6  agg_sort_ascending: the parity compare of H3 detects the mismatch
  W   no-write: listed main-tree / run H / worktree Step 3 paths unchanged,
      `git status --porcelain` unchanged, no new __pycache__ in the worktree
"""
from __future__ import annotations

import sys as _sys
_sys.dont_write_bytecode = True

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
BUILDER = HERE / "e2e_build_run_copy.py"
HARNESS = HERE / "e2e_step3_harness.py"
LAUNCHER = WT / "run_step3_full_scoring.sh"
MAIN = Path("~/distributed_prng_analysis").expanduser()
RUNH = Path("~/skipsw_run_thr3030").expanduser()
PY = sys.executable
MUTANT_ENV = "E2E_STEP3_SELFTEST_MUTANT"

WATCHED = [
    MAIN / "scoring_chunks", MAIN / "full_scoring_results", MAIN / "survivors_with_scores.json",
    MAIN / "scoring_jobs.json", Path("/dev/shm/prng"),
    RUNH / "scoring_chunks", RUNH / "full_scoring_results", RUNH / "survivors_with_scores.json",
    WT / "scoring_chunks", WT / "full_scoring_results", WT / "survivors_with_scores.json",
    WT / "scoring_jobs.json",
]

RESULTS = []


def gate(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def snapshot():
    out = {}
    for root in WATCHED:
        if not root.exists():
            out[str(root)] = None
        elif root.is_file():
            st = root.stat()
            out[str(root)] = (st.st_size, st.st_mtime_ns)
        else:
            rows = []
            for dp, dn, fn in os.walk(root):
                for n in sorted(dn + fn):
                    st = (Path(dp) / n).lstat()
                    rows.append((str(Path(dp) / n), st.st_size, st.st_mtime_ns))
            out[str(root)] = sorted(rows)
    return out


def pycache_dirs():
    return sorted(str(p) for p in WT.rglob("__pycache__") if "data" not in p.relative_to(WT).parts)


def git_status():
    """Every tracked change and every untracked file outside data/ (the
    designated output area, CLAUDE.md rule 7), listed file by file."""
    out = subprocess.run(["git", "-C", str(WT), "status", "--porcelain", "--untracked-files=all"],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    return sorted(l for l in out if not l[3:].startswith("data/"))


# ---------------------------------------------------------------------------
# synthetic fixture (no run H data)
# ---------------------------------------------------------------------------
CONTRACT = [("seeds", "uint32"), ("forward_matches", "float32"), ("reverse_matches", "float32"),
            ("window_size", "int32"), ("offset", "int32"), ("trial_number", "int32"),
            ("skip_min", "int32"), ("skip_max", "int32"), ("skip_range", "int32"),
            ("forward_count", "float32"), ("reverse_count", "float32"),
            ("bidirectional_count", "float32"), ("intersection_count", "float32"),
            ("intersection_ratio", "float32"), ("intersection_weight", "float32"),
            ("bidirectional_selectivity", "float32"), ("forward_only_count", "float32"),
            ("reverse_only_count", "float32"), ("survivor_overlap_ratio", "float32"),
            ("score", "float32"), ("skip_mode", "uint8"), ("prng_type", "uint8")]


def make_fixture(d: Path, rows: int, n_train: int, n_hold: int, seed: int = 20261123,
                 drop_array: str = None, dup_seed: bool = False) -> dict:
    d.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    seeds = rng.choice(2**32 - 1, size=rows, replace=False).astype(np.uint32)
    if dup_seed:
        seeds[1] = seeds[0]
    cols = {
        "seeds": seeds,
        "forward_matches": rng.uniform(0.3, 0.7, rows), "reverse_matches": rng.uniform(0.3, 0.6, rows),
        "window_size": np.full(rows, 12), "offset": np.full(rows, 25),
        "trial_number": np.zeros(rows), "skip_min": np.zeros(rows),
        "skip_max": np.full(rows, 16), "skip_range": np.full(rows, 16),
        "forward_count": rng.integers(1, 5, rows), "reverse_count": rng.integers(1, 5, rows),
        "bidirectional_count": np.full(rows, rows), "intersection_count": np.full(rows, rows),
        "intersection_ratio": rng.uniform(0, 1, rows), "intersection_weight": rng.uniform(0, 1, rows),
        "bidirectional_selectivity": rng.uniform(0, 1, rows),
        "forward_only_count": rng.integers(0, 3, rows), "reverse_only_count": rng.integers(0, 3, rows),
        "survivor_overlap_ratio": rng.uniform(0, 1, rows), "score": rng.uniform(0, 1, rows),
        "skip_mode": rng.integers(0, 2, rows),        # constant and variable rows both present
        "prng_type": np.ones(rows),
    }
    arrays = {k: np.asarray(cols[k]).astype(dt) for k, dt in CONTRACT if k != drop_array}
    npz = d / "survivors.npz"
    np.savez(npz, **arrays)
    draws = rng.integers(0, 1000, n_train + n_hold).tolist()
    start = _dt.date(2001, 1, 1)
    dataset = [{"date": (start + _dt.timedelta(days=i // 2)).isoformat(),
                "session": "midday" if i % 2 == 0 else "evening", "draw": int(v)}
               for i, v in enumerate(draws)]
    (d / "daily3.json").write_text(json.dumps(dataset, indent=2))
    (d / "train_history.json").write_text(json.dumps(draws[:n_train], indent=2))
    (d / "holdout_history.json").write_text(json.dumps(draws[n_train:], indent=2))
    (d / "optimal_scorer_config.json").write_text(json.dumps({"best_value": 0.0, "params": {}}))
    return {"dir": d, "npz": npz, "digest": sha(npz), "train": d / "train_history.json",
            "hold": d / "holdout_history.json", "dataset": d / "daily3.json",
            "config": d / "optimal_scorer_config.json", "draws": draws, "n_train": n_train}


def run_builder(fx, out_dir: Path, *extra, digest=None, label_count="60", sessions="all", dataset=None,
                npz=None):
    cmd = [PY, "-B", str(BUILDER), "--npz", str(npz or fx["npz"]),
           "--generation-digest", digest or fx["digest"], "--train-history", str(fx["train"]),
           "--holdout-history", str(fx["hold"]), "--dataset", str(dataset or fx["dataset"]),
           "--config", str(fx["config"]), "--label-start", "0", "--label-count", label_count,
           "--grade-sessions", sessions, "--out-dir", str(out_dir), *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


def run_harness(copy: Path, out_dir: Path, *extra, mutant=None):
    env = dict(os.environ)
    env.pop(MUTANT_ENV, None)
    if mutant:
        env[MUTANT_ENV] = mutant
    cmd = [PY, "-B", str(HARNESS), "--run-copy", str(copy), "--out-dir", str(out_dir),
           "--config", "configs/optimal_scorer_config.json", "--chunk-size", "40", *extra]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    p.seconds = time.time() - t0
    rec_p = out_dir / "run_record.json"
    p.record = json.loads(rec_p.read_text()) if rec_p.exists() else None
    return p


def strip_ts(rows):
    for r in rows:
        r.get("metadata", {}).pop("timestamp", None)
    return rows


def launcher_heredoc() -> tuple:
    lines = LAUNCHER.read_text().splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if "<< 'AGGREGATE_EOF'" in l)
    end = next(i for i in range(start + 1, len(lines)) if lines[i].strip() == "AGGREGATE_EOF")
    return "".join(lines[start + 1:end]), (start + 1, end + 1)


def parity(h_out: Path, rec: dict, train: Path, work: Path) -> tuple:
    """Run the launcher's own aggregation text on the harness's chunk files."""
    code, span = launcher_heredoc()
    work.mkdir()
    src = h_out / rec["results_dir"]
    dst = work / "full_scoring_results" / src.name
    dst.mkdir(parents=True)
    for f in sorted(src.glob("chunk_*.json")):
        shutil.copyfile(f, dst / f.name)
    env = dict(os.environ, TRAIN_HISTORY=str(train), PYTHONPATH=str(WT), PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([PY, "-B", "-"], input=code, cwd=work, capture_output=True, text=True, env=env)
    out = work / "survivors_with_scores.json"
    ran = p.returncode == 0 and "AGGREGATION COMPLETE" in p.stdout and out.exists()
    same = ran and sha(out) == sha(h_out / rec["output"]["name"])
    return ran, same, span, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--rows", type=int, default=100)
    args = ap.parse_args()

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = WT / "data" / f"e2e_step3_selftest_{stamp}"
    base.mkdir(parents=True)
    print(f"self-test dir: {base}", flush=True)
    snap0, git0, pyc0 = snapshot(), git_status(), pycache_dirs()
    t_all = time.time()

    fx = make_fixture(base / "fixture", rows=args.rows, n_train=300, n_hold=120)

    # ---------------- builder ----------------
    copy = base / "copy"
    p = run_builder(fx, copy, "--dataset-digest", sha(fx["dataset"]), label_count="60")
    rec = json.loads((copy / "run_record.json").read_text()) if p.returncode == 0 else {}
    ok = p.returncode == 0
    if ok:
        lb = json.loads((copy / "label_block_history.json").read_text())
        gr = json.loads((copy / "grade_block_records.json").read_text())
        outs = rec["outputs"]
        ok = (lb == fx["draws"][300:360] and len(gr) == 60
              and gr[0]["combined_index"] == 360 and gr[0]["draw"] == fx["draws"][360]
              and all(outs[k] == sha(copy / k) for k in outs)
              and rec["inputs"]["npz"]["sha256"] == fx["digest"]
              and rec["declared_parameters"]["label_count"] == 60
              and rec["blocks"]["label_block"]["combined_index"] == [300, 360]
              and not (copy / "holdout_history.json").exists())
    gate("B1 builder copy + record", ok, p.stderr.strip()[-300:])

    copy_m = base / "copy_midday"
    p = run_builder(fx, copy_m, sessions="midday")
    ok = p.returncode == 0
    if ok:
        gr = json.loads((copy_m / "grade_block_records.json").read_text())
        r = json.loads((copy_m / "run_record.json").read_text())["blocks"]["grade_block"]
        ok = (all(g["session"] == "midday" for g in gr) and len(gr) == 30
              and r["count_excluded_by_session_filter"] == 30 and r["count_before_session_filter"] == 60)
    gate("B2 grade session filter", ok)

    bad_ds = base / "fixture" / "daily3_bad.json"
    ds = json.loads(fx["dataset"].read_text())
    ds[5]["draw"] = (ds[5]["draw"] + 1) % 1000
    bad_ds.write_text(json.dumps(ds))
    fx_dup = make_fixture(base / "fixture_dup", rows=20, n_train=300, n_hold=120, dup_seed=True)
    fx_21 = make_fixture(base / "fixture_21", rows=20, n_train=300, n_hold=120, drop_array="score")
    cases = {
        "existing out-dir": run_builder(fx, copy),
        "out-dir outside data/": run_builder(fx, WT / "e2e_should_not_exist"),
        "out-dir under main tree": run_builder(fx, MAIN / "data" / "e2e_should_not_exist"),
        "wrong digest": run_builder(fx, base / "x1", digest="0" * 64),
        "wrong dataset digest": run_builder(fx, base / "x9", "--dataset-digest", "f" * 64),
        "dataset mismatch": run_builder(fx, base / "x2", dataset=bad_ds),
        "label/grade overlap": run_builder(fx, base / "x3", "--grade-start", "10"),
        "label-start != 0": subprocess.run([PY, "-B", str(BUILDER), "--npz", str(fx["npz"]),
                                            "--generation-digest", fx["digest"], "--train-history",
                                            str(fx["train"]), "--holdout-history", str(fx["hold"]),
                                            "--dataset", str(fx["dataset"]), "--label-start", "5",
                                            "--label-count", "10", "--grade-sessions", "all",
                                            "--out-dir", str(base / "x4")],
                                           capture_output=True, text=True),
        "empty grade block": run_builder(fx, base / "x5", label_count="120"),
        "duplicate seeds": run_builder(fx_dup, base / "x6"),
        "21-array NPZ": run_builder(fx_21, base / "x7"),
    }
    created = [str(q) for q in [WT / "e2e_should_not_exist", MAIN / "data" / "e2e_should_not_exist"]
               + [base / f"x{i}" for i in (1, 2, 3, 4, 5, 6, 7, 9)] if q.exists()]
    detail = {k: (v.returncode, "REFUSED" in v.stderr) for k, v in cases.items()}
    gate("B3 builder refusals", all(v.returncode == 2 and "REFUSED" in v.stderr for v in cases.values())
         and not created, f"{detail} created={created}")
    p = run_builder(fx_dup, base / "x8", "--allow-duplicate-seeds")
    gate("B3b duplicate seeds allowed by flag and recorded", p.returncode == 0 and json.loads(
        (base / "x8" / "run_record.json").read_text())["inputs"]["npz"]["distinct_seeds"] == 19)

    # ---------------- harness ----------------
    h1 = base / "h1"
    p1 = run_harness(copy, h1, "--workers", str(args.workers))
    r1 = p1.record or {}
    ok = (p1.returncode == 0 and r1.get("status") == "ok" and not r1.get("partial")
          and r1["checks"]["rows"] == args.rows and r1["effective"]["chunks_total"] == -(-args.rows // 40)
          and r1["guards_parent"]["socket_attempts"] == 0 and r1["guards_parent"]["subprocess_attempts"] == 0
          and all(c["guards"]["socket_attempts"] == 0 and c["guards"]["subprocess_attempts"] == 0
                  and c["torch_cuda_available"] is False and c["status"] == "ok" for c in r1["chunks"])
          and r1["peak_rss_kib"]["chunk_process_max"] > 0 and r1["wall_seconds"] > 0
          and len({c["pid"] for c in r1["chunks"]}) == len(r1["chunks"])
          and (h1 / "survivors_with_scores.json").exists())
    gate("H1 harness all chunks", ok, (p1.stderr or "")[-500:] if not ok else
         f"rows {r1['checks']['rows']} features {r1['checks']['feature_count']} "
         f"wall {r1['wall_seconds']}s chunk-RSS {r1['peak_rss_kib']['chunk_process_max']} KiB")

    h2 = base / "h2"
    p2 = run_harness(copy, h2, "--workers", "1")
    ok = p2.returncode == 0 and p1.returncode == 0
    if ok:
        a = strip_ts(json.loads((h1 / "survivors_with_scores.json").read_text()))
        b = strip_ts(json.loads((h2 / "survivors_with_scores.json").read_text()))
        ok = a == b
    gate("H2 workers 1 == workers W", ok)

    if p1.returncode == 0:
        ran, same, span, pp = parity(h1, r1, copy / "train_history.json", base / "parity_h1")
        gate("H3 parity with launcher AGGREGATE_EOF", ran and same,
             f"heredoc lines {span[0]}-{span[1]} of run_step3_full_scoring.sh; ran={ran}"
             + ("" if ran else f" stderr={pp.stderr[-300:]}"))
    else:
        gate("H3 parity with launcher AGGREGATE_EOF", False, "H1 failed")

    h4 = base / "h4"
    p4 = run_harness(copy, h4, "--chunks", "0")
    r4 = p4.record or {}
    gate("H4 --chunks 0 is PARTIAL", p4.returncode == 0 and r4.get("partial") is True
         and (h4 / "survivors_with_scores.PARTIAL.json").exists()
         and not (h4 / "survivors_with_scores.json").exists() and r4["checks"]["rows"] == 40)

    tampered = base / "copy_tampered"
    shutil.copytree(copy, tampered)
    t = json.loads((tampered / "train_history.json").read_text())
    t[0] = (t[0] + 1) % 1000
    (tampered / "train_history.json").write_text(json.dumps(t, indent=2))
    hcases = {
        "grade block as holdout": run_harness(copy, base / "y1", "--holdout-history", "grade_block_records.json"),
        "tampered input": run_harness(tampered, base / "y2"),
        "out-dir in main tree": run_harness(copy, MAIN / "data" / "e2e_should_not_exist"),
        "gpu-ids with cpu": run_harness(copy, base / "y3", "--gpu-ids", "0"),
        "out-dir exists": run_harness(copy, h1),
    }
    created = [str(q) for q in [base / "y1", base / "y2", base / "y3", MAIN / "data" / "e2e_should_not_exist"]
               if q.exists()]
    gate("H5 harness refusals", all(v.returncode == 2 and "REFUSED" in v.stderr for v in hcases.values())
         and not created, f"{ {k: v.returncode for k, v in hcases.items()} } created={created}")

    h6 = base / "h6"
    p6 = run_harness(copy, h6, "--min-features", "100000")
    r6 = p6.record or {}
    gate("H6 Phase 6 floor fails closed", p6.returncode == 1 and r6.get("status") == "failed"
         and "Phase 6" in (r6.get("aggregate_error") or "")
         and (h6 / "FAILED_survivors_with_scores.json").exists()
         and not (h6 / "survivors_with_scores.json").exists())

    # ---------------- mutants ----------------
    credited = {
        "batch_raise": ("sequential fallback", "--allow-sequential-fallback"),
        "torch_generate_raise": ("CPU generate fallback", "--allow-cpu-generate-fallback"),
        "holdout_quality_raise": ("holdout_quality failed", "--allow-holdout-quality-failure"),
        "seed_raise": ("seed errors", "--allow-seed-errors"),
    }
    for i, (m, (needle, flag)) in enumerate(credited.items(), 1):
        mo = base / f"m{i}"
        pm = run_harness(copy, mo, "--chunks", "0", mutant=m)
        rm = pm.record or {}
        c = (rm.get("chunks") or [{}])[0]
        caught = (pm.returncode == 1 and rm.get("status") == "failed" and rm.get("selftest_mutant") == m
                  and c.get("mutant") == m and c.get("mutant_calls", 0) > 0
                  and needle in (c.get("error") or "")
                  and (mo / rm["results_dir"] / "FAILED_chunk_0000.json").exists()
                  and not (mo / "survivors_with_scores.PARTIAL.json").exists())
        extra = [flag]
        if m == "seed_raise":
            # every row errors -> the batch fallback is also taken (by design of the mutant)
            extra.append("--allow-sequential-fallback")
        mo2 = base / f"m{i}_allowed"
        pa = run_harness(copy, mo2, "--chunks", "0", *extra, mutant=m)
        ra = pa.record or {}
        ca = (ra.get("chunks") or [{}])[0]
        if m == "seed_raise":
            # every row is an error record with no per-seed features -> the aggregate must still fail
            ae = ra.get("aggregate_error") or ""
            allowed_ok = ca.get("status") == "ok" and ra.get("status") == "failed" \
                and ("Phase 6" in ae or "feature key sets" in ae)
        else:
            allowed_ok = pa.returncode == 0 and ca.get("status") == "ok" and ca.get("mutant_calls", 0) > 0
        gate(f"M{i} mutant {m}", caught and allowed_ok,
             f"calls={c.get('mutant_calls')} error={str(c.get('error'))[:90]!r} flags={c.get('flags')} "
             f"allowed: rc={pa.returncode} chunk={ca.get('status')} agg_err={str(ra.get('aggregate_error'))[:70]!r}")

    for m in ("agg_drop_row", "agg_drop_feature"):
        mo = base / f"m5_{m}"
        pm = run_harness(copy, mo, "--chunks", "0", mutant=m)
        rm = pm.record or {}
        needle = "row count" if m == "agg_drop_row" else "feature key sets"
        gate(f"M5 mutant {m}", pm.returncode == 1 and rm.get("selftest_mutant_parent_applied") is True
             and needle in (rm.get("aggregate_error") or ""), str(rm.get("aggregate_error"))[:120])

    mo = base / "m6"
    pm = run_harness(copy, mo, mutant="agg_sort_ascending")
    rm = pm.record or {}
    if pm.returncode == 0:
        ran, same, _span, _pp = parity(mo, rm, copy / "train_history.json", base / "parity_m6")
        gate("M6 mutant agg_sort_ascending caught by parity", ran and not same
             and rm.get("selftest_mutant_parent_applied") is True)
    else:
        gate("M6 mutant agg_sort_ascending caught by parity", False, pm.stderr[-300:])

    # ---------------- no-write ----------------
    snap1, git1, pyc1 = snapshot(), git_status(), pycache_dirs()
    changed = [k for k in snap0 if snap0[k] != snap1[k]]
    gate("W no writes outside the self-test dir", not changed and git0 == git1 and pyc0 == pyc1,
         f"changed={changed} git_diff={sorted(set(git0) ^ set(git1))} new_pycache={sorted(set(pyc1) - set(pyc0))}")

    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} gates passed; total {time.time() - t_all:.1f}s; dir {base}",
          flush=True)
    summary = {"gates": RESULTS, "seconds": time.time() - t_all, "workers": args.workers,
               "rows": args.rows, "h1_record": {k: r1.get(k) for k in
                                                ("wall_seconds", "eval_seconds", "peak_rss_kib", "checks")}}
    (base / "selftest_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
