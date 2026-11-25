#!/usr/bin/env python3
"""Save run H's full vote matrix as an NPZ (TASK_MISS_VOTE_NPZ_20261124).

Rebuilds V exactly as e2e_miss_report.main() does (its arm loop, :515-551): the first arm's
ranking via e2e_miss_report.rebuild_ranking, then e2e_miss_report.full_vote over the spec's
vote_range. Module code is imported, not copied. Every declared pool record is checked against
V (pool == V[order[:N], t - t0]) and the second arm must share rows and skips; any mismatch
fails closed. No outcome, grade file or draw value is read.

Output: np.savez(V=int16[rows, draws], seeds=int64[rows], combined_index=int64[draws]).
Row i of V is survivor row i of the Step 3 file (file order, not ranked order); column j is
combined_index[j].
"""
import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU only; set before any project import

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e2e_miss_report as MR   # noqa: E402


def main(argv=None):
    import numpy as np
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--spec", required=True, help="The miss report spec JSON (survivors, vote_range, arms).")
    p.add_argument("--out", required=True, help="Output .npz; must not exist.")
    p.add_argument("--workers", type=int, default=2, help="Forked workers for full_vote (default 2).")
    a = p.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise MR.MissReportError(f"--out {out} exists")
    t_start = time.time()
    spec = json.loads(Path(a.spec).read_text())
    sys.path.insert(0, str(MR.REPO_ROOT))
    H = MR.load_harness()
    import logging
    logging.basicConfig(level=logging.WARNING)
    import prediction_generator as pg
    import utils.survivor_loader as sl
    signatures = H.check_signatures(pg, sl)
    if MR.sha256_file(spec["survivors"]["path"]) != spec["survivors"]["sha256"]:
        raise MR.MissReportError("survivors sha256 differs from the spec")
    t0, t1 = spec["vote_range"]
    V, base_rows, base_skips, vote_seconds, checked = None, None, None, None, {}
    for arm_id, arm in spec["arms"].items():
        R0, files = None, []
        for f in arm["pool_files"]:
            rec = json.loads((Path(f["run_dir"]) / "run_record.json").read_text())
            pp = Path(f["run_dir"]) / "pools.jsonl"
            if MR.sha256_file(pp) != f["pools_sha256"] or rec["pools_jsonl"]["sha256"] != f["pools_sha256"]:
                raise MR.MissReportError(f"{pp}: sha256 differs from spec/run record")
            if rec["effective_config"]["pool_size"] != f["pool_size"]:
                raise MR.MissReportError(f"{pp}: pool size differs")
            if R0 is None:
                R0 = MR.rebuild_ranking(H, pg, rec)
            elif rec["ranking_proof"]["scores_sha256"] != R0["scores_sha256"]:
                raise MR.MissReportError(f"{pp}: ranking differs within arm {arm_id}")
            files.append((f, pp))
        if base_rows is None:
            base_rows, base_skips = R0["rows"], R0["skips"]
            ts = time.time()
            V = MR.full_vote(R0["gen"], base_rows, base_skips, t0, t1, a.workers)
            vote_seconds = time.time() - ts
        else:
            if [int(r["seed"]) for r in R0["rows"]] != [int(r["seed"]) for r in base_rows]:
                raise MR.MissReportError("arms use different survivor rows")
            if R0["skips"] != base_skips:
                raise MR.MissReportError("arms use different skips")
        n = 0
        for f, pp in files:
            N = f["pool_size"]
            for line in pp.read_text().splitlines():
                r = json.loads(line)
                t = r["target"]["combined_index"]
                if r["abstain"]:
                    raise MR.MissReportError(f"{pp} t={t}: abstention; not handled")
                if r["pool"] != V[R0["order"][:N], t - t0].astype(int).tolist():
                    raise MR.MissReportError(f"{pp} t={t}: rebuilt vote != pool")
                n += 1
        checked[arm_id] = n
    seeds = np.array([int(r["seed"]) for r in base_rows], dtype=np.int64)
    cidx = np.arange(t0, t1, dtype=np.int64)
    np.savez(out, V=V, seeds=seeds, combined_index=cidx)
    rec = {"task": "TASK_MISS_VOTE_NPZ_20261124", "spec": {"path": str(a.spec), "sha256": MR.sha256_file(a.spec)},
           "npz": {"path": str(out), "sha256": MR.sha256_file(out)},
           "V": {"shape": list(V.shape), "dtype": str(V.dtype),
                 "sha256_of_bytes": hashlib.sha256(np.ascontiguousarray(V).tobytes()).hexdigest()},
           "seeds_unique": int(len(np.unique(seeds))), "vote_range": [t0, t1],
           "skips_distinct": sorted({int(s) for s in base_skips}),
           "pool_records_checked": checked, "vote_seconds": round(vote_seconds, 1), "workers": a.workers,
           "code": {"vote_npz_sha256": MR.sha256_file(__file__), "miss_report_sha256": MR.sha256_file(MR.__file__),
                    "harness_sha256": MR.sha256_file(MR.HARNESS_PATH),
                    "prediction_generator_sha256": MR.sha256_file(MR.REPO_ROOT / "prediction_generator.py"),
                    "survivor_scorer_sha256": MR.sha256_file(MR.REPO_ROOT / "survivor_scorer.py")},
           "signatures": signatures, "timing_seconds": round(time.time() - t_start, 1),
           "peak_rss_mib": {"self": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                            "largest_child": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1)}}
    Path(str(out) + ".record.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: rec[k] for k in ("npz", "V", "pool_records_checked", "vote_seconds", "timing_seconds",
                                         "peak_rss_mib", "skips_distinct")}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except MR.MissReportError as e:
        print(f"[vote_npz] FAIL-CLOSED: {e}", file=sys.stderr, flush=True)
        sys.exit(2)
