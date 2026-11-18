#!/usr/bin/env python3
"""cache_footprint.py — DRAFT (X-CACHE-POLICY v5). POST-RUN measurement of the per-run CuPy cache footprint.

Run once after the counted run's workers have exited (gate12_launch.sh ends in a live `tail -f`, so this is a
separate operator step, listed in the run checklist). Read-only over SSH.

  python3 scripts/cache_footprint.py --cache-tsv logs/gate12_<stamp>_cache_freeze/cache_records_<nonce>.tsv \
      --out logs/gate12_<stamp>_cache_freeze/cache_footprint_<nonce>.json

For every row of the FROZEN records TSV it measures the row's own path (so an allowed RIG_CUPY_CACHE_ROOT
override is measured correctly):
  * apparent_bytes  = `du -s --apparent-size --block-size=1` (sum of file sizes)
  * allocated_bytes = `du -s --block-size=1`               (disk blocks actually allocated)
  * entries         = number of directory entries in the path (`find -mindepth 1 -maxdepth 1`)
and, per CT, retained_run_dirs = number of `cupy_cache_run_*` directories under the path's parent.
Both byte figures are reported and labelled; neither is substituted for the other. Exit 0 only if every
row was measured; any SSH or command failure is recorded and gives exit 1 (the JSON is still written,
marked INCOMPLETE).
"""
import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time


def measure(ssh_cmd, endpoint, path):
    q = shlex.quote(path)
    parent = shlex.quote(os.path.dirname(path))
    script = ("set -o pipefail; "
              f"a=$(du -s --apparent-size --block-size=1 -- {q} | cut -f1) || exit 11; "
              f"b=$(du -s --block-size=1 -- {q} | cut -f1) || exit 12; "
              f"n=$(find {q} -mindepth 1 -maxdepth 1 -printf . | wc -c) || exit 13; "
              f"r=$(find {parent} -mindepth 1 -maxdepth 1 -type d -name 'cupy_cache_run_*' -printf . | wc -c) || exit 14; "
              'echo "$a $b $n $r"')
    p = subprocess.run(shlex.split(ssh_cmd) + [f"michael@{endpoint}", script], capture_output=True, text=True, timeout=30)
    if p.returncode != 0:
        return None, f"rc={p.returncode} {p.stderr.strip()[:80]}"
    parts = p.stdout.split()
    if len(parts) != 4 or not all(x.isdigit() for x in parts):
        return None, f"unparsable output {p.stdout[:80]!r}"
    return dict(zip(("apparent_bytes", "allocated_bytes", "entries", "retained_run_dirs_on_ct"), map(int, parts))), None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-tsv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ssh-cmd", default="ssh -n -o BatchMode=yes -o ConnectTimeout=8")
    a = ap.parse_args(argv)
    with open(a.cache_tsv, encoding="utf-8", newline="") as f:
        recs = list(csv.DictReader(f, delimiter="\t"))
    res = {"t": time.time(), "units": {"apparent_bytes": "du --apparent-size (sum of file sizes, bytes)",
                                       "allocated_bytes": "du (allocated disk blocks, bytes)"},
           "rows": [], "per_ct": {}, "defects": []}
    for r in recs:
        m, why = measure(a.ssh_cmd, r["endpoint"], r["path"])
        row = {"label": r["label"], "endpoint": r["endpoint"], "path": r["path"]}
        if m is None:
            res["defects"].append(f"{r['label']}: {why}")
            row["status"] = "UNMEASURED"
        else:
            row.update(m, status="OK")
            res["per_ct"][r["endpoint"]] = {"retained_run_dirs": m["retained_run_dirs_on_ct"]}
        res["rows"].append(row)
    ok = not res["defects"] and len(res["rows"]) == len(recs) > 0
    res["verdict"] = "COMPLETE" if ok else "INCOMPLETE"
    if ok:
        res["totals"] = {k: sum(x[k] for x in res["rows"]) for k in ("apparent_bytes", "allocated_bytes")}
    with open(a.out, "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"[cache-footprint] {res['verdict']}: {sum(1 for x in res['rows'] if x['status'] == 'OK')} of {len(recs)} measured -> {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
