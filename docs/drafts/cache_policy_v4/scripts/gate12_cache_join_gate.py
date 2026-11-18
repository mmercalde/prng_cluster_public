#!/usr/bin/env python3
"""gate12_cache_join_gate.py — DRAFT (X-CACHE-POLICY v4). Joins every rig worker's startup cache record to
its current-nonce, sentinel-correlated parked liveness row, and RE-READS CUPY_CACHE_DIR from the parked
process. Runs in gate12_launch.sh / run4_launch.sh right after the worker liveness gate, before the sampler
and the coordinator. Exit 0 only if all expected rig workers join exactly and every parked environment
carries exactly the startup path; anything missing, mismatched, ambiguous or unreadable refuses (exit 1).

  python3 scripts/gate12_cache_join_gate.py --run-nonce N --cache-tsv <LOGDIR>/cache_records_N.tsv \
      --liveness-json logs/gate12_<stamp>_liveness.json --expected-rig-slots 24 \
      --evidence-json logs/gate12_<stamp>_cache_join.json [--ssh-cmd 'ssh -n -o BatchMode=yes ...']

Join key, all literal: nonce (liveness run_nonce == cache tag == --run-nonce, and the parked argv carries
"--run-nonce <nonce>"), endpoint (liveness endpoint == cache-record endpoint), worker identity
(liveness worker_id == "<cache host>:gpu<gpu>"), gpu, pid (liveness pid == record pid == launch ACK pid).
The liveness row must be status OK (the liveness gate's own sentinel correlation is reused, not redone).
Zeus/local rows are recorded separately as local and are not part of the rig join.
The parked re-read is read-only: `tr '\\0' '\\n' < /proc/<pid>/environ` and `/proc/<pid>/cmdline` over SSH.
"""
import argparse
import csv
import json
import shlex
import subprocess
import sys
import time


def remote(ssh_cmd, endpoint, script, timeout=20):
    p = subprocess.run(shlex.split(ssh_cmd) + [f"michael@{endpoint}", script], capture_output=True,
                       text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-nonce", required=True)
    ap.add_argument("--cache-tsv", required=True)
    ap.add_argument("--liveness-json", required=True)
    ap.add_argument("--expected-rig-slots", type=int, required=True)
    ap.add_argument("--evidence-json", required=True)
    ap.add_argument("--ssh-cmd", default="ssh -n -o BatchMode=yes -o ConnectTimeout=8")
    a = ap.parse_args(argv)
    nonce = a.run_nonce
    ev = {"run_nonce": nonce, "t": time.time(), "rows": [], "local": [], "defects": []}

    def defect(msg):
        ev["defects"].append(msg)
        print(f"[cache-join] DEFECT {msg}", file=sys.stderr)

    try:
        with open(a.cache_tsv, encoding="utf-8", newline="") as f:
            recs = list(csv.DictReader(f, delimiter="\t"))
        live = json.load(open(a.liveness_json, encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"[cache-join] FATAL: input unreadable: {e}", file=sys.stderr)
        return 1
    if live.get("run_nonce") != nonce:
        defect(f"liveness run_nonce {live.get('run_nonce')!r} != {nonce!r}")
    rig = [r for r in live.get("results", []) if not r.get("local")]
    for r in live.get("results", []):
        if r.get("local"):
            ev["local"].append({k: r.get(k) for k in ("worker_id", "pid", "status")} | {"cache": "local (policy n/a)"})
    if len(rig) != a.expected_rig_slots:
        defect(f"{len(rig)} rig liveness rows != expected {a.expected_rig_slots}")
    if len(recs) != a.expected_rig_slots:
        defect(f"{len(recs)} cache records != expected {a.expected_rig_slots}")
    for rec in recs:
        if rec.get("tag") != nonce:
            defect(f"cache record tag {rec.get('tag')!r} != {nonce!r} ({rec.get('label')})")
    keyed = {}
    for rec in recs:
        k = (rec.get("endpoint"), f"{rec.get('host')}:gpu{rec.get('gpu')}")
        if k in keyed:
            defect(f"duplicate cache record for {k}")
        keyed[k] = rec
    seen = set()
    for lv in rig:
        k = (lv.get("endpoint"), lv.get("worker_id"))
        row = {"worker_id": lv.get("worker_id"), "endpoint": lv.get("endpoint"), "verdict": None}
        ev["rows"].append(row)
        if k in seen:
            defect(f"duplicate liveness row for {k}"); row["verdict"] = "duplicate liveness"; continue
        seen.add(k)
        rec = keyed.get(k)
        if rec is None:
            defect(f"no cache record joins liveness row {k}"); row["verdict"] = "no record"; continue
        if lv.get("status") != "OK":
            defect(f"{k}: liveness status {lv.get('status')!r}"); row["verdict"] = "liveness not OK"; continue
        if str(lv.get("gpu")) != rec.get("gpu"):
            defect(f"{k}: liveness gpu {lv.get('gpu')} != record gpu {rec.get('gpu')}"); row["verdict"] = "gpu"; continue
        if str(lv.get("pid")) != rec.get("pid"):
            defect(f"{k}: parked pid {lv.get('pid')} != startup pid {rec.get('pid')}"); row["verdict"] = "pid"; continue
        argv_ = lv.get("argv") or []
        if "--run-nonce" not in argv_ or argv_[argv_.index("--run-nonce") + 1] != nonce:
            defect(f"{k}: parked argv does not carry --run-nonce {nonce}"); row["verdict"] = "argv nonce"; continue
        pid = rec["pid"]
        script = (f"cmd=$(tr '\\0' ' ' < /proc/{pid}/cmdline) || exit 7; "
                  f"case \"$cmd\" in *'--run-nonce {nonce} '*) ;; *) exit 8;; esac; "
                  f"tr '\\0' '\\n' < /proc/{pid}/environ || exit 9")
        try:
            rc, out, err = remote(a.ssh_cmd, lv["endpoint"], script)
        except subprocess.TimeoutExpired:
            rc, out, err = 124, "", "timeout"
        if rc != 0:
            defect(f"{k}: parked environment unreadable (rc={rc} {err.strip()[:80]})"); row["verdict"] = f"unreadable rc={rc}"; continue
        vals = [l[len("CUPY_CACHE_DIR="):] for l in out.splitlines() if l.startswith("CUPY_CACHE_DIR=")]
        row.update(pid=pid, startup_path=rec["path"], parked_CUPY_CACHE_DIR=vals)
        if len(vals) != 1:
            defect(f"{k}: {len(vals)} CUPY_CACHE_DIR values in the parked environment"); row["verdict"] = "env count"; continue
        if vals[0] != rec["path"]:
            defect(f"{k}: parked CUPY_CACHE_DIR {vals[0]!r} != startup path {rec['path']!r}"); row["verdict"] = "env mismatch"; continue
        row["verdict"] = "OK"
    for k in keyed:
        if k not in seen:
            defect(f"cache record {k} joins no liveness row")
    ok = not ev["defects"] and sum(1 for r in ev["rows"] if r["verdict"] == "OK") == a.expected_rig_slots
    ev["verdict"] = "PASS" if ok else "REFUSE"
    try:
        with open(a.evidence_json, "x", encoding="utf-8") as f:
            json.dump(ev, f, indent=1)
    except OSError as e:
        print(f"[cache-join] FATAL: evidence not written: {e}", file=sys.stderr)
        return 1
    print(f"[cache-join] {ev['verdict']}: {sum(1 for r in ev['rows'] if r['verdict'] == 'OK')} of "
          f"{a.expected_rig_slots} rig workers joined; local recorded: {len(ev['local'])}; evidence {a.evidence_json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
