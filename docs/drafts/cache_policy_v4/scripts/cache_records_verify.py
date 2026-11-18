#!/usr/bin/env python3
"""cache_records_verify.py — DRAFT (X-CACHE-POLICY v4). Verifies the per-worker CuPy cache records of ONE
fleet dispatch, literally, and publishes the records TSV only after complete success.

Called by scripts/launch_fleet_manual.sh after every rig dispatch has been dispositioned:

  python3 scripts/cache_records_verify.py --tag <CACHE_TAG> --expect <expect.tsv> \
      --expected-slots <N> --out <LOGDIR>/cache_records_<tag>.tsv

expect.tsv (written by the launcher, one row per expected rig slot, tab-separated, no header):
  label  whost  endpoint  gpu  path  dispatch_log

For EVERY expected slot the dispatch log must contain:
  * exactly ONE line beginning "[cache] OK " whose tokens are exactly the ten key=value fields
    tag host gpu policy path created entries t_prep pid env_CUPY_CACHE_DIR (no extras, no repeats), with
    tag == --tag, host == whost (binds the record to the endpoint of its dispatch), gpu == expected gpu,
    policy == per_run, path == expected path, created == exclusive, entries == 0, t_prep a decimal time,
    pid all digits, env_CUPY_CACHE_DIR == expected path.   ALL COMPARISONS ARE LITERAL STRING EQUALITY.
  * exactly ONE launch ACK, a WHOLE line equal to "[dispatch] started pid=<digits> gpu=<digits>", with
    gpu == expected gpu and pid == the record's pid. A diagnostic that merely contains "started pid=" is
    not an ACK.
The expect file must hold exactly --expected-slots rows with distinct (whost, gpu) and distinct (endpoint, path)
(paths are per CT, so equal paths on different CTs are expected).

Exit 0 only if every slot passes AND the TSV was written in full (tmp file, flush, fsync, os.replace).
Exit 1 on any record defect (nothing is published); exit 2 if the TSV cannot be written in full (the tmp
file is removed, nothing is published); exit 3 on usage/expectation errors. An existing --out is refused
(a tag's TSV is written once).
"""
import argparse
import os
import re
import sys

FIELDS = ("tag", "host", "gpu", "policy", "path", "created", "entries", "t_prep", "pid", "env_CUPY_CACHE_DIR")
ACK_RE = re.compile(r"\[dispatch\] started pid=([0-9]+) gpu=([0-9]+)")      # used with fullmatch only
TPREP_RE = re.compile(r"[0-9]+\.[0-9]+")                                    # fullmatch only
DIGITS_RE = re.compile(r"[0-9]+")
HEADER = ("label", "tag", "host", "endpoint", "gpu", "path", "pid", "t_prep", "env_CUPY_CACHE_DIR", "dispatch_log")


def parse_record(line: str):
    """'[cache] OK k=v k=v ...' -> dict, or (None, reason). Literal: split on single spaces, '=' once."""
    body = line[len("[cache] OK "):]
    out = {}
    for tok in body.split(" "):
        if "=" not in tok:
            return None, f"token without '=': {tok!r}"
        k, v = tok.split("=", 1)
        if k in out:
            return None, f"repeated field {k}"
        out[k] = v
    if set(out) != set(FIELDS):
        return None, f"fields {sorted(out)} != {sorted(FIELDS)}"
    return out, None


def check_slot(tag, row):
    label, whost, endpoint, gpu, path, log = row
    try:
        with open(log, encoding="utf-8", errors="strict") as f:
            lines = f.read().splitlines()
    except (OSError, UnicodeDecodeError) as e:
        return None, f"dispatch log unreadable: {e}"
    recs = [l for l in lines if l.startswith("[cache] OK ")]
    if len(recs) != 1:
        return None, f"{len(recs)} cache OK records (need exactly 1)"
    rec, why = parse_record(recs[0])
    if rec is None:
        return None, f"record malformed: {why}"
    expect = {"tag": tag, "host": whost, "gpu": gpu, "policy": "per_run", "path": path,
              "created": "exclusive", "entries": "0", "env_CUPY_CACHE_DIR": path}
    for k, v in expect.items():
        if rec[k] != v:
            return None, f"field {k}={rec[k]!r} != expected {v!r}"
    if not TPREP_RE.fullmatch(rec["t_prep"]):
        return None, f"t_prep {rec['t_prep']!r} is not a decimal time"
    if not DIGITS_RE.fullmatch(rec["pid"]):
        return None, f"pid {rec['pid']!r} is not digits"
    acks = [ACK_RE.fullmatch(l) for l in lines]
    acks = [m for m in acks if m]
    if len(acks) != 1:
        return None, f"{len(acks)} launch ACK lines (need exactly 1)"
    apid, agpu = acks[0].group(1), acks[0].group(2)
    if agpu != gpu:
        return None, f"ACK gpu={agpu} != expected {gpu}"
    if apid != rec["pid"]:
        return None, f"ACK pid={apid} != record pid={rec['pid']}"
    return (label, tag, rec["host"], endpoint, gpu, path, rec["pid"], rec["t_prep"],
            rec["env_CUPY_CACHE_DIR"], log), None


def publish(out, rows):
    """Write the whole TSV to a temp file, fsync, then rename. Raises on any failure (caller cleans up)."""
    tmp = f"{out}.tmp.{os.getpid()}"
    try:
        with open(tmp, "x", encoding="utf-8") as f:
            f.write("\t".join(HEADER) + "\n")
            for r in rows:
                f.write("\t".join(r) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out)
        dfd = os.open(os.path.dirname(os.path.abspath(out)), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--expect", required=True)
    ap.add_argument("--expected-slots", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", a.tag):
        print(f"[cache-verify] FATAL: unsafe tag {a.tag!r}", file=sys.stderr); return 3
    if os.path.lexists(a.out):
        print(f"[cache-verify] FATAL: {a.out} already exists; a tag's records are written once", file=sys.stderr); return 3
    try:
        with open(a.expect, encoding="utf-8") as f:
            exp = [l.rstrip("\n").split("\t") for l in f if l.strip()]
    except OSError as e:
        print(f"[cache-verify] FATAL: expect file unreadable: {e}", file=sys.stderr); return 3
    if any(len(r) != 6 for r in exp):
        print("[cache-verify] FATAL: expect rows must have 6 tab-separated fields", file=sys.stderr); return 3
    if len(exp) != a.expected_slots:
        print(f"[cache-verify] FATAL: {len(exp)} expected rows != --expected-slots {a.expected_slots}", file=sys.stderr); return 3
    # paths are per CT: the same path on two CTs is legitimate; the same (endpoint, path) twice is not
    if len({(r[1], r[3]) for r in exp}) != len(exp) or len({(r[2], r[4]) for r in exp}) != len(exp):
        print("[cache-verify] FATAL: duplicate (host,gpu) or (endpoint,path) in expectations", file=sys.stderr); return 3
    rows, bad = [], 0
    for r in exp:
        res, why = check_slot(a.tag, r)
        if res is None:
            bad += 1
            print(f"[cache-verify] DEFECT {r[0]}: {why} (log {r[5]})", file=sys.stderr)
        else:
            rows.append(res)
    if bad:
        print(f"[cache-verify] {len(exp) - bad} OK of {len(exp)}: NOT VERIFIED; nothing published", file=sys.stderr)
        return 1
    try:
        publish(a.out, rows)
    except BaseException as e:      # incl. KeyboardInterrupt: an interrupted write is a failure
        print(f"[cache-verify] FATAL: records TSV could not be written in full: {e!r}", file=sys.stderr)
        return 2
    print(f"[cache-verify] {len(rows)} OK of {len(exp)} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
