#!/usr/bin/env python3
"""gate12_cache_join_gate.py — DRAFT (X-CACHE-POLICY v5).

Joins every rig worker's startup cache record to its current-nonce, sentinel-correlated parked liveness row,
RE-READS CUPY_CACHE_DIR from the parked process, and captures that worker's nonce- and PID-matched
SESSION_SENTINEL line (with its UTC timestamp) for the start-up timing. Runs in gate12_launch.sh and Route A
run4_launch.sh right after the worker liveness gate, before the sampler and the coordinator. Exit 0 only if
every expected rig worker joins exactly; anything missing, mismatched, ambiguous or unreadable refuses (1).

  python3 scripts/gate12_cache_join_gate.py --run-nonce N --cache-tsv <LOGDIR>/cache_records_N.tsv \
      --liveness-json logs/gate12_<stamp>_liveness.json --expected-rig-slots 24 \
      --evidence-json logs/gate12_<stamp>_cache_join.json --sentinel-tsv logs/gate12_<stamp>_cache_sentinels.tsv

Join key, all literal: nonce (liveness run_nonce == record tag == --run-nonce), endpoint, worker id
("<record host>:gpu<gpu>"), gpu, pid (liveness pid == record pid == launch ACK pid); liveness row status OK.

[v5, Beta X-CACHE-POLICY R1 Q1] The parked probe is NUL-safe and byte-exact. The remote shell returns
/proc/<pid>/cmdline, /proc/<pid>/environ and the sentinel lines BASE64-encoded, each with its own exit
status, so no NUL is turned into a newline and no byte is decoded as text before it is split:
  * environ is split on NUL into entries; exactly ONE entry may start with b"CUPY_CACHE_DIR=", and its whole
    value must equal the record's path bytes (a newline or any extra byte inside the value fails);
  * cmdline is split on NUL into argv tokens; the token after "--run-nonce" must equal the nonce exactly
    (adjacent tokens, no substring or prefix match), and "--run-nonce" must appear exactly once;
  * non-UTF-8 bytes in unrelated variables are carried as bytes and never decoded.
Any remote status other than 0 refuses, even if the output looked right.
Sentinel: the lines of the parked worker's log_path containing SESSION_SENTINEL; exactly one whose JSON has
run_nonce == nonce, pid == pid and worker_id == worker; its leading ISO-8601 UTC timestamp is recorded and
start-up seconds = sentinel time - t_prep (both epoch, same CT clock).
"""
import argparse
import base64
import csv
import datetime
import json
import shlex
import subprocess
import sys
import time

MARK = "@@TFMJOIN@@"


def remote(ssh_cmd, endpoint, script, timeout=20):
    p = subprocess.run(shlex.split(ssh_cmd) + [f"michael@{endpoint}", script], capture_output=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr.decode("utf-8", "replace")


def probe_script(pid: str, log_path: str) -> str:
    """Each section: marker, name, rc, base64 payload on one line. pid digits only; log_path is a plain path."""
    def sec(name, cmd):
        return (f'out=$({cmd} | base64 -w0; exit "${{PIPESTATUS[0]}}"); rc=$?; '
                f'printf "{MARK} {name} %s %s\\n" "$rc" "$out"; ')
    return ("set -o pipefail; " + sec("cmdline", f"cat /proc/{pid}/cmdline") + sec("environ", f"cat /proc/{pid}/environ")
            + sec("sentinel", f"grep -a SESSION_SENTINEL -- {shlex.quote(log_path)}") + "exit 0")


def parse_probe(stdout: bytes):
    out = {}
    for line in stdout.decode("ascii", "replace").splitlines():
        if not line.startswith(MARK + " "):
            continue
        parts = line.split(" ")
        if len(parts) not in (3, 4):
            return None, f"malformed probe line: {line[:80]!r}"
        name, rc = parts[1], parts[2]
        payload = parts[3] if len(parts) == 4 else ""
        if name in out:
            return None, f"duplicate probe section {name}"
        try:
            out[name] = (int(rc), base64.b64decode(payload, validate=True))
        except (ValueError, base64.binascii.Error) as e:
            return None, f"undecodable {name}: {e}"
    if set(out) != {"cmdline", "environ", "sentinel"}:
        return None, f"probe sections {sorted(out)}"
    return out, None


def cache_dir_values(environ: bytes):
    ents = [e for e in environ.split(b"\0") if e]
    return [e[len(b"CUPY_CACHE_DIR="):] for e in ents if e.startswith(b"CUPY_CACHE_DIR=")]


def nonce_ok(cmdline: bytes, nonce: str):
    toks = cmdline.split(b"\0")
    if toks and toks[-1] == b"":
        toks = toks[:-1]
    idx = [i for i, t in enumerate(toks) if t == b"--run-nonce"]
    if len(idx) != 1 or idx[0] + 1 >= len(toks):
        return False, f"--run-nonce occurs {len(idx)}x or has no value"
    if toks[idx[0] + 1] != nonce.encode():
        return False, f"--run-nonce value {toks[idx[0] + 1][:60]!r} != {nonce!r}"
    return True, None


def pick_sentinel(blob: bytes, nonce: str, pid: str, worker: str):
    hits = []
    for raw in blob.split(b"\n"):
        if b"SESSION_SENTINEL" not in raw:
            continue
        line = raw.decode("utf-8", "replace")
        j = line.find("SESSION_SENTINEL {")
        if j < 0:
            continue
        try:
            rec = json.loads(line[j + len("SESSION_SENTINEL "):])
        except ValueError:
            continue
        if rec.get("run_nonce") == nonce and str(rec.get("pid")) == pid and rec.get("worker_id") == worker:
            hits.append(line)
    if len(hits) != 1:
        return None, None, f"{len(hits)} nonce/pid/worker-matched SESSION_SENTINEL lines (need exactly 1)"
    ts = hits[0].split(" ", 1)[0]
    try:
        t = datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=datetime.timezone.utc).timestamp()
    except ValueError:
        return None, None, f"sentinel timestamp {ts!r} not ISO-8601 UTC"
    return hits[0], t, None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-nonce", required=True)
    ap.add_argument("--cache-tsv", required=True)
    ap.add_argument("--liveness-json", required=True)
    ap.add_argument("--expected-rig-slots", type=int, required=True)
    ap.add_argument("--evidence-json", required=True)
    ap.add_argument("--sentinel-tsv", required=True)
    ap.add_argument("--ssh-cmd", default="ssh -n -o BatchMode=yes -o ConnectTimeout=8")
    a = ap.parse_args(argv)
    nonce = a.run_nonce
    ev = {"run_nonce": nonce, "t": time.time(), "rows": [], "local": [], "defects": []}
    sent_rows = []

    def defect(msg):
        ev["defects"].append(msg)
        print(f"[cache-join] DEFECT {msg}", file=sys.stderr)

    try:
        with open(a.cache_tsv, encoding="utf-8", newline="") as f:
            recs = list(csv.DictReader(f, delimiter="\t"))
        with open(a.liveness_json, encoding="utf-8") as f:
            live = json.load(f)
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
    keyed = {}
    for rec in recs:
        if rec.get("tag") != nonce:
            defect(f"cache record tag {rec.get('tag')!r} != {nonce!r} ({rec.get('label')})")
        k = (rec.get("endpoint"), f"{rec.get('host')}:gpu{rec.get('gpu')}")
        if k in keyed:
            defect(f"duplicate cache record for {k}")
        keyed[k] = rec
    seen = set()
    for lv in rig:
        k = (lv.get("endpoint"), lv.get("worker_id"))
        row = {"worker_id": lv.get("worker_id"), "endpoint": lv.get("endpoint"), "verdict": None}
        ev["rows"].append(row)

        def bad(v, msg):
            defect(f"{k}: {msg}")
            row["verdict"] = v

        if k in seen:
            bad("duplicate liveness", "duplicate liveness row"); continue
        seen.add(k)
        rec = keyed.get(k)
        if rec is None:
            bad("no record", "no cache record joins this liveness row"); continue
        if lv.get("status") != "OK":
            bad("liveness not OK", f"liveness status {lv.get('status')!r}"); continue
        if str(lv.get("gpu")) != rec.get("gpu"):
            bad("gpu", f"liveness gpu {lv.get('gpu')} != record gpu {rec.get('gpu')}"); continue
        pid = rec.get("pid", "")
        if str(lv.get("pid")) != pid or not pid.isdigit():
            bad("pid", f"parked pid {lv.get('pid')} != startup pid {pid}"); continue
        argv_ = lv.get("argv") or []
        if argv_.count("--run-nonce") != 1 or argv_[argv_.index("--run-nonce") + 1:argv_.index("--run-nonce") + 2] != [nonce]:
            bad("argv nonce", f"liveness argv does not carry exactly --run-nonce {nonce}"); continue
        log_path = lv.get("log_path") or ""
        if not log_path.startswith("/") or any(c in log_path for c in "\0\n"):
            bad("log_path", f"liveness log_path {log_path!r} unusable"); continue
        try:
            rc, out, err = remote(a.ssh_cmd, lv["endpoint"], probe_script(pid, log_path))
        except subprocess.TimeoutExpired:
            rc, out, err = 124, b"", "timeout"
        if rc != 0:
            bad(f"transport rc={rc}", f"parked probe failed (rc={rc} {err.strip()[:80]})"); continue
        sec, why = parse_probe(out)
        if sec is None:
            bad("probe", why); continue
        # grep exits 1 for "no line": that is an empty sentinel set (refused below), not a read failure
        failed = [n for n, (r, _) in sec.items() if r != 0 and not (n == "sentinel" and r == 1)]
        if failed:
            bad("unreadable", f"parked read failed for {failed}"); continue
        ok, why = nonce_ok(sec["cmdline"][1], nonce)
        if not ok:
            bad("cmdline nonce", why); continue
        vals = cache_dir_values(sec["environ"][1])
        row.update(pid=pid, startup_path=rec["path"], parked_CUPY_CACHE_DIR=[v.decode("utf-8", "backslashreplace") for v in vals])
        if len(vals) != 1:
            bad("env count", f"{len(vals)} CUPY_CACHE_DIR entries in the parked environment"); continue
        if vals[0] != rec["path"].encode():
            bad("env mismatch", f"parked CUPY_CACHE_DIR {vals[0][:80]!r} != startup path {rec['path']!r}"); continue
        line, t_sent, why = pick_sentinel(sec["sentinel"][1], nonce, pid, lv["worker_id"])
        if line is None:
            bad("sentinel", why); continue
        try:
            startup_s = round(t_sent - float(rec["t_prep"]), 3)
        except ValueError:
            bad("t_prep", f"t_prep {rec.get('t_prep')!r} not numeric"); continue
        row.update(sentinel_line=line, sentinel_epoch=t_sent, t_prep=float(rec["t_prep"]), startup_s=startup_s)
        sent_rows.append((lv["worker_id"], lv["endpoint"], pid, nonce, f"{t_sent:.3f}", rec["t_prep"], f"{startup_s:.3f}", line))
        row["verdict"] = "OK"
    for k in keyed:
        if k not in seen:
            defect(f"cache record {k} joins no liveness row")
    n_ok = sum(1 for r in ev["rows"] if r["verdict"] == "OK")
    ok = not ev["defects"] and n_ok == a.expected_rig_slots
    ev["verdict"] = "PASS" if ok else "REFUSE"
    try:
        with open(a.evidence_json, "x", encoding="utf-8") as f:
            json.dump(ev, f, indent=1)
        if ok:
            with open(a.sentinel_tsv, "x", encoding="utf-8") as f:
                f.write("worker_id\tendpoint\tpid\trun_nonce\tsentinel_epoch\tt_prep\tstartup_s\tsentinel_line\n")
                for r in sent_rows:
                    f.write("\t".join(r) + "\n")
    except OSError as e:
        print(f"[cache-join] FATAL: evidence not written: {e}", file=sys.stderr)
        return 1
    print(f"[cache-join] {ev['verdict']}: {n_ok} of {a.expected_rig_slots} rig workers joined; "
          f"local recorded: {len(ev['local'])}; evidence {a.evidence_json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
