#!/usr/bin/env python3
"""Gates for scripts/run_authorization_gate.py (G3) — A-A..A-N and three mutants.

Everything is built in a throwaway directory under data/delegated_run_gates/
(removed at the end), including a throwaway git repository per fixture for the
tracked-and-unmodified checks. The clock is injected. No rig is contacted.

Prints COMPLETION SENTINEL: PASS only when every gate ran and passed.
"""

import contextlib
import copy
import datetime
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import run_authorization_gate as LIVE                               # noqa: E402

SRC = os.path.join(REPO, "scripts", "run_authorization_gate.py")
WORK_PARENT = os.path.join(REPO, "data", "delegated_run_gates")

TZ = "America/Los_Angeles"
from zoneinfo import ZoneInfo                                       # noqa: E402


def at_local(y, mo, d, h, mi=30):
    return datetime.datetime(y, mo, d, h, mi, tzinfo=ZoneInfo(TZ)).astimezone(
        datetime.timezone.utc)


NOW = at_local(2026, 11, 13, 10)          # Friday 10:30 Pacific

AUTH = {
    "run_class": "RUN4",
    "params": {
        "window_size": 12, "window_anchor": 25, "sessions": ["midday"],
        "forward_threshold": 0.71, "reverse_threshold": 0.47,
        "skip_min": 6, "skip_max": 99, "test_both_modes": True,
        "max_seeds": 2147483648, "worker_pool_size": {"min": 24, "max": 25},
        "n_parallel": 1, "end_step": 1,
    },
    "seed_start_min": 0, "seed_end_max_exclusive": 2147483648,
    "max_launches": 5, "max_wall_clock_s": 7200,
    "window_local": {"tz": TZ, "start_hour": 9, "end_hour": 17},
    "expires": "2026-12-31",
}

LAUNCH = {
    "window_size": 12, "window_anchor": 25, "sessions": ["midday"],
    "forward_threshold": 0.71, "reverse_threshold": 0.47,
    "skip_min": 6, "skip_max": 99, "test_both_modes": True,
    "max_seeds": 2147483648, "worker_pool_size": 25,
    "n_parallel": 1, "end_step": 1,
    "seed_start": 0, "seed_end_exclusive": 2147483648,
}


def git(repo, *args):
    p = subprocess.run(["git", "-C", repo, "-c", "user.name=test",
                        "-c", "user.email=test@example.invalid", *args],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"git {args} failed: {p.stderr}")
    return p.stdout


class Repo:
    def __init__(self, G, root, auth=None, commit=True):
        self.G = G
        self.repo = tempfile.mkdtemp(prefix="rag_", dir=root)
        git(self.repo, "init", "-q")
        with open(os.path.join(self.repo, "README"), "w") as fh:
            fh.write("throwaway\n")
        git(self.repo, "add", "README")
        git(self.repo, "commit", "-q", "-m", "init")
        self.rel = LIVE.auth_rel("RUN4")    # the file is always RUN4.auth (A-L)
        self.path = os.path.join(self.repo, self.rel)
        if auth is not None or commit is not None:
            self.write(auth if auth is not None else AUTH)
            if commit:
                git(self.repo, "add", self.rel)
                git(self.repo, "commit", "-q", "-m", "auth")

    def write(self, doc):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as fh:
            json.dump(doc, fh, indent=2, sort_keys=True)
            fh.write("\n")

    def run(self, argv, now=NOW):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = self.G.main(argv, repo_root=self.repo, clock=lambda: now)
        return rc, buf.getvalue()

    def verify(self, launch=None, cls="RUN4", now=NOW):
        return self.run(["verify", "--class", cls, "--params",
                         json.dumps(LAUNCH if launch is None else launch)], now=now)

    def consume(self, nonce, launch=None, now=NOW):
        return self.run(["consume", "--class", "RUN4", "--run-nonce", nonce,
                         "--params", json.dumps(LAUNCH if launch is None else launch)],
                        now=now)


def L(**kw):
    d = copy.deepcopy(LAUNCH)
    for k, v in kw.items():
        if v is DROP:
            d.pop(k)
        else:
            d[k] = v
    return d


def A(**kw):
    d = copy.deepcopy(AUTH)
    d.update(kw)
    return d


DROP = object()


def proceed(r):
    return r[0] == 0 and "PROCEED" in r[1]


def refuse(r, *needles):
    return r[0] == 1 and "REFUSE" in r[1] and all(n in r[1] for n in needles)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def g_AA(G, root):
    r = Repo(G, root, AUTH).verify()
    return proceed(r), r[1]


def g_AB(G, root):
    r = Repo(G, root, AUTH).verify(L(forward_threshold=0.72))
    return refuse(r, "'forward_threshold'"), r[1]


def g_AC(G, root):
    r = Repo(G, root, AUTH).verify(L(generator_phase=0))
    return refuse(r, "'generator_phase' is not named"), r[1]


def g_AD(G, root):
    r = Repo(G, root, AUTH).verify(L(window_anchor=DROP))
    return refuse(r, "'window_anchor' is absent"), r[1]


def g_AE(G, root):
    rp = Repo(G, root, AUTH)
    outs = [rp.verify(L(worker_pool_size=v)) for v in (24, 25, 23, 26)]
    ok = (proceed(outs[0]) and proceed(outs[1])
          and refuse(outs[2], "'worker_pool_size'") and refuse(outs[3], "'worker_pool_size'"))
    return ok, "\n---\n".join(o[1] for o in outs)


def g_AF(G, root):
    yesterday = Repo(G, root, A(expires="2026-11-12")).verify()
    today = Repo(G, root, A(expires="2026-11-13")).verify()
    ok = refuse(yesterday, "expired") and proceed(today)
    return ok, yesterday[1] + "\n---\n" + today[1]


def g_AG(G, root):
    rp = Repo(G, root, AUTH)
    outside = rp.verify(now=at_local(2026, 11, 13, 8, 59))
    start = rp.verify(now=at_local(2026, 11, 13, 9, 0))
    end = rp.verify(now=at_local(2026, 11, 13, 17, 0))
    ok = refuse(outside, "outside window") and proceed(start) and \
        refuse(end, "outside window")
    return ok, "\n---\n".join(x[1] for x in (outside, start, end))


def g_AH(G, root):
    rp = Repo(G, root, A(max_launches=2))
    v0 = rp.verify()
    c1 = rp.consume("n-1")
    v1 = rp.verify()
    c2 = rp.consume("n-2")
    v2 = rp.verify()
    ok = (proceed(v0) and c1[0] == 0 and proceed(v1) and c2[0] == 0
          and refuse(v2, "launches exhausted"))
    return ok, "\n---\n".join(x[1] for x in (v0, c1, v1, c2, v2))


def g_AI(G, root):
    rp = Repo(G, root, AUTH)
    first = rp.consume("same-nonce")
    second = rp.consume("same-nonce")
    ok = first[0] == 0 and refuse(second, "already recorded")
    return ok, first[1] + "\n---\n" + second[1]


def g_AJ(G, root):
    auth = A(seed_start_min=100, seed_end_max_exclusive=1000)
    del auth["params"]["max_seeds"]            # isolate the range rule from A-X1
    rp = Repo(G, root, auth)
    below = rp.verify(L(seed_start=99, seed_end_exclusive=1000, max_seeds=DROP))
    past = rp.verify(L(seed_start=100, seed_end_exclusive=1001, max_seeds=DROP))
    ok = refuse(below, "seed_start 99 < seed_start_min 100") and \
        refuse(past, "seed_end_exclusive 1001 > seed_end_max_exclusive 1000")
    return ok, below[1] + "\n---\n" + past[1]


def g_AK(G, root):
    untracked = Repo(G, root, AUTH, commit=False).verify()
    rp = Repo(G, root, AUTH)
    rp.write(A(expires="2026-12-30"))          # still valid content; only HEAD differs
    modified = rp.verify()
    ok = refuse(untracked, "UNTRACKED") and refuse(modified, "MODIFIED")
    return ok, untracked[1] + "\n---\n" + modified[1]


def g_AL(G, root):
    rp = Repo(G, root, A(run_class="SWEEP"))   # file RUN4.auth, content says SWEEP
    r = rp.verify(cls="RUN4")
    return refuse(r, "run_class 'SWEEP' in the file != --class 'RUN4'"), r[1]


def g_AM(G, root):
    r = Repo(G, root, AUTH).verify(L(test_both_modes="true"))
    return refuse(r, "'test_both_modes'"), r[1]


def g_AN(G, root):
    rp = Repo(G, root, A(max_launches=1))
    os.makedirs(os.path.join(rp.repo, LIVE.RECORD_DIR_REL), exist_ok=True)
    foreign = {"class": "RUN4", "nonce": "old-auth-run", "utc": "2026-11-01T00:00:00Z",
               "auth_sha256": "0" * 64, "params": LAUNCH}
    with open(os.path.join(rp.repo, LIVE.RECORD_DIR_REL, "RUN4.jsonl"), "w") as fh:
        fh.write(json.dumps(foreign) + "\n")
    r = rp.verify()
    ok = proceed(r) and "NOT COUNTED" in r[1] and "old-auth-run" in r[1]
    return ok, r[1]


def g_AX1(G, root):
    """Beyond the brief: a stated end that disagrees with seed_start + max_seeds."""
    r = Repo(G, root, AUTH).verify(L(seed_end_exclusive=1 << 30))
    return refuse(r, "!= seed_start 0 + max_seeds 2147483648"), r[1]


GATES = [("A-A", g_AA), ("A-B", g_AB), ("A-C", g_AC), ("A-D", g_AD), ("A-E", g_AE),
         ("A-F", g_AF), ("A-G", g_AG), ("A-H", g_AH), ("A-I", g_AI), ("A-J", g_AJ),
         ("A-K", g_AK), ("A-L", g_AL), ("A-M", g_AM), ("A-N", g_AN),
         ("A-X1", g_AX1)]

MUTANTS = {
    "A-MUT-1": ("unknown parameters allowed",
                "    for name in sorted(set(launch) - set(params) - set(SEED_KEYS)):\n",
                "    for name in []:\n", ["A-C"]),
    "A-MUT-2": ("expiry check removed",
                "    if local.date() > expires:\n", "    if False:\n", ["A-F"]),
    "A-MUT-3": ("launch count check removed",
                "    if len(counted) >= doc[\"max_launches\"]:\n", "    if False:\n",
                ["A-H"]),
}


def load_mutant(name, old, new, root):
    src = open(SRC, encoding="utf-8").read()
    n = src.count(old)
    if n != 1:
        raise RuntimeError(f"{name}: mutation anchor occurs {n} times (need 1) — "
                           f"mutant failed to apply")
    path = os.path.join(root, f"run_authorization_gate_{name.replace('-', '_')}.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src.replace(old, new))
    spec = importlib.util.spec_from_file_location(f"rag_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    os.makedirs(WORK_PARENT, exist_ok=True)
    root = tempfile.mkdtemp(prefix="test_run_auth_", dir=WORK_PARENT)
    results = []
    try:
        for name, fn in GATES:
            try:
                ok, out = fn(LIVE, root)
            except Exception as exc:                                # noqa: BLE001
                ok, out = False, f"EXCEPTION {type(exc).__name__}: {exc}"
            results.append((name, ok))
            print(f"[{'PASS' if ok else 'FAIL'}] {name}")
            if not ok:
                print("    " + out.replace("\n", "\n    "))
        gate_fns = dict(GATES)
        for mname, (desc, old, new, targets) in MUTANTS.items():
            try:
                mod = load_mutant(mname, old, new, root)
                reds = {t: not gate_fns[t](mod, root)[0] for t in targets}
                killed = all(reds.values())
                detail = ", ".join(f"{t} {'red' if r else 'STILL GREEN'}"
                                   for t, r in reds.items())
            except Exception as exc:                                # noqa: BLE001
                killed, detail = False, f"ERROR {type(exc).__name__}: {exc}"
            results.append((mname, killed))
            print(f"[{'PASS' if killed else 'FAIL'}] {mname} ({desc}): {detail}")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    expected = len(GATES) + len(MUTANTS)
    passed = sum(1 for _, ok in results if ok)
    print(f"\n{passed}/{expected} gates passed")
    if len(results) == expected and passed == expected:
        print("COMPLETION SENTINEL: PASS")
        return 0
    print("COMPLETION SENTINEL: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
