#!/usr/bin/env python3
"""PWC-BACKUP STAGE 3a — PWC-only items of the Stage 3 plan, CPU only.

  S3-13  WITHDRAWN (owner, 2026-11-20 21:32-21:35 PST: "Well range miner must be fixed
         none the less" / "Please do whatever method is the quickest and least
         recoding"). PWC keeps sending the trial's skip range, as at b2efdef; the miner
         gap (F-1) is to be fixed on the miner side. No skip-range gate here.
  F2     (Beta X-PWC-STAGE2B R1 Finding 2) a new-key job derives the anchor check and
         the window from ONE dataset read, as miner/range_miner_worker.py:904-932 does.
  S3-11  the coordinator side of the offset split already landed in Stage 2b (b2efdef);
         G-S311 re-asserts it on the wire (and that the job carries the range it is given).

Every repair has a mutant: a textual edit of the LIVE source, run in a separate module. A mutant counts as killed only if the detector passes on the live
source and fails on the mutant.

Run from the repository root:  python3 -u tests/test_pwc_stage3a.py
Writes only under logs/test_pwc_stage3a/ (ignored). No GPU, no network, no rigs.
"""
import contextlib
import copy
import importlib.util
import io
import json
import os
import subprocess
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "logs", "test_pwc_stage3a")
os.makedirs(OUT, exist_ok=True)
SF_PATH = os.path.join(ROOT, "sieve_filter.py")
BASE = os.environ.get("PWC_STAGE3A_BASE", "b2efdef")

# CuPy stand-in before anything imports sieve_filter
_cp = types.ModuleType("cupy")
class _Pool:
    def set_limit(self, *a, **k): pass
_cp.get_default_memory_pool = lambda: _Pool()
sys.modules.setdefault("cupy", _cp)

import persistent_worker_coordinator as PWC                 # noqa: E402
import sieve_filter as SF                                    # noqa: E402

RESULTS = []


def gate(name, fn):
    try:
        detail = fn()
        RESULTS.append((name, True))
        print(f"[PASS] {name}: {detail}", flush=True)
    except Exception as e:                                   # noqa: BLE001
        RESULTS.append((name, False))
        print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)


def _load_source(src, name, path):
    spec = importlib.util.spec_from_loader(name, loader=None, origin=path)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = path
    sys.modules[name] = mod
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod


def _mutate(src, old, new):
    n = src.count(old)
    assert n == 1, f"mutant anchor occurs {n} times: {old[:80]!r}"
    return src.replace(old, new)


class _FakeTransport:
    def __init__(self, provider):
        self.provider, self.jobs, self._r = provider, [], {}

    def ready_count(self):
        return 2

    def worker_count(self):
        return 2

    def submit_job(self, job):
        self.jobs.append(json.loads(json.dumps(job)))
        self._r[job["job_id"]] = {"status": "ok", "result": self.provider(job),
                                  "worker_id": "192_168_3_122_gpu0"}

    def recv_result(self, timeout_s=None, job_id=None):
        return self._r.pop(job_id, None)


def _payload(job):
    out = {"success": True, "seed_range": {"start": job["seed_start"], "end": job["seed_end"]},
           "min_match_threshold": job["min_match_threshold"], "survivors": [],
           "per_family": {job["prng_families"][0]: {"survivors": []}}}
    for k in ("window_anchor", "generator_phase"):
        if k in job:
            out[k] = job[k]
    return out


def _config_file():
    path = os.path.join(OUT, "nodes.json")
    nodes = [{"hostname": h, "gpu_type": "RX 6600 XT", "gpu_count": 8,
              "python_env": "/home/michael/rocm_env/bin/python",
              "script_path": "/home/michael/distributed_prng_analysis", "username": "michael"}
             for h in ("192.168.3.122", "192.168.3.156")]
    with open(path, "w") as f:
        json.dump({"nodes": nodes}, f)
    return path


def d_s311_jobs(M=PWC):
    """S3-11 on the wire: a TCP pass at anchor 25 sends window_anchor 25,
    generator_phase 0 and no offset, and carries the skip range it is given (the
    trial's, as at b2efdef; S3-13 withdrawn)."""
    p = M.PersistentWorkerCoordinator(config_file=_config_file(), worker_pool_size=6,
                                      seed_cap_amd=25_000, pwc_transport="tcp",
                                      pwc_coordinator_address="192.168.3.177")
    p._tcp_transport = _FakeTransport(_payload)
    p._effective_pool_by_host = {"192.168.3.122": 6}
    with contextlib.redirect_stderr(io.StringIO()):
        res = p.run_sieve_pass(prng_type="java_lcg", residues=[1], total_seeds=40_000,
                               threshold=0.71, window_size=8,
                               output_file=os.path.join(OUT, "pass.json"),
                               dataset_path="unused.json", sessions=["midday"],
                               skip_range=[6, 99], offset=25)
    jobs = p._tcp_transport.jobs
    assert jobs and all(j["window_anchor"] == 25 and j["generator_phase"] == 0
                        and "offset" not in j and j["skip_range"] == [6, 99] for j in jobs), jobs[:1]
    assert res["pwc_pass_accounting"]["complete"], res["pwc_pass_accounting"]
    return f"{len(jobs)} TCP jobs: window_anchor 25, generator_phase 0, no offset, skip_range [6, 99] as given"


# ═════════════════════════════════════════════════════════════════════════════
# F2 — one dataset read for a new-key job
# ═════════════════════════════════════════════════════════════════════════════
class _Recorder:
    def __init__(self):
        self.calls = []

    def factory(self):
        rec = self

        class FakeSieve:
            def __init__(self, gpu_id=0):
                pass

            def run_sieve(self, **kw):
                rec.calls.append(("run_sieve", copy.deepcopy(kw)))
                return {"family": kw["prng_family"], "survivors": [],
                        "seed_range": {"start": kw["seed_start"], "end": kw["seed_end"]},
                        "stats": {"seeds_tested": 1, "survivors_found": 0, "duration_ms": 1.0,
                                  "seeds_per_sec": 1.0}}
        return FakeSieve


def _rows(n, session="midday", start=0):
    return [{"draw": (start + i) * 7 % 1000, "session": session} for i in range(n)]


def run_with_reads(mod, job, contents_by_read):
    """execute_sieve_job with `open` on the dataset path answering read k with
    contents_by_read[min(k, last)]. Returns (result, sieve calls, number of reads)."""
    path = job["dataset_path"]
    reads = {"n": 0}
    real_open = open

    def fake_open(p, *a, **k):
        if os.path.abspath(p) == os.path.abspath(path):
            data = contents_by_read[min(reads["n"], len(contents_by_read) - 1)]
            reads["n"] += 1
            return io.StringIO(json.dumps(data))
        return real_open(p, *a, **k)

    rec = _Recorder()
    saved_gs, saved_open = mod.GPUSieve, mod.__dict__.get("open")
    mod.GPUSieve, mod.open = rec.factory(), fake_open
    try:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            out = mod.execute_sieve_job(copy.deepcopy(job), 0)
    finally:
        mod.GPUSieve = saved_gs
        if saved_open is None:
            del mod.open
        else:
            mod.open = saved_open
    return out, rec.calls, reads["n"]


def _job(**kw):
    j = {"job_id": "f2", "search_type": "residue_sieve",
         "dataset_path": os.path.join(OUT, "f2_dataset.json"),
         "seed_start": 0, "seed_end": 100, "window_size": 8, "min_match_threshold": 0.5,
         "skip_range": [0, 16], "sessions": ["midday"], "prng_families": ["java_lcg"],
         "strategies": None, "hybrid": False, "phase2_threshold": 0.5,
         "window_anchor": 2, "generator_phase": 0}
    j.update(kw)
    return j


def d_one_read(mod=SF):
    """Beta's fixture: ten single-session rows, width 8, anchor 2; the file shrinks to
    nine rows after the first read. One read: the window is rows[2:10] of the
    ten-row content, and it is the only read."""
    ten, nine = _rows(10), _rows(9)
    out, calls, n = run_with_reads(mod, _job(), [ten, nine])
    assert n == 1, f"the dataset was read {n} times for one new-key job"
    assert out.get("success") is True and out.get("window_anchor") == 2, out.get("error")
    want = [e["draw"] for e in ten[2:10]]
    assert calls and all(c[1]["residues"] == want for c in calls), \
        (want, [c[1]["residues"] for c in calls])
    return "one read; anchor 2 of ten rows gives rows[2:10]; a file shrunk after the read is never seen"


def d_one_read_domain(mod=SF):
    """The domain and the short-file refusal come from that same read, as in the miner."""
    out, calls, n = run_with_reads(mod, _job(window_anchor=3), [_rows(10)])
    assert n == 1 and out.get("success") is False and \
        "PWC_WINDOW_ANCHOR_OUT_OF_DOMAIN" in out.get("error", "") and calls == [], out
    out, calls, n = run_with_reads(mod, _job(window_anchor=0), [_rows(5)])
    assert n == 1 and out.get("success") is False and \
        "Dataset has only 5 entries" in out.get("error", "") and calls == [], out
    return "anchor 3 of 10 rows (w 8) refused out of domain; 5 rows < w 8 refused; one read each, no sieve"


def d_one_read_equals_miner(mod=SF):
    from miner.range_miner_worker import load_residue_window       # read-only reference
    data = []
    for i in range(60):
        e = {"draw": (i * 37 + 11) % 1000, "session": "midday" if i % 2 == 0 else "evening"}
        if i % 5 == 0:
            e["full_state"] = 100000 + i
        data.append(e)
    path = os.path.join(OUT, "f2_equiv.json")
    with open(path, "w") as f:
        json.dump(data, f)
    n_checked = 0
    for sess in (["midday", "evening"], ["midday"], ["evening"]):
        n_f = len([e for e in data if e["session"] in sess])
        for anchor in (0, 7, n_f - 8):
            out, calls, n = run_with_reads(mod, _job(dataset_path=path, sessions=sess,
                                                     window_anchor=anchor), [data])
            ref = load_residue_window(path, 8, sess, anchor)
            assert n == 1 and calls and calls[0][1]["residues"] == ref, (sess, anchor)
            n_checked += 1
    return f"{n_checked} windows (3 session sets x 3 anchors incl. the boundary) equal the miner's load_residue_window"


def d_legacy_unchanged(mod=SF):
    """Old-style jobs (legacy coordinator.py, ZMQ) still go through load_draws_from_daily3,
    its clamp included, exactly as at the base."""
    base = _load_source(subprocess.run(["git", "-C", ROOT, "show", f"{BASE}:sieve_filter.py"],
                                       check=True, capture_output=True, text=True).stdout,
                        "sieve_filter_base_3a", SF_PATH + "@BASE")
    for off in (0, 1, 5):                       # 5 is past the end of 10 rows / w 8: clamped
        j = {k: v for k, v in _job().items() if k not in ("window_anchor", "generator_phase")}
        j["offset"] = off
        ob, cb, _ = run_with_reads(base, j, [_rows(10)])
        on, cn, _ = run_with_reads(mod, j, [_rows(10)])
        assert cb == cn and json.dumps(ob, sort_keys=True) == json.dumps(on, sort_keys=True), off
    return f"old-style offsets 0/1/5 (5 clamped): sieve calls and results identical to {BASE}"


def d_one_read_base_control():
    """Positive control: the detector sees the base's two reads."""
    base = _load_source(subprocess.run(["git", "-C", ROOT, "show", f"{BASE}:sieve_filter.py"],
                                       check=True, capture_output=True, text=True).stdout,
                        "sieve_filter_base_3a_ctl", SF_PATH + "@BASE")
    out, calls, n = run_with_reads(base, _job(), [_rows(10), _rows(9)])
    assert n == 2, f"base read the file {n} times"
    return f"{BASE} reads the dataset twice for a new-key job (the case Beta reproduced)"


GATES = [
    ("G-S311-JOBS", d_s311_jobs),
    ("G-F2-ONE-READ", d_one_read),
    ("G-F2-DOMAIN", d_one_read_domain),
    ("G-F2-EQUALS-MINER", d_one_read_equals_miner),
    ("G-F2-LEGACY-UNCHANGED", d_legacy_unchanged),
    ("G-F2-BASE-CONTROL", d_one_read_base_control),
]


# ═════════════════════════════════════════════════════════════════════════════
# Mutants
# ═════════════════════════════════════════════════════════════════════════════
def _sf_mutant(old, new):
    return _load_source(_mutate(open(SF_PATH, encoding="utf-8").read(), old, new),
                        "sf_mutant_3a", SF_PATH)


def m_two_reads():
    d_one_read(_sf_mutant(
        "            draws = [int(e.get(\"full_state\", e[\"draw\"]))\n"
        "                     for e in _data[_pwc_window_anchor:_pwc_window_anchor + window_size]]\n",
        "            draws = load_draws_from_daily3(dataset_path, window_size, sessions,\n"
        "                                           _pwc_window_anchor)\n"))


def m_short_file_check_removed():
    d_one_read_domain(_sf_mutant(
        "            if len(_data) < window_size:\n                raise ValueError(\n",
        "            if False:\n                raise ValueError(\n"))


def m_window_off_by_one():
    d_one_read_equals_miner(_sf_mutant(
        "for e in _data[_pwc_window_anchor:_pwc_window_anchor + window_size]]\n",
        "for e in _data[_pwc_window_anchor + 1:_pwc_window_anchor + 1 + window_size]]\n"))


MUTANTS = [
    ("M-F2-1 second read restored", m_two_reads, d_one_read),
    ("M-F2-2 short-file refusal removed", m_short_file_check_removed, d_one_read_domain),
    ("M-F2-3 window off by one", m_window_off_by_one, d_one_read_equals_miner),
]


def main():
    for name, fn in GATES:
        gate(name, fn)
    killed = 0
    for name, mutant, detector in MUTANTS:
        try:
            detector()                                      # positive control on the live source
        except Exception as e:                              # noqa: BLE001
            print(f"[FAIL] {name}: detector fails on the LIVE source: {e}", flush=True)
            continue
        try:
            mutant()
        except AssertionError as e:
            killed += 1
            print(f"[PASS] {name} -> killed: {str(e)[:160]}", flush=True)
        except Exception as e:                              # noqa: BLE001
            print(f"[FAIL] {name}: mutant raised {type(e).__name__}, not the detector's "
                  f"AssertionError: {e}", flush=True)
        else:
            print(f"[FAIL] {name}: SURVIVED", flush=True)
    ok = sum(1 for _, r in RESULTS if r)
    print(f"GATES {ok}/{len(RESULTS)} PASS")
    print(f"MUTANTS {killed}/{len(MUTANTS)} killed")
    good = ok == len(RESULTS) and killed == len(MUTANTS)
    print(f"RESULT: {'PASS' if good else 'FAIL'}")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
