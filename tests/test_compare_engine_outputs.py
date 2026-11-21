#!/usr/bin/env python3
"""test_compare_engine_outputs.py — scripts/compare_engine_outputs.py on small synthetic
bundles. CPU only; no engine code, no GPU, no rig. Files go under <root>/logs/ (ignored).

Run from the repository root:   python3 -u tests/test_compare_engine_outputs.py

Clean controls: two bundles that differ only in fields F3 allows to differ (engine name,
run id, timestamps, provenance, artifact sha) must compare SAME, exit 0.
Mutants (each is ONE change to the B bundle, applied to a fresh copy of the clean pair;
each must be caught, with exit 1, by the named field and only by expected fields):
  M1 one changed survivor            M2 one changed match rate
  M3 two NPZ arrays in swapped order M4 one NPZ array missing
  plus: one changed NPZ value, one changed credit interval, one changed bidirectional
  set, one changed phase count, an unclassified top-level key, rows reordered inside an
  NPZ array, and inputs that differ (exit 2: not the same job).
"""
import contextlib
import copy
import io
import json
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
OUT = os.path.join(ROOT, "logs", "test_compare_engine_outputs")
TOOL = os.path.join(ROOT, "scripts", "compare_engine_outputs.py")

import importlib.util                                                   # noqa: E402
_spec = importlib.util.spec_from_file_location("compare_engine_outputs", TOOL)
CEO = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CEO)

N = 4


def arrays():
    rng = np.random.default_rng(7)
    out = {}
    for name, dt in CEO.CANONICAL_ARRAY_CONTRACT:
        if dt.kind == "u":
            out[name] = np.arange(10, 10 + N, dtype=dt)
        elif dt.kind == "i":
            out[name] = np.full(N, 8, dtype=dt)
        else:
            out[name] = rng.random(N).astype(dt)
    return out


def bundle_json(engine):
    return {
        "schema": CEO.SCHEMA,
        "engine": engine, "run_id": f"run-{engine}", "created_at": f"2026-11-20T00:00:{len(engine):02d}Z",
        "provenance": {"engine": engine}, "artifact_sha256": engine * 4,
        "inputs": {"prng_base": "java_lcg", "seed_start": 2147483648, "seed_end": 2147483648 + 1000,
                   "window_size": 8, "window_anchor": 0, "generator_phase": 0, "sessions": ["evening"],
                   "dataset_sha256": "ab" * 32, "forward_threshold": 0.5, "reverse_threshold": 0.5,
                   "constant_skip_range": [0, 16], "strategies": ["s1", "s2", "s3", "s4", "s5"],
                   "test_both_modes": True},
        "phases": {"1": {"seeds": [10, 11, 12, 13], "match_rates": [0.5, 0.625, 0.75, 0.5]},
                   "2": {"seeds": [11, 12, 20], "match_rates": [0.5, 0.875, 0.5]},
                   "3": {"seeds": [12], "match_rates": [0.625]},
                   "4": {"seeds": [12, 30], "match_rates": [0.75, 0.5]}},
        "bidirectional": {"constant": [11, 12], "variable": [12]},
        "phase_counts": {"1": {"survivors": 4}, "2": {"survivors": 3}, "3": {"survivors": 1},
                         "4": {"survivors": 2}},
        "credit": {"granted": True, "swept_intervals": {p: [2147483648, 2147483648 + 1000]
                                                        for p in ("1", "2", "3", "4")}},
        "npz": "survivors.npz",
    }


def write(dirname, bj, arrs, order=None):
    d = os.path.join(OUT, dirname)
    os.makedirs(d, exist_ok=True)
    order = order or [n for n, _ in CEO.CANONICAL_ARRAY_CONTRACT]
    # np.savez keeps keyword order in the zip, which is the order the tool checks
    np.savez(os.path.join(d, "survivors.npz"), **{n: arrs[n] for n in order if n in arrs})
    p = os.path.join(d, "bundle.json")
    with open(p, "w") as f:
        json.dump(bj, f)
    return p


def run(pa, pb):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = CEO.main([pa, pb])
    rep = CEO.compare(CEO.load_bundle(pa), CEO.load_bundle(pb))
    return rc, rep, buf.getvalue()


def differing(rep):
    return sorted(r["field"] for r in rep["outputs"] if not r["equal"])


def clean_pair(tag):
    a = write(f"{tag}_A", bundle_json("range_miner"), arrays())
    return a, bundle_json("pwc"), arrays()


def g_clean():
    a, bj, arrs = clean_pair("clean")
    b = write("clean_B", bj, arrs)
    rc, rep, txt = run(a, b)
    assert rc == 0 and rep["same"] and rep["comparable"], (rc, differing(rep), txt[-400:])
    assert set(rep["ignored"]) >= {"engine", "run_id", "created_at", "provenance", "artifact_sha256"}, rep["ignored"]
    assert "VERDICT: SAME" in txt
    n_npz = sum(1 for r in rep["outputs"] if r["field"].startswith("npz.") and "order" not in r["field"])
    assert n_npz == 22, n_npz
    return f"clean pair (only allowed fields differ): exit 0, SAME; all 22 arrays compared"


def mutant(tag, mutate, expect_fields, expect_rc=1):
    """Apply ONE change to B. The clean pair is the positive control (it must be SAME)."""
    g_clean()
    a, bj, arrs = clean_pair(tag)
    order = [n for n, _ in CEO.CANONICAL_ARRAY_CONTRACT]
    bj, arrs, order = mutate(copy.deepcopy(bj), {k: v.copy() for k, v in arrs.items()}, order[:])
    b = write(f"{tag}_B", bj, arrs, order)
    rc, rep, txt = run(a, b)
    got = differing(rep)
    assert rc == expect_rc, f"{tag}: exit {rc}, want {expect_rc}; differing={got}"
    for f in expect_fields:
        assert f in got, f"{tag}: {f} not reported; differing={got}"
    return f"caught (exit {rc}): {', '.join(got) or 'inputs'}"


def m1_survivor(bj, arrs, order):
    bj["phases"]["3"]["seeds"][0] = 99                    # one changed survivor
    return bj, arrs, order


def m2_match_rate(bj, arrs, order):
    bj["phases"]["1"]["match_rates"][2] = 0.7500001      # one changed match rate
    return bj, arrs, order


def m3_reordered_array(bj, arrs, order):
    i, j = order.index("forward_matches"), order.index("reverse_matches")
    order[i], order[j] = order[j], order[i]               # same data, swapped file order
    return bj, arrs, order


def m4_missing_array(bj, arrs, order):
    del arrs["score"]
    return bj, arrs, order


def m5_npz_value(bj, arrs, order):
    arrs["forward_matches"][1] = np.float32(arrs["forward_matches"][1] + 1e-3)
    return bj, arrs, order


def m6_credit(bj, arrs, order):
    bj["credit"]["swept_intervals"]["4"][1] -= 1
    return bj, arrs, order


def m7_bidirectional(bj, arrs, order):
    bj["bidirectional"]["constant"] = [11]
    return bj, arrs, order


def m8_phase_count(bj, arrs, order):
    bj["phase_counts"]["2"]["survivors"] = 4
    return bj, arrs, order


def m9_unclassified(bj, arrs, order):
    bj["new_field_nobody_classified"] = 1
    return bj, arrs, order


def m10_rows_reordered(bj, arrs, order):
    arrs["seeds"] = arrs["seeds"][::-1].copy()            # rows reordered inside one array
    return bj, arrs, order


def m11_inputs(bj, arrs, order):
    bj["inputs"]["constant_skip_range"] = [6, 99]         # not the same job
    return bj, arrs, order


def g_subprocess_exit():
    a, bj, arrs = clean_pair("sub")
    b = write("sub_B", bj, arrs)
    r = subprocess.run([sys.executable, TOOL, a, b], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "VERDICT: SAME" in r.stdout, (r.returncode, r.stderr[-300:])
    bj2 = copy.deepcopy(bj); bj2["phases"]["3"]["seeds"][0] = 99
    b2 = write("sub_B2", bj2, arrs)
    r = subprocess.run([sys.executable, TOOL, a, b2], capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and "VERDICT: DIFFERENT" in r.stdout, (r.returncode, r.stdout[-300:])
    return "as a program: exit 0 on SAME, exit 1 on a changed survivor"


GATES = [
    ("G-CLEAN", g_clean),
    ("M1 one changed survivor", lambda: mutant("m1", m1_survivor, ["phases.3.survivors"])),
    ("M2 one changed match rate", lambda: mutant("m2", m2_match_rate, ["phases.1.match_rates"])),
    ("M3 one reordered array", lambda: mutant("m3", m3_reordered_array, ["npz.order[B]"])),
    ("M4 one missing array", lambda: mutant("m4", m4_missing_array, ["npz.score"])),
    ("M5 one changed NPZ value", lambda: mutant("m5", m5_npz_value, ["npz.forward_matches"])),
    ("M6 one changed credit interval", lambda: mutant("m6", m6_credit, ["credit.swept_intervals.4"])),
    ("M7 one changed bidirectional set", lambda: mutant("m7", m7_bidirectional, ["bidirectional.constant"])),
    ("M8 one changed phase count", lambda: mutant("m8", m8_phase_count, ["phase_counts.2.survivors"])),
    ("M9 an unclassified top-level key", lambda: mutant("m9", m9_unclassified,
                                                        ["<unclassified>.new_field_nobody_classified"])),
    ("M10 rows reordered inside an NPZ array", lambda: mutant("m10", m10_rows_reordered, ["npz.seeds"])),
    ("M11 inputs differ (not the same job)", lambda: mutant("m11", m11_inputs, [], expect_rc=2)),
    ("G-SUBPROCESS-EXIT", g_subprocess_exit),
]


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    print("VERIFICATION test_compare_engine_outputs START", flush=True)
    passed = 0
    for name, fn in GATES:
        try:
            print(f"[PASS] {name}: {fn()}", flush=True)
            passed += 1
        except Exception as exc:                                    # noqa: BLE001
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)
    print(f"\n{passed}/{len(GATES)} PASS")
    print("RESULT: PASS" if passed == len(GATES) else "RESULT: FAIL")
    print("VERIFICATION test_compare_engine_outputs COMPLETE", flush=True)
    return 0 if passed == len(GATES) else 1


if __name__ == "__main__":
    sys.exit(main())
