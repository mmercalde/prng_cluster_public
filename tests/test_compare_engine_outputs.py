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

NOT COMPARED (Stage 4, 2026-11-21): a phase or section absent or null in BOTH bundles is
reported NOT COMPARED, never EQUAL, and the verdict is "SAME WHERE COMPARED" (exit 3),
never SAME:
  N1 phase 3 absent in both          N2 the whole phases section null in both
  N3 bidirectional.variable absent   N4 phase_counts empty in both
  N5 credit.granted null in both     N6 npz null in both
  N7 inputs empty in both (exit 2: no input compared, so not shown to be the same job)
  N8 one non-identity input null in both (exit 3)
  N9 a phase absent in ONE bundle only (DIFFERENT, exit 1)
  N10 NOT COMPARED plus a real difference (DIFFERENT wins, exit 1)
  N11 control: an EMPTY phase in both is data, compared EQUAL (exit 0)
  N12 a phase null in ONE bundle only (that engine does not persist it): NOT COMPARED,
      not DIFFERENT (exit 3)       N13 npz null in one only   N14 credit.granted null in one
Beta X-PWC-STAGE4 R1 Q2/Q9 (2026-11-21): every F2 job-identity key (F2_REQUIRED_INPUTS) must be
EQUAL in both bundles, and each bundle must carry granted credit with certified, linked and swept
intervals equal to the run interval; otherwise NOT COMPARABLE (exit 2). N5/N14/M6 now exit 2.
  N15 dataset digest null in both    N16 Beta's case: digest absent in one + a differing credit
  N17 both credit interval fields absent in both (was exit 0)          N18 a pin absent in both
  G-GRANTED-FALSE-BOTH  G-F2-IDENTITY-EACH-KEY (each of the 15 keys removed alone -> exit 2)
Code mutant: G-CODE-MUTANT patches the comparator's single NOT COMPARED constructor to
return EQUAL (the pre-fix behaviour) and requires every N gate above to FAIL under it.
Shared helpers: build_inputs() splits engine keys from job keys; exact_float32() keeps an
exact float32 and refuses one that is not.
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


S = 2147483648
JOB = {"prng_type": "java_lcg", "test_both_modes": True, "window_trials": 1, "n_parallel": 1,
       "warm_start_window": 12, "warm_start_offset": 25, "warm_start_session_idx": 1,
       "warm_start_fwd_thresh": 0.71, "warm_start_rev_thresh": 0.47,
       "warm_start_skip_min": 0, "warm_start_skip_max": 16, "seed_start": 0, "max_seeds": 1000}
PROV = {"prng_base": "java_lcg", "skip_modes_executed": ["constant", "variable"],
        "seed_start": S, "seed_end_exclusive": S + 1000, "repository_commit": "6d75f71" + "0" * 33}


def inputs_for(engine):
    """The inputs section the extractors write: build_inputs over engine-specific params."""
    eng = ({"use_range_miner": True, "use_persistent_workers": False, "worker_pool_size": 25}
           if engine == "range_miner" else
           {"use_range_miner": False, "use_persistent_workers": True, "seed_cap_amd": 25000})
    return CEO.build_inputs(dict(JOB, **eng), PROV, "ab" * 32)[0]


def bundle_json(engine):
    return {
        "schema": CEO.SCHEMA,
        "engine": engine, "run_id": f"run-{engine}", "created_at": f"2026-11-20T00:00:{len(engine):02d}Z",
        "provenance": {"engine": engine}, "artifact_sha256": engine * 4,
        "inputs": inputs_for(engine),
        "phases": {"1": {"seeds": [10, 11, 12, 13], "match_rates": [0.5, 0.625, 0.75, 0.5]},
                   "2": {"seeds": [11, 12, 20], "match_rates": [0.5, 0.875, 0.5]},
                   "3": {"seeds": [12], "match_rates": [0.625]},
                   "4": {"seeds": [12, 30], "match_rates": [0.75, 0.5]}},
        "bidirectional": {"constant": [11, 12], "variable": [12]},
        "phase_counts": {"1": {"survivors": 4}, "2": {"survivors": 3}, "3": {"survivors": 1},
                         "4": {"survivors": 2}},
        "credit": {"granted": True, "certified_interval": [S, S + 1000],
                   "linked_interval": [S, S + 1000],
                   "swept_intervals": {p: [S, S + 1000] for p in ("1", "2", "3", "4")}},
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
    return sorted(r["field"] for r in rep["outputs"] if r["status"] == CEO.DIFFERENT)


def not_compared(rep):
    return sorted(rep["not_compared"])


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
    bj["credit"]["swept_intervals"]["4"][1] -= 1         # B's phase 4 did not sweep the job
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
    bj["inputs"]["param.warm_start_skip_max"] = 99        # not the same job
    return bj, arrs, order


def nc_mutant(tag, mutate, expect_nc, expect_rc=3, expect_diff=()):
    """Apply one change to BOTH bundles (absence in both) or to B only, then require the
    named fields to be NOT COMPARED (never EQUAL) and the exit code to match."""
    a_bj, b_bj = bundle_json("range_miner"), bundle_json("pwc")
    a_bj, b_bj = mutate(copy.deepcopy(a_bj), copy.deepcopy(b_bj))
    a = write(f"{tag}_A", a_bj, arrays())
    b = write(f"{tag}_B", b_bj, arrays())
    rc, rep, txt = run(a, b)
    rows = {r["field"]: r for r in rep["inputs"] + rep["outputs"]}
    for f in expect_nc:
        assert f in rows, f"{tag}: no row {f}; rows={sorted(rows)}"
        assert rows[f]["status"] == CEO.NOT_COMPARED, f"{tag}: {f} is {rows[f]['status']}"
        assert not rows[f]["equal"], f"{tag}: {f} reads equal"
    for f in expect_diff:
        assert f in differing(rep), f"{tag}: {f} not DIFFERENT; differing={differing(rep)}"
    assert rc == expect_rc, f"{tag}: exit {rc}, want {expect_rc}; nc={not_compared(rep)} diff={differing(rep)}"
    assert not rep["same"], f"{tag}: report claims same"
    if expect_rc == 3:
        assert "VERDICT: SAME WHERE COMPARED" in txt and "VERDICT: SAME (" not in txt, txt[-300:]
    return f"exit {rc}; NOT COMPARED: {', '.join(not_compared(rep)) or '-'}" + (
        f"; DIFFERENT: {', '.join(differing(rep))}" if differing(rep) else "")


def _both(fn):
    return lambda a, b: (fn(a), fn(b))


def n1(bj):
    del bj["phases"]["3"]; return bj


def n2(bj):
    bj["phases"] = None; return bj


def n3(bj):
    del bj["bidirectional"]["variable"]; return bj


def n4(bj):
    bj["phase_counts"] = {}; return bj


def n5(bj):
    bj["credit"]["granted"] = None; return bj


def n6(bj):
    bj["npz"] = None; return bj


def n7(bj):
    bj["inputs"] = {}; return bj


def n8(bj):
    bj["inputs"]["param.window_trials"] = None; return bj   # not an F2 identity key


def n9(a, b):
    del b["phases"]["3"]; return a, b


def n10(a, b):
    del a["phases"]["3"]; del b["phases"]["3"]
    b["phases"]["1"]["seeds"][0] = 99
    return a, b


def n12(a, b):
    b["phases"]["3"] = None; return a, b


def n13(a, b):
    a["npz"] = None; return a, b


def n14(a, b):
    b["credit"]["granted"] = None; return a, b


def n15(bj):
    bj["inputs"]["run.dataset_sha256"] = None; return bj


def n16(a, b):
    """Beta's scratch case: a missing dataset digest and a differing credit field gave
    comparable=true, exit 1. Now: not comparable, exit 2."""
    del b["inputs"]["run.dataset_sha256"]
    b["credit"]["linked_interval"] = [S, S + 999]
    return a, b


def n17(bj):
    del bj["credit"]["certified_interval"]; del bj["credit"]["linked_interval"]; return bj


def n18(bj):
    del bj["inputs"]["param.warm_start_skip_min"]; return bj


def g_granted_false_both():
    a_bj, b_bj = bundle_json("range_miner"), bundle_json("pwc")
    for bj in (a_bj, b_bj):
        bj["credit"]["granted"] = False
    a = write("gf_A", a_bj, arrays()); b = write("gf_B", b_bj, arrays())
    rc, rep, txt = run(a, b)
    assert rc == 2 and not rep["comparable"], (rc, rep["credit_problems"])
    assert any("granted is False" in c for c in rep["credit_problems"]), rep["credit_problems"]
    assert "CREDIT EVIDENCE" in txt
    rep2 = CEO.compare(CEO.load_bundle(a), CEO.load_bundle(b), require_credit=False)
    assert CEO.exit_code(rep2) == 0, rep2["verdict"]
    return "granted=false in both: exit 2 (F3 credit not met); with --no-require-credit: exit 0"


def g_identity_list():
    """Every F2 key, removed from both bundles alone, makes the pair not comparable."""
    bad = []
    for k in CEO.F2_REQUIRED_INPUTS:
        a_bj, b_bj = bundle_json("range_miner"), bundle_json("pwc")
        assert k in a_bj["inputs"], f"fixture lacks {k}"
        del a_bj["inputs"][k]; del b_bj["inputs"][k]
        a = write("id_A", a_bj, arrays()); b = write("id_B", b_bj, arrays())
        rc, rep, _ = run(a, b)
        if rc != 2 or k not in rep["identity_missing"]:
            bad.append((k, rc))
    assert not bad, bad
    return f"each of the {len(CEO.F2_REQUIRED_INPUTS)} F2 keys, absent in both: exit 2"


def n11(bj):
    bj["phases"]["2"] = {"seeds": [], "match_rates": []}
    bj["bidirectional"]["variable"] = []
    return bj


def g_n11_empty_is_data():
    a = write("n11_A", n11(bundle_json("range_miner")), arrays())
    b = write("n11_B", n11(bundle_json("pwc")), arrays())
    rc, rep, txt = run(a, b)
    rows = {r["field"]: r for r in rep["outputs"]}
    assert rc == 0 and rep["same"], (rc, not_compared(rep), differing(rep))
    assert rows["phases.2.survivors"]["status"] == CEO.EQUAL, rows["phases.2.survivors"]
    assert rows["bidirectional.variable"]["status"] == CEO.EQUAL
    return "an empty phase and an empty set in both are compared: EQUAL, exit 0"


N_GATES = [
    ("N1 phase 3 absent in both", lambda: nc_mutant("n1", _both(n1), ["phases.3"])),
    ("N2 phases null in both", lambda: nc_mutant("n2", _both(n2), ["phases"])),
    ("N3 bidirectional.variable absent in both",
     lambda: nc_mutant("n3", _both(n3), ["bidirectional.variable"])),
    ("N4 phase_counts empty in both", lambda: nc_mutant(
        "n4", _both(n4), [f"phase_counts.{p}" for p in ("1", "2", "3", "4")])),
    ("N5 credit.granted null in both (credit required: exit 2)",
     lambda: nc_mutant("n5", _both(n5), ["credit.granted"], expect_rc=2)),
    ("N6 npz null in both", lambda: nc_mutant("n6", _both(n6), ["npz"])),
    ("N7 inputs empty in both", lambda: nc_mutant("n7", _both(n7), ["inputs"], expect_rc=2)),
    ("N8 one non-identity input null in both", lambda: nc_mutant("n8", _both(n8), ["inputs.param.window_trials"])),
    ("N9 phase absent in one bundle only", lambda: nc_mutant("n9", n9, [], expect_rc=1,
                                                             expect_diff=["phases.3"])),
    ("N10 NOT COMPARED plus a real difference", lambda: nc_mutant(
        "n10", n10, ["phases.3"], expect_rc=1, expect_diff=["phases.1.survivors"])),
    ("N11 empty phase in both is data (control)", g_n11_empty_is_data),
    ("N12 phase null in one bundle only", lambda: nc_mutant("n12", n12, ["phases.3"])),
    ("N13 npz null in one bundle only", lambda: nc_mutant("n13", n13, ["npz"])),
    ("N14 credit.granted null in one only (credit required: exit 2)",
     lambda: nc_mutant("n14", n14, ["credit.granted"], expect_rc=2)),
    ("N15 F2 identity: dataset digest null in both", lambda: nc_mutant(
        "n15", _both(n15), ["inputs.run.dataset_sha256"], expect_rc=2)),
    ("N16 F2 identity: dataset digest absent in one, credit field also differs", lambda: nc_mutant(
        "n16", n16, [], expect_rc=2)),
    ("N17 credit interval fields absent in both", lambda: nc_mutant(
        "n17", _both(n17), ["credit.certified_interval", "credit.linked_interval"], expect_rc=2)),
    ("N18 F2 identity: a pin absent in both", lambda: nc_mutant(
        "n18", _both(n18), ["inputs.param.warm_start_skip_min"], expect_rc=2)),
]


def g_code_mutant():
    """Revert the fix in memory: NOT COMPARED rows become EQUAL. Every N1-N8 and N10 gate
    must then FAIL; if one still passes, it does not test the fix."""
    real = CEO._nc
    CEO._nc = lambda field, why="": CEO._row(field, CEO.EQUAL, why)
    survived = []
    try:
        for name, fn in N_GATES:
            if name.startswith(("N9", "N11", "N16")):
                continue                    # these do not depend on NOT COMPARED rows
            try:
                fn()
                survived.append(name)
            except AssertionError:
                pass
    finally:
        CEO._nc = real
    assert not survived, f"the reverted comparator still passes: {survived}"
    return f"with NOT COMPARED reverted to EQUAL, all {len(N_GATES) - 2} dependent N gates fail"


def g_build_inputs():
    params = {"prng_type": "java_lcg", "use_range_miner": True, "worker_pool_size": 25,
              "seed_start": 0, "max_seeds": 268435456, "warm_start_skip_min": 0,
              "warm_start_skip_max": 16, "test_both_modes": True}
    prov = {"prng_base": "java_lcg", "skip_modes_executed": ["constant", "variable"],
            "seed_start": 2147483648, "seed_end_exclusive": 2415919104,
            "repository_commit": "6d75f71"}
    inp, geo = CEO.build_inputs(params, prov, "ab" * 32)
    assert set(geo) == {"use_range_miner", "worker_pool_size", "seed_start", "max_seeds"}, geo
    assert inp["param.warm_start_skip_max"] == 16 and "param.use_range_miner" not in inp
    assert inp["run.seed_end_exclusive"] == 2415919104 and inp["run.dataset_sha256"] == "ab" * 32
    p2 = dict(params, use_range_miner=False, use_persistent_workers=True, seed_cap_amd=25000)
    inp2, _ = CEO.build_inputs(p2, prov, "ab" * 32)
    assert inp2 == inp, "engine-selection keys leaked into inputs"
    inp3, _ = CEO.build_inputs(params, prov, None)
    assert inp3["run.dataset_sha256"] is None
    return "engine keys go to geometry; the two engines' params give equal inputs"


def g_exact_float32():
    good = [float(np.float32(x)) for x in (0.5, 0.71, 7 / 12, 1.0)]
    assert CEO.exact_float32(good, "t") == good
    for bad in ([7 / 12], [0.71], ["0.5"], [True]):
        try:
            CEO.exact_float32(bad, "t")
        except CEO.BundleError:
            continue
        raise AssertionError(f"{bad!r} was accepted")
    return "float32 values kept bit-exact; 7/12 and 0.71 as float64, strings, bools refused"


def g_subprocess_exit():
    a, bj, arrs = clean_pair("sub")
    b = write("sub_B", bj, arrs)
    r = subprocess.run([sys.executable, TOOL, a, b], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "VERDICT: SAME" in r.stdout, (r.returncode, r.stderr[-300:])
    bj2 = copy.deepcopy(bj); bj2["phases"]["3"]["seeds"][0] = 99
    b2 = write("sub_B2", bj2, arrs)
    r = subprocess.run([sys.executable, TOOL, a, b2], capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and "VERDICT: DIFFERENT" in r.stdout, (r.returncode, r.stdout[-300:])
    bj3 = copy.deepcopy(bj); del bj3["phases"]["3"]
    a3 = write("sub_A3", n1(bundle_json("range_miner")), arrays())
    b3 = write("sub_B3", bj3, arrs)
    r = subprocess.run([sys.executable, TOOL, a3, b3], capture_output=True, text=True, timeout=60)
    assert r.returncode == 3 and "NOT COMPARED  phases.3" in r.stdout, (r.returncode, r.stdout[-400:])
    return "as a program: exit 0 SAME, 1 changed survivor, 3 phase absent in both"


GATES = [
    ("G-CLEAN", g_clean),
    ("M1 one changed survivor", lambda: mutant("m1", m1_survivor, ["phases.3.survivors"])),
    ("M2 one changed match rate", lambda: mutant("m2", m2_match_rate, ["phases.1.match_rates"])),
    ("M3 one reordered array", lambda: mutant("m3", m3_reordered_array, ["npz.order[B]"])),
    ("M4 one missing array", lambda: mutant("m4", m4_missing_array, ["npz.score"])),
    ("M5 one changed NPZ value", lambda: mutant("m5", m5_npz_value, ["npz.forward_matches"])),
    ("M6 one changed credit interval (now credit evidence: exit 2)",
     lambda: mutant("m6", m6_credit, ["credit.swept_intervals.4"], expect_rc=2)),
    ("M7 one changed bidirectional set", lambda: mutant("m7", m7_bidirectional, ["bidirectional.constant"])),
    ("M8 one changed phase count", lambda: mutant("m8", m8_phase_count, ["phase_counts.2.survivors"])),
    ("M9 an unclassified top-level key", lambda: mutant("m9", m9_unclassified,
                                                        ["<unclassified>.new_field_nobody_classified"])),
    ("M10 rows reordered inside an NPZ array", lambda: mutant("m10", m10_rows_reordered, ["npz.seeds"])),
    ("M11 inputs differ (not the same job)", lambda: mutant("m11", m11_inputs, [], expect_rc=2)),
    *N_GATES,
    ("G-GRANTED-FALSE-BOTH", g_granted_false_both),
    ("G-F2-IDENTITY-EACH-KEY", g_identity_list),
    ("G-CODE-MUTANT (NOT COMPARED reverted to EQUAL)", g_code_mutant),
    ("G-BUILD-INPUTS", g_build_inputs),
    ("G-EXACT-FLOAT32", g_exact_float32),
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
