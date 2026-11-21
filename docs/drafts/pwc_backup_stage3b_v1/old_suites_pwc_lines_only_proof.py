"""Stage 3b: the three EXISTING suites changed only in their named PWC parts (owner allowance
2026-11-20). Whole file: scripts/prove_unchanged_outside.py compare() against b2efdef, with the
GATES / MUTANT_SPECS lists masked; those two lists are then compared ENTRY BY ENTRY and every
changed entry is listed by its label."""
import ast, importlib.util, json, subprocess, sys
spec = importlib.util.spec_from_file_location("p", "scripts/prove_unchanged_outside.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
LISTS = ("GATES", "MUTANT_SPECS")
CASES = {
 "tests/test_s172_phase1_scaffolding.py": (["gate5_watcher_manifest"], []),
 "tests/test_s172_miner_seed_start.py": (["g_r5_routes"], ["_PWC_STAGE3_ADDED_KW", "_canon_calls_without"]),
 "tests/test_step1_search_record_stage1.py": (["g1_pwc_zmq_refused", "g_miner_start", "run4_shape"], []),
}
def split(src):
    t = ast.parse(src); lists = {}
    for st in t.body:
        if isinstance(st, ast.Assign) and isinstance(st.targets[0], ast.Name) and st.targets[0].id in LISTS:
            lists[st.targets[0].id] = st.value; st.value = ast.Constant(0)
    return ast.unparse(t), lists
def label(e):
    return e.elts[0].value if isinstance(e, ast.Tuple) and isinstance(e.elts[0], ast.Constant) else ast.unparse(e)[:60]
ok = True
for path, (allow, added) in CASES.items():
    bsrc, nsrc = subprocess.check_output(["git", "show", f"b2efdef:{path}"], text=True), open(path).read()
    bm, bl = split(bsrc); nm, nl = split(nsrc)
    rep = P.compare(bm, nm, allow, added)
    print(f"{path}: {rep['checked_definitions']} definitions identical outside {allow} + added {added}: "
          f"{'OK' if rep['ok'] else 'PROBLEMS ' + json.dumps(rep['problems'])[:300]}")
    ok &= rep["ok"]
    for name in LISTS:
        if name not in bl:
            continue
        be = {ast.dump(e): label(e) for e in bl[name].elts}; ne = {ast.dump(e): label(e) for e in nl[name].elts}
        gone = sorted(be[k] for k in be.keys() - ne.keys()); new = sorted(ne[k] for k in ne.keys() - be.keys())
        same = len(be.keys() & ne.keys())
        print(f"   {name}: {same} entries identical; removed/changed {gone}; added/changed {new}")
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
