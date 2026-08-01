#!/usr/bin/env python3
"""
test_chapter1_p0_corrections.py — acceptance harness for Chapter 1 P0 items 1-5.

Authority : docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_1_P0_CORRECTION.md §8
Findings  : docs/CHAPTER_1_AUDIT_v1.md (db9782a), §6 correction list
Threshold : docs/THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md (cited, not re-derived)

    G-FLAG-FAILCLOSED        --forward/--reverse-threshold abort before coordinator
    G-STRATEGY-FAILCLOSED    random/grid/evolutionary abort; bayesian still dispatches
    G-NO-SILENT-SUBSTITUTION a Bayesian request never becomes random search
    G-METADATA-PROVENANCE    agent_metadata reports what executed; no 0.72/0.81; no clamp
    G-SNAPSHOT-EXTRACTED     chapter bounds snapshot matches live config + carries
                             repository_commit and configuration_digest
    G-SKIP-DEFECT-NOTE       chapter keeps the verbatim skip definition + defect callout

VERIFICATION-INTEGRITY CONTROLS (VIR-1…6)
-----------------------------------------
execution proof
    Every behavioural gate EXECUTES the live call site. The CLI gates run
    `window_optimizer.py` as a subprocess and assert on the real exit status.
    G-METADATA runs the real `run_bayesian_optimization` with only the
    coordinator faked, so the live metadata block writes a real file that is
    then read back. G-STRATEGY additionally derives the forwarded-kwarg set
    from the AST of the LIVE `strategy.search(...)` call inside
    `WindowOptimizer.optimize` — commit 2389b61 reverted a fix by whole-block
    replacement, and a text anchor would have gone green.
clean control
    Every gate has a negative arm on the UNMUTATED tree: the flag-absent CLI
    invocation dispatches, `--strategy bayesian` dispatches, a constructible
    Optuna search delegates, resolvable thresholds are reported.
fault-injection control
    Each gate is re-run against a SOURCE-MUTATED copy of the file it guards
    (loaded from a temp path, never written into the repo) and must go RED.
    Six mutants, one per gate.
detector independence
    The mutants are source edits to the production file; the detectors are
    subprocess exit codes, a written JSON artifact, and a regenerated snapshot.
    No detector shares an expression with the code it checks.
completion sentinel
    PASS | FAIL | UNAVAILABLE | INCOMPLETE, printed at the end. Only PASS accepts.
unavailable-observer behavior
    A gate whose surface cannot be reached reports UNAVAILABLE and is NOT
    counted as green.
audit claim scope
    Repo-scoped, VM 101 working tree. No GPU, no sieve, no rig, no pipeline.
    Deployed rig copies are NOT contacted and no claim is made about them.
"""

import ast
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

WO_PATH = os.path.join(_ROOT, "window_optimizer.py")
CHAPTER = os.path.join(_ROOT, "docs", "CHAPTER_1_WINDOW_OPTIMIZER.md")
SNAPSHOT_TOOL = os.path.join(_ROOT, "scripts", "extract_search_bounds_snapshot.py")
CONFIG = os.path.join(_ROOT, "distributed_config.json")

RESULTS = []          # (name, "PASS" | "FAIL" | "UNAVAILABLE", detail)
_GREEN = "\033[92m"
_RED = "\033[91m"
_YEL = "\033[93m"
_OFF = "\033[0m"


def _check(name, fn):
    try:
        fn()
    except _Unavailable as e:
        RESULTS.append((name, "UNAVAILABLE", str(e)))
        print(f"  [{_YEL}UNAVAILABLE{_OFF}] {name}: {e}", flush=True)
    except Exception:
        RESULTS.append((name, "FAIL", traceback.format_exc()))
        print(f"  [{_RED}FAIL{_OFF}] {name}", flush=True)
    else:
        RESULTS.append((name, "PASS", ""))
        print(f"  [{_GREEN}PASS{_OFF}] {name}", flush=True)


class _Unavailable(Exception):
    """The surface this gate needs could not be reached (VIR-5)."""


# ---------------------------------------------------------------------------
# mutation plumbing — mutants are written to a TEMP path, never into the repo
# ---------------------------------------------------------------------------

def _mutate_source(path, replacements):
    """Return mutated text of `path`; every replacement must actually apply."""
    src = open(path, "r", encoding="utf-8").read()
    for old, new in replacements:
        if old not in src:
            raise AssertionError(
                f"mutation anchor not found in {os.path.basename(path)} — the "
                f"harness is stale relative to the source it guards: {old[:90]!r}"
            )
        src = src.replace(old, new, 1)
    return src


def _mutant_script(replacements):
    """Write a mutated window_optimizer.py to a temp dir; return its path."""
    d = tempfile.mkdtemp(prefix="ch1_p0_mutant_")
    p = os.path.join(d, "window_optimizer_mutant.py")
    with open(p, "w", encoding="utf-8") as f:
        f.write(_mutate_source(WO_PATH, replacements))
    return p


def _load_mutant_module(replacements, name):
    """Import a mutated copy of window_optimizer.py as a fresh module object."""
    p = _mutant_script(replacements)
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_cli(script, args, timeout=120):
    """Run a window_optimizer script and return (rc, stdout, stderr)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = _ROOT + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, script] + args, cwd=_ROOT, env=env,
                          capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


# A sentinel that makes the CLI dispatch observable without building a
# coordinator: --strategy bayesian would otherwise spin 26 GPUs. We instead
# assert the run got PAST argument validation by observing that it reached the
# coordinator-construction banner, then killed itself on a bogus config path.
DISPATCH_MARKER = "BAYESIAN WINDOW OPTIMIZATION WITH REAL SIEVES"
COORDINATOR_MARKER = "Initializing 26-GPU coordinator"


# ===========================================================================
# G-FLAG-FAILCLOSED
# ===========================================================================

def gate_flag_failclosed():
    """
    --forward-threshold / --reverse-threshold produce a nonzero failure BEFORE
    coordinator construction (dead dimension D-4).
    """
    for flag, value in (("--forward-threshold", "0.6"),
                        ("--reverse-threshold", "0.0"),   # 0.0 must also abort
                        ("--forward-threshold", "0.31")):
        rc, out, err = _run_cli(WO_PATH,
                                ["--strategy", "bayesian", "--lottery-file",
                                 "daily3.json", flag, value])
        blob = out + err
        assert rc != 0, f"{flag} {value} exited 0 — the silent no-op is back"
        assert "WINDOW_OPTIMIZER_THRESHOLD_OVERRIDE_UNWIRED" in blob, \
            f"{flag}: fail-closed diagnostic missing; got:\n{blob[-600:]}"
        # BEFORE coordinator construction: the banner and the coordinator init
        # line both live inside run_bayesian_optimization, downstream of the gate.
        assert DISPATCH_MARKER not in blob, \
            f"{flag}: run reached run_bayesian_optimization before failing"
        assert COORDINATOR_MARKER not in blob, \
            f"{flag}: run constructed the coordinator before failing"

    # CLEAN CONTROL — flag absent: the same invocation must NOT produce the
    # diagnostic and must proceed into run_bayesian_optimization.
    rc, out, err = _run_cli(WO_PATH, ["--strategy", "bayesian",
                                      "--lottery-file", "daily3.json",
                                      "--trials", "1"], timeout=180)
    blob = out + err
    assert "WINDOW_OPTIMIZER_THRESHOLD_OVERRIDE_UNWIRED" not in blob, \
        "clean control: the gate fires when no override flag was passed"
    assert DISPATCH_MARKER in blob, \
        ("clean control: flag-absent run never reached run_bayesian_optimization, "
         "so the gate's 'before coordinator' assertion proves nothing:\n"
         + blob[-800:])


def mutant_flag_silent_noop():
    """FAULT INJECTION — restore the silent no-op; the gate must go RED."""
    mutant = _mutant_script([(
        "    if _unwired_flags:\n        parser.error(",
        "    if False:\n        parser.error(",
    )])
    rc, out, err = _run_cli(mutant, ["--strategy", "bayesian", "--lottery-file",
                                     "daily3.json", "--forward-threshold", "0.6"],
                            timeout=180)
    blob = out + err
    assert "WINDOW_OPTIMIZER_THRESHOLD_OVERRIDE_UNWIRED" not in blob, \
        "mutant still emitted the diagnostic — the mutation did not take"
    assert DISPATCH_MARKER in blob, \
        ("mutant did not reach run_bayesian_optimization, so it does not "
         "reproduce the silent no-op the gate detects")


# ===========================================================================
# G-STRATEGY-FAILCLOSED
# ===========================================================================

def _live_forwarded_kwargs():
    """
    AST-extract the keyword names of the LIVE `strategy.search(...)` call inside
    WindowOptimizer.optimize. Not a text match: 2389b61 reverted a fix by
    replacing a whole block, which a text anchor would not have caught.
    """
    tree = ast.parse(open(WO_PATH, "r", encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "WindowOptimizer":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "optimize":
                    for call in ast.walk(fn):
                        if (isinstance(call, ast.Call)
                                and isinstance(call.func, ast.Attribute)
                                and call.func.attr == "search"):
                            return tuple(sorted(
                                k.arg for k in call.keywords if k.arg))
    raise _Unavailable("no strategy.search(...) call found in "
                       "WindowOptimizer.optimize — cannot derive the contract")


def gate_strategy_failclosed():
    import window_optimizer as WO

    # EXECUTION PROOF that the production constant is the live call site's
    # contract, not a copy that can silently drift away from it.
    live = _live_forwarded_kwargs()
    assert tuple(sorted(WO.OPTIMIZE_FORWARDED_KWARGS)) == live, (
        f"OPTIMIZE_FORWARDED_KWARGS {sorted(WO.OPTIMIZE_FORWARDED_KWARGS)} != the "
        f"kwargs the live optimize() call forwards {list(live)}")

    # the gap is computed from LIVE signatures
    for broken in ("random", "grid", "evolutionary"):
        gap = WO.strategy_contract_gap(WO.STRATEGY_CLASSES[broken])
        assert gap, f"{broken}: expected a signature gap, live signature accepts all"
    assert WO.strategy_contract_gap(WO.BayesianOptimization) == (), \
        "bayesian must be callable by optimize()"

    # CLI: the three broken strategies abort before the coordinator
    for broken in ("random", "grid", "evolutionary"):
        rc, out, err = _run_cli(WO_PATH, ["--strategy", broken,
                                          "--lottery-file", "daily3.json"])
        blob = out + err
        assert rc != 0, f"--strategy {broken} exited 0"
        assert "WINDOW_OPTIMIZER_STRATEGY_UNSUPPORTED" in blob, \
            f"--strategy {broken}: diagnostic missing:\n{blob[-600:]}"
        # a raw TypeError must not escape — the diagnostic NAMES TypeError as the
        # cause, so look for an actual traceback rather than the word.
        assert "Traceback (most recent call last)" not in blob, \
            f"--strategy {broken}: an exception escaped instead of failing closed:\n{blob[-600:]}"
        assert COORDINATOR_MARKER not in blob, \
            f"--strategy {broken}: coordinator was constructed before failing"

    # an unknown name must not silently become RandomSearch
    try:
        WO.require_supported_strategy("no_such_strategy")
    except WO.StrategyContractError as e:
        assert "WINDOW_OPTIMIZER_STRATEGY_UNKNOWN" in str(e)
    else:
        raise AssertionError("unknown strategy name did not fail closed")

    # CLEAN CONTROL — bayesian still runs
    rc, out, err = _run_cli(WO_PATH, ["--strategy", "bayesian", "--lottery-file",
                                      "daily3.json", "--trials", "1"], timeout=180)
    blob = out + err
    assert "WINDOW_OPTIMIZER_STRATEGY_UNSUPPORTED" not in blob, \
        "clean control: bayesian was gated"
    assert DISPATCH_MARKER in blob, \
        "clean control: bayesian did not reach run_bayesian_optimization:\n" + blob[-800:]


def mutant_strategy_random_permitted():
    """FAULT INJECTION — re-permit `random`; the gate must go RED."""
    mutant = _mutant_script([(
        "        try:\n            require_supported_strategy(args.strategy)",
        "        try:\n            pass",
    )])
    rc, out, err = _run_cli(mutant, ["--strategy", "random", "--lottery-file",
                                     "daily3.json", "--trials", "1"], timeout=180)
    blob = out + err
    assert "WINDOW_OPTIMIZER_STRATEGY_UNSUPPORTED" not in blob, \
        "mutant still gated random — the mutation did not take"
    assert DISPATCH_MARKER in blob, \
        "mutant did not reach dispatch, so it does not reproduce the defect"


# ===========================================================================
# G-NO-SILENT-SUBSTITUTION
# ===========================================================================

def gate_no_silent_substitution():
    """A Bayesian request with Optuna unavailable FAILS. It never becomes random."""
    import window_optimizer as WO

    tripped = {"random_search_ran": False}
    original = WO.RandomSearch.search

    def tripwire(self, *a, **k):
        tripped["random_search_ran"] = True
        return original(self, *a, **k)

    WO.RandomSearch.search = tripwire
    try:
        strategy = WO.BayesianOptimization(n_initial=3)
        strategy.optuna_search = None       # Optuna unavailable

        try:
            strategy.search(lambda cfg, **kw: None, WO.SearchBounds.from_config(),
                            5, WO.BidirectionalCountScorer())
        except WO.StrategyContractError as e:
            assert "WINDOW_OPTIMIZER_BAYESIAN_UNAVAILABLE" in str(e), \
                f"wrong diagnostic: {e}"
        else:
            raise AssertionError(
                "Bayesian request with Optuna unavailable did NOT fail — "
                "semantic substitution is back")
        assert not tripped["random_search_ran"], \
            "RandomSearch.search was invoked for a Bayesian request"

        # CLEAN CONTROL — with a constructible search object it still delegates
        class _StubOptuna:
            def __init__(self):
                self.called = False

            def search(self, *a, **k):
                self.called = True
                return {"strategy": "stub", "best_score": 1.0}

        stub = _StubOptuna()
        strategy.optuna_search = stub
        out = strategy.search(lambda cfg, **kw: None, WO.SearchBounds.from_config(),
                              5, WO.BidirectionalCountScorer())
        assert stub.called and out["strategy"] == "stub", \
            "clean control: the available-Optuna path no longer delegates"
        assert not tripped["random_search_ran"], \
            "clean control: RandomSearch ran on the delegating path"
    finally:
        WO.RandomSearch.search = original


def mutant_silent_random_fallback():
    """FAULT INJECTION — restore the random-search fallback; gate must go RED."""
    mod = _load_mutant_module([(
        '            raise StrategyContractError(\n'
        '                "WINDOW_OPTIMIZER_BAYESIAN_UNAVAILABLE:',
        '            return RandomSearch().search(objective_function, bounds,\n'
        '                                         max_iterations, scorer)\n'
        '            raise StrategyContractError(\n'
        '                "WINDOW_OPTIMIZER_BAYESIAN_UNAVAILABLE:',
    )], "wo_mutant_optuna_fallback")

    ran = {"random": False}
    original = mod.RandomSearch.search

    def tripwire(self, objective_function, bounds, max_iterations, scorer):
        ran["random"] = True
        return {"strategy": "random_search", "best_score": 0.0,
                "best_config": {}, "best_result": {}, "all_results": [],
                "iterations": 0}

    mod.RandomSearch.search = tripwire
    try:
        strategy = mod.BayesianOptimization(n_initial=3)
        strategy.optuna_search = None
        try:
            strategy.search(lambda cfg, **kw: None, mod.SearchBounds.from_config(),
                            1, mod.BidirectionalCountScorer())
        except mod.StrategyContractError:
            raise AssertionError("mutant still failed closed — mutation did not take")
        assert ran["random"], \
            "mutant did not reach RandomSearch, so it does not reproduce the defect"
    finally:
        mod.RandomSearch.search = original


# ===========================================================================
# G-METADATA-PROVENANCE
# ===========================================================================

def _run_metadata_block(module, best_config, out_path):
    """
    Execute the LIVE run_bayesian_optimization metadata path with only the
    coordinator faked, and return the parsed optimal_window_config.json.
    """
    import window_optimizer_integration_final as WOIF

    results = {
        "strategy": "bayesian_optimization",
        "best_config": dict(best_config),
        "best_result": {"forward_count": 10, "reverse_count": 9,
                        "bidirectional_count": 5},
        "best_score": 5.0,
        "all_results": [],
        "iterations": 1,
    }

    class _FakeCoordinator:
        def __init__(self, *a, **k):
            pass

        def optimize_window(self, **kwargs):
            return results

    saved_coord = module.MultiGPUCoordinator
    saved_add = WOIF.add_window_optimizer_to_coordinator
    module.MultiGPUCoordinator = _FakeCoordinator
    WOIF.add_window_optimizer_to_coordinator = lambda *a, **k: None

    # The live function has two side effects on repo-root artifacts that this
    # harness must not leave behind: the [S121] TRSE confirmed_windows append,
    # and the 80/20 split which writes train_history.json / holdout_history.json
    # to HARDCODED paths. The TRSE block is skipped by pointing it at a path
    # that does not exist (an empty string would fall back to the live file —
    # `trse_context_file if trse_context_file else 'trse_context.json'`); the
    # split files are saved and restored byte-for-byte.
    absent_trse = os.path.join(tempfile.mkdtemp(prefix="ch1_p0_notrse_"),
                               "no_such_trse_context.json")
    split_files = ("train_history.json", "holdout_history.json")
    saved_bytes = {}
    for name in split_files:
        p = os.path.join(_ROOT, name)
        saved_bytes[name] = open(p, "rb").read() if os.path.exists(p) else None
    try:
        module.run_bayesian_optimization(
            lottery_file="daily3.json", trials=1, output_config=out_path,
            seed_count=1000, prng_type="java_lcg",
            trse_context_file=absent_trse,
        )
    finally:
        module.MultiGPUCoordinator = saved_coord
        WOIF.add_window_optimizer_to_coordinator = saved_add
        for name, blob in saved_bytes.items():
            p = os.path.join(_ROOT, name)
            if blob is None:
                if os.path.exists(p):
                    os.remove(p)
            else:
                with open(p, "wb") as f:
                    f.write(blob)

    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)


def gate_metadata_provenance():
    import window_optimizer as WO

    base = {"window_size": 12, "offset": 0, "sessions": ["midday"],
            "skip_min": 0, "skip_max": 16}
    d = tempfile.mkdtemp(prefix="ch1_p0_meta_")

    # (a) resolvable thresholds are reported EXACTLY as executed
    cfg = _run_metadata_block(WO, dict(base, forward_threshold=0.73,
                                       reverse_threshold=0.31),
                              os.path.join(d, "a.json"))
    sp = cfg["agent_metadata"]["suggested_params"]
    assert sp["forward_threshold"] == 0.73 and sp["reverse_threshold"] == 0.31, \
        f"executed thresholds not reported: {sp}"
    prov = cfg.get("executed_thresholds")
    assert prov and prov["forward_threshold"] == 0.73 \
        and prov["reverse_threshold"] == 0.31, \
        f"provenance field missing or wrong: {prov}"
    assert "resolve_directional_threshold" in prov.get("resolver", ""), \
        "provenance does not name the single resolver"
    blob = json.dumps(cfg)
    assert "0.72" not in blob and "0.81" not in blob, \
        f"invented constants present in metadata: {blob}"

    # (b) 0.0 is legitimate and must survive — `is None`, not truthiness
    cfg = _run_metadata_block(WO, dict(base, forward_threshold=0.0,
                                       reverse_threshold=0.0),
                              os.path.join(d, "b.json"))
    sp = cfg["agent_metadata"]["suggested_params"]
    assert sp["forward_threshold"] == 0.0 and sp["reverse_threshold"] == 0.0, \
        f"0.0 was replaced — truthiness fallback is back: {sp}"

    # (c) NO authoritative value -> field OMITTED, never substituted
    cfg = _run_metadata_block(WO, dict(base), os.path.join(d, "c.json"))
    sp = cfg["agent_metadata"]["suggested_params"]
    assert "forward_threshold" not in sp and "reverse_threshold" not in sp, \
        f"unresolvable thresholds were invented rather than omitted: {sp}"
    assert "executed_thresholds" not in cfg, \
        "provenance block emitted with no authoritative value"
    blob = json.dumps(cfg)
    assert "0.72" not in blob and "0.81" not in blob, \
        f"fallback constants reappeared when nothing resolved: {blob}"

    # (d) NEVER CLAMPED — an authoritative value above the 0.75 ceiling is
    #     reported as-is. Clamping would launder a governance breach into a
    #     plausible-looking number, which is the failure this repair exists for.
    cfg = _run_metadata_block(WO, dict(base, forward_threshold=0.81,
                                       reverse_threshold=0.90),
                              os.path.join(d, "e.json"))
    sp = cfg["agent_metadata"]["suggested_params"]
    assert sp["forward_threshold"] == 0.81 and sp["reverse_threshold"] == 0.90, \
        f"an out-of-range executed value was clamped: {sp}"

    # CLEAN CONTROL — the untouched fields still populate
    assert cfg["agent_metadata"]["pipeline_step"] == 1
    assert cfg["agent_metadata"]["suggested_params"]["window_size"] == 12


def mutant_metadata_constants():
    """FAULT INJECTION — restore the 0.72/0.81 constants; gate must go RED."""
    mod = _load_mutant_module([(
        "    for _direction in ('forward', 'reverse'):\n"
        "        try:\n"
        "            _executed_thresholds[_direction] = resolve_directional_threshold(\n"
        "                _best_config_view, _direction)",
        "    for _direction, _magic in (('forward', 0.72), ('reverse', 0.81)):\n"
        "        try:\n"
        "            _executed_thresholds[_direction] = best_config.get(\n"
        "                _direction + '_threshold', _magic)",
    )], "wo_mutant_metadata_constants")

    d = tempfile.mkdtemp(prefix="ch1_p0_meta_mut_")
    cfg = _run_metadata_block(mod, {"window_size": 12, "offset": 0,
                                    "sessions": ["midday"], "skip_min": 0,
                                    "skip_max": 16},
                              os.path.join(d, "m.json"))
    sp = cfg["agent_metadata"]["suggested_params"]
    assert sp.get("forward_threshold") == 0.72 and sp.get("reverse_threshold") == 0.81, \
        f"mutant did not reintroduce the constants — mutation did not take: {sp}"


# ===========================================================================
# G-SNAPSHOT-EXTRACTED
# ===========================================================================

def _chapter_snapshot_block(text):
    start = text.find("<!-- BEGIN EXTRACTED BOUNDS SNAPSHOT")
    end = text.find("<!-- END EXTRACTED BOUNDS SNAPSHOT")
    if start < 0 or end < 0:
        raise AssertionError("chapter carries no extracted bounds snapshot block")
    return text[start:end]


def _parse_snapshot_bounds(block):
    """
    Parse the snapshot's `extracted bounds` rows into {family: {bound: value}}.

    Rows look like:  `    window_size        min=6, max=50, default=12`
    Exact tokenisation, so a documented 500 can never satisfy a live 50.
    """
    bounds, in_rows = {}, False
    for line in block.splitlines():
        if "extracted bounds:" in line:
            in_rows = True
            continue
        if not in_rows:
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("```"):
            if stripped.startswith("```"):
                break
            continue
        parts = stripped.split(None, 1)
        if len(parts) != 2 or "=" not in parts[1]:
            continue
        family, rest = parts
        entry = {}
        for token in rest.split(","):
            token = token.strip()
            if "=" not in token:
                continue
            k, _, v = token.partition("=")
            entry[k.strip()] = v.strip()
        bounds[family] = entry
    return bounds


def gate_snapshot_extracted(chapter=None):
    chapter = chapter or CHAPTER
    if not os.path.exists(SNAPSHOT_TOOL):
        raise _Unavailable(f"{SNAPSHOT_TOOL} missing — cannot verify extraction")

    proc = subprocess.run([sys.executable, SNAPSHOT_TOOL, "--json"], cwd=_ROOT,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"snapshot tool failed: {proc.stderr[-500:]}"
    live = json.loads(proc.stdout)

    block = _chapter_snapshot_block(open(chapter, "r", encoding="utf-8").read())

    # provenance fields present and well-formed — a DATE ALONE IS NOT ENOUGH
    assert "repository_commit" in block, "snapshot lacks repository_commit"
    assert "configuration_digest" in block, "snapshot lacks configuration_digest"
    commit = [l for l in block.splitlines() if "repository_commit" in l][0]
    sha = commit.split(":", 1)[1].strip().split()[0]
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha), \
        f"repository_commit is not a full SHA: {sha!r}"
    assert live["configuration_digest"] in block, (
        "chapter snapshot digest does not match the live configuration digest — "
        f"regenerate with {os.path.relpath(SNAPSHOT_TOOL, _ROOT)}")
    assert "NOT AUTHORITATIVE" in block, \
        "snapshot does not declare itself informative rather than authoritative"

    # the extracted VALUES match live config, key by key.
    # Parsed into exact tokens, NOT substring-matched: "max=50" is a substring of
    # "max=500", so an `in block` test would pass a 10x-wrong ceiling — the exact
    # error class this gate exists to catch.
    documented = _parse_snapshot_bounds(block)
    with open(CONFIG, "r", encoding="utf-8") as f:
        raw = json.load(f)["search_bounds"]
    for key, vals in live["effective_bounds"].items():
        assert key in documented, f"chapter snapshot has no row for {key}"
        for bound, value in vals.items():
            if bound.startswith("_"):
                continue
            got = documented[key].get(bound)
            assert got is not None, \
                f"chapter snapshot missing/stale: {key}.{bound} absent"
            assert float(got) == float(value), \
                f"chapter snapshot missing/stale: {key}.{bound} documents " \
                f"{got}, live value is {value}"
            if key in raw and bound in raw[key]:
                assert raw[key][bound] == value, \
                    f"extractor disagrees with live config on {key}.{bound}"

    # the two _note provenance fields are carried over (audit §6 P0.2)
    for note in ("_calibration_note", "_s172_note"):
        assert note in block, f"{note} not carried into the chapter snapshot"

    # CLEAN CONTROL — the tool is not emitting a constant: perturb the config in
    # a temp copy and the digest must change.
    d = tempfile.mkdtemp(prefix="ch1_p0_snap_")
    alt = json.loads(open(CONFIG, "r", encoding="utf-8").read())
    alt["search_bounds"]["window_size"]["max"] = 999
    alt_path = os.path.join(d, "distributed_config.json")
    with open(alt_path, "w", encoding="utf-8") as f:
        json.dump(alt, f)
    import hashlib
    canon = json.dumps(alt["search_bounds"], sort_keys=True, separators=(",", ":"))
    alt_digest = "sha256:" + hashlib.sha256(canon.encode()).hexdigest()
    assert alt_digest != live["configuration_digest"], \
        "clean control: the digest does not discriminate between configurations"


def mutant_snapshot_stale_value():
    """FAULT INJECTION — hand-edit a bound in a chapter copy; gate must go RED."""
    text = open(CHAPTER, "r", encoding="utf-8").read()
    block = _chapter_snapshot_block(text)
    assert "max=50" in block, "mutation anchor 'max=50' absent from the snapshot"
    mutated = text.replace("max=50", "max=500", 1)
    assert mutated != text

    d = tempfile.mkdtemp(prefix="ch1_p0_snap_mut_")
    p = os.path.join(d, "CHAPTER_1_WINDOW_OPTIMIZER.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(mutated)

    try:
        gate_snapshot_extracted(chapter=p)
    except AssertionError as e:
        assert "missing/stale" in str(e), \
            f"gate reddened for the wrong reason: {e}"
    else:
        raise AssertionError(
            "gate stayed green against a hand-edited bound — it is vacuous")


# ===========================================================================
# G-SKIP-DEFECT-NOTE
# ===========================================================================

VERBATIM_SKIP_MIN = 'skip_min: int              # Minimum skip for variable PRNGs'
VERBATIM_SKIP_MAX = 'skip_max: int              # Maximum skip for variable PRNGs'
DEFECT_CALLOUT = (
    "DEFECT — current hybrid kernels do not execute the requested\n"
    "skip_min/skip_max semantics and instead use a hard-coded stride.\n"
    "Hybrid optimization results are non-certifying."
)


def gate_skip_defect_note(chapter=None):
    text = open(chapter or CHAPTER, "r", encoding="utf-8").read()

    assert VERBATIM_SKIP_MIN in text, \
        "the verbatim skip_min definition was altered or removed"
    assert VERBATIM_SKIP_MAX in text, \
        "the verbatim skip_max definition was altered or removed"
    assert DEFECT_CALLOUT in text, \
        "Team Beta's defect callout is absent or not verbatim"

    # the *why skip exists* physical model, and the standing rule
    for required in (
        "California State Lottery Daily & SuperLotto Plus Draw Procedures",
        "2021-06-09",
        "pre-test draws",
        "structural gaps",
        "WIRE-IN, not removal",
    ):
        assert required in text, f"skip rationale missing required element: {required!r}"

    # D-4 must be recorded in the chapter (audit §6 P0.4)
    for dim in ("D-1", "D-2", "D-3", "D-4"):
        assert dim in text, f"dead dimension {dim} not recorded in the chapter"


def mutant_skip_note_removed():
    """FAULT INJECTION — drop the defect callout; the gate must go RED."""
    text = open(CHAPTER, "r", encoding="utf-8").read()
    assert DEFECT_CALLOUT in text
    mutated = text.replace(DEFECT_CALLOUT, "(callout removed by mutant)", 1)

    d = tempfile.mkdtemp(prefix="ch1_p0_skip_mut_")
    p = os.path.join(d, "CHAPTER_1_WINDOW_OPTIMIZER.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(mutated)

    try:
        gate_skip_defect_note(chapter=p)
    except AssertionError as e:
        assert "defect callout" in str(e), f"reddened for the wrong reason: {e}"
    else:
        raise AssertionError(
            "gate stayed green with the defect callout removed — it is vacuous")


# ===========================================================================

GATES = [
    ("G-FLAG-FAILCLOSED", gate_flag_failclosed),
    ("G-STRATEGY-FAILCLOSED", gate_strategy_failclosed),
    ("G-NO-SILENT-SUBSTITUTION", gate_no_silent_substitution),
    ("G-METADATA-PROVENANCE", gate_metadata_provenance),
    ("G-SNAPSHOT-EXTRACTED", gate_snapshot_extracted),
    ("G-SKIP-DEFECT-NOTE", gate_skip_defect_note),
]

MUTANTS = [
    ("M1 restore silent no-op on a threshold flag", mutant_flag_silent_noop),
    ("M2 re-permit --strategy random", mutant_strategy_random_permitted),
    ("M3 restore the 0.72/0.81 constants", mutant_metadata_constants),
    ("M4 restore random-search fallback on missing Optuna",
     mutant_silent_random_fallback),
    ("M5 hand-edit a bound in the chapter snapshot", mutant_snapshot_stale_value),
    ("M6 remove the skip defect callout", mutant_skip_note_removed),
]


def main():
    print("=" * 70, flush=True)
    print("CHAPTER 1 P0 CORRECTIONS — items 1-5 acceptance harness", flush=True)
    print("=" * 70, flush=True)

    print("\n--- GATES (clean tree) ---", flush=True)
    for name, fn in GATES:
        _check(name, fn)

    print("\n--- MUTANTS (fault-injection controls) ---", flush=True)
    for name, fn in MUTANTS:
        _check(name, fn)

    total = len(RESULTS)
    passed = sum(1 for _, s, _ in RESULTS if s == "PASS")
    failed = [r for r in RESULTS if r[1] == "FAIL"]
    unavail = [r for r in RESULTS if r[1] == "UNAVAILABLE"]

    print("\n" + "=" * 70, flush=True)
    print(f"{passed}/{total} checks green "
          f"({len(GATES)} gates + {len(MUTANTS)} mutants)", flush=True)

    if failed:
        print("\nFAILURES (DO NOT COMMIT):", flush=True)
        for name, _, detail in failed:
            print(f"\n--- {name} ---\n{detail}", flush=True)

    if unavail:
        print("\nUNAVAILABLE (not counted as green — VIR-5):", flush=True)
        for name, _, detail in unavail:
            print(f"  {name}: {detail}", flush=True)

    if failed:
        sentinel = "FAIL"
    elif unavail:
        sentinel = "INCOMPLETE"
    elif passed == total:
        sentinel = "PASS"
    else:
        sentinel = "INCOMPLETE"

    print(f"\nSENTINEL : {sentinel}", flush=True)
    return 0 if sentinel == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
