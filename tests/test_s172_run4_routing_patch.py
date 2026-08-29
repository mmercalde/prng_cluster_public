#!/usr/bin/env python3
"""S172 RUN-4 ROUTE-A ROUTING PATCH — acceptance suite.

Governing artifacts:
  docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md          (§5, ten requirements)
  docs/TB_RULING_RUN4_ROUTING_PATCH_BRIEF_REVIEW.md           (Blockers 1 and 2)
  docs/S172_RUN4_ROUTING_PATCH_BRIEF_ROUTE_A.md               (revision 2)

Every gate terminates PASS | FAIL | UNAVAILABLE | INCOMPLETE; only PASS accepts
(VIR-3). No gate asserts a tally. Mutants are APPLIED, EXECUTED, DETECTED and
rebound against PRODUCTION module globals (the A8-B2 escape).
"""
import ast
import copy
import hashlib
import importlib
import itertools
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from fixtures import run4_routing_clean_control as FX          # noqa: E402
from fixtures import run4_pin_harness as H                     # noqa: E402

import agents.watcher_agent as W                               # noqa: E402

CONTROL = os.path.join(HERE, "fixtures", "run4_clean_control_69ca910.txt")
SEVEN = set(FX.SEVEN)
PINS = dict(FX.PIN_VALUES)
RESULTS = []


def record(name, verdict, detail=""):
    RESULTS.append((name, verdict, detail))
    print(f"  [{verdict:10s}] {name}" + (f"  — {detail}" if detail else ""))
    return verdict == "PASS"


def load_control():
    with open(CONTROL) as fh:
        return json.load(fh)


# ═══════════════════════════════════════════════════════════════════════════
# GATES
# ═══════════════════════════════════════════════════════════════════════════

def g_allowlist_exact():
    """8b — the allowlist is an exact literal seven, never derived."""
    got = set(W.STEP1_EXPLICIT_PIN_KEYS)
    if got != SEVEN:
        return record("G-ALLOWLIST-EXACT", "FAIL", f"set mismatch: {got ^ SEVEN}")
    hazardous = {"forward_threshold", "reverse_threshold",
                 "search_strategy", "seed_count"}
    leaked = hazardous & got
    if leaked:
        return record("G-ALLOWLIST-EXACT", "FAIL", f"hazardous names present: {leaked}")
    if "warm_start_session" in got:
        return record("G-ALLOWLIST-EXACT", "FAIL", "eighth key in allowlist")

    # AST arm: the constant is a literal frozenset of Str, not built from data.
    src = open(os.path.join(ROOT, "agents/watcher_agent.py")).read()
    tree = ast.parse(src)
    node = None
    for n in ast.walk(tree):
        if (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "STEP1_EXPLICIT_PIN_KEYS"):
            node = n.value
    if node is None:
        return record("G-ALLOWLIST-EXACT", "INCOMPLETE", "assignment not found")
    ok = (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
          and node.func.id == "frozenset" and len(node.args) == 1
          and isinstance(node.args[0], ast.Set)
          and all(isinstance(e, ast.Constant) and isinstance(e.value, str)
                  for e in node.args[0].elts))
    if not ok:
        return record("G-ALLOWLIST-EXACT", "FAIL",
                      "constant is not a literal frozenset of strings")
    for bad in ("args_map", "default_params", "json", "manifest"):
        if bad in ast.dump(node):
            return record("G-ALLOWLIST-EXACT", "FAIL", f"derived from {bad}")
    return record("G-ALLOWLIST-EXACT", "PASS",
                  f"7 literals, 4 hazardous names absent, eighth key absent")


def g_unpinned_identical():
    """4 — unpinned argv list-equal to the STEP-0 pre-edit capture."""
    try:
        ctl = load_control()
    except Exception as e:
        return record("G-UNPINNED-IDENTICAL", "UNAVAILABLE", f"control unreadable: {e}")
    if FX.fixture_sha256() != ctl["fixture_module_sha256"]:
        return record("G-UNPINNED-IDENTICAL", "INCOMPLETE",
                      "fixture module changed since capture — oracle invalid")
    try:
        got = FX.build_argv(W, FX.CONTROL_PARAMS)
    except RuntimeError as e:
        return record("G-UNPINNED-IDENTICAL", "INCOMPLETE", str(e))
    want = ctl["argv_unpinned"]
    if got != want:
        diff = [(i, a, b) for i, (a, b) in
                enumerate(itertools.zip_longest(got, want)) if a != b]
        return record("G-UNPINNED-IDENTICAL", "FAIL", f"argv diverged at {diff[:4]}")
    return record("G-UNPINNED-IDENTICAL", "PASS",
                  f"list-equal, {len(got)} tokens, vs {ctl['base_commit'][:7]}")


def g_chain():
    """1 — all seven survive WATCHER --params -> argv."""
    argv = H.build_argv_with_pin(W, FX.CONTROL_PARAMS, PINS)
    pairs = H.warm_pairs(argv)
    if set(pairs) != SEVEN:
        return record("G-CHAIN", "FAIL", f"routed {sorted(pairs)}")
    if len([v for v in PINS.values() if v is not None]) != 7:
        return record("G-CHAIN", "INCOMPLETE", "fixture did not supply seven values")
    return record("G-CHAIN", "PASS", f"7/7 routed to argv ({len(argv)} tokens)")


def g_exact():
    """2 — value AND type equality; never object identity."""
    argv = H.build_argv_with_pin(W, FX.CONTROL_PARAMS, PINS)
    pairs = H.warm_pairs(argv)
    bad = []
    for k, want in PINS.items():
        raw = pairs.get(k)
        if raw is None:
            bad.append(f"{k}: absent")
            continue
        # argv carries str(value); parse back in the declared type and require
        # both value AND type equality. `is` is deliberately never used.
        got = type(want)(raw) if not isinstance(want, bool) else (raw == "True")
        if type(got) is not type(want) or got != want:
            bad.append(f"{k}: {got!r}({type(got).__name__}) != {want!r}({type(want).__name__})")
    if bad:
        return record("G-EXACT", "FAIL", "; ".join(bad))
    if pairs["warm_start_session_idx"] != "1":
        return record("G-EXACT", "FAIL",
                      f"session_idx={pairs['warm_start_session_idx']} (the :780 default is 0)")
    return record("G-EXACT", "PASS",
                  "7/7 value+type equal; session_idx==1 as int; thresholds 0.71/0.47 intact")


def g_partial_closed():
    """5 — every one of the 126 proper non-empty subsets fails closed."""
    keys = sorted(SEVEN)
    checked = 0
    for r in range(1, 7):
        for combo in itertools.combinations(keys, r):
            params = dict(FX.CONTROL_PARAMS)
            params.update({k: PINS[k] for k in combo})
            try:
                W.capture_step1_pin_bundle(params)
            except W.Step1PinBundleError as e:
                msg = str(e)
                if not all(k in msg for k in combo):
                    return record("G-PARTIAL-CLOSED", "FAIL",
                                  f"error omits supplied keys for {combo}")
                checked += 1
                continue
            return record("G-PARTIAL-CLOSED", "FAIL", f"partial pin ACCEPTED: {combo}")
    if checked != 126:
        return record("G-PARTIAL-CLOSED", "INCOMPLETE", f"only {checked} subsets exercised")
    # The six-key subset that drops session_idx is the dangerous one (brief §3.4)
    six = tuple(k for k in keys if k != "warm_start_session_idx")
    p = dict(FX.CONTROL_PARAMS); p.update({k: PINS[k] for k in six})
    try:
        W.capture_step1_pin_bundle(p)
        return record("G-PARTIAL-CLOSED", "FAIL", "six-key/no-session_idx accepted")
    except W.Step1PinBundleError:
        pass
    return record("G-PARTIAL-CLOSED", "PASS",
                  "126/126 subsets rejected, incl. the session_idx-dropping six")


def g_eighth():
    """5b — warm_start_session never routes, pinned or unpinned."""
    for label, bundle in (("unpinned", None), ("pinned", PINS)):
        params = dict(FX.CONTROL_PARAMS)
        params["warm_start_session"] = "midday"
        argv = (FX.build_argv(W, params) if bundle is None
                else H.build_argv_with_pin(W, params, bundle))
        if any(t.startswith("--warm-start-session") and not
               t.startswith("--warm-start-session-idx") for t in argv):
            return record("G-EIGHTH", "FAIL", f"eighth key routed on {label} path")
    return record("G-EIGHTH", "PASS", "absent on both paths")


def g_no_synth():
    """6 — retry carry-forward cannot synthesize; a real bundle may replay."""
    # (a) WATCHER manufactures the seven into retry_params; frozen record empty.
    synthesized = dict(FX.CONTROL_PARAMS)
    synthesized.update(PINS)                       # as if retry/LLM injected them
    argv = FX.build_argv(W, synthesized)           # no _pin_bundle threaded
    if H.warm_pairs(argv):
        return record("G-NO-SYNTH", "FAIL", "synthesized values reached argv")
    # (b) an operator who DID supply all seven replays legitimately
    argv2 = H.build_argv_with_pin(W, synthesized, PINS)
    if set(H.warm_pairs(argv2)) != SEVEN:
        return record("G-NO-SYNTH", "FAIL", "legitimate replay was blocked")
    return record("G-NO-SYNTH", "PASS",
                  "synthesis routes 0/7; operator-supplied bundle replays 7/7")


def g_no_llm():
    """7 — LLM proposals cannot create these keys."""
    for step in (1, 5):
        proposed = {k: PINS[k] for k in SEVEN}
        bundle = W.capture_step1_pin_bundle(proposed)   # complete, but LLM-origin
        # The authority is the INVOCATION bundle, not any dict shaped like one:
        # run_pipeline captures before the LLM loop, so an LLM-built dict is
        # never the bundle threaded to run_step.
        if W._step1_explicit_pin(step, None) is not None:
            return record("G-NO-LLM", "FAIL", f"pin asserted with no bundle at step {step}")
        if step != 1 and W._step1_explicit_pin(step, bundle) is not None:
            return record("G-NO-LLM", "FAIL", f"pin asserted on step {step}")
    argv = FX.build_argv(W, {**FX.CONTROL_PARAMS, **PINS})
    if H.warm_pairs(argv):
        return record("G-NO-LLM", "FAIL", "LLM-shaped params reached argv unthreaded")
    return record("G-NO-LLM", "PASS", "no bundle -> no pin, step-scoped, 0/7 routed")


def g_provenance():
    """8 — marker present pinned, ABSENT unpinned."""
    import logging

    class Cap(logging.Handler):
        def __init__(self): super().__init__(); self.lines = []
        def emit(self, r): self.lines.append(r.getMessage())

    out = {}
    for label, bundle in (("unpinned", None), ("pinned", PINS)):
        cap = Cap()
        W.logger.addHandler(cap)
        try:
            if bundle is None:
                FX.build_argv(W, FX.CONTROL_PARAMS)
            else:
                H.build_argv_with_pin(W, FX.CONTROL_PARAMS, bundle)
        finally:
            W.logger.removeHandler(cap)
        out[label] = [l for l in cap.lines if "step1_pin_source" in l]
    if out["unpinned"]:
        return record("G-PROVENANCE", "FAIL",
                      f"marker present on unpinned run: {out['unpinned'][:1]}")
    if not out["pinned"]:
        return record("G-PROVENANCE", "FAIL", "marker absent on pinned run")
    if W.STEP1_PIN_SOURCE_MARKER not in " ".join(out["pinned"]):
        return record("G-PROVENANCE", "FAIL", "marker value wrong")

    # ---- structured-result arm, both directions (brief §4) ----------------
    # The log record alone leaves the acceptance record dependent on scraping
    # text. Brief §4 promises the marker in the step's STRUCTURED RESULT too.
    kmark, kargv = W.STEP1_PIN_PROVENANCE_KEY, W.STEP1_PIN_ARGV_KEY

    argv_p, res_p = H.build_result(W, FX.CONTROL_PARAMS, PINS)
    if not isinstance(res_p, dict):
        return record("G-PROVENANCE", "INCOMPLETE",
                      f"pinned dispatch returned {type(res_p).__name__}, not a result dict")
    if kmark not in res_p:
        return record("G-PROVENANCE", "FAIL",
                      f"structured result carries no {kmark} on the pinned path")
    if res_p[kmark] != W.STEP1_PIN_SOURCE_MARKER:
        return record("G-PROVENANCE", "FAIL",
                      f"structured {kmark}={res_p[kmark]!r}")
    if kargv not in res_p:
        return record("G-PROVENANCE", "FAIL",
                      f"structured result carries no {kargv} — the marker proves "
                      "authority but nothing records what it requested")
    if not isinstance(res_p[kargv], list):
        return record("G-PROVENANCE", "FAIL",
                      f"{kargv} is {type(res_p[kargv]).__name__}, not a list")
    # A FAITHFUL record of the argv actually built — deliberately NOT an
    # assertion that warm-start flags are present. Authority and routing are
    # separate facts; routing is G-CHAIN's. This is what keeps the gate green
    # under M1b, where the pin is genuinely authorized and genuinely stripped.
    if res_p[kargv] != argv_p:
        return record("G-PROVENANCE", "FAIL",
                      f"{kargv} does not match the argv actually built")

    argv_u, res_u = H.build_result(W, FX.CONTROL_PARAMS, None)
    if not isinstance(res_u, dict):
        return record("G-PROVENANCE", "INCOMPLETE",
                      f"unpinned dispatch returned {type(res_u).__name__}, not a result dict")
    present = [k for k in (kmark, kargv) if k in res_u]
    if present:
        # ABSENT, not None: `in` is the test, so a None placeholder fails here.
        return record("G-PROVENANCE", "FAIL",
                      f"unpinned structured result carries {present} "
                      f"(values {[res_u[k] for k in present]!r})")

    return record("G-PROVENANCE", "PASS",
                  f"log + structured result: {kmark}/{kargv} present pinned "
                  f"({len(res_p[kargv])}-token argv recorded), both keys ABSENT "
                  "(not null/empty) unpinned")


def _drive_pipeline(agent, params, operator_pin=None):
    """Run one run_pipeline invocation, capturing the argv its run_step builds.

    `operator_pin` goes through the OPERATOR-AUTHORITY CHANNEL (R1 Blocker 1),
    exactly as the real CLI seam supplies it. Ordinary `params` never carries
    the seven on a legitimate invocation, because the CLI MOVES them.
    """
    seen = {}
    real = W.WatcherAgent.run_step

    def spy(self, step, p=None, *, _pin_bundle=None):
        # Record what run_pipeline threaded, then build the argv through the
        # GENUINE run_step. The spy must be un-installed for that call: the
        # harness resolves `WatcherAgent.run_step` at call time, so leaving the
        # spy installed makes it capture itself and recurse.
        seen["bundle"] = _pin_bundle
        W.WatcherAgent.run_step = real
        try:
            seen["argv"] = H.build_argv_with_pin(W, p, _pin_bundle)
        finally:
            W.WatcherAgent.run_step = spy
        return {"success": True, "skipped": False}

    W.WatcherAgent.run_step = spy
    saved = {}
    for n, v in (("check_safety", lambda: True),
                 ("notify_telegram", lambda *a, **k: None),
                 ("PROGRESS_DISPLAY_AVAILABLE", False),
                 ("check_training_health", lambda *a, **k: {"action": "PROCEED"})):
        saved[n] = getattr(W, n, None)
        setattr(W, n, v)
    agent._pipeline_running = True
    agent._get_step_trials = lambda *a, **k: 1
    _dec = types.SimpleNamespace(
        recommended_action="proceed", confidence=1.0, reasoning="gate",
        warnings=[], suggested_param_adjustments={})
    agent.evaluate_results = lambda *a, **k: (_dec, {})
    # False -> `should_continue` is false -> the loop breaks after ONE step.
    # Exactly one dispatch per invocation is what this gate needs.
    agent.execute_decision = lambda *a, **k: False
    try:
        seen["returned"] = W.WatcherAgent.run_pipeline(
            agent, 1, 1, params, _operator_pin_params=operator_pin)
    finally:
        W.WatcherAgent.run_step = real
        for n, v in saved.items():
            setattr(W, n, v)
    return seen


def g_invocation_isolation():
    """10b — two run_pipeline invocations on ONE agent; no leakage."""
    agent = W.WatcherAgent.__new__(W.WatcherAgent)
    agent.config = types.SimpleNamespace(
        manifests_dir=os.path.join(ROOT, "agent_manifests"),
        force_steps=set(), get_step_timeout_minutes=lambda s: 600,
        daemon_state_file=os.path.join(ROOT, "daemon_state.json"))
    agent.retry_counts = {}
    agent.current_step = 1

    # The CLI seam MOVES the seven, so a legitimate pinned invocation carries
    # ordinary params WITHOUT them plus the authority channel WITH them.
    ordinary, authority = W.split_operator_pin_params({**FX.CONTROL_PARAMS, **PINS})
    first = _drive_pipeline(agent, ordinary, authority)
    if first.get("bundle") is None:
        return record("G-INVOCATION-ISOLATION", "INCOMPLETE",
                      "first invocation carried no bundle — nothing to leak")
    if set(H.warm_pairs(first["argv"])) != SEVEN:
        return record("G-INVOCATION-ISOLATION", "FAIL", "pinned invocation routed < 7")

    agent.current_step = 1
    second = _drive_pipeline(agent, dict(FX.CONTROL_PARAMS))
    if second.get("bundle") is not None:
        return record("G-INVOCATION-ISOLATION", "FAIL",
                      "second invocation inherited a pin bundle")
    if H.warm_pairs(second["argv"]):
        return record("G-INVOCATION-ISOLATION", "FAIL",
                      "warm-start routed on the second, unpinned invocation")

    leaked = [a for a in vars(agent)
              if any(k in str(getattr(agent, a, "")) for k in SEVEN)]
    if leaked:
        return record("G-INVOCATION-ISOLATION", "FAIL",
                      f"instance attributes hold pins: {leaked}")
    try:
        with open(agent.config.daemon_state_file) as fh:
            blob = fh.read()
        if any(k in blob for k in SEVEN):
            return record("G-INVOCATION-ISOLATION", "FAIL", "pins persisted to daemon state")
        state = "daemon_state read, clean"
    except FileNotFoundError:
        state = "daemon_state absent"
    except Exception as e:
        return record("G-INVOCATION-ISOLATION", "UNAVAILABLE", f"state unreadable: {e}")
    return record("G-INVOCATION-ISOLATION", "PASS",
                  f"invocation 1 routes 7/7, invocation 2 routes 0/7; {state}")


def g_one_trial():
    """3 — window_trials=1 reaches the optimizer as --trials 1 (one pinned trial)."""
    argv = H.build_argv_with_pin(W, FX.CONTROL_PARAMS, PINS)
    if "--trials" not in argv:
        return record("G-ONE-TRIAL", "INCOMPLETE", "--trials absent from argv")
    if argv[argv.index("--trials") + 1] != "1":
        return record("G-ONE-TRIAL", "FAIL",
                      f"--trials {argv[argv.index('--trials') + 1]}")
    return record("G-ONE-TRIAL", "PASS",
                  "--trials 1 with all seven pinned; the single trial is the enqueued one "
                  "(enqueue is window_optimizer_bayesian.py:774-786, unchanged by this patch)")


def g_origin():
    """R1 BLOCKER 1 — authority is ORIGIN, not presence.

    The seven in ordinary `run_pipeline(params=...)` with no authority must
    route 0/7 and carry no provenance; the same seven through the operator
    authority channel must route 7/7 with provenance. This is the distinction
    the previous patch could not make: it inferred authority from the presence
    of a complete bundle in generic params, which a live programmatic caller
    (`chapter_13_triggers.py:616`) can also produce.
    """
    def _fresh_agent():
        a = W.WatcherAgent.__new__(W.WatcherAgent)
        a.config = types.SimpleNamespace(
            manifests_dir=os.path.join(ROOT, "agent_manifests"),
            force_steps=set(), get_step_timeout_minutes=lambda s: 600,
            daemon_state_file=os.path.join(ROOT, "daemon_state.json"))
        a.retry_counts = {}
        a.current_step = 1
        return a

    # ---- arm 1: seven in ordinary params, NO authority --------------------
    impostor = {**FX.CONTROL_PARAMS, **PINS}
    unauth = _drive_pipeline(_fresh_agent(), impostor, None)
    if unauth.get("bundle") is not None:
        return record("G-ORIGIN", "FAIL",
                      "a caller acquired pin authority from ordinary params")
    if unauth.get("argv") and H.warm_pairs(unauth["argv"]):
        return record("G-ORIGIN", "FAIL",
                      "warm-start flags routed without operator authority")
    ret = unauth.get("returned") or {}
    # Beta prefers fail-LOUD; the essential property is that authority is never
    # acquired. Assert both: blocked, and blocked for the stated reason.
    if ret.get("blocked_by") != "step1_unauthorized_warm_start_pin":
        return record("G-ORIGIN", "FAIL",
                      f"unauthorized pin was not failed loud: blocked_by="
                      f"{ret.get('blocked_by')!r}")
    if W.STEP1_PIN_PROVENANCE_KEY in ret:
        return record("G-ORIGIN", "FAIL", "provenance stamped on a blocked run")

    # ---- arm 2: the same seven through the authority channel --------------
    ordinary, authority = W.split_operator_pin_params(dict(impostor))
    if set(authority) != SEVEN:
        return record("G-ORIGIN", "INCOMPLETE",
                      f"CLI seam did not move the seven: {sorted(authority)}")
    if SEVEN & set(ordinary):
        return record("G-ORIGIN", "FAIL",
                      f"seam DUPLICATED instead of MOVED: {sorted(SEVEN & set(ordinary))}")
    auth_run = _drive_pipeline(_fresh_agent(), ordinary, authority)
    if auth_run.get("bundle") is None:
        return record("G-ORIGIN", "FAIL", "authority channel granted no bundle")
    if set(H.warm_pairs(auth_run["argv"])) != SEVEN:
        return record("G-ORIGIN", "FAIL",
                      f"authorized run routed {sorted(H.warm_pairs(auth_run['argv']))}")

    # ---- arm 3: the two live programmatic callers default to no authority --
    import inspect
    sig = inspect.signature(W.WatcherAgent.run_pipeline)
    prm = sig.parameters.get("_operator_pin_params")
    if prm is None or prm.default is not None:
        return record("G-ORIGIN", "FAIL",
                      "authority channel is not a defaulted keyword — existing "
                      "callers would not default to zero authority")
    if prm.kind is not inspect.Parameter.KEYWORD_ONLY:
        return record("G-ORIGIN", "FAIL",
                      "authority channel is positional — a caller could supply "
                      "it by accident")
    return record("G-ORIGIN", "PASS",
                  "ordinary params -> 0/7 + fail-loud + no provenance; "
                  "authority channel -> 7/7 + provenance; channel is "
                  "keyword-only defaulting to None")


def g_value_usable():
    """R1 BLOCKER 2 — present-but-non-routable values fail loud.

    G-PARTIAL-CLOSED tests ABSENT keys. This sibling tests keys that are
    PRESENT but carry a value the `:2009-2018` command builder treats as
    absent, which previously produced a complete "authorized" bundle that
    logged a seven-key pin while the builder routed fewer, or zero.
    """
    cases = []
    keys = sorted(SEVEN)

    # 7 individually None, 7 individually '' (Beta's stated minimum), plus the
    # two bool cases the builder's OTHER branch drops or mangles.
    for bad, label in ((None, "None"), ("", "empty"), (False, "False"), (True, "True")):
        for k in keys:
            t = dict(PINS)
            t[k] = bad
            cases.append((f"{k}={label}", t))
    # all-seven variants
    for bad, label in ((None, "None"), ("", "empty")):
        cases.append((f"all-seven={label}", {k: bad for k in keys}))

    accepted = []
    for label, bundle in cases:
        try:
            got = W.capture_step1_pin_bundle(bundle)
        except W.Step1PinBundleError:
            continue
        # Silent collapse to "unpinned" is ALSO a failure: the ruling requires
        # a malformed explicit pin to fail, not to vanish.
        accepted.append(f"{label} -> {'ACCEPTED' if got else 'silently unpinned'}")
    if accepted:
        return record("G-VALUE-USABLE", "FAIL",
                      f"{len(accepted)} non-routable bundle(s) not rejected: {accepted[:4]}")
    if len(cases) < 15:
        return record("G-VALUE-USABLE", "INCOMPLETE",
                      f"only {len(cases)} negative cases (ruling requires >= 15)")

    # Non-vacuity: the very same shape with usable values must be ACCEPTED,
    # or this gate would pass on a capture that rejects everything.
    ok = W.capture_step1_pin_bundle(dict(PINS))
    if ok is None or set(ok) != SEVEN:
        return record("G-VALUE-USABLE", "INCOMPLETE",
                      "the valid control bundle was not accepted — the "
                      "rejections above prove nothing")
    if list(ok) != sorted(SEVEN):
        return record("G-VALUE-USABLE", "FAIL",
                      f"bundle key order is not deterministic: {list(ok)}")
    return record("G-VALUE-USABLE", "PASS",
                  f"{len(cases)} present-but-non-routable bundles rejected "
                  f"(7 None, 7 '', 7 False, 7 True, 2 all-seven); valid control "
                  "accepted in sorted key order")


def _capture_resolution_kwargs(params):
    """Drive the REAL `_ensure_execution_set` and return the kwargs it hands to
    `resolve_execution_set`. `_ensure_execution_set` imports from
    `execution_set` INSIDE the function, so patching the module attributes
    reaches the call site. Nothing contacts the fleet."""
    import execution_set as ES

    seen = {}
    names = ("_peek_execution_set", "resolve_execution_set", "freeze_execution_set")
    saved = {n: getattr(ES, n) for n in names}

    # `_ensure_execution_set` logs `s.describe()` on the frozen set, so the
    # sentinel must satisfy that surface — a bare string makes the gate raise
    # on the stub rather than on the property under test.
    sentinel = types.SimpleNamespace(describe=lambda: "parity-lock-sentinel")

    def _resolve(**kw):
        seen.update(kw)
        return sentinel

    ES._peek_execution_set = lambda: None
    ES.resolve_execution_set = _resolve
    ES.freeze_execution_set = lambda s: s
    try:
        agent = W.WatcherAgent.__new__(W.WatcherAgent)
        agent.config = types.SimpleNamespace(
            manifests_dir=os.path.join(ROOT, "agent_manifests"))
        W.WatcherAgent._ensure_execution_set(agent, params)
    finally:
        for n, v in saved.items():
            setattr(ES, n, v)
    return seen


def g_parity_inert():
    """PARITY REGRESSION LOCK — the mirror is live, and provably cannot move
    execution-set resolution.

    `_step1_declared_params` mirrors the seven so the two notions of "declared"
    cannot drift (brief §3.1). That mirror runs on ANY params dict carrying the
    seven, with or without pin authority, so its inertness for fleet resolution
    is load-bearing and must be locked by gate rather than by inspection.

    Three arms. Arm 1: the mirror is real (not dead code). Arm 2: the kwargs
    actually handed to `resolve_execution_set` are identical with and without
    the seven. Arm 3 is this gate's own fault-injection control — a key that
    SHOULD move resolution does move it, so arm 2's equality is not vacuous.
    """
    agent = W.WatcherAgent.__new__(W.WatcherAgent)
    agent.config = types.SimpleNamespace(
        manifests_dir=os.path.join(ROOT, "agent_manifests"))

    plain = dict(FX.CONTROL_PARAMS)
    seven = {**FX.CONTROL_PARAMS, **PINS}

    # ---- arm 1: parity holds — the mirror is not dead ----------------------
    d_plain = W.WatcherAgent._step1_declared_params(agent, plain)
    d_seven = W.WatcherAgent._step1_declared_params(agent, seven)
    mirrored = {k: d_seven.get(k) for k in SEVEN if k in d_seven}
    if set(mirrored) != SEVEN:
        return record("G-PARITY-INERT", "FAIL",
                      f"mirror dropped keys: {sorted(SEVEN - set(mirrored))}")
    if any(mirrored[k] != PINS[k] for k in SEVEN):
        return record("G-PARITY-INERT", "FAIL", "mirror altered a value")
    leaked = SEVEN & set(d_plain)
    if leaked:
        return record("G-PARITY-INERT", "FAIL",
                      f"mirror invented keys the caller never supplied: {sorted(leaked)}")

    # ---- arm 2: resolution is byte-identical with and without the seven ----
    kw_plain = _capture_resolution_kwargs(plain)
    kw_seven = _capture_resolution_kwargs(seven)
    if not kw_plain:
        return record("G-PARITY-INERT", "INCOMPLETE",
                      "resolve_execution_set was never reached — nothing was compared")
    if kw_plain != kw_seven:
        diff = {k: (kw_plain.get(k), kw_seven.get(k))
                for k in set(kw_plain) | set(kw_seven)
                if kw_plain.get(k) != kw_seven.get(k)}
        return record("G-PARITY-INERT", "FAIL",
                      f"the seven moved execution-set resolution: {diff}")

    # ---- arm 3: fault-injection control — the comparison CAN see a change --
    moved = dict(FX.CONTROL_PARAMS)
    moved["use_range_miner"] = False
    moved["use_persistent_workers"] = True
    kw_moved = _capture_resolution_kwargs(moved)
    if kw_moved == kw_plain:
        return record("G-PARITY-INERT", "INCOMPLETE",
                      "control key changed nothing — arm 2's equality proves nothing")
    if kw_moved.get("backend") != "pwc" or kw_plain.get("backend") != "miner":
        return record("G-PARITY-INERT", "INCOMPLETE",
                      f"backend control did not behave: {kw_plain.get('backend')} "
                      f"-> {kw_moved.get('backend')}")

    return record("G-PARITY-INERT", "PASS",
                  f"mirror routes 7/7 into declared; resolve kwargs identical "
                  f"({sorted(kw_plain)}); backend miner->pwc control moves them")


# ═══════════════════════════════════════════════════════════════════════════
# MUTANTS — applied, executed, detected, rebound to production globals
# ═══════════════════════════════════════════════════════════════════════════

def _run_gate_under_mutant(gate):
    """Run one gate and return (verdict, detail).

    Verdict is the gate's OWN terminal verdict. An unexpected exception is
    reported as `RAISED` and is deliberately NOT a verdict: brief §7 makes
    "any exception counts as detection" the recorded vacuity failure (§2.44),
    so a raising gate can neither credit a mutant as detected nor be counted as
    having stayed green.
    """
    before = len(RESULTS)
    try:
        gate()
    except Exception as e:                               # noqa: BLE001
        del RESULTS[before:]
        return "RAISED", f"{type(e).__name__}: {e}"
    if len(RESULTS) <= before:
        del RESULTS[before:]
        return "INCOMPLETE", "gate recorded no verdict"
    _, verdict, detail = RESULTS[before]
    del RESULTS[before:]
    return verdict, detail


def _mutant(name, apply_fn, restore_fn, must_red, must_stay_green):
    print(f"\n  -- {name} --")
    apply_fn()
    applied = True
    try:
        red, undetected, raised = [], [], []
        for gate in must_red:
            v, d = _run_gate_under_mutant(gate)
            if v == "RAISED":
                raised.append((gate.__name__, d))
            elif v == "PASS":
                undetected.append((gate.__name__, v))
            else:
                red.append((gate.__name__, v))
        stayed, broke = [], []
        for gate in must_stay_green:
            v, d = _run_gate_under_mutant(gate)
            if v == "RAISED":
                raised.append((gate.__name__, d))
            elif v != "PASS":
                broke.append((gate.__name__, v))
            else:
                stayed.append((gate.__name__, v))
    finally:
        restore_fn()
    detail = (f"applied={applied} detected_by={[g for g, _ in red]} "
              f"still_green={[g for g, _ in stayed]}")
    if raised:
        # Vacuity guard: credit only a specific assertion failure. A gate that
        # blew up under the mutant proves nothing about what the gate detects.
        return record(name, "INCOMPLETE",
                      f"gate raised under mutant (not credited): {raised}")
    if undetected:
        return record(name, "FAIL", f"NOT DETECTED by {[g for g, _ in undetected]}")
    if broke:
        return record(name, "FAIL", f"reds a gate it must not: {broke}")
    if not red:
        return record(name, "FAIL", "mutant applied but no gate reddened")
    return record(name, "PASS", detail)


def m1():
    """Restore WALL 2 unconditional eight-name stripping."""
    orig = W.STEP1_EXPLICIT_PIN_KEYS
    return _mutant(
        "M1-unconditional-strip",
        lambda: setattr(W, "STEP1_EXPLICIT_PIN_KEYS", frozenset()),
        lambda: setattr(W, "STEP1_EXPLICIT_PIN_KEYS", orig),
        must_red=[g_chain, g_exact],
        must_stay_green=[g_unpinned_identical])


_WATCHER_SRC = os.path.join(ROOT, "agents", "watcher_agent.py")

_WALL2_LIVE = (
    "            if _step1_pin:\n"
    "                _INTERNAL_ONLY_PARAMS = (\n"
    "                    _INTERNAL_ONLY_PARAMS - STEP1_EXPLICIT_PIN_KEYS\n"
    "                )\n"
)
_WALL2_DEAD = (
    "            if False:  # M1b mutant: WALL 2 strips unconditionally\n"
    "                _INTERNAL_ONLY_PARAMS = (\n"
    "                    _INTERNAL_ONLY_PARAMS - STEP1_EXPLICIT_PIN_KEYS\n"
    "                )\n"
)

_M1B_STATE = {}


def _m1b_apply():
    """Kill ONLY WALL 2's narrowing, in source, leaving the allowlist and
    `capture_step1_pin_bundle` untouched. Line count is preserved so every
    other file:line anchor stays valid."""
    original = open(_WATCHER_SRC, "rb").read()
    _M1B_STATE["bytes"] = original
    _M1B_STATE["sha"] = hashlib.sha256(original).hexdigest()
    text = original.decode()
    if text.count(_WALL2_LIVE) != 1:
        raise RuntimeError(
            f"M1b anchor is not unique ({text.count(_WALL2_LIVE)} occurrences) "
            "— refusing to mutate production source")
    open(_WATCHER_SRC, "w").write(text.replace(_WALL2_LIVE, _WALL2_DEAD, 1))
    importlib.reload(W)


def _m1b_restore():
    """Write the ORIGINAL bytes back and prove it by digest. A mutation harness
    that cannot prove restoration has left the tree in an unknown state."""
    original = _M1B_STATE.get("bytes")
    if original is None:
        return
    open(_WATCHER_SRC, "wb").write(original)
    importlib.reload(W)
    now = hashlib.sha256(open(_WATCHER_SRC, "rb").read()).hexdigest()
    if now != _M1B_STATE["sha"]:
        raise RuntimeError(
            f"M1b RESTORE FAILED: {now} != {_M1B_STATE['sha']} — "
            "agents/watcher_agent.py is NOT back to its pre-mutant bytes")
    _M1B_STATE.clear()


def m1b():
    """WALL 2 alone reverts to the unconditional eight-name strip.

    Distinct from M1: the allowlist constant and `capture_step1_pin_bundle` are
    untouched, so G-ALLOWLIST-EXACT and G-PARTIAL-CLOSED must stay green. This
    is the surgical WALL-2 defect — and the dangerous shape is that WALL 1 still
    fires, so the pin is ACCEPTED and its provenance LOGGED while nothing
    routes. G-PROVENANCE staying green under M1b is what proves the routing
    gates, not the marker, are what certify routing.
    """
    return _mutant(
        "M1b-wall2-only-unconditional-strip",
        _m1b_apply, _m1b_restore,
        must_red=[g_chain, g_exact],
        must_stay_green=[g_unpinned_identical, g_allowlist_exact,
                         g_partial_closed, g_provenance])


def m2a():
    """Invocation-local authority becomes agent-lifetime authority."""
    orig = W.capture_step1_pin_bundle
    cache = {}

    def leaky(params):
        got = orig(params)
        if got is not None:
            cache["b"] = got
        return got if got is not None else cache.get("b")

    return _mutant(
        "M2a-lifetime-authority",
        lambda: setattr(W, "capture_step1_pin_bundle", leaky),
        lambda: (setattr(W, "capture_step1_pin_bundle", orig), cache.clear()),
        must_red=[g_invocation_isolation],
        must_stay_green=[g_chain, g_unpinned_identical])


def m2b():
    """Pins contaminate ordinary/default final_params."""
    orig = W._step1_explicit_pin
    return _mutant(
        "M2b-default-contamination",
        lambda: setattr(W, "_step1_explicit_pin",
                        lambda step, bundle: dict(PINS) if step == 1 else None),
        lambda: setattr(W, "_step1_explicit_pin", orig),
        must_red=[g_unpinned_identical, g_provenance],
        must_stay_green=[g_chain])


def m4():
    """R1 BLOCKER 1 mutant — revert the capture source to ordinary `params`.

    This is the pre-R1 defect exactly: `run_pipeline` ignores the authority
    channel and captures from generic `params`, so any caller that can put the
    seven in `params` is misclassified as an explicit operator. The fail-loud
    guard is disabled alongside it, because in the pre-R1 shape it did not
    exist — reverting only one half would be a strawman that no real regression
    could produce.
    """
    real_pipeline = W.WatcherAgent.run_pipeline
    real_assert = W.assert_no_unauthorized_pin_keys

    def reverted(self, start_step=1, end_step=6, params=None, *,
                 _operator_pin_params=None):
        # capture from ORDINARY params, ignoring the authority channel
        return real_pipeline(self, start_step, end_step, params,
                             _operator_pin_params=params)

    return _mutant(
        "M4-origin-from-ordinary-params",
        lambda: (setattr(W, "assert_no_unauthorized_pin_keys",
                         lambda params, context: None),
                 setattr(W.WatcherAgent, "run_pipeline", reverted)),
        lambda: (setattr(W, "assert_no_unauthorized_pin_keys", real_assert),
                 setattr(W.WatcherAgent, "run_pipeline", real_pipeline)),
        must_red=[g_origin],
        must_stay_green=[g_chain, g_unpinned_identical, g_value_usable])


def m3():
    """Allowlist derived from args_map orphans instead of literals."""
    orig = W.STEP1_EXPLICIT_PIN_KEYS
    derived = frozenset(SEVEN | {"forward_threshold", "reverse_threshold",
                                 "search_strategy", "seed_count"})
    return _mutant(
        "M3-derived-allowlist",
        lambda: setattr(W, "STEP1_EXPLICIT_PIN_KEYS", derived),
        lambda: setattr(W, "STEP1_EXPLICIT_PIN_KEYS", orig),
        must_red=[g_allowlist_exact],
        must_stay_green=[g_chain, g_exact])


# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 74)
    print("S172 RUN-4 ROUTE-A ROUTING PATCH — ACCEPTANCE SUITE")
    print("=" * 74)
    print("\nGATES")
    gates = [g_allowlist_exact, g_unpinned_identical, g_chain, g_exact,
             g_partial_closed, g_eighth, g_no_synth, g_no_llm, g_provenance,
             g_invocation_isolation, g_one_trial, g_parity_inert,
             g_origin, g_value_usable]
    for g in gates:
        try:
            g()
        except Exception as e:                           # noqa: BLE001
            import traceback; traceback.print_exc()
            record(g.__name__, "FAIL", f"raised {type(e).__name__}: {e}")

    print("\nMUTANTS")
    for m in (m1, m1b, m2a, m2b, m3, m4):
        try:
            m()
        except Exception as e:                           # noqa: BLE001
            import traceback; traceback.print_exc()
            record(m.__name__, "FAIL", f"raised {type(e).__name__}: {e}")

    passed = sum(1 for _, v, _ in RESULTS if v == "PASS")
    total = len(RESULTS)
    print("\n" + "=" * 74)
    for n, v, d in RESULTS:
        if v != "PASS":
            print(f"  NOT-PASS: {n} [{v}] {d}")
    print(f"  {passed}/{total} PASS")
    if passed == total:
        print("  S172_RUN4_ROUTING_SUITE_COMPLETE")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
