"""L-5 probe — WATCHER keeps its own INFO records after Step 1's comparison imports
window_optimizer. Run as a SUBPROCESS by tests/test_s172_small_repairs_part3.py.

  open <dir>                 open a PINNED Run-4 search (production opener, production
                             configuration) in a throwaway ledger under <dir>; print its
                             path. This process imports window_optimizer; the next does not.
  run <ledger> [--mutant]    a FRESH process: window_optimizer is not imported. One real
                             run_step(1) on the pinned path with dispatch intercepted
                             (the successor routing harness), against the ledger opened
                             above. Every log record reaching the root logger is captured.
                             Prints one JSON line. `--mutant` installs the BASE (7b9184a)
                             `step1_presented_configuration`, i.e. L-5 unrepaired.
"""
import json
import logging
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
BASE = "7b9184a95cd125c14685008cd306fe0fd9ff3040"


def _open(d):
    import agents.watcher_agent as W
    from fixtures import run4_routing_search_control as FXS
    from fixtures import run4_search_fixture as S
    os.makedirs(d, exist_ok=True)
    S._SCRATCH = d
    agent = W.WatcherAgent.__new__(W.WatcherAgent)
    import types
    agent.config = types.SimpleNamespace(
        manifests_dir=os.path.join(ROOT, "agent_manifests"), force_steps=set(),
        get_step_timeout_minutes=lambda step: 600)
    agent.retry_counts = {}
    S.open_run4_search(W, agent, dict(FXS.CONTROL_PARAMS), dict(FXS.PIN_VALUES))
    print(json.dumps({"ledger": S.SearchDB.ledger_path}))


class _Grab(logging.Handler):
    def __init__(self):
        super().__init__(logging.NOTSET)
        self.records = []

    def emit(self, record):
        self.records.append((record.name, record.levelname, record.getMessage()))


def _run(ledger, mutant):
    out = {"wo_imported_before": "window_optimizer" in sys.modules}
    import agents.watcher_agent as W
    from fixtures import run4_routing_search_control as FXS
    from fixtures import run4_search_fixture as S
    if mutant:
        import ast
        import textwrap
        src = subprocess.check_output(["git", "-C", ROOT, "show",
                                       f"{BASE}:agents/watcher_agent.py"]).decode()
        node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)
                    and n.name == "step1_presented_configuration")
        lines = src.splitlines(keepends=True)[node.lineno - 1:node.end_lineno]
        ns = {}
        exec(compile(textwrap.dedent("".join(lines)), "<base step1_presented_configuration>",
                     "exec"), W.__dict__, ns)
        W.step1_presented_configuration = ns["step1_presented_configuration"]
    out["wo_imported_after_watcher_import"] = "window_optimizer" in sys.modules
    root = logging.getLogger()
    grab = _Grab()
    root.addHandler(grab)
    out["root_level_before"] = root.level

    # Point the harness at the search opened by the OTHER process, instead of opening
    # one here (opening computes the configuration, which would import window_optimizer
    # before run_step does — exactly the ordering that hides L-5).
    def _use_existing(agent_mod, agent, params, pin_bundle):
        S.SearchDB.ledger_path = ledger

        def cleanup():
            S.SearchDB.ledger_path = None
        return cleanup
    S.open_run4_search = _use_existing

    real = W.WatcherAgent.run_step
    pins = dict(FXS.PIN_VALUES)

    def _wrapper(self, step, p=None, *, _pin_bundle=None):
        return real(self, step, p, _pin_bundle=pins)
    W.WatcherAgent.run_step = _wrapper
    S.SearchDB.pin_bundle = pins
    try:
        argv = FXS.build_argv(W, dict(FXS.CONTROL_PARAMS))
        out["argv_built"] = True
        out["argv_len"] = len(argv)
    except Exception as e:   # noqa: BLE001
        out["argv_built"] = False
        out["error"] = f"{type(e).__name__}: {e}"
    finally:
        W.WatcherAgent.run_step = real
        S.SearchDB.pin_bundle = None
    out["wo_imported_after_run_step"] = "window_optimizer" in sys.modules
    out["root_level_after"] = root.level
    msgs = [m for _n, _l, m in grab.records]
    out["exec_cmd_records"] = sum(1 for m in msgs if m.startswith("EXEC CMD:"))
    out["pin_records"] = sum(1 for m in msgs if m.startswith("[S172-PIN] step1_pin_source="))
    out["watcher_info_records"] = sum(1 for n, lv, _m in grab.records
                                      if n == "WatcherAgent" and lv == "INFO")
    print("L5PROBE " + json.dumps(out))


if __name__ == "__main__":
    if sys.argv[1] == "open":
        _open(sys.argv[2])
    else:
        _run(sys.argv[2], "--mutant" in sys.argv[3:])
