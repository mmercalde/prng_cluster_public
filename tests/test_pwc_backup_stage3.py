#!/usr/bin/env python3
"""test_pwc_backup_stage3.py — PWC full backup, Stage 3b (shared launch side).

Files under test: window_optimizer.py, window_optimizer_integration_final.py,
agent_manifests/window_optimizer.json. Base: main b2efdef. Plan:
docs/drafts/pwc_backup_stage3_v1/PLAN.md (S3-1..S3-5, S3-8, S3-12; Beta X-PWC-STAGE2B R1 Q5).

Two kinds of gate:

RANGE-MINER UNCHANGED (the owner's "DO NOT break range miner"):
  G-MINER-SUBTREE   statement-level AST: every statement of run_bidirectional_test except
                    the single `if _use_pw:` block is identical to the base, including the
                    whole `if _use_miner:` subtree and the preamble.
  G-MINER-DEFS      every definition the miner path reaches (_miner_residues_for_config,
                    _build_test_result_from_miner, _step1_miner_sweep_evidence, …) and every
                    other definition and top-level statement in both files is identical, except
                    the named edited functions and the named added PWC helpers.
  G-WO-FUNCS        inside the edited window_optimizer.py functions, the new AST minus the
                    declared PWC insertions equals the base AST, statement for statement.
  G-MINER-SETTINGS  RUNTIME: run_bayesian_optimization(use_range_miner=True, Run-4 shape)
                    builds a coordinator with exactly the same attributes and values as the
                    base, captured when optimize_window is called.
  G-MINER-JOBS      RUNTIME: run_bidirectional_test with use_range_miner=True hands
                    run_trial_miner exactly the same arguments as the base.
  G-MANIFEST        JSON key-path diff: only the ruled default flip and the added PWC key.

PWC BEHAVIOUR (rulings 2, 3, 15:53 PST default; Beta Q5):
  G-GUARD, G-PWC-CALL, G-INCOMPLETE, G-PROV, G-CREDIT, G-CLI, G-FLEET.

Every gate is also run against MUTANTS (textual edits of the live source); a mutant is
killed only if the edit applies once, the detector passes on the live source, and the
detector fails on the mutant with its own AssertionError (VIR-2).

CPU only. No GPU, no rig, no network, no live ledger, no dataset read (the dataset readers
are replaced). Run from the repository root: python3 -u tests/test_pwc_backup_stage3.py
"""
import ast
import contextlib
import copy
import importlib.util
import io
import json
import os
import subprocess
import sys
import traceback
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
BASE = "b2efdef"
WO_PATH = os.path.join(ROOT, "window_optimizer.py")
WOI_PATH = os.path.join(ROOT, "window_optimizer_integration_final.py")
MAN_PATH = os.path.join(ROOT, "agent_manifests", "window_optimizer.json")


def git_show(rel):
    return subprocess.check_output(["git", "-C", ROOT, "show", f"{BASE}:{rel}"], text=True)


def live(path):
    return open(path, encoding="utf-8").read()


def load_module(src, name, path):
    spec = importlib.util.spec_from_loader(name, loader=None, origin=path)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = path
    sys.modules[name] = mod
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, path, "exec"), mod.__dict__)
    return mod


def dump(node):
    return ast.dump(node, include_attributes=False)


def func(tree, name):
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise AssertionError(f"function {name} not found")


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


# [SKIP-DELIVERY SWITCH] (docs/drafts/miner_skip_switch_v1/PLAN.md §7; Beta
# X-MINER-SKIP-SWITCH R1 Q3). The switch's hunks, VERBATIM and ANCHORED: each pair is
# (the live text, the HEAD text) of one hunk WITH unchanged context on each side that
# reaches a line occurring exactly once in the HEAD file (Beta X-MINER-SKIP-SWITCH R3
# Q3: a blank-line or `)` anchor let the whole hunk move unnoticed). `strip_switch` requires each live text to occur exactly once and puts the HEAD
# text back, so the static gates then compare exactly what they compared before. An
# edit inside a hunk, a hunk moved to another place (its context lines no longer
# surround it), or any other change still fails. The runtime gates (G-MINER-SETTINGS,
# G-MINER-JOBS) carry no allowance.
SWITCH_WOI = [('                                    seed_count: int, resume_checkpoint: str,\n                                    resume_study: bool, skip_delivery_on: bool = False):\n    """Build the D6.2 run context, honouring §4.4\'s combination matrix.\n', '                                    seed_count: int, resume_checkpoint: str,\n                                    resume_study: bool):\n    """Build the D6.2 run context, honouring §4.4\'s combination matrix.\n'), ("        execution_set_id=_checkpoint_execution_set_id(),\n    )\n    if skip_delivery_on:\n        # [SKIP-DELIVERY SWITCH] the arm is part of the run's identity ONLY when on, so a\n        # baseline digest is unchanged and recovery refuses the other arm's checkpoint\n        # (utils/checkpoint_d6_2.py _require_context; Beta R1 Q2)\n        _components = dict(_components, **{SKIP_DELIVERY_CHECKPOINT_KEY: SKIP_DELIVERY_ON})\n    _digest = _build_run_context_digest(_components)\n", '        execution_set_id=_checkpoint_execution_set_id(),\n    )\n    _digest = _build_run_context_digest(_components)\n'), ('            "interval": (int(seed_start), int(seed_start) + int(seed_count))}\n\n\nSKIP_DELIVERY_SEARCH_KEY = "constant_skip_delivery"      # utils/step1_search_ledger.py\nSKIP_DELIVERY_STUDY_ATTR = "constant_skip_delivery"      # Optuna study user attribute\nSKIP_DELIVERY_CHECKPOINT_KEY = "constant_skip_delivery"  # D6.2 run-context component\nSKIP_DELIVERY_ON = "trial_range"\n\n\ndef _skip_delivery_on(coordinator):\n    """The run\'s switch through the ONE validator (miner.range_miner_coordinator.\n    validate_skip_delivery): an absent attribute and `False` are off, `True` is on, any\n    other value is refused before any study or dispatch."""\n    from miner.range_miner_coordinator import validate_skip_delivery, MinerMetadataError\n    try:\n        return validate_skip_delivery(getattr(coordinator, \'deliver_skip_range\', False),\n                                      where="Step 1 child")\n    except MinerMetadataError as exc:\n        raise Step1SearchPreflightError(str(exc)) from exc\n\n\ndef _step1_skip_delivery_preflight(search_ctx, deliver_on, resume_study=False,\n                                   study_name=\'\'):\n    """[SKIP-DELIVERY SWITCH] Switched-on trials never share a search, a study or (see\n    `_prepare_checkpoint_run_context`) a checkpoint with baseline trials\n    (docs/drafts/miner_skip_switch_v1/PLAN.md §4; Beta X-MINER-SKIP-SWITCH R1 Q2).\n    Raises Step1SearchPreflightError BEFORE any study is created or loaded and before any\n    checkpoint is recovered; refuses, never adopts."""\n    recorded = None\n    if search_ctx is not None:\n        space = search_ctx["record"].search_space or {}\n        recorded = space.get(SKIP_DELIVERY_SEARCH_KEY) if isinstance(space, dict) else None\n    if deliver_on and search_ctx is None:\n        raise Step1SearchPreflightError(\n            "--deliver-skip-range needs a Step 1 search (--step1-search-id) opened with "\n            "deliver_skip_range: true, so its trials get a study of their own")\n    if deliver_on and recorded != SKIP_DELIVERY_ON:\n        raise Step1SearchPreflightError(\n            f"--deliver-skip-range: search {search_ctx[\'record\'].search_id} was opened "\n            f"without it (search_space.{SKIP_DELIVERY_SEARCH_KEY}={recorded!r}); open a "\n            f"separate search with deliver_skip_range: true")\n    if not deliver_on and recorded is not None:\n        raise Step1SearchPreflightError(\n            f"search {search_ctx[\'record\'].search_id} was opened with "\n            f"deliver_skip_range ({SKIP_DELIVERY_SEARCH_KEY}={recorded!r}); a baseline "\n            f"run may not add trials to it")\n    # the study: an existing study carries its arm as a user attribute, set on the\n    # first trial of a switched study (_skip_delivery_study_check); a legacy or baseline\n    # study has none\n    if deliver_on and resume_study and not study_name:\n        raise Step1SearchPreflightError(\n            "--deliver-skip-range with --resume-study but no --study-name: automatic "\n            "selection could pick a baseline study; name the switched study or start fresh")\n    if resume_study and study_name and os.path.exists(f"optuna_studies/{study_name}.db"):\n        import optuna as _op\n        tag = (_op.load_study(study_name=study_name,\n                              storage=_step1_storage_url(study_name)).user_attrs or {}\n               ).get(SKIP_DELIVERY_STUDY_ATTR)\n        if deliver_on and tag != SKIP_DELIVERY_ON:\n            raise Step1SearchPreflightError(\n                f"--deliver-skip-range: study {study_name!r} was not created by a "\n                f"switched-on run ({SKIP_DELIVERY_STUDY_ATTR}={tag!r}); a switched search "\n                f"never adopts a baseline or legacy study")\n        if not deliver_on and tag is not None:\n            raise Step1SearchPreflightError(\n                f"study {study_name!r} belongs to a switched-on run "\n                f"({SKIP_DELIVERY_STUDY_ATTR}={tag!r}); a baseline run may not resume it")\n\n\ndef _skip_delivery_study_check(study, deliver_on, trial_number):\n    """[SKIP-DELIVERY SWITCH] The backstop at the first trial, before its sieve, for the\n    study the optimizer actually opened (this also covers automatic resume selection).\n    On: a fresh study is tagged; an untagged study with earlier trials is refused. Off:\n    a tagged study is refused; nothing is written."""\n    tag = (study.user_attrs or {}).get(SKIP_DELIVERY_STUDY_ATTR)\n    if deliver_on:\n        if tag == SKIP_DELIVERY_ON:\n            return\n        earlier = [t for t in study.trials if t.number < int(trial_number)]\n        if tag is None and not earlier:\n            study.set_user_attr(SKIP_DELIVERY_STUDY_ATTR, SKIP_DELIVERY_ON)\n            return\n        raise Step1SearchPreflightError(\n            f"--deliver-skip-range: study {study.study_name!r} holds {len(earlier)} trial(s) "\n            f"of another arm ({SKIP_DELIVERY_STUDY_ATTR}={tag!r}); refused before the sieve")\n    if tag is not None:\n        raise Step1SearchPreflightError(\n            f"study {study.study_name!r} belongs to a switched-on run "\n            f"({SKIP_DELIVERY_STUDY_ATTR}={tag!r}); a baseline run may not add trials to it")\n\n\nSKIP_DELIVERY_ROOT_MARKER = os.path.join(".s172_accumulator", "skip_delivery_arm")\n\n\ndef _step1_skip_delivery_root_arm(deliver_on, output_root=None, create=False):\n    """[SKIP-DELIVERY SWITCH] ONE OUTPUT ROOT PER ARM (Beta X-MINER-SKIP-SWITCH R1 finding 2,\n    R3 Q2). The finalizer merges whatever generation is current in the output root, and a\n    generation becomes current BEFORE its coverage and link rows exist\n    (docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md §2.3), so no rule that reads the\n    link can know a generation\'s arm in time. The arm is therefore a property of the ROOT:\n    a switched run marks `<root>/.s172_accumulator/skip_delivery_arm` (ignored by git,\n    never read by the finalizer) before its first trial, and every generation ever\n    published there is switched. Reads no SQLite at all, so no WAL can make it stale.\n\n      on : marker present   -> must say trial_range\n           marker absent    -> a current generation means a baseline root: refused;\n                               an empty root is marked (only when `create`); at the\n                               pre-publication recheck (`create=False`) it is refused\n      off: marker present   -> refused (a switched root)\n           marker absent    -> nothing read but the marker\'s existence, nothing written:\n                               exactly as before\n    Called at preflight (create=True) and again immediately before finalize_run, against\n    the root whose current generation the finalizer is about to merge.\n\n    SCOPE OF THE ISOLATION CLAIM (Beta X-MINER-SKIP-SWITCH-2 R1 Q2; owner path: state the\n    limit, no lock): it holds for SEPARATE output roots per arm with ONE publisher at a\n    time, as Stage 4 ran (docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md). Two Step 1\n    publishers overlapping in one initially unmarked root are NOT isolated: the finalizer\n    selects its merge parent from `current` independently of this check. Any backup or\n    restore of the root must carry the marker; a root restored without it admits off."""\n    root = output_root or os.getcwd()\n    marker = os.path.join(root, SKIP_DELIVERY_ROOT_MARKER)\n    current = os.path.join(root, ".s172_accumulator", "current")\n    if not deliver_on:\n        if os.path.lexists(marker):\n            raise Step1SearchPreflightError(\n                f"the output root {root} belongs to the switched-on arm ({marker}); a baseline "\n                f"run may not publish into it. Use a separate output root per arm, one publisher "\n                f"at a time (the switch\'s isolation holds only so)")\n        return\n    if os.path.lexists(marker):\n        with open(marker, encoding="utf-8") as fh:\n            arm = fh.read().strip()\n        if arm != SKIP_DELIVERY_ON:\n            raise Step1SearchPreflightError(f"{marker} says {arm!r}, not {SKIP_DELIVERY_ON!r}")\n        return\n    if os.path.lexists(current):\n        raise Step1SearchPreflightError(\n            f"--deliver-skip-range: the output root {root} already holds a generation "\n            f"published without the switch; the finalizer would merge it. Use a separate, "\n            f"empty output root for the switched arm, one publisher at a time")\n    if not create:\n        raise Step1SearchPreflightError(\n            f"--deliver-skip-range: {marker} is missing at publication; the root was not "\n            f"marked for the switched arm")\n    print(f"[SKIP-DELIVERY] marking {root} as the switched-on arm\'s output root. Isolation "\n          f"holds only with a separate output root per arm and ONE publisher at a time: no "\n          f"baseline Step 1 may publish into this root while this run is live. Any backup or "\n          f"restore of this root must carry {SKIP_DELIVERY_ROOT_MARKER}.", flush=True)\n    os.makedirs(os.path.dirname(marker), exist_ok=True)\n    tmp = f"{marker}.tmp-{os.getpid()}"\n    with open(tmp, "x", encoding="utf-8") as fh:\n        fh.write(SKIP_DELIVERY_ON + "\\n")\n    os.replace(tmp, marker)\n\n\ndef _step1_skip_delivery_exact_pin(deliver_on, pinned, trials, resume_study, skip_min, skip_max):\n    """[SKIP-DELIVERY SWITCH] Beta X-MINER-SKIP-SWITCH-2 R2 Q4: a switched run that cannot\n    sample (THE rule, utils.step1_search_ledger.skip_delivery_can_sample) consumes its pin\n    EXACTLY: returns [skip_min, skip_max], checked by the one validator, for the study\n    body (window_optimizer_bayesian.py) to draw as single-point distributions. Every\n    other run, and every off run, returns None and the study body is unchanged."""\n    if not deliver_on:\n        return None\n    from utils.step1_search_ledger import skip_delivery_can_sample\n    if skip_delivery_can_sample(pinned, trials, resume_study):\n        return None\n    from miner.range_miner_coordinator import validate_skip_delivery, MinerMetadataError\n    try:\n        validate_skip_delivery(True, skip_min, skip_max, where="Step 1 child (exact pin)")\n    except MinerMetadataError as exc:\n        raise Step1SearchPreflightError(str(exc)) from exc\n    return [int(skip_min), int(skip_max)]\n\n\ndef _step1_skip_delivery_bounds_check(deliver_on, pinned=False, trials=None,\n                                     resume_study=False):\n    """[SKIP-DELIVERY SWITCH] Before any study is opened or loaded: a switched run that can\n    sample checks the full sampling domains under THE rule search open uses\n    (utils.step1_search_ledger.skip_delivery_can_sample / validate_skip_sampling_domains;\n    Beta X-MINER-SKIP-SWITCH-2 R1 Q4), so the two cannot disagree. Returns the CHECKED\n    SearchBounds object, which the caller hands to the sampler; off returns None and the\n    caller loads its bounds exactly as before."""\n    if not deliver_on:\n        return None\n    from window_optimizer import SearchBounds as _SB\n    from miner.range_miner_coordinator import MinerMetadataError\n    from utils.step1_search_ledger import skip_delivery_can_sample, validate_skip_sampling_domains\n    b = _SB.from_config()\n    if skip_delivery_can_sample(pinned, trials, resume_study):\n        try:\n            validate_skip_sampling_domains(b.min_skip_min, b.max_skip_min, b.min_skip_max,\n                                           b.max_skip_max, where="Step 1 child (sampling bounds)")\n        except MinerMetadataError as exc:\n            raise Step1SearchPreflightError(str(exc)) from exc\n    return b\n\n\nclass Step1ChunkState:\n', '            "interval": (int(seed_start), int(seed_start) + int(seed_count))}\n\n\nclass Step1ChunkState:\n'), ('            self.study_name = optuna_trial.study.study_name\n            _skip_delivery_study_check(optuna_trial.study,\n                                       getattr(self, "skip_delivery_on", False), n)\n            if self.search_ctx is not None:\n                prior = sum(1 for t in optuna_trial.study.trials if t.number < n)\n', '            self.study_name = optuna_trial.study.study_name\n            if self.search_ctx is not None:\n                prior = sum(1 for t in optuna_trial.study.trials if t.number < n)\n'), ('            phase5_sink            = _miner_sink,\n            # [SKIP-DELIVERY SWITCH] passed ONLY when on, so a baseline call is unchanged\n            **({"deliver_skip_range": True} if _skip_delivery_on(coordinator) else {}),\n        )\n        if _miner_result.get("pruned"):\n', '            phase5_sink            = _miner_sink,\n        )\n        if _miner_result.get("pruned"):\n'), ('            study_name=study_name, n_parallel=n_parallel)\n        _s1_skip_delivery_on = _skip_delivery_on(self)\n        # the bounds first: before the named-study check loads any store\n        _s1_pinned = None not in (warm_start_window, warm_start_offset, warm_start_skip_min,\n                                  warm_start_skip_max, warm_start_fwd_thresh,\n                                  warm_start_rev_thresh, warm_start_session_idx)\n        _s1_skip_bounds = _step1_skip_delivery_bounds_check(\n            _s1_skip_delivery_on, pinned=_s1_pinned,\n            trials=max_iterations, resume_study=resume_study)\n        _s1_skip_exact_pin = _step1_skip_delivery_exact_pin(\n            _s1_skip_delivery_on, _s1_pinned, max_iterations, resume_study,\n            warm_start_skip_min, warm_start_skip_max)\n        _step1_skip_delivery_preflight(_s1_search_ctx, _s1_skip_delivery_on,\n                                       resume_study=resume_study, study_name=study_name)\n        _step1_skip_delivery_root_arm(_s1_skip_delivery_on, create=True)\n        _step1_budget_preflight(resume_study=resume_study, study_name=study_name,\n', '            study_name=study_name, n_parallel=n_parallel)\n        _step1_budget_preflight(resume_study=resume_study, study_name=study_name,\n'), ('                                    seed_start, seed_count, max_iterations, None)\n        _s1_chunk.skip_delivery_on = _s1_skip_delivery_on\n        # S115 M1/M4: Partition map (IPs from distributed_config.json)\n', '                                    seed_start, seed_count, max_iterations, None)\n        # S115 M1/M4: Partition map (IPs from distributed_config.json)\n'), ('                resume_study=resume_study,\n                skip_delivery_on=_s1_skip_delivery_on,\n            )\n        _install_flush_run_context(_d6_2_context)\n', '                resume_study=resume_study,\n            )\n        _install_flush_run_context(_d6_2_context)\n'), ("            optimizer = WindowOptimizer(self, dataset_path)\n            # [SKIP-DELIVERY SWITCH] on: the bounds object checked at preflight is the one\n            # the sampler uses; off: loaded exactly as before\n            bounds = _s1_skip_bounds if _s1_skip_bounds is not None else SearchBounds.from_config()\n            # [S172 D6.2 §4.4] The record's `trial_number` comes from THIS\n", "            optimizer = WindowOptimizer(self, dataset_path)\n            bounds = SearchBounds.from_config()\n            # [S172 D6.2 §4.4] The record's `trial_number` comes from THIS\n"), ("            'warm_start_session_idx': warm_start_session_idx,  # [S166] required by Optuna\n        }\n        if _s1_skip_exact_pin is not None:\n            # [SKIP-DELIVERY SWITCH] only a switched run that cannot sample carries it\n            _trial_history_ctx['skip_delivery_exact_pin'] = _s1_skip_exact_pin\n\n        if not _np2_complete:  # [S142-B] skip single-process search for NP2\n", "            'warm_start_session_idx': warm_start_session_idx,  # [S166] required by Optuna\n        }\n\n        if not _np2_complete:  # [S142-B] skip single-process search for NP2\n"), ("        _skip_modes_d3_5 = (\n            ('constant', 'variable')\n            if (test_both_modes and not prng_base.endswith('_hybrid'))\n            else ('constant',)\n        )\n\n        # [SKIP-DELIVERY SWITCH] the arm, rechecked against the root whose current\n        # generation the finalizer merges (Beta R3 Q2)\n        _step1_skip_delivery_root_arm(_s1_skip_delivery_on)\n        _artifact_d3_5 = _finalize_run_d3_5(\n", "        _skip_modes_d3_5 = (\n            ('constant', 'variable')\n            if (test_both_modes and not prng_base.endswith('_hybrid'))\n            else ('constant',)\n        )\n\n        _artifact_d3_5 = _finalize_run_d3_5(\n")]
SWITCH_WO = [('    step1_search_id: str = \'\',              # [STEP1-SEARCH stage 1] hop 2 of 3\n    pwc_coordinator_address: str = None,    # [PWC-STAGE3] dial address; PWC only\n    *,\n    deliver_skip_range: bool = False,       # [SKIP-DELIVERY SWITCH] hop 2 of 3; keyword-only\n) -> Dict[str, Any]:\n    """\n    Run Bayesian optimization to find optimal window parameters\n', '    step1_search_id: str = \'\',              # [STEP1-SEARCH stage 1] hop 2 of 3\n    pwc_coordinator_address: str = None,    # [PWC-STAGE3] dial address; PWC only\n) -> Dict[str, Any]:\n    """\n    Run Bayesian optimization to find optimal window parameters\n'), ('        Dictionary with optimization results\n    """\n    # [S145] PRE-DISPATCH SEED-DOMAIN WALL — the CLI entry point.\n', '        Dictionary with optimization results\n    """\n\n    # [S145] PRE-DISPATCH SEED-DOMAIN WALL — the CLI entry point.\n'), ('              "execution, no staging, no coverage mutation.")\n        raise\n\n    # [SKIP-DELIVERY SWITCH] the one validator, right after the S145 wall (which must stay\n    # the first statement) and before any coordinator exists: True or False only; None,\n    # "true", 1, ... are refused, never read as off (Beta X-MINER-SKIP-SWITCH R3 Q4)\n    from miner.range_miner_coordinator import validate_skip_delivery as _vsd\n    _vsd(deliver_skip_range, where="run_bayesian_optimization")\n\n    # [MINER SEED-START R4/R5] G-MINER-START is replaced: the RANGE-MINER route now\n', '              "execution, no staging, no coverage mutation.")\n        raise\n\n    # [MINER SEED-START R4/R5] G-MINER-START is replaced: the RANGE-MINER route now\n'), ('    coordinator.use_range_miner   = use_range_miner\n    if deliver_skip_range is True:\n        # [SKIP-DELIVERY SWITCH] set ONLY when on: a baseline coordinator is unchanged\n        coordinator.deliver_skip_range = True\n        print("   [SKIP-DELIVERY] ON: RANGE-MINER phases 1-2 sieve each trial\'s own "\n              "[skip_min, skip_max] (baseline: [0, 16])")\n    coordinator.miner_stripe_size = miner_stripe_size\n', '    coordinator.use_range_miner   = use_range_miner\n    coordinator.miner_stripe_size = miner_stripe_size\n'), ('                        help=\'[S172] Use RANGE-MINER stripe backend (opt-in, mutex vs PWC/ZMQ)\')\n    parser.add_argument(\'--deliver-skip-range\', action=\'store_true\', default=False,\n                        help=\'[SKIP-DELIVERY SWITCH] RANGE-MINER phases 1-2 sieve the \'\n                             "trial\'s own [skip_min, skip_max] instead of the baseline "\n                             \'[0, 16]. Off by default; needs --use-range-miner and a \'\n                             \'Step 1 search opened with deliver_skip_range: true\')\n    parser.add_argument(\'--miner-stripe-size\', type=int, default=67_108_864,\n', "                        help='[S172] Use RANGE-MINER stripe backend (opt-in, mutex vs PWC/ZMQ)')\n    parser.add_argument('--miner-stripe-size', type=int, default=67_108_864,\n"), ('            f"--use-range-miner may be set (got: {\', \'.join(\'--\' + n.replace(\'_\',\'-\') for n in _enabled)})"\n        )\n\n    if getattr(args, \'deliver_skip_range\', False) and not args.use_range_miner:\n        parser.error("--deliver-skip-range applies to the RANGE-MINER route only "\n                     "(--use-range-miner); PWC already sends the trial\'s skip range")\n    if getattr(args, \'deliver_skip_range\', False):\n        # [SKIP-DELIVERY SWITCH] the one validator (miner.range_miner_coordinator); a\n        # pinned pair is checked HERE, before any search, study or dispatch\n        from miner.range_miner_coordinator import (\n            validate_skip_delivery as _vsd, MinerMetadataError as _MME)\n        _pins = (getattr(args, \'warm_start_skip_min\', None),\n                 getattr(args, \'warm_start_skip_max\', None))\n        try:\n            _vsd(True, *_pins, where="--deliver-skip-range") if _pins != (None, None) \\\n                else _vsd(True, where="--deliver-skip-range")\n        except _MME as _e:\n            parser.error(str(_e))\n        if getattr(args, \'config_file\', None):\n            parser.error("--deliver-skip-range is not carried by the --config-file route; "\n                         "refused rather than dropped (use the Bayesian search route with "\n                         "a Step 1 search opened with deliver_skip_range: true)")\n\n    # ========================================================================\n    # [RESOLVED EXECUTION SET] RUN-START FLEET AUTHORITY — resolve once, freeze\n', '            f"--use-range-miner may be set (got: {\', \'.join(\'--\' + n.replace(\'_\',\'-\') for n in _enabled)})"\n        )\n\n    # ========================================================================\n    # [RESOLVED EXECUTION SET] RUN-START FLEET AUTHORITY — resolve once, freeze\n'), ("            use_range_miner=getattr(args, 'use_range_miner', False),\n            deliver_skip_range=getattr(args, 'deliver_skip_range', False),  # SKIP-DELIVERY\n            miner_stripe_size=getattr(args, 'miner_stripe_size', 67_108_864),\n", "            use_range_miner=getattr(args, 'use_range_miner', False),\n            miner_stripe_size=getattr(args, 'miner_stripe_size', 67_108_864),\n")]


def strip_switch(src, hunks, label):
    for live_txt, base_txt in hunks:
        n = src.count(live_txt)
        assert n == 1, (f"{label}: a skip-switch hunk occurs {n} times at its anchored place "
                        f"(it must occur exactly once, verbatim, between its context lines): "
                        f"{live_txt[:120]!r}")
        src = src.replace(live_txt, base_txt)
    return src


# ═════════════════════════════════════════════════════════════════════════════
# RANGE-MINER unchanged — static
# ═════════════════════════════════════════════════════════════════════════════
def _is_use_pw_if(stmt):
    return isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Name) and stmt.test.id == "_use_pw"


def d_miner_subtree(woi_src=None):
    base = func(ast.parse(git_show("window_optimizer_integration_final.py")), "run_bidirectional_test")
    new = func(ast.parse(strip_switch(woi_src or live(WOI_PATH), SWITCH_WOI, "integration")),
               "run_bidirectional_test")
    assert dump(base.args) == dump(new.args), "run_bidirectional_test signature changed"
    bi = [i for i, s in enumerate(base.body) if _is_use_pw_if(s)]
    ni = [i for i, s in enumerate(new.body) if _is_use_pw_if(s)]
    assert len(bi) == 1 and len(ni) == 1, f"`if _use_pw:` blocks: base {bi}, new {ni}"
    bpre, npre = base.body[:bi[0]], new.body[:ni[0]]
    assert len(bpre) == len(npre), f"preamble statement count {len(bpre)} -> {len(npre)}"
    for k, (a, b) in enumerate(zip(bpre, npre)):
        assert dump(a) == dump(b), (f"statement {k} before `if _use_pw:` differs "
                                    f"(line {b.lineno}): {ast.unparse(b)[:120]!r}")
    mi = [s for s in npre if isinstance(s, ast.If) and getattr(s.test, "id", "") == "_use_miner"]
    assert len(mi) == 1, "the `if _use_miner:` block is not where it was"
    bpost, npost = base.body[bi[0] + 1:], new.body[ni[0] + 1:]
    assert [dump(s) for s in bpost] == [dump(s) for s in npost], "statements after `if _use_pw:` differ"
    return (f"run_bidirectional_test: {len(npre)} statements before `if _use_pw:` (incl. the "
            f"`if _use_miner:` subtree, {sum(1 for _ in ast.walk(mi[0]))} AST nodes) and "
            f"{len(npost)} after it are identical to {BASE}; signature identical")


# Edited and added names, declared. Everything else in both files must be identical.
WOI_EDITED = ["run_bidirectional_test"]
WOI_ADDED = ["PwcTrialIncompleteError", "PwcThresholdProvenanceError",
             "_PWC_TRIAL_COMPLETION_SCHEMA", "_pwc_required_passes",
             "_pwc_require_complete_trial", "_pwc_require_threshold_provenance",
             "_step1_pwc_sweep_evidence"]
WO_EDITED = ["assert_step1_route_seed_start_supported", "run_bayesian_optimization",
             "run_with_config", "main"]
MINER_REACHED = ["_miner_residues_for_config", "_build_test_result_from_miner",
                 "_step1_miner_sweep_evidence", "_step1_required_phases",
                 "_step1_ledger_rows_via_copy", "_get_residues_for_config",
                 "extract_survivor_records", "resolve_directional_threshold"]


def _prove(base_src, new_src, allow, allow_added):
    spec = importlib.util.spec_from_file_location("prove_unchanged_outside",
                                                  os.path.join(ROOT, "scripts", "prove_unchanged_outside.py"))
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    rep = P.compare(base_src, new_src, allow, allow_added)
    return rep


def d_miner_defs(woi_src=None, wo_src=None):
    woi_stripped = strip_switch(woi_src or live(WOI_PATH), SWITCH_WOI, "integration")
    r1 = _prove(git_show("window_optimizer_integration_final.py"), woi_stripped,
                WOI_EDITED, WOI_ADDED)
    assert r1["ok"], f"integration file: {json.dumps(r1['problems'])[:400]}"
    r2 = _prove(git_show("window_optimizer.py"),
                strip_switch(wo_src or live(WO_PATH), SWITCH_WO, "window_optimizer.py"), WO_EDITED, [])
    assert r2["ok"], f"window_optimizer.py: {json.dumps(r2['problems'])[:400]}"
    tree = ast.parse(woi_stripped)
    btree = ast.parse(git_show("window_optimizer_integration_final.py"))
    for name in MINER_REACHED:
        assert dump(func(btree, name)) == dump(func(tree, name)), f"{name} changed"
    return (f"integration: {r1['checked_definitions']} definitions identical outside "
            f"{WOI_EDITED} + added {len(WOI_ADDED)} PWC names; window_optimizer.py: "
            f"{r2['checked_definitions']} identical outside {WO_EDITED}; miner-reached "
            f"{MINER_REACHED[:3]}… identical")


# ── inside the edited window_optimizer.py functions: new minus declared insertions == base
_PWC_KW = {"pwc_coordinator_address", "require_fleet"}


class _StripPwc(ast.NodeTransformer):
    """Remove exactly the declared Stage-3 insertions from a window_optimizer.py AST."""

    def visit_Call(self, node):
        self.generic_visit(node)
        node.keywords = [k for k in node.keywords if k.arg not in _PWC_KW]
        return node

    def visit_arguments(self, node):
        keep = [(a, d) for a, d in zip(node.args[len(node.args) - len(node.defaults):], node.defaults)]
        n_no_default = len(node.args) - len(node.defaults)
        args = node.args[:n_no_default] + [a for a, _ in keep if a.arg != "pwc_coordinator_address"]
        defaults = [d for a, d in keep if a.arg != "pwc_coordinator_address"]
        node.args, node.defaults = args, defaults
        return node

    def _filter(self, body):
        out = []
        for s in body:
            # the inserted `if use_persistent_workers: coordinator.pwc_coordinator_address = …`
            if (isinstance(s, ast.If) and isinstance(s.test, ast.Name)
                    and s.test.id == "use_persistent_workers" and len(s.body) == 1
                    and isinstance(s.body[0], ast.Assign)
                    and isinstance(s.body[0].targets[0], ast.Attribute)
                    and s.body[0].targets[0].attr == "pwc_coordinator_address"):
                continue
            # the inserted argparse flag
            if (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                    and getattr(s.value.func, "attr", "") == "add_argument" and s.value.args
                    and isinstance(s.value.args[0], ast.Constant)
                    and s.value.args[0].value == "--pwc-coordinator-address"):
                continue
            out.append(s)
        return out

    def generic_visit(self, node):
        for field in ("body", "orelse", "finalbody"):
            if isinstance(getattr(node, field, None), list):
                setattr(node, field, self._filter(getattr(node, field)))
        return super().generic_visit(node)


def d_wo_funcs(wo_src=None):
    btree = ast.parse(git_show("window_optimizer.py"))
    ntree = ast.parse(strip_switch(wo_src or live(WO_PATH), SWITCH_WO, "window_optimizer.py"))
    out = []
    for name in ["run_bayesian_optimization", "run_with_config", "main"]:
        b, n = func(btree, name), func(ntree, name)
        ns = _StripPwc().visit(copy.deepcopy(n))
        assert dump(b) == dump(ns), (f"{name}: differs from {BASE} beyond the declared PWC "
                                     f"insertions (dial-address parameter/attribute/flag/keywords, "
                                     f"require_fleet)")
        out.append(name)
    # the guard: only its condition and comments changed; behaviour table in G-GUARD
    return f"{', '.join(out)}: new AST minus the declared PWC insertions == {BASE}"


# ═════════════════════════════════════════════════════════════════════════════
# RANGE-MINER unchanged — runtime
# ═════════════════════════════════════════════════════════════════════════════
class _Captured(Exception):
    pass


def _settings(wo_mod, **kw):
    """Run run_bayesian_optimization until it hands off to optimize_window; return the
    coordinator's attributes at that moment."""
    seen = {}

    class Coord:
        def __init__(self, *a, **k):
            self._ctor = (a, tuple(sorted(k.items())))

        def optimize_window(self, **k):
            seen["attrs"] = dict(vars(self))
            seen["optimize_kwargs"] = {x: (y if isinstance(y, (int, float, str, bool, type(None)))
                                           else type(y).__name__) for x, y in k.items()}
            raise _Captured()
    # run_bayesian_optimization imports add_window_optimizer_to_coordinator from the
    # integration module inside the function (window_optimizer.py), so it is replaced there
    import window_optimizer_integration_final as WOI
    saved = (wo_mod.MultiGPUCoordinator, wo_mod.COORDINATOR_AVAILABLE,
             WOI.add_window_optimizer_to_coordinator)
    wo_mod.MultiGPUCoordinator, wo_mod.COORDINATOR_AVAILABLE = Coord, True
    WOI.add_window_optimizer_to_coordinator = lambda: None
    try:
        with quiet():
            try:
                wo_mod.run_bayesian_optimization(lottery_file="/nonexistent.json", trials=1,
                                                 output_config="/dev/null", **kw)
            except _Captured:
                pass
    finally:
        (wo_mod.MultiGPUCoordinator, wo_mod.COORDINATOR_AVAILABLE,
         WOI.add_window_optimizer_to_coordinator) = saved
    assert "attrs" in seen, "optimize_window was never reached"
    return seen


RUN4_SHAPE = dict(seed_count=2147483648, seed_start=2147483648, prng_type="java_lcg",
                  test_both_modes=True, use_range_miner=True, worker_pool_size=25,
                  miner_stripe_size=67108864, n_parallel=1, staging_dir="/nonexistent/staging")


def d_miner_settings(wo_mod=None):
    base = load_module(git_show("window_optimizer.py"), "wo_base_s3", WO_PATH)
    new = wo_mod or __import__("window_optimizer")
    a, b = _settings(base, **RUN4_SHAPE), _settings(new, **RUN4_SHAPE)
    assert a["attrs"] == b["attrs"], (
        f"RANGE-MINER coordinator settings differ: only base "
        f"{sorted(set(a['attrs']) - set(b['attrs']))}, only new "
        f"{sorted(set(b['attrs']) - set(a['attrs']))}, changed "
        f"{sorted(k for k in set(a['attrs']) & set(b['attrs']) if a['attrs'][k] != b['attrs'][k])}")
    assert a["optimize_kwargs"] == b["optimize_kwargs"], "optimize_window arguments differ"
    # control: the PWC run DOES differ, by exactly the dial address
    pa = _settings(base, seed_count=1000, use_persistent_workers=True)["attrs"]
    pb = _settings(new, seed_count=1000, use_persistent_workers=True,
                   pwc_coordinator_address="192.168.3.177")["attrs"]
    assert set(pb) - set(pa) == {"pwc_coordinator_address"} and pb["pwc_coordinator_address"] == "192.168.3.177", \
        (sorted(set(pb) - set(pa)))
    return (f"use_range_miner=True (Run-4 shape): {len(b['attrs'])} coordinator attributes and the "
            f"optimize_window arguments identical to {BASE}; control: a PWC run gains exactly "
            f"pwc_coordinator_address")


def _cfg():
    from window_optimizer import WindowConfig
    return WindowConfig(window_size=8, offset=0, sessions=["evening"], skip_min=0, skip_max=16,
                        forward_threshold=0.5, reverse_threshold=0.5)


def _canon(v):
    if isinstance(v, (int, float, str, bool, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    if isinstance(v, dict):
        return {k: _canon(x) for k, x in sorted(v.items())}
    if hasattr(v, "__dict__"):
        return {"__type__": type(v).__name__, **{k: _canon(x) for k, x in sorted(vars(v).items())}}
    return type(v).__name__


def _miner_jobs(woi_mod):
    rec = []

    def run_trial_miner(**kw):
        rec.append(_canon(kw))
        return {"pruned": True}
    saved = {k: getattr(woi_mod, k) for k in ("run_trial_miner", "build_assembling_sink",
                                              "resolve_assembly_backend", "_miner_residues_for_config")}
    woi_mod.run_trial_miner = run_trial_miner
    woi_mod.build_assembling_sink = lambda backend: types.SimpleNamespace(backend=backend)
    woi_mod.resolve_assembly_backend = lambda name, **kw: f"backend:{name}:{sorted(kw)}"
    woi_mod._miner_residues_for_config = lambda config, path: [1, 2, 3, 4, 5, 6, 7, 8]
    coord = types.SimpleNamespace(use_range_miner=True, use_persistent_workers=False,
                                  use_zmq_sqlite=False, config_file="distributed_config.json",
                                  worker_pool_size=25, miner_stripe_size=67108864,
                                  staging_dir="/nonexistent/staging", seed_cap_amd=2_000_000)
    try:
        with quiet():
            woi_mod.run_bidirectional_test(coord, _cfg(), "/nonexistent/daily3.json",
                                           seed_start=2147483648, seed_count=2147483648,
                                           prng_base="java_lcg", test_both_modes=True,
                                           forward_threshold=0.5, reverse_threshold=0.5,
                                           trial_number=3, accumulator=None)
    finally:
        for k, v in saved.items():
            setattr(woi_mod, k, v)
    return rec


def d_miner_jobs(woi_mod=None):
    base = load_module(git_show("window_optimizer_integration_final.py"), "woi_base_s3", WOI_PATH)
    new = woi_mod or __import__("window_optimizer_integration_final")
    a, b = _miner_jobs(base), _miner_jobs(new)
    assert len(a) == len(b) == 1, (len(a), len(b))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), (
        "run_trial_miner arguments differ: " +
        str(sorted(k for k in set(a[0]) | set(b[0]) if a[0].get(k) != b[0].get(k))))
    return f"use_range_miner=True: run_trial_miner got the same {len(b[0])} arguments as {BASE}"


def d_manifest(man_src=None):
    base = json.loads(git_show("agent_manifests/window_optimizer.json"))
    new = json.loads(man_src or live(MAN_PATH))

    def flat(d, p=""):
        out = {}
        if isinstance(d, dict):
            for k, v in d.items():
                out.update(flat(v, f"{p}/{k}"))
        elif isinstance(d, list):
            for i, v in enumerate(d):
                out.update(flat(v, f"{p}[{i}]"))
        else:
            out[p] = d
        return out
    fb, fn = flat(base), flat(new)
    changed = sorted(k for k in set(fb) & set(fn) if fb[k] != fn[k])
    added, removed = sorted(set(fn) - set(fb)), sorted(set(fb) - set(fn))
    want_changed = ["/default_params/use_persistent_workers", "/default_params/use_range_miner",
                    "/parameter_bounds/use_range_miner/default",
                    "/parameter_bounds/use_range_miner/description"]
    want_added = ["/actions[0]/args_map/pwc-coordinator-address",
                  "/default_params/deliver_skip_range",          # [SKIP-DELIVERY SWITCH]
                  "/default_params/pwc_coordinator_address"]
    assert changed == want_changed and added == want_added and not removed, (changed, added, removed)
    assert fn["/default_params/use_range_miner"] is True and fn["/default_params/use_persistent_workers"] is False
    assert fn["/default_params/pwc_coordinator_address"] is None
    assert fn["/default_params/deliver_skip_range"] is False, "the skip switch must default off"
    # what a Run-4 launch merges: WATCHER starts from default_params and overrides the
    # declared keys (agents/watcher_agent.py run_step); Run 4 names both engine flags
    run4 = {"use_persistent_workers": False, "use_range_miner": True, "worker_pool_size": 25,
            "seed_start": 0, "max_seeds": 2147483648, "miner_stripe_size": 67108864,
            "test_both_modes": True, "prng_type": "java_lcg", "window_trials": 1, "n_parallel": 1}

    def merge(dp):
        f = dict(dp)
        f.update({k: v for k, v in run4.items() if k in dp})
        return f
    mb, mn = merge(base["default_params"]), merge(new["default_params"])
    diff = sorted(k for k in set(mb) | set(mn) if mb.get(k, "<absent>") != mn.get(k, "<absent>"))
    assert diff == ["deliver_skip_range", "pwc_coordinator_address"] and \
        mn["pwc_coordinator_address"] is None and mn["deliver_skip_range"] is False, diff
    wsrc = live(os.path.join(ROOT, "agents", "watcher_agent.py"))
    assert "if value is None or value == '':\n                    continue" in wsrc, \
        "WATCHER no longer skips None params; a null key would reach argv"
    assert "if isinstance(value, bool):\n                    if value:" in wsrc, \
        "WATCHER no longer omits a False bool; deliver_skip_range=false would reach argv"
    return ("manifest: only the ruled default flip (+ use_range_miner spec) and one added null "
            "PWC key; Run-4 params merge to the same values except that null key, which WATCHER "
            "skips (agents/watcher_agent.py run_step, `if value is None`)")


# ═════════════════════════════════════════════════════════════════════════════
# PWC behaviour
# ═════════════════════════════════════════════════════════════════════════════
def d_guard(wo_mod=None):
    WO = wo_mod or __import__("window_optimizer")
    T31 = 2 ** 31
    rows = []
    for ss in (0, T31):
        for route, kw in (("miner", {}), ("pwc", {"use_persistent_workers": True}),
                          ("zmq", {"use_zmq_sqlite": True}), ("legacy", {})):
            for form in ("flags", "route"):
                args = kw if form == "flags" else ({"route": route} if route in ("pwc", "zmq") else {})
                try:
                    WO.assert_step1_route_seed_start_supported(ss, context="t", **args)
                    raised = False
                except WO.Step1RouteStartUnsupportedError:
                    raised = True
                want = (route == "zmq" and ss != 0)
                assert raised == want, f"seed_start={ss} route={route} ({form}): raised={raised}"
                rows.append((ss, route, raised))
    return "only ZMQ at 2^31 refuses; PWC, miner and legacy at 0 and 2^31 pass (flags and route=)"


def _complete_result(transport="tcp", phases=("1", "2", "3", "4"), lo=2 ** 31, hi=2 ** 31 + 1000,
                     verified=True, survivors=True):
    passes = {}
    for ph in ("1", "2", "3", "4"):
        if ph in phases:
            passes[ph] = {"status": "complete", "accounting": {
                "transport": transport, "complete": True,
                "threshold": {"agree": True, "verified": verified}}}
        else:
            passes[ph] = {"status": "not_run", "accounting": None}
    comp = {"schema": "pwc_trial_completion_v1", "trial_complete": True, "passes": passes,
            "incomplete_passes": [], "swept_intervals": {ph: [lo, hi] for ph in phases}}
    m = {7: 0.9} if survivors else {}
    return {"pwc_completion": comp, "pwc_trial_complete": True,
            "forward_map_constant": dict(m), "reverse_map_constant": dict(m),
            "forward_map_variable": {}, "reverse_map_variable": {}}


def _pwc_drive(woi_mod, result, *, seed_start=2 ** 31, both=True, coord_over=None):
    calls = {"rtp": [], "builder": []}
    saved = {k: getattr(woi_mod, k) for k in ("run_trial_persistent", "_build_test_result_from_pw",
                                              "_get_residues_for_config")}

    def rtp(**kw):
        calls["rtp"].append(kw)
        return copy.deepcopy(result)

    def builder(pw_result, accumulator, *a, **k):
        calls["builder"].append(pw_result)
        if accumulator is not None:
            accumulator["bidirectional"].append("x")
        return types.SimpleNamespace()
    woi_mod.run_trial_persistent, woi_mod._build_test_result_from_pw = rtp, builder
    woi_mod._get_residues_for_config = lambda config, path: [1, 2, 3]
    coord = types.SimpleNamespace(use_range_miner=False, use_persistent_workers=True,
                                  use_zmq_sqlite=False, config_file="distributed_config.json",
                                  pwc_coordinator_address="192.168.3.177", pwc_min_workers=24)
    for k, v in (coord_over or {}).items():
        setattr(coord, k, v)
    acc = {"bidirectional": [], "forward": [], "reverse": []}
    exc, tr = None, None
    try:
        with quiet():
            tr = woi_mod.run_bidirectional_test(coord, _cfg(), "/nonexistent/daily3.json",
                                                seed_start=seed_start, seed_count=1000,
                                                prng_base="java_lcg", test_both_modes=both,
                                                forward_threshold=0.5, reverse_threshold=0.5,
                                                trial_number=5, accumulator=acc)
    except Exception as e:                                          # noqa: BLE001
        exc = e
    finally:
        for k, v in saved.items():
            setattr(woi_mod, k, v)
    return calls, acc, exc, tr


def d_pwc_call(woi_mod=None):
    W = woi_mod or __import__("window_optimizer_integration_final")
    calls, acc, exc, tr = _pwc_drive(W, _complete_result())
    assert exc is None, f"{type(exc).__name__}: {exc}"
    kw = calls["rtp"][0]
    assert kw.get("seed_start") == 2 ** 31, f"seed_start not handed on: {kw.get('seed_start')}"
    assert kw.get("pwc_coordinator_address") == "192.168.3.177", kw.get("pwc_coordinator_address")
    # [Beta X-PWC-STAGE3 R1 3b-Q7] the coordinator's pool size (24) is NOT passed as the
    # ready minimum: b2efdef's call shape, effective minimum 1 (owner ruling 2026-08-04).
    assert "min_workers" not in kw, f"min_workers passed: {kw.get('min_workers')}"
    assert kw.get("total_seeds") == 1000
    return "PWC at seed_start 2^31: dispatched with seed_start and the dial address; min_workers not passed (b2efdef shape)"


def d_incomplete(woi_mod=None):
    W = woi_mod or __import__("window_optimizer_integration_final")
    bad = {}
    r = _complete_result(); r["pwc_trial_complete"] = False; r["pwc_completion"]["trial_complete"] = False
    r["pwc_completion"]["incomplete_passes"] = ["3"]; bad["trial incomplete"] = r
    r = _complete_result(); del r["pwc_completion"]; bad["no completion record"] = r
    bad["pruned (old build)"] = {"pruned": True, "reason": "forward_zero"}
    r = _complete_result(); r["pwc_completion"]["passes"]["4"]["status"] = "incomplete"; bad["pass 4 incomplete"] = r
    r = _complete_result(transport="ssh"); bad["ssh pass"] = r
    for name, res in bad.items():
        calls, acc, exc, tr = _pwc_drive(W, res)
        assert isinstance(exc, W.PwcTrialIncompleteError), f"{name}: {type(exc).__name__}: {exc}"
        assert not calls["builder"] and acc["bidirectional"] == [], f"{name}: reached the accumulator"
    calls, acc, exc, tr = _pwc_drive(W, _complete_result())
    assert exc is None and len(calls["builder"]) == 1, (exc, calls["builder"])
    # constant-only: passes 3-4 not run is complete
    calls, acc, exc, tr = _pwc_drive(W, _complete_result(phases=("1", "2")), both=False)
    assert exc is None and len(calls["builder"]) == 1, exc
    return (f"{len(bad)} incomplete shapes refused before the accumulator and before a score; "
            f"complete four-pass and constant-only trials go through")


def d_prov(woi_mod=None):
    W = woi_mod or __import__("window_optimizer_integration_final")
    calls, acc, exc, tr = _pwc_drive(W, _complete_result(verified=False))
    assert isinstance(exc, W.PwcThresholdProvenanceError), f"{type(exc).__name__}: {exc}"
    assert not calls["builder"] and acc["bidirectional"] == []
    return "a pass without a verified threshold record is refused before ingress"


def d_credit(woi_mod=None):
    W = woi_mod or __import__("window_optimizer_integration_final")
    from utils.step1_search_ledger import credit_decision
    lo, hi = 2 ** 31, 2 ** 31 + 1000

    def credit(ev, phases):
        return credit_decision(requested_interval=(lo, hi), required_phases=phases,
                               evidence_by_trial={0: ev}, trial_states={0: "COMPLETE"})
    ev = W._step1_pwc_sweep_evidence(_complete_result(), "java_lcg", True)
    ok, code, det, _ = credit(ev, (1, 2, 3, 4))
    assert ok, f"complete four-pass trial not credited: {code} {det} {ev}"
    ev = W._step1_pwc_sweep_evidence(_complete_result(survivors=False), "java_lcg", True)
    assert credit(ev, (1, 2, 3, 4))[0], "zero-survivor complete trial not credited"
    ev = W._step1_pwc_sweep_evidence(_complete_result(phases=("1", "2")), "java_lcg", False)
    assert credit(ev, (1, 2))[0], "constant-only complete trial not credited"
    for name, res in (("ssh", _complete_result(transport="ssh")),
                      ("unverified", _complete_result(verified=False)),
                      ("short interval", _complete_result(hi=hi - 1))):
        ev = W._step1_pwc_sweep_evidence(res, "java_lcg", True)
        assert not credit(ev, (1, 2, 3, 4))[0], f"{name} credited: {ev}"
    r = _complete_result(); r["pwc_trial_complete"] = False
    assert W._step1_pwc_sweep_evidence(r, "java_lcg", True) is None
    # wired: the branch stamps this evidence on the trial result
    calls, acc, exc, tr = _pwc_drive(W, _complete_result())
    assert tr.step1_sweep_evidence["swept_intervals"]["4"] == [lo, hi], tr.step1_sweep_evidence
    return ("complete four-pass, zero-survivor and constant-only trials credited by the unchanged "
            "credit_decision; SSH, unverified and short-interval trials not; wired on the trial")


def d_cli(wo_src=None):
    tree = ast.parse(wo_src or live(WO_PATH))
    m = func(tree, "main")
    flags = [n for n in ast.walk(m) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument"
             and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--pwc-coordinator-address"]
    assert len(flags) == 1, "no --pwc-coordinator-address flag"
    dflt = [k.value for k in flags[0].keywords if k.arg == "default"]
    assert dflt and isinstance(dflt[0], ast.Constant) and dflt[0].value is None, "flag has a default"
    # only the calls that can select PWC (they pass use_persistent_workers); main's random,
    # grid and evolutionary calls pass no engine flag at all (pre-existing, not stage 3)
    calls = [n for n in ast.walk(m) if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") in ("run_bayesian_optimization", "run_with_config")
             and any(k.arg == "use_persistent_workers" for k in n.keywords)]
    assert len(calls) == 2, f"expected the two PWC-capable calls, found {len(calls)}"
    for c in calls:
        assert any(k.arg == "pwc_coordinator_address" for k in c.keywords), \
            f"{c.func.id} call does not pass the dial address"
    return "--pwc-coordinator-address (no default) is passed by main to both entry points"


def d_fleet(wo_src=None):
    tree = ast.parse(wo_src or live(WO_PATH))
    calls = [n for n in ast.walk(func(tree, "main")) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "run_start_dataset_gate"]
    assert len(calls) == 1
    kw = {k.arg: k.value for k in calls[0].keywords}
    assert "require_fleet" in kw and "use_persistent_workers" in ast.unparse(kw["require_fleet"]), \
        "run_start_dataset_gate is not told to require the fleet for PWC"
    assert "use_range_miner" in ast.unparse(kw["miner_backed"]), "miner_backed changed"
    from miner import dataset_authority as DA          # read-only use of the unchanged function
    for miner_backed, require_fleet, raises in ((True, False, True), (False, True, True),
                                                (False, False, False)):
        try:
            DA.resolve_absent_fleet_status(None, manifest_path="/nonexistent", miner_backed=miner_backed,
                                           require_fleet=require_fleet, context="t")
            r = False
        except Exception:                                           # noqa: BLE001
            r = True
        assert r == raises, (miner_backed, require_fleet, r)
    return ("main passes require_fleet=use_persistent_workers; the unchanged authority then fails "
            "a PWC run closed on a missing provisioning manifest, as it does the miner")


WATCHER_PATH = os.path.join(ROOT, "agents", "watcher_agent.py")


def d_watcher_fleet(w_src=None):
    """[PWC-STAGE3 S3-9, owner yes 2026-11-20] WATCHER passes require_fleet for PWC at its
    one resolve_absent_fleet_status call. Removing exactly that keyword must give a module
    AST-identical to the base: nothing else in WATCHER moved."""
    base = ast.parse(git_show("agents/watcher_agent.py"))
    new = ast.parse(w_src or live(WATCHER_PATH))
    calls = [n for n in ast.walk(new) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "resolve_absent_fleet_status"]
    assert len(calls) == 1, f"{len(calls)} resolve_absent_fleet_status calls"
    kw = {k.arg: k.value for k in calls[0].keywords}
    assert "require_fleet" in kw and "use_persistent_workers" in ast.unparse(kw["require_fleet"]), \
        "WATCHER does not require the fleet for PWC"
    assert ast.unparse(kw["miner_backed"]) == "bool(final_params.get('use_range_miner'))", "miner_backed changed"

    class Strip(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if getattr(node.func, "attr", "") == "resolve_absent_fleet_status":
                node.keywords = [k for k in node.keywords if k.arg != "require_fleet"]
            return node
    assert dump(Strip().visit(new)) == dump(base), "agents/watcher_agent.py differs beyond the one keyword"
    return ("WATCHER: require_fleet=use_persistent_workers on its one fleet-status call; the "
            "module minus that keyword is AST-identical to " + BASE)


GATES = [
    ("G-MINER-SUBTREE", d_miner_subtree),
    ("G-MINER-DEFS", d_miner_defs),
    ("G-WO-FUNCS", d_wo_funcs),
    ("G-MINER-SETTINGS", d_miner_settings),
    ("G-MINER-JOBS", d_miner_jobs),
    ("G-MANIFEST", d_manifest),
    ("G-GUARD", d_guard),
    ("G-PWC-CALL", d_pwc_call),
    ("G-INCOMPLETE", d_incomplete),
    ("G-PROV", d_prov),
    ("G-CREDIT", d_credit),
    ("G-CLI", d_cli),
    ("G-FLEET", d_fleet),
    ("G-WATCHER-FLEET", d_watcher_fleet),
]

# (label, file, old, new, detector, how)  how: "woi_src" | "wo_src" | "woi_mod" | "wo_mod"
MUTANTS = [
    ("M-MINER-1 miner subtree: serve_timeout default changed", WOI_PATH,
     "            serve_timeout          = getattr(coordinator, 'serve_timeout', None),\n",
     "            serve_timeout          = getattr(coordinator, 'serve_timeout', 30),\n",
     d_miner_subtree, "woi_src"),
    ("M-MINER-2a preamble: the miner switch's default changed", WOI_PATH,
     "    _use_miner = getattr(coordinator, 'use_range_miner', False)\n",
     "    _use_miner = getattr(coordinator, 'use_range_miner', None)\n",
     d_miner_subtree, "woi_src"),
    ("M-MINER-2b miner subtree: trial wall context changed", WOI_PATH,
     '                         context=f"run_bidirectional_test miner trial {trial_number}")\n',
     '                         context=f"miner trial {trial_number}")\n',
     d_miner_subtree, "woi_src"),
    ("M-MINER-3 _miner_residues_for_config changed", WOI_PATH,
     "def _miner_residues_for_config(config, dataset_path: str):\n",
     "def _miner_residues_for_config(config, dataset_path: str = ''):\n",
     d_miner_defs, "woi_src"),
    ("M-MINER-4 miner wiring in run_bayesian_optimization changed", WO_PATH,
     "    coordinator.miner_substripes  = miner_substripes\n",
     "    coordinator.miner_substripes  = miner_substripes + 0\n",
     d_wo_funcs, "wo_src"),
    ("M-MINER-5 main: miner_backed keyword changed", WO_PATH,
     "            miner_backed=bool(getattr(args, 'use_range_miner', False)),\n",
     "            miner_backed=bool(getattr(args, 'use_range_miner', True)),\n",
     d_wo_funcs, "wo_src"),
    ("M-MINER-6 dial address set on every coordinator (runtime)", WO_PATH,
     "    if use_persistent_workers:\n        # [PWC-STAGE3] set only on a PWC run,",
     "    if True:\n        # [PWC-STAGE3] set only on a PWC run,",
     d_miner_settings, "wo_mod"),
    ("M-MINER-7 run_trial_miner argument changed (runtime)", WOI_PATH,
     "            miner_substripes       = getattr(coordinator, 'miner_substripes', 8),\n",
     "            miner_substripes       = getattr(coordinator, 'miner_substripes', 9),\n",
     d_miner_jobs, "woi_mod"),
    ("M-PWC-1 seed-start guard still refuses PWC", WO_PATH,
     "    if seed_start != 0 and use_zmq_sqlite:\n",
     "    if seed_start != 0 and (use_zmq_sqlite or use_persistent_workers):\n",
     d_guard, "wo_mod"),
    ("M-PWC-2 PWC not handed seed_start", WOI_PATH,
     "            seed_start        = seed_start,                                   # [PWC-STAGE3 ruling 3]\n",
     "", d_pwc_call, "woi_mod"),
    ("M-PWC-3 completeness check removed (Beta Q5)", WOI_PATH,
     "        _pw_comp = _pwc_require_complete_trial(_pw_result, prng_base, test_both_modes,\n"
     "                                               trial_number)\n",
     "        _pw_comp = (_pw_result.get('pwc_completion') or {'passes': {}})\n",
     d_incomplete, "woi_mod"),
    ("M-PWC-4 provenance check removed", WOI_PATH,
     "        _pwc_require_threshold_provenance(_pw_comp, prng_base, test_both_modes, trial_number)\n",
     "", d_prov, "woi_mod"),
    ("M-PWC-5 transport not checked", WOI_PATH,
     '        if acc.get("transport") != "tcp":\n',
     '        if False:\n', d_credit, "woi_mod"),
    ("M-PWC-6 evidence built from the requested interval", WOI_PATH,
     "        out[ph] = [int(iv[0]), int(iv[1])]\n",
     "        out[ph] = [2 ** 31, 2 ** 31 + 1000]\n", d_credit, "woi_mod"),
    ("M-PWC-7 require_fleet not passed", WO_PATH,
     "            require_fleet=bool(getattr(args, 'use_persistent_workers', False)),\n",
     "", d_fleet, "wo_src"),
    ("M-WATCHER-1 require_fleet keyword dropped", WATCHER_PATH,
     "                        require_fleet=bool(final_params.get(\"use_persistent_workers\")),\n",
     "", d_watcher_fleet, "w_src"),
    ("M-WATCHER-2 another WATCHER line changed", WATCHER_PATH,
     "                        context=f\"watcher_step{step}\",\n                        # [PWC-STAGE3]",
     "                        context=f\"watcher-step{step}\",\n                        # [PWC-STAGE3]",
     d_watcher_fleet, "w_src"),
    ("M-PWC-8 dial address not passed by main", WO_PATH,
     "            pwc_coordinator_address=getattr(args, 'pwc_coordinator_address', None),  # PWC-STAGE3\n        )",
     "        )", d_cli, "wo_src"),
    # [SKIP-DELIVERY SWITCH] the allowance is exact
    ("M-SW-1 the allowed miner keyword altered", WOI_PATH,
     '            **({"deliver_skip_range": True} if _skip_delivery_on(coordinator) else {}),\n',
     '            **({"deliver_skip_range": True} if _skip_delivery_on(coordinator) is not None else {}),\n',
     d_miner_subtree, "woi_src"),
    ("M-SW-2 the allowed preflight helper altered", WOI_PATH,
     "    if not deliver_on and recorded is not None:\n",
     "    if False:\n", d_miner_defs, "woi_src"),
    ("M-SW-3 an extra statement beside the allowed preflight call", WOI_PATH,
     "        _step1_skip_delivery_root_arm(_s1_skip_delivery_on, create=True)\n",
     "        _step1_skip_delivery_root_arm(_s1_skip_delivery_on, create=True)\n"
     "        _s1_extra_statement = None\n", d_miner_defs, "woi_src"),
    ("M-SW-4 window_optimizer: the attribute set while off", WO_PATH,
     "    if deliver_skip_range is True:\n        # [SKIP-DELIVERY SWITCH] set ONLY when on",
     "    if deliver_skip_range is not None:\n        # [SKIP-DELIVERY SWITCH] set ONLY when on",
     d_wo_funcs, "wo_src"),
    ("M-SW-5 manifest: the switch defaults on", MAN_PATH,
     '    "deliver_skip_range": false,\n', '    "deliver_skip_range": true,\n',
     d_manifest, "man_src"),
    ("M-SW-6 window_optimizer: attribute set while off (runtime)", WO_PATH,
     "    if deliver_skip_range is True:\n        # [SKIP-DELIVERY SWITCH] set ONLY when on",
     "    coordinator.deliver_skip_range = deliver_skip_range\n    if deliver_skip_range is True:\n        # [SKIP-DELIVERY SWITCH] set ONLY when on",
     d_miner_settings, "wo_mod"),
    ("M-SW-8 Beta R1 Q3: the preflight call moved after checkpoint recovery", WOI_PATH,
     ('        _step1_skip_delivery_preflight(_s1_search_ctx, _s1_skip_delivery_on,\n                                       resume_study=resume_study, study_name=study_name)\n', '                skip_delivery_on=_s1_skip_delivery_on,\n            )\n'),
     ("", '                skip_delivery_on=_s1_skip_delivery_on,\n            )\n        _step1_skip_delivery_preflight(_s1_search_ctx, _s1_skip_delivery_on,\n                                       resume_study=resume_study, study_name=study_name)\n'),
     d_miner_subtree, "woi_src"),
    ("M-SW-9 the same relocation, through the definitions gate", WOI_PATH,
     ('        _step1_skip_delivery_preflight(_s1_search_ctx, _s1_skip_delivery_on,\n                                       resume_study=resume_study, study_name=study_name)\n', '                skip_delivery_on=_s1_skip_delivery_on,\n            )\n'),
     ("", '                skip_delivery_on=_s1_skip_delivery_on,\n            )\n        _step1_skip_delivery_preflight(_s1_search_ctx, _s1_skip_delivery_on,\n                                       resume_study=resume_study, study_name=study_name)\n'),
     d_miner_defs, "woi_src"),
    ("M-SW-10 Beta R3 Q3: the CLI preflight block moved after the Bayesian call", WO_PATH,
     ('    if getattr(args, \'deliver_skip_range\', False) and not args.use_range_miner:\n        parser.error("--deliver-skip-range applies to the RANGE-MINER route only "\n                     "(--use-range-miner); PWC already sends the trial\'s skip range")\n    if getattr(args, \'deliver_skip_range\', False):\n        # [SKIP-DELIVERY SWITCH] the one validator (miner.range_miner_coordinator); a\n        # pinned pair is checked HERE, before any search, study or dispatch\n        from miner.range_miner_coordinator import (\n            validate_skip_delivery as _vsd, MinerMetadataError as _MME)\n        _pins = (getattr(args, \'warm_start_skip_min\', None),\n                 getattr(args, \'warm_start_skip_max\', None))\n        try:\n            _vsd(True, *_pins, where="--deliver-skip-range") if _pins != (None, None) \\\n                else _vsd(True, where="--deliver-skip-range")\n        except _MME as _e:\n            parser.error(str(_e))\n        if getattr(args, \'config_file\', None):\n            parser.error("--deliver-skip-range is not carried by the --config-file route; "\n                         "refused rather than dropped (use the Bayesian search route with "\n                         "a Step 1 search opened with deliver_skip_range: true)")\n', "            deliver_skip_range=getattr(args, 'deliver_skip_range', False),  # SKIP-DELIVERY\n            miner_stripe_size=getattr(args, 'miner_stripe_size', 67_108_864),\n            miner_substripes=getattr(args, 'miner_substripes', 8),\n            miner_output_dir=getattr(args, 'miner_output_dir', None),\n            # [S172 Staging Part B] hop 2 of 3 — CLI end of the canonical route.\n            staging_dir=getattr(args, 'staging_dir', None),\n            # [S172-BP §3] hop 2 of 3 — CLI end of the capacity-control route.\n            staging_workers=getattr(args, 'staging_workers', 4),\n            staging_queue_depth=getattr(args, 'staging_queue_depth', 2),\n            staging_deferred_max=getattr(args, 'staging_deferred_max', None),\n            staging_capacity_timeout=getattr(args, 'staging_capacity_timeout', 600.0),\n            # [S172 STAGING-CAPACITY AMENDMENT §1.3] hop 2 of 3 for BOTH high-waters.\n            staging_high_water_files=getattr(args, 'staging_high_water_files', None),\n            staging_high_water_bytes=getattr(\n                args, 'staging_high_water_bytes', 16 * 1024 ** 3),\n            # [S172 Phase-5 D6.2] the operator route's CLI end of hop 2.\n            resume_checkpoint=getattr(args, 'resume_checkpoint', ''),\n            # [STEP1-SEARCH stage 1] hop 2 of 3, CLI end.\n            step1_search_id=getattr(args, 'step1_search_id', ''),\n        )\n"),
     ("", '            deliver_skip_range=getattr(args, \'deliver_skip_range\', False),  # SKIP-DELIVERY\n            miner_stripe_size=getattr(args, \'miner_stripe_size\', 67_108_864),\n            miner_substripes=getattr(args, \'miner_substripes\', 8),\n            miner_output_dir=getattr(args, \'miner_output_dir\', None),\n            # [S172 Staging Part B] hop 2 of 3 — CLI end of the canonical route.\n            staging_dir=getattr(args, \'staging_dir\', None),\n            # [S172-BP §3] hop 2 of 3 — CLI end of the capacity-control route.\n            staging_workers=getattr(args, \'staging_workers\', 4),\n            staging_queue_depth=getattr(args, \'staging_queue_depth\', 2),\n            staging_deferred_max=getattr(args, \'staging_deferred_max\', None),\n            staging_capacity_timeout=getattr(args, \'staging_capacity_timeout\', 600.0),\n            # [S172 STAGING-CAPACITY AMENDMENT §1.3] hop 2 of 3 for BOTH high-waters.\n            staging_high_water_files=getattr(args, \'staging_high_water_files\', None),\n            staging_high_water_bytes=getattr(\n                args, \'staging_high_water_bytes\', 16 * 1024 ** 3),\n            # [S172 Phase-5 D6.2] the operator route\'s CLI end of hop 2.\n            resume_checkpoint=getattr(args, \'resume_checkpoint\', \'\'),\n            # [STEP1-SEARCH stage 1] hop 2 of 3, CLI end.\n            step1_search_id=getattr(args, \'step1_search_id\', \'\'),\n        )\n        if getattr(args, \'deliver_skip_range\', False) and not args.use_range_miner:\n            parser.error("--deliver-skip-range applies to the RANGE-MINER route only "\n                         "(--use-range-miner); PWC already sends the trial\'s skip range")\n        if getattr(args, \'deliver_skip_range\', False):\n            # [SKIP-DELIVERY SWITCH] the one validator (miner.range_miner_coordinator); a\n            # pinned pair is checked HERE, before any search, study or dispatch\n            from miner.range_miner_coordinator import (\n                validate_skip_delivery as _vsd, MinerMetadataError as _MME)\n            _pins = (getattr(args, \'warm_start_skip_min\', None),\n                     getattr(args, \'warm_start_skip_max\', None))\n            try:\n                _vsd(True, *_pins, where="--deliver-skip-range") if _pins != (None, None) \\\n                    else _vsd(True, where="--deliver-skip-range")\n            except _MME as _e:\n                parser.error(str(_e))\n            if getattr(args, \'config_file\', None):\n                parser.error("--deliver-skip-range is not carried by the --config-file route; "\n                             "refused rather than dropped (use the Bayesian search route with "\n                             "a Step 1 search opened with deliver_skip_range: true)")\n'),
     d_wo_funcs, "wo_src"),
    ("M-SW-7 integration: the switch passed while off (runtime)", WOI_PATH,
     '            **({"deliver_skip_range": True} if _skip_delivery_on(coordinator) else {}),\n',
     '            **({"deliver_skip_range": True} if True else {}),\n', d_miner_jobs, "woi_mod"),
]


def run_mutant(label, path, old, new, detector, how):
    src = live(path)
    edits = list(zip(old, new)) if isinstance(old, tuple) else [(old, new)]
    msrc = src
    for o, n_ in edits:
        n = msrc.count(o)
        assert n == 1, f"mutant anchor occurs {n} times: {o[:80]!r}"
        msrc = msrc.replace(o, n_)
    detector()                                                      # positive control
    if how.endswith("_src"):
        arg = msrc
    else:
        name = "woi_mut_s3" if path == WOI_PATH else "wo_mut_s3"
        arg = load_module(msrc, name, path)
    try:
        with quiet():
            detector(arg)
    except AssertionError as exc:
        return f"killed: {str(exc).splitlines()[0][:110]}"
    except Exception as exc:                                        # noqa: BLE001
        raise AssertionError(f"UNATTRIBUTABLE kill: {type(exc).__name__}: {exc}")
    raise AssertionError(f"SURVIVED: {label}")


def main():
    import logging
    logging.disable(logging.CRITICAL)
    print("VERIFICATION test_pwc_backup_stage3 START", flush=True)
    passed = 0
    for name, fn in GATES:
        try:
            print(f"[PASS] {name}: {fn()}", flush=True)
            passed += 1
        except Exception as exc:                                    # noqa: BLE001
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
    killed = 0
    for m in MUTANTS:
        try:
            print(f"[PASS] {m[0]} -> {run_mutant(*m)}", flush=True)
            killed += 1
        except Exception as exc:                                    # noqa: BLE001
            print(f"[FAIL] {m[0]}: {type(exc).__name__}: {exc}", flush=True)
    print(f"\nGATES {passed}/{len(GATES)} PASS")
    print(f"MUTANTS {killed}/{len(MUTANTS)} killed")
    ok = passed == len(GATES) and killed == len(MUTANTS)
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    print("VERIFICATION test_pwc_backup_stage3 COMPLETE", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
