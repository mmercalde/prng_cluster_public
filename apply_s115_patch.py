#!/usr/bin/env python3
"""
apply_s115_patch.py  (v3 — anchors verified against LIVE Zeus files)
=====================================================================
Applies all S115 proposal changes using anchor-based replacement.
Anchors verified against actual Zeus file hashes:
  coordinator.py                    2782 lines  44a1d40b...
  window_optimizer_bayesian.py       713 lines  d2110591...
  window_optimizer.py               1043 lines  4cc965c3...
  window_optimizer_integration_final.py 599 lines  269bed7e...

Usage:
    python3 apply_s115_patch.py [--dry-run] [--repo-path PATH]

After patching:
    python3 dry_run_s115.py   (require 12/12 PASS)
"""
import sys, os, shutil, argparse
from pathlib import Path
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--dry-run', action='store_true')
parser.add_argument('--repo-path', type=str, default='.')
args = parser.parse_args()
REPO    = Path(args.repo_path).resolve()
DRY_RUN = args.dry_run

G="\033[92m"; R="\033[91m"; Y="\033[93m"; B="\033[1m"; Z="\033[0m"
applied=[]; failed=[]
_backed_up = set()

def read(p):
    with open(p) as f: return f.read()

def write(p, c):
    if not DRY_RUN:
        with open(p,'w') as f: f.write(c)

def backup(path):
    if str(path) not in _backed_up and not DRY_RUN:
        ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
        bak = Path(str(path)+f'.bak_s115_{ts}')
        shutil.copy2(path, bak)
        print(f"   💾 Backup: {bak.name}")
        _backed_up.add(str(path))

def patch(pid, filepath, anchor, replacement, desc):
    path = REPO / filepath
    src  = read(path)
    if anchor not in src:
        print(f"{R}❌ FAIL{Z}  {pid}  anchor not found in {filepath}")
        print(f"         anchor[:80]: {repr(anchor[:80])}")
        failed.append((pid, desc, "anchor not found"))
        return False
    n = src.count(anchor)
    if n > 1:
        print(f"{R}❌ FAIL{Z}  {pid}  anchor appears {n} times (not unique)")
        failed.append((pid, desc, f"anchor not unique ({n}x)"))
        return False
    if replacement in src:
        print(f"{Y}⏭  SKIP{Z}  {pid}  already applied  [{desc}]")
        applied.append((pid,"SKIP")); return True
    backup(path)
    write(path, src.replace(anchor, replacement, 1))
    mode = "[DRY RUN] " if DRY_RUN else ""
    print(f"{G}✅ OK  {Z}  {mode}{pid}  [{desc}]")
    applied.append((pid,"OK")); return True

print(f"\n{B}{'='*65}{Z}")
print(f"{B}S115 PATCH SCRIPT  (v3 — anchors from live Zeus files){Z}")
print(f"{'  [DRY RUN]' if DRY_RUN else ''}")
print(f"{B}{'='*65}{Z}\n")

# ── P1: coordinator.__init__ — node_allowlist param ───────────────────────────
# Zeus: unchanged from GitHub (coordinator.py hash matches zeus_sim exactly)
patch("P1","coordinator.py",
"""    def __init__(self, config_file: str = "distributed_config.json",
                 seed_cap_nvidia: int = 40000, seed_cap_amd: int = 19000, seed_cap_default: int = 19000,
                 max_concurrent: int = 8, max_per_node: int = 4, max_local_concurrent: Optional[int] = None,
                 job_timeout: int = 600, resume_policy: str = 'prompt'):
        self.config_file = config_file
        self.nodes: List[WorkerNode] = []""",
"""    def __init__(self, config_file: str = "distributed_config.json",
                 seed_cap_nvidia: int = 40000, seed_cap_amd: int = 19000, seed_cap_default: int = 19000,
                 max_concurrent: int = 8, max_per_node: int = 4, max_local_concurrent: Optional[int] = None,
                 job_timeout: int = 600, resume_policy: str = 'prompt',
                 node_allowlist: Optional[List[str]] = None):
        # S115 M1/M4: CRITICAL — set before load_configuration() which runs inside __init__
        self.node_allowlist = node_allowlist
        self.config_file = config_file
        self.nodes: List[WorkerNode] = []""",
"coordinator.__init__: node_allowlist param (M1/M4)")

# ── P2: coordinator.load_configuration — allowlist filter + guard ─────────────
patch("P2","coordinator.py",
"""                self.nodes.append(node)
                print(f\"Configured node {node.hostname}: {node.gpu_count}x {node.gpu_type}\")
            # Load sieve defaults from config""",
"""                self.nodes.append(node)
                print(f\"Configured node {node.hostname}: {node.gpu_count}x {node.gpu_type}\")
            # S115 M1/M4: apply allowlist filter after all nodes loaded
            if getattr(self, 'node_allowlist', None) is not None:
                _all = list(self.nodes)
                self.nodes = [n for n in self.nodes if n.hostname in self.node_allowlist]
                print(f\"   Node allowlist active: {len(self.nodes)}/{len(_all)} nodes \"
                      f\"({[n.hostname for n in self.nodes]})\")
                if not self.nodes:
                    raise ValueError(
                        f\"node_allowlist {self.node_allowlist} matched no nodes.\\n\"
                        f\"Available hostnames: {[n.hostname for n in _all]}\\n\"
                        f\"Check for hostname vs IP mismatch in distributed_config.json.\"
                    )
            # Load sieve defaults from config""",
"load_configuration: allowlist filter + zero-node guard (M1/M4)")

# ── P3: integration_final — optuna import guard at top ────────────────────────
patch("P3","window_optimizer_integration_final.py",
'#!/usr/bin/env python3\n"""\nWindow Optimizer Integration - WITH VARIABLE SKIP SUPPORT',
"""#!/usr/bin/env python3
# S115 N2: guarded optuna import — pruning only fires if optuna present
try:
    import optuna as _optuna_module
    _OPTUNA_AVAILABLE = True
except ImportError:
    _optuna_module = None
    _OPTUNA_AVAILABLE = False
\"\"\"
Window Optimizer Integration - WITH VARIABLE SKIP SUPPORT""",
"integration_final: optuna import guard (N2)")

# ── P4: run_bidirectional_test — add optuna_trial param ───────────────────────
patch("P4","window_optimizer_integration_final.py",
"""def run_bidirectional_test(coordinator,
                           config: WindowConfig,
                           dataset_path: str,
                           seed_start: int,
                           seed_count: int,
                           prng_base: str = 'java_lcg',
                           test_both_modes: bool = False,
                           forward_threshold: float = 0.01,
                           reverse_threshold: float = 0.01,
                           trial_number: int = 0,
                           accumulator: Dict[str, List] = None) -> TestResult:""",
"""def run_bidirectional_test(coordinator,
                           config: WindowConfig,
                           dataset_path: str,
                           seed_start: int,
                           seed_count: int,
                           prng_base: str = 'java_lcg',
                           test_both_modes: bool = False,
                           forward_threshold: float = 0.01,
                           reverse_threshold: float = 0.01,
                           trial_number: int = 0,
                           accumulator: Dict[str, List] = None,
                           optuna_trial=None) -> TestResult:  # S115 M2""",
"run_bidirectional_test: optuna_trial param (M2)")

# ── P5a-d: output path suffixes ───────────────────────────────────────────────
patch("P5a","window_optimizer_integration_final.py",
"        f'results/window_opt_forward_{config.window_size}_{config.offset}.json',",
"        f'results/window_opt_forward_{config.window_size}_{config.offset}_t{trial_number}.json',  # S115 M3",
"output path: forward constant (M3)")

patch("P5b","window_optimizer_integration_final.py",
"        f'results/window_opt_reverse_{config.window_size}_{config.offset}.json',",
"        f'results/window_opt_reverse_{config.window_size}_{config.offset}_t{trial_number}.json',  # S115 M3",
"output path: reverse constant (M3)")

patch("P5c","window_optimizer_integration_final.py",
"            f'results/window_opt_forward_hybrid_{config.window_size}_{config.offset}.json',",
"            f'results/window_opt_forward_hybrid_{config.window_size}_{config.offset}_t{trial_number}.json',  # S115 M3",
"output path: forward hybrid (M3)")

patch("P5d","window_optimizer_integration_final.py",
"            f'results/window_opt_reverse_hybrid_{config.window_size}_{config.offset}.json',",
"            f'results/window_opt_reverse_hybrid_{config.window_size}_{config.offset}_t{trial_number}.json',  # S115 M3",
"output path: reverse hybrid (M3)")

# ── P6: M2 pruning hook ────────────────────────────────────────────────────────
patch("P6","window_optimizer_integration_final.py",
"""    # v3.0: Extract full records with per-seed match_rate
    forward_records = extract_survivor_records(forward_result)
    print(f\"      Forward: {len(forward_records):,} survivors\")

    print(f\"    Running REVERSE sieve""",
"""    # v3.0: Extract full records with per-seed match_rate
    forward_records = extract_survivor_records(forward_result)
    print(f\"      Forward: {len(forward_records):,} survivors\")

    # S115 M2: prune dead trials (forward==0) before expensive reverse sieve
    if optuna_trial is not None:
        if not _OPTUNA_AVAILABLE:
            print(\"      \u26a0\ufe0f  optuna_trial passed but Optuna not installed \u2014 pruning disabled.\")
        elif len(forward_records) == 0:
            print(f\"      \u2702\ufe0f  PRUNED  trial={optuna_trial.number}  \"
                  f\"window={config.window_size}  offset={config.offset}  \"
                  f\"skip={config.skip_min}-{config.skip_max}  forward_count=0\")
            raise _optuna_module.exceptions.TrialPruned()

    print(f\"    Running REVERSE sieve""",
"pruning hook: forward_count==0 raises TrialPruned (M2/N2)")

# ── P7: optimize_window — n_parallel + partition cache ───────────────────────
# Zeus: S116 already added resume_study + study_name to signature
patch("P7","window_optimizer_integration_final.py",
"""    def optimize_window(self,
                        dataset_path: str,
                        seed_start: int = 0,
                        seed_count: int = 10_000_000,
                        prng_base: str = 'java_lcg',
                        test_both_modes: bool = False,
                        strategy_name: str = 'bayesian',
                        max_iterations: int = 50,
                        output_file: str = 'window_optimization.json',
                        resume_study: bool = False,
                        study_name: str = ''):""",
"""    def optimize_window(self,
                        dataset_path: str,
                        seed_start: int = 0,
                        seed_count: int = 10_000_000,
                        prng_base: str = 'java_lcg',
                        test_both_modes: bool = False,
                        strategy_name: str = 'bayesian',
                        max_iterations: int = 50,
                        output_file: str = 'window_optimization.json',
                        resume_study: bool = False,
                        study_name: str = '',
                        n_parallel: int = 1):  # S115 M1
        # S115 M1/M4: Partition map (IPs from distributed_config.json)
        # P0: localhost+192.168.3.120 (10 GPUs, ~141 TFLOPS)
        # P1: 192.168.3.154+192.168.3.162 (16 GPUs, ~142 TFLOPS)
        # M5: imbalance documented — TFLOPS near-equal; logged per trial
        _PARALLEL_PARTITIONS = {
            0: ['localhost', '192.168.3.120'],
            1: ['192.168.3.154', '192.168.3.162'],
        }
        _partition_coordinators = {}

        def _get_partition_coordinator(idx):
            if idx not in _partition_coordinators:
                from coordinator import MultiGPUCoordinator as _MCC
                coord = _MCC(
                    config_file=getattr(self, 'config_file', 'distributed_config.json'),
                    node_allowlist=_PARALLEL_PARTITIONS[idx % len(_PARALLEL_PARTITIONS)]
                )
                coord.load_configuration()
                coord.create_gpu_workers()
                _partition_coordinators[idx] = coord
                print(f\"   \U0001f500 Partition {idx} coordinator ready: {_PARALLEL_PARTITIONS[idx % len(_PARALLEL_PARTITIONS)]}\")
            return _partition_coordinators[idx]

        def _shutdown_partition_coordinators():
            for c in _partition_coordinators.values():
                try: c.ssh_pool.cleanup_all()
                except Exception: pass
            _partition_coordinators.clear()

        if False: pass  # indent anchor""",
"optimize_window: n_parallel + partition cache + shutdown (M1/M5/N3)")

# ── P8: test_config — routing + log ───────────────────────────────────────────
patch("P8","window_optimizer_integration_final.py",
"""        def test_config(config,
                        ss=seed_start, sc=seed_count,
                        ft=bounds.default_forward_threshold,
                        rt=bounds.default_reverse_threshold):
            trial_counter['count'] += 1""",
"""        def test_config(config,
                        ss=seed_start, sc=seed_count,
                        ft=bounds.default_forward_threshold,
                        rt=bounds.default_reverse_threshold,
                        optuna_trial=None):  # S115 M2
            trial_counter['count'] += 1
            # S115 M1/M5: route to partition coordinator
            if optuna_trial is not None and n_parallel > 1:
                _part = optuna_trial.number % n_parallel
                _coord = _get_partition_coordinator(_part)
                print(f\"   \U0001f500 Trial {optuna_trial.number} \u2192 Partition {_part} ({_PARALLEL_PARTITIONS[_part]})\")
            else:
                _coord = self""",
"test_config: optuna_trial + partition routing + log (M1/M5)")

# ── P9: OptunaBayesianSearch.__init__ — enable_pruning + n_parallel ───────────
# Zeus: __init__ unchanged from GitHub — anchor is exact
patch("P9","window_optimizer_bayesian.py",
"""    def __init__(self, n_startup_trials=5, seed=None):
        \"\"\"
        Args:
            n_startup_trials: Number of random trials before using TPE
            seed: Random seed for reproducibility
        \"\"\"
        if not OPTUNA_AVAILABLE:
            raise ImportError(\"Optuna not available. Install with: pip install optuna\")""",
"""    def __init__(self, n_startup_trials=5, seed=None,
                 enable_pruning=False, n_parallel=1):  # S115 R3
        \"\"\"
        Args:
            n_startup_trials: Number of random trials before using TPE
            seed: Random seed for reproducibility
            enable_pruning: If True enable forward_count==0 pruning (S115 M2)
            n_parallel: Number of parallel partitions (S115 M1)
        \"\"\"
        self.enable_pruning = enable_pruning
        self.n_parallel = n_parallel
        if not OPTUNA_AVAILABLE:
            raise ImportError(\"Optuna not available. Install with: pip install optuna\")""",
"OptunaBayesianSearch.__init__: enable_pruning + n_parallel (R3)")

# ── P10: study creation — upgrade Zeus S116 resume block with JournalStorage ──
# Zeus anchor: the actual S116 resume block (verified from live output)
patch("P10","window_optimizer_bayesian.py",
"""        # Create persistent storage for the study
        import time
        import glob as _glob
        import os as _os

        # --- Resume logic (S114 patch) ---
        # resume_study=True: find most recent incomplete DB and continue
        # resume_study=False (default): always create fresh study
        _resume = False
        _resumed_completed = 0
        _fresh_study_name = f\"window_opt_{int(time.time())}\"
        _fresh_storage_path = f\"sqlite:////home/michael/distributed_prng_analysis/optuna_studies/{_fresh_study_name}.db\"""",
"""        # Create persistent storage for the study
        import time
        import glob as _glob
        import os as _os

        # --- Resume logic (S114/S116 patch, upgraded S115 R4) ---
        # resume_study=True: find most recent incomplete DB and continue
        # resume_study=False (default): always create fresh study
        # S115 R4: n_parallel>1 uses JournalFileBackend (no SQLite write-lock contention)
        _resume = False
        _resumed_completed = 0
        _fresh_study_name = f\"window_opt_{int(time.time())}\"
        _fresh_storage_path = f\"sqlite:////home/michael/distributed_prng_analysis/optuna_studies/{_fresh_study_name}.db\"""",
"study block: tag S115 R4 comment (anchor for JournalStorage insertion)")

# ── P10b: insert JournalStorage branch just before study = optuna.create_study ─
patch("P10b","window_optimizer_bayesian.py",
"""        if not _resume:
            study_name = _fresh_study_name
            storage_path = _fresh_storage_path
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            direction='maximize',
            sampler=sampler,
            load_if_exists=_resume
        )""",
"""        if not _resume:
            # S115 R4: JournalFileBackend for n_parallel>1 (no SQLite write-lock)
            if self.n_parallel > 1:
                _journal_path = f\"/home/michael/distributed_prng_analysis/optuna_studies/{_fresh_study_name}.jsonl\"
                if _os.path.exists(_journal_path):
                    raise RuntimeError(f\"Journal file already exists: {_journal_path}\")
                storage_path = optuna.storages.JournalStorage(
                    optuna.storages.journal.JournalFileBackend(_journal_path)
                )
                study_name = _fresh_study_name
                print(f\"   \U0001f4ca Optuna study (journal): {_journal_path}\")
            else:
                study_name = _fresh_study_name
                storage_path = _fresh_storage_path

        # S115 M2: ThresholdPruner as secondary safety net
        _pruner = optuna.pruners.ThresholdPruner(lower=1.0) if self.enable_pruning else optuna.pruners.NopPruner()

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            direction='maximize',
            sampler=sampler,
            pruner=_pruner,
            load_if_exists=_resume
        )""",
"study creation: JournalStorage branch + ThresholdPruner (S115 R4/M2)")

# ── P11: pass optuna_trial to objective_function ──────────────────────────────
# Zeus line 315: result = objective_function(config) — same as expected
patch("P11","window_optimizer_bayesian.py",
"            # Evaluate configuration\n            result = objective_function(config)\n            result.iteration = trial.number",
"            # Evaluate configuration — S115 M2: pass trial for pruning hook\n            result = objective_function(config, optuna_trial=trial)\n            result.iteration = trial.number",
"optuna_objective: pass optuna_trial=trial (M2)")

# ── P12: prune telemetry + final summary ──────────────────────────────────────
# Zeus anchor: _trials_to_run = max_iterations - _resumed_completed (line 432)
# and study.optimize on line 440 with _incremental_callback
patch("P12","window_optimizer_bayesian.py",
"""        _trials_to_run = max_iterations - _resumed_completed
""",
"""        _trials_to_run = max_iterations - _resumed_completed
        if self.n_parallel > 1 and not _resume:
            _trials_to_run = max_iterations  # journal storage handles its own count
""",
"P12a: n_parallel journal storage trial count guard")

patch("P12b","window_optimizer_bayesian.py",
"        study.optimize(optuna_objective, n_trials=_trials_to_run, callbacks=[_incremental_callback])",
"""        # S115 R1: prune telemetry callback
        def _prune_telemetry(study, trial):
            _nt = len(study.trials)
            if _nt > 0 and _nt % 10 == 0:
                _np = sum(1 for t in study.trials if t.state.name=='PRUNED')
                _nc = sum(1 for t in study.trials if t.state.name=='COMPLETE')
                print(f\"   \U0001f4ca Prune telemetry ({_nt} trials): complete={_nc}  pruned={_np}  rate={_np/_nt*100:.0f}%\")

        study.optimize(optuna_objective, n_trials=_trials_to_run,
                       callbacks=[_incremental_callback, _prune_telemetry],
                       n_jobs=self.n_parallel)

        _nt = len(study.trials); _np = sum(1 for t in study.trials if t.state.name=='PRUNED')
        _nc = sum(1 for t in study.trials if t.state.name=='COMPLETE')
        if _nt > 0:
            print(f\"\\nPRUNING SUMMARY\\n  Total: {_nt}  Pruned: {_np} ({_np/_nt*100:.1f}%)  Complete: {_nc}\")""",
"prune telemetry + final summary, n_jobs=n_parallel (R1)")

# ── P13: BayesianOptimization.__init__ — forward enable_pruning + n_parallel ──
# Zeus: __init__ unchanged, search() already has resume params
patch("P13","window_optimizer.py",
"""    def __init__(self, n_initial=5):
        self.n_initial = n_initial
        self.optuna_search = None

        # Try to use real Optuna implementation
        if BAYESIAN_AVAILABLE:
            try:
                from window_optimizer_bayesian import OptunaBayesianSearch
                self.optuna_search = OptunaBayesianSearch(n_startup_trials=n_initial, seed=None)""",
"""    def __init__(self, n_initial=5, enable_pruning=False, n_parallel=1):  # S115 R3
        self.n_initial = n_initial
        self.enable_pruning = enable_pruning
        self.n_parallel = n_parallel
        self.optuna_search = None

        # Try to use real Optuna implementation
        if BAYESIAN_AVAILABLE:
            try:
                from window_optimizer_bayesian import OptunaBayesianSearch
                self.optuna_search = OptunaBayesianSearch(
                    n_startup_trials=n_initial, seed=None,
                    enable_pruning=enable_pruning, n_parallel=n_parallel)  # S115 R3""",
"BayesianOptimization.__init__: forward enable_pruning + n_parallel (R3)")

# ── P14: run_bayesian_optimization — add enable_pruning + n_parallel ──────────
# Zeus already has resume_study + study_name in signature
patch("P14","window_optimizer.py",
"""def run_bayesian_optimization(
    lottery_file: str,
    trials: int,
    output_config: str,
    seed_count: int = 10_000_000,
    prng_type: str = 'java_lcg',
    test_both_modes: bool = False,
    strategy_name: str = 'bayesian',  # 'bayesian' or 'random'
    resume_study: bool = False,
    study_name: str = ''
) -> Dict[str, Any]:""",
"""def run_bayesian_optimization(
    lottery_file: str,
    trials: int,
    output_config: str,
    seed_count: int = 10_000_000,
    prng_type: str = 'java_lcg',
    test_both_modes: bool = False,
    strategy_name: str = 'bayesian',  # 'bayesian' or 'random'
    resume_study: bool = False,
    study_name: str = '',
    enable_pruning: bool = False,     # S115 R3
    n_parallel: int = 1               # S115 M1
) -> Dict[str, Any]:""",
"run_bayesian_optimization: enable_pruning + n_parallel params (M1/R3)")

# ── P15: CLI flags — add --enable-pruning + --n-parallel ──────────────────────
# Zeus: --resume-study and --study-name appear BEFORE --test-both-modes
# with different help text than our simulation. Anchor on exact Zeus text.
patch("P15","window_optimizer.py",
"""    parser.add_argument('--resume-study', action='store_true',
                       help='Resume most recent incomplete Optuna study DB instead of starting fresh. '
                            'Skips warm-start enqueue if study already has trials. '
                            'Default: False (fresh study every run).')
    parser.add_argument('--study-name', type=str, default='',
                       help='Optuna study DB name to resume (e.g. window_opt_1772507547). '
                            'Empty string = auto-select most recent incomplete study. '
                            'Only used when --resume-study is set.')
    parser.add_argument('--test-both-modes', action='store_true',
                       help='Test BOTH constant and variable skip patterns (NEW!)')

    args = parser.parse_args()""",
"""    parser.add_argument('--resume-study', action='store_true',
                       help='Resume most recent incomplete Optuna study DB instead of starting fresh. '
                            'Skips warm-start enqueue if study already has trials. '
                            'Default: False (fresh study every run).')
    parser.add_argument('--study-name', type=str, default='',
                       help='Optuna study DB name to resume (e.g. window_opt_1772507547). '
                            'Empty string = auto-select most recent incomplete study. '
                            'Only used when --resume-study is set.')
    parser.add_argument('--test-both-modes', action='store_true',
                       help='Test BOTH constant and variable skip patterns (NEW!)')
    # S115 R3: pruning + parallelism flags
    parser.add_argument('--enable-pruning', action='store_true', default=False,
                       help='Enable forward_count==0 pruning (~1.7x speedup alone).')
    parser.add_argument('--n-parallel', type=int, default=1,
                       help='Parallel partitions: 1=serial (default), 2=dual-partition split.')

    args = parser.parse_args()""",
"CLI: --enable-pruning + --n-parallel after Zeus S116 argparse block (R3)")

# ── P16: optimizer.optimize() — already has resume params on Zeus (SKIP expected)
# Zeus lines 479-487 show resume_study + study_name already in the call.
# Patch included for completeness; will SKIP if already applied.
patch("P16","window_optimizer_integration_final.py",
"""        results = optimizer.optimize(
            strategy=strategy,
            bounds=bounds,
            max_iterations=max_iterations,
            scorer=BidirectionalCountScorer(),
            seed_start=seed_start,
            seed_count=seed_count,
            resume_study=resume_study,
            study_name=study_name
        )""",
"""        results = optimizer.optimize(
            strategy=strategy,
            bounds=bounds,
            max_iterations=max_iterations,
            scorer=BidirectionalCountScorer(),
            seed_start=seed_start,
            seed_count=seed_count,
            resume_study=resume_study,   # S116-Bug5 confirmed
            study_name=study_name        # S116-Bug5 confirmed
        )""",
"optimizer.optimize(): resume params already present (S116-Bug5) — SKIP expected")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{B}{'='*65}{Z}")
ok=sum(1 for _,s in applied if s=="OK"); sk=sum(1 for _,s in applied if s=="SKIP")
for pid,s in applied:
    print(f"  {G if s=='OK' else Y}{'✅' if s=='OK' else '⏭ '}{Z}  {pid}  {s}")
for pid,desc,reason in failed:
    print(f"  {R}❌{Z}  {pid}  FAIL  →  {reason}")
print(f"\n  Applied: {ok}   Skipped: {sk}   Failed: {len(failed)}")
if failed:
    print(f"\n  {R}⚠️  {len(failed)} failure(s) — do NOT run on Zeus until resolved{Z}")
    sys.exit(1)
elif DRY_RUN:
    print(f"\n  {Y}Dry run complete — no files written.{Z}")
    print(f"  Remove --dry-run to apply.")
else:
    print(f"\n  {G}All patches applied. Run harness:{Z}")
    print(f"    python3 dry_run_s115.py   (require 12/12 PASS)")
    sys.exit(0)
