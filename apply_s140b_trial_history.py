#!/usr/bin/env python3
"""
S140b Step 1 Trial History & Warm-Start — Deploy Script
========================================================
Implements the step1_trial_history table and closes the feedback loop:
  Step 1 trials → DB → WATCHER reads best params → warm-start TPE
  Chapter 13 writes accuracy back → next relaunch uses proven params

Files patched (7):
  1. database_system.py
  2. window_optimizer_bayesian.py
  3. window_optimizer.py
  4. window_optimizer_integration_final.py
  5. agents/watcher_agent.py
  6. chapter_13_orchestrator.py
  7. agent_manifests/window_optimizer.json
  8. agent_grammars/chapter_13.gbnf
  9. strategy_advisor.gbnf

Usage (from ~/distributed_prng_analysis/):
  python3 apply_s140b_trial_history.py [--dry-run]

PREREQUISITE: apply_s140_seed_coverage_tracker.py must be applied first.
"""

import sys, os, json, shutil, subprocess
from pathlib import Path

DRY_RUN = '--dry-run' in sys.argv

def backup(path):
    bak = path + '.bak_s140b'
    if not DRY_RUN:
        shutil.copy2(path, bak)
    print(f"  BAK  {bak}")

def write(path, content):
    if not DRY_RUN:
        with open(path, 'w') as f:
            f.write(content)
    print(f"  {'DRY' if DRY_RUN else 'WRT'} {path}")

def check(condition, msg):
    if not condition:
        print(f"  ABORT: {msg}")
        sys.exit(1)

def syntax_check(path):
    r = subprocess.run(['python3', '-m', 'py_compile', path], capture_output=True)
    ok = r.returncode == 0
    print(f"  {'✅' if ok else '❌'} syntax: {path}")
    if not ok:
        print(f"     {r.stderr.decode()}")
    return ok


# ── PATCH 1: database_system.py ──────────────────────────────────────────────
def patch_database_system():
    print("\n[1/9] database_system.py — step1_trial_history table + 3 methods")
    path = 'database_system.py'
    backup(path)
    with open(path) as f:
        content = f.read()

    if 'step1_trial_history' in content:
        print("  SKIP already patched")
        return

    old_indexes = """            # Create indices for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_cache_lookup ON cache_results(prng_type, mapping_type, seed, samples, parameter_hash)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_job_status ON search_jobs(status, priority)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_progress_search ON exhaustive_progress(search_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_lottery_hash ON lottery_draws(number_hash)')"""

    new_indexes = """            conn.execute(\'\'\'
                CREATE TABLE IF NOT EXISTS step1_trial_history (
                    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id                  TEXT NOT NULL,
                    study_name              TEXT NOT NULL,
                    trial_number            INTEGER NOT NULL,
                    recorded_at             TEXT NOT NULL,
                    prng_type               TEXT NOT NULL,
                    seed_range_start        INTEGER NOT NULL,
                    seed_range_end          INTEGER NOT NULL,
                    window_size             INTEGER,
                    offset                  INTEGER,
                    skip_min                INTEGER,
                    skip_max                INTEGER,
                    session                 TEXT,
                    forward_threshold       REAL,
                    reverse_threshold       REAL,
                    trial_score             REAL,
                    forward_survivors       INTEGER,
                    reverse_survivors       INTEGER,
                    bidirectional_survivors INTEGER,
                    pruned                  INTEGER DEFAULT 0,
                    hit_at_20               REAL DEFAULT NULL,
                    hit_at_100              REAL DEFAULT NULL,
                    hit_at_300              REAL DEFAULT NULL,
                    downstream_score        REAL DEFAULT NULL,
                    downstream_recorded_at  TEXT DEFAULT NULL,
                    UNIQUE(run_id, trial_number)
                )
            \'\'\')

            # Create indices for performance
            conn.execute(\'CREATE INDEX IF NOT EXISTS idx_cache_lookup ON cache_results(prng_type, mapping_type, seed, samples, parameter_hash)\')
            conn.execute(\'CREATE INDEX IF NOT EXISTS idx_job_status ON search_jobs(status, priority)\')
            conn.execute(\'CREATE INDEX IF NOT EXISTS idx_progress_search ON exhaustive_progress(search_id)\')
            conn.execute(\'CREATE INDEX IF NOT EXISTS idx_lottery_hash ON lottery_draws(number_hash)\')
            conn.execute(\'CREATE INDEX IF NOT EXISTS idx_step1_prng_recorded ON step1_trial_history(prng_type, recorded_at)\')
            conn.execute(\'CREATE INDEX IF NOT EXISTS idx_step1_prng_scores ON step1_trial_history(prng_type, downstream_score, trial_score)\')"""

    check(old_indexes in content, "index anchor not found in database_system.py")
    content = content.replace(old_indexes, new_indexes)

    old_method = "    def store_lottery_draw(self, lottery_name: str, draw_date: str, draw_number: int,"
    new_methods = '''    def write_step1_trial(self, run_id, study_name, trial_number, prng_type,
                          seed_range_start, seed_range_end, params,
                          trial_score, forward_survivors, reverse_survivors,
                          bidirectional_survivors, pruned=False):
        """[S140b] Write one Step 1 Optuna trial to step1_trial_history. INSERT OR IGNORE."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(\'\'\'
                    INSERT OR IGNORE INTO step1_trial_history
                    (run_id,study_name,trial_number,recorded_at,prng_type,
                     seed_range_start,seed_range_end,window_size,offset,
                     skip_min,skip_max,session,forward_threshold,reverse_threshold,
                     trial_score,forward_survivors,reverse_survivors,
                     bidirectional_survivors,pruned)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                \'\'\', (run_id,study_name,trial_number,datetime.now().isoformat(),
                       prng_type,seed_range_start,seed_range_end,
                       params.get(\'window_size\'),params.get(\'offset\'),
                       params.get(\'skip_min\'),params.get(\'skip_max\'),
                       params.get(\'time_of_day\',params.get(\'session\')),
                       params.get(\'forward_threshold\'),params.get(\'reverse_threshold\'),
                       trial_score,forward_survivors,reverse_survivors,
                       bidirectional_survivors,1 if pruned else 0))
        except Exception as e:
            import logging; logging.getLogger(__name__).warning(f"[TRIAL_HISTORY] write_step1_trial failed: {e}")

    def write_downstream_score(self, run_id, hit_at_20, hit_at_100, hit_at_300, downstream_score):
        """[S140b] Write Chapter 13 accuracy back to step1_trial_history rows for run_id."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(\'\'\'
                    UPDATE step1_trial_history
                    SET hit_at_20=?,hit_at_100=?,hit_at_300=?,
                        downstream_score=?,downstream_recorded_at=?
                    WHERE run_id=?
                \'\'\', (hit_at_20,hit_at_100,hit_at_300,downstream_score,
                       datetime.now().isoformat(),run_id))
                import logging; logging.getLogger(__name__).info(
                    f"[TRIAL_HISTORY] downstream_score written run_id={run_id} rows={cursor.rowcount}")
                return cursor.rowcount
        except Exception as e:
            import logging; logging.getLogger(__name__).warning(f"[TRIAL_HISTORY] write_downstream_score failed: {e}")
            return 0

    def get_best_step1_params(self, prng_type, limit=5):
        """[S140b] Return best Step 1 params ordered by COALESCE(downstream_score,trial_score) DESC."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(\'\'\'
                    SELECT window_size,offset,skip_min,skip_max,session,
                           forward_threshold,reverse_threshold,trial_score,
                           downstream_score,run_id,seed_range_start,seed_range_end
                    FROM step1_trial_history
                    WHERE prng_type=? AND pruned=0 AND window_size IS NOT NULL
                    ORDER BY COALESCE(downstream_score,trial_score) DESC, recorded_at DESC
                    LIMIT ?
                \'\'\', (prng_type, limit))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            import logging; logging.getLogger(__name__).warning(f"[TRIAL_HISTORY] get_best_step1_params failed: {e}")
            return []

    def store_lottery_draw(self, lottery_name: str, draw_date: str, draw_number: int,'''

    check(old_method in content, "store_lottery_draw anchor not found")
    content = content.replace(old_method, new_methods)
    write(path, content)
    syntax_check(path)
    print("  OK")


# ── PATCH 2: window_optimizer_bayesian.py ────────────────────────────────────
def patch_bayesian():
    print("\n[2/9] window_optimizer_bayesian.py — trial history write + dynamic warm-start")
    path = 'window_optimizer_bayesian.py'
    backup(path)
    with open(path) as f:
        content = f.read()

    if '[S140b]' in content:
        print("  SKIP already patched")
        return

    # 2a — callback signature
    old_sig = ('def create_incremental_save_callback(\n'
               '    output_config_path: str = "optimal_window_config.json",\n'
               '    output_survivors_path: str = "bidirectional_survivors.json",\n'
               '    total_trials: int = 50\n'
               '):\n'
               '    """\n'
               '    Optuna callback that saves best-so-far results after each trial.\n'
               '    Ensures crash recovery and WATCHER visibility.\n'
               '    """')
    new_sig = ('def create_incremental_save_callback(\n'
               '    output_config_path: str = "optimal_window_config.json",\n'
               '    output_survivors_path: str = "bidirectional_survivors.json",\n'
               '    total_trials: int = 50,\n'
               '    trial_history_context: dict = None,  # [S140b]\n'
               '):\n'
               '    """\n'
               '    Optuna callback that saves best-so-far results after each trial.\n'
               '    Ensures crash recovery and WATCHER visibility.\n'
               '    [S140b] Writes per-trial data to step1_trial_history when context provided.\n'
               '    """')
    check(old_sig in content, "callback sig anchor not found")
    content = content.replace(old_sig, new_sig)

    # 2b — trial history write at end of save_best_so_far
    old_return = '            temp_path.rename(output_config_path)\n    \n    return save_best_so_far'
    new_return = ('            temp_path.rename(output_config_path)\n'
                  '\n'
                  '        # [S140b] per-trial history write\n'
                  '        if trial_history_context:\n'
                  '            try:\n'
                  '                from database_system import DistributedPRNGDatabase as _DBTH\n'
                  '                _db_th = _DBTH()\n'
                  '                _params = trial.params if trial.params else {}\n'
                  '                _score  = trial.value if trial.value is not None else 0.0\n'
                  '                _pruned = trial.state.name == \'PRUNED\'\n'
                  '                _surv   = trial.user_attrs.get(\'bidirectional_survivors\', [])\n'
                  '                _bidi   = len(_surv) if isinstance(_surv, list) else 0\n'
                  '                _db_th.write_step1_trial(\n'
                  '                    run_id=trial_history_context.get(\'run_id\',\'unknown\'),\n'
                  '                    study_name=trial_history_context.get(\'study_name\',\'unknown\'),\n'
                  '                    trial_number=trial.number,\n'
                  '                    prng_type=trial_history_context.get(\'prng_type\',\'java_lcg\'),\n'
                  '                    seed_range_start=trial_history_context.get(\'seed_start\',0),\n'
                  '                    seed_range_end=trial_history_context.get(\'seed_end\',0),\n'
                  '                    params=_params,\n'
                  '                    trial_score=_score,\n'
                  '                    forward_survivors=trial.user_attrs.get(\'forward_count\',0),\n'
                  '                    reverse_survivors=trial.user_attrs.get(\'reverse_count\',0),\n'
                  '                    bidirectional_survivors=_bidi,\n'
                  '                    pruned=_pruned)\n'
                  '            except Exception as _e_th:\n'
                  '                print(f"   [TRIAL_HISTORY] write failed (non-fatal): {_e_th}")\n'
                  '    \n'
                  '    return save_best_so_far')
    check(old_return in content, "return anchor not found")
    content = content.replace(old_return, new_return)

    # 2c — OptunaBayesianSearch.search() signature
    old_search = ('    def search(self, \n'
                  '               objective_function: Callable,\n'
                  '               bounds: \'SearchBounds\',\n'
                  '               max_iterations: int,\n'
                  '               scorer: ResultScorer,\n'
                  '               resume_study: bool = False,\n'
                  '               study_name: str = \'\',\n'
                  '               trse_context_file: str = \'trse_context.json\') -> Dict:  # S121')
    new_search = ('    def search(self, \n'
                  '               objective_function: Callable,\n'
                  '               bounds: \'SearchBounds\',\n'
                  '               max_iterations: int,\n'
                  '               scorer: ResultScorer,\n'
                  '               resume_study: bool = False,\n'
                  '               study_name: str = \'\',\n'
                  '               trse_context_file: str = \'trse_context.json\',\n'
                  '               trial_history_context: dict = None) -> Dict:  # S121, [S140b]')
    check(old_search in content, "search sig anchor not found")
    content = content.replace(old_search, new_search)

    # 2d — store context on self
    old_print = '        print(f"   📊 Optuna study: optuna_studies/{study_name}.db")'
    new_print = ('        print(f"   📊 Optuna study: optuna_studies/{study_name}.db")\n'
                 '        self._trial_history_context = trial_history_context  # [S140b]')
    check(old_print in content, "print anchor not found")
    content = content.replace(old_print, new_print)

    # 2e — replace callback creation with context-aware version
    old_cb_create = ('        # Run optimization with incremental save callback\n'
                     '        _incremental_callback = create_incremental_save_callback(\n'
                     '            output_config_path="optimal_window_config.json",\n'
                     '            output_survivors_path="bidirectional_survivors.json",\n'
                     '            total_trials=max_iterations\n'
                     '        )')
    new_cb_create = ('        # Run optimization with incremental save callback\n'
                     '        # [S140b] trial_history_context flows from optimize_window\n'
                     '        _th_context = self._trial_history_context if hasattr(self, \'_trial_history_context\') else None\n'
                     '        _incremental_callback = create_incremental_save_callback(\n'
                     '            output_config_path="optimal_window_config.json",\n'
                     '            output_survivors_path="bidirectional_survivors.json",\n'
                     '            total_trials=max_iterations,\n'
                     '            trial_history_context=_th_context\n'
                     '        )')
    check(old_cb_create in content, "callback creation anchor not found")
    content = content.replace(old_cb_create, new_cb_create)

    # 2f — dynamic warm-start
    old_ws = ('        if not _resume:\n'
              '            study.enqueue_trial({\n'
              "                'window_size': 8,\n"
              "                'offset': 43,\n"
              "                'skip_min': 5,\n"
              "                'skip_max': 56,\n"
              "                'forward_threshold': 0.49,\n"
              "                'reverse_threshold': 0.49\n"
              '            })\n'
              '            print("   🌡️  Warm-start: enqueued W8_O43_S5-56 as trial 0 (S112 known-good config)")\n'
              '        else:\n'
              '            print(f"   ✅ Resume mode: skipping warm-start (already in DB)")')
    new_ws = ('        if not _resume:\n'
              '            _ws_params = {\'window_size\':8,\'offset\':43,\'skip_min\':5,\n'
              '                          \'skip_max\':56,\'forward_threshold\':0.49,\'reverse_threshold\':0.49}\n'
              "            _ws_source = 'S112 hardcoded default'\n"
              '            if trial_history_context:\n'
              "                _ww=trial_history_context.get('warm_start_window')\n"
              "                _wo=trial_history_context.get('warm_start_offset')\n"
              "                _wsk=trial_history_context.get('warm_start_skip_min')\n"
              "                _wsx=trial_history_context.get('warm_start_skip_max')\n"
              "                _wf=trial_history_context.get('warm_start_fwd_thresh')\n"
              "                _wr=trial_history_context.get('warm_start_rev_thresh')\n"
              '                if all(v is not None for v in [_ww,_wo,_wsk,_wsx,_wf,_wr]):\n'
              '                    _ws_params={\'window_size\':int(_ww),\'offset\':int(_wo),\n'
              '                               \'skip_min\':int(_wsk),\'skip_max\':int(_wsx),\n'
              '                               \'forward_threshold\':float(_wf),\'reverse_threshold\':float(_wr)}\n'
              "                    _ws_source=f'step1_trial_history (W{_ww}_O{_wo})'\n"
              '            study.enqueue_trial(_ws_params)\n'
              '            print(f"   🌡️  Warm-start: enqueued {_ws_source} as trial 0")  # [S140b]\n'
              '        else:\n'
              '            print(f"   ✅ Resume mode: skipping warm-start (already in DB)")')
    check(old_ws in content, "warm-start anchor not found")
    content = content.replace(old_ws, new_ws)

    write(path, content)
    syntax_check(path)
    print("  OK")


# ── PATCH 3: window_optimizer.py ─────────────────────────────────────────────
def patch_window_optimizer():
    print("\n[3/9] window_optimizer.py — thread trial_history_context")
    path = 'window_optimizer.py'
    backup(path)
    with open(path) as f:
        content = f.read()

    if '[S140b]' in content:
        print("  SKIP already patched")
        return

    old_bao = ('    def search(self, objective_function, bounds, max_iterations, scorer,\n'
               "               resume_study: bool = False, study_name: str = '',\n"
               "               trse_context_file: str = 'trse_context.json'):\n"
               '        """Run Bayesian optimization"""\n'
               '        if self.optuna_search:\n'
               '            # Use real Optuna implementation\n'
               '            return self.optuna_search.search(objective_function, bounds, max_iterations, scorer,\n'
               '                                             resume_study=resume_study, study_name=study_name,\n'
               '                                             trse_context_file=trse_context_file)')
    new_bao = ('    def search(self, objective_function, bounds, max_iterations, scorer,\n'
               "               resume_study: bool = False, study_name: str = '',\n"
               "               trse_context_file: str = 'trse_context.json',\n"
               '               trial_history_context: dict = None):  # [S140b]\n'
               '        """Run Bayesian optimization"""\n'
               '        if self.optuna_search:\n'
               '            # Use real Optuna implementation\n'
               '            return self.optuna_search.search(objective_function, bounds, max_iterations, scorer,\n'
               '                                             resume_study=resume_study, study_name=study_name,\n'
               '                                             trse_context_file=trse_context_file,\n'
               '                                             trial_history_context=trial_history_context)')
    check(old_bao in content, "BayesianOptimization.search anchor not found")
    content = content.replace(old_bao, new_bao)

    old_opt = ('    def optimize(self, strategy: SearchStrategy, bounds: SearchBounds,\n'
               '                max_iterations: int = 50, scorer: ScoringFunction = None,\n'
               '                seed_start: int = 0, seed_count: int = 10_000_000,\n'
               "                resume_study: bool = False, study_name: str = '',\n"
               "                trse_context_file: str = 'trse_context.json') -> Dict[str, Any]:")
    new_opt = ('    def optimize(self, strategy: SearchStrategy, bounds: SearchBounds,\n'
               '                max_iterations: int = 50, scorer: ScoringFunction = None,\n'
               '                seed_start: int = 0, seed_count: int = 10_000_000,\n'
               "                resume_study: bool = False, study_name: str = '',\n"
               "                trse_context_file: str = 'trse_context.json',\n"
               '                trial_history_context: dict = None) -> Dict[str, Any]:  # [S140b]')
    check(old_opt in content, "WindowOptimizer.optimize anchor not found")
    content = content.replace(old_opt, new_opt)

    old_strat = ('        return strategy.search(objective, bounds, max_iterations, scorer, '
                 'resume_study=resume_study, study_name=study_name, trse_context_file=trse_context_file)')
    new_strat = ('        return strategy.search(objective, bounds, max_iterations, scorer,\n'
                 '                              resume_study=resume_study, study_name=study_name,\n'
                 '                              trse_context_file=trse_context_file,\n'
                 '                              trial_history_context=trial_history_context)  # [S140b]')
    check(old_strat in content, "strategy.search anchor not found")
    content = content.replace(old_strat, new_strat)

    write(path, content)
    syntax_check(path)
    print("  OK")


# ── PATCH 4: window_optimizer_integration_final.py ───────────────────────────
def patch_integration_final():
    print("\n[4/9] window_optimizer_integration_final.py — build context at source")
    path = 'window_optimizer_integration_final.py'
    backup(path)
    with open(path) as f:
        content = f.read()

    if '[S140b]' in content:
        print("  SKIP already patched")
        return

    old_opt_call = ('        results = optimizer.optimize(\n'
                    '            strategy=strategy,\n'
                    '            bounds=bounds,\n'
                    '            max_iterations=max_iterations,\n'
                    '            scorer=BidirectionalCountScorer(),\n'
                    '            seed_start=seed_start,\n'
                    '            seed_count=seed_count,\n'
                    '            resume_study=resume_study,   # S116-Bug5 confirmed\n'
                    '            study_name=study_name,       # S116-Bug5 confirmed\n'
                    '            trse_context_file=trse_context_file  # S123 TRSE thread\n'
                    '        )')
    new_opt_call = ('        # [S140b] trial history context — flows to Optuna callback\n'
                    '        _trial_history_ctx = {\n'
                    "            'run_id':     f\"step1_{prng_base}_{int(seed_start)}\",\n"
                    "            'study_name': study_name,\n"
                    "            'prng_type':  prng_base,\n"
                    "            'seed_start': seed_start,\n"
                    "            'seed_end':   seed_start + seed_count,\n"
                    '        }\n'
                    '\n'
                    '        results = optimizer.optimize(\n'
                    '            strategy=strategy,\n'
                    '            bounds=bounds,\n'
                    '            max_iterations=max_iterations,\n'
                    '            scorer=BidirectionalCountScorer(),\n'
                    '            seed_start=seed_start,\n'
                    '            seed_count=seed_count,\n'
                    '            resume_study=resume_study,              # S116-Bug5 confirmed\n'
                    '            study_name=study_name,                  # S116-Bug5 confirmed\n'
                    '            trse_context_file=trse_context_file,    # S123 TRSE thread\n'
                    '            trial_history_context=_trial_history_ctx  # [S140b]\n'
                    '        )')
    check(old_opt_call in content, "optimizer.optimize call anchor not found")
    content = content.replace(old_opt_call, new_opt_call)
    write(path, content)
    syntax_check(path)
    print("  OK")


# ── PATCH 5: agents/watcher_agent.py ─────────────────────────────────────────
def patch_watcher():
    print("\n[5/9] agents/watcher_agent.py — warm-start read at relaunch")
    path = 'agents/watcher_agent.py'
    backup(path)
    with open(path) as f:
        content = f.read()

    if 'WARM_START' in content:
        print("  SKIP already patched")
        return

    check('SEED COVERAGE TRACKER' in content,
          "S140 coverage block not found — apply apply_s140_seed_coverage_tracker.py first")

    old_block = ('                else:\n'
                 '                    logger.info(\n'
                 '                        f"[COVERAGE] Step 1: no prior coverage for "\n'
                 '                        f"{_prng_type} — using seed_start=0"\n'
                 '                    )\n'
                 '            except Exception as _e:\n'
                 '                logger.warning(\n'
                 '                    f"[COVERAGE] Seed coverage lookup failed: {_e} — using seed_start=0"\n'
                 '                )')
    new_block = ('                else:\n'
                 '                    logger.info(\n'
                 '                        f"[COVERAGE] Step 1: no prior coverage for "\n'
                 '                        f"{_prng_type} — using seed_start=0"\n'
                 '                    )\n'
                 '\n'
                 '                # [S140b] WARM-START read\n'
                 '                try:\n'
                 '                    _best_prior = _db.get_best_step1_params(_prng_type, limit=1)\n'
                 '                    if _best_prior:\n'
                 '                        _bp = _best_prior[0]\n'
                 "                        final_params['warm_start_window']     = _bp.get('window_size')\n"
                 "                        final_params['warm_start_offset']     = _bp.get('offset')\n"
                 "                        final_params['warm_start_skip_min']   = _bp.get('skip_min')\n"
                 "                        final_params['warm_start_skip_max']   = _bp.get('skip_max')\n"
                 "                        final_params['warm_start_session']    = _bp.get('session')\n"
                 "                        final_params['warm_start_fwd_thresh'] = _bp.get('forward_threshold')\n"
                 "                        final_params['warm_start_rev_thresh'] = _bp.get('reverse_threshold')\n"
                 "                        _src = 'downstream' if _bp.get('downstream_score') else 'trial_score'\n"
                 '                        logger.info(\n'
                 '                            f"[WARM_START] Step 1: W{_bp.get(\'window_size\')}_"\n'
                 '                            f"O{_bp.get(\'offset\')}_{_bp.get(\'session\')} "\n'
                 '                            f"(ordered by {_src})"\n'
                 '                        )\n'
                 '                    else:\n'
                 '                        logger.info(f"[WARM_START] no prior history for {_prng_type}")\n'
                 '                except Exception as _ews:\n'
                 '                    logger.warning(f"[WARM_START] lookup failed (non-fatal): {_ews}")\n'
                 '\n'
                 '            except Exception as _e:\n'
                 '                logger.warning(\n'
                 '                    f"[COVERAGE] Seed coverage lookup failed: {_e} — using seed_start=0"\n'
                 '                )')
    check(old_block in content, "S140 coverage block anchor not found in watcher_agent.py")
    content = content.replace(old_block, new_block)
    write(path, content)
    syntax_check(path)
    print("  OK")


# ── PATCH 6: chapter_13_orchestrator.py ──────────────────────────────────────
def patch_chapter13():
    print("\n[6/9] chapter_13_orchestrator.py — downstream score write-back")
    path = 'chapter_13_orchestrator.py'
    backup(path)
    with open(path) as f:
        content = f.read()

    if 'S140b' in content:
        print("  SKIP already patched")
        return

    old_anchor = ('            # Save diagnostics\n'
                  '            save_diagnostics(diagnostics)')
    new_anchor = ('            # Save diagnostics\n'
                  '            save_diagnostics(diagnostics)\n'
                  '\n'
                  '            # [S140b] DOWNSTREAM SCORE WRITE-BACK — annotation only\n'
                  '            try:\n'
                  '                from database_system import DistributedPRNGDatabase as _DBCH13\n'
                  '                _db_ch13 = _DBCH13()\n'
                  "                _owc_path = Path('optimal_window_config.json')\n"
                  '                _run_id_ch13 = None\n'
                  '                if _owc_path.exists():\n'
                  '                    import json as _jch13\n'
                  '                    _owc = _jch13.loads(_owc_path.read_text())\n'
                  "                    _ptype = _owc.get('prng_type','java_lcg')\n"
                  "                    _ss    = _owc.get('seed_start',_owc.get('seed_range_start',0))\n"
                  '                    _run_id_ch13 = f"step1_{_ptype}_{int(_ss)}"\n'
                  '                if _run_id_ch13:\n'
                  "                    _pv       = diagnostics.get('prediction_validation',{})\n"
                  "                    _best_rank = _pv.get('best_rank')\n"
                  '                    _hit_at_20  = 1.0 if _best_rank is not None and _best_rank <= 20  else 0.0\n'
                  '                    _hit_at_100 = 1.0 if _best_rank is not None and _best_rank <= 100 else 0.0\n'
                  '                    _hit_at_300 = 1.0 if _best_rank is not None and _best_rank <= 300 else 0.0\n'
                  "                    _cov = _pv.get('pool_coverage',0.0)\n"
                  '                    _ds  = round((_hit_at_20*0.5)+(_hit_at_100*0.3)+\n'
                  '                                 (_hit_at_300*0.15)+(_cov*0.05),4)\n'
                  '                    _rows = _db_ch13.write_downstream_score(\n'
                  '                        run_id=_run_id_ch13,\n'
                  '                        hit_at_20=_hit_at_20, hit_at_100=_hit_at_100,\n'
                  '                        hit_at_300=_hit_at_300, downstream_score=_ds)\n'
                  '                    logger.info(f"[S140b] downstream written run_id={_run_id_ch13} rows={_rows}")\n'
                  '            except Exception as _e140b:\n'
                  '                logger.warning(f"[S140b] downstream write-back failed (non-fatal): {_e140b}")')
    check(old_anchor in content, "save_diagnostics anchor not found")
    content = content.replace(old_anchor, new_anchor)
    write(path, content)
    syntax_check(path)
    print("  OK")


# ── PATCH 7: agent_manifests/window_optimizer.json ───────────────────────────
def patch_manifest():
    print("\n[7/9] agent_manifests/window_optimizer.json — warm_start params")
    path = 'agent_manifests/window_optimizer.json'
    backup(path)
    with open(path) as f:
        manifest = json.load(f)

    if 'warm_start_window' in manifest.get('default_params', {}):
        print("  SKIP already patched")
        return

    for k in ['warm_start_window','warm_start_offset','warm_start_skip_min',
              'warm_start_skip_max','warm_start_session',
              'warm_start_fwd_thresh','warm_start_rev_thresh']:
        manifest['default_params'][k] = None
        print(f"  ADD default_params.{k}")

    write(path, json.dumps(manifest, indent=2))
    with open(path) as f: json.load(f)
    print("  JSON valid — OK")


# ── PATCH 8: agent_grammars/chapter_13.gbnf ──────────────────────────────────
def patch_ch13_gbnf():
    print("\n[8/9] agent_grammars/chapter_13.gbnf — add step1_relaunch")
    path = 'agent_grammars/chapter_13.gbnf'
    backup(path)
    with open(path) as f:
        content = f.read()

    if 'step1_relaunch' in content:
        print("  SKIP already patched")
        return

    old = ('scope-enum ::= "\\"steps_3_5_6\\"" | "\\"steps_5_6\\"" | '
           '"\\"step_6_only\\"" | "\\"full_pipeline\\""')
    new = ('scope-enum ::= "\\"steps_3_5_6\\"" | "\\"steps_5_6\\"" | '
           '"\\"step_6_only\\"" | "\\"full_pipeline\\"" | "\\"step1_relaunch\\""')
    check(old in content, "scope-enum anchor not found in chapter_13.gbnf")
    content = content.replace(old, new)
    write(path, content)
    print("  OK")


# ── PATCH 9: strategy_advisor.gbnf ───────────────────────────────────────────
def patch_advisor_gbnf():
    print("\n[9/9] strategy_advisor.gbnf — add steps_0_1 and step_1_only")
    path = 'strategy_advisor.gbnf'
    backup(path)
    with open(path) as f:
        content = f.read()

    if 'steps_0_1' in content:
        print("  SKIP already patched")
        return

    old = ('# Retrain scope enum\n'
           'retrain-scope ::= "\\"selfplay_only\\"" |\n'
           '                  "\\"steps_5_6\\"" |\n'
           '                  "\\"steps_3_5_6\\"" |\n'
           '                  "\\"full_pipeline\\""')
    new = ('# Retrain scope enum\n'
           'retrain-scope ::= "\\"selfplay_only\\"" |\n'
           '                  "\\"steps_5_6\\"" |\n'
           '                  "\\"steps_3_5_6\\"" |\n'
           '                  "\\"full_pipeline\\"" |\n'
           '                  "\\"steps_0_1\\"" |\n'
           '                  "\\"step_1_only\\""')
    check(old in content, "retrain-scope anchor not found in strategy_advisor.gbnf")
    content = content.replace(old, new)
    write(path, content)
    print("  OK")


def verify_all():
    print("\n=== Final syntax verification ===")
    all_ok = True
    py_files = [
        'database_system.py',
        'window_optimizer_bayesian.py',
        'window_optimizer.py',
        'window_optimizer_integration_final.py',
        'agents/watcher_agent.py',
        'chapter_13_orchestrator.py',
    ]
    for f in py_files:
        ok = syntax_check(f)
        all_ok = all_ok and ok
    try:
        with open('agent_manifests/window_optimizer.json') as f:
            json.load(f)
        print("  ✅ window_optimizer.json")
    except Exception as e:
        print(f"  ❌ window_optimizer.json: {e}")
        all_ok = False
    return all_ok


if __name__ == '__main__':
    if DRY_RUN:
        print("DRY RUN — no files modified\n")

    if not Path('window_optimizer.py').exists():
        print("ERROR: Run from ~/distributed_prng_analysis/")
        sys.exit(1)

    check('S140' in open('agents/watcher_agent.py').read(),
          "S140 patch not found — apply apply_s140_seed_coverage_tracker.py first")

    patch_database_system()
    patch_bayesian()
    patch_window_optimizer()
    patch_integration_final()
    patch_watcher()
    patch_chapter13()
    patch_manifest()
    patch_ch13_gbnf()
    patch_advisor_gbnf()

    ok = verify_all()

    print("\n" + "="*55)
    if ok:
        print("✅ S140b Trial History & Warm-Start — all patches applied")
        print("\nNext steps:")
        print("  git add database_system.py window_optimizer_bayesian.py \\")
        print("          window_optimizer.py window_optimizer_integration_final.py \\")
        print("          agents/watcher_agent.py chapter_13_orchestrator.py \\")
        print("          agent_manifests/window_optimizer.json \\")
        print("          agent_grammars/chapter_13.gbnf strategy_advisor.gbnf")
        print("  git commit -m 'S140b: step1 trial history, warm-start, downstream feedback loop'")
        print("  git push origin main && git push public main")
    else:
        print("❌ One or more patches failed")
        sys.exit(1)
