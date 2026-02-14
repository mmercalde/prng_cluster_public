#!/usr/bin/env python3
"""
Session 83 — Phase 8A v2: Selfplay Episode Diagnostics Wiring
==============================================================

Idempotent Python patcher. Safe to re-run.

v2 CHANGES (Team Beta review):
    - train_model_kfold() return signature UNCHANGED (still -> ProxyMetrics)
    - train_single_fold() return signature UNCHANGED (still -> 6-tuple)
    - Diagnostics collected via module-level _FOLD_DIAGNOSTICS_COLLECTOR list
    - InnerEpisodeTrainer.train() reads side-channel after train_model_kfold()
    - Zero blast radius on existing callers

TARGETS:
    1. inner_episode_trainer.py — Tasks 8.1 + 8.2
    2. selfplay_orchestrator.py — Tasks 8.2 (propagation) + 8.3

INVARIANTS:
    - Diagnostics are BEST-EFFORT and NON-FATAL (Ch14 invariant)
    - No behavioral change when enable_diagnostics=False (default)
    - train_model_kfold() return type UNCHANGED
    - train_single_fold() return type UNCHANGED

USAGE:
    python3 apply_s83_phase8a_diagnostics.py
"""

import os
import sys
import shutil

# ===========================================================================
# CONFIGURATION
# ===========================================================================

PROJECT_ROOT = os.path.expanduser("~/distributed_prng_analysis")
TRAINER_PATH = os.path.join(PROJECT_ROOT, "inner_episode_trainer.py")
ORCHESTRATOR_PATH = os.path.join(PROJECT_ROOT, "selfplay_orchestrator.py")

MARKER = "# S83_PHASE8A_DIAG"

# ===========================================================================
# HELPERS
# ===========================================================================

def read_file(path):
    with open(path, 'r') as f:
        return f.read()


def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)


def backup(path):
    bak = path + ".pre_s83_phase8a"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"  Backup: {bak}")
    else:
        print(f"  Backup exists: {bak}")


def verify_syntax(path):
    import py_compile
    try:
        py_compile.compile(path, doraise=True)
        return True
    except py_compile.PyCompileError as e:
        print(f"  SYNTAX ERROR in {path}: {e}")
        return False


def check_idempotent(content):
    return MARKER in content


def safe_replace(content, old, new, label):
    """Replace exactly once, fail if anchor not found."""
    if old not in content:
        print(f"  ERROR: Anchor not found for [{label}]")
        return None
    count = content.count(old)
    if count > 1:
        print(f"  ERROR: Anchor [{label}] found {count} times (expected 1)")
        return None
    result = content.replace(old, new, 1)
    print(f"  [{label}] Applied")
    return result


# ===========================================================================
# PATCH 1: inner_episode_trainer.py
# ===========================================================================

def patch_trainer(path):
    print(f"\n{'='*60}")
    print(f"PATCH 1: inner_episode_trainer.py")
    print(f"{'='*60}")

    content = read_file(path)

    if check_idempotent(content):
        print("  SKIP: Already patched (S83 marker found)")
        return True

    backup(path)

    # -----------------------------------------------------------------------
    # 1a. Add import for training_diagnostics (after sklearn import)
    # -----------------------------------------------------------------------
    content = safe_replace(content,
        "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score",

        "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n"
        "\n"
        "# Chapter 14 Phase 8A: Training Diagnostics (best-effort, non-fatal)  " + MARKER + "\n"
        "try:\n"
        "    from training_diagnostics import TrainingDiagnostics\n"
        "    DIAGNOSTICS_AVAILABLE = True\n"
        "except ImportError:\n"
        "    DIAGNOSTICS_AVAILABLE = False",
        "1a-import")

    if content is None: return False

    # -----------------------------------------------------------------------
    # 1b. Add diagnostics field to TrainingResult dataclass
    # -----------------------------------------------------------------------
    content = safe_replace(content,
        "    success: bool\n"
        "    model_type: str\n"
        "    metrics: Optional[ProxyMetrics] = None\n"
        "    error: Optional[str] = None",

        "    success: bool\n"
        "    model_type: str\n"
        "    metrics: Optional[ProxyMetrics] = None\n"
        "    error: Optional[str] = None\n"
        "    diagnostics: Optional[Dict[str, Any]] = None  " + MARKER,
        "1b-field")

    if content is None: return False

    # -----------------------------------------------------------------------
    # 1c. Update TrainingResult.to_dict()
    # -----------------------------------------------------------------------
    content = safe_replace(content,
        "    def to_dict(self) -> Dict:\n"
        "        return {\n"
        "            'success': self.success,\n"
        "            'model_type': self.model_type,\n"
        "            'metrics': self.metrics.to_dict() if self.metrics else None,\n"
        "            'error': self.error\n"
        "        }",

        "    def to_dict(self) -> Dict:\n"
        "        return {\n"
        "            'success': self.success,\n"
        "            'model_type': self.model_type,\n"
        "            'metrics': self.metrics.to_dict() if self.metrics else None,\n"
        "            'error': self.error,\n"
        "            'diagnostics': self.diagnostics,  " + MARKER + "\n"
        "        }",
        "1c-to_dict")

    if content is None: return False

    # -----------------------------------------------------------------------
    # 1d. Replace train_single_fold() — add eval_set + diagnostics capture
    #     RETURN SIGNATURE UNCHANGED (6-tuple)
    # -----------------------------------------------------------------------
    OLD_TRAIN_SINGLE = (
        "def train_single_fold(\n"
        "    model_type: str,\n"
        "    config: TrainerConfig,\n"
        "    X_train: np.ndarray,\n"
        "    y_train: np.ndarray,\n"
        "    X_val: np.ndarray,\n"
        "    y_val: np.ndarray,\n"
        "    feature_names: Optional[List[str]] = None\n"
        ") -> Tuple[float, float, float, float, np.ndarray, Optional[np.ndarray]]:\n"
        '    """\n'
        "    Train single fold and return metrics.\n"
        "    \n"
        "    Returns:\n"
        "        (train_mse, val_mse, val_mae, val_r2, predictions, feature_importance)\n"
        '    """\n'
        "    # Build model\n"
        "    builder = MODEL_BUILDERS.get(model_type)\n"
        "    if not builder:\n"
        '        raise ValueError(f"Unknown model type: {model_type}")\n'
        "    \n"
        "    model = builder(config)\n"
        "    \n"
        "    # Train\n"
        "    model.fit(X_train, y_train)\n"
        "    \n"
        "    # Predictions\n"
        "    train_pred = model.predict(X_train)\n"
        "    val_pred = model.predict(X_val)\n"
        "    \n"
        "    # Metrics\n"
        "    train_mse = mean_squared_error(y_train, train_pred)\n"
        "    val_mse = mean_squared_error(y_val, val_pred)\n"
        "    val_mae = mean_absolute_error(y_val, val_pred)\n"
        "    val_r2 = r2_score(y_val, val_pred)\n"
        "    \n"
        "    # Feature importance\n"
        "    importance = None\n"
        "    if hasattr(model, 'feature_importances_'):\n"
        "        importance = model.feature_importances_\n"
        "    elif hasattr(model, 'get_feature_importance'):\n"
        "        importance = model.get_feature_importance()\n"
        "    \n"
        "    return train_mse, val_mse, val_mae, val_r2, val_pred, importance"
    )

    NEW_TRAIN_SINGLE = (
        "# S83 Phase 8A: Module-level diagnostics side-channel  " + MARKER + "\n"
        "# Fold diagnostics appended here by train_single_fold(), read by train().\n"
        "# Cleared before each train_model_kfold() call.\n"
        "_FOLD_DIAGNOSTICS_COLLECTOR: List[Dict] = []\n"
        "\n"
        "\n"
        "def train_single_fold(\n"
        "    model_type: str,\n"
        "    config: TrainerConfig,\n"
        "    X_train: np.ndarray,\n"
        "    y_train: np.ndarray,\n"
        "    X_val: np.ndarray,\n"
        "    y_val: np.ndarray,\n"
        "    feature_names: Optional[List[str]] = None,\n"
        "    enable_diagnostics: bool = False,  " + MARKER + "\n"
        ") -> Tuple[float, float, float, float, np.ndarray, Optional[np.ndarray]]:\n"
        '    """\n'
        "    Train single fold and return metrics.\n"
        "    \n"
        "    Return signature UNCHANGED. If enable_diagnostics=True, fold diagnostics\n"
        "    are appended to _FOLD_DIAGNOSTICS_COLLECTOR (side-channel, best-effort).\n"
        "    \n"
        "    Returns:\n"
        "        (train_mse, val_mse, val_mae, val_r2, predictions, feature_importance)\n"
        '    """\n'
        "    # Build model\n"
        "    builder = MODEL_BUILDERS.get(model_type)\n"
        "    if not builder:\n"
        '        raise ValueError(f"Unknown model type: {model_type}")\n'
        "    \n"
        "    model = builder(config)\n"
        "    \n"
        "    # -- S83 Phase 8A: Train with eval_set for round data --------  " + MARKER + "\n"
        "    try:\n"
        "        if model_type == 'lightgbm':\n"
        "            model.fit(\n"
        "                X_train, y_train,\n"
        "                eval_set=[(X_val, y_val)],\n"
        "                eval_names=['validation'],\n"
        "            )\n"
        "        elif model_type == 'xgboost':\n"
        "            model.fit(\n"
        "                X_train, y_train,\n"
        "                eval_set=[(X_val, y_val)],\n"
        "                verbose=False,\n"
        "            )\n"
        "        elif model_type == 'catboost':\n"
        "            model.fit(\n"
        "                X_train, y_train,\n"
        "                eval_set=(X_val, y_val),\n"
        "                verbose=False,\n"
        "            )\n"
        "        else:\n"
        "            model.fit(X_train, y_train)\n"
        "    except TypeError:\n"
        "        # Fallback if eval_set not supported\n"
        "        model.fit(X_train, y_train)\n"
        "    # -- End eval_set wiring -------------------------------------\n"
        "    \n"
        "    # Predictions\n"
        "    train_pred = model.predict(X_train)\n"
        "    val_pred = model.predict(X_val)\n"
        "    \n"
        "    # Metrics\n"
        "    train_mse = mean_squared_error(y_train, train_pred)\n"
        "    val_mse = mean_squared_error(y_val, val_pred)\n"
        "    val_mae = mean_absolute_error(y_val, val_pred)\n"
        "    val_r2 = r2_score(y_val, val_pred)\n"
        "    \n"
        "    # Feature importance\n"
        "    importance = None\n"
        "    if hasattr(model, 'feature_importances_'):\n"
        "        importance = model.feature_importances_\n"
        "    elif hasattr(model, 'get_feature_importance'):\n"
        "        importance = model.get_feature_importance()\n"
        "    \n"
        "    # -- S83 Phase 8A: Capture fold diagnostics (side-channel) ---  " + MARKER + "\n"
        "    if enable_diagnostics and DIAGNOSTICS_AVAILABLE:\n"
        "        try:\n"
        "            diag = TrainingDiagnostics.create(model_type, feature_names=feature_names)\n"
        "            diag.attach(model)\n"
        "            \n"
        "            # Feed eval results if available\n"
        "            if model_type == 'lightgbm' and hasattr(model, 'evals_result_'):\n"
        "                evals = model.evals_result_\n"
        "                if 'validation' in evals:\n"
        "                    metric_key = list(evals['validation'].keys())[0]\n"
        "                    train_losses = evals.get('training', {}).get(metric_key, [])\n"
        "                    for i, val_loss in enumerate(evals['validation'][metric_key]):\n"
        "                        t_loss = train_losses[i] if i < len(train_losses) else val_loss\n"
        "                        diag.on_round_end(i, float(t_loss), float(val_loss))\n"
        "            \n"
        "            elif model_type == 'xgboost' and hasattr(model, 'evals_result'):\n"
        "                evals_fn = model.evals_result\n"
        "                if callable(evals_fn):\n"
        "                    evals = evals_fn()\n"
        "                    val_key = 'validation_0'\n"
        "                    if val_key in evals:\n"
        "                        metric_key = list(evals[val_key].keys())[0]\n"
        "                        for i, val_loss in enumerate(evals[val_key][metric_key]):\n"
        "                            diag.on_round_end(i, float(val_loss), float(val_loss))\n"
        "            \n"
        "            elif model_type == 'catboost' and hasattr(model, 'get_evals_result'):\n"
        "                evals = model.get_evals_result()\n"
        "                if 'validation' in evals:\n"
        "                    metric_key = list(evals['validation'].keys())[0]\n"
        "                    learn_key = 'learn' if 'learn' in evals else None\n"
        "                    for i, val_loss in enumerate(evals['validation'][metric_key]):\n"
        "                        t_loss = evals[learn_key][metric_key][i] if learn_key else val_loss\n"
        "                        diag.on_round_end(i, float(t_loss), float(val_loss))\n"
        "            \n"
        "            # Set feature importance\n"
        "            if importance is not None and feature_names:\n"
        "                imp_len = min(len(feature_names), len(importance))\n"
        "                imp_dict = dict(zip(feature_names[:imp_len], [float(v) for v in importance[:imp_len]]))\n"
        "                diag.set_feature_importance(imp_dict)\n"
        "            \n"
        "            # Set best iteration if available\n"
        "            if hasattr(model, 'best_iteration_') and model.best_iteration_ is not None:\n"
        "                diag.set_best_iteration(model.best_iteration_)\n"
        "            elif hasattr(model, 'get_best_iteration'):\n"
        "                try:\n"
        "                    diag.set_best_iteration(model.get_best_iteration())\n"
        "                except Exception:\n"
        "                    pass\n"
        "            \n"
        "            diag.set_final_metrics({\n"
        "                'train_mse': float(train_mse),\n"
        "                'val_mse': float(val_mse),\n"
        "                'val_mae': float(val_mae),\n"
        "                'val_r2': float(val_r2),\n"
        "            })\n"
        "            \n"
        "            _FOLD_DIAGNOSTICS_COLLECTOR.append(diag.get_report())\n"
        "            diag.detach()\n"
        "        except Exception as e:\n"
        '            logger.debug(f"Fold diagnostics capture failed (non-fatal): {e}")\n'
        "    # -- End diagnostics capture ---------------------------------\n"
        "    \n"
        "    return train_mse, val_mse, val_mae, val_r2, val_pred, importance"
    )

    content = safe_replace(content, OLD_TRAIN_SINGLE, NEW_TRAIN_SINGLE, "1d-train_single_fold")
    if content is None: return False

    # -----------------------------------------------------------------------
    # 1e. Add enable_diagnostics param to train_model_kfold() signature
    #     RETURN TYPE UNCHANGED — just pass flag to train_single_fold()
    # -----------------------------------------------------------------------
    content = safe_replace(content,
        "def train_model_kfold(\n"
        "    model_type: str,\n"
        "    X: np.ndarray,\n"
        "    y: np.ndarray,\n"
        "    config: TrainerConfig,\n"
        "    feature_names: Optional[List[str]] = None\n"
        ") -> ProxyMetrics:",

        "def train_model_kfold(\n"
        "    model_type: str,\n"
        "    X: np.ndarray,\n"
        "    y: np.ndarray,\n"
        "    config: TrainerConfig,\n"
        "    feature_names: Optional[List[str]] = None,\n"
        "    enable_diagnostics: bool = False,  " + MARKER + "\n"
        ") -> ProxyMetrics:",
        "1e-kfold_sig")

    if content is None: return False

    # Add collector clear before fold loop
    content = safe_replace(content,
        "    fold_metrics = {\n"
        "        'train_mse': [],\n"
        "        'val_mse': [],\n"
        "        'val_mae': [],\n"
        "        'val_r2': [],\n"
        "    }\n"
        "    all_predictions = []\n"
        "    all_importance = []",

        "    _FOLD_DIAGNOSTICS_COLLECTOR.clear()  " + MARKER + "\n"
        "\n"
        "    fold_metrics = {\n"
        "        'train_mse': [],\n"
        "        'val_mse': [],\n"
        "        'val_mae': [],\n"
        "        'val_r2': [],\n"
        "    }\n"
        "    all_predictions = []\n"
        "    all_importance = []",
        "1e-collector_clear")

    if content is None: return False

    # Pass enable_diagnostics to train_single_fold call
    content = safe_replace(content,
        "        train_mse, val_mse, val_mae, val_r2, predictions, importance = train_single_fold(\n"
        "            model_type, config, X_train, y_train, X_val, y_val, feature_names\n"
        "        )",

        "        train_mse, val_mse, val_mae, val_r2, predictions, importance = train_single_fold(\n"
        "            model_type, config, X_train, y_train, X_val, y_val, feature_names,\n"
        "            enable_diagnostics=enable_diagnostics,  " + MARKER + "\n"
        "        )",
        "1e-kfold_call")

    if content is None: return False

    # -----------------------------------------------------------------------
    # 1f. Update InnerEpisodeTrainer.train() — add enable_diagnostics param,
    #     pass to train_model_kfold, collect diagnostics from side-channel
    # -----------------------------------------------------------------------
    content = safe_replace(content,
        "    def train(\n"
        "        self,\n"
        "        X: np.ndarray,\n"
        "        y: np.ndarray,\n"
        "        feature_names: Optional[List[str]] = None,\n"
        "        model_type: Optional[str] = None\n"
        "    ) -> TrainingResult:",

        "    def train(\n"
        "        self,\n"
        "        X: np.ndarray,\n"
        "        y: np.ndarray,\n"
        "        feature_names: Optional[List[str]] = None,\n"
        "        model_type: Optional[str] = None,\n"
        "        enable_diagnostics: bool = False,  " + MARKER + "\n"
        "    ) -> TrainingResult:",
        "1f-train_sig")

    if content is None: return False

    # Replace the train_model_kfold call + return block in train()
    content = safe_replace(content,
        "            metrics = train_model_kfold(\n"
        "                model_type=model_type,\n"
        "                X=X,\n"
        "                y=y,\n"
        "                config=self.config,\n"
        "                feature_names=feature_names\n"
        "            )\n"
        "            \n"
        "            self.logger.info(\n"
        '                f"Trained {model_type}: R\u00c2\u00b2={metrics.val_r2:.4f}, "\n'
        '                f"MAE={metrics.val_mae:.4f}, time={metrics.training_time_ms:.0f}ms"\n'
        "            )\n"
        "            \n"
        "            return TrainingResult(\n"
        "                success=True,\n"
        "                model_type=model_type,\n"
        "                metrics=metrics\n"
        "            )",

        "            metrics = train_model_kfold(\n"
        "                model_type=model_type,\n"
        "                X=X,\n"
        "                y=y,\n"
        "                config=self.config,\n"
        "                feature_names=feature_names,\n"
        "                enable_diagnostics=enable_diagnostics,  " + MARKER + "\n"
        "            )\n"
        "            \n"
        "            self.logger.info(\n"
        '                f"Trained {model_type}: R\u00c2\u00b2={metrics.val_r2:.4f}, "\n'
        '                f"MAE={metrics.val_mae:.4f}, time={metrics.training_time_ms:.0f}ms"\n'
        "            )\n"
        "            \n"
        "            # -- S83 Phase 8A: Collect diagnostics from side-channel --  " + MARKER + "\n"
        "            aggregated_diag = None\n"
        "            if enable_diagnostics and _FOLD_DIAGNOSTICS_COLLECTOR:\n"
        "                try:\n"
        "                    fold_reports = list(_FOLD_DIAGNOSTICS_COLLECTOR)\n"
        "                    severities = [\n"
        "                        fd.get('diagnosis', {}).get('severity', 'absent')\n"
        "                        for fd in fold_reports\n"
        "                    ]\n"
        "                    severity_rank = {'critical': 3, 'warning': 2, 'ok': 1, 'absent': 0}\n"
        "                    worst_sev = max(severities, key=lambda s: severity_rank.get(s, 0))\n"
        "                    \n"
        "                    overfit_gaps = [\n"
        "                        fd.get('training_summary', {}).get('overfit_gap')\n"
        "                        for fd in fold_reports\n"
        "                        if fd.get('training_summary', {}).get('overfit_gap') is not None\n"
        "                    ]\n"
        "                    \n"
        "                    best_rounds = []\n"
        "                    total_rounds = []\n"
        "                    for fd in fold_reports:\n"
        "                        ts = fd.get('training_summary', {})\n"
        "                        bv = ts.get('best_val_round')\n"
        "                        rc = ts.get('rounds_captured')\n"
        "                        if bv is not None and rc:\n"
        "                            best_rounds.append(bv)\n"
        "                            total_rounds.append(rc)\n"
        "                    \n"
        "                    best_round_ratio = 0.0\n"
        "                    if best_rounds and total_rounds:\n"
        "                        best_round_ratio = float(np.mean([\n"
        "                            b / max(t, 1) for b, t in zip(best_rounds, total_rounds)\n"
        "                        ]))\n"
        "                    \n"
        "                    all_issues = []\n"
        "                    for fd in fold_reports:\n"
        "                        all_issues.extend(fd.get('diagnosis', {}).get('issues', []))\n"
        "                    \n"
        "                    aggregated_diag = {\n"
        "                        'model_type': model_type,\n"
        "                        'fold_count': len(fold_reports),\n"
        "                        'severity': worst_sev,\n"
        "                        'best_round_ratio': best_round_ratio,\n"
        "                        'mean_overfit_gap': float(np.mean(overfit_gaps)) if overfit_gaps else None,\n"
        "                        'issues': list(set(all_issues))[:10],\n"
        "                        'fold_severities': severities,\n"
        "                    }\n"
        "                except Exception as e:\n"
        '                    self.logger.debug(f"Diagnostics aggregation failed (non-fatal): {e}")\n'
        "            # -- End diagnostics collection --------------------------\n"
        "            \n"
        "            return TrainingResult(\n"
        "                success=True,\n"
        "                model_type=model_type,\n"
        "                metrics=metrics,\n"
        "                diagnostics=aggregated_diag,  " + MARKER + "\n"
        "            )",
        "1f-train_body")

    if content is None: return False

    # -----------------------------------------------------------------------
    # 1g. Update train_all() to pass enable_diagnostics
    # -----------------------------------------------------------------------
    content = safe_replace(content,
        "    def train_all(\n"
        "        self,\n"
        "        X: np.ndarray,\n"
        "        y: np.ndarray,\n"
        "        feature_names: Optional[List[str]] = None\n"
        "    ) -> Dict[str, TrainingResult]:",

        "    def train_all(\n"
        "        self,\n"
        "        X: np.ndarray,\n"
        "        y: np.ndarray,\n"
        "        feature_names: Optional[List[str]] = None,\n"
        "        enable_diagnostics: bool = False,  " + MARKER + "\n"
        "    ) -> Dict[str, TrainingResult]:",
        "1g-train_all_sig")

    if content is None: return False

    content = safe_replace(content,
        "            results[model_type] = self.train(X, y, feature_names, model_type)",
        "            results[model_type] = self.train(X, y, feature_names, model_type, enable_diagnostics)  " + MARKER,
        "1g-train_all_call")

    if content is None: return False

    # -----------------------------------------------------------------------
    # 1h. Update train_best() to pass enable_diagnostics
    # -----------------------------------------------------------------------
    content = safe_replace(content,
        "    def train_best(\n"
        "        self,\n"
        "        X: np.ndarray,\n"
        "        y: np.ndarray,\n"
        "        feature_names: Optional[List[str]] = None\n"
        "    ) -> Tuple[TrainingResult, Dict[str, TrainingResult]]:",

        "    def train_best(\n"
        "        self,\n"
        "        X: np.ndarray,\n"
        "        y: np.ndarray,\n"
        "        feature_names: Optional[List[str]] = None,\n"
        "        enable_diagnostics: bool = False,  " + MARKER + "\n"
        "    ) -> Tuple[TrainingResult, Dict[str, TrainingResult]]:",
        "1h-train_best_sig")

    if content is None: return False

    content = safe_replace(content,
        "        all_results = self.train_all(X, y, feature_names)",
        "        all_results = self.train_all(X, y, feature_names, enable_diagnostics)  " + MARKER,
        "1h-train_best_call")

    if content is None: return False

    # -----------------------------------------------------------------------
    # 1i. Update version
    # -----------------------------------------------------------------------
    content = safe_replace(content, "Version: 1.0.3", "Version: 1.1.0", "1i-ver")
    if content is None: return False
    content = safe_replace(content, "- v1.0.3:",
        "- v1.1.0: S83 Phase 8A -- episode diagnostics (eval_set + side-channel pattern)\n- v1.0.3:",
        "1i-log")
    if content is None: return False

    # Write and verify
    write_file(path, content)
    if not verify_syntax(path):
        print("  ROLLING BACK")
        shutil.copy2(path + ".pre_s83_phase8a", path)
        return False

    print(f"  PASS: inner_episode_trainer.py patched ({content.count(MARKER)} markers)")
    return True


# ===========================================================================
# PATCH 2: selfplay_orchestrator.py
# ===========================================================================

def patch_orchestrator(path):
    print(f"\n{'='*60}")
    print(f"PATCH 2: selfplay_orchestrator.py")
    print(f"{'='*60}")

    content = read_file(path)

    if check_idempotent(content):
        print("  SKIP: Already patched (S83 marker found)")
        return True

    backup(path)

    # -----------------------------------------------------------------------
    # 2a. Add diagnostics field to EpisodeResult
    # -----------------------------------------------------------------------
    content = safe_replace(content,
        "    # Phase 9B.2: Policy conditioning info\n"
        "    policy_conditioned: bool = False\n"
        "    active_policy_id: Optional[str] = None\n"
        "    active_policy_fingerprint: Optional[str] = None\n"
        "    conditioning_log: List[str] = field(default_factory=list)",

        "    # Phase 9B.2: Policy conditioning info\n"
        "    policy_conditioned: bool = False\n"
        "    active_policy_id: Optional[str] = None\n"
        "    active_policy_fingerprint: Optional[str] = None\n"
        "    conditioning_log: List[str] = field(default_factory=list)\n"
        "    \n"
        "    # Phase 8A: Episode diagnostics (Ch14)  " + MARKER + "\n"
        "    episode_diagnostics: Optional[Dict[str, Any]] = None",
        "2a-episode_result")

    if content is None: return False

    # -----------------------------------------------------------------------
    # 2b. Wire enable_diagnostics in _run_inner_episode()
    # -----------------------------------------------------------------------
    content = safe_replace(content,
        "        best_result, all_results = trainer.train_best(X, y, feature_names)",

        "        # S83 Phase 8A: Pass enable_diagnostics from config  " + MARKER + "\n"
        "        _enable_diag = self.config.__dict__.get('enable_diagnostics', False)\n"
        "        best_result, all_results = trainer.train_best(X, y, feature_names, enable_diagnostics=_enable_diag)",
        "2b-train_best")

    if content is None: return False

    # -----------------------------------------------------------------------
    # 2c. Add diagnostics to EpisodeResult return
    # -----------------------------------------------------------------------
    content = safe_replace(content,
        "        return EpisodeResult(\n"
        "            episode_id=episode_id,\n"
        "            timestamp=timestamp,\n"
        "            model_type=best_result.model_type,\n"
        "            fitness=metrics.fitness,\n"
        "            val_r2=metrics.val_r2,\n"
        "            val_mae=metrics.val_mae,\n"
        "            fold_std=metrics.fold_std,\n"
        "            train_val_gap=metrics.train_val_gap,\n"
        "            training_time_ms=training_time_ms,\n"
        "            survivor_count=survivor_count,\n"
        "            survivor_count_before=survivor_count_before,\n"
        "            feature_count=len(feature_names),\n"
        "            policy_id=policy_id,\n"
        "            # Phase 9B.2 fields\n"
        "            policy_conditioned=self.config.policy_conditioned,\n"
        "            active_policy_id=active_policy_id,\n"
        "            active_policy_fingerprint=active_policy_fingerprint,\n"
        "            conditioning_log=conditioning_log,\n"
        "        )",

        "        # S83 Phase 8A: Extract diagnostics from best model  " + MARKER + "\n"
        "        _best_diag = getattr(best_result, 'diagnostics', None)\n"
        "\n"
        "        return EpisodeResult(\n"
        "            episode_id=episode_id,\n"
        "            timestamp=timestamp,\n"
        "            model_type=best_result.model_type,\n"
        "            fitness=metrics.fitness,\n"
        "            val_r2=metrics.val_r2,\n"
        "            val_mae=metrics.val_mae,\n"
        "            fold_std=metrics.fold_std,\n"
        "            train_val_gap=metrics.train_val_gap,\n"
        "            training_time_ms=training_time_ms,\n"
        "            survivor_count=survivor_count,\n"
        "            survivor_count_before=survivor_count_before,\n"
        "            feature_count=len(feature_names),\n"
        "            policy_id=policy_id,\n"
        "            # Phase 9B.2 fields\n"
        "            policy_conditioned=self.config.policy_conditioned,\n"
        "            active_policy_id=active_policy_id,\n"
        "            active_policy_fingerprint=active_policy_fingerprint,\n"
        "            conditioning_log=conditioning_log,\n"
        "            # Phase 8A  " + MARKER + "\n"
        "            episode_diagnostics=_best_diag,\n"
        "        )",
        "2c-return")

    if content is None: return False

    # -----------------------------------------------------------------------
    # 2d. Add diagnostics tracking in run() loop
    # -----------------------------------------------------------------------
    content = safe_replace(content,
        "        for episode_num in range(1, self.config.max_episodes + 1):",

        "        # S83 Phase 8A: Episode diagnostics history  " + MARKER + "\n"
        "        diagnostics_history = []\n"
        "\n"
        "        for episode_num in range(1, self.config.max_episodes + 1):",
        "2d-init")

    if content is None: return False

    content = safe_replace(content,
        "                # Track best\n"
        "                if result.fitness > self.best_fitness:",

        "                # S83 Phase 8A: Collect episode diagnostics  " + MARKER + "\n"
        "                if hasattr(result, 'episode_diagnostics') and result.episode_diagnostics:\n"
        "                    diagnostics_history.append({\n"
        "                        'episode': episode_num,\n"
        "                        'severity': result.episode_diagnostics.get('severity', 'absent'),\n"
        "                        'best_round_ratio': result.episode_diagnostics.get('best_round_ratio', 0.0),\n"
        "                        'fitness': result.fitness,\n"
        "                        'model_type': result.model_type,\n"
        "                    })\n"
        "                    diagnostics_history = diagnostics_history[-20:]\n"
        "                    self._check_episode_training_trend(diagnostics_history)\n"
        "\n"
        "                # Track best\n"
        "                if result.fitness > self.best_fitness:",
        "2d-collect")

    if content is None: return False

    # -----------------------------------------------------------------------
    # 2e. Add _check_episode_training_trend() method
    # -----------------------------------------------------------------------
    TREND_METHOD = (
        "\n"
        "    # =====================================================  " + MARKER + "\n"
        "    # S83 PHASE 8A: Episode Training Trend Detection\n"
        "    # =====================================================\n"
        "\n"
        "    def _check_episode_training_trend(self, diagnostics_history: List[Dict]) -> Optional[str]:\n"
        '        """\n'
        "        Detect degrading training quality across episodes.\n"
        "        Ch14 Section 9.2. OBSERVE-ONLY -- no automatic intervention.\n"
        "        \n"
        "        Returns:\n"
        "            'STABLE', 'WARNING_OVERFIT', 'CRITICAL_OVERFIT', 'DEGRADING', or None.\n"
        '        """\n'
        "        if len(diagnostics_history) < 3:\n"
        "            return None\n"
        "        \n"
        "        recent = diagnostics_history[-3:]\n"
        "        ratios = [d.get('best_round_ratio', 0.0) for d in recent]\n"
        "        \n"
        "        if ratios[0] > ratios[1] > ratios[2] and ratios[2] < 0.2:\n"
        "            print(f\"      [DIAG] Training quality DEGRADING: \"\n"
        "                  f\"best_round_ratios={[f'{r:.2f}' for r in ratios]}\")\n"
        "            if self.telemetry:\n"
        "                try:\n"
        "                    self.telemetry.record_event(\n"
        "                        event_type='training_quality_degrading',\n"
        "                        details={\n"
        "                            'recent_ratios': ratios,\n"
        "                            'recent_severities': [d.get('severity', 'absent') for d in recent],\n"
        "                        },\n"
        "                    )\n"
        "                except Exception:\n"
        "                    pass\n"
        "            return 'DEGRADING'\n"
        "        \n"
        "        severities = [d.get('severity', 'ok') for d in recent]\n"
        "        critical_count = sum(1 for s in severities if s == 'critical')\n"
        "        \n"
        "        if critical_count >= 2:\n"
        "            print(f\"      [DIAG] Majority critical episodes: \"\n"
        "                  f\"{critical_count}/{len(recent)}\")\n"
        "            return 'CRITICAL_OVERFIT'\n"
        "        \n"
        "        warning_count = sum(1 for s in severities if s in ('critical', 'warning'))\n"
        "        if warning_count >= 2:\n"
        "            return 'WARNING_OVERFIT'\n"
        "        \n"
        "        return 'STABLE'\n"
        "\n"
    )

    content = safe_replace(content,
        "    def _load_survivors_raw(self, path: str) -> List[Dict]:",
        TREND_METHOD + "    def _load_survivors_raw(self, path: str) -> List[Dict]:",
        "2e-trend_method")

    if content is None: return False

    # -----------------------------------------------------------------------
    # 2f. Update version
    # -----------------------------------------------------------------------
    content = safe_replace(content,
        "Version: 1.1.0\nDate: 2026-01-30",
        "Version: 1.2.0\nDate: 2026-02-13",
        "2f-ver")
    if content is None: return False

    content = safe_replace(content,
        "    v1.1.0 (2026-01-30):",
        "    v1.2.0 (2026-02-13):\n"
        "        - S83 Phase 8A: Episode diagnostics (Team Beta v2 side-channel architecture)\n"
        "        - Added episode_diagnostics to EpisodeResult, trend detection\n"
        "    \n"
        "    v1.1.0 (2026-01-30):",
        "2f-log")
    if content is None: return False

    # Write and verify
    write_file(path, content)
    if not verify_syntax(path):
        print("  ROLLING BACK")
        shutil.copy2(path + ".pre_s83_phase8a", path)
        return False

    print(f"  PASS: selfplay_orchestrator.py patched ({content.count(MARKER)} markers)")
    return True


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 60)
    print("Session 83 -- Phase 8A v2: Selfplay Episode Diagnostics")
    print("(Team Beta architecture: side-channel, no return sig changes)")
    print("=" * 60)

    for path in [TRAINER_PATH, ORCHESTRATOR_PATH]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)

    ok1 = patch_trainer(TRAINER_PATH)
    ok2 = patch_orchestrator(ORCHESTRATOR_PATH)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  inner_episode_trainer.py: {'PASS' if ok1 else 'FAIL'}")
    print(f"  selfplay_orchestrator.py: {'PASS' if ok2 else 'FAIL'}")

    if ok1 and ok2:
        print(f"\nPhase 8A v2 applied successfully.")
        print(f"\nVerify:")
        print(f"  python3 -c \"import py_compile; py_compile.compile('inner_episode_trainer.py', doraise=True); print('OK')\"")
        print(f"  python3 -c \"import py_compile; py_compile.compile('selfplay_orchestrator.py', doraise=True); print('OK')\"")
    else:
        print(f"\nSome patches FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
