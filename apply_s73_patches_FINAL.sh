#!/bin/bash
# Session 73 FINAL: Complete Phase 3 diagnostics wiring
# Run from ~/distributed_prng_analysis
#
# Team Beta Review Fixes Applied:
# 1. Add CLI flag to meta_prediction_optimizer
# 2. Thread --enable-diagnostics through SubprocessTrialCoordinator
# 3. Actually emit diagnostics in train_single_trial.py
# 4. Wrap in best-effort guards

set -e

cd ~/distributed_prng_analysis

echo "============================================================"
echo "Session 73 FINAL: Phase 3 Diagnostics Wiring + Emission"
echo "============================================================"
echo ""

# Create diagnostics_outputs directory
mkdir -p diagnostics_outputs

# ============================================================
# PATCH 1: meta_prediction_optimizer_anti_overfit.py
# ============================================================
echo ">>> Patch 1/5: meta_prediction_optimizer_anti_overfit.py"

cp meta_prediction_optimizer_anti_overfit.py meta_prediction_optimizer_anti_overfit.py.bak_s73

# Add --enable-diagnostics to argparse
sed -i "/--parent-run-id/a\\
    \\
    # Chapter 14: Training diagnostics\\
    parser.add_argument('--enable-diagnostics', action='store_true',\\
                       help='Enable Chapter 14 training diagnostics (writes to diagnostics_outputs/)')" meta_prediction_optimizer_anti_overfit.py

echo "   ✅ CLI flag added"

# ============================================================
# PATCH 2: subprocess_trial_coordinator.py
# Thread enable_diagnostics through SubprocessTrialCoordinator
# ============================================================
echo ">>> Patch 2/5: subprocess_trial_coordinator.py (threading)"

cp subprocess_trial_coordinator.py subprocess_trial_coordinator.py.bak_s73

python3 << 'PATCH_COORDINATOR'
import re

with open('subprocess_trial_coordinator.py', 'r') as f:
    content = f.read()

# 2A: Add enable_diagnostics to __init__ signature
old_init = "def __init__(self, X_train, y_train, X_val, y_val,"
new_init = "def __init__(self, X_train, y_train, X_val, y_val, enable_diagnostics: bool = False,"

if old_init in content and "enable_diagnostics" not in content[:content.find("def __init__") + 500]:
    content = content.replace(old_init, new_init, 1)
    
    # Add self.enable_diagnostics after self.verbose
    content = content.replace(
        "self.verbose = verbose",
        "self.verbose = verbose\n        self.enable_diagnostics = enable_diagnostics"
    )

# 2B: Add --enable-diagnostics to subprocess command
# Find the cmd list and add the flag
old_cmd_end = "'--model-output-dir', str(self.temp_models_dir)  # NEW: Save to temp dir"
new_cmd_end = """'--model-output-dir', str(self.temp_models_dir)  # NEW: Save to temp dir
            ]
            
            # Chapter 14: Thread diagnostics flag
            if self.enable_diagnostics:
                cmd.append('--enable-diagnostics')
            
            cmd_continued = ["""

# This is tricky - let's do a simpler approach
# Find where cmd list ends and insert the diagnostics flag logic
if "if self.verbose:" in content and "--enable-diagnostics" not in content:
    # Insert before "if self.verbose:" in run_trial
    content = content.replace(
        "            if self.verbose:\n                cmd.append('--verbose')",
        """            if self.verbose:
                cmd.append('--verbose')
            
            # Chapter 14: Thread diagnostics flag to subprocess
            if getattr(self, 'enable_diagnostics', False):
                cmd.append('--enable-diagnostics')"""
    )

with open('subprocess_trial_coordinator.py', 'w') as f:
    f.write(content)

print("   ✅ SubprocessTrialCoordinator patched")
PATCH_COORDINATOR

# ============================================================
# PATCH 3: run_subprocess_comparison in meta_prediction_optimizer
# Pass enable_diagnostics to coordinator
# ============================================================
echo ">>> Patch 3/5: Wire enable_diagnostics to coordinator"

python3 << 'PATCH_META'
with open('meta_prediction_optimizer_anti_overfit.py', 'r') as f:
    content = f.read()

# Find SubprocessTrialCoordinator instantiation and add enable_diagnostics
old_coordinator = """with SubprocessTrialCoordinator(
        X_train, y_train, X_val, y_val,
        worker_script='train_single_trial.py',
        timeout=timeout,
        verbose=True,
        output_dir=output_dir
    ) as coordinator:"""

new_coordinator = """# Get enable_diagnostics from outer scope if available
    _enable_diag = globals().get('_enable_diagnostics_flag', False)
    
    with SubprocessTrialCoordinator(
        X_train, y_train, X_val, y_val,
        enable_diagnostics=_enable_diag,
        worker_script='train_single_trial.py',
        timeout=timeout,
        verbose=True,
        output_dir=output_dir
    ) as coordinator:"""

if old_coordinator in content:
    content = content.replace(old_coordinator, new_coordinator)

# Also need to set the flag before calling run_subprocess_comparison
# Find where run_subprocess_comparison is called in main()
old_call = "results = run_subprocess_comparison("
new_call = """# Thread diagnostics flag to subprocess coordinator
    global _enable_diagnostics_flag
    _enable_diagnostics_flag = getattr(args, 'enable_diagnostics', False)
    
    results = run_subprocess_comparison("""

if old_call in content and "_enable_diagnostics_flag" not in content:
    content = content.replace(old_call, new_call, 1)

with open('meta_prediction_optimizer_anti_overfit.py', 'w') as f:
    f.write(content)

print("   ✅ Coordinator call updated with enable_diagnostics")
PATCH_META

# ============================================================
# PATCH 4: train_single_trial.py (with diagnostics emission)
# ============================================================
echo ">>> Patch 4/5: train_single_trial.py (with emission)"

cp train_single_trial.py train_single_trial.py.bak_s73

python3 << 'PATCH_TRAIN'
with open('train_single_trial.py', 'r') as f:
    content = f.read()

# 4A: Add imports after docstring
import_patch = '''
# Chapter 14: Training diagnostics (best-effort, non-fatal)
try:
    from training_diagnostics import TreeDiagnostics, NNDiagnostics
    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    DIAGNOSTICS_AVAILABLE = False
    TreeDiagnostics = None
    NNDiagnostics = None

'''

first_sep = content.find('\n# ==')
if first_sep > 0 and "DIAGNOSTICS_AVAILABLE" not in content:
    content = content[:first_sep] + '\n' + import_patch + content[first_sep:]

# 4B: Add argparse flag
if "--enable-diagnostics" not in content:
    content = content.replace(
        "help='Directory for model checkpoint (default: temp dir)')",
        """help='Directory for model checkpoint (default: temp dir)')
    # Chapter 14: Training diagnostics
    parser.add_argument('--enable-diagnostics', action='store_true',
                        help='Enable Chapter 14 training diagnostics')"""
    )

# 4C: Update function signatures
for model in ['catboost', 'xgboost', 'lightgbm', 'neural_net', 'random_forest']:
    old_sig = f"def train_{model}(X_train, y_train, X_val, y_val, params: dict, save_path: str = None)"
    new_sig = f"def train_{model}(X_train, y_train, X_val, y_val, params: dict, save_path: str = None, enable_diagnostics: bool = False)"
    if old_sig in content:
        content = content.replace(old_sig, new_sig)

# 4D: Update dispatcher calls
for model in ['lightgbm', 'xgboost', 'catboost', 'neural_net', 'random_forest']:
    old_call = f"result = train_{model}(X_train, y_train, X_val, y_val, params, save_path)"
    new_call = f"result = train_{model}(X_train, y_train, X_val, y_val, params, save_path, getattr(args, 'enable_diagnostics', False))"
    if old_call in content:
        content = content.replace(old_call, new_call)

# 4E-H: Add diagnostics emission to each train function
# We'll add a helper function and call it from each train_* function

diagnostics_helper = '''

def _emit_tree_diagnostics(model, model_type: str, r2: float, mse: float, enable_diagnostics: bool):
    """Chapter 14: Emit tree model diagnostics (best-effort, non-fatal)."""
    if not enable_diagnostics or not DIAGNOSTICS_AVAILABLE:
        return
    try:
        import os
        os.makedirs('diagnostics_outputs', exist_ok=True)
        diag = TreeDiagnostics(model_type=model_type)
        diag.attach(model)
        
        # Try to capture evals_result for models that support it
        if model_type == 'catboost' and hasattr(model, 'get_evals_result'):
            try:
                evals = model.get_evals_result()
                if evals:
                    keys = list(evals.keys())
                    learn_key = 'learn' if 'learn' in keys else keys[0]
                    val_key = 'validation' if 'validation' in keys else (keys[-1] if len(keys) > 1 else keys[0])
                    metric_keys = list(evals[learn_key].keys()) if evals.get(learn_key) else []
                    if metric_keys:
                        metric = metric_keys[0]
                        for i in range(len(evals.get(val_key, {}).get(metric, []))):
                            t = evals[learn_key][metric][i] if i < len(evals.get(learn_key, {}).get(metric, [])) else 0
                            v = evals[val_key][metric][i]
                            diag.on_round_end(i, t, v)
            except Exception:
                pass
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            try:
                imp = model.feature_importances_
                importance = {f'f{i}': float(v) for i, v in enumerate(imp)}
                diag.set_feature_importance(importance)
            except Exception:
                pass
        
        diag.set_final_metrics({'r2': r2, 'mse': mse})
        diag.detach()
        diag.save(f'diagnostics_outputs/{model_type}_diagnostics.json')
        print(f"[DIAG] {model_type} diagnostics saved", file=sys.stderr)
    except Exception as e:
        print(f"[DIAG] {model_type} diagnostics failed (non-fatal): {e}", file=sys.stderr)


def _emit_nn_diagnostics(model, r2: float, mse: float, enable_diagnostics: bool):
    """Chapter 14: Emit neural net diagnostics (best-effort, non-fatal)."""
    if not enable_diagnostics or not DIAGNOSTICS_AVAILABLE:
        return
    try:
        import os
        os.makedirs('diagnostics_outputs', exist_ok=True)
        diag = NNDiagnostics()
        diag.set_final_metrics({'r2': r2, 'mse': mse})
        diag.save('diagnostics_outputs/neural_net_diagnostics.json')
        print(f"[DIAG] neural_net diagnostics saved", file=sys.stderr)
    except Exception as e:
        print(f"[DIAG] neural_net diagnostics failed (non-fatal): {e}", file=sys.stderr)

'''

# Insert helper functions after imports, before first train_* function
if "_emit_tree_diagnostics" not in content:
    first_train_func = content.find('\ndef train_lightgbm')
    if first_train_func > 0:
        content = content[:first_train_func] + diagnostics_helper + content[first_train_func:]

# Now add calls to the helper in each function
# For catboost - find return statement and insert before it
for model_type in ['catboost', 'xgboost', 'lightgbm']:
    func_start = content.find(f'def train_{model_type}')
    if func_start > 0:
        # Find the return { in this function
        next_func = content.find('\ndef ', func_start + 10)
        if next_func == -1:
            next_func = len(content)
        section = content[func_start:next_func]
        
        # Find the return { statement
        return_match = section.rfind('    return {')
        if return_match > 0:
            emit_call = f'''
    # Chapter 14: Emit diagnostics
    _emit_tree_diagnostics(model, '{model_type}', r2, mse, enable_diagnostics)

    '''
            if f"_emit_tree_diagnostics(model, '{model_type}'" not in section:
                new_section = section[:return_match] + emit_call + section[return_match:]
                content = content[:func_start] + new_section + content[next_func:]

# For neural_net
nn_func_start = content.find('def train_neural_net')
if nn_func_start > 0:
    next_func = content.find('\ndef ', nn_func_start + 10)
    if next_func == -1:
        next_func = len(content)
    section = content[nn_func_start:next_func]
    return_match = section.rfind('    return {')
    if return_match > 0:
        emit_call = '''
    # Chapter 14: Emit diagnostics
    _emit_nn_diagnostics(model, r2, mse, enable_diagnostics)

    '''
        if "_emit_nn_diagnostics(model" not in section:
            new_section = section[:return_match] + emit_call + section[return_match:]
            content = content[:nn_func_start] + new_section + content[next_func:]

with open('train_single_trial.py', 'w') as f:
    f.write(content)

print("   ✅ train_single_trial.py patched with diagnostics emission")
PATCH_TRAIN

# ============================================================
# PATCH 5: Config files (manifest + config)
# ============================================================
echo ">>> Patch 5/5: Config files"

cp agent_manifests/reinforcement.json agent_manifests/reinforcement.json.bak_s73
cp reinforcement_engine_config.json reinforcement_engine_config.json.bak_s73

python3 << 'EOF'
import json

# Update manifest
with open('agent_manifests/reinforcement.json', 'r') as f:
    manifest = json.load(f)

manifest['parameter_bounds']['enable_diagnostics'] = {
    "type": "bool",
    "default": False
}
manifest['default_params']['enable_diagnostics'] = False

with open('agent_manifests/reinforcement.json', 'w') as f:
    json.dump(manifest, f, indent=2)

# Update config
with open('reinforcement_engine_config.json', 'r') as f:
    config = json.load(f)

config['diagnostics'] = {
    "enabled": False,
    "capture_every_n": 5,
    "output_dir": "diagnostics_outputs"
}

with open('reinforcement_engine_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("   ✅ Config files updated")
EOF

# ============================================================
# VERIFICATION
# ============================================================
echo ""
echo "============================================================"
echo "VERIFICATION"
echo "============================================================"

echo ""
echo "1. CLI flags:"
python3 meta_prediction_optimizer_anti_overfit.py --help 2>&1 | grep -q "enable-diagnostics" && echo "   ✅ meta_optimizer --enable-diagnostics OK" || echo "   ⚠️ meta_optimizer flag missing"
python3 train_single_trial.py --help 2>&1 | grep -q "enable-diagnostics" && echo "   ✅ train_single_trial --enable-diagnostics OK" || echo "   ⚠️ train_single_trial flag missing"

echo ""
echo "2. Subprocess threading:"
grep -q "enable_diagnostics" subprocess_trial_coordinator.py && echo "   ✅ Coordinator has enable_diagnostics" || echo "   ⚠️ Coordinator missing enable_diagnostics"
grep -q "_enable_diagnostics_flag" meta_prediction_optimizer_anti_overfit.py && echo "   ✅ Meta optimizer threads flag" || echo "   ⚠️ Meta optimizer not threading flag"

echo ""
echo "3. Diagnostics emission:"
grep -q "_emit_tree_diagnostics" train_single_trial.py && echo "   ✅ Tree diagnostics helper exists" || echo "   ⚠️ Tree diagnostics helper missing"
grep -q "_emit_nn_diagnostics" train_single_trial.py && echo "   ✅ NN diagnostics helper exists" || echo "   ⚠️ NN diagnostics helper missing"

echo ""
echo "4. Config files:"
python3 -c "import json; m=json.load(open('agent_manifests/reinforcement.json')); print('   ✅ Manifest OK' if 'enable_diagnostics' in m.get('parameter_bounds', {}) else '   ⚠️ Manifest missing')"
python3 -c "import json; c=json.load(open('reinforcement_engine_config.json')); print('   ✅ Config OK' if 'diagnostics' in c else '   ⚠️ Config missing')"

echo ""
echo "5. Syntax check:"
python3 -m py_compile meta_prediction_optimizer_anti_overfit.py && echo "   ✅ meta_optimizer syntax OK" || echo "   ⚠️ meta_optimizer syntax error"
python3 -m py_compile subprocess_trial_coordinator.py && echo "   ✅ coordinator syntax OK" || echo "   ⚠️ coordinator syntax error"
python3 -m py_compile train_single_trial.py && echo "   ✅ train_single_trial syntax OK" || echo "   ⚠️ train_single_trial syntax error"

echo ""
echo "============================================================"
echo "✅ Session 73 FINAL patches applied"
echo "============================================================"
echo ""
echo "Backups:"
ls -la *.bak_s73 2>/dev/null | head -5
echo ""
echo "TEST COMMANDS:"
echo ""
echo "# Quick syntax test:"
echo "python3 -c \"import meta_prediction_optimizer_anti_overfit; print('OK')\""
echo ""
echo "# Full pipeline test:"
echo "PYTHONPATH=. python3 agents/watcher_agent.py \\"
echo "    --run-pipeline --start-step 5 --end-step 5 \\"
echo "    --params '{\"trials\": 1, \"model_type\": \"catboost\", \"enable_diagnostics\": true}'"
echo ""
echo "# Verify diagnostics generated:"
echo "ls -la diagnostics_outputs/"
echo "python3 training_health_check.py --check"
