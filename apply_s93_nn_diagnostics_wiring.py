#!/usr/bin/env python3
"""
S93 NN Diagnostics Hook Wiring — v3 (Team Beta final)

Wires NNDiagnostics hooks into train_neural_net() training loop.
Previously, _emit_nn_diagnostics() created an empty object AFTER training.
Now: attach hooks before training, record each epoch, save live data.

Team Beta fixes (v2):
- Fix #1: DataParallel .module unwrap before attach
- Fix #2: val_mse/mse variable fallback (NameError safe)
- Fix #3: Function-scoped anchor search (only patch inside train_neural_net)
- Fix A/B/C: Resilient anchoring, try-wrapped privates, detach before save

Team Beta fixes (v3):
- Red flag #1: Idempotency guard — safe to run multiple times
- Red flag #2: Tightened optimizer.step() anchor (loss.backward + step)
- Nice-to-have A: _write_canonical_diagnostics existence check
- Nice-to-have B: None fallback instead of 0 for missing MSE

Usage:
    cd ~/distributed_prng_analysis
    source ~/venvs/torch/bin/activate
    python3 apply_s93_nn_diagnostics_wiring.py
"""

import os, sys, shutil, re

TARGET = "train_single_trial.py"
BACKUP = TARGET + ".pre_s93_diag_wiring"

SENTINEL = "S93: Wire NNDiagnostics hooks into training loop"


def find_function_range(content, func_name):
    """Find the start and end positions of a top-level function in the file.
    Returns (start, end) character positions."""
    pattern = re.compile(r'^def ' + re.escape(func_name) + r'\(', re.MULTILINE)
    match = pattern.search(content)
    if not match:
        return None, None
    start = match.start()

    # Find next top-level def or end of file
    next_def = re.compile(r'^def ', re.MULTILINE)
    next_match = next_def.search(content, match.end())
    end = next_match.start() if next_match else len(content)

    return start, end


def main():
    if not os.path.isfile(TARGET):
        print(f"❌ {TARGET} not found. Run from ~/distributed_prng_analysis/")
        sys.exit(1)

    content = open(TARGET).read()

    # Find train_neural_net function boundaries
    fn_start, fn_end = find_function_range(content, "train_neural_net")
    if fn_start is None:
        print("❌ Could not find train_neural_net() function")
        sys.exit(1)

    fn_body = content[fn_start:fn_end]

    # =========================================================================
    # IDEMPOTENCY GUARD (TB v3 Red flag #1)
    # =========================================================================
    if SENTINEL in fn_body:
        print(f"⏭️  SKIP: S93 diagnostics wiring already present in train_neural_net()")
        print(f"   (sentinel found: '{SENTINEL}')")
        print(f"   To force re-apply, restore from backup first:")
        print(f"   cp {BACKUP} {TARGET}")
        sys.exit(0)

    # Also check for partial application
    partial_markers = ["_epoch_train_loss", "_nn_diag = None", "_epoch_batches"]
    found_partial = [m for m in partial_markers if m in fn_body]
    if found_partial:
        print(f"⚠️  Partial application detected: {found_partial}")
        print(f"   Restore from backup before re-applying:")
        print(f"   cp {BACKUP} {TARGET}")
        sys.exit(1)

    print(f"Found train_neural_net() at chars {fn_start}-{fn_end} ({len(fn_body)} chars)")

    # Backup
    if not os.path.exists(BACKUP):
        shutil.copy2(TARGET, BACKUP)
        print(f"📦 Backup: {BACKUP}")
    else:
        print(f"⚠️  Backup exists: {BACKUP}")

    patches_applied = 0

    # =========================================================================
    # PATCH 1: Initialize and attach diagnostics BEFORE the training loop
    # TB Fix #1: Unwrap DataParallel before attaching hooks
    # =========================================================================

    anchor1 = "    for epoch in range(epochs):"

    if anchor1 not in fn_body:
        print(f"❌ PATCH 1: 'for epoch in range(epochs):' not found in train_neural_net")
        sys.exit(1)

    insert1 = """    # {sentinel}
    _nn_diag = None
    _base_model = model.module if hasattr(model, 'module') else model  # TB Fix #1: DataParallel
    if enable_diagnostics and DIAGNOSTICS_AVAILABLE:
        try:
            _nn_diag = NNDiagnostics()
            _nn_diag.attach(_base_model)
            print(f"[DIAG] NNDiagnostics attached ({{len(_nn_diag._layer_names)}} layers)", file=sys.stderr)
        except Exception as _diag_err:
            print(f"[DIAG] NNDiagnostics attach failed (non-fatal): {{_diag_err}}", file=sys.stderr)
            _nn_diag = None

""".format(sentinel=SENTINEL)

    idx1 = fn_body.index(anchor1)
    fn_body = fn_body[:idx1] + insert1 + fn_body[idx1:]
    patches_applied += 1
    print(f"✅ PATCH 1: NNDiagnostics init + attach (DataParallel-safe)")

    # =========================================================================
    # PATCH 2a: Add loss accumulator at the start of each epoch
    # =========================================================================

    anchor2a = "    for epoch in range(epochs):\n        model.train()"

    if anchor2a not in fn_body:
        print(f"❌ PATCH 2a: epoch loop start not found")
        sys.exit(1)

    replacement2a = """    for epoch in range(epochs):
        model.train()
        _epoch_train_loss = 0.0
        _epoch_batches = 0"""

    fn_body = fn_body.replace(anchor2a, replacement2a, 1)
    patches_applied += 1
    print(f"✅ PATCH 2a: Epoch loss accumulator added")

    # =========================================================================
    # PATCH 2b: Accumulate batch loss after optimizer.step()
    # TB v3 Red flag #2: Tightened anchor — match loss.backward() + step() pair
    # =========================================================================

    # Use regex to match the loss.backward() / optimizer.step() pair
    # This is more specific than just optimizer.step() alone
    anchor2b_pattern = re.compile(
        r'(            loss\.backward\(\)\n'
        r'            optimizer\.step\(\))'
    )

    match2b = anchor2b_pattern.search(fn_body)
    if not match2b:
        print(f"❌ PATCH 2b: loss.backward()/optimizer.step() pair not found")
        sys.exit(1)

    old2b = match2b.group(1)
    replacement2b = old2b + "\n            _epoch_train_loss += loss.item()\n            _epoch_batches += 1"

    fn_body = fn_body[:match2b.start()] + replacement2b + fn_body[match2b.end():]
    patches_applied += 1
    print(f"✅ PATCH 2b: Batch loss accumulation (tightened anchor)")

    # =========================================================================
    # PATCH 3: Call on_round_end() after validation, before early stopping
    # TB Fix #3: Function-scoped (we're already in fn_body)
    # =========================================================================

    anchor3 = "        if val_loss < best_val_loss:"

    if anchor3 not in fn_body:
        print(f"❌ PATCH 3: early stopping check not found in train_neural_net")
        sys.exit(1)

    insert3 = """        # S93: Record epoch diagnostics
        if _nn_diag is not None:
            try:
                _avg_train_loss = _epoch_train_loss / max(_epoch_batches, 1)
                _nn_diag.on_round_end(
                    round_num=epoch,
                    train_loss=_avg_train_loss,
                    val_loss=val_loss,
                    learning_rate=lr,
                )
            except Exception as _rnd_err:
                if epoch == 0:
                    print(f"[DIAG] on_round_end failed (non-fatal): {_rnd_err}", file=sys.stderr)

"""

    # Insert before first occurrence (function-scoped)
    idx3 = fn_body.index(anchor3)
    fn_body = fn_body[:idx3] + insert3 + fn_body[idx3:]
    patches_applied += 1
    print(f"✅ PATCH 3: on_round_end() called each epoch (function-scoped)")

    # =========================================================================
    # PATCH 4: Replace post-hoc stub with live diagnostics save
    # TB Fix #2: val_mse/mse fallback via locals()
    # TB Fix C: detach in separate try block
    # TB Fix B: try-wrapped private attr access
    # TB v3 Nice-to-have A: _write_canonical_diagnostics existence check
    # TB v3 Nice-to-have B: None fallback (not 0) for missing MSE
    # =========================================================================

    emit_pattern = re.compile(r'    _emit_nn_diagnostics\([^)]+\)')
    emit_match = emit_pattern.search(fn_body)

    if not emit_match:
        print(f"❌ PATCH 4: _emit_nn_diagnostics() call not found in train_neural_net")
        sys.exit(1)

    old_emit = emit_match.group()
    print(f"   Found: '{old_emit.strip()}'")

    replacement4 = """    # S93: Save live diagnostics (replaces empty post-hoc stub)
    # TB Fix C: Always detach hooks (separate try)
    if _nn_diag is not None:
        try:
            _nn_diag.detach()
        except Exception:
            pass
    # TB Fix #2: Robust MSE resolution (None if neither exists — v3 nice-to-have B)
    if _nn_diag is not None and enable_diagnostics:
        try:
            _mse_val = locals().get('val_mse', locals().get('mse', None))
            _nn_diag.set_final_metrics({'r2': r2, 'mse': _mse_val})
            os.makedirs('diagnostics_outputs', exist_ok=True)
            _nn_diag.save('diagnostics_outputs/neural_net_diagnostics.json')
            # v3 nice-to-have A: check existence before calling
            if '_write_canonical_diagnostics' in dir() or '_write_canonical_diagnostics' in globals():
                _write_canonical_diagnostics('diagnostics_outputs/neural_net_diagnostics.json')
            # TB Fix B: try-wrapped private attr access
            try:
                _rnd_count = len(_nn_diag._round_data)
                _lyr_count = len(_nn_diag._layer_names)
            except Exception:
                _rnd_count = _lyr_count = '?'
            print(f"[DIAG] NN diagnostics saved: {_rnd_count} rounds, {_lyr_count} layers", file=sys.stderr)
        except Exception as _save_err:
            print(f"[DIAG] NN diagnostics save failed (non-fatal): {_save_err}", file=sys.stderr)
    elif enable_diagnostics:
        _emit_nn_diagnostics(model, r2, locals().get('val_mse', locals().get('mse', None)), enable_diagnostics)"""

    fn_body = fn_body[:emit_match.start()] + replacement4 + fn_body[emit_match.end():]
    patches_applied += 1
    print(f"✅ PATCH 4: Live diagnostics save (all TB fixes applied)")

    # =========================================================================
    # Reassemble file
    # =========================================================================

    content = content[:fn_start] + fn_body + content[fn_end:]
    open(TARGET, 'w').write(content)

    # Syntax check
    import py_compile
    try:
        py_compile.compile(TARGET, doraise=True)
        print(f"\n✅ Syntax check PASSED")
    except py_compile.PyCompileError as e:
        print(f"\n❌ SYNTAX ERROR: {e}")
        shutil.copy2(BACKUP, TARGET)
        print(f"Restored from backup")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"S93 NN Diagnostics Wiring v3 — {patches_applied}/6 patches applied")
    print(f"{'='*60}")
    print(f"\nTeam Beta fixes (all):")
    print(f"  ✅ #1: DataParallel .module unwrap")
    print(f"  ✅ #2: val_mse/mse robust fallback (None not 0)")
    print(f"  ✅ #3: Function-scoped anchor search")
    print(f"  ✅ RF1: Idempotency guard (sentinel + partial check)")
    print(f"  ✅ RF2: Tightened optimizer.step() anchor (backward+step pair)")
    print(f"  ✅ A: _write_canonical_diagnostics existence check")
    print(f"  ✅ B: try-wrapped private attr access")
    print(f"  ✅ C: detach in separate try block")
    print(f"\nTo re-apply after changes:")
    print(f"  cp {BACKUP} {TARGET}")
    print(f"  python3 apply_s93_nn_diagnostics_wiring.py")
    print(f"\nTest:")
    print(f"  python3 train_single_trial.py --model-type neural_net --trials 1 --enable-diagnostics 2>&1 | tee /tmp/nn_diag.log")
    print(f"  cat diagnostics_outputs/neural_net_diagnostics.json | python3 -m json.tool | head -40")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
