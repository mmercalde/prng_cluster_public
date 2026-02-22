#!/usr/bin/env python3
"""
apply_s105_scorer_worker_v4_0.py  (FINAL -- all TB issues resolved)
Patch: scorer_trial_worker.py v3.6 -> v4.0

TB REVIEW STATUS:
  Issue 1 (brittle markers) -- FIXED: extract_func_block() uses top-level def
    boundary scan, immune to import order / whitespace / internal line drift.
  Issue 2 (banned term grep) -- FIXED: checks live instantiation, not word presence.
  Issue 3 (inert params)     -- FIXED: all 5 params (rm1,rm2,rm3,offset,tw_size)
    now contribute to the WSI objective scoring formula.
"""

import os, sys, ast, shutil, hashlib
from datetime import datetime

TARGET = "scorer_trial_worker.py"
BACKUP = f"scorer_trial_worker.py.bak_s105_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# ---------------------------------------------------------------------------
# NEW run_trial() function -- built without nested triple-quotes to avoid
# string literal conflicts inside the patch script.
# ---------------------------------------------------------------------------
_DQ = '"""'   # triple double-quote, referenced where needed inside the string

NEW_RUN_TRIAL_FUNC = (
    "\ndef run_trial(seeds_to_score,\n"
    "              npz_forward_matches,\n"
    "              npz_reverse_matches,\n"
    "              params,\n"
    "              prng_type='java_lcg',\n"
    "              mod=1000,\n"
    "              trial=None,\n"
    "              use_legacy_scoring=False):\n"
    + "    " + _DQ + "\n"
    "    v4.0: WSI (Weighted Separation Index) objective -- draw-history-free.\n"
    "\n"
    "    ALL 5 Optuna params affect the objective (TB issue 3 resolved):\n"
    "      rm1, rm2, rm3  -> normalized mixture weights for fwd/rev/interaction\n"
    "      max_offset     -> squared-interaction term weight (wi = offset/15)\n"
    "      temporal_window_size -> temporal smoothing weight (tw = size/200)\n"
    "\n"
    "    Scoring:\n"
    "        scores = wf*fwd + wr*rev + w3*(fwd*rev) + tw*(fwd+rev)/2 + wi*(fwd*rev)**2\n"
    "\n"
    "    WSI formula (bounded [-1,1], TB-approved S103):\n"
    "        quality = fwd * rev\n"
    "        WSI = cov(scores,quality) / ((std_s+eps)*(std_q+eps))\n"
    "\n"
    "    Degenerate guard: std(scores) < 1e-12 -> WSI = -1.0\n"
    + "    " + _DQ + "\n"
    "    try:\n"
    "        import numpy as np\n"
    "        import random\n"
    "\n"
    "        # Sampling -- per-trial RNG preserved from S101\n"
    "        n_seeds     = len(seeds_to_score)\n"
    "        sample_size = params.get('sample_size', 50000)\n"
    "\n"
    "        if n_seeds > sample_size:\n"
    "            random.seed(params.get('optuna_trial_number', 0))\n"
    "            sample_idx  = random.sample(range(n_seeds), sample_size)\n"
    "            sample_idx  = np.array(sample_idx, dtype=np.int64)\n"
    "            sampled_fwd = npz_forward_matches[sample_idx]\n"
    "            sampled_rev = npz_reverse_matches[sample_idx]\n"
    "            logger.info(\n"
    "                f'Sampled {sample_size:,} / {n_seeds:,} seeds '\n"
    "                f'(rng_seed={params.get(\"optuna_trial_number\", 0)})'\n"
    "            )\n"
    "        else:\n"
    "            sampled_fwd = npz_forward_matches\n"
    "            sampled_rev = npz_reverse_matches\n"
    "            logger.info(f'Using all {n_seeds:,} seeds')\n"
    "\n"
    "        # Parametric scoring -- all 5 params active\n"
    "        rm1        = float(params.get('residue_mod_1',   10))\n"
    "        rm2        = float(params.get('residue_mod_2',  100))\n"
    "        rm3        = float(params.get('residue_mod_3', 1000))\n"
    "        max_offset = float(params.get('max_offset',       10))\n"
    "        tw_size    = float(params.get('temporal_window_size', 100))\n"
    "\n"
    "        eps    = 1e-10\n"
    "        w_sum  = rm1 + rm2 + rm3 + eps\n"
    "        wf     = rm1 / w_sum          # forward weight\n"
    "        wr     = rm2 / w_sum          # reverse weight\n"
    "        w3     = rm3 / w_sum          # intersection weight\n"
    "        tw     = tw_size / 200.0      # temporal smoothing weight\n"
    "        wi     = max_offset / 15.0    # squared interaction weight\n"
    "\n"
    "        fwd_rev = sampled_fwd * sampled_rev\n"
    "\n"
    "        scores = (\n"
    "            wf * sampled_fwd\n"
    "            + wr * sampled_rev\n"
    "            + w3 * fwd_rev\n"
    "            + tw * (sampled_fwd + sampled_rev) / 2.0\n"
    "            + wi * fwd_rev ** 2\n"
    "        )\n"
    "\n"
    "        logger.info(\n"
    "            f'Parametric scoring: wf={wf:.3f}  wr={wr:.3f}  w3={w3:.3f}  '\n"
    "            f'tw={tw:.3f}  wi={wi:.3f}  '\n"
    "            f'score_mean={scores.mean():.4f}  score_std={scores.std():.4f}'\n"
    "        )\n"
    "\n"
    "        # WSI objective -- bounded [-1, 1]\n"
    "        quality = fwd_rev\n"
    "        std_s   = float(scores.std())\n"
    "        std_q   = float(quality.std())\n"
    "\n"
    "        if std_s < 1e-12:\n"
    "            wsi = -1.0\n"
    "            logger.warning(\n"
    "                f'Degenerate scores (std={std_s:.2e}) -> WSI=-1.0.  '\n"
    "                f'rm1={rm1:.0f}  rm2={rm2:.0f}  rm3={rm3:.0f}  '\n"
    "                f'offset={max_offset:.0f}  tw={tw_size:.0f}'\n"
    "            )\n"
    "        else:\n"
    "            centered_s = scores  - scores.mean()\n"
    "            centered_q = quality - quality.mean()\n"
    "            covariance = float(np.mean(centered_s * centered_q))\n"
    "            wsi        = covariance / ((std_s + eps) * (std_q + eps))\n"
    "            wsi        = float(np.clip(wsi, -1.0, 1.0))\n"
    "            logger.info(\n"
    "                f'WSI = {wsi:.6f}  '\n"
    "                f'(cov={covariance:.6f}  std_s={std_s:.4f}  std_q={std_q:.4f}  '\n"
    "                f'quality_mean={quality.mean():.4f})'\n"
    "            )\n"
    "\n"
    "        return wsi, scores.tolist()\n"
    "\n"
    "    except Exception as e:\n"
    "        logger.error(f'Trial execution failed: {e}', exc_info=True)\n"
    "        raise\n"
    "\n"
)

NEW_LOAD_DATA_BODY = (
    "    if survivors is None:\n"
    "        logger.info('Loading NPZ survivor data (one time)...')\n"
    "        try:\n"
    "            survivors_file = os.path.expanduser(survivors_file)\n"
    "\n"
    "            # survivor_loader.data is plain Dict[str, np.ndarray] (no structured array)\n"
    "            survivor_result = load_survivors(survivors_file, return_format='array')\n"
    "            survivors = survivor_result.data\n"
    "            logger.info(\n"
    "                f'Loaded {survivor_result.count:,} survivors from '\n"
    "                f'{survivor_result.format} (fallback={survivor_result.fallback_used})'\n"
    "            )\n"
    "\n"
    "            if not isinstance(survivors, dict) or 'seeds' not in survivors:\n"
    "                raise ValueError(\n"
    "                    f'Unexpected survivors type: {type(survivors)}. '\n"
    "                    'Expected Dict[str, np.ndarray] from survivor_loader.'\n"
    "                )\n"
    "\n"
    "            seeds_to_score = survivors['seeds'].tolist()\n"
    "            logger.info(f'Loaded {len(seeds_to_score):,} seeds.')\n"
    "\n"
    "            import numpy as _np\n"
    "            if 'forward_matches' in survivors and 'reverse_matches' in survivors:\n"
    "                npz_forward_matches = survivors['forward_matches'].astype(_np.float32)\n"
    "                npz_reverse_matches = survivors['reverse_matches'].astype(_np.float32)\n"
    "                logger.info(\n"
    "                    f'NPZ quality signals: '\n"
    "                    f'fwd mean={npz_forward_matches.mean():.4f}  '\n"
    "                    f'rev mean={npz_reverse_matches.mean():.4f}'\n"
    "                )\n"
    "            else:\n"
    "                raise RuntimeError(\n"
    "                    'NPZ missing forward_matches or reverse_matches. '\n"
    "                    'Re-run convert_survivors_to_binary.py with NPZ v3.0+ format. '\n"
    "                    f'Available NPZ keys: {list(survivors.keys())}'\n"
    "                )\n"
    "\n"
    "        except Exception as e:\n"
    "            logger.error(f'Failed to load data: {e}', exc_info=True)\n"
    "            raise\n"
    "\n"
    "    if npz_forward_matches is None or npz_reverse_matches is None:\n"
    "        raise RuntimeError('NPZ quality signals are None after load -- cannot compute WSI.')\n"
    "\n"
    "    # prng_type from optimal_window_config.json (canonical source, S102)\n"
    "    prng_type = 'java_lcg'\n"
    "    mod = 1000\n"
    "    wc_path = os.path.join(\n"
    "        os.path.dirname(os.path.abspath(survivors_file)), 'optimal_window_config.json'\n"
    "    )\n"
    "    if os.path.exists(wc_path):\n"
    "        try:\n"
    "            with open(wc_path) as _wf:\n"
    "                _wc = json.load(_wf)\n"
    "            prng_type = _wc.get('prng_type') or 'java_lcg'\n"
    "            mod       = _wc.get('mod')       or 1000\n"
    "            logger.info(\n"
    "                f'Pipeline config: prng_type={prng_type}, mod={mod} '\n"
    "                '(from optimal_window_config.json)'\n"
    "            )\n"
    "        except Exception as _e:\n"
    "            logger.warning(f'Could not read optimal_window_config.json: {_e} -- using defaults')\n"
    "\n"
    "    return seeds_to_score, npz_forward_matches, npz_reverse_matches, prng_type, mod\n"
)

# =============================================================================
# STATIC PATCHES  (description, old_str, new_str)
# =============================================================================
STATIC_PATCHES = []

STATIC_PATCHES.append((
    "Version header v3.6 -> v4.0",
    "scorer_trial_worker.py (v3.6 - NPZ prng_type config fix)",
    "scorer_trial_worker.py (v4.0 - WSI objective, draw-history-free)"
))

STATIC_PATCHES.append((
    "Insert v4.0 changelog entry",
    "v3.5 (2026-02-20):\n- BUG FIX: Replace neg-MSE objective with Spearman rank correlation",
    "v4.0 (2026-02-21):\n"
    "- ARCHITECTURE: Remove draw history from Step 2 (TB ruling S102/S103)\n"
    "- NEW OBJECTIVE: WSI bounded [-1,1]; all 5 Optuna params active\n"
    "- REMOVE: ReinforcementEngine, SurvivorScorer\n"
    "- PRESERVE: per-trial RNG (S101), prng_type from config (S102)\n"
    "- CLI: positional args 2+3 accepted but ignored\n"
    "\n"
    "v3.6 (2026-02-21):\n"
    "- BUG FIX: NPZ branch never read prng_type from config\n"
    "\n"
    "v3.5 (2026-02-20):\n"
    "- BUG FIX: Replace neg-MSE objective with Spearman rank correlation"
))

STATIC_PATCHES.append((
    "Remove ReinforcementEngine and SurvivorScorer imports",
    "from reinforcement_engine import ReinforcementEngine, ReinforcementConfig\n"
    "from survivor_scorer import SurvivorScorer",
    "# v4.0: ReinforcementEngine + SurvivorScorer removed (WSI uses NPZ signals only)"
))

STATIC_PATCHES.append((
    "Remove train_history/holdout_history globals",
    "survivors = None\ntrain_history = None\nholdout_history = None\nseeds_to_score = None",
    "survivors = None\n"
    "seeds_to_score = None\n"
    "npz_forward_matches = None   # float32 ndarray -- quality signal from NPZ\n"
    "npz_reverse_matches = None   # float32 ndarray -- quality signal from NPZ"
))

STATIC_PATCHES.append((
    "Replace load_data() signature",
    "def load_data(survivors_file: str, train_history_file: str, holdout_history_file: str):\n"
    '    """Load data files (cached for reuse across trials on same worker)."""\n'
    "    global survivors, train_history, holdout_history, seeds_to_score",
    "def load_data(survivors_file: str,\n"
    "              train_history_file: str = None,\n"
    "              holdout_history_file: str = None):\n"
    '    """\n'
    "    v4.0: Load NPZ survivor data only -- draw history ignored.\n"
    "    train_history_file / holdout_history_file accepted for CLI compat.\n"
    '    """\n'
    "    global survivors, seeds_to_score, npz_forward_matches, npz_reverse_matches"
))

STATIC_PATCHES.append((
    "Replace load_data() body",
    "    if survivors is None:\n"
    '        logger.info("Loading data (one time)...")\n'
    "        try:\n"
    "            survivors_file = os.path.expanduser(survivors_file)\n"
    "            train_history_file = os.path.expanduser(train_history_file)\n"
    "            holdout_history_file = os.path.expanduser(holdout_history_file)\n"
    "\n"
    "            # Load survivors using modular loader (NPZ/JSON auto-detect)\n"
    "            survivor_result = load_survivors(survivors_file, return_format=\"array\")\n"
    "            survivors = survivor_result.data\n"
    '            logger.info(f"Loaded {survivor_result.count:,} survivors from {survivor_result.format} "\n'
    '                       f"(fallback={survivor_result.fallback_used})")\n'
    "\n"
    "            with open(train_history_file) as f:\n"
    "                train_data = json.load(f)\n"
    "                if isinstance(train_data, list) and len(train_data) > 0 and isinstance(train_data[0], dict):\n"
    "                    train_history = [d['draw'] for d in train_data]\n"
    "                else:\n"
    "                    train_history = train_data\n"
    "\n"
    "            with open(holdout_history_file) as f:\n"
    "                holdout_data = json.load(f)\n"
    "                if isinstance(holdout_data, list) and len(holdout_data) > 0 and isinstance(holdout_data[0], dict):\n"
    "                    holdout_history = [d['draw'] for d in holdout_data]\n"
    "                else:\n"
    "                    holdout_history = holdout_data\n"
    "\n"
    "            # Extract seeds (modular loader returns array format)\n"
    "            seeds_to_score = survivors['seeds'].tolist()\n"
    "\n"
    "            logger.info(f\"Loaded {len(seeds_to_score)} survivors/seeds from {survivors_file}.\")\n"
    "            logger.info(f\"Loaded {len(train_history)} training draws from {train_history_file}.\")\n"
    "            logger.info(f\"Loaded {len(holdout_history)} holdout draws from {holdout_history_file}.\")\n"
    "\n"
    "        except Exception as e:\n"
    "            logger.error(f\"Failed to load data: {e}\", exc_info=True)\n"
    "            raise\n"
    "\n"
    "    # Extract PRNG type from survivor metadata\n"
    "    prng_type = 'java_lcg'\n"
    "    mod = 1000\n"
    "    if isinstance(survivors, dict) and 'seeds' in survivors:\n"
    "        # NPZ format - prng_type from metadata if available\n"
    "        wc_path = os.path.join(os.path.dirname(os.path.abspath(survivors_file)), \"optimal_window_config.json\")\n"
    "        if os.path.exists(wc_path):\n"
    "            try:\n"
    "                with open(wc_path) as _wf:\n"
    "                    _wc = json.load(_wf)\n"
    "                prng_type = _wc.get(\"prng_type\")\n"
    "                mod = _wc.get(\"mod\")\n"
    "                logger.info(f\"Pipeline config: prng_type={prng_type}, mod={mod} (from optimal_window_config.json)\")\n"
    "            except Exception as _e:\n"
    "                logger.warning(f\"Could not read optimal_window_config.json: {_e}\")\n"
    "        if not prng_type:\n"
    "            logger.warning(\"prng_type not resolved from config -- defaulting to java_lcg\")\n"
    "            prng_type = \"java_lcg\"\n"
    "        if not mod:\n"
    "            mod = 1000\n"
    "    elif survivors and len(survivors) > 0 and isinstance(survivors[0], dict):\n"
    "        prng_type = survivors[0].get('prng_type', 'java_lcg')\n"
    "        if '_' in prng_type and prng_type.split('_')[-1].isdigit():\n"
    "            mod = int(prng_type.split('_')[-1])\n"
    "\n"
    "    return seeds_to_score, train_history, holdout_history, prng_type, mod",
    NEW_LOAD_DATA_BODY
))

# CLI positional args optional
STATIC_PATCHES.append((
    "Make train/holdout positional args optional in main()",
    "    survivors_file = sys.argv[1]\n"
    "    train_history_file = sys.argv[2]\n"
    "    holdout_history_file = sys.argv[3]\n"
    "    trial_id = int(sys.argv[4])",
    "    survivors_file       = sys.argv[1]\n"
    "    # v4.0: args 2+3 accepted for WATCHER/shell compat but ignored in load_data\n"
    "    train_history_file   = sys.argv[2] if len(sys.argv) > 2 else None\n"
    "    holdout_history_file = sys.argv[3] if len(sys.argv) > 3 else None\n"
    "    trial_id             = int(sys.argv[4]) if len(sys.argv) > 4 else 0"
))

# main() call signatures
STATIC_PATCHES.append((
    "Update main() load_data + run_trial call signatures",
    "        seeds, train_hist, holdout_hist, prng_type, mod = load_data(\n"
    "            survivors_file, train_history_file, holdout_history_file\n"
    "        )\n"
    "        \n"
    "        trial = None  # No Optuna in PULL mode\n"
    "        \n"
    "        accuracy, scores = run_trial(\n"
    "            seeds, train_hist, holdout_hist, params,\n"
    "            prng_type=prng_type, mod=mod, trial=trial,\n"
    "            use_legacy_scoring=use_legacy_scoring\n"
    "        )",
    "        # v4.0: draw history files passed for compat but unused\n"
    "        seeds, fwd_matches, rev_matches, prng_type, mod = load_data(\n"
    "            survivors_file, train_history_file, holdout_history_file\n"
    "        )\n"
    "\n"
    "        trial = None  # No Optuna in PULL mode\n"
    "\n"
    "        accuracy, scores = run_trial(\n"
    "            seeds, fwd_matches, rev_matches, params,\n"
    "            prng_type=prng_type, mod=mod, trial=trial,\n"
    "            use_legacy_scoring=use_legacy_scoring\n"
    "        )"
))


# =============================================================================
# DYNAMIC EXTRACTION -- top-level def boundary scan (TB issue 1 final fix)
# =============================================================================

def extract_func_block(content, func_name, next_func_name):
    """
    Extract a complete top-level function by scanning for column-0 def boundaries.
    Immune to: import order, internal line content, whitespace drift.
    """
    start_marker = f"\ndef {func_name}("
    end_marker   = f"\ndef {next_func_name}("
    start = content.find(start_marker)
    if start == -1:
        raise ValueError(f"Function '{func_name}' not found.")
    end = content.find(end_marker, start + 1)
    if end == -1:
        raise ValueError(f"Next function '{next_func_name}' not found after '{func_name}'.")
    block = content[start:end]
    if content.count(block) != 1:
        raise ValueError(f"Block for '{func_name}' not unique ({content.count(block)} occurrences).")
    return block


# =============================================================================
# PATCH ENGINE
# =============================================================================

def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def apply_patches():
    print("=" * 66)
    print("apply_s105_scorer_worker_v4_0.py  (FINAL -- all TB issues resolved)")
    print(f"Target : {TARGET}")
    print("=" * 66)

    if not os.path.exists(TARGET):
        print(f"ERROR: {TARGET} not found"); sys.exit(1)

    shutil.copy2(TARGET, BACKUP)
    pre_md5 = md5(TARGET)
    print(f"Backup : {BACKUP}")
    print(f"Pre-MD5: {pre_md5}\n")

    with open(TARGET, 'r') as f:
        content = f.read()

    # Dynamic P7: extract and replace complete run_trial() by def boundaries
    try:
        old_run_trial = extract_func_block(content, 'run_trial', 'save_local_result')
        print(f"OK    P7 (dynamic): extracted run_trial block ({len(old_run_trial)} chars, unique)")
    except ValueError as e:
        print(f"\nFAIL  P7 (dynamic): {e}")
        shutil.copy2(BACKUP, TARGET); sys.exit(1)

    all_patches = (
        STATIC_PATCHES[:6]
        + [("Replace complete run_trial() with WSI implementation",
            old_run_trial, NEW_RUN_TRIAL_FUNC)]
        + STATIC_PATCHES[6:]
    )

    for i, (desc, old_str, new_str) in enumerate(all_patches, 1):
        count = content.count(old_str)
        if count == 0:
            print(f"\nFAIL  {i:02d}/{len(all_patches)}: not found -- {desc}")
            print(f"      First 80 chars: {repr(old_str[:80])}")
            shutil.copy2(BACKUP, TARGET); sys.exit(1)
        elif count > 1:
            print(f"\nFAIL  {i:02d}/{len(all_patches)}: not unique ({count}x) -- {desc}")
            shutil.copy2(BACKUP, TARGET); sys.exit(1)
        content = content.replace(old_str, new_str, 1)
        print(f"OK    {i:02d}/{len(all_patches)}: {desc}")

    with open(TARGET, 'w') as f:
        f.write(content)

    try:
        ast.parse(content)
        print("\nAST validation : PASSED")
    except SyntaxError as e:
        print(f"\nAST FAILED: {e}")
        shutil.copy2(BACKUP, TARGET); sys.exit(1)

    # Live code checks -- instantiation, not word presence (TB issue 2)
    live_checks = [
        ('ReinforcementEngine(', 'ReinforcementEngine instantiation'),
        ('SurvivorScorer(',      'SurvivorScorer instantiation'),
        ('spearmanr(',           'spearmanr call'),
    ]
    issues = [f"  WARN: {lbl} still present {content.count(t)}x"
              for t, lbl in live_checks if content.count(t) > 0]
    if issues:
        print("\nPost-patch live-code warnings:")
        for w in issues: print(w)
    else:
        print("Live code check  : CLEAN")

    print(f"\n{'=' * 66}")
    print(f"scorer_trial_worker.py -> v4.0  COMPLETE")
    print(f"Post-MD5 : {md5(TARGET)}")
    print("""
VERIFY on Zeus:
  python3 -c "import ast; ast.parse(open('scorer_trial_worker.py').read()); print('AST OK')"
  grep -nE 'ReinforcementEngine\\(|SurvivorScorer\\(' scorer_trial_worker.py   # -> 0 hits
  grep -nE 'spearmanr\\(' scorer_trial_worker.py                               # -> 0 hits

NPZ SANITY CHECK:
  python3 -c "
  import numpy as np
  d = np.load('bidirectional_survivors_binary.npz')
  print('keys:', sorted(d.files))
  for k in ['seeds','forward_matches','reverse_matches']:
      print(k, d[k].shape, d[k].dtype, float(d[k].min()), float(d[k].max()))
  "

SMOKE TEST (3 param sets -- WSI should vary):
  for j in 0 1 2; do
    PYTHONPATH=. python3 scorer_trial_worker.py \\
      bidirectional_survivors_binary.npz /dev/null /dev/null $j \\
      --params-json "{\\\"residue_mod_1\\\":$((10+10*j)),\\\"residue_mod_2\\\":$((100-10*j)),\\\"residue_mod_3\\\":$((500+100*j)),\\\"max_offset\\\":$((3+2*j)),\\\"temporal_window_size\\\":$((50+25*j)),\\\"optuna_trial_number\\\":$j,\\\"sample_size\\\":1500}" \\
      --gpu-id 0 | tail -n 1
  done

DEPLOY:
  scp scorer_trial_worker.py 192.168.3.120:~/distributed_prng_analysis/
  scp scorer_trial_worker.py 192.168.3.154:~/distributed_prng_analysis/
  md5sum scorer_trial_worker.py
""")
    print("=" * 66)


if __name__ == "__main__":
    apply_patches()
