# S170 Config-Mode Parity Patch

**TB ruling:** APPROVE OPTION 1 — parity restoration, not a feature change.

**Severity:** HIGH — silent execution path divergence
**Scope:** `window_optimizer.py` only
**Lines changed:** 9 added (3 in signature, 6 in body, 4 in call site)
**Behavior change:** None outside `--config-file` mode

## What it fixes

`run_with_config()` was missing two coordinator attribute assignments that the Bayesian path has at lines 614-616:

```python
coordinator.use_persistent_workers = use_persistent_workers
coordinator.pwc_transport          = pwc_transport
```

Without these, the integration layer's `getattr(coordinator, 'use_persistent_workers', False)` defaulted to False and execution silently fell through to the legacy SSH job distribution path — even when `--use-persistent-workers --pwc-transport tcp` were passed on the CLI.

## Files in this delivery

| File | Purpose |
|:---|:---|
| `patch_config_mode_parity.py` | Idempotent patcher. Apply once on Zeus. |
| `README_S170_PARITY.md` | This file. |

## Apply on Zeus

```bash
cd ~/distributed_prng_analysis

# Backup
cp window_optimizer.py window_optimizer.py.s170_config_parity_bak

# Apply
python3 patch_config_mode_parity.py window_optimizer.py

# Verify syntax
python3 -c "import ast; ast.parse(open('window_optimizer.py').read()); print('AST: OK')"

# Verify markers (expect 4 lines)
grep -n "S170-PARITY" window_optimizer.py
```

If anything fails, restore:
```bash
cp window_optimizer.py.s170_config_parity_bak window_optimizer.py
```

## TB's required validation

**Pre-launch verification:**
```bash
grep -n "use_persistent_workers" window_optimizer.py
grep -n "pwc_transport" window_optimizer.py
```

Expect to see the new lines in `run_with_config()` body and call site.

**At launch, the log MUST show:**
```
[PWC-TCP] TCP transport mode — skipping SSH worker spawn
```

(NOT `🚀 Using Parallel Dynamic Job Distribution Mode`, which is the legacy SSH path.)

**During run:**
- 26 PWC workers spawn on the rigs (8 per AMD rig + 2 on Zeus)
- No SSH "Authentication timeout" / "No existing session" errors
- Thresholds in run banner match config (W6_O64_evening_S3-37_FT0.68_RT0.70)

## Diff (what gets inserted)

**Edit 1 — function signature** (line ~816):
```diff
     output_holdout: str = 'holdout_history.json',
+    use_persistent_workers: bool = False,   # [S170-PARITY] use_persistent_workers
+    pwc_transport: str = 'tcp',             # [S170-PARITY] use_persistent_workers
 ) -> Dict[str, Any]:
```

**Edit 2 — function body** (after `add_window_optimizer_to_coordinator()`):
```diff
     # Add integration
     add_window_optimizer_to_coordinator()

+    # [S170-PARITY] propagate persistent worker / transport — match Bayesian path
+    # (lines 614-616). Without these, --config-file mode silently downgrades to
+    # legacy SSH distribution regardless of CLI flags.
+    coordinator.use_persistent_workers = use_persistent_workers
+    coordinator.pwc_transport          = pwc_transport
+
     # Create WindowConfig object
```

**Edit 3 — call site** (line ~1201):
```diff
         results = run_with_config(
             config_file=args.config_file,
             ...
-            output_holdout=args.output_holdout
+            output_holdout=args.output_holdout,
+            # [S170-PARITY] CLI passthrough — same defaults as Bayesian call site
+            use_persistent_workers=getattr(args, 'use_persistent_workers', False),
+            pwc_transport=getattr(args, 'pwc_transport', 'tcp'),
         )
```

## Implications beyond S170

Per TB: any prior use of `--config-file` mode for distributed runs may have been silently using the legacy SSH path. This includes any debugging runs, replay attempts, or validation experiments. Results from those runs should be reviewed with this knowledge.

## Secondary issue (per TB)

The run banner showed `FT=0.4_RT=0.45` instead of the config's `FT=0.68_RT=0.70`. After patch is applied and rerun launches successfully, verify the actual thresholds used match the config. This may be a separate display-vs-execution bug worth a follow-up proposal.
