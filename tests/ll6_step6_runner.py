#!/usr/bin/env python3
"""
tests/ll6_step6_runner.py - [LL-6 self-test helper] run Step 6's real CLI entry point,
prediction_generator.main(), once, in a fresh process, from a chosen file.

  --pg-file FILE    the prediction_generator.py to run (this tree's, a mutant copy, or the
                    baseline copy written by the test from `git show <full sha>:prediction_generator.py`)
  --code-dir DIR    optional; put DIR first on sys.path (a mutant code dir)
  --repo-root DIR   the tree under test; second on sys.path (every module LL-6 does not change)
  --models-dir, --survivors, --history   Step 6 inputs (fixture files under the test's --out)
  --cwd DIR         working directory; Step 6 writes predictions/ relative to it, so it lies
                    under the test's --out
  --out FILE        result JSON: exit code, the canonical prediction file, the log lines

CPU only: CUDA is hidden and the module's GPU_AVAILABLE is set to False before main() runs
(cupy imports on this box even with no visible device). The log format drops the wall-clock
field so two runs can be compared line by line. The capture handler is installed again after the
module import, because an import Step 6 makes resets the root logger to WARNING.
"""
import argparse
import importlib.util
import io
import json
import logging
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg-file", required=True)
    ap.add_argument("--code-dir", default=None)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--survivors", required=True)
    ap.add_argument("--history", required=True)
    ap.add_argument("--cwd", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.repo_root)
    if a.code_dir:
        sys.path.insert(0, a.code_dir)
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
    root = logging.getLogger()
    root.handlers = [h]                       # basicConfig in setup_logging is then a no-op
    root.setLevel(logging.INFO)
    if a.code_dir:
        # prediction_generator puts its own directory first on sys.path when it runs
        # (prediction_generator.py:57); import the mutant's models package first so it is used.
        import models.inference_bundle  # noqa: F401
    spec = importlib.util.spec_from_file_location("prediction_generator", a.pg_file)
    pg = importlib.util.module_from_spec(spec)
    sys.modules["prediction_generator"] = pg
    spec.loader.exec_module(pg)
    # An import made by Step 6 resets the root logger (found after Beta X-LL6-BUNDLE R1: level
    # WARNING after import, so nothing at INFO was captured and the log comparison was vacuous).
    # Re-install the capture after the import, on both sides alike, and record what was found.
    after_import = {"root_level": root.level, "capture_handler_kept": root.handlers == [h]}
    root.handlers = [h]
    root.setLevel(logging.INFO)
    pg.GPU_AVAILABLE = False
    os.makedirs(a.cwd, exist_ok=True)
    os.chdir(a.cwd)
    sys.argv = ["prediction_generator.py", "--survivors-forward", a.survivors,
                "--lottery-history", a.history, "--models-dir", a.models_dir,
                "--config", os.path.join(a.cwd, "no_config.json")]
    rc = pg.main()
    canon = os.path.join(a.cwd, "predictions", "next_draw_prediction.json")
    result = json.load(open(canon)) if os.path.exists(canon) else None
    json.dump({"rc": rc, "pg_file": a.pg_file, "result": result, "logging_after_import": after_import,
               "log_lines": buf.getvalue().splitlines()}, open(a.out, "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
