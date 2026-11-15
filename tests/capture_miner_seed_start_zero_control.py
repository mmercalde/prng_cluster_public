#!/usr/bin/env python3
"""Re-capture the R7 zero-start control on untouched b11b6fd with the NUDGER-FREE driver.

Why a re-capture. G-R7-ZERO (tests/test_s172_miner_seed_start.py) refuses unless the
driver's sha256 equals the one recorded in the control. The L-4 repair removed the pump
nudger from tests/fixtures/miner_seed_start_driver.py, so the original capture (taken
with the nudger-bearing driver, sha 2d264a66...) no longer binds.

Why a liveness aid HERE ONLY. b11b6fd has L-4: its A9 shape (4 workers) stalls without a
pump timer. This script therefore runs the base coordinator's OWN `_pump_deferred` on a
0.5 s timer — exactly the removed nudger — during the capture, and nowhere else. It
changes when deferred frames resume, never what is assigned or recorded.

Why the content is unchanged. The script also loads the ORIGINAL control (the version
committed in the base tree's checkout of this file is irrelevant; it reads the original
artifact from `--original`) and refuses to write unless every shape's digest — over
stripes, StripeAssign payloads and evidence — equals the original capture's. A changed
driver that altered any compared field would therefore refuse here.

Usage (in a checkout whose HEAD is b11b6fd, tracked files unmodified, the nudger-free
driver placed at tests/fixtures/miner_seed_start_driver.py untracked):
  python3 tests/capture_miner_seed_start_zero_control.py --original <old control> --out <new>
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
BASE = "b11b6fde7ab57321b42c418f87af8cc2c75c814d"


def sh(*a):
    return subprocess.run(a, cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    head = sh("git", "rev-parse", "HEAD")
    dirty = sh("git", "status", "--porcelain", "--untracked-files=no")
    if head != BASE or dirty:
        print(f"REFUSED: HEAD {head} tracked-dirty {dirty!r}")
        return 3
    orig = json.load(open(a.original))
    if orig.get("head_at_capture") != BASE:
        print("REFUSED: original control is not from b11b6fd")
        return 3
    from fixtures import miner_seed_start_driver as D
    if hasattr(D, "pump_nudger"):
        print("REFUSED: the driver still carries the nudger")
        return 3
    import logging
    logging.disable(logging.CRITICAL)
    import miner.range_miner_coordinator as C
    import window_optimizer_integration_final as W

    # ---- capture-only liveness aid (b11b6fd has L-4) -------------------------
    made, stop = [], threading.Event()
    real = C.build_coordinator

    def capture(**kw):
        c = real(**kw)
        made.append(c)
        return c

    def loop():
        while not stop.wait(0.5):
            for c in list(made):
                try:
                    c._pump_deferred()
                except Exception:
                    pass
    C.build_coordinator = capture
    th = threading.Thread(target=loop, daemon=True)
    th.start()

    scratch = os.path.join(ROOT, "data", "miner_seed_start_recapture")
    os.makedirs(scratch, exist_ok=True)
    shapes = {}
    try:
        for name, shape in orig["shapes"].items():
            d = tempfile.mkdtemp(prefix=f"r7{name}_", dir=scratch)
            try:
                r = D.run_shape(d, **shape["params"])          # seed_start OMITTED
                if not r["committed"]:
                    print(f"REFUSED: {name} did not commit: {r['state']}")
                    return 4
                coord = __import__("types").SimpleNamespace(
                    staging_dir=os.path.dirname(r["ledger_path"]), miner_output_dir=None)
                ev = W._step1_miner_sweep_evidence(
                    {"run_id": r["run_id"], "committed": r["committed"]}, coord)
                ev = D._normalize(ev, [(r["run_id"], "<RUN>")])
            finally:
                shutil.rmtree(d, ignore_errors=True)
            body = {"stripes": r["stripes"], "assigns": r["assigns"], "evidence": ev}
            dg = hashlib.sha256(D.canonical(body).encode()).hexdigest()
            if dg != shape["digest"]:
                print(f"REFUSED: {name} digest {dg} != original {shape['digest']}")
                return 5
            shapes[name] = {"params": shape["params"], "digest": dg, **body,
                            "committed": r["committed"], "state": r["state"],
                            "sink": r["sink"]}
            print(f"{name}: digest {dg[:12]} == original", flush=True)
    finally:
        stop.set()
        C.build_coordinator = real
    coord_bytes = subprocess.run(["git", "show", f"{BASE}:miner/range_miner_coordinator.py"],
                                 cwd=ROOT, capture_output=True).stdout
    art = {"purpose": ("R7 zero-start control, RE-CAPTURED on b11b6fd with the nudger-free "
                       "driver (L-4 repair); content digests equal the original capture"),
           "head_at_capture": head, "tracked_tree_clean": True,
           "coordinator_sha256": hashlib.sha256(coord_bytes).hexdigest(),
           "driver_sha256": D.driver_sha256(),
           "original_driver_sha256": orig["driver_sha256"],
           "capture_liveness_aid": ("b11b6fd has L-4: the base coordinator's own "
                                    "_pump_deferred on a 0.5 s timer during THIS capture "
                                    "only; the gate runs with none"),
           "shapes": shapes, "completion_sentinel": "CAPTURE COMPLETE"}
    with open(a.out, "w") as f:
        json.dump(art, f, indent=1, sort_keys=True)
    print("CAPTURED", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
