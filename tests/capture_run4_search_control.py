#!/usr/bin/env python3
"""Capture the Run-4 routing SUCCESSOR control from untouched b11b6fd.

Beta X-MINER-START R1 Q6: a successor check needs an independently established control
captured from the tree BEFORE the miner seed-start change, under the new search
collaborator — never an oracle captured only from the candidate under review.

Run ONCE, in a checkout whose HEAD is b11b6fd and whose TRACKED files are unmodified.
The three fixture modules it imports (the frozen `run4_routing_clean_control`, the
successor harness `run4_routing_search_control`, and the search collaborator
`run4_search_fixture`) are placed there untracked, byte-identical to the ones the gate
uses; all three sha256s are recorded and the gate refuses if any differs.

Records: base commit, the committed WATCHER digest, the three fixture digests, the
unpinned params, the exact generated argv, and a completion sentinel.
"""
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

BASE_COMMIT = "b11b6fde7ab57321b42c418f87af8cc2c75c814d"
TARGET = "agents/watcher_agent.py"
ARTIFACT = os.path.join(HERE, "fixtures", "run4_search_control_b11b6fd.txt")


def sh(*a):
    return subprocess.run(a, cwd=ROOT, capture_output=True, text=True).stdout.strip()


def file_sha(rel):
    with open(os.path.join(ROOT, rel), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    head = sh("git", "rev-parse", "HEAD")
    dirty = sh("git", "status", "--porcelain", "--untracked-files=no")
    print(f"HEAD {head}; tracked changes {dirty!r}")
    if head != BASE_COMMIT:
        print(f"REFUSED: HEAD {head} != {BASE_COMMIT}")
        return 3
    if dirty:
        print("REFUSED: tracked files are modified — this would not be the base tree")
        return 3
    committed = subprocess.run(["git", "show", f"{BASE_COMMIT}:{TARGET}"], cwd=ROOT,
                               capture_output=True).stdout
    if hashlib.sha256(committed).hexdigest() != file_sha(TARGET):
        print("REFUSED: the WATCHER on disk is not the committed one")
        return 3

    from fixtures import run4_routing_clean_control as FX
    from fixtures import run4_routing_search_control as FXS
    from fixtures import run4_search_fixture as S
    import agents.watcher_agent as W

    argv = FXS.build_argv(W, FXS.CONTROL_PARAMS)
    art = {
        "artifact": "run4_search_control",
        "purpose": ("Run-4 routing successor control: the unpinned Step 1 argv on the "
                    "tree BEFORE the miner seed-start change, under the stage-1 search "
                    "collaborator (Beta X-MINER-START R1 Q6)"),
        "base_commit": BASE_COMMIT,
        "watcher_sha256_committed": hashlib.sha256(committed).hexdigest(),
        "frozen_fixture_sha256": FX.fixture_sha256(),
        "successor_module_sha256": FXS.module_sha256(),
        "search_fixture_sha256": file_sha("tests/fixtures/run4_search_fixture.py"),
        "search_range": list(S.RUN4_RANGE),
        "fixture_params_unpinned": FXS.CONTROL_PARAMS,
        "stage1_scheduling_changes": list(FXS.STAGE1_SCHEDULING_CHANGES),
        "argv_unpinned": argv,
        "completion_sentinel": "CAPTURE COMPLETE",
    }
    with open(ARTIFACT, "w") as fh:
        json.dump(art, fh, indent=1, sort_keys=True)
    print(f"CAPTURED {ARTIFACT}: {len(argv)} tokens")
    print(" ".join(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
