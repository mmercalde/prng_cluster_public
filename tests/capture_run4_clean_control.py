#!/usr/bin/env python3
"""STEP 0 -- capture the pre-edit clean control from untouched 69ca910.

TB Blocker 2 (preferred fix). Run ONCE, BEFORE the first edit to
agents/watcher_agent.py. Refuses to run if that file is modified.

Records: base commit, worktree digests, fixture inputs + fixture module hash,
the exact generated argv, the artifact sha256, and a completion sentinel.
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

from fixtures import run4_routing_clean_control as FX   # noqa: E402

ARTIFACT = os.path.join(HERE, "fixtures", "run4_clean_control_69ca910.txt")
TARGET = "agents/watcher_agent.py"


def sh(*a):
    return subprocess.run(a, cwd=ROOT, capture_output=True, text=True).stdout.strip()


def digest(rel):
    with open(os.path.join(ROOT, rel), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    head = sh("git", "rev-parse", "HEAD")
    dirty = sh("git", "status", "--porcelain", TARGET)

    print("=" * 70)
    print("STEP 0 -- PRE-EDIT CLEAN CONTROL CAPTURE")
    print("=" * 70)
    print(f"  HEAD                 : {head}")
    print(f"  {TARGET} status : {dirty!r}")

    if head != FX.BASE_COMMIT:
        print(f"\nREFUSED: HEAD {head} != fixture BASE_COMMIT {FX.BASE_COMMIT}")
        return 3
    if dirty:
        print(f"\nREFUSED: {TARGET} is MODIFIED. The clean-control window is closed.")
        print("An oracle captured from an edited tree is not a pre-patch oracle.")
        return 3

    committed = sh("git", "show", f"{FX.BASE_COMMIT}:{TARGET}")
    committed_sha = hashlib.sha256((committed + "\n").encode()).hexdigest()
    worktree_sha = digest(TARGET)
    print(f"  worktree digest      : {worktree_sha}")
    if worktree_sha != digest(TARGET):
        return 3

    sys.path.insert(0, os.path.join(ROOT, "agents"))
    import importlib
    agent_mod = importlib.import_module("agents.watcher_agent")

    print("\n  building UNPINNED control argv ...")
    argv_unpinned = FX.build_argv(agent_mod, FX.CONTROL_PARAMS)
    print(f"    {len(argv_unpinned)} tokens")

    print("  building SUPPLIED-SEVEN diagnostic argv (pre-patch: must be dropped) ...")
    supplied = dict(FX.CONTROL_PARAMS)
    supplied.update(FX.PIN_VALUES)
    argv_supplied = FX.build_argv(agent_mod, supplied)
    print(f"    {len(argv_supplied)} tokens")

    dead_chain_proven = (argv_unpinned == argv_supplied)
    warm_tokens = [t for t in argv_supplied if "warm-start" in t]
    print(f"\n  pre-patch dead chain proven (argv identical): {dead_chain_proven}")
    print(f"  warm-start tokens in supplied-seven argv    : {len(warm_tokens)} (expect 0)")
    if not dead_chain_proven or warm_tokens:
        print("\nREFUSED: pre-patch behavior is not what the brief describes.")
        return 3

    record = {
        "artifact": "run4_routing_clean_control",
        "purpose": ("pre-edit unpinned command oracle for G-UNPINNED-IDENTICAL; "
                    "TB_RULING_RUN4_ROUTING_PATCH_BRIEF_REVIEW.md Blocker 2"),
        "captured_before_any_edit": True,
        "base_commit": FX.BASE_COMMIT,
        "head_at_capture": head,
        "target_file": TARGET,
        "target_worktree_sha256": worktree_sha,
        "target_committed_sha256_note": committed_sha,
        "manifest_sha256": digest("agent_manifests/window_optimizer.json"),
        "fixture_module": "tests/fixtures/run4_routing_clean_control.py",
        "fixture_module_sha256": FX.fixture_sha256(),
        "fixture_params_unpinned": FX.CONTROL_PARAMS,
        "fixture_params_supplied_seven": supplied,
        "argv_unpinned": argv_unpinned,
        "argv_supplied_seven_DIAGNOSTIC": argv_supplied,
        "pre_patch_dead_chain_proven": dead_chain_proven,
        "stub_boundary": [
            "self._ensure_execution_set", "self._run_preflight_check",
            "self._run_step_streaming (interception point)",
            "self._run_post_step_cleanup", "self._find_results",
            "check_output_freshness", "p05_freeze_dataset",
            "p05_resolve_dataset_path", "database_system.DistributedPRNGDatabase",
            "miner.dataset_authority._active_execution_set",
            "miner.dataset_authority.load_provisioning_nodes",
            "miner.dataset_authority.fleet_preflight",
            "miner.dataset_authority.resolve_absent_fleet_status",
            "miner.dataset_authority.default_provisioning_manifest_path",
            "miner.dataset_authority.write_run_provenance",
        ],
        "not_stubbed": ["assert_seed_domain_preflight (pure arithmetic, real)"],
        "completion_sentinel": "CLEAN_CONTROL_CAPTURE_COMPLETE",
    }
    body = json.dumps(record, indent=2, sort_keys=True)
    payload_sha = hashlib.sha256(body.encode()).hexdigest()
    final = json.dumps({**record, "artifact_sha256_of_payload": payload_sha},
                       indent=2, sort_keys=True)
    with open(ARTIFACT, "w") as fh:
        fh.write(final + "\n")

    print("\n  argv_unpinned:")
    for t in argv_unpinned:
        print(f"      {t}")
    print(f"\n  artifact          : {os.path.relpath(ARTIFACT, ROOT)}")
    print(f"  payload sha256    : {payload_sha}")
    print(f"  artifact sha256   : {digest(os.path.relpath(ARTIFACT, ROOT))}")
    print("\nCLEAN_CONTROL_CAPTURE_COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
