#!/usr/bin/env python3
"""provision_dataset_fleet.py — S172 Phase 6-P0.5 fleet dataset provisioning.

Provisioning is an **explicit step in rig bring-up, not a side effect of
cloning** (`docs/RUNTIME_DATASET_PROVISIONING_CONTRACT.md` §4). `daily3.json` is
Git-ignored, so `git clone` is not a complete rig deployment — Phase 6.0
discovered that on CT100, and repeated manual `scp` is not a provisioning
contract. This is the tool that replaces the remembered command.

WHAT IT DOES
    1. resolves the pointer manifest `daily3_current.json` and freezes the
       identity (version, absolute path, sha256, size, record count);
    2. copies that immutable version to every node in the provisioning manifest,
       at the same absolute path the coordinator dispatches;
    3. re-derives the digest **on each target node** and compares it with the
       frozen value;
    4. writes a run-provenance record and terminates in an explicit
       PASS / FAIL / UNAVAILABLE / INCOMPLETE per node (VIR-3).

EVERY NODE IS PROVISIONED, INCLUDING ONES THAT ALREADY LOOK CORRECT.
    `--verify-only` exists for auditing, but the default provisions
    unconditionally. A provisioning step that skips a node it believes is already
    correct is a provisioning step that cannot detect the case it exists for. On
    2026-08-01 `rrig6600` held a hand-placed Phase-6.0 copy of the alias whose
    digest happened to match the published version; nothing had verified it, and
    it was correct by the accuracy of a manual copy rather than by any mechanism.
    That is exactly what this tool replaces.

IT NEVER PUBLISHES. It copies an already-published immutable version. It does not
create versions, move the pointer, or touch `daily3.json` on the source.

Usage (from VM 101 — the source of truth):
    cd ~/distributed_prng_analysis
    source ~/venvs/torch/bin/activate
    PYTHONPATH=. python3 scripts/provision_dataset_fleet.py            # provision + verify
    PYTHONPATH=. python3 scripts/provision_dataset_fleet.py --verify-only
    PYTHONPATH=. python3 scripts/provision_dataset_fleet.py --node rrig6600b

Exit 0 = every node PASS. Exit 1 = any node not PASS (fail closed).
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from miner import dataset_authority as D                             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=None,
                    help=f"provisioning manifest (default: <repo>/{D.PROVISIONING_MANIFEST_NAME})")
    ap.add_argument("--pointer", default=None,
                    help=f"pointer manifest (default: <repo>/{D.POINTER_MANIFEST_NAME})")
    ap.add_argument("--node", action="append", default=None,
                    help="restrict to this node_id (repeatable)")
    ap.add_argument("--verify-only", action="store_true",
                    help="verify on each target without transferring")
    ap.add_argument("--run-label", default="provision_dataset_fleet")
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    pointer = args.pointer or os.path.join(_ROOT, D.POINTER_MANIFEST_NAME)

    print("=" * 74)
    print("S172 Phase 6-P0.5 — fleet dataset provisioning")
    print("=" * 74)

    try:
        frozen = D.resolve_pointer(pointer)
    except D.DatasetAuthorityError as exc:
        print(f"\n❌ POINTER RESOLUTION FAILED\n   {exc}")
        print("\nRESULT: FAIL")
        return 1

    print(f"\nfrozen dataset : {frozen.describe()}")
    print(f"version_id     : {frozen.version_id}")
    print(f"lineage        : {frozen.dataset_lineage_id}")
    print(f"pointer sha256 : {frozen.manifest_sha256}")

    nodes = D.load_provisioning_nodes(args.manifest)
    if nodes is None:
        print(f"\n❌ no provisioning manifest at "
              f"{args.manifest or D.default_provisioning_manifest_path()}")
        print("   UNAVAILABLE — not clean (VIR-5). Nothing was provisioned.")
        print("\nRESULT: UNAVAILABLE")
        return 1
    if not nodes:
        print("\n❌ provisioning manifest declares no nodes for 'daily3'.")
        print("\nRESULT: INCOMPLETE")
        return 1
    if args.node:
        wanted = set(args.node)
        nodes = [n for n in nodes if n.node_id in wanted]
        if not nodes:
            print(f"\n❌ no manifest node matched {sorted(wanted)}")
            print("\nRESULT: INCOMPLETE")
            return 1

    verb = "verifying" if args.verify_only else "provisioning"
    print(f"\n{verb} {len(nodes)} node(s): "
          f"{', '.join(f'{n.node_id} ({n.ssh_address})' for n in nodes)}\n")

    records = []
    for node in nodes:
        print(f"── {node.node_id} ({node.ssh_address}) "
              + "─" * max(0, 48 - len(node.node_id) - len(node.ssh_address)))
        if args.verify_only:
            rec = D.verify_node_dataset(frozen, node, timeout=args.timeout)
        else:
            rec = D.provision_node_dataset(frozen, node, timeout=args.timeout)
        records.append(rec)
        icon = {"PASS": "✅", "FAIL": "❌",
                "UNAVAILABLE": "⚠️ ", "INCOMPLETE": "⚠️ "}.get(rec.status, "?")
        print(f"   {icon} {rec.status}: {rec.message}")
        if rec.digest:
            print(f"      digest on target : {rec.digest}")
            print(f"      expected         : {rec.expected_digest}")
        print()

    prov_path = D.write_run_provenance(
        args.run_label, frozen, records,
        fleet_status=("PASS" if all(r.status == "PASS" for r in records)
                      else "FAIL"),
        extra={"tool": "scripts/provision_dataset_fleet.py",
               "verify_only": bool(args.verify_only)},
    )
    print(f"run provenance : {prov_path}")

    print("=" * 74)
    for r in records:
        print(f"  {r.status:12} {r.node_id:12} {r.ssh_address}")
    print("=" * 74)

    failed = [r for r in records if r.status != "PASS"]
    if failed:
        print(f"\n{len(failed)} of {len(records)} node(s) did not pass. "
              f"A run must not start with a partial fleet (contract §3).")
        print("\nRESULT: FAIL")
        return 1
    print(f"\nAll {len(records)} node(s) verified ON TARGET against the frozen "
          f"identity.")
    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
