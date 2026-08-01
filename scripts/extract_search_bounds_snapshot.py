#!/usr/bin/env python3
"""
extract_search_bounds_snapshot.py — programmatic bounds snapshot for Chapter 1.

WHY THIS EXISTS
---------------
Every numeric search bound in `docs/CHAPTER_1_WINDOW_OPTIMIZER.md` was wrong
(audit `db9782a`, §6 P0.2): documented thresholds `[0.15, 0.60] default 0.25`
against live `[0.30, 0.75] default 0.30`, window ceiling 10x too large, skip
ceiling 2x too large. Hand-copied numbers drift silently. This tool emits the
chapter's bounds block from the LIVE authority so the only thing that can go
stale is the snapshot header, not the values.

A DATE IS NOT SUFFICIENT PROVENANCE. Multiple code states share a date, so the
snapshot carries `repository_commit` (which tree) and `configuration_digest`
(which configuration bytes) alongside `generated_at`.

The snapshot is INFORMATIVE, NOT AUTHORITATIVE. The authority is
`distributed_config.json -> search_bounds`, merged over the code defaults in
`window_optimizer.load_search_bounds_from_config()`. Config wins
(`window_optimizer.py:57-61`).

Usage:
    python3 scripts/extract_search_bounds_snapshot.py            # markdown block
    python3 scripts/extract_search_bounds_snapshot.py --json     # machine-readable
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_ROOT, "distributed_config.json")

# The bound families the chapter documents. Window and skip get the same
# treatment as thresholds — audit §6 P0.2 is explicit that all of them were wrong.
BOUND_KEYS = (
    "window_size",
    "offset",
    "skip_min",
    "skip_max",
    "forward_threshold",
    "reverse_threshold",
)

# Provenance notes carried in distributed_config.json. These are the ONLY in-repo
# record of *why* the window floor is 6, so the snapshot must carry them forward
# rather than reducing the config to bare numbers.
NOTE_PREFIX = "_"


def repository_commit():
    """Full SHA of the tree the snapshot was taken from, plus dirty marker."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, OSError):
        return "UNAVAILABLE (not a git checkout)"
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "distributed_config.json"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, OSError):
        dirty = ""
    return sha + ("  (distributed_config.json MODIFIED in working tree)" if dirty else "")


def load_authority():
    """Read the live search_bounds block. Fail closed — never fabricate bounds."""
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    bounds = cfg.get("search_bounds")
    if not bounds:
        raise SystemExit(
            "SEARCH_BOUNDS_AUTHORITY_MISSING: distributed_config.json carries no "
            "'search_bounds' block; refusing to emit a snapshot of nothing."
        )
    return bounds


def configuration_digest(bounds):
    """
    sha256 over the canonicalised search_bounds block.

    Two trees that share a date but differ in configuration produce different
    digests — which is exactly the discrimination a date alone cannot provide.
    """
    canonical = json.dumps(bounds, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def effective_bounds():
    """
    Resolve the bounds the optimizer actually uses, through the live loader —
    i.e. code defaults merged under config override. Imported, not re-implemented,
    so this cannot drift from the merge rule at window_optimizer.py:57-61.
    """
    sys.path.insert(0, REPO_ROOT)
    from window_optimizer import load_search_bounds_from_config  # noqa: E402
    return load_search_bounds_from_config(CONFIG_PATH)


def collect():
    bounds = load_authority()
    effective = effective_bounds()
    notes = {}
    for key in BOUND_KEYS:
        for nk, nv in (bounds.get(key) or {}).items():
            if nk.startswith(NOTE_PREFIX):
                notes.setdefault(key, {})[nk] = nv
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repository_commit": repository_commit(),
        "configuration_digest": configuration_digest(bounds),
        "authority": "distributed_config.json -> search_bounds "
                     "(merged over code defaults by window_optimizer."
                     "load_search_bounds_from_config; config wins, "
                     "window_optimizer.py:57-61)",
        "status": "INFORMATIVE SNAPSHOT — NOT AUTHORITATIVE. "
                  "Read the authority above for the binding values.",
        "effective_bounds": {k: effective[k] for k in BOUND_KEYS if k in effective},
        "provenance_notes": notes,
    }


def render_markdown(snap):
    out = []
    out.append("```")
    out.append("Authority:")
    out.append("  " + snap["authority"])
    out.append("")
    out.append("Snapshot:")
    out.append(f"  generated_at         : {snap['generated_at']}")
    out.append(f"  repository_commit    : {snap['repository_commit']}")
    out.append(f"  configuration_digest : {snap['configuration_digest']}")
    out.append(f"  status               : {snap['status']}")
    out.append("")
    out.append("  extracted bounds:")
    for key in BOUND_KEYS:
        vals = snap["effective_bounds"].get(key)
        if vals is None:
            continue
        pairs = ", ".join(
            f"{k}={v}" for k, v in vals.items() if not k.startswith(NOTE_PREFIX)
        )
        out.append(f"    {key:<18} {pairs}")
    out.append("```")
    if snap["provenance_notes"]:
        out.append("")
        out.append("**Provenance notes carried from `distributed_config.json`** "
                   "(the only in-repo record of *why* these values are what they are):")
        out.append("")
        for key, notes in snap["provenance_notes"].items():
            for nk, nv in notes.items():
                out.append(f"- `{key}.{nk}` — {nv}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--json", action="store_true",
                    help="emit the machine-readable snapshot instead of markdown")
    args = ap.parse_args()
    snap = collect()
    print(json.dumps(snap, indent=2) if args.json else render_markdown(snap))


if __name__ == "__main__":
    main()
