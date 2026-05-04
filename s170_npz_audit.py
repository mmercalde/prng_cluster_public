#!/usr/bin/env python3
"""
S170 post-run NPZ + JSON integrity audit
=========================================

Runs after a Step 1 completion. Inspects both the JSON survivor file
and the NPZ binary, checks that per-survivor metadata fields are
populated (not zero-shells), and reports any meta.json / NPZ row
mismatches.

This is the diagnostic for the data-integrity concern raised in S170:
the April 18 bidirectional_survivors_binary.npz had all metadata
fields zero-filled despite carrying 20,916 seeds, plus a row count
mismatch with its meta.json.

USAGE:
    cd ~/distributed_prng_analysis
    python3 s170_npz_audit.py

    # or point at a specific JSON / NPZ pair:
    python3 s170_npz_audit.py --json bidirectional_survivors.json \
                              --npz  bidirectional_survivors_binary.npz \
                              --meta bidirectional_survivors_binary.meta.json

EXIT CODE:
    0 = JSON and NPZ both populated correctly
    1 = JSON survivors are missing required fields (writer-side bug upstream)
    2 = NPZ has zero-shell columns (writer-side bug in convert_survivors_to_binary.py)
    3 = meta.json row count disagrees with NPZ row count
    4 = file(s) missing
    5 = other / multiple integrity issues
"""

import argparse
import json
import sys
from pathlib import Path

REQUIRED_JSON_FIELDS = [
    "window_size", "offset", "skip_min", "skip_max",
    "trial_number", "forward_count", "reverse_count",
    "bidirectional_count",
]

REQUIRED_NPZ_FIELDS = [
    "window_size", "offset", "skip_min", "skip_max",
    "trial_number", "forward_count", "reverse_count",
    "bidirectional_count",
]


def audit_json(path):
    """Returns (status_str, issues_list, sample_record)."""
    if not path.exists():
        return "MISSING", [f"file does not exist: {path}"], None

    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        return "PARSE_FAIL", [f"json parse error: {e}"], None

    if isinstance(data, list):
        survivors = data
    elif isinstance(data, dict) and "survivors" in data:
        survivors = data["survivors"]
    else:
        return "SCHEMA_UNKNOWN", [f"unexpected top-level type: {type(data).__name__}"], None

    issues = []
    if len(survivors) == 0:
        return "EMPTY", ["survivor list is empty"], None

    sample = survivors[0]
    if not isinstance(sample, dict):
        return "SCHEMA_UNKNOWN", [f"first survivor is not a dict: {type(sample).__name__}"], sample

    # Check required fields are PRESENT
    missing = [f for f in REQUIRED_JSON_FIELDS if f not in sample]
    if missing:
        issues.append(f"first survivor missing fields: {missing}")

    # Spot-check: are these fields non-zero across the file?
    # (zeros could be valid for some fields, but window_size=0 is NEVER valid)
    zeros = []
    for fld in ["window_size", "skip_max"]:
        if fld in sample:
            try:
                vals = set(s.get(fld, "MISSING") for s in survivors[:1000])
                # if all zero, that's a problem
                if vals == {0}:
                    zeros.append(fld)
            except Exception:
                pass
    if zeros:
        issues.append(f"first 1000 survivors have all-zero values in: {zeros}")

    if issues:
        return "FAIL", issues, sample

    return "OK", [], sample


def audit_npz(npz_path, meta_path):
    """Returns (status_str, issues_list, summary_dict)."""
    if not npz_path.exists():
        return "MISSING", [f"NPZ does not exist: {npz_path}"], None

    try:
        import numpy as np
    except ImportError:
        return "NO_NUMPY", ["numpy not importable in this env"], None

    try:
        d = np.load(str(npz_path), allow_pickle=True)
    except Exception as e:
        return "LOAD_FAIL", [f"NPZ load error: {e}"], None

    issues = []
    summary = {"npz_files": list(d.files)}

    # Row count from seeds array
    if "seeds" in d.files:
        npz_rows = int(d["seeds"].shape[0])
    else:
        return "NO_SEEDS", ["NPZ has no 'seeds' array"], summary
    summary["npz_rows"] = npz_rows

    # Cross-check meta.json
    if meta_path.exists():
        try:
            meta = json.load(open(meta_path))
            meta_rows = meta.get("survivor_count")
            summary["meta_rows"] = meta_rows
            if meta_rows is not None and int(meta_rows) != npz_rows:
                issues.append(f"meta.json says {meta_rows} survivors but NPZ has {npz_rows} rows (MISMATCH)")
        except Exception as e:
            issues.append(f"could not read meta.json: {e}")
    else:
        summary["meta_rows"] = None
        issues.append(f"meta.json not found at {meta_path}")

    # Check each required NPZ field is present and not all zeros
    for fld in REQUIRED_NPZ_FIELDS:
        if fld not in d.files:
            issues.append(f"NPZ missing required field: {fld}")
            continue

        a = d[fld]
        try:
            uniq = np.unique(a)
        except Exception as e:
            issues.append(f"NPZ field {fld}: could not compute uniques ({e})")
            continue

        summary[f"{fld}_unique_count"] = int(len(uniq))
        summary[f"{fld}_first"] = a[0].item() if a.size > 0 else None

        # zero-shell detection: window_size, offset, skip_max all-zero is a bug
        if fld in ("window_size", "skip_max", "trial_number"):
            if len(uniq) == 1 and uniq[0] == 0:
                issues.append(f"NPZ field {fld}: all values are 0 (zero-shell, writer bug)")

    if issues:
        return "FAIL", issues, summary

    return "OK", [], summary


def main():
    ap = argparse.ArgumentParser(description="S170 post-run NPZ + JSON integrity audit")
    ap.add_argument("--json", default="bidirectional_survivors.json",
                    help="path to bidirectional_survivors.json (default: cwd)")
    ap.add_argument("--npz", default="bidirectional_survivors_binary.npz",
                    help="path to bidirectional_survivors_binary.npz")
    ap.add_argument("--meta", default="bidirectional_survivors_binary.meta.json",
                    help="path to bidirectional_survivors_binary.meta.json")
    args = ap.parse_args()

    json_path = Path(args.json)
    npz_path = Path(args.npz)
    meta_path = Path(args.meta)

    print("=" * 70)
    print("S170 NPZ + JSON Integrity Audit")
    print("=" * 70)
    print(f"JSON: {json_path}")
    print(f"NPZ:  {npz_path}")
    print(f"Meta: {meta_path}")
    print()

    # ---- JSON ----
    print("--- JSON survivor file ---")
    json_status, json_issues, sample = audit_json(json_path)
    print(f"Status: {json_status}")
    if sample is not None and isinstance(sample, dict):
        print(f"First survivor keys: {sorted(sample.keys())[:20]}")
        for fld in ("window_size", "offset", "skip_min", "skip_max",
                    "forward_match_rate", "reverse_match_rate", "score"):
            if fld in sample:
                print(f"  {fld:22s}: {sample[fld]}")
    for issue in json_issues:
        print(f"  ISSUE: {issue}")

    print()
    print("--- NPZ binary file ---")
    npz_status, npz_issues, summary = audit_npz(npz_path, meta_path)
    print(f"Status: {npz_status}")
    if summary:
        if "npz_rows" in summary:
            print(f"NPZ rows: {summary['npz_rows']}")
        if "meta_rows" in summary:
            print(f"Meta rows: {summary['meta_rows']}")
        if "npz_files" in summary:
            print(f"NPZ arrays: {summary['npz_files']}")
        for fld in REQUIRED_NPZ_FIELDS:
            uniq_key = f"{fld}_unique_count"
            first_key = f"{fld}_first"
            if uniq_key in summary:
                print(f"  {fld:22s}: {summary[uniq_key]} unique, first = {summary[first_key]}")
    for issue in npz_issues:
        print(f"  ISSUE: {issue}")

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    if json_status == "OK" and npz_status == "OK":
        print("PASS: JSON and NPZ both populated correctly. Pipeline data integrity OK.")
        sys.exit(0)
    if json_status == "MISSING" or npz_status == "MISSING":
        print("FAIL: required file(s) missing — Step 1 may not have completed.")
        sys.exit(4)
    if json_status == "FAIL" and npz_status == "FAIL":
        # Both bad — fundamental writer-side issue
        print("FAIL: BOTH JSON and NPZ have integrity issues (multiple bugs).")
        sys.exit(5)
    if json_status == "FAIL":
        print("FAIL: JSON survivors lack required fields. Bug is UPSTREAM of NPZ converter")
        print("      (in window_optimizer_integration_final.py or coordinator).")
        sys.exit(1)
    if npz_status == "FAIL":
        print("FAIL: NPZ has zero-shell or mismatch issues. Bug is in")
        print("      convert_survivors_to_binary.py (does not propagate JSON metadata).")
        sys.exit(2)

    print("UNKNOWN: see issues above.")
    sys.exit(5)


if __name__ == "__main__":
    main()
