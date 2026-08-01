#!/usr/bin/env python3
"""[Phase 6-P0] Read-only verifier for a published dataset version + pointer manifest.

Re-derives the published version file's sha256 and confirms it agrees with BOTH the
pointer manifest and the compatibility alias (`daily3.json`).

Scoping §8.1.6 is the reason this exists: "without it P0 has published something
nothing has ever checked."  This is P0's clean control.

READ-ONLY BY CONSTRUCTION.  It verifies; it never repairs, never writes, never
renames and never deletes.  Every file is opened in a read mode.  A failure is
reported, not corrected.

No runtime consumer: nothing in the pipeline imports this module.  It is a hand-run
audit tool, the same shape as scripts/extract_search_bounds_snapshot.py.

Contract verified: docs/DATASET_PUBLICATION_SCHEMA_v1.md (manifest_schema_version 1).

Exit status:  0 = PASS,  1 = FAIL,  2 = UNAVAILABLE (an input could not be read at all).

Usage:
    python3 scripts/verify_dataset_publication.py
    python3 scripts/verify_dataset_publication.py --manifest /path/to/daily3_current.json
    python3 scripts/verify_dataset_publication.py --manifest M --alias A --quiet
"""

import argparse
import hashlib
import json
import os
import re
import sys

# Anchored on __file__, never os.getcwd() — scoping §2.3 records __file__/absolute
# anchoring as the non-CWD-hazardous pattern, and §2.1 shows what CWD anchoring costs.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MANIFEST = os.path.join(_REPO_ROOT, "daily3_current.json")
DEFAULT_ALIAS = os.path.join(_REPO_ROOT, "daily3.json")

SUPPORTED_SCHEMA_VERSIONS = (1,)

REQUIRED_FIELDS = (
    "manifest_schema_version",
    "version_id",
    "filename",
    "sha256",
    "size_bytes",
    "record_count",
    "first_draw",
    "last_draw",
    "published_utc",
    "dataset_lineage_id",
    "predecessor_sha256",
    "notes",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^daily3-(\d{8}T\d{12})Z-([0-9a-f]{12})$")

_READ_CHUNK = 1024 * 1024


def _sha256_file(path):
    """Streaming digest.  Read-only: the handle is opened 'rb' and never written."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_READ_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class _Report(object):
    """Accumulates check results.  Nothing here mutates the filesystem."""

    def __init__(self, quiet=False):
        self.rows = []
        self.quiet = quiet

    def check(self, cid, description, ok, detail=""):
        self.rows.append((cid, description, bool(ok), detail))
        if not self.quiet:
            mark = "PASS" if ok else "FAIL"
            line = "  [%s] %-6s %s" % (cid, mark, description)
            if detail:
                line += "\n           %s" % detail
            print(line, flush=True)
        return bool(ok)

    @property
    def failed(self):
        return [r for r in self.rows if not r[2]]


def verify(manifest_path, alias_path, quiet=False):
    """Returns (terminal_state, report).  terminal_state in PASS/FAIL/UNAVAILABLE."""
    rep = _Report(quiet=quiet)

    if not quiet:
        print("Phase 6-P0 dataset publication verifier (read-only)", flush=True)
        print("  manifest: %s" % manifest_path, flush=True)
        print("  alias:    %s" % alias_path, flush=True)
        print("", flush=True)

    # ---- the manifest itself ------------------------------------------------
    if not os.path.isfile(manifest_path):
        rep.check("M0", "pointer manifest exists", False,
                  "not found: %s" % manifest_path)
        return "UNAVAILABLE", rep
    rep.check("M0", "pointer manifest exists", True)

    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except Exception as exc:  # noqa: BLE001 - any parse failure is a verifier FAIL
        rep.check("M1", "pointer manifest is valid JSON", False, "%s: %s"
                  % (type(exc).__name__, exc))
        return "FAIL", rep
    rep.check("M1", "pointer manifest is valid JSON", True)

    if not isinstance(manifest, dict):
        rep.check("M2", "pointer manifest is a JSON object", False,
                  "top level is %s" % type(manifest).__name__)
        return "FAIL", rep
    rep.check("M2", "pointer manifest is a JSON object", True)

    missing = [f for f in REQUIRED_FIELDS if f not in manifest]
    if not rep.check("M3", "all required fields present", not missing,
                     "missing: %s" % ", ".join(missing) if missing else ""):
        return "FAIL", rep

    schema_version = manifest["manifest_schema_version"]
    rep.check("M4", "manifest_schema_version is supported",
              schema_version in SUPPORTED_SCHEMA_VERSIONS,
              "got %r, supported %r" % (schema_version, list(SUPPORTED_SCHEMA_VERSIONS)))

    version_id = manifest["version_id"]
    filename = manifest["filename"]
    declared_sha = manifest["sha256"]
    declared_size = manifest["size_bytes"]
    declared_records = manifest["record_count"]

    rep.check("M5", "sha256 field is a 64-hex digest",
              isinstance(declared_sha, str) and bool(_SHA256_RE.match(declared_sha)),
              "got %r" % (declared_sha,))

    pred = manifest["predecessor_sha256"]
    rep.check("M6", "predecessor_sha256 is null or a 64-hex digest",
              pred is None or (isinstance(pred, str) and bool(_SHA256_RE.match(pred))),
              "got %r" % (pred,))

    # ---- the filename grammar ----------------------------------------------
    rep.check("F1", "filename == version_id + '.json'",
              filename == version_id + ".json",
              "filename=%r version_id=%r" % (filename, version_id))

    m = _VERSION_RE.match(version_id)
    rep.check("F2", "version_id matches daily3-<UTC>Z-<sha256[:12]>", bool(m),
              "got %r" % (version_id,))

    if m and isinstance(declared_sha, str):
        embedded = m.group(2)
        rep.check("F3", "digest embedded in the filename matches the manifest sha256",
                  embedded == declared_sha[:12],
                  "filename carries %r, manifest sha256[:12] is %r"
                  % (embedded, declared_sha[:12]))

    # ---- the version file ---------------------------------------------------
    # Resolved relative to the MANIFEST's own directory, so a clone or a scratch
    # copy verifies without edit.  Never relative to os.getcwd().
    version_path = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), filename)

    if not os.path.isfile(version_path):
        rep.check("V0", "version file exists", False, "not found: %s" % version_path)
        return "FAIL", rep
    rep.check("V0", "version file exists", True, version_path)

    actual_size = os.path.getsize(version_path)
    rep.check("V1", "version file size matches size_bytes",
              actual_size == declared_size,
              "on disk %s, manifest %s" % (actual_size, declared_size))

    actual_sha = _sha256_file(version_path)
    rep.check("V2", "re-derived version-file sha256 matches the manifest",
              actual_sha == declared_sha,
              "re-derived %s\n           manifest   %s" % (actual_sha, declared_sha))

    try:
        with open(version_path, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        parsed = True
    except Exception as exc:  # noqa: BLE001
        rep.check("V3", "version file is valid JSON", False,
                  "%s: %s" % (type(exc).__name__, exc))
        parsed = False
        records = None
    if parsed:
        rep.check("V3", "version file is valid JSON", True)
        is_list = isinstance(records, list)
        rep.check("V4", "version file top level is a JSON array", is_list,
                  "" if is_list else "top level is %s" % type(records).__name__)
        if is_list:
            rep.check("V5", "record_count matches the manifest",
                      len(records) == declared_records,
                      "counted %s, manifest %s" % (len(records), declared_records))
            if records:
                first = records[0]
                last = records[-1]
                want_first = manifest["first_draw"]
                want_last = manifest["last_draw"]
                rep.check(
                    "V6", "first record matches first_draw",
                    isinstance(first, dict)
                    and isinstance(want_first, dict)
                    and first.get("date") == want_first.get("date")
                    and first.get("session") == want_first.get("session"),
                    "file %r vs manifest %r"
                    % ({"date": (first or {}).get("date") if isinstance(first, dict) else first,
                        "session": (first or {}).get("session") if isinstance(first, dict) else None},
                       want_first))
                rep.check(
                    "V7", "last record matches last_draw",
                    isinstance(last, dict)
                    and isinstance(want_last, dict)
                    and last.get("date") == want_last.get("date")
                    and last.get("session") == want_last.get("session"),
                    "file %r vs manifest %r"
                    % ({"date": (last or {}).get("date") if isinstance(last, dict) else last,
                        "session": (last or {}).get("session") if isinstance(last, dict) else None},
                       want_last))

    # ---- the compatibility alias -------------------------------------------
    # The instruction's core requirement: the digest must agree with BOTH the
    # manifest and the alias.  The alias is what all 17 existing consumers open.
    if not os.path.isfile(alias_path):
        rep.check("A0", "compatibility alias exists", False, "not found: %s" % alias_path)
        return "FAIL", rep
    rep.check("A0", "compatibility alias exists", True,
              "%s%s" % (alias_path,
                        " -> %s" % os.readlink(alias_path)
                        if os.path.islink(alias_path) else " (regular file)"))

    alias_sha = _sha256_file(alias_path)
    rep.check("A1", "alias sha256 matches the published version",
              alias_sha == declared_sha,
              "alias    %s\n           manifest %s" % (alias_sha, declared_sha))

    state = "FAIL" if rep.failed else "PASS"
    return state, rep


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Read-only verifier for the Phase 6-P0 dataset publication.")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="pointer manifest path (default: <repo>/daily3_current.json)")
    ap.add_argument("--alias", default=DEFAULT_ALIAS,
                    help="compatibility alias path (default: <repo>/daily3.json)")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress per-check output; print the sentinel only")
    args = ap.parse_args(argv)

    state, rep = verify(args.manifest, args.alias, quiet=args.quiet)

    total = len(rep.rows)
    failed = len(rep.failed)
    print("", flush=True)
    print("DATASET_PUBLICATION_VERIFIER — SENTINEL", flush=True)
    print("  checks run:    %d" % total, flush=True)
    print("  checks failed: %d" % failed, flush=True)
    if failed:
        for cid, desc, _ok, detail in rep.failed:
            print("    - [%s] %s" % (cid, desc), flush=True)
    print("  TERMINAL STATE: %s" % state, flush=True)

    return {"PASS": 0, "FAIL": 1, "UNAVAILABLE": 2}[state]


if __name__ == "__main__":
    sys.exit(main())
