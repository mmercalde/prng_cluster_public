#!/usr/bin/env python3
"""
models/feature_manifest.py - [LL-1, Ch14 learning loop, 2026-11-23]

One feature manifest: every Step 3 feature column labelled by ORIGIN group,
with named schema versions, and the checks Step 5 and Step 6 use.

Authority: models/feature_manifest.json (this module only reads it).
The JSON was built once by build_manifest() from the live Step 3 output of
run H (91 names) plus the code constants in survivor_scorer.py and
models/global_state_tracker.py; tests/test_ll_feature_settings.py rebuilds it
and checks it is unchanged, so neither side can drift alone.

Origin groups (system map v1.2 §7.1, §8):
  npz_metadata    13  merged from the NPZ by survivor_scorer.METADATA_MERGE_FIELDS
  dead             5  merge names with no producer (always 0.0)
  seed_history    34  computed in Step 3 from the regenerated sequence vs train history
  battery         23  batt_* (S113), computed from the regenerated sequence
  global          14  global_* from GlobalStateTracker(train_history), one value per run
  excluded         2  score, confidence (in the feature dict, never trained on;
                      also computed, so the map's '36 computed' = 34 + these 2)
  per_seed_sieve   2  forward_matches, reverse_matches (schema v2 only, LL-2)
Labels (holdout_hits, holdout_quality) sit OUTSIDE the feature dict.

Variance class (per-seed / per-trial / per-run constant) is a property of a
run, not of a name, and is MEASURED per run (variance_report), never typed in.

No torch / numpy import: safe to import from anything.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

MANIFEST_PATH = Path(__file__).with_name("feature_manifest.json")
MANIFEST_FORMAT = "tfm_feature_manifest/1"

GROUPS = ("npz_metadata", "dead", "seed_history", "battery", "global",
          "excluded", "per_seed_sieve")
TRAINABLE_GROUPS = ("npz_metadata", "dead", "seed_history", "battery", "global",
                    "per_seed_sieve")
SCHEMA_CHECK_MODES = ("off", "warn", "fail")

# Step 5's exclude list before LL-3 (meta_prediction_optimizer_anti_overfit.py
# _load_data). Kept here as the declared default; the test checks Step 5 still
# uses exactly this list when no LL-3 setting is given.
STEP5_DEFAULT_EXCLUDE = ("score", "confidence", "holdout_hits", "holdout_quality")


class ManifestError(ValueError):
    pass


def schema_hash(names: Iterable[str]) -> str:
    """The hash Step 5 writes and models.feature_schema.validate_feature_schema_hash
    checks: sha256 of the sorted names joined by ',', first 16 hex."""
    return hashlib.sha256(",".join(sorted(names)).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# build (run once, and by the test to prove the JSON is unchanged)
# ---------------------------------------------------------------------------
def classify(name: str, merge_fields: Sequence[str], dead_fields: Sequence[str],
             global_names: Sequence[str], sieve_fields: Sequence[str]) -> str:
    if name in ("score", "confidence"):
        return "excluded"
    if name in sieve_fields:
        return "per_seed_sieve"
    if name in dead_fields:
        return "dead"
    if name in merge_fields:
        return "npz_metadata"
    if name.startswith("global_"):
        if name[len("global_"):] not in global_names:
            raise ManifestError(f"{name}: global_ prefix but not a GlobalStateTracker name")
        return "global"
    if name.startswith("batt_"):
        return "battery"
    return "seed_history"


def build_manifest(v1_feature_names: Sequence[str], source: dict) -> dict:
    """Build the manifest from the live v1 name list and the code constants."""
    from survivor_scorer import (METADATA_MERGE_FIELDS, DEAD_PLACEHOLDER_FIELDS,
                                 PER_SEED_SIEVE_FIELDS)
    from models.global_state_tracker import GLOBAL_FEATURE_NAMES
    v1 = sorted(v1_feature_names)
    if len(set(v1)) != len(v1):
        raise ManifestError("duplicate names in the v1 list")
    if set(v1) & set(PER_SEED_SIEVE_FIELDS):
        raise ManifestError("v1 must not already contain the per-seed sieve fields")
    names = {}
    for n in v1 + list(PER_SEED_SIEVE_FIELDS):
        names[n] = classify(n, METADATA_MERGE_FIELDS, DEAD_PLACEHOLDER_FIELDS,
                            GLOBAL_FEATURE_NAMES, PER_SEED_SIEVE_FIELDS)
    v2 = sorted(v1 + list(PER_SEED_SIEVE_FIELDS))

    def trained(lst):
        return [n for n in lst if n not in STEP5_DEFAULT_EXCLUDE]

    return {
        "format": MANIFEST_FORMAT,
        "source": source,
        "groups": {g: sorted(n for n, gg in names.items() if gg == g) for g in GROUPS},
        "aliases": {"survivor_local": ["npz_metadata", "seed_history", "battery",
                                       "per_seed_sieve"],
                    "run_context": ["global"]},
        "labels_outside_features": ["holdout_hits", "holdout_quality"],
        "versions": {
            "v1": {"extracted": v1, "extracted_count": len(v1),
                   "trained_by_default": trained(v1),
                   "trained_by_default_count": len(trained(v1)),
                   "trained_by_default_hash": schema_hash(trained(v1)),
                   "settings": {"per_seed_sieve_features": "off"}},
            "v2": {"extracted": v2, "extracted_count": len(v2),
                   "trained_by_default": trained(v2),
                   "trained_by_default_count": len(trained(v2)),
                   "trained_by_default_hash": schema_hash(trained(v2)),
                   "settings": {"per_seed_sieve_features": "on"}},
        },
    }


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------
def manifest_sha256(path: Path = MANIFEST_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    m = json.loads(Path(path).read_text())
    if m.get("format") != MANIFEST_FORMAT:
        raise ManifestError(f"{path}: format {m.get('format')!r} != {MANIFEST_FORMAT!r}")
    seen = {}
    for g, members in m["groups"].items():
        if g not in GROUPS:
            raise ManifestError(f"unknown group {g!r}")
        for n in members:
            if n in seen:
                raise ManifestError(f"{n} is in both {seen[n]} and {g}")
            seen[n] = g
    for v, spec in m["versions"].items():
        missing = [n for n in spec["extracted"] if n not in seen]
        if missing:
            raise ManifestError(f"version {v}: names with no group: {missing[:5]}")
    return m


def group_of(manifest: dict) -> Dict[str, str]:
    return {n: g for g, members in manifest["groups"].items() for n in members}


def expand_groups(manifest: dict, groups: Sequence[str]) -> List[str]:
    """Group or alias names -> sorted unique atomic group names (fail closed)."""
    out = []
    for g in groups:
        g = g.strip()
        if not g:
            continue
        if g in manifest["groups"]:
            out.append(g)
        elif g in manifest.get("aliases", {}):
            out.extend(manifest["aliases"][g])
        else:
            raise ManifestError(f"unknown feature group {g!r}; known: "
                                f"{sorted(manifest['groups'])} + aliases "
                                f"{sorted(manifest.get('aliases', {}))}")
    return sorted(set(out))


def identify_version(manifest: dict, extracted_names: Iterable[str]) -> Optional[str]:
    s = set(extracted_names)
    for v, spec in manifest["versions"].items():
        if s == set(spec["extracted"]):
            return v
    return None


LL3_ALLOW_TARGETS_FLAG = "--allow-target-columns-as-features"


def _ll3_requested(exclude_features, include_groups, exclude_groups, allow_target_columns) -> dict:
    """The settings as given (None = flag omitted), recorded in the sidecar so a caller can
    check what each run actually applied (lead TASK_LL_SWITCHES_ADDENDUM_2 Fix 2)."""
    lst = lambda v: None if v is None else [str(x) for x in v]
    return {"exclude_features": lst(exclude_features), "include_groups": lst(include_groups),
            "exclude_groups": lst(exclude_groups),
            "allow_target_columns_as_features": bool(allow_target_columns)}


def validate_step5_settings(data_feature_names: Sequence[str],
                            exclude_features: Optional[Sequence[str]] = None,
                            include_groups: Optional[Sequence[str]] = None,
                            exclude_groups: Optional[Sequence[str]] = None,
                            allow_target_columns: bool = False) -> None:
    """[LL-3] Refuse a bad request before any fitting (Beta X-LL-SWITCHES R1 Q2; lead
    TASK_LL_SWITCHES_ADDENDUM_2 Fix 1). None means the flag was omitted; a supplied value is
    never read as "omitted"."""
    # (c) supplied but empty ('' and ',' parse to []): never the same as omitted
    for flag, v in (("--exclude-features", exclude_features),
                    ("--include-feature-groups", include_groups),
                    ("--exclude-feature-groups", exclude_groups)):
        if v is not None and not [x for x in v if str(x).strip()]:
            raise ManifestError(f"{flag} was given an empty value ({list(v)!r}); omit the flag "
                                f"for the default; refused, nothing trained")
    if allow_target_columns and exclude_features is None:
        raise ManifestError(f"{LL3_ALLOW_TARGETS_FLAG} has no effect without --exclude-features; "
                            f"refused, nothing trained")
    if exclude_features is not None:
        # (a) every name must be a feature of the survivors file (the four default exclusions
        #     are allowed even where a file carries some of them outside 'features')
        known = set(data_feature_names) | set(STEP5_DEFAULT_EXCLUDE)
        unknown = [n for n in exclude_features if n not in known]
        if unknown:
            raise ManifestError(f"--exclude-features names not among the survivors file's "
                                f"features: {unknown}; refused, nothing trained")
        # (b) the default exclusions stay excluded unless explicitly allowed
        missing = [n for n in STEP5_DEFAULT_EXCLUDE if n not in exclude_features]
        if missing and not allow_target_columns:
            raise ManifestError(f"--exclude-features {list(exclude_features)} omits the default "
                                f"exclusion(s) {missing}, which would train as features; list them, or "
                                f"pass {LL3_ALLOW_TARGETS_FLAG}; refused, nothing trained")


def resolve_step5_exclusions(data_feature_names: Sequence[str],
                             exclude_features: Optional[Sequence[str]] = None,
                             include_groups: Optional[Sequence[str]] = None,
                             exclude_groups: Optional[Sequence[str]] = None,
                             manifest_path: Path = MANIFEST_PATH,
                             allow_target_columns: bool = False) -> dict:
    """[LL-3] Resolve Step 5's exclude list from the declared settings.

    With no setting given this returns STEP5_DEFAULT_EXCLUDE and touches no
    manifest (today's behaviour). With a group setting, every data column must
    be in the manifest (fail closed: an unknown column cannot be classified).
    Any other request is validated first (validate_step5_settings).
    Returns {"exclude": [...], "selection": {...} or None}.
    """
    if (exclude_features is None and include_groups is None and exclude_groups is None
            and not allow_target_columns):
        return {"exclude": list(STEP5_DEFAULT_EXCLUDE), "selection": None}
    validate_step5_settings(data_feature_names, exclude_features, include_groups, exclude_groups,
                            allow_target_columns)
    requested = _ll3_requested(exclude_features, include_groups, exclude_groups, allow_target_columns)
    base = list(STEP5_DEFAULT_EXCLUDE if exclude_features is None else exclude_features)
    if include_groups is None and exclude_groups is None:
        # (d) explicit names must leave a feature to train on, as the group path requires
        if not [n for n in data_feature_names if n not in base]:
            raise ManifestError("--exclude-features leaves no feature to train on; refused, nothing trained")
        return {"exclude": base,
                "selection": {"exclude_features": base, "include_groups": None,
                              "exclude_groups": None, "manifest": None,
                              "allow_target_columns_as_features": bool(allow_target_columns),
                              "requested": requested}}
    m = load_manifest(manifest_path)
    gof = group_of(m)
    unknown = sorted(n for n in data_feature_names if n not in gof)
    if unknown:
        raise ManifestError(f"columns not in the feature manifest: {unknown[:10]} "
                            f"({len(unknown)} total); a group setting cannot classify them")
    inc = expand_groups(m, include_groups) if include_groups else None
    exc = expand_groups(m, exclude_groups) if exclude_groups else []
    if inc is not None and set(inc) & set(exc):
        raise ManifestError(f"groups both included and excluded: {sorted(set(inc) & set(exc))}")
    drop = set(base)
    for n in data_feature_names:
        g = gof[n]
        if (inc is not None and g not in inc) or g in exc:
            drop.add(n)
    if not drop:
        # load_survivors_with_features treats an empty list as "use the default"
        raise ManifestError("resolved exclude list is empty; Step 5 would silently use its default")
    kept = [n for n in data_feature_names if n not in drop]
    if not kept:
        raise ManifestError("the feature-group settings leave no feature to train on")
    return {
        "exclude": sorted(drop),
        "selection": {
            "exclude_features": base,
            "include_groups": inc, "exclude_groups": exc,
            "manifest": {"path": str(manifest_path), "sha256": manifest_sha256(manifest_path),
                         "data_version": identify_version(m, data_feature_names)},
            "dropped_by_groups": sorted(drop - set(base)),
            "allow_target_columns_as_features": bool(allow_target_columns),
            "requested": requested,
        },
    }


# ---------------------------------------------------------------------------
# Step 6 check
# ---------------------------------------------------------------------------
def check_rows_against_schema(feature_names: Sequence[str], expected_hash: Optional[str],
                              rows: Iterable[dict], mode: str = "off",
                              logger: Optional[logging.Logger] = None) -> dict:
    """[LL-1] Step 6 schema check. mode 'off' returns at once (today's behaviour).

    Checks: (1) the sidecar's names hash to its own stated hash; (2) every
    sidecar name is present in every row's feature dict, so nothing is
    zero-filled (prediction_generator builds X with features.get(name, 0.0)).
    'warn' logs; 'fail' raises ManifestError.
    """
    if mode not in SCHEMA_CHECK_MODES:
        raise ManifestError(f"schema_check={mode!r} not in {SCHEMA_CHECK_MODES}")
    if mode == "off":
        return {"mode": "off"}
    log = logger or logging.getLogger(__name__)
    problems = []
    if not feature_names:
        problems.append("sidecar has no feature names")
    hash_skipped = isinstance(expected_hash, str) and expected_hash.startswith("n/a:")
    if hash_skipped:
        pass
    elif expected_hash is None:
        problems.append("sidecar has no feature_schema_hash")
    elif feature_names and schema_hash(feature_names) != expected_hash:
        problems.append(f"sidecar names hash to {schema_hash(feature_names)}, "
                        f"sidecar states {expected_hash}")
    want = list(feature_names or [])
    n_rows = 0
    n_rows_missing = 0
    missing_names = {}
    for r in rows:
        n_rows += 1
        feats = r.get("features", {}) if isinstance(r, dict) else {}
        miss = [n for n in want if n not in feats]
        if miss:
            n_rows_missing += 1
            for n in miss:
                missing_names[n] = missing_names.get(n, 0) + 1
    if n_rows_missing:
        top = sorted(missing_names.items(), key=lambda kv: -kv[1])[:5]
        problems.append(f"{n_rows_missing} of {n_rows} rows lack sidecar features "
                        f"(would be zero-filled); most missing: {top}")
    report = {"mode": mode, "hash_test": "skipped " + expected_hash if hash_skipped else "run",
              "rows": n_rows, "rows_missing": n_rows_missing,
              "missing_names": missing_names, "problems": problems}
    try:
        m = load_manifest()
        report["sidecar_trained_version"] = next(
            (v for v, spec in m["versions"].items()
             if set(spec["trained_by_default"]) == set(want)), None)
    except Exception as e:  # the manifest is advisory here; never blocks 'warn'
        report["sidecar_trained_version"] = f"manifest unreadable: {e}"
    for p in problems:
        log.warning(f"[LL-1 schema-check] {p}")
    if problems and mode == "fail":
        raise ManifestError("[LL-1 schema-check] " + "; ".join(problems))
    return report


def variance_report(rows: Iterable[dict], names: Sequence[str]) -> Dict[str, int]:
    """Number of distinct values per name over the rows (measured, not declared)."""
    seen = {n: set() for n in names}
    for r in rows:
        f = r.get("features", {})
        for n in names:
            if n in f:
                seen[n].add(f[n])
    return {n: len(v) for n, v in seen.items()}


if __name__ == "__main__":
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="Build or show the feature manifest (LL-1)")
    ap.add_argument("--build-from", help="a Step 3 survivors_with_scores.json (reads row 0's names)")
    ap.add_argument("--out", default=str(MANIFEST_PATH))
    a = ap.parse_args()
    if a.build_from:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        src = Path(a.build_from)
        with open(src) as f:
            rows = json.load(f)
        man = build_manifest(sorted(rows[0]["features"]),
                             {"step3_output_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                              "rows_read_for_names": 1,
                              "note": "names only; row 0 of a run H Step 3 output"})
        Path(a.out).write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
        print(f"wrote {a.out}: v1 {man['versions']['v1']['extracted_count']} extracted / "
              f"{man['versions']['v1']['trained_by_default_count']} trained, hash "
              f"{man['versions']['v1']['trained_by_default_hash']}")
    else:
        m = load_manifest()
        print(json.dumps({g: len(v) for g, v in m["groups"].items()}))
