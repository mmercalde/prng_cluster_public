#!/usr/bin/env python3
"""G3 RUN-AUTHORIZATION GATE — refuse any launch the owner has not authorized in writing.

WHAT THIS IS
------------
`docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md` §5.2 / §7.1. One authorization
file per run class, JSON text, tracked in git at
`config_manifests/run_authorizations/<CLASS>.auth`. Only the owner commits, so a
file that is tracked and byte-identical to `HEAD` is the owner's.

    {"run_class": "RUN4",
     "params": {"<name>": <exact value> | {"min": n, "max": n}, ...},
     "seed_start_min": int, "seed_end_max_exclusive": int,
     "max_launches": int, "max_wall_clock_s": int,
     "window_local": {"tz": "America/Los_Angeles", "start_hour": 0-23, "end_hour": 1-24},
     "expires": "YYYY-MM-DD"}

LAUNCH PARAMETERS (`--params`, JSON object or @file)
---------------------------------------------------
Every key is a launch parameter and must be named in the authorization, EXCEPT the
two reserved keys that state the launch's seed range and are checked against the
authorized range: `seed_start` and `seed_end_exclusive` (both required). When the
launch also carries `max_seeds`, `seed_end_exclusive` must equal
`seed_start + max_seeds`.

THE RULE — anything not explicitly authorized is refused, not interpreted:
    file absent / unparseable / missing a field / wrong type      -> REFUSE
    file untracked, or differs from HEAD; git cannot run          -> REFUSE
    run_class != --class                                          -> REFUSE
    today (window tz) after `expires`                             -> REFUSE
    hour (window tz) outside [start_hour, end_hour)               -> REFUSE
    launches recorded under this file's sha256 >= max_launches     -> REFUSE
    a launch parameter not named in `params`                      -> REFUSE
    a named parameter absent from the launch                      -> REFUSE
    a value != its exact value, or outside min/max (inclusive)    -> REFUSE
    seed range not inside [seed_start_min, seed_end_max_exclusive) -> REFUSE
    end_step != 1 or n_parallel != 1, when supplied               -> REFUSE
    everything holds                                              -> PROCEED
Comparison: integers exact; if either side is a float, float() on both, no
rounding; booleans only equal booleans (True never equals 1 or "true"); a string
never equals a number; lists compare element-wise under the same rules.

LAUNCH RECORD
-------------
Append-only JSON lines in `logs/run_authorizations/<CLASS>.jsonl` (`logs/` is
ignored). `consume` re-runs the full verification under an exclusive lock and then
appends {class, nonce, utc, auth_sha256, params}. A nonce already recorded is
REFUSE. Records made under a different authorization sha256 are listed and not
counted. KNOWN LIMIT: the record lives under the agent's own account, so it can be
deleted — the count is cross-checkable, not tamper-proof.

Exit status: 0 = PROCEED, 1 = REFUSE.
"""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(REPO_ROOT, "scripts")
for _p in (REPO_ROOT, _SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import env_freeze_gate as EF                                        # noqa: E402

try:
    from zoneinfo import ZoneInfo                                   # py >= 3.9
except ImportError:                                                 # pragma: no cover
    ZoneInfo = None

EXIT_PROCEED = EF.EXIT_PROCEED
EXIT_REFUSE = EF.EXIT_REFUSE

AUTH_DIR_REL = "config_manifests/run_authorizations"
RECORD_DIR_REL = "logs/run_authorizations"
CLASS_RE = re.compile(r"^[A-Z0-9_]{1,32}$")
NONCE_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")

REQUIRED_FIELDS = ("run_class", "params", "seed_start_min", "seed_end_max_exclusive",
                   "max_launches", "max_wall_clock_s", "window_local", "expires")
SEED_KEYS = ("seed_start", "seed_end_exclusive")
# Absolute, whatever the authorization says (brief §3; proposal §9 SOAK; RUN4 §1).
MUST_BE_ONE = (("end_step", "--end-step", "end-step"),
               ("n_parallel", "--n-parallel", "n-parallel"))


class AuthError(ValueError):
    pass


def _reject_constant(name):
    raise ValueError(f"non-finite number {name} is not permitted")


def _loads(text):
    return json.loads(text, parse_constant=_reject_constant)


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def values_equal(expected, actual):
    """The comparison rule, in one place."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return isinstance(expected, bool) and isinstance(actual, bool) \
            and expected is actual
    if _is_int(expected) and _is_int(actual):
        return expected == actual
    if _is_num(expected) and _is_num(actual):
        return float(expected) == float(actual)
    if isinstance(expected, str) and isinstance(actual, str):
        return expected == actual
    if expected is None and actual is None:
        return True
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            values_equal(e, a) for e, a in zip(expected, actual))
    return False


def _is_bound(spec):
    return isinstance(spec, dict) and set(spec) == {"min", "max"}


def within_bound(spec, actual):
    if not _is_num(actual):
        return False
    lo, hi = spec["min"], spec["max"]
    if _is_int(lo) and _is_int(hi) and _is_int(actual):
        return lo <= actual <= hi
    return float(lo) <= float(actual) <= float(hi)


# ---------------------------------------------------------------------------
# the authorization file
# ---------------------------------------------------------------------------

def auth_rel(run_class):
    return f"{AUTH_DIR_REL}/{run_class}.auth"


def validate_auth(doc):
    """Structural validation. Raises AuthError naming the first defect."""
    if not isinstance(doc, dict):
        raise AuthError("authorization is not a JSON object")
    missing = [f for f in REQUIRED_FIELDS if f not in doc]
    if missing:
        raise AuthError(f"missing required field(s): {missing}")
    if not isinstance(doc["run_class"], str):
        raise AuthError("run_class is not a string")
    params = doc["params"]
    if not isinstance(params, dict) or not params:
        raise AuthError("params is not a non-empty object")
    for name, spec in params.items():
        if isinstance(spec, dict):
            if not _is_bound(spec) or not (_is_num(spec["min"]) and _is_num(spec["max"])):
                raise AuthError(f"params.{name}: an object must be exactly "
                                f"{{'min': n, 'max': n}} with numbers")
            if not float(spec["min"]) <= float(spec["max"]):
                raise AuthError(f"params.{name}: min > max")
    for f in ("seed_start_min", "seed_end_max_exclusive", "max_launches",
              "max_wall_clock_s"):
        if not _is_int(doc[f]):
            raise AuthError(f"{f} is not an integer")
    if doc["seed_start_min"] < 0 or doc["seed_end_max_exclusive"] <= doc["seed_start_min"]:
        raise AuthError("seed range is empty or negative")
    if doc["max_launches"] < 1 or doc["max_wall_clock_s"] < 1:
        raise AuthError("max_launches and max_wall_clock_s must be >= 1")
    w = doc["window_local"]
    if not isinstance(w, dict) or set(w) != {"tz", "start_hour", "end_hour"}:
        raise AuthError("window_local must hold exactly tz, start_hour, end_hour")
    if not (_is_int(w["start_hour"]) and _is_int(w["end_hour"])
            and 0 <= w["start_hour"] <= 23 and 0 <= w["end_hour"] <= 24
            and w["start_hour"] < w["end_hour"]):
        raise AuthError("window_local hours must satisfy 0 <= start_hour < end_hour <= 24 "
                        "(a window crossing midnight is not supported)")
    if ZoneInfo is None:
        raise AuthError("zoneinfo unavailable; cannot evaluate window_local")
    try:
        ZoneInfo(str(w["tz"]))
    except Exception as exc:                                        # noqa: BLE001
        raise AuthError(f"window_local.tz {w['tz']!r} is not a known zone: {exc}")
    if not isinstance(doc["expires"], str):
        raise AuthError("expires is not a string")
    try:
        datetime.date.fromisoformat(doc["expires"])
    except ValueError:
        raise AuthError(f"expires {doc['expires']!r} is not YYYY-MM-DD")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", doc["expires"]):
        raise AuthError(f"expires {doc['expires']!r} is not YYYY-MM-DD")


def load_authorization(repo_root, run_class, runner=None):
    """(doc, sha256, None) or (None, None, refusal)."""
    rel = auth_rel(run_class)
    status, reason, data = EF.tracked_unmodified(repo_root, rel, runner)
    if status != "OK":
        return None, None, f"authorization {status}: {reason}"
    try:
        doc = _loads(data.decode("utf-8"))
        validate_auth(doc)
    except Exception as exc:                                        # noqa: BLE001
        return None, None, f"authorization {rel} unparseable or malformed: {exc}"
    return doc, hashlib.sha256(data).hexdigest(), None


# ---------------------------------------------------------------------------
# the launch record
# ---------------------------------------------------------------------------

def record_path(repo_root, run_class):
    return os.path.join(repo_root, RECORD_DIR_REL, f"{run_class}.jsonl")


def read_records(path):
    """List of dicts. A malformed line raises — an uncountable record is a refusal."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if not line.strip():
                continue
            rec = _loads(line)
            if not isinstance(rec, dict) or not all(
                    k in rec for k in ("class", "nonce", "utc", "auth_sha256", "params")):
                raise ValueError(f"{path}:{i}: malformed launch record")
            out.append(rec)
    return out


def split_records(records, auth_sha):
    counted = [r for r in records if r["auth_sha256"] == auth_sha]
    other = [r for r in records if r["auth_sha256"] != auth_sha]
    return counted, other


# ---------------------------------------------------------------------------
# the verdict
# ---------------------------------------------------------------------------

def check(doc, auth_sha, run_class, launch, records, now_utc):
    """Pure. Returns (refusals, info). `now_utc` is an aware datetime."""
    refusals = []
    if doc["run_class"] != run_class:
        refusals.append(f"run_class {doc['run_class']!r} in the file != --class "
                        f"{run_class!r}")

    w = doc["window_local"]
    local = now_utc.astimezone(ZoneInfo(w["tz"]))
    expires = datetime.date.fromisoformat(doc["expires"])
    if local.date() > expires:
        refusals.append(f"expired: today {local.date()} ({w['tz']}) is after "
                        f"expires {expires}")
    if not (w["start_hour"] <= local.hour < w["end_hour"]):
        refusals.append(f"outside window: local hour {local.hour} ({w['tz']}, "
                        f"{local.isoformat()}) not in "
                        f"[{w['start_hour']}, {w['end_hour']})")

    counted, other = split_records(records, auth_sha)
    if len(counted) >= doc["max_launches"]:
        refusals.append(f"launches exhausted: {len(counted)} recorded under this "
                        f"authorization >= max_launches {doc['max_launches']}")

    if not isinstance(launch, dict):
        refusals.append("launch parameters are not a JSON object")
        launch = {}
    params = doc["params"]
    for name in sorted(set(launch) - set(params) - set(SEED_KEYS)):
        refusals.append(f"launch parameter {name!r} is not named in the "
                        f"authorization (an unknown parameter is never assumed "
                        f"harmless)")
    for name in sorted(params):
        if name not in launch:
            refusals.append(f"authorized parameter {name!r} is absent from the launch")
            continue
        spec, val = params[name], launch[name]
        if _is_bound(spec):
            if not within_bound(spec, val):
                refusals.append(f"parameter {name!r} = {val!r} outside "
                                f"[{spec['min']!r}, {spec['max']!r}]")
        elif not values_equal(spec, val):
            refusals.append(f"parameter {name!r} = {val!r} "
                            f"({type(val).__name__}) != authorized {spec!r} "
                            f"({type(spec).__name__})")

    s, e = launch.get("seed_start"), launch.get("seed_end_exclusive")
    if not (_is_int(s) and _is_int(e)):
        refusals.append(f"launch seed range not stated as integers "
                        f"seed_start={s!r} seed_end_exclusive={e!r}")
    else:
        if s < doc["seed_start_min"]:
            refusals.append(f"seed_start {s} < seed_start_min {doc['seed_start_min']}")
        if e > doc["seed_end_max_exclusive"]:
            refusals.append(f"seed_end_exclusive {e} > seed_end_max_exclusive "
                            f"{doc['seed_end_max_exclusive']}")
        if e <= s:
            refusals.append(f"seed range [{s}, {e}) is empty")
        # The real launch states its range as seed_start + max_seeds
        # (gate12_launch.sh --params). When both are present the stated end must
        # agree, so a launcher cannot declare one range and run another.
        if "max_seeds" in launch and not (_is_int(launch["max_seeds"])
                                          and e == s + launch["max_seeds"]):
            refusals.append(f"seed_end_exclusive {e} != seed_start {s} + max_seeds "
                            f"{launch['max_seeds']!r}")

    for spellings in MUST_BE_ONE:
        for key in spellings:
            if key in launch and not values_equal(1, launch[key]):
                refusals.append(f"{key} = {launch[key]!r}; must be 1 whenever supplied")

    info = {"counted": counted, "other_hash": other,
            "remaining": max(doc["max_launches"] - len(counted), 0),
            "local_time": local.isoformat()}
    return refusals, info


def _now(clock):
    t = (clock or (lambda: datetime.datetime.now(datetime.timezone.utc)))()
    if t.tzinfo is None:
        raise ValueError("clock returned a naive datetime")
    return t


def _parse_params(arg):
    if arg.startswith("@"):
        with open(arg[1:], encoding="utf-8") as fh:
            arg = fh.read()
    return _loads(arg)


def _report(run_class, refusals, info, word="REFUSE"):
    if info:
        for r in info["other_hash"]:
            print(f"  NOT COUNTED (different authorization sha256 "
                  f"{r['auth_sha256'][:16]}…): nonce={r['nonce']} utc={r['utc']}")
    if refusals:
        print(f"RUN-AUTHORIZATION GATE [{run_class}] : {word} — {len(refusals)} reason(s):")
        for line in refusals:
            print(f"  * {line}")
        return EXIT_REFUSE
    print(f"RUN-AUTHORIZATION GATE [{run_class}] : PROCEED — "
          f"{info['remaining']} launch(es) remain before this one is consumed.")
    return EXIT_PROCEED


def _write_evidence(path, payload):
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")


def _verify_core(repo_root, run_class, params_arg, runner, clock):
    """(refusals, info, doc, sha, launch). Every failure path is a refusal."""
    if not CLASS_RE.match(run_class or ""):
        return [f"--class {run_class!r} is not [A-Z0-9_]{{1,32}}"], None, None, None, None
    doc, sha, why = load_authorization(repo_root, run_class, runner)
    if why:
        return [why], None, None, None, None
    try:
        launch = _parse_params(params_arg)
    except Exception as exc:                                        # noqa: BLE001
        return [f"--params unparseable: {exc}"], None, doc, sha, None
    try:
        records = read_records(record_path(repo_root, run_class))
    except Exception as exc:                                        # noqa: BLE001
        return [f"launch record unreadable, so launches cannot be counted: {exc}"], \
            None, doc, sha, launch
    try:
        now = _now(clock)
    except Exception as exc:                                        # noqa: BLE001
        return [f"clock unavailable: {exc}"], None, doc, sha, launch
    refusals, info = check(doc, sha, run_class, launch, records, now)
    return refusals, info, doc, sha, launch


def cmd_verify(args, repo_root, runner, clock):
    refusals, info, _doc, sha, launch = _verify_core(
        repo_root, args.run_class, args.params, runner, clock)
    _write_evidence(args.evidence_json, {
        "gate": "run_authorization_gate", "class": args.run_class,
        "auth_sha256": sha, "launch": launch,
        "verdict": "REFUSE" if refusals else "PROCEED", "reasons": refusals,
        "not_counted": (info or {}).get("other_hash", [])})
    return _report(args.run_class, refusals, info)


def cmd_consume(args, repo_root, runner, clock):
    if not NONCE_RE.match(args.run_nonce or ""):
        return _report(args.run_class, [f"--run-nonce {args.run_nonce!r} is not "
                                        f"[A-Za-z0-9._:-]{{1,128}}"], None)
    if not CLASS_RE.match(args.run_class or ""):
        return _report(args.run_class, [f"--class {args.run_class!r} invalid"], None)
    path = record_path(repo_root, args.run_class)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a+", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        refusals, info, _doc, sha, launch = _verify_core(
            repo_root, args.run_class, args.params, runner, clock)
        if not refusals:
            existing = read_records(path)
            if any(r["nonce"] == args.run_nonce for r in existing):
                refusals = [f"nonce {args.run_nonce!r} already recorded"]
        if refusals:
            return _report(args.run_class, refusals, info)
        rec = {"class": args.run_class, "nonce": args.run_nonce,
               "utc": _now(clock).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "auth_sha256": sha, "params": launch}
        fh.seek(0, os.SEEK_END)
        fh.write(json.dumps(rec, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    print(f"RUN-AUTHORIZATION GATE [{args.run_class}] : CONSUMED nonce "
          f"{args.run_nonce} ({info['remaining'] - 1} remain)")
    return EXIT_PROCEED


def cmd_show(args, repo_root, runner, clock):
    if not CLASS_RE.match(args.run_class or ""):
        print(f"--class {args.run_class!r} invalid")
        return EXIT_REFUSE
    doc, sha, why = load_authorization(repo_root, args.run_class, runner)
    if why:
        print(f"RUN-AUTHORIZATION GATE [{args.run_class}] : REFUSE — {why}")
        return EXIT_REFUSE
    try:
        records = read_records(record_path(repo_root, args.run_class))
    except Exception as exc:                                        # noqa: BLE001
        print(f"RUN-AUTHORIZATION GATE [{args.run_class}] : REFUSE — {exc}")
        return EXIT_REFUSE
    counted, other = split_records(records, sha)
    print(json.dumps(doc, indent=2, sort_keys=True))
    print(f"authorization sha256 : {sha}")
    print(f"launches used        : {len(counted)} / {doc['max_launches']}")
    for r in counted:
        print(f"  counted     nonce={r['nonce']} utc={r['utc']}")
    for r in other:
        print(f"  NOT COUNTED nonce={r['nonce']} utc={r['utc']} "
              f"auth_sha256={r['auth_sha256']}")
    return EXIT_PROCEED


def main(argv=None, *, repo_root=REPO_ROOT, runner=None, clock=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--class", dest="run_class", required=True)
    v.add_argument("--params", required=True, help="JSON object, or @file")
    v.add_argument("--evidence-json", default=None)
    c = sub.add_parser("consume", help="record one launch; re-verifies under lock. "
                       "Called by a launcher only after every gate has passed.")
    c.add_argument("--class", dest="run_class", required=True)
    c.add_argument("--run-nonce", required=True)
    c.add_argument("--params", required=True, help="JSON object, or @file")
    s = sub.add_parser("show")
    s.add_argument("--class", dest="run_class", required=True)
    args = ap.parse_args(argv)
    fn = {"verify": cmd_verify, "consume": cmd_consume, "show": cmd_show}[args.cmd]
    return fn(args, repo_root, runner, clock)


if __name__ == "__main__":
    sys.exit(main())
