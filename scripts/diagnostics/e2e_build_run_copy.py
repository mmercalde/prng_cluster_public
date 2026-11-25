#!/usr/bin/env python3
"""
scripts/diagnostics/e2e_build_run_copy.py -- NON-CERTIFYING DIAGNOSTIC (e2e thin path).

Builds an isolated RUN COPY of one Step 1 generation, for the end-to-end thin
path (docs/drafts/e2e_v1/MAP_E2E_THIN_PATH_20261122.md §4 step 2; lead task
H3, logs/e2e_session/TASK_STEP3.txt item 1).

WHAT IT WRITES (all under --out-dir, which must not exist and must lie under
<this worktree>/data/):
  bidirectional_survivors_binary.npz   the resolved NPZ, byte copy; sha256 must
                                       equal --generation-digest before AND after
  train_history.json                   byte copy of --train-history
  label_block_history.json             holdout[label_start : label_start+label_count],
                                       a bare int list (Step 3's holdout format).
                                       Positionally contiguous: Step 3's holdout law
                                       is offset = len(train_history)
                                       (full_scoring_worker.py:274-305), so the label
                                       block is NEVER session-filtered.
  grade_block_records.json             holdout[grade_start:] as records
                                       {combined_index, date, session, draw}, filtered
                                       by --grade-sessions. Nothing upstream reads it.
  configs/<name>                       byte copies of every --config
  run_record.json                      every declared parameter, every input and
                                       output sha256, block boundaries (index and date),
                                       per-session counts, code identity.

WHAT IT CHECKS (fail-closed, exit 2 with REFUSED):
  * --out-dir: absent, under <worktree>/data/, not under any forbidden root
    (main tree, any ~/skipsw_run_* tree, /dev/shm/prng, ~/miner_staging,
    ~/miner_ledger_archive) and not a symlink path into one.
  * NPZ: sha256 == --generation-digest; loads with allow_pickle=False; passes
    utils.canonical_arrays.validate_array_bundle (22 arrays, order, dtypes);
    seeds unique unless --allow-duplicate-seeds.
  * dataset: sha256 == --dataset-digest when given; a list of {date, session, draw}
    records; its draw sequence equals
    train + holdout exactly (the Step 1 split, window_optimizer.py:1170-1175).
  * the split: 0 <= label_start, label_count >= 1, grade block non-empty,
    label block ends at or before the grade block starts (no overlap).

NOTHING IS REMOVED: the source files are only read. Rows are not filtered. The
session filter only selects which grade-block draws are listed for grading;
the excluded draws are counted in the record.

The owner has not ruled the split (map Q1) or the session filter: both are
arguments with no default. Do NOT run this on run H before Q1 is answered.
"""
from __future__ import annotations

import sys as _sys
_sys.dont_write_bytecode = True

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BUILDER_VERSION = "1.0.0-draft-20261123"
LABEL = "NON-CERTIFYING DIAGNOSTIC"
REPO_ROOT = Path(__file__).resolve().parents[2]
HOME = Path("~").expanduser()

# Forbidden output roots. Extendable by --forbid-root; never shrinkable.
DEFAULT_FORBIDDEN_ROOTS = (
    HOME / "distributed_prng_analysis",
    Path("/dev/shm/prng"),
    HOME / "miner_staging",
    HOME / "miner_ledger_archive",
)
FORBIDDEN_GLOBS = ("skipsw_run_*",)          # every run tree under $HOME, run H included

KNOWN_SESSIONS = ("midday", "evening")


class Refused(SystemExit):
    def __init__(self, msg: str):
        super().__init__(2)
        self.msg = msg


def refuse(msg: str):
    raise Refused(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_under(p: Path, root: Path) -> bool:
    try:
        p.relative_to(root.resolve())
        return True
    except ValueError:
        return False


def forbidden_roots(extra) -> list:
    roots = [r.resolve() if r.exists() else r for r in DEFAULT_FORBIDDEN_ROOTS]
    for g in FORBIDDEN_GLOBS:
        roots.extend(sorted(p.resolve() for p in HOME.glob(g)))
    roots.extend(Path(os.path.expanduser(x)).resolve() for x in (extra or []))
    return roots


def resolve_out(out_dir: Path) -> Path:
    # resolve the deepest existing ancestor, then re-append the rest, so a
    # symlinked parent cannot smuggle the copy into a forbidden tree
    parts = []
    p = out_dir.absolute()
    while not p.exists():
        parts.append(p.name)
        p = p.parent
    r = p.resolve()
    for name in reversed(parts):
        r = r / name
    return r


def check_out_dir(out_dir: Path, extra_forbidden) -> Path:
    if out_dir.exists() or out_dir.is_symlink():
        refuse(f"--out-dir already exists: {out_dir}")
    res = resolve_out(out_dir)
    data_root = (REPO_ROOT / "data").resolve() if (REPO_ROOT / "data").exists() else REPO_ROOT / "data"
    if not is_under(res, data_root) or res == data_root:
        refuse(f"--out-dir must lie under {data_root} (CLAUDE.md rule 7), got {res}")
    for root in forbidden_roots(extra_forbidden):
        if is_under(res, root):
            refuse(f"--out-dir is under forbidden root {root}")
    return res


def load_int_list(path: Path, what: str) -> list:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        refuse(f"{what} must be a non-empty JSON list: {path}")
    for i, v in enumerate(data):
        if isinstance(v, bool) or not isinstance(v, int):
            refuse(f"{what}[{i}] is not an int ({type(v).__name__}): {path}")
    return data


def load_dataset(path: Path) -> list:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        refuse(f"--dataset must be a non-empty JSON list of records: {path}")
    for i, r in enumerate(data):
        if not isinstance(r, dict) or not {"date", "session", "draw"} <= set(r):
            refuse(f"--dataset record {i} lacks date/session/draw: {path}")
        if isinstance(r["draw"], bool) or not isinstance(r["draw"], int):
            refuse(f"--dataset record {i} draw is not an int")
    return data


def check_npz(npz: Path, allow_dup: bool) -> dict:
    import numpy as np
    sys.path.insert(0, str(REPO_ROOT))
    from utils.canonical_arrays import validate_array_bundle
    try:
        with np.load(npz, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    except ValueError as e:
        refuse(f"NPZ needs pickle or is unreadable ({e}): {npz}")
    try:
        validate_array_bundle(arrays)
    except Exception as e:
        refuse(f"NPZ fails the 22-array contract: {type(e).__name__}: {e}")
    seeds = arrays["seeds"]
    rows = int(seeds.shape[0])
    if rows == 0:
        refuse("NPZ has zero rows")
    distinct = int(np.unique(seeds).size)
    if distinct != rows and not allow_dup:
        refuse(f"NPZ seeds are not unique ({distinct} distinct of {rows}); "
               f"Step 3 keys metadata by seed (full_scoring_worker.py:584). "
               f"Pass --allow-duplicate-seeds to proceed and record it.")

    def counts(name):
        u, c = np.unique(arrays[name], return_counts=True)
        return {str(int(a)): int(b) for a, b in zip(u, c)}

    return {"rows": rows, "distinct_seeds": distinct, "keys": list(arrays.keys()),
            "skip_mode_counts": counts("skip_mode"), "prng_type_counts": counts("prng_type"),
            "skip_min_range": [int(arrays["skip_min"].min()), int(arrays["skip_min"].max())],
            "skip_max_range": [int(arrays["skip_max"].min()), int(arrays["skip_max"].max())]}


def git_identity() -> dict:
    out = {"repo_root": str(REPO_ROOT)}
    me = Path(__file__).resolve()
    try:
        out["head"] = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                     capture_output=True, text=True, check=True).stdout.strip()
        out["status_of_builder"] = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", str(me)],
            capture_output=True, text=True, check=True).stdout.splitlines()
    except Exception as e:  # recorded, not fatal
        out["error"] = f"{type(e).__name__}: {e}"
    out["builder_sha256"] = sha256_file(me)
    out["canonical_arrays_sha256"] = sha256_file(REPO_ROOT / "utils" / "canonical_arrays.py")
    return out


def parse_sessions(s: str) -> list:
    vals = [x.strip() for x in s.split(",") if x.strip()]
    if not vals:
        refuse("--grade-sessions is empty")
    if vals == ["all"]:
        return list(KNOWN_SESSIONS)
    bad = [v for v in vals if v not in KNOWN_SESSIONS]
    if bad:
        refuse(f"--grade-sessions has unknown values {bad}; known: {list(KNOWN_SESSIONS)} or 'all'")
    return vals


def build(args) -> dict:
    t0 = time.time()
    npz = Path(os.path.expanduser(args.npz))
    train_p = Path(os.path.expanduser(args.train_history))
    hold_p = Path(os.path.expanduser(args.holdout_history))
    data_p = Path(os.path.expanduser(args.dataset))
    configs = [Path(os.path.expanduser(c)) for c in (args.config or [])]
    out_dir = Path(os.path.expanduser(args.out_dir))

    for p, what in [(npz, "--npz"), (train_p, "--train-history"), (hold_p, "--holdout-history"),
                    (data_p, "--dataset")] + [(c, "--config") for c in configs]:
        if not p.is_file():
            refuse(f"{what} is not a file: {p}")
    if npz.suffix != ".npz":
        refuse(f"--npz must be a .npz file: {npz}")
    digest = args.generation_digest.lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        refuse("--generation-digest must be a 64-hex sha256")
    names = [c.name for c in configs]
    if len(set(names)) != len(names):
        refuse(f"--config basenames collide: {names}")

    out_res = check_out_dir(out_dir, args.forbid_root)

    # ---- inputs -------------------------------------------------------------
    npz_real = npz.resolve()
    npz_sha = sha256_file(npz_real)
    if npz_sha != digest:
        refuse(f"NPZ sha256 {npz_sha} != --generation-digest {digest}")
    npz_info = check_npz(npz_real, args.allow_duplicate_seeds)

    train = load_int_list(train_p, "--train-history")
    hold = load_int_list(hold_p, "--holdout-history")
    dataset = load_dataset(data_p)
    data_sha = sha256_file(data_p)
    if args.dataset_digest is not None and data_sha != args.dataset_digest.lower():
        refuse(f"dataset sha256 {data_sha} != --dataset-digest {args.dataset_digest}")
    if [r["draw"] for r in dataset] != train + hold:
        refuse("dataset draw sequence != train_history + holdout_history "
               "(the Step 1 split, window_optimizer.py:1170-1175)")
    T, H = len(train), len(hold)

    # ---- split (owner Q1: no defaults) --------------------------------------
    ls, lc = args.label_start, args.label_count
    gs = args.grade_start if args.grade_start is not None else ls + lc
    if ls < 0 or lc < 1:
        refuse("--label-start must be >= 0 and --label-count >= 1")
    if ls + lc > H:
        refuse(f"label block [{ls}, {ls + lc}) exceeds the holdout ({H} draws)")
    if gs < ls + lc:
        refuse(f"grade block start {gs} overlaps the label block ending at {ls + lc}")
    if gs >= H:
        refuse(f"grade block start {gs} leaves no grade draws (holdout has {H})")
    if ls != 0:
        # Step 3's holdout law reads the holdout file from combined index T
        # (offset = len(train_history), full_scoring_worker.py:299-305). A label
        # block not starting at holdout index 0 would be read at the wrong
        # positions. Refused rather than silently misaligned.
        refuse("--label-start != 0 is refused: Step 3 reads the holdout file from "
               "combined index len(train_history) (full_scoring_worker.py:299-305)")
    sessions = parse_sessions(args.grade_sessions)

    label_block = hold[ls:ls + lc]
    grade_all = [{"combined_index": T + j, "holdout_index": j, "date": dataset[T + j]["date"],
                  "session": dataset[T + j]["session"], "draw": dataset[T + j]["draw"]}
                 for j in range(gs, H)]
    grade = [r for r in grade_all if r["session"] in sessions]
    if not grade:
        refuse(f"no grade draws left after --grade-sessions {sessions}")

    def sess_counts(recs):
        c = {}
        for r in recs:
            c[r["session"]] = c.get(r["session"], 0) + 1
        return dict(sorted(c.items()))

    label_recs = [dataset[T + j] for j in range(ls, ls + lc)]

    # ---- write --------------------------------------------------------------
    os.makedirs(out_res, exist_ok=False)
    (out_res / "configs").mkdir()
    outputs = {}

    def put_copy(src: Path, rel: str, expect_sha: str = None):
        dst = out_res / rel
        shutil.copyfile(src, dst)
        s = sha256_file(dst)
        if expect_sha is not None and s != expect_sha:
            refuse(f"copy of {src} changed in transit ({s} != {expect_sha})")
        outputs[rel] = s
        return s

    def put_json(obj, rel: str):
        dst = out_res / rel
        with open(dst, "w") as f:
            json.dump(obj, f, indent=2)
        outputs[rel] = sha256_file(dst)

    put_copy(npz_real, "bidirectional_survivors_binary.npz", npz_sha)
    if sha256_file(npz_real) != npz_sha:
        refuse("source NPZ changed while being copied")
    train_sha = sha256_file(train_p)
    put_copy(train_p, "train_history.json", train_sha)
    put_json(label_block, "label_block_history.json")
    put_json(grade, "grade_block_records.json")
    cfg_rec = {}
    for c in configs:
        cs = sha256_file(c)
        put_copy(c, f"configs/{c.name}", cs)
        cfg_rec[c.name] = {"source": str(c), "source_resolved": str(c.resolve()), "sha256": cs}

    record = {
        "label": LABEL,
        "kind": "e2e_run_copy",
        "builder_version": BUILDER_VERSION,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "host": os.uname().nodename,
        "argv": sys.argv,
        "args": vars(args),
        "declared_parameters": {
            "generation_digest": digest,
            "label_start": ls, "label_count": lc, "grade_start": gs,
            "grade_sessions": sessions,
            "label_block_session_filter": "none (refused by design: Step 3 position law)",
            "allow_duplicate_seeds": args.allow_duplicate_seeds,
        },
        "inputs": {
            "npz": {"path": str(npz), "resolved": str(npz_real), "sha256": npz_sha, **npz_info},
            "train_history": {"path": str(train_p), "resolved": str(train_p.resolve()),
                              "sha256": train_sha, "count": T},
            "holdout_history": {"path": str(hold_p), "resolved": str(hold_p.resolve()),
                                "sha256": sha256_file(hold_p), "count": H,
                                "note": "read only; NOT copied, so no consumer can be fed the full holdout by default"},
            "dataset": {"path": str(data_p), "resolved": str(data_p.resolve()),
                        "sha256": data_sha, "expected_sha256": args.dataset_digest,
                        "count": len(dataset)},
            "configs": cfg_rec,
        },
        "blocks": {
            "train": {"combined_index": [0, T], "first_date": dataset[0]["date"],
                      "last_date": dataset[T - 1]["date"]},
            "label_block": {"holdout_index": [ls, ls + lc], "combined_index": [T + ls, T + ls + lc],
                            "count": lc, "first_date": label_recs[0]["date"],
                            "last_date": label_recs[-1]["date"],
                            "session_counts": sess_counts(label_recs)},
            "grade_block": {"holdout_index": [gs, H], "combined_index": [T + gs, T + H],
                            "count_before_session_filter": len(grade_all),
                            "count_listed": len(grade),
                            "count_excluded_by_session_filter": len(grade_all) - len(grade),
                            "first_date": grade_all[0]["date"], "last_date": grade_all[-1]["date"],
                            "session_counts_before_filter": sess_counts(grade_all),
                            "session_counts_listed": sess_counts(grade)},
            "gap_between_label_and_grade": gs - (ls + lc),
        },
        "outputs": outputs,
        "code": git_identity(),
        "wall_seconds": None,
    }
    record["wall_seconds"] = round(time.time() - t0, 3)
    with open(out_res / "run_record.json", "w") as f:
        json.dump(record, f, indent=2)
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=f"{LABEL}: build an isolated e2e run copy.")
    ap.add_argument("--npz", required=True, help="survivor NPZ (symlink allowed; the real file is copied)")
    ap.add_argument("--generation-digest", required=True,
                    help="expected sha256 of the NPZ file (generation artifact_sha256)")
    ap.add_argument("--train-history", required=True)
    ap.add_argument("--holdout-history", required=True)
    ap.add_argument("--dataset", required=True,
                    help="the dated, sessioned record list the split was cut from (daily3.json)")
    ap.add_argument("--dataset-digest", default=None,
                    help="expected sha256 of --dataset (e.g. the generation receipt's dataset_sha256)")
    ap.add_argument("--config", action="append", default=[],
                    help="config file to copy into configs/ (repeatable)")
    ap.add_argument("--label-start", type=int, required=True,
                    help="holdout index where the label block starts (only 0 is accepted)")
    ap.add_argument("--label-count", type=int, required=True,
                    help="number of holdout draws in the label block (owner Q1)")
    ap.add_argument("--grade-start", type=int, default=None,
                    help="holdout index where the grade block starts (default: label end)")
    ap.add_argument("--grade-sessions", required=True,
                    help="comma list of sessions listed for grading: midday,evening or 'all' (owner Q1)")
    ap.add_argument("--out-dir", required=True, help="created; must lie under <worktree>/data/")
    ap.add_argument("--forbid-root", action="append", default=[],
                    help="extra forbidden output root (repeatable; adds to the defaults)")
    ap.add_argument("--allow-duplicate-seeds", action="store_true")
    args = ap.parse_args(argv)
    try:
        rec = build(args)
    except Refused as e:
        print(f"[{LABEL}] REFUSED: {e.msg}", file=sys.stderr)
        return 2
    b = rec["blocks"]
    print(f"[{LABEL}] run copy written: {args.out_dir}")
    print(f"[{LABEL}] NPZ rows {rec['inputs']['npz']['rows']} sha256 {rec['inputs']['npz']['sha256']}")
    print(f"[{LABEL}] train {rec['inputs']['train_history']['count']}  "
          f"label block {b['label_block']['count']} {b['label_block']['session_counts']}  "
          f"grade listed {b['grade_block']['count_listed']} of {b['grade_block']['count_before_session_filter']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
