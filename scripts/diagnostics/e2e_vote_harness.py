#!/usr/bin/env python3
"""
e2e_vote_harness.py - Harness H6: the run-scoped Step 6 vote (TFM thin path).

Status: DRAFT, uncommitted, Team Alpha (Claude), worktree e2e-vote, 2026-11-23.
Route: docs/drafts/e2e_v1/MAP_E2E_THIN_PATH_20261122.md section 2.4 (4), worktree e2e-map.
Task: logs/e2e_session/TASK_VOTE.txt (H6) under TASK_COMMON.txt, main tree.

WHAT IT DOES
  For every combined draw index t in the half-open block [--block-start, --block-end) it
  asks Step 6's own code for the pool Step 6 would emit for the draw at index t:

      PredictionGenerator.generate_predictions(survivors, history[:t], None, k=config.k)

  (prediction_generator.py:559-738, which calls _build_prediction_pool :752-862). Inside
  that call the module builds X from each survivor's `features` dict (:777-809), scores it
  with the model (:827-833), ranks with argsort (:836), and votes seq[next_idx] with
  next_idx = len(history passed) = t and skip = the survivor's skip (:839-850). The harness
  does NOT copy any of that. It passes Step 3 survivor dicts (so features are never
  recomputed, :782-784) and the history prefix (so next_idx = t).

  "Take the ranking once": the generator's model is wrapped in RankOnce. The first
  predict(X) call runs the real model (or the declared unlearned key). Every later call
  must present a byte-identical X (sha256 over dtype, shape and bytes) and gets the same
  score array back. A different X aborts the run. The ranking is computed once, and the
  harness proves it on every draw.

YARDSTICK (CLAUDE.md): seeds are vehicles. This harness writes pools. It never judges a
seed and never writes an outcome or a hit. It reads the dataset only to bind identity
(sha256, date, session) and to check that the history files match it. The vote for target t
is handed history[:t] only. Grading is the grader's job (TASK G).

OUTPUT (lead decision 2026-11-23: one format, defined once, by the grader's reader):
  pools.jsonl            tfm.e2e.pool.v1 records, exactly the fields e2e_grader.py reads
  vote_diagnostics.jsonl per-draw H6 detail (vote index, raw scores, model type, why empty)
  run_record.json        parameters, input hashes, ranking proof, counts, time, memory
An empty pool is written as abstain=false with pool [] so the grader's declared empty_pool
rule decides. Only a Step 6 signal-gate skip is written as abstain=true.

NOTHING FIXED: k, pool size, block, vote-position arm, ranking source and the key fields
are CLI arguments. Declared defaults are listed in --help and written to run_record.json.

FAIL-CLOSED: the run stops if any of these happens:
  - a survivor row is not a dict with a `seed` and a `features` dict (the module would
    drop it with a warning at :791-793);
  - the config file is missing or unreadable (the CLI falls back to defaults at :1012-1016);
  - the sidecar ranking path loads no model (the module falls back to the feature mean at
    :831-833, Ch7 B2);
  - X changes between draws;
  - a result status is unknown;
  - the out-dir already exists.
Abstentions are written as lines. No draw in the block is ever dropped.

GPU: off by default. CUDA_VISIBLE_DEVICES="" is set before any project import, unless
--allow-gpu is given.

Shortcuts (TODO-E2E-n, see the READY note):
  TODO-E2E-8   the harness calls the module's ranking and vote, and the self-test proves
               byte identity with the CLI at t = len(history) (prediction_generator.py:777-850).
  TODO-E2E-9   the dual-sieve path is untouched. The harness passes survivors_reverse=None
               and use_dual_sieve=False (:603-618).
  TODO-E2E-12  the vote position and skip follow Step 6's convention (:839, :846).
  TODO-E2E-V1  the unlearned ranking arms construct PredictionGenerator with
               MULTI_MODEL_AVAILABLE patched to False for the duration of __init__. The
               module has no sidecar-free constructor (:288-294). See _construct_unlearned.
  TODO-E2E-V2  the CLI plumbing (config overrides :1018-1029, survivors/history file
               loaders :1042-1056) is mirrored here, because main() is not factored.
               The ranking and vote are not mirrored.
  TODO-E2E-V3  the midday_stream_index vote arm is a declared hook only. It raises
               NotImplementedError until WP-1 Stage B / Ch7 O3 fix the law.
  TODO-E2E-V4  --tie-break (default `module` = numpy argsort at :836, the CLI's behaviour).
               A declared secondary key is applied by turning the scores into distinct rank
               positions before the module's argsort. The tie report in run_record.json says
               how many pool slots tie order decided.
  TODO-E2E-V5  --on-missing-feature (default fail). The module zero-fills a missing schema
               column (:798, Ch6 B5). `record` reproduces the CLI and counts the rows.
  TODO-E2E-V10 --seal-from: dataset rows at or after the seal are read for date/session only;
               their draw values are discarded at parse time (load_dataset) and votes past
               the seal get a None-padded history (history_for). Self-test G14.
  TODO-E2E-V9  neural_net sidecars must have the input scaler and the wrapper's y_mean/y_std
               loaded, or the run stops (check_nn_normalisation; lead rule 2026-11-23).
  TODO-E2E-V11 LL-5 (PLAN_CH14_LEARNING_LOOP.md, worktree ch14-learn): --emit-ranked-predictions
               (default off) writes ranked_predictions.jsonl, one Chapter 13 reader-shaped object
               per target (chapter_13_orchestrator.py:901-929). The survivor behind pool slot i
               is found by the same argsort the module uses (:836) over the RankOnce scores; each
               slot's raw score is checked against that survivor's score, so a wrong mapping
               stops the run. The plan puts LL-5 in prediction_generator.py; it is here because
               that file is not edited. draw_id follows chapter_13_diagnostics.py:163
               (draw_<len(draws)>, so the draw at index t is draw_<t+1>). Off: no file, no
               record key, pools and config_id unchanged.
  TODO-E2E-V12 --emit-value-ordering full_vote (TASK_LOOP_ADDENDUM_3, h6 item 1; default off):
               every pool record also carries value_scores, the grader's R24 field
               (e2e_grader.py 3ab8ce2a, header R24; draw_ranks). The ordering is Step 6's own vote
               rule extended to all survivors: value v scores -(rank of the first ranked survivor
               voting v), values no survivor votes tie at -(n_survivors + 1). The votes are seq[t]
               of the module's generator for every survivor (survivor_scorer.py:119-125, skip by
               the module's rule, pg:795, :846), built once per run as a matrix (the method of
               e2e_miss_report.py full_vote). The module itself votes only ranked_idx[:pool_size]
               (:841): identity is proven per draw (the ordering's first len(pool) votes equal the
               pool, and its top distinct values equal the pool's distinct values in order), or
               the run stops. On: "value_ordering" enters config_id. Off: unchanged.
  TODO-E2E-V13 --ranking-key-round-decimals D (h6 item 2; default off): for --ranking field_sum,
               the key is rounded to D decimals before ranking, so sums of the same match-rate
               class that differ only by float rounding tie (run H A0: 7 float values, 5
               classes). The declared tie-break then orders the class. On: enters config_id.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

# Pool record schema: defined ONCE, by the grader that reads it
# (scripts/diagnostics/e2e_grader.py in this repo, INPUT FORMATS, SCHEMA_POOL,
# resolve_records). H6 writes exactly those fields and nothing else into pools.jsonl:
#   schema, arm_id, config_id, dataset_sha256, data_cutoff_index,
#   target{combined_index, date, session}, pool, pool_sha256, abstain, abstain_reason
# (abstain_reason only when abstain is true). Everything else H6 knows per draw goes to
# vote_diagnostics.jsonl, which the grader never reads. Lead decision, 2026-11-23.
FORMAT_ID = "tfm.e2e.pool.v1"
DIAG_FORMAT_ID = "tfm.e2e.vote_diagnostics.v1"
RECORD_FORMAT_ID = "tfm_e2e_vote_run_record_v1"
HARNESS_VERSION = "0.1.0-draft"

REPO_ROOT = Path(__file__).resolve().parents[2]

RANKING_SOURCES = ("sidecar", "file_order", "seed_asc", "field_sum")
TIE_BREAKS = ("module", "seed_asc", "file_order")
MISSING_FEATURE_RULES = ("fail", "record")
VOTE_ARMS = ("step6_baseline", "per_skip", "midday_stream_index")
RANKED_FORMAT_ID = "tfm.e2e.ranked_predictions.v1"
EMIT_RANKED = ("off", "on")
DRAW_ID_RULES = ("c13_fallback",)   # chapter_13_diagnostics.py:163: draw_<len(draws)>
VALUE_ORDERINGS = ("off", "full_vote")
FULL_VOTE_RANK_BASIS = ("full_vote_first_rank_v1: value v scores -(rank of the first survivor, in "
                        "Step 6 ranking order, whose vote seq[t] equals v), over all survivors; "
                        "values no survivor votes tie at -(n_survivors+1). Its top distinct "
                        "values are the pool's distinct values in first-position order.")

# Expected parameter lists of the production callables (CLAUDE.md §1.7). Checked with
# inspect.signature at start-up. Any drift stops the run.
EXPECTED_SIGNATURES = {
    "PredictionGenerator.__init__": ["self", "config", "logger"],
    "PredictionGenerator.generate_predictions": [
        "self", "survivors_forward", "lottery_history", "survivors_reverse", "k"],
    "PredictionGenerator._build_prediction_pool": [
        "self", "survivors", "lottery_history", "pool_size"],
    "PredictionGenerator._get_survivor_skip": ["self", "survivor", "fallback_skip"],
    "PredictionConfig.from_json": ["path"],
    "load_survivors": ["path", "allow_fallback", "return_format", "logger"],
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
class HarnessError(RuntimeError):
    """Fail-closed stop. The message says which input or invariant failed."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def pool_sha256(pool):
    """The grader's definition (e2e_grader.pool_sha256): sha256 of canonical_json(raw pool)
    as ASCII, order and repeats kept."""
    return hashlib.sha256(canonical_json(pool).encode("ascii")).hexdigest()


def peak_rss_mib():
    self_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return {"self_peak_rss_mib": round(self_kb / 1024.0, 1),
            "largest_child_peak_rss_mib": round(child_kb / 1024.0, 1)}


def git_head():
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# input loaders (mirror the CLI's rules; TODO-E2E-V2)
# ---------------------------------------------------------------------------
def load_survivor_rows(path):
    """Same unwrap rule as prediction_generator.py:1042-1044, then strict validation."""
    with open(path, "r") as f:
        data = json.load(f)
    rows = data if isinstance(data, list) else data.get("survivors", data)
    if not isinstance(rows, list) or not rows:
        raise HarnessError(f"survivors file {path}: expected a non-empty list of rows")
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise HarnessError(f"survivors row {i}: type {type(r).__name__}, expected dict "
                               "(the module would drop or re-featurise it, :785-793)")
        if "seed" not in r:
            raise HarnessError(f"survivors row {i}: no 'seed'")
        if not isinstance(r.get("features"), dict):
            raise HarnessError(f"survivors row {i} (seed {r.get('seed')}): no 'features' dict")
        try:
            int(r["seed"])
        except (TypeError, ValueError):
            raise HarnessError(f"survivors row {i}: seed {r['seed']!r} is not an integer")
    return rows


def load_history_file(path):
    """Same rule as prediction_generator.py:1052-1056."""
    with open(path, "r") as f:
        data = json.load(f)
    hist = data if isinstance(data, list) else data.get("draws", data)
    if hist and isinstance(hist[0], dict):
        hist = [d["draw"] for d in hist]
    if not isinstance(hist, list):
        raise HarnessError(f"history file {path}: not a list after unwrapping")
    for i, v in enumerate(hist):
        if isinstance(v, bool) or not isinstance(v, int):
            raise HarnessError(f"history file {path} entry {i}: {v!r} is not an int")
    return hist


class _SealedValue:
    """Stands in for a dataset `draw` the harness must not hold (row index >= --seal-from)."""
    __slots__ = ()

    def __repr__(self):
        return "<sealed>"


SEALED = _SealedValue()

# Sealed-history state for the current main() call (fork-inherited by workers). TODO-E2E-V10.
_SEAL = {"from": None, "n_rows": None}


def load_dataset(path, seal_from=None):
    """The frozen dataset the grader binds by sha256: a list of {date, session, draw}; the row
    index is the combined index. Returns ([(date, session)], [draw]).

    seal_from (TODO-E2E-V10): rows at index >= seal_from are sealed. The JSON decoder hands
    each row's key/value pairs to the hook below in document order; the hook replaces the
    `draw` value of a sealed row by SEALED before the row dict exists. Nothing stores,
    validates, compares or returns a sealed draw value. The returned draws list then has
    length seal_from exactly."""
    counter = [0]

    def hook(pairs):
        keys = {k for k, _ in pairs}
        if {"date", "session"} <= keys:
            idx = counter[0]
            counter[0] += 1
            if seal_from is not None and idx >= seal_from:
                return {k: (SEALED if k == "draw" else v) for k, v in pairs}
        return dict(pairs)

    with open(path, "r") as f:
        rows = json.load(f, object_pairs_hook=hook)
    if not isinstance(rows, list) or not rows:
        raise HarnessError(f"dataset {path}: expected a non-empty list of {{date, session, draw}}")
    if counter[0] != len(rows):
        raise HarnessError(f"dataset {path}: {counter[0]} row objects seen by the parse hook, "
                           f"{len(rows)} rows (nested objects?); the seal cannot be trusted")
    if seal_from is not None and not (0 < seal_from <= len(rows)):
        raise HarnessError(f"--seal-from {seal_from} outside (0, {len(rows)}]")
    ident, draws = [], []
    for i, r in enumerate(rows):
        if not isinstance(r, dict) or not {"date", "session", "draw"} <= set(r):
            raise HarnessError(f"dataset row {i}: needs date, session, draw")
        if not isinstance(r["date"], str) or not isinstance(r["session"], str):
            raise HarnessError(f"dataset row {i}: date and session must be strings")
        ident.append((r["date"], r["session"]))
        if seal_from is not None and i >= seal_from:
            if r["draw"] is not SEALED:
                raise HarnessError(f"dataset row {i}: sealed but the draw value was kept")
            continue
        v = r["draw"]
        if isinstance(v, bool) or not isinstance(v, int):
            raise HarnessError(f"dataset row {i}: draw {v!r} is not an int")
        draws.append(v)
    return ident, draws


def history_for(combined, vi):
    """The history list handed to the module for vote index vi. Unsealed: combined[:vi].
    Sealed (TODO-E2E-V10): past the seal the list is padded with None up to length vi. For
    dict survivors the module reads only len(lottery_history) (pg:839; the only other use,
    pg:788, is for int survivors, which load_survivor_rows refuses), so the pool is unchanged
    (self-test G14). A None read by any future module code raises instead of computing."""
    if vi <= len(combined):
        return combined[:vi]
    if _SEAL["from"] is None or vi > _SEAL["n_rows"]:
        raise HarnessError(f"vote index {vi} past the history ({len(combined)}) and not sealed")
    return combined + [None] * (vi - len(combined))


def check_aligned(combined, draws):
    """--history files must equal the dataset's draws value for value, or the run stops."""
    if len(combined) != len(draws):
        raise HarnessError(f"history has {len(combined)} draws, dataset has {len(draws)} "
                           "(the files are not aligned)")
    for i, (a, b) in enumerate(zip(combined, draws)):
        if a != b:
            raise HarnessError(f"history[{i}] = {a} != dataset draw {b} (the files are not aligned)")


# ---------------------------------------------------------------------------
# ranking: computed once, proven on every draw
# ---------------------------------------------------------------------------
class RankOnce:
    """Wraps the object at PredictionGenerator.model (used at :827-828).

    The first predict(X) runs the inner model. Later calls must present a byte-identical X
    and get a copy of the same score array. A different X raises HarnessError.
    """

    def __init__(self, inner, secondary=None):
        self.inner = inner
        # tie-break (declared, default None = the module's own argsort decides ties, :836).
        # With a secondary key, the scores are replaced by distinct rank positions:
        # primary descending, then secondary ascending. The module's argsort then yields
        # exactly that order.
        self.secondary = secondary
        self.x_sha256 = None
        self.x_shape = None
        self.scores = None
        self.raw_scores = None
        self.inner_calls = 0
        self.cached_calls = 0

    @staticmethod
    def _x_hash(X):
        import numpy as np
        X = np.ascontiguousarray(X)
        h = hashlib.sha256()
        h.update(str(X.dtype).encode())
        h.update(str(X.shape).encode())
        h.update(X.tobytes())
        return h.hexdigest()

    def predict(self, X):
        import numpy as np
        hx = self._x_hash(X)
        if self.x_sha256 is None:
            scores = np.asarray(self.inner.predict(X))
            if scores.shape[0] != X.shape[0]:
                raise HarnessError(f"model returned {scores.shape[0]} scores for {X.shape[0]} rows")
            self.raw_scores = scores.copy()
            if self.secondary is not None:
                sec = np.asarray(self.secondary, dtype=np.float64)
                if sec.shape[0] != scores.shape[0]:
                    raise HarnessError("tie-break key length != rows")
                order = np.lexsort((sec, -scores.astype(np.float64)))
                pos = np.empty(len(order), dtype=np.float64)
                pos[order] = -np.arange(len(order), dtype=np.float64)
                scores = pos
            self.scores = scores.copy()
            self.scores.setflags(write=False)
            self.x_sha256 = hx
            self.x_shape = list(X.shape)
            self.inner_calls += 1
            return self.scores.copy()
        if hx != self.x_sha256:
            raise HarnessError("ranking input X changed between draws "
                               f"({hx[:12]} != {self.x_sha256[:12]}): the ranking is not fixed")
        self.cached_calls += 1
        return self.scores.copy()

    def scores_sha256(self):
        if self.scores is None:
            return None
        h = hashlib.sha256()
        h.update(str(self.scores.dtype).encode())
        h.update(self.scores.tobytes())
        return h.hexdigest()

    def tie_report(self, pool_size):
        """Diagnostics only, from the raw (pre-tie-break) scores. Never used for a pool."""
        import numpy as np
        if self.raw_scores is None:
            return None
        s = np.sort(self.raw_scores.astype(np.float64))[::-1]
        n = len(s)
        rep = {"rows": int(n), "distinct_scores": int(len(np.unique(s))),
               "pool_size": int(pool_size), "boundary_tie": False}
        if 0 < pool_size < n:
            b = s[pool_size - 1]
            if s[pool_size] == b:
                grp = int(np.sum(s == b))
                inside = int(np.sum(s[:pool_size] == b))
                rep.update(boundary_tie=True, boundary_score=float(b),
                           boundary_group_size=grp, boundary_group_in_pool=inside,
                           pool_slots_set_by_tie_order=inside)
        return rep


class KeyRanker:
    """Unlearned ranking: returns a declared per-row key, aligned with survivor order.
    X row r is survivor r (the harness rejects rows the module would skip), so it ignores
    X's columns and checks only the row count."""

    def __init__(self, key_values):
        import numpy as np
        self.key = np.asarray(key_values, dtype=np.float64)

    def predict(self, X):
        if X.shape[0] != self.key.shape[0]:
            raise HarnessError(f"KeyRanker: X has {X.shape[0]} rows, key has {self.key.shape[0]}")
        return self.key.copy()


def _lookup_field(row, name, npz_map):
    for container in (row, row.get("metadata") or {}, row.get("features") or {}):
        if name in container:
            return container[name], "row"
    if npz_map is not None and name in npz_map:
        seed = int(row["seed"])
        if seed in npz_map[name]:
            return npz_map[name][seed], "npz"
    raise HarnessError(f"ranking field '{name}' missing for seed {row.get('seed')} "
                       "(looked in row, metadata, features, --ranking-npz)")


def _load_npz_fields(path, fields):
    """Attach Step 1 fields by seed, using utils.survivor_loader.load_survivors (module code)."""
    from utils.survivor_loader import load_survivors
    sd = load_survivors(path, allow_fallback=False, return_format="array")
    if sd.format != "npz":
        raise HarnessError(f"--ranking-npz {path}: loaded format {sd.format}, expected npz")
    arrays = sd.data
    if "seeds" not in arrays:
        raise HarnessError(f"--ranking-npz {path}: no 'seeds' array")
    seeds = arrays["seeds"]
    out = {}
    for name in fields:
        if name not in arrays:
            continue
        m = {}
        for s, v in zip(seeds.tolist(), arrays[name].tolist()):
            s = int(s)
            if s in m and m[s] != v:
                raise HarnessError(f"--ranking-npz: seed {s} has conflicting '{name}' values "
                                   f"{m[s]} and {v}")
            m[s] = v
        out[name] = m
    return out, sd.count


def build_key(rows, ranking, fields, npz_map, round_decimals=None):
    """Higher key = ranked first (the module sorts descending, :836). round_decimals
    (TODO-E2E-V13, field_sum only, default None = off) rounds each row's sum."""
    import math
    if ranking == "file_order":
        return [-float(i) for i in range(len(rows))], {"rule": "key = -row_index"}
    if ranking == "seed_asc":
        return [-float(int(r["seed"])) for r in rows], {"rule": "key = -seed"}
    if ranking == "field_sum":
        if not fields:
            raise HarnessError("--ranking field_sum needs --ranking-fields")
        key, sources = [], {}
        for r in rows:
            total = 0.0
            for name in fields:
                v, src = _lookup_field(r, name, npz_map)
                sources[f"{name}:{src}"] = sources.get(f"{name}:{src}", 0) + 1
                fv = float(v)
                if not math.isfinite(fv):
                    raise HarnessError(f"ranking field '{name}' not finite for seed {r['seed']}")
                total += fv
            key.append(total)
        info = {"rule": "key = sum(" + "+".join(fields) + ")", "field_sources": sources}
        if round_decimals is not None:
            raw_distinct = len(set(key))
            key = [round(k, round_decimals) for k in key]
            info.update(rule=info["rule"] + f", rounded to {round_decimals} decimals",
                        round_decimals=round_decimals, distinct_before_round=raw_distinct,
                        distinct_after_round=len(set(key)))
        return key, info
    raise HarnessError(f"build_key: ranking {ranking!r} is not an unlearned key")


# ---------------------------------------------------------------------------
# vote-position arms
# ---------------------------------------------------------------------------
def vote_index_for(arm, t, ctx):
    """Map a combined draw index t to the index Step 6 votes at (len(history passed)).

    step6_baseline       next_idx = t (Step 6's own convention, :839; TODO-E2E-12).
    per_skip             next_idx = t. Skip is overridden per survivor (see survivors_for_arm).
    midday_stream_index  HOOK ONLY (TODO-E2E-V3). Its law is [UNVERIFIED] (Ch7 O3, WP-1 Stage B).
    """
    if arm in ("step6_baseline", "per_skip"):
        return t
    if arm == "midday_stream_index":
        raise NotImplementedError(
            "vote arm midday_stream_index is a declared hook only: the combined-index to "
            "midday-stream law is not fixed (Ch7 O3, WP-1 Stage B). Implement here, then test.")
    raise HarnessError(f"unknown vote arm {arm!r}")


def survivors_for_arm(arm, rows, arm_skip):
    """per_skip: shallow-copy each row with a top-level 'skip', which
    PredictionGenerator._get_survivor_skip reads first (:539-541). The features are not
    touched, so X and the ranking stay the same (RankOnce proves this)."""
    if arm != "per_skip":
        return rows, {}
    if arm_skip is None or arm_skip < 0:
        raise HarnessError("--vote-arm per_skip needs --arm-skip >= 0")
    overridden = sum(1 for r in rows if "skip" in r)
    return [dict(r, skip=int(arm_skip)) for r in rows], {
        "arm_skip": int(arm_skip), "rows_with_prior_top_level_skip": overridden}


# ---------------------------------------------------------------------------
# generator construction
# ---------------------------------------------------------------------------
def check_signatures(pg, survivor_loader_mod):
    import inspect
    got = {
        "PredictionGenerator.__init__": pg.PredictionGenerator.__init__,
        "PredictionGenerator.generate_predictions": pg.PredictionGenerator.generate_predictions,
        "PredictionGenerator._build_prediction_pool": pg.PredictionGenerator._build_prediction_pool,
        "PredictionGenerator._get_survivor_skip": pg.PredictionGenerator._get_survivor_skip,
        "PredictionConfig.from_json": pg.PredictionConfig.from_json,
        "load_survivors": survivor_loader_mod.load_survivors,
    }
    out = {}
    for name, fn in got.items():
        params = list(inspect.signature(fn).parameters)
        if params != EXPECTED_SIGNATURES[name]:
            raise HarnessError(f"signature drift: {name}{params} != {EXPECTED_SIGNATURES[name]}")
        out[name] = str(inspect.signature(fn))
    return out


CAPS_PATH = Path(__file__).resolve().parent / "e2e_pipeline_caps.py"


def ll1_check(config_path, root=None) -> dict:
    """LL-1 capability check (owner TASK_LAND_WF_VOTE_20261125_ADDENDUM_1, lead ADDENDUM_2).
    A --config whose prediction section sets schema_check to anything but 'off' needs the LL-1
    switch in the prediction_generator.py this harness imports (REPO_ROOT, :131). Main's
    PredictionConfig.from_json drops an unknown key silently, so without this check such a run
    would go ahead unchecked under the config's label. Read as source by e2e_pipeline_caps.py
    (nothing imported); called before any work. Refuses with HarnessError (exit 2)."""
    root = Path(root) if root is not None else REPO_ROOT
    if not CAPS_PATH.is_file():
        try:
            cd = json.load(open(config_path))
            cd = cd.get("prediction", cd) if isinstance(cd, dict) else {}
            req = cd.get("schema_check", "off") if isinstance(cd, dict) else "off"
        except (OSError, ValueError):
            req = "off"
        if req not in (None, "off"):
            raise HarnessError(f"LL-1 switch needed by {{'schema_check': {req!r}}} cannot be checked: "
                               f"{CAPS_PATH} missing; nothing run")
        return {"requested": {}, "pipeline_has_switch": None, "files_checked": [],
                "applied": "default (switch not used)", "note": f"{CAPS_PATH.name} absent"}
    spec = importlib.util.spec_from_file_location("e2e_pipeline_caps", CAPS_PATH)
    caps = importlib.util.module_from_spec(spec)
    _dwb, sys.dont_write_bytecode = sys.dont_write_bytecode, True     # no __pycache__ beside it
    try:
        spec.loader.exec_module(caps)
    finally:
        sys.dont_write_bytecode = _dwb
    try:
        return caps.check({"LL-1": caps.requested_ll1(config_path)},
                          {"LL-1": root, "LL-2": root, "LL-3": root})["LL-1"]
    except caps.MissingSwitch as e:
        raise HarnessError(f"pipeline switch missing, nothing run: {e}")


def build_config(pg, args):
    """PredictionConfig.from_json, then the CLI's own override order (:1018-1029)."""
    if not Path(args.config).is_file():
        raise HarnessError(f"--config {args.config} not found (the CLI would silently use "
                           "PredictionConfig() defaults, :1012-1016; the harness refuses)")
    config = pg.PredictionConfig.from_json(args.config)
    if args.k is not None:
        config.k = args.k
    if args.pool_size is not None:
        config.pool_size = args.pool_size
    if args.log_level is not None:
        config.log_level = args.log_level
    if args.models_dir:
        config.models_dir = args.models_dir
    if args.no_require_schema:
        config.require_feature_schema = False
    # Harness-forced settings, recorded in the run record:
    config.use_dual_sieve = False      # TODO-E2E-9: dual-sieve path left untouched
    config.save_predictions = False    # no writes to ./predictions; the harness writes out-dir
    return config


def _construct_unlearned(pg, config, logger):
    """TODO-E2E-V1: there is no sidecar-free constructor (:288-294 raise without a sidecar).
    MULTI_MODEL_AVAILABLE is patched to False for __init__ only, then restored."""
    saved = pg.MULTI_MODEL_AVAILABLE
    pg.MULTI_MODEL_AVAILABLE = False
    try:
        gen = pg.PredictionGenerator(config, logger)
    finally:
        pg.MULTI_MODEL_AVAILABLE = saved
    return gen


def tie_break_key(rule, rows):
    if rule == "module":
        return None
    if rule == "seed_asc":
        return [float(int(r["seed"])) for r in rows]
    if rule == "file_order":
        return [float(i) for i in range(len(rows))]
    raise HarnessError(f"unknown tie-break {rule!r}")


def missing_schema_features(rows, names):
    """Rows lacking a schema column. The module zero-fills these silently (:798, Ch6 B5)."""
    per_name, n_rows = {}, 0
    for r in rows:
        f = r["features"]
        miss = [nm for nm in names if nm not in f]
        if miss:
            n_rows += 1
            for nm in miss:
                per_name[nm] = per_name.get(nm, 0) + 1
    return {"rows_with_missing": n_rows, "per_feature": per_name}


def check_nn_normalisation(gen):
    """Lead rule 2026-11-23 (TODO-E2E-V9). A neural_net sidecar is accepted only if:
      - the generator loaded the input scaler;
      - the scaler length equals the schema length;
      - the wrapper loaded y_mean and y_std (neural_net_wrapper.py:437-441). Without them
        predict() returns normalised outputs, with no inverse transform (:357-359).
    [1b, 2026-11-26] Step 6 now refuses the same cases itself
    (prediction_generator.py, _require_nn_normalisation), so build_generator normally stops
    before this is reached. It stays as a second line (a prediction_generator.py without 1b)
    and it writes the nn_normalisation report."""
    import math
    import numpy as np
    names = gen.feature_names or []
    m, sc = getattr(gen, "_scaler_mean", None), getattr(gen, "_scaler_scale", None)
    normalize = bool(getattr(gen, "_normalize_features", False))
    ym = getattr(gen.model, "_y_mean", None)
    ys = getattr(gen.model, "_y_std", None)
    problems = []
    if not normalize or m is None or sc is None:
        problems.append("input scaler not loaded (prediction_generator.py:385-449 proceeds "
                        "without it, :814-824)")
    elif len(m) != len(names) or len(sc) != len(names):
        problems.append(f"scaler length {len(m)}/{len(sc)} != schema {len(names)} "
                        "(prediction_generator.py:817-824 would skip it)")
    elif not (np.all(np.isfinite(m)) and np.all(np.isfinite(sc))):
        problems.append("scaler arrays not finite")
    if ym is None or ys is None:
        problems.append("wrapper y_mean/y_std not loaded (neural_net_wrapper.py:437-441; "
                        "predict() would return normalised outputs, :357-359)")
    elif not (math.isfinite(float(ym)) and math.isfinite(float(ys)) and float(ys) > 0):
        problems.append(f"y_mean/y_std not usable: {ym!r}/{ys!r}")
    report = {"normalize_features": normalize,
              "scaler_len": None if m is None else int(len(m)), "schema_len": len(names),
              "y_mean": None if ym is None else float(ym),
              "y_std": None if ys is None else float(ys),
              "device": str(getattr(gen.model, "device", None))}
    if problems:
        raise HarnessError("neural_net sidecar refused: " + "; ".join(problems))
    return report


def build_generator(pg, config, logger, args, rows, npz_map):
    info = {"ranking": args.ranking, "tie_break": args.tie_break}
    secondary = tie_break_key(args.tie_break, rows)
    if args.ranking == "sidecar":
        if not pg.MULTI_MODEL_AVAILABLE:
            raise HarnessError("models.model_factory not importable: no sidecar ranking possible")
        try:
            gen = pg.PredictionGenerator(config, logger)
        except getattr(pg, "NNNormalisationError", ()) as e:
            # [1b] Step 6 itself now refuses a neural_net without its training normalisation
            raise HarnessError("neural_net sidecar refused by Step 6: " + " ".join(str(e).splitlines()[:2]))
        meta_path = Path(config.models_dir) / "best_model.meta.json"
        info["sidecar_path"] = str(meta_path)
        info["sidecar_sha256"] = sha256_file(meta_path)
        if gen._signal_gate_blocked:
            info["signal_gate_blocked"] = True
            info["model_type"] = (gen.model_meta or {}).get("model_type")
            return gen, None, info
        if gen.model is None:
            raise HarnessError("sidecar ranking: no model loaded; the module would rank by the "
                               "feature mean and report success (:831-833, Ch7 B2). Refused.")
        ck = gen.model_checkpoint_path
        ck_res = ck if ck and Path(ck).exists() else str(Path(config.models_dir) / Path(ck).name)
        info["checkpoint_path"] = ck_res
        info["checkpoint_sha256"] = sha256_file(ck_res) if Path(ck_res).exists() else None
        info["model_type"] = gen.model_meta.get("model_type")
        info["feature_names_count"] = len(gen.feature_names or [])
        info["feature_names_sha256"] = sha256_text(canonical_json(gen.feature_names or []))
        info["nn_scaler_applied"] = bool(getattr(gen, "_normalize_features", False))
        info["signal_gate_blocked"] = False
        if info["model_type"] == "neural_net":
            info["nn_normalisation"] = check_nn_normalisation(gen)
        if gen.feature_names:
            miss = missing_schema_features(rows, gen.feature_names)
            info["missing_schema_features"] = miss
            if miss["rows_with_missing"] and args.on_missing_feature == "fail":
                raise HarnessError(
                    f"{miss['rows_with_missing']} rows lack schema features "
                    f"{sorted(miss['per_feature'])[:5]}...; the module would zero-fill them "
                    "(:798, Ch6 B5). Use --on-missing-feature record to proceed as the CLI does.")
        rank = RankOnce(gen.model, secondary)
        gen.model = rank
        return gen, rank, info
    # unlearned arms
    if args.models_dir_given:
        raise HarnessError("--models-dir is meaningless for an unlearned ranking; drop it")
    config.require_feature_schema = False  # no schema exists; X columns are unused by KeyRanker
    gen = _construct_unlearned(pg, config, logger)
    key, key_info = build_key(rows, args.ranking, args.ranking_fields, npz_map,
                              getattr(args, "ranking_key_round_decimals", None))
    info.update(key_info)
    info["model_type"] = f"unlearned:{args.ranking}"
    gen.model_meta = {"model_type": info["model_type"],
                      "signal_quality": {"prediction_allowed": True,
                                         "signal_status": "unlearned_arm"}}
    rank = RankOnce(KeyRanker(key), secondary)
    gen.model = rank
    info["signal_gate_blocked"] = False
    return gen, rank, info


# ---------------------------------------------------------------------------
# per-draw vote
# ---------------------------------------------------------------------------
_W = {}  # fork-inherited worker state


def vote_one(gen, survivors, combined, t, arm, k):
    vi = vote_index_for(arm, t, None)
    limit = len(combined) if _SEAL["from"] is None else _SEAL["n_rows"]
    if vi < 0 or vi > limit:
        raise HarnessError(f"vote index {vi} for t={t} outside [0, {limit}]")
    result = gen.generate_predictions(survivors, history_for(combined, vi), None, k=k)
    status = result.get("status")
    preds = result.get("predictions", [])
    if status in ("prediction_skipped", "PREDICTION_SKIPPED"):
        return vi, "abstain", f"signal_gate:{result.get('reason')}", [], [], None
    if status == "success":
        if not preds:
            # written as an empty pool, not an abstention: the grader's declared
            # empty_pool rule decides (e2e_grader.parse_abstention)
            return vi, "empty", "empty_pool", [], [], None
        return (vi, "pool", None, [int(p) for p in preds],
                [float(x) for x in result.get("raw_scores", [])],
                result.get("metadata", {}).get("model_type"))
    if status is None and not preds and "error" in result.get("metadata", {}):
        # :631-640, no predictions generated (the module returns no status)
        return vi, "empty", "no_predictions:" + str(result["metadata"]["error"]), [], [], None
    raise HarnessError(f"t={t}: unknown result status {status!r}")


def target_for(t, ctx):
    if t < len(ctx["identity"]):
        d, s_ = ctx["identity"][t]
    else:  # pending: the draw after the dataset; date/session are declared arguments
        d, s_ = ctx["pending_date"], ctx["pending_session"]
    return {"combined_index": t, "date": d, "session": s_}


def cutoff_for(t, ctx):
    return t if ctx["data_cutoff"] == "t" else ctx["data_cutoff"]


def make_line(t, i, vi, status, reason, pool, raw_scores, model_type, ctx):
    """Returns {"pool": <tfm.e2e.pool.v1 record>, "diag": <diagnostics record>}."""
    abstain = status == "abstain"
    rec = {
        "schema": FORMAT_ID,
        "arm_id": ctx["arm_id"],
        "config_id": ctx["config_id"],
        "dataset_sha256": ctx["dataset_sha256"],
        "data_cutoff_index": cutoff_for(t, ctx),
        "target": target_for(t, ctx),
        "pool": pool,
        "pool_sha256": pool_sha256(pool),
        "abstain": abstain,
    }
    if abstain:
        rec["abstain_reason"] = reason
    diag = {
        "schema": DIAG_FORMAT_ID,
        "arm_id": ctx["arm_id"],
        "combined_index": t,
        "block_pos": i,
        "vote_index": vi,
        "pending": t >= ctx["history_len"],
        "vote_status": status,          # pool | abstain | empty
        "note": reason,                 # abstain reason, or why the pool is empty
        "pool_len": len(pool),
        "raw_scores": raw_scores,
        "k_requested": ctx["k"],
        "pool_size_requested": ctx["pool_size"],
        "model_type": model_type,
    }
    out = {"pool": rec, "diag": diag}
    if ctx.get("ranked_spec") is not None:
        out["ranked"] = ranked_line(t, status, pool, raw_scores, rec, ctx)
    return out


# ---------------------------------------------------------------------------
# LL-5: ranked predictions (TODO-E2E-V11), default off
# ---------------------------------------------------------------------------
def draw_id_for(t, rule):
    """chapter_13_diagnostics.py:163: the latest draw's id is draw_<len(draws)>; the draw at
    combined index t is the latest once len(draws) = t + 1."""
    if rule == "c13_fallback":
        return f"draw_{t + 1}"
    raise HarnessError(f"unknown draw-id rule {rule!r}")


def ranked_order(scores):
    """The module's own ranking line (prediction_generator.py:836), on the RankOnce scores."""
    import numpy as np
    return np.argsort(scores)[::-1]


def ranked_line(t, status, pool, raw_scores, rec, ctx):
    import math
    spec = ctx["ranked_spec"]
    rank = spec["rank"]
    if spec.get("order") is None:          # computed once per process, after the first draw
        spec["order"] = ranked_order(rank.scores)
        spec["model_id"] = {"model_type": spec["model_type"],
                            "checkpoint_sha256": spec["checkpoint_sha256"],
                            "scores_sha256": rank.scores_sha256()}
    obj = {"schema": RANKED_FORMAT_ID, "draw_id": draw_id_for(t, spec["draw_id_rule"]),
           "combined_index": t, "target": rec["target"], "arm_id": rec["arm_id"],
           "config_id": rec["config_id"], "model_id": spec["model_id"],
           "feature_schema_sha256": spec["names_sha256"],
           "feature_names_source": spec["names_source"], "vote_status": status,
           "predictions": []}
    if status != "pool":
        return obj
    depth = spec["depth"]
    if len(pool) < depth or len(raw_scores) < depth:
        raise HarnessError(f"t={t}: pool has {len(pool)} votes, ranked depth {depth} declared")
    order, rows, names = spec["order"], spec["rows"], spec["names"]
    preds = []
    for i in range(depth):
        r = int(order[i])
        want = float(rank.scores[r])
        if raw_scores[i] != want:
            raise HarnessError(f"t={t} slot {i}: raw score {raw_scores[i]} != score {want} of "
                               f"ranked survivor row {r}; the slot-to-survivor map is wrong")
        feats = rows[r].get("features", {})
        fmap = {}
        for n in names:
            v = float(feats.get(n, 0.0))      # the value the module put in X (:799)
            if not math.isfinite(v):
                raise HarnessError(f"t={t}: feature {n} of seed {rows[r]['seed']} not finite")
            fmap[n] = v
        preds.append({"rank": i + 1, "seed": int(rows[r]["seed"]), "row_index": r,
                      "prediction": int(pool[i]), "score": want, "features": fmap})
    obj["predictions"] = preds
    return obj


# ---------------------------------------------------------------------------
# full-vote value ordering (TODO-E2E-V12), default off
# ---------------------------------------------------------------------------
_VW = {}


def _vote_rows(chunk):
    import numpy as np
    sc, rows, skips, vis = _VW["scorer"], _VW["rows"], _VW["skips"], _VW["vis"]
    n_len = int(vis.max()) + 1
    out = np.empty((len(chunk), len(vis)), dtype=np.int16)
    for j, r in enumerate(chunk):
        seq = sc._generate_sequence(int(rows[r]["seed"]), n_len, skip=skips[r])
        if len(seq) < n_len:
            raise HarnessError(f"sequence for survivor row {r} shorter than {n_len}")
        out[j] = seq[vis]
    return chunk[0], out


def full_vote_matrix(gen, survivors, default_skip, vis, workers):
    """V[row, j] = seq[vis[j]] of survivor row's own sequence, by the module's generator and
    the module's per-survivor skip. Rows in survivor (file) order."""
    import multiprocessing as mp
    import numpy as np
    n = len(survivors)
    vis = np.asarray(vis, dtype=np.int64)
    skips = [gen._get_survivor_skip(r, default_skip) for r in survivors]
    V = np.empty((n, len(vis)), dtype=np.int16)
    _VW.update(scorer=gen.scorer, rows=survivors, skips=skips, vis=vis)
    idx = list(range(n))
    size = max(1, -(-n // (max(workers, 1) * 8)))
    chunks = [idx[i:i + size] for i in range(0, n, size)]
    if workers <= 1:
        for start, block in map(_vote_rows, chunks):
            V[start:start + len(block)] = block
    else:
        with mp.get_context("fork").Pool(workers) as p:
            for start, block in p.imap_unordered(_vote_rows, chunks):
                V[start:start + len(block)] = block
    return V


def value_scores_for(col_ranked, n_values):
    """R24 value_scores from one draw's votes in ranking order: -(first rank) per value,
    -(n+1) for values nobody votes."""
    import numpy as np
    n = len(col_ranked)
    vals, first = np.unique(col_ranked, return_index=True)
    s = np.full(n_values, -(n + 1), dtype=np.int64)
    if len(vals) and (vals.min() < 0 or vals.max() >= n_values):
        raise HarnessError(f"vote value outside 0..{n_values - 1}")
    s[vals] = -(first + 1)
    return s


def attach_value_orderings(lines, gen, rank, survivors, config, workers):
    """Adds value_scores + rank_basis to every pool line; proves the ordering against the
    module's pool on every draw. Returns the record block."""
    import numpy as np
    t0 = time.time()
    pool_lines = [ln for ln in lines if ln["diag"]["vote_status"] == "pool"]
    if not pool_lines:
        return {"draws_with_ordering": 0, "seconds": 0.0}
    vis = [ln["diag"]["vote_index"] for ln in pool_lines]
    V = full_vote_matrix(gen, survivors, config.default_skip, vis, workers)
    t_matrix = time.time() - t0
    order = np.asarray(ranked_order(rank.scores))
    Vr = V[order]
    n_values = int(config.mod)
    for j, ln in enumerate(pool_lines):
        rec = ln["pool"]
        pool = rec["pool"]
        col = Vr[:, j]
        if col[:len(pool)].astype(int).tolist() != pool:
            raise HarnessError(f"t={rec['target']['combined_index']}: full-vote matrix first "
                               f"{len(pool)} votes != the module's pool")
        s = value_scores_for(col, n_values)
        distinct = list(dict.fromkeys(pool))
        top = np.argsort(-s, kind="stable")[:len(distinct)].tolist()
        if top != distinct:
            raise HarnessError(f"t={rec['target']['combined_index']}: ordering top-k != pool's "
                               f"distinct values in first-position order")
        rec["value_scores"] = s.tolist()
        rec["rank_basis"] = FULL_VOTE_RANK_BASIS
    return {"draws_with_ordering": len(pool_lines), "survivors": int(V.shape[0]),
            "matrix_shape": list(V.shape), "max_vote_index": int(max(vis)),
            "matrix_seconds": round(t_matrix, 3), "seconds": round(time.time() - t0, 3),
            "workers": workers, "value_space": n_values, "format": "value_scores (R24)",
            "rank_basis": FULL_VOTE_RANK_BASIS,
            "per_draw_checks": ["first len(pool) ordering votes == pool",
                                "top distinct values == pool distinct values in order"]}


def _worker_chunk(chunk):
    gen, survivors, combined = _W["gen"], _W["survivors"], _W["combined"]
    rank, ctx = _W["rank"], _W["ctx"]
    inner_before = rank.inner_calls if rank else 0
    cached_before = rank.cached_calls if rank else 0
    lines = []
    for i, t in chunk:
        vi, st, rs, pool, raw, mt = vote_one(gen, survivors, combined, t, ctx["vote_arm"], ctx["k"])
        lines.append(make_line(t, i, vi, st, rs, pool, raw, mt, ctx))
    inner_after = rank.inner_calls if rank else 0
    cached_after = rank.cached_calls if rank else 0
    return lines, cached_after - cached_before, inner_after - inner_before


def run_block(gen, rank, survivors, combined, ts, ctx, workers):
    """Draw ts[0] runs in the parent and fixes the ranking. The rest run in forked workers,
    which inherit the cached scores and never call the model (asserted)."""
    import multiprocessing as mp
    lines = []
    i0, t0 = 0, ts[0]
    vi, st, rs, pool, raw, mt = vote_one(gen, survivors, combined, t0, ctx["vote_arm"], ctx["k"])
    lines.append(make_line(t0, i0, vi, st, rs, pool, raw, mt, ctx))
    rest = [(i, t) for i, t in enumerate(ts) if i > 0]
    child_cached, child_inner = 0, 0
    if rest:
        _W.update(gen=gen, survivors=survivors, combined=combined, rank=rank, ctx=ctx)
        if workers <= 1:
            out, c, n = _worker_chunk(rest)
            lines.extend(out)
            child_cached += c
            child_inner += n
        else:
            n_chunks = min(len(rest), workers * 4)
            size = -(-len(rest) // n_chunks)
            chunks = [rest[j:j + size] for j in range(0, len(rest), size)]
            with mp.get_context("fork").Pool(workers) as p:
                for out, c, n in p.imap(_worker_chunk, chunks):
                    lines.extend(out)
                    child_cached += c
                    child_inner += n
    if child_inner != 0:
        raise HarnessError(f"model was called {child_inner} more times after the first draw")
    return lines, child_cached


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="H6: run-scoped Step 6 vote over a block of draw indices (CPU, fail-closed).")
    p.add_argument("--survivors", required=True,
                   help="Step 3 survivors_with_scores.json (list of dicts with seed+features).")
    p.add_argument("--dataset", required=True,
                   help="Frozen dataset, list of {date, session, draw}; row index = combined "
                        "index. Its sha256 goes on every record (the grader binds it). Supplies "
                        "target.date/target.session.")
    p.add_argument("--history", default=None, nargs="+",
                   help="Optional history file(s) as Step 6 would read them, concatenated in "
                        "order (e.g. train_history.json holdout_history.json). Must equal the "
                        "dataset's draws value for value. Default: the dataset's draws.")
    p.add_argument("--seal-from", type=int, default=None,
                   help="Seal the dataset from this combined index on (TODO-E2E-V10): the "
                        "draw values of rows >= INDEX are discarded at parse time and never "
                        "held. --history must then cover exactly [0, INDEX). Votes past it get "
                        "a None-padded history of the right length (the module reads only its "
                        "length). Default: no seal.")
    p.add_argument("--expect-sidecar-sha256", default=None,
                   help="Fail closed unless the loaded best_model.meta.json has this sha256.")
    p.add_argument("--expect-checkpoint-sha256", default=None,
                   help="Fail closed unless the checkpoint the module loaded has this sha256.")
    p.add_argument("--data-cutoff-index", required=True,
                   help="Declared data_cutoff_index written on every record: an integer (must "
                        "be <= every target in the block) or the literal 't' (cutoff = the "
                        "target index, i.e. history[:t]). No default: it is a declaration.")
    p.add_argument("--pending-date", default=None,
                   help="target.date for t = len(dataset) (pending). Required if the block "
                        "includes it.")
    p.add_argument("--pending-session", default=None,
                   help="target.session for t = len(dataset) (pending). Required likewise.")
    p.add_argument("--block-start", type=int, required=True,
                   help="First combined index t (inclusive).")
    p.add_argument("--block-end", type=int, required=True,
                   help="End combined index (exclusive). May be len(dataset)+1, so the CLI's "
                        "own next-draw pool (t = len(dataset), pending) is included.")
    p.add_argument("--config", required=True,
                   help="Step 6 config JSON, as the CLI's --config. Must exist.")
    p.add_argument("--models-dir", default=None,
                   help="Directory holding best_model.meta.json (ranking=sidecar). "
                        "Default: the config's models_dir.")
    p.add_argument("--k", type=int, default=None,
                   help="Same as the CLI --k. Default: the config's k.")
    p.add_argument("--pool-size", type=int, default=None,
                   help="Same as the CLI --pool-size. Default: the config's pool_size.")
    p.add_argument("--no-require-schema", action="store_true",
                   help="Same as the CLI flag (UNSAFE in the module).")
    p.add_argument("--ranking", choices=RANKING_SOURCES, default="sidecar",
                   help="Ranking source. sidecar = learned Step 5 model (default). "
                        "file_order / seed_asc / field_sum = unlearned keys (arm A0).")
    p.add_argument("--ranking-fields", default=None,
                   help="Comma list for field_sum, e.g. forward_matches,reverse_matches.")
    p.add_argument("--ranking-npz", default=None,
                   help="Optional Step 1 NPZ. field_sum fields absent from the rows are read "
                        "from it by seed (utils.survivor_loader.load_survivors).")
    p.add_argument("--tie-break", choices=TIE_BREAKS, default="module",
                   help="How equal ranking scores are ordered. module = numpy argsort as "
                        "Step 6 does (:836), the default and the only CLI-identical choice. "
                        "seed_asc / file_order = declared secondary key.")
    p.add_argument("--on-missing-feature", choices=MISSING_FEATURE_RULES, default="fail",
                   help="Rows lacking a sidecar schema feature: fail (default) or record and "
                        "let the module zero-fill as the CLI does (:798).")
    p.add_argument("--vote-arm", choices=VOTE_ARMS, default="step6_baseline",
                   help="Vote-position arm. step6_baseline = Step 6 convention (default). "
                        "per_skip = every survivor votes at --arm-skip. "
                        "midday_stream_index = declared hook, not implemented.")
    p.add_argument("--arm-skip", type=int, default=None, help="Skip for --vote-arm per_skip.")
    p.add_argument("--arm-id", default=None,
                   help="Arm label written on every line. Default: <ranking>/<vote-arm>[/skipN].")
    p.add_argument("--emit-ranked-predictions", choices=EMIT_RANKED, default="off",
                   help="LL-5 (TODO-E2E-V11): off (default, today's behaviour) or on = also write "
                        "ranked_predictions.jsonl, one Chapter 13 reader-shaped object per target: "
                        "seed, rank, prediction, score, name-keyed feature map, draw_id, model id, "
                        "feature-schema hash. Pools and config_id are the same either way.")
    p.add_argument("--ranked-predictions-depth", type=int, default=20,
                   help="Ranked entries per target when on (default 20: Chapter 13 reads the "
                        "top 20, chapter_13_orchestrator.py:690). Must be <= k and <= pool size.")
    p.add_argument("--ranked-draw-id-rule", choices=DRAW_ID_RULES, default="c13_fallback",
                   help="draw_id per target (default c13_fallback = draw_<t+1>, "
                        "chapter_13_diagnostics.py:163).")
    p.add_argument("--emit-value-ordering", choices=VALUE_ORDERINGS, default="off",
                   help="TODO-E2E-V12: off (default) or full_vote = every pool record also carries "
                        "value_scores (grader R24): all values ordered by Step 6's vote over all "
                        "survivors, proven against the pool per draw. Enters config_id when on.")
    p.add_argument("--ranking-key-round-decimals", type=int, default=None,
                   help="TODO-E2E-V13: field_sum only. Round the key to D decimals before ranking "
                        "(default off). Enters config_id when given.")
    p.add_argument("--workers", type=int, default=1,
                   help="Forked worker processes for draws after the first (default 1).")
    p.add_argument("--out-dir", required=True, help="New directory. Must not exist.")
    p.add_argument("--allow-gpu", action="store_true",
                   help="Do not hide CUDA (default: CUDA_VISIBLE_DEVICES='' before imports).")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="WARNING",
                   help="Module log level (default WARNING; INFO logs every draw).")
    a = p.parse_args(argv)
    a.models_dir_given = a.models_dir is not None
    if a.data_cutoff_index != "t":
        try:
            a.data_cutoff_index = int(a.data_cutoff_index)
        except ValueError:
            raise HarnessError(f"--data-cutoff-index {a.data_cutoff_index!r}: integer or 't'")
    a.ranking_fields = [s.strip() for s in a.ranking_fields.split(",") if s.strip()] \
        if a.ranking_fields else None
    return a


def main(argv=None):
    t_start = time.time()
    args = parse_args(argv)
    if not args.allow_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise HarnessError(f"--out-dir {out_dir} exists; refusing to overwrite")
    if args.block_end <= args.block_start:
        raise HarnessError("empty block: --block-end must be > --block-start")
    if args.workers < 1:
        raise HarnessError("--workers must be >= 1")
    if args.ranking != "field_sum" and (args.ranking_fields or args.ranking_npz):
        raise HarnessError("--ranking-fields/--ranking-npz only apply to --ranking field_sum")
    if args.ranking_key_round_decimals is not None and (args.ranking != "field_sum"
                                                        or args.ranking_key_round_decimals < 0):
        raise HarnessError("--ranking-key-round-decimals needs --ranking field_sum and D >= 0")
    if args.vote_arm != "per_skip" and args.arm_skip is not None:
        raise HarnessError("--arm-skip only applies to --vote-arm per_skip")
    vote_index_for(args.vote_arm, args.block_start, None)  # hook arms fail here, before work
    if not Path(args.config).is_file():
        raise HarnessError(f"--config {args.config} not found")
    ll1 = ll1_check(args.config)    # LL-1 switch (ADDENDUM_1/2): refuses here, before any work

    sys.path.insert(0, str(REPO_ROOT))
    import logging
    import prediction_generator as pg
    import utils.survivor_loader as sl
    signatures = check_signatures(pg, sl)

    _SEAL.update({"from": None, "n_rows": None})
    if args.seal_from is not None and not args.history:
        raise HarnessError("--seal-from needs --history covering exactly [0, seal)")
    rows = load_survivor_rows(args.survivors)
    identity, ds_draws = load_dataset(args.dataset, args.seal_from)
    dataset_sha256 = sha256_file(args.dataset)
    n_ident = len(identity)
    combined, hist_meta = [], []
    if args.history:
        for hp in args.history:
            h = load_history_file(hp)
            hist_meta.append({"path": str(Path(hp).resolve()), "sha256": sha256_file(hp),
                              "len": len(h), "combined_start": len(combined)})
            combined.extend(h)
        check_aligned(combined, ds_draws)
    else:
        combined = list(ds_draws)
        hist_meta.append({"path": "<dataset draws>", "sha256": dataset_sha256,
                          "len": len(combined), "combined_start": 0})
    if args.seal_from is not None:
        _SEAL.update({"from": args.seal_from, "n_rows": n_ident})
        if args.data_cutoff_index != "t" and args.data_cutoff_index < min(args.block_start,
                                                                          args.seal_from):
            raise HarnessError(f"--data-cutoff-index {args.data_cutoff_index} is below the "
                               f"history the module is handed (min(block start, seal) = "
                               f"{min(args.block_start, args.seal_from)})")
    if args.block_start < 0 or args.block_end > n_ident + 1:
        raise HarnessError(f"block [{args.block_start}, {args.block_end}) outside "
                           f"[0, {n_ident + 1}) for a dataset of {n_ident} rows")
    if args.block_end > n_ident:  # the pending draw is in the block
        if not (isinstance(args.pending_date, str) and args.pending_date
                and isinstance(args.pending_session, str) and args.pending_session):
            raise HarnessError("the block includes the pending draw t = len(dataset); declare "
                               "--pending-date and --pending-session (the grader requires them)")
    elif args.pending_date or args.pending_session:
        raise HarnessError("--pending-date/--pending-session given but the block has no pending draw")
    if args.data_cutoff_index != "t":
        if args.data_cutoff_index < 0 or args.data_cutoff_index > args.block_start:
            raise HarnessError(f"--data-cutoff-index {args.data_cutoff_index} must be in "
                               f"[0, block start {args.block_start}] (cutoff <= every target)")

    npz_map, npz_info = None, None
    if args.ranking_npz:
        npz_map, n_npz = _load_npz_fields(args.ranking_npz, args.ranking_fields or [])
        npz_info = {"path": str(Path(args.ranking_npz).resolve()),
                    "sha256": sha256_file(args.ranking_npz), "rows": n_npz,
                    "fields_found": sorted(npz_map)}

    config = build_config(pg, args)
    ll1["applied_schema_check"] = getattr(config, "schema_check", "off")   # what the module will use
    if ll1["requested"] and ll1["applied_schema_check"] != ll1["requested"]["schema_check"]:
        raise HarnessError(f"LL-1: config asks schema_check={ll1['requested']['schema_check']!r} but "
                           f"PredictionConfig carries {ll1['applied_schema_check']!r}; nothing run")
    logger = logging.getLogger("e2e_vote_harness.pg")
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")
    logger.setLevel(getattr(logging, args.log_level))

    gen, rank, rank_info = build_generator(pg, config, logger, args, rows, npz_map)
    for flag, key in (("expect_sidecar_sha256", "sidecar_sha256"),
                      ("expect_checkpoint_sha256", "checkpoint_sha256")):
        want = getattr(args, flag)
        if want is not None and rank_info.get(key) != want:
            raise HarnessError(f"--{flag.replace('_', '-')} {want}: loaded {key} is "
                               f"{rank_info.get(key)}")
    survivors, arm_info = survivors_for_arm(args.vote_arm, rows, args.arm_skip)
    skips = {}
    for r in survivors:  # the skip each survivor votes with, by the module's own rule
        sk = gen._get_survivor_skip(r, config.default_skip)
        skips[sk] = skips.get(sk, 0) + 1
    arm_info["effective_skip_counts"] = {str(k): v for k, v in sorted(skips.items())}

    arm_id = args.arm_id or "/".join(
        [args.ranking, args.vote_arm] + ([f"skip{args.arm_skip}"] if args.vote_arm == "per_skip" else [])
    + ([f"tie-{args.tie_break}"] if args.tie_break != "module" else []))
    code = {"harness_sha256": sha256_file(__file__),
            "prediction_generator_sha256": sha256_file(REPO_ROOT / "prediction_generator.py"),
            "survivor_scorer_sha256": sha256_file(REPO_ROOT / "survivor_scorer.py"),
            "git_head": git_head()}
    identity_inputs = {
        "survivors_sha256": sha256_file(args.survivors),
        "config_sha256": sha256_file(args.config),
        "history": [{"sha256": m["sha256"], "len": m["len"]} for m in hist_meta],
        "dataset_sha256": dataset_sha256,
        "data_cutoff_index": args.data_cutoff_index,
        "seal_from": args.seal_from,
        "ranking": args.ranking, "ranking_fields": args.ranking_fields,
        "tie_break": args.tie_break,
        "ranking_npz_sha256": npz_info["sha256"] if npz_info else None,
        "sidecar_sha256": rank_info.get("sidecar_sha256"),
        "checkpoint_sha256": rank_info.get("checkpoint_sha256"),
        "vote_arm": args.vote_arm, "arm_skip": args.arm_skip,
        "effective_config": config.to_dict(),
        "harness_sha256": code["harness_sha256"],
        "prediction_generator_sha256": code["prediction_generator_sha256"],
    }
    if args.emit_value_ordering != "off":   # keys present only when on: off ids unchanged
        identity_inputs["value_ordering"] = args.emit_value_ordering
    if args.ranking_key_round_decimals is not None:
        identity_inputs["ranking_key_round_decimals"] = args.ranking_key_round_decimals
    config_id = sha256_text(canonical_json(identity_inputs))

    ranked_spec = None
    if args.emit_ranked_predictions == "on":
        depth = args.ranked_predictions_depth
        if depth < 1 or depth > min(config.k, config.pool_size):
            raise HarnessError(f"--ranked-predictions-depth {depth} must be in [1, min(k, pool "
                               f"size) = {min(config.k, config.pool_size)}]")
        if rank is None or rank_info.get("signal_gate_blocked"):
            raise HarnessError("--emit-ranked-predictions on needs a ranking; the signal gate "
                               "blocked this arm")
        if gen.feature_names:
            names, src = list(gen.feature_names), "sidecar feature_schema (pg:797-798)"
        else:
            names = sorted({k for r in survivors for k in r.get("features", {})
                            if k not in ("score", "confidence", "seed", "skip")})
            src = "row features, sorted, minus score/confidence/seed/skip (pg:800-801)"
        ranked_spec = {"rank": rank, "rows": survivors, "names": names, "names_source": src,
                       "names_sha256": sha256_text(canonical_json(names)), "depth": depth,
                       "draw_id_rule": args.ranked_draw_id_rule,
                       "model_type": rank_info.get("model_type"),
                       "checkpoint_sha256": rank_info.get("checkpoint_sha256"), "order": None}
    elif args.ranked_predictions_depth != 20 or args.ranked_draw_id_rule != "c13_fallback":
        raise HarnessError("--ranked-predictions-depth/--ranked-draw-id-rule need "
                           "--emit-ranked-predictions on")

    ctx = {"arm_id": arm_id, "config_id": config_id, "history_len": n_ident,
           "identity": identity, "k": config.k, "pool_size": config.pool_size,
           "vote_arm": args.vote_arm, "dataset_sha256": dataset_sha256,
           "data_cutoff": args.data_cutoff_index, "pending_date": args.pending_date,
           "pending_session": args.pending_session, "ranked_spec": ranked_spec}
    ts = list(range(args.block_start, args.block_end))
    t_vote = time.time()
    lines, cached = run_block(gen, rank, survivors, combined, ts, ctx, args.workers)
    vote_seconds = time.time() - t_vote

    # integrity: one record per t, in order, pool and diagnostics in step
    if [ln["pool"]["target"]["combined_index"] for ln in lines] != ts or \
            [ln["diag"]["combined_index"] for ln in lines] != ts:
        raise HarnessError("output lines do not match the block one-for-one, in order")
    n_pool = sum(1 for ln in lines if ln["diag"]["vote_status"] == "pool")
    abstain, empty = {}, {}
    for ln in lines:
        st, note = ln["diag"]["vote_status"], ln["diag"]["note"]
        if st == "abstain":
            abstain[note] = abstain.get(note, 0) + 1
        elif st == "empty":
            empty[note] = empty.get(note, 0) + 1
    if rank is not None and not rank_info.get("signal_gate_blocked"):
        if rank.inner_calls != 1 or rank.x_sha256 is None:
            raise HarnessError(f"ranking computed {rank.inner_calls} times, expected 1")
        if cached != len(ts) - 1:
            raise HarnessError(f"{cached} draws after the first reused the ranking, "
                               f"expected {len(ts) - 1}")

    value_ordering = None
    if args.emit_value_ordering == "full_vote":
        if rank is None or rank_info.get("signal_gate_blocked"):
            raise HarnessError("--emit-value-ordering needs a ranking; the signal gate blocked it")
        value_ordering = attach_value_orderings(lines, gen, rank, survivors, config, args.workers)

    out_dir.mkdir(parents=True)
    pools_path = out_dir / "pools.jsonl"
    diag_path = out_dir / "vote_diagnostics.jsonl"
    for path, key in ((pools_path, "pool"), (diag_path, "diag")):
        tmp = path.with_suffix(".jsonl.partial")
        with open(tmp, "w") as f:
            for ln in lines:
                f.write(canonical_json(ln[key]) + "\n")
        os.replace(tmp, path)
    ranked_path = None
    if ranked_spec is not None:
        ranked_path = out_dir / "ranked_predictions.jsonl"
        tmp = ranked_path.with_suffix(".jsonl.partial")
        with open(tmp, "w") as f:
            for ln in lines:
                f.write(canonical_json(ln["ranked"]) + "\n")
        os.replace(tmp, ranked_path)

    record = {
        "format": RECORD_FORMAT_ID, "pool_schema": FORMAT_ID, "diagnostics_schema": DIAG_FORMAT_ID,
        "harness_version": HARNESS_VERSION,
        "pipeline_switches": {"LL-1": ll1},
        "argv": sys.argv if argv is None else ["e2e_vote_harness.py"] + list(argv),
        "cwd": os.getcwd(), "host": platform.node(), "python": sys.version.split()[0],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pg_gpu_available_flag": bool(pg.GPU_AVAILABLE),
        "arm_id": arm_id, "config_id": config_id, "config_identity_inputs": identity_inputs,
        "declared_parameters": {
            "k": config.k, "pool_size": config.pool_size,
            "block_start": args.block_start, "block_end": args.block_end,
            "data_cutoff_index": args.data_cutoff_index,
            "seal_from": args.seal_from,
            "expect_sidecar_sha256": args.expect_sidecar_sha256,
            "expect_checkpoint_sha256": args.expect_checkpoint_sha256,
            "pending_date": args.pending_date, "pending_session": args.pending_session,
            "vote_arm": args.vote_arm, "arm_skip": args.arm_skip,
            "ranking": args.ranking, "ranking_fields": args.ranking_fields,
            "tie_break": args.tie_break, "on_missing_feature": args.on_missing_feature,
            "workers": args.workers, "allow_gpu": args.allow_gpu,
            "harness_forced": {"use_dual_sieve": False, "save_predictions": False,
                               "survivors_reverse": None,
                               "require_feature_schema_forced_false":
                                   args.ranking != "sidecar"},
        },
        "inputs": {"survivors": {"path": str(Path(args.survivors).resolve()),
                                 "sha256": identity_inputs["survivors_sha256"],
                                 "rows": len(rows)},
                   "history": hist_meta, "combined_history_len": len(combined),
                   "dataset": {"path": str(Path(args.dataset).resolve()),
                               "sha256": dataset_sha256, "rows": len(identity),
                               "seal": (None if args.seal_from is None else {
                                   "seal_from": args.seal_from,
                                   "draw_values_held_rows": len(ds_draws),
                                   "draw_values_discarded_at_parse_rows":
                                       len(identity) - len(ds_draws),
                                   "history_pad_value": None,
                                   "note": "rows >= seal_from: date/session only; the module "
                                           "reads only len(history) for dict survivors (pg:839)"})},
                   "config": {"path": str(Path(args.config).resolve()),
                              "sha256": identity_inputs["config_sha256"]},
                   "ranking_npz": npz_info},
        "effective_config": config.to_dict(),
        "prng": {"prng_type": config.prng_type, "mod": config.mod,
                 "default_skip": config.default_skip,
                 "upstream_config_file": config.upstream_config_file},
        "ranking_info": rank_info,
        "arm_info": arm_info,
        "ranking_proof": ({"x_sha256": rank.x_sha256, "x_shape": rank.x_shape,
                           "scores_sha256": rank.scores_sha256(),
                           "model_calls": rank.inner_calls,
                           "draws_reusing_ranking": cached,
                           "ties": rank.tie_report(config.pool_size)}
                          if rank is not None else None),
        "signatures": signatures,
        "code": code,
        "counts": {"draws_in_block": len(ts), "lines_written": len(lines),
                   "pools": n_pool, "abstentions": abstain,
                   "abstentions_total": sum(abstain.values()),
                   "empty_pools": empty, "empty_pools_total": sum(empty.values()),
                   "pending_lines": sum(1 for ln in lines if ln["diag"]["pending"])},
        "timing": {"total_seconds": round(time.time() - t_start, 3),
                   "vote_seconds": round(vote_seconds, 3)},
        "memory": peak_rss_mib(),
        "pools_jsonl": {"path": str(pools_path.resolve()), "sha256": sha256_file(pools_path),
                        "lines": len(lines)},
        "vote_diagnostics_jsonl": {"path": str(diag_path.resolve()),
                                   "sha256": sha256_file(diag_path), "lines": len(lines)},
    }
    if value_ordering is not None:   # present only when on
        record["value_ordering"] = value_ordering
        record["declared_parameters"]["emit_value_ordering"] = args.emit_value_ordering
    if args.ranking_key_round_decimals is not None:
        record["declared_parameters"]["ranking_key_round_decimals"] = args.ranking_key_round_decimals
    if ranked_spec is not None:   # key present only when on: off records are unchanged
        record["ranked_predictions"] = {
            "path": str(ranked_path.resolve()), "sha256": sha256_file(ranked_path),
            "lines": len(lines), "schema": RANKED_FORMAT_ID,
            "depth": ranked_spec["depth"], "draw_id_rule": ranked_spec["draw_id_rule"],
            "feature_names_count": len(ranked_spec["names"]),
            "feature_names_source": ranked_spec["names_source"],
            "feature_schema_sha256": ranked_spec["names_sha256"],
            "not_in_config_id": "the ranked file is derived output; pools and config_id do not "
                                "depend on it"}
    rec_path = out_dir / "run_record.json"
    tmp = rec_path.with_suffix(".json.partial")
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    os.replace(tmp, rec_path)
    print(f"[H6] {len(lines)} records ({n_pool} pools, {sum(abstain.values())} abstentions, "
          f"{sum(empty.values())} empty) -> "
          f"{pools_path}  sha256={record['pools_jsonl']['sha256'][:16]}  "
          f"vote {vote_seconds:.1f}s  peak {record['memory']}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (HarnessError, NotImplementedError) as e:
        print(f"[H6] FAIL-CLOSED: {e}", file=sys.stderr, flush=True)
        sys.exit(2)
