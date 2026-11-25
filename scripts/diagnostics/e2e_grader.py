#!/usr/bin/env python3
"""
e2e_grader.py — TFM hit-rate grader, first form of the WP-3 instrument (TODO-E2E-11).

Status: DRAFT, uncommitted, not Beta-reviewed. Its output is NOT a counted result until
Beta has reviewed it and the owner has given the go for a counted grade (TASK_COMMON rule 2;
WP-3 "Beta reviews before commit", docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:86).

It reads frozen per-draw pools and a frozen dataset, and launches nothing. CPU only.
It makes NO claim about TFM (WP-3, :86). It never judges a seed: seeds are vehicles; the
only quantity it measures is pool membership hits against k/1000 on draws nothing upstream
saw (CLAUDE.md "The yardstick"; docs/PITFALL_STATE_RECOVERY_YARDSTICK.md).

==========================================================================================
RULES (fixed here before any run; each is enforced in code at the function named)
==========================================================================================
Sources, read 2026-11-23 at 41ad509:
  [WP3]  docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md §2 WP-3, lines 78-86; §1 :36-42
  [E]    docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md §E, :115-128
  [F]    same addendum §F, :130-135
  [B4]   docs/exchange/from_beta/X-REVERSE-SIEVE_R2_BETA_REALIGNMENT.md §4, :112-127
         (the "R2 §4" that WP-3 adopts, WP3 :16, :81)
  [BQ5]  docs/exchange/from_beta/X-FINDINGS-PLAN_R2_BETA_REVIEW.md Q5, :69-87 (the omissions
         Beta found in WP-3 + §F taken together; suggested §F text at :85)
  [PC]   docs/TB_RULING_S176_WATCHER_KPI.md Phase C, :692-727; §4.1 null baselines :309-333

 R1  Yardstick. Pool hits against k/1000 on draws unseen upstream. [WP3 :38-42]
 R2  Unseen means unseen by the sieve AND by training, ranking-target construction, feature
     fitting, Optuna selection, threshold tuning and configuration selection for that
     forecast. Training-holdout labels are not automatically final test data.
     [B4 :114-117; WP3 :38-41; BQ5 item 1 :77]
     -> each arm declares `upstream_unseen_from_index`; a graded target below it is an
        ERROR (resolve_records). The declaration is the operator's; the grader cannot see
        upstream, it can only refuse a target the declaration says was seen.
 R3  Causal walk-forward: no data from draw t or later may enter prediction for t.
     [PC :720-727] -> every pool record carries `data_cutoff_index` <= target index, else
     ERROR (resolve_records).
 R4  k_t = number of DISTINCT values in the pool. [WP3 :82; B4 :119-120]
     -> validate_pool_values.
 R5  One membership hit per target draw; duplicated values or several winning entries do
     not inflate hits or coverage. [WP3 :82; B4 :121; E.1 :117-122]
     -> grade_draw. This is a new, clearly named field (`hit`); it does not reuse or
        reinterpret Chapter 13's `exact_hits` (E.1 :120-121).
 R6  Values are integers in [0, draw_space-1] (default draw_space 1000, i.e. [0, 999]);
     wrong type (bool, float, str) or out of range is an ERROR, not a miss. [F :132; E.3 :126-127]
     A draw of 000 is a valid draw and a valid pool value. [E.3 :128]
 R7  Expected hits = sum over graded draws of k_t / draw_space. [WP3 :83; B4 :120-121;
     PC §4.1 :316-325 for lift@K = observed / random]
 R8  Uncertainty for varying k: the hit count under the uniform null is Poisson-binomial
     with p_t = k_t/draw_space. One-sided exact p-value P(X >= H). Interval: exact
     test-inversion interval for a common lift factor lambda (p_t -> min(1, lambda*p_t)).
     [F :134; WP3 :84; B4 :125-126] -> pb_pmf, pb_upper_tail, lift_bounds.
 R9  Primary pool size, evaluation period, abstention rule and comparison rule are declared
     before results are inspected. [WP3 :83-84; B4 :124-125; BQ5 item 3 :79]
     -> they are input FILES (grade block, arm list, abstention rule), each sha256-bound in
        the summary. The grader cannot prove when they were written; the run record must.
 R10 Correction for the number of tests declared in advance. [B4 :127-128 "repeated
     comparisons"; BQ5 item 5 :81] -> family = every declared arm-vs-chance test plus every
     declared comparison (default; `--multiplicity-family arms_only` is the alternative).
     Bonferroni by default, Holm selectable in the arm list.
 R11 A point estimate above baseline alone is not evidence of lift. [B4 :125-126]
     -> no headline carries lift without its p-value, adjusted p-value and interval.
 R12 Paired comparison: same target draws, comparable coverage/budget or explicit
     accounting. [B4 :122-124; BQ5 item 2 :78] -> compare_arms grades only common draws,
     reports both budgets, and computes the exact McNemar test only when k_t is equal on
     every common draw; otherwise it reports "not tested: unequal budget".
 R13 If results guide configuration choice, evaluate on a further untouched period.
     [B4 :126-127; BQ5 item 4 :80] -> each arm may declare `previously_inspected_blocks`;
     a grade block overlapping one of them is an ERROR.
 R14 Missing, stale, malformed or duplicate inputs are ERRORS, not misses.
     [WP3 :84-85; B4 §3 item 3 :103-104; E.3 :126-128]
     stale = record dataset_sha256 differs from the bound dataset, or target date/session
     differs from the dataset row, or config_id differs from the arm's declaration.
 R15 Each frozen pool is bound to its target, cutoff and configuration; duplicate target
     records are rejected or resolved by a DECLARED, outcome-independent rule.
     [F :132-133; E.2 :123-125] -> `duplicate_rule` in {error (default), first_in_file,
     max_data_cutoff}.
 R16 A target with no outcome yet is PENDING, not a miss. [E.3 :127]
 R17 Abstentions are recorded with their denominator. [F :134]
 R18 Acceptance of the instrument: chance on uniform-random pools within a DECLARED
     tolerance (not exact equality); the known count on a fixture. [WP3 :85; F :135]
     -> uniform_random_control (tolerance in standard errors, `--control-tolerance-z`),
        known_count_control (runtime, on the real dataset) and the self-test fixture.
 R19 Ordered per-draw records and configuration identities are preserved so dependence
     and repeated comparisons can be assessed. [B4 :127-128; BQ5 item 5 :81]
     -> per_draw.jsonl in grade order; lag-1 autocorrelation of (hit - p) is reported as
        a diagnostic (no verdict).
 R20 Null results lead the report; every headline carries its denominator.
     (CLAUDE.md box rule 11) -> write_outputs orders results nulls-first.
 R21 Outputs only under a NEW out-dir, never /tmp or /dev/shm (CLAUDE.md box rule 7).
 R22 Development mode (TASK_LOOP_GRADE item 1, 2026-11-23; walk-forward frame of
     TASK_LOOP_COMMON: blocks already looked at are DEVELOPMENT evidence, never a counted
     test). `--mode {counted,development}`, default `counted` = the behaviour authorised for
     the run H counted grade at sha256 a4d96fe7... (counted outputs are unchanged apart from
     the recorded `mode` fields; self-test gate test_counted_default_equivalence).
     -> development must be chosen explicitly (`--mode development`); it is stamped into every
        output: summary `mode`/`status`/`run_label`, every per_draw row, every detail row,
        SUMMARY.md title, every headline, and a `DEVELOPMENT` marker file listed in SHA256SUMS.
     -> it never produces a counted verdict: each result's `above_chance` and
        `counted_verdict` are null and the summary's `above_chance` list is empty; the
        comparison against the modelled background is reported as
        `development_above_background` with development wording only (run, _dev_headline).
     -> in development mode an overlap with `previously_inspected_blocks` is RECORDED per
        arm (`previously_inspected_overlaps`), not refused. In counted mode R13 is unchanged.
     -> a grade block that declares `"mode": "development"` is refused in counted mode, so a
        development block cannot be graded as counted by omission of the flag (run).
 R23 Per-draw detail (TASK_LOOP_GRADE item 2). `--per-draw-detail {off,on}`, default `off`
     (= today: no extra file). `on` writes per_draw_detail.jsonl, one row per record of
     every declared arm (graded, abstained, pending, excluded): draw index, date, session,
     drawn number (graded rows only), pool size (distinct k and raw length), hit or miss,
     and the drawn number's rank in the vote if present: `vote_rank` = 1-based FIRST
     position of the drawn number in the raw pool AS WRITTEN, `vote_count` = its number of
     occurrences. Rank equals vote order only if the writer orders the pool by vote; H6
     does (prediction_generator.py:835-848 descending predicted quality, raw pool kept at
     e2e_vote_harness.py:690); an optional record field `pool_order` (string) is passed
     through so the writer can say so (detail_rows).
 R24 Rank (TASK_LOOP_ADDENDUM_1 item 2, 2026-11-23 17:50; owner 17:42 "Score every draw by
     rank"). Every graded draw can be scored by the rank (1..draw_space, 1 = top of the vote)
     of the drawn number in a full ordering of the outcome space. Source of the ordering,
     per record (`--rank-source`, default `record_or_pool`):
       `value_ranking`: a list of all draw_space distinct values, best first (no ties); or
       `value_scores`:  a list of draw_space finite numbers, one per value, higher = better
                        (ties allowed); or, if the record has neither,
       pool-derived:    the distinct pool values in first-position order, every value not in
                        the pool tied below them.
     `record_only` refuses a record without a full ordering; `pool_only` ignores it.
     Ties take the AVERAGE (mid-)rank (scipy.stats.rankdata method "average"); the tie-group
     size of the drawn number is reported per draw. An optional record string `rank_basis`
     is passed through (the writer says what the ordering is).
     -> `--rank-summary {off,on}` (default off = today): per arm, mean rank against the exact
        uniform expectation (draw_space+1)/2, the exact null mean and sd of the rank sum
        (from each draw's own mid-ranks, so ties are accounted for), a normal-approximation z,
        and a Monte Carlo test under the exact per-draw uniform null (drawn value uniform,
        its mid-rank from that draw's ordering), one-sided toward small ranks,
        p = (1 + #{sim <= obs}) / (R + 1), R = `--rank-test-reps` (default 10000), seed
        `--rank-test-seed`. Runtime control: the simulated mean must match the exact mean
        within `--control-tolerance-z` (rank_summary).
     -> DESCRIPTIVE in counted and development mode alike: the rank test is never in the
        multiplicity family and never changes `above_chance`, `m` or any headline. The
        counted verdict rule is unchanged.
     -> rank fields are validated only when ranks are computed (rank summary or per-draw
        detail on); with both off a record's rank fields are ignored, as today.
 R25 Sessions each on their own and pooled (TASK_LOOP_ADDENDUM_1 item 3).
     `--per-session-report {off,on}` (default off = today): each arm's result gains
     `per_session`, the same statistics (and the rank summary if on) for each session of the
     grade block separately, DESCRIPTIVE and unadjusted; the arm's own result is the pooled
     one. The counted-test rule (owner ruling 08:08) is not changed: m and the verdict
     use the declared grade block only.

Controls run on every invocation (R18, map §2.7):
  - uniform-random pools of the same k_t, through the same grading path, R reps;
  - draw-frequency control: the k_t most frequent values in a declared frequency block
    that lies wholly before every graded draw (ties: lower value first);
  - known-count control: pools built to contain the outcome on a fixed half of the graded
    draws and to exclude it on the rest; the grader must return exactly that count;
  - the §E cases (duplicates, out-of-range, wrong type, 000, pending vs miss, stale target,
    repeated grading, missing and malformed artifacts) are the self-test's fixtures:
    tests/test_e2e_grader.py.

Exit codes: 0 graded and controls passed; 2 input error (nothing written);
            3 outputs written but a control FAILED (results must not be read).

==========================================================================================
INPUT FORMATS (schema tfm.e2e.*.v1)
==========================================================================================
--pools PATH (JSONL, one object per line):
  {"schema": "tfm.e2e.pool.v1", "arm_id": "A0", "config_id": "<id>",
   "dataset_sha256": "<hex>", "data_cutoff_index": 14454,
   "target": {"combined_index": 14454, "date": "2021-03-17", "session": "midday"},
   "pool": [int, ...],                       # raw, as written; repeats allowed
   "pool_sha256": "<hex>"?,                  # optional; verified if present
   "abstain": false?, "abstain_reason": "<str>"?,
   "pool_order": "<str>"?,                   # optional, passed to per_draw_detail (R23)
   "value_ranking": [int x draw_space]? | "value_scores": [number x draw_space]?,
   "rank_basis": "<str>"?}                   # optional full ordering for rank (R24)
--dataset PATH + --dataset-sha256 HEX: a list of {"date","session","draw"}; row index is
  the combined index (window_optimizer.py:1159-1171 builds train/holdout this way).
--grade-block PATH: {"schema": "tfm.e2e.grade_block.v1", "name": str,
   "start_index": int, "end_index": int (exclusive; may exceed the dataset -> pending),
   "sessions": ["midday", ...],
   "frequency_control_block": {"start_index": int, "end_index": int,
                               "sessions": [...]?},
   "mode": "counted"|"development"?}         # optional; "development" refused in counted mode (R22)
--arms PATH: {"schema": "tfm.e2e.arms.v1",
   "arms": [{"arm_id", "config_id", "upstream_unseen_from_index": int,
             "max_k": int?, "min_k": int?, "previously_inspected_blocks": [[s,e],...]?,
             "description": str?}],
   "comparisons": [{"arm": "A1", "baseline": "A0"}]?,
   "family_alpha": 0.05?, "correction": "bonferroni"|"holm"?}
--abstention PATH: {"schema": "tfm.e2e.abstention.v1", "allow_abstention": bool,
   "allowed_reasons": [str]|null, "empty_pool": "error"|"abstain"}
"""

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import binom, norm, rankdata

GRADER_VERSION = "e2e_grader 0.2.0-draft (WP-3 first form + development mode, per-draw detail)"
MODES = ("counted", "development")
SCHEMA_POOL = "tfm.e2e.pool.v1"
SCHEMA_GRADE_BLOCK = "tfm.e2e.grade_block.v1"
SCHEMA_ARMS = "tfm.e2e.arms.v1"
SCHEMA_ABSTENTION = "tfm.e2e.abstention.v1"
SCHEMA_SUMMARY = "tfm.e2e.grade_summary.v1"

# Declared defaults. Each is overridable (CLI or the arm list) and recorded in the summary.
DEFAULTS = {
    "draw_space": 1000,              # outcome space 000-999 (S176 §4.1 :313)
    "family_alpha": 0.05,
    "correction": "bonferroni",
    "multiplicity_family": "all_declared_tests",
    "duplicate_rule": "error",
    "out_of_block_records": "error",
    "require_record_dataset_sha256": True,
    "control_reps": 200,
    "control_seed": 20261123,
    "control_tolerance_z": 4.0,
    "workers": max(1, min(4, (os.cpu_count() or 1) - 1)),
    "descriptive_ci_level": 0.95,
    "mode": "counted",               # R22; development must be chosen explicitly
    "per_draw_detail": "off",        # R23
    "rank_summary": "off",           # R24
    "rank_source": "record_or_pool",  # R24
    "rank_test_reps": 10000,         # R24
    "rank_test_seed": 20261123,      # R24
    "per_session_report": "off",     # R25
}
RANK_SOURCES = ("record_or_pool", "record_only", "pool_only")
FORBIDDEN_OUT_PREFIXES = ("/tmp", "/dev/shm", "/var/tmp")


class GraderError(Exception):
    """Fail-closed input error. Nothing is written."""


# ------------------------------------------------------------------------------ hashing --
def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def pool_sha256(raw_pool):
    """sha256 of the raw pool exactly as written (order and repeats kept)."""
    return sha256_bytes(canonical_json(raw_pool).encode("ascii"))


# --------------------------------------------------------------------------- validators --
def _is_int(x):
    # bool is a subclass of int; reject it explicitly (pattern from
    # watcher_kpi_metricC_deterministic_v2_2.py:4 "REJECT BOOLEAN").
    return isinstance(x, int) and not isinstance(x, bool)


def _req_int(obj, key, where):
    if key not in obj:
        raise GraderError(f"{where}: missing required field '{key}'")
    v = obj[key]
    if not _is_int(v):
        raise GraderError(f"{where}: field '{key}' must be an integer, got {v!r}")
    return v


def validate_pool_values(raw_pool, draw_space, where):
    """R4, R6. Returns (distinct sorted tuple, n_raw, n_duplicates). Raises on bad values."""
    if not isinstance(raw_pool, list):
        raise GraderError(f"{where}: 'pool' must be a list, got {type(raw_pool).__name__}")
    for v in raw_pool:
        if not _is_int(v):
            raise GraderError(f"{where}: pool value {v!r} is not an integer (wrong type)")
        if v < 0 or v > draw_space - 1:
            raise GraderError(f"{where}: pool value {v} outside [0, {draw_space - 1}]")
    distinct = tuple(sorted(set(raw_pool)))
    return distinct, len(raw_pool), len(raw_pool) - len(distinct)


def grade_draw(distinct_values, outcome):
    """R5: one membership hit per draw. `outcome` 0 (draw 000) is a valid value."""
    return 1 if outcome in distinct_values else 0


# ------------------------------------------------------------------------------ loading --
def load_dataset(path, expected_sha256, draw_space):
    """Dataset bound by sha256 (R14). Returns list of (date, session, draw)."""
    p = Path(path)
    if not p.is_file():
        raise GraderError(f"dataset not found: {path}")
    actual = sha256_file(p)
    if actual != expected_sha256.lower():
        raise GraderError(f"dataset sha256 mismatch: expected {expected_sha256}, got {actual}")
    try:
        data = json.loads(p.read_bytes())
    except json.JSONDecodeError as e:
        raise GraderError(f"dataset is not valid JSON: {e}")
    if not isinstance(data, list) or not data:
        raise GraderError("dataset must be a non-empty list of {date, session, draw}")
    rows = []
    for i, d in enumerate(data):
        if not isinstance(d, dict) or not {"date", "session", "draw"} <= set(d):
            raise GraderError(f"dataset row {i}: needs date, session, draw")
        v = d["draw"]
        if not _is_int(v) or v < 0 or v > draw_space - 1:
            raise GraderError(f"dataset row {i}: draw {v!r} not an integer in [0, {draw_space - 1}]")
        rows.append((str(d["date"]), str(d["session"]), v))
    return rows, actual


def load_declaration(path, schema, label):
    p = Path(path)
    if not p.is_file():
        raise GraderError(f"{label} file not found: {path}")
    raw = p.read_bytes()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise GraderError(f"{label} file is not valid JSON: {e}")
    if not isinstance(obj, dict) or obj.get("schema") != schema:
        raise GraderError(f"{label} file must be an object with schema '{schema}'")
    return obj, sha256_bytes(raw)


def parse_grade_block(obj):
    gb = {"name": str(obj.get("name", "")),
          "start_index": _req_int(obj, "start_index", "grade block"),
          "end_index": _req_int(obj, "end_index", "grade block")}
    if gb["start_index"] < 0 or gb["end_index"] <= gb["start_index"]:
        raise GraderError("grade block: need 0 <= start_index < end_index")
    s = obj.get("sessions")
    if not isinstance(s, list) or not s or not all(isinstance(x, str) for x in s):
        raise GraderError("grade block: 'sessions' must be a non-empty list of strings")
    gb["sessions"] = sorted(set(s))
    fb = obj.get("frequency_control_block")
    if not isinstance(fb, dict):
        raise GraderError("grade block: 'frequency_control_block' is required (draw-frequency control)")
    fcb = {"start_index": _req_int(fb, "start_index", "frequency_control_block"),
           "end_index": _req_int(fb, "end_index", "frequency_control_block")}
    if fcb["start_index"] < 0 or fcb["end_index"] <= fcb["start_index"]:
        raise GraderError("frequency_control_block: need 0 <= start_index < end_index")
    fs = fb.get("sessions", gb["sessions"])
    if not isinstance(fs, list) or not fs:
        raise GraderError("frequency_control_block: 'sessions' must be a non-empty list")
    fcb["sessions"] = sorted(set(fs))
    gb["frequency_control_block"] = fcb
    if "mode" in obj:                       # R22; absent = today's file, no key added
        if obj["mode"] not in MODES:
            raise GraderError(f"grade block: 'mode' must be one of {list(MODES)}")
        gb["mode"] = obj["mode"]
    return gb


def parse_arms(obj):
    arms_in = obj.get("arms")
    if not isinstance(arms_in, list) or not arms_in:
        raise GraderError("arm list: 'arms' must be a non-empty list")
    arms = {}
    for a in arms_in:
        if not isinstance(a, dict):
            raise GraderError("arm list: each arm must be an object")
        aid = a.get("arm_id")
        cid = a.get("config_id")
        if not isinstance(aid, str) or not aid or not isinstance(cid, str) or not cid:
            raise GraderError("arm list: each arm needs non-empty string arm_id and config_id")
        if aid in arms:
            raise GraderError(f"arm list: duplicate arm_id {aid}")
        arm = {"arm_id": aid, "config_id": cid,
               "upstream_unseen_from_index": _req_int(a, "upstream_unseen_from_index", f"arm {aid}"),
               "description": str(a.get("description", "")),
               "min_k": a.get("min_k", 1), "max_k": a.get("max_k")}
        if not _is_int(arm["min_k"]) or arm["min_k"] < 1:
            raise GraderError(f"arm {aid}: min_k must be an integer >= 1")
        if arm["max_k"] is not None and (not _is_int(arm["max_k"]) or arm["max_k"] < arm["min_k"]):
            raise GraderError(f"arm {aid}: max_k must be an integer >= min_k")
        pib = a.get("previously_inspected_blocks", [])
        if not isinstance(pib, list):
            raise GraderError(f"arm {aid}: previously_inspected_blocks must be a list")
        blocks = []
        for b in pib:
            if (not isinstance(b, list) or len(b) != 2 or not all(_is_int(x) for x in b)
                    or b[1] <= b[0]):
                raise GraderError(f"arm {aid}: bad previously_inspected_block {b!r}")
            blocks.append((b[0], b[1]))
        arm["previously_inspected_blocks"] = blocks
        arms[aid] = arm
    comps = []
    for c in obj.get("comparisons", []) or []:
        if (not isinstance(c, dict) or c.get("arm") not in arms or c.get("baseline") not in arms
                or c["arm"] == c["baseline"]):
            raise GraderError(f"arm list: bad comparison {c!r} (both arms must be declared)")
        comps.append({"arm": c["arm"], "baseline": c["baseline"]})
    alpha = obj.get("family_alpha", DEFAULTS["family_alpha"])
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise GraderError("arm list: family_alpha must be in (0, 1)")
    corr = obj.get("correction", DEFAULTS["correction"])
    if corr not in ("bonferroni", "holm"):
        raise GraderError("arm list: correction must be 'bonferroni' or 'holm'")
    return {"arms": arms, "comparisons": comps, "family_alpha": float(alpha), "correction": corr}


def parse_abstention(obj):
    allow = obj.get("allow_abstention")
    if not isinstance(allow, bool):
        raise GraderError("abstention rule: 'allow_abstention' must be true or false")
    reasons = obj.get("allowed_reasons")
    if reasons is not None and (not isinstance(reasons, list)
                                or not all(isinstance(r, str) for r in reasons)):
        raise GraderError("abstention rule: 'allowed_reasons' must be null or a list of strings")
    empty = obj.get("empty_pool", "error")
    if empty not in ("error", "abstain"):
        raise GraderError("abstention rule: 'empty_pool' must be 'error' or 'abstain'")
    return {"allow_abstention": allow, "allowed_reasons": reasons, "empty_pool": empty}


def read_pool_records(path):
    """Malformed lines are errors (R14)."""
    p = Path(path)
    if not p.is_file():
        raise GraderError(f"pools file not found: {path}")
    recs = []
    with open(p, "rb") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                raise GraderError(f"pools line {lineno}: blank line (malformed artifact)")
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise GraderError(f"pools line {lineno}: malformed JSON: {e}")
            if not isinstance(obj, dict):
                raise GraderError(f"pools line {lineno}: not a JSON object")
            recs.append((lineno, obj))
    if not recs:
        raise GraderError("pools file is empty")
    return recs


# ----------------------------------------------------------------------- record resolve --
def _overlaps(a0, a1, b0, b1):
    return a0 < b1 and b0 < a1


def resolve_records(records, arms_decl, gb, abst, dataset, dataset_sha, cfg):
    """Validate every record and bind it to one (arm, target). R2, R3, R6, R13-R17.

    Returns per_arm: {arm_id: {"graded": [...], "abstained": [...], "pending": [...],
                               "excluded": [...], "duplicates_resolved": [...]}}
    in ascending target order.
    """
    draw_space = cfg["draw_space"]
    n_data = len(dataset)
    arms = arms_decl["arms"]
    development = cfg.get("mode", "counted") == "development"
    inspected = {aid: [] for aid in arms}
    for aid, a in arms.items():
        for (b0, b1) in a["previously_inspected_blocks"]:
            if _overlaps(gb["start_index"], gb["end_index"], b0, b1):
                if development:             # R22: recorded, not refused; counted mode raises
                    inspected[aid].append([b0, b1])
                    continue
                raise GraderError(f"arm {aid}: grade block [{gb['start_index']},{gb['end_index']}) "
                                  f"overlaps previously inspected block [{b0},{b1}) (R13)")
    by_key = {}
    excluded = {aid: [] for aid in arms}
    for lineno, r in records:
        where = f"pools line {lineno}"
        if r.get("schema") != SCHEMA_POOL:
            raise GraderError(f"{where}: schema must be '{SCHEMA_POOL}'")
        aid = r.get("arm_id")
        if aid not in arms:
            raise GraderError(f"{where}: arm_id {aid!r} is not declared in the arm list")
        arm = arms[aid]
        if r.get("config_id") != arm["config_id"]:
            raise GraderError(f"{where}: config_id {r.get('config_id')!r} differs from arm "
                              f"{aid}'s declared {arm['config_id']!r} (stale or foreign record)")
        if "dataset_sha256" in r or cfg["require_record_dataset_sha256"]:
            if r.get("dataset_sha256") != dataset_sha:
                raise GraderError(f"{where}: dataset_sha256 {r.get('dataset_sha256')!r} differs "
                                  f"from the bound dataset (stale record)")
        tgt = r.get("target")
        if not isinstance(tgt, dict):
            raise GraderError(f"{where}: missing 'target' object")
        t = _req_int(tgt, "combined_index", where + " target")
        if t < 0:
            raise GraderError(f"{where}: negative target index")
        cutoff = _req_int(r, "data_cutoff_index", where)
        if cutoff > t:
            raise GraderError(f"{where}: data_cutoff_index {cutoff} > target {t}: the forecast "
                              f"used data from the target or later (R3, S176 Phase C)")
        if t < arm["upstream_unseen_from_index"]:
            raise GraderError(f"{where}: target {t} < arm {aid} upstream_unseen_from_index "
                              f"{arm['upstream_unseen_from_index']}: the declaration says this "
                              f"draw was seen upstream (R2)")
        pending = t >= n_data
        if not pending:
            d_date, d_sess, _ = dataset[t]
            if tgt.get("date") != d_date or tgt.get("session") != d_sess:
                raise GraderError(f"{where}: target ({tgt.get('date')}, {tgt.get('session')}) "
                                  f"differs from dataset row {t} ({d_date}, {d_sess}) (stale target)")
            session = d_sess
        else:
            if not isinstance(tgt.get("date"), str) or not isinstance(tgt.get("session"), str):
                raise GraderError(f"{where}: pending target needs date and session strings")
            session = tgt["session"]
        in_block = gb["start_index"] <= t < gb["end_index"] and session in gb["sessions"]
        abstain = r.get("abstain", False)
        if not isinstance(abstain, bool):
            raise GraderError(f"{where}: 'abstain' must be a boolean")
        raw_pool = r.get("pool", [])
        if abstain:
            if not abst["allow_abstention"]:
                raise GraderError(f"{where}: abstention not allowed by the declared rule")
            reason = r.get("abstain_reason")
            if not isinstance(reason, str) or not reason:
                raise GraderError(f"{where}: abstention needs a non-empty abstain_reason")
            if abst["allowed_reasons"] is not None and reason not in abst["allowed_reasons"]:
                raise GraderError(f"{where}: abstain_reason {reason!r} not in the declared list")
            distinct, n_raw, n_dup = (), 0, 0
        else:
            distinct, n_raw, n_dup = validate_pool_values(raw_pool, draw_space, where)
            if len(distinct) == 0:
                if abst["empty_pool"] == "abstain" and abst["allow_abstention"]:
                    abstain, reason = True, "empty_pool"
                else:
                    raise GraderError(f"{where}: empty pool (declared rule: error)")
            else:
                reason = None
                if len(distinct) < arm["min_k"] or (arm["max_k"] is not None
                                                     and len(distinct) > arm["max_k"]):
                    raise GraderError(f"{where}: distinct k {len(distinct)} outside arm {aid}'s "
                                      f"declared [{arm['min_k']}, {arm['max_k']}]")
        psha = pool_sha256(raw_pool)
        if "pool_sha256" in r and r["pool_sha256"] != psha:
            raise GraderError(f"{where}: pool_sha256 does not match the pool as written")
        pool_order = r.get("pool_order")
        if pool_order is not None and not isinstance(pool_order, str):
            raise GraderError(f"{where}: 'pool_order' must be a string if present")
        entry = {"arm_id": aid, "config_id": arm["config_id"], "combined_index": t,
                 "date": tgt.get("date"), "session": session, "data_cutoff_index": cutoff,
                 "pool_sha256": psha, "k_raw": n_raw, "k": len(distinct),
                 "n_duplicate_values": n_dup, "distinct": distinct, "abstain": abstain,
                 "abstain_reason": reason, "pending": pending, "source_line": lineno,
                 "raw_pool": list(raw_pool) if not abstain else [], "pool_order": pool_order,
                 "value_ranking": r.get("value_ranking"), "value_scores": r.get("value_scores"),
                 "rank_basis": r.get("rank_basis")}       # R24: validated only when used
        if not in_block:
            if cfg["out_of_block_records"] == "error":
                raise GraderError(f"{where}: target {t} ({session}) is outside the grade block "
                                  f"(declared rule: error)")
            excluded[aid].append(entry)
            continue
        key = (aid, t)
        by_key.setdefault(key, []).append(entry)

    per_arm = {aid: {"graded": [], "abstained": [], "pending": [], "excluded": excluded[aid],
                     "duplicates_resolved": [], "previously_inspected_overlaps": inspected[aid]}
               for aid in arms}
    for (aid, t), entries in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if len(entries) > 1:
            rule = cfg["duplicate_rule"]
            lines = [e["source_line"] for e in entries]
            if rule == "error":
                raise GraderError(f"arm {aid} target {t}: {len(entries)} records (lines {lines}); "
                                  f"declared duplicate rule: error (R15)")
            if rule == "first_in_file":
                chosen = min(entries, key=lambda e: e["source_line"])
            elif rule == "max_data_cutoff":
                top = max(e["data_cutoff_index"] for e in entries)
                cands = [e for e in entries if e["data_cutoff_index"] == top]
                if len(cands) > 1:
                    raise GraderError(f"arm {aid} target {t}: max_data_cutoff tie on lines "
                                      f"{[e['source_line'] for e in cands]}")
                chosen = cands[0]
            else:
                raise GraderError(f"unknown duplicate rule {rule}")
            per_arm[aid]["duplicates_resolved"].append(
                {"combined_index": t, "lines": lines, "chosen_line": chosen["source_line"],
                 "rule": rule})
        else:
            chosen = entries[0]
        if chosen["pending"]:
            per_arm[aid]["pending"].append(chosen)
        elif chosen["abstain"]:
            per_arm[aid]["abstained"].append(chosen)
        else:
            per_arm[aid]["graded"].append(chosen)

    # Completeness: every eligible (non-pending) grade-block draw needs one record per arm.
    eligible = [i for i in range(gb["start_index"], min(gb["end_index"], n_data))
                if dataset[i][1] in gb["sessions"]]
    if not eligible:
        raise GraderError("grade block contains no eligible draw in the dataset")
    for aid in arms:
        have = {e["combined_index"] for e in per_arm[aid]["graded"] + per_arm[aid]["abstained"]}
        missing = [i for i in eligible if i not in have]
        if missing:
            raise GraderError(f"arm {aid}: {len(missing)} of {len(eligible)} eligible grade-block "
                              f"draws have no pool record (first: {missing[:5]}) — missing input "
                              f"is an error, not a miss (R14)")
    return per_arm, eligible


# ------------------------------------------------------------------------- statistics ----
def pb_pmf(ps):
    """Poisson-binomial PMF. Draws with equal p are grouped as binomials (scipy) and the
    groups convolved; exact up to float rounding."""
    ps = np.asarray(ps, dtype=float)
    if ps.size == 0:
        return np.array([1.0])
    vals, counts = np.unique(ps, return_counts=True)
    pmf = np.array([1.0])
    for p, c in zip(vals, counts):
        g = binom.pmf(np.arange(c + 1), c, min(max(p, 0.0), 1.0))
        pmf = np.convolve(pmf, g)
    pmf = np.clip(pmf, 0.0, None)
    return pmf / pmf.sum()


def pb_upper_tail(ps, h):
    """P(X >= h) under the Poisson-binomial with probabilities ps."""
    pmf = pb_pmf(ps)
    if h <= 0:
        return 1.0
    if h >= pmf.size:
        return 0.0
    return float(min(1.0, pmf[h:].sum()))


def pb_lower_tail(ps, h):
    """P(X <= h)."""
    pmf = pb_pmf(ps)
    if h < 0:
        return 0.0
    return float(min(1.0, pmf[:h + 1].sum()))


def _scaled(ps, lam):
    return np.minimum(1.0, lam * np.asarray(ps, dtype=float))


def lift_bounds(ps, h, a_lower, a_upper, iters=80):
    """Exact test-inversion bounds for a common lift lambda (R8).
    lower: largest lambda with P_lambda(X >= h) <= a_lower  (one-sided at a_lower);
    upper: smallest lambda with P_lambda(X <= h) <= a_upper; None if unbounded."""
    ps = np.asarray(ps, dtype=float)
    n = ps.size
    lam_max = 1.0 / ps.min()
    lo_b = None
    if a_lower is not None:
        if h == 0:
            lo_b = 0.0
        else:
            lo, hi = 0.0, lam_max
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                if pb_upper_tail(_scaled(ps, mid), h) <= a_lower:
                    lo = mid
                else:
                    hi = mid
            lo_b = lo
    up_b = None
    if a_upper is not None and h < n:
        if pb_lower_tail(_scaled(ps, lam_max), h) > a_upper:
            up_b = None
        else:
            lo, hi = 0.0, lam_max
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                if pb_lower_tail(_scaled(ps, mid), h) <= a_upper:
                    hi = mid
                else:
                    lo = mid
            up_b = hi
    return lo_b, up_b


def lag1_autocorr(x):
    x = np.asarray(x, dtype=float)
    if x.size < 3 or np.allclose(x, x[0]):
        return None
    a, b = x[:-1] - x[:-1].mean(), x[1:] - x[1:].mean()
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return None if den == 0 else float((a * b).sum() / den)


def adjust_pvalues(pvals, method):
    """Family-wise adjustment over the declared tests (R10). None = not tested, still
    counted in m."""
    m = len(pvals)
    if method == "bonferroni":
        return [None if p is None else min(1.0, m * p) for p in pvals]
    order = sorted([i for i, p in enumerate(pvals) if p is not None], key=lambda i: pvals[i])
    adj = [None] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def arm_statistics(graded_rows, draw_space):
    hits = [r["hit"] for r in graded_rows]
    ps = [r["k"] / draw_space for r in graded_rows]
    H = int(sum(hits))
    E = float(sum(ps))
    V = float(sum(p * (1 - p) for p in ps))
    return {"n_graded": len(graded_rows), "hits": H, "expected_hits": E,
            "null_sd": math.sqrt(V), "sum_k": int(sum(r["k"] for r in graded_rows)),
            "hit_rate": H / len(graded_rows) if graded_rows else None,
            "baseline_rate": E / len(graded_rows) if graded_rows else None,
            "lift_point": (H / E) if E > 0 else None,
            "p_one_sided": pb_upper_tail(ps, H),
            "lag1_autocorr_residual": lag1_autocorr([h - p for h, p in zip(hits, ps)])}, ps


# ---------------------------------------------------------------------------- controls ---
_CTRL = {}


def _ctrl_init(ks, outcomes, draw_space, seed):
    _CTRL.update(ks=ks, outcomes=outcomes, draw_space=draw_space, seed=seed)


def _ctrl_rep(rep):
    """One uniform-random replicate through the SAME validate/grade path (R18)."""
    ss = np.random.SeedSequence([_CTRL["seed"], rep])
    rng = np.random.default_rng(ss)
    ds = _CTRL["draw_space"]
    h = 0
    for k, y in zip(_CTRL["ks"], _CTRL["outcomes"]):
        pool = rng.permutation(ds)[:k].tolist()
        distinct, _, _ = validate_pool_values(pool, ds, "uniform-random control")
        h += grade_draw(distinct, y)
    return h


def uniform_random_control(graded_rows, outcomes, cfg, alpha_test):
    ks = [r["k"] for r in graded_rows]
    ps = [k / cfg["draw_space"] for k in ks]
    E = float(sum(ps))
    V = float(sum(p * (1 - p) for p in ps))
    R = cfg["control_reps"]
    args = (ks, outcomes, cfg["draw_space"], cfg["control_seed"])
    if cfg["workers"] > 1 and R > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(cfg["workers"], initializer=_ctrl_init, initargs=args) as pool:
            reps = pool.map(_ctrl_rep, range(R), chunksize=max(1, R // (4 * cfg["workers"])))
    else:
        _ctrl_init(*args)
        reps = [_ctrl_rep(i) for i in range(R)]
    mean = float(np.mean(reps))
    se = math.sqrt(V / R) if R > 0 else float("inf")
    z = (mean - E) / se if se > 0 else 0.0
    # Rejection rate of the arm test on random pools: must not exceed alpha beyond tolerance.
    crit = None
    pmf = pb_pmf(ps)
    tail = np.cumsum(pmf[::-1])[::-1]          # tail[h] = P(X >= h)
    ok_h = np.nonzero(tail <= alpha_test)[0]
    crit = int(ok_h[0]) if ok_h.size else None
    rej = float(np.mean([h >= crit for h in reps])) if crit is not None else 0.0
    exact_size = float(tail[crit]) if crit is not None else 0.0
    rej_se = math.sqrt(max(exact_size * (1 - exact_size), 1e-12) / R) if R > 0 else float("inf")
    tol = cfg["control_tolerance_z"]
    passed = abs(z) <= tol and rej <= exact_size + tol * rej_se
    return {"reps": R, "seed": cfg["control_seed"], "expected_hits": E, "mean_hits": mean,
            "se_of_mean": se, "z": z, "tolerance_z": tol, "critical_hits": crit,
            "exact_test_size": exact_size, "empirical_rejection_rate": rej,
            "passed": bool(passed), "rep_hits": reps}


def frequency_control(graded_rows, dataset, fcb, draw_space):
    """Draw-frequency control: the k_t most frequent values in the declared block (which
    must end before every graded draw); ties broken by lower value."""
    if graded_rows and fcb["end_index"] > min(r["combined_index"] for r in graded_rows):
        raise GraderError(f"frequency_control_block ends at {fcb['end_index']}, after the first "
                          f"graded draw {min(r['combined_index'] for r in graded_rows)} (not causal)")
    if fcb["end_index"] > len(dataset):
        raise GraderError("frequency_control_block extends past the dataset")
    counts = np.zeros(draw_space, dtype=np.int64)
    n_rows = 0
    for i in range(fcb["start_index"], fcb["end_index"]):
        if dataset[i][1] in fcb["sessions"]:
            counts[dataset[i][2]] += 1
            n_rows += 1
    if n_rows == 0:
        raise GraderError("frequency_control_block selects no rows")
    order = sorted(range(draw_space), key=lambda v: (-int(counts[v]), v))
    hits, ps = [], []
    for r in graded_rows:
        distinct, _, _ = validate_pool_values(order[:r["k"]], draw_space, "frequency control")
        hits.append(grade_draw(distinct, r["outcome"]))
        ps.append(r["k"] / draw_space)
    H = int(sum(hits))
    return {"block": fcb, "n_block_rows": n_rows, "n_graded": len(graded_rows), "hits": H,
            "expected_hits": float(sum(ps)), "p_one_sided_unadjusted": pb_upper_tail(ps, H),
            "top_values": order[:20]}


def known_count_control(graded_rows, draw_space):
    """Runtime positive control on the real outcomes: include the outcome on even positions,
    exclude it on odd ones; the grading path must return exactly that count."""
    expected = 0
    got = 0
    for i, r in enumerate(graded_rows):
        y, k = r["outcome"], r["k"]
        others = [v for v in range(draw_space) if v != y]
        if i % 2 == 0 or k >= draw_space:
            pool = [y] + others[:k - 1]
            expected += 1
        else:
            pool = others[:k]
        distinct, _, _ = validate_pool_values(pool, draw_space, "known-count control")
        got += grade_draw(distinct, y)
    return {"n": len(graded_rows), "expected_hits": expected, "hits": got,
            "passed": got == expected}


def compare_arms(rows_a, rows_b, draw_space):
    """R12: same target draws only; exact one-sided McNemar when budgets are equal."""
    a = {r["combined_index"]: r for r in rows_a}
    b = {r["combined_index"]: r for r in rows_b}
    common = sorted(set(a) & set(b))
    out = {"n_common": len(common), "n_only_arm": len(set(a) - set(b)),
           "n_only_baseline": len(set(b) - set(a))}
    if not common:
        out.update(tested=False, reason="no common graded draws", p_one_sided=None)
        return out
    ha = sum(a[t]["hit"] for t in common)
    hb = sum(b[t]["hit"] for t in common)
    ea = sum(a[t]["k"] for t in common) / draw_space
    eb = sum(b[t]["k"] for t in common) / draw_space
    disc_ab = sum(1 for t in common if a[t]["hit"] == 1 and b[t]["hit"] == 0)
    disc_ba = sum(1 for t in common if a[t]["hit"] == 0 and b[t]["hit"] == 1)
    equal_budget = all(a[t]["k"] == b[t]["k"] for t in common)
    out.update(hits_arm=ha, hits_baseline=hb, expected_arm=ea, expected_baseline=eb,
               sum_k_arm=int(round(ea * draw_space)), sum_k_baseline=int(round(eb * draw_space)),
               excess_arm=ha - ea, excess_baseline=hb - eb,
               discordant_arm_only=disc_ab, discordant_baseline_only=disc_ba,
               equal_budget=equal_budget)
    if not equal_budget:
        out.update(tested=False, reason="not tested: unequal budget (k_t differs on some common "
                   "draw); excess over expected reported for accounting", p_one_sided=None)
    else:
        nd = disc_ab + disc_ba
        p = 1.0 if nd == 0 else float(binom.sf(disc_ab - 1, nd, 0.5))
        out.update(tested=True, reason="exact one-sided McNemar on discordant draws",
                   p_one_sided=p)
    return out


# ------------------------------------------------------------------------------- rank ---
def draw_ranks(e, draw_space, rank_source):
    """R24: mid-ranks (1 = best) of every value 0..draw_space-1 for one record, and the
    ordering's source. Raises GraderError on a malformed or missing (record_only) ordering."""
    where = f"pools line {e.get('source_line')}"
    vr, vs = e.get("value_ranking"), e.get("value_scores")
    if vr is not None and vs is not None:
        raise GraderError(f"{where}: give value_ranking OR value_scores, not both (R24)")
    use_record = (vr is not None or vs is not None) and rank_source != "pool_only"
    if not use_record and rank_source == "record_only":
        raise GraderError(f"{where}: no full ordering (value_ranking/value_scores) and "
                          f"--rank-source record_only (R24)")
    scores = np.zeros(draw_space, dtype=float)
    if use_record and vr is not None:
        if (not isinstance(vr, list) or len(vr) != draw_space or not all(_is_int(v) for v in vr)
                or len(set(vr)) != draw_space or min(vr) < 0 or max(vr) > draw_space - 1):
            raise GraderError(f"{where}: value_ranking must list all {draw_space} distinct values "
                              f"0..{draw_space - 1} once each (R24)")
        for pos, v in enumerate(vr):
            scores[v] = draw_space - pos
        source = "record_value_ranking"
        # TODO-E2E-G10: a supplied ordering is not checked against the record's pool (e.g. that
        # the pool's distinct values are its top entries); the writer's rank_basis states it.
    elif use_record:
        if (not isinstance(vs, list) or len(vs) != draw_space
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                           and math.isfinite(x) for x in vs)):
            raise GraderError(f"{where}: value_scores must be {draw_space} finite numbers (R24)")
        scores = np.asarray(vs, dtype=float)
        source = "record_value_scores"
    else:
        seen = []
        for v in e.get("raw_pool") or []:
            if v not in seen:
                seen.append(v)
        for pos, v in enumerate(seen):
            scores[v] = draw_space - pos          # > 0; values not in the pool stay 0, tied
        source = "pool_first_position_rest_tied"
    return rankdata(-scores, method="average"), source


def rank_summary(rows, cfg):
    """R24: rank of the drawn number against the exact per-draw uniform null. Descriptive."""
    ds = cfg["draw_space"]
    n = len(rows)
    obs = np.array([r["rank"] for r in rows], dtype=float)
    M = np.stack([r["_ranks"] for r in rows]).astype(np.float64)
    mu = M.mean(axis=1)
    var = (M * M).mean(axis=1) - mu * mu
    exact_mean = float(mu.sum())
    exact_sd = math.sqrt(max(float(var.sum()), 0.0))
    S = float(obs.sum())
    z = (S - exact_mean) / exact_sd if exact_sd > 0 else 0.0
    R = cfg["rank_test_reps"]
    rng = np.random.default_rng(np.random.SeedSequence([cfg["rank_test_seed"], 0x52414E4B]))
    chunk = max(1, int(2_000_000 // max(n, 1)))
    le = ge = 0
    tot = tot2 = 0.0
    rows_ix = np.arange(n)[None, :]
    for start in range(0, R, chunk):
        c = min(chunk, R - start)
        idx = rng.integers(0, ds, size=(c, n))
        sims = M[rows_ix, idx].sum(axis=1)
        le += int((sims <= S + 1e-6).sum())
        ge += int((sims >= S - 1e-6).sum())
        tot += float(sims.sum())
        tot2 += float((sims * sims).sum())
    mc_mean = tot / R
    mc_se = exact_sd / math.sqrt(R) if exact_sd > 0 else 0.0
    zc = (mc_mean - exact_mean) / mc_se if mc_se > 0 else (0.0 if abs(mc_mean - exact_mean) < 1e-6 else float("inf"))
    tol = cfg["control_tolerance_z"]
    sources = {}
    for r in rows:
        sources[r["rank_source"]] = sources.get(r["rank_source"], 0) + 1
    n_tied_struct = int(sum(1 for i in range(n) if np.unique(M[i]).size < ds))
    return {"n": n, "mean_rank": S / n if n else None,
            "expected_mean_rank": exact_mean / n if n else None,
            "sum_rank": S, "exact_null_mean_sum": exact_mean, "exact_null_sd_sum": exact_sd,
            "z_normal_approx": z, "p_lower_normal_approx": float(norm.cdf(z)) if exact_sd > 0 else None,
            "p_lower_monte_carlo": (1 + le) / (R + 1), "p_upper_monte_carlo": (1 + ge) / (R + 1),
            "reps": R, "seed": cfg["rank_test_seed"],
            "test": ("Monte Carlo under the exact per-draw uniform null: the drawn value is uniform "
                     "on the outcome space and its mid-rank is read from that draw's own ordering; "
                     "one-sided toward small ranks (1 = top of the vote); p = (1 + #{sim <= obs}) "
                     "/ (R + 1)"),
            "ties": {"method": "average (mid-rank), scipy.stats.rankdata",
                     "draws_whose_ordering_has_ties": n_tied_struct,
                     "draws_where_drawn_value_is_tied": int(sum(1 for r in rows if r["rank_tie_size"] > 1))},
            "rank_sources": sources,
            "role": "DESCRIPTIVE: not in the multiplicity family; the counted verdict rule is unchanged",
            "control": {"mc_mean_sum": mc_mean, "exact_null_mean_sum": exact_mean, "z": zc,
                        "tolerance_z": tol, "passed": bool(abs(zc) <= tol)}}


def session_breakdown(graded, sessions, cfg, rank_on):
    """R25: the same statistics per session, descriptive and unadjusted."""
    out = {}
    for s in sessions:
        sub = [e for e in graded if e["session"] == s]
        if not sub:
            out[s] = {"n_graded": 0}
            continue
        st, _ = arm_statistics(sub, cfg["draw_space"])
        out[s] = {"n_graded": st["n_graded"], "hits": st["hits"], "expected_hits": st["expected_hits"],
                  "null_sd": st["null_sd"], "sum_k": st["sum_k"], "lift_point": st["lift_point"],
                  "p_one_sided_unadjusted": st["p_one_sided"],
                  "role": "DESCRIPTIVE per-session breakdown; unadjusted; not a counted test"}
        if rank_on:
            out[s]["rank_summary"] = rank_summary(sub, cfg)
    return out


# -------------------------------------------------------------------------- provenance ---
def resolve_provenance(repo_root, authoritative):
    """Explicit-root provenance, fatal on failure when authoritative (pattern:
    watcher_kpi_metricC_deterministic_v2_2.py:23-34)."""
    prov = {"authoritative": authoritative, "grader_path": str(Path(__file__).resolve()),
            "grader_sha256": sha256_file(Path(__file__).resolve()),
            "grader_version": GRADER_VERSION, "python": platform.python_version(),
            "numpy": np.__version__, "argv": sys.argv}
    try:
        import scipy
        prov["scipy"] = scipy.__version__
    except Exception:  # pragma: no cover
        prov["scipy"] = None
    if not authoritative:
        prov.update(repo_root=None, source_commit=None, tree_dirty=None)
        return prov
    try:
        head = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        st = subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain"],
                            capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise GraderError(f"provenance: cannot resolve git state at {repo_root}: {e} "
                          f"(use --no-provenance for a non-authoritative run)")
    prov.update(repo_root=str(Path(repo_root).resolve()), source_commit=head,
                tree_dirty=bool(st.strip()))
    return prov


# ----------------------------------------------------------------------------- outputs ---
def _check_out_dir(out_dir):
    p = Path(out_dir).resolve()
    for pref in FORBIDDEN_OUT_PREFIXES:
        if str(p) == pref or str(p).startswith(pref + "/"):
            raise GraderError(f"out-dir {p} is under {pref}; write under the project's data/ (R21)")
    if p.exists():
        raise GraderError(f"out-dir {p} already exists; the grader writes only to a new out-dir")
    if not p.parent.is_dir():
        raise GraderError(f"out-dir parent {p.parent} does not exist")
    return p


def _row_out(e):
    return {k: e.get(k) for k in ("arm_id", "config_id", "combined_index", "date", "session",
                                  "data_cutoff_index", "pool_sha256", "k_raw", "k",
                                  "n_duplicate_values", "p_baseline", "outcome", "hit",
                                  "status", "abstain_reason", "source_line")}


def detail_row(e, mode):
    """R23: one per-draw detail row. `vote_rank` is the 1-based FIRST position of the drawn
    number in the raw pool as written (None if absent or not graded)."""
    graded = e.get("status") == "graded"
    y = e.get("outcome") if graded else None
    raw = e.get("raw_pool") or []
    rank = (raw.index(y) + 1) if (graded and y in raw) else None
    row = {"arm_id": e.get("arm_id"), "config_id": e.get("config_id"),
           "combined_index": e.get("combined_index"), "date": e.get("date"),
           "session": e.get("session"), "status": e.get("status"), "drawn": y,
           "k": e.get("k"), "k_raw": e.get("k_raw"), "hit": e.get("hit") if graded else None,
           "vote_rank": rank, "vote_count": raw.count(y) if graded else None,
           "pool_order": e.get("pool_order"), "pool_sha256": e.get("pool_sha256"),
           "rank": e.get("rank") if graded else None,                   # R24
           "rank_tie_size": e.get("rank_tie_size") if graded else None,
           "rank_source": e.get("rank_source") if graded else None,
           "rank_basis": e.get("rank_basis")}
    if mode == "development":
        row["mode"] = "DEVELOPMENT"
    return row


def _dev_headline(r, n_elig):
    """R22: development wording only; never the counted verdict phrases."""
    s = r["stats"]
    return (f"DEVELOPMENT, NOT A COUNTED VERDICT. {r['arm_id']}: {s['hits']} hits of "
            f"{s['n_graded']} graded draws (expected {s['expected_hits']:.2f} at "
            f"k/{r['draw_space']}; abstained {r['n_abstained']} of {n_elig} eligible; pending "
            f"{r['n_pending']}); one-sided p {s['p_one_sided']:.4g}, adjusted p "
            f"{r['p_adjusted']:.4g} (m={r['m']}, {r['correction']}); lift lower bound "
            f"{r['lift_lower_bound_adjusted']:.3f}; rate "
            f"{'above' if r['development_above_background'] else 'not above'} the modelled "
            f"background after correction")


def _headline(r, n_elig):
    s = r["stats"]
    lb = r["lift_lower_bound_adjusted"]
    return (f"{r['arm_id']}: {s['hits']} hits of {s['n_graded']} graded draws "
            f"(expected {s['expected_hits']:.2f} at k/{r['draw_space']}; "
            f"abstained {r['n_abstained']} of {n_elig} eligible; pending {r['n_pending']}); "
            f"one-sided p {s['p_one_sided']:.4g}, adjusted p {r['p_adjusted']:.4g} "
            f"(m={r['m']}, {r['correction']}); lift lower bound {lb:.3f}; "
            f"{'ABOVE CHANCE after correction' if r['above_chance'] else 'NULL: not above chance after correction'}")


def write_outputs(out_dir, per_draw_rows, summary, detail_rows=None):
    out_dir.mkdir(parents=False, exist_ok=False)
    development = summary.get("mode") == "DEVELOPMENT"
    extra = []
    if development:                         # R22: marker file, listed in SHA256SUMS
        (out_dir / "DEVELOPMENT").write_text(
            "DEVELOPMENT evidence: blocks looked at before; never a counted verdict (R22).\n",
            encoding="ascii")
        extra.append("DEVELOPMENT")
    pd = out_dir / "per_draw.jsonl"
    with open(pd, "w", encoding="ascii") as f:
        for row in per_draw_rows:
            f.write(canonical_json(row) + "\n")
    summary["outputs"] = {"per_draw.jsonl": {"sha256": sha256_file(pd),
                                             "lines": len(per_draw_rows)}}
    if detail_rows is not None:             # R23
        dp = out_dir / "per_draw_detail.jsonl"
        with open(dp, "w", encoding="ascii") as f:
            for row in detail_rows:
                f.write(canonical_json(row) + "\n")
        summary["outputs"]["per_draw_detail.jsonl"] = {"sha256": sha256_file(dp),
                                                       "lines": len(detail_rows)}
        extra.append("per_draw_detail.jsonl")
    sp = out_dir / "summary.json"
    sp.write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n", encoding="ascii")
    md = out_dir / "SUMMARY.md"
    title = "DEVELOPMENT grade summary, NOT COUNTED" if development else "Grade summary"
    lines = [f"# {title}: {summary['grade_block']['name']} ({summary['generated_utc']})", "",
             f"Instrument: {GRADER_VERSION}. Not Beta-reviewed; not a counted result unless the "
             f"run record says the owner gave the go.", "",
             "## Controls", ""]
    for aid, c in summary["controls"].items():
        u, kc, fq = c["uniform_random"], c["known_count"], c["frequency"]
        lines.append(f"- {aid} uniform-random: mean {u['mean_hits']:.2f} vs expected "
                     f"{u['expected_hits']:.2f} over {u['reps']} reps (z {u['z']:.2f}, tolerance "
                     f"{u['tolerance_z']}), rejection rate {u['empirical_rejection_rate']:.3f} vs "
                     f"size {u['exact_test_size']:.3f}: {'PASS' if u['passed'] else 'FAIL'}")
        lines.append(f"- {aid} known-count: {kc['hits']} of expected {kc['expected_hits']} "
                     f"over {kc['n']} draws: {'PASS' if kc['passed'] else 'FAIL'}")
        lines.append(f"- {aid} draw-frequency control: {fq['hits']} hits of {fq['n_graded']} "
                     f"(expected {fq['expected_hits']:.2f}; unadjusted p "
                     f"{fq['p_one_sided_unadjusted']:.4g}); block rows {fq['n_block_rows']}")
    lines += ["", "## Arm results (null results first)", ""]
    for r in summary["results"]:
        lines.append("- " + r["headline"])
    for r in summary["results"]:              # R24, R25: descriptive sections, only when on
        rs = r.get("rank_summary")
        if rs:
            lines.append(f"- {r['arm_id']} rank (DESCRIPTIVE, not in the family): mean rank "
                         f"{rs['mean_rank']:.2f} vs {rs['expected_mean_rank']:.2f} expected over "
                         f"{rs['n']} draws; z {rs['z_normal_approx']:.2f}; one-sided Monte Carlo p "
                         f"{rs['p_lower_monte_carlo']:.4g} ({rs['reps']} reps); ties: mid-rank, "
                         f"drawn value tied on {rs['ties']['draws_where_drawn_value_is_tied']} draws")
        for s, ps_ in (r.get("per_session") or {}).items():
            if ps_.get("n_graded"):
                lines.append(f"- {r['arm_id']} {s} only (DESCRIPTIVE, unadjusted): {ps_['hits']} hits "
                             f"of {ps_['n_graded']} (expected {ps_['expected_hits']:.2f}); p "
                             f"{ps_['p_one_sided_unadjusted']:.4g}")
    if summary["comparisons"]:
        lines += ["", "## Declared comparisons", ""]
        for c in summary["comparisons"]:
            lines.append(f"- {c['arm']} vs {c['baseline']}: {c['n_common']} common draws; hits "
                         f"{c.get('hits_arm')} vs {c.get('hits_baseline')} (sum k "
                         f"{c.get('sum_k_arm')} vs {c.get('sum_k_baseline')}); {c['reason']}; "
                         f"p {c['p_one_sided']}, adjusted p {c['p_adjusted']}")
    md.write_text("\n".join(lines) + "\n", encoding="ascii")
    sums = [f"{sha256_file(out_dir / n)}  {n}" for n in ("per_draw.jsonl", "summary.json",
                                                         "SUMMARY.md") + tuple(extra)]
    (out_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="ascii")


# -------------------------------------------------------------------------------- main ---
def run(args):
    cfg = {"draw_space": args.draw_space, "duplicate_rule": args.duplicate_rule,
           "out_of_block_records": args.out_of_block_records,
           "require_record_dataset_sha256": not args.allow_record_without_dataset_sha256,
           "control_reps": args.control_reps, "control_seed": args.control_seed,
           "control_tolerance_z": args.control_tolerance_z, "workers": args.workers,
           "multiplicity_family": args.multiplicity_family,
           "descriptive_ci_level": args.descriptive_ci_level,
           "mode": args.mode, "per_draw_detail": args.per_draw_detail,
           "rank_summary": args.rank_summary, "rank_source": args.rank_source,
           "rank_test_reps": args.rank_test_reps, "rank_test_seed": args.rank_test_seed,
           "per_session_report": args.per_session_report}
    if cfg["mode"] not in MODES or cfg["per_draw_detail"] not in ("off", "on"):
        raise GraderError("mode must be counted|development and per_draw_detail off|on")
    if (cfg["rank_summary"] not in ("off", "on") or cfg["rank_source"] not in RANK_SOURCES
            or cfg["per_session_report"] not in ("off", "on") or cfg["rank_test_reps"] < 1):
        raise GraderError("rank/per-session settings out of range (R24, R25)")
    rank_on = cfg["rank_summary"] == "on"
    need_ranks = rank_on or cfg["per_draw_detail"] == "on"
    development = cfg["mode"] == "development"
    if cfg["draw_space"] < 2:
        raise GraderError("draw_space must be >= 2")
    if cfg["control_reps"] < 1 or cfg["workers"] < 1:
        raise GraderError("control_reps and workers must be >= 1")
    out_dir = _check_out_dir(args.out_dir)
    prov = resolve_provenance(args.repo_root, authoritative=not args.no_provenance)

    dataset, dsha = load_dataset(args.dataset, args.dataset_sha256, cfg["draw_space"])
    gb_obj, gb_sha = load_declaration(args.grade_block, SCHEMA_GRADE_BLOCK, "grade block")
    arms_obj, arms_sha = load_declaration(args.arms, SCHEMA_ARMS, "arm list")
    ab_obj, ab_sha = load_declaration(args.abstention, SCHEMA_ABSTENTION, "abstention rule")
    gb, arms_decl, abst = parse_grade_block(gb_obj), parse_arms(arms_obj), parse_abstention(ab_obj)
    if gb.get("mode") == "development" and not development:
        raise GraderError("grade block declares mode 'development'; it cannot be graded in "
                          "counted mode (R22). Pass --mode development explicitly.")
    pools_sha =sha256_file(args.pools) if Path(args.pools).is_file() else None
    if args.pools_sha256 and pools_sha != args.pools_sha256.lower():
        raise GraderError(f"pools sha256 mismatch: expected {args.pools_sha256}, got {pools_sha}")
    records = read_pool_records(args.pools)
    per_arm, eligible = resolve_records(records, arms_decl, gb, abst, dataset, dsha, cfg)

    per_draw_rows = []
    detail_rows = [] if cfg["per_draw_detail"] == "on" else None

    def _emit(e):
        row = _row_out(e)
        if development:                     # R22: stamped into every per_draw row
            row["mode"] = "DEVELOPMENT"
        per_draw_rows.append(row)
        if detail_rows is not None:         # R23
            detail_rows.append(detail_row(e, cfg["mode"]))

    arm_results, controls, graded_by_arm = [], {}, {}
    m_arms = len(arms_decl["arms"])
    m = m_arms + (len(arms_decl["comparisons"])
                  if cfg["multiplicity_family"] == "all_declared_tests" else 0)
    alpha = arms_decl["family_alpha"]
    alpha_test = alpha / m          # Bonferroni level; also Holm's first step
    controls_ok = True
    for aid in arms_decl["arms"]:
        bucket = per_arm[aid]
        graded = []
        for e in bucket["graded"]:
            y = dataset[e["combined_index"]][2]
            e = dict(e, outcome=y, hit=grade_draw(e["distinct"], y),
                     p_baseline=e["k"] / cfg["draw_space"], status="graded")
            graded.append(e)
        if not graded:
            raise GraderError(f"arm {aid}: every eligible draw abstained; nothing to grade")
        graded_by_arm[aid] = graded
        if need_ranks:                      # R24: rank of the drawn number, per graded draw
            for e in graded:
                rk, src = draw_ranks(e, cfg["draw_space"], cfg["rank_source"])
                y = e["outcome"]
                e["rank"] = float(rk[y])
                e["rank_tie_size"] = int((rk == rk[y]).sum())
                e["rank_source"] = src
                e["_ranks"] = rk.astype(np.float32)
        stats, ps = arm_statistics(graded, cfg["draw_space"])
        ci_a = (1 - cfg["descriptive_ci_level"]) / 2
        d_lo, d_hi = lift_bounds(ps, stats["hits"], ci_a, ci_a)
        arm_results.append({"arm_id": aid, "config_id": arms_decl["arms"][aid]["config_id"],
                            "description": arms_decl["arms"][aid]["description"],
                            "draw_space": cfg["draw_space"], "stats": stats,
                            "n_eligible": len(eligible), "n_abstained": len(bucket["abstained"]),
                            "n_pending": len(bucket["pending"]),
                            "n_excluded_out_of_block": len(bucket["excluded"]),
                            "duplicates_resolved": bucket["duplicates_resolved"],
                            "abstention_reasons": sorted({e["abstain_reason"] for e in bucket["abstained"]}),
                            "lift_ci_descriptive": {"level": cfg["descriptive_ci_level"],
                                                    "lower": d_lo, "upper": d_hi},
                            "_ps": ps})
        outcomes = [e["outcome"] for e in graded]
        urc = uniform_random_control(graded, outcomes, cfg, alpha_test)
        kcc = known_count_control(graded, cfg["draw_space"])
        frq = frequency_control(graded, dataset, gb["frequency_control_block"], cfg["draw_space"])
        controls[aid] = {"uniform_random": urc, "known_count": kcc, "frequency": frq}
        controls_ok = controls_ok and urc["passed"] and kcc["passed"]
        if rank_on:                         # R24, descriptive; its MC control gates outputs
            rs = rank_summary(graded, cfg)
            arm_results[-1]["rank_summary"] = rs
            controls[aid]["rank_test"] = rs["control"]
            controls_ok = controls_ok and rs["control"]["passed"]
        if cfg["per_session_report"] == "on":   # R25
            arm_results[-1]["per_session"] = session_breakdown(graded, gb["sessions"], cfg, rank_on)
        for e in graded:
            e.pop("_ranks", None)
        for e in graded:
            _emit(e)
        for e in bucket["abstained"]:
            _emit(dict(e, status="abstained", outcome=None, hit=None, p_baseline=None))
        for e in bucket["pending"]:
            _emit(dict(e, status="pending", outcome=None, hit=None,
                       p_baseline=e["k"] / cfg["draw_space"]))
        for e in bucket["excluded"]:
            _emit(dict(e, status="excluded_out_of_block", outcome=None, hit=None,
                       p_baseline=None))
    per_draw_rows.sort(key=lambda r: (r["arm_id"], r["combined_index"]))
    if detail_rows is not None:
        detail_rows.sort(key=lambda r: (r["arm_id"], r["combined_index"]))

    comparisons = [dict(c, **compare_arms(graded_by_arm[c["arm"]], graded_by_arm[c["baseline"]],
                                          cfg["draw_space"])) for c in arms_decl["comparisons"]]
    pvals = [r["stats"]["p_one_sided"] for r in arm_results]
    if cfg["multiplicity_family"] == "all_declared_tests":
        pvals += [c["p_one_sided"] for c in comparisons]
    adj = adjust_pvalues(pvals, arms_decl["correction"])
    if len(adj) != m:
        raise GraderError("internal: multiplicity family size mismatch")
    for i, r in enumerate(arm_results):
        r["m"], r["correction"], r["p_adjusted"] = m, arms_decl["correction"], adj[i]
        r["above_chance"] = bool(adj[i] is not None and adj[i] <= alpha)
        lo, _ = lift_bounds(r.pop("_ps"), r["stats"]["hits"], alpha_test, None)
        r["lift_lower_bound_adjusted"] = lo
        r["lift_lower_bound_level"] = alpha_test
        r["headline"] = _headline(r, len(eligible))
    for j, c in enumerate(comparisons):
        c["p_adjusted"] = adj[m_arms + j] if cfg["multiplicity_family"] == "all_declared_tests" else None
        c["m"] = m
    arm_results.sort(key=lambda r: (r["above_chance"], r["arm_id"]))    # nulls first (R20)
    null_heads = [r["headline"] for r in arm_results if not r["above_chance"]]
    above_heads = [r["headline"] for r in arm_results if r["above_chance"]]
    dev_above_heads = []
    if development:                         # R22: no counted verdict can leave this path
        for r in arm_results:
            r["development_above_background"] = r["above_chance"]
            r["above_chance"] = None
            r["counted_verdict"] = None
            r["previously_inspected_overlaps"] = per_arm[r["arm_id"]]["previously_inspected_overlaps"]
            r["headline"] = _dev_headline(r, len(eligible))
        null_heads = [r["headline"] for r in arm_results if not r["development_above_background"]]
        dev_above_heads = [r["headline"] for r in arm_results if r["development_above_background"]]
        above_heads = []

    summary = {
        "schema": SCHEMA_SUMMARY,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "DEVELOPMENT" if development else "COUNTED",
        "status": ("DEVELOPMENT: previously inspected or walk-forward blocks; never a counted "
                   "verdict (R22)" if development else
                   "DRAFT instrument, not Beta-reviewed; not a counted result"),
        "run_label": ("DEVELOPMENT: " + args.run_label) if development else args.run_label,
        "controls_ok": bool(controls_ok),
        "null_results_first": null_heads,
        "above_chance": above_heads,
        "denominators": {"eligible_grade_block_draws": len(eligible),
                         "per_arm": {r["arm_id"]: {"graded": r["stats"]["n_graded"],
                                                   "abstained": r["n_abstained"],
                                                   "pending": r["n_pending"],
                                                   "excluded_out_of_block": r["n_excluded_out_of_block"]}
                                     for r in arm_results}},
        "inputs": {"pools": {"path": str(Path(args.pools).resolve()), "sha256": pools_sha,
                             "records": len(records)},
                   "dataset": {"path": str(Path(args.dataset).resolve()), "sha256": dsha,
                               "rows": len(dataset)},
                   "grade_block": {"path": str(Path(args.grade_block).resolve()), "sha256": gb_sha},
                   "arms": {"path": str(Path(args.arms).resolve()), "sha256": arms_sha},
                   "abstention": {"path": str(Path(args.abstention).resolve()), "sha256": ab_sha}},
        "grade_block": gb, "abstention_rule": abst,
        "declared": {"family_alpha": alpha, "correction": arms_decl["correction"],
                     "multiplicity_family": cfg["multiplicity_family"], "m": m,
                     "arms": [{k: v for k, v in a.items()} for a in arms_decl["arms"].values()],
                     "comparisons": arms_decl["comparisons"]},
        "config": cfg, "defaults": DEFAULTS, "provenance": prov,
        "results": arm_results, "comparisons": comparisons, "controls": controls,
        "claims": "None about TFM (WP-3). Hit counts against k/draw_space on the declared "
                  "draws, under the declared upstream-unseen boundaries, only.",
    }
    if development:
        summary["development_above_background"] = dev_above_heads
    write_outputs(out_dir, per_draw_rows, summary, detail_rows)
    for h in summary["null_results_first"] + summary["above_chance"] + dev_above_heads:
        print(h, flush=True)
    print(f"controls_ok={controls_ok} out_dir={out_dir}", flush=True)
    return 0 if controls_ok else 3


def build_parser():
    ap = argparse.ArgumentParser(description="TFM hit-rate grader (WP-3 first form, draft).")
    ap.add_argument("--pools", required=True, help="per-draw pools JSONL (tfm.e2e.pool.v1)")
    ap.add_argument("--pools-sha256", default=None, help="optional expected sha256 of --pools")
    ap.add_argument("--dataset", required=True, help="frozen dataset JSON [{date,session,draw}]")
    ap.add_argument("--dataset-sha256", required=True, help="expected sha256 of --dataset")
    ap.add_argument("--grade-block", required=True, help="grade block definition JSON")
    ap.add_argument("--arms", required=True, help="declared arm list JSON")
    ap.add_argument("--abstention", required=True, help="declared abstention rule JSON")
    ap.add_argument("--out-dir", required=True, help="NEW directory to write (must not exist)")
    ap.add_argument("--run-label", default="", help="free-text label recorded in the summary")
    ap.add_argument("--draw-space", type=int, default=DEFAULTS["draw_space"])
    ap.add_argument("--duplicate-rule", choices=["error", "first_in_file", "max_data_cutoff"],
                    default=DEFAULTS["duplicate_rule"])
    ap.add_argument("--out-of-block-records", choices=["error", "exclude_and_count"],
                    default=DEFAULTS["out_of_block_records"])
    ap.add_argument("--allow-record-without-dataset-sha256", action="store_true",
                    help="accept pool records that omit dataset_sha256 (a present one must match)")
    ap.add_argument("--multiplicity-family", choices=["all_declared_tests", "arms_only"],
                    default=DEFAULTS["multiplicity_family"])
    ap.add_argument("--control-reps", type=int, default=DEFAULTS["control_reps"])
    ap.add_argument("--control-seed", type=int, default=DEFAULTS["control_seed"])
    ap.add_argument("--control-tolerance-z", type=float, default=DEFAULTS["control_tolerance_z"])
    ap.add_argument("--workers", type=int, default=DEFAULTS["workers"],
                    help="processes for the uniform-random control (multi-core)")
    ap.add_argument("--descriptive-ci-level", type=float, default=DEFAULTS["descriptive_ci_level"])
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]),
                    help="repository whose commit is recorded (default: this file's repo)")
    ap.add_argument("--mode", choices=list(MODES), default=DEFAULTS["mode"],
                    help="counted (default) or development (R22: must be chosen explicitly; "
                         "never a counted verdict)")
    ap.add_argument("--per-draw-detail", choices=["off", "on"], default=DEFAULTS["per_draw_detail"],
                    help="on: also write per_draw_detail.jsonl (R23)")
    ap.add_argument("--rank-summary", choices=["off", "on"], default=DEFAULTS["rank_summary"],
                    help="on: per-arm rank of the drawn number vs uniform (R24, descriptive)")
    ap.add_argument("--rank-source", choices=list(RANK_SOURCES), default=DEFAULTS["rank_source"])
    ap.add_argument("--rank-test-reps", type=int, default=DEFAULTS["rank_test_reps"])
    ap.add_argument("--rank-test-seed", type=int, default=DEFAULTS["rank_test_seed"])
    ap.add_argument("--per-session-report", choices=["off", "on"], default=DEFAULTS["per_session_report"],
                    help="on: each session on its own beside the pooled result (R25, descriptive)")
    ap.add_argument("--no-provenance", action="store_true",
                    help="non-authoritative run: do not resolve git state")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except GraderError as e:
        print(f"GRADER ERROR (nothing written): {e}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
