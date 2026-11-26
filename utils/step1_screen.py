"""
utils/step1_screen.py - Step 1 cheap screen: score a trial by its OWN unlearned pools on a
selection block (TASK_CHEAP_SCREEN, lead 2026-11-23 20:21; owner 20:17/20:20; LL-17b of
PLAN_CH14_LEARNING_LOOP.md). Team Alpha (Claude). Default off.

Design: docs/drafts/anchor_v1/DESIGN_CHEAP_SCREEN_20261123.md.

Setting: --step1-objective {bidirectional_count, screen_rank, screen_hits}, default
bidirectional_count (= today: BidirectionalCountScorer, window_optimizer.py). With the default
nothing in this module runs.

WHAT A SCREENED TRIAL IS SCORED BY
  * Survivors: the trial's own bidirectional records (the canonical records of its miner
    assembly, miner/step1_ingress.py:219-220), unlearned.
  * Ranking: the A0 key of the run H unlearned arm, sum of the declared match-rate fields as
    float32 (the NPZ dtype, npz forward_matches == canonical forward_match_rate), ties by the
    declared tie-break (default seed_asc) - e2e_vote_harness.py build_key/tie_break_key and
    run H full_A0/run_record.json.
  * Vote: Step 6's own rule, prediction_generator.py:836-850: for target t, the top-k ranked
    survivors each vote seq[t] of their own sequence (SurvivorScorer._generate_sequence, the
    survivor's skip by PredictionGenerator._get_survivor_skip, fallback default_skip).
  * Selection block: [s0, s1) in combined index, s0 = (combined index of the window's last
    draw) + 1 + gap, s1 = s0 + selection_draws. It must end at or before selection_end_max
    (default the Step 1 holdout split) and at or before every confirmation block's start, and
    may not overlap one. The anchor domain is narrowed so every sampled anchor fits (never
    clamped: a pinned anchor that does not fit is refused).
  * Score: screen_hits = -log10 P(X >= H) under the grader's exact Poisson-binomial null
    (k_t distinct, p_t = k_t / draw_space; e2e_grader.py R4, R7, R8: pb_upper_tail);
    screen_rank = -z of the drawn number's mid-rank in the full-vote ordering against the
    grader's exact per-draw null (R24: draw_ranks, rank_summary). Both are 0 for a trial
    with no survivors (empty pools: p = 1, no rank spread).
  * Mode: every screen number is DEVELOPMENT evidence (R22 wording); never a counted result.

PHASE C (docs/TB_RULING_S176_WATCHER_KPI.md:692-727): the only draw VALUES this module reads
are those of the selection block, through ScreenPlan.draw_value, which refuses any other
index. The vote for target t uses the index t only (Step 6 passes history[:t] and reads only
its length at :839). Nothing at or after s1 is read; nothing in a confirmation block is read.

The grader is called, not copied: it is loaded from a path pinned by sha256 (the walk-forward
driver's rule, e2e_walk_forward.py build_declaration). The grader file is a draft in another
worktree (TODO-SCREEN-1).
"""

import hashlib
import importlib.util
import inspect
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

STEP1_OBJECTIVES = ("bidirectional_count", "screen_rank", "screen_hits")
DEFAULT_STEP1_OBJECTIVE = "bidirectional_count"
SCREEN_OBJECTIVES = ("screen_rank", "screen_hits")
SCREEN_SCHEMA = "tfm.step1_screen.v1"
SCREEN_STUDY_ATTR = "step1_objective"
SCREEN_TRIAL_ATTR = "step1_screen"

# Declared defaults. Each is a key of the --screen-config JSON; the resolved settings and
# their sha256 are recorded with the run and bound to the study (screen_study_check).
SCREEN_DEFAULTS: Dict[str, Any] = {
    "schema": SCREEN_SCHEMA,
    "grader_path": None,              # REQUIRED: e2e_grader.py with R22/R24
    "grader_sha256": None,            # REQUIRED
    "pool_size": 20,                  # k: Step 6's smallest pool (20/100/300); run H A0 k=20
    "selection_draws": 600,           # combined rows per selection block (WF block length)
    "selection_gap": 0,               # rows skipped between the window's last draw and s0
    "selection_sessions": "all",      # "all" | "trial" | list of session names, graded rows
    "selection_end_max": "holdout_split",   # "holdout_split" | int (exclusive bound on s1)
    "confirmation_blocks": "holdout",  # "holdout" = [[split, N]] | list of [start, end)
    "ranking_fields": ["forward_match_rate", "reverse_match_rate"],   # A0 key (NPZ names
                                      # forward_matches/reverse_matches hold these values)
    "tie_break": "seed_asc",          # run H A0
    "prng_type": "java_lcg",          # prediction_generator_config.json:23
    "mod": 1000,                      # prediction_generator_config.json:24
    "draw_space": 1000,
    "default_skip": 0,                # run H A0 effective_skip_counts {"0": 12156}
    "rank_survivor_cap": 0,           # screen_rank: top-M ranked survivors build the
                                      # ordering (0 = all, Step 6's ordering exactly)
    "rank_test_reps": 2000,           # grader R24 Monte Carlo reps (its control check)
    "rank_test_seed": 0,
    "control_tolerance_z": 4.0,
    "workers": 1,                     # >1: spawn pool (Step 1 holds threads/sockets)
    "mode": "development",
}
TIE_BREAKS = ("seed_asc",)
GRADER_FUNCS = {"pb_upper_tail": ["ps", "h"],
                "grade_draw": ["distinct_values", "outcome"],
                "validate_pool_values": ["raw_pool", "draw_space", "where"],
                "draw_ranks": ["e", "draw_space", "rank_source"],
                "rank_summary": ["rows", "cfg"]}


class Step1ScreenError(ValueError):
    """A screen setting, block or input is refused. Never a silent fallback."""


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def validate_step1_objective(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or value not in STEP1_OBJECTIVES:
        raise Step1ScreenError(f"{where}: step1_objective must be one of {STEP1_OBJECTIVES}, "
                               f"got {value!r}")
    return value


def load_grader(path: str, expected_sha256: str):
    """Load the pinned grader module; refuse a sha256 mismatch or a signature drift."""
    if not path or not expected_sha256:
        raise Step1ScreenError("screen: grader_path and grader_sha256 are required")
    try:
        got = sha256_file(path)
    except OSError as exc:
        raise Step1ScreenError(f"screen: grader {path!r} unreadable: {exc}") from exc
    if got != str(expected_sha256).lower():
        raise Step1ScreenError(f"screen: grader {path} sha256 {got} != pinned {expected_sha256}")
    spec = importlib.util.spec_from_file_location(f"_tfm_screen_grader_{got[:12]}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name, params in GRADER_FUNCS.items():
        fn = getattr(mod, name, None)
        if fn is None or list(inspect.signature(fn).parameters) != params:
            raise Step1ScreenError(f"screen: grader {path} {name} missing or signature != {params}")
    return mod, got


def resolve_settings(user: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    s = dict(SCREEN_DEFAULTS)
    for k, v in (user or {}).items():
        if k not in SCREEN_DEFAULTS:
            raise Step1ScreenError(f"screen config: unknown key {k!r}")
        s[k] = v
    if s["schema"] != SCREEN_SCHEMA:
        raise Step1ScreenError(f"screen config: schema {s['schema']!r} != {SCREEN_SCHEMA}")
    for k in ("pool_size", "selection_draws", "draw_space", "mod", "rank_test_reps", "workers"):
        if isinstance(s[k], bool) or not isinstance(s[k], int) or s[k] < 1:
            raise Step1ScreenError(f"screen config: {k} must be an int >= 1, got {s[k]!r}")
    for k in ("selection_gap", "default_skip", "rank_survivor_cap", "rank_test_seed"):
        if isinstance(s[k], bool) or not isinstance(s[k], int) or s[k] < 0:
            raise Step1ScreenError(f"screen config: {k} must be an int >= 0, got {s[k]!r}")
    if s["tie_break"] not in TIE_BREAKS:
        raise Step1ScreenError(f"screen config: tie_break must be one of {TIE_BREAKS}")
    if (not isinstance(s["ranking_fields"], list) or not s["ranking_fields"]
            or not all(isinstance(x, str) for x in s["ranking_fields"])):
        raise Step1ScreenError("screen config: ranking_fields must be a non-empty list of names")
    if s["mode"] != "development":
        raise Step1ScreenError("screen config: mode must be 'development' (a screen score is "
                               "never a counted result)")
    ss = s["selection_sessions"]
    if not (ss in ("all", "trial") or (isinstance(ss, list) and ss
                                        and all(isinstance(x, str) for x in ss))):
        raise Step1ScreenError("screen config: selection_sessions must be 'all', 'trial' or a list")
    return s


@dataclass
class ScreenPlan:
    """One run's screen, resolved once before the first trial."""
    objective: str
    settings: Dict[str, Any]
    settings_sha256: str
    dataset_path: str
    dataset_sha256: str
    sessions_by_row: List[str]
    combined_by_option: Dict[Tuple[str, ...], List[int]]
    selection_end_max: int
    confirmation_blocks: List[Tuple[int, int]]
    grader: Any = field(repr=False, default=None)
    grader_sha256: str = ""
    _draws: List[int] = field(repr=False, default_factory=list)
    _open_block: Optional[Tuple[int, int]] = field(repr=False, default=None)

    # ---- identity -------------------------------------------------------------------
    def tag(self) -> str:
        return f"{self.objective}:{self.settings_sha256}"

    def run_record(self) -> Dict[str, Any]:
        return {"objective": self.objective, "tag": self.tag(), "schema": SCREEN_SCHEMA,
                "settings": self.settings, "settings_sha256": self.settings_sha256,
                "dataset_sha256": self.dataset_sha256, "grader_sha256": self.grader_sha256,
                "selection_end_max": self.selection_end_max,
                "confirmation_blocks": [list(b) for b in self.confirmation_blocks],
                "selection_region": [0, self.selection_end_max],
                "mode": "development",
                "label": "DEVELOPMENT: selection-block evidence; never a counted result",
                "walk_forward_rule": (f"grade blocks of any run that uses a window chosen by this "
                                      f"study start at or after {self.selection_end_max} "
                                      f"(--range-start {self.selection_end_max}); the selection "
                                      f"blocks used are per trial in user attr "
                                      f"'{SCREEN_TRIAL_ATTR}' and go to the grader as "
                                      f"previously_inspected_blocks")}

    # ---- geometry -------------------------------------------------------------------
    def _comb(self, sessions: Sequence[str]) -> List[int]:
        key = tuple(sorted(sessions))
        if key not in self.combined_by_option:
            raise Step1ScreenError(f"screen: sessions {list(key)} not a resolved option")
        return self.combined_by_option[key]

    def window_combined(self, anchor: int, window_size: int,
                        sessions: Sequence[str]) -> Tuple[int, int]:
        """Combined indices of the window's first and last draw (filtered [a, a+w))."""
        comb = self._comb(sessions)
        if anchor < 0 or anchor + window_size > len(comb):
            raise Step1ScreenError(f"screen: window [{anchor}, {anchor + window_size}) outside "
                                   f"the {len(comb)} filtered rows of {sorted(sessions)}")
        return comb[anchor], comb[anchor + window_size - 1]

    def _limit(self) -> int:
        lim = self.selection_end_max
        for b0, _ in self.confirmation_blocks:
            lim = min(lim, b0)
        return lim

    def anchor_max(self, window_size: int, sessions: Sequence[str]) -> int:
        """Largest anchor whose selection block fits under the limit; -1 if none does."""
        comb = self._comb(sessions)
        need = self._limit() - self.settings["selection_draws"] - self.settings["selection_gap"] - 1
        # largest filtered index f with comb[f] <= need
        import bisect
        f = bisect.bisect_right(comb, need) - 1
        return f - int(window_size) + 1

    def selection_block(self, anchor: int, window_size: int,
                        sessions: Sequence[str]) -> Tuple[int, int]:
        _, c_last = self.window_combined(anchor, window_size, sessions)
        s0 = c_last + 1 + self.settings["selection_gap"]
        s1 = s0 + self.settings["selection_draws"]
        self.check_block(s0, s1, c_last)
        return s0, s1

    def check_block(self, s0: int, s1: int, window_last: int) -> None:
        """Refusals, each named. Never adjusted."""
        if s0 <= window_last:
            raise Step1ScreenError(f"screen: selection block [{s0}, {s1}) does not start after "
                                   f"the sieve window (last draw {window_last})")
        if s1 > len(self.sessions_by_row):
            raise Step1ScreenError(f"screen: selection block [{s0}, {s1}) runs past the dataset "
                                   f"({len(self.sessions_by_row)} rows)")
        if s1 > self.selection_end_max:
            raise Step1ScreenError(f"screen: selection block [{s0}, {s1}) ends after "
                                   f"selection_end_max {self.selection_end_max}")
        for b0, b1 in self.confirmation_blocks:
            if s0 < b1 and b0 < s1:
                raise Step1ScreenError(f"screen: selection block [{s0}, {s1}) overlaps "
                                       f"confirmation block [{b0}, {b1})")
            if s1 > b0:
                raise Step1ScreenError(f"screen: selection block [{s0}, {s1}) ends after the "
                                       f"start of confirmation block [{b0}, {b1}) (S176 "
                                       f"Phase C: no later draw selects a window for it)")

    def graded_targets(self, s0: int, s1: int, sessions: Sequence[str]) -> List[int]:
        ss = self.settings["selection_sessions"]
        allowed = (None if ss == "all" else set(sessions) if ss == "trial" else set(ss))
        return [t for t in range(s0, s1)
                if allowed is None or self.sessions_by_row[t] in allowed]

    def draw_value(self, t: int) -> int:
        """The ONLY read of a draw value (Phase C). Inside the open selection block only."""
        blk = self._open_block
        if blk is None or not (blk[0] <= t < blk[1]):
            raise Step1ScreenError(f"screen: draw value at {t} read outside the selection "
                                   f"block {blk} (S176 Phase C)")
        return int(self._draws[t])

    # ---- scoring --------------------------------------------------------------------
    def rank_order(self, records: Sequence[Mapping[str, Any]]) -> List[int]:
        import numpy as np
        keys = []
        for i, r in enumerate(records):
            tot = 0.0
            for name in self.settings["ranking_fields"]:
                if name not in r:
                    raise Step1ScreenError(f"screen: record {i} (seed {r.get('seed')}) has no "
                                           f"ranking field {name!r}")
                v = float(np.float32(r[name]))
                if not math.isfinite(v):
                    raise Step1ScreenError(f"screen: {name} not finite for seed {r.get('seed')}")
                tot += v
            keys.append(tot)
        seeds = [int(r["seed"]) for r in records]
        return sorted(range(len(records)), key=lambda i: (-keys[i], seeds[i]))

    def _skips(self, records):
        from prediction_generator import PredictionGenerator
        fb = self.settings["default_skip"]
        # unbound call: _get_survivor_skip reads only the survivor (prediction_generator.py:530-552)
        return [PredictionGenerator._get_survivor_skip(None, dict(r), fb) for r in records]

    def score(self, records: Sequence[Mapping[str, Any]], anchor: int, window_size: int,
              sessions: Sequence[str]) -> Tuple[float, Dict[str, Any]]:
        t0 = time.time()
        s = self.settings
        s0, s1 = self.selection_block(anchor, window_size, sessions)
        targets = self.graded_targets(s0, s1, sessions)
        # Variable-mode canonical records carry <base>_hybrid (utils/canonical_records.py:343-344).
        # Step 6 votes every survivor with its config prng_type (prediction_generator.py:263-266),
        # so both kinds vote with the base type here; anything else is refused. TODO-SCREEN-4.
        kinds = {}
        allowed = (s["prng_type"], s["prng_type"] + "_hybrid")
        for r in records:
            pt = r.get("prng_type")
            if pt is not None and pt not in allowed:
                raise Step1ScreenError(f"screen: record prng_type {pt!r} is not one of {allowed} "
                                       f"(refused, not re-generated)")
            kinds[str(pt)] = kinds.get(str(pt), 0) + 1
        order = self.rank_order(records)
        skips = self._skips(records)
        k = s["pool_size"]
        need_rank = self.objective == "screen_rank"
        cap = s["rank_survivor_cap"]
        n_vote = (len(order) if (need_rank and cap == 0) else
                  min(len(order), max(k, cap)) if need_rank else min(len(order), k))
        voters = order[:n_vote]
        V = vote_matrix([int(records[i]["seed"]) for i in voters], [skips[i] for i in voters],
                        targets, s["prng_type"], s["mod"], s["workers"])
        g = self.grader
        self._open_block = (s0, s1)
        try:
            ps, hits, rows = [], 0, []
            for j, t in enumerate(targets):
                drawn = self.draw_value(t)
                raw_pool = [int(x) for x in V[:k, j]] if len(voters) else []
                distinct, _, _ = g.validate_pool_values(raw_pool, s["draw_space"], f"t={t}")
                kt = len(distinct)
                ps.append(kt / s["draw_space"])
                hits += g.grade_draw(distinct, drawn)
                if need_rank:
                    vs = value_scores_for(V[:, j], s["draw_space"])
                    ranks, src = g.draw_ranks({"value_scores": vs.tolist(), "source_line": t},
                                              s["draw_space"], "record_only")
                    rk = float(ranks[drawn])
                    rows.append({"rank": rk, "_ranks": ranks, "rank_source": src,
                                 "rank_tie_size": int((ranks == rk).sum())})
        finally:
            self._open_block = None
        p_hits = float(g.pb_upper_tail(ps, hits))
        rec = {"objective": self.objective, "selection_block": [s0, s1],
               "graded_draws": len(targets), "survivors": len(records), "voters": len(voters),
               "pool_size": k, "hits": int(hits), "expected_hits": float(sum(ps)),
               "p_hits_exact": p_hits, "screen_hits": -math.log10(max(p_hits, 1e-300)),
               "mode": "development", "tag": self.tag(), "record_prng_types": kinds,
               "vote_prng_type": s["prng_type"]}
        if need_rank:
            if rows:
                cfg = {"draw_space": s["draw_space"], "rank_test_reps": s["rank_test_reps"],
                       "rank_test_seed": s["rank_test_seed"],
                       "control_tolerance_z": s["control_tolerance_z"]}
                rs = g.rank_summary(rows, cfg)
                if not rs["control"]["passed"]:
                    raise Step1ScreenError(f"screen: grader rank control failed {rs['control']}")
                z = float(rs["z_normal_approx"])
                rec.update(mean_rank=rs["mean_rank"], expected_mean_rank=rs["expected_mean_rank"],
                           rank_z=z, rank_p_lower_mc=rs["p_lower_monte_carlo"],
                           rank_survivor_cap=cap)
            else:
                z = 0.0
                rec.update(mean_rank=None, rank_z=0.0)
            rec["screen_rank"] = -z
        value = rec["screen_rank"] if need_rank else rec["screen_hits"]
        value = float(value) + 0.0          # -0.0 (log10 of p = 1) recorded as 0.0
        rec["value"] = value
        rec["seconds"] = round(time.time() - t0, 3)
        return float(value), rec


def value_scores_for(col_ranked, n_values):
    """Grader R24 value_scores from one draw's votes in ranking order: -(first rank) per
    value, -(n+1) for values nobody votes. Same rule as e2e_vote_harness.py value_scores_for
    (FULL_VOTE_RANK_BASIS); the self-test proves equality against that file."""
    import numpy as np
    col = np.asarray(col_ranked)
    n = len(col)
    s = np.full(n_values, -(n + 1), dtype=np.int64)
    if n == 0:
        return s
    vals, first = np.unique(col, return_index=True)
    if vals.min() < 0 or vals.max() >= n_values:
        raise Step1ScreenError(f"vote value outside 0..{n_values - 1}")
    s[vals] = -(first + 1)
    return s


def _vote_chunk(args):
    prng_type, mod, seeds, skips, ts = args
    import numpy as np
    from survivor_scorer import SurvivorScorer
    sc = SurvivorScorer(prng_type=prng_type, mod=mod)
    n_len = (max(ts) + 1) if ts else 1
    out = np.empty((len(seeds), len(ts)), dtype=np.int16)
    for j, (sd, sk) in enumerate(zip(seeds, skips)):
        seq = sc._generate_sequence(int(sd), n_len, skip=int(sk))
        if len(seq) < n_len:
            raise Step1ScreenError(f"sequence for seed {sd} shorter than {n_len}")
        out[j] = seq[ts]
    return out


def vote_matrix(seeds, skips, ts, prng_type, mod, workers):
    """V[i, j] = seq_i[ts[j]] (Step 6: prediction_generator.py:846-849), rows in rank order."""
    import numpy as np
    if not seeds or not ts:
        return np.empty((len(seeds), len(ts)), dtype=np.int16)
    if workers <= 1 or len(seeds) < 2 * workers:
        return _vote_chunk((prng_type, mod, list(seeds), list(skips), list(ts)))
    import multiprocessing as mp
    size = -(-len(seeds) // (workers * 4))
    chunks = [(prng_type, mod, seeds[i:i + size], skips[i:i + size], list(ts))
              for i in range(0, len(seeds), size)]
    with mp.get_context("spawn").Pool(workers) as p:
        parts = p.map(_vote_chunk, chunks)
    return np.concatenate(parts, axis=0)


def build_screen_plan(objective: str, user_settings: Optional[Mapping[str, Any]],
                      dataset_path: str, session_options: Sequence[Sequence[str]], *,
                      split_point_fn, expected_sha256: Optional[str] = None) -> ScreenPlan:
    """Resolve the plan once, before any study or trial. Refuses rather than adjusts."""
    validate_step1_objective(objective, where="build_screen_plan")
    if objective not in SCREEN_OBJECTIVES:
        raise Step1ScreenError(f"build_screen_plan: {objective!r} is not a screen objective")
    s = resolve_settings(user_settings)
    grader, gsha = load_grader(s["grader_path"], s["grader_sha256"])
    sha = sha256_file(dataset_path)
    if expected_sha256 is not None and sha != expected_sha256:
        raise Step1ScreenError(f"screen: dataset sha256 {sha} != frozen {expected_sha256}")
    with open(dataset_path) as f:
        data = json.load(f)
    seen = {}
    for i, r in enumerate(data):            # grader R26 (e2e_grader.py load_dataset); contract :444
        key = (str(r["date"]), str(r["session"]))
        if key in seen:
            raise Step1ScreenError(f"screen: dataset rows {seen[key]} and {i}: repeated (date, session) {key}")
        seen[key] = i
    n = len(data)
    sessions_by_row = [str(r["session"]) for r in data]
    draws = [int(r["draw"]) for r in data]
    split = int(split_point_fn(n))
    end_max = split if s["selection_end_max"] == "holdout_split" else s["selection_end_max"]
    if isinstance(end_max, bool) or not isinstance(end_max, int) or not 0 < end_max <= n:
        raise Step1ScreenError(f"screen: selection_end_max {end_max!r} outside (0, {n}]")
    cb = s["confirmation_blocks"]
    blocks = [(split, n)] if cb == "holdout" else cb
    if not isinstance(blocks, list) and not isinstance(blocks, tuple):
        raise Step1ScreenError("screen: confirmation_blocks must be 'holdout' or a list")
    conf = []
    for b in blocks:
        if (not isinstance(b, (list, tuple)) or len(b) != 2 or not all(
                isinstance(x, int) and not isinstance(x, bool) for x in b) or not 0 <= b[0] < b[1] <= n):
            raise Step1ScreenError(f"screen: bad confirmation block {b!r}")
        if b[0] < end_max:
            raise Step1ScreenError(
                f"screen: confirmation block [{b[0]}, {b[1]}) starts before selection_end_max "
                f"{end_max}: a selection block could overlap it or come after it (refused)")
        conf.append((int(b[0]), int(b[1])))
    if not conf:
        raise Step1ScreenError("screen: at least one confirmation block must be declared")
    comb = {}
    for opt in session_options:
        key = tuple(sorted(opt))
        comb[key] = [i for i, ss in enumerate(sessions_by_row) if ss in key]
    settings_sha = hashlib.sha256(canonical_json(s).encode()).hexdigest()
    return ScreenPlan(objective=objective, settings=s, settings_sha256=settings_sha,
                      dataset_path=dataset_path, dataset_sha256=sha,
                      sessions_by_row=sessions_by_row, combined_by_option=comb,
                      selection_end_max=end_max, confirmation_blocks=conf,
                      grader=grader, grader_sha256=gsha, _draws=draws)


def load_screen_config(path: Optional[str]) -> Tuple[Dict[str, Any], Optional[str]]:
    if not path:
        raise Step1ScreenError("a screen objective needs --screen-config (grader pin required)")
    with open(path) as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise Step1ScreenError(f"--screen-config {path}: not a JSON object")
    return obj, sha256_file(path)


def screen_study_check(study, run_tag: Optional[str], trial_number: int) -> None:
    """Arm separation, as anchor_domain_study_check: the first trial of a screen run tags
    the study; a run whose tag differs (count objective, other screen settings) is refused."""
    attrs = study.user_attrs
    existing = attrs.get(SCREEN_STUDY_ATTR)
    if existing is None:
        if run_tag is not None:
            done = [t for t in study.get_trials(deepcopy=False)
                    if t.state.name == "COMPLETE"]
            if done:
                raise Step1ScreenError(f"screen run {run_tag} refused: study {study.study_name} "
                                       f"already has {len(done)} untagged completed trials "
                                       f"(survivor-count objective)")
            study.set_user_attr(SCREEN_STUDY_ATTR, run_tag)
        return
    if existing != run_tag:
        raise Step1ScreenError(f"trial {trial_number}: study {study.study_name} is tagged "
                               f"{SCREEN_STUDY_ATTR}={existing!r}; this run is {run_tag!r} "
                               f"(refused before the sieve)")
