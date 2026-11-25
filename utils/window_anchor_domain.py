"""Window-anchor domain: the one validator every Step-1 window choice passes through.

Window-anchor Brief II, flag-selectable (docs/drafts/anchor_v1/STATUS_AND_PLAN_ANCHOR.md §D;
design docs/PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md §4.2, §4.3).

A window is `filtered[anchor : anchor + window_size]` over the session-filtered history.
`window_anchor` is a window START INDEX. It is never a record envelope (100 is the
historical anchor ceiling; 149 = 100 + 50 - 1 is the historical envelope ceiling) and it is
never the generator phase, which stays pinned at 0 and is not read or written here.

Two modes, selected by ONE declared setting, `derived_anchor_domain` (default False):
  * off ("bounded"): today's behaviour. Nothing in this module runs.
  * on  ("derived"): the anchor domain is derived from the data after the session filter,
        effective_max = derived_max if max_cap is None else min(max_cap, derived_max),
        derived_max   = N_filtered - window_size,
    with `min` and `max_cap` from `distributed_config.json` `search_bounds.window_anchor`
    when present, else the declared defaults below (min 0, max_cap None — v1.1 §4.3).

The session filter has ONE authority, `miner.range_miner_worker.load_residue_window`
(see its docstring: "Do NOT reintroduce a second session-filter implementation"). This
module counts N_filtered and then PROVES the count against that function — anchor N-1 must
load and anchor N must be refused — so a divergent count fails closed instead of producing
a domain the worker would reject.

A second declared setting, `anchor_holdout_guard` (default False, meaningful only with the
derived domain), caps the anchor so the whole sieve window ends before the first holdout
record of Step 1's train/holdout split for that session filter:
    holdout_max = N_train(sessions) - window_size,
    N_train(sessions) = records of that session filter in data[:split_point],
    split_point = window_optimizer.step1_holdout_split_point(len(data)) (the split's own helper).
It narrows the domain; it never widens it and never clamps a choice.

Refusals raise WindowAnchorDomainError and always name the anchor, the domain, the
session set and the dataset sha256. Nothing is clamped.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

ANCHOR_DOMAIN_MODE_BOUNDED = "bounded"
ANCHOR_DOMAIN_MODE_DERIVED = "derived"
# The arm tag a study / search carries: the mode, plus the guard when it is on.
ANCHOR_DOMAIN_MODE_DERIVED_GUARDED = "derived_holdout_guard"

# Declared defaults (v1.1 §4.3, exact machine representation). Overridden by
# `search_bounds.window_anchor` in distributed_config.json when that key exists.
DEFAULT_WINDOW_ANCHOR_BOUNDS: Dict[str, Any] = {"min": 0, "max_cap": None}

# Names used on the study, the trial, the search space and the output config. Named once.
ANCHOR_DOMAIN_STUDY_ATTR = "window_anchor_domain"
ANCHOR_DOMAIN_TRIAL_ATTR = "window_anchor_domain"
ANCHOR_DOMAIN_SEARCH_KEY = "window_anchor_domain"
ANCHOR_DOMAIN_OUTPUT_KEY = "window_anchor_domain"


class WindowAnchorDomainError(ValueError):
    """A window choice outside its declared domain, or a domain that cannot be resolved."""


def _validate_bool_setting(name: str, value: Any, where: str) -> bool:
    if value is True or value is False:
        return value
    raise WindowAnchorDomainError(
        f"{where}: {name} must be true or false, got {value!r} "
        f"({type(value).__name__}); refused rather than read as off")


def validate_derived_anchor_domain(value: Any, *, where: str) -> bool:
    """The setting's one validator: True or False only. None, "true", 1 ... are refused,
    never read as off (same rule as the skip-delivery switch)."""
    return _validate_bool_setting("derived_anchor_domain", value, where)


def validate_anchor_holdout_guard(value: Any, derived_on: bool, *, where: str) -> bool:
    """The guard's one validator: a bool, and on only together with the derived domain
    (the bounded domain has no resolver to cap; on there it would be a silent no-op)."""
    on = _validate_bool_setting("anchor_holdout_guard", value, where)
    if on and not derived_on:
        raise WindowAnchorDomainError(
            f"{where}: anchor_holdout_guard is on but derived_anchor_domain is off; the guard "
            f"caps the derived domain only, so it is refused rather than ignored")
    return on


def anchor_domain_tag(derived_on: bool, guard_on: bool) -> Optional[str]:
    """The arm tag: None (bounded), "derived", or "derived_holdout_guard"."""
    if not derived_on:
        return None
    return ANCHOR_DOMAIN_MODE_DERIVED_GUARDED if guard_on else ANCHOR_DOMAIN_MODE_DERIVED


def _session_key(sessions: Sequence[str]) -> Tuple[str, ...]:
    return tuple(sorted(str(s) for s in sessions))


def load_window_anchor_bounds(config_path: str = "distributed_config.json") -> Dict[str, Any]:
    """`search_bounds.window_anchor` from the config, merged over the declared defaults.
    A malformed value fails closed. A missing file or key gives the defaults."""
    out = dict(DEFAULT_WINDOW_ANCHOR_BOUNDS)
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        return out
    raw = (cfg.get("search_bounds") or {}).get("window_anchor")
    if raw is None:
        return out
    if not isinstance(raw, dict):
        raise WindowAnchorDomainError(
            f"{config_path}: search_bounds.window_anchor must be an object, got {raw!r}")
    unknown = sorted(set(raw) - {"min", "max_cap"} - {k for k in raw if k.startswith("_")})
    if unknown:
        raise WindowAnchorDomainError(
            f"{config_path}: search_bounds.window_anchor has unknown key(s) {unknown}; "
            f"only 'min' and 'max_cap' are defined")
    out.update({k: raw[k] for k in ("min", "max_cap") if k in raw})
    mn, cap = out["min"], out["max_cap"]
    if isinstance(mn, bool) or not isinstance(mn, int) or mn < 0:
        raise WindowAnchorDomainError(
            f"{config_path}: search_bounds.window_anchor.min must be an int >= 0, got {mn!r}")
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 0):
        raise WindowAnchorDomainError(
            f"{config_path}: search_bounds.window_anchor.max_cap must be null or an int >= 0, "
            f"got {cap!r}")
    return out


@dataclass(frozen=True)
class AnchorDomain:
    """The resolved derived domain for one run. Built once, before the first trial."""
    dataset_path: str
    dataset_sha256: str
    n_filtered: Mapping[Tuple[str, ...], int]
    anchor_min: int
    anchor_max_cap: Optional[int]
    window_size_min: int
    window_size_max: int
    bounded_anchor_max: int                    # live search_bounds.offset.max (today 100)
    session_options: Tuple[Tuple[str, ...], ...] = field(default=())
    holdout_guard: bool = False
    holdout_split_point: Optional[int] = None  # combined-record index of the first holdout record
    n_train: Mapping[Tuple[str, ...], int] = field(default_factory=dict)
    # [STEP1 CHEAP SCREEN] None (default) = unchanged. Set ONLY by a screen objective
    # (utils/step1_screen.py ScreenPlan.anchor_max) via dataclasses.replace: the largest anchor
    # whose selection block still fits before selection_end_max and every confirmation block.
    anchor_max_hook: Optional[Callable[[int, Sequence[str]], int]] = field(
        default=None, compare=False, repr=False)

    # ---- the domain ---------------------------------------------------------------
    def _n(self, sessions: Sequence[str]) -> int:
        key = _session_key(sessions)
        if key not in self.n_filtered:
            raise WindowAnchorDomainError(
                f"sessions={list(key)} is not a resolved session option "
                f"{[list(k) for k in self.n_filtered]}; dataset sha256={self.dataset_sha256}")
        return int(self.n_filtered[key])

    def derived_max(self, window_size: int, sessions: Sequence[str]) -> int:
        return self._n(sessions) - int(window_size)

    def holdout_max(self, window_size: int, sessions: Sequence[str]) -> Optional[int]:
        """With the guard on: the largest anchor whose window [a, a + w) ends before the first
        holdout record of this session filter. None with the guard off."""
        if not self.holdout_guard:
            return None
        return int(self.n_train[_session_key(sessions)]) - int(window_size)

    def effective_range(self, window_size: int, sessions: Sequence[str]) -> Tuple[int, int]:
        """[min, effective_max]; raises if empty. effective_max never exceeds derived_max."""
        dmax = self.derived_max(window_size, sessions)
        emax = dmax if self.anchor_max_cap is None else min(int(self.anchor_max_cap), dmax)
        hmax = self.holdout_max(window_size, sessions)
        if hmax is not None:
            emax = min(emax, hmax)
        smax = None
        if self.anchor_max_hook is not None:                       # [STEP1 CHEAP SCREEN]
            smax = int(self.anchor_max_hook(int(window_size), sessions))
            emax = min(emax, smax)
        if self.anchor_min > emax:
            raise WindowAnchorDomainError(
                f"empty anchor domain [{self.anchor_min}, {emax}] for window_size={window_size}, "
                f"sessions={sorted(sessions)}, N_filtered={self._n(sessions)}, "
                f"max_cap={self.anchor_max_cap}, holdout_guard={self.holdout_guard}"
                f"{'' if hmax is None else f' (holdout_max={hmax})'}"
                f"{'' if smax is None else f' (screen selection-block max={smax})'}; "
                f"dataset sha256={self.dataset_sha256}")
        return int(self.anchor_min), int(emax)

    def control_era_range(self, window_size: int, sessions: Sequence[str]) -> Tuple[int, int]:
        """v1.1 §4.2 control era: [min, min(historical ANCHOR ceiling, effective_max)].
        The ceiling is the bounded-mode anchor max read live from search_bounds.offset.max.
        It is an ANCHOR bound; the record envelope (ceiling + window_size - 1) is never used."""
        lo, emax = self.effective_range(window_size, sessions)
        return lo, min(int(self.bounded_anchor_max), emax)

    # ---- the one route every proposer passes through --------------------------------
    def check_window_size(self, window_size: Any, sessions: Sequence[str]) -> int:
        if isinstance(window_size, bool) or not isinstance(window_size, int):
            raise WindowAnchorDomainError(
                f"window_size must be an int, got {window_size!r}; sessions={sorted(sessions)}, "
                f"dataset sha256={self.dataset_sha256}")
        if not self.window_size_min <= window_size <= self.window_size_max:
            raise WindowAnchorDomainError(
                f"window_size {window_size} is outside the recorded bounds "
                f"[{self.window_size_min}, {self.window_size_max}] (search_bounds.window_size); "
                f"sessions={sorted(sessions)}, dataset sha256={self.dataset_sha256}")
        return window_size

    def check_window_choice(self, window_size: Any, anchor: Any,
                            sessions: Sequence[str]) -> Tuple[int, int]:
        """Validate one proposed window, whoever proposed it (Optuna sample, operator pin,
        or a later learner). Returns (window_size, anchor). Never clamps."""
        ws = self.check_window_size(window_size, sessions)
        if isinstance(anchor, bool) or not isinstance(anchor, int):
            raise WindowAnchorDomainError(
                f"window_anchor must be an int, got {anchor!r}; sessions={sorted(sessions)}, "
                f"dataset sha256={self.dataset_sha256}")
        lo, hi = self.effective_range(ws, sessions)
        if not lo <= anchor <= hi:
            raise WindowAnchorDomainError(
                f"window_anchor {anchor} is outside the derived domain [{lo}, {hi}] "
                f"(window_size={ws}, sessions={sorted(sessions)}, "
                f"N_filtered={self._n(sessions)}, max_cap={self.anchor_max_cap}, "
                f"holdout_guard={self.holdout_guard}"
                f"{'' if not self.holdout_guard else f', N_train={self.n_train[_session_key(sessions)]}, holdout_split_point={self.holdout_split_point}'}); "
                f"dataset sha256={self.dataset_sha256}. Validated, never clamped.")
        return ws, anchor

    # ---- records ------------------------------------------------------------------
    def tag(self) -> str:
        return anchor_domain_tag(True, self.holdout_guard)

    def run_record(self) -> Dict[str, Any]:
        rec = {
            "mode": ANCHOR_DOMAIN_MODE_DERIVED,
            "tag": self.tag(),
            "min": self.anchor_min,
            "max_cap": self.anchor_max_cap,
            "dataset_sha256": self.dataset_sha256,
            "n_filtered": {"+".join(k): v for k, v in sorted(self.n_filtered.items())},
            "window_size_bounds": [self.window_size_min, self.window_size_max],
            "rule": "effective_max = derived_max if max_cap is null else min(max_cap, derived_max); "
                    "derived_max = N_filtered(sessions) - window_size",
            "holdout_guard": self.holdout_guard,
        }
        if self.holdout_guard:
            rec["holdout_split_point"] = self.holdout_split_point
            rec["n_train"] = {"+".join(k): v for k, v in sorted(self.n_train.items())}
            rec["holdout_rule"] = ("effective_max = min(effective_max, N_train(sessions) - "
                                   "window_size): the window ends before the first holdout record")
        return rec

    def trial_record(self, window_size: int, sessions: Sequence[str]) -> Dict[str, Any]:
        lo, hi = self.effective_range(window_size, sessions)
        rec = {"mode": ANCHOR_DOMAIN_MODE_DERIVED, "tag": self.tag(),
               "sessions": sorted(sessions),
               "n_filtered": self._n(sessions), "window_size": int(window_size),
               "anchor_range": [lo, hi], "dataset_sha256": self.dataset_sha256,
               "holdout_guard": self.holdout_guard}
        if self.holdout_guard:
            rec["n_train"] = int(self.n_train[_session_key(sessions)])
            rec["holdout_split_point"] = self.holdout_split_point
        if self.anchor_max_hook is not None:                       # [STEP1 CHEAP SCREEN]
            rec["screen_anchor_max"] = int(self.anchor_max_hook(int(window_size), sessions))
        return rec


def resolve_anchor_domain(dataset_path: str,
                          session_options: Sequence[Sequence[str]],
                          *,
                          window_size_min: int,
                          window_size_max: int,
                          bounded_anchor_max: int,
                          anchor_bounds: Optional[Mapping[str, Any]] = None,
                          expected_sha256: Optional[str] = None,
                          holdout_guard: bool = False,
                          holdout_split_point_fn=None) -> AnchorDomain:
    """Resolve the derived domain once, from the dataset file the run sieves.

    `expected_sha256`: when the run froze a dataset (miner.dataset_authority), its digest;
    a different file is refused. N_filtered is counted per session option and then proved
    against miner.range_miner_worker.load_residue_window, the session filter's authority."""
    from miner.range_miner_worker import (load_residue_window, _sha256_file,
                                          ResidueResolutionError)
    bounds = dict(DEFAULT_WINDOW_ANCHOR_BOUNDS if anchor_bounds is None else anchor_bounds)
    sha = _sha256_file(dataset_path)
    if expected_sha256 is not None and sha != expected_sha256:
        raise WindowAnchorDomainError(
            f"dataset {dataset_path!r} has sha256 {sha}, but the run froze {expected_sha256}; "
            f"the anchor domain is resolved only against the frozen dataset")
    with open(dataset_path, "r") as f:
        data = json.load(f)
    split_point: Optional[int] = None
    if holdout_guard:
        if holdout_split_point_fn is None:
            raise WindowAnchorDomainError(
                "anchor_holdout_guard is on but no split-point function was supplied; the guard "
                "reads the split from Step 1's own helper and never assumes one")
        split_point = int(holdout_split_point_fn(len(data)))
        if not 0 <= split_point <= len(data):
            raise WindowAnchorDomainError(
                f"holdout split point {split_point} is outside [0, {len(data)}]; "
                f"dataset sha256={sha}")
    n_train: Dict[Tuple[str, ...], int] = {}
    n_filtered: Dict[Tuple[str, ...], int] = {}
    for opt in session_options:
        key = _session_key(opt)
        n = sum(1 for e in data if e.get("session") in list(opt))
        # Prove the count against the authority: the last index loads, one past it is refused.
        try:
            if n >= 1:
                load_residue_window(dataset_path, 1, list(opt), n - 1)
        except ResidueResolutionError as exc:
            raise WindowAnchorDomainError(
                f"N_filtered={n} for sessions={list(key)} disagrees with load_residue_window "
                f"(anchor {n - 1} refused: {exc}); dataset sha256={sha}") from exc
        try:
            load_residue_window(dataset_path, 1, list(opt), n)
        except ResidueResolutionError:
            pass
        else:
            raise WindowAnchorDomainError(
                f"N_filtered={n} for sessions={list(key)} disagrees with load_residue_window "
                f"(anchor {n} loaded); dataset sha256={sha}")
        n_filtered[key] = n
        if holdout_guard:
            n_train[key] = sum(1 for e in data[:split_point] if e.get("session") in list(opt))
    return AnchorDomain(
        dataset_path=dataset_path, dataset_sha256=sha, n_filtered=n_filtered,
        anchor_min=int(bounds["min"]),
        anchor_max_cap=None if bounds.get("max_cap") is None else int(bounds["max_cap"]),
        window_size_min=int(window_size_min), window_size_max=int(window_size_max),
        bounded_anchor_max=int(bounded_anchor_max),
        session_options=tuple(_session_key(o) for o in session_options),
        holdout_guard=bool(holdout_guard), holdout_split_point=split_point, n_train=n_train)


# ---- study and search arm separation (the skip-delivery precedent) -------------------
def anchor_domain_study_check(study, run_tag: Optional[str], trial_number: int) -> None:
    """At the first trial of a run, before its sieve. `run_tag` is anchor_domain_tag(...):
    None for a bounded run. A derived run tags a fresh study and refuses any study with
    earlier trials of another arm (bounded, or the other guard value). A bounded run refuses
    a tagged study and writes nothing. Pre-existing studies carry no tag, so the bounded
    path is unchanged for them."""
    tag = (study.user_attrs or {}).get(ANCHOR_DOMAIN_STUDY_ATTR)
    if run_tag is not None:
        if tag == run_tag:
            return
        earlier = [t for t in study.trials if t.number < int(trial_number)]
        if tag is None and not earlier:
            study.set_user_attr(ANCHOR_DOMAIN_STUDY_ATTR, run_tag)
            return
        raise WindowAnchorDomainError(
            f"--derived-anchor-domain ({run_tag}): study {study.study_name!r} holds "
            f"{len(earlier)} trial(s) of another arm ({ANCHOR_DOMAIN_STUDY_ATTR}={tag!r}); "
            f"refused before the sieve")
    if tag is not None:
        raise WindowAnchorDomainError(
            f"study {study.study_name!r} belongs to a derived-anchor-domain run "
            f"({ANCHOR_DOMAIN_STUDY_ATTR}={tag!r}); a bounded run may not add trials to it")


def anchor_domain_search_check(search_space: Any, run_tag: Optional[str], search_id: str) -> None:
    """A Step 1 search opened in one arm refuses runs of another."""
    recorded = search_space.get(ANCHOR_DOMAIN_SEARCH_KEY) if isinstance(search_space, dict) else None
    if recorded != run_tag:
        raise WindowAnchorDomainError(
            f"search {search_id} was opened with {ANCHOR_DOMAIN_SEARCH_KEY}={recorded!r} but this "
            f"run is {run_tag!r} (None = bounded); open a separate search for this arm")
