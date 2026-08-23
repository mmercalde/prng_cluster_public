#!/usr/bin/env python3
"""
canonical_records.py — S172 Phase-5 Deliverable D3.25: the ONE shared canonical
record builder, plus the versioned `step1_trial_populations_v2` producer
contract validated at BOTH boundaries.

Spec: docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3_25.md (REV3), rebased onto
HEAD c207e3a.

SCOPE — exactly one transformation, in the opposite direction from D3:

    four directional maps + explicit trial context  ->  canonical 24-field records

Sibling module: `utils/canonical_arrays.py` (D3) maps 24-field records -> the
typed 22-array bundle. D3.25 produces what D3 consumes; the two are deliberately
NOT coupled in the production path (REV3 §4a) — D3's validators are a *gate-side*
conformance check on this module's output, never a substitute for producing
correct records here.

DEPENDENCY DIRECTION (REV3 §3, correction [C2]):

    generic utilities  <-  { miner, PWC, ZMQ, adapter }

so this module imports NOTHING from `miner/`, takes NO dependency on the
`WindowConfig` class, and performs NO generic attribute lookup on a caller
object. D1 receives validated manifest metadata; PWC/ZMQ receive a
`WindowConfig`; this utility sits BELOW both and is handed explicit values.

THE TWO PUBLIC ENTRY POINTS:

  * `build_mode_records()` — the semantics-preserving extraction of
    `miner/range_miner_npz_writer._mode_records` (:432 at HEAD c207e3a). Moved
    UNCHANGED: same intersection, same `max(..., 1)` denominators, same
    deliberate `bidirectional_count` / `intersection_count` duplication, same
    ascending seed order, same `prng_type is None -> (both, [])` sentinel for a
    mode that did not run, and the same direct placement of the context's
    `sessions` object into every record ([C5] — D1's accepted shared-reference
    behavior is NOT silently changed here).

  * `normalize_trial_populations()` — the PWC/ZMQ wrapper. It owns the
    canonical field forms D1 already satisfies but the legacy adapter did not:
    integer `skip_range`, canonical `list` `sessions` (with its own defensive
    copy, so a caller mutating its original list cannot reach an
    already-produced record), and per-mode `prng_type` derivation.

WHAT THIS MODULE DELIBERATELY DOES NOT DO:
  * it selects no winner, merges no prior, writes no NPZ and orders no final
    rows — D3.25 orders candidate rows only at ingress (trial-major,
    mode-minor), and D3.5's explicit L2 key remains authoritative;
  * it never reconstructs a directional map from a record list (REV3 §0.3,
    binding): `forward_records_hybrid` / `reverse_records_hybrid` are telemetry
    whose provenance differs between PWC (raw survivor sequence) and ZMQ (map
    keys), so a repeated raw seed makes list and map inequivalent. The four
    explicit maps are the only authority;
  * it never silently repairs a producer/consumer disagreement. A returned
    bidirectional set that disagrees with its own map intersection is producer
    state corruption and fails closed at BOTH boundaries — never resolved by
    preferring the set, and never by preferring a recomputed intersection.
"""
from __future__ import annotations

import math
import numbers
from typing import Any, Dict, List, Mapping, Optional, Tuple

__all__ = [
    "CANONICAL_RECORD_FIELDS",
    "TRIAL_POPULATIONS_SCHEMA_VERSION",
    "TRIAL_POPULATION_MAP_FIELDS",
    "TRIAL_POPULATION_SET_FIELDS",
    "TRIAL_POPULATION_FIELDS",
    "CanonicalRecordContractError",
    "TrialPopulationContractError",
    "build_mode_records",
    "build_trial_populations",
    "canonical_sessions",
    "normalize_trial_populations",
    "validate_trial_populations",
]


# ---------------------------------------------------------------------------
# Errors. Both derive from ValueError for the same reason D3's do: a caller who
# only knows the broad contract ("bad input raises ValueError") keeps working,
# while a caller who wants to distinguish a record-level rejection from a
# producer-contract violation can.
# ---------------------------------------------------------------------------
class CanonicalRecordContractError(ValueError):
    """A canonical record could not be built from the values supplied."""


class TrialPopulationContractError(ValueError):
    """A backend result violated the `step1_trial_populations_v2` contract."""


# ---------------------------------------------------------------------------
# §6 — the frozen 24-field canonical record, RELOCATED here from
# `miner/range_miner_npz_writer.py:150` ([C3]) so it sits beside the builder
# that produces it. The writer re-exports it, so D1.1's G9 harness — which
# hand-transcribes an independent oracle and compares — is unaffected.
#
# Order and membership reproduce the LIVE Step-1 insertion order exactly:
# seed/rates/score followed by `metadata_base`
# (window_optimizer_integration_final.py:683-694 + :652-676 for constant;
# identically :785-796 + :756-780 for the hybrid/variable block).
#
# `threshold_used` is deliberately NOT a 25th field: it is manifest identity /
# validation metadata (D1 §5.1), not a record field.
#
# `utils/canonical_arrays.py` holds its OWN copy on purpose (its :130-142 note):
# D3 is the executable definition of a valid record and must not derive that
# definition from the module it validates. The duplication is deliberate and
# gate-checked, not an oversight.
# ---------------------------------------------------------------------------
CANONICAL_RECORD_FIELDS: Tuple[str, ...] = (
    "seed", "forward_match_rate", "reverse_match_rate", "score",
    "window_size", "offset", "skip_min", "skip_max", "skip_range", "sessions",
    "trial_number", "prng_base", "skip_mode", "prng_type",
    "forward_count", "reverse_count", "bidirectional_count",
    "intersection_count", "intersection_ratio",
    "forward_only_count", "reverse_only_count",
    "survivor_overlap_ratio", "bidirectional_selectivity", "intersection_weight",
)


# ---------------------------------------------------------------------------
# §2 — the versioned producer contract.
#
# The shape NEVER varies. A constant-only trial, a both-mode trial, a
# forward-zero pruned return and every other supported pruned return all carry
# the complete shape with all four maps present (empty where the mode did not
# run). A MISSING field is NOT interpreted as an empty field — that is the
# whole point of the version stamp, and it is why both PWC's pruned early
# return (:1621-1629) and ZMQ's (:1091-1099) had to be rewritten rather than
# left to the adapter's old `.get(..., {})`.
# ---------------------------------------------------------------------------
TRIAL_POPULATIONS_SCHEMA_VERSION = "step1_trial_populations_v2"

TRIAL_POPULATION_MAP_FIELDS: Tuple[str, ...] = (
    "forward_map_constant", "reverse_map_constant",
    "forward_map_variable", "reverse_map_variable",
)

TRIAL_POPULATION_SET_FIELDS: Tuple[str, ...] = (
    "bidirectional_constant", "bidirectional_variable",
)

TRIAL_POPULATION_FIELDS: Tuple[str, ...] = (
    ("schema_version",)
    + TRIAL_POPULATION_MAP_FIELDS
    + TRIAL_POPULATION_SET_FIELDS
    + ("pruned", "reason")
)

# mode -> (forward map field, reverse map field, bidirectional set field)
_MODE_FIELDS: Dict[str, Tuple[str, str, str]] = {
    "constant": ("forward_map_constant", "reverse_map_constant",
                 "bidirectional_constant"),
    "variable": ("forward_map_variable", "reverse_map_variable",
                 "bidirectional_variable"),
}

# The two canonical skip modes, in the frozen trial-major/mode-minor order.
CANONICAL_SKIP_MODES: Tuple[str, ...] = ("constant", "variable")


def _is_int(value: Any) -> bool:
    """Integer test with bool EXCLUDED — True == 1, so a bare isinstance check
    would accept a Boolean anywhere an integer is required (the same exclusion
    D1 applies at `range_miner_npz_writer._is_int`)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: Any) -> bool:
    """Finite real-number test, bool excluded for the same reason."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    return math.isfinite(float(value))


# ---------------------------------------------------------------------------
# §5.5/§6 — derived fields + canonical records for ONE mode.
#
# EXTRACTED VERBATIM from `miner/range_miner_npz_writer._mode_records` (:432 at
# HEAD c207e3a). Semantics-preserving is a hard requirement, and D1.1 18/18 +
# D2 7/7 staying green is its proof — not a re-reading of this docstring. Do
# not "improve" anything below: not the `max(..., 1)` guards, not the
# `bidirectional_count`/`intersection_count` duplication, not the ordering, and
# not the direct placement of `ctx["sessions"]` into every record ([C5]).
# ---------------------------------------------------------------------------
def build_mode_records(
    forward_map: Mapping[int, float],
    reverse_map: Mapping[int, float],
    context: Mapping[str, Any],
    skip_mode: str,
    prng_type: Optional[str],
) -> Tuple[set, List[Dict[str, Any]]]:
    """Intersection + the frozen 24-field canonical records for one skip mode.

    Semantics-preserving extraction of `_mode_records`. Every derived field is
    frozen from the live constant block
    (window_optimizer_integration_final.py:652-694) and variable block
    (:756-796), including the `max(..., 1)` denominators and the deliberate
    duplication of `bidirectional_count` / `intersection_count`.

    `prng_type is None` means the mode did not run (D1 §5.4): the intersection
    is still returned, and the record list is empty.
    """
    fwd_map, rev_map, ctx = forward_map, reverse_map, context
    fwd_set, rev_set = set(fwd_map), set(rev_map)
    both = fwd_set & rev_set
    if prng_type is None:                       # the mode did not run (§5.4)
        return both, []
    union = len(fwd_set | rev_set)
    metadata_base = {
        "window_size":               ctx["window_size"],
        # [WINDOW-ANCHOR BRIEF I — TB ruling, production-shape failure at 48a8705]
        # Canonical array 4 `offset` is a LEGACY WIRE NAME with exactly ONE
        # post-F-4 meaning: it IS the window anchor. It is NEVER the generator
        # phase, at any phase value — not merely while v1 pins the phase to 0.
        # Generator phase remains independently represented in versioned
        # generation metadata and never enters this array.
        # The name is frozen by the 22-array contract (index 4) and does not
        # change; only the source of its value is corrected here.
        "offset":                    ctx["window_anchor"],
        "skip_min":                  ctx["skip_min"],
        "skip_max":                  ctx["skip_max"],
        "skip_range":                ctx["skip_max"] - ctx["skip_min"],
        "sessions":                  ctx["sessions"],
        "trial_number":              ctx["trial_number"],
        "prng_base":                 ctx["prng_base"],
        "skip_mode":                 skip_mode,
        "prng_type":                 prng_type,
        "forward_count":             len(fwd_map),
        "reverse_count":             len(rev_map),
        "bidirectional_count":       len(both),
        "intersection_count":        len(both),
        "intersection_ratio":        len(both) / max(union, 1),
        "forward_only_count":        len(fwd_set - rev_set),
        "reverse_only_count":        len(rev_set - fwd_set),
        "survivor_overlap_ratio":    len(both) / max(len(fwd_set), 1),
        "bidirectional_selectivity": len(fwd_set) / max(len(rev_set), 1),
        "intersection_weight":       len(both) / max(len(fwd_set) + len(rev_set), 1),
    }
    records = []
    for seed in sorted(both):                   # ascending seed order (§6)
        fwd_rate, rev_rate = fwd_map[seed], rev_map[seed]
        record = {
            "seed":               seed,
            "forward_match_rate": fwd_rate,
            "reverse_match_rate": rev_rate,
            "score":              (fwd_rate + rev_rate) / 2.0,
        }
        record.update(metadata_base)
        # The frozen 24 keys, in the frozen order — gate-enforced (G9). Rebuilt
        # explicitly rather than trusted from insertion order.
        records.append({k: record[k] for k in CANONICAL_RECORD_FIELDS})
    return both, records


# ---------------------------------------------------------------------------
# §3.4 — the canonical `sessions` form.
#
# The rules are asymmetric ON PURPOSE:
#   list / tuple  -> defensive list copy   (the tuple is a shape mismatch a
#                                           producer can be forgiven for; D3
#                                           rejects a tuple outright)
#   None          -> []                    (an explicitly absent session set)
#   absent        -> FAIL CLOSED           (the caller never looked)
#   scalar "all"  -> FAIL CLOSED           (do NOT convert to ["all"] — that is
#                                           the exact legacy fabrication
#                                           `getattr(config, 'sessions', 'all')`
#                                           produced, and inventing a session
#                                           name is worse than refusing)
# ---------------------------------------------------------------------------
def canonical_sessions(sessions: Any) -> List[str]:
    """Return the canonical `list[str]` form of a `sessions` value.

    The copy is defensive: a caller that mutates its own list afterwards must
    not be able to reach a record this function already produced (G6).
    """
    if sessions is None:
        return []
    if isinstance(sessions, str):
        raise CanonicalRecordContractError(
            f"sessions must be a list of session names, got the scalar string "
            f"{sessions!r}. A scalar is NOT converted to [{sessions!r}] — the "
            f"legacy `getattr(config, 'sessions', 'all')` fallback fabricated a "
            f"session name and D3.25 fails closed instead."
        )
    if not isinstance(sessions, (list, tuple)):
        raise CanonicalRecordContractError(
            f"sessions must be a list or tuple of session names, got "
            f"{type(sessions).__name__}."
        )
    out = list(sessions)
    for pos, item in enumerate(out):
        if not isinstance(item, str):
            raise CanonicalRecordContractError(
                f"sessions[{pos}] must be str, got {type(item).__name__}."
            )
    return out


def _require_int(value: Any, name: str) -> int:
    if not _is_int(value):
        raise CanonicalRecordContractError(
            f"{name} must be an integer (bool excluded), got {value!r} "
            f"({type(value).__name__}). D3.25 takes explicit values only — "
            f"there is no coercion and no default."
        )
    return int(value)


# ---------------------------------------------------------------------------
# §3 — the PWC/ZMQ wrapper.
# ---------------------------------------------------------------------------
def normalize_trial_populations(
    forward_map_constant: Mapping[int, float],
    reverse_map_constant: Mapping[int, float],
    forward_map_variable: Mapping[int, float],
    reverse_map_variable: Mapping[int, float],
    *,
    window_size: int,
    offset: int,
    skip_min: int,
    skip_max: int,
    sessions: Any,
    trial_number: int,
    prng_base: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build `(constant_records, variable_records)` from the four v2 maps.

    Explicit values only — no config object, no `getattr`, no default. Each
    mode's records are derived ONLY from that mode's pair of maps (§3.3), so a
    variable record can never carry a constant-mode rate or a constant-mode
    aggregate.

    Canonical forms this wrapper owns (§3.4):
      * `skip_range` = int(skip_max) - int(skip_min) — the legacy
        `f"{skip_min}-{skip_max}"` string form is prohibited;
      * `sessions` normalized through `canonical_sessions`, with its own
        defensive copy so the caller's list is not shared into the records;
      * `prng_type` derived per mode: constant -> prng_base,
        variable -> prng_base + "_hybrid".
    """
    if not isinstance(prng_base, str) or not prng_base:
        raise CanonicalRecordContractError(
            f"prng_base must be a nonempty str, got {prng_base!r}."
        )
    if prng_base.endswith("_hybrid") or prng_base.endswith("_reverse"):
        raise CanonicalRecordContractError(
            f"prng_base {prng_base!r} is a derived identity, not a base family. "
            f"A record's prng_type is a MODE label derived FROM prng_base; "
            f"passing a derived identity would produce e.g. "
            f"{prng_base + '_hybrid'!r}."
        )

    skip_min_i = _require_int(skip_min, "skip_min")
    skip_max_i = _require_int(skip_max, "skip_max")

    # ONE defensive copy per trial, shared by that trial's records exactly as
    # D1 shares its already-validated context object ([C5]). The caller's
    # original list is never referenced again.
    ctx: Dict[str, Any] = {
        "window_size":  _require_int(window_size, "window_size"),
        # [WINDOW-ANCHOR BRIEF I — TB ruling, production-shape failure at 48a8705]
        # The CONTEXT key is `window_anchor`; the CALLER'S KEYWORD stays `offset`
        # and the emitted RECORD FIELD stays `offset` (frozen canonical array 4).
        # Beta: array 4 `offset` is a legacy WIRE NAME whose one post-F-4 meaning
        # IS the window anchor — never the generator phase, at any phase value.
        # This is the PWC/ZMQ wrapper's own context; `build_mode_records` reads
        # `window_anchor` from it, exactly as the miner assembly path does.
        "window_anchor": _require_int(offset, "offset"),
        "skip_min":     skip_min_i,
        "skip_max":     skip_max_i,
        "sessions":     canonical_sessions(sessions),
        "trial_number": _require_int(trial_number, "trial_number"),
        "prng_base":    prng_base,
    }

    # `build_mode_records` computes skip_range as ctx["skip_max"] - ctx["skip_min"];
    # both are ints above, so the result is the required int difference.
    _, constant_records = build_mode_records(
        forward_map_constant, reverse_map_constant, ctx, "constant", prng_base)
    _, variable_records = build_mode_records(
        forward_map_variable, reverse_map_variable, ctx, "variable",
        prng_base + "_hybrid")
    return constant_records, variable_records


# ---------------------------------------------------------------------------
# §2 — the contract itself, validated at BOTH boundaries ([C4]).
#
# `validate_trial_populations` is called
#   * by PWC and ZMQ immediately BEFORE returning  (producer egress), and
#   * by `_build_test_result_from_pw` on arrival   (adapter ingress),
# and the adapter calls it BEFORE it touches the accumulator, so a malformed or
# test-mutated result fails before even one candidate is appended (G4).
#
# Two independent boundaries is what makes the check meaningful: deleting
# either one must be caught (G11).
# ---------------------------------------------------------------------------
def validate_trial_populations(result: Mapping[str, Any], *, origin: str) -> None:
    """Validate a backend result against `step1_trial_populations_v2`.

    Checks, all fail-closed:
      * `schema_version` present and EXACTLY the expected string;
      * all four maps present, each a dict of int seed -> finite real rate;
      * both bidirectional sets present, each a set of ints;
      * `bidirectional_M == set(forward_map_M) & set(reverse_map_M)` for both
        modes;
      * the pruned shape: `pruned` is a bool, and `reason` is a nonempty str
        when pruned and None when not.

    A missing field is a contract violation, NEVER an empty field: presence is
    tested before type, so `.get(name, {})` can never be simulated here.

    Neither boundary may repair a set/map disagreement by preferring the
    returned set or a recomputed intersection — a mismatch is producer-state
    corruption.
    """
    if not isinstance(result, Mapping):
        raise TrialPopulationContractError(
            f"{origin}: expected a Mapping carrying "
            f"{TRIAL_POPULATIONS_SCHEMA_VERSION}, got {type(result).__name__}."
        )

    if "schema_version" not in result:
        raise TrialPopulationContractError(
            f"{origin}: missing 'schema_version'. The v2 contract is versioned "
            f"precisely so an unstamped legacy result cannot be mistaken for a "
            f"complete one; there is no default."
        )
    version = result["schema_version"]
    if version != TRIAL_POPULATIONS_SCHEMA_VERSION:
        raise TrialPopulationContractError(
            f"{origin}: schema_version {version!r} != "
            f"{TRIAL_POPULATIONS_SCHEMA_VERSION!r}."
        )

    for name in TRIAL_POPULATION_MAP_FIELDS:
        if name not in result:
            raise TrialPopulationContractError(
                f"{origin}: missing required map {name!r}. A missing map is NOT "
                f"an empty map — every trial (constant-only, both-mode, and "
                f"every pruned return) carries all four, empty where the mode "
                f"did not run."
            )
        value = result[name]
        if not isinstance(value, dict):
            raise TrialPopulationContractError(
                f"{origin}: {name!r} must be a dict of seed -> match_rate, got "
                f"{type(value).__name__}."
            )
        for seed, rate in value.items():
            if not _is_int(seed):
                raise TrialPopulationContractError(
                    f"{origin}: {name!r} key {seed!r} is not an integer seed "
                    f"(bool excluded)."
                )
            if not _is_real(rate):
                raise TrialPopulationContractError(
                    f"{origin}: {name!r}[{seed}] match_rate {rate!r} is not a "
                    f"finite real number (bool excluded)."
                )

    for name in TRIAL_POPULATION_SET_FIELDS:
        if name not in result:
            raise TrialPopulationContractError(
                f"{origin}: missing required population {name!r}."
            )
        value = result[name]
        if not isinstance(value, (set, frozenset)):
            raise TrialPopulationContractError(
                f"{origin}: {name!r} must be a set of seeds, got "
                f"{type(value).__name__}."
            )
        for seed in value:
            if not _is_int(seed):
                raise TrialPopulationContractError(
                    f"{origin}: {name!r} member {seed!r} is not an integer seed "
                    f"(bool excluded)."
                )

    for mode, (fwd_name, rev_name, set_name) in _MODE_FIELDS.items():
        expected = set(result[fwd_name]) & set(result[rev_name])
        actual = set(result[set_name])
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise TrialPopulationContractError(
                f"{origin}: {set_name!r} disagrees with "
                f"set({fwd_name}) & set({rev_name}) for the {mode} mode — "
                f"absent from the set: {missing}; present but not in the "
                f"intersection: {extra}. This is producer-state corruption and "
                f"fails closed; it is NEVER repaired by preferring the returned "
                f"set or a recomputed intersection."
            )

    if "pruned" not in result:
        raise TrialPopulationContractError(f"{origin}: missing 'pruned'.")
    pruned = result["pruned"]
    if not isinstance(pruned, bool):
        raise TrialPopulationContractError(
            f"{origin}: 'pruned' must be a bool, got {type(pruned).__name__}."
        )
    if "reason" not in result:
        raise TrialPopulationContractError(
            f"{origin}: missing 'reason'. A non-pruned trial carries an "
            f"explicit None, not an absent key."
        )
    reason = result["reason"]
    if pruned:
        if not isinstance(reason, str) or not reason:
            raise TrialPopulationContractError(
                f"{origin}: a pruned result must name its reason as a nonempty "
                f"str, got {reason!r}."
            )
    elif reason is not None:
        raise TrialPopulationContractError(
            f"{origin}: a non-pruned result must carry reason=None, got "
            f"{reason!r}."
        )


def build_trial_populations(
    *,
    forward_map_constant: Mapping[int, float],
    reverse_map_constant: Mapping[int, float],
    forward_map_variable: Mapping[int, float],
    reverse_map_variable: Mapping[int, float],
    bidirectional_constant: Any,
    bidirectional_variable: Any,
    pruned: bool,
    reason: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
    origin: str = "producer-egress",
) -> Dict[str, Any]:
    """Assemble + egress-validate the v2 block a backend returns.

    PWC and ZMQ both build their return dict through here, so the invariant
    `bidirectional_M == set(forward_map_M) & set(reverse_map_M)` is asserted at
    producer egress for EVERY return path — the full return, PWC's pruned
    early return (:1621-1629) and ZMQ's (:1091-1099) alike.

    `extra` carries each backend's legacy/telemetry keys (`forward_map`,
    `reverse_map`, `bidirectional_count`, the four record lists). They may
    remain temporarily; the v2 adapter must never read them (G7), and `extra`
    is applied FIRST so it can never shadow a v2 field.
    """
    block: Dict[str, Any] = dict(extra or {})
    block.update({
        "schema_version":         TRIAL_POPULATIONS_SCHEMA_VERSION,
        "forward_map_constant":   dict(forward_map_constant),
        "reverse_map_constant":   dict(reverse_map_constant),
        "forward_map_variable":   dict(forward_map_variable),
        "reverse_map_variable":   dict(reverse_map_variable),
        "bidirectional_constant": set(bidirectional_constant),
        "bidirectional_variable": set(bidirectional_variable),
        "pruned":                 bool(pruned),
        "reason":                 reason,
    })
    validate_trial_populations(block, origin=origin)
    return block
