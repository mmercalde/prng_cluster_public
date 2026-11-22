"""utils/step1_search_ledger.py — the Step 1 SEARCH record (stage 1).

Design: docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md §2.1-§2.7.
Brief:  docs/CLAUDE_CODE_INSTRUCTIONS_STEP1_SEARCH_RECORD_STAGE1_20261114.md (S1, S8, S9).

TWO RECORDS, TWO MEANINGS (v1.1 §2.1)
  * HISTORY — the coverage ledger's table (utils/seed_coverage_ledger.py, NOT EDITED): which seed
    intervals were swept and published for a family and mode set.
  * SEARCH  — this module: one named piece of Step 1 work, what it ran on, and how far it
    has got. It decides where WATCHER's next Step 1 chunk starts.

THIS MODULE IS NOT A SECOND CERTIFICATION DOOR. It never writes the history table. A
chunk is credited to a search only through `record_chunk_link`, which requires the
canonical publication witness (`RunArtifactResult`) AND the history row the one existing
door already wrote from that witness, and derives every interval/family/mode/generation
field from those two objects. The caller cannot assert them.

APPEND-ONLY. Every table carries BEFORE UPDATE and BEFORE DELETE triggers that ABORT, and
`recursive_triggers` is ON per connection so `INSERT OR REPLACE` cannot delete either. The
only write statements in this module are plain INSERTs.

ONE OPEN SEARCH PER (family, mode set), enforced inside a single `BEGIN IMMEDIATE`
transaction: the check and the insert cannot interleave with another opener.

THE CURSOR is `first_uncovered_seed` (utils/seed_coverage_ledger.py:551-605), imported and
applied to THIS search's credited (linked) chunks inside its declared range. It is
assigned even when it is 0 (v1.1 §2.4).

The yardstick is prediction-pool hit rate against k/1000 on unseen draws. Nothing here is a
statement about predictive value; this module records which work was run.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from utils.seed_coverage_ledger import (
    SEED_DOMAIN_EXCLUSIVE_MAX,
    SEED_DOMAIN_START,
    CoverageLedger,
    CursorResult,
    canonical_coverage_identity,
    first_uncovered_seed,
)

SEARCH_SCHEMA_VERSION = 1
SEARCH_ID_ENCODING_VERSION = 1

T_META = "step1_search_schema_meta"
T_SEARCH = "step1_search"
T_EVENT = "step1_search_event"
T_CHUNK = "step1_chunk_open"
T_LINK = "step1_chunk_link"
T_UNCREDITED = "step1_uncredited_attempt"
#: [Beta X-STEP1-SEARCH-CODE R1 Q5] The durable record written AFTER the history row and
#: BEFORE the link: what a fresh process needs to rebuild a link whose write failed after
#: publication, without a new trial and without a new generation.
T_PENDING = "step1_link_pending"
ALL_TABLES = (T_META, T_SEARCH, T_EVENT, T_CHUNK, T_LINK, T_UNCREDITED, T_PENDING)

EV_OPENED, EV_CLOSED, EV_SUPERSEDED, EV_COMPLETED = "opened", "closed", "superseded", "completed"
TERMINAL_EVENTS = (EV_CLOSED, EV_SUPERSEDED, EV_COMPLETED)

OPEN_REASONS = ("operator", "regime_shift", "soak", "history_import")

# The three durable WATCHER results (v1.1 §2.4).
SEL_NO_OPEN_SEARCH = "no_open_search"
SEL_SEARCH_COMPLETE = "search_complete"
SEL_GAP = "gap"

#: The session-mapping rule as it is executed today. Recorded with every search so the
#: index-to-label mapping is never the only thing stored (v1.1 §2.2 `session_filter_rule`).
SESSION_FILTER_RULE_V1 = (
    "v1: session_idx indexes session_options (label lists, in order, from "
    "window_optimizer.SearchBounds.session_options); a label list maps to the sieve "
    "session filter by set equality: {midday,evening}->both, {midday}->midday, "
    "otherwise evening (window_optimizer_integration_final.run_bidirectional_test)"
)

#: The seven operator pin keys (mirrors agents/watcher_agent.STEP1_EXPLICIT_PIN_KEYS; the
#: suite asserts the two are equal so they cannot drift).
PIN_KEYS = (
    "warm_start_fwd_thresh", "warm_start_offset", "warm_start_rev_thresh",
    "warm_start_session_idx", "warm_start_skip_max", "warm_start_skip_min",
    "warm_start_window",
)

#: The configuration fields compared field by field before every dispatch (S4).
CONFIG_FIELDS = (
    "prng_base", "required_modes", "session_options", "session_filter_rule",
    "search_space", "search_space_digest", "sampler", "pruning",
    "trials_per_chunk", "chunk_seeds",
)


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
class Step1SearchError(RuntimeError):
    """Base class. Every refusal names what was presented and what was recorded."""


class Step1SearchSchemaError(Step1SearchError):
    pass


class Step1SearchRefused(Step1SearchError):
    pass


class Step1SearchLookupError(Step1SearchError):
    pass


class Step1StudyOwnershipError(Step1SearchRefused):
    pass


class Step1LinkError(Step1SearchError):
    pass


# ---------------------------------------------------------------------------
# canonical encoding
# ---------------------------------------------------------------------------
def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def selector_key(prng_base: str, required_modes: Sequence[str]) -> str:
    return f"{prng_base}|{','.join(required_modes)}"


def _modes_tuple(modes: Iterable[str]) -> Tuple[str, ...]:
    order = ("constant", "variable")
    vals = set(modes)
    return tuple(m for m in order if m in vals)


# ---------------------------------------------------------------------------
# the presented configuration — ONE function for opening and for every dispatch
# ---------------------------------------------------------------------------
def skip_delivery_can_sample(pinned: bool, trials: Any, resume_study: bool) -> bool:
    """[SKIP-DELIVERY SWITCH] THE one rule, used by search open and by the Step 1 child
    (Beta X-MINER-SKIP-SWITCH-2 R1 Q4), for whether a switched run can SAMPLE skip bounds.
    Only a FRESH run with all seven pins and a one-trial budget cannot: its single trial is
    the enqueued pin (docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:121). A resumed
    study enqueues no pin (window_optimizer_bayesian.py, `if not _resume`), so it samples."""
    return not (pinned and trials is not None and int(trials) == 1 and not resume_study)


def validate_skip_sampling_domains(min_skip_min, max_skip_min, min_skip_max, max_skip_max,
                                   where: str) -> None:
    """[SKIP-DELIVERY SWITCH] The full sampling domains, through the one validator
    (miner.range_miner_coordinator.validate_skip_delivery): each range is plain ints in
    0..255 with lo <= hi, and skip_min.max <= skip_max.max. Optuna draws skip_max from
    [max(skip_min, skip_max.min), skip_max.max] (window_optimizer_bayesian.py), so the
    ordering is exactly what keeps every draw non-empty and every pair legal. Raises
    MinerMetadataError."""
    from miner.range_miner_coordinator import validate_skip_delivery
    validate_skip_delivery(True, min_skip_min, max_skip_min, where=f"{where} (skip_min range)")
    validate_skip_delivery(True, min_skip_max, max_skip_max, where=f"{where} (skip_max range)")
    validate_skip_delivery(True, max_skip_min, max_skip_max,
                           where=f"{where} (skip_min.max <= skip_max.max)")


def presented_configuration(final_params: Mapping[str, Any],
                            pin_bundle: Optional[Mapping[str, Any]],
                            bounds: Mapping[str, Any],
                            session_options: Sequence[Sequence[str]]) -> Dict[str, Any]:
    """The configuration a Step 1 run presents, in effective terms.

    `final_params` — WATCHER's step-scoped merged params (manifest defaults + operator).
    `pin_bundle`   — the invocation-local operator pin (all seven) or None. It is READ, it
                     is never produced here, and nothing here writes it anywhere.
    `bounds`       — `window_optimizer.load_search_bounds_from_config(...)` output.
    `session_options` — `SearchBounds().session_options` (label lists, in index order).

    The same function computes what is recorded at opening and what is compared before
    every dispatch, so the two cannot be computed differently.
    """
    prng_base, modes = canonical_coverage_identity(
        final_params.get("prng_type", "java_lcg"),
        test_both_modes=bool(final_params.get("test_both_modes", False)))
    sess = [list(s) for s in session_options]
    if pin_bundle:
        missing = [k for k in PIN_KEYS if k not in pin_bundle]
        if missing:
            raise Step1SearchRefused(f"pin bundle is missing {missing}; all seven or none")
        idx = int(pin_bundle["warm_start_session_idx"])
        if not 0 <= idx < len(sess):
            raise Step1SearchRefused(
                f"warm_start_session_idx={idx} is outside session_options {sess}")
        space = {
            "kind": "pinned",
            "pins": {k: pin_bundle[k] for k in PIN_KEYS},
            "effective": {
                "window_size": int(pin_bundle["warm_start_window"]),
                "window_anchor": int(pin_bundle["warm_start_offset"]),
                # Pinned in code, not a parameter (window_optimizer_integration_final.py:1525).
                "generator_phase": 0,
                "skip_min": int(pin_bundle["warm_start_skip_min"]),
                "skip_max": int(pin_bundle["warm_start_skip_max"]),
                "forward_threshold": float(pin_bundle["warm_start_fwd_thresh"]),
                "reverse_threshold": float(pin_bundle["warm_start_rev_thresh"]),
                "sessions": sess[idx],
            },
        }
    else:
        space = {"kind": "bounds",
                 "bounds": {k: dict(v) for k, v in sorted(bounds.items())},
                 "generator_phase": 0}
    # [SKIP-DELIVERY SWITCH] The one validator (miner.range_miner_coordinator), here for
    # both WATCHER paths that reach a search: opening it and every dispatch (S4). An
    # absent key and `false` are the same off state; any other value (null, "", "true",
    # 1) is REFUSED, never read as off. When on, the pinned pair, or the extremes of the
    # sampled bounds, must lie in 0..255 BEFORE a search is opened (Beta R1 Q4).
    from miner.range_miner_coordinator import validate_skip_delivery, MinerMetadataError
    _sd_value = final_params["deliver_skip_range"] if "deliver_skip_range" in final_params \
        else False
    try:
        if pin_bundle:
            _sd_on = validate_skip_delivery(
                _sd_value, pin_bundle["warm_start_skip_min"], pin_bundle["warm_start_skip_max"],
                where="presented_configuration")
        else:
            _sd_on = validate_skip_delivery(_sd_value, where="presented_configuration")
        # A search that can SAMPLE uses the live bounds after the enqueued pin: the full
        # domains are checked under the same rule the Step 1 child applies (Beta
        # X-MINER-SKIP-SWITCH R3 Q4, X-MINER-SKIP-SWITCH-2 R1 Q4).
        # The EFFECTIVE resume flag: under WATCHER a Step 1 search resumes exactly when a
        # study is named (agents/watcher_agent.py, "Study continuity": resume_study =
        # bool(study_name), assigned AFTER this check), so search open and the child see
        # the same flag (Beta X-MINER-SKIP-SWITCH-2 R2 Q4; pinned by G-SW3-RESUME-RULE).
        if _sd_on and skip_delivery_can_sample(bool(pin_bundle), final_params.get("window_trials"),
                                               bool(final_params.get("study_name"))):
            _smin, _smax = bounds.get("skip_min") or {}, bounds.get("skip_max") or {}
            validate_skip_sampling_domains(_smin.get("min"), _smin.get("max"), _smax.get("min"),
                                           _smax.get("max"),
                                           where="presented_configuration (sampling bounds)")
    except MinerMetadataError as exc:
        raise Step1SearchRefused(str(exc)) from exc
    if _sd_on:
        # Only when on, so every baseline digest and search id is unchanged; a
        # switched-on run therefore opens, and must use, a search of its own.
        space["constant_skip_delivery"] = "trial_range"
    trials = final_params.get("window_trials")
    chunk = final_params.get("max_seeds")
    return {
        "prng_base": prng_base,
        "required_modes": list(_modes_tuple(modes)),
        "session_options": sess,
        "session_filter_rule": SESSION_FILTER_RULE_V1,
        "search_space": space,
        "search_space_digest": sha256_text(canonical_json(space)),
        "sampler": {"strategy": final_params.get("strategy"),
                    "n_parallel": final_params.get("n_parallel")},
        "pruning": bool(final_params.get("enable_pruning", False)),
        "trials_per_chunk": None if trials is None else int(trials),
        "chunk_seeds": None if chunk is None else int(chunk),
    }


def compare_configuration(presented: Mapping[str, Any],
                          record: "SearchRecord") -> List[str]:
    """Each mismatching field, naming both values. Empty list = equal."""
    rec = record.configuration()
    out = []
    for f in CONFIG_FIELDS:
        if canonical_json(presented.get(f)) != canonical_json(rec.get(f)):
            out.append(f"{f}: presented={canonical_json(presented.get(f))} "
                       f"recorded={canonical_json(rec.get(f))}")
    return out


def compute_search_id(identity: Mapping[str, Any]) -> str:
    """Hash of a versioned canonical encoding. A retry of the same opening gives the
    same id; an explicit repeat (`repeat_of` set) gives a new one. Provenance that varies
    between retries (time, commit, invocation ref) is deliberately NOT in the identity."""
    payload = {"search_id_encoding_version": SEARCH_ID_ENCODING_VERSION,
               **{k: identity[k] for k in sorted(identity)}}
    return "s1s-" + sha256_text(canonical_json(payload))


IDENTITY_FIELDS = (
    "prng_base", "required_modes", "range_start", "range_end_exclusive",
    "dataset_version_id", "dataset_sha256", "input_manifest_sha256", "dataset_lineage_id",
    "session_options", "session_filter_rule", "search_space", "search_space_digest",
    "sampler", "pruning", "trials_per_chunk", "chunk_seeds", "opened_reason", "repeat_of",
)


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------
_DDL = {
    T_META: f"""CREATE TABLE IF NOT EXISTS {T_META} (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
)""",
    T_SEARCH: f"""CREATE TABLE IF NOT EXISTS {T_SEARCH} (
    search_id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    identity_json TEXT NOT NULL,
    prng_base TEXT NOT NULL,
    required_modes TEXT NOT NULL,
    selector_key TEXT NOT NULL,
    range_start INTEGER NOT NULL,
    range_end_exclusive INTEGER NOT NULL,
    dataset_version_id TEXT NOT NULL,
    dataset_sha256 TEXT NOT NULL,
    input_manifest_sha256 TEXT NOT NULL,
    dataset_lineage_id TEXT NOT NULL,
    session_options_json TEXT NOT NULL,
    session_filter_rule TEXT NOT NULL,
    search_space_json TEXT NOT NULL,
    search_space_digest TEXT NOT NULL,
    sampler_json TEXT NOT NULL,
    pruning INTEGER NOT NULL,
    study_name TEXT,
    trials_per_chunk INTEGER NOT NULL,
    chunk_seeds INTEGER NOT NULL,
    opened_reason TEXT NOT NULL,
    opened_by_ref TEXT NOT NULL,
    repeat_of TEXT,
    opened_utc TEXT NOT NULL,
    repository_commit TEXT NOT NULL,
    CHECK (range_start >= {SEED_DOMAIN_START}),
    CHECK (range_end_exclusive > range_start),
    CHECK (range_end_exclusive <= {SEED_DOMAIN_EXCLUSIVE_MAX}),
    CHECK (trials_per_chunk >= 1),
    CHECK (chunk_seeds >= 1)
)""",
    T_EVENT: f"""CREATE TABLE IF NOT EXISTS {T_EVENT} (
    event_id TEXT PRIMARY KEY,
    search_id TEXT NOT NULL,
    selector_key TEXT NOT NULL,
    event_type TEXT NOT NULL,
    reason TEXT NOT NULL,
    recorded_utc TEXT NOT NULL,
    seq INTEGER NOT NULL,
    CHECK (event_type IN ('opened','closed','superseded','completed'))
)""",
    T_CHUNK: f"""CREATE TABLE IF NOT EXISTS {T_CHUNK} (
    chunk_id TEXT PRIMARY KEY,
    search_id TEXT NOT NULL,
    seed_start INTEGER NOT NULL,
    seed_end_exclusive INTEGER NOT NULL,
    study_name TEXT NOT NULL,
    study_trials_at_open INTEGER NOT NULL,
    budget_target INTEGER NOT NULL,
    checkpoint_run_id TEXT NOT NULL,
    recorded_utc TEXT NOT NULL
)""",
    T_LINK: f"""CREATE TABLE IF NOT EXISTS {T_LINK} (
    link_id TEXT PRIMARY KEY,
    coverage_id TEXT NOT NULL UNIQUE,
    search_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    prng_base TEXT NOT NULL,
    skip_modes_executed TEXT NOT NULL,
    seed_start INTEGER NOT NULL,
    seed_end_exclusive INTEGER NOT NULL,
    generation_id TEXT NOT NULL,
    trials_complete INTEGER NOT NULL,
    trials_pruned INTEGER NOT NULL,
    trials_fail INTEGER NOT NULL,
    trials_other INTEGER NOT NULL,
    recorded_utc TEXT NOT NULL
)""",
    T_PENDING: f"""CREATE TABLE IF NOT EXISTS {T_PENDING} (
    pending_id TEXT PRIMARY KEY,
    coverage_id TEXT NOT NULL UNIQUE,
    search_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    trials_complete INTEGER NOT NULL,
    trials_pruned INTEGER NOT NULL,
    trials_fail INTEGER NOT NULL,
    trials_other INTEGER NOT NULL,
    recorded_utc TEXT NOT NULL
)""",
    T_UNCREDITED: f"""CREATE TABLE IF NOT EXISTS {T_UNCREDITED} (
    attempt_id TEXT PRIMARY KEY,
    search_id TEXT,
    chunk_id TEXT,
    prng_base TEXT NOT NULL,
    seed_start INTEGER NOT NULL,
    seed_end_exclusive INTEGER NOT NULL,
    reason_code TEXT NOT NULL,
    reason_detail TEXT NOT NULL,
    recorded_utc TEXT NOT NULL
)""",
}


def _trigger_sql(table: str, op: str) -> str:
    return (f"CREATE TRIGGER IF NOT EXISTS {table}_no_{op.lower()} BEFORE {op} ON {table} "
            f"BEGIN SELECT RAISE(ABORT, '{table} is append-only: {op} is forbidden'); END")


def _norm_sql(sql: Optional[str]) -> str:
    return " ".join((sql or "").replace("IF NOT EXISTS ", "").split())


def _expected_columns(ddl: str) -> Tuple[str, ...]:
    cols = []
    for line in ddl.split("\n")[1:]:
        tok = line.strip().split(" ")[0]
        if tok and tok not in ("CHECK", ")") and not tok.startswith("CHECK"):
            cols.append(tok.rstrip(","))
    return tuple(cols)


EXPECTED_COLUMNS = {t: _expected_columns(d) for t, d in _DDL.items()}


def _parse_utc(text: str):
    """Both ledgers' ISO timestamps ('...Z' here, '+00:00' in the coverage ledger)."""
    from datetime import datetime
    return datetime.fromisoformat(str(text).replace("Z", "+00:00"))


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class SearchRecord:
    search_id: str
    prng_base: str
    required_modes: Tuple[str, ...]
    range_start: int
    range_end_exclusive: int
    dataset_version_id: str
    dataset_sha256: str
    input_manifest_sha256: str
    dataset_lineage_id: str
    session_options: Any
    session_filter_rule: str
    search_space: Any
    search_space_digest: str
    sampler: Any
    pruning: bool
    study_name: Optional[str]
    trials_per_chunk: int
    chunk_seeds: int
    opened_reason: str
    opened_by_ref: str
    repeat_of: Optional[str]
    opened_utc: str
    repository_commit: str

    def configuration(self) -> Dict[str, Any]:
        return {"prng_base": self.prng_base, "required_modes": list(self.required_modes),
                "session_options": self.session_options,
                "session_filter_rule": self.session_filter_rule,
                "search_space": self.search_space,
                "search_space_digest": self.search_space_digest,
                "sampler": self.sampler, "pruning": self.pruning,
                "trials_per_chunk": self.trials_per_chunk, "chunk_seeds": self.chunk_seeds}


@dataclass(frozen=True)
class Selection:
    """One of the three durable WATCHER results (v1.1 §2.4)."""
    state: str
    search: Optional[SearchRecord]
    cursor: Optional[CursorResult]
    repaired: Tuple[str, ...] = ()        # coverage_ids whose stranded link was rebuilt

    @property
    def next_seed_start(self) -> Optional[int]:
        return None if self.cursor is None else self.cursor.next_seed_start


# ---------------------------------------------------------------------------
# the ledger
# ---------------------------------------------------------------------------
class Step1SearchLedger:
    """The append-only Step 1 search record, in the same SQLite file as the history."""

    def __init__(self, db_path: str = "prng_analysis.db") -> None:
        self.db_path = str(db_path)
        self.init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        conn.execute("PRAGMA recursive_triggers = ON")
        conn.row_factory = sqlite3.Row
        return conn

    # -- schema -----------------------------------------------------------
    def init_schema(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                for t in ALL_TABLES:
                    conn.execute(_DDL[t])
                for t in ALL_TABLES:
                    conn.execute(_trigger_sql(t, "UPDATE"))
                    conn.execute(_trigger_sql(t, "DELETE"))
                row = conn.execute(f"SELECT value FROM {T_META} WHERE key='schema_version'"
                                   ).fetchone()
                if row is None:
                    conn.execute(f"INSERT INTO {T_META}(key, value) VALUES ('schema_version', ?)",
                                 (str(SEARCH_SCHEMA_VERSION),))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            self._assert_schema_exact(conn)

    @staticmethod
    def _assert_schema_exact(conn: sqlite3.Connection) -> None:
        """Exact-schema validation: version, every table's column list, every trigger."""
        v = conn.execute(f"SELECT value FROM {T_META} WHERE key='schema_version'").fetchone()
        if v is None or str(v[0]) != str(SEARCH_SCHEMA_VERSION):
            raise Step1SearchSchemaError(
                f"{T_META}.schema_version is {None if v is None else v[0]!r}; this module "
                f"writes version {SEARCH_SCHEMA_VERSION}. Refusing to read or write.")
        for t in ALL_TABLES:
            found = tuple(r[1] for r in conn.execute(f"PRAGMA table_info({t})"))
            if found != EXPECTED_COLUMNS[t]:
                raise Step1SearchSchemaError(
                    f"{t} has an unexpected shape.\n  expected: {list(EXPECTED_COLUMNS[t])}"
                    f"\n  found:    {list(found)}")
        # [Beta X-STEP1-SEARCH-CODE R1 Q4] EXACT BY DEFINITION, not by name: the stored
        # CREATE text of every table and every append-only trigger must equal this
        # module's own DDL (SQLite stores it with `IF NOT EXISTS ` removed). A same-named
        # trigger with a different (e.g. ineffective) body, or a column whose type or
        # constraint changed, refuses.
        stored = {(r[0], r[1]): r[2] for r in conn.execute(
            "SELECT type, name, sql FROM sqlite_master WHERE type IN ('table','trigger')")}
        need = {("table", t): _DDL[t] for t in ALL_TABLES}
        need.update({("trigger", f"{t}_no_{op.lower()}"): _trigger_sql(t, op)
                     for t in ALL_TABLES for op in ("UPDATE", "DELETE")})
        missing = sorted(n for (k, n) in need if (k, n) not in stored)
        if missing:
            raise Step1SearchSchemaError(f"schema objects missing: {missing}")
        drift = sorted(n for (k, n), ddl in need.items()
                       if _norm_sql(stored[(k, n)]) != _norm_sql(ddl))
        if drift:
            raise Step1SearchSchemaError(
                f"schema objects whose stored definition differs from this module's: {drift}")

    # -- reads ------------------------------------------------------------
    @staticmethod
    def _row_to_record(r: sqlite3.Row) -> SearchRecord:
        return SearchRecord(
            search_id=r["search_id"], prng_base=r["prng_base"],
            required_modes=tuple(r["required_modes"].split(",")),
            range_start=r["range_start"], range_end_exclusive=r["range_end_exclusive"],
            dataset_version_id=r["dataset_version_id"], dataset_sha256=r["dataset_sha256"],
            input_manifest_sha256=r["input_manifest_sha256"],
            dataset_lineage_id=r["dataset_lineage_id"],
            session_options=json.loads(r["session_options_json"]),
            session_filter_rule=r["session_filter_rule"],
            search_space=json.loads(r["search_space_json"]),
            search_space_digest=r["search_space_digest"],
            sampler=json.loads(r["sampler_json"]), pruning=bool(r["pruning"]),
            study_name=r["study_name"], trials_per_chunk=r["trials_per_chunk"],
            chunk_seeds=r["chunk_seeds"], opened_reason=r["opened_reason"],
            opened_by_ref=r["opened_by_ref"], repeat_of=r["repeat_of"],
            opened_utc=r["opened_utc"], repository_commit=r["repository_commit"])

    def get_search(self, search_id: str) -> Optional[SearchRecord]:
        with closing(self._connect()) as conn:
            r = conn.execute(f"SELECT * FROM {T_SEARCH} WHERE search_id=?", (search_id,)
                             ).fetchone()
        return None if r is None else self._row_to_record(r)

    def events(self, search_id: str) -> List[Dict[str, Any]]:
        with closing(self._connect()) as conn:
            return [dict(r) for r in conn.execute(
                f"SELECT * FROM {T_EVENT} WHERE search_id=? ORDER BY seq", (search_id,))]

    @staticmethod
    def _open_ids(conn: sqlite3.Connection, sel: str) -> List[str]:
        rows = conn.execute(
            f"SELECT search_id, event_type FROM {T_EVENT} WHERE selector_key=? ORDER BY seq",
            (sel,)).fetchall()
        opened, ended = [], set()
        for r in rows:
            if r["event_type"] == EV_OPENED:
                opened.append(r["search_id"])
            elif r["event_type"] in TERMINAL_EVENTS:
                ended.add(r["search_id"])
        return [s for s in opened if s not in ended]

    def _search_state(self, conn: sqlite3.Connection, search_id: str) -> str:
        types = [r[0] for r in conn.execute(
            f"SELECT event_type FROM {T_EVENT} WHERE search_id=? ORDER BY seq", (search_id,))]
        if not types:
            return "unknown"
        for t in reversed(types):
            if t in TERMINAL_EVENTS:
                return t
        return "open"

    def search_state(self, search_id: str) -> str:
        with closing(self._connect()) as conn:
            return self._search_state(conn, search_id)

    def credited_intervals(self, search_id: str) -> List[Tuple[int, int]]:
        with closing(self._connect()) as conn:
            return [(r[0], r[1]) for r in conn.execute(
                f"SELECT seed_start, seed_end_exclusive FROM {T_LINK} WHERE search_id=? "
                f"ORDER BY seed_start", (search_id,))]

    def cursor(self, record: SearchRecord) -> CursorResult:
        """THE search cursor: the first gap among this search's credited chunks inside its
        declared range — `first_uncovered_seed`, imported, not re-implemented."""
        return first_uncovered_seed(self.credited_intervals(record.search_id),
                                    domain_start=record.range_start,
                                    domain_end_exclusive=record.range_end_exclusive)

    def select(self, prng_base: str, required_modes: Sequence[str]) -> Selection:
        """Which search WATCHER runs, as one of three durable results. Raises
        Step1SearchLookupError on any failure — there is no default and no seed 0."""
        try:
            sel = selector_key(prng_base, _modes_tuple(required_modes))
            with closing(self._connect()) as conn:
                open_ids = self._open_ids(conn, sel)
                if len(open_ids) > 1:
                    raise Step1SearchLookupError(
                        f"{len(open_ids)} open searches for {sel}: {open_ids}. The one-open "
                        f"rule is violated; refusing to choose.")
                if open_ids:
                    rec = self.get_search(open_ids[0])
                    if rec is None:
                        raise Step1SearchLookupError(f"open search {open_ids[0]} has no record")
                    # [Beta R1 Q5] a link that failed after publication is rebuilt HERE,
                    # before the gap is chosen, so the published chunk is never re-run and
                    # never stranded.
                    rep = tuple(r["coverage_id"] for r in self.repair_pending_links(rec.search_id))
                    cur = self.cursor(rec)
                    if cur.is_complete:
                        self._append_event(rec.search_id, sel, EV_COMPLETED,
                                           "every chunk of the declared range is credited",
                                           deterministic=True)
                        return Selection(SEL_SEARCH_COMPLETE, rec, cur, rep)
                    return Selection(SEL_GAP, rec, cur, rep)
                last = conn.execute(
                    f"SELECT search_id, event_type FROM {T_EVENT} WHERE selector_key=? "
                    f"ORDER BY seq DESC LIMIT 1", (sel,)).fetchone()
                if last is not None and last["event_type"] == EV_COMPLETED:
                    rec = self.get_search(last["search_id"])
                    return Selection(SEL_SEARCH_COMPLETE, rec, self.cursor(rec))
                return Selection(SEL_NO_OPEN_SEARCH, None, None)
        except Step1SearchError:
            raise
        except Exception as exc:
            raise Step1SearchLookupError(f"search lookup failed: {type(exc).__name__}: {exc}"
                                         ) from exc

    # -- writes: lifecycle -------------------------------------------------
    def _append_event(self, search_id: str, sel: str, event_type: str, reason: str, *,
                      deterministic: bool = False, conn: Optional[sqlite3.Connection] = None):
        own = conn is None
        c = conn or self._connect()
        try:
            if own:
                c.execute("BEGIN IMMEDIATE")
            eid = (sha256_text(f"{search_id}|{event_type}") if deterministic
                   else f"ev-{uuid.uuid4().hex}")
            if c.execute(f"SELECT 1 FROM {T_EVENT} WHERE event_id=?", (eid,)).fetchone() is None:
                seq = c.execute(f"SELECT COALESCE(MAX(seq),0)+1 FROM {T_EVENT}").fetchone()[0]
                c.execute(f"INSERT INTO {T_EVENT}(event_id, search_id, selector_key, event_type,"
                          f" reason, recorded_utc, seq) VALUES (?,?,?,?,?,?,?)",
                          (eid, search_id, sel, event_type, reason, _utc_now(), seq))
            if own:
                c.execute("COMMIT")
        except BaseException:
            if own:
                c.execute("ROLLBACK")
            raise
        finally:
            if own:
                c.close()

    def open_search(self, *, configuration: Mapping[str, Any], range_start: int,
                    range_end_exclusive: int, dataset: Mapping[str, Any], opened_reason: str,
                    opened_by_ref: str, repository_commit: str,
                    repeat_of: Optional[str] = None) -> Tuple[SearchRecord, bool]:
        """Open a search. Returns (record, created). One transaction:
          * same identity already open      -> the same record, created=False (retry);
          * same identity, no longer open   -> refuse (an explicit repeat needs repeat_of);
          * another search open on selector -> refuse;
          * otherwise insert record + `opened` event.
        """
        if opened_reason not in OPEN_REASONS:
            raise Step1SearchRefused(f"opened_reason {opened_reason!r} not in {OPEN_REASONS}")
        for k in ("dataset_version_id", "dataset_sha256", "input_manifest_sha256",
                  "dataset_lineage_id"):
            if not dataset.get(k):
                raise Step1SearchRefused(
                    f"the current frozen dataset does not resolve {k} ({dataset.get(k)!r}); "
                    f"a search cannot be opened without its full dataset identity")
        rs, re_ = int(range_start), int(range_end_exclusive)
        if not (SEED_DOMAIN_START <= rs < re_ <= SEED_DOMAIN_EXCLUSIVE_MAX):
            raise Step1SearchRefused(
                f"declared range [{rs}, {re_}) is not a nonempty interval inside "
                f"[{SEED_DOMAIN_START}, {SEED_DOMAIN_EXCLUSIVE_MAX})")
        cfg = dict(configuration)
        for f in ("trials_per_chunk", "chunk_seeds"):
            if not isinstance(cfg.get(f), int) or cfg[f] < 1:
                raise Step1SearchRefused(f"{f} must be a positive integer, got {cfg.get(f)!r}")
        identity = {**{f: cfg[f] for f in CONFIG_FIELDS}, "range_start": rs,
                    "range_end_exclusive": re_,
                    **{k: dataset[k] for k in ("dataset_version_id", "dataset_sha256",
                                               "input_manifest_sha256", "dataset_lineage_id")},
                    "opened_reason": opened_reason, "repeat_of": repeat_of}
        assert set(identity) == set(IDENTITY_FIELDS), sorted(set(identity) ^ set(IDENTITY_FIELDS))
        search_id = compute_search_id(identity)
        sel = selector_key(cfg["prng_base"], cfg["required_modes"])
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                if repeat_of is not None and conn.execute(
                        f"SELECT 1 FROM {T_SEARCH} WHERE search_id=?", (repeat_of,)
                        ).fetchone() is None:
                    raise Step1SearchRefused(f"repeat_of {repeat_of!r} names no search")
                existing = conn.execute(f"SELECT * FROM {T_SEARCH} WHERE search_id=?",
                                        (search_id,)).fetchone()
                open_ids = self._open_ids(conn, sel)
                if existing is not None:
                    state = self._search_state(conn, search_id)
                    conn.execute("ROLLBACK")
                    if state == "open":
                        return self._row_to_record(existing), False
                    raise Step1SearchRefused(
                        f"search {search_id} has the same identity and is {state}; a "
                        f"retry does not reopen it. Open an explicit repeat with "
                        f"repeat_of={search_id}.")
                if open_ids:
                    raise Step1SearchRefused(
                        f"a search is already open for {sel}: {open_ids[0]}. At most one "
                        f"open search per family and mode set; close it first.")
                conn.execute(
                    f"INSERT INTO {T_SEARCH} (search_id, schema_version, identity_json, "
                    f"prng_base, required_modes, selector_key, range_start, range_end_exclusive,"
                    f" dataset_version_id, dataset_sha256, input_manifest_sha256, "
                    f"dataset_lineage_id, session_options_json, session_filter_rule, "
                    f"search_space_json, search_space_digest, sampler_json, pruning, "
                    f"study_name, trials_per_chunk, chunk_seeds, opened_reason, opened_by_ref,"
                    f" repeat_of, opened_utc, repository_commit) VALUES "
                    f"(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (search_id, SEARCH_SCHEMA_VERSION, canonical_json(identity),
                     cfg["prng_base"], ",".join(cfg["required_modes"]), sel, rs, re_,
                     dataset["dataset_version_id"], dataset["dataset_sha256"],
                     dataset["input_manifest_sha256"], dataset["dataset_lineage_id"],
                     canonical_json(cfg["session_options"]), cfg["session_filter_rule"],
                     canonical_json(cfg["search_space"]), cfg["search_space_digest"],
                     canonical_json(cfg["sampler"]), 1 if cfg["pruning"] else 0,
                     None,  # bound by the search's first chunk-open (S9; see chunk_open)
                     cfg["trials_per_chunk"], cfg["chunk_seeds"], opened_reason,
                     opened_by_ref, repeat_of, _utc_now(), repository_commit))
                self._append_event(search_id, sel, EV_OPENED, f"opened: {opened_reason}",
                                   deterministic=True, conn=conn)
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
        return self.get_search(search_id), True

    def close_search(self, search_id: str, reason: str) -> None:
        if not reason or not str(reason).strip():
            raise Step1SearchRefused("closing a search requires a reason")
        rec = self.get_search(search_id)
        if rec is None:
            raise Step1SearchRefused(f"no search {search_id!r}")
        state = self.search_state(search_id)
        if state not in ("open", EV_COMPLETED):
            raise Step1SearchRefused(f"search {search_id} is {state}; nothing to close")
        self._append_event(search_id, selector_key(rec.prng_base, rec.required_modes),
                           EV_CLOSED, str(reason))

    # -- writes: chunks ----------------------------------------------------
    @staticmethod
    def chunk_id_for(search_id: str, seed_start: int, seed_end_exclusive: int) -> str:
        return "chk-" + sha256_text(f"{search_id}|{int(seed_start)}|{int(seed_end_exclusive)}")

    def study_bound_to(self, search_id: str) -> Optional[str]:
        with closing(self._connect()) as conn:
            r = conn.execute(f"SELECT study_name FROM {T_CHUNK} WHERE search_id=? "
                             f"ORDER BY recorded_utc LIMIT 1", (search_id,)).fetchone()
        return None if r is None else r[0]

    def study_owner(self, study_name: str) -> Optional[str]:
        with closing(self._connect()) as conn:
            r = conn.execute(f"SELECT search_id FROM {T_CHUNK} WHERE study_name=? "
                             f"ORDER BY recorded_utc LIMIT 1", (study_name,)).fetchone()
        return None if r is None else r[0]

    def assert_study_usable(self, search_id: str, study_name: str) -> None:
        """S9. A search owns one study (bound by its first chunk-open); a study already
        owned by another search, or a name other than this search's bound study, refuses."""
        owner = self.study_owner(study_name)
        if owner is not None and owner != search_id:
            raise Step1StudyOwnershipError(
                f"study {study_name!r} already belongs to search {owner}; presented for "
                f"search {search_id}. A study is owned by one search.")
        bound = self.study_bound_to(search_id)
        if bound is not None and bound != study_name:
            raise Step1StudyOwnershipError(
                f"search {search_id} owns study {bound!r}; presented study {study_name!r}.")

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        with closing(self._connect()) as conn:
            r = conn.execute(f"SELECT * FROM {T_CHUNK} WHERE chunk_id=?", (chunk_id,)).fetchone()
        return None if r is None else dict(r)

    def chunk_open(self, *, search_id: str, seed_start: int, seed_end_exclusive: int,
                   study_name: str, study_trials_at_open: int, budget_target: int,
                   checkpoint_run_id: str) -> Dict[str, Any]:
        """Before a chunk's first trial. Idempotent: a restart READS the existing event
        and does not recompute a baseline."""
        rec = self.get_search(search_id)
        if rec is None:
            raise Step1SearchRefused(f"chunk-open for unknown search {search_id!r}")
        s, e = int(seed_start), int(seed_end_exclusive)
        if not (rec.range_start <= s < e <= rec.range_end_exclusive):
            raise Step1SearchRefused(
                f"chunk [{s}, {e}) is outside search {search_id} range "
                f"[{rec.range_start}, {rec.range_end_exclusive})")
        self.assert_study_usable(search_id, study_name)
        cid = self.chunk_id_for(search_id, s, e)
        existing = self.get_chunk(cid)
        if existing is not None:
            if existing["study_name"] != study_name:
                raise Step1StudyOwnershipError(
                    f"chunk {cid} was opened on study {existing['study_name']!r}; presented "
                    f"{study_name!r}")
            return existing
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    f"INSERT INTO {T_CHUNK}(chunk_id, search_id, seed_start, seed_end_exclusive,"
                    f" study_name, study_trials_at_open, budget_target, checkpoint_run_id, "
                    f"recorded_utc) VALUES (?,?,?,?,?,?,?,?,?)",
                    (cid, search_id, s, e, study_name, int(study_trials_at_open),
                     int(budget_target), str(checkpoint_run_id), _utc_now()))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return self.get_chunk(cid)

    def record_uncredited(self, *, search_id: Optional[str], chunk_id: Optional[str],
                          prng_base: str, seed_start: int, seed_end_exclusive: int,
                          reason_code: str, reason_detail: str) -> str:
        """The NON-CERTIFYING record of an attempt that earned no credit, with its reason."""
        aid = f"unc-{uuid.uuid4().hex}"
        with closing(self._connect()) as conn:
            conn.execute(
                f"INSERT INTO {T_UNCREDITED}(attempt_id, search_id, chunk_id, prng_base, "
                f"seed_start, seed_end_exclusive, reason_code, reason_detail, recorded_utc) "
                f"VALUES (?,?,?,?,?,?,?,?,?)",
                (aid, search_id, chunk_id, prng_base, int(seed_start), int(seed_end_exclusive),
                 reason_code, reason_detail, _utc_now()))
        return aid

    # -- the link: one check-and-insert path for the first write and for repair --------
    def _link_problems(self, rec: SearchRecord, chunk: Mapping[str, Any], *, prng_base: str,
                       modes: Tuple[str, ...], iv: Tuple[int, int]) -> List[str]:
        problems = []
        if prng_base != rec.prng_base:
            problems.append(f"witness family {prng_base} != search {rec.prng_base}")
        if tuple(modes) != tuple(rec.required_modes):
            problems.append(f"witness modes {list(modes)} != search {list(rec.required_modes)}")
        if iv != (chunk["seed_start"], chunk["seed_end_exclusive"]):
            problems.append(f"witness interval {iv} != chunk "
                            f"{(chunk['seed_start'], chunk['seed_end_exclusive'])}")
        if not (rec.range_start <= iv[0] < iv[1] <= rec.range_end_exclusive):
            problems.append(f"witness interval {iv} outside search range "
                            f"[{rec.range_start}, {rec.range_end_exclusive})")
        return problems

    @staticmethod
    def _witness_vs_coverage(artifact: Any, coverage: Any) -> Tuple[Tuple[str, ...],
                                                                    Tuple[int, int], List[str]]:
        modes = _modes_tuple(artifact.skip_modes_executed)
        a_iv = (int(artifact.seed_start), int(artifact.seed_end_exclusive))
        problems = []
        if coverage.prng_base != artifact.prng_base:
            problems.append(f"coverage prng_base {coverage.prng_base} != witness "
                            f"{artifact.prng_base}")
        if coverage.as_interval() != a_iv:
            problems.append(f"coverage interval {coverage.as_interval()} != witness {a_iv}")
        if coverage.generation_id != artifact.generation_id:
            problems.append(f"coverage generation {coverage.generation_id} != witness "
                            f"{artifact.generation_id}")
        if coverage.skip_modes_executed != ",".join(modes):
            problems.append(f"coverage modes {coverage.skip_modes_executed} != witness "
                            f"{','.join(modes)}")
        if coverage.artifact_sha256 != artifact.artifact_sha256:
            problems.append("coverage artifact_sha256 != witness artifact_sha256")
        return modes, a_iv, problems

    def _search_and_chunk(self, search_id: str, chunk_id: str):
        rec = self.get_search(search_id)
        if rec is None:
            raise Step1LinkError(f"link for unknown search {search_id!r}")
        chunk = self.get_chunk(chunk_id)
        if chunk is None or chunk["search_id"] != search_id:
            raise Step1LinkError(f"chunk {chunk_id!r} is not a chunk of search {search_id}")
        return rec, chunk

    def _insert_idempotent(self, table: str, key_col: str, row: Dict[str, Any]) -> Dict[str, Any]:
        """Insert `row` into an append-only table, idempotent on `key_col`: an identical
        existing row is returned; a different one refuses. Never half-writes."""
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                ex = conn.execute(f"SELECT * FROM {table} WHERE {key_col}=?",
                                  (row[key_col],)).fetchone()
                if ex is not None:
                    conn.execute("ROLLBACK")
                    exd = dict(ex)
                    if not all(exd[k] == v for k, v in row.items()):
                        raise Step1LinkError(
                            f"{table}: {key_col} {row[key_col]} already recorded with different "
                            f"content (search {exd.get('search_id')}, chunk {exd.get('chunk_id')})")
                    return exd
                cols = list(row) + ["recorded_utc"]
                conn.execute(f"INSERT INTO {table}({', '.join(cols)}) VALUES "
                             f"({', '.join('?' for _ in cols)})",
                             tuple(row.values()) + (_utc_now(),))
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
        with closing(self._connect()) as conn:
            return dict(conn.execute(f"SELECT * FROM {table} WHERE {key_col}=?",
                                     (row[key_col],)).fetchone())

    @staticmethod
    def _counts(trial_counts: Mapping[str, int]) -> Dict[str, int]:
        return {k: int(trial_counts.get(k, 0)) for k in ("complete", "pruned", "fail", "other")}

    def _link_row(self, *, search_id, chunk_id, coverage_id, prng_base, modes, iv,
                  generation_id, counts) -> Dict[str, Any]:
        return {"link_id": "lnk-" + sha256_text(f"{coverage_id}|{search_id}"),
                "coverage_id": coverage_id, "search_id": search_id, "chunk_id": chunk_id,
                "prng_base": prng_base, "skip_modes_executed": ",".join(modes),
                "seed_start": iv[0], "seed_end_exclusive": iv[1],
                "generation_id": generation_id, "trials_complete": counts["complete"],
                "trials_pruned": counts["pruned"], "trials_fail": counts["fail"],
                "trials_other": counts["other"]}

    def record_link_pending(self, *, search_id: str, chunk_id: str, artifact: Any,
                            coverage: Any, trial_counts: Mapping[str, int]) -> Dict[str, Any]:
        """[Beta R1 Q5] Written after the history row and BEFORE the link, with the same
        checks the link applies. It is what lets a fresh process repair a link whose write
        failed after publication. Idempotent on `coverage_id`."""
        CoverageLedger._require_publication_witness(artifact)
        rec, chunk = self._search_and_chunk(search_id, chunk_id)
        modes, iv, problems = self._witness_vs_coverage(artifact, coverage)
        problems += self._link_problems(rec, chunk, prng_base=artifact.prng_base, modes=modes,
                                        iv=iv)
        if problems:
            raise Step1LinkError("pending link refused: " + "; ".join(problems))
        c = self._counts(trial_counts)
        return self._insert_idempotent(T_PENDING, "coverage_id", {
            "pending_id": "pnd-" + sha256_text(f"{coverage.coverage_id}|{search_id}"),
            "coverage_id": coverage.coverage_id, "search_id": search_id, "chunk_id": chunk_id,
            "generation_id": artifact.generation_id, "trials_complete": c["complete"],
            "trials_pruned": c["pruned"], "trials_fail": c["fail"], "trials_other": c["other"]})

    def record_chunk_link(self, *, search_id: str, chunk_id: str, artifact: Any,
                          coverage: Any, trial_counts: Mapping[str, int]) -> Dict[str, Any]:
        """The link, after a credited publication. Every interval / family / mode /
        generation field is DERIVED from the publication witness (`artifact`, a
        RunArtifactResult) and the history row the existing door wrote from it
        (`coverage`, a CertifiedInterval), and checked against the search and the chunk.
        Idempotent on `coverage_id`. Raises Step1LinkError; never half-writes."""
        CoverageLedger._require_publication_witness(artifact)
        rec, chunk = self._search_and_chunk(search_id, chunk_id)
        modes, iv, problems = self._witness_vs_coverage(artifact, coverage)
        problems += self._link_problems(rec, chunk, prng_base=artifact.prng_base, modes=modes,
                                        iv=iv)
        if problems:
            raise Step1LinkError("link refused: " + "; ".join(problems))
        return self._insert_idempotent(T_LINK, "coverage_id", self._link_row(
            search_id=search_id, chunk_id=chunk_id, coverage_id=coverage.coverage_id,
            prng_base=artifact.prng_base, modes=modes, iv=iv,
            generation_id=artifact.generation_id, counts=self._counts(trial_counts)))

    def repair_pending_links(self, search_id: str) -> List[Dict[str, Any]]:
        """[Beta R1 Q5] Rebuild every link of `search_id` whose pending record exists but
        whose link does not. The fields come from the IMMUTABLE history row the one
        certification door wrote from the publication witness (read by its coverage_id),
        checked against the pending record, the chunk and the search — no new trial, no new
        generation, no caller-asserted value. Returns the links written (empty if none)."""
        with closing(self._connect()) as conn:
            pend = [dict(r) for r in conn.execute(
                f"SELECT p.* FROM {T_PENDING} p LEFT JOIN {T_LINK} l "
                f"ON l.coverage_id = p.coverage_id WHERE p.search_id=? AND l.coverage_id IS NULL "
                f"ORDER BY p.recorded_utc", (search_id,))]
        out = []
        if not pend:
            return out + self._reconstruct_links(search_id)
        hist = CoverageLedger(self.db_path)
        for p in pend:
            rec, chunk = self._search_and_chunk(search_id, p["chunk_id"])
            cov = hist.get(p["coverage_id"])
            if cov is None:
                raise Step1LinkError(
                    f"pending link {p['pending_id']}: history row {p['coverage_id']} not found")
            modes = _modes_tuple(cov.skip_modes_executed.split(","))
            iv = cov.as_interval()
            problems = self._link_problems(rec, chunk, prng_base=cov.prng_base, modes=modes,
                                           iv=iv)
            if cov.generation_id != p["generation_id"]:
                problems.append(f"history generation {cov.generation_id} != pending "
                                f"{p['generation_id']}")
            if problems:
                raise Step1LinkError("link repair refused: " + "; ".join(problems))
            out.append(self._insert_idempotent(T_LINK, "coverage_id", self._link_row(
                search_id=search_id, chunk_id=p["chunk_id"], coverage_id=cov.coverage_id,
                prng_base=cov.prng_base, modes=modes, iv=iv, generation_id=cov.generation_id,
                counts={"complete": p["trials_complete"], "pruned": p["trials_pruned"],
                        "fail": p["trials_fail"], "other": p["trials_other"]})))
        return out + self._reconstruct_links(search_id)

    def _reconstruct_links(self, search_id: str) -> List[Dict[str, Any]]:
        """[Beta X-STEP1-SEARCH-CODE R2 Q2, LIVE] Recovery when even the PENDING write
        failed after the history row. The binding comes from two rows that both exist
        before or by the history commit:
          * the chunk-open (written BEFORE the chunk's first trial), naming the chunk's
            interval and its Optuna study; S9 makes that study owned by this search only;
          * the history row, which the one certification door wrote from the publication
            witness, carrying the caller-supplied provenance `study_identity` = that study
            (window_optimizer_integration_final.py, search chunks only).
        A history row is bound to a chunk only if its interval, family, mode set, dataset
        digest and study_identity all equal the chunk's and the search's, it was recorded
        AFTER the chunk-open, and it is linked to nothing. Exactly one candidate links
        (trial counts unknown: recorded as -1). None: nothing to repair. More than one:
        refuse — never a guess."""
        rec = self.get_search(search_id)
        if rec is None:
            return []
        with closing(self._connect()) as conn:
            chunks = [dict(r) for r in conn.execute(
                f"SELECT c.* FROM {T_CHUNK} c LEFT JOIN {T_LINK} l ON l.chunk_id = c.chunk_id "
                f"LEFT JOIN {T_PENDING} p ON p.chunk_id = c.chunk_id "
                f"WHERE c.search_id=? AND l.chunk_id IS NULL AND p.chunk_id IS NULL",
                (search_id,))]
            linked = {r[0] for r in conn.execute(f"SELECT coverage_id FROM {T_LINK}")}
        if not chunks:
            return []
        hist = CoverageLedger(self.db_path)
        rows = hist.certified_records(rec.prng_base)
        out = []
        for ch in chunks:
            opened = _parse_utc(ch["recorded_utc"])
            cands = [r for r in rows
                     if r.as_interval() == (ch["seed_start"], ch["seed_end_exclusive"])
                     and r.skip_modes_executed == ",".join(rec.required_modes)
                     and r.dataset_sha256 == rec.dataset_sha256
                     and r.study_identity == ch["study_name"]
                     and _parse_utc(r.recorded_at) >= opened
                     and r.coverage_id not in linked]
            if not cands:
                continue
            if len(cands) > 1:
                raise Step1LinkError(
                    f"link reconstruction refused for chunk {ch['chunk_id']}: "
                    f"{len(cands)} history rows match "
                    f"{[c.coverage_id[:16] for c in cands]}; refusing to choose")
            cov = cands[0]
            modes = _modes_tuple(cov.skip_modes_executed.split(","))
            problems = self._link_problems(rec, ch, prng_base=cov.prng_base, modes=modes,
                                           iv=cov.as_interval())
            if problems:
                raise Step1LinkError("link reconstruction refused: " + "; ".join(problems))
            out.append(self._insert_idempotent(T_LINK, "coverage_id", self._link_row(
                search_id=search_id, chunk_id=ch["chunk_id"], coverage_id=cov.coverage_id,
                prng_base=cov.prng_base, modes=modes, iv=cov.as_interval(),
                generation_id=cov.generation_id,
                counts={"complete": -1, "pruned": -1, "fail": -1, "other": -1})))
        return out


# ---------------------------------------------------------------------------
# S7 — the credit decision (pure; called by the integration layer before publication)
# ---------------------------------------------------------------------------
def credit_decision(*, requested_interval: Tuple[int, int], required_phases: Sequence[int],
                    evidence_by_trial: Mapping[int, Mapping[str, Any]],
                    trial_states: Mapping[int, str]) -> Tuple[bool, str, str, List[int]]:
    """Return (credited, reason_code, reason_detail, credited_trial_numbers).

    Credited only if at least one trial is in state COMPLETE, has sweep evidence recorded
    IN THIS CHUNK (by this process, at the trial), finished every required phase, and
    every phase's swept interval EQUALS the requested interval (owner ruling 2026-11-14,
    item 2a, second guard). Zero survivors are irrelevant: credit needs execution evidence,
    not a survivor count.
    """
    req = (int(requested_interval[0]), int(requested_interval[1]))
    need = sorted(int(p) for p in required_phases)
    complete = sorted(n for n, s in trial_states.items() if s == "COMPLETE")
    if not complete:
        states = sorted(set(trial_states.values()))
        return (False, "no_complete_trial",
                f"no trial of this chunk is COMPLETE (states seen: {states or 'none'})", [])
    good, why = [], []
    for n in complete:
        ev = evidence_by_trial.get(n)
        if ev is None:
            why.append(f"trial {n}: COMPLETE in the study but no sweep evidence from this chunk")
            continue
        phases = ev.get("swept_intervals") or {}
        missing = [p for p in need if str(p) not in phases and p not in phases]
        if missing:
            why.append(f"trial {n}: phases {missing} not finished")
            continue
        bad = []
        for p in need:
            iv = phases.get(str(p), phases.get(p))
            if iv is None or (int(iv[0]), int(iv[1])) != req:
                bad.append(f"phase {p} swept {iv}")
        if bad:
            why.append(f"trial {n}: swept interval != requested {list(req)}: " + "; ".join(bad))
            continue
        good.append(n)
    if not good:
        code = ("interval_mismatch" if any("swept interval" in w for w in why)
                else "no_sweep_evidence")
        return False, code, " | ".join(why), []
    return True, "credited", f"trials {good} completed every required phase over {list(req)}", good
