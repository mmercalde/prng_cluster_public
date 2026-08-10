#!/usr/bin/env python3
"""
GATE-12 CONCURRENCY SAMPLER — measures the POST-F1 state model
==============================================================

WHY THIS EXISTS
  The 2026-08-09 sampler asked

      count(distinct claimed_by) WHERE state IN ('claimed','staging')

  and never looked at `pending`. Under the certified F1 scheduler that query is
  wrong on both terms, and the term it omits is the one Beta's criterion turns
  on:

    * `claimed`  now means COMPUTE-ACTIVE — exactly one row per serial worker.
      This is the right term, but only together with the next two.
    * `staging`  means the worker already returned StripeComplete and its
      compute slot was RELEASED (`compute_busy_worker_ids`,
      range_miner_coordinator.py, excludes it deliberately). Counting it
      OVERSTATES occupancy.
    * `pending`  is now a REAL coordinator-owned backlog — 24 rows at W=8,
      7 at W=25 — because F1 creates the full stripe geometry born
      pending/claimed_by NULL/lease NULL and hands stripes out at real handoff.
      QUEUE DEPTH IS EXACTLY WHAT BETA'S CRITERION REQUIRES and the old query
      could not see it.

BETA'S CRITERION — what this tool must be able to prove
      "An observation window in which >=25 DISTINCT workers were simultaneously
       compute-active AND queued stripes remained available."

  Explicitly insufficient, and reported separately as such:
      "25 workers connected" · "25 distinct workers eventually used"
      "32 stripes eventually completed"

  SIMULTANEITY IS THE WHOLE POINT. A maximum-over-time of distinct worker ids
  is a union across instants: 25 workers that each ran alone, one after another,
  produce the same union as 25 running together. This tool therefore evaluates
  the criterion PER SAMPLE and reports consecutive satisfying samples as a
  window; the union is computed too, and printed under a heading that says it
  does not qualify.

TWO VERDICTS, NEVER COLLAPSED (Beta R1 §5)
  Simultaneity proves the queue was NON-EMPTY. It does not prove the queue was
  CONSUMED. 32 stripes were chosen over the 25-stripe minimum precisely so seven
  queued stripes would exercise scheduler turnover, completion, reassignment,
  staging and back-pressure UNDER full occupancy — and a run holding 25 claimed
  / 7 pending frozen forever satisfies simultaneity while demonstrating none of
  it. So:
      VERDICT 1  sustained simultaneity  — the qualifying window
      VERDICT 2  turnover under full occupancy — queued work actually consumed
                 DURING that window (pending strictly decreased, and/or stripes
                 transitioned into done/staging), with the pending delta and the
                 transition count both reported.
  They are stated separately and labelled. A run may satisfy one and fail the
  other; that distinction is the point.

ONE SAMPLE IS ONE SNAPSHOT (Beta R1 §3)
  The occupancy read and the queue-depth read are wrapped in a single explicit
  `BEGIN DEFERRED` ... `COMMIT` on the read-only connection (`read_snapshot`).
  Under WAL a deferred read transaction takes its snapshot at the first read
  statement and holds it until COMMIT, so both statements observe the same
  instant even while the coordinator commits between them. Two autocommit reads
  are two independent read transactions, and the sample that decides the verdict
  is the one most likely to straddle a transition.

ESTAB IS CONTEXT, AND IT IS HONEST (Beta R1 §4)
  `ss` unavailable, non-zero exit, timeout or unparseable output is recorded as
  UNAVAILABLE / None — never as 0, which is the identical "unobservable rendered
  as a definite zero" defect the GPU probe was just certified for fixing. ESTAB
  is not a term in either criterion and cannot move either verdict.

SAFETY
  * Read-only: every connection is opened `file:...?mode=ro` with uri=True.
  * A denylist refuses production analysis databases outright. A prior harness
    CREATED A REAL TABLE in the live DB through cwd-relative resolution.
  * The ledger path is derived from agent_manifests/window_optimizer.json
    (`staging_dir`), never hardcoded; --ledger overrides.

USAGE
  scripts/gate12_concurrency_sampler.py --out logs/x.tsv --summary logs/x.txt \
      --watch-pid <watcher_pid> --port 5700 --threshold 25
"""

import argparse
import contextlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

REPO = Path(__file__).resolve().parents[1]

# The certified F1 stripe states (range_miner_coordinator.py ST_* block).
ST_PENDING = "pending"
ST_CLAIMED = "claimed"
ST_STAGING = "staging"
ST_DONE = "done"
ST_FAILED = "failed"
ST_CANCELLED = "cancelled"

# Databases this tool must never open, even read-only: pointing a sampler at a
# production analysis DB is how a harness once created a table in one.
FORBIDDEN_DB_NAMES = {
    "prng_analysis.db",
    "miner_ledger_prod.db",
    "optuna_studies.db",
}

TSV_COLUMNS = [
    "ts_iso", "epoch", "run_id",
    "obs_status",         # OBSERVED | UNOBSERVED — a failed read is not a zero
    "obs_reason",         # why the ledger read did not happen, when it did not
    "compute_active",     # distinct workers with a COMPUTE-ACTIVE claim
    "queued_pending",     # coordinator-owned backlog
    "claimed_rows",       # rows in 'claimed'; == compute_active under F1
    "staging", "done", "cancelled", "failed",   # context, NOT occupancy
    "estab",              # context, NOT occupancy; UNAVAILABLE is not 0
    "estab_reason",       # why ESTAB is unavailable, when it is
    "satisfies",          # this SAMPLE meets the simultaneity criterion
]

# ESTAB outcome vocabulary. Same discipline as the certified preflight GPU probe:
# an attempted-and-failed observation is UNAVAILABLE, never a definite zero.
ESTAB_OK = "OK"
ESTAB_UNAVAILABLE = "UNAVAILABLE"
ESTAB_TIMEOUT_SECONDS = 5

# LEDGER-READ outcome vocabulary — the same discipline, applied where the
# CRITERION actually lives (Beta R2 §1). A sample whose ledger read failed is
# UNOBSERVED: it is not a measurement of an empty fleet, it is the absence of a
# measurement, and it may never reach the verdict as `compute_active=0`.
OBS_OBSERVED = "OBSERVED"
OBS_UNOBSERVED = "UNOBSERVED"

# Every per-sample ledger quantity. Listed once so `unobserved_row` cannot drift
# out of step with `sample_run` and silently leave a stale zero behind.
LEDGER_FIELDS = ("compute_active", "queued_pending", "claimed_rows",
                 "staging", "done", "cancelled", "failed")


# ──────────────────────────────────────────────────────────────────────────────
# ledger access — read-only, scoped to one run
# ──────────────────────────────────────────────────────────────────────────────

def default_ledger_path() -> str:
    """staging_dir from the committed manifest; the coordinator joins the same
    'miner_ledger.db' onto it (range_miner_coordinator.py run_trial_miner)."""
    manifest = REPO / "agent_manifests" / "window_optimizer.json"
    staging = None
    try:
        with open(manifest) as fh:
            staging = (json.load(fh).get("default_params") or {}).get("staging_dir")
    except Exception:
        staging = None
    if not staging:
        # Fall back to a top-level key rather than guessing a filesystem path.
        try:
            with open(manifest) as fh:
                staging = json.load(fh).get("staging_dir")
        except Exception:
            staging = None
    if not staging:
        raise SystemExit(
            "cannot derive staging_dir from agent_manifests/window_optimizer.json "
            "— pass --ledger explicitly rather than assuming a path")
    return os.path.join(staging, "miner_ledger.db")


def guard_db_path(path: str) -> str:
    name = os.path.basename(path)
    if name in FORBIDDEN_DB_NAMES:
        raise SystemExit(f"REFUSED: {path} is a production database, not the "
                         f"miner ledger. This tool never opens it.")
    if not os.path.exists(path):
        raise SystemExit(f"REFUSED: ledger {path} does not exist. The sampler "
                         f"never creates a database.")
    return path


def connect_ro(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    # Manual transaction control. With the default isolation_level the sqlite3
    # module issues implicit BEGINs only ahead of DML, so SELECTs run in
    # autocommit and `read_snapshot`'s explicit BEGIN would be racing the
    # module's own bookkeeping. None means the module never emits BEGIN/COMMIT
    # of its own and our transaction is the only one there is.
    conn.isolation_level = None
    return conn


@contextlib.contextmanager
def read_snapshot(conn: sqlite3.Connection):
    """ONE WAL snapshot for every read inside the block.

    WHY (Beta R1 §3). A sample is two reads — the occupancy set and the queue
    depth. Issued in autocommit they are two INDEPENDENT read transactions, and
    a stripe transitioning between them yields a sample whose two halves
    describe different instants. The window in which that happens is exactly the
    interesting one: 32 stripes, 25 workers, turnover in progress. Both a false
    positive (occupancy read before a release, queue read before the drain) and
    a false negative are reachable.

    WHY THIS IS SUFFICIENT UNDER WAL. `BEGIN DEFERRED` opens no transaction
    immediately; the first read statement takes a snapshot of the WAL as of that
    moment (the reader records the wal-index end mark) and every later read in
    the same transaction is served from that same mark, regardless of what any
    writer commits meanwhile. WAL readers never block writers and are never
    invalidated by them, so the coordinator continues at full speed and the
    sample is still one instant. COMMIT ends the read transaction and releases
    the mark — required, or the sampler would serve every later sample from the
    first snapshot it ever took.

    DEFERRED specifically: `BEGIN IMMEDIATE` would try to take a write lock,
    which a `mode=ro` connection cannot have.
    """
    conn.execute("BEGIN DEFERRED")
    try:
        yield conn
    finally:
        conn.execute("COMMIT")


def assert_is_ledger(path: str) -> None:
    with connect_ro(path) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='stripes'"
        ).fetchone()
    if row is None:
        raise SystemExit(f"REFUSED: {path} has no `stripes` table — this is not "
                         f"the miner ledger.")


def discover_run_id(conn: sqlite3.Connection, since_epoch: float) -> Optional[str]:
    """The run under observation: the earliest run whose stripe rows were created
    AFTER the sampler started.

    Scoping matters — the ledger accumulates runs, and an unscoped
    `count(*) FROM stripes` sums this trial with every previous one. Latching on
    `created_at >= sampler start` also proves the sampler was running before the
    first stripe existed, which is the evidence the 2026-08-09 attempt lacked.
    """
    row = conn.execute(
        """SELECT run_id, MIN(created_at) AS first_row
             FROM stripes
            WHERE created_at >= ?
         GROUP BY run_id
         ORDER BY first_row ASC
            LIMIT 1""",
        (since_epoch,),
    ).fetchone()
    return row["run_id"] if row else None


def sample_run(conn: sqlite3.Connection, run_id: str) -> Dict[str, Any]:
    """One observation of the post-F1 state model, scoped to `run_id`.

    Both reads are inside ONE `read_snapshot`, so the occupancy set and the
    counts describe the same instant. See `read_snapshot` for why that is
    load-bearing rather than tidy.
    """
    with read_snapshot(conn):
        # COMPUTE-ACTIVE OCCUPANCY. Mirrors the production authority
        # `MinerLedger.compute_busy_worker_ids`: state='claimed' only, staging
        # deliberately excluded because StripeComplete already freed the slot.
        active_rows = conn.execute(
            """SELECT DISTINCT claimed_by FROM stripes
                WHERE run_id=? AND state=? AND claimed_by IS NOT NULL""",
            (run_id, ST_CLAIMED),
        ).fetchall()
        active_workers: Set[str] = {r["claimed_by"] for r in active_rows}

        counts = {r["state"]: r["n"] for r in conn.execute(
            "SELECT state, COUNT(*) AS n FROM stripes WHERE run_id=? GROUP BY state",
            (run_id,),
        ).fetchall()}

    return {
        "obs_status": OBS_OBSERVED,
        "obs_reason": None,
        "compute_active": len(active_workers),
        "active_workers": active_workers,
        "queued_pending": counts.get(ST_PENDING, 0),
        "claimed_rows": counts.get(ST_CLAIMED, 0),
        "staging": counts.get(ST_STAGING, 0),
        "done": counts.get(ST_DONE, 0),
        "cancelled": counts.get(ST_CANCELLED, 0),
        "failed": counts.get(ST_FAILED, 0),
    }


def unobserved_row(epoch: float, ts_iso: str, reason: str) -> Dict[str, Any]:
    """A sample whose ledger read did not happen.

    EVERY ledger quantity is None, not 0 (Beta R2 §1). The pre-R2 loop
    pre-seeded the row with zeros and let a failed read fall through, so a
    locked or unreadable ledger was written into the evidence file as a definite
    `compute_active=0, queued_pending=0` observation AND appended to `samples` —
    the comment above the handler claimed the opposite. That is the GPU probe's
    own defect, reproduced in the tool measuring the same run, and on the term
    the criterion is actually made of rather than on a context field.
    """
    row = {"epoch": epoch, "ts_iso": ts_iso,
           "obs_status": OBS_UNOBSERVED, "obs_reason": reason,
           "active_workers": set()}
    row.update({f: None for f in LEDGER_FIELDS})
    return row


def is_observed(sample: Dict[str, Any]) -> bool:
    """Absence of the marker means observed: a sample dict built before R2 (and
    every fixture that predates it) is a real observation."""
    return sample.get("obs_status", OBS_OBSERVED) == OBS_OBSERVED


def _estab_unavailable(reason: str, detail: bytes = b"") -> Dict[str, Any]:
    text = detail.decode(errors="replace").strip() if detail else ""
    if text:
        reason = f"{reason}:{text.splitlines()[0][:60]}"
    return {"estab": None, "estab_status": ESTAB_UNAVAILABLE,
            "estab_reason": reason}


def estab_observation(port: int) -> Dict[str, Any]:
    """Established connections to the coordinator port. CONTEXT ONLY — a
    connected worker is not an occupied worker, and conflating the two is how
    'the fleet was saturated' gets claimed without evidence.

    THREE OUTCOMES, NEVER TWO (Beta R1 §4). The previous version returned the
    line count of an empty stdout, so `ss` exiting non-zero — no `ss` on a
    minimal PATH, a filter the local iproute2 rejects — was recorded as a
    definite `0` established connections. That is the identical defect the GPU
    probe was certified for fixing, reproduced inside the evidence tool for the
    same run. Here: unavailable, non-zero exit, timeout and unparseable output
    all yield `estab=None` / status UNAVAILABLE with a reason; only a genuine
    successful observation of nothing yields 0.

    ESTAB is not part of the saturation criterion and is not an input to either
    verdict — `evaluate()` cannot see it.
    """
    try:
        out = subprocess.run(
            ["ss", "-tnH", "state", "established", f"( sport = :{port} )"],
            capture_output=True, timeout=ESTAB_TIMEOUT_SECONDS)
    except FileNotFoundError:
        return _estab_unavailable("ss_not_found")
    except subprocess.TimeoutExpired:
        return _estab_unavailable("timeout")
    except Exception as e:                       # noqa: BLE001 — never a count
        return _estab_unavailable(f"{type(e).__name__}")

    if out.returncode != 0:
        return _estab_unavailable(f"ss_exit_{out.returncode}", out.stderr)

    try:
        text = out.stdout.decode()
    except Exception:
        return _estab_unavailable("undecodable_output")

    lines = [l for l in text.splitlines() if l.strip()]
    # `ss -tnH` emits one connection per line, Recv-Q/Send-Q/local/peer. Fewer
    # than four fields means this is not the output being parsed, and a count
    # taken off it would be fiction.
    for line in lines:
        if len(line.split()) < 4:
            return _estab_unavailable("unparseable_ss_output",
                                      line.encode(errors="replace"))
    return {"estab": len(lines), "estab_status": ESTAB_OK, "estab_reason": None}


def render_estab(value: Optional[int]) -> str:
    """An unobservable ESTAB renders as UNAVAILABLE in the TSV, never as 0."""
    return ESTAB_UNAVAILABLE if value is None else str(value)


def render_ledger_value(value: Optional[int]) -> str:
    """An unobserved ledger quantity renders as UNOBSERVED in the TSV, never as
    0. An analyst scanning the column must not be able to read a gap as an idle
    fleet."""
    return OBS_UNOBSERVED if value is None else str(value)


def format_tsv_row(row: Dict[str, Any], run_id: Optional[str],
                   satisfies: Optional[bool]) -> str:
    # `satisfies` is None for an unobserved sample and renders as '-': the
    # criterion was not evaluated. A `0` there would claim it was evaluated and
    # failed, which is the same lie one column to the left.
    return "\t".join(str(x) for x in [
        row["ts_iso"], f"{row['epoch']:.3f}", run_id or "-",
        row.get("obs_status", OBS_OBSERVED), row.get("obs_reason") or "-",
        *[render_ledger_value(row.get(f)) for f in LEDGER_FIELDS],
        render_estab(row.get("estab")), row.get("estab_reason") or "-",
        "-" if satisfies is None else int(satisfies),
    ]) + "\n"


def summarize_estab(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate ESTAB for the summary. Deliberately NOT part of `evaluate()`:
    keeping it out of the verdict function is what makes 'ESTAB cannot move the
    verdict' a structural property rather than a promise."""
    seen = [s["estab"] for s in samples if s.get("estab") is not None]
    missing = [s for s in samples if s.get("estab") is None]
    reasons = sorted({(s.get("estab_reason") or "unspecified").split(":")[0]
                      for s in missing})
    return {
        "observed_samples": len(seen),
        "unavailable_samples": len(missing),
        "max": max(seen) if seen else None,
        "min": min(seen) if seen else None,
        "reasons": reasons,
    }


# ──────────────────────────────────────────────────────────────────────────────
# verdict
# ──────────────────────────────────────────────────────────────────────────────

def _turnover(window: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """VERDICT 2 — was queued work actually CONSUMED during the qualifying
    window (Beta R1 §5)?

    Simultaneity proves the queue was non-empty, not that it moved. A run
    holding 25 claimed and 7 pending statically for the whole window satisfies
    verdict 1 and demonstrates no scheduler turnover, no completion, no
    reassignment, no staging and no back-pressure — which is the entire reason
    32 stripes were chosen over the 25-stripe minimum.

    THE TURNOVER WINDOW IS THE QUALIFYING SIMULTANEITY WINDOW — the single
    longest run of consecutive satisfying samples, the same interval verdict 1
    is decided on. It is NOT the whole run. Measuring over the run would let a
    drain that happened while the fleet was half-idle count as turnover under
    full occupancy, which is the opposite of the claim.

    Two independent pieces of evidence, either sufficient:
      * `pending` DRAINED across steps inside the window — the backlog moved;
      * stripes TRANSITIONED into done/staging during the window — counted as
        the sum of positive step-deltas of (done + staging), so a stripe moving
        staging -> done is not counted a second time.
    Both are summed STEP-WISE over consecutive pairs inside the window, and
    every sample in that window is at or above the occupancy threshold by
    construction — so each counted step is bracketed by two at-threshold
    samples. That is what "while occupancy remained at the threshold" means
    here, and it is why a run-wide monotonic `pending` decrease does not qualify
    on its own.

    `done_delta` is reported alongside because it is the completion count on its
    own, unmixed with staging.
    """
    if not window:
        return {"turnover_satisfied": False,
                "turnover_reason": "no qualifying window to measure turnover in",
                "turnover_pending_delta": None,
                "turnover_pending_drained": None,
                "turnover_transitions": None,
                "turnover_done_delta": None,
                "turnover_pending_first": None,
                "turnover_pending_last": None,
                "turnover_window_samples": 0,
                "turnover_window_start": None,
                "turnover_window_end": None,
                "turnover_window_min_active": None}

    pending_first = window[0]["queued_pending"]
    pending_last = window[-1]["queued_pending"]
    pending_delta = pending_first - pending_last          # positive == drained
    done_delta = window[-1].get("done", 0) - window[0].get("done", 0)

    # Both terms are STEP-WISE, over consecutive pairs INSIDE the window.
    # That is what pairs consumption with sustained occupancy (Beta R2 §2.4):
    # every step counted is bracketed by two samples that are themselves at or
    # above the threshold, so no drain that happened while the fleet was empty
    # can be credited. A run-wide monotonic decrease proves neither.
    pending_drained = 0
    transitions = 0
    for a, b in zip(window, window[1:]):
        drop = a["queued_pending"] - b["queued_pending"]
        if drop > 0:
            pending_drained += drop
        step = ((b.get("done", 0) + b.get("staging", 0))
                - (a.get("done", 0) + a.get("staging", 0)))
        if step > 0:
            transitions += step

    satisfied = pending_drained > 0 or transitions > 0
    if satisfied:
        reason = "queued work was consumed under full occupancy"
    else:
        reason = ("occupancy held at the threshold but nothing moved: pending "
                  f"stayed at {pending_last} and no stripe entered done/staging")
    return {"turnover_satisfied": satisfied,
            "turnover_reason": reason,
            "turnover_pending_delta": pending_delta,
            "turnover_pending_drained": pending_drained,
            "turnover_transitions": transitions,
            "turnover_done_delta": done_delta,
            "turnover_pending_first": pending_first,
            "turnover_pending_last": pending_last,
            "turnover_window_samples": len(window),
            "turnover_window_start": window[0]["ts_iso"],
            "turnover_window_end": window[-1]["ts_iso"],
            "turnover_window_min_active": min(s["compute_active"] for s in window)}


def evaluate(samples: List[Dict[str, Any]], threshold: int,
             min_window_samples: int) -> Dict[str, Any]:
    """The two criteria, evaluated per sample and aggregated into windows.

    VERDICT 1, sustained simultaneity: a sample satisfies iff compute_active >=
    threshold AND queued_pending >= 1. Both conjuncts are required: 25 workers
    busy with an EMPTY queue does not show the scheduler under load, and a deep
    queue with 8 busy workers does not show the fleet saturated. A window is
    `min_window_samples` consecutive satisfying samples.

    VERDICT 2, turnover under full occupancy: see `_turnover`. It is computed
    over the qualifying window and reported SEPARATELY — the two are never
    collapsed into one pass/fail, because a run can satisfy the first and fail
    the second and that distinction is the point.

    UNOBSERVED SAMPLES (Beta R2 §1). A sample whose ledger read failed is not
    evidence of anything. It is:
      * excluded from the criterion — it is neither satisfying nor
        non-satisfying, it is unevaluated;
      * excluded from peak occupancy and from the worker union;
      * counted, so the verdict has a KNOWN denominator;
      * and it BREAKS a window rather than being spanned by one. See below.

    THE GAP RULE, and why this one. Beta allowed either breaking the window or
    annotating the verdict. This implementation BREAKS. A window is a claim of
    SUSTAINED simultaneity, and sustained is precisely the property an unknown
    interior instant destroys: across a gap the fleet may have emptied and
    refilled, and nothing in the evidence file can distinguish that from
    continuity. Breaking makes the claim true by construction rather than true
    by assumption, and it fails closed — the worst a gap can do is understate.
    Annotation would leave a reader to decide whether to believe a window that
    the tool itself could not vouch for. Both are still reported: the gap count
    is in the summary either way, so nothing is hidden by the choice.

    ESTAB is not an input here, by construction.
    """
    scored = []
    for s in samples:
        if not is_observed(s):
            scored.append(dict(s, satisfies=None))
            continue
        ok = (s["compute_active"] >= threshold and s["queued_pending"] >= 1)
        scored.append(dict(s, satisfies=ok))

    observed = [s for s in scored if is_observed(s)]
    unobserved = [s for s in scored if not is_observed(s)]

    windows = []
    current = []
    for s in scored:
        # `satisfies is True` — not truthiness. An unobserved sample carries
        # None and must close the run rather than extend it.
        if s["satisfies"] is True:
            current.append(s)
        elif current:
            windows.append(current)
            current = []
    if current:
        windows.append(current)

    best = max(windows, key=lambda w: (len(w), w[-1]["epoch"] - w[0]["epoch"]),
               default=None)

    # Peak SIMULTANEOUS occupancy, and the queue depth at that same instant.
    peak = max(observed, key=lambda s: s["compute_active"], default=None)

    # The union across instants — computed so it can be reported as NOT
    # qualifying, never as evidence.
    union: Set[str] = set()
    for s in observed:
        union |= s.get("active_workers", set())

    satisfied = best is not None and len(best) >= min_window_samples

    qualifying_windows = [w for w in windows if len(w) >= min_window_samples]
    windows_detail = [{
        "samples": len(w),
        "seconds": w[-1]["epoch"] - w[0]["epoch"],
        "start": w[0]["ts_iso"], "end": w[-1]["ts_iso"],
        "min_active": min(x["compute_active"] for x in w),
        "min_queued": min(x["queued_pending"] for x in w),
    } for w in qualifying_windows]

    # Turnover is only meaningful inside a window that actually qualifies.
    qualifying = best if satisfied else None

    return {
        "satisfied": satisfied,
        **_turnover(qualifying),
        "threshold": threshold,
        "min_window_samples": min_window_samples,
        "samples_total": len(scored),
        "samples_observed": len(observed),
        "samples_unobserved": len(unobserved),
        "unobserved_reasons": sorted({(s.get("obs_reason") or "unspecified").split(":")[0]
                                      for s in unobserved}),
        "samples_satisfying": sum(1 for s in scored if s["satisfies"] is True),
        "window_count": len(windows),
        "qualifying_window_count": len(qualifying_windows),
        "windows_detail": windows_detail,
        "longest_window_samples": len(best) if best else 0,
        "longest_window_seconds": (best[-1]["epoch"] - best[0]["epoch"]) if best else 0.0,
        "longest_window_start": best[0]["ts_iso"] if best else None,
        "longest_window_end": best[-1]["ts_iso"] if best else None,
        "longest_window_min_active": min(s["compute_active"] for s in best) if best else None,
        "longest_window_min_queued": min(s["queued_pending"] for s in best) if best else None,
        "peak_simultaneous_active": peak["compute_active"] if peak else 0,
        "peak_at": peak["ts_iso"] if peak else None,
        "queued_at_peak": peak["queued_pending"] if peak else None,
        "distinct_workers_union": len(union),
    }


LABEL_SIMULTANEITY = "VERDICT 1 — SUSTAINED SIMULTANEITY"
LABEL_TURNOVER = "VERDICT 2 — TURNOVER UNDER FULL OCCUPANCY"
_LABEL_WIDTH = 42


def exit_code(v: Dict[str, Any]) -> int:
    """The two verdicts stay distinguishable even in the exit status. A single
    boolean would collapse exactly the distinction Beta requires be kept."""
    if not v["satisfied"]:
        return 2
    return 0 if v["turnover_satisfied"] else 3


EXIT_CODE_LEGEND = (
    "0 = both criteria satisfied · "
    "2 = criterion 1 (simultaneity) NOT satisfied · "
    "3 = criterion 1 satisfied, criterion 2 (turnover) NOT satisfied"
)


def render_summary(v: Dict[str, Any], run_id: Optional[str],
                   started_iso: str, ended_iso: str, ledger: str,
                   estab: Optional[Dict[str, Any]] = None,
                   interval: Optional[float] = None) -> str:
    """The evidence file must stand alone. Beta reads this WITHOUT the report
    beside it, so every number it prints is accompanied by the predicate that
    produced it, the interval it was measured over, and what was not measured.
    """
    verdict = "SATISFIED" if v["satisfied"] else "NOT SATISFIED"
    turnover = "SATISFIED" if v["turnover_satisfied"] else "NOT SATISFIED"
    t = v["threshold"]
    lines = [
        "=" * 74,
        "GATE-12 CONCURRENCY VERDICT — Beta's saturation criteria",
        "=" * 74,
        f"run_id            : {run_id or '<none observed>'}",
        f"ledger            : {ledger}",
        f"sampling          : {started_iso} -> {ended_iso}",
        f"sample interval   : {'unrecorded' if interval is None else f'{interval:g}s'}",
        f"occupancy threshold : {t} distinct compute-active workers",
        f"window minimum    : {v['min_window_samples']} consecutive satisfying samples",
        "",
        "-- sample census (the verdict's denominator) --",
        f"samples emitted   : {v['samples_total']}",
        f"  OBSERVED        : {v['samples_observed']}   (ledger read succeeded)",
        f"  UNOBSERVED      : {v['samples_unobserved']}   (ledger read FAILED — "
        f"not evidence of an idle fleet)"
        + (f"  reasons: {', '.join(v['unobserved_reasons'])}"
           if v["unobserved_reasons"] else ""),
        "  An UNOBSERVED sample is excluded from both criteria and BREAKS any",
        "  window it falls inside: a window is a claim of SUSTAINED occupancy,",
        "  and an unknown interior instant is exactly what makes 'sustained'",
        "  unprovable. Gaps can only understate this verdict, never inflate it.",
        "",
        "EXACT PREDICATES — this file is self-describing; nothing below depends",
        "on a companion document.",
        "",
        f"CRITERION 1 (sustained simultaneity), per sample:",
        f"      satisfies  <=>  compute_active >= {t}  AND  queued_pending >= 1",
        f"    VERDICT 1 satisfied  <=>  there exists a run of >= "
        f"{v['min_window_samples']} CONSECUTIVE",
        "      satisfying OBSERVED samples (that run is 'the qualifying window').",
        "      compute_active = COUNT(DISTINCT claimed_by) WHERE state='claimed';",
        "      staging is deliberately excluded — StripeComplete has already",
        "      released that worker's compute slot, so counting it overstates.",
        "",
        "CRITERION 2 (turnover under full occupancy), measured over THE",
        "  QUALIFYING SIMULTANEITY WINDOW — the same interval criterion 1 was",
        "  decided on, NOT the whole run:",
        "      pending_drained = SUM over consecutive pairs (a,b) in the window",
        "                        of max(0, a.pending - b.pending)",
        "      transitions     = SUM over the same pairs",
        "                        of max(0, (b.done+b.staging) - (a.done+a.staging))",
        "    VERDICT 2 satisfied  <=>  pending_drained > 0  OR  transitions > 0",
        "      Every counted step is bracketed by two samples that are themselves",
        "      at or above the threshold, so consumption is paired with sustained",
        "      occupancy across the SAME samples. A run-wide monotonic decrease",
        "      does not qualify on its own.",
        "",
        "THE TWO ARE SEPARATE AND ARE NOT COLLAPSED. Criterion 1 proves the queue",
        "was non-empty; only criterion 2 proves it was consumed.",
        "",
        f"{LABEL_SIMULTANEITY:<{_LABEL_WIDTH}}: {verdict}",
        f"{LABEL_TURNOVER:<{_LABEL_WIDTH}}: {turnover}",
        f"{'EXIT CODE':<{_LABEL_WIDTH}}: {exit_code(v)}",
        f"  {EXIT_CODE_LEGEND}",
        "",
        "-- verdict 1 evidence: simultaneity --",
        f"peak simultaneous compute-active workers : {v['peak_simultaneous_active']}",
        f"  observed at                            : {v['peak_at']}",
        f"  queued (pending) at that same instant  : {v['queued_at_peak']}",
        f"satisfying samples                       : {v['samples_satisfying']}/"
        f"{v['samples_observed']} observed",
        f"runs of satisfying samples               : {v['window_count']}",
        f"  of those, QUALIFYING (>= {v['min_window_samples']} samples)         : "
        f"{v['qualifying_window_count']}",
    ]
    if v["windows_detail"]:
        for i, w in enumerate(v["windows_detail"], 1):
            lines.append(
                f"    window {i}: {w['samples']} samples, {w['seconds']:.1f}s, "
                f"{w['start']} -> {w['end']}, "
                f"min_active={w['min_active']} min_queued={w['min_queued']}")
    else:
        lines.append("    (none)")
    lines += [
        f"longest window                           : {v['longest_window_samples']} samples, "
        f"{v['longest_window_seconds']:.1f}s",
        f"  from / to                              : {v['longest_window_start']} -> {v['longest_window_end']}",
        f"  min compute-active within window       : {v['longest_window_min_active']}",
        f"  min queued within window               : {v['longest_window_min_queued']}",
        "",
        "-- verdict 2 evidence: turnover WITHIN the qualifying window --",
        f"turnover window                          : "
        f"{v['turnover_window_samples']} samples, "
        f"{v['turnover_window_start']} -> {v['turnover_window_end']}",
        f"  (this is the qualifying simultaneity window, not the whole run)",
        f"  min compute-active across it           : {v['turnover_window_min_active']}",
        f"pending at window start / end            : "
        f"{v['turnover_pending_first']} -> {v['turnover_pending_last']}",
        f"  pending DRAINED step-wise in window    : {v['turnover_pending_drained']}",
        f"  endpoint delta (context only)          : {v['turnover_pending_delta']}",
        f"stripes transitioned into done/staging   : {v['turnover_transitions']}",
        f"  of which reached done                  : {v['turnover_done_delta']}",
        f"finding                                  : {v['turnover_reason']}",
        "",
        "-- NOT evidence of saturation (recorded so it is not mistaken for it) --",
        f"distinct workers ever seen active (union across instants) : "
        f"{v['distinct_workers_union']}",
        "  A union over time is not simultaneity: 25 workers running strictly one",
        "  after another produce the same number as 25 running together. This",
        "  figure CANNOT satisfy the criterion and is printed only for context.",
    ]

    # ESTAB: context only, and honest about not knowing.
    lines += ["", "-- ESTAB (context only — NOT a term in either criterion) --"]
    if estab is None:
        lines.append("established connections to the coordinator port : not aggregated")
    else:
        if estab["max"] is None:
            lines.append(f"established connections            : {ESTAB_UNAVAILABLE} "
                         f"— never successfully observed")
        else:
            # "samples where ss succeeded", NOT "observed samples": R2 gives
            # OBSERVED/UNOBSERVED a specific ledger meaning elsewhere in this
            # same file, and a reader with no report beside them must not have
            # to guess which vocabulary a word belongs to. Display only.
            lines.append(f"established connections            : max={estab['max']} "
                         f"min={estab['min']} over {estab['observed_samples']} "
                         f"sample(s) where ss succeeded")
        lines.append(f"samples where ss was {ESTAB_UNAVAILABLE:<13} : "
                     f"{estab['unavailable_samples']}"
                     + (f"  (reasons: {', '.join(estab['reasons'])})"
                        if estab["reasons"] else ""))
        lines.append("  An unobservable ss is recorded UNAVAILABLE, never as 0 — a")
        lines.append("  connection count nobody could take is not a count of zero.")
    lines += [
        "  A connected worker is not an occupied worker. ESTAB is not an input to",
        "  either verdict above and cannot change one.",
        "=" * 74,
    ]

    if not v["satisfied"]:
        reason = []
        if v["peak_simultaneous_active"] < v["threshold"]:
            reason.append(
                f"peak simultaneous occupancy was {v['peak_simultaneous_active']}, "
                f"below the required {v['threshold']}")
        elif v["samples_satisfying"] == 0:
            reason.append(
                "occupancy reached the threshold but the queue was empty at every "
                "such instant — the fleet was full with nothing left queued, which "
                "does not demonstrate the scheduler under load")
        else:
            reason.append(
                f"the longest satisfying window was {v['longest_window_samples']} "
                f"sample(s), short of the required {v['min_window_samples']}")
        lines += ["WHY NOT (verdict 1): " + "; ".join(reason), "=" * 74]
    if v["satisfied"] and not v["turnover_satisfied"]:
        lines += [
            "WHY NOT (verdict 2): " + str(v["turnover_reason"]),
            "  The fleet was full and the queue was non-empty, but the queue did",
            "  not move. Seven stripes held behind a saturated fleet that never",
            "  drains exercises no scheduler turnover, no completion path, no",
            "  reassignment and no staging back-pressure — which is what the",
            "  32-stripe geometry exists to demonstrate.",
            "=" * 74,
        ]
    return "\n".join(lines) + "\n"


# ──────────────────────────────────────────────────────────────────────────────
# main loop
# ──────────────────────────────────────────────────────────────────────────────

def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError) as e:
        return isinstance(e, PermissionError)
    except Exception:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default=None,
                    help="miner ledger path (default: derived from the manifest)")
    ap.add_argument("--out", required=True, help="per-sample TSV")
    ap.add_argument("--summary", required=True, help="verdict text file")
    ap.add_argument("--interval", type=float, default=2.0,
                    help="seconds between samples (default 2)")
    ap.add_argument("--threshold", type=int, default=25,
                    help="distinct simultaneous compute-active workers required")
    ap.add_argument("--min-window-samples", type=int, default=2,
                    help="consecutive satisfying samples that constitute a window")
    ap.add_argument("--port", type=int, default=5700, help="coordinator port")
    ap.add_argument("--watch-pid", type=int, default=None,
                    help="stop when this pid exits (the run's own process)")
    ap.add_argument("--run-id", default=None,
                    help="explicit run_id; default is auto-latch on the first "
                         "run created after sampling starts")
    ap.add_argument("--quiesce-seconds", type=float, default=180.0,
                    help="after latch, stop if no runnable stripe remains this long")
    ap.add_argument("--max-seconds", type=float, default=7200.0,
                    help="hard cap so the loop can never outlive the day")
    args = ap.parse_args(argv)

    ledger = guard_db_path(args.ledger or default_ledger_path())
    assert_is_ledger(ledger)

    start = time.time()
    started_iso = datetime.now().isoformat(timespec="seconds")
    run_id: Optional[str] = args.run_id
    samples: List[Dict[str, Any]] = []
    quiet_since: Optional[float] = None
    stop = {"flag": False}

    def _stop(signum, frame):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out = open(args.out, "w", buffering=1)
    out.write("\t".join(TSV_COLUMNS) + "\n")
    print(f"[sampler] ledger={ledger}")
    print(f"[sampler] armed at {started_iso} — BEFORE any StripeAssign can be issued")

    try:
        while not stop["flag"]:
            now = time.time()
            if now - start > args.max_seconds:
                print("[sampler] max-seconds reached")
                break

            ts_iso = datetime.now().isoformat(timespec="seconds")
            # Born UNOBSERVED. The row only becomes an observation if a ledger
            # read actually succeeds — the pre-R2 loop pre-seeded zeros and let
            # a failure fall through as `compute_active=0`, which is the defect
            # this ordering makes structurally impossible.
            row = unobserved_row(now, ts_iso, "not_yet_read")
            try:
                with connect_ro(ledger) as conn:
                    if run_id is None:
                        run_id = discover_run_id(conn, start)
                        if run_id:
                            print(f"[sampler] latched run_id={run_id}")
                    if run_id:
                        row.update(sample_run(conn, run_id))
                    else:
                        # No run to sample yet. That is a KNOWN state — the
                        # ledger was read and holds no run of ours — not a
                        # failed read, so it is a genuine observation of zero.
                        row.update({f: 0 for f in LEDGER_FIELDS})
                        row["obs_status"] = OBS_OBSERVED
                        row["obs_reason"] = None
            except Exception as e:
                # ANY failure to read the ledger — sqlite3.Error, a vanished
                # file, a permissions change — is UNOBSERVED, never zero
                # occupancy. Catching broadly is deliberate: an escaping
                # exception would kill the loop and the summary with it.
                row = unobserved_row(now, ts_iso, f"{type(e).__name__}:{e}")
                print(f"[sampler] ledger read UNOBSERVED: {type(e).__name__}: {e}")

            row.update(estab_observation(args.port))
            observed = is_observed(row)
            satisfies = None if not observed else (
                row["compute_active"] >= args.threshold
                and row["queued_pending"] >= 1)
            out.write(format_tsv_row(row, run_id, satisfies))
            if run_id:
                samples.append(row)

            # --- termination: the run, not a wall clock ---
            if args.watch_pid is not None and not pid_alive(args.watch_pid):
                print(f"[sampler] watched pid {args.watch_pid} exited — "
                      f"taking a final sample and stopping")
                stop["flag"] = True
            # Quiescence is decided ONLY on observed samples. An unobserved one
            # summed to `0 + 0 + 0` under the pre-R2 shape, so a spell of failed
            # ledger reads started the quiesce timer and could stop the sampler
            # with "run is over" while the run was very much alive — the same
            # fall-through, ending the observation rather than merely corrupting
            # it. It neither starts nor clears the timer now.
            if run_id and is_observed(row):
                runnable = row["queued_pending"] + row["claimed_rows"] + row["staging"]
                if runnable == 0:
                    quiet_since = quiet_since or now
                    if now - quiet_since >= args.quiesce_seconds:
                        print("[sampler] no runnable stripe for "
                              f"{args.quiesce_seconds:.0f}s — run is over")
                        stop["flag"] = True
                else:
                    quiet_since = None

            if not stop["flag"]:
                time.sleep(args.interval)
    finally:
        out.close()

    verdict = evaluate(samples, args.threshold, args.min_window_samples)
    text = render_summary(verdict, run_id, started_iso,
                          datetime.now().isoformat(timespec="seconds"), ledger,
                          estab=summarize_estab(samples),
                          interval=args.interval)
    Path(args.summary).write_text(text)
    print(text)
    return exit_code(verdict)


if __name__ == "__main__":
    sys.exit(main())
