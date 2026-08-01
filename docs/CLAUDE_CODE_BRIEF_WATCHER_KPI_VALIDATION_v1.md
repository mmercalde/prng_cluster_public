# Claude Code Brief — WATCHER KPI Validation & Step-0 Heuristic Validation (v1)

**Runs on:** VM101 (canonical dev box), as `michael`, from project root
`/home/michael/distributed_prng_analysis`.
**Objective:** Establish valid, evidence-backed baselines for the WATCHER
governance KPIs, and validate the Step-0 (TRSE) advisory heuristics on real data.

---

## Hard rules (do not violate)

- **Do NOT `git commit` / `git push`.** Deliver files only; Michael commits and
  dual-pushes. (You are Team Alpha, not Team Beta, not the committer.)
- **Do NOT run `watcher_agent.py --run-pipeline`.** It is out of scope here.
- **Do NOT launch the multi-hour cluster walk-forward yourself.** You prepare and
  dry-run it; Michael `nohup`-launches the full run. (See Task 4.)
- **Do NOT change any value in `watcher_policies.json`.** These tools *recommend*;
  changes route through `THRESHOLD_GOVERNANCE` + Team Beta.
- **Read live source before every claim.** Do not assert a definition, signature,
  or file behavior you have not read. Use `inspect.signature()` before calling any
  production function (the harnesses already do this; keep the discipline).
- **Verify background processes.** If an SSH/subprocess returns only a prompt with
  no output, it did not launch — check with `ps` or the log immediately.

## Files delivered with this brief (place in project root)

- `watcher_kpi_resolvability_probe.py` — analytic (Type-1) KPI resolvability.
- `step0_heuristic_validation.py` — Track A, real-data Step-0 validation.
- `watcher_kpi_baseline.py` — Track B aggregator (consumes backtest output).

All three are CPU-only, read-only (write only their findings file), and already
smoke-tested. Confirm they import and run before use.

---

## Task 0 — Pin the hit-rate definition (BLOCKING; do this first)

Every downstream baseline depends on the operational definition of "hit rate":
per-draw probability, and pool size **K**. Extract it from source — do not assume.

```bash
grep -rn -iE "hit_rate|hit@|def .*hit|pool.*size|top_?k|hits/" \
    evaluate_pools.py build_pools.py watcher_agent.py agents/ modules/ 2>/dev/null | less
```

**Deliverable 0:** a 3–5 line note stating (a) how hit rate is computed, (b) the
production pool size **K**, (c) which `--hit-column` (hit_20 / hit_100 / hit_300)
corresponds to production, and (d) the measured or expected baseline value if
recorded anywhere. This note feeds Tasks 2 and 5.

**Acceptance:** the definition is quoted from a real file with path + line numbers.

---

## Task 1 — Confirm the tools run

```bash
python3 -c "import trse_step0, inspect; print(inspect.signature(trse_step0.detect_offset_periodicity))"
python3 watcher_kpi_resolvability_probe.py --help
python3 step0_heuristic_validation.py --help
python3 watcher_kpi_baseline.py --help
```

**Acceptance:** all import/help cleanly. If `step0_heuristic_validation.py` reports
"trse_step0 API drift," STOP and report the drift — do not edit around it.

---

## Task 2 — Analytic KPI resolvability (CPU, now)

Run with the REAL baseline from Task 0 (not the 0.05 placeholder).

```bash
python3 watcher_kpi_resolvability_probe.py \
    --policies watcher_policies.json \
    --baseline-hit-rate <REAL_FROM_TASK_0> \
    --pool-size <K_FROM_TASK_0> \
    --out watcher_kpi_resolvability_findings.json
```

**Deliverable 2:** the findings JSON + a one-paragraph summary of which KPIs are
RESOLVABLE vs DECORATIVE vs INSUFFICIENT-N at their configured values.

**Acceptance:** verdicts are reported against the real baseline, and the hit-rate
assumption line is included so the verdicts are interpretable.

---

## Task 3 — Track A: real-data Step-0 validation (CPU, now)

```bash
python3 step0_heuristic_validation.py \
    --lottery-data daily3.json --permutations 2000 \
    --out step0_validation_findings.json
```

Then strengthen to out-of-sample (optional but valuable):

1. **Scraper URL check first** (the CA scraper header says to verify before a full
   pull). Do a single small fetch and confirm the page still parses. If it 404s or
   the table structure changed, report and stop — do not blind-scrape.
2. Refresh `daily3.json`, then run Track A on the **newest slice only** (draws that
   postdate when the heuristics were fixed) — that is true out-of-sample.
3. Run the PA scraper to produce `pa_pick3.json`, then run Track A on it — the
   cross-dataset check.

**Deliverable 3:** findings JSON(s) + a note per heuristic: SIGNIFICANT vs
NOT-DISTINGUISHABLE (permutation), STABLE vs DRIFT (time-split), and — if the
out-of-sample steps ran — whether offset/skip/duality replicate on unseen data.

**Acceptance:** results are from real `daily3.json`; any scrape was preceded by a
verified page-parse check.

---

## Task 4 — Prepare the walk-forward (PREP + DRY-RUN only; Michael launches)

The walk-forward "runs the past forward" to manufacture the real hit/miss +
survivor stream. It is the ONE cluster step. **You prepare and validate it; you do
not launch the full run.**

1. Set production-matching params (confirm from source/config — do not hardcode):
   `--prng-type` (TFM sieve targets `java_lcg`; confirm), full/production `--seed`
   range (not the 100k demo default), and the production `--window`.
2. **Minimal functional dry-run only** — 1–2 draws — to prove the subprocess chain
   (`coordinator.py` → `build_pools.py` → `evaluate_pools.py`) is wired and writes
   `backtest_results.json`. If even this spins GPUs beyond a trivial check, pause
   and get Michael's go-ahead first.
3. Write the exact `nohup` command for Michael to launch the pilot
   (~100–200 contiguous recent draws), e.g.:

```bash
nohup python3 backtest_pools.py --dataset daily3.json \
    --start <YYYY-MM-DD> --end <YYYY-MM-DD> --session midday \
    --prng-type java_lcg --seed-start 0 --seed-end <PRODUCTION_END> \
    --window <PROD_WINDOW> --out backtest_results.json \
    > logs/backtest_pilot.log 2>&1 &
```

**Deliverable 4:** the validated launch command + a note confirming the chain
produced a well-formed `backtest_results.json` on the dry-run. **Then hand off.**

**Acceptance:** no full multi-hundred-draw run was started by you; the command is
ready for Michael.

---

## Task 5 — Track B: baseline the KPIs (CPU, AFTER Michael's walk-forward)

Once `backtest_results.json` exists from the pilot:

```bash
python3 watcher_kpi_baseline.py \
    --backtest-results backtest_results.json \
    --hit-column <FROM_TASK_0> \
    --policies watcher_policies.json \
    --out watcher_kpi_baseline_findings.json
```

**Deliverable 5:** the baseline findings JSON: real baseline hit rate + Wilson CI,
survivor distribution, empirical firing rate of each hit/survivor-family KPI on the
real stream, and recommended threshold values. Note that `confidence_drift`,
`window_decay`, and `llm_confidence` are NOT covered here (they need the heavier
full-pipeline-per-draw run) — say so explicitly.

**Acceptance:** baselines derive from the real stream; if n < 100, they are flagged
provisional and a longer walk-forward is recommended before governance.

---

## Task 6 — Assemble the governance findings doc

Draft `WATCHER_KPI_CALIBRATION_FINDINGS_S<N>.md` mirroring the format of
`THRESHOLD_CALIBRATION_FINDINGS_S148.md`. For each KPI: configured value, evidence
(analytic verdict from Task 2 + empirical baseline from Task 5), and a recommended
value with justification. Include the Task-0 hit-rate definition as the stated
assumption. **Recommend only — do not edit `watcher_policies.json`.**

Write `SESSION_CHANGELOG_YYYYMMDD_S<N>.md` for the session.

**Deliverable 6:** the findings MD + changelog, delivered to Michael for Team Beta
review, then commit + dual-push (by Michael).

---

## What this proves (state it plainly in the findings doc)

- **Step-0 heuristics** are real temporal structure vs noise (Track A), and — with
  the scraped/PA steps — whether they hold out-of-sample.
- **WATCHER hit/survivor KPIs** are calibrated to the system's real behavior, not
  leftover guesses; several may be shown decorative by two independent methods
  (analytic probe + real-stream replay converging).
- **Out of scope (do not claim):** that predictions beat chance out-of-sample —
  that is the holdout/prospective question and is deliberately separate.

## Delivery flow (unchanged)

Tools/findings land in the project working dir → Michael downloads to ser8
`~/Downloads/` → scp to VM101 as needed → **Michael** commits and dual-pushes
(`git push origin main && git push public main`) after Team Beta review.
