# Claude Code Brief — S176 TB Ruling Follow-Up (Items 1–3) — v1

**Runs on:** VM101, as `michael`, from `/home/michael/distributed_prng_analysis`.
**Context:** Team Beta has ruled on the S176 WATCHER-KPI findings (ruling doc:
`docs/TB_RULING_S176_WATCHER_KPI.md` — Michael will place it; read it first).
This brief implements the three work items the ruling unblocks. The cluster-heavy
Phase C walk-forward is explicitly OUT of scope — it queues behind S172 Phase 5–7
on the RANGE-MINER path.

---

## Hard rules (unchanged)

- NO `git commit` / `git push`. Deliver files; Michael commits + dual-pushes.
- NO `watcher_agent.py --run-pipeline`. NO cluster runs of any kind.
- NO changes to `watcher_policies.json` or any live policy/config value.
  Item 3 produces a PROPOSAL document + example schema, not an edit.
- Read the exact source before every claim; cite file:line. Where the TB ruling
  and the working tree disagree, report the disagreement — do not paper over it.
- Every deliverable is recommend-only pending TB review of the outputs.

---

## Item 1 — The two verifications TB conditioned on (do FIRST; blocking)

### 1a. Zeus/VM101 working tree vs GitHub `main` @ `0c3166a` (D1/D2 disposition)

TB ruled D1/D2 "not confirmed as current defects on `main`" because live `main`
uses manifest-defined primary outputs. S176 compared against a step-file map
that may be stale. Your job: determine whether the working tree this box runs
differs from `main` in the relevant files.

```bash
cd /home/michael/distributed_prng_analysis
git fetch origin && git rev-parse HEAD && git status --porcelain
git diff 0c3166a630be321809f415bb28af28e319d0fe1b -- \
    agents/ chapter_13_diagnostics.py chapter_13_triggers.py \
    adaptive_meta_optimizer.py run_step3_full_scoring.sh \
    '*manifest*' watcher_policies.json | head -200
```

Then read the actual manifest(s) WATCHER loads at runtime (trace
`manifest.get("primary_output")` to the file it reads) and confirm what Step 3
and Step 4 declare **on this tree**.

**Deliverable 1a:** a short verdict per defect:
- `D1/D2 RESOLVED` (working tree matches main's manifest wiring; S176 compared
  a stale expectation) → recommend removing both from the defect list, OR
- `D1/D2 STANDS` (working tree differs from main; show the diff hunks), OR
- `MIXED` (state which and why).
Cite the exact manifest file(s) + lines. If the S176 step-file map artifact that
produced the original claim still exists, name it so the stale source is on
record.

### 1b. Final `minimum_hit_rate` consumer search on the exact working tree

TB: deprecation is approved only after a definitive repo-wide search here.

```bash
grep -rn "minimum_hit_rate" . \
    --include='*.py' --include='*.json' --include='*.sh' \
    --include='*.md' --include='*.gbnf' 2>/dev/null | grep -v '.git/'
```

Classify every hit: RUNTIME CONSUMER / CONFIG DEFINITION / DOCS-ONLY /
TEST-ONLY. **Deliverable 1b:** the classified table + a one-line verdict:
"zero runtime consumers confirmed on tree <HEAD sha>" or the consumer found.

---

## Item 2 — Revise the deterministic analyzer per TB §8

Revise `watcher_kpi_metricC_deterministic.py` → save as
`watcher_kpi_metricC_deterministic_v2.py` (keep v1 untouched for the audit
trail). Apply ALL twelve §8 points:

1. Validate `pool_size > 0`; 2. validate `draw_space > 0`; 3. validate
`max_misses >= 1`; 4. no hardcoded `window_UNUSED=20`; 5. no hardcoded
`minimum_hit_rate=0.05`; 6. read policy values from `--policies` (or require
explicit args — fail loudly if neither given, no silent defaults);
7. rename `chance_hit_probability` → `uniform_null_hit_probability`;
8. keep `assumed_healthy_hit_rate` strictly separate from the null;
9. assert/report the unique-pool-size assumption; 10. replace
`mean_gap > max_misses` with a defined false-alarm-horizon / expected-waiting-
time test (`--false-alarm-horizon`, default explicit and documented);
11. state in output that only TWO live triggers consume metric C and
`minimum_hit_rate` is a configured target, not a live trigger; 12. describe
Metric-A/Metric-C as complementary views of the same Bernoulli event, not
independent evidence.

Also adopt TB's wording discipline everywhere in the output: findings are
stated **at the uniform random null**, never as "healthy TFM" claims.

**Deliverable 2:** the v2 tool + a run of it against the real
`watcher_policies.json` values, output saved to
`watcher_kpi_metricC_v2_findings.json`, + a 5-line diff-summary of v1→v2.

---

## Item 3 — Draft the BOOTSTRAP-state proposal (document only)

Draft `docs/PROPOSAL_WATCHER_KPI_GOVERNANCE_STATES_v1_0.md` implementing TB's
BOOTSTRAP → CALIBRATING → GOVERNED architecture as a scoped change proposal for
TB code review. It must contain:

1. **The state model** — the three states, entry/exit criteria per TB Phase
   A–E, and TB's rule that triggers may transition individually.
2. **The `kpi_governance` schema** — start from TB's Q3 example (hit20/100/300
   blocks: `null_rate`, `empirical_baseline: null`, `minimum_samples: null`,
   `collapse_threshold: null`, `enforcement: "audit_only"`), extend with the
   pool-structure metric class TB listed in Q1 (weight shares, unique count,
   breadth, entropy, effective pool size, duplicates, stability) — kept as a
   SEPARATE class from Hit@K per TB's explicit separation requirement.
3. **Audit-only trigger mode** — exactly which code paths change so
   uncalibrated performance triggers RECORD hypothetical decisions (shadow log)
   without dispatching retraining; which structural/catastrophic gates stay
   ACTIVE (use TB §4.2's list, mapped to the checks WATCHER already performs —
   cite where each existing check lives).
4. **KPI recording plumbing** — what gets persisted per draw (TB Phase B/C
   lists), file/format, and where in the diagnostics path it hooks (file:line).
5. **`minimum_hit_rate` disposition** — conditional on Deliverable 1b:
   deprecation path + replacement by the schema above.
6. **Metric-name uniqueness audit** (TB Phase A item 6) — enumerate every
   "hit rate"-family name in the tree (`current_hit_rate`, `hit_K_rate`,
   `exact_hits`, `pool_coverage`, `minimum_hit_rate`, advisor-band names) and
   state each one's single definition + owner file. Flag any name with two
   meanings.
7. **Explicit non-scope** — Phase C walk-forward deferred to post-S172-Phase-7
   on the RANGE-MINER path (state the GCVM launch-storm rationale in one
   sentence); no thresholds selected; no autonomous enforcement enabled.

**Deliverable 3:** the proposal doc. Do NOT implement any of it — this is the
document TB reviews before implementation is scoped.

---

## Order, changelog, stop condition

Item 1 first (1a before 1b is fine, both before 2–3 since Item 3 §5 depends on
1b). Then Item 2, then Item 3. Write
`docs/SESSION_CHANGELOG_YYYYMMDD_S<N>.md`. Deliver all files for Michael's
review → TB. **Stop after delivering.** Do not begin Phase A implementation,
Phase C, Track A/Task 3, or any walk-forward work.
