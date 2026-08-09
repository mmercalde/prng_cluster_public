# CLAUDE CODE REPORT — S172 STAGING-CAPACITY AMENDMENT + `elapsed_s` PERSISTENCE

**Host:** VM101 (`zeus-ubuntu-vm`, `192.168.3.177`) · repo `~/distributed_prng_analysis`
**Base:** `c7058d8` · venv `~/venvs/torch` active for every run
**Status:** implementation complete; **not committed, not pushed, no pipeline/fleet launch, no
port 5700 bind.** Gate 12 and the Phase-7 soak remain HELD.

**Authority implemented:** Beta *"S172 GATE-12 STAGING-CAPACITY DEADLOCK"* (2026-08-07) §§2-6
(Part 1) and Beta *"STEP-1 SEARCH GEOMETRY…"* (2026-08-08) **R4** (Part 2).

---

## 0. Base verification — one deviation, owner-cleared

| check | required | actual | |
|---|---|---|---|
| `git log --oneline -1` | `c7058d8` | `c7058d8` | ✅ |
| `tests/test_s172_staging_backpressure.py` | 35/35 | **35/35**, exit 0, sentinel PASS | ✅ |
| `git status --porcelain` | clean | **6 untracked entries** | ⚠ deviation |

The **tracked tree was clean** (zero modified, zero staged). The six entries were the brief
itself, four live SQLite WAL sidecars (`miner_ledger.db-shm/-wal`, `prng_analysis.db-shm/-wal`)
and a 2026-05-11 runtime rotation (`optimal_window_config.json.stale_1786149572`). **None was a
`.py`.** I stopped and reported; the owner cleared me to proceed.

**Incidental finding (reported, not fixed — outside scope):** four of the six are visible only
because of a `.gitignore` pattern gap. `.gitignore:113` is `*.db`, which covers
`miner_ledger.db`/`prng_analysis.db` but not their `-shm`/`-wal` sidecars; `.gitignore:115` is the
exact string `optimal_window_config.json`, which the `.stale_<epoch>` suffix escapes. **The
project already solved this class once and did not generalise it** — `.gitignore:86-87` explicitly
pairs `optuna_studies/*.db` with `optuna_studies/*.db-journal`.

---

## 1. Per-ruling-section implementation notes

### §1.1 Release lifecycle — Beta Option C

`commit_trial` (`miner/range_miner_coordinator.py:4753`) previously released **nothing**. It now,
**only after the sink's successful commit return** (`:4798`), discharges every trial-owned
reservation in one pass, deletes the staged files, records a durable cleanup status and pumps
deferred work.

- **Reuse, not a second release path.** The discharge loop calls the existing
  **`ack_by_event_id`** (`:4381`, previously **zero production callers** — the brief's own
  observation). Its required semantics hold **by construction, not by inspection**: it is keyed
  solely by the reservation's immutable `event_id`, and it refuses any row not still `held`, so
  exactly-once survives a duplicate commit, a crash between rows and a concurrent abort. It also
  marks the shard acked and locally-deleted rather than merely dropping the reservation. (Owner
  rule 2026-08-06: take the structurally stronger mechanism.)
- **No incremental assembly, no mid-trial ack** (Beta §2.1 refuses both): the discharge is one
  pass after the terminal success.
- **Failure retains.** On a sink exception the method returns early (`:4788`) with
  `released_reservations = 0`; manifests, staged files and reservations are all retained and the
  same `event_id` stays retryable. **This is what preserves D1.1's failed-commit retry contract** —
  releasing on failure would delete the very spools the retry must re-read.
- **Durable cleanup status on a NEW column.** `trials.commit_cleanup_status` (`:873`, setter
  `set_trial_commit_cleanup_status` `:1310`). I deliberately did **not** reuse
  `abort_cleanup_status`: a reviewer reading `abort_cleanup=done` on a committed trial would be
  reading a lie.
- **Observable outcome:** the returned event carries `released_reservations` and
  `staged_files_deleted`, so gates assert the absence of a second release rather than infer it.

### §1.2 Derived retention bound

- **Pure function** `trial_retention_files_required` (`:579`), placed beside the existing burst
  bounds. It implements Beta §3 verbatim — Σ over planned phases, Σ over planned stripes, of
  max-over-eligible-workers expected sub-stripes — by **reusing
  `staging_burst_bound_conservative` per stage**. That function already computes the inner two
  sums and resolves caps through `advertised_effective_cap`, the coordinator's single cap path
  shared with the worker. A second derivation here is exactly the drift Beta named in §2.
- **Coordinator surface:** `trial_retention_requirement` (`:3251`) takes spans from the **real**
  `partition_macro_stripes` output, so a short final stripe counts at its true span;
  `preflight_trial_retention` (`:3318`) resolves and enforces; `effective_high_water_files`
  (`:3280`) is the single resolution accessor and never returns `None`.
- **Fail-closed placement.** The preflight runs in `serve_trial` **above `assign_stripes`**
  (`:5730` is its failure log), latched once per trial. A trial that cannot fit therefore creates
  no stripe rows, claims no stripe, sends **zero StripeAssign** and generates zero result traffic.
  It runs at the first stage because that is the earliest point the eligible set is known
  (admission has completed); the requirement itself is whole-trial, not per-stage, because nothing
  is released before commit.
- **Failure type:** `StagingRetentionSizingError(StagingConfigurationError)` (`:1769`) — permanent
  and non-retryable, so it cannot burn a Q3 retry against a bound that cannot change mid-trial.
- **All three `4096` sites handled** — `CoordinatorConfig` (`:238`), `build_coordinator` and
  `run_trial_miner` — all now `Optional[int] = None` meaning **derive**. **1,028 is not hardcoded
  anywhere.**

**One design point that needs Beta's eye.** `effective_high_water_files` has a third branch for
callers with **no trial plan** (bare-API/gate paths): it returns
`max(_derive_bound_from_current_state(), ledger.total_expected_substripes())` (`:1144`). I first
implemented this branch as **fail-closed**, which is the F5-consistent posture — and it **red 8
existing gates**, which the brief forbids. The two-derivation max is honest (both terms are
derived from live geometry/caps, never a constant) and is **not a production source**, because the
preflight always resolves branch 1 or 2 before any production reservation. I flag it because it is
the one place where "derive" degrades rather than refuses.

### §1.3 Route both high-waters

Both now travel the complete governed path, mirroring the four controls already routed:

| hop | file | change |
|---|---|---|
| 1 | `agent_manifests/window_optimizer.json` | `default_params.staging_high_water_files = null`, `…_bytes = 17179869184`; `actions[0].args_map` gains both kebab flags |
| 2 | `window_optimizer.py` | signature params, `--staging-high-water-files/-bytes` argparse, `coordinator.<key> = …` assignment, call-site forwarding |
| 3 | `window_optimizer_integration_final.py:1467-1478` | reads off the coordinator → `run_trial_miner` → `build_coordinator` → `CoordinatorConfig` |

**The stale literal is gone.** `getattr(coordinator, 'staging_high_water_files', 512)` was the only
production source of the file ceiling **and it never matched the committed dataclass default**
(4096 at `8bbe79e`), so the number production actually took corresponded to no reviewed value at
all. The files default is now `None` (= derive); no `getattr(..., <literal>)` remains the only
production source of either high-water.

**Manifest edit discipline:** the manifest is a gitignored JSON, so I verified the change by
**diffing the decoded structures**, not the text (skill §7 — a whole-file JSON rewrite is the
`2389b61` mechanism). Result: **4 keys added, 0 removed, 0 changed.**

### §1.4 Capacity timeout observes the executor

**The defect, located and confirmed:** `_run_staging_job:4358-4397` is
`while True: … except StagingBackPressure: time.sleep(0.02); continue` — an unbounded spin with no
clock. It pauses no connection, so `staging_capacity_timeout_expired()` read an empty
`_paused_connections` and truthfully answered "nothing is waiting" while the thread spun.

- **New registry** `_staging_reservation_waits`, deliberately under **the same `_pause_lock`** as
  the pause registry, so "oldest blocker across both classes" is one atomic read — two locks could
  race and let each registry look younger than the bound while their true oldest exceeded it.
- `register_staging_reservation_wait` (`:3807`) keeps the **original** entry time on re-register,
  for the same reason the pause registry does: a 20 ms spin loop resetting the clock would make the
  bound unreachable by construction. `clear_staging_reservation_wait` (`:3828`) is called from a
  **`finally`** (`:4438`), so no exit path can leak an ever-ageing phantom blocker.
- `_capacity_blockers_locked` (`:3837`) unifies both classes; `staging_capacity_timeout_expired`
  (`:3862`) now takes the oldest across both. **This widens the observer, not the classification
  law** — the reader-side timeout keeps its exact previous meaning and terminal classification.
- **One episode, fully attributed.** The F3 snapshot now carries `blocker_class`
  (`reader_pause`/`staging_reservation`), the full `trigger` record (run/stripe/attempt/sub-index,
  worker), `blocker_count`, `reserved_files`, `reserved_bytes`, `high_water_files`,
  `high_water_bytes` and `derived_required_files`. The terminal reason names them.
  `_high_water_files_for_report` (`:3931`) cannot raise — a terminal snapshot must never mask the
  termination it describes (the G-SUMMARY-NO-MASK discipline).
- **No new terminal path.** Termination still flows through the serve loop's single permitted
  direct `fail_trial`; it simply now sees a blocker it previously could not. **No worker retry
  matrix**, proven by the gate.

### §1.3 byte bound — **required statement (report item 9)**

> **No authoritative maximum shard-size contract exists in the tree.**

Searched: `miner/`, `utils/`, `docs/*.md`, and `PROPOSAL_S172_RANGE_MINER_v1_4_4.md`. The only
byte constants are `MAX_FRAME_BYTES = 64 MiB` (`range_miner_protocol.py:44`, a protocol frame cap)
and `INLINE_BYTE_LIMIT = 48 MiB` (`range_miner_worker.py:1031`). **Neither bounds a shard.**
`INLINE_BYTE_LIMIT` is the **inline-vs-spool selector** — exceeding it is precisely what routes a
shard to the spool path, and the spool path carries no size bound at all.

Per Beta §4.1 I therefore **invented no byte bound**. The byte ceiling stays **runtime-enforced**,
protected by the §1.4 timeout, and is routed as an operator value with its existing 16 GiB default.

---

## 2. The derived bound, and the number it produces

**Formula as implemented** (`:579`):

```
trial_retention_files_required =
    Σ over planned workflow stages (family, phase)
        staging_burst_bound_conservative(all planned stripe spans, eligible workers, phase, family)
  where the inner term = Σ over planned stripes of
        max over eligible workers of ceil(span / applicable_seed_cap(worker, phase, family))
```

> ## ⚠ THIS SECTION'S CONCLUSION WAS WRONG AND IS RETRACTED — see
> ## `docs/CLAUDE_CODE_REPORT_S172_STAGING_CAPACITY_R1.md` §3.
>
> The retracted text derived **816 files** from a **4-macro-stripe** geometry, called that
> "the recorded gate-12 geometry", and concluded that the observed 1,028 **"implies roughly
> five planned macro-stripes"**.
>
> **Team Beta corrected this (ruling 2026-08-08 §3). The conclusion is deleted, not amended.**
> The real 2026-08-07 gate-12 production geometry is:
>
> ```
> max_seeds         = 1,073,741,824
> miner_stripe_size =    67,108,864
> macro-stripes     = 16 per stage
> stage 0 = 504 files · stage 1 = 524 files · total = 1,028
> ```
>
> **1,028 is simply stages 0 and 1 of a SIXTEEN-stripe production run** hitting the 512
> ceiling. It implies nothing whatever about a five-stripe plan.
>
> The 4-stripe / 116-exact figure is the **2026-08-05 staging-back-pressure fixture**, built to
> demonstrate the exact-vs-conservative burst-bound distinction. It is retained in the suite as a
> compact mathematical arm under its true provenance, and the real 16-stripe geometry now has its
> own regression (`gate_trial_retention_preflight_gate12_geometry`).
>
> Retained above the strike-through only as the record of what was submitted; **do not cite the
> 816 figure or the five-stripe inference for anything.**

---

## 3. Red-first evidence (worktree at `c7058d8`)

Worktree: `git worktree add --detach … c7058d8`. The amended suite was copied in; the pre-amendment
tree lacks `StagingRetentionSizingError`, so a raw copy dies at import and yields a blanket red
instead of per-gate reds. **A shim was applied to the throwaway worktree copy only** (never to the
committed suite) so each gate reds on its own assertion.

`/tmp/s172_cap_REDFIRST.log` — **35/42, all seven new arms RED, all 35 pre-existing gates GREEN**:

| gate | red-first reason at `c7058d8` |
|---|---|
| G-HIGHWATER-ROUTE | `hop 1a: manifest default_params lacks staging_high_water_files` |
| G-TRIAL-RETENTION-PREFLIGHT | `'RangeMinerCoordinator' object has no attribute 'trial_retention_requirement'` |
| G-TRIAL-RETENTION-PREFLIGHT (serve) | `terminal reason does not lead with the retention sizing classification: 'runRPS…_s0: non-retryable failure'` |
| G-COMMIT-RELEASE | `'released_reservations'` (commit released nothing) |
| G-COMMIT-FAIL-RETAINS | `'released_reservations'` |
| G-EXECUTOR-CAPACITY-TIMEOUT | `no attribute 'staging_reservation_wait_count'` |
| G-SEQUENTIAL-TRIAL-REUSE | `'released_reservations'` |

**The serve-path arm is the most informative red:** at `c7058d8` an impossible ceiling **dispatched
stripes anyway** and then died through a stripe-failure path — the defect in one line.

**Part 2** red-first, `/tmp/s172_elapsed_REDFIRST.log`: **0/6**, each arm for its own reason
(`'elapsed_s'` KeyError ×3; `absent field decoded as 0.0`; `the additive column was not added on
open`; `record_stripe_complete() got an unexpected keyword argument 'elapsed_s'`).

## 4. Mutation evidence (where the gates specify it)

- **G-HIGHWATER-ROUTE** — hop 1 is dropped from the DECLARED set, reproducing WATCHER's
  step-scoped filter (`watcher_agent.py:1290-1314`). The gate first asserts the mutation is **not
  inert** (the key really does die in the filter), then asserts the injected 777 never reaches
  `CoordinatorConfig`. Both hold.
- **G-EXECUTOR-CAPACITY-TIMEOUT** — `_capacity_blockers_locked` is monkeypatched back to the
  reader-only oldest-pause logic while a real executor wait is outstanding and the bound has
  elapsed. The gate asserts the timeout **stays unexpired** (i.e. remains wedged/red), then
  restores the real observer and asserts it fires. This is what proves the widening is load-bearing
  rather than decorative.

---

## 5. Final-state runs (all AFTER the last edit, VM101, venv active)

| suite | result |
|---|---|
| `test_s172_staging_backpressure.py` ×3 | **42/42, 42/42, 42/42** — exit 0, sentinel PASS ×3 |
| `test_s172_staging_partb.py` | **24/24** — exit 0, sentinel PASS |
| `tests/test_s172_elapsed_roundtrip.py` (Part 2) | **6/6** — exit 0, sentinel PASS |
| `test_s172_phase4_coordinator.py` | **61/63** — see the differential below |

**Phase-4 by the accepted isolated-production-diff method.** The worktree was restored to pristine
`c7058d8` (my copied test files removed) so the differential isolates production code. Both runs
used their own committed, unmodified phase-4 suite.

```
baseline c7058d8 : 63/63
patched          : 61/63
DIFFERENTIAL (the only gates chargeable to this change):
  Gate 22   PASS -> FAIL
  Gate 37   PASS -> FAIL
61 verdicts unchanged
```

- **Gate 22** — `unexpected changed .py files: {tests/test_s172_staging_backpressure.py,
  tests/test_s172_elapsed_roundtrip.py}`. This is the **documented untracked/changed-`.py`
  sensitivity** (skill §7): Gate 22 builds `changed_py` from `git status --porcelain`. Expected
  during development, **not a regression**, resolved by committing the files. Not a reason to widen
  Gate 22.
- **Gate 37** — see §7. This one is a genuine disagreement and is reported, not worked around.

## 6. Programmatic assertion-unchanged proof

AST comparison of every pre-existing gate function between `git show
c7058d8:tests/test_s172_staging_backpressure.py` and the working tree (the round-2 method — the
committed baseline read from git, not from the shimmed worktree):

```
pre-existing functions : 53
assertion-IDENTICAL    : 53
assertion-CHANGED      : NONE
removed                : NONE
added (new)            : 8  (7 gate fns + 1 helper)

F1/credit            assertion-unchanged=True
F1-R/handoff         assertion-unchanged=True
F1-R2b/predecode     assertion-unchanged=True
F2/lease handoff     assertion-unchanged=True
F3/snapshot          assertion-unchanged=True
F4/bound-pause       assertion-unchanged=True
F5/sizing            assertion-unchanged=True
summary              assertion-unchanged=True
matrix-diff          assertion-unchanged=True
```

**No existing assertion was edited, reordered or removed** in the backpressure suite.

---

## 7. Disagreements — reported, not worked around

### 7.1 Gate 37 vs Beta Option C — an unavoidable conflict

The brief requires **both** that a successful commit deletes staged files (§1.1) **and** that all
existing gates stay green and assertion-unchanged. For exactly one gate these cannot both hold.

`test_s172_phase4_coordinator.py:2575` asserts, on a trial it has just confirmed
`state == "committed"` (`:2578`):

```python
assert m["local_spool_path"] and os.path.isfile(m["local_spool_path"])
```

**Mechanism proven, not inferred** — Gate 37 run in isolation with INFO logging:

```
[S172-CAP] retention preflight run=… mode=derived required=1 resolved=1 stages=1 stripes=1
[S172-CAP] trial … committed — released 1 trial-owned reservation(s), deleted 1 staged file(s);
           held_files=0 held_bytes=0
```

The manifest's `local_spool_path` **is** the reservation's `staged_path` (`_build_manifest:2693`
vs `ack_by_event_id:4389`), so Option C necessarily removes it. Gate 37 passes at `c7058d8` and
reds here **because the amendment did what it was told to do.**

**This is the Gate 56 precedent.** Skill §2.19 records that Gate 56 of this same phase-4 suite
"changed disposition under D (bound-proof retained); its old assertion text is SUPERSEDED — do not
cite it."

**I made no change.** Owner ruling 2026-08-08: leave Gate 37 red and report it. **Beta's
adjudication is required** — either Gate 37's file-exists assertion is superseded by Option C (my
reading), or Option C's "delete the staged files" needs qualifying.

### 7.2 Two files outside the brief's expected list

Item 7 anticipated `miner/range_miner_coordinator.py`, the three C-route hops and the suite.
Two additions, both justified:

- **`miner/range_miner_protocol.py`** (1 hunk, `elapsed_s: float = 0.0` → `Optional[float] = None`).
  **Gate G-ELAPSED-ROUNDTRIP's "a completion without the field leaves NULL rather than 0" is
  unsatisfiable without it** — with a `0.0` default, an absent field and a genuine zero decode
  identically. `Optional[…] = None` for "not reported" is the **established idiom in this very
  dataclass** (`effective_threshold`, `:147`), it preserves the CLAUDE.md §6 invariant that every
  envelope field carries a default, and it changes nothing in production because the worker always
  sets the field (`range_miner_worker.py:1345`). **No worker code was touched.**
- **`tests/test_s172_elapsed_roundtrip.py`** (new file). Part 2's gate is required to be
  independent of Part 1. A separate module makes that **structural** rather than a property of my
  inspection — see §8.

### 7.3 The `effective_high_water_files` fallback

Recorded in §1.2 above: the no-plan branch degrades to a derived value rather than refusing,
because fail-closed there red 8 existing gates. Flagged for Beta.

---

## 8. Part 2 separability (`elapsed_s`, Beta R4)

**Implemented FIRST, before any Part 1 code existed**, so its independence holds by construction.

- **Scope, exactly as ruled.** One additive ledger column (`stripes.elapsed_s REAL`, `:790`, with
  an idempotent `PRAGMA`-guarded `ALTER TABLE` migration so a pre-R4 `miner_ledger.db` gains it and
  old rows stay NULL); one persistence path (`record_stripe_complete` `:1584`, new `elapsed_s`
  parameter `:1592`, threaded from the call site `:6445`).
- **The worker-reported value is persisted verbatim; nothing is synthesized.** A coordinator-side
  clock measures arrival-to-arrival, a different quantity.
- **Idempotency is inherited, not added.** The UPDATE is guarded on `state = ST_CLAIMED`, which the
  first call clears to `ST_STAGING`; a replay matches zero rows. The gate proves this with a
  *hostile* replay carrying a different value.
- **Nothing from the excluded list** (`gpu_name`, `vram_bytes`, `gpu_id`, heartbeat counters,
  `StripeError.error`/`traceback`, `MinerStatusMessage`) was touched.

**Lift-out instructions:** revert `miner/range_miner_protocol.py`, delete
`tests/test_s172_elapsed_roundtrip.py`, and drop the four `elapsed_s` hunks in
`range_miner_coordinator.py` (`:790` schema, the `elapsed_s` migration block, `record_stripe_complete`
`:1584-1620`, call site `:6443-6445`). Part 1 touches none of them.

### Measurement caveat — recorded in source **and** here (Beta R4)

`elapsed_s` is a trustworthy **stripe SERVICE TIME** measurement, sufficient for per-stripe and
per-worker rate calculations and for sizing work. It is **NOT aggregate cluster wall-clock
throughput** — concurrent worker intervals **overlap**. **Do not reconstruct fleet throughput by
summing or averaging per-stripe seeds/sec;** any fleet-level figure needs an overlap-aware makespan
denominator. Carried in `record_stripe_complete`'s docstring, on the schema column, and in the gate
module header.

---

## 9. Files changed

| file | why |
|---|---|
| `miner/range_miner_coordinator.py` | §1.1 release-on-commit + cleanup column · §1.2 derivation, preflight, resolution · §1.4 wait registry + widened observer · Part 2 column/migration/persistence |
| `window_optimizer.py` | §1.3 hop 2 (signature, argparse, coordinator attrs, call site) |
| `window_optimizer_integration_final.py` | §1.3 hop 3 (stale `512` literal removed) |
| `agent_manifests/window_optimizer.json` | §1.3 hop 1 (`default_params` + `args_map`) |
| `tests/test_s172_staging_backpressure.py` | the six §1.5 gates (7 arms) — **all 53 existing functions assertion-identical** |
| `miner/range_miner_protocol.py` | Part 2 — `elapsed_s` default `0.0` → `None`, justified in §7.2 |
| `tests/test_s172_elapsed_roundtrip.py` *(new)* | Part 2 gate, separable by construction |

`git diff -w`: **555 insertions, 13 deletions** in the coordinator — the remainder of the raw
688-line diffstat is the `_run_staging_job` re-indent required to wrap its loop in the
`try/finally` that clears the wait record. All hunks are confined to `CoordinatorConfig`,
`MinerLedger`, `RangeMinerCoordinator`, the two `build_*` factories, the exception hierarchy, and
the module-level bound helpers.

**Not touched, as required:** worker code, seed caps, stripe geometry,
`tests/gate_s172_prod_shape.py`.

---

## 10. Verification-integrity controls (VIR-1…6)

- **execution proof:** every suite prints a completion sentinel and an exit code; all logs under
  `/tmp/final_*.log`, `/tmp/s172_cap_REDFIRST.log`, `/tmp/s172_elapsed_REDFIRST.log`.
- **clean control:** 35/35 at `c7058d8` before any edit; 61 phase-4 verdicts unchanged in the
  differential.
- **fault-injection control:** two mutations (§4), each first asserted **non-inert**; plus
  G-ELAPSED-ROUNDTRIP's genuine-zero arm, which exists so "absent → NULL" cannot pass an
  implementation that stores NULL unconditionally.
- **completion sentinel:** present in all four suites.
- **unavailable-observer behavior:** `_high_water_files_for_report` degrades to `None` rather than
  raising inside a terminal snapshot.
- **audit claim scope:** this repo tree on VM101 at `c7058d8` + working changes. **No claim about
  live fleet behaviour** — gate 12 and the soak were not run and remain HELD.
- **searched surfaces:** tracked repo; gitignored files (`agent_manifests/*.json` read live,
  `/bin/grep` used because the shell `grep` wrapper honours `.gitignore` and skips `*.json`);
  `git show` of the committed baseline; live VM101 filesystem; live Python imports and execution.
- **unavailable surfaces:** the failed gate-12 run's `total_seeds` (see §2); live rigs; any GPU
  path.
- **governance trail searched:** `docs/TB_RULING_*`, `PROPOSAL_S172_RANGE_MINER_v1_4_4.md`,
  `docs/CLAUDE_CODE_INSTRUCTIONS_*`; skill v19 §§2.7, 2.19, 2.15, §4, §7.
- **chapters searched:** not applicable — no claim here concerns sieve semantics or intent.

---

## 11. What is NOT done

- **Not committed, not pushed.** Michael commits and dual-pushes.
- **Gate 12 / G-PROD-SHAPE and the Phase-7 soak remain HELD** — untouched and unrun.
- **Gate 37 left red** pending Beta (§7.1).
- The `.gitignore` sidecar gap (§0) is reported only.
- **Staging a commit:** build the `git add` list from §9, never from recall — and note that
  committing the two test files is what clears Gate 22.
