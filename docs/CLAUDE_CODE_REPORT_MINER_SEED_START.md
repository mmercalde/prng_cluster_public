# CLAUDE CODE REPORT — THE SEED START REACHES THE SIEVE; RUN 4 EVIDENCE REFRESHED

## Prior record

**Searches run** (worktree `/home/michael/tfm_worktrees/miner-start`, base `b11b6fd`):

- The brief's own searches (listed in its Prior record) were taken as given and their anchors
  re-read live; every anchor quoted below was read in this session.
- Added here, as the brief asked:
  `/bin/grep -nI -e base_start -e seed_start -e start_seed -e G-MINER-START -e assign_stripes -e partition_macro_stripes docs/PROJECT_FILE_CATALOG.md docs/CHAPTER_* docs/*_AUDIT.md`.
- For the two new findings (§8): `/bin/grep -rlnI finalize_stripe docs`, and
  `/bin/grep -rlnI -e "StripeComplete.*pump" -e "pump.*StripeComplete" -e "no pump" -e "never pumped" -e "deferred.*stall" -e "stall.*deferred" docs`;
  `/bin/grep -nI pump docs/BACKLOG.md docs/LEADS.md docs/TODO_CONSOLIDATED.md`.

**Documents found that decide, schedule or record the subject**

| Document | What it says | Label |
|---|---|---|
| The brief's table (stage-1 report §6, Beta X-STEP1-SEARCH-CODE R1 Q10/Q13, `window_optimizer.py:675-703`, owner 2026-11-14 on PWC/ZMQ, survivor-volume M-B, proposal v1.1 §6/§8, Run 4 ruling `:154-179`, `gate12_parity_gate.py:174`) | As the brief states | ALREADY RECORDED |
| `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:109-114`, `docs/CHAPTER_2_SOURCE_MAP_v1.md:223`, `docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md:1502` | Describe `partition_macro_stripes` as tiling `[base_start, base_start + total_seeds)`. True of the function; silent on the caller passing 0 | ALREADY RECORDED (description only, nothing decided) |
| `docs/PROJECT_FILE_CATALOG.md:235` | `assign_stripes` appears only in the fleet-state ruling (admission guard) | Not about the start |
| `docs/CHAPTER_1_*`, `docs/CHAPTER_1_AUDIT_v1.md`, `docs/HYBRID_SKIP_BOUND_AUDIT.md` hits on `seed_start` | Optimizer signatures and the legacy sieve; nothing on the miner start | Not about this subject |
| Pump / StripeComplete ordering (§8 L-4) | No hit decides or records it. The only "no pump" hits are field-6 (`TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md:157`, field-6 report/brief): instrumentation `UNOBSERVED` when no pump pass ran | **NEW** |
| Root logger raised by `window_optimizer` import (§8 L-5) | Not found | **NEW** |
| `docs/exchange/from_beta/X-RUN4-RANGE_R1_BETA_REVIEW.md` **Q4** (`:87-97`) and `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:79-109`, `:145-153` | Q4: preserve the frozen cursor-zero oracle for its original comparison and add a separate real-cursor case; the R2 ruling closes unpinned behaviour on the 47-token `69ca910` oracle and sets the post-commit steps. **Missed by the brief's and this request's inventory; supplied by Beta R1 Finding 1** | ALREADY RECORDED — and in tension with Part 2 (§5, §9) |

**Items this report raises**

| Item | Label |
|---|---|
| R1–R7 implementation and tests | Implements the brief |
| `trial_retention_requirement`'s `partition_macro_stripes(…, 0)` (`:5104`) | NEW, read here: span-only, start-invariant (§2) |
| G-UNPINNED-IDENTICAL cannot pass on today's path | NEW consequence of stage 1 (`915d382`) |
| L-4: staging admission held by a DONE stripe, nothing pumps | **NEW** (registered `docs/LEADS.md` L-4) |
| L-5: WATCHER loses INFO records after Step 1's search comparison | **NEW** (registered `docs/LEADS.md` L-5) |
| A certified suite probes the live staging directory | NEW to me; a rule deviation of this session (§6.4) |

**Not searched:** VM101 host state beyond the live files snapshotted (§6.4), ser8 archives, the
private repository, project PDFs, any rig.

---

## For the owner, in plain words

- **What now happens when Run 4 is launched.** WATCHER takes the start from the open search
  (`--seed-start 2147483648 --max-seeds 2147483648`). The child's entry check passes (the
  miner route is no longer refused). The coordinator now partitions the stripes from that
  start: 32 stripes per phase over `[2147483648, 4294967296)`, in all four phases, and the
  workers are told those ranges. The evidence reader returns `[2³¹, 2³²)` for every phase, so
  the search can credit the chunk, and the history then reads complete over `[0, 2³²)`.
  Shown on CPU, end to end, with 25 fake workers (G-E2E, §5).
- **PWC and ZMQ** at a non-zero start now refuse before any dispatch, naming the reason. At 0
  they behave exactly as at `b11b6fd` (argument-identical, shown against the base source).
- **The rigs need the new `miner/range_miner_coordinator.py` before the parity gate can pass
  at power-on.** It is in the parity set (`scripts/gate12_parity_gate.py:174`); the gate will
  refuse until the rigs carry this exact file (sha256 in §10). That is your step at power-on.
- **Not verified:** no GPU, no fleet, no ROCm, no real worker. Everything here is CPU with
  fake workers that answer instantly.
- **Two new findings you should see** (§8, both registered OPEN in `docs/LEADS.md`):
  - **L-4:** the coordinator can stall: if a stripe's last staging job finishes before its
    StripeComplete is processed, the staging admission stays held by a finished stripe and
    nothing resumes the deferred work. With production's `serve_timeout=None` that would be
    an indefinite wait. Reproduced on CPU (1 of 10 two-worker runs; 4 of 4 runs at 4 workers ×
    2²⁶ stripes); not observed on the fleet. My tests work around it with a labelled
    harness timer; the coordinator is not changed for it.
  - **L-5:** since stage 1, a WATCHER process's first Step 1 search comparison imports
    `window_optimizer`, whose imports raise the root logger to WARNING. WATCHER's `EXEC CMD`
    and later INFO lines then disappear from its log. Measured: 0 `EXEC CMD` records vs 1 in
    the control.
- **One rule deviation by this session:** a certified suite I ran
  (`test_s172_staging_partb.py`, `gate_manifest_staging_dir_is_usable_here`) writes and
  removes a probe file in the live `/home/michael/miner_staging`. Directory contents and
  ledger bytes are identical to my pre-run snapshot, but I should have skipped that case.

---

## 1. First step

| Item | Result |
|---|---|
| `pgrep -af beta_relay.py` | no relay (the only match was the shell running the command) |
| `git rev-parse HEAD` (main) | `b11b6fde7ab57321b42c418f87af8cc2c75c814d` |
| `git status --porcelain` (main) | clean |
| Worktree | `git worktree add --detach /home/michael/tfm_worktrees/miner-start HEAD` → `b11b6fd` |
| venv | `~/venvs/torch` (Python 3.10.12) |
| Dataset files, copied read-only (mode 0444), sha256 equal to main | `daily3.json` `513648160d35…68f6`; `daily3-20260801T145551443433Z-513648160d35.json` `513648160d35…68f6`; `daily3_current.json` `ef3275264a90…250a` |
| Catalog / chapter / audit search | Prior record above |
| Live-state snapshot | `data/miner_seed_start/live_snapshot_before.log` (worktree) |
| Baseline | 27 suites, sequential, CPU only, before any edit (§6) |

## 2. M — the stage-1 demonstration, reproduced before any edit

Script: the stage-1 demo, unchanged except its output name
(`data/miner_seed_start/M/m_result.json`).

- Real `run_bidirectional_test`, miner route, `seed_start=2147483648`: `run_trial_miner` got
  39 keywords, `total_seeds=1000`, **no keyword carried the start**.
- Real `run_trial_miner` → `serve_trial` → `assign_stripes`, two loopback fake workers,
  `seed_start=2147483648` offered: stripes **`[0,250) [250,500) [500,750) [750,1000)`**,
  trial committed. The offered `seed_start=` was silently swallowed by the `**kwargs` tail.

**What `:5104` is.** `RangeMinerCoordinator.trial_retention_requirement`
(`miner/range_miner_coordinator.py:5089`), called by the retention preflight. It calls
`partition_macro_stripes(total_seeds, miner_stripe_size, 0)` and keeps **only the counts**
(`spans = [count for (_idx, _start, count) in macro]`, `:5106`). Spans depend on
`total_seeds` and the stripe size, never on the origin; nothing it returns carries a seed
position. It creates no stripe and places nothing. It is left unchanged; editing it would
change a definition for no behavioural effect and widen SR-1's declared sets.

## 3. Part 1 — requirements

| # | What was done | Anchors (worktree) |
|---|---|---|
| R1 | `run_bidirectional_test` passes `seed_start=seed_start` to `run_trial_miner`. `run_trial_miner` has an explicit `seed_start: Optional[int] = None` (no longer swallowed by `**kwargs`) and puts it on the serve context. `serve_trial` reads `context["seed_start"]` with no default and passes `base_start=` to its single `assign_stripes` call. `assign_stripes` and `partition_macro_stripes` were already origin-aware and are unchanged. The only stripe-row creator is `assign_stripes` → `ledger.add_stripe` (`:4056`). Retries, lease expiry and re-placement re-claim existing rows; the scheduler payload reads `row["seed_start"]` (`:4231`) and dispatch sends `stripe["seed_start"]` (`:12006`). No coordinator-process resume exists (certified boundary: coordinator death = interrupted, not resumable); a Step 1 checkpoint resume re-enters `run_bidirectional_test` with the argv's start, which G-HOP covers | integration `:1741`; coordinator `:12400` (param), `:12452-12468` (check, `if` at `:12461`), `:12535` (context), `:9517-9525` (serve read+check), `:10339` (assign) |
| R2 | Integration: the S145 domain wall (`assert_seed_domain_preflight`, the finalizer's own constant, imported) on `(seed_start, seed_count)` immediately before anything is built for the miner. Coordinator: a fail-closed check that the start is an explicit non-negative non-bool int, first statement of `run_trial_miner` (before staging, ledger, trial row) and before the dataset digest and socket in `serve_trial`. The upper bound is not repeated in the coordinator: importing the constant there would widen the module's import closure, which the rig parity gate governs (`execution_set.py` is in the governed set for exactly that reason) | integration `:1713-1722`; coordinator `:12461`, `:9517-9518` |
| R3 | No code change. `_step1_miner_sweep_evidence` (`window_optimizer_integration_final.py:1426`) already derives each phase's interval from the run's stripe rows (`seed_start`, `seed_count`, `state='done'`, contiguous). It now yields `[seed_start, seed_start+total)` because the rows are right. No ledger schema change; the migration note at `:1400` was read and does not apply | `:1426-1470` |
| R4 | `assert_miner_seed_start_supported` / `Step1MinerStartUnsupportedError` are removed and replaced by `assert_step1_route_seed_start_supported` / `Step1RouteStartUnsupportedError`, which does not refuse the miner route. The second guard (`credit_decision`, `utils/step1_search_ledger.py:1044`) is untouched | `window_optimizer.py:674-712`, call `:819-821` |
| R5 | The same guard refuses PWC/ZMQ at a non-zero start: at the child's entry (`run_bayesian_optimization`, after the S145 wall, before the coordinator exists), for every planned iteration of `run_with_config` (iteration *i* starts at `i × max_seeds`, so PWC with ≥2 iterations is refused as a plan), and per trial at the head of the PWC and ZMQ branches of `run_bidirectional_test`, before `run_trial_persistent` or the ZMQ session coordinator. Neither coordinator file is edited | `window_optimizer.py:819`, `:1253-1259`; integration `:1876-1883`, `:1937-1944` |
| R6 | `tests/gate_s172_prod_shape.py` read in full: it checks trial/stripe/shard states, phases, acks, cleanup, remote deletes and reservations. **It never reads `seed_start` or any seed range** (`:230-345`). Nothing assumes a zero start; **not edited** | — |
| R7 | A zero-start control captured on the unmodified tree (twice, identical digests), compared byte-for-byte after (§4, G-R7-ZERO) | `tests/fixtures/miner_seed_start_zero_control_b11b6fd.txt` |

**SR-1 (`DECLARED_CHANGED`).** AST diff against `HEAD`: the coordinator changes exactly
`run_trial_miner` and `RangeMinerCoordinator.serve_trial`; **no definition added or removed**.
Both are already in R-1's and MP-1's declared sets (Window-Anchor Brief I added them), so no
declared set needed an edit; R-1 44/44 and MP-1 PASS after. `window_optimizer.py` changes
`run_bayesian_optimization`, `run_with_config`, adds `assert_step1_route_seed_start_supported`,
removes `assert_miner_seed_start_supported`; the integration file changes
`run_bidirectional_test` only. The MP-1-pinned admission subtree in `serve_trial`
(`:10202-10238`) does not contain the edited call (`:10339`).

## 4. Tests — `tests/test_s172_miner_seed_start.py`

CPU, real coordinator, fake workers over real framed loopback sockets
(`tests/fixtures/miner_seed_start_driver.py`). **10/10 gates, 11/11 mutants detected,
`COMPLETION SENTINEL: PASS`.** Mutants are function-level, compiled against the production
module globals.

| Gate | Shows |
|---|---|
| G-HOP | `run_trial_miner(seed_start=2147483648, total_seeds=1000)`; AST: the call's `seed_start=` is the function's own parameter |
| G-RUN4-GEOM | start 2³¹, total 2³¹, stripe 2²⁶, both modes: 32 stripes per phase in phases 1–4, first `2147483648`, last ending `4294967296`, contiguous, all `done`, every stripe 2²⁶; the 128 StripeAssigns the workers received equal the ledger rows |
| G-SHORT | start 1000, total 1050, stripe 250: `[1000,1250) … [2000,2050)`, short last = 50 |
| G-RETRY | hybrid, one worker fails retryably: the stripe is re-assigned (attempt 1) at the same start; committed |
| G-LEASE | hybrid, one worker silent, lease 1.5 s: expiry, re-placement at the same start; committed |
| G-R2-ENDS | integration wall: `[0,1000)` and `[2³²−1000, 2³²)` reach `run_trial_miner`; `[−1,999)`, `[2³²−999, 2³²+1)`, `[2³², 2³²+1)` raise `SeedDomainPreflightError` and do not. Coordinator: omitted, `None`, `True`, `1.0`, `−1` raise `ValueError` naming `seed_start` before any ledger file exists; `0` reaches the context. `serve_trial` with no start refuses before binding a socket |
| G-R3-EVID | the real G-RUN4-GEOM run: evidence `{1,2,3,4: [2³¹, 2³²)}`, **credited**; the same evidence against `[0, 2³¹)` → `interval_mismatch`. Mutated ledger copies: gap → `no_sweep_evidence`; wrong start (phase 1 shifted by −2³¹) → `interval_mismatch`; one stripe not `done` → `no_sweep_evidence`; overlap by one seed → `no_sweep_evidence` |
| G-R5-ROUTES | PWC and ZMQ at 1000 raise `Step1RouteStartUnsupportedError` naming the route and start, with **no** dispatch seam called (`run_trial_persistent`, `ZMQSQLiteCoordinator`, `run_trial_zmq_sqlite` recorded). At 0, the recorded calls are **argument-identical to `b11b6fd`'s `run_bidirectional_test`** (base source executed in the same globals). Child entry: PWC/ZMQ at 2³¹ refused before the coordinator; miner and legacy at 2³¹, PWC/ZMQ at 0 reach it. `run_with_config`: PWC × 2 iterations refused as a plan; PWC × 1 and legacy × 2 reach the coordinator |
| G-R7-ZERO | seed_start = 0, attempt 9's shape (2³¹, 2²⁶, both modes: 128 stripes / 128 assigns) and the stage-1 demo shape (1000/250, phase 3): ledger rows, every StripeAssign including its payload, and the evidence are **byte-equal** to the control (digests `972d23b4e5a3…`, `737e803754f8…`). The gate checks the control came from the base coordinator (`git show b11b6fd:` sha), that the live coordinator differs (not vacuous) and that the driver is the captured one |
| G-E2E | stage-1 WATCHER harness: history `[0, 2³¹)`, open pinned search `[2³¹, 2³²)`, real `run_step` → argv `--seed-start 2147483648 --max-seeds 2147483648`, route RANGE-MINER only; the child's entry with those values reaches the coordinator (tripwire); the **real** `run_bidirectional_test` → **real** `run_trial_miner` with 25 fake workers at the production caps (sub-stripes as the real worker partitions) → 4 × 32 stripes over `[2³¹, 2³²)`, committed; evidence = request; `credit_decision` credits; credit chunk linked; certified cursor **COMPLETE, 2³² covered**; the search reads `search_complete` |

| Mutant | Detected by |
|---|---|
| M1 integration drops `seed_start=` (the stage-1 shape) | G-HOP |
| M2 `serve_trial` assigns without `base_start` | G-SHORT (stripes from 0) |
| M3 `run_trial_miner` puts 0 on the context | G-SHORT |
| M4 coordinator start check disabled | G-R2-ENDS |
| M5 integration domain wall removed | G-R2-ENDS (`[−1,999)` reached the miner) |
| M6 / M7 PWC / ZMQ branch guard removed | G-R5-ROUTES |
| M8 guard predicate ignores ZMQ | G-R5-ROUTES |
| M9 `run_with_config` plan guard removed | G-R5-ROUTES |
| M10 evidence reader accepts non-`done` stripes | G-R3-EVID (credited) |
| M11 zero-start geometry perturbed by one seed | G-R7-ZERO |

**Labelled intercepts.** (1) Fake workers, not GPUs. (2) A probe Phase-5 sink (no assembly)
in the driver; G-E2E stops after the coordinator returns, before Phase-5 assembly. (3) In
G-E2E the history row is written from a synthetic publication witness (D3.5 publication
needs a real NPZ); the chunk link is the production one. (4) **The pump nudger**
(`tests/fixtures/miner_seed_start_driver.py`, `pump_nudger`): calls the coordinator's own
`_pump_deferred()` every 0.5 s, because fake workers hit finding L-4 almost every run. It
patches no code and changes when deferred frames resume, never what a stripe is assigned.

**Changes to existing suites.**
- `tests/test_step1_search_record_stage1.py`: RUN4-SHAPE and G-MINER-START asserted the guard
  this brief replaces. RUN4-SHAPE now asserts the child's guard **accepts** the miner argv and
  refuses the same start on PWC; G-MINER-START → G-ROUTE-START (PWC/ZMQ at 2³¹ refused; miner
  at 2³¹ and 0, legacy at 2³¹, PWC/ZMQ at 0 reach the coordinator); its mutant moved to the
  new predicate. 39/39 gates, 33/33 mutants.
- An explicit `seed_start=0` was added, by AST-located insertion, to every existing test call
  of `run_trial_miner` and to the one dict-literal serve context, in `test_s172_admission_liveness`,
  `attempt6_remediation`, `f1_lease_origin`, `phase1_scaffolding`, `phase4_coordinator`,
  `phase5_d0`, `phase5_d1_workflow`, `phase5_d2_directional_uniqueness`,
  `staging_backpressure`, `staging_partb`. Nothing else on those lines changed. Without it the
  new fail-closed start refuses those runs, which is the intended behaviour.

## 5. Part 2 — the Run 4 evidence

Fixtures refreshed: a new `tests/fixtures/run4_search_fixture.py` (the stage-1 collaborator:
a throwaway Step 1 search ledger, opened by the production `open_search` with the
configuration from the production `_step1_cli_final_params` + `step1_presented_configuration`,
range `[2³¹, 2³²)`, pinned when the build is pinned); the stub DB and stub frozen dataset in
`run4_routing_clean_control.py` and `run4_prelaunch_provenance_check.py` now come from it;
`run4_pin_harness.py` tells the fixture which kind of build it is making. **No assertion was
changed.**

| Suite | Before | After |
|---|---|---|
| `tests/run4_prelaunch_provenance_check.py` | 3/4 printed, `ARGV-BUILT` FAIL (lookup failed) | **10/10, `RUN4_PRELAUNCH_PROVENANCE_OK`**. The built command carries `--seed-start 2147483648 --max-seeds 2147483648`, the seven pins at the approved values, `--use-range-miner`, `--step1-search-id s1s-…` |
| `tests/test_s172_run4_routing_patch.py` | 4/20 | **15/20** |

**The five that stay red, and why they cannot pass without weakening.**
`G-UNPINNED-IDENTICAL` compares today's unpinned argv list-for-list with the argv captured at
`69ca910` and refuses if the frozen fixture module's sha changed. The difference today is
exactly, and only: `--seed-start 0 → 2147483648` (the search's first gap — the purpose of the
search record and of this brief) and an inserted `--step1-search-id <id>` (stage-1 S6). Making
it pass would mean re-capturing the oracle at today's tree or allowing that delta; both
weaken it. It reports `INCOMPLETE` ("fixture module changed since capture — oracle invalid"),
which is true: editing the module was required to give it a search lookup. M1, M1b, M2a and M4
are each **detected by their own gates**, but each lists `g_unpinned_identical` as
must-stay-green, so the harness cannot credit them. M2b passes, but its credit through
`g_unpinned_identical` is vacuous for the same reason; its `g_provenance` detection is real.
Re-capturing the routing control is an owner/Beta decision.

**Two fixture defects found and fixed while refreshing** (both would have shown as gate
failures that were not about routing): the fixture first passed the raw bundle to
`step1_presented_configuration` where `run_step` passes `_step1_explicit_pin(step, bundle)`
(identical in production, different under M2b); and computing the configuration before
`run_step` imported `window_optimizer` early, which suppressed WATCHER's pin-provenance INFO
record (the fixture now restores the root level; the production side is L-5).

## 6. Suites before and after

CPU only (`CUDA_VISIBLE_DEVICES=""`), one at a time. Logs:
`data/miner_seed_start/{baseline,after}/` in the worktree.

| Suite | Baseline | After |
|---|---|---|
| test_step1_search_record_stage1 | PASS 39/39, 33/33 | PASS 39/39, 33/33 |
| test_s172_run4_routing_patch | 4/20 | 15/20 (§5) |
| run4_prelaunch_provenance_check | FAIL | 10/10 PASS |
| test_s172_phase4_coordinator | 63/63 | 62/63 — **Gate 22 only**: modified/untracked `.py` files; transitional, self-clears on commit, allowlist not to be widened |
| test_s172_r1_drain_remedy | 44/44 | 44/44 |
| test_s172_mp1_drain_attribution | PASS | PASS |
| test_s172_f1_f2_active_lease | 16/16 | 16/16 |
| test_s172_f1_lease_origin | PASS | PASS |
| test_s172_admission_binding | 11/20 | 11/20, same failing gates |
| test_s172_admission_liveness | 16/16 | 16/16 |
| test_s172_attempt6_remediation | 76/78 (RXP-3/6, 3/7) | same failing gates |
| test_s172_defect_a_transport_recovery | 29/29 | 29/29 |
| test_s172_staging_backpressure | PASS | PASS |
| test_s172_staging_partb | PASS | PASS (see §6.4) |
| test_s172_window_anchor_brief_i | FAIL | FAIL, same failing gates |
| test_s172_window_anchor_brief_i_mutants | PASS | PASS |
| test_s172_phase5_d0 | 0/12 | 0/12, same |
| test_s172_phase5_d6_production_adapter | positive control fails | same |
| test_s172_phase5_d6_threshold_path | positive control fails | same |
| test_s172_threshold_propagation | FAIL | FAIL, same failing gates |
| test_s172_h1h2_instrumentation | PASS | PASS |
| test_s172_elapsed_roundtrip | PASS | PASS |
| test_s172_d6_integration_repair (parity gate) | PASS | PASS |
| test_s172_resolved_execution_set | 34/34 | 34/34 |
| test_s172_phase1_scaffolding | 6/6 | 6/6 |
| test_seed_domain_cursor_amendment | PASS | PASS |
| test_gate_prod_shape_scoping | PASS | PASS |
| **test_s172_miner_seed_start (new)** | — | **PASS 10/10, 11/11** |

"Same failing gates" = the sorted set of failing gate names is identical before and after
(compared by digest). Those reds predate this work and are not charged to it.

### 6.4 Live state

Snapshot before any suite, compared after: halt file absent both times; the live miner
ledger and its `-wal`/`-shm`, and the main tree's four `.db` files, **byte-identical**;
`optuna_studies/` 104 entries both times. **One deviation:** `/home/michael/miner_staging`'s
directory mtime moved (15:47:28) while its contents stayed identical. The cause is
`tests/test_s172_staging_partb.py:599-606`, `gate_manifest_staging_dir_is_usable_here`, which
runs `validate_coordinator_staging_dir` — including its atomic-rename probe — on the
manifest's live staging path. It ran in all three of my passes. The brief forbids writes
there; I should have skipped that case. Recorded, not repeated.

## 7. Beta's last position per question

Thread X-MINER-START, **one round** (R1). Relay verdict `REVIEW RECEIVED`, exit 0, 386 s, no
boundary violation. Review: `docs/exchange/from_beta/X-MINER-START_R1_BETA_REVIEW.md` (main
tree).

| Q | Beta R1 | What was done |
|---|---|---|
| 1 Start reaches every partition / re-assignment | NO OBJECTION | — Beta: covers the Step-1 route, not arbitrary direct API calls (the raw miner entry checks type and lower bound, not `start + total ≤ 2³²`); no coordinator restart/resume certified |
| 2 Interval from execution evidence | NO OBJECTION | — Beta: interval accounting, not authentication of work |
| 3 Zero start unchanged | NO OBJECTION | — Beta verified the R7 construction; rerunning it was UNAVAILABLE in its sandbox (no loopback sockets) |
| 4 PWC/ZMQ refusal before dispatch, nothing else changed | NO OBJECTION | — Beta reran G-R5-ROUTES itself: PASS |
| 5 G-PROD-SHAPE | NO OBJECTION | — |
| 6 Run 4 fixtures, any assertion weakened | NO OBJECTION | — but "means only that the submitted assertions were not weakened"; 15/20 is **not** a renewed routing certificate. A successor comparison needs an independently established control, not one captured from this candidate |
| 7 What remains unverified; any reason not to launch | **OBJECTION · LIVE** | **Stopped here (stop gate).** L-4 can strand staging admission in a normal multi-worker run with `serve_timeout=None`; the new suite completes only with the harness pump. Beta: resolve L-4 through a separately reviewed repair, and exercise staging-before-StripeComplete with multiple workers **without** the harness pump, before launch; do not fold it into this change. I agree. The fix is outside this brief's scope, so no repair was attempted and no further round was run |
| 8 Missed governing record | SUGGESTION | **Adopted** (in scope, documentation only): Prior record above now carries X-RUN4-RANGE R1 Q4 and the R2 routing ruling; §5 and §9 state the tension with Part 2 |

Beta's independent execution: stage-1 39/39 + 33/33; prelaunch 10/10; routing 13/14 positive
and containment gates (frozen oracle INCOMPLETE), M1b UNAVAILABLE; new suite G-HOP, G-R2-ENDS,
G-R5-ROUTES PASS, the seven socket-dependent gates UNAVAILABLE; mutants M1, M4–M9 meaningfully
detected, M2/M3/M10/M11 UNAVAILABLE (a `PermissionError` is not a detection); L-4 mechanism
confirmed by an injected DONE admission holder on a real coordinator (admission stays false;
pruning restores it). Beta did not run the live-path staging probe case.

No severity is contested.

## 8. New findings (both OPEN, registered in `docs/LEADS.md`)

**L-4 — staging admission can stay held by a DONE stripe, and then nothing pumps.**
The coordinator stages one attempt at a time (`_try_admit_locked`, `:4832`, refuses while
`_admitted` is non-empty). An admitted key is pruned only by `_prune_admitted_locked`
(`:4783`), only inside `_pump_deferred`, once its stripe is `done`. `_run_staging_job`
finalizes then pumps (`:8230`, `:8234`); `_serve_dispatch`'s StripeComplete branch finalizes
and does not pump (`:11897`); `enqueue_staging` does not prune. If the last staging job of a
stripe finishes before its StripeComplete is processed, the stripe becomes `done` on the
non-pumping path, and every later attempt is deferred with nothing to resume it.
- Observed on the live object of a stalled run: `_admitted` = {s5}, s5 `done`;
  `_deferred` = [s6, s7].
- Rate: 2 workers × 4 small stripes without the nudger, **1/10** stalled; 1 worker, **0/10**;
  4 workers × 2²⁶ stripes, **4/4** before the nudger.
- The stage-1 report's demo lost its 4th stripe the same way (read then as a fake-worker limit).
- **Not observed on the fleet.** Real staging (fsync, larger payloads) may usually finish after
  the StripeComplete is processed. With `serve_timeout=None` the failure mode is an
  indefinite wait. Open follow-ups and fix options are in the register; the choice is Beta's
  and yours. SR-1 applies to any fix.

**L-5 — WATCHER loses its own INFO records after Step 1's search comparison.**
`step1_presented_configuration` (`agents/watcher_agent.py:769`) imports `window_optimizer`;
its chain (sklearn → `scipy.stats` → `numpy.testing` → `numpy.distutils.log`) sets the root
logger to WARNING. `run_step` calls it in the S4 comparison, so `EXEC CMD` and the
`[S172-PIN]` record (`:2424`) are dropped from that point. Fresh process, real `run_step`:
root 20 → 30; `EXEC CMD` 0 records; control 1. The structured provenance is unaffected.

## 9. Unresolved

- **Run 4 is not ready to launch** on Beta's LIVE Q7 objection, which I agree with: L-4 needs
  its own reviewed repair and a multi-worker staging-before-StripeComplete test without the
  harness pump. That is a new brief.
- **Part 2 vs X-RUN4-RANGE R1 Q4.** Q4 says to preserve the frozen cursor-zero oracle and add a
  separate real-cursor case; this brief said to refresh the fixtures, and the search lookup
  could only be supplied by editing the frozen module, which invalidates that oracle (the
  capture file `run4_clean_control_69ca910.txt` is untouched). Two ways forward for Alpha and
  the owner: keep the edited module and record an explicitly scoped successor comparison with
  an independently established control (Beta Q6), or restore the frozen module byte-for-byte
  and supply the search collaborator only through a separate real-cursor harness. Either way
  G-UNPINNED-IDENTICAL and the four mutants that rely on it stay uncredited until then.
- L-4, L-5 (§8).
- `docs/CLAUDE_CODE_REPORT_STEP1_PURPOSE_LINEAGE.md:213-215` (the lineage claim that every
  seed in `[seed_start, …)` is visited) is now true for the miner route and still false for
  PWC/ZMQ, which now refuse instead. The doc correction stays with Alpha, as the brief says.
- `_step1_miner_sweep_evidence` opens the live ledger with `mode=ro` without `immutable=1`; a
  memory note records that this form wrote `-wal`/`-shm` beside the live ledger before. Not
  changed (outside R3's need); listed for Alpha.
- `.claude/scheduled_tasks.lock` (15:14) and another session's X-PWC-BACKUP relay files are in
  the main tree; not mine, not touched.

## 10. Landing list

**Group A — files the landing script takes**

| File | sha256 |
|---|---|
| `window_optimizer.py` | `a0e7e90505bf61830038093ddcd44fa2d9d6d6473e0eb5d9e853b8c2a290a728` |
| `window_optimizer_integration_final.py` | `86041fd0c1c6d91d3fbf1d55e6d6481b77af5feacf4d86c57bc0f095143fc987` |
| `tests/test_s172_miner_seed_start.py` (new) | `81ca284cb80fb48757c4b2596fdb827fc304b3affb95cc2f268b318f49e72c42` |
| `tests/fixtures/miner_seed_start_driver.py` (new) | `2d264a66fdaa047daf91e8149952df4a64e44ab39d6b8e64856f3a61c0c6b542` |
| `tests/fixtures/miner_seed_start_zero_control_b11b6fd.txt` (new) | `9ccde08087a189f1f0534532bf8b8c65c63f194d2b427ffcf68ccdc95f216f81` |
| `tests/fixtures/run4_search_fixture.py` (new) | `d45f9d2a8123ada201a91499376c1398c66311fd82becf1f5836e9054e897733` |
| `tests/fixtures/run4_routing_clean_control.py` | `73e205c14d0d542d3af580f9ff0b4f48a849079a92b4771e3f670c2888dcfc6b` |
| `tests/fixtures/run4_pin_harness.py` | `31c452b05eb93960dfc8f40d7c0d6eb1ab17ec7dd4cecc7ad74d6f6bde0d1dee` |
| `tests/run4_prelaunch_provenance_check.py` | `70c222a4d201f2a00df9621a748c15015b38f8fc07c35aa29965409eb03a794e` |
| `tests/test_step1_search_record_stage1.py` | `70f312a62698b4a48188b06fbd7f313435037319f3a09295418a91c0d2a67d2f` |
| `tests/test_s172_admission_liveness.py` | `9c3082ab3a4b1bf1e77aff258e577fc1b59866fd0958edadf94d939bc0388316` |
| `tests/test_s172_attempt6_remediation.py` | `e88e71979e2b1863201cba498d69ff8ffad18b72b2c79823b44507de78563dc1` |
| `tests/test_s172_f1_lease_origin.py` | `42083609eb76bbb6c4923676456deba6665ddbc8805aedc60b1a3f043f6695be` |
| `tests/test_s172_phase1_scaffolding.py` | `8b3e24cf665022087dce6e21b6780726035c07857d6acece4b1901515ff7f076` |
| `tests/test_s172_phase4_coordinator.py` | `3ab5f4df277d18fa8d52cb3def7fb63b353788693df34a155668800252da9021` |
| `tests/test_s172_phase5_d0.py` | `a78d166dcbd6f5e2f0124e25722fa050d33623bb6d73784dda1ecd3c541e9176` |
| `tests/test_s172_phase5_d1_workflow.py` | `832ceb5fa7d3e886f43297c91cf72528962ff29e43840b67d2ef06c132aeeb77` |
| `tests/test_s172_phase5_d2_directional_uniqueness.py` | `a70e71e6f60be0f7c5b417823d63436de0f23acc8a0701c541110c851c90291c` |
| `tests/test_s172_staging_backpressure.py` | `20214f1b97b572d446a60b0c24045e9056388c72217135b44e08f9aecf967e65` |
| `tests/test_s172_staging_partb.py` | `5cd668c68578540f59b2aec10f2813cce0790069574d845c653306e4825c7ab2` |
| `docs/LEADS.md` (L-4, L-5 appended) | see §11 |
| `docs/CLAUDE_CODE_REPORT_MINER_SEED_START.md` (this file) | see §11 |

**Group B — hand-landed** (`scripts/owner_land.py:86-87`: `miner/` is hand-landed)

| File | sha256 |
|---|---|
| `miner/range_miner_coordinator.py` | `6f2d14f3a48a585c8b59ecf5c295b89bbc1c74cec1d48b569804637194c5c7f9` |

This file is in the rig parity set: **the three rigs must carry these exact bytes before the
parity gate passes at power-on.** Not edited: `miner/range_miner_worker.py`,
`range_miner_protocol.py`, `dataset_authority.py`, `prng_registry.py`, `sieve_gpu_worker.py`,
the kernels, the three `utils/` ledgers/finalizer, `agents/watcher_agent.py`, both PWC/ZMQ
coordinators, `gate12_launch.sh`, the 22-array contract.

## 11. Review by Beta — status at stop

One round, stopped at the §5 stop gate (a LIVE objection whose fix is outside scope). Files in
the main tree from this cycle, for Michael: this session's request
`docs/exchange/to_beta/X-MINER-START_R1_ALPHA_REVIEW_REQUEST_seed_start_into_sieve.md`, Beta's
review `docs/exchange/from_beta/X-MINER-START_R1_BETA_REVIEW.md`, and the relay's row in
`docs/exchange/LEDGER.md` (which also carries another session's X-PWC-BACKUP row). Relay
stdout: `data/miner_seed_start/relay_r1.out` (worktree). Nothing was committed or pushed.
Final hashes of the two documents in Group A are given in the session summary, because a
file cannot carry its own hash.
