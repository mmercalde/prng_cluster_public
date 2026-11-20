# NOTE · Run O (first full-range single run) · inbound_saturation_timeout · read-only analysis

## Prior record

**Searches run** (2026-11-19, read-only, main tree `/home/michael/distributed_prng_analysis` at `7c725c8`, plus the fact
file `~/.claude/skills/tfm-project-facts/SKILL.md` = FF):
- `/bin/grep -rlE 'MP-1|R-1 DRAIN|drain starvation|X-MINER-STALL|inbound_saturation|maxsize=1024|INBOUND_SATURATION|ADMISSION_STALL|queue_depth=2|staging_queue_depth|capacity_timeout' docs/ --include='*.md'`
  This gave 65 hits, covering `PROJECT_FILE_CATALOG.md`, `TB_RULING_*`, `TB_SUBMISSION_*`, `CLAUDE_CODE_*`, `LEADS.md`,
  `BACKLOG.md`, `TODO_CONSOLIDATED.md`, `exchange/` (LEDGER, DEFERRED, X-MINER-STALL, X-LEDGER-RELEASE,
  X-STEP1-CREDIT-WAL), the changelogs and the handoffs.
- Targeted reads of the hits below. One read-only Opus sub-agent did the document reading; its citations were spot-checked
  only where quoted here.

**Found: what decides or records the subject.**

| Item | Record | Label for this note |
|---|---|---|
| MP-1 drain starvation | `GATE12_MP1_RUN_FORENSIC.md:239–244` ("681 s of a 969 s trial blocked inside enqueue_staging"). `SESSION_CHANGELOG_20260816_MP1_DRAIN_ATTRIBUTION.md:104–113` (chain: "msg per frame grows -> fewer frames per pass -> inbound backlog grows"). FF §2.47 :2364–2391 | ALREADY RECORDED: same lock, same chain |
| R-1..R-4 remedy and its closure | FF §2.48 :2398–2407. FF:13 and :2501 "**MP-1 drain starvation is CLOSED — do not reopen R-1**". `TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md:17, :74` ("Do not reopen R-1"), **:76 "The absolute pump total remains worth watching as a future efficiency opportunity, but it is no longer producing the lock-starvation failure"**, :104 "No additional drain remedy is authorized" | ALREADY RECORDED (closure). Run O's measurements bear on :76; any code change in §3 C/D/E would **CHANGE A RULING** and needs the owner |
| Options rejected in R-1 | `SESSION_CHANGELOG_20260817_R1_R4_DRAIN_REMEDY.md:62–66`: R-1b "move the ledger I/O out from under the lock" was killed, as was R-1c "do not pump on every completion". :258 records a residual, `_deferred_retained_bytes()` under the lock in `_defer_locked`, "deliberately left alone". FF:2421–2423: R-1 did NOT change the one-attempt-at-a-time invariant or the pump call rate | ALREADY RECORDED. §3 D is R-1c. §3 C (reuse of the connection) is **not** R-1b (it leaves the reads under the lock), but it changes the cost R-1 was judged on |
| X-MINER-STALL (L-4) | `CLAUDE_CODE_REPORT_MINER_ADMISSION_STALL.md:136–148` ("a liveness defect, not contention … R-1 is not reopened"). :208: StripeComplete now submits a pump, "a harmless extra pump". `X-MINER-STALL_R1_BETA_REVIEW.md:92` ("MP-1 remains a contention diagnosis") | ALREADY RECORDED. It does not touch the inbound queue; `enqueue_staging`/`_pump_deferred` are byte-identical |
| F1/F2 active lease | `CLAUDE_CODE_INSTRUCTIONS_F1_F2_ACTIVE_LEASE.md:124–137` (§F1.5 pause grace) | ALREADY RECORDED; nothing on inbound saturation. Run O had 0 pauses |
| D6 refusal after a terminal | FF §2.36 :2045–2052 ("secondary symptom … D6 is EXONERATED — do not modify it"). `RUN4_LAUNCH_CHECKLIST.md:477` (`TC_INBOUND_SATURATION_TIMEOUT` = "staging or ingress capacity terminal") and :479 (`MinerIngressError … validated=False` = "a symptom. Look for the first terminal line before it") | ALREADY RECORDED. Run O's MinerIngressError is that symptom |
| Inbound bound | LEADS L-2 :92–100, :165 and BACKLOG §20 :495–534 ("bounded by COUNT and unbounded in BYTES … OPEN"; "confirm whether staging throughput degrades … in a way that raises queue occupancy"). `TODO_CONSOLIDATED.md:100` | ALREADY RECORDED as an open lead. Run O is **NEW** evidence for it |
| Full-range plans | `docs/drafts/NOTE_RUN_SCOPE_DECISION_20261118.md:8–12` ("run the full [0, 2³²) in one run later"); §2 :24–30 files only. `TFM_SESSION_HANDOFF_20261118_ADDENDUM.md:68` names only the survivor cap as a blocker. `S172_PHASE3_SURVIVOR_CAPACITY_CHARACTERIZATION.md` §4.1/§4.3 use 32-stripe arithmetic only | ALREADY RECORDED: nothing predicted an inbound or serve-loop limit at 64 stripes |

**NEW in this note:**
- The phase-3 serve-loop collapse from ≈40 to ≈2.5 frames/s, measured in L, M2 and O.
- The "remainder after collapse" explanation of why only the full range saturates.
- The pump/job ratio of 2.00.
- The list of changes in §3.

None of these is a ruling.

**Not searched:**
- `~/dashboard_work/`, which per `PROJECT_FILE_CATALOG.md:1051` holds `ATTEMPT6_REMEDIATION_DESIGN.md`,
  `R1_DRAIN_REMEDY.md` and `MP1_DRAIN_ATTRIBUTION.md`. The §8.2.4 / §2.4 / §8.5.1 design text for `S` and `D` lives there.
- ser8 archives, the private repository, the claude.ai project files, the rigs (and rig dmesg), `~/miner_staging` and
  the live ledger.
- `docs/exchange/from_beta/` beyond X-MINER-STALL and X-CACHE-POLICY-D6.
- Tests.

---

Scope and limits. Read-only: console, gate-12 TSV, coordinator source in the main tree (read only, HEAD `7c725c8`,
`miner/range_miner_coordinator.py` sha256 `b8e259b3…`, the same bytes the parity gate matched on the rigs). Nothing was run:
no tests, no replays, no simulations, because live runs P and Q share this CPU. No verdict on whether to change anything.
Run O = `distributed_config_t1_80ba5d40`. Comparison: Run L = `…_258866c1`, Run M2 = `…_9f5c4c73` (both 2026-11-18, half range).

Abbreviations: **frame** = one `sub_stripe_result` message. **Serve loop** = the coordinator's main thread.
**Pump** = `_pump_deferred`. Console = `/home/michael/runO2_evidence_20261119/runO2_20261119/console.log` (`O:` + line).

## 1 · What filled the inbound queue, and why it did not drain (Q1)

### 1.1 Timeline (Run O)

| Time | Event | Source |
|---|---|---|
| 18:41:52 | Stage 0 (phase 1). 64 stripes; 25 claimed, 39 queued. Each stripe = 34 frames | O:339, O:341 |
| 18:42:19–18:46:12 | Phase 1/2: inbound queue sits at 735–912 (high-water 920), then drains at each stage end | SERVE_LOOP_WINDOW lines O:347–O:589 |
| 18:46:52.6 | Stage 2 = **phase 3, java_lcg_hybrid**. 64 stripes × **68 frames** = 4,352 frames. Queue 0 | O:589–590, O:339 |
| 18:47:03–18:47:33 | Serve loop drains **369–463 frames per 10 s (≈40/s)**, queue 0 | windows |
| 18:47:43–18:47:53 | **Collapse:** 101, then 72, then **23–29 frames per 10 s (≈2.5/s)** for the rest of the run. Queue 247 → 490 → 819 | windows |
| 18:48:10.9 | Queue **1024/1024**. First `inbound_saturation_begin` (conn18); 25 of 25 connections within 9 s | O:712–761 |
| 18:48:10–18:51:16 | Queue stays at 1024. TSV: stripes in `staging` 22 → 26 → 16; `done` 132 → 146 (≈1 stripe / 13.8 s); `queued` 13 → 5 | concurrency TSV |
| 18:51:17.1 | conn25 (rrig6600c:gpu4) reaches its cumulative 180.09 s of blocked `put` → `READER_EXIT INFRASTRUCTURE_TERMINAL_EXIT`, `EMERGENCY_TERMINAL_REQUEST` | O:1235–1236 |
| 18:51:17.3 | `emergency_terminal` → `[F1/F2] TRIAL TERMINAL class=inbound_saturation_timeout` | O:1240–1241 |
| 18:51:17–18 | 7 more connections hit 180 s (8 requests in total; 1 acted on, 7 residual) | O:1237–1249, O:1272 |
| 18:51:21 | Step 1 then refuses ingress: `MinerIngressError … without a VALIDATED threshold provenance record` (D6). This follows the trial terminal; it is not a second fault — see §4 | O:~1330 |

146 of 256 stripes were done: 64 + 64 + 18 of phase 3. Rigs and GPUs: no fault lines in the console. [UNVERIFIED: rig dmesg not read.]

### 1.2 What the serve loop and staging were doing (run totals, `O:1273–1275`)

| Figure | Run O | Run L | Run M2 |
|---|---|---|---|
| Serve-loop wall (`loop_seconds`) | 572.1 s | 746.5 s | 743.0 s |
| `msg_total` (main thread handling frames) | 511.1 s | 266.0 s | 317.1 s |
| of which `staging` = `enqueue_staging` on the main thread | **476.2 s (83 % of the loop)**, 6,576 calls, **72 ms/call** | 242.1 s, 41 ms/call | 292.2 s, 49 ms/call |
| `pump` on each of 4 staging threads | 474–475 s each, 2,696–2,750 calls | 622–625 s each | 623–626 s each |
| Pump calls ÷ staging jobs | 10,895 ÷ 5,447 = **2.00** | 11,792 ÷ 5,895 = 2.00 | 11,955 ÷ 5,977 = 2.00 |
| `drain_stop_deadline` / `drain_stop_empty` | **1,416 / 271** | 714 / 2,340 | 852 / 2,045 |
| `inbound_qsize_high_water` | **1024** | 662 | 658 |
| `deferred_high_water` / bound in force | 1,649 / 4,377 | 1,721 / 2,201 | 1,618 / 2,201 |
| `inbound_saturation_seconds_total` / events | 3,223 s / 12,887 | 0 / 0 | 0 / 0 |
| `pump_liveness_probes_high_water` | 683 | 126 | 126 |
| `deferred_distinct_attempts_high_water` | 34 | 30 | 30 |
| `paused_high_water`, `pause_events`, `capacity_timeout_terminations` | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 |
| Last `SERVE_LOOP_ITERATION_PROFILE` frame classes | 6,576 sub-results, 162 stripe_complete, 600 heartbeats | 5,895 / 128 / 1,200 | 5,977 / 128 / 1,200 |

Phase 3 of Run O: per-stripe `stripe_rx` records show **68 of 68 frames deferred** for every completed stripe (e.g. 18:47:52–18:51:10).

### 1.3 The mechanism, link by link (file:line in `miner/range_miner_coordinator.py` at `7c725c8`)

1. **One reader thread per connection puts each decoded frame into one shared bounded queue** —
   `inbound = _queue.Queue(maxsize=1024)` at `:9616`. The put is retried in 0.25 s quanta
   (`INBOUND_PUT_QUANTUM_S`, `:457`; loop `:11286–11330`).
2. **Each failed quantum is charged to that connection, cumulatively**, by measured wait (`charge_inbound_saturation`,
   `:11316`). At `S` the reader exits with `RX_INFRASTRUCTURE_TERMINAL_EXIT` and discards its held frame (`:11322–11334`).
   `S = DEFAULT_INBOUND_SATURATION_BUDGET_SECONDS = DEFAULT_WORKER_ADMISSION_TIMEOUT = 180.0` (`:453`, `:417`).
   The emergency request ends the trial (`:9762`).
3. **The serve loop drains that queue for at most `D = 0.25 s` per iteration** (`DEFAULT_DRAIN_BUDGET_SECONDS`, `:433`).
   The deadline is tested before each message, so one slow message uses up the whole turn. Run O hit the deadline
   1,416 times and found the queue empty 271 times.
4. **For each frame the serve loop calls `enqueue_staging` synchronously** (`:11858`, `:11872`). That call takes
   `_admission_lock` (`:7588`) and, under it, makes three ledger reads: `get_stripe` (`:7589`), `get_shards` (`:7597`, via
   `_attempt_actual_bytes` `:4823`) and, in `_try_admit_locked`, a second `get_stripe` (`:4853`).
5. **Each ledger read opens a new SQLite connection and runs three PRAGMAs** (`MinerLedger._conn`, `:1207–1213`).
6. **Staging is serialised by attempt:** `_try_admit_locked` admits only one attempt at a time (`:4832–4856`,
   "Correction 6 … approach A"). Every frame of every other attempt is put in `_deferred` (`:7611–7615`). That is why
   phase 3 shows 68/68 frames deferred for each stripe.
7. **Every staging job runs the pump twice:** at the tail of `_run_staging_job` (`:8234`) and in the completion callback
   `_on_done` (`:7733–7739`). The measured ratio is exactly 2.00 in all three runs.
8. **The pump holds `_admission_lock` for a full scan of `_deferred`** (`:7895–8010`). It makes a ledger probe
   (`_attempt_live_locked`, `:4769–4781`: 1–2 new connections) per probe, plus `_prune_admitted_locked` (`:4783`). Its own
   docstring gives the per-pass probe count:
   *frames of the one admitted attempt + 2 × live non-admitted attempts + dead entries* (`:7768–7774`). It also says the
   pump "holds `_admission_lock` across a full scan of `_deferred` with a ledger liveness query per entry, while the serve
   loop needed that same lock once per sub-result" (`:7760–7762`, `:7813–7815`).

**Result:** four staging threads, each running two lock-holding pump passes per job, compete with the serve loop for one
lock. The serve loop's share fell to about one frame per 0.25–0.40 s (SLOW_MSG lines at the end show 0.35–0.39 s per
sub-result, `O:` e.g. 18:49:58 and 18:51:16). The GPUs kept sending at the pre-collapse rate. The queue filled in about
30 s, and every reader then accumulated blocked time until the first one reached 180 s.

**Measured, not inferred:** the collapse, the 72 ms/call average, the 2.00 pump/job ratio, the queue at 1024 for 183 s.
**[UNVERIFIED], needs a measurement:**
- The split of each pump pass between waiting for the lock and holding it. The `pump` charge wraps the whole method,
  lock wait included (`:7891–7893`).
- The per-probe SQLite cost.
- Why the collapse sets in about 50 s into the stage, in both L and O. The pump only reports a high-water of probes
  per pass, not a time series.
- Run O's 683-probe high-water does not fit the formula with 34 attempts and 68 frames (that predicts about 136). It may
  come from the clean-up pass after cancellation.

## 2 · Why the half-range runs do not hit it (Q2)

### 2.1 The collapse is the same in L and O; only the remainder differs

| | Run L (phase 3) | Run O (phase 3) |
|---|---|---|
| Stage start | 18:47:03 | 18:46:52 |
| Fast phase (frames / 10 s) | 381, 417, 411, 397 | 463, 390, 369, 427 |
| Collapse (~50 s in) | 18:47:53 → 26–32 per 10 s | 18:47:43 → 23–29 per 10 s |
| Frames in the stage | 32 × 68 = **2,176** | 64 × 68 = **4,352** |
| Frames taken in before collapse | ≈1,600–1,720 (deferred high-water 1,721) | ≈1,650 (deferred high-water 1,649) |
| Frames still to come after collapse | ≈450–570 | ≈2,700 |
| Room in the inbound queue (+1 held frame per reader) | 1,024 + 25 | 1,024 + 25 |
| Outcome | Queue peaked at 269 in phase 3 (662 in the run); then **no frames for 2 min 40 s** while staging drained `_deferred` (18:48:54–18:51:36 windows show 0–3 frames) | Queue full at 18:48:10 and stays full; 180 s budget spent at 18:51:17 |

Run L's phase 4 shows the same pattern (18:51:46: 703 and 948 frames per 10 s, then 27–30 per 10 s, queue peak 180).

### 2.2 Which quantity doubles

| Quantity | Half range (L/M) | Full range (O) | Changes? | Source |
|---|---|---|---|---|
| Stripes per stage | 32 | 64 | **×2** | `retention preflight` lines; `max_seeds / miner_stripe_size` |
| Queued stripes behind the 25 claimed | 7 | 39 | **×5.6**: workers never run dry during the collapse | `burst_exact … queued=` |
| Frames per stripe, phase 1/2 · 3/4 | 34 · 68 | 34 · 68 | no (`ceil(67,108,864 / cap)`, caps 2 M / 1 M AMD) | `:786–793`, `:599–601`, worker `:2948` |
| Frames per stage, phase 3/4 | 2,176 | 4,352 | **×2** | retention preflight |
| Survivors per frame, phase 3 | ≈23.5 k / 2,176 ≈ 11 average | ≈ same density (≈47 k / 4,352) | no per frame; **×2 per phase in total** (hence cap 25 k → 50 k, `7c725c8`) | `config_manifests/survivor_volume.cap` evidence text [per-frame figure is an average, UNVERIFIED per frame] |
| Concurrent attempts (cohort) | 25 (distinct-attempt high-water 30) | 25 (34) | about the same | BP summary |
| Staging backlog (`_deferred`) peak | 1,618–1,721 | 1,649 | **no** — it is capped by what the serve loop can move, not by the range | BP summary |
| Pump passes per job | 2 | 2 | no | §1.2 |

**The bottleneck is the serve loop's per-frame cost:** lock contention on `_admission_lock` with the pump, with
connection-per-query ledger reads inside that lock. It is not the readers: they block only because the queue is full.
It is not the staging writers' I/O: the staging threads spend about 83 % of their time inside `pump`, which includes lock
waits. It is not the disk, which has 306.85 GiB free and is not the limit (`O:309`). It is not ledger writes or
capacity: 0 capacity terminations, `deferred` well below its bound of 4,377.

**What doubles at full range** is the number of frames that arrive after the collapse while workers still have queued
stripes. At half range the stage runs out of work before the inbound queue fills. At full range it does not.
[UNVERIFIED: the point at which a half-range stage would also fail was not measured. L and M peaked at 662/1024.]

## 3 · Every setting or constant that would have to change (Q3)

"Parameter" = settable through the existing WATCHER manifest or CLI route, with no source edit. "Code" = a source edit
(and, for coordinator/worker files, a before-commit Beta review class under CLAUDE.md).

| # | Item | file:line | Current value | How derived | Parameter or code | What it would do · what else it risks |
|---|---|---|---|---|---|---|
| A | Inbound queue size | `range_miner_coordinator.py:9616` | `maxsize=1024` | Literal; Defect 6 (C3) comment gives no derivation | **Code** | Room for the post-collapse remainder (≈2,700 at full range, so ≥ about 2,750 plus margin, or derive from the stage's frame count as the deferred bound is). Risks: (i) RAM — each entry holds a decoded frame, inline payloads up to `INLINE_BYTE_LIMIT` 48 MiB (`range_miner_worker.py:1473`); typical phase-3 frames are small (≈11 survivors) [size per frame UNVERIFIED]. (ii) **Heartbeats and `stripe_complete` share this FIFO**: a deeper queue means longer waits before lease renewal is seen (`compute_lease_timeout` 300 s, `:618`). At 2.5 frames/s, 2,700 frames take ≈18 min, so lease expiry could become the next terminal [UNVERIFIED]. (iii) It does not change the drain rate; it only moves the wall |
| B | Saturation budget `S` | `:453` (from `:417`); threaded at `:12614–12616`, validated `:7204–7220`, read `:9472–9479` | 180.0 s | Derived from the worker admission window by design ("read here, never redefined") | **Code**: `run_trial_miner` accepts the kwarg, but `window_optimizer_integration_final.py` does not pass it, and the comment at `:12599–12606` says it is deliberately kept out of the WATCHER manifest | Gives readers longer before the trial dies. It only passes if the backlog clears before `S`. During Run O's saturation `_deferred` shrank by about (staging ≈5 jobs/s − ingest ≈2.5/s), so ≈1,650 entries would need ≈11 min [estimate, UNVERIFIED]. Same lease risk as A(ii). Breaks the recorded derivation link to the admission timeout |
| C | Pump lock-hold cost: SQLite connection per query | `MinerLedger._conn :1207–1213` (used by `get_stripe :2377`, `get_shards :2385`, `get_trial`) | New connection + 3 PRAGMAs per read | Not derived; the R-1 docstring names it as the cost driver (`:7807–7815`) | **Code** (ledger) | The candidate root cause of the slow per-frame cost. Risks: thread-safety of shared connections (sqlite3 objects are per-thread by default); WAL read snapshots; every MP-1/R-1 gate's cost identity; and the read-only ledger rules (`mode=ro&immutable=1` memory). **Touches the cost model R-1 was closed on, so the "do not reopen R-1" record applies — owner decision** |
| D | Pump scan under the lock, twice per job | `_pump_deferred :7743–8010`; callers `:7739`, `:8234` | Full scan, 2 passes per job, under `_admission_lock` | R-1 docstring states what was deliberately **not** done: no lock release mid-pass, no first-fit stop, **no coalescing of pump calls** (`:7871–7889`) | **Code** | Removing one of the two passes, or coalescing, roughly halves lock demand per job. The docstring says the `finally` is the F1 resume-credit point and "a dropped pump is a dropped resume credit". This is R-1c, which R-1 recorded as rejected (R1–R4 changelog :62–66). **Reopens R-1 — owner decision** |
| E | Serialised attempt admission | `_try_admit_locked :4832–4856` | One attempt stages at a time | Correction 6, Beta "approach A" (no circular wait between attempts) | **Code** | Allowing more than one admitted attempt would shrink `_deferred` and the probe count. Risks: the circular-wait argument that approach A was chosen for; the byte model (`:4838–4848`) |
| F | Hybrid seed cap (frames per stripe in phase 3/4) | coordinator `seed_cap_amd_hybrid :601` / `window_optimizer_integration_final.py:1835`; **worker** `--seed-cap-amd-hybrid` default 1,000,000 (`range_miner_worker.py:2948`); not in the WATCHER manifest (only the non-hybrid caps are, `agent_manifests/window_optimizer.json:39–40`) | 1,000,000 (68 frames per stripe) | Per-family VRAM caps (TB Q2) | **Code / launcher**: the worker default and the coordinator default must agree; the rig code-parity gate governs the worker | Doubling it gives 34 frames per stripe, so 2,176 per full-range stage, the same as a half-range stage today. Risks: VRAM on the 8 GB RX 6600 XT under the hybrid kernel, which is why the hybrid cap is half [UNVERIFIED that 2 M fits]; survivors per frame double; NPZ byte-identity (§11.A); retention preflight recomputes. Proof needs rigs |
| G | `staging_workers` / `staging_queue_depth` | config `:636`, `:641`; manifest `:57–58`, `:275–276`; CLI `window_optimizer.py:1952–1953` | 4 / 2 | Defect 4 / Defect 2 (C3): bounds in-flight RAM | **Parameter** | More staging threads means more pump passes competing for the same lock: likely **worse** [UNVERIFIED]. Fewer threads cuts lock demand but also staging throughput. Neither addresses the per-probe cost |
| H | `staging_deferred_max` | `:659`; manifest | None (derived 4,377 in phase 3) | S172-BP §2, Beta A | Parameter | **Not binding in Run O** (1,649 < 4,377). No change needed |
| I | `staging_capacity_timeout` | `:667` | 600 s | S172-BP §1.5 | Parameter | **Not binding** (0 pauses, 0 capacity terminations). No change needed |
| J | `staging_high_water_bytes` / files | `:604`, `:620` | 16 GiB / derived 13,056 | Staging-capacity amendment | Parameter | **Not binding**: preflight admitted 13,056 files (`O:339`) |
| K | Drain budget `D`, `DRAIN_COUNT_GUARD` | `:433`, `:439` | 0.25 s, 256 | ATTEMPT-6 §2.4 / §8.5.1 | Code (kept out of the manifest, `:12599`) | A larger `D` does not raise throughput when a single frame takes ≥0.25 s; the comment forbids lowering the guard |
| L | `cap_survivors_per_phase` | `config_manifests/survivor_volume.cap` | 50,000 (`7c725c8`) | Owner decision 2026-11-19 | Owner parameter | Already raised for full range; not involved in this terminal (the trial never reached the check) [UNVERIFIED that no cap check ran] |
| M | Run geometry (operational) | launcher `run4_full_launch.sh:468` `max_seeds` | 2^32 | Owner run plan | **Parameter** (no code) | Two half-range trials (as P and Q tonight) stay inside the measured envelope (L/M peaks 662/1024). It does not make a single full-range trial pass |

## 4 · Recorded design reasons for these limits (Q4)

| Limit | Recorded reason | Where |
|---|---|---|
| Inbound `maxsize=1024` | **None found.** It is only ever recorded as "unchanged" (`TB_SUBMISSION_ATTEMPT6_IMPLEMENTATION.md:294`). It is listed as an open lead: count-bounded, byte-unbounded (LEADS L-2, BACKLOG §20) | — |
| `S = 180 s` | Derived from the worker admission window, which "matches the PWC readiness window" (`_tcp_wait_ready(…180.0)`). It is cumulative and "a successful put never resets it". Persistent saturation is meant to be a **whole-trial infrastructure terminal, never a shed worker** | coordinator `:413–417`, `:450–453`; `TB_SUBMISSION_ATTEMPT6_IMPLEMENTATION.md:195–200`; FF:2245–2251 |
| `D = 0.25 s`, guard 256 | The design's declared default (§2.4 / §8.5.1). The guard exists because `while drained < 256` once let one iteration spend 940.971 s draining. Lowering it bare is forbidden. The claim is "structural, NOT a wall-clock guarantee" | `:431–438`; `SESSION_CHANGELOG_20260813_S180.md:22–23, :60`; FF:2263–2265 |
| `staging_workers=4`, `queue_depth=2` | They bound in-flight payload RAM (Defect 2) and keep staging off the dispatcher (Defect 4). **"Beta explicitly did not rule a new number — 'tune after measurement.'"** An Alpha note already called "6 concurrent staging jobs for a 25-worker fleet … the underlying imbalance" | `CLAUDE_CODE_INSTRUCTIONS_S172_STAGING_BACKPRESSURE_REMEDIATION.md:209–211`; `TEAM_ALPHA_DEFERRED_QUEUE_NOTE.md:135`; Corrections 2/3 |
| `staging_deferred_max` DERIVED | The constant 64 was deleted (S172-BP §2, Beta A). The bound is the conservative burst plus a margin. The recorded worked figure (2,201) is the 32-stripe one | FF:1103–1107; capacity doc §4.3 :280–285 |
| `staging_capacity_timeout` 600 s | "Same class as `staging_timeout`; flag the default for Beta" | remediation brief :144 |
| One attempt stages at a time (Correction 6, approach A) | It "trades a little cross-attempt parallelism for correctness — acceptable, since … **the miner's throughput bottleneck is GPU compute, not staging concurrency**." Run O measured the opposite in phase 3 (GPUs ≈40 frames/s, serve loop ≈2.5 frames/s). That is a measured fact against the recorded assumption; it does not change the ruling | `CLAUDE_CODE_CORRECTION6_S172_PHASE4_ADMISSION.md:77–79` |
| Pump on every completion, I/O under the lock | Kept on purpose in R-1. Moving I/O out (R-1b) and not pumping on every completion (R-1c) were both rejected | R1–R4 changelog :62–66; coordinator docstring `:7871–7889` |
| `staging_high_water_bytes` 16 GiB | Default; "NOT reachable" at the recorded geometry | `CLAUDE_CODE_INSTRUCTIONS_STAGING_DIR_FIX.md:151`; capacity doc §4.2 |
| Survivor cap 50,000 | Owner decision 2026-11-19 for this run (sum of the L and M phase-4 counts = 46,978) | `config_manifests/survivor_volume.cap`; `7c725c8` |

**In short:**
- `S`, `D`, the guard and the deferred bound have recorded derivations.
- `maxsize=1024`, `staging_workers=4` and `queue_depth=2` do not; for the staging pair Beta asked to tune after measurement.
- The serialised admission rests on a recorded throughput assumption that Run O's phase 3 measured as untrue for this geometry.

## 5 · Smallest change sets, in order, each with a test that needs no rigs (Q5)

These are options, not recommendations. Every code item touches a before-commit Beta class (coordinator). C and D also
touch the R-1 closure, which needs the owner. **All tests below are CPU-only and must wait until the owner says the
live runs are over.**

0. **No change.** Full range as two half-range trials (operational, item M). *Proof already exists:* L and M (clean,
   inbound peaks 662/658 of 1024). Limit: the margin at half range is about 360 frames and has not been measured
   under worst case.
1. **Measure before changing** (instrumentation only, no behaviour change). Split the `pump` charge into lock-wait and
   lock-hold. Record per-pass probe count and duration as a time series, and per-read SQLite open time.
   *Test:* a synthetic CPU harness that loads `_deferred` with Run O's shape (34 attempts, about 1,650 frames, one
   admitted attempt, a temporary ledger), runs 4 pumping threads and the serve-loop `enqueue_staging` call, and checks
   that the new counters add up to the existing `pump`/`staging` totals. This settles whether C or D is the cost.
2. **C alone: a reused read connection per thread for ledger reads.** *Test:* the same harness before and after, with
   `enqueue_staging` p50/p99 and pump-pass duration at 1,650 deferred entries. The pass condition is a per-frame
   serve-loop cost below the fast-phase arrival interval (1/40 s = 25 ms) at Run O's population, as a stated target.
   The existing MP-1/R-1 gates (G8b probe identity, G8g) and the coordinator suites must also stay green. Suites must
   run in sequence (memory: concurrent runs flake G-VAL-6).
3. **A (and optionally B) as a safety margin.** Derive the inbound size from the stage's conservative frame count, as
   the deferred bound is derived (`staging_burst_bound_conservative`, `:943`). *Test:* a synthetic serve-loop test with
   25 fake connections sending 4,352 small frames at 40 frames/s against an artificially slowed `enqueue_staging`
   (≈0.3 s/frame). It asserts no saturation terminal, and it measures the longest wait from heartbeat enqueue to
   dequeue against `compute_lease_timeout`. That exposes the A(ii) lease risk.
4. **F (hybrid cap 1 M → 2 M) only if 1–3 are rejected.** *CPU test:* the retention preflight and
   `staging_burst_bound_conservative` give 34 frames per stripe and 2,176 per full-range stage. *Not provable without
   rigs:* VRAM headroom and NPZ byte identity.

D and E are listed for completeness. Each one reopens a recorded design argument (R-1 non-coalescing; Correction 6
approach A). Neither is in the smallest set.
