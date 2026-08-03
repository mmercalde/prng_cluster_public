# PROJECT FILE CATALOG — INTENT-INDEXED

**TFM / `distributed_prng_analysis`. Regenerated 2026-08-03 on VM101 (`192.168.3.177`) as `michael`, venv `~/venvs/torch`, at HEAD `9e79a26`.**
**Authority:** `docs/CLAUDE_CODE_INSTRUCTIONS_PROJECT_CATALOG_REGENERATION.md` (REV2).
**Replaces** the catalog compiled 2026-02-04, which was name-indexed and six months stale.

> ## How to use this file
>
> **Every entry answers *"what question does this document settle?"* — not *"what is this file called?"***
>
> This catalog exists because two failures happened on 2026-08-02, and **neither was caused by
> missing information.** Alpha claimed the three-lane CRT test was undocumented while the answer sat
> in `CHAPTER_2_BIDIRECTIONAL_SIEVE.md` §6, committed. Alpha separately nearly submitted a finding to
> Team Beta that **Beta had already ruled on** in `TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md`, also
> committed. `ls docs/` was available both times and did not help, because a filename does not say
> what question a document answers.
>
> **Read §1 (the governance trail) before submitting anything to Beta.** It is the section that was
> missed.
>
> **This catalog is a snapshot, and a snapshot expires.** Anchors here were read on 2026-08-03.
> Re-verify before acting on any of them.

**Coverage (execution proof).** `docs/` contains **562 files** — 552 at top level plus 10 across
three subdirectories (`audit/`, `proposals/`, `phase6_evidence/`). **All 562 are accounted for:
394 indexed individually** (372 top-level `.md`, 12 top-level non-`.md`, 10 subdirectory files)
**and 168 session changelogs summarised as a group** under §3.11, as REV2 §1.3 permits.

**Audit claim scope: repo-scoped.** This catalog indexes what is committed to
`/home/michael/distributed_prng_analysis` at HEAD `9e79a26`. **Host state (systemd units, cron,
deployed uncommitted files) and the pre-repository archives on ser8 are OUT OF SCOPE and are not
implied by anything below.** Gitignored files are named where relevant but their contents are not
catalogued.

---

# 1. THE GOVERNANCE TRAIL

**The single most valuable section.** Every `TB_*`, `TEAM_ALPHA_*` and `TEAM_BETA_*` document, what
it settles, and — where they exist — the ruling and the implementation commit it pairs with.

**Disposition vocabulary:** `RULED` · `AWAITING RULING` · `SUPERSEDED BY <name>` · `RECORD` (a
notice or forensic record that asked for nothing).

**One standing caution.** A ruling *request* is the least reliable source on whether the thing it
asked for happened. Where the implementation commit is named below, it was verified by `git log` in
this pass. Where it is not, **this catalog does not claim one way or the other** — that is an audit,
and this is an index.

## 1.1 Team Beta rulings and ruling requests (`TB_*`)

| file | what it rules on / asks | disposition | still binding? |
|---|---|---|---|
| `TB_BINDING_RULINGS_S172_PHASE4.md` | Beta's binding answers to the two Phase-4 requests: **(1)** the worker must reject a stripe assignment that **omits** `dataset_sha256` as well as one that mismatches — Option C, explicitly **overriding** Alpha's recommended compare-when-present, because compare-when-present lets a coordinator regression silently bypass identity; **(2)** the L7 abort-discard interface. | **RULED**, 2026-07-18. Pairs with both requests below; implemented via `CLAUDE_CODE_INSTRUCTIONS_S172_PHASE4.md` Stage 0 + Stage 4. | **YES** |
| `TB_RULING_REQUEST_BLOCKER6_DATASET_SHA_S172_PHASE4.md` | Narrow question: what does the **worker** do when a payload arrives with **no** `dataset_sha256` key at all — tolerate absence, or fail closed? | **RULED** → Option C (reject on absence), in `TB_BINDING_RULINGS_S172_PHASE4.md` Ruling 1. | superseded by the ruling |
| `TB_RULING_REQUEST_L7_ABORT_DISCARD_S172_PHASE4.md` | Sync `abort_trial()` whose return guarantees Phase 5 holds no trial-owned path, vs async `TrialAbort → TrialAbortAck`. Explicitly a selection, not a scope change; P2, non-blocking for Stages 0–3. | **RULED** in `TB_BINDING_RULINGS_S172_PHASE4.md`. | superseded by the ruling |
| `TB_RULING_REQUEST_CPU_PARALLELISM_S175.md` | Should host-side survivor collection + 22-array NPZ assembly be parallelised across the coordinator CPU, or inherit the single-threaded GIL-bound S152 pattern? I.e. does the miner *remove* PWC's high-survivor throughput collapse or merely *relocate* it from transport to assembly? | **RULED** — the binding S175 ruling is the stated change driver for `PROPOSAL_S172_RANGE_MINER_v1_4_5.md` (remote spool staging, staged A+C parallel assembly, high-survivor acceptance, three-way verification). Realised as D5 `process_sharded`. | **YES** — but D5 is **available and UNPROMOTED** |
| `TB_RULING_REQUEST_D5_EXCEPTION_PRECEDENCE.md` | D5's read-all-then-merge is **not** the semantics-preserving no-op it was specified as: the pre-D5 `assemble_trial` interleaves read and merge, so a duplicate in an earlier-order spool raises **before** a later spool is read. A vs B; Alpha recommended B. | **RULED → Option B** (preserve the original deterministic precedence; spool-read errors returned as typed data and replayed by the parent in manifest order). Recorded in `CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D5_REV2_ADDENDUM.md`. | **YES** |
| `TB_RULING_REQUEST_IPC_SERIALIZATION_S150.md` | High-survivor trials collapse **27×** (≈1.99M s/s → ≈73K s/s) purely from result-payload serialisation — workers return full survivor records as JSON over stdout. Should the IPC path change? | **RULED → Option A / `slim_v1`.** Design in `IPC_SERIALIZATION_FIX_IMPLEMENTATION_GUIDE_S150.md`; applied by root `apply_s150_slim_v1_ipc.py`. | superseded in practice by the RANGE-MINER pivot |
| `TB_RULING_REQUEST_ROCR_ISOLATION_S149.md` | AMD rigs at ~1,050 seeds/sec vs an ~787K/GPU baseline, **0% GPU utilisation with 4 live workers**, 1.3% worker CPU. Is per-worker `ROCR_VISIBLE_DEVICES` isolation the fix, and may `worker_pool_size` exceed the S146 hard cap of 4? | **RULED** (approved — the file itself carries no ruling text). Implementation evidence: `apply_s149a_rocr_isolation.py`, `verify_s149a_rocr_fix.py`, `SESSION_CHANGELOG_20260321_S149.md`. | historical — pool-size limits later re-derived |
| `TB_RULING_REQUEST_STEP2_v4_1_OBJECTIVE.md` | **The WSI v4.0 tautology.** Smoke test returned WSI = 0.9997 on trial 1 because the scoring formula contains `quality = fwd*rev` as its dominant term (w3 ≈ 0.82) — the objective measures itself. Asks which NPZ fields are legitimate independent quality signals. | **RULED** → v4.1 deployed cleanly (19/19 checks). **SUPERSEDED BY `TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md`** in the same session. | superseded |
| `TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md` | ★ **The document Alpha nearly re-reported to Beta as a new finding on 2026-08-02.** v4.1 deployed and the objective still cannot optimise: `sel_score = 0.0000` on **every** passing trial, because `bidirectional_selectivity` sits at floor (98.8%) and carries no variance. Proposes swapping it for `npz_bidirectional_count`. | **RULED** → v4.2; lineage continued to v4.3 and v4.4 in the same session (`S107_session_log.md`). **Read this before reporting any Step-2 objective blindness.** | **YES** — it is the diagnosis behind the approved v4.1→v4.2→v4.3 remediation |
| `TB_RULING_S176_WATCHER_KPI.md` (827L) | Beta's review of the S176 WATCHER retrain-KPI findings, independently checked against live public `main` at `0c3166a`. Executive ruling + item-by-item disposition. | **RULED.** Follow-up work items scoped in `CLAUDE_CODE_BRIEF_S176_FOLLOWUP_v1.md`. | **YES** |
| `TB_RULING_S177_KPI_GOVERNANCE.md` (661L) | Beta's review of KPI governance proposal v1.0 + analyzer v2. | **RULED: CONDITIONAL APPROVAL — REVISION REQUIRED BEFORE IMPLEMENTATION.** Eight proposal blockers + six analyzer fixes. Resubmission brief: `CLAUDE_CODE_BRIEF_S177_RESUBMISSION_v1.md`. | superseded by S178 |
| `TB_RULING_S178_KPI_GOVERNANCE.md` (591L) | Beta's review of proposal **v1.1** + analyzer **v2.1**. | **RULED: ARCHITECTURE APPROVED IN PRINCIPLE — FOUR MANDATORY AMENDMENTS.** "A complete rewrite is not required." Follow-up: `CLAUDE_CODE_BRIEF_S178_FOLLOWUP_v1.md`. | superseded by S179 |
| `TB_RULING_S179_IMPLEMENTATION_AUTH.md` (552L) | Beta's binding review of the **v1.2 addendum** + analyzer **v2.2**. | **RULED: APPROVED FOR IMPLEMENTATION WITH THREE BINDING CODE-LEVEL CONDITIONS.** Whether implementation has landed was **not established in this pass** — see §7 gap 8. | **YES** — this is the live authority for KPI governance |
| `TB_SUBMISSION_S159G_RIG6600_CRASHES.md` | P0: rrig6600 crashes consistently under ZMQ multi-rig runs; rrig6600b never has. Netconsole capture of the `GCVM_L2_PROTECTION_FAULT` escalation. Positive finding: ZMQ SQLite lease expiry recovered the run — the coordinator architecture is crash-resilient. | **RECORD** + P0 ruling requested. Answered by the two updates below. | historical (pre-Proxmox, pre-miner) |
| `TB_UPDATE_S159G_ENV_PROPAGATION.md` | Follows the P0 ruling to verify `HSA_ENABLE_SDMA=0` actually reaches live worker PIDs. It does not: rrig6600's workers carry **one** ROCm variable; the stable rigs carry four. | **RECORD** — root cause identified as environment propagation, not kernel logic. | historical |
| `TB_FINAL_UPDATE_S159G_ROOT_CAUSE_CONFIRMED.md` | Confirms the above at PID level across all three rigs; fix identified, implementation pending at time of writing. | **RECORD** | historical |
| `TB_UPDATE_S162_OPTION_B_RESULTS.md` | Does `AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3` on rrig6600c delay the crash? **No** — crash at ~1–2 min of trial runtime, no meaningful improvement over prior 18 s / 50 min. | **RULED** on the diagnostic; feeds `PROPOSAL_S162_RRIG6600C_CRASH_ROOT_CAUSE_v1_0.md`. | historical |
| `TB_Incident_Report_rrig6600c_S163KARG.md` (305L) | Forensic incident report: the rrig6600c fatal crash during S163-KARG trial 3 is a **GPU virtual-memory PTE invalidation while kernels are mid-execution across devices** — **distinct** from the earlier int32/int64 kernel-arg mismatch the KARG patch fixed. Also records a secondary silent crash on rrig6600b GPU1 that self-recovered. | **RECORD**, classified BLOCKING at the time. | historical — the class of failure that motivated RANGE-MINER |
| `TB_SUMMARY_S163.md` | Session-scope summary of three items: the NPZ `UnboundLocalError` (a duplicate `import numpy` shadowing the module-level name), `free_all_blocks()` removal, and staged 500K→2M validation. | **RECORD** | historical |
| `TB_UPDATE_SELFPLAY_REFRAMING_2026-07-28.md` | **Correction of framing, not of architecture.** Self-play is a *discovery front-end* to an already-built grade→attribute→concentrate→reinforce loop — not "the" learning system, as the proposal treated it. Issued **before** REV2.1 was drafted so the addendum would be written against an accurate picture. | **RECORD** + three confirmations requested. | **YES** — governs how self-play may be described |

## 1.2 Team Alpha submissions (`TEAM_ALPHA_*`)

Grouped by the thread each belongs to; within a thread, in sequence.

### The S172 Phase-4 coordinator review chain — seven rounds

Beta rejected Phase 4 six times. Each Alpha review record pairs with the correction brief Beta's
rejection produced. **Read this chain before assuming an adversarial gate is excessive** — three of
the six defect classes were in functions Alpha had already read line-by-line and passed.

| file | what it records | pairs with | disposition |
|---|---|---|---|
| `TEAM_ALPHA_REVIEW_S172_PHASE4.md` | Rev-1: 63-line stub → 2,141-line coordinator; 36/36 brief gates. | `CLAUDE_CODE_INSTRUCTIONS_S172_PHASE4.md` | SUPERSEDED BY REV2 |
| `..._REV2.md` | Beta rejected on **one** blocker: `run_trial_miner()` had **no real server** — gate 20 exercised a harness-built loop, not the coordinator's own serve path. Alpha's rev-1 had accepted "live serve loop out of scope" as legitimate; it was the missing deliverable. | `CLAUDE_CODE_CORRECTION_S172_PHASE4_SERVE.md` | SUPERSEDED BY REV3 |
| `..._REV3.md` | Six release-blocking serve-path/ledger/wiring defects. **Accountability note:** rev-1/rev-2 traced the intended path and pattern-matched instead of constructing the adversarial case. | `CLAUDE_CODE_CORRECTION2_S172_PHASE4_SIX_DEFECTS.md` | SUPERSEDED BY REV4 |
| `..._REV4.md` | Six async-staging / socket defects: orphan late-write, unbounded-queue RAM, capacity deadlock, reconcile→matrix routing, hybrid hash-mismatch retry, silent/partial socket. 54/54. | `CLAUDE_CODE_CORRECTION3_S172_PHASE4_ASYNC_SOCKET.md` | SUPERSEDED BY REV5 |
| `..._REV5.md` | Four overload / heterogeneous-worker freezes Beta **reproduced**. 59/59. | `CLAUDE_CODE_CORRECTION4_S172_PHASE4_OVERLOAD.md` | SUPERSEDED BY REV6 |
| `..._REV6.md` | ★ **The theme worth carrying forward:** two of three prior failures were gates checking **bookkeeping** (registry entry count, a static byte estimate) instead of the **real resource** — so the gate passed while the resource leaked. Fixes now assert bytes on disk and live `threading.enumerate()`. | `CLAUDE_CODE_CORRECTION5_S172_PHASE4_REALBOUNDS.md` | SUPERSEDED BY REV7 |
| `..._REV7.md` | The last blocker: admission used **two contradictory byte models** — actual advertised `size_bytes` in `enqueue_staging` vs a static `expected_substripes × 48 MiB` in `_try_admit_locked`. Fixed by Beta's Approach A: `_try_admit_locked` becomes a pure serialisation gate. | `CLAUDE_CODE_CORRECTION6_S172_PHASE4_ADMISSION.md` | **RULED — Phase 4 closed at 63/63** |

### The S172 Phase-5 deliverable chain (D0 → D6.2)

| file | what it settles | disposition |
|---|---|---|
| `TEAM_ALPHA_REVIEW_S172_PHASE5_D0_REV2.md` | D0 correction round: `INSERT OR IGNORE` silently accepted a **conflicting** trial context → replaced with compare-and-insert in one DB transaction under the write lock. D0 9/9. | SUPERSEDED BY REV4 |
| `TEAM_ALPHA_REVIEW_S172_PHASE5_D0_REV4.md` | Round 4: `window_size`/`offset` **fabrication** in `run_trial_miner` removed, plus a third-order coercion site and Beta's vacuity catch on gate B4. D0 12/12, verified **from the extracted archive**, not the working tree. | **RULED — D0 closed** |
| `TEAM_ALPHA_REVIEW_S172_PHASE5_D1_0.md` | D1.0: workflow bidirectionality + abort/commit terminal-race correction. AST-verified scope: exactly two tracked modifications. | RULED |
| `TEAM_ALPHA_REVIEW_S172_PHASE5_D1_1.md` | D1.1: the shared four-population assembly engine + concrete `Phase5Sink`. Requests the standing gate-22 rule be extended from "the deliverable's new *harness* path" to "new file paths". | RULED |
| `TEAM_ALPHA_REVIEW_S172_PHASE5_D2.md` | D2: directional uniqueness enforced at **both** layers — producer overlap rejection and a Phase-5 fail-closed probe. No production change. | RULED |
| `TEAM_ALPHA_REVIEW_S172_PHASE5_D3.md` | D3: the shared backend-neutral **24→22 columnizer** + independent structural validator. Legacy paths deliberately left intact and in use. | RULED |
| `TEAM_ALPHA_REVIEW_S172_PHASE5_D3_0.md` | D3.0: canonical encoding seam + rectangular empty NPZ. **The untouched-accumulator claim holds structurally, not by assertion** — merge/supersede/backfill/sort are outside the diff entirely. Four follow-ups escalated. | RULED — but see `TEAM_ALPHA_D3_0_B_AND_ITEM1_NOTICE.md` |
| `TEAM_ALPHA_REVIEW_S172_PHASE5_D3_25.md` | D3.25: mode-preserving backend result contract + canonical candidate-ingress normalisation. **The deliverable that ends the "PWC/ZMQ untouched" era.** | RULED |
| `TEAM_ALPHA_REVIEW_S172_PHASE5_D3_5.md` | D3.5 (REV2, completed): the shared run finalizer. Byte-identity of the three components the retired migration gates certified, proven twice over. | RULED |
| `TEAM_ALPHA_REVIEW_S172_PHASE5_D3_5_B.md` | D3.5-B: Seed-Domain v1.1. **[R1]** Alpha had asserted the recursive chain walk compares `prng_base`/`schema_version`/`canonical_map_hash` **without verifying**; it contained zero such occurrences. | RULED |
| `TEAM_ALPHA_REVIEW_S172_PHASE5_D4.md` | D4: `serial_reference` behind the frozen two-backend interface. No production module modified; every must-not-modify file SHA-verified. | RULED |
| `TEAM_ALPHA_D5_ADVANCE_TO_BETA_REV3.md` | D5 blocker resolved: **lossless dual-encoding** for seed projection — fast `int64` path iff every seed satisfies −2⁶³ ≤ s ≤ 2⁶³−1, else the whole spool falls back to `signed_bytes`. Boundary confirmed at the condition itself, not from the report. | **RULED — D5 closed** |
| `TEAM_ALPHA_D6_ADVANCE_TO_BETA.md` | D6: production integration adapter — real miner trial → coordinator/staged-spool lifecycle → selected backend → `MinerTrialAssembly` → shared `finalize_run` → certified generation → miner candidates into the Step-1 accumulator, with an unchanged `TestResult` shape. | RULED (with correction) |
| `TEAM_ALPHA_D6_CORRECTION_RETURN_TO_BETA.md` | D6 held by Beta on one correctness blocker: **the miner ignored configured thresholds and filtered at a hardcoded `0.25`**, so the optimizer certified results for a configuration it had not requested. Correction adds a single canonical threshold path with requested/payload/effective provenance. Alpha states plainly it **under-disclosed the writer seam** the prior round. | **RULED** → `2be51d5` |
| `TEAM_ALPHA_D6_1_RETURN_TO_BETA.md` | D6.1 + a **scope-change ruling request**: a fifth defect made the briefed in-place flush repair unsafe. The checkpoint targeted relative `bidirectional_survivors_all.npz` / `..._binary.npz` — which since D3.5 are **finalizer-owned compatibility symlinks** that make `finalize_run` raise `PublicationError` if a regular file appears there. D1 reproduced: numpy 1.22.0 wrote `...flush.tmp.npz`, `os.replace` raised `FileNotFoundError`. **Incremental durability had never existed.** | **RULED** — relocate to `.s172_checkpoint/<run_id>/` |
| `TEAM_ALPHA_D6_2_IDENTITY_ADDENDUM.md` | Raised **mid-review** because D6.3 depends on a field D6.2 was already positioned to add: the checkpoint directory and the published generation use **two different run identities**, defined ~2,150 lines apart in the same file. | **RULED** |
| `TEAM_ALPHA_D6_2_CERTIFICATION_SUBMISSION.md` | D6.2 at `f7583bc`: 24-field canonical flush, `_FLUSH_CLEAR_IN_MEMORY = True`, finalizer fed reconstructed cumulative state. **The S166 OOM protection is real for the first time.** 29/29 gates, 377 assertions, 23/23 mutants. | RETURNED by Beta |
| `TEAM_ALPHA_D6_2_RECERTIFICATION_SUBMISSION.md` | Bounded repair at `18a2419`. ★ **Method note worth keeping:** Beta found blocker 1 by reading the live objective; Alpha had read *the report's description of it*. This submission was reviewed **from the diff, not the report**. 31/31 gates, 25/25 mutants, Phase 4 restored to 63/63. | **RULED — D6.2 CERTIFIED** at `18a2419` |

### Phase 6 — dataset authority, ROCm parity, certification

| file | what it settles | disposition |
|---|---|---|
| `TEAM_ALPHA_PWC_COMPARATOR_SCOPE_CORRECTION.md` | ★ **Scope drift, named and corrected.** The RANGE-MINER rule was **interface compatibility** — "the remaining steps must not be able to tell which engine produced the data" — a statement about the 22-array shape and its consumers, **not about values**. It had silently become "prove RANGE-MINER produces output identical to PWC", with PWC designated authoritative comparator. | **RULED** — PWC **retired from certifying authority**; flag-selectable, non-certifying diagnostic |
| `TEAM_ALPHA_PROPOSAL_PHASE_6_0_ROCM_SMOKE.md` | Proposes a cheap single-rig ROCm smoke *before* full Phase 6, because everything certified through D6 ran on **one RTX 3080 Ti under CUDA** — and the AMD rigs are the hardware class the whole rearchitecture was motivated by. Also records that the Proxmox migration blocker is cleared. | **RULED — approved**, with a required parity addition |
| `TEAM_ALPHA_PHASE_6_0_RETURN_TO_BETA.md` | ★ **The headline result.** The miner production path ran on an RX 6600 XT under ROCm and produced a **byte-identical certified artifact** to the CUDA run — `artifact_sha256 0e0092fe…c4b0` identical across the D6 release-grade CUDA generation, the 6.0 CUDA control and the 6.0 ROCm run; 22/22 arrays equal; forward 398,156 / reverse 383 / bidirectional 319. | **RULED** |
| `TEAM_ALPHA_DATASET_LIFECYCLE_FINDINGS.md` | Four findings that invalidate one field of the provisioning contract: **a fixed `expected_sha256` cannot work**, because the dataset is not immutable (append changes the digest every scrape; rewrite does not even guarantee prior content is stable). The invariant is **fleet consistency**, not immutability. | **RULED** — five rulings issued |
| `TEAM_ALPHA_APPEND_ONLY_SIMPLIFICATION.md` | Accepts all five lifecycle rulings and proposes a producer-side simplification: **rewrite mode will not exist.** Michael's decision; rewrite was a one-time bootstrap need, not an operating requirement. Publication model becomes **dated immutable files**. Also records `daily3scraper.service` now `disable --now`, unit retained. | **RULED** |
| `TEAM_ALPHA_PUSHBACK_ORDERING_AND_THRESHOLD_REGRESSION.md` | ★ **Alpha contesting a Beta ruling with evidence, and winning.** Ruling 20's premise — that the combined array is one generator's output stream — is factually wrong: the CA draw procedures select equipment **per session**, so midday and evening are different PRNG streams. Also reports a live threshold regression that outranks both rulings. | **RULED** — Ruling 20 withdrawn; combined-session sequential sieve now non-certifying |
| `TEAM_ALPHA_PHASE_6_P0_SUBMISSION.md` | P0: dataset version one published in place — immutable `daily3-<UTC>Z-<sha256[:12]>.json` (1,380,711 bytes, 18,068 records, sha `513648…68f6`, digest **re-derived from disk after writing**) + atomic pointer manifest; `daily3.json` untouched. Discloses that **Alpha proceeded without prior plan approval** and names two decisions for explicit ratification. | **RULED** → `131787d`. Beta granted the procedural exception and stated it is **not precedent** |
| `TEAM_ALPHA_PHASE_6_P0_5_SUBMISSION.md` | P0.5, the behavioural cutover: pointer resolution, one-time run-start freeze (manifest/version, absolute path, sha256, size, record count), absolute-path dispatch, fail-before-first-worker-dispatch, per-node provisioning with **on-target** verification, run provenance. | **RULED** → `d4ff1e4`; one closure condition → `8600e75` |
| `TEAM_ALPHA_FLEET_STATE_SUBMISSION.md` | The fleet-state investigation Beta required before ruling on Q1 — **and a finding that outranks Q1**: `assign_stripes`, `_dispatch_pending`, `process_lease_expiry` **and** the stage advance are all behind one `len(eligible) >= expected_workers` guard, while `serve_timeout` defaults to `None`. A worker loss crossing the threshold means the trial **neither completes nor fails**. | **RULED** → repaired `ee0db06` |
| `FLEET_SUBMISSION_CORRECTIONS.md` | Two **wording** corrections Beta required in the submission above. Both are corrections, not reversals — Beta confirmed the findings. The audit itself repeats neither error. | RULED |
| `TEAM_ALPHA_EXECUTION_SET_AND_CHAPTER2_SUBMISSION.md` | The **Resolved Execution Set** built (34/34 gates, 5/5 mutants) — one frozen run-scoped fleet authority created after backend and rig-profile selection and **before** dataset verification, GPU verification, coordinator construction and dispatch. Both entry points verified live to the same `set_id`. Plus Chapter 2 restored. | **RULED** → `63e627f`; Beta **withheld** Phase-7 closure pending two repairs |
| `TEAM_ALPHA_ADMISSION_BINDING_SUBMISSION.md` | ★ **A retraction in place.** Alpha claimed the freeze-after-read ordering could not be violated; **Beta's refutation was correct** — `active_execution_set()` incremented `_READS` only inside `if _ACTIVE is not None`, so a consumer could read `None`, take the legacy path, and a freeze could still follow. Counter now unconditional. `expected_workers` now sourced from the frozen set's effective `admission_count`. 20/20 · 34/34 · 16/16 · 63/63. | **RULED** → `eff6616`; **Phase-7 closure granted** |
| `TEAM_ALPHA_BOUNDED_PHASE_6_SUBMISSION.md` (731L) | **The certification the whole PWC → RANGE-MINER pivot was built toward.** Wall A (interface + consumer) PASS, Wall B (determinism + platform) PASS, Miner Known-Answer Transfer Gate PASS, RandomSampler control arm PASS **(NON-CERTIFYING)**, non-regression 22/22 suites exit 0. Two items flagged for Beta's decision rather than claimed complete. | **RULED — CERTIFIED and CLOSED** → `d98298c` |
| `TEAM_ALPHA_WALL_C_SUBMISSION.md` | ★ **"Alpha scoped Wall C as new work. That was wrong."** Known-answer validation is documented **pre-repository practice** — `prng_registry.py` is in the initial commit (`0101306`, 2025-11-29), the oldest session record is S73 (2026-02-08), so **the repository's history begins after the work was finished and cannot evidence it either way.** The evidence survives in Michael's archive. | **RULED — Wall C struck** as a Phase-6 precondition |
| `TEAM_ALPHA_D3_0_B_AND_ITEM1_NOTICE.md` | One ruling request + one disclosure: **D3.0-B was never completed and Phase 6 certified anyway.** No commit completing it exists, and the defect is live at HEAD — `convert_survivors_to_binary.py:184` still silently defaults a record carrying **neither** `prng_type` **nor** `prng_base` to `'java_lcg'`. | **RULED — D3.0-B accepted as OPEN**; Beta disclosed the governance error unprompted |
| `TEAM_ALPHA_PHASE7_LAUNCH_NOTICE.md` | **Not a ruling request** — Michael, as owner, ordered the Phase-7 soak to launch. Records the disposition of Beta's three points and the one criterion that will report `UNAVAILABLE` rather than `PASS`. **Alpha will not invoke the legacy converter until D3.0-B closes.** | RECORD — owner decision |
| `TEAM_ALPHA_6P2_TRANSITION_RULING_REQUEST.md` | The one 6-P2 item Alpha will not decide: `daily3.json` ends **mid-day** (18,068 records, terminal record `2026-02-26 midday`, evening absent), so under Beta's ordering the next scrape's `2026-02-26 evening` sorts *before* the terminal record — **publication one is a backfill, not an append**, and REV3 §2.3 halts it with `NON_APPEND_INSERTION_REQUIRED`. | **AWAITING RULING** — blocks 6-P2 publication one |
| `TEAM_ALPHA_SCRAPER_RECENT_SAFETY_NOTICE.md` | ★ **Operational hazard, held independently of 6-P2's timeline.** The scraper's `--recent` path **destroys the dataset**: `main()` writes only what it just scraped over `daily3.json` and never reads the existing file. **Note the audit surface** — `daily3_scraper.py` has **never been in git history and is not gitignored**; no repo-scoped audit could previously have seen any of it. | RECORD — safety notice |

### Audits, corrections and design submissions

| file | what it settles | disposition |
|---|---|---|
| `TEAM_ALPHA_CHAPTER_1_AUDIT_SUBMISSION.md` | Chapter 1 audit findings + a scope request. **Sentinel FAIL — on the chapter, not the audit.** 41 claims: **9 accurate · 19 stale · 5 superseded · 7 contradicted-by-code · 1 unverifiable.** A fifth dead dimension, and the first **operator-facing** one: `--forward-threshold`/`--reverse-threshold` are declared and never referenced after `parse_args()` — a silent no-op that still reports success. | **RULED** → P0 tranche `ddd2ac8`; closed `ef4b1c6` |
| `TEAM_ALPHA_CHAPTER_2_RECOVERY_SUBMISSION.md` | ★ **A Beta ruling made on incomplete information, corrected with forensics.** Beta ruled Chapter 2 "MISSING CORE CONTENT / repair type: reconstruction". It was not missing: 743 lines exist at `d14dcdd`, destroyed by a stale-copy overwrite at `248e48c` ("chore: move CHAPTER docs to docs/ folder", −709). | **RULED — re-scoped to restore-and-audit** |
| `TEAM_ALPHA_SKIP_SEMANTICS_SUBMISSION.md` | ★ **The premise of the earlier audit was false.** `HYBRID_SKIP_BOUND_AUDIT.md:318` recorded hybrid skip semantics as "unspecified"; they are specified in two committed documents. This was the **fourth falsified absence claim** of that session — and the sharpest, because the audit's own VIR-6 declared a full-tree grep for `skip_min` that **reached the exact line and did not read it.** Michael's decision: stop discarding `skip_sequences`, revive the three dead skip-shape features. **Output-statistic reading only — no kernel change, and NOT the input-bound question.** | **RULED** — skip-OUTPUT work approved; input bounds remain open |
| `TEAM_ALPHA_AUTONOMY_CONTROL_SURFACE_SUBMISSION.md` | Four autonomy chains, four ruling requests. Opens with **a correction to Alpha's own reporting**: Alpha stated repeatedly that the LLM parameter-application seam "does not exist" — it exists at `agents/watcher_agent.py:1789-1793` and is reachable. **Two of the four findings may not be defects at all** (Chains C and D); Alpha will not treat a deliberate safety control as a bug. | **RULED** — Chain C reporting hotfix `f8b751c`; Chain D `pending_approval` upheld as a valid authority boundary |
| `TEAM_ALPHA_TRSE_FIX_PROPOSAL.md` | **TRSE's mathematics is sound.** Rule A verified working producer→consumer. **Rules B and C are ADVISORY-BY-DESIGN, not dropped wires** — three independent citations, including a governing ruling at `SESSION_CHANGELOG_20260307_S122.md:56`. The v1.15 spec text describing them as applied is **SUPERSEDED**. | **AWAITING RULING** on the proposed fix |
| `TEAM_BETA_SUMMARY_20260107.md` | Beta's A/B test of four LLM options and the resulting production architecture: dual-LLM (Qwen orchestration + Qwen-Math routing) → **DeepSeek-R1-14B primary + Claude backup**. | historical — see §3.8 |
| `TEAM_BETA_REVIEW_kfolds_S100.docx` | Beta's k-folds review, S100. **Binary `.docx` — not read in this pass; contents UNVERIFIED.** | UNVERIFIED |

### The Phase-3 permanent audit trail (`audit/S172_PHASE3/`)

Preserved because Phase 3 required **three review rounds and uncovered two real production defects** —
Beta policy grants a permanent trail when a review cycle finds a material correctness/safety/data-loss
defect, changes a frozen requirement, needs multiple binding rejections, or sets precedent.

| file | disposition |
|---|---|
| `audit/S172_PHASE3/README.md` | The policy statement above; index of the round sequence. |
| `audit/S172_PHASE3/PHASE3_INITIAL_REVIEW.md` | SUPERSEDED BY `PHASE3_FIX_BRIEF_REV2.md` |
| `audit/S172_PHASE3/PHASE3_FIX_BRIEF_REV2.md` | Binding fix brief. SUPERSEDED BY `PHASE3_FINAL_APPROVAL_REV3.md` |
| `audit/S172_PHASE3/PHASE3_FINAL_APPROVAL_REV3.md` | **Approved.** 14/14 gates on RTX 3080 Ti; ROCm deploy validation deferred to Phase 6. |

---

# 2. CHAPTERS — status and currency

**Chapter numbers are not step numbers.** Chapter 3 documents **Step 2.5 / WATCHER step 2**; the
bidirectional sieve runs inside **Step 1**. See §5.2.

| chapter | file | lines | audited? | currency | known-stale sections |
|---|---|---|---|---|---|
| 1 — Window Optimizer (Step 1) | `CHAPTER_1_WINDOW_OPTIMIZER.md` | 2,303 | ✅ **YES** — `CHAPTER_1_AUDIT_v1.md`, **9 of 41 claims accurate** | **CLOSED at `81ef3f1`, 2026-08-02** — "verified-and-bounded, not finished". §17 carries the closure statement. Remediation: P0 `ddd2ac8`, then P1/P2, closure `ef4b1c6` | Chapter revision "3.1" is **a documentation-only number with no source counterpart** — the module docstring says `Version: 2.0`, and `3.1` appears in no source file. Do not read it as a code version. |
| 2 — Bidirectional Sieve (Step 2) | `CHAPTER_2_BIDIRECTIONAL_SIEVE.md` | **1,463** | ✅ **YES** | **CLOSED at `81ef3f1`, 2026-08-02.** Destroyed at `248e48c` (709 lines removed), restored from `d14dcdd` (743L) at `e1225a7` → 1,089L, corrected `e50e35f`, closed `81ef3f1`, content gate `09bbfbf` (6 gates + 6 mutants, 12/12, all proven red against the actual 34-line fragment). §14 carries the closure statement. | ★ **§6 contains the three-lane CRT proof** — the thing Alpha claimed was undocumented. |
| 3 — Scorer Meta-Optimizer (**Step 2.5 / WATCHER step 2**) | `CHAPTER_3_SCORER_META_OPTIMIZER.md` | 958 | ✅ **YES** — `CHAPTER_3_ALIGNMENT_AUDIT.md`, **55 claims: 17 accurate / 9 stale / 24 false / 5 unverifiable** | **NOT corrected.** The audit was read-only and **no fix was authorised.** v4.2, last touched `05b0e6b`. | **§8, §9 and §14.2 describe GPU scoring deleted at v4.0.** §9 confines the soak to `--start-step 1 --end-step 1`. |
| 4 — Full Scoring (Step 3) | `CHAPTER_4_FULL_SCORING.md` | 1,037 | ❌ **NO** | v2.0.0 (Holdout Integration); claims "~550 lines" across two files | unknown — unaudited |
| 5 — Adaptive Meta-Optimizer (Step 4) | `CHAPTER_5_ML_ARCHITECTURE_OPTIMIZER_v2.md` | 423 | ❌ **NO** | v2.0.0 "(Corrected)" | unknown — unaudited |
| 6 — Anti-Overfit Training (Step 5) | `CHAPTER_6_ANTI_OVERFIT_TRAINING.md` | 738 | ❌ **NO** | v3.1.0 | unknown — unaudited |
| 7 — Prediction Generator (Step 6) | `CHAPTER_7_PREDICTION_GENERATOR.md` | 933 | ❌ **NO** | v1.0 | unknown — unaudited |
| 8 — PRNG Registry | `CHAPTER_8_PRNG_REGISTRY.md` | 1,058 | ❌ **NO** | v2.4; claims `prng_registry.py` = 4,323 lines | unknown — unaudited |
| 9 — GPU Cluster Infrastructure | `CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md` | 980 | ❌ **NO** | v2.0.0 (Consolidated) — predates the Proxmox migration | topology; see §6.4 |
| 9 addendum | `CHAPTER_9_ADDENDUM_v2_2_0.md` | 152 | ❌ | Diagnostic battery + ramdisk v2.1.0, Jan 2026 | — |
| 10 — Autonomous Agent Framework | `CHAPTER_10_AUTONOMOUS_AGENT_FRAMEWORK_v3.md` | 586 | ❌ | v3.1.0, 2026-02-03, "Full Autonomous Operation — Phase 7 Complete" **(that is the *old* Phase 7, not S172 Phase 7 — see §5.2)** | `_v2.md` (1,553L) superseded; `.bak` is a stale duplicate |
| 11 — Feature Importance & Visualization | `CHAPTER_11_FEATURE_IMPORTANCE_VISUALIZATION.md` | 1,099 | ❌ | — | patched by `PATCH_Chapter11_LLM_Update_v2.md` / `apply_chapter11_patch.sh` |
| 12 — WATCHER Agent & Fingerprint Registry | `CHAPTER_12_WATCHER_AGENT.md` | 880 | ❌ | v1.4.0, 2026-02-03 | two addenda not folded in: `CHAPTER_12_ADDENDUM_v1_3_0.md` (preflight/cleanup), `CHAPTER_12_ADDENDUM_PHASE1_v1_1_2.md` (freshness + HARD/SOFT preflight) |
| 13 — Live Feedback Loop | `CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md` | 1,231 | ❌ **NO** | "Architecture-Final" | `CHAPTER_13_LIVE_FEEDBACK_LOOP.md` (1,229L) is the superseded v1.0; §19 checklist superseded by `CHAPTER_13_SECTION_19_UPDATED.md` |
| 14 — Training Diagnostics | `CHAPTER_14_TRAINING_DIAGNOSTICS.md` | 3,199 | ❌ | v1.2.0, "ACTIVE — Phases 1, 3, 5, 6 Complete (S69–S73)" | header superseded by `CHAPTER_14_HEADER_PATCH.md` |

**Chapters 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 and 14 are UNAUDITED.** `BACKLOG.md` §1 sets the prior
explicitly: *"Chapter 1's audit found 9 of 41 claims accurate. The base rate for an unaudited chapter
in this project is not 'mostly right with a few stale lines.'"*

**Chapter working documents** (not chapters themselves): `CHAPTER_1_AUDIT_v1.md` (736L) ·
`CHAPTER_1_PATCH_S114.md` (**⛔ SUPERSEDED — never merged, and its central mechanism was deleted from
the code while it sat unmerged; retained for history only**) · `CHAPTER_2_SOURCE_MAP_v1.md` (654L,
reconnaissance for the restoration) · `CHAPTER_3_ALIGNMENT_AUDIT.md` (925L) ·
`chapter2_interchunk_cleanup_section.md` (46L — an inter-chunk GPU cleanup section drafted for
Chapter 2, Jan 2026) · `CHAPTER_13_IMPLEMENTATION_PROGRESS*.md` (**14 versions**, v1.1 → v3.9,
2026-01-11 → 2026-02-15 — read **v3.9 only**; note the duplicate
`CHAPTER_13_IMPLEMENTATION_PROGRESS_v1_2 .md` with a space in the filename) ·
`CHAPTER_14_IMPLEMENTATION_PROPOSAL_S69.md` (398L).

---

# 3. THE INTENT INDEX

Grouped by theme. **Test for every line: if someone asked "where is X documented", would this line
let them find it?**

## 3.1 Why the design is what it is — foundations

| file | the question it answers |
|---|---|
| `BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md` (167L) | **Why bidirectional, and why thresholds must stay loose.** Derives the filtering power: for incorrect seeds forward and reverse survival are approximately independent, so `e^(−cn) → e^(−2cn)` — bidirectional **squares the exponent**. §7 is the counter-intuitive part that gets misread every time: exact sieves eliminate all variance, leaving no ranking, no gradients and **no learning signal**. |
| `TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md` (292L) | **The canonical system map** — what the learning architecture actually is, end to end. v1.2 supersedes v1.1 and v1.0; §7's corrections are load-bearing. Where this map and `S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md` disagree, **the report wins** — it read live source; v1.0 of the map read a public clone. |
| `TRIANGULATED_FUNCTIONAL_MIMICRY_VERIFIED_v1_0.md` (673L) | What "Triangulated Functional Mimicry" means as a technical method, using **verified metrics only** — the reference for describing the system without overclaiming. |
| `SKIP_SEMANTICS_SEARCH_v1.md` (407L) | ★ **Is the meaning of `skip_min`/`skip_max` for the hybrid kernels written down anywhere? VERDICT: FOUND.** The search that falsified the "nobody documented this" claim. Read this before proposing any change to skip. |
| `HYBRID_SKIP_BOUND_AUDIT.md` (376L) | Do the trial's sampled `skip_min`/`skip_max` reach the hybrid kernels? (They do not; they die at `_hybrid_prefix`.) **Its line 318 premise — that hybrid skip semantics are "unspecified" — is FALSE**, per the search above. Read it for the wiring trace, **not** for the semantics verdict. |
| `TFM_PROJECT_FACTS_SKILL.md` (997L) | The committed copy of the `tfm-project-facts` skill: foundations, settled facts, the superseded list, the mandatory verification procedure. **Currency is stated in its own header — check it against the live skill before relying on either.** |
| `VERIFICATION_INTEGRITY_STANDARD.md` (159L) | **VIR-1 … VIR-5 (VIR-6 added later).** What makes a check a check: a verification must prove its own execution; vacuous-capable detectors need a clean control *and* a fault-injection control; every gate terminates in `PASS \| FAIL \| UNAVAILABLE \| INCOMPLETE`. Adopted by Beta in the Phase 6.0 final ruling after three incidents of *a check that was not checking, presenting as a pass*. Referenced by every implementation brief. |
| `KNOWN_ANSWER_VALIDATION_INVENTORY.md` (430L) | **Does known-answer sieve validation already exist? YES, substantially.** Every registry PRNG was driven through the sieves during pipeline development in constant-forward, constant-reverse and hybrid variable-skip modes. The inventory that struck Wall C. |
| `THRESHOLD_GOVERNANCE.md` (157L; also at repo root) | The governance model for sieve thresholds — who may change them and within what bounds. **Its synthetic-era defaults are superseded by `THRESHOLD_CALIBRATION_FINDINGS_S148.md`.** |
| `DESIGN_INVARIANT_GPU_ISOLATION.md` (158L) | **MANDATORY, non-negotiable:** GPU-accelerated code must never run in the coordinating process when subprocess isolation is in use. Enforced since S72. |

## 3.2 Frozen contracts and specifications

| file | the question it answers |
|---|---|
| `PROPOSAL_S172_RANGE_MINER_v1_4_5.md` (344L) | **The authoritative RANGE-MINER architecture.** Absorbs the binding S175 ruling (remote spool staging, staged A+C parallel assembly, high-survivor acceptance, three-way verification). Where it and v1.4.4 conflict, **v1.4.5 governs.** |
| `PROPOSAL_S172_RANGE_MINER_v1_4_4.md` (747L) | **SUPERSEDED**, retained for the audit trail. Authoritative **only** for sections v1.4.5 explicitly marks PRESERVED. Frozen at `1f6c0c5`; still the named spec for Phase 3. |
| `DATASET_PUBLICATION_SCHEMA_v1.md` (175L) | **FROZEN**, `manifest_schema_version: 1`. The dataset publication schema. Covers the **combined** `daily3.json` only — the split `daily3_midday.json` / `daily3_evening.json` are explicitly **not** covered and remain unversioned. **Where a brief and this schema differ, the schema wins.** |
| `RUNTIME_DATASET_PROVISIONING_CONTRACT.md` (175L) | What a run must guarantee about dataset identity across nodes. **Phase attribution corrected: its obligations are Phase 6-P0.5, not P0.** Its `expected_sha256`-as-static-config field is **superseded** — see §6.1. |
| `PROVISIONING_CONTRACT_AMENDMENT.md` (141L) | The Beta-mandated amendment inserting fail-before-dispatch and per-node verification into the contract above as explicit **P0.5 obligations**, so contract and phase boundary stop disagreeing. |
| `DAILY3_CONSUMER_CONTRACT_v1.md` (514L) | **What the pipeline actually requires of the draw dataset**, established by tracing live code rather than assumption — so a rewritten producer cannot silently break or alter downstream steps. Read-only; this document was the only artifact. |
| `S172_INFRASTRUCTURE_INTERFACE_v1_0.md` (200L) | The miner ↔ rig deployment contract: what the RANGE-MINER track and the Proxmox migration track may each assume about the other, so they converge cleanly at Phase 6. Source of the CT100-hostname-equals-rig-name identity rule. |
| `CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md` (274L) | **RATIFIED.** Authority boundaries between selfplay, Chapter 13 and WATCHER — who may decide what. Binding on all future implementation. |
| `CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md` (735L) | The LLM-guided selfplay Strategy Advisor contract (Beta-authored): what the advisor may emit and under what activation gate. |
| `CONTRACT_SECTION_8_5_ADDENDUM.md` (61L) | Adds §8.5 (LLM lifecycle dependency) to the contract above; Beta-approved S67. |
| `SPEC_BUNDLE_FACTORY_v1_1_0.md` (373L) | The bundle-factory specification — how step-awareness bundles are constructed. |
| `ADDENDUM_A_STEP_AWARENESS_BUNDLES_v1_0.md` (364L) | **LOCKED, joint Alpha + Beta.** The step-awareness bundle format itself (`agents/contexts/bundle_factory.py`). |
| `TRSE_v1_15_SPEC.md` (333L) | The TRSE (Temporal Regime Segmentation) v1.15 specification. **SPEC ONLY — its text describing Rules B and C as applied is SUPERSEDED** (they are advisory by design). |
| `TRSE_INTEGRATION_PLAN_S121.md` (192L) | ★ **How TRSE integrates with Step 1 — and the file that keeps being cited without being opened.** §2B lists the manifest's `default_params`; §2C specifies the **PASSIVE** integration (Step 1 reads `trse_context.json` itself; WATCHER parses and injects nothing) and the `min(rec_ws * 4, …)` rule that makes `8 × 4 = 32`. |
| `PHASE_9B2_INTEGRATION_SPEC.md` (413L) | Beta-approved integration spec for `selfplay_orchestrator.py` v1.0.6 → v1.1.0. |

## 3.3 RANGE-MINER (S172) — implementation briefs

`CLAUDE_CODE_INSTRUCTIONS_*` are implementation briefs written *to* Claude Code on VM101. Each names
a base commit, forbids commit/push/WATCHER, and stops at a gate. **A brief describes what was
authorised, not necessarily what shipped** — pair it with the review record in §1.2.

| file | the question it answers |
|---|---|
| `CLAUDE_CODE_INSTRUCTIONS_S172_PHASE4.md` (328L) | Staged Phase-4 coordinator implementation (Stage 0 = the Blocker-6 `ResidueResolver` patch, Stage 4 = the L7 abort interface). |
| `CLAUDE_CODE_CORRECTION_S172_PHASE4_SERVE.md` → `..._CORRECTION6_..._ADMISSION.md` (6 files) | The six Beta rejection rounds in order: real serve path → six serve/ledger defects → six async-staging/socket defects → four overload freezes → three real-resource bounds → one admission byte-model. Pair each with the matching `TEAM_ALPHA_REVIEW_S172_PHASE4_REV*.md`. |
| `..._S172_PHASE5.md` (459L) | Phase-5 umbrella: NPZ writer + assembly, plus the prerequisite Phase-4 seam correction. |
| `..._S172_PHASE5_D1.md` (907L, REV5) | D1.0 workflow/terminal-race + D1.1 four-population assembly engine and concrete `Phase5Sink`. |
| `..._S172_PHASE5_D2.md` (227L) | Directional uniqueness at **both** layers — producer overlap rejection and a Phase-5 fail-closed probe. |
| `..._S172_PHASE5_D3.md` (396L, REV3) | The shared backend-neutral 24→22 columnizer + independent structural validator. **[A1]** `prng_base` restricted to a forward, non-hybrid base family — registry membership alone was insufficient. |
| `..._S172_PHASE5_D3_0.md` (193L) | Legacy seam correction: canonical PRNG/skip encoding + rectangular 22-array empty output. |
| `..._S172_PHASE5_D3_25.md` (383L, REV3) | Mode-preserving backend result contract + canonical candidate-ingress normalisation across the miner, PWC and ZMQ adapters. |
| `..._S172_PHASE5_D3_5.md` (819L, REV3.1) | The shared run finalizer. **[D1]** the chain **tip** is authenticated — the generation directory is named `<generation_id>--<sidecar_sha256>`, making the atomic `current` pointer the trust anchor for the newest generation. |
| `..._S172_PHASE5_D3_5_B.md` (276L, REV2) | Seed-Domain v1.1. **[R1]** the recursive chain walk did **not** compare `prng_base`/`schema_version`/`canonical_map_hash`; the per-link contract is now specified explicitly. |
| `..._S172_PHASE5_D4.md` (335L, REV3) | `serial_reference` behind a frozen two-backend interface. **[B1]** the return contract frozen as `BackendAssemblyResult` + `AssemblyMeasurement` — REV2 had left four incompatible APIs open. |
| `..._S172_PHASE5_D5.md` (436L) + `..._D5_REV2_ADDENDUM.md` + `..._D5_REV3_ADDENDUM.md` | The `process_sharded` backend. REV2 = Beta ruled **Option B** (preserve deterministic exception precedence). REV3 = the `int64` seed-projection divergence and its lossless-fallback resolution. |
| `..._S172_PHASE5_D6.md` (234L) + `..._S172_PHASE5_D6_CORRECTION.md` (188L) | D6, the production integration adapter — **the first deliverable touching real silicon and real Step-1/Step-2 wiring.** The correction is the hardcoded-`0.25` threshold repair. |
| `..._S172_D6_1_FLUSH_DURABILITY.md` (176L) | Incremental NPZ atomic flush and durability. Beta's framing is the operative one: **incremental durability does not currently exist.** |
| `..._S172_D6_2_CHECKPOINT_RECONCILIATION.md` (467L, REV5) | The 24-field checkpoint, canonical reconciliation and the finalizer resume path. |
| `..._S172_D6_2_REV5_BINDING_ADDENDUM.md` (176L) | **BINDING — where this and REV5 differ, THIS WINS.** Four normative items; Beta authorised implementation with no REV6. |
| `..._S172_D6_2_BOUNDED_REPAIR.md` (205L) | The two execution-path defects the 29 D6.2 gates do not exercise. Repair **on top of** `f7583bc`; do not revert. |
| `..._S172_D6_3_RETENTION_INVESTIGATION.md` (129L) | **READ-ONLY.** Checkpoint retention: authorises no fix, no policy and no deletion. Run *after* the import gate. |
| `..._S172_PROCESS_SHARDED_IMPORT_GATE.md` (169L) | **Beta-REQUIRED hardening**, test-side only: prove `assert_cpu_only` reds on a module-level GPU import, in a fresh spawned interpreter, against the **production** forbidden list, covering both `torch` and `cupy`. |
| `..._S172_THRESHOLD_REPAIR.md` (171L, REV2) | The optimizer threshold-propagation repair. REV2 shrinks scope because Beta retired PWC/ZMQ from certifying authority, so repair 3 is no longer a Phase-6 blocker. |
| `..._S172_PHASE_6_0_ROCM_PARITY.md` (208L) | Single-rig ROCm smoke against an identical CUDA control. Beta's required addition: **schema-valid ROCm output alone does not establish computational parity** — a platform-specific kernel defect could still produce a structurally valid generation. |
| `..._BOUNDED_PHASE_6.md` (170L) | **The Phase-6 certification brief.** "This is the certification the whole PWC → RANGE-MINER pivot was built toward." |
| `..._PHASE_6_P0_SCOPING.md` (164L) | **READ-ONLY.** Where does a published dataset live, and what breaks when it moves? |
| `..._PHASE_6_P0_IMPLEMENTATION.md` (166L) | **P0 CREATES FILES. P0 DOES NOT CHANGE RUNNING CODE.** That boundary is the deliverable's defining constraint. |
| `..._PHASE_6_P0_5_IMPLEMENTATION.md` (176L) | P0.5, the behavioural cutover. The inversion is deliberate: every behavioural change lands together against a published baseline, so the first post-publication certification has **one** cause to attribute. |
| `..._P0_5_Q2_CLOSURE.md` (117L) | The single P0.5 closure condition: a missing provisioning manifest must hard-fail a miner-backed run. **If the change grows beyond that path and its negative gate, stop and report.** |
| `..._ADMISSION_LIVENESS_REPAIR.md` (148L) | The §4.3 silent hang: separate **admission** (bounded) from **execution maintenance** (unbounded). |
| `..._RESOLVED_EXECUTION_SET.md` (146L) | **A Phase-7 blocker.** One frozen run-scoped fleet authority; all six existing mechanisms become consumers. |
| `..._ADMISSION_BINDING_REPAIR.md` (142L) | The two repairs Beta required before granting Phase-7 closure: bind admission to the frozen set, and repair the false freeze-after-read property. |
| `..._S172_PHASE_6_P2_SCRAPER.md` (385L, REV4 **DRAFT**) | Append-only immutable dataset publication. **Pending Beta**; the schema wins where they differ. |
| `..._S172_PHASE_7_SOAK.md` (189L, REV1 **DRAFT**) | Phase 7: 50-trial WATCHER soak, **≥5 high-survivor and ≥5 low-survivor trials, mixed constant/hybrid, per-trial cleanup verification.** |
| `..._CHAPTER_1_AUDIT.md` · `..._CHAPTER_1_P0_CORRECTION.md` · `..._CHAPTER_1_P1_P2_CORRECTION.md` · `..._CHAPTER_2_SOURCE_GATHERING.md` · `..._CHAPTER_2_RESTORE.md` · `..._CHAPTER_2_CORRECTIONS.md` · `..._CHAPTER_1_AND_2_CLOSURE.md` · `..._CHAPTER_3_ALIGNMENT_AUDIT.md` | The eight chapter-track briefs, in execution order. Each states its own scope limit — audit-only, documentation-only, or code-and-docs — and three of them say **explicitly that no fix is authorised.** |
| `..._TRSE_STEP0_AUDIT.md` · `..._STRATEGY_ADVISOR_AUDIT.md` · `..._SAMPLER_BEARING.md` · `..._CHAIN_C_TRUTHFULNESS_HOTFIX.md` · `..._TOPOLOGY_DOC_CORRECTION_v2.md` · `..._PROJECT_CATALOG_REGENERATION.md` | The audit / scoping / hotfix briefs — see §3.5 and §3.8 for what each produced. |
| `S172_PHASE4_BRIEF.md` (535L, rev-4) | The Phase-4 implementation brief itself — Blockers 1–7, Decisions A/B, L1–L8, gates 1–36. Cited constantly; **open it rather than citing it.** |
| `S172_PHASE5_D5_CHAT_PROMPT.md` (73L) | The D5 kickoff prompt for Claude Code on VM101. |

## 3.4 RANGE-MINER (S172) — evidence and reports

| file | the question it answers |
|---|---|
| `S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md` (486L) | The ROCm/CUDA parity evidence record. Notes the base-commit substitution (`8e2f5bf` is docs-only on top of the `3823b56` the brief names). |
| `D6_RELEASE_GRADE_CERTIFICATION_RECORD.md` (153L) | The release-grade certified generation produced from the clean real repository (Beta step 3), 2026-07-29. Raw evidence: `D6_RELEASE_GRADE_SMOKE_20260729.log` (16K). |
| `D6_FOLLOWUP_BOTH_MODES_SMOKE.md` (107L) | Proves the **variable-skip / hybrid column** (phases 3+4) end-to-end on real silicon — a different kernel and different seed caps from D6's constant-skip 3.B smoke. |
| `S172_D6_2_IMPLEMENTATION_REPORT.md` (737L) | Full D6.2 implementation evidence. |
| `S172_D6_2_BOUNDED_REPAIR_REPORT.md` (406L) | The bounded-repair evidence; completion sentinel `PASS`. |
| `S172_THRESHOLD_PROPAGATION_REPAIR_REPORT.md` (419L) | Evidence for the threshold repair against `THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md`. |
| `S172_PROCESS_SHARDED_IMPORT_GATE_REPORT.md` (502L) | The import-gate deliverable, incl. the mid-session addendum (contamination guard; the mutant must red **for the right reason**). |
| `S172_PHASE_7_PREREQ_REPORT.md` (826L) | §1 prerequisite measurement for the soak: items 5 and 7 measured, a 25-worker execution set proven freezable **by construction**, §6 checkpoint census. **No soak launched.** |
| `S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md` (681L) | ★ Two independent questions answered from live source with `file:line` anchors: **the survivor feature schema** (91 extracted / 89 trained; the three namespaces; the five dead placeholders) and **per-survivor attribution** (implemented, invoked, unreachable, unconsumed). The primary evidence behind the system map. |
| `S172_SIEVE_PATH_VERIFICATION_SCOPE.md` (100L) | ★ **What is and is not proven about the four sieve paths** — so nobody mistakes "Phase 3 green" for "the sieve computes correct survivors through the miner". Standing reference; update the status column as phases land. |
| `STEP2_BIDIRECTIONAL_SIEVE_DESCRIPTIVE_TRACE.md` (1,205L) | A read-only descriptive survey of Step 2 as built, 2026-07-28. |
| `ROCm_Saturation_Report_S172.md` (195L) | ROCm driver-level saturation: boundary mapping, failure analysis, mitigation. **The measurement behind the PWC → RANGE-MINER pivot.** |
| `PROVENANCE_DISPOSITION_ACCUMULATOR_20260725.md` (268L, REV2) | Accumulator provenance disposition. **REV2 incorporates all four Beta corrections — a copy without the REV2 banner is the superseded draft; discard it.** |
| `PHASE6_PREREQS.md` (441L, REV5) | Operational prerequisites for real-silicon testing. **REV4 corrected five of seven statuses from live measurement** and records that D3.0-B, stated here as mandatory before Phase 6 certification, was never completed. |
| `PHASE7_PREREQUISITES.md` (119L) | The durable answer to "what stands between us and the Phase 7 soak?" — a question that kept being re-derived. Current as of 2026-07-30, at D6.1. |
| `PHASE_6_P0_SCOPING_v1.md` (702L) | The read-only scoping report: where a published dataset lives and the blast radius of moving it. Nothing was created, moved or modified. |
| `PHASE_6_P0_IMPLEMENTATION_v1.md` (363L) | The P0 implementation record — dataset version one published in place. |
| `SAMPLER_BEARING_v1.md` · `STRATEGY_ADVISOR_AUDIT_v1.md` · `STRATEGY_ORIGIN_AUDIT.md` · `TRSE_STEP0_AUDIT_v1.md` · `FLEET_STATE_REQUIREMENTS_v1.md` · `THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md` | see §3.5 |
| `docs/phase6_evidence/wall_ab.json` · `known_answer_gate.json` · `sampler_control_arm.json` | The machine-readable Phase-6 evidence the bounded-Phase-6 sentinels cite. |

## 3.5 Audits and read-only investigations

**These are the documents most likely to already contain the answer you are about to go looking for.**

| file | the falsifiable question it answers |
|---|---|
| `CHAPTER_1_AUDIT_v1.md` (736L) | Does Chapter 1 describe the live window optimizer? **9 of 41 claims accurate.** |
| `CHAPTER_3_ALIGNMENT_AUDIT.md` (925L) | Does Chapter 3 describe the code today, and does its stage still align after RANGE-MINER replaced the Step-2 engine? **55 claims: 17 / 9 / 24 / 5.** Sentinel `PASS`. **NO FIX WAS AUTHORISED.** |
| `CHAPTER_2_SOURCE_MAP_v1.md` (654L) | Where would the material for a Chapter-2 reconstruction come from? Reconnaissance — superseded in purpose once restore-and-audit was ruled. |
| `TRSE_STEP0_AUDIT_v1.md` (537L) | What does TRSE compute, and do its outputs reach anything? |
| `STRATEGY_ADVISOR_AUDIT_v1.md` (779L) | What does the Strategy Advisor emit, what validates it, what applies it, what executes it — **and where does the chain break?** The four-chain analysis. |
| `SAMPLER_BEARING_v1.md` (662L) | **Cost and blast radius** of four working Optuna samplers in Step 1. READ-ONLY scoping — not an implementation and not an authorisation. Optuna 4.4.0. |
| `STRATEGY_ORIGIN_AUDIT.md` (396L) | Were RandomSearch / GridSearch / EvolutionarySearch **ever** Optuna-backed? A read-only *history* investigation. |
| `THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md` (384L) | In the window-optimizer / PWC route, does a configured threshold reach the kernel, or is it dropped for a default? **Explicitly adjudicates a `tfm-project-facts` §2.7 claim** — read it before repeating that claim. |
| `FLEET_STATE_REQUIREMENTS_v1.md` (548L) | What does a run actually demand of the fleet, and do the mechanisms agree? **They do not** — six checks, three granularities, two address sets. |
| `DOCUMENTATION_AUDIT_20260131.md` (179L) | Which project-knowledge documents were stale as of 2026-01-31? (8, all predating rig-6600c integration.) **Itself now stale — a historical record of a staleness sweep.** |
| `S107_session_log.md` (108L) | The Step-2 v4.1→v4.4 repair narrative, incl. **`sample_size=450` hardcoded in the shell script** so WATCHER's `{"sample_size":5000}` override was silently dropped, and `.162` missing from both the scp push loop and `ml_coordinator_config.json` (25 results per run silently lost). |
| `WATCHER_KPI_CALIBRATION_FINDINGS_S176.md` (269L) | Analytic + deterministic validation of the WATCHER hit/survivor KPIs. **Recommend-only — changes nothing in `watcher_policies.json`.** |
| `THRESHOLD_CALIBRATION_FINDINGS_S148.md` (373L) | **Authoritative** empirically-grounded sieve threshold defaults; supersedes the synthetic-era values in `THRESHOLD_GOVERNANCE.md`. |
| `STEP1_EXECUTION_FLOW_AND_PRUNING_S147.md` (196L) | Both Step-1 execution paths and the pruning logic, as of S147. |
| `STEP1_GPU_BENCHMARK_SUITE.md` (702L) | The systematic benchmark methodology for preventing GPU overload during Step 1 (modelled on the Step-2/3 work that established `sample_size=450 @ max_concurrent=12`). |
| `GPU_THROUGHPUT_INVESTIGATION_PLAN_v1_0.md` (244L) | The planned throughput investigation (S126). **PLANNED — pending execution.** |
| `ROOT_CAUSE_ANALYSIS_RRIG6600C_S151.md` (183L) | rrig6600c persistent crashes — **RESOLVED**, S151. |

## 3.6 How the system is operated

| file | the question it answers |
|---|---|
| `COMPLETE_OPERATING_GUIDE_v2_0.md` (1,181L) | The current operating guide (v2.2.0, updated S135 / 2026-03-10). **Predates RANGE-MINER, the Proxmox migration and dataset authority.** |
| `COMPLETE_OPERATING_GUIDE_v1_1.md` (760L) | SUPERSEDED by v2.0. |
| `Cluster_operating_manual.txt` (96K) | The cluster operating manual. Carries the `skip_min`/`skip_max` **input** reading verbatim at `:948-949`. |
| `Cluster_operating_manual_v1_1_update.md` (136L) | Session-17 changes for the manual — includes the record that **Step 0 PRNG fingerprinting was ARCHIVED**: mathematical analysis proved fingerprinting impossible under mod-1000 projection. |
| `instructions.txt` (152K) | The long-running operating document. ★ **`:1182-1183` is the load-bearing line** — the `skip_min`/`skip_max` *input* (element-wise pattern bound) reading, with the hybrid default `[0,16]`; **`:1230-1245`** is the Oct-2025 output spec declaring `skip_pattern` and `pattern_stats`, the literal ancestor of the three dead skip features. |
| `INSTRUCTIONS_NPZ_ADDITION.md` (260L) | The NPZ v3.0 binary survivor format section written for `instructions.txt`. |
| `CANONICAL_PIPELINE_AND_CH13_WITH_STARTUP_COMPLETE.md` (340L) | End-to-end operational walkthrough of the canonical pipeline plus Chapter 13, with no placeholders. |
| `complete_workflow_guide_v2_PULL_UPDATED.md` (2,728L) | The v2.0 workflow guide — manual per-step vs orchestrated, with variable-skip support. |
| `complete_workflow_guide_update_v2_1.md` (205L) | The v2.1 delta: `scripts_coordinator.py` v1.4.0. |
| `README.md` (339L) | The docs-tree README (a copy of the root README). See §6.4 on its framing. |
| `PROJECT_STATUS.md` (85L) | Component-readiness snapshot at **S109 / 2026-02-23**. Historical. |
| `REMOTE_NODE_SETUP_CHECKLIST.md` (278L) | How to stand up a new remote worker node. **Bare-metal era — predates the CT100 model.** |
| `TELEGRAM_NOTIFICATION_SYSTEM_REFERENCE.md` (323L) | The cluster notification system. **v2.1, Proxmox topology incl. pzeus — supersedes the 2026-04-03 bare-metal edition.** The most topology-current operational document in the tree. |
| `LLM_INFRASTRUCTURE.md` (35L) | **A pointer document, deliberately.** TFM shares Proxmox/LXC infrastructure with the local LLM cluster; the canonical docs live in the `rx6600-llm-inference` repo, not here. |
| `WATCHER_POLICIES_REFERENCE.md` (211L) | **The canonical meaning of every flag in `watcher_policies.json`.** |
| `SOAK_TEST_PLAN_PHASE7_v1_0.md` (850L) | The *old* Phase-7 (WATCHER dispatch) soak plan — **not** S172 Phase 7. |
| `SOAK_TEST_HANDOFF_PROMPT.md` (316L) | A resumable context prompt. Carries the standing framing: **"This is NOT specifically a lottery system"** — PRNG-agnostic by design, all generator behaviour abstracted via `prng_registry.py`. |
| `SOAK_C_GAPS_AND_PATCHES_v1_0.md` (620L) | Soak-C integration gaps (the acceptance engine did not honour `test_mode` flags) and proposed patches. |
| `SUBPROCESS_ISOLATION_INTEGRATION_GUIDE.md` (335L) | How to integrate subprocess isolation into `meta_prediction_optimizer_anti_overfit.py`. |
| `GBNF_DEPLOYMENT_README.md` (279L) | Deploying the GBNF grammars that constrain LLM output. |
| `Distributed_PRNG_Pipeline_Overview_for_Novices.pdf` · `Distributed_PRNG_Pipeline_Technical_Addendum.pdf` (16K each) | Onboarding overview and technical addendum. **PDFs — not read in this pass; contents UNVERIFIED.** |

## 3.7 Backlog, TODO, status and handoff

| file | the question it answers |
|---|---|
| `BACKLOG.md` (274L) | ★ **The live register.** Everything known, deliberately deferred, and **not** a Phase-7 blocker — written down so it is not rediscovered as a surprise finding. 14 numbered entries incl. unaudited chapters, skip-output sequencing, the sampler-comparison correction, the three `[WATCHER][RETRY]` log lines, session-separated dataset authority, `dataset_provenance/*.json` never pruned, the two Beta-required pre-Phase-7 audits, and `_RusageChildrenSampler` measuring the wrong thing. **Currency: HEAD `6892661`. New findings go here.** |
| `TODO_MASTER_S*.md` (**19 files**: S101, S114, S120, S122, S125b, S126, S127, S132, S135, S139, S143, S145, S148, S150, S152, S154, S163, S163KARG, S170) | A rolling P0/P1 master list, each compiled from the previous plus intervening changelogs. **Read only the latest (`TODO_MASTER_S170.md`, 2026-04-24) — and note it predates the entire S172 track.** Their real value is historical: each header states the cluster and pipeline state on its date. |
| `TODO_SELFPLAY_AND_LLM_AUTONOMY.md` (425L) | The autonomy last-mile track — **tracked and unstarted.** Part B, 20 tasks. **Task B3 auto-extracts tunable parameters from `agent_manifests/*.json`**, which is exactly why the next entry exists. |
| `D6_THRESHOLD_AUTONOMY_SIGNPOSTS.md` (91L) | Anti-bite signposts placed so the threshold disconnect cannot be picked up by autonomy work — **they take on no autonomy work themselves.** |
| `TODO_PHASE7_WATCHER_INTEGRATION_REVISED.md` / `_v3.md` | The old Phase-7 (WATCHER integration) task list. v3 is marked ALL PARTS COMPLETE. (v3's header contains mojibake — `â€"` — from an encoding round-trip.) |
| `TODO_DISPLAY_AND_VISUALIZATION.md` (150L) | Terminal display + visualisation improvements. **PENDING since 2025-12-06.** |
| `SESSION_61_HANDOFF.md` · `SESSION_81_HANDOFF.md` · `SESSION_HANDOFF_20260204.md` · `SESSION_CONTINUATION_PHASE7_PART_B.md` · `SESSION_NOTES_20260102.md` · `SESSION_NOTES_20260118_PIPELINE_TEST.md` | Point-in-time handoffs. Useful only for reconstructing what was believed on a given date. |
| `S1xx_CHAT_PROMPT.md` (**13 files**: S142, S144, S147, S149, S151, S152, S155, S157, S159, S160, S163, S164, S170) | Session-opening context prompts, each stating the cluster state, HEAD and P0/P1 priorities of its day. **Historical state records, not instructions.** (`S148_CHAT_PROMPT.md` and `S162_CHAT_PROMPT.md` are the same class but live at the repo root.) |
| `DOCUMENTATION_INDEX_v1_0.md` / `_v1_1.md` (274L each) | The Session-78/80 documentation indexes. **Name-indexed — superseded by this catalog, and the reason this catalog is intent-indexed instead.** |
| `DOCUMENTATION_UPDATES_S70.md` · `_S71.md` · `_S146.md` | Which documents needed updating after S70, S71 and S146, and with what. Useful for dating a doc's last intended sync. |
| `PROJECT_FILE_CATALOG.md` | this file |

## 3.8 Autonomy, WATCHER, LLM and selfplay

| file | the question it answers |
|---|---|
| `PROPOSAL_WATCHER_KPI_GOVERNANCE_STATES_v1_0.md` → `_v1_1.md` → `_v1_2_ADDENDUM.md` | The BOOTSTRAP → CALIBRATING → GOVERNED KPI state machine. **All three are recommend-only and change nothing; no thresholds are selected.** Read **v1.1 + the v1.2 addendum** together — v1.0 is superseded. Rulings: S177 → S178 → S179. |
| `CLAUDE_CODE_BRIEF_WATCHER_KPI_VALIDATION_v1.md` (200L) | Establish evidence-backed baselines for the WATCHER governance KPIs and validate the Step-0 (TRSE) advisory heuristics on real data. |
| `CLAUDE_CODE_BRIEF_S176_FOLLOWUP_v1.md` / `_S177_RESUBMISSION_v1.md` / `_S178_FOLLOWUP_v1.md` | The three implementation briefs the S176/S177/S178 rulings unblocked. Each names its ruling as the spec. The cluster-heavy Phase-C walk-forward is explicitly **out of scope** in S176. |
| `WATCHER_PHASE1_PATCH_v1.1_FINAL.md` (544L) | Stale-output prevention for WATCHER — the freshness check. Supersedes `WATCHER_PHASE1_PATCH_FOR_REVIEW.md` (337L). |
| `CLAUDE_CODE_INSTRUCTIONS_STRATEGY_ADVISOR_AUDIT.md` (151L) | The audit brief for the advisor. **A parked reference document** — the chapter track takes priority; nothing here authorises implementation. |
| `CLAUDE_CODE_INSTRUCTIONS_SAMPLER_BEARING.md` (146L) | The bearing brief: what would four working samplers cost? **A cost estimate, not an authorisation.** |
| `CLAUDE_CODE_INSTRUCTIONS_CHAIN_C_TRUTHFULNESS_HOTFIX.md` (133L) | **A narrow truthfulness repair, deliberately small.** WATCHER reports `Applied:` for LLM parameter proposals that never execute. Beta ruled the **filtering correct** — only the reporting is defective. If the change grows beyond the log message, its status field and a rejection reason, **stop and report.** |
| `CLAUDE_CODE_INSTRUCTIONS_TRSE_STEP0_AUDIT.md` (145L) | **"Do not assume a defect exists."** The TRSE audit brief. |
| `SELFPLAY_ARCHITECTURE_PROPOSAL_v1_0.md` (651L) | Multi-model inner-episode training. **APPROVED by Beta + user.** Reframed by `TB_UPDATE_SELFPLAY_REFRAMING_2026-07-28.md` — read both. |
| `SELFPLAY_INTEGRATION_PROGRESS_v1_0.md` (391L) | Selfplay integration progress, 2026-01-29. |
| `PROPOSAL_EPISTEMIC_AUTONOMY_UNIFIED_v1_3.md` (913L) | The unified epistemic-autonomy architecture, v1.3 (supersedes v1.2/v1.1). **Implementation-Complete** as of 2026-02-10. |
| `PROPOSAL_v1_3_FINAL_ACCEPTANCE.md` (551L) | The acceptance record for the above. |
| `PROPOSAL_LLM_Architecture_v2_0_0.md` (387L) | **DeepSeek-R1-14B primary + Claude Opus backup**, validated by A/B testing. Replaces the dual-LLM schema v1.0.4 design (`PROPOSAL_Schema_v1_0_4_Dual_LLM_Architecture.txt`, at the repo root). |
| `LLM_STRATEGY_DEEPSEEK_CLAUDE_DOCUMENTED.md` (489L) | The documented LLM roles and decision-making, extracted from the actual implementation. |
| `PROPOSAL_LLM_Infrastructure_Optimization_v1_1.md` (857L) | LLM subsystem optimisation + grammar completion (`llm_services/`, `agent_grammars/`). |
| `PROPOSAL_LLM_Reasoning_Refactor_v1_0.md` / `_v1_1.md` | v1.1 is **TEAM BETA APPROVED (with conditions)**; v1.0 superseded. |
| `PROPOSAL_LLM_Router_v2_1_0_Merge.md` (344L) | `llm_services/llm_router.py` API restoration + missing method. |
| `PROPOSAL_LLM_Terminology_Solution_v1_0_0.md` (557L) | The LLM terminology-drift solution — **HIGH priority, blocking agent autonomy** at the time. |
| `PROPOSAL_SEARCH_STRATEGY_VISIBILITY_FIX_v1_0.md` (415L) | Advisory-layer blindness to the search strategy actually in use. |
| `PROPOSAL_STRATEGY_ADVISOR_LIFECYCLE_INTEGRATION_v1_0.md` (428L) | LLM lifecycle integration + heuristic demotion for `parameter_advisor.py`. |
| `PROGRESS_BUNDLE_FACTORY_AND_STRATEGY_ADVISOR_v1_0.md` (412L) | Execution plan for bundle factory v1.1.0 + the advisor. |
| `PROPOSAL_Unified_Agent_Context_Framework_v3_2_*.md` (**12 files**: v3.2.0, .1, .3, .4, .5, .6, .7, .8, .9, .10 + two addenda) | The agent-context framework lineage, Dec 2025; each supersedes the last. **Read v3.2.10 only.** The two addenda carry the distinct content: the **threshold addendum** records that sieve thresholds were hardcoded at 0.01, and **Addendum B** records the creation of the Steps 2–6 manifests with their `parameter_bounds`. |
| `proposals/PROPOSAL_Universal_Agent_Architecture_v1_1.md` (1,375L) + `_ADDENDUM.md` (1,244L) | The Dec-2025 universal agent architecture, both **DRAFT — Pending Review**. The framework above is what actually shipped. |
| `Multi-Model_Architecture_integration_autonomy.md` (396L) | How the multi-model architecture integrates with `watcher_agent.py`. |
| `NOTE_Step7_Not_Required_for_Autonomy.md` (182L) | **DECISION RECORDED** (Beta): Step 7 is not required for autonomy. **Replaces `PROPOSAL_Step7_PostPipeline_Export_v1_1.md`.** |
| `PROPOSAL_Modular_Step_Automation_Framework_v1_1.md` (1,077L) | The step-automation framework — core autonomy infrastructure, Jan 2026. |

## 3.9 ML, features, thresholds and objectives

| file | the question it answers |
|---|---|
| `PROPOSAL_ML_Architecture_Remediation_v2_0.md` (458L) | ★ The complete ML-architecture diagnosis. **Its `:150-158` is the source of the `skip_min`/`skip_max` *output* reading** — "minimum/maximum gap that **worked**", with *"tight skip range = stronger hypothesis"*. |
| `PROPOSAL_Feature_Implementation_Remediation_v1_0.md` (309L) | ★ **Of 64 defined features, only 2 had actual variance** — the rest constant, hardcoded placeholders, or never computed. The finding that made ML training effectively useless. |
| `IMPLEMENTATION_OUTLINE_Feature_Remediation_v1_1.md` (436L) | **APPROVED — READY FOR IMPLEMENTATION.** The decision table for the remediation. |
| `TECHNICAL_SPEC_Feature_Remediation_Phases_2_4.md` (1,018L) | Phase 2–4 technical specs — including the **skip-metadata pipeline** and its `skip_metadata.json` schema. |
| `PROGRESS_Feature_Remediation_v1_0.md` (352L) | The remediation progress tracker. |
| `PROPOSAL_STEP2_OBJECTIVE_FUNCTION_v1_3_0.md` (333L) | Two confirmed Step-2 bugs: objective-function collapse and deterministic sampling lock. |
| `HOLDOUT_HITS_IMPLEMENTATION_SUMMARY.md` (202L) | What `holdout_hits` is, and the position-offset bug found in it. **`holdout_hits` is itself superseded as the target — see §6.1.** |
| `S111_IMPLEMENTATION_PLAN_FINAL.md` (514L) | The holdout-validation redesign, verified against live Zeus code. |
| `S111_TEAM_BETA_BRIEFING.md` (166L) | Phase-1 results: `holdout_quality` live as the ML target at `1cb90aa`; 49,882 survivors at 100% coverage. |
| `PROPOSAL_CATEGORY_B_NN_TRAINING_ENHANCEMENTS_v1_0.md` (357L) | Neural-net training enhancements. DRAFT — awaiting Beta. |
| `PROPOSAL_DUAL_GPU_PARALLEL_TRAINING_v2_0.md` (246L) | Zeus GPU 1 sits 100% idle during 2-hour NN training; halve it with dual-GPU Optuna trials. |
| `PROPOSAL_PHASE3_CONCURRENT_TRIAL_BATCHING_v1_0.md` (388L) | Concurrent NN trial batching + tree-model parallelism. DRAFT. |
| `PROPOSAL_S96B_PERSISTENT_GPU_WORKERS_v1_0.md` (288L) | Persistent GPU worker processes for NN training. |
| `PROPOSAL_Multi_Model_Architecture_Addendum_F.md` (347L) | **IMPLEMENTED.** Subprocess isolation for OpenCL/CUDA compatibility — the origin of the GPU-isolation invariant. |
| `ADDENDUM_M_Multi_Model_Architecture.md` (380L) | Multi-model ML architecture v3.1.2. IMPLEMENTED. |
| `PROPOSAL_Addendum_M.md` (94L) | `scripts_coordinator.py` v1.4.0 as a universal ML script orchestrator. IMPLEMENTED & TESTED. |
| `ADDENDUM_N_scripts_coordinator_Universal.md` (264L) | The same orchestrator, fuller treatment. IMPLEMENTED. |
| `PROPOSAL_Addendum_H_Feature_Importance_Integration.md` (261L) | Feature-importance integration, phase 2. IMPLEMENTED. |
| `IMPLEMENTATION_CHECKLIST.md` (263L) | Multi-model v3.1.3 — **COMPLETE + PRODUCTION VALIDATED**, Dec 2025. |
| `PREDICTION_STRATEGIES_DOCUMENTED.md` (435L) | Prediction-rate improvement strategies, extracted from project documentation. |
| `PROPOSAL_S145_R1_Progressive_Empirical_Sweep.md` (352L) | Progressive sweep of seed IDs 0→2³² with cross-session survivor accumulation and persistent Optuna continuity. **Beta-approved conditionally; supersedes the rejected `PROPOSAL_S145_Complete_Seed_Space_Sweep.md`.** |

## 3.10 Infrastructure, GPU faults and cluster stability

| file | the question it answers |
|---|---|
| `PROPOSAL_Proxmox_LXC_Rig_Migration_v1_0.md` (150L) | The LXC migration plan for the three rigs. **Superseded on acceptance criteria** — see the reconciliation below. |
| `PROPOSAL_Infrastructure_Reconciliation_S172_v1_0.md` (172L) | **APPROVED (Beta)** — the Proxmox container strategy; LXC-vs-VM decision pending the rrig6600c trial. |
| `CLAUDE_CODE_INSTRUCTIONS_TOPOLOGY_DOC_CORRECTION_v2.md` (209L) | Correct the topology **documentation** to the boot-selector model. **DOCUMENTATION ONLY — no `.py`, `.sh` or `.json` may be edited.** Supersedes v1 and the earlier "CT100 IP Migration Cleanup" draft. |
| `PROPOSAL_FINAL_ROCm_HIP_Init_Fix_v2_1.md` (262L) | **APPROVED FOR IMPLEMENTATION, CRITICAL.** Joint Alpha + Beta: parallel HIP initialisation on ROCm. |
| `PROPOSAL_FREE_ALL_BLOCKS_REPLACEMENT_v1_0.md` (416L) | Replace `free_all_blocks()` in `sieve_gpu_worker.py`'s `_best_effort_gpu_cleanup()` with safe GPU memory management. |
| `PROPOSAL_S155_CUPY_POOL_FIX.md` (178L) | **P0 — blocks all production runs.** The CuPy memory-pool fix. |
| `PROPOSAL_S162_RRIG6600C_CRASH_ROOT_CAUSE_v1_0.md` (410L) | rrig6600c crash root cause + fix options across `pwc_worker_service.py`, `pwc_transport_tcp.py`, `sieve_filter.py`. |
| `PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md` (350L) | **APPROVED WITH MODIFICATIONS** — the persistent-worker-coordinator lifecycle fix. |
| `PROPOSAL_ZMQ_SQLITE_COORDINATOR_S158D_v1_0.md` (134L) | The ZMQ+SQLite distributed sieve coordinator (new standalone files; three existing files minimally modified). |
| `PROPOSAL_Job_Batching_Pipeline_Stability.md` / `_v2.md` | Job batching for pipeline stability. **v2 supersedes v1.** |
| `PROPOSAL_RAM_Disk_Data_Preloading_v1_0.md` (297L) | RAM-disk preloading for distributed workers. |
| `PROPOSAL_Unified_Ramdisk_Steps_3_and_5_v1_1.md` (639L) | Extending the ramdisk to Steps 3 and 5, plus lifecycle management. |
| `PROPOSAL_Infrastructure_Improvements_v1_0.md` (309L) | Preflight checks, smarter parameter tuning and GPU health monitoring, after a 2-hour WATCHER run failed silently (zero survivors + an undetected rig crash + SMU degradation). |
| `PROPOSAL_Incremental_Output_Writing_v1_0.md` (435L) | Incremental output writing for the window optimizer. |
| `PROPOSAL_NPZ_Auto_Conversion_Step2.md` (271L) | NPZ auto-conversion for Step 2.5 — a pipeline gap, not a blocker. |
| `IPC_SERIALIZATION_FIX_IMPLEMENTATION_GUIDE_S150.md` (286L) | The approved `slim_v1` IPC design. |
| `PROPOSAL_Documentation_Paradigm_Correction_v1_2.md` (182L) | ★ The functional-mimicry language cleanup — **the origin of the naming rule.** Documentation only. |

## 3.11 Session changelogs — **168 files, summarised as a group**

`SESSION_CHANGELOG_*.md`. **Two naming forms:** 162 dated
(`SESSION_CHANGELOG_YYYYMMDD[_TAG].md`, **2026-01-09 → 2026-08-02**) and 6 session-ID-only
(`SESSION_CHANGELOG_S16x*.md`), plus `SESSION_CHANGELOG_TEMPLATE.md`. Session IDs span **S1 → S184**.

**What they are good for, and only this:**
- **Establishing when a behaviour changed**, and under whose authority. `git log --grep` over a
  symbol plus the changelog for that date is the fastest route to *why*.
- **Recovering a governing decision that never became its own document.** The canonical example:
  `SESSION_CHANGELOG_20260307_S122.md:56` carries the ruling *"disabled per TB + S121 shuffle test"* —
  one of the three citations proving TRSE Rules B and C are advisory **by design**, not dropped
  wires. That ruling exists nowhere else.

**What they are not:** a status source. A changelog states what was believed on its date. Six months
of them do not compose into a current picture — use §1, §2 and `BACKLOG.md` for that.

**The eight most recent, which between them cover everything since dataset authority:**
`20260801_PHASE_6_P0_5` · `20260801_P0_5_Q2_CLOSURE` · `20260801_S1_ADMISSION_LIVENESS` ·
`20260801_S184_BOUNDED_PHASE_6` · `20260801_RESOLVED_EXECUTION_SET` · `20260801_ADMISSION_BINDING` ·
`20260802_CHAPTER_1_AND_2_CLOSURE` · `20260802_CHAPTER_2_CONTENT_GATE`.

## 3.12 Non-`.md` files in `docs/`

| file | what it is |
|---|---|
| `instructions.txt` (152K) · `Cluster_operating_manual.txt` (96K) | see §3.6 — both are load-bearing for skip semantics |
| `window_optimizer_integration_final.py` (100K) | **A `.py` file living in `docs/`** — a copy of the Step-1 integration layer, placed here as reference material. **Do not edit it as if it were the live module**; the live one is at the repo root. |
| `apply_chapter11_patch.sh` (4K) | Applies `PATCH_Chapter11_LLM_Update_v2.md` to Chapter 11. |
| `D6_RELEASE_GRADE_SMOKE_20260729.log` (16K) | Raw evidence for `D6_RELEASE_GRADE_CERTIFICATION_RECORD.md`. |
| `CHAPTER_1_WINDOW_OPTIMIZER.md.bak` · `CHAPTER_3_...bak` · `CHAPTER_4_...bak` · `CHAPTER_10_...v3.md.bak` | **Stale duplicates. Do not read; do not cite.** |
| `TEAM_BETA_REVIEW_kfolds_S100.docx` (12K) | Beta's k-folds review. **Binary — UNVERIFIED.** |
| `Distributed_PRNG_Pipeline_Overview_for_Novices.pdf` · `..._Technical_Addendum.pdf` | see §3.6. **UNVERIFIED.** |
| `phase6_evidence/*.json` (3) | see §3.4 |

---

# 4. CODE INVENTORY, BY ROLE

**Scope note:** a role map, not a completeness audit. The repo root holds several hundred files, the
great majority of them one-shot `apply_*.py` / `fix_*.py` patch scripts and `test_*.py` throwaways
from 180+ sessions; those are characterised as a class in §4.8 rather than enumerated.

## 4.1 FROZEN — reuse, never reimplement

**Importing these is mandatory; forking them is the defect they exist to prevent.**

| symbol / file | anchor | why frozen |
|---|---|---|
| `_l2_sort_key` | `utils/run_finalizer.py:690` | Highest **float32** score → lowest `trial_number` → constant-before-variable, *within a trial only*. **Comparing pre-rounding float64 is the defect this converts away.** |
| `_select_l2_winners` | `utils/run_finalizer.py:714` | Same-trial/same-mode collision raises `AccumulatorConsistencyError`. |
| `canonical_map_hash` | `utils/run_finalizer.py:486` | The map-identity anchor carried through the generation chain. |
| `CANONICAL_ARRAY_CONTRACT` | `utils/canonical_arrays.py:98` | The frozen 22-array NPZ contract. Consumed at `run_finalizer.py:803` and `canonical_arrays.py:582`. |
| `CANONICAL_RECORD_FIELDS` | `utils/canonical_arrays.py:143` **and** `utils/canonical_records.py:115` | The 24-field canonical record. **Defined in two modules — check which one your consumer imports.** Also consumed at `utils/checkpoint_d6_2.py:288`. |
| `utils/prng_encoding.py` | whole module | The shared registry-derived PRNG type encoding (Phase 0, `2389b61`). |
| The finalizer validators | `run_finalizer.py:522, 558, 585, 634, 665, 884, 1004, 1069, 1113, 1176` | Ten `_validate_*` functions incl. `_validate_chain` (`:1176`) and `_validate_current_pointer` (`:1113`). |
| **D3.5 finalizer-owned root symlinks** | `run_finalizer.py` ~`:1400-1404` | `bidirectional_survivors_all.npz` / `..._binary.npz` are **symlinks the finalizer owns.** A regular file appearing there makes `finalize_run` raise `PublicationError` — this is what made the D6.1 briefed repair unsafe. |

## 4.2 `miner/` — the RANGE-MINER engine (Step 2)

| file | role |
|---|---|
| `range_miner_protocol.py` | Length-prefixed JSON framing (4-byte big-endian + compact UTF-8, 64 MB cap); 8 message types; `from_dict()` filters unknown kwargs via `dataclasses.fields()`; unknown `message_type` → `ValueError`. **All envelope fields carry defaults** — deliberately unlike `persistent/pwc_protocol.py`. |
| `range_miner_worker.py` | The per-GPU daemon: READY handshake, sub-stripe loop, `ResidueResolver`, threshold consumption and effective-threshold reporting. |
| `range_miner_coordinator.py` | Stripe assignment, admission, staging, lease expiry, the retry matrix, `serve_trial`. |
| `range_miner_npz_writer.py` | Trial assembly and the 22-array NPZ write-back; `AssemblingPhase5Sink`. |
| `assembly_backends.py` | The frozen two-backend interface; `serial_reference` (default). |
| `assembly_shard_worker.py` | `process_sharded`'s shard worker. Owns `assert_cpu_only()` and `_FORBIDDEN_GPU_MODULES` — **the import gate exercises these; it does not duplicate them.** |
| `dataset_authority.py` | Pointer resolution, the run-start freeze, per-node provisioning and on-target verification; `DatasetProvisioningError`. |
| `step1_ingress.py` | Miner candidates → the Step-1 accumulator (D6). |

## 4.3 `utils/` — the shared authorities

`run_finalizer.py` (the generation chain and publication) · `canonical_arrays.py` (24→22 columnizer
+ structural validator) · `canonical_records.py` (canonical record normalisation) ·
`checkpoint_d6_2.py` (24-field checkpoint, both digest layers, reconciliation, path confinement) ·
`prng_encoding.py` · `survivor_loader.py` · `metrics_extractor.py`. **Most of §4.1 lives here.**

## 4.4 Pipeline steps (repo root)

| step | primary module(s) |
|---|---|
| 0 — Regime Segmentation (TRSE) | `trse_step0.py`, `trse_calibration_probe.py`, `trse_entropy_probe.py`, `step0_heuristic_validation.py` |
| 1 — Window Optimizer | `window_optimizer.py`, `window_optimizer_bayesian.py`, `window_optimizer_integration_final.py` |
| 2 — Bidirectional Sieve | **`miner/`** (current) · legacy: `sieve_filter.py`, `sieve_gpu_worker.py`, `reverse_sieve_filter.py`, `hybrid_strategy.py` · kernels: `prng_registry.py` |
| 2.5 — Scorer Meta-Optimizer | `run_scorer_meta_optimizer.py` / `.sh`, `generate_scorer_jobs.py`, `scorer_trial_worker.py` |
| 3 — Full Scoring | `run_step3_full_scoring.sh`, `generate_full_scoring_jobs.py`, `full_scoring_worker.py`, `survivor_scorer.py` |
| 4 — Adaptive Meta-Optimizer | `adaptive_meta_optimizer.py` |
| 5 — Anti-Overfit Training | `meta_prediction_optimizer_anti_overfit.py`, `train_single_trial.py`, `nn_gpu_worker.py`, `inner_episode_trainer.py` |
| 6 — Prediction Generator | `prediction_generator.py`, `build_pools.py`, `evaluate_pools.py`, `backtest_pools.py` |
| Feedback (Ch. 13) | `chapter_13_orchestrator.py`, `chapter_13_triggers.py`, `chapter_13_acceptance.py`, `chapter_13_diagnostics.py`, `chapter_13_llm_advisor.py`, `draw_ingestion_daemon.py`, `per_survivor_attribution.py` |
| Ch. 14 diagnostics | `training_diagnostics.py`, `training_health_check.py`, `diagnostics_llm_analyzer.py`, `diagnostics_analysis_schema.py` |
| Selfplay | `selfplay_orchestrator.py`, `policy_conditioned_episode.py`, `policy_transform.py`, `reinforcement_engine.py` |
| Fleet authority | `execution_set.py` |
| Legacy engines (non-certifying) | `persistent_worker_coordinator.py`, `persistent/`, `zmq_sqlite_coordinator.py`, `zmq_sqlite_worker.py`, `coordinator.py`, `distributed_worker.py` |
| Data | `daily3_scraper.py`, `pa_pick3_scraper.py`, `convert_survivors_to_binary.py`, `validate_survivors.py` |

## 4.5 `agents/`

`watcher_agent.py` (the orchestrator; `STEP_SCRIPTS`/`STEP_MANIFESTS`/`STEP_NAMES` at `:387-417`) ·
`watcher_dispatch.py` · `watcher_registry_hooks.py` · `fingerprint_registry.py` · `agent_core.py` ·
`agent_decision.py` · `doctrine.py` · `full_agent_context.py` · `prompt_builder.py` ·
`registry_inspector.py` · `threshold_guardrail.py` · `progress_display.py` · subpackages
`contexts/` (bundle factory), `step_runner/`, `manifest/`, `parameters/`, `pipeline/`, `registry/`,
`runtime/`, `safety/`, `prompts/`, `data/`, `history/`.
**Note:** `agents/` also holds three stale in-place backups — `agent_core.py.bak2`,
`watcher_agent.py.bak2`, `watcher_agent.py(bakpregrammar)`.

## 4.6 `tests/` — 33 entries

Phase/deliverable-scoped gates, one per governed deliverable:
`test_s172_phase1_scaffolding` · `phase2_protocol` · `phase3_worker` (17/17) ·
`phase4_coordinator` (63/63) · `phase5_d0` · `d1_workflow` · `d1_engine` ·
`d2_directional_uniqueness` · `d3_columnizer` · `d3_0_encoding_contract` ·
`d3_25_candidate_ingress` · `d3_5_finalizer` · `d4_serial_backend` · `d5_process_sharded` ·
`d6_production_adapter` · `d6_threshold_path` · `d6_1_flush_durability` ·
`d6_2_checkpoint_reconciliation` · `phase6_p05_dataset_authority` (38/38 `--fleet`) ·
`process_sharded_import_gate` · `threshold_propagation` · `resolved_execution_set` (34/34) ·
`admission_binding` (20/20) · `admission_liveness` (16/16) · `test_chapter1_p0_corrections` (12/12) ·
`test_chapter2_content_gate` (12/12) · `test_prng_encoding` · `test_watcher_llm_integration` ·
`smoke_s172_phase5_d6_zeus_single_gpu.py` · plus `phase6/` (`wall_ab_gate.py`,
`known_answer_gate.py`) and `fixtures/`.
**`tests/test_s172_phase4_coordinator.py` gate 22 is the coexistence gate** — it reds on any stray
untracked `.py`, which is why every deliverable registers its new paths in that whitelist.

## 4.7 `scripts/`

`provision_dataset_fleet.py` · `verify_dataset_publication.py` · `extract_search_bounds_snapshot.py` ·
`apply_caps.py` · boot-notify installers (`install_boot_notify_amd.sh`, `install_boot_notify_pzeus.sh`,
`update_boot_notify_v2.sh`, `cluster_boot_notify.sh`) · rig probes (`probe_phase_A_amd.sh`,
`probe_phase_A_rtx.sh`, `probe_phase_C_stability.sh`) · Telegram diagnostics.

## 4.8 The one-shot patch corpus (repo root)

**~250 `apply_s*.py` / `fix_s*.py` / `patch_*.py` / `apply_*.sh` files**, each a session-scoped,
already-applied edit script named for its session (S73 → S174), plus **~200
`test_*.py` / `create_*_test.py` / `launch_*.sh`** exploratory harnesses from the same period.
**Their value is forensic** — `apply_s149a_rocr_isolation.py` is the evidence that the S149 ROCR
ruling was implemented. **They are not a runnable interface and must not be re-executed.**
The root also holds **~40 `tmp*.json` scratch files**, several `.bak` / `.save` / `(broken`
copies of `sieve_filter.py`, `prng_registry.py`, `coordinator.py`, `survivor_scorer.py` and
`window_optimizer.py`, and four `agents_*.tar.gz` archives.

---

# 5. THE STEP MAP — structural inventory

> **⚠ SCOPE LIMIT.** This section records what each map **declares** and how many parameters each
> manifest **lists**. It does **not** trace parameter reachability, does **not** determine whether a
> declared parameter is consumed, and does **not** call anything unused. **A count and a filename
> are facts; "nothing reads this" is a claim, and it is out of scope here.**
>
> That work is a separate brief and must be done **with the governance trail in hand** — a parameter
> that looks dead is very often a known, escalated, mid-remediation item. Chapter 3's audit reported
> the Step-2 objective as blind to 7 of 11 sampled dimensions; the condition was already diagnosed
> with live Zeus stats in `TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md` and is the subject of an approved
> v4.1→v4.2→v4.3 remediation. **Reported as a discovery, it would have told Beta about its own ruling.**

**Sources:** `agents/watcher_agent.py` — `STEP_SCRIPTS` at `:387-395`, `STEP_MANIFESTS` at
`:398-406`, `STEP_NAMES` at `:409-417` — and the manifests under `agent_manifests/`. All counts
below were parsed from the live JSON in this pass.

| step | `STEP_NAMES` | `STEP_MANIFESTS` entry | manifest version / `pipeline_step` | `STEP_SCRIPTS` entry | manifest `actions` → scripts | `default_params` count | documenting chapter |
|---|---|---|---|---|---|---|---|
| **0** | Regime Segmentation (TRSE) | `trse.json` | 1.15.1 / 0 | `trse_step0.py` | *(no `actions` key)* | **7** | **none** — the spec is `TRSE_v1_15_SPEC.md` + `TRSE_INTEGRATION_PLAN_S121.md` |
| **1** | Window Optimizer | `window_optimizer.json` | 1.8.0 / 1 | `window_optimizer.py` | `window_optimizer.py` ✅ **agrees** | **25** | **Chapter 1** ✅ audited, closed |
| **2** | Scorer Meta-Optimizer | `scorer_meta.json` | 1.3.0 / 2 | `run_scorer_meta_optimizer.sh` | `generate_scorer_jobs.py`, `scorer_trial_worker.py` ⚠ **DIVERGES** | **8** | **Chapter 3** — numbered **3**, documents **Step 2.5** ⚠ |
| **3** | Full Scoring | `full_scoring.json` | 1.3.0 / 3 | `run_step3_full_scoring.sh` | `generate_full_scoring_jobs.py`, `full_scoring_worker.py`, `aggregate_scoring_results.py` ⚠ **DIVERGES** | **10** | **Chapter 4** ❌ unaudited |
| **4** | ML Meta-Optimizer | `ml_meta.json` | 2.0.0 / 4 | `adaptive_meta_optimizer.py` | `adaptive_meta_optimizer.py` ✅ **agrees** | **4** | **Chapter 5** ❌ unaudited |
| **5** | Anti-Overfit Training | `reinforcement.json` | 1.10.0 / 5 | `meta_prediction_optimizer_anti_overfit.py` | *(no `actions` key)* | **10** | **Chapter 6** ❌ unaudited |
| **6** | Prediction Generator | `prediction.json` | 1.5.0 / 6 | `prediction_generator.py` | `prediction_generator.py` ✅ **agrees** | **7** | **Chapter 7** ❌ unaudited |

**Total declared `default_params` across the seven step manifests: 71.**
Every manifest's `pipeline_step` field matches its `STEP_MANIFESTS` key — **no key/field mismatch.**

## 5.1 Divergences between the two maps — reported, not diagnosed

1. **Step 2 — `STEP_MANIFESTS[2]` and `STEP_SCRIPTS[2]` name different things.** The manifest is
   `scorer_meta.json`, whose `actions` invoke `generate_scorer_jobs.py` and `scorer_trial_worker.py`;
   `STEP_SCRIPTS[2]` is `run_scorer_meta_optimizer.sh`, which appears in no manifest action.
   **This divergence is how a soak hazard reached launch day** — see `CHAPTER_3_ALIGNMENT_AUDIT.md`.
2. **Step 3 has the identical shape.** `STEP_SCRIPTS[3]` is `run_step3_full_scoring.sh`; the
   manifest's three actions name `generate_full_scoring_jobs.py`, `full_scoring_worker.py` and
   `aggregate_scoring_results.py`, none of which is the shell script. **Recorded here as a
   structural fact; see §7 gap 2 for what was and was not searched.**
3. **Steps 0 and 5 declare no `actions` at all**, so the two maps cannot be compared for them.
   `trse.json` instead carries `skip_on_fail: true` with a stated `skip_on_fail_reason` — that
   silent-failure behaviour is **architected**, per `TRSE_INTEGRATION_PLAN_S121.md` §2C.

## 5.2 Chapter-numbering hazards

- **Chapter numbers are not step numbers.** Chapter 3 documents **Step 2.5 / WATCHER step 2**. The
  bidirectional sieve documented in **Chapter 2** runs inside **Step 1**, not WATCHER step 2.
- **"Phase 7" is overloaded.** The Phase 7 marked COMPLETE in Chapters 10, 12 and 13 and in
  `TODO_PHASE7_WATCHER_INTEGRATION_REVISED_v3.md` is **WATCHER dispatch integration (Feb 2026)**.
  **S172 Phase 7 is the 26-GPU saturation + WATCHER soak** and is a different thing entirely.
- **Step 0 has no documenting chapter** (§7 gap 1).

## 5.3 Manifest inventory notes

- `agent_manifests/` holds **9 files**: the 7 step manifests, `definitions.json`, and
  `scorer_meta.json.bak` (stale).
- **All `agent_manifests/*.json` match `.gitignore:41` (`*.json`)** — confirmed by
  `git check-ignore -v --no-index`. The 7 step manifests are nonetheless **tracked** (force-added).
  `agent_manifests/trse.json` was force-added at `93918f5` (2026-08-01) and now has **exactly one
  commit** of history.
- **`definitions.json` is the only manifest still untracked and ignored.** It carries no
  `default_params`; its keys are `schema_version`, `pipeline_steps`, `sidecar_schema`,
  `watcher_protocol`, `description`, `updated_at`. **A fresh clone does not have it.**
- **⚠ This corrects a stale statement in the `tfm-project-facts` skill**, which says
  `agent_manifests/trse.json` is the only gitignored manifest and has no git history. As of
  `93918f5` it is tracked and has history; `definitions.json` is now the untracked one. **Noted for
  Alpha — not fixed here.**

---

# 6. SUPERSEDED / DO-NOT-CITE

Anything below could be mistaken for current by a reader who found it by filename.

## 6.1 Superseded facts and targets

| do not cite | cite instead |
|---|---|
| **R² as the ML objective** (0.000155 — zero signal) | `0.50·hit@20 + 0.30·hit@100 + 0.15·hit@300 + 0.05·pool_coverage` (S140b) |
| **`holdout_hits` as the ML target** | `holdout_quality` (`1cb90aa`; `S111_TEAM_BETA_BRIEFING.md`) |
| **"~62 features"**, and `feature_importance.py`'s 60-name list | **91 extracted / 89 trained**; `S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md` |
| `full_scoring_worker.py` "50 features" | as above |
| **`bidirectional_survivors.json` as survivor data** | the 22-array NPZ contract |
| **PWC / ZMQ as certifying comparators** | RANGE-MINER; PWC is a flag-selectable **non-certifying** diagnostic, and PWC hybrid is additionally quarantined |
| `RUNTIME_DATASET_PROVISIONING_CONTRACT.md`'s `expected_sha256` **as static config** | run-start freeze + fleet consistency (`TEAM_ALPHA_DATASET_LIFECYCLE_FINDINGS.md`) |
| scraper `--rewrite` mode | eliminated by owner decision (`TEAM_ALPHA_APPEND_ONLY_SIMPLIFICATION.md`) |
| **"RX 6600" on the rigs** | they are **RX 6600 XT**, gfx1032, 32 CUs, 8 per rig |
| `HYBRID_SKIP_BOUND_AUDIT.md:318` "semantics unspecified" | **FALSE** — `SKIP_SEMANTICS_SEARCH_v1.md`; `instructions.txt:1182-1183` |
| `TRSE_v1_15_SPEC.md` describing Rules B and C as **applied** | they are advisory by design (`TEAM_ALPHA_TRSE_FIX_PROPOSAL.md`) |
| `THRESHOLD_GOVERNANCE.md` synthetic-era defaults | `THRESHOLD_CALIBRATION_FINDINGS_S148.md` |
| `run_full_scoring.sh` | `run_step3_full_scoring.sh` |
| **Run-1 / sweep-era pool-size and threshold figures in any `S1xx_CHAT_PROMPT.md` or `TODO_MASTER_*`** | current manifests and `BACKLOG.md` |

## 6.2 ★ The v1 catalog's "Runtime Data" table — **DO NOT CARRY FORWARD**

REV2 §1.5 named this table a candidate for retirement and asked that its figures be verified before
carrying them forward. **They were, and they do not survive.** The February-2026 table against what
is on disk today at HEAD `9e79a26`:

| catalog claim | measured 2026-08-03 |
|---|---|
| `bidirectional_survivors.json` — **258 MB** | **2 bytes** (mtime May 11). Also superseded as a data carrier (§6.1). |
| `survivors_with_scores.json` — **500+ MB** | **621 KB** (mtime Mar 11) |
| `bidirectional_survivors_binary.npz` — 0.6 MB | **not present at the repo root.** Since D3.5 that path and `bidirectional_survivors_all.npz` are **finalizer-owned compatibility symlinks** created at publication, not standing files; `.s172_accumulator/` is not present in this working tree either. |
| Optuna study DBs — 10–50 MB each | `optuna_studies/` totals **8.9 MB**; individual `window_opt_*.db` are **112 KB** |
| "Agent Manifests (JSON) — 6" | **7 tracked + 1 untracked** (§5.3) |

**These are point-in-time filesystem measurements, not defect claims.** The general lesson for the
catalog is that **runtime-artifact sizes are not catalogue facts**, and none are reproduced in this
regeneration.

## 6.3 Superseded document versions (read the last one only)

- **`PROPOSAL_Unified_Agent_Context_Framework_v3_2_*`** — 10 versions + 2 addenda → **v3.2.10**
- **`CHAPTER_13_IMPLEMENTATION_PROGRESS_*`** — 14 versions → **v3.9**
- **`CHAPTER_13_LIVE_FEEDBACK_LOOP.md`** → **`_v1_1.md`**
- **`CHAPTER_10_AUTONOMOUS_AGENT_FRAMEWORK_v2.md`** → **`_v3.md`**
- **`COMPLETE_OPERATING_GUIDE_v1_1.md`** → **`_v2_0.md`**
- **`PROPOSAL_S172_RANGE_MINER_v1_4_4.md`** → **`_v1_4_5.md`** (v1.4.4 authoritative only where v1.4.5 marks PRESERVED)
- **`PROPOSAL_Job_Batching_Pipeline_Stability.md`** → **`_v2.md`**
- **`PROPOSAL_LLM_Reasoning_Refactor_v1_0.md`** → **`_v1_1.md`**
- **`WATCHER_PHASE1_PATCH_FOR_REVIEW.md`** → **`_v1.1_FINAL.md`**
- **`PROPOSAL_WATCHER_KPI_GOVERNANCE_STATES_v1_0.md`** → **v1.1 + the v1.2 addendum**
- **`TODO_PHASE7_WATCHER_INTEGRATION_REVISED.md`** → **`_v3.md`**
- **`DOCUMENTATION_INDEX_v1_0.md`** → **`_v1_1.md`** → **this catalog**
- **`PROPOSAL_Step7_PostPipeline_Export_v1_1.md`** → replaced by **`NOTE_Step7_Not_Required_for_Autonomy.md`**
- **`PROPOSAL_Proxmox_LXC_Rig_Migration_v1_0.md`** → superseded on acceptance criteria by **`PROPOSAL_Infrastructure_Reconciliation_S172_v1_0.md`**
- **`audit/S172_PHASE3/PHASE3_INITIAL_REVIEW.md`** → **`PHASE3_FIX_BRIEF_REV2.md`** → **`PHASE3_FINAL_APPROVAL_REV3.md`**
- **`CHAPTER_1_PATCH_S114.md`** — ⛔ **UNMERGED *and* SUPERSEDED.** Its central mechanism was deleted from the code while it sat unmerged. History only.
- **`PROVENANCE_DISPOSITION_ACCUMULATOR_20260725.md`** — a copy **without** the REV2 banner is the superseded draft.
- **All four `docs/*.md.bak` files** and **`agent_manifests/scorer_meta.json.bak`** — stale duplicates.
- **`CHAPTER_13_IMPLEMENTATION_PROGRESS_v1_2 .md`** (space in the filename) — duplicate of the v1.2 line.

## 6.4 Whole-document staleness warnings

- **`CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md`, `REMOTE_NODE_SETUP_CHECKLIST.md`, and every
  `TODO_MASTER_*` / `S1xx_CHAT_PROMPT.md`** describe the **bare-metal** cluster. The rigs are
  boot-selectors; when booted into Proxmox the worker endpoints are the CT100 addresses
  `.122` / `.156` / `.164`. **`distributed_config.json`'s bare-metal addresses are deliberate**
  (they match the *default* boot target) and **must not be "corrected".**
- **Everything dated before 2026-07-07** predates RANGE-MINER replacing the Step-2 engine.
- **Everything dated before 2026-08-01** predates dataset authority (P0/P0.5) and the Resolved
  Execution Set.
- **`docs/README.md` and `docs/proposals/README.md`** open with **"Seed Reconstruction"** /
  "Reverse-engineer PRNG behavior" framing. The project's naming rule is **functional mimicry, not
  seed recovery** (`PROPOSAL_Documentation_Paradigm_Correction_v1_2.md`). **Noted for Alpha — not
  fixed here.**

---

# 7. KNOWN GAPS

**Every entry names the search that establishes it.** Nothing is listed here by assumption. All
absence statements are **repo-scoped** and carry the standing caveat that host state and the ser8
archives were not searched.

| # | gap | the search that establishes it |
|---|---|---|
| 1 | **Step 0 has no documenting chapter.** | `ls docs/CHAPTER_*` enumerated all 46 chapter-family files in this pass (§2). There are chapters for steps 1–6 and for the registry, infrastructure, agents, features, WATCHER, feedback and diagnostics. None documents Step 0; TRSE's design authority is `TRSE_v1_15_SPEC.md` + `TRSE_INTEGRATION_PLAN_S121.md`, neither of which is a chapter. |
| 2 | **The `STEP_SCRIPTS[3]` ↔ `full_scoring.json` divergence is not separately reported.** | The Step-2 divergence is documented (`CHAPTER_3_ALIGNMENT_AUDIT.md`, and REV2 of this catalog's own brief). Step 3 has the identical shape (§5.1 item 2) and was not found named in any of the 372 documents indexed individually in this pass. **Stated as "not found in the documents indexed here", not as "undocumented"** — Chapter 4 is unaudited and was not read line-by-line. |
| 3 | **The CA draw-procedures PDF is not in the repository.** | `ls docs/*.pdf` returns exactly two files, both pipeline-overview documents. The *California State Lottery — Daily & SuperLotto Plus Draw Procedures* (eff. 2021-06-09) is cited as external authoritative evidence in `TEAM_ALPHA_PUSHBACK_ORDERING_AND_THRESHOLD_REGRESSION.md` and underpins the physical model of skip. **It exists only as a citation.** Already tracked in `BACKLOG.md`. |
| 4 | **`agent_manifests/definitions.json` is untracked and gitignored — a fresh clone does not have it.** | `git ls-files agent_manifests/` returns 7 files; `ls agent_manifests/` returns 9; `git check-ignore -v --no-index` matches `.gitignore:41:*.json` for all of them. **Its role is not described in any of the 372 documents indexed here**, and no repo-scoped audit can see its contents. |
| 5 | **`daily3_scraper.py` has no design document.** | `TEAM_ALPHA_SCRAPER_RECENT_SAFETY_NOTICE.md` establishes it was **never in git history and not gitignored**, and is only now being tracked. `DAILY3_CONSUMER_CONTRACT_v1.md` documents what consumers require of the *dataset*; `DATASET_PUBLICATION_SCHEMA_v1.md` freezes the *publication* schema. **Neither documents the producer**, and 6-P2 — the brief that would — is a REV4 **draft pending Beta**. |
| 6 | **Eleven chapters have never been audited against source.** | Enumerated in §2 from the file headers read in this pass; `BACKLOG.md` §1 names 3, 5, 6, 8 and 13 explicitly. The gap is wider than that backlog entry: **Chapters 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 and 14 — roughly 11,000 lines — are unverified**, against a measured base rate of 9/41 (Chapter 1) and 17/55 (Chapter 3). |
| 7 | **`TEAM_BETA_REVIEW_kfolds_S100.docx` and the two PDFs were not read.** | Binary formats; this pass read text. Their subjects are inferred from filenames only and are marked **UNVERIFIED** in §1.2, §3.6 and §3.12. **If the k-folds review contains a binding ruling, this catalog does not carry it.** |
| 8 | **Whether the S179-authorised KPI governance implementation has landed was not established.** | `TB_RULING_S179_IMPLEMENTATION_AUTH.md` reads "APPROVED FOR IMPLEMENTATION WITH THREE BINDING CODE-LEVEL CONDITIONS". No `SESSION_CHANGELOG_*` naming KPI governance after S179 was found in the filename sweep, and tracing the code is **out of this catalog's scope** — it is an index, not an audit. **This is not a claim that it did not land.** |

---

# 8. VERIFICATION-INTEGRITY CONTROLS (VIR-1…6)

- **execution proof:** `docs/` contains **562 files** (552 top-level + 10 across three
  subdirectories, by `find docs -type f`). **394 indexed individually; 168 session changelogs
  summarised as a group; 562 accounted for.** Every individually-indexed file was opened — heading
  structure plus opening section, per REV2 §2 — not summarised from its filename.
- **clean control:** `NOT_APPLICABLE` — this deliverable produces an index, not a detector.
- **fault-injection control:** `NOT_APPLICABLE` — same reason.
- **completion sentinel:** **`PASS`.** All 562 files are accounted for and all seven REV2 sections
  are delivered. Three files (`TEAM_BETA_REVIEW_kfolds_S100.docx` and the two PDFs) are indexed by
  filename with their contents explicitly marked **UNVERIFIED** and recorded in §7 gap 7, rather
  than silently omitted.
- **unavailable-observer behaviour:** binary formats and gitignored files are named and marked, not
  inferred. Anything not opened is labelled.
- **audit claim scope:** **repo-scoped**, HEAD `9e79a26`. This catalog indexes what is committed.
  **Host state (systemd, cron, deployed uncommitted files) and the pre-repository ser8 archives are
  out of scope and are not implied.**
- **searched surfaces:** `docs/` (all 552 top-level files) · `docs/audit/S172_PHASE3/` ·
  `docs/proposals/` (incl. the empty `archived/`) · `docs/phase6_evidence/` · the repo-root file
  listing · `agent_manifests/` (all 9 files; the 8 non-`.bak` parsed) ·
  `agents/watcher_agent.py:375-420` · `miner/` · `utils/` · `tests/` · `scripts/` · `.gitignore` ·
  `git log` for the 26 commits cited.
- **unavailable surfaces:** `agent_manifests/definitions.json` (gitignored; contents not
  catalogued) · `daily3_scraper.py` history (never in git) · the two PDFs and the `.docx` (binary,
  unread) · **host state and the ser8 pre-repository archives (not searched, out of scope).**

---

*Regenerated 2026-08-03 by Claude Code on VM101 under `docs/CLAUDE_CODE_INSTRUCTIONS_PROJECT_CATALOG_REGENERATION.md` REV2. Read-only except this file: no code, config, manifest, chapter or gate was changed; nothing was committed or pushed; WATCHER and the pipeline were not run; no finding was fixed.*
