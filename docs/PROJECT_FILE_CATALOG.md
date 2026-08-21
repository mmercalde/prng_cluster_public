# PROJECT FILE CATALOG — INTENT-INDEXED

**TFM / `distributed_prng_analysis`. Regenerated 2026-08-20 on VM101 (`192.168.3.177`) as `michael`, venv `~/venvs/torch`, at HEAD `0a4cef1`.**
**Authority:** `docs/CLAUDE_CODE_INSTRUCTIONS_PROJECT_CATALOG_REGENERATION.md` (REV2), as delta'd by
`docs/CLAUDE_CODE_INSTRUCTIONS_CATALOG_REGENERATION_20260820.md`.
**Replaces** the catalog regenerated 2026-08-03 at `9e79a26` (corrected `f8cb1c5`), which predates the
entire Gate-12 campaign, the drain-remedy series, the field-6 repair and four Beta rulings.

> **Anchor note.** The regeneration brief names HEAD `1bf49a5`. `0a4cef1` — the brief's own commit —
> landed on top of it before this pass ran, so this catalog is anchored at `0a4cef1` as REV2 §2
> requires ("regenerate at current HEAD"). No content difference: `0a4cef1` adds only the brief.

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
> missed. **§1.0 carries the two STANDING RULES** that otherwise live only inside ruling bodies.
>
> **This catalog is a snapshot, and a snapshot expires.** Anchors here were read on 2026-08-20.
> Re-verify before acting on any of them.

**Coverage (execution proof).** `docs/` contains **699 files** — 688 at top level plus 11 across four
subdirectories (`audit/S172_PHASE3/` 4, `phase6_evidence/` 3, `proposals/` 3, `reference/` 1).
**All 699 are accounted for: 517 indexed individually** (493 top-level `.md`, 13 top-level non-`.md`,
11 subdirectory files) **and 181 committed session changelogs summarised as a group** under §3.13, as
REV2 §1.3 permits. The 182nd changelog, `SESSION_CHANGELOG_20260819_S1.md`, is **untracked,
unattributed and deliberately NOT indexed as governance** — see §7 gap 1.

**Audit claim scope: repo-scoped.** This catalog indexes what is committed to
`/home/michael/distributed_prng_analysis` at HEAD `0a4cef1`. **Host state (systemd units, cron,
deployed uncommitted files), the `~/dashboard_work` brief tree, and the pre-repository archives on
ser8 are OUT OF SCOPE and are not implied by anything below.** Gitignored files are named where
relevant but their contents are not catalogued.

**Currency headline, because two things in the previous catalog are now wrong.**
**Gate 12 is PASSED** — attempt 9, 2026-08-17, run `distributed_config_t1_554463d3`, launch commit
**`e9ca800`, tagged `gate12-passed-attempt9`**, which is the certified pre-change reference for all
window-anchor work. **Phase 7 is UNBLOCKED** and the field-6 repair that had to precede it landed at
`d8b21e3`. Anywhere an older document describes Gate 12 as pending, held or failing, or Phase 7 as
blocked, it is describing a state that ended on 2026-08-17. See §6.5.

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

## 1.0 ⚑ STANDING RULES — binding today, and they live only inside ruling bodies

**Two rules currently have no home outside the ruling that created them. A session that misses
either will red a certified gate or violate a naming ruling without knowing why.**

### SR-1 — `DECLARED_CHANGED` maintenance (`TB_RULING_FIELD6_IMPLEMENTATION.md` §2, 2026-08-20)

> **Any authorized commit that changes a definition in `miner/range_miner_coordinator.py` covered by
> a historical exact live-vs-anchor scope gate MUST update every affected historical
> `DECLARED_CHANGED` set before that commit is accepted.**

Four binding constraints: **(1)** do not move the historical anchor forward; **(2)** do not relax
`changed == DECLARED_CHANGED` to subset/superset logic; **(3)** add only definitions actually
changed by the newly authorized work; **(4)** every added entry carries **provenance** naming the
later change that owns it, so an R-1 or MP-1 suite never falsely claims authorship of a later change.
Reverse protection is retained: a later revert must red the old suite as *declared-but-unchanged*,
never silently stay green. Beta's preferred future housekeeping (immutable original owned set +
provenance-tagged post-anchor delta set, assertion exact over their union) is **recorded, not
authorised** — it was explicitly not to be introduced in Field 6.

### SR-2 — session-changelog naming (`TB_RULING_CHANGELOG_NUMBERING.md`, 2026-08-18)

`SESSION_CHANGELOG_YYYYMMDD_<TOPIC>.md` is **canonical**. **No new S-numbers.** No retro-numbering of
the three existing topic-named sessions. At SER8-backlog import, **one** deliberate reconciliation
pass with a single explicit Beta ruling: restore the S-sequence or formally retire it. Date+topic is
canonical until then — *not* a temporary exception. The reason is load-bearing: `S185` is only the
**highest visible** number while ~20 SER8-only changelogs await backfill, so guessing `S186` converts
unknown history into asserted governance state.

## 1.1 Team Beta rulings and ruling requests (`TB_*`) — the current era

| file | what it rules on / asks | disposition | still binding? |
|---|---|---|---|
| `TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md` (222L) | ★ **The ruling that ended the Gate-12 campaign.** Attempt 9 accepted as the first successful production-class Gate-12 run: fresh zero-credit cursor → 4 stages / 128 stripes → saturation satisfied at 25 compute-active → **zero lease expiries** → certified `[0, 2^31)` coverage both modes. Also rules Fields 1 and 2 **MISSED AS WRITTEN** and refuses to renormalise them into passes; Field 6 **UNOBSERVED (instrumentation-output defect)**; publication symlinks no blocker but `.s172_accumulator/generations/` is durable data plane needing a backup policy. | **RULED**, 2026-08-17. Pairs with `SESSION_CHANGELOG_20260817_GATE12_ATTEMPT9.md`. Ruled sequence: anchor `e9ca800` → window-anchor merge → field-6 repair → Phase 7. | **YES** — Gate 12 PASSED, MP-1 drain defect CLOSED (do not reopen R-1), Phase 7 and the window-anchor merge **UNBLOCKED** |
| `TB_RULING_REQUEST_WINDOW_ANCHOR_SEQUENCING.md` (106L) | ★ **Alpha reporting that a Beta ruling presupposed an artifact that does not exist.** Sequencing item 2 said "perform the window-anchor production merge"; Alpha searched both remotes (all refs/branches/tags), VM101 (branches, worktrees, stashes, date-keyed `find`), ser8 `~/Downloads` and prior sessions, and found **no design artifact and no implementation branch** — only the problem characterization (Chapter 2 **F-4** at `:1133`, `AUDIT_STEP1_OFFSET_REACH.md`, report §D.1). | **RULED** → below. | superseded by the ruling |
| `TB_RULING_WINDOW_ANCHOR_SEQUENCING.md` (66L) | **APPROVED WITH TWO CORRECTIONS.** Alpha's finding accepted — Beta's "merge" wording was a sequencing error, no hidden Beta design exists. Proposal phase authorized. Scope **(a) separation + (b) hybrid semantics MANDATORY, (c) `skip_min`/`skip_max` hybrid search bounds OUT.** Correction: *"forward hybrids receive no offset"* is **too broad** — `lcg32_hybrid` and `pcg32_hybrid` **do** carry a phase argument. **Semantic contract BINDING:** `window_anchor` = which observed records form the residue window; `generator_phase` = how many generator advances precede the first comparison; **never reconstructed from one another.** Kernel ABI **FROZEN for v1**; `generator_phase = 0`, not an Optuna dimension; `offset = train_history_len` (Step 3) is a **consumer continuation law** and out of scope. Phase-7 soak classified **NON-CERTIFYING for anchor semantics.** | **RULED**, 2026-08-17. | **YES** — this is the governing sequence for the window-anchor track |
| `PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_0.md` (215L) | The design: one scalar `offset` doing two unrelated jobs (host residue-slice selection at `range_miner_worker.py:649-650`; device pre-advance via `BuildContext(offset)` → `ScalarArg(int32)` → `prng_registry.py:974-976`), coherent only at skip = 0. Per-variant capability matrix, fused-key hard reject, derived anchor domain, phase pinned 0, ABI-v2 dependency recorded, five questions for Beta. | **RULED** → below. **SUPERSEDED BY `_v1_1.md`** (retained for audit). | superseded |
| `TB_RULING_WINDOW_ANCHOR_PROPOSAL_V1_0.md` (77L) | ★ **ARCHITECTURE ACCEPTED; REVISION REQUIRED.** All five questions ruled. **One genuine semantic error: `[0,149]` as an anchor range is REJECTED** — 100 is the historical **anchor** ceiling, 149 the historical **record-envelope** ceiling (anchor 149 + window 50 reaches record 198, outside history). Correct control domain: **`control_anchor = [0, min(100, N_filtered − window_size)]`**. Also: `search_bounds.offset` **removed outright** (no tombstone — JSON has no comments); metadata addition approved with the **22-array wall staying closed**; `anchor_era` is **provenance, not authority**; header-only freezing of `reverse_sieve_filter.py` **not sufficient**; two sequential briefs (I then II). | **RULED**, 2026-08-18. | **YES** |
| `PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md` (257L) | **The current design.** Bounded corrections only, no new semantics: Q4 fixed everywhere plus a **terminology law** (anchor = start index, envelope = reachable records, never interchanged); the error encoded as a **permanent regression test** (AC3: anchor 149 with window 50 is NOT in `control_era`); exact machine representation `{min, max_cap}` with `effective_max = min(max_cap, N_filtered − window_size)` — widening impossible by construction; new §4.8 legacy-engine closure; AC1 strengthened via synthetic nonzero phase on a supported ABI. §8 records that nothing remains open. | **DESIGN GATE CLOSED** per the v1.0 ruling's disposition. Next artifact: **Implementation Brief I** (not yet written). | **YES — the live design** |
| `CLAUDE_CODE_INSTRUCTIONS_FIELD6_OBSERVABILITY_REPAIR.md` (192L) | Sequencing item 3: make the two R-1 falsifier fields (`deferred_distinct_attempts_high_water`, `pump_liveness_probes_high_water`) actually observable on the `[S172-BP] summary` line. Observability-only, own gate, must land before the Phase-7 soak. Names the trap in advance: an `int()` cast over a `None` seed raises into the blanket `except` and leaves both fields `None` forever while looking like a working feature. | **RULED** → below. | superseded by the ruling |
| `CLAUDE_CODE_REPORT_FIELD6_OBSERVABILITY_REPAIR.md` (483L) | The implementation evidence: `G-FIELD6` (3 arms) + `G-MUT-FIELD6` (3 mutants, all DETECTED); AST proof the added-definition set is unchanged; §5 **discloses three certified-suite edits outside the brief's enumerated scope** and argues each. | **RULED** → ratified. | historical evidence |
| `TB_RULING_FIELD6_IMPLEMENTATION.md` (147L) | ★ **ACCEPTED**, implemented `d8b21e3` (6 files, clean-tree battery 52/52). **FAIR-4/0 replacement RATIFIED — do not revert to `50/50`**: an exact count was never a sound authorization proof (an unauthorized gate swap still reads 50/50; two legitimate additions falsely red at 52/52). The three out-of-brief suite edits **RATIFIED**. Gate 22's red was the expected **transitional** dirty-tree condition — **do not widen the allowlist.** Carries **SR-1** (§1.0). | **RULED**, 2026-08-20. | **YES** |
| `TB_RULING_REQUEST_CHANGELOG_NUMBERING.md` (34L) | Which naming convention governs, given `S185` is only the highest *visible* number while ~20 SER8-only changelogs await backfill? Option A (date+topic now, reconcile once at import) vs Option B (resume S-numbering now). | **RULED** → Option A. | superseded by the ruling |
| `TB_RULING_CHANGELOG_NUMBERING.md` (53L) | **Option A APPROVED and governs immediately.** Carries **SR-2** (§1.0). | **RULED**, 2026-08-18. | **YES** |
| `TB_RULING_20260815_H1H2_R7_CERTIFICATION.md` (176L) | ★ **R1→R7 H1/H2 instrumentation CERTIFIED.** Beta independently reproduced the exact artifact (bytes, sha256, AST-call and call-site counts) and **executed the counterexamples itself**: the nested-`def` shape that defeated R6 now reds; six indirect-reach shapes of `fail_trial` all red while `getattr(msg, 'stripe_id', None)` stays clean. The R1→R7 technical review is closed. | **RULED**, 2026-08-15. Pairs with `SESSION_CHANGELOG_20260815_S185.md` and spec `ATTEMPT6_RIG_LOG_FORENSIC_v1_0.md` §7. | historical — the instrument that decided H1/H2 |

## 1.2 Team Beta rulings and ruling requests (`TB_*`) — earlier eras

| file | what it rules on / asks | disposition | still binding? |
|---|---|---|---|
| `TB_BINDING_RULINGS_S172_PHASE4.md` | Beta's binding answers to the two Phase-4 requests: **(1)** the worker must reject a stripe assignment that **omits** `dataset_sha256` as well as one that mismatches — Option C, explicitly **overriding** Alpha's recommended compare-when-present, because compare-when-present lets a coordinator regression silently bypass identity; **(2)** the L7 abort-discard interface. | **RULED**, 2026-07-18. Implemented via `CLAUDE_CODE_INSTRUCTIONS_S172_PHASE4.md` Stage 0 + Stage 4. | **YES** |
| `TB_RULING_REQUEST_BLOCKER6_DATASET_SHA_S172_PHASE4.md` | What does the **worker** do when a payload arrives with **no** `dataset_sha256` key at all — tolerate absence, or fail closed? | **RULED** → Option C, in `TB_BINDING_RULINGS_S172_PHASE4.md` Ruling 1. | superseded by the ruling |
| `TB_RULING_REQUEST_L7_ABORT_DISCARD_S172_PHASE4.md` | Sync `abort_trial()` whose return guarantees Phase 5 holds no trial-owned path, vs async `TrialAbort → TrialAbortAck`. A selection, not a scope change. | **RULED** in `TB_BINDING_RULINGS_S172_PHASE4.md`. | superseded by the ruling |
| `TB_RULING_REQUEST_CPU_PARALLELISM_S175.md` | Should host-side survivor collection + 22-array NPZ assembly be parallelised across the coordinator CPU, or inherit the single-threaded GIL-bound S152 pattern? I.e. does the miner *remove* PWC's high-survivor throughput collapse or merely *relocate* it from transport to assembly? | **RULED** — the binding S175 ruling is the stated change driver for `PROPOSAL_S172_RANGE_MINER_v1_4_5.md`. Realised as D5 `process_sharded`. | **YES** — but D5 is **available and UNPROMOTED** |
| `TB_RULING_REQUEST_D5_EXCEPTION_PRECEDENCE.md` | D5's read-all-then-merge is **not** the semantics-preserving no-op it was specified as: the pre-D5 `assemble_trial` interleaves read and merge, so a duplicate in an earlier-order spool raises **before** a later spool is read. | **RULED → Option B** (preserve deterministic precedence; spool-read errors returned as typed data, replayed by the parent in manifest order). Recorded in `CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D5_REV2_ADDENDUM.md`. | **YES** |
| `TB_RULING_REQUEST_IPC_SERIALIZATION_S150.md` | High-survivor trials collapse **27×** (≈1.99M s/s → ≈73K s/s) purely from result-payload serialisation. Should the IPC path change? | **RULED → Option A / `slim_v1`.** Design in `IPC_SERIALIZATION_FIX_IMPLEMENTATION_GUIDE_S150.md`; applied by root `apply_s150_slim_v1_ipc.py`. | superseded in practice by the RANGE-MINER pivot |
| `TB_RULING_REQUEST_ROCR_ISOLATION_S149.md` | AMD rigs at ~1,050 seeds/sec vs an ~787K/GPU baseline, **0% GPU utilisation with 4 live workers**. Is per-worker `ROCR_VISIBLE_DEVICES` isolation the fix? | **RULED** (approved — the file itself carries no ruling text). Evidence: `apply_s149a_rocr_isolation.py`, `verify_s149a_rocr_fix.py`, `SESSION_CHANGELOG_20260321_S149.md`. | historical — pool-size limits later re-derived |
| `TB_RULING_REQUEST_STEP2_v4_1_OBJECTIVE.md` | **The WSI v4.0 tautology.** WSI = 0.9997 on trial 1 because the formula contains `quality = fwd*rev` as its dominant term (w3 ≈ 0.82) — the objective measures itself. | **RULED** → v4.1 deployed cleanly. **SUPERSEDED BY `TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md`** in the same session. | superseded |
| `TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md` | ★ **The document Alpha nearly re-reported to Beta as a new finding on 2026-08-02.** v4.1 deployed and the objective still cannot optimise: `sel_score = 0.0000` on **every** passing trial because `bidirectional_selectivity` sits at floor (98.8%) with no variance. | **RULED** → v4.2; lineage to v4.3/v4.4 (`S107_session_log.md`). **Read this before reporting any Step-2 objective blindness.** | **YES** — and see `CLAUDE_CODE_REPORT_SELECTIVITY_PER_SEED_AUDIT.md` (§3.6), which re-derives *why*: the quantity is trial-level by construction |
| `TB_RULING_S176_WATCHER_KPI.md` (827L) | Beta's review of the S176 WATCHER retrain-KPI findings, independently checked against live public `main` at `0c3166a`. | **RULED.** Follow-up: `CLAUDE_CODE_BRIEF_S176_FOLLOWUP_v1.md`. | **YES** |
| `TB_RULING_S177_KPI_GOVERNANCE.md` (661L) | Beta's review of KPI governance proposal v1.0 + analyzer v2. | **RULED: CONDITIONAL APPROVAL — REVISION REQUIRED.** Eight proposal blockers + six analyzer fixes. | superseded by S178 |
| `TB_RULING_S178_KPI_GOVERNANCE.md` (591L) | Beta's review of proposal **v1.1** + analyzer **v2.1**. | **RULED: ARCHITECTURE APPROVED IN PRINCIPLE — FOUR MANDATORY AMENDMENTS.** | superseded by S179 |
| `TB_RULING_S179_IMPLEMENTATION_AUTH.md` (552L) | Beta's binding review of the **v1.2 addendum** + analyzer **v2.2**. | **RULED: APPROVED FOR IMPLEMENTATION WITH THREE BINDING CODE-LEVEL CONDITIONS.** Whether implementation has landed was **not established in this pass either** — see §7 gap 8. | **YES** — the live authority for KPI governance |
| `TB_SUBMISSION_S159G_RIG6600_CRASHES.md` | P0: rrig6600 crashes consistently under ZMQ multi-rig runs. Netconsole capture of the `GCVM_L2_PROTECTION_FAULT` escalation. Positive finding: ZMQ SQLite lease expiry recovered the run. | **RECORD** + P0 ruling requested. | historical (pre-Proxmox, pre-miner) |
| `TB_UPDATE_S159G_ENV_PROPAGATION.md` | Does `HSA_ENABLE_SDMA=0` actually reach live worker PIDs? It does not: rrig6600's workers carry **one** ROCm variable; the stable rigs carry four. | **RECORD** — root cause is environment propagation, not kernel logic. | historical |
| `TB_FINAL_UPDATE_S159G_ROOT_CAUSE_CONFIRMED.md` | Confirms the above at PID level across all three rigs. | **RECORD** | historical |
| `TB_UPDATE_S162_OPTION_B_RESULTS.md` | Does `AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3` on rrig6600c delay the crash? **No.** | **RULED** on the diagnostic; feeds `PROPOSAL_S162_RRIG6600C_CRASH_ROOT_CAUSE_v1_0.md`. | historical |
| `TB_Incident_Report_rrig6600c_S163KARG.md` (305L) | Forensic: the rrig6600c fatal crash is a **GPU virtual-memory PTE invalidation while kernels are mid-execution across devices** — **distinct** from the int32/int64 kernel-arg mismatch the KARG patch fixed. | **RECORD**, BLOCKING at the time. | historical — the failure class that motivated RANGE-MINER |
| `TB_SUMMARY_S163.md` | Three items: the NPZ `UnboundLocalError` (a duplicate `import numpy` shadowing the module-level name), `free_all_blocks()` removal, staged 500K→2M validation. | **RECORD** | historical |
| `TB_UPDATE_SELFPLAY_REFRAMING_2026-07-28.md` | **Correction of framing, not architecture.** Self-play is a *discovery front-end* to an already-built grade→attribute→concentrate→reinforce loop — not "the" learning system. | **RECORD** + three confirmations requested. | **YES** — governs how self-play may be described |

## 1.3 ★ THE GATE-12 CAMPAIGN — nine attempts, 2026-08-04 → 2026-08-17

**The dominant governance material of this catalog's delta, and the reason it exists.** Gate 12 is
the production-class Step-1 run that certifies the miner end to end on the real fleet. Attempts 1–8
all failed; **nothing composes from a failed attempt** and every attempt used a fresh nonce, all
prior nonces burned. **Attempt 9 passed** and is anchored at `e9ca800` / tag `gate12-passed-attempt9`.

**Read this section before proposing any coordinator change.** Nearly every defect below was found
by an instrument built *after* the previous attempt proved the prior instruments could not see it.

### The chain, in order

| round | ruling request / finding | brief | report / evidence | submission | outcome |
|---|---|---|---|---|---|
| **staging_dir** | `TEAM_ALPHA_STAGING_DIR_NOTE.md` — ★ **the finding that outranks the bug**: the miner-backed WATCHER path could not stage a single sub-stripe result, and **no gate had ever driven the production path.** RANGE-MINER also has **no worker-launch mechanism** (the soak ran against a hand-started 25-daemon fleet). | `..._STAGING_DIR_FIX.md` (Part A) → `..._STAGING_DIR_PART_B.md` (supersedes Part B of the first) · `..._MANUAL_FLEET_LAUNCH.md` | `S172_STAGING_PART_B_REPORT.md` (455L) — **COMPLETION SENTINEL: INCOMPLETE.** Repair implemented and gated 24/24 (each proven red pre-fix); **`G-PROD-SHAPE` NOT executed** — the pipeline launch is Michael-initiated by `CLAUDE.md` rule 3. | — | repair landed; `tests/gate_s172_prod_shape.py` created and **still never run** (§4.6) |
| **deferred queue** | `TEAM_ALPHA_DEFERRED_QUEUE_NOTE.md` — `staging_deferred_max = 64` was **Alpha's number, not Beta's**, and it terminated the first production-shape trial that ever got past staging. *"The sizing is the smaller question; the classification is the larger one."* | `..._S172_STAGING_BACKPRESSURE_REMEDIATION.md` (B+D+A+C; **E REJECTED** — no seed-cap or stripe-geometry change, for any reason) | `CLAUDE_CODE_REPORT_S172_STAGING_BACKPRESSURE.md` (494L) | `TB_SUBMISSION_S172_STAGING_BACKPRESSURE.md` + `TEAM_ALPHA_REVIEW_S172_STAGING_BACKPRESSURE.md` (independently executed on a **second host** from a fresh clone) | Beta ruling 2026-08-05 → committed `4b1aad6`. **Disclosed deviation:** committed at the owner's direction ahead of Beta's review, recorded in the commit message |
| **F1–F5** | Beta HOLD, *"targeted fix-forward required"*, 2026-08-06 | `..._S172_BP_AMENDMENT.md` → `_R2.md` → `_R3.md` | `CLAUDE_CODE_REPORT_S172_BP_AMENDMENT{,_R2,_R3}.md` | `TB_SUBMISSION_S172_BP_AMENDMENT_DELTA.md` → `_BP_R2_DELTA.md` → `_BP_R3_DELTA.md` | three HOLD rounds; ends with **exact-envelope credit tokenization** (`credit_id` minted under `_pause_lock` before `event.set()`) and a **pre-decode barrier** |
| **dispatch stall** | read-only diagnosis, 2026-08-07 | `..._S172_GATE12_DISPATCH_STALL.md` | `CLAUDE_CODE_REPORT_S172_GATE12_DISPATCH_STALL.md` (413L) — every command a read (SQLite `mode=ro`, read-only SSH probes) | — | fed the two 2026-08-07 rulings below |
| **staging capacity** | Beta *"S172 GATE-12 STAGING-CAPACITY DEADLOCK"* (2026-08-07) + *"STEP-1 SEARCH GEOMETRY"* R4 (`elapsed_s`) | `..._STAGING_CAPACITY_AMENDMENT.md` → `..._S172_STAGING_CAPACITY_R1.md` → `_R2.md` | `CLAUDE_CODE_REPORT_S172_STAGING_CAPACITY_{AMENDMENT,R1,R2}.md` | `TB_SUBMISSION_S172_STAGING_CAPACITY_{AMENDMENT,R1,R2}.md` | 42/42 → 48/48 → 50/50; **committed `4dd5535`** |
| **seed domain** | Beta *"S145 / SEED-DOMAIN SWEEP TERMINUS AND COVERAGE AUTHORITY"* (2026-08-07). Beta was explicit these are **separate amendments** — *"different authorities, different test suites, different rollback surfaces"* | `..._SEED_DOMAIN_CURSOR_AMENDMENT.md` → `_R1.md` → `_R2.md` (**TEST-ONLY**) | `CLAUDE_CODE_REPORT_SEED_DOMAIN_CURSOR_{AMENDMENT,R1,R2}.md` | `TB_SUBMISSION_SEED_DOMAIN_CURSOR_{AMENDMENT,R1,R2}.md` | 29/29 → 39/39 → 40/40; **committed `a3bb4da`**; produced **`utils/seed_coverage_ledger.py`** — `[0, 2^32)` shared terminus, legacy-tracker deauthorization, first-gap cursor, append-only content-derived ledger |
| **attempt 1** | `TB_NOTE_AMENDMENTS_COMMITTED_GATE12_REQUEST.md` — both amendments committed, execution authorization requested | — | — | — | authorized |
| **attempt 1 FAIL** | `TB_GATE12_EVIDENCE_PACKAGE_FAIL.md` — Alpha's own determination: **GATE 12 FAILED**, two findings, one of them an Alpha run-shape error. **No 25-GPU saturation claim made.** | `..._GATE12_FAILURE_FORENSICS.md` (read-only) | `CLAUDE_CODE_REPORT_GATE12_FAILURE_FORENSICS.md` (726L) | `TB_SUBMISSION_GATE12_FORENSICS_F1_DEFECT.md` — ★ **PRODUCTION DEFECT FOUND.** Alpha requests **no rerun**, and specifically **not** the `worker_pool_size = 25` correction, because it would **mask** the defect | committed `eecfff7` |
| **F1/F2** | Beta *"GATE-12 F1 FORENSICS / LEASE AMENDMENT"* (2026-08-09) — **Beta chose the remedy; Alpha did not substitute one** | `..._F1_F2_ACTIVE_LEASE.md` → `_R1.md` → `_R2.md` | `CLAUDE_CODE_REPORT_F1_F2_{ACTIVE_LEASE,R1,R2}.md` | `TB_SUBMISSION_F1_F2_{ACTIVE_LEASE,R1,R2}.md` · `TB_NOTE_F1_F2_SUBMITTED.md` · `TB_NOTE_R1_INFLIGHT_AND_ACCESS_PATTERN.md` | **Beta's §0 prediction held: the dispatcher needed no change at all.** One-active enforced **in SQL** inside `claim_stripe`; F2's terminal record written by the state transition itself. Certified, **committed `c4e0037`** |
| **pre-rerun** | two Beta-required items before any rerun request: a truthful GPU probe and a post-F1 concurrency sampler | `..._PRERUN_PROBE_AND_SAMPLER.md` → `_R1.md` → `_R2.md` → `_R3.md` | `CLAUDE_CODE_REPORT_PRERUN_{PROBE_AND_SAMPLER,R1,R2,R3}.md` | `TB_SUBMISSION_PRERUN_{ITEMS,R1,R2,R3}_AND_RERUN_REQUEST.md` | GPU probe certified at R1 and frozen; sampler returned three times. **`49ff9b4`** (R2) then **`4643a11`** (R3). ★ **A hash correction worth remembering:** Alpha cited the F1/F2 commit as `d3f8f00`, a hash that does not exist — misread from a terminal image; caught by *attempting the lookup* rather than trusting it |
| **attempt 2 FAIL** | `TB_GATE12_ATTEMPT2_FORENSIC_REPORT.md` — ★ *"Not hardware, not the certified F1/F2 scheduler, not staging."* The per-stage admission gate requires `expected_workers` **live registered** connections at the start of **every** stage; by stage 4 two of the 25 non-persistent workers' TCP connections had left the registered set, 23 were eligible, the 180 s window elapsed, the trial aborted. **The whole fail-closed chain below it behaved exactly as designed.** | Defect A: `..._DEFECT_A_TRANSPORT_RECOVERY.md` → `..._DEFECT_A_R2_DEADLINE.md` · Defect B: `..._DEFECT_B_TURNOVER_AGGREGATION.md` | `CLAUDE_CODE_REPORT_DEFECT_A_{TRANSPORT_RECOVERY,R2_DEADLINE}.md` · `..._DEFECT_B_TURNOVER_AGGREGATION.md` | `TB_SUBMISSION_DEFECT_A_{TRANSPORT_RECOVERY,R2_DEADLINE}.md` · `TB_SUBMISSION_DEFECT_B_{TURNOVER_AGGREGATION,CLOSURE_EVIDENCE}.md` | Defect A certified-except-§14 at `acd6f13`, then §14 deadline enforcement; Defect B closed on an **evidence-completeness hold, not a revision order** |
| **attempt 3 FAIL** | refused **at publication**: `RunParameterError: repository_tree_clean is False`. Two defects, both in the launch harness, neither in D3.5 — `gate12_launch.sh:54` **printed** the tree state and never **tested** it | — | `SESSION_CHANGELOG_20260811_CLEANTREE_ADMISSION.md` (236L) | Beta RETURN FOR NARROW R1 (two suite-provenance defects, both in the test file); 25/25 → 31/31 | produced `scripts/gate12_cleantree_gate.py` |
| **attempt 4 FAIL** | F1 lease-origin: `serve_trial` captured `now` once per iteration and stamped every `claim_stripe` of the pass with `now + compute_lease_timeout` — two stripes carrying `lease_expires_at` **identical to the microsecond** | — | `SESSION_CHANGELOG_20260811_F1_LEASE_ORIGIN.md` (241L) | Beta RRR: **the repair itself was ACCEPTED and not altered** — all four defects were in the new `ServeLoopTiming` instrumentation. 13/13 → 18/18 | — |
| **attempts 4+5** | two facts unrecoverable from their own artifacts: **every reader exit was indistinguishable at the point of drop** (nine ways out of `_conn_reader_loop`, none recording anything), and the serve loop's inbound drain had **no time term** — one attempt-5 iteration spent **940.971 s** inside the drain while lease expiry, admission, dispatch and stage advance did not run | — | `SESSION_CHANGELOG_20260813_S180.md` · `..._20260814_S181.md` (R1 **and** R2 in one file, deliberately) | — | attempt-6 R3 remediation implemented. **It does not diagnose attempt 5** — the initiating cause of the two lost reader sessions remains **UNRESOLVED**; Part A only makes the *next* occurrence self-describing |
| **D6 dry runs** | dry run #1: the rigs carried a `range_miner_worker.py` last deployed 2026-08-02 — **24 of 25 workers died at argparse**. Dry run #2: `mkdir && cd && worker … & echo started`, where `&` binds to the whole `&&` list, so the forked remote subshell held the SSH channel | — | `SESSION_CHANGELOG_20260814_S182.md` (code-parity gate + launcher wait set) · `..._20260815_S183.md` (D6-I1 remote dispatch detachment + D6-I2 sentinel-correlated liveness) · `..._20260815_S184.md` (**dry run #3 PASS**) | — | produced `scripts/gate12_parity_gate.py` (806L; **acceptance authority is content identity, never Git identity**), `scripts/gate12_worker_liveness_gate.py`, `scripts/launch_fleet_manual.sh` |
| **attempt 6** | `ATTEMPT6_RIG_LOG_FORENSIC_v1_0.md` (420L) — **BOUNDED-UNRESOLVED, no remedy proposed.** Can the rig-side logs distinguish *"the three long-held stripes were still computing"* from *"their results were sent and not processed"*? §7 (A–F) became the H1/H2 instrumentation spec | — | `SESSION_CHANGELOG_20260815_S185.md` (438L) | — | `TB_RULING_20260815_H1H2_R7_CERTIFICATION.md` (§1.1) |
| **attempt 7 FAIL** | `GATE12_ATTEMPT7_H1H2_FORENSIC.md` (249L) — ★ **H1 REFUTED · H2 CONFIRMED**, by direct measurement, on the first run the instrumentation was live for. Failed at stage 2 on a compute-lease expiry. Read-only; **no remedy proposed — the choice of mechanism is Beta's** (§2.26 precedent) | `~/dashboard_work/CCODE_BRIEF_MP1_DRAIN_ATTRIBUTION_v1_0.md` (out of tree) | `SESSION_CHANGELOG_20260816_MP1_DRAIN_ATTRIBUTION.md` (220L) — three-level per-iteration attribution, every level summing to a **named** remainder; `PhaseCharge` thread-keyed inclusive/exclusive time; per-connection drain census built from the **live connection set** so *"never serviced"* is a measured 0 row, not an absent one | — | **MEASUREMENT BEFORE REMEDY** (Beta) |
| **MP-1 run FAIL** | `GATE12_MP1_RUN_FORENSIC.md` (297L) — same wall as attempt 7, **but the cause is now named and measured**: `_pump_deferred` evaluated `_attempt_live_locked` **once per deferred entry**, under `_admission_lock`, on **every** staging-job completion, and `MinerLedger._conn` opens a new sqlite3 connection + 3 PRAGMAs per query. **Liveness is a per-attempt property being paid for per-frame** | R-1…R-4 briefs (out of tree) | `SESSION_CHANGELOG_20260817_R1_R4_DRAIN_REMEDY.md` (276L) — **3,401 → 233 ledger reads · 1,975 ms → 136 ms · 14.6×**, with both correctness blockers Beta raised against the *faster* intermediate forms (R-1 staged on cached positives; R-2 refused a frame at the capacity boundary) **closed rather than accepted** | — | **Beta CERTIFIED R-1 … R-4** → `e9ca800` |
| **attempt 9 PASS** | — | — | `SESSION_CHANGELOG_20260817_GATE12_ATTEMPT9.md` (344L) — 128/128 stripes, 4 stages, serve loop 738.162 s, **zero lease expiries, zero ERROR/Traceback**, coverage `[0, 2,147,483,648)`, `coverage_id c6f28aedf7af12cd`. §2 is a method note worth keeping: **the oracle had to be calibrated before it could be trusted** | — | `TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md` (§1.1) |
| **governance** | — | — | `SESSION_CHANGELOG_20260817_GOVERNANCE_RULINGS.md` (137L) · `SESSION_CHANGELOG_20260820_FIELD6_AND_WINDOW_ANCHOR_DESIGN.md` (145L) | — | the two rulings recorded, sequencing items 1 and 3 executed, the window-anchor absence finding made |

**Mandated phrasing, still in force.** Until the falsifiers are first observed in production
(**which is the Phase-7 soak**), the complexity result must be stated as:

> R-3's scaling model is gate- and benchmark-certified and strongly corroborated by Attempt 9's
> per-call cost, but its two dedicated production falsifier fields were not persisted and therefore
> were not observed in Attempt 9.

The field-6 repair (`d8b21e3`) makes them **observable**; it does **not** claim they were observed.

## 1.4 Team Alpha submissions (`TEAM_ALPHA_*`) — earlier threads

Grouped by the thread each belongs to; within a thread, in sequence.

### The S172 Phase-4 coordinator review chain — seven rounds

Beta rejected Phase 4 six times. **Read this chain before assuming an adversarial gate is
excessive** — three of the six defect classes were in functions Alpha had already read line-by-line
and passed.

| file | what it records | pairs with | disposition |
|---|---|---|---|
| `TEAM_ALPHA_REVIEW_S172_PHASE4.md` | Rev-1: 63-line stub → 2,141-line coordinator; 36/36 brief gates. | `CLAUDE_CODE_INSTRUCTIONS_S172_PHASE4.md` | SUPERSEDED BY REV2 |
| `..._REV2.md` | Beta rejected on **one** blocker: `run_trial_miner()` had **no real server** — gate 20 exercised a harness-built loop, not the coordinator's own serve path. | `CLAUDE_CODE_CORRECTION_S172_PHASE4_SERVE.md` | SUPERSEDED BY REV3 |
| `..._REV3.md` | Six release-blocking serve-path/ledger/wiring defects. **Accountability note:** rev-1/rev-2 traced the intended path and pattern-matched instead of constructing the adversarial case. | `CLAUDE_CODE_CORRECTION2_S172_PHASE4_SIX_DEFECTS.md` | SUPERSEDED BY REV4 |
| `..._REV4.md` | Six async-staging / socket defects: orphan late-write, unbounded-queue RAM, capacity deadlock, reconcile→matrix routing, hybrid hash-mismatch retry, silent/partial socket. 54/54. | `CLAUDE_CODE_CORRECTION3_S172_PHASE4_ASYNC_SOCKET.md` | SUPERSEDED BY REV5 |
| `..._REV5.md` | Four overload / heterogeneous-worker freezes Beta **reproduced**. 59/59. | `CLAUDE_CODE_CORRECTION4_S172_PHASE4_OVERLOAD.md` | SUPERSEDED BY REV6 |
| `..._REV6.md` | ★ **The theme worth carrying forward:** two of three prior failures were gates checking **bookkeeping** (registry entry count, a static byte estimate) instead of the **real resource** — so the gate passed while the resource leaked. Fixes now assert bytes on disk and live `threading.enumerate()`. | `CLAUDE_CODE_CORRECTION5_S172_PHASE4_REALBOUNDS.md` | SUPERSEDED BY REV7 |
| `..._REV7.md` | The last blocker: admission used **two contradictory byte models**. Fixed by Beta's Approach A — `_try_admit_locked` becomes a pure serialisation gate. | `CLAUDE_CODE_CORRECTION6_S172_PHASE4_ADMISSION.md` | **RULED — Phase 4 closed at 63/63** |

### The S172 Phase-5 deliverable chain (D0 → D6.2)

| file | what it settles | disposition |
|---|---|---|
| `TEAM_ALPHA_REVIEW_S172_PHASE5_D0_REV2.md` | `INSERT OR IGNORE` silently accepted a **conflicting** trial context → compare-and-insert in one DB transaction under the write lock. D0 9/9. | SUPERSEDED BY REV4 |
| `TEAM_ALPHA_REVIEW_S172_PHASE5_D0_REV4.md` | `window_size`/`offset` **fabrication** in `run_trial_miner` removed, plus a third-order coercion site and Beta's vacuity catch on gate B4. D0 12/12, verified **from the extracted archive**, not the working tree. | **RULED — D0 closed** |
| `TEAM_ALPHA_REVIEW_S172_PHASE5_D1_0.md` | Workflow bidirectionality + abort/commit terminal-race correction. AST-verified scope. | RULED |
| `TEAM_ALPHA_REVIEW_S172_PHASE5_D1_1.md` | Shared four-population assembly engine + concrete `Phase5Sink`. Requests the gate-22 rule be extended from "the deliverable's new *harness* path" to "new file paths". | RULED |
| `TEAM_ALPHA_REVIEW_S172_PHASE5_D2.md` | Directional uniqueness enforced at **both** layers. No production change. | RULED |
| `TEAM_ALPHA_REVIEW_S172_PHASE5_D3.md` | The shared backend-neutral **24→22 columnizer** + independent structural validator. | RULED |
| `TEAM_ALPHA_REVIEW_S172_PHASE5_D3_0.md` | Canonical encoding seam + rectangular empty NPZ. **The untouched-accumulator claim holds structurally, not by assertion.** | RULED — but see `TEAM_ALPHA_D3_0_B_AND_ITEM1_NOTICE.md` |
| `TEAM_ALPHA_REVIEW_S172_PHASE5_D3_25.md` | Mode-preserving backend result contract + canonical candidate-ingress normalisation. **Ends the "PWC/ZMQ untouched" era.** | RULED |
| `TEAM_ALPHA_REVIEW_S172_PHASE5_D3_5.md` | The shared run finalizer. Byte-identity of the three components the retired migration gates certified, proven twice over. | RULED |
| `TEAM_ALPHA_REVIEW_S172_PHASE5_D3_5_B.md` | Seed-Domain v1.1. **[R1]** Alpha had asserted the recursive chain walk compares `prng_base`/`schema_version`/`canonical_map_hash` **without verifying**; it contained zero such occurrences. | RULED |
| `TEAM_ALPHA_REVIEW_S172_PHASE5_D4.md` | `serial_reference` behind the frozen two-backend interface. Every must-not-modify file SHA-verified. | RULED |
| `TEAM_ALPHA_D5_ADVANCE_TO_BETA_REV3.md` | **Lossless dual-encoding** for seed projection — fast `int64` path iff every seed satisfies −2⁶³ ≤ s ≤ 2⁶³−1, else the whole spool falls back to `signed_bytes`. | **RULED — D5 closed** |
| `TEAM_ALPHA_D6_ADVANCE_TO_BETA.md` | D6: production integration adapter, real miner trial → finalizer → certified generation → Step-1 accumulator, with an unchanged `TestResult` shape. | RULED (with correction) |
| `TEAM_ALPHA_D6_CORRECTION_RETURN_TO_BETA.md` | Held on one blocker: **the miner ignored configured thresholds and filtered at a hardcoded `0.25`**, so the optimizer certified results for a configuration it had not requested. Alpha states plainly it **under-disclosed the writer seam** the prior round. | **RULED** → `2be51d5` |
| `TEAM_ALPHA_D6_1_RETURN_TO_BETA.md` | D6.1 + a scope-change ruling request: the briefed in-place flush repair targeted paths that since D3.5 are **finalizer-owned compatibility symlinks**. D1 reproduced: numpy 1.22.0 wrote `...flush.tmp.npz`, `os.replace` raised `FileNotFoundError`. **Incremental durability had never existed.** | **RULED** — relocate to `.s172_checkpoint/<run_id>/` |
| `TEAM_ALPHA_D6_2_IDENTITY_ADDENDUM.md` | Raised **mid-review**: the checkpoint directory and the published generation use **two different run identities**, defined ~2,150 lines apart in the same file. | **RULED** |
| `TEAM_ALPHA_D6_2_CERTIFICATION_SUBMISSION.md` | D6.2 at `f7583bc`: 24-field canonical flush, finalizer fed reconstructed cumulative state. **The S166 OOM protection is real for the first time.** | RETURNED by Beta |
| `TEAM_ALPHA_D6_2_RECERTIFICATION_SUBMISSION.md` | Bounded repair at `18a2419`. ★ **Method note:** Beta found blocker 1 by reading the live objective; Alpha had read *the report's description of it*. This submission was reviewed **from the diff, not the report**. | **RULED — D6.2 CERTIFIED** at `18a2419` |

### Phase 6 — dataset authority, ROCm parity, certification

| file | what it settles | disposition |
|---|---|---|
| `TEAM_ALPHA_PWC_COMPARATOR_SCOPE_CORRECTION.md` | ★ **Scope drift, named and corrected.** The RANGE-MINER rule was **interface compatibility** — a statement about the 22-array shape and its consumers, **not about values**. It had silently become "prove RANGE-MINER produces output identical to PWC". | **RULED** — PWC **retired from certifying authority**; flag-selectable, non-certifying diagnostic |
| `TEAM_ALPHA_PROPOSAL_PHASE_6_0_ROCM_SMOKE.md` | A cheap single-rig ROCm smoke *before* full Phase 6, because everything certified through D6 ran on **one RTX 3080 Ti under CUDA**. | **RULED — approved**, with a required parity addition |
| `TEAM_ALPHA_PHASE_6_0_RETURN_TO_BETA.md` | ★ **The headline result.** The miner production path ran on an RX 6600 XT under ROCm and produced a **byte-identical certified artifact** to the CUDA run — `artifact_sha256 0e0092fe…c4b0` identical across three runs; 22/22 arrays equal; forward 398,156 / reverse 383 / bidirectional 319. | **RULED** |
| `TEAM_ALPHA_DATASET_LIFECYCLE_FINDINGS.md` | **A fixed `expected_sha256` cannot work** — the dataset is not immutable. The invariant is **fleet consistency**, not immutability. | **RULED** — five rulings issued |
| `TEAM_ALPHA_APPEND_ONLY_SIMPLIFICATION.md` | **Rewrite mode will not exist.** Publication becomes **dated immutable files**. Records `daily3scraper.service` now `disable --now`, unit retained. | **RULED** |
| `TEAM_ALPHA_PUSHBACK_ORDERING_AND_THRESHOLD_REGRESSION.md` | ★ **Alpha contesting a Beta ruling with evidence, and winning.** Ruling 20's premise is factually wrong: the CA draw procedures select equipment **per session**, so midday and evening are different PRNG streams. | **RULED** — Ruling 20 withdrawn; combined-session sequential sieve now non-certifying |
| `TEAM_ALPHA_PHASE_6_P0_SUBMISSION.md` | Dataset version one published in place — immutable `daily3-<UTC>Z-<sha256[:12]>.json` + atomic pointer manifest. Discloses that **Alpha proceeded without prior plan approval**. | **RULED** → `131787d`. Beta granted the procedural exception and stated it is **not precedent** |
| `TEAM_ALPHA_PHASE_6_P0_5_SUBMISSION.md` | The behavioural cutover: pointer resolution, one-time run-start freeze, absolute-path dispatch, fail-before-first-worker-dispatch, per-node provisioning with **on-target** verification. | **RULED** → `d4ff1e4`; closure condition → `8600e75` |
| `TEAM_ALPHA_FLEET_STATE_SUBMISSION.md` | The fleet-state investigation **and a finding that outranks it**: `assign_stripes`, `_dispatch_pending`, `process_lease_expiry` **and** the stage advance are all behind one `len(eligible) >= expected_workers` guard while `serve_timeout` defaults to `None`. A worker loss crossing the threshold means the trial **neither completes nor fails**. | **RULED** → repaired `ee0db06`. **This is the same guard the attempt-2 forensic later re-encountered per-stage** (§1.3) |
| `FLEET_SUBMISSION_CORRECTIONS.md` | Two **wording** corrections Beta required. Both corrections, not reversals — Beta confirmed the findings. | RULED |
| `TEAM_ALPHA_EXECUTION_SET_AND_CHAPTER2_SUBMISSION.md` | The **Resolved Execution Set** — one frozen run-scoped fleet authority created before dataset verification, GPU verification, coordinator construction and dispatch. | **RULED** → `63e627f`; Beta **withheld** Phase-7 closure pending two repairs |
| `TEAM_ALPHA_ADMISSION_BINDING_SUBMISSION.md` | ★ **A retraction in place.** Alpha claimed the freeze-after-read ordering could not be violated; **Beta's refutation was correct.** Counter now unconditional. | **RULED** → `eff6616`; **Phase-7 closure granted** |
| `TEAM_ALPHA_BOUNDED_PHASE_6_SUBMISSION.md` (731L) | **The certification the whole PWC → RANGE-MINER pivot was built toward.** Wall A PASS, Wall B PASS, Known-Answer Transfer Gate PASS, RandomSampler control arm PASS **(NON-CERTIFYING)**, 22/22 suites exit 0. | **RULED — CERTIFIED and CLOSED** → `d98298c` |
| `TEAM_ALPHA_WALL_C_SUBMISSION.md` | ★ *"Alpha scoped Wall C as new work. That was wrong."* Known-answer validation is documented **pre-repository practice** — the repository's history begins **after** the work was finished and cannot evidence it either way. | **RULED — Wall C struck** as a Phase-6 precondition |
| `TEAM_ALPHA_D3_0_B_AND_ITEM1_NOTICE.md` | **D3.0-B was never completed and Phase 6 certified anyway.** The defect is live at HEAD — `convert_survivors_to_binary.py:184` still silently defaults a record carrying **neither** `prng_type` **nor** `prng_base` to `'java_lcg'`. | **RULED — D3.0-B accepted as OPEN**; Beta disclosed the governance error unprompted. Tracked in `BACKLOG.md` §12 |
| `TEAM_ALPHA_PHASE7_LAUNCH_NOTICE.md` | **Not a ruling request** — Michael, as owner, ordered the Phase-7 soak to launch. **Alpha will not invoke the legacy converter until D3.0-B closes.** | RECORD — owner decision. **The soak it announces is the 2026-08-04 one that hit the `staging_dir` defect** (§1.3) |
| `TEAM_ALPHA_6P2_TRANSITION_RULING_REQUEST.md` | `daily3.json` ends **mid-day** (18,068 records, terminal `2026-02-26 midday`), so the next scrape's `2026-02-26 evening` sorts *before* the terminal record — **publication one is a backfill, not an append**, and REV3 §2.3 halts it with `NON_APPEND_INSERTION_REQUIRED`. | **AWAITING RULING** — blocks 6-P2 publication one |
| `TEAM_ALPHA_SCRAPER_RECENT_SAFETY_NOTICE.md` | ★ **Operational hazard.** The scraper's `--recent` path **destroys the dataset**: `main()` writes only what it just scraped over `daily3.json` and never reads the existing file. | RECORD — safety notice. **`daily3_scraper.py` is now tracked** (`334dacf`), which changes the audit surface it describes |

### Audits, corrections and design submissions

| file | what it settles | disposition |
|---|---|---|
| `TEAM_ALPHA_CHAPTER_1_AUDIT_SUBMISSION.md` | **Sentinel FAIL — on the chapter, not the audit.** 41 claims: **9 accurate · 19 stale · 5 superseded · 7 contradicted-by-code · 1 unverifiable.** A fifth dead dimension, and the first **operator-facing** one: `--forward-threshold`/`--reverse-threshold` declared and never referenced after `parse_args()`. | **RULED** → P0 tranche `ddd2ac8`; closed `ef4b1c6` |
| `TEAM_ALPHA_CHAPTER_2_RECOVERY_SUBMISSION.md` | ★ **A Beta ruling made on incomplete information, corrected with forensics.** Chapter 2 was not missing: 743 lines exist at `d14dcdd`, destroyed by a stale-copy overwrite at `248e48c`. | **RULED — re-scoped to restore-and-audit** |
| `TEAM_ALPHA_SKIP_SEMANTICS_SUBMISSION.md` | ★ **The premise of the earlier audit was false.** `HYBRID_SKIP_BOUND_AUDIT.md:318` recorded hybrid skip semantics as "unspecified"; they are specified in two committed documents. The audit's own VIR-6 declared a full-tree grep for `skip_min` that **reached the exact line and did not read it.** | **RULED** — skip-OUTPUT work approved; input bounds remain open |
| `TEAM_ALPHA_AUTONOMY_CONTROL_SURFACE_SUBMISSION.md` | Four autonomy chains. Opens with **a correction to Alpha's own reporting**: the LLM parameter-application seam exists at `agents/watcher_agent.py:1789-1793`. **Two of the four findings may not be defects at all.** | **RULED** — Chain C hotfix `f8b751c`; Chain D `pending_approval` upheld as a valid authority boundary |
| `TEAM_ALPHA_TRSE_FIX_PROPOSAL.md` | **TRSE's mathematics is sound.** Rules B and C are **ADVISORY-BY-DESIGN, not dropped wires** — three citations, including `SESSION_CHANGELOG_20260307_S122.md:56`. | **AWAITING RULING** on the proposed fix |
| `TEAM_BETA_SUMMARY_20260107.md` | Beta's A/B test of four LLM options → **DeepSeek-R1-14B primary + Claude backup**. | historical — see §3.10 |
| `TEAM_BETA_REVIEW_kfolds_S100.docx` | Beta's k-folds review, S100. **Binary `.docx` — not read in this pass; contents UNVERIFIED.** | UNVERIFIED (§7 gap 7) |

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
| 1 — Window Optimizer (Step 1) | `CHAPTER_1_WINDOW_OPTIMIZER.md` | 2,303 | ✅ **YES** — `CHAPTER_1_AUDIT_v1.md`, **9 of 41 claims accurate** | **CLOSED at `ef4b1c6`, 2026-08-02** — "verified-and-bounded, not finished". §17 carries the closure statement. Remediation: P0 `ddd2ac8`, then P1/P2, then closure. **`81ef3f1` only commissioned the closure brief and edits neither chapter file** — do not cite it as the closure. | Chapter revision "3.1" is **a documentation-only number with no source counterpart** — the module docstring says `Version: 2.0`. Do not read it as a code version. |
| 2 — Bidirectional Sieve (Step 2) | `CHAPTER_2_BIDIRECTIONAL_SIEVE.md` | **1,463** | ✅ **YES** | **CLOSED at `ef4b1c6`, 2026-08-02.** Destroyed at `248e48c` (−709), restored from `d14dcdd` (743L) at `e1225a7` → 1,089L, corrected `e50e35f`, closed `ef4b1c6`, content gate `09bbfbf` (6 gates + 6 mutants, 12/12, all proven red against the actual 34-line fragment). | ★ **§6 contains the three-lane CRT proof** — the thing Alpha claimed was undocumented. ★ **`:1133` carries F-4** — the fused `offset` defect, recorded **CONFIRMED, not repaired**. It is the defect `PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md` now designs the repair for (§1.1). |
| 3 — Scorer Meta-Optimizer (**Step 2.5 / WATCHER step 2**) | `CHAPTER_3_SCORER_META_OPTIMIZER.md` | 958 | ✅ **YES** — `CHAPTER_3_ALIGNMENT_AUDIT.md`, **55 claims: 17 accurate / 9 stale / 24 false / 5 unverifiable** | **NOT corrected.** The audit was read-only and **no fix was authorised.** v4.2, last touched `05b0e6b`. | **§8, §9 and §14.2 describe GPU scoring deleted at v4.0.** §9 confines the soak to `--start-step 1 --end-step 1`. |
| 4 — Full Scoring (Step 3) | `CHAPTER_4_FULL_SCORING.md` | 1,037 | ❌ **NO** | v2.0.0 (Holdout Integration); claims "~550 lines" across two files | unknown — unaudited. See `BACKLOG.md` §15: Step 3's output validation floor is three contracts stale |
| 5 — Adaptive Meta-Optimizer (Step 4) | `CHAPTER_5_ML_ARCHITECTURE_OPTIMIZER_v2.md` | 423 | ❌ **NO** | v2.0.0 "(Corrected)" | unknown — unaudited |
| 6 — Anti-Overfit Training (Step 5) | `CHAPTER_6_ANTI_OVERFIT_TRAINING.md` | 738 | ❌ **NO** | v3.1.0 | unknown — unaudited |
| 7 — Prediction Generator (Step 6) | `CHAPTER_7_PREDICTION_GENERATOR.md` | 933 | ❌ **NO** | v1.0 | unknown — unaudited |
| 8 — PRNG Registry | `CHAPTER_8_PRNG_REGISTRY.md` | 1,058 | ❌ **NO** | v2.4; claims `prng_registry.py` = 4,323 lines | unknown — unaudited |
| 9 — GPU Cluster Infrastructure | `CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md` | 980 | ❌ **NO** | v2.0.0 (Consolidated) — predates the Proxmox migration | topology; see §6.4. Also the source of the `reverse_sieve` coordinator-job description that made Beta reject header-only freezing of `reverse_sieve_filter.py` (§1.1, Q3) |
| 9 addendum | `CHAPTER_9_ADDENDUM_v2_2_0.md` | 152 | ❌ | Diagnostic battery + ramdisk v2.1.0, Jan 2026 | — |
| 10 — Autonomous Agent Framework | `CHAPTER_10_AUTONOMOUS_AGENT_FRAMEWORK_v3.md` | 586 | ❌ | v3.1.0, 2026-02-03, "Phase 7 Complete" **(that is the *old* Phase 7 — see §5.2)** | `_v2.md` (1,553L) superseded; `.bak` is a stale duplicate |
| 11 — Feature Importance & Visualization | `CHAPTER_11_FEATURE_IMPORTANCE_VISUALIZATION.md` | 1,099 | ❌ | — | patched by `PATCH_Chapter11_LLM_Update_v2.md` / `apply_chapter11_patch.sh` |
| 12 — WATCHER Agent & Fingerprint Registry | `CHAPTER_12_WATCHER_AGENT.md` | 880 | ❌ | v1.4.0, 2026-02-03 | two addenda not folded in: `CHAPTER_12_ADDENDUM_v1_3_0.md`, `CHAPTER_12_ADDENDUM_PHASE1_v1_1_2.md` |
| 13 — Live Feedback Loop | `CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md` | 1,231 | ❌ **NO** | "Architecture-Final" | `CHAPTER_13_LIVE_FEEDBACK_LOOP.md` (1,229L) is the superseded v1.0; §19 checklist superseded by `CHAPTER_13_SECTION_19_UPDATED.md` |
| 14 — Training Diagnostics | `CHAPTER_14_TRAINING_DIAGNOSTICS.md` | 3,199 | ❌ | v1.2.0, "ACTIVE — Phases 1, 3, 5, 6 Complete (S69–S73)" | header superseded by `CHAPTER_14_HEADER_PATCH.md` |

**Chapters 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 and 14 are UNAUDITED.** `BACKLOG.md` §1 sets the prior
explicitly: *"Chapter 1's audit found 9 of 41 claims accurate. The base rate for an unaudited chapter
in this project is not 'mostly right with a few stale lines.'"*

**Chapter working documents** (not chapters themselves): `CHAPTER_1_AUDIT_v1.md` (736L) ·
`CHAPTER_1_PATCH_S114.md` (**⛔ SUPERSEDED — never merged, and its central mechanism was deleted from
the code while it sat unmerged; retained for history only**) · `CHAPTER_2_SOURCE_MAP_v1.md` (654L) ·
`CHAPTER_3_ALIGNMENT_AUDIT.md` (925L) · `chapter2_interchunk_cleanup_section.md` (46L) ·
`CHAPTER_13_IMPLEMENTATION_PROGRESS*.md` (**14 versions**, v1.1 → v3.9, 2026-01-11 → 2026-02-15 —
read **v3.9 only**; note the duplicate `CHAPTER_13_IMPLEMENTATION_PROGRESS_v1_2 .md` with a space in
the filename) · `CHAPTER_14_IMPLEMENTATION_PROPOSAL_S69.md` (398L).

---

# 3. THE INTENT INDEX

Grouped by theme. **Test for every line: if someone asked "where is X documented", would this line
let them find it?**

## 3.1 Why the design is what it is — foundations

| file | the question it answers |
|---|---|
| `BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md` (167L) | **Why bidirectional, and why thresholds must stay loose.** For incorrect seeds forward and reverse survival are approximately independent, so `e^(−cn) → e^(−2cn)` — bidirectional **squares the exponent**. §7 is the counter-intuitive part that gets misread every time: exact sieves eliminate all variance, leaving no ranking, no gradients and **no learning signal**. |
| `TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md` (292L) | **The canonical system map**, end to end. v1.2 supersedes v1.1/v1.0. Where this and `S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md` disagree, **the report wins** — it read live source. |
| `TRIANGULATED_FUNCTIONAL_MIMICRY_VERIFIED_v1_0.md` (673L) | What "Triangulated Functional Mimicry" means as a technical method, using **verified metrics only** — the reference for describing the system without overclaiming. |
| `PIPELINE_BEHAVIOUR_MODEL.md` (1,617L, REV1) | ★ **The authority on how the pipeline behaves and why, and the second mandatory read after this catalog.** Not an audit — nothing here claims anything is broken. **Every behaviour carries two anchors** (a documentary one and a source one), and behaviours with a WHAT but no WHY are marked `INCOMPLETE` — that list is the priority list for the ser8 archive import. Produced 2026-08-03 at `49c13ad`. |
| `CLAUDE_CODE_INSTRUCTIONS_PIPELINE_BEHAVIOUR_MODEL.md` (141L, REV1) | Its brief. §0 is the load-bearing part: **"This is NOT an audit"** — the project has learned twice that *an auditor without the governance trail generates false findings at a high rate.* |
| `SKIP_SEMANTICS_SEARCH_v1.md` (407L) | ★ **Is the meaning of `skip_min`/`skip_max` for the hybrid kernels written down anywhere? VERDICT: FOUND.** The search that falsified the "nobody documented this" claim. |
| `HYBRID_SKIP_BOUND_AUDIT.md` (376L) | Do the trial's sampled `skip_min`/`skip_max` reach the hybrid kernels? (They do not; they die at `_hybrid_prefix`.) **Its line 318 premise — that hybrid skip semantics are "unspecified" — is FALSE.** Read it for the wiring trace, **not** the semantics verdict. |
| `TFM_PROJECT_FACTS_SKILL.md` (2,810L) | The committed copy of the `tfm-project-facts` skill: foundations, settled facts, the superseded list, the mandatory verification procedure. **Currency v26, 2026-08-17 — and its own most recent changelog says v26 is now STALE** (three days of rulings, the field-6 repair, both standing rules and the window-anchor design are unrecorded; `G-PROD-SHAPE` was never in it). A v27 pass is owed. **Check it against the live skill before relying on either.** |
| `CLAUDE_CODE_INSTRUCTIONS_SKILL_V13_RESTRUCTURE.md` (154L) | The brief that restructured the skill from v12 (1,047L) to v13. §0 is the durable part: **what the skill is FOR, and what it must stop being** — it is the only thing a fresh session knows before it reads anything. |
| `VERIFICATION_INTEGRITY_STANDARD.md` (159L) | **VIR-1 … VIR-5 (VIR-6 added later).** What makes a check a check: a verification must prove its own execution; vacuous-capable detectors need a clean control *and* a fault-injection control; every gate terminates in `PASS \| FAIL \| UNAVAILABLE \| INCOMPLETE`. Adopted by Beta after three incidents of *a check that was not checking, presenting as a pass*. |
| `KNOWN_ANSWER_VALIDATION_INVENTORY.md` (430L) | **Does known-answer sieve validation already exist? YES, substantially.** The inventory that struck Wall C. |
| `THRESHOLD_GOVERNANCE.md` (157L; also at repo root) | Who may change sieve thresholds and within what bounds. **Its synthetic-era defaults are superseded by `THRESHOLD_CALIBRATION_FINDINGS_S148.md`.** |
| `DESIGN_INVARIANT_GPU_ISOLATION.md` (158L) | **MANDATORY, non-negotiable:** GPU-accelerated code must never run in the coordinating process when subprocess isolation is in use. Enforced since S72. |

## 3.2 Frozen contracts and specifications

| file | the question it answers |
|---|---|
| `PROPOSAL_S172_RANGE_MINER_v1_4_5.md` (344L) | **The authoritative RANGE-MINER architecture.** Absorbs the binding S175 ruling. Where it and v1.4.4 conflict, **v1.4.5 governs.** |
| `PROPOSAL_S172_RANGE_MINER_v1_4_4.md` (747L) | **SUPERSEDED**, retained for the audit trail. Authoritative **only** for sections v1.4.5 marks PRESERVED. Frozen at `1f6c0c5`; still the named spec for Phase 3. |
| `PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md` (257L) | ★ **The live design for the F-4 repair** — see §1.1 for the full disposition. Contains the **terminology law** (anchor = start index, envelope = reachable records) and the derived-maximum machine representation. |
| `DATASET_PUBLICATION_SCHEMA_v1.md` (175L) | **FROZEN**, `manifest_schema_version: 1`. Covers the **combined** `daily3.json` only — the split files are explicitly **not** covered. **Where a brief and this schema differ, the schema wins.** |
| `RUNTIME_DATASET_PROVISIONING_CONTRACT.md` (175L) | What a run must guarantee about dataset identity across nodes. Obligations are Phase 6-P0.5, not P0. Its `expected_sha256`-as-static-config field is **superseded** — see §6.1. |
| `PROVISIONING_CONTRACT_AMENDMENT.md` (141L) | The Beta-mandated amendment inserting fail-before-dispatch and per-node verification as explicit **P0.5 obligations**. |
| `DAILY3_CONSUMER_CONTRACT_v1.md` (514L) | **What the pipeline actually requires of the draw dataset**, established by tracing live code — so a rewritten producer cannot silently break downstream steps. |
| `S172_INFRASTRUCTURE_INTERFACE_v1_0.md` (200L+) | The miner ↔ rig deployment contract. Source of the CT100-hostname-equals-rig-name identity rule. **Extended since the last catalog** (+46 lines) with the fleet-launch surface the Gate-12 campaign built. |
| `CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md` (274L) | **RATIFIED.** Authority boundaries between selfplay, Chapter 13 and WATCHER. Binding on all future implementation. |
| `CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md` (735L) | The LLM-guided selfplay Strategy Advisor contract (Beta-authored): what the advisor may emit and under what activation gate. |
| `CONTRACT_SECTION_8_5_ADDENDUM.md` (61L) | Adds §8.5 (LLM lifecycle dependency) to the contract above; Beta-approved S67. |
| `SPEC_BUNDLE_FACTORY_v1_1_0.md` (373L) | How step-awareness bundles are constructed. |
| `ADDENDUM_A_STEP_AWARENESS_BUNDLES_v1_0.md` (364L) | **LOCKED, joint Alpha + Beta.** The bundle format itself (`agents/contexts/bundle_factory.py`). |
| `TRSE_v1_15_SPEC.md` (333L) | The TRSE (Temporal Regime Segmentation) v1.15 specification. **SPEC ONLY — its text describing Rules B and C as applied is SUPERSEDED.** |
| `TRSE_INTEGRATION_PLAN_S121.md` (192L) | ★ **How TRSE integrates with Step 1 — and the file that keeps being cited without being opened.** §2B lists the manifest's `default_params`; §2C specifies the **PASSIVE** integration and the `min(rec_ws * 4, …)` rule that makes `8 × 4 = 32`. |
| `PHASE_9B2_INTEGRATION_SPEC.md` (413L) | Beta-approved integration spec for `selfplay_orchestrator.py` v1.0.6 → v1.1.0. |

## 3.3 RANGE-MINER (S172) — implementation briefs, Phases 4–6

`CLAUDE_CODE_INSTRUCTIONS_*` are implementation briefs written *to* Claude Code on VM101. Each names
a base commit, forbids commit/push/WATCHER, and stops at a gate. **A brief describes what was
authorised, not necessarily what shipped** — pair it with the review record in §1.4.

| file | the question it answers |
|---|---|
| `CLAUDE_CODE_INSTRUCTIONS_S172_PHASE4.md` (328L) | Staged Phase-4 coordinator implementation (Stage 0 = the Blocker-6 `ResidueResolver` patch, Stage 4 = the L7 abort interface). |
| `CLAUDE_CODE_CORRECTION_S172_PHASE4_SERVE.md` → `..._CORRECTION6_..._ADMISSION.md` (6 files) | The six Beta rejection rounds in order: real serve path → six serve/ledger defects → six async-staging/socket defects → four overload freezes → three real-resource bounds → one admission byte-model. Pair each with the matching `TEAM_ALPHA_REVIEW_S172_PHASE4_REV*.md`. |
| `..._S172_PHASE5.md` (459L) | Phase-5 umbrella: NPZ writer + assembly, plus the prerequisite Phase-4 seam correction. |
| `..._S172_PHASE5_D1.md` (907L, REV5) | D1.0 workflow/terminal-race + D1.1 four-population assembly engine and concrete `Phase5Sink`. |
| `..._S172_PHASE5_D2.md` (227L) | Directional uniqueness at **both** layers. |
| `..._S172_PHASE5_D3.md` (396L, REV3) | The shared backend-neutral 24→22 columnizer + independent structural validator. **[A1]** `prng_base` restricted to a forward, non-hybrid base family. |
| `..._S172_PHASE5_D3_0.md` (193L) | Legacy seam correction: canonical PRNG/skip encoding + rectangular 22-array empty output. |
| `..._S172_PHASE5_D3_25.md` (383L, REV3) | Mode-preserving backend result contract + canonical candidate-ingress normalisation. |
| `..._S172_PHASE5_D3_5.md` (819L, REV3.1) | The shared run finalizer. **[D1]** the chain **tip** is authenticated — the generation directory is named `<generation_id>--<sidecar_sha256>`, making the atomic `current` pointer the trust anchor. |
| `..._S172_PHASE5_D3_5_B.md` (276L, REV2) | Seed-Domain v1.1; the per-link chain contract specified explicitly. |
| `..._S172_PHASE5_D4.md` (335L, REV3) | `serial_reference` behind a frozen two-backend interface; `BackendAssemblyResult` + `AssemblyMeasurement` frozen. |
| `..._S172_PHASE5_D5.md` (436L) + `..._D5_REV2_ADDENDUM.md` + `..._D5_REV3_ADDENDUM.md` | The `process_sharded` backend. REV2 = Beta ruled **Option B**. REV3 = the `int64` seed-projection divergence and its lossless fallback. |
| `..._S172_PHASE5_D6.md` (234L) + `..._S172_PHASE5_D6_CORRECTION.md` (188L) | D6, the production integration adapter — **the first deliverable touching real silicon and real Step-1/Step-2 wiring.** The correction is the hardcoded-`0.25` threshold repair. |
| `CLAUDE_CODE_CORRECTION_S172_PHASE5_D0_REV2.md` · `_REV3.md` · `_REV4.md` | The three D0 correction rounds, in order. |
| `..._S172_D6_1_FLUSH_DURABILITY.md` (176L) | Incremental NPZ atomic flush and durability. Beta's framing is operative: **incremental durability does not currently exist.** |
| `..._S172_D6_2_CHECKPOINT_RECONCILIATION.md` (467L, REV5) | The 24-field checkpoint, canonical reconciliation and the finalizer resume path. |
| `..._S172_D6_2_REV5_BINDING_ADDENDUM.md` (176L) | **BINDING — where this and REV5 differ, THIS WINS.** Four normative items; no REV6. |
| `..._S172_D6_2_BOUNDED_REPAIR.md` (205L) | The two execution-path defects the 29 D6.2 gates do not exercise. Repair **on top of** `f7583bc`; do not revert. |
| `..._S172_D6_3_RETENTION_INVESTIGATION.md` (129L) | **READ-ONLY.** Checkpoint retention: authorises no fix, no policy and no deletion. |
| `..._S172_PROCESS_SHARDED_IMPORT_GATE.md` (169L) | **Beta-REQUIRED hardening**, test-side only: prove `assert_cpu_only` reds on a module-level GPU import, in a fresh spawned interpreter, against the **production** forbidden list. |
| `..._S172_THRESHOLD_REPAIR.md` (171L, REV2) | The optimizer threshold-propagation repair. REV2 shrinks scope because Beta retired PWC/ZMQ from certifying authority. |
| `..._S172_PHASE_6_0_ROCM_PARITY.md` (208L) | Single-rig ROCm smoke against an identical CUDA control. Beta's required addition: **schema-valid ROCm output alone does not establish computational parity.** |
| `..._BOUNDED_PHASE_6.md` (170L) | **The Phase-6 certification brief.** |
| `..._PHASE_6_P0_SCOPING.md` (164L) | **READ-ONLY.** Where does a published dataset live, and what breaks when it moves? |
| `..._PHASE_6_P0_IMPLEMENTATION.md` (166L) | **P0 CREATES FILES. P0 DOES NOT CHANGE RUNNING CODE.** |
| `..._PHASE_6_P0_5_IMPLEMENTATION.md` (176L) | P0.5, the behavioural cutover. The inversion is deliberate: every behavioural change lands together against a published baseline, so the first post-publication certification has **one** cause to attribute. |
| `..._P0_5_Q2_CLOSURE.md` (117L) | The single P0.5 closure condition: a missing provisioning manifest must hard-fail a miner-backed run. |
| `..._ADMISSION_LIVENESS_REPAIR.md` (148L) | The §4.3 silent hang: separate **admission** (bounded) from **execution maintenance** (unbounded). |
| `..._RESOLVED_EXECUTION_SET.md` (146L) | One frozen run-scoped fleet authority; all six existing mechanisms become consumers. |
| `..._ADMISSION_BINDING_REPAIR.md` (142L) | The two repairs Beta required before granting Phase-7 closure. |
| `..._S172_PHASE_6_P2_SCRAPER.md` (385L, REV4 **DRAFT**) | Append-only immutable dataset publication. **Pending Beta**; the schema wins where they differ. |
| `..._S172_PHASE_7_SOAK.md` (189L, REV1 **DRAFT**) | Phase 7: 50-trial WATCHER soak, **≥5 high-survivor and ≥5 low-survivor trials, mixed constant/hybrid, per-trial cleanup verification.** |
| `CLAUDE_CODE_INSTRUCTIONS_CHAPTER_1_AUDIT.md` · `..._CHAPTER_1_P0_CORRECTION.md` · `..._CHAPTER_1_P1_P2_CORRECTION.md` · `..._CHAPTER_2_SOURCE_GATHERING.md` · `..._CHAPTER_2_RESTORE.md` · `..._CHAPTER_2_CORRECTIONS.md` · `..._CHAPTER_1_AND_2_CLOSURE.md` · `..._CHAPTER_3_ALIGNMENT_AUDIT.md` | The eight chapter-track briefs, in execution order. Each states its own scope limit — audit-only, documentation-only, or code-and-docs — and **three of them say explicitly that no fix is authorised.** |
| `S172_PHASE4_BRIEF.md` (535L, rev-4) | The Phase-4 implementation brief itself — Blockers 1–7, Decisions A/B, L1–L8, gates 1–36. Cited constantly; **open it rather than citing it.** |
| `S172_PHASE5_D5_CHAT_PROMPT.md` (73L) | The D5 kickoff prompt. |

## 3.4 The Gate-12 campaign — briefs, reports and submissions

**All 60-odd files of this chain are laid out in execution order in §1.3.** This subsection exists so
a filename search lands somewhere useful; **§1.3 is where you read the story.**

| group | files | what the group settles |
|---|---|---|
| staging repair | `CLAUDE_CODE_INSTRUCTIONS_STAGING_DIR_FIX.md` (269L) · `..._STAGING_DIR_PART_B.md` (209L) · `S172_STAGING_PART_B_REPORT.md` (455L) · `TEAM_ALPHA_STAGING_DIR_NOTE.md` (159L) | The miner-backed WATCHER path could not stage a result, **and no gate had ever driven the production path.** Produced `tests/gate_s172_prod_shape.py`, still unrun. |
| fleet launch | `CLAUDE_CODE_INSTRUCTIONS_MANUAL_FLEET_LAUNCH.md` (174L) | **RANGE-MINER has no worker-launch mechanism and has never had one.** How to bring up 25 daemons by hand. Later superseded operationally by `scripts/launch_fleet_manual.sh`. |
| back-pressure | `..._S172_STAGING_BACKPRESSURE_REMEDIATION.md` (263L) · `CLAUDE_CODE_REPORT_S172_STAGING_BACKPRESSURE.md` (494L) · `TEAM_ALPHA_REVIEW_S172_STAGING_BACKPRESSURE.md` (117L) · `TB_SUBMISSION_S172_STAGING_BACKPRESSURE.md` (161L) · `TEAM_ALPHA_DEFERRED_QUEUE_NOTE.md` (166L) | The deferred-queue bound and its classification. **Option E (seed-cap / stripe-geometry change) REJECTED outright.** |
| F1–F5 amendment | `..._S172_BP_AMENDMENT{,_R2,_R3}.md` · `CLAUDE_CODE_REPORT_S172_BP_AMENDMENT{,_R2,_R3}.md` · `TB_SUBMISSION_S172_BP_{AMENDMENT_DELTA,R2_DELTA,R3_DELTA}.md` | Three HOLD rounds ending in exact-envelope credit tokenization and a pre-decode barrier. |
| staging capacity | `..._STAGING_CAPACITY_AMENDMENT.md` · `..._S172_STAGING_CAPACITY_R1.md` · `_R2.md` + the three reports + three submissions | Option-C architecture, `elapsed_s` persistence, the 16-stripe geometry and the 3,264 derivation, Gate 37 supersession. Committed `4dd5535`. |
| seed domain | `..._SEED_DOMAIN_CURSOR_AMENDMENT.md` · `_R1.md` · `_R2.md` + three reports + three submissions | `[0, 2^32)` shared terminus, legacy-tracker deauthorization, first-gap cursor, append-only ledger. Committed `a3bb4da`. **R2 was TEST-ONLY: "do not modify production for this finding."** |
| dispatch stall | `..._S172_GATE12_DISPATCH_STALL.md` (122L) · `CLAUDE_CODE_REPORT_S172_GATE12_DISPATCH_STALL.md` (413L) | Read-only diagnosis. Its constraint block is the canonical statement of **"you do not launch anything"** (`CLAUDE.md` rule 3, harness-enforced). |
| attempt-1 forensics | `TB_GATE12_EVIDENCE_PACKAGE_FAIL.md` (159L) · `..._GATE12_FAILURE_FORENSICS.md` (161L) · `CLAUDE_CODE_REPORT_GATE12_FAILURE_FORENSICS.md` (726L) · `TB_SUBMISSION_GATE12_FORENSICS_F1_DEFECT.md` (150L) | The F1 production defect, and **why Alpha refused the `worker_pool_size = 25` "fix"**. |
| F1/F2 amendment | `..._F1_F2_ACTIVE_LEASE.md` (272L) · `_R1.md` · `_R2.md` + three reports + three submissions + `TB_NOTE_F1_F2_SUBMITTED.md` · `TB_NOTE_R1_INFLIGHT_AND_ACCESS_PATTERN.md` | Active-lease scheduler + durable terminal observability. Committed `c4e0037`. |
| pre-rerun | `..._PRERUN_PROBE_AND_SAMPLER.md` · `_R1.md` · `_R2.md` · `_R3.md` + four reports + four submissions | Truthful GPU probe (certified and frozen at R1) and the post-F1 concurrency sampler. `49ff9b4`, then `4643a11`. |
| attempt-2 forensics | `TB_GATE12_ATTEMPT2_FORENSIC_REPORT.md` (176L) | The per-stage `expected_workers` admission window. |
| Defect A / B | `..._DEFECT_A_TRANSPORT_RECOVERY.md` · `..._DEFECT_A_R2_DEADLINE.md` · `..._DEFECT_B_TURNOVER_AGGREGATION.md` + three reports + four submissions | Worker transport-session reconnect/re-register, and all-qualifying-window turnover aggregation. **Criterion 1 (simultaneity) must not be weakened.** |
| attempt-6 forensics | `ATTEMPT6_RIG_LOG_FORENSIC_v1_0.md` (420L) | **BOUNDED-UNRESOLVED.** §7 (A–F) is the H1/H2 instrumentation spec. |
| attempt-7 / MP-1 | `GATE12_ATTEMPT7_H1H2_FORENSIC.md` (249L) · `GATE12_MP1_RUN_FORENSIC.md` (297L) | **H1 REFUTED · H2 CONFIRMED**, then the drain-starvation cause named and measured. Neither proposes a remedy — **the choice of mechanism is Beta's.** |
| attempt-6 implementation | `TB_SUBMISSION_ATTEMPT6_IMPLEMENTATION.md` (401L) | R2 for certification; the scope proof **proves rather than asserts** that the coordinator and worker are unchanged from R1 (sha256 of the digest body compared to the preserved R1 reference). |
| field 6 | `..._FIELD6_OBSERVABILITY_REPAIR.md` (192L) · `CLAUDE_CODE_REPORT_FIELD6_OBSERVABILITY_REPAIR.md` (483L) | See §1.1. Landed `d8b21e3`. |

## 3.5 RANGE-MINER (S172) — evidence and reports

| file | the question it answers |
|---|---|
| `S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md` (486L) | The ROCm/CUDA parity evidence record. Notes the base-commit substitution (`8e2f5bf` is docs-only on top of the `3823b56` the brief names). |
| `D6_RELEASE_GRADE_CERTIFICATION_RECORD.md` (153L) | The release-grade certified generation from the clean real repository, 2026-07-29. Raw evidence: `D6_RELEASE_GRADE_SMOKE_20260729.log` (16K). |
| `D6_FOLLOWUP_BOTH_MODES_SMOKE.md` (107L) | Proves the **variable-skip / hybrid column** end-to-end on real silicon — a different kernel and different seed caps from D6's constant-skip smoke. |
| `S172_D6_2_IMPLEMENTATION_REPORT.md` (737L) | Full D6.2 implementation evidence. |
| `S172_D6_2_BOUNDED_REPAIR_REPORT.md` (406L) | The bounded-repair evidence; completion sentinel `PASS`. |
| `S172_THRESHOLD_PROPAGATION_REPAIR_REPORT.md` (419L) | Evidence for the threshold repair against `THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md`. |
| `S172_PROCESS_SHARDED_IMPORT_GATE_REPORT.md` (502L) | The import-gate deliverable, incl. the contamination guard (the mutant must red **for the right reason**). |
| `S172_PHASE_7_PREREQ_REPORT.md` (826L) | §1 prerequisite measurement for the soak; a 25-worker execution set proven freezable **by construction**; §6 checkpoint census. **No soak launched.** **Predates the 2026-08-04 soak attempt and the whole Gate-12 campaign.** |
| `S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md` (681L) | ★ Two independent questions answered from live source with `file:line` anchors: **the survivor feature schema** (91 extracted / 89 trained; three namespaces; five dead placeholders) and **per-survivor attribution** (implemented, invoked, unreachable, unconsumed). |
| `S172_SIEVE_PATH_VERIFICATION_SCOPE.md` (100L) | ★ **What is and is not proven about the four sieve paths** — so nobody mistakes "Phase 3 green" for "the sieve computes correct survivors through the miner". Standing reference. |
| `STEP2_BIDIRECTIONAL_SIEVE_DESCRIPTIVE_TRACE.md` (1,205L) | A read-only descriptive survey of Step 2 as built, 2026-07-28. `:501` already governs the trial-level nature of `bidirectional_selectivity` — see §3.6. |
| `ROCm_Saturation_Report_S172.md` (195L) | ROCm driver-level saturation: boundary mapping, failure analysis, mitigation. **The measurement behind the PWC → RANGE-MINER pivot.** |
| `PROVENANCE_DISPOSITION_ACCUMULATOR_20260725.md` (268L, REV2) | Accumulator provenance disposition. **REV2 incorporates all four Beta corrections — a copy without the REV2 banner is the superseded draft.** |
| `PHASE6_PREREQS.md` (441L, REV5) | Operational prerequisites for real-silicon testing. **REV4 corrected five of seven statuses from live measurement** and records that D3.0-B was never completed. |
| `PHASE7_PREREQUISITES.md` (119L) | The durable answer to "what stands between us and the Phase 7 soak?" **Current as of 2026-07-30, at D6.1 — it predates the entire Gate-12 campaign and every prerequisite that campaign added. Do not read it as the current answer.** |
| `PHASE_6_P0_SCOPING_v1.md` (702L) | The read-only scoping report: where a published dataset lives and the blast radius of moving it. |
| `PHASE_6_P0_IMPLEMENTATION_v1.md` (363L) | The P0 implementation record. |
| `FALLBACK_PARITY_PASS1_20260815.md` (198L) + `FALLBACK_PARITY_PASS1_20260815_pipfreeze.txt` | ★ **Pass 1 of the two-pass `CLAUDE.md` §5 fallback-parity review, and it does NOT convert the verdict.** Beta had ruled the S183/S184 parity line *not measured, not credited, and not to be silently converted to PASS.* This runs the half that is runnable while Zeus is booted into Proxmox, records the VM101 baseline pass 2 will diff against, and leaves the overall verdict **UNRESOLVED.** The `pipfreeze` is that baseline. |
| `docs/phase6_evidence/wall_ab.json` · `known_answer_gate.json` · `sampler_control_arm.json` | The machine-readable Phase-6 evidence the bounded-Phase-6 sentinels cite. |

## 3.6 Audits and read-only investigations

**These are the documents most likely to already contain the answer you are about to go looking for.**

| file | the falsifiable question it answers |
|---|---|
| `CLAUDE_CODE_REPORT_PIPELINE_OVERVIEW.md` (1,013L) | **What does the pipeline do, end to end?** Read-only, 2026-08-08, search order governance trail → chapters → code. Deliberately does **not** duplicate `PIPELINE_BEHAVIOUR_MODEL.md`, which remains the authority; it answers the brief's specific questions and cites the model for the rest. Pairs with `CLAUDE_CODE_INSTRUCTIONS_PIPELINE_OVERVIEW.md` (76L). |
| `CLAUDE_CODE_REPORT_STEP1_PURPOSE_LINEAGE.md` (786L) | ★ **What is Step 1 *for*?** A read-only *historical* reconstruction from full git history — every quote re-read from the live file, every DB figure from a read-only connection opened in the session. Brief: `..._INSTRUCTIONS_STEP1_PURPOSE_LINEAGE.md` (85L). |
| `CLAUDE_CODE_REPORT_SIEVE_CONTINUITY_MODEL.md` (753L) | ★ **What continuity does the sieve assume?** **Q1: one continuous generator state carried across the entire window, initialised once per (seed, skip-hypothesis) pass and thereafter only advanced** — no kernel reinitialises, reseeds or resets between observations. **Q2: reseed/breakpoint/regime concepts exist in the trail and in code but none reaches the sieve's continuity model**; machine identity and A/B-RNG selection: **no evidence found anywhere.** **Q3: skip is an abstraction over several physical causes at once**; constant mode applies it uniformly across the window, hybrid per-gap, and `skip_min`/`skip_max` bind **only** constant mode. Brief: `..._INSTRUCTIONS_SIEVE_CONTINUITY_MODEL.md` (85L). |
| `CLAUDE_CODE_REPORT_ATTACK_PLAN_FROM_PROCEDURES.md` (1,096L) | **What attack does the official draw procedure document actually license?** Read-only analysis producing a proposal, from `docs/reference/CA_DAILY_SLP_DRAW_PROCEDURES_20210609.pdf` (23 pages, sha256 `7048b255…f74`, identity verified in the report). **§D.1 is the load-bearing section** — it describes the window-anchor/generator-phase change, says of itself *"described only, not implemented"*, and calls itself *"a proposal to Beta, and the highest-value one in this report."* That is the origin of the window-anchor track. Brief: `..._INSTRUCTIONS_ATTACK_PLAN_FROM_PROCEDURES.md` (117L). |
| `CLAUDE_CODE_REPORT_ATTACK_PLAN_BLACKBOX_REEVAL.md` (565L) | ★ **The re-evaluation that retracts the report above.** *"Two of my three load-bearing negative conclusions were wrong, and one of them was wrong twice."* The owner identified a framing error invalidating parts of C, D and E. **Read this before citing any negative conclusion from the procedures report.** Brief: `..._INSTRUCTIONS_ATTACK_PLAN_BLACKBOX_REEVAL.md` (135L). |
| `CLAUDE_CODE_REPORT_SELECTIVITY_PER_SEED_AUDIT.md` (469L) | ★ **Is `bidirectional_selectivity` a per-seed quantity? VERDICT: TRIAL-LEVEL.** Computed once per (trial × skip-mode) as a ratio of two set cardinalities and stamped verbatim onto every survivor of that trial — **zero per-seed information, by construction.** Explicitly reported **as status, not as a discovery**, because `STEP2_BIDIRECTIONAL_SIEVE_DESCRIPTIVE_TRACE.md:501` already governed it. **This is the structural reason behind `TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md`.** Brief: `..._INSTRUCTIONS_SELECTIVITY_PER_SEED_AUDIT.md` (84L). |
| `CLAUDE_CODE_REPORT_OPTUNA_TO_FULL_SWEEP_PIVOT.md` (557L) | ★ **Did Step 1 pivot from Optuna-over-partial-ranges to sweeping the full 2³² space?** *"Michael's recollection is substantially CORRECT, and anchored in a real, ruled, implemented decision — but one word in it was struck by Team Beta, and that word is what makes today's wiring look like a contradiction."* Brief: `..._INSTRUCTIONS_OPTUNA_PIVOT_SEARCH.md` (71L). |
| `AUDIT_STEP1_OFFSET_REACH.md` (432L) | ★ **Was Step 1's `offset` bounded such that the sieve only ever examined a small slice at the front of the 18,068-draw history? CONFIRMED — and it was already recorded in five places, one of them a RANK-1 open item.** The binding constraint is the **pair** `offset ≤ 100` **and** the window size, not `offset` alone. **No fix authorized, none made.** This is the audit the window-anchor proposal repairs. |
| `CLAUDE_CODE_INSTRUCTIONS_STEP3_SCRIPT_READ.md` (97L) | The brief that asks what `run_step3_full_scoring.sh` actually does, given `STEP_SCRIPTS[3]` names it while `full_scoring.json`'s actions name three Python modules instead. **Deliverable is the final message only — "create NO file"** because the tree had to stay clean for the soak. **No report artifact exists**, which is why §7 gap 2 remains open. |
| `SER8_ARCHIVE_INVENTORY.md` (256L) | ★ **What is in the pre-repository ser8 archive? Completion sentinel: `UNAVAILABLE`.** ser8 is reachable from VM101 and up, but **VM101 holds no credential ser8 accepts** — no directory listing was obtained, no file opened, no inventory exists. The brief's own rule was followed rather than substituting guesses from filenames seen in a screenshot. Brief: `..._INSTRUCTIONS_SER8_ARCHIVE_INVENTORY.md` (156L). |
| `CLAUDE_CODE_INSTRUCTIONS_2389B61_AUDIT_AND_WARMSTART_RESTORE.md` (175L) + `..._WARMSTART_RESTORE_PART_B.md` (180L) | ★ **`2389b61` has now reverted TWO independent fixes, three months apart** — a whole-block overwrite that silently reverts unrelated work behind an accurate-sounding commit message. Part A is a read-only audit; Part A found **three** out-of-scope reverts and **only one is restored** by Part B. The canonical instance of that defect class. |
| `CLAUDE_CODE_INSTRUCTIONS_BACKLOG_MAINTENANCE.md` (163L) | The brief that repairs and updates `BACKLOG.md` — and states what that register is *for*. |
| `CLAUDE_CODE_INSTRUCTIONS_CATALOG_CORRECTIONS.md` (134L) | The 2026-08-03 correction pass on this catalog and `PIPELINE_BEHAVIOUR_MODEL.md` — two corrections and one coverage report, no restructure. Applied at `f8cb1c5`. |
| `CLAUDE_CODE_INSTRUCTIONS_PROJECT_CATALOG_REGENERATION.md` (REV2) + `CLAUDE_CODE_INSTRUCTIONS_CATALOG_REGENERATION_20260820.md` (107L) | **The authority for this file.** REV2 governs in full; the 2026-08-20 delta adds the anchor, the currency delta and the must-not-miss list. |
| `CHAPTER_1_AUDIT_v1.md` (736L) | Does Chapter 1 describe the live window optimizer? **9 of 41 claims accurate.** |
| `CHAPTER_3_ALIGNMENT_AUDIT.md` (925L) | Does Chapter 3 describe the code today? **55 claims: 17 / 9 / 24 / 5.** Sentinel `PASS`. **NO FIX WAS AUTHORISED.** |
| `CHAPTER_2_SOURCE_MAP_v1.md` (654L) | Where would the material for a Chapter-2 reconstruction come from? Superseded in purpose once restore-and-audit was ruled. |
| `TRSE_STEP0_AUDIT_v1.md` (537L) | What does TRSE compute, and do its outputs reach anything? |
| `STRATEGY_ADVISOR_AUDIT_v1.md` (779L) | What does the Strategy Advisor emit, what validates it, what applies it, what executes it — **and where does the chain break?** |
| `SAMPLER_BEARING_v1.md` (662L) | **Cost and blast radius** of four working Optuna samplers in Step 1. READ-ONLY scoping — not an implementation and not an authorisation. Optuna 4.4.0. |
| `STRATEGY_ORIGIN_AUDIT.md` (396L) | Were RandomSearch / GridSearch / EvolutionarySearch **ever** Optuna-backed? A read-only *history* investigation. |
| `THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md` (384L) | In the window-optimizer / PWC route, does a configured threshold reach the kernel, or is it dropped for a default? **Explicitly adjudicates a `tfm-project-facts` §2.7 claim.** |
| `FLEET_STATE_REQUIREMENTS_v1.md` (548L+) | What does a run actually demand of the fleet, and do the mechanisms agree? **They do not** — six checks, three granularities, two address sets. **Extended since the last catalog** (+66/−…) with the Gate-12-era fleet findings. |
| `DOCUMENTATION_AUDIT_20260131.md` (179L) | Which project-knowledge documents were stale as of 2026-01-31? **Itself now stale — a historical record of a staleness sweep.** |
| `S107_session_log.md` (108L) | The Step-2 v4.1→v4.4 repair narrative, incl. **`sample_size=450` hardcoded in the shell script** so WATCHER's override was silently dropped, and `.162` missing from both the scp push loop and `ml_coordinator_config.json`. |
| `WATCHER_KPI_CALIBRATION_FINDINGS_S176.md` (269L) | Analytic + deterministic validation of the WATCHER hit/survivor KPIs. **Recommend-only — changes nothing in `watcher_policies.json`.** |
| `THRESHOLD_CALIBRATION_FINDINGS_S148.md` (373L) | **Authoritative** empirically-grounded sieve threshold defaults; supersedes the synthetic-era values in `THRESHOLD_GOVERNANCE.md`. |
| `STEP1_EXECUTION_FLOW_AND_PRUNING_S147.md` (196L) | Both Step-1 execution paths and the pruning logic, as of S147. |
| `STEP1_GPU_BENCHMARK_SUITE.md` (702L) | The systematic benchmark methodology for preventing GPU overload during Step 1. |
| `GPU_THROUGHPUT_INVESTIGATION_PLAN_v1_0.md` (244L) | The planned throughput investigation (S126). **PLANNED — pending execution.** |
| `ROOT_CAUSE_ANALYSIS_RRIG6600C_S151.md` (183L) | rrig6600c persistent crashes — **RESOLVED**, S151. |

## 3.7 How the system is operated

| file | the question it answers |
|---|---|
| `COMPLETE_OPERATING_GUIDE_v2_0.md` (1,181L) | The current operating guide (v2.2.0, S135 / 2026-03-10). **Predates RANGE-MINER, the Proxmox migration, dataset authority and the entire Gate-12 launch harness.** |
| `COMPLETE_OPERATING_GUIDE_v1_1.md` (760L) | SUPERSEDED by v2.0. |
| `Cluster_operating_manual.txt` (96K) | The cluster operating manual. Carries the `skip_min`/`skip_max` **input** reading verbatim at `:948-949`. |
| `Cluster_operating_manual_v1_1_update.md` (136L) | Session-17 changes — includes the record that **Step 0 PRNG fingerprinting was ARCHIVED**: mathematical analysis proved fingerprinting impossible under mod-1000 projection. |
| `instructions.txt` (152K) | The long-running operating document. ★ **`:1182-1183` is the load-bearing line** — the `skip_min`/`skip_max` *input* (element-wise pattern bound) reading, hybrid default `[0,16]`; **`:1230-1245`** is the Oct-2025 output spec declaring `skip_pattern` and `pattern_stats`, the literal ancestor of the three dead skip features. |
| `INSTRUCTIONS_NPZ_ADDITION.md` (260L) | The NPZ v3.0 binary survivor format section written for `instructions.txt`. |
| `CANONICAL_PIPELINE_AND_CH13_WITH_STARTUP_COMPLETE.md` (340L) | End-to-end operational walkthrough of the canonical pipeline plus Chapter 13. |
| `complete_workflow_guide_v2_PULL_UPDATED.md` (2,728L) | The v2.0 workflow guide — manual per-step vs orchestrated, with variable-skip support. |
| `complete_workflow_guide_update_v2_1.md` (205L) | The v2.1 delta: `scripts_coordinator.py` v1.4.0. |
| `README.md` (339L) | The docs-tree README (a copy of the root README). See §6.4 on its framing. |
| `PROJECT_STATUS.md` (85L) | Component-readiness snapshot at **S109 / 2026-02-23**. Historical. |
| `REMOTE_NODE_SETUP_CHECKLIST.md` (278L) | How to stand up a new remote worker node. **Bare-metal era — predates the CT100 model.** |
| `TELEGRAM_NOTIFICATION_SYSTEM_REFERENCE.md` (323L) | The cluster notification system. **v2.1, Proxmox topology incl. pzeus.** The most topology-current operational document in the tree. |
| `LLM_INFRASTRUCTURE.md` (35L) | **A pointer document, deliberately.** The canonical LLM-cluster docs live in the `rx6600-llm-inference` repo, not here. |
| `WATCHER_POLICIES_REFERENCE.md` (211L) | **The canonical meaning of every flag in `watcher_policies.json`.** |
| `SOAK_TEST_PLAN_PHASE7_v1_0.md` (850L) | The *old* Phase-7 (WATCHER dispatch) soak plan — **not** S172 Phase 7. |
| `SOAK_TEST_HANDOFF_PROMPT.md` (316L) | A resumable context prompt. Carries the standing framing: **"This is NOT specifically a lottery system"** — PRNG-agnostic by design, all generator behaviour abstracted via `prng_registry.py`. |
| `SOAK_C_GAPS_AND_PATCHES_v1_0.md` (620L) | Soak-C integration gaps (the acceptance engine did not honour `test_mode` flags) and proposed patches. |
| `SUBPROCESS_ISOLATION_INTEGRATION_GUIDE.md` (335L) | How to integrate subprocess isolation into `meta_prediction_optimizer_anti_overfit.py`. |
| `GBNF_DEPLOYMENT_README.md` (279L) | Deploying the GBNF grammars that constrain LLM output. |
| `Distributed_PRNG_Pipeline_Overview_for_Novices.pdf` · `Distributed_PRNG_Pipeline_Technical_Addendum.pdf` (16K each) | Onboarding overview and technical addendum. **PDFs — not read in this pass; contents UNVERIFIED.** |

## 3.8 External source material

| file | the question it answers |
|---|---|
| `reference/CA_DAILY_SLP_DRAW_PROCEDURES_20210609.pdf` (666,629 bytes, sha256 `7048b255…f74`) | ★ **The primary external source for the physical model of skip and session-separated streams** — *California State Lottery, Daily & SuperLotto Plus Draw Procedures*, MODIFIED for Release for Solicitation, effective 2021-06-09, 23 pages, text-extractable. **This closes what the previous catalog listed as gap 3 ("exists only as a citation").** It underpins `TEAM_ALPHA_PUSHBACK_ORDERING_AND_THRESHOLD_REGRESSION.md`'s per-session equipment argument and is the source `CLAUDE_CODE_REPORT_ATTACK_PLAN_FROM_PROCEDURES.md` derives from. **Binary — its content is quoted in that report, which was read in this pass; the PDF itself was not re-parsed here.** |

## 3.9 Backlog, TODO, status and handoff

| file | the question it answers |
|---|---|
| `BACKLOG.md` (**449L, 19 entries**) | ★ **The live register.** Everything known, deliberately deferred, and **not** a Phase-7 blocker — written down so it is not rediscovered as a surprise finding. **Currency stated in-file as 2026-08-03; §19 was added 2026-08-17, so the header understates it.** Entries: 1 unaudited chapters · 2 skip-output work · 3 sampler-comparison sequencing · 4 three `[WATCHER][RETRY]` log lines · 5 session-separated dataset authority · 6 non-terminating multi-stripe loopback **[UNVERIFIED]** · 7 `dataset_provenance/*.json` never pruned · 8 two Beta-required pre-Phase-7 audits · 9 small/verified/unfixed · 10 standing reminders · 11 `_RusageChildrenSampler` measures the wrong thing · 12 **D3.0-B OPEN — it NARROWS what Phase 6 certified** · 13 NP2 checkpoint transaction design · 14 `2019-01-25` is evening-only · 15 Step 3's output validation floor is three contracts stale · 16 `full_scoring.json` declares 26 GPUs while the frozen Phase-7 set is 25 · 17 a skill revision lives in three places and committing updates one · 18 `chapter_13_triggers.py` reaches Step 3 outside `--end-step` · **19 `.s172_accumulator/generations/` is durable data plane with no backup policy — ruled real, ruled non-blocking, unowned.** New findings go here. |
| `TODO_MASTER_S*.md` (**19 files**: S101, S114, S120, S122, S125b, S126, S127, S132, S135, S139, S143, S145, S148, S150, S152, S154, S163, S163KARG, S170) | A rolling P0/P1 master list. **Read only the latest (`TODO_MASTER_S170.md`, 2026-04-24) — and note it predates the entire S172 track.** Their real value is historical: each header states the cluster and pipeline state on its date. |
| `TODO_SELFPLAY_AND_LLM_AUTONOMY.md` (425L) | The autonomy last-mile track — **tracked and unstarted.** Part B, 20 tasks. **Task B3 auto-extracts tunable parameters from `agent_manifests/*.json`**, which is exactly why the next entry exists. |
| `D6_THRESHOLD_AUTONOMY_SIGNPOSTS.md` (91L) | Anti-bite signposts so the threshold disconnect cannot be picked up by autonomy work — **they take on no autonomy work themselves.** |
| `TODO_PHASE7_WATCHER_INTEGRATION_REVISED.md` / `_v3.md` | The old Phase-7 (WATCHER integration) task list. v3 marked ALL PARTS COMPLETE. (v3's header contains mojibake from an encoding round-trip.) |
| `TODO_DISPLAY_AND_VISUALIZATION.md` (150L) | Terminal display + visualisation improvements. **PENDING since 2025-12-06.** |
| `SESSION_61_HANDOFF.md` · `SESSION_81_HANDOFF.md` · `SESSION_HANDOFF_20260204.md` · `SESSION_CONTINUATION_PHASE7_PART_B.md` · `SESSION_NOTES_20260102.md` · `SESSION_NOTES_20260118_PIPELINE_TEST.md` | Point-in-time handoffs. Useful only for reconstructing what was believed on a given date. |
| `S1xx_CHAT_PROMPT.md` (**13 files**: S142, S144, S147, S149, S151, S152, S155, S157, S159, S160, S163, S164, S170) | Session-opening context prompts, each stating the cluster state, HEAD and P0/P1 priorities of its day. **Historical state records, not instructions.** (`S148_CHAT_PROMPT.md` and `S162_CHAT_PROMPT.md` are the same class but live at the repo root.) |
| `DOCUMENTATION_INDEX_v1_0.md` / `_v1_1.md` (274L each) | The Session-78/80 documentation indexes. **Name-indexed — superseded by this catalog, and the reason this catalog is intent-indexed instead.** |
| `DOCUMENTATION_UPDATES_S70.md` · `_S71.md` · `_S146.md` | Which documents needed updating after S70, S71 and S146. Useful for dating a doc's last intended sync. |
| `PROJECT_FILE_CATALOG.md` | this file |

## 3.10 Autonomy, WATCHER, LLM and selfplay

| file | the question it answers |
|---|---|
| `PROPOSAL_WATCHER_KPI_GOVERNANCE_STATES_v1_0.md` → `_v1_1.md` → `_v1_2_ADDENDUM.md` | The BOOTSTRAP → CALIBRATING → GOVERNED KPI state machine. **All three are recommend-only and change nothing; no thresholds are selected.** Read **v1.1 + the v1.2 addendum** together — v1.0 is superseded. Rulings: S177 → S178 → S179. |
| `CLAUDE_CODE_BRIEF_WATCHER_KPI_VALIDATION_v1.md` (200L) | Establish evidence-backed baselines for the WATCHER governance KPIs and validate the Step-0 (TRSE) advisory heuristics on real data. |
| `CLAUDE_CODE_BRIEF_S176_FOLLOWUP_v1.md` / `_S177_RESUBMISSION_v1.md` / `_S178_FOLLOWUP_v1.md` | The three implementation briefs the S176/S177/S178 rulings unblocked. The cluster-heavy Phase-C walk-forward is explicitly **out of scope** in S176. |
| `WATCHER_PHASE1_PATCH_v1.1_FINAL.md` (544L) | Stale-output prevention for WATCHER — the freshness check. Supersedes `WATCHER_PHASE1_PATCH_FOR_REVIEW.md` (337L). |
| `CLAUDE_CODE_INSTRUCTIONS_STRATEGY_ADVISOR_AUDIT.md` (151L) | The audit brief for the advisor. **A parked reference document.** |
| `CLAUDE_CODE_INSTRUCTIONS_SAMPLER_BEARING.md` (146L) | What would four working samplers cost? **A cost estimate, not an authorisation.** |
| `CLAUDE_CODE_INSTRUCTIONS_CHAIN_C_TRUTHFULNESS_HOTFIX.md` (133L) | **A narrow truthfulness repair, deliberately small.** WATCHER reports `Applied:` for LLM parameter proposals that never execute. Beta ruled the **filtering correct** — only the reporting is defective. |
| `CLAUDE_CODE_INSTRUCTIONS_TRSE_STEP0_AUDIT.md` (145L) | **"Do not assume a defect exists."** The TRSE audit brief. |
| `SELFPLAY_ARCHITECTURE_PROPOSAL_v1_0.md` (651L) | Multi-model inner-episode training. **APPROVED by Beta + user.** Reframed by `TB_UPDATE_SELFPLAY_REFRAMING_2026-07-28.md` — read both. |
| `SELFPLAY_INTEGRATION_PROGRESS_v1_0.md` (391L) | Selfplay integration progress, 2026-01-29. |
| `PROPOSAL_EPISTEMIC_AUTONOMY_UNIFIED_v1_3.md` (913L) | The unified epistemic-autonomy architecture, v1.3. **Implementation-Complete** as of 2026-02-10. |
| `PROPOSAL_v1_3_FINAL_ACCEPTANCE.md` (551L) | The acceptance record for the above. |
| `PROPOSAL_LLM_Architecture_v2_0_0.md` (387L) | **DeepSeek-R1-14B primary + Claude Opus backup**, validated by A/B testing. |
| `LLM_STRATEGY_DEEPSEEK_CLAUDE_DOCUMENTED.md` (489L) | The documented LLM roles and decision-making, extracted from the actual implementation. |
| `PROPOSAL_LLM_Infrastructure_Optimization_v1_1.md` (857L) | LLM subsystem optimisation + grammar completion. |
| `PROPOSAL_LLM_Reasoning_Refactor_v1_0.md` / `_v1_1.md` | v1.1 is **TEAM BETA APPROVED (with conditions)**; v1.0 superseded. |
| `PROPOSAL_LLM_Router_v2_1_0_Merge.md` (344L) | `llm_services/llm_router.py` API restoration + missing method. |
| `PROPOSAL_LLM_Terminology_Solution_v1_0_0.md` (557L) | The LLM terminology-drift solution — **HIGH priority, blocking agent autonomy** at the time. |
| `PROPOSAL_SEARCH_STRATEGY_VISIBILITY_FIX_v1_0.md` (415L) | Advisory-layer blindness to the search strategy actually in use. |
| `PROPOSAL_STRATEGY_ADVISOR_LIFECYCLE_INTEGRATION_v1_0.md` (428L) | LLM lifecycle integration + heuristic demotion for `parameter_advisor.py`. |
| `PROGRESS_BUNDLE_FACTORY_AND_STRATEGY_ADVISOR_v1_0.md` (412L) | Execution plan for bundle factory v1.1.0 + the advisor. |
| `PROPOSAL_Unified_Agent_Context_Framework_v3_2_*.md` (**12 files**: v3.2.0, .1, .3, .4, .5, .6, .7, .8, .9, .10 + two addenda) | The agent-context framework lineage, Dec 2025. **Read v3.2.10 only.** The two addenda carry the distinct content: the **threshold addendum** records that sieve thresholds were hardcoded at 0.01, and **Addendum B** records the creation of the Steps 2–6 manifests with their `parameter_bounds`. |
| `proposals/PROPOSAL_Universal_Agent_Architecture_v1_1.md` (1,375L) + `_ADDENDUM.md` (1,244L) | The Dec-2025 universal agent architecture, both **DRAFT — Pending Review**. The framework above is what shipped. |
| `proposals/README.md` | The proposals-directory README. See §6.4 on its framing. |
| `Multi-Model_Architecture_integration_autonomy.md` (396L) | How the multi-model architecture integrates with `watcher_agent.py`. |
| `NOTE_Step7_Not_Required_for_Autonomy.md` (182L) | **DECISION RECORDED** (Beta): Step 7 is not required for autonomy. **Replaces `PROPOSAL_Step7_PostPipeline_Export_v1_1.md`.** |
| `PROPOSAL_Modular_Step_Automation_Framework_v1_1.md` (1,077L) | The step-automation framework — core autonomy infrastructure, Jan 2026. |

## 3.11 ML, features, thresholds and objectives

| file | the question it answers |
|---|---|
| `PROPOSAL_ML_Architecture_Remediation_v2_0.md` (458L) | ★ The complete ML-architecture diagnosis. **Its `:150-158` is the source of the `skip_min`/`skip_max` *output* reading** — "minimum/maximum gap that **worked**", with *"tight skip range = stronger hypothesis"*. |
| `PROPOSAL_Feature_Implementation_Remediation_v1_0.md` (309L) | ★ **Of 64 defined features, only 2 had actual variance.** The finding that made ML training effectively useless. |
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
| `PROPOSAL_S145_R1_Progressive_Empirical_Sweep.md` (352L) | Progressive sweep of seed IDs 0→2³² with cross-session survivor accumulation and persistent Optuna continuity. **Beta-approved conditionally; supersedes the rejected `PROPOSAL_S145_Complete_Seed_Space_Sweep.md`.** **Its terminus and cursor law are now executable in `utils/seed_coverage_ledger.py`** (§1.3). |

## 3.12 Infrastructure, GPU faults and cluster stability

| file | the question it answers |
|---|---|
| `PROPOSAL_Proxmox_LXC_Rig_Migration_v1_0.md` (150L) | The LXC migration plan for the three rigs. **Superseded on acceptance criteria.** |
| `PROPOSAL_Infrastructure_Reconciliation_S172_v1_0.md` (172L) | **APPROVED (Beta)** — the Proxmox container strategy. |
| `CLAUDE_CODE_INSTRUCTIONS_TOPOLOGY_DOC_CORRECTION_v2.md` (209L) | Correct the topology **documentation** to the boot-selector model. **DOCUMENTATION ONLY — no `.py`, `.sh` or `.json` may be edited.** |
| `PROPOSAL_FINAL_ROCm_HIP_Init_Fix_v2_1.md` (262L) | **APPROVED FOR IMPLEMENTATION, CRITICAL.** Parallel HIP initialisation on ROCm. |
| `PROPOSAL_FREE_ALL_BLOCKS_REPLACEMENT_v1_0.md` (416L) | Replace `free_all_blocks()` in `sieve_gpu_worker.py`'s cleanup with safe GPU memory management. |
| `PROPOSAL_S155_CUPY_POOL_FIX.md` (178L) | **P0 — blocks all production runs.** The CuPy memory-pool fix. |
| `PROPOSAL_S162_RRIG6600C_CRASH_ROOT_CAUSE_v1_0.md` (410L) | rrig6600c crash root cause + fix options. |
| `PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md` (350L) | **APPROVED WITH MODIFICATIONS** — the persistent-worker-coordinator lifecycle fix. |
| `PROPOSAL_ZMQ_SQLITE_COORDINATOR_S158D_v1_0.md` (134L) | The ZMQ+SQLite distributed sieve coordinator. |
| `PROPOSAL_Job_Batching_Pipeline_Stability.md` / `_v2.md` | Job batching for pipeline stability. **v2 supersedes v1.** |
| `PROPOSAL_RAM_Disk_Data_Preloading_v1_0.md` (297L) | RAM-disk preloading for distributed workers. |
| `PROPOSAL_Unified_Ramdisk_Steps_3_and_5_v1_1.md` (639L) | Extending the ramdisk to Steps 3 and 5, plus lifecycle management. |
| `PROPOSAL_Infrastructure_Improvements_v1_0.md` (309L) | Preflight checks, smarter parameter tuning and GPU health monitoring, after a 2-hour WATCHER run failed silently. |
| `PROPOSAL_Incremental_Output_Writing_v1_0.md` (435L) | Incremental output writing for the window optimizer. |
| `PROPOSAL_NPZ_Auto_Conversion_Step2.md` (271L) | NPZ auto-conversion for Step 2.5 — a pipeline gap, not a blocker. |
| `IPC_SERIALIZATION_FIX_IMPLEMENTATION_GUIDE_S150.md` (286L) | The approved `slim_v1` IPC design. |
| `PROPOSAL_Documentation_Paradigm_Correction_v1_2.md` (182L) | ★ The functional-mimicry language cleanup — **the origin of the naming rule.** Documentation only. |

## 3.13 Session changelogs — **181 committed files, summarised as a group**

`SESSION_CHANGELOG_*.md`. **Two naming forms:** **175 dated**
(`SESSION_CHANGELOG_YYYYMMDD[_TAG].md`, **2026-01-09 → 2026-08-20**) and **5 session-ID-only**
(`S160`, `S161`, `S162`, `S162_FINAL`, `S162_VICTORY`), plus `SESSION_CHANGELOG_TEMPLATE.md`
— **181 committed in total** (`git ls-files docs/ | grep SESSION_CHANGELOG`).
Session IDs span **S1 → S185**. **SR-2 (§1.0) governs all new ones: date + topic, no new S-numbers.**

**What they are good for, and only this:**
- **Establishing when a behaviour changed**, and under whose authority. `git log --grep` over a
  symbol plus the changelog for that date is the fastest route to *why*.
- **Recovering a governing decision that never became its own document.** The canonical example:
  `SESSION_CHANGELOG_20260307_S122.md:56` carries the ruling *"disabled per TB + S121 shuffle test"* —
  one of the three citations proving TRSE Rules B and C are advisory **by design**, not dropped
  wires. That ruling exists nowhere else.

**What they are not:** a status source. A changelog states what was believed on its date. Eight
months of them do not compose into a current picture — use §1, §2 and `BACKLOG.md` for that.

**The fourteen most recent, which between them cover the entire Gate-12 campaign** (all are indexed
individually in §1.3): `20260811_CLEANTREE_ADMISSION` · `20260811_F1_LEASE_ORIGIN` · `20260813_S180` ·
`20260814_S181` · `20260814_S182` · `20260815_S183` · `20260815_S184` · `20260815_S185` ·
`20260816_MP1_DRAIN_ATTRIBUTION` · `20260817_R1_R4_DRAIN_REMEDY` · `20260817_GATE12_ATTEMPT9` ·
`20260817_GOVERNANCE_RULINGS` · `20260820_FIELD6_AND_WINDOW_ANCHOR_DESIGN`. A fourteenth file sits
between them on disk — `20260819_S1` — and is **untracked and excluded**; see §7 gap 1.

**A note on where the briefs live.** From 2026-08-11 onward the Gate-12 briefs and reports were
written to `~/dashboard_work/` and are **not in the repository** — the changelogs name them
(`CCODE_BRIEF_*_v1_0.md`, `*_REPORT.md`) but the artifacts themselves are host state and out of this
catalog's scope. The changelog is the committed record of what those briefs required.

## 3.14 Non-`.md` files in `docs/` (13)

| file | what it is |
|---|---|
| `instructions.txt` (152K) · `Cluster_operating_manual.txt` (96K) | see §3.7 — both load-bearing for skip semantics |
| `FALLBACK_PARITY_PASS1_20260815_pipfreeze.txt` | The VM101 `pip freeze` baseline that fallback-parity pass 2 must diff against. See §3.5. |
| `window_optimizer_integration_final.py` (100K) | **A `.py` file living in `docs/`** — a copy of the Step-1 integration layer, placed here as reference material. **Do not edit it as if it were the live module**; the live one is at the repo root, and it was modified during the Gate-12 campaign. |
| `apply_chapter11_patch.sh` (4K) | Applies `PATCH_Chapter11_LLM_Update_v2.md` to Chapter 11. |
| `D6_RELEASE_GRADE_SMOKE_20260729.log` (16K) | Raw evidence for `D6_RELEASE_GRADE_CERTIFICATION_RECORD.md`. |
| `CHAPTER_1_WINDOW_OPTIMIZER.md.bak` · `CHAPTER_3_SCORER_META_OPTIMIZER.md.bak` · `CHAPTER_4_FULL_SCORING.md.bak` · `CHAPTER_10_AUTONOMOUS_AGENT_FRAMEWORK_v3.md.bak` | **Stale duplicates. Do not read; do not cite.** |
| `TEAM_BETA_REVIEW_kfolds_S100.docx` (12K) | Beta's k-folds review. **Binary — UNVERIFIED.** |
| `Distributed_PRNG_Pipeline_Overview_for_Novices.pdf` · `..._Technical_Addendum.pdf` | see §3.7. **UNVERIFIED.** |

*(`reference/CA_DAILY_SLP_DRAW_PROCEDURES_20210609.pdf` and `phase6_evidence/*.json` are subdirectory
files — see §3.8 and §3.5.)*

---

# 4. CODE INVENTORY, BY ROLE

**Scope note:** a role map, not a completeness audit. The repo root holds **1,026 files**, the great
majority one-shot `apply_*.py` / `fix_*.py` patch scripts and `test_*.py` throwaways from 180+
sessions; those are characterised as a class in §4.9 rather than enumerated.

## 4.1 FROZEN — reuse, never reimplement

**Importing these is mandatory; forking them is the defect they exist to prevent.**

| symbol / file | anchor | why frozen |
|---|---|---|
| `_l2_sort_key` | `utils/run_finalizer.py:690` | Highest **float32** score → lowest `trial_number` → constant-before-variable, *within a trial only*. **Comparing pre-rounding float64 is the defect this converts away.** |
| `_select_l2_winners` | `utils/run_finalizer.py:714` | Same-trial/same-mode collision raises `AccumulatorConsistencyError`. |
| `canonical_map_hash` | `utils/run_finalizer.py:486` | The map-identity anchor carried through the generation chain. |
| `CANONICAL_ARRAY_CONTRACT` | `utils/canonical_arrays.py:98` | The frozen 22-array NPZ contract. Consumed at `run_finalizer.py:803` and `canonical_arrays.py:582`. **The window-anchor ruling keeps this wall closed** — metadata may gain `window_anchor`/`generator_phase`/`anchor_era` provided no array is added, removed, reordered, retyped or reshaped. |
| `CANONICAL_RECORD_FIELDS` | `utils/canonical_arrays.py:143` **and** `utils/canonical_records.py:115` | The 24-field canonical record. **Defined in two modules — check which one your consumer imports.** Also consumed at `utils/checkpoint_d6_2.py:288`. |
| `utils/prng_encoding.py` | whole module | The shared registry-derived PRNG type encoding (Phase 0, `2389b61` — and see §3.6 on what else that commit silently reverted). |
| The finalizer validators | `run_finalizer.py:522, 558, 585, 634, 665, 884, 1004, 1069, 1113, 1176` | Ten `_validate_*` functions incl. `_validate_chain` (`:1176`) and `_validate_current_pointer` (`:1113`). |
| **D3.5 finalizer-owned root symlinks** | `run_finalizer.py` ~`:1400-1404` | `bidirectional_survivors_all.npz` / `..._binary.npz` are **symlinks the finalizer owns.** A regular file appearing there makes `finalize_run` raise `PublicationError`. **Both are now gitignored** — see §4.8. |
| **The kernel ABI** | `prng_registry.py` per-family builders; `range_miner_worker.py:197-198`, `:220`, `:948` | **FROZEN AND BINDING for window-anchor v1** (`TB_RULING_WINDOW_ANCHOR_SEQUENCING.md` Q3). Certified signatures byte-for-byte. `lcg32_hybrid`/`pcg32_hybrid` carry a phase argument; `java_lcg`/`minstd`/`xorshift32`/`xorshift128` hybrids do not; all covered reverse hybrids carry a trailing `int32(offset)`. Any need for independent phase on the four no-phase forward hybrids is a **separate kernel-ABI v2 dependency** with its own certification cycle. |

## 4.2 `miner/` — the RANGE-MINER engine (Step 2)

| file | role |
|---|---|
| `range_miner_protocol.py` | Length-prefixed JSON framing (4-byte big-endian + compact UTF-8, 64 MB cap); 8 message types; `from_dict()` filters unknown kwargs via `dataclasses.fields()`; unknown `message_type` → `ValueError`. **All envelope fields carry defaults** — deliberately unlike `persistent/pwc_protocol.py`. |
| `range_miner_worker.py` | The per-GPU daemon: READY handshake, sub-stripe loop, `ResidueResolver`, threshold consumption, effective-threshold reporting. **Carries the transport-session recovery state machine (Defect A) and the H1/H2 stripe-lifecycle instrumentation** added during the Gate-12 campaign. `:649-650` is the host-side residue slice the window-anchor design splits. |
| `range_miner_coordinator.py` | Stripe assignment, admission, staging, lease expiry, the retry matrix, `serve_trial`. **The single most-amended file of the Gate-12 campaign** — back-pressure, credit tokenization, active-lease scheduling, the serve-loop `[S172-SL]` seam, MP-1 three-level attribution, the R-1…R-4 drain remedy and the field-6 emitter all live here. **SR-1 (§1.0) governs any further change to it.** |
| `range_miner_npz_writer.py` | Trial assembly and the 22-array NPZ write-back; `AssemblingPhase5Sink`. |
| `assembly_backends.py` | The frozen two-backend interface; `serial_reference` (default). |
| `assembly_shard_worker.py` | `process_sharded`'s shard worker. Owns `assert_cpu_only()` and `_FORBIDDEN_GPU_MODULES` — **the import gate exercises these; it does not duplicate them.** |
| `dataset_authority.py` | Pointer resolution, the run-start freeze, per-node provisioning and on-target verification; `DatasetProvisioningError`. |
| `step1_ingress.py` | Miner candidates → the Step-1 accumulator (D6). |

## 4.3 `utils/` — the shared authorities

`run_finalizer.py` (the generation chain and publication) · `canonical_arrays.py` (24→22 columnizer
+ structural validator) · `canonical_records.py` · `checkpoint_d6_2.py` (24-field checkpoint, both
digest layers, reconciliation, path confinement) · `prng_encoding.py` · `survivor_loader.py` ·
`metrics_extractor.py` · **`seed_coverage_ledger.py` (NEW)**. **Most of §4.1 lives here.**

**`utils/seed_coverage_ledger.py`** is the executable form of §§1-7 of Beta's *"S145 / SEED-DOMAIN
SWEEP TERMINUS AND COVERAGE AUTHORITY"* ruling (2026-08-07): the `[0, 2^32)` terminus, Coverage
Ledger v1 and the cursor law. It **replaces** `database_system.get_next_seed_start()`, which meant
`MAX(seed_range_end)` over `exhaustive_progress` — a table Beta **deauthorized wholesale** (rows
retained, zero certified progress) because it advanced past the governed frontier with no terminus.
Append-only, content-derived identity, bare `INSERT`, triggers, `recursive_triggers=ON`. Beta called
it *"a strong solution to the exact clobber class that damaged the old tracker."*

## 4.4 Pipeline steps (repo root)

| step | primary module(s) |
|---|---|
| 0 — Regime Segmentation (TRSE) | `trse_step0.py`, `trse_calibration_probe.py`, `trse_entropy_probe.py`, `step0_heuristic_validation.py` |
| 1 — Window Optimizer | `window_optimizer.py`, `window_optimizer_bayesian.py`, `window_optimizer_integration_final.py` |
| 2 — Bidirectional Sieve | **`miner/`** (current) · legacy: `sieve_filter.py`, `sieve_gpu_worker.py`, `reverse_sieve_filter.py`, `hybrid_strategy.py` · kernels: `prng_registry.py` |
| 2.5 — Scorer Meta-Optimizer | `run_scorer_meta_optimizer.py` / `.sh`, `generate_scorer_jobs.py`, `scorer_trial_worker.py` |
| 3 — Full Scoring | `run_step3_full_scoring.sh`, `generate_full_scoring_jobs.py`, `full_scoring_worker.py`, `aggregate_scoring_results.py`, `survivor_scorer.py` |
| 4 — Adaptive Meta-Optimizer | `adaptive_meta_optimizer.py` |
| 5 — Anti-Overfit Training | `meta_prediction_optimizer_anti_overfit.py`, `train_single_trial.py`, `nn_gpu_worker.py`, `inner_episode_trainer.py` |
| 6 — Prediction Generator | `prediction_generator.py`, `build_pools.py`, `evaluate_pools.py`, `backtest_pools.py` |
| Feedback (Ch. 13) | `chapter_13_orchestrator.py`, `chapter_13_triggers.py`, `chapter_13_acceptance.py`, `chapter_13_diagnostics.py`, `chapter_13_llm_advisor.py`, `draw_ingestion_daemon.py`, `per_survivor_attribution.py` |
| Ch. 14 diagnostics | `training_diagnostics.py`, `training_health_check.py`, `diagnostics_llm_analyzer.py`, `diagnostics_analysis_schema.py` |
| Selfplay | `selfplay_orchestrator.py`, `policy_conditioned_episode.py`, `policy_transform.py`, `reinforcement_engine.py` |
| Fleet authority | `execution_set.py` |
| Preflight | `preflight_check.py` — **carries the certified three-outcome GPU probe** (`UNAVAILABLE` ≠ 0 with `gpu_count None`, located binary, stderr surfaced, advisory gating preserved). **Certified and frozen at pre-rerun R1; do not modify or re-verify it as a side effect of other work.** |
| Legacy engines (non-certifying) | `persistent_worker_coordinator.py`, `persistent/`, `zmq_sqlite_coordinator.py`, `zmq_sqlite_worker.py`, `coordinator.py`, `distributed_worker.py` |
| Data | `daily3_scraper.py` (**now tracked**, `334dacf`), `pa_pick3_scraper.py`, `convert_survivors_to_binary.py`, `validate_survivors.py` |
| Launch harness | `gate12_launch.sh` (23K, root, **tracked**) — the Gate-12 launcher; `:54` was the line that *printed* tree state instead of testing it, `§0.6`/`§2.6` carry the parity and liveness walls added by S182/S183 |

## 4.5 `agents/`

`watcher_agent.py` (the orchestrator; `STEP_SCRIPTS`/`STEP_MANIFESTS`/`STEP_NAMES` at `:387-417`) ·
`watcher_dispatch.py` · `watcher_registry_hooks.py` · `fingerprint_registry.py` · `agent_core.py` ·
`agent_decision.py` · `doctrine.py` · `full_agent_context.py` · `prompt_builder.py` ·
`registry_inspector.py` · `threshold_guardrail.py` · `progress_display.py` · subpackages
`contexts/` (bundle factory), `step_runner/`, `manifest/`, `parameters/`, `pipeline/`, `registry/`,
`runtime/`, `safety/`, `prompts/`, `data/`, `history/`.
**Note:** `agents/` also holds three stale in-place backups — `agent_core.py.bak2`,
`watcher_agent.py.bak2`, `watcher_agent.py(bakpregrammar)`.

## 4.6 `tests/` — 51 entries (was 33)

**The certifying verifier that has never run.** `tests/gate_s172_prod_shape.py` (336L) — **G-PROD-SHAPE**,
*"THE gate whose ABSENCE caused this defect."* It verifies that a **real** production call shape ran
end to end: real WATCHER execution → manifest defaults → `window_optimizer.py` → real
`MultiGPUCoordinator` → RANGE-MINER backend → coordinator staging → all required trial phases →
committed 22-array NPZ → Step-2 load-back with `fallback_used=False`. **It is a verifier, not a
driver** — it reads the artifacts and ledger a completed run leaves behind, plus three
**anti-fabrication** checks (no `self.staging_dir = …` substitute coordinator; no CLI-only
`--miner-output-dir` standing in for the manifest; the canonical staging value originated from the
manifest). **Every previously-certified miner run failed exactly those three checks**, which is why a
defect that kills every production run survived Phase-6 certification. Status: **built, proven red
against the failed 2026-08-04 soak log (9 pass / 5 fail), NOT RUN, Michael-initiated only, requires a
live 25-daemon fleet.** It is Phase 7's certifying verifier — see §5.2.

Phase/deliverable-scoped gates, one per governed deliverable:
`test_s172_phase1_scaffolding` · `phase2_protocol` · `phase3_worker` (17/17) ·
`phase4_coordinator` (63/63; **gate 22 is the coexistence gate**, gate 37 superseded during the
staging-capacity amendment) · `phase5_d0` · `d1_workflow` · `d1_engine` ·
`d2_directional_uniqueness` · `d3_columnizer` · `d3_0_encoding_contract` ·
`d3_25_candidate_ingress` · `d3_5_finalizer` · `d4_serial_backend` · `d5_process_sharded` ·
`d6_production_adapter` · `d6_threshold_path` · `d6_1_flush_durability` ·
`d6_2_checkpoint_reconciliation` · `phase6_p05_dataset_authority` (38/38 `--fleet`) ·
`process_sharded_import_gate` · `threshold_propagation` · `resolved_execution_set` (34/34) ·
`admission_binding` (20/20) · `admission_liveness` (16/16) · `test_chapter1_p0_corrections` (12/12) ·
`test_chapter2_content_gate` (12/12) · `test_prng_encoding` · `test_watcher_llm_integration` ·
`dry_run_s115.py` · `smoke_s172_phase5_d6_zeus_single_gpu.py` · plus `phase6/`
(`wall_ab_gate.py`, `known_answer_gate.py`) and `fixtures/`.

**The Gate-12 campaign suites (new since the last catalog):**
`test_s172_staging_backpressure` (→ 50/50) · `test_s172_staging_partb` (24/24) ·
`test_s172_elapsed_roundtrip` (6/6) · `test_seed_domain_cursor_amendment` (→ 40/40) ·
`test_gate12_cleantree_admission` (→ 31/31) · `test_gate12_concurrency_sampler` (→ 49/49) ·
`test_gate12_gpu_gate` (9/9) · `test_preflight_gpu_probe` (12/12) ·
`test_s172_f1_f2_active_lease` (→ 16/16) · `test_s172_f1_lease_origin` (→ 18/18) ·
`test_s172_defect_a_transport_recovery` · `test_s172_d6_integration_repair` (65 → 82) ·
`test_s172_d6_liveness_gate` (59/59) · `test_s172_attempt6_remediation` ·
`test_s172_h1h2_instrumentation` (**Beta-certified R1–R7**) · `test_s172_mp1_drain_attribution` ·
`test_s172_r1_drain_remedy`.

**⚠ Two standing operational facts about this suite set.**
**(1) Gate 22 (`test_s172_phase4_coordinator.py`) reds on any stray untracked `.py`** — that is why
every deliverable registers its new paths in the whitelist, and why a dirty-tree red there is a
**transitional** condition that self-clears on commit (`TB_RULING_FIELD6_IMPLEMENTATION.md` §4:
**do not widen the allowlist** to make it green).
**(2) The S172 suites must be run SEQUENTIALLY.** Concurrent runs flake Part B `G-VAL-6` on a
free-space race that reads exactly like a regression from your own diff.

## 4.7 `scripts/` — 20 entries

**Gate-12 gates (all new):** `gate12_cleantree_gate.py` (the tree-state test the launcher previously
only printed) · `gate12_gpu_gate.py` · `gate12_concurrency_sampler.py` (the post-F1 occupancy sampler;
`state='claimed'` semantics certified) · `gate12_parity_gate.py` (806L — fail-closed rig **source**
parity before worker dispatch; **expected values are the full 64-hex SHA256 of the canonical clean
local tree derived at run time**, an AST arm forbids any hex run ≥ 12 in a non-docstring literal, and
**acceptance authority is content identity, never Git identity**) · `gate12_sentinel_gate.py` (the
same-record `SESSION_SENTINEL`-carrying-the-nonce conjunction) · `gate12_worker_liveness_gate.py` ·
`launch_fleet_manual.sh` (remote dispatch detachment + launch ACK).

**Pre-existing:** `provision_dataset_fleet.py` · `verify_dataset_publication.py` ·
`extract_search_bounds_snapshot.py` · `apply_caps.py` · boot-notify installers
(`install_boot_notify_amd.sh`, `install_boot_notify_pzeus.sh`, `update_boot_notify_v2.sh`,
`cluster_boot_notify.sh`) · rig probes (`probe_phase_A_amd.sh`, `probe_phase_A_rtx.sh`,
`probe_phase_C_stability.sh`) · Telegram diagnostics (`check_telegram_rrig6600b.sh`,
`diagnose_conf_rrig6600b.sh`).

## 4.8 `.gitignore` — one change that carries a ruling

`.gitignore` gained a documented block (and lost a stale negation) after the attempt-9 tree-state
question: **`.s172_accumulator/`, `bidirectional_survivors_all.npz` and
`bidirectional_survivors_binary.npz` are now ignored.** The comment block is worth reading in place —
it records that this **supersedes `006623c`** (*"restore …binary.npz to tracking — real survivor
data, must stay in git"*), which was correct for what the path was **then** (mode 100644, a regular
data file) and wrong for what `46a3828`/D3.5 made it (a finalizer-owned symlink retargeted every
run). The removed `!bidirectional_survivors_all.npz` negation dated from `ad5ab8d` and was **doubly
inert**: it sat under `*.json`, which cannot match a `.npz`, and it carried a trailing `#` comment,
which `.gitignore` treats as part of the pattern.

## 4.9 The one-shot patch corpus (repo root)

**~207 `apply_s*.py` / `fix_s*.py` / `patch_*.py` files**, each a session-scoped, already-applied
edit script named for its session (S73 → S174), plus **~208 `test_*.py` / `create_*_test.py` /
`launch_*.sh`** exploratory harnesses from the same period. **Their value is forensic** —
`apply_s149a_rocr_isolation.py` is the evidence that the S149 ROCR ruling was implemented. **They are
not a runnable interface and must not be re-executed.** The root also holds **41 `tmp*.json`** scratch
files, 27 `.md` files (incl. `S148_CHAT_PROMPT.md`, `S162_CHAT_PROMPT.md`, `THRESHOLD_GOVERNANCE.md`,
`PROPOSAL_Schema_v1_0_4_Dual_LLM_Architecture.txt`), several `.bak` / `.save` / `(broken` copies of
`sieve_filter.py`, `prng_registry.py`, `coordinator.py`, `survivor_scorer.py` and
`window_optimizer.py`, and four `agents_*.tar.gz` archives. **1,026 files at the root in total.**

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
below were parsed from the live JSON in this pass. **All three maps are byte-unchanged since the
previous catalog.**

| step | `STEP_NAMES` | `STEP_MANIFESTS` entry | manifest version / `pipeline_step` | `STEP_SCRIPTS` entry | manifest `actions` → scripts | `default_params` count | documenting chapter |
|---|---|---|---|---|---|---|---|
| **0** | Regime Segmentation (TRSE) | `trse.json` | 1.15.1 / 0 | `trse_step0.py` | *(no `actions` key)* | **7** | **none** — the spec is `TRSE_v1_15_SPEC.md` + `TRSE_INTEGRATION_PLAN_S121.md` |
| **1** | Window Optimizer | `window_optimizer.json` | 1.8.0 / 1 | `window_optimizer.py` | `window_optimizer.py` ✅ **agrees** | **32** ⬆ *(was 25)* | **Chapter 1** ✅ audited, closed |
| **2** | Scorer Meta-Optimizer | `scorer_meta.json` | 1.3.0 / 2 | `run_scorer_meta_optimizer.sh` | `generate_scorer_jobs.py`, `scorer_trial_worker.py` ⚠ **DIVERGES** | **8** | **Chapter 3** — numbered **3**, documents **Step 2.5** ⚠ |
| **3** | Full Scoring | `full_scoring.json` | 1.3.0 / 3 | `run_step3_full_scoring.sh` | `generate_full_scoring_jobs.py`, `full_scoring_worker.py`, `aggregate_scoring_results.py` ⚠ **DIVERGES** | **10** | **Chapter 4** ❌ unaudited |
| **4** | ML Meta-Optimizer | `ml_meta.json` | 2.0.0 / 4 | `adaptive_meta_optimizer.py` | `adaptive_meta_optimizer.py` ✅ **agrees** | **4** | **Chapter 5** ❌ unaudited |
| **5** | Anti-Overfit Training | `reinforcement.json` | 1.10.0 / 5 | `meta_prediction_optimizer_anti_overfit.py` | *(no `actions` key)* | **10** | **Chapter 6** ❌ unaudited |
| **6** | Prediction Generator | `prediction.json` | 1.5.0 / 6 | `prediction_generator.py` | `prediction_generator.py` ✅ **agrees** | **7** | **Chapter 7** ❌ unaudited |

**Total declared `default_params` across the seven step manifests: 78** *(was 71)*.
Every manifest's `pipeline_step` field matches its `STEP_MANIFESTS` key — **no key/field mismatch.**

**What moved.** `window_optimizer.json` gained **seven staging parameters** — `staging_dir`,
`staging_workers`, `staging_queue_depth`, `staging_deferred_max`, `staging_capacity_timeout`,
`staging_high_water_files`, `staging_high_water_bytes` — in `arg_map`, `default_params` and
`evaluation_params`. This is the manifest half of the staging-capacity amendment: it is what makes
`staging_dir` **manifest-supplied** rather than CLI-only, which is precisely what `G-PROD-SHAPE`'s
third anti-fabrication check asserts. The manifest's own note on `staging_workers` is worth
quoting — *"Beta did NOT rule a new number — 'tune after measurement'; this is today's value made
REACHABLE, not a retune."* (Four non-`default_params` mojibake repairs also landed in the same file.)

## 5.1 Divergences between the two maps — reported, not diagnosed

1. **Step 2 — `STEP_MANIFESTS[2]` and `STEP_SCRIPTS[2]` name different things.** The manifest is
   `scorer_meta.json`, whose `actions` invoke `generate_scorer_jobs.py` and `scorer_trial_worker.py`;
   `STEP_SCRIPTS[2]` is `run_scorer_meta_optimizer.sh`, which appears in no manifest action.
   **This divergence is how a soak hazard reached launch day** — see `CHAPTER_3_ALIGNMENT_AUDIT.md`.
2. **Step 3 has the identical shape.** `STEP_SCRIPTS[3]` is `run_step3_full_scoring.sh`; the
   manifest's three actions name `generate_full_scoring_jobs.py`, `full_scoring_worker.py` and
   `aggregate_scoring_results.py`, none of which is the shell script. **A brief to read that script
   now exists** (`CLAUDE_CODE_INSTRUCTIONS_STEP3_SCRIPT_READ.md`) — but it deliberately produced **no
   file**, so no committed artifact records the answer. See §7 gap 2.
3. **Steps 0 and 5 declare no `actions` at all**, so the two maps cannot be compared for them.
   `trse.json` instead carries `skip_on_fail: true` with a stated `skip_on_fail_reason` — that
   silent-failure behaviour is **architected**, per `TRSE_INTEGRATION_PLAN_S121.md` §2C.

## 5.2 Chapter-numbering and phase-name hazards

- **Chapter numbers are not step numbers.** Chapter 3 documents **Step 2.5 / WATCHER step 2**. The
  bidirectional sieve documented in **Chapter 2** runs inside **Step 1**, not WATCHER step 2.
- **"Phase 7" is overloaded.** The Phase 7 marked COMPLETE in Chapters 10, 12 and 13 and in
  `TODO_PHASE7_WATCHER_INTEGRATION_REVISED_v3.md` is **WATCHER dispatch integration (Feb 2026)**.
  **S172 Phase 7 is the 25-GPU saturation + WATCHER soak**, and it is **UNBLOCKED as of 2026-08-17**.
  (**25 = 24 AMD RX 6600 XT + one VM101 RTX 3080 Ti**; the second 3080 Ti stays on VM100. Owner-ruled,
  **Team Beta ratified the waiver**. Frozen execution set `bea580e7…f67a8`. Older "26-GPU" wording
  predates the ruling — and `full_scoring.json` still declares 26, `BACKLOG.md` §16.)
- **"Gate 12" and "the Phase-7 soak" are different runs.** Gate 12 is the production-class Step-1
  acceptance run (`--end-step 1`); it PASSED at attempt 9. The Phase-7 soak is the 50-trial WATCHER
  soak certified by `G-PROD-SHAPE`, and it has **not** run. Beta additionally classified the soak
  **NON-CERTIFYING for window-anchor semantics** — it is observability/autonomy evidence only and
  must not later be cited as acceptance evidence for the window-anchor merge.
- **Step 0 has no documenting chapter** (§7 gap 3).

## 5.3 Manifest inventory notes

- `agent_manifests/` holds **9 files**: the 7 step manifests, `definitions.json`, and
  `scorer_meta.json.bak` (stale).
- **All `agent_manifests/*.json` match `.gitignore:41` (`*.json`).** The 7 step manifests are
  nonetheless **tracked** (force-added). `agent_manifests/trse.json` was force-added at `93918f5`
  (2026-08-01).
- **`definitions.json` is the only manifest still untracked and ignored.** Its keys are
  `schema_version`, `pipeline_steps`, `sidecar_schema`, `watcher_protocol`, `description`,
  `updated_at`; it carries no `default_params`. **A fresh clone does not have it.**
- **⚠ This still corrects a statement in the `tfm-project-facts` skill**, which says
  `agent_manifests/trse.json` is the only gitignored manifest and has no git history. **Noted for
  Alpha — not fixed here**, and it belongs in the owed v27 skill pass.

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
| **PWC / ZMQ as certifying comparators** | RANGE-MINER; PWC is a flag-selectable **non-certifying** diagnostic, PWC hybrid additionally quarantined |
| `RUNTIME_DATASET_PROVISIONING_CONTRACT.md`'s `expected_sha256` **as static config** | run-start freeze + fleet consistency (`TEAM_ALPHA_DATASET_LIFECYCLE_FINDINGS.md`) |
| scraper `--rewrite` mode | eliminated by owner decision (`TEAM_ALPHA_APPEND_ONLY_SIMPLIFICATION.md`) |
| **"RX 6600" on the rigs** | they are **RX 6600 XT**, gfx1032, 32 CUs, 8 per rig |
| `HYBRID_SKIP_BOUND_AUDIT.md:318` "semantics unspecified" | **FALSE** — `SKIP_SEMANTICS_SEARCH_v1.md`; `instructions.txt:1182-1183` |
| `TRSE_v1_15_SPEC.md` describing Rules B and C as **applied** | they are advisory by design (`TEAM_ALPHA_TRSE_FIX_PROPOSAL.md`) |
| `THRESHOLD_GOVERNANCE.md` synthetic-era defaults | `THRESHOLD_CALIBRATION_FINDINGS_S148.md` |
| `run_full_scoring.sh` | `run_step3_full_scoring.sh` |
| **`database_system.get_next_seed_start()` / `exhaustive_progress` as coverage authority** | **`utils/seed_coverage_ledger.py`** — Beta **deauthorized** the old tracker wholesale (rows retained, zero certified progress) |
| **`search_bounds.offset` as a live configuration key** | **removed outright** by the window-anchor ruling — no shim, no tombstone comment, no `offset → window_anchor` or `offset → generator_phase` mapping anywhere in the new path |
| **`[0,149]` as an anchor range**, anywhere it appears | **`control_anchor = [0, min(100, N_filtered − window_size)]`**. 100 is the historical **anchor** ceiling; 149 is the historical **record-envelope** ceiling. `PROPOSAL_..._v1_1.md` AC3 encodes the distinction as a permanent regression test |
| **`[0,100]` as a generator-phase law** | it was an **optimizer search bound**, not a mathematical law. It is neither inherited as a phase bound nor raised to ~7,000 |
| **"forward hybrid kernels receive no offset"** (too broad) | `lcg32_hybrid` and `pcg32_hybrid` **do** carry a phase argument; the four others do not. Use the per-variant capability matrix |
| **the FAIR-4/0 gate's old `50/50` literal** | the current form, **RATIFIED — do not revert.** An exact count was never a sound authorization proof |
| **`S185` as "the latest session number"** | SR-2 (§1.0): no new S-numbers; `S185` is only the highest *visible* number while ~20 SER8-only changelogs await backfill |
| **Run-1 / sweep-era pool-size and threshold figures in any `S1xx_CHAT_PROMPT.md` or `TODO_MASTER_*`** | current manifests and `BACKLOG.md` |

## 6.2 ★ Runtime-artifact figures — **DO NOT CARRY FORWARD, from any catalog**

The v1 catalog's "Runtime Data" table was retired in the 2026-08-03 pass because every figure in it
had failed against disk. **The general lesson stands and is the reason no such table appears here:
runtime-artifact sizes are not catalogue facts.** They were point-in-time filesystem measurements,
never defect claims, and they expire faster than any other class of statement in this file.
Two of the paths involved no longer exist as regular files at all: since D3.5 both
`bidirectional_survivors_all.npz` and `bidirectional_survivors_binary.npz` are **finalizer-owned
symlinks retargeted every run**, and as of the attempt-9 `.gitignore` block (§4.8) all three of those
paths plus `.s172_accumulator/` are ignored.

## 6.3 Superseded document versions (read the last one only)

- **`PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_0.md`** → **`_v1_1.md`**
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
- **`CLAUDE_CODE_INSTRUCTIONS_STAGING_DIR_FIX.md` Part B** → **`..._STAGING_DIR_PART_B.md`**
- **`CLAUDE_CODE_INSTRUCTIONS_2389B61_AUDIT_AND_WARMSTART_RESTORE.md` Part B** → **`..._WARMSTART_RESTORE_PART_B.md`**
- **`CLAUDE_CODE_REPORT_ATTACK_PLAN_FROM_PROCEDURES.md` parts C/D/E** → **`..._ATTACK_PLAN_BLACKBOX_REEVAL.md`** (Parts A/B stand)
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
- **Everything dated before 2026-08-11** predates the Gate-12 launch harness — the clean-tree gate,
  the parity gate, the sentinel gate, the liveness gate and the manual fleet launcher. A document
  that describes launching Gate 12 without them is describing a procedure that has since been
  rebuilt four times.
- **`docs/README.md` and `docs/proposals/README.md`** open with **"Seed Reconstruction"** /
  "Reverse-engineer PRNG behavior" framing. The project's naming rule is **functional mimicry, not
  seed recovery** (`PROPOSAL_Documentation_Paradigm_Correction_v1_2.md`). **Noted for Alpha — not
  fixed here.**

## 6.5 ★ Statements that were true on 2026-08-03 and are false now

**The previous catalog is the most likely source of each of these.** Re-read this list before citing
any status from it.

| said then | true now |
|---|---|
| Gate 12 pending / held / failing | **PASSED**, attempt 9, 2026-08-17, launch commit `e9ca800`, tag `gate12-passed-attempt9` |
| The Phase-7 soak is blocked | **UNBLOCKED.** Its remaining precondition — the field-6 repair — landed at `d8b21e3`. It is Michael-initiated and requires a live 25-daemon fleet |
| The MP-1 / drain-starvation defect is open | **CLOSED** (R-1 … R-4, Beta-certified). **Do not reopen R-1** |
| The window-anchor merge is next / a design exists | The design **did not exist**; a proposal phase was authorized, ran two rounds and **closed at v1.1**. The next artifact is **Implementation Brief I**, which does not exist yet |
| `BACKLOG.md` has 14 entries at `6892661` | **19 entries, 449 lines.** §19 (accumulator backup/recovery) was added 2026-08-17 — ruled real, ruled non-blocking, **unowned** |
| `tests/` has 33 entries | **51** |
| Step-1 manifest declares 25 `default_params`; 71 in total | **32** and **78** |
| The CA draw-procedures PDF exists only as a citation | It is committed at `docs/reference/CA_DAILY_SLP_DRAW_PROCEDURES_20210609.pdf` |
| `daily3_scraper.py` has never been in git history | It is **tracked** as of `334dacf`. Its *design document* still does not exist (§7 gap 5) |
| `bidirectional_survivors_*.npz` tracking is governed by `006623c` | **Superseded** by the attempt-9 `.gitignore` block (§4.8) |
| the skill (`TFM_PROJECT_FACTS_SKILL.md`) is current | **v26 (2026-08-17) is stale by its own most recent changelog** — a v27 pass is owed (§7 gap 10) |

---

# 7. KNOWN GAPS

**Every entry names the search that establishes it.** Nothing is listed here by assumption. All
absence statements are **repo-scoped** and carry the standing caveat that host state,
`~/dashboard_work` and the ser8 archives were not searched as content.

| # | gap | the search that establishes it |
|---|---|---|
| 1 | ★ **`docs/SESSION_CHANGELOG_20260819_S1.md` is untracked, unattributed, and violates the naming ruling.** It carries an **S-number**, which SR-2 (§1.0) forbids for any changelog written after 2026-08-18; its owner is unidentified; it is excluded from every commit pending Michael's disposition. **This catalog does not index it as governance and makes no claim about its contents.** | `git status --porcelain docs/` returns exactly one `??` line, this file. `git ls-files docs/ \| grep -c SESSION_CHANGELOG` = **181**; `ls docs/SESSION_CHANGELOG_*.md \| wc -l` = **182**. `SESSION_CHANGELOG_20260820_FIELD6_AND_WINDOW_ANCHOR_DESIGN.md` §5 records it as pending owner disposition. |
| 2 | **The `STEP_SCRIPTS[3]` ↔ `full_scoring.json` divergence has a brief but no committed answer.** | `/bin/grep -rl 'aggregate_scoring_results' docs/` returns exactly three files: this catalog, `PIPELINE_BEHAVIOUR_MODEL.md`, and `CLAUDE_CODE_INSTRUCTIONS_STEP3_SCRIPT_READ.md` — whose §Deliverable says *"your final message only. Create NO file"* because the tree had to stay clean for the soak. So the read was commissioned and the divergence is named, but **no committed artifact records what `run_step3_full_scoring.sh` does.** Chapter 4 remains unaudited and was not read line-by-line here. |
| 3 | **Step 0 has no documenting chapter.** | `ls docs/CHAPTER_*` enumerates all **46** chapter-family files in this pass (§2). There are chapters for steps 1–6 and for the registry, infrastructure, agents, features, WATCHER, feedback and diagnostics. None documents Step 0; TRSE's design authority is `TRSE_v1_15_SPEC.md` + `TRSE_INTEGRATION_PLAN_S121.md`, neither of which is a chapter. |
| 4 | **`agent_manifests/definitions.json` is untracked and gitignored — a fresh clone does not have it.** | `git ls-files agent_manifests/` returns 7 files; `ls agent_manifests/` returns 9; `.gitignore:41` (`*.json`) matches all of them. **Its role is not described in any of the 517 documents indexed here.** |
| 5 | **`daily3_scraper.py` has no design document.** | It is now **tracked** (`git log -1 -- daily3_scraper.py` → `334dacf`), so the audit surface `TEAM_ALPHA_SCRAPER_RECENT_SAFETY_NOTICE.md` describes has changed — but `DAILY3_CONSUMER_CONTRACT_v1.md` documents what consumers require of the *dataset* and `DATASET_PUBLICATION_SCHEMA_v1.md` freezes the *publication* schema. **Neither documents the producer**, and 6-P2 — the brief that would — is a REV4 **draft pending Beta**, with `TEAM_ALPHA_6P2_TRANSITION_RULING_REQUEST.md` still **AWAITING RULING**. |
| 6 | **Eleven chapters have never been audited against source.** | Enumerated in §2 from the file headers read in this pass; `BACKLOG.md` §1 names 3, 5, 6, 8 and 13 explicitly. The gap is wider: **Chapters 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 and 14 — roughly 11,000 lines — are unverified**, against a measured base rate of 9/41 (Chapter 1) and 17/55 (Chapter 3). |
| 7 | **Three binary files in `docs/` were not read.** | `find docs -name '*.pdf'` returns three; two (`Distributed_PRNG_Pipeline_*.pdf`) plus `TEAM_BETA_REVIEW_kfolds_S100.docx` were **not opened in this pass** and are marked UNVERIFIED in §1.4, §3.7 and §3.14. The third, `reference/CA_DAILY_SLP_DRAW_PROCEDURES_20210609.pdf`, **was not re-parsed here either** — its identity and content are carried from `CLAUDE_CODE_REPORT_ATTACK_PLAN_FROM_PROCEDURES.md`, which *was* read. **If the k-folds review contains a binding ruling, this catalog does not carry it.** |
| 8 | **Whether the S179-authorised KPI governance implementation has landed is still not established.** | `TB_RULING_S179_IMPLEMENTATION_AUTH.md` reads "APPROVED FOR IMPLEMENTATION WITH THREE BINDING CODE-LEVEL CONDITIONS". `ls docs/ \| grep -i kpi` returns eight files, the newest of which is the S178-era proposal set; `/bin/grep -rl 'KPI' docs/SESSION_CHANGELOG_2026081*.md docs/SESSION_CHANGELOG_2026082*.md` returns **nothing** — no changelog in the entire Gate-12 window mentions KPI governance. Tracing the code is **out of this catalog's scope**. **This is not a claim that it did not land**; it is a claim that fourteen days of committed session records do not mention it. |
| 9 | ★ **The Gate-12 brief corpus is not in the repository.** From 2026-08-11 onward the operative briefs and implementation reports (`CCODE_BRIEF_*_v1_0.md`, `ATTEMPT6_REMEDIATION_DESIGN.md`, `R1_DRAIN_REMEDY.md`, `MP1_DRAIN_ATTRIBUTION.md`, `H1H2_INSTRUMENTATION.md`, `D6_DRYRUN_PROCEDURE.md`, the attempt-4/5 forensics) were written to **`~/dashboard_work/`**. The changelogs name them precisely, so the trail is traversable, but **the artifacts are host state.** | `git ls-files \| grep -ci ccode_brief` = **0**. `ls ~/dashboard_work` shows the files exist on this host. **Host state is out of this catalog's scope by REV2 §5** — named here so the absence is not later reported as a missing governance record. |
| 10 | **The `tfm-project-facts` skill is stale, by its own record.** | `docs/TFM_PROJECT_FACTS_SKILL.md` header reads **v26, 2026-08-17**. `SESSION_CHANGELOG_20260820_FIELD6_AND_WINDOW_ANCHOR_DESIGN.md` §5 states plainly that v26 is now stale: three days of rulings, the field-6 repair, both standing rules and the window-anchor design are unrecorded, and **`G-PROD-SHAPE` was never in it.** A v27 pass is **owed and unwritten**. `BACKLOG.md` §17 additionally records that a skill revision lives in three places and committing updates one. |
| 11 | **The window-anchor Implementation Brief I does not exist.** | The design gate closed at `PROPOSAL_..._v1_1.md`; `TB_RULING_WINDOW_ANCHOR_PROPOSAL_V1_0.md` Q5 approves **two sequential briefs**, Brief II starting from the accepted Brief-I commit rather than independently from `e9ca800`. `ls docs/ \| grep -i 'BRIEF_I'` returns nothing and no `CLAUDE_CODE_INSTRUCTIONS_*WINDOW_ANCHOR*` file exists. **Expected next artifact, not a defect.** |
| 12 | **Fallback parity is UNRESOLVED and pass 2 has never run.** | `FALLBACK_PARITY_PASS1_20260815.md` states its own verdict: *"pass 1 RUN. Fallback parity remains UNRESOLVED and is NOT converted to PASS."* Pass 2 requires booting `.127`, and Zeus runs one OS at a time. `ls docs/FALLBACK_PARITY_*` returns only the pass-1 pair. |
| 13 | **The ser8 pre-repository archive has never been inventoried.** | `SER8_ARCHIVE_INVENTORY.md` completion sentinel: **`UNAVAILABLE`** — ser8 is reachable and up, but VM101 holds no credential ser8 accepts; no listing was obtained. This blocks both the ~20-changelog backfill that SR-2 defers the numbering decision to, and the `PIPELINE_BEHAVIOUR_MODEL.md` `INCOMPLETE` list it was meant to feed. |

---

# 8. VERIFICATION-INTEGRITY CONTROLS (VIR-1…6)

- **execution proof:** `docs/` contains **699 files** (688 top-level + 11 across four subdirectories,
  by `find docs -type f`). **517 indexed individually; 181 committed session changelogs summarised as
  a group; 1 untracked changelog excluded and recorded in §7 gap 1; 699 accounted for.** Every
  individually-indexed file was opened — heading structure plus opening section, per REV2 §2 — not
  summarised from its filename. **139 of them are new since the previous catalog** (123 non-changelog
  documents + 16 session changelogs), enumerated by
  `git diff --name-status 9e79a26..HEAD -- docs/`.
- **clean control:** `NOT_APPLICABLE` — this deliverable produces an index, not a detector.
- **fault-injection control:** `NOT_APPLICABLE` — same reason.
- **completion sentinel:** **`PASS`.** All 699 files are accounted for and all seven REV2 sections are
  delivered. Four binary files (the `.docx` and three PDFs) are indexed with their contents marked
  **UNVERIFIED** and recorded in §7 gap 7, rather than silently omitted.
- **unavailable-observer behaviour:** binary formats, gitignored files, host-state artifacts and the
  untracked changelog are **named and marked**, never inferred. Anything not opened is labelled.
- **audit claim scope:** **repo-scoped**, HEAD `0a4cef1`. This catalog indexes what is committed.
  **Host state (systemd, cron, deployed uncommitted files), `~/dashboard_work`, and the
  pre-repository ser8 archives are out of scope and are not implied.**
- **searched surfaces:** `docs/` (all 688 top-level files) · `docs/audit/S172_PHASE3/` ·
  `docs/proposals/` (incl. the empty `archived/`) · `docs/phase6_evidence/` · `docs/reference/` ·
  the repo-root file listing (1,026 files) · `agent_manifests/` (all 9; the 8 non-`.bak` parsed) ·
  `agents/watcher_agent.py:387-417` · `miner/` · `utils/` · `tests/` (51) · `scripts/` (20) ·
  `.gitignore` · `git diff 9e79a26..HEAD` over `docs/` and over the code tree · `git ls-files` ·
  `git status --porcelain`.
- **unavailable surfaces:** `agent_manifests/definitions.json` (gitignored; contents not catalogued) ·
  the `.docx` and three PDFs (binary, unread here) · `~/dashboard_work` (listed only, contents not
  read) · **host state and the ser8 pre-repository archives (not searched, out of scope).**

---

*Regenerated 2026-08-20 by Claude Code on VM101 under `docs/CLAUDE_CODE_INSTRUCTIONS_PROJECT_CATALOG_REGENERATION.md` REV2 and its 2026-08-20 delta. Read-only except this file: no code, config, manifest, chapter, gate or backlog entry was changed; nothing was committed or pushed; WATCHER and the pipeline were not run; no finding was fixed.*
