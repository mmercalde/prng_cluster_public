# READY TO LAND — RANGE-MINER constant-skip delivery switch (`deliver_skip_range`)

**X-MINER-SKIP-SWITCH-2 R3 IS ACCEPTED by the Alpha lead** (2026-11-22).
- **Beta's positions:** three NO OBJECTION and one test SUGGESTION. Beta matched the four attachment hashes and both
  tracked diffs byte for byte.
- **No LIVE objection holds the commit.**
- **Relay boundary flag, recorded as it happened.** The relay returned `BOUNDARY VIOLATION` (exit 3;
  `docs/exchange/LEDGER.md`, row X-MINER-SKIP-SWITCH-2 R3, 09:14 PST). It listed two changes outside `from_beta/`:
  `docs/drafts/miner_skip_switch_v1/PLAN.md` and this file.
  - **Cause:** Alpha edited the plan and this landing page while the relay was running. The edits were made on the
    Alpha lead's instruction to refresh the landing page while Beta read.
  - **No code or test file changed during the run.**
  - Beta's review cites `PLAN.md:302-316`, the SW3 section written during the run. It describes the fix under
    review.
  - Under `docs/exchange/RELAY_PROTOCOL.md` §4, a boundary verdict is Michael's decision. The lead accepts the
    review.

**Standing condition: do not land while rig-6600 is unhealthy** (owner, 2026-11-21).
- **Status 2026-11-22, as reported by the Alpha lead:** cause closed (failed PSU breakout board, commit `d58e120`);
  all three rigs up on Proxmox and idle.
- The commit is in history. **The rig state was not checked from this session** [UNVERIFIED].
- Whether the condition is met is the owner's judgement.

Nothing here is committed; the owner commits and dual-pushes.

- **Base:** `ab2d330bb5bd74deed716c823f2cfb9efd7fa60c`.
- **Worktree:** `/home/michael/tfm_worktrees/pwc-offset` (detached).
- **Plan:** `docs/drafts/miner_skip_switch_v1/PLAN.md`.
- **Default:** off. A run without `--deliver-skip-range` behaves byte-for-byte as `ab2d330`. Evidence:
  - the `G-OFF-*`, `G-Q1-*`, `G-SW2-*` and `G-SW3-Q4-EXACT-PIN` off-vs-HEAD gates;
  - Beta's Q1 NO OBJECTION in X-MINER-SKIP-SWITCH R3 and X-MINER-SKIP-SWITCH-2 R1 and R2.

## 1. Owner decisions (all taken)

1. **D6.2 §4.3 amendment: APPROVED**, 2026-11-22 08:34 PST. Owner's words: "i approve as long as it doesnt
   break,confuse or in any other way slow down development".
   - `constant_skip_delivery` joins the run-context digest components only when the switch is on.
   - The rule it amends: `docs/CLAUDE_CODE_INSTRUCTIONS_S172_D6_2_CHECKPOINT_RECONCILIATION.md:194-199`.
2. **One Beta round on the SW2 close-out:** "sure", 08:37. That round became X-MINER-SKIP-SWITCH-2 R2, which raised
   one LIVE objection (Q4).
3. **The fix and its round:** "sounds like the fix is required- lets get it doen and on to testing. please", 09:03.
   That is X-MINER-SKIP-SWITCH-2 R3.

## 2. Files and sha256 (worktree state when this page was written, after the R3 fix)

**`main` has moved since `ab2d330`.** It is now `d58e120`, three docs-only commits ahead: `9034af8`, `7125511`,
`d58e120`. None touches a code or test file in this change. **`docs/exchange/LEDGER.md` changed on both sides**, so
landing must merge it: keep main's rows plus this worktree's relay rows. `DEFERRED.md` changed only here.

**Check before landing:** `git diff --name-only ab2d330` must list exactly the tracked files in 2a and 2b plus the
two exchange ledgers. Re-hash them and compare with this table. `agents/` is **not** in the change
(`git diff --quiet ab2d330 -- agents/` exits 0).

### 2a. The owner lands these by hand

`scripts/owner_land.py:85-87` refuses anything under `miner/` or `agent_manifests/`.

| File | sha256 |
|---|---|
| `miner/range_miner_coordinator.py` | `b57f3c5a7eeaa10204347668dfb8a848d3fa01a9d44bfff437b317f7912d81fa` |
| `agent_manifests/window_optimizer.json` | `1591e9964dabcad53242e130b9e901271e0e038afb233fb8a889fb721bc90894` |

### 2b. Through `scripts/owner_land.py`

| File | sha256 |
|---|---|
| `window_optimizer.py` | `3d76916291818b919552401a324a67a759bf8aa0c3903a6c645c94e269feab27` |
| `window_optimizer_integration_final.py` | `5ec53da4f887f1a6640982ea4f81b137cd53567a1fbed07329d01260a8e2f51b` |
| `window_optimizer_bayesian.py` | `3cf0f0f1313847f78bae4613e2f21899b4a9820369e7b5cc170e6f978fa4dcbb` |
| `utils/step1_search_ledger.py` | `3d7e8f63ce558c0fd4074f25805982986c5727bf675900115a0d2fabfdc70d2c` |
| `utils/survivor_volume_estimator.py` | `2c0a45e78409f3961145851607a831a4af5d736f521c8acfebf0b623c7de9281` |
| `scripts/survivor_volume_gate.py` | `495862b3e4c7eeb3850a67aad101185115524118aa149fff744f78336c43cff6` |
| `scripts/extract_miner_bundle.py` | `e4ebaf07629a86bbe868ad3f6405a0f741e54fabfff0194ddeb3eca197608050` |
| `tests/test_miner_skip_switch.py` (new, untracked) | `8f66fe3e11ba23de2edb0f8bf5f4148651c5ebd66191c9af15b4684abd65463b` |
| `tests/test_pwc_backup_stage3.py` | `138b2978691e115441a7ebf65218ada53e180ee88f4334663653252a72bbb03e` |
| `tests/test_extract_miner_bundle.py` | `3ba0a91e586c2143f278813affeddc1da69c7d95daf1d0097c9fabbdd70dbfa1` |
| `tests/test_s172_mp1_drain_attribution.py` | `364b1c1afba85484f7a4011bca2de78217036977a3f632e6acd7a6cf0f082219` |
| `tests/test_s172_r1_drain_remedy.py` | `885d2a786c9035ade07f4a22817e6ba130a7aabbdda683998b7e84fa9042274e` |
| `docs/drafts/miner_skip_switch_v1/` (PLAN.md, this file, the diffs) | hash at landing |
| `docs/drafts/SKIP_DESIGN_RECORD_20261121.md` | hash at landing |
| `docs/exchange/LEDGER.md`, `docs/exchange/DEFERRED.md` | written by the relay; hash at landing |
| `docs/exchange/to_beta/` and `docs/exchange/from_beta/`: X-SKIP-F1 R1, X-MINER-SKIP-SWITCH R1–R3, X-MINER-SKIP-SWITCH-2 R1–R3 | from_beta files are Beta's; never edited |

**Diffs against `ab2d330`** (in `docs/drafts/miner_skip_switch_v1/`):

| File | sha256 | What it holds |
|---|---|---|
| `code_sw3.diff` | `147a9a7b…` | the source |
| `stage3_allowance_sw3.diff` | `b495d39f…` | the Stage 3 suite |
| `tests_r3.diff` | `d2c080d2…` | the MP-1/R-1 `DECLARED_ADDED` and extractor tests |
| `sw3_delta_from_sw2r2.diff` | `248009ff…` | this round only |

## 3. Rig redeploy and gate12 parity, required on all three rigs

`miner/range_miner_coordinator.py` is in gate12's governed set (`scripts/gate12_parity_gate.py:174-185`). **After
landing, gate12 refuses every RANGE-MINER launch, with the switch on or off, until all three rigs hold the new
file.** The worker and the other nine governed files are unchanged.

1. **Boot state.** All three are reported up on Proxmox, so the workload endpoints are CT100 `192.168.3.122`
   (rrig6600), `.156` (rrig6600b) and `.164` (rrig6600c) (CLAUDE.md §3). The fleet's declared default profile is
   `proxmox` (`rig_profiles_config.json`). Confirm with `ssh -o BatchMode=yes michael@<ip> hostname` before
   copying. **rig-6600 must be healthy.**
2. **Export.** On VM 101, from the landed commit:
   `git archive <landed-sha> miner/range_miner_coordinator.py | tar -x -C <scratch-export-dir>`.
   Never `git pull` on a rig. Deploy is an export plus `scp`, and parity is checked by digest (`docs/PHASE6_PREREQS.md`, as cited in
   `docs/CLAUDE_CODE_REPORT_RUN4_LAUNCH_PREP.md:97`).
3. **Copy.** For each CT100: `ssh michael@<ip> mkdir -p /home/michael/distributed_prng_analysis/miner`, then
   `scp <scratch-export-dir>/miner/range_miner_coordinator.py michael@<ip>:/home/michael/distributed_prng_analysis/miner/range_miner_coordinator.py`.
   Use absolute paths; the deploy path is each node's `script_path` in `distributed_config.json`.
4. **Parity.** From the clean landed tree on VM 101:
   `python3 scripts/gate12_parity_gate.py --rig-profile proxmox --evidence-json <path under data/>`.
   It must report every governed file matching on **all three rigs**: 10 files × 3 rigs = **30/30**, the shape
   recorded at `48a8705` (`docs/SESSION_CHANGELOG_20260822_BRIEF_I_PRODUCTION_SHAPE_REPAIR.md:16-18`). Keep the
   evidence bundle.
5. **First launch:** only after step 4 is green, and only when the owner initiates it (`nohup`, never `tmux`).
   - **A switched-on run** needs a Step 1 search opened with `deliver_skip_range: true`, and its own output root:
     its own checkout and working directory (§4).
   - **Before any switched-on production run**, run the survivor-volume gate for that geometry (§5).

## 4. The isolation claim, and its limits

- **Scope.** A switched run and a baseline run are isolated only when:
  - each arm has its own output root, run from its **own checkout root**, so that Optuna storage follows the
    checkout (owner ruling 2026-11-14, `docs/CLAUDE_CODE_REPORT_STEP1_SEARCH_RECORD_STAGE1.md:82-87`);
  - **one Step 1 publisher runs at a time**, as Stage 4 ran
    (`docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:91-115`).
- **Known gap.** Two publishers overlapping in one unmarked root are not isolated: the second inherits the first's
  rows (D3.5 §8b). Beta reproduced this, and `G-SW3-Q2-OVERLAP` asserts it. No root lock is built.
- **Marker.** A switched run marks its root with `.s172_accumulator/skip_delivery_arm`. **Any backup or restore of
  that root must carry the marker**, because a root restored without it admits a baseline run. This belongs in any
  future accumulator backup policy (BACKLOG §19).
- **Also in place:**
  - the search identity and S9 study ownership;
  - the Optuna study tag, and auto-select skipping tagged studies;
  - the D6.2 digest component (§1);
  - the extractor taking the switch from the linked search record.

## 5. Not verified

- **No rig run with the switch on.** There has been no GPU run and no WATCHER end-to-end launch with it. All
  evidence is CPU-only. The study-body proof uses a stub sieve objective.
- **Wide-range lease and survivor-cap behaviour are unmeasured.** At `[6, 99]`:
  - a seed has 94 skip hypotheses against 17 (5.53×), and 33.1× the Java LCG advances (X-SKIP-F1 R1 Q4);
  - lease renewal gaps and phase 1–2 time are unmeasured;
  - survivor volume against the 50,000-per-phase cap (`config_manifests/survivor_volume.cap`) is unmeasured. The
    estimator prints this warning when the switch is on.
- **No hit-rate comparison between the arms.** That needs independently attributable pools on the same unseen draws
  at comparable budgets, compared with `k/1000`.
- **No full cross-arm checkpoint pair from a real run.** Beta tested synthetic D6.2 pairs in both directions.
- **Open, ALREADY RECORDED:** binding a checkpoint to its search, chunk and study is a separate D6.2 issue.
- **Not taken:** Beta's X-MINER-SKIP-SWITCH-2 R2 Q2 SUGGESTION, which is wording only. It asks to qualify the
  docstring's opening "every generation is switched" and to name the own-checkout rule in the docstring. §4 above
  states both.

## 6. Tests (CPU, worktree, one at a time)

| Suite | Gates | Mutants |
|---|---|---|
| `tests/test_miner_skip_switch.py` | 32/32 | 48/48 |
| `tests/test_pwc_backup_stage3.py` | 14/14 | 28/28 |
| `tests/test_extract_miner_bundle.py` | 31/31 | 28/28 |

- **Affected suites after the R3 fix:** 20. Every exit code equals the previous run's and HEAD's.
  - Failure lines differ only in Gate 22's changed-file list (it reds on any changed `.py`, by design), set and
    hash ordering, generated run ids and timestamps.
  - After the whitespace strip, the readers of `window_optimizer_bayesian.py` were re-run with the same results.
  - Logs: `logs/switch_suites/` (ignored).
- **Red at HEAD too**, so not caused by this change:
  - `test_chapter1_p0_corrections`;
  - `test_s172_d6_2_checkpoint_reconciliation`;
  - `test_s172_miner_seed_start`;
  - `test_s172_phase5_d0`;
  - `test_s172_phase5_d6_*`;
  - `test_s172_phase6_p05_dataset_authority`;
  - `test_s172_threshold_propagation`;
  - `test_seed_domain_cursor_amendment`;
  - `test_step1_search_record_stage1`;
  - `test_s172_run4_routing_patch`;
  - Gate 22 in `test_s172_phase4_coordinator`, `test_s172_phase5_d1_workflow` and
    `test_s172_phase5_d2_directional_uniqueness`.

## 7. Beta verdict history (`docs/exchange/LEDGER.md`; reviews in `docs/exchange/from_beta/`)

| Thread · round | Time (PST) | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| X-SKIP-F1 R1 (design record) | 11-21 09:33 | NO OBJECTION | SUGGESTION (a selectable option, `[0, 16]` as the baseline) | SUGGESTION (after Stage 4) | SUGGESTION (costs, the 255 bound, a fresh study) |
| X-MINER-SKIP-SWITCH R1 | 11-21 18:17 | OBJECTION LIVE | OBJECTION LIVE | OBJECTION THEORETICAL | OBJECTION LIVE |
| X-MINER-SKIP-SWITCH R2 | 11-21 19:02 | NEEDS EVIDENCE | NEEDS EVIDENCE | NEEDS EVIDENCE | NEEDS EVIDENCE (Prior record listed documents, not searches) |
| X-MINER-SKIP-SWITCH R3 (closed) | 11-21 19:35 | NO OBJECTION | OBJECTION LIVE (output root, unnamed resume) | OBJECTION THEORETICAL (anchors) | OBJECTION LIVE (pinned sampling, launcher keyword) |
| X-MINER-SKIP-SWITCH-2 R1 | 11-21 21:16 | NO OBJECTION | OBJECTION THEORETICAL (overlapping publishers) | NO OBJECTION (anchors) | OBJECTION THEORETICAL (full domains, rule agreement) |
| X-MINER-SKIP-SWITCH-2 R2 | 11-22 | NO OBJECTION | SUGGESTION (scope wording) | — (Tests: SUGGESTION) | OBJECTION LIVE (a one-trial pin still built sampled ranges, so `[251, 255]` failed in Optuna) · THEORETICAL (WATCHER resume flag) |
| X-MINER-SKIP-SWITCH-2 R3 | 11-22 09:14 | NO OBJECTION | — (Q4 exact pin: NO OBJECTION) | — (Q4 resume flag: NO OBJECTION) | Tests: SUGGESTION · relay verdict **BOUNDARY VIOLATION** (draft edits during the run; accepted by the lead, see top) |

- The THEORETICAL items are in `docs/exchange/DEFERRED.md`.
- R3 answers R2's Q4 LIVE and THEORETICAL items and both test suggestions.
- **R3 Tests SUGGESTION, taken (test-only):**
  - the exact-pin gate loads the tracked `distributed_config.json` by explicit path;
  - it asserts each switched trial ends COMPLETE with single-point skip distributions, for `[251, 255]`, `[6, 99]`,
    `[0, 255]`, `[0, 0]` and `[255, 255]`;
  - it adds R2's inverted `skip_min` `[5, 3]` case: the exact pin completes, and without the key the range is
    empty;
  - off-vs-HEAD now also requires both runs to finish without error;
  - the overlap gate compares every array in both NPZ files of the two generations.
  - Suite result: 32/32 gates, 48/48 mutants.
- **R3 Tests SUGGESTION, deferred:** Beta's extra check, that WATCHER serializes the resume flag into the child's
  arguments. It would mean executing code extracted from `agents/watcher_agent.py` (owner-landed), so it is not
  small. It is recorded here and not in `docs/exchange/DEFERRED.md`, because that file is written only by
  `scripts/beta_relay.py` and holds THEORETICAL objections, not suggestions. `G-SW3-RESUME-RULE` still pins
  WATCHER's continuity block verbatim.
