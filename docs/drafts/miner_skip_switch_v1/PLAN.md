# PLAN: RANGE-MINER constant-skip delivery as a switch (`deliver_skip_range`)

2026-11-21. Worktree `/home/michael/tfm_worktrees/pwc-offset`, detached at main `ab2d330`. No commits, no rig
contact, no live ledger. Anchors are at `ab2d330`.

**Decision being implemented:**
- **The owner's choice, 2026-11-21 09:54 PST, verbatim:** "sounds resonalble". It answered: build the RANGE-MINER
  skip-range delivery as a switch with `[0, 16]` as the baseline, as its own reviewed change after PWC Stage 4,
  settled by a paired hit-rate comparison on unseen draws.
- **Go-ahead, 16:54 PST:** "yes".
- **What that means:**
  - **default (off):** phases 1–2 keep the worker's `[0, 16]`, byte-identical to today;
  - **on:** phases 1–2 sieve the trial's own `[skip_min, skip_max]`.

The switch changes what phases 1–2 sieve. Its value is settled only by the paired hit-rate comparison against
`k/1000` on unseen draws (CLAUDE.md yardstick; Beta X-SKIP-F1 R1 Q2), never by seed recovery. This change builds the
switch; it does not run that comparison.

## Prior record

- **Searches:**
  - `ls docs/drafts | grep -i skip`;
  - `grep -rn "skip_range\|deliver_skip" miner/ window_optimizer*.py utils/step1_search_ledger.py utils/survivor_volume_estimator.py agent_manifests/window_optimizer.json`;
  - `grep -n "default_params\|not declared" agents/watcher_agent.py`;
  - the study and search ownership code (`utils/step1_search_ledger.py`, `window_optimizer_integration_final.py`
    S4/S9, `window_optimizer_bayesian.py` study creation and resume).
- **Found, and read:**

| Record | What it settles | Label |
|---|---|---|
| `docs/drafts/SKIP_DESIGN_RECORD_20261121.md` (untracked, this worktree) | F-1 traced in full. Every design source says phases 1–2 search the trial's range; RANGE-MINER never delivered it. §7 carries Beta's opinion and the owner's choice above | ALREADY RECORDED |
| `docs/exchange/from_beta/X-SKIP-F1_R1_BETA_REVIEW.md` (untracked here; relayed 09:33, ledger row restored) | **Q2:** a run-selectable option, `[0, 16]` as the baseline, record requested and effective pairs and the selected behaviour. **Q4:** validate `0 ≤ skip_min ≤ skip_max ≤ 255` at the common assignment boundary; fresh study identity; cost 5.53× hypotheses and 33.1× Java LCG advances at `[6, 99]`; lease and volume measurement first; Gate 12 parity changes with the coordinator | ALREADY RECORDED; this plan follows it |
| `docs/drafts/pwc_backup_stage3a_v1/SCOPE_MINER_SKIP_RANGE_FIX.md` | Where the range is dropped: `serve_trial` → `_dispatch_pending` → `build_stripe_assign_payload`, and the worker default at `miner/range_miner_worker.py:1293`. The minimal edit: the builder adds `skip_range` for the constant phases only; keep `_dispatch_pending`'s arity, because `tests/test_s172_attempt6_remediation.py` runs pinned `serve_trial` source with 14 positional arguments. It listed option (B), the flag | ALREADY RECORDED; this plan is its option (B) |
| `docs/TFM_SESSION_HANDOFF_20261114.md:76-80` | F-1 recorded as the owner's decision | ALREADY RECORDED |
| `docs/RESULT_PWC_BACKUP_STAGE4_20261121.md` | Stage 4 closed: both engines 0/0/13/2995 with the trial pinned to `[0, 16]` | ALREADY RECORDED (the baseline this switch must not move) |
| CLAUDE.md yardstick step 4 | "Propose any alternative as a flag-selectable configuration. The current algorithm stays as the baseline." | Standing rule |
| `docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md` §2.3 (`:184-201`; Beta X-MINER-SKIP-SWITCH R3 Finding 1) | "The link. … If it fails after publication, the failure surfaces, the search is not advanced, and the retry uses the same witness." A generation is current **before** its link exists. So the arm of a generation **cannot** be read from the link in time; this is why R3 binds the arm to the output root (§R3 below) | ALREADY RECORDED; missed in R1/R2 |
| `docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md` §A (`:24-44`), applied in `docs/CLAUDE_CODE_INSTRUCTIONS_STEP1_CREDIT_WAL_20261115.md` (`:72-94`) (Beta R3 Finding 1) | An `immutable=1` read is stale when `-wal` is non-empty. The R2 output-root guard read the run ledger `immutable` and so could miss a committed link. The R3 guard reads **no SQLite at all** (a file under the root), so no WAL can make it stale | ALREADY RECORDED; missed in R1/R2 |
| `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:49-50` (Beta R3 Finding 1) | n_parallel > 1 is switched off by ruling on the miner path, so the partition branch is outside the active route; "Please do whatever method is the quickest and least recoding" (21:35, on the miner skip fix) | ALREADY RECORDED; the reason R3 picks the smaller change |

- **Labels:** the switch, its route and its guards are **NEW**. Nothing **CHANGES A RULING**.
- **Not searched:** the host, the ser8 archives, the private repository, the rigs, the live ledger.

## 1. The route, hop by hop (smallest change that is explicit end to end)

| # | File (governed / pinned?) | Change | Off-state |
|---|---|---|---|
| 1 | `agent_manifests/window_optimizer.json` (**governed, owner hand-lands**) | `default_params.deliver_skip_range: false`. WATCHER merges only `--params` keys the manifest's `default_params` declares (`agents/watcher_agent.py:2024-2034`), so without this an operator's `deliver_skip_range: true` is silently dropped. Not routed: the fallback `--deliver-skip-range` is built from the key (`:2413-2415`) | `false` is omitted from argv (`:2418-2422`): **EXEC CMD unchanged** |
| 2 | `window_optimizer.py` (shared launch file) | CLI `--deliver-skip-range` (store_true); `run_bayesian_optimization(..., deliver_skip_range=False)`; sets `coordinator.deliver_skip_range = True` **only when on**; `parser.error` if given without `--use-range-miner`; one banner line when on | attribute not set: coordinator attributes identical |
| 3 | `window_optimizer_integration_final.py` (Step 1 launch path) | (a) the miner call to `run_trial_miner` passes `deliver_skip_range=True` **only when on**, through a conditional `**{...}`; (b) `optimize_window`: when on, refuse a run outside a Step 1 search, and refuse a search whose recorded search space does not carry the switch (and the reverse) | `run_trial_miner` arguments identical; no new check fires |
| 4 | `utils/step1_search_ledger.py` (Step 1 search record) | `presented_configuration`: when `final_params["deliver_skip_range"] is True`, `search_space["constant_skip_delivery"] = "trial_range"` | key absent: `search_space_digest` and search id identical |
| 5 | `miner/range_miner_coordinator.py` (**gate12-pinned, owner hand-lands, rig-deployed**) | (a) `run_trial_miner(..., deliver_skip_range=False)`: when on, validate the pair and record it for the run, before anything is created; the serve result carries a `constant_skip_delivery` record. (b) `_dispatch_pending`: hands `skip_range=` from that record (arity unchanged). (c) `build_stripe_assign_payload(..., skip_range=None)`: keyword-only. When not None it validates, and for **constant phases only** adds `"skip_range": [min, max]` | `None`: payload dict **byte-identical**; dispatch identical |
| 6 | `utils/survivor_volume_estimator.py` (not pinned) | when the geometry carries `deliver_skip_range: true`, the sampled payload gets the same `skip_range`, and the note says so | key absent: identical |

**Added while coding, kept to the same rules:**
- Row 7, `scripts/extract_miner_bundle.py`: its F-1 code reading now recognises the switch, the builder's
  keyword-only `skip_range=None`. The effective range comes from the run's own params, which B2 binds to the trial.
  Before this, it read the switched builder as "not known".
- The range check is written **inline** at its two sites (row 5), not as a new `def`. The MP-1 and R-1 AST scope
  proofs pin the coordinator's added-definition set exactly (`tests/test_s172_r1_drain_remedy.py:2165-2170` records
  the same choice). The three changed functions were already declared there.

**Not changed:**
- `miner/range_miner_worker.py`, which already reads `payload["skip_range"]` (`:1293`) into `BuildContext` →
  `_constant_prefix` → the two `int32` kernel scalars (`:259-271`, `:1385-1390`). Its gate12 digest is unchanged.
- The kernels, `prng_registry.py` and the hybrid ABI (no skip bound).
- PWC, which already sends the trial's range.
- The NPZ contract, whose `skip_min`/`skip_max` stamps already come from the trial context.
- `agents/watcher_agent.py`.

## 2. How the switch is set and seen

- **Set:** `--params '{..., "deliver_skip_range": true}'` through WATCHER, or `--deliver-skip-range` on the CLI. It
  is never defaulted on. It is a bool; any other type is refused at `run_trial_miner`.
- **Visible in the run record:**
  - the WATCHER `EXEC CMD` line carries `--deliver-skip-range`;
  - the Step 1 **search record** carries `search_space.constant_skip_delivery = "trial_range"`, durable in
    `prng_analysis.db`;
  - `window_optimizer.py` prints `[SKIP-DELIVERY] ON`;
  - each phase 1–2 stripe payload carries `skip_range`;
  - `run_trial_miner`'s result carries `constant_skip_delivery = {"requested": [a, b], "effective": [a, b],
    "phases": [1, 2]}`. When off, the key is absent; effective is `[0, 16]` by the worker default.

## 3. Validation

**When on**, at two places:
- `run_trial_miner` entry, before the staging dir, ledger or trial exist;
- `build_stripe_assign_payload`, the common assignment boundary Beta named.

**The rule:** `skip_min` and `skip_max` are plain ints (not bool), and `0 ≤ skip_min ≤ skip_max ≤ 255`. The bound
comes from `best_skips` being `uint8` (`miner/range_miner_worker.py:1382`), where 256 would serialize as 0 (Beta Q4).
A violation raises `MinerMetadataError`, before dispatch.

**When off,** nothing new is validated, because the baseline must be identical.

## 4. Fresh study identity, so switched-on trials never mix with baseline trials

- **A switched-on run must name a Step 1 search** (`--step1-search-id`), and that search's recorded search space must
  carry the switch. A baseline run presented to a switched-on search is refused, in both WATCHER (S4, through the
  digest) and the child (§1 row 3b).
- **The search owns its study** (S9: `utils/step1_search_ledger.py:743-760`; enforced before dispatch at
  `window_optimizer_integration_final.py:1715-1724`). A switched-on search therefore never resumes or feeds a baseline
  study, and the reverse.
- **No Optuna or `window_optimizer_bayesian.py` change is needed.**
- **Not covered:** a baseline run **outside** any search, using `--resume-study` with no `--study-name`, auto-selects
  the newest `optuna_studies/window_opt_*.db` (`window_optimizer_bayesian.py:676-708`). It could pick a switched-on
  search's study. That is the legacy auto-resume path, which WATCHER never uses (`agents/watcher_agent.py:2314-2318`).
  It is recorded here, not changed.

## 5. Risks at wide ranges (measure before any switched-on production run)

- **Work:** `[6, 99]` is 94 hypotheses per seed against 17, which is 5.53×. At `k=12` the Java LCG work is 33.1× (Beta
  Q4). `[0, 250]` is 251 hypotheses. Fleet timing is **UNAVAILABLE**.
- **Lease:** the 300 s compute lease is renewed by the worker heartbeat (30 s) and by each accepted sub-stripe. A
  longer phase 1–2 stripe can still complete. A 300 s renewal gap aborts the trial. **No lease or cap change is made
  here.** Measure the slowest rig's cold and warm sub-stripe times and renewal gaps first.
- **Survivor volume, 50,000-per-phase cap** (`config_manifests/survivor_volume.cap`, owner decision 2026-11-19; R1 said 25,000, the earlier cap — corrected per Beta R1 Finding 1): measured only at `[0, 16]`. With §1 row 6 the estimator samples the delivered range; the gate must be re-run with `deliver_skip_range: true` for each intended geometry before relying on the cap. A baseline gate result is not load evidence for a switched launch.
- **When on with a width above 17,** `run_trial_miner` prints a warning naming the hypothesis multiplier and these
  measurements. It does not refuse.

## 6. Proof that off is byte-identical

Each check compares against HEAD's own code, loaded from `git show ab2d330:<file>` beside the changed file:

1. **Payload bytes:** `build_stripe_assign_payload` for phases 1–4, with `skip_range` omitted, gives canonical bytes
   equal to HEAD's builder.
2. **Dispatch:** `_dispatch_pending` with no delivery record sends the same payloads (captured `StripeAssignMessage`).
3. **The serve context:** `run_trial_miner` with the default, through its `_serve` seam, builds the same serve
   context as HEAD's, and the coordinator carries no delivery record.
4. **The Step 1 call:** `run_bidirectional_test` hands `run_trial_miner` the same arguments
   (`tests/test_pwc_backup_stage3.py` G-MINER-JOBS).
5. **The coordinator:** `run_bayesian_optimization` builds a coordinator with the same attributes
   (G-MINER-SETTINGS).
6. **The search identity:** `presented_configuration` gives the same `search_space_digest` with the key `false` and
   absent.
7. **WATCHER argv:** unchanged with the manifest default `false`.
8. **Records and NPZ:** no record, manifest, assembly, canonical record or finalizer code reads the switch. The only
   GPU input is the payload, proven identical in 1–2. So the NPZ and records are identical by construction, shown
   statically with a grep over the readers.

**When on:**
- phase 1–2 payloads carry the trial's pair, and phases 3–4 are byte-identical to off;
- the worker's own parse (`tuple(payload.get("skip_range", [0, 16]))`) fed into `BuildContext`/`_constant_prefix`
  yields `ScalarArg(skip_min)`/`ScalarArg(skip_max)` equal to the pair. The CPU reference is the worker's argument
  builder; no GPU is used;
- the validation refusals;
- the study and search refusals.

**Mutants:**
- the key added when off;
- the key added on phase 3/4;
- the bounds swapped;
- the 255 bound dropped;
- delivery not handed from `_dispatch_pending`;
- the search-space key dropped;
- the child check dropped.

## 7. Guards that this change trips by design (done: exact-text allowance, 7 mutants, 14/14 and 25/25)

`tests/test_pwc_backup_stage3.py` pins the miner path, both launch files and the manifest to `b2efdef`.
G-MINER-SUBTREE, G-MINER-DEFS, G-WO-FUNCS and G-MANIFEST must be told about exactly these additions, by exact-AST /
exact-key allowance, with mutants proving that any other change still fails. The runtime gates (G-MINER-SETTINGS,
G-MINER-JOBS) must pass unchanged, because off is identical. This is a test edit in a landed suite; it is listed for
review. Other suites are run before and after; any change in result is reported.

## 8. Consequences the owner schedules (not blocking the code)

- **Rig redeploy:** `miner/range_miner_coordinator.py` is in the gate12 governed set
  (`scripts/gate12_parity_gate.py:174-185`). After landing, gate12 refuses every launch, **switch on or off**, until
  all three CT100s hold the new file.
- **Owner hand-landing:** `miner/` and `agent_manifests/` are hand-landed by the owner.
- **Review:** coordinator, launch path and manifest are before-commit review classes, so Beta review comes before
  landing. Not sent, as instructed.

**No item above needs an owner decision before coding:** the design follows the 09:54 choice and Beta's Q2/Q4. The
code follows.


## R2 — answers to Beta X-MINER-SKIP-SWITCH R1 (owner 18:30 PST: "Let's do all we can without the rigs")

- **One validator** (Q4): `miner/range_miner_coordinator.validate_skip_delivery(value, *pair, where)`. It is the
  only check of the switch, called by:
  - the CLI (`window_optimizer.py` main);
  - WATCHER params and search open (`utils/step1_search_ledger.presented_configuration`, reached by WATCHER's S4 and
    by search open);
  - the Step 1 child (`_skip_delivery_on`, the miner call and `optimize_window`);
  - the estimator (`validate_geometry`);
  - the coordinator (`run_trial_miner`, `build_stripe_assign_payload`).

  The switch must be exactly `true` or `false`, and absent means `false`. Anything else is refused. When on, the
  pair, the pins or the sampled-bound extremes must be plain ints in `0..255`, checked **before a search is
  opened**. The `--config-file` route refuses the switch.

  The validator is a new top-level def, so it is declared in the MP-1 and R-1 scope proofs' `DECLARED_ADDED` (SR-1).
  The coordinator imports no new module, so gate12's derived closure is unchanged.
- **One off state** (Q1): the estimator drops an explicit `false` from the validated geometry, and the gate drops it
  from its recorded params. Absent and `false` give the same digest, sample, payloads and evidence record.
- **Isolation** (Q2):
  - **Checkpoint:** the switch joins the D6.2 run-context components only when on, so the digest differs per arm.
    Recovery's context check (`utils/checkpoint_d6_2.py:1022-1046`) refuses the other arm's member. The off digest
    equals HEAD's.
  - **Study:** an existing study carries its arm as a user attribute, set on the first trial of a switched study.
    Before any study is opened, a switched run refuses: a named study that is untagged or baseline, and automatic
    resume selection. A baseline run refuses a named switched study.
  - **First-trial backstop:** before the sieve, it refuses an untagged study with earlier trials for an on run, and
    a tagged study for an off run. The off path writes nothing.
- **Output root** (finding 2): a run refuses to publish on top of the other arm's current generation. The arm is read
  from the generation's `step1_chunk_link` → search record, read-only and `immutable`.
- **Placement** (Q3): the Stage 3 allowance pins every hunk together with one unchanged context line on each side.
  A moved hunk fails, and Beta's relocation is mutants M-SW-8 and M-SW-9.
- **Extractor** (finding 3): B2b takes the switch from the linked search record. Params that disagree are refused;
  with no record, the effective range is left unverified.
- **Keyword-only** (finding 4): `deliver_skip_range` is keyword-only in `run_trial_miner` and
  `run_bayesian_optimization`; the positional slots equal HEAD's.

## R3 — fixes for Beta X-MINER-SKIP-SWITCH R3 (thread closed at R3; lead's instruction, owner decides on X-MINER-SKIP-SWITCH-2)

- **Output root, one root per arm** (Q2 LIVE). Beta offered two fixes: stamp each generation with its arm at
  publication and check it against the exact parent chosen for the merge, or enforce a separate root per arm.
  **This plan takes the second, because it is the smaller change.**
  - Stamping needs `utils/run_finalizer.py`, the certified publication path PWC shares, to write an arm and to
    check its parent. Every finalizer suite and the extractor would change.
  - One root per arm touches only the Step 1 child. A switched run writes
    `<root>/.s172_accumulator/skip_delivery_arm` (`trial_range`) at preflight, before its first trial. Every
    generation ever published in that root is therefore switched, including one whose link was never written
    (§2.3 above).
  - The marker is ignored by git (`.gitignore:72`). The finalizer never reads it and accepts it: two real
    generations were published into a marked root.
  - Rules (`_step1_skip_delivery_root_arm`):
    - off with a marker is refused;
    - on with an unmarked root that has a `current` generation is refused;
    - on with an empty root marks it, at preflight only;
    - at the pre-publication recheck, an unmarked root is refused.
  - The recheck is the statement immediately before `_finalize_run_d3_5`, against the root whose current
    generation the finalizer merges.
  - No SQLite is read, so the binding addendum's stale-WAL case cannot arise. The R2 ledger-reading guard is
    removed.
  - **Limit:** a root used by a switched run before this change carries no marker. No switched run has
    happened, because the switch has never landed.
- **Automatic study selection** (Q2 LIVE). `window_optimizer_bayesian.py`'s unnamed `--resume-study` loop skips a
  candidate study tagged `constant_skip_delivery`, before any trial is allocated. A named study is unaffected; the
  named-study checks already cover it. A switched run cannot reach this loop, because it is refused earlier.
- **Sampling bounds** (Q4 LIVE).
  - **Search open:** a switched search that can sample (unpinned, more than one trial, or no stated budget)
    validates the sampling bounds as well as the pins.
  - **Child:** before any study opens, it validates `SearchBounds.from_config()`'s `min_skip_min`/`max_skip_max`.
    The tracked bound is 250.
  - The per-trial check at the coordinator stays.
- **Launcher keyword** (Q4 LIVE). `run_bayesian_optimization` calls the one validator on `deliver_skip_range` as its
  first statement, before any coordinator exists. `None`, `"true"`, `1`, `0` and `""` are refused.
- **Anchors** (Q3 THEORETICAL). In the Stage 3 allowance, each hunk's context now grows on each side until it holds a
  line that occurs exactly once in the HEAD file. A blank line or a `)` can no longer anchor a hunk. Beta's
  relocation (the CLI preflight block moved after the Bayesian call) is mutant M-SW-10.
- **Unchanged:**
  - the D6.2 §4.3 component question, which is the owner's call; Beta supports the narrow amendment;
  - the separate D6.2 search/chunk/study binding, ALREADY RECORDED.

## SW2 — close-out of Beta X-MINER-SKIP-SWITCH-2 R1 (Q2, Q4 THEORETICAL; Alpha lead's instruction)

- **The isolation claim, as stated** (Q2; the owner path was to state the limit, with no lock):
  - Switched-on and baseline trials are isolated only with **a separate output root per arm and one Step 1 publisher
    at a time**, as Stage 4 ran (`docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:91-115`: a distinct checkout
    and working directory per arm).
  - Two publishers overlapping in one initially unmarked root are **not** isolated. The finalizer picks its merge
    parent from `current` (D3.5 REV3.1 §8b, `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3_5.md:316-350`),
    independently of the marker check. Beta reproduced this.
  - **Any backup or restore of an output root must carry `.s172_accumulator/skip_delivery_arm`**; a root restored
    without it admits a baseline run. This also belongs in any future accumulator backup policy (BACKLOG §19).
  - The guard's docstring, both refusals and the banner a switched run prints when it marks its root all say this.
    No root lock is built.
- **One sampling rule** (Q4), in `utils/step1_search_ledger.py`, used by both search open and the Step 1 child:
  - **`skip_delivery_can_sample(pinned, trials, resume_study)`:** only a fresh run with all seven pins and one trial
    cannot sample (TB_RULING_RUN4 `:121`). A resumed study enqueues no pin, so it samples.
  - **`validate_skip_sampling_domains`:** both full ranges go through the one validator, plus
    `skip_min.max ≤ skip_max.max`. Optuna draws `skip_max` from `[max(skip_min, skip_max.min), skip_max.max]`, so
    that ordering keeps every draw non-empty and every pair legal.
  - The child runs the check **before** the named-study check loads any store. It returns the checked
    `SearchBounds` object, which is the one handed to the sampler. With the switch off, it returns `None` and the
    bounds load exactly as before.
  - The per-trial check at the coordinator stays.
- **Regressions:**
  - `G-SW2-Q4-DOMAINS`: 72 cases (6 bound sets × pinned × trials × resume) in which search open and the child must
    give the same verdict, including Beta's `skip_min.max=256` / `skip_max.max=250`.
  - `G-SW2-Q2-SCOPE`: Beta's overlap, run with the real finalizer, as the stated limit; the scope text; the serial
    orders refused.
  - Mutants MS34–MS35 (repointed) and MS37–MS42.

## Owner rulings, 2026-11-22

- **08:34 PST, D6.2 §4.3 amendment APPROVED.** `constant_skip_delivery` joins the run-context digest components when
  the switch is on; with it off, the component set and digest are unchanged. Owner, verbatim: "i approve as long as
  it doesnt break,confuse or in any other way slow down development". This was a ruling change, so it needed the
  owner: `docs/CLAUDE_CODE_INSTRUCTIONS_S172_D6_2_CHECKPOINT_RECONCILIATION.md:194-199`. Beta supported the narrow
  amendment in X-MINER-SKIP-SWITCH R3 Q2.
- **08:37 PST, one more Beta round on the SW2 close-out before landing.** Owner, verbatim: "sure". The round is
  X-MINER-SKIP-SWITCH-2 R2. Its scope is only the Q2 scope text and the Q4 single rule.
- **Rig status, as reported by the Alpha lead:** the rig-6600 cause is closed (failed PSU breakout board, commit
  `d58e120`, "rig-6600 resets 2026-11-21 closed: failed PSU breakout board (owner); all three rigs up on Proxmox").
  All three rigs are up on Proxmox and idle. The commit exists in this repository's history; **the rig state itself
  was not checked from this session** (no rig access) [UNVERIFIED].

## SW3 — fix for Beta X-MINER-SKIP-SWITCH-2 R2 Q4 (LIVE); owner 2026-11-22 09:03 PST: "sounds like the fix is required- lets get it doen and on to testing. please"

- **Exact pin.** A switched, fresh, one-trial, fully pinned run consumes its pin exactly:
  - The child (`_step1_skip_delivery_exact_pin`, under the shared `skip_delivery_can_sample` rule and the one
    validator) puts `skip_delivery_exact_pin` into `trial_history_context`.
  - When that key is present, the study body draws single-point distributions, so no sampled skip range is built.
    That closes the `[251, 255]` case and meets TB_RULING_RUN4 `:120`, "each actual enqueued value equals the
    requested value exactly".
  - With the key absent, which is every off run, the original calls run.
  - An exact pin past trial 0 fails closed.
- **Resume flag.** Search open uses WATCHER's effective flag, `bool(study_name)`, taken from the "Study continuity"
  block (`PROPOSAL_S145_R1_Progressive_Empirical_Sweep.md:124-130`). `agents/watcher_agent.py` is not edited, and
  `G-SW3-RESUME-RULE` pins its rule verbatim.
- **Tests:** `G-SW3-Q4-EXACT-PIN` (real Optuna study body), `G-SW3-RESUME-RULE` and `G-SW3-Q2-OVERLAP` (both orders,
  both finalizer calls); MS43–MS47. Suite totals: 32/32 gates, 48/48 mutants.
