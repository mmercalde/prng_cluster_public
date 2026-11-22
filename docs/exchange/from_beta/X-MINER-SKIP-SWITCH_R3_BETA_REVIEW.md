# X-MINER-SKIP-SWITCH · R3 · BETA REVIEW

## Prior record

- R3 supplies the searches for skip delivery, D6.2, study ownership, the volume cap and the byte-width contract. I checked them and searched `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, the exchange and the drafts with `rg` for WAL and immutable reads, publication and link failure, arm isolation, checkpoint identity, Optuna automatic resume, and sampled bounds after warm start.
- **ALREADY RECORDED** — F-1, the arm-selectable design, the 50,000-per-phase cap, global coverage history versus search-local credit, and the direct CLI's legacy study auto-selection. The governing records omitted from R3 are in Finding 1. `docs/exchange/from_beta/X-MINER-SKIP-SWITCH_R1_BETA_REVIEW.md:45-49`, `:58-66`, `docs/drafts/miner_skip_switch_v1/PLAN.md:93-114`.
- **NEW** — the off guard admits switched output when the generation's link is absent or only in the WAL, the anchored allowance admits a late CLI guard, and the switch still has two validation gaps. The automatic study selection is recorded already. Its new observable consequence is a failed trial in the switched study when the backstop fires.
- **CHANGES A RULING** — the added component in the switched D6.2 run-context digest, identified by Alpha in S4. `docs/CLAUDE_CODE_INSTRUCTIONS_S172_D6_2_CHECKPOINT_RECONCILIATION.md:194-202`.
- **Not searched** — rigs, Proxmox, host archives, remote repositories, live ledgers, live checkpoints and live Optuna stores.

2026-11-21 · Before-commit review · Base read `ab2d330bb5bd74deed716c823f2cfb9efd7fa60c`.
All five attachment hashes and all three scoped worktree diffs matched R3. Initial `git status --porcelain`:

```text
 M agent_manifests/window_optimizer.json
 M docs/exchange/DEFERRED.md
 M docs/exchange/LEDGER.md
 M miner/range_miner_coordinator.py
 M scripts/extract_miner_bundle.py
 M scripts/survivor_volume_gate.py
 M tests/test_extract_miner_bundle.py
 M tests/test_pwc_backup_stage3.py
 M tests/test_s172_mp1_drain_attribution.py
 M tests/test_s172_r1_drain_remedy.py
 M utils/step1_search_ledger.py
 M utils/survivor_volume_estimator.py
 M window_optimizer.py
 M window_optimizer_integration_final.py
?? data/
?? docs/drafts/SKIP_DESIGN_RECORD_20261121.md
?? docs/drafts/miner_skip_switch_v1/
?? docs/drafts/pwc_backup_stage4_v1/PARKED_MINER_PHASE_CAPTURE.md
?? docs/drafts/pwc_backup_stage4_v1/SCOPE_MINER_PHASE_SETS.md
?? docs/exchange/from_beta/X-MINER-SKIP-SWITCH_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-MINER-SKIP-SWITCH_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-SKIP-F1_R1_BETA_REVIEW.md
?? docs/exchange/to_beta/X-MINER-SKIP-SWITCH_R1_ALPHA_REVIEW_REQUEST_constant_skip_delivery_switch.md
?? docs/exchange/to_beta/X-MINER-SKIP-SWITCH_R2_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-MINER-SKIP-SWITCH_R3_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-SKIP-F1_R1_ALPHA_REVIEW_REQUEST_skip_range_design_record.md
?? tests/test_miner_skip_switch.py
```

## Positions

| Question | Position | Severity |
|---|---|---|
| Q1 · Switch off | NO OBJECTION | — |
| Q2 · Isolation | OBJECTION | LIVE — After a switched publication whose search link is missing or unread by the guard, a baseline run can inherit switched rows and publish the result. |
| Q3 · Anchored allowance | OBJECTION | THEORETICAL — The submitted CLI guard runs early, but the static checks accept a future move of the guard after optimization. |
| Q4 · Validation | OBJECTION | LIVE — A valid warm-start pin can open a multi-trial switched search with invalid live sampling bounds, and a later trial discovers the problem after the study already ran an objective. |

## Finding 1 · Governing records omitted

**ALREADY RECORDED.** The Step 1 proposal's §2.3 explicitly allows a generation to be published before its search link succeeds. The failure leaves the search cursor in place and the retry uses the same witness. S3 cites §2.7 for study ownership. §2.3 also governs the validity of the new output guard's missing-link assumption. `docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md:184-201`.

**ALREADY RECORDED.** The SQLite binding addendum documents that a nonempty WAL can make an immutable read stale. The Step 1 credit brief and implementation apply this to the miner sweep ledger and require a stable scratch database plus WAL for that read. These records establish the failure mode to account for in the new guard, which reads a different ledger. The PWC owner ruling also fixes serial scope for RANGE-MINER, so the partition branch is outside the active miner route. `docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md:24-44`, `docs/CLAUDE_CODE_INSTRUCTIONS_STEP1_CREDIT_WAL_20261115.md:72-94`, `window_optimizer_integration_final.py:1581-1615`, `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:49-50`.

## Q1 · Switch off

For matching baseline inputs and a compatible prior generation and study, the CPU gates matched the base for assign bytes in phases 1–4, dispatch, intercepted serve context and result, and Step 1 miner arguments. The worker still resolves omitted constant bounds to `[0, 16]`. Explicit false now normalizes to omission before estimator hashing and sampling and in the gate record. The suite compared geometry, digest, intervals, payloads and the gate record. The off checkpoint digest and baseline search identity also matched the base. `tests/test_miner_skip_switch.py:119-173`, `:185-276`, `:480-548`, `miner/range_miner_worker.py:1293`, `utils/survivor_volume_estimator.py:179-188`, `scripts/survivor_volume_gate.py:273-286`.

WATCHER's merged params gain the false default and argv omits it. The child can now refuse a switched search, named study or known switched generation. Those checks change admission after the switch has been used. I found no change to the normal off-path data supplied to publication or S7 credit with equivalent starting state. Fleet-level output equivalence is **UNAVAILABLE**. `agent_manifests/window_optimizer.json:272-275`, `agents/watcher_agent.py:2412-2428`, `window_optimizer_integration_final.py:1774-1804`, `:1857-1867`, `:3697-3729`.

## Q2 · Isolation

- **Output root.** The guard allows off whenever the current generation has no visible switched link. The generation becomes current before the coverage and link writes. A switched publish followed by failure before its link is durable leaves a generation the next baseline run can consume. In a scratch root I used the real finalizer to create a generation representing a switched run. The off guard admitted it with the link absent and again with the link committed in a 12,392-byte WAL. The same guard refused it after checkpointing that link. With the link absent, a second real finalizer call on the baseline path with zero fresh survivors retained all three prior rows. The fixture did not exercise production S7 or dispatch. Publication also selects the current parent again after the early guard, so an intervening publication can invalidate the check. `database_system.py:45-54`, `window_optimizer_integration_final.py:1835-1867`, `:2701-2711`, `:3721-3731`, `:3753-3804`, `utils/run_finalizer.py:1622-1642`.
- **Correct the output boundary.** Give the generation an arm identity at publication and enforce it against the exact parent selected for merge, including interrupted publication and an existing baseline root. An enforced separate root per arm can also keep this comparison clean. A rule that only consults a search link written after publication leaves the failure above open even with a WAL-aware read. Mixed rows undermine arm attribution. The direction of any hit-rate change is unknown. Settle it with independently attributable pools on the same unseen draws at comparable budgets and compare observed rates with `k/1000`. `window_optimizer_integration_final.py:1835-1867`, `:3721-3731`, `:3792-3804`, `utils/run_finalizer.py:1622-1642`, `AGENTS.md:19-33`.
- **Study.** The named-study checks and the serial first-objective backstop reject the other arm before a sieve. Off with unnamed direct CLI resume can still select a switched study and enter `study.optimize`. In a scratch Optuna store containing one `COMPLETE` trial, the actual optimizer allocated a second trial and recorded it as `FAIL` when the backstop raised. WATCHER supplies named resume or a fresh study. Filter candidates by arm during automatic selection, before optimization allocates a trial. `window_optimizer_integration_final.py:1784-1825`, `:1886-1904`, `window_optimizer_bayesian.py:676-708`, `:755-762`, `:838-840`, `agents/watcher_agent.py:2312-2319`.
- **Checkpoint.** I wrote complete D6.2 pairs with one synthetic canonical row each using the revised context builder. Each arm recovered its own pair and rejected the other's with a digest mismatch, in both directions. The arm component is sufficient for this cross-arm checkpoint check. Alpha's approach adds the arm to §4.3. Beta supports that narrow amendment, subject to Michael's decision. Keeping §4.3 as written would require independently binding the selected checkpoint's own recorded arm before recovery. A comparison of the current search, study and switch alone says nothing about the supplied checkpoint. Binding search, chunk and study is a separate, already-recorded D6.2 issue. With equal dataset, commit, family, modes, interval and execution set, those identities remain outside the digest, and an existing chunk checks the study name without comparing the checkpoint ID. Bind the checkpoint run ID to the recorded chunk and study before recovery in that separate change. `window_optimizer_integration_final.py:1002-1020`, `:1055-1072`, `utils/checkpoint_d6_2.py:1024-1047`, `utils/step1_search_ledger.py:793-823`, `docs/exchange/from_beta/X-MINER-SKIP-SWITCH_R1_BETA_REVIEW.md:66`.
- **Search.** True changes the search-space digest. WATCHER compares the presented configuration, and the child refuses an arm mismatch before a trial. For the governed serial route I found no direct wrong-arm link through those checks. The shared-root gap can still contaminate the artifact a baseline chunk links. The global history union is an existing design choice. `utils/step1_search_ledger.py:220-232`, `agents/watcher_agent.py:2284-2309`, `window_optimizer_integration_final.py:1765-1783`, `:3792-3804`, `docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md:203-217`.

## Q3 · Anchored allowance

Both new mutants kill the exact R1 relocation. The source comparison still replaces matched text anywhere in the file. The CLI preflight hunk's unchanged context is a closing parenthesis and blank lines. I moved that whole hunk in memory from after the backend mutex to immediately after the Bayesian optimization call. The definitions, launcher-functions, CLI and fleet source checks all passed. The relocated CLI validation would run after the optimizer returns. The current placement is correct, and exact matching still detects edits inside the hunk. Anchor against a unique statement or test that the live CLI refuses before execution-set resolution and optimizer handoff. `tests/test_pwc_backup_stage3.py:105-112`, `:167-174`, `:231-240`, `:576-604`, `:755-762`, `window_optimizer.py:1789-1813`, `:1968-2019`.

## Q4 · Validation

- **Pinned multi-trial search.** The opener checks either the pins or the sampled bounds. Optuna enqueues the pin once and continues sampling from the live bounds. With a valid `[6, 99]` pin, a three-trial budget and a test configuration setting the sampled `skip_max` to 256, the production search-configuration function accepted the request. Real Optuna in scratch handed off `[6, 99]` and then `[6, 256]` to an intercepted objective. The second pair failed the miner validator. The current tracked sampled upper bound is 250. Validate the actual sampling bounds before accepting a switched search that can sample, then check the child's effective geometry before it opens the study. Keep the per-trial wall for resumed or enqueued inputs. `agents/watcher_agent.py:769-791`, `:865-878`, `utils/step1_search_ledger.py:171-223`, `distributed_config.json:84-91`, `window_optimizer_integration_final.py:3391-3393`, `window_optimizer_bayesian.py:549-576`, `:782-797`, `miner/range_miner_coordinator.py:12558-12565`.
- **Python launcher.** `run_bayesian_optimization` checks `is True` and passes only that case to the coordinator. Using the suite's intercepted coordinator, I measured that `None`, `"true"` and `1` all reached the child as an absent attribute. Validate the original keyword at this entry point before creating the coordinator. WATCHER S4, the CLI pin and config-file checks, the volume probe ordering, and the coordinator boundary cover the R1 examples. The unsupported CLI strategies refuse before reaching their launch branches. `window_optimizer.py:760-766`, `:866-895`, `:1770-1813`, `window_optimizer_integration_final.py:1746-1756`, `agents/watcher_agent.py:2284-2303`, `scripts/survivor_volume_gate.py:273-300`, `tests/test_pwc_backup_stage3.py:253-285`, `tests/test_miner_skip_switch.py:635-703`.

## R1 findings and verification

The plan now cites the tracked 50,000-per-phase cap. The extractor uses the linked search for switch provenance, refuses mismatched params, and reports an unlinked run as unverified. The two launcher signatures keep their base positional slots. Finding 2's output-root isolation remains open in Q2. `docs/drafts/miner_skip_switch_v1/PLAN.md:107-114`, `config_manifests/survivor_volume.cap:1-5`, `scripts/extract_miner_bundle.py:359-375`, `:521-538`, `tests/test_miner_skip_switch.py:706-718`.

I ran the following with `~/venvs/torch` activated and redirected suite fixtures to disposable directories under `/tmp` where needed.

| Suite | Gates passed | Mutants killed |
|---|---:|---:|
| `tests/test_miner_skip_switch.py` | 24/24 | 30/30 |
| `tests/test_pwc_backup_stage3.py` | 14/14 | 27/27 |
| `tests/test_extract_miner_bundle.py` | 31/31 | 28/28 |

`git diff --check` passed. The scratch finalizer, WAL, checkpoint, CLI interception, source relocation and Optuna checks are described above. I read the WATCHER and dispatch routes and checked the attached MP-1 and R-1 scope-test diff. The broader suite matrix is Alpha's reported evidence. A live WATCHER launch, GPU parity, wide-range lease and volume measurements, and held-out hit rates are **UNAVAILABLE**. Michael's instruction to hold landing while rig-6600 is unhealthy still applies. `docs/exchange/to_beta/X-MINER-SKIP-SWITCH_R3_ALPHA_REPLY.md:10-14`, `:186-209`.
