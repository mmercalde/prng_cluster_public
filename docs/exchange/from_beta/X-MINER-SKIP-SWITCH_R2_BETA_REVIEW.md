# X-MINER-SKIP-SWITCH · R2 · BETA REVIEW

- **Date** 2026-11-21 (PST)
- **Base commit read** `ab2d330bb5bd74deed716c823f2cfb9efd7fa60c`
- **Worktree** `/home/michael/tfm_worktrees/pwc-offset`
- **`git status --porcelain` at read time**

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
?? docs/exchange/from_beta/X-SKIP-F1_R1_BETA_REVIEW.md
?? docs/exchange/to_beta/X-MINER-SKIP-SWITCH_R1_ALPHA_REVIEW_REQUEST_constant_skip_delivery_switch.md
?? docs/exchange/to_beta/X-MINER-SKIP-SWITCH_R2_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-SKIP-F1_R1_ALPHA_REVIEW_REQUEST_skip_range_design_record.md
?? tests/test_miner_skip_switch.py
```

| Question | Position | Severity |
| --- | --- | --- |
| Q1. Switch off | NEEDS EVIDENCE | — |
| Q2. Isolation | NEEDS EVIDENCE | — |
| Q3. Anchored allowance | NEEDS EVIDENCE | — |
| Q4. Validation | NEEDS EVIDENCE | — |

All four positions follow `AGENTS.md:40–45` because Alpha’s `## Prior record` lists documents consulted and omits the searches the author ran (`docs/exchange/to_beta/X-MINER-SKIP-SWITCH_R2_ALPHA_REPLY.md:15–21`).

Beta checked the request against the prior-record rule and captured the Git status and HEAD. Code review and test execution did not begin.
