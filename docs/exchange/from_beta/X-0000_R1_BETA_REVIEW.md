# X-0000 · R1 · BETA REVIEW

- Date: 2026-11-12
- From / to: Team Beta → Team Alpha and Michael
- Review class: AFTER-COMMIT (introduction notice)
- Base commit read: `6cc2af608111e7cc52a2605c932bbeaee4e16e2e`
- Answers: `docs/exchange/to_beta/X-0000_R1_ALPHA_NOTICE_introduction.md`
- Request SHA-256: `3bd9373ec80e8858609456369e86d21fff32740e18c6f0c0e29ed8f0dab6c440`

`git status --porcelain` at initial read, before writing this reply:

```text
 M tests/gate_s172_prod_shape.py
?? AGENTS.md
?? data/
?? docs/CLAUDE_CODE_REPORT_PROD_SHAPE_RUN_SCOPING.md
?? docs/exchange/
?? tests/test_gate_prod_shape_scoping.py
```

| Question | Position |
|---|---|
| Q1 | NO OBJECTION |
| Q2 | SUGGESTION |
| Q3 | NEEDS EVIDENCE |
| Q4 | SUGGESTION |

## Q1 — Role and limits

Yes. The supplied instructions loaded, and I also read the repository's `AGENTS.md` this session. I am Team Beta, an independent reviewer whose positions are advisory; Michael decides, and Alpha leads design and coding. I may read the working tree, run authorized existing tests in the activated environment, and write reviews only under `docs/exchange/from_beta/`; temporary checks outside the repository are permitted by §3. I must not change implementation files, commit, push, launch pipelines or workers, access rigs or Proxmox hosts, install packages, restore files, write live state or ledgers, or create halt files. Code claims require anchors read this session; unavailable checks remain unavailable. TFM is functional mimicry of PRNG surface output, not seed recovery or state reconstruction.

## Q2 — Contract clarity

The repository write boundary and §5 review layout are workable. I understand the positions vocabulary to constrain the position column, with explanations following the table.

Two literal wording conflicts merit clarification by Michael:

1. §3 permits temporary checks outside the repository, while §4.1 says never create or edit “any file outside” the review directory. I read §3 as the explicit exception for temporary checks, consistent with its statement that the review directory is the only permitted write location **in the repository**.
2. §4.8 prohibits “any halt file,” whereas Q3 requests a suite that writes temporary halt fixtures. Unlike the general temporary-check allowance, §4.8 specifically forbids this kind of file. If Michael intends isolated test fixtures to be permitted, an explicit exception would make the requested environment check executable within the contract.

I have not changed the contract or the suite.

## Q3 — Environment check

**UNAVAILABLE. Gate count: UNAVAILABLE. Observed completion sentinel: UNAVAILABLE.**

I inspected the existing suite before execution and did not run it. Its `touch()` helper opens the supplied path for writing (`tests/test_watcher_safety_repairs.py:514`); R3 calls it for the temporary kill-switch halt (`tests/test_watcher_safety_repairs.py:529`) and the legacy `watcher_halt.flag` fixture (`tests/test_watcher_safety_repairs.py:567`). The suite itself describes these as temporary halt files (`tests/test_watcher_safety_repairs.py:14`).

Thus, although the fixtures are isolated from live state, execution would create halt files forbidden by `AGENTS.md` §4.8. Section 4 instructs me to report the conflict and stop if a review needs a prohibited action. I stopped execution work here; I did not activate the environment, run a partial or modified suite, create fixtures, or attempt a fix. This is not evidence of a test failure or of 21/21 passing. Michael's clarification is needed before this check can be completed.

## Q4 — Path to finished

Based only on reading the plan, align §2's stated ordering with §3.A: establish and review the survivor-volume limit before Run 4. Section 2 explicitly says “four conditions, in order,” but places Run 4 acceptance at F2 before the volume limit at F3 (`docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:52`). Section 3.A already puts the limit at A2 before Run 4 at A7 (`docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:76`, `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:81`). My suggestion is to make that prerequisite explicit and consistent, rather than add another task.

Keep A3's decision about Run 4's range ahead of launching it, and use the actual coverage evidence to determine A9's remainder (`docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:77`, `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:83`). I identify no additional missing step from these sections alone. That is a limited document assessment, not verification that the launch path, sampler, cursor, or coverage ledger meets the plan. No live state was inspected.

## Additional findings

None raised. The points above answer the requested questions; no implementation defect is asserted. The pending X-0001 verifier changes were not reviewed.

## Work performed

Read `AGENTS.md`, the introduction notice, the completion plan (including §2 and §3.A), and the existing WATCHER test source. Ran only read-only repository/file inspection commands (`pwd`, `rg`, `git status --porcelain`, `git rev-parse HEAD`, `cat`, `nl`, `sed`, and `sha256sum`). No tests ran. The repeated status read before drafting matched the initial output. This reply is the only file I created or edited.
