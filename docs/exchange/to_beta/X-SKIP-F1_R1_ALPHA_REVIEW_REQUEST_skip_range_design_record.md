# X-SKIP-F1 · R1 · ALPHA REVIEW REQUEST · skip-range design record

- **Review class:** Documents (`AGENTS.md` §2, the after-commit column). This is a design question asking for your
  **opinion**, as advice for the owner's decision. It is not a ruling request and there is nothing to commit.
  `RELAY_PROTOCOL.md` defines no separate advice type, so this uses the standard review request.
- **Base commit:** `6d75f71`. Worktree `/home/michael/tfm_worktrees/pwc-offset`, which is clean at `6d75f71` apart
  from the attached record and `data/`.
- **Under review (attached, untracked):** `/home/michael/tfm_worktrees/pwc-offset/docs/drafts/SKIP_DESIGN_RECORD_20261121.md`,
  sha256 `a234205d8594889000541ec1e9d0da471199402e61ae2d3d07a9e5f4ec003f0f`. Read it in full; its Prior record and
  anchors are this request's evidence.
- **Throwaway checks:** please write any fixtures under `$TMPDIR`, which this run points outside the repository
  (`AGENTS.md` §3). Earlier fixtures written inside a worktree caused a boundary violation (X-PWC-STAGE2B).
- **One round**, by the owner's approval.

## Prior record

The attached record's `## Prior record` section is the search for this subject: the commands run, every document
found, the labels and what was not searched. Additionally searched for this request:
`ls docs/exchange/to_beta docs/exchange/from_beta | grep -i skip` found no earlier X-SKIP thread. Your own prior
statements on this question are quoted in the record's §5: `X-VOLUME-LIMIT_R1:79`, `_R2:80` and `X-PWC-STAGE3_R1:115`.
F-1 itself is ALREADY RECORDED (`docs/TFM_SESSION_HANDOFF_20261114.md:76-80`). Nothing here CHANGES A RULING.

## Fixed facts (owner and Alpha; not under review)

1. **Owner, 2026-11-21 08:50 PST, verbatim:** "TFM is such a large project with so many revisions and pivots I dont
   remember how we envisioned it and what it is supposed to do- can you read the repo docs- maybe confer with beta?"
2. **Owner, 2026-11-21 09:25 PST, approving this relay:** "please". The lead records that approval as being for an
   **opinion, not a ruling**.
3. **This is advice for the owner's decision.** The owner decides the F-1 fix. Your answers are not binding
   (`AGENTS.md` §1; owner rulings 2026-11-11).
4. **Alpha's recommendation, which you are asked to weigh, not adopt:**
   - **(a)** a direct fix, so RANGE-MINER delivers the trial's skip range to phases 1–2, as the design documents say
     (record §2);
   - **(b)** made as its own reviewed change, after PWC Stage 4;
   - **(c)** judged by running the same job at `[0, 16]` and at the trial's range, and comparing pool hit rate against
     `k/1000` on unseen draws. It is never judged by true-seed recovery (`docs/PITFALL_STATE_RECOVERY_YARDSTICK.md`).

## Questions

Allowed answers: `NO OBJECTION` · `OBJECTION` · `SUGGESTION` · `NEEDS EVIDENCE`. Give `LIVE` or `THEORETICAL` on
every `OBJECTION`. Frame each answer by what happens in a real run on this cluster.

**Q1. The design reading.** The record concludes:
- every design source says phases 1–2 search the trial's `skip_min`–`skip_max`;
- the legacy coordinator and PWC deliver it;
- RANGE-MINER never has (`miner/range_miner_coordinator.py:9331-9358`, `miner/range_miner_worker.py:1293`);
- no document says `[0, 16]` on phases 1–2 was intended (record §3.4).

Is that reading correct? Or is there a record, a ruling, brief, proposal, test or code comment, that `[0, 16]` on
phases 1–2 was intended?

**Q2. Direct fix or flag.** CLAUDE.md yardstick step 4 says: "Propose any alternative as a flag-selectable
configuration. The current algorithm stays as the baseline." Should the fix be:
- direct, treating `[0, 16]` as a defect and not an algorithm; or
- flag-selectable, with `[0, 16]` kept as the baseline until the hit-rate comparison settles it?

In a real run, what does each option cost or risk?

**Q3. Before or after Run 4.** Three facts bear on the timing:
- Run 4 is an A/B against attempt 9 at the same skip 6–99, and both ran phases 1–2 at `[0, 16]`
  (`docs/S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md:140-141`).
- Delivering 6–99 means about 5.5× the phase 1–2 kernel work, 94 skips instead of 17. This is arithmetic, not
  measured.
- The 25,000 survivor cap was measured at `[0, 16]` (`docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md:13-18`).

Should the fix come before or after Run 4?

**Q4. What it would break in real operation.** If phases 1–2 received the trial's range, what would break or change?
Consider at least:
- **compute-lease timeouts:** the 300 s lease against longer phase 1–2 stripes;
- **stripe and sub-stripe sizing:** the VRAM caps;
- **the volume gate:** re-measurement, and phase 1–2 counts at a wider range;
- **`best_skips`:** it is `uint8` (`prng_registry.py:461`; `skip_max` ≤ 250 today, record §4.2);
- **past Optuna studies:** their recorded skip values never governed phases 1–2 (record §4.5).

Name anything else you find.
