# SESSION CHANGELOG — 2026-11-15 — Step 1 credit on a non-empty ledger `-wal`

Brief: `docs/CLAUDE_CODE_INSTRUCTIONS_STEP1_CREDIT_WAL_20261115.md`. Report (the full record):
`docs/CLAUDE_CODE_REPORT_STEP1_CREDIT_WAL.md`. Worktree `/home/michael/tfm_worktrees/step1-credit-wal`,
base `c47680a`. Claude Code session on VM101 as `michael`.

## What happened

- **Q1 executed.** Run 4 attempt 1 (1 attempt, 0 credited) was refused credit because the gate12
  sampler, a read-only reader, was connected when the coordinator's last writing connection
  closed. A read-only last closer neither checkpoints nor deletes the WAL, so the `-wal` stayed
  non-empty, and the base S7 reader refuses any non-empty `-wal`. This corrects the brief's
  point 4: what decides is who closes last.
- **Change:** `window_optimizer_integration_final.py`. When the `-wal` is non-empty, Step 1 opens
  nothing on the original. It copies the `.db` and `-wal` to private system-temp scratch,
  requires the source to be stable across the copy, reads the copy, and fails closed on any
  error. The `-wal`-empty immutable branch is unchanged.
- **Tests:** new `tests/test_step1_credit_wal.py`, 12 gates and 5 mutants.
  `tests/test_s172_small_repairs_part3.py` G-EVID-WAL is reworded under the owner ruling, with a
  content-and-mtime directory check.
- **Frozen Run 4 ledger, on a copy:** phases 1-4 each `[2^31, 2^32)`. Today's sweep would have
  been credited. Nothing was credited.
- **Q2 dry run on a fixture database:** a second launch on the same search refuses in the child
  before dispatch. Close, then open with `--search-repeat-of`, works. Owner commands are in
  report §5.
- **Owner ruling, 2026-11-15 12:41 PST, in chat (NEW):** approves the narrow automatic-copy
  exception, system temp for the scratch copy, and the part3 rewording (report §0).
- **Beta:** R1 and R2 both `REVIEW RECEIVED`, with no objections. The suggestions were adopted
  (report §9). The final part3 mtime addition and the report cleanup came after R2 and are not
  re-reviewed; R3 is still available.

## Suites (sequential, patched tree, pass 3 + part3 after R2)

part3 3/3, 3/3 · stage-1 39/39, 33/33 · seed_start 10/10, 11/11 · credit-WAL 12/12, 5/5. All PASS.

## Not done (by rule)

No commit, push, fleet launch, search open or close, or `prng_analysis.db` write. The live ledger
was not opened.

fallback parity: not assessed this session (not a phase boundary).
