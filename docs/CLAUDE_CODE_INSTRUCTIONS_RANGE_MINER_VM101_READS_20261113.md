# CLAUDE CODE INSTRUCTIONS — RANGE-MINER: TWO VM101 READS BEFORE RUN 4 (READ-ONLY) — 2026-11-13

**From:** Team Alpha · **Owner:** Michael · **Run on:** VM101, as `michael`, venv
`source ~/venvs/torch/bin/activate` · **Model:** Opus
**Base commit:** `26d3147`. Anchors were read at that commit on the public mirror; re-read
them on VM101.
**Plan items:** A3 and A4 of `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md` §3.A (packet W0-5).
**Review class:** document (Beta after commit).

**Why.** Run 4 is the next RANGE-MINER run and it is prepared but not launched. Two facts it
depends on exist only on this host and have never been read into the record: where the
coverage cursor sits, and how many bidirectional survivors attempt 9 published. This session
reads both and reports. It changes nothing.

---

## 0. Rules for this session

- **Read-only. You edit no existing file.** You do not commit or push. You launch nothing.
- **The fleet is off unless Michael says otherwise.** No rig SSH, no power, wake or reboot.
- **Work in a worktree.** `git worktree add --detach /home/michael/tfm_worktrees/rm_reads HEAD`.
  Write only inside it: `docs/CLAUDE_CODE_REPORT_RANGE_MINER_VM101_READS.md`, and scratch
  under `data/rm_reads/`.
- **Live state is opened read-only.** SQLite files with a `mode=ro` URI, never a writer
  connection. NPZ files with `numpy.load(..., mmap_mode="r")` where possible. Nothing under
  `~/miner_staging`, `~/miner_ledger_archive`, `.s172_accumulator/`, `.s172_checkpoint/` or
  any generation directory is written, moved, renamed or deleted. Do not run the finalizer,
  the coordinator, WATCHER, or any gate that writes.
- **Nothing is installed or upgraded.**
- **Absence claims** carry the search that produced them. What cannot be read is `UNAVAILABLE`.
- **Lead handling.** If the bidirectional count is not what noise predicts, record it as an
  observation with its limits. Do not call it chance, and do not call it a finding. See
  `CLAUDE.md`, "Lead handling".

## 1. What Alpha read at `26d3147`

| Anchor | What it shows |
|---|---|
| `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md` §3.A rows A3, A4; §9 (`:336`) | the two reads, and the note that "cursor at 2³¹" is the fact file's record, not a reading |
| `utils/seed_coverage_ledger.py:436` | `covered_seed_count` |
| `agents/watcher_agent.py:2073`, `database_system.py:428` | where the cursor is reported |
| `utils/run_finalizer.py:1624` | generations live under `accumulator_root / GENERATIONS_DIRNAME` |
| `docs/CLAUDE_CODE_REPORT_PROD_SHAPE_RUN_SCOPING.md:266` | on 2026-11-12 no `gen-*` directory was found in the repo root |
| `docs/S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md` | Run 4 sets `max_seeds = 2³¹` and names no `seed_start` |
| `docs/TFM_PROJECT_FACTS_SKILL.md` §2.49 | attempt 9: run `distributed_config_t1_554463d3`, `coverage_id c6f28aedf7af12cd`, `covered_seed_count=2147483648`, `certified_interval_count=1` |
| `TFM_SESSION_HANDOFF_20261111.md` §5 (**not in the repo** at `26d3147`; it is a claude.ai project file) | Alpha's 2026-11-11 simulation estimate: noise predicts about 0.01 bidirectional survivors at attempt-9 settings. Treat as Alpha's estimate, not a repo fact |

## 2. What to read and report

**R0 — State of the host.** `git status --porcelain`, `git worktree list`,
`git log --oneline -3`; any coordinator, WATCHER or window-optimizer process; whether port
5700 is listening; which Claude Code or Codex sessions have files open for edit, if that can
be seen. This doubles as the check that the findings work's no-touch list
(`docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md` §0) covers every file in play: list any
file in play that is not on it.

**R1 — Coverage cursor (A3).** Locate the seed-coverage ledger on this host. Report its path,
size and modification time, and, per PRNG type: `covered_seed_count`, the certified
intervals, and the `coverage_id`. State whether it matches the fact file's record of
attempt 9.

**R2 — Run 4's seed range (A3).** With the cursor where R1 found it, what would the committed
Run 4 command do: repeat `[0, 2³¹)`, be refused as already covered, or continue from the
cursor? Trace it in the code (read-only) and cite the lines. Do not run it. Set out the
options for the owner; do not choose.

**R3 — Attempt-9 bidirectional count (A4).** Locate the generation published by attempt 9
(run `distributed_config_t1_554463d3`). The 2026-11-12 scoping report found none in the repo
root, so search the accumulator root and the archive locations and name every directory
searched. If found, report: path, sidecar identity, the number of rows in the 22-array NPZ,
and how many are forward-only, reverse-only and bidirectional, with the code that defines
each. If not found, report `UNAVAILABLE` and say what would have to exist to answer it.

**R4 — Anything else Run 4 would trip on.** From what you read only: leftover staging files,
a halt file, a stale lock, a ledger from an aborted run. List; do not clean.

## 3. Deliverable

`docs/CLAUDE_CODE_REPORT_RANGE_MINER_VM101_READS.md`: R0–R4 with paths, commands and
`file:line` anchors read in this session, then a short "for the owner" section with the
Run 4 seed-range options.

## 4. Stop gate

Stop when the report is written. Tell Michael the report path. He commits it.
