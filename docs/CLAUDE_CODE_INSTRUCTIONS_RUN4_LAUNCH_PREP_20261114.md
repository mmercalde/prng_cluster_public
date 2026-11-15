# CLAUDE CODE INSTRUCTIONS — RUN 4 LAUNCH PREPARATION (no fleet; ready for the owner in the morning)

## Prior record

**Searches run (public mirror, HEAD `80d2bb4`, 2026-11-14):**

- `ls docs | grep -iE "RUNBOOK|POWER_ON|PREREQ|LAUNCH"`; `ls scripts | grep -iE "deploy|provision|sync|parity|gpu_gate|netconsole"`.
- `grep -n -i "redeploy\|scp \|rsync\|git pull\|deploy" docs/PHASE6_PREREQS.md`; `grep -rn -i "boot-proxmox" docs/PHASE6_PREREQS.md docs/RUNBOOK_NETCONSOLE_REARM.md CLAUDE.md AGENTS.md`.
- Read: `docs/CLAUDE_CODE_REPORT_MINER_ADMISSION_STALL.md` "Before Run 4" (`:402-415`); `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md` §3.A A5–A9; `docs/exchange/from_beta/X-MINER-STALL_R1_BETA_REVIEW.md` positions and Q8; `docs/PHASE6_PREREQS.md` section list and `:121-129`.
- **Not read by Alpha:** `docs/PHASE6_PREREQS.md` in full, `docs/RUNBOOK_NETCONSOLE_REARM.md`, `docs/CLAUDE_CODE_INSTRUCTIONS_MANUAL_FLEET_LAUNCH.md`, `docs/PHASE7_PREREQUISITES.md`, `docs/S172_PHASE_7_PREREQ_REPORT.md`, `docs/TEAM_ALPHA_PHASE7_LAUNCH_NOTICE.md`, `gate12_launch.sh`, the gate scripts. **Reading them is this session's job.**

**Documents that already decide this subject:**

| Document | What it settles | Label |
|---|---|---|
| `docs/CLAUDE_CODE_REPORT_MINER_ADMISSION_STALL.md` `:402-415` | The remaining steps before Run 4: clean tree and Gate 22; power-on with `boot-proxmox`, GPU gate 8/8 ×3, coordinator redeployed (sha256 `b8e259b3…`), parity 30/30; netconsole re-armed with one proof line per host; the Route-A pre-launch check on the committed tree; the owner authorizes | ALREADY RECORDED. This brief prepares each step |
| `docs/exchange/from_beta/X-MINER-STALL_R1_BETA_REVIEW.md` Q8 | Beta's objection to launching is withdrawn for the reviewed files. Landing, clean-tree and Gate 22 checks, rig parity and launch provenance keep their roles | ALREADY RECORDED |
| `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md` §3.A A5–A8 | Power-on, netconsole, Run 4 with the committed Route-A command, then `G-PROD-SHAPE` and the closure report | ALREADY RECORDED |
| `docs/PHASE6_PREREQS.md:121-129` | The rigs are deployment targets, not working copies. Files go to them by `scp`, the same way on every rig | ALREADY RECORDED |
| `docs/RUNBOOK_NETCONSOLE_REARM.md:58` | A rebooted rig comes back as bare Ubuntu; `boot-proxmox` is required; never reboot a host without being asked | ALREADY RECORDED |
| `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:154-179` | Run 4's claim boundary | ALREADY RECORDED. Not widened |
| `docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md` M-B; `config_manifests/survivor_volume.cap` | Exact survivor counts for Run 4 on `[2³¹, 2³²)`: 0 / 0 / 60 / 23,568. Cap 25,000 by owner decision | ALREADY RECORDED |
| `docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md` §2.5, §2.6, §2.9 | Run 4 opens its own pinned search over `[2³¹, 2³²)`, by the operator. A search on a dataset other than the one behind the accumulated generation is refused | ALREADY RECORDED. See T3 |

**Items this brief raises:** none new. It assembles recorded steps into one checklist and checks state.
**Not searched:** the rigs (off tonight), ser8 archives, the private repository.

---

**From:** Team Alpha · **Owner:** Michael · **Run on:** VM101, as `michael`, venv
`source ~/venvs/torch/bin/activate` · **Model:** Opus
**Base commit:** record `git rev-parse HEAD` (expected `80d2bb4` or later).
**Review class:** document. No relay. **Outputs, both in your worktree:**
`docs/RUN4_LAUNCH_CHECKLIST.md` and `docs/CLAUDE_CODE_REPORT_RUN4_LAUNCH_PREP.md`.

**Why.** The owner launches Run 4 first thing tomorrow. Everything that can be done or
checked without the fleet is done tonight, and every remaining step is written down with
its exact command, so the morning is a checklist and not a search.

**The yardstick (CLAUDE.md).** Prediction-pool hit rate against `k/1000` on unseen draws.
Run 4 is an operational run. It measures no hit rate, and nothing here claims it does.

## 0. Rules for this session

- **You launch nothing.** No pipeline step, no coordinator, no WATCHER run, no rig SSH, no
  reboot, no GPU job. The rigs are off.
- **You do not open, close or change a Step 1 search.** That is the owner's act.
- **You edit no existing file.** You write only the two output documents.
- **Worktree:** `git worktree add --detach /home/michael/tfm_worktrees/run4-prep HEAD`. It is
  a clean checkout of the committed tree; run the suites there.
- **Live state is read-only**, as `docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md`
  §A says. Do not run `gate_manifest_staging_dir_is_usable_here`.
- **No command in the checklist is invented.** Each comes from a recorded document or from
  the code, with `file:line`. Where the record gives no command for a step, write
  `UNAVAILABLE` and say what is missing. A guessed command on a fleet run is worse than a gap.
- No scheduled wake-ups or background timers. Keep going on small things. There is no stop
  gate except a write to live state.

## 1. T1 — the committed tree is green

In the clean worktree at HEAD, run every suite the four reports of 2026-11-14 name
(`docs/CLAUDE_CODE_REPORT_STEP1_SEARCH_RECORD_STAGE1.md`,
`docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md`, `docs/CLAUDE_CODE_REPORT_MINER_SEED_START.md`,
`docs/CLAUDE_CODE_REPORT_MINER_ADMISSION_STALL.md`), one at a time. One table: suite, result
tonight, result the report gave, difference explained. In particular:

- Gate 22 and the other commit-sensitive gates, which were red only on uncommitted files.
  No allowlist is widened.
- `tests/run4_prelaunch_provenance_check.py`: 10 of 10, and print the command it builds.
- `tests/test_s172_run4_routing_patch.py`: 21 of 22 with the frozen gate INCOMPLETE.
- R-1 and MP-1.

A suite that is red tonight and was green in its report goes at the top of your report.

## 2. T2 — the checklist

`docs/RUN4_LAUNCH_CHECKLIST.md`. Numbered steps in the order they are done. For each: who
does it (OWNER, or a session at the owner's word), the host tag (`# run on ser8`,
`# run on VM101`, `# run on pve-zeus`, a rig), the exact command, what a pass looks like,
what to do on a fail, and the source `file:line`. Cover at least:

1. Rigs powered on and booted into Proxmox (`boot-proxmox`); how to confirm each is in the
   right OS before anything else.
2. GPU gate, 8 of 8 on each of three rigs.
3. The new coordinator to each rig by the recorded `scp` method, from committed HEAD, with a
   sha256 check on the rig against `b8e259b3e12b95297fa38018a41215d5c52f4d9b3f4095e384829a3ec1f1517d`.
   List every other parity-set file that differs between HEAD and what the record says the
   rigs carry; if you cannot tell from the record, say so and make the parity gate the test.
4. Parity gate, 30 of 30.
5. Netconsole re-armed, one proof line per host.
6. Clean tree on VM101; the dataset frozen and present on the fleet as the record requires.
7. The survivor-volume gate for Run 4's request.
8. Opening Run 4's search: the exact operator command, with the range `[2³¹, 2³²)` and the
   seven pins. Then the status command and what it should print.
9. The Run 4 launch command, exactly as the committed pre-launch check builds it, under
   `nohup`, with where its log goes.
10. What to watch while it runs: the expected survivor counts per phase (0 / 0 / 60 / 23,568),
    attempt 9's timing as a guide, the lines that mean trouble, and how to stop it cleanly.
11. After it finishes: the verifier (`G-PROD-SHAPE`) command, what to save, and what the
    search and history should then report.

Put a short "if something goes wrong" section at the end: the halt file, a hung run, a rig
that drops, and which of those the owner should stop for.

## 3. T3 — live state, read-only

Report each as a fact with the command used:

- The history ledger: its rows for `java_lcg`. Expected: attempt 9, `[0, 2147483648)`.
- The search ledger: does it exist yet on the live database; is any search open.
- **The dataset.** Which dataset version is current now, and is it the one attempt 9 ran on
  (`daily3-20260801T145551443433Z-513648160d35.json`, sha256 `513648160d35…68f6`)? If new
  draws have moved the pointer since August, say so at the top of the report: Run 4's search
  would be opened on the new dataset and §2.6 would refuse it against the accumulated
  generation. That is the owner's decision before launch, not something to work around.
- The halt file; the live staging directory (anything left from earlier runs); the miner
  ledger's `-wal` size (the evidence reader now refuses credit if it is not empty).
- The Optuna study folder: anything that a resume could pick up by mistake.
- Whether an authorization file or an environment baseline exists for the delegated gates
  (`scripts/run_authorization_gate.py`, `scripts/env_freeze_gate.py`). The owner is launching
  by hand, so these do not gate Run 4; say what exists.

## 4. Report

`docs/CLAUDE_CODE_REPORT_RUN4_LAUNCH_PREP.md`, opening with `## Prior record` (your reading
of the documents Alpha did not read). At the top, for the owner, in plain words: is the
committed tree green; is anything in the way of launching tomorrow; what he must decide.
Then T1, T3, every `UNAVAILABLE` in the checklist, and the sha256 of both documents.
