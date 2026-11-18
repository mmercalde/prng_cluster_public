# Run 4: first counted run with the v7 launcher (DRAFT, 2026-11-18)

## Prior record
- Searched: `/bin/grep` over `docs/RUN4_LAUNCH_CHECKLIST.md`, `docs/LEADS.md` (L-7), `docs/exchange/`, `docs/drafts/cache_policy_v7/README.md`, `docs/CLAUDE_CODE_REPORT_MINER_NO_TORCH.md`. Main is at `fe763fe`; v7 landed in `9034e18`.
- This draft is built from `docs/RUN4_LAUNCH_CHECKLIST.md`, which supplies the step numbers (§N) and the exact commands, the v7 README's "First counted run", and Beta X-CACHE-POLICY-D6 R1 Q3 items 1 and 2.
- Labels: everything here is ALREADY RECORDED in those documents. The step order and the clean-run wording are NEW and need the owner's OK. No ruling is changed.
- Not searched: ser8, the private repo, the rigs (off), Proxmox.

Run every VM101 command from `/home/michael/distributed_prng_analysis` with `~/venvs/torch` active. A failed step means stop; do not work around it.

## Before the rigs are on
1. **NEEDS OWNER GO.** Authorise the run: which search, the exact seed range, and why. Run 4 attempt 3 was already credited clean (2026-11-15 17:59; `CLAUDE_CODE_REPORT_MINER_NO_TORCH.md`). The upper half may therefore have no gap left to run.
2. **Routine.** Check the search record's status, using the old checklist §11's status command with `RUN4_PARAMS` (`:320`).
   - `"state": "gap"`: go on.
   - `"search_complete"`: WATCHER will not generate a Step 1 run. Return to step 1 for a new scope.
3. **NEEDS OWNER GO.** If the scope changes, open the new search (§11, `--step1-search-open`; only the owner opens a search). Then re-run step 2 and confirm `gap` and the expected `next_seed_start`.
4. **Routine.** Clean tree with HEAD recorded (§2): `git status --porcelain` prints nothing and `gate12_cleantree_gate.py --phase admission` gives rc 0. HEAD must contain `9034e18`.
5. **Routine, cache preflight.** On the committed tree, `python3 -u tests/test_s172_d6_integration_repair.py` must give 84/84. Then `python3 -B tests/test_cache_policy_v7.py --landed . --route-a /home/michael/run4_launch.sh` must be green.
6. **Routine, Route A wrapper comparison.**
   - Record `sha256sum /home/michael/run4_launch.sh`. It read `5ca23e3c…` today, which equals the landed fixture.
   - `diff gate12_launch.sh /home/michael/run4_launch.sh` must show only the `--params` block.
   - Check that block against `RUN4_PARAMS` value by value and type by type, and against any new scope from step 3.
   - Run `tests/run4_prelaunch_provenance_check.py` (§12): 10/10.
   - Keep the digest, the parameter comparison and the provenance output as three separate evidence files.
7. **Routine.** Survivor-volume gate (§10) for the exact parameters, run before the fleet parks on the 3080 Ti: PROCEED, rc 0.

## Rig bring-up and guard checks
8. **NEEDS OWNER GO.** Power on the rigs and run `boot-proxmox` on each console (§1). Confirm each CT100 answers with its own name and `ROCm_OK`. Never reboot a host to fix one.
9. **Routine.** No workers running on VM101 or any CT (§3, bracket `pgrep`).
10. **Routine.** GPU gate (§4): 3/3 rigs at 8/8. Never launch on fewer GPUs.
11. **Routine.** Parity from the committed tree (Beta D6 Q3.1). Run `gate12_parity_gate.py` (§5) and read its table.
    - 30/30: go on.
    - Any mismatch: deploy only the named files from committed HEAD, the §6 method (`git archive`, `mkdir -p`, `scp -p`, verify, clear `__pycache__`). Do not use the old `b8e259b3` and 27/30 expectations.
    - Either way, a fresh 30/30 is required at admission (§7).
12. **NEEDS OWNER GO.** Re-arm netconsole as root on .121/.155/.163 (§8). Get one proof line per host. Do not start a second listener on 6667.
13. **Routine, timing setup.**
    - Note on VM101 the time each cache join and freeze starts and ends.
    - Before any later dispatch reuses the logs, copy all 25 workers' nonce-matched park, release and timeout lines out of `/tmp/minerlogs/gpu<N>.log` on each rig and Zeus's local log.

## Launch
14. **NEEDS OWNER GO.** `nohup bash /home/michael/run4_launch.sh` (§13, Route A). `RIG_CUPY_CACHE_POLICY` stays unset, so `per_run` applies.
    - Before the sampler starts, the harness must show 24 verified rig cache rows, a passing join (24 joined, Zeus local) and `sha256sum -c SHA256SUMS` passing in `logs/gate12_<stamp>_cache_freeze/`.
    - Any refusal stops the run before the coordinator starts. That is a refusal, not a run.
15. **Routine.** While it runs, follow §14. Watch netconsole for page-fault lines.

## After the run
16. **Routine.** After the workers exit, run the printed `scripts/cache_footprint.py` command. The JSON must say `COMPLETE`; hash it into the run's evidence bundle. `INCOMPLETE` is reported as outstanding, not ignored.
17. **Routine.** Write the timing log:
    - per-slot `startup_s` and `STAGGER`;
    - the time from first park to release;
    - the join and freeze duration;
    - the maximum `SESSION_RELEASED.waited_s` against the 900 s deadline;
    - any `SESSION_RELEASE_TIMEOUT`;
    - any worker evidence that is missing.
18. **Routine.** Run `G-PROD-SHAPE` with `--run-id` (§15). Check search and history: `search_complete` for the run's search, with coverage matching. Save the files listed in §15.
19. **NEEDS OWNER GO.** Accept or reject the result.

**Clean run (needs owner confirmation)**, all of:
- Step 1 passed and the search is credited;
- `G-PROD-SHAPE` PASS for the run;
- cache freeze verified;
- footprint `COMPLETE`;
- no page-fault line on netconsole from launch to finish;
- no `SESSION_RELEASE_TIMEOUT`.

This follows L-7's usage: "credited clean", "no page-fault line".

## Dropped from the old checklist as out of date
- §5–§7's "27/30, coordinator-only mismatch, deploy `b8e259b3`": `048fd47` changed a governed worker file and was already deployed at 30/30 on 2026-11-15, so parity is now read live (step 11).
- §11's "first write of the search tables, open `[2³¹, 2³²)`": attempt 3 credited that range, so the open is now conditional on the status check (steps 2–3). §13's "D-1 decision, edit lines 379–383": Route A already exists, and today it matches the landed fixture.
