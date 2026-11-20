# X-PWC-STAGE1 · R1 · Team Beta review

Date 2026-11-20 PST. Base read `27eb9910ea8ec41ec197d707758dd7246e7dcfe5`. Reviewed the worktree and checked that the submitted diff matches its changed source files.

`git status --porcelain` at intake

```text
 M persistent/pwc_protocol.py
 M persistent/pwc_transport_base.py
 M persistent/pwc_transport_tcp.py
 M persistent_worker_coordinator.py
 M test_persistent_worker_harness.py
 M test_s149_rocr_isolation_harness.py
 M tests/test_pwc_cache_reset.py
?? data/
?? docs/drafts/pwc_backup_stage1_v1/
?? docs/exchange/to_beta/X-PWC-STAGE1_R1_ALPHA_REVIEW_REQUEST_real_world_scope.md
?? scripts/pwc_parity_gate.py
?? tests/test_pwc_backup_step1.py
?? tests/test_pwc_stage1.py
```

| Question | Position | Severity |
|---|---|---|
| Q1 · B5 process scope | NO OBJECTION | — |
| Q2 · B5 refusal and wait | NO OBJECTION | — |
| Q3 · A3 level 1 and same-ID workers | SUGGESTION | — |
| Q4 · Rejected worker retries | SUGGESTION | — |
| Q5 · Held worker hunk | NO OBJECTION | — |
| Q6 · Pipeline PWC regression | SUGGESTION | — |
| Q7 · RANGE-MINER | NO OBJECTION | — |

**Disposition.** No `LIVE` objection to landing Stage 1 in the agreed sequence. The counted launch gate refused this dirty worktree. On the reviewed runtime path I found no change to the miner's dispatch or results. Stage 1 also removes a possible VM101-only PWC diagnostic run when rigs stay in the configuration. That consequence belongs in the Stage 3 handoff.

## Prior record

- **Searches run.** Searched `docs/` with `rg`, including `TB_RULING_*`, `PROPOSAL_*`, `CHAPTER_*_AUDIT.md`, `TODO_CONSOLIDATED.md`, the catalog, backlog, leads and prior exchanges. Searched for S156, `pkill` self-match, `pgrep` status and markers, orphan reconnects, launch-set revocation, `PWC_HOST`, local-only PWC, Gate 22 and counted-launch clean-tree admission. Checked the request's referenced scope, the Stage 1 package and the applicable launch and worker code.
- **ALREADY RECORDED.** Prior cache-reset reviews and the carried scope identify the orphan-reconnect gap, the one-ready-worker rule and the staged token work. The Field 6 and Run 4 Route A rulings govern the dirty-tree test failures. The verification-integrity standard and the S182 count precedent are addressed in Finding 1. `docs/exchange/from_beta/X-PWC-CACHE-RESET_R2_BETA_REVIEW.md:93-103`, `docs/FLEET_STATE_REQUIREMENTS_v1.md:170-190`, `docs/drafts/SCOPE_PWC_FULL_BACKUP_20261119.md:231-255`, `docs/TB_RULING_FIELD6_IMPLEMENTATION.md:112-120`, `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:145-151`.
- **NEW.** The successful marker on probe error and the dispatch to an already admitted ID after revocation were reproduced with fixtures. Q6 records the specific consequence of the existing one-ready-worker policy for a VM101-only diagnostic with rigs configured. No owner ruling changes.
- **Outside the search.** Rigs and Proxmox, deployed `/tmp` and service state, live staging and ledgers, ser8 archives, the private repository and private project files.

## Finding 1 · The prior record omits the verification-integrity standard

**ALREADY RECORDED.** The scope carries S156. Add `docs/VERIFICATION_INTEGRITY_STANDARD.md` directly to the Stage 1 record. VIR-4 applies to the self-matching reporter. VIR-1 and VIR-5 also require evidence that the process check ran and could observe its target. S182 already distinguished `pgrep -c` exit 1 with a count of zero from probe errors. Those rules apply to the new cleanup marker as well as the previous self-kill. `docs/VERIFICATION_INTEGRITY_STANDARD.md:30-35`, `:93-117`, `docs/SESSION_CHANGELOG_20260814_S182.md:122-126`, `docs/exchange/from_beta/X-PWC-CACHE-RESET_R2_BETA_REVIEW.md:5`.

## Q1 · B5 process scope

The counted CT100 workers execute `miner/range_miner_worker.py`. The local worker uses the same entry point, and the counted rig cache has a run-specific path. I found no process required by that launch whose command line matches either B5 pattern. A still-active legacy SSH-PWC worker on a rig does match `sieve_gpu_worker.py --persistent`, so starting another PWC run on the same rig would kill it. The broad pattern checks the command line and cannot determine which run owns the process. The reviewed operating case is a single backup run with stale workers to reap. `scripts/launch_fleet_manual.sh:189-191`, `:344-351`, `:393-403`, `gate12_launch.sh:270`, `persistent_worker_coordinator.py:182-194`, `:1093-1098`.

In the private PID namespace, the base TCP shell died with exit 137, emitted no marker and left the sieve dummy alive. The new TCP command reaped both dummies and printed the marker. The S156 test reproduced the old find matching its own shell and the new find returning `none`. Startup runs S156 before the transport split, so that preliminary reap also runs on TCP startups. The confirmed cleanup that controls rig admission is the later TCP step. CT100 execution is **UNAVAILABLE**. `27eb991:persistent_worker_coordinator.py:757-768`, `persistent_worker_coordinator.py:891-961`, `:1196-1221`, `tests/test_pwc_stage1.py:153-234`.

## Q2 · Refusal and wait

Keep the two-second cutoff for this stage. If a visible worker is still stuck in the driver, skipping its rig avoids resetting its cache and launching another GPU worker over it. A worker that exits just after the check can cost a healthy rig's capacity for that attempt. If every other worker also fails to become ready, the startup gate can then refuse the attempt. A longer wait would help that specific recovery case. There is no CT100 timing evidence here to pick a better cutoff. TCP distributes chunks among ready workers, so a skipped rig can make the pass slower and still leave full coverage. If chunks go unswept, S1 marks the pass incomplete. S156 on the SSH route still proceeds after a reap warning. `persistent_worker_coordinator.py:188-194`, `:931-958`, `:1202-1221`, `:1406-1442`, `:1803-1846`, `:2101-2138`, `:2240-2254`.

## Q3 · Same-ID workers

**TO-DO for Stage 2.** With a successful rig process check and a single active launch, I found no repository-managed path for a killed same-ID rig worker to reappear. A process started after the scan can reuse the ID. An old launch script run again after cleanup is one route if it targets the active listener. I found no evidence of automatic replay on these rigs. VM101 has a separate gap. Its local kill ignores the result and logs success. If that kill fails after an interrupted PWC session, the old loopback worker can retry after `zeus_gpu0` is allowed for the new launch. Once admitted, another connection with that ID can overwrite its registry entry, and either handler's disconnect can remove that entry or reclaim work under it. This is a reason to implement the scheduled token and to check local cleanup. A local kill failure or post-scan relaunch on this cluster is **UNAVAILABLE**. `persistent_worker_coordinator.py:1158-1167`, `:1202-1221`, `:1238-1267`, `:1283-1298`, `:1324-1353`, `persistent/pwc_worker_service.py:169-220`, `persistent/pwc_transport_tcp.py:427-461`, `:624-636`.

## Q4 · Rejected worker retries

**TO-DO for Stage 2.** A refused hello ends before registration, leasing or the dispatch semaphore. A real refusal is retried because the worker's finite counter looks for a different error after a previous session. The retry can generate up to about 720 warnings per worker-hour, or 18,000 across 25 slots, plus short-lived connections. The rig launch appends those warnings to per-worker `/tmp` logs. I found no evidence of a run-length load or disk failure from that volume here. A previously initialized process can retain its GPU state while it retries because the retry closes the socket. Exiting on an explicit rejection addresses both the log growth and lingering process. `persistent/pwc_transport_tcp.py:400-416`, `:431-449`, `:563-599`, `:624-636`, `persistent/pwc_worker_service.py:39`, `:169-220`, `:357-371`, `persistent_worker_coordinator.py:1238`, `:1262-1265`.

## Q5 · Held worker hunk

Keep the worker hunk held for Stage 1. A hand start without the variables takes the fallback. A retained pre-Stage-1 `/tmp/pwc_tcp_launch_gpuN.sh` can do the same because the old template supplied CLI arguments and the worker reads the environment. That is a real launch mechanism. Its presence on a CT100 and any systemd unit are **UNAVAILABLE**. The new TCP rig and VM101 launches set both variables. While `.127` is down, a fallback launch can retry and log but cannot receive work there. Bare-metal boot creates a risk if a PWC listener is also accepting hellos at `.127:5600`. Boot by itself only makes the destination reachable. The base transport accepted every hello, so an old listener could serve the worker. `27eb991:persistent_worker_coordinator.py:784-827`, `27eb991:persistent/pwc_transport_tcp.py:393-410`, `persistent/pwc_worker_service.py:340-361`, `:757-773`, `persistent_worker_coordinator.py:1251-1265`, `:1326-1345`.

## Q6 · Pipeline PWC regression

**TO-DO for Stage 3.** I found no evidence of a previously working VM101 TCP run with participating rigs. There is a concrete route for a small constant-skip diagnostic with the rigs still configured. Before Stage 1, VM101's loopback worker could become the sole ready worker, satisfy the effective minimum of one and pull every chunk from the common TCP queue. After Stage 1 the same invocation refuses before the listener or local worker starts because the call omits the rig dial address. A CPU fixture with localhost and one configured rig confirmed that base dispatch with one synthetic ready worker issued both chunks covering `[0,20)`, while the patched startup check refused that configuration. End-to-end GPU success is **UNAVAILABLE**. Explicit SSH bypasses the address check and can still newly refuse on the sizing envelope. Record the degraded TCP route as a behavior change alongside the planned shared address and sizing work. The owner has already reserved PWC for backup and diagnostics, so I do not hold Stage 1 for the planned fail-closed behavior. `window_optimizer_integration_final.py:1968-1986`, `27eb991:persistent_worker_coordinator.py:850-870`, `:933-950`, `:1296-1329`, `:1371-1377`, `:1691-1698`, `27eb991:persistent/pwc_transport_tcp.py:213-221`, `:449-451`, `persistent_worker_coordinator.py:423-438`, `:862-896`, `:2369-2382`, `docs/FLEET_STATE_REQUIREMENTS_v1.md:170-190`, `docs/drafts/SCOPE_PWC_FULL_BACKUP_20261119.md:465-473`.

## Q7 · RANGE-MINER

I found no package line that changes worker launch, sieve dispatch or results for the Run 4 RANGE-MINER route on a clean checkout. The launcher sets miner on and PWC off, and the integration's miner branch returns before the PWC branch. The shared integration import of the PWC coordinator succeeded in a fresh process. All ten local parity pins matched `27eb991` byte for byte. Rig-deployed parity and a counted run from a committed Stage 1 tree are **UNAVAILABLE**. `gate12_launch.sh:314-317`, `:465-473`, `scripts/gate12_parity_gate.py:174-185`, `window_optimizer_integration_final.py:57-77`, `:1761-1767`, `:1926-1955`, `tests/test_pwc_stage1.py:493-503`.

**The current dirty checkout is a real launch refusal.** I ran only the existing clean-tree admission script against this worktree. It exited 1 and reported all 13 intake entries, including `data/` and the untracked review materials. The counted launcher runs the gate before fleet dispatch, in the production checkout to which it changes directory. The refusal applies there if the same dirty paths are present. Gate 22's allowlist and its S7 assertion inspect porcelain, as does the threshold suite's frozen-file check. On a clean commit those particular assertions have no dirty paths to reject. The separate D6 baseline failure remains recorded. Rerun the commit-sensitive checks on the committed tree with the existing rules. `gate12_launch.sh:79`, `:125-139`, `scripts/gate12_cleantree_gate.py:108-134`, `:232-245`, `window_optimizer_integration_final.py:101-122`, `tests/test_s172_phase4_coordinator.py:1621-1629`, `:2417-2428`, `tests/test_s172_threshold_propagation.py:544-551`, `:607-614`, `docs/exchange/from_beta/X-PWC-BACKUP_R2_BETA_REVIEW.md:79-83`.

## Additional findings

2. **NEW · TO-DO. An unsuccessful process probe can produce a clean marker.** I ran the generated shell strings with `pgrep` stubbed to exit 2. TCP exited 0 with `PWC_STALE_KILL_OK`. S156 find and reap produced `none` and `reaped:0`. Require an explicit no-match status from a functioning probe, refuse the TCP rig on other statuses, and report an unavailable S156 probe as such. Its count fallback also retains the already recorded S182 double-zero construct. No observation failure from CT100 was established in this review. `persistent_worker_coordinator.py:188-194`, `:911-949`, `:1202-1217`, `docs/SESSION_CHANGELOG_20260814_S182.md:122-126`.
3. **NEW · TO-DO. Revocation applies to future hellos.** If a worker connects while the SSH launch is in progress and the SSH reply then fails or loses its PID line, the coordinator revokes the ID but leaves the accepted socket eligible. A fake framed connection reported ready and took a job after revocation. A missing PID line also falls through to the launched counts and can consume the 60-second online wait. Make the launch outcome and accepted connection state agree, and count only confirmed launches. The fixture establishes the handler behavior. A live SSH failure in that window is **UNAVAILABLE**. `persistent_worker_coordinator.py:1282-1315`, `:1362-1392`, `persistent/pwc_transport_tcp.py:192-199`, `:427-461`, `:477-501`, `:586-599`.

## Evidence run and limits

- Activated `~/venvs/torch`, disabled GPU visibility and bytecode writes, and redirected the Stage 1 test fixture output to a temporary directory. Eight of the nine named gates passed, including the private PID namespace checks, the launch wiring with subprocess fakes, B6 and the fresh import. The real loopback gate was **UNAVAILABLE** because the sandbox denied TCP socket creation. Alpha reported its loopback pass. The fake framed-connection check exercised revocation separately.
- Ran the two probe and admission fixtures described above, the baseline dispatch fixture, the counted clean-tree check, the ten-file byte comparison, `git apply --reverse --check` against the package and `git diff --check`. Read Alpha's wider suite report. The full suite and live staging probe were excluded, since the latter calls the validator on the manifest's real staging directory. `tests/test_s172_staging_partb.py:599-609`, `miner/range_miner_coordinator.py:12190-12205`, `docs/drafts/pwc_backup_stage1_v1/README.md:189-259`.
- The B6 fixture confirmed both skip-field names. That is parser evidence. PWC credit and prediction-pool hit rate against `k/1000` on unseen draws remain **UNAVAILABLE** at Stage 1. `tests/test_pwc_stage1.py:474-487`, `persistent_worker_coordinator.py:1635-1640`, `sieve_filter.py:485-501`, `window_optimizer_integration_final.py:1997-2007`.
