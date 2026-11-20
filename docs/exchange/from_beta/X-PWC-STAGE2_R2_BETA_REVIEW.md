# X-PWC-STAGE2 · R2 · Team Beta review

Read 2026-11-20 PST at base `ee394047ddb7c3f9d7aadc0e4baf1d9cc474a0e6`.

`git status --porcelain` at intake

```text
 M docs/exchange/LEDGER.md
 M persistent/pwc_transport_tcp.py
 M persistent/pwc_worker_service.py
 M persistent_worker_coordinator.py
 M test_persistent_worker_harness.py
 M tests/test_pwc_backup_step1.py
 M tests/test_pwc_stage1.py
 M tests/test_s172_phase5_d3_25_candidate_ingress.py
 M tests/test_s172_threshold_propagation.py
?? data/
?? docs/drafts/pwc_backup_stage2_v1/
?? docs/exchange/from_beta/X-PWC-STAGE2_R1_BETA_REVIEW.md
?? docs/exchange/to_beta/X-PWC-STAGE2_R1_ALPHA_REVIEW_REQUEST_real_world_scope.md
?? docs/exchange/to_beta/X-PWC-STAGE2_R2_ALPHA_REPLY.md
?? tests/test_pwc_backup_stage2.py
```

| Question | Position | Severity |
|---|---|---|
| R2-Q1 · Reconnect and timeout eviction | OBJECTION | LIVE — When a silent AMD session reconnects at the timeout boundary, the reaper can evict the fresh connection using the old connection’s timestamp and requeue or fail the replacement’s lease. |
| R2-Q2 · Late results | NO OBJECTION | — |
| R2-Q3 · Refusal exit | NO OBJECTION | — |
| R2-Q4 · RANGE-MINER | NO OBJECTION | — |

Q1 holds Stage 2 from Beta’s review position. Michael decides whether the package lands.

## Prior record

- **Searches run.** Read both rounds, the package, the owner rulings, and the five records R1 asked Alpha to add. Searched `docs/` with `rg`, including `TB_RULING_*`, `PROPOSAL_*`, `CHAPTER_*_AUDIT.md`, `TODO_CONSOLIDATED.md`, and the exchanges, for timeout and last-seen ownership, reconnects, late results, refusal telemetry, and Run 4 admission.
- **ALREADY RECORDED.** The five R1 omissions are now cited. R1 Q6, Finding 2 and Q7 are the carryovers. The five-strategy GPU question remains `NEEDS EVIDENCE` for the owner’s Stage 4 sessions. `docs/exchange/to_beta/X-PWC-STAGE2_R2_ALPHA_REPLY.md:11-24`, `docs/exchange/from_beta/X-PWC-STAGE2_R1_BETA_REVIEW.md:58-62`, `:78-88`, `:103`, `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:25-29`.
- **NEW.** Timeout eviction can pair an old timestamp with a replacement socket. The cross-phase fixture has an independent scheduling assumption, described in Finding 1.
- **Scope of search.** External project records, rigs, Proxmox, and live staging and ledgers were outside the search.

## Reasoning

### R2-Q1 · Reconnect and timeout eviction

The R1 disconnect interleaving is closed. The terminating handler checks ownership and clears ID state inside `_workers_lock`. The reclaim call targets its own socket. Alpha’s reconnect-cleanup check passed with in-memory connections. `persistent/pwc_transport_tcp.py:375-406`, `:705-716`, `tests/test_pwc_backup_stage2.py:771-821`.

Timeout eviction still has a race. The replacement publishes its socket under `_workers_lock` and refreshes last-seen in a separate critical section. If the old socket is still registered and its timestamp is past 300 seconds, the reaper can take `_workers_lock` in that gap and select the replacement using the old timestamp. The reaper then releases `_last_seen_lock` before removing the registry entry and clearing readiness. The replacement can refresh its timestamp, report ready, and take a pending job during that interval. The reaper then shuts down the replacement and reclaims its lease. The worker reconnects with the same ID on a transport error, and AMD session drops are recorded. `persistent/pwc_transport_tcp.py:31-36`, `:419-447`, `:482-485`, `:512-515`, `:531-559`, `:622-658`, `persistent/pwc_worker_service.py:177-190`, `:198-214`, `:347-355`, `docs/TODO_CONSOLIDATED.md:307-308`.

I forced the two lock transitions with barriers around the actual handlers and reaper. Once the replacement had refreshed last-seen and received its assignment, I resumed the reaper. Registry and ready counts went to zero. The first-attempt lease was requeued. The third-attempt lease returned `max attempts (3) exceeded after worker disconnect`. With a fresh prior timestamp, the control kept the replacement ready and delivered the result. The fixture establishes the failure sequence. Its frequency on the rigs is unavailable. A terminal disconnect error bypasses the coordinator’s transient retry branch, and a drop below `min_workers` can block the next pass. `persistent/pwc_transport_tcp.py:383-406`, `:433-447`, `persistent_worker_coordinator.py:1762-1767`, `:1988-1997`, `:2103-2115`.

Publish the replacement connection and its fresh timestamp in the same critical section the reaper uses. Tie last-seen and readiness updates to the registered socket. Add a same-ID reconnect case at the expiry boundary. The current eviction gate uses two different IDs. `persistent/pwc_transport_tcp.py:433-442`, `:512-515`, `:531-552`, `tests/test_pwc_backup_stage2.py:824-851`.

### R2-Q2 · Late results

R1 Finding 2 is closed for the reviewed path. The pass ID scopes job IDs to a pass and a retry reuses its chunk’s job. Both inline and spooled results pass through the same locked check for job ID, active lease ID, and owning connection before normalization and delivery. The worker echoes the assigned lease ID. The equal-threshold ID check passed with the suite’s fake transport, and the active-lease check passed with in-memory connections. I also forced the intended A/B assignments in the cross-phase check. With both thresholds at `0.5`, the second pass completed and excluded the delayed P1 result. `persistent_worker_coordinator.py:2041-2064`, `:2080-2083`, `:2103-2115`, `persistent/pwc_transport_tcp.py:341-373`, `:660-679`, `persistent/pwc_worker_service.py:420-422`, `:461`, `:600-617`, `tests/test_pwc_backup_stage2.py:133-160`, `:857-977`.

### R2-Q3 · Refusal exit

The refusal logs the reason, closes the connection, and raises `SystemExit(43)` before the connected heartbeat, sieve import, or GPU device validation. I called `_connect` with a synthetic refusal after setting up heartbeat paths in a temp directory. The socket closed, the exit code was 43, the heartbeat method received zero calls, and the directory stayed empty. `persistent/pwc_worker_service.py:39-44`, `:170-221`, `:298-318`, `:341-379`.

### R2-Q4 · RANGE-MINER

I found no runtime effect on the clean Run 4 route. The launcher selects RANGE-MINER and disables PWC. The integration imports the PWC entry point, so I ran the fresh-process import gate and it passed. The miner branch returns before PWC dispatch. The job-ID allocation lives inside the PWC pass, and the miner protocol implements its own framing. My diff against `ee39404` found zero changes in the miner, launch and integration paths, and all ten pinned files. `gate12_launch.sh:465-473`, `window_optimizer_integration_final.py:51-76`, `:1761-1763`, `:1804-1819`, `:1934-1955`, `persistent_worker_coordinator.py:2041-2064`, `miner/range_miner_protocol.py:230-260`, `scripts/gate12_parity_gate.py:174-185`, `tests/test_pwc_backup_stage2.py:1039-1046`.

The standalone clean-tree admission check refused this checkout with exit 1 before fleet work, and Gate 22 tripped on six uncommitted PWC Python files. Those were the expected pre-commit results. The clean-tree and Gate 22 checks on the committed package are `UNAVAILABLE` at this stage. Rerun them after an owner commit under the existing gate rules. `gate12_launch.sh:125-139`, `window_optimizer_integration_final.py:114-122`, `scripts/gate12_cleantree_gate.py:232-245`, `tests/test_s172_phase4_coordinator.py:1621-1630`, `:2417`, `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:145-151`, `docs/TB_RULING_FIELD6_IMPLEMENTATION.md:112-120`.

## Findings

1. **NEW · SUGGESTION · Make the cross-phase regression assign its intended workers deterministically.** A’s first assignment can race B’s pulls. Once A sends the late result, it immediately resumes polling, while B waits 0.4 seconds before returning its reverse result. The gate requires a survivor from B, although it does not reserve a P2 chunk for B. In one run using in-memory connections and the original fixture scheduling, P2 completed and excluded `111`, but the assertion failed because B’s `222` was absent. It passed with barriers that ensured A got a P1 job and B got a P2 job. Put those assignment conditions in the fixture so the check measures late-result handling. `tests/test_pwc_backup_stage2.py:928-971`.

## Verification scope

I used the torch virtualenv. The sandbox refused loopback binds, so the real-loopback gates are `UNAVAILABLE` here. The in-memory checks used the actual framed handler and coordinator with synthetic connections and CPU results. The worker-child refusal gate, Alpha’s full suite and mutant tally, and fleet execution were outside my runs. I read Alpha’s reported test evidence in the request and verified that the submitted diff passed `git apply --reverse --check` against the reviewed tree. `tests/test_pwc_backup_stage2.py:618-665`, `:989-1004`, `docs/exchange/to_beta/X-PWC-STAGE2_R2_ALPHA_REPLY.md:85-113`.
