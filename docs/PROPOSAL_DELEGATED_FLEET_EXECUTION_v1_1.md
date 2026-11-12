# DELEGATED FLEET EXECUTION (amendment to `CLAUDE.md` rule 3) — v1.1

**Author:** Team Alpha · **Date:** 2026-11-11 · **Owner:** Michael
**Status:** ADOPTED IN PRINCIPLE by the owner, 2026-11-11. Nothing is built yet. No agent
launches a run until the guard rails in §7 exist and §11 is met.
**Supersedes:** v1.0 (same date), written as a request for Team Beta's approval. Under the
owner's rulings Beta reviews and does not approve, so this is now the owner's amendment with
Beta's review invited.
**Base:** public repo at `a15b524` (2026-08-28). Not checked against live VM101 or the rigs.
**Amends:** `CLAUDE.md` §1 rule 3. **Does not touch:** rule 1 (commits), rule 2 (dual-push),
`CLAUDE.md` §4 (rigs pinned and frozen), SR-1, SR-2.

---

## 0. WHAT CHANGED FROM v1.0

| v1.0 | v1.1 | Source |
|---|---|---|
| Proposal awaiting Beta's ruling | Owner's decision; Beta reviews | owner ruling 1 |
| Circuit check or GPU power caps required before unattended runs | **Removed** | owner ruling 3 |
| Power cutoff among the stop conditions | **Removed.** Netconsole stays | owner ruling 3 |
| A ticket written by the owner for every run | A **standing authorization** per run class, so agents can launch without the owner present | owner, in session |
| Counted runs (Run 4, soak) delegated only if Beta agrees | **Delegated and counted.** Beta reviews the evidence before the result is accepted | owner, in session |
| Freeze-reproduction and high-volume stress classes | **Removed.** No run is launched to overwhelm the rigs | owner, in session |
| No volume control | **Survivor-volume limit added** as a launch refusal | owner, in session |
| Agent model unspecified | Opus 5.5 | owner ruling 6 |

## 1. The owner's ruling

> Agents can launch. That is the whole point. We need guard rails. I do not want them updating
> the kernels or crashing my rigs.

And earlier the same day: the rule that kept agents off the rigs was written to limit the damage
an agent could do. Agents may now control the rigs **with authentication and in limited scope**,
and are **never** allowed to update the system.

So there are two things the guard rails must prevent:

1. **Any system change.** PyTorch, drivers, kernel, OS, on any host.
2. **Overwhelming the rigs.** "The rigs are fine so long as they are not overwhelmed."

## 2. Design principle

A principle with no enforcing gate is not a control. The rig-parity rule was written down, and
three Gate-12 attempts ran without anyone checking it (fact file §2.44).

Every limit below is paired with something that **refuses**, at the operating-system or
launcher level. A sentence in an agent's instructions is not counted as a control.

## 3. What a delegated agent MAY do

| # | Action | Bound |
|---|---|---|
| A1 | Read-only telemetry on VM101 and the CT100 workers | no state change |
| A2 | Run the existing prelaunch gates (`gate12_gpu_gate`, `gate12_parity_gate`, `gate12_cleantree_gate`, `gate12_sentinel_gate`, `gate12_worker_liveness_gate`) | unmodified, from a clean committed tree |
| A3 | Start and stop the worker fleet via `scripts/launch_fleet_manual.sh` | fresh nonce every launch |
| A4 | Launch a run through the certified launch path | only inside a valid authorization (§5), one trial at a time, `n_parallel = 1` |
| A5 | Arm samplers and the netconsole listener before dispatch | armed before the first stripe is assigned |
| A6 | Deploy the ten governed project files to the rigs | from committed HEAD only, verified `30/30 MATCH` by the parity gate afterwards |
| A7 | Collect and hash-freeze evidence bundles; write reports | uncommitted; the owner commits |

A6 is project source, not system software. A commit that changes a governed file makes the
parity gate refuse until the rigs are redeployed.

## 4. What a delegated agent may NEVER do

No authorization, run class or failure unlocks any of these.

1. **No system or environment change on any rig, CT or Proxmox host.** No `apt`, `pip`, `conda`
   or equivalent install, upgrade, downgrade or removal. No change to PyTorch, CuPy, ROCm, GPU
   drivers, kernel, firmware, BIOS, grub, kernel parameters or module parameters. No edit to a
   rig environment.
2. **No kernel, GPU-driver, CUDA, PyTorch or OS change on VM101 or the Zeus host.** No
   `apt upgrade`, no `do-release-upgrade`.
3. **No "repair" of an environment difference.** `rrig6600c` runs Python 3.10.12 against 3.10.4
   on the other two. It is recorded in the baseline and left alone.
4. **No reboot, power-off or boot-target change of any host.** A rebooted rig returns as Ubuntu
   and needs `boot-proxmox`; that step is the owner's.
5. **No Proxmox configuration change** (`qm`, `pct`, passthrough, storage, network).
6. **No commit and no push.**
7. **No edit to a tracked file during a run session.**
8. **No gate bypass or weakening.** A refusal is a result.
9. **No relaunch after a failure.** One terminal event ends the session (§8).
10. **No launch outside the authorized parameter envelope**, and no launch whose predicted
    survivor volume exceeds the cap (§7.3).
11. **No write to the dataset, the pointer manifest or the coverage ledger** outside the
    certified publication path.
12. **No access to `.127` bare-metal Zeus.**

## 5. Authentication and authorization

### 5.1 Who is acting: a dedicated agent identity

- A dedicated SSH keypair for delegated execution, usable **from VM101 only**.
- On the CT100 workers: a non-root account with **no `sudo`**, restricted to worker launch, stop
  and telemetry.
- On the Proxmox hosts: no access.
- The owner's own keys are not shared with any agent.
- The run operator is an Opus 5.5 agent.

### 5.2 What is authorized: a standing run authorization

One file per run class, written by the owner. The launcher refuses without one. It states:

```
run class            RUN4 | SWEEP | SOAK                           (see §9)
parameter envelope   the exact pinned parameters, or the bounds each may take
seed range           start and end the class may cover
max launches         how many runs this authorization covers
max wall-clock       per run                                      OWNER TO SET
permitted window     hours during which runs may start            OWNER TO SET
expiry               date after which the authorization is void
```

An authorization covers exactly what it names. Anything outside it is refused, not interpreted.
The owner writes it once per class, not once per run.

## 6. Counted runs

Run 4, the sweep and the soak may be agent-launched and still count.

The reason: what makes a run count is the evidence it leaves, not who started it. The clean-tree,
parity, publication and coverage walls each refuse on their own evidence, and `G-PROD-SHAPE`
reads what a completed run left behind.

Beta reviews a counted run's evidence **before its result is accepted**. The launch does not
wait on Beta. If Beta objects and Alpha disagrees, both positions go to Michael.

Beta's earlier recorded position was that delegated runs should be non-certifying. That position
is on record; the owner has ruled otherwise.

## 7. Guard rails to be built

Each gets a clean control, a fault-injection control and mutants.

### 7.1 Authorization gate

A fail-closed prelaunch gate before the fleet launch: valid authorization, unexpired, inside the
permitted window, launches remaining, launch parameters inside the envelope. Missing or
malformed is `REFUSE`.

### 7.2 Environment-freeze gate

The enforcement of §4 items 1 and 2, modelled on `scripts/gate12_parity_gate.py`.

- The owner captures a **baseline per host** once: package list, Python version, kernel release,
  ROCm or CUDA version, GPU driver version, and digests of the rig environment.
- The gate compares live state to the baseline **before every delegated launch and again after
  the run**.
- Any difference is `REFUSE` before launch, and a reported violation after. An unreachable host
  is `UNAVAILABLE`, never clean.
- Only the owner changes the baseline.

### 7.3 Survivor-volume limit

The enforcement of "do not overwhelm the rigs".

- Before launch, a small sample of the seed range is run through the production kernel on the
  run's own residue window, and the survivor count per phase is projected to the full range. The
  Run 4 proposal already predicts volume this way (`S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md` §1.1:
  predicted 96 and 22,656 against attempt 9's actual 59 and 23,515).
- The launch is `REFUSE` if any phase's projection exceeds the cap.
- **Starting cap: 23,515 survivors in one phase**, the largest volume carried through a complete
  certified run (attempt 9, phase 4). Run 3 froze after 13,146,485. The cap is raised only by the
  owner, on evidence from completed runs.
- Until this gate exists, agents launch **pinned-geometry runs only**, where the projection has
  already been done by hand.

### 7.4 Operating-system limits

- No `sudo` for the agent account on the rigs.
- Rig environments read-only to the agent account.
- These hold even if a gate is skipped or an agent misbehaves.

### 7.5 Session record

Every delegated session writes an append-only record: authorization, commands issued on each
host, gate results, start and end environment digests, terminal event. It is part of the
evidence bundle.

## 8. Stop conditions

Any one ends the session. The agent stops the fleet cleanly where it can, preserves and hashes
the evidence, notifies the owner, and does nothing else.

- any prelaunch gate refuses
- a host or worker goes silent, or the visible GPU count drops on any rig
- a staging failure, lease expiry or trial terminal of any class
- any kernel message on netconsole
- actual survivor volume in any phase passes the cap
- the wall-clock limit is reached
- coordinator process death (the run is interrupted, not resumable)
- an environment-freeze violation

## 9. Run classes

| Class | Purpose | Geometry | Counts |
|---|---|---|---|
| **RUN4** | Run 4: the committed Route-A launch command | pinned (attempt 9) | Yes |
| **SWEEP** | Cover the rest of `[0, 2³²)` for `java_lcg` | pinned | Yes |
| **SOAK** | 50 trials, `--start-step 1 --end-step 1`, `n_parallel = 1` | sampled; **requires §7.3** | Yes |

No class exists for reproducing the August freeze or for deliberately high-volume runs.

## 10. Against Beta's recorded terms

The fact file (§2.30) records that Beta expected a delegated-execution amendment to meet an
eight-part contract and names five parts. The source document is not in the public repo.

| Beta's term (as recorded) | This amendment |
|---|---|
| frozen hypothesis | The run class and its pinned parameters (§5.2) |
| hard resource bounds | One trial at a time, `n_parallel = 1`, wall-clock limit, volume cap, permitted window |
| non-certifying | **Not adopted.** Owner ruling, §6 |
| no authority mutation | §4 items 6, 7, 8, 11 |
| no self-healing relaunch | §4 item 9 and §8 |
| three further terms | Unknown; Beta is asked to state them (§13) |

## 11. Before the first delegated run

1. Fleet powered on and `boot-proxmox` done **by the owner**.
2. Netconsole re-armed, one proof line confirmed per host.
3. Environment baselines captured by the owner (§7.2).
4. Restricted agent account and key installed (§5.1).
5. Authorization gate and environment-freeze gate built, tested and committed.
6. A standing authorization written by the owner for the class being run.

## 12. Replacement text for `CLAUDE.md` rule 3

> **3. Fleet runs are owner-initiated, or delegated under a valid run authorization.** A pipeline
> or fleet launch needs either Michael at the keyboard, or an unexpired owner-issued run
> authorization accepted by the authorization gate, a passing environment-freeze gate, and a
> predicted survivor volume under the cap. A delegated agent never changes system software on
> any host (packages, PyTorch, CuPy, ROCm, drivers, kernel, firmware, OS), never reboots a host,
> never commits or pushes, never bypasses a gate, and never relaunches after a failure. Always
> `nohup`, never `tmux`.

The current rule says the pipeline "spins 26 GPUs". The frozen execution set is 25. The
replacement drops the count.

## 13. Review questions for Beta

Beta's review of this amendment is after the fact. Its positions are recorded; objections Alpha
disagrees with go to Michael.

- **Q1.** What are the three terms of the eight-part contract that the repo does not record?
- **Q2.** Is the environment-freeze gate (§7.2), with the OS limits in §7.4, sufficient
  enforcement of "no system updates"? Should it also run mid-session?
- **Q3.** Is a pre-run sample a sound basis for the volume cap (§7.3), and is 23,515 a
  defensible starting value?
- **Q4.** Is A6, redeploying the ten governed files from committed HEAD, sound as a delegated
  action?
- **Q5.** What evidence does Beta want in a counted run's bundle to review it in one pass?

## 14. What this document does not claim

- It does not claim agents are safe by capability. The limits are meant to hold by construction.
- Nothing here is built. No key, account, gate or authorization exists yet.
- It does not explain the 2026-08-22 freeze. The volume cap guards against one of its two
  candidate causes; the other, supply sag, is not addressed, by owner ruling.
- It was written from the public repo. Live host state on 2026-11-11 is unobserved.
