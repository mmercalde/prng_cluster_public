# CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_7_SOAK.md — REV1 (DRAFT)

**S172 — Phase 7: the full-fleet WATCHER soak.**

**Base:** `18a2419` (D6.2 CERTIFIED). Claude Code on **VM101** as `michael`, venv `~/venvs/torch`.

**Authoritative definition** (`docs/PROPOSAL_S172_RANGE_MINER_v1_4_5.md`, Final architecture):

> **Phase 7 (soak):** 50-trial WATCHER soak with **≥5 high-survivor and ≥5 low-survivor trials**,
> **mixed constant/hybrid**, **per-trial cleanup verification**.

**§17 backend promotion is PHASE 6's, not Phase 7's** — the spec places
*"benchmark-driven default backend selection (§17)"* in the Phase 6 block. **Do not attempt a
promotion decision during this soak.** `serial_reference` remains the default.

---

## 0. Two binding constraints, one from Beta and one from the owner

### 0.1 `n_parallel = 1` — BINDING (Beta, D6.2 certification)

D6.2 checkpoint recovery and the S166 validated-checkpoint-then-clear protection are **certified
only for `n_parallel == 1`.** **The soak must pin it until NP2 receives a separate checkpoint
transaction design.**

**That path still distributes each individual sieve trial across the whole fleet** — the limit is
on concurrent *Optuna partition* execution, not on cluster participation. A 25-GPU trial at
`n_parallel=1` is exactly what is intended.

⚠ **The `:1979` rejection only fires for `resume_checkpoint` + `n_parallel > 1`.** A soak launched
at `n_parallel=8` **without** a checkpoint trips no guard and silently leaves the certified
envelope. **`n_parallel=1` must be passed explicitly and verified in the run log.**

### 0.2 25 GPUs — OWNER-MANDATED

**Michael, as owner, has mandated the soak runs at 25 GPUs.** VM101 currently has **one** RTX
3080 Ti passed through (`nvidia-smi -L`, verified); the second is still assigned to VM100.
**24 AMD (3 rigs × 8 RX 6600 XT) + 1 NVIDIA = 25.** No Beta approval is sought for this
configuration.

**The technical consequence, which is NOT a permission question:**

> Beta's Resolved Execution Set ruling: *"A partial set must be explicit and frozen before the run
> — never inferred from which workers happened to answer."*

**The run must freeze a 25-worker set BY CONSTRUCTION.** A 26-worker set that happens to receive 25
answers is precisely the defect `eff6616` closed: `admission_count = min(requested, selected
identities)`, and a trial expecting a worker the set itself declares unreachable **burns its entire
180 s `worker_admission_timeout` failing to meet a threshold that is unmeetable.**

**Verify before launch:** the frozen set's `admission_count` reads **25**, and both
`requested_admission_count` and `admission_count` appear in `set_id` so the run is auditable as a
25-GPU run rather than a 26-GPU run that came up short.

---

## 1. Prerequisites — the checklist is STALE IN BOTH DIRECTIONS

`docs/PHASE6_PREREQS.md` (TB-approved REV3) lists seven items, **all `☐ open`**, status column last
touched **2026-07-25**. **Four of those statuses are now wrong.** Measured on VM101 this session:

| # | item | doc | **measured** |
|---|---|---|---|
| 1 | second 3080Ti in VM101 | ☐ | **☐ OPEN** — `nvidia-smi -L` shows **one** GPU. Owner-mandated 25-GPU run proceeds (§0.2) |
| 2 | `michael → CT100` SSH | ☐ | **☑ DONE** — `.122`, `.156`, `.164` all answer under `BatchMode=yes`, no prompt |
| 3 | `rrig6600` Proxmox migration | ☐ | **☑ DONE** — `.120` silent, `.122` responds |
| 4 | VM101 stable address | ☐ | **☐ OPEN** — `inet 192.168.3.177/24 … dynamic` (DHCP) |
| 5 | publication filesystem + clean-tree preflight | ☐ | **NOT MEASURED** — §1.1 |
| 6 | code/clock parity | ☐ | **☑ DONE** — clock synchronized, NTP active |
| 7 | transport reachability + firewall | ☐ | **NOT MEASURED** — §1.1 |

**First deliverable: correct the status column in `docs/PHASE6_PREREQS.md`** from measurement, and
say in the commit that four statuses were stale. **A TB-approved checklist carrying wrong statuses
is worse than no checklist** — the next reader trusts it.

### 1.1 Items 5 and 7 must be measured, not assumed

Both were gated by Phase 6, which certified — so they are **probably** satisfied. **"Probably" is
not the standard for a release gate.** Measure and report both:

- **Item 5** — publication filesystem writable, temp and destination on the same filesystem,
  clean-tree preflight passes, no regular file at either finalizer-owned alias.
- **Item 7** — transport reachability to all three CT100 workers and the local GPU, firewall clear.
  *(Zeus has no firewall; never suggest `ufw`.)*

### 1.2 Item 4 — pin the address before launch

**DHCP for a multi-hour 50-trial run is a live risk, not a formality.** If the lease moves
mid-soak, every worker loses the coordinator and the run dies in a way that costs a day to
attribute. **A router-side DHCP reservation for `192.168.3.177` is zero-risk to the VM and
sufficient** — a static netplan edit carries lockout risk and buys nothing extra. **Confirm the
address survives a reboot before the soak starts.**

---

## 2. What this soak is actually testing

**This is the first run in the project's history where `_FLUSH_CLEAR_IN_MEMORY = True`.** The S166
OOM protection has never executed at scale — until `f7583bc` the write always failed, and until
`18a2419` the resume path rejected itself.

**So the headline question is not throughput. It is: does the candidate list stay bounded across 50
trials, and does the finalizer still receive complete 24-field input after every clear?**

Secondary: per-trial cleanup verification, and stability at 25-GPU saturation without
`GCVM_L2_PROTECTION_FAULT` — the fault RANGE-MINER exists to eliminate.

---

## 3. Configuration

| parameter | value | why |
|---|---|---|
| `n_parallel` | **1** | §0.1, **binding** |
| trials | **50** | spec. **Manifest default is `window_trials: 3` — must be overridden** |
| survivor mix | **≥5 high-survivor, ≥5 low-survivor** | spec |
| skip modes | **mixed constant/hybrid** | spec |
| backend | **`serial_reference`** | §17 is Phase 6's; no promotion here |
| `resume_checkpoint` | **empty** | a fresh soak; the resume path is certified but not under test here |
| execution set | **25 workers, explicit and frozen** | §0.2 |

**Hybrid caveat, carried forward:** hybrid exploration is **non-certifying**; `skip_min`/`skip_max`
still die at `_hybrid_prefix` and the dead-dimension caveat stands. **Mixed const/hybrid is a
stability requirement here, not a search-quality claim.** Do not read hybrid trial results as
evidence about skip semantics.

---

## 4. Launch

**Always `nohup`, never `tmux`,** for the pipeline itself.

Before launch, capture and report: the frozen `set_id` and its `admission_count` · the dataset
manifest identity and sha256 · `git rev-parse HEAD` and a clean-tree confirmation · free RAM and
disk on VM101 and all three rigs · the **current `.s172_checkpoint/` directory count** (25 at last
census — see §6).

## 5. Abort criteria — stop the run and report

- any `GCVM_L2_PROTECTION_FAULT` or GPU reset in a host kernel log;
- candidate-list RAM **growing monotonically across trials** — that is the S166 protection failing,
  and it is the primary thing under test;
- a trial that **neither completes nor fails** (the §4.3 hang class — repaired at `ee0db06`, but
  this is its first full-scale exercise);
- swap usage on VM101 or any rig;
- an admission failure naming an expected worker count **other than 25**;
- any publication error at the finalizer.

**Do not restart into the same run id after an abort.** Capture the state first — a soak failure is
evidence, and restarting destroys it.

## 6. Free data — record it for D6.3

D6.3 (checkpoint retention) is unstarted and its open question is **how fast
`.s172_checkpoint/<run_id>/` grows.** Nobody has measured it. **Record the directory count and
total bytes before and after the soak.** That is D6.3's Q3 answered at zero cost, and it decides
whether D6.3 is a real blocker or a slow-burn item.

**Do not delete any checkpoint directory.** Beta's D6.3 constraint: *never remove active,
unresolved or audit-retained state merely for exceeding an age or count threshold.*

## 7. Report

`docs/S172_PHASE_7_SOAK_REPORT.md`: the corrected §1 prerequisite table with measurements · items 5
and 7 measured · the frozen `set_id` with `admission_count=25` · `n_parallel=1` **confirmed in the
run log, not merely in the command** · 50 trials with the high/low-survivor and const/hybrid
breakdown · **RAM across trials, as a series not a peak** · per-trial cleanup verification · any
kernel-log GPU events · the §6 checkpoint growth figures · every abort criterion explicitly
evaluated. Then STOP for Team Alpha review.

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** the run log names the frozen `set_id`, the admitted worker count and
  `n_parallel` per trial — a soak that "looks fine" without those is unverified.
- **clean control:** the pre-launch capture in §4 is the baseline every post-run figure is read
  against.
- **fault-injection control:** `NOT_APPLICABLE` — this is an execution run, not a detector under
  validation. **Say so; do not write `PASS`.**
- **completion sentinel:** `PASS | FAIL | UNAVAILABLE | INCOMPLETE`. **A soak that stops early is
  `INCOMPLETE`, never `PASS`.**
- **unavailable-observer behavior:** a rig that cannot be reached at launch makes the run
  `UNAVAILABLE`, **not** a 24-GPU soak. The set is frozen before the run and is not renegotiated by
  who answers.
- **audit claim scope:** live-fleet, VM101 plus three CT100 workers, at `18a2419`.
- **searched surfaces:** to be enumerated in the report.
- **unavailable surfaces:** the second 3080Ti (assigned to VM100 — §0.2); anything requiring the
  bare-metal rig profile, since the fleet runs Proxmox CT100 endpoints.
